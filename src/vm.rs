//! vm.rs — Estrutura principal da M³-AVM e loop de execução
//!
//! - Loop infinito com fetch/decode/execute
//! - Escalonador estrito RED > BLUE > GREEN (preempção)
//! - Dispatch dos 6 opcodes via `opcodes.rs`
//! - Integração com `memory.rs` (CoW + mmap) e `context.rs`
//!
//! Meta de throughput: 1_000_000 IPS (debug) — sem otimização CUDA, pura CPU.
//! O pipeline é simples (sem JIT), mas já alinha instruções em 32 bytes para
//! beneficiar prefetch e facilitar futura tradução para GPU/NIF.

use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use tokio::sync::mpsc;

use crate::context::{Context, Priority, Scheduler};
use crate::memory::{DType, MemoryManager};
#[cfg(feature = "wgpu")]
use crate::memory_wgpu::WgpuMemoryManager;
#[cfg(feature = "wgpu")]
use pollster;
use crate::opcodes::{
    Instruction, ProgramInstr, INSTR_SIZE, OP_ABORT, OP_ADD, OP_ATTN, OP_AUDIO_ALIGN, OP_CODEC_DEC, OP_CODEC_ENC,
    OP_COMPARE, OP_CTX_SWITCH, OP_DISTANCE, OP_EMBED, OP_FFN, OP_FORK, OP_GATHER, OP_HALT, OP_IF_EQUAL, OP_IF_INTERRUPT, OP_JUMP,
    OP_MATVEC, OP_MUL, OP_NOP, OP_NORM, OP_RANK1_UPDATE, OP_ROPE, OP_SAMPLE, OP_SENSE, OP_SILU, OP_SSM_RESET, OP_SSM_SCAN,
    OP_STREAM, OP_TENSOR, SENSE_AUDIO, SENSE_AUDIO_PCM, SENSE_CODEC_FRAME, SENSE_TOKEN, SENSE_USER_INPUT,
    SENSE_VAD, STREAM_FLAG_BLOCKING, OP_RNG_SEED, OP_RNG_NEXT, OP_RNG_UNIFORM, OP_RNG_NORMAL,
    OP_HASH, OP_CHECKSUM, OP_HMAC, OP_CYCLES_COUNT, OP_TRACE_EVENT, OP_SANITY_CHECK,
    OP_PREEMPT_CHECK, OP_ASSERT, OP_DUMP, OP_YIELD, OP_SET_DEADLINE, OP_GET_DEADLINE,
    OP_PRIORITY_SET, OP_PRIORITY_GET, OP_LOCK, OP_UNLOCK, OP_FENCE, OP_LOADI, OP_MOV, OP_KV_TRUNCATE,
    OP_FOREST, OP_DENOISE_STEP, OP_ODE_STEP, OP_SPIKE_STEP, OP_CONV,
    OP_REMOTE_SPAWN, OP_SIGNAL, OP_SEND_TENSOR, OP_BARRIER, OP_SLICE,
    OP_ARENA_ALLOC, OP_ARENA_RESET, OP_MEMCPY, OP_MEMSET, MEMCPY_DIR_HOST,
    OP_SNAPSHOT, OP_RESTORE, OP_PREFETCH, OP_RESHAPE, OP_CONCAT, SNAP_MASK_ALL,
    OP_CAST, OP_QUANTIZE, OP_DEQUANT, CAST_DST_F32, CAST_DST_F16, CAST_DST_BF16,
    CAST_DST_I8, CAST_DST_U8, QUANTIZE_Q4_0, QUANTIZE_Q8_0,
    OP_ADD_IMM, OP_SUB_IMM, OP_STEPS,
    OP_SORT, OP_TOPK, OP_ARGMAX, OP_REDUCE, OP_BROADCAST, OP_PAD, OP_TILE,
    OP_TRANSPOSE, SORT_ASC, SORT_DESC, REDUCE_SUM, REDUCE_MEAN, REDUCE_MAX,
    REDUCE_MIN, REDUCE_PROD,
    OP_KV_COMPRESS, OP_FLASH_ATTN, OP_ATTN_SPARSE, KVCOMP_MODE_SINK_WINDOW,
    SPARSE_METRIC_DOT,
    OP_STREAM_MERGE, OP_VAD_DETECT, OP_AUDIO_RESAMPLE, OP_AUDIO_FILTER,
    OP_AUDIO_WINDOW, VAD_MODE_ENERGY, VAD_MODE_ZCR, VAD_MODE_ML,
    FILTER_MODE_FIR, FILTER_MODE_IIR, WINDOW_HANN, WINDOW_HAMMING,
    OP_DEPFORMER, OP_CALL, OP_RET,
    OP_SOFTMAX, OP_GELU, OP_SIGMOID, OP_TANH, OP_RELU, OP_EXP, OP_LOG, OP_CLIP,
};
use crate::utils::{log_debug, log_info, log_warn, ThroughputMeter};

/// Configuração da VM
#[derive(Debug, Clone)]
pub struct VmConfig {
    pub max_steps: Option<u64>,
    pub enable_tracing: bool,
    pub ips_target: u64,
}

impl Default for VmConfig {
    fn default() -> Self {
        Self {
            max_steps: None,
            enable_tracing: false,
            ips_target: 1_000_000,
        }
    }
}

/// Estatísticas de execução
#[derive(Debug, Default, Clone)]
pub struct VmStats {
    pub steps: u64,
    pub tensor_allocs: u64,
    pub attn_execs: u64,
    pub norm_execs: u64,
    pub ffn_execs: u64,
    pub embed_execs: u64,
    pub add_execs: u64,
    pub sample_execs: u64,
    pub matvec_execs: u64,
    pub interrupt_checks: u64,
    pub streams: u64,
    pub forks: u64,
    pub aborts: u64,
    pub senses: u64,
    pub ssm_scans: u64,
    pub ssm_resets: u64,
    pub codec_encs: u64,
    pub codec_decs: u64,
    pub audio_aligns: u64,
    pub ctx_switches: u64,
    pub ropes: u64,
    pub gather_execs: u64,
    pub distance_execs: u64,
    pub rank1_execs: u64,
    pub rng_seed_execs: u64,
    pub rng_next_execs: u64,
    pub rng_uniform_execs: u64,
    pub rng_normal_execs: u64,
    pub hash_execs: u64,
    pub checksum_execs: u64,
    pub hmac_execs: u64,
    pub cycles_execs: u64,
    pub trace_execs: u64,
    pub sanity_execs: u64,
    pub preempt_check_execs: u64,
    pub assert_execs: u64,
    pub dump_execs: u64,
    pub yield_execs: u64,
    pub set_deadline_execs: u64,
    pub get_deadline_execs: u64,
    pub priority_set_execs: u64,
    pub priority_get_execs: u64,
    pub lock_execs: u64,
    pub unlock_execs: u64,
    pub fence_execs: u64,
    pub loadi_execs: u64,
    pub mov_execs: u64,
    pub kv_truncate_execs: u64,
    pub slice_execs: u64,
    pub forest_execs: u64,
    pub denoise_execs: u64,
    pub ode_execs: u64,
    pub spike_execs: u64,
    pub conv_execs: u64,
    pub remote_spawn_execs: u64,
    pub signal_execs: u64,
    pub send_tensor_execs: u64,
    pub barrier_execs: u64,
    pub arena_alloc_execs: u64,
    pub arena_reset_execs: u64,
    pub memcpy_execs: u64,
    pub memset_execs: u64,
    pub snapshot_execs: u64,
    pub restore_execs: u64,
    pub prefetch_execs: u64,
    pub reshape_execs: u64,
    pub concat_execs: u64,
    pub cast_execs: u64,
    pub quantize_execs: u64,
    pub dequant_execs: u64,
    pub add_imm_execs: u64,
    pub sub_imm_execs: u64,
    pub steps_execs: u64,
    pub sort_execs: u64,
    pub topk_execs: u64,
    pub argmax_execs: u64,
    pub reduce_execs: u64,
    pub broadcast_execs: u64,
    pub pad_execs: u64,
    pub tile_execs: u64,
    pub transpose_execs: u64,
    pub softmax_execs: u64,
    pub gelu_execs: u64,
    pub sigmoid_execs: u64,
    pub tanh_execs: u64,
    pub relu_execs: u64,
    pub exp_execs: u64,
    pub log_execs: u64,
    pub clip_execs: u64,
    pub kv_compress_execs: u64,
    pub flash_attn_execs: u64,
    pub attn_sparse_execs: u64,
    pub stream_merge_execs: u64,
    pub vad_detect_execs: u64,
    pub audio_resample_execs: u64,
    pub audio_filter_execs: u64,
    pub audio_window_execs: u64,
    pub depformer_execs: u64,
    pub call_execs: u64,
    pub ret_execs: u64,
    pub start_ns: u64,
}

impl VmStats {
    pub fn new() -> Self {
        Self {
            start_ns: crate::utils::now_ns(),
            ..Default::default()
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        crate::utils::now_ns().saturating_sub(self.start_ns) / 1_000_000
    }
}

/// Backend de memória — CPU ou Vega 8 (wgpu) — auto-detectado
pub enum MemBackend {
    Cpu(MemoryManager),
    #[cfg(feature = "wgpu")]
    Gpu(WgpuMemoryManager),
}

impl MemBackend {
    pub fn as_cpu_mut(&mut self) -> Option<&mut MemoryManager> {
        match self { MemBackend::Cpu(m) => Some(m), #[cfg(feature = "wgpu")] _ => None }
    }
    pub fn is_gpu(&self) -> bool {
        match self { MemBackend::Cpu(_) => false, #[cfg(feature = "wgpu")] MemBackend::Gpu(_) => true }
    }
    pub fn stats(&self) -> String {
        match self { MemBackend::Cpu(m) => m.stats(), #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.stats() }
    }
    // Delega alloc/read/write para Vm não quebrar — dispatch
    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> anyhow::Result<u128> {
        match self {
            MemBackend::Cpu(m) => m.alloc_tensor(shape, dtype),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.alloc_tensor(shape, dtype),
        }
    }
    pub fn alloc_sparse_tensor(&mut self, shape: &[usize], dtype: DType, density: f32) -> anyhow::Result<u128> {
        match self {
            MemBackend::Cpu(m) => m.alloc_sparse_tensor(shape, dtype, density),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.alloc_sparse_tensor(shape, dtype, density),
        }
    }
    pub fn is_sparse(&self, addr: u128) -> bool {
        match self {
            MemBackend::Cpu(m) => m.is_sparse(addr),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.is_sparse(addr),
        }
    }
    pub fn get_sparse(&self, addr: u128) -> Option<&crate::sparse::SparseTensor> {
        match self {
            MemBackend::Cpu(m) => m.get_sparse(addr),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.get_sparse(addr),
        }
    }
    pub fn get_sparse_mut(&mut self, addr: u128) -> Option<&mut crate::sparse::SparseTensor> {
        match self {
            MemBackend::Cpu(m) => m.get_sparse_mut(addr),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.get_sparse_mut(addr),
        }
    }
    pub fn get_tensor_meta(&self, addr: u128) -> Option<&crate::memory::TensorMeta> {
        match self {
            MemBackend::Cpu(m) => m.get_tensor_meta(addr),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.get_tensor_meta(addr),
        }
    }
    pub fn read(&self, addr: u128, len: usize) -> anyhow::Result<Vec<u8>> {
        match self {
            MemBackend::Cpu(m) => m.read(addr, len),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.read(addr, len),
        }
    }
    /// Zero-copy do mmap do modelo (`None` = backend sem mmap; caller faz `read`).
    pub fn read_model_raw(&self, file_offset: u64, len: usize) -> Option<&[u8]> {
        match self {
            MemBackend::Cpu(m) => m.get_model_raw_slice(file_offset, len),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(_) => None,
        }
    }
    pub fn write(&mut self, addr: u128, data: &[u8]) -> anyhow::Result<()> {
        match self {
            MemBackend::Cpu(m) => m.write(addr, data),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.write(addr, data),
        }
    }
    /// Alocação GLOBAL bruta (V-1b dia 3: preload `.data`).
    pub fn alloc_global(&mut self, size: usize) -> anyhow::Result<u128> {
        match self {
            MemBackend::Cpu(m) => m.alloc_global(size),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.alloc_global(size),
        }
    }
    /// Alocação com byte_len explícito (layouts em bloco, RFC-0025).
    /// GPU veta explícito (sem meta em blocos lá).
    pub fn alloc_tensor_bytes(&mut self, shape: &[usize], dtype: crate::memory::DType, byte_len: usize) -> anyhow::Result<u128> {
        match self {
            MemBackend::Cpu(m) => m.alloc_tensor_bytes(shape, dtype, byte_len),
            #[cfg(feature = "wgpu")]
            MemBackend::Gpu(_) => Err(anyhow::anyhow!("alloc em blocos no backend GPU (RFC futura)")),
        }
    }
    pub fn write_f32_tensor(&mut self, addr: u128, data: &[f32]) -> anyhow::Result<()> {
        match self {
            MemBackend::Cpu(m) => m.write_f32_tensor(addr, data),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.write_f32_tensor(addr, data),
        }
    }
    pub fn read_f32_tensor(&self, addr: u128, count: usize) -> anyhow::Result<Vec<f32>> {
        match self {
            MemBackend::Cpu(m) => m.read_f32_tensor(addr, count),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.read_f32_tensor(addr, count),
        }
    }
    pub fn snapshot(&mut self) -> u64 {
        match self {
            MemBackend::Cpu(m) => m.snapshot(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.snapshot(),
        }
    }
    pub fn snapshot_count(&self) -> usize {
        match self {
            MemBackend::Cpu(m) => m.snapshot_count(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.snapshot_count(),
        }
    }
    pub fn restore(&mut self, v: u64) -> anyhow::Result<()> {
        match self {
            MemBackend::Cpu(m) => m.restore(v),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.restore(v),
        }
    }
    pub fn remove_tensor(&mut self, addr: u128) -> bool {
        match self {
            MemBackend::Cpu(m) => m.remove_tensor(addr),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.remove_tensor(addr),
        }
    }
    pub fn current_version(&self) -> u64 {
        match self {
            MemBackend::Cpu(m) => m.current_version(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.current_version(),
        }
    }
    pub fn persistent_flush(&self) -> anyhow::Result<()> {
        match self {
            MemBackend::Cpu(m) => m.persistent_flush(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.persistent_flush(),
        }
    }
    pub fn temporal_push(&mut self, data: &[u8]) -> anyhow::Result<u128> {
        match self {
            MemBackend::Cpu(m) => m.temporal_push(data),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.temporal_push(data),
        }
    }
    pub fn temporal_read(&self, addr: u128, len: usize) -> anyhow::Result<Vec<u8>> {
        match self {
            MemBackend::Cpu(m) => m.temporal_read(addr, len),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.temporal_read(addr, len),
        }
    }
    pub fn load_gguf_model(&mut self, path: &str) -> anyhow::Result<u128> {
        match self {
            MemBackend::Cpu(m) => m.load_gguf_model(path),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.load_gguf_model(path),
        }
    }
    pub fn load_gguf_bytes(&mut self, bytes: &[u8]) -> anyhow::Result<u128> {
        match self {
            MemBackend::Cpu(m) => m.load_gguf_bytes(bytes),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.load_gguf_bytes(bytes),
        }
    }
    // KV_CACHE delegação
    pub fn kv_cache_init(&mut self, n_layers: usize, hidden: usize) {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_init(n_layers, hidden),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_init(n_layers, hidden),
        }
    }
    pub fn kv_cache_append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> anyhow::Result<()> {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_append(layer, k, v),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_append(layer, k, v),
        }
    }
    pub fn kv_cache_get_k(&self, layer: usize) -> Option<Vec<f32>> {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_get_k(layer).map(|s| s.to_vec()),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_get_k(layer).map(|s| s.to_vec()),
        }
    }
    pub fn kv_cache_get_v(&self, layer: usize) -> Option<Vec<f32>> {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_get_v(layer).map(|s| s.to_vec()),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_get_v(layer).map(|s| s.to_vec()),
        }
    }
    pub fn kv_cache_seq_len(&self) -> usize {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_seq_len(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_seq_len(),
        }
    }
    pub fn kv_cache_truncate(&mut self, seq: usize) {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_truncate(seq),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_truncate(seq),
        }
    }
    pub fn kv_cache_truncate_stream(&mut self, sid: u16, seq: usize) {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_truncate_stream(sid, seq),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_truncate_stream(sid, seq),
        }
    }
    pub fn kv_cache_compress_sink_window_stream(&mut self, sid: u16, sink: usize, window: usize) {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_compress_sink_window_stream(sid, sink, window),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_compress_sink_window_stream(sid, sink, window),
        }
    }
    pub fn kv_cache_append_stream(&mut self, sid: u16, layer: usize, k: &[f32], v: &[f32]) -> anyhow::Result<()> {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_append_stream(sid, layer, k, v),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_append_stream(sid, layer, k, v),
        }
    }
    pub fn kv_cache_seq_len_stream(&self, sid: u16) -> usize {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_seq_len_stream(sid),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_seq_len_stream(sid),
        }
    }
    pub fn kv_cache_compress_sink_window(&mut self, sink: usize, window: usize) {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_compress_sink_window(sink, window),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_compress_sink_window(sink, window),
        }
    }
    pub fn kv_cache_clear(&mut self) {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_clear(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_clear(),
        }
    }
    pub fn kv_cache_attention(&self, layer: usize, q: &[f32]) -> anyhow::Result<Vec<f32>> {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_attention(layer, q),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_attention(layer, q),
        }
    }
    pub fn kv_cache_stats(&self) -> String {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_stats(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_stats(),
        }
    }
    pub fn kv_cache_n_layers(&self) -> usize {
        match self {
            MemBackend::Cpu(m) => m.kv_cache_n_layers(),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.kv_cache_n_layers(),
        }
    }
}

/// Máquina Virtual M³-AVM
/// Teto do anel de trace (TRACE_EVENT 0x6B, RFC-0006): além disso, descarta o
/// mais antigo. O limite existe para a primitiva nunca virar vetor de flood.
pub const TRACE_CAP: usize = 1024;

pub struct Vm {
    pub memory: MemBackend,
    pub scheduler: Scheduler,
    /// Programa carregado (frames decodificados, largura mista).
    /// `pc_offsets[i]` = byte offset da instrução `i` a partir de
    /// `program_base`; o stride vem de `ProgramInstr::byte_len()`
    /// (W1-remainder). Programas só-32B têm offsets uniformes.
    pub(crate) program: Vec<ProgramInstr>,
    /// Byte offsets por instrução (mesmo len que `program`).
    pub(crate) pc_offsets: Vec<u128>,
    /// Base address do programa em GLOBAL (onde o código reside conceitualmente)
    pub(crate) program_base: u128,
    pub config: VmConfig,
    pub stats: VmStats,
    /// Canais para STREAM (stub tokio mpsc)
    stream_channels: HashMap<u128, mpsc::Sender<Vec<u8>>>,
    meter: ThroughputMeter,
    /// Tokenizer GGUF (opcional) e último token amostrado (host-side SAMPLE)
    pub tokenizer: Option<crate::tokenizer::M3Tokenizer>,
    pub last_sample: u32,
    /// Fila de entrada do usuário (SENSE USER_INPUT consome sem bloquear)
    pub user_input: VecDeque<String>,
    /// Estado recorrente Mamba por camada (fase 2 híbrida; MVP usa tensores,
    /// Rh==0xFF endereça este vetor via payload layer_id).
    pub ssm_states: Vec<crate::ssm::MambaState>,
    /// Pilha de snapshots SSM para ABORT/rollback: (versão da memória no
    /// FORK, estados). RFC-0011: pop versionado (ts!=0) ou LIFO legado (ts=0).
    ssm_snapshots: Vec<(u64, Vec<crate::ssm::MambaState>)>,
    /// Handle corrente da matriz H por camada (RANK1_UPDATE 0x24, RFC-0004).
    /// H em si vive em tensores imutáveis (CoW por passo); aqui só o mapa
    /// camada->addr, com push no FORK e pop no ABORT (espelha ssm_snapshots).
    pub rank1_layers: HashMap<u8, u128>,
    rank1_snapshots: Vec<(u64, HashMap<u8, u128>)>,
    /// Handles SNN por camada (SPIKE_STEP 0x20, RFC-0015): layer ->
    /// (addr V, addr refr). Tensores imutáveis (CoW); mapa com
    /// push no FORK e pop versionado no ABORT (espelha rank1_layers).
    pub snn_layers: HashMap<u8, (u128, u128)>,
    snn_snapshots: Vec<(u64, HashMap<u8, (u128, u128)>)>,
    /// Arenas bump por id (ARENA_ALLOC 0x26, RFC-0023): id -> Arena.
    /// Push (clone) no FORK, pop versionado no ABORT (espelha
    /// rank1_layers); custo do snapshot O(bytes totais), ver RFC.
    pub arenas: HashMap<u8, crate::arena::Arena>,
    arena_snapshots: Vec<(u64, HashMap<u8, crate::arena::Arena>)>,
    /// KV do depformer por (stream, layer) (DEPFORMER 0x44, RFC-0032).
    /// Chaveado desde o dia um (17 streams: RFC-0033 adiciona streams,
    /// nunca reobra). Push no FORK + SNAPSHOT, pop versionado no
    /// ABORT + RESTORE — as quatro casas, sem exceção.
    pub dep_kv: HashMap<(u8, u8), crate::depformer::DepKV>,
    dep_snapshots: Vec<(u64, HashMap<(u8, u8), crate::depformer::DepKV>)>,
    /// Barreiras locais (BARRIER 0x1D, RFC-0018): id -> estado one-shot.
    /// Removida no RELEASE e no timeout (sem reuso silencioso de geração).
    pub barriers: HashMap<u32, BarrierState>,
    /// Espera por barreira (RFC-0018): id -> ctxs bloqueados. Drenado no
    /// RELEASE/timeout (só ctxs ainda existentes são reacordados).
    pub barrier_waiters: HashMap<u32, Vec<u64>>,
    /// Locks cross-context (LOCK 0x75 / UNLOCK 0x76, RFC-0006): id -> holder.
    /// Try-lock não-bloqueante; sem filas de espera (EDF real é follow-up).
    pub locks: HashMap<u32, u64>,
    /// Anel de trace (TRACE_EVENT 0x6B): (event_id, data), teto TRACE_CAP.
    pub trace: VecDeque<(u64, u128)>,
}

/// Estado one-shot de uma barreira local (RFC-0018).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarrierState {
    pub expected: u16,
    pub arrived: u32,
    pub deadline_ns: u64, // u64::MAX = sem timeout
    pub epoch: u32,
}

impl Vm {
    pub fn new(config: VmConfig) -> Result<Self> {
        Self::new_with_persistent_size(config, 64 * 1024 * 1024)
    }

    pub fn new_with_persistent_size(config: VmConfig, persistent_bytes: usize) -> Result<Self> {
        // Auto-detecta Vega 8 se feature wgpu ativa — fallback para CPU.
        // M3_GPU=0 força CPU (útil p/ comparar e em CI sem GPU).
        #[cfg(feature = "wgpu")]
        let gpu_off = std::env::var("M3_GPU").map(|v| v == "0").unwrap_or(false);
        #[cfg(feature = "wgpu")]
        let memory = if gpu_off {
            eprintln!("[vm] M3_GPU=0 — backend CPU forçado");
            MemBackend::Cpu(MemoryManager::new_with_size(persistent_bytes)?)
        } else {
            match pollster::block_on(WgpuMemoryManager::new()) {
                Ok(gpu) => {
                    eprintln!("[vm] Vega 8 detectada — usando WgpuMemoryManager (Vulkan RADV)");
                    MemBackend::Gpu(gpu)
                }
                Err(e) => {
                    eprintln!("[vm] wgpu não disponível ({}), fallback CPU", e);
                    MemBackend::Cpu(MemoryManager::new_with_size(persistent_bytes)?)
                }
            }
        };
        #[cfg(not(feature = "wgpu"))]
        let memory = MemBackend::Cpu(MemoryManager::new_with_size(persistent_bytes)?);
        let scheduler = Scheduler::new();
        Ok(Self {
            memory,
            scheduler,
            program: Vec::new(),
            pc_offsets: Vec::new(),
            program_base: 0x1000, // programa começa em GLOBAL 0x1000
            config,
            stats: VmStats::new(),
            stream_channels: HashMap::new(),
            meter: ThroughputMeter::new(),
            tokenizer: None,
            last_sample: 0,
            user_input: VecDeque::new(),
            ssm_states: Vec::new(),
            ssm_snapshots: Vec::new(),
            rank1_layers: HashMap::new(),
            rank1_snapshots: Vec::new(),
            snn_layers: HashMap::new(),
            snn_snapshots: Vec::new(),
            arenas: HashMap::new(),
            arena_snapshots: Vec::new(),
            dep_kv: HashMap::new(),
            dep_snapshots: Vec::new(),
            barriers: HashMap::new(),
            barrier_waiters: HashMap::new(),
            locks: HashMap::new(),
            trace: VecDeque::new(),
        })
    }

    /// Construtor sem persistência (testes) — sempre CPU para determinismo
    /// (init de device ~800ms na Raven quebraria thresholds de timing; o path
    /// GPU tem testes próprios via WgpuMemoryManager::new_blocking).
    pub fn new_in_memory(config: VmConfig) -> Self {
        let memory = MemBackend::Cpu(MemoryManager::new_in_memory());
        Self {
            memory,
            scheduler: Scheduler::new(),
            program: Vec::new(),
            pc_offsets: Vec::new(),
            program_base: 0x1000,
            config,
            stats: VmStats::new(),
            stream_channels: HashMap::new(),
            meter: ThroughputMeter::new(),
            tokenizer: None,
            last_sample: 0,
            user_input: VecDeque::new(),
            ssm_states: Vec::new(),
            ssm_snapshots: Vec::new(),
            rank1_layers: HashMap::new(),
            rank1_snapshots: Vec::new(),
            snn_layers: HashMap::new(),
            snn_snapshots: Vec::new(),
            arenas: HashMap::new(),
            arena_snapshots: Vec::new(),
            dep_kv: HashMap::new(),
            dep_snapshots: Vec::new(),
            barriers: HashMap::new(),
            barrier_waiters: HashMap::new(),
            locks: HashMap::new(),
            trace: VecDeque::new(),
        }
    }

    /// Construtor explícito CPU (para testes determinísticos)
    pub fn new_cpu(config: VmConfig) -> Result<Self> {
        Ok(Self {
            memory: MemBackend::Cpu(MemoryManager::new()?),
            scheduler: Scheduler::new(),
            program: Vec::new(),
            pc_offsets: Vec::new(),
            program_base: 0x1000,
            config,
            stats: VmStats::new(),
            stream_channels: HashMap::new(),
            meter: ThroughputMeter::new(),
            tokenizer: None,
            last_sample: 0,
            user_input: VecDeque::new(),
            ssm_states: Vec::new(),
            ssm_snapshots: Vec::new(),
            rank1_layers: HashMap::new(),
            rank1_snapshots: Vec::new(),
            snn_layers: HashMap::new(),
            snn_snapshots: Vec::new(),
            arenas: HashMap::new(),
            arena_snapshots: Vec::new(),
            dep_kv: HashMap::new(),
            dep_snapshots: Vec::new(),
            barriers: HashMap::new(),
            barrier_waiters: HashMap::new(),
            locks: HashMap::new(),
            trace: VecDeque::new(),
        })
    }

    pub fn set_tokenizer(&mut self, tok: crate::tokenizer::M3Tokenizer) {
        self.tokenizer = Some(tok);
    }

    /// Enfileira entrada do usuário (consumida por SENSE USER_INPUT).
    pub fn push_input(&mut self, text: String) {
        self.user_input.push_back(text);
    }

    /// Host-side SAMPLE: softmax + random weighted sampling (temperatura 1.0)
    /// Se logits são constantes (dummy), adiciona jitter para variar tokens e provar pipeline
    /// LEGADO (thread_rng): preservado p/ compat externa; dentro da Vm todo
    /// sampling passa por `sample_logits_ctx` (RFC-0009). Será removido.
    #[deprecated(note = "use sample_logits_ctx (seeded, RFC-0009) ou determinism::sample_logits_host")]
    pub fn sample_logits(&self, logits: &[f32]) -> u32 {
        if logits.is_empty() { return 0; }
        // Adiciona jitter aleatório para dummy logits constantes (ex: 0.7) não ficarem sempre token 0
        let mut rng = rand::thread_rng();
        let jittered: Vec<f32> = logits.iter().map(|&v| v + rng.gen_range(-0.5..0.5)).collect();
        let max = jittered.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        let mut exps = Vec::with_capacity(jittered.len());
        for &v in &jittered {
            let e = (v - max).exp();
            exps.push(e);
            sum += e;
        }
        // Weighted sampling
        let mut r: f32 = rng.gen_range(0.0..sum);
        for (i, e) in exps.iter().enumerate() {
            r -= *e;
            if r <= 0.0 { return i as u32; }
        }
        // Fallback argmax
        let mut best_idx = 0usize;
        let mut best_p = 0.0f32;
        for (i, e) in exps.iter().enumerate() {
            let p = e / sum;
            if p > best_p { best_p = p; best_idx = i; }
        }
        best_idx as u32
    }

    /// Amostragem seeded pelo contexto (RFC-0009): mesma seed + mesmos
    /// logits => mesmo token. Avança `rng_state` do contexto.
    pub fn sample_logits_ctx(&mut self, ctx_id: u64, logits: &[f32]) -> Result<u32> {
        let mut st = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.rng_state;
        let tok = crate::determinism::sample_logits_seeded(&mut st, logits);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.rng_state = st;
        }
        Ok(tok)
    }

    /// Carrega programa 32B a partir de Vec<Instruction> (envolve em
    /// `ProgramInstr::W32`; todos os chamadores existentes inalterados).
    pub fn load_program(&mut self, prog: Vec<Instruction>) {
        self.load_mixed(prog.into_iter().map(ProgramInstr::W32).collect());
    }

    /// V-1b dia 3: carrega programa montado com `.data` (Opção A).
    /// Preload como PRIMEIRA alocação GLOBAL (base determinística
    /// `DATA_LOAD_BASE`); qualquer desvio => erro alto, nunca silencioso
    /// (ex.: VM não-fresca ou segundo preload — endereços `@` foram
    /// congelados no assemble para a base).
    pub fn load_assembled(&mut self, prog: &crate::opcodes::AssembledProgram) -> anyhow::Result<()> {
        use crate::opcodes::{data_layout_addrs, data_layout_total, DATA_LOAD_BASE};
        if !prog.data.blobs.is_empty() {
            let total = data_layout_total(&prog.data);
            let base = self.memory.alloc_global(total)?;
            if base != DATA_LOAD_BASE {
                return Err(anyhow::anyhow!(
                    "load_assembled: base GLOBAL 0x{:x} != DATA_LOAD_BASE 0x{:x} (preload exige VM fresca, antes de qualquer alloc)",
                    base,
                    DATA_LOAD_BASE
                ));
            }
            for (name, addr) in data_layout_addrs(&prog.data) {
                let blob = prog.data.find(&name).ok_or_else(|| {
                    anyhow::anyhow!("load_assembled: blob '{}' sumiu do layout", name)
                })?;
                self.memory.write(addr, &blob.bytes)?;
            }
        }
        self.load_program(prog.instrs.clone());
        Ok(())
    }

    /// Carrega programa de largura mista + reconstrói a tabela de offsets
    /// (base do fetch com stride variável, W1-remainder).
    pub fn load_mixed(&mut self, prog: Vec<ProgramInstr>) {
        // Invariante: classe de largura do frame == classe do opcode
        // (W64(0x00/0xFF) e W32(0x80+) são inconstructíveis via decode).
        debug_assert!(prog.iter().all(|f| match f {
            ProgramInstr::W32(i) =>
                crate::opcodes::instr_width(i.opcode) == crate::opcodes::InstrWidth::Fixed32,
            ProgramInstr::W64(g) =>
                crate::opcodes::instr_width(g.opcode) == crate::opcodes::InstrWidth::Fixed64,
        }));
        let base = self.program_base;
        self.program = prog;
        self.rebuild_offsets();
        // Cria contexto inicial GREEN se não houver nenhum
        if self.scheduler.count() == 0 {
            let root = self.memory.current_version();
            let entry_pc = base; // PC inicial aponta para primeira instrução
            let ctx_id = self.scheduler.create_context(Priority::Green, entry_pc, root);
            log_info("vm", &format!("programa carregado: {} instr, ctx inicial {} @ PC {:032x}", self.program.len(), ctx_id, entry_pc));
        }
    }

    /// Reconstrói `pc_offsets` a partir das larguras reais.
    fn rebuild_offsets(&mut self) {
        self.pc_offsets.clear();
        let mut off = 0u128;
        for ins in &self.program {
            self.pc_offsets.push(off);
            off += ins.byte_len() as u128;
        }
    }

    /// Fim do programa em endereço absoluto (para validação de jump).
    fn program_end(&self) -> u128 {
        match (self.pc_offsets.last(), self.program.last()) {
            (Some(&off), Some(last)) => self.program_base + off + last.byte_len() as u128,
            _ => self.program_base,
        }
    }

    /// Carrega de bytes brutos .m3bin (32 bytes por instrução)
    pub fn load_bin(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() % INSTR_SIZE != 0 {
            return Err(anyhow!(".m3bin tamanho não múltiplo de 32: {}", bytes.len()));
        }
        let mut prog = Vec::new();
        for chunk in bytes.chunks_exact(INSTR_SIZE) {
            let instr = Instruction::decode(chunk)?;
            instr.validate().map_err(|e| anyhow!("decode: {}", e))?;
            prog.push(instr);
        }
        self.load_program(prog);
        Ok(())
    }

    /// Traduz PC (u128) para índice no programa via tabela de offsets
    /// (busca binária; endereço no meio de instrução larga => None).
    fn pc_to_index(&self, pc: u128) -> Option<usize> {
        if pc < self.program_base {
            return None;
        }
        let offset = pc - self.program_base;
        self.pc_offsets.binary_search(&offset).ok()
    }

    /// Busca instrução num PC absoluto (compartilhado com main/reactor,
    /// que indexavam `program` com `/32` — ver `fetch_at`).
    pub(crate) fn fetch_at(&self, pc: u128) -> Option<ProgramInstr> {
        self.pc_to_index(pc).map(|idx| self.program[idx])
    }

    /// Busca instrução no PC do contexto
    fn fetch(&self, ctx: &Context) -> Result<ProgramInstr> {
        if let Some(idx) = self.pc_to_index(ctx.pc) {
            Ok(self.program[idx])
        } else {
            Err(anyhow!(
                "PC fora do programa: ctx {} PC {:032x} (program len {})",
                ctx.id,
                ctx.pc,
                self.program.len()
            ))
        }
    }

    // -----------------------------------------------------------------------
    // Loop principal
    // -----------------------------------------------------------------------

    /// Executa até HALT, esgotar max_steps ou não haver contexto pronto.
    pub fn run(&mut self) -> Result<VmStats> {
        log_info("vm", &format!("VM iniciando — target {} IPS", self.config.ips_target));
        loop {
            if let Some(max) = self.config.max_steps {
                if self.stats.steps >= max {
                    log_info("vm", &format!("max_steps {} atingido", max));
                    break;
                }
            }

            // Escalonamento estrito
            let ctx_id = match self.scheduler.pick_next() {
                Ok(id) => id,
                Err(_) => {
                    log_info("vm", "nenhum contexto pronto — VM ociosa, encerrando");
                    break;
                }
            };

            // Preempção: se durante pick um RED entrou, já foi priorizado

            // Fetch
            let instr = {
                let ctx = self.scheduler.get(ctx_id).unwrap();
                match self.fetch(ctx) {
                    Ok(i) => i,
                    Err(e) => {
                        log_warn("vm", &format!("ctx {} fetch falhou: {} — terminando contexto", ctx_id, e));
                        if let Some(c) = self.scheduler.get_mut(ctx_id) {
                            c.state = crate::context::ContextState::Terminated;
                        }
                        continue;
                    }
                }
            };

            if self.config.enable_tracing {
                let ctx = self.scheduler.get(ctx_id).unwrap();
                log_debug("vm", &format!("ctx {} PC {:032x} {}", ctx.id, ctx.pc, instr));
            }

            // HALT / NOP são tratados sem dispatch genérico para permitir PC advance correto
            match instr.opcode() {
                OP_HALT => {
                    log_info("vm", &format!("HALT em ctx {} step {}", ctx_id, self.stats.steps));
                    if let Some(c) = self.scheduler.get_mut(ctx_id) {
                        c.state = crate::context::ContextState::Terminated;
                    }
                    self.stats.steps += 1;
                    self.meter.tick(1);
                    // Se todos terminaram, encerra
                    if self.scheduler.active_count() == 0 {
                        break;
                    }
                    continue;
                }
                OP_NOP => {
                    // NOP: só avança PC (NOP é sempre 32B por R2).
                    debug_assert_eq!(instr.byte_len(), 32);
                    if let Some(c) = self.scheduler.get_mut(ctx_id) {
                        c.advance_pc();
                        c.state = crate::context::ContextState::Ready;
                    }
                    self.scheduler.yield_current();
                    self.stats.steps += 1;
                    self.meter.tick(1);
                    continue;
                }
                _ => {}
            }

            // Stride real da instrução buscada (W1-remainder): o avanço
            // de PC usa largura, nunca constante.
            let fetched_width = instr.byte_len() as u64;
            // Formas 64B: fetch OK; dispatch chega nas Fases 7/9. Rejeição
            // limpa aqui (nunca misdispatch): termina só o contexto.
            let instr: Instruction = match instr {
                ProgramInstr::W32(i) => i,
                ProgramInstr::W64(g) => {
                    log_warn("vm", &format!("ctx {} opcode 0x{:02x} 64B sem dispatch (Fases 7/9) — terminando contexto", ctx_id, g.opcode));
                    if let Some(c) = self.scheduler.get_mut(ctx_id) {
                        c.state = crate::context::ContextState::Terminated;
                    }
                    self.scheduler.yield_current();
                    self.stats.steps += 1;
                    self.meter.tick(1);
                    continue;
                }
            };

            // Execute — erro não deve derrubar VM inteira, apenas o contexto
            let should_advance_pc = match self.execute_instruction(ctx_id, &instr) {
                Ok(v) => v,
                Err(e) => {
                    log_warn("vm", &format!("ctx {} execução falhou ({}): {} — terminando contexto", ctx_id, instr.mnemonic(), e));
                    if let Some(c) = self.scheduler.get_mut(ctx_id) {
                        c.state = crate::context::ContextState::Terminated;
                    }
                    self.scheduler.yield_current();
                    self.stats.steps += 1;
                    self.meter.tick(1);
                    continue;
                }
            };

            // Atualiza PC e re-enfileira (se não foi terminado/abortado/fork já fez)
            if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                if ctx.state == crate::context::ContextState::Running {
                    if should_advance_pc {
                        ctx.advance_pc_by(fetched_width);
                    }
                    ctx.state = crate::context::ContextState::Ready;
                }
            }
            // Round-robin dentro da prioridade
            self.scheduler.yield_current();

            self.stats.steps += 1;
            self.meter.tick(1);

            // Checagem periódica de throughput (a cada 100k)
            if self.stats.steps % 100_000 == 0 {
                let ips = self.meter.ips();
                log_info("vm", &format!("progresso {} steps, {:.0} IPS {}", self.stats.steps, ips, self.memory.stats()));
                if ips < self.config.ips_target as f64 * 0.5 && self.stats.steps > 200_000 {
                    log_warn("vm", &format!("IPS abaixo do target: {:.0} < {}", ips, self.config.ips_target));
                }
            }
        }

        let elapsed = self.stats.elapsed_ms();
        log_info(
            "vm",
            &format!(
                "VM encerrada: {} steps em {}ms, {:.0} IPS | {:?}",
                self.stats.steps,
                elapsed,
                self.meter.ips(),
                self.stats
            ),
        );
        Ok(self.stats.clone())
    }

    /// Executa uma instrução no contexto; retorna `true` se PC deve avançar automaticamente.
    fn execute_instruction(&mut self, ctx_id: u64, instr: &Instruction) -> Result<bool> {
        match instr.opcode {
            OP_TENSOR => {
                self.exec_tensor(ctx_id, instr)?;
                Ok(true)
            }
            OP_ATTN => {
                self.exec_attn(ctx_id, instr)?;
                Ok(true)
            }
            OP_STREAM => {
                self.exec_stream(ctx_id, instr)?;
                Ok(true)
            }
            OP_FORK => {
                self.exec_fork(ctx_id, instr)?;
                // FORK já avançou PC do pai? Não, pai avança normalmente, filho já nasce com PC+32
                Ok(true)
            }
            OP_ABORT => {
                self.exec_abort(ctx_id, instr)?;
                // ABORT não avança PC do alvo (ele é terminado), mas o autor avança
                Ok(true)
            }
            OP_SENSE => {
                self.exec_sense(ctx_id, instr)?;
                Ok(true)
            }
            OP_NORM => {
                self.exec_norm(ctx_id, instr)?;
                Ok(true)
            }
            OP_FFN => {
                self.exec_ffn(ctx_id, instr)?;
                Ok(true)
            }
            OP_EMBED => {
                self.exec_embed(ctx_id, instr)?;
                Ok(true)
            }
            OP_ADD => {
                self.exec_add(ctx_id, instr)?;
                Ok(true)
            }
            OP_SAMPLE => {
                self.exec_sample(ctx_id, instr)?;
                Ok(true)
            }
            OP_COMPARE => {
                self.exec_compare(ctx_id, instr)?;
                Ok(true)
            }
            OP_JUMP => {
                self.exec_jump(ctx_id, instr)?;
                // JUMP define PC diretamente — sem avanço automático
                Ok(false)
            }
            OP_IF_EQUAL => {
                let jumped = self.exec_if_equal(ctx_id, instr)?;
                Ok(!jumped)
            }
            OP_IF_INTERRUPT => {
                let jumped = self.exec_if_interrupt(ctx_id, instr)?;
                Ok(!jumped)
            }
            OP_MATVEC => {
                self.exec_matvec(ctx_id, instr)?;
                Ok(true)
            }
            OP_MUL => {
                self.exec_mul(ctx_id, instr)?;
                Ok(true)
            }
            OP_SILU => {
                self.exec_silu(ctx_id, instr)?;
                Ok(true)
            }
            OP_SSM_SCAN => {
                self.exec_ssm_scan(ctx_id, instr)?;
                Ok(true)
            }
            OP_SSM_RESET => {
                self.exec_ssm_reset(ctx_id, instr)?;
                Ok(true)
            }
            OP_CODEC_ENC => {
                self.exec_codec_enc(ctx_id, instr)?;
                Ok(true)
            }
            OP_CODEC_DEC => {
                self.exec_codec_dec(ctx_id, instr)?;
                Ok(true)
            }
            OP_AUDIO_ALIGN => {
                self.exec_audio_align(ctx_id, instr)?;
                Ok(true)
            }
            OP_CTX_SWITCH => {
                self.exec_ctx_switch(ctx_id, instr)?;
                Ok(true)
            }
            OP_ROPE => {
                self.exec_rope(ctx_id, instr)?;
                Ok(true)
            }
            OP_GATHER => {
                self.exec_gather(ctx_id, instr)?;
                Ok(true)
            }
            OP_DISTANCE => {
                self.exec_distance(ctx_id, instr)?;
                Ok(true)
            }
            OP_RANK1_UPDATE => {
                self.exec_rank1_update(ctx_id, instr)?;
                Ok(true)
            }
            OP_RNG_SEED => {
                self.exec_rng_seed(ctx_id, instr)?;
                Ok(true)
            }
            OP_RNG_NEXT => {
                self.exec_rng_next(ctx_id, instr)?;
                Ok(true)
            }
            OP_RNG_UNIFORM => {
                self.exec_rng_uniform(ctx_id, instr)?;
                Ok(true)
            }
            OP_RNG_NORMAL => {
                self.exec_rng_normal(ctx_id, instr)?;
                Ok(true)
            }
            OP_HASH => {
                self.exec_hash(ctx_id, instr)?;
                Ok(true)
            }
            OP_CHECKSUM => {
                self.exec_checksum(ctx_id, instr)?;
                Ok(true)
            }
            OP_HMAC => {
                self.exec_hmac(ctx_id, instr)?;
                Ok(true)
            }
            OP_CYCLES_COUNT => {
                self.exec_cycles_count(ctx_id, instr)?;
                Ok(true)
            }
            OP_TRACE_EVENT => {
                self.exec_trace_event(ctx_id, instr)?;
                Ok(true)
            }
            OP_SANITY_CHECK => {
                self.exec_sanity_check(ctx_id, instr)?;
                Ok(true)
            }
            OP_PREEMPT_CHECK => {
                self.exec_preempt_check(ctx_id, instr)?;
                Ok(true)
            }
            OP_ASSERT => {
                self.exec_assert(ctx_id, instr)?;
                Ok(true)
            }
            OP_DUMP => {
                self.exec_dump(ctx_id, instr)?;
                Ok(true)
            }
            OP_YIELD => {
                self.exec_yield(ctx_id, instr)?;
                Ok(true)
            }
            OP_SET_DEADLINE => {
                self.exec_set_deadline(ctx_id, instr)?;
                Ok(true)
            }
            OP_GET_DEADLINE => {
                self.exec_get_deadline(ctx_id, instr)?;
                Ok(true)
            }
            OP_PRIORITY_SET => {
                self.exec_priority_set(ctx_id, instr)?;
                Ok(true)
            }
            OP_PRIORITY_GET => {
                self.exec_priority_get(ctx_id, instr)?;
                Ok(true)
            }
            OP_LOCK => {
                self.exec_lock(ctx_id, instr)?;
                Ok(true)
            }
            OP_UNLOCK => {
                self.exec_unlock(ctx_id, instr)?;
                Ok(true)
            }
            OP_FENCE => {
                self.exec_fence(ctx_id, instr)?;
                Ok(true)
            }
            OP_LOADI => {
                self.exec_loadi(ctx_id, instr)?;
                Ok(true)
            }
            OP_MOV => {
                self.exec_mov(ctx_id, instr)?;
                Ok(true)
            }
            OP_KV_TRUNCATE => {
                self.exec_kv_truncate(ctx_id, instr)?;
                Ok(true)
            }
            OP_DENOISE_STEP => {
                self.exec_denoise_step(ctx_id, instr)?;
                Ok(true)
            }
            OP_ODE_STEP => {
                self.exec_ode_step(ctx_id, instr)?;
                Ok(true)
            }
            OP_SPIKE_STEP => {
                self.exec_spike_step(ctx_id, instr)?;
                Ok(true)
            }
            OP_CONV => {
                self.exec_conv(ctx_id, instr)?;
                Ok(true)
            }
            OP_REMOTE_SPAWN => {
                self.exec_remote_spawn(ctx_id, instr)?;
                Ok(true)
            }
            OP_SIGNAL => {
                self.exec_signal(ctx_id, instr)?;
                Ok(true)
            }
            OP_SEND_TENSOR => {
                self.exec_send_tensor(ctx_id, instr)?;
                Ok(true)
            }
            OP_BARRIER => {
                self.exec_barrier(ctx_id, instr)?;
                Ok(true)
            }
            OP_SLICE => {
                self.exec_slice(ctx_id, instr)?;
                Ok(true)
            }
            OP_ARENA_ALLOC => {
                self.exec_arena_alloc(ctx_id, instr)?;
                Ok(true)
            }
            OP_ARENA_RESET => {
                self.exec_arena_reset(ctx_id, instr)?;
                Ok(true)
            }
            OP_MEMCPY => {
                self.exec_memcpy(ctx_id, instr)?;
                Ok(true)
            }
            OP_MEMSET => {
                self.exec_memset(ctx_id, instr)?;
                Ok(true)
            }
            OP_SNAPSHOT => {
                self.exec_snapshot(ctx_id, instr)?;
                Ok(true)
            }
            OP_RESTORE => {
                self.exec_restore(ctx_id, instr)?;
                Ok(true)
            }
            OP_PREFETCH => {
                self.exec_prefetch(ctx_id, instr)?;
                Ok(true)
            }
            OP_RESHAPE => {
                self.exec_reshape(ctx_id, instr)?;
                Ok(true)
            }
            OP_CONCAT => {
                self.exec_concat(ctx_id, instr)?;
                Ok(true)
            }
            OP_CAST => {
                self.exec_cast(ctx_id, instr)?;
                Ok(true)
            }
            OP_QUANTIZE => {
                self.exec_quantize(ctx_id, instr)?;
                Ok(true)
            }
            OP_DEQUANT => {
                self.exec_dequant(ctx_id, instr)?;
                Ok(true)
            }
            OP_ADD_IMM => {
                self.exec_add_imm(ctx_id, instr)?;
                Ok(true)
            }
            OP_SUB_IMM => {
                self.exec_sub_imm(ctx_id, instr)?;
                Ok(true)
            }
            OP_STEPS => {
                self.exec_steps(ctx_id, instr)?;
                Ok(true)
            }
            OP_SORT => {
                self.exec_sort(ctx_id, instr)?;
                Ok(true)
            }
            OP_TOPK => {
                self.exec_topk(ctx_id, instr)?;
                Ok(true)
            }
            OP_ARGMAX => {
                self.exec_argmax(ctx_id, instr)?;
                Ok(true)
            }
            OP_REDUCE => {
                self.exec_reduce(ctx_id, instr)?;
                Ok(true)
            }
            OP_BROADCAST => {
                self.exec_broadcast(ctx_id, instr)?;
                Ok(true)
            }
            OP_PAD => {
                self.exec_pad(ctx_id, instr)?;
                Ok(true)
            }
            OP_TILE => {
                self.exec_tile(ctx_id, instr)?;
                Ok(true)
            }
            OP_TRANSPOSE => {
                self.exec_transpose(ctx_id, instr)?;
                Ok(true)
            }
            OP_SOFTMAX => {
                self.exec_softmax(ctx_id, instr)?;
                Ok(true)
            }
            OP_GELU => {
                self.exec_unary(ctx_id, instr, crate::activations::gelu, "GELU")?;
                self.stats.gelu_execs += 1;
                Ok(true)
            }
            OP_SIGMOID => {
                self.exec_unary(ctx_id, instr, crate::activations::sigmoid, "SIGMOID")?;
                self.stats.sigmoid_execs += 1;
                Ok(true)
            }
            OP_TANH => {
                self.exec_unary(ctx_id, instr, f32::tanh, "TANH")?;
                self.stats.tanh_execs += 1;
                Ok(true)
            }
            OP_RELU => {
                self.exec_unary(ctx_id, instr, |x: f32| x.max(0.0), "RELU")?;
                self.stats.relu_execs += 1;
                Ok(true)
            }
            OP_EXP => {
                self.exec_unary(ctx_id, instr, f32::exp, "EXP")?;
                self.stats.exp_execs += 1;
                Ok(true)
            }
            OP_LOG => {
                self.exec_unary(ctx_id, instr, f32::ln, "LOG")?;
                self.stats.log_execs += 1;
                Ok(true)
            }
            OP_CLIP => {
                self.exec_clip(ctx_id, instr)?;
                Ok(true)
            }
            OP_FOREST => {
                self.exec_forest(ctx_id, instr)?;
                Ok(true)
            }
            OP_KV_COMPRESS => {
                self.exec_kv_compress(ctx_id, instr)?;
                Ok(true)
            }
            OP_FLASH_ATTN => {
                self.exec_flash_attn(ctx_id, instr)?;
                Ok(true)
            }
            OP_ATTN_SPARSE => {
                self.exec_attn_sparse(ctx_id, instr)?;
                Ok(true)
            }
            OP_STREAM_MERGE => {
                self.exec_stream_merge(ctx_id, instr)?;
                Ok(true)
            }
            OP_VAD_DETECT => {
                self.exec_vad_detect(ctx_id, instr)?;
                Ok(true)
            }
            OP_AUDIO_RESAMPLE => {
                self.exec_audio_resample(ctx_id, instr)?;
                Ok(true)
            }
            OP_AUDIO_FILTER => {
                self.exec_audio_filter(ctx_id, instr)?;
                Ok(true)
            }
            OP_AUDIO_WINDOW => {
                self.exec_audio_window(ctx_id, instr)?;
                Ok(true)
            }
            OP_DEPFORMER => {
                self.exec_depformer(ctx_id, instr)?;
                Ok(true)
            }
            OP_CALL => {
                self.exec_call(ctx_id, instr)?;
                // CALL define PC diretamente — sem avanço automático
                Ok(false)
            }
            OP_RET => {
                self.exec_ret(ctx_id, instr)?;
                // RET define PC diretamente — sem avanço automático
                Ok(false)
            }
            _ => Err(anyhow!("opcode não implementado: 0x{:02x}", instr.opcode)),
        }
    }

    /// Passo único público: executa 1 instrução do contexto; retorna se o PC
    /// deve avançar (pulos retornam `false`).
    ///
    /// Contrato de interpretadores: o modo `--interactive` (main.rs) delega
    /// aqui as ops puras (TENSOR/EMBED/ADD/SAMPLE/COMPARE/JUMP/IF_EQUAL/
    /// IF_INTERRUPT) para não duplicar a ISA. ATTN/NORM/FFN/STREAM/FORK/ABORT/
    /// SENSE seguem stubs de demo no modo interativo (instrumentação de
    /// preempção, outputs e checkpoints da tese).
    pub fn step_instruction(&mut self, ctx_id: u64, instr: &Instruction) -> Result<bool> {
        self.execute_instruction(ctx_id, instr)
    }

    // -----------------------------------------------------------------------
    // Implementações dos opcodes
    // -----------------------------------------------------------------------

    fn exec_tensor(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // TENSOR rdest, rsrc1(shape), rsrc2(dtype) com payload fallback rows/cols/dtype
        let (rows, cols) = instr.tensor_shape();
        let dtype_byte = instr.tensor_dtype();
        let dtype = DType::from_u8(dtype_byte);

        // Se rsrc1/rsrc2 forem registradores válidos, tenta ler shape/dtype deles como override?
        // Para stub, prioriza payload; se payload for default 2x2 e registrador contiver valor, usa registrador.
        let mut final_rows = rows;
        let final_cols = cols;
        let mut final_dtype = dtype;

        // Permite que shape venha via registradores como imediato (compatibilidade com prompt)
        if instr.rsrc1 != 0xFF {
            if let Some(ctx) = self.scheduler.get(ctx_id) {
                if let Ok(val) = ctx.reg(instr.rsrc1) {
                    if val != 0 && val < 1_000_000 {
                        // heurística: se valor pequeno e payload era default, usa val como rows*cols?
                        // Vamos interpretar val como total elementos; tenta fatorar em quadrado
                        // Simplificação: val = rows
                        final_rows = val as usize;
                    }
                }
            }
        }
        if instr.rsrc2 != 0xFF {
            if let Some(ctx) = self.scheduler.get(ctx_id) {
                if let Ok(val) = ctx.reg(instr.rsrc2) {
                    if val <= 3 {
                        final_dtype = DType::from_u8(val as u8);
                    }
                }
            }
        }

        let shape = vec![final_rows, final_cols];
        let elems = final_rows * final_cols;
        // NOP+Sparse: se flag SPARSE, aloca CSR
        let addr = if instr.is_sparse() {
            let dens = instr.sparse_density();
            self.memory.alloc_sparse_tensor(&shape, final_dtype, dens)?
        } else {
            let addr = self.memory.alloc_tensor(&shape, final_dtype)?;
            // Se foi mapeado para PERSISTENTE (GGUF zero-copy), não inicializa — dados já estão no mmap
            let is_persist = crate::memory::region_of(addr) == crate::memory::Region::Persistent;
            if !is_persist {
                // RFC-0019: FILL escalar (só F32; resto veta explícito).
                if let Some(f) = instr.tensor_fill() {
                    if final_dtype != DType::F32 {
                        return Err(anyhow!("TENSOR FILL só em f32 (dtype {:?})", final_dtype));
                    }
                    let init_data: Vec<f32> = vec![f; elems];
                    self.memory.write_f32_tensor(addr, &init_data)?;
                } else {
                match final_dtype {
                    DType::F32 => {
                        let init_data: Vec<f32> = (0..elems).map(|i| (i as f32 + 1.0) * 0.5).collect();
                        self.memory.write_f32_tensor(addr, &init_data)?;
                    }
                    DType::F16 => {
                        // F16: escreve como bytes (2 por elem) — mock com f32 truncado
                        let init_bytes: Vec<u8> = (0..elems*2).map(|i| (i % 256) as u8).collect();
                        self.memory.write(addr, &init_bytes)?;
                    }
                    DType::U8 | DType::I8 => {
                        let init_bytes: Vec<u8> = (0..elems).map(|i| (i % 256) as u8).collect();
                        self.memory.write(addr, &init_bytes)?;
                    }
                    _ => {
                        if final_dtype.is_quantized() {
                            // Quantizado: não inicializa, virá do GGUF se shape bater
                        }
                    }
                }
                }
            }
            addr
        };
        // Para esparso, dados já foram preenchidos aleatoriamente em alloc_sparse_tensor
        if instr.is_sparse() {
            log_debug("tensor", &format!("ctx {} TENSOR-SPARSE r{} <- 0x{:032x} shape {:?} density {}", ctx_id, instr.rdest, addr, shape, instr.sparse_density()));
        }

        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, addr)?;
        }

        self.stats.tensor_allocs += 1;
        log_debug(
            "tensor",
            &format!(
                "ctx {} TENSOR r{} <- 0x{:032x} shape {:?} dtype {:?} ({} elems)",
                ctx_id, instr.rdest, addr, shape, final_dtype, elems
            ),
        );
        Ok(())
    }

    fn exec_attn(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // ATTN rdest, rQ, rK, rV — suporta denso e esparso (CSR) com NOP por head
        let (addr_q, addr_k, addr_v) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let aq = ctx.reg(instr.rsrc1)?;
            let ak = ctx.reg(instr.rsrc2)?;
            let av = ctx.reg(instr.rsrc3)?;
            (aq, ak, av)
        };

        let is_sparse = self.memory.is_sparse(addr_q) || self.memory.is_sparse(addr_k) || self.memory.is_sparse(addr_v);
        let notify_each_head = (instr.flags & crate::opcodes::ATTN_FLAG_NOTIFY_EACH_HEAD) != 0;
        // KV_CACHE (0x30) — tese 2.1: se K ou V estiver em KV_CACHE, usa cache acumulado
        let is_kv = crate::memory::region_of(addr_k) == crate::memory::Region::KvCache || crate::memory::region_of(addr_v) == crate::memory::Region::KvCache;
        if is_kv {
            // Tenta attention via KV cache do MemoryManager (loop 22)
            // Para VM, usa layer 0 se não houver mapping específico, ou infere layer do offset
            let n_layers = self.memory.kv_cache_n_layers();
            if n_layers > 0 {
                // Infere layer do offset K (addr_k % n_layers) para distribuir 22 camadas
                let offset_k = crate::memory::kv_cache_offset(addr_k);
                let layer = if n_layers > 0 { (offset_k / 4096) % n_layers } else { 0 };
                // Lê Q como vetor hidden (assume shape [1, hidden] ou [hidden])
                let meta_q = self.memory.get_tensor_meta(addr_q).cloned();
                let shape_q = meta_q.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![1, 8]);
                let n_q: usize = shape_q.iter().product();
                let q_data = self.memory.read_f32_tensor(addr_q, n_q).unwrap_or(vec![0.5; 8]);
                let hidden = shape_q.last().cloned().unwrap_or(8);
                let q_vec = if q_data.len() >= hidden { q_data[q_data.len()-hidden..].to_vec() } else { q_data.clone() };
                // Se cache vazio para layer, inicializa com K/V atuais (fallback para primeiro token)
                if self.memory.kv_cache_seq_len() == 0 {
                    // Para demo, apenas aloca saída densa dummy e não falha
                } else {
                    // Tenta attn via cache; se falhar, fallback para denso
                    if let Ok(attn_out) = self.memory.kv_cache_attention(layer, &q_vec) {
                        // Projeta via O (se houver) — para simplificar, retorna attn_out direto como saída
                        let out_vec = attn_out;
                        let out_shape = vec![shape_q[0], hidden];
                        if let Ok(out_addr) = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32) {
                            let _ = self.memory.write_f32_tensor(out_addr, &out_vec);
                            if let Some(ctx) = self.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, out_addr); }
                            self.stats.attn_execs += 1;
                            return Ok(());
                        }
                    }
                }
            }
            // Fallback: trata como denso normal se cache não inicializado
        }

        if is_sparse {
            // Caminho esparso PIM: sem mover dados, computa onde está
            let q_st = self.memory.get_sparse(addr_q).cloned();
            let k_st = self.memory.get_sparse(addr_k).cloned();
            let v_st = self.memory.get_sparse(addr_v).cloned();
            // Fallback: se algum não for esparso, converte denso->esparso temporariamente
            let q_sp = if let Some(s) = q_st { s } else {
                let meta = self.memory.get_tensor_meta(addr_q).cloned().unwrap_or_else(|| crate::memory::TensorMeta { addr: addr_q, shape: vec![2,2], dtype: DType::F32, byte_len: 16, is_sparse: false, density: 1.0 });
                let n: usize = meta.shape.iter().product();
                let data = self.memory.read_f32_tensor(addr_q, n).unwrap_or(vec![0.5; n]);
                crate::sparse::SparseTensor::from_dense(&data, (meta.shape[0], meta.shape[1]))
            };
            let k_sp = if let Some(s) = k_st { s } else {
                let meta = self.memory.get_tensor_meta(addr_k).cloned().unwrap_or_else(|| crate::memory::TensorMeta { addr: addr_k, shape: vec![2,2], dtype: DType::F32, byte_len: 16, is_sparse: false, density: 1.0 });
                let n: usize = meta.shape.iter().product();
                let data = self.memory.read_f32_tensor(addr_k, n).unwrap_or(vec![0.5; n]);
                crate::sparse::SparseTensor::from_dense(&data, (meta.shape[0], meta.shape[1]))
            };
            let v_sp = if let Some(s) = v_st { s } else {
                let meta = self.memory.get_tensor_meta(addr_v).cloned().unwrap_or_else(|| crate::memory::TensorMeta { addr: addr_v, shape: vec![2,2], dtype: DType::F32, byte_len: 16, is_sparse: false, density: 1.0 });
                let n: usize = meta.shape.iter().product();
                let data = self.memory.read_f32_tensor(addr_v, n).unwrap_or(vec![0.5; n]);
                crate::sparse::SparseTensor::from_dense(&data, (meta.shape[0], meta.shape[1]))
            };

            // Se NOTIFY_EACH_HEAD, simula notificação por head via log (reator faz watch real)
            if notify_each_head {
                log_debug("attn-sparse", &format!("ctx {} ATTN-SPARSE NOTIFY_EACH_HEAD: {} heads", ctx_id, q_sp.shape.0));
            }

            let out_sp = crate::sparse::attn_sparse(&q_sp, &k_sp, &v_sp, None)?;
            let out_shape = vec![out_sp.shape.0, out_sp.shape.1];
            // Aloca saída esparsa (densidade herdada)
            let dens = out_sp.density;
            let out_addr = self.memory.alloc_sparse_tensor(&out_shape, DType::F32, dens)?;
            // Substitui CSR gerado aleatoriamente pelo resultado real
            if let Some(slot) = self.memory.get_sparse_mut(out_addr) {
                *slot = out_sp;
            }
            if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
            self.stats.attn_execs += 1;
            log_debug("attn", &format!("ctx {} ATTN-SPARSE r{} <- Q:0x{:x} K:0x{:x} V:0x{:x} => 0x{:x} shape {:?} dens {:.2}", ctx_id, instr.rdest, addr_q, addr_k, addr_v, out_addr, out_shape, dens));
            return Ok(());
        }

        // Caminho Vega 8 (wgpu) — só para ATTN grande: medido que GPU perde
        // feio abaixo de ~10M elems (launch+buffers+readback) e só vence no
        // grande (bench_gpu_gemv). Demos 64x64 ficam no CPU (rápido e testado).
        #[cfg(feature = "wgpu")]
        if self.memory.is_gpu() {
            let meta_q = self.memory.get_tensor_meta(addr_q).cloned();
            let meta_k = self.memory.get_tensor_meta(addr_k).cloned();
            let meta_v = self.memory.get_tensor_meta(addr_v).cloned();
            let shape_q = meta_q.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2,2]);
            let shape_k = meta_k.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2,2]);
            let shape_v = meta_v.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2,2]);
            // FLOPs aprox: QK^T (2*m*n*d) + SM*V (2*m*n*p)
            let flops = 2 * shape_q[0] * shape_k[0] * shape_q[1]
                + 2 * shape_q[0] * shape_k[0] * shape_v[1];
            if flops >= 8_000_000 {
                // Tenta ATTN na Vega 8 — extrai shapes antes do borrow mutável
                let q_sh = (shape_q[0], shape_q[1]);
                let k_sh = (shape_k[0], shape_k[1]);
                let v_sh = (shape_v[0], shape_v[1]);
                let gpu_result: Option<Result<(u128, Vec<f32>)>> = if let MemBackend::Gpu(gpu_mem) = &mut self.memory {
                    Some(gpu_mem.attn_wgpu(addr_q, addr_k, addr_v, q_sh, k_sh, v_sh))
                } else { None };
                if let Some(Ok((out_addr, out_data))) = gpu_result {
                    if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                        ctx.set_reg(instr.rdest, out_addr)?;
                    }
                    self.stats.attn_execs += 1;
                    log_info("attn-wgpu", &format!("ctx {} ATTN-WGPU Vega8 r{} => 0x{:x} {} elems em GPU", ctx_id, instr.rdest, out_addr, out_data.len()));
                    return Ok(());
                } else if let Some(Err(e)) = gpu_result {
                    log_warn("attn-wgpu", &format!("Vega 8 fallback CPU: {}", e));
                }
            }
        }

        // Caminho denso (legado)
        // Recupera metas para shape
        let meta_q = self.memory.get_tensor_meta(addr_q).cloned();
        let meta_k = self.memory.get_tensor_meta(addr_k).cloned();
        let meta_v = self.memory.get_tensor_meta(addr_v).cloned();

        // Fallback para 2x2 se meta não encontrada (robustez)
        let shape_q = meta_q.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2, 2]);
        let shape_k = meta_k.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2, 2]);
        let shape_v = meta_v.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2, 2]);

        // Lê tensores como f32
        let n_q: usize = shape_q.iter().product();
        let n_k: usize = shape_k.iter().product();
        let n_v: usize = shape_v.iter().product();

        let q_data = self.memory.read_f32_tensor(addr_q, n_q)?;
        let k_data = self.memory.read_f32_tensor(addr_k, n_k)?;
        let v_data = self.memory.read_f32_tensor(addr_v, n_v)?;

        // Converte para ndarray Array2
        // Assume 2D [rows, cols]
        let q_rows = shape_q[0];
        let q_cols = shape_q[1];
        let k_rows = shape_k[0];
        let k_cols = shape_k[1];
        let v_rows = shape_v[0];
        let v_cols = shape_v[1];

        // Validação mínima de compatibilidade para atenção
        if q_cols != k_cols {
            return Err(anyhow!("ATTN incompatível: Q cols {} != K cols {}", q_cols, k_cols));
        }
        if k_rows != v_rows {
            return Err(anyhow!("ATTN incompatível: K rows {} != V rows {}", k_rows, v_rows));
        }

        let q = Array2::from_shape_vec((q_rows, q_cols), q_data)
            .map_err(|e| anyhow!("q reshape: {}", e))?;
        let k = Array2::from_shape_vec((k_rows, k_cols), k_data)
            .map_err(|e| anyhow!("k reshape: {}", e))?;
        let v = Array2::from_shape_vec((v_rows, v_cols), v_data)
            .map_err(|e| anyhow!("v reshape: {}", e))?;

        // FlashAttention simplificada:
        //   scores = Q * K^T / sqrt(d_k)
        //   softmax por linha
        //   out = scores * V
        let d_k = q_cols as f32;
        let scale = 1.0 / d_k.sqrt();

        let kt = k.t();
        let mut scores = q.dot(&kt); // (q_rows, k_rows)
        scores.mapv_inplace(|x| x * scale);

        // Softmax por linha (estável)
        for mut row in scores.axis_iter_mut(Axis(0)) {
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v /= sum;
            }
        }

        let out = scores.dot(&v); // (q_rows, v_cols)
        let out_rows = out.nrows();
        let out_cols = out.ncols();
        let out_vec: Vec<f32> = out.iter().cloned().collect();

        // Aloca tensor de saída
        let out_shape = vec![out_rows, out_cols];
        let out_addr = self.memory.alloc_tensor(&out_shape, DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;

        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }

        self.stats.attn_execs += 1;
        log_debug(
            "attn",
            &format!(
                "ctx {} ATTN r{} <- attn(Q:0x{:x} K:0x{:x} V:0x{:x}) => 0x{:x} shape {:?}",
                ctx_id, instr.rdest, addr_q, addr_k, addr_v, out_addr, out_shape
            ),
        );
        Ok(())
    }

    fn exec_stream(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // Periféricos especiais detectados via raw rsrc2 (sem dereferenciar) para suportar `STREAM r, SAMPLE`
        if instr.rsrc2 == 2 {
            let src_addr = {
                let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
                if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? } else { 0 }
            };
            let logits: Vec<f32> = if let Some(meta) = self.memory.get_tensor_meta(src_addr).cloned() {
                let n: usize = meta.shape.iter().product();
                self.memory.read_f32_tensor(src_addr, n).unwrap_or_else(|_| vec![0.5; 4])
            } else {
                self.memory.read(src_addr, 64).ok().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()).unwrap_or(vec![0.5;4])
            };
            let tok = self.sample_logits_ctx(ctx_id, &logits)?;
            self.last_sample = tok;
            log_info("sample", &format!("ctx {} SAMPLE logits {} -> token {} ({} logits)", ctx_id, src_addr, tok, logits.len()));
            self.stats.streams += 1;
            return Ok(());
        }
        if instr.rsrc2 == 4 {
            let src_addr = {
                let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
                if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? } else { 0 }
            };
            let token_id: u32 = if src_addr < 100_000 && src_addr != 0 {
                src_addr as u32
            } else if src_addr != 0 {
                self.memory.read(src_addr, 4).ok().and_then(|b| if b.len()>=4 { Some(u32::from_le_bytes([b[0],b[1],b[2],b[3]])) } else { None }).unwrap_or(self.last_sample)
            } else {
                self.last_sample
            };
            let text = if let Some(tok) = &self.tokenizer { tok.decode(token_id) } else { let vocab = ["Olá","mundo","M³","AVM","DeepSeek","Qwen","token","teste"]; vocab[(token_id as usize) % vocab.len()].to_string() };
            print!("{}", text);
            use std::io::Write;
            let _ = std::io::stdout().flush();
            log_info("sample", &format!("ctx {} OUTPUT_DECODED token {} -> '{}'", ctx_id, token_id, text));
            self.stats.streams += 1;
            return Ok(());
        }

        let (src_addr, sink_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let s = if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? } else { 0 };
            let d = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? } else { 0 };
            (s, d)
        };

        // --- Periféricos especiais host-side via valor dereferenciado (compat) ---
        if sink_addr == crate::opcodes::STREAM_PERIPHERAL_SAMPLE {
            // SAMPLE: src é tensor de logits f32 -> softmax+argmax -> guarda last_sample
            let logits: Vec<f32> = if let Some(meta) = self.memory.get_tensor_meta(src_addr).cloned() {
                let n: usize = meta.shape.iter().product();
                self.memory.read_f32_tensor(src_addr, n).unwrap_or_else(|_| vec![0.5; 4])
            } else {
                // Tenta ler 64 bytes como f32
                self.memory.read(src_addr, 64).ok().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()).unwrap_or(vec![0.5;4])
            };
            let tok = self.sample_logits_ctx(ctx_id, &logits)?;
            self.last_sample = tok;
            log_info("sample", &format!("ctx {} SAMPLE logits {} -> token {} ({} logits)", ctx_id, src_addr, tok, logits.len()));
            self.stats.streams += 1;
            return Ok(());
        }
        if sink_addr == crate::opcodes::STREAM_PERIPHERAL_OUTPUT_DECODED {
            // OUTPUT_DECODED: src contém token id (u32) -> decodifica via tokenizer
            let token_id: u32 = if src_addr < 100_000 {
                src_addr as u32 // valor direto se pequeno (ex: 0..vocab)
            } else {
                // Lê 4 bytes de memória
                self.memory.read(src_addr, 4).ok().and_then(|b| if b.len()>=4 { Some(u32::from_le_bytes([b[0],b[1],b[2],b[3]])) } else { None }).unwrap_or(self.last_sample)
            };
            let text = if let Some(tok) = &self.tokenizer {
                tok.decode(token_id)
            } else {
                // fallback mock
                let vocab = ["Olá","mundo","M³","AVM","DeepSeek","Qwen","token","teste"];
                vocab[(token_id as usize) % vocab.len()].to_string()
            };
            print!("{}", text);
            use std::io::Write;
            let _ = std::io::stdout().flush();
            log_info("sample", &format!("ctx {} OUTPUT_DECODED token {} -> '{}'", ctx_id, token_id, text));
            self.stats.streams += 1;
            return Ok(());
        }

        let is_blocking = (instr.flags & STREAM_FLAG_BLOCKING) != 0;

        // Stub: tenta ler dado da fonte (se for endereço GLOBAL válido, lê 64 bytes)
        // Caso contrário, usa payload dummy.
        let data = if src_addr != 0 {
            match self.memory.read(src_addr, 64) {
                Ok(b) => b,
                Err(_) => format!("stream src 0x{:x} sink 0x{:x} blocking={}", src_addr, sink_addr, is_blocking)
                    .into_bytes(),
            }
        } else {
            // Se src é zero (ex: stdout), apenas loga
            b"<stream stdout>".to_vec()
        };

        // Simula backpressure via tokio mpsc
        // Se não há canal para sink, cria um.
        let chan_key = sink_addr;
        if !self.stream_channels.contains_key(&chan_key) {
            let (tx, mut rx) = mpsc::channel::<Vec<u8>>(16);
            self.stream_channels.insert(chan_key, tx);
            // Spawn consumidor que drena e imprime (não bloqueia VM)
            tokio::spawn(async move {
                while let Some(payload) = rx.recv().await {
                    // Em prod, escreveria em socket/file/GPU
                    // Aqui só conta bytes
                    let _ = payload.len();
                }
            });
        }

        if let Some(tx) = self.stream_channels.get(&chan_key) {
            let tx = tx.clone();
            let payload = data.clone();
            if is_blocking {
                // BLOCKING: tenta enviar com timeout curto; se cheio, espera (backpressure)
                // Como estamos em contexto síncrono, usamos try_send + fallback
                match tx.try_send(payload.clone()) {
                    Ok(_) => {},
                    Err(mpsc::error::TrySendError::Full(_)) => {
                        log_warn("stream", &format!("ctx {} STREAM backpressure FULL (BLOCKING) — drop ou wait", ctx_id));
                        // Em modo BLOCKING real, bloquearia; aqui dropamos após warn para não travar
                        // Alternativa: block_in_place com tokio
                    },
                    Err(mpsc::error::TrySendError::Closed(_)) => {
                        log_warn("stream", "canal fechado");
                    }
                }
            } else {
                // DROP: descarta se cheio
                let _ = tx.try_send(payload);
            }
        }

        // Também imprime em stdout para o exemplo .m3asm
        // Se sink_addr mapeia para stdout (convencional 0), escreve
        if sink_addr == 0 || sink_addr == 0xFF {
            println!("[STREAM ctx {}] {} bytes: {:02x?} (src 0x{:x} -> sink 0x{:x})", ctx_id, data.len(), &data[..data.len().min(16)], src_addr, sink_addr);
        }

        // Se rdest for válido, escreve número de bytes enviados
        if instr.rdest != 0xFF {
            if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                ctx.set_reg(instr.rdest, data.len() as u128)?;
            }
        }

        self.stats.streams += 1;
        log_debug("stream", &format!("ctx {} STREAM 0x{:x} -> 0x{:x} {} bytes flags 0x{:02x}", ctx_id, src_addr, sink_addr, data.len(), instr.flags));
        Ok(())
    }

    fn exec_fork(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let parent = self.scheduler.get(ctx_id).cloned().ok_or_else(|| anyhow!("ctx {} não encontrado para FORK", ctx_id))?;
        let child_prio = Priority::from_flags(instr.flags);
        let snap_version = self.memory.snapshot();
        // Forma com rótulo (FORK Rd, LABEL): filho começa no alvo; senão PC+32
        let child_pc = {
            let label_pc = instr.imm_u128();
            if label_pc != 0 {
                self.check_jump_target(label_pc)?;
                label_pc
            } else {
                parent.pc.wrapping_add(32)
            }
        };
        let new_id = self.scheduler.create_context(child_prio, child_pc, snap_version);
        // Copia registradores do pai para filho
        if let Some(child) = self.scheduler.get_mut(new_id) {
            child.regs = parent.regs;
            child.rng_state = parent.rng_state; // RFC-0005: herda stream RNG
            child.call_stack = parent.call_stack.clone(); // RFC-0034: Unix (retorna pela cadeia)
            child.root_version = snap_version;
        }
        // Retorna ID do filho em rdest do pai
        if let Some(parent_mut) = self.scheduler.get_mut(ctx_id) {
            parent_mut.set_reg(instr.rdest, new_id as u128)?;
        }
        // Snapshot SSM para rollback, carimbado com a versão (RFC-0011).
        self.ssm_snapshots.push((snap_version, self.ssm_states.clone()));
        // Snapshot dos handles RANK1 (mapa barato; tensores H são imutáveis).
        self.rank1_snapshots.push((snap_version, self.rank1_layers.clone()));
        // Snapshot dos handles SNN (idem; V/refr imutáveis por passo).
        self.snn_snapshots.push((snap_version, self.snn_layers.clone()));
        // Snapshot das arenas (mapa; buffers clonados — custo O(total), RFC-0023).
        self.arena_snapshots.push((snap_version, self.arenas.clone()));
        // Snapshot do KV depformer (RFC-0032; mesma disciplina).
        self.dep_snapshots.push((snap_version, self.dep_kv.clone()));
        self.stats.forks += 1;
        log_info("fork", &format!("ctx {} FORK -> child {} prio {} (snap v{})", ctx_id, new_id, child_prio, snap_version));
        Ok(())
    }

    /// EMBED rdest, rtoken, rtable — lookup de embedding: linha `token_id` da
    /// tabela [rows, hidden] vira tensor [1, hidden] em rdest.
    /// rtoken com valor < 1M é id direto; senão lê u32 LE do endereço.
    fn exec_embed(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (tok_val, table_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let token_id: usize = if tok_val < 1_000_000 {
            tok_val as usize
        } else {
            let b = self.memory.read(tok_val, 4)?;
            u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize
        };
        let meta = self.memory.get_tensor_meta(table_addr).cloned()
            .ok_or_else(|| anyhow!("EMBED: tabela não encontrada 0x{:x}", table_addr))?;
        if meta.shape.len() != 2 {
            return Err(anyhow!("EMBED: tabela precisa ser 2D, shape {:?}", meta.shape));
        }
        if meta.is_sparse {
            return Err(anyhow!("EMBED: tabela esparsa não suportada"));
        }
        let (rows, hidden) = (meta.shape[0], meta.shape[1]);
        let row = token_id % rows.max(1);
        let table = self.memory.read_f32_tensor(table_addr, rows * hidden)?;
        let out_vec = table[row * hidden..(row + 1) * hidden].to_vec();
        let out_addr = self.memory.alloc_tensor(&[1, hidden], DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.embed_execs += 1;
        log_debug("embed", &format!("ctx {} EMBED r{} <- tok {} tabela 0x{:x} => 0x{:x} [1,{}]", ctx_id, instr.rdest, token_id, table_addr, out_addr, hidden));
        Ok(())
    }

    /// ADD rdest, r1, r2 — soma elemento a elemento (f32 denso, shapes iguais).
    fn exec_add(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (a1, a2) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let m1 = self.memory.get_tensor_meta(a1).cloned()
            .ok_or_else(|| anyhow!("ADD: tensor 0x{:x} não encontrado", a1))?;
        let m2 = self.memory.get_tensor_meta(a2).cloned()
            .ok_or_else(|| anyhow!("ADD: tensor 0x{:x} não encontrado", a2))?;
        if m1.is_sparse || m2.is_sparse {
            return Err(anyhow!("ADD: esparso não suportado"));
        }
        if m1.shape != m2.shape {
            return Err(anyhow!("ADD: shapes {:?} != {:?}", m1.shape, m2.shape));
        }
        let n: usize = m1.shape.iter().product();
        let d1 = self.memory.read_f32_tensor(a1, n)?;
        let d2 = self.memory.read_f32_tensor(a2, n)?;
        let out_vec: Vec<f32> = d1.iter().zip(d2.iter()).map(|(a, b)| a + b).collect();
        let out_addr = self.memory.alloc_tensor(&m1.shape, DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.add_execs += 1;
        log_debug("add", &format!("ctx {} ADD r{} <- 0x{:x} + 0x{:x} => 0x{:x}", ctx_id, instr.rdest, a1, a2, out_addr));
        Ok(())
    }

    /// SAMPLE rdest, rlogits [TEMP=x] — softmax + amostragem; guarda token em
    /// rdest e em `last_sample` (igual ao sink SAMPLE do STREAM).
    fn exec_sample(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let src_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? } else { 0 }
        };
        let logits: Vec<f32> = if let Some(meta) = self.memory.get_tensor_meta(src_addr).cloned() {
            let n: usize = meta.shape.iter().product();
            self.memory.read_f32_tensor(src_addr, n).unwrap_or_else(|_| vec![0.5; 4])
        } else {
            self.memory.read(src_addr, 64).ok()
                .map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
                .unwrap_or(vec![0.5; 4])
        };
        // TOPK mode (RFC-0004 §4): payload[4..6]=k>0 escreve tensor de
        // índices [1,k] (top-k logits, desempate por menor índice) e NÃO
        // toca em last_sample. payload zero = amostragem legada.
        let topk = instr.sample_topk();
        if topk > 0 {
            let k = (topk as usize).min(logits.len());
            if k == 0 {
                return Err(anyhow!("SAMPLE: TOPK com logits vazios"));
            }
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| {
                logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            });
            order.truncate(k);
            let out: Vec<f32> = order.iter().map(|&i| i as f32).collect();
            let out_addr = self.memory.alloc_tensor(&[1, k], crate::memory::DType::F32)?;
            self.memory.write_f32_tensor(out_addr, &out)?;
            if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                if instr.rdest != 0xFF {
                    ctx.set_reg(instr.rdest, out_addr)?;
                }
            }
            self.stats.sample_execs += 1;
            log_debug("sample", &format!("ctx {} SAMPLE TOPK={} -> 0x{:x}", ctx_id, k, out_addr));
            return Ok(());
        }
        // Temp vai no payload[0..4] (instr_sample); <=0 ou inválido => 1.0
        let mut tb = [0u8; 4];
        tb.copy_from_slice(&instr.payload[0..4]);
        let temp = f32::from_le_bytes(tb);
        let scaled: Vec<f32> = if temp > 0.0 && temp.is_finite() && (temp - 1.0).abs() > f32::EPSILON {
            logits.iter().map(|v| v / temp).collect()
        } else {
            logits.clone()
        };
        let tok = self.sample_logits_ctx(ctx_id, &scaled)?;
        self.last_sample = tok;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, tok as u128)?;
            }
        }
        self.stats.sample_execs += 1;
        log_debug("sample", &format!("ctx {} SAMPLE {} logits -> token {}", ctx_id, scaled.len(), tok));
        Ok(())
    }

    /// COMPARE r1, r2|imm — `cmp_equal = (reg[r1] == reg[r2] ou imm)`.
    fn exec_compare(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (v1, v2) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let a = ctx.reg(instr.rsrc1)?;
            let b = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? } else { instr.imm_u128() };
            (a, b)
        };
        // Predicado (RFC-0007, payload[16]; 0 = EQ legado). u128 sem sinal:
        // regs carregam endereços/ids/contadores, não floats.
        use crate::opcodes::{CMP_EQ, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE};
        let result = match instr.compare_pred() {
            CMP_EQ => v1 == v2,
            CMP_NE => v1 != v2,
            CMP_LT => v1 < v2,
            CMP_LE => v1 <= v2,
            CMP_GT => v1 > v2,
            CMP_GE => v1 >= v2,
            p => return Err(anyhow!("COMPARE: PRED {} inválido (0-5)", p)),
        };
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.cmp_equal = result;
        }
        log_debug("compare", &format!("ctx {} COMPARE {} pred={} {} -> {}", ctx_id, v1, instr.compare_pred(), v2, result));
        Ok(())
    }

    /// Valida que um PC alvo cai em início de instrução do programa
    /// (tabela de offsets; em programa só-32B equivale ao `% 32` anterior).
    fn check_jump_target(&self, target: u128) -> Result<()> {
        if target < self.program_base {
            return Err(anyhow!("JUMP para {:032x} antes da base {:032x}", target, self.program_base));
        }
        if self.pc_to_index(target).is_some() {
            return Ok(());
        }
        if target >= self.program_end() {
            return Err(anyhow!("JUMP para {:032x} fora do programa (len {})", target, self.program.len()));
        }
        Err(anyhow!("JUMP para {:032x} desalinhado (não cai em início de instrução)", target))
    }

    /// JUMP alvo — pc = alvo (chamador não avança).
    fn exec_jump(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let target = instr.imm_u128();
        self.check_jump_target(target)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.pc = target;
        }
        log_debug("jump", &format!("ctx {} JUMP -> {:032x}", ctx_id, target));
        Ok(())
    }

    /// IF_EQUAL alvo — pula se `cmp_equal`; retorna se pulou.
    fn exec_if_equal(&mut self, ctx_id: u64, instr: &Instruction) -> Result<bool> {
        let take = self.scheduler.get(ctx_id).map(|c| c.cmp_equal).unwrap_or(false);
        if take {
            self.exec_jump(ctx_id, instr)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// IF_INTERRUPT [rcond,] alvo — pula se houver interrupção pendente:
    /// com registrador, `reg != 0`; sem, `interrupt_flag` (consumida no pulo).
    fn exec_if_interrupt(&mut self, ctx_id: u64, instr: &Instruction) -> Result<bool> {
        self.stats.interrupt_checks += 1;
        let take = if instr.rsrc1 != 0xFF {
            self.scheduler.get(ctx_id).map(|c| c.reg(instr.rsrc1).unwrap_or(0) != 0).unwrap_or(false)
        } else {
            self.scheduler.get(ctx_id).map(|c| c.interrupt_flag).unwrap_or(false)
        };
        if take {
            if instr.rsrc1 == 0xFF {
                if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                    ctx.interrupt_flag = false;
                }
            }
            self.exec_jump(ctx_id, instr)?;
            log_debug("if_interrupt", &format!("ctx {} IF_INTERRUPT tomado", ctx_id));
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn exec_abort(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (target_id, ts_version) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let tid = if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? as u64 } else { 0 };
            let ts = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? as u64 } else { 0 };
            (tid, ts)
        };

        if target_id == 0 {
            log_warn("abort", &format!("ctx {} ABORT alvo 0 — ignorado", ctx_id));
            return Ok(());
        }

        // Remove contexto alvo
        if let Some(removed) = self.scheduler.remove(target_id) {
            log_info("abort", &format!("ctx {} ABORT matou ctx {} (state {})", ctx_id, target_id, removed.state));
        } else {
            log_warn("abort", &format!("ctx {} ABORT alvo {} não encontrado", ctx_id, target_id));
        }

        // Restaura snapshot se ts_version != 0
        if ts_version != 0 {
            match self.memory.restore(ts_version) {
                Ok(_) => log_info("abort", &format!("ABORT restaurou memória para v{}", ts_version)),
                Err(e) => log_warn("abort", &format!("falha ao restaurar v{}: {}", ts_version, e)),
            }
        }
        // Rollback SSM versionado (RFC-0011): com ts != 0, descarta entradas
        // mais novas que o alvo e aplica a entrada exata, se houver; com
        // ts == 0, pop único legado. Pilha vazia = no-op (nunca panic).
        if ts_version != 0 {
            while self.ssm_snapshots.last().map(|(v, _)| *v > ts_version).unwrap_or(false) {
                self.ssm_snapshots.pop();
            }
            if self.ssm_snapshots.last().map(|(v, _)| *v == ts_version).unwrap_or(false) {
                if let Some((_, snap)) = self.ssm_snapshots.pop() {
                    self.ssm_states = snap;
                    log_info("abort", &format!("ctx {} ABORT restaurou {} estados SSM (v{})", ctx_id, self.ssm_states.len(), ts_version));
                }
            } else {
                log_info("abort", &format!("ctx {} ABORT sem snapshot SSM em v{} (mantido)", ctx_id, ts_version));
            }
        } else if let Some((_, snap)) = self.ssm_snapshots.pop() {
            self.ssm_states = snap;
            log_info("abort", &format!("ctx {} ABORT restaurou {} estados SSM", ctx_id, self.ssm_states.len()));
        }
        // Rollback RANK1: mesma disciplina sobre os handles (tensores H
        // antigos seguem válidos por CoW).
        if ts_version != 0 {
            while self.rank1_snapshots.last().map(|(v, _)| *v > ts_version).unwrap_or(false) {
                self.rank1_snapshots.pop();
            }
            if self.rank1_snapshots.last().map(|(v, _)| *v == ts_version).unwrap_or(false) {
                if let Some((_, snap)) = self.rank1_snapshots.pop() {
                    self.rank1_layers = snap;
                }
            }
        } else if let Some((_, snap)) = self.rank1_snapshots.pop() {
            self.rank1_layers = snap;
        }
        // Rollback SNN: idem (handles V/refr).
        if ts_version != 0 {
            while self.snn_snapshots.last().map(|(v, _)| *v > ts_version).unwrap_or(false) {
                self.snn_snapshots.pop();
            }
            if self.snn_snapshots.last().map(|(v, _)| *v == ts_version).unwrap_or(false) {
                if let Some((_, snap)) = self.snn_snapshots.pop() {
                    self.snn_layers = snap;
                }
            }
        } else if let Some((_, snap)) = self.snn_snapshots.pop() {
            self.snn_layers = snap;
        }
        // Rollback arenas: idem (buffers do filho descartados com o mapa).
        if ts_version != 0 {
            while self.arena_snapshots.last().map(|(v, _)| *v > ts_version).unwrap_or(false) {
                self.arena_snapshots.pop();
            }
            if self.arena_snapshots.last().map(|(v, _)| *v == ts_version).unwrap_or(false) {
                if let Some((_, snap)) = self.arena_snapshots.pop() {
                    self.arenas = snap;
                }
            }
        } else if let Some((_, snap)) = self.arena_snapshots.pop() {
            self.arenas = snap;
        }
        // Rollback depformer: idem (RFC-0032).
        if ts_version != 0 {
            while self.dep_snapshots.last().map(|(v, _)| *v > ts_version).unwrap_or(false) {
                self.dep_snapshots.pop();
            }
            if self.dep_snapshots.last().map(|(v, _)| *v == ts_version).unwrap_or(false) {
                if let Some((_, snap)) = self.dep_snapshots.pop() {
                    self.dep_kv = snap;
                }
            }
        } else if let Some((_, snap)) = self.dep_snapshots.pop() {
            self.dep_kv = snap;
        }

        self.stats.aborts += 1;
        Ok(())
    }

    fn exec_norm(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // NORM Rd, Rsrc, Rgamma, Rbeta  — RMSNorm
        //  x_norm = x / sqrt(mean(x^2) + eps) * gamma + beta
        //  eps = 1e-5, eixo = último (cols)
        let (src_addr, gamma_addr, beta_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let s = ctx.reg(instr.rsrc1)?;
            let g = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? } else { 0 };
            let b = if instr.rsrc3 != 0xFF { ctx.reg(instr.rsrc3)? } else { 0 };
            (s, g, b)
        };

        let meta_src = self.memory.get_tensor_meta(src_addr).cloned()
            .ok_or_else(|| anyhow!("NORM: src tensor não encontrado 0x{:x}", src_addr))?;
        let shape = meta_src.shape.clone();
        if shape.len() != 2 {
            return Err(anyhow!("NORM: apenas 2D suportado, shape {:?}", shape));
        }
        let rows = shape[0];
        let cols = shape[1];
        let n: usize = rows * cols;
        let src_data = self.memory.read_f32_tensor(src_addr, n)?;

        // Gamma: se endereço não zero e meta existe, lê; senão usa 1.0
        let gamma_data: Vec<f32> = if gamma_addr != 0 {
            if let Some(meta) = self.memory.get_tensor_meta(gamma_addr).cloned() {
                let g_n: usize = meta.shape.iter().product();
                let d = self.memory.read_f32_tensor(gamma_addr, g_n).unwrap_or(vec![1.0; cols]);
                // broadcast: se shape [cols] ou [1, cols] ou [rows, cols], expande
                if d.len() == cols {
                    // expand per row: repeat
                    let mut expanded = Vec::with_capacity(n);
                    for _ in 0..rows { expanded.extend_from_slice(&d); }
                    expanded
                } else if d.len() == n {
                    d
                } else {
                    vec![1.0; n]
                }
            } else {
                vec![1.0; n]
            }
        } else {
            vec![1.0; n]
        };

        let beta_data: Vec<f32> = if beta_addr != 0 {
            if let Some(meta) = self.memory.get_tensor_meta(beta_addr).cloned() {
                let b_n: usize = meta.shape.iter().product();
                let d = self.memory.read_f32_tensor(beta_addr, b_n).unwrap_or(vec![0.0; cols]);
                if d.len() == cols {
                    let mut expanded = Vec::with_capacity(n);
                    for _ in 0..rows { expanded.extend_from_slice(&d); }
                    expanded
                } else if d.len() == n {
                    d
                } else {
                    vec![0.0; n]
                }
            } else {
                vec![0.0; n]
            }
        } else {
            vec![0.0; n]
        };

        let eps = 1e-5_f32;
        let src_arr = Array2::from_shape_vec((rows, cols), src_data).map_err(|e| anyhow!("norm reshape src: {}", e))?;
        let gamma_arr = Array2::from_shape_vec((rows, cols), gamma_data).map_err(|e| anyhow!("norm reshape gamma: {}", e))?;
        let beta_arr = Array2::from_shape_vec((rows, cols), beta_data).map_err(|e| anyhow!("norm reshape beta: {}", e))?;

        // RMS per row
        let mut out = Array2::<f32>::zeros((rows, cols));
        for r in 0..rows {
            let row = src_arr.row(r);
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let mean_sq = sum_sq / cols as f32;
            let rms = (mean_sq + eps).sqrt();
            let inv_rms = 1.0 / rms;
            for c in 0..cols {
                out[[r, c]] = src_arr[[r, c]] * inv_rms * gamma_arr[[r, c]] + beta_arr[[r, c]];
            }
        }

        let out_vec: Vec<f32> = out.iter().cloned().collect();
        let out_addr = self.memory.alloc_tensor(&shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;

        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.norm_execs += 1;
        log_debug("norm", &format!("ctx {} NORM r{} <- rmsnorm(src 0x{:x} gamma 0x{:x} beta 0x{:x}) => 0x{:x} shape {:?}", ctx_id, instr.rdest, src_addr, gamma_addr, beta_addr, out_addr, shape));
        Ok(())
    }

    fn exec_ffn(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // FFN Rd, Rsrc, Rw1, Rw2 (+ payload Rb1, Rb2) — SwiGLU simplificado
        //   hidden = x · W1 + b1
        //   gated  = hidden * sigmoid(hidden)
        //   out    = gated · W2 + b2
        let (src_addr, w1_addr, w2_addr, b1_addr, b2_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let s = ctx.reg(instr.rsrc1)?;
            let w1 = ctx.reg(instr.rsrc2)?;
            let w2 = ctx.reg(instr.rsrc3)?;
            let (rb1, rb2) = instr.ffn_bias_regs();
            let b1 = if rb1 != 0xFF { ctx.reg(rb1).unwrap_or(0) } else { 0 };
            let b2 = if rb2 != 0xFF { ctx.reg(rb2).unwrap_or(0) } else { 0 };
            (s, w1, w2, b1, b2)
        };

        let meta_x = self.memory.get_tensor_meta(src_addr).cloned()
            .ok_or_else(|| anyhow!("FFN: src tensor não encontrado 0x{:x}", src_addr))?;
        let meta_w1 = self.memory.get_tensor_meta(w1_addr).cloned()
            .ok_or_else(|| anyhow!("FFN: W1 tensor não encontrado 0x{:x}", w1_addr))?;
        let meta_w2 = self.memory.get_tensor_meta(w2_addr).cloned()
            .ok_or_else(|| anyhow!("FFN: W2 tensor não encontrado 0x{:x}", w2_addr))?;

        // Shapes: x [M, D], W1 [D, H], W2 [H, D_out]
        if meta_x.shape.len() != 2 || meta_w1.shape.len() != 2 || meta_w2.shape.len() != 2 {
            return Err(anyhow!("FFN: apenas 2D suportado x{:?} w1{:?} w2{:?}", meta_x.shape, meta_w1.shape, meta_w2.shape));
        }
        let m = meta_x.shape[0];
        let d = meta_x.shape[1];
        let d_w1 = meta_w1.shape[0];
        let h = meta_w1.shape[1];
        let h_w2 = meta_w2.shape[0];
        let d_out = meta_w2.shape[1];

        if d != d_w1 {
            return Err(anyhow!("FFN incompatível: x cols {} != W1 rows {}", d, d_w1));
        }
        if h != h_w2 {
            return Err(anyhow!("FFN incompatível: W1 cols {} != W2 rows {}", h, h_w2));
        }

        let n_x = m * d;
        let n_w1 = d * h;
        let n_w2 = h * d_out;

        let x_data = self.memory.read_f32_tensor(src_addr, n_x)?;
        let w1_data = self.memory.read_f32_tensor(w1_addr, n_w1)?;
        let w2_data = self.memory.read_f32_tensor(w2_addr, n_w2)?;

        let x = Array2::from_shape_vec((m, d), x_data).map_err(|e| anyhow!("ffn x reshape: {}", e))?;
        let w1 = Array2::from_shape_vec((d, h), w1_data).map_err(|e| anyhow!("ffn w1 reshape: {}", e))?;
        let w2 = Array2::from_shape_vec((h, d_out), w2_data).map_err(|e| anyhow!("ffn w2 reshape: {}", e))?;

        // hidden = x · W1
        let mut hidden = x.dot(&w1); // [M, H]
        // + b1 broadcast
        if b1_addr != 0 {
            if let Some(meta_b1) = self.memory.get_tensor_meta(b1_addr).cloned() {
                let n_b1: usize = meta_b1.shape.iter().product();
                let b1_data = self.memory.read_f32_tensor(b1_addr, n_b1).unwrap_or(vec![0.0; h]);
                let bias = if b1_data.len() == h {
                    // broadcast per row
                    Array2::from_shape_vec((1, h), b1_data).unwrap().broadcast((m, h)).unwrap().to_owned()
                } else if b1_data.len() == m * h {
                    Array2::from_shape_vec((m, h), b1_data).unwrap()
                } else {
                    Array2::zeros((m, h))
                };
                hidden = hidden + bias;
            }
        }

        // SwiGLU simplified: hidden * sigmoid(hidden)
        let gated = hidden.mapv(|v| {
            let sig = 1.0 / (1.0 + (-v).exp());
            v * sig
        });

        // out = gated · W2
        let mut out = gated.dot(&w2); // [M, D_out]
        // + b2 broadcast
        if b2_addr != 0 {
            if let Some(meta_b2) = self.memory.get_tensor_meta(b2_addr).cloned() {
                let n_b2: usize = meta_b2.shape.iter().product();
                let b2_data = self.memory.read_f32_tensor(b2_addr, n_b2).unwrap_or(vec![0.0; d_out]);
                let bias = if b2_data.len() == d_out {
                    Array2::from_shape_vec((1, d_out), b2_data).unwrap().broadcast((m, d_out)).unwrap().to_owned()
                } else if b2_data.len() == m * d_out {
                    Array2::from_shape_vec((m, d_out), b2_data).unwrap()
                } else {
                    Array2::zeros((m, d_out))
                };
                out = out + bias;
            }
        }

        let out_vec: Vec<f32> = out.iter().cloned().collect();
        let out_shape = vec![m, d_out];
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;

        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.ffn_execs += 1;
        log_debug("ffn", &format!("ctx {} FFN r{} <- ffn(src 0x{:x} w1 0x{:x} w2 0x{:x} b1 0x{:x} b2 0x{:x}) => 0x{:x} shape {:?}", ctx_id, instr.rdest, src_addr, w1_addr, w2_addr, b1_addr, b2_addr, out_addr, out_shape));
        Ok(())
    }

    fn exec_sense(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let peripheral = instr.rsrc1;
        let data: Vec<u8> = match peripheral {
            SENSE_AUDIO | SENSE_AUDIO_PCM => {
                // F2 Mimi: frame determinístico 24kHz/80ms (1920 samples senoide
                // 440Hz) em vez de white-noise rand. Mesmo bytes por chamada,
                // codificável via `crate::mimi::encode_frame` (16cb@12.5Hz).
                // SENSE_AUDIO_PCM (6) é alias explícito p/ CODEC_ENC.
                let frame = crate::mimi::synth_frame_440hz();
                debug_assert_eq!(frame.len(), crate::mimi::MIMI_SAMPLES_PER_FRAME);
                crate::mimi::pcm_to_bytes(&frame)
            }
            SENSE_CODEC_FRAME => {
                // Frame já codificado Mimi: 32B LE (16xu16) do synth 440Hz.
                let frame = crate::mimi::synth_frame_440hz();
                let codes = crate::mimi::encode_frame(&frame);
                crate::mimi::codes_to_bytes(&codes).to_vec()
            }
            SENSE_VAD => {
                // VAD stub: 1 byte 0/1 aleatório + timestamp
                let mut rng = rand::thread_rng();
                let vad: u8 = if rng.gen_bool(0.3) { 1 } else { 0 };
                vec![vad]
            }
            crate::opcodes::SENSE_TOKEN => {
                // PERIPHERAL_TOKEN: retorna last_sample como u32 LE
                self.last_sample.to_le_bytes().to_vec()
            }
            SENSE_USER_INPUT => {
                // Entrada do usuário (não-bloqueante): consome 1 item da fila.
                // Semântica de valor: rdest = 1/0 + interrupt_flag — casa com
                // `IF_INTERRUPT rdest, ALVO` do loop thinking.
                let item = self.user_input.pop_front();
                let has = item.is_some();
                if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                    ctx.set_reg(instr.rdest, if has { 1 } else { 0 })?;
                    ctx.interrupt_flag = has;
                }
                self.stats.senses += 1;
                log_debug("sense", &format!("ctx {} SENSE USER_INPUT -> {} (fila {})", ctx_id, has as u8, self.user_input.len()));
                if let Some(text) = item {
                    let addr = self.memory.temporal_push(text.as_bytes())?;
                    log_debug("sense", &format!("ctx {} USER_INPUT {} bytes em 0x{:032x}", ctx_id, text.len(), addr));
                }
                return Ok(());
            }
            _ => {
                // Lê do stdin se periférico desconhecido (para teste manual)
                b"sense_default".to_vec()
            }
        };

        let addr = self.memory.temporal_push(&data)?;

        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, addr)?;
        }

        self.stats.senses += 1;
        log_debug("sense", &format!("ctx {} SENSE periph {} -> 0x{:032x} {} bytes", ctx_id, peripheral, addr, data.len()));
        Ok(())
    }

    fn exec_matvec(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // MATVEC Rd, Rx, Rw — GEMV: y = x · W
        // x: [M, K] ou [K], W: [K, N] → y: [M, N] ou [N]
        let (x_addr, w_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let meta_x = self.memory.get_tensor_meta(x_addr).cloned()
            .ok_or_else(|| anyhow!("MATVEC: x tensor não encontrado 0x{:x}", x_addr))?;
        let meta_w = self.memory.get_tensor_meta(w_addr).cloned()
            .ok_or_else(|| anyhow!("MATVEC: W tensor não encontrado 0x{:x}", w_addr))?;
        if meta_x.shape.len() != 2 || meta_w.shape.len() != 2 {
            return Err(anyhow!("MATVEC: apenas 2D suportado x{:?} w{:?}", meta_x.shape, meta_w.shape));
        }
        let m = meta_x.shape[0];
        let k = meta_x.shape[1];
        let k_w = meta_w.shape[0];
        let n = meta_w.shape[1];
        if k != k_w {
            return Err(anyhow!("MATVEC incompatível: x cols {} != W rows {}", k, k_w));
        }
        let n_x = m * k;
        let n_w = k * n;
        let x_data = self.memory.read_f32_tensor(x_addr, n_x)?;
        let w_data = self.memory.read_f32_tensor(w_addr, n_w)?;
        // Usa faer/matvec_quant path: para M==1 usa matvec direto, senão batched via ndarray
        let out_vec: Vec<f32> = if m == 1 {
            crate::matvec::matvec(&x_data, &w_data, k, n)
        } else {
            let x_arr = Array2::from_shape_vec((m, k), x_data).map_err(|e| anyhow!("matvec x reshape: {}", e))?;
            let w_arr = Array2::from_shape_vec((k, n), w_data).map_err(|e| anyhow!("matvec w reshape: {}", e))?;
            x_arr.dot(&w_arr).iter().cloned().collect()
        };
        let out_shape = vec![m, n];
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.matvec_execs += 1;
        log_debug("matvec", &format!("ctx {} MATVEC r{} <- matvec(x 0x{:x} {:?} * W 0x{:x} {:?}) => 0x{:x} {:?}", ctx_id, instr.rdest, x_addr, meta_x.shape, w_addr, meta_w.shape, out_addr, out_shape));
        Ok(())
    }

    fn exec_mul(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (a1, a2) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let m1 = self.memory.get_tensor_meta(a1).cloned().ok_or_else(|| anyhow!("MUL: tensor 0x{:x} não encontrado", a1))?;
        let m2 = self.memory.get_tensor_meta(a2).cloned().ok_or_else(|| anyhow!("MUL: tensor 0x{:x} não encontrado", a2))?;
        if m1.shape != m2.shape {
            return Err(anyhow!("MUL: shapes {:?} != {:?}", m1.shape, m2.shape));
        }
        let n: usize = m1.shape.iter().product();
        let d1 = self.memory.read_f32_tensor(a1, n)?;
        let d2 = self.memory.read_f32_tensor(a2, n)?;
        let out_vec: Vec<f32> = d1.iter().zip(d2.iter()).map(|(a, b)| a * b).collect();
        let out_addr = self.memory.alloc_tensor(&m1.shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.add_execs += 1;
        Ok(())
    }

    fn exec_silu(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let src = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let meta = self.memory.get_tensor_meta(src).cloned().ok_or_else(|| anyhow!("SILU: tensor 0x{:x} não encontrado", src))?;
        let n: usize = meta.shape.iter().product();
        let data = self.memory.read_f32_tensor(src, n)?;
        let out_vec: Vec<f32> = data.iter().map(|&v| {
            let sig = 1.0 / (1.0 + (-v).exp());
            v * sig
        }).collect();
        let out_addr = self.memory.alloc_tensor(&meta.shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Novos opcodes 0x13..0x19 (Mamba / Codec / Align / CtxSwitch / RoPE)
    // -----------------------------------------------------------------------

    /// Garante `ssm_states[layer]` compatível com (di, ds); preserva d_conv.
    fn ensure_ssm_state(&mut self, layer: usize, d_inner: usize, d_state: usize) {
        if self.ssm_states.len() <= layer {
            self.ssm_states.resize_with(layer + 1, || crate::ssm::MambaState::new(d_inner, d_state, 4));
        }
        let st = &mut self.ssm_states[layer];
        if !st.is_compatible(d_inner, d_state, st.d_conv) {
            let dc = st.d_conv;
            *st = crate::ssm::MambaState::new(d_inner, d_state, dc);
        }
    }

    /// Lê pack de params do SSM: dt[di]+A[di*ds]+B[ds]+C[ds]+D[di].
    /// Retorna `None` se pack ausente/incompatível (caller usa defaults).
    fn read_ssm_pack(&self, pack_addr: u128, d_inner: usize, d_state: usize) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let meta = self.memory.get_tensor_meta(pack_addr)?;
        let n: usize = meta.shape.iter().product();
        let expect = d_inner + d_inner * d_state + d_state + d_state + d_inner;
        if n != expect {
            return None;
        }
        let v = self.memory.read_f32_tensor(pack_addr, n).ok()?;
        let mut off = 0;
        let dt = v[off..off + d_inner].to_vec(); off += d_inner;
        let a = v[off..off + d_inner * d_state].to_vec(); off += d_inner * d_state;
        let b = v[off..off + d_state].to_vec(); off += d_state;
        let c = v[off..off + d_state].to_vec(); off += d_state;
        let d = v[off..off + d_inner].to_vec();
        Some((dt, a, b, c, d))
    }

    fn default_ssm_pack(&self, d_inner: usize, d_state: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // Defaults determinísticos p/ demo sem pack: dt=1, A=-1, B=1, C=1, D=0.
        // Coincidem com o golden `ssm.rs::test_scan_decay_and_leak`.
        (vec![1.0; d_inner], vec![-1.0; d_inner * d_state], vec![1.0; d_state], vec![1.0; d_state], vec![0.0; d_inner])
    }

    fn exec_ssm_scan(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // Flags CONV/GATE são reservadas: falhar explícito em vez de ignorar
        // silenciosamente (evita programa que parece fazer conv/gate sem fazer).
        if instr.flags & crate::opcodes::SSM_SCAN_FLAG_CONV != 0 {
            return Err(anyhow!("SSM_SCAN CONV ainda não implementado (use scan puro + MATVEC p/ conv depthwise)"));
        }
        if instr.flags & crate::opcodes::SSM_SCAN_FLAG_GATE != 0 {
            return Err(anyhow!("SSM_SCAN GATE ainda não implementado (use SILU+MUL explícitos p/ gating)"));
        }
        let (d_inner, d_state, layer_id) = instr.ssm_dims();
        if d_inner == 0 || d_state == 0 || d_inner > 16384 || d_state > 1024 {
            return Err(anyhow!("SSM_SCAN dims inválidas di={} ds={}", d_inner, d_state));
        }
        let (x_addr, h_addr, p_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2).unwrap_or(0), ctx.reg(instr.rsrc3).unwrap_or(0))
        };
        // x: tensor [1,di] (ou produto==di); trunca/pad para di.
        let x_meta = self.memory.get_tensor_meta(x_addr).cloned()
            .ok_or_else(|| anyhow!("SSM_SCAN: x tensor 0x{:x} não encontrado", x_addr))?;
        let nx: usize = x_meta.shape.iter().product();
        let mut x = self.memory.read_f32_tensor(x_addr, nx)?;
        x.resize(d_inner, 0.0);
        x.truncate(d_inner);
        // params: pack tensor ou defaults.
        let (dt, a, b, c, d) = if p_addr != 0 && self.memory.get_tensor_meta(p_addr).is_some() {
            self.read_ssm_pack(p_addr, d_inner, d_state).unwrap_or_else(|| self.default_ssm_pack(d_inner, d_state))
        } else {
            self.default_ssm_pack(d_inner, d_state)
        };
        let y = if instr.rsrc2 == 0xFF {
            // Path híbrido: estado na Vm (fase 2), indexado por layer_id.
            let layer = layer_id as usize;
            self.ensure_ssm_state(layer, d_inner, d_state);
            let st = &mut self.ssm_states[layer];
            debug_assert_eq!(st.ssm.len(), d_inner * d_state);
            crate::ssm::selective_scan_update(&mut st.ssm, &x, &dt, &a, &b, &c, &d, d_inner, d_state)
        } else {
            // Path tensor puro (MVP): h é tensor [di,ds] atualizado in-place.
            let h_meta = self.memory.get_tensor_meta(h_addr).cloned()
                .ok_or_else(|| anyhow!("SSM_SCAN: h tensor 0x{:x} não encontrado", h_addr))?;
            let nh: usize = h_meta.shape.iter().product();
            if nh != d_inner * d_state {
                return Err(anyhow!("SSM_SCAN: h shape {:?} != [{} x {}]", h_meta.shape, d_inner, d_state));
            }
            let mut h = self.memory.read_f32_tensor(h_addr, nh)?;
            let y = crate::ssm::selective_scan_update(&mut h, &x, &dt, &a, &b, &c, &d, d_inner, d_state);
            self.memory.write_f32_tensor(h_addr, &h)?;
            y
        };
        let out_addr = self.memory.alloc_tensor(&[1, d_inner], crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &y)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.ssm_scans += 1;
        log_debug("ssm", &format!("ctx {} SSM_SCAN di={} ds={} layer={} -> 0x{:x}", ctx_id, d_inner, d_state, layer_id, out_addr));
        Ok(())
    }

    fn exec_ssm_reset(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (d_inner, d_state, layer_id) = instr.ssm_dims();
        if instr.rsrc1 == 0xFF {
            let layer = layer_id as usize;
            self.ensure_ssm_state(layer, d_inner.max(1), d_state.max(1));
            self.ssm_states[layer].reset();
        } else {
            let h_addr = {
                let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
                ctx.reg(instr.rsrc1)?
            };
            if let Some(meta) = self.memory.get_tensor_meta(h_addr).cloned() {
                let n: usize = meta.shape.iter().product();
                self.memory.write_f32_tensor(h_addr, &vec![0.0; n])?;
            } else {
                let layer = layer_id as usize;
                self.ensure_ssm_state(layer, d_inner.max(1), d_state.max(1));
                self.ssm_states[layer].reset();
            }
        }
        self.stats.ssm_resets += 1;
        log_debug("ssm", &format!("ctx {} SSM_RESET layer={}", ctx_id, layer_id));
        Ok(())
    }

    /// Lê PCM 1920xf32 de um addr (tensor f32 ou bytes TEMPORAL LE).
    fn read_pcm_frame(&self, addr: u128) -> Result<Vec<f32>> {
        use crate::mimi::MIMI_SAMPLES_PER_FRAME;
        if let Some(meta) = self.memory.get_tensor_meta(addr).cloned() {
            let n: usize = meta.shape.iter().product();
            let mut v = self.memory.read_f32_tensor(addr, n)?;
            v.resize(MIMI_SAMPLES_PER_FRAME, 0.0);
            v.truncate(MIMI_SAMPLES_PER_FRAME);
            return Ok(v);
        }
        // Fallback bytes (SENSE AUDIO_PCM em TEMPORAL).
        let raw = self.memory.read(addr, MIMI_SAMPLES_PER_FRAME * 4)?;
        let mut v = crate::mimi::pcm_from_bytes(&raw);
        v.resize(MIMI_SAMPLES_PER_FRAME, 0.0);
        v.truncate(MIMI_SAMPLES_PER_FRAME);
        Ok(v)
    }

    /// Lê 16 códigos de um addr (tensor [1,16] f32 ou 32B LE).
    fn read_codes(&self, addr: u128) -> Result<crate::mimi::MimiCodes> {
        use crate::mimi::{MIMI_CODEBOOK_SIZE, MIMI_N_CODEBOOKS};
        if let Some(meta) = self.memory.get_tensor_meta(addr).cloned() {
            let n: usize = meta.shape.iter().product();
            let v = self.memory.read_f32_tensor(addr, n)?;
            let mut codes = [0u16; MIMI_N_CODEBOOKS];
            for (i, c) in codes.iter_mut().enumerate() {
                let f = v.get(i).copied().unwrap_or(0.0).round() as i32;
                *c = f.clamp(0, (MIMI_CODEBOOK_SIZE - 1) as i32) as u16;
            }
            return Ok(codes);
        }
        let raw = self.memory.read(addr, MIMI_N_CODEBOOKS * 2)?;
        crate::mimi::codes_from_bytes(&raw).ok_or_else(|| anyhow!("CODEC: codes inválidos em 0x{:x}", addr))
    }

    fn exec_codec_enc(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::CODEC_FLAG_AS_TENSOR;
        let pcm_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let pcm = self.read_pcm_frame(pcm_addr)?;
        let codes = crate::mimi::encode_frame(&pcm);
        let as_tensor = instr.flags & CODEC_FLAG_AS_TENSOR != 0;
        let out_addr = if as_tensor {
            let a = self.memory.alloc_tensor(&[1, codes.len()], crate::memory::DType::F32)?;
            let f: Vec<f32> = codes.iter().map(|&c| c as f32).collect();
            self.memory.write_f32_tensor(a, &f)?;
            a
        } else {
            let b = crate::mimi::codes_to_bytes(&codes);
            self.memory.temporal_push(&b)?
        };
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.codec_encs += 1;
        log_debug("codec", &format!("ctx {} CODEC_ENC 0x{:x} -> 0x{:x}", ctx_id, pcm_addr, out_addr));
        Ok(())
    }

    fn exec_codec_dec(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::CODEC_FLAG_AS_TENSOR;
        let codes_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let codes = self.read_codes(codes_addr)?;
        let frame = crate::mimi::decode_frame(&codes);
        let as_tensor = instr.flags & CODEC_FLAG_AS_TENSOR != 0;
        let out_addr = if as_tensor {
            let a = self.memory.alloc_tensor(&[1, frame.len()], crate::memory::DType::F32)?;
            self.memory.write_f32_tensor(a, &frame)?;
            a
        } else {
            let b = crate::mimi::pcm_to_bytes(&frame);
            self.memory.temporal_push(&b)?
        };
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.codec_decs += 1;
        log_debug("codec", &format!("ctx {} CODEC_DEC 0x{:x} -> 0x{:x}", ctx_id, codes_addr, out_addr));
        Ok(())
    }

    /// Lê timestamp u64 de um registrador: tensor `[n]` (primeiro `f32`),
    /// bytes `TEMPORAL` (8B LE), ou o valor imediato do registrador.
    /// Bytes só são lidos quando a região é `TEMPORAL` — nunca de `GLOBAL`,
    /// para não interpretar peso/tensor como timestamp.
    fn read_ts(&self, addr_or_imm: u128) -> u64 {
        if let Some(meta) = self.memory.get_tensor_meta(addr_or_imm).cloned() {
            let n: usize = meta.shape.iter().product();
            if let Ok(v) = self.memory.read_f32_tensor(addr_or_imm, n) {
                if let Some(&f) = v.first() {
                    return f as u64;
                }
            }
            return 0;
        }
        if crate::memory::region_of(addr_or_imm) == crate::memory::Region::Temporal {
            if let Ok(raw) = self.memory.read(addr_or_imm, 8) {
                if raw.len() >= 8 {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(&raw[..8]);
                    return u64::from_le_bytes(b);
                }
            }
            return 0;
        }
        // Imediato: valor do registrador é o próprio timestamp.
        addr_or_imm as u64
    }

    fn exec_audio_align(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (u_addr, ai_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let t_user = self.read_ts(u_addr);
        let t_ai = self.read_ts(ai_addr);
        let delta = t_ai.wrapping_sub(t_user);
        // frame_id = delta_ms / 80ms (Mimi frame). Usa u64 ns.
        let delta_ms = (t_ai.saturating_sub(t_user)) as f64 / 1_000_000.0;
        let frame_id = (delta_ms / 80.0).floor() as u64;
        let out = vec![t_user as f32, t_ai as f32, delta as f32, frame_id as f32];
        let out_addr = self.memory.alloc_tensor(&[1, 4], crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.audio_aligns += 1;
        log_debug("audio", &format!("ctx {} AUDIO_ALIGN tu={} tai={} d={} f={}", ctx_id, t_user, t_ai, delta, frame_id));
        Ok(())
    }

    fn exec_ctx_switch(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        // Pipe canônico via payload+MAGIC (ver instr_ctx_switch); rsrc1 é legado.
        let pipe = crate::opcodes::ctx_switch_pipe(instr);
        if pipe > 2 {
            return Err(anyhow!("CTX_SWITCH pipe inválido {} (use MAMBA/TRANSFORMER/AUDIO)", pipe));
        }
        let prio = Priority::from_flags(instr.flags);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.pipeline = pipe;
            // Só re-enfileira se prioridade mudou (evita duplicar fila no path comum).
            if ctx.priority != prio {
                ctx.priority = prio;
            }
        }
        // Fence: checkpoint de memória p/ rollback + reavalia preempção.
        let v = self.memory.snapshot();
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.root_version = v;
        }
        self.scheduler.maybe_preempt();
        self.stats.ctx_switches += 1;
        log_debug("ctx", &format!("ctx {} CTX_SWITCH pipe={} prio={} fence=v{}", ctx_id, pipe, prio, v));
        Ok(())
    }

    fn exec_rope(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (pos, head_dim, n_heads, theta) = instr.rope_params();
        if head_dim % 2 != 0 || head_dim == 0 {
            return Err(anyhow!("ROPE head_dim inválido {}", head_dim));
        }
        let src_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let meta = self.memory.get_tensor_meta(src_addr).cloned()
            .ok_or_else(|| anyhow!("ROPE: tensor 0x{:x} não encontrado", src_addr))?;
        let n: usize = meta.shape.iter().product();
        if n % (n_heads * head_dim) != 0 && n != n_heads * head_dim {
            return Err(anyhow!("ROPE shape {:?} incompatível com n_heads={} head_dim={}", meta.shape, n_heads, head_dim));
        }
        let mut v = self.memory.read_f32_tensor(src_addr, n)?;
        // freqs por posição (mesma fórmula de moshi.rs/inference.rs).
        let half = head_dim / 2;
        let freqs: Vec<(f32, f32)> = (0..half.max(1))
            .map(|i| {
                let f = pos as f32 * theta.powf(-2.0 * i as f32 / head_dim.max(1) as f32);
                (f.cos(), f.sin())
            })
            .collect();
        // Aplica por bloco [n_heads*head_dim] (suporta batch M>1).
        let block = n_heads * head_dim;
        for chunk in v.chunks_mut(block) {
            crate::inference::apply_rope(chunk, n_heads, head_dim, &freqs);
        }
        let out_addr = self.memory.alloc_tensor(&meta.shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &v)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.ropes += 1;
        log_debug("rope", &format!("ctx {} ROPE pos={} hd={} nh={} -> 0x{:x}", ctx_id, pos, head_dim, n_heads, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0004: GATHER (0x1F) / DISTANCE (0x23) / RANK1_UPDATE (0x24)
    // -----------------------------------------------------------------------

    /// GATHER rD, rTable, rIdx [, rAcc] — indexação indireta N-D row-major.
    /// Índices em f32 (interop com DISTANCE/SAMPLE-TOPK): devem ser integrais
    /// e não-negativos; o limite é dim(axis) da tabela (GATHER) ou do
    /// acumulador (scatter) — fora disso, erro determinístico (nunca wrap).
    /// Denso apenas; tabela esparsa retorna erro explícito.
    fn exec_gather(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{
            GATHER_MODE_GATHER, GATHER_MODE_SCATTER_ADD, GATHER_MODE_SCATTER_MAX,
        };
        let (t_addr, i_addr, acc_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?, ctx.reg(instr.rsrc3).unwrap_or(0xFF))
        };
        let (axis, mode, _hint) = instr.gather_params();
        let tmeta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("GATHER: tabela 0x{:x} não encontrada", t_addr))?;
        if tmeta.is_sparse {
            return Err(anyhow!("GATHER: tabela esparsa não suportada (RFC-0004: denso)"));
        }
        let rank = tmeta.shape.len();
        if rank == 0 || (axis as usize) >= rank {
            return Err(anyhow!("GATHER: axis {} inválido para rank {}", axis, rank));
        }
        let ax = axis as usize;
        let imeta = self.memory.get_tensor_meta(i_addr).cloned()
            .ok_or_else(|| anyhow!("GATHER: índices 0x{:x} não encontrados", i_addr))?;
        if imeta.is_sparse {
            return Err(anyhow!("GATHER: índices esparsos não suportados"));
        }
        let n_idx: usize = imeta.shape.iter().product();
        let idata = self.memory.read_f32_tensor(i_addr, n_idx)?;
        let dim: usize = tmeta.shape[ax];
        let mut idx = Vec::with_capacity(n_idx);
        for (j, &f) in idata.iter().enumerate() {
            // Integralidade e não-negatividade sempre; o limite superior
            // depende do modo (tabela no GATHER, acumulador no scatter).
            if !f.is_finite() || f.fract() != 0.0 || f < 0.0 {
                return Err(anyhow!("GATHER: índice inválido na posição {} (valor {})", j, f));
            }
            if mode == GATHER_MODE_GATHER && f as usize >= dim {
                return Err(anyhow!("GATHER: índice OOB na posição {} (valor {}, dim {})", j, f, dim));
            }
            idx.push(f as usize);
        }
        let outer: usize = tmeta.shape[..ax].iter().product();
        let inner: usize = tmeta.shape[ax + 1..].iter().product();
        // Eixo da tabela/valores em rsrc1: `dim` (tabela cheia, modo GATHER)
        // ou `n_idx` (valores a espalhar, modo scatter). Leitura usa o real.
        let t_ax = tmeta.shape[ax];
        let tflat = self.memory.read_f32_tensor(t_addr, outer * t_ax * inner)?;
        let mut out_shape = tmeta.shape.clone();
        out_shape[ax] = n_idx;
        let out: Vec<f32> = match mode {
            GATHER_MODE_GATHER => {
                let mut v = vec![0.0f32; outer * n_idx * inner];
                for o in 0..outer {
                    for (j, &r) in idx.iter().enumerate() {
                        for k in 0..inner {
                            v[(o * n_idx + j) * inner + k] = tflat[(o * dim + r) * inner + k];
                        }
                    }
                }
                v
            }
            GATHER_MODE_SCATTER_ADD | GATHER_MODE_SCATTER_MAX => {
                if instr.rsrc3 == 0xFF {
                    return Err(anyhow!("GATHER: modo scatter exige acumulador em rsrc3"));
                }
                let ameta = self.memory.get_tensor_meta(acc_addr).cloned()
                    .ok_or_else(|| anyhow!("GATHER: acumulador 0x{:x} não encontrado", acc_addr))?;
                if ameta.is_sparse {
                    return Err(anyhow!("GATHER: acumulador esparso não suportado"));
                }
                // Valores (rsrc1): eixo = n_idx; demais eixos = shape do acumulador.
                if t_ax != n_idx {
                    return Err(anyhow!("GATHER: scatter espera valores com eixo {} = n_idx {}, achado {}", ax, n_idx, t_ax));
                }
                if ameta.shape.len() != rank {
                    return Err(anyhow!("GATHER: acumulador rank {} != rank {} da tabela", ameta.shape.len(), rank));
                }
                for (a, (&vs, &as_)) in tmeta.shape.iter().zip(ameta.shape.iter()).enumerate() {
                    if a != ax && vs != as_ {
                        return Err(anyhow!("GATHER: eixo {}: valores {} != acumulador {}", a, vs, as_));
                    }
                }
                let dst_dim: usize = ameta.shape[ax];
                let mut v = self.memory.read_f32_tensor(acc_addr, outer * dst_dim * inner)?;
                let vals = tflat; // rsrc1 = valores no modo scatter
                for o in 0..outer {
                    for (j, &r) in idx.iter().enumerate() {
                        if r >= dst_dim {
                            return Err(anyhow!("GATHER: índice scatter OOB na posição {} (valor {}, dim {})", j, r, dst_dim));
                        }
                        for k in 0..inner {
                            let dst = (o * dst_dim + r) * inner + k;
                            let src = (o * n_idx + j) * inner + k;
                            if mode == GATHER_MODE_SCATTER_ADD {
                                v[dst] += vals[src];
                            } else {
                                v[dst] = v[dst].max(vals[src]);
                            }
                        }
                    }
                }
                out_shape = ameta.shape.clone(); // scatter devolve o shape cheio
                v
            }
            _ => return Err(anyhow!("GATHER: MODE {} inválido (0/1/2)", mode)),
        };
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.gather_execs += 1;
        log_debug("gather", &format!("ctx {} GATHER axis={} mode={} n={} -> 0x{:x}", ctx_id, axis, mode, n_idx, out_addr));
        Ok(())
    }

    /// DISTANCE rD, rQuery[1,D], rBank[N,D] — 4 métricas + top-k fundido.
    /// Semântica de "distância" (menor = mais próximo): EUCLID=L2,
    /// COSINE=1-cos (norma zero => 1.0), MANHATTAN=L1, DOT=-dot.
    /// TOPK=0: saída [1,N] em ordem; TOPK=T: saída [1,2T] (dists, índices
    /// como f32, exatos para N<2^24), desempate por menor índice.
    fn exec_distance(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{
            DIST_METRIC_COSINE, DIST_METRIC_DOT, DIST_METRIC_EUCLID, DIST_METRIC_MANHATTAN,
        };
        let (q_addr, b_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let (metric, topk) = instr.distance_params();
        if !matches!(metric, DIST_METRIC_EUCLID | DIST_METRIC_COSINE | DIST_METRIC_MANHATTAN | DIST_METRIC_DOT) {
            return Err(anyhow!("DISTANCE: METRIC {} inválida (0-3)", metric));
        }
        let qmeta = self.memory.get_tensor_meta(q_addr).cloned()
            .ok_or_else(|| anyhow!("DISTANCE: query 0x{:x} não encontrada", q_addr))?;
        let bmeta = self.memory.get_tensor_meta(b_addr).cloned()
            .ok_or_else(|| anyhow!("DISTANCE: banco 0x{:x} não encontrado", b_addr))?;
        if qmeta.is_sparse || bmeta.is_sparse {
            return Err(anyhow!("DISTANCE: esparso não suportado (RFC-0004: denso)"));
        }
        if qmeta.shape.len() != 2 || qmeta.shape[0] != 1 {
            return Err(anyhow!("DISTANCE: query deve ser [1,D], achado {:?}", qmeta.shape));
        }
        if bmeta.shape.len() != 2 || bmeta.shape[1] != qmeta.shape[1] {
            return Err(anyhow!("DISTANCE: banco deve ser [N,D] com D={}, achado {:?}", qmeta.shape[1], bmeta.shape));
        }
        let d = qmeta.shape[1];
        let n = bmeta.shape[0];
        if n == 0 || d == 0 {
            return Err(anyhow!("DISTANCE: query/banco vazios (N={}, D={})", n, d));
        }
        let q = self.memory.read_f32_tensor(q_addr, d)?;
        let bank = self.memory.read_f32_tensor(b_addr, n * d)?;
        let qnorm2: f32 = q.iter().map(|x| x * x).sum();
        let mut dists = Vec::with_capacity(n);
        for i in 0..n {
            let row = &bank[i * d..(i + 1) * d];
            let dist = match metric {
                DIST_METRIC_EUCLID => row.iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt(),
                DIST_METRIC_MANHATTAN => row.iter().zip(q.iter()).map(|(a, b)| (a - b).abs()).sum(),
                DIST_METRIC_DOT => -row.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>(),
                _ => {
                    let rn: f32 = row.iter().map(|x| x * x).sum();
                    if qnorm2 == 0.0 || rn == 0.0 {
                        1.0
                    } else {
                        let dot: f32 = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                        1.0 - dot / (qnorm2.sqrt() * rn.sqrt())
                    }
                }
            };
            dists.push(if dist.is_finite() { dist } else { f32::MAX });
        }
        let (out_shape, out): (Vec<usize>, Vec<f32>) = if topk == 0 {
            (vec![1, n], dists)
        } else {
            let t = (topk as usize).min(n);
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                dists[a].partial_cmp(&dists[b]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
            });
            order.truncate(t);
            let mut v = Vec::with_capacity(2 * t);
            v.extend(order.iter().map(|&i| dists[i]));
            v.extend(order.iter().map(|&i| i as f32));
            (vec![1, 2 * t], v)
        };
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out_addr)?;
        }
        self.stats.distance_execs += 1;
        log_debug("distance", &format!("ctx {} DISTANCE metric={} topk={} N={} D={} -> 0x{:x}", ctx_id, metric, topk, n, d, out_addr));
        Ok(())
    }

    /// RANK1_UPDATE rH[d,k], rV[d], rK[k] — H' CoW (logicamente in-place).
    /// hebbian: H'=αH+β·v⊗k · delta: H'=αH+β·((v−Hk)⊗k)/(1+β‖k‖²) ·
    /// forget: H'=α·(m⊙H)+β·v⊗k (m=rsrc3 ou 1; α validado em (0,1]).
    /// Novo H alocado por passo; handle da camada religado (I-Persist-safe).
    fn exec_rank1_update(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{RANK1_MODE_DELTA, RANK1_MODE_FORGET, RANK1_MODE_HEBBIAN};
        let h_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rdest)?
        };
        let (v_addr, k_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let (alpha, beta, mode, layer) = instr.rank1_params();
        if !matches!(mode, RANK1_MODE_HEBBIAN | RANK1_MODE_DELTA | RANK1_MODE_FORGET) {
            return Err(anyhow!("RANK1_UPDATE: MODE {} inválido (0/1/2)", mode));
        }
        if !(alpha.is_finite() && beta.is_finite()) {
            return Err(anyhow!("RANK1_UPDATE: ALPHA/BETA não-finitos"));
        }
        let hmeta = self.memory.get_tensor_meta(h_addr).cloned()
            .ok_or_else(|| anyhow!("RANK1_UPDATE: H 0x{:x} não encontrado", h_addr))?;
        if hmeta.is_sparse || hmeta.shape.len() != 2 {
            return Err(anyhow!("RANK1_UPDATE: H deve ser denso [d,k], achado {:?}", hmeta.shape));
        }
        let (dd, kk) = (hmeta.shape[0], hmeta.shape[1]);
        let read_vec = |addr: u128, expect: usize, what: &str| -> Result<Vec<f32>> {
            let m = self.memory.get_tensor_meta(addr).cloned()
                .ok_or_else(|| anyhow!("RANK1_UPDATE: {} 0x{:x} não encontrado", what, addr))?;
            if m.is_sparse {
                return Err(anyhow!("RANK1_UPDATE: {} esparso não suportado", what));
            }
            let flat: usize = m.shape.iter().product();
            if flat != expect {
                return Err(anyhow!("RANK1_UPDATE: {} com {} elems, esperado {}", what, flat, expect));
            }
            self.memory.read_f32_tensor(addr, expect)
        };
        let v = read_vec(v_addr, dd, "V")?;
        let k = read_vec(k_addr, kk, "K")?;
        let h = self.memory.read_f32_tensor(h_addr, dd * kk)?;
        let mask: Option<Vec<f32>> = if instr.rsrc3 != 0xFF {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let m_addr = ctx.reg(instr.rsrc3)?;
            Some(read_vec(m_addr, dd * kk, "máscara")?)
        } else {
            None
        };
        if mode == RANK1_MODE_FORGET {
            if !(alpha > 0.0 && alpha <= 1.0) {
                return Err(anyhow!("RANK1_UPDATE: FORGET exige ALPHA em (0,1], achado {}", alpha));
            }
            if mask.is_none() && (alpha - 1.0).abs() > f32::EPSILON {
                // Sem máscara, FORGET escalar com α<1 ainda é definido; segue.
            }
        }
        // Hk e ‖k‖² (compartilhados pelo modo delta).
        let mut hk = vec![0.0f32; dd];
        let mut knorm2 = 0.0f32;
        for x in k.iter() {
            knorm2 += x * x;
        }
        for i in 0..dd {
            let mut s = 0.0f32;
            for j in 0..kk {
                s += h[i * kk + j] * k[j];
            }
            hk[i] = s;
        }
        let denom = 1.0 + beta * knorm2;
        let mut out = vec![0.0f32; dd * kk];
        for i in 0..dd {
            for j in 0..kk {
                let gate = mask.as_ref().map(|m| m[i * kk + j]).unwrap_or(1.0);
                let upd = match mode {
                    RANK1_MODE_DELTA => beta * (v[i] - hk[i]) * k[j] / denom,
                    _ => beta * v[i] * k[j],
                };
                out[i * kk + j] = alpha * gate * h[i * kk + j] + upd;
            }
        }
        if !out.iter().all(|x| x.is_finite()) {
            return Err(anyhow!("RANK1_UPDATE: resultado não-finito (ALPHA/BETA/entradas?)"));
        }
        let new_addr = self.memory.alloc_tensor(&[dd, kk], crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(new_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, new_addr)?;
        }
        self.rank1_layers.insert(layer, new_addr);
        self.stats.rank1_execs += 1;
        log_debug("rank1", &format!("ctx {} RANK1_UPDATE layer={} mode={} [{}x{}] 0x{:x} -> 0x{:x}", ctx_id, layer, mode, dd, kk, h_addr, new_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0005: determinismo (0x60–0x66). Escalares via regs (u128, precedente
    // SAMPLE); tensores como stream canônico f32-LE.
    // -----------------------------------------------------------------------

    fn exec_rng_seed(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let seed = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            if instr.rsrc1 != 0xFF {
                ctx.reg(instr.rsrc1)? as u64
            } else {
                crate::determinism::DEFAULT_RNG_SEED
            }
        };
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.rng_state = seed;
        }
        self.stats.rng_seed_execs += 1;
        log_debug("rng", &format!("ctx {} RNG_SEED 0x{:016x}", ctx_id, seed));
        Ok(())
    }

    fn exec_rng_next(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let mut st = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.rng_state;
        let v = crate::determinism::splitmix64(&mut st);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.rng_state = st;
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, v as u128)?;
            }
        }
        self.stats.rng_next_execs += 1;
        log_debug("rng", &format!("ctx {} RNG_NEXT 0x{:016x}", ctx_id, v));
        Ok(())
    }

    fn exec_rng_uniform(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (a, b) = instr.uniform_range();
        if !(a.is_finite() && b.is_finite() && a < b) {
            return Err(anyhow!("RNG_UNIFORM: intervalo inválido [{}, {})", a, b));
        }
        let mut st = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.rng_state;
        let x = crate::determinism::uniform_f32(&mut st, a, b);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.rng_state = st;
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, (x.to_bits() as u64) as u128)?;
            }
        }
        self.stats.rng_uniform_execs += 1;
        log_debug("rng", &format!("ctx {} RNG_UNIFORM [{},{}) -> {}", ctx_id, a, b, x));
        Ok(())
    }

    fn exec_rng_normal(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (mean, std) = instr.normal_params();
        if !(mean.is_finite() && std.is_finite() && std > 0.0) {
            return Err(anyhow!("RNG_NORMAL: parâmetros inválidos (mean={}, std={})", mean, std));
        }
        let mut st = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.rng_state;
        let x = crate::determinism::normal_f32(&mut st, mean, std);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.rng_state = st;
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, (x.to_bits() as u64) as u128)?;
            }
        }
        self.stats.rng_normal_execs += 1;
        log_debug("rng", &format!("ctx {} RNG_NORMAL({}, {}) -> {}", ctx_id, mean, std, x));
        Ok(())
    }

    /// Stream canônico de bytes de um tensor denso: f32-LE concatenados.
    /// Esparso e vazio são erro explícito (hash de identidade silenciosa
    /// seria mentira de integridade).
    fn tensor_canonical_bytes(&self, addr: u128, what: &str) -> Result<Vec<u8>> {
        let m = self.memory.get_tensor_meta(addr).cloned()
            .ok_or_else(|| anyhow!("{}: tensor 0x{:x} não encontrado", what, addr))?;
        if m.is_sparse {
            return Err(anyhow!("{}: tensor esparso não suportado", what));
        }
        let n: usize = m.shape.iter().product();
        if n == 0 {
            return Err(anyhow!("{}: tensor vazio não tem hash", what));
        }
        let v = self.memory.read_f32_tensor(addr, n)?;
        let mut out = Vec::with_capacity(n * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        Ok(out)
    }

    fn exec_hash(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let bytes = self.tensor_canonical_bytes(t_addr, "HASH")?;
        let h = crate::determinism::fnv1a64(&bytes);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, h as u128)?;
            }
        }
        self.stats.hash_execs += 1;
        log_debug("hash", &format!("ctx {} HASH 0x{:x} ({}B) -> 0x{:016x}", ctx_id, t_addr, bytes.len(), h));
        Ok(())
    }

    fn exec_checksum(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let bytes = self.tensor_canonical_bytes(t_addr, "CHECKSUM")?;
        let c = crate::determinism::crc32_ieee(&bytes);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, c as u128)?;
            }
        }
        self.stats.checksum_execs += 1;
        log_debug("hash", &format!("ctx {} CHECKSUM 0x{:x} -> 0x{:08x}", ctx_id, t_addr, c));
        Ok(())
    }

    fn exec_hmac(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (k_addr, m_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let key = self.tensor_canonical_bytes(k_addr, "HMAC(key)")?;
        let msg = self.tensor_canonical_bytes(m_addr, "HMAC(msg)")?;
        let t = crate::determinism::hmac_sha256_trunc64(&key, &msg);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, t as u128)?;
            }
        }
        self.stats.hmac_execs += 1;
        log_debug("hash", &format!("ctx {} HMAC -> 0x{:016x}", ctx_id, t));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0006: telemetria (0x6A–0x6F) + scheduler (0x70–0x77).
    // -----------------------------------------------------------------------

    fn exec_cycles_count(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let t = crate::utils::now_ns();
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, t as u128)?;
            }
        } else {
            return Err(anyhow!("ctx {} não encontrado", ctx_id));
        }
        self.stats.cycles_execs += 1;
        log_debug("tele", &format!("ctx {} CYCLES_COUNT {}", ctx_id, t));
        Ok(())
    }

    fn exec_trace_event(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (ev, data) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2).unwrap_or(0))
        };
        if self.trace.len() >= TRACE_CAP {
            self.trace.pop_front();
        }
        self.trace.push_back((ev as u64, data));
        self.stats.trace_execs += 1;
        log_debug("tele", &format!("ctx {} TRACE_EVENT ev={} data=0x{:x}", ctx_id, ev, data));
        Ok(())
    }

    /// SANITY_CHECK rD, rT [, rCount]: copia CoW com não-finitos zerados;
    /// rD = novo addr; rCount (!=0xFF) = nº de absorvidos.
    fn exec_sanity_check(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("SANITY_CHECK: tensor 0x{:x} não encontrado", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("SANITY_CHECK: esparso não suportado"));
        }
        let n: usize = meta.shape.iter().product();
        let data = self.memory.read_f32_tensor(t_addr, n)?;
        let mut out = Vec::with_capacity(n);
        let mut absorbed = 0u64;
        for x in data {
            if x.is_finite() {
                out.push(x);
            } else {
                out.push(0.0);
                absorbed += 1;
            }
        }
        let out_addr = self.memory.alloc_tensor(&meta.shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
            if instr.rsrc3 != 0xFF {
                ctx.set_reg(instr.rsrc3, absorbed as u128)?;
            }
        }
        self.stats.sanity_execs += 1;
        log_debug("tele", &format!("ctx {} SANITY_CHECK 0x{:x} absorvidos={} -> 0x{:x}", ctx_id, t_addr, absorbed, out_addr));
        Ok(())
    }

    /// PREEMPT_CHECK rD: rdest <- interrupt_flag SEM consumir (poll puro;
    /// IF_INTERRUPT consome — ver tese §3.15).
    fn exec_preempt_check(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let flag = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.interrupt_flag;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, u128::from(flag))?;
            }
        }
        self.stats.preempt_check_execs += 1;
        log_debug("tele", &format!("ctx {} PREEMPT_CHECK {}", ctx_id, flag));
        Ok(())
    }

    /// ASSERT Rs [CODE]: trap limpo (Err, sem commit parcial) se reg == 0.
    fn exec_assert(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let v = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        if v == 0 {
            return Err(anyhow!("ASSERT falhou (code {})", instr.assert_code()));
        }
        self.stats.assert_execs += 1;
        log_debug("tele", &format!("ctx {} ASSERT ok", ctx_id));
        Ok(())
    }

    /// DUMP: log de debug do contexto corrente (só o próprio ctx).
    fn exec_dump(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let _ = instr;
        if let Some(ctx) = self.scheduler.get(ctx_id) {
            log_info("dump", &format!(
                "ctx {} pc=0x{:x} root_v={} prio={} pipe={} deadline={} cmp={} irq={} regs={:x?}",
                ctx.id, ctx.pc, ctx.root_version, ctx.priority.as_str(),
                ctx.pipeline, ctx.deadline, ctx.cmp_equal, ctx.interrupt_flag, ctx.regs,
            ));
        } else {
            return Err(anyhow!("ctx {} não encontrado", ctx_id));
        }
        self.stats.dump_execs += 1;
        Ok(())
    }

    /// YIELD: cede + maybe_preempt (um RED em espera preempta não-RED).
    fn exec_yield(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let _ = instr;
        if self.scheduler.get(ctx_id).is_none() {
            return Err(anyhow!("ctx {} não encontrado", ctx_id));
        }
        self.scheduler.maybe_preempt();
        self.stats.yield_execs += 1;
        log_debug("sched", &format!("ctx {} YIELD", ctx_id));
        Ok(())
    }

    fn exec_set_deadline(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let dl = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)? as u64
        };
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.deadline = dl;
        }
        self.stats.set_deadline_execs += 1;
        log_debug("sched", &format!("ctx {} SET_DEADLINE {}", ctx_id, dl));
        Ok(())
    }

    fn exec_get_deadline(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let dl = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.deadline;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, dl as u128)?;
            }
        }
        self.stats.get_deadline_execs += 1;
        log_debug("sched", &format!("ctx {} GET_DEADLINE {}", ctx_id, dl));
        Ok(())
    }

    /// PRIORITY_SET: 0/1/2 válido, senão Err. Move de fila de verdade
    /// (dequeue + set + enqueue + preempt) — mais estrito que CTX_SWITCH.
    fn exec_priority_set(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::context::Priority;
        let raw = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let prio = match raw {
            0 => Priority::Green,
            1 => Priority::Blue,
            2 => Priority::Red,
            _ => return Err(anyhow!("PRIORITY_SET: valor {} inválido (0/1/2)", raw)),
        };
        // Troca simples de campo (como CTX_SWITCH): o loop de run
        // re-enfileira uma vez via yield_current. (Enfileirar aqui
        // duplicaria presença — corrigido na RFC-0018.)
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.priority = prio;
        }
        self.scheduler.maybe_preempt();
        self.stats.priority_set_execs += 1;
        log_debug("sched", &format!("ctx {} PRIORITY_SET {}", ctx_id, prio.as_str()));
        Ok(())
    }

    fn exec_priority_get(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let p = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.priority;
        let v = match p {
            crate::context::Priority::Green => 0u128,
            crate::context::Priority::Blue => 1u128,
            crate::context::Priority::Red => 2u128,
        };
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, v)?;
            }
        }
        self.stats.priority_get_execs += 1;
        log_debug("sched", &format!("ctx {} PRIORITY_GET {}", ctx_id, v));
        Ok(())
    }

    /// LOCK: try-lock não-bloqueante. Livre ou próprio => adquire (reentrante
    /// p/ o dono); de outro ctx => Err. Sem filas de espera (follow-up EDF).
    fn exec_lock(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let id = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)? & 0xFFFF_FFFF) as u32
        };
        match self.locks.get(&id).copied() {
            None => {
                self.locks.insert(id, ctx_id);
            }
            Some(holder) if holder == ctx_id => {}
            Some(holder) => {
                return Err(anyhow!("LOCK {} retido por ctx {} (try-lock)", id, holder));
            }
        }
        self.stats.lock_execs += 1;
        log_debug("sched", &format!("ctx {} LOCK {} ok", ctx_id, id));
        Ok(())
    }

    /// UNLOCK: libera se dono; senão Err (não-possuído / de outro).
    fn exec_unlock(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let id = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)? & 0xFFFF_FFFF) as u32
        };
        match self.locks.get(&id).copied() {
            Some(holder) if holder == ctx_id => {
                self.locks.remove(&id);
            }
            Some(holder) => {
                return Err(anyhow!("UNLOCK {} pertence a ctx {}", id, holder));
            }
            None => {
                return Err(anyhow!("UNLOCK {} não retido", id));
            }
        }
        self.stats.unlock_execs += 1;
        log_debug("sched", &format!("ctx {} UNLOCK {} ok", ctx_id, id));
        Ok(())
    }

    /// FENCE: marcador de visibilidade + compiler fence. Neste alvo não há
    /// store buffer separado: documentado, não forjado (RFC-0006).
    fn exec_fence(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let _ = instr;
        if self.scheduler.get(ctx_id).is_none() {
            return Err(anyhow!("ctx {} não encontrado", ctx_id));
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
        self.stats.fence_execs += 1;
        log_debug("sched", &format!("ctx {} FENCE", ctx_id));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0007: LOADI (0x78) / MOV (0x79).
    // -----------------------------------------------------------------------

    /// LOADI rD, imm — materializa literal u128 (payload[0..16]).
    fn exec_loadi(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let imm = instr.imm_u128();
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, imm)?;
            }
        } else {
            return Err(anyhow!("ctx {} não encontrado", ctx_id));
        }
        self.stats.loadi_execs += 1;
        log_debug("ctrl", &format!("ctx {} LOADI r{} <- {}", ctx_id, instr.rdest, imm));
        Ok(())
    }

    /// MOV rD, rS — copia registrador (aliasing, spill/fill, args).
    fn exec_mov(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let v = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, v)?;
            }
        }
        self.stats.mov_execs += 1;
        log_debug("ctrl", &format!("ctx {} MOV r{} <- r{} ({})", ctx_id, instr.rdest, instr.rsrc1, v));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0010: KV_TRUNCATE (0x38). Wrapper fino sobre kv_cache_truncate
    // (maquinário + cobertura de snapshot já existentes e testados).
    // -----------------------------------------------------------------------

    /// KV_TRUNCATE Rs_len [, STREAM=sid] — trunca TODAS as camadas p/
    /// min(atual, len). len>=atual e cache vazio: no-op Ok. sid != 0: Err
    /// explícito (17 streams é follow-up; sem truncamento parcial fantasma).
    fn exec_kv_truncate(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let stream = instr.kv_stream();
        if stream > crate::memory::KV_MAX_STREAM {
            return Err(anyhow!("KV_TRUNCATE: STREAM={} fora de 0-16 (17 streams)", stream));
        }
        let len = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)? as usize
        };
        self.memory.kv_cache_truncate_stream(stream, len);
        self.stats.kv_truncate_execs += 1;
        log_debug("kv", &format!("ctx {} KV_TRUNCATE stream={} len={} seq={}", ctx_id, stream, len, self.memory.kv_cache_seq_len_stream(stream)));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0018: cluster F1 — semântica estritamente local (node_id == 0).
    // node_id != 0 veta limpo (transporte é F2+). Sem sockets aqui.
    // -----------------------------------------------------------------------

    /// REMOTE_SPAWN rD, node, entry_pc, prio — node 0: cria contexto local
    /// com regs zerados (spawn ≠ fork: sem herança), Rd = ctx_id.
    fn exec_remote_spawn(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::context::Priority;
        let (node, entry_pc, prio) = instr.remote_spawn_params();
        if node != 0 {
            return Err(anyhow!("REMOTE_SPAWN: nó {} remoto sem transporte (F2+; F1 é local)", node));
        }
        let prio = match prio {
            0 => Priority::Green,
            1 => Priority::Blue,
            2 => Priority::Red,
            _ => return Err(anyhow!("REMOTE_SPAWN: prio {} inválida (0/1/2)", prio)),
        };
        self.check_jump_target(entry_pc as u128)?;
        let root = self.memory.current_version();
        let new_id = self.scheduler.create_context(prio, entry_pc as u128, root);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, new_id as u128)?;
            }
        }
        self.stats.remote_spawn_execs += 1;
        log_debug("cluster", &format!("ctx {} REMOTE_SPAWN local -> ctx {} @ 0x{:x}", ctx_id, new_id, entry_pc));
        Ok(())
    }

    /// SIGNAL rD, kind, node, ctx, seq — fora-de-banda local.
    /// ABORT: termina + pops de engine (sem restore de heap — sem versão;
    /// use ABORT p/ isso). HALT/KILL: termina. PING: ack 0. FORK_REQ: Err
    /// explícito (use FORK; request remoto é F3). rdest = 0 no sucesso.
    fn exec_signal(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{
            SIGNAL_KIND_ABORT, SIGNAL_KIND_FORK_REQ, SIGNAL_KIND_HALT, SIGNAL_KIND_PING,
        };
        let kind = instr.rsrc1;
        let (node, target, seq) = instr.signal_params();
        if node != 0 {
            return Err(anyhow!("SIGNAL: nó {} remoto sem transporte (F2+)", node));
        }
        if kind != SIGNAL_KIND_ABORT
            && kind != SIGNAL_KIND_FORK_REQ
            && kind != SIGNAL_KIND_HALT
            && kind != SIGNAL_KIND_PING
        {
            return Err(anyhow!("SIGNAL: KIND {} inválido (0/1/2/3)", kind));
        }
        if kind == SIGNAL_KIND_FORK_REQ {
            return Err(anyhow!("SIGNAL FORK_REQ local: use FORK (request remoto é F3)"));
        }
        if kind == SIGNAL_KIND_PING {
            if target != 0 && self.scheduler.get(target).is_none() {
                return Err(anyhow!("SIGNAL PING: ctx {} inexistente", target));
            }
            if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                if instr.rdest != 0xFF {
                    ctx.set_reg(instr.rdest, 0)?;
                }
            }
            self.stats.signal_execs += 1;
            log_debug("cluster", &format!("ctx {} SIGNAL PING -> {} ack (seq {})", ctx_id, target, seq));
            return Ok(());
        }
        // ABORT / HALT: alvo obrigatório.
        if target == 0 {
            return Err(anyhow!("SIGNAL: KIND={} exige CTX alvo != 0", kind));
        }
        if self.scheduler.get(target).is_none() {
            return Err(anyhow!("SIGNAL: alvo ctx {} inexistente", target));
        }
        self.scheduler.remove(target);
        if kind == SIGNAL_KIND_ABORT {
            // Pops de engine como exec_abort, sem restore de heap.
            if let Some((_, snap)) = self.ssm_snapshots.pop() {
                self.ssm_states = snap;
            }
            if let Some((_, snap)) = self.rank1_snapshots.pop() {
                self.rank1_layers = snap;
            }
            if let Some((_, snap)) = self.snn_snapshots.pop() {
                self.snn_layers = snap;
            }
        }
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, 0)?;
            }
        }
        self.stats.signal_execs += 1;
        log_debug("cluster", &format!("ctx {} SIGNAL kind={} -> {} seq={}", ctx_id, kind, target, seq));
        Ok(())
    }

    /// SEND_TENSOR rS, rD|0xFF, node, off, len, mode — cópia local de bytes
    /// [off, off+len) do tensor fonte. COPY: tensor novo (ou destino dado
    /// com tamanho exato). MOVE: COPY + invalidação TOTAL da fonte
    /// (heap+meta+esparso). len 0: Err (não no-op silencioso).
    fn exec_send_tensor(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{SEND_MODE_COPY, SEND_MODE_MOVE};
        let (src_addr, dst_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let s = ctx.reg(instr.rdest)?;
            let d = if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? } else { 0xFF as u128 };
            (s, d)
        };
        let (node, off, len, mode) = instr.send_tensor_params();
        if node != 0 {
            return Err(anyhow!("SEND_TENSOR: nó {} remoto sem transporte (F2+)", node));
        }
        if mode != SEND_MODE_COPY && mode != SEND_MODE_MOVE {
            return Err(anyhow!("SEND_TENSOR: MODE {} inválido (0=COPY,1=MOVE)", mode));
        }
        if len == 0 {
            return Err(anyhow!("SEND_TENSOR: LEN 0 (bulk vazio é bug do chamador)"));
        }
        let src_meta = self.memory.get_tensor_meta(src_addr).cloned()
            .ok_or_else(|| anyhow!("SEND_TENSOR: fonte 0x{:x} não encontrada", src_addr))?;
        if src_meta.is_sparse {
            return Err(anyhow!("SEND_TENSOR: fonte esparsa (F2+: serialização CSR)"));
        }
        let total: usize = src_meta.shape.iter().product::<usize>() * 4;
        let (off, len) = (off as usize, len as usize);
        if off.saturating_add(len) > total {
            return Err(anyhow!("SEND_TENSOR: janela [{}..{}] fora do tensor ({}B)", off, off + len, total));
        }
        let bytes = self.memory.read(src_addr, total)?[off..off + len].to_vec();
        if instr.rsrc1 != 0xFF {
            let dmeta = self.memory.get_tensor_meta(dst_addr).cloned()
                .ok_or_else(|| anyhow!("SEND_TENSOR: destino 0x{:x} não encontrado", dst_addr))?;
            if dmeta.is_sparse {
                return Err(anyhow!("SEND_TENSOR: destino esparso não suportado"));
            }
            let dtotal: usize = dmeta.shape.iter().product::<usize>() * 4;
            // Escrita espelhada no mesmo offset (montagem parcial honesta);
            // destino menor que a janela => erro, nunca truncamento.
            if off.saturating_add(len) > dtotal {
                return Err(anyhow!("SEND_TENSOR: janela [{}..{}] fora do destino ({}B)", off, off + len, dtotal));
            }
            // Escreve o bulk deslocado: lê, emenda, escreve de volta
            // (preserva o restante do destino).
            let mut cur = self.memory.read(dst_addr, dtotal)?;
            cur[off..off + len].copy_from_slice(&bytes);
            self.memory.write(dst_addr, &cur)?;
        } else {
            // F1: sem driver não há para onde devolver tensor alocado (rdest
            // carrega a FONTE pelo encoding congelado) — destino explícito
            // exigido; a forma 0xFF volta com o transporte (F2+).
            return Err(anyhow!("SEND_TENSOR: destino 0xFF exige transporte (F2+); passe tensor destino explícito"));
        }
        if mode == SEND_MODE_MOVE {
            // Invalidação total: heap + meta + esparso (nada ressuscitável).
            if !self.memory.remove_tensor(src_addr) {
                return Err(anyhow!("SEND_TENSOR MOVE: fonte 0x{:x} já ausente", src_addr));
            }
            log_debug("cluster", &format!("ctx {} SEND_TENSOR MOVE 0x{:x} invalidado", ctx_id, src_addr));
        }
        self.stats.send_tensor_execs += 1;
        log_debug("cluster", &format!("ctx {} SEND_TENSOR mode={} {}B", ctx_id, mode, len));
        Ok(())
    }

    /// SLICE rD, rT — fatia flat [start, start+len) => tensor novo [1,len].
    /// Bounds validados pré-alloc; len 0 e OOB são Err (nunca clamp).
    /// Denso f32 apenas (RFC-0019).
    fn exec_slice(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (start, len) = instr.slice_params();
        let (start, len) = (start as usize, len as usize);
        if len == 0 {
            return Err(anyhow!("SLICE: LEN 0 (fatia vazia é bug do chamador)"));
        }
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("SLICE: tensor 0x{:x} não encontrado", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("SLICE: esparso não suportado"));
        }
        let flat: usize = meta.shape.iter().product();
        if start.saturating_add(len) > flat {
            return Err(anyhow!("SLICE: janela [{}..{}] fora do tensor ({} elems)", start, start + len, flat));
        }
        let data = self.memory.read_f32_tensor(t_addr, flat)?;
        let out_addr = self.memory.alloc_tensor(&[1, len], crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &data[start..start + len])?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.slice_execs += 1;
        log_debug("slice", &format!("ctx {} SLICE 0x{:x}[{}..{}] -> 0x{:x}", ctx_id, t_addr, start, start + len, out_addr));
        Ok(())
    }

    /// ARENA_ALLOC rD, SIZE=n [ALIGN=n] [ARENA=id] — bump-alloc na arena
    /// (criada no primeiro uso); rdest <- offset em bytes. FORK clona o
    /// mapa, ABORT restaura (disciplina rank1_layers, RFC-0011).
    fn exec_arena_alloc(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (size_p, align_p, arena_id) = instr.arena_alloc_params();
        let size = usize::try_from(size_p).map_err(|_| anyhow!("ARENA_ALLOC: SIZE {} não cabe", size_p))?;
        let align = usize::try_from(align_p).map_err(|_| anyhow!("ARENA_ALLOC: ALIGN {} não cabe", align_p))?;
        let arena = self.arenas.entry(arena_id).or_default();
        let off = arena.alloc(size, align).map_err(|e| anyhow!("ARENA_ALLOC: {}", e))?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, off as u128)?;
            }
        }
        self.stats.arena_alloc_execs += 1;
        log_debug("arena", &format!("ctx {} ARENA_ALLOC arena={} size={} align={} -> off={}", ctx_id, arena_id, size, align_p, off));
        Ok(())
    }

    /// ARENA_RESET [ARENA=id] — cursor=0 O(1), capacidade mantida.
    /// Arena inexistente é Err (sem criação silenciosa).
    fn exec_arena_reset(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let arena_id = instr.arena_id();
        let arena = self.arenas.get_mut(&arena_id)
            .ok_or_else(|| anyhow!("ARENA_RESET: arena {} inexistente (sem criação silenciosa)", arena_id))?;
        arena.reset();
        self.stats.arena_reset_execs += 1;
        log_debug("arena", &format!("ctx {} ARENA_RESET arena={}", ctx_id, arena_id));
        Ok(())
    }

    /// MEMCPY rDst, rSrc [LEN=n] [SRC_OFF=n] [DST_OFF=n] [DIR=HOST] —
    /// cópia byte-exata tensor->tensor. rDst/rSrc guardam addrs (não são
    /// reescritos). Lê tudo e reescreve: caminhos exatos de read/write
    /// (sem depender de busca-por-base com offset), overlap-safe (buffer
    /// próprio) e CoW-safe (`Arc::make_mut` no write => snapshots
    /// intactos, RFC-0016). Custo O(src+dst), documentado.
    fn exec_memcpy(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("MEMCPY precisa de rDst e rSrc com endereços de tensores"));
        }
        let (dst_addr, src_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rdest)?, ctx.reg(instr.rsrc1)?)
        };
        let (len_p, src_off_p, dst_off_p, dir) = instr.memcpy_params();
        if dir != MEMCPY_DIR_HOST {
            return Err(anyhow!("MEMCPY: DIR={} sem driver (só HOST; GPU/NIC são RFC futura)", dir));
        }
        let src_meta = self.memory.get_tensor_meta(src_addr).cloned()
            .ok_or_else(|| anyhow!("MEMCPY: fonte 0x{:x} não é tensor", src_addr))?;
        let dst_meta = self.memory.get_tensor_meta(dst_addr).cloned()
            .ok_or_else(|| anyhow!("MEMCPY: destino 0x{:x} não é tensor", dst_addr))?;
        if src_meta.is_sparse || dst_meta.is_sparse {
            return Err(anyhow!("MEMCPY: esparso não suportado (denso nesta RFC)"));
        }
        if crate::memory::region_of(dst_addr) == crate::memory::Region::Persistent {
            return Err(anyhow!("MEMCPY: destino em PERSISTENTE (WEIGHTS sempre RO, ESPEC §7.4)"));
        }
        let (src_bytes, dst_bytes) = (src_meta.byte_len, dst_meta.byte_len);
        let (len, so, doff) = if len_p == 0 {
            if src_off_p != 0 || dst_off_p != 0 {
                return Err(anyhow!("MEMCPY: LEN=0 seleciona o tensor fonte inteiro e exige offsets 0"));
            }
            (src_bytes, 0usize, 0usize)
        } else {
            let len = usize::try_from(len_p).map_err(|_| anyhow!("MEMCPY: LEN {} não cabe", len_p))?;
            let so = usize::try_from(src_off_p).map_err(|_| anyhow!("MEMCPY: SRC_OFF {} não cabe", src_off_p))?;
            let doff = usize::try_from(dst_off_p).map_err(|_| anyhow!("MEMCPY: DST_OFF {} não cabe", dst_off_p))?;
            (len, so, doff)
        };
        if len == 0 {
            return Err(anyhow!("MEMCPY: nada a copiar (fonte vazia)"));
        }
        if so.saturating_add(len) > src_bytes {
            return Err(anyhow!("MEMCPY: janela fonte [{}..{}] fora do tensor ({} bytes)", so, so + len, src_bytes));
        }
        if doff.saturating_add(len) > dst_bytes {
            return Err(anyhow!("MEMCPY: janela destino [{}..{}] fora do tensor ({} bytes)", doff, doff + len, dst_bytes));
        }
        let src_full = self.memory.read(src_addr, src_bytes)?;
        let mut dst_full = self.memory.read(dst_addr, dst_bytes)?;
        dst_full[doff..doff + len].copy_from_slice(&src_full[so..so + len]);
        self.memory.write(dst_addr, &dst_full)?;
        self.stats.memcpy_execs += 1;
        log_debug("memcpy", &format!("ctx {} MEMCPY 0x{:x}[{}..{}] -> 0x{:x}[{}..{}]", ctx_id, src_addr, so, so + len, dst_addr, doff, doff + len));
        Ok(())
    }

    /// MEMSET rT, PATTERN=n [LEN=n] [OFF=n] — fill byte-level (não
    /// value-level; value fill é TENSOR FILL, RFC-0019). Mesmas regras
    /// de região/esparso do MEMCPY.
    fn exec_memset(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF {
            return Err(anyhow!("MEMSET precisa de rTensor com endereço de tensor"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rdest)?
        };
        let (pattern, len_p, off_p) = instr.memset_params();
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("MEMSET: 0x{:x} não é tensor", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("MEMSET: esparso não suportado (denso nesta RFC)"));
        }
        if crate::memory::region_of(t_addr) == crate::memory::Region::Persistent {
            return Err(anyhow!("MEMSET: destino em PERSISTENTE (WEIGHTS sempre RO, ESPEC §7.4)"));
        }
        let total = meta.byte_len;
        let (len, off) = if len_p == 0 {
            if off_p != 0 {
                return Err(anyhow!("MEMSET: LEN=0 seleciona o tensor inteiro e exige OFF=0"));
            }
            (total, 0usize)
        } else {
            let len = usize::try_from(len_p).map_err(|_| anyhow!("MEMSET: LEN {} não cabe", len_p))?;
            let off = usize::try_from(off_p).map_err(|_| anyhow!("MEMSET: OFF {} não cabe", off_p))?;
            (len, off)
        };
        if len == 0 {
            return Err(anyhow!("MEMSET: nada a preencher (tensor vazio)"));
        }
        if off.saturating_add(len) > total {
            return Err(anyhow!("MEMSET: janela [{}..{}] fora do tensor ({} bytes)", off, off + len, total));
        }
        let mut full = self.memory.read(t_addr, total)?;
        full[off..off + len].fill(pattern);
        self.memory.write(t_addr, &full)?;
        self.stats.memset_execs += 1;
        log_debug("memset", &format!("ctx {} MEMSET 0x{:x}[{}..{}] <- 0x{:02x}", ctx_id, t_addr, off, off + len, pattern));
        Ok(())
    }

    /// SNAPSHOT rD [MASK=n] — snapshot nomeado: memória + 4 mapas de
    /// engine (mesmas 4 linhas de push do FORK; dedup é follow-up, os
    /// caminhos auditados ficam intocados). rdest <- version u64.
    /// Só MASK=0b111 executa (maquinário é tudo-ou-nada).
    fn exec_snapshot(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let mask = instr.snapshot_mask();
        if mask != SNAP_MASK_ALL {
            return Err(anyhow!("SNAPSHOT: MASK=0b{:03b} parcial não suportado (só 0b111; parcial é RFC futura)", mask));
        }
        let v = self.memory.snapshot();
        self.ssm_snapshots.push((v, self.ssm_states.clone()));
        self.rank1_snapshots.push((v, self.rank1_layers.clone()));
        self.snn_snapshots.push((v, self.snn_layers.clone()));
        self.arena_snapshots.push((v, self.arenas.clone()));
        self.dep_snapshots.push((v, self.dep_kv.clone()));
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, v as u128)?;
            }
        }
        self.stats.snapshot_execs += 1;
        log_debug("snapshot", &format!("ctx {} SNAPSHOT v{} (retidos {})", ctx_id, v, self.memory.snapshot_count()));
        Ok(())
    }

    /// RESTORE rV — rewind p/ version: `memory.restore` + pops
    /// versionados dos 4 mapas (mesma disciplina do ABORT ts!=0; ver
    /// exec_abort). Contextos intocados (ninguém morre). Versão
    /// desconhecida/expirada => Err alto. Contador nunca rebaixa
    /// (I-Mono, via `memory.restore` — teorema `restoreFix` inalterado).
    fn exec_restore(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rsrc1 == 0xFF {
            return Err(anyhow!("RESTORE precisa de rV com version u64"));
        }
        let version = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)? as u64
        };
        self.memory.restore(version)?;
        while self.ssm_snapshots.last().map(|(v, _)| *v > version).unwrap_or(false) {
            self.ssm_snapshots.pop();
        }
        if self.ssm_snapshots.last().map(|(v, _)| *v == version).unwrap_or(false) {
            if let Some((_, snap)) = self.ssm_snapshots.pop() {
                self.ssm_states = snap;
            }
        }
        while self.rank1_snapshots.last().map(|(v, _)| *v > version).unwrap_or(false) {
            self.rank1_snapshots.pop();
        }
        if self.rank1_snapshots.last().map(|(v, _)| *v == version).unwrap_or(false) {
            if let Some((_, snap)) = self.rank1_snapshots.pop() {
                self.rank1_layers = snap;
            }
        }
        while self.snn_snapshots.last().map(|(v, _)| *v > version).unwrap_or(false) {
            self.snn_snapshots.pop();
        }
        if self.snn_snapshots.last().map(|(v, _)| *v == version).unwrap_or(false) {
            if let Some((_, snap)) = self.snn_snapshots.pop() {
                self.snn_layers = snap;
            }
        }
        while self.arena_snapshots.last().map(|(v, _)| *v > version).unwrap_or(false) {
            self.arena_snapshots.pop();
        }
        if self.arena_snapshots.last().map(|(v, _)| *v == version).unwrap_or(false) {
            if let Some((_, snap)) = self.arena_snapshots.pop() {
                self.arenas = snap;
            }
        }
        while self.dep_snapshots.last().map(|(v, _)| *v > version).unwrap_or(false) {
            self.dep_snapshots.pop();
        }
        if self.dep_snapshots.last().map(|(v, _)| *v == version).unwrap_or(false) {
            if let Some((_, snap)) = self.dep_snapshots.pop() {
                self.dep_kv = snap;
            }
        }
        self.stats.restore_execs += 1;
        log_debug("restore", &format!("ctx {} RESTORE v{}", ctx_id, version));
        Ok(())
    }

    /// PREFETCH rT [LEN=n] [OFF=n] — hint de cache, só leitura. Valida
    /// tensor + janela (senão Err; esparso Err) e toca as páginas da
    /// janela (a cópia do read já falta; soma com black_box impede
    /// eliminação). Best-effort, sem promessa de desempenho.
    fn exec_prefetch(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF {
            return Err(anyhow!("PREFETCH precisa de rTensor com endereço de tensor"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rdest)?
        };
        let (len_p, off_p) = instr.prefetch_params();
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("PREFETCH: 0x{:x} não é tensor", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("PREFETCH: esparso não suportado (denso nesta RFC)"));
        }
        let total = meta.byte_len;
        let (len, off) = if len_p == 0 {
            if off_p != 0 {
                return Err(anyhow!("PREFETCH: LEN=0 seleciona o tensor inteiro e exige OFF=0"));
            }
            (total, 0usize)
        } else {
            let len = usize::try_from(len_p).map_err(|_| anyhow!("PREFETCH: LEN {} não cabe", len_p))?;
            let off = usize::try_from(off_p).map_err(|_| anyhow!("PREFETCH: OFF {} não cabe", off_p))?;
            (len, off)
        };
        if len == 0 {
            return Err(anyhow!("PREFETCH: nada a pré-carregar (tensor vazio)"));
        }
        if off.saturating_add(len) > total {
            return Err(anyhow!("PREFETCH: janela [{}..{}] fora do tensor ({} bytes)", off, off + len, total));
        }
        let full = self.memory.read(t_addr, total)?;
        let acc = full[off..off + len].iter().fold(0u64, |a, &b| a.wrapping_add(b as u64));
        std::hint::black_box(acc);
        self.stats.prefetch_execs += 1;
        log_debug("prefetch", &format!("ctx {} PREFETCH 0x{:x}[{}..{}]", ctx_id, t_addr, off, off + len));
        Ok(())
    }

    /// RESHAPE rD, rT (cópia com novo shape; NÃO é view — views com stride
    /// são follow-up). numel deve casar, senão Err. Denso; dtype preservado;
    /// layouts não-planos vetam (guarda defensiva p/ quantizados).
    fn exec_reshape(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("RESHAPE precisa de rdest e rTensor com endereços"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (ndim, dims) = instr.reshape_shape();
        if !(1..=4).contains(&ndim) {
            return Err(anyhow!("RESHAPE: ndim {} fora de 1-4", ndim));
        }
        let dims = &dims[..ndim as usize];
        if dims.iter().any(|&d| d == 0) {
            return Err(anyhow!("RESHAPE: dim 0 (dims > 0)"));
        }
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("RESHAPE: 0x{:x} não é tensor", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("RESHAPE: esparso não suportado (denso nesta RFC)"));
        }
        let old_numel: usize = meta.shape.iter().product();
        let new_numel: usize = dims.iter()
            .map(|&d| d as usize)
            .try_fold(1usize, |a, d| a.checked_mul(d))
            .ok_or_else(|| anyhow!("RESHAPE: produto das dims estoura"))?;
        if old_numel != new_numel {
            return Err(anyhow!("RESHAPE: numel {} != {} (shape deve preservar elementos)", new_numel, old_numel));
        }
        let width = meta.dtype.byte_width();
        if meta.byte_len != old_numel.saturating_mul(width) {
            return Err(anyhow!("RESHAPE: layout não-plano (quantizado é RFC futura)"));
        }
        let data = self.memory.read(t_addr, meta.byte_len)?;
        let new_shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let out_addr = self.memory.alloc_tensor(&new_shape, meta.dtype)?;
        self.memory.write(out_addr, &data)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.reshape_execs += 1;
        log_debug("reshape", &format!("ctx {} RESHAPE 0x{:x} {:?} -> {:?} 0x{:x}", ctx_id, t_addr, meta.shape, new_shape, out_addr));
        Ok(())
    }

    /// CONCAT rD, rA, rB [AXIS=n] — montagem N-D genérica ao longo do eixo:
    /// mesmo rank, demais dims iguais, mesmo dtype, layouts planos. Cópia
    /// (sem aliasing). Esparso veta.
    fn exec_concat(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rsrc1 == 0xFF || instr.rsrc2 == 0xFF {
            return Err(anyhow!("CONCAT precisa de rA e rB com endereços de tensores"));
        }
        let (a_addr, b_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let axis = instr.concat_axis() as usize;
        let ma = self.memory.get_tensor_meta(a_addr).cloned()
            .ok_or_else(|| anyhow!("CONCAT: A 0x{:x} não é tensor", a_addr))?;
        let mb = self.memory.get_tensor_meta(b_addr).cloned()
            .ok_or_else(|| anyhow!("CONCAT: B 0x{:x} não é tensor", b_addr))?;
        if ma.is_sparse || mb.is_sparse {
            return Err(anyhow!("CONCAT: esparso não suportado (denso nesta RFC)"));
        }
        if ma.dtype != mb.dtype {
            return Err(anyhow!("CONCAT: dtypes {:?} != {:?} (sem cast silencioso)", ma.dtype, mb.dtype));
        }
        if ma.shape.len() != mb.shape.len() || ma.shape.is_empty() {
            return Err(anyhow!("CONCAT: ranks {:?} vs {:?} (mesmo rank ≥ 1)", ma.shape, mb.shape));
        }
        let rank = ma.shape.len();
        if axis >= rank {
            return Err(anyhow!("CONCAT: AXIS {} fora do rank {}", axis, rank));
        }
        for (i, (da, db)) in ma.shape.iter().zip(mb.shape.iter()).enumerate() {
            if i != axis && da != db {
                return Err(anyhow!("CONCAT: dim {} difere ({} vs {}) fora do eixo", i, da, db));
            }
        }
        let width = ma.dtype.byte_width();
        for (m, tag) in [(&ma, "A"), (&mb, "B")] {
            let numel: usize = m.shape.iter().product();
            if m.byte_len != numel.saturating_mul(width) {
                return Err(anyhow!("CONCAT: {} com layout não-plano (quantizado é RFC futura)", tag));
            }
        }
        let (a_ax, b_ax) = (ma.shape[axis], mb.shape[axis]);
        let outer: usize = ma.shape[..axis].iter().product();
        let inner_elems: usize = ma.shape[axis + 1..].iter().product();
        let inner_bytes = inner_elems.checked_mul(width).ok_or_else(|| anyhow!("CONCAT: inner_bytes estoura"))?;
        let a_chunk = a_ax.checked_mul(inner_bytes).ok_or_else(|| anyhow!("CONCAT: janela A estoura"))?;
        let b_chunk = b_ax.checked_mul(inner_bytes).ok_or_else(|| anyhow!("CONCAT: janela B estoura"))?;
        let a_full = self.memory.read(a_addr, ma.byte_len)?;
        let b_full = self.memory.read(b_addr, mb.byte_len)?;
        if a_full.len() < outer.saturating_mul(a_chunk) || b_full.len() < outer.saturating_mul(b_chunk) {
            return Err(anyhow!("CONCAT: layout inconsistente com shape (sem leitura OOB)"));
        }
        let mut out_shape = ma.shape.clone();
        out_shape[axis] = a_ax.checked_add(b_ax).ok_or_else(|| anyhow!("CONCAT: dim resultante estoura"))?;
        let mut out_bytes = Vec::with_capacity(out_shape[axis].saturating_mul(outer).saturating_mul(inner_bytes));
        for o in 0..outer {
            let a_base = o.saturating_mul(a_chunk);
            out_bytes.extend_from_slice(&a_full[a_base..a_base + a_chunk]);
            let b_base = o.saturating_mul(b_chunk);
            out_bytes.extend_from_slice(&b_full[b_base..b_base + b_chunk]);
        }
        let out_addr = self.memory.alloc_tensor(&out_shape, ma.dtype)?;
        self.memory.write(out_addr, &out_bytes)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.concat_execs += 1;
        log_debug("concat", &format!("ctx {} CONCAT 0x{:x}{:?} + 0x{:x}{:?} AXIS={} -> 0x{:x}{:?}", ctx_id, a_addr, ma.shape, b_addr, mb.shape, axis, out_addr, out_shape));
        Ok(())
    }

    /// CAST rD, rT DST=... — conversão de valor para tensor NOVO (src
    /// intacto). Enforcement da Precision Rule (§11) no path 32B: par
    /// explícito ou trap — nunca fallback silencioso. Identidade veta
    /// (no-op é bug do chamador; use MOV).
    fn exec_cast(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("CAST precisa de rdest e rTensor com endereços"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let dst_code = instr.cast_dst();
        let dst_dtype = match dst_code {
            CAST_DST_F32 => DType::F32,
            CAST_DST_F16 => DType::F16,
            CAST_DST_BF16 => DType::BF16,
            CAST_DST_I8 => DType::I8,
            CAST_DST_U8 => DType::U8,
            _ => return Err(anyhow!("CAST: DST={} sem precisão suportada (use F32/F16/BF16/I8/U8)", dst_code)),
        };
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("CAST: 0x{:x} não é tensor", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("CAST: esparso não suportado (denso nesta RFC)"));
        }
        let src_dtype = meta.dtype;
        let plain = |d: DType| matches!(d, DType::F32 | DType::F16 | DType::BF16 | DType::I8 | DType::U8);
        if !plain(src_dtype) {
            return Err(anyhow!("CAST: fonte {:?} quantizada (use DEQUANT)", src_dtype));
        }
        if src_dtype == dst_dtype {
            return Err(anyhow!("CAST: identidade {:?}->mesmo (no-op; use MOV)", src_dtype));
        }
        if !(src_dtype == DType::F32 || dst_dtype == DType::F32) {
            return Err(anyhow!("CAST: só pares com F32 (transitivo via F32; direto é RFC futura)"));
        }
        let numel: usize = meta.shape.iter().product();
        if numel == 0 {
            return Err(anyhow!("CAST: tensor vazio"));
        }
        if meta.byte_len != numel.saturating_mul(src_dtype.byte_width()) {
            return Err(anyhow!("CAST: layout não-plano na fonte"));
        }
        let data = self.memory.read(t_addr, meta.byte_len)?;
        // Decodifica tudo para f32 (ponto comum honesto), depois codifica.
        let f32s: Vec<f32> = match src_dtype {
            DType::F32 => data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            DType::F16 => data.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect(),
            DType::BF16 => data.chunks_exact(2).map(|c| crate::quant::bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            DType::I8 => data.iter().map(|&b| (b as i8) as f32).collect(),
            DType::U8 => data.iter().map(|&b| b as f32).collect(),
            _ => return Err(anyhow!("CAST: fonte {:?} inalcançável", src_dtype)),
        };
        if f32s.len() != numel {
            return Err(anyhow!("CAST: bytes {} != numel {} (layout inconsistente)", data.len(), numel));
        }
        let out_bytes: Vec<u8> = match dst_dtype {
            DType::F32 => f32s.iter().flat_map(|v| v.to_le_bytes()).collect(),
            DType::F16 => f32s.iter().flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes()).collect(),
            DType::BF16 => f32s.iter().flat_map(|v| crate::quant::f32_to_bf16_bits(*v).to_le_bytes()).collect(),
            DType::I8 => f32s.iter().map(|v| v.round().clamp(-128.0, 127.0) as i8 as u8).collect(),
            DType::U8 => f32s.iter().map(|v| v.round().clamp(0.0, 255.0) as u8).collect(),
            _ => return Err(anyhow!("CAST: destino {:?} inalcançável", dst_dtype)),
        };
        let out_addr = self.memory.alloc_tensor(&meta.shape, dst_dtype)?;
        if crate::memory::region_of(out_addr) == crate::memory::Region::Persistent {
            return Err(anyhow!("CAST: saída colidiu com PERSISTENTE (alias de pesos; recuse alto)"));
        }
        self.memory.write(out_addr, &out_bytes)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.cast_execs += 1;
        log_debug("cast", &format!("ctx {} CAST 0x{:x} {:?} -> {:?} 0x{:x}", ctx_id, t_addr, src_dtype, dst_dtype, out_addr));
        Ok(())
    }

    /// QUANTIZE rD, rT Q=... — F32 denso -> blocos Q4_0/Q8_0 em tensor
    /// NOVO (shape lógica preservada). Entrada não-finita, numel fora de
    /// múltiplo de 32, ou tipo sem encoder => Err alto. Bound: |err| <=
    /// d/2 com o d armazenado (ver quant.rs).
    fn exec_quantize(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("QUANTIZE precisa de rdest e rTensor com endereços"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let q = instr.quantize_type();
        let (blk_elems, blk_bytes) = crate::quant::quant_block_info(q as u32)
            .ok_or_else(|| anyhow!("QUANTIZE: tipo {} sem encoder (só Q4_0/Q8_0; Q4_K/Q6_K são RFC futura)", q))?;
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("QUANTIZE: 0x{:x} não é tensor", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("QUANTIZE: esparso não suportado (denso nesta RFC)"));
        }
        if meta.dtype != DType::F32 {
            return Err(anyhow!("QUANTIZE: fonte {:?} precisa ser F32 denso", meta.dtype));
        }
        let numel: usize = meta.shape.iter().product();
        if numel == 0 || numel % blk_elems != 0 {
            return Err(anyhow!("QUANTIZE: numel {} fora de múltiplo de bloco {}", numel, blk_elems));
        }
        if meta.byte_len != numel.saturating_mul(4) {
            return Err(anyhow!("QUANTIZE: layout não-plano na fonte"));
        }
        let data = self.memory.read(t_addr, meta.byte_len)?;
        let f32s: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        if f32s.iter().any(|v| !v.is_finite()) {
            return Err(anyhow!("QUANTIZE: entrada não-finita (bound mentiria; sanitize antes)"));
        }
        let nblocks = numel / blk_elems;
        let mut out = vec![0u8; nblocks * blk_bytes];
        for b in 0..nblocks {
            let ok = if q == QUANTIZE_Q4_0 {
                crate::quant::quantize_q4_0(&f32s[b * blk_elems..(b + 1) * blk_elems], &mut out[b * blk_bytes..(b + 1) * blk_bytes])
            } else {
                crate::quant::quantize_q8_0(&f32s[b * blk_elems..(b + 1) * blk_elems], &mut out[b * blk_bytes..(b + 1) * blk_bytes])
            };
            if !ok {
                return Err(anyhow!("QUANTIZE: bloco {} rejeitado (interno; já validado acima)", b));
            }
        }
        let dtype = if q == QUANTIZE_Q4_0 { DType::Q4_0 } else { DType::Q8_0 };
        // Alocação com bytes exatos do layout em blocos (nunca via
        // alloc_tensor: numel*width mente p/ bloqueado; nunca via GGUF:
        // saída nova não pode ser alias de pesos).
        let out_addr = self.memory.alloc_tensor_bytes(&meta.shape, dtype, nblocks * blk_bytes)?;
        if crate::memory::region_of(out_addr) == crate::memory::Region::Persistent {
            return Err(anyhow!("QUANTIZE: saída colidiu com PERSISTENTE (alias de pesos; recuse alto)"));
        }
        self.memory.write(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.quantize_execs += 1;
        log_debug("quantize", &format!("ctx {} QUANTIZE 0x{:x} -> {:?} 0x{:x} ({} blocos)", ctx_id, t_addr, dtype, out_addr, nblocks));
        Ok(())
    }

    /// DEQUANT rD, rT — blocos -> F32 em tensor NOVO (mesmo shape).
    /// Tipos servidos pelo dispatcher existente (F32/F16/Q4_0/Q4_K/Q6_K/
    /// Q8_0); resto => Err. byte_len deve casar com os blocos.
    fn exec_dequant(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("DEQUANT precisa de rdest e rTensor com endereços"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let meta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("DEQUANT: 0x{:x} não é tensor", t_addr))?;
        if meta.is_sparse {
            return Err(anyhow!("DEQUANT: esparso não suportado (denso nesta RFC)"));
        }
        let (blk_elems, blk_bytes) = match meta.dtype {
            DType::F32 => (1usize, 4usize),
            DType::F16 => (1usize, 2usize),
            DType::Q4_0 => (32usize, 18usize),
            DType::Q8_0 => (32usize, 34usize),
            DType::Q4_K | DType::Q5_K => (256usize, 144usize),
            DType::Q6_K => (256usize, 210usize),
            _ => return Err(anyhow!("DEQUANT: {:?} sem decoder (dispatcher cobre F32/F16/Q4_0/Q4_K/Q6_K/Q8_0)", meta.dtype)),
        };
        let numel: usize = meta.shape.iter().product();
        if numel == 0 || numel % blk_elems != 0 {
            return Err(anyhow!("DEQUANT: numel {} fora de múltiplo de bloco {}", numel, blk_elems));
        }
        if meta.byte_len != (numel / blk_elems).saturating_mul(blk_bytes) {
            return Err(anyhow!("DEQUANT: byte_len {} inconsistente com blocos (esperado {})", meta.byte_len, (numel / blk_elems).saturating_mul(blk_bytes)));
        }
        let data = self.memory.read(t_addr, meta.byte_len)?;
        let mut out = vec![0.0f32; numel];
        // Números do dispatcher (GGML-ish), NÃO discriminantes DType
        // (Q4_0 é 2 lá, 4 aqui) — mapeamento explícito, sem `as u32`.
        let disp_dtype: u32 = match meta.dtype {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::Q4_0 => 2,
            DType::Q8_0 => 8,
            DType::Q4_K => 12,
            DType::Q5_K => 13,
            DType::Q6_K => 14,
            _ => return Err(anyhow!("DEQUANT: {:?} sem número no dispatcher", meta.dtype)),
        };
        if !crate::quant::dequantize(&data, disp_dtype, &mut out, numel) {
            return Err(anyhow!("DEQUANT: dispatcher rejeitou (interno; já validado acima)"));
        }
        let out_addr = self.memory.alloc_tensor(&meta.shape, DType::F32)?;
        if crate::memory::region_of(out_addr) == crate::memory::Region::Persistent {
            return Err(anyhow!("DEQUANT: saída colidiu com PERSISTENTE (alias de pesos; recuse alto)"));
        }
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.dequant_execs += 1;
        log_debug("dequant", &format!("ctx {} DEQUANT 0x{:x} {:?} -> F32 0x{:x}", ctx_id, t_addr, meta.dtype, out_addr));
        Ok(())
    }

    /// ADD_IMM rD, rS, IMM=n — rdest <- rsrc wrapping_add imm. Total
    /// (wrapping documentado, como `advance_pc`); só regs nomeados vetam.
    fn exec_add_imm(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("ADD_IMM precisa de rdest e rSrc (0xFF não é registrador)"));
        }
        let (v, imm) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, instr.imm_u128())
        };
        let out = v.wrapping_add(imm);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out)?;
        }
        self.stats.add_imm_execs += 1;
        log_debug("alu", &format!("ctx {} ADD_IMM r{} + {} -> r{}", ctx_id, instr.rsrc1, imm, instr.rdest));
        Ok(())
    }

    /// SUB_IMM rD, rS, IMM=n — rdest <- rsrc wrapping_sub imm. Único
    /// SUB do ISA (antes só havia a construção ADD+COMPARE).
    fn exec_sub_imm(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("SUB_IMM precisa de rdest e rSrc (0xFF não é registrador)"));
        }
        let (v, imm) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, instr.imm_u128())
        };
        let out = v.wrapping_sub(imm);
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, out)?;
        }
        self.stats.sub_imm_execs += 1;
        log_debug("alu", &format!("ctx {} SUB_IMM r{} - {} -> r{}", ctx_id, instr.rsrc1, imm, instr.rdest));
        Ok(())
    }

    /// STEPS rD — rdest <- instruções retiradas (u64). Lê o contador de
    /// COMPLETADAS (o loop incrementa pós-execute): determinístico para
    /// um caminho de programa — o relógio replay-exato ao lado do
    /// wall-clock `CYCLES_COUNT` (RFC-0005: tempo nunca dirige lógica).
    fn exec_steps(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF {
            return Err(anyhow!("STEPS precisa de rdest (0xFF não é registrador)"));
        }
        let n = self.stats.steps;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rdest, n as u128)?;
        }
        self.stats.steps_execs += 1;
        log_debug("alu", &format!("ctx {} STEPS -> r{} = {}", ctx_id, instr.rdest, n));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0027: forma (0x30-0x37). Invariantes de índice: todo índice
    // escrito é < out.len() por construção (cada produto parcial <= total
    // final <= usize::MAX); estouros de parse (u64) usam try_from/checked.
    // -----------------------------------------------------------------------

    /// Decomposição (outer, dim, inner) de um eixo. None se eixo >= rank.
    fn axis_decomp(shape: &[usize], axis: usize) -> Option<(usize, usize, usize)> {
        if axis >= shape.len() {
            return None;
        }
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        Some((outer, shape[axis], inner))
    }

    /// Ordem total determinística (RFC-0027): NaN como +inf; empate
    /// (incl. -0.0/0.0) sempre resolve para o menor índice original,
    /// nas duas direções.
    fn lane_cmp(a: (f32, usize), b: (f32, usize), desc: bool) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        let ord = match (a.0.is_nan(), b.0.is_nan()) {
            (true, true) => Equal,
            (true, false) => Greater,
            (false, true) => Less,
            (false, false) => a.0.partial_cmp(&b.0).unwrap_or(Equal),
        };
        let ord = if desc && ord != Equal { ord.reverse() } else { ord };
        if ord == Equal {
            a.1.cmp(&b.1)
        } else {
            ord
        }
    }

    /// Lê tensor denso F32 + shape (guards uniformes RFC-0027: meta
    /// existe, denso, F32, layout plano, não-vazio).
    fn read_dense_f32(&self, addr: u128, op: &str) -> Result<(Vec<usize>, Vec<f32>)> {
        let meta = self.memory.get_tensor_meta(addr).cloned()
            .ok_or_else(|| anyhow!("{}: 0x{:x} não é tensor", op, addr))?;
        if meta.is_sparse {
            return Err(anyhow!("{}: esparso não suportado (denso nesta RFC)", op));
        }
        if meta.dtype != crate::memory::DType::F32 {
            return Err(anyhow!("{}: dtype {:?} (só F32 nesta RFC)", op, meta.dtype));
        }
        let numel: usize = meta.shape.iter().product();
        if numel == 0 {
            return Err(anyhow!("{}: tensor vazio", op));
        }
        if meta.byte_len != numel.saturating_mul(4) {
            return Err(anyhow!("{}: layout não-plano", op));
        }
        let data = self.memory.read_f32_tensor(addr, numel)?;
        Ok((meta.shape, data))
    }

    /// SORT rD, rT [AXIS] [ORDER] — ordenação por eixo em tensor NOVO.
    fn exec_sort(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("SORT precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (axis_p, order) = instr.sort_params();
        if order != SORT_ASC && order != SORT_DESC {
            return Err(anyhow!("SORT: ORDER={} inválido (0=ASC,1=DESC)", order));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "SORT")?;
        let axis = axis_p as usize;
        let (outer, dim, inner) = Self::axis_decomp(&shape, axis)
            .ok_or_else(|| anyhow!("SORT: AXIS {} fora do rank {}", axis, shape.len()))?;
        let desc = order == SORT_DESC;
        let mut out = data.clone();
        for o in 0..outer {
            for i in 0..inner {
                let mut lane: Vec<(f32, usize)> =
                    (0..dim).map(|k| (data[o * dim * inner + k * inner + i], k)).collect();
                lane.sort_by(|a, b| Self::lane_cmp(*a, *b, desc));
                for (k, (v, _)) in lane.iter().enumerate() {
                    out[o * dim * inner + k * inner + i] = *v;
                }
            }
        }
        let out_addr = self.memory.alloc_tensor(&shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.sort_execs += 1;
        log_debug("sort", &format!("ctx {} SORT 0x{:x}{:?} AXIS={} -> 0x{:x}", ctx_id, t_addr, shape, axis, out_addr));
        Ok(())
    }

    /// TOPK rD, rT — [k valores | k índices] ao longo do eixo (pack
    /// DISTANCE, RFC-0004). Saída: mesmo rank, dim do eixo = 2k.
    fn exec_topk(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("TOPK precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (axis_p, k_p, largest, sorted) = instr.topk_params();
        if largest > 1 || sorted > 1 {
            return Err(anyhow!("TOPK: flags fora de 0/1"));
        }
        if k_p == 0 {
            return Err(anyhow!("TOPK: K 0 (seleção vazia é bug do chamador)"));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "TOPK")?;
        let axis = axis_p as usize;
        let (outer, dim, inner) = Self::axis_decomp(&shape, axis)
            .ok_or_else(|| anyhow!("TOPK: AXIS {} fora do rank {}", axis, shape.len()))?;
        let k = k_p as usize;
        if k > dim {
            return Err(anyhow!("TOPK: K={} > dim={} (sem clamp silencioso)", k, dim));
        }
        let desc = largest != 0;
        let mut out_shape = shape.clone();
        out_shape[axis] = 2 * k;
        let mut out = vec![0.0f32; outer * (2 * k) * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut lane: Vec<(f32, usize)> =
                    (0..dim).map(|kk| (data[o * dim * inner + kk * inner + i], kk)).collect();
                lane.sort_by(|a, b| Self::lane_cmp(*a, *b, desc));
                let mut sel = lane[..k].to_vec();
                if sorted == 0 {
                    sel.sort_by_key(|&(_, idx)| idx); // ordem original — ainda determinístico
                }
                for (j, (v, idx)) in sel.iter().enumerate() {
                    out[(o * (2 * k) + j) * inner + i] = *v;
                    out[(o * (2 * k) + k + j) * inner + i] = *idx as f32;
                }
            }
        }
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.topk_execs += 1;
        log_debug("topk", &format!("ctx {} TOPK 0x{:x}{:?} K={} AXIS={} -> 0x{:x}", ctx_id, t_addr, shape, k, axis, out_addr));
        Ok(())
    }

    /// ARGMAX rD, rT — índices como f32 (exatos p/ dim < 2^24), dim do
    /// eixo vira 1. NaN ignorado salvo tudo-NaN (menor índice) —
    /// maxNum-consistente com REDUCE (não numpy, documentado).
    fn exec_argmax(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("ARGMAX precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let axis = instr.argmax_axis() as usize;
        let (shape, data) = self.read_dense_f32(t_addr, "ARGMAX")?;
        let (outer, dim, inner) = Self::axis_decomp(&shape, axis)
            .ok_or_else(|| anyhow!("ARGMAX: AXIS {} fora do rank {}", axis, shape.len()))?;
        let mut out_shape = shape.clone();
        out_shape[axis] = 1;
        let mut out = vec![0.0f32; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut best_idx = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                let mut seen = false;
                let mut nan_idx: Option<usize> = None;
                for k in 0..dim {
                    let v = data[o * dim * inner + k * inner + i];
                    if v.is_nan() {
                        if nan_idx.is_none() {
                            nan_idx = Some(k);
                        }
                    } else if !seen || v > best_val {
                        best_val = v;
                        best_idx = k;
                        seen = true;
                    }
                }
                out[o * inner + i] = nan_idx.map_or(best_idx, |n| if seen { best_idx } else { n }) as f32;
            }
        }
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.argmax_execs += 1;
        log_debug("argmax", &format!("ctx {} ARGMAX 0x{:x}{:?} AXIS={} -> 0x{:x}", ctx_id, t_addr, shape, axis, out_addr));
        Ok(())
    }

    /// REDUCE rD, rT OP=... [AXIS=n] — redução em tensor NOVO (eixo vira
    /// 1; sem eixo = escalar [1]). SUM/MEAN/PROD propagam NaN (aritmética
    /// f32); MAX/MIN pulam NaN (f32::max), tudo-NaN => NaN.
    fn exec_reduce(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("REDUCE precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (op, axis_p) = instr.reduce_params();
        if op > REDUCE_PROD {
            return Err(anyhow!("REDUCE: OP={} inválido (0-4)", op));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "REDUCE")?;
        // Sem eixo (0xFF): escalar [1]; com eixo: dim vira 1.
        let (out_shape, lanes): (Vec<usize>, Vec<(usize, usize, usize)>) = if axis_p == 0xFF {
            (vec![1], vec![(0, shape.iter().product(), 1)])
        } else {
            let axis = axis_p as usize;
            let (outer, dim, inner) = Self::axis_decomp(&shape, axis)
                .ok_or_else(|| anyhow!("REDUCE: AXIS {} fora do rank {}", axis, shape.len()))?;
            let mut os = shape.clone();
            os[axis] = 1;
            let mut ls = Vec::new();
            for o in 0..outer {
                for i in 0..inner {
                    // Cada lane: base + k*inner, k em 0..dim.
                    ls.push((o * dim * inner + i, dim, inner));
                }
            }
            (os, ls)
        };
        let mut out = Vec::with_capacity(lanes.len());
        for (base, dim, stride) in lanes {
            let lane: Vec<f32> = (0..dim).map(|k| data[base + k * stride]).collect();
            let v = match op {
                REDUCE_SUM => lane.iter().sum(),
                REDUCE_MEAN => lane.iter().sum::<f32>() / dim as f32,
                REDUCE_PROD => lane.iter().product(),
                REDUCE_MAX => lane.iter().copied().reduce(f32::max).unwrap_or(f32::NAN),
                _ => lane.iter().copied().reduce(f32::min).unwrap_or(f32::NAN),
            };
            out.push(v);
        }
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.reduce_execs += 1;
        log_debug("reduce", &format!("ctx {} REDUCE 0x{:x}{:?} OP={} -> 0x{:x}", ctx_id, t_addr, shape, op, out_addr));
        Ok(())
    }

    /// BROADCAST rD, rT SHAPE=... — expansão right-aligned (cópia).
    fn exec_broadcast(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("BROADCAST precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (ndim, dims4) = instr.broadcast_shape();
        if !(1..=4).contains(&ndim) {
            return Err(anyhow!("BROADCAST: ndim {} fora de 1-4", ndim));
        }
        let target: Vec<usize> = dims4[..ndim as usize].iter().map(|&d| d as usize).collect();
        if target.iter().any(|&d| d == 0) {
            return Err(anyhow!("BROADCAST: dim 0 (dims > 0)"));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "BROADCAST")?;
        if target.len() < shape.len() {
            return Err(anyhow!("BROADCAST: alvo {:?} afunila rank {:?} (sem squeeze silencioso)", target, shape));
        }
        let off = target.len() - shape.len();
        for (i, &sd) in shape.iter().enumerate() {
            let td = target[off + i];
            if sd != 1 && sd != td {
                return Err(anyhow!("BROADCAST: dim {} incompatível ({} vs {})", off + i, sd, td));
            }
        }
        // Strides da fonte (0 onde broadcast: dim fonte 1 repete ou é
        // constante; stride real só quando fonte e alvo coincidem > 1).
        let mut src_strides = vec![0usize; target.len()];
        let mut stride = 1usize;
        for i in (0..target.len()).rev() {
            let sd = if i < off { 1 } else { shape[i - off] };
            src_strides[i] = if target[i] == sd && sd != 1 { stride } else { 0 };
            stride = stride.checked_mul(target[i])
                .ok_or_else(|| anyhow!("BROADCAST: shape alvo estoura"))?;
        }
        let total: usize = target.iter().try_fold(1usize, |a, &d| a.checked_mul(d))
            .ok_or_else(|| anyhow!("BROADCAST: shape alvo estoura"))?;
        if total == 0 {
            return Err(anyhow!("BROADCAST: alvo vazio"));
        }
        // Odômetro sobre o alvo.
        let mut out = Vec::with_capacity(total);
        let mut coord = vec![0usize; target.len()];
        for _ in 0..total {
            let mut src_flat = 0usize;
            for (i, &c) in coord.iter().enumerate() {
                src_flat += c * src_strides[i];
            }
            out.push(data[src_flat]);
            // incrementa odômetro (eixo 0 mais significativo)
            for i in (0..target.len()).rev() {
                coord[i] += 1;
                if coord[i] < target[i] {
                    break;
                }
                coord[i] = 0;
            }
        }
        let out_addr = self.memory.alloc_tensor(&target, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.broadcast_execs += 1;
        log_debug("broadcast", &format!("ctx {} BROADCAST 0x{:x}{:?} -> {:?} 0x{:x}", ctx_id, t_addr, shape, target, out_addr));
        Ok(())
    }

    /// PAD rD, rT — crescimento single-axis com valor (tensor NOVO).
    /// Multi-eixo compõe por repetição (desenho). before==after==0 veta
    /// (no-op; use MOV).
    fn exec_pad(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("PAD precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (value, axis_p, before_p, after_p) = instr.pad_params();
        let (shape, data) = self.read_dense_f32(t_addr, "PAD")?;
        let axis = axis_p as usize;
        let (outer, dim, inner) = Self::axis_decomp(&shape, axis)
            .ok_or_else(|| anyhow!("PAD: AXIS {} fora do rank {}", axis, shape.len()))?;
        let (before, after) = (before_p as usize, after_p as usize);
        if before == 0 && after == 0 {
            return Err(anyhow!("PAD: BEFORE=AFTER=0 (no-op; use MOV)"));
        }
        let new_dim = dim.checked_add(before).and_then(|d| d.checked_add(after))
            .ok_or_else(|| anyhow!("PAD: dim resultante estoura"))?;
        let mut out_shape = shape.clone();
        out_shape[axis] = new_dim;
        let total = outer.checked_mul(new_dim).and_then(|t| t.checked_mul(inner))
            .ok_or_else(|| anyhow!("PAD: shape resultante estoura"))?;
        let mut out = vec![value; total];
        for o in 0..outer {
            for i in 0..inner {
                for k in 0..dim {
                    out[(o * new_dim + before + k) * inner + i] =
                        data[(o * dim + k) * inner + i];
                }
            }
        }
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.pad_execs += 1;
        log_debug("pad", &format!("ctx {} PAD 0x{:x}{:?} AXIS={}+{}+{} -> 0x{:x}", ctx_id, t_addr, shape, before, dim, after, out_addr));
        Ok(())
    }

    /// TILE rD, rT — repetição single-axis (tensor NOVO). REPS=0 veta.
    fn exec_tile(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("TILE precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (reps_p, axis_p) = instr.tile_params();
        if reps_p == 0 {
            return Err(anyhow!("TILE: REPS 0 (saída vazia é bug do chamador)"));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "TILE")?;
        let axis = axis_p as usize;
        let (outer, dim, inner) = Self::axis_decomp(&shape, axis)
            .ok_or_else(|| anyhow!("TILE: AXIS {} fora do rank {}", axis, shape.len()))?;
        let reps = reps_p as usize;
        let new_dim = dim.checked_mul(reps)
            .ok_or_else(|| anyhow!("TILE: dim resultante estoura"))?;
        let mut out_shape = shape.clone();
        out_shape[axis] = new_dim;
        let total = outer.checked_mul(new_dim).and_then(|t| t.checked_mul(inner))
            .ok_or_else(|| anyhow!("TILE: shape resultante estoura"))?;
        let mut out = Vec::with_capacity(total);
        for o in 0..outer {
            for _ in 0..reps {
                for k in 0..dim {
                    out.extend_from_slice(&data[(o * dim + k) * inner..(o * dim + k) * inner + inner]);
                }
            }
        }
        debug_assert_eq!(out.len(), total);
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.tile_execs += 1;
        log_debug("tile", &format!("ctx {} TILE 0x{:x}{:?} AXIS={}x{} -> 0x{:x}", ctx_id, t_addr, shape, axis, reps, out_addr));
        Ok(())
    }

    /// TRANSPOSE rD, rT — permuta N-D genérica (tensor NOVO, cópia).
    /// ndim deve igualar o rank; perm total (cada eixo 1×, todos < ndim).
    fn exec_transpose(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("TRANSPOSE precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (ndim_p, perm4) = instr.transpose_perm();
        let (shape, data) = self.read_dense_f32(t_addr, "TRANSPOSE")?;
        let ndim = ndim_p as usize;
        if ndim != shape.len() || !(1..=4).contains(&ndim) {
            return Err(anyhow!("TRANSPOSE: ndim {} != rank {} (1-4, exato)", ndim, shape.len()));
        }
        let perm = &perm4[..ndim];
        let mut seen = [false; 4];
        for &p in perm {
            if (p as usize) >= ndim {
                return Err(anyhow!("TRANSPOSE: eixo {} fora do rank {}", p, ndim));
            }
            if seen[p as usize] {
                return Err(anyhow!("TRANSPOSE: eixo {} repetido (permutação total)", p));
            }
            seen[p as usize] = true;
        }
        let out_shape: Vec<usize> = perm.iter().map(|&p| shape[p as usize]).collect();
        // Strides row-major (entrada e saída). in_flat abaixo é sempre
        // < numel por construção (permutação bijetora de coords válidas),
        // sem clamp: indexação direta, sem máscara de bug.
        let mut in_strides = vec![0usize; ndim];
        let mut st = 1usize;
        for i in (0..ndim).rev() {
            in_strides[i] = st;
            st *= shape[i];
        }
        let mut out_strides = vec![0usize; ndim];
        st = 1usize;
        for i in (0..ndim).rev() {
            out_strides[i] = st;
            st *= out_shape[i];
        }
        let numel = data.len();
        let mut out = vec![0.0f32; numel];
        for out_flat in 0..numel {
            let mut tmp = out_flat;
            let mut in_flat = 0usize;
            for (i, &p) in perm.iter().enumerate() {
                let c = tmp / out_strides[i];
                tmp %= out_strides[i];
                in_flat += c * in_strides[p as usize];
            }
            out[out_flat] = data[in_flat];
        }
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.transpose_execs += 1;
        log_debug("transpose", &format!("ctx {} TRANSPOSE 0x{:x}{:?} {:?} -> 0x{:x}", ctx_id, t_addr, shape, perm, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0028: ativações (0x3C-0x43). Núcleo em activations.rs; aqui só
    // leitura/escrita de tensores (guards uniformes) e dispatch.
    // -----------------------------------------------------------------------

    /// Op elementar unário rD, rT em tensor NOVO (f: total sobre f32).
    fn exec_unary(&mut self, ctx_id: u64, instr: &Instruction, f: fn(f32) -> f32, name: &str) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("{} precisa de rdest e rTensor (0xFF não é registrador)", name));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (shape, data) = self.read_dense_f32(t_addr, name)?;
        let out: Vec<f32> = data.iter().map(|&x| f(x)).collect();
        let out_addr = self.memory.alloc_tensor(&shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        log_debug("unary", &format!("ctx {} {} 0x{:x}{:?} -> 0x{:x}", ctx_id, name, t_addr, shape, out_addr));
        Ok(())
    }

    /// SOFTMAX rD, rT [AXIS] [TEMP] — normalização estável por lane em
    /// tensor NOVO. TEMP NaN/<=0 veta (+inf = uniforme, documentado).
    fn exec_softmax(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("SOFTMAX precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (axis_p, temp) = instr.softmax_params();
        if temp.is_nan() || temp <= 0.0 {
            return Err(anyhow!("SOFTMAX: TEMP={} inválido (> 0 ou +inf)", temp));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "SOFTMAX")?;
        if shape.is_empty() {
            return Err(anyhow!("SOFTMAX: escalar não tem eixo"));
        }
        let axis = if axis_p == 0xFF { shape.len() - 1 } else { axis_p as usize };
        let (outer, dim, inner) = Self::axis_decomp(&shape, axis)
            .ok_or_else(|| anyhow!("SOFTMAX: AXIS {} fora do rank {}", axis, shape.len()))?;
        let mut out = data.clone();
        for o in 0..outer {
            for i in 0..inner {
                let mut lane = vec![0.0f32; dim];
                for k in 0..dim {
                    lane[k] = out[o * dim * inner + k * inner + i];
                }
                crate::activations::softmax_lane(&mut lane, temp);
                for (k, v) in lane.iter().enumerate() {
                    out[o * dim * inner + k * inner + i] = *v;
                }
            }
        }
        let out_addr = self.memory.alloc_tensor(&shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.softmax_execs += 1;
        log_debug("softmax", &format!("ctx {} SOFTMAX 0x{:x}{:?} AXIS={} TEMP={} -> 0x{:x}", ctx_id, t_addr, shape, axis, temp, out_addr));
        Ok(())
    }

    /// CLIP rD, rT MIN=x MAX=x — clamp em tensor NOVO. Bounds finitos e
    /// min<=max, senão Err (`clamp` entraria em pânico com NaN).
    fn exec_clip(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("CLIP precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (min, max) = instr.clip_params();
        if !min.is_finite() || !max.is_finite() {
            return Err(anyhow!("CLIP: bounds não-finitos (min={}, max={})", min, max));
        }
        if min > max {
            return Err(anyhow!("CLIP: MIN={} > MAX={} (janela invertida)", min, max));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "CLIP")?;
        let out: Vec<f32> = data.iter().map(|&x| x.clamp(min, max)).collect();
        let out_addr = self.memory.alloc_tensor(&shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.clip_execs += 1;
        log_debug("clip", &format!("ctx {} CLIP 0x{:x}{:?} [{},{}] -> 0x{:x}", ctx_id, t_addr, shape, min, max, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0029: KV/attention (0x39/0x3A/0x3B). Equivalência com ATTN denso
    // verificada por TOLERÂNCIA (associação fp difere do path ndarray) —
    // nunca afirmada bit-exata.
    // -----------------------------------------------------------------------

    /// KV_COMPRESS SINK=n WINDOW=n — evicção sink+janela por camada
    /// (StreamingLLM-style). Sem registradores. No-op Ok quando já cabe;
    /// (0,0) veta (degenerado). Rollback via snapshots (não snapshota).
    fn exec_kv_compress(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (sink_p, window_p, mode, stream) = instr.kv_compress_params();
        if mode != KVCOMP_MODE_SINK_WINDOW {
            return Err(anyhow!("KV_COMPRESS: MODE={} inválido (só SINK_WINDOW)", mode));
        }
        if stream > crate::memory::KV_MAX_STREAM {
            return Err(anyhow!("KV_COMPRESS: STREAM={} fora de 0-16 (17 streams)", stream));
        }
        if sink_p == 0 && window_p == 0 {
            return Err(anyhow!("KV_COMPRESS: SINK=WINDOW=0 esvaziaria o cache (degenerado)"));
        }
        let before = self.memory.kv_cache_seq_len_stream(stream);
        self.memory.kv_cache_compress_sink_window_stream(stream, sink_p as usize, window_p as usize);
        let after = self.memory.kv_cache_seq_len_stream(stream);
        self.stats.kv_compress_execs += 1;
        log_debug("kv", &format!("ctx {} KV_COMPRESS stream={} sink={} window={} seq {} -> {}", ctx_id, stream, sink_p, window_p, before, after));
        Ok(())
    }

    /// Lê Q/K/V densos F32 2-D (guards uniformes; formas validadas como
    /// ATTN: Q.cols==K.cols, K.rows==V.rows). Retorna (M, N, D, Vd, q, k, v).
    fn read_qkv(&self, ctx_id: u64, instr: &Instruction, op: &str) -> Result<(usize, usize, usize, usize, Vec<f32>, Vec<f32>, Vec<f32>)> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF || instr.rsrc2 == 0xFF || instr.rsrc3 == 0xFF {
            return Err(anyhow!("{} precisa de rdest, rQ, rK, rV (0xFF não é registrador)", op));
        }
        let (aq, ak, av) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?, ctx.reg(instr.rsrc3)?)
        };
        let (sq, q) = self.read_dense_f32(aq, op)?;
        let (sk, k) = self.read_dense_f32(ak, op)?;
        let (sv, v) = self.read_dense_f32(av, op)?;
        if sq.len() != 2 || sk.len() != 2 || sv.len() != 2 {
            return Err(anyhow!("{}: tensores 2-D (Q{:?} K{:?} V{:?})", op, sq, sk, sv));
        }
        if sq[1] != sk[1] {
            return Err(anyhow!("{} incompatível: Q cols {} != K cols {}", op, sq[1], sk[1]));
        }
        if sk[0] != sv[0] {
            return Err(anyhow!("{} incompatível: K rows {} != V rows {}", op, sk[0], sv[0]));
        }
        Ok((sq[0], sk[0], sq[1], sv[1], q, k, v))
    }

    /// FLASH_ATTN rD, rQ, rK, rV [BLOCK=n] — atenção em blocos com online
    /// softmax (o algoritmo FlashAttention de verdade, tamanho CPU):
    /// máximo/normalizador/acumulador correntes com reescala por bloco.
    /// Flags devem ser 0 (NOTIFY não existe no path em blocos — alto, não
    /// descartado quieto). Saída [M, Vd], tolerância-verificada vs ATTN.
    fn exec_flash_attn(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.flags != 0 {
            return Err(anyhow!("FLASH_ATTN: flags 0x{:02x} sem suporte no path em blocos", instr.flags));
        }
        let block_p = instr.flash_block();
        let block = if block_p == 0 { 32 } else { block_p as usize };
        let (mq, n, d, vd, q, k, v) = self.read_qkv(ctx_id, instr, "FLASH_ATTN")?;
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; mq * vd];
        for m in 0..mq {
            let mut rm = f32::NEG_INFINITY;
            let mut rl = 0.0f32;
            let mut ro = vec![0.0f32; vd];
            let mut b0 = 0usize;
            while b0 < n {
                let b1 = (b0 + block).min(n);
                // Máximo do bloco (duas passadas: clareza > micro-ótimo;
                // fundir é follow-up de perf, semântica idêntica).
                let mut bm = f32::NEG_INFINITY;
                for j in b0..b1 {
                    let mut s = 0.0f32;
                    for dd in 0..d {
                        s += q[m * d + dd] * k[j * d + dd];
                    }
                    let s = s * scale;
                    if s > bm {
                        bm = s;
                    }
                }
                let new_m = if bm > rm { bm } else { rm };
                let a_old = (rm - new_m).exp();
                rl *= a_old;
                for x in ro.iter_mut() {
                    *x *= a_old;
                }
                for j in b0..b1 {
                    let mut s = 0.0f32;
                    for dd in 0..d {
                        s += q[m * d + dd] * k[j * d + dd];
                    }
                    let p = ((s * scale) - new_m).exp();
                    rl += p;
                    for vv in 0..vd {
                        ro[vv] += p * v[j * vd + vv];
                    }
                }
                rm = new_m;
                b0 = b1;
            }
            for vv in 0..vd {
                out[m * vd + vv] = ro[vv] / rl;
            }
        }
        let out_shape = vec![mq, vd];
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.flash_attn_execs += 1;
        log_debug("flash", &format!("ctx {} FLASH_ATTN [{}x{}]x[{}x{}] BLOCK={} -> 0x{:x}", ctx_id, mq, d, n, vd, block, out_addr));
        Ok(())
    }

    /// ATTN_SPARSE rD, rQ, rK, rV [TOPK] — top-k fundido por DOT (scores
    /// são dots) + softmax + V. Mesma ordem total da casa (lane_cmp:
    /// NaN-primeiro em desc, empate menor idx). k=0 = todos (mesmo
    /// kernel, sem caso especial). Fecha o follow-up da RFC-0004.
    fn exec_attn_sparse(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (metric, topk_p) = instr.sparse_params();
        if metric != SPARSE_METRIC_DOT {
            return Err(anyhow!("ATTN_SPARSE: METRIC={} sem implementação (só DOT)", metric));
        }
        let (mq, n, d, vd, q, k, v) = self.read_qkv(ctx_id, instr, "ATTN_SPARSE")?;
        let kk = if topk_p == 0 { n } else { topk_p as usize };
        if kk > n {
            return Err(anyhow!("ATTN_SPARSE: TOPK={} > N={} (sem clamp silencioso)", kk, n));
        }
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![0.0f32; mq * vd];
        for m in 0..mq {
            // Scores + seleção top-k (ordem total da casa).
            let mut scored: Vec<(f32, usize)> = (0..n)
                .map(|j| {
                    let mut s = 0.0f32;
                    for dd in 0..d {
                        s += q[m * d + dd] * k[j * d + dd];
                    }
                    (s * scale, j)
                })
                .collect();
            scored.sort_by(|a, b| Self::lane_cmp(*a, *b, true));
            let mut sel: Vec<f32> = scored[..kk].iter().map(|&(s, _)| s).collect();
            crate::activations::softmax_lane(&mut sel, 1.0);
            for (t, w) in sel.iter().enumerate() {
                let j = scored[t].1;
                for vv in 0..vd {
                    out[m * vd + vv] += *w * v[j * vd + vv];
                }
            }
        }
        let out_shape = vec![mq, vd];
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.attn_sparse_execs += 1;
        log_debug("sparse-attn", &format!("ctx {} ATTN_SPARSE [{}x{}] TOPK={}/{} -> 0x{:x}", ctx_id, mq, d, kk, n, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0031: DSP de áudio (0x45-0x49). Tensores densos F32, saídas novas.
    // -----------------------------------------------------------------------

    /// VAD_DETECT rD, rT [MODE] — score [1,1]: RMS cru (ENERGY, >= 0) ou
    /// taxa de cruzamento de zero (ZCR, [0,1], troca estrita de sinal).
    /// ML parseia e veta (sem modelo). n < 2 com ZCR veta.
    fn exec_vad_detect(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("VAD_DETECT precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let mode = instr.vad_mode();
        if mode != VAD_MODE_ENERGY && mode != VAD_MODE_ZCR && mode != VAD_MODE_ML {
            return Err(anyhow!("VAD_DETECT: MODE={} inválido (0/1/2)", mode));
        }
        if mode == VAD_MODE_ML {
            return Err(anyhow!("VAD_DETECT: MODE=ML sem modelo (RFC futura)"));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "VAD_DETECT")?;
        let n = data.len();
        let score = if mode == VAD_MODE_ENERGY {
            let sum_sq: f32 = data.iter().map(|&x| x * x).sum();
            (sum_sq / n as f32).sqrt()
        } else {
            if n < 2 {
                return Err(anyhow!("VAD_DETECT: ZCR precisa de >= 2 amostras"));
            }
            let mut cross = 0usize;
            for w in data.windows(2) {
                if w[0] * w[1] < 0.0 {
                    cross += 1;
                }
            }
            cross as f32 / (n - 1) as f32
        };
        let out_addr = self.memory.alloc_tensor(&[1, 1], crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &[score])?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.vad_detect_execs += 1;
        log_debug("vad", &format!("ctx {} VAD_DETECT 0x{:x}{:?} MODE={} -> {}", ctx_id, t_addr, shape, mode, score));
        Ok(())
    }

    /// STREAM_MERGE rD, rA, rB — out = GAIN*A + (1-GAIN)*B, shapes
    /// idênticas (sem broadcast silencioso). Ganho finito; N_STREAMS
    /// 0=default 2, < 2 veta; modo reservado nonzero veta.
    fn exec_stream_merge(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF || instr.rsrc2 == 0xFF {
            return Err(anyhow!("STREAM_MERGE precisa de rdest, rA, rB (0xFF não é registrador)"));
        }
        let (a_addr, b_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let (gain, n_p, mode) = instr.merge_params();
        if !gain.is_finite() {
            return Err(anyhow!("STREAM_MERGE: GAIN={} não-finito", gain));
        }
        if mode != 0 {
            return Err(anyhow!("STREAM_MERGE: modo {} reservado", mode));
        }
        let n_streams = if n_p == 0 { 2 } else { n_p };
        if n_streams < 2 {
            return Err(anyhow!("STREAM_MERGE: N_STREAMS={} contradiz mix de 2 (mínimo 2)", n_p));
        }
        let (sa, a) = self.read_dense_f32(a_addr, "STREAM_MERGE")?;
        let (sb, b) = self.read_dense_f32(b_addr, "STREAM_MERGE")?;
        if sa != sb {
            return Err(anyhow!("STREAM_MERGE: shapes {:?} vs {:?} (idênticos, sem broadcast)", sa, sb));
        }
        let g = 1.0 - gain;
        let out: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| gain * x + g * y).collect();
        let out_addr = self.memory.alloc_tensor(&sa, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.stream_merge_execs += 1;
        log_debug("merge", &format!("ctx {} STREAM_MERGE 0x{:x}+0x{:x}{:?} GAIN={} -> 0x{:x}", ctx_id, a_addr, b_addr, sa, gain, out_addr));
        Ok(())
    }

    /// AUDIO_RESAMPLE rD, rT — interpolação linear p/ nova taxa.
    /// out_len = floor(in*dst/src) >= 1; cauda clampada (documentado).
    fn exec_audio_resample(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("AUDIO_RESAMPLE precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let (src_rate, dst_rate) = instr.resample_rates();
        if src_rate == 0 || dst_rate == 0 {
            return Err(anyhow!("AUDIO_RESAMPLE: taxas {}->{} inválidas (> 0)", src_rate, dst_rate));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "AUDIO_RESAMPLE")?;
        let n_in = data.len();
        let n_out = (n_in as u64 * dst_rate as u64 / src_rate as u64) as usize;
        if n_out == 0 {
            return Err(anyhow!("AUDIO_RESAMPLE: saída vazia ({} @{}->@{})", n_in, src_rate, dst_rate));
        }
        let mut out = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let pos = i as f32 * src_rate as f32 / dst_rate as f32;
            // min() blinda erro de arredondamento do f32 (pos < n_in por
            // construção, mas floor() pode devolver n_in no limite).
            let i0 = (pos.floor() as usize).min(n_in - 1);
            let frac = (pos - i0 as f32).clamp(0.0, 1.0);
            let i1 = (i0 + 1).min(n_in - 1);
            out.push(data[i0] * (1.0 - frac) + data[i1] * frac);
        }
        let out_shape = vec![n_out];
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.audio_resample_execs += 1;
        log_debug("resample", &format!("ctx {} RESAMPLE 0x{:x}[{}] @{}->@{} [{}] -> 0x{:x}", ctx_id, t_addr, n_in, src_rate, dst_rate, n_out, out_addr));
        Ok(())
    }

    /// AUDIO_FILTER rD, rX, rB — FIR same-size centrado, zero-pad nas
    /// bordas: tap j alinha em x[n-(j-(M-1)/2)] (divisão inteira).
    fn exec_audio_filter(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF || instr.rsrc2 == 0xFF {
            return Err(anyhow!("AUDIO_FILTER precisa de rdest, rX, rB (0xFF não é registrador)"));
        }
        let (x_addr, b_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let mode = instr.filter_mode();
        if mode != FILTER_MODE_FIR {
            return Err(anyhow!("AUDIO_FILTER: MODE={} sem implementação (só FIR; IIR precisa projeto de coefs A)", mode));
        }
        let (sx, x) = self.read_dense_f32(x_addr, "AUDIO_FILTER")?;
        let (_, b) = self.read_dense_f32(b_addr, "AUDIO_FILTER")?;
        let m = b.len();
        if m == 0 {
            return Err(anyhow!("AUDIO_FILTER: taps vazios"));
        }
        let n = x.len();
        let off = (m - 1) / 2;
        let mut out = vec![0.0f32; n];
        for nn in 0..n {
            let mut acc = 0.0f32;
            for (j, &bj) in b.iter().enumerate() {
                let k = nn as isize - (j as isize - off as isize);
                if k >= 0 && (k as usize) < n {
                    acc += bj * x[k as usize];
                }
            }
            out[nn] = acc;
        }
        let out_addr = self.memory.alloc_tensor(&sx, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.audio_filter_execs += 1;
        log_debug("filter", &format!("ctx {} FILTER FIR M={} 0x{:x}{:?} -> 0x{:x}", ctx_id, m, x_addr, sx, out_addr));
        Ok(())
    }

    /// AUDIO_WINDOW rD, rT — Hann/Hamming periódica flat (N = numel).
    /// N=1 dá 0.0 (fórmula honesta, documentado).
    fn exec_audio_window(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF {
            return Err(anyhow!("AUDIO_WINDOW precisa de rdest e rTensor (0xFF não é registrador)"));
        }
        let t_addr = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            ctx.reg(instr.rsrc1)?
        };
        let wtype = instr.window_type();
        if wtype != WINDOW_HANN && wtype != WINDOW_HAMMING {
            return Err(anyhow!("AUDIO_WINDOW: TYPE={} inválido (0=HANN,1=HAMMING)", wtype));
        }
        let (shape, data) = self.read_dense_f32(t_addr, "AUDIO_WINDOW")?;
        let n = data.len();
        let out: Vec<f32> = data
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let c = (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
                let w = if wtype == WINDOW_HANN { 0.5 * (1.0 - c) } else { 0.54 - 0.46 * c };
                x * w
            })
            .collect();
        let out_addr = self.memory.alloc_tensor(&shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.audio_window_execs += 1;
        log_debug("window", &format!("ctx {} WINDOW 0x{:x}{:?} TYPE={} -> 0x{:x}", ctx_id, t_addr, shape, wtype, out_addr));
        Ok(())
    }

    /// DEPFORMER rD, rX, rW — um passo de camada depformer (RFC-0032):
    /// proj Q/K/V/O + attn causal sobre KV deslizante (sem RoPE,
    /// paridade moshi) + amostragem por codebook. Pesos via tabela de
    /// 5 addrs u64 (+3 reservados zerados). Saída pack [Y|codes]
    /// (fatiar com SLICE). Sem residual/norma/FFN (programa compõe).
    fn exec_depformer(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rdest == 0xFF || instr.rsrc1 == 0xFF || instr.rsrc2 == 0xFF {
            return Err(anyhow!("DEPFORMER precisa de rdest, rX, rW (0xFF não é registrador)"));
        }
        let (x_addr, w_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let (stream, layer, ncb_p, nheads_p, q_p, ctx_p, temp, topk_p) = instr.depformer_params();
        let (ncb, nheads, qlevels, context) = (
            if ncb_p == 0 { crate::depformer::DEPFORMER_DEFAULT_NCB } else { ncb_p as usize },
            if nheads_p == 0 { crate::depformer::DEPFORMER_DEFAULT_NHEADS } else { nheads_p as usize },
            if q_p == 0 { crate::depformer::DEPFORMER_DEFAULT_LEVELS } else { q_p as usize },
            if ctx_p == 0 { crate::depformer::DEPFORMER_DEFAULT_CONTEXT } else { ctx_p as usize },
        );
        if ncb == 0 || nheads == 0 || qlevels == 0 || context == 0 {
            return Err(anyhow!("DEPFORMER: dims inválidas (NCB/NHEADS/LEVELS/CONTEXT >= 1)"));
        }
        if !temp.is_finite() {
            return Err(anyhow!("DEPFORMER: TEMP não-finito"));
        }
        if topk_p != 0 && temp <= 0.0 {
            return Err(anyhow!("DEPFORMER: TEMP={} precisa ser > 0 p/ amostragem", temp));
        }
        // Entrada: denso F32, D = numel.
        let x_meta = self.memory.get_tensor_meta(x_addr).cloned()
            .ok_or_else(|| anyhow!("DEPFORMER: rX 0x{:x} não é tensor", x_addr))?;
        if x_meta.is_sparse {
            return Err(anyhow!("DEPFORMER: rX esparso (denso nesta RFC)"));
        }
        if x_meta.dtype != crate::memory::DType::F32 {
            return Err(anyhow!("DEPFORMER: rX dtype {:?} (só F32)", x_meta.dtype));
        }
        let d: usize = x_meta.shape.iter().product();
        if d == 0 {
            return Err(anyhow!("DEPFORMER: rX vazio"));
        }
        if d % nheads != 0 {
            return Err(anyhow!("DEPFORMER: D={} não divisível por NHEADS={}", d, nheads));
        }
        // Tabela: 8×u64 LE ([Wq,Wk,Wv,Wo,Wh,0,0,0]); cada peso [D,D]
        // denso F32, Whead [D,NCB*Q].
        let w_bytes = self.memory.read(w_addr, 64).map_err(|_| anyhow!("DEPFORMER: tabela rW 0x{:x} precisa de 64 bytes (8×u64)", w_addr))?;
        let mut waddrs = [0u64; 8];
        for (i, w) in waddrs.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&w_bytes[i * 8..(i + 1) * 8]);
            *w = u64::from_le_bytes(b);
        }
        for (i, r) in waddrs[5..8].iter().enumerate() {
            if *r != 0 {
                return Err(anyhow!("DEPFORMER: reservado[{}] != 0 (forward-compat alto)", i));
            }
        }
        let qw = d.checked_mul(d).ok_or_else(|| anyhow!("DEPFORMER: D*D estoura"))?;
        let hw = d.checked_mul(ncb).and_then(|x| x.checked_mul(qlevels))
            .ok_or_else(|| anyhow!("DEPFORMER: D*NCB*Q estoura"))?;
        let names = ["Wq", "Wk", "Wv", "Wo"];
        let mut wmats: Vec<Vec<f32>> = Vec::with_capacity(5);
        for (wi, waddr) in waddrs[..4].iter().enumerate() {
            let wa = *waddr as u128;
            let meta = self.memory.get_tensor_meta(wa).cloned()
                .ok_or_else(|| anyhow!("DEPFORMER: {} 0x{:x} não é tensor", names[wi], wa))?;
            if meta.is_sparse || meta.dtype != crate::memory::DType::F32 {
                return Err(anyhow!("DEPFORMER: {} dtype/esparso inválido", names[wi]));
            }
            if meta.byte_len != qw.saturating_mul(4) {
                return Err(anyhow!("DEPFORMER: {} com shape {:?} (esperado [{},{}])", names[wi], meta.shape, d, d));
            }
            wmats.push(self.memory.read_f32_tensor(wa, qw)?);
        }
        let wh_addr = waddrs[4] as u128;
        let wh_meta = self.memory.get_tensor_meta(wh_addr).cloned()
            .ok_or_else(|| anyhow!("DEPFORMER: Whead 0x{:x} não é tensor", wh_addr))?;
        if wh_meta.is_sparse || wh_meta.dtype != crate::memory::DType::F32 {
            return Err(anyhow!("DEPFORMER: Whead dtype/esparso inválido"));
        }
        if wh_meta.byte_len != hw.saturating_mul(4) {
            return Err(anyhow!("DEPFORMER: Whead com {} bytes (esperado {} = D*NCB*Q*4)", wh_meta.byte_len, hw * 4));
        }
        let whead = self.memory.read_f32_tensor(wh_addr, hw)?;
        let x = self.memory.read_f32_tensor(x_addr, d)?;
        // Q/K/V = x @ W (GEMV linha; laço escalar como FLASH_ATTN).
        let matvec = |m: &[f32], v: &[f32]| -> Vec<f32> {
            let mut y = vec![0.0f32; d];
            for (j, yj) in y.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for (i, &xi) in v.iter().enumerate() {
                    s += xi * m[i * d + j];
                }
                *yj = s;
            }
            y
        };
        let qv = matvec(&wmats[0], &x);
        let kv = matvec(&wmats[1], &x);
        let vv = matvec(&wmats[2], &x);
        // KV deslizante por (stream, layer).
        let key = (stream, layer);
        let entry = self.dep_kv.entry(key).or_insert_with(|| crate::depformer::DepKV::new(d));
        if entry.d != d {
            return Err(anyhow!("DEPFORMER: camada (s={},l={}) reconfigurada D={} (era {})", stream, layer, d, entry.d));
        }
        entry.push(&kv, &vv, context).map_err(|e| anyhow!("DEPFORMER: {}", e))?;
        let rows = entry.rows();
        let attn = crate::depformer::dep_attention(&qv, &entry.k, &entry.v, d, nheads, rows);
        // Y = attn @ Wo (sem residual — programa compõe).
        let y = matvec(&wmats[3], &attn);
        // Códigos: logits_c = Y @ Whead[:, c*Q..]; argmax (TOPK=0,
        // determinístico) ou top-k + softmax + RNG (SAMPLE).
        let topk = topk_p as usize;
        if topk > qlevels {
            return Err(anyhow!("DEPFORMER: TOPK={} > LEVELS={} (sem clamp)", topk, qlevels));
        }
        let mut codes = Vec::with_capacity(ncb);
        for c in 0..ncb {
            let logits: Vec<f32> = (0..qlevels)
                .map(|q| {
                    let mut s = 0.0f32;
                    for (i, &yi) in y.iter().enumerate() {
                        s += yi * whead[i * ncb * qlevels + c * qlevels + q];
                    }
                    s
                })
                .collect();
            let code = if topk == 0 {
                // Argmax determinístico (menor índice no empate).
                let mut bi = 0usize;
                let mut bv = logits[0];
                for (i, &v) in logits.iter().enumerate().skip(1) {
                    if v > bv {
                        bv = v;
                        bi = i;
                    }
                }
                bi as u32
            } else {
                // Top-k + softmax + RNG com seed (padrão SAMPLE).
                let mut order: Vec<usize> = (0..qlevels).collect();
                order.sort_by(|&a, &b| {
                    logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
                });
                order.truncate(topk);
                let mut probs: Vec<f32> = order.iter().map(|&i| logits[i]).collect();
                let mx = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for p in probs.iter_mut() {
                    *p = ((*p - mx) / temp).exp();
                    sum += *p;
                }
                for p in probs.iter_mut() {
                    *p /= sum;
                }
                let draw = self.sample_logits_ctx(ctx_id, &probs)?;
                order[draw as usize % order.len()] as u32
            };
            codes.push(code);
        }
        // Pack [Y | codes] (precedente DISTANCE; fatiar com SLICE).
        let mut packed = Vec::with_capacity(d + ncb);
        packed.extend_from_slice(&y);
        packed.extend(codes.iter().map(|&c| c as f32));
        let out_shape = vec![1, d + ncb];
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &packed)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.depformer_execs += 1;
        log_debug("depformer", &format!("ctx {} DEPFORMER s={} l={} D={} NCB={} rows={} -> 0x{:x}", ctx_id, stream, layer, d, ncb, rows, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0034: sub-rotinas (0x7D/0x7E). Pilha por contexto; FORK clona,
    // ABORT descarta com o contexto; fora de versionamento (código imutável).
    // -----------------------------------------------------------------------

    /// CALL LABEL — empilha pc+32 (a CALL é sempre 32B) e pula p/ alvo
    /// (validado como JUMP). Teto `MAX_CALL_DEPTH`, alto em vez de estourar.
    fn exec_call(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let target = instr.imm_u128();
        self.check_jump_target(target)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if ctx.call_stack.len() >= crate::context::MAX_CALL_DEPTH {
                return Err(anyhow!("CALL: pilha cheia ({} frames)", crate::context::MAX_CALL_DEPTH));
            }
            let ret = ctx.pc.wrapping_add(32);
            ctx.call_stack.push(ret);
            ctx.pc = target;
        }
        self.stats.call_execs += 1;
        log_debug("call", &format!("ctx {} CALL -> {:032x}", ctx_id, target));
        Ok(())
    }

    /// RET — desempilha retorno p/ pc. Pilha vazia veta alto (nunca cai
    /// em bytes aleatórios).
    fn exec_ret(&mut self, ctx_id: u64, _instr: &Instruction) -> Result<()> {
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            match ctx.call_stack.pop() {
                Some(ret) => ctx.pc = ret,
                None => return Err(anyhow!("RET sem CALL (pilha vazia)")),
            }
        }
        self.stats.ret_execs += 1;
        log_debug("ret", &format!("ctx {} RET", ctx_id));
        Ok(())
    }

    /// Reacorda (Ready + enqueue) os ctxs à espera da barreira `id`,
    /// ignorando os que morreram no meio (ABORT remove do scheduler, mas
    /// não desta lista — guarda `is_some`). O arrivante NÃO é tocado: o
    /// loop de run avança o PC e re-enfileira quem segue Running; mexer
    /// aqui duplicaria presença na fila e pularia o avanço de PC.
    fn barrier_wake(&mut self, id: u32) {
        if let Some(waiters) = self.barrier_waiters.remove(&id) {
            for w in waiters {
                if let Some(ctx) = self.scheduler.get_mut(w) {
                    ctx.state = crate::context::ContextState::Ready;
                    self.scheduler.enqueue(w);
                }
            }
        }
    }

    /// BARRIER id/expected/timeout/epoch — one-shot local. Chegada
    /// junta-ou-cria; expected atingido => RELEASE (reacorda + apaga);
    /// timeout (0=sem) verificado preguiçosamente na chegada (NACK: o
    /// arrivante morre com Err, os demais são liberados); geração
    /// divergente (expected/epoch) => Err sem tocar em nada.
    fn exec_barrier(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (id, expected, timeout_ms, epoch) = instr.barrier_params();
        if expected == 0 {
            return Err(anyhow!("BARRIER: EXPECT 0 (barreira vazia é bug do chamador)"));
        }
        let now = crate::utils::now_ns();
        let arrived_new = match self.barriers.get(&id).copied() {
            None => {
                if expected <= 1 {
                    // EXPECT=1: libera na hora, sem bloquear nem registrar.
                    self.stats.barrier_execs += 1;
                    log_debug("cluster", &format!("ctx {} BARRIER id={} RELEASE-imediato (1/1)", ctx_id, id));
                    return Ok(());
                }
                let deadline = if timeout_ms == 0 {
                    u64::MAX
                } else {
                    now.saturating_add(timeout_ms as u64 * 1_000_000)
                };
                self.barriers.insert(id, BarrierState { expected, arrived: 1, deadline_ns: deadline, epoch });
                1u32
            }
            Some(st) => {
                if st.expected != expected || st.epoch != epoch {
                    return Err(anyhow!(
                        "BARRIER id={}: geração divergente (have expect={} epoch={}, want expect={} epoch={})",
                        id, st.expected, st.epoch, expected, epoch
                    ));
                }
                if now > st.deadline_ns {
                    // Timeout: apaga, libera quem esperava; o arrivante
                    // recebe NACK como Err (morre — documentado na RFC).
                    self.barriers.remove(&id);
                    self.barrier_wake(id);
                    return Err(anyhow!("BARRIER id={}: timeout (NACK)", id));
                }
                let n = st.arrived + 1;
                if n >= st.expected as u32 {
                    self.barriers.remove(&id);
                    self.barrier_wake(id);
                    self.stats.barrier_execs += 1;
                    log_debug("cluster", &format!("ctx {} BARRIER id={} RELEASE ({}/{})", ctx_id, id, n, expected));
                    return Ok(());
                }
                if let Some(stm) = self.barriers.get_mut(&id) {
                    stm.arrived = n;
                }
                n
            }
        };
        // Ainda faltam participantes: registra espera (sem duplicar o mesmo
        // ctx — re-step manual não deve dar fatia dobrada) e bloqueia (o
        // scheduler pula Blocked; o loop NÃO re-enfileira: estado != Running).
        {
            let waiters = self.barrier_waiters.entry(id).or_default();
            if !waiters.contains(&ctx_id) {
                waiters.push(ctx_id);
            }
        }
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.state = crate::context::ContextState::Blocked;
        }
        self.stats.barrier_execs += 1;
        log_debug("cluster", &format!("ctx {} BARRIER id={} WAIT ({}/{})", ctx_id, id, arrived_new, expected));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0013: DENOISE_STEP (0x21). Stateless por passo (função pura):
    // estado multi-passo viaja em tensores (snapshots existentes cobrem).
    // -----------------------------------------------------------------------

    /// DENOISE_STEP rD, rX, rE — x_{t-1} = (x_t - coef*eps)/sqrt(α) + σz.
    /// DDPM ancestral (Ho et al. 2020); σ=0 determinístico e NÃO consome RNG.
    /// rsrc3 != 0xFF: Err explícito (schedule futuro). Resultado não-finito:
    /// trap (fail-closed, mesma regra de RANK1/FOREST).
    fn exec_denoise_step(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        if instr.rsrc3 != 0xFF {
            return Err(anyhow!("DENOISE_STEP: rsrc3 com tensor de schedule não suportado (use 0xFF)"));
        }
        let (x_addr, e_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let (alpha_bar, beta, sigma, timestep) = instr.denoise_params();
        if !(alpha_bar.is_finite() && alpha_bar > 0.0 && alpha_bar <= 1.0) {
            return Err(anyhow!("DENOISE_STEP: ALPHA_BAR {} inválido (0,1]", alpha_bar));
        }
        if !(beta.is_finite() && beta >= 0.0 && beta < 1.0) {
            return Err(anyhow!("DENOISE_STEP: BETA {} inválido [0,1)", beta));
        }
        if !(sigma.is_finite() && sigma >= 0.0) {
            return Err(anyhow!("DENOISE_STEP: SIGMA {} inválido (>=0)", sigma));
        }
        let alpha = 1.0 - beta;
        if alpha <= 0.0 {
            return Err(anyhow!("DENOISE_STEP: ALPHA derivado {} inválido", alpha));
        }
        let xmeta = self.memory.get_tensor_meta(x_addr).cloned()
            .ok_or_else(|| anyhow!("DENOISE_STEP: x 0x{:x} não encontrado", x_addr))?;
        let emeta = self.memory.get_tensor_meta(e_addr).cloned()
            .ok_or_else(|| anyhow!("DENOISE_STEP: eps 0x{:x} não encontrado", e_addr))?;
        if xmeta.is_sparse || emeta.is_sparse {
            return Err(anyhow!("DENOISE_STEP: esparso não suportado"));
        }
        if xmeta.shape != emeta.shape {
            return Err(anyhow!("DENOISE_STEP: shapes {:?} != {:?}", xmeta.shape, emeta.shape));
        }
        let n: usize = xmeta.shape.iter().product();
        if n == 0 {
            return Err(anyhow!("DENOISE_STEP: tensor vazio"));
        }
        let x = self.memory.read_f32_tensor(x_addr, n)?;
        let e = self.memory.read_f32_tensor(e_addr, n)?;
        let one_minus_bar = 1.0 - alpha_bar;
        if one_minus_bar <= 0.0 && e.iter().any(|v| *v != 0.0) {
            return Err(anyhow!("DENOISE_STEP: ALPHA_BAR=1 com eps não-nulo (divisão por zero)"));
        }
        let coef = if one_minus_bar <= 0.0 { 0.0 } else { beta / one_minus_bar.sqrt() };
        let inv_sqrt_alpha = 1.0 / alpha.sqrt();
        // Draw estocástico SOMENTE se sigma > 0 (sigma=0 não toca no RNG).
        let mut noise = vec![0.0f32; n];
        if sigma > 0.0 {
            let mut st = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?.rng_state;
            for z in noise.iter_mut() {
                *z = crate::determinism::normal_f32(&mut st, 0.0, 1.0);
            }
            if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
                ctx.rng_state = st;
            }
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push((x[i] - coef * e[i]) * inv_sqrt_alpha + sigma * noise[i]);
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err(anyhow!("DENOISE_STEP: resultado não-finito (t={})", timestep));
        }
        let out_addr = self.memory.alloc_tensor(&xmeta.shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.denoise_execs += 1;
        log_debug("denoise", &format!("ctx {} DENOISE_STEP t={} sigma={} -> 0x{:x}", ctx_id, timestep, sigma, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0014: ODE_STEP (0x25). Stateless por passo; trajetórias em tensores.
    // -----------------------------------------------------------------------

    /// Campo f(x,u) = SILU(W·x + b + û), û = u com pad/truncate p/ n.
    fn ode_field(x: &[f32], u: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut s = b[i];
            for j in 0..n {
                s += w[i * n + j] * x[j];
            }
            if i < u.len() {
                s += u[i];
            }
            // SiLU: x·sigmoid(x), estável p/ |x| grande via ramo.
            let g = if s >= 0.0 {
                s / (1.0 + (-s).exp())
            } else {
                let e = s.exp();
                s * e / (1.0 + e)
            };
            out.push(g);
        }
        out
    }

    /// ODE_STEP rD, rX, rU [, rWb] — x(t+dt) Euler/RK2/RK4.
    /// rU=0xFF => sem controle; rWb=0xFF => campo default contrativo
    /// (W=-0.1·I, b=0), documentado. Resultado não-finito: trap.
    fn exec_ode_step(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{ODE_METHOD_EULER, ODE_METHOD_RK2, ODE_METHOD_RK4};
        let (x_addr, u_addr, w_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            let xa = ctx.reg(instr.rsrc1)?;
            let ua = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? } else { 0xFF };
            let wa = if instr.rsrc3 != 0xFF { ctx.reg(instr.rsrc3)? } else { 0xFF };
            (xa, ua, wa)
        };
        let (dt, method, _layer) = instr.ode_params();
        if !(dt.is_finite() && dt > 0.0) {
            return Err(anyhow!("ODE_STEP: DT {} inválido (>0)", dt));
        }
        if method != ODE_METHOD_EULER && method != ODE_METHOD_RK2 && method != ODE_METHOD_RK4 {
            return Err(anyhow!("ODE_STEP: METHOD {} inválido (0/1/2)", method));
        }
        let xmeta = self.memory.get_tensor_meta(x_addr).cloned()
            .ok_or_else(|| anyhow!("ODE_STEP: x 0x{:x} não encontrado", x_addr))?;
        if xmeta.is_sparse {
            return Err(anyhow!("ODE_STEP: esparso não suportado"));
        }
        let n: usize = xmeta.shape.iter().product();
        if n == 0 {
            return Err(anyhow!("ODE_STEP: estado vazio"));
        }
        let x = self.memory.read_f32_tensor(x_addr, n)?;
        let u: Vec<f32> = if u_addr != 0xFF {
            let um = self.memory.get_tensor_meta(u_addr).cloned()
                .ok_or_else(|| anyhow!("ODE_STEP: u 0x{:x} não encontrado", u_addr))?;
            if um.is_sparse {
                return Err(anyhow!("ODE_STEP: controle esparso não suportado"));
            }
            let m: usize = um.shape.iter().product();
            self.memory.read_f32_tensor(u_addr, m)?
        } else {
            Vec::new()
        };
        let (w, b): (Vec<f32>, Vec<f32>) = if w_addr != 0xFF {
            let wm = self.memory.get_tensor_meta(w_addr).cloned()
                .ok_or_else(|| anyhow!("ODE_STEP: pack Wb 0x{:x} não encontrado", w_addr))?;
            if wm.is_sparse {
                return Err(anyhow!("ODE_STEP: pack esparso não suportado"));
            }
            let flat: usize = wm.shape.iter().product();
            if flat != n * (n + 1) {
                return Err(anyhow!("ODE_STEP: pack com {} elems, esperado {} ([n,n+1])", flat, n * (n + 1)));
            }
            let pack = self.memory.read_f32_tensor(w_addr, flat)?;
            (pack[..n * n].to_vec(), pack[n * n..].to_vec())
        } else {
            // Default contrativo: W=-0.1·I, b=0.
            let mut w0 = vec![0.0f32; n * n];
            for i in 0..n {
                w0[i * n + i] = -0.1;
            }
            (w0, vec![0.0f32; n])
        };
        let f = |s: &[f32]| Self::ode_field(s, &u, &w, &b);
        let fx = f(&x);
        let out: Vec<f32> = match method {
            ODE_METHOD_EULER => x.iter().zip(fx.iter()).map(|(a, k)| a + dt * k).collect(),
            ODE_METHOD_RK2 => {
                let mid: Vec<f32> = x.iter().zip(fx.iter()).map(|(a, k)| a + dt * 0.5 * k).collect();
                let k2 = f(&mid);
                x.iter().zip(k2.iter()).map(|(a, k)| a + dt * k).collect()
            }
            _ => {
                let k1 = fx;
                let s2: Vec<f32> = x.iter().zip(k1.iter()).map(|(a, k)| a + dt * 0.5 * k).collect();
                let k2 = f(&s2);
                let s3: Vec<f32> = x.iter().zip(k2.iter()).map(|(a, k)| a + dt * 0.5 * k).collect();
                let k3 = f(&s3);
                let s4: Vec<f32> = x.iter().zip(k3.iter()).map(|(a, k)| a + dt * k).collect();
                let k4 = f(&s4);
                x.iter()
                    .zip(k1.iter())
                    .zip(k2.iter())
                    .zip(k3.iter())
                    .zip(k4.iter())
                    .map(|((((a, k1), k2), k3), k4)| a + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
                    .collect()
            }
        };
        if !out.iter().all(|v| v.is_finite()) {
            return Err(anyhow!("ODE_STEP: resultado não-finito (dt={}, method={})", dt, method));
        }
        let out_addr = self.memory.alloc_tensor(&xmeta.shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.ode_execs += 1;
        log_debug("ode", &format!("ctx {} ODE_STEP n={} dt={} m={} -> 0x{:x}", ctx_id, n, dt, method, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0015: SPIKE_STEP (0x20). LIF discreto; estado em tensores CoW
    // (V + refr), handles por camada com push/pop como rank1_layers.
    // -----------------------------------------------------------------------

    /// SPIKE_STEP rD, rV, rI [, rPack] — integrate-and-fire discreto.
    /// rsrc1 (V) religado ao novo tensor (CoW); rdest = spikes [1,n] 0/1.
    /// Pack [thresh,decay,reset,refr] vence payload. Refrão em countdown f32.
    fn exec_spike_step(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (v_addr, i_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let (mut thresh, mut decay, mut reset, layer, mut refr_steps) = instr.spike_params();
        if instr.rsrc3 != 0xFF {
            let p_addr = {
                let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
                ctx.reg(instr.rsrc3)?
            };
            let pmeta = self.memory.get_tensor_meta(p_addr).cloned()
                .ok_or_else(|| anyhow!("SPIKE_STEP: pack 0x{:x} não encontrado", p_addr))?;
            if pmeta.is_sparse {
                return Err(anyhow!("SPIKE_STEP: pack esparso não suportado"));
            }
            let pflat: usize = pmeta.shape.iter().product();
            if pflat != 4 {
                return Err(anyhow!("SPIKE_STEP: pack com {} elems, esperado 4 [thresh,decay,reset,refr]", pflat));
            }
            let p = self.memory.read_f32_tensor(p_addr, 4)?;
            thresh = p[0];
            decay = p[1];
            reset = p[2];
            if !(p[3].is_finite() && p[3] >= 0.0 && p[3] <= 255.0 && p[3].fract() == 0.0) {
                return Err(anyhow!("SPIKE_STEP: refr_steps {} inválido (0-255 integral)", p[3]));
            }
            refr_steps = p[3] as u8;
        }
        if !thresh.is_finite() {
            return Err(anyhow!("SPIKE_STEP: THRESH não-finito"));
        }
        if !(decay.is_finite() && (0.0..=1.0).contains(&decay)) {
            return Err(anyhow!("SPIKE_STEP: DECAY {} inválido [0,1]", decay));
        }
        if !reset.is_finite() {
            return Err(anyhow!("SPIKE_STEP: RESET não-finito"));
        }
        let vmeta = self.memory.get_tensor_meta(v_addr).cloned()
            .ok_or_else(|| anyhow!("SPIKE_STEP: V 0x{:x} não encontrado", v_addr))?;
        let imeta = self.memory.get_tensor_meta(i_addr).cloned()
            .ok_or_else(|| anyhow!("SPIKE_STEP: I 0x{:x} não encontrado", i_addr))?;
        if vmeta.is_sparse || imeta.is_sparse {
            return Err(anyhow!("SPIKE_STEP: esparso não suportado"));
        }
        let n: usize = vmeta.shape.iter().product();
        let m: usize = imeta.shape.iter().product();
        if n == 0 || m == 0 {
            return Err(anyhow!("SPIKE_STEP: tensor vazio"));
        }
        if m != n {
            return Err(anyhow!("SPIKE_STEP: V com {} elems != I com {} (shapes {:?} vs {:?})", n, m, vmeta.shape, imeta.shape));
        }
        let v = self.memory.read_f32_tensor(v_addr, n)?;
        let inp = self.memory.read_f32_tensor(i_addr, n)?;
        if !v.iter().chain(inp.iter()).all(|z| z.is_finite()) {
            return Err(anyhow!("SPIKE_STEP: V/I não-finitos na entrada"));
        }
        // Refrão corrente: mapa da camada, ou zeros na 1ª execução.
        let refr_cur: Vec<f32> = match self.snn_layers.get(&layer) {
            Some((_, r_addr)) => {
                let rm = self.memory.get_tensor_meta(*r_addr).cloned()
                    .ok_or_else(|| anyhow!("SPIKE_STEP: refr da camada {} perdido", layer))?;
                let rn: usize = rm.shape.iter().product();
                if rn != n {
                    return Err(anyhow!("SPIKE_STEP: refr da camada {} com {} elems, esperado {}", layer, rn, n));
                }
                self.memory.read_f32_tensor(*r_addr, n)?
            }
            None => vec![0.0f32; n],
        };
        let mut v_new = Vec::with_capacity(n);
        let mut r_new = Vec::with_capacity(n);
        let mut spikes = Vec::with_capacity(n);
        for i in 0..n {
            let mut vv = (v[i] + inp[i]) * decay;
            let mut rr = if refr_cur[i] > 0.0 { refr_cur[i] - 1.0 } else { 0.0 };
            let fired = refr_cur[i] <= 0.0 && vv >= thresh;
            if fired {
                vv = reset;
                rr = refr_steps as f32;
                spikes.push(1.0);
            } else {
                spikes.push(0.0);
            }
            v_new.push(vv);
            r_new.push(rr);
        }
        if !v_new.iter().all(|z| z.is_finite()) {
            return Err(anyhow!("SPIKE_STEP: V resultante não-finito (layer {})", layer));
        }
        let shape1 = vec![1, n];
        let v_addr_new = self.memory.alloc_tensor(&shape1, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(v_addr_new, &v_new)?;
        let r_addr_new = self.memory.alloc_tensor(&shape1, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(r_addr_new, &r_new)?;
        let s_addr = self.memory.alloc_tensor(&shape1, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(s_addr, &spikes)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.set_reg(instr.rsrc1, v_addr_new)?;
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, s_addr)?;
            }
        }
        self.snn_layers.insert(layer, (v_addr_new, r_addr_new));
        self.stats.spike_execs += 1;
        log_debug("spike", &format!("ctx {} SPIKE_STEP layer={} n={} spikes={} -> 0x{:x}", ctx_id, layer, n,
            spikes.iter().filter(|&&s| s == 1.0).count(), s_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0017: CONV (0x1E). Deslizamento direto 1D/2D (sem im2col),
    // groups/depthwise, ativação fundida. Stateless (Family 1).
    // -----------------------------------------------------------------------

    fn conv_activate(x: f32, act: u8) -> f32 {
        use crate::opcodes::{CONV_ACT_RELU, CONV_ACT_SILU};
        if act == CONV_ACT_SILU {
            if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) }
        } else if act == CONV_ACT_RELU {
            x.max(0.0)
        } else {
            x
        }
    }

    /// CONV rD, rX, rW [, rB] — cross-correlação (kernel NÃO flipado).
    /// Adaptação de rank: [H,W]=>[1,1,H,W]; [C,L]=>[1,C,L]; [N,C,…] direto
    /// (só 1D/2D); kernel cheio [Cout,Cin,K…] ou reduzido [K]/[KH,KW]
    /// (single). Geometria validada antes de alocar.
    fn exec_conv(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{CONV_ACT_NONE, CONV_ACT_RELU, CONV_ACT_SILU};
        let (x_addr, w_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?)
        };
        let b_addr = if instr.rsrc3 != 0xFF {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            Some(ctx.reg(instr.rsrc3)?)
        } else {
            None
        };
        let (stride, pad, dilation, groups, act) = instr.conv_params();
        if act != CONV_ACT_NONE && act != CONV_ACT_SILU && act != CONV_ACT_RELU {
            return Err(anyhow!("CONV: ACT {} inválido (0/1/2)", act));
        }
        if stride == 0 || dilation == 0 {
            return Err(anyhow!("CONV: STRIDE/DILATION devem ser >= 1"));
        }
        if groups == 0 {
            return Err(anyhow!("CONV: GROUPS inválido"));
        }
        let xm = self.memory.get_tensor_meta(x_addr).cloned()
            .ok_or_else(|| anyhow!("CONV: input 0x{:x} não encontrado", x_addr))?;
        let wm = self.memory.get_tensor_meta(w_addr).cloned()
            .ok_or_else(|| anyhow!("CONV: kernel 0x{:x} não encontrado", w_addr))?;
        if xm.is_sparse || wm.is_sparse {
            return Err(anyhow!("CONV: esparso não suportado"));
        }
        // Adaptação de rank do input.
        let (batch, cin, spatial): (usize, usize, Vec<usize>) = match xm.shape.len() {
            2 => (1, 1, vec![xm.shape[0], xm.shape[1]]),
            3 => (xm.shape[0], xm.shape[1], xm.shape[2..].to_vec()),
            4 => (xm.shape[0], xm.shape[1], xm.shape[2..].to_vec()),
            r => return Err(anyhow!("CONV: input rank {} (só 1D/2D: [H,W]/[N,C,L]/[N,C,..])", r)),
        };
        let ndim = spatial.len();
        // Adaptação de rank do kernel.
        let (cout, kcin, kspatial): (usize, usize, Vec<usize>) = match wm.shape.len() {
            1 if ndim == 1 => (1, 1, vec![wm.shape[0]]),
            2 if ndim == 2 => (1, 1, vec![wm.shape[0], wm.shape[1]]),
            r if r == ndim + 2 => (wm.shape[0], wm.shape[1], wm.shape[2..].to_vec()),
            _ => return Err(anyhow!("CONV: kernel rank {:?} incompatível com input {}D", wm.shape, ndim)),
        };
        let g = groups as usize;
        if g == 0 {
            return Err(anyhow!("CONV: GROUPS inválido"));
        }
        if kcin != cin / g {
            // Layout padrão de groups: kernel [Cout, Cin/G, K...].
            // (G=1 => [Cout, Cin, K...] usual; depthwise => [C,1,K...].)
            return Err(anyhow!("CONV: Cin kernel {} != Cin/Grupos {}/{} (kernel [Cout,Cin/G,K..])", kcin, cin, g));
        }
        if cin % g != 0 || cout % g != 0 {
            return Err(anyhow!("CONV: GROUPS {} não divide Cin={} Cout={}", g, cin, cout));
        }
        // Geometria por eixo (tudo validado antes de alocar).
        let s = stride as usize;
        let p = pad as usize;
        let d = dilation as usize;
        let mut out_spatial = Vec::with_capacity(ndim);
        for (a, (&ii, &kk)) in spatial.iter().zip(kspatial.iter()).enumerate() {
            let eff = d * (kk - 1) + 1;
            if ii + 2 * p < eff {
                return Err(anyhow!("CONV: kernel maior que input no eixo {} (I={} K={} pad={} dil={})", a, ii, kk, p, d));
            }
            out_spatial.push((ii + 2 * p - eff) / s + 1);
        }
        // Bias: ausente ou flat len == Cout.
        let bias: Vec<f32> = match b_addr {
            None => vec![0.0f32; cout],
            Some(ba) => {
                let bm = self.memory.get_tensor_meta(ba).cloned()
                    .ok_or_else(|| anyhow!("CONV: bias 0x{:x} não encontrado", ba))?;
                if bm.is_sparse {
                    return Err(anyhow!("CONV: bias esparso não suportado"));
                }
                let flat: usize = bm.shape.iter().product();
                if flat != cout {
                    return Err(anyhow!("CONV: bias com {} elems, esperado Cout={}", flat, cout));
                }
                self.memory.read_f32_tensor(ba, flat)?
            }
        };
        let xflat = self.memory.read_f32_tensor(x_addr, batch * cin * spatial.iter().product::<usize>())?;
        let wflat = self.memory.read_f32_tensor(w_addr, cout * kcin * kspatial.iter().product::<usize>())?;
        let cin_g = cin / g;
        let cout_g = cout / g;
        let mut out_shape = vec![batch, cout];
        out_shape.extend_from_slice(&out_spatial);
        let out_len: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; out_len];
        if ndim == 1 {
            let (li, lk, lo) = (spatial[0], kspatial[0], out_spatial[0]);
            for n in 0..batch {
                for co in 0..cout {
                    let gz = co / cout_g;
                    for o in 0..lo {
                        let mut acc = bias[co];
                        for ci in 0..cin_g {
                            let c = gz * cin_g + ci;
                            for k in 0..lk {
                                let ii = o as isize * s as isize - p as isize + k as isize * d as isize;
                                if ii < 0 || ii >= li as isize {
                                    continue;
                                }
                                let xv = xflat[(n * cin + c) * li + ii as usize];
                                let wv = wflat[(co * cin_g + ci) * lk + k];
                                acc += xv * wv;
                            }
                        }
                        out[((n * cout + co) * lo) + o] = Self::conv_activate(acc, act);
                    }
                }
            }
        } else {
            let (hi, wi) = (spatial[0], spatial[1]);
            let (hk, wk) = (kspatial[0], kspatial[1]);
            let (ho, wo) = (out_spatial[0], out_spatial[1]);
            for n in 0..batch {
                for co in 0..cout {
                    let gz = co / cout_g;
                    for oh in 0..ho {
                        for ow in 0..wo {
                            let mut acc = bias[co];
                            for ci in 0..cin_g {
                                let c = gz * cin_g + ci;
                                for kh in 0..hk {
                                    let ih = oh as isize * s as isize - p as isize + kh as isize * d as isize;
                                    if ih < 0 || ih >= hi as isize {
                                        continue;
                                    }
                                    for kw in 0..wk {
                                        let iw = ow as isize * s as isize - p as isize + kw as isize * d as isize;
                                        if iw < 0 || iw >= wi as isize {
                                            continue;
                                        }
                                        let xv = xflat[((n * cin + c) * hi + ih as usize) * wi + iw as usize];
                                        let wv = wflat[((co * cin_g + ci) * hk + kh) * wk + kw];
                                        acc += xv * wv;
                                    }
                                }
                            }
                            out[(((n * cout + co) * ho) + oh) * wo + ow] = Self::conv_activate(acc, act);
                        }
                    }
                }
            }
        }
        if !out.iter().all(|v| v.is_finite()) {
            return Err(anyhow!("CONV: resultado não-finito"));
        }
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.conv_execs += 1;
        log_debug("conv", &format!("ctx {} CONV {}D N={} Cin={} Cout={} G={} act={} -> 0x{:x}", ctx_id, ndim, batch, cin, cout, g, act, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // RFC-0012: FOREST (0x22). Stateless (Family 1): sem snapshot, preempção
    // descarta o chunk. Walk vetorizado com teto max_depth (terminação
    // garantida mesmo p/ tabelas cíclicas).
    // -----------------------------------------------------------------------

    /// FOREST rD, rF, rT, rL — ensemble sobre tabela plana (ver RFC-0012).
    /// Folha = left < 0; valor NaN em folha = Err (fail-closed).
    fn exec_forest(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        use crate::opcodes::{FOREST_MODE_MEAN, FOREST_MODE_VOTE};
        let (f_addr, t_addr, l_addr) = {
            let ctx = self.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx {} não encontrado", ctx_id))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?, ctx.reg(instr.rsrc3)?)
        };
        let (n_trees, max_depth, mode) = instr.forest_params();
        if mode != FOREST_MODE_VOTE && mode != FOREST_MODE_MEAN {
            return Err(anyhow!("FOREST: MODE {} inválido (0/1)", mode));
        }
        if n_trees == 0 || max_depth == 0 || max_depth > 16 {
            return Err(anyhow!("FOREST: TREES/DEPTH fora da faixa"));
        }
        let fmeta = self.memory.get_tensor_meta(f_addr).cloned()
            .ok_or_else(|| anyhow!("FOREST: features 0x{:x} não encontradas", f_addr))?;
        let tmeta = self.memory.get_tensor_meta(t_addr).cloned()
            .ok_or_else(|| anyhow!("FOREST: tabela 0x{:x} não encontrada", t_addr))?;
        let lmeta = self.memory.get_tensor_meta(l_addr).cloned()
            .ok_or_else(|| anyhow!("FOREST: folhas 0x{:x} não encontradas", l_addr))?;
        if fmeta.is_sparse || tmeta.is_sparse || lmeta.is_sparse {
            return Err(anyhow!("FOREST: esparso não suportado"));
        }
        let n_feat: usize = fmeta.shape.iter().product();
        if n_feat == 0 {
            return Err(anyhow!("FOREST: features vazias"));
        }
        let stride = (1usize << max_depth) - 1;
        let need_rows = (n_trees as usize) * stride;
        let t_rows = if tmeta.shape.len() == 2 { tmeta.shape[0] } else { 0 };
        let t_cols = if tmeta.shape.len() == 2 { tmeta.shape[1] } else { 0 };
        if tmeta.shape.len() != 2 || t_cols != 4 || t_rows < need_rows {
            return Err(anyhow!("FOREST: tabela deve ser [>={}, 4], achado {:?}", need_rows, tmeta.shape));
        }
        let l_flat: usize = lmeta.shape.iter().product();
        if l_flat < need_rows {
            return Err(anyhow!("FOREST: folhas com {} elems, esperado >={}", l_flat, need_rows));
        }
        let feats = self.memory.read_f32_tensor(f_addr, n_feat)?;
        let table = self.memory.read_f32_tensor(t_addr, t_rows * 4)?;
        let leaves = self.memory.read_f32_tensor(l_addr, l_flat)?;
        let mut scores = Vec::with_capacity(n_trees as usize);
        for t in 0..(n_trees as usize) {
            let base = t * stride;
            let mut node = 0usize;
            let mut value = leaves[base];
            for _ in 0..max_depth {
                let row = base + node;
                let fi = table[row * 4] as usize;
                // feat_idx deve ser integral, finito e < F.
                if !table[row * 4].is_finite() || table[row * 4].fract() != 0.0 || fi >= n_feat {
                    return Err(anyhow!("FOREST: feat_idx inválido na árvore {} nó {}", t, node));
                }
                let (thresh, left, right) = (table[row * 4 + 1], table[row * 4 + 2], table[row * 4 + 3]);
                if left < 0.0 {
                    value = leaves[base + node];
                    break;
                }
                if !left.is_finite() || !right.is_finite() || left.fract() != 0.0 || right.fract() != 0.0 {
                    return Err(anyhow!("FOREST: filho inválido na árvore {} nó {}", t, node));
                }
                let (l, r) = (left as usize, right as usize);
                if l >= stride || r >= stride {
                    return Err(anyhow!("FOREST: filho OOB na árvore {} nó {} (stride {})", t, node, stride));
                }
                node = if feats[fi] > thresh { r } else { l };
                value = leaves[base + node];
            }
            if !value.is_finite() {
                return Err(anyhow!("FOREST: folha NaN/Inf na árvore {} (use SANITY_CHECK a montante)", t));
            }
            scores.push(value);
        }
        let (out_shape, out): (Vec<usize>, Vec<f32>) = if mode == FOREST_MODE_MEAN {
            let m = scores.iter().sum::<f32>() / scores.len() as f32;
            if !m.is_finite() {
                return Err(anyhow!("FOREST: média não-finita"));
            }
            (vec![1, 1], vec![m])
        } else {
            (vec![1, scores.len()], scores)
        };
        let out_addr = self.memory.alloc_tensor(&out_shape, crate::memory::DType::F32)?;
        self.memory.write_f32_tensor(out_addr, &out)?;
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            if instr.rdest != 0xFF {
                ctx.set_reg(instr.rdest, out_addr)?;
            }
        }
        self.stats.forest_execs += 1;
        log_debug("forest", &format!("ctx {} FOREST trees={} depth={} mode={} -> 0x{:x}", ctx_id, n_trees, max_depth, mode, out_addr));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Benchmark helper
    // -----------------------------------------------------------------------

    /// Roda micro-benchmark de IPS com programa NOP
    pub fn bench_ips(&mut self, n: usize) -> f64 {
        let prog = vec![crate::opcodes::instr_nop(); n];
        self.load_program(prog);
        let start = crate::utils::now_ns();
        let _ = self.run();
        let elapsed = crate::utils::now_ns() - start;
        n as f64 / (elapsed as f64 / 1e9)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcodes::{instr_attn, instr_halt, instr_stream, instr_tensor};

    #[tokio::test]
    async fn test_vm_tensor_attn_stream() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });

        // Programa: TENSOR r0 2x2, TENSOR r1 2x2, TENSOR r2 2x2, ATTN r3 r0 r1 r2, STREAM r3 r0, HALT
        let prog = vec![
            instr_tensor(0, 0xFF, 0xFF, 2, 2, 0),
            instr_tensor(1, 0xFF, 0xFF, 2, 2, 0),
            instr_tensor(2, 0xFF, 0xFF, 2, 2, 0),
            instr_attn(3, 0, 1, 2),
            instr_stream(3, 0, true),
            instr_halt(),
        ];
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert!(stats.tensor_allocs == 3);
        assert!(stats.attn_execs == 1);
        assert!(stats.streams == 1);
    }

    #[tokio::test]
    async fn test_fork_and_abort() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        use crate::opcodes::{instr_abort, instr_fork};
        let prog = vec![
            instr_fork(0, 2), // FORK r0 RED
            instr_abort(0, 0xFF), // ABORT child
            instr_halt(),
        ];
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert!(stats.forks == 1);
    }

    #[tokio::test]
    async fn test_sense() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        use crate::opcodes::instr_sense;
        let prog = vec![
            instr_sense(0, SENSE_AUDIO),
            instr_sense(1, SENSE_VAD),
            instr_halt(),
        ];
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.senses, 2);
    }

    #[tokio::test]
    async fn test_norm_rms() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        use crate::opcodes::{instr_norm, instr_tensor};
        // x 1x4, gamma 1x4 (1.0), beta 1x4 (0.0) — teste simples
        let prog = vec![
            instr_tensor(0, 0xFF, 0xFF, 2, 4, 0), // x
            instr_tensor(1, 0xFF, 0xFF, 1, 4, 0), // gamma
            instr_tensor(2, 0xFF, 0xFF, 1, 4, 0), // beta
            instr_norm(3, 0, 1, 2),
            instr_halt(),
        ];
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.norm_execs, 1);
        let ctx = vm.scheduler.get(1).unwrap();
        let out_addr = ctx.regs[3];
        let out = vm.memory.read_f32_tensor(out_addr, 8).unwrap();
        // Verifica não-nan e que normalização ocorreu (rms ~ sqrt(mean(x^2)))
        for v in &out { assert!(!v.is_nan()); }
        // Para x = [0.5,1.0,1.5,2.0] por linha, rms ~ sqrt((0.25+1+2.25+4)/4)=1.369...
        // out[0] = 0.5/1.369 * gamma ~0.365
        assert!(out[0].abs() < 2.0);
    }

    #[tokio::test]
    async fn test_ffn_swiglu() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        use crate::opcodes::{instr_ffn, instr_tensor};
        // x 2x4, W1 4x8, W2 8x4 -> out 2x4
        let prog = vec![
            instr_tensor(0, 0xFF, 0xFF, 2, 4, 0), // x
            instr_tensor(1, 0xFF, 0xFF, 4, 8, 0), // W1
            instr_tensor(2, 0xFF, 0xFF, 8, 4, 0), // W2
            instr_ffn(3, 0, 1, 2),
            instr_halt(),
        ];
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.ffn_execs, 1);
        let ctx = vm.scheduler.get(1).unwrap();
        let out_addr = ctx.regs[3];
        let out = vm.memory.read_f32_tensor(out_addr, 8).unwrap();
        for v in &out { assert!(!v.is_nan() && v.is_finite()); }
    }

    #[tokio::test]
    async fn test_transformer_block_norm_attn_ffn() {
        // Fluxo completo de uma camada transformer: NORM -> ATTN -> NORM -> FFN
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        use crate::opcodes::{instr_attn, instr_ffn, instr_norm, instr_tensor};
        let prog = vec![
            instr_tensor(0, 0xFF, 0xFF, 2, 4, 0), // hidden
            instr_tensor(1, 0xFF, 0xFF, 1, 4, 0), // gamma
            instr_tensor(2, 0xFF, 0xFF, 1, 4, 0), // beta
            instr_tensor(3, 0xFF, 0xFF, 4, 4, 0), // Q/K/V proj shared size
            instr_tensor(4, 0xFF, 0xFF, 4, 8, 0), // W1
            instr_tensor(5, 0xFF, 0xFF, 8, 4, 0), // W2
            instr_norm(6, 0, 1, 2),              // NORM hidden
            instr_attn(7, 6, 3, 3),              // ATTN (self)
            instr_norm(8, 7, 1, 2),              // NORM att out
            instr_ffn(9, 8, 4, 5),               // FFN
            instr_halt(),
        ];
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.norm_execs, 2);
        assert_eq!(stats.attn_execs, 1);
        assert_eq!(stats.ffn_execs, 1);
    }

    #[tokio::test]
    async fn test_embed_add_sample_compare_jump() {
        // Cadeia EMBED/ADD/SAMPLE + COMPARE/IF_EQUAL/JUMP com rótulos:
        // prova assembler 2-pass, dispatch e pulos condicionais.
        use crate::opcodes::assemble;
        let src = r#"
            TENSOR r0 2 2 f32
            TENSOR r1 2 2 f32
            ADD r2, r0, r1
            SAMPLE r3, r2
            EMBED r4, r7, r0
            COMPARE r2, r2
            IF_EQUAL DO_ADD
            JUMP DONE
        DO_ADD:
            ADD r5, r0, r0
            JUMP DONE
        DONE:
            HALT
        "#;
        let prog = assemble(src).unwrap();
        assert_eq!(prog.len(), 11);
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.embed_execs, 1);
        assert_eq!(stats.add_execs, 2);
        assert_eq!(stats.sample_execs, 1);
        // TENSOR 2x2 inicializa (i+1)*0.5 => [0.5,1,1.5,2]; ADD dobra
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let a2 = ctx.reg(2).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a2, 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        // IF_EQUAL tomado: JUMP DONE pulado, ADD r5 executou, JUMP DONE caiu no HALT
        let a5 = ctx.reg(5).unwrap();
        assert_ne!(a5, 0);
        assert_eq!(vm.memory.read_f32_tensor(a5, 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        // EMBED r4, r7(=0), r0 -> linha 0 da tabela [0.5,1.0] em tensor [1,2]
        let a4 = ctx.reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a4, 2).unwrap(), vec![0.5, 1.0]);
        // SAMPLE guardou token em r3 e last_sample
        let r3 = ctx.reg(3).unwrap();
        assert_eq!(vm.last_sample as u128, r3);
    }

    #[test]
    fn test_mixed_width_fetch_stride() {
        // W1-remainder: programa [NOP32][X(0x84)64B][NOP32].
        // Offsets: [0, 32, 96]; fim em 128. Meio de 64B nunca faz fetch.
        use crate::opcodes::{instr_nop, Instr64, ProgramInstr};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        vm.load_mixed(vec![
            ProgramInstr::W32(instr_nop()),
            ProgramInstr::W64(Instr64::new(0x84, 0xFF, [0xFF; 5])),
            ProgramInstr::W32(instr_nop()),
        ]);
        let base = vm.program_base;
        assert_eq!(vm.pc_offsets, vec![0, 32, 96]);
        assert!(matches!(vm.fetch_at(base), Some(ProgramInstr::W32(_))));
        assert!(matches!(vm.fetch_at(base + 32), Some(ProgramInstr::W64(_))));
        assert!(matches!(vm.fetch_at(base + 96), Some(ProgramInstr::W32(_))));
        // Meio da instrução larga => None (nunca misfetch/misdispatch).
        assert_eq!(vm.fetch_at(base + 64), None);
        assert_eq!(vm.fetch_at(base + 128), None); // além do fim
        // Jumps: inícios OK; meio de 64B e além-do-fim rejeitados.
        assert!(vm.check_jump_target(base + 32).is_ok());
        assert!(vm.check_jump_target(base + 96).is_ok());
        assert!(vm.check_jump_target(base + 64).is_err());
        assert!(vm.check_jump_target(base + 128).is_err());
        // Run: NOP avança 32 (stride real), W64 termina o contexto com
        // mensagem limpa — sem pânico, sem misdispatch. 2 steps.
        let stats = vm.run().unwrap();
        assert_eq!(stats.steps, 2);
        let ctx = vm.scheduler.get(1).unwrap();
        assert_eq!(ctx.pc, base + 32);
        assert_eq!(ctx.state, crate::context::ContextState::Terminated);
    }

    #[tokio::test]
    async fn test_sense_user_input_if_interrupt() {
        use crate::opcodes::assemble;
        let src = r#"
            SENSE r0, USER_INPUT
            IF_INTERRUPT DONE
            TENSOR r1 1 1 f32
        DONE:
            HALT
        "#;
        // Com input: r0=1, IF_INTERRUPT pula o TENSOR (r1 fica 0)
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        vm.push_input("hello".to_string());
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(0).unwrap(), 1);
        assert_eq!(ctx.reg(1).unwrap(), 0);
        assert_eq!(stats.senses, 1);
        assert_eq!(stats.interrupt_checks, 1);
        // Sem input: r0=0, cai no TENSOR (r1 vira endereço válido)
        let prog2 = assemble(src).unwrap();
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        vm2.load_program(prog2);
        let stats2 = vm2.run().unwrap();
        let ctx2 = vm2.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx2.reg(0).unwrap(), 0);
        assert_ne!(ctx2.reg(1).unwrap(), 0);
        assert_eq!(stats2.tensor_allocs, 1);
    }

    #[tokio::test]
    async fn test_ssm_scan_tensor_matches_reference() {
        use crate::opcodes::{instr_ssm_reset, instr_ssm_scan};
        // di=1, ds=1, defaults dt=1 A=-1 B=C=1 D=0: x=1 -> h=1, y=1 (golden ssm.rs).
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let prog = vec![
            instr_tensor(0, 0xFF, 0xFF, 1, 1, 0), // x
            instr_tensor(1, 0xFF, 0xFF, 1, 1, 0), // h
            instr_ssm_scan(2, 0, 1, 0xFF, 1, 1, 0),
            instr_ssm_reset(1, 1, 1, 0),
            instr_halt(),
        ];
        vm.load_program(prog);
        // x=ones? TENSOR init é (i+1)*0.5 => x=[0.5]. h init=[0.5]! Sobrescreve p/ golden.
        let ctx_id = 1;
        // Ajusta x=1.0, h=0.0 antes do scan executando manualmente os 2 primeiros passos:
        // mais simples: roda programa parcial via step e reescreve.
        // Aqui validamos via step_instruction direto para controle total.
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let xa = vm2.memory.alloc_tensor(&[1, 1], crate::memory::DType::F32).unwrap();
        vm2.memory.write_f32_tensor(xa, &[1.0]).unwrap();
        let ha = vm2.memory.alloc_tensor(&[1, 1], crate::memory::DType::F32).unwrap();
        vm2.memory.write_f32_tensor(ha, &[0.0]).unwrap();
        let root = vm2.memory.current_version();
        let cid = vm2.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        vm2.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
        vm2.scheduler.get_mut(cid).unwrap().set_reg(1, ha).unwrap();
        let scan = instr_ssm_scan(2, 0, 1, 0xFF, 1, 1, 0);
        vm2.step_instruction(cid, &scan).unwrap();
        let y_addr = vm2.scheduler.get(cid).unwrap().reg(2).unwrap();
        assert!((vm2.memory.read_f32_tensor(y_addr, 1).unwrap()[0] - 1.0).abs() < 1e-5);
        assert!((vm2.memory.read_f32_tensor(ha, 1).unwrap()[0] - 1.0).abs() < 1e-5);
        // RESET zera h
        let reset = instr_ssm_reset(1, 1, 1, 0);
        vm2.step_instruction(cid, &reset).unwrap();
        assert_eq!(vm2.memory.read_f32_tensor(ha, 1).unwrap()[0], 0.0);
        assert_eq!(vm2.stats.ssm_scans, 1);
        assert_eq!(vm2.stats.ssm_resets, 1);
        // Programa completo roda sem erro (init não-golden, só smoke).
        let stats = vm.run().unwrap();
        assert_eq!(stats.ssm_scans, 1);
        assert_eq!(stats.ssm_resets, 1);
        let _ = ctx_id;
    }

    #[tokio::test]
    async fn test_ssm_scan_hybrid_state_and_abort_rollback() {
        use crate::opcodes::{instr_fork, instr_ssm_reset, instr_ssm_scan};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let xa = vm.memory.alloc_tensor(&[1, 2], crate::memory::DType::F32).unwrap();
        vm.memory.write_f32_tensor(xa, &[1.0, 0.5]).unwrap();
        let root = vm.memory.current_version();
        let cid = vm.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
        // Rh=0xFF => estado na Vm, layer 0, di=2 ds=2
        let scan = instr_ssm_scan(2, 0, 0xFF, 0xFF, 2, 2, 0);
        vm.step_instruction(cid, &scan).unwrap();
        assert_eq!(vm.ssm_states.len(), 1);
        assert!(vm.ssm_states[0].ssm.iter().any(|&v| v != 0.0));
        // FORK empilha snapshot; novo scan suja; ABORT restaura.
        let fork = instr_fork(5, 0);
        vm.step_instruction(cid, &fork).unwrap();
        vm.step_instruction(cid, &scan).unwrap();
        let dirty: Vec<f32> = vm.ssm_states[0].ssm.clone();
        // ABORT no filho (id em r5) faz pop do snapshot
        let child = vm.scheduler.get(cid).unwrap().reg(5).unwrap() as u64;
        assert_ne!(child, 0);
        let abort = crate::opcodes::instr_abort(5, 0xFF);
        vm.step_instruction(cid, &abort).unwrap();
        assert_ne!(vm.ssm_states[0].ssm, dirty);
        // RESET híbrido zera
        let reset = instr_ssm_reset(0xFF, 2, 2, 0);
        let mut reset = reset;
        reset.rsrc1 = 0xFF;
        vm.step_instruction(cid, &reset).unwrap();
        assert!(vm.ssm_states[0].ssm.iter().all(|&v| v == 0.0));
    }

    #[tokio::test]
    async fn test_codec_enc_dec_roundtrip() {
        use crate::opcodes::{instr_codec_dec, instr_codec_enc, instr_sense};
        use crate::opcodes::{SENSE_AUDIO_PCM, SENSE_CODEC_FRAME};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        // SENSE PCM (7680B) -> ENC (32B) -> DEC (7680B)
        let prog = vec![
            instr_sense(0, SENSE_AUDIO_PCM),
            instr_codec_enc(1, 0, false),
            instr_codec_dec(2, 1, false),
            instr_sense(3, SENSE_CODEC_FRAME),
            instr_halt(),
        ];
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.codec_encs, 1);
        assert_eq!(stats.codec_decs, 1);
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let codes_addr = ctx.reg(1).unwrap();
        let raw = vm.memory.read(codes_addr, 32).unwrap();
        let codes = crate::mimi::codes_from_bytes(&raw).unwrap();
        // synth 440Hz não é silêncio
        assert!(codes.iter().any(|&c| c != crate::mimi::MIMI_SILENCE_CODE));
        let pcm_addr = ctx.reg(2).unwrap();
        let raw_pcm = vm.memory.read(pcm_addr, 1920 * 4).unwrap();
        assert_eq!(raw_pcm.len(), 1920 * 4);
        let cf_addr = ctx.reg(3).unwrap();
        assert_eq!(vm.memory.read(cf_addr, 32).unwrap().len(), 32);
        // Path tensor: ENC TENSOR -> DEC TENSOR preserva energia grosseiramente
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let pa = vm2.memory.alloc_tensor(&[1, 1920], crate::memory::DType::F32).unwrap();
        let synth = crate::mimi::synth_frame_440hz();
        vm2.memory.write_f32_tensor(pa, &synth).unwrap();
        let root = vm2.memory.current_version();
        let cid = vm2.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        vm2.scheduler.get_mut(cid).unwrap().set_reg(0, pa).unwrap();
        let mut enc = instr_codec_enc(1, 0, true);
        enc.flags = crate::opcodes::CODEC_FLAG_AS_TENSOR;
        vm2.step_instruction(cid, &enc).unwrap();
        let ca = vm2.scheduler.get(cid).unwrap().reg(1).unwrap();
        let mut dec = instr_codec_dec(2, 1, true);
        dec.flags = crate::opcodes::CODEC_FLAG_AS_TENSOR;
        vm2.scheduler.get_mut(cid).unwrap().set_reg(1, ca).unwrap();
        vm2.step_instruction(cid, &dec).unwrap();
        let da = vm2.scheduler.get(cid).unwrap().reg(2).unwrap();
        let back = vm2.memory.read_f32_tensor(da, 1920).unwrap();
        let e_in: f32 = synth.iter().map(|v| v * v).sum::<f32>() / synth.len() as f32;
        let e_out: f32 = back.iter().map(|v| v * v).sum::<f32>() / back.len() as f32;
        assert!((e_out - e_in).abs() / e_in < 0.9, "e_in {} e_out {}", e_in, e_out);
    }

    #[tokio::test]
    async fn test_rope_identity_and_rotation() {
        use crate::opcodes::instr_rope;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // pos=0 => identidade
        let xa = vm.memory.alloc_tensor(&[1, 4], crate::memory::DType::F32).unwrap();
        vm.memory.write_f32_tensor(xa, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let root = vm.memory.current_version();
        let cid = vm.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
        vm.step_instruction(cid, &instr_rope(1, 0, 0, 4, 1, 10_000.0)).unwrap();
        let ya = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
        let y = vm.memory.read_f32_tensor(ya, 4).unwrap();
        for (a, b) in [1.0, 2.0, 3.0, 4.0].iter().zip(y.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        // pos!=0 gira (norma preservada, valores mudam)
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
        vm.step_instruction(cid, &instr_rope(1, 0, 7, 4, 1, 10_000.0)).unwrap();
        let ya2 = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
        let y2 = vm.memory.read_f32_tensor(ya2, 4).unwrap();
        let n0: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|v| v * v).sum();
        let n1: f32 = y2.iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-3, "{} vs {}", n0, n1);
        assert!((y2[0] - 1.0).abs() + (y2[1] - 2.0).abs() > 1e-3);
        assert_eq!(vm.stats.ropes, 2);
    }

    #[tokio::test]
    async fn test_audio_align_and_ctx_switch() {
        use crate::opcodes::{instr_audio_align, instr_ctx_switch, PIPE_MAMBA};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let root = vm.memory.current_version();
        let cid = vm.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        // ts imediatos via regs (sem tensor): r0=1s, r1=1.08s (80ms depois = 1 frame Mimi)
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, 1_000_000_000).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, 1_080_000_000).unwrap();
        vm.step_instruction(cid, &instr_audio_align(2, 0, 1)).unwrap();
        let aa = vm.scheduler.get(cid).unwrap().reg(2).unwrap();
        let v = vm.memory.read_f32_tensor(aa, 4).unwrap();
        assert_eq!(v[3] as u64, 1); // 80ms => frame 1
        assert_eq!(vm.stats.audio_aligns, 1);
        // CTX_SWITCH MAMBA, RED
        assert_eq!(vm.scheduler.get(cid).unwrap().pipeline, crate::context::PIPE_TRANSFORMER_CTX);
        vm.step_instruction(cid, &instr_ctx_switch(PIPE_MAMBA, 0b10)).unwrap();
        let ctx = vm.scheduler.get(cid).unwrap();
        assert_eq!(ctx.pipeline, PIPE_MAMBA);
        assert_eq!(ctx.priority, crate::context::Priority::Red);
        assert_eq!(vm.stats.ctx_switches, 1);
    }

    #[tokio::test]
    async fn test_ssm_scan_flags_reservadas_falham_explicito() {
        use crate::opcodes::{instr_ssm_scan, SSM_SCAN_FLAG_CONV, SSM_SCAN_FLAG_GATE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        let root = vm.memory.current_version();
        let cid = vm.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        for flag in [SSM_SCAN_FLAG_CONV, SSM_SCAN_FLAG_GATE] {
            let mut scan = instr_ssm_scan(2, 0, 0xFF, 0xFF, 1, 1, 0);
            scan.flags = flag;
            assert!(vm.step_instruction(cid, &scan).is_err(), "flag 0x{:02x} deveria falhar", flag);
        }
        assert_eq!(vm.stats.ssm_scans, 0);
    }

    #[tokio::test]
    async fn test_ssm_scan_com_pack_explicito() {
        use crate::opcodes::instr_ssm_scan;
        // Pack montado com o helper público == referência ssm::selective_scan_update.
        let (di, ds) = (2usize, 2usize);
        let x = vec![1.0f32, 0.5];
        let dt = vec![0.5f32, 1.0];
        let a = vec![-1.0f32; 4];
        let b = vec![1.0f32, 0.5];
        let c = vec![1.0f32, 1.0];
        let d = vec![0.1f32, 0.0];
        let pack = crate::ssm::pack_params(&dt, &a, &b, &c, &d, di, ds);
        let mut expect_h = vec![0.0f32; 4];
        let expect_y = crate::ssm::selective_scan_update(&mut expect_h, &x, &dt, &a, &b, &c, &d, di, ds);

        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        let xa = vm.memory.alloc_tensor(&[1, di], crate::memory::DType::F32).unwrap();
        vm.memory.write_f32_tensor(xa, &x).unwrap();
        let ha = vm.memory.alloc_tensor(&[di, ds], crate::memory::DType::F32).unwrap();
        vm.memory.write_f32_tensor(ha, &[0.0; 4]).unwrap();
        let pa = vm.memory.alloc_tensor(&[1, pack.len()], crate::memory::DType::F32).unwrap();
        vm.memory.write_f32_tensor(pa, &pack).unwrap();
        let root = vm.memory.current_version();
        let cid = vm.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, ha).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, pa).unwrap();
        vm.step_instruction(cid, &instr_ssm_scan(3, 0, 1, 2, di, ds, 0)).unwrap();
        let ya = vm.scheduler.get(cid).unwrap().reg(3).unwrap();
        let y = vm.memory.read_f32_tensor(ya, di).unwrap();
        for (got, want) in y.iter().zip(expect_y.iter()) {
            assert!((got - want).abs() < 1e-5, "{} vs {}", got, want);
        }
        assert_eq!(vm.memory.read_f32_tensor(ha, 4).unwrap(), expect_h);
    }

    #[tokio::test]
    async fn test_new_isa_assembled_program_runs() {
        use crate::opcodes::assemble;
        let src = r#"
            TENSOR r0 1 2 f32
            TENSOR r1 2 1 f32
            SSM_SCAN r5, r0, r1, r0 D_INNER=2 D_STATE=1 LAYER=0
            SSM_RESET r1 D_INNER=2 D_STATE=1
            SENSE r6, AUDIO_PCM
            CODEC_ENC r7, r6
            CODEC_DEC r8, r7
            AUDIO_ALIGN r4, r0, r0
            CTX_SWITCH TRANSFORMER, GREEN
            ROPE r2, r0 POS=0 HDIM=2 NHEADS=1
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.ssm_scans, 1);
        assert_eq!(stats.ssm_resets, 1);
        assert_eq!(stats.codec_encs, 1);
        assert_eq!(stats.codec_decs, 1);
        assert_eq!(stats.audio_aligns, 1);
        assert_eq!(stats.ctx_switches, 1);
        assert_eq!(stats.ropes, 1);
    }

    // ---- RFC-0004: goldens GATHER / DISTANCE / RANK1 / SAMPLE-TOPK ----

    fn rfc0004_ctx_with(vm: &mut Vm, regs: &[(u8, u128)]) -> u64 {
        let root = vm.memory.current_version();
        let cid = vm.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        for (r, v) in regs {
            vm.scheduler.get_mut(cid).unwrap().set_reg(*r, *v).unwrap();
        }
        cid
    }

    fn rfc0004_f32(vm: &mut Vm, shape: &[usize], data: &[f32]) -> u128 {
        let a = vm.memory.alloc_tensor(shape, crate::memory::DType::F32).unwrap();
        vm.memory.write_f32_tensor(a, data).unwrap();
        a
    }

    #[tokio::test]
    async fn test_gather_golden_and_oob() {
        use crate::opcodes::{
            instr_gather, GATHER_MODE_GATHER, GATHER_MODE_SCATTER_ADD, GATHER_MODE_SCATTER_MAX,
        };
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let table = rfc0004_f32(&mut vm, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let idx = rfc0004_f32(&mut vm, &[2], &[1.0, 0.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, table), (1, idx)]);
        // axis 0: linhas [1,0] => [[3,4],[1,2]]
        vm.step_instruction(cid, &instr_gather(5, 0, 1, 0xFF, 0, GATHER_MODE_GATHER)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(5).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 4).unwrap(), vec![3.0, 4.0, 1.0, 2.0]);
        // axis 1: coluna [1] => [[2],[4]]
        let idx1 = rfc0004_f32(&mut vm, &[1], &[1.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, idx1).unwrap();
        vm.step_instruction(cid, &instr_gather(5, 0, 1, 0xFF, 1, GATHER_MODE_GATHER)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(5).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 2).unwrap(), vec![2.0, 4.0]);
        // scatter_add: acc zeros + valores [[10,20]] idx [1] => [[0,0],[10,20]]
        let acc = rfc0004_f32(&mut vm, &[2, 2], &[0.0; 4]);
        let vals = rfc0004_f32(&mut vm, &[1, 2], &[10.0, 20.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, vals).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, acc).unwrap();
        vm.step_instruction(cid, &instr_gather(5, 0, 1, 2, 0, GATHER_MODE_SCATTER_ADD)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(5).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 4).unwrap(), vec![0.0, 0.0, 10.0, 20.0]);
        // scatter_max sobre o resultado: max com [[5,25]] idx [1] => [[0,0],[10,25]]
        let vals2 = rfc0004_f32(&mut vm, &[1, 2], &[5.0, 25.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, vals2).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, out).unwrap();
        vm.step_instruction(cid, &instr_gather(5, 0, 1, 2, 0, GATHER_MODE_SCATTER_MAX)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(5).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 4).unwrap(), vec![0.0, 0.0, 10.0, 25.0]);
        // OOB e erros determinísticos.
        let bad = rfc0004_f32(&mut vm, &[1], &[5.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, bad).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, table).unwrap();
        assert!(vm.step_instruction(cid, &instr_gather(5, 0, 1, 0xFF, 0, GATHER_MODE_GATHER)).is_err());
        assert!(vm.step_instruction(cid, &instr_gather(5, 0, 1, 0xFF, 7, GATHER_MODE_GATHER)).is_err());
        assert!(vm.step_instruction(cid, &instr_gather(5, 0, 1, 0xFF, 0, 9)).is_err());
        assert_eq!(vm.stats.gather_execs, 4);
    }

    #[tokio::test]
    async fn test_distance_four_metrics_and_topk() {
        use crate::opcodes::{
            instr_distance, DIST_METRIC_COSINE, DIST_METRIC_DOT, DIST_METRIC_EUCLID,
            DIST_METRIC_MANHATTAN,
        };
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let q = rfc0004_f32(&mut vm, &[1, 2], &[1.0, 0.0]);
        let bank = rfc0004_f32(&mut vm, &[3, 2], &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, q), (1, bank)]);
        let run = |vm: &mut Vm, cid: u64, metric: u8, topk: u16| -> (Vec<usize>, Vec<f32>) {
            vm.step_instruction(cid, &instr_distance(5, 0, 1, metric, topk)).unwrap();
            let out = vm.scheduler.get(cid).unwrap().reg(5).unwrap();
            let meta = vm.memory.get_tensor_meta(out).cloned().unwrap();
            let n: usize = meta.shape.iter().product();
            (meta.shape.clone(), vm.memory.read_f32_tensor(out, n).unwrap())
        };
        let (_, e) = run(&mut vm, cid, DIST_METRIC_EUCLID, 0);
        assert!((e[0] - 0.0).abs() < 1e-5 && (e[1] - 2f32.sqrt()).abs() < 1e-5 && (e[2] - 2.0).abs() < 1e-5);
        let (_, c) = run(&mut vm, cid, DIST_METRIC_COSINE, 0);
        assert!((c[0] - 0.0).abs() < 1e-5 && (c[1] - 1.0).abs() < 1e-5 && (c[2] - 2.0).abs() < 1e-5);
        let (_, m) = run(&mut vm, cid, DIST_METRIC_MANHATTAN, 0);
        assert_eq!(m, vec![0.0, 2.0, 2.0]);
        let (_, d) = run(&mut vm, cid, DIST_METRIC_DOT, 0);
        assert_eq!(d, vec![-1.0, 0.0, 1.0]);
        // topk=2 euclid: dists [0, √2], índices [0, 1], shape [1,4].
        let (shape, t) = run(&mut vm, cid, DIST_METRIC_EUCLID, 2);
        assert_eq!(shape, vec![1, 4]);
        assert!((t[0] - 0.0).abs() < 1e-5 && (t[1] - 2f32.sqrt()).abs() < 1e-5);
        assert_eq!(&t[2..], &[0.0, 1.0]);
        // Métrica inválida e shape inválido: erro limpo.
        let mut badm = instr_distance(5, 0, 1, 0, 0);
        badm.payload[0] = 9;
        assert!(vm.step_instruction(cid, &badm).is_err());
        assert_eq!(vm.stats.distance_execs, 5);
    }

    #[tokio::test]
    async fn test_rank1_modes_and_rollback() {
        use crate::opcodes::{
            instr_fork, instr_rank1_update, RANK1_MODE_DELTA, RANK1_MODE_FORGET,
            RANK1_MODE_HEBBIAN,
        };
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let h = rfc0004_f32(&mut vm, &[2, 2], &[0.0; 4]);
        let v = rfc0004_f32(&mut vm, &[2], &[1.0, 2.0]);
        let k = rfc0004_f32(&mut vm, &[2], &[3.0, 4.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(7, h), (1, v), (2, k)]);
        // hebbian α=1 β=1: H = v⊗k = [[3,4],[6,8]].
        vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 1.0, 1.0, RANK1_MODE_HEBBIAN, 0)).unwrap();
        let h1 = vm.scheduler.get(cid).unwrap().reg(7).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(h1, 4).unwrap(), vec![3.0, 4.0, 6.0, 8.0]);
        assert_eq!(vm.rank1_layers.get(&0), Some(&h1));
        // delta sobre H identidade: v=[1,0], k=[1,0] => Hk=[1,0], denom=2,
        // upd=0 => H inalterada.
        let hi = rfc0004_f32(&mut vm, &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let v2 = rfc0004_f32(&mut vm, &[2], &[1.0, 0.0]);
        let k2 = rfc0004_f32(&mut vm, &[2], &[1.0, 0.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, hi).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, v2).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, k2).unwrap();
        vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 1.0, 1.0, RANK1_MODE_DELTA, 1)).unwrap();
        let h2 = vm.scheduler.get(cid).unwrap().reg(7).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(h2, 4).unwrap(), vec![1.0, 0.0, 0.0, 1.0]);
        // forget com máscara: H'=0.5*(m⊙H)+v⊗k, m=[[1,0],[0,1]], H=ones.
        let ho = rfc0004_f32(&mut vm, &[2, 2], &[1.0; 4]);
        let m = rfc0004_f32(&mut vm, &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, ho).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, v).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, k).unwrap();
        let mut f = instr_rank1_update(7, 1, 2, 0.5, 1.0, RANK1_MODE_FORGET, 2);
        f.rsrc3 = 3;
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, m).unwrap();
        vm.step_instruction(cid, &f).unwrap();
        let h3 = vm.scheduler.get(cid).unwrap().reg(7).unwrap();
        // 0.5*[[1,0],[0,1]] + [[3,4],[6,8]] = [[3.5,4],[6,8.5]]
        assert_eq!(vm.memory.read_f32_tensor(h3, 4).unwrap(), vec![3.5, 4.0, 6.0, 8.5]);
        // FORK empilha handles; passo suja; ABORT restaura o handle (CoW: o
        // tensor antigo segue intacto e legível).
        let before = vm.rank1_layers.clone();
        vm.step_instruction(cid, &instr_fork(5, 0)).unwrap();
        vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 1.0, 1.0, RANK1_MODE_HEBBIAN, 2)).unwrap();
        let dirty = vm.scheduler.get(cid).unwrap().reg(7).unwrap();
        assert_ne!(dirty, h3);
        let child = vm.scheduler.get(cid).unwrap().reg(5).unwrap() as u64;
        assert_ne!(child, cid);
        // r5 = id do filho, ts = 0xFF (sem restore de memória; isola o mapa RANK1).
        vm.step_instruction(cid, &crate::opcodes::instr_abort(5, 0xFF)).unwrap();
        assert_eq!(vm.rank1_layers, before);
        // O handle da camada voltou a apontar ao H pré-FORK e o tensor
        // antigo segue intacto (CoW). NOTA: registradores do contexto NÃO
        // fazem parte do rollback (desenho pré-existente do FORK/ABORT, vale
        // p/ todos os opcodes) — r7 segue com o addr sujo; o estado
        // autoritativo é o mapa restaurado. Ver follow-up na RFC-0004.
        assert_eq!(vm.rank1_layers.get(&2), Some(&h3));
        assert_eq!(vm.memory.read_f32_tensor(h3, 4).unwrap(), vec![3.5, 4.0, 6.0, 8.5]);
        // Modo inválido e ALPHA fora de (0,1] no FORGET: erro limpo.
        assert!(vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 1.0, 1.0, 9, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 2.0, 1.0, RANK1_MODE_FORGET, 0)).is_err());
        assert_eq!(vm.stats.rank1_execs, 4);
    }

    #[tokio::test]
    async fn test_sample_topk_indices() {
        use crate::opcodes::instr_sample_topk;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // Empate 0.9/0.9 desempata por menor índice: [1, 3].
        let l = rfc0004_f32(&mut vm, &[4], &[0.1, 0.9, 0.5, 0.9]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, l)]);
        let before = vm.last_sample;
        vm.step_instruction(cid, &instr_sample_topk(4, 0, 2)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 2).unwrap(), vec![1.0, 3.0]);
        assert_eq!(vm.last_sample, before, "TOPK não toca em last_sample");
    }

    #[tokio::test]
    async fn test_rfc0004_assembled_program_runs() {
        use crate::opcodes::assemble;
        // Pipeline combinado: SAMPLE-TOPK -> GATHER, DISTANCE, RANK1, FORK/ABORT.
        let src = r#"
            TENSOR r0 1 4 f32
            TENSOR r1 4 2 f32
            SAMPLE r2, r0 TOPK=1
            GATHER r3, r1, r2 AXIS=0
            TENSOR r4 1 4 f32
            TENSOR r5 8 4 f32
            DISTANCE r6, r4, r5 METRIC=DOT TOPK=2
            TENSOR r7 2 2 f32
            TENSOR r8 2 1 f32
            TENSOR r9 2 1 f32
            RANK1_UPDATE r7, r8, r9 ALPHA=1.0 BETA=1.0 MODE=HEBBIAN LAYER=0
            FORK r10, GREEN
            RANK1_UPDATE r7, r8, r9 ALPHA=1.0 BETA=1.0 MODE=HEBBIAN LAYER=0
            ABORT r10, r11
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.sample_execs, 1);
        assert_eq!(stats.gather_execs, 1);
        assert_eq!(stats.distance_execs, 1);
        // RANK1 x3: pai antes + pai depois do FORK + filho (nasce após o FORK
        // e executa o RANK1 pós-FORK antes do seu ABORT-alvo-0, que não conta).
        assert_eq!(stats.rank1_execs, 3);
        assert_eq!(stats.forks, 1);
        assert_eq!(stats.aborts, 1);
    }

    // ---- RFC-0023: núcleo de memória --------------------------------

    #[test]
    fn test_rfc0023_arena_alloc_reset() {
        use crate::opcodes::{instr_arena_alloc, instr_arena_reset};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        vm.step_instruction(cid, &instr_arena_alloc(2, 64, 16, 1)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 2), 0);
        vm.step_instruction(cid, &instr_arena_alloc(3, 32, 16, 1)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 3), 64);
        // cursor=96 já é múltiplo de 32 => 96 (align-up real, não arredonda à toa).
        vm.step_instruction(cid, &instr_arena_alloc(4, 1, 32, 1)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 4), 96);
        // align-up de verdade: cursor=97, align 32 => 128.
        vm.step_instruction(cid, &instr_arena_alloc(7, 1, 32, 1)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 7), 128);
        // Arenas separadas por id.
        vm.step_instruction(cid, &instr_arena_alloc(5, 16, 0, 7)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 5), 0);
        // RESET + reuso O(1).
        vm.step_instruction(cid, &instr_arena_reset(1)).unwrap();
        vm.step_instruction(cid, &instr_arena_alloc(6, 16, 0, 1)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 6), 0);
        // Erros: size 0, align não-potência, reset de arena inexistente.
        assert!(vm.step_instruction(cid, &instr_arena_alloc(6, 0, 16, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_arena_alloc(6, 8, 3, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_arena_reset(9)).is_err());
        assert_eq!(vm.stats.arena_alloc_execs, 6);
        assert_eq!(vm.stats.arena_reset_execs, 1);
    }

    #[test]
    fn test_rfc0023_memcpy_memset() {
        use crate::opcodes::{instr_memcpy, instr_memset, MEMCPY_DIR_GPU, MEMCPY_DIR_HOST};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let a1 = rfc0004_f32(&mut vm, &[2, 2], &[0.0; 4]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0), (1, a1)]);
        // Cópia inteira bit-exata.
        vm.step_instruction(cid, &instr_memcpy(1, 0, 0, 0, 0, MEMCPY_DIR_HOST)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a1, 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        // MEMSET parcial: primeiros 4 bytes zeram o 1º f32.
        vm.step_instruction(cid, &instr_memset(1, 0, 4, 0)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a1, 4).unwrap(), vec![0.0, 2.0, 3.0, 4.0]);
        // Janela parcial: bytes [4..8] de a0 (2.0f32) sobre a1[8..12].
        vm.step_instruction(cid, &instr_memcpy(1, 0, 4, 4, 8, MEMCPY_DIR_HOST)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a1, 4).unwrap(), vec![0.0, 2.0, 2.0, 4.0]);
        // Self-copy com overlap: memmove via buffer próprio.
        vm.step_instruction(cid, &instr_memcpy(1, 1, 4, 0, 4, MEMCPY_DIR_HOST)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a1, 4).unwrap(), vec![0.0, 0.0, 2.0, 4.0]);
        // Erros: OOB fonte/destino, dir, meta ausente, esparso, PERSISTENTE.
        assert!(vm.step_instruction(cid, &instr_memcpy(1, 0, 99, 0, 0, MEMCPY_DIR_HOST)).is_err());
        assert!(vm.step_instruction(cid, &instr_memcpy(1, 0, 4, 0, 13, MEMCPY_DIR_HOST)).is_err());
        assert!(vm.step_instruction(cid, &instr_memcpy(1, 0, 0, 0, 0, MEMCPY_DIR_GPU)).is_err());
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, 0xdead).unwrap();
        assert!(vm.step_instruction(cid, &instr_memcpy(2, 0, 0, 0, 0, MEMCPY_DIR_HOST)).is_err());
        assert!(vm.step_instruction(cid, &instr_memset(1, 0xAB, 99, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_memset(1, 0, 0, 4)).is_err());
        let a_sparse = vm.memory.alloc_sparse_tensor(&[2, 2], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, a_sparse).unwrap();
        assert!(vm.step_instruction(cid, &instr_memcpy(3, 0, 0, 0, 0, MEMCPY_DIR_HOST)).is_err());
        assert!(vm.step_instruction(cid, &instr_memcpy(1, 3, 0, 0, 0, MEMCPY_DIR_HOST)).is_err());
        assert!(vm.step_instruction(cid, &instr_memset(3, 0, 0, 0)).is_err());
        // Destino PERSISTENTE: meta existe, região veta (WEIGHTS RO).
        let paddr = 0x2000_0000_0000_0000_0000_0000_0000_1000u128;
        vm.memory.as_cpu_mut().unwrap().tensor_meta.insert(paddr, crate::memory::TensorMeta {
            addr: paddr, shape: vec![4], dtype: crate::memory::DType::F32,
            byte_len: 16, is_sparse: false, density: 1.0,
        });
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, paddr).unwrap();
        assert!(vm.step_instruction(cid, &instr_memcpy(4, 0, 0, 0, 0, MEMCPY_DIR_HOST)).is_err());
        assert!(vm.step_instruction(cid, &instr_memset(4, 0, 0, 0)).is_err());
        assert_eq!(vm.stats.memcpy_execs, 3);
        assert_eq!(vm.stats.memset_execs, 1);
    }

    #[test]
    fn test_rfc0023_arena_fork_abort() {
        use crate::opcodes::{instr_abort, instr_arena_alloc, instr_fork, FORK_FLAG_GREEN};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let parent = rfc0004_ctx_with(&mut vm, &[]);
        vm.step_instruction(parent, &instr_arena_alloc(0, 64, 16, 0)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, parent, 0), 0);
        vm.step_instruction(parent, &instr_fork(1, FORK_FLAG_GREEN)).unwrap();
        let child = rfc0005_reg_u64(&vm, parent, 1);
        // Mapa vivo compartilhado até o ABORT (disciplina rank1_layers):
        // filho aloca em 64, cursor vai a 96.
        vm.step_instruction(child, &instr_arena_alloc(2, 32, 16, 0)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, child, 2), 64);
        // Pai vê o cursor movido (sem cópia concorrente — isolamento é
        // por snapshot, não por cópia).
        vm.step_instruction(parent, &instr_arena_alloc(3, 8, 16, 0)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, parent, 3), 96);
        // ABORT ts=0 restaura o snapshot do FORK (cursor 64): os dois
        // allocs pós-FORK desfazem, como rank1_layers.
        vm.scheduler.get_mut(parent).unwrap().set_reg(4, child as u128).unwrap();
        vm.scheduler.get_mut(parent).unwrap().set_reg(5, 0).unwrap();
        vm.step_instruction(parent, &instr_abort(4, 5)).unwrap();
        vm.step_instruction(parent, &instr_arena_alloc(6, 8, 16, 0)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, parent, 6), 64);
        assert_eq!(vm.stats.arena_alloc_execs, 4);
        assert_eq!(vm.stats.forks, 1);
        assert_eq!(vm.stats.aborts, 1);
    }

    #[tokio::test]
    async fn test_rfc0023_assembled_program_runs() {
        use crate::opcodes::assemble;
        // Pipeline combinado: FILL -> MEMCPY -> MEMSET + ARENA com reuso.
        let src = r#"
            TENSOR r0 2 2 f32 FILL=1
            TENSOR r1 2 2 f32
            MEMCPY r1, r0
            MEMSET r1 PATTERN=0 LEN=4
            ARENA_ALLOC r2, SIZE=64
            ARENA_ALLOC r3, SIZE=32 ALIGN=32
            ARENA_RESET
            ARENA_ALLOC r4, SIZE=16
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.memcpy_execs, 1);
        assert_eq!(stats.memset_execs, 1);
        assert_eq!(stats.arena_alloc_execs, 3);
        assert_eq!(stats.arena_reset_execs, 1);
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let a1 = ctx.reg(1).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a1, 4).unwrap(), vec![0.0, 1.0, 1.0, 1.0]);
        // RESET reuso: r2=0, r3=64 (align 32 de cursor 64), r4=0.
        assert_eq!(ctx.reg(2).unwrap(), 0);
        assert_eq!(ctx.reg(3).unwrap(), 64);
        assert_eq!(ctx.reg(4).unwrap(), 0);
    }

    #[tokio::test]
    async fn test_rfc0023_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/arena_memcpy_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.arena_alloc_execs, stats.arena_reset_execs, stats.memcpy_execs, stats.memset_execs), (3, 1, 1, 1));
    }

    // ---- RFC-0024: views & versions ---------------------------------

    #[test]
    fn test_rfc0024_snapshot_restore_roundtrip() {
        use crate::opcodes::{instr_memset, instr_restore, instr_snapshot, SNAP_MASK_ALL};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[2, 2], &[1.0; 4]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0), (5, 0), (6, 0), (7, 9999)]);
        // Snapshot nomeado: fresh boot => v1.
        vm.step_instruction(cid, &instr_snapshot(5, SNAP_MASK_ALL)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 5), 1);
        // Sujidade + rewind bit-exato.
        vm.step_instruction(cid, &instr_memset(0, 0, 0, 0)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a0, 4).unwrap(), vec![0.0; 4]);
        vm.step_instruction(cid, &instr_restore(5)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a0, 4).unwrap(), vec![1.0; 4]);
        // Monotônico: próximo snapshot é v1+1 (restore nunca rebaixa).
        vm.step_instruction(cid, &instr_snapshot(6, SNAP_MASK_ALL)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 6), 2);
        // Erros: MASK parcial, versão desconhecida, rV ausente.
        assert!(vm.step_instruction(cid, &crate::opcodes::instr_snapshot(5, 0b001)).is_err());
        assert!(vm.step_instruction(cid, &instr_restore(7)).is_err());
        assert!(vm.step_instruction(cid, &instr_restore(0xFF)).is_err());
        assert_eq!(vm.stats.snapshot_execs, 2);
        assert_eq!(vm.stats.restore_execs, 1);
    }

    #[test]
    fn test_rfc0024_restore_rewinds_engine() {
        use crate::opcodes::{instr_arena_alloc, instr_restore, instr_snapshot, SNAP_MASK_ALL};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        vm.step_instruction(cid, &instr_arena_alloc(0, 64, 16, 0)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 0), 0);
        vm.step_instruction(cid, &instr_snapshot(5, SNAP_MASK_ALL)).unwrap();
        vm.step_instruction(cid, &instr_arena_alloc(1, 32, 16, 0)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 1), 64);
        // RESTORE rebobina o mapa (cursor volta a 64).
        vm.step_instruction(cid, &instr_restore(5)).unwrap();
        vm.step_instruction(cid, &instr_arena_alloc(2, 8, 16, 0)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 2), 64);
    }

    #[test]
    fn test_rfc0024_prefetch_reshape_concat() {
        use crate::opcodes::{instr_concat, instr_prefetch, instr_reshape};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b0 = rfc0004_f32(&mut vm, &[2, 4], &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0), (1, b0), (9, 0xdead)]);
        // PREFETCH inteiro + janela + erros.
        vm.step_instruction(cid, &instr_prefetch(0, 0, 0)).unwrap();
        vm.step_instruction(cid, &instr_prefetch(0, 8, 8)).unwrap();
        assert!(vm.step_instruction(cid, &instr_prefetch(0, 99, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_prefetch(9, 0, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_prefetch(0xFF, 0, 0)).is_err());
        // RESHAPE [2,4] -> [8]: valores preservados; ndim muda p/ [2,2,2].
        vm.step_instruction(cid, &instr_reshape(2, 0, &[8])).unwrap();
        let r1 = rfc0005_reg_u64(&vm, cid, 2) as u128;
        assert_eq!(vm.memory.read_f32_tensor(r1, 8).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        vm.step_instruction(cid, &instr_reshape(3, 0, &[2, 2, 2])).unwrap();
        let r2 = rfc0005_reg_u64(&vm, cid, 3) as u128;
        assert_eq!(vm.memory.read_f32_tensor(r2, 8).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!(vm.step_instruction(cid, &instr_reshape(4, 0, &[3])).is_err());
        assert!(vm.step_instruction(cid, &instr_reshape(4, 9, &[8])).is_err());
        // CONCAT eixo 0: [2,4]+[2,4] -> [4,4] (A em cima, B embaixo).
        vm.step_instruction(cid, &instr_concat(4, 0, 1, 0)).unwrap();
        let rc = rfc0005_reg_u64(&vm, cid, 4) as u128;
        assert_eq!(
            vm.memory.read_f32_tensor(rc, 16).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
        );
        // CONCAT eixo 1: [2,4]+[2,4] -> [2,8] (lado a lado por linha).
        vm.step_instruction(cid, &instr_concat(5, 0, 1, 1)).unwrap();
        let rc1 = rfc0005_reg_u64(&vm, cid, 5) as u128;
        assert_eq!(
            vm.memory.read_f32_tensor(rc1, 16).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 8.0, 50.0, 60.0, 70.0, 80.0]
        );
        // Erros: eixo fora, dims fora do eixo, dtype, esparso.
        assert!(vm.step_instruction(cid, &instr_concat(6, 0, 1, 2)).is_err());
        let c_mismatch = rfc0004_f32(&mut vm, &[2, 3], &[0.0; 6]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, c_mismatch).unwrap();
        assert!(vm.step_instruction(cid, &instr_concat(7, 0, 6, 0)).is_err());
        let d_f16 = vm.memory.alloc_tensor(&[2, 4], crate::memory::DType::F16).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, d_f16).unwrap();
        assert!(vm.step_instruction(cid, &instr_concat(8, 0, 7, 0)).is_err());
        let a_sparse = vm.memory.alloc_sparse_tensor(&[2, 4], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(8, a_sparse).unwrap();
        assert!(vm.step_instruction(cid, &instr_concat(0, 8, 1, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_reshape(0, 8, &[8])).is_err());
        assert!(vm.step_instruction(cid, &instr_prefetch(8, 0, 0)).is_err());
        assert_eq!(vm.stats.prefetch_execs, 2);
        assert_eq!(vm.stats.reshape_execs, 2);
        assert_eq!(vm.stats.concat_execs, 2);
    }

    #[tokio::test]
    async fn test_rfc0024_assembled_program_runs() {
        use crate::opcodes::assemble;
        // Pipeline combinado: SNAPSHOT -> sujidade -> RESTORE + PREFETCH +
        // RESHAPE + CONCAT.
        let src = r#"
            TENSOR r0 2 2 f32 FILL=1
            SNAPSHOT r5
            MEMSET r0 PATTERN=0
            RESTORE r5
            PREFETCH r0
            RESHAPE r1, r0 SHAPE=1x4
            CONCAT r2, r1, r1 AXIS=0
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.snapshot_execs, 1);
        assert_eq!(stats.restore_execs, 1);
        assert_eq!(stats.prefetch_execs, 1);
        assert_eq!(stats.reshape_execs, 1);
        assert_eq!(stats.concat_execs, 1);
        let ctx = vm.scheduler.get(1).unwrap().clone();
        // r0 restaurado a ones; r2 = [2,4] de ones.
        let a0 = ctx.reg(0).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a0, 4).unwrap(), vec![1.0; 4]);
        let a2 = ctx.reg(2).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a2, 8).unwrap(), vec![1.0; 8]);
    }

    #[tokio::test]
    async fn test_rfc0024_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/snap_concat_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.snapshot_execs, stats.restore_execs, stats.prefetch_execs, stats.reshape_execs, stats.concat_execs), (1, 1, 1, 1, 1));
    }

    // ---- RFC-0025: conversão ----------------------------------------

    #[test]
    fn test_rfc0025_cast_pairs() {
        use crate::opcodes::{instr_cast, CAST_DST_F16, CAST_DST_BF16, CAST_DST_I8, CAST_DST_U8};
        use crate::memory::DType;
        assert_eq!(DType::from_u8(64), DType::BF16);
        assert_eq!(DType::BF16.byte_width(), 2);
        assert!(!DType::BF16.is_quantized());
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[4], &[1.0, -2.5, 0.5, 100.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0)]);
        // F16 ida e volta exata (valores representáveis em meia precisão).
        vm.step_instruction(cid, &instr_cast(1, 0, CAST_DST_F16)).unwrap();
        let m1 = vm.memory.get_tensor_meta(rfc0005_reg_u64(&vm, cid, 1) as u128).unwrap().clone();
        assert_eq!((m1.dtype, m1.byte_len), (DType::F16, 8));
        vm.step_instruction(cid, &instr_cast(2, 1, crate::opcodes::CAST_DST_F32)).unwrap();
        let b = rfc0005_reg_u64(&vm, cid, 2) as u128;
        assert_eq!(vm.memory.read_f32_tensor(b, 4).unwrap(), vec![1.0, -2.5, 0.5, 100.0]);
        // BF16 ida e volta exata (8 bits de mantissa cobrem estes valores).
        vm.step_instruction(cid, &instr_cast(3, 0, CAST_DST_BF16)).unwrap();
        let m3 = vm.memory.get_tensor_meta(rfc0005_reg_u64(&vm, cid, 3) as u128).unwrap().clone();
        assert_eq!((m3.dtype, m3.byte_len), (DType::BF16, 8));
        vm.step_instruction(cid, &instr_cast(4, 3, crate::opcodes::CAST_DST_F32)).unwrap();
        let b4 = rfc0005_reg_u64(&vm, cid, 4) as u128;
        assert_eq!(vm.memory.read_f32_tensor(b4, 4).unwrap(), vec![1.0, -2.5, 0.5, 100.0]);
        // INT8 com saturação documentada: [1,-3,1,100].
        vm.step_instruction(cid, &instr_cast(5, 0, CAST_DST_I8)).unwrap();
        vm.step_instruction(cid, &instr_cast(6, 5, crate::opcodes::CAST_DST_F32)).unwrap();
        let b6 = rfc0005_reg_u64(&vm, cid, 6) as u128;
        assert_eq!(vm.memory.read_f32_tensor(b6, 4).unwrap(), vec![1.0, -3.0, 1.0, 100.0]);
        // U8 com clamp: [1,0,1,100].
        vm.step_instruction(cid, &instr_cast(7, 0, CAST_DST_U8)).unwrap();
        vm.step_instruction(cid, &instr_cast(8, 7, crate::opcodes::CAST_DST_F32)).unwrap();
        let b8 = rfc0005_reg_u64(&vm, cid, 8) as u128;
        assert_eq!(vm.memory.read_f32_tensor(b8, 4).unwrap(), vec![1.0, 0.0, 1.0, 100.0]);
        // Erros: identidade, código ruim, fonte quantizada, meta ausente, rdest 0xFF.
        assert!(vm.step_instruction(cid, &instr_cast(1, 0, crate::opcodes::CAST_DST_F32)).is_err());
        let mut bad_code = instr_cast(1, 0, CAST_DST_F16);
        bad_code.set_cast_dst(9);
        assert!(vm.step_instruction(cid, &bad_code).is_err());
        let aq = vm.memory.alloc_tensor(&[32], DType::F32).unwrap();
        vm.memory.as_cpu_mut().unwrap().tensor_meta.insert(aq, crate::memory::TensorMeta {
            addr: aq, shape: vec![32], dtype: DType::Q4_0, byte_len: 18, is_sparse: false, density: 1.0,
        });
        vm.scheduler.get_mut(cid).unwrap().set_reg(9, aq).unwrap();
        assert!(vm.step_instruction(cid, &instr_cast(1, 9, CAST_DST_F32)).is_err());
        vm.scheduler.get_mut(cid).unwrap().set_reg(9, 0xdead).unwrap();
        assert!(vm.step_instruction(cid, &instr_cast(1, 9, CAST_DST_F32)).is_err());
        assert!(vm.step_instruction(cid, &instr_cast(0xFF, 0, CAST_DST_F16)).is_err());
        assert_eq!(vm.stats.cast_execs, 8);
    }

    #[test]
    fn test_rfc0025_quantize_dequant() {
        use crate::opcodes::{instr_dequant, instr_quantize, QUANTIZE_Q4_0, QUANTIZE_Q8_0};
        use crate::memory::DType;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let ramp: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let ar = rfc0004_f32(&mut vm, &[32], &ramp);
        let ao = rfc0004_f32(&mut vm, &[32], &[1.0; 32]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, ar), (1, ao)]);
        // Q4_0 golden: bloco de ones => d=f16(1/7), quants 0xFF.
        vm.step_instruction(cid, &instr_quantize(2, 1, QUANTIZE_Q4_0)).unwrap();
        let q4 = rfc0005_reg_u64(&vm, cid, 2) as u128;
        let mq = vm.memory.get_tensor_meta(q4).unwrap().clone();
        assert_eq!((mq.dtype, mq.byte_len), (DType::Q4_0, 18));
        let raw = vm.memory.read(q4, 18).unwrap();
        let d_exp = half::f16::from_f32(1.0 / 7.0).to_bits().to_le_bytes();
        assert_eq!(&raw[0..2], &d_exp);
        assert!(raw[2..].iter().all(|&b| b == 0xFF));
        // Q4_0 roundtrip na rampa com bound do d armazenado.
        vm.step_instruction(cid, &instr_quantize(3, 0, QUANTIZE_Q4_0)).unwrap();
        let q4r = rfc0005_reg_u64(&vm, cid, 3) as u128;
        vm.step_instruction(cid, &instr_dequant(4, 3)).unwrap();
        let back = rfc0005_reg_u64(&vm, cid, 4) as u128;
        let got = vm.memory.read_f32_tensor(back, 32).unwrap();
        let draw = vm.memory.read(q4r, 18).unwrap();
        let d = half::f16::from_bits(u16::from_le_bytes([draw[0], draw[1]])).to_f32();
        for (x, y) in ramp.iter().zip(got.iter()) {
            assert!((x - y).abs() <= d / 2.0 + 1e-4, "x={} y={} d={}", x, y, d);
        }
        // Q8_0 ida e volta.
        vm.step_instruction(cid, &instr_quantize(5, 1, QUANTIZE_Q8_0)).unwrap();
        let q8 = rfc0005_reg_u64(&vm, cid, 5) as u128;
        let mq8 = vm.memory.get_tensor_meta(q8).unwrap().clone();
        assert_eq!((mq8.dtype, mq8.byte_len), (DType::Q8_0, 34));
        vm.step_instruction(cid, &instr_dequant(6, 5)).unwrap();
        let back8 = rfc0005_reg_u64(&vm, cid, 6) as u128;
        let got8 = vm.memory.read_f32_tensor(back8, 32).unwrap();
        let draw8 = vm.memory.read(q8, 34).unwrap();
        let d8 = half::f16::from_bits(u16::from_le_bytes([draw8[0], draw8[1]])).to_f32();
        for y in got8 {
            assert!((y - 1.0).abs() <= d8 / 2.0 + 1e-6, "y={} d={}", y, d8);
        }
        // Erros: fora de múltiplo de 32, não-finito, fonte não-F32,
        // tipo sem encoder, esparso, meta ausente.
        let a30 = rfc0004_f32(&mut vm, &[30], &[1.0; 30]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, a30).unwrap();
        assert!(vm.step_instruction(cid, &instr_quantize(8, 7, QUANTIZE_Q4_0)).is_err());
        let ai = rfc0004_f32(&mut vm, &[32], &[1.0; 32]);
        let mut inf = vm.memory.read_f32_tensor(ai, 32).unwrap();
        inf[0] = f32::INFINITY;
        vm.memory.write_f32_tensor(ai, &inf).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, ai).unwrap();
        assert!(vm.step_instruction(cid, &instr_quantize(8, 7, QUANTIZE_Q4_0)).is_err());
        let af = vm.memory.alloc_tensor(&[32], DType::F16).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, af).unwrap();
        assert!(vm.step_instruction(cid, &instr_quantize(8, 7, QUANTIZE_Q4_0)).is_err());
        assert!(vm.step_instruction(cid, &instr_quantize(8, 0, 12)).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[4, 8], DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_quantize(8, 7, QUANTIZE_Q4_0)).is_err());
        // DEQUANT: tipo sem decoder (Q4_1) e byte_len inconsistente.
        let aq1 = vm.memory.alloc_tensor(&[32], DType::F32).unwrap();
        vm.memory.as_cpu_mut().unwrap().tensor_meta.insert(aq1, crate::memory::TensorMeta {
            addr: aq1, shape: vec![32], dtype: DType::Q4_1, byte_len: 18, is_sparse: false, density: 1.0,
        });
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, aq1).unwrap();
        assert!(vm.step_instruction(cid, &instr_dequant(8, 7)).is_err());
        let aq2 = vm.memory.alloc_tensor(&[32], DType::F32).unwrap();
        vm.memory.as_cpu_mut().unwrap().tensor_meta.insert(aq2, crate::memory::TensorMeta {
            addr: aq2, shape: vec![32], dtype: DType::Q4_0, byte_len: 20, is_sparse: false, density: 1.0,
        });
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, aq2).unwrap();
        assert!(vm.step_instruction(cid, &instr_dequant(8, 7)).is_err());
        assert_eq!(vm.stats.quantize_execs, 3);
        assert_eq!(vm.stats.dequant_execs, 2);
    }

    #[tokio::test]
    async fn test_rfc0025_assembled_program_runs() {
        use crate::opcodes::assemble;
        // Pipeline combinado: CAST ida/volta + QUANTIZE + DEQUANT.
        let src = r#"
            TENSOR r0 8 4 f32 FILL=0.5
            CAST r1, r0 DST=F16
            CAST r2, r1 DST=F32
            QUANTIZE r3, r0 Q=Q8_0
            DEQUANT r4, r3
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.cast_execs, 2);
        assert_eq!(stats.quantize_execs, 1);
        assert_eq!(stats.dequant_execs, 1);
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let a2 = ctx.reg(2).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a2, 32).unwrap(), vec![0.5; 32]);
        let a4 = ctx.reg(4).unwrap();
        for v in vm.memory.read_f32_tensor(a4, 32).unwrap() {
            assert!((v - 0.5).abs() < 0.01, "v={}", v);
        }
    }

    #[tokio::test]
    async fn test_rfc0025_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/cast_quant_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.cast_execs, stats.quantize_execs, stats.dequant_execs), (2, 1, 1));
    }

    // ---- RFC-0026: ALU de control-plane -------------------------------

    #[test]
    fn test_rfc0026_alu_wrapping() {
        use crate::opcodes::{instr_add_imm, instr_sub_imm};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[(0, 100)]);
        vm.step_instruction(cid, &instr_add_imm(1, 0, 23)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 1) as u128, 123);
        vm.step_instruction(cid, &instr_sub_imm(2, 1, 23)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 2) as u128, 100);
        // Wrapping especificado: MAX+100 == 99; 100-101 == MAX.
        vm.step_instruction(cid, &instr_add_imm(3, 0, u128::MAX)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(3).unwrap(), 99);
        vm.step_instruction(cid, &instr_sub_imm(4, 0, 101)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(4).unwrap(), u128::MAX);
        vm.step_instruction(cid, &instr_sub_imm(5, 0, 100)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(5).unwrap(), 0);
        // 0xFF não é registrador.
        assert!(vm.step_instruction(cid, &instr_add_imm(0xFF, 0, 1)).is_err());
        assert!(vm.step_instruction(cid, &instr_add_imm(1, 0xFF, 1)).is_err());
        assert!(vm.step_instruction(cid, &instr_sub_imm(0xFF, 0, 1)).is_err());
        assert_eq!(vm.stats.add_imm_execs, 2);
        assert_eq!(vm.stats.sub_imm_execs, 3);
    }

    #[tokio::test]
    async fn test_rfc0026_steps_exact() {
        use crate::opcodes::assemble;
        // O incremento é pós-execute: STEPS lê as COMPLETADAS.
        let src = "NOP\nSTEPS r0\nNOP\nSTEPS r1\nHALT\n";
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        vm.load_program(prog);
        vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(0).unwrap(), 1);
        assert_eq!(ctx.reg(1).unwrap(), 3);
        assert_eq!(vm.stats.steps_execs, 2);
    }

    #[tokio::test]
    async fn test_rfc0026_assembled_program_runs() {
        use crate::opcodes::assemble;
        let src = r#"
            LOADI r0, 100
            ADD_IMM r1, r0 IMM=23
            SUB_IMM r2, r1 IMM=23
            STEPS r3
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.add_imm_execs, stats.sub_imm_execs, stats.steps_execs), (1, 1, 1));
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(1).unwrap(), 123);
        assert_eq!(ctx.reg(2).unwrap(), 100);
        assert_eq!(ctx.reg(3).unwrap(), 3);
    }

    #[tokio::test]
    async fn test_rfc0026_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/alu_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.add_imm_execs, stats.sub_imm_execs, stats.steps_execs), (1, 1, 1));
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!((ctx.reg(1).unwrap(), ctx.reg(2).unwrap(), ctx.reg(3).unwrap()), (123, 100, 3));
    }

    // ---- RFC-0027: forma ------------------------------------------------

    #[test]
    fn test_rfc0027_sort_topk_argmax() {
        use crate::opcodes::{instr_argmax, instr_sort, instr_topk, SORT_DESC};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[2, 4], &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0)]);
        // SORT DESC por linha.
        vm.step_instruction(cid, &instr_sort(1, 0, 1, SORT_DESC)).unwrap();
        let s = rfc0005_reg_u64(&vm, cid, 1) as u128;
        assert_eq!(vm.memory.read_f32_tensor(s, 8).unwrap(), vec![2.0, 1.5, 1.0, 0.5, 4.0, 3.5, 3.0, 2.5]);
        // SORT ASC por coluna (já ordenado => idêntico).
        vm.step_instruction(cid, &instr_sort(2, 0, 0, 0)).unwrap();
        let s0 = rfc0005_reg_u64(&vm, cid, 2) as u128;
        assert_eq!(vm.memory.read_f32_tensor(s0, 8).unwrap(), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
        // TOPK k=2 largest sorted: [vals | idx].
        vm.step_instruction(cid, &instr_topk(3, 0, 1, 2, 1, 1)).unwrap();
        let t = rfc0005_reg_u64(&vm, cid, 3) as u128;
        assert_eq!(vm.memory.read_f32_tensor(t, 8).unwrap(), vec![2.0, 1.5, 3.0, 2.0, 4.0, 3.5, 3.0, 2.0]);
        // TOPK smallest unsorted: ordem original dos índices.
        vm.step_instruction(cid, &instr_topk(4, 0, 1, 2, 0, 0)).unwrap();
        let t4 = rfc0005_reg_u64(&vm, cid, 4) as u128;
        assert_eq!(vm.memory.read_f32_tensor(t4, 8).unwrap(), vec![0.5, 1.0, 0.0, 1.0, 2.5, 3.0, 0.0, 1.0]);
        // ARGMAX por linha.
        vm.step_instruction(cid, &instr_argmax(5, 0, 1)).unwrap();
        let am = rfc0005_reg_u64(&vm, cid, 5) as u128;
        assert_eq!(vm.memory.read_f32_tensor(am, 2).unwrap(), vec![3.0, 3.0]);
        // NaN: ASC põe por último; ARGMAX ignora (maxNum); tudo-NaN => idx 0.
        let an = rfc0004_f32(&mut vm, &[3], &[3.0, f32::NAN, 1.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, an).unwrap();
        vm.step_instruction(cid, &instr_sort(7, 6, 0, 0)).unwrap();
        let sn = rfc0005_reg_u64(&vm, cid, 7) as u128;
        let got = vm.memory.read_f32_tensor(sn, 3).unwrap();
        assert_eq!(got[0], 1.0);
        assert_eq!(got[1], 3.0);
        assert!(got[2].is_nan());
        vm.step_instruction(cid, &instr_argmax(8, 6, 0)).unwrap();
        let an2 = rfc0005_reg_u64(&vm, cid, 8) as u128;
        assert_eq!(vm.memory.read_f32_tensor(an2, 1).unwrap(), vec![0.0]);
        let allnan = rfc0004_f32(&mut vm, &[2], &[f32::NAN, f32::NAN]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, allnan).unwrap();
        vm.step_instruction(cid, &instr_argmax(8, 6, 0)).unwrap();
        let an3 = rfc0005_reg_u64(&vm, cid, 8) as u128;
        assert_eq!(vm.memory.read_f32_tensor(an3, 1).unwrap(), vec![0.0]);
        // Erros: eixo fora, K=0, K>dim, esparso, meta ausente.
        assert!(vm.step_instruction(cid, &instr_sort(1, 0, 5, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_topk(1, 0, 1, 0, 1, 1)).is_err());
        assert!(vm.step_instruction(cid, &instr_topk(1, 0, 1, 5, 1, 1)).is_err());
        assert!(vm.step_instruction(cid, &instr_argmax(1, 0, 2)).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[2, 4], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(9, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_sort(1, 9, 0, 0)).is_err());
        vm.scheduler.get_mut(cid).unwrap().set_reg(9, 0xdead).unwrap();
        assert!(vm.step_instruction(cid, &instr_topk(1, 9, 0, 1, 1, 1)).is_err());
        assert_eq!(vm.stats.sort_execs, 3);
        assert_eq!(vm.stats.topk_execs, 2);
        assert_eq!(vm.stats.argmax_execs, 3);
    }

    #[test]
    fn test_rfc0027_reduce() {
        use crate::opcodes::{instr_reduce, REDUCE_SUM, REDUCE_MEAN, REDUCE_MAX, REDUCE_MIN, REDUCE_PROD};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0)]);
        let rd = |vm: &Vm, r: u8, n: usize| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, n).unwrap();
        // Total: SUM=36, MEAN=4.5, PROD=40320, MAX=8, MIN=1.
        vm.step_instruction(cid, &instr_reduce(1, 0, REDUCE_SUM, 0xFF)).unwrap();
        assert_eq!(rd(&vm, 1, 1), vec![36.0]);
        vm.step_instruction(cid, &instr_reduce(1, 0, REDUCE_MEAN, 0xFF)).unwrap();
        assert_eq!(rd(&vm, 1, 1), vec![4.5]);
        vm.step_instruction(cid, &instr_reduce(1, 0, REDUCE_PROD, 0xFF)).unwrap();
        assert_eq!(rd(&vm, 1, 1), vec![40320.0]);
        vm.step_instruction(cid, &instr_reduce(1, 0, REDUCE_MAX, 0xFF)).unwrap();
        assert_eq!(rd(&vm, 1, 1), vec![8.0]);
        vm.step_instruction(cid, &instr_reduce(1, 0, REDUCE_MIN, 0xFF)).unwrap();
        assert_eq!(rd(&vm, 1, 1), vec![1.0]);
        // Por eixo: SUM eixo 0 => [6,8,10,12]; MEAN eixo 1 => [2.5,6.5].
        vm.step_instruction(cid, &instr_reduce(2, 0, REDUCE_SUM, 0)).unwrap();
        assert_eq!(rd(&vm, 2, 4), vec![6.0, 8.0, 10.0, 12.0]);
        vm.step_instruction(cid, &instr_reduce(3, 0, REDUCE_MEAN, 1)).unwrap();
        assert_eq!(rd(&vm, 3, 2), vec![2.5, 6.5]);
        // NaN: MAX pula (2.0), tudo-NaN => NaN.
        let an = rfc0004_f32(&mut vm, &[2], &[f32::NAN, 2.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, an).unwrap();
        vm.step_instruction(cid, &instr_reduce(5, 4, REDUCE_MAX, 0xFF)).unwrap();
        assert_eq!(rd(&vm, 5, 1), vec![2.0]);
        let allnan = rfc0004_f32(&mut vm, &[2], &[f32::NAN, f32::NAN]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, allnan).unwrap();
        vm.step_instruction(cid, &instr_reduce(5, 4, REDUCE_MAX, 0xFF)).unwrap();
        assert!(rd(&vm, 5, 1)[0].is_nan());
        // Erros: OP ruim, eixo fora, esparso.
        let mut bad = instr_reduce(1, 0, REDUCE_SUM, 0xFF);
        bad.set_reduce_params(9, 0xFF);
        assert!(vm.step_instruction(cid, &bad).is_err());
        assert!(vm.step_instruction(cid, &instr_reduce(1, 0, REDUCE_SUM, 3)).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[2, 4], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_reduce(1, 4, REDUCE_SUM, 0xFF)).is_err());
        assert_eq!(vm.stats.reduce_execs, 9);
    }

    #[test]
    fn test_rfc0027_broadcast_pad_tile_transpose() {
        use crate::opcodes::{instr_broadcast, instr_pad, instr_tile, instr_transpose};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let a1 = rfc0004_f32(&mut vm, &[4], &[1.0, 2.0, 3.0, 4.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0), (1, a1)]);
        let rd = |vm: &Vm, r: u8, n: usize| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, n).unwrap();
        // BROADCAST [4] -> [2,4]: linhas repetidas.
        vm.step_instruction(cid, &instr_broadcast(2, 1, &[2, 4])).unwrap();
        assert_eq!(rd(&vm, 2, 8), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        // BROADCAST [1] -> [2,2]: constante.
        let aone = rfc0004_f32(&mut vm, &[1], &[7.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, aone).unwrap();
        vm.step_instruction(cid, &instr_broadcast(4, 3, &[2, 2])).unwrap();
        assert_eq!(rd(&vm, 4, 4), vec![7.0; 4]);
        // Incompatível e afunilamento vetam.
        let a23 = rfc0004_f32(&mut vm, &[2, 3], &[0.0; 6]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, a23).unwrap();
        assert!(vm.step_instruction(cid, &instr_broadcast(4, 0, &[2, 3])).is_err());
        assert!(vm.step_instruction(cid, &instr_broadcast(4, 0, &[2])).is_err());
        // PAD eixo 1: [2,2] +1/+1 com 9.0.
        vm.step_instruction(cid, &instr_pad(5, 0, 9.0, 1, 1, 1)).unwrap();
        assert_eq!(rd(&vm, 5, 8), vec![9.0, 1.0, 2.0, 9.0, 9.0, 3.0, 4.0, 9.0]);
        // PAD no-op e eixo fora vetam.
        assert!(vm.step_instruction(cid, &instr_pad(5, 0, 9.0, 1, 0, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_pad(5, 0, 9.0, 5, 1, 1)).is_err());
        // TILE eixo 0 x2: bloco do eixo repetido ([1,2],[3,4] ×2).
        vm.step_instruction(cid, &instr_tile(6, 0, 2, 0)).unwrap();
        assert_eq!(rd(&vm, 6, 8), vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
        assert!(vm.step_instruction(cid, &instr_tile(6, 0, 0, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_tile(6, 0, 2, 7)).is_err());
        // TRANSPOSE 2-D + 3-D pontual.
        vm.step_instruction(cid, &instr_transpose(7, 0, &[1, 0])).unwrap();
        assert_eq!(rd(&vm, 7, 4), vec![1.0, 3.0, 2.0, 4.0]);
        let a222 = rfc0004_f32(&mut vm, &[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, a222).unwrap();
        vm.step_instruction(cid, &instr_transpose(8, 3, &[2, 0, 1])).unwrap();
        // numpy: in_coord[perm[i]] = out_coord[i], i.e. out[a,b,c]=in[b,c,a].
        assert_eq!(rd(&vm, 8, 8), vec![1.0, 3.0, 5.0, 7.0, 2.0, 4.0, 6.0, 8.0]);
        // Perm inválida: repetida, fora, ndim != rank.
        assert!(vm.step_instruction(cid, &instr_transpose(8, 0, &[0, 0])).is_err());
        assert!(vm.step_instruction(cid, &instr_transpose(8, 0, &[2, 0])).is_err());
        assert!(vm.step_instruction(cid, &instr_transpose(8, 0, &[0])).is_err());
        assert_eq!(vm.stats.broadcast_execs, 2);
        assert_eq!(vm.stats.pad_execs, 1);
        assert_eq!(vm.stats.tile_execs, 1);
        assert_eq!(vm.stats.transpose_execs, 2);
    }

    #[tokio::test]
    async fn test_rfc0027_assembled_program_runs() {
        use crate::opcodes::assemble;
        // Pipeline combinado: as 8 formas numa passada (rampa 0.5..4.0).
        let src = r#"
            TENSOR r0 2 4 f32
            SORT r1, r0 AXIS=1 ORDER=DESC
            TOPK r2, r0 K=2 AXIS=1
            ARGMAX r3, r0 AXIS=1
            REDUCE r4, r0 OP=SUM
            BROADCAST r5, r4 SHAPE=2x2
            PAD r6, r0 VALUE=0 AXIS=1 BEFORE=1 AFTER=1
            TILE r7, r0 REPS=2 AXIS=0
            TRANSPOSE r8, r0 AXES=1x0
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.sort_execs, stats.topk_execs, stats.argmax_execs, stats.reduce_execs), (1, 1, 1, 1));
        assert_eq!((stats.broadcast_execs, stats.pad_execs, stats.tile_execs, stats.transpose_execs), (1, 1, 1, 1));
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let rd = |r: u8, n: usize| vm.memory.read_f32_tensor(ctx.reg(r).unwrap(), n).unwrap();
        assert_eq!(rd(1, 8), vec![2.0, 1.5, 1.0, 0.5, 4.0, 3.5, 3.0, 2.5]);
        assert_eq!(rd(2, 8), vec![2.0, 1.5, 3.0, 2.0, 4.0, 3.5, 3.0, 2.0]);
        assert_eq!(rd(3, 2), vec![3.0, 3.0]);
        assert_eq!(rd(4, 1), vec![18.0]);
        assert_eq!(rd(5, 4), vec![18.0; 4]);
        assert_eq!(rd(6, 12), vec![0.0, 0.5, 1.0, 1.5, 2.0, 0.0, 0.0, 2.5, 3.0, 3.5, 4.0, 0.0]);
        assert_eq!(rd(7, 16), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
        assert_eq!(rd(8, 8), vec![0.5, 2.5, 1.0, 3.0, 1.5, 3.5, 2.0, 4.0]);
    }

    #[tokio::test]
    async fn test_rfc0027_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/shape_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.sort_execs, stats.topk_execs, stats.argmax_execs, stats.reduce_execs), (1, 1, 1, 1));
        assert_eq!((stats.broadcast_execs, stats.pad_execs, stats.tile_execs, stats.transpose_execs), (1, 1, 1, 1));
    }

    // ---- RFC-0028: ativações ------------------------------------------

    #[test]
    fn test_rfc0028_unary_goldens() {
        use crate::opcodes::{instr_clip, instr_exp, instr_gelu, instr_log, instr_relu, instr_sigmoid, instr_tanh};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(80), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[4], &[0.0, 1.0, -1.0, 2.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0)]);
        let rd = |vm: &Vm, r: u8| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, 4).unwrap();
        let close = |got: &[f32], exp: &[f32], eps: f32| {
            for (g, e) in got.iter().zip(exp.iter()) {
                assert!((g - e).abs() <= eps, "got {:?} exp {:?} eps {}", got, exp, eps);
            }
        };
        vm.step_instruction(cid, &instr_gelu(1, 0)).unwrap();
        close(&rd(&vm, 1), &[0.0, 0.8413447, -0.1586553, 1.9544997], 1e-5);
        vm.step_instruction(cid, &instr_sigmoid(2, 0)).unwrap();
        close(&rd(&vm, 2), &[0.5, 0.7310586, 0.2689414, 0.8807971], 1e-6);
        vm.step_instruction(cid, &instr_tanh(3, 0)).unwrap();
        close(&rd(&vm, 3), &[0.0, 0.7615942, -0.7615942, 0.9640276], 1e-6);
        vm.step_instruction(cid, &instr_relu(4, 0)).unwrap();
        assert_eq!(rd(&vm, 4), vec![0.0, 1.0, 0.0, 2.0]);
        // -0.0 normaliza para +0.0 (bits zero).
        let an = rfc0004_f32(&mut vm, &[1], &[-0.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, an).unwrap();
        vm.step_instruction(cid, &instr_relu(6, 5)).unwrap();
        let r6 = rfc0005_reg_u64(&vm, cid, 6) as u128;
        assert_eq!(vm.memory.read(r6, 4).unwrap(), vec![0u8; 4]);
        vm.step_instruction(cid, &instr_exp(7, 0)).unwrap();
        close(&rd(&vm, 7), &[1.0, 2.7182818, 0.3678795, 7.389056], 1e-5);
        let al = rfc0004_f32(&mut vm, &[3], &[1.0, 2.7182818, 0.5]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, al).unwrap();
        vm.step_instruction(cid, &instr_log(8, 5)).unwrap();
        let r8 = rfc0005_reg_u64(&vm, cid, 8) as u128;
        close(&vm.memory.read_f32_tensor(r8, 3).unwrap(), &[0.0, 1.0, -0.6931472], 1e-5);
        vm.step_instruction(cid, &instr_clip(9, 0, -0.5, 1.5)).unwrap();
        assert_eq!(rd(&vm, 9), vec![0.0, 1.0, -0.5, 1.5]);
        // NaN: GELU/SIGMOID/TANH/EXP/LOG propagam; RELU => +0.0; CLIP => max.
        let anan = rfc0004_f32(&mut vm, &[1], &[f32::NAN]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, anan).unwrap();
        vm.step_instruction(cid, &instr_gelu(1, 5)).unwrap();
        assert!(rd(&vm, 1)[0].is_nan());
        vm.step_instruction(cid, &instr_sigmoid(1, 5)).unwrap();
        assert!(rd(&vm, 1)[0].is_nan());
        vm.step_instruction(cid, &instr_tanh(1, 5)).unwrap();
        assert!(rd(&vm, 1)[0].is_nan());
        vm.step_instruction(cid, &instr_relu(1, 5)).unwrap();
        assert_eq!(vm.memory.read(rfc0005_reg_u64(&vm, cid, 1) as u128, 4).unwrap(), vec![0u8; 4]);
        vm.step_instruction(cid, &instr_exp(1, 5)).unwrap();
        assert!(rd(&vm, 1)[0].is_nan());
        vm.step_instruction(cid, &instr_log(1, 5)).unwrap();
        assert!(rd(&vm, 1)[0].is_nan());
        vm.step_instruction(cid, &instr_clip(1, 5, 0.0, 1.0)).unwrap();
        let rc = rfc0005_reg_u64(&vm, cid, 1) as u128;
        assert!(vm.memory.read_f32_tensor(rc, 1).unwrap()[0].is_nan());
        // Erros: esparso, meta ausente, não-F32, rdest 0xFF, bounds.
        let asp = vm.memory.alloc_sparse_tensor(&[2, 2], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_gelu(1, 5)).is_err());
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, 0xdead).unwrap();
        assert!(vm.step_instruction(cid, &instr_exp(1, 5)).is_err());
        let af = vm.memory.alloc_tensor(&[4], crate::memory::DType::F16).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, af).unwrap();
        assert!(vm.step_instruction(cid, &instr_tanh(1, 5)).is_err());
        assert!(vm.step_instruction(cid, &instr_relu(0xFF, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_clip(1, 0, 2.0, 1.0)).is_err());
        let mut bad = instr_clip(1, 0, 0.0, 1.0);
        bad.set_clip_params(f32::NAN, 1.0);
        assert!(vm.step_instruction(cid, &bad).is_err());
        assert_eq!(vm.stats.gelu_execs, 2);
        assert_eq!(vm.stats.sigmoid_execs, 2);
        assert_eq!(vm.stats.tanh_execs, 2);
        assert_eq!(vm.stats.relu_execs, 3);
        assert_eq!(vm.stats.exp_execs, 2);
        assert_eq!(vm.stats.log_execs, 2);
        assert_eq!(vm.stats.clip_execs, 2);
    }

    #[test]
    fn test_rfc0028_softmax() {
        use crate::opcodes::instr_softmax;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let a0 = rfc0004_f32(&mut vm, &[3], &[1.0, 2.0, 3.0]);
        let au = rfc0004_f32(&mut vm, &[4], &[0.0; 4]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a0), (1, au)]);
        let rd = |vm: &Vm, r: u8, n: usize| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, n).unwrap();
        // Vetor conhecido + uniforme exato.
        vm.step_instruction(cid, &instr_softmax(2, 0, 0xFF, 1.0)).unwrap();
        let got = rd(&vm, 2, 3);
        assert!((got[0] - 0.0900306).abs() < 1e-5, "{:?}", got);
        assert!((got[1] - 0.2447285).abs() < 1e-5, "{:?}", got);
        assert!((got[2] - 0.6652410).abs() < 1e-5, "{:?}", got);
        vm.step_instruction(cid, &instr_softmax(3, 1, 0xFF, 1.0)).unwrap();
        assert_eq!(rd(&vm, 3, 4), vec![0.25; 4]);
        // Temperatura: afia e achata.
        vm.step_instruction(cid, &instr_softmax(4, 0, 0xFF, 0.1)).unwrap();
        assert!(rd(&vm, 4, 3)[2] > 0.999);
        vm.step_instruction(cid, &instr_softmax(5, 0, 0xFF, 100.0)).unwrap();
        for v in rd(&vm, 5, 3) {
            assert!((v - 1.0 / 3.0).abs() < 0.01);
        }
        // Por eixo: [2,2] AXIS=0 normaliza colunas.
        let a2 = rfc0004_f32(&mut vm, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, a2).unwrap();
        vm.step_instruction(cid, &instr_softmax(7, 6, 0, 1.0)).unwrap();
        let got7 = rd(&vm, 7, 4);
        assert!((got7[0] - 0.1192029).abs() < 1e-5, "{:?}", got7);
        assert!((got7[1] - 0.1192029).abs() < 1e-5, "{:?}", got7);
        assert!((got7[2] - 0.8807971).abs() < 1e-5, "{:?}", got7);
        assert!((got7[3] - 0.8807971).abs() < 1e-5, "{:?}", got7);
        // NaN envenena a lane (documentado).
        let an = rfc0004_f32(&mut vm, &[2], &[1.0, f32::NAN]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, an).unwrap();
        vm.step_instruction(cid, &instr_softmax(7, 6, 0xFF, 1.0)).unwrap();
        assert!(rd(&vm, 7, 2).iter().all(|x| x.is_nan()));
        // Erros: TEMP 0/neg/NaN, eixo fora, esparso.
        assert!(vm.step_instruction(cid, &instr_softmax(2, 0, 0xFF, 0.0)).is_err());
        assert!(vm.step_instruction(cid, &instr_softmax(2, 0, 0xFF, -1.0)).is_err());
        assert!(vm.step_instruction(cid, &instr_softmax(2, 0, 0xFF, f32::NAN)).is_err());
        assert!(vm.step_instruction(cid, &instr_softmax(2, 0, 5, 1.0)).is_err());
        assert_eq!(vm.stats.softmax_execs, 6);
    }

    #[tokio::test]
    async fn test_rfc0028_assembled_program_runs() {
        use crate::opcodes::assemble;
        let src = r#"
            TENSOR r0 2 2 f32 FILL=0.5
            SIGMOID r1, r0
            TANH r2, r0
            RELU r3, r0
            GELU r4, r0
            EXP r5, r0
            LOG r6, r5
            SOFTMAX r7, r0 AXIS=1
            CLIP r8, r0 MIN=0.0 MAX=0.4
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.sigmoid_execs, stats.tanh_execs, stats.relu_execs, stats.gelu_execs), (1, 1, 1, 1));
        assert_eq!((stats.exp_execs, stats.log_execs, stats.softmax_execs, stats.clip_execs), (1, 1, 1, 1));
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let rd = |r: u8| vm.memory.read_f32_tensor(ctx.reg(r).unwrap(), 4).unwrap();
        for v in rd(1) {
            assert!((v - 0.6224594).abs() < 1e-6, "sigmoid(0.5)");
        }
        for v in rd(2) {
            assert!((v - 0.4621172).abs() < 1e-6, "tanh(0.5)");
        }
        assert_eq!(rd(3), vec![0.5; 4]);
        for v in rd(4) {
            assert!((v - 0.3457312).abs() < 1e-5, "gelu(0.5)");
        }
        for v in rd(6) {
            assert!((v - 0.5).abs() < 1e-6, "ln(exp(0.5))");
        }
        // Lanes de 2 (eixo 1 de [2,2]): softmax([0.5,0.5]) = [0.5,0.5].
        assert_eq!(rd(7), vec![0.5; 4]);
        assert_eq!(rd(8), vec![0.4; 4]);
    }

    #[tokio::test]
    async fn test_rfc0028_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/activation_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.sigmoid_execs, stats.tanh_execs, stats.relu_execs, stats.gelu_execs), (1, 1, 1, 1));
        assert_eq!((stats.exp_execs, stats.log_execs, stats.softmax_execs, stats.clip_execs), (1, 1, 1, 1));
    }

    // ---- RFC-0029: KV/attention ---------------------------------------

    #[test]
    fn test_rfc0029_kv_compress() {
        use crate::opcodes::instr_kv_compress;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.memory.kv_cache_init(2, 4);
        for t in 0..10u32 {
            for layer in 0..2usize {
                let k = vec![(layer as u32 * 100 + t) as f32; 4];
                let v = vec![(layer as u32 * 1000 + t) as f32; 4];
                vm.memory.kv_cache_append(layer, &k, &v).unwrap();
            }
        }
        assert_eq!(vm.memory.kv_cache_seq_len(), 10);
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        let row = |vm: &mut Vm, l: usize, p: usize| {
            vm.memory.as_cpu_mut().unwrap().kv_cache_row(l, p).unwrap()
        };
        // SINK=2 WINDOW=3: mantém pos 0,1 + 7,8,9.
        vm.step_instruction(cid, &instr_kv_compress(2, 3, 0, 0)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 5);
        assert_eq!(row(&mut vm, 0, 0).0, vec![0.0; 4]);
        assert_eq!(row(&mut vm, 0, 1).0, vec![1.0; 4]);
        assert_eq!(row(&mut vm, 0, 2).0, vec![7.0; 4]);
        assert_eq!(row(&mut vm, 0, 4).1, vec![9.0; 4]);
        assert_eq!(row(&mut vm, 1, 0).0, vec![100.0; 4]);
        assert_eq!(row(&mut vm, 1, 4).1, vec![1009.0; 4]);
        // No-op quando já cabe (chamada por passo não deve falhar).
        vm.step_instruction(cid, &instr_kv_compress(5, 5, 0, 0)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 5);
        // Erros: degenerado, modo; stream roteia (RFC-0033).
        assert!(vm.step_instruction(cid, &instr_kv_compress(0, 0, 0, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_kv_compress(2, 3, 9, 0)).is_err());
        // STREAM=1 roteia p/ stream inexistente: no-op Ok (RFC-0033).
        vm.step_instruction(cid, &instr_kv_compress(2, 3, 0, 1)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len_stream(1), 0);
        // sid > 16 veta alto.
        assert!(vm.step_instruction(cid, &instr_kv_compress(2, 3, 0, 99)).is_err());
        // +1 sucesso (no-op roteado acima).
        // Rollback: snapshot -> append -> compress -> restore volta tudo.
        let snap = vm.memory.snapshot();
        for layer in 0..2usize {
            vm.memory.kv_cache_append(layer, &[50.0; 4], &[60.0; 4]).unwrap();
            vm.memory.kv_cache_append(layer, &[51.0; 4], &[61.0; 4]).unwrap();
        }
        assert_eq!(vm.memory.kv_cache_seq_len(), 7);
        vm.step_instruction(cid, &instr_kv_compress(1, 1, 0, 0)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 2);
        vm.memory.restore(snap).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 5);
        assert_eq!(row(&mut vm, 0, 2).0, vec![7.0; 4]);
        assert_eq!(vm.stats.kv_compress_execs, 4);
    }

    #[test]
    fn test_rfc0029_flash_attn() {
        use crate::opcodes::{instr_attn, instr_flash_attn};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let q = rfc0004_f32(&mut vm, &[2, 4], &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
        let k = rfc0004_f32(&mut vm, &[4, 4], &[0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.5]);
        let v = rfc0004_f32(&mut vm, &[4, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 2.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, q), (1, k), (2, v)]);
        let rd = |vm: &Vm, r: u8| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, 4).unwrap();
        // Referência densa + flash em blocos: tolerância, nunca bits.
        vm.step_instruction(cid, &instr_attn(10, 0, 1, 2)).unwrap();
        let aref = rd(&vm, 10);
        vm.step_instruction(cid, &instr_flash_attn(11, 0, 1, 2, 2)).unwrap();
        let got2 = rd(&vm, 11);
        for (a, b) in aref.iter().zip(got2.iter()) {
            assert!((a - b).abs() <= 1e-4, "attn={:?} flash={:?}", aref, got2);
        }
        // Invariância de tiling: BLOCK=default vs 2 vs 64.
        vm.step_instruction(cid, &instr_flash_attn(12, 0, 1, 2, 0)).unwrap();
        vm.step_instruction(cid, &instr_flash_attn(13, 0, 1, 2, 64)).unwrap();
        let g0 = rd(&vm, 12);
        let g64 = rd(&vm, 13);
        for (a, b) in got2.iter().zip(g0.iter()).chain(got2.iter().zip(g64.iter())) {
            assert!((a - b).abs() <= 1e-4);
        }
        // Erros: incompatível, esparso, flags.
        let kbad = rfc0004_f32(&mut vm, &[4, 3], &[0.0; 12]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, kbad).unwrap();
        assert!(vm.step_instruction(cid, &instr_flash_attn(11, 0, 3, 2, 2)).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[2, 4], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_flash_attn(11, 3, 1, 2, 2)).is_err());
        let mut flagged = instr_flash_attn(11, 0, 1, 2, 2);
        flagged.flags = 1;
        assert!(vm.step_instruction(cid, &flagged).is_err());
        assert_eq!(vm.stats.flash_attn_execs, 3);
    }

    #[test]
    fn test_rfc0029_attn_sparse() {
        use crate::opcodes::{instr_attn, instr_attn_sparse, SPARSE_METRIC_DOT};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        // K quase-ortogonal escalado: seleção top-2 óbvia por linha.
        let q = rfc0004_f32(&mut vm, &[1, 4], &[1.0, 0.0, 0.0, 0.0]);
        let k = rfc0004_f32(&mut vm, &[4, 4], &[10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 10.0]);
        let v = rfc0004_f32(&mut vm, &[4, 2], &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, q), (1, k), (2, v)]);
        // scores/2 = [5,0,0,0]: top-2 = idx {0,1}; softmax = [w0,w1].
        vm.step_instruction(cid, &instr_attn_sparse(3, 0, 1, 2, SPARSE_METRIC_DOT, 2)).unwrap();
        let got = vm.memory.read_f32_tensor(rfc0005_reg_u64(&vm, cid, 3) as u128, 2).unwrap();
        let e5 = 5.0f64.exp();
        let w0 = e5 / (e5 + 1.0);
        let w1 = 1.0 / (e5 + 1.0);
        let e0 = w0 * 0.0 + w1 * 10.0;
        let e1 = w0 * 1.0 + w1 * 11.0;
        assert!((got[0] as f64 - e0).abs() < 1e-4, "{:?} vs [{},{}]", got, e0, e1);
        assert!((got[1] as f64 - e1).abs() < 1e-4, "{:?} vs [{},{}]", got, e0, e1);
        // TOPK=0 e k=N: mesmo kernel, tolerância vs ATTN (não bits).
        vm.step_instruction(cid, &instr_attn(4, 0, 1, 2)).unwrap();
        let aref = vm.memory.read_f32_tensor(rfc0005_reg_u64(&vm, cid, 4) as u128, 2).unwrap();
        vm.step_instruction(cid, &instr_attn_sparse(5, 0, 1, 2, SPARSE_METRIC_DOT, 0)).unwrap();
        vm.step_instruction(cid, &instr_attn_sparse(6, 0, 1, 2, SPARSE_METRIC_DOT, 4)).unwrap();
        for r in [5u8, 6u8] {
            let g = vm.memory.read_f32_tensor(rfc0005_reg_u64(&vm, cid, r) as u128, 2).unwrap();
            for (a, b) in aref.iter().zip(g.iter()) {
                assert!((a - b).abs() <= 1e-4, "attn={:?} sparse={:?}", aref, g);
            }
        }
        // Erros: k > N, métrica, esparso.
        assert!(vm.step_instruction(cid, &instr_attn_sparse(3, 0, 1, 2, SPARSE_METRIC_DOT, 5)).is_err());
        assert!(vm.step_instruction(cid, &instr_attn_sparse(3, 0, 1, 2, 1, 2)).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[1, 4], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_attn_sparse(3, 7, 1, 2, SPARSE_METRIC_DOT, 2)).is_err());
        assert_eq!(vm.stats.attn_sparse_execs, 3);
    }

    #[tokio::test]
    async fn test_rfc0029_assembled_program_runs() {
        use crate::opcodes::assemble;
        let src = r#"
            TENSOR r0 2 4 f32
            TENSOR r1 4 4 f32
            TENSOR r2 4 2 f32
            KV_COMPRESS SINK=2 WINDOW=3
            ATTN_SPARSE r3, r0, r1, r2 TOPK=2
            FLASH_ATTN r4, r0, r1, r2 BLOCK=2
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.kv_compress_execs, stats.attn_sparse_execs, stats.flash_attn_execs), (1, 1, 1));
    }

    #[tokio::test]
    async fn test_rfc0029_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/sparse_attn_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.kv_compress_execs, stats.attn_sparse_execs, stats.flash_attn_execs), (1, 1, 1));
    }

    // ---- RFC-0030: Trilha P (lowerings executáveis) --------------------

    #[tokio::test]
    async fn test_rfc0030_lowering_runs() {
        use crate::opcodes::assemble;
        // Newton p/ 1/sqrt(2) sem DIV: x1 = 0.5*(3-2*0.25)/2 = 0.625 exato.
        let src = include_str!("../programs/newton_rsqrt_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let a10 = ctx.reg(10).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a10, 1).unwrap(), vec![0.625]);
        // Lloyd k=2 desenrolado: q0==c0 -> idx 0; q1==c1 -> idx 1.
        let src = include_str!("../programs/kmeans_assign_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm2.load_program(prog);
        vm2.run().unwrap();
        let ctx2 = vm2.scheduler.get(1).unwrap().clone();
        let a4 = ctx2.reg(4).unwrap();
        assert_eq!(vm2.memory.read_f32_tensor(a4, 1).unwrap(), vec![0.0]);
        let a7 = ctx2.reg(7).unwrap();
        assert_eq!(vm2.memory.read_f32_tensor(a7, 1).unwrap(), vec![1.0]);
    }

    // ---- RFC-0031: DSP de áudio -----------------------------------------

    #[test]
    fn test_rfc0031_vad() {
        use crate::opcodes::{instr_vad_detect, VAD_MODE_ENERGY, VAD_MODE_ML, VAD_MODE_ZCR};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let adc = rfc0004_f32(&mut vm, &[8], &[0.5; 8]);
        let azero = rfc0004_f32(&mut vm, &[8], &[0.0; 8]);
        let aalt = rfc0004_f32(&mut vm, &[4], &[1.0, -1.0, 1.0, -1.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, adc), (1, azero), (2, aalt)]);
        let rd1 = |vm: &Vm, r: u8| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, 1).unwrap()[0];
        // DC 0.5: ENERGY = RMS = 0.5 exato; ZCR = 0 (sem troca de sinal).
        vm.step_instruction(cid, &instr_vad_detect(3, 0, VAD_MODE_ENERGY)).unwrap();
        assert_eq!(rd1(&vm, 3), 0.5);
        vm.step_instruction(cid, &instr_vad_detect(3, 0, VAD_MODE_ZCR)).unwrap();
        assert_eq!(rd1(&vm, 3), 0.0);
        // Silêncio: 0 e 0.
        vm.step_instruction(cid, &instr_vad_detect(3, 1, VAD_MODE_ENERGY)).unwrap();
        assert_eq!(rd1(&vm, 3), 0.0);
        vm.step_instruction(cid, &instr_vad_detect(3, 1, VAD_MODE_ZCR)).unwrap();
        assert_eq!(rd1(&vm, 3), 0.0);
        // Alternado: ENERGY = 1.0, ZCR = 3/3 = 1.0.
        vm.step_instruction(cid, &instr_vad_detect(3, 2, VAD_MODE_ENERGY)).unwrap();
        assert_eq!(rd1(&vm, 3), 1.0);
        vm.step_instruction(cid, &instr_vad_detect(3, 2, VAD_MODE_ZCR)).unwrap();
        assert_eq!(rd1(&vm, 3), 1.0);
        // Senoide 440Hz real: RMS ≈ 0.5/√2; ZCR na faixa teórica.
        let frame = crate::mimi::synth_frame_440hz();
        let n = frame.len();
        let asine = vm.memory.alloc_tensor(&[n], crate::memory::DType::F32).unwrap();
        vm.memory.write_f32_tensor(asine, &frame).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, asine).unwrap();
        vm.step_instruction(cid, &instr_vad_detect(3, 4, VAD_MODE_ENERGY)).unwrap();
        assert!((rd1(&vm, 3) - 0.3536).abs() < 5e-3, "rms={}", rd1(&vm, 3));
        vm.step_instruction(cid, &instr_vad_detect(3, 4, VAD_MODE_ZCR)).unwrap();
        let z = rd1(&vm, 3);
        assert!(z > 0.03 && z < 0.05, "zcr={}", z);
        // Erros: ML (parseia, veta), modo ruim, esparso, meta ausente, n=1 p/ ZCR.
        assert!(vm.step_instruction(cid, &instr_vad_detect(3, 0, VAD_MODE_ML)).is_err());
        let mut bad = instr_vad_detect(3, 0, VAD_MODE_ENERGY);
        bad.set_vad_mode(9);
        assert!(vm.step_instruction(cid, &bad).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[2, 4], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_vad_detect(3, 4, VAD_MODE_ENERGY)).is_err());
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, 0xdead).unwrap();
        assert!(vm.step_instruction(cid, &instr_vad_detect(3, 4, VAD_MODE_ENERGY)).is_err());
        let a1 = rfc0004_f32(&mut vm, &[1], &[0.5]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, a1).unwrap();
        assert!(vm.step_instruction(cid, &instr_vad_detect(3, 4, VAD_MODE_ZCR)).is_err());
        assert_eq!(vm.stats.vad_detect_execs, 8);
    }

    #[test]
    fn test_rfc0031_merge_resample() {
        use crate::opcodes::{instr_audio_resample, instr_stream_merge};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let a = rfc0004_f32(&mut vm, &[2], &[1.0, 2.0]);
        let b = rfc0004_f32(&mut vm, &[2], &[10.0, 20.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, a), (1, b)]);
        let rd = |vm: &Vm, r: u8| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, 2).unwrap();
        // GAIN=0.25: [7.75, 15.5] exatos.
        vm.step_instruction(cid, &instr_stream_merge(2, 0, 1, 0.25, 0)).unwrap();
        assert_eq!(rd(&vm, 2), vec![7.75, 15.5]);
        // N_STREAMS explícito aceito (metadado).
        vm.step_instruction(cid, &instr_stream_merge(2, 0, 1, 0.5, 17)).unwrap();
        assert_eq!(rd(&vm, 2), vec![5.5, 11.0]);
        // Downsample [0,1,2,3] 4->2: [0,2]. Upsample [0,10] 1->2: [0,5,10,10].
        let r4 = rfc0004_f32(&mut vm, &[4], &[0.0, 1.0, 2.0, 3.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, r4).unwrap();
        vm.step_instruction(cid, &instr_audio_resample(4, 3, 4, 2)).unwrap();
        let rd4 = rfc0005_reg_u64(&vm, cid, 4) as u128;
        assert_eq!(vm.memory.read_f32_tensor(rd4, 2).unwrap(), vec![0.0, 2.0]);
        let r2v = rfc0004_f32(&mut vm, &[2], &[0.0, 10.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, r2v).unwrap();
        vm.step_instruction(cid, &instr_audio_resample(4, 3, 1, 2)).unwrap();
        let rd4b = rfc0005_reg_u64(&vm, cid, 4) as u128;
        assert_eq!(vm.memory.read_f32_tensor(rd4b, 4).unwrap(), vec![0.0, 5.0, 10.0, 10.0]);
        // Erros: shapes, ganho NaN, N_STREAMS=1, modo, taxas, saída vazia.
        let c3 = rfc0004_f32(&mut vm, &[3], &[0.0; 3]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, c3).unwrap();
        assert!(vm.step_instruction(cid, &instr_stream_merge(2, 0, 5, 0.5, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_stream_merge(2, 0, 1, f32::NAN, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_stream_merge(2, 0, 1, 0.5, 1)).is_err());
        let mut badm = instr_stream_merge(2, 0, 1, 0.5, 0);
        badm.set_merge_params(0.5, 0, 7);
        assert!(vm.step_instruction(cid, &badm).is_err());
        assert!(vm.step_instruction(cid, &instr_audio_resample(4, 3, 0, 2)).is_err());
        let r1v = rfc0004_f32(&mut vm, &[1], &[5.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, r1v).unwrap();
        assert!(vm.step_instruction(cid, &instr_audio_resample(4, 3, 2, 1)).is_err());
        assert_eq!(vm.stats.stream_merge_execs, 2);
        assert_eq!(vm.stats.audio_resample_execs, 2);
    }

    #[test]
    fn test_rfc0031_filter_window() {
        use crate::opcodes::{instr_audio_filter, instr_audio_window, FILTER_MODE_FIR, FILTER_MODE_IIR, WINDOW_HANN, WINDOW_HAMMING};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let ax = rfc0004_f32(&mut vm, &[4], &[1.0, 2.0, 3.0, 4.0]);
        let ab = rfc0004_f32(&mut vm, &[2], &[0.5, 0.5]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, ax), (1, ab)]);
        let rd = |vm: &Vm, r: u8| vm.memory.read_f32_tensor(rfc0005_reg_u64(vm, cid, r) as u128, 4).unwrap();
        // Média móvel 2, same centrado: [0.5,1.5,2.5,3.5] exato.
        vm.step_instruction(cid, &instr_audio_filter(2, 0, 1, FILTER_MODE_FIR)).unwrap();
        assert_eq!(rd(&vm, 2), vec![0.5, 1.5, 2.5, 3.5]);
        // Taps [1]: identidade.
        let ab1 = rfc0004_f32(&mut vm, &[1], &[1.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, ab1).unwrap();
        vm.step_instruction(cid, &instr_audio_filter(2, 0, 3, FILTER_MODE_FIR)).unwrap();
        assert_eq!(rd(&vm, 2), vec![1.0, 2.0, 3.0, 4.0]);
        // IIR veta; eixo... (filter não tem eixo); taps esparso veta.
        assert!(vm.step_instruction(cid, &instr_audio_filter(2, 0, 1, FILTER_MODE_IIR)).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[2, 2], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_audio_filter(2, 0, 3, FILTER_MODE_FIR)).is_err());
        // Hann N=4: [0,0.5,1,0.5]; Hamming: [0.08,0.54,1,0.54].
        // Janela × entrada [1,2,3,4]: Hann dá [0,1,3,2].
        vm.step_instruction(cid, &instr_audio_window(4, 0, WINDOW_HANN)).unwrap();
        for (g, e) in rd(&vm, 4).iter().zip([0.0, 1.0, 3.0, 2.0].iter()) {
            assert!((g - e).abs() < 1e-5, "hann {:?}", rd(&vm, 4));
        }
        // Hamming × entrada: [0.08,1.08,3.0,2.16].
        vm.step_instruction(cid, &instr_audio_window(4, 0, WINDOW_HAMMING)).unwrap();
        for (g, e) in rd(&vm, 4).iter().zip([0.08, 1.08, 3.0, 2.16].iter()) {
            assert!((g - e).abs() < 1e-5, "hamming {:?}", rd(&vm, 4));
        }
        let mut badw = instr_audio_window(4, 0, WINDOW_HANN);
        badw.set_window_type(7);
        assert!(vm.step_instruction(cid, &badw).is_err());
        assert_eq!(vm.stats.audio_filter_execs, 2);
        assert_eq!(vm.stats.audio_window_execs, 2);
    }

    #[tokio::test]
    async fn test_rfc0031_assembled_program_runs() {
        use crate::opcodes::assemble;
        let src = r#"
            TENSOR r0 1 1920 f32 FILL=0.5
            VAD_DETECT r1, r0 MODE=ENERGY
            VAD_DETECT r7, r0 MODE=ZCR
            AUDIO_RESAMPLE r2, r0 SRC=24000 DST=16000
            AUDIO_WINDOW r3, r2 TYPE=HANN
            TENSOR r4 1 8 f32 FILL=0.125
            AUDIO_FILTER r5, r3, r4 MODE=FIR
            STREAM_MERGE r6, r5, r5 GAIN=0.5
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.vad_detect_execs, 2);
        assert_eq!((stats.audio_resample_execs, stats.audio_window_execs, stats.audio_filter_execs, stats.stream_merge_execs), (1, 1, 1, 1));
        let ctx = vm.scheduler.get(1).unwrap().clone();
        let a1 = ctx.reg(1).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a1, 1).unwrap(), vec![0.5]);
        let a7 = ctx.reg(7).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a7, 1).unwrap(), vec![0.0]);
        // r2 tem 1280 elems (1920@24k -> @16k); r6 == r5 (mix 0.5+0.5).
        let a2 = ctx.reg(2).unwrap();
        let m2 = vm.memory.get_tensor_meta(a2).unwrap().clone();
        assert_eq!(m2.byte_len, 1280 * 4);
        let a5 = ctx.reg(5).unwrap();
        let a6 = ctx.reg(6).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a6, 1280).unwrap(), vm.memory.read_f32_tensor(a5, 1280).unwrap());
    }

    #[tokio::test]
    async fn test_rfc0031_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/audio_dsp_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.vad_detect_execs, 2);
        assert_eq!((stats.audio_resample_execs, stats.audio_window_execs, stats.audio_filter_execs, stats.stream_merge_execs), (1, 1, 1, 1));
    }

    // ---- RFC-0032: passo depformer --------------------------------------

    /// Monta tabela de pesos [Wq,Wk,Wv,Wo,Wh,0,0,0] (u64 LE) + matrizes
    /// FILL=fill. `.m3asm` não soletra u64 (sem STORE/.data — follow-up
    /// Fase 9); tabelas nascem aqui, em Rust.
    fn rfc0032_table(vm: &mut Vm, d: usize, ncb: usize, q: usize, fill: f32) -> (u128, u128) {
        use crate::memory::DType;
        let mat = |vm: &mut Vm, rows: usize, cols: usize| {
            let a = vm.memory.alloc_tensor(&[rows, cols], DType::F32).unwrap();
            vm.memory.write_f32_tensor(a, &vec![fill; rows * cols]).unwrap();
            a
        };
        let (wq, wk, wv, wo) = (mat(vm, d, d), mat(vm, d, d), mat(vm, d, d), mat(vm, d, d));
        let wh = mat(vm, d, ncb * q);
        let t = vm.memory.alloc_tensor(&[64], DType::U8).unwrap();
        let mut bytes = Vec::new();
        for a in [wq, wk, wv, wo, wh] {
            bytes.extend_from_slice(&u64::try_from(a).unwrap().to_le_bytes());
        }
        bytes.extend_from_slice(&[0u8; 24]);
        vm.memory.write(t, &bytes).unwrap();
        // rX [1,d] FILL=1.
        let x = vm.memory.alloc_tensor(&[1, d], DType::F32).unwrap();
        vm.memory.write_f32_tensor(x, &vec![1.0; d]).unwrap();
        (x, t)
    }

    #[test]
    fn test_rfc0032_depformer_all_ones() {
        use crate::opcodes::instr_depformer;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        // D=4, NCB=2, Q=2, NHEADS=2, tudo FILL=1: Y=[16×4], codes=[0,0].
        let (x, t) = rfc0032_table(&mut vm, 4, 2, 2, 1.0);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, t)]);
        vm.step_instruction(cid, &instr_depformer(9, 0, 1, 0, 0, 2, 2, 2, 8, 1.0, 0)).unwrap();
        let p = rfc0005_reg_u64(&vm, cid, 9) as u128;
        assert_eq!(vm.memory.read_f32_tensor(p, 6).unwrap(), vec![16.0, 16.0, 16.0, 16.0, 0.0, 0.0]);
        // Pack fatiável: Y=[16×4], codes=[0,0].
        vm.step_instruction(cid, &crate::opcodes::instr_slice(2, 9, 0, 4)).unwrap();
        let y = rfc0005_reg_u64(&vm, cid, 2) as u128;
        assert_eq!(vm.memory.read_f32_tensor(y, 4).unwrap(), vec![16.0; 4]);
        vm.step_instruction(cid, &crate::opcodes::instr_slice(3, 9, 4, 2)).unwrap();
        let c = rfc0005_reg_u64(&vm, cid, 3) as u128;
        assert_eq!(vm.memory.read_f32_tensor(c, 2).unwrap(), vec![0.0, 0.0]);
        // Determinístico: segunda chamada, mesmos códigos.
        vm.step_instruction(cid, &instr_depformer(9, 0, 1, 0, 0, 2, 2, 2, 8, 1.0, 0)).unwrap();
        let p2 = rfc0005_reg_u64(&vm, cid, 9) as u128;
        assert_eq!(vm.memory.read_f32_tensor(p2, 6).unwrap(), vec![16.0, 16.0, 16.0, 16.0, 0.0, 0.0]);
        assert_eq!(vm.dep_kv.get(&(0, 0)).unwrap().rows(), 2);
        assert_eq!(vm.stats.depformer_execs, 2);
    }

    #[test]
    fn test_rfc0032_kv_window_and_rollback() {
        use crate::opcodes::{instr_arena_alloc, instr_depformer, instr_restore, instr_snapshot, SNAP_MASK_ALL};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let (x, t) = rfc0032_table(&mut vm, 4, 2, 2, 1.0);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, t)]);
        let run = |vm: &mut Vm| {
            vm.step_instruction(cid, &instr_depformer(9, 0, 1, 0, 0, 2, 2, 2, 2, 1.0, 0)).unwrap();
        };
        run(&mut vm);
        assert_eq!(vm.dep_kv.get(&(0, 0)).unwrap().rows(), 1);
        // SNAPSHOT nomeado cobre o mapa (4ª casa, como rank1).
        vm.step_instruction(cid, &instr_snapshot(5, SNAP_MASK_ALL)).unwrap();
        run(&mut vm);
        run(&mut vm);
        assert_eq!(vm.dep_kv.get(&(0, 0)).unwrap().rows(), 2); // CONTEXT=2 drena
        vm.step_instruction(cid, &instr_restore(5)).unwrap();
        assert_eq!(vm.dep_kv.get(&(0, 0)).unwrap().rows(), 1);
        // FORK/ABORT idem (2ª/3ª casas): filho cresce, abort restaura.
        vm.step_instruction(cid, &crate::opcodes::instr_fork(6, crate::opcodes::FORK_FLAG_GREEN)).unwrap();
        let child = rfc0005_reg_u64(&vm, cid, 6);
        vm.step_instruction(child, &instr_depformer(9, 0, 1, 0, 0, 2, 2, 2, 8, 1.0, 0)).unwrap();
        assert_eq!(vm.dep_kv.get(&(0, 0)).unwrap().rows(), 2);
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, child as u128).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(8, 0).unwrap();
        vm.step_instruction(cid, &crate::opcodes::instr_abort(7, 8)).unwrap();
        assert_eq!(vm.dep_kv.get(&(0, 0)).unwrap().rows(), 1);
        // ARENA coexistindo não quebra o push (regressão de tupla).
        vm.step_instruction(cid, &instr_arena_alloc(0, 16, 0, 0)).unwrap();
        assert_eq!(vm.stats.depformer_execs, 4);
    }

    #[test]
    fn test_rfc0032_sampling_seeded() {
        use crate::opcodes::{instr_depformer, instr_rng_seed};
        // Mesmo seed => mesmos códigos (acordo entre runs, padrão RFC-0009).
        let run_seeded = |seed: u64| {
            let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
            let (x, t) = rfc0032_table(&mut vm, 4, 2, 4, 0.5);
            let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, t), (2, seed as u128)]);
            vm.step_instruction(cid, &instr_rng_seed(2)).unwrap();
            vm.step_instruction(cid, &instr_depformer(9, 0, 1, 0, 0, 2, 4, 4, 8, 1.0, 4)).unwrap();
            let p = rfc0005_reg_u64(&vm, cid, 9) as u128;
            vm.memory.read_f32_tensor(p, 6).unwrap()
        };
        let a = run_seeded(42);
        let b = run_seeded(42);
        assert_eq!(a, b);
        // Códigos dentro do range [0,4) e seeds distintas ao menos tentam divergir
        // (não garantido estatisticamente com 2 draws — só range é assert).
        for v in a[4..6].iter() {
            assert!(*v >= 0.0 && *v < 4.0, "{:?}", a);
        }
    }

    #[test]
    fn test_rfc0032_errors() {
        use crate::opcodes::instr_depformer;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let (x, t) = rfc0032_table(&mut vm, 4, 2, 2, 1.0);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, t)]);
        let ok = || instr_depformer(9, 0, 1, 0, 0, 2, 2, 2, 8, 1.0, 0);
        vm.step_instruction(cid, &ok()).unwrap();
        // NHEADS ∤ D; TOPK > Q; TEMP NaN c/ amostragem; tabela curta;
        // peso ausente; reservado != 0; rX esparso; camada reconfigurada.
        assert!(vm.step_instruction(cid, &instr_depformer(9, 0, 1, 0, 0, 2, 3, 2, 8, 1.0, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_depformer(9, 0, 1, 0, 0, 2, 2, 2, 8, 1.0, 3)).is_err());
        assert!(vm.step_instruction(cid, &instr_depformer(9, 0, 1, 0, 0, 2, 2, 2, 8, f32::NAN, 2)).is_err());
        let tshort = vm.memory.alloc_tensor(&[4], crate::memory::DType::U8).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, tshort).unwrap();
        assert!(vm.step_instruction(cid, &instr_depformer(9, 0, 2, 0, 0, 2, 2, 2, 8, 1.0, 0)).is_err());
        let tbad = vm.memory.alloc_tensor(&[64], crate::memory::DType::U8).unwrap();
        let mut raw = vec![0u8; 64];
        raw[0..8].copy_from_slice(&0xdeadbeefu64.to_le_bytes());
        vm.memory.write(tbad, &raw).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, tbad).unwrap();
        assert!(vm.step_instruction(cid, &instr_depformer(9, 0, 2, 0, 0, 2, 2, 2, 8, 1.0, 0)).is_err());
        let trsv = vm.memory.alloc_tensor(&[64], crate::memory::DType::U8).unwrap();
        // tabela válida mas reservado[0] != 0.
        let mut tb = vec![0u8; 64];
        // copia tabela válida e suja o reservado.
        let good = vm.memory.read(t, 64).unwrap();
        tb.copy_from_slice(&good);
        tb[40] = 1;
        vm.memory.write(trsv, &tb).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, trsv).unwrap();
        assert!(vm.step_instruction(cid, &instr_depformer(9, 0, 2, 0, 0, 2, 2, 2, 8, 1.0, 0)).is_err());
        let asp = vm.memory.alloc_sparse_tensor(&[1, 4], crate::memory::DType::F32, 0.05).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, asp).unwrap();
        assert!(vm.step_instruction(cid, &instr_depformer(9, 2, 1, 0, 0, 2, 2, 2, 8, 1.0, 0)).is_err());
        // Reconfiguração real: mesma (stream,layer), D=8 com pesos [8,8].
        let (x8, t8) = rfc0032_table(&mut vm, 8, 2, 2, 1.0);
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, x8).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, t8).unwrap();
        assert!(vm.step_instruction(cid, &instr_depformer(9, 2, 3, 0, 0, 2, 2, 2, 8, 1.0, 0)).is_err());
        assert!(vm.step_instruction(cid, &instr_depformer(0xFF, 0, 1, 0, 0, 2, 2, 2, 8, 1.0, 0)).is_err());
        assert_eq!(vm.stats.depformer_execs, 1);
    }

    #[tokio::test]
    async fn test_rfc0032_combined_runs() {
        use crate::opcodes::assemble;
        // Programa .m3asm de verdade dirigindo tabelas feitas em Rust
        // (u64 não se soletra em .m3asm — Fase 9): prova o caminho CLI.
        let src = "DEPFORMER r9, r0, r1 LAYER=0 NCB=2 NHEADS=2 LEVELS=2 CONTEXT=8\nHALT\n";
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let (x, t) = rfc0032_table(&mut vm, 4, 2, 2, 1.0);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, t)]);
        vm.load_program(prog);
        // load_program cria ctx novo se vazio — aqui já há ctx com regs.
        let stats = vm.run().unwrap();
        assert_eq!(stats.depformer_execs, 1);
        let ctx = vm.scheduler.get(cid).unwrap().clone();
        let p = ctx.reg(9).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(p, 6).unwrap(), vec![16.0, 16.0, 16.0, 16.0, 0.0, 0.0]);
    }

    // ---- RFC-0033: KV por stream ----------------------------------------

    #[test]
    fn test_rfc0033_stream_isolation() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.memory.kv_cache_init(2, 4);
        // Stream 0: 3 linhas [t;4]. Stream 1: 5 linhas [100+t;4].
        for t in 0..3u32 {
            for layer in 0..2usize {
                vm.memory.kv_cache_append(layer, &vec![t as f32; 4], &vec![t as f32; 4]).unwrap();
            }
        }
        for t in 0..5u32 {
            for layer in 0..2usize {
                vm.memory.kv_cache_append_stream(1, layer, &vec![(100 + t) as f32; 4], &vec![(200 + t) as f32; 4]).unwrap();
            }
        }
        assert_eq!(vm.memory.kv_cache_seq_len(), 3);
        assert_eq!(vm.memory.kv_cache_seq_len_stream(1), 5);
        assert_eq!(vm.memory.kv_cache_seq_len_stream(2), 0); // inexistente
        // Valores isolados por stream.
        let cpu = vm.memory.as_cpu_mut().unwrap();
        assert_eq!(cpu.kv_cache_row_stream(0, 0, 2).unwrap().0, vec![2.0; 4]);
        assert_eq!(cpu.kv_cache_row_stream(1, 0, 4).unwrap().0, vec![104.0; 4]);
        assert_eq!(cpu.kv_cache_row_stream(1, 1, 0).unwrap().1, vec![200.0; 4]);
        assert!(cpu.kv_cache_row_stream(2, 0, 0).is_none());
        // Stream 0 intocado pelas escritas do stream 1.
        assert_eq!(cpu.kv_cache_row_stream(0, 1, 0).unwrap().1, vec![0.0; 4]);
        // Geometria herdada: mesma (n_layers, hidden); layer OOB veta.
        assert!(vm.memory.kv_cache_append_stream(1, 9, &[0.0; 4], &[0.0; 4]).is_err());
        assert!(vm.memory.kv_cache_append_stream(1, 0, &[0.0; 3], &[0.0; 4]).is_err());
    }

    #[test]
    fn test_rfc0033_stream_uninit_and_noop() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // Sem init: append em extra veta (stream 0 sem geometria).
        assert!(vm.memory.kv_cache_append_stream(1, 0, &[0.0; 4], &[0.0; 4]).is_err());
        // Truncate/compress em stream inexistente: no-op Ok, nada materializa.
        vm.memory.kv_cache_truncate_stream(4, 0);
        vm.memory.kv_cache_compress_sink_window_stream(4, 2, 2);
        assert_eq!(vm.memory.kv_cache_seq_len_stream(4), 0);
        assert!(vm.memory.as_cpu_mut().unwrap().kv_cache_row_stream(4, 0, 0).is_none());
    }

    #[test]
    fn test_rfc0033_exec_routing() {
        use crate::opcodes::{instr_kv_compress, instr_kv_truncate};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.memory.kv_cache_init(2, 4);
        for t in 0..6u32 {
            for layer in 0..2usize {
                vm.memory.kv_cache_append(layer, &[t as f32; 4], &[t as f32; 4]).unwrap();
                vm.memory.kv_cache_append_stream(1, layer, &[(50 + t) as f32; 4], &[(60 + t) as f32; 4]).unwrap();
            }
        }
        let cid = rfc0004_ctx_with(&mut vm, &[(0, 4u128)]);
        // TRUNCATE STREAM=1 não toca o stream 0.
        let mut tr1 = instr_kv_truncate(0, 0);
        tr1.set_kv_stream(1);
        vm.step_instruction(cid, &tr1).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 6);
        assert_eq!(vm.memory.kv_cache_seq_len_stream(1), 4);
        // COMPRESS STREAM=1: SINK=1 WINDOW=1 em seq 4 => [50,53].
        vm.step_instruction(cid, &instr_kv_compress(1, 1, 0, 1)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len_stream(1), 2);
        assert_eq!(vm.memory.kv_cache_seq_len(), 6);
        let cpu = vm.memory.as_cpu_mut().unwrap();
        assert_eq!(cpu.kv_cache_row_stream(1, 0, 0).unwrap().0, vec![50.0; 4]);
        assert_eq!(cpu.kv_cache_row_stream(1, 0, 1).unwrap().0, vec![53.0; 4]);
        // sid > 16 veta alto nos dois ops.
        let mut tr99 = instr_kv_truncate(0, 0);
        tr99.set_kv_stream(99);
        assert!(vm.step_instruction(cid, &tr99).is_err());
        assert!(vm.step_instruction(cid, &instr_kv_compress(1, 1, 0, 17)).is_err());
        assert_eq!(vm.stats.kv_truncate_execs, 1);
        assert_eq!(vm.stats.kv_compress_execs, 1);
    }

    #[test]
    fn test_rfc0033_snapshot_restore_streams() {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        vm.memory.kv_cache_init(1, 2);
        for t in 0..3u32 {
            vm.memory.kv_cache_append(0, &[t as f32; 2], &[t as f32; 2]).unwrap();
            vm.memory.kv_cache_append_stream(1, 0, &[(10 + t) as f32; 2], &[0.0; 2]).unwrap();
        }
        let snap = vm.memory.snapshot();
        // Muta tudo: append + compress no extra.
        vm.memory.kv_cache_append(0, &[99.0; 2], &[99.0; 2]).unwrap();
        vm.memory.kv_cache_append_stream(1, 0, &[99.0; 2], &[99.0; 2]).unwrap();
        vm.memory.kv_cache_compress_sink_window_stream(1, 1, 1);
        assert_eq!(vm.memory.kv_cache_seq_len_stream(1), 2);
        vm.memory.restore(snap).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 3);
        assert_eq!(vm.memory.kv_cache_seq_len_stream(1), 3);
        let cpu = vm.memory.as_cpu_mut().unwrap();
        assert_eq!(cpu.kv_cache_row_stream(1, 0, 2).unwrap().0, vec![12.0; 2]);
        // Retenção ainda limitada com streams (janela conta versões).
        vm.memory.as_cpu_mut().unwrap().set_snapshot_window(2);
        let _ = vm.memory.snapshot();
        let _ = vm.memory.snapshot();
        let _ = vm.memory.snapshot();
        assert!(vm.memory.snapshot_count() <= 2);
        // Versão expirada falha alto (incl. p/ streams).
        assert!(vm.memory.restore(snap).is_err());
    }

    #[test]
    fn test_rfc0033_depformer_stream_foresight() {
        use crate::opcodes::instr_depformer;
        // DEPFORMER (RFC-0032) já chaveava por stream: stream 1 não toca (0,0).
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let mk = |vm: &mut Vm, rows: usize, cols: usize| {
            let a = vm.memory.alloc_tensor(&[rows, cols], crate::memory::DType::F32).unwrap();
            vm.memory.write_f32_tensor(a, &vec![1.0; rows * cols]).unwrap();
            a
        };
        let x = mk(&mut vm, 1, 4);
        let wq = mk(&mut vm, 4, 4);
        let wk = mk(&mut vm, 4, 4);
        let wv = mk(&mut vm, 4, 4);
        let wo = mk(&mut vm, 4, 4);
        let wh = mk(&mut vm, 4, 4);
        let t = vm.memory.alloc_tensor(&[64], crate::memory::DType::U8).unwrap();
        let mut tb = Vec::new();
        for a in [wq, wk, wv, wo, wh] {
            tb.extend_from_slice(&u64::try_from(a).unwrap().to_le_bytes());
        }
        tb.extend_from_slice(&[0u8; 24]);
        vm.memory.write(t, &tb).unwrap();
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, t)]);
        vm.step_instruction(cid, &instr_depformer(9, 0, 1, 1, 0, 2, 2, 2, 8, 1.0, 0)).unwrap();
        assert!(vm.dep_kv.contains_key(&(1, 0)));
        assert!(!vm.dep_kv.contains_key(&(0, 0)));
        assert_eq!(vm.stats.depformer_execs, 1);
    }

    #[tokio::test]
    async fn test_rfc0033_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/kv_stream_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.kv_truncate_execs, stats.kv_compress_execs), (1, 1));
    }

    // ---- RFC-0034: sub-rotinas ------------------------------------------

    #[test]
    fn test_rfc0034_ret_underflow() {
        use crate::opcodes::instr_ret;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // Pilha vazia: RET veta alto em vez de cair em bytes aleatórios.
        assert!(vm.step_instruction(cid, &instr_ret()).is_err());
        assert_eq!(vm.stats.ret_execs, 0);
    }

    #[test]
    fn test_rfc0034_depth_overflow() {
        use crate::opcodes::instr_call;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(5000), ..Default::default() });
        let prog = crate::opcodes::assemble("HALT").unwrap();
        vm.load_program(prog);
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // 1024 CALLs aninhados passam; o 1025º veta (teto exato).
        for _ in 0..crate::context::MAX_CALL_DEPTH {
            vm.step_instruction(cid, &instr_call(0x1000)).unwrap();
        }
        assert_eq!(vm.scheduler.get(cid).unwrap().call_stack.len(), crate::context::MAX_CALL_DEPTH);
        assert!(vm.step_instruction(cid, &instr_call(0x1000)).is_err());
        assert_eq!(vm.stats.call_execs, crate::context::MAX_CALL_DEPTH as u64);
    }

    #[test]
    fn test_rfc0034_fork_inherits_stack() {
        use crate::opcodes::{instr_abort, instr_call, instr_fork, FORK_FLAG_GREEN};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let prog = crate::opcodes::assemble("CALL SUB\nHALT\nSUB:\nRET").unwrap();
        vm.load_program(prog);
        let parent = rfc0004_ctx_with(&mut vm, &[]);
        // Entra na sub-rotina: pilha do pai = [ret].
        vm.step_instruction(parent, &instr_call(0x1000 + 64)).unwrap();
        assert_eq!(vm.scheduler.get(parent).unwrap().call_stack.len(), 1);
        // FORK clona a pilha (semântica Unix).
        vm.step_instruction(parent, &instr_fork(0, FORK_FLAG_GREEN)).unwrap();
        let child = rfc0005_reg_u64(&vm, parent, 0);
        assert_eq!(vm.scheduler.get(child).unwrap().call_stack.len(), 1);
        assert_eq!(
            vm.scheduler.get(child).unwrap().call_stack,
            vm.scheduler.get(parent).unwrap().call_stack
        );
        // ABORT mata o filho (pilha morre junto); pai intacto.
        vm.scheduler.get_mut(parent).unwrap().set_reg(1, child as u128).unwrap();
        vm.scheduler.get_mut(parent).unwrap().set_reg(2, 0).unwrap();
        vm.step_instruction(parent, &instr_abort(1, 2)).unwrap();
        assert!(vm.scheduler.get(child).is_none());
        assert_eq!(vm.scheduler.get(parent).unwrap().call_stack.len(), 1);
    }

    #[tokio::test]
    async fn test_rfc0034_assembled_program_runs() {
        use crate::opcodes::assemble;
        let src = r#"
            LOADI r0, 5
            CALL LEVEL1
            HALT
        LEVEL1:
            ADD_IMM r0, r0 IMM=1
            CALL LEVEL2
            RET
        LEVEL2:
            ADD_IMM r0, r0 IMM=10
            RET
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.call_execs, stats.ret_execs), (2, 2));
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(0).unwrap(), 16);
        assert!(ctx.call_stack.is_empty());
    }

    #[tokio::test]
    async fn test_rfc0034_demo_runs() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/call_ret_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!((stats.call_execs, stats.ret_execs), (2, 2));
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(0).unwrap(), 16);
        assert!(ctx.call_stack.is_empty());
    }

    // ---- PLANO_VISAO Fase V-0 (demos de comportamento) --------------------

    #[tokio::test]
    async fn test_planoV0_pause_paths() {
        use crate::opcodes::assemble;
        // Caminho retomada: 1 evento de fala => RESUMED(3), VAD pontuado.
        let src = include_str!("../programs/pause_turn_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        vm.load_program(prog);
        vm.push_input("oi".to_string());
        vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(0).unwrap(), 3);
        let vscore = ctx.reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(vscore, 1).unwrap(), vec![0.5]);
        // Caminho timeout: sem eventos, 64 iters quietas => TURN_END(4)+HALT.
        let prog = assemble(src).unwrap();
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(300), ..Default::default() });
        vm2.load_program(prog);
        vm2.run().unwrap();
        let ctx2 = vm2.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx2.reg(0).unwrap(), 4);
    }

    #[tokio::test]
    async fn test_planoV0_filler_synth() {
        use crate::opcodes::assemble;
        let src = include_str!("../programs/filler_synth_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        // Crossfade 0.25*0.5+0.75*0.5 = 0.5 bit-exato; 3 iterações.
        let out = ctx.reg(7).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 8).unwrap(), vec![0.5; 8]);
        assert_eq!(ctx.reg(8).unwrap(), 3);
        // Sorteio com seed: bits f32 zero-estendidos ([0,1)).
        let pick = ctx.reg(1).unwrap();
        let pickf = f32::from_bits(pick as u32);
        assert!(pickf >= 0.0 && pickf < 1.0, "pick={}", pickf);
    }

    #[tokio::test]
    async fn test_planoV1_equ_consts() {
        use crate::opcodes::assemble;
        // RFC-0037 (V-1a): `.equ`/`.text` executam com semântica idêntica
        // aos literais. r0=10, r1=42, r2=32, r4=1 (branch tomado), r3=[2,2] FILL=0.5.
        let src = include_str!("../programs/equ_const_demo.m3asm");
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(0).unwrap(), 10);
        assert_eq!(ctx.reg(1).unwrap(), 42);
        assert_eq!(ctx.reg(2).unwrap(), 32);
        assert_eq!(ctx.reg(4).unwrap(), 1);
        let tab = ctx.reg(3).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(tab, 4).unwrap(), vec![0.5; 4]);
    }

    #[tokio::test]
    async fn test_planoV3_voice_loop() {
        use crate::opcodes::assemble;
        // V-3/G6 (MVP voice-loop, emulador): SENSE->VAD->CODEC->DEPFORMER->
        // CODEC->STREAM + filler isolado + estados + barge + deadline.
        // Tabela DEPFORMER nasce no harness (nota harness_table no .m3asm).
        let template = include_str!("../programs/voice_loop_demo.m3asm");
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(4000), ..Default::default() });
        let (_x, t) = rfc0032_table(&mut vm, 16, 2, 2, 1.0);
        let src = template.replace(".equ W_TAB 0", &format!(".equ W_TAB {}", t));
        let prog = assemble(&src).unwrap();
        vm.load_program(prog);
        // 1 evento => 1 barge no frame 1; frames 2-3 limpos; COMPLETE.
        vm.push_input("oi".to_string());
        vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        // (1) Estados: todos os 5 visitados; final COMPLETE; 2 frames limpos.
        assert_eq!(ctx.reg(0).unwrap(), 4, "rState=COMPLETE");
        assert_eq!(ctx.reg(1).unwrap(), 2, "rFrames=2 limpos");
        assert_eq!(ctx.reg(11).unwrap(), 1, "rBarged: RESUMED aconteceu");
        // (2) Barge-in: interrompeu e voltou a INCOMPLETE (completou depois).
        // (coberto por rBarged==1 + COMPLETE acima; canário abaixo prova restore.)
        // (3) Filler isolado: rY = 2*(7.777^2) exato; output sem marcador.
        let x = 7.777f32;
        let expect_fill = x * x + x * x;
        let y = ctx.reg(6).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(y, 8).unwrap(), vec![expect_fill; 8]);
        let out = ctx.reg(7).unwrap();
        let pcm = vm.memory.read_f32_tensor(out, 1920).unwrap();
        assert_eq!(pcm.len(), 1920);
        assert!(!pcm.iter().any(|&v| v == 7.777f32 || v == expect_fill), "filler vazou p/ output");
        // (4) Rollback: canário pré-snapshot intacto pós-RESTORE + coerência
        // funcional após restore (COMPLETE acima). Rewind de mapa/ssm/KV é
        // propriedade unitária (RFC-0011/RFC-0003); aqui: não-corrupção + direta.
        let can = ctx.reg(14).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(can, 4).unwrap(), vec![1.0; 4]);
        assert!(ctx.reg(8).unwrap() > 0, "rSnap: versão válida");
        // VAD determinístico sobre rCan ([1;4] => energia 1.0).
        let vad = ctx.reg(5).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(vad, 1).unwrap(), vec![1.0]);
        // (5) Deadline: nenhum frame estourou o orçamento STEPS.
        assert_eq!(ctx.reg(12).unwrap(), 0, "rOver: budget estourado");
        // Cobertura dos estágios (exatos: 1 barge + 2 limpos).
        assert_eq!(vm.stats.vad_detect_execs, 2);
        assert_eq!(vm.stats.codec_encs, 2);
        assert_eq!(vm.stats.codec_decs, 2);
        assert_eq!(vm.stats.streams, 2);
    }

    #[tokio::test]
    async fn test_planoV1b_data_loader() {
        use crate::opcodes::assemble_with_data;
        // V-1b dia 3 (Opção A): `@nome` resolve em assemble para o layout
        // fixo; `load_assembled` posiciona os bytes; leitura confirma.
        let src = include_str!("../programs/data_addr_demo.m3asm");
        let prog = assemble_with_data(src).unwrap();
        assert_eq!(prog.data.blobs.len(), 4);
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_assembled(&prog).unwrap();
        vm.run().unwrap();
        let ctx = vm.scheduler.get(1).unwrap().clone();
        assert_eq!(ctx.reg(0).unwrap(), 0x1000);
        assert_eq!(ctx.reg(1).unwrap(), 0x1008);
        assert_eq!(ctx.reg(2).unwrap(), 0x1010);
        assert_eq!(ctx.reg(3).unwrap(), 0x1018);
        assert_eq!(vm.memory.read(0x1000, 4).unwrap(), 1920u32.to_le_bytes().to_vec());
        assert_eq!(vm.memory.read(0x1008, 4).unwrap(), 0.5f32.to_le_bytes().to_vec());
        assert_eq!(vm.memory.read(0x1010, 2).unwrap(), b"hi".to_vec());
        assert_eq!(vm.memory.read_f32_tensor(0x1018, 4).unwrap(), vec![1.0, 0.0, 0.0, 1.0]);
        // Guarda honesta: segundo preload (ou VM não-fresca) erra alto —
        // endereços `@` foram congelados para a base.
        assert!(vm.load_assembled(&prog).is_err());
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        let setup = crate::opcodes::assemble("TENSOR r0 2 2 f32\nHALT").unwrap();
        vm2.load_program(setup);
        vm2.run().unwrap();
        assert!(vm2.load_assembled(&prog).is_err());
    }

    // ---- RFC-0005: determinismo -------------------------------------

    fn rfc0005_reg_u64(vm: &Vm, cid: u64, r: u8) -> u64 {
        vm.scheduler.get(cid).unwrap().reg(r).unwrap() as u64
    }

    #[tokio::test]
    async fn test_rng_determinism_reseed_fork() {
        use crate::opcodes::{instr_fork, instr_rng_next, instr_rng_seed};
        async fn run_seq(seed: Option<u64>) -> Vec<u64> {
            let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
            let cid = rfc0004_ctx_with(&mut vm, &[]);
            match seed {
                Some(s) => {
                    vm.scheduler.get_mut(cid).unwrap().set_reg(0, s as u128).unwrap();
                    vm.step_instruction(cid, &instr_rng_seed(0)).unwrap();
                }
                None => {
                    vm.step_instruction(cid, &instr_rng_seed(0xFF)).unwrap();
                }
            }
            let mut out = Vec::new();
            for _ in 0..5 {
                vm.step_instruction(cid, &instr_rng_next(1)).unwrap();
                out.push(rfc0005_reg_u64(&vm, cid, 1));
            }
            out
        }
        // Mesma semente explícita => mesma sequência; default == default.
        assert_eq!(run_seq(Some(12345)).await, run_seq(Some(12345)).await);
        assert_eq!(run_seq(None).await, run_seq(None).await);
        assert_ne!(run_seq(Some(12345)).await, run_seq(Some(999)).await);
        // FORK herda o stream: pai e filho geram o mesmo próximo valor.
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, 777u128).unwrap();
        vm.step_instruction(cid, &instr_rng_seed(0)).unwrap();
        vm.step_instruction(cid, &instr_rng_next(1)).unwrap();
        let first = rfc0005_reg_u64(&vm, cid, 1);
        vm.step_instruction(cid, &instr_fork(2, 0)).unwrap();
        let child = vm.scheduler.get(cid).unwrap().reg(2).unwrap() as u64;
        // Filho continua de onde o pai parou (1 NEXT consumido por ambos os
        // lados de forma independente => próximos valores divergem a partir
        // do mesmo estado herdado: pai e filho geram IGUAL agora).
        vm.step_instruction(cid, &instr_rng_next(3)).unwrap();
        vm.step_instruction(child, &instr_rng_next(3)).unwrap();
        assert_eq!(rfc0005_reg_u64(&vm, cid, 3), rfc0005_reg_u64(&vm, child, 3));
        assert_ne!(first, rfc0005_reg_u64(&vm, cid, 3));
    }

    #[tokio::test]
    async fn test_rng_uniform_normal_hash_goldens() {
        use crate::opcodes::{instr_checksum, instr_hash, instr_hmac, instr_rng_normal, instr_rng_uniform};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let t = rfc0004_f32(&mut vm, &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, t)]);
        // UNIFORM dentro de [a,b).
        vm.step_instruction(cid, &instr_rng_uniform(1, -2.0, 5.0)).unwrap();
        let x = f32::from_bits(rfc0005_reg_u64(&vm, cid, 1) as u32);
        assert!((-2.0..5.0).contains(&x));
        // Intervalo inválido: erro limpo.
        assert!(vm.step_instruction(cid, &instr_rng_uniform(1, 5.0, 5.0)).is_err());
        assert!(vm.step_instruction(cid, &instr_rng_uniform(1, f32::NAN, 1.0)).is_err());
        // NORMAL finita; std inválido: erro limpo.
        vm.step_instruction(cid, &instr_rng_normal(1, 10.0, 2.0)).unwrap();
        let y = f32::from_bits(rfc0005_reg_u64(&vm, cid, 1) as u32);
        assert!(y.is_finite());
        assert!(vm.step_instruction(cid, &instr_rng_normal(1, 0.0, -1.0)).is_err());
        assert!(vm.step_instruction(cid, &instr_rng_normal(1, 0.0, f32::INFINITY)).is_err());
        // HASH/CHECKSUM batem com as funções puras sobre o mesmo stream.
        vm.step_instruction(cid, &instr_hash(1, 0)).unwrap();
        vm.step_instruction(cid, &instr_checksum(2, 0)).unwrap();
        let flat = vm.memory.read_f32_tensor(t, 4).unwrap();
        let mut bytes = Vec::new();
        for v in &flat {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(rfc0005_reg_u64(&vm, cid, 1), crate::determinism::fnv1a64(&bytes));
        assert_eq!(rfc0005_reg_u64(&vm, cid, 2), crate::determinism::crc32_ieee(&bytes) as u64);
        // HMAC contra a primitiva direta (vetor RFC 4231 vive em determinism.rs).
        let k = rfc0004_f32(&mut vm, &[5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, k).unwrap();
        vm.step_instruction(cid, &instr_hmac(4, 3, 0)).unwrap();
        let kb = {
            let kv = vm.memory.read_f32_tensor(k, 5).unwrap();
            let mut b = Vec::new();
            for v in &kv {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b
        };
        assert_eq!(
            rfc0005_reg_u64(&vm, cid, 4),
            crate::determinism::hmac_sha256_trunc64(&kb, &bytes)
        );
        assert_eq!(vm.stats.rng_uniform_execs, 1);
        assert_eq!(vm.stats.rng_normal_execs, 1);
        assert_eq!(vm.stats.hash_execs, 1);
        assert_eq!(vm.stats.checksum_execs, 1);
        assert_eq!(vm.stats.hmac_execs, 1);
    }

    #[tokio::test]
    async fn test_rfc0005_assembled_program_runs() {
        use crate::opcodes::assemble;
        let src = r#"
            RNG_SEED DEFAULT
            RNG_NEXT r1
            RNG_NEXT r2
            RNG_UNIFORM r3 A=-2.0 B=5.0
            RNG_NORMAL r4 MEAN=10.0 STD=2.0
            TENSOR r5 2 2 f32
            HASH r6, r5
            CHECKSUM r7, r5
            HMAC r8, r5, r5
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.rng_seed_execs, 1);
        assert_eq!(stats.rng_next_execs, 2);
        assert_eq!(stats.rng_uniform_execs, 1);
        assert_eq!(stats.rng_normal_execs, 1);
        assert_eq!(stats.hash_execs, 1);
        assert_eq!(stats.checksum_execs, 1);
        assert_eq!(stats.hmac_execs, 1);
        // Determinismo ponta a ponta: mesma seed => mesmos registradores.
        let main_id = *vm.scheduler.contexts().keys().next().expect("ctx principal");
        let regs_once = {
            let ctx = vm.scheduler.get(main_id).unwrap();
            (ctx.reg(1).unwrap(), ctx.reg(2).unwrap())
        };
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        vm2.load_program(assemble(src).unwrap());
        vm2.run().unwrap();
        let main2 = *vm2.scheduler.contexts().keys().next().expect("ctx principal");
        let ctx2 = vm2.scheduler.get(main2).unwrap();
        assert_eq!((ctx2.reg(1).unwrap(), ctx2.reg(2).unwrap()), regs_once);
    }

    // ---- RFC-0011: stacks versionadas ---------------------------------

    fn rfc0011_set_ssm(vm: &mut Vm, val: f32) {
        if vm.ssm_states.is_empty() {
            vm.ssm_states.push(crate::ssm::MambaState::new(2, 1, 4));
        }
        for x in vm.ssm_states[0].ssm.iter_mut() {
            *x = val;
        }
    }

    fn rfc0011_ssm(vm: &Vm) -> Vec<f32> {
        vm.ssm_states[0].ssm.clone()
    }

    #[tokio::test]
    async fn test_abort_version_gated_nested() {
        use crate::opcodes::{instr_abort, instr_fork};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // Estado A, FORK(v1), estado B, FORK(v2), estado C.
        rfc0011_set_ssm(&mut vm, 1.0);
        vm.step_instruction(cid, &instr_fork(5, 0)).unwrap();
        let v1 = vm.memory.current_version();
        rfc0011_set_ssm(&mut vm, 2.0);
        vm.step_instruction(cid, &instr_fork(6, 0)).unwrap();
        let v2 = vm.memory.current_version();
        assert!(v2 > v1);
        rfc0011_set_ssm(&mut vm, 3.0);
        // ABORT nomeando v1 (alvo: filho 2 em r6; ts=v1 em r7):
        // descarta (v2,B), aplica (v1,A).
        vm.scheduler.get_mut(cid).unwrap().set_reg(7, v1 as u128).unwrap();
        vm.step_instruction(cid, &instr_abort(6, 7)).unwrap();
        assert_eq!(rfc0011_ssm(&vm), vec![1.0; 2]);
        assert!(vm.ssm_snapshots.is_empty(), "entradas consumidas, sem resíduo");
        // Segundo ABORT no mesmo ts (alvo já removido => warn, segue):
        // no-op estável (antes: comia o outer).
        vm.step_instruction(cid, &instr_abort(6, 7)).unwrap();
        assert_eq!(rfc0011_ssm(&vm), vec![1.0; 2]);
    }

    #[tokio::test]
    async fn test_abort_without_fork_is_noop() {
        use crate::opcodes::instr_abort;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        rfc0011_set_ssm(&mut vm, 9.0);
        // Alvo válido (o próprio ctx sobrevive? não — usa ctx inexistente:
        // remove falha com warn, stacks vazias => no-op, estado intacto).
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, 4242u128).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, 0u128).unwrap();
        vm.step_instruction(cid, &instr_abort(5, 6)).unwrap();
        assert_eq!(rfc0011_ssm(&vm), vec![9.0; 2]);
        assert!(vm.ssm_snapshots.is_empty());
        assert!(vm.rank1_snapshots.is_empty());
    }

    #[tokio::test]
    async fn test_rank1_stack_version_gated() {
        use crate::opcodes::{instr_abort, instr_fork, instr_rank1_update, RANK1_MODE_HEBBIAN};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let h = rfc0004_f32(&mut vm, &[2, 2], &[0.0; 4]);
        let v = rfc0004_f32(&mut vm, &[2], &[1.0, 1.0]);
        let k = rfc0004_f32(&mut vm, &[2], &[1.0, 1.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(7, h), (1, v), (2, k)]);
        // Passo base ANTES da 1ª FORK: v1 carrega {0: h_a}.
        vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 1.0, 1.0, RANK1_MODE_HEBBIAN, 0)).unwrap();
        let ha = vm.rank1_layers.get(&0).copied().unwrap();
        assert_eq!(vm.memory.read_f32_tensor(ha, 4).unwrap(), vec![1.0; 4]);
        vm.step_instruction(cid, &instr_fork(5, 0)).unwrap();
        let v1 = vm.memory.current_version();
        vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 1.0, 1.0, RANK1_MODE_HEBBIAN, 0)).unwrap();
        let dirty = vm.rank1_layers.get(&0).copied().unwrap();
        assert_ne!(dirty, ha);
        vm.step_instruction(cid, &instr_fork(6, 0)).unwrap();
        vm.step_instruction(cid, &instr_rank1_update(7, 1, 2, 1.0, 1.0, RANK1_MODE_HEBBIAN, 0)).unwrap();
        assert_ne!(vm.rank1_layers.get(&0).copied().unwrap(), dirty);
        // ABORT nomeando v1 (alvo: filho 2 em r6; ts=v1 em r3):
        // descarta v2, aplica v1 => mapa volta a {0: h_a}.
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, v1 as u128).unwrap();
        vm.step_instruction(cid, &instr_abort(6, 3)).unwrap();
        assert_eq!(vm.rank1_layers.get(&0).copied(), Some(ha));
        // H_a intacto (CoW): 4 uns.
        assert_eq!(vm.memory.read_f32_tensor(ha, 4).unwrap(), vec![1.0; 4]);
    }

    // ---- RFC-0010: KV_TRUNCATE --------------------------------------

    #[tokio::test]
    async fn test_kv_truncate_shrink_noop_rollback() {
        use crate::opcodes::instr_kv_truncate;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        vm.memory.kv_cache_init(4, 8);
        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        for _ in 0..5 {
            for layer in 0..4 {
                vm.memory.kv_cache_append(layer, &k, &v).unwrap();
            }
        }
        assert_eq!(vm.memory.kv_cache_seq_len(), 5);
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // Shrink p/ 2 via reg.
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, 2u128).unwrap();
        vm.step_instruction(cid, &instr_kv_truncate(0, 0)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 2);
        // len >= atual: no-op Ok.
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, 99u128).unwrap();
        vm.step_instruction(cid, &instr_kv_truncate(0, 0)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 2);
        // Rollback: snapshot -> append -> truncate -> restore volta.
        let snap = vm.memory.snapshot();
        for layer in 0..4 {
            vm.memory.kv_cache_append(layer, &k, &v).unwrap();
        }
        assert_eq!(vm.memory.kv_cache_seq_len(), 3);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, 1u128).unwrap();
        vm.step_instruction(cid, &instr_kv_truncate(0, 0)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 1);
        vm.memory.restore(snap).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 2);
        // Stream != 0 roteia (RFC-0033): stream 3 inexistente => no-op
        // Ok, stream 0 intocado.
        let mut s3 = instr_kv_truncate(0, 0);
        s3.set_kv_stream(3);
        vm.step_instruction(cid, &s3).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 2);
        assert_eq!(vm.memory.kv_cache_seq_len_stream(3), 0);
        // sid > 16 veta alto (17 streams).
        let mut bad = instr_kv_truncate(0, 0);
        bad.set_kv_stream(99);
        assert!(vm.step_instruction(cid, &bad).is_err());
        assert_eq!(vm.stats.kv_truncate_execs, 4);
    }

    #[tokio::test]
    async fn test_kv_truncate_empty_is_noop() {
        use crate::opcodes::instr_kv_truncate;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        vm.memory.kv_cache_init(2, 4);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, 0u128)]);
        vm.step_instruction(cid, &instr_kv_truncate(0, 0)).unwrap();
        assert_eq!(vm.memory.kv_cache_seq_len(), 0);
        assert_eq!(vm.stats.kv_truncate_execs, 1);
    }

    // ---- RFC-0009: SAMPLE seeded ------------------------------------

    #[tokio::test]
    async fn test_sample_seeded_two_vms_agree() {
        use crate::opcodes::{instr_rng_seed, instr_sample};
        // Programa: seed fixa -> SAMPLE. Duas VMs => mesmo token, last igual.
        async fn run_once() -> (u32, u32) {
            let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
            let cid = rfc0004_ctx_with(&mut vm, &[]);
            let la = rfc0004_f32(&mut vm, &[4], &[0.2, 1.5, -0.7, 0.9]);
            vm.scheduler.get_mut(cid).unwrap().set_reg(0, la).unwrap();
            vm.scheduler.get_mut(cid).unwrap().set_reg(1, 424242u128).unwrap();
            vm.step_instruction(cid, &instr_rng_seed(1)).unwrap();
            vm.step_instruction(cid, &instr_sample(2, 0, 1.0)).unwrap();
            let tok = vm.scheduler.get(cid).unwrap().reg(2).unwrap() as u32;
            (tok, vm.last_sample)
        }
        let (t1, l1) = run_once().await;
        let (t2, l2) = run_once().await;
        assert_eq!((t1, l1), (t2, l2), "seed fixa => replay exato entre VMs");
        // Reseed replays: mesma seed de novo => mesmo token.
        let (t3, _) = run_once().await;
        assert_eq!(t1, t3);
    }

    // ---- RFC-0006: telemetria + scheduler ---------------------------

    #[tokio::test]
    async fn test_telemetry_sanity_preempt_assert_dump() {
        use crate::opcodes::{
            instr_assert, instr_cycles_count, instr_dump, instr_preempt_check,
            instr_sanity_check, instr_trace_event,
        };
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        // Tensor com NaN/Inf: [1.0, NaN, 2.0, +Inf] => saneado [1,0,2,0], count 2.
        let t = rfc0004_f32(&mut vm, &[4], &[1.0, f32::NAN, 2.0, f32::INFINITY]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, t)]);
        vm.step_instruction(cid, &instr_sanity_check(4, 0, 1)).unwrap();
        let clean = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(clean, 4).unwrap(), vec![1.0, 0.0, 2.0, 0.0]);
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(1).unwrap(), 2);
        // Original intacto (CoW): ainda tem NaN.
        assert!(vm.memory.read_f32_tensor(t, 4).unwrap()[1].is_nan());
        // Sem rCount: funciona, descarta contagem.
        vm.step_instruction(cid, &instr_sanity_check(4, 0, 0xFF)).unwrap();
        // PREEMPT_CHECK sem consumir: flag 0 => 0; com flag => 1 e segue setada.
        vm.step_instruction(cid, &instr_preempt_check(5)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(5).unwrap(), 0);
        vm.scheduler.get_mut(cid).unwrap().interrupt_flag = true;
        vm.step_instruction(cid, &instr_preempt_check(5)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(5).unwrap(), 1);
        assert!(vm.scheduler.get(cid).unwrap().interrupt_flag, "PREEMPT_CHECK não consome");
        vm.scheduler.get_mut(cid).unwrap().interrupt_flag = false;
        // ASSERT: passa com != 0 (qualquer code), trap com == 0 e code exato.
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, 7).unwrap();
        vm.step_instruction(cid, &instr_assert(6, 42)).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, 0).unwrap();
        let err = vm.step_instruction(cid, &instr_assert(6, 42)).unwrap_err();
        assert!(err.to_string().contains("42"), "trap carrega o code: {}", err);
        // CYCLES_COUNT / TRACE_EVENT / DUMP executam e contabilizam.
        vm.step_instruction(cid, &instr_cycles_count(7)).unwrap();
        assert!(vm.scheduler.get(cid).unwrap().reg(7).unwrap() > 0);
        vm.scheduler.get_mut(cid).unwrap().set_reg(8, 99).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(9, 0xABCD).unwrap();
        vm.step_instruction(cid, &instr_trace_event(8, 9)).unwrap();
        assert_eq!(vm.trace.back(), Some(&(99, 0xABCD)));
        vm.step_instruction(cid, &instr_dump()).unwrap();
        assert_eq!(vm.stats.sanity_execs, 2);
        assert_eq!(vm.stats.preempt_check_execs, 2);
        assert_eq!(vm.stats.assert_execs, 1);
        assert_eq!(vm.stats.cycles_execs, 1);
        assert_eq!(vm.stats.trace_execs, 1);
        assert_eq!(vm.stats.dump_execs, 1);
    }

    #[tokio::test]
    async fn test_scheduler_deadline_priority_locks_fence() {
        use crate::opcodes::{
            instr_fence, instr_get_deadline, instr_lock, instr_priority_get,
            instr_priority_set, instr_set_deadline, instr_unlock, instr_yield,
        };
        use crate::context::Priority;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // Deadline default = MAX (best-effort); set/get roundtrip.
        vm.step_instruction(cid, &instr_get_deadline(0)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(0).unwrap(), u64::MAX as u128);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, 1_000_000u128).unwrap();
        vm.step_instruction(cid, &instr_set_deadline(0)).unwrap();
        vm.step_instruction(cid, &instr_get_deadline(1)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(1).unwrap(), 1_000_000);
        // Prioridade: GET=Green(0); SET RED via reg; GET=2; volta p/ Green.
        vm.step_instruction(cid, &instr_priority_get(2)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(2).unwrap(), 0);
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, 2u128).unwrap();
        vm.step_instruction(cid, &instr_priority_set(3)).unwrap();
        assert!(matches!(vm.scheduler.get(cid).unwrap().priority, Priority::Red));
        vm.step_instruction(cid, &instr_priority_get(2)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(2).unwrap(), 2);
        // Valor inválido: erro limpo, prioridade intacta.
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, 7u128).unwrap();
        assert!(vm.step_instruction(cid, &instr_priority_set(3)).is_err());
        assert!(matches!(vm.scheduler.get(cid).unwrap().priority, Priority::Red));
        vm.scheduler.get_mut(cid).unwrap().set_reg(3, 0u128).unwrap();
        vm.step_instruction(cid, &instr_priority_set(3)).unwrap();
        assert!(matches!(vm.scheduler.get(cid).unwrap().priority, Priority::Green));
        // LOCK: adquire, reentrante p/ o dono, UNLOCK libera.
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, 77u128).unwrap();
        vm.step_instruction(cid, &instr_lock(4)).unwrap();
        assert_eq!(vm.locks.get(&77), Some(&cid));
        vm.step_instruction(cid, &instr_lock(4)).unwrap(); // reentrante ok
        vm.step_instruction(cid, &instr_unlock(4)).unwrap();
        assert!(!vm.locks.contains_key(&77));
        // UNLOCK sem posse: erro; LOCK de outro ctx: contenção.
        assert!(vm.step_instruction(cid, &instr_unlock(4)).is_err());
        let cid2 = rfc0004_ctx_with(&mut vm, &[(4, 77u128)]);
        vm.step_instruction(cid, &instr_lock(4)).unwrap();
        assert!(vm.step_instruction(cid2, &instr_lock(4)).is_err());
        assert!(vm.step_instruction(cid2, &instr_unlock(4)).is_err());
        vm.step_instruction(cid, &instr_unlock(4)).unwrap();
        vm.step_instruction(cid2, &instr_lock(4)).unwrap(); // liberou => ok
        // YIELD + FENCE executam.
        vm.step_instruction(cid, &instr_yield()).unwrap();
        vm.step_instruction(cid, &instr_fence()).unwrap();
        assert_eq!(vm.stats.set_deadline_execs, 1);
        assert_eq!(vm.stats.priority_set_execs, 2);
        assert_eq!(vm.stats.lock_execs, 4);
        assert_eq!(vm.stats.unlock_execs, 2);
        assert_eq!(vm.stats.yield_execs, 1);
        assert_eq!(vm.stats.fence_execs, 1);
    }

    #[tokio::test]
    async fn test_rfc0006_assembled_program_runs() {
        use crate::opcodes::assemble;
        // ASSERT sobre r0 (CYCLES>0 garante passa; ASSERT sobre r4==0 travaria).
        let src = r#"
            CYCLES_COUNT r0
            TRACE_EVENT r0, r0
            TENSOR r1 2 2 f32
            SANITY_CHECK r2, r1, r3
            PREEMPT_CHECK r4
            ASSERT r0
            DUMP
            YIELD
            SET_DEADLINE r0
            GET_DEADLINE r5
            PRIORITY_GET r6
            FENCE
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.cycles_execs, 1);
        assert_eq!(stats.trace_execs, 1);
        assert_eq!(stats.sanity_execs, 1);
        assert_eq!(stats.preempt_check_execs, 1);
        assert_eq!(stats.assert_execs, 1);
        assert_eq!(stats.dump_execs, 1);
        assert_eq!(stats.yield_execs, 1);
        assert_eq!(stats.set_deadline_execs, 1);
        assert_eq!(stats.get_deadline_execs, 1);
        assert_eq!(stats.priority_get_execs, 1);
        assert_eq!(stats.fence_execs, 1);
    }

    // ---- RFC-0007: LOADI / MOV / predicados -------------------------

    #[tokio::test]
    async fn test_loadi_mov_chain() {
        use crate::opcodes::{instr_loadi, instr_mov};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        vm.step_instruction(cid, &instr_loadi(0, 80_000_000)).unwrap();
        vm.step_instruction(cid, &instr_loadi(1, u128::MAX)).unwrap();
        vm.step_instruction(cid, &instr_mov(2, 0)).unwrap();
        vm.step_instruction(cid, &instr_mov(3, 1)).unwrap();
        let ctx = vm.scheduler.get(cid).unwrap();
        assert_eq!(ctx.reg(0).unwrap(), 80_000_000);
        assert_eq!(ctx.reg(1).unwrap(), u128::MAX);
        assert_eq!(ctx.reg(2).unwrap(), 80_000_000);
        assert_eq!(ctx.reg(3).unwrap(), u128::MAX);
        assert_eq!(vm.stats.loadi_execs, 2);
        assert_eq!(vm.stats.mov_execs, 2);
    }

    #[tokio::test]
    async fn test_compare_all_predicates() {
        use crate::opcodes::{instr_compare, CMP_GE, CMP_GT, CMP_LE, CMP_LT, CMP_NE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // (a, b, pred) -> esperado. Inclui fronteiras 0 e MAX.
        let cases: Vec<(u128, u128, u8, bool)> = vec![
            (5, 5, crate::opcodes::CMP_EQ, true),
            (5, 6, crate::opcodes::CMP_EQ, false),
            (5, 6, CMP_NE, true),
            (5, 5, CMP_NE, false),
            (5, 6, CMP_LT, true),
            (6, 5, CMP_LT, false),
            (5, 5, CMP_LT, false),
            (5, 5, CMP_LE, true),
            (6, 5, CMP_LE, false),
            (6, 5, CMP_GT, true),
            (5, 6, CMP_GT, false),
            (5, 5, CMP_GE, true),
            (4, 5, CMP_GE, false),
            (0, 0, CMP_LE, true),
            (0, 1, CMP_LT, true),
            (u128::MAX, u128::MAX, CMP_GE, true),
            (u128::MAX, 0, CMP_GT, true),
            (0, u128::MAX, CMP_LT, true),
        ];
        for (a, b, pred, want) in cases {
            let mut ins = instr_compare(0, 0xFF, b);
            // rsrc1=0 carrega `a` via reg 0.
            vm.scheduler.get_mut(cid).unwrap().set_reg(0, a).unwrap();
            ins.rsrc1 = 0;
            ins.set_compare_pred(pred);
            vm.step_instruction(cid, &ins).unwrap();
            assert_eq!(vm.scheduler.get(cid).unwrap().cmp_equal, want, "cmp {} pred={} {}", a, pred, b);
        }
        // Predicado inválido: trap limpo.
        let mut bad = instr_compare(0, 0xFF, 1);
        bad.set_compare_pred(9);
        assert!(vm.step_instruction(cid, &bad).is_err());
    }

    #[tokio::test]
    async fn test_rfc0007_assembled_program_runs() {
        use crate::opcodes::assemble;
        // Deadline + lock-id + prioridade como literais (o gap chicken-and-egg).
        let src = r#"
            LOADI r0, 80000000
            LOADI r1, 77
            LOADI r2, 2
            MOV r3, r0
            SET_DEADLINE r0
            GET_DEADLINE r4
            PRIORITY_SET r2
            PRIORITY_GET r5
            LOCK r1
            UNLOCK r1
            COMPARE r3, r0 PRED=EQ
            HALT
        "#;
        let prog = assemble(src).unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(60), ..Default::default() });
        vm.load_program(prog);
        let stats = vm.run().unwrap();
        assert_eq!(stats.loadi_execs, 3);
        assert_eq!(stats.mov_execs, 1);
        assert_eq!(stats.set_deadline_execs, 1);
        assert_eq!(stats.priority_set_execs, 1);
        assert_eq!(stats.lock_execs, 1);
        assert_eq!(stats.unlock_execs, 1);
        let ctx_id = *vm.scheduler.contexts().keys().next().expect("ctx");
        let ctx = vm.scheduler.get(ctx_id).unwrap();
        assert_eq!(ctx.reg(3).unwrap(), 80_000_000);
        assert_eq!(ctx.reg(4).unwrap(), 80_000_000);
        assert!(ctx.cmp_equal, "80M == 80M via PRED=EQ");
    }

    // ---- RFC-0012: FOREST goldens -----------------------------------

    /// Ensemble de referência: 2 árvores, depth 2 (stride 3).
    /// T0: f0>0.5 ? L20 : L10 · T1: f1>1.0 ? L50 : L40.
    fn rfc0012_ensemble(vm: &mut Vm) -> (u128, u128, u128) {
        let feats = rfc0004_f32(vm, &[2], &[0.7, 0.3]);
        let table = rfc0004_f32(
            vm,
            &[6, 4],
            &[
                0.0, 0.5, 1.0, 2.0, // t0n0 root
                0.0, 0.0, -1.0, -1.0, // t0n1 leaf
                0.0, 0.0, -1.0, -1.0, // t0n2 leaf
                1.0, 1.0, 1.0, 2.0, // t1n0 root
                0.0, 0.0, -1.0, -1.0, // t1n1 leaf
                0.0, 0.0, -1.0, -1.0, // t1n2 leaf
            ],
        );
        let leaves = rfc0004_f32(vm, &[6], &[9.0, 10.0, 20.0, 30.0, 40.0, 50.0]);
        (feats, table, leaves)
    }

    #[tokio::test]
    async fn test_forest_golden_vote_mean() {
        use crate::opcodes::{instr_forest, FOREST_MODE_MEAN, FOREST_MODE_VOTE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let (f, t, l) = rfc0012_ensemble(&mut vm);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, f), (1, t), (2, l)]);
        // f=[0.7,0.3]: T0 -> dir (20), T1 -> esq (40).
        vm.step_instruction(cid, &instr_forest(4, 0, 1, 2, 2, 2, FOREST_MODE_VOTE)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 2).unwrap(), vec![20.0, 40.0]);
        vm.step_instruction(cid, &instr_forest(4, 0, 1, 2, 2, 2, FOREST_MODE_MEAN)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 1).unwrap(), vec![30.0]);
        assert_eq!(vm.stats.forest_execs, 2);
    }

    #[tokio::test]
    async fn test_forest_errors_and_termination() {
        use crate::opcodes::{instr_forest, FOREST_MODE_VOTE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let (f, t, l) = rfc0012_ensemble(&mut vm);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, f), (1, t), (2, l)]);
        // feat OOB: tabela com feat_idx=5, F=2.
        let bad_t = rfc0004_f32(&mut vm, &[3, 4], &[5.0, 0.0, 1.0, 2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, bad_t).unwrap();
        assert!(vm.step_instruction(cid, &instr_forest(4, 0, 1, 2, 1, 2, FOREST_MODE_VOTE)).is_err());
        // filho OOB: right=99, stride=3.
        let bad_c = rfc0004_f32(&mut vm, &[3, 4], &[0.0, 0.0, 1.0, 99.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, bad_c).unwrap();
        assert!(vm.step_instruction(cid, &instr_forest(4, 0, 1, 2, 1, 2, FOREST_MODE_VOTE)).is_err());
        // Folha NaN: trap (fail-closed). feats[0]=0.7>0.5 => slot 2 => NaN.
        let one = rfc0004_f32(&mut vm, &[3, 4], &[0.0, 0.5, 1.0, 2.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0]);
        let nan_l = rfc0004_f32(&mut vm, &[3], &[0.0, 0.0, f32::NAN]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, one).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, nan_l).unwrap();
        assert!(vm.step_instruction(cid, &instr_forest(4, 0, 1, 2, 1, 2, FOREST_MODE_VOTE)).is_err());
        // Tabela cíclica (zeros): termina pelo teto depth, sem hang.
        let cyc = rfc0004_f32(&mut vm, &[3, 4], &[0.0; 12]);
        let cyc_l = rfc0004_f32(&mut vm, &[3], &[7.0, 0.0, 0.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, cyc).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, cyc_l).unwrap();
        vm.step_instruction(cid, &instr_forest(4, 0, 1, 2, 1, 2, FOREST_MODE_VOTE)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 1).unwrap(), vec![7.0]);
        // Modo inválido via ctor direto: erro limpo.
        let mut badm = instr_forest(4, 0, 1, 2, 1, 2, FOREST_MODE_VOTE);
        badm.payload[4] = 9;
        assert!(vm.step_instruction(cid, &badm).is_err());
    }

    #[tokio::test]
    async fn test_rfc0012_ramp_table_traps() {
        // TENSOR inicializa em ramp (i+1)*0.5: tabelas ramp NUNCA têm
        // índices integrais => FOREST sobre TENSOR cru dá Err determinístico.
        // Programa .m3asm não constrói tabelas válidas sem fill de literais
        // (follow-up RFC-0012); goldens acima cobrem a execução real.
        use crate::opcodes::assemble;
        let prog = assemble("TENSOR r0 1 2 f32\nTENSOR r1 2 4 f32\nTENSOR r2 2 f32\nFOREST r4, r0, r1, r2 TREES=1 DEPTH=1 MODE=VOTE\nHALT").unwrap();
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let root = vm.memory.current_version();
        let cid = vm.scheduler.create_context(crate::context::Priority::Green, 0x1000, root);
        for ins in prog.iter().take(3) {
            vm.step_instruction(cid, ins).unwrap();
        }
        assert!(vm.step_instruction(cid, &prog[3]).is_err());
        assert_eq!(vm.stats.forest_execs, 0);
    }

    // ---- RFC-0013: DENOISE goldens ----------------------------------

    #[tokio::test]
    async fn test_denoise_sigma0_manual() {
        use crate::opcodes::instr_denoise_step;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // x=[4,2], eps=[1,0], abar=0.25, beta=0.36:
        // alpha=0.64, coef=0.36/sqrt(0.75), out=[4.480385, 2.5] (1e-4).
        let x = rfc0004_f32(&mut vm, &[2], &[4.0, 2.0]);
        let e = rfc0004_f32(&mut vm, &[2], &[1.0, 0.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, e)]);
        let st_before = vm.scheduler.get(cid).unwrap().rng_state;
        vm.step_instruction(cid, &instr_denoise_step(4, 0, 1, 0xFF, 0.25, 0.36, 0.0, 7)).unwrap();
        // sigma=0 NÃO consome RNG (reseed semantics limpas).
        assert_eq!(vm.scheduler.get(cid).unwrap().rng_state, st_before);
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        let v = vm.memory.read_f32_tensor(out, 2).unwrap();
        assert!((v[0] - 4.480385).abs() < 1e-4, "got {}", v[0]);
        assert!((v[1] - 2.5).abs() < 1e-4, "got {}", v[1]);
        assert_eq!(vm.stats.denoise_execs, 1);
    }

    #[tokio::test]
    async fn test_denoise_seeded_replay() {
        use crate::opcodes::{instr_denoise_step, instr_rng_seed};
        async fn run_once() -> (Vec<f32>, u64) {
            let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
            let x = rfc0004_f32(&mut vm, &[4], &[0.5, -1.0, 2.0, 0.0]);
            let e = rfc0004_f32(&mut vm, &[4], &[0.1, 0.2, -0.3, 0.4]);
            let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, e), (2, 777u128)]);
            vm.step_instruction(cid, &instr_rng_seed(2)).unwrap();
            vm.step_instruction(cid, &instr_denoise_step(4, 0, 1, 0xFF, 0.9, 0.05, 0.3, 42)).unwrap();
            let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
            let st = vm.scheduler.get(cid).unwrap().rng_state;
            (vm.memory.read_f32_tensor(out, 4).unwrap(), st)
        }
        let (a, sa) = run_once().await;
        let (b, sb) = run_once().await;
        assert!(a.iter().all(|v| v.is_finite()));
        assert_eq!(a, b, "mesma seed => mesmo rollout estocástico");
        assert_eq!(sa, sb, "stream avança igual");
        // Checagem independente da fiação: mesmos draws manuais.
        let mut st = 777u64;
        // NOTE: RNG_SEED usa o valor do reg como semente direta.
        let mut draws = Vec::new();
        for _ in 0..4 {
            draws.push(crate::determinism::normal_f32(&mut st, 0.0, 1.0));
        }
        let (abar, beta, sigma) = (0.9f32, 0.05f32, 0.3f32);
        let alpha = 1.0 - beta;
        let coef = beta / (1.0 - abar).sqrt();
        let inv = 1.0 / alpha.sqrt();
        let x = [0.5f32, -1.0, 2.0, 0.0];
        let e = [0.1f32, 0.2, -0.3, 0.4];
        for (i, (got, z)) in a.iter().zip(draws.iter()).enumerate() {
            let want = (x[i] - coef * e[i]) * inv + sigma * z;
            assert!((got - want).abs() < 1e-5, "i={} got={} want={}", i, got, want);
        }
    }

    #[tokio::test]
    async fn test_denoise_param_errors() {
        use crate::opcodes::instr_denoise_step;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let x = rfc0004_f32(&mut vm, &[2], &[1.0, 2.0]);
        let e = rfc0004_f32(&mut vm, &[2], &[0.1, 0.2]);
        let bad = rfc0004_f32(&mut vm, &[3], &[0.0; 3]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, e), (2, bad)]);
        // Parâmetros inválidos.
        for (a, b, s) in [
            (0.0, 0.02, 0.0),   // abar <= 0
            (-0.5, 0.02, 0.0),  // abar negativo
            (1.5, 0.02, 0.0),   // abar > 1
            (0.9, -0.1, 0.0),   // beta negativo
            (0.9, 1.0, 0.0),    // beta >= 1
            (0.9, 0.02, -0.5),  // sigma negativo
            (f32::NAN, 0.02, 0.0),
            (0.9, f32::INFINITY, 0.0),
        ] {
            assert!(
                vm.step_instruction(cid, &instr_denoise_step(4, 0, 1, 0xFF, a, b, s, 1)).is_err(),
                "params ({},{},{}) devem falhar",
                a, b, s
            );
        }
        // abar=1 com eps não-nulo: divisão por zero.
        assert!(vm.step_instruction(cid, &instr_denoise_step(4, 0, 1, 0xFF, 1.0, 0.02, 0.0, 1)).is_err());
        // Shape mismatch e rsrc3 com tensor.
        assert!(vm.step_instruction(cid, &instr_denoise_step(4, 0, 2, 0xFF, 0.9, 0.02, 0.0, 1)).is_err());
        assert!(vm.step_instruction(cid, &instr_denoise_step(4, 0, 1, 2, 0.9, 0.02, 0.0, 1)).is_err());
        assert_eq!(vm.stats.denoise_execs, 0, "erros não contabilizam");
    }

    // ---- RFC-0014: ODE goldens --------------------------------------

    fn rfc0014_pack(vm: &mut Vm, w: &[f32], b: &[f32]) -> u128 {
        let mut flat = w.to_vec();
        flat.extend_from_slice(b);
        rfc0004_f32(vm, &[flat.len()], &flat)
    }

    #[tokio::test]
    async fn test_ode_euler_manual() {
        use crate::opcodes::{instr_ode_step, ODE_METHOD_EULER};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // n=1, W=[0], b=[1]: f = SILU(1) = 0.7310586; x0=2, dt=0.5:
        // out = 2 + 0.5*0.7310586 = 2.3655293.
        let x = rfc0004_f32(&mut vm, &[1], &[2.0]);
        let wb = rfc0014_pack(&mut vm, &[0.0], &[1.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (2, wb)]);
        vm.step_instruction(cid, &instr_ode_step(3, 0, 0xFF, 2, 0.5, ODE_METHOD_EULER, 0)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(3).unwrap();
        let v = vm.memory.read_f32_tensor(out, 1).unwrap();
        assert!((v[0] - 2.3655293).abs() < 1e-5, "got {}", v[0]);
        assert_eq!(vm.stats.ode_execs, 1);
    }

    #[tokio::test]
    async fn test_ode_rk4_vs_fine_euler() {
        use crate::opcodes::{instr_ode_step, ODE_METHOD_RK4};
        // Campo default (W=-0.1I) com controle: RK4 dt=0.1 vs referência
        // Euler dt=0.0001 (implementação independente inline, não ode_field).
        fn silu_ref(v: f32) -> f32 {
            if v >= 0.0 { v / (1.0 + (-v).exp()) } else { let e = v.exp(); v * e / (1.0 + e) }
        }
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let x = rfc0004_f32(&mut vm, &[2], &[1.0, -0.5]);
        let u = rfc0004_f32(&mut vm, &[1], &[0.2]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, u)]);
        vm.step_instruction(cid, &instr_ode_step(3, 0, 1, 0xFF, 0.1, ODE_METHOD_RK4, 0)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(3).unwrap();
        let got = vm.memory.read_f32_tensor(out, 2).unwrap();
        // Referência: Euler fino, campo reescrito à mão.
        let mut xr = [1.0f32, -0.5f32];
        let dt = 0.0001f32;
        for _ in 0..1000 {
            // f = SILU(-0.1x + û), û=[0.2, 0].
            let f0 = silu_ref(-0.1 * xr[0] + 0.2);
            let f1 = silu_ref(-0.1 * xr[1]);
            xr[0] += dt * f0;
            xr[1] += dt * f1;
        }
        assert!((got[0] - xr[0]).abs() < 1e-3, "got {} want {}", got[0], xr[0]);
        assert!((got[1] - xr[1]).abs() < 1e-3, "got {} want {}", got[1], xr[1]);
    }

    #[tokio::test]
    async fn test_ode_default_contracts_and_errors() {
        use crate::opcodes::{instr_ode_step, ODE_METHOD_EULER};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let x = rfc0004_f32(&mut vm, &[2], &[2.0, -2.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x)]);
        // Default contrativo: |x| encolhe num passo Euler.
        vm.step_instruction(cid, &instr_ode_step(3, 0, 0xFF, 0xFF, 0.1, ODE_METHOD_EULER, 0)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(3).unwrap();
        let v = vm.memory.read_f32_tensor(out, 2).unwrap();
        assert!(v[0] < 2.0 && v[1] > -2.0, "contração: {:?}", v);
        assert!(v.iter().all(|z| z.is_finite()));
        // Params inválidos e shapes ruins: erro limpo, sem contabilizar.
        let bad_dt = instr_ode_step(3, 0, 0xFF, 0xFF, 0.0, ODE_METHOD_EULER, 0);
        assert!(vm.step_instruction(cid, &bad_dt).is_err());
        let mut bad_dt = instr_ode_step(3, 0, 0xFF, 0xFF, 0.1, ODE_METHOD_EULER, 0);
        bad_dt.payload[0..4].copy_from_slice(&(-1.0f32).to_le_bytes());
        assert!(vm.step_instruction(cid, &bad_dt).is_err());
        let mut bad_m = instr_ode_step(3, 0, 0xFF, 0xFF, 0.1, ODE_METHOD_EULER, 0);
        bad_m.payload[4] = 9;
        assert!(vm.step_instruction(cid, &bad_m).is_err());
        // Pack com tamanho errado: n=2 precisa 2*3=6 elems.
        let short = rfc0004_f32(&mut vm, &[4], &[1.0; 4]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, short).unwrap();
        assert!(vm.step_instruction(cid, &instr_ode_step(3, 0, 0xFF, 5, 0.1, ODE_METHOD_EULER, 0)).is_err());
        // x inexistente.
        let nox = instr_ode_step(3, 0, 0xFF, 0xFF, 0.1, ODE_METHOD_EULER, 0);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, 0xDEADu128).unwrap();
        assert!(vm.step_instruction(cid, &nox).is_err());
        assert_eq!(vm.stats.ode_execs, 1);
    }

    // ---- RFC-0015: SPIKE goldens ------------------------------------

    #[tokio::test]
    async fn test_spike_lif_golden_refractory() {
        use crate::opcodes::instr_spike_step;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        // n=3, thresh=1, decay=0.5, reset=0, refr=2, I=[2,0.5,4] constante.
        // Passo 1: V=[1,.25,2] -> spikes [1,0,1] (==thresh dispara),
        //   V=[0,.25,0], refr=[2,0,2].
        // Passo 2: V=[1,.375,2], refr=[2,0,2] => ninguém dispara (refratários
        //   seguram 0 e 2), spikes [0,0,0], refr=[1,0,1].
        // Passo 4 (após passo 3 sem disparos): [1,0,1] de novo (ciclo).
        let v = rfc0004_f32(&mut vm, &[3], &[0.0, 0.0, 0.0]);
        let inp = rfc0004_f32(&mut vm, &[3], &[2.0, 0.5, 4.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(1, v), (2, inp)]);
        let step = instr_spike_step(3, 1, 2, 0xFF, 1.0, 0.5, 0.0, 0, 2);
        let run = |vm: &mut Vm, cid: u64| -> (Vec<f32>, Vec<f32>) {
            vm.step_instruction(cid, &step).unwrap();
            let s = vm.scheduler.get(cid).unwrap().reg(3).unwrap();
            let vv = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
            (
                vm.memory.read_f32_tensor(s, 3).unwrap(),
                vm.memory.read_f32_tensor(vv, 3).unwrap(),
            )
        };
        // Passo 1: V inicial é religado (CoW) — realimenta via r1.
        let (s1, v1) = run(&mut vm, cid);
        assert_eq!(s1, vec![1.0, 0.0, 1.0]);
        assert_eq!(v1, vec![0.0, 0.25, 0.0]);
        let (s2, v2) = run(&mut vm, cid);
        assert_eq!(s2, vec![0.0, 0.0, 0.0]);
        assert_eq!(v2, vec![1.0, 0.375, 2.0]);
        let (s3, _) = run(&mut vm, cid);
        assert_eq!(s3, vec![0.0, 0.0, 0.0]);
        let (s4, v4) = run(&mut vm, cid);
        assert_eq!(s4, vec![1.0, 0.0, 1.0], "ciclo refratário fecha");
        assert_eq!(v4, vec![0.0, 0.46875, 0.0]);
        assert_eq!(vm.stats.spike_execs, 4);
    }

    #[tokio::test]
    async fn test_spike_pack_errors_rollback() {
        use crate::opcodes::{instr_abort, instr_fork, instr_spike_step};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(40), ..Default::default() });
        let v = rfc0004_f32(&mut vm, &[2], &[0.0, 0.0]);
        let inp = rfc0004_f32(&mut vm, &[2], &[5.0, 0.1]);
        let cid = rfc0004_ctx_with(&mut vm, &[(1, v), (2, inp)]);
        // Pack [thresh,decay,reset,refr] vence payload.
        let pack = rfc0004_f32(&mut vm, &[4], &[0.5, 1.0, -1.0, 1.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, pack).unwrap();
        // Payload diria thresh=99 (nunca dispara); pack diz 0.5 (dispara n0).
        vm.step_instruction(cid, &instr_spike_step(3, 1, 2, 0, 99.0, 0.9, 0.0, 1, 2)).unwrap();
        let s = vm.scheduler.get(cid).unwrap().reg(3).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(s, 2).unwrap(), vec![1.0, 0.0]);
        // V religado: n0 resetou p/ -1 (do pack), n1 = (0+0.1)*1 = 0.1.
        let vv = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(vv, 2).unwrap(), vec![-1.0, 0.1]);
        // Pack inválido: len, refr fracionário, decay fora de [0,1].
        let bad_len = rfc0004_f32(&mut vm, &[3], &[1.0; 3]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, bad_len).unwrap();
        assert!(vm.step_instruction(cid, &instr_spike_step(3, 1, 2, 0, 1.0, 0.9, 0.0, 1, 2)).is_err());
        let bad_r = rfc0004_f32(&mut vm, &[4], &[1.0, 0.9, 0.0, 2.5]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, bad_r).unwrap();
        assert!(vm.step_instruction(cid, &instr_spike_step(3, 1, 2, 0, 1.0, 0.9, 0.0, 1, 2)).is_err());
        // decay via payload fora de [0,1].
        let mut bad_d = instr_spike_step(3, 1, 2, 0xFF, 1.0, 1.5, 0.0, 1, 2);
        let _ = &bad_d;
        assert!(vm.step_instruction(cid, &bad_d).is_err());
        // Shape mismatch V/I.
        let bad_i = rfc0004_f32(&mut vm, &[3], &[1.0; 3]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, bad_i).unwrap();
        assert!(vm.step_instruction(cid, &instr_spike_step(3, 1, 2, 0xFF, 1.0, 0.9, 0.0, 1, 2)).is_err());
        // Rollback: FORK, passo suja, ABORT(ts=v1) restaura handles; V velho intacto.
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, inp).unwrap();
        vm.step_instruction(cid, &instr_fork(5, 0)).unwrap();
        let v1 = vm.memory.current_version();
        let before = vm.snn_layers.clone();
        vm.step_instruction(cid, &instr_spike_step(3, 1, 2, 0xFF, 1.0, 0.9, 0.0, 1, 2)).unwrap();
        assert_ne!(vm.snn_layers.get(&1).copied(), before.get(&1).copied());
        vm.scheduler.get_mut(cid).unwrap().set_reg(4, v1 as u128).unwrap();
        vm.step_instruction(cid, &instr_abort(5, 4)).unwrap();
        vm.step_instruction(cid, &instr_abort(5, 4)).unwrap();
        assert_eq!(vm.snn_layers, before);
        assert_eq!(vm.stats.spike_execs, 2);
    }

    // ---- RFC-0017: CONV goldens -------------------------------------

    #[tokio::test]
    async fn test_conv_1d_basic() {
        use crate::opcodes::{instr_conv, CONV_ACT_NONE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // x=[1,1,5]=[1,2,3,4,5], k=[1,1,3]=[1,0,-1] (edge), stride 1.
        // out[0]=1*1+2*0+3*(-1)=-2; out[1]=2-4=-2; out[2]=3-5=-2.
        let x = rfc0004_f32(&mut vm, &[1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let k = rfc0004_f32(&mut vm, &[1, 1, 3], &[1.0, 0.0, -1.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, k)]);
        vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 1, CONV_ACT_NONE)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        let meta = vm.memory.get_tensor_meta(out).cloned().unwrap();
        assert_eq!(meta.shape, vec![1, 1, 3]);
        assert_eq!(vm.memory.read_f32_tensor(out, 3).unwrap(), vec![-2.0, -2.0, -2.0]);
        assert_eq!(vm.stats.conv_execs, 1);
    }

    #[tokio::test]
    async fn test_conv_2d_golden_stride_pad() {
        use crate::opcodes::{instr_conv, CONV_ACT_NONE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // x=[1,1,3,3]=1..9, k=[1,1,2,2]=[[1,0],[0,1]] (identidade diagonal).
        // stride 1, sem pad => [1,1,2,2]:
        // [0,0]=1*1+2*0+4*0+5*1=6; [0,1]=2+6=8; [1,0]=4+8=12; [1,1]=5+9=14.
        let x = rfc0004_f32(&mut vm, &[1, 1, 3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let k = rfc0004_f32(&mut vm, &[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, k)]);
        vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 1, CONV_ACT_NONE)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 4).unwrap(), vec![6.0, 8.0, 12.0, 14.0]);
        // stride 2, pad 1, bias 10: O=floor((3+2-2)/2)+1=2.
        // Padded 5x5 (borda 0); janela (oh,ow) cobre (2oh-1..2oh, 2ow-1..2ow)
        // com kernel [[1,0],[0,1]]: (0,0)->x[0,0]=1; (0,1)->x[0,2]=3;
        // (1,0)->x[2,0]=7; (1,1)->x[1,1]+x[2,2]=5+9=14. +10 cada.
        let b = rfc0004_f32(&mut vm, &[1], &[10.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, b).unwrap();
        vm.step_instruction(cid, &instr_conv(4, 0, 1, 2, 2, 1, 1, 1, CONV_ACT_NONE)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 4).unwrap(), vec![11.0, 13.0, 17.0, 24.0]);
    }

    #[tokio::test]
    async fn test_conv_depthwise_fused_dilation() {
        use crate::opcodes::{instr_conv, CONV_ACT_NONE, CONV_ACT_RELU, CONV_ACT_SILU};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        // Depthwise: Cin=Cout=2, G=2, kernel [2,1,3] (por canal).
        // canal0: x=[1,2,3,4,5], k=[1,0,-1] => [-2,-2,-2]; canal1: x=[5,4,3,2,1] => [2,2,2].
        let x = rfc0004_f32(&mut vm, &[1, 2, 5], &[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let k = rfc0004_f32(&mut vm, &[2, 1, 3], &[1.0, 0.0, -1.0, 1.0, 0.0, -1.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, k)]);
        vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 2, CONV_ACT_NONE)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        let meta = vm.memory.get_tensor_meta(out).cloned().unwrap();
        assert_eq!(meta.shape, vec![1, 2, 3]);
        assert_eq!(vm.memory.read_f32_tensor(out, 6).unwrap(), vec![-2.0, -2.0, -2.0, 2.0, 2.0, 2.0]);
        // Fused ReLU zera negativos; tamanho mantido.
        vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 2, CONV_ACT_RELU)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 6).unwrap(), vec![0.0, 0.0, 0.0, 2.0, 2.0, 2.0]);
        // Dilation 2 com o mesmo kernel [2,1,3]: eff=1+2*2=5, L=5 => O=1.
        // canal0: x[0]*1+x[2]*0+x[4]*-1 = 1-5 = -4; canal1: 5-1 = 4.
        vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 2, 2, CONV_ACT_NONE)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 2).unwrap(), vec![-4.0, 4.0]);
        // SILU fundido preserva shape e finitude.
        vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 2, CONV_ACT_SILU)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(4).unwrap();
        let v = vm.memory.read_f32_tensor(out, 6).unwrap();
        assert!(v.iter().all(|z| z.is_finite()));
        assert_eq!(vm.stats.conv_execs, 4);
    }

    #[tokio::test]
    async fn test_conv_errors() {
        use crate::opcodes::{instr_conv, CONV_ACT_NONE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let x = rfc0004_f32(&mut vm, &[1, 2, 4], &[1.0; 8]);
        let k = rfc0004_f32(&mut vm, &[2, 2, 2], &[1.0; 8]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, x), (1, k)]);
        // Cin kernel (2) != Cin/Grupos (2/2=1): layout exige [Cout,Cin/G].
        // (GROUPS=2 com kernel [2,2,2]: kcin=2 vs cin/g=1.)
        let ok2 = instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 2, CONV_ACT_NONE);
        assert!(vm.step_instruction(cid, &ok2).is_err());
        // Kernel maior que o input.
        let big = rfc0004_f32(&mut vm, &[1, 1, 9], &[1.0; 9]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, big).unwrap();
        assert!(vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 1, CONV_ACT_NONE)).is_err());
        // Groups que não dividem.
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, k).unwrap();
        assert!(vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 3, CONV_ACT_NONE)).is_err());
        // ACT inválido e stride 0.
        let mut bad = instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 1, CONV_ACT_NONE);
        bad.payload[7] = 9;
        assert!(vm.step_instruction(cid, &bad).is_err());
        let mut bads = instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 1, CONV_ACT_NONE);
        bads.payload[0] = 0;
        bads.payload[1] = 0;
        assert!(vm.step_instruction(cid, &bads).is_err());
        // Bias com tamanho errado.
        let bb = rfc0004_f32(&mut vm, &[3], &[0.0; 3]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, bb).unwrap();
        assert!(vm.step_instruction(cid, &instr_conv(4, 0, 1, 2, 1, 0, 1, 1, CONV_ACT_NONE)).is_err());
        // Rank 3D de volume (não suportado no MVP): kernel 4D com
        // input 1D (rank 3 == ndim+2 exigiria rank 3 no kernel).
        let v3 = rfc0004_f32(&mut vm, &[2, 2, 2], &[1.0; 8]);
        let k3 = rfc0004_f32(&mut vm, &[2, 2, 2], &[1.0; 8]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, v3).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, k3).unwrap();
        // [2,2,2] como input 1D-3ch [N=1,C=2,L=2] é válido! usa kernel 4D p/ falhar:
        let k4 = rfc0004_f32(&mut vm, &[1, 1, 1, 9], &[1.0; 9]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, k4).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, k4).unwrap();
        assert!(vm.step_instruction(cid, &instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 1, CONV_ACT_NONE)).is_err());
    }

    // ---- RFC-0018: cluster F1 ---------------------------------------

    #[tokio::test]
    async fn test_remote_spawn_and_signal_local() {
        use crate::opcodes::{
            instr_remote_spawn, instr_signal, SIGNAL_KIND_ABORT,
            SIGNAL_KIND_FORK_REQ, SIGNAL_KIND_HALT, SIGNAL_KIND_PING,
        };
        use crate::context::Priority;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        // Programa com label: SPAWN entra no HALT (filho termina sozinho).
        let prog = crate::opcodes::assemble("MAIN:\nHALT\n").unwrap();
        vm.load_program(prog);
        let base = vm.program_base;
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // SPAWN local (prio RED=2): filho nasce no HALT, regs zerados.
        vm.step_instruction(cid, &instr_remote_spawn(0, 0, base as u64, 2)).unwrap();
        let child = vm.scheduler.get(cid).unwrap().reg(0).unwrap() as u64;
        assert_ne!(child, cid);
        let cctx = vm.scheduler.get(child).unwrap();
        assert_eq!((cctx.pc, cctx.priority), (base, Priority::Red));
        assert_eq!(cctx.reg(0).unwrap(), 0, "spawn não herda regs (≠ fork)");
        // PING em ctx existente: ack 0.
        vm.step_instruction(cid, &instr_signal(1, SIGNAL_KIND_PING, 0, child, 7)).unwrap();
        assert_eq!(vm.scheduler.get(cid).unwrap().reg(1).unwrap(), 0);
        // PING em ctx inexistente: erro.
        assert!(vm.step_instruction(cid, &instr_signal(1, SIGNAL_KIND_PING, 0, 4242, 7)).is_err());
        // FORK_REQ local: erro explícito (use FORK).
        assert!(vm.step_instruction(cid, &instr_signal(1, SIGNAL_KIND_FORK_REQ, 0, child, 7)).is_err());
        // KIND inválido e nó remoto: erro.
        let mut badk = instr_signal(1, SIGNAL_KIND_PING, 0, child, 7);
        badk.rsrc1 = 9;
        assert!(vm.step_instruction(cid, &badk).is_err());
        assert!(vm.step_instruction(cid, &instr_signal(1, SIGNAL_KIND_PING, 1, child, 7)).is_err());
        assert!(vm.step_instruction(cid, &instr_remote_spawn(0, 3, base as u64, 0)).is_err());
        // ABORT via SIGNAL mata o filho.
        vm.step_instruction(cid, &instr_signal(1, SIGNAL_KIND_ABORT, 0, child, 7)).unwrap();
        assert!(vm.scheduler.get(child).is_none());
        // HALT mata sem pops: empilha ssm, HALT, conta intacta.
        let cid2 = rfc0004_ctx_with(&mut vm, &[]);
        vm.ssm_states.push(crate::ssm::MambaState::new(2, 1, 4));
        vm.step_instruction(cid2, &instr_remote_spawn(0, 0, base as u64, 0)).unwrap();
        let child2 = vm.scheduler.get(cid2).unwrap().reg(0).unwrap() as u64;
        let snaps_before = vm.ssm_snapshots.len();
        vm.step_instruction(cid2, &instr_signal(1, SIGNAL_KIND_HALT, 0, child2, 7)).unwrap();
        assert!(vm.scheduler.get(child2).is_none());
        assert_eq!(vm.ssm_snapshots.len(), snaps_before, "HALT não toca nas pilhas");
        assert_eq!(vm.stats.remote_spawn_execs, 2);
        assert_eq!(vm.stats.signal_execs, 3);
    }

    #[tokio::test]
    async fn test_send_tensor_copy_move() {
        use crate::opcodes::{instr_send_tensor, SEND_MODE_COPY, SEND_MODE_MOVE};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        // Fonte [4] f32 = 16B: [1,2,3,4]; destino [4] zeros.
        let s = rfc0004_f32(&mut vm, &[4], &[1.0, 2.0, 3.0, 4.0]);
        let d = rfc0004_f32(&mut vm, &[4], &[0.0; 4]);
        let cid = rfc0004_ctx_with(&mut vm, &[(5, s), (6, d)]);
        // COPY integral.
        vm.step_instruction(cid, &instr_send_tensor(5, 6, 0, 0, 16, SEND_MODE_COPY)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(d, 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        // COPY parcial [4..12) = bytes de [2.0, 3.0] sobre destino NÃO
        // zerado (prova preservação do restante): [9,9,9,9] -> [9,2,3,9].
        let d2 = rfc0004_f32(&mut vm, &[4], &[9.0; 4]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, d2).unwrap();
        vm.step_instruction(cid, &instr_send_tensor(5, 6, 0, 4, 8, SEND_MODE_COPY)).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(d2, 4).unwrap(), vec![9.0, 2.0, 3.0, 9.0]);
        // MOVE invalida a fonte de verdade (heap + meta).
        vm.step_instruction(cid, &instr_send_tensor(5, 6, 0, 0, 16, SEND_MODE_MOVE)).unwrap();
        assert!(vm.memory.get_tensor_meta(s).is_none(), "MOVE remove a meta");
        assert!(vm.memory.read(s, 16).is_err(), "MOVE remove o heap");
        assert_eq!(vm.memory.read_f32_tensor(d2, 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        // Erros: LEN 0, janela OOB, destino incompatível, dst 0xFF (F1),
        // modo inválido, nó remoto, fonte ausente.
        let s2 = rfc0004_f32(&mut vm, &[2], &[9.0, 9.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(5, s2).unwrap();
        assert!(vm.step_instruction(cid, &instr_send_tensor(5, 6, 0, 0, 0, SEND_MODE_COPY)).is_err());
        assert!(vm.step_instruction(cid, &instr_send_tensor(5, 6, 0, 4, 8, SEND_MODE_COPY)).is_err());
        let small = rfc0004_f32(&mut vm, &[1], &[0.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(6, small).unwrap();
        assert!(vm.step_instruction(cid, &instr_send_tensor(5, 6, 0, 0, 8, SEND_MODE_COPY)).is_err());
        assert!(vm.step_instruction(cid, &instr_send_tensor(5, 0xFF, 0, 0, 8, SEND_MODE_COPY)).is_err());
        let mut badm = instr_send_tensor(5, 6, 0, 0, 8, SEND_MODE_COPY);
        badm.payload[16] = 7;
        assert!(vm.step_instruction(cid, &badm).is_err());
        assert!(vm.step_instruction(cid, &instr_send_tensor(5, 6, 2, 0, 8, SEND_MODE_COPY)).is_err());
        assert_eq!(vm.stats.send_tensor_execs, 3);
    }

    #[tokio::test]
    async fn test_barrier_two_party_release() {
        use crate::opcodes::instr_barrier;
        use crate::context::ContextState;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let a = rfc0004_ctx_with(&mut vm, &[]);
        let b = rfc0004_ctx_with(&mut vm, &[]);
        // A chega primeiro: bloqueia.
        vm.step_instruction(a, &instr_barrier(7, 2, 0, 1)).unwrap();
        assert!(matches!(vm.scheduler.get(a).unwrap().state, ContextState::Blocked));
        assert!(vm.barriers.contains_key(&7));
        // B chega: RELEASE; A reacorda; entrada some.
        vm.step_instruction(b, &instr_barrier(7, 2, 0, 1)).unwrap();
        assert!(!vm.barriers.contains_key(&7));
        assert!(matches!(vm.scheduler.get(a).unwrap().state, ContextState::Ready));
        // EXPECT=1: libera na hora, nunca bloqueia (estado segue Ready —
        // step direto não passa por pick_next, que poria Running).
        vm.step_instruction(a, &instr_barrier(8, 1, 0, 1)).unwrap();
        assert!(matches!(vm.scheduler.get(a).unwrap().state, ContextState::Ready));
        assert!(!vm.barriers.contains_key(&8));
        // Geração divergente e EXPECT=0: erro sem tocar em nada.
        vm.step_instruction(a, &instr_barrier(9, 2, 0, 1)).unwrap();
        assert!(vm.step_instruction(b, &instr_barrier(9, 3, 0, 1)).is_err());
        assert!(vm.step_instruction(b, &instr_barrier(9, 2, 0, 2)).is_err());
        assert!(vm.step_instruction(a, &instr_barrier(10, 0, 0, 1)).is_err());
        // 4 execuções contabilizadas (A7, B7-release, A8-imediato, A9-wait);
        // os 3 erros não contabilizam.
        assert_eq!(vm.stats.barrier_execs, 4);
    }

    #[tokio::test]
    async fn test_barrier_timeout_nack() {
        use crate::opcodes::instr_barrier;
        use crate::context::ContextState;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let a = rfc0004_ctx_with(&mut vm, &[]);
        let b = rfc0004_ctx_with(&mut vm, &[]);
        // A espera em id=11 (sem timeout). Entrada expirada forjada p/ id=12
        // com A como waiter: B chega atrasado => NACK (morre), A liberado.
        vm.step_instruction(a, &instr_barrier(11, 2, 0, 1)).unwrap();
        vm.barriers.insert(12, crate::vm::BarrierState { expected: 2, arrived: 1, deadline_ns: 0, epoch: 1 });
        vm.barrier_waiters.insert(12, vec![a]);
        assert!(vm.step_instruction(b, &instr_barrier(12, 2, 0, 1)).is_err());
        assert!(!vm.barriers.contains_key(&12));
        assert!(matches!(vm.scheduler.get(a).unwrap().state, ContextState::Ready));
        assert!(vm.scheduler.get(b).is_none() || !matches!(vm.scheduler.get(b).unwrap().state, ContextState::Running));
    }

    #[tokio::test]
    async fn test_priority_set_single_presence() {
        // Regressão RFC-0018: PRIORITY_SET enfileirava aqui + o loop de run
        // re-enfileirava de novo (presença dupla => fatia dobrada).
        // Simula o ciclo do run: pick (pop) -> step -> avanço+Ready -> yield.
        use crate::opcodes::{instr_priority_set};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[(0, 2u128)]);
        let picked = vm.scheduler.pick_next().unwrap();
        assert_eq!(picked, cid);
        vm.step_instruction(cid, &instr_priority_set(0)).unwrap();
        assert!(matches!(vm.scheduler.get(cid).unwrap().priority, crate::context::Priority::Red));
        // Cauda do loop: advance + yield (cópia fiel de Vm::run).
        if let Some(ctx) = vm.scheduler.get_mut(cid) {
            if ctx.state == crate::context::ContextState::Running {
                ctx.advance_pc();
                ctx.state = crate::context::ContextState::Ready;
            }
        }
        vm.scheduler.yield_current();
        let (r, bl, g) = vm.scheduler.queue_lengths();
        assert_eq!((r, bl, g), (1, 0, 0), "exatamente uma presença (RED)");
        // E o programa completo termina com contadores exatos.
        let prog = crate::opcodes::assemble("LOADI r0, 2\nPRIORITY_SET r0\nNOP\nHALT").unwrap();
        let mut vm2 = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        vm2.load_program(prog);
        let stats = vm2.run().unwrap();
        assert_eq!((stats.priority_set_execs, stats.steps), (1, 4));
    }

    // ---- RFC-0019: FILL + SLICE ---------------------------------------

    #[tokio::test]
    async fn test_tensor_fill_zeros_vs_ramp() {
        use crate::opcodes::instr_tensor;
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let cid = rfc0004_ctx_with(&mut vm, &[]);
        // Default: ramp (legado intacto).
        vm.step_instruction(cid, &instr_tensor(0, 0xFF, 0xFF, 2, 2, 0)).unwrap();
        let a = vm.scheduler.get(cid).unwrap().reg(0).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(a, 4).unwrap(), vec![0.5, 1.0, 1.5, 2.0]);
        // FILL=0: zeros exatos.
        let mut z = instr_tensor(1, 0xFF, 0xFF, 2, 2, 0);
        z.set_tensor_fill(0.0);
        vm.step_instruction(cid, &z).unwrap();
        let b = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(b, 4).unwrap(), vec![0.0; 4]);
        // FILL=2.5 via assembler.
        let prog = crate::opcodes::assemble("TENSOR r2 2 2 f32 FILL=2.5").unwrap();
        vm.step_instruction(cid, &prog[0]).unwrap();
        let c = vm.scheduler.get(cid).unwrap().reg(2).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(c, 4).unwrap(), vec![2.5; 4]);
    }

    #[tokio::test]
    async fn test_slice_exact_and_errors() {
        use crate::opcodes::{instr_distance, instr_slice, DIST_METRIC_DOT};
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
        // Simula packed do DISTANCE: [d0,d1,d2,i0,i1,i2] => fatia [3..6).
        let p = rfc0004_f32(&mut vm, &[6], &[9.0, 8.0, 7.0, 0.0, 2.0, 1.0]);
        let cid = rfc0004_ctx_with(&mut vm, &[(0, p)]);
        vm.step_instruction(cid, &instr_slice(1, 0, 3, 3)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
        let meta = vm.memory.get_tensor_meta(out).cloned().unwrap();
        assert_eq!(meta.shape, vec![1, 3]);
        assert_eq!(vm.memory.read_f32_tensor(out, 3).unwrap(), vec![0.0, 2.0, 1.0]);
        // Metade das dists.
        vm.step_instruction(cid, &instr_slice(1, 0, 0, 3)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 3).unwrap(), vec![9.0, 8.0, 7.0]);
        // Erros: OOB, além do fim, LEN 0.
        assert!(vm.step_instruction(cid, &instr_slice(1, 0, 4, 3)).is_err());
        assert!(vm.step_instruction(cid, &instr_slice(1, 0, 0, 7)).is_err());
        assert!(vm.step_instruction(cid, &instr_slice(1, 0, 2, 0)).is_err());
        // Integração: DISTANCE TOPK=2 real => SLICE Yeni índices => GATHER.
        let q = rfc0004_f32(&mut vm, &[1, 2], &[1.0, 0.0]);
        let bank = rfc0004_f32(&mut vm, &[3, 2], &[1.0, 0.0, 0.0, 1.0, -1.0, 0.0]);
        let tab = rfc0004_f32(&mut vm, &[3, 1], &[10.0, 20.0, 30.0]);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, q).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, bank).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(2, tab).unwrap();
        vm.step_instruction(cid, &instr_distance(3, 0, 1, DIST_METRIC_DOT, 2)).unwrap();
        // DOT: [-1, 0, 1] => top-2: [-1(idx0), 0(idx1)] packed [1,4].
        vm.step_instruction(cid, &instr_slice(4, 3, 2, 2)).unwrap();
        vm.step_instruction(cid, &crate::opcodes::instr_gather(5, 2, 4, 0xFF, 0, crate::opcodes::GATHER_MODE_GATHER)).unwrap();
        let out = vm.scheduler.get(cid).unwrap().reg(5).unwrap();
        assert_eq!(vm.memory.read_f32_tensor(out, 2).unwrap(), vec![10.0, 20.0]);
    }
}
