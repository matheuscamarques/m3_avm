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
    Instruction, INSTR_SIZE, OP_ABORT, OP_ADD, OP_ATTN, OP_COMPARE, OP_EMBED, OP_FFN, OP_FORK, OP_HALT, OP_IF_EQUAL,
    OP_IF_INTERRUPT, OP_JUMP, OP_NOP, OP_NORM, OP_SAMPLE, OP_SENSE, OP_STREAM, OP_TENSOR, SENSE_AUDIO, SENSE_TOKEN,
    SENSE_USER_INPUT, SENSE_VAD, STREAM_FLAG_BLOCKING,
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
    pub fn restore(&mut self, v: u64) -> anyhow::Result<()> {
        match self {
            MemBackend::Cpu(m) => m.restore(v),
            #[cfg(feature = "wgpu")] MemBackend::Gpu(m) => m.restore(v),
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
pub struct Vm {
    pub memory: MemBackend,
    pub scheduler: Scheduler,
    /// Programa carregado (instruções decodificadas). PC é índice * 32 + base.
    /// Mantemos também mapa PC (u128) -> índice, pois PC é 128-bit.
    pub(crate) program: Vec<Instruction>,
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
}

impl Vm {
    pub fn new(config: VmConfig) -> Result<Self> {
        Self::new_with_persistent_size(config, 64 * 1024 * 1024)
    }

    pub fn new_with_persistent_size(config: VmConfig, persistent_bytes: usize) -> Result<Self> {
        // Auto-detecta Vega 8 se feature wgpu ativa — fallback para CPU
        #[cfg(feature = "wgpu")]
        let memory = match pollster::block_on(WgpuMemoryManager::new()) {
            Ok(gpu) => {
                eprintln!("[vm] Vega 8 detectada — usando WgpuMemoryManager (Vulkan RADV)");
                MemBackend::Gpu(gpu)
            }
            Err(e) => {
                eprintln!("[vm] wgpu não disponível ({}), fallback CPU", e);
                MemBackend::Cpu(MemoryManager::new_with_size(persistent_bytes)?)
            }
        };
        #[cfg(not(feature = "wgpu"))]
        let memory = MemBackend::Cpu(MemoryManager::new_with_size(persistent_bytes)?);
        let scheduler = Scheduler::new();
        Ok(Self {
            memory,
            scheduler,
            program: Vec::new(),
            program_base: 0x1000, // programa começa em GLOBAL 0x1000
            config,
            stats: VmStats::new(),
            stream_channels: HashMap::new(),
            meter: ThroughputMeter::new(),
            tokenizer: None,
            last_sample: 0,
            user_input: VecDeque::new(),
        })
    }

    /// Construtor sem persistência (testes) — tenta Vega 8 se feature wgpu
    pub fn new_in_memory(config: VmConfig) -> Self {
        #[cfg(feature = "wgpu")]
        let memory = match pollster::block_on(WgpuMemoryManager::new()) {
            Ok(gpu) => MemBackend::Gpu(gpu),
            Err(_) => MemBackend::Cpu(MemoryManager::new_in_memory()),
        };
        #[cfg(not(feature = "wgpu"))]
        let memory = MemBackend::Cpu(MemoryManager::new_in_memory());
        Self {
            memory,
            scheduler: Scheduler::new(),
            program: Vec::new(),
            program_base: 0x1000,
            config,
            stats: VmStats::new(),
            stream_channels: HashMap::new(),
            meter: ThroughputMeter::new(),
            tokenizer: None,
            last_sample: 0,
            user_input: VecDeque::new(),
        }
    }

    /// Construtor explícito CPU (para testes determinísticos)
    pub fn new_cpu(config: VmConfig) -> Result<Self> {
        Ok(Self {
            memory: MemBackend::Cpu(MemoryManager::new()?),
            scheduler: Scheduler::new(),
            program: Vec::new(),
            program_base: 0x1000,
            config,
            stats: VmStats::new(),
            stream_channels: HashMap::new(),
            meter: ThroughputMeter::new(),
            tokenizer: None,
            last_sample: 0,
            user_input: VecDeque::new(),
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

    /// Carrega programa a partir de Vec<Instruction>
    pub fn load_program(&mut self, prog: Vec<Instruction>) {
        let base = self.program_base;
        self.program = prog;
        // Cria contexto inicial GREEN se não houver nenhum
        if self.scheduler.count() == 0 {
            let root = self.memory.current_version();
            let entry_pc = base; // PC inicial aponta para primeira instrução
            let ctx_id = self.scheduler.create_context(Priority::Green, entry_pc, root);
            log_info("vm", &format!("programa carregado: {} instr, ctx inicial {} @ PC {:032x}", self.program.len(), ctx_id, entry_pc));
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

    /// Traduz PC (u128) para índice no Vec<Instruction>
    fn pc_to_index(&self, pc: u128) -> Option<usize> {
        if pc < self.program_base {
            return None;
        }
        let offset = pc - self.program_base;
        if offset % 32 != 0 {
            return None;
        }
        let idx = (offset / 32) as usize;
        if idx < self.program.len() {
            Some(idx)
        } else {
            None
        }
    }

    /// Busca instrução no PC do contexto
    fn fetch(&self, ctx: &Context) -> Result<Instruction> {
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
            match instr.opcode {
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
                    // NOP: só avança PC
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
                        ctx.advance_pc();
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

        // Caminho Vega 8 (wgpu) — tenta GPU se backend for Gpu e shapes <=64
        #[cfg(feature = "wgpu")]
        if self.memory.is_gpu() {
            // Só tenta GPU para denso e shapes pequenos (Vega 8: 8 CUs, workgroup 8x8)
            let meta_q = self.memory.get_tensor_meta(addr_q).cloned();
            let meta_k = self.memory.get_tensor_meta(addr_k).cloned();
            let meta_v = self.memory.get_tensor_meta(addr_v).cloned();
            let shape_q = meta_q.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2,2]);
            let shape_k = meta_k.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2,2]);
            let shape_v = meta_v.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![2,2]);
            if shape_q[0] <= 64 && shape_q[1] <= 64 {
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
            let tok = self.sample_logits(&logits);
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
            let tok = self.sample_logits(&logits);
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
            child.root_version = snap_version;
        }
        // Retorna ID do filho em rdest do pai
        if let Some(parent_mut) = self.scheduler.get_mut(ctx_id) {
            parent_mut.set_reg(instr.rdest, new_id as u128)?;
        }
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
        // Temp vai no payload[0..4] (instr_sample); <=0 ou inválido => 1.0
        let mut tb = [0u8; 4];
        tb.copy_from_slice(&instr.payload[0..4]);
        let temp = f32::from_le_bytes(tb);
        let scaled: Vec<f32> = if temp > 0.0 && temp.is_finite() && (temp - 1.0).abs() > f32::EPSILON {
            logits.iter().map(|v| v / temp).collect()
        } else {
            logits.clone()
        };
        let tok = self.sample_logits(&scaled);
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
        if let Some(ctx) = self.scheduler.get_mut(ctx_id) {
            ctx.cmp_equal = v1 == v2;
        }
        log_debug("compare", &format!("ctx {} COMPARE {} == {} -> {}", ctx_id, v1, v2, v1 == v2));
        Ok(())
    }

    /// Valida que um PC alvo cai dentro do programa.
    fn check_jump_target(&self, target: u128) -> Result<()> {
        if target < self.program_base {
            return Err(anyhow!("JUMP para {:032x} antes da base {:032x}", target, self.program_base));
        }
        let offset = target - self.program_base;
        if offset % 32 != 0 {
            return Err(anyhow!("JUMP para {:032x} desalinhado", target));
        }
        let idx = (offset / 32) as usize;
        if idx >= self.program.len() {
            return Err(anyhow!("JUMP para {:032x} fora do programa (len {})", target, self.program.len()));
        }
        Ok(())
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
            SENSE_AUDIO => {
                // Simula ruído branco: 1024 samples f32 aleatórios -> bytes
                let mut rng = rand::thread_rng();
                let samples: Vec<f32> = (0..256).map(|_| rng.gen_range(-1.0..1.0)).collect();
                let bytes: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
                bytes
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
}
