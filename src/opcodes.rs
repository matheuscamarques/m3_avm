//! opcodes.rs — Decodificador e dispatcher da ISA M³-AVM
//!
//! Formato fixo 32 bytes (simplifica pipeline / prefetch):
//!   offset 0: opcode (1 byte)
//!   offset 1: flags  (1 byte)
//!   offset 2: rdest  (1 byte) — 0..15, 0xFF = não usado
//!   offset 3: rsrc1  (1 byte)
//!   offset 4: rsrc2  (1 byte)
//!   offset 5: rsrc3  (1 byte)
//!   offset 6..31: payload (26 bytes) — imediatos 128-bit, shape, etc.
//!
//! ISA — 8 opcodes (tese: preempção + computação inseparáveis):
//!   0x01 TENSOR  — aloca tensor em GLOBAL
//!   0x02 ATTN    — FlashAttention simplificada via ndarray
//!   0x03 STREAM  — conecta fonte→sumidouro com backpressure
//!   0x04 FORK    — clona contexto (CoW)
//!   0x05 ABORT   — aborta contexto e restaura snapshot
//!   0x06 SENSE   — lê periférico → TEMPORAL
//!   0x07 NORM    — RMSNorm / LayerNorm
//!   0x08 FFN     — Feed-Forward com SwiGLU

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use thiserror::Error;
use crate::memory::GLOBAL_HEAP_START;

pub const INSTR_SIZE: usize = 32;

// RFC-0002 (ESPEC-V2 §4): largura 64B das formas estendidas e teto do
// payload de escape (RFC-0001 §9.1: ext_len <= 1 MiB).
pub const INSTR_SIZE_64: usize = 64;
pub const EXT_MAX_LEN: usize = 1024 * 1024;

// Opcodes
pub const OP_TENSOR: u8 = 0x01;
pub const OP_ATTN: u8 = 0x02;
pub const OP_STREAM: u8 = 0x03;
pub const OP_FORK: u8 = 0x04;
pub const OP_ABORT: u8 = 0x05;
pub const OP_SENSE: u8 = 0x06;
pub const OP_NORM: u8 = 0x07;
pub const OP_FFN: u8 = 0x08;
// Novos opcodes para LLM Thinking & Controle de Fluxo
pub const OP_EMBED: u8 = 0x09;
pub const OP_ADD: u8 = 0x0A;
pub const OP_SAMPLE: u8 = 0x0B;
pub const OP_COMPARE: u8 = 0x0C;
pub const OP_IF_EQUAL: u8 = 0x0D;
pub const OP_JUMP: u8 = 0x0E;
pub const OP_IF_INTERRUPT: u8 = 0x0F;
pub const OP_MATVEC: u8 = 0x10;
pub const OP_MUL: u8 = 0x11;
pub const OP_SILU: u8 = 0x12;
// Mamba / SSM + Audio Codec + controle híbrido (ISA 0x13..0x19)
pub const OP_SSM_SCAN: u8 = 0x13; // h*=exp(dt*A)+x*B*dt; y=h·C+D*x (ssm.rs)
pub const OP_SSM_RESET: u8 = 0x14; // zera/restaura h_t O(1)
pub const OP_CODEC_ENC: u8 = 0x15; // PCM 1920xf32 -> 16xu16 (mimi.rs)
pub const OP_CODEC_DEC: u8 = 0x16; // 16xu16 -> PCM 1920xf32
pub const OP_AUDIO_ALIGN: u8 = 0x17; // t_user/t_ai/delta/frame_id p/ barge-in
pub const OP_CTX_SWITCH: u8 = 0x18; // troca pipeline + fence + prioridade
pub const OP_ROPE: u8 = 0x19; // Rotary Position Embedding nativo
// Cluster F1 local (RFC-0018): sem transporte; node!=0 veta explícito.
pub const OP_REMOTE_SPAWN: u8 = 0x1A; // cria contexto (local: entry_pc)
pub const OP_SIGNAL: u8 = 0x1B; // controle: ABORT/FORK_REQ/HALT/PING
pub const OP_SEND_TENSOR: u8 = 0x1C; // COPY (+MOVE com invalidação real)
pub const OP_BARRIER: u8 = 0x1D; // barreira one-shot com timeout
// Kinds — SIGNAL (rsrc1 como valor imediato 0..3).
pub const SIGNAL_KIND_ABORT: u8 = 0;
pub const SIGNAL_KIND_FORK_REQ: u8 = 1;
pub const SIGNAL_KIND_HALT: u8 = 2;
pub const SIGNAL_KIND_PING: u8 = 3;
// Modos — SEND_TENSOR (payload[16]): 0=COPY, 1=MOVE.
pub const SEND_MODE_COPY: u8 = 0;
pub const SEND_MODE_MOVE: u8 = 1;
// Universal onda 1/2 — G1/H1 (RFC-0004): GATHER + DISTANCE + RANK1_UPDATE.
// (0x1E CONV, 0x20-0x22, 0x25 seguem reservados; ver ESPEC-V2 §3.3.)
pub const OP_GATHER: u8 = 0x1F; // gather/scatter por índice (MoE, GNN, e-bag)
pub const OP_DISTANCE: u8 = 0x23; // distância em lote + top-k fundido (RAG)
pub const OP_RANK1_UPDATE: u8 = 0x24; // H = α·H + β·v⊗k (DeltaNet/Titans)
// Determinismo — bloco RNG + hashing (RFC-0005, ESPEC-V2 §3.10 parcial).
pub const OP_RNG_SEED: u8 = 0x60; // re-sementeia rng_state do contexto
pub const OP_RNG_NEXT: u8 = 0x61; // rdest <- próximo u64
pub const OP_RNG_NORMAL: u8 = 0x62; // rdest <- normal(mean,std)
pub const OP_RNG_UNIFORM: u8 = 0x63; // rdest <- f32 em [a,b)
pub const OP_HASH: u8 = 0x64; // rdest <- FNV-1a/64 do tensor
pub const OP_CHECKSUM: u8 = 0x65; // rdest <- CRC32-IEEE do tensor
pub const OP_HMAC: u8 = 0x66; // rdest <- HMAC-SHA256 truncado em 64 bits
// Telemetria + scheduler (RFC-0006, ESPEC-V2 §3.10 parcial).
pub const OP_CYCLES_COUNT: u8 = 0x6A; // rdest <- now_ns (métrica)
pub const OP_TRACE_EVENT: u8 = 0x6B; // anel de trace (event,data)
pub const OP_SANITY_CHECK: u8 = 0x6C; // absorve NaN/Inf (CoW + contagem)
pub const OP_PREEMPT_CHECK: u8 = 0x6D; // rdest <- interrupt_flag (sem consumir)
pub const OP_ASSERT: u8 = 0x6E; // trap se reg == 0
pub const OP_DUMP: u8 = 0x6F; // log de debug do contexto
pub const OP_YIELD: u8 = 0x70; // cede + maybe_preempt
pub const OP_SET_DEADLINE: u8 = 0x71; // deadline EDF absoluto (ns)
pub const OP_GET_DEADLINE: u8 = 0x72; // rdest <- deadline
pub const OP_PRIORITY_SET: u8 = 0x73; // 0/1/2 + re-enfileira
pub const OP_PRIORITY_GET: u8 = 0x74; // rdest <- prioridade
pub const OP_LOCK: u8 = 0x75; // try-lock cross-context
pub const OP_UNLOCK: u8 = 0x76; // libera lock próprio
pub const OP_FENCE: u8 = 0x77; // barreira (marcador + compiler fence)
// Programabilidade (RFC-0007): imediatos + predicados de comparação.
pub const OP_LOADI: u8 = 0x78; // rdest <- imm u128 (payload[0..16])
pub const OP_MOV: u8 = 0x79; // rdest <- reg[rsrc1]
// W4 pull-forward (RFC-0010): KV hygiene (0x38; resto da onda segue RSVD).
pub const OP_KV_TRUNCATE: u8 = 0x38; // trunca todas as camadas KV p/ len
// Onda 1 universal, último (RFC-0013): passo de difusão fundido.
pub const OP_DENOISE_STEP: u8 = 0x21; // x_t -> x_{t-1} (DDPM/DDIM)
// Onda 2 / W4 (RFC-0014): passo ODE fundido (Liquid/NCPS).
pub const OP_ODE_STEP: u8 = 0x25; // x(t+dt) via Euler/RK2/RK4
// W4 (RFC-0015): LIF integrate-and-fire (SNN, 3º motor stateful).
pub const OP_SPIKE_STEP: u8 = 0x20; // spikes binários + V(t) CoW
// Onda 1, último buraco (RFC-0017): convolução deslizante 1D/2D.
pub const OP_CONV: u8 = 0x1E; // N-dim por shape; aqui 1D/2D + groups
// Fused act — CONV (payload[7]): 0=none, 1=silu, 2=relu.
pub const CONV_ACT_NONE: u8 = 0;
pub const CONV_ACT_SILU: u8 = 1;
pub const CONV_ACT_RELU: u8 = 2;
// Métodos — ODE_STEP (payload[4]): 0=Euler, 1=RK2, 2=RK4.
pub const ODE_METHOD_EULER: u8 = 0;
pub const ODE_METHOD_RK2: u8 = 1;
pub const ODE_METHOD_RK4: u8 = 2;
// W4 pull-forward (RFC-0012): ensemble de árvores vetorizado.
pub const OP_FOREST: u8 = 0x22; // XGBoost/RF: walk sobre tabela plana
// RFC-0019: FILL escalar em TENSOR + SLICE flat (0x2E).
pub const TENSOR_FLAG_FILL: u8 = 0b01; // payload[22..26] = fill f32 LE
pub const OP_SLICE: u8 = 0x2E; // fatia flat [start,start+len) => [1,len]
// RFC-0023: núcleo de memória (0x26/0x27/0x2A/0x2B; resto de 0x26-0x2F na RFC-0024).
pub const OP_ARENA_ALLOC: u8 = 0x26; // bump-alloc size+align -> offset
pub const OP_ARENA_RESET: u8 = 0x27; // cursor=0 O(1)
pub const OP_MEMCPY: u8 = 0x2A; // cópia byte-exata tensor->tensor
pub const OP_MEMSET: u8 = 0x2B; // fill de padrão byte
// Direções — MEMCPY (payload[24]): só HOST executa; GPU/NIC vetam explícito.
pub const MEMCPY_DIR_HOST: u8 = 0;
pub const MEMCPY_DIR_GPU: u8 = 1;
pub const MEMCPY_DIR_NIC: u8 = 2;
// RFC-0026: ALU de control-plane (0x7A-0x7C; 0x7D-0x7F seguem RSVD).
pub const OP_ADD_IMM: u8 = 0x7A; // rD = rS wrapping_add imm
pub const OP_SUB_IMM: u8 = 0x7B; // rD = rS wrapping_sub imm (único SUB do ISA)
pub const OP_STEPS: u8 = 0x7C; // rD <- instruções retiradas (determinístico)
// RFC-0038: retrieval (0x50-0x53/0x56-0x57, v1.15; turno 1: ADD/DEL).
pub const OP_RAG_INDEX_ADD: u8 = 0x50; // anexa vetor (rDb=0 cria; rD <- id/contagem)
pub const OP_RAG_INDEX_DEL: u8 = 0x51; // remove por id (rD <- restante)
// RFC-0025: bloco de conversão (0x67/0x68/0x69, v1.7).
pub const OP_CAST: u8 = 0x67; // conversão de valor FP32<->F16/BF16/I8/U8
pub const OP_QUANTIZE: u8 = 0x68; // F32 -> blocos Q4_0/Q8_0
pub const OP_DEQUANT: u8 = 0x69; // blocos -> F32 (dispatcher existente)
// Destinos — CAST (payload[0]): códigos locais do op (não confundir com
// discriminantes DType; o mapeamento é explícito no exec).
pub const CAST_DST_F32: u8 = 0;
pub const CAST_DST_F16: u8 = 1;
pub const CAST_DST_BF16: u8 = 2;
pub const CAST_DST_I8: u8 = 3;
pub const CAST_DST_U8: u8 = 4;
// Tipos — QUANTIZE (payload[0]): discriminante DType (4=Q4_0, 8=Q8_0).
// Só o par simétrico com encoder+decoder parseia (Q4_K/Q6_K vetam no
// assembler — sem soletrar o inexecutável).
pub const QUANTIZE_Q4_0: u8 = 4;
pub const QUANTIZE_Q8_0: u8 = 8;
// RFC-0027: forma (0x30-0x37, v1.9; resto de 0x30-0x43 nas partes 2-3).
// RFC-0028: ativações (0x3C-0x43, v1.10; 0x39-0x3B na parte 3).
// RFC-0032: passo depformer (0x44, v1.13; 17 streams na RFC-0033).
pub const OP_DEPFORMER: u8 = 0x44; // proj+KV deslizante+attn+codes
// RFC-0034: sub-rotinas (0x7D/0x7E, v1.14; 0x7F segue RSVD).
pub const OP_CALL: u8 = 0x7D; // push retorno + jump p/ label
pub const OP_RET: u8 = 0x7E; // pop retorno (pilha vazia veta)
// RFC-0029: KV/attention (0x39/0x3A/0x3B, v1.11; fecha 0x30-0x43).
// RFC-0031: DSP de áudio (0x45-0x49, v1.12; 0x44 na parte 2).
pub const OP_STREAM_MERGE: u8 = 0x45; // mix com ganho explícito
pub const OP_VAD_DETECT: u8 = 0x46; // score de voz (energy/zcr)
pub const OP_AUDIO_RESAMPLE: u8 = 0x47; // interpolação linear
pub const OP_AUDIO_FILTER: u8 = 0x48; // FIR same-size centrado
pub const OP_AUDIO_WINDOW: u8 = 0x49; // Hann/Hamming periódica
// Modos — VAD (payload[0]): 0=ENERGY, 1=ZCR, 2=ML (veta no exec).
pub const VAD_MODE_ENERGY: u8 = 0;
pub const VAD_MODE_ZCR: u8 = 1;
pub const VAD_MODE_ML: u8 = 2;
// Modo — FILTER (payload[0]): 0=FIR (IIR veta no exec).
pub const FILTER_MODE_FIR: u8 = 0;
pub const FILTER_MODE_IIR: u8 = 1;
// Tipo — WINDOW (payload[0]): 0=HANN, 1=HAMMING.
pub const WINDOW_HANN: u8 = 0;
pub const WINDOW_HAMMING: u8 = 1;
pub const OP_KV_COMPRESS: u8 = 0x39; // evicção sink+janela no KV_CACHE
pub const OP_FLASH_ATTN: u8 = 0x3A; // atenção em blocos, online-softmax
pub const OP_ATTN_SPARSE: u8 = 0x3B; // top-k fundido por DOT + atenção
// Modo — KV_COMPRESS (payload[4]): só SINK_WINDOW.
pub const KVCOMP_MODE_SINK_WINDOW: u8 = 0;
// Métrica — ATTN_SPARSE (payload[0]): só DOT (scores são dots).
pub const SPARSE_METRIC_DOT: u8 = 0;
pub const OP_SOFTMAX: u8 = 0x3C; // softmax estável por eixo + temperatura
pub const OP_GELU: u8 = 0x3D; // gelu exato (erf A&S)
pub const OP_SIGMOID: u8 = 0x3E; // 1/(1+e^-x)
pub const OP_TANH: u8 = 0x3F; // tangente hiperbólica
pub const OP_RELU: u8 = 0x40; // max(x,+0)
pub const OP_EXP: u8 = 0x41; // exponencial
pub const OP_LOG: u8 = 0x42; // logaritmo natural (IEEE)
pub const OP_CLIP: u8 = 0x43; // clamp [min,max]
pub const OP_SORT: u8 = 0x30; // ordenação por eixo (NaN=+inf, empate=menor idx)
pub const OP_TOPK: u8 = 0x31; // top-k fundido [vals|idx] (pack DISTANCE)
pub const OP_ARGMAX: u8 = 0x32; // índices como f32 (ignora NaN, cf. maxNum)
pub const OP_REDUCE: u8 = 0x33; // sum/mean/max/min/prod por eixo ou total
pub const OP_BROADCAST: u8 = 0x34; // expansão p/ shape alvo (cópia)
pub const OP_PAD: u8 = 0x35; // crescimento single-axis (compõe p/ N-D)
pub const OP_TILE: u8 = 0x36; // repetição single-axis
pub const OP_TRANSPOSE: u8 = 0x37; // permuta N-D genérica (cópia)
// Ordem — SORT (payload[1]): 0=ASC, 1=DESC.
pub const SORT_ASC: u8 = 0;
pub const SORT_DESC: u8 = 1;
// Operações — REDUCE (payload[0]).
pub const REDUCE_SUM: u8 = 0;
pub const REDUCE_MEAN: u8 = 1;
pub const REDUCE_MAX: u8 = 2;
pub const REDUCE_MIN: u8 = 3;
pub const REDUCE_PROD: u8 = 4;
// RFC-0024: views & versions (0x28/0x29/0x2C/0x2D/0x2F; fecha 0x26-0x2F, v1.6).
pub const OP_SNAPSHOT: u8 = 0x28; // snapshot nomeado -> version handle
pub const OP_RESTORE: u8 = 0x29; // rewind p/ version (memória + mapas)
pub const OP_PREFETCH: u8 = 0x2C; // hint de cache (só leitura)
pub const OP_RESHAPE: u8 = 0x2D; // cópia com novo shape
pub const OP_CONCAT: u8 = 0x2F; // montagem ao longo de eixo
// Máscara — SNAPSHOT (payload[0]): só o conjunto cheio executa.
pub const SNAP_MASK_GLOBAL: u8 = 0b001;
pub const SNAP_MASK_KV: u8 = 0b010;
pub const SNAP_MASK_ENGINE: u8 = 0b100;
pub const SNAP_MASK_ALL: u8 = 0b111;
// Modos — FOREST (payload[4]): 0 = valores por árvore, 1 = média.
pub const FOREST_MODE_VOTE: u8 = 0;
pub const FOREST_MODE_MEAN: u8 = 1;
// Predicados — COMPARE (payload[16]; 0 = EQ legado)
pub const CMP_EQ: u8 = 0;
pub const CMP_NE: u8 = 1;
pub const CMP_LT: u8 = 2;
pub const CMP_LE: u8 = 3;
pub const CMP_GT: u8 = 4;
pub const CMP_GE: u8 = 5;
pub const OP_HALT: u8 = 0x00; // não oficial, usado para encerrar programa
pub const OP_NOP: u8 = 0xFF;

// Modos — GATHER (payload[1])
pub const GATHER_MODE_GATHER: u8 = 0;
pub const GATHER_MODE_SCATTER_ADD: u8 = 1;
pub const GATHER_MODE_SCATTER_MAX: u8 = 2;
// Métricas — DISTANCE (payload[0])
pub const DIST_METRIC_EUCLID: u8 = 0;
pub const DIST_METRIC_COSINE: u8 = 1;
pub const DIST_METRIC_MANHATTAN: u8 = 2;
pub const DIST_METRIC_DOT: u8 = 3;
// Modos — RANK1_UPDATE (payload[8])
pub const RANK1_MODE_HEBBIAN: u8 = 0;
pub const RANK1_MODE_DELTA: u8 = 1;
pub const RANK1_MODE_FORGET: u8 = 2;

// Flags — STREAM
pub const STREAM_FLAG_BLOCKING: u8 = 0b01;
pub const STREAM_FLAG_DROP: u8 = 0b10;

// Flags — FORK (prioridade do filho)
pub const FORK_FLAG_GREEN: u8 = 0b00;
pub const FORK_FLAG_BLUE: u8 = 0b01;
pub const FORK_FLAG_RED: u8 = 0b10;
// NOP — FORK com notificação imediata ao scheduler (1 ciclo)
pub const FORK_FLAG_NOTIFY: u8 = 0b100;

// Flags — ATTN
pub const ATTN_FLAG_NOTIFY_EACH_HEAD: u8 = 0b01;

// Flags — SENSE periféricos (em rsrc1)
pub const SENSE_AUDIO: u8 = 0;
pub const SENSE_VAD: u8 = 1;
pub const SENSE_TOKEN: u8 = 3; // PERIPHERAL_TOKEN lido via SENSE Rtoken, TOKEN
pub const SENSE_USER_INPUT: u8 = 5; // PERIPHERAL_USER_INPUT lido via SENSE Rd, USER_INPUT
pub const SENSE_AUDIO_PCM: u8 = 6; // PCM bruto 24kHz/80ms (1920xf32) p/ CODEC_ENC
pub const SENSE_CODEC_FRAME: u8 = 7; // Frame já codificado Mimi (32B, 16xu16)

// Flags — SSM_SCAN
pub const SSM_SCAN_FLAG_CONV: u8 = 0b01; // (reservado) aplica conv causal antes do scan
pub const SSM_SCAN_FLAG_GATE: u8 = 0b10; // (reservado) aplica gating silu(z) após scan
// Flags — CODEC_ENC/DEC
pub const CODEC_FLAG_AS_TENSOR: u8 = 0b01; // Rd/Rpcm como tensor em vez de TEMPORAL bytes
// Flags — ROPE
pub const ROPE_FLAG_INPLACE: u8 = 0b01; // Rd == Rsrc1, atualiza in-place
// Pipeline ids — CTX_SWITCH (em rsrc1)
pub const PIPE_MAMBA: u8 = 0;
pub const PIPE_TRANSFORMER: u8 = 1;
pub const PIPE_AUDIO: u8 = 2;
// STREAM sinks periféricos (valor do registrador Rsink ou imediato Rsink==periph id)
pub const STREAM_PERIPHERAL_SAMPLE: u128 = 2; // sink 2 = amostra logits -> token
pub const STREAM_PERIPHERAL_OUTPUT_DECODED: u128 = 4; // sink 4 = decodifica token e imprime
pub const STREAM_PERIPHERAL_INPUT: u128 = 1; // sink 1 = entrada do usuário / prompt
pub const STREAM_PERIPHERAL_OUTPUT: u128 = 0; // sink 0 = stdout
pub const EOS_TOKEN_DEFAULT: u128 = 2;

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("opcode desconhecido: 0x{0:02x}")]
    UnknownOpcode(u8),
    #[error("registrador inválido r{0}")]
    InvalidRegister(u8),
    #[error("instrução incompleta: esperado 32 bytes, recebido {0}")]
    Incomplete(usize),
    /// RFC-0002: este decoder v1.x só executa 32B. Opcode de zona 64B/escape
    /// em buffer 32B é rejeitado limpo em vez de misdecodificado.
    #[error("largura não suportada neste decoder (opcode 0x{0:02x}, use decoder 64B)")]
    UnsupportedWidth(u8),
    /// RFC-0002: faixa reservada (0xB7-0xFE). Nenhum tamanho assumido.
    #[error("opcode reservado: 0x{0:02x}")]
    ReservedOpcode(u8),
    /// RFC-0002: ext_len de ESCAPE fora dos limites (válido: >=64, múltiplo
    /// de 8, <=1MiB).
    #[error("ext_len inválido em ESCAPE: {0}")]
    InvalidExtLen(u64),
    /// W1-remainder: decoder 64B recebeu menos de 64 bytes.
    #[error("instrução incompleta: esperado 64 bytes, recebido {0}")]
    Incomplete64(usize),
    /// Cabeça ESCAPE 0xB0: exige layout ext_len ([R] §3.2), não decode
    /// 64B plano.
    #[error("cabeça ESCAPE 0x{0:02x}: exige layout ext_len, não decode 64B plano")]
    EscapeHead(u8),
}

/// Largura de instrução por opcode (RFC-0002, ESPEC-V2 §4.3).
/// `0xFF` é a única exceção: sempre 32B (R2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstrWidth {
    Fixed32,
    Fixed64,
    Fixed128,
    Fixed256,
    /// `0xB3 ESCAPE_VAR`: tamanho total em `ext_len` (validar com
    /// `escape_total_len`).
    EscapeVar,
    /// `0xB7-0xFE`: reservado, nenhum tamanho assumido.
    Reserved,
}

/// Regra de dual-mode (RFC-0002). Pura e total sobre os 256 opcodes.
pub fn instr_width(op: u8) -> InstrWidth {
    match op {
        0x00..=0x7F | 0xFF => InstrWidth::Fixed32,
        0x80..=0xAF | 0xB4..=0xB6 => InstrWidth::Fixed64,
        0xB0 => InstrWidth::Fixed64, // cabeça; total via `escape_total_len`
        0xB1 => InstrWidth::Fixed128,
        0xB2 => InstrWidth::Fixed256,
        0xB3 => InstrWidth::EscapeVar,
        _ => InstrWidth::Reserved, // 0xB7-0xFE
    }
}

/// Tamanho fixo em bytes, se houver (`None` para `EscapeVar`/`Reserved`).
pub fn fixed_size(w: InstrWidth) -> Option<usize> {
    match w {
        InstrWidth::Fixed32 => Some(INSTR_SIZE),
        InstrWidth::Fixed64 => Some(INSTR_SIZE_64),
        InstrWidth::Fixed128 => Some(128),
        InstrWidth::Fixed256 => Some(256),
        InstrWidth::EscapeVar | InstrWidth::Reserved => None,
    }
}

/// Valida `ext_len` de ESCAPE (RFC-0001 §3.2 + §9.1, corrigido pela
/// RFC-0002): `>= 64`, múltiplo de 8, `<= 1 MiB`.
pub fn escape_total_len(ext_len: u64) -> Result<usize, DecodeError> {
    if ext_len < 64 || ext_len % 8 != 0 || ext_len > EXT_MAX_LEN as u64 {
        return Err(DecodeError::InvalidExtLen(ext_len));
    }
    Ok(ext_len as usize)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    pub opcode: u8,
    pub flags: u8,
    pub rdest: u8,
    pub rsrc1: u8,
    pub rsrc2: u8,
    pub rsrc3: u8,
    pub payload: [u8; 26],
}

impl Instruction {
    pub fn new(opcode: u8, flags: u8, rdest: u8, rsrc1: u8, rsrc2: u8, rsrc3: u8) -> Self {
        Self {
            opcode,
            flags,
            rdest,
            rsrc1,
            rsrc2,
            rsrc3,
            payload: [0u8; 26],
        }
    }

    pub fn with_payload(mut self, payload: [u8; 26]) -> Self {
        self.payload = payload;
        self
    }

    /// Decodifica 32 bytes brutos em Instruction (decoder v1.x: 32B).
    /// Opcodes de zona 64B/escape/reservada são rejeitados limpo
    /// (RFC-0002) em vez de misdecodificados.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < INSTR_SIZE {
            return Err(DecodeError::Incomplete(bytes.len()));
        }
        match instr_width(bytes[0]) {
            InstrWidth::Fixed32 => {}
            InstrWidth::EscapeVar => {
                return Err(DecodeError::UnsupportedWidth(bytes[0]));
            }
            InstrWidth::Reserved => {
                return Err(DecodeError::ReservedOpcode(bytes[0]));
            }
            _ => {
                return Err(DecodeError::UnsupportedWidth(bytes[0]));
            }
        }
        let mut payload = [0u8; 26];
        payload.copy_from_slice(&bytes[6..32]);
        Ok(Self {
            opcode: bytes[0],
            flags: bytes[1],
            rdest: bytes[2],
            rsrc1: bytes[3],
            rsrc2: bytes[4],
            rsrc3: bytes[5],
            payload,
        })
    }

    /// Codifica para 32 bytes.
    pub fn encode(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        out[0] = self.opcode;
        out[1] = self.flags;
        out[2] = self.rdest;
        out[3] = self.rsrc1;
        out[4] = self.rsrc2;
        out[5] = self.rsrc3;
        out[6..32].copy_from_slice(&self.payload);
        out
    }

    /// Valida registradores (0..15 ou 0xFF para unused).
    pub fn validate(&self) -> Result<(), DecodeError> {
        for r in [self.rdest, self.rsrc1, self.rsrc2, self.rsrc3] {
            if r != 0xFF && r >= 16 {
                return Err(DecodeError::InvalidRegister(r));
            }
        }
        Ok(())
    }

    /// Interpreta payload[0..16] como u128 LE (imediato).
    pub fn imm_u128(&self) -> u128 {
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&self.payload[0..16]);
        u128::from_le_bytes(buf)
    }

    /// Escreve u128 LE em payload[0..16].
    pub fn set_imm_u128(&mut self, v: u128) {
        self.payload[0..16].copy_from_slice(&v.to_le_bytes());
    }

    /// Helpers para TENSOR: payload codifica shape como 2× u64 LE + dtype em payload[16].
    /// payload[0..8] = rows (u64 LE), payload[8..16] = cols (u64 LE), payload[16] = dtype.
    pub fn tensor_shape(&self) -> (usize, usize) {
        let mut r = [0u8; 8];
        let mut c = [0u8; 8];
        r.copy_from_slice(&self.payload[0..8]);
        c.copy_from_slice(&self.payload[8..16]);
        let rows = u64::from_le_bytes(r) as usize;
        let cols = u64::from_le_bytes(c) as usize;
        // defaults sensatos se payload zero
        let rows = if rows == 0 { 2 } else { rows };
        let cols = if cols == 0 { 2 } else { cols };
        (rows, cols)
    }

    pub fn tensor_dtype(&self) -> u8 {
        self.payload[16]
    }

    /// NOP+Sparse: payload[17]=1 indica CSR, payload[18..22]=density f32 LE
    pub fn is_sparse(&self) -> bool {
        self.payload[17] == 1
    }

    pub fn sparse_density(&self) -> f32 {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(&self.payload[18..22]);
        let d = f32::from_le_bytes(buf);
        if d <= 0.0 || d > 1.0 { 0.05 } else { d }
    }

    pub fn set_sparse(&mut self, is_sparse: bool, density: f32) {
        self.payload[17] = if is_sparse { 1 } else { 0 };
        self.payload[18..22].copy_from_slice(&density.to_le_bytes());
    }

    pub fn mnemonic(&self) -> &'static str {
        match self.opcode {
            OP_TENSOR => "TENSOR",
            OP_ATTN => "ATTN",
            OP_STREAM => "STREAM",
            OP_FORK => "FORK",
            OP_ABORT => "ABORT",
            OP_SENSE => "SENSE",
            OP_NORM => "NORM",
            OP_FFN => "FFN",
            OP_EMBED => "EMBED",
            OP_ADD => "ADD",
            OP_SAMPLE => "SAMPLE",
            OP_COMPARE => "COMPARE",
            OP_IF_EQUAL => "IF_EQUAL",
            OP_JUMP => "JUMP",
            OP_IF_INTERRUPT => "IF_INTERRUPT",
            OP_MATVEC => "MATVEC",
            OP_MUL => "MUL",
            OP_SILU => "SILU",
            OP_SSM_SCAN => "SSM_SCAN",
            OP_SSM_RESET => "SSM_RESET",
            OP_CODEC_ENC => "CODEC_ENC",
            OP_CODEC_DEC => "CODEC_DEC",
            OP_AUDIO_ALIGN => "AUDIO_ALIGN",
            OP_CTX_SWITCH => "CTX_SWITCH",
            OP_ROPE => "ROPE",
            OP_REMOTE_SPAWN => "REMOTE_SPAWN",
            OP_SIGNAL => "SIGNAL",
            OP_SEND_TENSOR => "SEND_TENSOR",
            OP_BARRIER => "BARRIER",
            OP_GATHER => "GATHER",
            OP_DISTANCE => "DISTANCE",
            OP_RANK1_UPDATE => "RANK1_UPDATE",
            OP_RNG_SEED => "RNG_SEED",
            OP_RNG_NEXT => "RNG_NEXT",
            OP_RNG_UNIFORM => "RNG_UNIFORM",
            OP_RNG_NORMAL => "RNG_NORMAL",
            OP_HASH => "HASH",
            OP_CHECKSUM => "CHECKSUM",
            OP_HMAC => "HMAC",
            OP_CYCLES_COUNT => "CYCLES_COUNT",
            OP_TRACE_EVENT => "TRACE_EVENT",
            OP_SANITY_CHECK => "SANITY_CHECK",
            OP_PREEMPT_CHECK => "PREEMPT_CHECK",
            OP_ASSERT => "ASSERT",
            OP_DUMP => "DUMP",
            OP_YIELD => "YIELD",
            OP_SET_DEADLINE => "SET_DEADLINE",
            OP_GET_DEADLINE => "GET_DEADLINE",
            OP_PRIORITY_SET => "PRIORITY_SET",
            OP_PRIORITY_GET => "PRIORITY_GET",
            OP_LOCK => "LOCK",
            OP_UNLOCK => "UNLOCK",
            OP_FENCE => "FENCE",
            OP_LOADI => "LOADI",
            OP_MOV => "MOV",
            OP_KV_TRUNCATE => "KV_TRUNCATE",
            OP_SLICE => "SLICE",
            OP_ARENA_ALLOC => "ARENA_ALLOC",
            OP_ARENA_RESET => "ARENA_RESET",
            OP_MEMCPY => "MEMCPY",
            OP_MEMSET => "MEMSET",
            OP_SNAPSHOT => "SNAPSHOT",
            OP_RESTORE => "RESTORE",
            OP_PREFETCH => "PREFETCH",
            OP_RESHAPE => "RESHAPE",
            OP_CONCAT => "CONCAT",
            OP_CAST => "CAST",
            OP_QUANTIZE => "QUANTIZE",
            OP_DEQUANT => "DEQUANT",
            OP_ADD_IMM => "ADD_IMM",
            OP_SUB_IMM => "SUB_IMM",
            OP_STEPS => "STEPS",
            OP_SORT => "SORT",
            OP_TOPK => "TOPK",
            OP_ARGMAX => "ARGMAX",
            OP_REDUCE => "REDUCE",
            OP_BROADCAST => "BROADCAST",
            OP_PAD => "PAD",
            OP_TILE => "TILE",
            OP_TRANSPOSE => "TRANSPOSE",
            OP_SOFTMAX => "SOFTMAX",
            OP_GELU => "GELU",
            OP_SIGMOID => "SIGMOID",
            OP_TANH => "TANH",
            OP_RELU => "RELU",
            OP_EXP => "EXP",
            OP_LOG => "LOG",
            OP_CLIP => "CLIP",
            OP_DEPFORMER => "DEPFORMER",
            OP_CALL => "CALL",
            OP_RET => "RET",
            OP_KV_COMPRESS => "KV_COMPRESS",
            OP_FLASH_ATTN => "FLASH_ATTN",
            OP_ATTN_SPARSE => "ATTN_SPARSE",
            OP_STREAM_MERGE => "STREAM_MERGE",
            OP_VAD_DETECT => "VAD_DETECT",
            OP_AUDIO_RESAMPLE => "AUDIO_RESAMPLE",
            OP_AUDIO_FILTER => "AUDIO_FILTER",
            OP_AUDIO_WINDOW => "AUDIO_WINDOW",
            OP_DENOISE_STEP => "DENOISE_STEP",
            OP_ODE_STEP => "ODE_STEP",
            OP_SPIKE_STEP => "SPIKE_STEP",
            OP_CONV => "CONV",
            OP_FOREST => "FOREST",
            OP_RAG_INDEX_ADD => "RAG_INDEX_ADD",
            OP_RAG_INDEX_DEL => "RAG_INDEX_DEL",
            OP_HALT => "HALT",
            OP_NOP => "NOP",
            _ => "UNKNOWN",
        }
    }

    /// Helpers para FFN: recupera registradores de bias no payload
    pub fn ffn_bias_regs(&self) -> (u8, u8) {
        let rb1 = self.payload[0];
        let rb2 = self.payload[1];
        (rb1, rb2)
    }

    pub fn set_ffn_bias_regs(&mut self, rb1: u8, rb2: u8) {
        self.payload[0] = rb1;
        self.payload[1] = rb2;
    }

    /// Helpers SSM_SCAN/RESET: payload[0..2]=d_inner u16 LE, [2..4]=d_state u16 LE,
    /// [4]=layer_id, [5..8]=reserved. Fase 2 usa layer_id p/ Vm.ssm_states.
    pub fn ssm_dims(&self) -> (usize, usize, u8) {
        let di = u16::from_le_bytes([self.payload[0], self.payload[1]]) as usize;
        let ds = u16::from_le_bytes([self.payload[2], self.payload[3]]) as usize;
        (if di == 0 { 1 } else { di }, if ds == 0 { 1 } else { ds }, self.payload[4])
    }

    pub fn set_ssm_dims(&mut self, d_inner: usize, d_state: usize, layer_id: u8) {
        self.payload[0..2].copy_from_slice(&(d_inner.min(65535) as u16).to_le_bytes());
        self.payload[2..4].copy_from_slice(&(d_state.min(65535) as u16).to_le_bytes());
        self.payload[4] = layer_id;
    }

    /// Helpers ROPE: payload[0..4]=pos u32 LE, [4..6]=head_dim u16, [6..8]=n_heads u16,
    /// [8..12]=theta f32 LE.
    pub fn rope_params(&self) -> (u32, usize, usize, f32) {
        let mut b4 = [0u8; 4];
        b4.copy_from_slice(&self.payload[0..4]);
        let pos = u32::from_le_bytes(b4);
        let hd = u16::from_le_bytes([self.payload[4], self.payload[5]]) as usize;
        let nh = u16::from_le_bytes([self.payload[6], self.payload[7]]) as usize;
        let mut bt = [0u8; 4];
        bt.copy_from_slice(&self.payload[8..12]);
        let theta = f32::from_le_bytes(bt);
        (pos, if hd == 0 { 2 } else { hd }, if nh == 0 { 1 } else { nh }, if theta <= 0.0 { 10_000.0 } else { theta })
    }

    pub fn set_rope_params(&mut self, pos: u32, head_dim: usize, n_heads: usize, theta: f32) {
        self.payload[0..4].copy_from_slice(&pos.to_le_bytes());
        self.payload[4..6].copy_from_slice(&(head_dim.min(65535) as u16).to_le_bytes());
        self.payload[6..8].copy_from_slice(&(n_heads.min(65535) as u16).to_le_bytes());
        self.payload[8..12].copy_from_slice(&theta.to_le_bytes());
    }

    /// Helpers AUDIO_ALIGN: payload[0..4]=sample_rate u32, [4..8]=samples_per_frame u32,
    /// [8..12]=frame_hz_x100 u32, [12..16]=delay_ms_x100 u32.
    pub fn audio_align_params(&self) -> (u32, u32, f32, f32) {
        let sr = u32::from_le_bytes([self.payload[0], self.payload[1], self.payload[2], self.payload[3]]);
        let spf = u32::from_le_bytes([self.payload[4], self.payload[5], self.payload[6], self.payload[7]]);
        let fh100 = u32::from_le_bytes([self.payload[8], self.payload[9], self.payload[10], self.payload[11]]);
        let dl100 = u32::from_le_bytes([self.payload[12], self.payload[13], self.payload[14], self.payload[15]]);
        (
            if sr == 0 { 24_000 } else { sr },
            if spf == 0 { 1920 } else { spf },
            if fh100 == 0 { 12.5 } else { fh100 as f32 / 100.0 },
            if dl100 == 0 { 160.0 } else { dl100 as f32 / 100.0 },
        )
    }

    pub fn set_audio_align_params(&mut self, sample_rate: u32, samples_per_frame: u32, frame_hz: f32, delay_ms: f32) {
        self.payload[0..4].copy_from_slice(&sample_rate.to_le_bytes());
        self.payload[4..8].copy_from_slice(&samples_per_frame.to_le_bytes());
        self.payload[8..12].copy_from_slice(&((frame_hz * 100.0).round() as u32).to_le_bytes());
        self.payload[12..16].copy_from_slice(&((delay_ms * 100.0).round() as u32).to_le_bytes());
    }
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:<7} flags=0x{:02x} rdest=r{} rsrc1=r{} rsrc2=r{} rsrc3=r{} payload={:02x?}",
            self.mnemonic(),
            self.flags,
            self.rdest,
            self.rsrc1,
            self.rsrc2,
            self.rsrc3,
            &self.payload[..8]
        )
    }
}

// ---------------------------------------------------------------------------
// Instrução 64B (ESPEC-V2 §4.2, W1-remainder). Representação decodificada
// das formas estendidas 0x80-0xAF/0xB4-0xB6. Nomes/mnemonics chegam com as
// RFCs das Fases 7/9 — até lá, o dispatch rejeita com erro nomeado.
// ---------------------------------------------------------------------------

/// Instrução 64B decodificada (layout §4.2, tudo LE):
/// `[0-1]` opcode/flags (canônico, §5) · `[2-7]` rdest/rsrc1-5 ·
/// `[8-15]` LAMPORT u64 (0=local) · `[16-23]` DEADLINE u64 (MAX=best-effort) ·
/// `[24-31]` PAYLOAD_EXT (HMAC/RDMA/checksum, 0=none) ·
/// `[32-63]` PAYLOAD_CORE (32B, por opcode).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instr64 {
    pub opcode: u8,
    pub flags: u8,
    pub rdest: u8,
    pub rsrc: [u8; 5],
    pub lamport: u64,
    pub deadline: u64,
    pub payload_ext: [u8; 8],
    pub payload_core: [u8; 32],
}

impl Instr64 {
    /// Construtor com extensão neutra (R4: LAMPORT=0, DEADLINE=MAX,
    /// PAYLOAD_EXT=0) e núcleo zerado.
    pub fn new(opcode: u8, rdest: u8, rsrc: [u8; 5]) -> Self {
        Self {
            opcode,
            flags: 0,
            rdest,
            rsrc,
            lamport: 0,
            deadline: u64::MAX,
            payload_ext: [0u8; 8],
            payload_core: [0u8; 32],
        }
    }

    /// Decodifica 64 bytes. Aceita SOMENTE `Fixed64` plano (0x80-0xAF,
    /// 0xB4-0xB6); 0xB0 (cabeça ESCAPE), 0xB1/B2/B3, reservados e opcodes
    /// 32B são rejeitados com erro nomeado — nunca misdecodificados.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < INSTR_SIZE_64 {
            return Err(DecodeError::Incomplete64(bytes.len()));
        }
        let op = bytes[0];
        if op == 0xB0 {
            return Err(DecodeError::EscapeHead(op));
        }
        match instr_width(op) {
            InstrWidth::Fixed64 => {}
            InstrWidth::Reserved => return Err(DecodeError::ReservedOpcode(op)),
            _ => return Err(DecodeError::UnsupportedWidth(op)),
        }
        let mut rsrc = [0u8; 5];
        rsrc.copy_from_slice(&bytes[3..8]);
        let mut lamport_b = [0u8; 8];
        lamport_b.copy_from_slice(&bytes[8..16]);
        let mut deadline_b = [0u8; 8];
        deadline_b.copy_from_slice(&bytes[16..24]);
        let mut payload_ext = [0u8; 8];
        payload_ext.copy_from_slice(&bytes[24..32]);
        let mut payload_core = [0u8; 32];
        payload_core.copy_from_slice(&bytes[32..64]);
        let ins = Self {
            opcode: op,
            flags: bytes[1],
            rdest: bytes[2],
            rsrc,
            lamport: u64::from_le_bytes(lamport_b),
            deadline: u64::from_le_bytes(deadline_b),
            payload_ext,
            payload_core,
        };
        ins.validate()?;
        Ok(ins)
    }

    /// Codifica para 64 bytes (inverso exato de `decode`).
    pub fn encode(&self) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[0] = self.opcode;
        out[1] = self.flags;
        out[2] = self.rdest;
        out[3..8].copy_from_slice(&self.rsrc);
        out[8..16].copy_from_slice(&self.lamport.to_le_bytes());
        out[16..24].copy_from_slice(&self.deadline.to_le_bytes());
        out[24..32].copy_from_slice(&self.payload_ext);
        out[32..64].copy_from_slice(&self.payload_core);
        out
    }

    /// Largura em bytes (sempre 64; existe para o stride do fetch ser
    /// table-driven quando a VM ganhar o programa de largura mista).
    pub fn byte_len(&self) -> usize {
        INSTR_SIZE_64
    }

    /// Extensão neutra (R4: equivalência semântica com a forma básica).
    pub fn has_neutral_ext(&self) -> bool {
        self.lamport == 0 && self.deadline == u64::MAX && self.payload_ext == [0u8; 8]
    }

    /// Nomes chegam com as RFCs das Fases 7/9. Até lá: UNKNOWN honesto.
    pub fn mnemonic(&self) -> &'static str {
        "UNKNOWN"
    }

    /// Valida registradores (0..15 ou 0xFF para unused/descarte).
    pub fn validate(&self) -> Result<(), DecodeError> {
        if self.rdest != 0xFF && self.rdest >= 16 {
            return Err(DecodeError::InvalidRegister(self.rdest));
        }
        for r in self.rsrc {
            if r != 0xFF && r >= 16 {
                return Err(DecodeError::InvalidRegister(r));
            }
        }
        Ok(())
    }
}

impl std::fmt::Display for Instr64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "X_{:02x} flags=0x{:02x} rdest=r{} rsrc={:?} lamport={} deadline={} core={:02x?}",
            self.opcode,
            self.flags,
            self.rdest,
            self.rsrc,
            self.lamport,
            self.deadline,
            &self.payload_core[..8]
        )
    }
}

// ---------------------------------------------------------------------------
// Programa de largura mista (W1-remainder). O `Vm.program` guarda frames
// já decodificados; o stride vem de `byte_len()`, nunca de constante.
// Programas só-32B (todo o corpus atual) têm offsets uniformes e
// comportamento bit-idêntico ao anterior.
// ---------------------------------------------------------------------------

/// Uma instrução do programa: 32B (`Instruction`) ou 64B (`Instr64`).
/// `Copy` de propósito: o fetch retorna por valor, sem empréstimo do `Vm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProgramInstr {
    W32(Instruction),
    W64(Instr64),
}

impl ProgramInstr {
    /// Opcode (byte 0 em ambas as larguras).
    pub fn opcode(&self) -> u8 {
        match *self {
            ProgramInstr::W32(i) => i.opcode,
            ProgramInstr::W64(g) => g.opcode,
        }
    }

    /// Stride em bytes desta instrução (32 ou 64).
    pub fn byte_len(&self) -> usize {
        match *self {
            ProgramInstr::W32(_) => INSTR_SIZE,
            ProgramInstr::W64(g) => g.byte_len(),
        }
    }

    /// Mnemônico (W64: "UNKNOWN" até as RFCs das Fases 7/9).
    pub fn mnemonic(&self) -> &'static str {
        match *self {
            ProgramInstr::W32(i) => i.mnemonic(),
            ProgramInstr::W64(g) => g.mnemonic(),
        }
    }
}

impl std::fmt::Display for ProgramInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ProgramInstr::W32(i) => write!(f, "{}", i),
            ProgramInstr::W64(g) => write!(f, "{}", g),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers de construção (ergonomia para assembler / testes)
// ---------------------------------------------------------------------------

pub fn instr_tensor(rdest: u8, rsrc1_shape: u8, rsrc2_dtype: u8, rows: u64, cols: u64, dtype: u8) -> Instruction {
    let mut payload = [0u8; 26];
    payload[0..8].copy_from_slice(&rows.to_le_bytes());
    payload[8..16].copy_from_slice(&cols.to_le_bytes());
    payload[16] = dtype;
    Instruction {
        opcode: OP_TENSOR,
        flags: 0,
        rdest,
        rsrc1: rsrc1_shape,
        rsrc2: rsrc2_dtype,
        rsrc3: 0xFF,
        payload,
    }
}

pub fn instr_tensor_sparse(rdest: u8, rows: u64, cols: u64, dtype: u8, density: f32) -> Instruction {
    let mut instr = instr_tensor(rdest, 0xFF, 0xFF, rows, cols, dtype);
    instr.set_sparse(true, density);
    instr
}

pub fn instr_attn(rdest: u8, r_q: u8, r_k: u8, r_v: u8) -> Instruction {
    Instruction::new(OP_ATTN, 0, rdest, r_q, r_k, r_v)
}

pub fn instr_attn_notify(rdest: u8, r_q: u8, r_k: u8, r_v: u8) -> Instruction {
    Instruction::new(OP_ATTN, ATTN_FLAG_NOTIFY_EACH_HEAD, rdest, r_q, r_k, r_v)
}

pub fn instr_stream(r_src: u8, r_sink: u8, blocking: bool) -> Instruction {
    let flags = if blocking {
        STREAM_FLAG_BLOCKING
    } else {
        STREAM_FLAG_DROP
    };
    Instruction::new(OP_STREAM, flags, 0xFF, r_src, r_sink, 0xFF)
}

pub fn instr_fork(rdest: u8, priority_flag: u8) -> Instruction {
    Instruction::new(OP_FORK, priority_flag, rdest, 0xFF, 0xFF, 0xFF)
}

pub fn instr_abort(r_target: u8, r_timestamp: u8) -> Instruction {
    Instruction::new(OP_ABORT, 0, 0xFF, r_target, r_timestamp, 0xFF)
}

pub fn instr_sense(rdest: u8, peripheral: u8) -> Instruction {
    Instruction::new(OP_SENSE, 0, rdest, peripheral, 0xFF, 0xFF)
}

pub fn instr_norm(rdest: u8, r_src: u8, r_gamma: u8, r_beta: u8) -> Instruction {
    Instruction::new(OP_NORM, 0, rdest, r_src, r_gamma, r_beta)
}

pub fn instr_ffn(rdest: u8, r_src: u8, r_w1: u8, r_w2: u8) -> Instruction {
    let mut instr = Instruction::new(OP_FFN, 0, rdest, r_src, r_w1, r_w2);
    instr.payload[0] = 0xFF;
    instr.payload[1] = 0xFF;
    instr
}

pub fn instr_ffn_with_bias(rdest: u8, r_src: u8, r_w1: u8, r_b1: u8, r_w2: u8, r_b2: u8) -> Instruction {
    let mut instr = Instruction::new(OP_FFN, 0, rdest, r_src, r_w1, r_w2);
    instr.payload[0] = r_b1;
    instr.payload[1] = r_b2;
    instr
}

pub fn instr_halt() -> Instruction {
    Instruction::new(OP_HALT, 0, 0xFF, 0xFF, 0xFF, 0xFF)
}

pub fn instr_nop() -> Instruction {
    Instruction::new(OP_NOP, 0, 0xFF, 0xFF, 0xFF, 0xFF)
}

pub fn instr_embed(rdest: u8, rtoken: u8, rtable: u8) -> Instruction {
    Instruction::new(OP_EMBED, 0, rdest, rtoken, rtable, 0xFF)
}

pub fn instr_add(rdest: u8, rsrc1: u8, rsrc2: u8) -> Instruction {
    Instruction::new(OP_ADD, 0, rdest, rsrc1, rsrc2, 0xFF)
}

pub fn instr_sample(rdest: u8, rlogits: u8, temp: f32) -> Instruction {
    let mut instr = Instruction::new(OP_SAMPLE, 0, rdest, rlogits, 0xFF, 0xFF);
    instr.payload[0..4].copy_from_slice(&temp.to_le_bytes());
    instr
}

pub fn instr_compare(rsrc1: u8, rsrc2: u8, imm: u128) -> Instruction {
    let mut instr = Instruction::new(OP_COMPARE, 0, 0xFF, rsrc1, rsrc2, 0xFF);
    instr.set_imm_u128(imm);
    instr
}

pub fn instr_if_equal(target_pc: u128) -> Instruction {
    let mut instr = Instruction::new(OP_IF_EQUAL, 0, 0xFF, 0xFF, 0xFF, 0xFF);
    instr.set_imm_u128(target_pc);
    instr
}

pub fn instr_jump(target_pc: u128) -> Instruction {
    let mut instr = Instruction::new(OP_JUMP, 0, 0xFF, 0xFF, 0xFF, 0xFF);
    instr.set_imm_u128(target_pc);
    instr
}

pub fn instr_if_interrupt(rcond: u8, target_pc: u128) -> Instruction {
    let mut instr = Instruction::new(OP_IF_INTERRUPT, 0, 0xFF, rcond, 0xFF, 0xFF);
    instr.set_imm_u128(target_pc);
    instr
}

pub fn instr_matvec(rdest: u8, r_x: u8, r_w: u8) -> Instruction {
    Instruction::new(OP_MATVEC, 0, rdest, r_x, r_w, 0xFF)
}

pub fn instr_mul(rdest: u8, rsrc1: u8, rsrc2: u8) -> Instruction {
    Instruction::new(OP_MUL, 0, rdest, rsrc1, rsrc2, 0xFF)
}

pub fn instr_silu(rdest: u8, rsrc: u8) -> Instruction {
    Instruction::new(OP_SILU, 0, rdest, rsrc, 0xFF, 0xFF)
}

/// SSM_SCAN Rd,Rx,Rh,Rp — y = scan(x, h, dt/A/B/C/D). Rp = pack TEMPORAL/GLOBAL
/// com dt[d_inner]+A[d_inner*d_state]+B[d_state]+C[d_state]+D[d_inner] f32 LE.
/// Se Rh == 0xFF, usa Vm.ssm_states[layer_id] (fase 2 híbrida).
pub fn instr_ssm_scan(rdest: u8, r_x: u8, r_h: u8, r_params: u8, d_inner: usize, d_state: usize, layer_id: u8) -> Instruction {
    let mut instr = Instruction::new(OP_SSM_SCAN, 0, rdest, r_x, r_h, r_params);
    instr.set_ssm_dims(d_inner, d_state, layer_id);
    instr
}

pub fn instr_ssm_reset(r_h: u8, d_inner: usize, d_state: usize, layer_id: u8) -> Instruction {
    let mut instr = Instruction::new(OP_SSM_RESET, 0, 0xFF, r_h, 0xFF, 0xFF);
    instr.set_ssm_dims(d_inner, d_state, layer_id);
    instr
}

pub fn instr_codec_enc(rdest: u8, r_pcm: u8, as_tensor: bool) -> Instruction {
    let flags = if as_tensor { CODEC_FLAG_AS_TENSOR } else { 0 };
    Instruction::new(OP_CODEC_ENC, flags, rdest, r_pcm, 0xFF, 0xFF)
}

pub fn instr_codec_dec(rdest: u8, r_codes: u8, as_tensor: bool) -> Instruction {
    let flags = if as_tensor { CODEC_FLAG_AS_TENSOR } else { 0 };
    Instruction::new(OP_CODEC_DEC, flags, rdest, r_codes, 0xFF, 0xFF)
}

pub fn instr_audio_align(rdest: u8, r_user: u8, r_ai: u8) -> Instruction {
    let mut instr = Instruction::new(OP_AUDIO_ALIGN, 0, rdest, r_user, r_ai, 0xFF);
    instr.set_audio_align_params(24_000, 1920, 12.5, 160.0);
    instr
}

/// Marcador de que `payload[0]` carrega o pipe imediato (ver `ctx_switch_pipe`).
pub const CTX_SWITCH_PAYLOAD_MAGIC: u8 = 0xA5;

/// `rsrc1` carrega o pipe imediato (0/1/2) por compatibilidade histórica, mas
/// `payload[0..2]=[pipe, MAGIC]` é a fonte canônica — elimina a ambiguidade
/// com índices de registrador `r0–r2`.
pub fn instr_ctx_switch(pipe_id: u8, priority_flag: u8) -> Instruction {
    let mut instr = Instruction::new(OP_CTX_SWITCH, priority_flag, 0xFF, pipe_id, 0xFF, 0xFF);
    instr.payload[0] = pipe_id;
    instr.payload[1] = CTX_SWITCH_PAYLOAD_MAGIC;
    instr
}

/// Resolve o pipeline alvo: prefere `payload[0]` com MAGIC; senão `rsrc1`
/// (legado: montagem manual sem o construtor).
pub fn ctx_switch_pipe(instr: &Instruction) -> u8 {
    if instr.payload[1] == CTX_SWITCH_PAYLOAD_MAGIC {
        instr.payload[0]
    } else {
        instr.rsrc1
    }
}

pub fn instr_rope(rdest: u8, r_src: u8, pos: u32, head_dim: usize, n_heads: usize, theta: f32) -> Instruction {
    let mut instr = Instruction::new(OP_ROPE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_rope_params(pos, head_dim, n_heads, theta);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0018: cluster F1 (0x1A–0x1D). Payloads nos 26B, exatos do plano
// congelado (node u32 + ids u64). node_id != 0 veta no exec (F2+).
// ---------------------------------------------------------------------------

impl Instruction {
    /// REMOTE_SPAWN: payload[0..4]=node u32, [4..12]=entry_pc u64,
    /// [12]=prio (0=GREEN,1=BLUE,2=RED).
    pub fn remote_spawn_params(&self) -> (u32, u64, u8) {
        let mut bn = [0u8; 4];
        bn.copy_from_slice(&self.payload[0..4]);
        let mut be = [0u8; 8];
        be.copy_from_slice(&self.payload[4..12]);
        (u32::from_le_bytes(bn), u64::from_le_bytes(be), self.payload[12])
    }

    pub fn set_remote_spawn_params(&mut self, node: u32, entry_pc: u64, prio: u8) {
        self.payload[0..4].copy_from_slice(&node.to_le_bytes());
        self.payload[4..12].copy_from_slice(&entry_pc.to_le_bytes());
        self.payload[12] = prio;
    }

    /// SIGNAL: payload[0..4]=node u32, [4..12]=ctx_id u64, [12..20]=seq u64.
    /// kind via rsrc1 como IMEDIATO 0..3 (não registrador: evita indireção
    /// no caminho crítico de preempção).
    pub fn signal_params(&self) -> (u32, u64, u64) {
        let mut bn = [0u8; 4];
        bn.copy_from_slice(&self.payload[0..4]);
        let mut bc = [0u8; 8];
        bc.copy_from_slice(&self.payload[4..12]);
        let mut bs = [0u8; 8];
        bs.copy_from_slice(&self.payload[12..20]);
        (u32::from_le_bytes(bn), u64::from_le_bytes(bc), u64::from_le_bytes(bs))
    }

    pub fn set_signal_params(&mut self, node: u32, ctx_id: u64, seq: u64) {
        self.payload[0..4].copy_from_slice(&node.to_le_bytes());
        self.payload[4..12].copy_from_slice(&ctx_id.to_le_bytes());
        self.payload[12..20].copy_from_slice(&seq.to_le_bytes());
    }

    /// SEND_TENSOR: payload[0..4]=node u32, [4..12]=byte_offset u64,
    /// [12..16]=byte_len u32, [16]=mode (0=COPY,1=MOVE).
    pub fn send_tensor_params(&self) -> (u32, u64, u32, u8) {
        let mut bn = [0u8; 4];
        bn.copy_from_slice(&self.payload[0..4]);
        let mut bo = [0u8; 8];
        bo.copy_from_slice(&self.payload[4..12]);
        let mut bl = [0u8; 4];
        bl.copy_from_slice(&self.payload[12..16]);
        (
            u32::from_le_bytes(bn),
            u64::from_le_bytes(bo),
            u32::from_le_bytes(bl),
            self.payload[16],
        )
    }

    pub fn set_send_tensor_params(&mut self, node: u32, offset: u64, len: u32, mode: u8) {
        self.payload[0..4].copy_from_slice(&node.to_le_bytes());
        self.payload[4..12].copy_from_slice(&offset.to_le_bytes());
        self.payload[12..16].copy_from_slice(&len.to_le_bytes());
        self.payload[16] = mode;
    }

    /// BARRIER: payload[0..4]=barrier_id u32, [4..6]=expected u16,
    /// [6..8]=timeout_ms u16, [8..12]=epoch u32.
    pub fn barrier_params(&self) -> (u32, u16, u16, u32) {
        let mut bi = [0u8; 4];
        bi.copy_from_slice(&self.payload[0..4]);
        let ex = u16::from_le_bytes([self.payload[4], self.payload[5]]);
        let to = u16::from_le_bytes([self.payload[6], self.payload[7]]);
        let mut be = [0u8; 4];
        be.copy_from_slice(&self.payload[8..12]);
        (u32::from_le_bytes(bi), ex, to, u32::from_le_bytes(be))
    }

    pub fn set_barrier_params(&mut self, id: u32, expected: u16, timeout_ms: u16, epoch: u32) {
        self.payload[0..4].copy_from_slice(&id.to_le_bytes());
        self.payload[4..6].copy_from_slice(&expected.to_le_bytes());
        self.payload[6..8].copy_from_slice(&timeout_ms.to_le_bytes());
        self.payload[8..12].copy_from_slice(&epoch.to_le_bytes());
    }
}

/// REMOTE_SPAWN rD, node, entry_pc, prio — rdest recebe ctx_id remoto.
/// (Assembler: NODE=n ENTRY=label|pc PRI=...; node!=0 monta, exec veta.)
pub fn instr_remote_spawn(rdest: u8, node: u32, entry_pc: u64, prio: u8) -> Instruction {
    let mut instr = Instruction::new(OP_REMOTE_SPAWN, 0, rdest, 0xFF, 0xFF, 0xFF);
    instr.set_remote_spawn_params(node, entry_pc, prio);
    instr
}

/// SIGNAL rD, kind(0..3 imm), node, ctx_id, seq — rdest = status/ack.
pub fn instr_signal(rdest: u8, kind: u8, node: u32, ctx_id: u64, seq: u64) -> Instruction {
    let mut instr = Instruction::new(OP_SIGNAL, 0, rdest, kind, 0xFF, 0xFF);
    instr.set_signal_params(node, ctx_id, seq);
    instr
}

/// SEND_TENSOR rSrc, rDst|0xFF, node, offset, len, mode.
pub fn instr_send_tensor(r_src: u8, r_dst: u8, node: u32, offset: u64, len: u32, mode: u8) -> Instruction {
    let mut instr = Instruction::new(OP_SEND_TENSOR, 0, r_src, r_dst, 0xFF, 0xFF);
    instr.set_send_tensor_params(node, offset, len, mode);
    instr
}

/// BARRIER id, expected, timeout_ms, epoch.
pub fn instr_barrier(id: u32, expected: u16, timeout_ms: u16, epoch: u32) -> Instruction {
    let mut instr = Instruction::new(OP_BARRIER, 0, 0xFF, 0xFF, 0xFF, 0xFF);
    instr.set_barrier_params(id, expected, timeout_ms, epoch);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0004: GATHER (0x1F) / DISTANCE (0x23) / RANK1_UPDATE (0x24)
// Layouts idênticos aos congelados (ESPEC §5.5 / PLANO_ISA_UNIVERSAL).
// ---------------------------------------------------------------------------

impl Instruction {
    /// GATHER: payload[0]=axis u8, [1]=mode u8, [2..4]=nnz_hint u16 LE.
    pub fn gather_params(&self) -> (u8, u8, u16) {
        let axis = self.payload[0];
        let mode = self.payload[1];
        let hint = u16::from_le_bytes([self.payload[2], self.payload[3]]);
        (axis, mode, hint)
    }

    pub fn set_gather_params(&mut self, axis: u8, mode: u8, nnz_hint: u16) {
        self.payload[0] = axis;
        self.payload[1] = mode;
        self.payload[2..4].copy_from_slice(&nnz_hint.to_le_bytes());
    }

    /// DISTANCE: payload[0]=metric u8, [1..3]=topk u16 LE (0 = todas).
    pub fn distance_params(&self) -> (u8, u16) {
        let metric = self.payload[0];
        let topk = u16::from_le_bytes([self.payload[1], self.payload[2]]);
        (metric, topk)
    }

    pub fn set_distance_params(&mut self, metric: u8, topk: u16) {
        self.payload[0] = metric;
        self.payload[1..3].copy_from_slice(&topk.to_le_bytes());
    }

    /// RANK1_UPDATE: payload[0..4]=alpha f32 LE, [4..8]=beta f32 LE,
    /// [8]=mode u8, [9]=layer_id u8.
    pub fn rank1_params(&self) -> (f32, f32, u8, u8) {
        let mut ba = [0u8; 4];
        ba.copy_from_slice(&self.payload[0..4]);
        let mut bb = [0u8; 4];
        bb.copy_from_slice(&self.payload[4..8]);
        (
            f32::from_le_bytes(ba),
            f32::from_le_bytes(bb),
            self.payload[8],
            self.payload[9],
        )
    }

    pub fn set_rank1_params(&mut self, alpha: f32, beta: f32, mode: u8, layer_id: u8) {
        self.payload[0..4].copy_from_slice(&alpha.to_le_bytes());
        self.payload[4..8].copy_from_slice(&beta.to_le_bytes());
        self.payload[8] = mode;
        self.payload[9] = layer_id;
    }

    /// SAMPLE TOPK: payload[4..6]=topk u16 LE (0 = amostragem legada).
    /// payload[0..4] segue TEMP (irrelevante no modo TOPK).
    pub fn sample_topk(&self) -> u16 {
        u16::from_le_bytes([self.payload[4], self.payload[5]])
    }

    pub fn set_sample_topk(&mut self, topk: u16) {
        self.payload[4..6].copy_from_slice(&topk.to_le_bytes());
    }
}

/// GATHER rD, rTable, rIdx [, rAcc] [AXIS=n] [MODE=GATHER|SCATTER_ADD|SCATTER_MAX]
pub fn instr_gather(rdest: u8, r_table: u8, r_idx: u8, r_acc: u8, axis: u8, mode: u8) -> Instruction {
    let mut instr = Instruction::new(OP_GATHER, 0, rdest, r_table, r_idx, r_acc);
    instr.set_gather_params(axis, mode, 0);
    instr
}

/// DISTANCE rD, rQuery, rBank [METRIC=...] [TOPK=n]
pub fn instr_distance(rdest: u8, r_query: u8, r_bank: u8, metric: u8, topk: u16) -> Instruction {
    let mut instr = Instruction::new(OP_DISTANCE, 0, rdest, r_query, r_bank, 0xFF);
    instr.set_distance_params(metric, topk);
    instr
}

/// RANK1_UPDATE rH, rV, rK [ALPHA=a] [BETA=b] [MODE=...] [LAYER=n]
pub fn instr_rank1_update(r_h: u8, r_v: u8, r_k: u8, alpha: f32, beta: f32, mode: u8, layer_id: u8) -> Instruction {
    let mut instr = Instruction::new(OP_RANK1_UPDATE, 0, r_h, r_v, r_k, 0xFF);
    instr.set_rank1_params(alpha, beta, mode, layer_id);
    instr
}

/// SAMPLE rD, rLogits TOPK=k — escreve tensor de índices [1,k].
pub fn instr_sample_topk(rdest: u8, r_logits: u8, topk: u16) -> Instruction {
    let mut instr = instr_sample(rdest, r_logits, 1.0);
    instr.set_sample_topk(topk);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0005: determinismo (0x60–0x66). Payloads: UNIFORM a/b em [0..8],
// NORMAL mean/std em [0..8]; SEED/NEXT/HASH/CHECKSUM/HMAC sem payload.
// ---------------------------------------------------------------------------

impl Instruction {
    /// UNIFORM: payload[0..4]=a f32 LE, [4..8]=b f32 LE (default 0/1).
    pub fn uniform_range(&self) -> (f32, f32) {
        let mut ba = [0u8; 4];
        ba.copy_from_slice(&self.payload[0..4]);
        let mut bb = [0u8; 4];
        bb.copy_from_slice(&self.payload[4..8]);
        let (a, b) = (f32::from_le_bytes(ba), f32::from_le_bytes(bb));
        // Payload zerado = default [0,1) (compat: construtor sem args).
        if a == 0.0 && b == 0.0 {
            (0.0, 1.0)
        } else {
            (a, b)
        }
    }

    pub fn set_uniform_range(&mut self, a: f32, b: f32) {
        self.payload[0..4].copy_from_slice(&a.to_le_bytes());
        self.payload[4..8].copy_from_slice(&b.to_le_bytes());
    }

    /// NORMAL: payload[0..4]=mean f32 LE, [4..8]=std f32 LE.
    /// std==0 carrega o default 1.0 (mesma convenção do UNIFORM).
    pub fn normal_params(&self) -> (f32, f32) {
        let mut bm = [0u8; 4];
        bm.copy_from_slice(&self.payload[0..4]);
        let mut bs = [0u8; 4];
        bs.copy_from_slice(&self.payload[4..8]);
        let std = f32::from_le_bytes(bs);
        (f32::from_le_bytes(bm), if std == 0.0 { 1.0 } else { std })
    }

    pub fn set_normal_params(&mut self, mean: f32, std: f32) {
        self.payload[0..4].copy_from_slice(&mean.to_le_bytes());
        self.payload[4..8].copy_from_slice(&std.to_le_bytes());
    }
}

/// RNG_SEED Rs|0xFF — rsrc1 com a semente u64 (0xFF = default fixo).
pub fn instr_rng_seed(r_seed: u8) -> Instruction {
    Instruction::new(OP_RNG_SEED, 0, 0xFF, r_seed, 0xFF, 0xFF)
}

/// RNG_NEXT rD — rdest <- próximo u64.
pub fn instr_rng_next(rdest: u8) -> Instruction {
    Instruction::new(OP_RNG_NEXT, 0, rdest, 0xFF, 0xFF, 0xFF)
}

/// RNG_UNIFORM rD [A=a] [B=b] — rdest <- f32 em [a,b).
pub fn instr_rng_uniform(rdest: u8, a: f32, b: f32) -> Instruction {
    let mut instr = Instruction::new(OP_RNG_UNIFORM, 0, rdest, 0xFF, 0xFF, 0xFF);
    instr.set_uniform_range(a, b);
    instr
}

/// RNG_NORMAL rD [MEAN=m] [STD=s] — rdest <- normal(m,s).
pub fn instr_rng_normal(rdest: u8, mean: f32, std: f32) -> Instruction {
    let mut instr = Instruction::new(OP_RNG_NORMAL, 0, rdest, 0xFF, 0xFF, 0xFF);
    instr.set_normal_params(mean, std);
    instr
}

/// HASH rD, rT — rdest <- FNV-1a/64 do tensor.
pub fn instr_hash(rdest: u8, r_tensor: u8) -> Instruction {
    Instruction::new(OP_HASH, 0, rdest, r_tensor, 0xFF, 0xFF)
}

/// CHECKSUM rD, rT — rdest <- CRC32-IEEE do tensor.
pub fn instr_checksum(rdest: u8, r_tensor: u8) -> Instruction {
    Instruction::new(OP_CHECKSUM, 0, rdest, r_tensor, 0xFF, 0xFF)
}

/// HMAC rD, rKey, rMsg — rdest <- HMAC-SHA256 truncado (u64 BE).
pub fn instr_hmac(rdest: u8, r_key: u8, r_msg: u8) -> Instruction {
    Instruction::new(OP_HMAC, 0, rdest, r_key, r_msg, 0xFF)
}

// ---------------------------------------------------------------------------
// RFC-0006: telemetria (0x6A–0x6F) + scheduler (0x70–0x77).
// ---------------------------------------------------------------------------

impl Instruction {
    /// ASSERT: payload[0..2] = code u16 LE (default 0).
    pub fn assert_code(&self) -> u16 {
        u16::from_le_bytes([self.payload[0], self.payload[1]])
    }

    pub fn set_assert_code(&mut self, code: u16) {
        self.payload[0..2].copy_from_slice(&code.to_le_bytes());
    }
}

/// CYCLES_COUNT rD — rdest <- now_ns (métrica, não tempo real).
pub fn instr_cycles_count(rdest: u8) -> Instruction {
    Instruction::new(OP_CYCLES_COUNT, 0, rdest, 0xFF, 0xFF, 0xFF)
}

/// TRACE_EVENT rEv, rData — anexa (evento, dado) ao anel de trace.
pub fn instr_trace_event(r_event: u8, r_data: u8) -> Instruction {
    Instruction::new(OP_TRACE_EVENT, 0, 0xFF, r_event, r_data, 0xFF)
}

/// SANITY_CHECK rD, rT [, rCount] — tensor saneado (CoW) em rD;
/// rCount (se != 0xFF) <- nº de elementos não-finitos absorvidos.
pub fn instr_sanity_check(rdest: u8, r_tensor: u8, r_count: u8) -> Instruction {
    Instruction::new(OP_SANITY_CHECK, 0, rdest, r_tensor, 0xFF, r_count)
}

/// PREEMPT_CHECK rD — rdest <- interrupt_flag (NÃO consome).
pub fn instr_preempt_check(rdest: u8) -> Instruction {
    Instruction::new(OP_PREEMPT_CHECK, 0, rdest, 0xFF, 0xFF, 0xFF)
}

/// ASSERT Rs [CODE=n] — trap se reg == 0.
pub fn instr_assert(r_cond: u8, code: u16) -> Instruction {
    let mut instr = Instruction::new(OP_ASSERT, 0, 0xFF, r_cond, 0xFF, 0xFF);
    instr.set_assert_code(code);
    instr
}

/// DUMP — log de debug do contexto corrente (sem operandos).
pub fn instr_dump() -> Instruction {
    Instruction::new(OP_DUMP, 0, 0xFF, 0xFF, 0xFF, 0xFF)
}

/// YIELD — cede + maybe_preempt.
pub fn instr_yield() -> Instruction {
    Instruction::new(OP_YIELD, 0, 0xFF, 0xFF, 0xFF, 0xFF)
}

/// SET_DEADLINE Rs — deadline EDF absoluto (ns) do valor no reg.
pub fn instr_set_deadline(r_deadline: u8) -> Instruction {
    Instruction::new(OP_SET_DEADLINE, 0, 0xFF, r_deadline, 0xFF, 0xFF)
}

/// GET_DEADLINE rD — rdest <- deadline corrente.
pub fn instr_get_deadline(rdest: u8) -> Instruction {
    Instruction::new(OP_GET_DEADLINE, 0, rdest, 0xFF, 0xFF, 0xFF)
}

/// PRIORITY_SET Rs — reg com 0/1/2 (GREEN/BLUE/RED) + re-enfileira.
pub fn instr_priority_set(r_prio: u8) -> Instruction {
    Instruction::new(OP_PRIORITY_SET, 0, 0xFF, r_prio, 0xFF, 0xFF)
}

/// PRIORITY_GET rD — rdest <- prioridade (0/1/2).
pub fn instr_priority_get(rdest: u8) -> Instruction {
    Instruction::new(OP_PRIORITY_GET, 0, rdest, 0xFF, 0xFF, 0xFF)
}

/// LOCK Rs — try-lock do id no reg (não-bloqueante).
pub fn instr_lock(r_id: u8) -> Instruction {
    Instruction::new(OP_LOCK, 0, 0xFF, r_id, 0xFF, 0xFF)
}

/// UNLOCK Rs — libera lock próprio do id no reg.
pub fn instr_unlock(r_id: u8) -> Instruction {
    Instruction::new(OP_UNLOCK, 0, 0xFF, r_id, 0xFF, 0xFF)
}

/// FENCE — barreira (marcador + compiler fence).
pub fn instr_fence() -> Instruction {
    Instruction::new(OP_FENCE, 0, 0xFF, 0xFF, 0xFF, 0xFF)
}

// ---------------------------------------------------------------------------
// RFC-0007: LOADI (0x78) / MOV (0x79) + predicados COMPARE.
// ---------------------------------------------------------------------------

impl Instruction {
    /// COMPARE: payload[16] = predicado (0=EQ legado; resto ver CMP_*).
    pub fn compare_pred(&self) -> u8 {
        self.payload[16]
    }

    pub fn set_compare_pred(&mut self, pred: u8) {
        self.payload[16] = pred;
    }
}

/// LOADI rD, imm — rdest <- imm u128 (payload[0..16], convenção imm_u128).
pub fn instr_loadi(rdest: u8, imm: u128) -> Instruction {
    let mut instr = Instruction::new(OP_LOADI, 0, rdest, 0xFF, 0xFF, 0xFF);
    instr.set_imm_u128(imm);
    instr
}

/// MOV rD, rS — rdest <- reg[rsrc1].
pub fn instr_mov(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_MOV, 0, rdest, r_src, 0xFF, 0xFF)
}

// ---------------------------------------------------------------------------
// RFC-0010: KV_TRUNCATE (0x38). payload[0..2] = stream_id u16 (0 = todas;
// nonzero reservado p/ KV 17-streams futuro).
// ---------------------------------------------------------------------------

impl Instruction {
    pub fn kv_stream(&self) -> u16 {
        u16::from_le_bytes([self.payload[0], self.payload[1]])
    }

    pub fn set_kv_stream(&mut self, stream: u16) {
        self.payload[0..2].copy_from_slice(&stream.to_le_bytes());
    }
}

/// KV_TRUNCATE Rs_len [, STREAM=sid] — trunca todas as camadas p/ len.
pub fn instr_kv_truncate(r_len: u8, stream: u16) -> Instruction {
    let mut instr = Instruction::new(OP_KV_TRUNCATE, 0, 0xFF, r_len, 0xFF, 0xFF);
    instr.set_kv_stream(stream);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0019: TENSOR FILL + SLICE (0x2E). Fill escalar em payload[22..26]
// (bytes zero em binários antigos = sem fill); slice flat [start,len).
// ---------------------------------------------------------------------------

impl Instruction {
    /// FILL: (presente, valor). Presença = flags & TENSOR_FLAG_FILL.
    pub fn tensor_fill(&self) -> Option<f32> {
        if self.opcode == OP_TENSOR && (self.flags & TENSOR_FLAG_FILL) != 0 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&self.payload[22..26]);
            Some(f32::from_le_bytes(b))
        } else {
            None
        }
    }

    pub fn set_tensor_fill(&mut self, v: f32) {
        self.flags |= TENSOR_FLAG_FILL;
        self.payload[22..26].copy_from_slice(&v.to_le_bytes());
    }

    /// SLICE: payload[0..4]=start u32, [4..8]=len u32.
    pub fn slice_params(&self) -> (u32, u32) {
        let mut bs = [0u8; 4];
        bs.copy_from_slice(&self.payload[0..4]);
        let mut bl = [0u8; 4];
        bl.copy_from_slice(&self.payload[4..8]);
        (u32::from_le_bytes(bs), u32::from_le_bytes(bl))
    }

    pub fn set_slice_params(&mut self, start: u32, len: u32) {
        self.payload[0..4].copy_from_slice(&start.to_le_bytes());
        self.payload[4..8].copy_from_slice(&len.to_le_bytes());
    }
}

/// SLICE rD, rT START=n LEN=n — fatia flat => tensor [1,len].
pub fn instr_slice(rdest: u8, r_tensor: u8, start: u32, len: u32) -> Instruction {
    let mut instr = Instruction::new(OP_SLICE, 0, rdest, r_tensor, 0xFF, 0xFF);
    instr.set_slice_params(start, len);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0023: ARENA_ALLOC (0x26) / ARENA_RESET (0x27) / MEMCPY (0x2A) /
// MEMSET (0x2B). Layouts nos 26B congelados (ver RFC).
// ---------------------------------------------------------------------------

impl Instruction {
    /// ARENA_ALLOC: payload[0..8]=size u64 LE, [8..12]=align u32 LE
    /// (0=default), [12]=arena_id u8.
    pub fn arena_alloc_params(&self) -> (u64, u32, u8) {
        let mut bs = [0u8; 8];
        bs.copy_from_slice(&self.payload[0..8]);
        let mut ba = [0u8; 4];
        ba.copy_from_slice(&self.payload[8..12]);
        (
            u64::from_le_bytes(bs),
            u32::from_le_bytes(ba),
            self.payload[12],
        )
    }

    pub fn set_arena_alloc_params(&mut self, size: u64, align: u32, arena: u8) {
        self.payload[0..8].copy_from_slice(&size.to_le_bytes());
        self.payload[8..12].copy_from_slice(&align.to_le_bytes());
        self.payload[12] = arena;
    }

    /// ARENA_RESET: payload[0]=arena_id u8.
    pub fn arena_id(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_arena_id(&mut self, arena: u8) {
        self.payload[0] = arena;
    }

    /// MEMCPY: payload[0..8]=len u64 LE (0=tensor fonte inteiro),
    /// [8..16]=src_off u64, [16..24]=dst_off u64, [24]=dir u8.
    pub fn memcpy_params(&self) -> (u64, u64, u64, u8) {
        let mut bl = [0u8; 8];
        bl.copy_from_slice(&self.payload[0..8]);
        let mut bs = [0u8; 8];
        bs.copy_from_slice(&self.payload[8..16]);
        let mut bd = [0u8; 8];
        bd.copy_from_slice(&self.payload[16..24]);
        (
            u64::from_le_bytes(bl),
            u64::from_le_bytes(bs),
            u64::from_le_bytes(bd),
            self.payload[24],
        )
    }

    pub fn set_memcpy_params(&mut self, len: u64, src_off: u64, dst_off: u64, dir: u8) {
        self.payload[0..8].copy_from_slice(&len.to_le_bytes());
        self.payload[8..16].copy_from_slice(&src_off.to_le_bytes());
        self.payload[16..24].copy_from_slice(&dst_off.to_le_bytes());
        self.payload[24] = dir;
    }

    /// MEMSET: payload[0]=pattern byte, [1..5]=len u32 LE (0=tensor
    /// inteiro), [5..13]=offset u64 LE. Byte-level, não value-level.
    pub fn memset_params(&self) -> (u8, u32, u64) {
        let mut bl = [0u8; 4];
        bl.copy_from_slice(&self.payload[1..5]);
        let mut bo = [0u8; 8];
        bo.copy_from_slice(&self.payload[5..13]);
        (
            self.payload[0],
            u32::from_le_bytes(bl),
            u64::from_le_bytes(bo),
        )
    }

    pub fn set_memset_params(&mut self, pattern: u8, len: u32, offset: u64) {
        self.payload[0] = pattern;
        self.payload[1..5].copy_from_slice(&len.to_le_bytes());
        self.payload[5..13].copy_from_slice(&offset.to_le_bytes());
    }
}

/// ARENA_ALLOC rD, SIZE=n [ALIGN=n] [ARENA=id] — rdest <- offset em bytes.
pub fn instr_arena_alloc(rdest: u8, size: u64, align: u32, arena: u8) -> Instruction {
    let mut instr = Instruction::new(OP_ARENA_ALLOC, 0, rdest, 0xFF, 0xFF, 0xFF);
    instr.set_arena_alloc_params(size, align, arena);
    instr
}

/// ARENA_RESET [ARENA=id] — cursor=0 O(1), capacidade mantida.
pub fn instr_arena_reset(arena: u8) -> Instruction {
    let mut instr = Instruction::new(OP_ARENA_RESET, 0, 0xFF, 0xFF, 0xFF, 0xFF);
    instr.set_arena_id(arena);
    instr
}

/// MEMCPY rDst, rSrc [LEN=n] [SRC_OFF=n] [DST_OFF=n] [DIR=HOST] —
/// rDst/rSrc guardam addrs de tensores (não são reescritos).
pub fn instr_memcpy(r_dst: u8, r_src: u8, len: u64, src_off: u64, dst_off: u64, dir: u8) -> Instruction {
    let mut instr = Instruction::new(OP_MEMCPY, 0, r_dst, r_src, 0xFF, 0xFF);
    instr.set_memcpy_params(len, src_off, dst_off, dir);
    instr
}

/// MEMSET rT, PATTERN=n [LEN=n] [OFF=n] — fill byte-level no tensor.
pub fn instr_memset(r_tensor: u8, pattern: u8, len: u32, offset: u64) -> Instruction {
    let mut instr = Instruction::new(OP_MEMSET, 0, r_tensor, 0xFF, 0xFF, 0xFF);
    instr.set_memset_params(pattern, len, offset);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0024: SNAPSHOT (0x28) / RESTORE (0x29) / PREFETCH (0x2C) /
// RESHAPE (0x2D) / CONCAT (0x2F). Layouts nos 26B congelados (ver RFC).
// ---------------------------------------------------------------------------

impl Instruction {
    /// SNAPSHOT: payload[0]=mask u8 (só 0b111 executa; resto veta).
    pub fn snapshot_mask(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_snapshot_mask(&mut self, mask: u8) {
        self.payload[0] = mask;
    }

    /// PREFETCH: payload[0..4]=len u32 LE (0=tensor inteiro),
    /// [4..12]=offset u64 LE.
    pub fn prefetch_params(&self) -> (u32, u64) {
        let mut bl = [0u8; 4];
        bl.copy_from_slice(&self.payload[0..4]);
        let mut bo = [0u8; 8];
        bo.copy_from_slice(&self.payload[4..12]);
        (u32::from_le_bytes(bl), u64::from_le_bytes(bo))
    }

    pub fn set_prefetch_params(&mut self, len: u32, offset: u64) {
        self.payload[0..4].copy_from_slice(&len.to_le_bytes());
        self.payload[4..12].copy_from_slice(&offset.to_le_bytes());
    }

    /// RESHAPE: payload[0]=ndim u8 (1-4), [1..5]=d0 u32 LE, [5..9]=d1,
    /// [9..13]=d2, [13..17]=d3. Cópia (não view); numel validado no exec.
    /// Layout compartilhado com BROADCAST (fns livres abaixo).
    pub fn reshape_shape(&self) -> (u8, [u32; 4]) {
        reshape_dims_of(self)
    }

    pub fn set_reshape_shape(&mut self, ndim: u8, dims: [u32; 4]) {
        set_reshape_dims(self, ndim, dims);
    }

    /// CONCAT: payload[0]=axis u8 (default 0).
    pub fn concat_axis(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_concat_axis(&mut self, axis: u8) {
        self.payload[0] = axis;
    }
}

/// SNAPSHOT rD [MASK=n] — rdest <- version u64 (só MASK=0b111 executa).
pub fn instr_snapshot(rdest: u8, mask: u8) -> Instruction {
    let mut instr = Instruction::new(OP_SNAPSHOT, 0, rdest, 0xFF, 0xFF, 0xFF);
    instr.set_snapshot_mask(mask);
    instr
}

/// RESTORE rV — rV guarda version u64; rewind sem matar contexto.
pub fn instr_restore(r_version: u8) -> Instruction {
    Instruction::new(OP_RESTORE, 0, 0xFF, r_version, 0xFF, 0xFF)
}

/// PREFETCH rT [LEN=n] [OFF=n] — hint de cache, só leitura.
pub fn instr_prefetch(r_tensor: u8, len: u32, offset: u64) -> Instruction {
    let mut instr = Instruction::new(OP_PREFETCH, 0, r_tensor, 0xFF, 0xFF, 0xFF);
    instr.set_prefetch_params(len, offset);
    instr
}

/// RESHAPE rD, rT, dims — cópia com novo shape (1-4 dims, validadas no exec).
pub fn instr_reshape(rdest: u8, r_src: u8, dims: &[u32]) -> Instruction {
    debug_assert!((1..=4).contains(&dims.len()));
    let mut arr = [0u32; 4];
    for (i, d) in dims.iter().take(4).enumerate() {
        arr[i] = *d;
    }
    let mut instr = Instruction::new(OP_RESHAPE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_reshape_shape(dims.len().min(255) as u8, arr);
    instr
}

// ---------------------------------------------------------------------------
// Layout de shape compartilhado RESHAPE/BROADCAST (RFC-0024/0027):
// payload[0]=ndim (1-4), [1..17]=4×u32 LE. Parse `SHAPE=AxBxC` idem.
// ---------------------------------------------------------------------------

/// Lê (ndim, dims) do layout de shape (corpo de `reshape_shape` e
/// `broadcast_shape` — uma implementação só).
pub fn reshape_dims_of(instr: &Instruction) -> (u8, [u32; 4]) {
    let mut dims = [0u32; 4];
    for (i, d) in dims.iter_mut().enumerate() {
        let mut b = [0u8; 4];
        b.copy_from_slice(&instr.payload[1 + i * 4..5 + i * 4]);
        *d = u32::from_le_bytes(b);
    }
    (instr.payload[0], dims)
}

/// Escreve (ndim, dims) no layout de shape (corpo de `set_reshape_shape`
/// e `set_broadcast_shape`).
pub fn set_reshape_dims(instr: &mut Instruction, ndim: u8, dims: [u32; 4]) {
    instr.payload[0] = ndim;
    for (i, d) in dims.iter().enumerate() {
        instr.payload[1 + i * 4..5 + i * 4].copy_from_slice(&d.to_le_bytes());
    }
}

/// Parseia valor `SHAPE=` x-separado em 1-4 dims > 0. Usado por RESHAPE
/// e BROADCAST (mesma regra, uma implementação só). `v` é o valor após
/// `SHAPE=`; `op` nomeia o opcode nas mensagens.
pub fn parse_shape_dims(v: &str, op: &str) -> Result<Vec<u32>> {
    let mut dims = Vec::new();
    for d in v.split(|c| c == 'x' || c == 'X') {
        let n = d.parse::<u32>().map_err(|_| anyhow!("{} SHAPE inválido '{}' (use AxBxC)", op, v))?;
        if n == 0 {
            return Err(anyhow!("{} SHAPE com dim 0 (dims > 0)", op));
        }
        dims.push(n);
    }
    if dims.is_empty() || dims.len() > 4 {
        return Err(anyhow!("{} SHAPE '{}' precisa de 1-4 dims", op, v));
    }
    Ok(dims)
}

/// CONCAT rD, rA, rB [AXIS=n] — montagem ao longo do eixo.
pub fn instr_concat(rdest: u8, r_a: u8, r_b: u8, axis: u8) -> Instruction {
    let mut instr = Instruction::new(OP_CONCAT, 0, rdest, r_a, r_b, 0xFF);
    instr.set_concat_axis(axis);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0025: CAST (0x67) / QUANTIZE (0x68) / DEQUANT (0x69). Layouts nos
// 26B congelados (ver RFC). Regra de precisão (§11) no path 32B: par
// explícito ou trap — nunca fallback silencioso.
// ---------------------------------------------------------------------------

impl Instruction {
    /// CAST: payload[0]=dst code (0=F32,1=F16,2=BF16,3=I8,4=U8).
    pub fn cast_dst(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_cast_dst(&mut self, dst: u8) {
        self.payload[0] = dst;
    }

    /// QUANTIZE: payload[0]=quant dtype discriminant (4=Q4_0, 8=Q8_0).
    pub fn quantize_type(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_quantize_type(&mut self, qtype: u8) {
        self.payload[0] = qtype;
    }
}

/// CAST rD, rT DST=... — rdest <- NOVO tensor convertido (src intacto).
pub fn instr_cast(rdest: u8, r_src: u8, dst: u8) -> Instruction {
    let mut instr = Instruction::new(OP_CAST, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_cast_dst(dst);
    instr
}

/// QUANTIZE rD, rT Q=... — rdest <- NOVO tensor em blocos.
pub fn instr_quantize(rdest: u8, r_src: u8, qtype: u8) -> Instruction {
    let mut instr = Instruction::new(OP_QUANTIZE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_quantize_type(qtype);
    instr
}

/// DEQUANT rD, rT — rdest <- NOVO tensor F32 (mesmo shape do src).
pub fn instr_dequant(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_DEQUANT, 0, rdest, r_src, 0xFF, 0xFF)
}

// ---------------------------------------------------------------------------
// RFC-0027: SORT (0x30) / TOPK (0x31) / ARGMAX (0x32) / REDUCE (0x33) /
// BROADCAST (0x34) / PAD (0x35) / TILE (0x36) / TRANSPOSE (0x37).
// ---------------------------------------------------------------------------

impl Instruction {
    /// SORT: payload[0]=axis (default 0), [1]=order (0=ASC,1=DESC).
    pub fn sort_params(&self) -> (u8, u8) {
        (self.payload[0], self.payload[1])
    }

    pub fn set_sort_params(&mut self, axis: u8, order: u8) {
        self.payload[0] = axis;
        self.payload[1] = order;
    }

    /// TOPK: payload[0]=axis, [1..3]=k u16 LE, [3]=largest, [4]=sorted.
    pub fn topk_params(&self) -> (u8, u16, u8, u8) {
        let k = u16::from_le_bytes([self.payload[1], self.payload[2]]);
        (self.payload[0], k, self.payload[3], self.payload[4])
    }

    pub fn set_topk_params(&mut self, axis: u8, k: u16, largest: u8, sorted: u8) {
        self.payload[0] = axis;
        self.payload[1..3].copy_from_slice(&k.to_le_bytes());
        self.payload[3] = largest;
        self.payload[4] = sorted;
    }

    /// ARGMAX: payload[0]=axis (default 0).
    pub fn argmax_axis(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_argmax_axis(&mut self, axis: u8) {
        self.payload[0] = axis;
    }

    /// REDUCE: payload[0]=op (0-4), [1]=axis (0xFF=total).
    pub fn reduce_params(&self) -> (u8, u8) {
        (self.payload[0], self.payload[1])
    }

    pub fn set_reduce_params(&mut self, op: u8, axis: u8) {
        self.payload[0] = op;
        self.payload[1] = axis;
    }

    /// BROADCAST: payload[0]=ndim (1-4), [1..17]=4×u32 dims (RESHAPE).
    pub fn broadcast_shape(&self) -> (u8, [u32; 4]) {
        reshape_dims_of(self)
    }

    pub fn set_broadcast_shape(&mut self, ndim: u8, dims: [u32; 4]) {
        set_reshape_dims(self, ndim, dims);
    }

    /// PAD: payload[0..4]=value f32 LE, [4]=axis, [5..9]=before u32,
    /// [9..13]=after u32.
    pub fn pad_params(&self) -> (f32, u8, u32, u32) {
        let mut bv = [0u8; 4];
        bv.copy_from_slice(&self.payload[0..4]);
        let mut bb = [0u8; 4];
        bb.copy_from_slice(&self.payload[5..9]);
        let mut ba = [0u8; 4];
        ba.copy_from_slice(&self.payload[9..13]);
        (
            f32::from_le_bytes(bv),
            self.payload[4],
            u32::from_le_bytes(bb),
            u32::from_le_bytes(ba),
        )
    }

    pub fn set_pad_params(&mut self, value: f32, axis: u8, before: u32, after: u32) {
        self.payload[0..4].copy_from_slice(&value.to_le_bytes());
        self.payload[4] = axis;
        self.payload[5..9].copy_from_slice(&before.to_le_bytes());
        self.payload[9..13].copy_from_slice(&after.to_le_bytes());
    }

    /// TILE: payload[0..4]=reps u32 LE, [4]=axis.
    pub fn tile_params(&self) -> (u32, u8) {
        let mut br = [0u8; 4];
        br.copy_from_slice(&self.payload[0..4]);
        (u32::from_le_bytes(br), self.payload[4])
    }

    pub fn set_tile_params(&mut self, reps: u32, axis: u8) {
        self.payload[0..4].copy_from_slice(&reps.to_le_bytes());
        self.payload[4] = axis;
    }

    /// TRANSPOSE: payload[0]=ndim (1-4), [1..5]=perm u8 ×4.
    pub fn transpose_perm(&self) -> (u8, [u8; 4]) {
        let mut perm = [0u8; 4];
        perm.copy_from_slice(&self.payload[1..5]);
        (self.payload[0], perm)
    }

    pub fn set_transpose_perm(&mut self, ndim: u8, perm: [u8; 4]) {
        self.payload[0] = ndim;
        self.payload[1..5].copy_from_slice(&perm);
    }
}

/// SORT rD, rT [AXIS=n] [ORDER=ASC|DESC].
pub fn instr_sort(rdest: u8, r_src: u8, axis: u8, order: u8) -> Instruction {
    let mut instr = Instruction::new(OP_SORT, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_sort_params(axis, order);
    instr
}

/// TOPK rD, rT K=k [AXIS=n] [largest] [sorted].
pub fn instr_topk(rdest: u8, r_src: u8, axis: u8, k: u16, largest: u8, sorted: u8) -> Instruction {
    let mut instr = Instruction::new(OP_TOPK, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_topk_params(axis, k, largest, sorted);
    instr
}

/// ARGMAX rD, rT [AXIS=n] — índices como f32.
pub fn instr_argmax(rdest: u8, r_src: u8, axis: u8) -> Instruction {
    let mut instr = Instruction::new(OP_ARGMAX, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_argmax_axis(axis);
    instr
}

/// REDUCE rD, rT OP=... [AXIS=n] (0xFF = total).
pub fn instr_reduce(rdest: u8, r_src: u8, op: u8, axis: u8) -> Instruction {
    let mut instr = Instruction::new(OP_REDUCE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_reduce_params(op, axis);
    instr
}

/// BROADCAST rD, rT SHAPE=... — expansão (cópia).
pub fn instr_broadcast(rdest: u8, r_src: u8, dims: &[u32]) -> Instruction {
    debug_assert!((1..=4).contains(&dims.len()));
    let mut arr = [0u32; 4];
    for (i, d) in dims.iter().take(4).enumerate() {
        arr[i] = *d;
    }
    let mut instr = Instruction::new(OP_BROADCAST, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_broadcast_shape(dims.len().min(255) as u8, arr);
    instr
}

/// PAD rD, rT VALUE=x [AXIS=n] [BEFORE=n] [AFTER=n].
pub fn instr_pad(rdest: u8, r_src: u8, value: f32, axis: u8, before: u32, after: u32) -> Instruction {
    let mut instr = Instruction::new(OP_PAD, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_pad_params(value, axis, before, after);
    instr
}

/// TILE rD, rT REPS=n [AXIS=n].
pub fn instr_tile(rdest: u8, r_src: u8, reps: u32, axis: u8) -> Instruction {
    let mut instr = Instruction::new(OP_TILE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_tile_params(reps, axis);
    instr
}

/// TRANSPOSE rD, rT perm — permuta N-D (validada no exec).
pub fn instr_transpose(rdest: u8, r_src: u8, perm: &[u8]) -> Instruction {
    debug_assert!((1..=4).contains(&perm.len()));
    let mut arr = [0u8; 4];
    for (i, p) in perm.iter().take(4).enumerate() {
        arr[i] = *p;
    }
    let mut instr = Instruction::new(OP_TRANSPOSE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_transpose_perm(perm.len().min(255) as u8, arr);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0028: SOFTMAX (0x3C) + ativações (0x3D-0x43). Semântica IEEE total
// em activations.rs; aqui só codificação.
// ---------------------------------------------------------------------------

impl Instruction {
    /// SOFTMAX: payload[0]=axis (0xFF=último), [1..5]=temp f32 LE.
    pub fn softmax_params(&self) -> (u8, f32) {
        let mut bt = [0u8; 4];
        bt.copy_from_slice(&self.payload[1..5]);
        (self.payload[0], f32::from_le_bytes(bt))
    }

    pub fn set_softmax_params(&mut self, axis: u8, temp: f32) {
        self.payload[0] = axis;
        self.payload[1..5].copy_from_slice(&temp.to_le_bytes());
    }

    /// CLIP: payload[0..4]=min f32 LE, [4..8]=max f32 LE (ambos exigidos).
    pub fn clip_params(&self) -> (f32, f32) {
        let mut bn = [0u8; 4];
        bn.copy_from_slice(&self.payload[0..4]);
        let mut bx = [0u8; 4];
        bx.copy_from_slice(&self.payload[4..8]);
        (f32::from_le_bytes(bn), f32::from_le_bytes(bx))
    }

    pub fn set_clip_params(&mut self, min: f32, max: f32) {
        self.payload[0..4].copy_from_slice(&min.to_le_bytes());
        self.payload[4..8].copy_from_slice(&max.to_le_bytes());
    }
}

/// SOFTMAX rD, rT [AXIS=n] [TEMP=x] (assembler escreve defaults).
pub fn instr_softmax(rdest: u8, r_src: u8, axis: u8, temp: f32) -> Instruction {
    let mut instr = Instruction::new(OP_SOFTMAX, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_softmax_params(axis, temp);
    instr
}

/// CLIP rD, rT MIN=x MAX=x.
pub fn instr_clip(rdest: u8, r_src: u8, min: f32, max: f32) -> Instruction {
    let mut instr = Instruction::new(OP_CLIP, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_clip_params(min, max);
    instr
}

/// GELU rD, rT (sem payload).
pub fn instr_gelu(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_GELU, 0, rdest, r_src, 0xFF, 0xFF)
}

/// SIGMOID rD, rT (sem payload).
pub fn instr_sigmoid(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_SIGMOID, 0, rdest, r_src, 0xFF, 0xFF)
}

/// TANH rD, rT (sem payload).
pub fn instr_tanh(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_TANH, 0, rdest, r_src, 0xFF, 0xFF)
}

/// RELU rD, rT (sem payload).
pub fn instr_relu(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_RELU, 0, rdest, r_src, 0xFF, 0xFF)
}

/// EXP rD, rT (sem payload).
pub fn instr_exp(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_EXP, 0, rdest, r_src, 0xFF, 0xFF)
}

/// LOG rD, rT (sem payload).
pub fn instr_log(rdest: u8, r_src: u8) -> Instruction {
    Instruction::new(OP_LOG, 0, rdest, r_src, 0xFF, 0xFF)
}

// ---------------------------------------------------------------------------
// RFC-0032: DEPFORMER (0x44). Tabela de pesos com 5 addrs u64 LE +
// 3 reservados; resto em payload (ver RFC).
// ---------------------------------------------------------------------------

impl Instruction {
    /// DEPFORMER: payload[0]=stream, [1]=layer, [2..4]=ncb u16 (0=16),
    /// [4..6]=nheads u16 (0=16), [6..8]=levels Q u16 (0=1024),
    /// [8..10]=context u16 (0=8), [10..14]=temp f32, [14..16]=topk u16
    /// (0=argmax determinístico).
    pub fn depformer_params(&self) -> (u8, u8, u16, u16, u16, u16, f32, u16) {
        let u16le = |r: std::ops::Range<usize>| {
            u16::from_le_bytes([self.payload[r.start], self.payload[r.start + 1]])
        };
        let mut bt = [0u8; 4];
        bt.copy_from_slice(&self.payload[10..14]);
        (
            self.payload[0],
            self.payload[1],
            u16le(2..4),
            u16le(4..6),
            u16le(6..8),
            u16le(8..10),
            f32::from_le_bytes(bt),
            u16le(14..16),
        )
    }

    pub fn set_depformer_params(
        &mut self, stream: u8, layer: u8, ncb: u16, nheads: u16,
        levels: u16, context: u16, temp: f32, topk: u16,
    ) {
        self.payload[0] = stream;
        self.payload[1] = layer;
        self.payload[2..4].copy_from_slice(&ncb.to_le_bytes());
        self.payload[4..6].copy_from_slice(&nheads.to_le_bytes());
        self.payload[6..8].copy_from_slice(&levels.to_le_bytes());
        self.payload[8..10].copy_from_slice(&context.to_le_bytes());
        self.payload[10..14].copy_from_slice(&temp.to_le_bytes());
        self.payload[14..16].copy_from_slice(&topk.to_le_bytes());
    }
}

/// DEPFORMER rD, rX, rW [...] — rdest <- pack [Y|codes] (fatiar c/ SLICE).
#[allow(clippy::too_many_arguments)]
pub fn instr_depformer(
    rdest: u8, r_x: u8, r_w: u8, stream: u8, layer: u8, ncb: u16,
    nheads: u16, levels: u16, context: u16, temp: f32, topk: u16,
) -> Instruction {
    let mut instr = Instruction::new(OP_DEPFORMER, 0, rdest, r_x, r_w, 0xFF);
    instr.set_depformer_params(stream, layer, ncb, nheads, levels, context, temp, topk);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0034: CALL (0x7D) / RET (0x7E). Alvo em imm_u128 (label resolvido);
// RET sem operandos.
// ---------------------------------------------------------------------------

/// CALL LABEL — alvo PC imediato (validado no exec como JUMP).
pub fn instr_call(target_pc: u128) -> Instruction {
    let mut instr = Instruction::new(OP_CALL, 0, 0xFF, 0xFF, 0xFF, 0xFF);
    instr.set_imm_u128(target_pc);
    instr
}

/// RET — sem operandos.
pub fn instr_ret() -> Instruction {
    Instruction::new(OP_RET, 0, 0xFF, 0xFF, 0xFF, 0xFF)
}

// ---------------------------------------------------------------------------
// RFC-0029: KV_COMPRESS (0x39) / FLASH_ATTN (0x3A) / ATTN_SPARSE (0x3B).
// ---------------------------------------------------------------------------

impl Instruction {
    /// KV_COMPRESS: payload[0..2]=sink u16 LE, [2..4]=window u16 LE,
    /// [4]=mode (0=SINK_WINDOW), [5..7]=stream u16 LE (0=só).
    pub fn kv_compress_params(&self) -> (u16, u16, u8, u16) {
        let sink = u16::from_le_bytes([self.payload[0], self.payload[1]]);
        let window = u16::from_le_bytes([self.payload[2], self.payload[3]]);
        let stream = u16::from_le_bytes([self.payload[5], self.payload[6]]);
        (sink, window, self.payload[4], stream)
    }

    pub fn set_kv_compress_params(&mut self, sink: u16, window: u16, mode: u8, stream: u16) {
        self.payload[0..2].copy_from_slice(&sink.to_le_bytes());
        self.payload[2..4].copy_from_slice(&window.to_le_bytes());
        self.payload[4] = mode;
        self.payload[5..7].copy_from_slice(&stream.to_le_bytes());
    }

    /// FLASH_ATTN: payload[0..2]=block_rows u16 LE (0=default 32).
    pub fn flash_block(&self) -> u16 {
        u16::from_le_bytes([self.payload[0], self.payload[1]])
    }

    pub fn set_flash_block(&mut self, block: u16) {
        self.payload[0..2].copy_from_slice(&block.to_le_bytes());
    }

    /// ATTN_SPARSE: payload[0]=metric (0=DOT), [1..3]=topk u16 LE
    /// (0=todos = caminho denso-equivalente).
    pub fn sparse_params(&self) -> (u8, u16) {
        let topk = u16::from_le_bytes([self.payload[1], self.payload[2]]);
        (self.payload[0], topk)
    }

    pub fn set_sparse_params(&mut self, metric: u8, topk: u16) {
        self.payload[0] = metric;
        self.payload[1..3].copy_from_slice(&topk.to_le_bytes());
    }
}

/// KV_COMPRESS SINK=n WINDOW=n [MODE=..] [STREAM=0] (sem registradores).
pub fn instr_kv_compress(sink: u16, window: u16, mode: u8, stream: u16) -> Instruction {
    let mut instr = Instruction::new(OP_KV_COMPRESS, 0, 0xFF, 0xFF, 0xFF, 0xFF);
    instr.set_kv_compress_params(sink, window, mode, stream);
    instr
}

/// FLASH_ATTN rD, rQ, rK, rV [BLOCK=n] (0=default 32).
pub fn instr_flash_attn(rdest: u8, r_q: u8, r_k: u8, r_v: u8, block: u16) -> Instruction {
    let mut instr = Instruction::new(OP_FLASH_ATTN, 0, rdest, r_q, r_k, r_v);
    instr.set_flash_block(block);
    instr
}

/// ATTN_SPARSE rD, rQ, rK, rV [TOPK=n] [METRIC=DOT].
pub fn instr_attn_sparse(rdest: u8, r_q: u8, r_k: u8, r_v: u8, metric: u8, topk: u16) -> Instruction {
    let mut instr = Instruction::new(OP_ATTN_SPARSE, 0, rdest, r_q, r_k, r_v);
    instr.set_sparse_params(metric, topk);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0031: STREAM_MERGE (0x45) / VAD_DETECT (0x46) / AUDIO_RESAMPLE (0x47)
// / AUDIO_FILTER (0x48) / AUDIO_WINDOW (0x49).
// ---------------------------------------------------------------------------

impl Instruction {
    /// VAD_DETECT: payload[0]=mode (0=ENERGY,1=ZCR,2=ML-veta).
    pub fn vad_mode(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_vad_mode(&mut self, mode: u8) {
        self.payload[0] = mode;
    }

    /// STREAM_MERGE: payload[0..4]=gain f32 LE, [4..6]=n_streams u16 LE
    /// (0=default 2), [6]=mode (reservado 0).
    pub fn merge_params(&self) -> (f32, u16, u8) {
        let mut bg = [0u8; 4];
        bg.copy_from_slice(&self.payload[0..4]);
        let n = u16::from_le_bytes([self.payload[4], self.payload[5]]);
        (f32::from_le_bytes(bg), n, self.payload[6])
    }

    pub fn set_merge_params(&mut self, gain: f32, n_streams: u16, mode: u8) {
        self.payload[0..4].copy_from_slice(&gain.to_le_bytes());
        self.payload[4..6].copy_from_slice(&n_streams.to_le_bytes());
        self.payload[6] = mode;
    }

    /// AUDIO_RESAMPLE: payload[0..4]=src_rate u32 LE, [4..8]=dst_rate u32.
    pub fn resample_rates(&self) -> (u32, u32) {
        let mut bs = [0u8; 4];
        bs.copy_from_slice(&self.payload[0..4]);
        let mut bd = [0u8; 4];
        bd.copy_from_slice(&self.payload[4..8]);
        (u32::from_le_bytes(bs), u32::from_le_bytes(bd))
    }

    pub fn set_resample_rates(&mut self, src_rate: u32, dst_rate: u32) {
        self.payload[0..4].copy_from_slice(&src_rate.to_le_bytes());
        self.payload[4..8].copy_from_slice(&dst_rate.to_le_bytes());
    }

    /// AUDIO_FILTER: payload[0]=mode (0=FIR,1=IIR-veta).
    pub fn filter_mode(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_filter_mode(&mut self, mode: u8) {
        self.payload[0] = mode;
    }

    /// AUDIO_WINDOW: payload[0]=type (0=HANN,1=HAMMING).
    pub fn window_type(&self) -> u8 {
        self.payload[0]
    }

    pub fn set_window_type(&mut self, wtype: u8) {
        self.payload[0] = wtype;
    }
}

/// VAD_DETECT rD, rT [MODE=ENERGY|ZCR] — rdest <- score [1,1].
pub fn instr_vad_detect(rdest: u8, r_src: u8, mode: u8) -> Instruction {
    let mut instr = Instruction::new(OP_VAD_DETECT, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_vad_mode(mode);
    instr
}

/// STREAM_MERGE rD, rA, rB GAIN=x [N_STREAMS=n].
pub fn instr_stream_merge(rdest: u8, r_a: u8, r_b: u8, gain: f32, n_streams: u16) -> Instruction {
    let mut instr = Instruction::new(OP_STREAM_MERGE, 0, rdest, r_a, r_b, 0xFF);
    instr.set_merge_params(gain, n_streams, 0);
    instr
}

/// AUDIO_RESAMPLE rD, rT SRC=n DST=n.
pub fn instr_audio_resample(rdest: u8, r_src: u8, src_rate: u32, dst_rate: u32) -> Instruction {
    let mut instr = Instruction::new(OP_AUDIO_RESAMPLE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_resample_rates(src_rate, dst_rate);
    instr
}

/// AUDIO_FILTER rD, rX, rB [MODE=FIR].
pub fn instr_audio_filter(rdest: u8, r_x: u8, r_b: u8, mode: u8) -> Instruction {
    let mut instr = Instruction::new(OP_AUDIO_FILTER, 0, rdest, r_x, r_b, 0xFF);
    instr.set_filter_mode(mode);
    instr
}

/// AUDIO_WINDOW rD, rT [TYPE=HANN|HAMMING].
pub fn instr_audio_window(rdest: u8, r_src: u8, wtype: u8) -> Instruction {
    let mut instr = Instruction::new(OP_AUDIO_WINDOW, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_window_type(wtype);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0026: ADD_IMM (0x7A) / SUB_IMM (0x7B) / STEPS (0x7C). Imediato u128
// na convenção imm_u128 (payload[0..16]); STEPS sem payload.
// ---------------------------------------------------------------------------

/// ADD_IMM rD, rS, IMM=n — rdest <- rsrc wrapping_add imm.
pub fn instr_add_imm(rdest: u8, r_src: u8, imm: u128) -> Instruction {
    let mut instr = Instruction::new(OP_ADD_IMM, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_imm_u128(imm);
    instr
}

/// SUB_IMM rD, rS, IMM=n — rdest <- rsrc wrapping_sub imm.
pub fn instr_sub_imm(rdest: u8, r_src: u8, imm: u128) -> Instruction {
    let mut instr = Instruction::new(OP_SUB_IMM, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_imm_u128(imm);
    instr
}

/// STEPS rD — rdest <- contador de retiradas (u64).
pub fn instr_steps(rdest: u8) -> Instruction {
    Instruction::new(OP_STEPS, 0, rdest, 0xFF, 0xFF, 0xFF)
}

// ---------------------------------------------------------------------------
// RFC-0013: DENOISE_STEP (0x21). payload[0..4]=alpha_bar_t f32,
// [4..8]=beta_t f32, [8..12]=sigma_t f32, [12..16]=timestep u32.
// ---------------------------------------------------------------------------

impl Instruction {
    pub fn denoise_params(&self) -> (f32, f32, f32, u32) {
        let mut ba = [0u8; 4];
        ba.copy_from_slice(&self.payload[0..4]);
        let mut bb = [0u8; 4];
        bb.copy_from_slice(&self.payload[4..8]);
        let mut bs = [0u8; 4];
        bs.copy_from_slice(&self.payload[8..12]);
        let mut bt = [0u8; 4];
        bt.copy_from_slice(&self.payload[12..16]);
        (
            f32::from_le_bytes(ba),
            f32::from_le_bytes(bb),
            f32::from_le_bytes(bs),
            u32::from_le_bytes(bt),
        )
    }

    pub fn set_denoise_params(&mut self, alpha_bar: f32, beta: f32, sigma: f32, timestep: u32) {
        self.payload[0..4].copy_from_slice(&alpha_bar.to_le_bytes());
        self.payload[4..8].copy_from_slice(&beta.to_le_bytes());
        self.payload[8..12].copy_from_slice(&sigma.to_le_bytes());
        self.payload[12..16].copy_from_slice(&timestep.to_le_bytes());
    }
}

/// DENOISE_STEP rD, rX, rE [, rS] [ALPHA=a] [BETA=b] [SIGMA=s] [T=t].
/// rS aceito no parse e vetado no exec salvo 0xFF (schedule futuro).
pub fn instr_denoise_step(rdest: u8, r_x: u8, r_eps: u8, r_sched: u8, alpha_bar: f32, beta: f32, sigma: f32, timestep: u32) -> Instruction {
    let mut instr = Instruction::new(OP_DENOISE_STEP, 0, rdest, r_x, r_eps, r_sched);
    instr.set_denoise_params(alpha_bar, beta, sigma, timestep);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0014: ODE_STEP (0x25). payload[0..4]=dt f32, [4]=method u8,
// [5]=layer_id u8 (carregado, reservado p/ namespacing futuro).
// ---------------------------------------------------------------------------

impl Instruction {
    pub fn ode_params(&self) -> (f32, u8, u8) {
        let mut b = [0u8; 4];
        b.copy_from_slice(&self.payload[0..4]);
        (f32::from_le_bytes(b), self.payload[4], self.payload[5])
    }

    pub fn set_ode_params(&mut self, dt: f32, method: u8, layer_id: u8) {
        self.payload[0..4].copy_from_slice(&dt.to_le_bytes());
        self.payload[4] = method;
        self.payload[5] = layer_id;
    }
}

/// ODE_STEP rD, rX, rU [, rWb] [DT=t] [METHOD=...] [LAYER=n].
/// rU/rWb ausentes => 0xFF (sem controle / campo default contrativo).
pub fn instr_ode_step(rdest: u8, r_x: u8, r_u: u8, r_wb: u8, dt: f32, method: u8, layer_id: u8) -> Instruction {
    let mut instr = Instruction::new(OP_ODE_STEP, 0, rdest, r_x, r_u, r_wb);
    instr.set_ode_params(dt, method, layer_id);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0015: SPIKE_STEP (0x20). payload[0..4]=V_threshold f32,
// [4..8]=decay f32, [8..12]=V_reset f32, [12]=layer_id u8,
// [13]=refractory_steps u8.
// ---------------------------------------------------------------------------

impl Instruction {
    pub fn spike_params(&self) -> (f32, f32, f32, u8, u8) {
        let mut bt = [0u8; 4];
        bt.copy_from_slice(&self.payload[0..4]);
        let mut bd = [0u8; 4];
        bd.copy_from_slice(&self.payload[4..8]);
        let mut br = [0u8; 4];
        br.copy_from_slice(&self.payload[8..12]);
        (
            f32::from_le_bytes(bt),
            f32::from_le_bytes(bd),
            f32::from_le_bytes(br),
            self.payload[12],
            self.payload[13],
        )
    }

    pub fn set_spike_params(&mut self, thresh: f32, decay: f32, reset: f32, layer_id: u8, refr: u8) {
        self.payload[0..4].copy_from_slice(&thresh.to_le_bytes());
        self.payload[4..8].copy_from_slice(&decay.to_le_bytes());
        self.payload[8..12].copy_from_slice(&reset.to_le_bytes());
        self.payload[12] = layer_id;
        self.payload[13] = refr;
    }
}

/// SPIKE_STEP rD, rV, rI [, rPack] [THRESH=t] [DECAY=d] [RESET=r]
/// [LAYER=n] [REFRACT=k]. Pack [thresh,decay,reset,refr] vence payload.
pub fn instr_spike_step(
    rdest: u8, r_v: u8, r_i: u8, r_pack: u8,
    thresh: f32, decay: f32, reset: f32, layer_id: u8, refr: u8,
) -> Instruction {
    let mut instr = Instruction::new(OP_SPIKE_STEP, 0, rdest, r_v, r_i, r_pack);
    instr.set_spike_params(thresh, decay, reset, layer_id, refr);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0017: CONV (0x1E). payload[0..2]=stride u16, [2..4]=pad u16,
// [4..6]=dilation u16, [6]=groups u8 (0=1), [7]=fused_act u8.
// ---------------------------------------------------------------------------

impl Instruction {
    pub fn conv_params(&self) -> (u16, u16, u16, u8, u8) {
        let s = u16::from_le_bytes([self.payload[0], self.payload[1]]);
        let p = u16::from_le_bytes([self.payload[2], self.payload[3]]);
        let d = u16::from_le_bytes([self.payload[4], self.payload[5]]);
        let g = self.payload[6];
        (s, p, d, if g == 0 { 1 } else { g }, self.payload[7])
    }

    pub fn set_conv_params(&mut self, stride: u16, pad: u16, dilation: u16, groups: u8, act: u8) {
        self.payload[0..2].copy_from_slice(&stride.to_le_bytes());
        self.payload[2..4].copy_from_slice(&pad.to_le_bytes());
        self.payload[4..6].copy_from_slice(&dilation.to_le_bytes());
        self.payload[6] = groups;
        self.payload[7] = act;
    }
}

/// CONV rD, rX, rW [, rB] [STRIDE=n] [PAD=n] [DILATION=n] [GROUPS=n]
/// [ACT=NONE|SILU|RELU]. rB ausente => 0xFF (sem bias).
pub fn instr_conv(
    rdest: u8, r_x: u8, r_w: u8, r_b: u8,
    stride: u16, pad: u16, dilation: u16, groups: u8, act: u8,
) -> Instruction {
    let mut instr = Instruction::new(OP_CONV, 0, rdest, r_x, r_w, r_b);
    instr.set_conv_params(stride, pad, dilation, groups, act);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0012: FOREST (0x22). payload[0..2]=n_trees u16, [2..4]=max_depth u16,
// [4]=mode (0=VOTE/valores por árvore, 1=MEAN/média).
// Tabela: [n_trees*stride, 4] f32, stride = 2^max_depth - 1;
// linha = [feat_idx, thresh, left, right]; folha = left < 0.
// Folhas: [n_trees*stride] f32 alinhado slot a slot.
// ---------------------------------------------------------------------------

impl Instruction {
    pub fn forest_params(&self) -> (u16, u16, u8) {
        let n = u16::from_le_bytes([self.payload[0], self.payload[1]]);
        let d = u16::from_le_bytes([self.payload[2], self.payload[3]]);
        (n, d, self.payload[4])
    }

    pub fn set_forest_params(&mut self, n_trees: u16, max_depth: u16, mode: u8) {
        self.payload[0..2].copy_from_slice(&n_trees.to_le_bytes());
        self.payload[2..4].copy_from_slice(&max_depth.to_le_bytes());
        self.payload[4] = mode;
    }
}

/// FOREST rD, rF, rT, rL [TREES=n] [DEPTH=d] [MODE=VOTE|MEAN]
pub fn instr_forest(rdest: u8, r_feat: u8, r_table: u8, r_leaves: u8, n_trees: u16, max_depth: u16, mode: u8) -> Instruction {
    let mut instr = Instruction::new(OP_FOREST, 0, rdest, r_feat, r_table, r_leaves);
    instr.set_forest_params(n_trees, max_depth, mode);
    instr
}

// ---------------------------------------------------------------------------
// RFC-0038: retrieval (turno 1: ADD/DEL). Sem payload próprio (dim/id
// vivem no IndexStore / regs); rId ausente = 0xFF = sequencial.
// ---------------------------------------------------------------------------

/// RAG_INDEX_ADD rD, rDb, rVec [, rId] — rDb=0 cria (rD <- id do store).
pub fn instr_rag_index_add(rdest: u8, r_db: u8, r_vec: u8, r_id: u8) -> Instruction {
    Instruction::new(OP_RAG_INDEX_ADD, 0, rdest, r_db, r_vec, r_id)
}

/// RAG_INDEX_DEL rD, rDb, rId — rD <- contagem restante.
pub fn instr_rag_index_del(rdest: u8, r_db: u8, r_id: u8) -> Instruction {
    Instruction::new(OP_RAG_INDEX_DEL, 0, rdest, r_db, r_id, 0xFF)
}

/// Detecta uso da faixa V-2 (`0x50-0x53/0x56-0x57`) p/ o bit REQUIRED
/// do container (RFC-0038; valor do bit em `m3bc.rs`, sem ciclo).
pub fn uses_v2_retrieval(prog: &[Instruction]) -> bool {
    prog.iter().any(|i| matches!(i.opcode, 0x50..=0x53 | 0x56..=0x57))
}

// ---------------------------------------------------------------------------
// Assembler textual simples (.m3asm)
// ---------------------------------------------------------------------------

/// Parseia um arquivo .m3asm textual para Vec<Instruction>.
///
/// Sintaxe suportada (case-insensitive, comentários `;` ou `#`):
/// ```text
/// ; exemplo
/// TENSOR r0, r1, r2, 2x2 f32    # r0 = alloc 2x2 f32 (payload rows/cols/dtype)
/// TENSOR r3 2 2 f32            # forma alternativa
/// ATTN r3, r0, r1, r2          # r3 = attn(Q=r0,K=r1,V=r2)
/// NORM r2, r0, r1, r3          # r2 = rmsnorm(r0, gamma=r1, beta=r3)
/// FFN r3, r0, r1, r2           # r3 = ffn(x=r0,W1=r1,W2=r2)
/// FFN r3, r0, r1, r2, r4, r5   # com bias: r4=b1, r5=b2 (payload)
/// STREAM r3, r4 BLOCKING       # ou DROP
/// FORK r5, RED                 # RED | BLUE | GREEN
/// ABORT r1, r2
/// SENSE r0, AUDIO              # ou VAD / 0 / 1
/// HALT
/// NOP
/// ```
pub fn assemble(text: &str) -> Result<Vec<Instruction>> {
    let full = assemble_full(text, 0x1000, DATA_LOAD_BASE)?;
    if !full.data.blobs.is_empty() {
        return Err(anyhow!(
            "programa tem seção `.data` ({} blob(s)) — use `assemble_with_data` (V-1b; loader dia 3)",
            full.data.blobs.len()
        ));
    }
    Ok(full.instrs)
}

/// V-1b (dia 1): monta código + seção `.data`, sem exigir loader.
/// `instrs` valem para `load_program` hoje; `data` é sidecar para o
/// preload do dia 3 (endereços ainda sem binding — semântica de
/// execução inalterada).
pub fn assemble_with_data(text: &str) -> Result<AssembledProgram> {
    assemble_full(text, 0x1000, DATA_LOAD_BASE)
}

/// Variante dia 3b: base de carga explícita (harness que aloca tabelas
/// antes do preload; o valor viaja em `AssembledProgram.data_base` e o
/// loader verifica — sem relocation, sem lookup).
pub fn assemble_with_data_base(text: &str, data_base: u128) -> Result<AssembledProgram> {
    assemble_full(text, 0x1000, data_base)
}

/// RFC-0008 (modo estrito): rejeita tokens extras desconhecidos.
/// `known`: palavras exatas ("NOTIFY") ou prefixos KV ("AXIS=").
/// Chamado com o restante NÃO consumido posicionalmente por cada arm.
fn reject_unknown(op: &str, rest: &[&str], known: &[&str]) -> Result<()> {
    for p in rest {
        let up = p.to_ascii_uppercase();
        let ok = known.iter().any(|k| {
            if k.ends_with('=') {
                up.starts_with(k)
            } else {
                up == *k
            }
        });
        if !ok {
            return Err(anyhow!("{}: token desconhecido '{}' (modo estrito, RFC-0008)", op, p));
        }
    }
    Ok(())
}

pub fn assemble_with_base(text: &str, program_base: u128) -> Result<Vec<Instruction>> {
    let full = assemble_full(text, program_base, DATA_LOAD_BASE)?;
    if !full.data.blobs.is_empty() {
        return Err(anyhow!(
            "programa tem seção `.data` ({} blob(s)) — use `assemble_with_data` (V-1b; loader dia 3)",
            full.data.blobs.len()
        ));
    }
    Ok(full.instrs)
}

/// Núcleo comum: duas seções (`.text` código, `.data` blobs), tabela
/// de símbolos única por chamada (`.reg` + `.equ` valem no arquivo
/// todo, independente de seção). `program_base` = PCs do código;
/// `data_base` = cursor GLOBAL assumido p/ `@` (loader verifica).
fn assemble_full(text: &str, program_base: u128, data_base: u128) -> Result<AssembledProgram> {
    let mut labels: HashMap<String, u128> = HashMap::new();
    let mut syms = SymbolTable::new();
    let mut instr_lines: Vec<(usize, String)> = Vec::new();
    let mut data = DataSection::default();
    // Linhas `.data` cruas: parseadas SÓ após todas as diretivas (`.reg`/
    // `.equ` valem no arquivo todo, como no Passo 2 do código).
    let mut data_lines: Vec<(usize, String)> = Vec::new();
    let mut in_data = false;

    // Passo 1: Varredura de rótulos, diretivas `.reg` e mapeamento
    for (lineno, raw) in text.lines().enumerate() {
        let mut line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }

        // Diretiva `.reg <nome> <rN>` (RFC-0036): vincula apelido
        // simbólico a registrador físico. Não emite instrução.
        // Qualquer outra linha com ponto inicial cai no erro padrão
        // de opcode desconhecido (nada afrouxado).
        {
            let words: Vec<&str> = line.split_whitespace().collect();
            if !words.is_empty() && words[0].eq_ignore_ascii_case(".reg") {
                if words.len() != 3 {
                    return Err(anyhow!(
                        "linha {}: `.reg` precisa de nome e registrador — ex: `.reg rTranscript r0` — '{}'",
                        lineno + 1,
                        line
                    ));
                }
                let name = words[1].trim_end_matches(',').to_lowercase();
                let name = name.strip_prefix('r').ok_or_else(|| {
                    anyhow!("linha {}: nome simbólico deve começar com 'r' — '{}'", lineno + 1, line)
                })?;
                if name.is_empty()
                    || !name.bytes().next().map_or(false, |b| b.is_ascii_alphabetic() || b == b'_')
                    || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
                {
                    return Err(anyhow!("linha {}: nome simbólico inválido '{}'", lineno + 1, words[1]));
                }
                let phys = words[2].trim_end_matches(',').trim_start_matches('r');
                let reg: u8 = phys.parse().map_err(|_| {
                    anyhow!("linha {}: registrador físico inválido '{}' (use r0..r15)", lineno + 1, words[2])
                })?;
                syms.declare(name, reg).map_err(|e| anyhow!("linha {}: {}", lineno + 1, e))?;
                continue;
            }
        }

        // Diretivas V-1a (RFC-0037): `.equ NAME valor` (constante inteira
        // p/ imediatos) e `.text` (marcador de seção, no-op — convenção
        // para a V-1b). `.data`/`.str` erram explícito apontando a V-1b
        // (antes caíam em "opcode desconhecido"; continuam errando).
        {
            let words: Vec<&str> = line.split_whitespace().collect();
            if !words.is_empty() {
                if words[0].eq_ignore_ascii_case(".equ") {
                    if words.len() != 3 {
                        return Err(anyhow!(
                            "linha {}: `.equ` precisa de nome e valor — ex: `.equ RAG_TOPK 5` — '{}'",
                            lineno + 1,
                            line
                        ));
                    }
                    let name = words[1].trim_end_matches(',').to_ascii_uppercase();
                    let ok_ident = !name.is_empty()
                        && name.bytes().next().map_or(false, |b| b.is_ascii_alphabetic() || b == b'_')
                        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
                    if !ok_ident {
                        return Err(anyhow!("linha {}: nome de constante inválido '{}'", lineno + 1, words[1]));
                    }
                    if ["F32", "F16", "BF16", "I8", "U8", "SPARSE", "EOS_TOKEN"].contains(&name.as_str()) {
                        return Err(anyhow!(
                            "linha {}: nome '{}' reservado (dtype/keyword) — '{}'",
                            lineno + 1,
                            words[1],
                            line
                        ));
                    }
                    if name.len() > 1 && name.starts_with('R') && name[1..].bytes().all(|b| b.is_ascii_digit()) {
                        return Err(anyhow!(
                            "linha {}: nome '{}' colide com registrador — '{}'",
                            lineno + 1,
                            words[1],
                            line
                        ));
                    }
                    let vs = words[2].trim_end_matches(',');
                    let val = if let Some(hex) = vs.strip_prefix("0x").or_else(|| vs.strip_prefix("0X")) {
                        u128::from_str_radix(hex, 16).map_err(|_| {
                            anyhow!("linha {}: valor `.equ` hex inválido '{}'", lineno + 1, words[2])
                        })?
                    } else if vs.starts_with('-') || vs.starts_with('+') {
                        return Err(anyhow!(
                            "linha {}: valor `.equ` '{}' inválido (u128 sem sinal)",
                            lineno + 1,
                            words[2]
                        ));
                    } else {
                        vs.parse::<u128>().map_err(|_| {
                            anyhow!("linha {}: valor `.equ` '{}' inválido (u128 decimal ou 0x-hex)", lineno + 1, words[2])
                        })?
                    };
                    syms.declare_const(&name, val).map_err(|e| anyhow!("linha {}: {}", lineno + 1, e))?;
                    continue;
                }
                if words[0].eq_ignore_ascii_case(".text") {
                    if words.len() != 1 {
                        return Err(anyhow!("linha {}: `.text` não recebe operandos — '{}'", lineno + 1, line));
                    }
                    in_data = false;
                    continue;
                }
                // V-1b (dia 1): `.data` abre a seção de blobs (tipos
                // escalares + `.str`; multi-valor é dia 2). `.str` sozinho
                // continua erro: é tipo de blob, não diretiva.
                if words[0].eq_ignore_ascii_case(".data") {
                    if words.len() != 1 {
                        return Err(anyhow!("linha {}: `.data` não recebe operandos — '{}'", lineno + 1, line));
                    }
                    in_data = true;
                    continue;
                }
                if words[0].eq_ignore_ascii_case(".str") {
                    return Err(anyhow!(
                        "linha {}: `.str` é tipo de blob dentro de `.data` (ex: `filler_01: .str \"...\"`) — '{}'",
                        lineno + 1,
                        line
                    ));
                }
            }
        }

        // Seção `.data`: só `nome: .tipo valor` (dia 1: escalar/`.str`).
        // Instrução ou rótulo de código aqui é erro alto. O parse é
        // adiado p/ o passo 1b (diretivas valem no arquivo todo).
        if in_data {
            data_lines.push((lineno + 1, line.to_string()));
            continue;
        }

        // Verifica se há rótulo no início da linha (ex: "MAIN_LOOP:" ou "MAIN_LOOP: SENSE ...")
        if let Some(colon_idx) = line.find(':') {
            let label_cand = line[..colon_idx].trim();
            if !label_cand.is_empty() && !label_cand.contains(' ') {
                let target_pc = program_base + (instr_lines.len() as u128) * (INSTR_SIZE as u128);
                labels.insert(label_cand.to_uppercase(), target_pc);
                line = line[colon_idx + 1..].trim();
            }
        }

        if !line.is_empty() {
            instr_lines.push((lineno + 1, line.to_string()));
        }
    }

    // Passo 1b: blobs `.data` (diretivas já todas coletadas acima).
    for (lineno, line) in &data_lines {
        let blob = parse_data_line(line, &syms)
            .map_err(|e| anyhow!("linha {}: {} — '{}'", lineno, e, line))?;
        if labels.contains_key(blob.name.to_ascii_uppercase().as_str()) {
            return Err(anyhow!(
                "linha {}: blob '{}' colide com rótulo de código — '{}'",
                lineno,
                blob.name,
                line
            ));
        }
        // Dia 3: namespace `nome` compartilhado `.equ`×`.data` —
        // colisão = erro (evita confusão valor-vs-endereço; endereço
        // explícito é `@nome`).
        if syms.is_const(&blob.name) {
            return Err(anyhow!(
                "linha {}: blob '{}' colide com constante `.equ` (use outro nome; endereço é `@{}`) — '{}'",
                lineno,
                blob.name,
                blob.name,
                line
            ));
        }
        data.declare(blob).map_err(|e| anyhow!("linha {}: {}", lineno, e))?;
    }
    // Dia 3: layout fixo + bind de `@nome` (antes do Passo 2, que resolve
    // os LOADI de endereço). Base explícita (dia 3b).
    syms.bind_addrs(&data_layout_addrs_at(&data, data_base));

    // Passo 2: Montagem com resolução de rótulos (+ símbolos do Passo 1)
    let mut out = Vec::with_capacity(instr_lines.len());
    for (lineno, line) in instr_lines {
        let instr = parse_line(&line, &labels, &mut syms).map_err(|e| anyhow!("linha {}: {} — '{}'", lineno, e, line))?;
        out.push(instr);
    }
    Ok(AssembledProgram { instrs: out, data, data_base })
}

fn strip_comment(s: &str) -> &str {
    // corta em primeiro ';' ou '#'
    let mut end = s.len();
    for (i, ch) in s.char_indices() {
        if ch == ';' || ch == '#' {
            end = i;
            break;
        }
    }
    &s[..end]
}

/// Tabela de símbolos do assembler (RFC-0036, passo V-1 do PLANO_VISAO;
/// constantes RFC-0037). Apelidos simbólicos (`rTranscript`) vinculados
/// EXPLICITAMENTE via diretiva `.reg` a um dos 16 GPRs normativos (R5:
/// sem R255, sem faixas, sem container). Uso sem declaração continua
/// erro — a auto-alocação foi deliberadamente rejeitada para preservar
/// o gate do RFC-0008 (typo em posição de registrador deve falhar, ex.
/// `SANITY_CHECK r4, r0, FOO`). Constantes `.equ` (inteiras u128,
/// namespace separado, chave em maiúsculas) valem em posições de
/// imediato (LOADI/COMPARE/ADD_IMM/SUB_IMM/dims TENSOR/TREES/DEPTH/
/// START/LEN/CODE); fora delas, literais como antes. Determinístico;
/// tabela nova por chamada.
#[derive(Debug, Default)]
struct SymbolTable {
    map: HashMap<String, u8>,
    used: [bool; 16],
    consts: HashMap<String, u128>,
    /// Endereços `.data` (dia 3): nome (minúsculas) -> endereço load-time.
    /// Preenchido após o passo 1b (layout precisa de todos os blobs).
    addrs: HashMap<String, u128>,
}

impl SymbolTable {
    fn new() -> Self {
        Self::default()
    }

    /// Vincula `name` (já em minúsculas, com o `r` inicial removido) ao
    /// registrador físico `reg`. Nome repetido ou físico ocupado: erro.
    fn declare(&mut self, name: &str, reg: u8) -> Result<()> {
        if self.map.contains_key(name) {
            return Err(anyhow!("símbolo '{}' redeclarado (uma definição por arquivo)", name));
        }
        if reg >= 16 {
            return Err(anyhow!("registrador físico r{} fora de 0..15", reg));
        }
        if self.used[reg as usize] {
            return Err(anyhow!("registrador físico r{} já vinculado (sem alias implícito)", reg));
        }
        self.used[reg as usize] = true;
        self.map.insert(name.to_string(), reg);
        Ok(())
    }

    /// Resolve nome declarado. Desconhecido => erro que sugere `.reg`
    /// (nunca aloca sozinho: RFC-0008 acima de conveniência).
    fn resolve(&self, name: &str) -> Result<u8> {
        self.map.get(name).copied().ok_or_else(|| {
            anyhow!(
                "símbolo '{}' não declarado (declare com `.reg {} rN`)",
                name,
                name
            )
        })
    }

    /// Vincula `name` (já em maiúsculas) ao imediato `val` (RFC-0037).
    /// Nome repetido: erro. Não emite instrução.
    fn declare_const(&mut self, name: &str, val: u128) -> Result<()> {
        if self.consts.contains_key(name) {
            return Err(anyhow!("constante '{}' redeclarada (uma definição por arquivo)", name));
        }
        self.consts.insert(name.to_string(), val);
        Ok(())
    }

    /// Resolve constante `.equ` (lookup case-insensitive). Desconhecida
    /// => erro que sugere `.reg`/`.equ` conforme o caso de uso.
    fn resolve_const(&self, name: &str) -> Result<u128> {
        let key = name.trim().trim_end_matches(',').to_ascii_uppercase();
        self.consts.get(&key).copied().ok_or_else(|| {
            anyhow!("constante '{}' não declarada (declare com `.equ {} <valor>`, ou use literal)", name, name)
        })
    }

    /// Sonda sem errar (detecção de forma: dims TENSOR).
    fn is_const(&self, name: &str) -> bool {
        let key = name.trim().trim_end_matches(',').to_ascii_uppercase();
        self.consts.contains_key(&key)
    }

    /// Vincula endereços `.data` (dia 3, após o passo 1b).
    fn bind_addrs(&mut self, addrs: &[(String, u128)]) {
        for (name, addr) in addrs {
            self.addrs.insert(name.clone(), *addr);
        }
    }

    /// Resolve `@nome` (dia 3): SÓ blobs `.data`. `@` de `.equ`/nome
    /// inexistente => erro alto (namespace de endereço separado).
    fn resolve_addr(&self, name: &str) -> Result<u128> {
        let key = name.trim().trim_end_matches(',').to_ascii_lowercase();
        self.addrs.get(&key).copied().ok_or_else(|| {
            anyhow!("endereço '@{}' não declarado (só blobs `.data` têm endereço; `.equ` usa referência sem `@`)", name)
        })
    }
}

fn parse_reg(tok: &str, syms: &SymbolTable) -> Result<u8> {
    let t = tok.trim().trim_end_matches(',').to_lowercase();
    if t == "_" || t == "x" || t == "-" || t == "ff" {
        return Ok(0xFF);
    }
    let t = t.trim_start_matches('r');
    // Caminho legado: numérico (quirks preservados, ex. "rr1").
    if let Ok(n) = t.parse::<u8>() {
        if n >= 16 {
            return Err(anyhow!("registrador r{} fora de 0..15", n));
        }
        return Ok(n);
    }
    // Novo (RFC-0036): simbólico r+identificador previamente declarado.
    // Formas inválidas ("r1x", "r", "rfoo-bar") erram como antes.
    let is_ident = !t.is_empty()
        && t.bytes().next().map_or(false, |b| b.is_ascii_alphabetic() || b == b'_')
        && t.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if is_ident {
        return syms.resolve(t);
    }
    Err(anyhow!("registrador inválido '{}'", tok))
}

/// Imediato u128 (RFC-0037): literal decimal/0x-hex (sem sinal) ou
/// constante `.equ`. Literal primeiro (legado intacto); desconhecido
/// => erro que sugere `.equ`. Sítios com mensagem própria (LOADI,
/// COMPARE, ADD_IMM, ...) a preservam — este helper é o núcleo.
fn parse_imm_u128(tok: &str, syms: &SymbolTable) -> Result<u128> {
    let t = tok.trim().trim_end_matches(',');
    if let Some(hex) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        return u128::from_str_radix(hex, 16).map_err(|_| anyhow!("imediato hex inválido '{}'", tok));
    }
    if t.starts_with('-') || t.starts_with('+') {
        return Err(anyhow!("imediato '{}' inválido (u128: decimal ou 0x-hex, sem sinal)", tok));
    }
    if let Ok(n) = t.parse::<u128>() {
        return Ok(n);
    }
    syms.resolve_const(t)
}

/// Variante decimal-ou-const (sem hex): preserva o domínio exato dos
/// sítios que hoje só aceitam decimal (COMPARE/ADD_IMM/SUB_IMM e os
/// KVs numéricos abaixo) — `.equ` soma-se, nada se afrouxa.
fn parse_imm_dec_or_const(tok: &str, syms: &SymbolTable) -> Result<u128> {
    let t = tok.trim().trim_end_matches(',');
    if let Ok(n) = t.parse::<u128>() {
        return Ok(n);
    }
    syms.resolve_const(t)
}

/// Dimensão TENSOR (u64): literal decimal ou `.equ` (com checagem u64).
fn parse_dim_u64(tok: &str, syms: &SymbolTable) -> Result<u64> {
    if let Ok(n) = tok.trim().trim_end_matches(',').parse::<u64>() {
        return Ok(n);
    }
    let c = syms.resolve_const(tok)?;
    u64::try_from(c).map_err(|_| anyhow!("dimensão '{}' fora da faixa u64", tok))
}

// ---------------------------------------------------------------------------
// V-1b (dia 1): seção `.data` — sidecar de blobs (load-time, sem opcode).
// Tipos do dia 1: `.u32`/`.i32`/`.f32` escalares + `.str` UTF-8.
// Multi-valor (`[...]`) é dia 2 e erra explícito. O loader (dia 3) dá
// binding aos endereços; até lá, `data` não afeta execução.
// ---------------------------------------------------------------------------

/// Tipo de um blob `.data` (dia 1: escalar ou string).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataDtype {
    U32,
    I32,
    F32,
    Str,
}

/// Um blob nomeado da seção `.data`: nome (minúsculas), tipo, shape e
/// bytes LE (dia 1: `shape` vazio = escalar/`.str`; dia 2: `.f32` com
/// shape explícito `[N]`/`[R,C]` e valores em little-endian).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataBlob {
    pub name: String,
    pub dtype: DataDtype,
    pub shape: Vec<u32>,
    pub bytes: Vec<u8>,
}

/// Sidecar de dados de um programa montado com `.data`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DataSection {
    pub blobs: Vec<DataBlob>,
}

impl DataSection {
    /// Registra um blob; nome repetido: erro (uma definição por arquivo,
    /// mesma regra de `.reg`/`.equ`).
    pub fn declare(&mut self, blob: DataBlob) -> Result<()> {
        if self.blobs.iter().any(|b| b.name == blob.name) {
            return Err(anyhow!("blob '{}' redeclarado (uma definição por arquivo)", blob.name));
        }
        self.blobs.push(blob);
        Ok(())
    }

    /// Soma dos `bytes` (para dimensionar a região do loader, dia 3).
    pub fn total_bytes(&self) -> usize {
        self.blobs.iter().map(|b| b.bytes.len()).sum()
    }

    /// Busca case-insensitive (nomes guardados em minúsculas).
    pub fn find(&self, name: &str) -> Option<&DataBlob> {
        let key = name.trim().to_ascii_lowercase();
        self.blobs.iter().find(|b| b.name == key)
    }
}

/// Programa montado com dados: instruções + sidecar `.data` + base de
/// carga assumida (dia 3b: harness pode alocar tabelas antes; a base
/// viaja com o artefato e o loader verifica — Opção A pura, sem lookup).
#[derive(Debug, Clone)]
pub struct AssembledProgram {
    pub instrs: Vec<Instruction>,
    pub data: DataSection,
    pub data_base: u128,
}

/// `nome: .tipo valor` na seção `.data` (dia 1).
/// Limitação honesta do dia 1: a linha já passou por `strip_comment`
/// (`;`/`#` cortam), então strings não podem conter `;`/`#` — dia 2
/// trata aspas antes do corte de comentário.
fn parse_data_line(line: &str, syms: &SymbolTable) -> Result<DataBlob> {
    let colon = line.find(':').ok_or_else(|| {
        anyhow!("seção `.data` só aceita `nome: .tipo valor` (instrução vive em `.text`)")
    })?;
    let (name_raw, rest_raw) = (&line[..colon], &line[colon + 1..]);
    let name = name_raw.trim().to_ascii_lowercase();
    let ok_ident = !name.is_empty()
        && name.bytes().next().map_or(false, |b| b.is_ascii_alphabetic() || b == b'_')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_');
    if !ok_ident || name_raw.trim().split_whitespace().count() != 1 {
        return Err(anyhow!("nome de blob inválido '{}' (ident único)", name_raw.trim()));
    }
    let rest = rest_raw.trim();
    let mut words = rest.split_whitespace();
    let ty = words.next().ok_or_else(|| {
        anyhow!("blob '{}' sem tipo (use .u32/.i32/.f32/.str)", name)
    })?;
    let after_ty = rest[ty.len()..].trim();
    match ty.to_ascii_lowercase().as_str() {
        ".u32" => {
            if after_ty.is_empty() || after_ty.split_whitespace().count() != 1 {
                return Err(anyhow!("blob '{}': `.u32` precisa de um valor — ex: `{}: .u32 1920`", name, name));
            }
            // Literal decimal/0x-hex ou `.equ` (faixa u32 checada).
            let v = parse_imm_u128(after_ty, syms)
                .map_err(|_| anyhow!("blob '{}': valor `.u32` inválido '{}'", name, after_ty))?;
            let n = u32::try_from(v).map_err(|_| anyhow!("blob '{}': '{}' fora da faixa u32", name, after_ty))?;
            Ok(DataBlob { name, dtype: DataDtype::U32, shape: vec![], bytes: n.to_le_bytes().to_vec() })
        }
        ".i32" => {
            if after_ty.is_empty() || after_ty.split_whitespace().count() != 1 {
                return Err(anyhow!("blob '{}': `.i32` precisa de um valor — ex: `{}: .i32 -5`", name, name));
            }
            // Literal com sinal ou `.equ` (faixa i32 checada).
            let n = if let Ok(n) = after_ty.parse::<i32>() {
                n
            } else if let Ok(c) = syms.resolve_const(after_ty) {
                i32::try_from(c).map_err(|_| anyhow!("blob '{}': '{}' fora da faixa i32", name, after_ty))?
            } else {
                return Err(anyhow!("blob '{}': valor `.i32` inválido '{}'", name, after_ty));
            };
            Ok(DataBlob { name, dtype: DataDtype::I32, shape: vec![], bytes: n.to_le_bytes().to_vec() })
        }
        ".f32" => {
            // Dia 2: `[shape] [valores]` (nested, shape explícito) ou
            // escalar literal. `.f32 [...]` sem shape erra (sem inferência).
            if after_ty.starts_with('[') {
                return parse_nested_f32(&name, after_ty, syms);
            }
            if after_ty.is_empty() || after_ty.split_whitespace().count() != 1 {
                return Err(anyhow!("blob '{}': `.f32` precisa de um valor — ex: `{}: .f32 0.5`", name, name));
            }
            // Dia 1: literal float apenas (constante int→float seria
            // conversão com perda silenciosa — recusada por estrito).
            let f: f32 = after_ty
                .parse()
                .map_err(|_| anyhow!("blob '{}': valor `.f32` inválido '{}'", name, after_ty))?;
            if !f.is_finite() {
                return Err(anyhow!("blob '{}': `.f32` não-finito '{}' (NaN/Inf recusados)", name, after_ty));
            }
            Ok(DataBlob { name, dtype: DataDtype::F32, shape: vec![], bytes: f.to_le_bytes().to_vec() })
        }
        ".str" => {
            if after_ty.len() < 2 || !after_ty.starts_with('"') || !after_ty.ends_with('"') {
                return Err(anyhow!(
                    "blob '{}': `.str` precisa de aspas duplas — ex: `{}: .str \"...\"`",
                    name,
                    name
                ));
            }
            let inner = &after_ty[1..after_ty.len() - 1];
            if inner.contains('"') {
                return Err(anyhow!("blob '{}': `.str` sem escapes no dia 1 (aspas internas recusadas)", name));
            }
            Ok(DataBlob { name, dtype: DataDtype::Str, shape: vec![], bytes: inner.as_bytes().to_vec() })
        }
        _ if ty.starts_with('[') || after_ty.starts_with('[') => {
            Err(anyhow!("blob '{}': init multi-valor só `.f32` no dia 2 (ex: `{}: .f32 [2, 2] [1.0, 0.0, 0.0, 1.0]`)", name, name))
        }
        _ => Err(anyhow!("blob '{}': tipo '{}' inválido (use .u32/.i32/.f32/.str)", name, ty)),
    }
}

/// Multi-valor nested dia 2: `.f32 [R] [v...]` / `.f32 [R, C] [v...]`.
/// Gramática fechada, tudo numa linha (parser é line-based):
/// - shape explícito e obrigatório (sem inferência: `.f32 [...]` erra);
/// - rank 1–2 (rank 3+ erra); dims decimais/0x-hex/`.equ`, não-zero;
/// - valores separados por vírgula (trailing/leading/dupla vírgula erra);
/// - `count == prod(shape)` ou erro `esperados N, obtidos M`;
/// - `prod(shape) ≤ u32::MAX` (checado no parse, não em runtime);
/// - floats finitos; `[[...]]` estilo-JSON recusado explícito.
fn parse_nested_f32(name: &str, after_ty: &str, syms: &SymbolTable) -> Result<DataBlob> {
    // Primeiro grupo: shape.
    let s_close = after_ty.find(']').ok_or_else(|| {
        anyhow!("blob '{}': shape sem `]` de fechamento — ex: `{}: .f32 [2, 2] [...]`", name, name)
    })?;
    let shape_raw = after_ty[1..s_close].trim();
    let rest = after_ty[s_close + 1..].trim();
    if shape_raw.is_empty() {
        return Err(anyhow!(
            "blob '{}': `.f32 [...]` sem shape (inferência recusada) — ex: `{}: .f32 [4] [...]`",
            name,
            name
        ));
    }
    let mut shape: Vec<u32> = Vec::new();
    for d in shape_raw.split(',') {
        let d = d.trim();
        if d.is_empty() {
            return Err(anyhow!("blob '{}': vírgula vazia no shape `[{}]`", name, shape_raw));
        }
        if d.contains('.') {
            return Err(anyhow!(
                "blob '{}': `[{}]` parece lista de valores sem shape (shape explícito exigido) — ex: `{}: .f32 [2] [...]`",
                name,
                shape_raw,
                name
            ));
        }
        let v = parse_imm_u128(d, syms)
            .map_err(|_| anyhow!("blob '{}': dim inválida '{}' no shape `[{}]`", name, d, shape_raw))?;
        let n = u32::try_from(v).map_err(|_| anyhow!("blob '{}': dim '{}' fora da faixa u32", name, d))?;
        if n == 0 {
            return Err(anyhow!("blob '{}': dim zero no shape `[{}]` (tensor vazio recusado)", name, shape_raw));
        }
        shape.push(n);
    }
    if shape.len() > 2 {
        return Err(anyhow!(
            "blob '{}': rank {} não suportado no dia 2 (ranks 1–2) — shape `[{}]`",
            name,
            shape.len(),
            shape_raw
        ));
    }
    let prod: u128 = shape.iter().map(|&n| n as u128).product();
    if prod > u32::MAX as u128 {
        return Err(anyhow!("blob '{}': prod(shape)={} acima de u32::MAX", name, prod));
    }
    let expected = prod as usize;
    // Segundo grupo: valores (obrigatório; nada após ele).
    if rest.is_empty() {
        return Err(anyhow!(
            "blob '{}': shape `[{}]` sem bloco de valores — ex: `{}: .f32 [{}] [...]`",
            name,
            shape_raw,
            name,
            shape_raw
        ));
    }
    if !rest.starts_with('[') || !rest.ends_with(']') {
        return Err(anyhow!(
            "blob '{}': após o shape, esperado `[v, ...]` — obtido '{}'",
            name,
            rest
        ));
    }
    let vals_raw = rest[1..rest.len() - 1].trim();
    // Vírgula pendente/líder explícita (antes de tokenizar).
    if vals_raw.is_empty() {
        return Err(anyhow!("blob '{}': esperados {} valores, obtidos 0", name, expected));
    }
    if vals_raw.starts_with(',') || vals_raw.ends_with(',') {
        return Err(anyhow!("blob '{}': vírgula pendente/líder em `[{}]` (trailing comma recusada)", name, vals_raw));
    }
    let mut bytes: Vec<u8> = Vec::with_capacity(expected * 4);
    let mut got = 0usize;
    for tok in vals_raw.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            return Err(anyhow!("blob '{}': vírgula dupla/vazia em `[{}]`", name, vals_raw));
        }
        if tok.starts_with('[') {
            return Err(anyhow!("blob '{}': `[[...]]` estilo-JSON recusado (forma nested: `[shape] [v...]`)", name));
        }
        let f: f32 = tok
            .parse()
            .map_err(|_| anyhow!("blob '{}': valor `.f32` inválido '{}'", name, tok))?;
        if !f.is_finite() {
            return Err(anyhow!("blob '{}': `.f32` não-finito '{}' (NaN/Inf recusados)", name, tok));
        }
        bytes.extend_from_slice(&f.to_le_bytes());
        got += 1;
    }
    if got != expected {
        return Err(anyhow!("blob '{}': esperados {} valores, obtidos {}", name, expected, got));
    }
    Ok(DataBlob { name: name.to_string(), dtype: DataDtype::F32, shape, bytes })
}

// ---------------------------------------------------------------------------
// V-1b (dia 3): layout load-time — substituição em assemble-time (Opção A).
// Base fixa = GLOBAL_HEAP_START (o preload como PRIMEIRA alocação cai
// exatamente ali; o loader verifica, nunca assume). Sem ASLR, sem
// relocation, sem lookup em runtime: `@nome` vira o endereço literal.
// Correção registrada: região GLOBAL é 0x00 (top byte), logo endereço =
// offset puro (NÃO `(region<<60)|offset`).
// ---------------------------------------------------------------------------

/// Base de carga da seção `.data` (= início do heap GLOBAL fresco).
pub const DATA_LOAD_BASE: u128 = GLOBAL_HEAP_START;

/// Padding entre blobs (regra congelada dia 3; cache-line, futuro SIMD).
pub const DATA_BLOB_PAD: usize = 8;

/// Alinhamento por tipo (regra congelada dia 3): escalares/blobs f32 =
/// 4 (hardware natural); `.str` = 1 (sem requisito).
pub fn data_dtype_align(dtype: DataDtype) -> usize {
    match dtype {
        DataDtype::U32 | DataDtype::I32 | DataDtype::F32 => 4,
        DataDtype::Str => 1,
    }
}

fn align_up(off: usize, align: usize) -> usize {
    (off + align - 1) & !(align - 1)
}

/// Layout determinístico compartilhado assembler↔loader: `(nome, addr)`
/// na ordem de declaração. `start=align(off,tipo)`; próximo blob em
/// `align8(start+len)`; total inclui o pad final (base+total 8-alinhado).
/// `base` = cursor GLOBAL onde o loader vai posicionar (default
/// DATA_LOAD_BASE; dia 3b permite outra quando o harness aloca antes).
pub fn data_layout_addrs(data: &DataSection) -> Vec<(String, u128)> {
    data_layout_addrs_at(data, DATA_LOAD_BASE)
}

/// Variante com base explícita (dia 3b).
pub fn data_layout_addrs_at(data: &DataSection, base: u128) -> Vec<(String, u128)> {
    let mut out = Vec::with_capacity(data.blobs.len());
    let mut off = 0usize;
    for b in &data.blobs {
        let start = align_up(off, data_dtype_align(b.dtype));
        out.push((b.name.clone(), base + start as u128));
        off = align_up(start + b.bytes.len(), DATA_BLOB_PAD);
    }
    out
}

/// Tamanho total do layout (para `alloc_global` do loader).
pub fn data_layout_total(data: &DataSection) -> usize {
    let mut off = 0usize;
    for b in &data.blobs {
        let start = align_up(off, data_dtype_align(b.dtype));
        off = align_up(start + b.bytes.len(), DATA_BLOB_PAD);
    }
    off
}

fn parse_line(line: &str, labels: &HashMap<String, u128>, syms: &mut SymbolTable) -> Result<Instruction> {
    // Normaliza vírgulas -> espaços
    let normalized = line.replace(',', " ");
    let parts: Vec<&str> = normalized.split_whitespace().collect();
    if parts.is_empty() {
        return Err(anyhow!("linha vazia"));
    }
    let op = parts[0].to_ascii_uppercase();
    match op.as_str() {
        "TENSOR" => {
            // Formas:
            // TENSOR rdest rsrc1 rsrc2
            // TENSOR rdest 2 2 f32
            // TENSOR rdest, rows, cols, dtype
            if parts.len() < 2 {
                return Err(anyhow!("TENSOR precisa de rdest"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            // Tenta detectar forma com shape literal
            // Helper para detectar SPARSE/DENSITY nos tokens restantes
            let detect_sparse = |parts: &[&str]| -> (bool, f32) {
                let mut is_sparse = false;
                let mut dens = 0.05f32;
                for p in parts {
                    let up = p.to_ascii_uppercase();
                    if up == "SPARSE" { is_sparse = true; }
                    if up.starts_with("DENSITY") {
                        if let Some(eq) = p.find('=') {
                            if let Ok(v) = p[eq+1..].parse::<f32>() { dens = v; is_sparse = true; }
                        } else if let Ok(v) = p.parse::<f32>() { dens = v; }
                    }
                }
                (is_sparse, dens)
            };
            if parts.len() >= 4 {
                // Se parts[2] é número, `LxC`, ou constante `.equ` (RFC-0037)
                let maybe_rows = parts[2].replace('x', " ").trim().to_string();
                // Checa se é literal numérico
                if maybe_rows.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) || parts[2].contains('x') || syms.is_const(parts[2]) {
                    // Literal. Rastreia tokens consumidos (shape/dtype) para
                    // o restante (SPARSE/DENSITY/FILL) não colidir com eles.
                    let mut consumed = vec![false; parts.len()];
                    consumed[0] = true;
                    consumed[1] = true;
                    consumed[2] = true;
                    let (rows, cols) = if parts[2].contains('x') {
                        let mut split = parts[2].split('x');
                        let r = split.next().unwrap().parse::<u64>().map_err(|_| anyhow!("rows inválido"))?;
                        let c = split.next().unwrap().parse::<u64>().map_err(|_| anyhow!("cols inválido"))?;
                        (r, c)
                    } else if parts.len() >= 4 && (parts[3].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) || syms.is_const(parts[3])) {
                        // rows/cols: literal decimal ou `.equ` (RFC-0037).
                        let r = parse_dim_u64(parts[2], syms)
                            .map_err(|_| anyhow!("rows inválido"))?;
                        let c = parse_dim_u64(parts[3], syms)
                            .map_err(|_| anyhow!("cols inválido"))?;
                        consumed[3] = true;
                        (r, c)
                    } else if syms.is_const(parts[2]) {
                        // rows é `.equ` mas cols ausente/não-numérico.
                        return Err(anyhow!("TENSOR precisa de rows, cols — ex: TENSOR r0 2 2 f32"));
                    } else {
                        (2, 2)
                    };
                    // dtype: primeiro não-consumido sem '=' que parseie.
                    let mut dtype_str = "f32";
                    for (i, p) in parts.iter().enumerate().skip(2) {
                        if consumed[i] || p.contains('=') {
                            continue;
                        }
                        if parse_dtype(p).is_ok() {
                            dtype_str = p;
                            consumed[i] = true;
                            break;
                        }
                    }
                    let dtype = parse_dtype(dtype_str)?;
                    // Restante: SPARSE/DENSITY/FILL (posições livres).
                    let remaining: Vec<&str> = parts.iter().enumerate()
                        .skip(2)
                        .filter(|(i, _)| !consumed[*i])
                        .map(|(_, p)| *p)
                        .collect();
                    let (is_sparse, dens) = detect_sparse(&remaining);
                    reject_unknown("TENSOR", &remaining, &["SPARSE", "DENSITY=", "FILL="])?;
                    let mut instr = instr_tensor(rdest, 0xFF, 0xFF, rows, cols, dtype);
                    if is_sparse { instr.set_sparse(true, dens); }
                    // RFC-0019: FILL escalar (payload[22..26] + flag).
                    for p in &remaining {
                        let up = p.to_ascii_uppercase();
                        if let Some(v) = up.strip_prefix("FILL=") {
                            let f = v.parse::<f32>().map_err(|_| anyhow!("TENSOR FILL inválido '{}'", p))?;
                            if is_sparse {
                                return Err(anyhow!("TENSOR: FILL + SPARSE contraditórios"));
                            }
                            instr.set_tensor_fill(f);
                        }
                    }
                    return Ok(instr);
                }
            }
            // Forma com registradores
            let rsrc1 = if parts.len() > 2 { parse_reg(parts[2], syms)? } else { 0xFF };
            let rsrc2 = if parts.len() > 3 { parse_reg(parts[3], syms)? } else { 0xFF };
            // Tenta extrair payload rows/cols/dtype se houver tokens extras
            let mut rows = 2u64;
            let mut cols = 2u64;
            let mut dtype = 0u8;
            if parts.len() > 4 {
                if parts[4].contains('x') {
                    let mut s = parts[4].split('x');
                    rows = s.next().unwrap().parse().unwrap_or(2);
                    cols = s.next().unwrap().parse().unwrap_or(2);
                } else {
                    // RFC-0008: 4º token sem 'x' só pode ser dtype (antes:
                    // ignorado em silêncio, mantendo 2x2).
                    dtype = parse_dtype(parts[4])
                        .map_err(|_| anyhow!("TENSOR token '{}' inválido (use LxC, dtype ou SPARSE)", parts[4]))?;
                }
            }
            if parts.len() > 5 {
                let up5 = parts[5].to_ascii_uppercase();
                if up5 == "SPARSE" || up5.starts_with("DENSITY") {
                    // Vai para detect_sparse abaixo; dtype segue 0/f32.
                } else {
                    dtype = parse_dtype(parts[5])
                        .map_err(|_| anyhow!("TENSOR dtype '{}' inválido (use f32/f16/i8/u8)", parts[5]))?;
                }
            }
            let remaining = if parts.len() > 6 { &parts[6..] } else if parts.len() > 5 && parts[5].to_ascii_uppercase() == "SPARSE" { &parts[5..] } else { &[] as &[&str] };
            let (is_sparse, dens) = detect_sparse(remaining);
            reject_unknown("TENSOR", remaining, &["SPARSE", "DENSITY=", "FILL="])?;
            let mut instr = instr_tensor(rdest, rsrc1, rsrc2, rows, cols, dtype);
            if is_sparse { instr.set_sparse(true, dens); }
            for p in remaining {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("FILL=") {
                    let f = v.parse::<f32>().map_err(|_| anyhow!("TENSOR FILL inválido '{}'", p))?;
                    if is_sparse {
                        return Err(anyhow!("TENSOR: FILL + SPARSE contraditórios"));
                    }
                    instr.set_tensor_fill(f);
                }
            }
            Ok(instr)
        }
        "ATTN" => {
            if parts.len() < 5 {
                return Err(anyhow!("ATTN precisa de rdest, rQ, rK, rV — ex: ATTN r3, r0, r1, r2"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rq = parse_reg(parts[2], syms)?;
            let rk = parse_reg(parts[3], syms)?;
            let rv = parse_reg(parts[4], syms)?;
            let notify = if parts.len() > 5 {
                parts[5..].iter().any(|p| {
                    let up = p.to_ascii_uppercase();
                    up == "NOTIFY_EACH_HEAD" || up == "NOTIFY" || up == "NOTIFY_EACH"
                })
            } else { false };
            if parts.len() > 5 {
                reject_unknown("ATTN", &parts[5..], &["NOTIFY", "NOTIFY_EACH_HEAD", "NOTIFY_EACH"])?;
            }
            if notify {
                Ok(instr_attn_notify(rdest, rq, rk, rv))
            } else {
                Ok(instr_attn(rdest, rq, rk, rv))
            }
        }
        "STREAM" => {
            if parts.len() < 3 {
                return Err(anyhow!("STREAM precisa de r_src, r_sink"));
            }
            let rsrc = parse_reg(parts[1], syms)?;
            // Suporta periféricos simbólicos: INPUT (1), SAMPLE (2), DECODED (4), OUTPUT (0)
            let rsink = match parts[2].to_ascii_uppercase().as_str() {
                "INPUT" | "PERIPHERAL_INPUT" | "1" => 1,
                "SAMPLE" | "PERIPHERAL_SAMPLE" | "2" => 2,
                "DECODED" | "OUTPUT_DECODED" | "PERIPHERAL_OUTPUT_DECODED" | "4" => 4,
                "OUTPUT" | "PERIPHERAL_OUTPUT" | "0" => 0,
                _ => parse_reg(parts[2], syms)?,
            };
            let blocking = if parts.len() > 3 {
                // RFC-0008: modo deve ser explícito; qualquer outra coisa
                // (antes: silenciosamente BLOCKING) agora é erro.
                match parts[3].to_ascii_uppercase().as_str() {
                    "BLOCKING" => true,
                    "DROP" => false,
                    _ => return Err(anyhow!("STREAM modo '{}' inválido (use BLOCKING ou DROP)", parts[3])),
                }
            } else {
                true
            };
            if parts.len() > 4 {
                reject_unknown("STREAM", &parts[4..], &[])?;
            }
            Ok(instr_stream(rsrc, rsink, blocking))
        }
        "FORK" => {
            if parts.len() < 3 {
                return Err(anyhow!("FORK precisa de rdest, prioridade (RED/BLUE/GREEN) ou rótulo"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let second = parts[2].to_ascii_uppercase();

            // Se segundo argumento é um rótulo conhecido (ex: FORK R12, MAIN_LOOP, GREEN)
            if let Some(&target_pc) = labels.get(&second) {
                let prio = if parts.len() > 3 {
                    match parts[3].to_ascii_uppercase().as_str() {
                        "RED" | "2" => FORK_FLAG_RED,
                        "BLUE" | "1" => FORK_FLAG_BLUE,
                        "GREEN" | "0" => FORK_FLAG_GREEN,
                        "NOTIFY" | "NOTIFY_SCHEDULER" => FORK_FLAG_GREEN,
                        _ => return Err(anyhow!("FORK prioridade '{}' inválida (use RED/BLUE/GREEN)", parts[3])),
                    }
                } else {
                    FORK_FLAG_GREEN
                };
                let notify = parts.iter().skip(3).any(|p| {
                    let up = p.to_ascii_uppercase();
                    up == "NOTIFY" || up == "NOTIFY_SCHEDULER"
                });
                reject_unknown("FORK", &parts[3..], &["RED", "BLUE", "GREEN", "0", "1", "2", "NOTIFY", "NOTIFY_SCHEDULER"])?;
                let flags = if notify { prio | FORK_FLAG_NOTIFY } else { prio };
                let mut instr = instr_fork(rdest, flags);
                instr.set_imm_u128(target_pc);
                Ok(instr)
            } else {
                // Sintaxe clássica: FORK Rd, PRIORITY [, NOTIFY]
                // (aqui `second` NÃO é rótulo conhecido — ramo acima cobre)
                let prio = match second.as_str() {
                    "RED" | "2" => FORK_FLAG_RED,
                    "BLUE" | "1" => FORK_FLAG_BLUE,
                    "GREEN" | "0" => FORK_FLAG_GREEN,
                    _ => return Err(anyhow!("FORK prioridade/rótulo '{}' inválida (use RED/BLUE/GREEN ou rótulo)", parts[2])),
                };
                if parts.len() > 3 {
                    reject_unknown("FORK", &parts[3..], &["NOTIFY", "NOTIFY_SCHEDULER"])?;
                }
                let notify = if parts.len() > 3 {
                    parts[3].to_ascii_uppercase() == "NOTIFY" || parts[3].to_ascii_uppercase() == "NOTIFY_SCHEDULER"
                } else { false };
                let flags = if notify { prio | FORK_FLAG_NOTIFY } else { prio };
                Ok(instr_fork(rdest, flags))
            }
        }
        "ABORT" => {
            if parts.len() < 3 {
                return Err(anyhow!("ABORT precisa de r_target, r_timestamp"));
            }
            let rt = parse_reg(parts[1], syms)?;
            let ts = parse_reg(parts[2], syms)?;
            if parts.len() > 3 {
                reject_unknown("ABORT", &parts[3..], &[])?;
            }
            Ok(instr_abort(rt, ts))
        }
        "SENSE" => {
            if parts.len() < 3 {
                return Err(anyhow!("SENSE precisa de rdest, periférico (AUDIO/VAD/TOKEN/USER_INPUT/AUDIO_PCM/CODEC_FRAME/0/1/3/5/6/7)"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let periph = match parts[2].to_ascii_uppercase().as_str() {
                "AUDIO" | "0" => SENSE_AUDIO,
                "VAD" | "1" => SENSE_VAD,
                "TOKEN" | "3" => SENSE_TOKEN,
                "USER_INPUT" | "PERIPHERAL_USER_INPUT" | "5" => SENSE_USER_INPUT,
                "AUDIO_PCM" | "PCM" | "6" => SENSE_AUDIO_PCM,
                "CODEC_FRAME" | "MIMI" | "7" => SENSE_CODEC_FRAME,
                _ => {
                    if let Ok(n) = parts[2].parse::<u8>() {
                        n
                    } else {
                        // RFC-0008: periférico desconhecido é erro (antes:
                        // silenciosamente AUDIO). Numérico u8 segue aceito.
                        return Err(anyhow!("SENSE periférico '{}' inválido (use AUDIO/VAD/TOKEN/USER_INPUT/AUDIO_PCM/CODEC_FRAME ou 0/1/3/5/6/7)", parts[2]));
                    }
                }
            };
            if parts.len() > 3 {
                reject_unknown("SENSE", &parts[3..], &[])?;
            }
            Ok(instr_sense(rdest, periph))
        }
        "NORM" => {
            if parts.len() < 3 {
                return Err(anyhow!("NORM precisa de rdest, r_src, r_gamma — ex: NORM r2, r0, r1, r3"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rsrc = parse_reg(parts[2], syms)?;
            let rgamma = if parts.len() > 3 { parse_reg(parts[3], syms)? } else { 0xFF };
            let rbeta = if parts.len() > 4 { parse_reg(parts[4], syms)? } else { 0xFF };
            if parts.len() > 5 {
                reject_unknown("NORM", &parts[5..], &[])?;
            }
            Ok(instr_norm(rdest, rsrc, rgamma, rbeta))
        }
        "FFN" => {
            if parts.len() < 4 {
                return Err(anyhow!("FFN precisa de rdest, r_src, r_w1, r_w2 — ex: FFN r3, r0, r1, r2 ou FFN r3, r0, r1, r4, r2, r5 (com bias)"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rsrc = parse_reg(parts[2], syms)?;
            let rw1 = parse_reg(parts[3], syms)?;
            if parts.len() >= 7 {
                let rb1 = parse_reg(parts[4], syms)?;
                let rw2 = parse_reg(parts[5], syms)?;
                let rb2 = parse_reg(parts[6], syms)?;
                if parts.len() > 7 {
                    reject_unknown("FFN", &parts[7..], &[])?;
                }
                Ok(instr_ffn_with_bias(rdest, rsrc, rw1, rb1, rw2, rb2))
            } else if parts.len() == 6 {
                let rw2 = parse_reg(parts[4], syms)?;
                let maybe_rb = parse_reg(parts[5], syms)?;
                let mut instr = instr_ffn(rdest, rsrc, rw1, rw2);
                instr.payload[1] = maybe_rb;
                Ok(instr)
            } else {
                let rw2 = parse_reg(parts[4], syms)?;
                Ok(instr_ffn(rdest, rsrc, rw1, rw2))
            }
        }
        "EMBED" => {
            if parts.len() < 4 {
                return Err(anyhow!("EMBED precisa de rdest, r_token, r_table — ex: EMBED r10, r11, r1"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rtoken = parse_reg(parts[2], syms)?;
            let rtable = parse_reg(parts[3], syms)?;
            if parts.len() > 4 {
                reject_unknown("EMBED", &parts[4..], &[])?;
            }
            Ok(instr_embed(rdest, rtoken, rtable))
        }
        "ADD" => {
            if parts.len() < 4 {
                return Err(anyhow!("ADD precisa de rdest, r_src1, r_src2 — ex: ADD r10, r10, r14"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rsrc1 = parse_reg(parts[2], syms)?;
            let rsrc2 = parse_reg(parts[3], syms)?;
            if parts.len() > 4 {
                reject_unknown("ADD", &parts[4..], &[])?;
            }
            Ok(instr_add(rdest, rsrc1, rsrc2))
        }
        "SAMPLE" => {
            if parts.len() < 3 {
                return Err(anyhow!("SAMPLE precisa de rdest, r_logits — ex: SAMPLE r11, r15"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rlogits = parse_reg(parts[2], syms)?;
            let mut temp = 1.0f32;
            let mut topk = 0u16;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if up.starts_with("TEMP=") {
                    if let Ok(v) = up["TEMP=".len()..].parse::<f32>() { temp = v; }
                    else { return Err(anyhow!("SAMPLE TEMP inválido '{}'", p)); }
                } else if up.starts_with("TOPK=") {
                    if let Ok(v) = up["TOPK=".len()..].parse::<u16>() { topk = v; }
                    else { return Err(anyhow!("SAMPLE TOPK inválido '{}'", p)); }
                } else if let Ok(v) = p.parse::<f32>() {
                    temp = v;
                } else {
                    return Err(anyhow!("SAMPLE token desconhecido '{}' (use TEMP=/TOPK=/número)", p));
                }
            }
            let mut instr = instr_sample(rdest, rlogits, temp);
            if topk > 0 {
                instr.set_sample_topk(topk);
            }
            Ok(instr)
        }
        "COMPARE" => {
            // COMPARE R1, R2|imm|EOS_TOKEN [PRED=EQ|NE|LT|LE|GT|GE]
            // (RFC-0007: predicado em payload[16]; default EQ = legado)
            if parts.len() < 3 {
                return Err(anyhow!("COMPARE precisa de r_src1, r_src2/imediato — ex: COMPARE r11, EOS_TOKEN"));
            }
            let rsrc1 = parse_reg(parts[1], syms)?;
            let second = parts[2].to_ascii_uppercase();
            let mut instr = if second == "EOS_TOKEN" {
                instr_compare(rsrc1, 0xFF, EOS_TOKEN_DEFAULT)
            } else if let Ok(r2) = parse_reg(parts[2], syms) {
                instr_compare(rsrc1, r2, 0)
            } else if let Ok(n) = parse_imm_dec_or_const(parts[2], syms) {
                instr_compare(rsrc1, 0xFF, n)
            } else {
                return Err(anyhow!("COMPARE segundo operando inválido '{}'", parts[2]));
            };
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("PRED=") {
                    let pred = match v {
                        "EQ" => CMP_EQ,
                        "NE" => CMP_NE,
                        "LT" => CMP_LT,
                        "LE" => CMP_LE,
                        "GT" => CMP_GT,
                        "GE" => CMP_GE,
                        _ => return Err(anyhow!("COMPARE PRED inválido '{}' (use EQ/NE/LT/LE/GT/GE)", p)),
                    };
                    instr.set_compare_pred(pred);
                } else {
                    return Err(anyhow!("COMPARE token desconhecido '{}' (use PRED=)", p));
                }
            }
            Ok(instr)
        }
        "IF_EQUAL" => {
            if parts.len() < 2 {
                return Err(anyhow!("IF_EQUAL precisa de rótulo alvo — ex: IF_EQUAL PROGRAM_END"));
            }
            let label = parts[1].to_ascii_uppercase();
            let target_pc = *labels.get(&label)
                .ok_or_else(|| anyhow!("rótulo '{}' não encontrado para IF_EQUAL", label))?;
            if parts.len() > 2 {
                reject_unknown("IF_EQUAL", &parts[2..], &[])?;
            }
            Ok(instr_if_equal(target_pc))
        }
        "JUMP" | "JMP" => {
            if parts.len() < 2 {
                return Err(anyhow!("JUMP precisa de rótulo alvo — ex: JUMP MAIN_LOOP"));
            }
            let label = parts[1].to_ascii_uppercase();
            let target_pc = *labels.get(&label)
                .ok_or_else(|| anyhow!("rótulo '{}' não encontrado para JUMP", label))?;
            if parts.len() > 2 {
                reject_unknown("JUMP", &parts[2..], &[])?;
            }
            Ok(instr_jump(target_pc))
        }
        "IF_INTERRUPT" => {
            if parts.len() == 2 {
                let label = parts[1].to_ascii_uppercase();
                let target_pc = *labels.get(&label)
                    .ok_or_else(|| anyhow!("rótulo '{}' não encontrado para IF_INTERRUPT", label))?;
                Ok(instr_if_interrupt(0xFF, target_pc))
            } else if parts.len() >= 3 {
                let rcond = parse_reg(parts[1], syms)?;
                let label = parts[2].to_ascii_uppercase();
                let target_pc = *labels.get(&label)
                    .ok_or_else(|| anyhow!("rótulo '{}' não encontrado para IF_INTERRUPT", label))?;
                if parts.len() > 3 {
                    reject_unknown("IF_INTERRUPT", &parts[3..], &[])?;
                }
                Ok(instr_if_interrupt(rcond, target_pc))
            } else {
                Err(anyhow!("IF_INTERRUPT precisa de rótulo ou rcond, rótulo"))
            }
        }
        "MATVEC" => {
            if parts.len() < 4 {
                return Err(anyhow!("MATVEC precisa de rdest, r_x, r_w — ex: MATVEC r2, r0, r1"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rx = parse_reg(parts[2], syms)?;
            let rw = parse_reg(parts[3], syms)?;
            if parts.len() > 4 {
                reject_unknown("MATVEC", &parts[4..], &[])?;
            }
            Ok(instr_matvec(rdest, rx, rw))
        }
        "MUL" => {
            if parts.len() < 4 {
                return Err(anyhow!("MUL precisa de rdest, r1, r2 — ex: MUL r6, r6, r7"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let r1 = parse_reg(parts[2], syms)?;
            let r2 = parse_reg(parts[3], syms)?;
            if parts.len() > 4 {
                reject_unknown("MUL", &parts[4..], &[])?;
            }
            Ok(instr_mul(rdest, r1, r2))
        }
        "SILU" => {
            if parts.len() < 3 {
                return Err(anyhow!("SILU precisa de rdest, rsrc — ex: SILU r6, r6"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rsrc = parse_reg(parts[2], syms)?;
            if parts.len() > 3 {
                reject_unknown("SILU", &parts[3..], &[])?;
            }
            Ok(instr_silu(rdest, rsrc))
        }
        "SSM_SCAN" => {
            // SSM_SCAN rY, rX, rH, rP [D_INNER=n D_STATE=n LAYER=n] [CONV] [GATE]
            if parts.len() < 5 {
                return Err(anyhow!("SSM_SCAN precisa de rdest, r_x, r_h, r_params — ex: SSM_SCAN r5, r0, r1, r2 D_INNER=8 D_STATE=4"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rx = parse_reg(parts[2], syms)?;
            let rh = parse_reg(parts[3], syms)?;
            let rp = parse_reg(parts[4], syms)?;
            let (mut di, mut ds, mut layer) = (1usize, 1usize, 0u8);
            let mut flags = 0u8;
            for p in &parts[5..] {
                let up = p.to_ascii_uppercase();
                if up == "CONV" { flags |= SSM_SCAN_FLAG_CONV; }
                else if up == "GATE" { flags |= SSM_SCAN_FLAG_GATE; }
                else if let Some(v) = up.strip_prefix("D_INNER=") { if let Ok(n) = v.parse::<usize>() { di = n; } else { return Err(anyhow!("SSM_SCAN D_INNER inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("D_STATE=") { if let Ok(n) = v.parse::<usize>() { ds = n; } else { return Err(anyhow!("SSM_SCAN D_STATE inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("LAYER=") { if let Ok(n) = v.parse::<u8>() { layer = n; } else { return Err(anyhow!("SSM_SCAN LAYER inválido '{}'", p)); } }
                else { return Err(anyhow!("SSM_SCAN token desconhecido '{}' (use D_INNER=/D_STATE=/LAYER=/CONV/GATE)", p)); }
            }
            let mut instr = instr_ssm_scan(rdest, rx, rh, rp, di, ds, layer);
            instr.flags = flags;
            Ok(instr)
        }
        "SSM_RESET" => {
            // SSM_RESET rH [D_INNER=n D_STATE=n LAYER=n]
            if parts.len() < 2 {
                return Err(anyhow!("SSM_RESET precisa de r_h — ex: SSM_RESET r1"));
            }
            let rh = parse_reg(parts[1], syms)?;
            let (mut di, mut ds, mut layer) = (1usize, 1usize, 0u8);
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("D_INNER=") { if let Ok(n) = v.parse::<usize>() { di = n; } else { return Err(anyhow!("SSM_RESET D_INNER inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("D_STATE=") { if let Ok(n) = v.parse::<usize>() { ds = n; } else { return Err(anyhow!("SSM_RESET D_STATE inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("LAYER=") { if let Ok(n) = v.parse::<u8>() { layer = n; } else { return Err(anyhow!("SSM_RESET LAYER inválido '{}'", p)); } }
                else { return Err(anyhow!("SSM_RESET token desconhecido '{}' (use D_INNER=/D_STATE=/LAYER=)", p)); }
            }
            Ok(instr_ssm_reset(rh, di, ds, layer))
        }
        "CODEC_ENC" => {
            // CODEC_ENC rD, rS [TENSOR]
            if parts.len() < 3 {
                return Err(anyhow!("CODEC_ENC precisa de rdest, r_pcm — ex: CODEC_ENC r2, r0"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rpcm = parse_reg(parts[2], syms)?;
            let as_tensor = parts[3..].iter().any(|p| p.to_ascii_uppercase() == "TENSOR");
            if !as_tensor && !parts[3..].is_empty() {
                reject_unknown("CODEC_ENC", &parts[3..], &["TENSOR"])?;
            }
            Ok(instr_codec_enc(rdest, rpcm, as_tensor))
        }
        "CODEC_DEC" => {
            // CODEC_DEC rD, rS [TENSOR]
            if parts.len() < 3 {
                return Err(anyhow!("CODEC_DEC precisa de rdest, r_codes — ex: CODEC_DEC r3, r2"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rc = parse_reg(parts[2], syms)?;
            let as_tensor = parts[3..].iter().any(|p| p.to_ascii_uppercase() == "TENSOR");
            if !as_tensor && !parts[3..].is_empty() {
                reject_unknown("CODEC_DEC", &parts[3..], &["TENSOR"])?;
            }
            Ok(instr_codec_dec(rdest, rc, as_tensor))
        }
        "AUDIO_ALIGN" => {
            // AUDIO_ALIGN rD, rU, rA [SR=24000 FRAME=1920 HZ=12.5 DELAY=160]
            if parts.len() < 4 {
                return Err(anyhow!("AUDIO_ALIGN precisa de rdest, r_user, r_ai — ex: AUDIO_ALIGN r4, r0, r1"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let ru = parse_reg(parts[2], syms)?;
            let ra = parse_reg(parts[3], syms)?;
            let (mut sr, mut spf, mut hz, mut dl) = (24_000u32, 1920u32, 12.5f32, 160.0f32);
            for p in &parts[4..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("SR=") { if let Ok(n) = v.parse::<u32>() { sr = n; } else { return Err(anyhow!("AUDIO_ALIGN SR inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("FRAME=") { if let Ok(n) = v.parse::<u32>() { spf = n; } else { return Err(anyhow!("AUDIO_ALIGN FRAME inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("HZ=") { if let Ok(n) = v.parse::<f32>() { hz = n; } else { return Err(anyhow!("AUDIO_ALIGN HZ inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("DELAY=") { if let Ok(n) = v.parse::<f32>() { dl = n; } else { return Err(anyhow!("AUDIO_ALIGN DELAY inválido '{}'", p)); } }
                else { return Err(anyhow!("AUDIO_ALIGN token desconhecido '{}' (use SR=/FRAME=/HZ=/DELAY=)", p)); }
            }
            let mut instr = instr_audio_align(rdest, ru, ra);
            instr.set_audio_align_params(sr, spf, hz, dl);
            Ok(instr)
        }
        "CTX_SWITCH" => {
            // CTX_SWITCH MAMBA|TRANSFORMER|AUDIO [RED|BLUE|GREEN]
            if parts.len() < 2 {
                return Err(anyhow!("CTX_SWITCH precisa de pipeline — ex: CTX_SWITCH MAMBA, RED"));
            }
            let pipe = match parts[1].to_ascii_uppercase().as_str() {
                "MAMBA" | "SSM" | "0" => PIPE_MAMBA,
                "TRANSFORMER" | "LLM" | "1" => PIPE_TRANSFORMER,
                "AUDIO" | "MOSHI" | "DEPFORMER" | "2" => PIPE_AUDIO,
                _ => {
                    // RFC-0008: pipe desconhecido é erro (antes: silenciosamente
                    // TRANSFORMER, a menos que fosse registrador válido legado).
                    if parse_reg(parts[1], syms).is_ok() {
                        return Err(anyhow!("CTX_SWITCH pipe '{}' deve ser MAMBA/TRANSFORMER/AUDIO (forma legado por registrador removida no modo estrito)", parts[1]));
                    } else {
                        return Err(anyhow!("CTX_SWITCH pipe '{}' inválido (use MAMBA/TRANSFORMER/AUDIO)", parts[1]));
                    }
                }
            };
            let mut prio = 0u8;
            for p in &parts[2..] {
                match p.to_ascii_uppercase().as_str() {
                    "RED" | "2" => prio = 0b10,
                    "BLUE" | "1" => prio = 0b01,
                    "GREEN" | "0" => prio = 0b00,
                    _ => return Err(anyhow!("CTX_SWITCH token desconhecido '{}' (use RED/BLUE/GREEN)", p)),
                }
            }
            Ok(instr_ctx_switch(pipe, prio))
        }
        "ROPE" => {
            // ROPE rD, rS POS=n HDIM=n NHEADS=n [THETA=n] [INPLACE]
            if parts.len() < 3 {
                return Err(anyhow!("ROPE precisa de rdest, rsrc — ex: ROPE r2, r0 POS=0 HDIM=8 NHEADS=2"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rsrc = parse_reg(parts[2], syms)?;
            let (mut pos, mut hd, mut nh, mut theta) = (0u32, 2usize, 1usize, 10_000.0f32);
            let mut inplace = false;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if up == "INPLACE" { inplace = true; }
                else if let Some(v) = up.strip_prefix("POS=") { if let Ok(n) = v.parse::<u32>() { pos = n; } else { return Err(anyhow!("ROPE POS inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("HDIM=") { if let Ok(n) = v.parse::<usize>() { hd = n; } else { return Err(anyhow!("ROPE HDIM inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("NHEADS=") { if let Ok(n) = v.parse::<usize>() { nh = n; } else { return Err(anyhow!("ROPE NHEADS inválido '{}'", p)); } }
                else if let Some(v) = up.strip_prefix("THETA=") { if let Ok(n) = v.parse::<f32>() { theta = n; } else { return Err(anyhow!("ROPE THETA inválido '{}'", p)); } }
                else { return Err(anyhow!("ROPE token desconhecido '{}' (use POS=/HDIM=/NHEADS=/THETA=/INPLACE)", p)); }
            }
            let mut instr = instr_rope(rdest, rsrc, pos, hd, nh, theta);
            if inplace { instr.flags |= ROPE_FLAG_INPLACE; }
            Ok(instr)
        }
        "GATHER" | "SCATTER_ADD" => {
            // GATHER rD, rTable, rIdx [, rAcc] [AXIS=n] [MODE=...]
            // SCATTER_ADD rD, rTable, rIdx [AXIS=n] (alias MODE=SCATTER_ADD)
            if parts.len() < 4 {
                return Err(anyhow!("GATHER precisa de rdest, rTable, rIdx — ex: GATHER r5, r0, r1 AXIS=0"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rtable = parse_reg(parts[2], syms)?;
            let ridx = parse_reg(parts[3], syms)?;
            let (mut axis, mut mode, mut racc) = (0u8, GATHER_MODE_GATHER, 0xFF);
            let mut rpos = 4;
            if op == "SCATTER_ADD" {
                mode = GATHER_MODE_SCATTER_ADD;
            }
            // rAcc opcional: próximo token é registrador e não KV
            if parts.len() > rpos {
                let up = parts[rpos].to_ascii_uppercase();
                let is_kv = up.contains('=') || up == "AXIS" || up == "MODE"
                    || up.starts_with("AXIS=") || up.starts_with("MODE=");
                if !is_kv {
                    if let Ok(r) = parse_reg(parts[rpos], syms) {
                        racc = r;
                        rpos += 1;
                    }
                }
            }
            for p in &parts[rpos..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("AXIS=") {
                    if let Ok(n) = v.parse::<u8>() { axis = n; }
                    else { return Err(anyhow!("GATHER AXIS inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("MODE=") {
                    mode = match v {
                        "GATHER" => GATHER_MODE_GATHER,
                        "SCATTER_ADD" => GATHER_MODE_SCATTER_ADD,
                        "SCATTER_MAX" => GATHER_MODE_SCATTER_MAX,
                        _ => return Err(anyhow!("GATHER MODE inválido '{}' (use GATHER/SCATTER_ADD/SCATTER_MAX)", parts[rpos])),
                    };
                } else {
                    return Err(anyhow!("GATHER token desconhecido '{}' (use AXIS=/MODE=)", p));
                }
            }
            Ok(instr_gather(rdest, rtable, ridx, racc, axis, mode))
        }
        "DISTANCE" => {
            // DISTANCE rD, rQuery, rBank [METRIC=...] [TOPK=n]
            if parts.len() < 4 {
                return Err(anyhow!("DISTANCE precisa de rdest, rQuery, rBank — ex: DISTANCE r5, r0, r1 METRIC=COSINE TOPK=5"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let rquery = parse_reg(parts[2], syms)?;
            let rbank = parse_reg(parts[3], syms)?;
            let (mut metric, mut topk) = (DIST_METRIC_EUCLID, 0u16);
            for p in &parts[4..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("METRIC=") {
                    metric = match v {
                        "EUCLID" => DIST_METRIC_EUCLID,
                        "COSINE" => DIST_METRIC_COSINE,
                        "MANHATTAN" => DIST_METRIC_MANHATTAN,
                        "DOT" => DIST_METRIC_DOT,
                        _ => return Err(anyhow!("DISTANCE METRIC inválida '{}'", p)),
                    };
                } else if let Some(v) = up.strip_prefix("TOPK=") {
                    if let Ok(n) = v.parse::<u16>() { topk = n; }
                    else { return Err(anyhow!("DISTANCE TOPK inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("DISTANCE token desconhecido '{}' (use METRIC=/TOPK=)", p));
                }
            }
            Ok(instr_distance(rdest, rquery, rbank, metric, topk))
        }
        "RANK1_UPDATE" => {
            // RANK1_UPDATE rH, rV, rK [ALPHA=a] [BETA=b] [MODE=...] [LAYER=n]
            if parts.len() < 4 {
                return Err(anyhow!("RANK1_UPDATE precisa de rH, rV, rK — ex: RANK1_UPDATE rH, r1, r2 ALPHA=0.99 BETA=1.0"));
            }
            let rh = parse_reg(parts[1], syms)?;
            let rv = parse_reg(parts[2], syms)?;
            let rk = parse_reg(parts[3], syms)?;
            let (mut alpha, mut beta, mut mode, mut layer) = (1.0f32, 1.0f32, RANK1_MODE_HEBBIAN, 0u8);
            for p in &parts[4..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("ALPHA=") {
                    if let Ok(n) = v.parse::<f32>() { alpha = n; }
                    else { return Err(anyhow!("RANK1_UPDATE ALPHA inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("BETA=") {
                    if let Ok(n) = v.parse::<f32>() { beta = n; }
                    else { return Err(anyhow!("RANK1_UPDATE BETA inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("MODE=") {
                    mode = match v {
                        "HEBBIAN" => RANK1_MODE_HEBBIAN,
                        "DELTA" => RANK1_MODE_DELTA,
                        "FORGET" => RANK1_MODE_FORGET,
                        _ => return Err(anyhow!("RANK1_UPDATE MODE inválido '{}'", p)),
                    };
                } else if let Some(v) = up.strip_prefix("LAYER=") {
                    if let Ok(n) = v.parse::<u8>() { layer = n; }
                    else { return Err(anyhow!("RANK1_UPDATE LAYER inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("RANK1_UPDATE token desconhecido '{}' (use ALPHA=/BETA=/MODE=/LAYER=)", p));
                }
            }
            Ok(instr_rank1_update(rh, rv, rk, alpha, beta, mode, layer))
        }
        "RNG_SEED" => {
            // RNG_SEED Rs | DEFAULT — rsrc1 com a semente (0xFF = default fixo)
            if parts.len() < 2 {
                return Err(anyhow!("RNG_SEED precisa de Rs ou DEFAULT"));
            }
            if parts.len() > 2 {
                reject_unknown("RNG_SEED", &parts[2..], &[])?;
            }
            let up = parts[1].to_ascii_uppercase();
            if up == "DEFAULT" {
                Ok(instr_rng_seed(0xFF))
            } else {
                Ok(instr_rng_seed(parse_reg(parts[1], syms)?))
            }
        }
        "RNG_NEXT" => {
            if parts.len() < 2 {
                return Err(anyhow!("RNG_NEXT precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("RNG_NEXT", &parts[2..], &[])?;
            }
            Ok(instr_rng_next(parse_reg(parts[1], syms)?))
        }
        "RNG_UNIFORM" => {
            // RNG_UNIFORM rD [A=a] [B=b]
            if parts.len() < 2 {
                return Err(anyhow!("RNG_UNIFORM precisa de rdest"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let (mut a, mut b) = (0.0f32, 1.0f32);
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("A=") {
                    if let Ok(n) = v.parse::<f32>() { a = n; }
                    else { return Err(anyhow!("RNG_UNIFORM A inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("B=") {
                    if let Ok(n) = v.parse::<f32>() { b = n; }
                    else { return Err(anyhow!("RNG_UNIFORM B inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("RNG_UNIFORM token desconhecido '{}' (use A=/B=)", p));
                }
            }
            Ok(instr_rng_uniform(rdest, a, b))
        }
        "RNG_NORMAL" => {
            // RNG_NORMAL rD [MEAN=m] [STD=s]
            if parts.len() < 2 {
                return Err(anyhow!("RNG_NORMAL precisa de rdest"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let (mut mean, mut std) = (0.0f32, 1.0f32);
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("MEAN=") {
                    if let Ok(n) = v.parse::<f32>() { mean = n; }
                    else { return Err(anyhow!("RNG_NORMAL MEAN inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("STD=") {
                    if let Ok(n) = v.parse::<f32>() { std = n; }
                    else { return Err(anyhow!("RNG_NORMAL STD inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("RNG_NORMAL token desconhecido '{}' (use MEAN=/STD=)", p));
                }
            }
            Ok(instr_rng_normal(rdest, mean, std))
        }
        "HASH" => {
            if parts.len() < 3 {
                return Err(anyhow!("HASH precisa de rdest, rTensor — ex: HASH r4, r0"));
            }
            if parts.len() > 3 {
                reject_unknown("HASH", &parts[3..], &[])?;
            }
            Ok(instr_hash(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "CHECKSUM" => {
            if parts.len() < 3 {
                return Err(anyhow!("CHECKSUM precisa de rdest, rTensor — ex: CHECKSUM r4, r0"));
            }
            if parts.len() > 3 {
                reject_unknown("CHECKSUM", &parts[3..], &[])?;
            }
            Ok(instr_checksum(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "HMAC" => {
            if parts.len() < 4 {
                return Err(anyhow!("HMAC precisa de rdest, rKey, rMsg — ex: HMAC r4, r0, r1"));
            }
            if parts.len() > 4 {
                reject_unknown("HMAC", &parts[4..], &[])?;
            }
            Ok(instr_hmac(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, parse_reg(parts[3], syms)?))
        }
        "CYCLES_COUNT" => {
            if parts.len() < 2 {
                return Err(anyhow!("CYCLES_COUNT precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("CYCLES_COUNT", &parts[2..], &[])?;
            }
            Ok(instr_cycles_count(parse_reg(parts[1], syms)?))
        }
        "TRACE_EVENT" => {
            if parts.len() < 3 {
                return Err(anyhow!("TRACE_EVENT precisa de rEv, rData"));
            }
            if parts.len() > 3 {
                reject_unknown("TRACE_EVENT", &parts[3..], &[])?;
            }
            Ok(instr_trace_event(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "SANITY_CHECK" => {
            // SANITY_CHECK rD, rT [, rCount]
            if parts.len() < 3 {
                return Err(anyhow!("SANITY_CHECK precisa de rdest, rTensor — ex: SANITY_CHECK r4, r0"));
            }
            // RFC-0008: 4º token, se presente, DEVE ser registrador (antes:
            // lixo silenciosamente virava 0xFF = sem contagem).
            let rcount = if parts.len() > 3 {
                parse_reg(parts[3], syms).map_err(|_| anyhow!("SANITY_CHECK 4º operando '{}' inválido (use registrador)", parts[3]))?
            } else {
                0xFF
            };
            if parts.len() > 4 {
                reject_unknown("SANITY_CHECK", &parts[4..], &[])?;
            }
            Ok(instr_sanity_check(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, rcount))
        }
        "PREEMPT_CHECK" => {
            if parts.len() < 2 {
                return Err(anyhow!("PREEMPT_CHECK precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("PREEMPT_CHECK", &parts[2..], &[])?;
            }
            Ok(instr_preempt_check(parse_reg(parts[1], syms)?))
        }
        "ASSERT" => {
            // ASSERT Rs [CODE=n]
            if parts.len() < 2 {
                return Err(anyhow!("ASSERT precisa de Rs — ex: ASSERT r1 CODE=42"));
            }
            let mut code = 0u16;
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("CODE=") {
                    // Literal u16 ou `.equ` (RFC-0037; faixa checada).
                    if let Ok(n) = v.parse::<u16>() { code = n; }
                    else if let Ok(n) = syms.resolve_const(v) {
                        code = u16::try_from(n).map_err(|_| anyhow!("ASSERT CODE inválido '{}'", p))?;
                    }
                    else { return Err(anyhow!("ASSERT CODE inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("ASSERT token desconhecido '{}' (use CODE=)", p));
                }
            }
            Ok(instr_assert(parse_reg(parts[1], syms)?, code))
        }
        "DUMP" => {
            if parts.len() > 1 {
                reject_unknown("DUMP", &parts[1..], &[])?;
            }
            Ok(instr_dump())
        }
        "YIELD" => {
            if parts.len() > 1 {
                reject_unknown("YIELD", &parts[1..], &[])?;
            }
            Ok(instr_yield())
        }
        "SET_DEADLINE" => {
            if parts.len() < 2 {
                return Err(anyhow!("SET_DEADLINE precisa de Rs"));
            }
            if parts.len() > 2 {
                reject_unknown("SET_DEADLINE", &parts[2..], &[])?;
            }
            Ok(instr_set_deadline(parse_reg(parts[1], syms)?))
        }
        "GET_DEADLINE" => {
            if parts.len() < 2 {
                return Err(anyhow!("GET_DEADLINE precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("GET_DEADLINE", &parts[2..], &[])?;
            }
            Ok(instr_get_deadline(parse_reg(parts[1], syms)?))
        }
        "PRIORITY_SET" => {
            if parts.len() < 2 {
                return Err(anyhow!("PRIORITY_SET precisa de Rs (0/1/2)"));
            }
            if parts.len() > 2 {
                reject_unknown("PRIORITY_SET", &parts[2..], &[])?;
            }
            Ok(instr_priority_set(parse_reg(parts[1], syms)?))
        }
        "PRIORITY_GET" => {
            if parts.len() < 2 {
                return Err(anyhow!("PRIORITY_GET precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("PRIORITY_GET", &parts[2..], &[])?;
            }
            Ok(instr_priority_get(parse_reg(parts[1], syms)?))
        }
        "LOCK" => {
            if parts.len() < 2 {
                return Err(anyhow!("LOCK precisa de Rs (id)"));
            }
            if parts.len() > 2 {
                reject_unknown("LOCK", &parts[2..], &[])?;
            }
            Ok(instr_lock(parse_reg(parts[1], syms)?))
        }
        "UNLOCK" => {
            if parts.len() < 2 {
                return Err(anyhow!("UNLOCK precisa de Rs (id)"));
            }
            if parts.len() > 2 {
                reject_unknown("UNLOCK", &parts[2..], &[])?;
            }
            Ok(instr_unlock(parse_reg(parts[1], syms)?))
        }
        "FENCE" => {
            if parts.len() > 1 {
                reject_unknown("FENCE", &parts[1..], &[])?;
            }
            Ok(instr_fence())
        }
        "LOADI" => {
            // LOADI rD, imm — decimal, 0x-hex ou `.equ` (RFC-0037);
            // negativos rejeitados (u128).
            if parts.len() < 3 {
                return Err(anyhow!("LOADI precisa de rdest, imediato — ex: LOADI r0, 80000000"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            // Dia 3: `@nome` = endereço `.data` (substituição em
            // assemble-time; LOADI u128 comporta qualquer endereço).
            // Erro próprio (não cai na mensagem de imediato).
            let imm = match parts[2].trim().trim_end_matches(',').strip_prefix('@') {
                Some(nm) => syms
                    .resolve_addr(nm)
                    .map_err(|e| anyhow!("LOADI {}", e))?,
                None => parse_imm_u128(parts[2], syms).map_err(|_| {
                    if parts[2].trim().starts_with("0x") || parts[2].trim().starts_with("0X") {
                        anyhow!("LOADI imediato hex inválido '{}'", parts[2])
                    } else {
                        anyhow!("LOADI imediato '{}' inválido (u128: decimal, 0x-hex ou .equ)", parts[2])
                    }
                })?,
            };
            if parts.len() > 3 {
                reject_unknown("LOADI", &parts[3..], &[])?;
            }
            Ok(instr_loadi(rdest, imm))
        }
        "MOV" => {
            if parts.len() < 3 {
                return Err(anyhow!("MOV precisa de rdest, rsrc — ex: MOV r1, r0"));
            }
            if parts.len() > 3 {
                reject_unknown("MOV", &parts[3..], &[])?;
            }
            Ok(instr_mov(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "KV_TRUNCATE" => {
            // KV_TRUNCATE Rs_len [, STREAM=sid]
            if parts.len() < 2 {
                return Err(anyhow!("KV_TRUNCATE precisa de Rs_len — ex: KV_TRUNCATE r0"));
            }
            let mut stream = 0u16;
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("STREAM=") {
                    if let Ok(n) = v.parse::<u16>() { stream = n; }
                    else { return Err(anyhow!("KV_TRUNCATE STREAM inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("KV_TRUNCATE token desconhecido '{}' (use STREAM=)", p));
                }
            }
            Ok(instr_kv_truncate(parse_reg(parts[1], syms)?, stream))
        }
        "SLICE" => {
            // SLICE rD, rT START=n LEN=n
            if parts.len() < 3 {
                return Err(anyhow!("SLICE precisa de rdest, rTensor — ex: SLICE r4, r0 START=3 LEN=3"));
            }
            let (mut start, mut len, mut has_start, mut has_len) = (0u32, 0u32, false, false);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("START=") {
                    // Literal u32 ou `.equ` (RFC-0037; faixa checada).
                    if let Ok(n) = v.parse::<u32>() { start = n; }
                    else if let Ok(n) = syms.resolve_const(v) {
                        start = u32::try_from(n).map_err(|_| anyhow!("SLICE START inválido '{}'", p))?;
                    }
                    else { return Err(anyhow!("SLICE START inválido '{}'", p)); }
                    has_start = true;
                } else if let Some(v) = up.strip_prefix("LEN=") {
                    if let Ok(n) = v.parse::<u32>() { len = n; }
                    else if let Ok(n) = syms.resolve_const(v) {
                        len = u32::try_from(n).map_err(|_| anyhow!("SLICE LEN inválido '{}'", p))?;
                    }
                    else { return Err(anyhow!("SLICE LEN inválido '{}'", p)); }
                    has_len = true;
                } else {
                    return Err(anyhow!("SLICE token desconhecido '{}' (use START=/LEN=)", p));
                }
            }
            if !has_start || !has_len {
                return Err(anyhow!("SLICE precisa de START= e LEN= — ex: SLICE r4, r0 START=3 LEN=3"));
            }
            Ok(instr_slice(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, start, len))
        }
        "ARENA_ALLOC" => {
            // ARENA_ALLOC rD, SIZE=n [ALIGN=n] [ARENA=id]
            if parts.len() < 3 {
                return Err(anyhow!("ARENA_ALLOC precisa de rdest e SIZE= — ex: ARENA_ALLOC r2, SIZE=64 ALIGN=16"));
            }
            let (mut size, mut align, mut arena, mut has_size) = (0u64, 0u32, 0u8, false);
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("SIZE=") {
                    size = v.parse::<u64>().map_err(|_| anyhow!("ARENA_ALLOC SIZE inválido '{}'", p))?;
                    has_size = true;
                } else if let Some(v) = up.strip_prefix("ALIGN=") {
                    align = v.parse::<u32>().map_err(|_| anyhow!("ARENA_ALLOC ALIGN inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("ARENA=") {
                    arena = v.parse::<u8>().map_err(|_| anyhow!("ARENA_ALLOC ARENA inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("ARENA_ALLOC token desconhecido '{}' (use SIZE=/ALIGN=/ARENA=)", p));
                }
            }
            if !has_size {
                return Err(anyhow!("ARENA_ALLOC precisa de SIZE= — ex: ARENA_ALLOC r2, SIZE=64"));
            }
            Ok(instr_arena_alloc(parse_reg(parts[1], syms)?, size, align, arena))
        }
        "ARENA_RESET" => {
            // ARENA_RESET [ARENA=id]
            let mut arena = 0u8;
            for p in &parts[1..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("ARENA=") {
                    arena = v.parse::<u8>().map_err(|_| anyhow!("ARENA_RESET ARENA inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("ARENA_RESET token desconhecido '{}' (use ARENA=)", p));
                }
            }
            Ok(instr_arena_reset(arena))
        }
        "MEMCPY" => {
            // MEMCPY rDst, rSrc [LEN=n] [SRC_OFF=n] [DST_OFF=n] [DIR=HOST]
            if parts.len() < 3 {
                return Err(anyhow!("MEMCPY precisa de rDst, rSrc — ex: MEMCPY r1, r0 LEN=16"));
            }
            let (mut len, mut src_off, mut dst_off, mut dir) = (0u64, 0u64, 0u64, MEMCPY_DIR_HOST);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("LEN=") {
                    len = v.parse::<u64>().map_err(|_| anyhow!("MEMCPY LEN inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("SRC_OFF=") {
                    src_off = v.parse::<u64>().map_err(|_| anyhow!("MEMCPY SRC_OFF inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("DST_OFF=") {
                    dst_off = v.parse::<u64>().map_err(|_| anyhow!("MEMCPY DST_OFF inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("DIR=") {
                    dir = match v {
                        "HOST" => MEMCPY_DIR_HOST,
                        "GPU" => MEMCPY_DIR_GPU,
                        "NIC" => MEMCPY_DIR_NIC,
                        _ => v.parse::<u8>().map_err(|_| anyhow!("MEMCPY DIR inválido '{}' (use HOST/GPU/NIC)", p))?,
                    };
                    if dir > MEMCPY_DIR_NIC {
                        return Err(anyhow!("MEMCPY DIR inválido '{}' (use HOST/GPU/NIC)", p));
                    }
                } else {
                    return Err(anyhow!("MEMCPY token desconhecido '{}' (use LEN=/SRC_OFF=/DST_OFF=/DIR=)", p));
                }
            }
            Ok(instr_memcpy(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, len, src_off, dst_off, dir))
        }
        "MEMSET" => {
            // MEMSET rT, PATTERN=n [LEN=n] [OFF=n]
            if parts.len() < 3 {
                return Err(anyhow!("MEMSET precisa de rTensor e PATTERN= — ex: MEMSET r1, PATTERN=0 LEN=4"));
            }
            let (mut pattern, mut len, mut off, mut has_pat) = (0u8, 0u32, 0u64, false);
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("PATTERN=") {
                    pattern = v.parse::<u8>().map_err(|_| anyhow!("MEMSET PATTERN inválido '{}' (0-255)", p))?;
                    has_pat = true;
                } else if let Some(v) = up.strip_prefix("LEN=") {
                    len = v.parse::<u32>().map_err(|_| anyhow!("MEMSET LEN inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("OFF=") {
                    off = v.parse::<u64>().map_err(|_| anyhow!("MEMSET OFF inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("MEMSET token desconhecido '{}' (use PATTERN=/LEN=/OFF=)", p));
                }
            }
            if !has_pat {
                return Err(anyhow!("MEMSET precisa de PATTERN= — ex: MEMSET r1, PATTERN=0"));
            }
            Ok(instr_memset(parse_reg(parts[1], syms)?, pattern, len, off))
        }
        "SNAPSHOT" => {
            // SNAPSHOT rD [MASK=n] (só MASK=0b111 executa; resto veta no exec)
            if parts.len() < 2 {
                return Err(anyhow!("SNAPSHOT precisa de rdest — ex: SNAPSHOT r5"));
            }
            let mut mask = SNAP_MASK_ALL;
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("MASK=") {
                    mask = v.parse::<u8>().map_err(|_| anyhow!("SNAPSHOT MASK inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("SNAPSHOT token desconhecido '{}' (use MASK=)", p));
                }
            }
            Ok(instr_snapshot(parse_reg(parts[1], syms)?, mask))
        }
        "RESTORE" => {
            // RESTORE rV (rV guarda version u64; sem chaves)
            if parts.len() != 2 {
                return Err(anyhow!("RESTORE precisa de exatamente um registrador — ex: RESTORE r5"));
            }
            Ok(instr_restore(parse_reg(parts[1], syms)?))
        }
        "PREFETCH" => {
            // PREFETCH rT [LEN=n] [OFF=n]
            if parts.len() < 2 {
                return Err(anyhow!("PREFETCH precisa de rTensor — ex: PREFETCH r0 LEN=1024"));
            }
            let (mut len, mut off) = (0u32, 0u64);
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("LEN=") {
                    len = v.parse::<u32>().map_err(|_| anyhow!("PREFETCH LEN inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("OFF=") {
                    off = v.parse::<u64>().map_err(|_| anyhow!("PREFETCH OFF inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("PREFETCH token desconhecido '{}' (use LEN=/OFF=)", p));
                }
            }
            Ok(instr_prefetch(parse_reg(parts[1], syms)?, len, off))
        }
        "RESHAPE" => {
            // RESHAPE rD, rT SHAPE=AxBxC (x-separado, 1-4 dims; helper
            // compartilhado com BROADCAST — mesma regra).
            if parts.len() < 3 {
                return Err(anyhow!("RESHAPE precisa de rdest, rTensor e SHAPE= — ex: RESHAPE r1, r0 SHAPE=1x4"));
            }
            let mut dims: Option<Vec<u32>> = None;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("SHAPE=") {
                    dims = Some(parse_shape_dims(v, "RESHAPE")?);
                } else {
                    return Err(anyhow!("RESHAPE token desconhecido '{}' (use SHAPE=AxBxC)", p));
                }
            }
            match dims {
                Some(d) => Ok(instr_reshape(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, &d)),
                None => Err(anyhow!("RESHAPE precisa de SHAPE= — ex: RESHAPE r1, r0 SHAPE=1x4")),
            }
        }
        "CONCAT" => {
            // CONCAT rD, rA, rB [AXIS=n]
            if parts.len() < 4 {
                return Err(anyhow!("CONCAT precisa de rdest, rA, rB — ex: CONCAT r2, r0, r1 AXIS=0"));
            }
            let mut axis = 0u8;
            for p in &parts[4..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("CONCAT AXIS inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("CONCAT token desconhecido '{}' (use AXIS=)", p));
                }
            }
            Ok(instr_concat(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, parse_reg(parts[3], syms)?, axis))
        }
        "CAST" => {
            // CAST rD, rT DST=F32|F16|BF16|I8|U8 (0-4 também aceitos)
            if parts.len() < 3 {
                return Err(anyhow!("CAST precisa de rdest, rTensor e DST= — ex: CAST r1, r0 DST=F16"));
            }
            let mut dst = None;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("DST=") {
                    dst = Some(match v {
                        "F32" => CAST_DST_F32,
                        "F16" => CAST_DST_F16,
                        "BF16" => CAST_DST_BF16,
                        "I8" => CAST_DST_I8,
                        "U8" => CAST_DST_U8,
                        _ => v.parse::<u8>().map_err(|_| anyhow!("CAST DST inválido '{}' (use F32/F16/BF16/I8/U8)", p))?,
                    });
                    if dst > Some(4) {
                        return Err(anyhow!("CAST DST inválido '{}' (use F32/F16/BF16/I8/U8)", p));
                    }
                } else {
                    return Err(anyhow!("CAST token desconhecido '{}' (use DST=)", p));
                }
            }
            match dst {
                Some(d) => Ok(instr_cast(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, d)),
                None => Err(anyhow!("CAST precisa de DST= — ex: CAST r1, r0 DST=F16")),
            }
        }
        "QUANTIZE" => {
            // QUANTIZE rD, rT Q=Q4_0|Q8_0 (só o par simétrico parseia;
            // Q4_K/Q6_K vetam aqui — sem soletrar o inexecutável)
            if parts.len() < 3 {
                return Err(anyhow!("QUANTIZE precisa de rdest, rTensor e Q= — ex: QUANTIZE r3, r0 Q=Q8_0"));
            }
            let mut qtype = None;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("Q=") {
                    qtype = Some(match v {
                        "Q4_0" | "4" => QUANTIZE_Q4_0,
                        "Q8_0" | "8" => QUANTIZE_Q8_0,
                        _ => return Err(anyhow!("QUANTIZE Q='{}' sem encoder (só Q4_0/Q8_0; Q4_K/Q6_K são RFC futura)", p)),
                    });
                } else {
                    return Err(anyhow!("QUANTIZE token desconhecido '{}' (use Q=)", p));
                }
            }
            match qtype {
                Some(q) => Ok(instr_quantize(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, q)),
                None => Err(anyhow!("QUANTIZE precisa de Q= — ex: QUANTIZE r3, r0 Q=Q8_0")),
            }
        }
        "DEQUANT" => {
            // DEQUANT rD, rT (sem chaves; tipo vem do meta do tensor)
            if parts.len() != 3 {
                return Err(anyhow!("DEQUANT precisa de exatamente rdest, rTensor — ex: DEQUANT r4, r3"));
            }
            Ok(instr_dequant(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "ADD_IMM" => {
            // ADD_IMM rD, rS, IMM=n (u128 decimal ou `.equ`; "-5" erra — use SUB_IMM)
            if parts.len() < 3 {
                return Err(anyhow!("ADD_IMM precisa de rdest, rSrc e IMM= — ex: ADD_IMM r1, r0 IMM=23"));
            }
            let mut imm = None;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("IMM=") {
                    imm = Some(parse_imm_dec_or_const(v, syms).map_err(|_| anyhow!("ADD_IMM IMM inválido '{}' (u128 decimal ou .equ; negativo use SUB_IMM)", p))?);
                } else {
                    return Err(anyhow!("ADD_IMM token desconhecido '{}' (use IMM=)", p));
                }
            }
            match imm {
                Some(n) => Ok(instr_add_imm(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, n)),
                None => Err(anyhow!("ADD_IMM precisa de IMM= — ex: ADD_IMM r1, r0 IMM=23")),
            }
        }
        "SUB_IMM" => {
            // SUB_IMM rD, rS, IMM=n (u128 decimal ou `.equ`; wrapping; único SUB do ISA)
            if parts.len() < 3 {
                return Err(anyhow!("SUB_IMM precisa de rdest, rSrc e IMM= — ex: SUB_IMM r2, r1 IMM=23"));
            }
            let mut imm = None;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("IMM=") {
                    imm = Some(parse_imm_dec_or_const(v, syms).map_err(|_| anyhow!("SUB_IMM IMM inválido '{}' (u128 decimal ou .equ)", p))?);
                } else {
                    return Err(anyhow!("SUB_IMM token desconhecido '{}' (use IMM=)", p));
                }
            }
            match imm {
                Some(n) => Ok(instr_sub_imm(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, n)),
                None => Err(anyhow!("SUB_IMM precisa de IMM= — ex: SUB_IMM r2, r1 IMM=23")),
            }
        }
        "STEPS" => {
            // STEPS rD (sem chaves)
            if parts.len() != 2 {
                return Err(anyhow!("STEPS precisa de exatamente um registrador — ex: STEPS r3"));
            }
            Ok(instr_steps(parse_reg(parts[1], syms)?))
        }
        "SORT" => {
            // SORT rD, rT [AXIS=n] [ORDER=ASC|DESC]
            if parts.len() < 3 {
                return Err(anyhow!("SORT precisa de rdest, rTensor — ex: SORT r1, r0 AXIS=1 ORDER=DESC"));
            }
            let (mut axis, mut order) = (0u8, SORT_ASC);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("SORT AXIS inválido '{}'", p))?;
                } else if up == "ASC" {
                    order = SORT_ASC;
                } else if up == "DESC" {
                    order = SORT_DESC;
                } else if let Some(v) = up.strip_prefix("ORDER=") {
                    order = match v {
                        "ASC" => SORT_ASC,
                        "DESC" => SORT_DESC,
                        _ => return Err(anyhow!("SORT ORDER '{}' inválido (use ASC/DESC)", p)),
                    };
                } else {
                    return Err(anyhow!("SORT token desconhecido '{}' (use AXIS=/ORDER=)", p));
                }
            }
            Ok(instr_sort(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, axis, order))
        }
        "TOPK" => {
            // TOPK rD, rT K=n [AXIS=n] [LARGEST|SMALLEST] [SORTED|UNSORTED]
            if parts.len() < 3 {
                return Err(anyhow!("TOPK precisa de rdest, rTensor e K= — ex: TOPK r2, r0 K=2 AXIS=1"));
            }
            let (mut axis, mut k, mut has_k, mut largest, mut sorted) = (0u8, 0u16, false, 1u8, 1u8);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("K=") {
                    k = v.parse::<u16>().map_err(|_| anyhow!("TOPK K inválido '{}'", p))?;
                    has_k = true;
                } else if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("TOPK AXIS inválido '{}'", p))?;
                } else if up == "LARGEST" {
                    largest = 1;
                } else if up == "SMALLEST" {
                    largest = 0;
                } else if up == "SORTED" {
                    sorted = 1;
                } else if up == "UNSORTED" {
                    sorted = 0;
                } else {
                    return Err(anyhow!("TOPK token desconhecido '{}' (use K=/AXIS=/LARGEST/SMALLEST/SORTED/UNSORTED)", p));
                }
            }
            if !has_k {
                return Err(anyhow!("TOPK precisa de K= — ex: TOPK r2, r0 K=2"));
            }
            Ok(instr_topk(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, axis, k, largest, sorted))
        }
        "ARGMAX" => {
            // ARGMAX rD, rT [AXIS=n]
            if parts.len() < 3 {
                return Err(anyhow!("ARGMAX precisa de rdest, rTensor — ex: ARGMAX r3, r0 AXIS=1"));
            }
            let mut axis = 0u8;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("ARGMAX AXIS inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("ARGMAX token desconhecido '{}' (use AXIS=)", p));
                }
            }
            Ok(instr_argmax(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, axis))
        }
        "REDUCE" => {
            // REDUCE rD, rT OP=SUM|MEAN|MAX|MIN|PROD [AXIS=n]
            if parts.len() < 3 {
                return Err(anyhow!("REDUCE precisa de rdest, rTensor e OP= — ex: REDUCE r4, r0 OP=SUM"));
            }
            let (mut op, mut has_op, mut axis) = (0u8, false, 0xFFu8);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("OP=") {
                    op = match v {
                        "SUM" | "0" => REDUCE_SUM,
                        "MEAN" | "1" => REDUCE_MEAN,
                        "MAX" | "2" => REDUCE_MAX,
                        "MIN" | "3" => REDUCE_MIN,
                        "PROD" | "4" => REDUCE_PROD,
                        _ => return Err(anyhow!("REDUCE OP '{}' inválido (use SUM/MEAN/MAX/MIN/PROD)", p)),
                    };
                    has_op = true;
                } else if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("REDUCE AXIS inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("REDUCE token desconhecido '{}' (use OP=/AXIS=)", p));
                }
            }
            if !has_op {
                return Err(anyhow!("REDUCE precisa de OP= — ex: REDUCE r4, r0 OP=SUM"));
            }
            Ok(instr_reduce(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, op, axis))
        }
        "BROADCAST" => {
            // BROADCAST rD, rT SHAPE=AxBxC (mesma regra do RESHAPE)
            if parts.len() < 3 {
                return Err(anyhow!("BROADCAST precisa de rdest, rTensor e SHAPE= — ex: BROADCAST r5, r4 SHAPE=2x2"));
            }
            let mut dims: Option<Vec<u32>> = None;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("SHAPE=") {
                    dims = Some(parse_shape_dims(v, "BROADCAST")?);
                } else {
                    return Err(anyhow!("BROADCAST token desconhecido '{}' (use SHAPE=AxBxC)", p));
                }
            }
            match dims {
                Some(d) => Ok(instr_broadcast(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, &d)),
                None => Err(anyhow!("BROADCAST precisa de SHAPE= — ex: BROADCAST r5, r4 SHAPE=2x2")),
            }
        }
        "PAD" => {
            // PAD rD, rT VALUE=x [AXIS=n] [BEFORE=n] [AFTER=n]
            if parts.len() < 3 {
                return Err(anyhow!("PAD precisa de rdest, rTensor e VALUE= — ex: PAD r6, r0 VALUE=0 AXIS=1 BEFORE=1 AFTER=1"));
            }
            let (mut value, mut has_value, mut axis, mut before, mut after) = (0.0f32, false, 0u8, 0u32, 0u32);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("VALUE=") {
                    value = v.parse::<f32>().map_err(|_| anyhow!("PAD VALUE inválido '{}'", p))?;
                    has_value = true;
                } else if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("PAD AXIS inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("BEFORE=") {
                    before = v.parse::<u32>().map_err(|_| anyhow!("PAD BEFORE inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("AFTER=") {
                    after = v.parse::<u32>().map_err(|_| anyhow!("PAD AFTER inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("PAD token desconhecido '{}' (use VALUE=/AXIS=/BEFORE=/AFTER=)", p));
                }
            }
            if !has_value {
                return Err(anyhow!("PAD precisa de VALUE= — ex: PAD r6, r0 VALUE=0 BEFORE=1 AFTER=1"));
            }
            Ok(instr_pad(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, value, axis, before, after))
        }
        "TILE" => {
            // TILE rD, rT REPS=n [AXIS=n]
            if parts.len() < 3 {
                return Err(anyhow!("TILE precisa de rdest, rTensor e REPS= — ex: TILE r7, r0 REPS=2 AXIS=0"));
            }
            let (mut reps, mut has_reps, mut axis) = (0u32, false, 0u8);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("REPS=") {
                    reps = v.parse::<u32>().map_err(|_| anyhow!("TILE REPS inválido '{}'", p))?;
                    has_reps = true;
                } else if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("TILE AXIS inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("TILE token desconhecido '{}' (use REPS=/AXIS=)", p));
                }
            }
            if !has_reps {
                return Err(anyhow!("TILE precisa de REPS= — ex: TILE r7, r0 REPS=2"));
            }
            Ok(instr_tile(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, reps, axis))
        }
        "TRANSPOSE" => {
            // TRANSPOSE rD, rT AXES=AxBxC (x-separado; vírgula quebraria o
            // split do assembler — por isso x, como SHAPE=)
            if parts.len() < 3 {
                return Err(anyhow!("TRANSPOSE precisa de rdest, rTensor e AXES= — ex: TRANSPOSE r8, r0 AXES=1x0"));
            }
            let mut perm: Option<Vec<u8>> = None;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("AXES=") {
                    let mut parsed = Vec::new();
                    for a in v.split(|c| c == 'x' || c == 'X') {
                        parsed.push(a.parse::<u8>().map_err(|_| anyhow!("TRANSPOSE AXES inválido '{}' (use AxBxC)", p))?);
                    }
                    if parsed.is_empty() || parsed.len() > 4 {
                        return Err(anyhow!("TRANSPOSE AXES '{}' precisa de 1-4 eixos", p));
                    }
                    perm = Some(parsed);
                } else {
                    return Err(anyhow!("TRANSPOSE token desconhecido '{}' (use AXES=AxBxC)", p));
                }
            }
            match perm {
                Some(pm) => Ok(instr_transpose(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, &pm)),
                None => Err(anyhow!("TRANSPOSE precisa de AXES= — ex: TRANSPOSE r8, r0 AXES=1x0")),
            }
        }
        "SOFTMAX" => {
            // SOFTMAX rD, rT [AXIS=n] [TEMP=x] (defaults: último eixo, 1.0)
            if parts.len() < 3 {
                return Err(anyhow!("SOFTMAX precisa de rdest, rTensor — ex: SOFTMAX r7, r0 AXIS=1"));
            }
            let (mut axis, mut temp) = (0xFFu8, 1.0f32);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("AXIS=") {
                    axis = v.parse::<u8>().map_err(|_| anyhow!("SOFTMAX AXIS inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("TEMP=") {
                    temp = v.parse::<f32>().map_err(|_| anyhow!("SOFTMAX TEMP inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("SOFTMAX token desconhecido '{}' (use AXIS=/TEMP=)", p));
                }
            }
            Ok(instr_softmax(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, axis, temp))
        }
        "GELU" => {
            if parts.len() != 3 {
                return Err(anyhow!("GELU precisa de exatamente rdest, rTensor — ex: GELU r4, r0"));
            }
            Ok(instr_gelu(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "SIGMOID" => {
            if parts.len() != 3 {
                return Err(anyhow!("SIGMOID precisa de exatamente rdest, rTensor — ex: SIGMOID r1, r0"));
            }
            Ok(instr_sigmoid(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "TANH" => {
            if parts.len() != 3 {
                return Err(anyhow!("TANH precisa de exatamente rdest, rTensor — ex: TANH r2, r0"));
            }
            Ok(instr_tanh(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "RELU" => {
            if parts.len() != 3 {
                return Err(anyhow!("RELU precisa de exatamente rdest, rTensor — ex: RELU r3, r0"));
            }
            Ok(instr_relu(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "EXP" => {
            if parts.len() != 3 {
                return Err(anyhow!("EXP precisa de exatamente rdest, rTensor — ex: EXP r5, r0"));
            }
            Ok(instr_exp(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "LOG" => {
            if parts.len() != 3 {
                return Err(anyhow!("LOG precisa de exatamente rdest, rTensor — ex: LOG r6, r5"));
            }
            Ok(instr_log(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?))
        }
        "CLIP" => {
            // CLIP rD, rT MIN=x MAX=x (ambos exigidos: sem default mudo)
            if parts.len() < 3 {
                return Err(anyhow!("CLIP precisa de rdest, rTensor, MIN= e MAX= — ex: CLIP r8, r0 MIN=0 MAX=0.4"));
            }
            let (mut min, mut max, mut has_min, mut has_max) = (0.0f32, 0.0f32, false, false);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("MIN=") {
                    min = v.parse::<f32>().map_err(|_| anyhow!("CLIP MIN inválido '{}'", p))?;
                    has_min = true;
                } else if let Some(v) = up.strip_prefix("MAX=") {
                    max = v.parse::<f32>().map_err(|_| anyhow!("CLIP MAX inválido '{}'", p))?;
                    has_max = true;
                } else {
                    return Err(anyhow!("CLIP token desconhecido '{}' (use MIN=/MAX=)", p));
                }
            }
            if !has_min || !has_max {
                return Err(anyhow!("CLIP precisa de MIN= e MAX= — ex: CLIP r8, r0 MIN=0 MAX=0.4"));
            }
            Ok(instr_clip(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, min, max))
        }
        "KV_COMPRESS" => {
            // KV_COMPRESS SINK=n WINDOW=n [MODE=SINK_WINDOW] [STREAM=0]
            // (sem registradores; DUMP-precedente: operandos ignorados N/A)
            let (mut sink, mut window, mut has_sink, mut has_window) = (0u16, 0u16, false, false);
            let (mut mode, mut stream) = (KVCOMP_MODE_SINK_WINDOW, 0u16);
            for p in &parts[1..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("SINK=") {
                    sink = v.parse::<u16>().map_err(|_| anyhow!("KV_COMPRESS SINK inválido '{}'", p))?;
                    has_sink = true;
                } else if let Some(v) = up.strip_prefix("WINDOW=") {
                    window = v.parse::<u16>().map_err(|_| anyhow!("KV_COMPRESS WINDOW inválido '{}'", p))?;
                    has_window = true;
                } else if let Some(v) = up.strip_prefix("MODE=") {
                    mode = match v {
                        "SINK_WINDOW" | "0" => KVCOMP_MODE_SINK_WINDOW,
                        _ => return Err(anyhow!("KV_COMPRESS MODE '{}' inválido (só SINK_WINDOW)", p)),
                    };
                } else if let Some(v) = up.strip_prefix("STREAM=") {
                    stream = v.parse::<u16>().map_err(|_| anyhow!("KV_COMPRESS STREAM inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("KV_COMPRESS token desconhecido '{}' (use SINK=/WINDOW=/MODE=/STREAM=)", p));
                }
            }
            if !has_sink || !has_window {
                return Err(anyhow!("KV_COMPRESS precisa de SINK= e WINDOW= — ex: KV_COMPRESS SINK=2 WINDOW=3"));
            }
            Ok(instr_kv_compress(sink, window, mode, stream))
        }
        "FLASH_ATTN" => {
            // FLASH_ATTN rD, rQ, rK, rV [BLOCK=n] (0/default 32)
            if parts.len() < 5 {
                return Err(anyhow!("FLASH_ATTN precisa de rdest, rQ, rK, rV — ex: FLASH_ATTN r4, r0, r1, r2 BLOCK=2"));
            }
            let mut block = 0u16;
            for p in &parts[5..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("BLOCK=") {
                    block = v.parse::<u16>().map_err(|_| anyhow!("FLASH_ATTN BLOCK inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("FLASH_ATTN token desconhecido '{}' (use BLOCK=)", p));
                }
            }
            Ok(instr_flash_attn(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, parse_reg(parts[3], syms)?, parse_reg(parts[4], syms)?, block))
        }
        "ATTN_SPARSE" => {
            // ATTN_SPARSE rD, rQ, rK, rV [TOPK=n] [METRIC=DOT]
            // (só DOT parseia — sem soletrar métrica inexecutável)
            if parts.len() < 5 {
                return Err(anyhow!("ATTN_SPARSE precisa de rdest, rQ, rK, rV — ex: ATTN_SPARSE r3, r0, r1, r2 TOPK=2"));
            }
            let (mut topk, mut metric) = (0u16, SPARSE_METRIC_DOT);
            for p in &parts[5..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("TOPK=") {
                    topk = v.parse::<u16>().map_err(|_| anyhow!("ATTN_SPARSE TOPK inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("METRIC=") {
                    metric = match v {
                        "DOT" | "0" => SPARSE_METRIC_DOT,
                        _ => return Err(anyhow!("ATTN_SPARSE METRIC '{}' sem implementação (só DOT)", p)),
                    };
                } else {
                    return Err(anyhow!("ATTN_SPARSE token desconhecido '{}' (use TOPK=/METRIC=)", p));
                }
            }
            Ok(instr_attn_sparse(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, parse_reg(parts[3], syms)?, parse_reg(parts[4], syms)?, metric, topk))
        }
        "VAD_DETECT" => {
            // VAD_DETECT rD, rT [MODE=ENERGY|ZCR] (ML parseia, exec veta)
            if parts.len() < 3 {
                return Err(anyhow!("VAD_DETECT precisa de rdest, rTensor — ex: VAD_DETECT r1, r0 MODE=ENERGY"));
            }
            let mut mode = VAD_MODE_ENERGY;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("MODE=") {
                    mode = match v {
                        "ENERGY" | "0" => VAD_MODE_ENERGY,
                        "ZCR" | "1" => VAD_MODE_ZCR,
                        "ML" | "2" => VAD_MODE_ML,
                        _ => return Err(anyhow!("VAD_DETECT MODE '{}' inválido (use ENERGY/ZCR)", p)),
                    };
                } else {
                    return Err(anyhow!("VAD_DETECT token desconhecido '{}' (use MODE=)", p));
                }
            }
            Ok(instr_vad_detect(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, mode))
        }
        "STREAM_MERGE" => {
            // STREAM_MERGE rD, rA, rB GAIN=x [N_STREAMS=n]
            if parts.len() < 4 {
                return Err(anyhow!("STREAM_MERGE precisa de rdest, rA, rB e GAIN= — ex: STREAM_MERGE r6, r5, r5 GAIN=0.5"));
            }
            let (mut gain, mut has_gain, mut n_streams) = (0.0f32, false, 0u16);
            for p in &parts[4..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("GAIN=") {
                    gain = v.parse::<f32>().map_err(|_| anyhow!("STREAM_MERGE GAIN inválido '{}'", p))?;
                    has_gain = true;
                } else if let Some(v) = up.strip_prefix("N_STREAMS=") {
                    n_streams = v.parse::<u16>().map_err(|_| anyhow!("STREAM_MERGE N_STREAMS inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("STREAM_MERGE token desconhecido '{}' (use GAIN=/N_STREAMS=)", p));
                }
            }
            if !has_gain {
                return Err(anyhow!("STREAM_MERGE precisa de GAIN= — ex: STREAM_MERGE r6, r5, r5 GAIN=0.5"));
            }
            Ok(instr_stream_merge(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, parse_reg(parts[3], syms)?, gain, n_streams))
        }
        "AUDIO_RESAMPLE" => {
            // AUDIO_RESAMPLE rD, rT SRC=n DST=n (ambos exigidos)
            if parts.len() < 3 {
                return Err(anyhow!("AUDIO_RESAMPLE precisa de rdest, rTensor, SRC= e DST= — ex: AUDIO_RESAMPLE r2, r0 SRC=24000 DST=16000"));
            }
            let (mut src, mut dst, mut has_src, mut has_dst) = (0u32, 0u32, false, false);
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("SRC=") {
                    src = v.parse::<u32>().map_err(|_| anyhow!("AUDIO_RESAMPLE SRC inválido '{}'", p))?;
                    has_src = true;
                } else if let Some(v) = up.strip_prefix("DST=") {
                    dst = v.parse::<u32>().map_err(|_| anyhow!("AUDIO_RESAMPLE DST inválido '{}'", p))?;
                    has_dst = true;
                } else {
                    return Err(anyhow!("AUDIO_RESAMPLE token desconhecido '{}' (use SRC=/DST=)", p));
                }
            }
            if !has_src || !has_dst {
                return Err(anyhow!("AUDIO_RESAMPLE precisa de SRC= e DST= — ex: AUDIO_RESAMPLE r2, r0 SRC=24000 DST=16000"));
            }
            Ok(instr_audio_resample(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, src, dst))
        }
        "AUDIO_FILTER" => {
            // AUDIO_FILTER rD, rX, rB [MODE=FIR] (IIR parseia, exec veta)
            if parts.len() < 4 {
                return Err(anyhow!("AUDIO_FILTER precisa de rdest, rX, rB — ex: AUDIO_FILTER r5, r3, r4 MODE=FIR"));
            }
            let mut mode = FILTER_MODE_FIR;
            for p in &parts[4..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("MODE=") {
                    mode = match v {
                        "FIR" | "0" => FILTER_MODE_FIR,
                        "IIR" | "1" => FILTER_MODE_IIR,
                        _ => return Err(anyhow!("AUDIO_FILTER MODE '{}' inválido (use FIR)", p)),
                    };
                } else {
                    return Err(anyhow!("AUDIO_FILTER token desconhecido '{}' (use MODE=)", p));
                }
            }
            Ok(instr_audio_filter(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, parse_reg(parts[3], syms)?, mode))
        }
        "AUDIO_WINDOW" => {
            // AUDIO_WINDOW rD, rT [TYPE=HANN|HAMMING]
            if parts.len() < 3 {
                return Err(anyhow!("AUDIO_WINDOW precisa de rdest, rTensor — ex: AUDIO_WINDOW r3, r2 TYPE=HANN"));
            }
            let mut wtype = WINDOW_HANN;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("TYPE=") {
                    wtype = match v {
                        "HANN" | "0" => WINDOW_HANN,
                        "HAMMING" | "1" => WINDOW_HAMMING,
                        _ => return Err(anyhow!("AUDIO_WINDOW TYPE '{}' inválido (use HANN/HAMMING)", p)),
                    };
                } else {
                    return Err(anyhow!("AUDIO_WINDOW token desconhecido '{}' (use TYPE=)", p));
                }
            }
            Ok(instr_audio_window(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, wtype))
        }
        "DEPFORMER" => {
            // DEPFORMER rD, rX, rW [STREAM=] [LAYER=] [NCB=] [NHEADS=]
            // [LEVELS=] [CONTEXT=] [TEMP=] [TOPK=] (tudo default 0)
            if parts.len() < 4 {
                return Err(anyhow!("DEPFORMER precisa de rdest, rX, rW — ex: DEPFORMER rD, rX, rW LAYER=0"));
            }
            let (mut stream, mut layer) = (0u8, 0u8);
            let (mut ncb, mut nheads, mut levels, mut context, mut topk) = (0u16, 0u16, 0u16, 0u16, 0u16);
            let mut temp = 1.0f32;
            for p in &parts[4..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("STREAM=") {
                    stream = v.parse::<u8>().map_err(|_| anyhow!("DEPFORMER STREAM inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("LAYER=") {
                    layer = v.parse::<u8>().map_err(|_| anyhow!("DEPFORMER LAYER inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("NCB=") {
                    ncb = v.parse::<u16>().map_err(|_| anyhow!("DEPFORMER NCB inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("NHEADS=") {
                    nheads = v.parse::<u16>().map_err(|_| anyhow!("DEPFORMER NHEADS inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("LEVELS=") {
                    levels = v.parse::<u16>().map_err(|_| anyhow!("DEPFORMER LEVELS inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("CONTEXT=") {
                    context = v.parse::<u16>().map_err(|_| anyhow!("DEPFORMER CONTEXT inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("TEMP=") {
                    temp = v.parse::<f32>().map_err(|_| anyhow!("DEPFORMER TEMP inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("TOPK=") {
                    topk = v.parse::<u16>().map_err(|_| anyhow!("DEPFORMER TOPK inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("DEPFORMER token desconhecido '{}' (use STREAM=/LAYER=/NCB=/NHEADS=/LEVELS=/CONTEXT=/TEMP=/TOPK=)", p));
                }
            }
            Ok(instr_depformer(parse_reg(parts[1], syms)?, parse_reg(parts[2], syms)?, parse_reg(parts[3], syms)?, stream, layer, ncb, nheads, levels, context, temp, topk))
        }
        "CALL" => {
            // CALL LABEL (espelha JUMP: rótulo resolvido, resto veta).
            if parts.len() < 2 {
                return Err(anyhow!("CALL precisa de rótulo alvo — ex: CALL ADD_TWO"));
            }
            let label = parts[1].to_ascii_uppercase();
            let target_pc = *labels.get(&label)
                .ok_or_else(|| anyhow!("rótulo '{}' não encontrado para CALL", label))?;
            if parts.len() > 2 {
                reject_unknown("CALL", &parts[2..], &[])?;
            }
            Ok(instr_call(target_pc))
        }
        "RET" => {
            // RET (sem operandos; resto veta).
            if !parts[1..].is_empty() {
                reject_unknown("RET", &parts[1..], &[])?;
            }
            Ok(instr_ret())
        }
        "REMOTE_SPAWN" => {
            // REMOTE_SPAWN rD NODE=n ENTRY=label|pc PRI=GREEN|BLUE|RED|0|1|2
            // (NODE textual => Err: tabela de roteamento é F3, sem chute.)
            if parts.len() < 2 {
                return Err(anyhow!("REMOTE_SPAWN precisa de rdest — ex: REMOTE_SPAWN r3 NODE=0 ENTRY=MAIN GREEN"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let (mut node, mut entry, mut prio) = (0u32, 0u64, 0u8);
            let mut has_entry = false;
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("NODE=") {
                    node = v.parse::<u32>().map_err(|_| anyhow!("REMOTE_SPAWN NODE '{}' inválido (use id numérico; nomes são F3)", p))?;
                } else if let Some(v) = up.strip_prefix("ENTRY=") {
                    if let Some(&pc) = labels.get(v) {
                        entry = pc as u64;
                    } else if let Ok(n) = v.parse::<u64>() {
                        entry = n;
                    } else {
                        return Err(anyhow!("REMOTE_SPAWN ENTRY '{}' inválido (use label ou pc)", p));
                    }
                    has_entry = true;
                } else if let Some(v) = up.strip_prefix("PRI=") {
                    prio = match v {
                        "GREEN" | "0" => 0,
                        "BLUE" | "1" => 1,
                        "RED" | "2" => 2,
                        _ => return Err(anyhow!("REMOTE_SPAWN PRI inválida '{}'", p)),
                    };
                } else if matches!(up.as_str(), "GREEN" | "BLUE" | "RED" | "0" | "1" | "2") {
                    prio = match up.as_str() {
                        "BLUE" | "1" => 1,
                        "RED" | "2" => 2,
                        _ => 0,
                    };
                } else {
                    return Err(anyhow!("REMOTE_SPAWN token desconhecido '{}' (use NODE=/ENTRY=/PRI=)", p));
                }
            }
            if !has_entry {
                return Err(anyhow!("REMOTE_SPAWN precisa de ENTRY="));
            }
            Ok(instr_remote_spawn(rdest, node, entry, prio))
        }
        "SIGNAL" => {
            // SIGNAL rD KIND=ABORT|FORK_REQ|HALT|PING|0..3 NODE=n CTX=n SEQ=n
            if parts.len() < 2 {
                return Err(anyhow!("SIGNAL precisa de rdest — ex: SIGNAL r2 KIND=PING NODE=0 CTX=1"));
            }
            let rdest = parse_reg(parts[1], syms)?;
            let (mut kind, mut node, mut ctx, mut seq) = (SIGNAL_KIND_PING, 0u32, 0u64, 0u64);
            for p in &parts[2..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("KIND=") {
                    kind = match v {
                        "ABORT" | "0" => SIGNAL_KIND_ABORT,
                        "FORK_REQ" | "1" => SIGNAL_KIND_FORK_REQ,
                        "HALT" | "KILL" | "2" => SIGNAL_KIND_HALT,
                        "PING" | "3" => SIGNAL_KIND_PING,
                        _ => return Err(anyhow!("SIGNAL KIND inválido '{}'", p)),
                    };
                } else if let Some(v) = up.strip_prefix("NODE=") {
                    node = v.parse::<u32>().map_err(|_| anyhow!("SIGNAL NODE '{}' inválido (use id numérico)", p))?;
                } else if let Some(v) = up.strip_prefix("CTX=") {
                    ctx = v.parse::<u64>().map_err(|_| anyhow!("SIGNAL CTX '{}' inválido (use id numérico)", p))?;
                } else if let Some(v) = up.strip_prefix("SEQ=") {
                    seq = v.parse::<u64>().map_err(|_| anyhow!("SIGNAL SEQ inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("SIGNAL token desconhecido '{}' (use KIND=/NODE=/CTX=/SEQ=)", p));
                }
            }
            Ok(instr_signal(rdest, kind, node, ctx, seq))
        }
        "SEND_TENSOR" => {
            // SEND_TENSOR rS [, rD] [NODE=n] [OFF=n] [LEN=n] [COPY|MOVE]
            if parts.len() < 2 {
                return Err(anyhow!("SEND_TENSOR precisa de rSrc — ex: SEND_TENSOR r5 NODE=0 LEN=64"));
            }
            let rsrc = parse_reg(parts[1], syms)?;
            let (mut rdst, mut node, mut off, mut len, mut mode) = (0xFF, 0u32, 0u64, 0u32, SEND_MODE_COPY);
            let mut rpos = 2;
            if parts.len() > rpos {
                if let Ok(r) = parse_reg(parts[rpos], syms) {
                    rdst = r;
                    rpos += 1;
                }
            }
            for p in &parts[rpos..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("NODE=") {
                    node = v.parse::<u32>().map_err(|_| anyhow!("SEND_TENSOR NODE '{}' inválido", p))?;
                } else if let Some(v) = up.strip_prefix("OFF=") {
                    off = v.parse::<u64>().map_err(|_| anyhow!("SEND_TENSOR OFF inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("LEN=") {
                    len = v.parse::<u32>().map_err(|_| anyhow!("SEND_TENSOR LEN inválido '{}'", p))?;
                } else if up == "MOVE" {
                    mode = SEND_MODE_MOVE;
                } else if up == "COPY" {
                    mode = SEND_MODE_COPY;
                } else {
                    return Err(anyhow!("SEND_TENSOR token desconhecido '{}' (use rDst/NODE=/OFF=/LEN=/COPY/MOVE)", p));
                }
            }
            Ok(instr_send_tensor(rsrc, rdst, node, off, len, mode))
        }
        "BARRIER" => {
            // BARRIER id= EXPECT= TIMEOUT= [EPOCH=]
            let (mut id, mut exp, mut to, mut epoch, mut has_id, mut has_exp) = (0u32, 0u16, 0u16, 0u32, false, false);
            if parts.len() < 2 {
                return Err(anyhow!("BARRIER precisa de id= e EXPECT= — ex: BARRIER id=7 EXPECT=2 TIMEOUT=500"));
            }
            for p in parts[1..].iter() {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("ID=") {
                    id = v.parse::<u32>().map_err(|_| anyhow!("BARRIER ID inválido '{}'", p))?;
                    has_id = true;
                } else if let Some(v) = up.strip_prefix("EXPECT=") {
                    exp = v.parse::<u16>().map_err(|_| anyhow!("BARRIER EXPECT inválido '{}'", p))?;
                    has_exp = true;
                } else if let Some(v) = up.strip_prefix("TIMEOUT=") {
                    to = v.parse::<u16>().map_err(|_| anyhow!("BARRIER TIMEOUT inválido '{}'", p))?;
                } else if let Some(v) = up.strip_prefix("EPOCH=") {
                    epoch = v.parse::<u32>().map_err(|_| anyhow!("BARRIER EPOCH inválido '{}'", p))?;
                } else {
                    return Err(anyhow!("BARRIER token desconhecido '{}' (use id=/EXPECT=/TIMEOUT=/EPOCH=)", p));
                }
            }
            if !has_id || !has_exp {
                return Err(anyhow!("BARRIER precisa de id= e EXPECT= — ex: BARRIER id=7 EXPECT=2 TIMEOUT=500"));
            }
            Ok(instr_barrier(id, exp, to, epoch))
        }
        "DENOISE_STEP" => {
            // DENOISE_STEP rD, rX, rE [, rS] [ALPHA=a] [BETA=b] [SIGMA=s] [T=t]
            if parts.len() < 4 {
                return Err(anyhow!("DENOISE_STEP precisa de rdest, rX, rEps — ex: DENOISE_STEP r4, r0, r1 ALPHA=0.98 BETA=0.02 SIGMA=0.0 T=500"));
            }
            let (mut alpha, mut beta, mut sigma, mut t) = (0.98f32, 0.02f32, 0.0f32, 500u32);
            let mut rpos = 4;
            // 4º reg opcional (schedule futuro; exec veta salvo 0xFF).
            let mut rsched = 0xFF;
            if parts.len() > rpos {
                if let Ok(r) = parse_reg(parts[rpos], syms) {
                    let up = parts[rpos].to_ascii_uppercase();
                    // Registrador de verdade, não KV (KV contém '=').
                    if !up.contains('=') {
                        rsched = r;
                        rpos += 1;
                    }
                }
            }
            for p in &parts[rpos..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("ALPHA=") {
                    if let Ok(n) = v.parse::<f32>() { alpha = n; }
                    else { return Err(anyhow!("DENOISE_STEP ALPHA inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("BETA=") {
                    if let Ok(n) = v.parse::<f32>() { beta = n; }
                    else { return Err(anyhow!("DENOISE_STEP BETA inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("SIGMA=") {
                    if let Ok(n) = v.parse::<f32>() { sigma = n; }
                    else { return Err(anyhow!("DENOISE_STEP SIGMA inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("T=") {
                    if let Ok(n) = v.parse::<u32>() { t = n; }
                    else { return Err(anyhow!("DENOISE_STEP T inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("DENOISE_STEP token desconhecido '{}' (use ALPHA=/BETA=/SIGMA=/T=)", p));
                }
            }
            Ok(instr_denoise_step(
                parse_reg(parts[1], syms)?,
                parse_reg(parts[2], syms)?,
                parse_reg(parts[3], syms)?,
                rsched,
                alpha,
                beta,
                sigma,
                t,
            ))
        }
        "ODE_STEP" => {
            // ODE_STEP rD, rX, rU [, rWb] [DT=t] [METHOD=...] [LAYER=n]
            if parts.len() < 4 {
                return Err(anyhow!("ODE_STEP precisa de rdest, rX, rU — ex: ODE_STEP r2, r0, r1 DT=0.01 METHOD=RK2"));
            }
            let (mut dt, mut method, mut layer) = (0.01f32, ODE_METHOD_EULER, 0u8);
            let mut rpos = 4;
            // 4º reg opcional (pack Wb); KV nunca é reg válido aqui.
            let mut rwb = 0xFF;
            if parts.len() > rpos {
                if let Ok(r) = parse_reg(parts[rpos], syms) {
                    rwb = r;
                    rpos += 1;
                }
            }
            for p in &parts[rpos..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("DT=") {
                    if let Ok(n) = v.parse::<f32>() { dt = n; }
                    else { return Err(anyhow!("ODE_STEP DT inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("METHOD=") {
                    method = match v {
                        "EULER" => ODE_METHOD_EULER,
                        "RK2" => ODE_METHOD_RK2,
                        "RK4" => ODE_METHOD_RK4,
                        _ => return Err(anyhow!("ODE_STEP METHOD inválido '{}' (use EULER/RK2/RK4)", p)),
                    };
                } else if let Some(v) = up.strip_prefix("LAYER=") {
                    if let Ok(n) = v.parse::<u8>() { layer = n; }
                    else { return Err(anyhow!("ODE_STEP LAYER inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("ODE_STEP token desconhecido '{}' (use DT=/METHOD=/LAYER=)", p));
                }
            }
            Ok(instr_ode_step(
                parse_reg(parts[1], syms)?,
                parse_reg(parts[2], syms)?,
                parse_reg(parts[3], syms)?,
                rwb,
                dt,
                method,
                layer,
            ))
        }
        "SPIKE_STEP" => {
            // SPIKE_STEP rD, rV, rI [, rPack] [THRESH=] [DECAY=] [RESET=] [LAYER=] [REFRACT=]
            if parts.len() < 4 {
                return Err(anyhow!("SPIKE_STEP precisa de rdest, rV, rI — ex: SPIKE_STEP r3, r1, r2 THRESH=1.0 DECAY=0.9"));
            }
            let (mut thresh, mut decay, mut reset, mut layer, mut refr) =
                (1.0f32, 0.9f32, 0.0f32, 0u8, 2u8);
            let mut rpos = 4;
            // 4º reg opcional (pack); KV nunca é reg válido aqui.
            let mut rpack = 0xFF;
            if parts.len() > rpos {
                if let Ok(r) = parse_reg(parts[rpos], syms) {
                    rpack = r;
                    rpos += 1;
                }
            }
            for p in &parts[rpos..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("THRESH=") {
                    if let Ok(n) = v.parse::<f32>() { thresh = n; }
                    else { return Err(anyhow!("SPIKE_STEP THRESH inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("DECAY=") {
                    if let Ok(n) = v.parse::<f32>() { decay = n; }
                    else { return Err(anyhow!("SPIKE_STEP DECAY inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("RESET=") {
                    if let Ok(n) = v.parse::<f32>() { reset = n; }
                    else { return Err(anyhow!("SPIKE_STEP RESET inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("LAYER=") {
                    if let Ok(n) = v.parse::<u8>() { layer = n; }
                    else { return Err(anyhow!("SPIKE_STEP LAYER inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("REFRACT=") {
                    if let Ok(n) = v.parse::<u8>() { refr = n; }
                    else { return Err(anyhow!("SPIKE_STEP REFRACT inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("SPIKE_STEP token desconhecido '{}' (use THRESH=/DECAY=/RESET=/LAYER=/REFRACT=)", p));
                }
            }
            Ok(instr_spike_step(
                parse_reg(parts[1], syms)?,
                parse_reg(parts[2], syms)?,
                parse_reg(parts[3], syms)?,
                rpack,
                thresh,
                decay,
                reset,
                layer,
                refr,
            ))
        }
        "CONV" => {
            // CONV rD, rX, rW [, rB] [STRIDE=n] [PAD=n] [DILATION=n]
            // [GROUPS=n] [ACT=NONE|SILU|RELU]
            if parts.len() < 4 {
                return Err(anyhow!("CONV precisa de rdest, rX, rW — ex: CONV r4, r0, r1 STRIDE=1 PAD=1"));
            }
            let (mut stride, mut pad, mut dilation, mut groups, mut act) =
                (1u16, 0u16, 1u16, 1u8, CONV_ACT_NONE);
            let mut rpos = 4;
            // 4º reg opcional (bias); KV nunca é reg válido aqui.
            let mut rbias = 0xFF;
            if parts.len() > rpos {
                if let Ok(r) = parse_reg(parts[rpos], syms) {
                    rbias = r;
                    rpos += 1;
                }
            }
            for p in &parts[rpos..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("STRIDE=") {
                    if let Ok(n) = v.parse::<u16>() { stride = n; }
                    else { return Err(anyhow!("CONV STRIDE inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("PAD=") {
                    if let Ok(n) = v.parse::<u16>() { pad = n; }
                    else { return Err(anyhow!("CONV PAD inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("DILATION=") {
                    if let Ok(n) = v.parse::<u16>() { dilation = n; }
                    else { return Err(anyhow!("CONV DILATION inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("GROUPS=") {
                    if let Ok(n) = v.parse::<u8>() { groups = n; }
                    else { return Err(anyhow!("CONV GROUPS inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("ACT=") {
                    act = match v {
                        "NONE" => CONV_ACT_NONE,
                        "SILU" => CONV_ACT_SILU,
                        "RELU" => CONV_ACT_RELU,
                        _ => return Err(anyhow!("CONV ACT inválido '{}' (use NONE/SILU/RELU)", p)),
                    };
                } else {
                    return Err(anyhow!("CONV token desconhecido '{}' (use STRIDE=/PAD=/DILATION=/GROUPS=/ACT=)", p));
                }
            }
            Ok(instr_conv(
                parse_reg(parts[1], syms)?,
                parse_reg(parts[2], syms)?,
                parse_reg(parts[3], syms)?,
                rbias,
                stride,
                pad,
                dilation,
                groups,
                act,
            ))
        }
        "FOREST" => {
            // FOREST rD, rF, rT, rL [TREES=n] [DEPTH=d] [MODE=VOTE|MEAN]
            if parts.len() < 5 {
                return Err(anyhow!("FOREST precisa de rdest, rFeat, rTable, rLeaves — ex: FOREST r4, r0, r1, r2 TREES=8 DEPTH=6"));
            }
            let (mut n_trees, mut depth, mut mode) = (1u16, 1u16, FOREST_MODE_VOTE);
            for p in &parts[5..] {
                let up = p.to_ascii_uppercase();
                if let Some(v) = up.strip_prefix("TREES=") {
                    // Literal u16 ou `.equ` (RFC-0037; faixa checada abaixo).
                    if let Ok(n) = v.parse::<u16>() { n_trees = n; }
                    else if let Ok(n) = syms.resolve_const(v) {
                        n_trees = u16::try_from(n).map_err(|_| anyhow!("FOREST TREES inválido '{}'", p))?;
                    }
                    else { return Err(anyhow!("FOREST TREES inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("DEPTH=") {
                    if let Ok(n) = v.parse::<u16>() { depth = n; }
                    else if let Ok(n) = syms.resolve_const(v) {
                        depth = u16::try_from(n).map_err(|_| anyhow!("FOREST DEPTH inválido '{}'", p))?;
                    }
                    else { return Err(anyhow!("FOREST DEPTH inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("MODE=") {
                    mode = match v {
                        "VOTE" => FOREST_MODE_VOTE,
                        "MEAN" => FOREST_MODE_MEAN,
                        _ => return Err(anyhow!("FOREST MODE inválido '{}' (use VOTE/MEAN)", p)),
                    };
                } else {
                    return Err(anyhow!("FOREST token desconhecido '{}' (use TREES=/DEPTH=/MODE=)", p));
                }
            }
            if n_trees == 0 || depth == 0 || depth > 16 {
                return Err(anyhow!("FOREST TREES/DEPTH fora da faixa (trees>=1, 1<=depth<=16)"));
            }
            Ok(instr_forest(
                parse_reg(parts[1], syms)?,
                parse_reg(parts[2], syms)?,
                parse_reg(parts[3], syms)?,
                parse_reg(parts[4], syms)?,
                n_trees,
                depth,
                mode,
            ))
        }
        "RAG_INDEX_ADD" => {
            // RAG_INDEX_ADD rD, rDb, rVec [, rId] (rDb=0 cria; rId ausente=seq).
            if parts.len() < 4 {
                return Err(anyhow!("RAG_INDEX_ADD precisa de rdest, rDb, rVec — ex: RAG_INDEX_ADD r4, r0, r1"));
            }
            // 4º token, se presente, DEVE ser registrador (nada silencioso).
            let rid = if parts.len() > 4 {
                Some(parse_reg(parts[4], syms).map_err(|_| anyhow!("RAG_INDEX_ADD 4º operando '{}' inválido (use registrador rId)", parts[4]))?)
            } else {
                None
            };
            if parts.len() > 5 {
                reject_unknown("RAG_INDEX_ADD", &parts[5..], &[])?;
            }
            Ok(instr_rag_index_add(
                parse_reg(parts[1], syms)?,
                parse_reg(parts[2], syms)?,
                parse_reg(parts[3], syms)?,
                rid.unwrap_or(0xFF),
            ))
        }
        "RAG_INDEX_DEL" => {
            // RAG_INDEX_DEL rD, rDb, rId (id exigido; ausente não deleta).
            if parts.len() != 4 {
                return Err(anyhow!("RAG_INDEX_DEL precisa de rdest, rDb, rId — ex: RAG_INDEX_DEL r4, r0, r1"));
            }
            Ok(instr_rag_index_del(
                parse_reg(parts[1], syms)?,
                parse_reg(parts[2], syms)?,
                parse_reg(parts[3], syms)?,
            ))
        }
        "HALT" => {
            if parts.len() > 1 {
                reject_unknown("HALT", &parts[1..], &[])?;
            }
            Ok(instr_halt())
        }
        "NOP" => {
            if parts.len() > 1 {
                reject_unknown("NOP", &parts[1..], &[])?;
            }
            Ok(instr_nop())
        }
        _ => Err(anyhow!("opcode desconhecido '{}'", parts[0])),
    }
}

fn parse_dtype(s: &str) -> Result<u8> {
    match s.to_ascii_lowercase().as_str() {
        "f32" | "fp32" | "0" => Ok(0),
        "f16" | "fp16" | "1" => Ok(1),
        "i8" | "2" => Ok(2),
        "u8" | "3" => Ok(3),
        "bf16" | "bfloat16" | "64" => Ok(64),
        _ => Err(anyhow!("dtype desconhecido '{}' (use f32/f16/bf16/i8/u8)", s)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let instr = instr_tensor(0, 1, 2, 2, 2, 0);
        let bytes = instr.encode();
        assert_eq!(bytes.len(), 32);
        let decoded = Instruction::decode(&bytes).unwrap();
        assert_eq!(decoded.opcode, OP_TENSOR);
        assert_eq!(decoded.rdest, 0);
        assert_eq!(decoded.tensor_shape(), (2, 2));
    }

    #[test]
    fn test_assemble_simple() {
        let src = r#"
            ; aloca 2x2
            TENSOR r0 2 2 f32
            TENSOR r1 2 2 f32
            TENSOR r2 2 2 f32
            ATTN r3, r0, r1, r2
            STREAM r3, r0 BLOCKING
            HALT
        "#;
        let prog = assemble(src).unwrap();
        assert_eq!(prog.len(), 6);
        assert_eq!(prog[0].opcode, OP_TENSOR);
        assert_eq!(prog[3].opcode, OP_ATTN);
        assert_eq!(prog[4].opcode, OP_STREAM);
        assert_eq!(prog[5].opcode, OP_HALT);
    }

    #[test]
    fn test_fork_flags() {
        let src = "FORK r0, RED";
        let prog = assemble(src).unwrap();
        assert_eq!(prog[0].flags, FORK_FLAG_RED);
    }

    #[test]
    fn test_norm_ffn_assemble() {
        let src = r#"
            TENSOR r0 2 2 f32
            TENSOR r1 2 2 f32
            TENSOR r2 2 2 f32
            NORM r3, r0, r1, r2
            FFN r4, r3, r0, r1
            FFN r5, r3, r0, r1, r2, r0
            HALT
        "#;
        let prog = assemble(src).unwrap();
        assert_eq!(prog[3].opcode, OP_NORM);
        assert_eq!(prog[4].opcode, OP_FFN);
        assert_eq!(prog[4].rsrc1, 3);
        assert_eq!(prog[5].opcode, OP_FFN);
        // FFN com bias: FFN r5,r3,r0,r1,r2,r0 => rw1=r0, rb1=r1, rw2=r2, rb2=r0
        assert_eq!(prog[5].payload[0], 1);
        assert_eq!(prog[5].payload[1], 0);
    }

    #[test]
    fn test_ffn_bias_decode() {
        let instr = instr_ffn_with_bias(3, 0, 1, 4, 2, 5);
        assert_eq!(instr.opcode, OP_FFN);
        assert_eq!(instr.rdest, 3);
        assert_eq!(instr.rsrc1, 0);
        assert_eq!(instr.rsrc2, 1);
        assert_eq!(instr.rsrc3, 2);
        assert_eq!(instr.payload[0], 4);
        assert_eq!(instr.payload[1], 5);
        let bytes = instr.encode();
        let dec = Instruction::decode(&bytes).unwrap();
        assert_eq!(dec.ffn_bias_regs(), (4, 5));
    }

    #[test]
    fn test_new_opcodes_encode_decode() {
        let instr_emb = instr_embed(10, 11, 1);
        assert_eq!(instr_emb.opcode, OP_EMBED);
        assert_eq!(instr_emb.rdest, 10);
        assert_eq!(instr_emb.rsrc1, 11);
        assert_eq!(instr_emb.rsrc2, 1);

        let instr_a = instr_add(10, 10, 14);
        assert_eq!(instr_a.opcode, OP_ADD);
        assert_eq!(instr_a.rdest, 10);
        assert_eq!(instr_a.rsrc1, 10);
        assert_eq!(instr_a.rsrc2, 14);

        let instr_s = instr_sample(11, 15, 0.7);
        assert_eq!(instr_s.opcode, OP_SAMPLE);
        assert_eq!(instr_s.rdest, 11);
        assert_eq!(instr_s.rsrc1, 15);
        let mut b = [0u8; 4];
        b.copy_from_slice(&instr_s.payload[0..4]);
        assert_eq!(f32::from_le_bytes(b), 0.7);
    }

    #[test]
    fn test_labels_and_control_flow_assemble() {
        let src = r#"
            MAIN_LOOP:
                SENSE r5, USER_INPUT
                IF_INTERRUPT r5, HANDLE_ABORT
                EMBED r10, r11, r1
                ADD r10, r10, r14
                SAMPLE r11, r15
                COMPARE r11, EOS_TOKEN
                IF_EQUAL PROGRAM_END
                JUMP MAIN_LOOP

            HANDLE_ABORT:
                ABORT r12, r0
                HALT

            PROGRAM_END:
                HALT
        "#;
        let prog = assemble(src).unwrap();
        assert_eq!(prog[0].opcode, OP_SENSE);
        assert_eq!(prog[1].opcode, OP_IF_INTERRUPT);
        assert_eq!(prog[2].opcode, OP_EMBED);
        assert_eq!(prog[3].opcode, OP_ADD);
        assert_eq!(prog[4].opcode, OP_SAMPLE);
        assert_eq!(prog[5].opcode, OP_COMPARE);
        assert_eq!(prog[6].opcode, OP_IF_EQUAL);
        assert_eq!(prog[7].opcode, OP_JUMP);

        // JUMP MAIN_LOOP aponta para a primeira instrução (PC = 0x1000)
        assert_eq!(prog[7].imm_u128(), 0x1000);

        // HANDLE_ABORT está no índice 8 (PC = 0x1000 + 8 * 32 = 0x1100)
        assert_eq!(prog[1].imm_u128(), 0x1000 + 8 * 32);

        // PROGRAM_END está no índice 10 (PC = 0x1000 + 10 * 32 = 0x1140)
        assert_eq!(prog[6].imm_u128(), 0x1000 + 10 * 32);
    }

    #[test]
    fn test_new_isa_0x13_0x19_roundtrip() {
        let s = instr_ssm_scan(5, 0, 1, 2, 8, 4, 3);
        assert_eq!(s.opcode, OP_SSM_SCAN);
        assert_eq!(s.ssm_dims(), (8, 4, 3));
        let dec = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(dec.opcode, OP_SSM_SCAN);
        assert_eq!(dec.ssm_dims(), (8, 4, 3));

        let r = instr_rope(2, 0, 7, 8, 2, 10_000.0);
        assert_eq!(r.opcode, OP_ROPE);
        let (pos, hd, nh, th) = r.rope_params();
        assert_eq!((pos, hd, nh), (7, 8, 2));
        assert!((th - 10_000.0).abs() < 1e-3);

        let a = instr_audio_align(4, 0, 1);
        let (sr, spf, hz, dl) = a.audio_align_params();
        assert_eq!((sr, spf), (24_000, 1920));
        assert!((hz - 12.5).abs() < 1e-3 && (dl - 160.0).abs() < 1e-3);

        // CTX_SWITCH: pipe canônico no payload (fim da ambiguidade r0–r2).
        let cs = instr_ctx_switch(PIPE_MAMBA, 0b10);
        assert_eq!(ctx_switch_pipe(&cs), PIPE_MAMBA);
        // Legado: instrução montada à mão sem o construtor usa rsrc1.
        let legacy = Instruction::new(OP_CTX_SWITCH, 0b01, 0xFF, PIPE_AUDIO, 0xFF, 0xFF);
        assert_eq!(ctx_switch_pipe(&legacy), PIPE_AUDIO);

        for (op, mk) in [
            (OP_SSM_RESET, instr_ssm_reset(1, 4, 4, 0)),
            (OP_CODEC_ENC, instr_codec_enc(2, 0, false)),
            (OP_CODEC_DEC, instr_codec_dec(3, 2, false)),
            (OP_CTX_SWITCH, instr_ctx_switch(PIPE_MAMBA, 0b10)),
        ] {
            assert_eq!(mk.opcode, op);
            let d = Instruction::decode(&mk.encode()).unwrap();
            assert_eq!(d.opcode, op);
            assert_eq!(d.mnemonic(), mk.mnemonic());
        }
    }

    #[test]
    fn test_new_isa_assemble() {
        let src = r#"
            SSM_SCAN r5, r0, r1, r2 D_INNER=2 D_STATE=2 LAYER=0
            SSM_RESET r1 D_INNER=2 D_STATE=2
            CODEC_ENC r2, r0
            CODEC_DEC r3, r2
            AUDIO_ALIGN r4, r0, r1
            CTX_SWITCH MAMBA, RED
            ROPE r2, r0 POS=0 HDIM=4 NHEADS=1
            SENSE r6, AUDIO_PCM
            SENSE r7, CODEC_FRAME
            HALT
        "#;
        let prog = assemble(src).unwrap();
        assert_eq!(prog[0].opcode, OP_SSM_SCAN);
        assert_eq!(prog[0].ssm_dims(), (2, 2, 0));
        assert_eq!(prog[1].opcode, OP_SSM_RESET);
        assert_eq!(prog[2].opcode, OP_CODEC_ENC);
        assert_eq!(prog[3].opcode, OP_CODEC_DEC);
        assert_eq!(prog[4].opcode, OP_AUDIO_ALIGN);
        assert_eq!(prog[5].opcode, OP_CTX_SWITCH);
        assert_eq!(prog[5].rsrc1, PIPE_MAMBA);
        assert_eq!(prog[6].opcode, OP_ROPE);
        assert_eq!(prog[7].opcode, OP_SENSE);
        assert_eq!(prog[7].rsrc1, SENSE_AUDIO_PCM);
        assert_eq!(prog[8].rsrc1, SENSE_CODEC_FRAME);
    }

    // ---- RFC-0002: dual-mode (ESPEC-V2 §4) ------------------------------

    #[test]
    fn test_instr_width_full_map() {
        // Total sobre os 256 opcodes, sem exceção além de 0xFF (R2).
        for op in 0x00u8..=0x7Fu8 {
            assert_eq!(instr_width(op), InstrWidth::Fixed32, "op 0x{:02x}", op);
        }
        for op in 0x80u8..=0xAFu8 {
            assert_eq!(instr_width(op), InstrWidth::Fixed64, "op 0x{:02x}", op);
        }
        assert_eq!(instr_width(0xB0), InstrWidth::Fixed64); // cabeça; total via ext_len
        assert_eq!(instr_width(0xB1), InstrWidth::Fixed128);
        assert_eq!(instr_width(0xB2), InstrWidth::Fixed256);
        assert_eq!(instr_width(0xB3), InstrWidth::EscapeVar);
        for op in 0xB4u8..=0xB6u8 {
            assert_eq!(instr_width(op), InstrWidth::Fixed64, "op 0x{:02x}", op);
        }
        for op in 0xB7u8..=0xFEu8 {
            assert_eq!(instr_width(op), InstrWidth::Reserved, "op 0x{:02x}", op);
        }
        assert_eq!(instr_width(0xFF), InstrWidth::Fixed32); // R2: única exceção
        assert_eq!(fixed_size(InstrWidth::Fixed32), Some(32));
        assert_eq!(fixed_size(InstrWidth::Fixed64), Some(64));
        assert_eq!(fixed_size(InstrWidth::Fixed128), Some(128));
        assert_eq!(fixed_size(InstrWidth::Fixed256), Some(256));
        assert_eq!(fixed_size(InstrWidth::EscapeVar), None);
        assert_eq!(fixed_size(InstrWidth::Reserved), None);
    }

    #[test]
    fn test_decode_width_rejection() {
        // Zona 64B/escape/reservada: rejeição limpa, nunca misdecode.
        for op in [0x80u8, 0x84, 0x9F, 0xA0, 0xB0, 0xB1, 0xB3, 0xB4] {
            let mut bytes = [0u8; 32];
            bytes[0] = op;
            assert!(
                matches!(Instruction::decode(&bytes), Err(DecodeError::UnsupportedWidth(o)) if o == op),
                "op 0x{:02x} deve ser UnsupportedWidth",
                op
            );
        }
        for op in [0xB7u8, 0xC0, 0xE0, 0xFE] {
            let mut bytes = [0u8; 32];
            bytes[0] = op;
            assert!(
                matches!(Instruction::decode(&bytes), Err(DecodeError::ReservedOpcode(o)) if o == op),
                "op 0x{:02x} deve ser ReservedOpcode",
                op
            );
        }
        // 0xFF (sempre 32B) e 0x1A (congelado, rejeitado só no execute) decodificam.
        for op in [0x00u8, 0x19, 0x1A, 0x25, 0xFF] {
            let mut bytes = [0u8; 32];
            bytes[0] = op;
            bytes[2] = 0xFF;
            bytes[3] = 0xFF;
            bytes[4] = 0xFF;
            bytes[5] = 0xFF;
            assert!(Instruction::decode(&bytes).is_ok(), "op 0x{:02x} deve decodificar", op);
        }
        // Truncado continua Incomplete (caminho de erro preservado).
        assert!(matches!(Instruction::decode(&[]), Err(DecodeError::Incomplete(0))));
        assert!(matches!(Instruction::decode(&[0u8; 31]), Err(DecodeError::Incomplete(31))));
    }

    #[test]
    fn test_escape_total_len_bounds() {
        assert_eq!(escape_total_len(64).unwrap(), 64);
        assert_eq!(escape_total_len(128).unwrap(), 128);
        assert_eq!(escape_total_len(1024 * 1024).unwrap(), 1024 * 1024);
        for bad in [0u64, 7, 63, 65, 68, 72 + 1, 1024 * 1024 + 8] {
            assert!(
                matches!(escape_total_len(bad), Err(DecodeError::InvalidExtLen(n)) if n == bad),
                "ext_len {} deve ser inválido",
                bad
            );
        }
    }

    // ---- W1-remainder: instrução 64B (ESPEC-V2 §4.2) --------------------

    fn sample_instr64() -> Instr64 {
        let mut ins = Instr64::new(0x84, 3, [0, 1, 2, 0xFF, 0xFF]);
        ins.flags = 0b0010_0001;
        ins.lamport = 7;
        ins.deadline = 1_000_000;
        ins.payload_ext = [9u8, 8, 7, 6, 5, 4, 3, 2];
        ins.payload_core[0] = 0xAA;
        ins.payload_core[31] = 0x55;
        ins
    }

    #[test]
    fn test_instr64_roundtrip() {
        let ins = sample_instr64();
        let bytes = ins.encode();
        assert_eq!(bytes.len(), 64);
        assert_eq!(bytes[0], 0x84);
        assert_eq!(bytes[1], 0b0010_0001);
        assert_eq!(bytes[2], 3);
        assert_eq!(&bytes[3..8], &[0, 1, 2, 0xFF, 0xFF]);
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 7);
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 1_000_000);
        assert_eq!(&bytes[24..32], &[9u8, 8, 7, 6, 5, 4, 3, 2]);
        assert_eq!(bytes[32], 0xAA);
        assert_eq!(bytes[63], 0x55);
        let back = Instr64::decode(&bytes).unwrap();
        assert_eq!(back, ins);
        assert_eq!(back.encode(), bytes); // byte-idêntico
        assert_eq!(back.byte_len(), 64);
    }

    #[test]
    fn test_instr64_new_neutral() {
        let ins = Instr64::new(0x80, 0xFF, [0xFF; 5]);
        assert!(ins.has_neutral_ext()); // R4-ready
        assert!(!sample_instr64().has_neutral_ext());
        assert_eq!(ins.mnemonic(), "UNKNOWN"); // nomes chegam nas RFCs
        assert!(ins.validate().is_ok());
    }

    #[test]
    fn test_instr64_rejections() {
        // Curto => Incomplete64 (não o Incomplete de 32B).
        assert!(matches!(Instr64::decode(&[0u8; 63]), Err(DecodeError::Incomplete64(63))));
        assert!(matches!(Instr64::decode(&[]), Err(DecodeError::Incomplete64(0))));
        // Opcode 32B => UnsupportedWidth (não misdecode como 64B).
        let mut b32 = [0u8; 64];
        b32[0] = 0x01;
        assert!(matches!(Instr64::decode(&b32), Err(DecodeError::UnsupportedWidth(0x01))));
        // 0xFF (sempre 32B, R2) também rejeita no decoder 64B.
        let mut bff = [0u8; 64];
        bff[0] = 0xFF;
        assert!(matches!(Instr64::decode(&bff), Err(DecodeError::UnsupportedWidth(0xFF))));
        // Reservado => ReservedOpcode, sem tamanho assumido.
        let mut br = [0u8; 64];
        br[0] = 0xB7;
        assert!(matches!(Instr64::decode(&br), Err(DecodeError::ReservedOpcode(0xB7))));
        // Cabeça ESCAPE 0xB0 => layout próprio, não plano.
        let mut be = [0u8; 64];
        be[0] = 0xB0;
        assert!(matches!(Instr64::decode(&be), Err(DecodeError::EscapeHead(0xB0))));
        // 0xB1/0xB2/0xB3 têm larguras próprias.
        for op in [0xB1u8, 0xB2, 0xB3] {
            let mut b = [0u8; 64];
            b[0] = op;
            assert!(Instr64::decode(&b).is_err(), "op 0x{:02x} não é 64B plano", op);
        }
        // Registrador inválido => InvalidRegister (rdest e rsrc).
        let mut bad = sample_instr64().encode();
        bad[2] = 16;
        assert!(matches!(Instr64::decode(&bad), Err(DecodeError::InvalidRegister(16))));
        let mut bad2 = sample_instr64().encode();
        bad2[5] = 42;
        assert!(matches!(Instr64::decode(&bad2), Err(DecodeError::InvalidRegister(42))));
    }

    #[test]
    fn test_instr64_display() {
        let s = format!("{}", sample_instr64());
        assert!(s.contains("X_84"), "display deve mostrar opcode: {}", s);
    }

    #[test]
    fn test_program_instr_accessors() {
        let a = ProgramInstr::W32(instr_nop());
        assert_eq!(a.opcode(), OP_NOP);
        assert_eq!(a.byte_len(), 32);
        assert_eq!(a.mnemonic(), "NOP");
        let b = ProgramInstr::W64(Instr64::new(0x84, 0, [0xFF; 5]));
        assert_eq!(b.opcode(), 0x84);
        assert_eq!(b.byte_len(), 64);
        assert_eq!(b.mnemonic(), "UNKNOWN");
        // Copy: o fetch retorna por valor, sem empréstimo do Vm.
        let c = b;
        assert_eq!(c, b);
        // Display delega para a largura interna.
        assert!(format!("{}", a).contains("NOP"));
        assert!(format!("{}", b).contains("X_84"));
    }

    // ---- RFC-0004: GATHER / DISTANCE / RANK1_UPDATE ---------------------

    #[test]
    fn test_rfc0004_ctor_roundtrip() {
        let g = instr_gather(5, 0, 1, 0xFF, 0, GATHER_MODE_GATHER);
        assert_eq!(g.opcode, OP_GATHER);
        assert_eq!(g.gather_params(), (0, GATHER_MODE_GATHER, 0));
        let d = Instruction::decode(&g.encode()).unwrap();
        assert_eq!((d.opcode, d.gather_params()), (OP_GATHER, (0, GATHER_MODE_GATHER, 0)));
        assert_eq!(d.mnemonic(), "GATHER");

        let s = instr_distance(5, 0, 1, DIST_METRIC_COSINE, 5);
        assert_eq!(s.opcode, OP_DISTANCE);
        assert_eq!(s.distance_params(), (DIST_METRIC_COSINE, 5));
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!((d.opcode, d.distance_params()), (OP_DISTANCE, (DIST_METRIC_COSINE, 5)));
        assert_eq!(d.mnemonic(), "DISTANCE");

        let r = instr_rank1_update(7, 1, 2, 0.99, 1.0, RANK1_MODE_DELTA, 3);
        assert_eq!(r.opcode, OP_RANK1_UPDATE);
        let (a, b, m, l) = r.rank1_params();
        assert!((a - 0.99).abs() < 1e-6 && (b - 1.0).abs() < 1e-9 && m == RANK1_MODE_DELTA && l == 3);
        let d = Instruction::decode(&r.encode()).unwrap();
        assert_eq!(d.mnemonic(), "RANK1_UPDATE");

        let t = instr_sample_topk(4, 0, 2);
        assert_eq!(t.sample_topk(), 2);
        let d = Instruction::decode(&t.encode()).unwrap();
        assert_eq!(d.sample_topk(), 2);
        // Legado: payload zerado => amostragem (topk 0).
        assert_eq!(instr_sample(4, 0, 1.0).sample_topk(), 0);
    }

    #[test]
    fn test_rfc0004_assemble() {
        let prog = assemble("GATHER r5, r0, r1 AXIS=0 MODE=GATHER").unwrap();
        assert_eq!(prog[0].opcode, OP_GATHER);
        assert_eq!(prog[0].gather_params().0, 0);
        let prog = assemble("SCATTER_ADD r5, r0, r1 AXIS=1").unwrap();
        assert_eq!(prog[0].gather_params(), (1, GATHER_MODE_SCATTER_ADD, 0));
        let prog = assemble("DISTANCE r5, r0, r1 METRIC=COSINE TOPK=5").unwrap();
        assert_eq!(prog[0].distance_params(), (DIST_METRIC_COSINE, 5));
        let prog = assemble("DISTANCE r5, r0, r1 METRIC=DOT").unwrap();
        assert_eq!(prog[0].distance_params(), (DIST_METRIC_DOT, 0));
        let prog = assemble("RANK1_UPDATE r7, r1, r2 ALPHA=0.99 BETA=1.0 MODE=DELTA LAYER=2").unwrap();
        let (a, b, m, l) = prog[0].rank1_params();
        assert!((a - 0.99).abs() < 1e-6 && m == RANK1_MODE_DELTA && l == 2 && (b - 1.0).abs() < 1e-9);
        let prog = assemble("SAMPLE r4, r0 TOPK=2").unwrap();
        assert_eq!(prog[0].sample_topk(), 2);
        assert!(assemble("GATHER r5, r0").is_err());
        assert!(assemble("DISTANCE r5, r0, r1 METRIC=HAMMING").is_err());
    }

    // ---- RFC-0005: determinismo -------------------------------------

    #[test]
    fn test_rfc0005_ctor_roundtrip() {
        let s = instr_rng_seed(3);
        assert_eq!((s.opcode, s.rsrc1), (OP_RNG_SEED, 3));
        assert_eq!(Instruction::decode(&s.encode()).unwrap().mnemonic(), "RNG_SEED");
        let n = instr_rng_next(4);
        assert_eq!(Instruction::decode(&n.encode()).unwrap().mnemonic(), "RNG_NEXT");
        let u = instr_rng_uniform(4, -2.0, 5.0);
        assert_eq!(u.uniform_range(), (-2.0, 5.0));
        assert_eq!(Instruction::decode(&u.encode()).unwrap().uniform_range(), (-2.0, 5.0));
        // Payload zerado => defaults.
        let u0 = Instruction::new(OP_RNG_UNIFORM, 0, 4, 0xFF, 0xFF, 0xFF);
        assert_eq!(u0.uniform_range(), (0.0, 1.0));
        let m = instr_rng_normal(4, 10.0, 2.0);
        assert_eq!(m.normal_params(), (10.0, 2.0));
        let m0 = Instruction::new(OP_RNG_NORMAL, 0, 4, 0xFF, 0xFF, 0xFF);
        assert_eq!(m0.normal_params(), (0.0, 1.0));
        for (mk, name) in [
            (instr_hash(4, 0), "HASH"),
            (instr_checksum(4, 0), "CHECKSUM"),
            (instr_hmac(4, 0, 1), "HMAC"),
        ] {
            let d = Instruction::decode(&mk.encode()).unwrap();
            assert_eq!(d.mnemonic(), name);
        }
    }

    #[test]
    fn test_rfc0005_assemble() {
        let prog = assemble("RNG_SEED r3").unwrap();
        assert_eq!((prog[0].opcode, prog[0].rsrc1), (OP_RNG_SEED, 3));
        let prog = assemble("RNG_SEED DEFAULT").unwrap();
        assert_eq!(prog[0].rsrc1, 0xFF);
        let prog = assemble("RNG_NEXT r4").unwrap();
        assert_eq!(prog[0].opcode, OP_RNG_NEXT);
        let prog = assemble("RNG_UNIFORM r4 A=-2.0 B=5.0").unwrap();
        assert_eq!(prog[0].uniform_range(), (-2.0, 5.0));
        let prog = assemble("RNG_NORMAL r4 MEAN=10.0 STD=2.0").unwrap();
        assert_eq!(prog[0].normal_params(), (10.0, 2.0));
        let prog = assemble("HASH r4, r0").unwrap();
        assert_eq!(prog[0].opcode, OP_HASH);
        let prog = assemble("CHECKSUM r4, r0").unwrap();
        assert_eq!(prog[0].opcode, OP_CHECKSUM);
        let prog = assemble("HMAC r4, r0, r1").unwrap();
        assert_eq!((prog[0].opcode, prog[0].rsrc2), (OP_HMAC, 1));
        assert!(assemble("RNG_NEXT").is_err());
        assert!(assemble("HASH r4").is_err());
    }

    // ---- RFC-0006: telemetria + scheduler ---------------------------

    #[test]
    fn test_rfc0006_ctor_roundtrip() {
        let c = instr_cycles_count(4);
        assert_eq!(c.opcode, OP_CYCLES_COUNT);
        assert_eq!(Instruction::decode(&c.encode()).unwrap().mnemonic(), "CYCLES_COUNT");
        let t = instr_trace_event(1, 2);
        assert_eq!((t.rsrc1, t.rsrc2), (1, 2));
        assert_eq!(Instruction::decode(&t.encode()).unwrap().mnemonic(), "TRACE_EVENT");
        let s = instr_sanity_check(4, 0, 1);
        assert_eq!((s.rdest, s.rsrc3), (4, 1));
        let p = instr_preempt_check(4);
        assert_eq!(Instruction::decode(&p.encode()).unwrap().mnemonic(), "PREEMPT_CHECK");
        let a = instr_assert(1, 42);
        assert_eq!(a.assert_code(), 42);
        assert_eq!(Instruction::decode(&a.encode()).unwrap().assert_code(), 42);
        for (mk, name) in [
            (instr_dump(), "DUMP"),
            (instr_yield(), "YIELD"),
            (instr_set_deadline(0), "SET_DEADLINE"),
            (instr_get_deadline(4), "GET_DEADLINE"),
            (instr_priority_set(0), "PRIORITY_SET"),
            (instr_priority_get(4), "PRIORITY_GET"),
            (instr_lock(0), "LOCK"),
            (instr_unlock(0), "UNLOCK"),
            (instr_fence(), "FENCE"),
        ] {
            assert_eq!(Instruction::decode(&mk.encode()).unwrap().mnemonic(), name);
        }
    }

    #[test]
    fn test_rfc0006_assemble() {
        let prog = assemble("CYCLES_COUNT r4").unwrap();
        assert_eq!(prog[0].opcode, OP_CYCLES_COUNT);
        let prog = assemble("TRACE_EVENT r1, r2").unwrap();
        assert_eq!((prog[0].rsrc1, prog[0].rsrc2), (1, 2));
        let prog = assemble("SANITY_CHECK r4, r0, r1").unwrap();
        assert_eq!((prog[0].rdest, prog[0].rsrc3), (4, 1));
        let prog = assemble("SANITY_CHECK r4, r0").unwrap();
        assert_eq!(prog[0].rsrc3, 0xFF);
        let prog = assemble("PREEMPT_CHECK r4").unwrap();
        assert_eq!(prog[0].opcode, OP_PREEMPT_CHECK);
        let prog = assemble("ASSERT r1 CODE=42").unwrap();
        assert_eq!(prog[0].assert_code(), 42);
        let prog = assemble("ASSERT r1").unwrap();
        assert_eq!(prog[0].assert_code(), 0);
        for (src, op) in [
            ("DUMP", OP_DUMP),
            ("YIELD", OP_YIELD),
            ("SET_DEADLINE r0", OP_SET_DEADLINE),
            ("GET_DEADLINE r4", OP_GET_DEADLINE),
            ("PRIORITY_SET r0", OP_PRIORITY_SET),
            ("PRIORITY_GET r4", OP_PRIORITY_GET),
            ("LOCK r0", OP_LOCK),
            ("UNLOCK r0", OP_UNLOCK),
            ("FENCE", OP_FENCE),
        ] {
            let prog = assemble(src).unwrap();
            assert_eq!(prog[0].opcode, op, "{}", src);
        }
        assert!(assemble("CYCLES_COUNT").is_err());
        assert!(assemble("ASSERT").is_err());
        assert!(assemble("LOCK").is_err());
    }

    // ---- RFC-0007: LOADI / MOV / COMPARE-PRED -----------------------

    #[test]
    fn test_rfc0007_ctor_roundtrip() {
        let l = instr_loadi(3, 80_000_000);
        assert_eq!(l.opcode, OP_LOADI);
        assert_eq!(l.imm_u128(), 80_000_000);
        let d = Instruction::decode(&l.encode()).unwrap();
        assert_eq!((d.opcode, d.imm_u128()), (OP_LOADI, 80_000_000));
        assert_eq!(d.mnemonic(), "LOADI");
        let m = instr_mov(2, 0);
        assert_eq!(Instruction::decode(&m.encode()).unwrap().mnemonic(), "MOV");
        let mut c = instr_compare(1, 0xFF, 7);
        assert_eq!(c.compare_pred(), CMP_EQ); // legado: zero => EQ
        c.set_compare_pred(CMP_LT);
        assert_eq!(Instruction::decode(&c.encode()).unwrap().compare_pred(), CMP_LT);
    }

    #[test]
    fn test_rfc0007_assemble() {
        let prog = assemble("LOADI r0, 80000000").unwrap();
        assert_eq!(prog[0].imm_u128(), 80_000_000);
        let prog = assemble("LOADI r1, 0xFF").unwrap();
        assert_eq!(prog[0].imm_u128(), 255);
        let prog = assemble("LOADI r2, 340282366920938463463374607431768211455").unwrap();
        assert_eq!(prog[0].imm_u128(), u128::MAX);
        assert!(assemble("LOADI r0, -1").is_err());
        assert!(assemble("LOADI r0, 1.5").is_err());
        assert!(assemble("LOADI r0").is_err());
        assert!(assemble("LOADI r16, 1").is_err());
        let prog = assemble("MOV r1, r0").unwrap();
        assert_eq!((prog[0].opcode, prog[0].rsrc1), (OP_MOV, 0));
        let prog = assemble("COMPARE r1, r2 PRED=LT").unwrap();
        assert_eq!(prog[0].compare_pred(), CMP_LT);
        let prog = assemble("COMPARE r1, 100 PRED=GE").unwrap();
        assert_eq!(prog[0].compare_pred(), CMP_GE);
        let prog = assemble("COMPARE r1, r2").unwrap();
        assert_eq!(prog[0].compare_pred(), CMP_EQ);
        assert!(assemble("LOADI r0").is_err());
        assert!(assemble("LOADI r16, 1").is_err());
        let prog = assemble("MOV r1, r0").unwrap();
        assert_eq!((prog[0].opcode, prog[0].rsrc1), (OP_MOV, 0));
        let prog = assemble("COMPARE r1, r2 PRED=LT").unwrap();
        assert_eq!(prog[0].compare_pred(), CMP_LT);
        let prog = assemble("COMPARE r1, 100 PRED=GE").unwrap();
        assert_eq!(prog[0].compare_pred(), CMP_GE);
        let prog = assemble("COMPARE r1, r2").unwrap();
        assert_eq!(prog[0].compare_pred(), CMP_EQ);
        assert!(assemble("COMPARE r1, r2 PRED=XX").is_err());
        assert!(assemble("MOV r1").is_err());
    }

    // ---- RFC-0010: KV_TRUNCATE --------------------------------------

    #[test]
    fn test_rfc0010_ctor_roundtrip() {
        let k = instr_kv_truncate(3, 0);
        assert_eq!(k.opcode, OP_KV_TRUNCATE);
        assert_eq!(k.kv_stream(), 0);
        let d = Instruction::decode(&k.encode()).unwrap();
        assert_eq!(d.mnemonic(), "KV_TRUNCATE");
        assert_eq!(d.kv_stream(), 0);
        let mut k2 = instr_kv_truncate(3, 0);
        k2.set_kv_stream(2);
        assert_eq!(k2.kv_stream(), 2);
    }

    #[test]
    fn test_rfc0010_assemble() {
        let prog = assemble("KV_TRUNCATE r0").unwrap();
        assert_eq!((prog[0].opcode, prog[0].kv_stream()), (OP_KV_TRUNCATE, 0));
        let prog = assemble("KV_TRUNCATE r0 STREAM=0").unwrap();
        assert_eq!(prog[0].kv_stream(), 0);
        let prog = assemble("KV_TRUNCATE r0 STREAM=3").unwrap();
        assert_eq!(prog[0].kv_stream(), 3); // parse aceita; exec veta
        assert!(assemble("KV_TRUNCATE").is_err());
        assert!(assemble("KV_TRUNCATE r0 FOO=1").is_err());
    }

    // ---- RFC-0012: FOREST -------------------------------------------

    #[test]
    fn test_rfc0012_ctor_roundtrip() {
        let f = instr_forest(4, 0, 1, 2, 8, 6, FOREST_MODE_MEAN);
        assert_eq!(f.opcode, OP_FOREST);
        assert_eq!(f.forest_params(), (8, 6, FOREST_MODE_MEAN));
        let d = Instruction::decode(&f.encode()).unwrap();
        assert_eq!((d.opcode, d.forest_params()), (OP_FOREST, (8, 6, FOREST_MODE_MEAN)));
        assert_eq!(d.mnemonic(), "FOREST");
    }

    #[test]
    fn test_rfc0012_assemble() {
        let prog = assemble("FOREST r4, r0, r1, r2 TREES=8 DEPTH=6 MODE=MEAN").unwrap();
        assert_eq!(prog[0].forest_params(), (8, 6, FOREST_MODE_MEAN));
        let prog = assemble("FOREST r4, r0, r1, r2").unwrap();
        assert_eq!(prog[0].forest_params(), (1, 1, FOREST_MODE_VOTE));
        assert!(assemble("FOREST r4, r0, r1").is_err());
        assert!(assemble("FOREST r4, r0, r1, r2 TREES=0").is_err());
        assert!(assemble("FOREST r4, r0, r1, r2 DEPTH=0").is_err());
        assert!(assemble("FOREST r4, r0, r1, r2 DEPTH=17").is_err());
        assert!(assemble("FOREST r4, r0, r1, r2 MODE=X").is_err());
        assert!(assemble("FOREST r4, r0, r1, r2 FOO=1").is_err());
    }

    // ---- RFC-0013: DENOISE_STEP -------------------------------------

    #[test]
    fn test_rfc0013_ctor_roundtrip() {
        let d = instr_denoise_step(4, 0, 1, 0xFF, 0.98, 0.02, 0.0, 500);
        assert_eq!(d.opcode, OP_DENOISE_STEP);
        let (a, b, s, t) = d.denoise_params();
        assert!((a - 0.98).abs() < 1e-6 && (b - 0.02).abs() < 1e-9 && s == 0.0 && t == 500);
        let r = Instruction::decode(&d.encode()).unwrap();
        assert_eq!(r.mnemonic(), "DENOISE_STEP");
        let (a2, _, _, _) = r.denoise_params();
        assert!((a2 - 0.98).abs() < 1e-6);
    }

    #[test]
    fn test_rfc0013_assemble() {
        let prog = assemble("DENOISE_STEP r4, r0, r1 ALPHA=0.98 BETA=0.02 SIGMA=0.0 T=500").unwrap();
        let (a, b, s, t) = prog[0].denoise_params();
        assert!((a - 0.98).abs() < 1e-6 && (b - 0.02).abs() < 1e-9 && s == 0.0 && t == 500);
        assert_eq!((prog[0].rsrc3), 0xFF);
        let prog = assemble("DENOISE_STEP r4, r0, r1").unwrap();
        let (a, b, s, t) = prog[0].denoise_params();
        assert!((a - 0.98).abs() < 1e-6 && (b - 0.02).abs() < 1e-9 && s == 0.0 && t == 500);
        assert!(assemble("DENOISE_STEP r4, r0").is_err());
        assert!(assemble("DENOISE_STEP r4, r0, r1 FOO=1").is_err());
        assert!(assemble("DENOISE_STEP r4, r0, r1 ALPHA=abc").is_err());
    }

    // ---- RFC-0014: ODE_STEP -------------------------------------------

    #[test]
    fn test_rfc0014_ctor_roundtrip() {
        let o = instr_ode_step(2, 0, 1, 0xFF, 0.01, ODE_METHOD_RK2, 3);
        assert_eq!(o.opcode, OP_ODE_STEP);
        let (dt, m, l) = o.ode_params();
        assert!((dt - 0.01).abs() < 1e-9 && m == ODE_METHOD_RK2 && l == 3);
        let d = Instruction::decode(&o.encode()).unwrap();
        assert_eq!(d.mnemonic(), "ODE_STEP");
        let (dt2, _, _) = d.ode_params();
        assert!((dt2 - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_rfc0014_assemble() {
        let prog = assemble("ODE_STEP r2, r0, r1 DT=0.01 METHOD=RK2 LAYER=0").unwrap();
        let (dt, m, l) = prog[0].ode_params();
        assert!((dt - 0.01).abs() < 1e-9 && m == ODE_METHOD_RK2 && l == 0);
        assert_eq!((prog[0].rsrc3), 0xFF);
        let prog = assemble("ODE_STEP r2, r0, r1, r3 DT=0.05 METHOD=RK4").unwrap();
        assert_eq!(prog[0].rsrc3, 3);
        let prog = assemble("ODE_STEP r2, r0, r1").unwrap();
        let (dt, m, _) = prog[0].ode_params();
        assert!((dt - 0.01).abs() < 1e-9 && m == ODE_METHOD_EULER);
        assert!(assemble("ODE_STEP r2, r0").is_err());
        assert!(assemble("ODE_STEP r2, r0, r1 METHOD=RK5").is_err());
        assert!(assemble("ODE_STEP r2, r0, r1 DT=abc").is_err());
        assert!(assemble("ODE_STEP r2, r0, r1 FOO=1").is_err());
    }

    // ---- RFC-0015: SPIKE_STEP -----------------------------------------

    #[test]
    fn test_rfc0015_ctor_roundtrip() {
        let s = instr_spike_step(3, 1, 2, 0xFF, 1.0, 0.9, 0.0, 4, 2);
        assert_eq!(s.opcode, OP_SPIKE_STEP);
        let (t, d, r, l, k) = s.spike_params();
        assert!((t - 1.0).abs() < 1e-9 && (d - 0.9).abs() < 1e-6 && r == 0.0 && l == 4 && k == 2);
        let x = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(x.mnemonic(), "SPIKE_STEP");
        let (_, _, _, _, k2) = x.spike_params();
        assert_eq!(k2, 2);
    }

    #[test]
    fn test_rfc0015_assemble() {
        let prog = assemble("SPIKE_STEP r3, r1, r2 THRESH=1.0 DECAY=0.9 RESET=0.0 LAYER=4 REFRACT=2").unwrap();
        assert_eq!(prog[0].spike_params(), (1.0, 0.9, 0.0, 4, 2));
        let prog = assemble("SPIKE_STEP r3, r1, r2").unwrap();
        assert_eq!(prog[0].spike_params(), (1.0, 0.9, 0.0, 0, 2));
        let prog = assemble("SPIKE_STEP r3, r1, r2, r0 THRESH=0.5").unwrap();
        assert_eq!(prog[0].rsrc3, 0);
        assert!(assemble("SPIKE_STEP r3, r1").is_err());
        assert!(assemble("SPIKE_STEP r3, r1, r2 THRESH=abc").is_err());
        assert!(assemble("SPIKE_STEP r3, r1, r2 FOO=1").is_err());
    }

    // ---- RFC-0017: CONV -------------------------------------------------

    #[test]
    fn test_rfc0017_ctor_roundtrip() {
        let c = instr_conv(4, 0, 1, 0xFF, 1, 2, 3, 4, CONV_ACT_SILU);
        assert_eq!(c.opcode, OP_CONV);
        assert_eq!(c.conv_params(), (1, 2, 3, 4, CONV_ACT_SILU));
        let d = Instruction::decode(&c.encode()).unwrap();
        assert_eq!(d.mnemonic(), "CONV");
        // groups=0 normaliza p/ 1 na leitura.
        let mut z = instr_conv(4, 0, 1, 0xFF, 1, 0, 1, 0, CONV_ACT_NONE);
        let _ = &mut z;
        let mut raw = Instruction::new(OP_CONV, 0, 4, 0, 1, 0xFF);
        raw.payload[6] = 0;
        assert_eq!(raw.conv_params().3, 1);
    }

    #[test]
    fn test_rfc0017_assemble() {
        let prog = assemble("CONV r4, r0, r1 STRIDE=2 PAD=1 DILATION=1 GROUPS=1 ACT=SILU").unwrap();
        assert_eq!(prog[0].conv_params(), (2, 1, 1, 1, CONV_ACT_SILU));
        let prog = assemble("CONV r4, r0, r1, r2 ACT=RELU").unwrap();
        assert_eq!(prog[0].rsrc3, 2);
        assert_eq!(prog[0].conv_params().4, CONV_ACT_RELU);
        let prog = assemble("CONV r4, r0, r1").unwrap();
        assert_eq!(prog[0].conv_params(), (1, 0, 1, 1, CONV_ACT_NONE));
        assert!(assemble("CONV r4, r0").is_err());
        assert!(assemble("CONV r4, r0, r1 ACT=TANH").is_err());
        assert!(assemble("CONV r4, r0, r1 FOO=1").is_err());
        assert!(assemble("CONV r4, r0, r1 STRIDE=x").is_err());
    }

    // ---- RFC-0018: cluster F1 -----------------------------------------

    #[test]
    fn test_rfc0018_ctor_roundtrip() {
        let s = instr_remote_spawn(3, 0, 0x1000, 2);
        assert_eq!(s.opcode, OP_REMOTE_SPAWN);
        assert_eq!(s.remote_spawn_params(), (0, 0x1000, 2));
        assert_eq!(Instruction::decode(&s.encode()).unwrap().mnemonic(), "REMOTE_SPAWN");
        let g = instr_signal(2, SIGNAL_KIND_ABORT, 0, 7, 99);
        assert_eq!(g.rsrc1, SIGNAL_KIND_ABORT);
        assert_eq!(g.signal_params(), (0, 7, 99));
        assert_eq!(Instruction::decode(&g.encode()).unwrap().mnemonic(), "SIGNAL");
        let t = instr_send_tensor(5, 6, 0, 8, 64, SEND_MODE_MOVE);
        assert_eq!(t.send_tensor_params(), (0, 8, 64, SEND_MODE_MOVE));
        assert_eq!(Instruction::decode(&t.encode()).unwrap().mnemonic(), "SEND_TENSOR");
        let b = instr_barrier(7, 2, 500, 1);
        assert_eq!(b.barrier_params(), (7, 2, 500, 1));
        assert_eq!(Instruction::decode(&b.encode()).unwrap().mnemonic(), "BARRIER");
    }

    #[test]
    fn test_rfc0018_assemble() {
        assert!(assemble("REMOTE_SPAWN r3 NODE=0 ENTRY=NOPE GREEN").is_err());
        assert!(assemble("REMOTE_SPAWN r3 NODE=abc ENTRY=1 GREEN").is_err());
        assert!(assemble("REMOTE_SPAWN r3 NODE=0 GREEN").is_err());
        assert!(assemble("REMOTE_SPAWN r3 NODE=0 ENTRY=1 FOO=1").is_err());
        let prog = assemble("MAIN:\nREMOTE_SPAWN r3 NODE=0 ENTRY=MAIN GREEN\nHALT").unwrap();
        assert_eq!(prog[0].remote_spawn_params().1, 0x1000);
        let prog = assemble("SIGNAL r2 KIND=ABORT NODE=0 CTX=5 SEQ=9").unwrap();
        assert_eq!(prog[0].rsrc1, SIGNAL_KIND_ABORT);
        assert_eq!(prog[0].signal_params(), (0, 5, 9));
        let prog = assemble("SIGNAL r2 KIND=HALT NODE=0 CTX=5").unwrap();
        assert_eq!(prog[0].rsrc1, SIGNAL_KIND_HALT);
        assert!(assemble("SIGNAL r2 KIND=XX NODE=0 CTX=5").is_err());
        assert!(assemble("SIGNAL r2 KIND=ABORT NODE=x CTX=5").is_err());
        let prog = assemble("SEND_TENSOR r5, r6 NODE=0 OFF=8 LEN=64 MOVE").unwrap();
        assert_eq!(prog[0].send_tensor_params(), (0, 8, 64, SEND_MODE_MOVE));
        let prog = assemble("SEND_TENSOR r5 NODE=0 LEN=16").unwrap();
        assert_eq!(prog[0].rsrc1, 0xFF);
        assert!(assemble("SEND_TENSOR r5 NODE=0 LEN=1 FOO=1").is_err());
        let prog = assemble("BARRIER id=7 EXPECT=2 TIMEOUT=500 EPOCH=1").unwrap();
        assert_eq!(prog[0].barrier_params(), (7, 2, 500, 1));
        assert!(assemble("BARRIER id=7").is_err());
        assert!(assemble("BARRIER").is_err());
    }

    // ---- RFC-0019: FILL + SLICE ---------------------------------------

    #[test]
    fn test_rfc0019_tensor_fill() {
        let mut t = instr_tensor(0, 0xFF, 0xFF, 2, 2, 0);
        assert_eq!(t.tensor_fill(), None);
        t.set_tensor_fill(0.0);
        assert_eq!(t.tensor_fill(), Some(0.0));
        assert!((t.flags & TENSOR_FLAG_FILL) != 0);
        let d = Instruction::decode(&t.encode()).unwrap();
        assert_eq!(d.tensor_fill(), Some(0.0));
        let prog = assemble("TENSOR r0 2 2 f32 FILL=0").unwrap();
        assert_eq!(prog[0].tensor_fill(), Some(0.0));
        let prog = assemble("TENSOR r0 2 2 f32").unwrap();
        assert_eq!(prog[0].tensor_fill(), None);
        assert!(assemble("TENSOR r0 2 2 f32 FILL=abc").is_err());
        assert!(assemble("TENSOR r0 2 2 f32 SPARSE DENSITY=0.1 FILL=0").is_err());
        assert!(assemble("TENSOR r0 2 2 f32 FOO=1").is_err());
    }

    #[test]
    fn test_rfc0019_slice_roundtrip() {
        let s = instr_slice(4, 0, 3, 3);
        assert_eq!(s.opcode, OP_SLICE);
        assert_eq!(s.slice_params(), (3, 3));
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(d.mnemonic(), "SLICE");
        assert_eq!(d.slice_params(), (3, 3));
        let prog = assemble("SLICE r4, r0 START=3 LEN=3").unwrap();
        assert_eq!(prog[0].slice_params(), (3, 3));
        assert!(assemble("SLICE r4, r0 START=3").is_err());
        assert!(assemble("SLICE r4, r0 LEN=3").is_err());
        assert!(assemble("SLICE r4").is_err());
        assert!(assemble("SLICE r4, r0 START=3 LEN=3 FOO=1").is_err());
    }

    // ---- RFC-0023: núcleo de memória --------------------------------

    #[test]
    fn test_rfc0023_ctor_roundtrip() {
        let a = instr_arena_alloc(2, 64, 16, 1);
        assert_eq!(a.opcode, OP_ARENA_ALLOC);
        assert_eq!(a.arena_alloc_params(), (64, 16, 1));
        let d = Instruction::decode(&a.encode()).unwrap();
        assert_eq!(d.mnemonic(), "ARENA_ALLOC");
        assert_eq!(d.arena_alloc_params(), (64, 16, 1));
        let r = instr_arena_reset(3);
        assert_eq!(r.opcode, OP_ARENA_RESET);
        assert_eq!(r.arena_id(), 3);
        let d = Instruction::decode(&r.encode()).unwrap();
        assert_eq!(d.mnemonic(), "ARENA_RESET");
        let c = instr_memcpy(1, 0, 16, 0, 8, MEMCPY_DIR_HOST);
        assert_eq!(c.opcode, OP_MEMCPY);
        assert_eq!(c.memcpy_params(), (16, 0, 8, MEMCPY_DIR_HOST));
        let d = Instruction::decode(&c.encode()).unwrap();
        assert_eq!(d.mnemonic(), "MEMCPY");
        assert_eq!(d.memcpy_params(), (16, 0, 8, MEMCPY_DIR_HOST));
        let s = instr_memset(1, 0xAB, 4, 8);
        assert_eq!(s.opcode, OP_MEMSET);
        assert_eq!(s.memset_params(), (0xAB, 4, 8));
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(d.mnemonic(), "MEMSET");
        assert_eq!(d.memset_params(), (0xAB, 4, 8));
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("ARENA_ALLOC r2, SIZE=64 ALIGN=16 ARENA=1").unwrap();
        assert_eq!(prog[0].arena_alloc_params(), (64, 16, 1));
        let prog = assemble("ARENA_RESET ARENA=3").unwrap();
        assert_eq!(prog[0].arena_id(), 3);
        let prog = assemble("MEMCPY r1, r0 LEN=16 DST_OFF=8 DIR=HOST").unwrap();
        assert_eq!(prog[0].memcpy_params(), (16, 0, 8, MEMCPY_DIR_HOST));
        let prog = assemble("MEMSET r1, PATTERN=171 LEN=4 OFF=8").unwrap();
        assert_eq!(prog[0].memset_params(), (171, 4, 8));
        assert!(assemble("ARENA_ALLOC r2").is_err());
        assert!(assemble("ARENA_ALLOC r2, SIZE=64 FOO=1").is_err());
        assert!(assemble("ARENA_RESET FOO=1").is_err());
        assert!(assemble("MEMCPY r1").is_err());
        assert!(assemble("MEMCPY r1, r0 DIR=GP").is_err());
        assert!(assemble("MEMCPY r1, r0 DIR=9").is_err());
        assert!(assemble("MEMCPY r1, r0 FOO=1").is_err());
        assert!(assemble("MEMSET r1").is_err());
        assert!(assemble("MEMSET r1, LEN=4").is_err());
        assert!(assemble("MEMSET r1, PATTERN=256").is_err());
        assert!(assemble("MEMSET r1, PATTERN=0 FOO=1").is_err());
    }

    // ---- RFC-0024: views & versions ---------------------------------

    #[test]
    fn test_rfc0024_ctor_roundtrip() {
        let s = instr_snapshot(5, SNAP_MASK_ALL);
        assert_eq!(s.opcode, OP_SNAPSHOT);
        assert_eq!(s.snapshot_mask(), SNAP_MASK_ALL);
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(d.mnemonic(), "SNAPSHOT");
        assert_eq!(d.snapshot_mask(), SNAP_MASK_ALL);
        let r = instr_restore(5);
        assert_eq!(r.opcode, OP_RESTORE);
        let d = Instruction::decode(&r.encode()).unwrap();
        assert_eq!(d.mnemonic(), "RESTORE");
        let p = instr_prefetch(0, 1024, 64);
        assert_eq!(p.opcode, OP_PREFETCH);
        assert_eq!(p.prefetch_params(), (1024, 64));
        let d = Instruction::decode(&p.encode()).unwrap();
        assert_eq!(d.mnemonic(), "PREFETCH");
        assert_eq!(d.prefetch_params(), (1024, 64));
        let h = instr_reshape(1, 0, &[1, 4]);
        assert_eq!(h.opcode, OP_RESHAPE);
        assert_eq!(h.reshape_shape(), (2, [1, 4, 0, 0]));
        let d = Instruction::decode(&h.encode()).unwrap();
        assert_eq!(d.mnemonic(), "RESHAPE");
        assert_eq!(d.reshape_shape(), (2, [1, 4, 0, 0]));
        let c = instr_concat(2, 0, 1, 1);
        assert_eq!(c.opcode, OP_CONCAT);
        assert_eq!(c.concat_axis(), 1);
        let d = Instruction::decode(&c.encode()).unwrap();
        assert_eq!(d.mnemonic(), "CONCAT");
        assert_eq!(d.concat_axis(), 1);
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("SNAPSHOT r5").unwrap();
        assert_eq!(prog[0].snapshot_mask(), SNAP_MASK_ALL);
        let prog = assemble("SNAPSHOT r5 MASK=7").unwrap();
        assert_eq!(prog[0].snapshot_mask(), 7);
        let prog = assemble("RESTORE r5").unwrap();
        assert_eq!(prog[0].rsrc1, 5);
        let prog = assemble("PREFETCH r0 LEN=1024 OFF=64").unwrap();
        assert_eq!(prog[0].prefetch_params(), (1024, 64));
        let prog = assemble("RESHAPE r1, r0 SHAPE=1x4").unwrap();
        assert_eq!(prog[0].reshape_shape(), (2, [1, 4, 0, 0]));
        let prog = assemble("RESHAPE r1, r0 SHAPE=2x2x2").unwrap();
        assert_eq!(prog[0].reshape_shape(), (3, [2, 2, 2, 0]));
        let prog = assemble("CONCAT r2, r0, r1 AXIS=1").unwrap();
        assert_eq!(prog[0].concat_axis(), 1);
        assert!(assemble("SNAPSHOT").is_err());
        assert!(assemble("SNAPSHOT r5 FOO=1").is_err());
        assert!(assemble("RESTORE").is_err());
        assert!(assemble("RESTORE r5 r6").is_err());
        assert!(assemble("PREFETCH").is_err());
        assert!(assemble("PREFETCH r0 FOO=1").is_err());
        assert!(assemble("RESHAPE r1, r0").is_err());
        assert!(assemble("RESHAPE r1, r0 SHAPE=0x4").is_err());
        assert!(assemble("RESHAPE r1, r0 SHAPE=1x2x3x4x5").is_err());
        assert!(assemble("RESHAPE r1, r0 SHAPE=abc").is_err());
        assert!(assemble("CONCAT r2, r0").is_err());
        assert!(assemble("CONCAT r2, r0, r1 FOO=1").is_err());
    }

    // ---- RFC-0025: conversão ----------------------------------------

    #[test]
    fn test_rfc0025_ctor_roundtrip() {
        let c = instr_cast(1, 0, CAST_DST_F16);
        assert_eq!(c.opcode, OP_CAST);
        assert_eq!(c.cast_dst(), CAST_DST_F16);
        let d = Instruction::decode(&c.encode()).unwrap();
        assert_eq!(d.mnemonic(), "CAST");
        assert_eq!(d.cast_dst(), CAST_DST_F16);
        let q = instr_quantize(3, 0, QUANTIZE_Q8_0);
        assert_eq!(q.opcode, OP_QUANTIZE);
        assert_eq!(q.quantize_type(), QUANTIZE_Q8_0);
        let d = Instruction::decode(&q.encode()).unwrap();
        assert_eq!(d.mnemonic(), "QUANTIZE");
        let e = instr_dequant(4, 3);
        assert_eq!(e.opcode, OP_DEQUANT);
        let d = Instruction::decode(&e.encode()).unwrap();
        assert_eq!(d.mnemonic(), "DEQUANT");
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("CAST r1, r0 DST=BF16").unwrap();
        assert_eq!(prog[0].cast_dst(), CAST_DST_BF16);
        let prog = assemble("CAST r1, r0 DST=2").unwrap();
        assert_eq!(prog[0].cast_dst(), 2);
        let prog = assemble("QUANTIZE r3, r0 Q=Q4_0").unwrap();
        assert_eq!(prog[0].quantize_type(), QUANTIZE_Q4_0);
        let prog = assemble("DEQUANT r4, r3").unwrap();
        assert_eq!(prog[0].opcode, OP_DEQUANT);
        assert!(assemble("CAST r1, r0").is_err());
        assert!(assemble("CAST r1, r0 DST=F64").is_err());
        assert!(assemble("CAST r1, r0 DST=9").is_err());
        assert!(assemble("CAST r1, r0 FOO=1").is_err());
        assert!(assemble("QUANTIZE r3, r0").is_err());
        assert!(assemble("QUANTIZE r3, r0 Q=Q4_K").is_err());
        assert!(assemble("QUANTIZE r3, r0 Q=Q6_K").is_err());
        assert!(assemble("QUANTIZE r3, r0 FOO=1").is_err());
        assert!(assemble("DEQUANT r4").is_err());
        assert!(assemble("DEQUANT r4, r3 EXTRA").is_err());
        // TENSOR ... bf16 aloca (DType::BF16 = 64, fora das numerações).
        let prog = assemble("TENSOR r0 2 2 bf16").unwrap();
        assert_eq!(prog[0].tensor_dtype(), 64);
        assert!(assemble("TENSOR r0 2 2 bf16x").is_err());
    }

    // ---- RFC-0026: ALU de control-plane -------------------------------

    #[test]
    fn test_rfc0026_ctor_roundtrip() {
        let a = instr_add_imm(1, 0, 23);
        assert_eq!(a.opcode, OP_ADD_IMM);
        assert_eq!(a.imm_u128(), 23);
        let d = Instruction::decode(&a.encode()).unwrap();
        assert_eq!(d.mnemonic(), "ADD_IMM");
        assert_eq!(d.imm_u128(), 23);
        let s = instr_sub_imm(2, 1, u128::MAX);
        assert_eq!(s.opcode, OP_SUB_IMM);
        assert_eq!(s.imm_u128(), u128::MAX);
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(d.mnemonic(), "SUB_IMM");
        let t = instr_steps(3);
        assert_eq!(t.opcode, OP_STEPS);
        let d = Instruction::decode(&t.encode()).unwrap();
        assert_eq!(d.mnemonic(), "STEPS");
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("ADD_IMM r1, r0 IMM=23").unwrap();
        assert_eq!(prog[0].imm_u128(), 23);
        let prog = assemble("SUB_IMM r2, r1 IMM=23").unwrap();
        assert_eq!(prog[0].imm_u128(), 23);
        let prog = assemble("STEPS r3").unwrap();
        assert_eq!(prog[0].opcode, OP_STEPS);
        assert!(assemble("ADD_IMM r1, r0").is_err());
        assert!(assemble("ADD_IMM r1, r0 IMM=-5").is_err());
        assert!(assemble("ADD_IMM r1, r0 IMM=abc").is_err());
        assert!(assemble("ADD_IMM r1, r0 FOO=1").is_err());
        assert!(assemble("SUB_IMM r2, r1").is_err());
        assert!(assemble("SUB_IMM r2, r1 FOO=1").is_err());
        assert!(assemble("STEPS").is_err());
        assert!(assemble("STEPS r3 EXTRA").is_err());
    }

    // ---- RFC-0027: forma ------------------------------------------------

    #[test]
    fn test_rfc0027_ctor_roundtrip() {
        let s = instr_sort(1, 0, 1, SORT_DESC);
        assert_eq!(s.opcode, OP_SORT);
        assert_eq!(s.sort_params(), (1, SORT_DESC));
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(d.mnemonic(), "SORT");
        let t = instr_topk(2, 0, 1, 2, 1, 1);
        assert_eq!(t.opcode, OP_TOPK);
        assert_eq!(t.topk_params(), (1, 2, 1, 1));
        let d = Instruction::decode(&t.encode()).unwrap();
        assert_eq!(d.mnemonic(), "TOPK");
        let a = instr_argmax(3, 0, 1);
        assert_eq!(a.opcode, OP_ARGMAX);
        assert_eq!(a.argmax_axis(), 1);
        let d = Instruction::decode(&a.encode()).unwrap();
        assert_eq!(d.mnemonic(), "ARGMAX");
        let r = instr_reduce(4, 0, REDUCE_SUM, 0xFF);
        assert_eq!(r.opcode, OP_REDUCE);
        assert_eq!(r.reduce_params(), (REDUCE_SUM, 0xFF));
        let d = Instruction::decode(&r.encode()).unwrap();
        assert_eq!(d.mnemonic(), "REDUCE");
        let b = instr_broadcast(5, 4, &[2, 2]);
        assert_eq!(b.opcode, OP_BROADCAST);
        assert_eq!(b.broadcast_shape(), (2, [2, 2, 0, 0]));
        let d = Instruction::decode(&b.encode()).unwrap();
        assert_eq!(d.mnemonic(), "BROADCAST");
        let p = instr_pad(6, 0, 0.5, 1, 1, 1);
        assert_eq!(p.opcode, OP_PAD);
        assert_eq!(p.pad_params(), (0.5, 1, 1, 1));
        let d = Instruction::decode(&p.encode()).unwrap();
        assert_eq!(d.mnemonic(), "PAD");
        let l = instr_tile(7, 0, 2, 0);
        assert_eq!(l.opcode, OP_TILE);
        assert_eq!(l.tile_params(), (2, 0));
        let d = Instruction::decode(&l.encode()).unwrap();
        assert_eq!(d.mnemonic(), "TILE");
        let x = instr_transpose(8, 0, &[1, 0]);
        assert_eq!(x.opcode, OP_TRANSPOSE);
        assert_eq!(x.transpose_perm(), (2, [1, 0, 0, 0]));
        let d = Instruction::decode(&x.encode()).unwrap();
        assert_eq!(d.mnemonic(), "TRANSPOSE");
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("SORT r1, r0 AXIS=1 ORDER=DESC").unwrap();
        assert_eq!(prog[0].sort_params(), (1, SORT_DESC));
        let prog = assemble("SORT r1, r0 DESC").unwrap();
        assert_eq!(prog[0].sort_params(), (0, SORT_DESC));
        let prog = assemble("TOPK r2, r0 K=2 AXIS=1 SMALLEST UNSORTED").unwrap();
        assert_eq!(prog[0].topk_params(), (1, 2, 0, 0));
        let prog = assemble("ARGMAX r3, r0 AXIS=1").unwrap();
        assert_eq!(prog[0].argmax_axis(), 1);
        let prog = assemble("REDUCE r4, r0 OP=MEAN AXIS=0").unwrap();
        assert_eq!(prog[0].reduce_params(), (REDUCE_MEAN, 0));
        let prog = assemble("REDUCE r4, r0 OP=SUM").unwrap();
        assert_eq!(prog[0].reduce_params(), (REDUCE_SUM, 0xFF));
        let prog = assemble("BROADCAST r5, r4 SHAPE=2x2").unwrap();
        assert_eq!(prog[0].broadcast_shape(), (2, [2, 2, 0, 0]));
        let prog = assemble("PAD r6, r0 VALUE=0.5 BEFORE=1 AFTER=2").unwrap();
        assert_eq!(prog[0].pad_params(), (0.5, 0, 1, 2));
        let prog = assemble("TILE r7, r0 REPS=3 AXIS=1").unwrap();
        assert_eq!(prog[0].tile_params(), (3, 1));
        let prog = assemble("TRANSPOSE r8, r0 AXES=1x0").unwrap();
        assert_eq!(prog[0].transpose_perm(), (2, [1, 0, 0, 0]));
        assert!(assemble("SORT r1").is_err());
        assert!(assemble("SORT r1, r0 ORDER=UP").is_err());
        assert!(assemble("SORT r1, r0 FOO=1").is_err());
        assert!(assemble("TOPK r2, r0").is_err());
        assert!(assemble("TOPK r2, r0 K=2 FOO=1").is_err());
        assert!(assemble("ARGMAX r3").is_err());
        assert!(assemble("REDUCE r4, r0").is_err());
        assert!(assemble("REDUCE r4, r0 OP=MEDIAN").is_err());
        assert!(assemble("BROADCAST r5, r4").is_err());
        assert!(assemble("PAD r6, r0").is_err());
        assert!(assemble("TILE r7, r0").is_err());
        assert!(assemble("TRANSPOSE r8, r0").is_err());
        assert!(assemble("TRANSPOSE r8, r0 AXES=1,0").is_err());
    }

    // ---- RFC-0028: ativações ------------------------------------------------

    #[test]
    fn test_rfc0028_ctor_roundtrip() {
        let s = instr_softmax(7, 0, 1, 0.5);
        assert_eq!(s.opcode, OP_SOFTMAX);
        assert_eq!(s.softmax_params(), (1, 0.5));
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(d.mnemonic(), "SOFTMAX");
        assert_eq!(d.softmax_params(), (1, 0.5));
        let g = instr_gelu(4, 0);
        assert_eq!(g.opcode, OP_GELU);
        let d = Instruction::decode(&g.encode()).unwrap();
        assert_eq!(d.mnemonic(), "GELU");
        for (ctor, name) in [
            (instr_sigmoid(1, 0), "SIGMOID"),
            (instr_tanh(2, 0), "TANH"),
            (instr_relu(3, 0), "RELU"),
            (instr_exp(5, 0), "EXP"),
            (instr_log(6, 0), "LOG"),
        ] {
            let d = Instruction::decode(&ctor.encode()).unwrap();
            assert_eq!(d.mnemonic(), name);
            assert_eq!((d.rdest, d.rsrc1), (ctor.rdest, ctor.rsrc1));
        }
        let c = instr_clip(8, 0, 0.0, 0.4);
        assert_eq!(c.opcode, OP_CLIP);
        assert_eq!(c.clip_params(), (0.0, 0.4));
        let d = Instruction::decode(&c.encode()).unwrap();
        assert_eq!(d.mnemonic(), "CLIP");
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("SOFTMAX r7, r0 AXIS=1 TEMP=0.5").unwrap();
        assert_eq!(prog[0].softmax_params(), (1, 0.5));
        let prog = assemble("SOFTMAX r7, r0").unwrap();
        assert_eq!(prog[0].softmax_params(), (0xFF, 1.0));
        let prog = assemble("CLIP r8, r0 MIN=0 MAX=0.4").unwrap();
        assert_eq!(prog[0].clip_params(), (0.0, 0.4));
        assert!(assemble("SOFTMAX r7").is_err());
        assert!(assemble("SOFTMAX r7, r0 FOO=1").is_err());
        assert!(assemble("GELU r4").is_err());
        assert!(assemble("GELU r4, r0 EXTRA").is_err());
        assert!(assemble("CLIP r8, r0").is_err());
        assert!(assemble("CLIP r8, r0 MIN=0").is_err());
        assert!(assemble("CLIP r8, r0 MIN=0 MAX=0.4 FOO=1").is_err());
    }

    // ---- RFC-0029: KV/attention -----------------------------------------

    #[test]
    fn test_rfc0029_ctor_roundtrip() {
        let k = instr_kv_compress(2, 3, KVCOMP_MODE_SINK_WINDOW, 0);
        assert_eq!(k.opcode, OP_KV_COMPRESS);
        assert_eq!(k.kv_compress_params(), (2, 3, KVCOMP_MODE_SINK_WINDOW, 0));
        let d = Instruction::decode(&k.encode()).unwrap();
        assert_eq!(d.mnemonic(), "KV_COMPRESS");
        let f = instr_flash_attn(4, 0, 1, 2, 2);
        assert_eq!(f.opcode, OP_FLASH_ATTN);
        assert_eq!(f.flash_block(), 2);
        assert_eq!((f.rdest, f.rsrc1, f.rsrc2, f.rsrc3), (4, 0, 1, 2));
        let d = Instruction::decode(&f.encode()).unwrap();
        assert_eq!(d.mnemonic(), "FLASH_ATTN");
        let s = instr_attn_sparse(3, 0, 1, 2, SPARSE_METRIC_DOT, 2);
        assert_eq!(s.opcode, OP_ATTN_SPARSE);
        assert_eq!(s.sparse_params(), (SPARSE_METRIC_DOT, 2));
        let d = Instruction::decode(&s.encode()).unwrap();
        assert_eq!(d.mnemonic(), "ATTN_SPARSE");
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("KV_COMPRESS SINK=2 WINDOW=3").unwrap();
        assert_eq!(prog[0].kv_compress_params(), (2, 3, KVCOMP_MODE_SINK_WINDOW, 0));
        let prog = assemble("FLASH_ATTN r4, r0, r1, r2 BLOCK=2").unwrap();
        assert_eq!(prog[0].flash_block(), 2);
        let prog = assemble("ATTN_SPARSE r3, r0, r1, r2 TOPK=2").unwrap();
        assert_eq!(prog[0].sparse_params(), (SPARSE_METRIC_DOT, 2));
        assert!(assemble("KV_COMPRESS SINK=2").is_err());
        assert!(assemble("KV_COMPRESS SINK=2 WINDOW=3 MODE=FOO").is_err());
        assert!(assemble("KV_COMPRESS SINK=2 WINDOW=3 FOO=1").is_err());
        assert!(assemble("FLASH_ATTN r4, r0, r1").is_err());
        assert!(assemble("FLASH_ATTN r4, r0, r1, r2 FOO=1").is_err());
        assert!(assemble("ATTN_SPARSE r3, r0, r1").is_err());
        assert!(assemble("ATTN_SPARSE r3, r0, r1, r2 METRIC=COSINE").is_err());
        assert!(assemble("ATTN_SPARSE r3, r0, r1, r2 FOO=1").is_err());
    }

    // ---- RFC-0031: DSP de áudio -----------------------------------------

    #[test]
    fn test_rfc0031_ctor_roundtrip() {
        let v = instr_vad_detect(1, 0, VAD_MODE_ZCR);
        assert_eq!(v.opcode, OP_VAD_DETECT);
        assert_eq!(v.vad_mode(), VAD_MODE_ZCR);
        let d = Instruction::decode(&v.encode()).unwrap();
        assert_eq!(d.mnemonic(), "VAD_DETECT");
        let m = instr_stream_merge(6, 5, 5, 0.5, 0);
        assert_eq!(m.opcode, OP_STREAM_MERGE);
        assert_eq!(m.merge_params(), (0.5, 0, 0));
        let d = Instruction::decode(&m.encode()).unwrap();
        assert_eq!(d.mnemonic(), "STREAM_MERGE");
        let r = instr_audio_resample(2, 0, 24000, 16000);
        assert_eq!(r.opcode, OP_AUDIO_RESAMPLE);
        assert_eq!(r.resample_rates(), (24000, 16000));
        let d = Instruction::decode(&r.encode()).unwrap();
        assert_eq!(d.mnemonic(), "AUDIO_RESAMPLE");
        let f = instr_audio_filter(5, 3, 4, FILTER_MODE_FIR);
        assert_eq!(f.opcode, OP_AUDIO_FILTER);
        assert_eq!(f.filter_mode(), FILTER_MODE_FIR);
        let d = Instruction::decode(&f.encode()).unwrap();
        assert_eq!(d.mnemonic(), "AUDIO_FILTER");
        let w = instr_audio_window(3, 2, WINDOW_HAMMING);
        assert_eq!(w.opcode, OP_AUDIO_WINDOW);
        assert_eq!(w.window_type(), WINDOW_HAMMING);
        let d = Instruction::decode(&w.encode()).unwrap();
        assert_eq!(d.mnemonic(), "AUDIO_WINDOW");
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("VAD_DETECT r1, r0 MODE=ZCR").unwrap();
        assert_eq!(prog[0].vad_mode(), VAD_MODE_ZCR);
        let prog = assemble("VAD_DETECT r1, r0").unwrap();
        assert_eq!(prog[0].vad_mode(), VAD_MODE_ENERGY);
        let prog = assemble("STREAM_MERGE r6, r5, r5 GAIN=0.5 N_STREAMS=17").unwrap();
        assert_eq!(prog[0].merge_params(), (0.5, 17, 0));
        let prog = assemble("AUDIO_RESAMPLE r2, r0 SRC=24000 DST=16000").unwrap();
        assert_eq!(prog[0].resample_rates(), (24000, 16000));
        let prog = assemble("AUDIO_FILTER r5, r3, r4 MODE=FIR").unwrap();
        assert_eq!(prog[0].filter_mode(), FILTER_MODE_FIR);
        let prog = assemble("AUDIO_WINDOW r3, r2 TYPE=HAMMING").unwrap();
        assert_eq!(prog[0].window_type(), WINDOW_HAMMING);
        assert!(assemble("VAD_DETECT r1").is_err());
        assert!(assemble("VAD_DETECT r1, r0 MODE=FOO").is_err());
        assert!(assemble("STREAM_MERGE r6, r5, r5").is_err());
        assert!(assemble("STREAM_MERGE r6, r5, r5 GAIN=0.5 FOO=1").is_err());
        assert!(assemble("AUDIO_RESAMPLE r2, r0 SRC=24000").is_err());
        assert!(assemble("AUDIO_FILTER r5, r3").is_err());
        assert!(assemble("AUDIO_FILTER r5, r3, r4 MODE=FOO").is_err());
        assert!(assemble("AUDIO_WINDOW r3").is_err());
        assert!(assemble("AUDIO_WINDOW r3, r2 TYPE=FOO").is_err());
    }

    // ---- RFC-0032: passo depformer ----------------------------------------

    #[test]
    fn test_rfc0032_ctor_roundtrip() {
        let d = instr_depformer(9, 0, 1, 2, 3, 4, 5, 6, 7, 0.8, 0);
        assert_eq!(d.opcode, OP_DEPFORMER);
        assert_eq!(d.depformer_params(), (2, 3, 4, 5, 6, 7, 0.8, 0));
        let r = Instruction::decode(&d.encode()).unwrap();
        assert_eq!(r.mnemonic(), "DEPFORMER");
        assert_eq!(r.depformer_params(), (2, 3, 4, 5, 6, 7, 0.8, 0));
        assert_eq!((r.rdest, r.rsrc1, r.rsrc2), (9, 0, 1));
        // Assembler: formas válidas + rejeições estritas.
        let prog = assemble("DEPFORMER r9, r0, r1 LAYER=2 NCB=4 NHEADS=2 LEVELS=8 CONTEXT=3 TEMP=0.8").unwrap();
        assert_eq!(prog[0].depformer_params(), (0, 2, 4, 2, 8, 3, 0.8, 0));
        let prog = assemble("DEPFORMER r9, r0, r1").unwrap();
        assert_eq!(prog[0].depformer_params(), (0, 0, 0, 0, 0, 0, 1.0, 0));
        assert!(assemble("DEPFORMER r9, r0").is_err());
        assert!(assemble("DEPFORMER r9, r0, r1 FOO=1").is_err());
        assert!(assemble("DEPFORMER r9, r0, r1 NCB=abc").is_err());
    }

    // ---- RFC-0034: sub-rotinas ------------------------------------------------

    #[test]
    fn test_rfc0034_ctor_roundtrip() {
        let c = instr_call(0x1040);
        assert_eq!(c.opcode, OP_CALL);
        assert_eq!(c.imm_u128(), 0x1040);
        let d = Instruction::decode(&c.encode()).unwrap();
        assert_eq!(d.mnemonic(), "CALL");
        assert_eq!(d.imm_u128(), 0x1040);
        let r = instr_ret();
        assert_eq!(r.opcode, OP_RET);
        let d = Instruction::decode(&r.encode()).unwrap();
        assert_eq!(d.mnemonic(), "RET");
        // Assembler: rótulos resolvidos + rejeições estritas.
        let prog = assemble("JUMP DONE\nDONE:\nCALL DONE\nRET\nHALT").unwrap();
        assert_eq!(prog[1].opcode, OP_CALL);
        assert_eq!(prog[1].imm_u128(), 0x1000 + 32); // DONE: = índice 1
        assert_eq!(prog[2].opcode, OP_RET);
        assert!(assemble("CALL").is_err());
        assert!(assemble("CALL NOPE").is_err());
        assert!(assemble("CALL DONE EXTRA").is_err());
        assert!(assemble("RET r0").is_err());
    }

    // ---- RFC-0008: modo estrito -------------------------------------

    #[test]
    fn test_rfc0008_rejects_unknown_tokens() {
        // Classes de silent-drop documentadas + formas legítimas preservadas.
        let bad = [
            "SENSE r0, AUDIO_PCM LEN=1920",   // extra ignorado antes
            "SENSE r0, FROBNICATE",           // periférico fantasma -> AUDIO
            "SAMPLE r1, r0 TEMPERATURE=0.5",  // TEMP= é o nome; resto ignorado
            "SAMPLE r1, r0 FOO",              // lixo total
            "ATTN r3, r0, r1, r2 NHEADS=32",  // KV de outra ISA
            "FORK r5, GRENN",                 // typo degradava p/ GREEN
            "FORK r5, RED, FOO",              // notificaria? não — ignorava
            "STREAM r3, r0 FLUSH",            // modo inventado (->BLOCKING)
            "STREAM r3, r0 BLOCKING EXTRA",   // cauda
            "CTX_SWITCH MAMBAH",              // pipe fantasma (->TRANSFORMER)
            "CTX_SWITCH MAMBA, FOO",          // prio fantasma ignorada
            "ROPE r2, r0 POS=0 HDIM=8 NHEADS=1 FOO=1",
            "GATHER r5, r0, r1 AXS=0",        // typo de AXIS=
            "DISTANCE r5, r0, r1 METRIC=HAMMING",
            "RANK1_UPDATE r7, r1, r2 ALPHA=abc",
            "RNG_UNIFORM r4 C=1.0",
            "ASSERT r1 CODE=abc",
            "SANITY_CHECK r4, r0, FOO",       // contagem silenciosamente 0xFF
            "LOCK r0 EXTRA",
            "HALT now",
            "DUMP verbose",
            "COMPARE r1, r2 FOO=1",
            "MATVEC r2, r0, r1 TRANSPOSE",    // TRANSPOSE é flag, não KV solto
            "TENSOR r0 2 2 f32 EXTRA",
            "NORM r2, r0, r1, r3 EXTRA",
            "JUMP DONE EXTRA",
        ];
        for src in bad {
            assert!(assemble(src).is_err(), "modo estrito deve rejeitar: {}", src);
        }
        // Formas legítimas (incl. posicionais e defaults) seguem passando.
        let good = [
            "TENSOR r0 2 2 f32",
            "TENSOR r0 32 32 f32 SPARSE DENSITY=0.05",
            "ATTN r3, r0, r1, r2 NOTIFY_EACH_HEAD",
            "STREAM r3, r0 BLOCKING",
            "STREAM r3, r0 DROP",
            "FORK r5, RED, NOTIFY",
            "FORK r5, MAIN_LOOP, GREEN",
            "SENSE r0, AUDIO_PCM",
            "SENSE r0, 6",
            "SAMPLE r1, r0",
            "SAMPLE r1, r0 0.7",
            "SAMPLE r1, r0 TEMP=0.7 TOPK=2",
            "COMPARE r1, r2 PRED=LT",
            "MATVEC r2, r0, r1",
            "SSM_SCAN r5, r0, r1, r0 D_INNER=2 D_STATE=1 LAYER=0",
            "ROPE r2, r0 POS=0 HDIM=8 NHEADS=1 INPLACE",
            "CTX_SWITCH MAMBA, RED",
            "CODEC_ENC r2, r0 TENSOR",
            "GATHER r5, r0, r1 AXIS=0",
            "SCATTER_ADD r5, r0, r1",
            "DISTANCE r5, r0, r1 METRIC=DOT TOPK=3",
            "RANK1_UPDATE r7, r1, r2 ALPHA=0.99 BETA=1.0 MODE=DELTA LAYER=0",
            "RNG_UNIFORM r4 A=-2.0 B=5.0",
            "ASSERT r1 CODE=42",
            "SANITY_CHECK r4, r0, r1",
            "LOCK r0",
            "HALT",
        ];
        for src in good {
            // FORK com rótulo precisa do label definido; testa isolado abaixo.
            if src.starts_with("FORK r5, MAIN_LOOP") {
                continue;
            }
            assert!(assemble(src).is_ok(), "forma legítima deve passar: {}", src);
        }
        let prog = assemble("MAIN_LOOP:\nFORK r5, MAIN_LOOP, GREEN\nJUMP MAIN_LOOP").unwrap();
        assert_eq!(prog[0].opcode, OP_FORK);
    }

    // ---- RFC-0036: apelidos simbólicos (passo V-1) ---------------------

    #[test]
    fn test_rfc0036_symbolic_regs() {
        // Declaração explícita vincula; uso resolve; numéricos intactos.
        // rBar sem declaração: erro que sugere `.reg` (RFC-0008 preservado).
        let err = assemble(".reg rFoo r5\nADD rFoo, rBar, rFoo").unwrap_err().to_string();
        assert!(err.contains(".reg"), "erro deve sugerir .reg: {}", err);
        // Programa completo declarado monta com os físicos certos.
        let prog = assemble(".reg rFoo r5\n.reg rBar r1\nADD rFoo, rBar, rFoo").unwrap();
        assert_eq!((prog[0].rdest, prog[0].rsrc1, prog[0].rsrc2), (5, 1, 5));
        // Maiúsculas/minúsculas: insensível (lowercase interno).
        let prog = assemble(".reg rFoo r5\nMOV rFoo, rFOO").unwrap();
        assert_eq!((prog[0].rdest, prog[0].rsrc1), (5, 5));
        // Tabela nova por chamada: determinístico entre arquivos.
        let a = assemble(".reg rFoo r5\nADD rFoo, r0, r1").unwrap();
        let b = assemble(".reg rFoo r5\nADD rFoo, r0, r1").unwrap();
        assert_eq!((a[0].rdest, b[0].rdest), (5, 5));
        // Redeclaração, físico ocupado, físico fora: tudo erro alto.
        assert!(assemble(".reg rFoo r5\n.reg rFoo r6\nADD rFoo, r0, r1").is_err());
        assert!(assemble(".reg rFoo r5\n.reg rBar r5\nADD rFoo, rBar, r1").is_err());
        assert!(assemble(".reg rFoo r16\nADD rFoo, r0, r1").is_err());
        assert!(assemble(".reg rFoo\nADD rFoo, r0, r1").is_err());
        assert!(assemble(".reg Foo r5\nADD rFoo, r0, r1").is_err());
        // Formas inválidas continuam erro (nada afrouxado).
        assert!(assemble("ADD r1x, r0, r1").is_err());
        assert!(assemble("ADD r, r0, r1").is_err());
        assert!(assemble("ADD r16, r0, r1").is_err());
        assert!(assemble("ADD rFoo-bar, r0, r1").is_err());
        // Diretiva não emite instrução: só o ADD aparece.
        let prog = assemble(".reg rFoo r5\nADD rFoo, r0, r1").unwrap();
        assert_eq!(prog.len(), 1);
    }

    // ---- RFC-0037: constantes `.equ` + `.text` (V-1a) -------------------

    #[test]
    fn test_rfc0037_equ_text() {
        // `.equ` vincula; uso em imediatos resolve (LOADI/ADD_IMM/SUB_IMM/
        // COMPARE/TENSOR-dims/TREES/DEPTH/START/LEN/CODE).
        let prog = assemble(".equ BASE 10\nLOADI r0, BASE").unwrap();
        assert_eq!(prog.len(), 1);
        assert_eq!(prog[0].mnemonic(), "LOADI");
        // Hex e case-insensitividade.
        let a = assemble(".equ STEP 0x20\nADD_IMM r1, r0 IMM=STEP").unwrap();
        let b = assemble(".equ step 32\nADD_IMM r1, r0 IMM=step").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        // `.text` é no-op: zero instruções emitidas, resto monta.
        let prog = assemble(".text\n.equ N 7\nLOADI r2, N\nHALT").unwrap();
        assert_eq!(prog.len(), 2);
        // `.equ` + `.reg` convivem (namespaces posicionais).
        let prog = assemble(".reg rFoo r5\n.equ N 7\nADD rFoo, r0, r1\nLOADI r2, N").unwrap();
        assert_eq!((prog[0].rdest, prog.len()), (5, 2));
        // TENSOR dims via const; mesma codificação que literal.
        let a = assemble(".equ NR 2\n.equ NC 4\nTENSOR r1 NR NC f32").unwrap();
        let b = assemble("TENSOR r1 2 4 f32").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        // FOREST TREES/DEPTH e SLICE/ASSERT via const.
        let a = assemble(".equ T 8\n.equ D 6\nFOREST r4, r0, r1, r2 TREES=T DEPTH=D").unwrap();
        let b = assemble("FOREST r4, r0, r1, r2 TREES=8 DEPTH=6").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        let a = assemble(".equ S 3\n.equ L 3\nSLICE r4, r0 START=S LEN=L").unwrap();
        let b = assemble("SLICE r4, r0 START=3 LEN=3").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        let a = assemble(".equ C 42\nASSERT r1 CODE=C").unwrap();
        let b = assemble("ASSERT r1 CODE=42").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        // COMPARE imediato via const.
        let a = assemble(".equ LIM 32\nCOMPARE r2, LIM PRED=EQ").unwrap();
        let b = assemble("COMPARE r2, 32 PRED=EQ").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        // SUB_IMM via const.
        let a = assemble(".equ BASE 10\nSUB_IMM r2, r1 IMM=BASE").unwrap();
        let b = assemble("SUB_IMM r2, r1 IMM=10").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        // Determinismo entre chamadas.
        let a = assemble(".equ N 7\nLOADI r0, N").unwrap();
        let b = assemble(".equ N 7\nLOADI r0, N").unwrap();
        assert_eq!(a[0].encode(), b[0].encode());
        // Erros altos (nada silencioso): sem declaração, redeclaração,
        // valor ruim, aridade ruim, nome ruim/reservado/colisão com reg.
        assert!(assemble("LOADI r0, NOPE").is_err());
        assert!(assemble("ADD_IMM r1, r0 IMM=NOPE").is_err());
        assert!(assemble("COMPARE r1, NOPE").is_err());
        assert!(assemble("TENSOR r1 NOPE 4 f32").is_err());
        assert!(assemble(".equ N 1\n.equ N 2\nLOADI r0, N").is_err());
        assert!(assemble(".equ N abc\nLOADI r0, N").is_err());
        assert!(assemble(".equ N -5\nLOADI r0, N").is_err());
        assert!(assemble(".equ N\nLOADI r0, N").is_err());
        assert!(assemble(".equ N 1 2\nLOADI r0, N").is_err());
        assert!(assemble(".equ 1N 5\nLOADI r0, 1N").is_err());
        assert!(assemble(".equ F32 5\nLOADI r0, F32").is_err());
        assert!(assemble(".equ R1 5\nLOADI r0, R1").is_err());
        assert!(assemble(".equ BIG 99999999999999999999999\nTENSOR r1 BIG 4 f32").is_err());
        assert!(assemble(".equ BIG 70000\nASSERT r1 CODE=BIG").is_err());
        assert!(assemble(".text extra\nHALT").is_err());
        // `.data`/`.str` (V-1b dia 1): `.data` abre seção de blobs —
        // instrução lá dentro erra; `assemble()` com blobs erra pedindo
        // `assemble_with_data`; `.str` nu erra (é tipo de blob).
        assert!(assemble(".data\nHALT").is_err());
        let err = assemble(".data\nx: .u32 1\n.text\nHALT").unwrap_err().to_string();
        assert!(err.contains("assemble_with_data"), "erro deve apontar assemble_with_data: {}", err);
        let full = assemble_with_data(".data\nx: .u32 1\n.text\nHALT").unwrap();
        assert_eq!((full.instrs.len(), full.data.blobs.len()), (1, 1));
        let err = assemble(".str\nHALT").unwrap_err().to_string();
        assert!(err.contains(".data"), "erro deve apontar .data: {}", err);
        // Faixa preservada: TREES=0 via const continua erro.
        assert!(assemble(".equ Z 0\nFOREST r4, r0, r1, r2 TREES=Z").is_err());
    }

    // ---- V-1b dia 1: `.data` escalar + `.str` (parser, sem loader) ----

    #[test]
    fn test_v1b_data_scalar_str() {
        use super::{assemble_with_data, DataDtype};
        // Escalares: bytes LE exatos.
        let full = assemble_with_data(".data\nframe_len: .u32 1920\n.text\nHALT").unwrap();
        assert_eq!(full.instrs.len(), 1);
        assert_eq!(full.data.blobs.len(), 1);
        let b = &full.data.blobs[0];
        assert_eq!((b.name.as_str(), b.dtype), ("frame_len", DataDtype::U32));
        assert_eq!(b.bytes, 1920u32.to_le_bytes().to_vec());
        // Hex + `.equ` no `.u32`; `.i32` negativo e via const.
        let full = assemble_with_data(".equ N 7\n.data\na: .u32 0x20\nb: .u32 N\nc: .i32 -5\nd: .i32 N\n.text\nHALT").unwrap();
        assert_eq!(full.data.blobs.len(), 4);
        assert_eq!(full.data.blobs[0].bytes, 32u32.to_le_bytes().to_vec());
        assert_eq!(full.data.blobs[1].bytes, 7u32.to_le_bytes().to_vec());
        assert_eq!(full.data.blobs[2].bytes, (-5i32).to_le_bytes().to_vec());
        assert_eq!(full.data.blobs[3].bytes, 7i32.to_le_bytes().to_vec());
        // `.f32` bit-exato; `.str` UTF-8 cru.
        let full = assemble_with_data(".data\nt: .f32 0.5\nf: .str \"Hmm...\"\n.text\nHALT").unwrap();
        assert_eq!(full.data.blobs[0].bytes, 0.5f32.to_le_bytes().to_vec());
        assert_eq!(full.data.blobs[1].dtype, DataDtype::Str);
        assert_eq!(full.data.blobs[1].bytes, b"Hmm...".to_vec());
        assert_eq!(full.data.total_bytes(), 4 + 6);
        assert_eq!(full.data.find("T").unwrap().dtype, DataDtype::F32);
        // `.reg`/`.equ` valem no arquivo todo, em qualquer seção.
        let full = assemble_with_data(".data\nn: .u32 BASE\n.text\n.reg rFoo r5\n.equ BASE 9\nADD rFoo, r0, r1").unwrap();
        assert_eq!(full.data.blobs[0].bytes, 9u32.to_le_bytes().to_vec());
        assert_eq!(full.instrs[0].rdest, 5);
        // Seção default é `.text`; idempotência de `.text`/`.data`.
        let full = assemble_with_data("HALT\n.text\nHALT\n.data\na: .u32 1\n.data\nb: .u32 2\n.text\nHALT").unwrap();
        assert_eq!((full.instrs.len(), full.data.blobs.len()), (3, 2));
        // Erros altos: instrução/rótulo-nu em `.data`, tipo ruim, valor
        // ruim, aridade ruim, nome ruim/duplicado, NaN/Inf, aspas,
        // multi-valor (dia 2), colisão com rótulo de código.
        assert!(assemble_with_data(".data\nHALT").is_err());
        assert!(assemble_with_data(".data\nfoo:\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .u64 1\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .u32\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .u32 1 2\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .u32 abc\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .u32 4294967296\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .i32 2147483648\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .f32 abc\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .f32 NaN\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .f32 inf\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .str hi\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .str \"a\"b\"\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\n1x: .u32 1\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nx: .u32 1\nx: .u32 2\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [0.1, 0.2]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nLOOP: .u32 1\n.text\nLOOP:\nHALT").is_err());
        assert!(assemble_with_data(".data extra\nHALT").is_err());
    }

    // ---- V-1b dia 2: multi-valor nested `.f32 [shape] [vals]` ----------

    #[test]
    fn test_v1b_data_nested() {
        use super::{assemble_with_data, DataDtype};
        fn f32s(v: &[f32]) -> Vec<u8> {
            v.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        // Vetor rank-1 e matriz rank-2: bytes LE bit-exatos + shape.
        let full = assemble_with_data(".data\nv: .f32 [4] [1.0, 2.0, 3.0, 4.0]\n.text\nHALT").unwrap();
        assert_eq!(full.data.blobs[0].shape, vec![4]);
        assert_eq!(full.data.blobs[0].bytes, f32s(&[1.0, 2.0, 3.0, 4.0]));
        let full = assemble_with_data(".data\nm: .f32 [2, 2] [1.0, 0.0, 0.0, 1.0]\n.text\nHALT").unwrap();
        assert_eq!(full.data.blobs[0].shape, vec![2, 2]);
        assert_eq!(full.data.blobs[0].bytes, f32s(&[1.0, 0.0, 0.0, 1.0]));
        assert_eq!(full.data.blobs[0].dtype, DataDtype::F32);
        // Whitespace normalizado: `[4,4]`, `[4 , 4]`, hex e `.equ` em dims.
        let a = assemble_with_data(".data\nm: .f32 [2,2] [1.0, 0.0, 0.0, 1.0]\n.text\nHALT").unwrap();
        let b = assemble_with_data(".data\nm: .f32 [2 , 2] [1.0, 0.0, 0.0, 1.0]\n.text\nHALT").unwrap();
        assert_eq!(a.data.blobs[0].bytes, b.data.blobs[0].bytes);
        let full = assemble_with_data(".equ N 2\n.data\nm: .f32 [N, 0x2] [1.0, 0.0, 0.0, 1.0]\n.text\nHALT").unwrap();
        assert_eq!(full.data.blobs[0].shape, vec![2, 2]);
        // Negativos e expoentes.
        let full = assemble_with_data(".data\nv: .f32 [3] [-1.5, 1e3, 0.25]\n.text\nHALT").unwrap();
        assert_eq!(full.data.blobs[0].bytes, f32s(&[-1.5, 1000.0, 0.25]));
        // Mismatch nomeia esperados/obtidos.
        let err = assemble_with_data(".data\nv: .f32 [4] [1.0, 2.0]\n.text\nHALT").unwrap_err().to_string();
        assert!(err.contains("esperados 4") && err.contains("obtidos 2"), "mismatch deve nomear N/M: {}", err);
        // Erros altos: zero, rank-3, trailing/leading/dupla vírgula, shape
        // sem valores, valores sem shape, JSON-style, vazio, f32-only,
        // NaN, sobra após bloco, colchete aberto, escalar intacto.
        assert!(assemble_with_data(".data\nm: .f32 [0, 4] [1.0, 0.0, 0.0, 1.0]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nt: .f32 [1, 2, 3] [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [2] [1.0, 2.0, ]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [2] [, 1.0, 2.0]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [2] [1.0,, 2.0]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [4]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [1.0, 2.0]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nm: .f32 [[1.0, 2.0], [3.0, 4.0]]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [2] []\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .u32 [2] [1, 2]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [2] [1.0, NaN]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [2] [1.0, 2.0] EXTRA\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [2 [1.0, 2.0]\n.text\nHALT").is_err());
        assert!(assemble_with_data(".data\nv: .f32 [65536, 65536] [1.0]\n.text\nHALT").is_err());
        // Escalar dia 1 intacto ao lado de nested.
        let full = assemble_with_data(".data\ns: .f32 0.5\nm: .f32 [2] [1.0, 2.0]\n.text\nHALT").unwrap();
        assert_eq!(full.data.blobs.len(), 2);
        assert_eq!(full.data.blobs[0].shape.len(), 0);
        assert_eq!(full.data.blobs[1].shape, vec![2]);
    }

    // ---- V-1b dia 3: `@nome` (Opção A, assemble-time) ------------------

    #[test]
    fn test_v1b_data_addr() {
        use super::{assemble_with_data, data_layout_addrs, data_layout_total, DATA_LOAD_BASE};
        // Layout fixo: base 0x1000, align 4/1, pad 8 entre blobs.
        let full = assemble_with_data(
            ".data\nframe_len: .u32 1920\nthreshold: .f32 0.5\ntag: .str \"hi\"\ntab: .f32 [2, 2] [1.0, 0.0, 0.0, 1.0]\n.text\nHALT",
        )
        .unwrap();
        let addrs = data_layout_addrs(&full.data);
        assert_eq!(
            addrs,
            vec![
                ("frame_len".to_string(), DATA_LOAD_BASE),
                ("threshold".to_string(), DATA_LOAD_BASE + 8),
                ("tag".to_string(), DATA_LOAD_BASE + 16),
                ("tab".to_string(), DATA_LOAD_BASE + 24),
            ]
        );
        assert_eq!(DATA_LOAD_BASE, 0x1000);
        assert_eq!(data_layout_total(&full.data), 40);
        assert!(data_layout_total(&super::DataSection::default()) == 0);
        // `@nome` num único LOADI = mesmo bytes do literal do endereço.
        let a = assemble_with_data(".data\nframe_len: .u32 1920\n.text\nLOADI r0, @frame_len").unwrap();
        let b = assemble("LOADI r0, 4096").unwrap();
        assert_eq!(a.instrs[0].encode(), b[0].encode());
        // Case-insensitive; `.equ` no caminho (forward-reference ok).
        let full = assemble_with_data(".equ N 3\n.data\nv: .f32 [N] [1.0, 2.0, 3.0]\n.text\nLOADI r0, @V").unwrap();
        assert_eq!(full.instrs.len(), 1);
        // Erros altos: `@` de `.equ` (só `.data` tem endereço), `@`
        // inexistente, `@` nu, `@` fora de LOADI, colisão blob×`.equ`.
        assert!(assemble_with_data(".equ N 5\n.text\nLOADI r0, @N").is_err());
        assert!(assemble_with_data(".text\nLOADI r0, @nope").is_err());
        assert!(assemble_with_data(".data\nx: .u32 1\n.text\nLOADI r0, @").is_err());
        assert!(assemble_with_data(".data\nx: .u32 1\n.text\nCOMPARE r0, @x").is_err());
        assert!(assemble_with_data(".data\nx: .u32 1\n.text\nADD_IMM r0, r1 IMM=@x").is_err());
        let err = assemble_with_data(".equ FOO 1\n.data\nfoo: .u32 2\n.text\nHALT")
            .unwrap_err()
            .to_string();
        assert!(err.contains("colide"), "colisão blob×const deve errar: {}", err);
        // `@` case-insensitive como todo o assembler (`@FOO` = `@foo`).
        let a = assemble_with_data(".data\nfoo: .u32 2\n.text\nLOADI r0, @FOO").unwrap();
        let b = assemble_with_data(".data\nfoo: .u32 2\n.text\nLOADI r0, @foo").unwrap();
        assert_eq!(a.instrs[0].encode(), b.instrs[0].encode());
        let err = assemble_with_data(".data\nfoo: .u32 2\n.text\nLOADI r0, @bar")
            .unwrap_err()
            .to_string();
        assert!(err.contains("@bar"), "erro deve nomear @: {}", err);
    }
}
