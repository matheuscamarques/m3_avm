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
// Métodos — ODE_STEP (payload[4]): 0=Euler, 1=RK2, 2=RK4.
pub const ODE_METHOD_EULER: u8 = 0;
pub const ODE_METHOD_RK2: u8 = 1;
pub const ODE_METHOD_RK4: u8 = 2;
// W4 pull-forward (RFC-0012): ensemble de árvores vetorizado.
pub const OP_FOREST: u8 = 0x22; // XGBoost/RF: walk sobre tabela plana
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
            OP_DENOISE_STEP => "DENOISE_STEP",
            OP_ODE_STEP => "ODE_STEP",
            OP_SPIKE_STEP => "SPIKE_STEP",
            OP_FOREST => "FOREST",
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
    assemble_with_base(text, 0x1000)
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
    let mut labels: HashMap<String, u128> = HashMap::new();
    let mut instr_lines: Vec<(usize, String)> = Vec::new();

    // Passo 1: Varredura de rótulos e mapeamento de instruções
    for (lineno, raw) in text.lines().enumerate() {
        let mut line = strip_comment(raw).trim();
        if line.is_empty() {
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

    // Passo 2: Montagem com resolução de rótulos
    let mut out = Vec::with_capacity(instr_lines.len());
    for (lineno, line) in instr_lines {
        let instr = parse_line(&line, &labels).map_err(|e| anyhow!("linha {}: {} — '{}'", lineno, e, line))?;
        out.push(instr);
    }
    Ok(out)
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

fn parse_reg(tok: &str) -> Result<u8> {
    let t = tok.trim().trim_end_matches(',').to_lowercase();
    if t == "_" || t == "x" || t == "-" || t == "ff" {
        return Ok(0xFF);
    }
    let t = t.trim_start_matches('r');
    let n: u8 = t.parse().map_err(|_| anyhow!("registrador inválido '{}'", tok))?;
    if n >= 16 {
        return Err(anyhow!("registrador r{} fora de 0..15", n));
    }
    Ok(n)
}

fn parse_line(line: &str, labels: &HashMap<String, u128>) -> Result<Instruction> {
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
            let rdest = parse_reg(parts[1])?;
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
                // Se parts[2] é número ou contém 'x'
                let maybe_rows = parts[2].replace('x', " ").trim().to_string();
                // Checa se é literal numérico
                if maybe_rows.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) || parts[2].contains('x') {
                    // Literal
                    let (rows, cols) = if parts[2].contains('x') {
                        let mut split = parts[2].split('x');
                        let r = split.next().unwrap().parse::<u64>().map_err(|_| anyhow!("rows inválido"))?;
                        let c = split.next().unwrap().parse::<u64>().map_err(|_| anyhow!("cols inválido"))?;
                        (r, c)
                    } else if parts.len() >= 4 && parts[3].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(false) {
                        let r = parts[2].parse::<u64>().map_err(|_| anyhow!("rows inválido"))?;
                        let c = parts[3].parse::<u64>().map_err(|_| anyhow!("cols inválido"))?;
                        (r, c)
                    } else {
                        (2, 2)
                    };
                    let dtype_str = if parts.len() >= 5 {
                        parts[4]
                    } else if parts.len() == 4 && !parts[3].chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true) {
                        parts[3]
                    } else {
                        "f32"
                    };
                    let dtype = parse_dtype(dtype_str)?;
                    // Detecta SPARSE/DENSITY nos tokens restantes (após dtype)
                    let remaining = if parts.len() > 5 { &parts[5..] } else { &[] as &[&str] };
                    let (is_sparse, dens) = detect_sparse(remaining);
                    reject_unknown("TENSOR", remaining, &["SPARSE", "DENSITY="])?;
                    let mut instr = instr_tensor(rdest, 0xFF, 0xFF, rows, cols, dtype);
                    if is_sparse { instr.set_sparse(true, dens); }
                    return Ok(instr);
                }
            }
            // Forma com registradores
            let rsrc1 = if parts.len() > 2 { parse_reg(parts[2])? } else { 0xFF };
            let rsrc2 = if parts.len() > 3 { parse_reg(parts[3])? } else { 0xFF };
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
            reject_unknown("TENSOR", remaining, &["SPARSE", "DENSITY="])?;
            let mut instr = instr_tensor(rdest, rsrc1, rsrc2, rows, cols, dtype);
            if is_sparse { instr.set_sparse(true, dens); }
            Ok(instr)
        }
        "ATTN" => {
            if parts.len() < 5 {
                return Err(anyhow!("ATTN precisa de rdest, rQ, rK, rV — ex: ATTN r3, r0, r1, r2"));
            }
            let rdest = parse_reg(parts[1])?;
            let rq = parse_reg(parts[2])?;
            let rk = parse_reg(parts[3])?;
            let rv = parse_reg(parts[4])?;
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
            let rsrc = parse_reg(parts[1])?;
            // Suporta periféricos simbólicos: INPUT (1), SAMPLE (2), DECODED (4), OUTPUT (0)
            let rsink = match parts[2].to_ascii_uppercase().as_str() {
                "INPUT" | "PERIPHERAL_INPUT" | "1" => 1,
                "SAMPLE" | "PERIPHERAL_SAMPLE" | "2" => 2,
                "DECODED" | "OUTPUT_DECODED" | "PERIPHERAL_OUTPUT_DECODED" | "4" => 4,
                "OUTPUT" | "PERIPHERAL_OUTPUT" | "0" => 0,
                _ => parse_reg(parts[2])?,
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
            let rdest = parse_reg(parts[1])?;
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
            let rt = parse_reg(parts[1])?;
            let ts = parse_reg(parts[2])?;
            if parts.len() > 3 {
                reject_unknown("ABORT", &parts[3..], &[])?;
            }
            Ok(instr_abort(rt, ts))
        }
        "SENSE" => {
            if parts.len() < 3 {
                return Err(anyhow!("SENSE precisa de rdest, periférico (AUDIO/VAD/TOKEN/USER_INPUT/AUDIO_PCM/CODEC_FRAME/0/1/3/5/6/7)"));
            }
            let rdest = parse_reg(parts[1])?;
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
            let rdest = parse_reg(parts[1])?;
            let rsrc = parse_reg(parts[2])?;
            let rgamma = if parts.len() > 3 { parse_reg(parts[3])? } else { 0xFF };
            let rbeta = if parts.len() > 4 { parse_reg(parts[4])? } else { 0xFF };
            if parts.len() > 5 {
                reject_unknown("NORM", &parts[5..], &[])?;
            }
            Ok(instr_norm(rdest, rsrc, rgamma, rbeta))
        }
        "FFN" => {
            if parts.len() < 4 {
                return Err(anyhow!("FFN precisa de rdest, r_src, r_w1, r_w2 — ex: FFN r3, r0, r1, r2 ou FFN r3, r0, r1, r4, r2, r5 (com bias)"));
            }
            let rdest = parse_reg(parts[1])?;
            let rsrc = parse_reg(parts[2])?;
            let rw1 = parse_reg(parts[3])?;
            if parts.len() >= 7 {
                let rb1 = parse_reg(parts[4])?;
                let rw2 = parse_reg(parts[5])?;
                let rb2 = parse_reg(parts[6])?;
                if parts.len() > 7 {
                    reject_unknown("FFN", &parts[7..], &[])?;
                }
                Ok(instr_ffn_with_bias(rdest, rsrc, rw1, rb1, rw2, rb2))
            } else if parts.len() == 6 {
                let rw2 = parse_reg(parts[4])?;
                let maybe_rb = parse_reg(parts[5])?;
                let mut instr = instr_ffn(rdest, rsrc, rw1, rw2);
                instr.payload[1] = maybe_rb;
                Ok(instr)
            } else {
                let rw2 = parse_reg(parts[4])?;
                Ok(instr_ffn(rdest, rsrc, rw1, rw2))
            }
        }
        "EMBED" => {
            if parts.len() < 4 {
                return Err(anyhow!("EMBED precisa de rdest, r_token, r_table — ex: EMBED r10, r11, r1"));
            }
            let rdest = parse_reg(parts[1])?;
            let rtoken = parse_reg(parts[2])?;
            let rtable = parse_reg(parts[3])?;
            if parts.len() > 4 {
                reject_unknown("EMBED", &parts[4..], &[])?;
            }
            Ok(instr_embed(rdest, rtoken, rtable))
        }
        "ADD" => {
            if parts.len() < 4 {
                return Err(anyhow!("ADD precisa de rdest, r_src1, r_src2 — ex: ADD r10, r10, r14"));
            }
            let rdest = parse_reg(parts[1])?;
            let rsrc1 = parse_reg(parts[2])?;
            let rsrc2 = parse_reg(parts[3])?;
            if parts.len() > 4 {
                reject_unknown("ADD", &parts[4..], &[])?;
            }
            Ok(instr_add(rdest, rsrc1, rsrc2))
        }
        "SAMPLE" => {
            if parts.len() < 3 {
                return Err(anyhow!("SAMPLE precisa de rdest, r_logits — ex: SAMPLE r11, r15"));
            }
            let rdest = parse_reg(parts[1])?;
            let rlogits = parse_reg(parts[2])?;
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
            let rsrc1 = parse_reg(parts[1])?;
            let second = parts[2].to_ascii_uppercase();
            let mut instr = if second == "EOS_TOKEN" {
                instr_compare(rsrc1, 0xFF, EOS_TOKEN_DEFAULT)
            } else if let Ok(r2) = parse_reg(parts[2]) {
                instr_compare(rsrc1, r2, 0)
            } else if let Ok(n) = parts[2].parse::<u128>() {
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
                let rcond = parse_reg(parts[1])?;
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
            let rdest = parse_reg(parts[1])?;
            let rx = parse_reg(parts[2])?;
            let rw = parse_reg(parts[3])?;
            if parts.len() > 4 {
                reject_unknown("MATVEC", &parts[4..], &[])?;
            }
            Ok(instr_matvec(rdest, rx, rw))
        }
        "MUL" => {
            if parts.len() < 4 {
                return Err(anyhow!("MUL precisa de rdest, r1, r2 — ex: MUL r6, r6, r7"));
            }
            let rdest = parse_reg(parts[1])?;
            let r1 = parse_reg(parts[2])?;
            let r2 = parse_reg(parts[3])?;
            if parts.len() > 4 {
                reject_unknown("MUL", &parts[4..], &[])?;
            }
            Ok(instr_mul(rdest, r1, r2))
        }
        "SILU" => {
            if parts.len() < 3 {
                return Err(anyhow!("SILU precisa de rdest, rsrc — ex: SILU r6, r6"));
            }
            let rdest = parse_reg(parts[1])?;
            let rsrc = parse_reg(parts[2])?;
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
            let rdest = parse_reg(parts[1])?;
            let rx = parse_reg(parts[2])?;
            let rh = parse_reg(parts[3])?;
            let rp = parse_reg(parts[4])?;
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
            let rh = parse_reg(parts[1])?;
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
            let rdest = parse_reg(parts[1])?;
            let rpcm = parse_reg(parts[2])?;
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
            let rdest = parse_reg(parts[1])?;
            let rc = parse_reg(parts[2])?;
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
            let rdest = parse_reg(parts[1])?;
            let ru = parse_reg(parts[2])?;
            let ra = parse_reg(parts[3])?;
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
                    if parse_reg(parts[1]).is_ok() {
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
            let rdest = parse_reg(parts[1])?;
            let rsrc = parse_reg(parts[2])?;
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
            let rdest = parse_reg(parts[1])?;
            let rtable = parse_reg(parts[2])?;
            let ridx = parse_reg(parts[3])?;
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
                    if let Ok(r) = parse_reg(parts[rpos]) {
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
            let rdest = parse_reg(parts[1])?;
            let rquery = parse_reg(parts[2])?;
            let rbank = parse_reg(parts[3])?;
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
            let rh = parse_reg(parts[1])?;
            let rv = parse_reg(parts[2])?;
            let rk = parse_reg(parts[3])?;
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
                Ok(instr_rng_seed(parse_reg(parts[1])?))
            }
        }
        "RNG_NEXT" => {
            if parts.len() < 2 {
                return Err(anyhow!("RNG_NEXT precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("RNG_NEXT", &parts[2..], &[])?;
            }
            Ok(instr_rng_next(parse_reg(parts[1])?))
        }
        "RNG_UNIFORM" => {
            // RNG_UNIFORM rD [A=a] [B=b]
            if parts.len() < 2 {
                return Err(anyhow!("RNG_UNIFORM precisa de rdest"));
            }
            let rdest = parse_reg(parts[1])?;
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
            let rdest = parse_reg(parts[1])?;
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
            Ok(instr_hash(parse_reg(parts[1])?, parse_reg(parts[2])?))
        }
        "CHECKSUM" => {
            if parts.len() < 3 {
                return Err(anyhow!("CHECKSUM precisa de rdest, rTensor — ex: CHECKSUM r4, r0"));
            }
            if parts.len() > 3 {
                reject_unknown("CHECKSUM", &parts[3..], &[])?;
            }
            Ok(instr_checksum(parse_reg(parts[1])?, parse_reg(parts[2])?))
        }
        "HMAC" => {
            if parts.len() < 4 {
                return Err(anyhow!("HMAC precisa de rdest, rKey, rMsg — ex: HMAC r4, r0, r1"));
            }
            if parts.len() > 4 {
                reject_unknown("HMAC", &parts[4..], &[])?;
            }
            Ok(instr_hmac(parse_reg(parts[1])?, parse_reg(parts[2])?, parse_reg(parts[3])?))
        }
        "CYCLES_COUNT" => {
            if parts.len() < 2 {
                return Err(anyhow!("CYCLES_COUNT precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("CYCLES_COUNT", &parts[2..], &[])?;
            }
            Ok(instr_cycles_count(parse_reg(parts[1])?))
        }
        "TRACE_EVENT" => {
            if parts.len() < 3 {
                return Err(anyhow!("TRACE_EVENT precisa de rEv, rData"));
            }
            if parts.len() > 3 {
                reject_unknown("TRACE_EVENT", &parts[3..], &[])?;
            }
            Ok(instr_trace_event(parse_reg(parts[1])?, parse_reg(parts[2])?))
        }
        "SANITY_CHECK" => {
            // SANITY_CHECK rD, rT [, rCount]
            if parts.len() < 3 {
                return Err(anyhow!("SANITY_CHECK precisa de rdest, rTensor — ex: SANITY_CHECK r4, r0"));
            }
            // RFC-0008: 4º token, se presente, DEVE ser registrador (antes:
            // lixo silenciosamente virava 0xFF = sem contagem).
            let rcount = if parts.len() > 3 {
                parse_reg(parts[3]).map_err(|_| anyhow!("SANITY_CHECK 4º operando '{}' inválido (use registrador)", parts[3]))?
            } else {
                0xFF
            };
            if parts.len() > 4 {
                reject_unknown("SANITY_CHECK", &parts[4..], &[])?;
            }
            Ok(instr_sanity_check(parse_reg(parts[1])?, parse_reg(parts[2])?, rcount))
        }
        "PREEMPT_CHECK" => {
            if parts.len() < 2 {
                return Err(anyhow!("PREEMPT_CHECK precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("PREEMPT_CHECK", &parts[2..], &[])?;
            }
            Ok(instr_preempt_check(parse_reg(parts[1])?))
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
                    if let Ok(n) = v.parse::<u16>() { code = n; }
                    else { return Err(anyhow!("ASSERT CODE inválido '{}'", p)); }
                } else {
                    return Err(anyhow!("ASSERT token desconhecido '{}' (use CODE=)", p));
                }
            }
            Ok(instr_assert(parse_reg(parts[1])?, code))
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
            Ok(instr_set_deadline(parse_reg(parts[1])?))
        }
        "GET_DEADLINE" => {
            if parts.len() < 2 {
                return Err(anyhow!("GET_DEADLINE precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("GET_DEADLINE", &parts[2..], &[])?;
            }
            Ok(instr_get_deadline(parse_reg(parts[1])?))
        }
        "PRIORITY_SET" => {
            if parts.len() < 2 {
                return Err(anyhow!("PRIORITY_SET precisa de Rs (0/1/2)"));
            }
            if parts.len() > 2 {
                reject_unknown("PRIORITY_SET", &parts[2..], &[])?;
            }
            Ok(instr_priority_set(parse_reg(parts[1])?))
        }
        "PRIORITY_GET" => {
            if parts.len() < 2 {
                return Err(anyhow!("PRIORITY_GET precisa de rdest"));
            }
            if parts.len() > 2 {
                reject_unknown("PRIORITY_GET", &parts[2..], &[])?;
            }
            Ok(instr_priority_get(parse_reg(parts[1])?))
        }
        "LOCK" => {
            if parts.len() < 2 {
                return Err(anyhow!("LOCK precisa de Rs (id)"));
            }
            if parts.len() > 2 {
                reject_unknown("LOCK", &parts[2..], &[])?;
            }
            Ok(instr_lock(parse_reg(parts[1])?))
        }
        "UNLOCK" => {
            if parts.len() < 2 {
                return Err(anyhow!("UNLOCK precisa de Rs (id)"));
            }
            if parts.len() > 2 {
                reject_unknown("UNLOCK", &parts[2..], &[])?;
            }
            Ok(instr_unlock(parse_reg(parts[1])?))
        }
        "FENCE" => {
            if parts.len() > 1 {
                reject_unknown("FENCE", &parts[1..], &[])?;
            }
            Ok(instr_fence())
        }
        "LOADI" => {
            // LOADI rD, imm — decimal ou 0x-hex; negativos rejeitados (u128).
            if parts.len() < 3 {
                return Err(anyhow!("LOADI precisa de rdest, imediato — ex: LOADI r0, 80000000"));
            }
            let rdest = parse_reg(parts[1])?;
            let s = parts[2].trim();
            let imm = if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
                u128::from_str_radix(hex, 16)
                    .map_err(|_| anyhow!("LOADI imediato hex inválido '{}'", parts[2]))?
            } else if s.starts_with('-') || s.starts_with('+') {
                return Err(anyhow!("LOADI imediato '{}' inválido (u128: decimal ou 0x-hex, sem sinal)", parts[2]));
            } else {
                s.parse::<u128>()
                    .map_err(|_| anyhow!("LOADI imediato '{}' inválido (u128: decimal ou 0x-hex)", parts[2]))?
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
            Ok(instr_mov(parse_reg(parts[1])?, parse_reg(parts[2])?))
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
            Ok(instr_kv_truncate(parse_reg(parts[1])?, stream))
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
                if let Ok(r) = parse_reg(parts[rpos]) {
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
                parse_reg(parts[1])?,
                parse_reg(parts[2])?,
                parse_reg(parts[3])?,
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
                if let Ok(r) = parse_reg(parts[rpos]) {
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
                parse_reg(parts[1])?,
                parse_reg(parts[2])?,
                parse_reg(parts[3])?,
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
                if let Ok(r) = parse_reg(parts[rpos]) {
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
                parse_reg(parts[1])?,
                parse_reg(parts[2])?,
                parse_reg(parts[3])?,
                rpack,
                thresh,
                decay,
                reset,
                layer,
                refr,
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
                    if let Ok(n) = v.parse::<u16>() { n_trees = n; }
                    else { return Err(anyhow!("FOREST TREES inválido '{}'", p)); }
                } else if let Some(v) = up.strip_prefix("DEPTH=") {
                    if let Ok(n) = v.parse::<u16>() { depth = n; }
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
                parse_reg(parts[1])?,
                parse_reg(parts[2])?,
                parse_reg(parts[3])?,
                parse_reg(parts[4])?,
                n_trees,
                depth,
                mode,
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
        _ => Err(anyhow!("dtype desconhecido '{}' (use f32/f16/i8/u8)", s)),
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
}
