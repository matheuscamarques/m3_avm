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
pub const OP_HALT: u8 = 0x00; // não oficial, usado para encerrar programa
pub const OP_NOP: u8 = 0xFF;

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

    /// Decodifica 32 bytes brutos em Instruction.
    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < INSTR_SIZE {
            return Err(DecodeError::Incomplete(bytes.len()));
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

pub fn instr_ctx_switch(pipe_id: u8, priority_flag: u8) -> Instruction {
    Instruction::new(OP_CTX_SWITCH, priority_flag, 0xFF, pipe_id, 0xFF, 0xFF)
}

pub fn instr_rope(rdest: u8, r_src: u8, pos: u32, head_dim: usize, n_heads: usize, theta: f32) -> Instruction {
    let mut instr = Instruction::new(OP_ROPE, 0, rdest, r_src, 0xFF, 0xFF);
    instr.set_rope_params(pos, head_dim, n_heads, theta);
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
                }
            }
            if parts.len() > 5 {
                dtype = parse_dtype(parts[5]).unwrap_or(0);
            }
            let remaining = if parts.len() > 6 { &parts[6..] } else if parts.len() > 5 && parts[5].to_ascii_uppercase() == "SPARSE" { &parts[5..] } else { &[] as &[&str] };
            let (is_sparse, dens) = detect_sparse(remaining);
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
                parts[3].to_ascii_uppercase() != "DROP"
            } else {
                true
            };
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
                        _ => FORK_FLAG_GREEN,
                    }
                } else {
                    FORK_FLAG_GREEN
                };
                let notify = parts.iter().skip(3).any(|p| {
                    let up = p.to_ascii_uppercase();
                    up == "NOTIFY" || up == "NOTIFY_SCHEDULER"
                });
                let flags = if notify { prio | FORK_FLAG_NOTIFY } else { prio };
                let mut instr = instr_fork(rdest, flags);
                instr.set_imm_u128(target_pc);
                Ok(instr)
            } else {
                // Sintaxe clássica: FORK Rd, PRIORITY [, NOTIFY]
                let prio = match second.as_str() {
                    "RED" | "2" => FORK_FLAG_RED,
                    "BLUE" | "1" => FORK_FLAG_BLUE,
                    _ => FORK_FLAG_GREEN,
                };
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
                        SENSE_AUDIO
                    }
                }
            };
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
            Ok(instr_embed(rdest, rtoken, rtable))
        }
        "ADD" => {
            if parts.len() < 4 {
                return Err(anyhow!("ADD precisa de rdest, r_src1, r_src2 — ex: ADD r10, r10, r14"));
            }
            let rdest = parse_reg(parts[1])?;
            let rsrc1 = parse_reg(parts[2])?;
            let rsrc2 = parse_reg(parts[3])?;
            Ok(instr_add(rdest, rsrc1, rsrc2))
        }
        "SAMPLE" => {
            if parts.len() < 3 {
                return Err(anyhow!("SAMPLE precisa de rdest, r_logits — ex: SAMPLE r11, r15"));
            }
            let rdest = parse_reg(parts[1])?;
            let rlogits = parse_reg(parts[2])?;
            let mut temp = 1.0f32;
            for p in &parts[3..] {
                let up = p.to_ascii_uppercase();
                if up.starts_with("TEMP=") {
                    if let Ok(v) = up["TEMP=".len()..].parse::<f32>() { temp = v; }
                } else if let Ok(v) = p.parse::<f32>() {
                    temp = v;
                }
            }
            Ok(instr_sample(rdest, rlogits, temp))
        }
        "COMPARE" => {
            if parts.len() < 3 {
                return Err(anyhow!("COMPARE precisa de r_src1, r_src2/imediato — ex: COMPARE r11, EOS_TOKEN"));
            }
            let rsrc1 = parse_reg(parts[1])?;
            let second = parts[2].to_ascii_uppercase();
            if second == "EOS_TOKEN" {
                Ok(instr_compare(rsrc1, 0xFF, EOS_TOKEN_DEFAULT))
            } else if let Ok(r2) = parse_reg(parts[2]) {
                Ok(instr_compare(rsrc1, r2, 0))
            } else if let Ok(n) = parts[2].parse::<u128>() {
                Ok(instr_compare(rsrc1, 0xFF, n))
            } else {
                Err(anyhow!("COMPARE segundo operando inválido '{}'", parts[2]))
            }
        }
        "IF_EQUAL" => {
            if parts.len() < 2 {
                return Err(anyhow!("IF_EQUAL precisa de rótulo alvo — ex: IF_EQUAL PROGRAM_END"));
            }
            let label = parts[1].to_ascii_uppercase();
            let target_pc = *labels.get(&label)
                .ok_or_else(|| anyhow!("rótulo '{}' não encontrado para IF_EQUAL", label))?;
            Ok(instr_if_equal(target_pc))
        }
        "JUMP" | "JMP" => {
            if parts.len() < 2 {
                return Err(anyhow!("JUMP precisa de rótulo alvo — ex: JUMP MAIN_LOOP"));
            }
            let label = parts[1].to_ascii_uppercase();
            let target_pc = *labels.get(&label)
                .ok_or_else(|| anyhow!("rótulo '{}' não encontrado para JUMP", label))?;
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
            Ok(instr_matvec(rdest, rx, rw))
        }
        "MUL" => {
            if parts.len() < 4 {
                return Err(anyhow!("MUL precisa de rdest, r1, r2 — ex: MUL r6, r6, r7"));
            }
            let rdest = parse_reg(parts[1])?;
            let r1 = parse_reg(parts[2])?;
            let r2 = parse_reg(parts[3])?;
            Ok(instr_mul(rdest, r1, r2))
        }
        "SILU" => {
            if parts.len() < 3 {
                return Err(anyhow!("SILU precisa de rdest, rsrc — ex: SILU r6, r6"));
            }
            let rdest = parse_reg(parts[1])?;
            let rsrc = parse_reg(parts[2])?;
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
                else if let Some(v) = up.strip_prefix("D_INNER=") { if let Ok(n) = v.parse::<usize>() { di = n; } }
                else if let Some(v) = up.strip_prefix("D_STATE=") { if let Ok(n) = v.parse::<usize>() { ds = n; } }
                else if let Some(v) = up.strip_prefix("LAYER=") { if let Ok(n) = v.parse::<u8>() { layer = n; } }
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
                if let Some(v) = up.strip_prefix("D_INNER=") { if let Ok(n) = v.parse::<usize>() { di = n; } }
                else if let Some(v) = up.strip_prefix("D_STATE=") { if let Ok(n) = v.parse::<usize>() { ds = n; } }
                else if let Some(v) = up.strip_prefix("LAYER=") { if let Ok(n) = v.parse::<u8>() { layer = n; } }
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
                if let Some(v) = up.strip_prefix("SR=") { if let Ok(n) = v.parse::<u32>() { sr = n; } }
                else if let Some(v) = up.strip_prefix("FRAME=") { if let Ok(n) = v.parse::<u32>() { spf = n; } }
                else if let Some(v) = up.strip_prefix("HZ=") { if let Ok(n) = v.parse::<f32>() { hz = n; } }
                else if let Some(v) = up.strip_prefix("DELAY=") { if let Ok(n) = v.parse::<f32>() { dl = n; } }
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
                _ => parse_reg(parts[1]).unwrap_or(PIPE_TRANSFORMER),
            };
            let mut prio = 0u8;
            for p in &parts[2..] {
                match p.to_ascii_uppercase().as_str() {
                    "RED" | "2" => prio = 0b10,
                    "BLUE" | "1" => prio = 0b01,
                    "GREEN" | "0" => prio = 0b00,
                    _ => {}
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
                else if let Some(v) = up.strip_prefix("POS=") { if let Ok(n) = v.parse::<u32>() { pos = n; } }
                else if let Some(v) = up.strip_prefix("HDIM=") { if let Ok(n) = v.parse::<usize>() { hd = n; } }
                else if let Some(v) = up.strip_prefix("NHEADS=") { if let Ok(n) = v.parse::<usize>() { nh = n; } }
                else if let Some(v) = up.strip_prefix("THETA=") { if let Ok(n) = v.parse::<f32>() { theta = n; } }
            }
            let mut instr = instr_rope(rdest, rsrc, pos, hd, nh, theta);
            if inplace { instr.flags |= ROPE_FLAG_INPLACE; }
            Ok(instr)
        }
        "HALT" => Ok(instr_halt()),
        "NOP" => Ok(instr_nop()),
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
}
