//! mimi.rs — Codec Mimi STUB determinístico (F2 do PLANO_MOSHI_NATIVO)
//!
//! Alvo real: Mimi `24kHz -> 12.5Hz (80ms = 1920 samples)`, `16 codebooks`
//! PersonaPlex (Moshi original usa 8). SEANet + transformer + RVQ de verdade
//! entram em F3; aqui o stub é determinístico e testável para ligar o
//! pipeline `SENSE AUDIO -> TEMPORAL -> codes -> PCM` sem `rand`.
//!
//! Convenções:
//! - PCM sempre `f32` mono. Mimi opera a 24kHz; STT/whisper usa 16kHz
//!   (`src/stt.rs`) — conversão linear em `resample_*`.
//! - 1 frame = 1920 samples = 32 bytes de codes (16x u16 LE) ou 7680 bytes PCM.
//! - Stub RVQ: codebook `c` resume o bloco `frame[c*120..(c+1)*120]`
//!   (1920/16 = 120) pela média quantizada em 10 bits (0..1023).
//!   Silêncio -> código central (~511) -> decode ~0. Documentado como STUB:
//!   sem paridade com `mimi` real.

/// Sample rate do Mimi.
pub const MIMI_SAMPLE_RATE: u32 = 24_000;
/// Frame rate do Mimi.
pub const MIMI_FRAME_HZ: f32 = 12.5;
/// Samples por frame: 24000/12.5 = 1920 (80ms).
pub const MIMI_SAMPLES_PER_FRAME: usize = 1920;
/// Codebooks PersonaPlex (Moshi original: 8).
pub const MIMI_N_CODEBOOKS: usize = 16;
/// Samples por codebook no stub: 1920/16 = 120.
pub const MIMI_SAMPLES_PER_CODEBOOK: usize = 120;
/// Tamanho do codebook stub (10 bits).
pub const MIMI_CODEBOOK_SIZE: u16 = 1024;
/// Código do silêncio: nível 0 -> (0*0.5+0.5)*1023 = 511.5 -> round = 512.
pub const MIMI_SILENCE_CODE: u16 = 512;

/// 1 frame codificado: 16 códigos RVQ.
pub type MimiCodes = [u16; MIMI_N_CODEBOOKS];

/// Quantiza um nível [-1,1] em código 10 bits.
#[inline]
fn level_to_code(level: f32) -> u16 {
    let v = ((level.clamp(-1.0, 1.0) * 0.5 + 0.5) * ((MIMI_CODEBOOK_SIZE - 1) as f32)).round() as i32;
    v.clamp(0, (MIMI_CODEBOOK_SIZE - 1) as i32) as u16
}

/// Dequantiza código 10 bits em nível [-1,1].
#[inline]
fn code_to_level(code: u16) -> f32 {
    (code.min(MIMI_CODEBOOK_SIZE - 1) as f32) / ((MIMI_CODEBOOK_SIZE - 1) as f32) * 2.0 - 1.0
}

/// Codifica 1 frame (1920 samples) em 16 códigos. Determinístico.
/// Frame curto é completado com zeros; longo é truncado.
///
/// STUB: cada código guarda o RMS do bloco (`frame[c*120..(c+1)*120]`)
/// mapeado em 10 bits via `level_to_code(rms)` (rms em [0,1] -> códigos em
/// [512,1023]; silêncio -> 512). RMS preserva energia de senoides (média
/// simples daria ~0 para 440Hz em janelas de 120 samples ≈ 2.2 períodos).
/// Sem paridade com o RVQ real.
pub fn encode_frame(frame: &[f32]) -> MimiCodes {
    let mut codes = [MIMI_SILENCE_CODE; MIMI_N_CODEBOOKS];
    for c in 0..MIMI_N_CODEBOOKS {
        let base = c * MIMI_SAMPLES_PER_CODEBOOK;
        let mut sum_sq = 0.0f32;
        for i in 0..MIMI_SAMPLES_PER_CODEBOOK {
            let v = frame.get(base + i).copied().unwrap_or(0.0);
            sum_sq += v * v;
        }
        let rms = (sum_sq / (MIMI_SAMPLES_PER_CODEBOOK as f32)).sqrt().clamp(0.0, 1.0);
        codes[c] = level_to_code(rms);
    }
    codes
}

/// Decodifica 16 códigos em 1 frame (1920 samples).
/// Cada bloco de 120 samples recebe o nível do código + dither senoidal
/// determinístico de baixa amplitude (evita degraus perfeitos, mantém
/// determinismo e teste de roundtrip simples).
pub fn decode_frame(codes: &MimiCodes) -> [f32; MIMI_SAMPLES_PER_FRAME] {
    let mut out = [0.0f32; MIMI_SAMPLES_PER_FRAME];
    for c in 0..MIMI_N_CODEBOOKS {
        let level = code_to_level(codes[c]) * 0.9;
        // Frequência do dither varia por codebook, fase fixa — determinístico.
        let freq = 200.0 + (c as f32) * 25.0;
        for i in 0..MIMI_SAMPLES_PER_CODEBOOK {
            let t = (c * MIMI_SAMPLES_PER_CODEBOOK + i) as f32 / (MIMI_SAMPLE_RATE as f32);
            let dither = (2.0 * std::f32::consts::PI * freq * t).sin() * 0.02;
            out[c * MIMI_SAMPLES_PER_CODEBOOK + i] = (level + dither).clamp(-1.0, 1.0);
        }
    }
    out
}

/// PCM 24kHz -> frames de 1920 (último com zero-pad).
pub fn pcm_to_frames(pcm: &[f32]) -> Vec<MimiCodes> {
    if pcm.is_empty() {
        return vec![];
    }
    pcm.chunks(MIMI_SAMPLES_PER_FRAME).map(encode_frame).collect()
}

/// Frames -> PCM 24kHz concatenado.
pub fn frames_to_pcm(frames: &[MimiCodes]) -> Vec<f32> {
    let mut out = Vec::with_capacity(frames.len() * MIMI_SAMPLES_PER_FRAME);
    for f in frames {
        out.extend_from_slice(&decode_frame(f));
    }
    out
}

/// Codes -> 32 bytes LE (16x u16). Para guardar em TEMPORAL.
pub fn codes_to_bytes(codes: &MimiCodes) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, c) in codes.iter().enumerate() {
        b[i * 2..i * 2 + 2].copy_from_slice(&c.to_le_bytes());
    }
    b
}

/// 32 bytes LE -> codes. Retorna `None` se `len < 32`.
pub fn codes_from_bytes(bytes: &[u8]) -> Option<MimiCodes> {
    if bytes.len() < 32 {
        return None;
    }
    let mut codes = [0u16; MIMI_N_CODEBOOKS];
    for i in 0..MIMI_N_CODEBOOKS {
        codes[i] = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) % MIMI_CODEBOOK_SIZE;
    }
    Some(codes)
}

/// PCM f32 -> bytes LE (para `TEMPORAL`, como `exec_sense` já faz).
pub fn pcm_to_bytes(pcm: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 4);
    for v in pcm {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Bytes LE -> PCM f32 (trunca resto incompleto).
pub fn pcm_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Frame sintético determinístico 440Hz @24kHz * 0.5 (para SENSE/testes).
/// Substitui o `rand` white-noise: mesmo bytes a cada chamada.
pub fn synth_frame_440hz() -> [f32; MIMI_SAMPLES_PER_FRAME] {
    let mut out = [0.0f32; MIMI_SAMPLES_PER_FRAME];
    for (i, v) in out.iter_mut().enumerate() {
        *v = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / MIMI_SAMPLE_RATE as f32).sin() * 0.5;
    }
    out
}

/// Frame de silêncio (zeros).
pub fn silence_frame() -> [f32; MIMI_SAMPLES_PER_FRAME] {
    [0.0f32; MIMI_SAMPLES_PER_FRAME]
}

/// Reamostra 16kHz -> 24kHz (fator 1.5, interpolação linear).
/// `out_len = ceil(in_len * 3 / 2)`.
pub fn resample_16k_to_24k(pcm16: &[f32]) -> Vec<f32> {
    if pcm16.is_empty() {
        return vec![];
    }
    if pcm16.len() == 1 {
        return vec![pcm16[0]; 2];
    }
    let out_len = (pcm16.len() * 3 + 1) / 2;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f32 * 2.0 / 3.0;
        let lo = pos.floor() as usize;
        let frac = pos - lo as f32;
        let hi = (lo + 1).min(pcm16.len() - 1);
        out.push(pcm16[lo] * (1.0 - frac) + pcm16[hi] * frac);
    }
    out
}

/// Reamostra 24kHz -> 16kHz (fator 2/3, interpolação linear).
pub fn resample_24k_to_16k(pcm24: &[f32]) -> Vec<f32> {
    if pcm24.is_empty() {
        return vec![];
    }
    if pcm24.len() == 1 {
        return vec![pcm24[0]];
    }
    let out_len = (pcm24.len() * 2 + 2) / 3;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f32 * 3.0 / 2.0;
        let lo = pos.floor() as usize;
        let frac = pos - lo as f32;
        let hi = (lo + 1).min(pcm24.len() - 1);
        out.push(pcm24[lo] * (1.0 - frac) + pcm24[hi] * frac);
    }
    out
}

/// Codec stateful (MVP sem estado; estrutura reservada para SEANet em F3).
#[derive(Debug, Clone, Default)]
pub struct MimiCodec;

impl MimiCodec {
    pub fn new() -> Self {
        Self
    }

    /// PCM 24kHz -> codes por frame.
    pub fn encode_pcm(&self, pcm24: &[f32]) -> Vec<MimiCodes> {
        pcm_to_frames(pcm24)
    }

    /// Codes -> PCM 24kHz.
    pub fn decode_codes(&self, frames: &[MimiCodes]) -> Vec<f32> {
        frames_to_pcm(frames)
    }

    /// PCM 16kHz (whisper/STT) -> codes (via upsample interno).
    pub fn encode_pcm16(&self, pcm16: &[f32]) -> Vec<MimiCodes> {
        self.encode_pcm(&resample_16k_to_24k(pcm16))
    }

    /// Codes -> PCM 16kHz (via downsample).
    pub fn decode_pcm16(&self, frames: &[MimiCodes]) -> Vec<f32> {
        resample_24k_to_16k(&self.decode_codes(frames))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constantes_batem_moshi() {
        assert_eq!(MIMI_SAMPLES_PER_FRAME, 1920);
        assert_eq!(MIMI_SAMPLE_RATE, 24_000);
        assert_eq!(MIMI_SAMPLES_PER_CODEBOOK, MIMI_SAMPLES_PER_FRAME / MIMI_N_CODEBOOKS);
        assert_eq!(crate::moshi::MOSHI_SAMPLE_RATE, MIMI_SAMPLE_RATE);
        assert!((crate::moshi::MOSHI_FRAME_HZ - MIMI_FRAME_HZ).abs() < 1e-6);
        assert_eq!(crate::moshi::MOSHI_N_CODEBOOKS, MIMI_N_CODEBOOKS);
        assert_eq!(crate::moshi::MoshiConfig::default().samples_per_frame(), MIMI_SAMPLES_PER_FRAME);
    }

    #[test]
    fn test_silencio_codigo_central_e_decode_quase_zero() {
        let f = silence_frame();
        let codes = encode_frame(&f);
        assert!(codes.iter().all(|&c| c == MIMI_SILENCE_CODE), "{:?}", &codes[..4]);
        let back = decode_frame(&codes);
        let max = back.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(max < 0.05, "max {}", max);
    }

    #[test]
    fn test_encode_deterministico_e_faixa() {
        let f = synth_frame_440hz();
        let a = encode_frame(&f);
        let b = encode_frame(&f);
        assert_eq!(a, b);
        assert!(a.iter().all(|&c| c < MIMI_CODEBOOK_SIZE));
        // senoide não é silêncio: algum código difere do central
        assert!(a.iter().any(|&c| c != MIMI_SILENCE_CODE));
    }

    #[test]
    fn test_roundtrip_frames_pcm_fiel_em_energia() {
        // Stub é lossy por bloco; checa energia preservada grosseiramente.
        let f = synth_frame_440hz();
        let codes = encode_frame(&f);
        let back = decode_frame(&codes);
        let e_in: f32 = f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32;
        let e_out: f32 = back.iter().map(|v| v * v).sum::<f32>() / back.len() as f32;
        assert!(e_in > 0.05, "e_in {}", e_in);
        assert!((e_out - e_in).abs() / e_in < 0.9, "e_in {} e_out {}", e_in, e_out);
    }

    #[test]
    fn test_pcm_frames_chunk_e_pad() {
        let pcm = vec![0.25f32; MIMI_SAMPLES_PER_FRAME * 2 + 100];
        let frames = pcm_to_frames(&pcm);
        assert_eq!(frames.len(), 3);
        let back = frames_to_pcm(&frames);
        assert_eq!(back.len(), 3 * MIMI_SAMPLES_PER_FRAME);
        assert!(pcm_to_frames(&[]).is_empty());
    }

    #[test]
    fn test_codes_bytes_roundtrip() {
        let codes = encode_frame(&synth_frame_440hz());
        let b = codes_to_bytes(&codes);
        assert_eq!(b.len(), 32);
        assert_eq!(codes_from_bytes(&b), Some(codes));
        assert_eq!(codes_from_bytes(&[0u8; 10]), None);
    }

    #[test]
    fn test_pcm_bytes_roundtrip() {
        let f = synth_frame_440hz();
        let b = pcm_to_bytes(&f);
        assert_eq!(b.len(), MIMI_SAMPLES_PER_FRAME * 4);
        let back = pcm_from_bytes(&b);
        assert_eq!(back.len(), f.len());
        assert!((back[100] - f[100]).abs() < 1e-6);
    }

    #[test]
    fn test_resample_comprimentos() {
        let p16 = vec![0.0f32, 1.0, 0.0, -1.0];
        let up = resample_16k_to_24k(&p16);
        assert_eq!(up.len(), (p16.len() * 3 + 1) / 2);
        let down = resample_24k_to_16k(&up);
        assert!(!down.is_empty());
        assert!(resample_16k_to_24k(&[]).is_empty());
        assert!(resample_24k_to_16k(&[]).is_empty());
        // constante preservada
        let c = vec![0.5f32; 16000];
        let upc = resample_16k_to_24k(&c);
        assert_eq!(upc.len(), 24000);
        assert!((upc[12000] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_codec_struct_16k_path() {
        let codec = MimiCodec::new();
        let p16 = vec![0.1f32; 16000];
        let frames = codec.encode_pcm16(&p16);
        assert!(!frames.is_empty());
        let back16 = codec.decode_pcm16(&frames);
        assert!(!back16.is_empty());
        for &v in &back16 {
            assert!(v.is_finite() && v.abs() <= 1.0);
        }
    }
}
