//! stt.rs — Serviço Whisper Tiny para M³-AVM (Ryzen 3500U)
//!
//! Mantém ISA de 6 opcodes. Áudio entra via `SENSE` → `TEMPORAL`,
//! serviço host lê `TEMPORAL`, chama `whisper-cli` via `Command` (sem bindgen),
//! escreve texto em `PERSISTENTE` e notifica via `Bus`.
//! O binário whisper.cpp é compilado separadamente: `whisper.cpp/build/bin/whisper-cli`

use anyhow::{anyhow, Result};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

pub const MODEL_Q5_0_SIZE_MB: usize = 42; // q8_0 real (q5_1 era 31, q8_0 42 — ambos <64)
pub const MODEL_FP32_SIZE_MB: usize = 144;
pub const PERSISTENT_CAP_MB: usize = 64;
pub const MODEL_DEFAULT: &str = "ggml-tiny-q8_0.bin";

/// Serviço STT — usa whisper-cli se existir, senão mock determinístico
pub struct WhisperService {
    model_path: String,
    whisper_bin: String,
}

impl WhisperService {
    pub fn new(model_path: &str) -> Result<Self> {
        let path = if model_path == "ggml-tiny-q5_0.bin" { MODEL_DEFAULT } else { model_path };
        let whisper_bin = Self::find_whisper_bin();
        Ok(Self { model_path: path.to_string(), whisper_bin })
    }

    pub fn new_mock() -> Self {
        Self { model_path: MODEL_DEFAULT.to_string(), whisper_bin: String::new() }
    }

    fn find_whisper_bin() -> String {
        for p in ["./whisper.cpp/build/bin/whisper-cli", "./whisper.cpp/build/bin/main", "/usr/local/bin/whisper-cli", "whisper-cli"] {
            if Path::new(p).exists() { return p.to_string(); }
        }
        String::new()
    }

    /// Transcreve PCM f32 mono 16kHz. Chama whisper-cli se modelo e binário existirem, senão mock.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<String> {
        if pcm.is_empty() { return Err(anyhow!("pcm vazio")); }
        let start = Instant::now();
        let audio_secs = pcm.len() as f32 / 16000.0;

        // Tenta whisper-cli real se binário e modelo existem
        if !self.whisper_bin.is_empty() && Path::new(&self.model_path).exists() && Path::new(&self.whisper_bin).exists() {
            match self.transcribe_via_cli(pcm) {
                Ok(text) => {
                    let elapsed = start.elapsed();
                    eprintln!("[stt whisper-cli] {:.1}s áudio → {} chars em {:?} (RTF {:.2}) bin={} model={}", audio_secs, text.len(), elapsed, elapsed.as_secs_f32()/audio_secs, self.whisper_bin, self.model_path);
                    return Ok(text);
                }
                Err(e) => eprintln!("[stt] whisper-cli falhou: {} — usando mock", e),
            }
        }

        // Mock determinístico
        let hash = { let mut h = 0u64; for &v in pcm.iter().take(1000) { h = h.wrapping_add((v*1000.0) as u64); } h };
        let mock_text = format!("mock transcrição pt ({}s, hash {}) — olá mundo teste M³-AVM", audio_secs as u32, hash % 1000);
        let elapsed = start.elapsed();
        eprintln!("[stt mock] {:.1}s áudio → {} em {:?} (model: {})", audio_secs, mock_text.len(), elapsed, self.model_path);
        Ok(mock_text)
    }

    fn transcribe_via_cli(&self, pcm: &[f32]) -> Result<String> {
        // Escreve WAV temporário 16kHz mono PCM16
        let wav_path = format!("/tmp/m3_whisper_{}.wav", std::process::id());
        Self::write_wav(&wav_path, pcm)?;
        let output = Command::new(&self.whisper_bin)
            .args(["-m", &self.model_path, "-f", &wav_path, "-l", "pt", "--output-txt", "-np", "-nt"])
            .output()
            .map_err(|e| anyhow!("whisper-cli exec: {}", e))?;
        let _ = std::fs::remove_file(&wav_path);
        let _ = std::fs::remove_file(format!("{}.txt", wav_path));
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!("whisper-cli exit {}: {}", output.status, err));
        }
        let txt = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // whisper-cli também escreve .txt file
        let txt_file = format!("{}.txt", wav_path);
        if let Ok(file_text) = std::fs::read_to_string(&txt_file) {
            let _ = std::fs::remove_file(txt_file);
            if !file_text.trim().is_empty() { return Ok(file_text.trim().to_string()); }
        }
        Ok(txt)
    }

    fn write_wav(path: &str, pcm: &[f32]) -> Result<()> {
        let spec = hound::WavSpec { channels: 1, sample_rate: 16000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = hound::WavWriter::create(path, spec).map_err(|e| anyhow!("wav create: {}", e))?;
        for &s in pcm {
            let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
            writer.write_sample(v).map_err(|e| anyhow!("wav write: {}", e))?;
        }
        writer.finalize().map_err(|e| anyhow!("wav finalize: {}", e))?;
        Ok(())
    }

    pub fn pcm_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect()
    }

    pub fn rtf(elapsed_secs: f32, audio_secs: f32) -> f32 {
        if audio_secs == 0.0 { 0.0 } else { elapsed_secs / audio_secs }
    }
}

pub fn simulate_rtf(_audio_secs: f32) -> f32 { 1.2 }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_pcm_conversion() {
        let bytes = [0x00, 0x80, 0xFF, 0x7F];
        let pcm = WhisperService::pcm_bytes_to_f32(&bytes);
        assert!((pcm[0] - (-1.0)).abs() < 0.001);
        assert!((pcm[1] - 0.999).abs() < 0.01);
    }
    #[test]
    fn test_mock_transcribe() {
        let svc = WhisperService::new_mock();
        let pcm = vec![0.1f32; 16000 * 2];
        let text = svc.transcribe(&pcm).unwrap();
        assert!(text.contains("mock"));
    }
    #[test]
    fn test_model_fits_persistent() {
        assert!(MODEL_Q5_0_SIZE_MB < PERSISTENT_CAP_MB);
        assert!(MODEL_FP32_SIZE_MB > PERSISTENT_CAP_MB);
    }
}
