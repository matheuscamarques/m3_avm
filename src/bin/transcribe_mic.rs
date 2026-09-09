//! transcribe_mic — teste direto do Whisper com mic.wav (sem VM)
//! Uso: cargo run --bin transcribe_mic -- mic.wav ggml-tiny-q5_1.bin
//! Sem --features whisper usa mock.

use std::path::PathBuf;
use m3_avm::stt::WhisperService;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let audio_path = args.get(1).map(|s| s.as_str()).unwrap_or("mic.wav");
    let model_path = args.get(2).map(|s| s.as_str()).unwrap_or("ggml-tiny-q5_1.bin");

    let bytes = std::fs::read(audio_path).map_err(|e| anyhow::anyhow!("{}: {}", audio_path, e))?;
    // WAV tem header 44 bytes, PCM começa após
    let pcm_bytes = if bytes.len() > 44 && &bytes[0..4] == b"RIFF" { &bytes[44..] } else { &bytes[..] };
    let pcm = WhisperService::pcm_bytes_to_f32(pcm_bytes);
    println!("Áudio: {} bytes → {} samples ({:.1}s) de {}", bytes.len(), pcm.len(), pcm.len() as f32/16000.0, audio_path);
    println!("Modelo: {} (exists={})", model_path, PathBuf::from(model_path).exists());

    let svc = if PathBuf::from(model_path).exists() {
        match WhisperService::new(model_path) {
            Ok(s) => s,
            Err(e) => { eprintln!("Modelo falhou ({}), usando mock: {}", model_path, e); WhisperService::new_mock() }
        }
    } else {
        eprintln!("Modelo não encontrado, usando mock");
        WhisperService::new_mock()
    };

    let start = std::time::Instant::now();
    let text = svc.transcribe(&pcm)?;
    let elapsed = start.elapsed();
    let rtf = WhisperService::rtf(elapsed.as_secs_f32(), pcm.len() as f32/16000.0);
    println!("\nTexto: {}", text);
    println!("Tempo: {:?} RTF: {:.2}", elapsed, rtf);
    if rtf < 2.5 { println!("✅ RTF ok (<2.5)"); } else { println!("⚠️ RTF alto (>2.5)"); }
    Ok(())
}
