//! bench_deepseek — Mede 2 tokens DeepSeek-Q4_K_M em release.
//!
//! Uso: `cargo run --release --bin bench_deepseek`
//! Imprime tempo do 1º/2º token, tamanho do cache f32 e RSS (prova que o
//! kernel int4 bypassa os ~7GB de pesos dequantizados).

use m3_avm::inference::RealInference;
use m3_avm::memory::MemoryManager;
use m3_avm::vm::MemBackend;

fn rss_mb() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| {
                l.split_whitespace().nth(1)?.parse::<usize>().ok()
            })
        })
        .map(|kb| kb / 1024)
        .unwrap_or(0)
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf".to_string());
    let mut inf = RealInference::new(&path)?;
    let mut mem_mgr = MemoryManager::new_in_memory();
    mem_mgr.load_gguf_model(&path)?;
    let mem = MemBackend::Cpu(mem_mgr);
    eprintln!("[bench] rss após mmap: {} MiB", rss_mb());

    let t0 = std::time::Instant::now();
    let logits1 = inf.forward_one(&mem, 1)?;
    let d0 = t0.elapsed();
    println!(
        "first token {:?} (logits {}, cache_f32 {}, rss {} MiB)",
        d0,
        logits1.len(),
        inf.weight_cache_len(),
        rss_mb()
    );

    let t1 = std::time::Instant::now();
    let logits2 = inf.forward_one(&mem, 2)?;
    let d1 = t1.elapsed();
    println!(
        "second token {:?} (logits {}, cache_f32 {}, rss {} MiB)",
        d1,
        logits2.len(),
        inf.weight_cache_len(),
        rss_mb()
    );
    println!("sample1={} sample2={}", inf.sample(&logits1), inf.sample(&logits2));
    Ok(())
}
