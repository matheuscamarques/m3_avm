//! probe_rank — completação crua (sem template): prefill + top12 dos logits.
//! Uso: `cargo run --release --bin probe_rank -- <modelo.gguf> <texto...>`
//! Se o pipeline está correto, "The capital of France is" ranqueia Paris no topo.

use m3_avm::inference::RealInference;
use m3_avm::memory::MemoryManager;
use m3_avm::vm::MemBackend;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).map(|s| s.as_str()).unwrap_or("./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf");
    let prompt = if args.len() > 2 { args[2..].join(" ") } else { "The capital of France is".to_string() };
    let mut inf = RealInference::new(path)?;
    let mut mem_mgr = MemoryManager::new_in_memory();
    mem_mgr.load_gguf_model(path)?;
    let mem = MemBackend::Cpu(mem_mgr);
    let toks = inf.tokenize(&prompt);
    println!("prompt {:?} -> {} toks: {:?}", prompt, toks.len(), toks);
    let mut last = Vec::new();
    for &t in &toks {
        last = inf.forward_one(&mem, t)?;
    }
    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("--- top12 ---");
    for &i in idx.iter().take(12) {
        println!("{:8} {:+.2} {:?}", i, last[i], inf.tokenizer.decode(i as u32));
    }
    Ok(())
}
