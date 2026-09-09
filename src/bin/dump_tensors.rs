//! dump_tensors — imprime nome, dims, dtype e n_elements dos tensores-chave.
//! Uso: `cargo run --release --bin dump_tensors -- <modelo.gguf> [filtro]`

use m3_avm::gguf::GgufFile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf".to_string());
    let filter = std::env::args().nth(2).unwrap_or_default();
    let gg = GgufFile::open(&path)?;
    println!("tensors={} data_offset={:#x}", gg.n_tensors, gg.data_offset);
    for t in &gg.tensors {
        if t.name.contains("blk.0.") || t.name.contains("embd") || t.name.contains("output") || t.name.contains("norm") {
            if filter.is_empty() || t.name.contains(&filter as &str) {
                println!("{:45} dims={:?} dtype={:3} n={:10} off={}", t.name, t.dims, t.dtype, t.n_elements, t.offset);
            }
        }
    }
    Ok(())
}
