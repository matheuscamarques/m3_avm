//! probe_head — inspeciona linhas do output.weight (Q6_K) e o token top1.
//! Uso: `cargo run --release --bin probe_head`

use m3_avm::gguf::GgufFile;
use m3_avm::inference::RealInference;
use m3_avm::memory::MemoryManager;
use m3_avm::quant::dequantize;
use m3_avm::tokenizer::M3Tokenizer;
use m3_avm::vm::MemBackend;

fn main() -> anyhow::Result<()> {
    let path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
    let gg = GgufFile::open(path)?;
    let tok = M3Tokenizer::from_gguf(path)?;
    println!("vocab[22573] = {:?}", tok.decode(22573));
    println!("vocab[4992]  = {:?}", tok.decode(4992));
    for (i, v) in tok.vocab.iter().enumerate() {
        if v.contains("im_start") || v == ">a" || v == ">Ċ" || v == "|i" {
            println!("vocab[{}] = {:?}", i, v);
        }
    }
    println!("vocab[151643] = {:?} vocab[151645] = {:?}", tok.decode(151643), tok.decode(151645));
    let mut mem_mgr = MemoryManager::new_in_memory();
    mem_mgr.load_gguf_model(path)?;
    let mem = MemBackend::Cpu(mem_mgr);
    let info = gg.find_tensor("output.weight").unwrap();
    println!("output.weight dtype={} n={}", info.dtype, info.n_elements);
    let h = 1536usize;
    // decodifica 3 linhas via dequantize de blocos alinhados (1536 = 6 blocos Q6_K)
    for row in [0usize, 22573, 4242] {
        let base = row * h;
        let raw_len = (h / 256) * 210;
        let file_off = gg.data_offset + info.offset + (base / 256) as u64 * 210;
        let paddr = m3_avm::memory::make_persistent_addr(file_off as u128);
        let raw = mem.read(paddr, raw_len)?;
        let mut dst = vec![0.0f32; h];
        assert!(dequantize(&raw, info.dtype, &mut dst, h));
        let mean: f32 = dst.iter().sum::<f32>() / h as f32;
        let amax = dst.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let std = (dst.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / h as f32).sqrt();
        println!("row {}: mean={:+.4} std={:.4} max|x|={:.2}", row, mean, std, amax);
    }
    // Nosso Q4_K nos mesmos 2 blocos do gate (comparar com ggml via xcheck_ggml.py:
    // ggml mean=-0.00038 std=0.03668 max=+0.107 min=-0.108)
    let ginfo = gg.find_tensor("blk.0.ffn_gate.weight").unwrap();
    let grawoff = gg.data_offset + ginfo.offset;
    let graw = mem.read(m3_avm::memory::make_persistent_addr(grawoff as u128), 144 * 2)?;
    let mut gdst = vec![0.0f32; 512];
    assert!(dequantize(&graw, ginfo.dtype, &mut gdst, 512));
    let gmean: f32 = gdst.iter().sum::<f32>() / 512.0;
    let gstd = (gdst.iter().map(|v| (v - gmean) * (v - gmean)).sum::<f32>() / 512.0).sqrt();
    println!("gate blk0-1 NOSSO: mean={:+.5} std={:.5} max={:+.3} min={:+.3}",
        gmean, gstd,
        gdst.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        gdst.iter().cloned().fold(f32::INFINITY, f32::min));
    // Top-12 dos logits após prefill curto: pico sistemático ou semântica?
    let prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n";
    let toks = inf_tokenize(path, prompt);
    println!("--- toks ({}): {:?} ---", toks.len(),
        toks.iter().map(|&t| tok.decode(t)).collect::<Vec<_>>());    let toks = inf_tokenize(path, prompt);
    let (mut inf2, mem2) = inf_new(path)?;
    let mut last = Vec::new();
    for &t in &toks {
        last = inf2.forward_one(&mem2, t)?;
    }
    let mut idx: Vec<usize> = (0..last.len()).collect();
    idx.sort_by(|&a, &b| last[b].partial_cmp(&last[a]).unwrap());
    println!("--- top12 após {:?} ({} toks) ---", prompt, toks.len());
    for &i in idx.iter().take(12) {
        println!("{:8} {:+.2} {:?}", i, last[i], tok.decode(i as u32));
    }
    Ok(())
}

fn inf_tokenize(path: &str, prompt: &str) -> Vec<u32> {
    let inf = RealInference::new(path).unwrap();
    inf.tokenize(prompt)
}

fn inf_new(path: &str) -> anyhow::Result<(RealInference, MemBackend)> {
    use m3_avm::inference::RealInference;
    let mut inf = RealInference::new(path)?;
    let mut mem_mgr = MemoryManager::new_in_memory();
    mem_mgr.load_gguf_model(path)?;
    Ok((inf, MemBackend::Cpu(mem_mgr)))
}
