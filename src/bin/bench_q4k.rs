//! bench_q4k — A/B escalar vs AVX2 no mesmo processo (cancela drift de load).
//!
//! Uso: `cargo run --release --bin bench_q4k`
//! Alterna escalar/AVX2 5× sobre q_proj blk0 do DeepSeek (1536×1536 Q4_K)
//! e imprime medianas.

use m3_avm::inference::RealInference;
use m3_avm::matvec_quant::{matvec_q4k_impl, quant_raw_len};
use m3_avm::memory::{make_persistent_addr, MemoryManager};
use m3_avm::vm::MemBackend;

fn median(mut v: Vec<u128>) -> u128 {
    v.sort_unstable();
    v[v.len() / 2]
}

fn main() -> anyhow::Result<()> {
    let path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
    let inf = RealInference::new(path)?;
    let mut mem_mgr = MemoryManager::new_in_memory();
    mem_mgr.load_gguf_model(path)?;
    let mem = MemBackend::Cpu(mem_mgr);
    let h = inf.config.hidden;
    let names = inf.config.try_get_tensor_names(0, "q");
    let name = names.iter().find(|n| inf.gguf.find_tensor(n).is_some()).unwrap();
    let info = inf.gguf.find_tensor(name).unwrap();
    let n = info.n_elements;
    let raw_len = quant_raw_len(info.dtype, n).unwrap();
    let raw = mem.read(make_persistent_addr((inf.gguf.data_offset + info.offset) as u128), raw_len)?;
    let x: Vec<f32> = (0..h).map(|i| ((i as f32 * 0.017).sin() * 0.6)).collect();

    // warmup
    let _ = matvec_q4k_impl(&x, &raw, h, h, false);
    let _ = matvec_q4k_impl(&x, &raw, h, h, true);

    let mut t_scalar = Vec::new();
    let mut t_avx2 = Vec::new();
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let a = matvec_q4k_impl(&x, &raw, h, h, false);
        t_scalar.push(t0.elapsed().as_micros());
        std::hint::black_box(a);
        let t1 = std::time::Instant::now();
        let b = matvec_q4k_impl(&x, &raw, h, h, true);
        t_avx2.push(t1.elapsed().as_micros());
        std::hint::black_box(b);
    }
    println!("scalar_us {:?} median {}", t_scalar, median(t_scalar.clone()));
    println!("avx2_us   {:?} median {}", t_avx2, median(t_avx2.clone()));
    println!(
        "speedup avx2/scalar = {:.2}x",
        median(t_scalar) as f64 / median(t_avx2).max(1) as f64
    );

    // A/B caminho de leitura: mem.read (cópia) vs read_model_raw (zero-copy),
    // ambos alimentando o kernel AVX2. Usa gate blk0 (13.8M elems, 7.7MB raw).
    let gnames = inf.config.try_get_tensor_names(0, "gate");
    let gname = gnames.iter().find(|n| inf.gguf.find_tensor(n).is_some()).unwrap();
    let ginfo = inf.gguf.find_tensor(gname).unwrap();
    let gn = ginfo.n_elements;
    let inter = inf.config.intermediate;
    let graw_len = quant_raw_len(ginfo.dtype, gn).unwrap();
    let goff = inf.gguf.data_offset + ginfo.offset;
    let gx: Vec<f32> = (0..h).map(|i| ((i as f32 * 0.031).sin() * 0.4)).collect();
    let mut t_copy = Vec::new();
    let mut t_zc = Vec::new();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let rc = mem.read(make_persistent_addr(goff as u128), graw_len)?;
        let yc = m3_avm::matvec_quant::matvec_quant(&gx, &rc, ginfo.dtype, h, inter).unwrap();
        t_copy.push(t0.elapsed().as_micros());
        std::hint::black_box(yc);
        let t1 = std::time::Instant::now();
        let rz = mem.read_model_raw(goff, graw_len).unwrap();
        let yz = m3_avm::matvec_quant::matvec_quant(&gx, rz, ginfo.dtype, h, inter).unwrap();
        t_zc.push(t1.elapsed().as_micros());
        std::hint::black_box(yz);
    }
    println!("read+kernel copy_us {:?} median {}", t_copy, median(t_copy.clone()));
    println!("read+kernel zc_us   {:?} median {}", t_zc, median(t_zc.clone()));
    println!("speedup zc/copy = {:.2}x", median(t_copy) as f64 / median(t_zc).max(1) as f64);
    Ok(())
}
