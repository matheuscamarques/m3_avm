//! benches/matvec_bench.rs — Fase 3.1: ndarray (baseline) vs faer (SIMD)
//!
//! Rode: `cargo bench --bench matvec_bench -- --quick`
//!
//! Tamanhos espelham o DeepSeek-R1-Distill-Qwen-1.5B (hidden 1536):
//!   - q_proj: 1536 -> 1536
//!   - gate/up: 1536 -> 896 (intermediate aprox; real 8960 em 7B, 1536 aqui reduzido)
//!   - pequeno: 512 -> 256 (rápido, sanity)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use m3_avm::matvec::{matvec_faer, matvec_ndarray};
use std::time::Duration;

fn sample(in_dim: usize, out_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32 * 0.013).sin() * 0.5)).collect();
    let w: Vec<f32> = (0..in_dim * out_dim)
        .map(|i| ((i as f32 * 0.002).sin() * 0.3))
        .collect();
    (x, w)
}

fn bench_matvec(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_fase3_1");
    group.measurement_time(Duration::from_secs(4));
    group.warm_up_time(Duration::from_secs(1));
    group.sample_size(10);

    for (name, in_dim, out_dim) in [("small_512x256", 512, 256), ("qproj_1536x1536", 1536, 1536)] {
        let (x, w) = sample(in_dim, out_dim);
        group.throughput(Throughput::Elements((in_dim * out_dim) as u64));
        group.bench_with_input(BenchmarkId::new("ndarray", name), &(in_dim, out_dim), |b, _| {
            b.iter(|| criterion::black_box(matvec_ndarray(&x, &w, in_dim, out_dim)));
        });
        group.bench_with_input(BenchmarkId::new("faer", name), &(in_dim, out_dim), |b, _| {
            b.iter(|| criterion::black_box(matvec_faer(&x, &w, in_dim, out_dim)));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matvec);
criterion_main!(benches);
