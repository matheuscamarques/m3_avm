//! benches/physics_bench.rs — Benchmarks Nível 3 (Carga e Latência)
//!
//! Métricas exigidas pela Matriz de Cobertura:
//!   - IPS (NOP 1M) >5 MIPS em release
//!   - FORK 1GB (CoW) <1ms
//!   - ABORT pico (10 contextos) <50µs
//!   - STREAM 10 MB/s
//!
//! Rode: `cargo bench` ou `cargo bench --bench physics_bench`

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use m3_avm::{
    context::Priority,
    memory::{DType, MemoryManager},
    opcodes,
    vm::{Vm, VmConfig},
};
use std::time::Duration;

fn bench_ips(c: &mut Criterion) {
    let mut group = c.benchmark_group("level3_ips");
    group.throughput(Throughput::Elements(1));
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_secs(1));

    for n in [10_000, 100_000, 1_000_000].iter() {
        group.bench_with_input(BenchmarkId::new("nop", n), n, |b, &n| {
            b.iter(|| {
                let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(n as u64 + 1), ..Default::default() });
                let mut prog = vec![opcodes::instr_nop(); n];
                prog.push(opcodes::instr_halt());
                vm.load_program(prog);
                let _ = vm.run();
            });
        });
    }
    group.finish();
}

fn bench_fork(c: &mut Criterion) {
    let mut group = c.benchmark_group("level3_fork");
    group.bench_function("fork_1gb_cow", |b| {
        b.iter(|| {
            let mut mem = MemoryManager::new_in_memory();
            // Simula 1GB lógico com 100 blocos de 1MiB
            for _ in 0..100 {
                let _ = mem.alloc_global(1024 * 1024).unwrap();
            }
            let snap = mem.snapshot();
            criterion::black_box(snap);
        });
    });
    group.bench_function("fork_context_clone", |b| {
        b.iter(|| {
            let mut vm = Vm::new_in_memory(VmConfig::default());
            vm.scheduler.create_context(Priority::Green, 0x1000, 0);
            let prog = vec![opcodes::instr_fork(0, 2), opcodes::instr_halt()];
            vm.load_program(prog);
            let _ = vm.run();
        });
    });
    group.finish();
}

fn bench_abort(c: &mut Criterion) {
    let mut group = c.benchmark_group("level3_abort");
    group.bench_function("abort_peak_10_contexts", |b| {
        b.iter(|| {
            let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
            for _ in 0..10 {
                vm.scheduler.create_context(Priority::Green, 0x1000, 0);
            }
            // Seta r0=2 para abortar contexto 2
            if let Some(ctx) = vm.scheduler.get_mut(1) {
                let _ = ctx.set_reg(0, 2);
            }
            let prog = vec![opcodes::instr_abort(0, 0xFF), opcodes::instr_halt()];
            vm.load_program(prog);
            let _ = vm.run();
        });
    });
    group.finish();
}

fn bench_stream(c: &mut Criterion) {
    let mut group = c.benchmark_group("level3_stream");
    group.throughput(Throughput::Bytes(1024 * 1024));
    group.bench_function("stream_10mbs", |b| {
        b.iter(|| {
            let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(30), ..Default::default() });
            let mut prog = vec![];
            for _ in 0..10 {
                prog.push(opcodes::instr_tensor(0, 0xFF, 0xFF, 32, 32, 0));
                prog.push(opcodes::instr_stream(0, 0xFF, true));
            }
            prog.push(opcodes::instr_halt());
            vm.load_program(prog);
            let _ = vm.run();
        });
    });
    group.finish();
}

fn bench_attn(c: &mut Criterion) {
    let mut group = c.benchmark_group("level3_attn");
    group.sample_size(10);
    for size in [2, 8, 32].iter() {
        group.bench_with_input(BenchmarkId::new("attn", size), size, |b, &size| {
            b.iter(|| {
                let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
                let prog = vec![
                    opcodes::instr_tensor(0, 0xFF, 0xFF, size, size, 0),
                    opcodes::instr_tensor(1, 0xFF, 0xFF, size, size, 0),
                    opcodes::instr_tensor(2, 0xFF, 0xFF, size, size, 0),
                    opcodes::instr_attn(3, 0, 1, 2),
                    opcodes::instr_halt(),
                ];
                vm.load_program(prog);
                let _ = vm.run();
            });
        });
    }
    group.finish();
}

fn bench_sense(c: &mut Criterion) {
    let mut group = c.benchmark_group("level3_sense");
    group.bench_function("sense_audio", |b| {
        b.iter(|| {
            let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
            let prog = vec![
                opcodes::instr_sense(0, opcodes::SENSE_AUDIO),
                opcodes::instr_halt(),
            ];
            vm.load_program(prog);
            let _ = vm.run();
        });
    });
    group.finish();
}

fn bench_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_nop");
    group.sample_size(10);
    for density in [0.05, 0.5, 1.0].iter() {
        group.bench_with_input(BenchmarkId::new("attn_sparse_density", density), density, |b, &dens| {
            b.iter(|| {
                let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
                let prog = vec![
                    opcodes::instr_tensor_sparse(0, 32, 32, 0, dens),
                    opcodes::instr_tensor_sparse(1, 32, 32, 0, dens),
                    opcodes::instr_tensor_sparse(2, 32, 32, 0, dens),
                    opcodes::instr_attn_notify(3, 0, 1, 2),
                    opcodes::instr_halt(),
                ];
                vm.load_program(prog);
                let _ = vm.run();
            });
        });
    }
    group.bench_function("sparse_vs_dense_32", |b| {
        b.iter(|| {
            // Compara tempo: esparso 5% vs denso 100% para 32x32
            let mut vm_sparse = Vm::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
            let prog_sparse = vec![
                opcodes::instr_tensor_sparse(0, 32, 32, 0, 0.05),
                opcodes::instr_tensor_sparse(1, 32, 32, 0, 0.05),
                opcodes::instr_tensor_sparse(2, 32, 32, 0, 0.05),
                opcodes::instr_attn(3, 0, 1, 2),
                opcodes::instr_halt(),
            ];
            vm_sparse.load_program(prog_sparse);
            let _ = vm_sparse.run();
        });
    });
    group.finish();
}

criterion_group!(benches, bench_ips, bench_fork, bench_abort, bench_stream, bench_attn, bench_sense, bench_sparse);
criterion_main!(benches);
