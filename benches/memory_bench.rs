//! benches/memory_bench.rs — Benches RFC-0023 (0x26/0x27/0x2A/0x2B).
//!
//! Por instrução via `step_instruction` (custo que programas pagam).
//! Rode: `cargo bench --bench memory_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::{instr_arena_alloc, instr_arena_reset, instr_memcpy, instr_memset, MEMCPY_DIR_HOST},
    vm::{Vm, VmConfig},
};
use std::time::Duration;

/// Vm pronta com um contexto GREEN em 0x1000.
fn ready_vm() -> (Vm, u64) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let root = vm.memory.current_version();
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, root);
    (vm, cid)
}

fn f32_tensor(vm: &mut Vm, n: usize) -> u128 {
    let a = vm.memory.alloc_tensor(&[n], DType::F32).unwrap();
    vm.memory.write_f32_tensor(a, &vec![1.0; n]).unwrap();
    a
}

fn bench_memcpy_4k(c: &mut Criterion) {
    let mut g = c.benchmark_group("memcpy");
    g.measurement_time(Duration::from_secs(2));
    // 1024 f32 = 4 KiB: janela representativa de compactação KV.
    let (mut vm, cid) = ready_vm();
    let a = f32_tensor(&mut vm, 1024);
    let b = f32_tensor(&mut vm, 1024);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, a).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(1, b).unwrap();
    let cp = instr_memcpy(1, 0, 0, 0, 0, MEMCPY_DIR_HOST);
    g.bench_function("memcpy_4KiB_whole", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &cp).unwrap();
        })
    });
    g.finish();
}

fn bench_memset_4k(c: &mut Criterion) {
    let mut g = c.benchmark_group("memset");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let a = f32_tensor(&mut vm, 1024);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, a).unwrap();
    let st = instr_memset(0, 0xAB, 0, 0);
    g.bench_function("memset_4KiB_whole", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &st).unwrap();
        })
    });
    g.finish();
}

fn bench_arena(c: &mut Criterion) {
    let mut g = c.benchmark_group("arena");
    g.measurement_time(Duration::from_secs(2));
    // alloc 64B + reset por iteração (sem crescimento ilimitado).
    let (mut vm, cid) = ready_vm();
    let al = instr_arena_alloc(0, 64, 16, 0);
    let rs = instr_arena_reset(0);
    g.bench_function("arena_alloc_reset_64B", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &al).unwrap();
            vm.step_instruction(cid, &rs).unwrap();
        })
    });
    g.finish();
}

criterion_group!(
    memory,
    bench_memcpy_4k,
    bench_memset_4k,
    bench_arena,
);
criterion_main!(memory);
