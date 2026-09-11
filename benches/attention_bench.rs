//! benches/attention_bench.rs — Benches RFC-0029 (0x3A/0x3B).
//!
//! flash em blocos vs top-k fundido, Q[4,32] K[16,32] V[16,8],
//! por instrução via `step_instruction`.
//! Rode: `cargo bench --bench attention_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::{instr_attn_sparse, instr_flash_attn, SPARSE_METRIC_DOT},
    vm::{Vm, VmConfig},
};
use std::time::Duration;

/// Vm pronta com Q/K/V rampa em r0/r1/r2.
fn ready_vm() -> (Vm, u64) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let root = vm.memory.current_version();
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, root);
    let mk = |vm: &mut Vm, shape: &[usize]| {
        let n: usize = shape.iter().product();
        let a = vm.memory.alloc_tensor(shape, DType::F32).unwrap();
        let ramp: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.25).collect();
        vm.memory.write_f32_tensor(a, &ramp).unwrap();
        a
    };
    let q = mk(&mut vm, &[4, 32]);
    let k = mk(&mut vm, &[16, 32]);
    let v = mk(&mut vm, &[16, 8]);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, q).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(1, k).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(2, v).unwrap();
    (vm, cid)
}

fn bench_attention(c: &mut Criterion) {
    let mut g = c.benchmark_group("attention");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let fl = instr_flash_attn(3, 0, 1, 2, 4);
    g.bench_function("flash_4x16_block4", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &fl).unwrap();
        })
    });
    let sp = instr_attn_sparse(4, 0, 1, 2, SPARSE_METRIC_DOT, 4);
    g.bench_function("sparse_top4_4x16", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &sp).unwrap();
        })
    });
    g.finish();
}

criterion_group!(attention, bench_attention,);
criterion_main!(attention);
