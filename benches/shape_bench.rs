//! benches/shape_bench.rs — Benches RFC-0027 (0x30-0x37, parcial).
//!
//! sort/topk/reduce sobre lane 1K, por instrução via `step_instruction`.
//! Rode: `cargo bench --bench shape_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::{instr_argmax, instr_reduce, instr_sort, instr_topk, REDUCE_SUM, SORT_ASC},
    vm::{Vm, VmConfig},
};
use std::time::Duration;

/// Vm pronta com tensor [1,1024] rampa em r0.
fn ready_vm() -> (Vm, u64) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let root = vm.memory.current_version();
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, root);
    let a = vm.memory.alloc_tensor(&[1, 1024], DType::F32).unwrap();
    let ramp: Vec<f32> = (0..1024).map(|i| (i as f32 + 1.0) * 0.5).collect();
    vm.memory.write_f32_tensor(a, &ramp).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, a).unwrap();
    (vm, cid)
}

fn bench_shape(c: &mut Criterion) {
    let mut g = c.benchmark_group("shape");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let sort = instr_sort(1, 0, 1, SORT_ASC);
    g.bench_function("sort_1K_lane", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &sort).unwrap();
        })
    });
    let topk = instr_topk(2, 0, 1, 8, 1, 1);
    g.bench_function("topk8_1K_lane", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &topk).unwrap();
        })
    });
    let red = instr_reduce(3, 0, REDUCE_SUM, 0xFF);
    g.bench_function("reduce_sum_1K", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &red).unwrap();
        })
    });
    let amx = instr_argmax(4, 0, 1);
    g.bench_function("argmax_1K_lane", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &amx).unwrap();
        })
    });
    g.finish();
}

criterion_group!(shape, bench_shape,);
criterion_main!(shape);
