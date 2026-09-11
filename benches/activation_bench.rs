//! benches/activation_bench.rs — Benches RFC-0028 (0x3C-0x43, parcial).
//!
//! softmax/gelu sobre lane 1K, por instrução via `step_instruction`.
//! Rode: `cargo bench --bench activation_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::{instr_gelu, instr_softmax},
    vm::{Vm, VmConfig},
};
use std::time::Duration;

/// Vm pronta com tensor [1,1024] rampa em r0.
fn ready_vm() -> (Vm, u64) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let root = vm.memory.current_version();
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, root);
    let a = vm.memory.alloc_tensor(&[1, 1024], DType::F32).unwrap();
    let ramp: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) / 512.0).collect();
    vm.memory.write_f32_tensor(a, &ramp).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, a).unwrap();
    (vm, cid)
}

fn bench_activation(c: &mut Criterion) {
    let mut g = c.benchmark_group("activation");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let sm = instr_softmax(1, 0, 0xFF, 1.0);
    g.bench_function("softmax_1K_lane", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &sm).unwrap();
        })
    });
    let ge = instr_gelu(2, 0);
    g.bench_function("gelu_1K", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &ge).unwrap();
        })
    });
    g.finish();
}

criterion_group!(activation, bench_activation,);
criterion_main!(activation);
