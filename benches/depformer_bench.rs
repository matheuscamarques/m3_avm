//! benches/depformer_bench.rs — Bench RFC-0032 (0x44).
//!
//! Passo depformer pequeno (D=32, NCB=4, Q=8, 2 heads), com tabela
//! montada em Rust (u64 não se soletra em `.m3asm`), por instrução
//! via `step_instruction`.
//! Rode: `cargo bench --bench depformer_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::instr_depformer,
    vm::{Vm, VmConfig},
};
use std::time::Duration;

/// Vm pronta com rX [1,32] e tabela (pesos FILL=1) em rW.
fn ready_vm() -> (Vm, u64) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let root = vm.memory.current_version();
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, root);
    let d = 32usize;
    let (ncb, q) = (4usize, 8usize);
    let mat = |vm: &mut Vm, rows: usize, cols: usize| {
        let a = vm.memory.alloc_tensor(&[rows, cols], DType::F32).unwrap();
        vm.memory.write_f32_tensor(a, &vec![1.0; rows * cols]).unwrap();
        a
    };
    let (wq, wk, wv, wo) = (mat(&mut vm, d, d), mat(&mut vm, d, d), mat(&mut vm, d, d), mat(&mut vm, d, d));
    let wh = mat(&mut vm, d, ncb * q);
    let t = vm.memory.alloc_tensor(&[64], DType::U8).unwrap();
    let mut bytes = Vec::new();
    for a in [wq, wk, wv, wo, wh] {
        bytes.extend_from_slice(&u64::try_from(a).unwrap().to_le_bytes());
    }
    bytes.extend_from_slice(&[0u8; 24]);
    vm.memory.write(t, &bytes).unwrap();
    let x = vm.memory.alloc_tensor(&[1, d], DType::F32).unwrap();
    vm.memory.write_f32_tensor(x, &vec![1.0; d]).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, x).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(1, t).unwrap();
    (vm, cid)
}

fn bench_depformer(c: &mut Criterion) {
    let mut g = c.benchmark_group("depformer");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let dp = instr_depformer(9, 0, 1, 0, 0, 4, 2, 8, 8, 1.0, 0);
    g.bench_function("depformer_step_d32", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &dp).unwrap();
        })
    });
    g.finish();
}

criterion_group!(depformer, bench_depformer,);
criterion_main!(depformer);
