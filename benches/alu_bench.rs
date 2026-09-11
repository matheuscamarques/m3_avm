//! benches/alu_bench.rs — Benches RFC-0026 (0x7A/0x7B/0x7C).
//!
//! Reg-ALU pura + contador, por instrução via `step_instruction`.
//! Rode: `cargo bench --bench alu_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    opcodes::{instr_add_imm, instr_halt, instr_steps, instr_sub_imm},
    vm::{Vm, VmConfig},
};
use std::time::Duration;

/// Vm pronta com r0 = 100 no contexto GREEN em 0x1000.
fn ready_vm() -> (Vm, u64) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let root = vm.memory.current_version();
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, root);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, 100).unwrap();
    // Programa mínimo p/ CALL validar alvo (RFC-0034).
    vm.load_program(vec![instr_halt()]);
    (vm, cid)
}

fn bench_alu(c: &mut Criterion) {
    let mut g = c.benchmark_group("alu");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let add = instr_add_imm(1, 0, 23);
    g.bench_function("add_imm", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &add).unwrap();
        })
    });
    let sub = instr_sub_imm(2, 0, 23);
    g.bench_function("sub_imm", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &sub).unwrap();
        })
    });
    let stp = instr_steps(3);
    g.bench_function("steps", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &stp).unwrap();
        })
    });
    // Par CALL+RET (empilha e desempilha; pilha nunca cresce aqui).
    let call = m3_avm::opcodes::instr_call(0x1000);
    let ret = m3_avm::opcodes::instr_ret();
    g.bench_function("call_ret", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &call).unwrap();
            vm.step_instruction(cid, &ret).unwrap();
        })
    });
    g.finish();
}

criterion_group!(alu, bench_alu,);
criterion_main!(alu);
