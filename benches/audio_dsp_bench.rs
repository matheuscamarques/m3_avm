//! benches/audio_dsp_bench.rs — Benches RFC-0031 (0x45-0x47, parcial).
//!
//! vad/resample/merge sobre frame 1920, por instrução via `step_instruction`.
//! Rode: `cargo bench --bench audio_dsp_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::{instr_audio_resample, instr_stream_merge, instr_vad_detect, VAD_MODE_ENERGY},
    vm::{Vm, VmConfig},
};
use std::time::Duration;

/// Vm pronta com frame 1920 de 0.5 em r0 (+ cópia em r1).
fn ready_vm() -> (Vm, u64) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let root = vm.memory.current_version();
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, root);
    let a = vm.memory.alloc_tensor(&[1920], DType::F32).unwrap();
    vm.memory.write_f32_tensor(a, &vec![0.5; 1920]).unwrap();
    let b = vm.memory.alloc_tensor(&[1920], DType::F32).unwrap();
    vm.memory.write_f32_tensor(b, &vec![0.5; 1920]).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, a).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(1, b).unwrap();
    (vm, cid)
}

fn bench_audio_dsp(c: &mut Criterion) {
    let mut g = c.benchmark_group("audio_dsp");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let vad = instr_vad_detect(2, 0, VAD_MODE_ENERGY);
    g.bench_function("vad_energy_1920", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &vad).unwrap();
        })
    });
    let rs = instr_audio_resample(3, 0, 24000, 16000);
    g.bench_function("resample_1920_24k_16k", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &rs).unwrap();
        })
    });
    let mg = instr_stream_merge(4, 0, 1, 0.5, 0);
    g.bench_function("merge_1920", |ben| {
        ben.iter(|| {
            vm.step_instruction(cid, &mg).unwrap();
        })
    });
    g.finish();
}

criterion_group!(audio_dsp, bench_audio_dsp,);
criterion_main!(audio_dsp);
