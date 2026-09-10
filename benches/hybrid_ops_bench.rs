//! benches/hybrid_ops_bench.rs — Benches dos opcodes 0x13–0x19 (ESPEC §16).
//!
//! Mede por instrução via `step_instruction` (custo que programas pagam,
//! incluindo alloc de saída), mais o rollup da janela de 80ms
//! (CODEC_ENC -> SSM_SCAN -> AUDIO_ALIGN) que ancora o T_proc do T6.
//! Rode: `cargo bench --bench hybrid_ops_bench [-- --quick]`

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::{
        instr_audio_align, instr_codec_dec, instr_codec_enc, instr_ctx_switch,
        instr_rope, instr_ssm_reset, instr_ssm_scan, PIPE_MAMBA,
        CODEC_FLAG_AS_TENSOR,
    },
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

fn f32_tensor(vm: &mut Vm, shape: &[usize], fill: f32) -> u128 {
    let n: usize = shape.iter().product();
    let a = vm.memory.alloc_tensor(shape, DType::F32).unwrap();
    vm.memory.write_f32_tensor(a, &vec![fill; n]).unwrap();
    a
}

fn bench_ssm_scan(c: &mut Criterion) {
    let mut g = c.benchmark_group("ssm_scan");
    g.measurement_time(Duration::from_secs(2));
    // di=64, ds=16: estado 1024 f32 (~4 KiB), representativo e rápido.
    let (mut vm, cid) = ready_vm();
    let xa = f32_tensor(&mut vm, &[1, 64], 0.5);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
    let scan = instr_ssm_scan(2, 0, 0xFF, 0xFF, 64, 16, 0);
    g.bench_function("di64_ds16", |b| {
        b.iter(|| {
            vm.step_instruction(cid, &scan).unwrap();
        })
    });
    g.finish();
}

fn bench_ssm_reset(c: &mut Criterion) {
    let mut g = c.benchmark_group("ssm_reset");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let reset = instr_ssm_reset(0xFF, 64, 16, 0);
    // Garante estado existente (custo O(1) real, não ensure).
    let xa = f32_tensor(&mut vm, &[1, 64], 0.5);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
    vm.step_instruction(cid, &instr_ssm_scan(2, 0, 0xFF, 0xFF, 64, 16, 0)).unwrap();
    g.bench_function("reset", |b| {
        b.iter(|| {
            vm.step_instruction(cid, &reset).unwrap();
        })
    });
    g.finish();
}

fn bench_codec(c: &mut Criterion) {
    let mut g = c.benchmark_group("codec");
    g.measurement_time(Duration::from_secs(3));
    // Path tensor [1,1920] com frame synth (pior caso honesto, sem I/O).
    let (mut vm, cid) = ready_vm();
    let pa = vm.memory.alloc_tensor(&[1, 1920], DType::F32).unwrap();
    let synth = m3_avm::mimi::synth_frame_440hz();
    vm.memory.write_f32_tensor(pa, &synth).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, pa).unwrap();
    let mut enc = instr_codec_enc(1, 0, true);
    enc.flags = CODEC_FLAG_AS_TENSOR;
    g.bench_function("enc_frame", |b| {
        b.iter(|| {
            vm.step_instruction(cid, &enc).unwrap();
        })
    });
    // DEC sobre codes fixos (saída do primeiro ENC).
    vm.step_instruction(cid, &enc).unwrap();
    let ca = vm.scheduler.get(cid).unwrap().reg(1).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(1, ca).unwrap();
    let mut dec = instr_codec_dec(2, 1, true);
    dec.flags = CODEC_FLAG_AS_TENSOR;
    g.bench_function("dec_frame", |b| {
        b.iter(|| {
            vm.scheduler.get_mut(cid).unwrap().set_reg(1, ca).unwrap();
            vm.step_instruction(cid, &dec).unwrap();
        })
    });
    g.finish();
}

fn bench_audio_align(c: &mut Criterion) {
    let mut g = c.benchmark_group("audio_align");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, 1_000_000_000).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(1, 1_080_000_000).unwrap();
    let al = instr_audio_align(2, 0, 1);
    g.bench_function("align", |b| {
        b.iter(|| {
            vm.step_instruction(cid, &al).unwrap();
        })
    });
    g.finish();
}

fn bench_ctx_switch(c: &mut Criterion) {
    let mut g = c.benchmark_group("ctx_switch");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    let cs = instr_ctx_switch(PIPE_MAMBA, 0b10);
    g.bench_function("fence", |b| {
        b.iter(|| {
            vm.step_instruction(cid, &cs).unwrap();
        })
    });
    g.finish();
}

fn bench_rope(c: &mut Criterion) {
    let mut g = c.benchmark_group("rope");
    g.measurement_time(Duration::from_secs(2));
    let (mut vm, cid) = ready_vm();
    // 32 heads x 128 = 4096 (Moshi-like por camada).
    let xa = f32_tensor(&mut vm, &[1, 4096], 0.25);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, xa).unwrap();
    let rp = instr_rope(1, 0, 7, 128, 32, 10_000.0);
    g.bench_function("4096_pos7", |b| {
        b.iter(|| {
            vm.step_instruction(cid, &rp).unwrap();
        })
    });
    g.finish();
}

/// Janela de 80ms ponta a ponta (T6): ENC -> SCAN -> ALIGN por iteração.
/// NOTA: o SCAN usa x dedicado [1,16] — alimentar o PCM [1,1920] faria o
/// op ler 1920 floats p/ truncar a 16 (custo real, mas não representativo;
/// ver follow-up de leitura parcial em exec_ssm_scan).
fn bench_audio_window_80ms(c: &mut Criterion) {
    let mut g = c.benchmark_group("audio_window_80ms");
    g.measurement_time(Duration::from_secs(4));
    let (mut vm, cid) = ready_vm();
    let pa = vm.memory.alloc_tensor(&[1, 1920], DType::F32).unwrap();
    let synth = m3_avm::mimi::synth_frame_440hz();
    vm.memory.write_f32_tensor(pa, &synth).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, pa).unwrap();
    let xa = f32_tensor(&mut vm, &[1, 16], 0.5);
    vm.scheduler.get_mut(cid).unwrap().set_reg(10, xa).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(8, 1_000_000_000).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(9, 1_080_000_000).unwrap();
    let mut enc = instr_codec_enc(1, 0, true);
    enc.flags = CODEC_FLAG_AS_TENSOR;
    let scan = instr_ssm_scan(2, 10, 0xFF, 0xFF, 16, 16, 0);
    let al = instr_audio_align(3, 8, 9);
    g.bench_function("enc_scan_align", |b| {
        b.iter(|| {
            vm.step_instruction(cid, &enc).unwrap();
            vm.step_instruction(cid, &scan).unwrap();
            vm.step_instruction(cid, &al).unwrap();
        })
    });
    g.finish();
}

criterion_group!(
    hybrid,
    bench_ssm_scan,
    bench_ssm_reset,
    bench_codec,
    bench_audio_align,
    bench_ctx_switch,
    bench_rope,
    bench_audio_window_80ms,
);
criterion_main!(hybrid);
