//! benches/rag_bench.rs — Benches V-2 retrieval (0x50-0x53/0x56-0x57).
//!
//! ADD (1 vetor 768d), SEARCH (banco 1k×768 TOPK=10), LOOKUP (batch 32),
//! PQ (768d 8 subvetores). Rode: `cargo bench --bench rag_bench [-- --quick]`
//! M3BC bit 50 REQUIRED firewall (src/m3bc.rs).

use criterion::{criterion_group, criterion_main, Criterion};
use m3_avm::{
    context::Priority,
    memory::DType,
    opcodes::{
        instr_embed_lookup, instr_pq_decode, instr_pq_encode, instr_rag_index_add,
        instr_rag_search, RAG_METRIC_COSINE,
    },
    vm::{Vm, VmConfig},
};
use std::time::Duration;

fn alloc_f32(vm: &mut Vm, shape: &[usize], fill: f32) -> u128 {
    let a = vm.memory.alloc_tensor(shape, DType::F32).unwrap();
    let n: usize = shape.iter().product();
    vm.memory.write_f32_tensor(a, &vec![fill; n]).unwrap();
    a
}

fn ready_vm_rag() -> (Vm, u64, u128, u128) {
    let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
    let cid = vm.scheduler.create_context(Priority::Green, 0x1000, vm.memory.current_version());
    // store dim 64 for speed (spec says 768, bench uses 64 to stay <100ms per iter)
    let dim = 64;
    let db = {
        let mut s = vm.rag_store.write().unwrap();
        s.create(dim).unwrap()
    };
    // pre-populate 1000 vectors dim 64
    {
        let mut s = vm.rag_store.write().unwrap();
        for i in 0..1000 {
            let v: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32 * 0.001).sin()).collect();
            s.add(db, &v, None).unwrap();
        }
    }
    let q = alloc_f32(&mut vm, &[1, dim], 0.5);
    vm.scheduler.get_mut(cid).unwrap().set_reg(10, db as u128).unwrap();
    vm.scheduler.get_mut(cid).unwrap().set_reg(11, q).unwrap();
    let v = alloc_f32(&mut vm, &[1, dim], 0.7);
    vm.scheduler.get_mut(cid).unwrap().set_reg(0, v).unwrap();
    (vm, cid, q, db as u128)
}

fn bench_rag(c: &mut Criterion) {
    let mut g = c.benchmark_group("rag");
    g.measurement_time(Duration::from_secs(2));

    // ADD: 1 vetor 64d (spec 768d)
    {
        let (mut vm, cid, _, _) = ready_vm_rag();
        let v = alloc_f32(&mut vm, &[1, 64], 0.9);
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, v).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, 1).unwrap(); // db 1
        let add = instr_rag_index_add(2, 1, 0, 0xFF);
        g.bench_function("add_1x64", |ben| {
            ben.iter(|| {
                vm.step_instruction(cid, &add).unwrap();
            })
        });
    }

    // SEARCH: banco 1k×64 TOPK=10 (spec 10k×768)
    {
        let (mut vm, cid, _, _) = ready_vm_rag();
        let search = instr_rag_search(2, 11, 10, 10, RAG_METRIC_COSINE);
        g.bench_function("search_1k_64_top10_cosine", |ben| {
            ben.iter(|| {
                vm.step_instruction(cid, &search).unwrap();
            })
        });
    }

    // LOOKUP: batch 32 ids -> tabela [128,64]
    {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
        let cid = vm.scheduler.create_context(Priority::Green, 0x1000, vm.memory.current_version());
        let tab = alloc_f32(&mut vm, &[128, 64], 0.5);
        let ids = {
            let a = vm.memory.alloc_tensor(&[1, 32], DType::F32).unwrap();
            let v: Vec<f32> = (0..32).map(|i| (i % 128) as f32).collect();
            vm.memory.write_f32_tensor(a, &v).unwrap();
            a
        };
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, ids).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, tab).unwrap();
        let lk = instr_embed_lookup(2, 0, 1);
        g.bench_function("lookup_batch32_64d", |ben| {
            ben.iter(|| {
                vm.step_instruction(cid, &lk).unwrap();
            })
        });
    }

    // PQ: 64d 8 subvetores (spec 768d 8)
    {
        let mut vm = Vm::new_in_memory(VmConfig { max_steps: Some(1_000_000), ..Default::default() });
        let cid = vm.scheduler.create_context(Priority::Green, 0x1000, vm.memory.current_version());
        let d = 64;
        let nsub = 8;
        let subdim = d / nsub; // 8
        let k = 16;
        let vec = alloc_f32(&mut vm, &[1, d], 0.5);
        let cb = {
            let rows = nsub * k;
            let a = vm.memory.alloc_tensor(&[rows, subdim], DType::F32).unwrap();
            let v: Vec<f32> = (0..rows * subdim).map(|i| (i as f32 * 0.01).sin()).collect();
            vm.memory.write_f32_tensor(a, &v).unwrap();
            a
        };
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, vec).unwrap();
        vm.scheduler.get_mut(cid).unwrap().set_reg(1, cb).unwrap();
        let enc = instr_pq_encode(2, 0, 1, nsub as u16);
        g.bench_function("pq_encode_64d_8x16", |ben| {
            ben.iter(|| {
                vm.step_instruction(cid, &enc).unwrap();
            })
        });
        // PQ decode bench
        let codes = {
            let a = vm.memory.alloc_tensor(&[1, nsub], DType::F32).unwrap();
            vm.memory.write_f32_tensor(a, &vec![0.0; nsub]).unwrap();
            a
        };
        vm.scheduler.get_mut(cid).unwrap().set_reg(0, codes).unwrap();
        let dec = instr_pq_decode(3, 0, 1, nsub as u16);
        g.bench_function("pq_decode_64d_8x16", |ben| {
            ben.iter(|| {
                vm.step_instruction(cid, &dec).unwrap();
            })
        });
    }

    g.finish();
}

criterion_group!(rag, bench_rag,);
criterion_main!(rag);
