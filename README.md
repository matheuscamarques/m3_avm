# M³-AVM: Research Prototype for Sparse Event-Driven Tensor Computation

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange)](https://www.rust-lang.org/)
[![ISA](https://img.shields.io/badge/ISA-6%20Opcodes-blueviolet)](docs/ISA.md)
[![Sparse](https://img.shields.io/badge/Support-Sparse%20%26%20Dense-brightgreen)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)]()

## Abstract — What This Emulator Actually Is

The M³-AVM is a software emulator in Rust (`src/lib.rs:1`, `src/vm.rs:1`) exploring primitives for interactive AI: preemption and sparse memory. It implements:

1. A fixed 6-opcode ISA `TENSOR, ATTN, STREAM, FORK, ABORT, SENSE` (`src/opcodes.rs:26`, `INSTR_SIZE=32` `src/opcodes.rs:23`).
2. A scheduler with strict priority `Red > Blue > Green` (`src/context.rs:15`, `src/context.rs:167`) and an optional event-driven reactor (`src/reactor.rs:1`, `src/bus.rs:20`) using `tokio::sync::watch`/`broadcast`.
3. Dense tensors via `ndarray` and sparse CSR via `nalgebra-sparse 0.10` + `sprs 0.11` (`Cargo.toml:22`, `src/sparse.rs:1`).

**Caveats (verified):** This runs on x86/ARM hosts. It does not bypass the Von Neumann bottleneck and does not run on optical hardware. It is a functional reference, not a production inference engine.

## 1. Problem Space

Sparsity (MoE routing, pruning, long-context KV cache) is common, but current stacks treat it as an optimization, not a primitive. GPU kernels, once launched, run to completion and cannot be preempted per row. The prototype tests whether exposing progress per row-chunk via notifications allows finer preemption, at the cost of scheduler complexity.

## 2. Architecture — Implemented

### Address Space (u128 virtual) `src/memory.rs:21`
- `GLOBAL (0x00…)` — `global_heap: HashMap<u128, Arc<Vec<u8>>>` + `sparse_heap: HashMap<u128, SparseTensor>` `src/memory.rs:122` (dense aligned to 64B `src/memory.rs:221`, sparse CSR).
- `TEMPORAL (0x10…)` — circular buffer 1 GiB logical / 64 MiB physical in dev (`src/memory.rs:27`, `src/memory.rs:31`).
- `PERSISTENTE (0x20…)` — `memmap2::MmapMut` `src/memory.rs:15`, survives restarts (`src/memory.rs:157`). File `m3_persistent.dat`.

| Tensor | Internal | Memory cost (actual) |
| :--- | :--- | :--- |
| Dense | `Vec<u8>` + `TensorMeta` `src/memory.rs:103` | `rows*cols*4` |
| Sparse | `CsrMatrix<f32>` `src/sparse.rs:30` | `(nnz*4)+(rows+1)*4+nnz*4` |

### Context `src/context.rs:84`
- 16 regs `u128`, `pc: u128`, `root_version: u64`, `priority: Priority`, `state: ContextState`.
- No `tensor_type` or `chunk_size` field in code — chunking is done inside `attn_sparse` per row (`src/sparse.rs:183`).

### NOP Buses `src/bus.rs:32`
- `interrupt_tx: watch::Sender<Option<InterruptSignal>>` — `SENSE`/`ABORT` → `ATTN` heads.
- `stream_tx: broadcast::Sender<StreamSignal>` — sink capacity → producer.
- `sched_tx: broadcast::Sender<SchedSignal>` — `FORK` → scheduler (`recv().await` in `src/reactor.rs:45`).
- Emulated latency is `watch`/`broadcast` (≈µs in tests `src/bus.rs:130`), not <10ns silicon.

## 3. ISA — As Implemented

| Opcode | Hex | Syntax (assembler `src/opcodes.rs:332`) | Behavior in emulator |
| :--- | :--- | :--- | :--- |
| `0x01` | **TENSOR** | `TENSOR Rd 32 32 f32` or `TENSOR Rd 32 32 f32 SPARSE DENSITY=0.05` | Dense: `alloc_tensor` `src/memory.rs:238`; Sparse: `alloc_sparse_tensor` `src/memory.rs:258` with `SparseTensor::random` `src/sparse.rs:54` (not `CsrMatrix::zero`). Payload `[17]=is_sparse` `src/opcodes.rs:168`. |
| `0x02` | **ATTN** | `ATTN Rd, Q, K, V` or `ATTN Rd, Q, K, V NOTIFY_EACH_HEAD` | Sparse-aware dispatcher `src/vm.rs:418`: if any sparse → `sparse::attn_sparse` `src/sparse.rs:146` (`Q*K^T` scaled, softmax per row, `*V`, notifies `AttentionEvent::HeadCompleted` per row `src/sparse.rs:183`); else dense `ndarray` `src/vm.rs:454`. Flag `ATTN_FLAG_NOTIFY_EACH_HEAD=0b01` `src/opcodes.rs:42`. |
| `0x03` | **STREAM** | `STREAM Rsrc, Rsink, [BLOCKING/DROP]` | `tokio::sync::mpsc` bounded 16 stub `src/vm.rs:82`; blocking uses `try_send` + warn, reactor uses `StreamSignal` `src/reactor.rs:273`. |
| `0x04` | **FORK** | `FORK Rd, RED` or `FORK Rd, RED, NOTIFY` | CoW via `Arc::clone` `src/memory.rs:475` and `sparse_snapshots` `src/memory.rs:149`; `FORK_FLAG_NOTIFY=0b100` `src/opcodes.rs:44` publishes `SchedSignal` `src/reactor.rs:313`. |
| `0x05` | **ABORT** | `ABORT Rs_ctx, Rs_ts` | Removes context `src/vm.rs:601`, restores snapshot `src/memory.rs:489` (`watch` publish `InterruptSignal` in reactor `src/reactor.rs:318`). |
| `0x06` | **SENSE** | `SENSE Rd, AUDIO/VAD` | Generates white noise `rand::thread_rng` `src/vm.rs:621` or `0/1`, pushes to `TEMPORAL` `src/memory.rs:430`. |

All instructions are 32 bytes `src/opcodes.rs:82`.

## 4. Implementation Status — Verified Gaps

- Sparse is CSR via `nalgebra-sparse`; no native `f16/i8` — cast to `f32` internally.
- No sparse FlashAttention; `attn_sparse` densifies for softmax `src/sparse.rs:190` (`to_dense` `src/sparse.rs:73`), losing memory savings for large heads.
- No BSR backend.
- No `block_size` param for `ATTN` — granularity is per row, not configurable block.
- `ndarray` dense path only `f32`.
- `watch`/`broadcast` are Tokio channels, not hardware crossbar.

Tests that pass on this host: `cargo test --lib` 67 tests `src/qa.rs:800` + `src/sparse.rs:240` + `src/bus.rs:120` + `src/reactor.rs:377`.

## 5. Benchmarking — Measured (not claimed)

All measurements on this host are for small matrices; large 2048x2048 numbers in the draft were hypothetical and are not included.

| Scenario (measured) | Matrix | Operation | Time (debug) | Note |
| :--- | :--- | :--- | :--- | :--- |
| `ATTN` dense | 32x32 | `attn_sparse` 1.0 density | ~5 ms iter `cargo bench sparse_nop` | |
| `ATTN` sparse | 32x32 5% | `attn_sparse` + `NOTIFY_EACH_HEAD` | ~5 ms + overhead <3x per head `src/qa.rs:860` | overhead measured with `watch` |
| Program `sparse_nop.m3asm` | 6 tensors 32x32 | 2x ATTN + 2x STREAM | 45 ms debug `cargo run -- run examples/sparse_nop.m3asm` | dense 4x4 part ~similar |
| Scheduler IPS | NOP loop | 100k `NOP` | 240k IPS debug, 2.1M IPS release (`cargo run --release -- bench --nops 1000000`) | target 1M met in release |
| FORK CoW | 100x1MiB | `snapshot` | <1 ms `src/qa.rs:270` | `Arc` clone |
| ABORT | 1 sparse ATTN 32x32 under load | `ABORT` | 25 ms measured `src/qa.rs:664` (not 50µs; 50µs is hypothetical HW) | watch publish <1ms `src/bus.rs:130` |
| Memory saving | 64x64 5% | CSR vs dense | sparse ~10x smaller `src/qa.rs:900` (`nnz*4+...` < `rows*cols*4`) | verified |

## 6. Research Gaps

1. No autograd for sparse Jacobians.
2. No BSR / FlashAttention-3 sparse.
3. `std::time::Instant` used for timestamps `src/utils.rs:9`, not synchronized across epochs.
4. No rate-limiting; `SENSE` flood could thrash reactor.

## 7. Comparison — Honest

| Aspect | CUDA/PyTorch | M³-AVM (this emulator) |
| :--- | :--- | :--- |
| Preemption | Kernel ~500 ms (reported) | Block-level via `watch` — measured ~25 ms for 32x32 sparse ATTN under load; publish <1ms |
| Sparse | cuSPARSE opaque | First-class CSR `src/sparse.rs`, not accelerated |
| I/O | ALSA/syscalls | `TEMPORAL` `mmap` + `rand` stub |
| Rollback | Disk checkpoint | CoW snapshot `src/memory.rs:475` — clones `Arc`+`HashMap` in µs for small heaps |
| Scheduler | FIFO | strict `Red>Blue>Green` `src/context.rs:194` |

This is a testbed, not a replacement.

## 8. Integration with JusrisOS

Current state: *Proposed, not integrated.* The emulator is standalone (`src/main.rs`). Integration via `persistent_term` dispatcher and `Rustler` NIF was reverted to focus on VM (`docs/PLANO_NOP.md`). A future adapter would be `lib/jusris_os_core/adapters/llm_m3.ex` via `Port` or `mmap`, but no NIF exists in this repo today (`native/` is Tauri shell, not Rustler).

## 8.1 Model Selection — Measured on Ryzen 3500U

Empirical: `ggml-tiny-q8_0.bin` 42MB (RTF 0.56 for 5s `mic.wav`, `encode 1847ms`) is 44% faster than `q5_1` 31MB (RTF 1.00, `encode 3043ms`) and stable in silence (5s silêncio 2441ms vs 6612ms, 10s 2334ms vs 24007ms). Both fit in `PERSISTENTE` 64MiB `src/memory.rs:32`; `FP32` 75MB does not. Default is now `MODEL_DEFAULT="ggml-tiny-q8_0.bin"` `src/stt.rs:20`.

## 9. Getting Started — Commands That Actually Work

Prerequisites: Rust 1.75+ (`rustc 1.93` tested), `cargo`, `cmake` for `whisper.cpp`.

```bash
git clone <repo> && cd m3_avm
cargo build
cargo run -- run examples/minimal.m3asm        # dense 2x2
cargo run -- run examples/sparse_nop.m3asm     # sparse 32x32 5% + NOTIFY_EACH_HEAD
cargo run -- run examples/nop_demo.m3asm
cargo run -- assemble examples/minimal.m3asm -o /tmp/minimal.m3bin
cargo run -- disassemble /tmp/minimal.m3bin
cargo test --lib                               # 67 tests
cargo test --lib sparse_nop -- --nocapture
cargo test --lib bus -- --nocapture
cargo test --lib reactor -- --nocapture
cargo bench --bench physics_bench -- --quick    # includes sparse_nop group
```

What does *not* exist: `cargo run --bin m3_tui` (no TUI), `cargo test test_sparse_attention` (actual names are `sparse::*` and `qa::sparse_nop::*`).

Example verified: `examples/sparse_nop.m3asm:1`
```asm
TENSOR r0 32 32 f32 SPARSE DENSITY=0.05
TENSOR r1 32 32 f32 SPARSE DENSITY=0.10
ATTN r3 r0 r1 r2 NOTIFY_EACH_HEAD
STREAM r3 r15 BLOCKING
```

## 10. Immediate Roadmap

- BSR backend for block-sparse attention.
- Keep softmax dense densification or implement sparse softmax.
- Validate 1024+ dimensions (currently tested up to 64 for speed).
- Re-introduce JusrisOS adapter when `TARGET` is defined.

## License

Apache 2.0. Contributions welcome, especially sparse-dense matmul optimizations.
