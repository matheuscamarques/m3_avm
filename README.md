# M³-AVM — Multimodal Multi-Model Abstract Virtual Machine

*Research prototype for sparse event-driven tensor computation.*

> **Author: Matheus de Camargo Marques** — Independent Researcher \
> ORCID: [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258) · Email: <matheuscamarques@gmail.com> · GitHub: [@matheuscamarques](https://github.com/matheuscamarques) \
> If you use this work, please cite it (see [CITATION.cff](CITATION.cff) / [How to cite](#license--citation)).

[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--4518--2258-A6CE39?style=flat&logo=orcid&logoColor=white)](https://orcid.org/0009-0003-4518-2258)
[![Author](https://img.shields.io/badge/Author-matheuscamarques-181717?style=flat&logo=github&logoColor=white)](https://github.com/matheuscamarques)
[![Email](https://img.shields.io/badge/Email-matheuscamarques%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:matheuscamarques@gmail.com)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange)](https://www.rust-lang.org/)
[![ISA](https://img.shields.io/badge/ISA-64%20Opcodes-blueviolet)](docs/ESPEC.md)
[![Sparse](https://img.shields.io/badge/Support-Sparse%20%26%20Dense-brightgreen)]()
[![License](https://img.shields.io/badge/License-AGPL_v3.0-blue)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow)]()
[![DOI](https://zenodo.org/badge/10.5281/zenodo.22694886.svg)](https://doi.org/10.5281/zenodo.22694886)

## Abstract — What This Emulator Actually Is

The M³-AVM is a software emulator in Rust (`src/lib.rs:1`, `src/vm.rs:1`) exploring primitives for interactive AI: preemption and sparse memory. It implements:

1. A fixed 87-opcode ISA v1.9 (`0x00–0x37`, `0x38`, `0x60–0x69`, `0x6A–0x79`, `0x7A–0x7C`, `0xFF`): `TENSOR, ATTN, STREAM, FORK, ABORT, SENSE` + `NORM, FFN` (transformer) + `EMBED, ADD, SAMPLE` (thinking loop) + `COMPARE, JUMP, IF_EQUAL, IF_INTERRUPT` (control flow) + `MATVEC, MUL, SILU` (GEMV blocks) + `SSM_SCAN, SSM_RESET` (Mamba) + `CODEC_ENC, CODEC_DEC, AUDIO_ALIGN` (Mimi full-duplex) + `CTX_SWITCH, ROPE` (hybrid control) + `REMOTE_SPAWN, SIGNAL, SEND_TENSOR, BARRIER` (local cluster — RFC-0018) + `CONV, GATHER, SPIKE_STEP, DENOISE_STEP, FOREST, DISTANCE, RANK1_UPDATE, ODE_STEP` (universal waves — RFC-0004/0012–0015/0017) + `KV_TRUNCATE` + `SLICE` (RFC-0019) + `SORT, TOPK, ARGMAX, REDUCE, BROADCAST, PAD, TILE, TRANSPOSE` (shape — RFC-0027) + `ARENA_ALLOC/RESET, MEMCPY/MEMSET` (memory core — RFC-0023) + `SNAPSHOT/RESTORE, PREFETCH/RESHAPE/CONCAT` (views & versions — RFC-0024) + `CAST, QUANTIZE, DEQUANT` (conversion — RFC-0025) + `RNG_*, HASH, CHECKSUM, HMAC` (determinism — RFC-0005) + telemetry/scheduler block (RFC-0006) + `LOADI, MOV` (RFC-0007) + `ADD_IMM, SUB_IMM, STEPS` (control-plane ALU — RFC-0026) (`INSTR_SIZE=32`, spec `docs/ESPEC.md`, RFCs `docs/RFC-*.md`). The assembler is 2-pass with labels (`LOOP:`, `JUMP LOOP`, `FORK Rd, LABEL`) and strict unknown-token rejection (RFC-0008).
2. A scheduler with strict priority `Red > Blue > Green` (`src/context.rs:15`, `src/context.rs:167`) and an optional event-driven reactor (`src/reactor.rs:1`, `src/bus.rs:20`) using `tokio::sync::watch`/`broadcast`.
3. Dense tensors via `ndarray` + `faer` SIMD and sparse CSR via `nalgebra-sparse 0.10` + `sprs 0.11` (`Cargo.toml:22`, `src/sparse.rs:1`), plus quantized dequant `Q4_0/Q4_K/Q6_K/Q8_0` `src/quant.rs:1` and fused `matvec_q4k` `AVX2` `src/matvec_quant.rs:1`.
4. Real inference from GGUF (`src/inference.rs:1`, `src/gguf.rs:1`) with `mmap` zero-copy in `PERSISTENTE 0x20` (`src/memory.rs:15`), `KV_CACHE 0x30` per-layer (`src/memory.rs:27`) and `AsmEmitter` (`src/asm_emitter.rs:1`) that lowers a model to `.m3asm` desenrolado.

**Caveats (verified):** This runs on x86/ARM hosts. It does not bypass the Von Neumann bottleneck and does not run on optical hardware. It is a functional reference, not a production inference engine. Vega 8 `wgpu` (`src/memory_wgpu.rs:1`) only accelerates `ATTN <=64x64`; the hot path is still `matvec` on CPU.

## 1. Problem Space

Sparsity (MoE routing, pruning, long-context KV cache) is common, but current stacks treat it as an optimization, not a primitive. GPU kernels, once launched, run to completion and cannot be preempted per row. The prototype tests whether exposing progress per row-chunk via notifications allows finer preemption, at the cost of scheduler complexity.

## 2. Architecture — Implemented

> Full target vision (diagram + heterogeneous engines + distributed roadmap): see [docs/ESPEC.md](docs/ESPEC.md).

### Address Space (u128 virtual) `src/memory.rs:21`
- `GLOBAL (0x00…)` — `global_heap: HashMap<u128, Arc<Vec<u8>>>` + `sparse_heap: HashMap<u128, SparseTensor>` `src/memory.rs:122` (dense aligned to 64B `src/memory.rs:221`, sparse CSR).
- `TEMPORAL (0x10…)` — circular buffer 1 GiB logical / 64 MiB physical in dev (`src/memory.rs:27`, `src/memory.rs:31`).
- `PERSISTENTE (0x20…)` — `memmap2::MmapMut` `src/memory.rs:15`, survives restarts (`src/memory.rs:157`). File `m3_persistent.dat`. GGUF weights are `mmap`ed read-only (`model_mmap: Arc<Mmap>`) with `madvise WillNeed` and exposed via `get_tensor_f32_slice` zero-copy (`src/memory.rs:506`).
- `KV_CACHE (0x30…)` — 22+ layers, `seq_len*hidden` per layer, 1 GiB logical / 64 MiB physical (`src/memory.rs:43`, `src/memory.rs:380`) with `snapshot`/`restore` for rollback.

| Tensor | Internal | Memory cost (actual) |
| :--- | :--- | :--- |
| Dense | `Vec<u8>` + `TensorMeta` `src/memory.rs:103` | `rows*cols*4` |
| Sparse | `CsrMatrix<f32>` `src/sparse.rs:30` | `(nnz*4)+(rows+1)*4+nnz*4` |
| Quant | `Q4_K 144B/256` `src/quant.rs:38` | `(n/256)*144` |

### Context `src/context.rs:84`
- 16 regs `u128`, `pc: u128`, `root_version: u64`, `priority: Priority`, `state: ContextState`, plus `cmp_equal: bool` (COMPARE/IF_EQUAL) and `interrupt_flag: bool` (SENSE USER_INPUT / IF_INTERRUPT).

### NOP Buses `src/bus.rs:32`
- `interrupt_tx: watch::Sender<Option<InterruptSignal>>` — `SENSE`/`ABORT` → `ATTN` heads.
- `stream_tx: broadcast::Sender<StreamSignal>` — sink capacity → producer.
- `sched_tx: broadcast::Sender<SchedSignal>` — `FORK` → scheduler (`recv().await` in `src/reactor.rs:45`).
- Emulated latency is `watch`/`broadcast` (≈µs in tests `src/bus.rs:130`), not <10ns silicon.

### Real Inference `src/inference.rs:1`
- `ModelConfig::from_gguf` parses `hidden/intermediate/n_layers/n_heads/n_kv_heads/vocab/arch` from GGUF KV (Qwen2/Llama/Mistral).
- `RealInference` holds `weight_cache: FxHashMap<String, Arc<Vec<f32>>>` (`src/inference.rs:180`) and `LayerNames` pre-resolved indices (`src/inference.rs:155`) — avoids `format!`+`HashMap` per `matvec` (154 lookups/token).
- `matvec_weight_pre` (`src/inference.rs:370`) tries `read_model_raw` zero-copy + `matvec_quant::matvec_q4k` `AVX2` fused (`src/matvec_quant.rs:249`) before falling back to `faer` (`src/matvec.rs:37`). `embedding_row` row-slice avoids materializing `934MB`.
- `forward_one` loop `src/inference.rs:527` does `RMSNorm → Q/K/V matvec → RoPE → KV append → per-head softmax → O proj → residual → FFN gate/up/down` for `22-28` layers, with `M3_PROFILE=1` breakdown.
- `AsmEmitter` (`src/asm_emitter.rs:1`) lowers the same `ModelConfig` to desenrolado `.m3asm` (`TENSOR r0 1 hidden` + `NORM/ATTN/FFN/ADD` per layer + `SENSE/IF_INTERRUPT`).

### Vega 8 (optional) `src/memory_wgpu.rs:1`
- `wgpu 0.19` `RADV RAVEN 8 CUs @1200MHz 1.1 TFLOPS` via `Vulkan`. `WgpuMemoryManager` mirrors `MemoryManager` with `STORAGE` buffers + `cpu_fallback` shadow. `attn_wgpu` `src/memory_wgpu.rs:185` does `QKT/softmax/SM*V` in 3 dispatches, `<=64x64`. For LLM `2048` the hot path stays on CPU — GPU gain `<5%` today, `~1.5x` est. with resident `GEMV.wgsl`.

## 3. ISA — As Implemented

| Opcode | Hex | Syntax (assembler `src/opcodes.rs:332`) | Behavior in emulator |
| :--- | :--- | :--- | :--- |
| `0x01` | **TENSOR** | `TENSOR Rd 32 32 f32` or `TENSOR Rd 32 32 f32 SPARSE DENSITY=0.05` | Dense: `alloc_tensor` `src/memory.rs:238`; Sparse: `alloc_sparse_tensor` `src/memory.rs:258` with `SparseTensor::random` `src/sparse.rs:54`. GGUF-shaped `F32 2048x5632` is auto-mapped to `PERSISTENTE` `Q4_K` via `try_gguf_tensor` (`src/memory.rs:380`), `vm.rs:718` skips init for `PERSISTENTE`. Payload `[17]=is_sparse` `src/opcodes.rs:168`. |
| `0x02` | **ATTN** | `ATTN Rd, Q, K, V` or `ATTN Rd, Q, K, V NOTIFY_EACH_HEAD` | Sparse-aware dispatcher `src/vm.rs:418`: if any sparse → `sparse::attn_sparse` `src/sparse.rs:146`; else dense `ndarray` `src/vm.rs:906` or `attn_wgpu` if `<=64` and `feature=wgpu` `src/vm.rs:875`. Flag `0b01` `src/opcodes.rs:42`. |
| `0x03` | **STREAM** | `STREAM Rsrc, Rsink, [BLOCKING/DROP]` or `STREAM Rsrc, SAMPLE/DECODED` | `tokio::sync::mpsc` bounded 16 stub `src/vm.rs:82`; `SAMPLE` (`2`) does `softmax+sampling` → `last_sample`, `DECODED` (`4`) decodes via `M3Tokenizer`. |
| `0x04` | **FORK** | `FORK Rd, RED` or `FORK Rd, RED, NOTIFY` or `FORK Rd, LABEL, GREEN` | CoW via `Arc::clone` `src/memory.rs:475` and `sparse_snapshots` `src/memory.rs:149`; `NOTIFY=0b100` `src/opcodes.rs:44` publishes `SchedSignal`. Label form starts child at label PC (2-pass). |
| `0x05` | **ABORT** | `ABORT Rs_ctx, Rs_ts` | Removes context `src/vm.rs:1350`, restores snapshot `src/memory.rs:489`. |
| `0x06` | **SENSE** | `SENSE Rd, AUDIO/VAD/TOKEN/USER_INPUT` | White noise `rand` `src/vm.rs:621` or `0/1`, pushes to `TEMPORAL`. `USER_INPUT` pops host queue `Vm::push_input`: writes `1/0` + `interrupt_flag` `src/vm.rs:1605` — pairs with `IF_INTERRUPT`. |
| `0x07` | **NORM** | `NORM Rd, Rsrc, Rgamma, Rbeta` | RMSNorm `x/sqrt(mean(x²)+eps)*gamma+beta` `src/vm.rs:1380`. |
| `0x08` | **FFN** | `FFN Rd, Rsrc, Rw1, Rw2[, Rb1, Rb2]` | SwiGLU `silu(x·W1+b1)*up ·W2+b2` `src/vm.rs:1476`. |
| `0x09` | **EMBED** | `EMBED Rd, Rtoken, Rtable` | Row `token_id % rows` of `[vocab, hidden]` → `[1, hidden]` `src/vm.rs:1172`. |
| `0x0A` | **ADD** | `ADD Rd, R1, R2` | Elementwise `f32` add `src/vm.rs:1250`. |
| `0x0B` | **SAMPLE** | `SAMPLE Rd, Rlogits [TEMP=x]` | Softmax + weighted sampling `src/vm.rs:1270`; writes `Rd` and `last_sample`. |
| `0x0C` | **COMPARE** | `COMPARE R1, R2` or `COMPARE R1, imm/EOS_TOKEN` | Sets `cmp_equal` `src/vm.rs:1300`. |
| `0x0D` | **IF_EQUAL** | `IF_EQUAL LABEL` | Jumps if `cmp_equal` `src/vm.rs:1330`. |
| `0x0E` | **JUMP** | `JUMP LABEL` | Unconditional `src/vm.rs:1315`, target validated. |
| `0x0F` | **IF_INTERRUPT** | `IF_INTERRUPT LABEL` or `IF_INTERRUPT Rcond, LABEL` | With reg `reg!=0`, without `interrupt_flag` (consumed) `src/vm.rs:1335`. |
| `0x10-0x12` | **MATVEC/MUL/SILU** | `MATVEC Rd, Rx, Rw` / `MUL Rd, R1, R2` / `SILU Rd, Rsrc` | GEMV `x·W` (`faer`), elementwise mul, SiLU `x·sigmoid(x)` — building blocks for SwiGLU/heads. |
| `0x13` | **SSM_SCAN** | `SSM_SCAN rY, rX, rH, rP [D_INNER=n D_STATE=n LAYER=n]` | Selective scan `h*=exp(dt·A)+x·B·dt; y=h·C+D·x` (`src/ssm.rs`); `rH=_` uses `Vm::ssm_states[layer]`; full spec `docs/ESPEC.md`. |
| `0x14` | **SSM_RESET** | `SSM_RESET rH [D_INNER=n D_STATE=n LAYER=n]` | Zeroes `h_t` (tensor or `ssm_states[layer]`); pairs with `ABORT` (auto pop of `FORK` snapshot) for Mamba rollback. |
| `0x15` | **CODEC_ENC** | `CODEC_ENC rD, rS [TENSOR]` | PCM `1920xf32` (tensor or `TEMPORAL`) → Mimi codes 32B / `[1,16]` (`src/mimi.rs`). |
| `0x16` | **CODEC_DEC** | `CODEC_DEC rD, rS [TENSOR]` | Inverse: codes → PCM frame. |
| `0x17` | **AUDIO_ALIGN** | `AUDIO_ALIGN rD, rU, rA` | `[t_user, t_ai, delta, frame_id]` for barge-in `t_interrupção` (80ms frames). |
| `0x18` | **CTX_SWITCH** | `CTX_SWITCH MAMBA\|TRANSFORMER\|AUDIO [,RED\|BLUE\|GREEN]` | Sets `ctx.pipeline` + priority + memory fence + `maybe_preempt()` (`src/context.rs`). |
| `0x19` | **ROPE** | `ROPE rD, rS POS=n HDIM=n NHEADS=n` | Native rotary embedding via `inference::apply_rope`; `pos=0` is identity. |
| `SENSE 6/7` | **AUDIO_PCM/CODEC_FRAME** | `SENSE Rd, AUDIO_PCM` / `SENSE Rd, CODEC_FRAME` | Deterministic 24kHz synth frame (7680B PCM) or its 32B codes — feeds `CODEC_ENC` directly. |

All instructions are 32 bytes `src/opcodes.rs:82`.

## 4. Implementation Status — Verified Gaps

- Sparse is CSR; no `f16/i8` native — cast to `f32`.
- No sparse FlashAttention; `attn_sparse` densifies for softmax `src/sparse.rs:190` (`to_dense` `src/sparse.rs:73`).
- No BSR backend.
- `watch`/`broadcast` are Tokio channels, not hardware crossbar.
- `wgpu` only for `ATTN <=64`; `GEMV` hot path is CPU `faer`/`matvec_quant` — unified `DDR4 19GB/s` limits Vega 8 to `~1.5x` est.

Tests that pass on this host: `cargo test --lib` 170 tests (ISA, sparse, bus, reactor, inference, asm_emitter, TUI, VM).

## 5. Benchmarking — Measured (not claimed)

| Scenario (measured) | Matrix | Operation | Time (debug) | Note |
| :--- | :--- | :--- | :--- | :--- |
| `ATTN` dense | 32x32 | `attn_sparse` 1.0 density | ~5 ms iter `cargo bench sparse_nop` | |
| `ATTN` sparse | 32x32 5% | `attn_sparse` + `NOTIFY_EACH_HEAD` | ~5 ms + overhead <3x `src/qa.rs:860` | |
| Program `sparse_nop.m3asm` | 6 tensors 32x32 | 2x ATTN + 2x STREAM | 45 ms debug `cargo run --bin m3_avm -- run examples/sparse_nop.m3asm` | |
| Scheduler IPS | NOP loop | 100k `NOP` | 240k IPS debug, 2.1M IPS release (`cargo run --release -- bench --nops 1000000`) | target 1M met in release |
| FORK CoW | 100x1MiB | `snapshot` | <1 ms `src/qa.rs:270` | `Arc` clone |
| ABORT | 1 sparse ATTN 32x32 under load | `ABORT` | 25 ms measured `src/qa.rs:664` (hypothetical 50µs is HW) | publish <1ms `src/bus.rs:130` |
| Memory saving | 64x64 5% | CSR vs dense | sparse ~10x smaller `src/qa.rs:900` | |
| DeepSeek 1.5B `Q4_K` | 1536 hidden 28 layers | `forward_one` `M3_PROFILE=1` `matvec_q4k AVX2` | ~1.2s/token `0.8 tok/s` `3500U` (layersTOTAL) | `wgpu` today `0%` gain |
| TinyLlama 1.1B `Q4_K` | 2048 hidden 22 layers | same | ~0.33s/token `3 tok/s` | `RUSTFLAGS="-C target-cpu=znver1"` `+10%` |

## 6. Research Gaps

1. No autograd for sparse Jacobians.
2. No BSR / FlashAttention-3 sparse.
3. `std::time::Instant` used for timestamps `src/utils.rs:9`, not synchronized.
4. No rate-limiting; `SENSE` flood could thrash reactor.
5. No resident `GEMV.wgsl` for `Q4_K` — needed for Vega 8 `1.5x`.

## 7. Comparison — Honest

| Aspect | CUDA/PyTorch | M³-AVM (this emulator) |
| :--- | :--- | :--- |
| Preemption | Kernel ~500 ms | Block-level via `watch` — `~25 ms` for 32x32; publish `<1ms` |
| Sparse | cuSPARSE opaque | First-class CSR `src/sparse.rs`, not accelerated |
| I/O | ALSA/syscalls | `TEMPORAL` `mmap` + `rand` stub |
| Rollback | Disk checkpoint | CoW snapshot `src/memory.rs:475` — `Arc`+`HashMap` µs for small heaps |
| Scheduler | FIFO | strict `Red>Blue>Green` `src/context.rs:194` |
| Throughput | cuBLAS 300+ tok/s | `faer` `3 tok/s` TinyLlama `3500U` — functional, `BW-bound` |

This is a testbed, not a replacement.

## 8. Model Selection — Measured on Ryzen 3500U

`ggml-tiny-q8_0.bin` 42MB (RTF 0.56 for 5s `mic.wav`, `encode 1847ms`) is 44% faster than `q5_1` 31MB (RTF 1.00) and stable in silence. Both fit in `PERSISTENTE` 64MiB `src/memory.rs:32`; `FP32` 75MB does not. Default `MODEL_DEFAULT="ggml-tiny-q8_0.bin"` `src/stt.rs:20`. For LLM, `TinyLlama 1.1B Q4_K 638MB` `DeepSeek 1.5B Q4_K 1.1GB` need `persistent_mib 64-2048` and `mmap` lazy.

## 9. Getting Started — Commands That Actually Work

Prerequisites: Rust 1.75+ (`rustc 1.93` tested), `cargo`, `cmake` for `whisper.cpp`.

```bash
git clone <repo> && cd m3_avm
cargo build
cargo run --bin m3_avm -- run examples/minimal.m3asm        # dense 2x2
cargo run --bin m3_avm -- run examples/sparse_nop.m3asm     # sparse 32x32 5% + NOTIFY_EACH_HEAD
cargo run --bin m3_avm -- run examples/nop_demo.m3asm
cargo run --bin m3_avm -- run programs/control_flow_demo.m3asm  # EMBED/ADD/SAMPLE + jumps

# Assembly roundtrip
cargo run --bin m3_avm -- assemble examples/minimal.m3asm -o /tmp/minimal.m3bin
cargo run --bin m3_avm -- disassemble /tmp/minimal.m3bin

# Real inference (GGUF mmap + KV_CACHE 0x30, interactive thèse)
cargo run --bin m3_avm -- run programs/thinking_sample.m3asm --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --real --interactive
M3_PROFILE=1 cargo run --bin m3_avm --release -- run --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --real --max-steps 3  # ~1.2s/token
cargo run --bin m3_avm -- run --model models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf --emit-asm /tmp/gen.m3asm examples/minimal.m3asm
cargo run --bin m3_avm -- run /tmp/gen.m3asm --max-steps 50  # 238 instr desenroladas, SENSE/IF_INTERRUPT per layer

# Vega 8 (optional, unified)
cargo run --bin m3_avm --features wgpu -- run --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --real --interactive  # [wgpu] RADV RAVEN
M3_PROFILE=1 cargo run --bin m3_avm --features wgpu --release -- run --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --real --max-steps 3

# Tests & benches
cargo test --lib                               # 170 tests
cargo test --lib sparse_nop -- --nocapture
cargo test --lib asm_emitter -- --nocapture
cargo bench --bench physics_bench -- --quick
```

What does *not* exist: a `m3_tui` binary (the TUI is a REPL module `src/tui.rs` used by `--real --interactive`, not a separate binary), `cargo test test_sparse_attention` (actual names are `sparse::*` and `qa::sparse_nop::*`).

Example verified: `examples/sparse_nop.m3asm:1`
```asm
TENSOR r0 32 32 f32 SPARSE DENSITY=0.05
TENSOR r1 32 32 f32 SPARSE DENSITY=0.10
ATTN r3 r0 r1 r2 NOTIFY_EACH_HEAD
STREAM r3 r15 BLOCKING
```

Thinking loop (`programs/thinking_sample.m3asm:1`, `HALT` in both `run` and `--interactive`):
```asm
    TENSOR r0 2 2 f32
    ADD r2, r0, r1
    SAMPLE r3, r2
    COMPARE r2, r2
    IF_EQUAL DO_ADD
    JUMP DONE
DO_ADD:
    ADD r5, r0, r0
    SENSE r6, USER_INPUT
    IF_INTERRUPT DONE
    JUMP DONE
DONE:
    HALT
```

Emitted program (`--emit-asm` `src/asm_emitter.rs:1`):
```asm
; arch=llama hidden=2048 layers=22
TENSOR r0 1 2048 f32
TENSOR r9 2048 64 f32
MAIN_LOOP:
    SENSE r11, TOKEN
    FORK r12, GREEN, NOTIFY
    NORM r1, r0, r0, r0
    ATTN r5, r1, r1, r1
    ADD r0, r0, r5
    FFN r8, r1, r9, r10
    ADD r0, r0, r8
    SAMPLE r11, r0
    STREAM r11, 4 BLOCKING
    COMPARE r11, 2
    IF_EQUAL PROGRAM_END
    JUMP MAIN_LOOP
HANDLE_ABORT:
    JUMP MAIN_LOOP
PROGRAM_END:
    HALT
```

## 10. Immediate Roadmap

- [x] `EMBED/ADD/SAMPLE` + `COMPARE/JUMP/IF_*` + 2-pass labels.
- [x] Real inference `Q4_K` `AVX2` + `KV_CACHE 0x30` + `AsmEmitter --emit-asm`.
- [x] `FxHash` + `LayerNames` pre-resolve (P0, ~20% `matvec`).
- [x] `MATVEC/MUL/SILU` + Mamba/SSM (`SSM_SCAN/RESET`) + audio codec (`CODEC_ENC/DEC`, `AUDIO_ALIGN`, `CTX_SWITCH`, `ROPE`) — see `docs/ESPEC.md`.
- [ ] Resident `GEMV.wgsl` `Q4_K` for Vega 8 (est. `1.5x`, `BW-bound`).
- [ ] BSR backend for block-sparse attention.
- [ ] Validate 1024+ dims (tested up to 64 for sparse, 2048 for dense).
- [ ] `AVM-Cluster`: distributed actors (`REMOTE_SPAWN/SIGNAL/SEND_TENSOR/BARRIER` `0x1A..0x1D`) — plan in `docs/ESPEC.md`.
- [ ] Universal ISA: `CONV/GATHER/SPIKE_STEP/DENOISE_STEP` (`0x1E..0x21`) + `FOREST/DISTANCE/RANK1_UPDATE/ODE_STEP` (`0x22..0x25`), incl. Titans/DeltaNet/trees/kNN/SVD coverage — map + aliases + canonical table in `docs/ESPEC.md`.

## License — Citation

**© 2026 Matheus de Camargo Marques. Licensed under AGPL-3.0-or-later — see [LICENSE](LICENSE).**

Network use of a modified version (e.g. hosted inference) requires offering the Corresponding Source to users (AGPL §13). For a proprietary/commercial license without copyleft obligations, contact the author.

**How to cite:**

```bibtex
@software{m3_avm_2026,
  author  = {Marques, Matheus de Camargo},
  title   = {M³-AVM: Multimodal Multi-Model Abstract Virtual Machine},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/matheuscamarques/m3_avm},
  license = {AGPL-3.0-or-later}
}
```

> DOI: 10.5281/zenodo.22694886

Or use the machine-readable [CITATION.cff](CITATION.cff) (GitHub "Cite this repository"). Contributions welcome, especially sparse-dense matmul optimizations — by contributing you agree your code is licensed under the same AGPL-3.0-or-later.
