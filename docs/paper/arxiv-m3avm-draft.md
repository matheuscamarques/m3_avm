# M³-AVM: A Sparse Event-Driven Tensor Virtual Machine with Deterministic Rollback for Interactive AI (draft for arXiv cs.DC)

**Author:** Matheus de Camargo Marques — ORCID https://orcid.org/0009-0003-4518-2258 — matheuscamarques@gmail.com — Independent Researcher
**Code:** https://github.com/matheuscamarques/m3_avm — License: AGPL-3.0-or-later — **Status:** research prototype, not production.
**Spec:** `docs/ESPEC.md` (normative, ISA v1.4) + `docs/ESPEC-V2.md` (draft). **Convert:** `pandoc docs/paper/arxiv-m3avm-draft.md -o /tmp/m3avm.pdf --pdf-engine=xelatex`.

---

## Abstract

Interactive AI (voice assistants with barge-in, agentic tree search, speculative decoding) needs fine-grained preemption and exact state rollback, but current stacks launch GPU kernels to completion and isolate models in separate processes behind serialization. We present M³-AVM, a software virtual machine in Rust that moves these concerns into a fixed 32-byte instruction set (ISA v1.4): first-class sparse (CSR) and quantized (Q4_K, fused AVX2) tensors, a strict-priority scheduler (Red > Blue > Green) with absolute per-context deadlines, and copy-on-write snapshot/rollback (`FORK`/`ABORT`) across Transformer KV cache, SSM recurrent state, and matrix-recurrence state with a monotonic version counter. The VM runs Transformer text models from GGUF via zero-copy `mmap` plus per-layer KV cache, with Mamba/SSM recurrence and a neural audio codec in the same address space, switchable without context switch (`CTX_SWITCH`). Cluster transport, 64-byte forms, composite deadlines (AND/OR/N_OF_M/CHAIN), device-aware EDF, and cross-model KV transfer are frozen/reserved future work, explicitly rejected at runtime. Measured on Ryzen 3500U: ATTN 32×32 ~5 ms, `FORK` 100×1 MiB <1 ms, `ABORT` ~25 ms debug (event publish <1 ms), TinyLlama-1.1B Q4_K ~0.33 s/token; `cargo test --lib` 228 passed / 1 model-data failure. Lean 4 proofs cover SSM contraction/Euler error, rollback roundtrip with the monotonic-counter fix, and windowed hybrid error arithmetic. Contribution: an open, citable reference showing where ISA-level preemption/rollback pays and where it does not.

**Keywords:** virtual machine, instruction set architecture, AI inference, sparse computation, deterministic rollback, KV cache, Mamba SSM.

## 1. Introduction

Serving stacks (Triton, Ray Serve, vLLM) optimize single-model throughput: paged KV management, continuous batching, disaggregated prefill/decode. Interaction primitives — interrupt an in-flight generation, roll back to the exact point of error, inject a correction, resume — live in application code, subject to allocator jitter, garbage collection, and inter-process copies. Sandbox checkpointing (DeltaBox: overlayfs + CRIU, ~11 ms hidden / ~2 ms fork-restore) works at OS/process granularity, not tensor granularity; KV-transfer work (cross-model ridge mapping, CacheBridge) studies the math, not runtime integration with rollback.

M³-AVM asks: what if preemption, sparse/quantized tensors, deadlines, and rollback were ISA primitives of a small VM, with honest measurements and machine-checked core lemmas? This paper reports the prototype, its limits, and what remains future work. Nothing here replaces production engines; the hot path stays CPU-bound (`faer`/AVX2, ~3 tok/s TinyLlama on the author's APU).

## 2. Related work

**Serving:** NVIDIA Triton (per-model schedulers, sequence batching, HTTP/gRPC/C API); Ray Serve / KServe / Seldon (per-service timeouts — Seldon documents per-node, not end-to-end, semantics); vLLM + PagedAttention (SOSP'23: paged KV blocks, block tables, CoW sharing within/across requests of the *same* model, 2–4× throughput). **Checkpoints:** DeltaBox (arXiv:2605.22781: DeltaFS layer insert / O(1) switch + DeltaCR template-fork restore). **Scheduling:** classic EDF/PIP/PCP/SRP/DFP theory; PHAROS (arXiv:2604.05308: EDF/FIFO on spatially-partitioned heterogeneous accelerators with schedulability-guided DSE — hardware design, no boolean deadline composition over media flows); Kubernetes 1.36 topology-aware scheduling / Run:ai (placement, not deadline inheritance). **KV reuse:** intra-model (prefix caching, cross-layer sharing); cross-model ridge transfer (arXiv:2608.03893), CacheBridge (2609.00891), cross-family sharing (2608.30963) — math without ISA/rollback integration. **Compression/eviction:** PyramidInfer (arXiv:2405.12532), FlexGen offload, SnapKV-family — single-model VRAM scope.

M³-AVM differs in combination: heterogeneous engines (Transformer + SSM + codec + retrieval/matrix primitives) in one address space, ISA-level CoW snapshot/restore across them, per-context absolute deadlines with strict priorities, per-context seeded RNG for replay — with cluster/composite/topology/transfer explicitly deferred.

## 3. Architecture (what is implemented)

**Machine model.** One process, 128-bit virtual space, 4 regions by high byte: GLOBAL 0x00 (immutable tensors/code), TEMPORAL 0x10 (1 GiB logical / 64 MiB physical-dev circular sensory buffer), PERSISTENT 0x20 (`mmap` file, GGUF weights read-only zero-copy with `madvise WillNeed`), KV_CACHE 0x30 (per-layer sequential K/V, MAX_SEQ 2048, append + truncate). Dense (`Vec<u8>` + meta, 64 B aligned), sparse CSR (`nalgebra-sparse`/`sprs`), quantized weights (Q4_0/Q4_K/Q6_K/Q8_0 dequant + fused AVX2 `matvec_q4k`; no native f16/i8 compute). Context: 16 `u128` regs, PC, version root, priority, `cmp_equal`, `interrupt_flag`, pipeline, deadline. Events: Tokio `watch` (interrupt) + `broadcast` (stream/scheduler) — microseconds measured, not silicon; optional reactor consumes via `select!`, plus inline per-chunk checks.

**Engines, one memory.** Transformer (`ATTN` sparse-aware dispatch, `FFN` SwiGLU, `NORM`, `ROPE`, `MATVEC/MUL/SILU`, `EMBED/ADD/SAMPLE`), SSM (`SSM_SCAN`: `h←h·exp(dt·A)+x·B·dt`, `y=⟨h,C⟩+D·x`; `SSM_RESET`), audio (`CODEC_ENC/DEC`: 1920 f32 ⇄ 16 codes; `AUDIO_ALIGN`), retrieval/matrix (`GATHER` + scatter modes with OOB trap, `DISTANCE` 4 metrics + fused top-k, `RANK1_UPDATE` CoW with 3 modes, `SAMPLE TOPK`, `FOREST` branchless walk — RFC-0004/0012, `DENOISE_STEP`, `ODE_STEP`), control (`FORK/ABORT/SENSE/STREAM/COMPARE/JUMP/IF_*`), determinism (`RNG_SEED/NEXT/NORMAL/UNIFORM`, `HASH/CHECKSUM/HMAC`), telemetry/scheduler (`CYCLES_COUNT/TRACE_EVENT/SANITY_CHECK/PREEMPT_CHECK/ASSERT/DUMP/YIELD/SET_DEADLINE/GET_DEADLINE/PRIORITY_SET/GET/LOCK/UNLOCK/FENCE`), immediates (`LOADI/MOV`, extended COMPARE predicates), hygiene (`KV_TRUNCATE`), `HALT/NOP`. Full list with payload layouts: `docs/ESPEC.md §5`.

**Instruction format.** Fixed 32 bytes: opcode 1 B | flags 1 B (legacy per-opcode) | 4 reg bytes | 26 B LE payload. Decoder rejects short instructions and bad regs; encodings immutable; 2-pass assembler with labels. A 64-bit-word sketch in the archive is superseded and must not be implemented. The 64-byte form (canonical flags, LAMPORT/DEADLINE/PAYLOAD_EXT + 32 B core) is draft-only (`ESPEC-V2 §4`); a v1.x decoder rejects `op ≥ 0x80` with `UnsupportedWidth`.

**Real inference.** `ModelConfig::from_gguf` (hidden/intermediate/layers/heads/vocab/arch for Qwen2/Llama/Mistral); `forward_one`: RMSNorm → Q/K/V matvec (zero-copy raw slice → AVX2 Q4_K → `faer` fallback; row-slice embeddings) → RoPE → KV append → per-head softmax → O proj → residual → FFN gate/up/down; `AsmEmitter` lowers the same config to unrolled `.m3asm` with per-layer `SENSE/IF_INTERRUPT`. `wgpu` accelerates only `ATTN ≤ 64×64`; GEMV stays CPU.

## 4. Preemption and rollback (conditional theorems)

**Mechanism.** `SENSE` publishes `InterruptSignal`; chunk-granular checks in `ATTN`/`SSM_SCAN` divert to `ABORT` + restore + prompt injection at the checkpoint. `FORK` pushes CoW snapshots (`Arc` clones per store + `ssm_states` + `rank1_layers` stacks + KV layers); `ABORT` pops/restores by pointer swap. Retention: sliding window k=16 (RFC-0003, `0` = explicit unbounded opt-out), oldest-first eviction, clean error on evicted restore. Discipline: **I-Mono** (counter never lowered on restore — implemented, arbiters green) and **I-Persist** (no mutation of snapshot-reachable tensors — required, *not yet enforced*, declared open).

**Theorems (conditional on the discipline; proofs in `formal/`).** T1 exact rollback: `restore(snapshot(s)) = s` bit-exact; required test 100-step rollout, abort at 50, +50 equals uninterrupted. T2 bounded preemption: `L_abort = t_detect + t_sched + t_restore` has no term in sequence length or parameter count (root/register swap only); absolute values are measurements (§6), never part of the theorem. Verified: SSM contraction/Euler O(dt²)/forgetting/linear-scan cost; rollback roundtrip + clobber counterexample for the old counter-lowering variant; windowed hybrid error arithmetic (physical instantiation open); TopK statement skeleton; cluster-WAL case skeleton.

**Determinism.** In-VM sampling draws from the calling context's splitmix64 stream (RFC-0009): same seed + same logits replays exactly, including across VMs; host/REPL sampling stays thread-random; `SENSE` is environmental; timestamps are metric-only (`Instant`, unsynchronized; ordering uses Lamport clocks in the cluster design). Bit-exactness is same-host/same-build (FP32/fused kernels may differ across targets).

## 5. Scheduler and deadlines (implemented vs reserved)

Implemented: three FIFOs, strict `Red > Blue > Green` (interrupts/abort/sense; audio I/O; heavy inference); `CTX_SWITCH` fence + re-evaluation; absolute per-context deadline ops; locks/fence/trace/sanity traps; `BARRIER`-with-timeout is reserved. Reserved direction (normative-draft): composite deadline trees (`AND/OR/N_OF_M/CHAIN`) with violation propagation, EDF with deadline inheritance (consumer blocked on a producer inherits the tighter deadline), device/NUMA affinity, urgent-over-bulk queues, heartbeat placement, SWIM states, WAL-retained migration — each wave needs its RFC + green tests + Lean delta before `DRAFT→IMPL`.

## 6. Measurements (not claims)

Author's host (Ryzen 3500U unless noted); debug values are emulator overhead: ATTN dense 32×32 ~5 ms/iter; sparse 5% + `NOTIFY_EACH_HEAD` ~5 ms + <3× overhead; `sparse_nop.m3asm` 45 ms debug; scheduler 240k IPS debug / 2.1M release per 1M NOPs; `FORK` CoW 100×1 MiB <1 ms; `ABORT` 32×32 ~25 ms measured (HW 50 µs hypothetical), event publish <1 ms; CSR 64×64@5% ~10× smaller; DeepSeek-1.5B Q4_K ~1.2 s/token, TinyLlama-1.1B Q4_K ~0.33 s/token (+10% with `znver1`). Pre-existing failure unrelated to the spec: `moshi::test_gguf_qkv_split_shapes` (local GGUF norm assertion; fails on pristine tree). No BSR/sparse-FlashAttention; attention densifies at softmax; no `SENSE` rate limiting yet.

## 7. Limitations (read before citing)

Single-host determinism only; I-Persist unenforced; context registers outside rollback (memory + handle maps only); no cluster transport in tree (`0x1A–0x1D` reject); `CONV/SPIKE` + most retrieval/classical/audio-full-duplex ops reserved; GPU path minimal; no autograd, no BSR, no resident GEMV shader. The multi-model voice scenario (`CENARIO-MULTIMODEL.md`) is archived input, explicitly non-compiling — the executable equivalents are the `*_demo.m3asm` programs.

## 8. Future work

Dual-mode 64-byte decoder + region table; I-Persist enforcement; pruned-attention fusion; telemetry-gated scheduling waves; retrieval wave with the lowering-inadequacy filter; Moshi full-duplex wave; cluster X-forms + WAL/SWIM; MMIO/system + `.m3bc` loader — in wave order with per-wave RFCs (`ESPEC-V2 §14`).

## References

- Kwon et al., PagedAttention/vLLM, SOSP'23. DOI:10.1145/3600006.3613165.
- DeltaBox, arXiv:2605.22781 (OS-level agent C/R).
- PHAROS, arXiv:2604.05308 (EDF heterogeneous accelerators).
- Heo et al., cross-model KV transfer, arXiv:2608.03893; CacheBridge, arXiv:2609.00891; cross-family sharing, arXiv:2608.30963.
- PyramidInfer, arXiv:2405.12532.
- NVIDIA Triton Inference Server docs (architecture/model execution); Seldon/KServe timeout docs; K8s topology-aware scheduling (KEP-5732) / Run:ai.
- M³-AVM spec: `docs/ESPEC.md` (v1.4, normative); `docs/ESPEC-V2.md` (draft); RFC-0002…0012; Lean proofs in `formal/Formal/`.

## Reproducibility

```bash
git clone https://github.com/matheuscamarques/m3_avm && cd m3_avm
cargo test --lib
cargo run --bin m3_avm -- run examples/sparse_nop.m3asm
cargo run --bin m3_avm -- run programs/control_flow_demo.m3asm
# real inference needs a local GGUF (not redistributed):
cargo run --bin m3_avm -- run --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --real --max-steps 3
```

*Cite as (CITATION.cff): Marques, M. de C. (2026). M³-AVM: Research Prototype for Sparse Event-Driven Tensor Computation (v0.1.0). AGPL-3.0-or-later. https://github.com/matheuscamarques/m3_avm — ORCID https://orcid.org/0009-0003-4518-2258.*
