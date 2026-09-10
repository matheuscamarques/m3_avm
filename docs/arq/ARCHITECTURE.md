# M³-AVM Architecture Overview

> **Author: Matheus de Camargo Marques** · https://github.com/matheuscamarques/m3_avm
> Determinism, Microsecond Rollback & Distributed Heterogeneous AI Virtual Machine.
>
> **Reading guide (honesty contract):** the diagram below is the *target architecture*.
> The [status table](#status-implemented-vs-vision) marks what is already implemented
> in this repo, what is experimental, and what is roadmap. Numbers cited as measured
> come from `cargo test`/`cargo bench` on the author's host (see `README.md` §5).
> The mathematical viability models and proofs are in [MATH_MVP.md](MATH_MVP.md).

```text
==================================================================================================
                                    M³-AVM ARCHITECTURE OVERVIEW
      (Determinism, Microsecond Rollback & Distributed Heterogeneous AI Virtual Machine)
==================================================================================================

  [ USER / HARDWARE INTERFACES ]
       │
       ├───────────────────────────────┐
       ▼                               ▼
┌──────────────┐             ┌──────────────────┐
│ Audio Mic /  │             │ Text Prompt /    │
│ PCM Stream   │             │ Token Stream     │
└──────┬───────┘             └─────────┬────────┘
       │                               │
       │ SENSE_AUDIO_PCM (0x06)        │ SENSE_USER_INPUT (0x06)
       ▼                               ▼
==================================================================================================
                        M³-AVM EXECUTION ENGINE & DISPATCHER (Rust)
==================================================================================================
   ┌──────────────────────────────────────────────────────────────────────────────────────────┐
   │ 32-BYTE FIXED INSTRUCTION FETCH & PREFETCH PIPELINE                                      │
   │ [ Opcode (1B) | Flags (1B) | Rdest (1B) | Rsrc1 (1B) | Rsrc2 (1B) | Rsrc3 (1B) | Payload (26B) ]│
   └──────────────────────────────┬───────────────────────────────────────────────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │  ATOMIC PREEMPTION & BARGE-IN DETECTOR│ <──── [ SIGNAL / NETWORK ]
               └──────────────────┬───────────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │ Interrupt Flag == True?       │
                  ├───────────────┬───────────────┤
                  │     YES       │      NO       │
                  └───────┬───────┴───────┬───────┘
                          │               │
    ┌─────────────────────┘               └──────────────────────┐
    ▼                                                            ▼
┌─────────────────────────┐                   ┌──────────────────────────────────────┐
│  OP_ABORT / OP_SSM_RESET│                   │         INSTRUCTION DISPATCHER       │
│ (~217µs Rollback / CoW) │                   └──────────────────┬───────────────────┘
└───────────┬─────────────┘                                      │
            │                                ┌───────────────────┼───────────────────┐
            │ Fast-Forward / Reset           │                   │                   │
            ▼                                ▼                   ▼                   ▼
┌─────────────────────────┐       ┌──────────────────┐┌──────────────────┐┌──────────────────┐
│ STATE & MEMORY REGIONS  │       │ Neural Codec Engine││ Mamba Engine     ││ Transformer Engine│
│ ┌─────────────────────┐ │       │ (PersonaPlex Audio)││ (State Space O(1))││ (KV Cache O(N))  │
│ │ TEMPORAL (0x0000...)│ │       ├──────────────────┤├──────────────────┤├──────────────────┤
│ ├─────────────────────┤ │       │  OP_CODEC_ENC    ││  OP_SSM_SCAN     ││  OP_ATTN (Flash) │
│ │ GLOBAL   (0x1000...)│ │ <──── │  OP_CODEC_DEC    ││  OP_SSM_RESET    ││  OP_FFN (SwiGLU) │
│ ├─────────────────────┤ │       │  OP_AUDIO_ALIGN  ││  State Vector ht ││  OP_ROPE / NORM  │
│ │ PERSISTENT (0x20...)│ │       └─────────┬────────┘└─────────┬────────┘└─────────┬────────┘
│ └─────────────────────┘ │                 │                   │                   │
└─────────────────────────┘                 └───────────────────┼───────────────────┘
                                                                │
                                                                ▼
                                              ┌──────────────────────────────────────┐
                                              │ OP_STREAM / LOGIT SAMPLER (0x0B/0x03) │
                                              └──────────────────┬───────────────────┘
                                                                 │
                                                                 ▼
                                              ┌──────────────────────────────────────┐
                                              │ PERIPHERAL OUTPUT DECODED / AUDIO    │
                                              └──────────────────────────────────────┘

==================================================================================================
                     DISTRIBUTED NETWORK LAYER (BEAM-Inspired Cluster Engine)
==================================================================================================
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │       NETWORK DISPATCHER (Rust)       │
                             │        (Async Mio / Tokio Loop)       │
                             └───┬───────────────┬───────────────┬───┘
                                 │               │               │
            ┌────────────────────┘               │               └────────────────────┐
            ▼                                    ▼                                    ▼
┌───────────────────────┐            ┌───────────────────────┐            ┌───────────────────────┐
│     OP_REMOTE_SPAWN   │            │       OP_SIGNAL       │            │     OP_SEND_TENSOR    │
│       (0x1A)          │            │        (0x1B)         │            │         (0x1C)        │
│ Spawns remote AVM ctx │            │ Sub-ms Out-of-Band    │            │ Zero-Copy RDMA/Socket │
│ on worker nodes       │            │ ABORT / Interrupts    │            │ KV Cache/State Transfer│
└───────────┬───────────┘            └───────────┬───────────┘            └───────────┬───────────┘
            │                                    │                                    │
            └────────────────────────────────────┼────────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │    CLUSTER BUS (QUIC / TCP / RDMA)    │
                             └───────────────────┬───────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             ┌─────────────────────┐                           ┌─────────────────────┐
             │   REMOTE NODE A     │                           │   REMOTE NODE B     │
             │ (Compute / Mamba)   │                           │ (Compute / Transf.) │
             └─────────────────────┘                           └─────────────────────┘
```

> Note: the fourth network opcode `OP_BARRIER` (`0x1D`, deterministic barrier/RELEASE across nodes) rides the same cluster bus; full spec in `docs/PLANO_AVM_CLUSTER.md` §2.

## Technical highlights

1. **ISA topology (32-byte fixed):** deterministic fetch and prefetch at the top of
   the instruction pipeline (`src/opcodes.rs`, `INSTR_SIZE=32`).
2. **Interrupt & barge-in loop:** if the atomic interrupt flag fires via local
   `SENSE` (or, in the future, `OP_SIGNAL` from the network), the dispatcher
   diverts straight to `OP_ABORT` / `OP_SSM_RESET` without waiting for the end
   of the generation block.
3. **Heterogeneous engines:** the three engines (neural codec for audio, Mamba
   for `O(1)` state, Transformer for heavy attention) read and write the same
   unified memory regions (`TEMPORAL`, `GLOBAL`, `PERSISTENT`, plus `KV_CACHE`).
4. **BEAM-style cluster:** the network bus injects `OP_SIGNAL` directly into the
   remote node's preemption flag, targeting distributed sub-millisecond interrupt.

## Status: implemented vs. vision

| Diagram element | Status | Evidence |
| :--- | :--- | :--- |
| 32-byte fetch/dispatcher, `TENSOR/ATTN/STREAM/FORK/ABORT/SENSE` (`0x01–0x06`) | ✅ Implemented | `src/opcodes.rs`, `src/vm.rs` |
| `NORM/FFN/EMBED/ADD/SAMPLE` + control flow (`0x07–0x0F`) | ✅ Implemented | `src/vm.rs`, `programs/control_flow_demo.m3asm` |
| `MATVEC/MUL/SILU`, GGUF `mmap` inference, `KV_CACHE` + snapshot/rollback | ✅ Implemented | `src/inference.rs`, `src/matvec_quant.rs`, `src/memory.rs` |
| `SSM_SCAN/SSM_RESET/CODEC_ENC/CODEC_DEC/AUDIO_ALIGN/CTX_SWITCH/ROPE` (`0x13–0x19`) | ✅ Implemented | `src/opcodes.rs`, `src/vm.rs` (`exec_ssm_scan/reset`, `exec_codec_enc/dec`, `exec_audio_align/ctx_switch/rope`) + `src/ssm.rs`, `src/mimi.rs`, `src/moshi.rs`; spec `docs/ISA_OPCODES_0x13_0x19.md`; demos `programs/mamba_scan_demo.m3asm`, `programs/codec_loop.m3asm`, `programs/moshi_loop_v2.m3asm` |
| Barge-in via `SENSE` + `IF_INTERRUPT`, `ABORT` rollback | ✅ Implemented (local) | `src/vm.rs`, `src/rollback.rs`, `src/bus.rs` |
| `REMOTE_SPAWN/SIGNAL/SEND_TENSOR/BARRIER` (`0x1A–0x1D`), network dispatcher, cluster bus (QUIC/TCP/RDMA) | 🔭 Roadmap (not implemented) | Reserved opcode range; no network transport in this repo; spec in `docs/PLANO_AVM_CLUSTER.md` |
| Latency figures in diagram (`~217µs`, sub-ms distributed) | 🎯 Target | Local measured baselines in `README.md` §5; distributed figures are design goals, not measurements |

## Planning / roadmap

- [x] Core ISA + scheduler + sparse/quant backends + real GGUF inference (done — see `README.md`).
- [x] VM wiring for `0x13–0x19` (SSM/codec/audio-align/ctx-switch/ROPE) with tests (166 lib green) — benches still open.
- [ ] Benches (`criterion`) for `0x13–0x19` + end-to-end smoke with a real Mamba GGUF.
- [ ] `GEMV.wgsl` resident kernels for the GPU path (`src/inference_gpu.rs`).
- [ ] Specify `0x1A–0x1D` wire format (envelope, addressing, auth) before any network code.
- [ ] Universal-AI opcodes `0x1E–0x25` (`CONV/GATHER/SPIKE_STEP/DENOISE_STEP` + `FOREST/DISTANCE/RANK1_UPDATE/ODE_STEP`) — plan in `docs/PLANO_ISA_UNIVERSAL.md`.
- [ ] Reference cluster transport (Tokio/QUIC first, RDMA later) + distributed preemption demo.
- [ ] Publish ISA spec as `docs/ISA.md` + whitepaper so the architecture above is citable.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
