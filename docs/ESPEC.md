# M³-AVM Specification

```text
Status:   Informational Draft (research prototype, not a standard)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258 — <https://github.com/matheuscamarques/m3_avm>
Date:     2026-09-10
Version:  ISA v1.10 (95 opcodes: 0x00-0x43 except 0x39-0x3B, + 0x60-0x69,0x6A-0x79,0x7A-0x7C + 0xFF implemented; cumulative since v1.9)
License:  AGPL-3.0-or-later (see LICENSE; Section 15)
Replaces: all documents under docs/arq/ (archived, non-normative)
```

## Abstract

This document is the single normative specification of the M³-AVM
(Abstract Machine of Matheus de Camargo Marques), a software virtual
machine in Rust exploring preemptible, sparse, event-driven tensor
computation for interactive AI. It defines the instruction encoding,
the opcode map, the memory model, the execution and scheduling model,
the preemption/rollback discipline, the numerical contracts, the
reserved cluster and universal-AI extensions, and the formal
verification status. Anything this document marks as a goal without a
measurement is a target, not a claim (see Section 1.3).

## Table of Contents

1. Introduction
2. Conventions and Terminology
3. Architecture Overview
4. Instruction Encoding
5. Instruction Set
6. Memory Model
7. Execution, Scheduling and Events
8. Preemption and Rollback
9. Determinism and Reproducibility
10. Numerical Contracts
11. Real-Time Audio
12. Cluster Extension (Reserved)
13. Universal-AI Extension (Reserved)
14. PersonaPlex/Moshi Profile
15. Formal Verification
16. Implementation Status
17. Measured Benchmarks
18. Security Considerations
19. Evolution and Versioning
20. References
A. Appendix: Mnemonic Quick Reference

---

## 1. Introduction

### 1.1 Scope

M³-AVM is a fixed-width instruction-set emulator with first-class
sparse tensors, quantized GEMV, a strict-priority scheduler, and
microsecond-scale abort/rollback of inference state. It runs
Transformer text models from GGUF files today; Mamba recurrence,
neural audio codec frames, hybrid pipelines, distributed actors, and
universal-AI operators are implemented, reserved, or planned exactly as
stated per opcode in Section 5.

### 1.2 Goals and non-goals

Goals: surgical correction of in-flight reasoning (interrupt, roll
back to the exact point of error, inject a correction, resume);
per-row-chunk preemption of long kernels; sparse and quantized
tensors as ISA-level primitives; bit-exact replay of committed
prefixes.

Non-goals: replacing production inference engines; training
(gradients exist only as documented lowerings); bypassing the Von
Neumann bottleneck; optical or wafer-scale hardware claims. The Vega 8
`wgpu` path accelerates only `ATTN <= 64x64`; the hot path is CPU.

### 1.3 Honesty contract

Normative keywords (Section 2) apply only to statements tagged
`[NORMATIVE]` or to the reserved encodings in Sections 12-13.
Performance figures are `measured` (host, command and commit cited) or
`target` (design goal, explicitly labeled). No absolute figure in this
document is part of any correctness theorem; latency theorems prove
the *absence of terms* in N or in parameter count, never a microsecond
value (Sections 8, 10).

## 2. Conventions and Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in RFC 2119.

Definitions:

- `Heap`: a dense (`Vec<u8>`) or sparse (CSR) tensor store.
- `Snapshot`: an immutable, version-identified capture of VM state.
- `Rollback`: replacing current state with a snapshot (pointer swap,
  never a recompute).
- `Fence`: a `CTX_SWITCH` memory fence plus scheduler re-evaluation.
- `Deadline`: an absolute logical-time bound attached to a blockable
  context (Section 7, Section 12).
- `WAL`: write-ahead log retained at a migration source until the
  destination acknowledges (Section 6.5, Section 12).
- `Window (W)`: recalibration period bounding hybrid error (Section 10).
- `Tail mass (tau)`: softmax probability mass outside the top-T set
  (Section 10).

Byte order on the wire and in payloads is little-endian.

## 3. Architecture Overview

```text
 [Audio PCM / Text prompt]
            |  SENSE (0x06)
            v
 +-------------------------------+
 | Fetch: fixed 32-byte instr    |
 | Preemption check (atomic flag)|
 +-------+-----------------------+
         |               |
    interrupt?       dispatch
     YES |               | NO
         v               v
   ABORT/SSM_RESET   Engines (shared memory regions):
   (rollback)          Codec (audio) | Mamba O(1) | Transformer O(N)
                                     +-- CTX_SWITCH fence --+
                                         v
                                   SAMPLE/STREAM output
```

Engines share the four memory regions (Section 6). A BEAM-inspired
cluster bus (Section 12) injects remote `SIGNAL` into the same atomic
preemption flag. Full diagram and engine topology: archived
`docs/arq/ARCHITECTURE.md` (informative only).

## 4. Instruction Encoding

Every instruction is exactly 32 bytes (`INSTR_SIZE = 32`,
`src/opcodes.rs`):

```text
 offset 0: opcode (1 byte)
 offset 1: flags  (1 byte)
 offset 2: rdest  (1 byte, 0..15, 0xFF = unused)
 offset 3: rsrc1  (1 byte)
 offset 4: rsrc2  (1 byte)
 offset 5: rsrc3  (1 byte)
 offset 6..31: payload (26 bytes)
```

Rules [NORMATIVE]:

1. Decoders MUST reject instructions shorter than 32 bytes.
2. Register fields MUST be `0..15` or `0xFF`; any other value is a
   decode error.
3. Payload integers and floats are little-endian; offsets per opcode
   are fixed in Section 5.
4. Encodings, once assigned, are NEVER reused. Names without an
   encoding are assembler aliases resolved to a canonical opcode
   (Section 13.3).
5. The assembler is 2-pass with labels (`LOOP:`, `JUMP LOOP`,
   `FORK Rd, LABEL`).

NOTE: an older 64-bit word sketch (`docs/arq/chat.md`) is superseded
by this section and MUST NOT be implemented.

## 5. Instruction Set

Status per opcode: `IMPL` (implemented), `RSVD` (encoding frozen,
emulator rejects with an explicit error), `PLAN` (no encoding yet).

### 5.1 Control, memory and thinking loop (`0x00-0x0F`) — IMPL

| Hex  | Mnemonic      | Operands (assembler)              | Effect |
|------|---------------|-----------------------------------|--------|
| 0x00 | `HALT`        | `HALT`                            | Terminate program. |
| 0x01 | `TENSOR`      | `TENSOR Rd rows cols dtype [SPARSE DENSITY=x] [PERSIST]` | Allocate dense tensor in GLOBAL, sparse CSR if flagged, or GGUF-backed persistent view. |
| 0x02 | `ATTN`        | `ATTN Rd, Q, K, V [NOTIFY_EACH_HEAD] [MASK] [BLOCK_SIZE=n]` | Scaled attention; sparse-aware dispatch; KV-cache backed when K/V live in `0x30`. |
| 0x03 | `STREAM`      | `STREAM Rsrc, Rsink [BLOCKING\|DROP]` | Move data with backpressure; sinks include `SAMPLE`/`DECODED`/`PERIPHERAL_OUTPUT`. |
| 0x04 | `FORK`        | `FORK Rd, LABEL [RED\|BLUE\|GREEN] [NOTIFY]` | Clone context copy-on-write; child starts at label; snapshot pushed. |
| 0x05 | `ABORT`       | `ABORT Rs_ctx, Rs_ts`             | Remove target context; restore its snapshot (Section 8). |
| 0x06 | `SENSE`       | `SENSE Rd, PERIPHERAL [NON_BLOCKING]` | Read peripheral into TEMPORAL; `USER_INPUT` sets the interrupt flag; modes `AUDIO_PCM`/`CODEC_FRAME` feed the codec. |
| 0x07 | `NORM`        | `NORM Rd, Rsrc, Rgamma, Rbeta`    | RMSNorm/LayerNorm `x/sqrt(mean(x^2)+eps)*gamma+beta`. |
| 0x08 | `FFN`         | `FFN Rd, Rsrc, Rw1, Rw2 [, Rb1, Rb2]` | SwiGLU feed-forward. |
| 0x09 | `EMBED`       | `EMBED Rd, Rtoken, Rtable`        | Row `token_id % rows` of `[vocab, hidden]`. |
| 0x0A | `ADD`         | `ADD Rd, R1, R2`                  | Elementwise add (residuals). |
| 0x0B | `SAMPLE`      | `SAMPLE Rd, Rlogits [TEMP=x]`     | Softmax + sampling; writes `Rd` and `last_sample`. A future `TOPK=k` mode writes indices (Section 13). |
| 0x0C | `COMPARE`     | `COMPARE R1, R2\|imm\|EOS_TOKEN`  | Set `cmp_equal`. |
| 0x0D | `IF_EQUAL`    | `IF_EQUAL LABEL`                  | Branch if `cmp_equal`. |
| 0x0E | `JUMP`        | `JUMP LABEL`                      | Unconditional branch (target validated). |
| 0x0F | `IF_INTERRUPT`| `IF_INTERRUPT [Rcond,] LABEL`     | Branch if `reg != 0`, else if the interrupt flag is set (flag consumed). |

### 5.2 GEMV blocks (`0x10-0x12`) — IMPL

| Hex  | Mnemonic | Operands | Effect |
|------|----------|----------|--------|
| 0x10 | `MATVEC` | `MATVEC Rd, Rx, Rw` | `y = x.W` (`faer`/AVX2; quantized fused path when weights are `Q4_K`). A future `TRANSPOSE` flag (no new opcode) is reserved for local backward use. |
| 0x11 | `MUL`    | `MUL Rd, R1, R2`    | Elementwise multiply. |
| 0x12 | `SILU`   | `SILU Rd, Rsrc`     | `x.sigmoid(x)`. With `MATVEC`+`MUL` composes SwiGLU/heads. |

### 5.3 Mamba, codec and hybrid control (`0x13-0x19`) — IMPL

| Hex  | Mnemonic     | Operands | Payload / flags |
|------|--------------|----------|-----------------|
| 0x13 | `SSM_SCAN`   | `SSM_SCAN rY,rX,rH,rP [D_INNER=n D_STATE=n LAYER=n]` | `rY=[1,di]` out; `rX=[1,di]`; `rH=[di,ds]` in-place or `_` = `Vm::ssm_states[layer]`; `rP` = packed `dt+A+B+C+D` (build with `ssm::pack_params`: `dt[di]+A[di*ds]+B[ds]+C[ds]+D[di]` f32 LE) or `_` = defaults `dt=1,A=-1,B=C=1,D=0`. `payload[0..2]=di u16, [2..4]=ds u16, [4]=layer`. `CONV`/`GATE` flags are rejected (conv via `MATVEC`, gate via `SILU+MUL`). |
| 0x14 | `SSM_RESET`  | `SSM_RESET rH [D_INNER=n D_STATE=n LAYER=n]` | Zero/restore `h_t` (tensor or `ssm_states[layer]`). Rollback pair of `SSM_SCAN`. |
| 0x15 | `CODEC_ENC`  | `CODEC_ENC rD,rS [TENSOR]` | PCM `1920xf32` (80 ms @ 24 kHz, tensor or `TEMPORAL` bytes) to 16 Mimi codes (32 B / `[1,16]` tensor). |
| 0x16 | `CODEC_DEC`  | `CODEC_DEC rD,rS [TENSOR]` | Inverse of `CODEC_ENC`. |
| 0x17 | `AUDIO_ALIGN`| `AUDIO_ALIGN rD,rU,rA [SR=.. FRAME=.. HZ=.. DELAY=..]` | `rD=[t_user,t_ai,delta,frame]`; `payload = sr u32, spf u32, hz*100 u32, delay*100 u32`. Byte timestamps accepted ONLY from `TEMPORAL`, never `GLOBAL`. |
| 0x18 | `CTX_SWITCH` | `CTX_SWITCH MAMBA\|TRANSFORMER\|AUDIO [,RED\|BLUE\|GREEN]` | `payload[0]=pipe (0/1/2), [1]=0xA5 magic`; sets pipeline + priority + memory fence + snapshot + `maybe_preempt()`. |
| 0x19 | `ROPE`       | `ROPE rD,rS POS=n HDIM=n NHEADS=n [THETA=n] [INPLACE]` | Native rotary embedding per `[nheads*hdim]` block; `pos=0` is identity; odd `hdim` is an error. |

Semantics: `SSM_SCAN` computes `h = h.exp(dt.A)+x.B.dt; y = h.C+D.x`
(`src/ssm.rs`). `FORK` pushes an `ssm_states` snapshot; `ABORT` pops it,
mirroring KV truncation. `--emit-asm` emits `ROPE` after Q/K `MATVEC`
and `SSM_SCAN`+`CTX_SWITCH MAMBA` when `arch` contains `mamba`.

### 5.4 Cluster (`0x1A-0x1D`) — IMPL-local (RFC-0018, Section 12)

`0x1A REMOTE_SPAWN`, `0x1B SIGNAL`, `0x1C SEND_TENSOR`, `0x1D BARRIER`.
Payload layouts frozen to the 26-byte budget; `node_id == 0` executes
inline (spawn/kill/ping/copy+move/barrier), nonzero node fails loudly
— wire format, routing table and driver remain F2+.

### 5.5 Universal AI, wave 1 (`0x1E-0x21`) — PARTIAL (Section 13)

`0x1E CONV` (RSVD), `0x1F GATHER` (**IMPL**, RFC-0004),
`0x20 SPIKE_STEP` (RSVD), `0x21 DENOISE_STEP` (RSVD).

### 5.6 Universal AI, wave 2 (`0x22-0x25`) — PARTIAL (Section 13)

`0x22 FOREST` (RSVD), `0x23 DISTANCE` (**IMPL**, RFC-0004),
`0x24 RANK1_UPDATE` (**IMPL**, RFC-0004), `0x25 ODE_STEP` (RSVD).

### 5.7 `0xFF NOP` — IMPL

No operation (scheduler/IP bench target).

### 5.8 Free range

`0x39-0x3B`, `0x44-0x79` and `0x7D-0xFE` are free, except `0x38
KV_TRUNCATE` (IMPL, RFC-0010). `0x26-0x2F` are fully allocated and IMPL
(RFC-0023/0024, incl. `0x2E SLICE` from RFC-0019); `0x30-0x37` allocated
and IMPL (RFC-0027); `0x3C-0x43` allocated and IMPL (RFC-0028);
`0x7A-0x7C` allocated and IMPL (RFC-0026). Allocation REQUIRES a
proposal following the stateful-opcode rule (Section 19).

## 6. Memory Model

### 6.1 Address space

A 128-bit virtual space, region tag in the top byte
(`src/memory.rs`):

| Region      | Tag  | Logical | Physical (dev) | Purpose |
|-------------|------|---------|----------------|---------|
| GLOBAL      | 0x00 | —       | heap          | Immutable tensors, program code, shared weights. |
| TEMPORAL    | 0x10 | 1 GiB   | 64 MiB circular | Sensory I/O (prompts, PCM, codes, timestamps). |
| PERSISTENT  | 0x20 | 64 MiB* | `mmap` file `m3_persistent.dat` | Survives restarts (simulated PCM/ReRAM); GGUF weights mapped read-only with `madvise WillNeed`, exposed zero-copy. |
| KV_CACHE    | 0x30 | 1 GiB   | 64 MiB        | Per-layer sequential K/V buffers, `MAX_SEQ 2048`, append + truncate. |

`*` configurable (`new_with_size`, `persistent_mib`).

### 6.2 Tensor representation

- Dense: `Vec<u8>` + `TensorMeta`, cost `rows.cols.4`, 64-byte aligned.
- Sparse: CSR `CsrMatrix<f32>` (`nalgebra-sparse`/`sprs`), cost
  `nnz.4 + (rows+1).4 + nnz.4` (~`2.rho` of dense for large `r`).
  CSR pays off for projections/FFN; attention densifies at softmax
  (Section 10) — no sparse FlashAttention exists.
- Quantized: `Q4_K` super-block 256 weights / 144 B (~0.56 B/weight,
  7.1x denser than f32); dequant `Q4_0/Q4_K/Q6_K/Q8_0` (`src/quant.rs`)
  and fused AVX2 `matvec_q4k` (`src/matvec_quant.rs`). No native
  `f16/i8` compute; values are cast to f32.

### 6.3 Versioned snapshots [NORMATIVE]

State is `(counter, snaps, cur)` per store:

- `GLOBAL` heap clones, sparse heap clones, `KV_CACHE` layers, KV byte
  heap — one snapshot map each (`src/memory.rs`).
- `Vm::ssm_states` uses a push-on-`FORK` / pop-on-`ABORT` stack
  (`src/vm.rs`).

Discipline [NORMATIVE]:

1. **I-Persist.** After a snapshot, kernels MUST NOT mutate any tensor
   reachable from it. Status: IMPL — audit (RFC-0016) proves the
   discipline holds via `Arc::make_mut` CoW (dense heaps), deep clones
   (sparse/KV/SSM), and CoW rebinding (SNN/RANK1 tensors); the single
   hole (unsnapshotted `tensor_meta`) closed with the fifth snapshot
   map; `TEMPORAL` deliberately excluded (input staging, not model
   state). The `Arc::clone`-only caveat is retired.
2. **I-Mono.** The version counter MUST be strictly monotonic;
   `restore` MUST NOT lower it. Status: IMPL (`src/memory.rs::restore`
   preserves the counter since restoreFix; arbiters
   `qa::snapshot_restore_monotonic_no_clobber` green; the old
   `version = v` behavior is documented only as the machine-checked
   counterexample `formal/Formal/Rollback.lean` `clobber_demo`, with the
   fixed variant `restoreFix` + `fresh_of_inv` verified in the same file).
3. Snapshots MUST be retained under an explicit policy. Status: IMPL —
   sliding window `DEFAULT_SNAPSHOT_WINDOW = 16` (`set_snapshot_window`;
   `0` = explicit unbounded opt-out), evicted oldest-first across all
   four maps, clean `Err` on evicted restore (RFC-0003).

Under I-Persist + I-Mono, `restore(snapshot(s)) = s` bit-exact and
successive ids are strictly increasing (T1, Section 8; proofs in
`formal/Formal/Rollback.lean`).

### 6.4 Stateful-engine taxonomy [NORMATIVE]

Every stateful opcode MUST declare (a) where its state lives,
(b) snapshot cost, (c) how `ABORT` restores it. No new stateful opcode
enters the ISA without this.

- FAMILY 1 STATELESS (preemption = drop chunk): FFN, `CONV`, aggregated
  `GATHER`, MoE gate, cache-less `ATTN`.
- FAMILY 2 STATEFUL (preemption = `ABORT` + O(1) pointer rollback):
  Transformer `KV_CACHE 0x30`, Mamba `ssm_states`, SNN `V(t)`, RNN
  cell, diffusion `x_t`, DeltaNet/Titans `H_t`.

### 6.5 Migration log [NORMATIVE]

Any tensor migration MUST be logged at the source (WAL) and retained
until destination ACK plus a retention window (Section 12, T5).
`MOVE = COPY + origin invalidation without retention` is FORBIDDEN:
it loses data if the source dies between invalidation and ACK
(counterexample `formal/Formal/ClusterWAL.lean moveWithoutWal_loses`).

## 7. Execution, Scheduling and Events

- Context: 16 `u128` registers (`R0` always zero per thesis; emulator
  `src/context.rs`), `pc: u128`, `root_version`, `priority`,
  `cmp_equal`, `interrupt_flag`, `pipeline`, `deadline`.
- Scheduler: strict `Red > Blue > Green`, three local FIFOs
  (`src/context.rs`). `Red` = interrupts/`ABORT`/`SENSE`;
  `Blue` = audio I/O; `Green` = heavy inference.
- Buses (`src/bus.rs`, Tokio, NOT silicon): `watch` interrupt channel
  (`SENSE`/`ABORT` to `ATTN` heads), `broadcast` stream channel
  (backpressure), `broadcast` scheduler channel (`FORK` events).
  An optional `Reactor` (`src/reactor.rs`) consumes them via
  `tokio::select!`; interrupt checks also happen inline per chunk.
- Direction [NORMATIVE for cluster]: every blockable context carries an
  absolute deadline; scheduling is earliest-deadline-first with
  priority inheritance (a consumer blocked on a remote producer
  inherits the tighter deadline). Rationale: without inheritance, a
  local `RED` holding a queue while waiting on a remote `BARRIER`
  deadlocks silently. All waits MUST have timeouts (Section 12).

## 8. Preemption and Rollback

Mechanism: `SENSE` (explicit peripheral poll or `AUDIO_PCM` frame)
publishes `InterruptSignal { ctx_id, timestamp, payload }`
(`payload = { new_prompt?, target_token_index? }`); chunk-granular
checks in `ATTN`/`SSM_SCAN` divert to `ABORT` + `restore` + prompt
injection at the checkpoint (`Theta' = Theta[0:eta] ++
tokenize(prompt') ++ Theta[eta:]`).

Theorems (conditional on Section 6.3):

- **T1 (exact rollback).** Defined in Section 6.3; verified in Lean.
  Required test: 100-step rollout, `ABORT` at step 50, 50 more steps
  MUST equal the uninterrupted rollout bit-exactly.
- **T2 (bounded preemption).** `L_abort = t_detect + t_sched +
  t_restore` contains NO term in N or in parameter count
  (`t_restore` swaps roots/registers only). Absolute values are
  measurements (Section 17), never part of the theorem. Distributed
  case adds one LAN RTT plus urgent-queue priority; REQUIRES the
  deadline discipline of Section 7.

## 9. Determinism and Reproducibility

Single-seed execution is deterministic: in-VM sampling (`SAMPLE`,
`STREAM`-to-`SAMPLE`) draws from the calling context's splitmix64 stream
(RFC-0009), so same seed + same logits replays exactly, including across
VMs. Host/REPL sampling stays thread-random by design (a human is in the
loop); `SENSE` stimuli stay environmental randomness. Timestamps use `std::time::Instant` (unsynchronized across
nodes — wall-clock is metric-only; ordering uses Lamport clocks,
Section 12). Bit-exactness is same-host, same-build only: FP32/fused
kernels may differ across x86/ARM/GPU targets, so cross-hardware replay
is NOT claimed (blind spot 13, `docs/arq/PLANO-PONTOS-CEGOS.md`). No rate limiting on `SENSE` exists yet (flood can thrash
the reactor — open gap).

## 10. Numerical Contracts

- **SSM (Mamba-1).** Continuous `h' = A.h + B.x, y = C.h + D.x` with
  fixed `A < 0`, `dt > 0` from `softplus`. Code implements explicit
  Euler: `a_d = exp(dt.A)` (contraction: `0 < a_d < 1`), `h <- h.a_d
  + x.B.dt`, `y = <h,C> + D.x`. Local error vs exact ZOH is `O(dt^2)`
  with explicit constant under `|A|.dt <= 1`; homogeneous recurrence
  forgets as `|h_n| <= a^n.|h_0|`; one layer-step costs `Theta(I.S)`
  with `I.S + I.d_conv` state, independent of N (e.g. `d=2048`:
  ~320 KiB/layer, ~7 MiB for 22 layers). Verified in
  `formal/Formal/SSM.lean`.
- **Hybrid crossover.** Dense attention `Theta(N^2.d)` per layer vs SSM
  scan `Theta(N.I.S)`; KV-cache `Theta(L.N.d)` vs SSM state
  `Theta(L.I.S)`. Above `N_0 = Theta(I.S/d)` (tens of tokens at
  `I.S/d = 32`) the quadratic term dominates — the region where
  rollback/preemption pays.
- **Quantization (Q4_K).** Per-vector error `|<e,x>| <=
  (s_max/2).||x||_1 <= (s_max/2).sqrt(n).||x||_2`. Perplexity impact is
  NOT covered — empirical bench required.
- **T3 (windowed hybrid bound) [NORMATIVE direction].** Unwindowed
  composition diverges as `O(N.(C1.dt^2 + C2.eps_q))`. With a
  `CTX_SWITCH` fence + Transformer recalibration every `W` steps,
  global error is `<= r + W.e` independent of N (contraction forbids
  exponential amplification inside the window). Arithmetic verified in
  `formal/Formal/WindowedError.lean`; physical instantiation of `e`
  is an open obligation. A Kalman-fusion upgrade is future work, not a
  premise.
- **T4 (top-k attention) [NORMATIVE direction].** If softmax tail mass
  `tau < eps`, pruned renormalized attention errs by `<=
  2.eps/(1-eps)` in L1; cost drops to `O(N.d + T^2.d)`. Requires fast
  QK decay AND `T << N`; otherwise pruning is decorative. Statement in
  `formal/Formal/TopK.lean` (proof obligation); `DISTANCE 0x23` fused
  top-k MUST feed pruned `ATTN` (never the reverse).

## 11. Real-Time Audio

`SENSE_AUDIO_PCM` delivers 24 kHz PCM in 80 ms windows (1920 f32).
**T6:** if the `CODEC_ENC -> SSM_SCAN -> AUDIO_ALIGN` pipeline
processes one window in `T_proc < 80 ms`, the queue never grows
(backlog <= 1) and barge-in detection latency is `<= 80 ms + T_proc`,
independent of conversation length. `CODEC_ENC` compresses 1920 f32 to
16 codes (120x); per-window SSM state is O(1), so `T_proc` is encoder-
dominated and bench-testable per 80 ms window. Cluster case adds one
RTT under the Section 7 deadline discipline.

## 12. Cluster Extension (Reserved) [NORMATIVE encodings]

Opcodes `0x1A-0x1D`; instruction stays 32 bytes; bulk travels outside
it in 64 KiB chunks; `NODE="name"` resolves via the node routing table.

```text
0x1A REMOTE_SPAWN  payload[0..4]=node_id u32, [4..12]=entry_pc u64,
                   [12]=prio (0=GREEN,1=BLUE,2=RED). Rd = remote ctx_id
                   or u128::MAX on failure.
0x1B SIGNAL        flags bit0 = highest priority. rsrc1 = kind register
                   (0=ABORT,1=FORK_REQ,2=HALT/KILL,3=PING).
                   payload[0..4]=node_id u32, [4..12]=ctx_id u64,
                   [12..20]=Lamport seq u64, [20..26] reserved.
                   Rich prompts MUST NOT travel in SIGNAL (use prior
                   SEND_TENSOR/side channel); the signal stays 32 B.
0x1C SEND_TENSOR   flags bit0 = 0 TCP / 1 RDMA-hint (MVP always TCP).
                   rdest = local source reg, rsrc1 = remote dest reg
                   (0xFF = allocate). payload[0..4]=node_id u32,
                   [4..12]=byte_offset u64, [12..16]=byte_len u32,
                   [16]=mode (0=COPY,1=MOVE). MOVE obeys Section 6.5.
0x1D BARRIER       payload[0..4]=barrier_id u32, [4..6]=expected u16,
                   [6..8]=timeout_ms u16, [8..12]=epoch/mask. Blocks the
                   context (wakeup, never scheduler spin); timeout yields
                   NACK + error unlock. Coordinator counts ARRIVE and
                   broadcasts RELEASE.
```

Runtime rules [NORMATIVE]: static mesh of 2..N nodes (`--peer`; no
gossip in MVP); `node_id u32` + `GlobalPid(node_id, ctx_id)`;
Erlang-style cookie (`--cookie`/`M3_COOKIE`, constant-time compare) at
handshake; frame `total_len BE | 32 B instr | bulk_len | bulk`;
timeouts `connect 2s, ack 500ms, heartbeat 200ms, suspect 600ms, dead
2s`; urgent (`SIGNAL`/`BARRIER`/`HEARTBEAT`) overtakes bulk
(`SEND_TENSOR`/`SPAWN`); heartbeat carries `free_pct + queue lengths`
for spawn placement; dead nodes leave routing and `SPAWN` retries the
next live node. **T5:** under Section 6.5 and single-node fail-stop, a
migrating tensor exists on >= 1 live node (case proof skeleton in
`formal/Formal/ClusterWAL.lean`). Double-simultaneous death is out of
scope (needs 3-way replication). Reference milestones: spec (F0),
local ISA without network (F1), driver + distributed barge-in `<5 ms`
loopback with `M3_PROFILE=1` (F2), spawn/barrier/supervision (F3),
tensor migration with per-chunk hash under `tc delay/loss` (F4),
hardening/benches/honest docs (F5). Order `F0->...->F5`; F2 MUST NOT
be skipped. TCP-Tokio suffices for a preprint (QUIC/RDMA reserved by
flags; TLS mutual is follow-up).

## 13. Universal-AI Extension (Reserved) [NORMATIVE encodings]

Design rule: native ONLY where lowering over existing opcodes is
inadequate (im2col-`MATVEC` memory blowup, missing tensor-indirect
indexing, per-element masked stores, vectorized tree walk, fused
top-k retrieval, outer-product state, fused ODE steps). RNN/LSTM/GRU,
RWKV scans, MoE gates, Titans TTT steps, fractal reductions, SVMs,
PCA/SVD solvers are documented lowerings, NOT new opcodes.

### 13.1 Wave 1 (`0x1E-0x21`)

```text
0x1E CONV         rdest=out [N,Cout,..], rsrc1=input, rsrc2=kernel
                 [Cout,Cin,K..], rsrc3=bias|0xFF.
                 payload[0..2]=stride u16, [2..4]=pad u16,
                 [4..6]=dilation u16, [6]=groups u8,
                 [7]=fused_act (0=none,1=silu,2=relu). N-dim, depthwise
                 from day one; direct sliding (no materialized im2col).
0x1F GATHER       rdest=out, rsrc1=table, rsrc2=u32/i64 indices,
                 rsrc3=0xFF (gather) | accumulator (scatter).
                 payload[0]=axis u8, [1]=mode
                 (0=GATHER,1=SCATTER_ADD,2=SCATTER_MAX),
                 [2..4]=nnz_hint u16. OOB is an explicit error, never wrap.
0x20 SPIKE_STEP   rdest=spikes [1,n] u8, rsrc1=V(t) membrane (in-place),
                 rsrc2=input current, rsrc3=weights/decay-pack|0xFF.
                 payload[0..4]=V_threshold f32, [4..8]=decay f32,
                 [8..12]=V_reset f32, [12]=layer_id,
                 [13]=refractory_steps u8. CoW like ssm_states.
0x21 DENOISE_STEP rdest=x_{t-1}, rsrc1=x_t, rsrc2=predicted noise,
                 rsrc3=schedule/t-embed|0xFF.
                 payload[0..4]=alpha_bar_t f32, [4..8]=beta_t f32,
                 [8..12]=sigma_t f32 (0=deterministic/DDIM),
                 [12..16]=timestep u32. Seeded VM RNG (replay-exact).
```

Order `G0->G1(GATHER)->G2(SPIKE)->G3(CONV)->G4(DENOISE)->G5(harden)`;
MoE top-k arrives as a `SAMPLE` mode, not an opcode.

### 13.2 Wave 2 (`0x22-0x25`) and flags

```text
0x22 FOREST       rdest=scores [1,n_trees]|aggregate, rsrc1=features [1,F],
                 rsrc2=flat node table, rsrc3=leaf values.
                 payload[0..2]=n_trees u16, [2..4]=max_depth u16,
                 [4]=mode (0=vote/sum,1=mean prob). Branchless walk.
0x23 DISTANCE     rdest=dists [1,N] (+indices when top-k),
                 rsrc1=query [1,D], rsrc2=bank [N,D].
                 payload[0]=metric (0=EUCLID,1=COSINE,2=MANHATTAN,3=DOT),
                 [1..3]=topk u16 (0=all). Fused partial-heap top-k.
0x24 RANK1_UPDATE rdest=H in-place [d,k] (matrix state), rsrc1=v [d],
                 rsrc2=k [k], rsrc3=0xFF|mask. payload[0..4]=alpha f32,
                 [4..8]=beta f32, [8]=mode (0=hebbian,1=delta,2=forget),
                 [9]=layer_id. H = alpha.H + beta.v k^T. CoW like ssm.
0x25 ODE_STEP     rdest=x(t+dt), rsrc1=x(t), rsrc2=u(t) params,
                 rsrc3=field weights|0xFF. payload[0..4]=dt f32,
                 [4]=method (0=Euler,1=RK2,2=RK4), [5]=layer_id.
```

New flags on existing opcodes (no new encodings): `MATVEC.TRANSPOSE`
(direct `W^T.x`, needed for local Titans backward) and
`SAMPLE.TOPK=k` (writes indices; serves MoE gate and standalone use).

### 13.3 Aliases (never encoded)

`MATRIX_RECURRENCE` -> `RANK1_UPDATE`, `LOOKUP_TREE` -> `FOREST`
(`TREES=1`), `ODE_SOLVER` -> `ODE_STEP`, `SCATTER_ADD` -> `GATHER
MODE=SCATTER_ADD`. Domain map: (1) dense matrix `MATVEC/MUL/CONV`;
(2) state/recurrence `ATTN/SSM_SCAN/RANK1_UPDATE/ODE_STEP/SPIKE_STEP`;
(3) control/flow `FORK/ABORT/SENSE/STREAM/COMPARE/JUMP/IF_*/SAMPLE/
CTX_SWITCH` + cluster `0x1A-0x1D`; (4) audio/codec `CODEC_ENC/
CODEC_DEC/AUDIO_ALIGN` + `SENSE PCM/CODEC_FRAME` + `ROPE`;
(5) classical/retrieval `FOREST/DISTANCE/GATHER/DENOISE_STEP`.

## 14. PersonaPlex/Moshi Profile

Target: PersonaPlex-7B (Kyutai Moshi base), `temporal 32x4096x32 +
depformer 6x1024x16 + Mimi 16 codebooks @12.5 Hz/24 kHz`, 17 streams
(1 text + 8 user + 8 agent), 80 ms frames, ~160 ms theoretical delay.
Reference quantization ~5 GB: temporal `Q4_K` (~4.4 GB incl. shards),
embeddings `Q4_0`, depformer/Mimi fp16. Tensor-name maps (GGUF
`blk.*` vs HF safetensors) and phase plan F0-F5 (map, loader
`src/moshi.rs`, `src/mimi.rs`, quantized temporal+depformer with
17-stream KV + voice conditioning, full-duplex `moshi_loop.m3asm`,
INT4 optimization) live in archived `docs/arq/PLANO_MOSHI_NATIVO.md`
and `docs/arq/MOSHI_MAP.md`. Demos: `programs/moshi_loop_v2.m3asm`,
`programs/codec_loop.m3asm`, `programs/mamba_scan_demo.m3asm`.

## 15. Formal Verification

| Claim | Artifact | Status |
|-------|----------|--------|
| SSM contraction, Euler `O(dt^2)`, forgetting, linear scan cost | `formal/Formal/SSM.lean` | Proved |
| Rollback roundtrip + monotonic-counter fix + clobber counterexample | `formal/Formal/Rollback.lean` | Proved; fix live in `src/memory.rs::restore` (arbiters green, §16) |
| Windowed hybrid bound arithmetic | `formal/Formal/WindowedError.lean` | Proved; physical instantiation open |
| Top-k statement + uniform no-gain lemma | `formal/Formal/TopK.lean` | Skeleton (`sorry` on main bound) |
| Migration conservation cases | `formal/Formal/ClusterWAL.lean` | Proved (model-level) |

`lake build` in `formal/` MUST stay green. No claim in Sections 8-12
outruns the Status column above.

## 16. Implementation Status

| Component | Status | Evidence |
|-----------|--------|----------|
| 32-byte fetch/dispatch, `0x01-0x0F` | IMPL | `src/opcodes.rs`, `src/vm.rs`, `programs/control_flow_demo.m3asm` |
| `0x10-0x12`, GGUF `mmap` inference, `KV_CACHE` + snapshot/rollback | IMPL | `src/inference.rs`, `src/matvec_quant.rs`, `src/memory.rs` |
| I-Mono (monotonic version counter, `restoreFix`) | IMPL | `src/memory.rs::restore` (no counter lowering); arbiters `qa::snapshot_restore_monotonic_no_clobber`, `qa::rollback_100_50_50_bit_exact`; model `formal/Formal/Rollback.lean` |
| `0x1F GATHER` + `0x23 DISTANCE` + `0x24 RANK1_UPDATE` + `SAMPLE TOPK` | IMPL | RFC-0004; goldens `vm::test_gather_golden_and_oob`, `test_distance_four_metrics_and_topk`, `test_rank1_modes_and_rollback`, `test_sample_topk_indices`; demos `gather_moe_demo`, `rag_search_demo`, `deltanet_demo` (`attn_topk_demo` removido: packed precisa de `SLICE`) |
| I-Persist (post-snapshot immutability) | IMPL | RFC-0016; make_mut/deep-clone/CoW audit + meta map + visibility proofs; TEMPORAL excluded by design |
| `0x60-0x66` RNG + hash (determinism block) | IMPL | RFC-0005; vectors NIST/FNV/CRC + fork-inheritance goldens; demo `rng_demo` |
| `0x6A-0x77` telemetry + scheduler | IMPL | RFC-0006; NaN-absorb/poll/trap/deadline/prio/try-lock goldens; demo `telemetry_demo` |
| `0x78 LOADI` + `0x79 MOV` + `COMPARE PRED=` | IMPL | RFC-0007; max-u128/chain/6-predicate goldens; demo `loadi_demo` |
| Strict assembler (unknown-token rejection) | IMPL | RFC-0008; 26 bad forms rejected, corpus gate 27/27 |
| Seeded sampling (`SAMPLE` via context RNG) | IMPL | RFC-0009; two-VM agreement + reseed replay; host paths explicitly non-deterministic |
| `0x38 KV_TRUNCATE` (rolling KV hygiene) | IMPL | RFC-0010; shrink/no-op/rollback/stream-gate; demo `kv_truncate_demo` |
| `0x21 DENOISE_STEP` (fused diffusion) | IMPL | RFC-0013; manual vector, seeded replay + no-draw, param gate; demo `denoise_demo` (exec verified) |
| `0x25 ODE_STEP` (fused Euler/RK) | IMPL | RFC-0014; Euler manual, RK4 vs fine-Euler, contraction, param gate; demo `ode_demo` (exec verified) |
| `0x20 SPIKE_STEP` (LIF integrate-and-fire) | IMPL | RFC-0015; 4-step refractory golden, pack precedence, rollback; demo `snn_demo` (exec verified) |
| `TENSOR FILL` + `0x2E SLICE` (literal data paths) | IMPL | RFC-0019; fill-vs-ramp, packed split, integration chain; demos `forest_demo` + `attn_topk_demo` ressuscitados (exec verificada) |
| `0x1E CONV` (sliding 1D/2D + groups) | IMPL | RFC-0017; edge/diagonal/depthwise/dilation goldens, error paths; demo `conv_demo` (exec verified) |
| Version-tagged engine stacks + host pruning | IMPL | RFC-0011; nested gating, no-op abort, double-abort stability, `prune_oldest` |
| `0x22 FOREST` (vectorized tree walk) | IMPL | RFC-0012; 2-tree golden, OOB/NaN/depth-bound, strict ramp-trap (demos `.m3asm` inviáveis sem TENSOR-INIT — ver RFC) |
| `0x1A-0x1D` cluster local (spawn/signal/send/barrier) | IMPL | RFC-0018; local-only, transport vetado; `remove_tensor`; demo `cluster_local_demo` (contadores exatos) |
| `0x13-0x19` wiring + tests | IMPL | `src/vm.rs` exec fns, `src/ssm.rs`, `src/mimi.rs`, `src/moshi.rs`, demos |
| Local barge-in (`SENSE`+`IF_INTERRUPT`, `ABORT`) | IMPL (local) | `src/vm.rs`, `src/rollback.rs`, `src/bus.rs` |
| `0x1A-0x1D` transport | ROADMAP | No network transport in tree |
| `0x1E-0x25` operators (minus GATHER/DISTANCE/RANK1) | ROADMAP | Rejected explicitly until specified |
| GPU path | PARTIAL | `ATTN<=64` only; GEMV stays CPU |
| Benches for `0x13-0x19` | IMPL | `benches/hybrid_ops_bench.rs` (7 ops + 80 ms window, §17) |
| `0x26/0x27 ARENA` + `0x2A/0x2B MEMCPY/MEMSET` (memory core) | IMPL | RFC-0023; bump+O(1) reset, bit-exact/OOB/RO/dir goldens, FORK-snapshot/ABORT-restore, demo `arena_memcpy_demo` (exec verified) |
| `0x28/0x29 SNAPSHOT/RESTORE` + `0x2C/0x2D/0x2F PREFETCH/RESHAPE/CONCAT` (views & versions) | IMPL | RFC-0024; named-version roundtrip, engine rewind, N-D concat/order goldens, demo `snap_concat_demo` (exec verified); v1.6 |
| `0x67/0x68/0x69 CAST/QUANTIZE/DEQUANT` (conversion) | IMPL | RFC-0025; explicit pairs/RNE/clamp goldens, Q4_0/Q8_0 error-bounded encoders, dispatcher-backed dequant, demo `cast_quant_demo` (exec verified); v1.7 |
| `0x7A/0x7B/0x7C ADD_IMM/SUB_IMM/STEPS` (control-plane ALU) | IMPL | RFC-0026; wrapping goldens, deterministic STEPS count, demo `alu_demo` (exec verified); v1.8 |
| `0x30-0x37 SORT/TOPK/ARGMAX/REDUCE/BROADCAST/PAD/TILE/TRANSPOSE` (shape) | IMPL | RFC-0027; total-order/NaN/tie goldens, N-D concat-style lanes, demo `shape_demo` (exec verified); v1.9 |
| `0x3C-0x43 SOFTMAX/GELU/SIGMOID/TANH/RELU/EXP/LOG/CLIP` (activations) | IMPL | RFC-0028; stable-softmax/temp goldens, exact-erf GELU, NaN table, demo `activation_demo` (exec verified); v1.10 |
| End-to-end Mamba GGUF smoke | OPEN | — |

`cargo test --lib`: 256 green + 4 RFC-0019 arbiters at last report (ISA, sparse, bus,
reactor, inference, asm_emitter, TUI, VM). One pre-existing failure
unrelated to this spec: `moshi::test_gguf_qkv_split_shapes` (norm-gamma
assertion on the local PersonaPlex GGUF; fails identically on the pristine
tree — model-data issue, not a spec regression).

## 17. Measured Benchmarks

Measured on the author's host (Ryzen 3500U unless noted); debug values
are emulator overhead, not silicon claims:

| Scenario | Result |
|----------|--------|
| `ATTN` dense 32x32 | ~5 ms/iter |
| `ATTN` sparse 32x32 5% + `NOTIFY_EACH_HEAD` | ~5 ms + overhead < 3x |
| `sparse_nop.m3asm` (6x 32x32) | 45 ms debug |
| Scheduler IPS | 240k debug / 2.1M release per 1M NOPs |
| `FORK` CoW (100x 1 MiB) | < 1 ms (`Arc` clone) |
| `ABORT` 32x32 under load | 25 ms measured (HW 50 us hypothetical); publish < 1 ms |
| CSR 64x64 @ 5% | ~10x smaller than dense |
| DeepSeek 1.5B `Q4_K` forward | ~1.2 s/token (0.8 tok/s) |
| TinyLlama 1.1B `Q4_K` forward | ~0.33 s/token (3 tok/s; +10% with `znver1`) |

Hybrid ops (`benches/hybrid_ops_bench.rs`, criterion medians, bench
profile, Ryzen 5 3500U / 8 threads, rustc 1.98.1 — per-instruction via
`step_instruction`, output alloc included):

| Op | Result (median) |
|----|-----------------|
| `SSM_SCAN` di=64, ds=16 | ~25.9 µs |
| `SSM_RESET` | ~0.87 µs |
| `CODEC_ENC` frame 1920 | ~10.2 µs |
| `CODEC_DEC` frame | ~74.6 µs |
| `AUDIO_ALIGN` | ~2.6 µs |
| `CTX_SWITCH` fence | ~1.5 µs |
| `ROPE` 4096 (32×128) pos=7 | ~21.3 µs |
| **Audio window 80 ms** (`ENC→SCAN→ALIGN`) | **~29.7 µs** (`T_proc`; T6 needs `< 80 ms` — headroom ~2700× on this host) |

Caveats: `--quick` runs on shared hosts inflate `AUDIO_ALIGN`/
`CTX_SWITCH` ~5× (noise; full runs above are stable). Feeding an
oversized x (e.g. `[1,1920]`) to a `di=16` scan pays the full 1920-float
read before truncation — representative inputs used above; partial-read
optimization is an open follow-up.

## 18. Security Considerations

- Cluster cookie (`M3_COOKIE`) is comparison-only auth; mutual TLS
  (Noise) is future work — MUST NOT expose cluster ports to untrusted
  networks.
- No gossip/peer authentication beyond the cookie; no rate limiting on
  `SENSE`; BARRIER particip
ation assumes non-Byzantine peers.
- Hosted inference of modified versions MUST offer Corresponding
  Source (AGPL Section 13).
- Resource caps: snapshot retention window (Section 6.3), `MAX_SEQ
  2048`, bounded `mpsc` channels are REQUIRED mitigations against
  memory/flood exhaustion; the `SENSE`-flood gap is open.

## 19. Evolution and Versioning

- `ISA v1.5` = everything in §5 marked IMPL (cumulative since v1.4:
  RFC-0004 through RFC-0018; intermediate minors were not cut).
  Minor bump per new opcode family from here on; existing encodings
  are immutable.
- `ISA v1.6` = v1.5 + memory/arena family `0x26-0x2F` complete
  (RFC-0023: `ARENA_ALLOC/RESET`, `MEMCPY/MEMSET`; RFC-0024:
  `SNAPSHOT/RESTORE`, `PREFETCH/RESHAPE/CONCAT`).
- `ISA v1.7` = v1.6 + conversion family `0x67-0x69` (RFC-0025:
  `CAST`, `QUANTIZE`, `DEQUANT`; Precision Rule enforced on 32B).
- `ISA v1.8` = v1.7 + control-plane ALU `0x7A-0x7C` (RFC-0026:
  `ADD_IMM`, `SUB_IMM`, `STEPS`; `0x7D-0x7F` stay RESERVED).
- `ISA v1.9` = v1.8 + shape family `0x30-0x37` (RFC-0027: `SORT`,
  `TOPK`, `ARGMAX`, `REDUCE`, `BROADCAST`, `PAD`, `TILE`, `TRANSPOSE`).
- `ISA v1.10` = v1.9 + activation family `0x3C-0x43` (RFC-0028:
  `SOFTMAX`, `GELU`, `SIGMOID`, `TANH`, `RELU`, `EXP`, `LOG`, `CLIP`).
  The 1.x line continues past 1.9 as 1.10+; `v2.x` stays reserved for
  the dual-mode freeze (never "the minor after 1.9").
- Draft v2.0 proposal (reconciled, non-normative): `docs/ESPEC-V2.md`
  (DRAFT — do not code against it; reservations in Sections 12-13 of
  this document remain the only binding future encodings).
- `0x26-0xFE` allocate ONLY via proposal satisfying the Section 6.4
  stateful-opcode rule.
- The 64-bit word sketch and any conflicting archived text are
  superseded by Sections 4-6 of this document.

## 20. References

- Code: `src/opcodes.rs` (encoding/assembler), `src/vm.rs`
  (dispatch), `src/memory.rs` (regions/snapshots), `src/context.rs`
  (scheduler), `src/bus.rs`/`src/reactor.rs` (events),
  `src/inference.rs`/`src/gguf.rs`/`src/quant.rs`/`src/matvec_quant.rs`
  (weights), `src/ssm.rs`/`src/mimi.rs`/`src/moshi.rs` (hybrid),
  `src/sparse.rs` (CSR), `src/rollback.rs` (entropy fallback).
- Demos: `examples/minimal.m3asm`, `examples/sparse_nop.m3asm`,
  `programs/control_flow_demo.m3asm`, `programs/thinking_sample.m3asm`,
  `programs/mamba_scan_demo.m3asm`, `programs/codec_loop.m3asm`,
  `programs/moshi_loop_v2.m3asm`.
- Proofs: `formal/` (Section 15).
- Archive (informative, superseded on conflict): `docs/arq/`.
- Citation: `CITATION.cff` (`@software{m3_avm_2026, ...}`,
  AGPL-3.0-or-later).

## Appendix A. Mnemonic Quick Reference

```text
TENSOR ATTN STREAM FORK ABORT SENSE NORM FFN EMBED ADD SAMPLE
COMPARE IF_EQUAL JUMP IF_INTERRUPT MATVEC MUL SILU
SSM_SCAN SSM_RESET CODEC_ENC CODEC_DEC AUDIO_ALIGN CTX_SWITCH ROPE
HALT NOP
(RSVD) REMOTE_SPAWN SIGNAL SEND_TENSOR BARRIER
(RSVD) CONV GATHER SPIKE_STEP DENOISE_STEP
(RSVD) FOREST DISTANCE RANK1_UPDATE ODE_STEP
```

```text
Canonical loop (thinking):
  SENSE r11, TOKEN / FORK r12, GREEN, NOTIFY / NORM / ATTN /
  ADD / FFN / ADD / SAMPLE / STREAM / COMPARE / IF_EQUAL / JUMP
```

```text
Barge-in:
  SENSE USER_INPUT -> IF_INTERRUPT HANDLE_ABORT ->
  ABORT (+ SSM pop / KV truncate / optional SSM_RESET) -> resume
```
