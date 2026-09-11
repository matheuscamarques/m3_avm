# M³-AVM Specification v2.0 (Draft, Reconciled)

```text
Status:   DRAFT — non-normative proposal. NOT IMPLEMENTED.
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258 — <https://github.com/matheuscamarques/m3_avm>
Date:     2026-09-10
Extends:  docs/ESPEC.md (ISA v1.4 + model, normative)
Reconciles:
  [B] docs/arq/ISA-V2-BLUEPRINT.md (v2.0 blueprint + critique, archived)
  [R] docs/arq/RFC-0001-ESCAPE.md (extension mechanism, archived)
  [C] implemented code (src/opcodes.rs, src/memory.rs, src/context.rs, src/vm.rs)
License:  AGPL-3.0-or-later (see LICENSE)
```

## Abstract

This document reconciles three mutually contradictory sources — the
implemented v1.3 machine, the v2.0 blueprint [B], and the extension RFC
[R] — into ONE canonical 256-opcode map with no duplicates, no ghosts,
and no silent rewrites of shipped encodings. Every conflict gets a
numbered resolution (Section 2) with evidence. Until each item's status
reads IMPL, code MUST NOT be written against it; reservations alone are
binding (Section 11).

Honesty contract (from ESPEC Section 1.3) applies in full: anything
marked DRAFT/RSVD/HELD is a target or a held encoding, never a claim.

## Table of Contents

1. Introduction
2. Reconciliation Resolutions (R1-R12)
3. Canonical Opcode Map (256 entries)
4. Instruction Formats and Size Function
5. Flags: Canonical Byte and 32B Legacy Compat
6. Registers
7. Memory: Logical Regions, Compat Profile, Region Table
8. Bytecode Container and Feature Negotiation
9. ESCAPE Mechanism (Fixed)
10. Stateful Taxonomy (Extension of ESPEC 6.4)
11. Precision Rule
12. Cluster: Basic Forms, X-Forms, WAL Naming
13. Determinism
14. Unified Wave Plan
15. Open Obligations
A. Appendix: Source Cross-Reference

---

## 1. Introduction

### 1.1 Why this document exists

[B] proposes ~130 new opcodes, 16 regions, 64B instructions and 360
registers. [R] proposes an ESCAPE mechanism, a versioned container, R255
bank switching and a Region Table. Both contradict the shipped machine
(32B instructions, `u128` addresses, 16 registers, 4 regions) and each
other (Section 2). This document keeps every good idea, deletes every
duplicate, and demotes everything unimplemented to DRAFT with a named
path to acceptance.

### 1.2 Precedence

1. Shipped encodings and behaviors (ESPEC v1.3) are immutable.
2. Frozen-but-unimplemented encodings (`0x1A-0x25`) keep their exact
   meaning; v2.0 may only ADD width variants alongside, never replace.
3. Between [B] and [R], the extension mechanism ([R] Sections 3-7) wins
   over placeholder reservations ([B] Section 5.12) wherever they
   collide, because placeholders carry no semantics.
4. Anything not in Section 3 does not exist.

## 2. Reconciliation Resolutions

**R1 — Address width stays 128-bit.** Code uses `u128` addresses,
registers and PC (`src/memory.rs:71-86`, `src/context.rs:93`). [B]
Section 1.1 specifies 64-bit `REGION:OFFSET`. Resolution: the
architectural address is 128-bit; the 64-bit compact form is the
canonical serialization inside 64B payloads and zero-extends into
`u128`. No code changes; [B] Section 1.1 is reinterpreted, not adopted
literally.

**R2 — Width function with two carved exceptions.** [B] Section 2.1
(`>=0x80` implies 64B) contradicts its own Section 5.13 (`0xFF NOP` is
32B). Resolution: size rule is Section 4 (table-driven): `<0x80`
implies 32B; `0x80-0xAF`, `0xB4-0xB6` imply 64B; `0xB0-0xB3` are
variable-length (ext_len); `0xFF` is ALWAYS 32B (sole exception,
preserves every shipped test and binary).

**R3 — B-block split: ESCAPE wins, federated moves.** [R] Section 3
allocates `0xB0-0xB6`; [B] Section 5.12 reserves `0xB0-0xBF` for
federated learning. Resolution: `0xB0-0xB6` are the ESCAPE/capability
block (mechanism beats placeholder); the federated-future placeholder
moves to `0xB7-0xBF`. [B] Section 5.12 is superseded on this range only.

**R4 — Cluster duplicates deleted; X-forms defined, not migrated.**
[B] defines MULTICAST/MIGRATE/HEARTBEAT/PING/REDUCE_REMOTE/GATHER_REMOTE
twice (Sections 5.5 at `0x4A-0x4F` and 5.9 at `0x84-0x8B`) and claims
`0x1A-0x1D` "migrate" to 64B (breaking frozen encodings). Resolution:
(a) `0x1A-0x1D` stay canonical 32B basic forms FOREVER (equivalence,
not migration); (b) single extended home at `0x80-0x8F` (64B X-forms
plus genuinely new ops); (c) `0x4A-0x4F` is RESERVED; any code citing
[B] Section 5.5's cluster rows must use Section 3 of this document.
Equivalence rule: a 64B X-form with neutral ext fields
(`LAMPORT=0`, `DEADLINE=MAX`, `PAYLOAD_EXT=0`) is semantically EQUAL to
its 32B basic form (Section 12).

**R5 — Registers: 16 GPR normative; files deferred; R255 rejected.**
Code has 16 `u128` registers (`src/context.rs:93`); decoder rejects
other values except `0xFF` (`src/opcodes.rs:169-172`). [B] Section 4
(256 GPR + VEC + STATE files) and [R] Section 5.1 (R255) require an
encoding change, which [R] Section 6.1 itself classifies as a MAJOR
bump. Resolution: (a) 16 GPR stay normative through v2.x; (b) SPECIAL
names SP0-SP7 adopted with per-name status (Section 6);
(c) STATE handles S0-S63 adopted as LOGICAL namespaces encoded in
payload immediates (already true: layer_id, stream_id, kv handles) —
not file entries; (d) VEC file and GPR-256 deferred to v3.0 (MAJOR);
(e) R255/RFH rejected in current form, remediated as: bank selection
arrives only with the v3.0 file it would select.

**R6 — Regions: 16 logical, 4 implemented, one table.**
Code has 4 top-byte-tagged regions (`src/memory.rs:24-27`); [B] has 16
with a 4-bit selector. Resolution: the 16 regions (Section 7) are the
LOGICAL model and the 64B address model; the 4 implemented regions are
the "32B compatibility profile" with a fixed mapping (Section 7).
32B instructions address profile regions only; 64B instructions may
address all 16. [R] Section 5.2 Region Table adopted with fix R7: it
lives at MMIO+0x0 (resolving the MMIO-vs-table double booking of `0xF`),
`N_REGIONS==0` selects the 16 defaults.

**R7 — TEMPORAL ghost fixed.** [B] Sections 5.2/6.3 cite a `TEMPORAL`
region its own Section 1.2 table lacks. Resolution: implemented
TEMPORAL maps to logical `STREAM_RING (0x9)`; CODEC rows corrected
accordingly in Section 3.

**R8 — Flags: canonical for 64B, legacy kept for 32B.**
[B] Section 3 fixes the flag byte; code uses per-opcode flags.
Resolution: canonical assignment (Section 5) is normative for 64B;
32B keeps legacy flags with a compat table (Section 5). PRECISION bits
are ADVISORY kernel routing governed by Section 11 (trap, never silent
fallback) — this answers [B]'s "fake precision" charge without faking
precision.

**R9 — One MIGRATE, two widths, one WAL rule.** Canonical MIGRATE is
`0x84` (64B, full WAL id + checksum + HMAC per [B] Section 6.5, kept).
32B `SEND_TENSOR` keeps basic COPY/WAL_BACKED with the WAL-entry naming
rule of Section 12 (no new fields needed). Retention `5s`,
backpressure at 80% (from [B] Section 1.3, kept as normative targets).

**R10 — Container is additive; legacy sniffing rule added.**
[R] Section 4's `.m3bc` header cannot apply to headerless v1.3 `.m3bin`
([R] Section 8.1 as written is unimplementable). Resolution: loader
sniffs MAGIC; absent MAGIC implies legacy v1.x path (no negotiation,
no rejection). Negotiation applies to `.m3bc` only. Feature registry
imported from [R] Section 7 with all bits DRAFT (nothing implemented).

**R11 — Governance adopted.** [R] Sections 6 (lifecycle, no-reuse
rules) and 6.4 (RFC template) adopted, extending ESPEC Section 19:
every `0x26+` allocation and every B-block use REQUIRES an RFC with
that template. Reference-implementation links ([R] Section 11) are void
until the files exist.

**R12 — Classical-ML and exotic placeholders HELD, not planned.**
[B] Sections 5.6 (second half), 5.12 reserve ethics-free semantics for
dozens of ML-classical and exotic (photonic/quantum/TEE/federated)
entries. Resolution: encodings HELD; any implementation first needs a
lowering-inadequacy proof under the ESPEC Section 13 filter
("native only where lowering is inadequate"). Placeholders carry no
semantics and no wave assignment.

## 3. Canonical Opcode Map (256 entries)

Legend — W: `32` fixed 32B · `64` fixed 64B · `128/256/VAR` escape sizes.
Status: `IMPL` shipped · `RSVD` frozen, emulator rejects ·
`DRAFT` v2.0, do not emit · `HELD` encoding held pending proof/RFC.
State: `S` stateless · `F2` stateful Family 2 (declares per ESPEC 6.4).

### 3.1 Base: control, memory, thinking (`0x00-0x0F`) — IMPL, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x00` | `HALT` | 32 | IMPL | S | Terminates context |
| `0x01` | `TENSOR` | 32 | IMPL | S | GLOBAL/ACTIVATION; dims+dtype |
| `0x02` | `ATTN` | 32 | IMPL | F2 | KV_CACHE; sparse-aware; pruned mode DRAFT (T4) |
| `0x03` | `STREAM` | 32 | IMPL | S | STREAM_RING; backpressure |
| `0x04` | `FORK` | 32 | IMPL | S | SNAPSHOT push; prio flag (legacy, see §5) |
| `0x05` | `ABORT` | 32 | IMPL | S | SNAPSHOT restore; T1 conditional on I-Persist+I-Mono |
| `0x06` | `SENSE` | 32 | IMPL | S | STREAM_RING; spike-train mode DRAFT |
| `0x07` | `NORM` | 32 | IMPL | S | RMS/Layer; Batch mode DRAFT |
| `0x08` | `FFN` | 32 | IMPL | S | SwiGLU; FUSED_ACT DRAFT |
| `0x09` | `EMBED` | 32 | IMPL | S | WEIGHTS read |
| `0x0A` | `ADD` | 32 | IMPL | S | Elementwise |
| `0x0B` | `SAMPLE` | 32 | IMPL | S | Softmax+sample; TOPK mode DRAFT |
| `0x0C` | `COMPARE` | 32 | IMPL | S | Predicates |
| `0x0D` | `IF_EQUAL` | 32 | IMPL | S | Conditional branch |
| `0x0E` | `JUMP` | 32 | IMPL | S | TEXT; validated target |
| `0x0F` | `IF_INTERRUPT` | 32 | IMPL | S | Flag branch (consumes) |

### 3.2 GEMV + hybrid (`0x10-0x19`) — IMPL, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x10` | `MATVEC` | 32 | IMPL | S | faer/AVX2/Q4K; TRANSPOSE+FUSED_ACT DRAFT |
| `0x11` | `MUL` | 32 | IMPL | S | Elementwise |
| `0x12` | `SILU` | 32 | IMPL | S | Elementwise |
| `0x13` | `SSM_SCAN` | 32 | IMPL | F2 | KV_CACHE (ssm_states); PRECISION DRAFT |
| `0x14` | `SSM_RESET` | 32 | IMPL | F2 | Rollback pair of SCAN |
| `0x15` | `CODEC_ENC` | 32 | IMPL | S | STREAM_RING (R7); PRECISION-fp16 DRAFT |
| `0x16` | `CODEC_DEC` | 32 | IMPL | S | STREAM_RING (R7) |
| `0x17` | `AUDIO_ALIGN` | 32 | IMPL | S | Timestamps; TEMPORAL-only byte rule kept |
| `0x18` | `CTX_SWITCH` | 32 | IMPL | S | Pipeline+prio+fence; T3 recalibration fence |
| `0x19` | `ROPE` | 32 | IMPL | S | Rotary; pos=0 identity |

### 3.3 Cluster basic + universal (`0x1A-0x25`) — PARTIAL (RSVD unless marked IMPL), 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x1A` | `REMOTE_SPAWN` | 32 | IMPL | S | Local spawn at label (RFC-0018; remote vetado) |
| `0x1B` | `SIGNAL` | 32 | IMPL | S | Local ABORT/HALT/PING; FORK_REQ vetado (RFC-0018) |
| `0x1C` | `SEND_TENSOR` | 32 | IMPL | S | Local COPY/MOVE+invalidation (RFC-0018) |
| `0x1D` | `BARRIER` | 32 | IMPL | S | One-shot local + lazy timeout (RFC-0018) |
| `0x1E` | `CONV` | 32 | IMPL | S | Sliding 1D/2D + groups + fused (RFC-0017) |
| `0x1F` | `GATHER` | 32 | IMPL | S | +SCATTER modes; OOB traps (RFC-0004) |
| `0x20` | `SPIKE_STEP` | 32 | IMPL | F2 | KV_CACHE (V(t)); CoW (RFC-0015) |
| `0x21` | `DENOISE_STEP` | 32 | IMPL | F2 | ACTIVATION (x_t); seeded/DDIM (RFC-0013) |
| `0x22` | `FOREST` | 32 | IMPL | S | Branchless walk (RFC-0012; table spec §7 fixed) |
| `0x23` | `DISTANCE` | 32 | IMPL | S | 4 metrics + fused top-k; feeds pruned ATTN (RFC-0004) |
| `0x24` | `RANK1_UPDATE` | 32 | IMPL | F2 | KV_CACHE (H_t); CoW (RFC-0004) |
| `0x25` | `ODE_STEP` | 32 | IMPL | S | Stateless fused Euler/RK (RFC-0014) |

### 3.4 Memory and arena (`0x26-0x2F`) — DRAFT, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x26` | `ARENA_ALLOC` | 32 | IMPL | S | Vm-side bump; size+align (RFC-0023) |
| `0x27` | `ARENA_RESET` | 32 | IMPL | S | O(1); unknown arena traps (RFC-0023) |
| `0x28` | `SNAPSHOT` | 32 | DRAFT | S | SNAPSHOT; region bitmask; retention k=16 |
| `0x29` | `RESTORE` | 32 | DRAFT | S | Version handle; monotonic (I-Mono) |
| `0x2A` | `MEMCPY` | 32 | IMPL | S | Tensor-only MVP; HOST dir; CoW-safe (RFC-0023) |
| `0x2B` | `MEMSET` | 32 | IMPL | S | Byte pattern+len; PERSISTENT-RO (RFC-0023) |
| `0x2C` | `PREFETCH` | 32 | DRAFT | S | Cache hint |
| `0x2D` | `RESHAPE` | 32 | DRAFT | S | View or copy |
| `0x2E` | `SLICE` | 32 | IMPL | S | Flat [start,len) => [1,len] (RFC-0019) |
| `0x2F` | `CONCAT` | 32 | DRAFT | S | Axis |

### 3.5 Advanced tensors + attention + activations (`0x30-0x43`) — DRAFT, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x30` | `SORT` | 32 | DRAFT | S | Axis+order |
| `0x31` | `TOPK` | 32 | DRAFT | S | k+axis+largest+sorted |
| `0x32` | `ARGMAX` | 32 | DRAFT | S | Axis |
| `0x33` | `REDUCE` | 32 | DRAFT | S | sum/mean/max/min/prod |
| `0x34` | `BROADCAST` | 32 | DRAFT | S | Target shape |
| `0x35` | `PAD` | 32 | DRAFT | S | Per-axis |
| `0x36` | `TILE` | 32 | DRAFT | S | Repeats |
| `0x37` | `TRANSPOSE` | 32 | DRAFT | S | Axis permute |
| `0x38` | `KV_TRUNCATE` | 32 | IMPL | F2 | KV_CACHE; len+stream gate (RFC-0010) |
| `0x39` | `KV_COMPRESS` | 32 | DRAFT | F2 | KV_CACHE |
| `0x3A` | `FLASH_ATTN` | 32 | DRAFT | F2 | KV_CACHE; IO-aware |
| `0x3B` | `ATTN_SPARSE` | 32 | DRAFT | F2 | KV_CACHE; top-k via DISTANCE (T4) |
| `0x3C` | `SOFTMAX` | 32 | DRAFT | S | Axis+temperature |
| `0x3D` | `GELU` | 32 | DRAFT | S | Elementwise |
| `0x3E` | `SIGMOID` | 32 | DRAFT | S | Elementwise |
| `0x3F` | `TANH` | 32 | DRAFT | S | Elementwise |
| `0x40` | `RELU` | 32 | DRAFT | S | Elementwise |
| `0x41` | `EXP` | 32 | DRAFT | S | Elementwise |
| `0x42` | `LOG` | 32 | DRAFT | S | Elementwise |
| `0x43` | `CLIP` | 32 | DRAFT | S | min/max |

### 3.6 Full-duplex audio (`0x44-0x49`) — DRAFT, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x44` | `DEPFORMER` | 32 | DRAFT | F2 | KV_CACHE (Depformer KV); 6x1024x16, 16 codebooks |
| `0x45` | `STREAM_MERGE` | 32 | DRAFT | S | STREAM_RING; 17-stream mix |
| `0x46` | `VAD_DETECT` | 32 | DRAFT | S | Energy/ZCR/ML modes |
| `0x47` | `AUDIO_RESAMPLE` | 32 | DRAFT | S | STREAM_RING; 24k/16k/48k |
| `0x48` | `AUDIO_FILTER` | 32 | DRAFT | S | STREAM_RING; FIR/IIR |
| `0x49` | `AUDIO_WINDOW` | 32 | DRAFT | S | STREAM_RING; Hann/Hamming |

### 3.7 Gap (`0x4A-0x4F`) — RESERVED

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x4A-0x4F` | — | — | RESERVED | — | Dedup R4: [B] §5.5 cluster rows deleted; single home §3.10 |

### 3.8 Retrieval (`0x50-0x57`) — DRAFT/HELD, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x50` | `RAG_INDEX_ADD` | 32 | DRAFT | S | RAG_INDEX insert |
| `0x51` | `RAG_INDEX_DEL` | 32 | DRAFT | S | RAG_INDEX delete |
| `0x52` | `RAG_SEARCH` | 32 | DRAFT | S | RAG_INDEX; flat/IVF/HNSW/PQ + top-k |
| `0x53` | `EMBED_LOOKUP` | 32 | DRAFT | S | WEIGHTS; embedding bag |
| `0x54` | `HASH_BUCKET` | 32 | HELD | S | Lowering-first (R12) |
| `0x55` | `QUANTIZE_VEC` | 32 | HELD | S | Lowering-first (R12) |
| `0x56` | `PQ_ENCODE` | 32 | DRAFT | S | RAG_INDEX; product quantizer |
| `0x57` | `PQ_DECODE` | 32 | DRAFT | S | RAG_INDEX; inverse |

### 3.9 Classical ML (`0x58-0x5F`) — HELD, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x58` | `KMEANS_STEP` | 32 | HELD | S | Lloyd step; needs lowering proof (R12) |
| `0x59` | `LINEAR_REG` | 32 | HELD | S | y=wTx+b (MATVEC covers; R12) |
| `0x5A` | `LOGISTIC_REG` | 32 | HELD | S | +sigmoid (R12) |
| `0x5B` | `NAIVE_BAYES` | 32 | HELD | S | Gaussian/multinomial (R12) |
| `0x5C` | `SVM_PREDICT` | 32 | HELD | S | Kernels (R12) |
| `0x5D` | `PCA_STEP` | 32 | HELD | S | Power iteration (R12) |
| `0x5E` | `STANDARDIZE` | 32 | HELD | S | Z-score/min-max (R12) |
| `0x5F` | `NEAREST_CENTROID` | 32 | HELD | S | Argmin distance (R12) |

### 3.10 Determinism, conversion, telemetry, scheduler (`0x60-0x77`) — DRAFT, 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x60` | `RNG_SEED` | 32 | IMPL | S | Per-context seed (closes replay gap, §13) |
| `0x61` | `RNG_NEXT` | 32 | IMPL | S | Next u64 |
| `0x62` | `RNG_NORMAL` | 32 | IMPL | S | Box-Muller |
| `0x63` | `RNG_UNIFORM` | 32 | IMPL | S | [a,b) |
| `0x64` | `HASH` | 32 | IMPL | S | FNV-1a/64 (downgrade honesto; RFC-0005) |
| `0x65` | `CHECKSUM` | 32 | IMPL | S | CRC32-IEEE |
| `0x66` | `HMAC` | 32 | IMPL | S | Truncated SHA256 |
| `0x67` | `CAST` | 32 | DRAFT | S | FP32/BF16/FP16/INT8 (§11) |
| `0x68` | `QUANTIZE` | 32 | DRAFT | S | WEIGHTS; Q4_0/Q4_K/Q6_K/Q8_0 |
| `0x69` | `DEQUANT` | 32 | DRAFT | S | Inverse |
| `0x6A` | `CYCLES_COUNT` | 32 | IMPL | S | RDTSC-like |
| `0x6B` | `TRACE_EVENT` | 32 | IMPL | S | Structured tracing |
| `0x6C` | `SANITY_CHECK` | 32 | IMPL | S | Absorbs NaN/Inf per actor |
| `0x6D` | `PREEMPT_CHECK` | 32 | IMPL | S | Reads atomic flag |
| `0x6E` | `ASSERT` | 32 | IMPL | S | Predicate trap |
| `0x6F` | `DUMP` | 32 | IMPL | S | Debug snapshot |
| `0x70` | `YIELD` | 32 | IMPL | S | Yield CPU |
| `0x71` | `SET_DEADLINE` | 32 | IMPL | S | EDF deadline |
| `0x72` | `GET_DEADLINE` | 32 | IMPL | S | Read deadline |
| `0x73` | `PRIORITY_SET` | 32 | IMPL | S | RED/BLUE/GREEN |
| `0x74` | `PRIORITY_GET` | 32 | IMPL | S | Read priority |
| `0x75` | `LOCK` | 32 | IMPL | S | SHARED mutex |
| `0x76` | `UNLOCK` | 32 | IMPL | S | Release |
| `0x77` | `FENCE` | 32 | IMPL | S | SHARED visibility barrier |

### 3.11 Immediates (`0x78-0x79`) + gap (`0x7A-0x7F`)

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x78` | `LOADI` | 32 | IMPL | S | rdest <- imm u128 (RFC-0007) |
| `0x79` | `MOV` | 32 | IMPL | S | Reg copy (RFC-0007) |
| `0x7A-0x7F` | — | — | RESERVED | — | Future 32B control |

### 3.12 Cluster extended (`0x80-0x8F`) — DRAFT, 64B

X-forms (`_X`) are strict supersets of the `0x1A-0x1D` basics with
identical basic-field offsets; neutral ext fields imply equality (R4,
Section 12).

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x80` | `REMOTE_SPAWN_X` | 64 | DRAFT | S | node+entry+prio+lamport+deadline |
| `0x81` | `SIGNAL_X` | 64 | DRAFT | S | +lamport+deadline+capability (T2) |
| `0x82` | `SEND_TENSOR_X` | 64 | DRAFT | S | CLUSTER_STAGING; +checksum+compression |
| `0x83` | `BARRIER_X` | 64 | DRAFT | S | +epoch+deadline |
| `0x84` | `MIGRATE` | 64 | DRAFT | S | WAL; mode COPY/WAL_BACKED/RDMA; §12 |
| `0x85` | `MULTICAST` | 64 | DRAFT | S | CLUSTER_STAGING; N destinations |
| `0x86` | `REDUCE_REMOTE` | 64 | DRAFT | S | Allreduce |
| `0x87` | `GATHER_REMOTE` | 64 | DRAFT | S | Allgather |
| `0x88` | `SCATTER_REMOTE` | 64 | DRAFT | S | Scatter |
| `0x89` | `BROADCAST_REMOTE` | 64 | DRAFT | S | Broadcast |
| `0x8A` | `HEARTBEAT` | 64 | DRAFT | S | free_pct+queues |
| `0x8B` | `PING` | 64 | DRAFT | S | RTT probe |
| `0x8C` | `NODE_JOIN` | 64 | DRAFT | S | Handshake+cookie |
| `0x8D` | `NODE_LEAVE` | 64 | DRAFT | S | Graceful shutdown |
| `0x8E` | `NODE_SUSPECT` | 64 | DRAFT | S | SWIM suspect |
| `0x8F` | `NODE_DEAD` | 64 | DRAFT | S | SWIM dead |

### 3.13 MMIO (`0x90-0x9F`) — DRAFT (ops) / RESERVED, 64B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0x90` | `GPU_LAUNCH` | 64 | DRAFT | S | SCRATCH_GPU; kernel+grid |
| `0x91` | `GPU_WAIT` | 64 | DRAFT | S | Event handle |
| `0x92` | `NIC_SEND` | 64 | DRAFT | S | CLUSTER_STAGING; zero-copy TX |
| `0x93` | `NIC_RECV` | 64 | DRAFT | S | CLUSTER_STAGING; zero-copy RX |
| `0x94` | `DMA_START` | 64 | DRAFT | S | Any region; dir+len |
| `0x95` | `DMA_WAIT` | 64 | DRAFT | S | Handle |
| `0x96` | `ATOMIC_CAS` | 64 | DRAFT | S | SHARED; compare-and-swap |
| `0x97` | `ATOMIC_ADD` | 64 | DRAFT | S | SHARED; fetch-and-add |
| `0x98` | `ATOMIC_XCHG` | 64 | DRAFT | S | SHARED; exchange |
| `0x99` | `WFI` | 64 | DRAFT | S | Wait-for-interrupt |
| `0x9A-0x9F` | — | — | RESERVED | — | Future MMIO |

### 3.14 System (`0xA0-0xAF`) — DRAFT (ops) / RESERVED, 64B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0xA0` | `LOAD_MODEL` | 64 | DRAFT | S | WEIGHTS; path+quant (multi-model) |
| `0xA1` | `UNLOAD_MODEL` | 64 | DRAFT | S | WEIGHTS handle |
| `0xA2` | `SPAWN_CONTEXT` | 64 | DRAFT | S | ARENA; entry+prio+model_id |
| `0xA3` | `KILL_CONTEXT` | 64 | DRAFT | S | ARENA; ctx_id |
| `0xA4` | `SET_AFFINITY` | 64 | DRAFT | S | CPU mask |
| `0xA5` | `PROFILE_START` | 64 | DRAFT | S | Tag |
| `0xA6` | `PROFILE_STOP` | 64 | DRAFT | S | Tag |
| `0xA7` | `SET_MODEL` | 64 | DRAFT | S | Context model swap |
| `0xA8` | `GET_MODEL` | 64 | DRAFT | S | Current model |
| `0xA9` | `MODEL_SWITCH` | 64 | DRAFT | S | Multi-model fence |
| `0xAA-0xAF` | — | — | RESERVED | — | Future system |

### 3.15 ESCAPE block (`0xB0-0xB6`) — RSVD mechanism, 64B+

Encodings frozen per [R] R3; unimplemented. Full layout in [R] Section
3.2 (imported by reference; ext_len bounds in Section 9).

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0xB0` | `ESCAPE` | 64 | RSVD | S | 64B extension prefix |
| `0xB1` | `ESCAPE_128` | 128 | RSVD | S | 128B extension prefix |
| `0xB2` | `ESCAPE_256` | 256 | RSVD | S | 256B extension prefix |
| `0xB3` | `ESCAPE_VAR` | VAR | RSVD | S | ext_len self-described |
| `0xB4` | `VERSION` | 64 | RSVD | S | Runtime version declare |
| `0xB5` | `CAPABILITY_QUERY` | 64 | RSVD | S | Capability bitmask report |
| `0xB6` | `CAPABILITY_ASSERT` | 64 | RSVD | S | Require-or-fail |

### 3.16 Placeholders (`0xB7-0xFE`) — RESERVED

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0xB7-0xBF` | — | — | RESERVED | — | Federated-future placeholder (R3, R12) |
| `0xC0-0xCF` | — | — | RESERVED | — | Confidential/TEE (R12) |
| `0xD0-0xDF` | — | — | RESERVED | — | Photonic/neuromorphic (R12) |
| `0xE0-0xEF` | — | — | RESERVED | — | Quantum simulator (R12) |
| `0xF0-0xFE` | — | — | RESERVED | — | Extended system (R12) |

### 3.17 Terminal (`0xFF`) — IMPL, always 32B

| Hex | Mnemonic | W | Status | State | Region / Notes |
|-----|----------|---|--------|-------|----------------|
| `0xFF` | `NOP` | 32 | IMPL | S | No-op; sole R2 exception |

## 4. Instruction Formats and Size Function

### 4.1 32B (v1.3, exact)

```text
Byte 0: opcode · Byte 1: flags (legacy per-opcode, see §5)
Bytes 2-5: rdest/rsrc1/rsrc2/rsrc3 · Bytes 6-31: payload (26B, LE)
```

Unchanged from ESPEC Section 4. Decoder behavior unchanged.

### 4.2 64B (v2.0)

```text
Bytes 0-1: opcode/flags(canonical, see §5) · Bytes 2-7: rdest/rsrc1-5
Bytes 8-15: LAMPORT u64 · Bytes 16-23: DEADLINE u64 (ns, MAX=best-effort)
Bytes 24-31: PAYLOAD_EXT (HMAC/RDMA key/checksum, 0=none)
Bytes 32-63: PAYLOAD_CORE (32B, per-opcode)
```

`0xFF` in RSRC4/5 means unused; `RDEST=0xFF` discards; `LAMPORT=0`
means local; `DEADLINE=MAX` means best-effort (neutral values from
[B] Section 2.4, kept).

### 4.3 Size function [NORMATIVE for v2.0 decoders]

```text
op < 0x80          -> 32
0x80-0xAF, B4-B6   -> 64
0xB0               -> 64 head, total ext_len (valid: >=64, multiple of 8, <=1MiB)
0xB1 / 0xB2        -> 128 / 256
0xB3               -> ext_len (same validity)
0xB7-0xFE          -> reserved: MUST trap cleanly (no length assumed)
0xFF               -> 32 (R2 exception)
```

A v1.x decoder (32B only) MUST reject `op >= 0x80` with an explicit
`UnsupportedWidth` error — never misdecode.

## 5. Flags: Canonical Byte and 32B Legacy Compat

Canonical byte (64B only), from [B] Section 3, kept:

```text
bits 0-1 PRECISION (0=FP32,1=BF16,2=FP16,3=INT8/FP8) · bit2 FUSED_ACT
bit3 INPLACE · bit4 TRANSPOSE · bit5 PRIORITY · bit6 ATOMIC · bit7 PERSIST
```

Per-opcode reinterpretation MUST be documented in that opcode's RFC;
silence means canonical.

32B legacy compat (unchanged behavior; mapping for future migration):

| Legacy flag | Canonical equivalent (64B) |
|-------------|----------------------------|
| FORK prio / CTX_SWITCH prio | PRIORITY bit5 (+ payload for 3 levels) |
| ROPE INPLACE | INPLACE bit3 |
| MATVEC future TRANSPOSE | TRANSPOSE bit4 |
| SAMPLE TOPK / ATTN NOTIFY/MASK/BLOCK_SIZE | Payload mode bits (no flag equivalent) |
| TENSOR SPARSE/DENSITY/PERSIST | PERSIST bit7; sparsity stays payload |
| CODEC TENSOR-out | Payload mode |
| SENSE modes | Operand, not flags (unchanged) |

## 6. Registers

- **GPR: 16 x u128, R0-R15, 0xFF=unused.** Normative through v2.x
  (code as-is). GPR-256 and VEC file deferred to v3.0 MAJOR (R5).
- **SPECIAL (names adopted, status per name):** SP0 PC (effective),
  SP1 SP / SP2 FP (reserved until ARENA), SP3 CTX_ID (effective),
  SP4 NODE_ID (tied to cluster), SP5 LAMPORT (tied to cluster/64B),
  SP6 DEADLINE (tied to EDF), SP7 STATUS (partial: interrupt flag
  effective; NaN/OOB/TIMEOUT bits tied to SANITY_CHECK).
- **STATE handles S0-S63:** logical namespaces in payload immediates
  (layer_id, stream_id, kv/ssm handles). No file; no encoding needed.
- **R255/RFH:** rejected (R5); bank selection deferred with the file.

## 7. Memory: Logical Regions, Compat Profile, Region Table

### 7.1 Logical regions (16, from [B] Section 1.2, kept with R6/R7 fixes)

TEXT 0x0 (RO code) · GLOBAL 0x1 (RO const) · WEIGHTS 0x2 (RO+mprotect)
· ACTIVATION 0x3 (RW, CoW) · KV_CACHE 0x4 (RW, CoW: K/V, ssm, V(t),
H_t) · ARENA 0x5 (RW, per-actor) · SHARED 0x6 (RW+fence, CoW) · WAL 0x7
(append-only ring, retention 5s) · SNAPSHOT 0x8 (RO, window k=16) ·
STREAM_RING 0x9 (ring) · RAG_INDEX 0xA · CLUSTER_STAGING 0xB (back-
pressure at 80% + EDF deadline) · FEDERATED 0xC (placeholder) ·
CONFIDENTIAL 0xD (placeholder) · SCRATCH_GPU 0xE · MMIO 0xF (table at
+0x0 per R6).

### 7.2 32B compatibility profile (implemented, normative)

| Implemented | Logical |
|-------------|---------|
| GLOBAL 0x00 (code+const heap) | TEXT 0x0 (program bytes) + GLOBAL 0x1 (rest) |
| TEMPORAL 0x10 (circular) | STREAM_RING 0x9 |
| PERSISTENT 0x20 (mmap) | WEIGHTS 0x2 (model weights, RO) + ACTIVATION spill (PERSIST flag) |
| KV_CACHE 0x30 (layers) | KV_CACHE 0x4 |

32B instructions address profile regions only (existing top-byte tags
unchanged). 64B instructions may address all 16 by compact
REGION:OFFSET (zero-extended to u128, R1).

### 7.3 Region Table (MMIO+0x0, from [R] Section 5.2 with R6 fix)

`N_REGIONS==0` selects the Section 7.1 defaults. Otherwise the table
replaces them (entries: BASE u64 + SIZE_LOG2 + FLAGS + ALIAS u16).
Validation rules from [R] Section 9.2 kept (count, alignment,
SIZE_LOG2 bound, alias conflicts, RO+CoW inconsistency). Table reads
are boot-time; writes go through a future RFC (not this document).

### 7.4 Invariants (kept from [B] Section 1.3, normative targets)

WEIGHTS always RO; stateful writes only to KV_CACHE/ARENA (never
SHARED/WEIGHTS); SNAPSHOT window k=16; WAL append-only ring;
SHARED requires FENCE 0x77; CLUSTER_STAGING backpressure at 80%.
Per-actor slicing (ACTIVATION/KV/ARENA/STREAM_RING) per [B] 1.4 kept
as the Erlang-isolation target.

## 8. Bytecode Container and Feature Negotiation

`.m3bc` container from [R] Section 4 kept: `MAGIC "M3BC"` + MAJOR/MINOR/
PATCH + REQUIRED/OPTIONAL u64 + ENTRY_PC u32 + CRC32. Negotiation steps
kept. **Added (R10):** absent MAGIC implies the legacy v1.x `.m3bin`
path (execute without negotiation; never reject for missing header).
Feature registry from [R] Section 7 kept with ALL bits DRAFT —
`CAPABILITY_QUERY` reports implemented bits only (today: none beyond
v1.3 baseline, which needs no bit).

## 9. ESCAPE Mechanism (Fixed)

[R] Sections 3.* kept with bounds: `ext_len` valid iff `>=64`,
multiple of 8, `<=1MiB`; at most one chained ESCAPE (no recursion);
checksum validated pre-execution; unknown-required fails
`UnsupportedExtension`, unknown-optional skips `ext_len`; `SHARED`
origin requires FENCE. B-block split per R3. `VERSION`,
`CAPABILITY_QUERY/ASSERT` reserved (unimplemented).

## 10. Stateful Taxonomy (Extension of ESPEC 6.4)

Family 1 (drop chunk): FFN, CONV, aggregated GATHER, MoE-gate,
cache-less ATTN, ROPE, SILU, elementwise, RESHAPE/SLICE/CONCAT views,
AUDIO_RESAMPLE/FILTER/WINDOW (frame-local).
Family 2 (CoW + ABORT path, all in KV_CACHE unless noted): ATTN KV,
SSM h_t, SNN V(t), RANK1 H_t, ODE x(t), DEPFORMER KV (17 streams),
DENOISE x_t (ACTIVATION), KV_TRUNCATE/COMPRESS/FLASH/SPARSE targets.
Every future stateful RFC declares (a) home region, (b) snapshot cost,
(c) ABORT path — no exceptions (carries ESPEC 6.4 + [B] Section 7).

## 11. Precision Rule [NORMATIVE]

PRECISION bits are ADVISORY kernel routing. A runtime MUST either
execute at the requested precision or trap `UnsupportedPrecision`.
Silent fallback to another precision is FORBIDDEN and voids any
numerical bound (this answers [B]'s charge honestly: agnosticism is a
routing contract + trap rule, not a compute claim). Effective today:
FP32 compute; quantized weight paths (Q4_0/Q4_K/Q6_K/Q8_0) as in
ESPEC Section 10.

## 12. Cluster: Basic Forms, X-Forms, WAL Naming

Basics `0x1A-0x1D` keep frozen 32B payloads (ESPEC Section 12).
X-forms `0x80-0x83` are field-compatible supersets (basic fields at
identical offsets, then LAMPORT/DEADLINE/PAYLOAD_EXT). Neutral ext
fields imply semantic equality with the basic form (R4).
Timeouts, cookie handshake, urgent-over-bulk queues, heartbeat
placement and SWIM states (`0x8C-0x8F`) per ESPEC Section 12 and
[B] Section 5.9 (kept). **WAL naming (R9):** a 32B WAL_BACKED send
names its log entry by `(source NODE_ID, sender SP5 LAMPORT at issue)`;
the 64B MIGRATE names it explicitly (WAL id u48 + checksum). Retention
5s post-ACK; T5 conservation proof obligation extends to both namings
(`formal/Formal/ClusterWAL.lean` follow-up).

## 13. Determinism

Seeded execution replays bit-exactly (ESPEC Section 9). The RNG block
(`0x60-0x63`) closes the genuine gap: no stochastic opcode
(`DENOISE_STEP` sigma>0, future sampling modes) may draw from an
unseeding source; default context seed is fixed and `RNG_SEED`
re-seeds explicitly. HASH/CHECKSUM/HMAC (`0x64-0x66`) serve payload
and WAL integrity.

## 14. Unified Wave Plan

W0 Freeze this document + RFC template adoption (R11). W1 Dual-mode
decoder + width errors (IMPLEMENTED for decode/load boundary:
RFC-0002 — `instr_width`, rejection variants, bounds, 3 conformance
tests; PENDING: 64B fetch stride in `Vm`, 64B `encode`). W2 Profile
mapping (PENDING) + retention k=16 (IMPLEMENTED: RFC-0003 — window,
opt-out, 3 conformance tests) + I-Mono (IMPLEMENTED: restoreFix) /
I-Persist (OPEN, own RFC).
W3 `GATHER` + `DISTANCE`/`RANK1` + pruned-ATTN demo (T4) (IMPLEMENTED:
RFC-0004 — 3 opcodes + SAMPLE-TOPK, 7 conformance tests, 3 demos exit 0
(`attn_topk_demo` removido em auditoria RFC-0012: packed precisa de
`SLICE`; goldens cobrem);
§3.3 statuses flipped to IMPL). W4
`SPIKE`/`ODE`-last + `FOREST`/`DENOISE`/`CONV` IMPLEMENTED (RFC-0012/0013/0017). W5 Telemetry+scheduler
(IMPLEMENTED: RFC-0006 — 14 opcodes, behavior goldens, demo exit 0;
determinism `0x60-0x66` already IMPLEMENTED: RFC-0005). W6
Memory/arena (`0x26-0x2F`) + `FENCE`/`LOCK`. W7 Retrieval
(`0x50-0x57`) with lowering filter. W8 Moshi full-duplex
(`0x44-0x49` + 17-stream KV). W9 Cluster X-forms + MIGRATE/WAL +
SWIM. W10 MMIO/system + `.m3bc` loader. Order is normative-draft;
each wave needs its RFC + green tests + Lean delta where bounds move.

## 15. Open Obligations

1. Lean: TopK main bound proof; stepError physical instantiation
   (carried from ESPEC Section 15).
2. New: dual-mode decoder model (size function + 0xFF exception);
   WAL dual-naming conservation; Region Table refinement (N==0 vs
   override); X-form equivalence (neutral fields).
3. Empirical: per-wave benches before any status flip DRAFT->IMPL.

## Appendix A. Source Cross-Reference

- Resolutions R1-R12 answer, in order: [B] Sections 1.1, 2.1+5.13,
  5.12 vs [R] Section 3, [B] 5.5 vs 5.9 + frozen `0x1A-0x1D`,
  [B] Section 4 + [R] 5.1 vs `src/context.rs:93`,
  4-region code vs [B] 1.2 vs [R] 5.2, [B] TEMPORAL ghost,
  [B] Section 3 vs legacy flags, single-MIGRATE decision,
  headerless v1.3 vs [R] 8.1, governance adoption, HELD policy.
- Nothing in [B] Sections 6.* (64B payload details for ATTN/SSM/
  DEPFORMER/MERGE/MIGRATE/RAG) is lost: they become the 64B
  PAYLOAD_CORE references for those opcodes (identical offsets kept
  where they fit the Section 4.2 layout; conflicts resolved per R4).
- [B] Section 10 (gains/costs/risks) and [R] Appendices A/B are kept
  as informative annexes in `docs/arq/`; normative content moved here.
