# RFC-0024 — Memory Views & Versions: `SNAPSHOT`/`RESTORE` + `PREFETCH`/`RESHAPE`/`CONCAT` (`0x28/0x29/0x2C/0x2D/0x2F`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.4 (0x28,0x29,0x2C,0x2D,0x2F DRAFT->IMPL),
              §14 W6 (done); docs/ESPEC.md §5.8, §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.5 -> v1.6: memory/arena family 0x26-0x2F complete,
              RFC-0023 + this RFC; intermediate minors were not cut)
```

## Abstract

Completes the memory/arena family: explicit named snapshots
(`SNAPSHOT` returns a version handle; `RESTORE` rewinds memory +
engine maps to it) plus the view/data-shape block (`PREFETCH` cache
hint, `RESHAPE` copy with new shape, `CONCAT` along an axis).
`SNAPSHOT` is fork-state without forking a context; `RESTORE` is
abort-state without killing one — both reuse the audited FORK/ABORT
machinery (`memory.snapshot/restore`, version-tagged engine stacks,
retention k=16, I-Mono never-lowers). The region bitmask exists but
only the full set is accepted (partial snapshots are follow-up, fail
loudly). `RESHAPE`/`CONCAT` are copies, not views (strided views are
RFC-0019 follow-up); sparse and dtype-mismatched inputs trap
explicitly.

## Motivation

Per the ESPEC Section 13 filter: nothing names a restorable point
without forking (FORK always spawns), nothing rewinds to it without
aborting (ABORT always kills), and nothing expresses shape change
(`SLICE` only narrows flat windows) or multi-tensor assembly
(`SCATTER_ADD` accumulates, never concatenates) over existing opcodes.
These five close the memory family opened by RFC-0023 and give
`CONCAT`-based programs (RAG packing, batch assembly) a native path.

## Specification

All 32B. Tensor addrs in regs (SAMPLE precedent); dense f32-first,
same honesty rules as RFC-0023 (OOB traps, WEIGHTS RO, sparse traps).

```text
0x28 SNAPSHOT  rdest <- version u64 (as u128).
  payload[0]=mask u8 (default 0b111 when absent... — see below).
  Takes memory.snapshot() (bumps version, Arc-clone 5 maps, evicts
  beyond k=16) AND pushes all 4 engine-map stacks tagged with the
  version (ssm/rank1/snn/arenas — same 4 lines as FORK).
  Mask bits: 0b001=GLOBAL, 0b010=KV_CACHE, 0b100=ENGINE MAPS.
  Only 0b111 accepted; anything else => Err PartialMask (partial
  snapshots are follow-up — the machinery is all-or-nothing today).
  Assembler default (no MASK=) = 0b111.
0x29 RESTORE  rsrc1 reg holds version u64. memory.restore(version)?
  then rewind all 4 engine stacks to <= version (same version-gated
  pops as ABORT ts!=0). Unknown/evicted version => Err (loud; aged-out
  versions fail exactly like ABORT-ts on evicted versions). Restoring
  never lowers the version counter (I-Mono, via memory.restore).
  Contexts are untouched (unlike ABORT, nobody dies).
0x2C PREFETCH  rdest reg holds tensor addr (read-only use).
  payload[0..4]=len u32 (0 = whole tensor), [4..12]=offset u64.
  Validates tensor resolves + bounds (else Err; sparse => Err), then
  reads the window to fault pages (best-effort cache hint; no
  performance claim — it is a hint). PERSISTENT readable (mmap fault
  is the honest case); no writes anywhere.
0x2D RESHAPE  rdest <- NEW tensor addr, rsrc1 reg holds src addr.
  payload[0]=ndim u8 (1-4), [1..5]=d0 u32, [5..9]=d1, [9..13]=d2,
  [13..17]=d3 (LE; only first ndim used). numel(new) must equal
  numel(src), else Err. Copy (alloc + byte copy, dtype preserved);
  NOT a view (strided views are follow-up). Sparse => Err. SRC in any
  readable region OK (meta-driven); DST is fresh GLOBAL (never
  PERSISTENT aliasing).
0x2F CONCAT  rdest <- NEW tensor addr, rsrc1=A addr, rsrc2=B addr.
  payload[0]=axis u8 (default 0). Generic N-D: same rank, all dims
  equal except axis, axis < ndim, else Err. Same dtype required, else
  Err (no silent cast); meta byte_len must equal shape.product *
  dtype width on both sides (defensive: quantized layouts are not
  plain row-major). Copy assembly (no aliasing). Sparse => Err.
```

Assembler:
```text
SNAPSHOT rD [MASK=n]
RESTORE rV
PREFETCH rT [LEN=n] [OFF=n]
RESHAPE rD, rT SHAPE=AxBxC (x-separated, 1-4 dims)
CONCAT rD, rA, rB [AXIS=n]
```
Strict lists gain exactly these keys; `SHAPE=` with 0 or 5+ dims,
non-numeric dims, or zero dims => Err.

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats). `0x28/0x29/0x2C/
0x2D/0x2F` previously errored at execute. No encoding touched. v1.5 ->
v1.6: family `0x26-0x2F` now fully IMPL (RFC-0023 + this RFC).

## Security Considerations

- `RESTORE` cannot fabricate state: unknown versions trap; the counter
  never lowers (no id-reuse clobber — the `restoreFix` theorem,
  `formal/Formal/Rollback.lean`, covers this path unchanged since the
  same `memory.restore` is called).
- Aged-out named versions fail loudly (retention is observable via
  `snapshot_count`, not silent).
- `MASK != 0b111` traps instead of silently snapshotting everything
  (a lying mask would be a correctness hazard for future partial
  restores).
- RESHAPE/CONCAT allocate fresh GLOBAL (no PERSISTENT aliasing, no
  in-place shape lies); OOB/numel/dtype guards as specified.
- PREFETCH performs no writes (read-only by construction).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_SNAPSHOT/RESTORE/PREFETCH/RESHAPE/CONCAT`,
  `SNAP_MASK_*` consts, params + setters, `instr_*` ctors, mnemonics,
  `parse_line` arms (`SHAPE=` x-separated), roundtrips.
- `src/vm.rs`: `exec_snapshot/restore/prefetch/reshape/concat`,
  dispatch arms, 5 counters. SNAPSHOT duplicates FORK's 4 push lines
  (comment points at canonical site; dedup refactor is follow-up —
  audited paths untouched). RESTORE = `memory.restore` + ABORT-style
  version-gated pops (contexts untouched).
- Conformance: snapshot returns fresh version + monotonic growth;
  restore roundtrip (memset-zero then restore ones); engine coherence
  (arena cursor rewinds); unknown version / MASK!=7 / OOB / sparse /
  numel / axis / dtype errors; reshape values preserved across ndim
  change; concat order on both axes; assembler roundtrips + strict
  rejections; combined program (`vm::test_rfc0024_assembled_program_
  runs`: counters exact); `programs/snap_concat_demo.m3asm` (all 5
  counters at 1, verified).
- Suite: `cargo test --lib` 292 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/memory_bench.rs` gains snapshot/restore +
  reshape + concat (per-instruction via `step_instruction`).

Follow-up (NOT this RFC): partial snapshots (MASK bits honrados);
strided zero-copy views; sparse RESHAPE/CONCAT; GPU/NIC MEMCPY dirs
(RFC-0023 follow-up); dedup refactor FORK/SNAPSHOT + ABORT/RESTORE
push/pop helpers; per-arena quotas.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0024-00 | 2026-09-11 | DRAFT: views & versions + demo |
| 0024-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.6 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
