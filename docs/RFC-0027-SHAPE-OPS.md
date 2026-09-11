# RFC-0027 — Shape Ops: `SORT`/`TOPK`/`ARGMAX`/`REDUCE`/`BROADCAST`/`PAD`/`TILE`/`TRANSPOSE` (`0x30-0x37`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.5-partial (0x30-0x37 DRAFT->IMPL);
              docs/ESPEC.md §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.8 -> v1.9: shape family 0x30-0x37; Fase 3 de 3 partes)
```

## Abstract

Implements the shape family (part 1 of Fase 3): lane-wise
sort/top-k/argmax/reduction along an axis plus shape constructors
(broadcast/pad/tile/transpose). All eight are dense-F32-only MVP with
explicit traps (sparse, other dtypes, bad axis, degenerate selections
— never silent). Ordering is total and deterministic: NaN sorts as
+infinity, ties always resolve to the lower original index, and
`ARGMAX`/`REDUCE MAX` ignore NaN unless all values are NaN (Rust
`f32::max` semantics, documented — deliberately NOT numpy's
propagate-everywhere). Outputs are fresh tensors (no aliasing, no
views). Multi-axis pad/tile compose via repetition (single-axis ops
by design).

## Motivation

Per the ESPEC Section 13 filter: sorting, top-k selection, argmax,
reduction and reshaping-by-construction have no expression over
`MATVEC/MUL/ADD` at acceptable cost, and `DISTANCE`'s fused top-k only
serves retrieval — standalone `TOPK`/`SORT` unblock MoE dispatch
loops, beam bookkeeping, and the `ATTN_SPARSE` leg (RFC-0004
follow-up, later RFC). `TRANSPOSE` is already assumed by the
`MATVEC.TRANSPOSE` flag design (§13.2).

## Specification

All 32B, stateless (S), dense F32 only. Lanes run along `axis` with
`outer = prod(shape[..axis])`, `dim = shape[axis]`, `inner =
prod(shape[axis+1..])` (shared decomposition).

```text
0x30 SORT  rdest <- NEW tensor, same shape, lanes sorted.
  payload[0]=axis (default 0), [1]=order (0=ASC,1=DESC, default 0).
  Total order: NaN as +inf (ASC: NaNs last; DESC: first); ties
  (incl. -0.0/0.0) resolve to the lower original index, both orders.
0x31 TOPK  rdest <- NEW tensor, same rank, axis dim = 2k:
  [k values | k indices-as-f32] (DISTANCE pack precedent, RFC-0004).
  payload[0]=axis (default 0), [1..3]=k u16 (REQUIRED; 0 => Err),
  [3]=largest (1=desc, default; 0=asc), [4]=sorted (1=ordered, default;
  0=original-index order — still deterministic). k > dim => Err (no
  silent clamp). Same NaN/tie rules as SORT.
0x32 ARGMAX  rdest <- NEW tensor, same rank, axis dim = 1, indices as
  f32 (exact for dim < 2^24). payload[0]=axis (default 0). NaN ignored
  unless all-NaN (then lowest index) — maxNum-consistent with REDUCE.
0x33 REDUCE  rdest <- NEW tensor. payload[0]=op (0=SUM,1=MEAN,2=MAX,
  3=MIN,4=PROD), [1]=axis (0xFF = full reduce to [1]; default full).
  With axis: that dim -> 1. SUM/MEAN/PROD propagate NaN (plain f32
  arithmetic); MAX/MIN skip NaN (f32::max semantics), all-NaN => NaN.
  MEAN divides by count as f32.
0x34 BROADCAST  rdest <- NEW tensor with target shape.
  payload[0]=ndim (1-4), [1..17]=4×u32 dims (RESHAPE encoding).
  Right-aligned compat (each dim 1 or equal), else Err. Copy.
0x35 PAD  rdest <- NEW tensor, single axis grown.
  payload[0..4]=value f32, [4]=axis (default 0), [5..9]=before u32,
  [9..13]=after u32. before==after==0 => Err (no-op; use MOV).
  Multi-axis = repeat (composable by design).
0x36 TILE  rdest <- NEW tensor, single axis repeated.
  payload[0..4]=reps u32 (>= 1; 0 => Err), [4]=axis (default 0).
  Overflow of axis dim => Err (checked).
0x37 TRANSPOSE  rdest <- NEW tensor, permuted (generic N-D, not just 2D).
  payload[0]=ndim (1-4, must equal tensor rank), [1..5]=perm u8 ×4
  (each < ndim, all distinct, else Err). Copy.
```

Assembler:
```text
SORT rD, rT [AXIS=n] [ORDER=ASC|DESC]
TOPK rD, rT K=n [AXIS=n] [LARGEST|SMALLEST] [SORTED|UNSORTED]
ARGMAX rD, rT [AXIS=n]
REDUCE rD, rT OP=SUM|MEAN|MAX|MIN|PROD [AXIS=n]
BROADCAST rD, rT SHAPE=AxBxC
PAD rD, rT VALUE=x [AXIS=n] [BEFORE=n] [AFTER=n]
TILE rD, rT REPS=n [AXIS=n]
TRANSPOSE rD, rT AXES=AxBxC (x-separated perm; NOT comma — assembler
  splits commas)
```
Strict lists gain exactly these keys. `SHAPE=` shares the `RESHAPE`
parser (factored helper; `RESHAPE` behavior unchanged, suite-guarded).

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats). `0x30-0x37`
previously errored at execute. No encoding touched. v1.8 -> v1.9.

## Security Considerations

- All bounds checked pre-alloc (axis < rank, k <= dim, compat dims,
  checked dim products) — no OOB read, no empty output, no overflow
  wrap (checked arithmetic, `Err` on overflow).
- NaN/total-order rules are total functions: no input panics, no
  data-dependent timing beyond input-size-fixed loops (full-lane sort;
  partial heaps are follow-up, documented as perf-only).
- Fresh outputs only (no aliasing, no PERSISTENT writes possible —
  `alloc_tensor` + `write` on GLOBAL).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_SORT/TOPK/ARGMAX/REDUCE/BROADCAST/PAD/TILE/
  TRANSPOSE` + mode consts, params + setters, `instr_*` ctors,
  mnemonics, `parse_line` arms, shared `parse_shape_dims` helper
  (RESHAPE refactored onto it, behavior identical), roundtrips.
- `src/vm.rs`: `axis_decomp` helper, 8 exec fns, dispatch arms,
  8 counters.
- Conformance: sort asc/desc incl. NaN placement + tie order, top-k
  pack layout + k-errors + smallest/unsorted modes, argmax incl.
  all-NaN, all 5 reduce ops incl. NaN rules + full/axis forms, bcast
  compat/incompat, pad/tile windows + no-op/zero errors, transpose
  2-D + 3-D + bad-perm errors, sparse/dtype/missing-meta errors,
  assembler roundtrips + strict rejections, combined program
  (`vm::test_rfc0027_assembled_program_runs`: counters exact),
  `programs/shape_demo.m3asm` (all 8 counters at 1, values verified).
- Suite: `cargo test --lib` 311 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/shape_bench.rs` (1K-lane via `step_instruction`;
  medians `--quick` neste host: sort ~29µs, topk8 ~16.8µs,
  reduce ~10.8µs, argmax ~8.7µs — referência).
- Benches: `benches/shape_bench.rs` (sort/topk/reduce 1K-lane,
  per-instruction via `step_instruction`).

Follow-up (NOT this RFC): partial-heap top-k (perf only); sparse
shape ops; non-F32 dtypes; strided views; `ATTN_SPARSE`/`KV_COMPRESS`/
`FLASH_ATTN`/`SOFTMAX`/activations (Fase 3 parts 2-3); TopK Lean main
bound (with the attention part, where the bound matters).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0027-00 | 2026-09-11 | DRAFT: shape family + demo |
| 0027-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.9 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
