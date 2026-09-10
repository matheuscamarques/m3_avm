# RFC-0019 — TENSOR FILL + SLICE (`0x2E`): Literal Data Paths

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x2E DRAFT->IMPL); TENSOR row (FILL mode)
Obsoletes   : None
Feature Bit : none
Bump        : none (same rule: next MINOR reserved for the next family)
```

## Abstract

Closes the audit loop opened in RFC-0012: `.m3asm` programs could not
build non-ramp data (no literal fill) nor split packed tensors (no
slice), which killed `forest_demo` and `attn_topk_demo`. Adds the two
minimal primitives: scalar `FILL` on `TENSOR` and flat `SLICE 0x2E`.
Both resurrect as verified-executed demos.

## Motivation

Every table-driven op (`FOREST`) and every packed layout (`DISTANCE`
top-k) was unreachable from handwritten programs. The goldens proved
the ops; these two primitives prove the programs.

## Specification

```text
TENSOR rD rows cols dtype [FILL=x] — payload[22..26] = fill f32 LE +
  flags bit TENSOR_FLAG_FILL (0b01). F32 only (other dtypes + FILL =
  explicit Err, not silent cast). SPARSE + FILL together = Err
  (contradictory init). Absent FILL = legacy ramp, unchanged.
0x2E SLICE rD, rT START=n LEN=n — flat slice [start, start+len) as
  new tensor [1, len] (consistent with DISTANCE/SAMPLE-TOPK shapes).
  Out-of-range (incl. empty selection len==0) = Err, never clamp.
  Payload [0..4]=start u32, [4..8]=len u32. Dense f32 only in this RFC.
```

Assembler: `TENSOR rD 2 2 f32 FILL=0`, `SLICE rD, rT START=3 LEN=3`.
Strict lists gain `FILL=` (TENSOR) and nothing new (SLICE numerics are
positional).

## Backwards Compatibility

Additive: one flag bit (previously always 0), one payload field use in
previously-zero bytes `[22..26]`, one new opcode. Old binaries have
flag 0 and zero bytes = legacy ramp path, bit-identical.

## Security Considerations

- `FILL` is a scalar broadcast: allocation size unchanged (shape still
  governs), no new exhaustion vector.
- `SLICE` bounds-checked pre-alloc: no OOB read, no empty tensor.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `TENSOR_FLAG_FILL`, `tensor_fill/set_tensor_fill`,
  `OP_SLICE`, `slice_params/set`, `instr_slice`, mnemonics, parse arms,
  roundtrips.
- `src/vm.rs`: FILL branch in `exec_tensor`, `exec_slice`, dispatch,
  `slice_execs`.
- Conformance: FILL zeros/ones vs ramp default, SPARSE+FILL
  conflict, non-F32 FILL error, SLICE exact incl. packed-DISTANCE
  split, OOB/empty errors, DISTANCE→SLICE→GATHER integration chain,
  resurrected `forest_demo` (forest_execs=2) + `attn_topk_demo`
  (distance/slice/gather×2/attn=1, all verified executed).
- Suite: `cargo test --lib` 256 passed (sole failure: pre-existing
  unrelated moshi norm-gamma).
- Audit note: strictifying TENSOR exposed a positional-dtype hazard
  (`TENSOR r2 2 f32 FILL=0` read FILL as dtype); rewritten with
  consumed-token tracking, corpus 34/34 still assembles.

Follow-up (NOT this RFC): multi-value init lists, axis-SLICE (N-D),
strided views (zero-copy), sparse FILL.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0019-00 | 2026-09-10 | DRAFT: FILL + SLICE + demo resurrections |
| 0019-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |
