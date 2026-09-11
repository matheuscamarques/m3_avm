# RFC-0025 — Conversion Block: `CAST` + `QUANTIZE` + `DEQUANT` (`0x67/0x68/0x69`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.10 (0x67,0x68,0x69 DRAFT->IMPL), §11 (note);
              docs/ESPEC.md §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.6 -> v1.7: conversion family 0x67-0x69)
```

## Abstract

Implements the conversion family: value `CAST` between FP32/F16/BF16/
INT8/UINT8 (explicit pairs, no silent fallback), block `QUANTIZE` to
Q4_0/Q8_0 (new encoders, exact inverses of this repo's decoders, error-
bounded), and `DEQUANT` over the existing dispatcher (F32/F16/Q4_0/
Q4_K/Q6_K/Q8_0). This is also the enforcement point of the Precision
Rule (§11) on the 32B path: every conversion either executes at an
explicitly supported precision or traps `UnsupportedPrecision`-style —
never falls back silently. Q4_K/Q6_K (and beyond) have NO encoder here:
writing a 6-bit super-block packer from scratch without reference
vectors would be silent-guess engineering, which this project forbids;
they trap explicitly until a vector-validated RFC lands.

## Motivation

Per the ESPEC Section 13 filter: nothing converts tensor dtype
(`TENSOR` fixes dtype at alloc; downstream ops assume F32), nothing
produces quantized blocks in-VM (only decoders existed), and nothing
gives the RNG→index path (`FILLER_SPEECH.md` §5.4 waits on `CAST`)
a clean float-to-int step. These three unblock mixed-precision
programs, offline quantization workflows, and the filler-phrase
dispatch without touching a single shipped encoding.

## Specification

All 32B. Outputs are FRESH tensors (alloc + write; never in-place
lies, never PERSISTENT aliasing — see §Security).

```text
0x67 CAST  rdest <- NEW tensor addr, rsrc1 reg holds src addr.
  payload[0]=dst code (0=F32,1=F16,2=BF16,3=I8,4=U8).
  Src dtype inferred from tensor meta (never re-declared: no
  mismatch class). Supported pairs: F32<->{F16,BF16,I8,U8}.
  Anything else (quantized input, unknown meta, same-dtype no-op,
  bad code) => Err. Identity (e.g. F32->F32) is Err (no-op is caller
  bug; use MOV).
  Semantics: F16 via half crate both ways; BF16 via round-to-nearest-
  even bit truncation with NaN guard (NaN stays NaN, Inf stays Inf);
  INT8/UINT8 via round + clamp (saturating, NaN->0 — defined, not a
  trap). Widening (back to F32) is exact.
0x68 QUANTIZE  rdest <- NEW tensor addr, rsrc1 reg holds F32 src addr.
  payload[0]=quant dtype discriminant (4=Q4_0, 8=Q8_0; matches DType).
  Input must be dense F32, all-finite (Inf/NaN => Err, else the error
  bound below would lie), numel % 32 == 0 (no partial blocks).
  Q4_0: per-32 amax, d = amax/7 stored f16 (repo decoder is the exact
  inverse: low nibble = y[0..16], high = y[16..32], bias -8);
  per-element error <= d/2 with the STORED d; amax=0 => exact-zero
  block. Q8_0: d = amax/127 stored f16, int8 quants clamped
  [-127,127]; same bound shape. Other types => Err UnsupportedQuant
  (no Q4_K/Q6_K encoder in this RFC — see Abstract).
0x69 DEQUANT  rdest <- NEW F32 tensor addr (same shape as src),
  rsrc1 reg holds quantized src addr. Types served by the existing
  dispatcher: F32 (identity copy), F16, Q4_0, Q4_K (+Q5_K stub, as the
  dispatcher does), Q6_K, Q8_0. Anything else => Err. byte_len must
  match blocks for numel (defensive: no partial-block reads).
```

`DType::BF16 = 64` (new variant; 64 is outside every numbering in play:
DType ≤ 15, GGML ≤ ~40 — zero behavior change in `from_u8/from_u32`
callers, verified by suite). `byte_width` 2, not quantized.
Assembler gains `bf16` in `parse_dtype`. TENSOR `... bf16` allocates
(zeroed, like the `_` arm); value init comes from CAST.

Assembler:
```text
CAST rD, rT DST=F32|F16|BF16|I8|U8 (numerics 0-4 also accepted)
QUANTIZE rD, rT Q=Q4_0|Q8_0 (nothing else parses — deliberately
  stricter than DIR=, see Abstract: unexecutable spellings fail here)
DEQUANT rD, rT (no keys)
```

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats + one enum variant
with full wildcard coverage — verified zero behavior change by suite).
`0x67/0x68/0x69` previously errored at execute. v1.6 -> v1.7.

## Security Considerations

- Fresh-output discipline: CAST/QUANTIZE/DEQUANT check
  `region_of(out) != PERSISTENT` post-alloc and refuse weight-aliasing
  writes with a named error (GGUF shape-collision guard). Pre-existing
  producers (SLICE/RESHAPE/CONCAT/TENSOR-FILL) predate this guard —
  noted as follow-up hardening, not silently fixed here.
- No precision lies: unsupported pairs trap; identity traps; partial
  blocks trap; non-finite quantize input traps. NaN/Inf through CAST
  follow documented total rules (half-crate / RNE-guard / saturate).
- `u64->usize` conversions via `try_from`; shape products checked;
  OOB impossible by construction (fresh allocs sized from verified
  numel).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/memory.rs`: `DType::BF16 = 64` + `from_u8` arm + `byte_width`
  (2); `parse_dtype` gains `bf16` (opcodes.rs).
- `src/quant.rs`: `f32_to_bf16_bits` (RNE + NaN guard) /
  `bf16_bits_to_f32`, `quantize_q4_0`, `quantize_q8_0` (bool-returning,
  no panics), `quant_block_info`, unit tests (RNE vectors, NaN/Inf,
  roundtrip bounds vs STORED d, zero-block, bad input).
- `src/opcodes.rs`: `OP_CAST/QUANTIZE/DEQUANT`, `CAST_DST_*` consts,
  params + setters, `instr_*`, mnemonics, `parse_line` arms, roundtrips.
- `src/vm.rs`: `exec_cast/quantize/dequant`, dispatch arms, 3 counters.
- Conformance: cast roundtrips per pair (bounds), bf16 RNE vectors,
  int clamp table, identity/unsupported/sparse/missing-meta errors,
  quantize golden bytes (all-ones block) + error bound on ramp +
  non-finite/non-mult32/non-F32/unsupported-Q errors, dequant hand-
  built blocks + dispatcher coverage + mismatch errors, assembler
  roundtrips + strict rejections, combined program
  (`vm::test_rfc0025_assembled_program_runs`: counters exact),
  `programs/cast_quant_demo.m3asm` (counters at 1, verified).
- Suite: `cargo test --lib` 300 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/memory_bench.rs` gains cast/quantize/dequant
  (per-instruction via `step_instruction`).

Follow-up (NOT this RFC): Q4_K/Q6_K(/beyond) encoders with reference
vectors; transitive CAST pairs (via F32 today); dtype-strict dispatch
hardening downstream (pre-existing F16/I8/U8 blind-read class, cf.
qa.rs:38); dedup FORK/SNAPSHOT + ABORT/RESTORE helpers (RFC-0024
follow-up); PERSISTENT-alias guard for pre-existing producers;
per-arena quotas; sparse MEMCPY.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0025-00 | 2026-09-11 | DRAFT: conversion block + precision enforcement |
| 0025-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.7 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
