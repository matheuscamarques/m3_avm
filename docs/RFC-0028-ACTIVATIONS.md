# RFC-0028 — Activations: `SOFTMAX` + `GELU`/`SIGMOID`/`TANH`/`RELU`/`EXP`/`LOG`/`CLIP` (`0x3C-0x43`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.5-partial (0x3C-0x43 DRAFT->IMPL);
              docs/ESPEC.md §5.8, §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.9 -> v1.10: activation family 0x3C-0x43. NOT
              v2.0: the 1.x line continues past 1.9 as 1.10+; v2.x stays
              reserved for the dual-mode freeze — §19 rule clarified)
```

## Abstract

Implements the activation family (Fase 3, part 2 of 3): numerically
stable `SOFTMAX` (axis + temperature) plus seven elementwise ops
(`GELU` exact-erf, `SIGMOID`, `TANH`, `RELU`, `EXP`, `LOG`, `CLIP`).
All eight are dense-F32-only MVP producing fresh tensors, with total
IEEE-documented behavior (no traps on values except malformed
parameters: bad TEMP, inverted/NaN CLIP bounds). `GELU` uses the exact
erf formulation (Abramowitz & Stegun 7.1.26, |err| ≤ 1.5e-7 — no
`libm` dependency, bound documented and tested). This RFC also freezes
the versioning clarification the bump forced: minor versions continue
1.9 → 1.10 → 1.11; `v2.x` means the dual-mode freeze, never "the next
minor after 1.9".

## Motivation

Per the ESPEC Section 13 filter: every transformer path in every
`.m3asm` program today either calls whole-model inference (opaque) or
cannot express activations at all — `FFN` fuses SwiGLU internally and
offers no elementwise vocabulary. These eight give programs the
standard nonlinearity set plus the only honest softmax (max-subtracted,
temperature-explicit), unblocking hand-written attention probes,
activation patching demos, and the `ATTN_SPARSE` leg (part 3).

## Specification

All 32B, stateless (S), dense F32 only (meta/sparse/layout guards via
the shared reader). Outputs are FRESH tensors, same shape.

```text
0x3C SOFTMAX  rdest <- NEW tensor, lanes normalized along axis.
  payload[0]=axis (0xFF = last, default), [1..5]=temp f32 (default 1.0,
  written by the assembler when absent). Stable: subtract lane max,
  divide by temp, exp, normalize. temp NaN or <= 0 => Err; temp=+inf
  is ALLOWED and means uniform (documented limit, explicit intent).
  NaN in lane => whole lane NaN (natural arithmetic, documented).
  Rank-0 (scalar) => Err (no axis).
0x3D GELU  exact: 0.5·x·(1+erf(x/√2)), erf via A&S 7.1.26.
0x3E SIGMOID  1/(1+exp(-x)) (f32 inf-arithmetic makes the tails exact).
0x3F TANH  x.tanh() (std).
0x40 RELU  max(x, +0.0): -0.0 normalizes to +0.0 (IEEE maxNum).
  NaN => +0.0 (f32::max rule — documented, not numpy).
0x41 EXP  x.exp() (overflow => +inf, defined).
0x42 LOG  x.ln() (0 => -inf, negatives => NaN; IEEE-defined, no trap).
0x43 CLIP  payload[0..4]=min f32, [4..8]=max f32 (both REQUIRED at
  assemble; no defaults — bare CLIP would silently zero everything).
  min > max => Err; non-finite bound => Err (clamp would panic).
  NaN value => max (f32::clamp rule, documented).
```

NaN policy summary (total functions, all documented): GELU/SIGMOID/
EXP/LOG/CLIP-value propagate NaN (plain arithmetic: `clamp` comparisons
are false for NaN, so it passes through); RELU maps NaN to +0.0
(`f32::max` rule); SOFTMAX poisons its lane.

Assembler:
```text
SOFTMAX rD, rT [AXIS=n] [TEMP=x]
GELU rD, rT
SIGMOID rD, rT
TANH rD, rT
RELU rD, rT
EXP rD, rT
LOG rD, rT
CLIP rD, rT MIN=x MAX=x
```
Strict lists gain exactly these keys. `TEMP=`/`MIN=`/`MAX=`/`VALUE=`
parse via f32 (inf/nan spellings accepted at parse; exec decides).

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats + one module).
`0x3C-0x43` previously errored at execute. No encoding touched.
v1.9 -> v1.10.

## Security Considerations

- No traps on values (except malformed parameters): every float input
  maps to a defined output — no panic path (`clamp` bounds validated;
  all indexing pre-bounded by construction).
- Fresh outputs only (GLOBAL; no aliasing, no PERSISTENT writes).
- `TEMP=+inf` is the only exotic-but-allowed input, and it is total
  (uniform) — documented, not a hole.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/activations.rs` (new): `erf_approx` (A&S, tested vectors),
  `gelu`/`sigmoid`/`softmax_lane` (stable, temp-explicit), unit tests
  (erf(1), gelu(1), sigmoid(0), uniform/known/temp/NaN lanes).
- `src/opcodes.rs`: `OP_SOFTMAX/GELU/SIGMOID/TANH/RELU/EXP/LOG/CLIP`,
  params + setters, `instr_*`, mnemonics, `parse_line` arms, roundtrips.
- `src/vm.rs`: 8 exec fns (thin over module + shared reader),
  dispatch arms, 8 counters.
- Conformance: per-op goldens (gelu(1)≈0.8413, softmax [1,2,3],
  temp shaping, clip bounds), NaN table, sparse/dtype/meta/temp/
  bounds/axis errors, assembler roundtrips + strict rejections,
  combined program (`vm::test_rfc0028_assembled_program_runs`:
  counters exact), `programs/activation_demo.m3asm` (all 8 at 1).
- Suite: `cargo test --lib` 320 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/activation_bench.rs` (softmax/gelu 1K-lane,
  per-instruction via `step_instruction`).

Follow-up (NOT this RFC): `KV_COMPRESS`/`FLASH_ATTN`/`ATTN_SPARSE`
(Fase 3 part 3 + TopK Lean bound); non-F32 dtypes; sparse activations;
fused ACT modes (`FFN FUSED_ACT`, `MATVEC FUSED_ACT` — DRAFT flags).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0028-00 | 2026-09-11 | DRAFT: activation family + v1.10 rule |
| 0028-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.10 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
