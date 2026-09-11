# RFC-0026 — Control-Plane ALU: `ADD_IMM`/`SUB_IMM`/`STEPS` (`0x7A/0x7B/0x7C`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.11 (0x7A,0x7B,0x7C RESERVED->IMPL);
              docs/ESPEC.md §5.8, §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.7 -> v1.8: control-plane family opens 0x7A-0x7F;
              0x7D-0x7F stay RESERVED)
```

## Abstract

Implements the first control-plane ops in the `0x7A-0x7F` gap:
register-immediate arithmetic (`ADD_IMM`, `SUB_IMM`, wrapping u128)
and a deterministic progress counter (`STEPS`). `SUB_IMM` closes the
mostDerivado gap in the ISA — subtraction previously required an
`ADD`+`COMPARE` construction (cf. the timeout code in
`PAUSA_DE_PENSAMENTO.md` §5). `STEPS` reads retired-instruction count,
giving `.m3asm` programs a replay-exact clock where `CYCLES_COUNT`
(`0x6A`) is deliberately wall-clock (RFC-0005 determinism-first:
wall time never drives program logic). All three are pure-register,
stateless, total functions — no memory, no traps except missing regs.

## Motivation

Per the ESPEC Section 13 filter: counters, deadlines and indices are
currently built from `LOADI` constants plus full-tensor `ADD`
(allocating nothing, but conceptually wrong layer), and there is NO
subtraction at all — every `now - last` in every spec sketch is
unwritable as drawn. Three tiny ops fix the layering: arithmetic where
arithmetic belongs, and a deterministic counter next to the
non-deterministic one, so programs can choose explicitly.

## Specification

All 32B, stateless (S). Immediates reuse the `imm_u128` convention
(payload`[0..16]`, like `COMPARE`/`JUMP`).

```text
0x7A ADD_IMM  rdest, rsrc1, IMM=n — rdest <- rsrc1 wrapping_add n.
  Total: never traps on values (wraps, documented — same rule as PC
  advance). `0xFF` operands => Err (must name real registers).
0x7B SUB_IMM  rdest, rsrc1, IMM=n — rdest <- rsrc1 wrapping_sub n.
  Same totality rule. This is the ONLY subtraction in the ISA.
0x7C STEPS  rdest — rdest <- retired instruction count (u64, as u128).
  Reads COMPLETED count (the loop increments post-execute): the value
  observed equals the number of previously retired instructions on
  this VM instance — deterministic for a given program path (unlike
  CYCLES_COUNT). No payload. Missing rdest (0xFF) => Err.
```

Wrapping is specified, not silent: `u128::MAX + 1 == 0` and
`0 - 1 == u128::MAX`, by definition here (matches `advance_pc`).

Assembler:
```text
ADD_IMM rD, rS, IMM=n (decimal u128; "-5" is Err — use SUB_IMM)
SUB_IMM rD, rS, IMM=n
STEPS rD
```
Strict lists gain exactly `IMM=`; anything else is `Err`.

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats). `0x7A/0x7B/0x7C`
previously errored at execute. No encoding touched. v1.7 -> v1.8.

## Security Considerations

- Wrapping arithmetic is total: no overflow trap to weaponize, no
  panic path (all `u128` ops are `wrapping_*`, never `+`/`-`).
- `STEPS` exposes only a counter (no memory, no timing side-channel
  beyond what `CYCLES_COUNT` already gives).
- No new state, no new exhaustion vector (zero allocation).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_ADD_IMM/SUB_IMM/STEPS`, `*_imm` accessors
  (reuse `imm_u128`/`set_imm_u128`), `instr_*` ctors, mnemonics,
  `parse_line` arms, roundtrips.
- `src/vm.rs`: `exec_add_imm/sub_imm/steps`, dispatch arms, 3 counters.
- Conformance: wrapping goldens (`MAX+1==0`, `0-1==MAX`, mixed),
  STEPS exact-count program (deterministic: `r0==1`, `r1==3` — the
  increment happens post-execute, documented), missing-reg errors,
  IMM-required/negative/bad-token rejections, combined program
  (`vm::test_rfc0026_assembled_program_runs`: counters exact),
  `programs/alu_demo.m3asm` (counters at 1, verified).
- Suite: `cargo test --lib` 305 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/alu_bench.rs` (add_imm/sub_imm/steps,
  per-instruction via `step_instruction`).

Follow-up (NOT this RFC): CALL/RET + call stack + `.data/.equ` +
string literals (RFC-0008 follow-ups, needs ISA design — own RFC);
`0x7D-0x7F` remain RESERVED; generic `COUNTER ID=` (only STEPS exists
— new counters need a registry RFC, not ad-hoc ops).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0026-00 | 2026-09-11 | DRAFT: control-plane ALU + deterministic counter |
| 0026-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.8 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
