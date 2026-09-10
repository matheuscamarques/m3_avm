# RFC-0007 — Programability: LOADI/MOV + COMPARE Predicates

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x78,0x79 RESERVED->IMPL), W5-tail;
              COMPARE semantics (ESPEC §5.1 row, predicate field)
Obsoletes   : None
Feature Bit : none (control-plane completions)
Bump        : none (control-plane completions under the v1.4 line; next
              MINOR reserved for the next opcode family: universal W4 or
              cluster F1 — same rule as RFC-0005/0006)
```

## Abstract

Closes the immediate gap: no way to materialize a numeric literal into
a register (deadlines, lock ids, priorities, seeds, masks) and no
comparison except equality. Adds `LOADI 0x78` (u128 immediate),
`MOV 0x79` (register copy), and a predicate field on `COMPARE`
(EQ/NE/LT/LE/GT/GE, unsigned u128). No encoding changes to existing
ops: `LOADI` reuses the `imm_u128` payload convention
(`payload[0..16]`, precedent: `COMPARE`/`JUMP` immediates); the
predicate lives in `COMPARE payload[16]`, which is zero in every
previously assembled binary (= EQ, behavior preserved).

## Motivation

Found by the scenario audits: `SET_DEADLINE rDl, 80000000` cannot be
built because nothing puts 80000000 anywhere; `IF_LESS`/`IF_GREATER`
do not exist; lock ids and priorities need tensor round-trips today.
Demos and all future control flow depend on this.

## Specification

```text
0x78 LOADI rD, imm — rdest <- imm u128 (payload[0..16] LE).
     Assembler literals: decimal (80000000), hex (0xFF..). Negatives
     rejected with a clear error (regs are u128; fixed-point
     conventions belong to programs, documented in §Backwards).
0x79 MOV rD, rS — rdest <- reg[rS]. Trivial copy (aliasing aid,
     spill/fill, arg passing to the future CALL convention).
COMPARE PRED= — payload[16] = 0 EQ (default/legacy), 1 NE, 2 LT,
     3 LE, 4 GT, 5 GE; other values trap. Unsigned u128 semantics
     (regs hold addresses/ids/counters). cmp_equal carries the result;
     IF_EQUAL unchanged.
```

Float thresholds (the scenarios' `COMPARE r, 0.5`) stay unrepresentable
by design: programs use scaled integers (e.g. permille). A float-compare
mode is a future RFC, not silent coercion here.

Explicitly NOT in scope: register ALU (`ADD_IMM`/`INC` — counters still
need a follow-up RFC), `IF_LESS`/`IF_GREATER` as separate opcodes
(covered by PRED + IF_EQUAL).

## Backwards Compatibility

Additive: two new arms, one new payload byte read defaulting to legacy
behavior (`payload[16]==0` in all old binaries = EQ). `COMPARE` results
identical for every previously valid program.

## Security Considerations

- `LOADI` of arbitrary u128 into regs is the program's own literal —
  no new authority (regs were already writable via `SAMPLE`/hash ops).
- Predicate byte validated (6 values, else trap): no silent fallthrough
  to EQ on corrupt input.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_LOADI/OP_MOV`, `instr_loadi/instr_mov`,
  `compare_pred/set_compare_pred`, mnemonics, `parse_line` arms
  (`LOADI rD, imm`; `MOV rD, rS`; `COMPARE … PRED=`), roundtrips.
- `src/vm.rs`: `exec_loadi/exec_mov`, predicate branch in
  `exec_compare`, dispatch arms, 2 counters (`loadi_execs`,
  `mov_execs`; compare reuses its counter).
- Conformance: max-u128/hex/negative-reject, MOV chain, all 6
  predicates incl. boundaries (0, MAX), legacy-EQ preserved, demo
  `programs/loadi_demo.m3asm` exit 0.
- Suite: `cargo test --lib` 207 passed; sole failure is the
  pre-existing, unrelated `moshi::test_gguf_qkv_split_shapes`.

Follow-up (NOT this RFC): reg-ALU (`ADD_IMM`/`INC`, real counters),
float-compare mode, `SAMPLE TOPK` from register.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0007-00 | 2026-09-10 | DRAFT: LOADI/MOV + COMPARE predicates + demo |
| 0007-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
