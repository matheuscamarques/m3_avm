# RFC-0036 — Apelidos Simbólicos no Assembler (`.reg`, sem auto-alocação)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : assembler §(parse_reg/assemble_with_base); PLANO_VISAO V-1 (passo 1)
Obsoletes   : None
Feature Bit : none (assembler-only)
Bump        : none (no encoding touched — Trilha P precedent)
```

## Abstract

Registers may now be named: `.reg rTranscript r0` binds an alias,
and later uses of `rTranscript` assemble to `r0`. Declaration is
EXPLICIT and mandatory — use-without-declare is a hard error naming
`.reg`. Auto-allocation was designed, prototyped, and DELIBERATELY
REJECTED in the same turn: it turned RFC-0008's conformance case
`SANITY_CHECK r4, r0, FOO` from a loud error into a silent `r2`
binding (proven by the failing gate before the revert). Strictness
outranks convenience in this assembler; typos must fail.

## Motivation

Vision programs are unreadable and unmaintainable with bare `r0–r15`
(`FOREST rOut, rFeatures, rXgb` vs `FOREST r5, r1, r9`), and hand
tracking of 16 registers across hundred-line programs is itself a
bug farm. Names fix readability without touching a single encoding.

## Specification

```text
.reg <name> <rN>   — binds name (r-prefixed identifier) to physical
                     reg 0–15. No instruction emitted. Case-insensitive.
```

- Name: `r` + `[A-Za-z_][A-Za-z0-9_]*` (stored lowercase). Anything
  else (`r1x`, bare `r`, `rfoo-bar`) errors exactly as before.
- Physical: numeric `r0–r15` only (parsed by the numeric path, so
  `r16`/`rr1`-style errors are unchanged).
- Fresh name + fresh physical required: redeclare → Err; occupied
  physical → Err (no implicit aliasing); unknown physical form → Err.
- Use of an undeclared symbolic in any reg position → Err suggesting
  `.reg` (keeps the RFC-0008 gate green verbatim — its 26 bad forms
  still reject, suite-guarded).
- One shared namespace, deterministic per file, fresh table per
  `assemble*` call. Mixing numeric and symbolic may alias (e.g. a
  hypothetical `rS1` lands on physical `r1`): allowed, documented,
  style guide says pick one per file. 16 GPR normative (R5) — this
  changes nothing about register count, ranges, files, or container.
- Unknown dot-words (`.data`, `.foo`) keep falling through to the
  existing unknown-opcode error (nothing loosened for Fase V-1's
  remaining directives).

## Backwards Compatibility

Pure assembler addition. Every previously-assembling program assembles
to byte-identical output (no new tokens in old programs match the
`.reg` gate; numeric path untouched including quirks). Corpus gate
verifies by CLI loop.

## Security Considerations

- Assembler-only: zero runtime surface, zero encoding change, nothing
  for an attacker to target beyond what exists. Deterministic mapping
  (no hash iteration order — `HashMap` lookups only, allocation order
  is program order).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `SymbolTable` (declare/resolve, no auto-alloc),
  `parse_reg` threaded with `&SymbolTable`, `.reg` handling in
  Passo 1 (emits nothing), `parse_line` signature extended (all
  call sites internal — no public API change).
- Conformance: `opcodes::test_rfc0036_symbolic_regs` (bind/use/reuse,
  case-insensitivity, determinism across calls, redeclare/occupied/
  out-of-range/malformed directive errors, unknown-symbol error with
  `.reg` hint, directive emits no instruction, legacy invalid forms
  still error); RFC-0008 suite green UNCHANGED (the load-bearing
  property).
- Suite: `cargo test --lib` 357 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop
  (byte-identical programs unaffected).

Follow-up (NOT this RFC): remaining V-1 assembler track (`.data`/
`.equ`/`.str`/`.text`, literals, multi-value init lists); each with
its own strictness proof like this one.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0036-00 | 2026-09-11 | DRAFT: symbolic regs (auto-alloc first, then rejected) |
| 0036-01 | 2026-09-11 | IMPLEMENTED: declare-only; gate green; no bump |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
