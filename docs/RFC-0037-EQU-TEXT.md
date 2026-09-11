# RFC-0037 — Constantes `.equ` + `.text` no Assembler (V-1a, sem opcode)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : assembler §(assemble_with_base Passo 1, parse_imm_*, 8 use-sites); PLANO_VISAO V-1 (passo 2)
Obsoletes   : None
Feature Bit : none (assembler-only)
Bump        : none (no encoding touched — Trilha P precedent)
```

## Abstract

Immediates may now be named: `.equ RAG_TOPK 5` binds an integer
constant, and later uses in immediate positions (`LOADI r0, RAG_TOPK`,
`ADD_IMM r1, r0 IMM=RAG_TOPK`, `TENSOR r1 NR NC f32`, `TREES=`/`DEPTH=`,
`START=`/`LEN=`, `CODE=`, `COMPARE` second operand) assemble to the
same bytes as the literal. `.text` is accepted as a no-op section
marker (convention for V-1b). `.data`/`.str` fail with an explicit
error pointing at V-1b — they previously fell through to the generic
unknown-opcode error and still fail; nothing was loosened.

## Motivation

Vision listings are full of magic numbers (`TEACH_MODE`, `RAG_TOPK`,
timeouts, table dims) that must stay consistent across a file by hand —
itself a bug farm, same argument as RFC-0036 for registers. Named
constants fix that without touching a single encoding. Multi-value
init (`.data` contents, `.str` bytes) deliberately stays out: giving it
real runtime effect requires either a new opcode (`STORE`, which needs
a Trilha P R12 dossier first) or a loader sidecar (container + address
binding) — both are a separate decision (V-1b), not smuggled in here.

## Specification

```text
.equ <NAME> <valor>  — binds NAME (ident, case-insensitive, stored UPPER)
                        to a u128 (decimal or 0x-hex, unsigned, same domain
                        as LOADI). No instruction emitted. File-scoped,
                        order-independent (collected in Passo 1, like `.reg`).
.text                  — section marker, no operands, no instruction emitted.
                         Idempotent. Default section stays `.text`.
.data / .str           — RESERVED: hard error naming V-1b (not silently dropped,
                         not assembled).
```

- Name: `[A-Za-z_][A-Za-z0-9_]*`. Reserved (rejected at declare):
  `F32/F16/BF16/I8/U8/SPARSE/EOS_TOKEN` (dtype/keyword shadowing) and
  `R<digits>` (register shadowing — keeps `TENSOR r0 r1 r2` in the
  register form and `COMPARE`'s try-reg-first order intact).
- Value: `u128` decimal or `0x`-hex, no sign (same as LOADI).
- Fresh name required: redeclare → Err. Unknown name in any extended
  position → the site's legacy error verbatim (e.g. COMPARE keeps
  "segundo operando inválido"; LOADI keeps its hex/dec messages).
- Immediate positions extended (RFC-0037): `LOADI` (full literal-or-hex
  domain + const), `COMPARE` 2nd operand, `ADD_IMM`/`SUB_IMM` `IMM=`
  (decimal-or-const, no hex — domain unchanged), `TENSOR` rows/cols
  (decimal-or-const, `u64`-checked; `NxM` single-token form stays
  literal-only), `FOREST` `TREES=`/`DEPTH=` (`u16`-checked, existing
  range gate unchanged), `SLICE` `START=`/`LEN=` (`u32`-checked),
  `ASSERT` `CODE=` (`u16`-checked).
- NOT extended (literals as before, mechanical V-1b follow-up):
  `TOPK=`, `AXIS=`, `POS=/HDIM=/NHEADS=/THETA=`, `A=/B=`, `MEAN=/STD=`,
  `FILL=` (f32 — int consts don't apply), `DENSITY=`, `STREAM=/START=/
  LEN=` (telemetry), `D_INNER=/D_STATE=/LAYER=`, `SR=/FRAME=/HZ=/
  DELAY=`, `SIZE=/ALIGN=/ARENA=`, `ALPHA=/BETA=/MODE=` (floats),
  `CYCLES`/misc.
- Namespaces are positional (RFC-0036 precedent: allowed, documented):
  a word declared in both `.reg` and `.equ` resolves as register in
  reg positions, as constant in immediate positions. `R<digits>` and
  dtype/keyword reservations keep the ambiguous cases unreachable.
- Unknown dot-words other than the four above keep falling through to
  the existing unknown-opcode error.

## Backwards Compatibility

Pure assembler addition. Every previously-assembling program assembles
to byte-identical output: new code paths trigger only on `.equ`/`.text`
(which old programs don't contain) or on declared-const lookup (undeclared
names keep the legacy errors verbatim). Corpus gate 40/40 `.m3asm` via
CLI loop, including the new demo.

## Security Considerations

- Assembler-only: zero runtime surface, zero encoding change. Constants
  are validated at declare (domain + reservations) and range-checked at
  each use site (`u16`/`u32`/`u64` try_from — overflow is a loud error,
  never a truncation).
- Deterministic: `HashMap` lookups only, no iteration order dependence;
  same file → same bytes across calls (test-guarded).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `SymbolTable.consts` + `declare_const`/`resolve_const`/
  `is_const`; `.equ`/`.text` handling + explicit `.data`/`.str` V-1b
  error in Passo 1; `parse_imm_u128` / `parse_imm_dec_or_const` /
  `parse_dim_u64`; 8 const-aware use-sites (LOADI, COMPARE, ADD_IMM,
  SUB_IMM, TENSOR-dims, FOREST, SLICE, ASSERT).
- Conformance: `opcodes::test_rfc0037_equ_text` (bind/use in all 8 sites
  with byte-equality vs literals, hex, case-insensitivity, `.text`
  no-op, `.reg`+`.equ` coexistence, determinism, 20 error cases incl.
  range overflow, `.data`/`.str` V-1b error, TREES=0-via-const still
  rejected); RFC-0008 suite green UNCHANGED.
- Demo: `programs/equ_const_demo.m3asm` + `vm::test_planoV1_equ_consts`
  (r0=10, r1=42, r2=32, branch taken r4=1, TENSOR [2,2] FILL=0.5).
- Suite: `cargo test --lib` 359 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate 40/40 by CLI loop.

Follow-up (NOT this RFC — V-1b, decisão registrada: sidecar).
`.data` contents + `.str` bytes + multi-value `TENSOR` init via
assembler sidecar (`assemble_with_data`) + loader preload + address
binding — sem opcode novo, sem bump (`.data` é load-time, não
runtime). `STORE` segue possível um dia, mas só com dossiê R12
(Trilha P) e 2º caso de uso concreto; V-1b não precisa dele. A V-1b
ganha RFC própria com sua prova de estrito como esta.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0037-00 | 2026-09-11 | DRAFT: V-1a scope (`.equ`/`.text` + 8 imm sites; `.data`/`.str` deferred) |
| 0037-01 | 2026-09-11 | IMPLEMENTED: gate green; corpus 40/40; no bump |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
