# RFC-0037 — Diretivas de Assembler (`.equ`/`.text`/`.data`, sem opcode)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : assembler §(assemble Passo 1/1b, parse_imm_*, 8 use-sites, DataSection, layout, `@`); loader §(load_assembled); PLANO_VISAO V-1 (passos 2–4)
Obsoletes   : None
Feature Bit : none (assembler-only)
Bump        : none (no encoding touched — Trilha P precedent)
```

## Abstract

Immediates may now be named: `.equ RAG_TOPK 5` binds an integer
constant, and later uses in immediate positions (`LOADI r0, RAG_TOPK`,
`ADD_IMM r1, r0 IMM=RAG_TOPK`, `TENSOR r1 NR NC f32`, `TREES=`/`DEPTH=`,
`START=`/`LEN=`, `CODE=`, `COMPARE` second operand) assemble to the
same bytes as the literal. `.text` is accepted as a section marker
(idempotent; default section stays `.text`). `.data` opens the data
section (V-1b dia 1: `.u32`/`.i32`/`.f32` escalares + `.str` UTF-8,
sidecar via `assemble_with_data`); multi-value nested dia 2
(`.f32 [R] [v...]` / `.f32 [R, C] [v...]`, shape explícito).
`.str` nu erra (é tipo de blob, não diretiva).

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
.data                  — opens the data section (V-1b dia 1), no operands,
                         no instruction emitted. Idempotent; `.text`
                         switches back. Blob lines, one per line:
                           <nome>: .u32 <dec/0x-hex/.equ>  — 4B LE
                           <nome>: .i32 <dec/.equ>         — 4B LE (sinal ok)
                           <nome>: .f32 <literal>          — 4B LE (só literal;
                                                              const int→float
                                                              recusada: perda
                                                              silenciosa)
                           <nome>: .str "<utf8>"           — bytes crus, sem NUL
                         Multi-valor nested dia 2, só `.f32`, tudo numa linha:
                           <nome>: .f32 [N] [v, ...]       — vetor rank-1
                           <nome>: .f32 [R, C] [v, ...]    — matriz rank-2
                         `.str` nu (fora de blob) erra.
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
- `.data` dia 1: blob `nome` é ident (`[A-Za-z_][A-Za-z0-9_]*`,
  guardado em minúsculas); duplicado → Err; colisão com rótulo de
  código → Err (namespaces separados, confusão recusada). `.u32`/`.i32`
  aceitam literal ou `.equ` (faixa checada, nunca truncada); `.f32`
  recusa `NaN`/`Inf` (bounds assumem finitos); `.str` sem escapes no
  dia 1 (aspas internas recusadas) e sem `;`/`#` (a linha passa por
  `strip_comment` antes do parse — dia 2 trata aspas antes do corte).
  Diretivas `.reg`/`.equ` valem no arquivo todo, independente de seção
  (blobs parseados no passo 1b — mesmo precedente de forward-reference
  do `.reg`).
- `.data` dia 2 (nested, só `.f32`): shape explícito e obrigatório
  (`.f32 [...]` sem shape erra — sem inferência); rank 1–2 (rank 3+
  erra); dims decimais/0x-hex/`.equ`, não-zero; valores separados por
  vírgula (trailing/leading/dupla vírgula erra); `count == prod(shape)`
  ou erro `esperados N, obtidos M`; `prod(shape) ≤ u32::MAX` no parse;
  floats finitos; `[[...]]` estilo-JSON recusado; nada após o bloco de
  valores (sem mistura flat/nested). `DataBlob.shape: Vec<u32>`
  (vazio = escalar dia 1).
- `.data` dia 3 (Opção A, assemble-time): `@nome` vira o endereço GLOBAL
  literal num único `LOADI` u128 (0x78 comporta qualquer endereço —
  verificado: sem `LOADI64`, sem dança de dois imediatos, sem bailout).
  `@` SÓ em `LOADI` (fora dela, erro legado); `@` SÓ resolve blob
  `.data` (`@` de `.equ`/inexistente/nú => erro alto); namespace `nome`
  compartilhado `.equ`×`.data` (colisão = erro — evita confusão
  valor-vs-endereço).
- Layout congelado (função única assembler↔loader): base
  `DATA_LOAD_BASE = GLOBAL_HEAP_START = 0x1000`; `start=align(off,tipo)`
  (escalares/blobs f32 = 4, `.str` = 1); próximo blob em
  `align8(start+len)` (pad 8 entre blobs; total inclui pad final); LE
  sempre. Correção registrada: GLOBAL é `0x00` (top byte) — endereço =
  offset puro, NÃO `(region<<60)|offset`.
- `Vm::load_assembled(&AssembledProgram)`: preload como PRIMEIRA
  alocação GLOBAL (`alloc_global(total)`); base retornada precisa ser
  `DATA_LOAD_BASE` senão erro alto (VM não-fresca ou 2º preload —
  endereços `@` foram congelados no assemble). GLOBAL não é read-only
  (CoW); sem enforcement novo (YAGNI).
- Bit `M3BC_OPTIONAL_HAS_DATA_SECTION = 1<<49`: NÚMERO CONGELADO,
  wiring pendente — o container ainda não carrega payload de dados
  (CLI `assemble` rejeita `.data`), então ligar o bit hoje seria
  mentira documentada. Trava de valor em teste; wiring com o layout.
- `assemble()` com blobs → Err nomeando `assemble_with_data` (nada
  descartado em silêncio); `assemble_with_data()` retorna
  `AssembledProgram { instrs, data }`. `instrs` valem para
  `load_program` hoje; `data` aguarda o preload do dia 3 (sem binding,
  sem efeito em execução).
- Unknown dot-words other than `.reg`/`.equ`/`.text`/`.data` keep
  falling through to the existing unknown-opcode error.

## Backwards Compatibility

Pure assembler addition. Every previously-assembling program assembles
to byte-identical output: new code paths trigger only on `.equ`/`.text`/
`.data` (which old programs don't contain) or on declared-const lookup
(undeclared names keep the legacy errors verbatim). Corpus gate 40/40
`.m3asm` via CLI loop, including the new demo.

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
- Dia 1 (V-1b): `DataSection`/`DataBlob`/`DataDtype`/`AssembledProgram` +
  `assemble_with_data()`; seção `.data` com passo 1b; `parse_data_line`
  (`.u32`/`.i32`/`.f32`/`.str`).
- Conformance dia 1: `opcodes::test_v1b_data_scalar_str` (bytes LE
  bit-exatos, hex/`.equ`/negativo, `.reg`+`.equ` cross-seção,
  idempotência de seção, 20 casos de erro incl. overflow de faixa,
  `NaN`/`Inf`, `[...]`→dia 2, colisão blob×rótulo); `assemble()` com
  `.data` erra pedindo `assemble_with_data`; RFC-0008 green UNCHANGED.
- Suite dia 1: `cargo test --lib` 360 passed (sole failure: pre-existing
  moshi); corpus 40/40.
- Dia 2: `parse_nested_f32` + `DataBlob.shape`; conformance
  `opcodes::test_v1b_data_nested` (bytes LE bit-exatos, whitespace/hex/
  `.equ` em dims, 20 casos incl. mismatch `esperados N, obtidos M`,
  zero, rank-3, vírgulas, `[[...]]`, `prod > u32::MAX`).
- Dia 3: `data_layout_addrs`/`data_layout_total` + `SymbolTable.addrs`
  + `@` no `LOADI` + `MemBackend::alloc_global` + `Vm::load_assembled`
  + `M3BC_OPTIONAL_HAS_DATA_SECTION`; conformance
  `opcodes::test_v1b_data_addr` (layout exato, byte-igualdade vs
  literal, 10 erros incl. `@` de `.equ`, colisão blob×const) +
  `vm::test_planoV1b_data_loader` (regs + bytes lidos de volta +
  guarda de 2º preload/VM suja) + demo `programs/data_addr_demo.m3asm`.
- Suite dia 3: `cargo test --lib` 362 passed (sole failure: pre-existing
  moshi); corpus 40/40.

Follow-up (fora desta RFC): layout `.data` no container `.m3bc` (aí o
bit 49 liga de verdade) e, depois de V-1b, `src/mimi.rs` antes de V-2
(RAG é bump de ISA; Mimi é implementação sobre `0x15`/`0x16` já
existentes).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0037-00 | 2026-09-11 | DRAFT: V-1a scope (`.equ`/`.text` + 8 imm sites; `.data`/`.str` deferred) |
| 0037-01 | 2026-09-11 | IMPLEMENTED: gate green; corpus 40/40; no bump |
| 0037-02 | 2026-09-11 | V-1b dia 1: `.data` escalar/`.str` + `assemble_with_data` (sem loader); `[...]`→dia 2 |
| 0037-03 | 2026-09-11 | V-1b dia 2: multi-valor nested `.f32 [shape] [vals]` (só explícito, ranks 1–2) |
| 0037-04 | 2026-09-11 | V-1b dia 3: `@nome` (Opção A) + layout congelado + `load_assembled`; bit 49 congelado, wiring pendente |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
