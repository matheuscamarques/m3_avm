# RFC-0002 — Dual-Mode Decoder and Width Errors

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §4 (decoding half of W1)
Obsoletes   : None
Feature Bit : 1 (DUAL_MODE, claimed on acceptance per RFC-0001 §7)
Bump        : none (decoder hardening; no new executable semantics, v1.3 unchanged)
```

## Abstract

Implements the decoding half of ESPEC-V2 W1: a total width function over
all 256 opcodes, clean rejection of 64B/escape/reserved widths at every
32B decode boundary (instead of silent misdecode), and validated bounds
for ESCAPE `ext_len`. Execution of 64B instructions (fetch-stride
changes in the executor) is explicitly OUT of scope and tracked as a
follow-up.

## Motivation

The v1.x decoder (`Instruction::decode`) reads any 32 bytes as an
instruction. A single `>=0x80` byte at a chunk start would misdecode
into garbage registers and payload rather than fail loudly — the exact
class of silent corruption the honesty contract forbids. Before any
wave emits wider encodings, the decoder must know every width and
refuse what it cannot execute.

## Specification

1. `instr_width(op: u8) -> InstrWidth` (`src/opcodes.rs`): total
   function. `<0x80` and `0xFF` map to `Fixed32` (`0xFF` is the sole
   R2 exception); `0x80-0xAF` and `0xB4-0xB6` to `Fixed64`; `0xB0` to
   `Fixed64` (head; total via rule 3); `0xB1`/`0xB2` to
   `Fixed128`/`Fixed256`; `0xB3` to `EscapeVar`; `0xB7-0xFE` to
   `Reserved`.
2. `fixed_size(w) -> Option<usize>`: 32/64/128/256; `None` for
   `EscapeVar`/`Reserved` (no length ever assumed for reserved).
3. `escape_total_len(ext_len: u64)`: valid iff `>= 64`, multiple of 8,
   `<= 1 MiB` (`EXT_MAX_LEN`, from RFC-0001 §9.1). Otherwise
   `InvalidExtLen`.
4. `Instruction::decode` (32B v1.x entry point): after the length
   check, opcodes that are not `Fixed32` are rejected — `UnsupportedWidth`
   for 64B/escape widths, `ReservedOpcode` for `0xB7-0xFE`. The
   `Incomplete` path (short input, including empty) is unchanged.
5. All existing `decode` callers (`Vm::load_bin`, assembler/disassembler
   chunk loops in `src/main.rs`) propagate the new variants via `?`
   with zero call-site changes; frozen-but-unimplemented opcodes
   (`0x1A-0x25`, all `<0x80`) still decode and are rejected later at
   execute, exactly as before.

## Backwards Compatibility

Purely additive refusal: every byte sequence the old decoder accepted
with `op < 0x80` or `op == 0xFF` decodes identically. Sequences with
`op >= 0x80` (except `0xFF`) previously misdecoded and now fail loudly
— that is the intended fix, and no shipped program or test contains
such bytes (full suite green, Section "Reference Implementation").

## Security Considerations

- Truncated input still yields `Incomplete` before any width logic
  (no out-of-bounds read; empty input safe).
- `ext_len` is validated BEFORE any allocation or skip (`<=1MiB`,
  alignment, floor), closing the length-bomb vector from RFC-0001 §9.1.
- Reserved range assumes no length: a future decoder must trap before
  reading past byte 0.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `INSTR_SIZE_64`, `EXT_MAX_LEN`, `InstrWidth`,
  `instr_width`, `fixed_size`, `escape_total_len`, new
  `DecodeError::{UnsupportedWidth, ReservedOpcode, InvalidExtLen}`,
  `decode` wiring.
- Conformance tests (all green): `opcodes::test_instr_width_full_map`
  (total over 256 opcodes), `opcodes::test_decode_width_rejection`
  (64B/escape/reserved rejected; `0x00/0x19/0x1A/0x25/0xFF` decode;
  truncation preserved), `opcodes::test_escape_total_len_bounds`.
- Suite: `cargo test --lib` 177 passed; sole failure is the
  pre-existing, unrelated `moshi::test_gguf_qkv_split_shapes`
  (norm-gamma assertion on the local PersonaPlex GGUF; fails
  identically without this change).

Follow-up (NOT this RFC): 64B fetch stride in `Vm` (`pc_to_index` and
the execute loop assume 32B), 64B `encode`, `.m3bc` loader. Tracked
under ESPEC-V2 W1-remainder / W10.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0002-00 | 2026-09-10 | DRAFT: width function + rejection + tests |
| 0002-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
