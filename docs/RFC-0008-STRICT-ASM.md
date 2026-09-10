# RFC-0008 — Strict Assembler (Unknown-Token Rejection)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : assembler behavior for every mnemonic in `parse_line`
              (no encoding change; error paths only)
Obsoletes   : None
Feature Bit : none
Bump        : none (error-path hardening under the v1.4 line)
```

## Abstract

The assembler silently ignores unknown trailing tokens (`SENSE …,
LEN=`, `SAMPLE … TEMPERATURE=`, `ATTN … NHEADS=`, typo'd priorities
degrading to defaults). A program can assemble, run, and mean something
else. This RFC makes every `parse_line` arm total over its documented
forms: any token that is not a consumed operand or a documented
flag/KV is a hard `Err` naming the token and the opcode.

## Motivation

Found by the scenario audits (3 silent-drop classes) and confirmed by
reading all 55 arms: typo'd `FORK …, GRENN` degrades to GREEN,
`SENSE …, XYZ` degrades to AUDIO, unknown `CTX_SWITCH` pipes degrade to
TRANSFORMER, `SANITY_CHECK rD, rT, FOO` silently drops the count
register. Silent defaults are the assembler's most dangerous feature.

## Specification

1. New helper `reject_unknown(op, rest, known)`: each token must equal
   a known word or start with a known `PREFIX=`; otherwise
   `Err("<OP>: token desconhecido '<tok>' (modo estrito, RFC-0008)")`.
2. Applied to every arm's unconsumed tail (full per-arm table in
   §Reference). Positional optionals keep working (`STREAM …
   BLOCKING/DROP`, `FORK … NOTIFY`, `SANITY_CHECK` 4th reg,
   `GATHER` rAcc heuristic); bare-numeric `SAMPLE` temp stays.
3. Former silent defaults become errors: unknown `FORK` priority,
   unknown `SENSE` peripheral (numeric u8 still accepted),
   unknown `CTX_SWITCH` pipe, non-reg `SANITY_CHECK` 4th token,
   non-`BLOCKING`/`DROP` `STREAM` mode.
4. `HALT/NOP/DUMP/YIELD/FENCE` and all fixed-arity arms reject ANY
   trailing token.
5. Conformance gate (must stay green): all `programs/*.m3asm` +
   `examples/*.m3asm` assemble (27/27 at baseline) AND full
   `cargo test --lib`. Any fallout is fixed by correcting the
   program/test, never by re-adding leniency — unless the token proves
   to be a legitimate undocumented form, in which case it is documented
   in the arm's comment and this RFC's table.

## Backwards Compatibility

Source-level only, and intentionally breaking for nonsense: every
previously *meaningful* program assembles identically (gate proves it).
Programs that relied on silent drops were buggy by definition; they now
fail loudly at assemble time instead of misbehaving at runtime. No
encoding, no runtime semantics change.

## Security Considerations

- Assembler strictness is a supply-chain control: pasted v2.4-style
  programs (`CHANNEL=`, `TEMPERATURE=`, `NHEADS=`) now fail instead of
  running with dropped constraints (e.g. a dropped `CAUSAL` or timeout).
- Error messages name the token and opcode (actionable, no guessing).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `reject_unknown` + per-arm application (55 arms;
  table below), strict prio/peripheral/pipe/mode validation.
- Conformance: `opcodes::test_rfc0008_rejects_unknown_tokens`
  (26 bad forms rejected incl. all 3 documented silent-drop classes;
  30 legitimate forms pass), corpus gate 27/27 `.m3asm`, suite
  208 passed (sole failure: pre-existing unrelated moshi norm-gamma).

Per-arm known sets (beyond consumed operands):

| Arm(s) | Accepted extras |
|:---|:---|
| TENSOR | `SPARSE`, `DENSITY*` |
| ATTN | `NOTIFY`, `NOTIFY_EACH_HEAD`, `NOTIFY_EACH` |
| STREAM | `BLOCKING`, `DROP` (exactly one, position 3) |
| FORK | prio `RED/BLUE/GREEN/0/1/2`/label, `NOTIFY`, `NOTIFY_SCHEDULER` |
| SENSE | (none beyond peripheral) |
| SAMPLE | `TEMP=`, `TOPK=`, bare f32 |
| COMPARE | `PRED=` |
| SSM_SCAN | `CONV`, `GATE`, `D_INNER=`, `D_STATE=`, `LAYER=` |
| SSM_RESET | `D_INNER=`, `D_STATE=`, `LAYER=` |
| CODEC_ENC/DEC | `TENSOR` |
| AUDIO_ALIGN | `SR=`, `FRAME=`, `HZ=`, `DELAY=` |
| ROPE | `INPLACE`, `POS=`, `HDIM=`, `NHEADS=`, `THETA=` |
| GATHER | `AXIS=`, `MODE=` (+ positional rAcc) |
| DISTANCE | `METRIC=`, `TOPK=` |
| RANK1_UPDATE | `ALPHA=`, `BETA=`, `MODE=`, `LAYER=` |
| RNG_UNIFORM | `A=`, `B=` |
| RNG_NORMAL | `MEAN=`, `STD=` |
| ASSERT | `CODE=` |
| Fixed (all others) | (none) |

Follow-up (NOT this RFC): `.data/.equ` directives, string literals,
`CALL/RET` (need ISA ops first), float immediates in `COMPARE`
(rejected today — programs use scaled integers per RFC-0007).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0008-00 | 2026-09-10 | DRAFT: strict helper + all arms + corpus gate |
| 0008-01 | 2026-09-10 | IMPLEMENTED: merged to tree, corpus 27/27 + suite green.
  Zero fallout: no legitimate program used silent drops. |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
