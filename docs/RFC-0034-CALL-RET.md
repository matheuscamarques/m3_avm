# RFC-0034 — Subroutines: `CALL`/`RET` + Call Stack (`0x7D/0x7E`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.11 (0x7D,0x7E RESERVED->IMPL);
              docs/ESPEC.md §5.8, §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.13 -> v1.14: call control 0x7D-0x7E;
              0x7F stays RESERVED)
```

## Abstract

Implements real subroutines (RFC-0008 follow-up: "`CALL/RET` (need
ISA ops first)"): `CALL` pushes the return address and jumps,
`RET` pops back. Each context owns a `call_stack: Vec<u128>`
(max 1024 frames, loud overflow); `FORK` clones it (Unix semantics —
the child continues the call chain); `ABORT` needs nothing (dead
contexts take their stacks). Deliberately OUT: stack participation
in SNAPSHOT/RESTORE versioning — return addresses name immutable
code, so rewound data can never invalidate a pending `RET`; stacks
are control flow, not versioned state (registers precedent,
RFC-0004 follow-up). This unblocks structured `.m3asm` (nesting,
libraries of subroutines) without touching a single shipped encoding.

## Motivation

Every multi-step `.m3asm` program today is a flat `JUMP` soup with
copy-pasted blocks: reusable subroutines are inexpressible, which is
exactly why the vision docs all say `CALL` (and none assemble).
The missing piece was never the jump — it was the disciplined return.

## Specification

All 32B, control-flow (return `false` = no auto-advance, like `JUMP`).

```text
0x7D CALL  target: imm_u128 label PC (assembler resolves; validated
  by check_jump_target at execute like JUMP/FORK). Pushes pc+32
  (the CALL itself is always 32B — no width threading needed),
  sets pc=target. Depth guard: len >= 1024 => Err (no silent stack
  exhaustion; 16 KiB worst case per context, documented).
0x7E RET  no operands. Pops return PC into pc. Empty stack => Err
  ("RET sem CALL" — loud, never falls through into garbage).
```

State: `Context.call_stack: Vec<u128>` (default empty). `FORK`
clones parent→child alongside regs/RNG. `ABORT`/death discards.
`HALT` inside a subroutine terminates (no unwinding needed).
`SNAPSHOT`/`RESTORE` ignore stacks (immutable-code argument above).

Assembler (mirrors `JUMP`, labels-only):
```text
CALL LABEL
RET
```
Any trailing token on either => strict `Err` (`reject_unknown`, empty
known-set — same as `JUMP`). Unknown label => `Err` at assemble.

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats + one `Context`
field defaulting to empty). `0x7D/0x7E` previously errored at
execute. No encoding touched. v1.13 -> v1.14.

## Security Considerations

- Bounded stack (1024) kills the only new exhaustion vector at a
  known constant; overflow and underflow both trap loudly (no wrap,
  no fallthrough into attacker-influenced bytes — there are none,
  code is immutable, but loudness is the rule anyway).
- `CALL` targets validate exactly like `JUMP` (same function, no
  second standard).
- Fork-cloned stacks cannot corrupt the parent (vectors clone, no
  aliasing — `Vec<u128>` move semantics, no `Arc` sharing).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_CALL/RET`, `instr_call/ret`, mnemonics,
  `parse_line` arms (labels-only + strict), roundtrips.
- `src/context.rs`: `call_stack` field + `MAX_CALL_DEPTH = 1024`.
- `src/vm.rs`: `exec_call/ret`, dispatch arms (`Ok(false)`),
  FORK-clone line, 2 counters.
- Conformance: nesting (2 levels, exact regs), underflow `Err`,
  overflow at exactly 1025th nested `CALL` (1024 succeed),
  FORK-in-subroutine inheritance, `RET` resumes caller PC exactly,
  assembler roundtrips + strict rejections (incl. unknown label),
  combined program (`vm::test_rfc0034_assembled_program_runs`:
  counters exact), `programs/call_ret_demo.m3asm` (values verified).
- Suite: `cargo test --lib` 354 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/alu_bench.rs` gains call/ret pair.

Follow-up (NOT this RFC): `.data/.equ`/literals (assembler-only,
pairs with this — tables for `DEPFORMER`-class ops); closures/
upvalues (needs design, not just a stack); `0x7F` stays RESERVED.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0034-00 | 2026-09-11 | DRAFT: subroutines + stack discipline |
| 0034-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.14 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
