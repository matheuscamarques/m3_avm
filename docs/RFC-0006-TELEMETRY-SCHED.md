# RFC-0006 — Telemetry + Scheduler Block (`0x6A–0x77`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.10 (0x6A-0x77 DRAFT->IMPL), W5
Obsoletes   : None
Feature Bit : none (core opcodes, v2.0-zone early implementation)
Bump        : none (v2.0-zone ops; v1.x line stays v1.4)
```

## Abstract

Implements the telemetry + scheduler block: six observability ops
(`CYCLES_COUNT/TRACE_EVENT/SANITY_CHECK/PREEMPT_CHECK/ASSERT/DUMP`)
and eight scheduler/sync ops (`YIELD/SET_DEADLINE/GET_DEADLINE/
PRIORITY_SET/PRIORITY_GET/LOCK/UNLOCK/FENCE`). Gives programs
self-observation (NaN absorption per actor, non-consuming preemption
poll, structured trace ring) and the first real deadline/sync
primitives the EDF direction needs — with honest limits (try-lock, not
blocking lock; compiler fence, not hardware fence).

## Motivation

`SANITY_CHECK` (absorb NaN/Inf at actor level) and `PREEMPT_CHECK`
were proposed as hardware audit ops as far back as `chat.md`;
`SET_DEADLINE` is the missing setter for the deadline field the T2
direction requires; `LOCK/UNLOCK/FENCE` are prerequisites for any
future `SHARED` region. All were reserved; none existed.

## Specification

All 32B. Scalars via regs (u128); tensors canonical f32-LE.

```text
0x6A CYCLES_COUNT rdest <- now_ns() (u64). "RDTSC-like": host clock,
                 metric-only (same caveat as all timestamps).
0x6B TRACE_EVENT rsrc1 = reg holding (event_id u64); rsrc2 = reg holding
                 data (u128). Appends (event,data) to a Vm ring
                 (cap 1024, oldest dropped). rdest unused.
0x6C SANITY_CHECK rD, rT [, rCount]: tensor sanitized CoW (non-finite
                 -> 0.0); rD = new tensor addr; rCount (if != 0xFF) =
                 sanitized element count (u64). Deterministic.
0x6D PREEMPT_CHECK rdest <- ctx.interrupt_flag as u64. Non-consuming
                 (unlike IF_INTERRUPT): pure poll for tight loops.
0x6E ASSERT Rs [, CODE=u16]: if reg == 0, trap Err("ASSERT <code>");
                 else Ok. Payload[0..2] = code (default 0).
0x6F DUMP (no operands; rdest/rsrc ignored): logs regs+pc+version+
                 pipeline+deadline of current ctx at info level.
0x70 YIELD: stats + scheduler.maybe_preempt() (a RED waiter preempts
                 a non-RED yielder). No operands.
0x71 SET_DEADLINE rsrc1 = reg holding absolute deadline (ns, u64).
                 Any value accepted; default u64::MAX (best-effort).
0x72 GET_DEADLINE rdest <- ctx.deadline.
0x73 PRIORITY_SET rsrc1 = reg holding 0/1/2 (GREEN/BLUE/RED), else Err.
                 Field swap only — the run loop re-enqueues exactly once
                 via yield_current (an explicit enqueue here would double
                 presence; fixed in RFC-0018 audit). maybe_preempt after.
0x74 PRIORITY_GET rdest <- priority as u64 (0/1/2).
0x75 LOCK rsrc1 = reg holding lock_id u32. TRY-lock (non-blocking):
                 free-or-self => acquire; held-by-other => Err Locked.
                 Table Vm::locks: id -> holder ctx. Reentrant for holder.
0x76 UNLOCK rsrc1 = reg holding lock_id. Held-by-self => release;
                 else Err (not-held / held-by-other).
0x77 FENCE: compiler fence SeqCst + stats. Visibility MARKER for the
                 future SHARED region; on this emulator no separate
                 store buffer exists, documented, not faked.
```

State: `Context.deadline: u64` (default `u64::MAX`); `Vm::locks:
HashMap<u32, u64>`; `Vm::trace: VecDeque<(u64, u128)>` cap 1024.

## Backwards Compatibility

Additive: one `u64` field with neutral default (all scheduling
decisions unchanged — strict Red>Blue>Green untouched, EDF still
future), two Vm collections born empty, 14 match arms, 14 counters.
v1.x line stays v1.4 (v2.0-zone early implementation, same rule as
RFC-0005).

## Security Considerations

- `LOCK` is try-only: no wait queues, hence no deadlock *through* the
  primitive (deadlock via lock ordering stays a program bug; documented).
- `ASSERT` traps are clean `Err` through the existing `?` chain (no
  partial state: failing instruction commits nothing).
- Trace ring is bounded (no flood vector); `DUMP` logs, never exposes
  other contexts' regs (only current ctx).
- `FENCE` is explicitly NOT a security boundary on this target.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/context.rs`: `deadline` field + default.
- `src/opcodes.rs`: 14 consts, `instr_*` ctors, ASSERT-code accessor,
  mnemonics, `parse_line` arms, roundtrips.
- `src/vm.rs`: `locks` + `trace` + cap, 14 exec fns, dispatch arms,
  14 counters.
- Conformance: per-op behavior tests (NaN absorb + count + CoW,
  non-consuming poll, ASSERT trap/code, deadline roundtrip, priority
  move + preempt, lock contend/release/error paths, fence), combined
  program (`vm::test_rfc0006_assembled_program_runs`, counters exact),
  `programs/telemetry_demo.m3asm` exit 0.
- Suite: `cargo test --lib` 202 passed; sole failure is the
  pre-existing, unrelated `moshi::test_gguf_qkv_split_shapes`.

Follow-up (NOT this RFC): blocking LOCK with wait queues + EDF
inheritance (needs scheduler wait-graph — the real T2 scheduler work);
`SHARED` region giving FENCE/LOCK a home; `TRACE_EVENT` export/sink.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0006-00 | 2026-09-10 | DRAFT: telemetry + scheduler block + demo |
| 0006-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green.
  Side-effect fix: pre-existing `mnemonic()` shadowing landmine
  (uncommitted `0x6A-0x77` arms compiled as bindings, mislabeling
  HALT/NOP/UNKNOWN); defining the consts restored correct dispatch. |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
