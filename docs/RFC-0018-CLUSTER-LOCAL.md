# RFC-0018 — Cluster F1: Local-Only `0x1A–0x1D` (No Transport)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC.md §5.4 (RSVD->IMPL-local), §19 (v1.4->v1.5);
              docs/ESPEC-V2.md §3 (0x1A-0x1D rows)
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.4 -> v1.5 cumulative: covers all opcode
              additions since v1.4 — 0004 through 0018. Intermediate
              minors were not cut per addition; v1.5 reconciles the line
              and minor-per-family resumes from here.)
```

## Abstract

Implements the four cluster opcodes with strictly local semantics
(`node_id == 0`): spawn contexts, signal (abort/kill/ping), tensor copy
(+ honest MOVE), and single-node barriers with timeout. Any nonzero
`node_id` fails loudly with a transport error — the wire format,
routing table, and driver remain F2+ (untouched by this RFC). This is
the plan's F1 milestone verbatim, plus two hardening decisions the plan
left open (MOVE source invalidation for real; barrier lazily-enforced
timeouts).

## Motivation

Distributed programs (spawn → barrier → signal) must be writable and
testable before any network code exists. Local execution validates
encodings, assembler forms, and coordination semantics; the transport
later moves bytes, not meanings.

## Specification

Payloads are exactly the frozen 26B layouts (no changes):

```text
0x1A REMOTE_SPAWN  [0..4]=node u32, [4..12]=entry_pc u64, [12]=prio.
  node==0: create context at entry_pc (validated via check_jump_target),
  fresh zeroed regs, Rd = ctx_id. node!=0: Err transport follow-up.
0x1B SIGNAL  flags bit0 = priority/interrupt; rsrc1 = kind reg
  (0=ABORT,1=FORK_REQ,2=HALT/KILL,3=PING); [0..4]=node, [4..12]=ctx_id,
  [12..20]=seq. rdest = status/ack.
  node==0: ABORT = terminate target + engine-stack pops (no heap
    restore — no version carried; use ABORT for that);
    HALT/KILL = terminate only; PING = ack 0, no side effects;
    FORK_REQ = explicit Err (use FORK; remote request flow is F3).
  node!=0 or kind>3: Err. rdest gets 0 on success.
0x1C SEND_TENSOR  rdest=src reg, rsrc1=dst reg (explícito; 0xFF
  reserva p/ F2 — destino fantasma seria mentira);
  [0..4]=node, [4..12]=offset u64, [12..16]=len u32, [16]=mode.
  node==0: COPY reads [offset, offset+len) of source tensor bytes and
    mirrors the window at the SAME offset of an explicit destination
    (honest partial assembly, rest preserved; undersized dst => Err,
    never truncation).
    MOVE = COPY + source invalidation for real (heap+meta removal;
    new MemoryManager::remove_tensor). Empty (len 0) = Err, not no-op
    (a zero-length transfer is a caller bug — fail loudly).
  node!=0: Err transport follow-up. Bulk stays out-of-instruction.
0x1D BARRIER  [0..4]=id u32, [4..6]=expected u16, [6..8]=timeout_ms u16,
  [8..12]=epoch. One-shot per id: arrival joins-or-creates
  {expected, arrived, deadline, epoch}; arrived>=expected releases all
  (unblock + log RELEASE) and deletes the entry; timeout (0 = none)
  enforced lazily at arrival (now > deadline => clear + Err NACK to the
  arriver + unblock waiters). Blocked contexts sleep (scheduler already
  skips Blocked). Mismatched expected/epoch on rejoin: Err (no silent
  merge of generations).
```

Assembler (numeric `NODE=` only; named nodes need the routing table,
F3 — explicit Err, not a hash guess):
```text
REMOTE_SPAWN NODE=0 ENTRY=label GREEN|BLUE|RED -> rD
SIGNAL NODE=0 CTX=rT KIND=ABORT|FORK_REQ|HALT|PING -> rD
SEND_TENSOR rS -> NODE=0 DEST=rD [OFF=n] [LEN=n] [MOVE]
BARRIER id=7 EXPECT=2 TIMEOUT=500 [EPOCH=n]
```

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats). `0x1A–0x1D`
previously errored at execute (`opcode não implementado`).

## Security Considerations

- No network surface added: nonzero node fails before touching any
  socket (there are none). Local-only cannot be confused into remote
  by a crafted `node_id` — the check is `== 0`, not a table lookup.
- `MOVE` invalidation is total within the VM (heap + meta); no dangling
  tensor survives to be resurrected.
- Barrier entries are one-shot and deleted on release/timeout: no
  cross-generation contamination by id reuse (epoch mismatch errors).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: 4 consts + kind/mode consts, payload accessors,
  4 ctors, mnemonics, parse arms, roundtrips.
- `src/memory.rs`: `remove_tensor` (heap+meta+sparse removal).
- `src/vm.rs`: `barriers` map, 4 exec fns, dispatch arms, 4 counters.
- Conformance: spawn-at-label (+zeroed regs, prio, transport/prio/
  entry errors), signal kill/ping/forkreq/kind/remote errors (HALT
  kills without stack pops), copy bit-exact + partial-preserve, MOVE total
  invalidation, all error paths, barrier two-party/wait/timeout/
  mismatch/empty, PRIORITY_SET single-presence regression,
  `programs/cluster_local_demo.m3asm` (4 counters at 1, verified).
- Suite: `cargo test --lib` 252 passed (sole failure: pre-existing
  unrelated moshi norm-gamma).
- Audit note: implementation caught a real RFC-0006 bug
  (PRIORITY_SET double-enqueue; fixed + doc corrected + regression
  test), a SEND_TENSOR 0xFF-dst hole (now explicit Err), and two
  requeue hazards in BARRIER (fixed before merge).

Follow-up (NOT this RFC): F2 driver/transport, routing table
(`NODE="name"`), RDMA hints, cyclic barriers, wait-graph timeouts
(real timers), remote `FORK_REQ` flow.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0018-00 | 2026-09-10 | DRAFT: local cluster ops + demo |
| 0018-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |
