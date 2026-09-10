# RFC-0011 — Rollback Robustness: Version-Tagged Stacks + Host Pruning

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : FORK/ABORT engine-state discipline (no encoding change);
              host checkpoint policy in `main.rs`
Obsoletes   : None
Feature Bit : none
Bump        : none (hardening under the v1.4 line)
```

## Abstract

Engine-state rollback (`ssm_states`, `rank1_layers`) rides LIFO stacks
pushed by `FORK` and popped unconditionally by `ABORT`. Two defects:
(1) an `ABORT` naming an old version pops the newest entry regardless —
restoring the wrong generation; (2) a second `ABORT` (or an `ABORT`
without `FORK`) consumes entries belonging to outer forks. This RFC
tags every pushed entry with the memory version at `FORK` time and
gates pops on the abort target; `ts=0` keeps the exact legacy
single-pop behavior. It also bounds the host-side `checkpoints` vec in
`main.rs` to the same k=16 window as retention (RFC-0003).

## Motivation

Nested forks with a single outer abort restore inner state instead of
outer state today; double-abort eats the outer fork's entry. Both are
silent. The memory heap does not have this bug (version-keyed maps +
I-Mono); engine stacks should match that discipline.

## Specification

1. `ssm_snapshots: Vec<(u64, Vec<MambaState>)>`,
   `rank1_snapshots: Vec<(u64, HashMap<u8, u128>)>`; `FORK` pushes
   `(snap_version, clone)` where `snap_version` is the `memory.snapshot()`
   id taken by that same `FORK`.
2. `ABORT` with `ts_version != 0`:
   a. discard (pop, drop) while top.version > ts;
   b. if the new top has version == ts, pop and apply it;
      otherwise apply nothing (no engine snapshot at-or-below ts —
      current state kept; documented, not silent: logged).
3. `ABORT` with `ts_version == 0`: pop once and apply if present
   (legacy best-effort, byte-identical behavior to before).
4. Memory `restore(ts)` failure changes nothing about (2): stack
   gating keys on versions, not on restore success (logged either way,
   as today).
5. Host: `rollback::prune_oldest(&mut checkpoints, 16)` after every
   push in `main.rs` (3 sites). Same K as retention — a host entry older
   than the window may already be evicted server-side (clean `Err`,
   already handled); pruning only bounds host memory. Generic helper,
   unit-tested in lib (`rollback.rs`).

## Backwards Compatibility

`ts=0` path is instruction-for-instruction the old behavior (all
current programs and demos use it). Version-gated path only triggers on
nonzero `ts`, which previously popped blindly — any program depending
on the blind pop with explicit versions was depending on the bug.

## Security Considerations

- No new authority: tags are internal bookkeeping, never read from
  bytecode (version comes from the aborting program's register as
  before; a forged ts can only select among already-pushed entries,
  never fabricate state).
- Bounded stacks in practice (one entry per unmatched FORK); full
  retention policy for engine stacks stays a follow-up (entries live
  until consumed, same as before — no worse).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/vm.rs`: tagged stack types, `FORK` push with version, gated
  `ABORT` pop, unchanged `ts=0` path.
- `src/rollback.rs`: `prune_oldest` (+ unit test); `src/main.rs`: 3
  call sites.
- Conformance: nested version gating (inner dropped, outer applied,
  stacks drained), abort-without-fork no-op, double-abort stability,
  rank1 map gating + CoW intactness, `prune_oldest` bounds, full suite
  218 passed (sole failure: pre-existing unrelated moshi norm-gamma).

Follow-up (NOT this RFC): engine-stack retention bounds, `ABORT` with
ts naming a non-fork memory snapshot (engine state left as-is — see
§2b), unifying host `Checkpoint` structs (two local definitions in
`main.rs`).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0011-00 | 2026-09-10 | DRAFT: version-tagged stacks + host pruning + tests |
| 0011-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
