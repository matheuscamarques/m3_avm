# RFC-0003 — Snapshot Retention Window (k=16)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC.md §6.3 item 3; docs/ESPEC-V2.md W2 (retention half)
Obsoletes   : None
Feature Bit : none (local policy, nothing to negotiate)
Bump        : MINOR (behavioral contract change, no encoding change)
```

## Abstract

Bounds snapshot memory to a sliding window (default k=16, configurable,
`0` = explicit unbounded opt-out). On every `snapshot()`, versions older
than the window are evicted from all snapshot maps (four at writing; five since RFC-0016); `restore()` of
an evicted version fails cleanly with the pre-existing not-found error.
This closes the `O(T·|Σ|)` unbounded-growth gap (ESPEC §6.3 item 3) and
partially mitigates the SENSE-flood resource gap (no rate limiting yet).

## Motivation

`CTX_SWITCH` snapshots on every switch and the interactive loop
snapshots per token; without eviction a long session accumulates one
full heap clone per snapshot in four maps (five since RFC-0016). I-Mono (RFC-0002 line of
work, `restoreFix`) made ids safe to keep; retention makes them safe to
drop. Window k=16 preserves the documented rollback depths (5-token
entropy fallback, 80 ms audio window, 17-stream Moshi) with margin.

## Specification

1. `MemoryManager` gains `max_snapshots: usize`;
   `DEFAULT_SNAPSHOT_WINDOW = 16`; all constructors initialize it.
2. `set_snapshot_window(n)`: `n > 0` sets the window and evicts
   immediately down to it; `n == 0` disables eviction (explicit
   unbounded opt-out, e.g. foreplanned long interactive sessions).
3. After every `snapshot()` insert, while `snapshots.len() > max`,
   remove the smallest version key from ALL snapshot maps (`snapshots`,
   `sparse_snapshots`, `kv_snapshots`, `kv_heap_snapshots`, plus `meta_snapshots` since RFC-0016) atomically
   as one step (same key or none — maps never diverge).
4. `restore(evicted)` returns the pre-existing not-found `Err`; no new
   error variant. Callers already handle `Err` (host REPL prints and
   continues; `ABORT` logs a warning).
5. The `version` counter is unaffected (keeps I-Mono monotonicity from
   the restoreFix change).

## Backwards Compatibility

Behavioral change, no encoding change: restores of versions older than
the window now fail where they previously succeeded. All in-tree
callers handle `Err` cleanly (verified by survey: `vm.rs` ABORT path,
`main.rs` checkpoint paths log-and-continue). Any out-of-tree session
holding >16 live snapshots must either raise the window via
`set_snapshot_window` or re-snapshot. Default k=16 covers every
documented rollback depth.

## Security Considerations

- Bounds the FORK/CTX_SWITCH snapshot-amplification vector: memory is
  now `O(k·|Σ|)` regardless of session length (partial mitigation for
  the open SENSE-flood gap; rate limiting still missing).
- Eviction is oldest-first by version key; no attacker-controlled input
  selects victims. `0` (unbounded) requires an explicit call, never a
  default.
- Evicted restores fail closed (error, never partial state): `restore`
  still applies all-or-nothing per-key lookup as before.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/memory.rs`: `DEFAULT_SNAPSHOT_WINDOW`, `max_snapshots` field
  (all 3 constructors), `set_snapshot_window`, `evict_old_snapshots`,
  eviction call in `snapshot()`.
- Conformance tests (all green): `qa::snapshot_window_evicts_oldest`
  (window 4 of 6: oldest two `Err`, newest exact),
  `qa::snapshot_window_zero_means_unbounded`,
  `qa::snapshot_window_default_is_16`.
- Suite: `cargo test --lib` 180 passed; sole failure is the
  pre-existing, unrelated `moshi::test_gguf_qkv_split_shapes`
  (norm-gamma assertion on the local PersonaPlex GGUF; fails
  identically without this change).

Follow-up (NOT this RFC): I-Persist enforcement (clone-before-write
discipline audit — separate RFC number); `ssm_snapshots` stack bounds
in `Vm`; host-side `checkpoints` pruning policy in `main.rs`.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0003-00 | 2026-09-10 | DRAFT: window mechanism + policy + tests |
| 0003-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
