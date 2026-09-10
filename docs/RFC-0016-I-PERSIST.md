# RFC-0016 — I-Persist: Snapshot-Coverage Completeness

```text
Status      : DRAFT
Category    : Standards Track
Updates     : docs/ESPEC.md §6.3 item 1 (I-Persist REQUIRED->IMPL)
Obsoletes   : None
Feature Bit : none
Bump        : none (correctness completion under the v1.4 line)
```

## Abstract

Closes the last T1 condition. The audit (this RFC) finds the discipline
ALREADY holds almost everywhere — dense heaps via `Arc::make_mut`
(`memory.rs` write paths), sparse/KV/SSM via deep clones, engine states
via CoW tensors — with exactly ONE hole: `tensor_meta` is not
snapshotted, so post-snapshot allocs leave dangling metas after
`restore` (clean `Err` today, divergence all the same). This RFC adds
the fifth snapshot map, proves each mechanism with a visibility test
(write-then-restore must not leak, including in-place sparse mutation),
and documents the one deliberate exclusion (`TEMPORAL`).

## Motivation

ESPEC Section 6.3 item 1 has been OPEN since the Σ model: "no kernel
may mutate state reachable by an active snapshot". Without closing it,
T1 (bit-exact rollback) rests on an assumption instead of a proof.

## Specification

1. New map `meta_snapshots: HashMap<u64, HashMap<u128, TensorMeta>>`
   (`TensorMeta: Clone` already); `snapshot()` inserts, `restore()`
   replaces, `evict_old_snapshots()` recycles — same key discipline as
   the other four maps (RFC-0003).
2. Coverage table (normative statement of what protects what):

| Store | Mechanism | Proof |
|:---|:---|:---|
| `global_heap`, `kv_heap` (`Arc<Vec<u8>>`) | `Arc::make_mut` CoW on every write path | dense visibility test (existing `rollback_100_50_50` + new) |
| `sparse_heap` (`SparseTensor`/owned CSR) | deep clone at snapshot; later in-place mutation (e.g. `vm.rs` sparse fill) cannot reach the copy | sparse-mutation invisibility test (new; would fail on shallow clone) |
| `kv_cache_layers` | wholesale `Vec` clone at snapshot | existing memory-level test |
| `ssm_states` | `Vec` clone at `FORK` | existing hybrid test |
| `snn/rank1` handles + tensors | CoW rebinding (RFC-0004/0015) | map-rollback tests |
| `tensor_meta` | NEW fifth map (this RFC) | meta-coherence test (new; fails before) |
| `TEMPORAL` circular buffer | EXCLUDED by design (below) | — |
| `next_*_offset` counters | monotonic, never rewound: no aliasing possible | by construction |

3. `TEMPORAL` exclusion (deliberate, not a gap): the circular buffer
   stages *inputs* (PCM, prompts), not model state. Rollback replays
   from restored heaps and re-injects input via `temporal_push`
   (append-only); rewinding a ring shared with live sensors would
   corrupt in-flight captures. Documented here so the exclusion is a
   decision, not an oversight.
4. No write path changes: the audit proves `make_mut`/deep-clone
   coverage; this RFC only adds the missing map plus proofs.

## Backwards Compatibility

One more map in already-versioned operations; eviction covers it;
`restore` of old snapshots (taken before this change — impossible
in-practice: snapshots are in-memory only, never serialized) N/A.
`restore` now also swaps metas: strictly more coherent, no caller
changes (all handle `Err` already).

## Security Considerations

- Dangling metas previously let `get_tensor_meta` succeed for dead
  tensors (clean read-`Err` downstream, but a confused-deputy shape:
  fixed by coherence).
- Snapshot cost grows by one small map (`TensorMeta` is metadata-only);
  retention bound (RFC-0003) covers it unchanged.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/memory.rs`: `meta_snapshots` field (3 ctors), insert/restore/
  evict, doc comment citing this RFC.
- Conformance: sparse-mutation invisibility, meta-coherence
  (alloc-restore-meta-absent), dense visibility (existing),
  full suite green.

Follow-up (NOT this RFC): `TEMPORAL` point-in-time capture (needs a
design decision on sensor sharing first); sparse snapshot cost
(O(nnz) clone per snapshot — fine at MVP sizes, revisit with PAMT).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0016-00 | 2026-09-10 | DRAFT: fifth map + visibility proofs + TEMPORAL ruling |
