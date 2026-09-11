# RFC-0033 — Stream-Aware KV: 17 Streams sem Novo Opcode

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC.md §16 (row; no version change)
Obsoletes   : None
Feature Bit : none (no new opcodes)
Bump        : none (deliberately: architecture, not an opcode family —
              same rule as Trilha P)
```

## Abstract

Makes the KV cache stream-aware (Fase 5, part 3 of 3): streams 0–16
(17, `MOSHI_N_STREAMS`), each an independent layer set with its own
sequence, retention coverage, and rollback. Stream 0 is today's store,
bit-identical behavior — all existing code paths untouched. `DEPFORMER`
needed zero changes (its KV was keyed by `(stream, layer)` since
RFC-0032 — the foresight pays here, proved by test). `KV_TRUNCATE` /
`KV_COMPRESS` route by their (previously vetoed) `STREAM=` fields.
No new opcode, no version bump: this is the memory architecture the
opcodes assumed.

## Motivation

Full-duplex Moshi runs 17 concurrent sequences (1 text + 8 user + 8
agent, ESPEC §14) through one KV store today — sequences would
collide on append/truncate. The alternative (17 separate VMs) breaks
shared weights and unified rollback. Per-stream layer sets inside one
`MemoryManager`, covered by the same snapshot/retention/rollback
machinery, is the minimal honest design. `ATTN` stream routing stays
out (needs ISA design for stream selection — follow-up, stated).

## Specification

- Streams `0–16` (`u16` ids; `> 16` => explicit `Err` at exec).
  Stream 0 = the pre-existing `kv_cache_layers` (untouched code).
- Extra streams live in `kv_extra_streams: HashMap<u16,
  Vec<KvCacheLayer>>`, lazy-created on first append/truncate/compress,
  inheriting stream-0 geometry (`n_layers`, `hidden`); stream 0
  uninitialized => `Err` (call `kv_cache_init` first — same rule as
  today, extended).
- Missing stream on read/truncate/compress => no-op `Ok` with empty
  results (never materializes empty state, never fails steady-state
  calls) — EXCEPT append, which creates (that's its job).
- Snapshots: `kv_extra_snapshots: HashMap<u64, HashMap<u16,
  Vec<KvCacheLayer>>>` alongside the five existing maps (SEIS mapas,
  same-key-or-none discipline — the "CINCO mapas" comment is updated).
  Eviction prunes it; restore rewinds it; the counter never lowers
  (I-Mono untouched — same `restore`, extended).
- `kv_cache_clear` clears ALL streams (new-generation semantics;
  single-stream callers observe identical behavior).
- `KV_TRUNCATE STREAM=s` / `KV_COMPRESS STREAM=s`: vetoes removed,
  routed per stream; `sid > 16` => explicit `Err`.
- `DEPFORMER` unchanged (Vm-side `dep_kv` was already keyed) — its
  stream test here is the RFC-0032 foresight proof. Note the two
  stores are distinct on purpose: `dep_kv` (Vm, depformer sliding
  windows) vs memory KV streams (transformer inference cache).

## Backwards Compatibility

Stream-0 paths byte-identical (same struct, same functions — the old
methods delegate with `sid 0`). No encoding touched. No version bump.

## Security Considerations

- No new exhaustion vector beyond stream count × window: 17 streams
  max (enforced), each under the same retention window and the same
  `kv_cache_max_seq` per-layer cap as stream 0.
- Missing-stream reads return `None`/empty (no fabricated rows, no
  cross-stream leakage — maps keyed, never indexed positionally
  across streams).
- `sid > 16` traps at the exec boundary (no silent mod-17 wrap that
  could alias stream 0).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/memory.rs`: `kv_extra_streams` + `kv_extra_snapshots` fields;
  `*_stream` methods (append/truncate/compress/seq_len/row);
  legacy methods delegate with `sid 0`; snapshot/restore/evict cover
  the sixth map; `kv_cache_clear` clears all.
- `src/vm.rs`: `MemBackend` forwarding for the stream methods used by
  exec; `WgpuMemoryManager` delegates to `cpu_fallback` (established
  pattern); `exec_kv_truncate`/`exec_kv_compress` route by `STREAM`
  (vetoes removed, `> 16` traps).
- Conformance: per-stream isolation (values + seq), lazy geometry
  inheritance, uninit errors, missing-stream no-ops, truncate/compress
  routing incl. `> 16` traps, snapshot/restore roundtrip across
  streams, retention bound holds, `DEPFORMER` on stream 1 (foresight
  proof), `programs/kv_stream_demo.m3asm` (routing + counters).
- Suite: `cargo test --lib` 348 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- No new bench (no new hot path: routing is one HashMap lookup, kernels
  unchanged — documented skip, not an omission).

Follow-up (NOT this RFC): `ATTN` stream selection (needs ISA design);
per-stream retention quotas (today: shared version window);
`KV_PIN`/`KV_RETRIEVE`; unifying `kv_cache_layers` into the keyed
store (cleanup once stable — deliberately NOT done here to keep the
stream-0 diff at zero).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0033-00 | 2026-09-11 | DRAFT: stream-aware KV + demo |
| 0033-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, no bump |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
