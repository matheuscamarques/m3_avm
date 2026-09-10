# RFC-0009 — SAMPLE via Context RNG (Deterministic Sampling)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC.md §9 (sampling replay now enforceable);
              RFC-0005 follow-up "SAMPLE routed through context RNG" (done)
Obsoletes   : None
Feature Bit : none (no encoding change; error paths unchanged)
Bump        : none (behavioral hardening under the v1.4 line)
```

## Abstract

`SAMPLE` (opcode and `STREAM`-to-`SAMPLE` sink) drew from
`rand::thread_rng`, making every sampled token unreproducible and the
ESPEC Section 9 replay claim aspirational for any stochastic program.
This RFC routes all in-VM sampling through the calling context's
splitmix64 stream (RFC-0005): same seed + same logits = same token,
`RNG_SEED` replays any prefix, `FORK` children share the stream from
the fork point.

## Motivation

The determinism story (RFC-0005) left its largest consumer on host
randomness. T1's rollout-invariance test and any future stochastic
`DENOISE`/`SAMPLE-TOPK`-style mode need seeded sampling first.

## Specification

1. New pure function `determinism::sample_logits_seeded(state: &mut
   u64, logits: &[f32]) -> u32`: jitter (`uniform[-0.5,0.5)` per logit,
   same anti-degeneracy role as before) → softmax → weighted draw from
   `uniform[0,sum)` → argmax fallback on non-finite/empty. Empty logits
   return 0 (legacy behavior preserved).
2. New `Vm::sample_logits_ctx(&mut self, ctx_id, logits) ->
   Result<u32>`: loads the context's `rng_state`, runs (1), stores the
   advanced state back. Unknown ctx is `Err` (no silent default).
3. All three in-VM call sites (`exec_sample`, both `STREAM`-`SAMPLE`
   sinks) use (2).
4. Host/REPL paths (`main.rs` interactive loop) keep thread randomness
   via `determinism::sample_logits_host` (documented
   non-deterministic-by-design: a human is in the loop there).
5. Boundary (unchanged, now documented): `SENSE` stimuli (white noise,
   VAD stub) stay environmental randomness — the VM does not seed the
   world, only itself.

## Backwards Compatibility

Token VALUES change vs the old implementation (different stream) — any
golden asserting a specific sampled token from the old RNG must be
re-baselined (none exist in-tree; verified by suite). Shapes, sinks,
`last_sample`, counters, error paths: unchanged.

## Security Considerations

- Deterministic sampling is replayability, not secrecy: MUST NOT seed
  security decisions from it (same caveat as RFC-0005 RNG).
- `RNG_SEED` from user input makes sampling predictable by design.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/determinism.rs`: `sample_logits_seeded`, `sample_logits_host`.
- `src/vm.rs`: `sample_logits_ctx`, 3 call sites migrated.
- `src/main.rs`: 2 host sites on `sample_logits_host` (unchanged values
  behavior: still random, now explicitly so).
- Conformance: determinism/reseed/extreme/empty unit test,
  two-VM agreement + reseed replay test, full suite 210 passed
  (sole failure: pre-existing unrelated moshi norm-gamma).

Follow-up (NOT this RFC): `SAMPLE-TOPK` stochastic tie-breaks (already
deterministic via index order — no change needed), DENOISE sigma>0
routing (needs the opcode first, W4).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0009-00 | 2026-09-10 | DRAFT: seeded sampling + boundary + tests |
| 0009-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
