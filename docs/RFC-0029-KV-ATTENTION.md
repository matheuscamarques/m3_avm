# RFC-0029 — KV & Attention: `KV_COMPRESS`/`FLASH_ATTN`/`ATTN_SPARSE` (`0x39/0x3A/0x3B`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.5 (0x39,0x3A,0x3B DRAFT->IMPL; §3.5 now
              fully IMPL); docs/ESPEC.md §5.8, §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.10 -> v1.11: KV/attention family 0x39-0x3B;
              Fase 3 de 3 partes completa)
```

## Abstract

Implements the KV/attention family (Fase 3, part 3 of 3), closing the
`0x30-0x43` block and the RFC-0004 follow-up (`ATTN_SPARSE`): sink+
window KV eviction (`KV_COMPRESS`, the eviction design RFC-0010
deferred), blocked online-softmax attention (`FLASH_ATTN`, the real
algorithm — tiling with rescaling, not a rename), and fused top-k
pruned attention (`ATTN_SPARSE`). Equivalence with dense `ATTN` is
verified by TOLERANCE (max abs diff ≤ 1e-4 on goldens), never claimed
bit-exact: the blocked/scalar summation order differs from the
ndarray path by floating-point association, and saying otherwise
would be a lie. This RFC does NOT close the TopK Lean obligation
(§15.1): recorded honestly in §6 as environment-blocked.

## Motivation

Per the ESPEC Section 13 filter: rolling truncation (`KV_TRUNCATE`)
cannot retain sinks while dropping middles (StreamingLLM showed sinks
matter); nothing computes attention without materializing the full
N×N score matrix (the IO-aware formulation is exactly the missing
primitive); and pruned attention exists only as a 4-op program
(`DISTANCE→SLICE→GATHER→ATTN`), never fused — the T4 empirical leg
needs the fused form to measure the real thing.

## Specification

All 32B. Dense F32 2-D only in this RFC (sparse/KV-cache-backed
inputs trap explicitly, like previous MVPs).

```text
0x39 KV_COMPRESS  no regs (rdest unused). payload[0..2]=sink u16,
  [2..4]=window u16, [4]=mode (0=SINK_WINDOW only), [5..7]=stream u16
  (0 only, like KV_TRUNCATE). Per layer: if seq <= sink+window, no-op
  Ok (steady-state calls must not fail); else keep rows [0..sink) +
  [seq-window..seq) in BOTH k and v, drop the middle, seq=sink+window.
  sink==window==0 => Err (degenerate; emptying the cache is not
  compression). Operates on kv_cache_layers only (kv_heap untouched —
  same scope as TRUNCATE). Rollback-safe via the snapshot machinery
  (mutates live layers like TRUNCATE; ABORT restores). Does NOT
  snapshot itself.
0x3A FLASH_ATTN  rdest, rQ, rK, rV (same regs as ATTN).
  payload[0..2]=block_rows u16 (0 = default 32; must be >= 1).
  Blocked forward pass with online softmax (per-row running max +
  normalizer + rescaled accumulator — the actual FlashAttention math,
  CPU-sized): Q[M,D], K[N,D], V[N,Vd] -> [M,Vd], scale 1/sqrt(D).
  Flags must be 0 (NOTIFY has no meaning on the blocked path — loud,
  not silently dropped). Shapes validated like ATTN (Q.cols==K.cols,
  K.rows==V.rows). Output tolerance-verified vs ATTN (see Reference).
0x3B ATTN_SPARSE  rdest, rQ, rK, rV. payload[0]=metric (0=DOT only;
  else Err — attention scores ARE dots, no fake choice), [1..3]=topk
  u16 (0 = all = dense-equivalent path through the same kernel).
  Per query row: top-k keys by score (ties: lower index, house rule),
  stable softmax over the k, weighted V. k > N => Err (no clamp).
  Closes the RFC-0004 follow-up; the DISTANCE-fed composed path
  (attn_topk_demo) stays as the cross-check, not replaced.
```

Equivalence contract (tolerance, not bits): on finite inputs,
`max|FLASH_ATTN - ATTN| <= 1e-4` and `max|ATTN_SPARSE(k=N) - ATTN| <=
1e-4` for the golden sizes (32-wide and below); NaN poisons lanes in
all three (documented, unspecified exact bits).

Assembler:
```text
KV_COMPRESS SINK=n WINDOW=n [MODE=SINK_WINDOW] [STREAM=0]
FLASH_ATTN rD, rQ, rK, rV [BLOCK=n]
ATTN_SPARSE rD, rQ, rK, rV [TOPK=n] [METRIC=DOT]
```
Strict lists gain exactly these keys. `MODE=` accepts `SINK_WINDOW`
or `0`; `METRIC=` accepts `DOT` or `0`.

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats + two small
`MemoryManager` methods). `0x39/0x3A/0x3B` previously errored at
execute. No encoding touched. v1.10 -> v1.11.

## Security Considerations

- `KV_COMPRESS` cannot fabricate rows (only drops middle, keeps
  prefix/suffix verbatim); degenerate (0,0) traps instead of wiping
  the cache silently.
- OOB impossible: k<=N enforced, windows bounded by seq, block math
  pre-bounded; `u64->usize` via `try_from`, products checked.
- Tolerance (not equality) is load-bearing honesty: any future
  optimization must re-verify the bound, never assume bits.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/memory.rs`: `kv_cache_compress_sink_window(sink, window)`
  (per-layer drain, mirrors `kv_cache_truncate` scope) +
  `kv_cache_row(layer, pos)` (read-only inspection for tests).
- `src/opcodes.rs`: `OP_KV_COMPRESS/FLASH_ATTN/ATTN_SPARSE`, mode/
  metric consts, params + setters, `instr_*`, mnemonics, `parse_line`
  arms, roundtrips.
- `src/vm.rs`: `exec_kv_compress/flash_attn/attn_sparse`, dispatch
  arms, 3 counters.
- Conformance: compress sink+window values (distinctive rows) +
  no-op + degenerate + stream errors + rollback-via-ABORT; flash vs
  ATTN tolerance + BLOCK=2-vs-64 invariance + shape/sparse/flag
  errors; sparse top-k values + k=N tolerance vs ATTN + k>N/metric/
  sparse errors; assembler roundtrips + strict rejections; combined
  program (`vm::test_rfc0029_assembled_program_runs`: counters exact);
  `programs/sparse_attn_demo.m3asm` (KV no-op path + fused sparse +
  tolerance cross-check vs composed path, verified).
- Suite: `cargo test --lib` 326 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/attention_bench.rs` (flash vs sparse 32-wide,
  per-instruction via `step_instruction`).

Follow-up (NOT this RFC): KV_PIN/KV_RETRIEVE; non-DOT prune metrics;
sparse-table ATTN_SPARSE; GPU-tiled FLASH_ATTN; cyclic barriers
(cluster); stepError physical instantiation (Lean, still open).

## §6. TopK Lean obligation — FECHADA (com correções de rota)

The `topk_error_bound` sorry in `formal/Formal/TopK.lean` is now a
machine-checked proof (`lake build` green, zero `sorry` in `formal/`).
What the attempt taught (recorded so the next proof goes faster):

1. The stated theorem was vacuous as written (`... ∨ True` closes by
   `Or.inr trivial` — proving nothing). It was RESTATED first: L1 ≤
   `2ε/(1−ε)` over `Finset` sums with a top-T subset, for ANY subset
   `T` ("top" matters only for the assumed hypothesis `τ < ε`, never
   in the proof). Restating is honest here because the build verifies
   the new statement immediately.
2. The environment CAN build it: toolchain present (elan,
   leanprover/lean4:v4.33.1) AND Mathlib vendored with prebuilt
   oleans (`formal/.lake`, 8322 oleans) — an early `ls` without `-a`
   hid `.lake` and nearly caused a false "blocked" verdict. Lesson:
   `ls` hides dotfiles; verify with the build itself.
3. Lemma names were verified empirically FIRST (`lake env lean` on a
   scratch file: `div_lt_iff₀`, `le_div_iff₀`, `Finset.sum_sdiff`,
   `Finset.sum_div`), which held the proof to two iterations:
   (a) `Finset.mul_sum` has the constant on the LEFT — the needed
   direction is `(Finset.sum_mul ..).symm`; (b) one `field_simp`
   closed its goal alone, leaving `ring` with "no goals" (removed).
4. The tolerance bound still lives in Rust tests too (this RFC) —
   Lean proves the math, tests prove the code honors it. Both, not
   either-or.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0029-00 | 2026-09-11 | DRAFT: KV/attention + Lean verdict |
| 0029-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.11 |
| 0029-02 | 2026-09-11 | TopK main bound proved (sorry removed, lake build green) |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
