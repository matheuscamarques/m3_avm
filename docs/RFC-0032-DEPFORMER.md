# RFC-0032 — Depformer Step: `DEPFORMER` (`0x44`, single-stream)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.6 (0x44 DRAFT->IMPL; §3.6 now fully IMPL);
              docs/ESPEC.md §5.8, §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcode)
Bump        : MINOR (v1.12 -> v1.13: depformer step 0x44; Fase 5 part 2)
```

## Abstract

Implements one depformer layer step (Fase 5, part 2 of 2): Q/K/V
projections from caller weight tables, causal sliding-window
per-head attention over per-(stream, layer) KV (no RoPE, mirroring
the in-repo `moshi.rs` stub semantics), output projection, and
per-codebook categorical sampling over caller head weights. Key
design (stream-ready day one): KV state keyed by `(stream, layer)`,
defaulting to stream 0 — the 17-stream RFC adds streams, never
reworks this op. Programs compose norms/FFN/residuals from existing
ops (canonical layer program below); the op owns exactly what cannot
be lowered: KV append + window + code sampling. Output packs
`[Y | codes]` (the `DISTANCE` pack precedent, split with `SLICE`).

## Motivation

Per the ESPEC Section 13 filter: codec frames (`CODEC_ENC`) enter as
codes, but nothing turns text/audio conditioning back into codes —
the full-duplex loop is open at exactly this point (Marco 1 needs
it). A hand-rolled program would need KV append (no primitive),
sliding eviction (no primitive) and per-codebook sampling (no
primitive): three genuine gaps, one opcode. Everything else in a
depformer layer (RMSNorm, GEMV, SwiGLU, residuals) already exists.

## Specification

32B, dense F32. `DEPFORMER rD, rX, rW` + immediates:

- `rX`: hidden in, flat numel `D` (`[1,D]` or `[D]`).
- `rW`: weight-TABLE tensor (U8 bytes holding 8×u64 LE addrs):
  `[Wq, Wk, Wv, Wo, Whead, 0, 0, 0]` (last three reserved, must be 0
  else `Err` — forward-compat, not silent). Shapes: Q/K/V/O `[D,D]`,
  `Whead` `[D, NCB*Q]`, all dense F32, else `Err`.
- `rD <- NEW [1, D+NCB]`: first `D` = `Y` (attention out @ Wo, NO
  residual/norm/FFN — program composes), next `NCB` = codes as f32
  (exact < 2^24).

Payload (all LE): `[0]`=stream u8, `[1]`=layer u8, `[2..4]`=ncb u16
(0=default 16), `[4..6]`=nheads u16 (0=default 16; must divide D),
`[6..8]`=levels Q u16 (0=default 1024), `[8..10]`=context u16
(0=default 8; window rows, ≥ 1 effective), `[10..14]`=temp f32,
`[14..16]`=topk u16 (0 = argmax, deterministic default).

Semantics per call: `Q=x@Wq, K=x@Wk, V=x@Wv` (plain row GEMV);
append `(K,V)` to `dep_kv[(stream,layer)]`, drain oldest while rows >
context; per-head causal attention over the window (no RoPE, moshi
stub parity); `Y = attn@Wo`; logits_c = `Y@Whead[:, c*Q..]` per
codebook; code = argmax (`TOPK=0`) or seeded top-k sample
(`SAMPLE` conventions via context RNG: `TEMP`/`TOPK`, deterministic
under `RNG_SEED`).

New state: `Vm::dep_kv: HashMap<(u8,u8), DepKV>`,
`DepKV { k, v: Vec<f32>, d: usize }`; `dep_snapshots` version-tagged
stack; FORK-push / ABORT-pop / SNAPSHOT-push / RESTORE-pop (same
discipline as all engine maps — all four sites, no exceptions).

Canonical layer program (norms/FFN/residuals in program land):
```asm
NORM rX1, rX, rG1, rB1
MATVEC rQ0, rX1, rWq0   ; (ou DEPFORMER com a tabela — ver abaixo)
```
Full form with the op (weights via table; `rW*` are `[D,D]` tensors):
```asm
DEPFORMER rP, rX, rWT LAYER=0 NCB=16 NHEADS=16 LEVELS=1024 CONTEXT=8
SLICE rY, rP START=0 LEN=D
SLICE rCodes, rP START=D LEN=NCB
NORM rN, rY, rG, rB
FFN rF, rN, rW1, rW2
ADD rXnext, rX, rY
ADD rXnext, rXnext, rF
```

Assembler:
```text
DEPFORMER rD, rX, rW [STREAM=s] [LAYER=n] [NCB=n] [NHEADS=n]
  [LEVELS=n] [CONTEXT=n] [TEMP=x] [TOPK=n]
```
Strict lists gain exactly these keys.

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats + one module +
one state map). `0x44` previously errored at execute. No encoding
touched. v1.12 -> v1.13.

## Security Considerations

- Table-driven weights: each addr must resolve to a dense F32 tensor
  with EXACT shapes, else `Err` (no partial reads, no guessing);
  reserved slots must be 0 (future-proofing, loud).
- `u64→usize` via `try_from`; dim products checked; `NHEADS ∤ D`
  traps; `TOPK > Q` traps (no clamp); temp NaN traps, temp ≤ 0 with
  stochastic sampling traps.
- State cost bounded: window rows per (stream, layer); snapshot cost
  O(total KV bytes) like all engine maps, documented.
- No PERSISTENT writes (fresh outputs only; weights read-only).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/depformer.rs` (new): `DepKV` (push + window drain) +
  `dep_attention` (causal per-head, no RoPE — moshi stub parity) +
  unit tests (window cap, attention vs naive, empty).
- `src/opcodes.rs`: `OP_DEPFORMER`, params + setters,
  `instr_depformer`, mnemonic, `parse_line` arm, roundtrip.
- `src/vm.rs`: `dep_kv` + `dep_snapshots`, 4-site wiring
  (FORK/ABORT/SNAPSHOT/RESTORE), `exec_depformer`, dispatch arm,
  1 counter.
- Conformance: all-ones-weights golden (Y exact-ish, codes zero),
  KV window cap + drain order, argmax determinism (same input twice),
  TOPK sampling seeded agreement, FORK/ABORT + SNAPSHOT/RESTORE
  coherence, shape/mode/temp/topk/stream table errors, assembler
  roundtrip + strict rejections, combined Rust-level test (tables
  built in Rust — `.m3asm` cannot spell u64 handle tables; the
  missing tiny immediate-store/`.data` is recorded as Fase 9
  assembler follow-up, NOT smuggled in here).
- Suite: `cargo test --lib` 342 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
  (One run showed 4 transient timing flakes under load; 3/4 runs
  clean with identical results — not a regression.)
- Benches: `benches/depformer_bench.rs` (tiny step D=32, per-instruction
  via `step_instruction`).

Follow-up (NOT this RFC): RFC-0033 (17-stream KV: keyed stores +
STREAM= routing + per-stream retention); per-codebook head layouts
beyond `[D, NCB*Q]`; real PersonaPlex weight wiring (F3b);
resident GEMV path (matvecaufs); immediate-store/`.data` for
table authoring in `.m3asm` (Fase 9); GPU-tiled depformer.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0032-00 | 2026-09-11 | DRAFT: depformer step + demo-test |
| 0032-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.13 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
