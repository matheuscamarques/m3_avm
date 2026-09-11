# RFC-0031 — Audio DSP: `STREAM_MERGE`/`VAD_DETECT`/`AUDIO_RESAMPLE`/`AUDIO_FILTER`/`AUDIO_WINDOW` (`0x45-0x49`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.6-partial (0x45-0x49 DRAFT->IMPL);
              docs/ESPEC.md §5.8, §16, §19, version line
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : MINOR (v1.11 -> v1.12: audio DSP family 0x45-0x49;
              0x44 DEPFORMER stays DRAFT — Fase 5 part 2)
```

## Abstract

Implements the DSP half of full-duplex audio (Fase 5, part 1 of 2):
voice-activity scoring (`VAD_DETECT`, energy + zero-crossing — the
current `SENSE VAD` is a random stub, documented in-repo), two-source
mixing (`STREAM_MERGE` with explicit gain), sample-rate conversion
(`AUDIO_RESAMPLE`, linear), single-axis FIR filtering
(`AUDIO_FILTER`), and window functions (`AUDIO_WINDOW`, Hann/Hamming
periodic). All five are dense-F32-only, fresh-tensor outputs, with
exact documented math (no silent defaults beyond the house
payload-zero convention). `DEPFORMER` (`0x44`) and 17-stream KV stay
DRAFT: they need model + memory-architecture work (part 2), not DSP.

## Motivation

Per the ESPEC Section 13 filter: barge-in needs a REAL voice/silence
signal over arbitrary PCM (the stub draws coins), full-duplex needs
mixing, and any 24kHz↔16kHz boundary needs resampling — none exists
below hand-rolled `MATVEC` abuse. Windowing + FIR are the two DSP
primitives every codec/frame pipeline assumes. Five small exact ops
instead.

## Specification

All 32B, stateless (S), dense F32 only (shared reader guards).
Fresh outputs (GLOBAL, never aliasing).

```text
0x46 VAD_DETECT  rdest <- NEW [1,1] score tensor, rsrc1 = PCM addr.
  payload[0]=mode (0=ENERGY,1=ZCR,2=ML). ENERGY = RMS (raw, >= 0 —
  unbounded by nature, NOT clamped to [0,1]: clamping energy would
  be a lie). ZCR = crossings/(n-1) in [0,1], crossing = strict sign
  change between consecutive samples (zeros don't cross). n < 2 with
  ZCR => Err (no pair to compare). ML parses but traps (no model —
  DIR-precedent: spelled for readability, vetoed at execute).
0x45 STREAM_MERGE  rdest <- NEW tensor, rsrc1=A addr, rsrc2=B addr:
  out = GAIN*A + (1-GAIN)*B, shapes must match EXACTLY else Err.
  payload[0..4]=gain f32 (any finite; NaN/inf => Err), [4..6]=
  n_streams u16 (0 = default 2; < 2 => Err — a 1-stream "mix" is
  contradictory metadata), [6]=mode (reserved 0; nonzero => Err —
  no MODE key exists yet by design, §Alternatives).
0x47 AUDIO_RESAMPLE  rdest <- NEW tensor, rsrc1 = src addr.
  payload[0..4]=src_rate u32, [4..8]=dst_rate u32 (both > 0 else Err).
  Linear interpolation; out_len = floor(in*dst/src), >= 1 else Err.
  Edge clamping at the tail (documented; matches mimi.rs precedent
  style, exact formula in Reference).
0x48 AUDIO_FILTER  rdest <- NEW tensor, rsrc1=X addr, rsrc2=B addr
  (FIR taps, flat, len >= 1 else Err). payload[0]=mode (0=FIR;
  1=IIR parses but traps — needs A-coeff tensor design, follow-up).
  SAME-size centered convolution, zero-padded edges: tap j aligns to
  x[n-(j-(M-1)/2)] (integer division; M=2 => causal-ish [n,n-1]).
0x49 AUDIO_WINDOW  rdest <- NEW tensor, rsrc1 = src addr.
  payload[0]=type (0=HANN,1=HAMMING; else Err). Periodic over ALL
  elements: Hann w[n]=0.5(1-cos(2πn/N)), Hamming 0.54-0.46cos.
  Applied flat (N = numel); N=1 yields 0.0 (formula-honest, documented).
```

Assembler:
```text
VAD_DETECT rD, rT [MODE=ENERGY|ZCR] (ML parses, exec vetoes)
STREAM_MERGE rD, rA, rB GAIN=x [N_STREAMS=n]
AUDIO_RESAMPLE rD, rT SRC=n DST=n (both required)
AUDIO_FILTER rD, rX, rB [MODE=FIR] (IIR parses, exec vetoes)
AUDIO_WINDOW rD, rT [TYPE=HANN|HAMMING]
```
Strict lists gain exactly these keys.

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats). `0x45-0x49`
previously errored at execute. No encoding touched. v1.11 -> v1.12.

## Security Considerations

- All sizes precomputed checked (`out_len`, `N*M` products); OOB
  impossible; empty selections trap (n<2 ZCR, empty taps, zero out).
- No aliasing (fresh GLOBAL outputs); no PERSISTENT writes possible
  (outputs never mapped — `alloc_tensor` plain path; GGUF-shape
  collision noted as the pre-existing class from RFC-0025, unchanged).
- VAD scores drive barge-in thresholds downstream: ENERGY unbounded
  is documented so callers scale via `COMPARE` on scaled integers
  (RFC-0007 pattern) instead of assuming [0,1].

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_STREAM_MERGE/VAD_DETECT/AUDIO_RESAMPLE/
  AUDIO_FILTER/AUDIO_WINDOW` + mode consts, params + setters,
  `instr_*`, mnemonics, `parse_line` arms, roundtrips.
- `src/vm.rs`: 5 exec fns, dispatch arms, 5 counters.
- Conformance: VAD ENERGY exact-DC + sine-RMS + silence, ZCR
  hand-vectors + sine range, resample up/down exact ramps, Hann/
  Hamming N=4 vectors, FIR moving-average exact, merge gain math,
  mode/shape/sparse/meta/rate errors, assembler roundtrips + strict
  rejections, combined program
  (`vm::test_rfc0031_assembled_program_runs`: counters exact),
  `programs/audio_dsp_demo.m3asm` (all 5 at 1, values verified).
- Suite: `cargo test --lib` 333 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.
- Benches: `benches/audio_dsp_bench.rs` (vad/resample/merge 1920-frame,
  per-instruction via `step_instruction`).

Follow-up (NOT this RFC): Fase 5 part 2 (`DEPFORMER` 0x44 + 17-stream
KV — model + memory architecture); TEMPORAL→tensor ingest bridge
(`SENSE PCM` lands in the ring, unreachable from tensor ops — real
gap found during this RFC); IIR with A-coeff design; ML-VAD model;
`N_STREAMS` routing semantics (metadata today).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0031-00 | 2026-09-11 | DRAFT: audio DSP family + demo |
| 0031-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green, v1.12 |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
