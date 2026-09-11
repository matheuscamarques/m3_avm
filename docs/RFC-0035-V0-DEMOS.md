# RFC-0035 — Fase V-0: Demos de Comportamento (pausa + filler sintético)

```text
Status      : IMPLEMENTED
Category    : Informational (programs; proposes no encodings)
Updates     : docs/PLANO_VISAO.md §4 (V-0 row)
Obsoletes   : None
Feature Bit : none (no new opcodes — that is the point)
Bump        : none (deliberately: programs-only, like Trilha P)
```

## Abstract

First execution of `PLANO_VISAO.md` Fase V-0: two behavior demos
built **exclusively** from already-implemented opcodes, proving that
turn-taking state machines and adaptive filler loops run on today's
ISA with zero `src/` changes. The pause demo drives its transitions
from the host input queue (`push_input` as speech events) with a
deterministic `STEPS`-based timeout; the filler demo proves the
`FILL`+`MUL`+`ADD` crossfade construction and seeded RNG picks.
 synthetic inputs throughout — voice enters in Fase V-3.

## Motivation

The vision family describes behaviors (pauses, filler speech) whose
mechanisms were designed on paper (`PAUSA_DE_PENSAMENTO.md` §5,
`FILLER_SPEECH.md` §5–§6) but never executed. V-0 closes exactly that
credibility gap at zero ISA cost, and in doing so exercises two
underused paths: `SENSE USER_INPUT`+`IF_INTERRUPT` as event driver,
and `STEPS` as a deterministic clock (vs wall-clock `CYCLES_COUNT`).

## Specification

No new encodings. Conventions used (all pre-existing):

- Turn states as `LOADI` enums: 1=INCOMPLETE, 3=RESUMED, 4=TURN_END
  (`THINKING` stays a design state until VAD-gated sensing lands —
  documented, not faked: the demo jumps INCOMPLETE→RESUMED).
- Timeout: `STEPS` + `ADD_IMM` deadline + `COMPARE PRED=LT`; the
  `CYCLES_COUNT` wall-clock variant is a one-immediate swap.
- Filler: crossfade `out = 0.5·A + 0.5·B` via `FILL` constants;
  3-iteration counter via `ADD_IMM`/`COMPARE`/`IF_EQUAL`;
  `RNG_SEED` + bare `RNG_UNIFORM` (defaults `[0,1)`).
- Register maps in file headers (numeric regs only — symbolic names
  do not assemble; same lesson as Trilha P sketches).

## Reference Implementation

- `programs/pause_turn_demo.m3asm` (17 instr): resume path
  (`rState==3`, VAD score `[0.5]`) and timeout path (64 quiet iters
  → `rState==4` + HALT), both deterministic.
- `programs/filler_synth_demo.m3asm` (15 instr): crossfade exact
  (`[0.5×8]` bit-exato), counter == 3, pick in `[0,1)`.
- `vm::test_planoV0_pause_paths`, `vm::test_planoV0_filler_synth`
  (value asserts, no timing dependence anywhere).
- Suite: `cargo test --lib` 356 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate verified by CLI loop.

Follow-up (NOT this RFC): VAD-gated (not queue-driven) branching
awaits tensor→scalar read (general gap, Trilha P §4); wall-clock
variant of the timeout; voice-path versions in Fase V-3.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0035-00 | 2026-09-11 | DRAFT: behavior demos |
| 0035-01 | 2026-09-11 | IMPLEMENTED: merged, suite green, no bump |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
