# RFC-0013 — DENOISE_STEP (`0x21`): Fused Diffusion Step

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x21 DRAFT->IMPL)
Obsoletes   : None
Feature Bit : none (v2.0-zone early implementation)
Bump        : none (same rule as RFC-0005/0006: next MINOR reserved for
              the next opcode family)
```

## Abstract

Implements `DENOISE_STEP 0x21`: one fused DDPM/DDIM reverse step
`x_t -> x_{t-1}`. Stateless per step (pure function of three tensors +
schedule scalars; multi-step state rides ordinary tensors, hence
existing snapshot/restore covers rollback with no new discipline).
Stochasticity draws from the calling context's splitmix64 stream
(RFC-0005/0009 family); `sigma=0` is exactly deterministic and consumes
no randomness.

## Motivation

Last of the wave-1 universal core in priority order (G4: lowest, but
tiny and fully specified). Unblocks diffusion programs (SD/Sora-style
loops unrolled or counter-driven later) and exercises the seeded-RNG
contract on a second consumer besides `SAMPLE`.

## Specification

```text
0x21 DENOISE_STEP  rdest=x_{t-1}, rsrc1=x_t, rsrc2=eps, rsrc3=0xFF.
  payload[0..4]=alpha_bar_t f32, [4..8]=beta_t f32,
  [8..12]=sigma_t f32, [12..16]=timestep u32 (informational).
```

Math (DDPM ancestral sampling, Ho et al. 2020):
- `alpha_t = 1 - beta_t`; require `0 < alpha_bar_t <= 1`,
  `0 <= beta_t < 1`, `sigma_t` finite `>= 0`, else deterministic Err.
- Shapes of `x_t`, `eps` must match exactly (any rank), else Err.
- `rsrc3 != 0xFF` is an explicit Err (schedule-tensor input is a
  follow-up; silently ignoring a data operand would violate RFC-0008
  spirit).
- `coef = beta_t / sqrt(1 - alpha_bar_t)` (guard `alpha_bar_t < 1`;
  `== 1` with nonzero eps is Err — division by zero, not a result);
  `x_{t-1} = (x_t - coef*eps)/sqrt(alpha_t) + (sigma_t>0 ? sigma_t*z : 0)`,
  `z ~ N(0,1)` from context stream (`normal_f32`).
- `sigma_t == 0` consumes NO randomness (reseed semantics stay clean).
- Non-finite result traps (fail-closed, same family rule as RANK1 out
  and FOREST leaves).
- Timestep is carried, not interpreted (schedule already parameterized).

Assembler: `DENOISE_STEP rD, rX, rE [, rS] [ALPHA=a] [BETA=b]
[SIGMA=s] [T=t]` (4th reg accepted by parse, rejected at exec unless
`0xFF` — parse/exec split documented here, not silent).

## Backwards Compatibility

Purely additive (const/ctor/parse/dispatch/stat). `0x21` previously
errored at execute.

## Security Considerations

- Seeded stochasticity: replayable by anyone holding the seed (same
  caveat as RFC-0009, not secrecy).
- Param validation precedes any draw: malformed schedules cannot
  advance (or desync) the RNG stream — `Err` leaves state untouched.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_DENOISE_STEP`, `denoise_params/set`,
  `instr_denoise_step`, mnemonic, parse arm, roundtrips.
- `src/vm.rs`: `exec_denoise_step`, dispatch arm, `denoise_execs`.
- Conformance: hand-computed sigma=0 vector, seeded replay +
  no-draw check, independent re-derivation of seeded draws, param/
  shape/rsrc3 errors, `programs/denoise_demo.m3asm` (7 steps,
  denoise_execs=3, verified executed — not just exit 0).
- Suite: `cargo test --lib` 228 passed (sole failure: pre-existing
  unrelated moshi norm-gamma).

Follow-up (NOT this RFC): schedule-tensor input (`rsrc3`), DDIM
eta-parameterization as distinct mode, adaptive step controllers
(`x(t)` loops belong in `.m3asm`, not the ISA).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0013-00 | 2026-09-10 | DRAFT: fused DDPM/DDIM step + demo |
| 0013-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
