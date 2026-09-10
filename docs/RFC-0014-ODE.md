# RFC-0014 — ODE_STEP (`0x25`): Fused ODE Step

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x25 DRAFT->IMPL)
Obsoletes   : None
Feature Bit : none (v2.0-zone early implementation)
Bump        : none (same rule as RFC-0005/0006: next MINOR reserved for
              the next opcode family)
```

## Abstract

Implements `ODE_STEP 0x25`: one fused step `x(t) -> x(t+dt)` with
Euler/RK2/RK4 methods over an explicit field `f(x,u) =
SILU(W·x + b + û)`. Stateless per step (pure function; trajectories
live in ordinary tensors). Refines the frozen spec's "campo linear
default" into two precise cases (packed weights vs documented
contractive default) instead of leaving the field undefined.

## Motivation

Liquid/ODE policies (robotics, IoT, time series) loop sensor-fast
integration where per-step dispatch through `ADD+MUL` dominates. One
fused op per step, plus a native RK4 that stays stable at larger `dt`.

## Specification

```text
0x25 ODE_STEP  rdest=x(t+dt), rsrc1=x(t) [n], rsrc2=u(t) [m] or 0xFF,
               rsrc3=Wb pack or 0xFF (default field).
  payload[0..4]=dt f32 (> 0 finite, else Err),
  [4]=method (0=Euler, 1=RK2-midpoint, 2=RK4; else Err),
  [5]=layer_id (carried, reserved for future namespacing).
```

Field (frozen "MATVEC+SILU fundido", made exact):
- `rsrc3 = Wb` packed tensor `[n*(n+1)]`: first `n*n` = W row-major,
  last `n` = b. `f(x,u) = SILU(W·x + b + û)`, `û` = u zero-padded (or
  truncated) to n. `u` empty (`rsrc2 == 0xFF`) => û = 0.
- `rsrc3 == 0xFF` => contractive default `W = -0.1·I`, `b = 0`
  (documented; stable for smoke tests, not a model).
- Methods: Euler `x+dt·f(x)`; RK2-midpoint `k1=f(x)`,
  `k2=f(x+dt/2·k1)`, `x+dt·k2`; RK4 classic. All field evaluations use
  the same `(W,b,u)`.
- Non-finite result traps (fail-closed, family rule).

Assembler: `ODE_STEP rD, rX, rU [DT=0.01] [METHOD=Euler|RK2|RK4]
[LAYER=n]` (`rU=0xFF` omits control; `rWb` 4th positional reg or absent
=> `0xFF`).

## Backwards Compatibility

Purely additive (const/ctor/parse/dispatch/stat). `0x25` previously
errored at execute.

## Security Considerations

- `dt`/method validated before any field evaluation (no wasted work,
  no state touched on Err).
- Cost is `O(stages·n²)` with caller-visible shapes — no hidden
  blowup; `n=0`/empty input is Err, not a degenerate loop.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_ODE_STEP`, `ode_params/set`,
  `instr_ode_step`, mnemonic, parse arm, roundtrips.
- `src/vm.rs`: `exec_ode_step` (+ `ode_field` helper), dispatch arm,
  `ode_execs`.
- Conformance: hand-computed Euler vector, RK4 vs fine-Euler
  reference (independent field rewrite), default-field contraction,
  param/shape errors, `programs/ode_demo.m3asm` (9 steps,
  ode_execs=5, verified executed).
- Suite: `cargo test --lib` 233 passed (sole failure: pre-existing
  unrelated moshi norm-gamma).

Follow-up (NOT this RFC): adaptive step control (belongs in `.m3asm`
loops per the universal plan, needs counters), staged-field reuse
across steps (cache Wb decode), stiff solvers (implicit Euler).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0014-00 | 2026-09-10 | DRAFT: fused Euler/RK2/RK4 + demo |
| 0014-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
