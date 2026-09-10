# RFC-0015 — SPIKE_STEP (`0x20`): LIF Integrate-and-Fire

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x20 DRAFT->IMPL)
Obsoletes   : None
Feature Bit : none (v2.0-zone early implementation)
Bump        : none (same rule as RFC-0005/0006: next MINOR reserved for
              the next opcode family)
```

## Abstract

Implements `SPIKE_STEP 0x20`: one discrete leaky integrate-and-fire
step over `n` neurons with threshold, decay, reset and refractory
period. State discipline copies `RANK1_UPDATE` (RFC-0004), not `SSM`
(legacy): membrane `V` and refractory countdown live in immutable
tensors, rebound per step (logically in-place, physically CoW), with a
per-layer handle map covered by `FORK`/`ABORT`. Output spikes are f32
0.0/1.0 (not u8: uniform tensor pipeline; exact values, no ambiguity).

## Motivation

Third stateful engine (after SSM and matrix memory), first spiking one.
Completes the Family-2 trio with a common CoW discipline, so the three
engines share one rollback story instead of three.

## Specification

```text
0x20 SPIKE_STEP  rdest=spikes[1,n], rsrc1=V[1,n]|V[n] (rebound CoW),
                 rsrc2=I[1,n]|I[n], rsrc3=pack[4]|0xFF.
  payload[0..4]=V_threshold f32, [4..8]=decay f32, [8..12]=V_reset f32,
  [12]=layer_id u8, [13]=refractory_steps u8.
  pack (rsrc3 tensor, 4 elems) = [thresh, decay, reset, refr_steps];
  payload wins when rsrc3 == 0xFF; pack wins when present (documented
  precedence; assembler uses payload unless PACKREG given — no silent mix).
```

Dynamics per neuron (deterministic, in order):
1. `V += I` (input current added; shapes must agree on `n`, else Err).
2. `V *= decay` (`decay` validated in `[0,1]`; `1` = pure integrate).
3. Fire iff `refr <= 0 AND V >= threshold` (`>=` fires: documented edge).
4. Fired: `spike=1`, `V=V_reset`, `refr=refractory_steps`.
   Else: `spike=0`, `refr = max(0, refr-1)` (membrane still decays
   during refractory — no freeze hack).
5. Threshold must be finite; `refr` countdown stored as f32 tensor
   (uniform pipeline; 4× bytes vs u8, irrelevant at MVP sizes).
6. New `V` + `refr` tensors allocated per step; `rsrc1` rebound;
   `Vm::snn_layers: layer -> (V_addr, refr_addr)` + fork/abort stack
   (mirrors `rank1_layers`). Non-finite `V` post-update traps
   (fail-closed, family rule).

Defaults (payload zeros = `instr` default path, NOT wire zeros):
assembler defaults `THRESH=1.0 DECAY=0.9 RESET=0.0 LAYER=0 REFRACT=2`
(per the frozen plan). Raw zero payload decodes to thresh=0/decay=0 —
a degenerate but VALID config (everything fires, full leak); exec does
not second-guess explicit zeros.

Assembler: `SPIKE_STEP rD, rV, rI [rPack] [THRESH=] [DECAY=]
[RESET=] [LAYER=] [REFRACT=]`.

## Backwards Compatibility

Purely additive. `0x20` previously errored at execute.

## Security Considerations

- Depth/size bounded by input tensors (no hidden allocation: two `[n]`
  tensors per step, caller-visible).
- No time source, no RNG: fully deterministic given inputs (replayable
  spike trains; SENSE spike-train encoding stays a follow-up).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_SPIKE_STEP`, `spike_params/set`,
  `instr_spike_step`, mnemonic, parse arm, roundtrips.
- `src/vm.rs`: `snn_layers` + snapshots, `exec_spike_step`,
  dispatch arm, `spike_execs`.
- Conformance: 3-neuron/4-step hand golden (fire/reset/refractory
  cycle incl. ==thresh edge), pack precedence, decay/thresh/shape/pack
  errors, fork/abort map rollback + CoW intactness,
  `programs/snn_demo.m3asm` (spike_execs=3, verified executed).
- Suite: `cargo test --lib` 237 passed (sole failure: pre-existing
  unrelated moshi norm-gamma).

Follow-up (NOT this RFC): SNN `SENSE` spike-train mode, u8-packed
spike trains, refractory as integer tensor, GPU walk.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0015-00 | 2026-09-10 | DRAFT: LIF + CoW state + demo |
| 0015-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
