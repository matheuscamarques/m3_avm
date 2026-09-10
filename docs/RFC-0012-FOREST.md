# RFC-0012 — FOREST (`0x22`): Tree-Ensemble Vectorized Walk

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x22 DRAFT->IMPL)
Obsoletes   : None
Feature Bit : none (v2.0-zone early implementation)
Bump        : none (same rule as RFC-0005/0006: next MINOR reserved for
              the next opcode family)
Note        : the archived blind-spot plan also titles a different
              "RFC-0012" (Clock Resilience, content-free). This registry
              assigns numbers by implementation; clock work will take the
              next free number when specified.
```

## Abstract

Implements `FOREST 0x22`: vectorized traversal of tree ensembles
(XGBoost/Random Forest style) over a flat node table. Stateless (Family
1): no snapshots, preemption drops the chunk. Fixes the table encoding
the frozen spec left implicit (stride layout, leaf marker, termination
bound) and proves parity against a scalar reference evaluator.

## Motivation

The scenarios' intent classifier is XGBoost; scalar `COMPARE+JUMP` per
node branch-mispredicts and does not scale to thousands of trees. A
single vectorized walk over contiguous tables is the right primitive —
and the same op covers Random Forest, LightGBM-style ensembles, and
`TREES=1` decision trees (`LOOKUP_TREE` alias).

## Specification

```text
0x22 FOREST  rdest=scores, rsrc1=features [1,F], rsrc2=node table,
             rsrc3=leaf values.
  payload[0..2]=n_trees u16, [2..4]=max_depth u16,
  [4]=mode (0=per-tree values, 1=mean).
```

Table encoding (frozen spec refined — this layout IS the spec):
- `table`: f32 tensor `[n_trees*stride, 4]`, `stride = 2^max_depth - 1`
  (complete-tree slots; unused slots ignored via traversal bounds).
  Row = `[feat_idx, thresh, left, right]` with feat/left/right integral.
- `leaves`: f32 tensor `[n_trees*stride]`, aligned slot-for-slot.
- Leaf marker: `left < 0` (any negative) => leaf; value =
  `leaves[t*stride + node]`. (Frozen spec was silent; negative-child
  sentinel chosen over NaN so NaN payloads trap instead of hiding.)
- Walk per tree: `node=0`; repeat at most `max_depth` times:
  `f = features[feat_idx]` (OOB feat_idx => deterministic Err);
  `node = (f > thresh) ? right : left` (right/left validated
  `< stride`, else Err); stop early on leaf marker.
- The `max_depth` iteration bound doubles as the termination guarantee:
  cyclic tables cannot hang the VM (a zero table walks node 0 and
  yields its leaf value).
- Output: mode 0 => `[1,n_trees]` per-tree leaf values (caller
  aggregates); mode 1 => `[1,1]` arithmetic mean. NaN leaf value =>
  deterministic Err (no silent poison; use SANITY_CHECK upstream).

Assembler: `FOREST rD, rF, rT, rL [TREES=n] [DEPTH=d]
[MODE=VOTE|MEAN]` (`VOTE` = per-tree values, legacy name kept).

## Backwards Compatibility

Purely additive (const/ctor/parse/dispatch/stat). `0x22` previously
errored at execute.

## Security Considerations

- All indices validated before dereference (feat `< F`, child `<
  stride`, slots within table len): malformed tables trap, never OOB.
- Depth bound makes worst-case cost `O(n_trees * max_depth)` —
  no input-dependent hang; no timeout needed.
- NaN leaves trap rather than propagate into decisions (fail-closed
  for classifier use).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_FOREST`, `forest_params/set`, `instr_forest`,
  mnemonic, parse arm, roundtrips.
- `src/vm.rs`: `exec_forest`, dispatch arm, `forest_execs`.
- Conformance: hand-built 2-tree golden (exact values), OOB
  feat/child errors, NaN-leaf trap, depth-bound termination on cyclic
  table, mode MEAN parity, strict program-level trap on TENSOR-ramp
  tables (`vm::test_rfc0012_ramp_table_traps`).
- Suite: `cargo test --lib` green (sole failure: pre-existing
  unrelated moshi norm-gamma).
- NOTA DE AUDITORIA, parte 2 (RFC-0019: RESOLVIDO): `TENSOR FILL=0`
  permite tabelas zeradas e `forest_demo.m3asm` RESSUSCITOU
  (forest_execs=2, terminação pelo teto depth verificada ao vivo).
  Segue valendo que tabelas *variadas* exigem INIT de literais
  (follow-up permanente).

Follow-up (NOT this RFC): categorical splits (float-only today),
missing-value default directions, `SCATTER_MAX`-style leaf
combinators, GPU walk, `TENSOR`-INIT/fill de literais (sem ele,
programas `.m3asm` não montam tabelas — ver nota de auditoria acima),
`SLICE 0x2E` (idem para o packed do `DISTANCE`).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0012-00 | 2026-09-10 | DRAFT: vectorized walk + table spec + demo |
| 0012-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green.
  Demos `.m3asm` removidos após auditoria (TENSOR-ramp inviabiliza
  tabelas; exit 0 escondia morte de contexto) — cobertura nos goldens. |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
