# RFC-0004 — GATHER / DISTANCE / RANK1_UPDATE (+ SAMPLE-TOPK)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC.md §5 (0x1F/0x23/0x24 RSVD->IMPL), §13 G1/H1;
              docs/ESPEC-V2.md §3.3 (status), W3
Obsoletes   : None
Feature Bit : none (core opcodes, negotiated by ISA minor)
Bump        : MINOR (v1.3 -> v1.4: three frozen opcodes implemented, no encoding change)
```

## Abstract

Implements the three highest-value frozen opcodes — `GATHER 0x1F`
(tensor-indirect indexing for MoE/GNN/embedding-bag),
`DISTANCE 0x23` (batched distance + fused top-k for RAG and attention
pruning), `RANK1_UPDATE 0x24` (outer-product matrix recurrence for
DeltaNet/Titans fast-weights) — plus the minimal `SAMPLE TOPK` mode the
MoE-dispatch loop needs. Unlocks MoE dispatch, RAG search, DeltaNet
recurrence and the DISTANCE-fed pruned-attention path (T4 empirical
leg) with zero encoding changes: all payloads are exactly the frozen
ones.

## Motivation

Per the ESPEC Section 13 filter (native only where lowering is
inadequate): nothing expresses tensor-indirect indexing (`GATHER`),
fused top-k retrieval (`DISTANCE`) or outer-product state
(`RANK1_UPDATE`) at acceptable cost over `MATVEC/MUL/ADD`. These three
unblock MoE (dispatch), RAG (search), DeltaNet (matrix memory) and the
only honest sparse-attention path (prune-before-softmax, T4).

## Specification

### 1. `GATHER 0x1F` (frozen payload, kept)

`rdest`=out, `rsrc1`=table, `rsrc2`=u32/i64 indices,
`rsrc3`=`0xFF` (gather) or accumulator (scatter).
`payload[0]`=axis u8, `[1]`=mode
(`0`=GATHER, `1`=SCATTER_ADD, `2`=SCATTER_MAX), `[2..4]`=`nnz_hint`
u16 (advisory). N-D row-major along any valid axis; out-of-bounds
index is a deterministic `Err` (never wrap). Dense only in this RFC;
sparse table yields an explicit `SparseUnsupported` error (no silent
densify). `SCATTER_ADD rD,rT,rI` is an assembler alias
(`MODE=SCATTER_ADD`).

### 2. `DISTANCE 0x23` (frozen payload + one documented refinement)

`rdest`=out, `rsrc1`=query `[1,D]`, `rsrc2`=bank `[N,D]`.
`payload[0]`=metric (`0`=EUCLID, `1`=COSINE, `2`=MANHATTAN, `3`=DOT),
`[1..3]`=topk u16 (`0`=all N). Refinement (no encoding change; the
frozen spec says "+ indices if top-k" without locating them): with
`TOPK=T>0`, `rdest` receives `[1,2T]` f32 — first T entries are the
top-T distances ascending, next T are their indices as f32 (exact for
`N < 2^24`). Ties broken by lower index (deterministic). With
`TOPK=0`, `rdest` is `[1,N]` distances in bank order.

### 3. `RANK1_UPDATE 0x24` (frozen payload + CoW refinement)

`rdest`=H `[d,k]`, `rsrc1`=v `[d]`, `rsrc2`=k `[k]`,
`rsrc3`=`0xFF`/mask. `payload[0..4]`=alpha f32, `[4..8]`=beta f32,
`[8]`=mode (`0`=hebbian `H=aH+b.vkT`, `1`=delta-normalized
`/ (1+b.||k||^2)`, `2`=forget-gated), `[9]`=layer_id.
Refinement: "in-place" is LOGICALLY in-place but PHYSICALLY CoW — each
step allocates the new H and rebinds the layer handle, so snapshots
stay valid (I-Persist-compatible by construction; satisfies the
Section 10 stateful rule with snapshot cost O(d.k) per step and
ABORT = handle rebind). Per-layer current handle lives in
`Vm::rank1_layers: HashMap<u8, u128>`; `FORK` pushes a map clone,
`ABORT` pops (mirrors `ssm_snapshots`).

### 4. `SAMPLE TOPK` mode (no new opcode)

`payload[4..6]`=topk u16 (`0` = legacy sampling; previously unused —
`instr_sample` only wrote `payload[0..4]`=TEMP). With `TOPK=k>0`,
`rdest` receives an indices tensor `[1,k]` (top-k logits descending,
ties by lower index); `last_sample` untouched. Assembler:
`SAMPLE rD, rL TOPK=k`. Serves the MoE gate
(`MATVEC -> SAMPLE TOPK -> GATHER` dispatch).

### 5. Pruned attention (program-level, no ATTN change)

T4's empirical leg was designed as a hand-wired demo
(`DISTANCE TOPK -> GATHER rows of K,V -> ATTN on the subset`), but the
demo was REMOVED in audit (see adendo below): without `SLICE`, the
packed `[dists|idx]` cannot feed `GATHER` cleanly. The leg lives in
`test_distance_four_metrics_and_topk`. `ATTN_SPARSE 0x3B` stays a
later wave.

## Backwards Compatibility

Additive only: three new match arms, one new payload field read
(`SAMPLE[4..6]`, zero in all previously assembled binaries = legacy
behavior), one new Vm map (empty by default), three stat counters.
`0x1F/0x23/0x24` previously errored at execute (`opcode não
implementado`) and still decode identically. ISA minor v1.4 -> v1.5.

## Security Considerations

- OOB indices trap (no wrap, no clamping): attacker-controlled index
  tensors cannot read out of bounds.
- `nnz_hint` is advisory only; allocation follows actual index count
  (no over-allocation oracle).
- Top-k ties broken deterministically (no data-dependent timing
  beyond the partial heap, which is input-size-fixed).
- CoW H means no aliasing between live state and snapshots by
  construction (I-Persist for this state family).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_GATHER/DISTANCE/RANK1_UPDATE` + metric/mode
  consts, `gather/distance/rank1_params` + setters, `instr_gather/
  instr_distance/instr_rank1_update/instr_sample_topk`, `mnemonic`
  arms, `parse_line` arms (`GATHER/SCATTER_ADD/DISTANCE/RANK1_UPDATE`,
  `SAMPLE TOPK=k`), roundtrip tests.
- `src/vm.rs`: `exec_gather/exec_distance/exec_rank1_update`,
  `rank1_layers` + FORK-push/ABORT-pop, TOPK branch in `exec_sample`,
  dispatch arms, `gather/distance/rank1` stat counters.
- Conformance: golden-vs-manual tests (gather N-D + 3 scatter modes,
  4 metrics + top-k pack, rank1 3 modes + map rollback, SAMPLE-TOPK
  ties), OOB determinism, assembler roundtrips, combined program
  (`vm::test_rfc0004_assembled_program_runs`: counters exact).
  Demos `gather_moe_demo`, `rag_search_demo`, `deltanet_demo`
  exit 0 under `run --max-steps 30`.
- Suite: `cargo test --lib` 187 passed; sole failure is the
  pre-existing, unrelated `moshi::test_gguf_qkv_split_shapes`.

Follow-up (NOT this RFC): sparse-table GATHER, `ATTN_SPARSE 0x3B`,
`SAMPLE TOPK` as index source inside `ATTN` (fused path), and the
stale-register hazard (context registers are not part of any rollback;
only memory + layer-handle maps are — pre-existing FORK/ABORT design).

### Adendo de auditoria (RFC-0012, posterior)

O demo `attn_topk_demo.m3asm` foi REMOVIDO: com `TENSOR` inicializando em
ramp `(i+1)*0.5`, o packed `[dists|idx]` do `DISTANCE` alimenta o `GATHER`
com distâncias (ex.: DOT negativas) como índices — OOB determinístico —
e não existe `SLICE 0x2E` para fatiar só os índices. O programa morria em
silêncio com exit 0. A perna empírica do T4 vive nos goldens
(`test_distance_four_metrics_and_topk`); o demo volta com `SLICE`.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0004-00 | 2026-09-10 | DRAFT: three opcodes + SAMPLE-TOPK + demos |
| 0004-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
