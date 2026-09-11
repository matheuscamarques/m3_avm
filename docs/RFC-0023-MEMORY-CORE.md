# RFC-0023 — Memory Core: `ARENA_ALLOC`/`ARENA_RESET` + `MEMCPY`/`MEMSET` (`0x26/0x27/0x2A/0x2B`)

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3.4 (0x26,0x27,0x2A,0x2B DRAFT->IMPL);
              docs/ESPEC.md §16 (row)
Obsoletes   : None
Feature Bit : none (core opcodes)
Bump        : none (same rule as RFC-0005/0006: next MINOR reserved for
              the family; RFC-0024 completes 0x26-0x2F and takes v1.5->v1.6)
Note        : desvio do RFC-0020 §3 (que previa arena+snapshot/restore
              aqui): MEMCPY/MEMSET vieram junto porque ALLOC sem mover
              dados é morto; SNAPSHOT/RESTORE (0x28/0x29) + PREFETCH/
              RESHAPE/CONCAT ficam para RFC-0024, mesmo minor.
```

## Abstract

Implements the memory-core family: bump-allocated per-context arenas
(`ARENA_ALLOC` with size+align, O(1) `ARENA_RESET`) plus byte-exact
tensor data movement (`MEMCPY` with offsets/len/dir, `MEMSET` with
pattern+len+offset). Arenas live Vm-side (`HashMap<u8, Arena>`) with
FORK-clone/ABORT-pop mirroring `rank1_layers` (RFC-0004); tensor writes
go through `MemoryManager::read/write`, whose `Arc::make_mut` gives
CoW — snapshots stay valid by construction (I-Persist, RFC-0016).
Writes to PERSISTENT are an explicit `Err` (WEIGHTS always RO, ESPEC
§7.4); GPU/NIC dirs are explicit `Err` (no driver yet); sparse tensors
are explicit `Err` (dense-only in this RFC, same rule as RFC-0004
GATHER). OOB is a deterministic `Err`, never truncation or clamping.

## Motivation

Per the ESPEC Section 13 filter: nothing expresses bump allocation
with O(1) reset, tensor-to-tensor copy with explicit offsets, or
pattern fill at acceptable cost over existing opcodes (`TENSOR FILL`
covers init-at-alloc only, RFC-0019). These four unblock arena-backed
scratch (codec frames, attention buffers), KV compaction loops, and
the `CONCAT` implementation (RFC-0024 needs MEMCPY internally).

## Specification

All 32B. Tensor addrs travel in regs (u128, SAMPLE precedent).

```text
0x26 ARENA_ALLOC  rdest <- byte offset u64 (as u128).
  payload[0..8]=size u64, [8..12]=align u32 (0 selects default 16),
  [12]=arena_id u8. Bump-allocates size bytes at align in arenas[id]
  (created on first use); grows by doubling; rdest = offset.
  size==0 => Err (empty allocation is a caller bug). align must be a
  power of two (default 16 when 0), else Err. ARENA_MAX_BYTES = 1 GiB
  per arena (explicit Err beyond; tunable, see §Security).
0x27 ARENA_RESET  payload[0]=arena_id u8. cursor=0, capacity kept: O(1).
  Unknown arena => Err (no silent create). rdest unused (0xFF).
0x2A MEMCPY  rdest reg holds DST tensor addr, rsrc1 reg holds SRC tensor
  addr. payload[0..8]=len u64 (0 = whole source), [8..16]=src_off u64,
  [16..24]=dst_off u64, [24]=dir (0=HOST,1=GPU,2=NIC).
  len==0 selects whole-source copy and requires both offsets 0, else Err.
  src_off+len <= src_bytes and dst_off+len <= dst_bytes, else Err (never
  truncate). dir!=0 => Err UnsupportedDir (no driver yet). Sparse either
  side => Err SparseUnsupported. DST in PERSISTENT => Err (WEIGHTS RO).
  Both sides must resolve via tensor_meta, else Err. Overlap-safe
  (read-then-write through an owned buffer: memmove semantics).
  CoW-safe: writes go through MemoryManager::write (Arc::make_mut).
0x2B MEMSET  rdest reg holds tensor addr. payload[0]=pattern byte,
  [1..5]=len u32 (0 = whole tensor), [5..13]=offset u64. Byte-level fill
  (not value-level; value fill is TENSOR FILL, RFC-0019). len==0 selects
  whole tensor and requires offset 0, else Err. Bounds-checked, else Err.
  Sparse => Err. PERSISTENT => Err (WEIGHTS RO). Tensor must resolve via
  tensor_meta, else Err.
```

Arena state: `Vm::arenas: HashMap<u8, Arena>`, `Arena { buf: Vec<u8>,
cursor: usize }` (`src/arena.rs`); `Vm::arena_snapshots:
Vec<(u64, HashMap<u8, Arena>)>`; FORK pushes a map clone (cost O(total
arena bytes), documented), ABORT version-gated-pops (mirrors
rank1_layers, RFC-0011 discipline).

Assembler (numeric forms only; strict lists gain nothing beyond these):
```text
ARENA_ALLOC rD, SIZE=n [ALIGN=n] [ARENA=id]
ARENA_RESET [ARENA=id]
MEMCPY rDst, rSrc [LEN=n] [SRC_OFF=n] [DST_OFF=n] [DIR=HOST]
MEMSET rT, PATTERN=n [LEN=n] [OFF=n]
```
`DIR=` accepts only `HOST` (numeric 0 also accepted); `GPU`/`NIC` parse
but trap at execute with `UnsupportedDir` (parsed, not silent — the
assembler knows the words so programs read clearly; execution vetoes).

## Backwards Compatibility

Purely additive (consts/ctors/parse/dispatch/stats). `0x26/0x27/0x2A/
0x2B` previously errored at execute (`opcode não implementado`).
`load_program`, corpus programs, and all 32B encodings untouched.

## Security Considerations

- `ARENA_MAX_BYTES` (1 GiB/arena) bounds the only new exhaustion
  vector; growth is doubling (amortized O(1) alloc, no fragmentation
  oracle beyond cursor position, which is program-visible anyway).
- PERSISTENT-RO enforcement keeps weights immutable through this path
  (a MEMSET over weights would be silent model corruption — now an
  explicit trap).
- OOB traps (no wrap, no clamp): attacker-controlled offset/len tensors
  cannot read or write out of bounds; `len` is explicit, never inferred
  from undersized destinations.
- TEMPORAL addrs are not tensors (no meta) => clean Err, not ring
  confusion. KV_CACHE writes ride CoW (ABORT restores).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/arena.rs` (new): `Arena { buf, cursor }`, `alloc(size, align)`
  with align-up + doubling growth + 1 GiB cap, `reset()` O(1),
  `ArenaError` variants, unit tests (offsets, alignment, reuse,
  cap, zero-size/align errors).
- `src/opcodes.rs`: `OP_ARENA_ALLOC/RESET/MEMCPY/MEMSET` + consts
  (`ARENA_DEFAULT_ALIGN=16`, `ARENA_MAX_BYTES`, `MEMCPY_DIR_HOST/GPU/
  NIC`), `arena_alloc/arena_reset/memcpy/memset_params` + setters,
  `instr_arena_alloc/reset/memcpy/memset`, mnemonics, `parse_line` arms,
  roundtrips.
- `src/vm.rs`: `arenas` + `arena_snapshots`, FORK-push/ABORT-pop,
  `exec_arena_alloc/reset/memcpy/memset`, dispatch arms, 4 counters.
- Conformance: alloc offsets/alignment/reuse-after-reset/cap, memcpy
  bit-exact + partial windows + whole-tensor + overlap self-copy + OOB
  src/dst + sparse + persistent-dst + dir + missing-meta errors, memset
  pattern/whole/bounds/sparse/persistent errors, arena FORK snapshot +
  ABORT restore (live-map compartilhado, disciplina rank1), assembler
  roundtrips + strict rejections, combined program
  (`vm::test_rfc0023_assembled_program_runs`: counters exact),
  `programs/arena_memcpy_demo.m3asm` (alloc=3, reset/memcpy/memset=1,
  verified).
- Suite: `cargo test --lib` 286 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate: all `programs/*.m3asm` +
  `examples/*.m3asm` assemble (verificado por loop CLI).
- Benches: `benches/memory_bench.rs` (memcpy/memset 4 KiB + arena
  64B alloc+reset, per-instruction via `step_instruction`; medians
  `--quick` neste host: memcpy ~1.45µs, memset ~723ns, arena ~783ns —
  referência, não §17).

Follow-up (NOT this RFC): RFC-0024 (`SNAPSHOT`/`RESTORE` + `PREFETCH`/
`RESHAPE`/`CONCAT`, same v1.6 minor); per-arena quotas; GPU/NIC dirs;
sparse MEMCPY; raw-region (non-tensor) forms; `ARENA_FREE` (bump has no
free by design — needs a different allocator, own RFC).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0023-00 | 2026-09-11 | DRAFT: memory-core family + demo |
| 0023-01 | 2026-09-11 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
