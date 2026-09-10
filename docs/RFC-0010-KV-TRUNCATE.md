# RFC-0010 — KV_TRUNCATE (`0x38`): Rolling KV Hygiene

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x38 DRAFT->IMPL)
Obsoletes   : None
Feature Bit : none (v2.0-zone early implementation)
Bump        : none (same rule as RFC-0005/0006: next MINOR reserved for
              the next opcode family)
```

## Abstract

Exposes the existing `kv_cache_truncate` machinery (memory-level,
already tested) as opcode `KV_TRUNCATE 0x38`: truncate all KV layers to
a register-held length, with an explicit stream-id reservation for the
future 17-stream Moshi cache. Thin by design — the hard parts (append
paths, snapshot/restore coverage) already exist and stay untouched.
Enables rolling-window KV management and the speculative-decode sync
narrative (`KV_TRUNCATE` after rejected tokens) without new state.

## Motivation

Long runs grow KV without bound (`MAX_SEQ 2048` then append errors);
rejected speculative tokens leave stale KV behind with no surgical
removal except full `ABORT`. A one-operand truncate closes both at
minimal surface.

## Specification

```text
0x38 KV_TRUNCATE Rs_len [, STREAM=sid]
  rsrc1 = reg holding new length (u64 -> usize; saturates at usize::MAX,
          never wraps).
  payload[0..2] = stream_id u16. MUST be 0 today (single stream);
          nonzero yields explicit Err("multi-stream: follow-up").
  Effect: every layer truncated to min(current, len). len >= current
          is a no-op Ok. Empty cache is a no-op Ok.
  Rollback: KV layers already ride memory snapshot/restore
          (kv_snapshots); no new snapshot discipline.
```

Assembler: `KV_TRUNCATE rLen [STREAM=n]` (default stream 0).

## Backwards Compatibility

Purely additive: one const/ctor/parse/dispatch/stat. No encoding,
payload, or snapshot-format change. `0x38` previously errored at
execute.

## Security Considerations

- Truncation only shrinks (never exposes uninitialized reads: `Vec`
  truncate keeps len ≤ capacity invariant; `seq_len` recomputed from
  actual lens).
- `usize` conversion saturates; no wrap on 128-bit hosts.
- Stream gate prevents silent cross-stream truncation before streams
  exist.

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_KV_TRUNCATE`, `kv_stream()`/`set_kv_stream()`,
  `instr_kv_truncate`, mnemonic, parse arm, roundtrips.
- `src/vm.rs`: `exec_kv_truncate`, dispatch arm, `kv_truncate_execs`.
- Conformance: shrink/no-op/empty/stream-gate/rollback tests +
  `programs/kv_truncate_demo.m3asm` exit 0. Suite 214 passed
  (sole failure: pre-existing unrelated moshi norm-gamma).

Follow-up (NOT this RFC): 17-stream KV (`STREAM=sid` real),
`KV_COMPRESS`/`KV_PIN`/`KV_RETRIEVE` (need eviction/recompute design),
stream-aware snapshot policy.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0010-00 | 2026-09-10 | DRAFT: thin truncate wrapper + demo |
| 0010-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
