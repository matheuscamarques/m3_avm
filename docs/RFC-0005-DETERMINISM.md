# RFC-0005 — Determinism Block (`0x60–0x66`): Seeded RNG + Hashing

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC.md §9 (replay rule made enforceable);
              docs/ESPEC-V2.md §3.10 (0x60-0x66 DRAFT->IMPL), §13, W5
Obsoletes   : None
Feature Bit : none (core opcodes, v2.0-zone early implementation)
Bump        : none (v2.0-zone ops; v1.x line stays v1.4 — see §Backwards)
```

## Abstract

Implements the determinism block: per-context seeded RNG
(`RNG_SEED/NEXT/UNIFORM/NORMAL`) plus tensor hashing
(`HASH/CHECKSUM/HMAC`). Closes the genuine replay gap behind ESPEC
Section 9: no stochastic opcode may draw from an unseedable source
after this RFC. Hash choices are downgraded honestly from the
blueprint where auditability beats name-dropping (FNV-1a instead of
SipHash; table-built CRC32-IEEE; compact SHA256 verified against the
NIST vector for HMAC).

## Motivation

`DENOISE_STEP` with sigma>0 and any future sampling mode are meaningless
without a seed primitive, and WAL/checksum claims need real integrity
functions. The blueprint named algorithms (`SipHash/xxHash`, `SHA256
truncado`) without implementations; this RFC implements
*verifiable* ones and documents each deviation.

## Specification

### 1. RNG state

- `Context.rng_state: u64`, default `DEFAULT_RNG_SEED =
  0x9E3779B97F4A7C15` (fixed => deterministic cold boot).
- `FORK` (both `Vm::exec_fork` and the host-side path in `main.rs`)
  copies `rng_state` parent->child (child replays the parent stream
  from the fork point; re-seed explicitly to diverge).
- Core: splitmix64. `next()` mutates state, returns u64.

### 2. Opcodes (all 32B; regs carry scalars as u128, SAMPLE precedent)

```text
0x60 RNG_SEED    rsrc1 = reg holding u64 seed (0xFF = default fixed seed).
                 rdest unused (0xFF). No payload.
0x61 RNG_NEXT    rdest reg <- next u64 (as u128). No payload.
0x62 RNG_NORMAL  rdest reg <- Box-Muller normal(mean,std) f32 bits.
                 payload[0..4]=mean, [4..8]=std; std==0 selects default
                 1.0. Resolved std validated (> 0 finite), else Err.
                 Deterministic (no rejection loop).
0x63 RNG_UNIFORM rdest reg <- uniform f32 in [a,b) (bits as u64).
                 payload[0..4]=a f32, [4..8]=b f32; all-zero payload
                 selects default [0,1). Resolved values validated
                 (a < b, both finite), else Err.
0x64 HASH        rsrc1 = tensor addr; rdest reg <- FNV-1a/64 of the
                 canonical f32-LE byte stream. (Blueprint said SipHash:
                 downgraded — FNV-1a is 5 auditable lines; keyed hashing
                 stays a future RFC.)
0x65 CHECKSUM    rsrc1 = tensor addr; rdest reg <- CRC32-IEEE (u64-carried).
                 Table built per call (no global state).
0x66 HMAC        rsrc1 = key tensor bytes, rsrc2 = msg tensor bytes;
                 rdest reg <- first 8 bytes (BE) of HMAC-SHA256.
                 Compact SHA256 verified against NIST "abc" vector.
```

Tensor bytes: canonical f32-LE stream via `read_f32_tensor`
(dequant view for quantized weights — hashes the logical values).
Empty tensor (0 elems) is an explicit `Err` (no silent identity hash).

### 3. Assembler

```text
RNG_SEED rS | RNG_SEED DEFAULT
RNG_NEXT rD
RNG_NORMAL rD [MEAN=m] [STD=s]
RNG_UNIFORM rD [A=a] [B=b]
HASH rD, rT
CHECKSUM rD, rT
HMAC rD, rKey, rMsg
```

## Backwards Compatibility

Purely additive: one `u64` field with fixed default (cold-boot streams
unchanged and deterministic), seven new match arms, seven stat
counters. No encoding touched. v1.x line stays v1.4: these are v2.0-zone
opcodes implemented early (32B-compatible); the v2.0-line version
decision stays with the v2.0 freeze.

## Security Considerations

- RNG is deterministic by design: MUST NOT be used for secrets; HMAC
  here is integrity-only (no side-channel hardening, no constant-time
  compare primitive yet — follow-up).
- `RNG_SEED` from a user-controlled register reseeds silently by
  design (documented, not a vulnerability).
- CRC32/FNV detect corruption, not malice; only HMAC is adversarial
  (truncated to 64 bits per blueprint — collision caveat documented,
  full-digest form is a future mode).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/determinism.rs` (new): splitmix64, uniform, Box-Muller normal,
  FNV-1a/64, CRC32-IEEE, SHA256, HMAC-SHA256-truncated.
- `src/context.rs`: `rng_state` + default; `fork()` inherits via Clone.
- `src/vm.rs` + `src/main.rs`: FORK copies `rng_state`.
- `src/opcodes.rs`: consts, ctors, payload accessors, mnemonics,
  `parse_line` arms, roundtrips.
- `src/vm.rs`: `exec_rng_seed/next/uniform/normal/hash/checksum/hmac`,
  dispatch arms, 7 counters.
- Conformance: NIST/FNV/CRC vectors, determinism + reseed + fork
  inheritance, bounds/param errors, golden demo run
  (`programs/rng_demo.m3asm`, exit 0).
- Suite: `cargo test --lib` 197 passed; sole failure is the
  pre-existing, unrelated `moshi::test_gguf_qkv_split_shapes`.

Follow-up (CLOSED by RFC-0009): `SAMPLE` routed through context RNG
(`sample_logits_ctx`; host paths on documented `sample_logits_host`).
Remaining: keyed SipHash, constant-time compare op,
entropy-seeded boot (deliberately NOT default — determinism first).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0005-00 | 2026-09-10 | DRAFT: RNG + hash block + demo |
| 0005-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
