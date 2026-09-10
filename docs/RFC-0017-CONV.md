# RFC-0017 — CONV (`0x1E`): Sliding Convolution 1D/2D

```text
Status      : IMPLEMENTED
Category    : Standards Track
Updates     : docs/ESPEC-V2.md §3 (0x1E DRAFT->IMPL)
Obsoletes   : None
Feature Bit : none (v2.0-zone early implementation)
Bump        : none (same rule as RFC-0005/0006: next MINOR reserved for
              the next opcode family)
```

## Abstract

Implements `CONV 0x1E`: direct sliding cross-correlation (no materialized
im2col) for 1D (`[N,C,L]`) and 2D (`[N,C,H,W]`) with stride/pad/dilation,
groups (incl. depthwise) and fused SiLU/ReLU. Stateless (Family 1).
Settles two things the frozen spec left open: rank adaptation (so
`.m3asm` programs built only from 2D `TENSOR`s can run real
convolutions) and the N-dim boundary (3D+ volumes are an explicit
follow-up, not silent misbehavior).

## Motivation

Last wave-1 holdout with a closed spec: CNN/ConvNeXt stems, temporal
convs adjacent to Mamba, and audio front-ends. Lowering via
im2col+`MATVEC` explodes memory `K·K·C` per position — the textbook case
of the ESPEC Section 13 filter.

## Specification

```text
0x1E CONV  rdest=out, rsrc1=input, rsrc2=kernel, rsrc3=bias|0xFF.
  payload[0..2]=stride u16 (broadcast a todos os eixos),
  [2..4]=pad u16, [4..6]=dilation u16, [6]=groups u8 (0 = 1),
  [7]=fused_act (0=none, 1=silu, 2=relu).
```

Semantics (cross-correlation, ML convention — kernel NOT flipped,
documented):
- Rank adaptation (enables real demos from 2D-only `TENSOR`):
  input `[H,W]` => `[1,1,H,W]`; `[N,C,L]` / `[N,C,H,W]` as-is
  (1D/2D only, else Err). Kernel full form
  `[Cout,Cin,K…]`; reduced forms `[K]` (1D single) and `[KH,KW]`
  (2D single, `Cout=Cin=1`); anything else is Err (no guessing).
- Groups: `Cin % G == 0`, `Cout % G == 0` (else Err); group `g`
  maps `Cin/G` inputs to `Cout/G` outputs; depthwise =
  `G == Cin == Cout`. `groups` payload 0 means 1.
- Geometry per axis: `O = floor((I + 2P - D*(K-1) - 1)/S) + 1`,
  must be `>= 1` (else Err, never empty output). Stride/dilation
  `>= 1` (0 = Err).
- Bias: `rsrc3 == 0xFF` = none; else flat length must equal `Cout`
  (any shape with product `Cout`), added per output channel.
- Fused act applied inline (none/SiLU/ReLU; other values Err).
- Dtypes: dense f32 path; quantized kernel weights flow through the
  existing dequant read (same as `MATVEC`), no special casing.

Assembler: `CONV rD, rX, rW [, rB] [STRIDE=n] [PAD=n]
[DILATION=n] [GROUPS=n] [ACT=NONE|SILU|RELU]`.

## Backwards Compatibility

Purely additive. `0x1E` previously errored at execute.

## Security Considerations

- All geometry validated before allocation (no zero/negative dims,
  no wrap on huge shapes: products computed in `usize` with the
  allocator as backstop).
- Cost is caller-visible `O(N·Cout·O·Cin/G·K)` — direct sliding has no
  hidden im2col blowup by construction (the point of the op).

## Reference Implementation

Working tree (no PR link; single-commit scope):

- `src/opcodes.rs`: `OP_CONV`, `conv_params/set`, `instr_conv`,
  mnemonic, parse arm, roundtrips.
- `src/vm.rs`: `exec_conv` (1D/2D + groups + fused), dispatch arm,
  `conv_execs`.
- Conformance: hand goldens (1D basic, 2D 3×3, stride/pad/dilation,
  depthwise, fused), all error paths, `programs/conv_demo.m3asm`
  (real 3×3 over 4×4, verified executed).

Follow-up (NOT this RFC): N-dim generic volumes (`0x26+` era needs
them least), Winograd/FFT paths, quantized-direct kernels (no-dequant
fast path), `CONV3D` (stays an alias-to-lowering per the universal
plan, not an opcode).

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0017-00 | 2026-09-10 | DRAFT: sliding 1D/2D + groups + fused + demo |
| 0017-01 | 2026-09-10 | IMPLEMENTED: merged to tree, suite green |
