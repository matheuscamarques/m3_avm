# M³-AVM ISA — Tabela Canônica (`0x00–0xFF`)

> Fonte de verdade do **mapa de opcodes**. Semântica detalhada por faixa nos docs
> linkados; implementação em `src/opcodes.rs` (assembler/disassembler) e `src/vm.rs`
> (dispatcher). Formato fixo 32B: `[op|flags|rdest|rsrc1|rsrc2|rsrc3|payload:26]`.
>
> Convenções: payload em little-endian; `u16/u32/f32` com offsets documentados por
> opcode; `_`/`0xFF` = registrador não usado. Estado stateful segue a taxonomia
> `PLANO_ISA_UNIVERSAL.md` §3 (stateless = drop; stateful = `FORK` empilha, `ABORT`
> restaura). Encodings **nunca são reutilizados**; aliases (§4) resolvem no assembler.

## 1. Implementados — `0x00–0x19` + `0xFF` (25 opcodes + HALT/NOP)

| Hex | Nome | Domínio | Estado | Spec |
|---|---|---|---|---|
| `0x00` | `HALT` | controle | — | `README.md` §3 |
| `0x01` | `TENSOR` | memória | — | `README.md` §3 |
| `0x02` | `ATTN` | recorrência | `KV_CACHE 0x30` | `README.md` §3 |
| `0x03` | `STREAM` | controle | — | `README.md` §3 |
| `0x04` | `FORK` | controle | snapshot CoW | `README.md` §3 |
| `0x05` | `ABORT` | controle | rollback | `README.md` §3 |
| `0x06` | `SENSE` | controle | — | `README.md` §3 |
| `0x07` | `NORM` | matriz | — | `README.md` §3 |
| `0x08` | `FFN` | matriz | — | `README.md` §3 |
| `0x09` | `EMBED` | matriz | — | `README.md` §3 |
| `0x0A` | `ADD` | matriz | — | `README.md` §3 |
| `0x0B` | `SAMPLE` | controle | — (+`TOPK` futuro, universal G1) | `README.md` §3 |
| `0x0C` | `COMPARE` | controle | — | `README.md` §3 |
| `0x0D` | `IF_EQUAL` | controle | — | `README.md` §3 |
| `0x0E` | `JUMP` | controle | — | `README.md` §3 |
| `0x0F` | `IF_INTERRUPT` | controle | — | `README.md` §3 |
| `0x10` | `MATVEC` | matriz | — (+flag `TRANSPOSE` futura, universal §8.3) | `README.md` §3 |
| `0x11` | `MUL` | matriz | — | `README.md` §3 |
| `0x12` | `SILU` | matriz | — | `README.md` §3 |
| `0x13` | `SSM_SCAN` | recorrência | `Vm::ssm_states` | `docs/ISA_OPCODES_0x13_0x19.md` |
| `0x14` | `SSM_RESET` | recorrência | `Vm::ssm_states` | `docs/ISA_OPCODES_0x13_0x19.md` |
| `0x15` | `CODEC_ENC` | áudio | — | `docs/ISA_OPCODES_0x13_0x19.md` |
| `0x16` | `CODEC_DEC` | áudio | — | `docs/ISA_OPCODES_0x13_0x19.md` |
| `0x17` | `AUDIO_ALIGN` | áudio | — | `docs/ISA_OPCODES_0x13_0x19.md` |
| `0x18` | `CTX_SWITCH` | controle | fence + snapshot | `docs/ISA_OPCODES_0x13_0x19.md` |
| `0x19` | `ROPE` | áudio/matriz | — | `docs/ISA_OPCODES_0x13_0x19.md` |
| `0xFF` | `NOP` | controle | — | `README.md` §3 |

## 2. Reservados — planos congelados (assembler/VM rejeitam com erro explícito)

| Faixa | Nome | Plano |
|---|---|---|
| `0x1A–0x1D` | `REMOTE_SPAWN, SIGNAL, SEND_TENSOR, BARRIER` (cluster) | `docs/PLANO_AVM_CLUSTER.md` §2 |
| `0x1E–0x21` | `CONV, GATHER, SPIKE_STEP, DENOISE_STEP` (universal onda 1) | `docs/PLANO_ISA_UNIVERSAL.md` §2 |
| `0x22–0x25` | `FOREST, DISTANCE, RANK1_UPDATE, ODE_STEP` (universal onda 2) | `docs/PLANO_ISA_UNIVERSAL.md` §8 |

## 3. Livres — `0x26–0xFE` (alocação só via proposta + §5 de `PLANO_ISA_UNIVERSAL.md`)

## 4. Aliases — nomes que **não** ganham encoding (resolvidos no assembler)

`MATRIX_RECURRENCE→RANK1_UPDATE`, `LOOKUP_TREE→FOREST` (`TREES=1`),
`ODE_SOLVER→ODE_STEP`, `SCATTER_ADD→GATHER MODE=SCATTER_ADD`.
Tabela completa + lowering (Titans-TTT, fractais, `DOT`, distâncias, SVD):
`docs/PLANO_ISA_UNIVERSAL.md` §11.

## 5. Versionamento

`ISA v1.3` = `0x00–0x19` + `0xFF` implementados. Minor bump a cada opcode novo
(`v1.4` = cluster `0x1A–0x1D`, `v1.5` = universal onda 1…); encodings existentes são
imutáveis. Demos por opcode em `programs/` (`mamba_scan_demo`, `moshi_loop_v2`,
`codec_loop`); futuros: `gather_moe_demo`, `snn_demo`, `conv_demo`, `denoise_demo`,
`rag_search_demo`, `deltanet_demo`, `forest_demo`, `ode_demo`.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
