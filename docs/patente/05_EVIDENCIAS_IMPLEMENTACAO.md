# EVIDÊNCIAS DE IMPLEMENTAÇÃO — reivindicação × código × teste

> Finalidade: dar ao agente de PI e ao perito o **lastro de suficiência
> descritiva (art. 24 LPI)**. Caminhos verificados em 2026-09-10 contra o
> tree. `(C2)` = sem implementação — indica payload congelado, não arquivo.
> Rode `cargo test --lib` e `lake build` em `formal/` e anexe saídas datadas.

## C1 — Habilitado

| Reiv. | Elemento | Código (arquivo:símbolo) | Teste / demo (comando) | Prova formal |
|---|---|---|---|---|
| 1a | Instrução 32B, regs, LE | `src/opcodes.rs:26 INSTR_SIZE`, `:34-102` opcodes, `docs/ESPEC.md §4` | `cargo test --lib opcodes` (roundtrip); `assemble/disassemble` em `README §9` | — |
| 1a | 4 regiões + u128 | `src/memory.rs:24-27` tags, `:71-86` u128, `GLOBAL/TEMPORAL/PERSISTENTE/KV_CACHE` | `qa::snapshot_restore_*`, `cargo test --lib memory` | `Rollback.lean` modelo |
| 1a | GGUF mmap zero-copy | `src/memory.rs:506 get_tensor_f32_slice`, `src/gguf.rs`, `src/inference.rs:180 weight_cache`, `:370 matvec_weight_pre` | `run --model tinyllama... --real --max-steps 3` (~0.33 s/tok) | — |
| 1b | Contexto 16 regs + prio + deadline | `src/context.rs:93` regs, `:20-28` Priority, scheduler 3 FIFOs | `context::*` tests; `SET_DEADLINE/GET_DEADLINE` goldens (RFC-0006) | — |
| 1c | Transformer IMPL | `src/vm.rs:418` dispatch ATTN, `:906` denso, `:1380` NORM, `:1476` FFN, `ROPE` via `inference::apply_rope` | `attn_topk_demo`, `control_flow_demo.m3asm` | `TopK.lean` (esqueleto), `SSM.lean` p/ SSM |
| 1c | SSM IMPL | `src/ssm.rs` scan, `src/vm.rs` exec `SSM_SCAN/RESET`, `ssm_states` stack | `mamba_scan_demo.m3asm`, `test_rank1_modes_and_rollback` (espelho) | `SSM.lean` provado |
| 1c | Áudio IMPL | `src/mimi.rs` codec, `src/moshi.rs`, `CODEC_ENC/DEC`, `AUDIO_ALIGN` | `codec_loop.m3asm`, `moshi_loop_v2.m3asm` | T6 argumento (`ESPEC §11`) |
| 1c | CTX_SWITCH fence | `src/context.rs` pipeline+prio+fence+`maybe_preempt()` | `ctx_switch` tests, demos híbridas | `WindowedError.lean` (aritmética) |
| 1d-e | FORK/ABORT CoW + I-Mono | `src/memory.rs:475 snapshot (Arc::clone)`, `:489 restore` sem rebaixar contador, `Vm::ssm_states` push/pop, `rank1_layers` idem, `DEFAULT_SNAPSHOT_WINDOW=16` | `qa::snapshot_restore_monotonic_no_clobber`, `qa::rollback_100_50_50_bit_exact`, `FORK CoW <1ms` bench | `Rollback.lean` (`restoreFix`, `fresh_of_inv`, `clobber_demo`) |
| 1f | Deadline abs + Red>Blue>Green | `src/opcodes.rs:82 OP_SET_DEADLINE (0x71)`, `src/vm.rs:873,2982`, `src/context.rs:15` | RFC-0006 goldens (deadline/prio/try-lock), `telemetry_demo` | — |
| 2 | Janela k=16 | `src/memory.rs:53 DEFAULT_SNAPSHOT_WINDOW`, RFC-0003 | 3 testes de conformidade RFC-0003 | — |
| 4 | GATHER/DISTANCE/RANK1+TOPK | `src/opcodes.rs:63-65`, `src/vm.rs:exec_gather/distance/rank1` | RFC-0004: `test_gather_golden_and_oob`, `test_distance_four_metrics_and_topk`, `test_rank1_modes_and_rollback`, demos `gather_moe_demo/rag_search_demo/deltanet_demo/attn_topk_demo` | T4 perna empírica |
| 5 | KV_TRUNCATE + SSM_RESET | `src/opcodes.rs:93 OP_KV_TRUNCATE 0x38`, RFC-0010 | 4 arbiters RFC-0010, `kv_truncate_demo` | — |
| 6 | RNG/telemetria | `0x60-0x66`, `0x6A-0x77`, RFC-0005/0006/0007/0009 | NIST/FNV/CRC vetores, fork-inheritance, two-VM agreement, 26 formas rejeitadas (RFC-0008) | — |
| 7 | Assembler 2-pass | `src/opcodes.rs:332` assembler, `:169-172` rejeição | corpus gate 27/27 (RFC-0008) | — |
| 11 | Sistema C1 | `src/main.rs`, `src/lib.rs`, `src/vm.rs` + `Cargo.toml` AGPL | `cargo test --lib` (237 verdes + 1 falha pré-existente `moshi::test_gguf_qkv_split_shapes`, dado do modelo, não regressão — ver `ESPEC §16`) | `lake build` verde exceto TopK `sorry` |
| 4/9 | FOREST/DENOISE/ODE/SPIKE_STEP | `src/opcodes.rs:99-103` (`0x22`/`0x21`/`0x25`/`0x20`), `src/vm.rs` exec + `snn_layers` + snapshots, RFC-0012–0015 | Goldens + demos `snn_demo/denoise_demo/ode_demo` exit 0 (`--max-steps 50`); `forest_demo` removido após auditoria RFC-0012 (TENSOR-ramp inviabiliza tabelas; cobertura nos goldens) | — |

**Obrigação aberta declarada (não afirmar como pronta):** I-Persist
(`ESPEC §6.3`, item 1) — REQUIRED, não fiscalizado. Registradores fora do
rollback (RFC-0004 follow-up). `SENSE`-flood sem rate-limit. Cross-hardware
replay não reivindicado. `wgpu` só `ATTN≤64`.

## C2 — Congelado, sem execução (não usar como prova de suficiência)

| Reiv. | Elemento | Codificação congelada | Habilitação exigida |
|---|---|---|---|
| 8–9 | AND/OR/N_OF_M/CHAIN, EDF, NUMA/NVLink, herança | **Nenhuma** (números `0xD0–0xD3` do rascunho original não existem no mapa; usar regra `ESPEC §6.4/§19`: novo RFC obrigatório) | RFC + testes + Lean (herança, equivalência X-form) |
| 10 | KV_TRANSFER ridge | Não existe (`grep KV_TRANSFER` vazio) | RFC + calibração 500×1024 + demo + prova T4 completa |
| 12 | 64B + 16 regiões + LOAD_MODEL/SET_AFFINITY | `ESPEC-V2 §§3–4,7` (DRAFT), `0xA0–0xA5` DRAFT | W1/W2/W10 (`ESPEC-V2 §14`) |
| 13 | Cluster WAL | `0x1A–0x1D` RSVD (`ESPEC §12`), X-forms `0x80–0x8F` DRAFT | F0→F5 sem pular F2; `ClusterWAL.lean` estendido p/ dupla nomeação |
| 14 | ChaCha/cookie/mTLS | Só cookie previsto (`ESPEC §12`); sem cripto no tree | RFC segurança + auditoria; **não afirmar ChaCha pronto** |
| 15 | OTLP/W3C | Só `TRACE_EVENT 0x6B` local | Exportador + conformidade OTLP |

## Notas de honestidade para o perito

1. `docs/arq/CENARIO-MULTIMODEL.md` está carimbado como **entrada arquivada,
   não normativa** e declara “nada disso compila hoje” — não é anterioridade
   de execução, é spec de alvo.
2. Medições fora de `ESPEC §17`/`README §5` são `alvo`, inclusive 5–50 ms
   cross-processo, 217 µs, 500 ms fim-a-fim, NVLink sem cópia.
3. Falha `moshi::test_gguf_qkv_split_shapes` é de dado do GGUF local, idêntica
   em tree pristino — documentada em `ESPEC §16`, não esconder.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
