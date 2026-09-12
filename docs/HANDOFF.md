# Handoff — M³-AVM (continuação do trabalho) — v6 (HEAD + V-4 v1.16)

Contexto denso para IA que vai continuar. Autocontido. Atualiza handoff v5 (V-3A v1.16, 2026-09-11).

---

## 1. Identidade do projeto

**Nome:** M³-AVM (Multimodal Multi-Model Abstract Virtual Machine)
**Autor:** Matheus de Camargo Marques — `matheuscamarques@gmail.com`
**ORCID:** `0009-0003-4518-2258`
**Licença:** AGPL-3.0-or-later (dual-licensing comercial opcional) `Cargo.toml:7` `LICENSE:1`
**Repo:** `https://github.com/matheuscamarques/m3_avm` `Cargo.toml:8`
**Local:** `~/projetos/m3_avm/` (com underscore; `~/projetos/m3avm` não existe)
**O que é:** VM heterogênea para inferência determinística, ISA própria (32B fixo), scheduler EDF RED>BLUE>GREEN, memória CoW+snapshot, Transformer+SSM+SNN+ML clássico+RAG+áudio full-duplex. Infra para décadas, sem quebra binária.
**Não é:** framework, wrapper PyTorch, lib de serving.

## 2. Estado atual (2026-09-11, HEAD + V-4 v1.16)

| Item | Valor (HEAD) |
|:---|:---|
| **ISA normativa** | `v1.16` `docs/ESPEC.md:7` (116 opcodes: `0x00-0x49`+`0x4A-0x4D`+`0x50-0x53,0x56-0x57`+`0x60-0x69,0x6A-0x79,0x7A-0x7E`+`0xFF`) |
| **ISA implementada** | `v1.16` — 116 opcodes (6 retrieval + 4 system) `src/opcodes.rs:142` |
| **Testes** | `379 passed; 1 failed` `moshi::test_gguf_qkv_split_shapes` (pré-existente) — `cargo test --lib` (V-4 smokes ~120s) |
| **RFCs** | `0002-0039` (0038+0039 IMPL; 0036-0037 IMPLEMENTED) |
| **Corpus `.m3asm`** | `45/45` monta (`system_demo` 10 instr, `rag_demo` 17) `programs/` |
| **Linguagem** | Rust (runtime); C++ collisor |

**Últimos commits (main):**
- `V-4` (este) — `test_v4_*_forward_real_smoke` (mamba-130m 42s + TinyLlama 74s) fecha `ESPEC §16 OPEN`
- `V-3A mini` — `0x4A-0x4D` + `bit51` + `system_demo.m3asm` + `test_system_ops_mini` (RFC-0039 v1.16)
- `V-2 turno4` — `rag_demo.m3asm` + `rag_bench.rs` + bump `v1.15`

## 3. Estrutura de arquivos

```
m3_avm/
├── src/
│   ├── opcodes.rs           — assembler/disassembler + encoding + @ + V-2 ctors (0x50-0x53/0x56-0x57)
│   ├── vm.rs                — dispatcher + exec_* (rag_* + pq_* + load_assembled)
│   ├── rag.rs               — IndexStore + pq_validate/pq_encode/pq_decode
│   ├── m3bc.rs              — container .m3bc + bits 49/50 + REQUIRED gate
│   ├── memory.rs            — Region impl (GLOBAL/TEMPORAL/PERSISTENT/KV_CACHE u128)
│   ├── bus.rs / context.rs / reactor.rs / rollback.rs
│   ├── inference.rs / matvec_quant.rs / quant.rs / ssm.rs / mimi.rs / moshi.rs
│   └── lib.rs / main.rs / asm_emitter.rs / arena.rs / activations.rs / depformer.rs
├── docs/
│   ├── ESPEC.md             — spec normativa v1.16 (116 opcodes)
│   ├── ESPEC-V2.md          — spec reconciliada v2.0 DRAFT (§3.7 shims IMPL, §3.8 IMPL)
│   ├── RFC-0036-SYMBOLIC-REGS.md  — .reg explícito
│   ├── RFC-0037-EQU-TEXT.md       — .equ/.text/.data/@ + layout
│   ├── RFC-0038-RETRIEVAL.md      — V-2 retrieval IMPL (0038-04)
│   ├── RFC-0039-SYSTEM-OPS.md     — V-3A mini 0x4A-0x4D local shims (bit 51)
│   ├── V-2_SCOPE.md               — scope travado 0x50-0x57
│   ├── PLANO_VISAO.md       — roadmap v1.16 base (V-3A ✅)
│   └── TURN_STATE.md / PLANO_AVM_CLUSTER.md etc.
├── programs/ (45)
│   ├── mamba_scan_demo.m3asm / moshi_loop_v2.m3asm / codec_loop.m3asm
│   ├── data_addr_demo.m3asm / english_tutor_demo.m3asm / voice_loop_demo.m3asm
│   ├── rag_demo.m3asm / system_demo.m3asm / rag_search_demo.m3asm
│   └── (39 outros)
└── Cargo.toml / benches/rag_bench.rs / tests/ / formal/
```

Correção vs handoff v1: `ISA_V2.md` -> `ESPEC-V2.md`; `RFC-0036.md` -> `RFC-0036-SYMBOLIC-REGS.md`; `RFC-0037.md` -> `RFC-0037-EQU-TEXT.md`; `RFC-0038.md` -> `RFC-0038-RETRIEVAL.md`.

## 4. ISA — convenções travadas

**Formato dual** `src/opcodes.rs:346` `docs/ESPEC-V2.md:466`:
- `0x00-0x7F` = 32B (`INSTR_SIZE=32` `src/opcodes.rs:27`)
- `0x80-0xFF` = 64B (`INSTR_SIZE_64=64` `src/opcodes.rs:31`) exceto `0xFF NOP` sempre 32B `docs/ESPEC-V2.md:82`

**Layout 32B:** `[op 1B|flags 1B|rdest 1B|rsrc1-3 3B|payload 26B]` `src/opcodes.rs:6`
**Layout 64B:** `[op|flags|rdest|rsrc1-5 5B|lamport 8B|deadline 8B|payload_ext 8B|payload_core 32B]` `docs/ESPEC-V2.md:455`

**Endereçamento:** implementado `u128` com tags top-byte `0x00 GLOBAL / 0x10 TEMPORAL / 0x20 PERSISTENT / 0x30 KV_CACHE` `src/memory.rs:24`. Modelo lógico 64b `REGION:OFFSET` (4b+60b) `docs/ESPEC-V2.md:520` é reconciliação `R1` `docs/ESPEC-V2.md:73` — compacto zero-estende para `u128`.

**Regiões lógicas 0x0-0xF (ESPEC-V2 §7.1):** `0x0 TEXT` `0x1 GLOBAL` `0x2 WEIGHTS` `0x3 ACTIVATION` `0x4 KV_CACHE` `0x5 ARENA` `0x6 SHARED` `0x7 WAL` `0x8 SNAPSHOT` `0x9 STREAM_RING` `0xA RAG_INDEX` `0xB CLUSTER_STAGING` `0xC FEDERATED` `0xD CONFIDENTIAL` `0xE SCRATCH_DEVICE` `0xF MMIO`. Perfil compat 32B implementa só 4.

**Feature bits** `src/m3bc.rs:56`:
- 0 ESCAPE 1 DUAL_MODE 2 LAMPORT 3 DEADLINE 4 WAL_MIGRATE 5 COW_SNAPSHOT 6 CLUSTER 7 GPU_MMIO 8 TENSOR_REGS 9 REGION_TABLE 10-16 RFC-0002
- 49 `M3BC_OPTIONAL_HAS_DATA_SECTION` (`.data` payload; número congelado, wiring pendente — CLI `assemble` ainda rejeita `.data` sem `assemble_with_data`)
- 50 `M3BC_REQUIRED_HAS_V2_RETRIEVAL = 1<<50` `src/m3bc.rs:61` REQUIRED `src/m3bc.rs:49` — runtime antigo falha alto em `UnknownOpcode` ou `REQUIRED` gate `src/m3bc.rs:240`

**Regras invioláveis:** encodings nunca reutilizados, feature bits nunca removidos, aliases no assembler, `0xFF`=não usado `src/opcodes.rs:107`.

## 5. Decisões arquiteturais travadas (não reabrir)

| Decisão | Onde | Por quê |
|:---|:---|:---|
| Não patentear | Estratégia | AGPL+Zenodo+arXiv prior art |
| AGPL v3 + dual | LICENSE | Proteção Big Tech |
| Inglês em testes/demos | Corpus | Global |
| Rust runtime, C++ collisor | Ecossistema | Cada projeto sua ling. |
| Sem Python runtime | m3_avm+collisor | Eliminado |
| Contrato via `m3-spec/` | Ecossistema | Zero código compartilhado `~/projetos/m3-spec/` com `AUDIO_CONTRACT.md`/`CONVENTIONS.md` |
| Linux-only collisor | m3-spec/CONVENTIONS.md | V4L2 direto |
| Regs simbólicos via `.reg` explícito | RFC-0036 | Auto-alocação quebrou `SANITY_CHECK r4,r0,FOO->r2` |
| `.data` chave seção | RFC-0037 | Consistente com `.equ/.text` |
| Multi-value nested obrigatório | V-1b dia2 | `.f32 [R,C] [...]` |
| `@name` namespace endereço | V-1b dia3 ff21f53 | Desambigua `.equ` valor vs `.data` endereço |
| Loader Opção A assemble-time | V-1b dia3 | `LOADI 0x78` `u128` comporta qualquer endereço `src/opcodes.rs:104` — sem `LOADI64` |
| Queue-driven event sources | TURN_STATE | VAD é evento, não op síncrona |
| V-2: um bit p/ V-2 inteiro | RFC-0038 | bit 50 REQUIRED |

## 6. V-1b dia 3 — FEITO (não mais pendente)

Bloqueio `LOADI 64-bit` verificado em `src/opcodes.rs:104` `src/vm.rs:1193` e `docs/RFC-0037-EQU-TEXT.md:42`: `LOADI rD, imm u128` (payload 0..16) basta. Implementado `ff21f53`:
- `data_layout_addrs`/`data_layout_total` `src/opcodes.rs` — base `DATA_LOAD_BASE=GLOBAL_HEAP_START=0x1000`, align 4 (`f32`/escalar) 1 (`.str`), pad `align8` entre blobs, LE
- `SymbolTable.addrs` + `@nome` só em `LOADI` (fora erra) + colisão blob×const erra
- `Vm::load_assembled(&AssembledProgram)` `src/vm.rs` — primeira `alloc_global(total)` deve dar `0x1000` senão erra (VM não-fresca/2º preload)
- Testes: `opcodes::test_v1b_data_addr` + `vm::test_planoV1b_data_loader` + demo `programs/data_addr_demo.m3asm`

## 7. V-2 — FECHADO v1.15 (4/4 turnos)

**Dossiê `docs/V-2_SCOPE.md:14` travado:**

| Item | Valor |
|:---|:---|
| **In-band** | `0x50 RAG_INDEX_ADD` `0x51 RAG_INDEX_DEL` `0x52 RAG_SEARCH` `0x53 EMBED_LOOKUP` `0x56 PQ_ENCODE` `0x57 PQ_DECODE` |
| **Out** | `0x54 HASH_BUCKET` `0x55 QUANTIZE_VEC` HELD `docs/ESPEC-V2.md:293` |
| **Bump** | `v1.14->v1.15` `docs/ESPEC.md:7` `docs/RFC-0038-RETRIEVAL.md:6` |
| **Bit** | `50 REQUIRED` `src/m3bc.rs:61` |
| **Goldens** | `V-2_SCOPE §5` (SEARCH bit-exato, PQ roundtrip, LOOKUP média, ADD/DEL lifecycle) |
| **Bench** | `benches/rag_bench.rs` (ADD 1×64, SEARCH 1k×64 TOPK10, LOOKUP 32, PQ 64d 8×16) |
| **Demo** | `programs/rag_demo.m3asm` 17 instr `ADD×3->SEARCH->LOOKUP->PQ->HALT` (`test_rag_demo_exec` verde) |

**5 decisões travadas `V-2_SCOPE §9` `src/rag.rs:10`:** `1024` / `f32` / `cosine` / `Arc<RwLock>` CoW / corpus `rag_demo`

**Slice 4 turnos — todos ✅:**
- **Turno1** `3986872` `ADD/DEL` + `bit50` + `test_rag_index_lifecycle`
- **Turno2** `2114213` `SEARCH`/`LOOKUP` + `test_rag_search_topk`
- **Turno3** `PQ_ENCODE/DECODE` + `test_rag_pq_encode_decode`
- **Turno4** `rag_bench.rs` + `rag_demo.m3asm` + bump `v1.15`

## 7b. V-3A — FECHADO v1.16 (mini, local shims)

**Dossiê `docs/RFC-0039-SYSTEM-OPS.md` (não V-3_SCOPE separado):**

| Item | Valor |
|:---|:---|
| **Shims** | `0x4A LOAD_MODEL` `0x4B SPAWN_CONTEXT` `0x4C KILL_CONTEXT` `0x4D SET_MODEL` (0x4E-0x4F RESERVED) |
| **64B** | `0xA0 LOAD_MODEL` `0xA2 SPAWN_CONTEXT` etc. seguem DRAFT `ESPEC-V2 §3.14` |
| **Bump** | `v1.15->v1.16` `docs/ESPEC.md:7` |
| **Bit** | `51 REQUIRED` `M3BC_REQUIRED_HAS_V3_SYSTEM=1<<51` `src/m3bc.rs:63` |
| **Demo** | `programs/system_demo.m3asm` 10 instr (`LOAD×2->SPAWN×2->SET->KILL->HALT`) + `programs/voice_loop_demo.m3asm` já verde (5 invariantes, `test_planoV3_voice_loop`) |
| **Bench** | sem bench novo (stub <1µs; `rag_bench` já cobre) |
| **Goldens** | `vm::test_system_ops_mini` (11 bad forms, `LOAD 1->1,2->2`, `SPAWN->child model`, `SET`, `KILL`, snapshot `models`, `.equ` em `MODEL`) |

Corte: PQ -> V-2b se esticar `V-2_SCOPE:11`.

## 8. Roadmap de fases

| Fase | Escopo | Status |
|:---|:---|:--:|
| V-1a | `.equ/.text` literais | ✅ `b4a950b` |
| V-1b d1 | `.data` escalar+string `AssembledProgram` | ✅ `8c4ad03` |
| V-1b d2 | Multi-value nested | ✅ `5d9fa62` |
| V-1b d3 | Loader `@name` + layout congelado | ✅ `ff21f53` |
| V-2 | Retrieval RAG/EMBED/PQ | ✅ v1.15 (4/4; 44/44) |
| V-3A | System local shims | ✅ v1.16 (0x4A-0x4D; 45/45; `system_demo`+`voice_loop` MVP 5 invariantes) |
| V-3 64B | System 64B X-forms | ⏸️ freeze dual-mode (W1 remainder) |
| V-4 | Pesos reais (fumaça Mamba) | ✅ V-4 smoke (130m 42s + TinyLlama 74s) fecha ESPEC OPEN |
| G7 | Safety doc (paralelo, sem código) | 🔜 próximo (bloqueante antes de demo pública) |

Milestone público: voice-loop end-to-end (V-3), não V-2. Container `.m3bc` com dados (bit49 wiring) é follow-up fora desta RFC `docs/RFC-0037-EQU-TEXT.md`.

## 9. Ecossistema (satélites)

| Projeto | Ling | Função | Estado | Local |
|:---|:---|:---|:---|:---|
| m3_avm | Rust | VM, ISA, scheduler | 379 passed, 116 ops, v1.16, V-4 verde | `~/projetos/m3_avm/` |
| vulkan_collisor_simulator | C++ | Kernels `.spv` + face | v0.1.0-mirror | `~/projetos/vulkan_collisor_simulator/` |
| m3-audio | Rust | I/O áudio real-time, AEC | Semana 1 | `~/projetos/m3-audio/` |
| m3-spec | Docs | `CONVENTIONS.md` `AUDIO_CONTRACT.md` `INTEGRATION.md` | Estável | `~/projetos/m3-spec/` |

Regra ouro: contrato via `m3-spec/`, zero código compartilhado, cada projeto headless.

## 10. Comandos úteis

```bash
cd ~/projetos/m3_avm

cargo test --lib                    # 379 passed; 1 failed moshi pré-existente (V-4 smokes 120s)
cargo test --lib -- test_rag        # 4 RAG (lifecycle+search+pq+demo)
cargo test --lib -- test_system     # V-3A shims
cargo test --lib -- test_v4 -- --nocapture # V-4 smokes (mamba 42s + tinyllama 74s)
cargo bench --bench rag_bench -- --quick  # 6 benches
cargo run --bin m3_avm -- run programs/rag_demo.m3asm --max-steps 50 # 17 instr
cargo run --bin m3_avm -- run programs/system_demo.m3asm --max-steps 50 # 10 instr
cargo run --bin m3_avm -- run programs/voice_loop_demo.m3asm --max-steps 4000 # 5 invariantes

cargo run --bin m3_avm -- run programs/control_flow_demo.m3asm --max-steps 10
cargo run --bin m3_avm -- asm programs/english_tutor_demo.m3asm --emit-asm
cat docs/ESPEC.md           # normativa v1.14
cat docs/ESPEC-V2.md        # DRAFT reconciliada
cat docs/V-2_SCOPE.md       # scope travado
cat docs/RFC-0038-RETRIEVAL.md
```

## 11. Documentos de referência

**Ler nesta ordem:**
1. `docs/PLANO_VISAO.md`
2. `docs/ESPEC.md` (normativa) + `docs/ESPEC-V2.md` (DRAFT)
3. `docs/RFC-0037-EQU-TEXT.md`
4. `docs/V-2_SCOPE.md`
5. `docs/RFC-0038-RETRIEVAL.md`
6. `src/rag.rs` + `src/m3bc.rs:49`

**Após implementar (turno4):** atualizar `docs/ESPEC.md:7` bump, `RFC-0038 changelog`, `README.md §3`, `docs/PLANO_VISAO.md §2`.

## 12. Fatos que a IA NÃO pode esquecer (atualizado)

1. Sem Python no runtime.
2. AGPL v3 — não sugerir MIT/Apache sem trade-off.
3. Opcodes imutáveis — novo = nova posição.
4. `LOADI 0x78` `u128` resolve `@name` — `LOADI64` nunca existiu; V-1b dia3 FEITO `ff21f53`.
5. 5 decisões V-2 TRAVADAS `V-2_SCOPE §9` — não reabrir.
6. V-2 não é milestone — voice-loop V-3 é.
7. Regs simbólicos só via `.reg` explícito — nunca auto-alocação.
8. `.data` é chave de seção.
9. Bit 50 REQUIRED — runtime antigo falha alto.
10. Queue-driven para event sources.
11. Não patentear — Zenodo+arXiv.
12. Todo commit `cargo test --lib` verde (exceto moshi pré-existente).

## 13. Convenções de código

Commits imperativo curto: `V-1b: add ...` `RFC-0037: clarify ...` `V-2: implement RAG_SEARCH`. Tags só após V-2 (Decisão 8). Cada feature -> teste. Docs atualizados no mesmo commit. Master estável.

## 14. Riscos críticos

| Risco | Mitigação |
|:---|:---|
| PQ estoura escopo | Corte para V-2b (registrado) |
| Wiring bit49 container | Follow-up separado; bit congelado `1<<49` já testado |
| MNN no collisor sem Vulkan | Fallback CPU |
| Autoria | ORCID+DOI Zenodo |
| Big Tech | AGPL força negociação |

## 15. Próxima ação imediata

**V-4 FECHADO — próximo G7 Safety (bloqueante antes de demo pública):**

1. G7 Safety doc (`docs/SAFETY.md` novo): política de crise, refusals, conteúdo sensível, *filler* seguro (cf. `docs/FILLER_SPEECH.md` + `docs/AVATAR_3D.md` Abordagem A)
2. Publicação defensiva: Zenodo `v1.16` + arXiv delta (sem pesos; DOI já 10.5281/zenodo.22694886)
3. Depois: transporte F2 (driver `m3-spec` + `L1-0` swarm) ou avatar sidecar `vulkan_collisor` (sem opcode)

**Gate V-4 já verde:** `cargo test --lib` 379 + `test_v4_*` 120s + corpus 45 + `rag_bench` + `REQUIRED bits 50|51` firewall.

---
*HEAD + V-4 v1.16 fechado — 2026-09-11.*
*Author: Matheus de Camargo Marques — ORCID 0009-0003-4518-2258 — AGPL-3.0-or-later*
