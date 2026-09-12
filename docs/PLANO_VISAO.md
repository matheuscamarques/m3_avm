# Plano-Objetivo: da Base v1.16 à Família de Visão

```text
Status:   PLANO vigente (roadmap, NÃO-normativo; muda por decisão registrada)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Family:   docs/VISAO_PREMIUM_250GB.md · docs/EXPERIENCIA_PREMIUM.md
          docs/PAUSA_DE_PENSAMENTO.md · docs/FILLER_SPEECH.md
          docs/MVP_ENGLISH_TUTOR.md · docs/AVATAR_3D.md
License:  AGPL-3.0-or-later (see LICENSE)
```

> **Contrato de honestidade (ESPEC §1.3):** este plano descreve metas e
> ordem de trabalho, não capacidade atual. Todo status ✅/🟡/🔴 aqui
> reflete a árvore na data acima; divergências futuras decidem-se por
> nova revisão deste arquivo, nunca por edição silenciosa.

## 1. Objetivo — o que "chegar lá" significa, por documento

| Documento | Chegar lá = |
|:---|:---|
| `VISAO_PREMIUM_250GB` | Marcos 1–5 operando (voz → RAG → deep path → barge-in 3 modelos → LoRA), medidos, não afirmados |
| `EXPERIENCIA_PREMIUM` | As 8 sensações reproduzíveis em demo (contato, casual, média, profunda, interrupção, troca de assunto, longa, ruído) |
| `PAUSA_DE_PENSAMENTO` | Turn-taking com os 4 comportamentos (pausa, retomada, reformulação, abandono) + antecipação |
| `FILLER_SPEECH` | Fala de preenchimento no deep path + biblioteca EN de 30 frases + duração adaptativa |
| `MVP_ENGLISH_TUTOR` | Tutor funcional **no emulador** (rig físico fora do horizonte atual — ver §6) |
| `AVATAR_3D` | v1 com **zero opcodes novos** (sidecars host-side), após MVP de voz |

## 2. Ponto de partida (verificado 2026-09-11 — pós V-4)

- ISA v1.16, 116 opcodes, suite 379 passed + 1 falha pré-existente
  (`moshi::test_gguf_qkv_split_shapes`, model-data, sem relação)
- Fases 0–5 + V-1 (assembler) + V-2 (retrieval) + V-3A (system) + V-4 (Mamba smoke) completas,
  Trilha P (10/10 sem opcode), CALL/RET, TopK-Lean provado (`lake build` verde)
- Zenodo DOI ativo; container `.m3bc` fim-a-fim com `REQUIRED bits 50|51`
- ESPEC §16 quase todo IMPL (retrieval+system+smoke IMPL, `benches/rag_bench.rs`, `programs/rag_demo`+`system_demo`); restam: transporte, ondas rejeitadas, GPU parcial

## 3. Requisitos — inventário consolidado (o que falta)

| # | Gap | Tamanho | Destrava |
|:---|:---|:---|:---|
| G1 | Demos executáveis de comportamento (pausa + filler sintético) | **Nenhum código novo** — só `.m3asm` + testes | `PAUSA` MVP, `FILLER` Marco 0, `EXPERIENCIA` (interrupção, pergunta profunda) |
| G2 | Assembler track: `.data/.equ/.str/.text`, literais, init multivalor (+ `STORE` só se passar no filtro R12) — V-1a (`.equ`/`.text`, RFC-0037) feito; V-1b (`.data`/`.str`/init) pendente | 1–2 turnos (só assembler + testes; zero opcode salvo R12) | Tabelas (FOREST/XGB/filler) e todos os programas da família — **hoje NENHUM programa de visão monta** |
| G3 | Fase 6 Retrieval (`0x50–0x53`, `0x56–0x57`; `0x54–0x55` já julgados sem-opcode) | **FEITO** v1.15 (RFC-0038 turnos 1-4; `rag_demo` + `rag_bench`; `bit 50 REQUIRED`) | Tutor RAG, antecipação da pausa, `RAG_SEARCH` dos programas |
| G4 | System ops (`LOAD_MODEL`, `SPAWN_CONTEXT`, `MODEL_SWITCH`…, zona `0xA0+`) | **FEITO** V-3A mini 0x4A-0x4D 32B (`bit 51`; `0xA0+` 64B fica DRAFT) | Programas multi-modelo que carregam/posicionam de verdade |
| G5 | Fumaça de pesos reais (Mamba GGUF smoke = OPEN; mapeamento GGUF existe) | Turno de integração + pesos externos | Sair do sintético sem comprar rig (TinyLlama/DeepSeek locais) |
| G6 | Integração voice-loop MVP (SENSE→VAD→CODEC→raciocínio stub→DEPFORMER→CODEC→STREAM, pausa+filler vivos) | 1–2 turnos (programa + cola, quase zero opcode) | Marco 1 adaptado ao emulador; `moshi_loop` de verdade |
| G7 | Safety policy (respostas em crise, cf. doc FILLER) | Doc próprio, sem código | Pré-requisito ético antes de qualquer demo pública |
| G8 | Fase 8 T2 + follow-ups cirúrgicos (blocking LOCK/EDF, dedup FORK/SNAPSHOT, guarda PERSISTENT nos produtores antigos, encoder Q4_K com vetores) | Fatiado, sob demanda | Robustez — **adiado** (só emulador; sem hardening de tempo real) |
| G9 | Fase 7 transporte (F2–F5) | **Adiada** (decisão single-rig, §6) | Nada no horizonte atual |
| G10 | Lean restante (stepError físico, decoder dual-mode, WAL dual-naming, Region Table, X-form) | Sob demanda, por obrigação | Paper (TopK, a que importava p/ T4, provada) |

Fora de escopo permanente: programa `.m3asm` completo da família antes de G2 · pesos no repo (sempre externos, git-ignored) · voz PT-BR (política EN-only até fine-tuning) · `v2.x` (só no freeze dual-mode) · transporte multi-nó (decisão §6).

## 4. Ordem de execução

### Fase V-0 — Quick wins, zero código em `src/` ← *primeira*
- Demo pausa: `VAD_DETECT` + `CONCAT` + `CYCLES_COUNT` + `COMPARE` (tudo IMPL) — máquina INCOMPLETE/THINKING/RESUMED + timeout 3s, com asserts.
- Demo filler Marco 0: crossfade `FILL`+`MUL`+`ADD` sobre PCM sintético + loop adaptativo com `RNG` — transição e duração sem modelo.
- **Critério de saída:** 2 arquivos em `programs/` + testes verdes; `git diff --stat src/` vazio.

### Fase V-1 — Assembler track
- `.data/.equ/.str/.text` + literais + init multivalor (+ `STORE` só com prova R12).
- **Critério:** tabelas FOREST/XGB/filler montáveis a partir de texto; corpus gate continua verde; nenhum programa de visão precisa montar ainda.

### Fase V-2 — Fase 6 Retrieval
- `RAG_INDEX_ADD/DEL/SEARCH`, `EMBED_LOOKUP`, `PQ_ENCODE/DECODE` + bump minor + demo RAG mínima (índice sintético primeiro).
- **Critério:** busca real sobre índice construído no repo; `RAG_SEARCH` dos programas deixa de ser ficção.

### Fase V-3 — System ops + voice-loop MVP (G4+G6 juntas) — ✅ V-3A mini (0x4A-0x4D 32B, bit 51; 0xA0+ 64B fica DRAFT)
- `LOAD_MODEL`/`SPAWN_CONTEXT`/`KILL_CONTEXT`/`SET_MODEL` locais → `system_demo.m3asm` + `voice_loop_demo` (MVP já verde; `VAD+CODEC+DEPFORMER`).
- **Critério:** conversa sintética completa no emulador; latências **medidas**, nunca afirmadas. V-3A cumpre: `SPAWN`/`LOAD` nomeiam modelos/contextos a partir de `.m3asm`.

### Fase V-4 — Pesos reais sem rig (G5)
- Fumaça Mamba GGUF (fecha o OPEN), validação do mapeamento com TinyLlama/DeepSeek locais, ingestão de corpus RAG mínimo em JSONL.
- **Critério:** mesmo loop V-3, agora com pesos reais onde couber na máquina atual.

### Contínuo / paralelo
- G7 safety **antes** de qualquer demo pública (bloqueante, não opcional).
- G8/G10 sob demanda, um item por vez, com RFC própria quando tocar encoding.
- Zenodo por release; papel só com medidas reais (nunca sintéticas).
- Revisar este plano a cada fase concluída (atualizar §2 e o quadro abaixo).

### Quadro de acompanhamento

| Fase | Status | Evidência de saída |
|:---|:---|:---|
| V-0 quick wins | ✅ feito (RFC-0035) | 2 demos + testes; `src/` intocado |
| V-1 assembler | ✅ feito (RFC-0036 `.reg`, RFC-0037 V-1a `.equ`/`.text` + V-1b dias 1–3: `.data`/sidecar/`@`/loader) | consts + blobs + `@` montáveis; corpus verde; container `.m3bc` com dados = follow-up (bit 49 congelado) |
| V-2 retrieval | ✅ feito (RFC-0038 v1.15) | `RAG_INDEX_ADD/DEL` + `SEARCH`/`LOOKUP` + `PQ_ENCODE/DECODE` + `rag_demo` + `rag_bench` + `bit 50` + corpus 44 |
| V-3 system | ✅ feito (RFC-0039 V-3A mini) | `LOAD_MODEL`/`SPAWN/KILL`/`SET_MODEL` 0x4A-0x4D + `system_demo` + `voice_loop` MVP (5 invariantes); `bit 51` + corpus 45 |
| V-4 pesos reais | ✅ feito (V-4 smoke) | `test_v4_mamba_forward_real_smoke` (130m 42s) + `test_v4_transformer_forward_real_smoke` (TinyLlama 74s) — fecha OPEN ESPEC §16 |
| G7 safety | ⬜ pendente | doc próprio antes de demo pública |

## 5. Riscos assumidos (declarados)

1. V-0 prova comportamento, não inteligência — pausas e fillers convincentes sobre raciocínio stub ainda são stub (o doc EXPERIÊNCIA já diz isso).
2. Pesos reais (V-4) podem revelar que tensores/ops assumem formas que os modelos não têm — retrabalho possível em V-2/V-3; por isso V-4 vem depois, não antes.
3. Sem rig, nenhuma latência abaixo de ~ms é alegável — todas as metas de µs da família seguem METAs até medição em hardware.
4. Programas de visão completos continuam não-montáveis até V-1 (+ system ops); qualquer demo parcial deve dizer o que falta, como fazem os docs da família.

## 6. Registro de decisões (vigente; muda só com nova entrada aqui)

| Data | Decisão |
|:---|:---|
| 2026-09-11 | Fio inicial: quick wins (V-0) antes de features |
| 2026-09-11 | Só emulador por ora; sem hardening de tempo real; sem compra de rig |
| 2026-09-11 | Single-rig; Fase 7 adiada por tempo indeterminado |
| 2026-09-11 | Voz PT-BR adiada (política EN-only até fine-tuning) |
| 2026-09-11 | `v2.x` reservado ao freeze dual-mode (regra RFC-0028) |
| 2026-09-11 | HELD sem opcode salvo prova R12 (regra Trilha P) |
| 2026-09-11 | Fronteira satélites: sem WGSL/shader novo no repo (wgpu ATTN≤64 existente congelado; kernels futuros via `.spv` do collisor); sem I/O de áudio (sem `cpal`/ALSA — WAV/arq + `SENSE` sintético; `m3-audio` via ring/file); headless por design; convenções só via `m3-spec/` (nada unilateral) |

---

## Satélites

Este projeto tem dois satélites:

- **vulkan_collisor_simulator** (C++, Linux-only) — kernel factory `.spv` + vetorização de rosto
- **m3-audio** (Rust) — I/O de áudio real-time, AEC, hotplug

Contrato compartilhado: `../m3-spec/`. Zero código compartilhado entre projetos.

Convergência no mês 6+ (avatar 3D + áudio real no MVP de voz).

---

*Plano vivo: atualizar §2, §4 e §6 a cada fase concluída. Nada aqui
altera ESPEC.md (normativo) — cada fase com encoding gera sua RFC.*

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
