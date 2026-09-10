# Dossiê de Patente M³-AVM — Índice e Aviso

> **Status:** minuta técnica para revisão por agente de PI. **Não é depósito.**
> **Autor/inventor:** Matheus de Camargo Marques — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258) · matheuscamarques@gmail.com · https://github.com/matheuscamarques/m3_avm
> **Data-base:** 2026-09-10 · **ISA referência:** v1.4 (`docs/ESPEC.md`) + proposta v2.0 DRAFT (`docs/ESPEC-V2.md`)
> **Licença do código:** AGPL-3.0-or-later (ver `LICENSE`, `CITATION.cff`)

## Estratégia adotada (resposta ao dilema IMPL vs DRAFT)

Seu rascunho original reivindicava como pronto o que no código é `RSVD/DRAFT`
(`FOREST`, `SPIKE_STEP`, `KV_TRANSFER`, deadlines `AND/OR/N_OF_M/CHAIN`,
scheduler EDF+NUMA/NVLink, instrução 64B, 11 regiões, ChaCha20-Poly1305, OTLP).
Depositar assim gera **insuficiência descritiva (art. 24 LPI)** e risco de nulidade.

A pedido, o dossiê foi reescrito em **2 camadas explícitas**:

- **Camada 1 — Núcleo habilitado (IMPL):** o que o código + testes + Lean provam hoje.
  É o que sustenta suficiência descritiva e melhor modo.
- **Camada 2 — Modalidade preferencial futura (DRAFT/RSVD):** extensões com
  codificação congelada em `docs/ESPEC.md §§12–13` e `docs/ESPEC-V2.md §3`,
  descritas como desenvolvimento previsto, com regra de habilitação
  (ex.: taxonomia stateful §6.4, payloads de 26B, janela `k=16`).

Nada neste dossiê afirma medição que não exista em `docs/ESPEC.md §17`.

## Arquivos

| # | Arquivo | Conteúdo | Leitor |
|---|---------|----------|--------|
| 0 | `00_LEIAME.md` (este) | índice + checklist | todos |
| 1 | `01_RELATORIO_DESCRITIVO.md` | relatório completo revisado (arts. 24 LPI) | agente PI + INPI |
| 2 | `02_QUADRO_REIVINDICATORIO.md` | 15 reivindicações em 2 camadas | agente PI + INPI |
| 3 | `03_RESUMO_INPI.md` | resumo ≤150 palavras (Ato Normativo INPI) | INPI |
| 4 | `04_DESENHOS.md` | Fig. 1–5 em texto + Mermaid p/ desenhista | desenhista técnico |
| 5 | `05_EVIDENCIAS_IMPLEMENTACAO.md` | tabela reivindicação × código × teste × prova Lean | agente PI + perito |

Documentos-fonte normativos (prevalecem sobre este dossiê em caso de conflito):
`docs/ESPEC.md` (normativo), `docs/ESPEC-V2.md` (DRAFT), `src/opcodes.rs`,
`src/memory.rs`, `src/context.rs`, `src/vm.rs`, `formal/Formal/*.lean`.

## Mapa rápido IMPL vs DRAFT (não esconder do agente de PI)

| Tema do rascunho original | Realidade auditada 2026-09-10 | Tratamento no dossiê |
|---|---|---|
| Instrução 32B ou 64B | **32B IMPL** (`INSTR_SIZE=32`, `src/opcodes.rs:26`); 64B só DRAFT (`ESPEC-V2 §4`, `INSTR_SIZE_64` const mas sem fetch 64B no `Vm`) | Reivindica 32B; 64B como modalidade futura |
| 11 regiões (TEXT/WEIGHTS/ARENA/SHARED/WAL/...) | **4 IMPL:** GLOBAL 0x00, TEMPORAL 0x10, PERSISTENTE 0x20, KV_CACHE 0x30 (`src/memory.rs:24-27`); 16 lógicas só DRAFT (`ESPEC-V2 §7`) | Descreve 4 + tabela de compatibilidade com 16 futuras |
| `ATTN/FFN/ROPE/SSM_SCAN/CODEC_*` | **IMPL** (`0x01–0x19`, `src/vm.rs`, `src/ssm.rs`, `src/mimi.rs`) | Camada 1 |
| `GATHER/DISTANCE/RANK1_UPDATE` + `FOREST/DENOISE/ODE/SPIKE_STEP` | **IMPL** via RFC-0004/0012–0015 (`0x1F/0x23/0x24`, `0x22/0x21/0x25/0x20`) | Camada 1 (com ressalva RFC-0012: sem demo `.m3asm` de FOREST) |
| `CONV/RAG_SEARCH/DEPFORMER/STREAM_MERGE/KV_TRANSFER` | **RSVD/DRAFT** (rejeitam com erro explícito) | Camada 2, com payload congelado citado |
| `FORK/ABORT` rollback conjunto | **IMPL parcial:** CoW `Arc::clone` + `ssm_states` push/pop + `rank1_layers` push/pop + `KV_TRUNCATE 0x38`; I-Mono IMPL, **I-Persist OPEN** (`ESPEC §6.3`) | Reivindica CoW + I-Mono; I-Persist como obrigação aberta declarada |
| Deadlines `AND/OR/N_OF_M/CHAIN` + EDF + NUMA/NVLink + herança cross-device | **Não implementado.** Existe `SET_DEADLINE 0x71/GET_DEADLINE 0x72` + prioridade estrita `Red>Blue>Green` (`src/context.rs:15`), sem EDF, sem NUMA, sem NVLink | Camada 1 = prioridade estrita + deadline absoluto; Camada 2 = composição + EDF + afinidade |
| Cluster WAL + ChaCha + OTLP | **RSVD.** `0x1A–0x1D` congelados mas sem transporte; sem cripto; `TRACE_EVENT 0x6B` local apenas | Camada 2 |
| Latências 5–50ms, 217µs, 500ms | Medido: `FORK <1ms`, `ABORT ~25ms debug / publish <1ms`, `ATTN 32×32 ~5ms`, `TinyLlama 0.33s/tok` (`ESPEC §17`). Restante é alvo | Só números medidos no relatório; resto marcado `alvo` |

## Checklist antes de procurar agente de PI

- [ ] Decidir titularidade (pessoa física vs empresa) e coautoria.
- [ ] Decidir: pedido principal restrito (Camada 1) + divisionais (Camada 2) — recomendado na seção 8 do relatório.
- [ ] Gerar `git log --oneline` + `cargo test --lib` + `lake build` datados como anexo de prova de posse.
- [ ] Publicar RFCs no Zenodo com DOI **ou** manter sigilo — não fazer os dois ao mesmo tempo sem orientação.
- [ ] Encomendar Fig. 1–5 a desenhista a partir de `04_DESENHOS.md` (traço técnico, sem texto excedente).
- [ ] Revisar `02_QUADRO_REIVINDICATORIO.md` com agente: clareza (art. 25 LPI), unidade de invenção, suporte no relatório.
- [ ] Contar palavras de `03_RESUMO_INPI.md` (≤150) na versão final protocolada.

## Comandos de prova (rode e anexe saída)

```bash
cargo test --lib 2>&1 | tail -5
cargo run --release -- bench --nops 1000000 2>&1 | tail -5
# se tiver Lean:
lake build 2>&1 | tail -5
git log --oneline -10
git status --short
```
