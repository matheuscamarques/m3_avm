# ANÁLISE DO RELATÓRIO DE PATENTEABILIDADE (10 invenções) — auditoria contra o código

**Data:** 2026-09-10 · **Auditor:** assistente técnico (não parecer legal).
**Base:** `src/opcodes.rs` (57 `OP_*`), `src/vm.rs` dispatch, `docs/ESPEC.md` v1.4, `docs/patente/07_ANTERIORIDADE.md`.
**Veredito geral:** o relatório acerta na direção (combinação heterogênea + ISA é o ativo), mas **superestima enablement em 6/10 invenções** e **subestima 2 riscos letais** (cross-model KV ago/2026; software puro em EPO/BR). Usável como tese, **não como depósito sem os cortes abaixo**.

## Método (o que conferi)

- `OP_*` existentes: `0x00–0x19`, `0x1F/0x23/0x24`, `0x38`, `0x21 DENOISE`, `0x22 FOREST` (RFC-0012, novo), `0x25 ODE`, `0x60–0x66`, `0x6A–0x77`, `0x78/0x79`, `0xFF`. Total ~50 opcodes úteis.
- **Inexistentes como opcode:** `SNAPSHOT/RESTORE` (só `FORK 0x04/ABORT 0x05`), `WEIGHTS_HINT`, `COMPRESS/DECOMPRESS/COMPRESS_QUERY/ENTROPY_ESTIMATE`, `DEPFORMER/STREAM_MERGE`, `RAG_INDEX_ADD/RAG_SEARCH/EMBED_LOOKUP/PQ_*`, `FOREST_EVAL/DISTANCE_CALC` (existe `DISTANCE 0x23` + `FOREST 0x22` — nomes do relatório estão errados), `WAL_APPEND/SEND_TENSOR/MIGRATE` (só consts/RSVD), `ESCAPE` com execução (só validação `ext_len` + `instr_width`), `Region Table/Register File Header` operacionais, tiers NVMe/remoto/S3, SNN `SPIKE_STEP` (`0x20` RSVD).
- Testes: `cargo test --lib` 228 passed / 1 failed (`moshi qkv`, dado do modelo).

## Casos de anterioridade do relatório (validação)

| Caso | Avaliação |
|---|---|
| **Caso 1 — US 8,161,269 B2 (Intel escape x86, 2012)** | Citação pertinente, diferenciação correta (hardware decode vs bytecode VM + `ext_len` + negociação + tabela). **Porém:** nossa negociação (`REQUIRED/OPTIONAL_FEATURES`, `.m3bc`, tabela em MMIO+0x0) é **DRAFT não executado** (`ESPEC-V2 §§7–9`, loader só fareja MAGIC). Não reivindicar como pronto; risco médio **procede**, mas por motivo maior: falta enablement, não só Intel/Sun. |
| **Caso 2 — PagedAttention/vLLM (SOSP'23)** | Correto e é a anterioridade mais perigosa para CoW. Diferenciação “heterogêneo + O(1) + 4 tiers” **parcialmente falsa**: O(1) por troca de ponteiros existe (C1), mas **4 tiers não existem** (só VRAM-simulada/CPU; sem NVMe/remoto, sem decisão `T_transf vs T_recomp`). Cortar tiers da claim; manter transversal heterogêneo. |
| **Caso 3 — PyramidInfer (arXiv:2405.12532)** | Citação boa, mas **omite as 3 anteriores letais de ago/set-2026**: 2608.03893 (ridge cross-model), 2609.00891 (CacheBridge), 2608.30963 (cross-family) — ver `07_ANTERIORIDADE.md §3`. Para Invenção 5/claim 10, PyramidInfer é o menor dos problemas. |

## Invenção por invenção (risco do relatório → risco corrigido)

**1. ISA ESCAPE + versionamento + indireção — relatório: médio.**
Corrigido: **médio-alto (por falta de enablement)**. `ext_len` só validado, `0xB0–0xB6` sem execução, `.m3bc`/tabela sem loader. **Corte:** reivindicar só `instr_width` + rejeição `UnsupportedWidth` + `0xFF=32B` (RFC-0002, IMPL) como C1; resto vira divisional futura. Nomes `Register File Header` não existem no código — remover ou definir.

**2. CoW determinístico + SNAPSHOT/RESTORE O(1) — relatório: baixo-médio.**
Corrigido: **médio (núcleo ok, nomes errados)**. Opcodes chamam-se `FORK/ABORT` (+`KV_TRUNCATE`), não `SNAPSHOT/RESTORE`; `version: u64` + I-Mono ok; **I-Persist aberto**; SNN `V(t)` não existe (só SSM `h_t` + RANK1 `H` + KV). **Corte:** trocar nomes, listar só estados existentes, declarar I-Persist como obrigação. Com isso, risco cai para baixo-médio e é a melhor divisional após multi-modelo.

**3. Scheduler EDF + deadline composto + cross-device — relatório: médio.**
Corrigido: **médio-alto (C2 puro)**. Existe só `SET/GET_DEADLINE` + `Red>Blue>Green` estrito; sem `AND/OR/N_OF_M/CHAIN` codificados, sem EDF, sem NVLink/NUMA medidos, sem failover <100ms (número sem fonte). HetSched/SMLP citados ok, mas faltam Seldon/KServe-timeout-fragmentado e K8s-1.36-topológico e SRP/DFP (ver `07 §5`). **Corte:** C1 = deadline absoluto + prioridade estrita; C2 = composição/herança como divisional com RFC obrigatório.

**4. Cluster WAL + Lamport + `t_wal_retention` — relatório: baixo.**
Corrigido: **alto (sem enablement)**. `0x1A–0x1D` RSVD sem transporte; sem `WAL_APPEND`, sem retenção implementada, sem ACK. “Sem precedentes” é falso como afirmação de execução — é precedente como **ideia** (WAL é clássico; novidade estaria no protocolo de tensores, ainda não escrito). **Corte:** virar divisional F0→F5 sem pular F2; nunca no principal.

**5. KV 4-tiers + evicção Q·K + recomputação — relatório: médio-alto.**
Corrigido: **alto (tiers + decisão inexistentes)**. Existe `KV_TRUNCATE` + `DISTANCE→GATHER→ATTN` podada (T4 perna empírica). Não existe tier NVMe/remoto, `T_transf vs T_recomp`, máscaras dinâmicas, FlexGen-offload próprio. **Corte:** reivindicar só higiene + poda (C1); tiers/decisão vão para divisional com benches por tier.

**6. Multi-modelo contexto único + rollback conjunto — relatório: muito baixo / prioridade extremamente alta.**
Corrigido: **baixo-moderado — concordo que é a claim 1 do principal.** É o único ponto com enablement (Transformer+SSM+codec+`GATHER/DISTANCE/RANK1/FOREST/DENOISE/ODE` no mesmo endereço, `FORK/ABORT` transversal) e sem antecipação direta (Triton/Ray são multi-processo). **Ajustes:** remover “SNN” da lista executada (RSVD) ou marcar C2; remover “depuração distribuída no bytecode” (só `TRACE_EVENT` local); corrigir “elimina IPC” para “elimina para os motores residentes” (cluster ainda precisaria rede).

**7. Prefetch 5 níveis + WEIGHTS_HINT — relatório: médio.**
Corrigido: **alto (opcode não existe)**. `mmap` GGUF + `madvise` + `matvec_q4k` existem; `WEIGHTS_HINT`, 5 níveis, RDMA/S3, descompressão inline não. **Corte:** fora do principal; no máximo dependente “GGUF mmap + KV por camada” (C1) + divisional futura para hint/prefetch.

**8. DEPFORMER/STREAM_MERGE full-duplex — relatório: baixo.**
Corrigido: **alto (opcodes DRAFT)**. Existe `CODEC_ENC/DEC/AUDIO_ALIGN` + `ssm/moshi` parcial + demos; `0x44/0x45` rejeitam. “17 fluxos, atraso acústico na ISA, barge-in atômico” mistura C1 (`SENSE→IF_INTERRUPT→ABORT`) com C2. **Corte:** C1 = codec+align+barge-in local; DEPFORMER/MERGE = divisional W8.

**9. RAG + ML clássico nativos — relatório: baixo.**
Corrigido: **médio, melhorou esta semana: `FOREST 0x22` e `DISTANCE 0x23` IMPL (RFC-0004/0012) — atualizar dossiê.** Mas `RAG_INDEX_ADD/SEARCH`, `EMBED_LOOKUP`, `PQ_*`, `SVM/KMEANS` seguem RSVD/HELD; região `RAG_INDEX` só lógica; mesmo motor EDF é C2. **Corte:** C1 = `GATHER/DISTANCE/RANK1/FOREST` + `SAMPLE TOPK`; resto divisional W7 com filtro lowering-inadequado.

**10. Compressão adaptativa (COMPRESS/ENTROPY) — relatório: baixo-médio.**
Corrigido: **alto (nada existe)**. Sem opcode, sem estimador de entropia, sem `SEND_TENSOR/MIGRATE` executando, sem nvcomp integrado. Q4_K é quantização de pesos, não compressão de transporte adaptativa. **Corte:** fora do principal; divisional futura ou abandono. Menor prioridade real do portfólio.

## Matriz corrigida (prioridade = valor × enablement / risco)

| # | Relatório (risco/prior.) | Corrigido | Destino |
|---|---|---|---|
| 6 multi-modelo | muito baixo / extrema | baixo-moderado | **Principal claim 1** |
| 2 CoW | baixo-médio / alta | médio→baixo-médio após renomear | Principal claims 2–5 / **Divisional 1ª** |
| 1 ESCAPE | médio / alta | médio-alto | Principal só `instr_width`; resto divisional |
| 3 EDF | médio / alta | médio-alto | C1 parcial; resto divisional |
| 4 WAL | baixo / alta | **alto** | Divisional (não no principal) |
| 8 áudio | baixo / alta | **alto** | C1 parcial; resto divisional W8 |
| 9 RAG/FOREST | baixo / média | médio (FOREST ok) | C1 parcial; resto divisional W7 |
| 5 KV tiers | médio-alto / média | **alto** | C1 parcial; resto divisional |
| 7 prefetch | médio / média | **alto** | Fora do principal |
| 10 compressão | baixo-médio / média | **alto** | Fora/divisional tardia ou abandono |

## Estratégia do relatório (guarda-chuva + 5 divisionais + PCT + AGPL) — ajustes

1. **Guarda-chuva com 10 invenções “prontas” não passa em suficiência (art. 24) nem unidade.** Reduzir o principal a C1 habilitado: multi-modelo (6) + CoW renomeado (2) + ESCAPE-width (1 parcial) + deadline absoluto/prioridade (3 parcial) + `GATHER/DISTANCE/RANK1/FOREST/KV_TRUNCATE/RNG` (9/5 parciais) + codec local (8 parcial). Todo o resto em divisionais condicionadas a RFC+testes+Lean (`ESPEC-V2 §14`).
2. **PCT:** correto (BR → 12m PCT → 30m fases US/EP/BR/CN/JP). Adendo BR: sem “provisional” brasileiro — o “BR-relâmpago” é pedido completo simples que gera prioridade. E: publicar Zenodo **antes** do BR mata novidade onde não há graça (UE/CN) — os 3 papers de ago/2026 já mostram a velocidade do campo.
3. **Defensiva + AGPL:** tese correta (DOI trava concorrente + FTO; AGPL §13 força recontribuição; dual-licensing como receita). Mas **defensiva e patente são excludentes para a mesma matéria no exterior** — escolher por família: patentear 6/2/1/3, publicar defensivamente o resto. Não fazer os dois para tudo.
4. **Buscas:** protocolo INPI/USPTO/Espacenet/Google/arXiv está bom; acrescentar CPCs `G06N3/0455?/G06F9/50` (scheduling), `G06F12/10` (paging), `H04L67` (streaming) e busca obrigatória em arXiv 2026 (o relatório parou em 2024 e perdeu os 3 papers críticos).
5. **Valoração:** concordo com o top-3 (6, 1, 2), mas reordenado por enablement: **6 > 2 > 9-parcial > 1-parcial > 3-parcial**; 4/5/7/8-full/10 como apostas divisionais.

## Ações aplicadas neste turno

- `02_QUADRO_REIVINDICATORIO.md` claim 10 já estreitada (mecanismo, não matemática).
- `01_RELATORIO_DESCRITIVO.md §2.1` já cita vLLM/Seldon/KServe/K8s-topológico/SRP-DFP/2608.03893.
- `07_ANTERIORIDADE.md §§3/5` já cobre cross-model KV e claims 8–9.
- **Pendente (para o agente):** renomear `SNAPSHOT/RESTORE→FORK/ABORT` e `FOREST_EVAL→FOREST` no relatório externo; remover tiers/SNN/WEIGHTS_HINT/COMPRESS do principal; atualizar `05_EVIDENCIAS` com `FOREST/DENOISE/ODE` IMPL.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
