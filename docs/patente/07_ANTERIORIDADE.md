# BUSCA DE ANTERIORIDADE — minuta técnica (assistente, não parecer legal)

**Data:** 2026-09-10 · **Base do pedido:** ISA v1.4 + dossiê C1/C2.
**Uso:** levar ao agente de PI para ajuste das claims 1, 5, 8–10. Não é busca exaustiva INPI.

## 1. vLLM / PagedAttention — Kwon et al., SOSP'23 (arXiv:2309.06180)

**O que antecipa:** KV cache paginado (blocos ≈ páginas, tabela de blocos, alocação sob demanda), **CoW com refcount** para sharing entre sequências (parallel sampling/beam search, −55% memória), scheduler central + preempção.
**Diferença da nossa C1:** vLLM é **mesmo modelo / mesma arquitetura**, foco em throughput por batching. Não tem: VM de ISA fixa com motores heterogêneos (Transformer+SSM+codec) no mesmo endereço, `FORK/ABORT` transversal a KV+`h_t`+`H` com contador monotônico e janela, deadline absoluto + `Red>Blue>Green`, RNG por contexto, `GATHER/DISTANCE/RANK1` nativos.
**Impacto:** cita como anterioridade mais próxima para rollback/CoW. **Claims 1–2 e 5 devem se ancorar no caráter transversal heterogêneo + ISA + deadline**, não em “CoW de KV” isolado. Incluir vLLM no relatório §2.1 (já citado genericamente — explicitar).

## 2. NVIDIA Triton Inference Server (docs oficiais, arquitetura/execução)

**O que antecipa:** serving multi-modelo concorrente, scheduler **por modelo** (dynamic/sequence batching), `instance_group`, stateful via `correlation ID` + sinais start/end, ensembles/BLS, protocolos HTTP/gRPC/C API, métricas.
**Diferença da nossa C1:** Triton = **backends/processos separados + serialização por request**, sem espaço de endereço único com regiões tipadas, sem `CTX_SWITCH` sem troca de contexto, sem restore atômico transversal bit-exato, sem deadline composto. C API in-process existe, mas é linkage, não ISA de tensores com snapshot.
**Impacto:** sustenta o problema técnico (§2.2 itens 1–4). Citar Triton nominalmente no relatório fortalece atividade inventiva da C1 (eliminar serialização + rollback conjunto).

## 3. Cross-model KV transfer — Heo et al., arXiv:2608.03893 (04-ago-2026) + seguidores

**O que antecipa (risco alto para claim 10 como redigida):**
- 2608.03893: mapper ridge **fechado, por cabeça**, top-k camadas fonte, **RoPE-strip**, calibração **500×1024** (FineWeb-Edu), 73–98% retenção em 4/6 pares, 2,7–25× vs re-prefill, bidirecional.
- 2609.00891 CacheBridge (01-set-2026): mesma interface afim, restrição por cabeça casada, sensibilidade atencional, kernel fundido, −8× storage, −10× dados calibração.
- 2608.30963 Universal Context-Reuse (31-ago-2026): cross-family (Qwen↔Gemma, Llama-70B→Qwen-7B, −67% prefill, 6,5× mais rápido), “context mobility” como abstração.
**Datas:** todos **anteriores a 2026-09-10** → anterioridade potencial contra matéria C2 não depositada.
**Diferença aproveitável:** nenhum propõe **opcode de runtime com integração a snapshot/rollback** (descarte em `ABORT`), CoW de `H/h_t`, nem composição com deadlines/scheduler. A matemática do ridge por cabeça está comprometida; o **mecanismo de execução** ainda é espaço livre.
**Ação recomendada ao agente:**
1. **Estreitar claim 10** para “instrução nativa que aplica mapeamento a KV/estado residente + integra a `FORK/ABORT` (descarte até commit) + alimenta `ATTN` podada via `DISTANCE`”, sem reivindicar o ridge em si.
2. Citar 2608.03893/2609.00891/2608.30963 no relatório como estado da técnica (boa-fé + delimita invento).
3. Se a graça BR for usada pós-Zenodo, lembrar que esses papers já são técnica anterior **independente** — a graça não os neutraliza.

## 4. Conclusão operacional

- **C1 (claims 1–7, 11):** anterioridade mapeada é contornável por heterogeneidade transversal + ISA + deadline. Risco moderado, defensável.
- **C2 claims 8–9 (deadlines compostos/EDF/NUMA):** ver §5 — contornáveis por composição nativa + integração ISA, mas estreitar redação.
- **C2 claim 10 (KV_TRANSFER):** como estava, **risco alto de falta de novidade/atividade**. Estreitada em `02_QUADRO_REIVINDICATORIO.md` para mecanismo de runtime (opcode + descarte em `ABORT` + `DISTANCE→ATTN`), sem reivindicar a matemática.

## 5. Complemento claims 8–9 — timeouts por modelo, topologia e EDF (2026-09-10)

### 5.1 Seldon Core / KServe — timeouts fragmentados (anterioridade que nos favorece)

- **Seldon:** `seldon.io/rest-timeout` e `grpc-timeout` por nó; doc oficial adverte que em REST **o timeout vale por nó, não para o grafo fim-a-fim** — cada sub-request pode consumir o teto cheio.
- **KServe:** `timeoutSeconds` por componente (`InferenceGraph`, `InferenceService`), `affinity`/`resources` por nó, autoscaling por concorrência/RPS — sem operador de composição `AND/OR/N_OF_M/CHAIN` nem propagação de violação.
- **Diferença C2:** nossa árvore de deadlines com semântica de propagação + relaxamento/migração + herança é justamente o que falta aos dois. **Manter claims 8–9, mas redigir como “composição nativa no ISA com propagação definida”,** citando Seldon/KServe como problema (fragmentação), não como antecipação.

### 5.2 Kubernetes / NVIDIA Run:ai / KAI — topologia ciente (placement, não deadline)

- **K8s 1.36 (KEP-5732) + Run:ai + KAI Scheduler:** placement por NVLink/NVSwitch/rack/zona, NUMA-aware via Topology Manager (`none/best-effort/restricted/single-numa-node`) + GFD labels + NFD `NodeResourceTopology`, ganho até ~10× em coletivo (NCCL). Escopo: **onde** o Pod roda.
- **Diferença C2:** nossa claim 9 é **quando + com que urgência herdar**, integrada ao ISA (`SET_DEADLINE`, `BARRIER` com timeout→NACK, fila urgente passa bulk, `prio_eff=(deadline,localidade,base)`). Topologia decide localidade; EDF+herança decide preempção. **Complementares, não antecipação.** Citar KEP-5732/Run:ai no relatório §2.1(e) e manter claim 9 ancorada na herança cross-device com corte ≤16 hops + backpressure 80%.

### 5.3 EDF + herança — SRP/DFP (fundamentos, não aplicação em ISA de IA)

- **Base clássica:** Sha–Rajkumar–Lehoczky (PIP/PCP, 1990), Baker SRP (1991, padrão OSEK/AUTOSAR), Burns et al. DFP deadline-floor (2014), revisões Davis/York — bloqueio limitado a 1 seção crítica, sem deadlock/transitividade, análise por demanda.
- **Diferença C2:** aplicamos herança de **deadline absoluto por contexto de IA** dentro de VM de tensores com `FORK/ABORT`, `LOCK/UNLOCK/FENCE` e `BARRIER` com NACK — domínio e mecanismo distintos de RTOS uniprocessador. **Citar SRP/DFP como fundamento** (boa-fé) e reivindicar a instanciação (herança cross-device + timeouts obrigatórios + integração com rollback).

## 6. Achados do inventor (busca própria, 2026-09-10) + verificação do assistente

**Método:** amostra verificada via busca (DeltaBox e PHAROS confirmados; `kv-cache-migrator` NÃO localizado — tratar como não-verificado até URL/commit). Referências 2026 não verificadas abaixo vão marcadas **[A VERIFICAR]** antes de citar no depósito.

| # | Achado do inventor | Verificação | Impacto na claim |
|---|---|---|---|
| 1 | JP 2005025741 A + US 8,161,269 B2 (escape x86) | Plausível (família Intel escape 0x0F38/3A conhecida) **[A VERIFICAR nº JP]** | Reforça: novidade só na tríade VM (escape + `ext_len` + negociação + tabela). Nada a mudar, só citar JP se confirmado. |
| 2 | **DeltaBox arXiv:2605.22781** (CoW SO, ~14ms ckpt / ~5ms rb) | **CONFIRMADO** (v1: 14ms/≤6ms; v2: 10.83ms oculto / 1.86ms fork-template; DeltaFS overlay + DeltaCR/CRIU, Firecracker) | **Eleva risco 2 para médio.** Diferencial mantido: SO/sandbox+arquivos+processos vs **ISA de tensores heterogêneos** (KV+`h_t`+`H`, `FORK/ABORT`, I-Mono, janela k=16). Explicitar na descrição. |
| 2b | vLLM #4907 CoW fork + TensorRT-LLM #13453 replay Mamba-2 | Plausível (CoW fork existe no vLLM; replay Mamba coerente) **[A VERIFICAR nºs]** | Se confirmados, reforçam saturação de rollback mono-modelo; citar como tal. |
| 3 | **PHAROS arXiv:2604.05308** (EDF HAs safety-critical) + US 2023/0376752 + EDF-AoT | **PHAROS CONFIRMADO** (EDF/FIFO em aceleradores particionados, DSE p/ schedulability, DAC 2026). Demais **[A VERIFICAR]** | **Eleva risco 3 para alto.** Diferencial: **deadline composto AND/OR/N_OF_M/CHAIN p/ fluxos multimodais em VM de inferência** (PHAROS é DSE de hardware automotivo, sem composição booleana de fluxos). |
| 4 | `kv-cache-migrator` (81.73ms, −95% VRAM, AGPL) + tierkv + Symphony + CXL-SpecKV | **NÃO LOCALIZADO** (busca retorna mirror obscuro + paper Pallas 2608.16477, outro tema). **[NÃO CITAR sem URL/commit]** | Risco 4 fica em **médio por WAL-ausente-nos-demais**, não por esse repo. Diferencial WAL+`t_wal_retention`+Lamport mantido, mas como **divisional sem enablement**. |
| 5 | Predictive Multi-Tier (6 tiers, 70–84% hit, RoPE-aware prefetch) + HiKV + CAKE | Coerente com campo saturado (FlexGen/HiKV/CAKE existem) **[A VERIFICAR título exato]** | **Eleva risco 5 para alto.** Diferencial restante: **alternância determinística recomputação-vs-transferência** (`T_transf` vs `T_recomp`) — ainda não habilitada; manter como divisional. |
| 6 | StreamWise 2603.05800 + AI OS + OpenChimera + void-box | StreamWise plausível (multi-modal serving) **[A VERIFICAR]** | Risco 6 vai a **baixo (não muito baixo)**. Diferencial: **mesmo endereço + rollback conjunto híbrido** — nenhum faz. Ver claim 6 revisada em `09_CLAIM6_REVISADA.md`. |
| 7 | NMOS (prefetch SSD→RAM→VRAM, 70B em 4GB) + DualDecoder + MoE-SSD | NMOS como nome não localizado; ideia = FlexGen/DeepSpeed-offload **[A VERIFICAR]** | **Eleva risco 7 para alto.** `WEIGHTS_HINT` inexistente agrava. Fora do principal. |
| 8 | vLLM-Omni PersonaPlex PR #4771 + wrapper ONNX + speech-core | PersonaPlex/vLLM-Omni existem como projeto; nº do PR **[A VERIFICAR]** | Risco 8 vai a **médio**: serving full-duplex existe, **opcode não**. C1 parcial mantida. |
| 9 | US 2013/0061045 (vector compare) | Plausível (classe de opcodes SIMD) | Risco 9 **baixo confirmado** — sem antecipação de RAG-em-VM. Melhor 2ª claim. |
| 10 | WO 2026147586 A1 (NVIDIA tensor codecs) + US 2017/0364443 + Nexus 2504.19636 | Família NVIDIA nvcomp plausível; números **[A VERIFICAR]** | Risco 10 em **médio**: seleção-por-entropia-no-opcode segue livre; sem enablement, fora do principal. |

**Regra para o depósito:** só citar referência com número/data confirmados (artigo: arXiv ID + v1; patente: nº + data pub; repo: URL + commit). Tudo marcado acima vai em anexo de busca, não no relatório, até confirmado.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
