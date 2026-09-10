> **ARQUIVO — material de entrada, NÃO normativo.**
> Recebido como "mais doc" (catálogo de capacidades v2.4) em 2026-09-10.
> Premissa do doc ("12 RFCs implementados, ~180 opcodes ativos") É FALSA
> neste repo (42 opcodes executáveis; ondas/modelo Σ em curso). Números de
> capacidade e citações de medição auditados na revisão — vários citam
> arquivos que não existem ou linhas com outro conteúdo. A §3 (limites
> honestos) é a melhor parte e foi aproveitada (qualificador cross-hardware
> já vive na ESPEC §9). Canônico: `docs/ESPEC.md` (+ `docs/ESPEC-V2.md`
> DRAFT). Nada aqui deve ser implementado sem reconciliação. Conteúdo
> original verbatim abaixo.
>
> ---

# M³-AVM v2.4 — Catálogo Completo de Capacidades

> **Premissa:** ISA v2.4 com todos os 12 RFCs implementados, ~180 opcodes ativos, 24 regiões, 19 registradores especiais, 47 feature bits.  
> **Formato:** capacidades concretas com números, não marketing.

---

## 1. Sumário Executivo

Com tudo implementado, a M³-AVM é capaz de:

- Rodar **qualquer arquitetura de IA atual** (Transformer, Mamba, MoE, SNN, difusão, ODE, CNN, GNN, ML clássico) **na mesma VM**, com o mesmo bytecode.
- Escalar de **laptop (1 CPU)** a **pod de datacenter (8 GPUs + 100 nós)** sem mudar o código de usuário.
- Servir **modelos de 1TB+** em hardware de 80GB, via storage tiering + KV tiering.
- Garantir **determinismo bit-exato** com rollback em ~217µs (local) e failover distribuído em <5ms (LAN).
- Operar **full-duplex audio+video+text** com deadlines compostos e barge-in sub-ms.
- Fazer **RAG, ML clássico, treino local (LoRA/TTT)** dentro do mesmo processo.
- Ser **auditada** (tracing OTLP), **protegida** (ChaCha20-Poly1305, ratchet), **monitorada** (power/thermal), **resiliente** (clock health, failover).

Abaixo, o detalhamento por domínio.

---

## 2. Capacidades por Domínio

### 2.1 Inferência — Arquiteturas Suportadas

| Arquitetura | Como é executada | Opcodes principais |
|:---|:---|:---|
| **Transformer denso** (Llama, Mistral, Qwen) | Direto | `ATTN 0x02`, `FFN 0x08`, `ROPE 0x19` |
| **Transformer com FlashAttention** | IO-aware | `FLASH_ATTN 0x3A` |
| **Transformer com atenção esparsa** | Top-k via distância | `ATTN_SPARSE 0x3B` + `DISTANCE 0x23` |
| **Mamba / SSM** | Scan seletivo O(1) | `SSM_SCAN 0x13`, `SSM_RESET 0x14` |
| **DeltaNet / HGRN / Titans** | Recurrence matricial | `RANK1_UPDATE 0x24` |
| **MoE** (Mixtral, DeepSeek) | Gate + top-k + dispatch | `MATVEC 0x10` + `SAMPLE TOPK` + `GATHER 0x1F` |
| **RNN / LSTM / GRU / RWKV** | Lowering | `MATVEC` + `MUL` + `ADD` + `SIGMOID 0x3E` |
| **SNN / LIF** | Integrate-and-fire | `SPIKE_STEP 0x20` |
| **Difusão** (DDPM/DDIM) | Passo fundido | `DENOISE_STEP 0x21` |
| **ODE / Liquid** | Euler/RK2/RK4 | `ODE_STEP 0x25` |
| **CNN / ConvNeXt** | N-dim conv | `CONV 0x1E` |
| **GNN** | Gather/scatter | `GATHER 0x1F` |
| **XGBoost / Random Forest** | Travessia vetorizada | `FOREST 0x22` |
| **SVM / Logística / Linear** | Lowering | `MATVEC`, `SVM_PREDICT 0x5C` |
| **kNN / kMeans / DBSCAN** | Distância em lote | `DISTANCE 0x23` |
| **PCA / SVD** | Power iteration | `PCA_STEP 0x5D` |
| **RAG / Vector DB** | IVF, HNSW, PQ | `RAG_SEARCH 0x52`, `PQ_* 0x56/0x57` |
| **Full-duplex áudio** (Moshi, PersonaPlex) | Temporal + Depformer + Mimi | `DEPFORMER 0x44`, `STREAM_MERGE 0x45` |
| **Multi-modal** (áudio+texto+vídeo) | Deadlines compostos | `DEADLINE_CHAIN 0xD3` |

**Consequência prática:** você pode rodar **Moshi (áudio) + Llama (texto) + XGBoost (decisão) + RAG (conhecimento)** no **mesmo contexto**, compartilhando memória.

---

### 2.2 Escala — De Laptop a Datacenter

| Cenário | Hardware | O que roda |
|:---|:---|:---|
| **Laptop** | 1 CPU, 16 GB RAM, sem GPU | TinyLlama-1.1B Q4_K, ML clássico, RAG pequeno |
| **Workstation** | 1 CPU, 64 GB RAM, 1 GPU 24 GB | Llama-8B Q4_K, Moshi-7B, fine-tuning LoRA |
| **Rig médio** | 2 CPUs, 256 GB RAM, 4 GPUs 80 GB | Llama-70B TP=4, MoE 8 experts, contexto 1M tokens |
| **Servidor** | 2 CPUs, 2 TB RAM, 8 GPUs H100 | Llama-405B Q4_K, contexto 10M tokens, multi-tenant |
| **Cluster LAN** | 10 nós × 8 GPUs | Llama-405B com pipeline paralelo, serving distribuído |
| **Cluster WAN** | 100 nós globais | Federated learning, serving geo-distribuído |

**O mesmo bytecode roda em todos.** O runtime decide tier, device, transporte.

---

### 2.3 Memória — 24 Regiões Especializadas

| Capacidade | Habilitado por |
|:---|:---|
| **Pesos de 1TB+ em hardware de 80GB** | `WEIGHTS_COLD/WARM/HOT` + `PREFETCH_TIER` + `STREAM_WEIGHTS` |
| **Contexto de 10M tokens** | `KV_CACHE` + `KV_CACHE_COLD` + `KV_CACHE_REMOTE` + `KV_RECOMPUTE` |
| **Rollback determinístico** | `SNAPSHOT` + `RESTORE` + CoW versionado |
| **Zero-copy cross-context** | `SHARED` + `FENCE` |
| **Migração transacional** | `WAL` + `MIGRATE` |
| **Isolamento Erlang por ator** | `ARENA` por contexto |
| **Vector DB persistente** | `RAG_INDEX` |
| **Estado de treino** | `GRADIENTS` + `OPTIMIZER_STATE` |
| **Chaves criptográficas isoladas** | `KEYS` em `CONFIDENTIAL` |
| **Staging de rede** | `CLUSTER_STAGING` com backpressure |

---

### 2.4 Distribuição — Três Topologias Unificadas

| Topologia | Transporte | Latência | Banda | Uso |
|:---|:---|:---|:---|:---|
| **Intra-device** | Silicon | <1 ns | TB/s | Registradores |
| **Intra-rig** | NVLink / PCIe / UPI | 200 ns – 1 µs | 64 GB/s – 900 GB/s | Tensor parallel, MoE, pipeline |
| **Intra-LAN** | RDMA / TCP | 2 µs – 50 µs | 100 GB/s | Serving distribuído, KV remoto |
| **Inter-WAN** | Internet | 50 ms | 1 GB/s | Federated, geo-distribuído |

**Comandos unificados:** `MIGRATE`, `REDUCE_LOCAL/REMOTE`, `BARRIER_LOCAL/REMOTE`, `SEND_TENSOR`, `SIGNAL`, `SPAWN`. O runtime escolhe transporte por `TOPOLOGY[i][j]`.

**Failover:**
- GPU morre → `DEVICE_HEALTH` detecta → `MIGRATE_LOCAL` em <100ms
- Nó morre → `NODE_SUSPECT` → `NODE_DEAD` → novo nó assume em <2s
- Relógio salta → `CLOCK_HEALTH` detecta → deadlines relaxados automaticamente

---

### 2.5 Full-Duplex Moshi — Capacidades Concretas

| Capacidade | Como |
|:---|:---|
| **Áudio 24kHz 80ms frame** | `SENSE AUDIO_PCM` + `CODEC_ENC 0x15` |
| **17 streams (1 texto + 8 user + 8 agent)** | `KV_CACHE` estendido + `STREAM_MERGE 0x45` |
| **Temporal 32 layers 4096 dim** | `ATTN 0x02` + `FFN 0x08` |
| **Depformer 6 layers × 16 codebooks** | `DEPFORMER 0x44` |
| **Mimi encoder/decoder 16 RVQ** | `CODEC_ENC/DEC 0x15/0x16` |
| **Barge-in sub-ms** | `SENSE USER_INPUT` → `IF_INTERRUPT 0x0F` → `ABORT 0x05` |
| **VAD** | `VAD_DETECT 0x46` |
| **Echo cancellation** | `STREAM_MERGE MODE=echo_cancel` |
| **Acoustic delay 1 frame** | `DEPFORMER ... DELAY=1` |
| **Rollback de KV 17-streams** | `FORK 0x04` + `ABORT 0x05` (CoW) |
| **Modelo 7B em ~5GB** | Q4_K + fp16 depformer/mimi |

**Resultado:** áudio full-duplex com **~180ms latency** em CPU, **~80ms** em GPU.

---

### 2.6 Multi-Modal com Deadlines Compostos

| Cenário | Deadline | Como |
|:---|:---|:---|
| **Áudio 80ms + Vídeo 33ms** | `AND` | `DEADLINE_AND 0xD0` |
| **Áudio OU texto (fallback)** | `OR` | `DEADLINE_OR 0xD1` |
| **3 de 5 streams sincronizados** | `N_OF_M` | `DEADLINE_N_OF_M 0xD2` |
| **Áudio→texto→decisão encadeados** | `CHAIN` | `DEADLINE_CHAIN 0xD3` |
| **Relaxar vídeo se áudio atrasar** | `RELAX` | `DEADLINE_RELAX 0xD5` |

**Exemplo:** assistente de vídeo-conferência que processa áudio (80ms), vídeo (33ms), legendas (200ms) e detecção de emoção (500ms) com sincronia labial garantida.

---

### 2.7 Determinismo e Replay

| Capacidade | Habitado por |
|:---|:---|
| **Rollback local ~217µs** | `FORK 0x04` + `ABORT 0x05` (CoW) |
| **Replay bit-exato** | `RNG_SEED 0x60` + `LAMPORT` inline |
| **Ordem causal distribuída** | `LAMPORT` em todo `SIGNAL`/`SEND_TENSOR` |
| **Deadline EDF com herança** | `DEADLINE` inline + `PRIORITY_SET 0x73` |
| **Detecção de drift de clock** | `CLOCK_QUERY 0x78` + `CLOCK_HEALTH` |
| **Relaxamento adaptativo** | `DEADLINE_RELAX 0xD5` + `DEADLINE_NEGOTIATE 0x7B` |

**Exemplo:** você roda 1000 passos de inferência, aborta no passo 500, roda mais 500. A saída final é **bit-exata** ao rollout sem abort. Isso é ouro para debugging, A/B testing e compliance.

---

### 2.8 Treino e Fine-Tuning

| Capacidade | Como |
|:---|:---|
| **LoRA local** | `LORA_APPLY 0xF8` + `LORA_MERGE 0xF9` |
| **TTT (test-time training)** | `RANK1_UPDATE 0x24` + `MATVEC TRANSPOSE` |
| **Gradient accumulation** | `GRAD_ACCUM 0xF4` |
| **Adam / SGD / Lion** | `OPTIMIZER_STEP 0xF6` |
| **Gradient checkpointing** | `CHECKPOINT_SAVE/RESTORE 0xFA/0xFB` |
| **Mixed precision** | `CAST 0x67` + `QUANTIZE 0x68` |
| **Fine-tuning distribuído** | `REDUCE_REMOTE 0x86` (allreduce de gradientes) |

**Exemplo:** fine-tuning de LoRA em Llama-8B com 1M tokens, em 1 GPU, durante idle time, sem parar inferência.

---

### 2.9 RAG e ML Clássico

| Capacidade | Como |
|:---|:---|
| **Vector DB de 1B embeddings** | `RAG_INDEX_ADD 0x50` + `RAG_SEARCH 0x52` |
| **IVF, HNSW, PQ** | `RAG_SEARCH MODE=IVF/HNSW/PQ` |
| **Similaridade cosseno/euclidiana/dot** | `DISTANCE 0x23 METRIC=...` |
| **XGBoost 1000 árvores** | `FOREST 0x22` |
| **SVM RBF/poly/sigmoid** | `SVM_PREDICT 0x5C` |
| **kNN / kMeans / DBSCAN** | `DISTANCE` + `KMEANS_STEP 0x58` |
| **PCA / SVD** | `PCA_STEP 0x5D` |

**Exemplo:** pipeline completo — RAG busca 5 documentos, Llama processa, XGBoost valida decisão, tudo no mesmo contexto.

---

### 2.10 Segurança

| Capacidade | Como |
|:---|:---|
| **Criptografia autenticada** | `CRYPTO_ENCRYPT 0xDE` (ChaCha20-Poly1305) |
| **Key agreement** | `KEY_AGREE 0xD8` (X25519) |
| **Rotação de chaves** | `KEY_ROTATE 0xD9` (60s default) |
| **Forward secrecy** | Ratchet automático |
| **Replay protection** | `REPLAY_CHECK 0xDB` (window 1024) |
| **Assinatura de payload** | `CRYPTO_SIGN 0xDC` |
| **Isolamento de chaves** | Região `KEYS` em `CONFIDENTIAL` |

**Exemplo:** cluster de 100 nós em rede pública, todos os `SIGNAL`/`SEND_TENSOR` autenticados e resistentes a replay.

---

### 2.11 Observabilidade

| Capacidade | Como |
|:---|:---|
| **Distributed tracing** | `TRACE_SPAN_START/END 0xE0/0xE1` |
| **Correlação cross-node** | `TRACE_INJECT/EXTRACT 0xE5/0xE6` |
| **W3C Trace Context** | Formato padrão |
| **OTLP export** | `TRACE_EXPORT 0xE4` para Jaeger/Tempo |
| **Atributos estruturados** | `TRACE_ATTR 0xE2` |
| **Span links** | `TRACE_LINK 0xE3` (cross-trace) |

**Exemplo:** debug de um `SIGNAL` que não chega — você vê o trace completo: emissão → rede → recepção → handler → rollback.

---

### 2.12 Power e Thermal

| Capacidade | Como |
|:---|:---|
| **Telemetria de energia** | `POWER_QUERY 0xC8` (mW) |
| **Budget por contexto** | `POWER_BUDGET_SET 0xC9` |
| **Telemetria térmica** | `THERMAL_QUERY 0xCB` (m°C) |
| **Políticas de throttling** | `THROTTLE_POLICY 0xCC` |
| **Hard cap de device** | `POWER_CAP 0xCD` |
| **Contabilização de energia** | `ENERGY_ACCOUNT 0xCE` |
| **Headroom térmico** | `THERMAL_HEADROOM 0xCF` |

**Exemplo:** pod de 8 H100 com budget de 5.6 kW. Quando temperatura passa de 80°C, runtime dropa contextos GREEN primeiro, depois BLUE, mantém RED com deadline apertado.

---

### 2.13 Compressão

| Capacidade | Como |
|:---|:---|
| **LZ4 em trânsito** | `COMPRESS 0xF0` |
| **Zstd adaptativo** | `COMPRESS_QUERY 0xF2` escolhe |
| **Q4_K-native** | Compressão nativa de tensores quantizados |
| **Estimativa de entropia** | `ENTROPY_ESTIMATE 0xF3` |

**Exemplo:** `MIGRATE` de 10GB de KV-cache entre nós: 12s sem compressão → **3s com Zstd**.

---

### 2.14 Multi-Backend

| Capacidade | Como |
|:---|:---|
| **GGUF (v1/v2/v3)** | `MODEL_LOAD_UNIFIED 0xFC` |
| **safetensors** | idem |
| **ONNX (subset)** | idem |
| **TorchScript (subset)** | idem |
| **Modelo híbrido** | Carrega cada parte do formato nativo |
| **Compilação para M3** | `MODEL_COMPILE 0xFD` |

**Exemplo:** Moshi (safetensors) + Llama (GGUF) + XGBoost (ONNX) no mesmo contexto, sem conversão manual.

---

## 3. O Que a VM **NÃO** Consegue Fazer (Limites Honestos)

Lista do que está **fora** do escopo atual:

| Limitação | Por quê | Quando resolver |
|:---|:---|:---|
| **Treino full (não LoRA)** | Faltam otimizadores distribuídos (ZeRO, FSDP) | RFC futuro |
| **Replay bit-exato cross-hardware** | FP32 em x86 ≠ ARM ≠ GPU | Impossível sem emulação |
| **Modelos quânticos reais** | Só simulador | Hardware futuro |
| **Fotônica real** | Sem hardware | Reservado em `0xD0–0xDF` |
| **Neuromórfico analógico** | Sem hardware | Reservado |
| **Multi-tenant com MIG completo** | Parcial (RFC-0002) | RFC futuro |
| **Compliance (GDPR, HIPAA, EU AI Act)** | Fora do escopo | RFC futuro |
| **Adversarial robustness** | Detecção de alucinação fora do escopo | RFC futuro |
| **Human-in-the-loop** | Fora do escopo | RFC futuro |
| **Provenance / SBOM** | Modelos assinados fora do escopo | RFC futuro |

---

## 4. Cenários Concretos

### Cenário 1 — Assistente de voz full-duplex
```
Hardware : 1 workstation (1 GPU 24GB, 64GB RAM)
Modelo   : Moshi-7B Q4_K + Llama-8B Q4_K (fallback)
Capacidades usadas:
  - SENSE AUDIO_PCM → CODEC_ENC → DEPFORMER
  - Barge-in via SENSE USER_INPUT → ABORT
  - DEADLINE_AND (áudio 80ms + texto 200ms)
  - Rollback de KV 17-streams
Latência : ~180ms CPU / ~80ms GPU
```

### Cenário 2 — Serving distribuído Llama-405B
```
Hardware : 4 nós × 8 GPUs H100 + NVSwitch + RDMA
Modelo   : Llama-405B Q4_K (~200GB) em WEIGHTS_COLD
Capacidades usadas:
  - STREAM_WEIGHTS (camadas 0-10 em VRAM)
  - SHARD TP=8 + PP=4
  - REDUCE_LOCAL (NVLink) + REDUCE_REMOTE (RDMA)
  - KV_CACHE_TIERING (contexto 1M tokens)
  - TRACE_SPAN cross-node
Throughput: ~2000 tokens/s
```

### Cenário 3 — RAG + ML clássico em produção
```
Hardware : 1 servidor CPU-only
Modelo   : Llama-3-8B Q4_K + XGBoost + BGE embeddings
Capacidades usadas:
  - RAG_INDEX (100M docs)
  - PQ_ENCODE (4-bit)
  - RAG_SEARCH (IVF, top-5)
  - FOREST (1000 árvores)
  - SVM_PREDICT
Latência : ~500ms end-to-end
```

### Cenário 4 — Fine-tuning LoRA em idle time
```
Hardware : 1 workstation (1 GPU 24GB)
Modelo   : Llama-8B + LoRA adapter
Capacidades usadas:
  - LORA_APPLY durante inferência
  - GRAD_ACCUM + OPTIMIZER_STEP em background
  - CHECKPOINT_SAVE a cada 1000 steps
  - POWER_BUDGET_SET (limita watts em idle)
Custo    : 0 (roda em idle) vs ~$50/dia em cloud
```

### Cenário 5 — Cluster geo-distribuído federated
```
Hardware : 100 nós globais (10 por região)
Modelo   : Llama-70B (federated)
Capacidades usadas:
  - NODE_JOIN + KEY_AGREE + CRYPTO_ENCRYPT
  - REDUCE_REMOTE (allreduce de gradientes)
  - CLOCK_QUERY (adapta deadlines cross-region)
  - TRACE_EXPORT para observabilidade global
Compliance: dados nunca saem da região (WEIGHTS_WARM local)
```

---

## 5. Métricas de Referência (metas honestas)

| Métrica | Alvo | Onde medir |
|:---|:---|:---|
| Rollback local | ~217µs | `src/qa.rs:664` |
| Failover GPU | <100ms | `benches/device_failover.rs` |
| Failover nó | <2s | `benches/cluster_failover.rs` |
| Barge-in LAN | <5ms | `benches/dist_barge.rs` |
| Throughput Moshi CPU | ~180ms/frame | `benches/moshi_full.rs` |
| Throughput Moshi GPU | ~80ms/frame | idem |
| KV eviction latency | ~10µs | `benches/kv_tiering.rs` |
| Compressão Zstd | 3–5× | `benches/compression.rs` |
| Tracing overhead | <2% | `benches/tracing.rs` |
| Security handshake | ~1ms | `benches/crypto.rs` |

---

## 6. Resumo em Uma Frase

**Com tudo implementado, a M³-AVM é o primeiro sistema operacional de inferência que roda qualquer arquitetura de IA — clássica ou moderna, single-model ou multi-model, local ou distribuída, segura e observável — em qualquer hardware, de um laptop a um datacenter, com o mesmo bytecode, sem quebra binária por décadas.**

Isso não é marketing. É a consequência direta de 12 RFCs bem definidos, 180 opcodes bem alocados e uma disciplina de compatibilidade que vem do mainframe.

**A pergunta que fica:** o que você quer construir em cima disso?

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
