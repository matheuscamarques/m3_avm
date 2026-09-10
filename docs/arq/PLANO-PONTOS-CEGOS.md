> **ARQUIVO — material de entrada, NÃO normativo.**
> Recebido como "novos dados" (pré-planejamento: 10 pontos cegos +
> meta-processo) em 2026-09-10. Colisões com o registro deste repo:
> (1) números RFC-0003–0006 JÁ EXISTEM aqui (Retention/GATHER/Determinism/
> Telemetry, todos IMPLEMENTED); (2) faixa `0xC0–0xFE` conflita com nossos
> placeholders R12; (3) regiões `0x10–0x17` exigem seletor de 5 bits
> (formato atual: 4 bits — mudança MAJOR); (4) feature bits 18–46 colidem
> com bit 32 (PHOTONIC) e presumem bits 10–17 do LCT (não adotados).
> A especificação canônica é `docs/ESPEC.md` (+ `docs/ESPEC-V2.md` DRAFT).
> Revisão na conversa de trabalho. Nada aqui deve ser implementado sem
> reconciliação (e, pelos próprios termos do doc, isto é mapa, não RFC).
> Conteúdo original preservado verbatim abaixo.
>
> ---

# Pré-planejamento: 10 Pontos Cegos + Meta-Processo

> **Objetivo:** transformar cada ponto cego em um RFC acionável, com prioridade, custo, dependências e MVP. Não é implementação — é o mapa do terreno antes de cavar.

---

## Metodologia

Para cada ponto cego, defino:

1. **Problema** — o que quebra
2. **Cenário de falha** — como aparece na prática
3. **RFC** — número e título
4. **MVP** — menor solução que resolve 80% do problema
5. **Solução completa** — visão de 10 anos
6. **Prioridade** — P0 (bloqueia produção), P1 (necessário para escala), P2 (desejável)
7. **Esforço** — semanas-homem
8. **Dependências** — o que precisa antes
9. **Mudanças na spec** — opcodes, regiões, feature bits

---

## Tabela Resumo

| # | Ponto Cego | RFC | Prio | Esforço | Dep. |
|:--:|:---|:---|:--:|:--:|:---|
| 1 | Armazenamento (NVMe/RDMA-storage) | RFC-0003 Storage Tiering | **P0** | 4–6 sem | — |
| 2 | Power/Thermal | RFC-0004 Power Management | P1 | 3–4 sem | — |
| 3 | Multi-modal (deadline composto) | RFC-0005 Composite Deadlines | P1 | 2–3 sem | RFC-0012 |
| 4 | Segurança de rede | RFC-0006 Cluster Security | **P0** | 4–6 sem | — |
| 5 | Observabilidade | RFC-0007 Observability | P1 | 3–4 sem | RFC-0006 |
| 6 | Estado gigante (KV 1TB) | RFC-0008 Tiered KV-Cache | **P0** | 6–8 sem | RFC-0003 |
| 7 | Compressão adaptativa | RFC-0009 Compression | P1 | 2–3 sem | — |
| 8 | Treino/fine-tuning | RFC-0010 Inference-Time Training | P2 | 6–8 sem | RFC-0003, RFC-0008 |
| 9 | Múltiplos backends | RFC-0011 Unified Model IR | P1 | 8–10 sem | RFC-0003 |
| 10 | Falha de relógio | RFC-0012 Clock Resilience | **P0** | 2–3 sem | — |

**Total sequencial:** ~40–55 semanas  
**Total paralelizado (2–3 trilhas):** ~20–25 semanas

---

## 1. Armazenamento (NVMe/RDMA-storage)

### Problema
Modelos de 1TB+ (Llama-405B, GPT-4-class) não cabem em VRAM (80GB) nem RAM (2TB típico). `WEIGHTS` é mmap único sem tiering, sem streaming, sem prefetch inteligente.

### Cenário de falha
Você carrega Llama-405B Q4_K (~200GB). `mmap` funciona, mas o page fault aleatório durante inferência mata a latência. `DEADLINE=80ms` estoura. O runtime não sabe que camadas 0–5 já foram usadas e camadas 6–10 vêm a seguir.

### RFC-0003 — Storage Tiering

**Hierarquia de camadas:**
```
L0: VRAM (HBM)         ~80 GB     ~3 TB/s   <1 µs
L1: RAM (DDR)          ~2 TB      ~200 GB/s ~100 ns
L2: NVMe (PCIe Gen5)   ~30 TB     ~14 GB/s  ~20 µs
L3: NVMe-oF (RDMA)     ~∞         ~400 GB/s ~2 µs
L4: S3/blob storage    ~∞         ~1 GB/s   ~50 ms
```

**Opcodes novos (`0xC0–0xC7`):**
- `PIN` — fixa tensor em tier específico
- `UNPIN` — libera
- `PREFETCH_TIER` — pré-carrega próximo tier
- `STREAM_WEIGHTS` — streaming sequencial de camadas
- `EVICT` — remove de tier
- `TIER_QUERY` — telemetria (hit rate, latência)
- `TIER_POLICY` — define LRU/LFU/pin-first
- `WEIGHTS_HINT` — pré-declara padrão de acesso

**Regiões novas:**
- `0x10`: `WEIGHTS_COLD` (NVMe-backed, mmap)
- `0x11`: `WEIGHTS_WARM` (RAM-backed)
- `0x12`: `WEIGHTS_HOT` (VRAM-backed)

**Feature bits:** `STORAGE_TIERING` (18), `NVME_STREAM` (19), `RDMA_STORAGE` (20)

**MVP:** mmap + prefetch sequencial + LRU eviction entre RAM↔NVMe. Sem VRAM tiering (fica para v2.1).

**Solução completa:** 5-tier com prefetch preditivo (baseado em sequência de camadas), RDMA-storage remoto, compressão inline.

**Esforço:** 4–6 semanas.

---

## 2. Power/Thermal

### Problema
Um pod com 8 H100 = 5.6 kW. Throttling térmico reduz clock em 30%+. `DEADLINE` EDF assume capacidade constante, mas o hardware não entrega.

### Cenário de falha
Você agenda 4 contextos RED com deadlines de 80ms. Quando todos rodam simultaneamente, o GPU cluster esquenta. Throttling entra. Deadlines estouram. Scheduler não sabe que precisa **reduzir carga**, só que precisa **cumprir prazo**.

### RFC-0004 — Power and Thermal Management

**Modelo:**
- Orçamento de energia por contexto (mW)
- Orçamento térmico por device (m°C)
- Políticas de degradação graciosa

**Opcodes novos (`0xC8–0xCF`):**
- `POWER_QUERY` — telemetria (mW, temperatura, throttling %)
- `POWER_BUDGET_SET` — define budget para contexto
- `POWER_BUDGET_GET` — lê budget
- `THERMAL_QUERY` — temperatura por device
- `THROTTLE_POLICY` — define degradação (drop GREEN, relax deadline, migrar device)
- `POWER_CAP` — hard cap de device
- `ENERGY_ACCOUNT` — contabiliza energia consumida
- `THERMAL_HEADROOM` — espaço antes de throttling

**Registradores especiais novos:**
- `SP12`: `POWER_BUDGET_MW`
- `SP13`: `THERMAL_HEADROOM_MC`
- `SP14`: `THROTTLE_STATE`

**Feature bits:** `POWER_AWARE` (21), `THERMAL_AWARE` (22), `ENERGY_ACCOUNTING` (23)

**MVP:** Query + report + degradação simples (drop GREEN contexts primeiro). Sem modelagem preditiva.

**Solução completa:** Modelo térmico preditivo, DVFS-aware scheduling, energy-aware placement cross-rig.

**Esforço:** 3–4 semanas.

---

## 3. Multi-modal (deadline composto)

### Problema
Full-duplex (áudio + vídeo + texto) tem deadlines que interagem: áudio 80ms, vídeo 33ms (30fps), texto 200ms. Um atraso no vídeo pode quebrar sincronia labial.

### Cenário de falha
Moshi roda áudio a 80ms. Adiciona vídeo a 33ms. Se áudio atrasa 20ms, o vídeo tem que esperar ou desincronizar. EDF atual só conhece um deadline por contexto.

### RFC-0005 — Composite Deadlines

**Modelo:**
- Deadline é uma árvore booleana de sub-deadlines
- Operadores: `AND`, `OR`, `N_OF_M`, `CHAIN`, `DEADLINE_OF`
- Sincronização entre streams via `CHAIN`

**Opcodes novos (`0xD0–0xD7`):**
- `DEADLINE_AND` — todos os sub-deadlines devem ser cumpridos
- `DEADLINE_OR` — pelo menos um
- `DEADLINE_N_OF_M` — N de M
- `DEADLINE_CHAIN` — encadeia (dependência temporal)
- `DEADLINE_QUERY` — status da árvore
- `DEADLINE_RELAX` — relaxa sub-deadline específico
- `DEADLINE_TIGHTEN` — aperta
- `DEADLINE_RESET` — reinicia árvore

**Registradores especiais:**
- `SP15`: `DEADLINE_TREE_ID`
- `SP16`: `DEADLINE_TREE_MASK`

**Feature bits:** `COMPOSITE_DEADLINE` (24), `MULTI_STREAM_SYNC` (25)

**MVP:** `AND`/`OR` apenas. `N_OF_M` fica para v2.1.

**Solução completa:** Deadlines hierárquicos com propagação de constraints, adaptação dinâmica baseada em sincronia audiovisual.

**Esforço:** 2–3 semanas (depende de RFC-0012).

**Dependência:** RFC-0012 (relógio confiável) — sem ele, sub-deadlines são imprecisos.

---

## 4. Segurança de rede

### Problema
`SIGNAL` carrega HMAC truncado em `PAYLOAD_EXT` (8 bytes). Sem rotação de chaves, sem forward secrecy, sem replay protection. Um adversário que captura um `SIGNAL` pode reenviar, forjar ou descriptografar.

### Cenário de falha
Node A envia `SIGNAL KIND=ABORT` para Node B. Atacante captura pacote, reenvia 100x. Node B aborta 100x em loop. Ou pior: atacante modifica `ctx_id` para abortar contexto errado.

### RFC-0006 — Cluster Security

**Modelo:**
- Handshake X25519 para key agreement
- ChaCha20-Poly1305 para authenticated encryption
- Rotação de chaves a cada `T_rotate` (default 60s)
- Replay window de 1024 mensagens
- Forward secrecy via ratchet

**Opcodes novos (`0xD8–0xDF`):**
- `KEY_AGREE` — X25519 handshake
- `KEY_ROTATE` — rotação de chave
- `KEY_DERIVE` — HKDF
- `REPLAY_CHECK` — valida contra window
- `CRYPTO_SIGN` — assina payload
- `CRYPTO_VERIFY` — verifica assinatura
- `CRYPTO_ENCRYPT` — encapsula
- `CRYPTO_DECRYPT` — desencapsula

**Região nova:**
- `0x13`: `KEYS` (em `CONFIDENTIAL 0xD`, isolada)

**Feature bits:** `NOISE_PROTOCOL` (26), `FORWARD_SECRECY` (27), `REPLAY_PROTECTION` (28), `KEY_ROTATION` (29)

**MVP:** X25519 + ChaCha20-Poly1305 + replay window 1024. Sem ratchet (fica para v2.1).

**Solução completa:** Noise Protocol Framework completo, ratchet de chaves, post-quantum (Kyber) fallback.

**Esforço:** 4–6 semanas.

**Prioridade:** P0. Não negociável para produção.

---

## 5. Observabilidade

### Problema
`TRACE_EVENT` sem formato, sem correlação cross-node, sem export. Impossível debugar cluster distribuído em produção.

### Cenário de falha
Um `SIGNAL` não chega. Você tem 10.000 traces locais sem correlação. Não sabe se o pacote foi perdido na rede, se o handler falhou, ou se o deadline expirou antes.

### RFC-0007 — Observability and Tracing

**Modelo:**
- W3C Trace Context (trace_id, span_id, parent_span_id)
- Spans abertos/fechados em torno de operações
- Atributos estruturados (key-value)
- Export via OTLP (OpenTelemetry Protocol)
- Correlação cross-node via header

**Opcodes novos (`0xE0–0xE7`):**
- `TRACE_SPAN_START`
- `TRACE_SPAN_END`
- `TRACE_ATTR`
- `TRACE_LINK` (span link, cross-trace)
- `TRACE_EXPORT`
- `TRACE_INJECT` (injeta contexto em `SIGNAL`/`SEND_TENSOR`)
- `TRACE_EXTRACT`
- `TRACE_QUERY`

**Feature bits:** `DISTRIBUTED_TRACING` (30), `OTLP_EXPORT` (31), `SPAN_LINKS` (32, extensão)

**MVP:** W3C Trace Context + buffer local + export OTLP para collector.

**Solução completa:** Tracing contínuo (head + tail sampling), profiling contínuo (eBPF-style), correlação com métricas e logs.

**Esforço:** 3–4 semanas.

**Dependência:** RFC-0006 (traces precisam de transporte seguro).

---

## 6. Estado gigante (KV-cache 1TB)

### Problema
Contexto de 10M tokens = 1TB de KV-cache (para modelo 70B, 32 layers, 128 heads, 128 dim). Não cabe em nenhuma região.

### Cenário de falha
Você roda Llama-70B com contexto de 1M tokens. KV-cache = 100GB. Não cabe em VRAM (80GB). Runtime tenta alocar, falha, aborta.

### RFC-0008 — Tiered KV-Cache

**Modelo:**
- KV-cache distribuído por tiers (VRAM → RAM → NVMe → remoto)
- Eviction inteligente (attention-aware, não LRU cego)
- Sparse attention com recall de tokens relevantes
- Recomputação sob demanda (trade-off compute vs memória)

**Opcodes novos (`0xE8–0xEF`):**
- `KV_TIER_MOVE` — move camada entre tiers
- `KV_EVICT` — remove camada (com política)
- `KV_PIN` — fixa camada
- `KV_RETRIEVE` — recall de camada evictada
- `KV_RECOMPUTE` — recomputa camada a partir de input
- `KV_COMPRESS_LOSSY` — quantização adaptativa
- `KV_SPARSE_MASK` — máscara de tokens relevantes
- `KV_MIGRATE_REMOTE` — move para nó remoto

**Regiões novas:**
- `0x14`: `KV_CACHE_COLD` (NVMe)
- `0x15`: `KV_CACHE_REMOTE` (cluster)

**Feature bits:** `KV_TIERING` (33), `KV_SPARSE_EVICT` (34), `KV_RECOMPUTE` (35)

**MVP:** 2-tier (VRAM + RAM) com LRU + atenção-aware eviction.

**Solução completa:** 4-tier com recomputação parcial, sparse attention unificada, KV-cache compartilhado entre contextos.

**Esforço:** 6–8 semanas.

**Dependência:** RFC-0003 (storage tiering).

---

## 7. Compressão adaptativa

### Problema
`SEND_TENSOR` manda 4GB sem compressão. Cluster com 10 GbE leva 3.2 segundos. Inaceitável.

### Cenário de falha
Você faz `MIGRATE` de um KV-cache de 10GB entre nós. Sem compressão, leva 80s. Com LZ4, 20s. Com Zstd nível 3, 12s.

### RFC-0009 — Adaptive Compression

**Modelo:**
- Compressão por tipo de dado (tensor denso, esparso, quantizado, KV-cache)
- Seleção de algoritmo por entropia estimada
- Compressão inline durante `SEND_TENSOR`/`MIGRATE`
- Descompressão zero-copy quando possível

**Opcodes novos (`0xF0–0xF3`):**
- `COMPRESS` — comprime tensor
- `DECOMPRESS` — descomprime
- `COMPRESS_QUERY` — retorna algoritmo ótimo
- `ENTROPY_ESTIMATE` — estima entropia

**Algoritmos suportados:**
- LZ4 (fast, 2–3× ratio)
- Zstd (balanced, 3–5× ratio)
- Q4_K-native (tensor quantizado, 7× ratio)
- Custom sparse (CSR + delta encoding)

**Feature bits:** `COMPRESSION` (36), `ADAPTIVE_COMPRESSION` (37)

**MVP:** LZ4 + Zstd + Q4_K-native. Sem custom sparse.

**Solução completa:** Compressão neural (autoencoder pequeno), delta compression entre tensores similares, compressão lossy-aware de KV-cache.

**Esforço:** 2–3 semanas.

---

## 8. Treino/fine-tuning

### Problema
Assumimos inferência. LoRA/TTT local existe como lowering, mas sem otimizador, sem gradient checkpointing, sem mixed precision manager.

### Cenário de falha
Você quer fazer fine-tuning de LoRA em um modelo 7B com 1M tokens de dados. Precisa de gradientes, optimizer states (Adam = 2× params), gradient accumulation. Nada disso existe.

### RFC-0010 — Inference-Time Training

**Modelo:**
- LoRA nativo (adapter por contexto)
- TTT (test-time training) para Titans/MIRAS
- Gradient accumulation
- Otimizador (Adam, SGD, Lion)
- Gradient checkpointing (recomputa ativações)
- Mixed precision (FP32 master weights, BF16 compute)

**Opcodes novos (`0xF4–0xFB`):**
- `GRAD_ACCUM` — acumula gradiente
- `GRAD_ZERO` — zera
- `OPTIMIZER_STEP` — aplica update
- `OPTIMIZER_INIT` — inicializa estado
- `LORA_APPLY` — aplica adapter
- `LORA_MERGE` — merge no peso base
- `CHECKPOINT_SAVE` — salva
- `CHECKPOINT_RESTORE` — restaura

**Regiões novas:**
- `0x16`: `GRADIENTS`
- `0x17`: `OPTIMIZER_STATE`

**Feature bits:** `TRAINING` (38), `LORA` (39), `GRAD_CHECKPOINT` (40), `MIXED_PRECISION` (41)

**MVP:** LoRA apenas, Adam, sem gradient checkpointing.

**Solução completa:** Full fine-tuning, distributed training (FSDP, ZeRO), mixed precision completo, gradient compression.

**Esforço:** 6–8 semanas.

**Dependência:** RFC-0003, RFC-0008.

**Prioridade:** P2 (inferência primeiro, treino depois).

---

## 9. Múltiplos backends

### Problema
GGUF + safetensors + ONNX + TorchScript híbridos. Sem loader unificado.

### Cenário de falha
Você quer rodar Moshi (safetensors) + Llama (GGUF) no mesmo contexto. Runtime tem `ModelConfig::from_gguf` só.

### RFC-0011 — Unified Model IR (M3IR)

**Modelo:**
- IR intermediária que todos os formatos compilam para
- Operações canônicas mapeadas para opcodes da ISA
- Quantização como atributo, não como formato
- Metadata unificada (shapes, layers, arch)

**Opcodes novos (`0xFC–0xFE`):**
- `MODEL_LOAD_UNIFIED` — carrega qualquer formato
- `MODEL_COMPILE` — compila para bytecode M3
- `MODEL_QUERY` — metadata

**Formatos suportados:**
- GGUF (v1, v2, v3)
- safetensors
- ONNX (subset)
- TorchScript (subset)
- MLX (futuro)

**Feature bits:** `M3IR` (42), `MULTI_BACKEND` (43)

**MVP:** GGUF + safetensors.

**Solução completa:** ONNX + TorchScript + MLX, com importação simbólica.

**Esforço:** 8–10 semanas.

**Dependência:** RFC-0003.

---

## 10. Falha de relógio

### Problema
`LAMPORT` e `DEADLINE` assumem relógio monotônico estável. NTP drift, VM migration, suspensão e hibernação quebram isso.

### Cenário de falha
VM migra entre hosts. `Instant::now()` salta 500ms para trás. Todos os deadlines viram imprecisos. Scheduler começa a violar EDF.

### RFC-0012 — Clock Resilience

**Modelo:**
- Monitoramento de saúde do relógio
- Detecção de jumps (`> 100ms` = suspeito)
- Fallback para relógio lógico (`LAMPORT` puro)
- Adaptação de deadlines (relax + renegociação)
- Sincronização via PTP/NTP com bounds

**Opcodes novos:**
- `CLOCK_QUERY` — estado do relógio
- `CLOCK_SYNC` — força sincronização
- `CLOCK_ADJUST` — aplica offset
- `DEADLINE_RELAX` — relaxa deadline após jump
- `DEADLINE_NEGOTIATE` — renegocia com deadline distribuído

**Registradores especiais:**
- `SP17`: `CLOCK_HEALTH` (ok, drift, jump, frozen)
- `SP18`: `CLOCK_OFFSET_NS`

**Feature bits:** `CLOCK_HEALTH` (44), `DEADLINE_ADAPT` (45), `PTP_SYNC` (46)

**MVP:** Detecção de jumps > 100ms, log, relaxa deadlines.

**Solução completa:** PTP com hardware timestamping, relógio lógico híbrido (HLC), deadlines adaptativos.

**Esforço:** 2–3 semanas.

---

## Matriz de Priorização

### P0 — Bloqueia produção (implementar primeiro)

| RFC | Por quê |
|:---|:---|
| **0006 Security** | Sem isso, cluster é inseguro |
| **0012 Clock** | Sem isso, deadlines são ficção |
| **0003 Storage** | Sem isso, modelos grandes não rodam |
| **0008 KV Tiering** | Sem isso, contexto longo não roda |

### P0 — Bloqueia produção (implementar primeiro)

| RFC | Por quê |
|:---|:---|
| **0006 Security** | Sem isso, cluster é inseguro |
| **0012 Clock** | Sem isso, deadlines são ficção |
| **0003 Storage** | Sem isso, modelos grandes não rodam |
| **0008 KV Tiering** | Sem isso, contexto longo não roda |

### P1 — Necessário para escala (implementar em paralelo)

| RFC | Por quê |
|:---|:---|
| **0004 Power** | Datacenter real precisa |
| **0005 Composite Deadline** | Multi-modal real precisa |
| **0007 Observability** | Produção precisa |
| **0009 Compression** | Cluster em escala precisa |
| **0011 Unified IR** | Modelos híbridos precisam |

### P2 — Desejável (implementar depois)

| RFC | Por quê |
|:---|:---|
| **0010 Training** | Inferência primeiro |

---

## Fases de Implementação

### Fase 1 — Fundação (4–6 semanas, paralelo)
- RFC-0012 (Clock) — 2–3 sem
- RFC-0006 (Security) — 4–6 sem

**Marco:** Cluster seguro com deadlines confiáveis.

### Fase 2 — Armazenamento (6–8 semanas, paralelo)
- RFC-0003 (Storage) — 4–6 sem
- RFC-0008 (KV Tiering) — 6–8 sem

**Marco:** Modelos de 1TB+ e contexto de 1M+ tokens funcionam.

### Fase 3 — Operação (5–7 semanas, paralelo)
- RFC-0004 (Power) — 3–4 sem
- RFC-0007 (Observability) — 3–4 sem
- RFC-0009 (Compression) — 2–3 sem

**Marco:** Produção em datacenter é viável.

### Fase 4 — Multi-modal (3–4 semanas)
- RFC-0005 (Composite Deadline) — 2–3 sem

**Marco:** Full-duplex audio+video funciona.

### Fase 5 — Modelos híbridos (8–10 semanas)
- RFC-0011 (Unified IR) — 8–10 sem

**Marco:** GGUF + safetensors + ONNX no mesmo contexto.

### Fase 6 — Treino (6–8 semanas)
- RFC-0010 (Training) — 6–8 sem

**Marco:** Fine-tuning local funciona.

**Total paralelizado:** ~20–25 semanas.

---

## Dependências entre RFCs

```
RFC-0012 (Clock) ──┬──> RFC-0005 (Composite Deadline)
                   │
RFC-0006 (Security)┴──> RFC-0007 (Observability)

RFC-0003 (Storage) ─┬──> RFC-0008 (KV Tiering)
                    ├──> RFC-0011 (Unified IR)
                    └──> RFC-0010 (Training) <── RFC-0008
```

**Caminho crítico:** RFC-0003 → RFC-0008 → RFC-0010. ~18 semanas.

**Paralelizável:** RFC-0004, 0005, 0007, 0009 são independentes.

---

## Meta-Processo: Como achar os próximos 10 pontos cegos

Estes 10 são os que eu vejo **agora**. Haverá outros. O meta-processo para encontrá-los:

### 1. Red team trimestral
A cada 3 meses, alguém (humano ou IA) faz revisão adversarial da spec. Objetivo: **quebrar** a spec, não elogiá-la.

### 2. Devil's advocate por RFC
Todo RFC tem um revisor designado que **deve** tentar refutar. Se não conseguir, RFC avança.

### 3. Cenários de stress obrigatórios
Para cada RFC, responder:
- "E se a GPU morrer?"
- "E se o relógio saltar?"
- "E se a rede particionar?"
- "E se o modelo for malicioso?"
- "E se o usuário for adversário?"
- "E se o hardware for heterogêneo?"

### 4. Cross-pollination
Estudar como HPC, embedded, aerospace, telecom resolvem problemas análogos. Eles têm 40+ anos de lições.

### 5. Post-mortems de outros sistemas
Kubernetes, Erlang, CUDA, Ray, PyTorch — cada um tem falhas documentadas. Estudar.

### 6. Adversarial users
Convidar alguém de segurança ofensiva para auditar. Eles acham em 1 dia o que o autor não acha em 1 ano.

---

## O que Ainda Posso Estar Cego (lista honesta)

Depois de gerar 10 pontos cegos, é justo listar mais 10 que **eu mesmo** suspeito existirem:

| # | Área | Ponto cego provável |
|:--:|:---|:---|
| 11 | **Compliance/audit** | GDPR, HIPAA, EU AI Act. Logs de auditoria, lineage, right-to-erasure |
| 12 | **Multi-tenancy** | Múltiplos usuários no mesmo rig. Isolamento hardware (MIG, confidential compute) |
| 13 | **Determinismo cross-hardware** | FP32 em x86 ≠ ARM ≠ GPU. Replay bit-exato é impossível sem emulação |
| 14 | **Warmup/coldstart** | Primeira inferência é 10× mais lenta. Políticas de priming |
| 15 | **Model drift** | Modelo desatualiza. Monitoramento, detecção de drift |
| 16 | **Adversarial inputs** | Modelo envenenado, GGUF malformado, prompt injection na VM |
| 17 | **Model provenance** | Modelos assinados, SBOM, supply chain |
| 18 | **Regulatory** | Export controls, data residency, soberania |
| 19 | **Failure modes of AI** | Detecção de alucinação, safety filters, moderação |
| 20 | **Human-in-the-loop** | Aprovação humana, escalation, override |

Cada um vira RFC quando você perguntar. O meta-processo é o que garante que eles **sejam encontrados**, não que eu os preveja.

---

## Próximos Passos Concretos

1. **Escrever RFC-0003 (Storage)** — desbloqueia 3 outros RFCs
2. **Escrever RFC-0012 (Clock)** — desbloqueia 1 outro RFC
3. **Escrever RFC-0006 (Security)** — obrigatório para produção
4. **Adicionar feature bits 18–46 ao RFC-0001** — reserva os bits
5. **Adicionar opcodes `0xC0–0xFE` à spec ISA v2.0** — reserva a faixa
6. **Adicionar regiões `0x10–0x17` à spec** — reserva as regiões
7. **Rodar red team trimestral** — institucionalizar o meta-processo

Se você quiser, o próximo passo natural é eu escrever o **RFC-0003 (Storage Tiering)** completo, no mesmo formato do RFC-0002, para desbloquear o caminho crítico. Quer?

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
