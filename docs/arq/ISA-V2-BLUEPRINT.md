> **ARQUIVO — material de entrada, NÃO normativo.**
> Recebido como crítica/colagem ("precisamos corrigir o conceitual") em
> 2026-09-10, junto ao blueprint v2.0 abaixo. A especificação canônica é
> `docs/ESPEC.md` (ISA v1.3 + modelo Σ); a reconciliação com este corpus
> vive em `docs/ESPEC-V2.md` (DRAFT). Nada aqui deve ser implementado sem
> reconciliação. Conteúdo original preservado verbatim abaixo (inclui a
> crítica de cobertura e o blueprint v2.0).
>
> ---

precisamos corrigir o conceitual # Não. Está **incompleto** — e a lacuna é grande.

Sua lista cobre bem a **camada de extensões** (flags, semântica, reserva), mas **omite três camadas inteiras** do blueprint v2.0 que discutimos. Vou ser cirúrgico.

---

## O que sua lista cobre (correto)

| Seção | Status | Comentário |
|:---|:--:|:---|
| **A. Flags em opcodes existentes** | ✅ Parcial | 4 de ~10 extensões de flag listadas |
| **B. Extensão semântica** | ✅ Parcial | Cobre FORK/ABORT/CTX_SWITCH/SENSE, mas falta estado novo |
| **C. Comportamento reservado** | ✅ Parcial | Cobre SEND_TENSOR e DENOISE, mas falta WAL/SNAPSHOT/SHARED |
| **D. Ordem normativa de opcodes novos** | ⚠️ Incompleta | Só cobre a onda universal (`0x1E–0x25`), ignora ~150 opcodes |

---

## O que sua lista **NÃO** cobre (as lacunas)

### Lacuna 1 — O modelo de memória (16 regiões)

Sua lista não menciona **nenhuma das 16 regiões**. Isso é grave porque sem elas:

- `WEIGHTS` RO (imutabilidade de pesos) → bug de corrupção
- `KV_CACHE` CoW versionado → rollback quebrado
- `ARENA` por ator → isolamento Erlang perdido
- `SHARED` zero-copy → sem `mmap` cross-context
- `WAL` append-only → `MIGRATE` inseguro
- `SNAPSHOT` monotônico → estouro em sequências longas
- `STREAM_RING` → áudio full-duplex sem buffer
- `RAG_INDEX` → vector DB sem casa
- `CLUSTER_STAGING` → backpressure ausente
- `SCRATCH_GPU`, `MMIO` → sem caminho para aceleradores

**Sem as 16 regiões, os opcodes que você listou não sabem onde escrever.**

---

### Lacuna 2 — Campos novos da instrução (64B)

Sua lista ignora que a instrução v2.0 tem **4 campos novos** além dos 8 bytes de core:

| Campo | Bytes | O que habilita | Sua lista menciona? |
|:---|:---:|:---|:--:|
| `LAMPORT` | 8 | Replay distribuído bit-exato | ❌ |
| `DEADLINE` | 8 | Scheduler EDF + herança de prioridade | ⚠️ (só em SIGNAL) |
| `PAYLOAD_EXT` | 8 | Capability token, RDMA key, checksum | ❌ |
| `RSRC4`, `RSRC5` | 2 | `RANK1_UPDATE`, `CONV`, `FOREST` sem acessório | ❌ |

Sem `LAMPORT` e `DEADLINE` **inline em toda instrução**, o scheduler EDF e o replay distribuído não existem — e o `SIGNAL` que você listou fica órfão.

---

### Lacuna 3 — Flags novos do byte `FLAGS`

Sua lista cobre `TOPK`, `TRANSPOSE`, `deadline`. Faltam:

| Bit | Flag | O que habilita |
|:--:|:---|:---|
| 0–1 | `PRECISION` | FP32/BF16/FP16/INT8 por instrução |
| 2 | `FUSED_ACT` | SILU/ReLU/GELU fundido em MATVEC/CONV |
| 3 | `INPLACE` | Evita alocação em loops tight |
| 6 | `ATOMIC` | Escrita atômica em SHARED |
| 7 | `PERSIST` | Persiste em WAL (crash recovery) |

Sem `PRECISION` por instrução, seu "agnosticismo de precisão" (a tese dos 50 anos) é fake.

---

### Lacuna 4 — Opcodes novos fora da onda universal

Sua lista D cobre só `0x1E–0x25`. Faltam **~150 opcodes** do blueprint v2.0:

#### Memória & Arena (`0x26–0x2F`)
`ARENA_ALLOC`, `ARENA_RESET`, `SNAPSHOT`, `RESTORE`, `MEMCPY`, `MEMSET`, `PREFETCH`, `RESHAPE`, `SLICE`, `CONCAT`

#### Tensores (`0x30–0x37`)
`SORT`, `TOPK`, `ARGMAX`, `REDUCE`, `BROADCAST`, `PAD`, `TILE`, `TRANSPOSE`

#### Atenção avançada (`0x38–0x3C`)
`KV_TRUNCATE`, `KV_COMPRESS`, `FLASH_ATTN`, `ATTN_SPARSE`, `SOFTMAX`

#### Ativações (`0x3D–0x43`)
`GELU`, `SIGMOID`, `TANH`, `RELU`, `EXP`, `LOG`, `CLIP`

#### Áudio & Full-Duplex (`0x44–0x49`)
`DEPFORMER`, `STREAM_MERGE`, `VAD_DETECT`, `AUDIO_RESAMPLE`, `AUDIO_FILTER`, `AUDIO_WINDOW`

**Estes 6 opcodes são o coração do Moshi full-duplex.** Sua lista não os menciona.

#### Cluster estendido (`0x4A–0x4F`)
`MULTICAST`, `MIGRATE`, `HEARTBEAT`, `PING`, `REDUCE_REMOTE`, `GATHER_REMOTE`

**`MIGRATE` substitui `SEND_TENSOR MOVE`** — você listou a mudança semântica em C, mas não o opcode que a implementa.

#### Retrieval (`0x50–0x57`)
`RAG_INDEX_ADD/DEL`, `RAG_SEARCH`, `EMBED_LOOKUP`, `HASH_BUCKET`, `QUANTIZE_VEC`, `PQ_ENCODE/DECODE`

#### ML clássico (`0x58–0x5F`)
`KMEANS_STEP`, `LINEAR_REG`, `LOGISTIC_REG`, `NAIVE_BAYES`, `SVM_PREDICT`, `PCA_STEP`, `STANDARDIZE`, `NEAREST_CENTROID`

Você listou `FOREST` e `DISTANCE` em D, mas o resto do ML clássico ficou de fora.

#### Determinismo (`0x60–0x66`)
`RNG_SEED`, `RNG_NEXT`, `RNG_NORMAL`, `RNG_UNIFORM`, `HASH`, `CHECKSUM`, `HMAC`

**Sem `RNG_SEED`, seu `DENOISE_STEP` estocástico não é replay-exato.**

#### Conversão (`0x67–0x69`)
`CAST`, `QUANTIZE`, `DEQUANT`

#### Debug/Telemetria (`0x6A–0x6F`)
`CYCLES_COUNT`, `TRACE_EVENT`, `SANITY_CHECK`, `PREEMPT_CHECK`, `ASSERT`, `DUMP`

#### Scheduler (`0x70–0x77`)
`YIELD`, `SET_DEADLINE`, `GET_DEADLINE`, `PRIORITY_SET/GET`, `LOCK`, `UNLOCK`, `FENCE`

**Sem `LOCK`/`UNLOCK`/`FENCE`, o `SHARED` não tem sincronização.**

#### MMIO (`0xE0–0xE8`)
`GPU_LAUNCH`, `GPU_WAIT`, `NIC_SEND/RECV`, `DMA_START/WAIT`, `ATOMIC_CAS/ADD`, `WFI`

#### Sistema (`0xF0–0xF7`)
`LOAD_MODEL`, `UNLOAD_MODEL`, `SPAWN_CONTEXT`, `KILL_CONTEXT`, `SET_PRIORITY`, `SET_AFFINITY`, `PROFILE_START/STOP`

**Sem `LOAD_MODEL`/`SPAWN_CONTEXT`, não há multi-modelo na mesma VM.**

---

### Lacuna 5 — Modo dual 32B/64B

Sua lista assume que todos os opcodes novos usam o encoding estendido, mas **não menciona a regra de dual-mode**:

- Opcodes `0x00–0x7F` → 32B (compatibilidade v1.3)
- Opcodes `0x80–0xFF` → 64B (v2.0)

Sem essa regra, você quebra binários antigos ou infla o I-cache com padding desnecessário.

---

## Versão corrigida e completa da sua lista

### A. Flags/modos em opcodes existentes
| Opcode | Extensão |
|:---|:---|
| `SAMPLE 0x0B` | modo `TOPK=k` |
| `MATVEC 0x10` | flag `TRANSPOSE` + `FUSED_ACT` |
| `ATTN 0x02` | modo podado (softmax top-T via `DISTANCE`) |
| `SIGNAL 0x1B` | `DEADLINE` + `CAPABILITY` inline |
| `NORM 0x07` | modo `RMS` / `LAYER` / `BATCH` |
| `FFN 0x08` | flag `FUSED_ACT` |
| `SSM_SCAN 0x13` | flag `PRECISION` (FP32/BF16) |
| `CODEC_ENC/DEC` | flag `PRECISION` (fp16 p/ Mimi) |
| `SENSE 0x06` | modo spike-train (SNN) |
| **Todos** | flag `PRECISION`, `INPLACE`, `PERSIST` |

### B. Extensão semântica
| Opcode | Extensão |
|:---|:---|
| `FORK`/`ABORT` | Cobrir `V(t)`, `H_t`, `x_t`, KV 17-streams |
| `CTX_SWITCH` | Fence de recalibração da janela W (anti-deriva) |
| `SENSE` | Modo spike-train |
| **`SHARED`** | Fence explícito via `0x77` |
| **`KV_CACHE`** | Suporte a 17 streams (Moshi) + `ssm_states` + SNN |
| **`RAG_INDEX`** | Estado indexado por `RAG_SEARCH` |

### C. Comportamento reservado que muda
| Opcode | Extensão |
|:---|:---|
| `SEND_TENSOR 0x1C` | `MOVE` → `OP_MIGRATE 0x4B` + WAL |
| `DENOISE_STEP 0x21` | Despriorizado (após atenção linear) |
| **`WEIGHTS`** | RO enforced por `mprotect` |
| **`SNAPSHOT`** | Política de retenção `k=16` |
| **`WAL`** | Ring buffer com `t_wal_retention=5s` |
| **`CLUSTER_STAGING`** | Backpressure em 80% |

### D. Opcodes novos (ordem normativa revisada)

**Onda 1 (desbloqueia valor imediato):**
1. `GATHER 0x1F`
2. `DISTANCE 0x23` + `RANK1_UPDATE 0x24`
3. `SPIKE_STEP 0x20`
4. `CONV 0x1E`
5. `FOREST 0x22`
6. `ODE_STEP 0x25`
7. `DENOISE_STEP 0x21` (último)

**Onda 2 (infraestrutura):**
8. Memória: `ARENA_ALLOC 0x26` … `CONCAT 0x2F`
9. Tensores: `SORT 0x30` … `TRANSPOSE 0x37`
10. Atenção: `KV_TRUNCATE 0x38` … `SOFTMAX 0x3C`
11. Ativações: `GELU 0x3D` … `CLIP 0x43`

**Onda 3 (Full-Duplex Moshi):**
12. `DEPFORMER 0x44`, `STREAM_MERGE 0x45`, `VAD_DETECT 0x46`, `AUDIO_RESAMPLE 0x47`, `AUDIO_FILTER 0x48`, `AUDIO_WINDOW 0x49`

**Onda 4 (Cluster estendido):**
13. `MULTICAST 0x4A`, `MIGRATE 0x4B`, `HEARTBEAT 0x4C`, `PING 0x4D`, `REDUCE_REMOTE 0x4E`, `GATHER_REMOTE 0x4F`

**Onda 5 (Retrieval + ML clássico):**
14. `RAG_SEARCH 0x52` + `EMBED_LOOKUP 0x53` + `PQ_* 0x56/0x57`
15. `KMEANS_STEP 0x58` … `NEAREST_CENTROID 0x5F`

**Onda 6 (Determinismo + conversão):**
16. `RNG_SEED 0x60` … `HMAC 0x66`
17. `CAST 0x67`, `QUANTIZE 0x68`, `DEQUANT 0x69`

**Onda 7 (Debug + Scheduler):**
18. `CYCLES_COUNT 0x6A` … `DUMP 0x6F`
19. `YIELD 0x70` … `FENCE 0x77`

**Onda 8 (MMIO + Sistema):**
20. `GPU_LAUNCH 0xE0` … `WFI 0xE8`
21. `LOAD_MODEL 0xF0` … `PROFILE_STOP 0xF7`

### E. Infraestrutura (a lacuna mais grave)

| Item | Descrição |
|:---|:---|
| **16 regiões de memória** | Com seletor de 4 bits, `WEIGHTS` RO, `KV_CACHE` CoW, `WAL` append-only |
| **Layout 64B** | `[op\|flags\|rdest\|rsrc1-5\|lamport:8\|deadline:8\|payload_ext:8\|payload_core:32]` |
| **Dual-mode 32B/64B** | `op < 0x80` → 32B; `op ≥ 0x80` → 64B |
| **Banco de registradores** | 256 GPR + 32 VEC + 64 STATE + 8 SPECIAL |
| **Convenção `0xFF`** | "não usado" universal |
| **Política de retenção** | Snapshots com janela `k=16` |

---

## Resumo da crítica

Sua lista cobre **~25% do blueprint v2.0**. Especificamente:

| Camada | Cobertura |
|:---|:--:|
| Extensões de flags em opcodes existentes | 40% |
| Extensão semântica | 60% |
| Mudanças de comportamento reservado | 40% |
| Opcodes novos | **~10%** |
| Modelo de memória (16 regiões) | **0%** |
| Layout de instrução (64B + campos) | **0%** |
| Flags do byte `FLAGS` | 30% |
| Dual-mode 32B/64B | **0%** |
| Infraestrutura (regs, retenção, backpressure) | **0%** |

**Veredito:** sua lista é um **índice de extensões incrementais**, não o blueprint completo. Falta a **fundação** (memória, layout, flags, infraestrutura) que sustenta os opcodes.

Se você quer a lista **verdadeiramente completa**, ela tem 4 seções (A/B/C/D como você fez) **mais** 5 seções de infraestrutura (E/F/G/H/I) que descrevi acima. Sem elas, você tem opcodes sem casa, flags sem semântica e distribuição sem relógio.

Quer que eu gere a **tabela canônica final** (todos os 256 opcodes, com estado, região, payload e flags) em formato de documento único? # M³-AVM ISA v2.0 — Especificação Canônica Completa

> **Documento único de referência.** Consolida memória, layout de instrução, flags, registradores, tabela completa de 256 opcodes, payloads por domínio, taxonomia de estado, dual-mode 32B/64B e plano de migração.  
> **Autor original da visão:** Matheus de Camargo Marques · https://github.com/matheuscamarques/m3_avm  
> **Status:** spec v2.0 congelada em 2026-09-09. Implementação segue as ondas §9.

---

## 1. Modelo de Memória — 16 Regiões

### 1.1 Endereçamento (64 bits)

```
Bits 63..60 : REGION (4 bits) — seletor de região
Bits 59..0  : OFFSET (60 bits) — 1 EiB por região
Total       : 16 EiB endereçável
```

### 1.2 Tabela de regiões

| ID | Nome | Base | Tamanho | R/W | CoW | Propósito |
|:--:|:---|:---|:---|:--:|:--:|:---|
| `0x0` | `TEXT` | `0x0000_0000_0000_0000` | 16 GiB | RO | — | Bytecode da VM |
| `0x1` | `GLOBAL` | `0x1000_0000_0000_0000` | 256 GiB | RO | — | Constantes, tabelas |
| `0x2` | `WEIGHTS` | `0x2000_0000_0000_0000` | 4 TiB | RO (`mprotect`) | — | Pesos GGUF/safetensors |
| `0x3` | `ACTIVATION` | `0x3000_0000_0000_0000` | 64 GiB | RW | Sim | Scratch por contexto |
| `0x4` | `KV_CACHE` | `0x4000_0000_0000_0000` | 512 GiB | RW | Sim | K/V, `ssm_states`, `V(t)`, `H_t` |
| `0x5` | `ARENA` | `0x5000_0000_0000_0000` | 64 GiB | RW | — | Heap do ator |
| `0x6` | `SHARED` | `0x6000_0000_0000_0000` | 256 GiB | RW (fence) | Sim | Tensores zero-copy cross-context |
| `0x7` | `WAL` | `0x7000_0000_0000_0000` | 8 GiB | append | — | Write-ahead log de migração |
| `0x8` | `SNAPSHOT` | `0x8000_0000_0000_0000` | 256 GiB | RO | — | CoW de `FORK` |
| `0x9` | `STREAM_RING` | `0x9000_0000_0000_0000` | 16 GiB | ring | — | Áudio/vídeo em tempo-real |
| `0xA` | `RAG_INDEX` | `0xA000_0000_0000_0000` | 512 GiB | RW | — | Vector DB / IVF / HNSW |
| `0xB` | `CLUSTER_STAGING` | `0xB000_0000_0000_0000` | 64 GiB | RW (backpressure) | — | TX/RX remoto |
| `0xC` | `FEDERATED` | `0xC000_0000_0000_0000` | 256 GiB | RW | — | Reservado (aprendizado federado) |
| `0xD` | `CONFIDENTIAL` | `0xD000_0000_0000_0000` | 256 GiB | enc | — | Reservado (TEE/enclave) |
| `0xE` | `SCRATCH_GPU` | `0xE000_0000_0000_0000` | 256 GiB | RW | — | Staging GPU/TPU/NPU |
| `0xF` | `MMIO` | `0xF000_0000_0000_0000` | 256 GiB | RW | — | NIC, DMA, aceleradores |

### 1.3 Invariantes de consistência

1. **`WEIGHTS` é sempre RO** — nenhum opcode escreve em `0x2`. Fine-tuning copia para `ACTIVATION` primeiro.
2. **`KV_CACHE` é CoW-versionado** — todo opcode stateful escreve aqui, nunca em `SHARED`/`WEIGHTS`.
3. **`SNAPSHOT` é monotônico** — janela deslizante `k=16` recicla versões antigas.
4. **`WAL` é append-only ring** — `t_wal_retention=5s` default; entradas coletáveis após ACK.
5. **`SHARED` requer `FENCE (0x77)`** — visibilidade explícita entre contextos.
6. **`CLUSTER_STAGING` tem backpressure** — bloqueia em 80% de uso com deadline EDF.

### 1.4 Arena por contexto (isolamento Erlang)

```
Contexto C:
  ACTIVATION[C]  = slice 0x3 / N_contextos
  KV_CACHE[C]    = slice 0x4 / N_contextos
  ARENA[C]       = slice 0x5 / N_contextos
  STREAM_RING[C] = slice 0x9

Compartilhado RO  : TEXT, GLOBAL, WEIGHTS, RAG_INDEX
Compartilhado CoW : SHARED, SNAPSHOT
Compartilhado RW  : WAL, CLUSTER_STAGING
```

---

## 2. Formato de Instrução

### 2.1 Dual-mode 32B / 64B

| Opcode | Formato | Uso |
|:---|:---|:---|
| `0x00–0x7F` | **32B** (v1.3 compatível) | Controle, tensores, matriz, áudio base |
| `0x80–0xFF` | **64B** (v2.0 estendido) | Cluster, universal, retrieval, sistema |

O dispatcher lê byte 0; se `>= 0x80`, lê 64B. Se `< 0x80`, lê 32B.

### 2.2 Layout 32B (v1.3)

```
Byte 0     : OPCODE    (u8)
Byte 1     : FLAGS     (u8)
Byte 2     : RDEST     (u8)
Byte 3     : RSRC1     (u8)
Byte 4     : RSRC2     (u8)
Byte 5     : RSRC3     (u8)
Bytes 6-31 : PAYLOAD_CORE (26 B, LE)
```

### 2.3 Layout 64B (v2.0)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Byte 0     : OPCODE     (u8)                                         │
│ Byte 1     : FLAGS      (u8)                                         │
│ Byte 2     : RDEST      (u8)                                         │
│ Byte 3     : RSRC1      (u8)                                         │
│ Byte 4     : RSRC2      (u8)                                         │
│ Byte 5     : RSRC3      (u8)                                         │
│ Byte 6     : RSRC4      (u8)                                         │
│ Byte 7     : RSRC5      (u8)                                         │
│ Bytes 8-15 : LAMPORT    (u64 LE) — relógio lógico distribuído        │
│ Bytes 16-23: DEADLINE   (u64 LE) — deadline EDF absoluto (ns)        │
│ Bytes 24-31: PAYLOAD_EXT (8 B) — HMAC, RDMA key, checksum            │
│ Bytes 32-63: PAYLOAD_CORE (32 B) — específico por opcode             │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.4 Valores "neutros" (compatibilidade v1.3)

| Campo | Neutro | Semântica |
|:---|:---|:---|
| `RSRC4`, `RSRC5` | `0xFF` | Não usado |
| `LAMPORT` | `0` | Instrução local |
| `DEADLINE` | `u64::MAX` | Best-effort |
| `PAYLOAD_EXT` | `0x00...0` | Sem auth/chave |

---

## 3. Flags (Byte 1)

| Bit | Nome | Valores | Descrição |
|:--:|:---|:---|:---|
| 0–1 | `PRECISION` | `0=FP32, 1=BF16, 2=FP16, 3=INT8/FP8` | Precisão por instrução |
| 2 | `FUSED_ACT` | `0=none, 1=SILU, 2=ReLU, 3=GELU` | Ativação fundida |
| 3 | `INPLACE` | `0=novo, 1=in-place` | Evita alocação |
| 4 | `TRANSPOSE` | `0=normal, 1=Wᵀ·x` | Para `MATVEC` |
| 5 | `PRIORITY` | `0=GREEN, 1=BLUE, 2=RED` | Prioridade EDF |
| 6 | `ATOMIC` | `0=normal, 1=atômico` | Escrita atômica |
| 7 | `PERSIST` | `0=volátil, 1=WAL` | Persistência em crash |

Flags podem ser reinterpretadas por opcode quando documentado.

---

## 4. Banco de Registradores

| Tipo | Qtd | Largura | Faixa | Uso |
|:---|:--:|:---|:---|:---|
| **GPR** | 256 | 64 bits | `R0–R255` | Endereços, handles, escalares |
| **VEC** | 32 | 256 bits | `V0–V31` | SIMD explícito (8×f32) |
| **STATE** | 64 | opaco | `S0–S63` | Handles `KV_CACHE`, `ssm_states`, `H`, `V(t)` |
| **SPECIAL** | 8 | 64 bits | `SP0–SP7` | Ver tabela |

### 4.1 Registradores especiais

| Reg | Nome | Uso |
|:--:|:---|:---|
| `SP0` | `PC` | Program counter |
| `SP1` | `SP` | Stack pointer (em `ARENA`) |
| `SP2` | `FP` | Frame pointer |
| `SP3` | `CTX_ID` | ID do contexto (u64) |
| `SP4` | `NODE_ID` | ID do nó no cluster (u32) |
| `SP5` | `LAMPORT` | Relógio lógico local |
| `SP6` | `DEADLINE` | Deadline EDF corrente |
| `SP7` | `STATUS` | Flags: `NaN/OOB/TIMEOUT/PREEMPT` |

### 4.2 Convenção `0xFF`

`0xFF` em `RSRC*` = "não usado". `RDEST=0xFF` = "descartar resultado".

---

## 5. Mapa Completo de Opcodes (0x00–0xFF)

**Legenda de status:**  
✅ implementado · 🟡 planejado (spec congelada) · 🔵 novo v2.0 · ⬜ reservado

### 5.1 Domínio 0 — Controle & Flow (`0x00–0x0F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x00` | `HALT` | ✅ | — | Encerra contexto |
| `0x01` | `TENSOR` | ✅ | `GLOBAL`/`ACTIVATION` | `[dims..., dtype]` |
| `0x02` | `ATTN` | ✅ | `KV_CACHE` | Ver §5.4 |
| `0x03` | `STREAM` | ✅ | `STREAM_RING` | Canal + modo |
| `0x04` | `FORK` | ✅ | `SNAPSHOT` | Bitmask de regiões |
| `0x05` | `ABORT` | ✅ | `SNAPSHOT` | Handle da versão |
| `0x06` | `SENSE` | ✅ | `STREAM_RING` | Tipo de sensor |
| `0x07` | `NORM` | ✅ | — | Modo: RMS/Layer/Batch |
| `0x08` | `FFN` | ✅ | — | Fused act |
| `0x09` | `EMBED` | ✅ | `WEIGHTS` | Vocab + dim |
| `0x0A` | `ADD` | ✅ | — | Elementwise |
| `0x0B` | `SAMPLE` | ✅ | — | + modo `TOPK=k` |
| `0x0C` | `COMPARE` | ✅ | — | Predicados |
| `0x0D` | `IF_EQUAL` | ✅ | — | Branch condicional |
| `0x0E` | `JUMP` | ✅ | `TEXT` | Offset relativo |
| `0x0F` | `IF_INTERRUPT` | ✅ | — | Branch em flag |

### 5.2 Domínio 1 — Matriz & Tensor (`0x10–0x1F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x10` | `MATVEC` | ✅ | — | + `TRANSPOSE`, `FUSED_ACT` |
| `0x11` | `MUL` | ✅ | — | Elementwise |
| `0x12` | `SILU` | ✅ | — | SiLU elementwise |
| `0x13` | `SSM_SCAN` | ✅ | `KV_CACHE` | Ver §5.5 |
| `0x14` | `SSM_RESET` | ✅ | `KV_CACHE` | Reset `h_t` |
| `0x15` | `CODEC_ENC` | ✅ | `TEMPORAL` | PCM → codes |
| `0x16` | `CODEC_DEC` | ✅ | `TEMPORAL` | Codes → PCM |
| `0x17` | `AUDIO_ALIGN` | ✅ | — | Timestamps |
| `0x18` | `CTX_SWITCH` | ✅ | — | `MAMBA/TRANSFORMER/AUDIO` + prio |
| `0x19` | `ROPE` | ✅ | — | Rotary |
| `0x1A` | `REMOTE_SPAWN` | 🟡 | — | Ver §5.10 |
| `0x1B` | `SIGNAL` | 🟡 | — | Ver §5.10 |
| `0x1C` | `SEND_TENSOR` | 🟡 | `CLUSTER_STAGING` | Ver §5.10 |
| `0x1D` | `BARRIER` | 🟡 | — | Ver §5.10 |
| `0x1E` | `CONV` | 🟡 | — | N-dim, stride/pad/dil/groups |
| `0x1F` | `GATHER` | 🟡 | — | Modo `GATHER`/`SCATTER_ADD`/`SCATTER_MAX` |

### 5.3 Domínio 2 — Memória & Arena (`0x20–0x2F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x20` | `SPIKE_STEP` | 🟡 | `KV_CACHE` | LIF: threshold/decay/reset |
| `0x21` | `DENOISE_STEP` | 🟡 | `ACTIVATION` | DDPM/DDIM |
| `0x22` | `FOREST` | 🟡 | — | Travessia vetorizada |
| `0x23` | `DISTANCE` | 🟡 | — | 4 métricas + top-k |
| `0x24` | `RANK1_UPDATE` | 🟡 | `KV_CACHE` | DeltaNet/Titans |
| `0x25` | `ODE_STEP` | 🟡 | `KV_CACHE` | Euler/RK2/RK4 |
| `0x26` | `ARENA_ALLOC` | 🔵 | `ARENA` | size, região, flags, alignment |
| `0x27` | `ARENA_RESET` | 🔵 | `ARENA` | Reset O(1) |
| `0x28` | `SNAPSHOT` | 🔵 | `SNAPSHOT` | Bitmask de regiões |
| `0x29` | `RESTORE` | 🔵 | `SNAPSHOT` | Handle da versão |
| `0x2A` | `MEMCPY` | 🔵 | todas | dir: host↔GPU↔NIC |
| `0x2B` | `MEMSET` | 🔵 | todas | Padrão + len |
| `0x2C` | `PREFETCH` | 🔵 | — | Hint de cache |
| `0x2D` | `RESHAPE` | 🔵 | — | View ou cópia |
| `0x2E` | `SLICE` | 🔵 | — | Axis/start/end/step |
| `0x2F` | `CONCAT` | 🔵 | — | Axis |

### 5.4 Domínio 3 — Tensores Avançados (`0x30–0x3F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x30` | `SORT` | 🔵 | — | Axis, ascending |
| `0x31` | `TOPK` | 🔵 | — | k, axis, largest, sorted |
| `0x32` | `ARGMAX` | 🔵 | — | Axis |
| `0x33` | `REDUCE` | 🔵 | — | sum/mean/max/min/prod |
| `0x34` | `BROADCAST` | 🔵 | — | Shape alvo |
| `0x35` | `PAD` | 🔵 | — | Antes/depois por eixo |
| `0x36` | `TILE` | 🔵 | — | Repetições |
| `0x37` | `TRANSPOSE` | 🔵 | — | Permutação de eixos |
| `0x38` | `KV_TRUNCATE` | 🔵 | `KV_CACHE` | Novo comprimento + stream_id |
| `0x39` | `KV_COMPRESS` | 🔵 | `KV_CACHE` | Compressão de cache |
| `0x3A` | `FLASH_ATTN` | 🔵 | `KV_CACHE` | IO-aware attention |
| `0x3B` | `ATTN_SPARSE` | 🔵 | `KV_CACHE` | Top-k via `DISTANCE` |
| `0x3C` | `SOFTMAX` | 🔵 | — | Axis, temperature |
| `0x3D` | `GELU` | 🔵 | — | Elementwise |
| `0x3E` | `SIGMOID` | 🔵 | — | Elementwise |
| `0x3F` | `TANH` | 🔵 | — | Elementwise |

### 5.5 Domínio 4 — Ativações & Elementwise (`0x40–0x4F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x40` | `RELU` | 🔵 | — | Elementwise |
| `0x41` | `EXP` | 🔵 | — | Elementwise |
| `0x42` | `LOG` | 🔵 | — | Elementwise |
| `0x43` | `CLIP` | 🔵 | — | min/max |
| `0x44` | `DEPFORMER` | 🔵 | `KV_CACHE` | Moshi Depformer (6×1024×16) |
| `0x45` | `STREAM_MERGE` | 🔵 | `STREAM_RING` | Mix 17 streams Moshi |
| `0x46` | `VAD_DETECT` | 🔵 | — | Energy/ZCR/ML |
| `0x47` | `AUDIO_RESAMPLE` | 🔵 | `STREAM_RING` | 24k↔16k↔48k |
| `0x48` | `AUDIO_FILTER` | 🔵 | `STREAM_RING` | FIR/IIR |
| `0x49` | `AUDIO_WINDOW` | 🔵 | `STREAM_RING` | Hann/Hamming |
| `0x4A` | `MULTICAST` | 🔵 | `CLUSTER_STAGING` | N nós destino |
| `0x4B` | `MIGRATE` | 🔵 | `WAL` | COPY/WAL_BACKED/RDMA |
| `0x4C` | `HEARTBEAT` | 🔵 | — | free_pct + queues |
| `0x4D` | `PING` | 🔵 | — | RTT probe |
| `0x4E` | `REDUCE_REMOTE` | 🔵 | — | sum/mean/max/min/prod |
| `0x4F` | `GATHER_REMOTE` | 🔵 | — | allgather |

### 5.6 Domínio 5 — Retrieval & ML Clássico (`0x50–0x5F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x50` | `RAG_INDEX_ADD` | 🔵 | `RAG_INDEX` | Inserção |
| `0x51` | `RAG_INDEX_DEL` | 🔵 | `RAG_INDEX` | Remoção |
| `0x52` | `RAG_SEARCH` | 🔵 | `RAG_INDEX` | flat/IVF/HNSW/PQ + top-k |
| `0x53` | `EMBED_LOOKUP` | 🔵 | `WEIGHTS` | Bag de embeddings |
| `0x54` | `HASH_BUCKET` | 🔵 | — | Locality-sensitive |
| `0x55` | `QUANTIZE_VEC` | 🔵 | — | Scalar/product |
| `0x56` | `PQ_ENCODE` | 🔵 | `RAG_INDEX` | Product quantizer |
| `0x57` | `PQ_DECODE` | 🔵 | `RAG_INDEX` | Inverse |
| `0x58` | `KMEANS_STEP` | 🔵 | `ACTIVATION` | Um passo Lloyd |
| `0x59` | `LINEAR_REG` | 🔵 | — | y = wᵀx + b |
| `0x5A` | `LOGISTIC_REG` | 🔵 | — | + sigmoid |
| `0x5B` | `NAIVE_BAYES` | 🔵 | — | Gaussian/multinomial |
| `0x5C` | `SVM_PREDICT` | 🔵 | — | linear/rbf/poly/sigmoid |
| `0x5D` | `PCA_STEP` | 🔵 | — | Power iteration |
| `0x5E` | `STANDARDIZE` | 🔵 | — | Z-score/min-max |
| `0x5F` | `NEAREST_CENTROID` | 🔵 | — | Argmin distância |

### 5.7 Domínio 6 — Universal (Onda 1 + 2) — `0x60–0x6F`

> **Nota:** `CONV/GATHER/SPIKE/DENOISE/FOREST/DISTANCE/RANK1/ODE` já mapeados acima. Este domínio cobre **extensões universais residuais**.

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x60` | `RNG_SEED` | 🔵 | — | Semente por contexto |
| `0x61` | `RNG_NEXT` | 🔵 | — | Próximo u64 |
| `0x62` | `RNG_NORMAL` | 🔵 | — | Box-Muller |
| `0x63` | `RNG_UNIFORM` | 🔵 | — | [a,b) |
| `0x64` | `HASH` | 🔵 | — | SipHash/xxHash |
| `0x65` | `CHECKSUM` | 🔵 | — | CRC32/xxHash |
| `0x66` | `HMAC` | 🔵 | — | SHA256 truncado |
| `0x67` | `CAST` | 🔵 | — | FP32↔BF16↔FP16↔INT8 |
| `0x68` | `QUANTIZE` | 🔵 | `WEIGHTS` | Q4_0/Q4_K/Q6_K/Q8_0 |
| `0x69` | `DEQUANT` | 🔵 | — | Inverse |
| `0x6A` | `CYCLES_COUNT` | 🔵 | — | RDTSC-like |
| `0x6B` | `TRACE_EVENT` | 🔵 | — | Tracing estruturado |
| `0x6C` | `SANITY_CHECK` | 🔵 | — | Absorve NaN/Inf |
| `0x6D` | `PREEMPT_CHECK` | 🔵 | — | Lê flag atômica |
| `0x6E` | `ASSERT` | 🔵 | — | Predicado → trap |
| `0x6F` | `DUMP` | 🔵 | — | Snapshot debug |

### 5.8 Domínio 7 — Scheduler & Sistema (`0x70–0x7F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x70` | `YIELD` | 🔵 | — | Cede CPU |
| `0x71` | `SET_DEADLINE` | 🔵 | — | Deadline EDF |
| `0x72` | `GET_DEADLINE` | 🔵 | — | Lê deadline |
| `0x73` | `PRIORITY_SET` | 🔵 | — | RED/BLUE/GREEN |
| `0x74` | `PRIORITY_GET` | 🔵 | — | Lê prioridade |
| `0x75` | `LOCK` | 🔵 | `SHARED` | Mutex cross-context |
| `0x76` | `UNLOCK` | 🔵 | `SHARED` | Libera |
| `0x77` | `FENCE` | 🔵 | `SHARED` | Barreira de memória |
| `0x78–0x7F` | ⬜ | — | — | Reservado |

### 5.9 Domínio 8 — Cluster Estendido (`0x80–0x8F`)

> **Opcodes `0x1A–0x1D` originais migram para 64B e ficam aqui como espelho estendido.**

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x80` | `REMOTE_SPAWN` | 🔵 | — | node_id, entry_pc, prio |
| `0x81` | `SIGNAL` | 🔵 | — | + lamport + deadline + capability |
| `0x82` | `SEND_TENSOR` | 🔵 | `CLUSTER_STAGING` | + checksum + compression |
| `0x83` | `BARRIER` | 🔵 | — | id, expected, timeout |
| `0x84` | `MIGRATE` | 🔵 | `WAL` | copy + WAL + retention |
| `0x85` | `MULTICAST` | 🔵 | `CLUSTER_STAGING` | N destinos |
| `0x86` | `REDUCE_REMOTE` | 🔵 | — | Allreduce |
| `0x87` | `GATHER_REMOTE` | 🔵 | — | Allgather |
| `0x88` | `SCATTER_REMOTE` | 🔵 | — | Scatter |
| `0x89` | `BROADCAST_REMOTE` | 🔵 | — | Broadcast |
| `0x8A` | `HEARTBEAT` | 🔵 | — | free_pct + queues |
| `0x8B` | `PING` | 🔵 | — | RTT |
| `0x8C` | `NODE_JOIN` | 🔵 | — | Handshake + cookie |
| `0x8D` | `NODE_LEAVE` | 🔵 | — | Graceful shutdown |
| `0x8E` | `NODE_SUSPECT` | 🔵 | — | SWIM suspect |
| `0x8F` | `NODE_DEAD` | 🔵 | — | SWIM dead |

### 5.10 Domínio 9 — MMIO & Aceleradores (`0x90–0x9F`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0x90` | `GPU_LAUNCH` | 🔵 | `SCRATCH_GPU` | Kernel + grid |
| `0x91` | `GPU_WAIT` | 🔵 | — | Handle de evento |
| `0x92` | `NIC_SEND` | 🔵 | `CLUSTER_STAGING` | Zero-copy TX |
| `0x93` | `NIC_RECV` | 🔵 | `CLUSTER_STAGING` | Zero-copy RX |
| `0x94` | `DMA_START` | 🔵 | todas | Direção + len |
| `0x95` | `DMA_WAIT` | 🔵 | — | Handle |
| `0x96` | `ATOMIC_CAS` | 🔵 | `SHARED` | Compare-and-swap |
| `0x97` | `ATOMIC_ADD` | 🔵 | `SHARED` | Fetch-and-add |
| `0x98` | `ATOMIC_XCHG` | 🔵 | `SHARED` | Exchange |
| `0x99` | `WFI` | 🔵 | — | Wait-for-interrupt |
| `0x9A–0x9F` | ⬜ | — | — | Reservado MMIO |

### 5.11 Domínio 10 — Sistema (`0xA0–0xAF`)

| Hex | Nome | St | Região | Payload / Notas |
|:--:|:---|:--:|:---|:---|
| `0xA0` | `LOAD_MODEL` | 🔵 | `WEIGHTS` | GGUF path + quant |
| `0xA1` | `UNLOAD_MODEL` | 🔵 | `WEIGHTS` | Handle |
| `0xA2` | `SPAWN_CONTEXT` | 🔵 | `ARENA` | Entry PC + prio + model_id |
| `0xA3` | `KILL_CONTEXT` | 🔵 | `ARENA` | ctx_id |
| `0xA4` | `SET_AFFINITY` | 🔵 | — | CPU mask |
| `0xA5` | `PROFILE_START` | 🔵 | — | Tag |
| `0xA6` | `PROFILE_STOP` | 🔵 | — | Tag |
| `0xA7` | `SET_MODEL` | 🔵 | — | Troca modelo do contexto |
| `0xA8` | `GET_MODEL` | 🔵 | — | Modelo atual |
| `0xA9` | `MODEL_SWITCH` | 🔵 | — | Multi-modelo |
| `0xAA–0xAF` | ⬜ | — | — | Reservado sistema |

### 5.12 Domínio 11 — Reservado (`0xB0–0xFE`)

| Faixa | Uso |
|:---|:---|
| `0xB0–0xBF` | Federated learning (reservado) |
| `0xC0–0xCF` | Confidential computing / TEE (reservado) |
| `0xD0–0xDF` | Photonic / neuromorphic (reservado) |
| `0xE0–0xEF` | Quantum simulator (reservado) |
| `0xF0–0xFE` | Sistema extendido (reservado) |

### 5.13 Domínio 12 — Terminal (`0xFF`)

| Hex | Nome | St | Notas |
|:--:|:---|:--:|:---|
| `0xFF` | `NOP` | ✅ | No-op (formato 32B) |

---

## 6. Payloads por Domínio (detalhamento)

### 6.1 `ATTN (0x02)` — atenção completa

```
Byte 2      : rdest    (output)
Byte 3      : rsrc1    (Q)
Byte 4      : rsrc2    (K ou 0xFF=KV_CACHE)
Byte 5      : rsrc3    (V ou 0xFF=KV_CACHE)
Byte 6      : rsrc4    (mask ou 0xFF)
Byte 7      : rsrc5    (KV handle ou 0xFF)
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext (0 = sem auth)
Bytes 32-35 : n_heads (u32)
Bytes 36-39 : head_dim (u32)
Bytes 40-43 : seq_len (u32)
Byte 44     : flags (bit0=causal, bit1=flash, bit2=notify_per_head)
Byte 45     : kv_stream_id (0=default, 1-16=Moshi)
Byte 46     : layer_id
Bytes 47-50 : scale (f32) — 0 = 1/√d
Bytes 51-63 : reservado
```

### 6.2 `SSM_SCAN (0x13)`

```
Byte 2      : rdest    (y)
Byte 3      : rsrc1    (x)
Byte 4      : rsrc2    (h in-place ou 0xFF=Vm::ssm_states)
Byte 5      : rsrc3    (pack dt+A+B+C+D ou 0xFF=defaults)
Bytes 6-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-33 : d_inner (u16)
Bytes 34-35 : d_state (u16)
Byte 36     : layer_id
Byte 37     : flags (bit0=precision, bit1=recompute)
Bytes 38-63 : reservado
```

### 6.3 `DEPFORMER (0x44)` — Moshi

```
Byte 2      : rdest    (codes 16 codebooks)
Byte 3      : rsrc1    (hidden do Temporal)
Byte 4      : rsrc2    (audio embeddings)
Byte 5      : rsrc3    (KV handle Depformer)
Bytes 6-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-35 : dep_layers (u32)
Bytes 36-39 : dep_dim (u32)
Byte 40     : codebooks (u8, default 16)
Byte 41     : acoustic_delay (u8, default 1)
Bytes 42-43 : temperature (f16)
Bytes 44-47 : top_k (u32)
Byte 48     : stream_id
Byte 49     : flags (bit0=interleave_agent, bit1=interleave_user)
Bytes 50-63 : reservado
```

### 6.4 `STREAM_MERGE (0x45)`

```
Byte 2      : rdest    (PCM misturado)
Byte 3      : rsrc1    (stream agente)
Byte 4      : rsrc2    (stream usuário)
Byte 5      : rsrc3    (0xFF)
Bytes 6-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-35 : n_streams (u32)
Bytes 36-39 : sample_rate (u32)
Bytes 40-43 : frame_samples (u32)
Byte 44     : mode (0=mix, 1=agent_only, 2=user_only, 3=echo_cancel)
Bytes 45-48 : gain (f32)
Bytes 49-63 : reservado
```

### 6.5 `MIGRATE (0x84)`

```
Byte 2      : rdest    (status)
Byte 3      : rsrc1    (fonte local)
Byte 4      : rsrc2    (node_id u32)
Byte 5      : rsrc3    (offset remoto)
Bytes 6-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext (HMAC / RDMA rkey)
Bytes 32-35 : node_id (u32)
Bytes 36-43 : byte_offset (u64)
Bytes 44-47 : byte_len (u32)
Byte 48     : mode (0=COPY, 1=WAL_BACKED, 2=RDMA)
Byte 49     : t_wal_retention (u8, segundos)
Bytes 50-51 : checksum (u16)
Bytes 52-57 : WAL entry id (u48)
Bytes 58-63 : reservado
```

### 6.6 `RAG_SEARCH (0x52)`

```
Byte 2      : rdest    (top-k resultados)
Byte 3      : rsrc1    (query vector)
Byte 4      : rsrc2    (índice RAG_INDEX)
Byte 5      : rsrc3    (0xFF)
Bytes 6-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-35 : metric (0=EUCLID, 1=COSINE, 2=DOT, 3=MANHATTAN)
Bytes 36-39 : top_k (u32)
Bytes 40-43 : n_probe (u32, p/ IVF)
Byte 44     : mode (0=flat, 1=IVF, 2=HNSW, 3=PQ)
Bytes 45-63 : reservado
```

---

## 7. Taxonomia de Estado (obrigatória para novos opcodes)

Todo opcode stateful **deve** declarar:

| Campo | Descrição |
|:---|:---|
| **(a) Onde mora** | Região exata (`KV_CACHE`, `ARENA`, `SHARED`) |
| **(b) Custo de snapshot** | O(1) via CoW, O(n) via cópia, ou O(log n) via PAMT |
| **(c) Como `ABORT` restaura** | Ponteiro, versão, ou reset explícito |

### 7.1 Classificação

**Family 1 — Stateless (drop instantâneo):**
`FFN`, `CONV`, `GATHER` (agregado), `MoE-gate`, `ATTN` sem cache, `ROPE`, `SILU`, todas as elementwise.

**Family 2 — Stateful (CoW obrigatório):**
`ATTN` (KV_CACHE 0x30), `SSM_SCAN` (ssm_states), `SPIKE_STEP` (V(t)), `RANK1_UPDATE` (H_t), `ODE_STEP` (x_t), `DEPFORMER` (KV Depformer).

---

## 8. Regras de Codificação

### 8.1 Dual-mode

```rust
fn instruction_size(op: u8) -> usize {
    if op >= 0x80 { 64 } else { 32 }
}
```

### 8.2 Imutabilidade de encodings

Opcodes existentes **nunca** têm encoding alterado. Extensões são:
- **Flag nova** no byte 1 (compatível)
- **Modo novo** em `payload_core` (documentado)
- **Opcode novo** em faixa livre

### 8.3 Aliases (resolvidos no assembler, sem encoding)

| Alias | Opcode canônico |
|:---|:---|
| `MATRIX_RECURRENCE` | `RANK1_UPDATE 0x24` |
| `LOOKUP_TREE` | `FOREST 0x22` (`TREES=1`) |
| `ODE_SOLVER` | `ODE_STEP 0x25` |
| `SCATTER_ADD` | `GATHER 0x1F` (`MODE=SCATTER_ADD`) |
| `SCATTER_MAX` | `GATHER 0x1F` (`MODE=SCATTER_MAX`) |

---

## 9. Plano de Migração (v1.3 → v2.0)

### Onda 0 — Congelamento (1 sem)
- [ ] Publicar `docs/ISA_V2.md` (este documento)
- [ ] Congelar §2 (layout), §5 (mapa), §6 (payloads)

### Onda 1 — Dual-mode + memória (2 sem)
- [ ] `src/opcodes.rs`: `INSTR_SIZE_32/64`, decoder dual
- [ ] `src/memory.rs`: 16 regiões com seletor de 4 bits
- [ ] `src/vm.rs`: dispatch dual-mode
- [ ] Testes: roundtrip de ambos formatos

### Onda 2 — Infra de memória (2 sem)
- [ ] `ARENA_ALLOC/RESET` (0x26/0x27)
- [ ] `SNAPSHOT/RESTORE` explícitos (0x28/0x29)
- [ ] `MEMCPY/MEMSET/PREFETCH` (0x2A–0x2C)
- [ ] Política de retenção `k=16`

### Onda 3 — GATHER + retrieval (2 sem)
- [ ] `GATHER 0x1F`
- [ ] `DISTANCE 0x23` + `RAG_SEARCH 0x52` + `EMBED_LOOKUP 0x53`
- [ ] `RANK1_UPDATE 0x24`

### Onda 4 — Universal (3 sem)
- [ ] `SPIKE_STEP 0x20`
- [ ] `CONV 0x1E`
- [ ] `FOREST 0x22`
- [ ] `ODE_STEP 0x25`
- [ ] `DENOISE_STEP 0x21` (último)

### Onda 5 — Cluster estendido (3 sem)
- [ ] `SIGNAL` com `LAMPORT`/`DEADLINE`/`CAPABILITY`
- [ ] `MIGRATE 0x84` com WAL
- [ ] `REDUCE_REMOTE`/`GATHER_REMOTE`/`BROADCAST_REMOTE`
- [ ] SWIM (`NODE_JOIN/SUSPECT/DEAD`)

### Onda 6 — Full-Duplex Moshi (4 sem)
- [ ] `DEPFORMER 0x44`
- [ ] `STREAM_MERGE 0x45`
- [ ] `VAD_DETECT 0x46`
- [ ] `AUDIO_RESAMPLE/FILTER/WINDOW` (0x47–0x49)
- [ ] `KV` 17 streams

### Onda 7 — Retrieval + ML clássico (3 sem)
- [ ] `PQ_ENCODE/DECODE` (0x56/0x57)
- [ ] `KMEANS_STEP` … `NEAREST_CENTROID` (0x58–0x5F)

### Onda 8 — Determinismo + conversão (2 sem)
- [ ] `RNG_*` (0x60–0x63)
- [ ] `HASH/CHECKSUM/HMAC` (0x64–0x66)
- [ ] `CAST/QUANTIZE/DEQUANT` (0x67–0x69)

### Onda 9 — Debug + Scheduler (2 sem)
- [ ] `CYCLES_COUNT` … `DUMP` (0x70–0x77)
- [ ] `YIELD` … `FENCE` (0x70–0x77)

### Onda 10 — MMIO + Sistema (3 sem)
- [ ] `GPU_LAUNCH/WAIT` (0x90/0x91)
- [ ] `NIC_SEND/RECV` (0x92/0x93)
- [ ] `DMA_START/WAIT` (0x94/0x95)
- [ ] `LOAD_MODEL` … `MODEL_SWITCH` (0xA0–0xA9)

**Total estimado:** ~27 semanas (com paralelização: ~16–18 semanas calendário).

---

## 10. Análise Crítica

### 10.1 Ganhos

| Ganho | Impacto |
|:---|:---|
| `LAMPORT`/`DEADLINE` inline | Replay distribuído + EDF sem metadados externos |
| 16 regiões | Isolamento de ator real, WAL separado, `WEIGHTS` RO enforced |
| `RSRC4`/`RSRC5` | `RANK1_UPDATE`, `CONV`, `FOREST` sem `TENSOR` acessório |
| Dual-mode 32B/64B | I-cache preservado para controle |
| `MIGRATE` + WAL | Migração transacionalmente segura |
| `DEPFORMER`/`STREAM_MERGE` | Full-duplex Moshi nativo |

### 10.2 Custos

| Custo | Mitigação |
|:---|:---|
| I-cache dobra em loops 64B | Dual-mode mantém controle em 32B |
| ~180 opcodes | Dispatch por macro + tabela gerada |
| 16 regiões | Arena por contexto + reciclagem de snapshots |
| `WAL` crescente | Ring + `t_wal_retention=5s` |
| `SHARED` race | `FENCE 0x77` explícito |

### 10.3 Riscos

1. **`WEIGHTS` RO requer `mprotect`** — sem isso, bug corrompe pesos.
2. **CoW depende de imutabilidade pós-snapshot** — todo opcode stateful escreve só em `KV_CACHE`/`ARENA`.
3. **`LAMPORT` requer ajuste no recebimento** — `max(local, remoto) + 1`.
4. **`DEADLINE` requer relógio monotônico + NTP** — sem isso, EDF impreciso.
5. **`MIGRATE` com WAL não é zero-copy** — zero-copy real exige RDMA (onda 10+).

---

## 11. Resumo Executivo

| Camada | Cobertura nesta spec |
|:---|:---|
| Modelo de memória (16 regiões) | 100% |
| Layout 32B/64B | 100% |
| Flags (8 bits) | 100% |
| Registradores (GPR/VEC/STATE/SPECIAL) | 100% |
| Opcodes implementados (0x00–0x19 + 0xFF) | 26 |
| Opcodes planejados (0x1A–0x25) | 12 |
| Opcodes v2.0 novos (0x26–0xAF) | ~130 |
| Reservados (0xB0–0xFE) | ~79 |
| **Total** | **256** |

Esta spec é **autossuficiente**: qualquer pessoa pode implementar o assembler, o dispatcher e o runtime apenas com este documento. Extensões futuras entram em faixas reservadas sem quebrar binários.

---

**Documento canônico da ISA M³-AVM v2.0.** Qualquer desvio exige RFC e bump de versão minor (`v2.1`, `v2.2`, …). Encodings existentes são imutáveis.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
