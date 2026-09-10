> **ARQUIVO — material de entrada, NÃO normativo.**
> Recebido como "novos dados" (mapa ISA v2.4, 256/256) em 2026-09-10.
> Premissa do doc ("todos os 12 RFCs implementados (0001–0012)") É FALSA
> neste repo: nossos RFC-0001 é entrada arquivada, 0002–0006 têm outros
> temas (0002 Dual-Mode, 0003 Retention, 0004 GATHER, 0005 Determinism,
> 0006 Telemetry — todos IMPLEMENTED), 0007–0012 não existem. Numeração
> de RFCs, blocos `0x4A–0x4F`/`0xB7–0xBF`/`0xC0–0xFE`, regiões `0x10–0x17`
> e bits colidem com o registro (ver revisão na conversa). Verificação
> mecânica deste arquivo: cobertura 256 valores OK; ressalvas: NOP listado
> 2× no sumário, bit 17 ausente, duplicatas MIGRATE/MULTICAST/HEARTBEAT/
> PING/REDUCE_REMOTE/GATHER_REMOTE entre `0x4A–0x4F` e `0x84–0x8B`.
> Canônico: `docs/ESPEC.md` (+ `docs/ESPEC-V2.md` DRAFT). Nada aqui deve
> ser implementado sem reconciliação. Conteúdo original verbatim abaixo.
>
> ---

# M³-AVM ISA v2.4 — Mapa Completo de Opcodes (256/256 alocados)

> **Premissa:** todos os 12 RFCs implementados (0001–0012).  
> **Layout dual:** `0x00–0x7F` = 32B, `0x80–0xFF` = 64B (RFC-0001).  
> **Total:** 256 opcodes, 100% alocados.

---

## Sumário Estatístico

| Categoria | Qtd | Faixa |
|:---|:--:|:---|
| Implementados (v1.3) | 26 | `0x00–0x19`, `0xFF` |
| Planejados v2.0 | 12 | `0x1A–0x25` |
| Novos v2.0 | 82 | `0x26–0x77` |
| RFC-0012 (Clock) | 4 | `0x78–0x7B` |
| Reservados | 4 | `0x7C–0x7F` |
| Cluster estendido | 16 | `0x80–0x8F` |
| MMIO | 16 | `0x90–0x9F` |
| Sistema | 16 | `0xA0–0xAF` |
| ESCAPE (RFC-0001) | 7 | `0xB0–0xB6` |
| LCT (RFC-0002) | 9 | `0xB7–0xBF` |
| Storage (RFC-0003) | 8 | `0xC0–0xC7` |
| Power (RFC-0004) | 8 | `0xC8–0xCF` |
| Composite Deadline (RFC-0005) | 8 | `0xD0–0xD7` |
| Security (RFC-0006) | 8 | `0xD8–0xDF` |
| Observability (RFC-0007) | 8 | `0xE0–0xE7` |
| KV Tiering (RFC-0008) | 8 | `0xE8–0xEF` |
| Compression (RFC-0009) | 4 | `0xF0–0xF3` |
| Training (RFC-0010) | 8 | `0xF4–0xFB` |
| Unified IR (RFC-0011) | 3 | `0xFC–0xFE` |
| NOP | 1 | `0xFF` |
| **TOTAL** | **256** | — |

---

## Domínio 0 — Controle & Flow (`0x00–0x0F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x00` | `HALT` | v1.3 |
| `0x01` | `TENSOR` | v1.3 |
| `0x02` | `ATTN` | v1.3 |
| `0x03` | `STREAM` | v1.3 |
| `0x04` | `FORK` | v1.3 |
| `0x05` | `ABORT` | v1.3 |
| `0x06` | `SENSE` | v1.3 |
| `0x07` | `NORM` | v1.3 |
| `0x08` | `FFN` | v1.3 |
| `0x09` | `EMBED` | v1.3 |
| `0x0A` | `ADD` | v1.3 |
| `0x0B` | `SAMPLE` | v1.3 (+ `TOPK`) |
| `0x0C` | `COMPARE` | v1.3 |
| `0x0D` | `IF_EQUAL` | v1.3 |
| `0x0E` | `JUMP` | v1.3 |
| `0x0F` | `IF_INTERRUPT` | v1.3 |

## Domínio 1 — Matriz & Recurrence (`0x10–0x1F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x10` | `MATVEC` | v1.3 (+ `TRANSPOSE`) |
| `0x11` | `MUL` | v1.3 |
| `0x12` | `SILU` | v1.3 |
| `0x13` | `SSM_SCAN` | v1.3 |
| `0x14` | `SSM_RESET` | v1.3 |
| `0x15` | `CODEC_ENC` | v1.3 |
| `0x16` | `CODEC_DEC` | v1.3 |
| `0x17` | `AUDIO_ALIGN` | v1.3 |
| `0x18` | `CTX_SWITCH` | v1.3 |
| `0x19` | `ROPE` | v1.3 |
| `0x1A` | `REMOTE_SPAWN` | cluster 32B |
| `0x1B` | `SIGNAL` | cluster 32B |
| `0x1C` | `SEND_TENSOR` | cluster 32B |
| `0x1D` | `BARRIER` | cluster 32B |
| `0x1E` | `CONV` | universal |
| `0x1F` | `GATHER` | universal |

## Domínio 2 — Memória & Universal (`0x20–0x2F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x20` | `SPIKE_STEP` | universal |
| `0x21` | `DENOISE_STEP` | universal |
| `0x22` | `FOREST` | universal |
| `0x23` | `DISTANCE` | universal |
| `0x24` | `RANK1_UPDATE` | universal |
| `0x25` | `ODE_STEP` | universal |
| `0x26` | `ARENA_ALLOC` | v2.0 |
| `0x27` | `ARENA_RESET` | v2.0 |
| `0x28` | `SNAPSHOT` | v2.0 |
| `0x29` | `RESTORE` | v2.0 |
| `0x2A` | `MEMCPY` | v2.0 |
| `0x2B` | `MEMSET` | v2.0 |
| `0x2C` | `PREFETCH` | v2.0 |
| `0x2D` | `RESHAPE` | v2.0 |
| `0x2E` | `SLICE` | v2.0 |
| `0x2F` | `CONCAT` | v2.0 |

## Domínio 3 — Tensores Avançados (`0x30–0x3F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x30` | `SORT` | v2.0 |
| `0x31` | `TOPK` | v2.0 |
| `0x32` | `ARGMAX` | v2.0 |
| `0x33` | `REDUCE` | v2.0 |
| `0x34` | `BROADCAST` | v2.0 |
| `0x35` | `PAD` | v2.0 |
| `0x36` | `TILE` | v2.0 |
| `0x37` | `TRANSPOSE` | v2.0 |
| `0x38` | `KV_TRUNCATE` | v2.0 |
| `0x39` | `KV_COMPRESS` | v2.0 |
| `0x3A` | `FLASH_ATTN` | v2.0 |
| `0x3B` | `ATTN_SPARSE` | v2.0 |
| `0x3C` | `SOFTMAX` | v2.0 |
| `0x3D` | `GELU` | v2.0 |
| `0x3E` | `SIGMOID` | v2.0 |
| `0x3F` | `TANH` | v2.0 |

## Domínio 4 — Ativações & Áudio (`0x40–0x4F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x40` | `RELU` | v2.0 |
| `0x41` | `EXP` | v2.0 |
| `0x42` | `LOG` | v2.0 |
| `0x43` | `CLIP` | v2.0 |
| `0x44` | `DEPFORMER` | Moshi |
| `0x45` | `STREAM_MERGE` | Moshi |
| `0x46` | `VAD_DETECT` | Moshi |
| `0x47` | `AUDIO_RESAMPLE` | Moshi |
| `0x48` | `AUDIO_FILTER` | Moshi |
| `0x49` | `AUDIO_WINDOW` | Moshi |
| `0x4A` | `MULTICAST` | cluster |
| `0x4B` | `MIGRATE` | cluster |
| `0x4C` | `HEARTBEAT` | cluster |
| `0x4D` | `PING` | cluster |
| `0x4E` | `REDUCE_REMOTE` | cluster |
| `0x4F` | `GATHER_REMOTE` | cluster |

## Domínio 5 — Retrieval & ML Clássico (`0x50–0x5F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x50` | `RAG_INDEX_ADD` | retrieval |
| `0x51` | `RAG_INDEX_DEL` | retrieval |
| `0x52` | `RAG_SEARCH` | retrieval |
| `0x53` | `EMBED_LOOKUP` | retrieval |
| `0x54` | `HASH_BUCKET` | retrieval |
| `0x55` | `QUANTIZE_VEC` | retrieval |
| `0x56` | `PQ_ENCODE` | retrieval |
| `0x57` | `PQ_DECODE` | retrieval |
| `0x58` | `KMEANS_STEP` | clássico |
| `0x59` | `LINEAR_REG` | clássico |
| `0x5A` | `LOGISTIC_REG` | clássico |
| `0x5B` | `NAIVE_BAYES` | clássico |
| `0x5C` | `SVM_PREDICT` | clássico |
| `0x5D` | `PCA_STEP` | clássico |
| `0x5E` | `STANDARDIZE` | clássico |
| `0x5F` | `NEAREST_CENTROID` | clássico |

## Domínio 6 — Determinismo & Conversão (`0x60–0x6F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x60` | `RNG_SEED` | v2.0 |
| `0x61` | `RNG_NEXT` | v2.0 |
| `0x62` | `RNG_NORMAL` | v2.0 |
| `0x63` | `RNG_UNIFORM` | v2.0 |
| `0x64` | `HASH` | v2.0 |
| `0x65` | `CHECKSUM` | v2.0 |
| `0x66` | `HMAC` | v2.0 |
| `0x67` | `CAST` | v2.0 |
| `0x68` | `QUANTIZE` | v2.0 |
| `0x69` | `DEQUANT` | v2.0 |
| `0x6A` | `CYCLES_COUNT` | debug |
| `0x6B` | `TRACE_EVENT` | debug |
| `0x6C` | `SANITY_CHECK` | debug |
| `0x6D` | `PREEMPT_CHECK` | debug |
| `0x6E` | `ASSERT` | debug |
| `0x6F` | `DUMP` | debug |

## Domínio 7 — Scheduler (`0x70–0x7F`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x70` | `YIELD` | scheduler |
| `0x71` | `SET_DEADLINE` | scheduler |
| `0x72` | `GET_DEADLINE` | scheduler |
| `0x73` | `PRIORITY_SET` | scheduler |
| `0x74` | `PRIORITY_GET` | scheduler |
| `0x75` | `LOCK` | scheduler |
| `0x76` | `UNLOCK` | scheduler |
| `0x77` | `FENCE` | scheduler |
| `0x78` | `CLOCK_QUERY` | **RFC-0012** |
| `0x79` | `CLOCK_SYNC` | **RFC-0012** |
| `0x7A` | `CLOCK_ADJUST` | **RFC-0012** |
| `0x7B` | `DEADLINE_NEGOTIATE` | **RFC-0012** |
| `0x7C–0x7F` | *Reservado* | — |

## Domínio 8 — Cluster Estendido (`0x80–0x8F`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x80` | `REMOTE_SPAWN_EXT` | cluster 64B |
| `0x81` | `SIGNAL_EXT` | cluster 64B (+ LAMPORT/DEADLINE/CAP) |
| `0x82` | `SEND_TENSOR_EXT` | cluster 64B (+ COMPRESS) |
| `0x83` | `BARRIER_EXT` | cluster 64B |
| `0x84` | `MIGRATE` | cluster 64B (+ WAL) |
| `0x85` | `MULTICAST_EXT` | cluster 64B |
| `0x86` | `REDUCE_REMOTE_EXT` | cluster 64B |
| `0x87` | `GATHER_REMOTE_EXT` | cluster 64B |
| `0x88` | `SCATTER_REMOTE` | cluster 64B |
| `0x89` | `BROADCAST_REMOTE` | cluster 64B |
| `0x8A` | `HEARTBEAT_EXT` | cluster 64B |
| `0x8B` | `PING_EXT` | cluster 64B |
| `0x8C` | `NODE_JOIN` | cluster |
| `0x8D` | `NODE_LEAVE` | cluster |
| `0x8E` | `NODE_SUSPECT` | cluster |
| `0x8F` | `NODE_DEAD` | cluster |

## Domínio 9 — MMIO & Aceleradores (`0x90–0x9F`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0x90` | `GPU_LAUNCH` | v2.0 |
| `0x91` | `GPU_WAIT` | v2.0 |
| `0x92` | `NIC_SEND` | v2.0 |
| `0x93` | `NIC_RECV` | v2.0 |
| `0x94` | `DMA_START` | v2.0 |
| `0x95` | `DMA_WAIT` | v2.0 |
| `0x96` | `ATOMIC_CAS` | v2.0 |
| `0x97` | `ATOMIC_ADD` | v2.0 |
| `0x98` | `ATOMIC_XCHG` | v2.0 |
| `0x99` | `WFI` | v2.0 |
| `0x9A–0x9F` | *Reservado MMIO* | — |

## Domínio 10 — Sistema (`0xA0–0xAF`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xA0` | `LOAD_MODEL` | sistema |
| `0xA1` | `UNLOAD_MODEL` | sistema |
| `0xA2` | `SPAWN_CONTEXT` | sistema |
| `0xA3` | `KILL_CONTEXT` | sistema |
| `0xA4` | `SET_AFFINITY` | sistema |
| `0xA5` | `PROFILE_START` | sistema |
| `0xA6` | `PROFILE_STOP` | sistema |
| `0xA7` | `SET_MODEL` | sistema |
| `0xA8` | `GET_MODEL` | sistema |
| `0xA9` | `MODEL_SWITCH` | sistema |
| `0xAA–0xAF` | *Reservado sistema* | — |

## Domínio 11 — ESCAPE (`0xB0–0xB6`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xB0` | `ESCAPE` | **RFC-0001** |
| `0xB1` | `ESCAPE_128` | **RFC-0001** |
| `0xB2` | `ESCAPE_256` | **RFC-0001** |
| `0xB3` | `ESCAPE_VAR` | **RFC-0001** |
| `0xB4` | `VERSION` | **RFC-0001** |
| `0xB5` | `CAPABILITY_QUERY` | **RFC-0001** |
| `0xB6` | `CAPABILITY_ASSERT` | **RFC-0001** |

## Domínio 12 — Local Compute Topology (`0xB7–0xBF`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xB7` | `DEVICE_QUERY` | **RFC-0002** |
| `0xB8` | `DEVICE_SELECT` | **RFC-0002** |
| `0xB9` | `DEVICE_HEALTH` | **RFC-0002** |
| `0xBA` | `PLACE` | **RFC-0002** |
| `0xBB` | `SHARD` | **RFC-0002** |
| `0xBC` | `REPLICATE` | **RFC-0002** |
| `0xBD` | `REDUCE_LOCAL` | **RFC-0002** |
| `0xBE` | `MIGRATE_LOCAL` | **RFC-0002** |
| `0xBF` | `BARRIER_LOCAL` | **RFC-0002** |

## Domínio 13 — Storage Tiering (`0xC0–0xC7`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xC0` | `PIN` | **RFC-0003** |
| `0xC1` | `UNPIN` | **RFC-0003** |
| `0xC2` | `PREFETCH_TIER` | **RFC-0003** |
| `0xC3` | `STREAM_WEIGHTS` | **RFC-0003** |
| `0xC4` | `EVICT` | **RFC-0003** |
| `0xC5` | `TIER_QUERY` | **RFC-0003** |
| `0xC6` | `TIER_POLICY` | **RFC-0003** |
| `0xC7` | `WEIGHTS_HINT` | **RFC-0003** |

## Domínio 14 — Power & Thermal (`0xC8–0xCF`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xC8` | `POWER_QUERY` | **RFC-0004** |
| `0xC9` | `POWER_BUDGET_SET` | **RFC-0004** |
| `0xCA` | `POWER_BUDGET_GET` | **RFC-0004** |
| `0xCB` | `THERMAL_QUERY` | **RFC-0004** |
| `0xCC` | `THROTTLE_POLICY` | **RFC-0004** |
| `0xCD` | `POWER_CAP` | **RFC-0004** |
| `0xCE` | `ENERGY_ACCOUNT` | **RFC-0004** |
| `0xCF` | `THERMAL_HEADROOM` | **RFC-0004** |

## Domínio 15 — Composite Deadlines (`0xD0–0xD7`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xD0` | `DEADLINE_AND` | **RFC-0005** |
| `0xD1` | `DEADLINE_OR` | **RFC-0005** |
| `0xD2` | `DEADLINE_N_OF_M` | **RFC-0005** |
| `0xD3` | `DEADLINE_CHAIN` | **RFC-0005** |
| `0xD4` | `DEADLINE_QUERY` | **RFC-0005** |
| `0xD5` | `DEADLINE_RELAX` | **RFC-0005** |
| `0xD6` | `DEADLINE_TIGHTEN` | **RFC-0005** |
| `0xD7` | `DEADLINE_RESET` | **RFC-0005** |

## Domínio 16 — Security (`0xD8–0xDF`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xD8` | `KEY_AGREE` | **RFC-0006** |
| `0xD9` | `KEY_ROTATE` | **RFC-0006** |
| `0xDA` | `KEY_DERIVE` | **RFC-0006** |
| `0xDB` | `REPLAY_CHECK` | **RFC-0006** |
| `0xDC` | `CRYPTO_SIGN` | **RFC-0006** |
| `0xDD` | `CRYPTO_VERIFY` | **RFC-0006** |
| `0xDE` | `CRYPTO_ENCRYPT` | **RFC-0006** |
| `0xDF` | `CRYPTO_DECRYPT` | **RFC-0006** |

## Domínio 17 — Observability (`0xE0–0xE7`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xE0` | `TRACE_SPAN_START` | **RFC-0007** |
| `0xE1` | `TRACE_SPAN_END` | **RFC-0007** |
| `0xE2` | `TRACE_ATTR` | **RFC-0007** |
| `0xE3` | `TRACE_LINK` | **RFC-0007** |
| `0xE4` | `TRACE_EXPORT` | **RFC-0007** |
| `0xE5` | `TRACE_INJECT` | **RFC-0007** |
| `0xE6` | `TRACE_EXTRACT` | **RFC-0007** |
| `0xE7` | `TRACE_QUERY` | **RFC-0007** |

## Domínio 18 — KV Tiering (`0xE8–0xEF`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xE8` | `KV_TIER_MOVE` | **RFC-0008** |
| `0xE9` | `KV_EVICT` | **RFC-0008** |
| `0xEA` | `KV_PIN` | **RFC-0008** |
| `0xEB` | `KV_RETRIEVE` | **RFC-0008** |
| `0xEC` | `KV_RECOMPUTE` | **RFC-0008** |
| `0xED` | `KV_COMPRESS_LOSSY` | **RFC-0008** |
| `0xEE` | `KV_SPARSE_MASK` | **RFC-0008** |
| `0xEF` | `KV_MIGRATE_REMOTE` | **RFC-0008** |

## Domínio 19 — Compression (`0xF0–0xF3`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xF0` | `COMPRESS` | **RFC-0009** |
| `0xF1` | `DECOMPRESS` | **RFC-0009** |
| `0xF2` | `COMPRESS_QUERY` | **RFC-0009** |
| `0xF3` | `ENTROPY_ESTIMATE` | **RFC-0009** |

## Domínio 20 — Training (`0xF4–0xFB`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xF4` | `GRAD_ACCUM` | **RFC-0010** |
| `0xF5` | `GRAD_ZERO` | **RFC-0010** |
| `0xF6` | `OPTIMIZER_STEP` | **RFC-0010** |
| `0xF7` | `OPTIMIZER_INIT` | **RFC-0010** |
| `0xF8` | `LORA_APPLY` | **RFC-0010** |
| `0xF9` | `LORA_MERGE` | **RFC-0010** |
| `0xFA` | `CHECKPOINT_SAVE` | **RFC-0010** |
| `0xFB` | `CHECKPOINT_RESTORE` | **RFC-0010** |

## Domínio 21 — Unified Model IR (`0xFC–0xFE`, 64B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xFC` | `MODEL_LOAD_UNIFIED` | **RFC-0011** |
| `0xFD` | `MODEL_COMPILE` | **RFC-0011** |
| `0xFE` | `MODEL_QUERY` | **RFC-0011** |

## Domínio 22 — Terminal (`0xFF`, 32B)

| Hex | Nome | Origem |
|:--:|:---|:---|
| `0xFF` | `NOP` | v1.3 |

---

## Registradores Especiais (consolidado)

| Reg | Nome | Origem |
|:--:|:---|:---|
| `SP0` | `PC` | v1.3 |
| `SP1` | `SP` | v1.3 |
| `SP2` | `FP` | v1.3 |
| `SP3` | `CTX_ID` | v1.3 |
| `SP4` | `NODE_ID` | v1.3 |
| `SP5` | `LAMPORT` | v2.0 |
| `SP6` | `DEADLINE` | v2.0 |
| `SP7` | `STATUS` | v2.0 |
| `SP8` | `DEVICE_ID` | **RFC-0002** |
| `SP9` | `DEVICE_AFFINITY` | **RFC-0002** |
| `SP10` | `DEVICE_HEALTH` | **RFC-0002** |
| `SP11` | `NUMA_NODE` | **RFC-0002** |
| `SP12` | `POWER_BUDGET_MW` | **RFC-0004** |
| `SP13` | `THERMAL_HEADROOM_MC` | **RFC-0004** |
| `SP14` | `THROTTLE_STATE` | **RFC-0004** |
| `SP15` | `DEADLINE_TREE_ID` | **RFC-0005** |
| `SP16` | `DEADLINE_TREE_MASK` | **RFC-0005** |
| `SP17` | `CLOCK_HEALTH` | **RFC-0012** |
| `SP18` | `CLOCK_OFFSET_NS` | **RFC-0012** |

---

## Regiões de Memória (consolidado)

| ID | Nome | Origem |
|:--:|:---|:---|
| `0x0` | `TEXT` | v2.0 |
| `0x1` | `GLOBAL` | v2.0 |
| `0x2` | `WEIGHTS` | v2.0 |
| `0x3` | `ACTIVATION` | v2.0 |
| `0x4` | `KV_CACHE` | v2.0 |
| `0x5` | `ARENA` | v2.0 |
| `0x6` | `SHARED` | v2.0 |
| `0x7` | `WAL` | v2.0 |
| `0x8` | `SNAPSHOT` | v2.0 |
| `0x9` | `STREAM_RING` | v2.0 |
| `0xA` | `RAG_INDEX` | v2.0 |
| `0xB` | `CLUSTER_STAGING` | v2.0 |
| `0xC` | `FEDERATED` | v2.0 |
| `0xD` | `CONFIDENTIAL` | v2.0 |
| `0xE` | `SCRATCH_DEVICE[i]` | **RFC-0002** |
| `0xF` | `MMIO` | v2.0 |
| `0x10` | `WEIGHTS_COLD` | **RFC-0003** |
| `0x11` | `WEIGHTS_WARM` | **RFC-0003** |
| `0x12` | `WEIGHTS_HOT` | **RFC-0003** |
| `0x13` | `KEYS` | **RFC-0006** |
| `0x14` | `KV_CACHE_COLD` | **RFC-0008** |
| `0x15` | `KV_CACHE_REMOTE` | **RFC-0008** |
| `0x16` | `GRADIENTS` | **RFC-0010** |
| `0x17` | `OPTIMIZER_STATE` | **RFC-0010** |

---

## Feature Bits (consolidado)

| Bit | Nome | RFC |
|:--:|:---|:---|
| 0 | `ESCAPE` | RFC-0001 |
| 1 | `DUAL_MODE` | RFC-0001 |
| 2 | `LAMPORT` | RFC-0001 |
| 3 | `DEADLINE` | RFC-0001 |
| 4 | `WAL_MIGRATE` | RFC-0001 |
| 5 | `COW_SNAPSHOT` | RFC-0001 |
| 6 | `CLUSTER` | RFC-0001 |
| 7 | `GPU_MMIO` | RFC-0001 |
| 8 | `TENSOR_REGS` | RFC-0001 |
| 9 | `REGION_TABLE` | RFC-0001 |
| 10 | `MULTI_DEVICE` | RFC-0002 |
| 11 | `DEVICE_TOPOLOGY` | RFC-0002 |
| 12 | `UNIFIED_MEMORY` | RFC-0002 |
| 13 | `DEVICE_FAILOVER` | RFC-0002 |
| 14 | `NUMA_AWARE` | RFC-0002 |
| 15 | `MIG_SLICES` | RFC-0002 |
| 16 | `MULTI_RIG` | RFC-0002 (futuro) |
| 18 | `STORAGE_TIERING` | RFC-0003 |
| 19 | `NVME_STREAM` | RFC-0003 |
| 20 | `RDMA_STORAGE` | RFC-0003 |
| 21 | `POWER_AWARE` | RFC-0004 |
| 22 | `THERMAL_AWARE` | RFC-0004 |
| 23 | `ENERGY_ACCOUNTING` | RFC-0004 |
| 24 | `COMPOSITE_DEADLINE` | RFC-0005 |
| 25 | `MULTI_STREAM_SYNC` | RFC-0005 |
| 26 | `NOISE_PROTOCOL` | RFC-0006 |
| 27 | `FORWARD_SECRECY` | RFC-0006 |
| 28 | `REPLAY_PROTECTION` | RFC-0006 |
| 29 | `KEY_ROTATION` | RFC-0006 |
| 30 | `DISTRIBUTED_TRACING` | RFC-0007 |
| 31 | `OTLP_EXPORT` | RFC-0007 |
| 32 | `SPAN_LINKS` | RFC-0007 |
| 33 | `KV_TIERING` | RFC-0008 |
| 34 | `KV_SPARSE_EVICT` | RFC-0008 |
| 35 | `KV_RECOMPUTE` | RFC-0008 |
| 36 | `COMPRESSION` | RFC-0009 |
| 37 | `ADAPTIVE_COMPRESSION` | RFC-0009 |
| 38 | `TRAINING` | RFC-0010 |
| 39 | `LORA` | RFC-0010 |
| 40 | `GRAD_CHECKPOINT` | RFC-0010 |
| 41 | `MIXED_PRECISION` | RFC-0010 |
| 42 | `M3IR` | RFC-0011 |
| 43 | `MULTI_BACKEND` | RFC-0011 |
| 44 | `CLOCK_HEALTH` | RFC-0012 |
| 45 | `DEADLINE_ADAPT` | RFC-0012 |
| 46 | `PTP_SYNC` | RFC-0012 |

---

## Veredito

Com os 12 RFCs implementados:

- **256/256 opcodes alocados** — nenhum espaço livre para opcodes não-ESCAPE
- **ESCAPE (`0xB0`)** é a **única** via de extensão futura — e é **infinita** via `ext_opcode` + `ext_len`
- **24 regiões** ativas (16 core + 8 RFC)
- **47 feature bits** ativos (0–46)
- **19 registradores especiais** (SP0–SP18)
- **Zero quebra binária** entre v1.3 e v2.4

Se você quiser adicionar algo **agora**, a única via é **ESCAPE**. Isso não é limitação — é **disciplina**. Foi o que permitiu ao x86, ARM e z/Architecture durarem décadas.

**A spec está fechada. Qualquer extensão futura passa por RFC + ESCAPE.**

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
