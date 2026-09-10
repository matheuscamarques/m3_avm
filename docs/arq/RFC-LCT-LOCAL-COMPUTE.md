> **ARQUIVO — material de entrada, NÃO normativo.**
> Recebido como "novos dados" em 2026-09-10 sob o título "RFC-0002 — Local
> Compute Topology (LCT)". ATENÇÃO — colisão de número: **RFC-0002 já existe
> neste repo** (`docs/RFC-0002-DUALMODE.md`, Dual-Mode Decoder,
> IMPLEMENTED). Este documento LCT precisará de renumeração (sugerido:
> RFC-0007, próximo livre) antes de qualquer adoção. Colisão de opcodes
> `0xB7–0xBF` idem (nossa reserva: placeholder federated, ESPEC-V2 R3).
> A especificação canônica é `docs/ESPEC.md` (+ `docs/ESPEC-V2.md` DRAFT).
> Revisão na conversa de trabalho. Nada aqui deve ser implementado sem
> reconciliação. Conteúdo original preservado verbatim abaixo.
>
> ---

# RFC-0002 — Local Compute Topology (LCT)

```
RFC Number  : 0002
Title       : Local Compute Topology for M³-AVM
Status      : PROPOSED
Category    : Standards Track
Author      : Matheus de Camargo Marques
Date        : 2026-09-10
Updates     : ISA v2.0 (docs/ISA_V2.md), RFC-0001 (extension mechanism)
Obsoletes   : None
```

---

## Abstract

Este RFC especifica a **camada de descoberta e orquestração de hardware local** da M³-AVM. Define como a VM enxerga, seleciona e distribui trabalho entre **múltiplas CPUs, GPUs, NPUs, TPUs, FPGAs e aceleradores heterogêneos** dentro da mesma máquina física.

Introduz:

1. **Device Registry** — descoberta de hardware no boot
2. **Topology Matrix** — custo de comunicação entre devices (NVLink, PCIe, UPI, CXL)
3. **9 opcodes novos** (`0xB7–0xBF`) para orquestração local
4. **Regiões `SCRATCH_DEVICE[i]`** para memória por device
5. **Device-aware EDF** — scheduler ciente de topologia
6. **Transporte unificado** — `MIGRATE`/`REDUCE`/`BARRIER` escolhem automaticamente NVLink, PCIe, RDMA ou TCP

O princípio central é: **rig é cluster, cluster é rig.** A diferença é apenas o barramento físico. O bytecode nunca sabe se está indo por NVLink ou TCP.

---

## 1. Motivação

A ISA M³-AVM v2.0 cobre distribuição entre nós (`PLANO_AVM_CLUSTER.md`), mas trata um rig multi-GPU como **um único device lógico**. Isso impede:

| Capacidade perdida | Impacto |
|:---|:---|
| **Tensor parallelism** | Llama-70B não cabe em 1 GPU; impossível dividir camada |
| **Pipeline parallelism** | Camadas diferentes não podem morar em GPUs diferentes |
| **Expert parallelism** | MoE com experts em GPUs distintas é inviável |
| **Data parallelism** | Mesmo modelo, batches distintos, sem replicação gerenciada |
| **NUMA-aware placement** | Memória alocada no socket errado degrada 2–3× |
| **NVLink zero-copy** | 900 GB/s de banda desperdiçada |
| **Unified memory** | CPU↔GPU exige cópia explícita |
| **Failover de device** | GPU morre → rig inteiro cai |

Em 2026, uma workstation típica de IA tem 1–4 GPUs. Um servidor tem 8. Um pod de treino tem 64+. A M³-AVM **precisa** saber disso.

A lição histórica é clara: sistemas operacionais que ignoram topologia de hardware morrem (ex: early Linux SMP, early Windows NT). Os que sobrevivem (Linux NUMA-aware, macOS Metal, CUDA) tratam hardware como **cidadão de primeira classe**.

Este RFC faz isso.

---

## 2. Convenções e Terminologia

- **MAY / MUST / SHOULD / MUST NOT** seguem RFC 2119
- **Device** = unidade de computação (CPU socket, GPU, NPU, TPU, FPGA, acelerador)
- **Transport** = meio físico de comunicação (NVLink, PCIe, UPI, CXL, RDMA, TCP)
- **Topology** = matriz de custos entre devices
- **Locality** = proximidade física (mesmo socket, mesmo PCIe root, mesmo NVLink domain)
- **Rig** = conjunto de devices na mesma máquina física
- **Node** = conjunto de rigs conectados por rede (cluster)
- **Placement** = decisão de onde alocar tensor/contexto

---

## 3. Device Registry

### 3.1 Estrutura

A região `0x1` (GLOBAL) MUST conter, no offset reservado `0x1000`, a estrutura `DEVICE_REGISTRY`:

```
Offset 0x1000 : MAGIC "M3DV" (0x4D 0x33 0x44 0x56)
Offset 0x1004 : N_DEVICES (u32 LE)
Offset 0x1008 : RESERVED (u32)
Offset 0x1010 : DEVICES[N_DEVICES] — array de entradas de 128 bytes

Entrada DEVICES[i] (128 bytes):
  Bytes 0-1    : device_id (u16 LE)     — 0 = boot CPU, 1..N = aceleradores
  Byte 2       : kind (u8)              — 0=CPU, 1=GPU, 2=NPU, 3=TPU, 4=FPGA,
                                           5=PHOTONIC, 6=NEUROMORPHIC, 7=QUANTUM
  Byte 3       : flags (u8)             — bit0=healthy, bit1=available,
                                           bit2=unified_mem, bit3=mig,
                                           bit4=ecc, bit5=throttling
  Bytes 4-7    : arch (u32 LE)          — vendor<<16 | model
  Bytes 8-11   : cores (u32 LE)         — SMs / threads / unidades
  Bytes 12-15  : clock_mhz (u32 LE)
  Bytes 16-23  : mem_bytes (u64 LE)
  Bytes 24-31  : mem_bandwidth_mbps (u64 LE)
  Bytes 32-35  : compute_fp32_gflops (u32 LE)
  Bytes 36-39  : compute_fp16_gflops (u32 LE)
  Bytes 40-43  : compute_int8_gops (u32 LE)
  Byte 44      : numa_node (u8)
  Byte 45      : pcie_root (u8)
  Byte 46      : nvlink_domain (u8)
  Byte 47      : nvlink_links (u8)      — 0 = sem NVLink
  Bytes 48-55  : capabilities (u64 LE)  — bitmask (ver §3.3)
  Bytes 56-63  : reserved (u64)
  Bytes 64-79  : name (char[16])        — ex: "H100-SXM-80G"
  Bytes 80-95  : driver_version (char[16])
  Bytes 96-103 : uuid (u64 LE)
  Bytes 104-111: reserved (u64)
  Bytes 112-127: vendor_data (16 bytes) — livre para driver
```

### 3.2 População no boot

O runtime MUST popular `DEVICE_REGISTRY` no boot usando APIs nativas:

| Plataforma | API |
|:---|:---|
| Linux + NVIDIA | `cudaGetDeviceProperties`, NVML |
| Linux + AMD | `hipGetDeviceProperties`, ROCm SMI |
| Linux + Intel | `zeDeviceGetProperties` (Level Zero), oneAPI |
| Linux + CPU | `/sys/devices/system/cpu`, `/sys/devices/system/node` |
| macOS + Apple | Metal `MTLCopyAllDevices` |
| Windows | DXGI, CUDA, NVML |

Se nenhuma API estiver disponível, o runtime MUST popular **pelo menos** `DEVICE_REGISTRY[0]` com a CPU boot e `N_DEVICES = 1`. Isso garante compatibilidade com ambientes restritos.

### 3.3 Capabilities bitmask

| Bit | Nome | Significado |
|:--:|:---|:---|
| 0 | `TENSOR_CORE` | Multiplicação de matriz acelerada |
| 1 | `FP8` | Suporte a FP8 |
| 2 | `INT4` | Suporte a INT4 |
| 3 | `INT8` | Suporte a INT8 |
| 4 | `BF16` | Suporte a BF16 |
| 5 | `FP16` | Suporte a FP16 |
| 6 | `FP32` | Suporte a FP32 |
| 7 | `FP64` | Suporte a FP64 |
| 8 | `UNIFIED_MEM` | Memória unificada CPU↔device |
| 9 | `MIG` | Suporte a MIG (Multi-Instance GPU) |
| 10 | `RDMA` | RDMA nativo |
| 11 | `NVLink` | NVLink para outros devices |
| 12 | `PCIe_P2P` | Peer-to-peer via PCIe |
| 13 | `ATOMIC_GLOBAL` | Operações atômicas globais |
| 14 | `ECC` | Correção de erro de memória |
| 15 | `SPARSE` | Suporte a tensores esparsos |
| 16-63 | RESERVED | Reservado |

---

## 4. Topology Matrix

### 4.1 Estrutura

A região `0x1` (GLOBAL) MUST conter, no offset reservado `0x2000`, a `TOPOLOGY` matrix:

```
Offset 0x2000 : MAGIC "M3TP" (0x4D 0x33 0x54 0x50)
Offset 0x2004 : N (u32 LE) — mesmo valor de N_DEVICES
Offset 0x2008 : RESERVED (u32)
Offset 0x2010 : MATRIX[N][N] — cada entrada 8 bytes

Entrada MATRIX[i][j] (8 bytes):
  Bytes 0-3 : latency_ns (u32 LE)  — latência de i para j
  Bytes 4-7 : bandwidth_mbps (u32 LE) — banda efetiva
```

### 4.2 Valores de referência (2026)

| Transport | Latência | Banda | Notas |
|:---|:---|:---|:---|
| Silicon (mesmo SM) | <1 ns | TB/s | — |
| L2 cache | ~10 ns | 5–10 TB/s | — |
| L3 cache | ~30 ns | 1–3 TB/s | — |
| NVLink 4.0 | ~200 ns | 900 GB/s | H100↔H100 |
| NVLink 5.0 | ~150 ns | 1.8 TB/s | B100↔B100 |
| NVSwitch | ~250 ns | 900 GB/s | via switch |
| UPI 2.0 | ~100 ns | 200 GB/s | Socket↔Socket Intel |
| Infinity Fabric | ~120 ns | 200 GB/s | Socket↔Socket AMD |
| PCIe Gen5 x16 | ~1 µs | 64 GB/s | CPU↔GPU |
| PCIe Gen6 x16 | ~700 ns | 128 GB/s | 2025+ |
| CXL 3.0 | ~300 ns | 64 GB/s | Memória compartilhada |
| RDMA (InfiniBand) | ~2 µs | 400 GB/s | Nó↔Nó |
| RDMA (RoCEv2) | ~5 µs | 200 GB/s | Nó↔Nó |
| TCP (10 GbE) | ~50 µs | 1.25 GB/s | Nó↔Nó |
| TCP (100 GbE) | ~20 µs | 12.5 GB/s | Nó↔Nó |
| Internet | ~50 ms | ~1 GB/s | WAN |

Runtime MUST popular `MATRIX[i][j]` com valores medidos quando possível, defaults documentados quando não.

### 4.3 População

| Plataforma | API de topologia |
|:---|:---|
| NVIDIA | `cudaDeviceGetP2PAttribute`, NVML |
| AMD | `hipDeviceGetP2PAttribute`, ROCm SMI |
| Intel | Level Zero `zeDeviceP2PGetProperties` |
| CPU NUMA | `/sys/devices/system/node/node*/distance` |
| PCIe | `/sys/bus/pci/devices/*/` |

Runtime MUST detectar loops (`MATRIX[i][i] = 0`) e simetria (`MATRIX[i][j] ≈ MATRIX[j][i]`). Assimetrias > 20% MUST ser logadas.

---

## 5. Registradores Especiais

### 5.1 Novos registradores

| Reg | Nome | Descrição |
|:--:|:---|:---|
| `SP8` | `DEVICE_ID` | Device ativo do contexto (default 0 = CPU) |
| `SP9` | `DEVICE_AFFINITY` | Bitmask de devices permitidos (default = todos) |
| `SP10` | `DEVICE_HEALTH` | Bitmask de saúde (atualizado por `DEVICE_HEALTH`) |
| `SP11` | `NUMA_NODE` | NUMA node do contexto (default = herdado) |

`SP8–SP11` só existem quando a feature `MULTI_DEVICE` (bit 10) está ativa. Runtime sem suporte a LCT trata acesso como erro `UnsupportedRegister`.

### 5.2 Compatibilidade

`SP8 = 0` significa "CPU boot" — comportamento idêntico ao v2.0 sem LCT. Bytecode v2.0 roda inalterado.

---

## 6. Opcodes de LCT (`0xB7–0xBF`)

### 6.1 Alocação

| Opcode | Nome | Formato | Função |
|:--:|:---|:---|:---|
| `0xB7` | `DEVICE_QUERY` | 64B | Popula info de device em registrador |
| `0xB8` | `DEVICE_SELECT` | 64B | Define `SP8` (device ativo) |
| `0xB9` | `DEVICE_HEALTH` | 64B | Atualiza `SP10` |
| `0xBA` | `PLACE` | 64B | Hint de placement para tensor |
| `0xBB` | `SHARD` | 64B | Divide tensor em N devices |
| `0xBC` | `REPLICATE` | 64B | Replica tensor em N devices |
| `0xBD` | `REDUCE_LOCAL` | 64B | Allreduce intra-rig |
| `0xBE` | `MIGRATE_LOCAL` | 64B | Migração device↔device |
| `0xBF` | `BARRIER_LOCAL` | 64B | Barreira intra-rig |

**Todos os 9 opcodes são reservados nesta especificação. Opcodes `0xB0–0xB6` permanecem como ESCAPE (RFC-0001). Opcodes `0xC0–0xFE` permanecem reservados.**

### 6.2 `DEVICE_QUERY (0xB7)`

```
Byte 0      : 0xB7
Byte 1      : flags (0 = query by id, 1 = query by capability)
Byte 2      : rdest — GPR que receberá índice do primeiro device matching
Byte 3      : rsrc1 — device_id (0..N) ou capability bitmask
Byte 4-7    : 0xFF (não usado)
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-35 : kind_filter (u32)  — 0 = qualquer, 1=GPU, 2=NPU, ...
Bytes 36-39 : capability_filter (u32)
Bytes 40-43 : min_mem_mb (u32)   — filtro de VRAM mínima
Bytes 44-47 : min_bandwidth_mbps (u32)
Bytes 48-55 : numa_affinity (u64) — bitmask de NUMA nodes preferidos
Bytes 56-63 : reservado
```

Semântica: escreve em `rdest` o `device_id` do primeiro device que satisfaz os filtros. Se nenhum, `rdest = 0xFFFF`.

### 6.3 `DEVICE_SELECT (0xB8)`

```
Byte 0      : 0xB8
Byte 1      : flags (0 = select by id, 1 = select by filter)
Byte 2      : rdest — status (0 = ok, 1 = unavailable, 2 = affinity violation)
Byte 3      : rsrc1 — device_id ou filtro handle
Byte 4-7    : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-39 : numa_mask (u64)
Bytes 40-47 : device_mask (u64) — bitmask dos devices permitidos
Bytes 48-51 : min_mem_mb (u32)
Bytes 52-55 : capability_filter (u32)
Bytes 56-63 : reservado
```

Semântica: define `SP8` (device ativo) e `SP9` (afinidade) para o contexto corrente. Falha se device indisponível ou viola afinidade.

### 6.4 `DEVICE_HEALTH (0xB9)`

```
Byte 0      : 0xB9
Byte 1      : flags
Byte 2      : rdest — status global (0=ok, 1=degraded, 2=failed)
Byte 3      : rsrc1 — device_id (0xFF = todos)
Bytes 4-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-35 : temperature_mc (u32) — millicelsius
Bytes 36-39 : power_mw (u32)       — milliwatts
Bytes 40-43 : utilization_pct (u32)
Bytes 44-47 : memory_used_mb (u32)
Bytes 48-51 : ecc_errors (u32)
Bytes 52-55 : throttling_pct (u32)
Bytes 56-63 : reservado
```

Semântica: atualiza `SP10` (bitmask de saúde) e escreve telemetria. Runtime MAY usar isso para decidir failover.

### 6.5 `PLACE (0xBA)`

```
Byte 0      : 0xBA
Byte 1      : flags (0 = hint, 1 = require)
Byte 2      : rdest — status (0 = placed, 1 = fallback, 2 = failed)
Byte 3      : rsrc1 — tensor handle
Bytes 4-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-39 : device_mask (u64) — devices preferidos
Bytes 40-47 : numa_mask (u64)
Bytes 48-51 : min_bandwidth_mbps (u32)
Bytes 52-55 : min_mem_mb (u32)
Bytes 56-63 : reservado
```

Semântica: informa ao runtime onde o tensor **deveria** morar. Se `flags.bit0=0`, runtime pode ignorar. Se `flags.bit0=1`, runtime MUST alocar no device matching ou falhar.

### 6.6 `SHARD (0xBB)`

```
Byte 0      : 0xBB
Byte 1      : flags — bit0=TP (tensor), bit1=PP (pipeline), bit2=EP (expert)
Byte 2      : rdest — handle do tensor shardado
Byte 3      : rsrc1 — tensor original
Byte 4      : rsrc2 — lista de devices (handle)
Byte 5      : rsrc3 — axis (u8) ou 0xFF = automático
Bytes 6-7   : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-39 : n_shards (u64)
Bytes 40-47 : device_mask (u64)
Bytes 48-55 : shard_offsets[2] (u64) — para PP
Bytes 56-63 : reservado
```

Semântica: divide tensor em `n_shards` ao longo de `axis`, distribuindo entre devices. Estratégias:
- `TP` (Tensor Parallel): divide pesos em 4 GPUs (Llama-70B TP=4)
- `PP` (Pipeline Parallel): divide camadas em 4 GPUs (GPT-NeoX PP=4)
- `EP` (Expert Parallel): experts de MoE em GPUs distintas

### 6.7 `REPLICATE (0xBC)`

```
Byte 0      : 0xBC
Byte 1      : flags — bit0=async, bit1=verify_hash
Byte 2      : rdest — handles das réplicas
Byte 3      : rsrc1 — tensor original
Byte 4      : rsrc2 — lista de devices
Byte 5-7    : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext (HMAC da réplica)
Bytes 32-39 : n_replicas (u64)
Bytes 40-47 : device_mask (u64)
Bytes 48-55 : checksum_seed (u64)
Bytes 56-63 : reservado
```

Semântica: replica tensor em N devices (data parallel). Runtime escolhe transporte por `TOPOLOGY[i][j]`.

### 6.8 `REDUCE_LOCAL (0xBD)`

```
Byte 0      : 0xBD
Byte 1      : flags — bit0=async, bit1=inplace
Byte 2      : rdest — tensor resultado
Byte 3      : rsrc1 — tensor local
Byte 4      : rsrc2 — lista de devices
Byte 5-7    : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-35 : op (0=allreduce_sum, 1=allreduce_mean, 2=allreduce_max,
                   3=allreduce_min, 4=allgather, 5=reduce_scatter,
                   6=alltoall, 7=broadcast)
Bytes 36-39 : n_devices (u32)
Bytes 40-47 : device_mask (u64)
Bytes 48-51 : transport_hint (u32) — 0=auto, 1=NVLink, 2=PCIe, 3=RDMA
Bytes 52-55 : timeout_ms (u32)
Bytes 56-63 : reservado
```

Semântica: collective operation intra-rig. Runtime MUST escolher transporte por `TOPOLOGY`. Se `transport_hint=0`, runtime escolhe o mais barato.

### 6.9 `MIGRATE_LOCAL (0xBE)`

```
Byte 0      : 0xBE
Byte 1      : flags — bit0=async, bit1=verify
Byte 2      : rdest — status
Byte 3      : rsrc1 — tensor origem
Byte 4      : rsrc2 — device destino (u8)
Byte 5-7    : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext (HMAC)
Bytes 32-39 : src_offset (u64)
Bytes 40-47 : byte_len (u64)
Bytes 48-55 : checksum_seed (u64)
Bytes 56-63 : reservado
```

Semântica: migra tensor device↔device. Transporte escolhido por `TOPOLOGY`. Se src == dst, no-op. Semântica transacional com WAL se `flags.bit0=0`.

### 6.10 `BARRIER_LOCAL (0xBF)`

```
Byte 0      : 0xBF
Byte 1      : flags — bit0=async
Byte 2      : rdest — status
Byte 3-4    : 0xFF
Bytes 8-15  : lamport
Bytes 16-23 : deadline
Bytes 24-31 : payload_ext
Bytes 32-35 : barrier_id (u32)
Bytes 36-39 : expected (u32) — número de participants
Bytes 40-43 : timeout_ms (u32)
Bytes 44-47 : mode (0=sync, 1=async_notify)
Bytes 48-55 : device_mask (u64)
Bytes 56-63 : reservado
```

Semântica: barreira intra-rig. Bloqueia até `expected` participants sinalizarem. Timeout → `NACK` + desbloqueio.

---

## 7. Regiões `SCRATCH_DEVICE[i]`

### 7.1 Substituição da região `0xE`

A região `0xE` (SCRATCH_GPU) deixa de ser única. Com Region Table (RFC-0001 §5.2):

```
0xE, alias 0x00 : SCRATCH_DEVICE[0]  — GPU0 (ou CPU0 se sem GPU)
0xE, alias 0x01 : SCRATCH_DEVICE[1]  — GPU1
0xE, alias 0x02 : SCRATCH_DEVICE[2]  — GPU2
...
0xE, alias 0xFF : SCRATCH_DEVICE[255] — futuro
```

Runtime MUST popular Region Table no boot com as entradas correspondentes aos devices descobertos. Runtime v2.0 sem LCT popula apenas `SCRATCH_DEVICE[0]` (comportamento atual).

### 7.2 Endereçamento

```
Endereço: 0xE000_0000_0000_0000 + (device_id << 56) + offset
```

`device_id` de 8 bits permite 256 devices. `offset` de 56 bits = 64 PiB por device.

### 7.3 Alocação

`ARENA_ALLOC (0x26)` com região `0xE` aloca no device corrente (`SP8`). Para alocar em device específico:

```asm
DEVICE_SELECT rStatus, DEVICE=3      ; GPU3
ARENA_ALLOC rTensor, SIZE=1GB, REGION=0xE
```

Se `SP8 = 0`, aloca no device 0 (compatibilidade).

---

## 8. Device-Aware EDF Scheduler

### 8.1 Modelo

Cada contexto tem:

```
Context {
  ...
  deadline        : u64          ; EDF
  device_affinity : u64 (bitmask)
  numa_affinity   : u64 (bitmask)
  device_current  : u8           ; SP8
  priority_base   : u8           ; RED/BLUE/GREEN
}
```

Prioridade efetiva:

```
prio_eff(ctx) = (deadline, device_locality, priority_base)
```

Onde `device_locality` mede quão perto o contexto está do device ideal:

```
device_locality = 0 se ctx.device_current ∈ ctx.device_affinity
                  1 se há fallback próximo (mesmo NUMA)
                  2 se há fallback distante (cross-NUMA)
                  ∞ se nenhum device compatível
```

Scheduler MUST escolher contexto com menor `prio_eff` primeiro.

### 8.2 Herança de prioridade cross-device

Quando um contexto `A` (device 2) espera por `BARRIER_LOCAL` de `B` (device 3):

- Runtime MUST elevar prioridade de `B` para a de `A`
- Se `B` estiver bloqueado por `C` (device 4), propaga
- Detecção de deadlock: se ciclo detectado em ≤ 16 hops, aborta `C`

### 8.3 Failover automático

Se `DEVICE_HEALTH` retorna `degraded` ou `failed`:

1. Runtime marca device como indisponível em `SP10`
2. Contextos com afinidade restrita ao device são notificados via `SIGNAL`
3. Se `FEATURE_DEVICE_FAILOVER` (bit 13) ativo:
   - Runtime tenta migrar tensores via `MIGRATE_LOCAL`
   - Se migração falha, contexto é movido para CPU fallback
   - Deadline é reavaliado (pode perder)

---

## 9. Transporte Unificado

### 9.1 Princípio

`MIGRATE`, `REDUCE`, `BARRIER`, `SPAWN` são **semanticamente idênticos** entre rig e cluster. O que muda é o transporte:

| Origem → Destino | Transporte | Decisão |
|:---|:---|:---|
| Registrador → Registrador | Silicon | Compilador (estático) |
| L2 → L2 | Cache | Hardware |
| Device → Device (NVLink) | NVLink | `TOPOLOGY[i][j]` |
| Device → Device (PCIe) | PCIe | `TOPOLOGY[i][j]` |
| Device → CPU | PCIe + UPI | `TOPOLOGY[i][j]` |
| Nó → Nó (LAN RDMA) | RDMA | `TOPOLOGY` cross-node |
| Nó → Nó (LAN TCP) | TCP | `TOPOLOGY` cross-node |
| Nó → Nó (WAN) | TCP | `TOPOLOGY` cross-node |

### 9.2 Algoritmo de escolha

```
função choose_transport(src, dst):
    if src.node == dst.node:
        if src.device == dst.device:
            return SILICON
        return MATRIX[src.device][dst.device].transport
    else:
        return CLUSTER_TOPOLOGY[src.node][dst.node].transport
```

Runtime MUST cachear decisões. Runtime MAY invalidar cache em eventos `DEVICE_HEALTH` ou `NODE_SUSPECT`.

### 9.3 Cost model

Runtime MUST considerar:
- **Latência** (ns)
- **Banda** (MB/s)
- **Custo de setup** (handshake, alocação)
- **Contenção** (fila de TX/RX)
- **Energia** (para datacenter verde)

Função de custo:

```
cost = α · latency + β / bandwidth + γ · setup + δ · contention
```

Pesos α, β, γ, δ configuráveis por política.

---

## 10. Feature Bits (RFC-0001 §7, atualização)

Novos bits core (adicionados à §7.1 do RFC-0001):

| Bit | Nome | Descrição | Introduzido |
|:--:|:---|:---|:---|
| 10 | `MULTI_DEVICE` | Suporte a múltiplos devices locais | v2.1 |
| 11 | `DEVICE_TOPOLOGY` | Descoberta de topologia NVLink/PCIe | v2.1 |
| 12 | `UNIFIED_MEMORY` | Memória unificada CPU↔device | v2.1 |
| 13 | `DEVICE_FAILOVER` | Migração automática em falha de device | v2.1 |
| 14 | `NUMA_AWARE` | Alocação ciente de NUMA | v2.1 |
| 15 | `MIG_SLICES` | Suporte a MIG slices | v2.1 |
| 16 | `MULTI_RIG` | Múltiplos rigs no mesmo nó lógico | futuro |
| 17–31 | RESERVED | Reservado | — |

---

## 11. Segurança

### 11.1 Device spoofing

`DEVICE_REGISTRY` é populado pelo runtime; bytecode não pode escrever. Runtime MUST:
- Marcar `DEVICE_REGISTRY` como read-only após boot
- Rejeitar instruções que tentem escrever via `MEMCPY`
- Validar `DEVICE_ID` a cada `DEVICE_SELECT`

### 11.2 Cross-device leak

Um contexto não pode ler memória de outro device sem permissão. Runtime MUST:
- Isolar `SCRATCH_DEVICE[i]` por permissão de contexto
- Validar acesso cross-device via `SP9` (afinidade)
- Logar violações como eventos de segurança

### 11.3 GPU MIG isolation

Se `MIG_SLICES` ativo, runtime MUST:
- Tratar cada MIG slice como device separado em `DEVICE_REGISTRY`
- Isolar memória e compute entre slices
- Rejeitar acesso cross-slice sem capability explícita

### 11.4 Poisoned topology

Topologia forjada pode direcionar tráfego por caminho ruim. Runtime MUST:
- Validar `MATRIX[i][j]` contra APIs nativas
- Rejeitar entradas com valores impossíveis (> 1s de latência, > 10 TB/s)
- Logar inconsistências

### 11.5 DoS via failover

Failover automático pode ser explorado. Runtime MUST:
- Limitar failovers a 1 por contexto por 10s
- Rejeitar failover se destino já tem 90% de carga
- Notificar observabilidade via `TRACE_EVENT`

---

## 12. IANA Considerations

Este RFC estabelece dois registros:

| Registro | Faixa | RFC que atualiza |
|:---|:---|:---|
| **M³-AVM Device Kind Registry** | u8 (0–255) | Este RFC §3.1 |
| **M³-AVM Transport Registry** | u8 (0–255) | Este RFC §9.1 |

Novas entradas via RFC Standard Track. Entradas nunca removidas.

---

## 13. Reference Implementation

Estrutura esperada no repo:

```
src/
  device/
    mod.rs           — DEVICE_REGISTRY, device discovery
    nvml.rs          — NVIDIA discovery via NVML
    rocm.rs          — AMD discovery via ROCm SMI
    level_zero.rs    — Intel discovery via Level Zero
    cpu_topology.rs  — CPU/NUMA discovery via /sys
    topology.rs      — TOPOLOGY matrix
    transport.rs     — transport selection
    scheduler.rs     — device-aware EDF
  opcodes.rs         — 0xB7–0xBF decode/encode
  vm.rs              — dispatcher arms for 0xB7–0xBF
  memory.rs          — SCRATCH_DEVICE[i] via Region Table
tests/
  rfc0002/
    device_discovery.rs
    topology_validation.rs
    opcode_roundtrip.rs
    scheduler_failover.rs
```

Testes de conformidade obrigatórios (RFC-0001 §8.3 estendido):

1. `test_device_registry_populated` — registry populado no boot
2. `test_topology_populated` — topology consistente
3. `test_device_query` — `DEVICE_QUERY` retorna device correto
4. `test_device_select` — `DEVICE_SELECT` muda `SP8`
5. `test_shard_tp` — tensor 1GB shardado em 4 GPUs
6. `test_reduce_local_nvlink` — allreduce via NVLink
7. `test_migrate_local` — migração device↔device
8. `test_barrier_local` — barreira intra-rig
9. `test_device_failover` — GPU falha, contexto migra
10. `test_scheduler_device_aware` — EDF respeita afinidade

---

## 14. Changelog

| Versão | Data | Mudanças |
|:---|:---|:---|
| 0002-00 | 2026-09-10 | Especificação inicial |

---

## 15. Referências

- **ISA v2.0** — `docs/ISA_V2.md`
- **RFC-0001** — Extension Mechanism for M³-AVM ISA
- **PLANO_AVM_CLUSTER.md** — Cluster (integração com LCT)
- **CUDA Programming Guide** — device model
- **ROCm Documentation** — AMD device model
- **NVML API Reference** — NVIDIA Management Library
- **Linux NUMA API** — `/sys/devices/system/node`
- **PCIe Base Specification 5.0/6.0** — barramento
- **NVLink 4.0/5.0** — interconexão NVIDIA
- **CXL 3.0 Specification** — memória compartilhada

---

## Appendix A — Exemplo completo: Llama-70B TP=4

**Rig alvo:** 2 CPUs + 4 GPUs H100 + NVSwitch.

```asm
; 1. Descoberta
DEVICE_QUERY rGpu0, KIND=GPU, MIN_MEM_MB=70000
DEVICE_QUERY rGpu1, KIND=GPU, MIN_MEM_MB=70000, EXCLUDE=rGpu0
DEVICE_QUERY rGpu2, KIND=GPU, MIN_MEM_MB=70000, EXCLUDE=rGpu0,rGpu1
DEVICE_QUERY rGpu3, KIND=GPU, MIN_MEM_MB=70000, EXCLUDE=rGpu0,rGpu1,rGpu2

; 2. Seleção de device
DEVICE_SELECT rStatus, DEVICE=rGpu0

; 3. Carregamento do modelo
LOAD_MODEL rModel, "llama-70b-Q4_K.gguf" QUANT=Q4_K

; 4. Sharding em TP=4
SHARD rShards, rModel, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3], STRATEGY=TP

; 5. Contexto de inferência
SPAWN_CONTEXT rWorker, ENTRY=forward, DEVICE_AFFINITY=0b111100

forward:
    ; QKV espalhados
    MATVEC rQ, rX, rWq DEVICE=rGpu0
    MATVEC rK, rX, rWk DEVICE=rGpu1
    MATVEC rV, rX, rWv DEVICE=rGpu2

    ; Allgather via NVLink
    REDUCE_LOCAL rQ, OP=ALLGATHER, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3], TRANSPORT=NVLINK
    REDUCE_LOCAL rK, OP=ALLGATHER, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3], TRANSPORT=NVLINK
    REDUCE_LOCAL rV, OP=ALLGATHER, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3], TRANSPORT=NVLINK

    ; Atenção
    ATTN rAttn, rQ, rK, rV NHEADS=64 HEADS_PER_DEVICE=16

    ; FFN + allreduce
    FFN rOut, rAttn DEVICE=SPLIT_TP
    REDUCE_LOCAL rOut, OP=ALLREDUCE, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3]

    ; Verificação de saúde
    DEVICE_HEALTH rHealth, DEVICE=0xFF
    COMPARE rHealth, 0
    IF_GREATER rHealth, failover

    JUMP forward

failover:
    ; GPU degradada → migrar
    MIGRATE_LOCAL rShards[rFailed], SRC=rFailed, DST=rSpare
    JUMP forward
```

**Comportamento esperado:**
- Allgather em NVLink: ~900 GB/s
- Latência de collective: ~5 µs para 4 GPUs
- Failover: <100 ms

---

## Appendix B — Exemplo: MoE com Expert Parallelism

**Modelo:** Mixtral-8x7B (8 experts) em 4 GPUs.

```asm
; 2 experts por GPU
SHARD rExperts, rMoEWeights, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3], STRATEGY=EP

forward:
    ; Gate
    MATVEC rGate, rX, rWgate DEVICE=rGpu0

    ; Top-2 experts
    SAMPLE rTopk, rGate TOPK=2

    ; Dispatch via ALLTOALL
    REDUCE_LOCAL rX, OP=ALLTOALL, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3]

    ; Expert FFN local
    FFN rOut, rX DEVICE=LOCAL

    ; Combine
    REDUCE_LOCAL rOut, OP=ALLTOALL, DEVICES=[rGpu0,rGpu1,rGpu2,rGpu3]

    JUMP forward
```

**Ganho:** 4× throughput em MoE vs single-GPU.

---

## Appendix C — Checklist de conformidade

Um runtime é **conforme a RFC-0002** se:

- [ ] Popula `DEVICE_REGISTRY` no boot (mínimo: CPU boot)
- [ ] Popula `TOPOLOGY` (mínimo: matriz identidade)
- [ ] Implementa `SP8` e `SP9`
- [ ] Implementa opcodes `0xB7–0xBF` (mesmo que parcialmente)
- [ ] Implementa `SCRATCH_DEVICE[i]` via Region Table
- [ ] Scheduler respeita `device_affinity`
- [ ] Detecta falha de device e responde
- [ ] Seleciona transporte por `TOPOLOGY`
- [ ] Passa nos 10 testes de §13
- [ ] Não quebra bytecode v2.0 sem LCT

---

**Fim do RFC-0002.**

Este documento transforma a M³-AVM de "VM de workstation" em "sistema operacional de inferência" para qualquer rig — de laptop a pod de treino com 64 GPUs. A regra permanece: **rig é cluster, cluster é rig, a diferença é só o barramento**.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
