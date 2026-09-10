# PLANO — AVM-Cluster: rede estilo Erlang/BEAM no núcleo da M³-AVM

**Objetivo:** elevar a M³-AVM de runtime local a VM distribuída: preempção (barge-in) e migração de estado entre nós em sub-milissegundos, com rede implementada **no runtime Rust (estilo BEAM), não em biblioteca de usuário**.
**ISA:** 4 opcodes novos `0x1A..0x1D` (`REMOTE_SPAWN, SIGNAL, SEND_TENSOR, BARRIER`), instrução fixa 32B preservada.
**Data:** 2026-09-09

> Síntese das duas propostas discutidas: (1) cluster estilo Erlang com `REMOTE_SPAWN/SEND_TENSOR/SIGNAL` para barge-in distribuído Edge→Compute; (2) correção BEAM — rede no núcleo da VM (sockets, PIDs, serialização, preempção fora-de-banda em Rust) + mapa final congelado `0x1B=SIGNAL / 0x1C=SEND_TENSOR / 0x1D=BARRIER`.

## 0. Contexto confirmado (lido no repo)

* ISA atual ocupa `0x00 HALT, 0x01..0x19` (26 opcodes), `0x1E..0x25` (8 universal), `0x38 KV_TRUNCATE`, `0x60..0x79` (16 RNG/hash), `0xFF NOP` — `src/opcodes.rs:28-57`. **`0x1A..0x1D` livres, sem colisão.**
* Formato real: `opcode 1B + flags 1B + rdest/rsrc1-3 4B + payload 26B (bytes 6..31)` — `src/opcodes.rs:2-10`. Qualquer tabela de payload de rede precisa caber em **26B**, não 28B.
* Bus local: `watch<Option<InterruptSignal>> + 2x broadcast` — `src/bus.rs:40-48`. Não atravessa processo; cluster entra como ponte, sem reescrever o bus.
* Scheduler estrito `Red > Blue > Green`, 3 FIFOs locais, `Context.id: u64` local — `src/context.rs:15,171-181`. Falta `GlobalPid = (node_id, ctx_id)` e tabela de roteamento.
* Memória 4 regiões `GLOBAL 0x00 / TEMPORAL 0x10 / PERSISTENTE 0x20 mmap / KV_CACHE 0x30` — `src/memory.rs:23-27`; backend `Cpu|Gpu` em `src/vm.rs:92`. `SEND_TENSOR` decide o que viaja: `TensorMeta + bytes` (header 32B anuncia, bulk em chunks).
* Runtime `Vm` monolocal + `Reactor` opt-in via `tokio::select!` sobre os buses — `src/vm.rs:1-250`, `src/reactor.rs:42-94`. Ponto de checagem de interrupção já existe (`has_interrupt()`, `ATTN` por head) — rede só adiciona flag atômica lida no mesmo ponto.
* Tokio hoje sem rede: `features=[sync,rt,rt-multi-thread,macros,time]` — `Cargo.toml:34`. Cluster exige `net,io-util` + `bytes`.
* Rollback cirúrgico (entropia + fallback 5 tokens) já existe — `src/rollback.rs:104-128`. ABORT remoto reutiliza `restore()/snapshot()` (`src/memory.rs`) + reset `ssm_states`.
* Gap de docs: `README` cita `docs/ISA.md` inexistente (`docs/` só tem `MOSHI_MAP.md`, `PLANO_MOSHI_NATIVO.md`).

## 1. Princípio BEAM espelhado (o que fica no Rust vs. no `.m3asm`)

| Camada | Na VM Rust (como `dist_util.c`/`epoll` da BEAM) | No `.m3asm` (como sintaxe Erlang) |
|---|---|---|
| Sockets/transporte | `src/network.rs`: TCP persistente, handshake, filas urgente/bulk, heartbeat | `SIGNAL NODE="b" ...`, `REMOTE_SPAWN ...` — sem socket visível |
| PIDs | `GlobalPid(node_id, ctx_id)`; `node_id==0` executa local, senão serializa 32B p/ fila TX | `Pid ! Msg` equivalente: programa não distingue local/remoto |
| Serialização | 32B instr LE + bulk em chunks (futura ETF/QUIC/RDMA) | passa `Rs`/`Rd`/label como operandos |
| Preempção | loop RX escreve flag atômica; `ATTN/SSM_SCAN` checam 1 `load(Relaxed)` por chunk; `SIGNAL` fura fila bulk | `SENSE VAD → SIGNAL → IF_INTERRUPT/HANDLE_ABORT` inalterado |

Caminho disruptivo alvo: `SENSE → OP_SIGNAL no socket → load atômico no peer → ABORT+rollback` em **<1ms LAN** (vs 50–200ms do caminho Python/gRPC→CUDA-cancel).

## 2. ISA congelada — 4 opcodes `0x1A..0x1D`

```rust
pub const OP_REMOTE_SPAWN: u8 = 0x1A; // cria contexto AVM em nó remoto
pub const OP_SIGNAL:       u8 = 0x1B; // controle fora-de-banda: ABORT/FORK/HALT/PING
pub const OP_SEND_TENSOR:  u8 = 0x1C; // transfere/mapeia tensor, KV-cache, h_t, frames
pub const OP_BARRIER:      u8 = 0x1D; // sincronização determinística entre nós
```

### 2.1 Codificação nos 26B de payload (ajuste ao layout real)

* **`OP_SIGNAL (0x1B)`** — `flags bit0 = prioridade máxima/interrupt`. `rsrc1` = reg com kind (`0=ABORT, 1=FORK_REQ, 2=HALT/KILL, 3=PING`). `payload[0..4]` = `node_id u32 LE`, `[4..12]` = `ctx_id u64`, `[12..20]` = `t_interrupt/seq u64` (Lamport p/ ordenação — wall-clock só p/ métrica, cf. gap de relógio `README §6`), `[20..26]` reservado. `rdest` = status/ack local.
* **`OP_SEND_TENSOR (0x1C)`** — `flags bit0 = 0 TCP / 1 RDMA-hint` (MVP sempre TCP; flag reserva caminho futuro). `rdest` = reg origem local, `rsrc1` = reg destino remoto (`0xFF` = alocar). `payload[0..4]` = `node_id u32`, `[4..12]` = `byte_offset u64`, `[12..16]` = `byte_len u32`, `[16]` = `mode (0=COPY, 1=MOVE)`. Bulk fora da instrução, em chunks 64KiB.
* **`OP_REMOTE_SPAWN (0x1A)`** — `payload[0..4]` = `node_id u32`, `[4..12]` = `entry_pc u64` (label resolvido no assembler 2-pass), `[12]` = `prio (0=GREEN,1=BLUE,2=RED)`. `Rd` recebe `ctx_id` remoto ou `u128::MAX` em falha.
* **`OP_BARRIER (0x1D)`** — `payload[0..4]` = `barrier_id u32`, `[4..6]` = `expected u16`, `[6..8]` = `timeout_ms u16`, `[8..12]` = `epoch/mask`. Bloqueia o contexto (`Blocked` + wakeup via `SchedSignal`), nunca gira scheduler. Timeout vira `NACK` + desbloqueio com erro.

Sintaxe assembler:

```asm
SIGNAL NODE="compute" CTX=r1 KIND=ABORT -> r2
SEND_TENSOR r5 -> NODE="compute" DEST=r7 OFF=0 LEN=65536 MOVE
REMOTE_SPAWN NODE="compute" ENTRY=MAIN_LOOP GREEN -> r3
BARRIER id=7 EXPECT=2 TIMEOUT=500
```

`NODE="nome"` via tabela de roteamento do nó (`nome -> node_id u32`); literal numérico também aceito. Prompt rico (`new_prompt`, `target_token_index` de `tese.md §6`) **não** viaja no `SIGNAL` — vai por `SEND_TENSOR` prévio/canal lateral, mantendo o sinal em 32B.

## 3. Arquitetura runtime

```
Nó A (Edge/PersonaPlex)                    Nó B (Compute Mamba/Transformer)
Vm + Scheduler local                       Vm + Scheduler local
Bus local (watch/bcast)                    Bus local
ClusterNode/Driver TCP  <--- 32B SIGNAL -- ClusterNode/Driver TCP
 SENSE VAD → ABORT local + SIGNAL           flag atômica → ABORT + restore KV/SSM
```

* Topologia MVP: **mesh estática 2..N nós** via `--peer` (sem gossip/SWIM no MVP).
* Identidade: `node_id u32` (hash do `node_name` persistido em `m3_node.id`), `GlobalPid(node_id, ctx_id)`. Cookie estilo Erlang (`--cookie`/`M3_COOKIE`, comparação constant-time) no handshake. TLS/Noise = follow-up.
* Regra do dispatcher (`src/vm.rs::execute`): `node_id == LOCAL (0)` → inline local; `!=` → serializa 32B p/ fila TX e retorna (SPAWN/BARRIER esperam via `Blocked`+wakeup, com timeout).
* Filas do driver: **urgente** (`SIGNAL/BARRIER/HEARTBEAT`) fura **bulk** (`SEND_TENSOR/SPAWN`), como `EXIT/KILL` furam mailbox na BEAM.
* `Red` atravessa nó, mas `Red` remoto nunca preempta `Red` local em execução (anti-inversão inter-nó).

## 4. Protocolo de fio (MVP TCP; QUIC/RDMA reservados)

```
Handshake: MAGIC "M3C1" + node_id u32 + cookie_len u8 + cookie + listen_addr
Frame:     total_len u32 BE | instr 32B | bulk_len u32 | bulk[bulk_len]
Tipos:     HELLO 0xF0, HEARTBEAT 0xF1, ACK 0xE0, NACK 0xE1, + 0x1A..0x1D corpo
Timeouts:  connect 2s, ack 500ms, heartbeat 200ms, suspect 600ms, dead 2s
```

Heartbeat carrega `free_pct + queue_lengths (R,B,G)` para `REMOTE_SPAWN` escolher peer menos carregado; nó `dead` sai do roteamento e `SPAWN` retenta no próximo vivo (`Let It Crash` mínimo).

## 5. Fases

### F0 — Spec + payload (0.5 sem)
- [ ] Escrever `docs/CLUSTER.md` (fio, handshake, kinds/flags, diagrama Edge→Compute) + criar `docs/ISA.md` (tabela completa `0x00..0x1D,0xFF`).
- [ ] Congelar §2 deste plano (qualquer mudança de byte exige revisão).
- Aceite: revisão da codificação antes de qualquer código.

### F1 — ISA local, sem rede (1 sem)
- [ ] `src/opcodes.rs`: consts, `instr_remote_spawn/signal/send_tensor/barrier()`, accessors `*_params()/set_*`, `mnemonic()`, `parse_line()` (NODE/LABEL/KIND/MOVE/BARRIER), testes roundtrip + labels.
- [ ] `src/vm.rs`: 4 braços no `execute` com `node_id==0` executando inline (ABORT/FORK/BARRIER-locais); 4 contadores em `VmStats`.
- Aceite: `cargo test --lib opcodes::` verde; `SIGNAL` local aborta como `ABORT`.

### F2 — Driver + SIGNAL = barge-in distribuído (1–2 sem)
- [ ] Novo `src/network.rs`: TX (`mpsc` urgente+bulk, TCP `nodelay/keepalive`), RX (`select!` listeners+peers → flag atômica + `Bus::publish_interrupt` + `interrupt_flag` p/ `IF_INTERRUPT` existente).
- [ ] Novo `src/cluster.rs` (~300 linhas) ou módulo em `network.rs`: `NodeId/GlobalPid/ClusterConfig/node_table/ctx_affinity`, handshake cookie.
- [ ] `Cargo.toml`: `tokio += ["net","io-util"]`, `bytes = "1"`; feature `cluster` opcional p/ não quebrar build default.
- [ ] CLI: `run --node-name/--listen/--peer/--cookie` + `cluster ping/signal` p/ debug.
- [ ] Demo `examples/cluster/edge_vad.m3asm` (Nó A) + `compute_reason.m3asm` (Nó B, `IF_INTERRUPT HANDLE_ABORT` com `restore()` KV/SSM existente).
- Aceite: 2 nós em localhost, `t(VAD)→t(abort)` com `M3_PROFILE=1` **<5ms**; `cargo test --lib network::` (loopback ABORT).

### F3 — SPAWN + BARRIER + supervisão (1 sem)
- [ ] `REMOTE_SPAWN`: peer cria contexto (`create_context + enqueue`), devolve id via `ACK`; falha → `u128::MAX` + retry no próximo vivo.
- [ ] `BARRIER`: coordenador conta `ARRIVE(barrier_id)` e difunde `RELEASE`; timeout → `NACK`.
- [ ] `Reactor` assina `ClusterNode` além do `Bus` (sem reescrever `run_nop`).
- Aceite: `tests/cluster_spawn_barrier.rs` (2 VMs em threads Tokio); kill -9 de 1 peer com failover de SPAWN `<2s`.

### F4 — SEND_TENSOR + migração de estado (2 sem)
- [ ] `COPY` depois `MOVE` (=`COPY` + invalidação da origem pós-ACK; documentar como emulação — zero-copy real só com RDMA futuro).
- [ ] Tipos: tensor `GLOBAL`, 1 camada KV (`kv_cache_get_k/v`), `h_t` Mamba (`ssm_states`), frame áudio 1920×f32 (`SENSE_AUDIO_PCM`).
- [ ] Reuso de `rollback.rs` pós-migração (rollback continua entrópico/local).
- Aceite: roundtrip com hash por chunk; `tc delay/loss` sem corrupção; teste speech 80ms.

### F5 — Hardening, benches, docs (1 sem)
- [ ] Fuzz do codec (`cargo fuzz` ou proptest do frame), auditoria do cookie, `BARRIER` sob partição (sem deadlock: sempre timeout).
- [ ] `benches/dist_barge.rs`: latência ABORT distribuído vs baseline local (`src/qa.rs:664` mede ~25ms p/ ABORT 32x32 hoje); IPS com rede idle (overhead <5%).
- [ ] Demo `moshi_loop` distribuído (Edge áudio ↔ Compute) + `docs/CLUSTER.md` final honesto (TCP ≠ zero-copy; AGPL §13 p/ hosted inference).
- Aceite: `cargo test --lib` 124+novos verdes; doc sem claims além do medido.

## 6. Ordem e não-objetivos

Ordem: `F0 → F1 → F2 → F3 → F4 → F5`. Não pular F2 — sem `SIGNAL` fora-de-banda não há tese.
Não fazer agora: QUIC (`quinn`)/RDMA real, gossip/SWIM completo, TLS mútuo, `BSR/FlashAttention sparse`, `GEMV.wgsl Q4_K`.

## 7. Comandos úteis

```bash
cargo test --lib opcodes:: network:: cluster:: -- --nocapture
cargo run --bin m3_avm -- run examples/cluster/edge_vad.m3asm --node-name edge --listen 127.0.0.1:5001 --peer 127.0.0.1:5002 --cookie segredo --trace
cargo run --bin m3_avm -- run examples/cluster/compute_reason.m3asm --node-name compute --listen 127.0.0.1:5002 --peer 127.0.0.1:5001 --cookie segredo
M3_PROFILE=1 cargo run --release -- run examples/cluster/compute_reason.m3asm --max-steps 1000
cargo bench --bench dist_barge -- --quick
```

## 8. Decisões pendentes (travar em F0)

1. Mapa `0x1B=SIGNAL / 0x1C=SEND_TENSOR / 0x1D=BARRIER` congelado? (Sim recomendado — §2.)
2. MVP prova valor só com `SIGNAL+SPAWN`, com `SEND_TENSOR COPY` no mesmo marco ou em F4?
3. TCP-Tokio puro basta p/ preprint (QUIC/RDMA como flag reservada)?
4. Demo de referência: par fixo Edge→Compute ou mesh N-nós genérico já no MVP?

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
