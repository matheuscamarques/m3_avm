# RFC-0039 — System local shims (`0x4A–0x4D`, v1.16; `0xA0+` DRAFT 64B fica)

```text
Status      : IMPLEMENTED (V-3A mini, local 32B; 0xA0+ X-forms DRAFT)
Category    : Standards Track
Updates     : ESPEC §5/§16 + ESPEC-V2 §3.14 (PARTIAL: 0x4A-0x4D IMPL, 0xA0+ stays DRAFT)
            : ISA v1.15 → v1.16 (família com encoding, sem re-encoding)
Feature Bit : M3BC_REQUIRED_HAS_V3_SYSTEM = 1<<51 (REQUIRED, não OPTIONAL)
Bump        : minor v1.15 → v1.16
```

## Abstract

Quatro shims 32B locais para o MVP voice-loop operarem sem esperar o
freeze dual-mode 64B: `LOAD_MODEL` (registry stub), `SPAWN_CONTEXT`
(fork-like com entry label + prio + model handle), `KILL_CONTEXT`
(terminação), `SET_MODEL` (troca de modelo por contexto). Os 64B
`0xA0–0xA9` (`LOAD_MODEL`/`SPAWN_CONTEXT`/`KILL_CONTEXT`/`SET_MODEL`
etc. com LAMPORT/DEADLINE) seguem `DRAFT` em `ESPEC-V2 §3.14` — os
shims são os operacionais até o freeze (precedente `0x1A–0x1D`
vs `0x80+` em `ESPEC-V2 §3.12`).

## Motivation

`programs/voice_loop_demo.m3asm` já roda fim-a-fim com zero opcode
novo (V-3/G6, `test_planoV3_voice_loop` 5 invariantes). Falta ao ISA,
porém, a capacidade de *nomear* modelos e *posicionar* contextos a
partir de `.m3asm` — sem isso `VISAO_PREMIUM_250GB`/`MVP_ENGLISH_TUTOR`
seguem ficção. O caminho 64B completo (fetch 64B + `0xA0` payloads
com `PATH`/`QUANT`/`LAMPORT`) bloqueia `V-4` (pesos reais) por 2 turnos.
Os shims 32B unlockam `V-3` em 1 turno, sem re-encoding, com bit
REQUIRED 51 (firewall idêntico ao `V-2 bit 50`).

## Specification

Convenções herdadas: `rdest` = resultado, `0xFF` = ausente, payload 26B
LE, `f32` só, `NaN/Inf` veta, braços `RFC-0008` desde o dia 1.

```text
LOAD_MODEL rD, MODEL=n                 ; 0x4A
  n u16 >=1 (payload[0..2] LE, .equ ok). rD <- handle u64 monotônico (>0,
  mapeia p/ model_id n em `Vm.models`). Sem path (harness mapeia n->arquivo
  GGUF externa; EMPTY path é follow-up 64B). Sem payload extra.

SPAWN_CONTEXT rD, LABEL, PRIO [, MODEL=rM] ; 0x4B
  LABEL = rótulo 2-pass (entry_pc u128 em payload[0..16] LE, como FORK).
  PRIO = RED|BLUE|GREEN (flags 0b00/01/10, como FORK; NOTIFY não aplicável).
  MODEL=rM opcional: registrador com handle de LOAD_MODEL; 0xFF = nenhum.
  rD <- child ctx_id u64 (>0). Cria contexto fresco (regs zero) em LABEL,
  prioridade PRIO, model_handle copiado se houver. Entry deve alinhar com
  início de instrução (pc_offsets); senão erro alto.

KILL_CONTEXT rTarget                    ; 0x4C
  rTarget com ctx_id u64; alvo 0 ou inexistente => erro alto (não warn).
  Remove do scheduler.

SET_MODEL rCtx, rModel                 ; 0x4D
  rCtx com ctx_id, rModel com handle. Ambos devem existir; senão erro alto.
  Troca `ctx.model_handle` (FORK herda).
```

`Vm`: `models: HashMap<u64,u16>` + `next_model_handle: u64` (1..),
`model_handle: u64` por `Context` (0 = nenhum). `snapshot`/`restore` e
`fork` (`ABORT`/`SNAPSHOT`/`RESTORE` vs `FORK` herança) cobrem `models`
como cobrem `rag_store` (4 casas: `FORK`/`SNAPSHOT` clone, `ABORT`/`RESTORE`
pop versionado).

## Out-of-band (não mexe nesta RFC)

* `0xA0 LOAD_MODEL` 64B, `0xA2 SPAWN_CONTEXT` 64B, `0xA3 KILL_CONTEXT`,
  `0xA7 SET_MODEL`, `0xA8 GET_MODEL`, `0xA9 MODEL_SWITCH`, etc. — seguem
  `DRAFT` em `ESPEC-V2 §3.14`. Requerem fetch 64B + payloads com `PATH`/
  `QUANT`/`LAMPORT`/`DEADLINE` (W1-remainder follow-up).
* `0x4E-0x4F` ficam `RESERVED` (futuro `GET_MODEL`/`UNLOAD_MODEL` 32B se
  precisar; hoje `KILL` + `SET` bastam para o MVP).

## Bump

`v1.15 (112)` → **`v1.16 (116)`** (`+4` shims). `M3BC_REQUIRED_HAS_V3_SYSTEM: u64 = 1<<51`
(`bit 51`, livre — `1<<50` já é `V-2`; `1<<49` é `OPTIONAL` `.data`).
`M3BC_SUPPORTED_REQUIRED` passa a `V2|V3`. Arquivos antigos: zero
`0x4A-0x4D` em uso (grep) => byte-idênticos; decoder antigo rejeita como
`UnknownOpcode` (nunca misdecode); loader `.m3bc` com `REQUIRED 51` em
runtime `v1.15` é rejeitado no `header.required` antes do fetch.

## Backwards Compatibility

Só adição. Corpus `44/44` passa a `45/45` com `programs/system_demo.m3asm`;
onde não usa `0x4A-0x4D` é byte-idêntico. `.m3bc` `V-3A` em runtime `v1.15`:
rejeitado por `REQUIRED` (com bits nomeados) ou por `UnknownOpcode`.

## Security Considerations

* Bounds: `MODEL` u16 `>=1`, `handle` `u64` checks `try_from` nunca truncar,
  entry `u128` valida alinhamento 32B + existência em `pc_offsets`.
* Sanitização: `F32`-only não aplica (handles são inteiros); `NaN/Inf` não
  entram (handles são `u64`).
* DoS: `models` sem quota nesta RFC (emulador; quota é `G8` follow-up).
* Determinismo: zero `rand`; `SPAWN` ordem `next_id` monotônica (precedente
  `FORK`).

## Reference Implementation

* `src/opcodes.rs`: `OP_LOAD_MODEL=0x4A` etc. + ctors + 4 braços + 11 bad
  forms + `uses_v3_system` (gate container).
* `src/context.rs`: `Context.model_handle: u64` (0 = nenhum; `FORK` herda).
* `src/vm.rs`: `Vm.models`/`next_model_handle`/`models_snapshots` + 4
  execs + `VmStats::{load,spawn,kill,set}_model_execs` + snapshot em
  `FORK`/`SNAPSHOT`/`ABORT`/`RESTORE`.
* `src/m3bc.rs`: `M3BC_REQUIRED_HAS_V3_SYSTEM=1<<51` em `SUPPORTED_REQUIRED`.
* `benches/`: sem bench novo (stub <1µs; `rag_bench` já cobre retrieval).
* `programs/system_demo.m3asm` + `vm::test_system_ops_mini` (11 bad forms,
  goldens `LOAD 1->1,2->2`, `SPAWN->child`, `SET`, `KILL`, snapshot `models`).
* `docs/HANDOFF.md` v4 + `ESPEC§16/19` + `ESPEC-V2§3.14` PARTIAL + `PLANO_VISAO §2`.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0039-00 | 2026-09-11 | IMPLEMENTED: 4 shims 0x4A-0x4D local 32B, bit 51 REQUIRED, bump v1.16, V-3A mini |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
