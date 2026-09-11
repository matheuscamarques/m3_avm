# TURN_STATE — máquina de turno (pausa/retomada)

```text
Status:   META-ALVO (doc de comportamento, NÃO-normativo, NÃO executado além da demo)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Companion: docs/PLANO_VISAO.md (Fase V-0) · docs/PAUSA_DE_PENSAMENTO.md · programs/pause_turn_demo.m3asm
License:  AGPL-3.0-or-later (see LICENSE)
```

> **Contrato de honestidade (ESPEC §1.3):** o único comportamento
> executável hoje é `programs/pause_turn_demo.m3asm` (RFC-0035,
> `test_planoV0_pause_paths` verde). Todo o resto abaixo é alvo.

## 1. Estados

| Código (demo) | Nome conceitual | Significado | Onde existe |
|:---:|:---|:---|:---|
| — | IDLE | sem turno ativo, aguardando `SENSE USER_INPUT` | alvo (não há estado IDLE na demo; `LOADI r0, 1` entra direto em INCOMPLETE) |
| 1 | INCOMPLETE | fala parcial recebida, aguardando continuação ou timeout | `pause_turn_demo.m3asm` (`LOADI r0, 1`, loop `SENSE` + `IF_INTERRUPT`) |
| 2 | THINKING | janela de silêncio sendo medida (`STEPS` vs deadline) | alvo (na demo o loop INCOMPLETE acumula esse papel; não há valor `r0=2` distinto) |
| 3 | RESUMED | barge-in / continuação detectada, deadline estendido +64 | `pause_turn_demo.m3asm` (`GOT_SPEECH: LOADI r0, 3`, `ADD_IMM r2, r7 IMM=64`) |
| 4 | TURN_END / COMPLETE | turno fechado: timeout (sem fala) ou HALT após processamento | `pause_turn_demo.m3asm` (`LOADI r0, 4`, `JUMP DONE`, `HALT`; teste asserts `r0==4` no caminho quieto) |

## 2. Transições (demo)

```text
INCOMPLETE(1) --SENSE USER_INPUT tem evento--> RESUMED(3) --JUMP LOOP--> INCOMPLETE(1)
INCOMPLETE(1) --64 iters sem evento (STEPS>=deadline)--> TURN_END(4) --> HALT
RESUMED(3)    --deadline+=64--> INCOMPLETE(1)  (mesmo loop; r0 fica em 3 até próximo timeout ou HALT)
```

- Timeout determinístico via `STEPS` (contador de retiradas, não wall-clock).
- `VAD_DETECT` pontua frames sintéticos em paralelo (assert no teste: `0.5`).
- Driver: fila `USER_INPUT` via `push_input` no teste.

## 3. Gaps até o alvo completo

1. Sem `IDLE` distinto: a demo começa em `INCOMPLETE`; o loop voz real (G6) precisará do estado de repouso + `SENSE` bloqueante/não-bloqueante bem definido.
2. Sem `THINKING=2` distinto: hoje o silêncio é medido dentro do loop `INCOMPLETE`; separar exige 1 imediato novo, sem opcode.
3. Sem `COMPLETE` distinto de `TURN_END`: hoje `4` cobre os dois; o voice-loop (V-3) precisará distinguir "abandono por timeout" de "resposta emitida".
4. Reformulação/abandono (`PAUSA` Exps. 2–4) e antecipação RAG seguem alvos V-2/V-3 (`RAG_SEARCH` 🟡, system ops G4).

## 4. Critério de promoção

Qualquer fatia que mexa nesta máquina só promove para `programs/`
quando montar (gate RFC-0008) e tiver teste verde como
`test_planoV0_pause_paths`; caso contrário vive como listagem-alvo
nos docs da família.
