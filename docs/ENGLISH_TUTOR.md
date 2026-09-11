# ENGLISH_TUTOR — Tutor de inglês no emulador (composição, não arquitetura)

```text
Status:   CONTRATO-EXECUTÁVEL (doc de composição; normativo só o que o teste asserir)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Base:     programs/voice_loop_demo.m3asm (fluxo) + V-1b `.str` (filler library)
License:  AGPL-3.0-or-later (see LICENSE)
```

> Composição sobre peças existentes (zero opcode, zero sintaxe, zero
> região, zero bit — por isso doc, não RFC). Fora: RAG (V-2), LoRA,
> avatar, áudio real (stub até `m3-audio` v0.1), >1 role-play, sessão >30s.

## 1. Fases (1 role-play: interview; `PHASE` injetado pelo harness 0/1/2)

| PHASE | Nome | Conteúdo stub |
|:---:|:---|:---|
| 0 | warm-up | conversa casual; sem correção (só response) |
| 1 | role-play interview | correção + response (ordem pelo modo) |
| 2 | review | repete erros do dia; correção + response (ordem pelo modo) |

## 2. Modos (`TEACH_MODE` injetado: 0=explicit, 1=implicit)

- **explicit** (B1): emite correção ANTES da resposta.
- **implicit** (B2/C1): emite resposta ANTES da correção.
- Ordem observável: blocos de emissão exclusivos (`rExpMark`/`rImpMark`,
  só um corre por run) + flag `rOrder` (1=correction-first, 2=response-first).

## 3. Invariantes (teste: 3 fases × 2 modos = 6 runs)

1. Sessão completa roda; estado final IDLE (por run de fase).
2. Ordem correta por modo (`rOrder` + marcador exclusivo do bloco).
3. Filler não vaza p/ output (`audio_out` sem marcador 9.999).
4. Filler não vaza p/ transcript (codes sem marcador; transcript real
   de tokens vem com STT/Mimi-real — aqui codes = "o que o usuário disse").
5. Abort no role-play restaura (`rTurnState`→INCOMPLETE, canário intacto,
   coerência funcional; rewind de mapa/ssm/KV é unitário RFC-0011/0003).

## 4. Filler library (`.str`, 1 frase por fase; endereço selecionado, nunca emitido)

`filler_warmup` / `filler_interview` / `filler_review` — programa faz
`LOADI` do endereço da frase da fase (prova seleção); bytes nunca entram
em tensor de output/transcript (invariantes 3–4 guardam).
