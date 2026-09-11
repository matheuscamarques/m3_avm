# Filler Speech — Pensar em Voz Alta no M³-AVM Premium

```text
Status:   DESIGN-ALVO (proposta de mecanismo, NÃO-normativa, NÃO implementada)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Companion: docs/EXPERIENCIA_PREMIUM.md (a sensação) + docs/PAUSA_DE_PENSAMENTO.md (o outro lado do silêncio)
License:  AGPL-3.0-or-later (see LICENSE)
```

> **Contrato de honestidade (ESPEC §1.3):** latências e comportamentos
> abaixo são **metas de design**. O §0 lista as correções factuais
> aplicadas ao rascunho; o §5 mostra o que sai com o ISA de hoje.
>
> **O par:** `PAUSA_DE_PENSAMENTO.md` cobre o silêncio do **usuário**
> (pausa, retomada, abandono). Este documento cobre o silêncio do
> **modelo** (os 700ms–2s de computação). Os dois juntos eliminam o
> silêncio morto da conversa.

## §0. Correções aplicadas ao rascunho

1. **`CLOCK_QUERY 🟡 0x78` errado (de novo):** relógio é `CYCLES_COUNT`
   (✅ `0x6A`). Convertido em todo o documento.
2. **`SUB` não existe** (🔴). Timeouts usam a construção `ADD` +
   `COMPARE PRED=` (§5 de `PAUSA_DE_PENSAMENTO.md`).
3. **`KIND=FILLER_START/STOP` não existem** — os 4 kinds de `SIGNAL` são
   congelados. O mecanismo correto é `PING` + palavra de controle (§5.1).
   Esforço continua trivial, mas **sem tocar encoding congelado**.
4. **`SENSE rSignal, LOCAL_SIGNAL` / `rSignal.kind` / `rSignal.phrase_id`**
   não existem (🔴). Padrão válido hoje: `PING` → `IF_INTERRUPT` no
   destino + leitura da palavra de controle em memória.
5. **`LOADSTR`, `.str`, `.data/.equ/.text`, `CALL`, `CROSSFADE`,
   `STREAM ... CHANNEL=`** — inexistentes (🔴). Crossfade **sai com ops
   de hoje** (`FILL`+`MUL`+`ADD`, §5.2); biblioteca de frases precisa
   init multivalor (follow-up RFC-0019, §5.3).
6. **`DEPFORMER ✅ mesmo modelo` → 🟡.** Mesmo *modelo*, mas o opcode
   (`0x44`) é DRAFT, Fase 5.
7. **`FOREST ... TREES=500` ✅** (kwargs `TREES=`/`DEPTH=` existem), mas
   `MODE=PROBABILITY` não — só `VOTE`/`MEAN`. E tabelas de 500 árvores
   esbarram no mesmo init-multivalor do item 5.
8. **Comparações com float imediato** (`COMPARE rComplexity, 0.4`) —
   usar inteiros escalados (RFC-0007/0008).
9. **`random(1, N)`** — `RNG_UNIFORM` (✅ `0x63`) existe, mas índice
   float→int precisa `CAST` (🟡 `0x67`, Fase 4) ou escada de `COMPARE`
   (§5.4).
10. **"20ms (XGBoost)" e "~1 semana"** — metas/estimativa do autor, não
    medições. `FOREST` 500×8 nunca foi bencheado (entrar na fila de
    benches por wave, ESPEC-V2 §15.3).

Legenda: ✅ IMPL · 🟡 DRAFT · 🔵 proposto-novo · 🔴 inexistente/bloqueado.

---

Quando você pergunta algo difícil para um amigo, ele não fica em silêncio por 2 segundos. Ele diz:

"Hmm... boa pergunta... deixa eu pensar..."

E enquanto ele fala isso, o cérebro dele está processando a resposta. Quando ele termina a frase de preenchimento, a resposta já está pronta. E ele continua naturalmente.

Isso se chama "filler speech" em linguística. É o que humanos fazem para não deixar silêncio no ar. E é o alvo deste documento na M³-AVM.

## O insight central

O tempo que o Llama-405B leva para processar (~700ms–2s, META) é exatamente o tempo que um humano leva para dizer "Hmm, deixa eu pensar...".

Não é coincidência. É a mesma janela de tempo.

Então em vez de silêncio, o sistema usa esse tempo para gerar fala natural.

## A arquitetura (alvo)

```text
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Thread RED (áudio)                                             │
│  ├─ Detecta pergunta complexa em ~20ms (XGBoost) [META]        │
│  ├─ Dispara FILLER_START imediatamente                          │
│  └─ Começa a tocar: "Hmm... boa pergunta..."                    │
│                                                                 │
│  Thread BLUE (raciocínio)                                       │
│  ├─ Em paralelo, Llama-405B processa                            │
│  ├─ 700ms–2s: gera tokens da resposta real  [META]             │
│  └─ Signal: FILLER_STOP quando primeiro token pronto            │
│                                                                 │
│  Thread de transição                                            │
│  └─ Quando FILLER_STOP chega, o áudio troca suavemente          │
│     de "deixa eu pensar..." para a resposta real                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

O usuário nunca fica em silêncio. Ele ouve pensamento humano. E a resposta chega quando está pronta.

> Mecanismo real dos "sinais": `PING` + palavra de controle (§5.1) —
> nenhum kind novo, nenhum encoding tocado.

## O Assembly (alvo, não monta hoje)

```asm
; ============================================================
; programs/thinking_assistant.m3asm  [ALVO — ver §0.2 da VISÃO]
; Voice assistant with natural filler speech
; ============================================================

.data                                                     ; 🔴 assembler
    ; Library of natural "thinking" phrases (PT-BR)
    FILLER_01   .str "Hmm... deixa eu pensar..."           ; 🔴 (.str + LOADSTR inexistentes; §5.3)
    FILLER_02   .str "Boa pergunta..."                     ; 🔴
    FILLER_03   .str "Ah, interessante..."                 ; 🔴
    FILLER_04   .str "Deixa eu organizar isso..."          ; 🔴
    FILLER_05   .str "Hmm, complexo isso..."               ; 🔴
    FILLER_06   .str "Peraí, deixa eu ver..."              ; 🔴
    FILLER_07   .str "Boa... isso merece atenção..."       ; 🔴
    FILLER_08   .str "Vou pensar com calma..."             ; 🔴

    FILLER_COUNT    .equ 8                                ; 🔴
    FILLER_DURATION .equ 1500000000    ; 1.5s média        ; 🔴

.text                                                     ; 🔴
reason_loop:
    ; ============ STEP 1: Detecta complexidade em ~20ms [META] ============
    EMBED rQueryEmb, rTranscript, rBge                    ; 🔴 modo DIM= (opcode ✅ 0x09)
    CONCAT rFeatures, rQueryEmb, rTurnMeta                ; 🟡 0x2F (Fase 2)
    FOREST rClassify, rFeatures, rXgb TREES=500           ; ✅ 0x22 (DEPTH= existe; tabelas: §5.3)
    SLICE rComplexity, rClassify, 1, 2                    ; ✅ 0x2E

    COMPARE rComplexity, 0.4                              ; 🔴 (inteiro escalado)
    IF_GREATER rComplexity, deep_path_with_filler         ; 🔴 (expansão: COMPARE PRED=GT + IF_EQUAL)
    JUMP fast_path                                        ; ✅ 0x0E

; ============================================================
; DEEP PATH — com filler speech imediato
; ============================================================
deep_path_with_filler:
    ; --- 1. Dispara filler IMEDIATAMENTE ---
    ; (não espera processar nada)
    SIGNAL NODE=LOCAL, CTX=rCtxAudioOut, KIND=FILLER_START  ; 🔴 KIND (mecanismo real no §5.1)
        PHRASE_ID=random(1, FILLER_COUNT)                   ; 🔴 (sorteio: §5.4)

    ; --- 2. Em paralelo, processa a resposta ---
    FORK rSnapDeep                                        ; 🔴 forma (opcode ✅ 0x04)
    CONCAT rPrompt, rTranscript, rContext                 ; 🟡
    EMBED rPromptEmb, rPrompt, rLlama.emb                 ; 🔴 modo (opcode ✅)

    ; Llama-405B começa a processar (leva 700ms-2s) [META]
    CALL llama_forward_loop rPromptEmb, rLlama, rOutput    ; 🔴 CALL

    ; --- 3. Sinaliza que a resposta real está pronta ---
    SIGNAL NODE=LOCAL, CTX=rCtxAudioOut, KIND=FILLER_STOP  ; 🔴 KIND (§5.1)
        NEW_TOKENS=rOutput                                ; 🔴 field-passing (palavra de controle, §5.1)

    JUMP synthesize_response                              ; ✅

; ============================================================
; AUDIO OUTPUT — Gerencia filler → resposta real
; ============================================================
audio_output_loop:
    SENSE rSignal, LOCAL_SIGNAL                           ; 🔴 (padrão real: PING + IF_INTERRUPT, §5.1)

    COMPARE rSignal.kind, FILLER_START                    ; 🔴 field access
    IF_EQUAL rSignal.kind, start_filler                   ; ✅ (com palavra de controle)

    COMPARE rSignal.kind, FILLER_STOP                     ; 🔴 field access
    IF_EQUAL rSignal.kind, stop_filler                    ; ✅ (idem)

    JUMP audio_output_loop                                ; ✅

start_filler:
    ; Toca frase de pensamento
    ; Usa o mesmo Depformer para consistência de voz
    LOADSTR rFillerText, FILLER_LIBRARY[rSignal.phrase_id]  ; 🔴 (biblioteca: §5.3)
    EMBED rFillerEmb, rFillerText, rPersona.emb            ; 🔴 modo (opcode ✅)
    DEPFORMER rFillerCodes, rFillerEmb, rAudioEmb, rKVDep  ; 🟡 0x44 (Fase 5)
    CODEC_DEC rFillerAudio, rFillerCodes                   ; ✅ 0x16
    STREAM rFillerAudio, CHANNEL=4 BLOCKING                ; 🔴 modo CHANNEL= (opcode ✅ 0x03)

    ; Marca estado
    HOLD_CONTEXT rAudioState, STATE=FILLER     ; 🔵 (ou enum em registrador, §5.1)
    JUMP audio_output_loop                                ; ✅

stop_filler:
    ; --- Transição suave ---
    ; 1. Termina a frase de filler naturalmente
    ;    (não corta no meio)

    ; 2. Emenda com o início da resposta real
    ;    usando crossfade de 200ms
    CROSSFADE rFillerTail, rResponseHead, 200ms           ; 🔴 opcode (construção com ops de hoje no §5.2)

    ; 3. Streama a resposta real
    EMBED rTextEmb, rSignal.new_tokens, rPersona.emb      ; 🔴 modo/field (opcode ✅)
    DEPFORMER rAudioCodes, rTextEmb, rAudioEmb, rKVDep    ; 🟡
    CODEC_DEC rAudio, rAudioCodes                         ; ✅
    STREAM rAudio, CHANNEL=4 BLOCKING                     ; 🔴 modo (opcode ✅)

    HOLD_CONTEXT rAudioState, STATE=RESPONDING  ; 🔵 (ou registrador)
    JUMP audio_output_loop                                ; ✅
```

## A Experiência (metas)

### Pergunta difícil sem filler (assistentes atuais)

```text
Você:  "Me explica decoerência quântica."

[SILÊNCIO DE 2 SEGUNDOS]

IA:    "Decoerência quântica é o processo pelo qual..."
```

O que você sente: "Está travado? Está processando? Será que entendeu?"

### Pergunta difícil com filler (M³-AVM, alvo)

```text
Você:  "Me explica decoerência quântica."

IA:    "Hmm... boa pergunta... deixa eu pensar..."
       [700ms - resposta sendo processada em paralelo]
       "...É o processo pelo qual um sistema quântico perde
        suas propriedades ao interagir com o ambiente..."

[TRANSIÇÃO SUAVE, SEM CORTE]
```

O que você sente: Como conversar com alguém que pensa antes de responder. A pausa de 700ms é natural — é o tempo que uma pessoa levaria para organizar o pensamento.

### Pergunta muito difícil (Llama-405B leva 2s) [META]

```text
Você:  "Me explica a diferença entre Mamba, Transformer e SSM,
        quando usar cada um, e como eles se relacionam com
        atenção linear."

IA:    "Hmm... complexo isso... deixa eu organizar..."
       [500ms]
       "...Boa... isso merece atenção..."
       [500ms]
       "...Então, vamos por partes. Mamba é um SSM com seleção..."
```

O que você sente: Ele realmente está pensando. As frases são naturais, variadas, e o ritmo é humano. Você não percebe que são "fillers" — parecem pensamento genuíno.

## O Detalhe Genial — Duração Adaptativa (alvo)

O filler adapta sua duração ao tempo real de computação:

```asm
adaptive_filler:
    ; Loop enquanto a resposta real não está pronta
    COMPARE rResponseReady, 1                             ; ✅ (flag como palavra de controle)
    IF_EQUAL rResponseReady, stop_filler                   ; ✅

    ; Se passou de 1.5s e ainda não há resposta,
    ; adiciona mais uma frase de pensamento
    CYCLES_COUNT rNow                                     ; ✅ (era CLOCK_QUERY)
    SUB rElapsed, rNow, rFillerStart                      ; 🔴 SUB (construção ADD+COMPARE, §5.1)
    COMPARE rElapsed, FILLER_DURATION                     ; ✅
    IF_GREATER rElapsed, add_more_filler                    ; 🔴 (expansão)

    YIELD                                                   ; ✅ 0x70
    JUMP adaptive_filler                                  ; ✅

add_more_filler:
    ; Escolhe outra frase (sem repetir a anterior)
    LOADI rNextPhrase, random_exclude(rLastPhrase, FILLER_COUNT)  ; 🔴 sorteio (§5.4)
    CALL play_filler_phrase rNextPhrase                    ; 🔴 CALL
    JUMP adaptive_filler                                  ; ✅
```

Se o Llama leva 2s, ele diz 2 frases. Se leva 3s, diz 3 frases. Nunca fica em silêncio.

## A Variação por Contexto (alvo)

O filler muda conforme o tipo de pergunta:

| Tipo de pergunta | Filler usado |
|:---|:---|
| Técnica complexa | "Hmm, deixa eu organizar isso..." |
| Pessoal | "Ah, boa pergunta..." |
| Matemática | "Peraí, deixa eu calcular..." |
| Filosófica | "Interessante... vou pensar com calma..." |
| Triste/emocional | "Hmm... isso é delicado..." |

O XGBoost classifica a emoção e o tipo da pergunta em ~20ms [META — `FOREST` 500×8 nunca bencheado; entrar na fila de benches]. Depois escolhe o filler mais apropriado.

## Por Que Isso é Importante

1. **Elimina a ansiedade do silêncio.** Silêncio de 2s faz o usuário pensar "travou?". Filler elimina isso.
2. **Aproveita tempo ocioso.** O Llama-405B está processando de qualquer jeito. Usar esse tempo para gerar fala natural é free.
3. **Humaniza.** Humanos fazem isso. O sistema parece mais inteligente porque parece pensar.
4. **Permite hardware mais lento.** Sem filler, você precisa de latência <200ms sempre. Com filler, você tolera 2s de processamento sem que o usuário perceba. (Com a ressalva de contenção single-GPU no §5.5.)

## O Que Falta (corrigido)

| Item | Status real |
|:---|:---|
| Biblioteca de fillers PT-BR (30–50 frases) | 🔵 criar conteúdo + armazenamento precisa init multivalor (follow-up RFC-0019, §5.3) |
| Classificador de tipo de pergunta | ✅ `FOREST` existe; 🟡 tabelas XGBoost 500×8 (construção: §5.3) + bench de 20ms nunca medido |
| Sinais FILLER_START/STOP | 🟡 "trivial" com mecanismo certo: `PING` + palavra de controle (§5.1) — **sem kinds novos** |
| Crossfade de 200ms | ✅ construção com `FILL`+`MUL`+`ADD` (§5.2) — `AUDIO_ALIGN` é alinhamento, não mixagem |
| Adaptação de duração | ✅ lógica + `RNG_UNIFORM` (sorteio com escada `COMPARE`, §5.4, até `CAST` Fase 4) |
| Depformer p/ filler (voz consistente) | 🟡 mesmo modelo, opcode DRAFT (`0x44`, Fase 5) |

Tempo de implementação (estimativa do autor): ~1 semana — **após** Fase 2 (`CONCAT`) + Fase 5 (`DEPFORMER` + biblioteca armazenável). Custo de hardware incremental: zero no alvo multi-GPU (§5.5).

## Resumo (alvo)

O sistema não precisa esperar 700ms em silêncio. Ele pode:

1. Detectar complexidade em ~20ms [META] (XGBoost via `FOREST`)
2. Começar a falar "Hmm, deixa eu pensar..." imediatamente
3. Processar a resposta real em paralelo (Llama-405B)
4. Transicionar suavemente quando a resposta está pronta
5. Adaptar a duração ao tempo real de processamento

Isso é o que humanos fazem. E é o que separa um assistente de uma pessoa.

---

## §5. Análise de engenharia

### §5.1 Sinais sem kinds novos (recomendado)

`FILLER_START` / `FILLER_STOP` / `RESUME` / `FINALIZE` cabem no ISA
congelado assim:

- Remetente: `SIGNAL rD, <ctx>, 0, 0, <seq>` com `KIND=PING` (✅ local)
  + escreve antes uma **palavra de controle** (tensor 1×N ou registrador
  combinado via `LOADI`) com `(verbo, phrase_id | handle_tokens)`.
- Destino: `IF_INTERRUPT` (✅ consome) → lê a palavra de controle →
  `COMPARE` + `IF_EQUAL` despacha o verbo.
- `HOLD_CONTEXT rAudioState, STATE=X` vira `LOADI rAudioState, <enum>`
  (convenção documentada na RFC do MVP, não opcode).

Nenhum encoding muda. Quando o transporte chegar (Fase 7), a palavra de
controle viaja no `SEND_TENSOR` que antecede o `SIGNAL` — o padrão já é
distribuível por construção.

### §5.2 Crossfade com ops de hoje

Crossfade linear de 200ms entre cauda do filler `x` e cabeça da
resposta `y`: `out = a·x + b·y`, `a+b=1`, rampas complementares.

- `TENSOR rA … FILL=<a0>` / `FILL=<b0>` por janela (✅ RFC-0019; uma
  rampa por chunk de 80ms, 3 chunks ≈ 200ms) → `MUL` (✅) → `ADD` (✅).
- Sem `CROSSFADE`, sem `AUDIO_ALIGN` (que é carimbo temporal, não
  mixagem). Qualidade de rampa por chunk é grosseira em 80ms — medir;
se artefato audível, rampa por amostra vira argumento para RFC
futura, com números na mão (§13).

### §5.3 Biblioteca de frases e tabelas FOREST (o gargalo real)

O rascunho subestima este item: 8–50 frases PT-BR como **tokens** e
tabelas `FOREST` 500×8 exigem **init multivalor de tensor** (follow-up
RFC-0019, Fase 10) — `FILL` escalar não basta. Caminhos honestos:

1. **Tensores assados offline** (hoje): gerar os bytes `.m3bin` de cada
   tabela/frase com script Python e carregar como `TENSOR` pré-preenchido
   (o loader GGUF/`mmap` já prova o padrão de bytes externos).
2. **`CONCAT` de peças `FILL`** (Fase 2): verboso, mas só-ISA.
3. **Init multivalor** (Fase 10): a solução definitiva.

Recomendação: (1) para o MVP — tira o MVP do caminho crítico do ISA.

### §5.4 Sorteio de frase sem `CAST`

`RNG_UNIFORM r, [0,8)` (✅) dá `f32` no registrador. Sem `CAST`
(🟡 `0x67`, Fase 4), índice via escada: 8× (`COMPARE r, <k+1>,
PRED=LT` + `IF_EQUAL`) — feio, O(8), mas funciona e é determinístico
(RNG com seed, RFC-0005). Com `CAST`, vira 2 instruções. Não bloquear o
MVP por isso; trocar quando a Fase 4 pousar. Anti-repetição =
`COMPARE` contra `rLastPhrase` + re-sortear (limitado a 2 tentativas,
depois aceita — evita loop patológico).

### §5.5 Ressalva de contenção

"Custo zero" vale no alvo multi-GPU (Depformer GPU0 ∥ Llama GPU1). Em
setup single-GPU, filler e resposta disputam o mesmo dispositivo —
medir com `M3_PROFILE=1`; o scheduler `Red>Blue>Green` + `YIELD`
(✅) é o mecanismo de convivência até haver placement real (Fase 9,
`PLACE` realocado — o `0xBA` do rascunho é proposta inválida).

## §6. Matriz de construtibilidade

| Construção | Hoje | Com o plano |
|:---|:---|:---|
| Classificar complexidade (`FOREST`+`SLICE`) | ✅ parcial (tabelas: §5.3) | Tabelas via (1)/(2)/(3) |
| Disparar filler (`PING`+controle) | ✅ | — |
| Tocar filler (Depformer→CODEC_DEC→STREAM) | 🔴 | 🟡 Fase 5 (`DEPFORMER`) |
| Sortear frase | ✅ (escada `COMPARE`, §5.4) | 🟡 `CAST` limpa (Fase 4) |
| Crossfade 200ms | ✅ (`FILL`+`MUL`+`ADD`, §5.2) | — |
| Duração adaptativa (timer+RNG) | ✅ | — |
| Variar por emoção/tipo | ✅ parcial (idem classificador) | Bench 20ms |
| Biblioteca 30–50 frases PT-BR | 🔴 conteúdo + init | Conteúdo agora; init §5.3 |

**Demo mínima honesta (só ISA v1.5):** o loop adaptativo + crossfade
sobre tons/PCM sintético (`SENSE AUDIO_PCM` ✅ gera frame determinístico)
— prova transição e duração sem nenhum modelo. É o "Marco 0" do filler.

## §7. Mapa de fases

| Capacidade | Desbloqueia em |
|:---|:---|
| Demo sintética (transição+duração) | Hoje (ISA v1.5) |
| Filler com voz real | Fase 5 (`DEPFORMER`) + biblioteca (§5.3) |
| Sorteio limpo + tabelas fáceis | Fase 4 (`CAST`) + Fase 10 (init multivalor) |
| Filler no deep path especulativo | Marco 3 (VISÃO Anexo B) |
| Filler cross-GPU sem contenção | Fase 9 (placement) |

---

*Documento companheiro de `docs/EXPERIENCIA_PREMIUM.md` (seção "Pergunta
Profunda") e `docs/PAUSA_DE_PENSAMENTO.md` (o silêncio do usuário).
Nada aqui altera ESPEC.md (normativo) — `CROSSFADE` como opcode, se um
dia proposto, exige RFC com prova §13 (hoje: construção com ops
existentes vence).*

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
