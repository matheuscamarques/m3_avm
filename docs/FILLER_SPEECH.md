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
| Biblioteca de fillers (30 frases EN nativas) | 🟡 conteúdo redigido (§8); falta assar como tabela (§5.3-1). PT-BR volta após fine-tuning |
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
| Filler com voz real | Fase 5 (`DEPFORMER`) + biblioteca (§5.3, conteúdo §8) |
| Sorteio limpo + tabelas fáceis | Fase 4 (`CAST`) + Fase 10 (init multivalor) |
| Filler no deep path especulativo | Marco 3 (VISÃO Anexo B) |
| Filler cross-GPU sem contenção | Fase 9 (placement) |

---

## §8. Biblioteca de fillers — English nativo

**Decisão registrada:** PersonaPlex é treinado em inglês; gerar filler em
PT-BR no Depformer inglês sairia com sotaque carregado ou errado.
Qualquer teste de voz é **em inglês** até o fine-tuning PT-BR (limitação
"sotaque" em `EXPERIENCIA_PREMIUM.md`). PT-BR volta como segunda
biblioteca após o fine-tuning — a arquitetura é agnóstica a idioma, só
a tabela assada troca (§5.3, caminho 1).

**Correção de contagem:** o rascunho diz `FILLER_COUNT .equ 32`, mas a
biblioteca tem **30 frases** (5+5+4+4+4+4+4). Os offsets do seletor
conferem com 30 (tech 0–4, personal 5–9, math 10–13, phil 14–17,
emo 18–21, long 22–25, trans 26–29). Valor correto: **30**.

```asm
.data                                                     ; 🔴 assembler (tabela assada offline, §5.3)
    ; ============================================================
    ; FILLER LIBRARY — English (native PersonaPlex)
    ; Organized by question type and emotional context
    ; ============================================================

    ; --- Type 0: TECHNICAL / COMPLEX (IDs 0-4) ---
    FILLER_TECH_01   .str "Hmm, let me think about that..."
    FILLER_TECH_02   .str "Okay, that's a good one..."
    FILLER_TECH_03   .str "Alright, let me organize this..."
    FILLER_TECH_04   .str "Good question... hold on..."
    FILLER_TECH_05   .str "Let me work through this..."

    ; --- Type 1: PERSONAL / CASUAL (IDs 5-9) ---
    FILLER_PERS_01   .str "Oh, interesting..."
    FILLER_PERS_02   .str "Hmm, let me think..."
    FILLER_PERS_03   .str "Ah, good question..."
    FILLER_PERS_04   .str "Yeah, I get what you mean..."
    FILLER_PERS_05   .str "Right, right..."

    ; --- Type 2: MATH / CALCULATION (IDs 10-13) ---
    FILLER_MATH_01   .str "Hold on, let me calculate..."
    FILLER_MATH_02   .str "Okay, let me work this out..."
    FILLER_MATH_03   .str "Hmm, let me check the numbers..."
    FILLER_MATH_04   .str "Alright, give me a sec..."

    ; --- Type 3: PHILOSOPHICAL / OPEN-ENDED (IDs 14-17) ---
    FILLER_PHIL_01   .str "Hmm, that's a deep one..."
    FILLER_PHIL_02   .str "Interesting... let me think carefully..."
    FILLER_PHIL_03   .str "That deserves a thoughtful answer..."
    FILLER_PHIL_04   .str "Wow, okay... let me gather my thoughts..."

    ; --- Type 4: EMOTIONAL / SENSITIVE (IDs 18-21) ---
    FILLER_EMO_01    .str "Hmm... that's a delicate one..."
    FILLER_EMO_02    .str "I see... let me think about this..."
    FILLER_EMO_03    .str "Okay... that's important..."
    FILLER_EMO_04    .str "Hmm, let me take this seriously..."

    ; --- Type 5: LONG PROCESSING >2s (IDs 22-25) ---
    FILLER_LONG_01   .str "Alright... this is complex..."
    FILLER_LONG_02   .str "Okay, let me break this down..."
    FILLER_LONG_03   .str "Hmm, there's a lot here..."
    FILLER_LONG_04   .str "Let me go step by step..."

    ; --- Type 6: TRANSITION / CONTINUATION (IDs 26-29) ---
    FILLER_TRANS_01  .str "Okay, so..."
    FILLER_TRANS_02  .str "Alright, here's the thing..."
    FILLER_TRANS_03  .str "So, the way I see it..."
    FILLER_TRANS_04  .str "Right, so basically..."

    FILLER_COUNT     .equ 30   ; corrigido (era 32; §8)
    FILLER_DURATION  .equ 1500000000    ; 1.5s average
```

### §8.1 Como o sistema escolhe o filler (alvo)

```asm
; ============================================================
; CLASSIFY QUESTION — ~20ms via XGBoost [META — nunca bencheado]
; ============================================================
classify_question_type:
    .param rTranscript, rType                             ; 🔴 assembler

    ; Extract features
    EMBED rQueryEmb, rTranscript, rBge                    ; ✅-opcode (DIM= 🔴)
    CONCAT rFeatures, rQueryEmb, rTurnMeta                ; 🟡 0x2F

    ; XGBoost predicts (type, confidence, complexity)
    FOREST rClassify, rFeatures, rXgb                     ; ✅ (MODE=PROBABILITY 🔴 — usar MEAN/VOTE)
        TREES=500
        DEPTH=8
        MODE=PROBABILITY

    SLICE rType, rClassify, 0, 1        ; 0-6 (type)      ; ✅ (layout de saída: convenção a confirmar na RFC)
    SLICE rConfidence, rClassify, 1, 2                    ; ✅ (idem)
    SLICE rComplexity, rClassify, 2, 3                    ; ✅ (idem)

    RET                                                   ; 🔴

; ============================================================
; SELECT FILLER — Based on type + complexity
; ============================================================
select_filler:
    .param rType, rComplexity, rFillerID                   ; 🔴

    ; If complexity > 0.7, use LONG fillers
    COMPARE rComplexity, 0.7                              ; 🔴 (inteiro escalado, ex. 7 em escala ×10)
    IF_GREATER rComplexity, use_long                       ; 🔴 (expansão PRED=GT + IF_EQUAL)

    ; Otherwise, map type to filler group
    COMPARE rType, 0                    ; TECHNICAL       ; ✅
    IF_EQUAL rType, use_tech                               ; ✅
    COMPARE rType, 1                    ; PERSONAL        ; ✅
    IF_EQUAL rType, use_personal                           ; ✅
    COMPARE rType, 2                    ; MATH            ; ✅
    IF_EQUAL rType, use_math                               ; ✅
    COMPARE rType, 3                    ; PHILOSOPHICAL   ; ✅
    IF_EQUAL rType, use_phil                               ; ✅
    COMPARE rType, 4                    ; EMOTIONAL       ; ✅
    IF_EQUAL rType, use_emotional                          ; ✅
    JUMP use_transition                 ; default         ; ✅

use_tech:
    LOADI rFillerID, random(1, 5) + 0                     ; 🔴 expressão (sorteio real: RNG + escada §5.4)
    RET                                                   ; 🔴
use_personal:
    LOADI rFillerID, random(1, 5) + 5                     ; 🔴 (idem)
    RET                                                   ; 🔴
use_math:
    LOADI rFillerID, random(1, 4) + 10                    ; 🔴 (idem)
    RET                                                   ; 🔴
use_phil:
    LOADI rFillerID, random(1, 4) + 14                    ; 🔴 (idem)
    RET                                                   ; 🔴
use_emotional:
    LOADI rFillerID, random(1, 4) + 18                    ; 🔴 (idem)
    RET                                                   ; 🔴
use_long:
    LOADI rFillerID, random(1, 4) + 22                    ; 🔴 (idem)
    RET                                                   ; 🔴
use_transition:
    LOADI rFillerID, random(1, 4) + 26                    ; 🔴 (idem)
    RET                                                   ; 🔴
```

> A estrutura de despacho (tabela tipo→faixa de IDs + override por
> complexidade) é válida e monta com ISA de hoje menos `CONCAT` e o
> sorteio. O conteúdo (30 frases EN) é o dado de entrada do caminho
> §5.3-1: assar como tabela offline.

### §8.2 Exemplos de conversa (English, nativo — metas)

**Pergunta técnica complexa:**

```text
User:  "Can you explain quantum decoherence?"

AI:    "Hmm, let me think about that..."
       [700ms — Llama-405B processing in parallel]

       "...Okay, so decoherence is the process by which a quantum
        system loses its quantum properties — superposition and
        entanglement — through interaction with the environment.
        Imagine an electron that's in two places at once. The moment
        it touches anything — an air molecule, a photon of light —
        that 'which-way' information leaks into the environment.
        The electron is forced to 'choose' one place. The
        superposition collapses. That's why quantum computers need
        extreme isolation: any vibration, any light, any heat
        destroys the computation. Decoherence is the biggest enemy
        of quantum computing. Want me to go deeper into any
        specific aspect?"
```

**Pergunta filosófica:**

```text
User:  "What's the meaning of life?"

AI:    "Hmm... that's a deep one..."
       [1.2s — philosophical question, longer processing]

       "...Let me think carefully. There's no single answer that
        works for everyone. Some find meaning in relationships,
        some in creation, some in understanding. The philosopher
        Susan Wolf argues meaning comes from active engagement
        with projects of worth. The psychologist Viktor Frankl
        found meaning through suffering, love, and work.
        Honestly? I think the question itself is part of the
        answer — the fact that you're asking means you're already
        searching, and that search is meaningful. What prompted
        the question?"
```

**Pergunta emocional:**

```text
User:  "I think I might be depressed."

AI:    "Hmm... that's a delicate one..."
       [800ms]

       "...I see. Thank you for telling me. That takes courage.
        I'm not a therapist, so I won't pretend to diagnose you —
        but I hear you. What you're feeling is real, and it's
        worth taking seriously. Can you tell me more about
        what's been going on? And if you're in crisis, please
        reach out to a professional — I can share resources if
        you want."
```

> Nota de segurança (trabalho futuro): o comportamento de *resposta*
> em crise (não-diagnóstico, oferta de recursos, escalação) precisa de
> política própria — documento de safety futuro, fora do escopo deste
> desenho. A camada de filler (frases EMO 18–21, tom de-escalador) já
> segue esse princípio e independe da política.

**Processamento longo (pergunta multipartes):**

```text
User:  "Explain the difference between Mamba, Transformer, and
        SSM, when to use each, and how they relate to linear
        attention."

AI:    "Alright... this is complex..."
       [500ms]
       "Okay, let me break this down..."
       [500ms]
       "Hmm, there's a lot here..."
       [500ms]
       "...Right. So let's go step by step. First, SSM stands for
        State Space Model — it's a mathematical framework. Mamba
        is a specific SSM implementation with selective scan.
        Transformer is a different architecture based on attention.
        Linear attention is an approximation of Transformer
        attention that makes it O(N) instead of O(N²)..."
```

Total filler time: 1.5s. Total processing: 2s. User perception: natural thought.

### §8.3 Ajuste no assembly original

A troca de idioma **não muda a arquitetura** — muda a tabela assada
(`FILLER_LIBRARY_PT` → `FILLER_LIBRARY_EN`, mesmo formato §5.3-1).
No assembly-alvo, o ponto de troca é o carregamento da frase:

```asm
start_filler:
    ; Antes (rascunho PT-BR):
    ; LOADSTR rFillerText, FILLER_LIBRARY_PT[rSignal.phrase_id]   ; 🔴 (LOADSTR/indexação inexistentes)

    ; Depois (correto — mesmo ponto, tabela EN):
    LOADSTR rFillerText, FILLER_LIBRARY_EN[rSignal.phrase_id]     ; 🔴 (idem; mecanismo real: tabela assada §5.3-1)

    EMBED rFillerEmb, rFillerText, rPersona.emb            ; 🔴 modo (opcode ✅)
    DEPFORMER rFillerCodes, rFillerEmb, rAudioEmb, rKVDep  ; 🟡 0x44
    CODEC_DEC rFillerAudio, rFillerCodes                   ; ✅
    STREAM rFillerAudio, CHANNEL=4 BLOCKING                ; 🔴 modo (opcode ✅)

    HOLD_CONTEXT rAudioState, STATE=FILLER     ; 🔵 (ou LOADI de enum, §5.1)
    JUMP audio_output_loop                                ; ✅
```

Nada mais muda. Toda a arquitetura permanece igual — e o teste em
inglês ainda *simplifica* a validação (distribuição nativa do
Depformer, sem risco de sotaque).

---

*Documento companheiro de `docs/EXPERIENCIA_PREMIUM.md` (seção "Pergunta
Profunda") e `docs/PAUSA_DE_PENSAMENTO.md` (o silêncio do usuário).
Nada aqui altera ESPEC.md (normativo) — `CROSSFADE` como opcode, se um
dia proposto, exige RFC com prova §13 (hoje: construção com ops
existentes vence).*

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
