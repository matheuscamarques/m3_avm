# Pausa de Pensamento — Turn-Taking no M³-AVM Premium

```text
Status:   DESIGN-ALVO (proposta de mecanismo, NÃO-normativa, NÃO implementada)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Companion: docs/EXPERIENCIA_PREMIUM.md (a sensação) + docs/VISAO_PREMIUM_250GB.md (a engenharia)
License:  AGPL-3.0-or-later (see LICENSE)
```

> **Contrato de honestidade (ESPEC §1.3):** comportamentos e latências
> abaixo são **metas de design**. O §0 lista as correções factuais
> aplicadas ao rascunho; o §5 responde com engenharia se os dois opcodes
> propostos são mesmo necessários.

## §0. Correções aplicadas ao rascunho

1. **"ainda não estão no v1.3" → v1.5.** A linha atual é ISA v1.5.
2. **`CLOCK_QUERY 🟡 0x78` está errado:** `0x78` é `LOADI`. Relógio
   monotônico existe como `CYCLES_COUNT` (✅ `0x6A`, RFC-0006). Todo
   `CLOCK_QUERY` do rascunho foi convertido para `CYCLES_COUNT`.
3. **`SUB ... ✅ 0x0A` está errado:** `0x0A` é `ADD`; **`SUB` não existe**
   no ISA. Todo `SUB` foi marcado 🔴, com construção alternativa no §5
   (`ADD` + `COMPARE PRED=`).
4. **`IF_LESS` / `IF_GREATER` não existem.** Só há `IF_EQUAL`,
   `IF_INTERRUPT`, `JUMP`. Expansão válida (RFC-0007 tem 6 predicados):
   `COMPARE X, T, PRED=LT` + `IF_EQUAL label`. Marcados 🔴 com a expansão
   anotada.
5. **`VAD_DETECT` não é ✅.** É 🟡 DRAFT (`0x46`, Fase 5). A tabela de
   "Honestidade" do rascunho foi corrigida.
6. **`KIND=FINALIZE` / `KIND=RESUME` não existem.** `SIGNAL` aceita só
   `ABORT/FORK_REQ/HALT/KILL` (+`PING` local). Marcados 🔴.
7. **`SENSE rSignal, SIGNAL_KIND`, `LOAD`, `COPY`, `CALL/RET/.param`,
   `.data/.equ/.text`** — inexistentes (🔴); `COPY`→`MOV` (✅ `0x79`),
   `LOAD`→`MOV`/`LOADI` existem.
8. **"Aprender padrão 🟡 RFC-0010 (LoRA)"** — RFC-0010 é `KV_TRUNCATE`;
   LoRA/training é **Trilha P** (sem RFC ainda). Corrigido.

Legenda: ✅ IMPL · 🟡 DRAFT em ESPEC-V2 (precisa RFC+código) ·
🔵 proposto-novo (este documento) · 🔴 inexistente/bloqueado.

---

Sim. E isso é o que separa uma conversa real de um atendimento automatizado.
O ponto mais interessante: o que acontece quando o usuário pausa no meio de um pensamento?

A resposta-alvo: o sistema segura o contexto, espera, e quando você retoma, ele continua de onde parou — ou reformula com base no que veio depois.

## O Que Acontece em uma Conversa Humana

Imagine este diálogo:

```text
Você:  "Me explica... deixa eu pensar..."
       [2 segundos de silêncio]
Você:  "Então, o que é decoerência quântica?"

Amigo: "Ah, agora sim. Decoerência é quando..."
```

Um humano faz três coisas:

1. **Percebe** que você está pensando (não é fim de turno)
2. **Espera** (não interrompe)
3. **Reformula** quando você completa a pergunta

O alvo do M³-AVM é fazer exatamente isso, com três mecanismos:

| Mecanismo | Opcode | Status |
|:---|:---|:---|
| Detectar pausa vs. fim de turno | `VAD_DETECT` + estado de turno | 🟡 DRAFT (`0x46`, Fase 5); estado = convenção, ver §5 |
| Manter contexto pendente | `HOLD_CONTEXT` | 🔵 proposto (necessidade em análise no §5) |
| Reformular quando retoma | `RESUME_QUERY` | 🔵 proposto (necessidade em análise no §5) |

## Como Isso Funciona na Prática (alvo)

### Fase 1 — Você começa a pergunta

```text
Você: "Me explica..."
```

```asm
; Audio input loop detecta fala
SENSE rPCM, AUDIO_PCM                                     ; ✅ 0x06 (LEN= 🔴 — não existe)
VAD_DETECT rVAD, rPCM THRESHOLD=0.3                       ; 🟡 0x46
COMPARE rVAD, 0.3                                         ; 🔴 float imediato (usar inteiro escalado, RFC-0007)
IF_GREATER rVAD, start_utterance                          ; 🔴 (expansão: COMPARE PRED=GT + IF_EQUAL)

start_utterance:
    ; Marca início de turno
    HOLD_CONTEXT rPending, STATE=INCOMPLETE   ; 🔵
    CODEC_ENC rCodes, rPCM                                ; ✅ 0x15
    STREAM_WRITE rAudioInRing, rCodes                      ; 🔴 modo (STREAM ✅ existe)
    SIGNAL NODE=LOCAL, CTX=rCtxReason, KIND=PING           ; 🔴 forma (opcode ✅ 0x1B)
```

O sistema não processa ainda. Ele marca que você começou a falar e espera.

### Fase 2 — Você pausa no meio

```text
Você: "...deixa eu pensar..."
       [2 segundos de silêncio]
```

```asm
; Audio input loop detecta silêncio
SENSE rPCM, AUDIO_PCM                                     ; ✅
VAD_DETECT rVAD, rPCM THRESHOLD=0.3                       ; 🟡
COMPARE rVAD, 0.3                                         ; 🔴 (inteiro escalado)
IF_LESS rVAD, check_pause                                 ; 🔴 (expansão: COMPARE PRED=LT + IF_EQUAL)

check_pause:
    ; Verifica se é pausa curta (pensamento) ou fim de turno
    CYCLES_COUNT rNow                                     ; ✅ 0x6A (era CLOCK_QUERY — corrigido §0.2)
    SUB rDelta, rNow, rLastSpeech                         ; 🔴 SUB não existe (construção alternativa no §5)
    COMPARE rDelta, 2000000000                 ; 2 segundos  ; ✅ (inteiro)
    IF_LESS rDelta, hold_pending                           ; 🔴 (expansão)

hold_pending:
    ; Não é fim de turno — mantém contexto pendente
    HOLD_CONTEXT rPending, STATE=THINKING      ; 🔵
    ; Não aciona o modelo ainda
    JUMP audio_input_loop                                 ; ✅

; Só depois de 3s de silêncio considera fim de turno
check_timeout:
    COMPARE rDelta, 3000000000                            ; ✅
    IF_GREATER rDelta, finalize_turn                       ; 🔴 (expansão)
    JUMP audio_input_loop                                 ; ✅

finalize_turn:
    ; Se o usuário não voltou em 3s, processa o que tem
    SIGNAL NODE=LOCAL, CTX=rCtxReason, KIND=FINALIZE       ; 🔴 KIND não existe
    HOLD_CONTEXT rPending, STATE=CLOSED        ; 🔵
```

O sistema espera. Ele não aciona o Llama ainda. Ele não gera áudio. Ele segura o contexto como "pendente".

### Fase 3 — Você retoma

```text
Você: "Então, o que é decoerência quântica?"
```

```asm
; Audio input loop detecta nova fala
start_utterance:
    ; Verifica se há contexto pendente
    LOAD rPendingState, rPending.STATE                     ; 🔴 LOAD/field-access não existem (convenção no §5)
    COMPARE rPendingState, THINKING                       ; ✅ (com enum em registrador)
    IF_EQUAL rPendingState, resume_utterance               ; ✅ 0x0D

resume_utterance:
    ; Concatena a nova fala com o contexto pendente
    HOLD_CONTEXT rPending, STATE=RESUMED       ; 🔵
    CODEC_ENC rCodesNew, rPCM                             ; ✅
    STREAM_WRITE rAudioInRing, rCodesNew                   ; 🔴 modo

    ; Notifica reasoning thread com contexto completo
    SIGNAL NODE=LOCAL, CTX=rCtxReason, KIND=RESUME         ; 🔴 KIND não existe
    JUMP audio_input_loop                                 ; ✅

; Na reasoning thread:
reason_loop:
    ; Recebe sinal RESUME
    SENSE rSignal, SIGNAL_KIND                            ; 🔴 periférico não existe
    COMPARE rSignal, RESUME                               ; ✅ (com enum)
    IF_EQUAL rSignal, handle_resume                       ; ✅

handle_resume:
    ; Concatena contexto pendente + nova fala
    CONCAT rFullQuery, rPending.Text, rNewText  ; 🟡 0x2E→0x2F: CONCAT é 0x2F (0x2E é SLICE)

    ; Agora sim processa a pergunta completa
    EMBED rQueryEmb, rFullQuery, rBge                     ; 🔴 modo DIM= (opcode ✅ 0x09)
    RAG_SEARCH rDocs, rQueryEmb, rRag TOPK=5              ; 🟡 0x52

    ; Decide fast vs deep path baseado na query COMPLETA
    FOREST rClassify, rFeatures, rXgb                     ; ✅ 0x22
    COMPARE rComplexity, 0.4                              ; 🔴 (inteiro escalado)
    IF_GREATER rComplexity, deep_path                      ; 🔴 (expansão)
    JUMP fast_path                                        ; ✅
```

Agora sim o sistema processa — com a pergunta completa ("Me explica... decoerência quântica"), não só a segunda parte.

## O Que Isso Significa na Prática (metas)

### Experiência 1 — Pausa de pensamento

```text
Você:  "Me explica..."
       [1.5s de silêncio — você está pensando]
Você:  "...como funciona atenção em Transformers?"

IA:    "Ah, atenção é o mecanismo central. Basicamente..."
```

A IA não interrompe durante a pausa. Ela espera. E quando você retoma, ela responde à pergunta completa.

### Experiência 2 — Reformulação

```text
Você:  "Me explica deco... não, espera, deixa eu reformular."
       [1s de silêncio]
Você:  "O que é emaranhamento quântico?"

IA:    "Emaranhamento é diferente de decoerência. Emaranhamento é..."
```

A IA detecta a reformulação e descarta a primeira parte. Ela não responde "decoerência" — responde "emaranhamento".

### Experiência 3 — Continuação

```text
Você:  "Qual é a diferença entre Mamba e Transformer? E..."
       [2s de silêncio — você está pensando]
Você:  "...e quando usar cada um?"

IA:    "Boa pergunta. Mamba é O(1) em estado, Transformer é O(N).
        Você usa Mamba quando precisa velocidade e contexto curto.
        Transformer quando precisa raciocínio profundo e contexto longo."
```

A IA lembra da primeira parte e integra com a segunda.

### Experiência 4 — Abandono de pensamento

```text
Você:  "Me explica..."
       [5s de silêncio — você desistiu]
Você:  "Deixa, esquece."

IA:    "Sem problema. Quando quiser, é só perguntar."
```

A IA detecta o abandono e não processa. Ela responde de forma natural, como um humano faria.

## Os Três Estados de Conversa

| Estado | Sinal | Ação do sistema |
|:---|:---|:---|
| Fim de turno | Silêncio > 3s OU entonação descendente | Processa e responde |
| Pausa de pensamento | Silêncio < 3s com pausa curta | Espera, mantém contexto |
| Abandono | "Deixa", "esquece", "nada não" | Descarta contexto |

### Como detectar cada um (alvo)

```asm
; Classificador de estado de turno
CALL classify_turn_state rPCM, rState                     ; 🔴 CALL/.param não existem

classify_turn_state:
    .param rPCM, rState                                   ; 🔴 assembler

    ; 1. Detecta silêncio
    VAD_DETECT rVAD, rPCM                                 ; 🟡
    COMPARE rVAD, 0.3                                     ; 🔴 (inteiro escalado)
    IF_LESS rVAD, check_silence_duration                   ; 🔴 (expansão)

    ; 2. Se há fala, detecta entonação
    LOADI rState, SPEAKING                                ; ✅ 0x78 (com enum)
    RET                                                   ; 🔴 RET não existe

check_silence_duration:
    CYCLES_COUNT rNow                                     ; ✅ (era CLOCK_QUERY)
    SUB rDelta, rNow, rLastSpeech                         ; 🔴 (construção no §5)

    ; 3. Silêncio < 3s = pausa de pensamento
    COMPARE rDelta, 3000000000                            ; ✅
    IF_LESS rDelta, state_pause                            ; 🔴 (expansão)

    ; 4. Silêncio > 3s = fim de turno
    LOADI rState, TURN_END                                ; ✅
    RET                                                   ; 🔴

state_pause:
    ; 5. Detecta palavras de abandono no texto acumulado
    CALL detect_abandon_phrases rPendingText, rAbandon     ; 🔴 CALL (+ precisa ASR-texto, ver §5)
    COMPARE rAbandon, 1                                   ; ✅
    IF_EQUAL rAbandon, state_abandon                       ; ✅

    LOADI rState, PAUSE                                   ; ✅
    RET                                                   ; 🔴

state_abandon:
    LOADI rState, ABANDON                                 ; ✅
    RET                                                   ; 🔴
```

## O Assembly Completo — MVP com Pausa de Pensamento (alvo, não monta hoje)

```asm
; ============================================================
; programs/mvp_voice_with_pause.m3asm  [ALVO — ver §0.2 da VISÃO]
; Voice assistant with thought-pause handling
; ============================================================

.data                                                     ; 🔴 assembler
    SILENCE_THRESHOLD   .equ 0.3                          ; 🔴
    PAUSE_TIMEOUT_NS    .equ 3000000000    ; 3s            ; 🔴
    THOUGHT_TIMEOUT_NS  .equ 2000000000    ; 2s            ; 🔴

.text                                                     ; 🔴
main:
    ; Setup (same as MVP)
    SPAWN_CONTEXT rCtxAudioIn,  ENTRY=audio_input_loop,  PRIORITY=RED    ; 🟡 0xA2
    SPAWN_CONTEXT rCtxReason,   ENTRY=reason_loop,       PRIORITY=BLUE   ; 🟡

; ============================================================
; AUDIO INPUT — Detecta pausa vs fim de turno
; ============================================================
audio_input_loop:
    SENSE rPCM, AUDIO_PCM, LEN=1920                       ; 🔴 modo (opcode ✅)
    VAD_DETECT rVAD, rPCM THRESHOLD=SILENCE_THRESHOLD     ; 🟡
    COMPARE rVAD, SILENCE_THRESHOLD                       ; 🔴 (inteiro escalado)
    IF_LESS rVAD, handle_silence                           ; 🔴 (expansão)
    JUMP handle_speech                                    ; ✅

handle_speech:
    ; Marca timestamp da última fala
    CYCLES_COUNT rNow                                     ; ✅
    COPY rLastSpeech, rNow                                ; 🔴 COPY (usar MOV ✅ 0x79)

    ; Se é continuação, marca como RESUMED
    LOAD rPendingState, rPending.STATE                    ; 🔴 (convenção no §5)
    COMPARE rPendingState, PAUSE                          ; ✅
    IF_EQUAL rPendingState, mark_resumed                   ; ✅

    ; Senão, é fala nova
    CODEC_ENC rCodes, rPCM                                ; ✅
    STREAM_WRITE rAudioInRing, rCodes                      ; 🔴 modo
    JUMP audio_input_loop                                 ; ✅

mark_resumed:
    HOLD_CONTEXT rPending, STATE=RESUMED       ; 🔵
    CODEC_ENC rCodes, rPCM                                ; ✅
    STREAM_WRITE rAudioInRing, rCodes                      ; 🔴 modo
    SIGNAL NODE=LOCAL, CTX=rCtxReason, KIND=RESUME         ; 🔴 KIND
    JUMP audio_input_loop                                 ; ✅

handle_silence:
    CYCLES_COUNT rNow                                     ; ✅
    SUB rDelta, rNow, rLastSpeech                         ; 🔴 (construção no §5)

    ; < 2s: pausa curta, mantém pendente
    COMPARE rDelta, THOUGHT_TIMEOUT_NS                    ; ✅
    IF_LESS rDelta, short_pause                            ; 🔴 (expansão)

    ; 2s–3s: pausa de pensamento
    COMPARE rDelta, PAUSE_TIMEOUT_NS                      ; ✅
    IF_LESS rDelta, thought_pause                          ; 🔴 (expansão)

    ; > 3s: fim de turno (se havia fala)
    LOAD rPendingState, rPending.STATE                    ; 🔴 (convenção no §5)
    COMPARE rPendingState, SPEAKING                       ; ✅
    IF_EQUAL rPendingState, finalize_turn                  ; ✅
    JUMP audio_input_loop                                 ; ✅

short_pause:
    ; Não faz nada, só espera
    JUMP audio_input_loop                                 ; ✅

thought_pause:
    HOLD_CONTEXT rPending, STATE=PAUSE         ; 🔵
    JUMP audio_input_loop                                 ; ✅

finalize_turn:
    ; Processa o que foi dito até agora
    HOLD_CONTEXT rPending, STATE=CLOSED         ; 🔵
    SIGNAL NODE=LOCAL, CTX=rCtxReason, KIND=FINALIZE       ; 🔴 KIND
    JUMP audio_input_loop                                 ; ✅
```

## O Que Isso Entrega na Experiência (metas)

| Momento | O que o usuário sente |
|:---|:---|
| Pausa de 1s | "Ele está esperando eu terminar" |
| Pausa de 2s | "Ele percebe que estou pensando" |
| Pausa de 3s+ | "Ele vai responder o que eu disse até agora" |
| Retoma no meio | "Ele lembra do que eu falei antes" |
| Reformulação | "Ele entendeu que eu mudei de ideia" |
| Abandono | "Ele respeitou que eu desisti" |

## A Parte Mais Interessante — Antecipação (visão)

Depois de 1000 turnos com o mesmo usuário, o sistema aprende seu padrão de pausa:

- Se você sempre pausa 1.5s antes de completar a pergunta, ele começa a preparar o RAG durante a pausa
- Se você sempre reformula, ele espera a reformulação antes de processar
- Se você sempre abandona, ele não desperdiça computação

Isso é feito pelo LoRA em background (thread GREEN). O modelo aprende a conversar com você.

> Requer RAG (Fase 6) + LoRA (Trilha P) — o mais distante deste documento.
> Nada disso existe hoje.

## Honestidade Sobre o Estado Atual (corrigida)

| Capacidade | Status real |
|:---|:---|
| Detectar pausa curta (VAD) | 🟡 `VAD_DETECT` é DRAFT (`0x46`, Fase 5) — **não** ✅ |
| Detectar fim de turno (3s) | 🟡 implementável com `CYCLES_COUNT`+`ADD`+`COMPARE` (construção no §5; `VAD_DETECT` pendente) |
| Manter contexto pendente | 🔵 `HOLD_CONTEXT` proposto — necessidade em análise (§5) |
| Reformular com nova fala | 🔵 `RESUME_QUERY` proposto — necessidade em análise (§5) |
| Detectar abandono ("deixa", "esquece") | 🔴 precisa caminho ASR→texto + comparação textual (Temporal 32-stream + text ops — após Fase 5) |
| Aprender padrão de pausa | 🔴 Trilha P (LoRA; RAG Fase 6 antes) |

## Resumo (alvo)

O sistema deve poder:

- Esperar você pensar (sem interromper)
- Manter o contexto pendente (não descarta)
- Reformular quando você retoma (integra as duas falas)
- Detectar abandono ("deixa, esquece")
- Aprender seu padrão (LoRA em background)

Isso é o que separa um assistente de uma pessoa. A pessoa entende que silêncio não é fim de conversa. A pessoa espera. A pessoa retoma.

---

## §5. Análise de engenharia — precisa mesmo de opcodes novos?

Aplicando o filtro do ESPEC §13 ("nativo só onde o lowering é inadequado")
a `HOLD_CONTEXT` / `RESUME_QUERY`:

### §5.1 O que cada peça realmente exige

| Peça | Veredito |
|:---|:---|
| **Estado de turno** (`INCOMPLETE/THINKING/RESUMED/PAUSE/CLOSED/SPEAKING/TURN_END/ABANDON`) | **Zero ISA.** É um enum num registrador (ou palavra de um tensor de controle). `LOAD rPendingState, rPending.STATE` vira `MOV`/`LOADI` + convenção documentada. |
| **Timeout de 2s/3s sem `SUB`** | **Zero ISA**, construção válida hoje: `ADD rDeadline, rLastSpeech, <timeout>` uma vez por turno; depois `COMPARE rNow, rDeadline, PRED=GE` + `IF_EQUAL`. `CYCLES_COUNT` (✅) + `ADD` (✅) + `COMPARE` com predicado (✅ RFC-0007) cobrem tudo. |
| **`IF_LESS/IF_GREATER`** | **Zero ISA:** `COMPARE X, T, PRED=LT/GT` + `IF_EQUAL label` (os 6 predicados têm goldens, ESPEC §16). |
| **Acumular fala pendente** (`STREAM_WRITE` anel) | `SENSE` já deposita PCM em TEMPORAL; `CODEC_ENC` gera códigos. Falta o **anel nomeado + append** — `CONCAT` (🟡 `0x2F`, Fase 2) resolve acumular tensores de códigos. Nenhum opcode novo se `CONCAT` existir. |
| **`HOLD_CONTEXT` (salvar estado pendente)** | **Redutível a:** `TENSOR` alloc (✅) + `CONCAT` (🟡) + enum de estado em registrador + `FORK` para rollback (✅). É convenção de programa, não opcode — *a menos que* se prove custo inadequado (cópia por append vs. view com offset; `SLICE` ✅ sugere views viáveis). **Pendente de prova, não de fé.** |
| **`RESUME_QUERY` (pendente + nova fala)** | **É `CONCAT` com outro nome** (🟡 `0x2F`). Se `CONCAT` existir, `RESUME_QUERY` é alias de programa. Recomenda-se **não codificar**. |
| **`KIND=FINALIZE/RESUME` no `SIGNAL`** | Os 4 kinds congelados não mudam. Alternativa com ISA atual: `PING` (✅ local) + palavra de controle em memória compartilhada (ou `TRACE_EVENT` como side-channel — sujo; melhor: registrador dedicado lido após `PING`). Ou estender kinds via RFC futura (muda encoding congelado — custo alto, evitar). |
| **Detecção de abandono por frase** | **Genuinamente futura:** precisa ASR (Temporal 32-stream, Fase 5) + comparação textual (sem op de string; `COMPARE` é numérico). Heurística intermediária honesta: abandono ≈ fala curta (<1s) seguida de silêncio longo — implementável com `VAD_DETECT` + timer, sem NLP. |
| **Antecipação (prefetch RAG na pausa)** | Genuinamente futura: RAG (Fase 6) + perfil de usuário (Trilha P). |

### §5.2 Opções de desenho

- **(a) Dois opcodes novos** (`HOLD_CONTEXT`, `RESUME_QUERY`) — proposta do rascunho. Custo: 2 encodings permanentes + RFCs + goldens + Lean (stateful? declarar região/custo/ABORT por ESPEC §6.4).
- **(b) Zero opcodes novos (recomendado):** `CONCAT` (Fase 2) + enum em registrador + construção de timeout (§5.1) + `PING` + palavra de controle. Todo o MVP de pausa sai sem tocar o ISA — e o §13 manda preferir lowering adequado.
- **(c) Um opcode só (`TURN_HOLD`, se (b) provar custo inadequado):** append com offset + máquina de estados + timestamp num único `FAMILY-2` declarado. Só se medição mostrar que `CONCAT`-por-append estoura retenção/latência.

### §5.3 Recomendação

**Congelar (b) como desenho.** Escrever a RFC do MVP de pausa *contra*
(b): programa `.m3asm` usando `VAD_DETECT` (Fase 5) + `CONCAT` (Fase 2) +
construção de timeout + convenção de registrador de estado. Se, ao
implementar, o append ingênuo violar janela de retenção ou latência de
80ms/frame com medição na mão, aí — e só aí — propor (c) com os números.
`RESUME_QUERY` como opcode está **desrecomendado** (alias de `CONCAT`).

A estimativa do rascunho ("talvez 2 semanas") passa a valer para o MVP
na opção (b), **após** Fase 2 (`CONCAT`) + Fase 5 (`VAD_DETECT`) —
nenhuma das duas existe hoje.

## §6. Matriz de construtibilidade do MVP

| Construção do MVP | Hoje | Com o plano |
|:---|:---|:---|
| Loop `SENSE` + timeout 2s/3s | ✅ (`CYCLES_COUNT`+`ADD`+`COMPARE PRED`) | — |
| Estados de turno (enum) | ✅ (registrador + `LOADI`/`MOV`/`COMPARE`/`IF_EQUAL`) | — |
| Acumular códigos pendentes | 🔴 | 🟡 `CONCAT` (Fase 2) |
| Detectar fala vs. silêncio | 🔴 | 🟡 `VAD_DETECT` (Fase 5) |
| Sinalizar reasoning thread | ✅ parcial (`PING` + palavra de controle) | 🟡 kinds ricos (Fase 7, evitar) |
| Concatenar retomada | 🔴 | 🟡 `CONCAT` (= `RESUME_QUERY` por alias) |
| Abandono heurístico (fala curta + silêncio) | 🔴 | 🟡 Fase 5 |
| Abandono por frase ("esquece") | 🔴 | 🔴 pós-Fase 5 (ASR-texto) |
| Antecipação de RAG/LoRA | 🔴 | 🔴 Fase 6 + Trilha P |

**Demo mínima honesta (só ISA v1.5 de hoje):** máquina de estados de
turno com timeout simulando VAD por `SENSE TOKEN` + `USER_INPUT`
(`programs/thinking_sample.m3asm` já demonstra o padrão
`SENSE`/`IF_INTERRUPT`) — prova a lógica de pausa sem áudio real.

## §7. Mapa de fases

| Capacidade | Desbloqueia em |
|:---|:---|
| Lógica de pausa (MVP simulado) | Hoje (ISA v1.5) |
| Pausa com áudio real | Fase 2 (`CONCAT`) + Fase 5 (`VAD_DETECT`, `STREAM_MERGE`) |
| Retomada integral | Fase 2 (`CONCAT` como `RESUME_QUERY`) |
| Abandono heurístico | Fase 5 |
| Abandono por frase | Pós-Fase 5 (caminho ASR→texto) |
| Antecipação | Fase 6 (RAG) + Trilha P (LoRA) |
| `HOLD_CONTEXT`/`RESUME_QUERY` como opcodes | **Somente se §5.2(b) falhar com medição** |

---

*Documento companheiro de `docs/EXPERIENCIA_PREMIUM.md` (a sensação —
seção "Interrupção") e `docs/VISAO_PREMIUM_250GB.md` (Anexo A/B).
Nada aqui altera ESPEC.md (normativo) — opcodes novos, se um dia
propostos, exigem RFC própria com prova §13.*

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
