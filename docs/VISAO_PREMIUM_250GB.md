# M³-AVM Premium — Assistente de Voz com 250GB VRAM

```text
Status:   META-ALVO (target vision, NÃO-normativo, NÃO implementado)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Extends:  docs/ESPEC.md (ISA v1.5, normativo) + docs/ESPEC-V2.md (v2.0 DRAFT)
License:  AGPL-3.0-or-later (see LICENSE)
```

> **Contrato de honestidade (ESPEC §1.3):** tudo neste documento é **alvo**,
> não afirmação. Nenhum número de latência, nenhum opcode e nenhuma
> capacidade aqui descrita existe até o Anexo A dizer `✅ IMPL`.
> O programa da Parte 2 **não monta hoje** — vive neste documento como
> listagem-alvo (ver §0.3), não em `programs/`.

## §0. Como ler este documento

**Legenda (corrigida contra a árvore em 2026-09-11; ISA v1.5):**

| Marcador | Significado |
|:---|:---|
| ✅ | **IMPL hoje** — opcode/modo existe na árvore e passa em teste |
| 🟡 | **DRAFT em ESPEC-V2 §3** — encoding reservado com semântica proposta; precisa RFC + implementação (Fases 2–9 do plano) |
| 🔴 | **Bloqueado** — HELD/RESERVED por R12, ou diretiva/modo de assembler inexistente (precisa prova R12 ou RFC de assembler — Trilha P / Fase 9) |

**Correções aplicadas ao rascunho original:** a legenda dizia "v1.3" (a linha
atual é **v1.5**); `YIELD`, `LOCK/UNLOCK`, `PREEMPT_CHECK`, `TRACE_EVENT`,
`SET_DEADLINE`, `FOREST`, `SLICE` estavam marcados 🟡 mas são ✅ IMPL;
`SIGNAL`, `PREEMPT_CHECK`, `DEADLINE` listados como "faltantes" na Parte 8
existem em forma local/básica — o que falta são as formas estendidas.
**A fonte da verdade é o Anexo A**, não os marcadores inline.

### §0.1 O que a Parte 2 exige para montar

O programa-alvo usa, além do ISA v1.5 implementado:

1. **Diretivas e controle de assembler** (follow-ups RFC-0008, 🔴):
   `.data/.equ/.str/.text/.param`, `CALL/RET`, `LOADSTR`, literais string —
   o assembler estrito atual rejeita todos.
2. **Modos com kwargs que não existem** (🔴, precisam RFC de extensão):
   `SENSE ... LEN=`, `STREAM ... CHANNEL=`, `ATTN ... NHEADS=/HDIM=/CAUSAL=/STREAM=/LAYER=`,
   `EMBED ... DIM=`, `FOREST ... MODE=PROBABILITY`, `TRACE_EVENT "str"/KV`,
   `SIGNAL NODE=<nome>/URGENCY=`, `FORK` sem prioridade/label.
   (As formas implementadas estão no Anexo A.)
3. **Opcodes DRAFT** (🟡, Fases 2–6): `LOAD_MODEL`, `SPAWN_CONTEXT`,
   `DEPFORMER`, `STREAM_MERGE`, `VAD_DETECT`, `RAG_INDEX_ADD/SEARCH`,
   `EMBED_LOOKUP`, `CONCAT`, `MEMSET`, `FLASH_ATTN`, `SOFTMAX`, `WFI`.
4. **Opcodes em faixas RESERVADAS** (🔴, conflitam com ESPEC-V2 R12 — ver
   Anexo A): `DEVICE_QUERY/SELECT/HEALTH` (`0xB7–0xB9`, placeholder
   federado), `PLACE` (`0xBA`), `PIN` (`0xC0`, faixa TEE), `DEADLINE_CHAIN`
   (`0xD3`, faixa fotônica), `TRACE_SPAN_*` (`0xE0`, faixa quântica),
   `GRAD_ZERO/OPTIMIZER_STEP/CHECKPOINT_SAVE` (`0xF5/0xF6/0xFA`, sistema
   estendido), `MODEL_LOAD_UNIFIED` (`0xFC`). **Esses números são proposta
   do rascunho, não reserva válida** — a implementação exige RFC que ou
   realoca para faixas livres ou apresenta prova R12.
5. **Transporte cluster real** (Fase 7): `SIGNAL` cross-node, tabela de
   roteamento (`NODE="nome"`), `MIGRATE`/WAL. Hoje: só local (RFC-0018).

### §0.2 Regra do programa-alvo

Enquanto §0.1 não estiver resolvido, `premium_voice_assistant.m3asm`
**não entra em `programs/`** (quebraria o gate do assembler estrito,
RFC-0008 §5). Ele promove para `programs/` fatiado por marcos, na ordem
da Parte 8 — cada fatia só entra quando monta e executa de verdade.

---

> **Configuração:** PersonaPlex-7B + Mamba-2.8B + Llama-405B + BGE-M3 + XGBoost + RAG (100M docs)
>
> **Hardware alvo:** 2× H200 (141GB cada) = 282GB VRAM / 4× A100 80GB = 320GB / 3× H100 80GB = 240GB
>
> **Latência (METAS):** first sound <200ms · barge-in <300µs · RAG <100ms · raciocínio profundo 1–2s

---

## Parte 1 — Layout de Hardware (250GB, alvo)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    HARDWARE TOPOLOGY (ALVO)                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  GPU 0 (H200, 141GB)                                                 │
│  ├─ PersonaPlex-7B (Temporal + Depformer + Mimi)     ~5 GB          │
│  ├─ Audio ring buffers + ativações                   ~5 GB          │
│  └─ KV cache PersonaPlex (17 streams × 128k ctx)    ~30 GB          │
│                                                                      │
│  GPU 1 (H200, 141GB)                                                 │
│  ├─ Llama-405B Q4_K                                ~200 GB          │
│  ├─ KV cache Llama (128k tokens)                    ~25 GB          │
│  └─ Ativações                                        ~5 GB          │
│                                                                      │
│  GPU 2 (H100, 80GB) [opcional / fallback]                            │
│  ├─ Mamba-2.8B Q4_K                                  ~2 GB          │
│  ├─ BGE-M3 (embeddings)                              ~1 GB          │
│  ├─ XGBoost (classificador)                        ~0.1 GB          │
│  ├─ RAG index (IVF, 100M docs)                      ~50 GB          │
│  └─ LoRA adapters (training)                         ~5 GB          │
│                                                                      │
│  Total: ~320 GB (cabe em 4× A100 80GB)                              │
│  Versão mínima: ~250 GB (2× H200 + 1× H100)                         │
└──────────────────────────────────────────────────────────────────────┘
```

### Topologia de interconexão (alvo)

```
  GPU 0 ←──NVLink 4.0 (900 GB/s, ~200ns)──→ GPU 1
    ↑                                          ↑
    │                                          │
    └──NVSwitch──→ GPU 2 ←──NVLink 4.0─────────┘

  CPU ←──PCIe Gen5 x16 (64 GB/s, ~1µs)──→ GPUs
```

> Números de banda/latência do tecido são especificação do fabricante,
> não medição nossa. `PLACE`/`PIN`/`DEVICE_*` (🔴) são a contrapartida em
> ISA desse layout — ver Anexo A.

---

## Parte 2 — O Programa Completo (listagem-alvo, NÃO monta hoje)

`programs/premium_voice_assistant.m3asm` _(destino futuro, ver §0.2)_

```asm
; ============================================================
; M³-AVM PREMIUM — Voice Assistant (250GB VRAM)
; File    : programs/premium_voice_assistant.m3asm  [ALVO — ver §0.2]
; Target  : 2× H200 + 1× H100 (ou 4× A100)
; Models  : PersonaPlex-7B + Mamba-2.8B + Llama-405B + BGE-M3 + XGBoost
; Latency : <200ms first sound · <300µs barge-in · <100ms RAG  [METAS]
; Author  : Matheus de Camargo Marques
; ORCID   : 0009-0003-4518-2258
; License : AGPL-3.0-or-later
; ============================================================

.data                                                        ; 🔴 assembler
    ; --- Audio ---
    PCM_FRAME_LEN       .equ 1920          ; 80ms @ 24kHz    ; 🔴 assembler
    N_CODEBOOKS         .equ 16                              ; 🔴 assembler
    N_STREAMS           .equ 17            ; 1 text + 8 user + 8 agent  ; 🔴

    ; --- Model paths ---
    PATH_PERSONA        .str "models/personaplex-7b-Q4_K.gguf"  ; 🔴 assembler
    PATH_MAMBA          .str "models/mamba-2.8b-Q4_K.gguf"      ; 🔴
    PATH_LLAMA          .str "models/llama-405b-Q4_K.gguf"      ; 🔴
    PATH_BGE            .str "models/bge-m3.safetensors"        ; 🔴
    PATH_XGB            .str "models/intent_classifier.onnx"    ; 🔴
    PATH_RAG            .str "models/rag_index_ptbr_100M.bin"   ; 🔴

    ; --- Deadlines (ns) ---
    DL_AUDIO            .equ 80000000      ; 80ms/frame      ; 🔴 assembler
    DL_FIRST_SOUND      .equ 200000000     ; 200ms           ; 🔴
    DL_BARGE_IN         .equ 300000        ; 300µs           ; 🔴
    DL_RAG              .equ 100000000     ; 100ms           ; 🔴
    DL_FAST_PATH        .equ 50000000      ; 50ms            ; 🔴
    DL_DEEP_PATH        .equ 2000000000    ; 2s              ; 🔴

    ; --- Thresholds ---
    VAD_THRESHOLD       .equ 0.5                             ; 🔴
    RAG_TOPK            .equ 5                               ; 🔴
    COMPLEXITY_THRESH   .equ 0.4           ; acima = deep path  ; 🔴
    XGB_CONF_THRESH     .equ 0.75                            ; 🔴
    TEMPERATURE_FAST    .equ 0.5                             ; 🔴
    TEMPERATURE_DEEP    .equ 0.3                             ; 🔴
    TOP_K               .equ 50                              ; 🔴
    MAX_TOKENS          .equ 200                             ; 🔴

    ; --- Context window ---
    LLAMA_CTX           .equ 131072        ; 128k            ; 🔴
    MAMBA_STATE         .equ 8192                            ; 🔴

.text                                                        ; 🔴 assembler

; ============================================================
; BOOTSTRAP — Carrega e distribui os 6 modelos
; ============================================================
main:
    TRACE_SPAN_START rSpanBoot, NAME="bootstrap_250gb"     ; 🔴 0xE0 (faixa reservada!)

    ; ============ STEP 1: Descoberta de hardware ============
    DEVICE_QUERY rGpu0, KIND=GPU, MIN_MEM_MB=140000        ; 🔴 0xB7 (faixa reservada!)
    DEVICE_QUERY rGpu1, KIND=GPU, MIN_MEM_MB=140000, EXCLUDE=rGpu0   ; 🔴
    DEVICE_QUERY rGpu2, KIND=GPU, MIN_MEM_MB=70000,  EXCLUDE=[rGpu0,rGpu1]  ; 🔴

    DEVICE_SELECT rStatus, DEVICE=rGpu0                    ; 🔴 0xB8 (faixa reservada!)

    ; ============ STEP 2: Carrega modelos em paralelo ============
    FORK rSnapBoot                                         ; 🔴 forma: FORK exige prio/label (✅ 0x04 existe)
    PARALLEL_START N=6                                     ; 🔴 assembler/controle inexistente

    LOAD_MODEL rPersona, PATH_PERSONA QUANT=Q4_K           ; 🟡 0xA0
    LOAD_MODEL rMamba,   PATH_MAMBA   QUANT=Q4_K           ; 🟡
    LOAD_MODEL rLlama,   PATH_LLAMA   QUANT=Q4_K           ; 🟡
    MODEL_LOAD_UNIFIED rBge, PATH_BGE                      ; 🔴 0xFC (faixa reservada!)
    MODEL_LOAD_UNIFIED rXgb, PATH_XGB                      ; 🔴
    RAG_INDEX_ADD rRag, FILE=PATH_RAG                      ; 🟡 0x50
        N_DOCS=100000000
        DIM=1024
        MODE=IVF
        N_PROBE=32

    PARALLEL_END                                           ; 🔴

    ; ============ STEP 3: Placement (LCT / RFC-0002) ============
    ; GPU0: PersonaPlex (deadline crítico de áudio)
    DEVICE_SELECT rStatus, DEVICE=rGpu0                    ; 🔴
    PLACE rPersona, DEVICE_MASK=0b0001                     ; 🔴 0xBA (faixa reservada!)
    PIN rPersona, TIER=HOT                                 ; 🔴 0xC0 (faixa TEE reservada!)

    ; GPU1: Llama-405B (raciocínio profundo, contexto longo)
    DEVICE_SELECT rStatus, DEVICE=rGpu1                    ; 🔴
    PLACE rLlama, DEVICE_MASK=0b0010                       ; 🔴
    PIN rLlama, TIER=HOT                                   ; 🔴

    ; GPU2: Mamba + BGE + XGB + RAG (fast path + retrieval)
    DEVICE_SELECT rStatus, DEVICE=rGpu2                    ; 🔴
    PLACE rMamba, DEVICE_MASK=0b0100                       ; 🔴
    PLACE rBge,   DEVICE_MASK=0b0100                       ; 🔴
    PLACE rXgb,   DEVICE_MASK=0b0100                       ; 🔴
    PIN rMamba, TIER=HOT                                   ; 🔴

    ; ============ STEP 4: Deadlines compostos (RFC-0005) ============
    DEADLINE_CHAIN rChain, [                               ; 🔴 0xD3 (faixa reservada!)
        rDlAudio, rDlFirstSound, rDlDeep
    ]
    SET_DEADLINE rDlAudio,       DL_AUDIO                  ; ✅ 0x71 existe (forma: SET_DEADLINE Rs)
    SET_DEADLINE rDlFirstSound,  DL_FIRST_SOUND            ; ✅
    SET_DEADLINE rDlDeep,        DL_DEEP_PATH              ; ✅

    ; ============ STEP 5: Spawn contexts ============
    SPAWN_CONTEXT rCtxAudioIn,  ENTRY=audio_input_loop,  PRIORITY=RED     ; 🟡 0xA2
    SPAWN_CONTEXT rCtxAudioOut, ENTRY=audio_output_loop, PRIORITY=RED     ; 🟡
    SPAWN_CONTEXT rCtxReason,   ENTRY=reason_loop,       PRIORITY=BLUE    ; 🟡
    SPAWN_CONTEXT rCtxTrain,    ENTRY=background_loop,   PRIORITY=GREEN   ; 🟡

    TRACE_SPAN_END rSpanBoot                               ; 🔴

    JUMP idle_loop                                        ; ✅ 0x0E


idle_loop:
    YIELD                                                   ; ✅ 0x70
    JUMP idle_loop                                        ; ✅


; ============================================================
; AUDIO INPUT (RED) — Captura contínua + VAD + barge-in
; ============================================================
audio_input_loop:
    ; --- Capture 80ms PCM frame ---
    SENSE rPCM, AUDIO_PCM, LEN=PCM_FRAME_LEN                ; 🔴 modo: SENSE existe (✅ 0x06) mas LEN= não existe

    ; --- VAD detection ---
    VAD_DETECT rVAD, rPCM THRESHOLD=VAD_THRESHOLD           ; 🟡 0x46
    COMPARE rVAD, VAD_THRESHOLD                             ; ✅ 0x0C (com imediato; predicados RFC-0007)
    IF_GREATER rVAD, handle_barge_in                       ; 🔴 não existe (só IF_EQUAL/IF_INTERRUPT; ver Anexo A)

    ; --- Encode to Mimi codes ---
    CODEC_ENC rCodesIn, rPCM TENSOR                         ; ✅ 0x15

    ; --- Publish to ring ---
    LOCK rRingLock                                          ; ✅ 0x75 (try-lock; blocking é Fase 8)
    STREAM_WRITE rAudioInRing, rCodesIn                     ; 🔴 modo inexistente (STREAM ✅ 0x03 existe)
    UNLOCK rRingLock                                        ; ✅ 0x76

    ; --- Notify reasoning thread ---
    SIGNAL NODE=LOCAL, CTX=rCtxReason, KIND=PING            ; 🔴 forma: SIGNAL existe (✅ 0x1B) mas só NODE=0 numérico + kind imediato

    JUMP audio_input_loop                                 ; ✅


handle_barge_in:
    ; ============ BARGE-IN: atomic abort across all models ============
    TRACE_EVENT "barge_in_detected" T=CLOCK_QUERY           ; 🔴 forma: TRACE_EVENT existe (✅ 0x6B) mas toma registradores, não string/KV

    ; --- Signal all contexts ---
    SIGNAL NODE=LOCAL, CTX=rCtxReason,   KIND=ABORT         ; 🔴 forma (opcode ✅)
    SIGNAL NODE=LOCAL, CTX=rCtxAudioOut, KIND=ABORT         ; 🔴 forma (opcode ✅)
    SIGNAL NODE=LOCAL, CTX=rCtxTrain,    KIND=ABORT         ; 🔴 forma (opcode ✅)

    ; --- Local rollback (audio input) ---
    ABORT rSnapAudioIn                                      ; ✅ 0x05 (forma: ABORT Rs_ctx, Rs_ts)

    ; Note: other contexts rollback their own state upon receiving ABORT.
    ; The rollback covers:
    ;   - PersonaPlex KV (17 streams × 32 layers)            🟡 RFC-0008 == KV 17-stream (Fase 5; RFC-0010 follow-up)
    ;   - Mamba ssm_states                                   ✅ 0x14
    ;   - Llama KV_CACHE                                     ✅ via ABORT CoW (KV 17-stream é 🟡)
    ;   - LoRA gradients (if training active)                🟡 RFC-0010 == KV_COMPRESS/PIN/RETRIEVE (Fase 3)

    JUMP audio_input_loop                                 ; ✅


; ============================================================
; AUDIO OUTPUT (RED) — Synthesize + play
; ============================================================
audio_output_loop:
    ; --- Wait for codes from reasoning thread ---
    LOCK rRingLock                                        ; ✅
    STREAM_READ rCodesOut, rAudioOutRing BLOCKING           ; 🔴 modo inexistente (opcode STREAM ✅)
    UNLOCK rRingLock                                        ; ✅

    ; --- Merge 17 streams ---
    STREAM_MERGE rMixed, rAgentOut, rUserIn                 ; 🟡 0x45
        N_STREAMS=N_STREAMS
        MODE=mix
        GAIN=0.85

    ; --- Decode Mimi ---
    CODEC_DEC rAudioOut, rMixed TENSOR                      ; ✅ 0x16

    ; --- Align timestamps ---
    AUDIO_ALIGN rAlign, rPCMRef, rAudioOut                  ; ✅ 0x17 (SR=/FRAME=/HZ=/DELAY= existem)
        SR=24000 FRAME=80 HZ=1250 DELAY=80

    ; --- Stream to speaker ---
    STREAM rAudioOut, CHANNEL=4 BLOCKING                    ; 🔴 modo: STREAM existe (✅) mas CHANNEL= não existe

    JUMP audio_output_loop                                ; ✅


; ============================================================
; REASONING (BLUE) — Orquestração PersonaPlex + Mamba + Llama
; ============================================================
reason_loop:
    TRACE_SPAN_START rSpanTurn, NAME="conversation_turn"    ; 🔴

    ; ============ STEP 1: ASR via PersonaPlex Temporal ============
    TRACE_SPAN_START rSpanAsr, NAME="asr_personaplex"       ; 🔴

    LOCK rRingLock                                        ; ✅
    STREAM_READ rCodesIn, rAudioInRing BLOCKING             ; 🔴 modo (opcode ✅)
    UNLOCK rRingLock                                        ; ✅

    ; Merge user streams only (for ASR)
    STREAM_MERGE rUserMixed, rCodesIn, rCodesIn             ; 🟡 0x45
        N_STREAMS=N_STREAMS
        MODE=user_only

    ; Temporal forward (32 layers)
    CALL temporal_forward rUserMixed, rHidden, 0            ; 🔴 CALL não existe (follow-up RFC-0008)

    ; Text head (with sampling)
    MATVEC rTextLogits, rHidden, rPersona.text_head         ; ✅ 0x10 (field access `a.b` é notação-alvo, 🔴 hoje)
    SAMPLE rTranscript, rTextLogits TEMPERATURE=0.1 TOPK=1  ; ✅ 0x0B (TEMP=/TOPK= existem)

    TRACE_SPAN_END rSpanAsr                                 ; 🔴

    ; ============ STEP 2: Complexity classification (XGBoost) ============
    TRACE_SPAN_START rSpanClassify, NAME="classify_intent"  ; 🔴

    ; Embed query with BGE-M3
    EMBED rQueryEmb, rTranscript, rBge DIM=1024             ; 🔴 modo: EMBED existe (✅ 0x09) mas DIM= não existe

    ; Build feature vector
    CONCAT rFeatures, rQueryEmb, rTurnMeta                  ; 🟡 0x2F

    ; XGBoost evaluation
    FOREST rClassify, rFeatures, rXgb                       ; ✅ 0x22 existe (kwargs TREES=/DEPTH=; MODE=PROBABILITY 🔴 — só VOTE/MEAN)
        TREES=500
        DEPTH=8
        MODE=PROBABILITY

    ; Extract (confidence, complexity)
    SLICE rConfidence, rClassify, 0, 1                      ; ✅ 0x2E (posicional START/LEN)
    SLICE rComplexity, rClassify, 1, 2                      ; ✅

    TRACE_SPAN_END rSpanClassify                             ; 🔴

    ; ============ STEP 3: RAG retrieval (always) ============
    TRACE_SPAN_START rSpanRag, NAME="rag_retrieval"         ; 🔴

    RAG_SEARCH rDocs, rQueryEmb, rRag                       ; 🟡 0x52
        METRIC=COSINE
        TOPK=RAG_TOPK
        MODE=IVF
        N_PROBE=32

    ; Load docs
    LOADI rI, 0                                           ; ✅ 0x78
rag_load_loop:
    COMPARE rI, RAG_TOPK                                    ; ✅
    IF_EQUAL rI, rag_done                                   ; ✅ 0x0D
    EMBED_LOOKUP rDoc, rRag.docs, rDocs[rI]                 ; 🟡 0x53 (indexação `r[rI]` é notação-alvo 🔴)
    CONCAT rContext, rContext, rDoc                         ; 🟡
    ADD rI, rI, 1                                           ; ✅ 0x0A
    JUMP rag_load_loop                                    ; ✅

rag_done:
    TRACE_SPAN_END rSpanRag                                 ; 🔴

    ; ============ STEP 4: Routing decision ============
    ; If confidence < threshold → escalate to human
    COMPARE rConfidence, XGB_CONF_THRESH                    ; ✅
    IF_LESS rConfidence, escalate_to_human                  ; 🔴 (só IF_EQUAL/IF_INTERRUPT)

    ; If complexity < threshold → fast path (Mamba)
    COMPARE rComplexity, COMPLEXITY_THRESH                  ; ✅
    IF_GREATER rComplexity, deep_path                       ; 🔴
    JUMP fast_path                                        ; ✅


; ============================================================
; FAST PATH — Mamba-2.8B only (~50ms) [META]
; ============================================================
fast_path:
    TRACE_SPAN_START rSpanFast, NAME="fast_path_mamba"      ; 🔴

    ; --- Set deadline for fast response ---
    SET_DEADLINE rDlFast, DL_FAST_PATH                      ; ✅ (forma: SET_DEADLINE Rs)

    ; --- Fork for rollback ---
    FORK rSnapFast                                          ; 🔴 forma (opcode ✅ 0x04)

    ; --- Switch to Mamba engine ---
    CTX_SWITCH MAMBA, BLUE                                  ; ✅ 0x18

    ; --- Build prompt ---
    CONCAT rPrompt, rTranscript, rContext                   ; 🟡
    EMBED rPromptEmb, rPrompt, rMamba.emb                   ; 🔴 modo DIM (opcode ✅)

    ; --- Mamba forward ---
    CALL mamba_forward rPromptEmb, rMamba, rOut             ; 🔴 CALL

    ; --- Sample response ---
    SAMPLE rResponse, rOut TEMPERATURE=TEMPERATURE_FAST TOPK=TOP_K  ; ✅

    TRACE_SPAN_END rSpanFast                               ; 🔴

    ; --- Check for barge-in before synthesis ---
    IF_INTERRUPT rFlag, interrupt_fast                      ; ✅ 0x0F

    JUMP synthesize_response                              ; ✅


interrupt_fast:
    ABORT rSnapFast                                       ; ✅
    SSM_RESET rH D_INNER=8192 D_STATE=16 LAYER=0            ; ✅ 0x14 (kwargs existem)
    JUMP reason_loop                                      ; ✅


; ============================================================
; DEEP PATH — Llama-405B + Mamba speculative (~1-2s) [META]
; ============================================================
deep_path:
    TRACE_SPAN_START rSpanDeep, NAME="deep_path_speculative" ; 🔴

    ; --- Set deep deadline ---
    SET_DEADLINE rDlDeep, DL_DEEP_PATH                      ; ✅

    ; --- Fork for rollback (covers KV + ssm_states) ---
    FORK rSnapDeep                                          ; 🔴 forma (opcode ✅)

    ; --- Build prompt with full context ---
    CONCAT rPrompt, rTranscript, rContext                   ; 🟡
    EMBED rPromptEmb, rPrompt, rLlama.emb                   ; 🔴 modo (opcode ✅)

    ; --- Speculative decoding loop ---
    ; Mamba drafts 5 tokens (GPU2), Llama verifies (GPU1)
    LOADI rTokenPos, 0                                    ; ✅
    LOADI rAccepted, 0                                    ; ✅
    LOADI rSpecLen, 5                                     ; ✅

speculative_loop:
    COMPARE rTokenPos, MAX_TOKENS                          ; ✅
    IF_EQUAL rTokenPos, spec_done                           ; ✅

    ; --- Draft: Mamba generates K tokens on GPU2 ---
    CTX_SWITCH MAMBA, BLUE                                ; ✅
    DEVICE_SELECT rStatus, DEVICE=rGpu2                    ; 🔴
    MATVEC rDraftLogits, rPromptEmb, rMamba.head           ; ✅ (field access 🔴)
    SAMPLE rDraftTokens, rDraftLogits TOPK=rSpecLen        ; 🔴 modo: TOPK= existe mas toma imediato, não registrador

    ; --- Verify: Llama processes on GPU1 ---
    CTX_SWITCH TRANSFORMER, BLUE                          ; ✅
    DEVICE_SELECT rStatus, DEVICE=rGpu1                    ; 🔴
    CALL llama_forward_step rPromptEmb, rLlama, rTargetLogits  ; 🔴 CALL

    ; --- Acceptance check ---
    CALL speculative_verify rDraftTokens, rTargetLogits, rAcceptedMask  ; 🔴 CALL

    ; --- Append accepted tokens ---
    CALL append_tokens rOutput, rDraftTokens, rAcceptedMask  ; 🔴 CALL

    ; --- Update states ---
    CALL mamba_update_state rOutput, rMamba                ; 🔴 CALL
    CALL llama_update_kv rOutput, rLlama                   ; 🔴 CALL

    ; --- Progress ---
    ADD rTokenPos, rTokenPos, rAccepted                    ; ✅
    JUMP speculative_loop                                 ; ✅

spec_done:
    SAMPLE rResponse, rOutput TEMPERATURE=TEMPERATURE_DEEP TOPK=TOP_K  ; ✅
    TRACE_SPAN_END rSpanDeep                              ; 🔴

    ; --- Check for barge-in ---
    IF_INTERRUPT rFlag, interrupt_deep                     ; ✅

    JUMP synthesize_response                              ; ✅


interrupt_deep:
    ; --- Rollback hybrid state ---
    ABORT rSnapDeep                                       ; ✅
    SSM_RESET rH D_INNER=8192 D_STATE=16 LAYER=0            ; ✅
    ; Llama KV rollback happens automatically via ABORT (CoW)
    JUMP reason_loop                                      ; ✅


; ============================================================
; ESCALATION — Low confidence → human operator
; ============================================================
escalate_to_human:
    TRACE_SPAN_START rSpanEsc, NAME="escalate"              ; 🔴

    SIGNAL NODE=HUMAN, KIND=PING URGENCY=HIGH               ; 🔴 forma (opcode ✅; nome/urgência exigem Fase 7)

    LOADSTR rResponse, "Vou transferir você para um especialista."  ; 🔴 LOADSTR não existe (LOADI ✅ existe)
    TRACE_SPAN_END rSpanEsc                               ; 🔴
    JUMP synthesize_response                              ; ✅


; ============================================================
; SYNTHESIZE RESPONSE — PersonaPlex Depformer
; ============================================================
synthesize_response:
    TRACE_SPAN_START rSpanSynth, NAME="synthesize_audio"    ; 🔴

    ; --- Switch to audio engine on GPU0 ---
    CTX_SWITCH AUDIO, RED                                 ; ✅
    DEVICE_SELECT rStatus, DEVICE=rGpu0                    ; 🔴

    ; --- Embed response text ---
    EMBED rTextEmb, rResponse, rPersona.emb                ; 🔴 modo (opcode ✅)

    ; --- Depformer generates audio codes ---
    DEPFORMER rAudioCodes, rTextEmb, rAudioEmb, rKVDep      ; 🟡 0x44
        DEP_LAYERS=6
        DEP_DIM=1024
        CODEBOOKS=N_CODEBOOKS
        ACOUSTIC_DELAY=1
        TEMPERATURE=0.8
        TOP_K=TOP_K
        STREAM_ID=0

    ; --- Publish to output ring ---
    LOCK rRingLock                                        ; ✅
    STREAM_WRITE rAudioOutRing, rAudioCodes                ; 🔴 modo (opcode ✅)
    UNLOCK rRingLock                                        ; ✅

    TRACE_SPAN_END rSpanSynth                             ; 🔴
    TRACE_SPAN_END rSpanTurn                               ; 🔴

    ; --- Log turn metrics ---
    TRACE_EVENT "turn_complete"                             ; 🔴 forma (opcode ✅ 0x6B)
        CONFIDENCE=rConfidence
        COMPLEXITY=rComplexity
        PATH=rChosenPath
        TOKENS=rTokenPos
        ACCEPTED=rAccepted

    JUMP reason_loop                                      ; ✅


; ============================================================
; BACKGROUND (GREEN) — Continuous LoRA learning
; ============================================================
background_loop:
    ; --- Check GPU idle ---
    DEVICE_HEALTH rHealth, DEVICE=rGpu2                     ; 🔴 0xB9 (faixa reservada!)
    COMPARE rHealth.utilization_pct, 20                     ; 🔴 field access (COMPARE ✅)
    IF_GREATER rHealth.utilization_pct, wait_and_yield      ; 🔴

    ; --- LoRA training step on recent conversation ---
    GRAD_ZERO rLora.grad                                    ; 🔴 0xF5 (faixa reservada!)
    CALL lora_train_step rConversationBuffer, rLora         ; 🔴 CALL
    OPTIMIZER_STEP rOpt, rLora.params                       ; 🔴 0xF6 (faixa reservada!)

    ; --- Every 1000 steps, save checkpoint ---
    MOD rTmp, rStep, 1000                                   ; 🔴 MOD não existe
    COMPARE rTmp, 0                                       ; ✅
    IF_EQUAL rTmp, save_checkpoint                         ; ✅

    ADD rStep, rStep, 1                                   ; ✅
    YIELD                                                   ; ✅
    JUMP background_loop                                  ; ✅

save_checkpoint:
    CHECKPOINT_SAVE rLora, PATH="/data/lora_ckpt"           ; 🔴 0xFA (faixa reservada!)
    JUMP background_loop                                  ; ✅

wait_and_yield:
    YIELD                                                   ; ✅
    WFI TIMEOUT=100ms                                       ; 🟡 0x99
    JUMP background_loop                                  ; ✅


; ============================================================
; SUB-ROUTINE: PersonaPlex Temporal forward (32 layers)
; ============================================================
; Input : rInput (embeddings)
; Output: rOutput (hidden states)
; Param : rStreamId (0 = text, 1-8 user, 9-16 agent)
; ============================================================
temporal_forward:
    .param rInput, rOutput, rStreamId                       ; 🔴 assembler

    TENSOR rHidden 1 4096 f32                             ; ✅ 0x01
    COPY rHidden, rInput                                    ; 🔴 COPY não existe (MOV ✅ 0x79 existe)

    LOADI rL, 0                                           ; ✅
temporal_layer_loop:
    COMPARE rL, 32                                        ; ✅
    IF_EQUAL rL, temporal_done                             ; ✅

    ; --- Self-attention ---
    NORM rX, rHidden, rPersona.normA[rL]                    ; ✅ 0x07 (indexação 🔴)
    MATVEC rQ, rX, rPersona.wq[rL]                          ; ✅ 0x10 (indexação 🔴)
    MATVEC rK, rX, rPersona.wk[rL]                          ; ✅
    MATVEC rV, rX, rPersona.wv[rL]                          ; ✅
    ROPE rQ, rQ POS=rPos HDIM=128 NHEADS=32                 ; ✅ 0x19 (kwargs existem)
    ROPE rK, rK POS=rPos HDIM=128 NHEADS=32                 ; ✅

    ATTN rAttn, rQ, rK, rV, rMask, rKV                      ; 🔴 modo: ATTN existe (✅ 0x02, 4 regs) mas NHEADS=/HDIM=/CAUSAL=/STREAM=/LAYER= não existem
        NHEADS=32
        HDIM=128
        CAUSAL
        STREAM=rStreamId
        LAYER=rL
        NOTIFY_EACH_HEAD              ; enables sub-layer preemption  (✅ este modo existe)

    MATVEC rProj, rAttn, rPersona.wo[rL]                    ; ✅ (indexação 🔴)
    ADD rHidden, rHidden, rProj                             ; ✅ 0x0A

    ; --- FFN ---
    NORM rX, rHidden, rPersona.normF[rL]                    ; ✅ (indexação 🔴)
    FFN rFFN, rX, rPersona.gate[rL], rPersona.up[rL], rPersona.down[rL]  ; ✅ 0x08 (indexação 🔴)
    ADD rHidden, rHidden, rFFN                            ; ✅

    ; --- Layer-level preemption check ---
    PREEMPT_CHECK rPreemptFlag                              ; ✅ 0x6D
    COMPARE rPreemptFlag, 1                               ; ✅
    IF_EQUAL rPreemptFlag, temporal_abort                   ; ✅

    ADD rL, rL, 1                                         ; ✅
    JUMP temporal_layer_loop                              ; ✅

temporal_abort:
    LOADI rStatus, 1                                      ; ✅
    RET                                                     ; 🔴 RET não existe

temporal_done:
    COPY rOutput, rHidden                                   ; 🔴 (MOV ✅ existe)
    LOADI rStatus, 0                                      ; ✅
    RET                                                     ; 🔴


; ============================================================
; SUB-ROUTINE: Mamba forward
; ============================================================
; Input : rEmb, rModel
; Output: rOut (logits)
; ============================================================
mamba_forward:
    .param rEmb, rModel, rOut                              ; 🔴

    TENSOR rH MAMBA_STATE 16 f32                           ; 🔴 forma (.equ; opcode ✅)

    LOADI rL, 0                                           ; ✅
mamba_layer_loop:
    COMPARE rL, rModel.n_layers                           ; ✅ (field access 🔴)
    IF_EQUAL rL, mamba_done                               ; ✅

    NORM rX, rEmb, rModel.norm[rL]                          ; ✅ (indexação 🔴)
    SSM_SCAN rY, rX, rH, rModel.pack[rL]                    ; ✅ 0x13 (D_INNER=/D_STATE=/LAYER= existem; indexação 🔴)
        D_INNER=MAMBA_STATE
        D_STATE=16
        LAYER=rL

    ADD rEmb, rEmb, rY                                    ; ✅

    NORM rX, rEmb, rModel.normF[rL]                         ; ✅ (indexação 🔴)
    FFN rFFN, rX, rModel.gate[rL], rModel.up[rL], rModel.down[rL]  ; ✅ (indexação 🔴)
    ADD rEmb, rEmb, rFFN                                  ; ✅

    ; --- Layer-level preemption ---
    PREEMPT_CHECK rPreemptFlag                              ; ✅
    COMPARE rPreemptFlag, 1                               ; ✅
    IF_EQUAL rPreemptFlag, mamba_abort                     ; ✅

    ADD rL, rL, 1                                         ; ✅
    JUMP mamba_layer_loop                                 ; ✅

mamba_abort:
    LOADI rStatus, 1                                      ; ✅
    RET                                                     ; 🔴

mamba_done:
    MATVEC rOut, rEmb, rModel.lm_head                      ; ✅ (field access 🔴)
    LOADI rStatus, 0                                      ; ✅
    RET                                                     ; 🔴


; ============================================================
; SUB-ROUTINE: Llama-405B forward (single step)
; ============================================================
; Input : rEmb, rModel
; Output: rLogits
; ============================================================
llama_forward_step:
    .param rEmb, rModel, rLogits                           ; 🔴

    LOADI rL, 0                                           ; ✅
llama_layer_loop:
    COMPARE rL, 126               ; Llama-405B has 126 layers  ; ✅
    IF_EQUAL rL, llama_done                               ; ✅

    ; --- Self-attention (with KV cache) ---
    NORM rX, rEmb, rModel.normA[rL]                         ; ✅ (indexação 🔴)
    MATVEC rQ, rX, rModel.wq[rL]                            ; ✅ (indexação 🔴)
    MATVEC rK, rX, rModel.wk[rL]                            ; ✅ (indexação 🔴)
    MATVEC rV, rX, rModel.wv[rL]                            ; ✅ (indexação 🔴)
    ROPE rQ, rQ POS=rPos HDIM=128 NHEADS=128                ; ✅
    ROPE rK, rK POS=rPos HDIM=128 NHEADS=128                ; ✅

    FLASH_ATTN rAttn, rQ, rK, rV, rMask, rKV                ; 🟡 0x3A
        NHEADS=128
        HDIM=128
        CAUSAL
        LAYER=rL

    MATVEC rProj, rAttn, rModel.wo[rL]                      ; ✅ (indexação 🔴)
    ADD rEmb, rEmb, rProj                                 ; ✅

    ; --- FFN (SwiGLU) ---
    NORM rX, rEmb, rModel.normF[rL]                         ; ✅ (indexação 🔴)
    FFN rFFN, rX, rModel.gate[rL], rModel.up[rL], rModel.down[rL]  ; ✅ (indexação 🔴)
    ADD rEmb, rEmb, rFFN                                  ; ✅

    ; --- Layer-level preemption (critical for barge-in) ---
    PREEMPT_CHECK rPreemptFlag                              ; ✅
    COMPARE rPreemptFlag, 1                               ; ✅
    IF_EQUAL rPreemptFlag, llama_abort                     ; ✅

    ADD rL, rL, 1                                         ; ✅
    JUMP llama_layer_loop                                 ; ✅

llama_abort:
    LOADI rStatus, 1                                      ; ✅
    RET                                                     ; 🔴

llama_done:
    MATVEC rLogits, rEmb, rModel.lm_head                   ; ✅ (field access 🔴)
    LOADI rStatus, 0                                      ; ✅
    RET                                                     ; 🔴


; ============================================================
; SUB-ROUTINE: Speculative verify
; ============================================================
speculative_verify:
    .param rDraft, rTarget, rMask                          ; 🔴

    SOFTMAX rDraftProbs, rDraft AXIS=-1                     ; 🟡 0x3C
    SOFTMAX rTargetProbs, rTarget AXIS=-1                   ; 🟡

    CALL kl_divergence rDraftProbs, rTargetProbs, rKL       ; 🔴 CALL
    COMPARE rKL, 0.1                                      ; ✅ (float imediato: RFC-0008 rejeita hoje — programas usam inteiros escalados, RFC-0007)
    IF_LESS rKL, accept                                    ; 🔴
    JUMP reject                                           ; ✅

accept:
    MEMSET rMask, 1 LEN=1                                   ; 🟡 0x2B
    RET                                                     ; 🔴

reject:
    MEMSET rMask, 0 LEN=1                                   ; 🟡
    SAMPLE rResampled, rTargetProbs TOPK=1                  ; ✅
    RET                                                     ; 🔴
```

---

## Parte 3 — Timeline de uma Conversa Real (metas, não medições)

### Turno 1 — Saudação simples [META]

```
T=0ms      User fala "Oi, tudo bem?"
T=80ms     Frame capturado, VAD detecta
T=90ms     CODEC_ENC → 16 codebooks
T=100ms    SIGNAL para reasoning thread
T=110ms    ASR via Temporal (PersonaPlex)
T=140ms    XGBoost: complexidade=0.15 (baixa) → FAST PATH
T=150ms    RAG retrieval (100ms típico)
T=170ms    Mamba forward (SSM_SCAN)
T=220ms    Resposta gerada: "Oi! Tudo ótimo, e você?"
T=230ms    Depformer gera áudio
T=250ms    CODEC_DEC reproduz
T=270ms    🔊 User ouve resposta
```

**First sound latency (META):** ~270ms. **Sensation (alvo):** instantâneo.

---

### Turno 2 — Pergunta profunda [META]

```
T=0ms      User fala "Me explica decoerência quântica"
T=80ms     Frame capturado
T=110ms    ASR completa
T=140ms    XGBoost: complexidade=0.85 (alta) → DEEP PATH
T=150ms    RAG retrieval (top-5 docs)
T=260ms    Llama-405B começa a gerar
T=400ms    Primeiro token gerado
T=420ms    Depformer começa a sintetizar
T=700ms    🔊 Primeiro som
T=2200ms   Resposta completa (200 tokens)
```

**First sound latency (META):** ~700ms. **Sensation (alvo):** "está pensando, mas não em silêncio."

---

### Turno 3 — Interrupção durante resposta [META]

```
T=0ms      User interrompe "não, não, eu quis dizer..."
T=80ms     VAD detecta nova fala
T=85ms     SIGNAL ABORT para todas as threads
T=86ms     Interrupt flag setada atomicamente
T=86.1ms   Llama verifica flag no próximo layer boundary
T=86.3ms   ABORT rSnapDeep → CoW rollback
T=86.5ms   PersonaPlex rollback KV 17-streams
T=86.6ms   Mamba rollback ssm_states
T=86.7ms   Estado restaurado atomicamente
T=87ms     Novo processamento começa
```

**Barge-in latency (META):** ~87µs + tempo de processamento do layer
corrente. Pior caso (alvo): **~200µs**. Meta do programa: 300µs.
Medir com `benches/barge_in.rs` (a criar — benches atuais cobrem
`physics_bench`/`hybrid_ops_bench`; ABORT 32×32 hoje mede ~25ms no
emulador CPU, ESPEC §17 — o caminho até µs é hardware + preempção por
chunk, não este emulador).

---

### Turno 4 — Aprendizado em background [META]

Enquanto o usuário dorme (ou entre turnos):

```
GREEN thread:
- GRAD_ACCUM sobre buffer de conversa
- OPTIMIZER_STEP (Adam)
- CHECKPOINT_SAVE a cada 1000 steps
- Monitoramento de temperatura via POWER_QUERY
```

Após 1000 turnos, o modelo começa a **antecipar** preferências do usuário.
(Training é non-goal do ESPEC §1.2 — esta parte é visão de produto, não
promessa do emulador.)

---

## Parte 4 — Métricas Alvo (metas, não medições)

| Métrica | Meta | Como medir |
|:---|:--:|:---|
| **First sound (fast path)** | <270ms | `M3_PROFILE=1` |
| **First sound (deep path)** | <700ms | `M3_PROFILE=1` |
| **Barge-in latency** | <300µs | `benches/barge_in.rs` (a criar) |
| **RAG search (100M)** | <100ms | Per query |
| **Mamba forward** | <50ms | Per step |
| **Llama-405B step** | <30ms | Per token |
| **Temporal (32 layers)** | <50ms | Per frame |
| **Depformer (6×16)** | <40ms | Per frame |
| **VRAM usage** | <250GB | `nvidia-smi` |
| **Conversation turns** | >50 sem degradação | Teste manual |
| **WER (ASR PT-BR)** | <12% | Dataset Common Voice |
| **MOS (audio quality)** | >4.2 | Teste subjetivo 10 pessoas |
| **LoRA improvement** | +15% preferência | A/B test 1000 turnos |

---

## Parte 5 — Orçamento (estimativa do rascunho, a validar)

| Item | Custo |
|:---|:--:|
| 2× H200 141GB (Lambda) | $8.50/h |
| 1× H100 80GB (RunPod) | $3.99/h |
| Storage 1TB NVMe | $0.10/GB/mês |
| **Total hora** | **~$13/h** |
| **8h/dia × 30 dias** | **~$3.120** |
| **3 meses de desenvolvimento** | **~$9.360** |

**Alternativa:** comprar 4× A100 80GB usado (~$40k) → break-even em 3.000h.

---

## Parte 6 — O Que Isso Entrega (visão)

Com 250GB e esse programa, o alvo é:

- **Conversa natural full-duplex** — interromper, rir, fazer "hmm"
- **Velocidade adaptativa** — Mamba responde em 50ms, Llama em 2s
- **Contexto real de 128k tokens** — lembrar da conversa de 30 min atrás
- **Barge-in atômico em <300µs** — estado híbrido restaurado junto
- **RAG de 100M docs** — conhecimento atualizado sem retreinar
- **Aprendizado contínuo** — LoRA em background
- **Classificação de intenção em <20ms** — XGBoost decide o path

---

## Parte 7 — Comparação com Estado da Arte (leitura do rascunho, datada)

| Sistema | Full-Duplex | Barge-in | Contexto | RAG | Aprendizado |
|:---|:--:|:--:|:--:|:--:|:--:|
| **ChatGPT Voice** | ❌ | parcial | 32k | externo | ❌ |
| **Claude Voice** | ❌ | ❌ | 200k | externo | ❌ |
| **Gemini Live** | ✅ | parcial | 1M | externo | ❌ |
| **Moshi (oficial)** | ✅ | ✅ | 8k | ❌ | ❌ |
| **M³-AVM Premium (META)** | ✅ | ✅ 300µs | 128k | nativo | ✅ LoRA |

> Tabela comparativa herdada do rascunho (2026-09-11); capacidades de
> terceiros mudam rápido — revalidar antes de citar.

---

## Parte 8 — Próximos Passos (reconciliados com o plano)

1. **Provisionar hardware:** 2× H200 + 1× H100 (~$13/h em Lambda)
2. **Implementar os opcodes/modos faltantes** (inventário real no Anexo A;
   o "18 opcodes" do rascunho subconta — só a Parte 2 usa ~15 DRAFT +
   ~20 bloqueados + ~25 modos de assembler):
   `LOAD_MODEL`, `SPAWN_CONTEXT`, `DEPFORMER`, `STREAM_MERGE`,
   `RAG_INDEX_*`, `EMBED_LOOKUP`, `CONCAT`, `MEMSET`, `FLASH_ATTN`,
   `SOFTMAX`, `VAD_DETECT`, `WFI`, `KV_COMPRESS`, `ATTN_SPARSE` +
   modos `SENSE LEN=`, `STREAM CHANNEL=`, `ATTN` estendido, `SIGNAL`
   cross-thread, diretivas/`CALL`/`RET`, branches `IF_GT/IF_LT`
3. **Baixar e quantizar modelos:** PersonaPlex-7B Q4_K, Mamba-2.8B Q4_K, Llama-405B Q4_K
4. **Construir índice RAG:** 100M docs PT-BR
5. **Fine-tuning:** PersonaPlex em corpus PT-BR (LoRA)
6. **Rodar e medir:** Ajustar thresholds com base em dados reais

**Marcos entregáveis (cada um monta+executa de verdade antes de entrar
em `programs/`):**

1. **Marco 1 (1× H100 80GB):** áudio in → transcrição → texto → áudio out
   (só ISA v1.5 de hoje + `VAD_DETECT` + `STREAM_MERGE` + `DEPFORMER`)
2. **Marco 2:** + Mamba fast path + RAG (`RAG_*`, `EMBED_LOOKUP`, `CONCAT`, `KV_COMPRESS`)
3. **Marco 3:** + Llama-405B deep path (`FLASH_ATTN`, `SOFTMAX`, `LOAD_MODEL`, `SPAWN_CONTEXT`)
4. **Marco 4:** barge-in coordenado 3 modelos (Fase 7 transporte + `ATTN_SPARSE`)
5. **Marco 5:** LoRA em background (Trilha P decide os opcodes `0xF5/0xF6/0xFA` ou aliases)

---

## Anexo A — Reconciliação com a árvore (fonte da verdade, 2026-09-11)

ISA v1.5 · suite 270 passed + 1 falha pré-existente (moshi norm-gamma).

### A.1 ✅ Já implementado (usar hoje)

| Item | Onde |
|:---|:---|
| `TENSOR/MATVEC/MUL/SILU/NORM/FFN/EMBED/ADD/SAMPLE/COMPARE` | `src/opcodes.rs`, RFC-0004/0007 |
| `SAMPLE TEMP=/TOPK=`, `COMPARE` predicados, `LOADI/MOV` | RFC-0007 |
| `FORK/ABORT`, `JUMP/IF_EQUAL/IF_INTERRUPT`, `HALT/NOP` | núcleo v1.x |
| `SENSE` (AUDIO/VAD/TOKEN/USER_INPUT/PCM/CODEC_FRAME) | `src/vm.rs` |
| `STREAM` (+`BLOCKING/DROP`), `SSM_SCAN/RESET`, `CODEC_ENC/DEC`, `AUDIO_ALIGN` (`SR=/FRAME=/HZ=/DELAY=`), `CTX_SWITCH`, `ROPE` (`POS=/HDIM=/NHEADS=/THETA=`) | híbridos IMPL |
| `GATHER/DISTANCE/RANK1_UPDATE`, `CONV`, `SPIKE/DENOISE/FOREST/ODE_STEP` | RFC-0004/0012–0015/0017 |
| `KV_TRUNCATE`, `TENSOR FILL`, `SLICE` (posicional), `REMOTE_SPAWN/SIGNAL/SEND_TENSOR/BARRIER` (local) | RFC-0010/0018/0019 |
| `RNG_*`, `HASH/CHECKSUM/HMAC` | RFC-0005 |
| `CYCLES_COUNT/TRACE_EVENT/SANITY_CHECK/PREEMPT_CHECK/ASSERT/DUMP/YIELD/SET-DEADLINE/GET-DEADLINE/PRIORITY_SET/GET/LOCK/UNLOCK/FENCE` | RFC-0006 |
| Assembler estrito (rejeita desconhecido) | RFC-0008 |

### A.2 🟡 DRAFT em ESPEC-V2 §3 (precisa RFC + código; Fases 2–6, 9)

| Item | Fase do plano |
|:---|:---|
| `LOAD_MODEL/UNLOAD_MODEL/SET/GET/MODEL_SWITCH` (`0xA0`, `0xA7–0xA9`), `SPAWN/KILL_CONTEXT` (`0xA2–0xA3`) | Fase 9 |
| `DEPFORMER` (`0x44`), `STREAM_MERGE` (`0x45`), `VAD_DETECT` (`0x46`), `AUDIO_RESAMPLE/FILTER/WINDOW` (`0x47–0x49`) | Fase 5 |
| `RAG_INDEX_ADD/DEL/SEARCH` (`0x50–0x52`), `EMBED_LOOKUP` (`0x53`), `PQ_ENCODE/DECODE` (`0x56–0x57`) | Fase 6 |
| `KV_COMPRESS` (`0x39`), `FLASH_ATTN` (`0x3A`), `ATTN_SPARSE` (`0x3B`), `SOFTMAX` (`0x3C`), ativações/unários (`0x3D–0x43`), shape ops (`0x30–0x37`) | Fase 3 |
| `CONCAT` (`0x2F`), `MEMCPY/MEMSET` (`0x2A–0x2B`), arena/snapshot (`0x26–0x29`), `PREFETCH/RESHAPE` (`0x2C–0x2D`) | Fase 2 |
| `CAST/QUANTIZE/DEQUANT` (`0x67–0x69`) | Fase 4 |
| `WFI` (`0x99`), `DMA_*`, `NIC_*`, `ATOMIC_*`, `GPU_LAUNCH/WAIT` (trap), `.m3bc` loader (parcial: `src/m3bc.rs`) | Fase 1/9 |
| KV 17-stream real, `SIGNAL` remoto, tabela de rotas, `MIGRATE`/WAL, SWIM | Fase 7 |

### A.3 🔴 Bloqueado — exige decisão antes de qualquer código

| Item | Bloqueio |
|:---|:---|
| `DEVICE_QUERY/SELECT/HEALTH` (`0xB7–0xB9`), `PLACE` (`0xBA`), `PIN` (`0xC0`), `DEADLINE_CHAIN` (`0xD3`), `TRACE_SPAN_*` (`0xE0`), `GRAD_ZERO`/`OPTIMIZER_STEP`/`CHECKPOINT_SAVE` (`0xF5/0xF6/0xFA`), `MODEL_LOAD_UNIFIED` (`0xFC`) | Faixas RESERVED (ESPEC-V2 R12). RFC deve **realocar** ou provar R12 |
| `.data/.equ/.str/.text/.param`, `CALL/RET`, `LOADSTR`, literais, `MOD`, `COPY` (usar `MOV`), `IF_GREATER/IF_LESS` | Assembler Fase 9 (follow-ups RFC-0008); `COPY`→`MOV`, `LOADSTR`→`LOADI`+dados |
| `SENSE LEN=`, `STREAM CHANNEL=`, `ATTN` kwargs, `EMBED DIM=`, `FOREST MODE=PROBABILITY`, `TRACE_EVENT` string/KV, `SIGNAL NODE=<nome>/URGENCY=`, `SAMPLE TOPK=<reg>`, `FORK` sem operando, `COMPARE` float imediato, field access `a.b`, indexação `r[i]` | Modos inexistentes — cada um precisa RFC de extensão (ou reescrita com o ISA atual: `IF_EQUAL`+`COMPARE PRED=`, `MOV`, `SLICE`, `GATHER`) |
| Training (LoRA/GRAD/OPTIMIZER) como opcodes | Non-goal ESPEC §1.2 + faixas reservadas → Trilha P decide (prova R12 ou aliases `.m3asm`) |

---

## Anexo B — Mapa para o plano de fases

| Marco (Parte 8) | Fases do plano | RFCs |
|:---|:---|:---|
| Marco 1 (voz básica) | Fase 5 (áudio) + Fase 2 (`CONCAT`) | 0028–0031, 0023–0024 |
| Marco 2 (fast+RAG) | Fase 6 (retrieval) + Fase 3 (`KV_COMPRESS`) | 0032, 0025–0027 |
| Marco 3 (deep path) | Fase 3 (attention) + Fase 9 (system) | 0025–0027, 0039–0040 |
| Marco 4 (barge-in 3 modelos) | Fase 7 (transporte) + Fase 8 (scheduler T2) | 0034–0038 |
| Marco 5 (LoRA bg) | Trilha P (provas R12) | 0033 |

---

*Meta-alvo viva: atualizar os marcadores ✅/🟡/🔴 à medida que as fases
do plano forem concluídas. Nada aqui altera ESPEC.md (normativo) nem
ESPEC-V2.md (draft) — mudanças de encoding exigem RFC própria.*

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
