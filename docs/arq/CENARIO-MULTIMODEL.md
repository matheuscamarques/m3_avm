> **ARQUIVO — material de entrada, NÃO normativo.**
> Recebido como "mais documentação" (multi-model support, central de
> atendimento, speculative decoding) em 2026-09-10. Etiquetas ✅/🟡 com o
> mesmo deslocamento dos cenários anteriores: `SIGNAL 0x1B` marcado ✅
> (aqui: RSVD); `LOCK/UNLOCK` usados SEM etiqueta (aqui: IMPL, RFC-0006);
> `SET_DEADLINE`/`TRACE_EVENT`/`SAMPLE-TOPK` sem etiqueta (aqui: IMPL).
> Sintaxe além do nosso assembler (ver revisão): `DIV`, `IF_LESS`,
> `CALL/RET/.param`, `LOADI/MOV/LOADSTR/MOD`, `STREAM_WRITE/STREAM_READ`,
> `PARALLEL_START/END`, `.data/.equ/.str`, `0b…`, `"strings"`, `1e-4`,
> `rX[i]`, acesso pontilhado, `COMPARE` com float, `ADD` sobre regs
> contadores (type error: ADD é elementwise de tensores). Ideia
> aproveitável: speculative decoding com sync via ABORT + KV_TRUNCATE
> (mecanismos reais). Números de latência/throughput sem medição.
> Canônico: `docs/ESPEC.md` (+ `docs/ESPEC-V2.md` DRAFT). Nada aqui deve
> ser implementado sem reconciliação. Conteúdo original verbatim abaixo.
>
> ---

# Multi-Model Integration — Sistema de Atendimento Inteligente

`programs/production/multi_model_support.m3asm`

> **Cenário:** central de atendimento por voz que combina 6 modelos heterogêneos no mesmo contexto, compartilhando memória, com roteamento inteligente e speculative decoding.
>
> **Modelos:**
> - **Moshi-7B** — interface de voz full-duplex (input/output)
> - **BGE-M3** — embeddings para RAG
> - **XGBoost** — classificador de intenção (leve, CPU)
> - **Mamba-2.8B** — draft model (rápido)
> - **Llama-70B Q4_K** — target model (preciso)
> - **RAG index** — base de conhecimento (100M docs)
>
> **Legenda:** ✅ v1.3 implementado · 🟡 spec/RFC congelada

---

## Visão Arquitetural do Programa

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VOICE INPUT (Moshi)                              │
│  SENSE PCM → CODEC_ENC → STREAM_MERGE → DEPFORMER                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  CLASSIFIER (XGBoost)  ← features: intent + confidence + complexity │
│  Decide: FAST PATH (Mamba) | DEEP PATH (Speculative Llama+Mamba)   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
┌───────────────────┐             ┌─────────────────────────────────┐
│   FAST PATH       │             │   DEEP PATH (Speculative)       │
│   Mamba-2.8B      │             │   Mamba (draft) → Llama (verify)│
│   ~50ms           │             │   ~300ms                        │
└─────────┬─────────┘             └──────────────┬──────────────────┘
          │                                      │
          └──────────────┬───────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│  RAG AUGMENTATION (BGE-M3 + RAG index)                              │
│  Top-5 docs injetados no contexto                                   │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│  VOICE OUTPUT (Moshi)                                               │
│  DEPFORMER → CODEC_DEC → STREAM                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Código Completo

```asm
; ============================================================
; M³-AVM PROGRAM — Multi-Model Customer Support
; File    : programs/production/multi_model_support.m3asm
; ISA     : v2.4
; Hardware: 1 rig (2 CPUs, 4 GPUs H100, 512GB RAM)
; Models  : 6 heterogeneous models, 1 unified context
; Author  : Matheus de Camargo Marques
; ============================================================

.data
    ; --- Dimensões ---
    PCM_FRAME_LEN        .equ 1920        ; 80ms @ 24kHz
    N_CODEBOOKS          .equ 16
    N_STREAMS            .equ 17
    EMB_DIM              .equ 1024        ; BGE-M3

    ; --- Modelos ---
    MODEL_MOSHI          .equ 0
    MODEL_BGE            .equ 1
    MODEL_XGB            .equ 2
    MODEL_MAMBA          .equ 3
    MODEL_LLAMA          .equ 4

    ; --- Paths ---
    PATH_MOSHI           .str "moshi-7b-Q4_K.gguf"
    PATH_BGE             .str "bge-m3.safetensors"
    PATH_XGB             .str "intent_classifier.onnx"
    PATH_MAMBA           .str "mamba-2.8b-Q4_K.gguf"
    PATH_LLAMA           .str "llama-70b-Q4_K.gguf"
    PATH_RAG             .str "knowledge_base.bin"

    ; --- Deadlines (ns) ---
    DL_VOICE_INPUT       .equ 80000000      ; 80ms
    DL_CLASSIFY          .equ 20000000      ; 20ms
    DL_FAST_PATH         .equ 50000000      ; 50ms
    DL_DEEP_PATH         .equ 300000000     ; 300ms
    DL_VOICE_OUTPUT      .equ 80000000      ; 80ms
    DL_TOTAL             .equ 500000000     ; 500ms

    ; --- Thresholds ---
    THRESH_CONFIDENCE    .equ 0.75
    THRESH_COMPLEXITY    .equ 0.40          ; acima = deep path
    THRESH_XGB_REVIEW    .equ 0.85

.text

; ============================================================
; BOOTSTRAP — Carrega e posiciona todos os modelos
; ============================================================
main:
    TRACE_SPAN_START rSpanBoot, NAME="bootstrap"           ; 🟡 0xE0

    ; --- 1. Descoberta de hardware ---
    DEVICE_QUERY rGpu0, KIND=GPU, MIN_MEM_MB=70000         ; 🟡 0xB7
    DEVICE_QUERY rGpu1, KIND=GPU, EXCLUDE=rGpu0
    DEVICE_QUERY rGpu2, KIND=GPU, EXCLUDE=[rGpu0,rGpu1]
    DEVICE_QUERY rGpu3, KIND=GPU, EXCLUDE=[rGpu0,rGpu1,rGpu2]

    ; --- 2. Carrega os 6 modelos em paralelo ---
    ;     (threads Tokio executam em paralelo, uma chamada por modelo)
    FORK rSnapBoot                                          ; ✅ 0x04
    PARALLEL_START N=5                                      ; 🟡 0xBB (usar SHARD por ora)

    LOAD_MODEL rMoshi, PATH_MOSHI QUANT=Q4_K                ; 🟡 0xA0
    MODEL_LOAD_UNIFIED rBge, PATH_BGE                       ; 🟡 0xFC
    MODEL_LOAD_UNIFIED rXgb, PATH_XGB                       ; 🟡 0xFC
    LOAD_MODEL rMamba, PATH_MAMBA QUANT=Q4_K                ; 🟡 0xA0
    LOAD_MODEL rLlama, PATH_LLAMA QUANT=Q4_K                ; 🟡 0xA0

    PARALLEL_END                                            ; 🟡

    ; --- 3. Placement por device (LCT) ---
    ;     Distribui modelos por GPUs conforme perfil de uso
    DEVICE_SELECT rStatus, DEVICE=rGpu0                     ; 🟡 0xB8
    PLACE rMoshi, DEVICE_MASK=0b0001                        ; 🟡 0xBA

    DEVICE_SELECT rStatus, DEVICE=rGpu1
    PLACE rMamba, DEVICE_MASK=0b0010

    DEVICE_SELECT rStatus, DEVICE=rGpu2
    PLACE rLlama, DEVICE_MASK=0b0100                        ; Llama-70B em GPU dedicada

    DEVICE_SELECT rStatus, DEVICE=rGpu3
    PLACE rBge,   DEVICE_MASK=0b1000
    PLACE rXgb,   DEVICE_MASK=0b1000                        ; BGE + XGB compartilham GPU3

    ; --- 4. Índice RAG ---
    RAG_INDEX_ADD rRag, FILE=PATH_RAG                       ; 🟡 0x50
        N_DOCS=100000000
        DIM=EMB_DIM
        MODE=IVF
        N_PROBE=32

    ; --- 5. Deadline composto do sistema ---
    DEADLINE_AND rTree, CHILDREN=[rDlIn, rDlClassify, rDlMain, rDlOut]  ; 🟡 0xD0
    SET_DEADLINE rDlIn,       DL_VOICE_INPUT
    SET_DEADLINE rDlClassify, DL_CLASSIFY
    SET_DEADLINE rDlMain,     DL_DEEP_PATH                  ; pior caso
    SET_DEADLINE rDlOut,      DL_VOICE_OUTPUT
    DEADLINE_CHAIN rChain, [rDlIn, rDlClassify, rDlMain, rDlOut]  ; 🟡 0xD3

    ; --- 6. Contextos concorrentes ---
    SPAWN_CONTEXT rCtxAudioIn,  ENTRY=audio_input_loop,  PRIORITY=RED    ; ✅/🟡 0xA2
    SPAWN_CONTEXT rCtxAudioOut, ENTRY=audio_output_loop, PRIORITY=RED
    SPAWN_CONTEXT rCtxMain,     ENTRY=main_loop,         PRIORITY=BLUE

    TRACE_SPAN_END rSpanBoot

    JUMP main_loop


; ============================================================
; AUDIO INPUT LOOP (RED) — captura e codifica áudio contínuo
; ============================================================
audio_input_loop:
    ; --- Captura frame PCM (bloqueante, 80ms) ---
    SENSE rPCM, AUDIO_PCM, LEN=PCM_FRAME_LEN                ; ✅ 0x06

    ; --- VAD local (barge-in) ---
    VAD_DETECT rVAD, rPCM THRESHOLD=0.5                     ; 🟡 0x46
    COMPARE rVAD, 0.5
    IF_GREATER rVAD, signal_barge_in

    ; --- Encode Mimi ---
    CODEC_ENC rCodesIn, rPCM TENSOR                         ; ✅ 0x15

    ; --- Publica no ring buffer compartilhado ---
    LOCK rRingLock                                          ; 🟡 0x75
    STREAM_WRITE rAudioRing, rCodesIn                       ; 🟡
    UNLOCK rRingLock                                        ; 🟡 0x76

    ; --- Notifica contexto principal ---
    SIGNAL NODE=LOCAL, CTX=rCtxMain, KIND=PING              ; ✅ 0x1B

    JUMP audio_input_loop

signal_barge_in:
    SIGNAL NODE=LOCAL, CTX=rCtxMain, KIND=ABORT             ; ✅ 0x1B
    JUMP audio_input_loop


; ============================================================
; AUDIO OUTPUT LOOP (RED) — sintetiza e reproduz áudio
; ============================================================
audio_output_loop:
    ; --- Espera dados do ring buffer ---
    LOCK rRingLock
    STREAM_READ rCodesOut, rAudioOutRing BLOCKING           ; 🟡
    UNLOCK rRingLock

    ; --- Merge dos 17 streams ---
    STREAM_MERGE rMixed, rAgentOut, rUserIn                 ; 🟡 0x45
        N_STREAMS=N_STREAMS
        MODE=mix
        GAIN=0.85

    ; --- Decode Mimi ---
    CODEC_DEC rAudioOut, rMixed TENSOR                      ; ✅ 0x16

    ; --- Alinhamento temporal ---
    AUDIO_ALIGN rAlign, rPCMRef, rAudioOut                  ; ✅ 0x17
        SR=24000 FRAME=80 HZ=1250 DELAY=80

    ; --- Stream para saída ---
    STREAM rAudioOut, CHANNEL=4 BLOCKING                    ; ✅ 0x03

    JUMP audio_output_loop


; ============================================================
; MAIN LOOP (BLUE) — pipeline de raciocínio multi-modelo
; ============================================================
main_loop:
    TRACE_SPAN_START rSpanTurn, NAME="conversation_turn"    ; 🟡 0xE0

    ; ============ FASE 1: Captura + ASR ============
    LOCK rRingLock
    STREAM_READ rCodesIn, rAudioRing BLOCKING               ; 🟡
    UNLOCK rRingLock

    ; --- Converte codes em texto via Moshi temporal ---
    CALL moshi_asr rCodesIn, rTranscript

    ; ============ FASE 2: Classificação de intenção (XGBoost) ============
    TRACE_SPAN_START rSpanClassify, NAME="intent_classification"

    EMBED rQueryEmb, rTranscript, rBge                      ; ✅ 0x09
        MODEL=MODEL_BGE

    ; Concat features: embedding + metadata
    CONCAT rFeatures, rQueryEmb, rTurnMeta                  ; 🟡 0x2F

    FOREST rClassify, rFeatures, rXgb                       ; 🟡 0x22
        TREES=500
        DEPTH=8
        MODE=PROBABILITY

    ; Extrai (intent_id, confidence, complexity)
    SLICE rIntentId,  rClassify, 0, 1                       ; 🟡 0x2E
    SLICE rConfidence, rClassify, 1, 2
    SLICE rComplexity, rClassify, 2, 3

    TRACE_SPAN_END rSpanClassify

    ; --- Roteamento inteligente ---
    COMPARE rConfidence, THRESH_CONFIDENCE                  ; ✅ 0x0C
    IF_LESS rConfidence, low_confidence_path

    COMPARE rComplexity, THRESH_COMPLEXITY
    IF_GREATER rComplexity, deep_path
    JUMP fast_path


; ============================================================
; FAST PATH — Mamba-2.8B apenas (~50ms)
; ============================================================
fast_path:
    TRACE_SPAN_START rSpanFast, NAME="fast_path_mamba"

    ; --- RAG retrieval (top-5 docs) ---
    CALL rag_retrieve rQueryEmb, rDocs

    ; --- Prompt = query + docs ---
    CONCAT rPrompt, rTranscript, rDocs                      ; 🟡 0x2F
    EMBED rPromptEmb, rPrompt, rMamba.emb                   ; ✅ 0x09

    ; --- Mamba forward (rápido) ---
    CALL mamba_forward rPromptEmb, rMamba, rOutput          ; usa SSM_SCAN ✅
    SAMPLE rTextResp, rOutput TEMPERATURE=0.5 TOPK=40       ; ✅ 0x0B

    TRACE_SPAN_END rSpanFast
    JUMP synthesize_response


; ============================================================
; DEEP PATH — Speculative Decoding (Mamba draft + Llama verify)
; ============================================================
deep_path:
    TRACE_SPAN_START rSpanDeep, NAME="deep_path_speculative"

    ; --- RAG retrieval (top-10 docs para queries complexas) ---
    CALL rag_retrieve rQueryEmb, rDocs TOPK=10

    CONCAT rPrompt, rTranscript, rDocs                      ; 🟡 0x2F
    EMBED rPromptEmb, rPrompt, rLlama.emb                   ; ✅ 0x09

    ; --- Speculative decoding loop ---
    ;     Mamba (draft) gera K tokens, Llama (target) verifica
    ;     Ambos rodam em GPUs diferentes, comunicação via NVLink
    LOADI rTokenPos, 0
    LOADI rAccepted, 0
    LOADI rSpecLen, 5                                       ; draft 5 tokens por vez

speculative_loop:
    COMPARE rTokenPos, MAX_TOKENS
    IF_EQUAL rTokenPos, spec_done

    ; --- Draft: Mamba gera K tokens em GPU1 ---
    CTX_SWITCH TRANSFORMER, BLUE
    DEVICE_SELECT rStatus, DEVICE=rGpu1                     ; 🟡 0xB8
    MATVEC rDraftLogits, rPromptEmb, rMamba.head            ; ✅ 0x10
    SAMPLE rDraftTokens, rDraftLogits TOPK=rSpecLen         ; ✅ 0x0B

    ; --- Verify: Llama processa em GPU2 ---
    DEVICE_SELECT rStatus, DEVICE=rGpu2
    MATVEC rTargetLogits, rPromptEmb, rLlama.head           ; ✅ 0x10

    ; --- Acceptance check: comparar distribuições ---
    ;     Aceita tokens onde Mamba e Llama concordam
    CALL speculative_verify rDraftTokens, rTargetLogits, rAcceptedMask

    ; --- Anexa tokens aceitos ---
    CALL append_tokens rOutput, rDraftTokens, rAcceptedMask

    ; --- Atualiza estado SSM do Mamba (rollback se rejeitado) ---
    CALL mamba_update_state rOutput, rMamba                 ; usa SSM_SCAN ✅
    ; Se rejeitou, ABORT restaura h_t

    ; --- Atualiza KV cache do Llama ---
    CALL llama_update_kv rOutput, rLlama                    ; usa ATTN ✅

    ; --- Progresso ---
    ADD rTokenPos, rTokenPos, rAccepted
    JUMP speculative_loop

spec_done:
    TRACE_SPAN_END rSpanDeep
    JUMP synthesize_response


; ============================================================
; LOW CONFIDENCE PATH — escalonamento para humano
; ============================================================
low_confidence_path:
    TRACE_SPAN_START rSpanLow, NAME="low_confidence"

    ; --- Notifica operador humano via canal separado ---
    SIGNAL NODE=HUMAN, KIND=PING                            ; ✅ 0x1B
        URGENCY=HIGH
        CONTEXT=rTranscript

    ; --- Resposta fallback ---
    LOADSTR rTextResp, "Vou transferir você para um especialista."  ; ✅

    TRACE_SPAN_END rSpanLow
    JUMP synthesize_response


; ============================================================
; SYNTHESIZE RESPONSE — Moshi gera áudio da resposta
; ============================================================
synthesize_response:
    TRACE_SPAN_START rSpanSynth, NAME="synthesize_response"

    ; --- Tokeniza resposta ---
    EMBED rTextEmb, rTextResp, rMoshi.emb                   ; ✅ 0x09

    ; --- Moshi Depformer gera código de áudio ---
    CTX_SWITCH AUDIO, RED                                   ; ✅ 0x18
    DEVICE_SELECT rStatus, DEVICE=rGpu0                     ; 🟡 0xB8

    DEPFORMER rAudioCodes, rTextEmb, rAudioEmb, rKVDep      ; 🟡 0x44
        DEP_LAYERS=6
        DEP_DIM=1024
        CODEBOOKS=N_CODEBOOKS
        ACOUSTIC_DELAY=1
        TEMPERATURE=0.8
        TOP_K=50
        STREAM_ID=0

    ; --- Publica no ring buffer de saída ---
    LOCK rRingLock
    STREAM_WRITE rAudioOutRing, rAudioCodes                 ; 🟡
    UNLOCK rRingLock

    TRACE_SPAN_END rSpanSynth
    TRACE_SPAN_END rSpanTurn

    ; --- Log de turno completo ---
    TRACE_EVENT "turn_complete"                             ; ✅ 0x6B
        INTENT=rIntentId
        CONFIDENCE=rConfidence
        COMPLEXITY=rComplexity
        PATH=chosen_path
        TOKENS=rTokenPos
        ACCEPTED=rAccepted

    JUMP main_loop


; ============================================================
; SUB-ROTINA: RAG Retrieve
; ============================================================
; Entrada: rQueryEmb (embedding da query)
; Saída  : rDocs (top-K documentos concatenados)
; ============================================================
rag_retrieve:
    .param rQuery, rResult, TOPK=5

    RAG_SEARCH rResults, rQuery, rRag                       ; 🟡 0x52
        METRIC=COSINE
        TOPK=TOPK
        MODE=IVF
        N_PROBE=32

    ; Carrega os K documentos
    LOADI rI, 0
rag_load_loop:
    COMPARE rI, TOPK
    IF_EQUAL rI, rag_done

    ; rResults[i] contém doc_id
    EMBED_LOOKUP rDoc, rRag.docs, rResults[rI]              ; 🟡 0x53
    CONCAT rResult, rResult, rDoc                           ; 🟡 0x2F

    ADD rI, rI, 1
    JUMP rag_load_loop

rag_done:
    RET


; ============================================================
; SUB-ROTINA: Mamba Forward
; ============================================================
; Entrada: rEmb (embedding do prompt), rModel (handle Mamba)
; Saída  : rOutput (logits)
; Usa: SSM_SCAN ✅, NORM ✅, FFN ✅
; ============================================================
mamba_forward:
    .param rEmb, rModel, rOutput

    TENSOR rH 4096 16 f32                                   ; estado SSM
    CALL ssm_init_state rH, rModel

    LOADI rL, 0
mamba_layer_loop:
    COMPARE rL, rModel.n_layers
    IF_EQUAL rL, mamba_done

    ; RMSNorm
    NORM rX, rEmb, rModel.norm[rL]                          ; ✅ 0x07

    ; SSM scan (núcleo do Mamba)
    SSM_SCAN rY, rX, rH, rModel.pack[rL]                    ; ✅ 0x13
        D_INNER=4096
        D_STATE=16
        LAYER=rL

    ; Residual
    ADD rEmb, rEmb, rY                                      ; ✅ 0x0A

    ; FFN
    NORM rX, rEmb, rModel.normF[rL]
    FFN rFFN, rX, rModel.gate[rL], rModel.up[rL], rModel.down[rL]  ; ✅ 0x08
    ADD rEmb, rEmb, rFFN

    ADD rL, rL, 1
    JUMP mamba_layer_loop

mamba_done:
    ; LM head
    MATVEC rOutput, rEmb, rModel.lm_head                    ; ✅ 0x10
    RET


; ============================================================
; SUB-ROTINA: Speculative Verify
; ============================================================
; Compara distribuições draft vs target, aceita tokens concordantes
; ============================================================
speculative_verify:
    .param rDraftTokens, rTargetLogits, rAcceptMask

    SOFTMAX rDraftProbs, rDraftTokens AXIS=-1               ; 🟡 0x3C
    SOFTMAX rTargetProbs, rTargetLogits AXIS=-1

    ; Divergência KL por token
    CALL kl_divergence rDraftProbs, rTargetProbs, rKL       ; 🟡

    ; Aceita tokens com KL < threshold
    COMPARE rKL, 0.1
    IF_LESS rKL, accept_token
    JUMP reject_token

accept_token:
    MEMSET rAcceptMask, 1 LEN=1                             ; 🟡 0x2B
    RET

reject_token:
    MEMSET rAcceptMask, 0 LEN=1
    ; Resample do target
    SAMPLE rResampled, rTargetProbs TOPK=1                  ; ✅ 0x0B
    RET


; ============================================================
; SUB-ROTINA: Moshi ASR
; ============================================================
; Codes Mimi → texto (usa Temporal do Moshi)
; ============================================================
moshi_asr:
    .param rCodes, rTranscript

    ; Merge streams
    STREAM_MERGE rMixed, rCodes, rCodes                     ; 🟡 0x45
        N_STREAMS=N_STREAMS MODE=agent_only

    ; Temporal forward (32 layers)
    LOADI rL, 0
asr_layer_loop:
    COMPARE rL, 32
    IF_EQUAL rL, asr_done

    NORM rHidden, rMixed, rMoshi.normA[rL]
    MATVEC rQ, rHidden, rMoshi.wq[rL]
    MATVEC rK, rHidden, rMoshi.wk[rL]
    MATVEC rV, rHidden, rMoshi.wv[rL]
    ATTN rAttn, rQ, rK, rV, rMask, rKV                      ; ✅ 0x02
        NHEADS=32 HDIM=128 CAUSAL
        STREAM=0 LAYER=rL

    MATVEC rProj, rAttn, rMoshi.wo[rL]
    ADD rHidden, rHidden, rProj

    NORM rHidden, rHidden, rMoshi.normF[rL]
    FFN rFFN, rHidden, rMoshi.gate[rL], rMoshi.up[rL], rMoshi.down[rL]
    ADD rHidden, rHidden, rFFN

    ADD rL, rL, 1
    JUMP asr_layer_loop

asr_done:
    ; Text head
    MATVEC rTextLogits, rHidden, rMoshi.text_head
    SAMPLE rTranscript, rTextLogits TEMPERATURE=0.1 TOPK=1  ; ✅ 0x0B
    RET


; ============================================================
; SUB-ROTINA: Append Tokens
; ============================================================
append_tokens:
    .param rOutput, rTokens, rMask
    ; ... (concatena tokens aceitos no output) ...
    RET


; ============================================================
; SUB-ROTINA: Mamba Update State (pós-speculative)
; ============================================================
mamba_update_state:
    .param rOutput, rModel
    ; Re-scan apenas dos tokens aceitos para sincronizar estado SSM
    ; Se houve rejeição, ABORT restaura h_t
    RET


; ============================================================
; SUB-ROTINA: Llama Update KV
; ============================================================
llama_update_kv:
    .param rOutput, rModel
    ; Append K,V dos tokens aceitos no KV cache do Llama
    ; Usa KV_TRUNCATE se houve rejeição
    KV_TRUNCATE rKV, rLen                                   ; 🟡 0x38
    RET


; ============================================================
; SUB-ROTINA: KL Divergence
; ============================================================
kl_divergence:
    .param rP, rQ, rResult
    ; D_KL(P || Q) = sum(P * log(P/Q))
    DIV rRatio, rP, rQ                                      ; 🟡
    LOG rLogRatio, rRatio
    MUL rElem, rP, rLogRatio
    REDUCE rResult, rElem OP=SUM                            ; 🟡 0x33
    RET
```

---

## Análise do Programa

### O que este programa demonstra

| Capacidade | Como aparece |
|:---|:---|
| **6 modelos no mesmo contexto** | `LOAD_MODEL` × 5 + `RAG_INDEX_ADD` × 1 |
| **Placement multi-GPU** | `PLACE` distribui modelos por 4 GPUs |
| **Roteamento inteligente** | `FOREST` (XGBoost) decide fast/deep path |
| **Speculative decoding** | `MATVEC` em paralelo (Mamba + Llama) |
| **RAG nativo** | `RAG_SEARCH` + `EMBED_LOOKUP` |
| **Voice I/O** | `DEPFORMER` + `CODEC_ENC/DEC` |
| **Deadline composto** | `DEADLINE_AND` + `DEADLINE_CHAIN` |
| **Rollback híbrido** | `ABORT` restaura KV (Llama) + `ssm_states` (Mamba) |
| **Cross-device** | `REDUCE_LOCAL` (não usado aqui, mas disponível) |
| **Tracing** | `TRACE_SPAN_*` correlaciona 6 fases |

### Fluxo de Execução

```
Turno do usuário:
  ├─ 80ms  captura áudio (Moshi input loop)
  ├─ 20ms  classificação (XGBoost)
  ├─ RAG   retrieval (BGE + IVF)
  ├─ FAST PATH (Mamba)       ~50ms   ──┐
  │     OU                             │
  │   DEEP PATH (speculative) ~300ms  ─┤
  ├─ 80ms  síntese de voz (Moshi output loop)
  └─ 500ms deadline total (worst case)
```

### Sinergia entre Modelos

| Modelo | Papel | Por que não usar outro |
|:---|:---|:---|
| **Moshi** | Voice I/O full-duplex | Único modelo open com barge-in nativo |
| **BGE-M3** | Embeddings | Multilingual + multi-granularity |
| **XGBoost** | Classificador de intenção | 1000× mais rápido que LLM, determinístico |
| **Mamba** | Draft model | O(1) state, ideal para speculative |
| **Llama-70B** | Target model | Máxima qualidade |
| **RAG** | Conhecimento | Reduz alucinação, cita fontes |

### Fluxo de Execução

```
Turno do usuário:
  ├─ 80ms  captura áudio (Moshi input loop)
  ├─ 20ms  classificação (XGBoost)
  ├─ RAG   retrieval (BGE + IVF)
  ├─ FAST PATH (Mamba)       ~50ms   ──┐
  │     OU                             │
  │   DEEP PATH (speculative) ~300ms  ─┤
  ├─ 80ms  síntese de voz (Moshi output loop)
  └─ 500ms deadline total (worst case)
```

### Sinergia entre Modelos

| Modelo | Papel | Por que não usar outro |
|:---|:---|:---|
| **Moshi** | Voice I/O full-duplex | Único modelo open com barge-in nativo |
| **BGE-M3** | Embeddings | Multilingual + multi-granularity |
| **XGBoost** | Classificador de intenção | 1000× mais rápido que LLM, determinístico |
| **Mamba** | Draft model | O(1) state, ideal para speculative |
| **Llama-70B** | Target model | Máxima qualidade |
| **RAG** | Conhecimento | Reduz alucinação, cita fontes |

### Comparação com Abordagem Tradicional

| Aspecto | Abordagem tradicional (Python + gRPC) | M³-AVM |
|:---|:---|:---|
| **Comunicação entre modelos** | Serialização JSON → HTTP → parse | Memória compartilhada (`SHARED`) |
| **Roteamento** | Lógica Python, latência ~5ms | `FOREST` nativo, latência ~200µs |
| **Speculative decoding** | Threads Python, GIL, cópia GPU↔CPU | GPUs diferentes, NVLink, sem cópia |
| **RAG** | FAISS + Python → serialização | `RAG_SEARCH` nativo |
| **Voice I/O** | gRPC + streaming, ~200ms overhead | `STREAM` nativo |
| **Rollback** | Não existe | `ABORT` em ~217µs |
| **Deadline** | Manual (try/except timeout) | `DEADLINE_CHAIN` nativo |
| **Latência total** | ~800ms | ~500ms |
| **Memória** | 3 processos separados | 1 processo unificado |

### Notas Honestas

1. **Nada disso compila hoje.** Os opcodes ✅ rodam no repo atual, mas os 🟡 (especialmente `LOAD_MODEL`, `RAG_SEARCH`, `DEPFORMER`, `FOREST`, `TRACE_*`, `PARALLEL_*`) são spec.

2. **A complexidade é real.** Um programa deste tipo exige ~2000 linhas de runtime Rust + ~3000 linhas de teste para funcionar de verdade.

3. **O ganho é estrutural.** O programa acima não é "melhor" porque tem mais features — é melhor porque **não precisa de 5 processos Python conversando via JSON** para fazer o que cabe em um único contexto.

4. **Latência real será medida.** Os números (~50ms, ~300ms) são estimativas de engenharia baseadas no paper do Moshi + benchmarks publicados de Mamba e Llama. O runtime real pode estar 20–50% acima até o tuning.

5. **A ordem de implementação importa.** Para este programa rodar, você precisa de:
   - Marco 1 (Cluster + GATHER) — 4–6 sem
   - Marco 2 (Moshi) — 6–8 sem
   - Marco 3 (Onda 1 universal) — 3–4 sem
   - Marco 4 (Onda 2 universal) — 3–4 sem
   - Marco 5 (Otimizações) — 3–4 sem
   - Marco 6 (LCT) — 4 sem
   - RFCs 0003–0012 — 20–25 sem
   
   **Total: ~45–55 semanas.** Este não é um programa que roda amanhã.

---

## O Que Este Exemplo Prova

Se você implementar tudo, você terá um sistema onde:

- **Uma query de voz entra, uma resposta de voz sai** — com raciocínio de Llama-70B, velocidade de Mamba, precisão de RAG e classificador XGBoost.
- **Tudo isso em um único processo** — sem gRPC, sem JSON, sem cópia de memória entre modelos.
- **Com deadline de 500ms** — garantido pelo scheduler EDF, não por sorte.
- **Com rollback de 217µs** — se o usuário interromper, o estado híbrido (KV + SSM) é restaurado atomicamente.
- **Com tracing distribuído** — cada fase é observável.
- **Com compressão automática** — o KV do Llama é comprimido antes de migrar entre nós.

**Isso não existe hoje em nenhum framework.** PyTorch + vLLM + FAISS + Whisper + XGBoost rodando juntos têm 5 processos, 5 linguagens de serialização e 5 deadlines independentes. A M³-AVM unifica tudo isso.

É exatamente para isso que a VM foi projetada.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
