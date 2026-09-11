# Avatar 3D Vetorizado — Abordagem A (Plano-Alvo)

```text
Status:   META-ALVO (design de produto, NÃO-normativo, NÃO implementado)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Companion: docs/MVP_ENGLISH_TUTOR.md (o rig) + docs/EXPERIENCIA_PREMIUM.md (a sensação)
License:  AGPL-3.0-or-later (see LICENSE)
```

> **Contrato de honestidade (ESPEC §1.3):** hardware, preços, latências
> e specs de terceiros abaixo são **pesquisa e metas do autor**. A §0
> lista as correções — a principal: **nada da §3 existe no ISA**,
> e os números propostos colidem com opcodes implementados.

## §0. Reconciliação (leitura obrigatória)

### §0.1 A tabela de opcodes da §3 é inválida como escrita

Cada opcode proposto colide com um opcode IMPL:

| Proposto | Colide com (IMPL) | Desde |
|:---|:---|:---|
| `0x22 AUDIO2FACE` | `FOREST` | RFC-0012 |
| `0x23 RENDER_FACE` | `DISTANCE` | RFC-0004 |
| `0x24 LIP_SYNC` | `RANK1_UPDATE` | RFC-0004 |
| `0x25 AVATAR_LOAD` | `ODE_STEP` | RFC-0014 |
| `0x26 AVATAR_POSE` | `ARENA_ALLOC` | RFC-0023 |
| `0x27 AVATAR_BLEND` | `ARENA_RESET` | RFC-0023 |
| `0x28 AVATAR_LOOK` | `SNAPSHOT` | RFC-0024 |
| `0x29 AVATAR_BLINK` | `RESTORE` | RFC-0024 |

Consequências: (a) "RFC-0013" do título da §3 também colide
(`RFC-0013` é `DENOISE_STEP`); a forma de proposta canônica hoje é o
template R11 (ESPEC-V2 R11, não "formato RFC-0001" — arquivada);
(b) os payloads em `payload_core` descrevem o layout 64B do ESPEC-V2
§4.2, mas sem opcode válido não há instrução; (c) as "Regiões Novas"
`0x18–0x1C` não existem — o modelo lógico tem 16 regiões `0x0–0xF`
(ESPEC-V2 §7); o equivalente honesto são tensores em GLOBAL +
buffers host-side; (d) "Feature Bit 48" exige a RFC de capabilities
(mecanismo RSVD, trilha W10/ESCAPE).

### §0.2 A descoberta que salva o plano

**O MVP do avatar precisa de ZERO opcodes novos.** Audio2Face e o
renderer são componentes host-side (sidecars): a VM entrega PCM pelos
ops existentes (`CODEC_DEC`, `STREAM`), o sidecar gera blendshapes e
renderiza. Uma família `AVATAR_*` no ISA só faria sentido para
orquestração in-VM futura — e aí exigiria RFC de alocação própria
(as faixas livres estão contestadas entre ESPEC §5.8 e ESPEC-V2
§3.7; a RFC resolveria primeiro). O Anexo A detalha.

### §0.3 Regras deste documento

- Programa da §4: listagem-alvo (não monta; mesmos motivos dos
  anteriores + ops inexistentes). Não entra em `programs/`.
- "Abordagem A" implica B inexistente: nenhuma Abordagem B foi anexada.
- "Como discutimos antes" (Vulkan): sem registro neste repo — tratar
  como contexto externo do autor.

---

> **Objetivo:** Adicionar um rosto 3D que fala em sincronia com o áudio do Moshi, usando apenas o hardware atual (GTX 1650 + 2× RTX 3060).
>
> **Custo adicional:** ~R$ 200 (mesh + SDK) [pesquisa do autor]
> **Latência adicional:** ~16ms [META]
> **VRAM adicional:** ~0.5 GB [estimativa]
> **Sem GPU nova.**

---

## 1. Arquitetura da Abordagem A (alvo)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE AVATAR 3D (ALVO)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Moshi (GPU 2)                                                              │
│  ┌──────────────────────────────────────────┐                              │
│  │  DEPFORMER → codes Mimi (16 codebooks)   │  ✅ op existe (RFC-0032)     │
│  └────────────────┬─────────────────────────┘                              │
│                   │                                                         │
│                   │ CODEC_DEC (Mimi)                       ✅ op existe     │
│                   ▼                                                         │
│  ┌──────────────────────────────────────────┐                              │
│  │  PCM 24kHz @ 80ms frames                 │                              │
│  └────────────────┬─────────────────────────┘                              │
│                   │                                                         │
│                   │ (paralelo: áudio vai pro speaker + avatar)              │
│                   │                                                         │
│         ┌─────────┴─────────┐                                              │
│         │                   │                                              │
│         ▼                   ▼                                              │
│  ┌─────────────┐    ┌─────────────────────┐                               │
│  │ 🔊 Speaker  │    │ AUDIO2FACE          │  🔴 host-side (não é opcode)  │
│  │             │    │ (GPU 0 ou CPU)      │                               │
│  └─────────────┘    │                     │                               │
│                     │ Áudio → 52 blendshapes ARKit                            │
│                     │ Latência: ~10ms [META/vendedor]  │                               │
│                     │ VRAM: ~1 GB [vendor]            │                               │
│                     └──────────┬──────────┘                               │
│                                │                                          │
│                                │ blendshapes (52 floats)                  │
│                                ▼                                          │
│                     ┌─────────────────────┐                               │
│                     │ AVATAR_RENDER       │  🔴 host-side (não é opcode)  │
│                     │ (GPU 1 ou 2)        │                               │
│                     │                     │                               │
│                     │ Mesh 3D + blendshapes → frame                            │
│                     │ Latência: ~5ms [META]     │                               │
│                     │ VRAM: ~0.5 GB [estimativa]│                               │
│                     └──────────┬──────────┘                               │
│                                │                                          │
│                                │ frame (1920×1080 @ 30fps)                │
│                                ▼                                          │
│                     ┌─────────────────────┐                               │
│                     │ STREAM (VIDEO)      │  🔴 modo (opcode STREAM ✅)   │
│                     │ (WebSocket → client)│                               │
│                     └─────────────────────┘                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pipeline total (alvo):**
1. Moshi gera áudio (ops existem; pesos+programa pendentes)
2. Áudio vai para o speaker **e** para o Audio2Face
3. Audio2Face gera 52 blendshapes ARKit (10ms) [vendor]
4. Render aplica blendshapes no mesh 3D (5ms) [META]
5. Frame vai para o cliente via WebSocket

**Latência visual total (META):** ~15ms (imperceptível).

---

## 2. Componentes Técnicos (pesquisa do autor; specs de terceiros não verificadas aqui)

### 2.1 Audio2Face (NVIDIA)

**O que é:** Rede neural que mapeia áudio → blendshapes faciais (52 ARKit).

| Spec | Valor |
|:---|:--:|
| **Modelo** | NVIDIA Audio2Face-3D |
| **Input** | Áudio PCM 16kHz mono |
| **Output** | 52 blendshapes ARKit (0–1) |
| **Latência** | ~10ms/frame |
| **VRAM** | ~1 GB |
| **Licença** | NVIDIA AI Foundation (gratuita para uso pessoal) |
| **Plataforma** | Windows, Linux (CUDA) |

**Como funciona internamente (descrição do vendor):**
```
Áudio (16kHz, 40ms windows)
    ↓
[Conv1D + LSTM encoder]
    ↓
[Cross-attention com embeddings de fonemas]
    ↓
[MLP decoder]
    ↓
52 blendshapes ARKit
    (browDownLeft, browDownRight, browInnerUp, ...
     jawOpen, jawLeft, jawRight, jawForward,
     mouthFunnel, mouthPucker, mouthLeft, mouthRight,
     mouthSmileLeft, mouthSmileRight, ...)
```

**Alternativas open source (tabela do autor):**
| Modelo | Qualidade | Latência | VRAM |
|:---|:--:|:--:|:--:|
| **NVIDIA Audio2Face** | ⭐⭐⭐⭐⭐ | 10ms | 1 GB |
| **OpenFace 2.0** | ⭐⭐⭐ | 15ms | 0.5 GB |
| **Rhubarb Lip Sync** | ⭐⭐ | 5ms | 0 GB (CPU) |
| **OVR LipSync** | ⭐⭐⭐ | 8ms | 0.5 GB |

**Recomendação (do autor):** NVIDIA Audio2Face (melhor qualidade, gratuito para uso pessoal).

### 2.2 Mesh 3D (pesquisa do autor)

| Fonte | Qualidade | Custo | Personalização | Formato |
|:---|:--:|:--:|:--:|:---|
| **Ready Player Me** | ⭐⭐⭐⭐ | Grátis | Alta | glTF |
| **MetaHuman Creator** | ⭐⭐⭐⭐⭐ | Grátis | Alta | USD/FBX |
| **VRoid Studio** | ⭐⭐⭐ | Grátis | Anime | VRM |
| **Mixamo** | ⭐⭐⭐ | Grátis | Média | FBX |
| **Reallusion Character Creator** | ⭐⭐⭐⭐⭐ | $300 | Alta | FBX |

**Recomendação (do autor):** **Ready Player Me** (2 min de selfie, glTF com blendshapes ARKit, licença comercial gratuita).
**Alternativa:** **MetaHuman Creator** (fotorrealista, pesado, exige Unreal p/ exportar).

### 2.3 Renderer (pesquisa do autor)

| Renderer | Latência | VRAM | Complexidade | Plataforma |
|:---|:--:|:--:|:--:|:---|
| **wgpu** (Rust) | ~5ms | ~0.5 GB | Média | Cross-platform |
| **OpenGL ES 3.0** | ~5ms | ~0.3 GB | Baixa | Cross-platform |
| **Vulkan** | ~4ms | ~0.4 GB | Alta | Cross-platform |
| **Bevy** (Rust) | ~6ms | ~1 GB | Baixa | Cross-platform |

**Recomendação (do autor):** **wgpu** (Rust nativo, cross-platform, glTF+PBR).

> Nota de engenharia: render gráfico via wgpu é workload distinto do
> path de computação da M³-AVM (hoje CPU-bound com `ATTN≤64` em wgpu).
> Sem conflito arquitetural; contenção de dispositivo mede-se no rig
> (ver §6, Opção A).

---

## 3. Opcodes Novos — INVÁLIDOS COMO ESCRITOS (ver §0.1)

> Mantidos abaixo para registro do intento original, com a correção ao
> lado. Nenhum deles existe; os hexas pertencem a outros opcodes.

### 3.1 Tabela (proposta original → correção)

| Opcode | Hex | Nome | Função | Correção |
|:---|:--:|:---|:---|:---|
| `0x22` | `AUDIO2FACE` | Áudio → blendshapes | **COLIDE com `FOREST`** (RFC-0012) |
| `0x23` | `RENDER_FACE` | Blendshapes → frame | **COLIDE com `DISTANCE`** (RFC-0004) |
| `0x24` | `LIP_SYNC` | Áudio → visemes | **COLIDE com `RANK1_UPDATE`** (RFC-0004) |
| `0x25` | `AVATAR_LOAD` | Carrega mesh | **COLIDE com `ODE_STEP`** (RFC-0014) |
| `0x26` | `AVATAR_POSE` | Define pose | **COLIDE com `ARENA_ALLOC`** (RFC-0023) |
| `0x27` | `AVATAR_BLEND` | Combina expressões | **COLIDE com `ARENA_RESET`** (RFC-0023) |
| `0x28` | `AVATAR_LOOK` | Direção do olhar | **COLIDE com `SNAPSHOT`** (RFC-0024) |
| `0x29` | `AVATAR_BLINK` | Piscar | **COLIDE com `RESTORE`** (RFC-0024) |

Caminho honesto (Anexo A): v1 sem opcode novo (sidecars host-side);
família futura só via RFC de alocação em faixa livre.

### 3.2 Payloads (rascunho original — sem efeito normativo)

Mantidos como ponto de partida para a futura RFC de alocação; campos
como `MODEL=` (qual modelo externo) e `FILE=` não existem no ISA
(caminhos são host-side; ver §0.2).

### 3.3 Regiões Novas (inválidas — ver §0.1)

| ID (proposto) | Nome | Uso |
|:--:|:---|:---|
| `0x18` | `AVATAR_MESH` | Mesh, texturas, materiais |
| `0x19` | `BLENDSHAPES` | 52 floats (ativa) |
| `0x1A` | `BLENDSHAPES_PREV` | Frame anterior (smoothing) |
| `0x1B` | `RENDER_BUFFER` | Frame RGBA 1080p |
| `0x1C` | `AVATAR_ANIM` | Animações idle, blink, look |

Equivalente honesto hoje: tensores em GLOBAL (52 floats = tensor
`[1,52]` trivial) + buffers do processo host. Nenhuma região nova é
necessária para o MVP.

### 3.4 Feature Bit (proposta — exige RFC de capabilities)

| Bit | Nome | Descrição |
|:--:|:---|:---|
| 48 | `VISUAL_AVATAR` | Suporte a avatar 3D com audio2face |

---

## 4. Programa `.m3asm` Completo (listagem-alvo, NÃO monta — ver §0.2)

`programs/english_tutor_avatar.m3asm` (destino futuro)

```asm
; ============================================================
; M³-AVM English Tutor with 3D Avatar (ALVO)
; Hardware : GTX 1650 + 2× RTX 3060
; Avatar   : Ready Player Me glTF + NVIDIA Audio2Face (host-side)
; Latência : ~15ms adicional (visual) [META]
; ============================================================

.data                                                        ; 🔴 assembler
    PCM_FRAME_LEN    .equ 1920                               ; 🔴
    N_CODEBOOKS      .equ 16                                 ; 🔴
    N_STREAMS        .equ 17                                 ; 🔴
    RAG_TOPK         .equ 5                                  ; 🔴
    AVATAR_FPS       .equ 30                                 ; 🔴
    AVATAR_FRAME_NS  .equ 33333333                           ; 🔴
    RENDER_WIDTH     .equ 1920                               ; 🔴
    RENDER_HEIGHT    .equ 1080                               ; 🔴
    BS_JAW_OPEN      .equ 0                                  ; 🔴 (+6 índices)
    ; ... (52 total)

.text                                                        ; 🔴
main:
    ; ============ STEP 1: Carregar modelos ============
    LOAD_MODEL rPersona, "personaplex-7b-Q4_K.gguf"          ; 🟡 0xA0
    LOAD_MODEL rLlama,   "llama-3.1-8b-Q4_K.gguf"            ; 🟡
    MODEL_LOAD_UNIFIED rBge, "bge-m3.safetensors"            ; 🔴 0xFC reservada!

    ; ============ STEP 2: Carregar avatar ============
    AVATAR_LOAD rAvatar, FILE="avatars/teacher.glb"         ; 🔴 opcode inexistente
    AVATAR_POSE rAvatar, POSE=idle                          ; 🔴 opcode inexistente

    ; ============ STEP 3: Placement ============
    PLACE rPersona, DEVICE=0    ; GTX 1650                   ; 🔴 reservada!
    PLACE rLlama,   DEVICE=1    ; RTX 3060 #1                ; 🔴
    PLACE rAvatar,  DEVICE=2    ; RTX 3060 #2 (junto com Moshi) ; 🔴

    ; ============ STEP 4: RAG ============
    RAG_INDEX_ADD rRag, FILE="grammar_b2.jsonl"              ; 🟡 0x50
    RAG_INDEX_ADD rRag, FILE="vocabulary_b2.jsonl"           ; 🟡

    ; ============ STEP 5: Contextos ============
    SPAWN_CONTEXT rAudioIn,  ENTRY=audio_input,  PRIORITY=RED    ; 🟡 0xA2
    SPAWN_CONTEXT rAudioOut, ENTRY=audio_output, PRIORITY=RED    ; 🟡
    SPAWN_CONTEXT rReason,   ENTRY=reason_loop,  PRIORITY=BLUE   ; 🟡
    SPAWN_CONTEXT rAvatarCtx, ENTRY=avatar_loop,  PRIORITY=BLUE  ; 🟡

    JUMP idle                                                ; ✅ 0x0E

; ============================================================
; ÁUDIO INPUT — captura + VAD + barge-in
; ============================================================
audio_input:
    SENSE rPCM, AUDIO_PCM, LEN=PCM_FRAME_LEN                ; 🔴 modo (opcode ✅)
    VAD_DETECT rVAD, rPCM THRESHOLD=0.3                     ; 🔴 modo (opcode ✅)
    COMPARE rVAD, 0.3                                       ; 🔴 float imediato
    IF_GREATER rVAD, barge_in                               ; 🔴 (só IF_EQUAL/IF_INTERRUPT)

    CODEC_ENC rCodes, rPCM                                  ; ✅ 0x15
    STREAM_WRITE rAudioIn, rCodes                            ; 🔴 modo
    SIGNAL CTX=rReason, KIND=PING                           ; 🔴 forma (opcode ✅)
    JUMP audio_input                                        ; ✅

barge_in:
    SIGNAL CTX=rReason,    KIND=ABORT                       ; 🔴 forma (opcode ✅)
    SIGNAL CTX=rAvatarCtx, KIND=ABORT                       ; 🔴 forma
    ABORT rSnapAudio                                        ; ✅ (forma: ABORT Rs_ctx, Rs_ts)
    JUMP audio_input                                        ; ✅

; ============================================================
; ÁUDIO OUTPUT — sintetiza + reproduz + envia pro avatar
; ============================================================
audio_output:
    STREAM_READ rCodesOut, rAudioOutRing BLOCKING           ; 🔴 modo
    STREAM_MERGE rMixed, rAgentOut, rUserIn N_STREAMS=N_STREAMS  ; 🔴 modo+kwsimbólico (opcode ✅)
    CODEC_DEC rPCMOut, rMixed                               ; ✅ 0x16 (forma: CODEC_DEC rD, rS [TENSOR])

    ; Reproduz no speaker
    STREAM rPCMOut, CHANNEL=4 BLOCKING                      ; 🔴 modo (opcode ✅)

    ; Publica para o avatar
    LOCK rRingLock                                          ; ✅ 0x75 (try-lock; blocking é Fase 8)
    STREAM_WRITE rPCMForAvatar, rPCMOut                     ; 🔴 modo
    UNLOCK rRingLock                                        ; ✅ 0x76

    JUMP audio_output                                     ; ✅

; ============================================================
; REASONING — pipeline de conversação
; ============================================================
reason_loop:
    ; --- ASR ---
    LOCK rRingLock                                        ; ✅
    STREAM_READ rCodes, rAudioIn BLOCKING                   ; 🔴 modo
    UNLOCK rRingLock                                        ; ✅

    STREAM_MERGE rUserMixed, rCodes, rCodes MODE=user_only  ; 🔴 modo (opcode ✅)
    CALL temporal_forward rUserMixed, rHidden               ; 🔴 CALL com args (só label)
    MATVEC rTextLogits, rHidden, rPersona.text_head         ; ✅ (field access 🔴)
    SAMPLE rTranscript, rTextLogits TEMPERATURE=0.1 TOPK=1  ; ✅ (kwargs existem)

    ; --- RAG ---
    EMBED rQueryEmb, rTranscript, rBge                      ; 🔴 modo DIM= (opcode ✅)
    RAG_SEARCH rDocs, rQueryEmb, rRag TOPK=RAG_TOPK          ; 🟡 (+símbolo .data 🔴)
    CALL load_docs rDocs, rContext                          ; 🔴 CALL com args

    ; --- Raciocínio ---
    CONCAT rPrompt, rTranscript, rContext                   ; 🟡 0x2F
    CALL llama_forward rPrompt, rLlama, rOut                ; 🔴 CALL com args
    SAMPLE rResponse, rOut TEMPERATURE=0.7 TOPK=50          ; ✅

    ; --- Envia para síntese ---
    LOCK rRingLock                                        ; ✅
    STREAM_WRITE rAudioOutRing, rResponse                   ; 🔴 modo
    UNLOCK rRingLock                                        ; ✅

    ; --- Atualiza expressão do avatar (emoção) ---
    CALL classify_emotion rResponse, rEmotion               ; 🔴 CALL com args
    SIGNAL CTX=rAvatarCtx, KIND=EMOTION, VALUE=rEmotion     ; 🔴 kind+forma (opcode ✅)

    JUMP reason_loop                                      ; ✅

; ============================================================
; AVATAR — loop de animação visual (30 fps) [META — host-side]
; ============================================================
avatar_loop:
    ; ============ STEP 1: Lê áudio para lip-sync ============
    LOCK rAvatarLock                                      ; ✅ (lock genérico; anel é convenção)
    STREAM_READ rPCMForAvatar, rPCMFrame NONBLOCKING       ; 🔴 modo
    UNLOCK rAvatarLock                                    ; ✅

    ; ============ STEP 2: Audio2Face ============
    ; Converte áudio em 52 blendshapes ARKit
    AUDIO2FACE rBlendshapes, rPCMFrame                  ; 🔴 opcode inexistente (host-side!)
        SAMPLE_RATE=24000
        FRAME_SIZE=1920
        MODEL=NVIDIA
        SMOOTH=1
        ALPHA=0.3

    ; ============ STEP 3: Combina com expressão emocional ============
    AVATAR_BLEND rFinalBlendshapes, rBlendshapes, rEmotionBlendshapes  ; 🔴 inexistente
        MODE=additive
        CLAMP=1

    ; ============ STEP 4: Piscar natural ============
    CLOCK_QUERY rNow                                     ; 🔴 (é CYCLES_COUNT ✅ 0x6A)
    SUB rTimeSinceBlink, rNow, rLastBlink                ; 🔴 SUB não existe (ADD_IMM/SUB_IMM ✅ p/ regs!)
    COMPARE rTimeSinceBlink, BLINK_INTERVAL              ; 🔴 símbolo+float (inteiros escalados)
    IF_GREATER rTimeSinceBlink, trigger_blink            ; 🔴
    JUMP continue_avatar                                 ; ✅

trigger_blink:
    AVATAR_BLINK rFinalBlendshapes, DURATION=100ms        ; 🔴 inexistente
    COPY rLastBlink, rNow                                ; 🔴 COPY (usar MOV ✅ 0x79)

continue_avatar:
    ; ============ STEP 5: Direção do olhar ============
    COMPARE rVAD, 0.3                                    ; 🔴 float (rVAD indefinido neste escopo, aliás)
    IF_GREATER rVAD, look_at_user                        ; 🔴
    JUMP look_idle                                     ; ✅

look_at_user:
    AVATAR_LOOK rFinalBlendshapes, TARGET=user            ; 🔴 inexistente
    JUMP render                                         ; ✅

look_idle:
    AVATAR_LOOK rFinalBlendshapes, TARGET=horizon         ; 🔴 inexistente
    JUMP render                                         ; ✅

render:
    ; ============ STEP 6: Renderizar frame ============
    RENDER_FACE rFrame, rAvatar, rFinalBlendshapes       ; 🔴 inexistente
        WIDTH=RENDER_WIDTH
        HEIGHT=RENDER_HEIGHT
        FORMAT=RGBA8
        BACKGROUND=scene
        CAMERA_DIST=1.2
        CAMERA_ANGLE=0.0

    ; ============ STEP 7: Enviar para o cliente ============
    STREAM rFrame, CHANNEL=VIDEO BLOCKING                ; 🔴 modo (opcode ✅)

    ; ============ STEP 8: Controle de FPS ============
    SET_DEADLINE rDlAvatar, AVATAR_FRAME_NS              ; ✅ 0x71 (forma: SET_DEADLINE Rs)
    YIELD                                                   ; ✅ 0x70
    JUMP avatar_loop                                    ; ✅
```

> Notas de lógica: `SUB rTimeSinceBlink…` tem substituto real
> (`SUB_IMM` ✅ em regs — o programa o ignora); `rVAD` é lido num
> contexto onde nunca foi escrito (bug de escopo do rascunho);
> `KIND=EMOTION`/`FILLER_*` seguem o padrão `PING`+palavra de controle
> (`FILLER_SPEECH.md` §5.1). `SET_DEADLINE`/`YIELD` para pacing de
> 30fps é o uso correto do bloco scheduler.

---

## 5. Timeline de um Turno com Avatar (metas)

```
T=0ms      🎤 Usuário fala "Hey, how are you?"
           │
           ▼
T=80ms     SENSE AUDIO_PCM → VAD → CODEC_ENC
           │
           ▼
T=90ms     SIGNAL para reasoning + avatar
           │
           ▼
T=100ms    ASR (Temporal) → "Hey, how are you?"
           │
           ▼
T=150ms    RAG retrieval (contexto)
           │
           ▼
T=200ms    Llama gera resposta: "I'm doing great! How about you?"
           │
           ▼
T=250ms    Depformer gera áudio
           │
           ▼
T=255ms    🔀 Split:
           │
           ├──► 🔊 Speaker: áudio do Moshi
           │
           └──► 🎭 Avatar: Audio2Face
                    │
                    ▼
T=265ms    Blendshapes geradas (52 floats)
           │
           ▼
T=270ms    Render frame (1920×1080)
           │
           ▼
T=275ms    Frame enviado para o cliente
           │
           ▼
T=280ms    🖥️ Usuário vê rosto 3D falando
           │
           ▼
T=280ms    🔊 Usuário ouve áudio sincronizado

LATÊNCIA VISUAL ADICIONAL: ~25ms [META; §1 dizia ~15ms — divergência
do próprio rascunho, registrada]
SINCRONIA ÁUDIO-VÍDEO: perfeita (±5ms) [META]
```

---

## 6. Hardware — Onde Roda Cada Componente (tabela do autor + conta conferida)

| Componente | GPU | VRAM | Latência |
|:---|:--:|:--:|:--:|
| **Moshi (PersonaPlex)** | GPU 2 (RTX 3060) | 8 GB | ~40ms/frame |
| **Audio2Face** | GPU 0 (GTX 1650) | 1 GB | ~10ms |
| **Avatar Renderer (wgpu)** | GPU 0 (GTX 1650) | 0.5 GB | ~5ms |
| **Llama-8B** | GPU 1 (RTX 3060) | 6.5 GB | ~80ms/token |
| **Mamba + BGE** | GPU 0 (GTX 1650) | 3.5 GB | ~50ms |

**VRAM total usada:**
- GPU 0: 3.5 (Mamba+BGE) + 1 (Audio2Face) + 0.5 (render) = **5 GB** → **cabe em 4 GB? NÃO.** [conta do próprio rascunho — correta, e é o achado honesto do documento]
- GPU 1: 6.5 GB (Llama) → cabe em 12 GB ✅
- GPU 2: 8 GB (Moshi) → cabe em 12 GB ✅

**Soluções (do autor):**

| Opção | Como | Custo |
|:---|:---|:--:|
| **A** | Mover Audio2Face + Render para GPU 1 (RTX 3060) | R$ 0 |
| **B** | Mover Audio2Face para CPU (mais lento) | R$ 0 (latência +20ms) |
| **C** | Adicionar 4ª GPU (GTX 1050 Ti) | R$ 500 |

**Recomendação (do autor):** **Opção A** (GPU 1: 6.5 + 1 + 0.5 = 8 GB ✅).

---

## 7. Roadmap de Implementação (alvo do autor; fases do repo entre colchetes)

### Fase 1 — Fundação (Semana 1–2)

| Semana | Tarefa |
|:--:|:---|
| 1 | Instalar `wgpu` + `nvidia-audio2face` no projeto [nota: wgpu já é dependência opcional do repo; audio2face é SDK externo] |
| 2 | Carregar mesh glTF + render básico (cubo → mesh) |

**Entregável:** mesh na tela, sem animação.

### Fase 2 — Áudio → Blendshapes (Semana 3–4)

| Semana | Tarefa |
|:--:|:---|
| 3 | Integrar Audio2Face ao runtime |
| 4 | Testar com áudio pré-gravado |

**Entregável:** mesh falando com áudio gravado.

### Fase 3 — Integração com Moshi (Semana 5–6)

| Semana | Tarefa |
|:--:|:---|
| 5 | Conectar áudio do Moshi ao Audio2Face |
| 6 | Sincronizar frame rate (30 fps) com áudio (80ms) |

**Entregável:** mesh falando em tempo real com Moshi. [Pré-requisito real: MVP de voz, ~Mês 4–6 do `MVP_ENGLISH_TUTOR.md`.]

### Fase 4 — Refinamento (Semana 7–8)

| Semana | Tarefa |
|:--:|:---|
| 7 | Adicionar piscar, olhar, expressões |
| 8 | Otimização + demo de 5 min |

**Entregável:** avatar completo em conversa real.

---

## 8. Orçamento (pesquisa do autor)

| Item | Custo |
|:---|:--:|
| **Ready Player Me** (mesh) | R$ 0 |
| **NVIDIA Audio2Face** (SDK) | R$ 0 (uso pessoal) |
| **wgpu** (biblioteca Rust) | R$ 0 |
| **Blender** (edição de mesh) | R$ 0 |
| **Opcional: MetaHuman** | R$ 0 |
| **Opcional: mesh premium** | ~R$ 200 |
| **Total** | **R$ 0–200** |

**Zero hardware adicional** (com a realocação da Opção A).

---

## 9. O Que Você Vai Ver (narrativa-alvo, família `EXPERIENCIA_*`)

### Turno 1 — Saudação

```
Você:  "Hey, how are you?"

[Avatar 3D aparece na tela]
[Rosto neutro, olhando para você]
[Boca começa a se mover em sincronia com o áudio]

Avatar:  "I'm doing great! How about you?"
         [Sobrancelhas sobem levemente]
         [Sorriso sutil no canto da boca]
         [Pisca naturalmente]
```

**Sensação (alvo):** como conversar com um personagem 3D bem-feito. Não é fotorrealista, mas é **expressivo e sincronizado**.

### Turno 2 — Pergunta

```
Você:  "Can you explain conditionals?"

[Avatar inclina levemente a cabeça]
[Expressão de "pensando"]

Avatar:  "Sure! A conditional is..."
         [Boca se move conforme fala]
         [Olhar fixo em você]
         [Microexpressões a cada frase]
```

**Sensação (alvo):** como conversar com um professor virtual. Funcional e agradável.

### Turno 3 — Interrupção

```
Avatar:  "...so conditionals are used to—"

Você:  "Wait, can you give an example?"

[Avatar para de falar instantaneamente]
[Expressão muda para "ouvindo"]
[Leve inclinação de cabeça]

Avatar:  "Of course! For example, 'If it rains, I'll stay home.'"
```

**Sensação (alvo):** o avatar **reage** à interrupção. Parece vivo.

---

## 10. O Que Isso Habilita (visão do autor)

### 10.1 English Tutor Visual

Você não apenas ouve o professor — **vê** ele falar:
- **Compreensão:** ver a boca ajuda a entender fonemas
- **Engajamento:** conversar com um rosto é mais natural
- **Retenção:** memória visual + auditiva
- **Pronúncia:** você imita o movimento labial

### 10.2 Paper Citável (rascunho de quote — datar antes de citar)

> *"M³-AVM extends to visual interaction with a real-time 3D avatar driven by Audio2Face, running entirely on a heterogeneous rig of R$ 8.900. The avatar speaks in sync with Moshi's output, with 25ms additional latency and no additional hardware."*

### 10.3 Demo Impressionante (roteiro do autor)

Um vídeo de 30 segundos mostrando: você fala → avatar responde com rosto 3D → boca em sincronia → você interrompe, avatar para → você pergunta de novo, avatar responde.

---

## 11. Resumo (do autor, com a correção que importa)

| Aspecto | Valor |
|:---|:--:|
| **Abordagem** | Avatar 3D vetorizado (Audio2Face + mesh) |
| **VRAM adicional** | ~1.5 GB (realocada) |
| **Latência adicional** | ~25ms [META] |
| **Hardware adicional** | Nenhum |
| **Custo** | R$ 0–200 |
| **Tempo de implementação** | 8 semanas [META, pós-MVP-voz] |
| **Qualidade visual** | Boa (não fotorrealista) |
| **Sincronia labial** | Perfeita [META] |
| **Expressões** | 52 blendshapes ARKit |
| **Olhar** | Sim (com barge-in) |
| **Piscar** | Natural (aleatório) |

**A Abordagem A é a escolha certa** (julgamento do autor). Cabe no hardware, custa quase nada, e transforma "falar com IA" em "conversar com alguém" — **e, pelo §0.2, a v1 não exige nenhum opcode novo**.

---

## Anexo A — o que uma futura família avatar exigiria de verdade

1. **Alocação**: os 8 hexas propostos são ocupados; nova família exige
   RFC de alocação em faixa livre (faixas livres estão contestadas
   entre ESPEC §5.8 e ESPEC-V2 §3.7 — resolver primeiro).
2. **Modelo de execução**: blendshapes como tensor `[1,52]` GLOBAL +
   sidecars host-side já bastam para v1; ops `AVATAR_*` só se
   justificariam para orquestração in-VM (filtro §13 — hoje não passa).
3. **Regiões**: nenhuma nova (modelo de 16); buffers de frame vivem no
   host ou em tensores.
4. **Sincronia 30fps-vs-80ms**: `SET_DEADLINE` + `YIELD` já dão o
   pacing (o programa os usa corretamente); o problema é de integração
   de relógios, não de ISA.
5. **Pré-requisito**: MVP de voz (Mês 4–6 do `MVP_ENGLISH_TUTOR.md`) —
   avatar sem voz é cartaz, não conversa.

---

*Documento companheiro de `docs/MVP_ENGLISH_TUTOR.md` (o rig),
`docs/VISAO_PREMIUM_250GB.md` (o alvo maior) e
`docs/EXPERIENCIA_PREMIUM.md` (a sensação). Nada aqui altera ESPEC.md
(normativo).*

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
