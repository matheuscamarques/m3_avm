# M³-AVM — Planejamento Unificado: Hardware + MVP para Treino de Inglês

```text
Status:   META-ALVO (plano de produto + rig, NÃO-normativo, NÃO executado)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Companion: docs/VISAO_PREMIUM_250GB.md (engenharia) + docs/EXPERIENCIA_PREMIUM.md (sensação)
License:  AGPL-3.0-or-later (see LICENSE)
```

> **Contrato de honestidade (ESPEC §1.3):** hardware, preços, prazos e
> metas de fluência abaixo são **plano e pesquisa do autor**, não
> afirmação do projeto. Onde o rascunho toca o ISA, vale a legenda do
> §0 — vários itens que ele chama de "faltantes" já estão
> implementados, e "v2.4" colide com a governança de versões (ver §0.2).

## §0. Reconciliação com a árvore (2026-09-11)

**Legenda:** ✅ IMPL hoje · 🟡 DRAFT (precisa RFC+código) · 🔴 bloqueado
(diretiva/modo inexistente ou faixa RESERVED).

### §0.1 Correções aplicadas ao rascunho

1. **"M³-AVM v2.4 com ~30 opcodes" → ISA v1.14 com 106 opcodes.**
   `v2.x` é reservado para o freeze dual-mode (regra RFC-0028); "v2.4"
   aqui lê-se como codinome do marco, não versão do ISA.
2. **"Implementar 18 opcodes" desatualizado:** `DEPFORMER` (✅ RFC-0032),
   `STREAM_MERGE` (✅ RFC-0031), `VAD_DETECT` (✅ RFC-0031), `FOREST`
   (✅ RFC-0012), `PREEMPT_CHECK` (✅ RFC-0006) já existem. O que falta
   de verdade está no Anexo A.
3. **`SIGNAL` cross-thread, `FILLER_START/STOP`, `TEACH_MODE`,
   `PLACE/PIN/DEVICE_*`, `MODEL_LOAD_UNIFIED`** não existem (🔴) —
   os mecanismos reais estão no Anexo A (ex.: `PING` + palavra de
   controle no lugar de `FILLER_*`, `docs/FILLER_SPEECH.md` §5.1).
4. **O programa da Parte 3 não monta hoje** (diretivas `.data/.equ`,
   `CALL` com args, `LOADSTR`, modos com kwargs) — vive aqui como
   listagem-alvo (§0.2), não em `programs/`.
5. **Pesos são externos** (`*.gguf`/`*.safetensors` — git-ignored,
   excluídos do pack Zenodo): o plano assume downloads+quantização
   pelo usuário; nenhum peso mora neste repo.

### §0.2 Regra do programa-alvo

`programs/english_tutor.m3asm` promove para `programs/` fatiado por
marcos executáveis, na ordem da Parte 8 da VISÃO — cada fatia só entra
quando monta e executa de verdade (gate RFC-0008).

---

> **Objetivo:** Construir um rig heterogêneo de R$ 10.000 que roda a M³-AVM como assistente de conversação full-duplex em inglês, com meta de atingir nível B2 em 6 meses e C1 em 12 meses.
>
> **Documento único:** hardware, software, cronograma, orçamento, e programa de treino.

---

## Parte 1 — Visão Executiva

### O que você vai construir

Um **assistente pessoal de conversação em inglês** que:
- Roda 100% no seu hardware (privacidade total)
- Custa ~R$ 0.45/hora de uso (vs R$ 80–200/hora de professor)
- Fala e ouve ao mesmo tempo (full-duplex)
- Pode ser interrompido em ~50ms (barge-in) [META — medido hoje: ABORT 32×32 ~25ms no emulador CPU, ESPEC §17; 300µs é alvo de hardware]
- Lembra do contexto de 30 minutos de conversa [META — 128k de contexto-alvo]
- Corrige gramática e vocabulário em tempo real [META — modo professor, §3.1 P2]
- Aprende seu nível e ajusta a dificuldade [META — LoRA/Trilha P]

### Por que isso importa

- **Para você:** ferramenta real que muda sua fluência em inglês
- **Para o projeto:** caso de uso concreto que valida a arquitetura M³-AVM
- **Para a comunidade:** primeiro demo real de multi-modelo em rig barato
- **Para o paper:** evidência citável de uso real, não benchmark sintético

### Números-chave (metas do autor)

| Item | Valor |
|:---|:--:|
| Investimento em hardware | R$ 8.900 |
| Custo por hora de uso | ~R$ 0.45 |
| Tempo até primeiro demo | 8 semanas |
| Tempo até uso diário | 4 meses |
| Nível alcançável em 6 meses | B2 |
| Nível alcançável em 12 meses | C1 |

---

## Parte 2 — Hardware (pesquisa do autor, a validar na compra)

### 2.1 Lista de componentes

| Componente | Especificação | Qtd | Preço unit. | Subtotal | Função |
|:---|:---|:--:|:--:|:--:|:---|
| **GPU 0** | GTX 1650 4GB (usada) | 1 | R$ 600 | R$ 600 | Mamba-2.8B + BGE-M3 |
| **GPU 1** | RTX 3060 12GB (usada) | 1 | R$ 1.500 | R$ 1.500 | Llama-8B |
| **GPU 2** | RTX 3060 12GB (usada) | 1 | R$ 1.500 | R$ 1.500 | Moshi-7B full-duplex |
| **Placa-mãe** | Huananzhi X99-F8D Plus (dual socket) | 1 | R$ 700 | R$ 700 | 3 slots PCIe x16 + 3 x8 |
| **CPU** | Intel Xeon E5-2670 v3 (12C/24T) | 2 | R$ 180 | R$ 360 | Orquestração, RAG, I/O |
| **RAM** | DDR4 ECC RDIMM 16GB 2133MHz | 4 | R$ 400 | R$ 1.600 | 64 GB total |
| **Fonte** | 850W 80 Plus Gold | 1 | R$ 550 | R$ 550 | Margem para picos |
| **Armazenamento** | SSD NVMe 1TB | 1 | R$ 1.100 | R$ 1.100 | SO + modelos + RAG |
| **Gabinete** | Rack 4U (E-ATX) | 1 | R$ 400 | R$ 400 | Ventilação + espaço |
| **Cooler CPU** | LGA 2011 | 2 | R$ 100 | R$ 200 | Dissipação térmica |
| **Risers PCIe** | x16, cabo 20cm, PCIe 3.0 | 3 | R$ 130 | R$ 390 | **Obrigatório** |
| **TOTAL** | | | | **R$ 8.900** | |

**Margem restante:** R$ 1.100 (para imprevistos ou upgrade).

### 2.2 Poder combinado (estimativa do autor)

| Recurso | Total | Comentário |
|:---|:--:|:---|
| **GPUs** | 3 (24 GB VRAM agregada) | 4 + 12 + 12 GB |
| **CPUs** | 2× Xeon E5-2670 v3 (24 cores, 48 threads) | Orquestração |
| **RAM** | 64 GB DDR4 ECC | Suficiente para RAG + fallback |
| **FP32 combinado** | ~28 TFLOPS | GPUs dominam |
| **Consumo** | ~500W em carga | ~R$ 27/mês (2h/dia) |

### 2.3 Topologia

```
┌─────────────────────────────────────────────────────────────────┐
│                    RIG M³-AVM                                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ GPU 0        │  │ GPU 1        │  │ GPU 2        │          │
│  │ GTX 1650     │  │ RTX 3060     │  │ RTX 3060     │          │
│  │ 4 GB         │  │ 12 GB        │  │ 12 GB        │          │
│  │              │  │              │  │              │          │
│  │ Mamba        │  │ Llama-8B     │  │ Moshi        │          │
│  │ BGE-M3       │  │              │  │              │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                  │
│         │  Riser PCIe     │  Riser PCIe     │  Riser PCIe      │
│         │                 │                 │                  │
│         └─────────────────┼─────────────────┘                  │
│                           │                                    │
│                  ┌────────▼────────┐                           │
│                  │ Huananzhi X99   │                           │
│                  │ (dual socket)   │                           │
│                  └────────┬────────┘                           │
│                           │                                    │
│         ┌─────────────────┼─────────────────┐                  │
│         │                 │                 │                  │
│    ┌────▼────┐       ┌────▼────┐      ┌────▼────┐            │
│    │ Xeon 0  │       │ Xeon 1  │      │ 64 GB   │            │
│    │ 12c/24t │       │ 12c/24t │      │ DDR4    │            │
│    └─────────┘       └─────────┘      └─────────┘            │
│                                                                 │
│  Storage: NVMe 1TB  │  Fonte: 850W Gold                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Onde comprar (pesquisa do autor)

| Componente | Onde | Dica |
|:---|:---|:---|
| GPUs usadas | OLX, HardMob, Mercado Livre | Peça vídeo de benchmark antes de pagar |
| Placa-mãe | AliExpress | Procure "Huananzhi X99-F8D Plus" |
| CPUs Xeon | OLX, AliExpress | Lotes de servidores desativados |
| RAM ECC | OLX, AliExpress | Verifique compatibilidade (RDIMM, não UDIMM) |
| Fonte / SSD / Gabinete | Kabum, Pichau, Terabyte | Novos, com garantia |
| Risers PCIe | AliExpress, Mercado Livre | Cabo ≥20cm, chip PLX se possível |

---

## Parte 3 — Software (estado real no Anexo A)

### 3.1 O que o rascunho pedia vs. o que existe (2026-09-11)

| Componente | Opcodes (rascunho) | Estado real |
|:---|:---|:--:|
| **Mimi codec** | `CODEC_ENC` (0x15), `CODEC_DEC` (0x16) | ✅ IMPL |
| **PersonaPlex Temporal** | `ATTN` (0x02), `FFN` (0x08), `ROPE` (0x19) | ✅ IMPL |
| **Depformer** | `DEPFORMER` (0x44) | ✅ IMPL (RFC-0032, single-stream) |
| **Stream merge** | `STREAM_MERGE` (0x45) | ✅ IMPL (RFC-0031, 2-mix) |
| **Barge-in** | `VAD_DETECT` (0x46), `SIGNAL` (0x1B), `ABORT` (0x05) | ✅ IMPL (local) |
| **RAG** | `RAG_SEARCH` (0x52), `EMBED_LOOKUP` (0x53) | 🟡 DRAFT (Fase 6) |
| **Filler speech** | `FILLER_START`, `FILLER_STOP` (novos) | 🔴 sem kinds novos — `PING`+controle (`FILLER_SPEECH.md` §5.1) |
| **Teach mode** | `TEACH_MODE` (novo) | 🔴 sem opcode (comportamento em programa, futuro) |

### 3.2 Cronograma de 8 semanas (alvo do autor, a revalidar)

| Semana | Foco | Entregável |
|:--:|:---|:---|
| **1** | Ambiente + Mimi | `SENSE AUDIO_PCM` + `CODEC_ENC/DEC` funcionando [✅ já funciona] |
| **2** | Temporal Transformer | ASR (voz → texto) em 1 frame [🟡 precisa pesos + programa] |
| **3** | Depformer | TTS (texto → voz) em 1 frame [🟡 op existe; precisa pesos] |
| **4** | Barge-in | Interrupção em <100ms [🟡ABORT medido ~25ms CPU; <100ms plausível] |
| **5** | RAG | Indexação + busca de vocabulário [🔴 Fase 6 pendente] |
| **6** | Orquestração | Threads RED/BLUE/GREEN coordenadas [🟡 scheduler existe; programa pendente] |
| **7** | Otimização | Latência reduzida, filler speech [🟡 design existe; `DEPFORMER`/voz pendente] |
| **8** | Demo | Sessão de 5 min em inglês [🔴 depende de 2–7] |

### 3.3 Programa `.m3asm` principal (listagem-alvo, NÃO monta — ver §0.2)

`programs/english_tutor.m3asm` (destino futuro)

```asm
; ============================================================
; M³-AVM English Tutor — MVP (ALVO)
; Hardware: GTX 1650 + 2× RTX 3060
; Uso: prática diária de inglês (B1 → C1)
; ============================================================

.data                                                        ; 🔴 assembler
    PCM_FRAME_LEN       .equ 1920          ; 80ms @ 24kHz    ; 🔴
    N_CODEBOOKS         .equ 16                              ; 🔴
    N_STREAMS           .equ 17                              ; 🔴
    RAG_TOPK            .equ 5                               ; 🔴
    TEACH_MODE          .equ 1             ; 0=conversa, 1=professor  ; 🔴
    LEVEL               .equ 2             ; 0=A2, 1=B1, 2=B2, 3=C1   ; 🔴

.text                                                        ; 🔴
main:
    ; --- Carrega modelos ---
    LOAD_MODEL rPersona, "personaplex-7b-Q4_K.gguf"          ; 🟡 0xA0
    LOAD_MODEL rMamba,   "mamba-2.8b-Q4_K.gguf"              ; 🟡
    LOAD_MODEL rLlama,   "llama-3.1-8b-Q4_K.gguf"            ; 🟡
    MODEL_LOAD_UNIFIED rBge, "bge-m3.safetensors"            ; 🔴 0xFC reservada!

    ; --- Placement ---
    PLACE rMamba,   DEVICE=0    ; GTX 1650                   ; 🔴 0xBA reservada!
    PLACE rPersona, DEVICE=1    ; RTX 3060 #1                ; 🔴
    PLACE rLlama,   DEVICE=2    ; RTX 3060 #2                ; 🔴
    PLACE rBge,   DEVICE=0    ; GTX 1650 (junto com Mamba)   ; 🔴

    ; --- RAG: material de estudo ---
    RAG_INDEX_ADD rRag, FILE="grammar_b2.jsonl"              ; 🟡 0x50
    RAG_INDEX_ADD rRag, FILE="vocabulary_b2.jsonl"           ; 🟡
    RAG_INDEX_ADD rRag, FILE="phrasal_verbs.jsonl"           ; 🟡
    RAG_INDEX_ADD rRag, FILE="business_english.jsonl"        ; 🟡

    ; --- Contextos ---
    SPAWN_CONTEXT rAudioIn,  ENTRY=audio_input,  PRIORITY=RED    ; 🟡 0xA2
    SPAWN_CONTEXT rAudioOut, ENTRY=audio_output, PRIORITY=RED    ; 🟡
    SPAWN_CONTEXT rReason,   ENTRY=reason_loop,  PRIORITY=BLUE   ; 🟡

    JUMP idle                                                ; ✅ 0x0E

; ============================================================
; ÁUDIO INPUT — Captura e VAD
; ============================================================
audio_input:
    SENSE rPCM, AUDIO_PCM, LEN=PCM_FRAME_LEN                ; 🔴 modo (opcode ✅ 0x06)
    VAD_DETECT rVAD, rPCM THRESHOLD=0.3                     ; 🔴 modo (opcode ✅ 0x46)
    COMPARE rVAD, 0.3                                       ; 🔴 float imediato (inteiros escalados)
    IF_GREATER rVAD, barge_in_check                         ; 🔴 (só IF_EQUAL/IF_INTERRUPT)

    CODEC_ENC rCodes, rPCM                                  ; ✅ 0x15
    STREAM_WRITE rAudioIn, rCodes                            ; 🔴 modo (opcode STREAM ✅)
    SIGNAL CTX=rReason, KIND=PING                           ; 🔴 forma (opcode ✅ 0x1B)
    JUMP audio_input                                        ; ✅

barge_in_check:
    SIGNAL CTX=rReason, KIND=ABORT                          ; 🔴 forma (opcode ✅)
    ABORT rSnapAudio                                        ; ✅ 0x05 (forma: ABORT Rs_ctx, Rs_ts)
    JUMP audio_input                                        ; ✅

; ============================================================
; RAZÃO — Pipeline de conversação
; ============================================================
reason_loop:
    ; --- ASR ---
    STREAM_READ rCodes, rAudioIn                            ; 🔴 modo (opcode ✅)
    STREAM_MERGE rUserMixed, rCodes, rCodes MODE=user_only  ; 🔴 modo (opcode ✅ 0x45)
    CALL temporal_forward rUserMixed, rHidden               ; 🔴 CALL com args (só label)
    MATVEC rTextLogits, rHidden, rPersona.text_head         ; ✅ 0x10 (field access 🔴)
    SAMPLE rTranscript, rTextLogits TEMPERATURE=0.1 TOPK=1  ; ✅ 0x0B (kwargs existem)

    ; --- Classificar complexidade ---
    EMBED rQueryEmb, rTranscript, rBge                      ; 🔴 modo DIM= (opcode ✅ 0x09)
    FOREST rClassify, rFeatures, rXgb                       ; ✅ 0x22 (rFeatures indefinido no escopo!)
    SLICE rComplexity, rClassify, 1, 2                      ; ✅ 0x2E (posicional)

    ; --- RAG retrieval ---
    RAG_SEARCH rDocs, rQueryEmb, rRag TOPK=RAG_TOPK          ; 🟡 0x52
    CALL load_docs rDocs, rContext                          ; 🔴 CALL com args

    ; --- Decidir path ---
    COMPARE rComplexity, 0.4                                ; 🔴 float imediato
    IF_GREATER rComplexity, deep_path                       ; 🔴
    JUMP fast_path                                        ; ✅

fast_path:
    ; Mamba (rápido) + filler
    SIGNAL CTX=rAudioOut, KIND=FILLER_START                 ; 🔴 kind (usar PING+controle)
    CALL mamba_forward rTranscript, rMamba, rOut            ; 🔴 CALL com args
    SIGNAL CTX=rAudioOut, KIND=FILLER_STOP                  ; 🔴 kind
    JUMP synthesize                                       ; ✅

deep_path:
    ; Llama (profundo) + filler
    SIGNAL CTX=rAudioOut, KIND=FILLER_START                 ; 🔴 kind
    CALL llama_forward rTranscript, rContext, rLlama, rOut  ; 🔴 CALL com args
    SIGNAL CTX=rAudioOut, KIND=FILLER_STOP                  ; 🔴 kind
    JUMP synthesize                                       ; ✅

synthesize:
    SAMPLE rResponse, rOut TEMPERATURE=0.7 TOPK=50          ; ✅

    ; --- Modo professor: correção ---
    COMPARE TEACH_MODE, 1                                   ; 🔴 símbolo .data
    IF_EQUAL TEACH_MODE, correct_english                    ; 🔴 (IF_EQUAL pula p/ label, não compara símbolo)

    JUMP emit_response                                    ; ✅

correct_english:
    ; Detectar erros na fala do usuário
    CALL detect_errors rTranscript, rErrors                 ; 🔴 CALL com args
    COMPARE rErrors.count, 0                                ; 🔴 field access
    IF_GREATER rErrors.count, emit_correction               ; 🔴 (duplo)

emit_response:
    EMBED rTextEmb, rResponse, rPersona.emb                 ; 🔴 modo (opcode ✅)
    DEPFORMER rCodesOut, rTextEmb, rAudioEmb, rKVDep        ; 🔴 forma (opcode ✅ 0x44; DEPFORMER real: rD,rX,rW+tabela)
    STREAM_WRITE rAudioOut, rCodesOut                       ; 🔴 modo
    JUMP reason_loop                                      ; ✅

emit_correction:
    ; Emitir correção primeiro
    CONCAT rFull, rErrors.correction, rResponse             ; 🟡 0x2F (field access 🔴)
    JUMP emit_response                                    ; ✅
```

> Nota de lógica (não só sintaxe): `IF_EQUAL TEACH_MODE, correct_english`
> compara registradores, não símbolos — o modo professor seria um
> `LOADI` + `COMPARE` + branch, quando `.data` existir. E `FOREST`
> lê `rFeatures`, que o programa nunca constrói (`CONCAT` 🟡 Exists).
> A auditoria acima marca forma; o fluxo de dados precisa do mesmo
> rigor antes de qualquer fatia promover para `programs/`.

---

## Parte 4 — Programa de Treino de Inglês (plano do autor)

### 4.1 Sessão diária (30 minutos)

| Fase | Duração | O que fazer |
|:---|:--:|:---|
| **Warm-up** | 3 min | "How was your day?" — conversa casual |
| **Vocabulário** | 7 min | 10 palavras novas com exemplos |
| **Gramática** | 5 min | Um tópico (present perfect, conditionals) |
| **Role-play** | 10 min | Simulação (entrevista, restaurante, reunião) |
| **Review** | 5 min | Erros do dia + repetição |

### 4.2 Progressão de nível (metas do autor)

| Mês | Nível | Foco | Meta |
|:--:|:---|:---|:---|
| 1–2 | B1 | Conversa básica, presente | Fluência em frases simples |
| 3–4 | B1+ | Passado, futuro, phrasal verbs | Conversa casual |
| 5–6 | B2 | Conditionals, modais, nuance | Discussões técnicas |
| 7–9 | B2+ | Idioms, ironia, cultura | Naturalidade |
| 10–12 | C1 | Debate, negociação, humor | Fluência profissional |

### 4.3 Material para RAG (lista do autor; arquivos externos, não no repo)

| Arquivo | Conteúdo | Tamanho |
|:---|:---|:--:|
| `grammar_b2.jsonl` | Gramática B2 (conditionals, modais) | ~1 MB |
| `vocabulary_b2.jsonl` | 5000 palavras B2 com exemplos | ~5 MB |
| `phrasal_verbs.jsonl` | 500 phrasal verbs | ~500 KB |
| `business_english.jsonl` | Inglês de negócios | ~2 MB |
| `idioms.jsonl` | 1000 idioms | ~1 MB |
| `pronunciation.jsonl` | IPA + áudio de referência | ~10 MB |

### 4.4 Modos de uso (propostas — nenhum existe; modos viriam como programas, não opcodes)

| Modo | Comando | Quando usar |
|:---|:---|:---|
| **Casual** | `MODE=casual` | Praticar fluência sem pressão |
| **Teacher** | `MODE=teacher` | Correção ativa + explicações |
| **Shadowing** | `MODE=shadow` | Repetir frases para pronúncia |
| **Role-play** | `MODE=role,SCENARIO=interview` | Simular situações |
| **Quiz** | `MODE=quiz,TOPIC=grammar` | Testar conhecimento |

---

## Parte 5 — Cronograma de 12 Meses (alvo do autor)

### Mês 1–2: Hardware + Base

| Semana | Foco |
|:--:|:---|
| 1–2 | Comprar componentes, montar rig |
| 3 | Instalar SO, drivers, CUDA |
| 4 | Compilar M³-AVM, rodar testes |
| 5 | Implementar Mimi codec [✅ já existe — pular] |
| 6 | Implementar Temporal Transformer [🟡 parcial: ops existem, programa+pesos pendentes] |
| 7 | Implementar Depformer [✅ op existe (RFC-0032); programa+pesos pendentes] |
| 8 | Primeiro demo: 1 frame de áudio |

**Marco:** áudio entra, áudio sai.

### Mês 3–4: Barge-in + RAG

| Semana | Foco |
|:--:|:---|
| 9–10 | Implementar VAD + barge-in [✅ ops existem; programa pendente] |
| 11–12 | Implementar RAG [🔴 Fase 6 pendente] |
| 13–14 | Implementar filler speech [🟡 design pronto (`FILLER_SPEECH.md`); voz pendente] |
| 15–16 | Primeira conversa de 5 minutos |

**Marco:** conversa interativa em inglês.

### Mês 5–6: Uso Diário

| Semana | Foco |
|:---|:--:|
| 17–18 | Implementar Teach Mode [🔴 comportamento em programa; depende de RAG] |
| 19–20 | Indexar material B1/B2 [🔴 depende de RAG] |
| 21–24 | Uso diário (30 min/dia) |

**Marco:** B1 → B2.

### Mês 7–9: Refinamento

| Semana | Foco |
|:---|:--:|
| 25–28 | Implementar Shadow Mode [🔴 programa futuro] |
| 29–32 | Fine-tuning LoRA (sotaque pessoal) [🔴 Trilha P] |
| 33–36 | Adicionar idiomas, gírias [🔴 futuro] |

**Marco:** B2+ consolidado.

### Mês 10–12: C1

| Semana | Foco |
|:---|:--:|
| 37–40 | Debate mode, negociação [🔴 programa futuro] |
| 41–44 | Humor, ironia, cultura [🔴 programa futuro] |
| 45–48 | Preparar certificação (IELTS, TOEFL) [meta pessoal] |

**Marco:** C1, pronto para certificação.

---

## Parte 6 — Riscos e Mitigações (do autor; estado atual entre colchetes)

| Risco | Prob. | Impacto | Mitigação [estado] |
|:---|:--:|:--:|:---|
| GPU usada com defeito | Média | Alto | Testar com benchmark antes de pagar [processo de compra — vale] |
| Sotaque PT-BR no Moshi | Alta | Médio | PersonaPlex é EN nativo; usar só em inglês [✅ política registrada em `FILLER_SPEECH.md` §8] |
| Llama-8B fraco | Alta | Médio | Aceitar limitação; usar Mamba para rápido [estratégia fast/deep já desenhada] |
| Latência > 1s | Média | Alto | Filler speech + quantização [🟡 design pronto; medição pendente] |
| Contexto enche em 30 min | Alta | Baixo | Reiniciar sessão; KV compress [✅ `KV_TRUNCATE` + `KV_COMPRESS` existem] |
| Custo de energia | Baixa | Médio | Rodar 2h/dia, não 24/7 [vale] |

---

## Parte 7 — Métricas de Sucesso

### Técnicas (metas; medir quando houver rig)

| Métrica | Meta | Como medir |
|:---|:--:|:---|
| First sound | <400ms | `M3_PROFILE=1` |
| Barge-in | <100ms | Benchmark |
| Throughput Llama | >10 tok/s | Benchmark |
| Throughput Moshi | >12.5 fps | Benchmark |
| Uptime | >99% | Log |

### Pessoais (metas do autor)

| Métrica | Meta (6 meses) | Meta (12 meses) |
|:---|:--:|:--:|
| Nível CEFR | B2 | C1 |
| Vocabulário ativo | 3000 palavras | 6000 palavras |
| Fluência | Conversa casual | Debate técnico |
| Sotaque | Compreensível | Neutro |
| Certificação | — | IELTS 7.0 |

---

## Parte 8 — Resultado Final (visão do autor)

### O que você terá em 12 meses

**Hardware:**
- Rig de R$ 8.900 rodando 24/7 (quando ativo)
- 3 GPUs, 24 GB VRAM, 64 GB RAM, 2 CPUs
- Consumo: ~500W, custo ~R$ 27/mês

**Software:**
- M³-AVM com assistente full-duplex, Teach/Shadow/Role-play, RAG com 5000+ palavras
- [Corrigido: a linha original dizia "v2.4 com ~30 opcodes" — ver §0.1.]

**Pessoal:**
- Nível C1 em inglês
- Ferramenta que você usa todos os dias
- Caso de uso real para o paper
- Propriedade intelectual citável

**Comunidade:**
- Repo no GitHub com demo real
- Paper no arXiv
- Post no Hacker News
- Possíveis contribuidores

### O pitch final (do autor)

> *"I built M³-AVM — an open-source virtual machine for deterministic AI inference — because I wanted to practice English with a full-duplex voice assistant on my own hardware. It runs on a R$ 8.900 heterogeneous rig, costs R$ 0.45/hour, and helped me go from B1 to C1 in 12 months. The architecture, the RFCs, and the reference implementation are all public."*

Isso é honesto, citável, e **verdadeiro** — como alvo. Como fato, hoje: o rig não foi comprado, o assistente não roda, o C1 não veio. O valor deste documento é o mapa, não o território.

---

## Anexo A — §3.1 destrinchado: o que destrava cada linha

| Linha do rascunho | Estado | Desbloqueio |
|:---|:---|:---|
| Mimi codec | ✅ feito | — |
| Temporal (ATTN/FFN/ROPE) | ✅ feito | programa + pesos |
| DEPFORMER | ✅ feito (RFC-0032) | programa + pesos |
| STREAM_MERGE | ✅ feito (RFC-0031) | programa |
| Barge-in (VAD/SIGNAL/ABORT) | ✅ feito (ops locais) | programa de coordenação |
| RAG (SEARCH/LOOKUP) | 🟡 Fase 6 | RFC + implementação |
| FILLER_START/STOP | 🔴 sem kinds novos | padrão `PING`+controle (existe) |
| TEACH_MODE | 🔴 sem opcode | comportamento em programa (pós-RAG) |

## Anexo B — programa `english_tutor`: checklist por construto

| Construto | Status | Caminho |
|:---|:---|:---|
| `.data/.equ/.str/.text` | 🔴 | Fase 9 (assembler) |
| `LOAD_MODEL`, `SPAWN_CONTEXT` | 🟡 | Fase 9 (system ops 0xA0–0xAF) |
| `MODEL_LOAD_UNIFIED`, `PLACE/PIN`, `DEVICE_*` | 🔴 | faixas RESERVED (R12) |
| `RAG_INDEX_ADD/SEARCH`, `EMBED_LOOKUP` | 🟡 | Fase 6 |
| `SENSE LEN=`, `STREAM_*`, `SIGNAL NODE=/URGENCY` | 🔴 modos | programas com formas atuais |
| `CALL` com args, `LOADSTR`, `CONCAT`+fields | 🔴/🟡 | `CALL` label (✅ RFC-0034) + resto Fase 9 |
| `FOREST`/`SAMPLE`/`MATVEC`/`DEPFORMER` formas base | ✅ | — |
| `WFI`, `CHECKPOINT_SAVE`, `GRAD_*`, `OPTIMIZER_*` | 🟡/🔴 | MMIO/system futuros + Trilha P |

---

*Documento companheiro de `docs/VISAO_PREMIUM_250GB.md` (o alvo de 250GB),
`docs/EXPERIENCIA_PREMIUM.md` (a sensação) e `docs/FILLER_SPEECH.md` (a voz
que preenche o silêncio). Nada aqui altera ESPEC.md (normativo).*

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
