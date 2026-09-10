# RELATÓRIO DESCRITIVO DA INVENÇÃO (minuta revisada — 2 camadas)

**Inventor:** Matheus de Camargo Marques — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258) · matheuscamarques@gmail.com · Independent Researcher

## Título
**SISTEMA E MÉTODO PARA ORQUESTRAÇÃO DE MÚLTIPLOS MODELOS DE INTELIGÊNCIA ARTIFICIAL HETEROGÊNEOS EM CONTEXTO ÚNICO DE MÁQUINA VIRTUAL COM ESTADO COMPARTILHADO E ROLLBACK CONJUNTO**

> **Aviso de leitura (vinculante para interpretação):** este relatório distingue
> **(C1) Núcleo habilitado** — implementado e testado em `src/*.rs` conforme
> `docs/ESPEC.md` ISA v1.4 — de **(C2) Modalidade preferencial futura** —
> codificações congeladas (`RSVD`) ou propostas (`DRAFT`) em `docs/ESPEC.md
> §§12–13` e `docs/ESPEC-V2.md §3`, ainda sem transporte/execução. (C2) é
> descrita como desenvolvimento previsto com regra de habilitação explícita,
> não como resultado medido. Números sem fonte em `docs/ESPEC.md §17` são
> marcados `alvo`. Versão final para depósito deve ser revisada por agente
> de PI credenciado.

---

## 1. CAMPO DA INVENÇÃO

A presente invenção pertence ao campo da computação, mais especificamente a:

- **Máquinas virtuais e runtime de execução** (CPC: G06F9/455)
- **Sistemas de inferência de inteligência artificial** (CPC: G06N3/063, G06N3/10)
- **Gerenciamento de memória para aprendizado de máquina** (CPC: G06F12/08)
- **Escalonamento de tarefas** (CPC: G06F9/4881)

A invenção é aplicável a qualquer sistema que necessite executar, de forma
coordenada, múltiplos modelos de IA de arquiteturas distintas em um único
processo computacional, com interrupção (barge-in), correção cirúrgica e
retomada bit-exata do prefixo confirmado.

**Melhor modo conhecido pelo inventor (art. 24 LPI):** o protótipo em Rust
(`src/lib.rs`, `src/vm.rs`, `src/memory.rs`, `src/context.rs`,
`src/inference.rs`, `src/ssm.rs`, `src/mimi.rs`), ISA v1.4 de 32 bytes,
4 regiões, scheduler de prioridade estrita, `FORK/ABORT` com CoW, executando
modelos Transformer GGUF via `mmap` + `KV_CACHE`, com laços Mamba/SSM e
codec de áudio no mesmo espaço de endereço. É esse modo que garante
suficiência descritiva; (C2) é extensão.

---

## 2. FUNDAMENTOS DA INVENÇÃO

### 2.1 Estado da Técnica

**(a) Processos isolados:** Ray Serve, NVIDIA Triton, TorchServe executam cada
modelo em processo separado, com serialização (gRPC/HTTP/JSON). A técnica
reconhece sobrecarga de cópia e latência por transição; valores variam por
carga e não são reivindicados aqui como medição própria, exceto quando
citados de `docs/ESPEC.md §17` para o protótipo.

**(b) Ausência de estado compartilhado nativo:** cada modelo mantém KV cache,
estado recorrente ou potenciais em heaps isolados, sem primitiva de
compartilhamento zero-copy cross-modelo no ISA.

**(c) Rollback por modelo:** vLLM com PagedAttention (Kwon et al., SOSP'23, arXiv:2309.06180) gerencia KV paginado com CoW **intra-modelo** (mesma arquitetura, foco em batching/throughput); TensorRT-LLM idem, sem restore atômico transversal a Transformer + SSM + matriz recorrente em ISA única.

**(d) Deadlines por modelo:** KServe (`timeoutSeconds` por componente) e Seldon (`rest-timeout`/`grpc-timeout` por nó — com ressalva oficial de que não cobrem o grafo fim-a-fim) tratam SLA por serviço, sem operador nativo de composição entre modelos interdependentes.

**(e) Scheduler genérico vs topológico:** Kubernetes padrão/Nomad escalonam por CPU/mem; a geração topológica (K8s 1.36 KEP-5732, NVIDIA Run:ai/KAI, Topology Manager + GFD) decide **onde** colocar (NVLink/NUMA/rack), não **quando preemptar por deadline com herança**; EDF clássico (SRP de Baker, DFP) é fundamento de RTOS, sem instanciação em ISA de tensores heterogêneos.

**(f) Transferência cross-model (matemática conhecida, runtime aberto):** Heo et al. arXiv:2608.03893 (ridge por cabeça, top-k camadas, RoPE-strip, 500×1024), CacheBridge arXiv:2609.00891 e Universal Context-Reuse arXiv:2608.30963 descrevem o mapeamento; nenhum propõe opcode com integração a snapshot/rollback — objeto da presente claim 10 de mecanismo.

### 2.2 Limitações do Estado da Técnica

1. Custo de serialização/cópia entre processos por transição entre modelos.
2. Sem restore coordenado, interrupção multi-modelo (ex.: barge-in durante
   raciocínio) deixa estados semanticamente inconsistentes.
3. Banda de interconexão subaproveitada por camadas de comunicação duplicadas.
4. Composição de deadlines exige lógica de aplicação ad hoc.
5. Transferência de estado entre modelos (ex.: cross-model KV) sem suporte
   de runtime nativo.

### 2.3 Problema Técnico a Ser Resolvido

**Como executar múltiplos modelos de IA heterogêneos em um único contexto
computacional, com compartilhamento de memória sem serialização, rollback
conjunto determinístico, deadlines por contexto e comutação nativa entre
motores, de forma habilitada em software convencional (CPU/GPU)?**

A solução reivindicada: **uma VM de 32 bytes por instrução (C1), com regiões
tipadas, motores sobre memória compartilhada, snapshot/restore CoW
transversal e scheduler por prioridade + deadline absoluto (C1), extensível a
deadlines compostos, afinidade de dispositivo e transferência nativa de
estado (C2).**

---

## 3. SUMÁRIO DA INVENÇÃO

### 3.1 Arquitetura de máquina virtual unificada (C1 + extensão C2)

**(C1 — habilitado.)** VM em um único processo, espaço virtual de 128 bits
(`u128`, `src/memory.rs:71-86`, `src/context.rs:93`), **4 regiões com tag no
byte alto**: `GLOBAL 0x00`, `TEMPORAL 0x10` (circular 1 GiB lógico / 64 MiB
físico dev), `PERSISTENTE 0x20` (`mmap` `m3_persistent.dat`, pesos GGUF
read-only zero-copy com `madvise WillNeed`), `KV_CACHE 0x30` (por camada,
`MAX_SEQ 2048`, append + truncate). Tensores densos (`Vec<u8>` + `TensorMeta`,
alinhados 64B), esparsos CSR (`nalgebra-sparse`/`sprs`) e pesos quantizados
(`Q4_0/Q4_K/Q6_K/Q8_0` + `matvec_q4k` AVX2 fundido). Isolamento por ator
(estilo Erlang/OTP) via fatiamento por contexto; `SHARED` zero-copy é
**(C2)** (região lógica `0x6`, `ESPEC-V2 §7`), hoje obtido por `GLOBAL` +
`FENCE 0x77` + `LOCK/UNLOCK 0x75/0x76`.

**(C2 — futuro, compatível.)** Modelo lógico de 16 regiões (`ESPEC-V2 §7.1`:
TEXT 0x0, GLOBAL 0x1, WEIGHTS 0x2, ACTIVATION 0x3, KV_CACHE 0x4, ARENA 0x5,
SHARED 0x6, WAL 0x7, SNAPSHOT 0x8, STREAM_RING 0x9, RAG_INDEX 0xA,
CLUSTER_STAGING 0xB, FEDERATED 0xC, CONFIDENTIAL 0xD, SCRATCH_GPU 0xE,
MMIO 0xF), com perfil de compatibilidade 32B = 4 regiões atuais (§7.2) e
tabela de regiões em MMIO+0x0. Instruções 64B (`ESPEC-V2 §4.2`) endereçam as
16; instruções 32B, só o perfil.

### 3.2 Conjunto de instruções especializado (C1 + C2)

**(C1 — IMPL, 32 bytes fixos.)** Formato (`src/opcodes.rs:1-10`,
`docs/ESPEC.md §4`): `opcode 1B | flags 1B | rdest/rsrc1/rsrc2/rsrc3 4B |
payload 26B LE`. `INSTR_SIZE=32`. Assembler 2-pass com labels.
Opcodes habilitados: `0x00 HALT`, `0x01 TENSOR`, `0x02 ATTN`, `0x03 STREAM`,
`0x04 FORK`, `0x05 ABORT`, `0x06 SENSE`, `0x07 NORM`, `0x08 FFN`,
`0x09 EMBED`, `0x0A ADD`, `0x0B SAMPLE`, `0x0C COMPARE`, `0x0D IF_EQUAL`,
`0x0E JUMP`, `0x0F IF_INTERRUPT`, `0x10 MATVEC`, `0x11 MUL`, `0x12 SILU`,
`0x13 SSM_SCAN`, `0x14 SSM_RESET`, `0x15 CODEC_ENC`, `0x16 CODEC_DEC`,
`0x17 AUDIO_ALIGN`, `0x18 CTX_SWITCH`, `0x19 ROPE`, `0x1F GATHER`,
`0x23 DISTANCE`, `0x24 RANK1_UPDATE`, `0x38 KV_TRUNCATE`,
`0x60–0x66` RNG/hash, `0x6A–0x77` telemetria/scheduler,
`0x78 LOADI`, `0x79 MOV`, `0xFF NOP`.

**(C2 — congelado/proposto.)** `0x1A–0x1D` cluster básico, `0x1E CONV`,
`0x20 SPIKE_STEP`, `0x21 DENOISE_STEP`, `0x22 FOREST`, `0x25 ODE_STEP`,
`0x26–0x2F` memória/arena, `0x30–0x43` tensores avançados, `0x44–0x49` áudio
(`DEPFORMER`, `STREAM_MERGE`), `0x50–0x57` retrieval (`RAG_SEARCH`,
`EMBED_LOOKUP`), `0x58–0x5F` ML clássico (`SVM_PREDICT`, `KMEANS_STEP`, …),
`0x80–0xAF` formas-X 64B, bloco ESCAPE `0xB0–0xB6`. O emulador **rejeita com
erro explícito** até especificação/execução (regra de governança
`ESPEC-V2 R11`).

### 3.3 Rollback conjunto determinístico (C1, com 1 obrigação aberta declarada)

`FORK` empilha snapshot CoW (`Arc::clone` dos heaps + mapas por store +
`ssm_states` + `rank1_layers` + camadas KV); `ABORT` desempilha e restaura por
troca de ponteiros (`src/memory.rs`, `src/vm.rs`, `src/rollback.rs`).
Cobre KV de Transformer, `h_t` de SSM, `H` de `RANK1_UPDATE` e truncamento KV
(`KV_TRUNCATE`). **I-Mono IMPL** (contador monotônico, `restore` não rebaixa;
`formal/Formal/Rollback.lean`, arbiters verdes). **I-Persist (não-mutação
pós-snapshot) é OBRIGAÇÃO ABERTA declarada** (`ESPEC §6.3`): requerida, ainda
não fiscalizada no código — o relatório não a afirma como pronta.
Determinismo: `SAMPLE` usa RNG splitmix64 por contexto (RFC-0009); mesma
semente + mesmos logits = replay exato no mesmo host/build.

### 3.4 Deadlines (C1 simples; C2 compostos)

**(C1.)** Todo contexto carrega `deadline` absoluto (ns) via
`SET_DEADLINE 0x71` / `GET_DEADLINE 0x72` + prioridade `RED/BLUE/GREEN`
(`PRIORITY_SET/GET 0x73/0x74`, `src/context.rs`). Scheduler: 3 FIFOs, preempção
estrita `Red > Blue > Green`; `Red` = interrupto/`ABORT`/`SENSE`,
`Blue` = áudio, `Green` = inferência pesada. `CTX_SWITCH` impõe fence +
reavaliação. `BARRIER` com timeout é (C2).

**(C2.)** Composição `AND/OR/N_OF_M/CHAIN` (`0xD0–0xD3` na numeração do rascunho
original; **sem codificação congelada** — requer RFC pela regra §6.4/§19),
EDF com herança de deadline cross-contexto/dispositivo, afinidade de
dispositivo/NUMA e backpressure de `CLUSTER_STAGING` a 80%. Descrito como
direção normativa (`ESPEC §7`, `ESPEC-V2 §§7/12`), não como medição.

### 3.5 Scheduler (C1 + C2)

(C1) acima. (C2) `prio_eff = (deadline, localidade, base)`, herança com
detecção de ciclo e `ABORT` anti-deadlock, placement por heartbeat
(`free_pct` + filas) — alvos, não medições.

### 3.6 Transferência de estado entre modelos (C2, com 1 perna habilitada)

`KV_TRANSFER` com regressão ridge por cabeça **não está implementado** e não
é reivindicado como pronto. O que está habilitado e sustenta a direção:
`GATHER` (reindexação), `DISTANCE` top-k fundido alimentando `ATTN` podada
(`programs/attn_topk_demo.m3asm`, T4), `RANK1_UPDATE` CoW e `KV_TRUNCATE`
higiene rolante — i.e., os primitivos de reindexação/poda/rollback sobre os
quais a transferência futura se apoia. A transferência plena requer RFC com
prova de inadequação de lowering (`ESPEC §13`).

---

## 4. BREVE DESCRIÇÃO DOS DESENHOS

- **Figura 1** — Blocos da VM: fetch 32B, motores sobre 4 regiões (C1) e 16 lógicas (C2 tracejado), fence `CTX_SWITCH`.
- **Figura 2** — `FORK` (empilha CoW) → mutação → `ABORT` (troca de ponteiros); I-Mono; I-Persist como guarda futura.
- **Figura 3** — Deadline absoluto (C1, sólido) e árvore `AND/OR/N_OF_M/CHAIN` (C2, tracejado).
- **Figura 4** — Filas `Red>Blue>Green` (C1, sólido) e EDF + afinidade + herança (C2, tracejado).
- **Figura 5** — Pipeline voz→RAG→decisão→voz com opcodes IMPL sólidos e RSVD tracejados.
- Detalhes executáveis em `04_DESENHOS.md`; traço final por desenhista técnico.

---

## 5. DESCRIÇÃO DETALHADA DA INVENÇÃO

### 5.1 Arquitetura geral (C1)

VM em processo único sobre CPU/GPU/NPU convencional. Prova em
`src/memory.rs:1-80` (4 regiões), `src/context.rs:1-80` (16 regs `u128`,
`pc`, `root_version`, `priority`, `cmp_equal`, `interrupt_flag`, `pipeline`,
`deadline`), `src/bus.rs` (3 barramentos Tokio: `watch` interrupto,
`broadcast` stream/scheduler — µs medidos, não silício), `src/reactor.rs`
(`tokio::select!` + checagem inline por chunk), `src/inference.rs`
(GGUF→`forward_one`: RMSNorm→Q/K/V `matvec`→RoPE→KV append→softmax por
cabeça→proj O→residual→FFN gate/up/down, 22–28 camadas; `matvec_q4k` AVX2
fundido antes de `faer`; `embedding_row` sem materializar 934 MB),
`src/asm_emitter.rs` (baixa `ModelConfig` para `.m3asm` desenrolado).

**Motores sobre a mesma memória (comutação sem troca de contexto):**
Transformer (`ATTN/FFN/NORM/ROPE/SAMPLE/MATVEC/MUL/SILU`), SSM
(`SSM_SCAN/RESET`, `src/ssm.rs`: `h←h·exp(dt·A)+x·B·dt`, `y=⟨h,C⟩+D·x`),
Áudio (`CODEC_ENC/DEC` 1920 f32⇄16 códigos, `AUDIO_ALIGN`, `src/mimi.rs`),
Clássico/retrieval parcial (`GATHER/DISTANCE/RANK1_UPDATE` + `SAMPLE TOPK`,
RFC-0004), `CTX_SWITCH` (`payload[0]=pipe`, `[1]=0xA5`) como fence +
snapshot + `maybe_preempt()`.

Medido (`ESPEC §17`, host do autor Ryzen 3500U salvo nota): ATTN 32×32 ~5 ms;
sparse 5% + `NOTIFY_EACH_HEAD` ~5 ms + overhead <3×; `sparse_nop` 45 ms debug;
IPS 240k debug / 2,1M release; `FORK` 100×1MiB <1 ms; `ABORT` 32×32 ~25 ms
debug (publish <1 ms; 50 µs HW é hipótese); CSR 64×64@5% ~10× menor;
DeepSeek-1.5B `Q4_K` ~1,2 s/tok; TinyLlama-1.1B `Q4_K` ~0,33 s/tok (+10% com
`znver1`). `wgpu` só `ATTN≤64` — ganho ~0% hoje, ~1,5× est. com `GEMV.wgsl`
residente. Nada além disso é afirmado como desempenho.

### 5.2 Formato de instrução (C1 normativo; C2 proposto)

**(C1)** 32 bytes: `opcode | flags | rdest/rsrc1/rsrc2/rsrc3 | payload 26B LE`
(`src/opcodes.rs:26`, `ESPEC §4`). Regras: rejeita <32B; regs `0..15` ou
`0xFF`; LE; encodings imutáveis; aliases nunca codificados; assembler 2-pass.
Rascunho de 64 bits (`docs/arq/chat.md`) **superado — não implementar**.

**(C2)** 64B: `opcode/flags canônicos | rdest/rsrc1-5 | LAMPORT u64 |
DEADLINE u64 ns | PAYLOAD_EXT 8B | PAYLOAD_CORE 32B` (`ESPEC-V2 §4.2`);
função de tamanho com exceção única `0xFF=32B`; decoder v1.x rejeita
`≥0x80` com `UnsupportedWidth`. Flags canônicos `PRECISION/FUSED_ACT/
INPLACE/TRANSPOSE/PRIORITY/ATOMIC/PERSIST`; `PRECISION` é roteamento
consultivo com trap (`UnsupportedPrecision`), **proibido fallback silencioso**
(`ESPEC-V2 §11`).

### 5.3 Rollback conjunto (C1)

#### 5.3.1 Snapshot (`FORK 0x04`)

Empilha: heaps GLOBAL/esparso (clone `Arc`), camadas KV + heap de bytes KV,
`ssm_states`, `rank1_layers`. Custo O(1) em ponteiros (não-cópia física até
write). Janela deslizante `DEFAULT_SNAPSHOT_WINDOW=16` (RFC-0003,
`0`=opt-out explícito ilimitado), despejo oldest-first, `Err` limpo em restore
de evictado. Ex.: `FORK r12, GREEN, NOTIFY` publica `SchedSignal`.

#### 5.3.2 Restore (`ABORT 0x05` + `SSM_RESET` + `KV_TRUNCATE`)

Troca raízes/registradores-mapa; `ssm_states`/`rank1_layers` pop; KV trunca.
Teoremas condicionais (`ESPEC §8`): **T1** rollback exato
(`restore(snapshot(s))=s`, teste exigido: 100 passos, `ABORT` no 50, +50 =
igual bit-exato); **T2** preempção sem termo em N ou nº de parâmetros
(`L_abort=t_detect+t_sched+t_restore`; valores absolutos são medição §17,
nunca parte do teorema). Provas: `formal/Formal/Rollback.lean`
(`restoreFix`, `fresh_of_inv`, contraexemplo `clobber_demo` da variante
antiga). Limite declarado: registradores de contexto não participam do
rollback (só memória + mapas de handles) — pré-existente, documentado em
RFC-0004 follow-up.

### 5.4 Deadlines (C1 + C2)

**(C1)** `SET_DEADLINE/GET_DEADLINE`, `PRIORITY_SET/GET`, `YIELD`,
`PREEMPT_CHECK`, `LOCK/UNLOCK/FENCE`, `TRACE_EVENT/CYCLES_COUNT`,
`SANITY_CHECK/ASSERT/DUMP` (RFC-0006). Semântica: deadline absoluto + 3 FIFOs.

**(C2)** Árvore binária de deadlines (folhas = absolutos ns; internos =
`AND/OR/N_OF_M/CHAIN`), propagação com rebaixamento/relaxamento/migração,
herança cross-device com corte ≤16 hops. Requer RFC + testes + delta Lean
antes de `DRAFT→IMPL` (plano de ondas `ESPEC-V2 §14`).

### 5.5 Scheduler (C1 + C2)

(C1) §5.4. (C2) EDF + `device_affinity` + `numa_affinity` + `priority_base`;
`device_locality` 0/1/2/∞; herança e anti-deadlock; placement por heartbeat;
timeouts obrigatórios (`connect 2s, ack 500ms, heartbeat 200ms, suspect
600ms, dead 2s`), urgente passa bulk, `BARRIER` com NACK+timeout. Tudo (C2).

### 5.6 Transferência de estado (C2 com base C1)

Fórmula de referência (não implementada; habilitação requer calibração de
~500×1024 + RFC + prova T4):
`K_tgt[l,h]=K_src[top_k,h]·W_K[l,h]+b_K[l,h]`,
`V_tgt[l,h]=V_src[top_k,h]·W_V[l,h]+b_V[l,h]`.
Integração com `ABORT`: estado transferido é descartável (não faz parte do
snapshot até commit). Base habilitada: `DISTANCE→GATHER→ATTN` podada,
`RANK1_UPDATE` CoW, `KV_TRUNCATE`.

### 5.7 Exemplo de operação (honesto: mescla C1 executável + C2 como spec)

Cenário: atendimento por voz (Moshi temporal + BGE-M3 emb + XGBoost via
`FOREST` (C2) + Mamba draft + Llama verify + RAG (C2)). O programa
`docs/arq/CENARIO-MULTIMODEL.md` **não compila hoje** (sintaxe além do
assembler: `DIV/IF_LESS/CALL/RET/.param/LOADSTR/STREAM_WRITE/PARALLEL_*`,
`COMPARE` com float, `ADD` sobre contadores) — é material de entrada
arquivado, não normativo. Núcleo executável equivalente (C1): laço
`SENSE TOKEN / FORK / NORM / ATTN / ADD / FFN / ADD / SAMPLE / STREAM /
COMPARE / IF_EQUAL / JUMP` (`AsmEmitter`), barge-in
`SENSE USER_INPUT → IF_INTERRUPT → ABORT (+pop SSM / KV_TRUNCATE /
SSM_RESET) → resume` (`programs/mamba_scan_demo.m3asm`,
`programs/codec_loop.m3asm`, `programs/moshi_loop_v2.m3asm`,
`programs/thinking_sample.m3asm`). Deadline total 500 ms e latências por
fase no cenário são **alvos de engenharia**, não medições.

Pipeline de referência (sólido=C1, tracejado=C2 — ver Fig. 5):

```text
SENSE PCM → CODEC_ENC → [STREAM_MERGE] → [DEPFORMER]
  → EMBED(BGE, via EMBED/GATHER) → [RAG_SEARCH] → CONCAT(C2)
  → [FOREST](XGB) → FAST(Mamba SSM_SCAN) ou DEEP(draft+verify, C2)
  → [DEPFORMER] → CODEC_DEC → STREAM
```

---

## 6. REIVINDICAÇÕES (resumo legível; texto jurídico em 02_QUADRO_REIVINDICATORIO.md)

1. Método de orquestração multi-modelo em VM de contexto único com ISA nativo,
   snapshot CoW e restore atômico, deadline absoluto e scheduler por
   prioridade (C1).
2. CoW por `Arc` + janela `k=16` + I-Mono.
3. Motores Transformer/SSM/áudio com `CTX_SWITCH` sem troca de contexto.
4. `GATHER/DISTANCE/RANK1_UPDATE` + `SAMPLE TOPK` para MoE/RAG/recorrência matricial.
5. `KV_TRUNCATE` + `SSM_RESET` como higiene/rollback.
6. RNG por contexto + telemetria + traps determinísticos.
7. Formato 32B + assembler 2-pass + encodings imutáveis.
8. (C2) Deadlines compostos `AND/OR/N_OF_M/CHAIN` + EDF + herança.
9. (C2) Afinidade de dispositivo/NUMA + placement + backpressure.
10. (C2) Transferência nativa de estado (`KV_TRANSFER`-tipo) com descarte em `ABORT`.
11. Sistema (produto) correspondente a 1–7 (C1).
12. (C2) Descoberta de hardware + registro de dispositivos.
13. (C2) Cluster transacional com WAL + retenção pós-ACK.
14. (C2) Segurança (cookie + handshake; mTLS futuro) — sem afirmar ChaCha/OTLP prontos.
15. (C2) Observabilidade (`TRACE_EVENT` local hoje; W3C/OTLP futuro).

---

## 7. RESUMO

Ver `03_RESUMO_INPI.md` (≤150 palavras). Essência: VM de 32B em processo único
com 4 regiões tipadas e ISA nativa (Transformer, SSM, áudio, retrieval e
recorrência matricial), snapshot CoW e restore atômico transversal
(`FORK/ABORT`), deadline absoluto e prioridade estrita, com extensão prevista
a deadlines compostos, scheduler topológico e transferência de estado —
permitindo pipelines multi-modelo (voz+RAG+decisão+voz) sem serialização
entre processos, com interrupção e retomada bit-exata do prefixo confirmado.

---

## 8. NOTAS PARA O DEPÓSITO

### 8.1 Documentos a complementar

- Desenhos Fig. 1–5 por desenhista (`04_DESENHOS.md` como briefing).
- Quadro reivindicatório final por agente de PI (`02_QUADRO_REIVINDICATORIO.md` como minuta).
- Resumo INPI (`03_RESUMO_INPI.md`, conferir 150 palavras na versão protocolada).
- GRU + dados de titularidade. Sem lista de sequências.

### 8.2 Estratégia de depósito (recomendada)

1. **Principal (C1):** reivindicações 1–7 + 11 (habilitadas). Exame em até 36 meses.
2. **Divisionais (C2):** (i) ISA 64B + ESCAPE + tabela de regiões; (ii) CoW/I-Persist/endurecido;
   (iii) scheduler EDF + deadlines compostos + afinidade; (iv) cluster WAL + formas-X + SWIM.
   Cada divisional só após RFC + testes verdes + delta Lean (regra `ESPEC-V2 §14`).
3. Sigilo 18 meses padrão; avaliar trilha de exame prioritário se houver implementação acelerada.
4. **Não afirmar** ChaCha20-Poly1305, OTLP/W3C, NVLink/PCIe P2P, `FOREST/SPIKE/DEPFORMER/RAG`
   prontos — estão como (C2) com erro explícito hoje; afirmação contrária é prova contra o depositante.

### 8.3 Custos estimados (ordem de grandeza, confirmar com agente)

| Item | Estimativa (R$) |
|:---|:---|
| Depósito PF | ~260 |
| Redação agente PI | 8.000–15.000 |
| Desenhos | 1.500–3.000 |
| Anuidades 20 anos | 20.000–30.000 |
| **Total** | **~30.000–48.000** |

### 8.4 Alternativa estratégica (mantida do rascunho, com ressalva)

Zenodo + DOI + AGPL protege anterioridade e impõe copyleft (inclusive §13 p/ rede),
mas **publicar antes de depositar destrói novidade**. Só publique após depósito ou
sob orientação expressa do agente. Avaliar tração 6–12 meses **após** depósito, não antes.

---

*Fim da minuta. Prevalecem `docs/ESPEC.md` (normativo) e código-fonte em caso de divergência.*
