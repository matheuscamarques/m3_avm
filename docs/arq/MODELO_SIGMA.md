# M³-AVM-Σ — Modelo Matemático (base teórica)

> **Status:** especificação verificável, não marketing.
> Fonte de verdade para alegações de correção, tempo-real e erro.
> Provas machine-checked em `formal/`; provas estilo paper aqui com hipótese explícita.
> Convenção de honestidade do `README.md` §5 e `ARCHITECTURE.md` vale para este doc:
> número sem bench citado é meta, não medição.

## 0. O que este modelo prova (e o que não prova)

Prova, condicionado aos invariantes I-Persist, I-Mono, I-WAL, I-Deadline:

1. Rollback bit-exato sem clobber (T1).
2. Latência de preempção sem termo em N nem |modelo| (T2).
3. Erro híbrido Mamba+quant limitado por janela, independente de N (T3).
4. Atenção top-k com erro limitado sob decaimento rápido (T4).
5. Migração sem perda sob falha fail-stop de 1 nó (T5).
6. Estabilidade tempo-real do áudio (T6).

Não prova: perplexidade/qualidade (exige bench com pesos reais),
throughput absoluto (exige `cargo bench` no host), segurança bizantina,
sobrevivência a 50 anos (tese de design, não teorema).

Mapa de código: `src/memory.rs` (snapshot/restore), `src/ssm.rs` (Euler),
`src/quant.rs` + `src/matvec_quant.rs` (Q4_K), `src/sparse.rs` (CSR),
`src/vm.rs` + `src/bus.rs` + `src/reactor.rs` (preempção),
`PLANO_AVM_CLUSTER.md` (rede), `PLANO_ISA_UNIVERSAL.md` (opcodes 0x1E–0x25).

## 1. Definições

**D1 (Estado).** `Σ = (M, H_ssm, K_kv, H_rank, V_snn, C, t_L)` onde:

- `M`: store versionado `(counter, snaps, cur)`.
- `H_ssm`: família de estados Mamba por layer (`I·S` escalares).
- `K_kv`: KV-cache por layer (`2·N·d` no pior caso denso).
- `H_rank`: matrizes `H_t [d,k]` (DeltaNet/Titans, `OP_RANK1_UPDATE 0x24`).
- `V_snn`: potenciais de membrana (SNN, `OP_SPIKE_STEP 0x20`).
- `C`: contextos `(regs, pc, pipeline, deadline d, prioridade)`.
- `t_L`: relógio lógico de Lamport (ordenação). Wall-clock só para métrica.

Instrução real: 32 bytes
`[op 1B | flags 1B | rdest | rsrc1 | rsrc2 | rsrc3 | payload 26B]`
(`src/opcodes.rs`, `docs/ISA.md`). O formato 64-bit do `docs/chat.md`
é visão futura, não o implementado.

**D2 (Passo).** Transição `Σ -[op]-> Σ'` determinística dado `(op, seed RNG)`.
Sampling estocástico (`DENOISE_STEP σ>0`) é determinístico sob mesma seed
(replay-exato). Sem seed fixada, determinismo não é alegado.

**D3 (Falha).** Modelo fail-stop de 1 nó: no máximo 1 nó do cluster
morre por episódio de migração; rede pode atrasar/perder (timeout cobre),
mas não corrompe (futura autenticação via cookie; TLS = follow-up).

## 2. Invariantes (hipóteses load-bearing)

- **I-Persist:** nenhum kernel muta estado alcançável por snapshot ativo.
  Implementação exigida: disciplina imutável / CoW estrito
  (`im::HashMap` ou clone-antes-de-escrever). O `Arc::clone` atual
  **não** garante isso sozinho — ver `MATH_MVP.md §3(ii)`.
- **I-Mono:** contador de versão estritamente monotônico; `restore` nunca
  rebaixa o contador. O código atual viola (`version := v`,
  `src/memory.rs:1027`, contraexemplo `formal/Formal/Rollback.lean:clobber_demo`).
  Fix exigido: `restoreFix` (preserva contador).
- **I-WAL:** todo tensor em `MOVE`/migração tem cópia retida na origem
  até ACK do destino + janela de retenção. `MOVE=COPY+invalidate` atual
  viola (ver T5).
- **I-Deadline:** todo contexto bloqueável (`BARRIER`, `SEND_TENSOR`,
  `SPAWN`) declara prazo absoluto + timeout; scheduler é EDF com herança:
  ao bloquear em produtor remoto, o consumidor herda o prazo mais restrito.
  `Red remoto nunca preempta Red local` sem herança viola (ver T2).

Sem I-Persist+I-Mono, T1 é falso. Sem I-WAL, T5 é falso. Sem I-Deadline,
T2 distribuído é falso. São as três correções Σ obrigatórias.

## 3. Teoremas

### T1 — Rollback exato (fortalece MATH_MVP Teo. 4)

**Enunciado.** Sob I-Persist + I-Mono, se `v = snapshot(Σ₀)` então
`restore(v) = Σ₀` bit-exato, e ids sucessivos são estritamente crescentes
(clobber impossível).

**Demonstração.** Machine-checked em `formal/Formal/Rollback.lean`:
`snapshot_restore_roundtrip`, `restoreFix_roundtrip`, `fresh_of_inv`,
`snapshot_counter_succ`. O arquivo também prova que a variante atual
sem o fix perde dados (`clobber_demo`, `clobber_loses`).
**Obrigação de código:** aplicar `restoreFix` em `src/memory.rs:1027`
e impor I-Persist em `vm.rs` (nenhum `&mut` em tensor snapshotado).
**Teste exigido:** rollout 100 passos, `ABORT` no 50, +50 passos =
bit-exato vs rollout sem abort (PAMT torna trivial; `Arc` mutável falha).

### T2 — Latência de preempção sem termo em N

**Enunciado.** `L_abort = t_detect + t_sched + t_restore` onde nenhum termo
é proporcional a N nem ao nº de parâmetros.

**Demonstração.** `t_detect` = 1 `load` atômico por chunk + `watch::broadcast`
(µs, `src/bus.rs`). `t_sched` = remoção O(1) de fila + comparação de prazo.
`t_restore` = troca de raízes + ponteiros (`O(|regs|+|roots|)`), não
`O(N·d)`. Logo a forma é estrutural, não empírica. Valores absolutos
(`25ms` debug 32×32, `publish <1ms`, metas `~217µs/39µs`) são medição,
não parte do teorema — ver `README.md §5`.
**Distribuído:** soma-se 1 RTT LAN + enfileiramento urgente-fura-bulk.
Requer I-Deadline (herança): sem ela, espera circular `BARRIER` × `RED`
local trava (fratura `chat.md #3`). Com timeout sempre + herança, espera
é limitada pelo prazo máximo, nunca circular.
**Obrigação Lean:** `formal/Formal/ClusterWAL.lean` (segurança) +
deadline como `Nat` monotônico (liveness por timeout, trivial por construção).

### T3 — Erro híbrido limitado por janela (resposta à deriva `chat.md #5`)

**Lemas base (já provados).**
(a) Passo SSM Euler vs ZOH: erro local `≤ |xB|·|A|·dt²` sob `|A|·dt ≤ 1`
(`formal/Formal/SSM.lean:euler_local_error`, `MATH_MVP.md Teo. 1`).
Contração `a_d = exp(dt·A) ∈ (0,1)` e esquecimento `|hₙ| ≤ aⁿ|h₀|`
(`ssm_contraction`, `ssm_forget`).
(b) Matvec Q4_K: `|⟨e,x⟩| ≤ (s_max/2)·‖x‖₁ ≤ (s_max/2)·√n·‖x‖₂`
(`MATH_MVP.md Teo. 5`).

**Enunciado.** Seja `e = C1·dt² + C2·s_max·√n·‖x‖₂` o erro por passo
(Euler + quant). Sem recalibração, após N passos o erro é `O(N·e)` —
diverge, confirmando a crítica. Com fence `CTX_SWITCH` + recalibração
Transformer a cada `W` passos (residual substitui/pondera o estado),
o erro global é `≤ r + W·e`, onde `r` é o erro do recalibrador,
**independente de N**. A contração `a_d < 1` impede amplificação
exponencial dentro da janela.

**Demonstração (esboço).** Por indução sobre janelas: dentro da janela,
telescópio linear com fator `≤1` acumula no máximo `W·e`; no fence,
o estado é reprojetado com erro `≤ r`, descartando o acúmulo anterior.
Formalização esqueleto em `formal/Formal/WindowedError.lean`
(lemas aritméticos provados; instanciação com `e` de (a)+(b) como obrigação).
**Consequência prática:** `W=80ms` de áudio (1 janela Mimi) ou `W=5` tokens
(checkpoint thinking) dá bound testável por bench. Filtro de Kalman do
`chat.md` é otimização futura de fusão, não premissa do bound.

### T4 — Atenção top-k com erro limitado (resposta a `chat.md #2`)

**Enunciado.** Seja `p` o softmax denso e `p̂` o softmax restrito aos
top-T scores, com massa de cauda `τ = 1 − Σ_{top-T} exp(sᵢ)/Z < ε`.
Então `‖p − p̂‖₁ ≤ 2ε/(1−ε)`.

**Demonstração (esboço).** Renormalização: `p̂ᵢ = pᵢ/(1−τ)` no top-T,
`0` fora. Distância L1 = `τ + (1−(1−τ)⁻¹)(1−τ) ≤ 2τ/(1−τ)`.
Esqueleto em `formal/Formal/TopK.lean` (definições + enunciado, prova
completa como obrigação; caso uniforme `τ≈1−T/N` mostra quando **não**
há ganho — honestidade da Prop. 7 mantida).
**Custo:** `O(N·d + N log T + T²·d)` vs `O(N²·d)` denso; vale sse `T≪N`
**e** `τ<ε` (decaimento rápido de QK). Caso contrário, poda é decorativa.
**Obrigação de código:** `OP_DISTANCE 0x23` top-k fundido alimenta
`OP_ATTN` podado (H1 antes de G4), nunca o inverso.

### T5 — Migração conserva estado (resposta a `chat.md #4`)

**Enunciado.** Sob I-WAL + fail-stop de 1 nó, após `OP_MIGRATE`
o tensor existe em `≥1` nó vivo (origem ou destino), nunca zero.

**Demonstração.** Por casos, machine-checked esqueleto em
`formal/Formal/ClusterWAL.lean`: origem loga no WAL → envia → espera ACK
→ marca coletável mas retém por janela. Se origem morre antes do ACK,
destino ainda não confirmou e origem retém (ou destino já tem cópia).
Se destino morre, origem retém e reenvia. Perda só com morte dupla
simultânea (fora do modelo; exigiria replicação 3-way).
O `MOVE=COPY+invalidate pós-ACK sem retenção` atual perde no caso
morre-origem-entre-invalidate-e-ACK — contraexemplo explícito no doc.
**Obrigação de código:** reescrever F4 do cluster como `OP_MIGRATE`+WAL.

### T6 — Tempo-real do áudio

**Enunciado.** Se `T_proc < 80ms` por janela de 1920 amostras @24kHz,
backlog `≤1` janela e atraso de barge-in `≤ 80ms + T_proc`,
independente do comprimento da conversa (`MATH_MVP.md Teo. 6`, prova
por estabilidade de fila determinística). No cluster, o mesmo bound
com `T_proc` incluindo 1 RTT sob I-Deadline.

## 4. Regra de evolução (tese §5 de PLANO_ISA_UNIVERSAL)

Todo opcode stateful novo declara (a) onde mora o estado, (b) custo de
snapshot, (c) como `ABORT` restaura. Sem isso, não entra na ISA.
Aliases resolvem no assembler, nunca ganham encoding (`docs/ISA.md §4`).

## 5. Obrigações empíricas (não substituíveis por prova)

- Bench por janela 80ms (`CODEC_ENC→SSM_SCAN→AUDIO_ALIGN`) com `M3_PROFILE=1`.
- Paridade 1 frame vs referência PyTorch `max_diff <1e-3` (F3 Moshi),
  depois invariância de rollout 100/50/50 bit-exata (T1+T3).
- `cargo test --lib` verde (170 atuais + novos `windowed/topk/wal`).
- `lake build` verde em `formal/` (SSM+Rollback provados; 3 novos como
  esqueleto com `sorry` marcado até fechamento).
- Nenhum número novo entra em `README.md §5` sem comando reprodutível.

## 6. Correspondência Lean

| Teorema | Arquivo | Status |
|---|---|---|
| T1 (rollback + fix) | `formal/Formal/Rollback.lean` | provado, inclui contraexemplo do bug atual |
| T3 lemas SSM | `formal/Formal/SSM.lean` | provado |
| T3 janela | `formal/Formal/WindowedError.lean` | esqueleto: aritmética provada, instanciação `sorry` |
| T4 poda | `formal/Formal/TopK.lean` | esqueleto: definições + enunciado, prova `sorry` |
| T2/T5 cluster | `formal/Formal/ClusterWAL.lean` | esqueleto: segurança por casos, liveness por timeout |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
