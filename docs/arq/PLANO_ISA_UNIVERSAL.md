# PLANO — ISA Universal de IA: `0x1E–0x21` (CONV / GATHER / SPIKE / DENOISE)

**Objetivo:** cobrir as famílias de arquiteturas que a M³-AVM ainda não executa nativamente (CNNs/visão, GNNs/grafos, SNNs/spikes, difusão, MoE, RNNs clássicas/RWKV, **memória viva tipo Titans, recorrência matricial tipo DeltaNet, ODEs/liquid, e ML clássico**) com **8 opcodes novos no total (onda 1: `0x1E–0x21`, onda 2: `0x22–0x25`)**, sem inflar a ISA: tudo o que pode ser *lowering* de opcodes existentes continua sendo lowering.
**Data:** 2026-09-09 · Doc canônico desta frente (espelha `docs/ISA_OPCODES_0x13_0x19.md` e `docs/PLANO_AVM_CLUSTER.md`).

> Tese: toda arquitetura de IA resume-se a 3 primitivas — (1) GEMM/tensor-contraction, (2) atualização de estado temporizada `S_t = α·S_{t-1} + β·f(X_t)`, (3) elementwise/não-linearidades. A M³-AVM já tem (1) via `MATVEC/MUL` e (3) via `SILU/NORM/SAMPLE`, e (2) via `SSM_SCAN/KV_CACHE`. Os 4 opcodes aqui cobrem **só o que não é expressível com custo aceitável** sobre essa base.

## 0. Contexto confirmado (lido no repo)

* ISA ocupa `0x00 HALT, 0x01..0x19` (25 opcodes) + `0xFF NOP` — `src/opcodes.rs:28-57`. Faixa de rede `0x1A..0x1D` reservada (`docs/PLANO_AVM_CLUSTER.md` §2). **`0x1E–0x21` livres, `0x22–0xFE` seguem livres.**
* Convenção vigente: `SSM_SCAN` com flags `CONV/GATE` é **rejeitado com erro explícito** — conv via `MATVEC`, gate via `SILU+MUL` (`src/vm.rs:1869`, `docs/ISA_OPCODES_0x13_0x19.md:16`). Este plano mantém a filosofia: nativo só onde lowering é inadequado.
* Formato fixo 32B: `opcode 1B + flags 1B + rdest/rsrc1-3 + payload 26B` — `src/opcodes.rs:2-10`.
* Estado recorrente já tem casa: `KV_CACHE 0x30` (Transformers), `Vm::ssm_states` (Mamba, com `FORK` empilha/`ABORT` desempilha snapshot). SNNs/MoE/difusão reusam o mesmo padrão CoW (ver §3).
* Não existe hoje: indexação indireta por tensor (`GATHER`), convolução deslizante, disparo por limiar com reset, top-k/MoE-dispatch, passo de difusão fundido. Busca em `src/` confirma: zero ocorrências de `CONV/GATHER/SPIKE/DENOISE/MOE` fora da flag reservada.

## 1. O que cada família precisa — e o veredito nativo vs. lowering

| Família | Estado na VM | Veredito |
|---|---|---|
| RNN/LSTM/GRU/RWKV/xLSTM | vetor célula `c_t` + oculto `h_t` fixos | **Lowering, sem opcode novo.** Passo RNN = `MATVEC+MUL+ADD` (+`SILU`); RWKV/`wkv` é scan linear → `SSM_SCAN` cobre. Falta só `sigmoid` — avaliar flag `SIGMOID` em `SILU` ou fundir no passo, não um `OP_RNN_STEP`. Reavaliar só se bench provar overhead de dispatch. |
| MoE (DeepSeek/Mixtral/GPT-4) | roteamento dinâmico token→especialistas | **Parcial.** Gate = `MATVEC` + top-k + dispatch. Top-k cabe como flag/modo de `SAMPLE` (hoje só amostra, `src/vm.rs:1270`); dispatch por índice **exige `OP_GATHER (0x1F)`**. Sem `OP_MOE_GATE` dedicado no MVP. |
| CNN/ConvNeXt/visão | feature maps em `GLOBAL` | **`OP_CONV (0x1E)` nativo.** Lowering via im2col+`MATVEC` explode memória (`K·K·C` por posição); deslizamento com stride/pad/dilation merece kernel próprio, mesma justificativa do `matvec_q4k` fundido. |
| GNNs (moléculas, KG) | adjacência + arestas | **`OP_GATHER (0x1F)` nativo.** Nenhum opcode hoje indexa por tensor; serve também a MoE-dispatch e embedding-bag. `SCATTER_ADD` como modo do mesmo opcode. |
| SNNs/spikes | potencial de membrana `V(t)` por neurônio | **`OP_SPIKE_STEP (0x20)` nativo.** Acumular+limiar+reset+refratário por neurônio exigiria `COMPARE`+escrita mascarada por elemento — não expressível (não há store mascarado). Estado em `TEMPORAL`/`GLOBAL` com snapshot CoW como `ssm_states`. |
| Difusão/EBMs (SD/Sora) | mapa de ruído + embedding de `t` | **`OP_DENOISE_STEP (0x21)` fundido, prioridade baixa.** É composição (`ADD+MUL+NORM` + schedule); nativo economiza tráfego de memória, mas lowering já funciona. Por último. |

Descartado do MVP: `OP_RNN_STEP` e `OP_MOE_GATE` dedicados (cobertos por lowering + `GATHER`); `OP_CONV3D` separado (`OP_CONV` é N-dimensional por shape); `OP_STATE_UPDATE/RESET` como aliases (já existem como `SSM_SCAN 0x13`/`SSM_RESET 0x14` — documentar a generalização em vez de duplicar).

## 2. Mapa congelado — 4 opcodes `0x1E–0x21`

```rust
pub const OP_CONV:         u8 = 0x1E; // convolução N-dim com stride/pad/dilation/groups
pub const OP_GATHER:       u8 = 0x1F; // gather/scatter por índice (grafos, MoE, embedding-bag)
pub const OP_SPIKE_STEP:   u8 = 0x20; // integrate-and-fire discreto (SNNs)
pub const OP_DENOISE_STEP: u8 = 0x21; // passo de difusão fundido x_t -> x_{t-1}
// Onda 2 — arquiteturas emergentes + ML clássico (§8)
pub const OP_FOREST:       u8 = 0x22; // ensemble de árvores (XGBoost/RF) vetorizado
pub const OP_DISTANCE:     u8 = 0x23; // distância/similaridade em lote + top-k (kNN, RAG)
pub const OP_RANK1_UPDATE: u8 = 0x24; // H = α·H + β·v⊗k (DeltaNet, HGRN, fast-weights/Titans)
pub const OP_ODE_STEP:     u8 = 0x25; // passo ODE fundido Euler/RK (Liquid/NCPS)
```

Mapa completo pós-frentas: `0x00 HALT · 0x01–0x0F base+thinking · 0x10–0x12 MATVEC/MUL/SILU · 0x13–0x19 SSM/codec/áudio · 0x1A–0x1D cluster · 0x1E–0x25 universal · 0x26–0xFE livres · 0xFF NOP`.

### 2.1 Codificação nos 26B de payload

* **`OP_CONV (0x1E)`** — `rdest`=out `[N,Cout,...]`, `rsrc1`=input, `rsrc2`=kernel `[Cout,Cin,K..]`, `rsrc3`=bias ou `0xFF`. `payload[0..2]`=`stride u16` (por eixo via repetição/broadcast), `[2..4]`=`pad u16`, `[4..6]`=`dilation u16`, `[6]`=`groups u8`, `[7]`=`fused_act (0=none,1=silu,2=relu)`. Dims espaciais derivadas dos shapes (sem novo descritor).
* **`OP_GATHER (0x1F)`** — `rdest`=out, `rsrc1`=tabela, `rsrc2`=índices `u32/i64`, `rsrc3`=`0xFF` (gather) ou acumulador (scatter). `payload[0]`=`axis u8`, `[1]`=`mode (0=GATHER,1=SCATTER_ADD,2=SCATTER_MAX)`, `[2..4]`=`nnz_hint u16` (pré-alocação esparsa). Bounds-check OOB → erro explícito (determinismo, nunca wrap).
* **`OP_SPIKE_STEP (0x20)`** — `rdest`=spikes `[1,n] u8/binário`, `rsrc1`=`V(t)` membrana (in-place), `rsrc2`=corrente de entrada, `rsrc3`=pesos/decay-pack ou `0xFF`=defaults. `payload[0..4]`=`V_threshold f32`, `[4..8]`=`decay τ f32`, `[8..12]`=`V_reset f32`, `[12]`=`layer_id`, `[13]`=`refractory_steps u8`. `FORK` empilha `V`, `ABORT` restaura (mesmo protocolo de `ssm_states`).
* **`OP_DENOISE_STEP (0x21)`** — `rdest`=`x_{t-1}`, `rsrc1`=`x_t`, `rsrc2`=ruído previsto `ε`, `rsrc3`=schedule/`t-embed` ou `0xFF`. `payload[0..4]`=`alpha_bar_t f32`, `[4..8]`=`beta_t f32`, `[8..12]`=`sigma_t f32` (0=determinístico/DDIM), `[12..16]`=`timestep u32`. Sampling estocástico via RNG da VM (semeado → determinístico/replay).

Sintaxe assembler:

```asm
CONV r4, r0, r1, r2 STRIDE=1 PAD=1 DILATION=1 GROUPS=1 ACT=SILU
GATHER r5, r0, r1 AXIS=0 MODE=GATHER
SCATTER_ADD r5, r0, r1 AXIS=0          ; alias de GATHER MODE=SCATTER_ADD
SPIKE_STEP r3, r1, r2 THRESH=1.0 DECAY=0.9 RESET=0.0 LAYER=0 REFRACT=2
DENOISE_STEP r4, r0, r1, r2 ALPHA=0.98 BETA=0.02 SIGMA=0.0 T=500
```

## 3. Taxonomia de estado (o "meio-termo" operacional)

```
FAMILY 1 STATELESS (preempção = drop instantâneo, sem snapshot)
  FFN · CNN (`OP_CONV`) · GATHER agregado · MoE-gate · ATTN sem cache
  → `ABORT` só descarta o chunk; nada a restaurar.

FAMILY 2 STATEFUL (preempção = ABORT + rollback de ponteiro O(1), padrão CoW)
  Transformer KV_CACHE 0x30 · Mamba ssm_states · SNN V(t) · célula RNN · difusão x_t
  → `FORK` empilha snapshot, `ABORT` desempilha; `OP_SIGNAL (0x1B)` remoto atinge
    todos pelo mesmo caminho da flag atômica (cf. PLANO_AVM_CLUSTER §3).
```

Regra: todo opcode stateful novo **deve** declarar (a) onde mora o estado, (b) custo de snapshot, (c) como `ABORT` o restaura. Sem isso, não entra na ISA.

## 4. Fases

### G0 — Spec (este doc) + reserva (0.5 sem)
- [x] Congelar §2/§2.1. Sem código.
- Aceite: revisão do payload (cabe em 26B? modos cobrem gather+scatter?) antes de codar.

### G1 — `OP_GATHER (0x1F)` (1 sem, maior valor/desbloqueio MoE+GNN)
- [ ] `src/opcodes.rs`: const, `instr_gather()`, `gather_params()/set_*`, `mnemonic()`, `parse_line()` (`GATHER/SCATTER_ADD`, `AXIS/MODE`).
- [ ] `src/vm.rs`: `exec_gather` denso (`ndarray` take/put) + esparso (reuso `SparseTensor` CSR); OOB = erro.
- [ ] Top-k p/ MoE como modo de `SAMPLE` (`TOPK=k`, escreve índices) — sem opcode novo.
- [ ] Testes: roundtrip, `gather/scatter_add` goldens, OOB determinístico; demo `programs/gather_moe_demo.m3asm` (gate `MATVEC→SAMPLE TOPK→GATHER` dispatch p/ 2 FFNs).
- Aceite: `cargo test --lib gather::` verde; dispatch MoE 2-expert bit-exato vs referência `ndarray`.

### G2 — `OP_SPIKE_STEP (0x20)` (1 sem)
- [ ] Novo `src/snn.rs`: `SnnState{V: Vec<f32>, refractory: Vec<u8>}`, `integrate_fire()` + golden LIF.
- [ ] `exec_spike_step` com `V` in-place + snapshot CoW (`FORK`/`ABORT` como `ssm_states`); RNG semeado só se `SIGMA/refractory estocástico` futuro — MVP determinístico.
- [ ] `SENSE` spike-train (modo de codificação rate/latency sobre `TEMPORAL`) — avaliar, não obrigatório no MVP.
- [ ] Demo `programs/snn_demo.m3asm` (LIF 64 neurônios, 100 passos, raster de spikes via `STREAM`).
- Aceite: paridade vs `snn.rs` golden `<1e-6`; `ABORT` no passo 50 restaura `V` bit-exato.

### G3 — `OP_CONV (0x1E)` (1–2 sem, maior custo)
- [ ] `exec_conv` direto (sem im2col materializado no MVP: deslizamento + `dot` por janela; `groups/depthwise` desde o dia 1 — é o caso de Mamba-adjacent e MobileNet).
- [ ] `fused_act` SILU/ReLU inline (evita passe extra de memória).
- [ ] Demo `programs/conv_demo.m3asm` (ConvNeXt-tiny stem ou depthwise 3×3) + paridade vs `ndarray`/`faer` referência.
- Aceite: bit-exato vs lowering im2col+`MATVEC` em shapes de teste; bench documenta speedup (alvo ≥2× vs im2col por economia de memória).

### G4 — `OP_DENOISE_STEP (0x21)` (1 sem, menor prioridade)
- [ ] `exec_denoise` DDPM/DDIM (`sigma=0` determinístico primeiro); schedule via `rsrc3` ou payload escalar.
- [ ] Demo `programs/denoise_demo.m3asm` (20 passos sobre tensor 32×32, `STREAM` a cada 5).
- Aceite: `sigma=0` bit-exato vs lowering `ADD+MUL+NORM`; modo estocástico replay-exato com mesma seed.

### G5 — Endurecer + tese (1 sem)
- [ ] `docs/ISA_UNIVERSAL.md` (spec final estilo `ISA_OPCODES_0x13_0x19.md`) + benches `criterion` por opcode + fumaça com pesos reais onde houver (ConvNeXt-tiny / GNN pequeno).
- [ ] Reavaliar `OP_RNN_STEP`/`OP_MOE_GATE` fundidos **só** com números de G1–G3 na mão.
- Aceite: `cargo test --lib` verde (166 + novos); doc honesto sobre o que é lowering vs. nativo.

## 5. Ordem e não-objetivos

Ordem: `G0 → G1 → G2 → G3 → G4 → G5`, com onda 2 intercalada por valor: `H1` logo após `G1` (desbloqueia RAG + DeltaNet), `H2/H3` após `G3`.
Não fazer agora: autograd/treino completo (só TTT local via `MATVEC` transposto + `RANK1_UPDATE`), SVD denso nativo (lowering iterativo), `OP_RNN_STEP` / `OP_MOE_GATE` / `OP_HIERARCHICAL_REDUCE` / `OP_BACKPROP_LOCAL` dedicados (todos cobertos abaixo), `0x26+` (reservar).

## 6. Comandos úteis

```bash
cargo test --lib gather:: snn:: conv:: denoise:: -- --nocapture
cargo run --bin m3_avm -- run programs/gather_moe_demo.m3asm --max-steps 200
cargo run --bin m3_avm -- run programs/snn_demo.m3asm --max-steps 500
cargo run --bin m3_avm -- run programs/conv_demo.m3asm --max-steps 200
cargo bench --bench physics_bench -- --quick
```

## 7. Decisões pendentes (travar em G0)

1. Mapa `0x1E=CONV / 0x1F=GATHER / 0x20=SPIKE_STEP / 0x21=DENOISE_STEP` congelado? (Recomendado — §2.)
2. Top-k como modo de `SAMPLE` vs. opcode próprio? (Recomendado: modo, sem opcode novo.)
3. `sigmoid` para RNN clássica: flag em `SILU`, lowering (`ADD+MUL` sobre exp — sem `EXP` nativo hoje), ou fundido futuro? (Recomendado: decidir em G5 com bench; RWKV/Mamba não precisam.)
4. `SCATTER_MAX` entra no MVP de G1 ou só `GATHER`+`SCATTER_ADD`? (Recomendado: só os dois primeiros; MAX é follow-up.)

## 8. Onda 2 — arquiteturas emergentes + ML clássico

Mesmo filtro da onda 1: nativo só onde lowering é inadequado. Resultado: **4 opcodes novos (`0x22–0x25`)**, o resto é lowering documentado.

### 8.1 Vereditos por família

| Família | Estado na VM | Veredito |
|---|---|---|
| Titans/MIRAS (memória-MLP com TTT) | fast-weights `W_mem` + gradiente local | **Parcial, sem `OP_BACKPROP_LOCAL`.** Passo TTT = `MATVEC` (+ flag `TRANSPOSE`, nova, ver §8.3) + `MUL/ADD` + `OP_RANK1_UPDATE (0x24)` p/ escrita Hebbiana/delta. Backprop de MLP pequeno é composição de primitivas existentes — opcode dedicado seria treino disfarçado, fora do escopo (cf. §5). |
| Gated DeltaNet / HGRN-2 | matriz `H_t [d,k]` (resolve o *recall gap* do Mamba) | **`OP_RANK1_UPDATE (0x24)` nativo.** `H = α·H + β·v⊗kᵀ` por token não é expressível (produto externo + decaimento matricial); cobre também fast-weights de Titans e Hebbian geral. Stateful CoW como `ssm_states`. |
| Fractal/hierárquicos (árvore de abstração) | pilha de níveis | **Lowering, sem `OP_TREE_STACK`/`OP_HIERARCHICAL_REDUCE`.** Reduce por nível = `GATHER`+`ADD`/`NORM`; ramos = `FORK` (subcontextos já existem). Recursão cabe no scheduler atual. |
| Liquid/ODE (robótica, IoT, séries) | estado contínuo `x(t)` | **`OP_ODE_STEP (0x25)` fundido, baixa prioridade.** Euler/RK são `ADD+MUL`+campo (lowering funciona); nativo economiza dispatch em loop sensorial rápido. Por último, como `DENOISE`. |
| Árvores/RF/XGBoost/LightGBM | tabelas planas feat/limiar/folha | **`OP_FOREST (0x22)` nativo.** `COMPARE+JUMP` por nó sofre *branch mispredict* e não escala p/ 1000s de árvores; travessia vetorizada em lote sobre tabelas contíguas (`OP_LOOKUP_TREE` generalizado) é a representação certa. |
| SVM/logística/linear | `w, b` | **Lowering (`MATVEC`/`MUL+ADD`).** Zero opcode novo — é o mesmo `y = wᵀx+b` do DL. |
| kNN/kMeans/DBSCAN + RAG | banco de vetores | **`OP_DISTANCE (0x23)` nativo com top-k fundido.** Distância em lote + seleção top-k num opcode só = busca vetorial RAG nativa, que a tese de inferência precisa de qualquer forma. |
| PCA/SVD/t-SNE | autovetores | **Lowering iterativo, sem `OP_SVD_STEP`/`OP_EIGEN_SOLVER`.** Power-iteration/QR via `MATVEC+NORM` em loop; solver denso nativo seria LAPACK inteiro — desproporcional. Reavaliar só com caso de uso medido. |

Caiu do mapa (dedicados desnecessários): `OP_BACKPROP_LOCAL`, `OP_MATRIX_RECURRENCE` (virou `RANK1_UPDATE`, mais geral), `OP_TREE_STACK`, `OP_HIERARCHICAL_REDUCE`, `OP_BRANCH`, `OP_EUCLID`/`OP_COSINE` separados (viraram modos de `DISTANCE`), `OP_SVD`/`OP_EIGEN`.

### 8.2 Codificação nos 26B

* **`OP_FOREST (0x22)`** — `rdest`=scores/votos `[1,n_trees]` ou agregado, `rsrc1`=features `[1,F]`, `rsrc2`=tabela de nós plana (feat_idx, thresh, left, right empacotados), `rsrc3`=valores de folha. `payload[0..2]`=`n_trees u16`, `[2..4]`=`max_depth u16`, `[4]`=`mode (0=voto/soma,1=probabilidade média)`, resto reservado. Travessia sem branch (índice aritmético), bit-exata vs referência escalar.
* **`OP_DISTANCE (0x23)`** — `rdest`=dists `[1,N]` (+ índices se top-k), `rsrc1`=query `[1,D]`, `rsrc2`=banco `[N,D]`. `payload[0]`=`metric (0=EUCLID,1=COSINE,2=MANHATTAN,3=DOT)`, `[1..3]`=`topk u16` (0=todas), resto reservado. Top-k fundido com heap parcial (não ordena o banco inteiro).
* **`OP_RANK1_UPDATE (0x24)`** — `rdest`=`H` in-place `[d,k]` (estado matricial), `rsrc1`=`v [d]`, `rsrc2`=`k [k]`, `rsrc3`=`0xFF` ou máscara/gate. `payload[0..4]`=`α (decay) f32`, `[4..8]`=`β f32`, `[8]`=`mode (0=hebbian,1=delta-normalizado,2=forget-gateado)`, `[9]`=`layer_id`. `FORK` empilha `H`, `ABORT` restaura (protocolo `ssm_states`).
* **`OP_ODE_STEP (0x25)`** — `rdest`=`x(t+dt)`, `rsrc1`=`x(t)`, `rsrc2`=controle/parâmetros `u(t)`, `rsrc3`=pesos do campo ou `0xFF` (campo linear default). `payload[0..4]`=`dt f32`, `[4]`=`method (0=Euler,1=RK2,2=RK4)`, `[5]`=`layer_id`. Campo = `MATVEC+SILU` fundido no MVP.

Sintaxe assembler:

```asm
FOREST r4, r0, r1, r2 TREES=512 DEPTH=8 MODE=VOTE
DISTANCE r5, r0, r1 METRIC=COSINE TOPK=5
RANK1_UPDATE rH, r1, r2 ALPHA=0.99 BETA=1.0 MODE=DELTA LAYER=0
ODE_STEP r2, r0, r1 DT=0.01 METHOD=RK2
```

### 8.3 Flags novas em opcodes existentes (sem opcode novo)

* `MATVEC` ganha flag `TRANSPOSE` (opera `Wᵀ·x` direto do `mmap`, sem transpor pesos — necessária p/ backward local de Titans e p/ atenção eficiente). Só flag + caminho `matvec` transposto.
* `SAMPLE` ganha modo `TOPK=k` (escreve índices, ver G1) — serve MoE-gate e reuso em `DISTANCE` quando separado.

## 9. Fases da onda 2

### H1 — `OP_DISTANCE (0x23)` + `OP_RANK1_UPDATE (0x24)` (2 sem, logo após G1)
- [ ] `exec_distance` (4 métricas, `faer`/SIMD, heap parcial p/ top-k) + `exec_rank1` (outer-product fundido com decaimento, denso; batched por layer).
- [ ] Demos `programs/rag_search_demo.m3asm` (query → top-5 sobre banco 1k×64) e `programs/deltanet_demo.m3asm` (recorrência matricial 8×8, 50 passos + `ABORT` restaura `H`).
- Aceite: distâncias `<1e-5` vs `ndarray`; top-k idêntico a sort completo; `RANK1` bit-exato vs lowering `MUL+ADD` triplo; rollback de `H` bit-exato.

### H2 — `OP_FOREST (0x22)` + flag `TRANSPOSE` (1–2 sem, após G3)
- [ ] `exec_forest` (travessia vetorizada, 2 modos) + flag `MATVEC_TRANSPOSE` com teste contra pesos transpostos materializados.
- [ ] Demo `programs/forest_demo.m3asm` (ensemble 128 árvores, profundidade 6, paridade vs avaliador escalar em Python/Rust).
- Aceite: paridade bit-exata escore por escore; throughput documentado vs cadeia `COMPARE+JUMP` (alvo ≥10× em 1k árvores).

### H3 — `OP_ODE_STEP (0x25)` (1 sem, após G4)
- [ ] `exec_ode` Euler + RK2 (+RK4 se custo couber), campo linear+SILU fundido.
- [ ] Demo `programs/ode_demo.m3asm` (oscilador amortecido 2-D, 200 passos `dt=0.01`, `STREAM` da trajetória).
- Aceite: Euler/RK2 `<1e-4` vs integrador de referência no passo 200 (mesmo `dt`).

## 10. Decisões pendentes (onda 2, travar antes de H1)

1. `DISTANCE` com top-k fundido vs. separado (`SAMPLE TOPK`)? (Recomendado: fundido — evita materializar N distâncias; RAG agradece.)
2. `RANK1` modo delta-normalizado (OMA-style, divide por `1+β·‖k‖²`) entra no MVP ou só hebbian+forget? (Recomendado: os 3 modos desde o dia 1 — é 1 divisão a mais e cobre DeltaNet fiel.)
3. `FOREST` suporta valores categóricos/one-hot no MVP ou só limiar float? (Recomendado: só float; categórico é follow-up via `GATHER` de embeddings.)
4. TTT de Titans: teto de tamanho da memória-MLP (ex. ≤2 layers × 512) para o update local não estourar o budget de preempção? (Recomendado: sim, documentar teto + medir `ABORT` sob TTT ativo.)

## 11. Consolidação — nomes propostos (Titans/DeltaNet/fractais/liquid/ML clássico) → mapa congelado

Origem: revisão de arquiteturas emergentes (contexto infinito, memória de longo prazo,
eficiência de hardware) + ML clássico em produção (risco/fraude/tabular). Regra mantida:
**nome proposto vira opcode nativo só onde lowering é inadequado** (§1); o resto vira
alias documentado ou lowering. Nada abaixo realoca `0x1E–0x25`.

### 11.1 Emergentes

| Nome proposto | Veredito | Onde mora |
|---|---|---|
| `OP_BACKPROP_LOCAL` (Titans/MIRAS, TTT em inferência) | **Lowering, sem opcode dedicado.** Passo TTT = `MATVEC` (+ flag `TRANSPOSE`, §8.3) + `MUL/ADD` + `OP_RANK1_UPDATE` p/ escrita. Com teto de MLP (§10.4) o dispatch extra cabe no budget de preempção; reavaliar só com bench. | §8.1, H2 |
| `OP_MATRIX_RECURRENCE` (Gated DeltaNet/HGRN-2, `H_t = H_{t-1} + v⊗kᵀ`) | **Alias de `OP_RANK1_UPDATE (0x24)`.** O nome congelado é mais geral (cobre Hebbian/delta/forget + fast-weights de Titans); "matrix recurrence" é o caso de uso DeltaNet, não um opcode distinto. | H1, §8.2 |
| `OP_TREE_STACK` / `OP_HIERARCHICAL_REDUCE` (fractais/hierárquicos) | **Lowering.** Reduce por nível = `GATHER+ADD/NORM`; ramos = `FORK` (subcontextos já existem). Recursão cabe no scheduler atual. | §8.1 |
| `OP_ODE_SOLVER` (Liquid/ODE, Euler/RK) | **Renomeado p/ `OP_ODE_STEP (0x25)`.** A VM executa *um passo fundido* (`x(t)→x(t+dt)`); "solver" sugere driver de loop adaptativo, que pertence ao `.m3asm` (loop + `COMPARE`), não à ISA. | H3, §8.2 |

### 11.2 ML clássico

| Nome proposto | Veredito | Onde mora |
|---|---|---|
| `OP_BRANCH` (árvores: `feature > limiar`) | **`COMPARE+JUMP` existentes (`0x0C–0x0E`).** Branch escalar continua válido p/ árvores pequenas; ensembles grandes usam `FOREST`. | — |
| `OP_LOOKUP` / `OP_LOOKUP_TREE` (travessia sem branch) | **`OP_FOREST (0x22)` p/ ensembles** (travessia vetorizada, §8.2); **embedding-bag via `OP_GATHER (0x1F)`**. `LOOKUP_TREE` é o caso `TREES=1` de `FOREST` — alias, não opcode. | G1, H2 |
| `OP_DOT_PROD` / `OP_MATVEC` (SVM/logística/linear, `y=wᵀx+b`) | **`OP_MATVEC (0x10)` cobre.** Dot é o caso `out_dim=1`; kernel/threshold via `MUL/ADD` + `SILU`. Zero opcode novo. | — |
| `OP_EUCLID` / `OP_EUCLIDEAN_DIST` / `OP_COSINE` / `OP_COSINE_SIM` (kNN/kMeans/DBSCAN) | **Modos `METRIC=` de `OP_DISTANCE (0x23)`.** Quatro opcodes p/ uma fórmula com flag seriam inflação; Manhattan/DOT entram no mesmo campo. | H1, §8.2 |
| `OP_SVD_STEP` / `OP_EIGEN` / `OP_EIGEN_SOLVER` (PCA/SVD) | **Lowering iterativo** (power-iteration/QR via `MATVEC+NORM` em loop). Solver denso nativo = um LAPACK inteiro — desproporcional sem caso medido. | §8.1 |

### 11.3 Os 5 domínios (mapa definitivo)

| Domínio | Opcodes | Famílias atendidas |
|---|---|---|
| **1. Dense Matrix Ops** | `MATVEC 0x10, MUL 0x11, CONV 0x1E` (+ `DOT` via `MATVEC`) | Transformers, CNNs/visão, SVM/lineares |
| **2. State & Recurrence** | `ATTN 0x02` (+`KV_CACHE`), `SSM_SCAN 0x13`/`SSM_RESET 0x14`, `RANK1_UPDATE 0x24`, `ODE_STEP 0x25`, `SPIKE_STEP 0x20` | Mamba/DeltaNet, Titans-TTT, Liquid, SNNs, RNNs (lowering) |
| **3. Control & Flow** | `FORK 0x04, ABORT 0x05, SENSE 0x06, STREAM 0x03, COMPARE 0x0C, JUMP 0x0E, IF_* 0x0D/0x0F, SAMPLE 0x0B` (+`TOPK`), `CTX_SWITCH 0x18`, cluster `0x1A–0x1D` | Thinking loops, preempção/rollback, MoE-dispatch (via `GATHER`), distribuído |
| **4. Audio & Codec** | `CODEC_ENC 0x15, CODEC_DEC 0x16, AUDIO_ALIGN 0x17`, `SENSE PCM/CODEC_FRAME`, `ROPE 0x19` | Moshi/Mimi full-duplex, híbridos voz+texto |
| **5. Classical & Retrieval** | `FOREST 0x22, DISTANCE 0x23, GATHER 0x1F, DENOISE_STEP 0x21` | XGBoost/RF, kNN/RAG, GNNs, difusão |

Regra de evolução (tese §5): opcode stateful novo declara (a) onde mora o estado,
(b) custo de snapshot, (c) como `ABORT` o restaura — ver taxonomia §3. Aliases desta
seção (`MATRIX_RECURRENCE`, `LOOKUP_TREE`, `ODE_SOLVER`…) são resolvidos no assembler
para o opcode canônico e **nunca** ganham encoding próprio.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
