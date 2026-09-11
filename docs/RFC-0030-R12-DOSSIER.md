# RFC-0030 — Trilha P, Dossiê R12: ML Clássico + Retrieval HELD + Placeholders Exóticos

```text
Status      : IMPLEMENTED
Category    : Informational (analysis; proposes no encodings)
Updates     : docs/ESPEC-V2.md §3.8/§3.9/§3.16 (R12 verdict notes);
              docs/ESPEC.md §13.3 (aliases)
Obsoletes   : None
Feature Bit : none (no new opcodes — that is the verdict)
Bump        : none (deliberately: a proof dossier bumps nothing)
```

## Abstract

First execution of Trilha P (plan §Trilha P): every HELD item gets
its lowering-inadequacy trial under the ESPEC Section 13 filter
("native only where lowering is inadequate"). Verdict, stated upfront:
**10 of 10 evaluated items need NO opcode** — each lowers to existing
ops, and all nine lowering sketches below were executed to HALT
(exit 0), two with value asserts (`test_rfc0030_lowering_runs`).
Exotic placeholders (`0xB7-0xFE`) stay RESERVED with entry criteria
(no lowering target exists yet — nothing to prove against). This is
the outcome the plan predicted as possible ("a prova pode concluir
que lowering é adequado → alias, não opcode").

## §0. Método e legendas

- Filtro §13: um opcode só se justifica onde o lowering explode em
  memória, exige primitiva inexistente sem substituto, ou impõe custo
  assintótico pior. Verbosidade ("é feio") não conta; "treino" não
  conta (non-goal, ESPEC §1.2).
- Tags de verificação (cada uma executada, não afirmada):
  `[RUNS]` = monta sob assembler estrito + roda até HALT exit 0;
  `[RUNS+ASSERT]` = acima + valores verificados em teste;
  `[ASSEMBLES]` = monta (comportamento em runtime coberto pelos
  componentes, já testados).
- Divisão treino-vs-serving: ajustar/fitar/iterar-até-convergir é
  treino (non-goal); avaliar/classificar/recuperar é serving e é o
  que se julga aqui.

## §1. ML clássico (`0x58-0x5F`) — nenhum opcode

### `0x58 KMEANS_STEP` — NO OPCODE
Atribuição Lloyd = `DISTANCE TOPK=1` + fatiamento do pack
(`SLICE START=N LEN=N`, padrão `attn_topk_demo`). `[RUNS+ASSERT]`:
`programs/kmeans_assign_demo.m3asm` (k=2, N=2 desenrolado; idx
exatos 0 e 1 verificados). Lote dinâmico de N arbitrário exigiria
`SLICE` com bounds em registrador (hoje imediatos) — desenrola-se
para N fixo; reagrupamento dinâmico pleno quer leitura
tensor→escalar (gap geral, §4). Refit/retreinamento é non-goal.

### `0x59 LINEAR_REG` — NO OPCODE
`y = MATVEC + ADD`, literalmente. `[RUNS]`. Alias registrado (§5).

### `0x5A LOGISTIC_REG` — NO OPCODE
`MATVEC + ADD + SIGMOID` (0x3E existe desde a RFC-0028). `[RUNS]`.
Alias registrado (§5).

### `0x5B NAIVE_BAYES` — NO OPCODE
Via log-domínio com tabelas pré-computadas: `GATHER` (lookup por
classe) + `LOG` + `REDUCE SUM`, mais prior em `ADD`. `[RUNS]`
(`nb_log`, com índices f32 exatos < 2^24 — caminho já provado pelo
`attn_topk_demo`). Gaussiana com variância via `E[x²]−E[x]²`
(`MUL`+`REDUCE`, sem `SUB` tensorial — ver padrão Newton, §4).
Fitting/contagem é treino (non-goal).

### `0x5C SVM_PREDICT` — NO OPCODE
Linear: `MATVEC + ADD` (função de decisão; limiar escalar via
`COMPARE PRED=` em regs). RBF: `DISTANCE EUCLID + MUL + EXP`
(kernel gaussiano exato). `[RUNS]` ambos. Sinal por elemento cai no
gap geral de predicado tensorial (§4) — que um op `SVM_PREDICT`
também teria, pois devolveria tensor do mesmo jeito: o op não compra
nada.

### `0x5D PCA_STEP` — NO OPCODE
Iteração de potência = `MATVEC` + normalização por Newton
(`MUL`/`ADD`/`LOADI`/`FILL` apenas — sem `DIV`/`SQRT`). `[RUNS+ASSERT]`:
`programs/newton_rsqrt_demo.m3asm` (uma iteração p/ 1/√2 desde 0.5 =
0.625 bit-exato, verificado). Convergência iterada desenrola ou
aguarda índices dinâmicos (§1-KMEANS); fitting é non-goal.

### `0x5E STANDARDIZE` — NO OPCODE
`REDUCE MEAN` + `MUL`/`ADD` (variância por momentos) + Newton p/
recíproco + `MUL`, com `BROADCAST` de volta ao shape (`[RUNS]` do
esqueleto; o loop Newton plugável é o do `0x5D`, `[RUNS+ASSERT]`).

### `0x5F NEAREST_CENTROID` — NO OPCODE
`DISTANCE TOPK=1` É nearest-centroid (não "como"). `[RUNS]` (este
dossiê + `attn_topk_demo` + `gather_moe_demo` pré-existentes).
Alias registrado (§5).

## §2. Retrieval HELD (`0x54-0x55`) — nenhum opcode

### `0x54 HASH_BUCKET` — NO OPCODE
LSH por projeção aleatória + empacotamento aritmético dos bits
(`MUL` por potências de 2 + `ADD` — sem ops de bit, sem problema).
`[RUNS]` do prefixo (`MATVEC` de projeção; tabela de hiperplanos
assume init multivalor — follow-up RFC-0019 já registrado).
Extração de sinal é o gap geral de predicado tensorial (§4):
um op `HASH_BUCKET` o teria igual — não compra nada. NN exato já é
nativo (`DISTANCE`).

### `0x55 QUANTIZE_VEC` — NO OPCODE
Encode VQ = `SLICE` (subvetores) + `DISTANCE TOPK=1` por subvetor
(`[RUNS]`); decode = `GATHER` no codebook (existente). Distinto de
`QUANTIZE` 0x68 (blocos escalares vs codebook — ambos cobertos,
cada um no seu lugar).

## §3. Placeholders exóticos (`0xB7-0xFE`) — seguem RESERVED

Sem alvo de lowering = nada a provar (R12 exige prova contra um
lowering candidato; aqui não há candidato). Critérios de entrada
honestos, por faixa:

- `0xB7-0xBF` (federado): exige transporte (Fase 7) + protocolo de
  agregação com prova de conservação — sistema, não opcode, até lá.
- `0xC0-0xCF` (TEE/confidencial): exige modelo de ameaça + mecanismo
  de atestação; emulador funcional não executa enclave.
- `0xD0-0xDF` (fotônico/neuromórfico), `0xE0-0xEF` (quântico):
  exigem backend executável (simulador conta, com semântica
  publicada) — papel aceita tudo, opcode não.
- `0xF0-0xFE` (sistema estendido): caso a caso via RFC com prova §13.

## §4. Gaps gerais encontrados (não são propostas de opcode)

Ventilados porque o filtro os revelou, registrados para não
re-descobrir — nenhum vira opcode neste dossiê:

1. **Predicado tensorial → máscara** (ex.: sinal por elemento,
   `x > 0` vetorizado). `COMPARE PRED=` é escalar (regs). Workaround
   hoje: `SIGMOID` íngreme / `SAMPLE` com seed / caminhos por
   `DISTANCE`. Um futuro `CMP_GT` tensorial seria ALU geral, nunca
   um op por algoritmo.
2. **Leitura tensor→escalar**: só via `SAMPLE` legado (estocástico;
   determinístico sob seed, RFC-0009) ou fatiamento estático. Loops
   dinâmicos sobre linhas aguardam `SLICE` com bounds em reg
   (extensão futura, não opcode novo).
3. **Sem `DIV`/`SQRT`/`SUB` tensorial**: padrão Newton
   (`MUL`/`ADD`/`FILL`/`LOADI`) cobre recíproco e raiz com bounds
   conhecidos (`[RUNS+ASSERT]`). Se um dia doer de verdade, a resposta
   é ALU geral, não opcodes de nicho.
4. **Init multivalor de tabelas** (já follow-up RFC-0019): tabelas
   FOREST/XGBoost/hiperplanos/bibliotecas assumem tensores assados
   offline (§5.3 de `FILLER_SPEECH.md`).

## §5. Aliases registrados (ESPEC §13.3, nunca codificados)

`LINEAR_REG` → `MATVEC+ADD` · `LOGISTIC_REG` → `MATVEC+ADD+SIGMOID` ·
`NEAREST_CENTROID` → `DISTANCE TOPK=1` (+`SLICE`) · `MOE_DISPATCH` →
`MATVEC`+`SAMPLE-TOPK`+`GATHER` (loop já demonstrado) ·
`VQ_ENCODE` → `SLICE`+`DISTANCE-TOPK` · `STANDARDIZE` →
`REDUCE`+`MUL`/`ADD` (Newton) · `POWER_ITER` → `MATVEC`+`MUL`/`ADD`
(Newton) · `RBF_KERNEL` → `DISTANCE`+`MUL`+`EXP`.

## Verificação (executada 2026-09-11)

- 9 sketches em `/tmp/r12/*.m3asm`: todos montam sob assembler
  estrito; todos rodam até HALT exit 0 (`--max-steps 40`).
- 2 demos commitadas em `programs/` com asserts de valores:
  `newton_rsqrt_demo` (0.625 exato), `kmeans_assign_demo` (idx 0, 1)
  — `vm::test_rfc0030_lowering_runs` verde.
- Demos pré-existentes citadas (`gather_moe_demo`,
  `rag_search_demo`, `attn_topk_demo`, `forest_demo`) seguem verdes
  na suite (não re-testadas aqui por serem de outras RFCs).
- Suite: `cargo test --lib` 327 passed (sole failure: pre-existing
  unrelated moshi norm-gamma); corpus gate (incl. os 2 demos novos)
  verificado por CLI loop.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0030-00 | 2026-09-11 | DRAFT: dossiê R12 + sketches verificados |
| 0030-01 | 2026-09-11 | IMPLEMENTED: 10/10 sem opcode; aliases; sem bump |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
