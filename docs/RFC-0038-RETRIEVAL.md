# RFC-0038 — Retrieval em ISA (`0x50–0x53`, `0x56–0x57`; bump v1.15)

```text
Status      : DRAFT (escopo travado em docs/V-2_SCOPE.md; sem código ainda)
Category    : Standards Track
Updates     : ESPEC-V2 §3.8 (DRAFT→IMPL); ISA v1.14→v1.15; m3bc REQUIRED (bit 50)
Obsoletes   : None
Feature Bit : M3BC_REQUIRED_HAS_V2_RETRIEVAL = 1<<50 (REQUIRED, não OPTIONAL)
Bump        : minor v1.14 → v1.15 (família com encoding; patch é p/ fix)
```

## Abstract

Seis opcodes de retrieval entram na ISA: índice (`ADD`/`DEL`), busca
(`SEARCH` sobre o núcleo `DISTANCE`), `EMBED_LOOKUP` (bag) e par
`PQ_ENCODE`/`DECODE`. `0x54`/`0x55` seguem HELD sem opcode (RFC-0030).
Negociação via bit REQUIRED 50: runtime antigo rejeita `.m3bc` V-2
antes do fetch (firewall, não chute).

## Motivation

RAG no ISA e não host-side: o tutor (Decisão 7) precisa de lookup de
vocabulário dentro do programa (branch por `RAG_SEARCH`, Tabelas via
harness), não via chamada externa — mesma razão que pôs `FOREST` e
`DISTANCE` na ISA em vez de callbacks. O núcleo matemático já existe
(`DISTANCE` + top-k fundido, provado por `rag_search_demo`); V-2
adiciona o estado (índice) e as projeções (lookup/PQ), não a métrica.

## Specification

Convenções herdadas (nada novo): `rdest` = resultado; `0xFF` = ausente;
payload 26B com inteiros LE; `f32` só; `NaN/Inf` de entrada => erro
alto; braços RFC-0008 desde o dia 1.

```text
RAG_INDEX_ADD rD, rDb, rVec [, rId]   ; 0x50
  rDb=0 cria store (rD <- novo id u128 sequencial, host-side IndexStore,
  fora do heap — como kv_heap); senão anexa (rD <- nova contagem u64).
  rVec [1,D] F32 (D fixado na criação; divergente => erro).
  rId ausente (0xFF) => id sequencial. payload: vazio.
RAG_INDEX_DEL rD, rDb, rId            ; 0x51
  rD <- contagem restante. Id ausente => erro (sem deleção silenciosa).
RAG_SEARCH rD, rQuery, rDb [TOPK=n] [METRIC=COSINE|EUCLID|DOT]  ; 0x52
  rD <- [1,2*TOPK] F32 (dists+índices; mesmo layout DISTANCE-TOPK).
  TOPK default 1; TOPK=0 ou >tamanho => erro. payload[0..2]=topk u16,
  payload[2]=metric u8 (0=COSINE,1=EUCLID,2=DOT).
EMBED_LOOKUP rD, rIds, rTable         ; 0x53
  rIds [1,N] F32 com valores integrais (ids); rTable [V,D] F32.
  rD <- [1,D] média das linhas (bag). Id fora de [0,V) => erro
  (sem clamp). N=0 => erro.
PQ_ENCODE rD, rVec, rCodebook         ; 0x56
  rVec [1,D], D % NSUB == 0; rCodebook [NSUB*LEVELS,subdim] F32.
  rD <- [1,NSUB] F32 carregando u16 (precedente CODEC_AS_TENSOR).
  payload[0..2]=nsub u16. Sem codebook => erro (nota harness_table).
PQ_DECODE rD, rCodes, rCodebook       ; 0x57
  inverso; rD <- [1,D] F32. Mesmas validações.
```

`IndexStore`: `id u128 -> {dim, Vec<(id_vec u64, Vec<f32>)>}`;
contadores monotônicos (store ids e vec ids); `DEL` não reutiliza ids.

## Out-of-band

`0x54 HASH_BUCKET`, `0x55 QUANTIZE_VEC`: HELD (RFC-0030, lowering
adequado). Reabrir exige re-dossiê R12.

## Bump

v1.14 → **v1.15**. `M3BC_REQUIRED_HAS_V2_RETRIEVAL: u64 = 1<<50`
(bit 50, livre — nenhum bit atribuído até hoje); implementação liga em
`M3BC_SUPPORTED_REQUIRED`. Arquivos antigos: zero `0x50–0x57` em uso
(verificado por grep) => byte-idênticos; decoder atual rejeita os
opcodes como `UnknownOpcode` (alto, nunca misdecode).

## Backwards Compatibility

Só adição. Corpus monta byte-idêntico (gate na implementação).
`.m3bc` V-2 em runtime v1.14: rejeitado no load por REQUIRED (com bits
nomeados) ou por `UnknownOpcode` — nunca execução parcial.

## Security Considerations

- Bounds: TOPK≤tamanho, ids em faixa, dims iguais, `D%NSUB==0`,
  `u16` checados (`try_from`, nunca truncar).
- Sanitização: dtype F32-only (quantizado veta explícito); `NaN/Inf`
  veta na entrada (precedente FOREST).
- DoS: `IndexStore` sem teto nesta RFC (emulador; quota é G8/follow-up
  documentado, não silencioso).
- Determinismo: zero `rand`; empate => menor id (precedente DISTANCE).

## Reference Implementation (plano; código no próximo turno)

- `src/opcodes.rs`: `OP_*` + ctors (idioma `instr_forest`) + 6 braços
  + testes `bad`/`good` (RFC-0008) + goldens de `V-2_SCOPE.md` §5.
- `src/memory.rs` ou `src/rag.rs`: `IndexStore` + cursor monotônico.
- `src/vm.rs`: 6 execs + `test_rfc0038_*`.
- `benches/rag_bench.rs` (padrão Criterion) + `programs/rag_demo.m3asm`.
- `src/m3bc.rs`: constante do bit + ligar em `SUPPORTED_REQUIRED`.
- ESPEC-V2 §3.8 DRAFT→IMPL; suite verde + corpus gate.

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0038-00 | 2026-09-11 | DRAFT: 6 in-band, `0x54`/`0x55` out, bump v1.15, bit 50 REQUIRED |

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
