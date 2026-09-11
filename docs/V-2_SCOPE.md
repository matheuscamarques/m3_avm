# V-2_SCOPE — Retrieval `0x50–0x57` (dossiê de escopo; RFC depois)

```text
Status:   ESCOPO TRAVADO (inventário + interfaces propostas; RFC-00XX congela)
Author:   Matheus de Camargo Marques <matheuscamarques@gmail.com> — ORCID https://orcid.org/0009-0003-4518-2258
Date:     2026-09-11
Base:     docs/ESPEC-V2.md §3.8 (DRAFT/HELD) · RFC-0030 (R12) · programs/rag_search_demo.m3asm
License:  AGPL-3.0-or-later (see LICENSE)
```

> Milestone público é voice-loop end-to-end (V-3), não V-2. V-2 é meio:
> se esticar, corta PQ (0x56–0x57 vão para V-2b) e fecha o núcleo 0x50–0x53.

## 1. Opcodes da faixa (ESPEC-V2 §3.8; nenhum implementado hoje)

| Opcode | Nome | Estado ESPEC | V-2 |
|:---:|:---|:---|:---|
| `0x50` | `RAG_INDEX_ADD` | DRAFT | in-band |
| `0x51` | `RAG_INDEX_DEL` | DRAFT | in-band |
| `0x52` | `RAG_SEARCH` | DRAFT | in-band |
| `0x53` | `EMBED_LOOKUP` | DRAFT | in-band |
| `0x54` | `HASH_BUCKET` | HELD | **out** (RFC-0030: lowering adequado, sem opcode) |
| `0x55` | `QUANTIZE_VEC` | HELD | **out** (RFC-0030: lowering adequado, sem opcode) |
| `0x56` | `PQ_ENCODE` | DRAFT | in-band (corta p/ V-2b se esticar) |
| `0x57` | `PQ_DECODE` | DRAFT | in-band (corta p/ V-2b se esticar) |

## 2. In-band: interfaces propostas (RFC congela; mesma disciplina V-1)

- `RAG_INDEX_ADD rDb, rVec [, rId]` — anexa vetor ao índice; `rDb` é
  handle u128 p/ `IndexStore` host-side (fora do heap de tensores, como
  `kv_heap`); sem `rId` => id sequencial. Erro alto em dtype≠F32/dim≠store.
- `RAG_INDEX_DEL rDb, rId` — remove; id ausente => erro alto (sem
  deleção silenciosa).
- `RAG_SEARCH rOut, rQuery, rDb [TOPK=n] [METRIC=COSINE|EUCLID|DOT]` —
  núcleo = `DISTANCE` existente; saída `[1,2*TOPK]` (dists+índices,
  mesmo layout do `rag_search_demo`); `TOPK` default 1.
- `EMBED_LOOKUP rOut, rIds, rTable` — bag: gather linhas + média;
  ids fora da tabela => erro alto (sem clamp silencioso).
- `PQ_ENCODE rCodes, rVec [, rCodebook]` — codebook via harness_table
  (nota RFC-0037); sem codebook => erro alto (sem default silencioso).
- `PQ_DECODE rVec, rCodes [, rCodebook]` — inverso; mesma regra.
- Determinismo: zero `rand` (seed explícita se precisar); `NaN/Inf`
  de entrada => erro alto (precedente FOREST/SANITY_CHECK).
- Strictness: braços RFC-0008 desde o dia 1 (teste `bad`/`good` como 0008/0036/0037).

## 3. Out-of-band (não mexe)

`0x54`/`0x55`: julgados RFC-0030 (lowering cobre). Reabrir exige
re-dossiê R12, não "parece útil".

## 4. Bump

`v1.14` → **`v1.15`** (família com encoding; governança: minor por
família; nada de `v1.14.x` — patch é p/ fix, não p/ família).

## 5. Goldens (por opcode; tolerância travada aqui)

- ADD/DEL: sequências (add 3, del 1, search TOPK=2) sobre banco
  `8×4` ramp (mesmo do `rag_search_demo`); ids exatos; del-fantasma erra.
- SEARCH: query ramp => índices exatos bit-a-bit (empate => menor id,
  precedente DISTANCE); distâncias f32 bit-exatas em CPU.
- EMBED_LOOKUP: tabela `4×4` identidade => bag = média exata das linhas.
- PQ: roundtrip `DECODE(ENCODE(v))` com erro ≤ 1 LSB/codebook-nível por
  componente (tolerância numérica explícita; resto bit-exato).
- Todos: `NaN` de entrada erra; shape errado erra; flag desconhecida erra.

## 6. Bench (por opcode; baseline CPU, meta)

Seguir `benches/` existentes (padrão `*_bench.rs` + Criterion): `ADD`
(1 vetor 768d), `SEARCH` (banco 10k×768, TOPK=10), `LOOKUP`
(batch 32), `PQ` (768d, 8 subvetores). Meta dia 1: medir e registrar
(sem alegar alvo de hardware — regra single-rig/emulador).

## 7. Demo

`programs/rag_demo.m3asm`: ADD ×3 → SEARCH TOPK=2 → LOOKUP → PQ
roundtrip → HALT; teste `test_rfcV2_rag_*` com os goldens acima.
Corpus gate passa a 42/43 + 2 API-only (regra vigente).

## 8. Fora (não reabrir sem dossiê)

HNSW/IVF aproximado (flat/entanumerado basta p/ V-2), índice
persistente em disco, `0x54`/`0x55`, streaming-add, multi-tenant.
