# ACOMPANHAMENTO PI — M³-AVM (assistente técnico)

**Inventor:** Matheus de Camargo Marques — ORCID 0009-0003-4518-2258 · matheuscamarques@gmail.com
**Data-base:** 2026-09-10 · **Papel do assistente:** técnico (não substitui agente de PI).

## 1. Estado da árvore (congelar antes de qualquer divulgação)

`git status` em 2026-09-10: **sujo** (~23 modificados + ~15 untracked, ex. `src/vm.rs`, `src/opcodes.rs`, `docs/ESPEC*.md`, `docs/RFC-*`, `docs/patente/`, `.zenodo.json`).
**Não publique no Zenodo assim.** Congele:

```bash
git add -A && git commit -m "chore(freeze): ISA v1.4 + dossiê patente + zenodo"
git tag v0.1.0 && git log --oneline -5 > PROVENANCE.txt
cargo test --lib 2>&1 | tail -5 >> PROVENANCE.txt
./docs/zenodo/pack.sh v0.1.0
```

Anote aqui: commit ___ tag ___ DOI ___ data Zenodo ___.

## 2. Pasta de contratação do agente (o que levar)

- [ ] `docs/patente/01_RELATORIO_DESCRITIVO.md` + `02_QUADRO` + `03_RESUMO` + `04_DESENHOS`
- [ ] `docs/patente/05_EVIDENCIAS_IMPLEMENTACAO.md` + `PROVENANCE.txt` + saída `cargo test`/`lake build`
- [ ] `docs/ESPEC.md` (normativo) + `docs/ESPEC-V2.md` (DRAFT) + ZIP Zenodo (ou plano de data)
- [ ] Declaração de período de graça (se já divulgou): forma/local/data + DOI + commit
- [ ] Perguntas ao agente (levar impresso):
  1. Principal C1 + divisionais C2, ou pedido único amplo?
  2. Vale depósito BR-relâmpago antes do Zenodo para salvar exterior?
  3. `FOREST/SPIKE/KV_TRANSFER/CHAIN` têm suporte suficiente ou saem do quadro?
  4. Desenhos Fig.1–5 aprovados? Reivindicação 8–10 (C2) mantêm unidade?
  5. Honorários + anuidades + exame em 36m: cronograma e custo fechado?

## 3. Prazos (preencher datas reais)

| Marco | Regra | Data | Status |
|---|---|---|---|
| 1ª divulgação (Zenodo/talk) | marco zero da graça BR | ___ | pendente |
| Depósito BR | ≤12m da divulgação (art. 12 LPI) | ___ | pendente |
| Prioridade exterior/PCT | ≤12m do depósito BR (CUP art. 4) | ___ | pendente |
| Exame técnico | requerer em ≤36m do depósito | ___ | pendente |
| Anuidades | anual após depósito | ___ | pendente |

Perdeu a graça BR = novidade destruída no BR. Publicou antes do BR = novidade destruída onde não há graça (UE/CN).

## 4. Próxima busca de anterioridade (a fazer)

Alvos: Triton Inference Server, vLLM PagedAttention, Ray Serve, TensorRT-LLM, KServe (rollback/scheduler/deadline), arXiv:2608.03893 (cross-model KV).
Entrega: tabela documento × o que antecipa × diferença da C1 × impacto nas claims 1–11.
