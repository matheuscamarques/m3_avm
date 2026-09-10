# DECISÃO — publicação defensiva, sem depósito de patente

**Data:** 2026-09-10 · **Decisor:** Matheus de Camargo Marques (ORCID 0009-0003-4518-2258)
**Decisão:** não depositar patente; publicar código + especificações como prior art defensivo.

## Efeitos

- O conteúdo de `docs/patente/` vira **anexo histórico de análise**, não minuta de depósito. Não citar como “patent pending”.
- A proteção passa a ser: **AGPL-3.0-or-later** (código, §13 p/ rede) + **CC BY 4.0** (textos no Zenodo) + DOI/arXiv como prova de data e autoria (freedom to operate).
- Sem custos de depósito/anuidade; sem exclusividade — qualquer um pode implementar o descrito (é o objetivo).

## Antes de publicar (obrigatório)

1. `git add -A && git commit && git tag v0.1.0` (árvore hoje suja — ver `PROVENANCE-2026-09-10.txt`).
2. `./docs/zenodo/pack.sh v0.1.0` → subir ZIP no Zenodo (tipo Software, AGPL) + PDFs das specs (tipo Publication, CC BY 4.0) — ver `docs/zenodo/README_ZENODO.md`.
3. Corrigir hype antes do DOI: sem “217µs”, sem “256 opcodes/50 anos”, sem “substitui PyTorch”; usar só `docs/ESPEC.md §17` (medido) e marcar resto como alvo. Título do depósito: ISA **v1.4**, não v2.4.
4. Após DOI: colar badge no `README.md`, adicionar Works no ORCID, criar perfil Google Scholar/Semantic Scholar.
