# Pacote Zenodo — M³-AVM (leia antes de publicar)

**Author:** Matheus de Camargo Marques — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258) · matheuscamarques@gmail.com · Independent Researcher

> **Congele antes de publicar.** O DOI torna o conteúdo anterioridade para
> fins de patente no exterior e marco do período de graça de 12 meses no
> Brasil (art. 12 LPI). Não publique sem decidir a estratégia em
> `docs/patente/00_LEIAME.md` (ideal: depósito BR primeiro, Zenodo no dia
> seguinte). **Não sou advogado — valide com agente de PI.**

## 1. O que este pacote contém (e o que NÃO afirma)

- **Código (AGPL-3.0-or-later):** `src/`, `Cargo.toml`, `Cargo.lock`,
  `examples/`, `programs/`, `benches/`, `tests/`, `formal/` (Lean 4).
- **Especificação normativa:** `docs/ESPEC.md` (ISA v1.4).
- **Proposta futura (DRAFT, não implementar):** `docs/ESPEC-V2.md` + RFCs
  `docs/RFC-*.md` + `docs/arq/` (arquivo, informativo).
- **Dossiê de patente (minuta, não depósito):** `docs/patente/` com camadas
  C1 (habilitado) / C2 (futuro) explícitas.
- **Excluído do ZIP:** `target/`, `m3_persistent.dat` (67 MB gerado),
  `models/*.gguf` (binários externos), `whisper.cpp/` (submódulo externo).

O que é comprovado vs futuro está em `docs/ESPEC.md §§15–17` e
`docs/patente/05_EVIDENCIAS_IMPLEMENTACAO.md`. Números fora dessas seções
são alvo, não medição.

## 2. Split de licenças (importante)

- **Software:** `AGPL-3.0-or-later` (`LICENSE` na raiz). Uso em rede de versão
  modificada exige oferta do fonte correspondente (§13).
- **Textos (ESPEC, RFCs, dossiê):** recomendamos **CC BY 4.0** no formulário
  Zenodo para o registro tipo *Publication/Preprint*; o campo `license` do
  `.zenodo.json` (`AGPL-3.0-or-later`) vale para o registro tipo *Software*
  via integração GitHub. Se fizer **um único depósito manual misto**, escolha
  **AGPL-3.0-or-later** e declare no `Description`: “Code under AGPL-3.0-or-later;
  texts under CC BY 4.0”.
- **Sem ORCID?** `.zenodo.json` atual está sem `orcid` (placeholder removido).
  Adicione o seu no Zenodo web + `CITATION.cff` antes de publicar.

## 3. Nomenclatura e estrutura do ZIP

```
m3_avm-v0.1.0-zenodo-AAAA-MM-DD.zip
├── README.md                 (este repo, com badge DOI após publicar)
├── CITATION.cff
├── LICENSE
├── .zenodo.json
├── Cargo.toml / Cargo.lock
├── src/  examples/  programs/  benches/  tests/
├── docs/ESPEC.md  docs/ESPEC-V2.md  docs/RFC-*.md
├── docs/patente/00_LEIAME.md … 05_EVIDENCIAS_IMPLEMENTACAO.md
├── docs/zenodo/README_ZENODO.md (este arquivo)
└── formal/ (fontes Lean; sem `formal/.lake/build`)
```

Nomes descritivos, sem `final_final`. Um `README` na raiz do ZIP é obrigatório.

## 4. Metadados sugeridos (copiar/colar no formulário)

- **Title:** `M³-AVM: Research Prototype for Sparse Event-Driven Tensor Computation (ISA v1.4 + proofs)`
- **Creators:** `Marques, Matheus de Camargo` (+ ORCID) — Independent Researcher
- **Description (≥150 palavras):** usar o campo `description` do `.zenodo.json`
  + acrescentar: métodos (emulador Rust, CSR, Q4_K AVX2, GGUF mmap, Lean),
  contribuições (preempção por chunk, rollback CoW transversal, RNG por
  contexto), limites (wgpu só ATTN≤64, I-Persist aberto, cluster/ondas
  universais RSVD) e estrutura de pastas acima.
- **Keywords (3–10):** `virtual machine`, `instruction set architecture`,
  `AI inference`, `sparse computation`, `heterogeneous computing`,
  `deterministic rollback`, `KV cache`, `Mamba SSM`
- **Resource type:** `Software` (código) — e, se separado, `Publication / Preprint`
  para ESPEC+RFCs em PDF/A.
- **Date:** data do tag (`AAAA-MM-DD`), igual ao `version`/`date-released`.
- **Related identifiers:** `isSupplementTo → https://github.com/matheuscamarques/m3_avm`
- **Communities:** submeter a `artificial-intelligence` se pertinente.

## 5. PDF/A (arquivamento)

O guia que você colou está correto: **PDF/A + fonte LaTeX junto** para textos.
Fluxo mínimo sem LaTeX:

```bash
# requer pandoc + texlive
pandoc docs/ESPEC.md -o /tmp/ESPEC-v1.4.pdf \
  --pdf-engine=xelatex -V colorlinks:true \
  --metadata title="M³-AVM ESPEC v1.4"
# PDF/A-2b (verificar com veraPDF antes de subir):
gs -dPDFA=2 -dBATCH -dNOPAUSE -sProcessColorModel=DeviceRGB \
   -sDEVICE=pdfwrite -sPDFACompatibilityPolicy=1 \
   -sOutputFile=/tmp/ESPEC-v1.4-PDFA.pdf /tmp/ESPEC-v1.4.pdf
```

Suba **tanto o `.md` quanto o `-PDFA.pdf`** (reusabilidade + preservação).
Sem pandoc instalado? Suba os `.md` + este README e gere o PDF/A depois como
nova versão do registro (Zenodo versiona DOIs).

## 6. Automação GitHub → Zenodo

1. Zenodo.org → Log in with GitHub → GitHub → flip **ON** em `m3_avm`.
2. No GitHub, crie Release `v0.1.0` (tag = `version` do `.zenodo.json` e
   `CITATION.cff`). O webhook do Zenodo arquiva o tag e minta DOI de versão +
   DOI conceito.
3. Após publicar, cole no `README.md`:
   `[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)`
4. `.zenodo.json` tem precedência sobre `CITATION.cff` no GitHub-Zenodo;
   mantenha título/autores/licença iguais nos dois (já alinhados).

## 7. Checklist anti-perda de direitos (faça nesta ordem)

- [ ] `git tag v0.1.0 && git log --oneline -5 > PROVENANCE.txt && cargo test --lib 2>&1 | tail -5 >> PROVENANCE.txt`
- [ ] ZIP sem `target/`, `*.dat`, `*.gguf`, `whisper.cpp/build`
- [ ] Declaração de graça pronta (forma/local/data da divulgação) se for depositar BR em ≤12 meses
- [ ] Decisão exterior: publicar = abrir mão de novidade onde não há graça (UE/CN). Alternativa: depósito BR hoje, Zenodo amanhã.
