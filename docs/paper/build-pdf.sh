#!/usr/bin/env bash
# Gera PDFs (e PDF/A) do paper arXiv + ESPEC para o deposito Zenodo.
# Uso: ./docs/paper/build-pdf.sh [outdir]   (padrao: /tmp)
# Requer: pandoc + texlive-xetex (PDF), ghostscript (PDF/A). Sem eles,
# o script diz exatamente o que instalar e sai sem fingir resultado.
set -euo pipefail
OUT="${1:-/tmp}"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

need() { command -v "$1" >/dev/null 2>&1 || { echo "FALTA: $1 — instale com: sudo apt install $2"; MISSING=1; }; }
MISSING=0
need pandoc pandoc
need xelatex texlive-xetex
need gs ghostscript
[ "$MISSING" -ne 0 ] && { echo "Instale acima e rode de novo. Nada foi gerado."; exit 1; }

build() { # $1=md $2=pdf-base $3=title
  echo "== $1 -> $2.pdf"
  pandoc "$1" -o "$OUT/$2.pdf" --pdf-engine=xelatex \
    -V colorlinks:true -V geometry:margin=2.5cm \
    --metadata title="$3" \
    --metadata author="Matheus de Camargo Marques"
  echo "== $2.pdf -> $2-PDFA.pdf (PDF/A-2b)"
  gs -dPDFA=2 -dBATCH -dNOPAUSE -sProcessColorModel=DeviceRGB \
     -sDEVICE=pdfwrite -sPDFACompatibilityPolicy=1 \
     -sOutputFile="$OUT/$2-PDFA.pdf" "$OUT/$2.pdf" >/dev/null
  ls -la "$OUT/$2.pdf" "$OUT/$2-PDFA.pdf"
}

build docs/paper/arxiv-m3avm-draft.md m3avm-paper "M3-AVM: Sparse Event-Driven Tensor VM (draft)"
build docs/ESPEC.md m3avm-espec-v1.4 "M3-AVM Specification v1.4"

if command -v verapdf >/dev/null 2>&1; then
  verapdf --format text "$OUT/m3avm-paper-PDFA.pdf" | tail -3
else
  echo "(veraPDF ausente — valide em https://demo.verapdf.org antes de subir no Zenodo)"
fi
echo "OK em $OUT"
