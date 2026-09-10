#!/usr/bin/env bash
# Empacota release Zenodo do M³-AVM (exclui binários/gerados).
# Uso: ./docs/zenodo/pack.sh [VERSAO]  (ex.: ./docs/zenodo/pack.sh v0.1.0)
set -euo pipefail
VER="${1:-v0.1.0}"
DATE="$(date +%F)"
OUT="/tmp/m3_avm-${VER}-zenodo-${DATE}.zip"
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
git log --oneline -5 > /tmp/PROVENANCE.txt
(cargo test --lib 2>&1 | tail -5 >> /tmp/PROVENANCE.txt) || true
cp /tmp/PROVENANCE.txt ./PROVENANCE.txt
zip -r "$OUT" \
  README.md CITATION.cff LICENSE .zenodo.json AUTHORS PROVENANCE.txt \
  Cargo.toml Cargo.lock \
  src examples programs benches tests formal \
  docs/ESPEC.md docs/ESPEC-V2.md docs/RFC-*.md docs/zenodo docs/patente docs/paper \
  -x "formal/.lake/*" "target/*" "*.dat" "*.gguf" "whisper.cpp/*"
rm ./PROVENANCE.txt
echo "OK: $OUT"
unzip -l "$OUT" | tail -8
