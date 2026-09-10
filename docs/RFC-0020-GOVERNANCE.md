# RFC-0020 — Governança W0 + Baseline

```text
Status      : IMPLEMENTED (process)
Category    : Standards Track
Updates     : this plan; ESPEC §19; RFC template adoption
Obsoletes   : None
Feature Bit : none (process only)
Bump        : none
```

## Abstract

Adota este documento como a ordem normativa-draft que sequencia as RFCs e waves do M³-AVM a partir da baseline v1.5 (256 testes verdes + 1 falha pré-existente). Congela a ordem Fase 0→Fase 10 e o template RFC para todas as subsequentes.

## Motivation

Sem uma ordem explícita, as RFCs futuras podem ser escritas em qualquer sequência, criar sobreposições, ou ignorar gates (TRilha P, prova R12, lean). Este RFC freezes a roadmap atual como referência para todos os contribuintes e para o Zenodo release.

## Specification

### 1. Adoção de template RFC

Todo novo RFC deve seguir o template da seção 6.4 de [R] (R11), estendendo ESPEC §19, com os seguintes campos obrigatórios:

- `Status`: IMPLEMENTED / DRAFT / HELD / RESERVED
- `Category`: Standards Track / Informational
- `Updates`: referência(s) a ESPEC § e a wave(s) do plano
- `Feature Bit`: None / u64 bit assignment
- `Bump`: none / MINOR / MAJOR (regra: minor por família 32B a partir de v1.5; next MINOR = v1.6)
- `Changelog`: modelo `| Version | Date | Changes |`

### 2. Baseline de testes

Baseline congelada para todas as fases subsequentes:

- `cargo test --lib`: **256 passed** + **1 failed** (`moshi::test_gguf_qkv_split_shapes`, norm-gamma assertion, issue de modelo-data, não relacionado ao spec — aceito como baseline)
- Tag de referência: `v0.1.0` (ISA v1.5, RFCs 0002–0019 IMPLEMENTED)
- `PROVENANCE.txt` gerado por `./docs/zenodo/pack.sh v0.1.0` contendo últimos 5 commits + resultados de teste

### 3. Ordem de phases (freeze)

A seguinte ordem é normativa-draft para todas as RFCs subsequentes:

```
Fase 0: Governança W0 + baseline (RFC-0020) — CONCLUÍDA
Fase 1: Infra 64B + container (RFC-0021/0022) — W1-resto + W10-loader
Fase 2: Arena/memória 32B (RFC-0023/0024) — W6
Fase 3: Tensores avançados + attention (RFC-0025/0026/0027) — W3+W4
Fase 4: Conversão/precisão (RFC-0028/0029) — §11 + gap 0x7A-0x7F
Fase 5: Áudio full-duplex (RFC-0030/0031) — W8
Trilha P: Provas R12 + HELD (paralela a Fases 2–6)
Fase 7: Cluster X-forms + transporte F2–F5 (RFC-0034/0035/0036) — W9 + §12
Fase 8: Scheduler T2 + robustez (RFC-0037/0038) — follow-ups + §18
Fase 9: System/MMIO/ESCAPE (RFC-0039/0040) — W10-resto + §9
Fase 10: Polimento + provas restantes + freeze v2.0 (RFC-0041)
```

### 4. Versionamento

- Famílias 32B = **minor bumps** (v1.6, v1.7… a partir da baseline v1.5)
- Work 64B entra como *"v2.0-zone early implementation"* (regra RFC-0005/0006), sem bump até o freeze v2.0 na Fase 10
- Cada RFC deve declarar seu bump na header; mudanças que tocam encodings shipped exigem consenso

### 5. Gate de aceitação por RFC (padrão uniforme)

Todo RFC deve passar por esta pipeline antes de ser marcado IMPLEMENTED:

1. Doc RFC (template R11) com template R11 preenchido
2. Consts/ctors/parse/roundtrip/dispatch/counters
3. Goldens + error paths + OOB-determinismo
4. Demo `.m3asm` exec-verificada
5. Corpus gate (programas existentes continuam a montar)
6. Suite vs baseline (regressão `cargo test --lib`)
7. Bench criterion (medianas via `step_instruction`, output alloc incluído)
8. Lean delta onde bounds movem (se applicable)
9. ESPEC §5+§16+README atualizados
10. Tag + `pack.sh` + release Zenodo

## Changelog

| Version | Date | Changes |
|:---|:---|:---|
| 0020-00 | YYYY-MM-DD | DRAFT: governance + baseline freeze + template RFC adoption |
| 0020-01 | YYYY-MM-DD | IMPLEMENTED: merged to tree, suite green, baseline registrada |

---

*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258)*