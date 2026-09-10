# CLAIM 6 REVISADA — orquestração multi-modelo em contexto único (pós-busca 2026-09-10)

> Minuta do assistente técnico. Diferenciais inseridos a partir dos achados do inventor:
> StreamWise (multi-modal sem mesmo-endereço), DeltaBox (CoW em SO, não em ISA de tensores),
> vLLM CoW fork (mono-modelo), Triton/Ray (multi-processo). SNN removida do executado (RSVD 0x20).
> Requer validação do agente de PI (unidade, clareza art. 25, suporte em `01_RELATORIO` + `05_EVIDENCIAS`).

## Reivindicação 6.1 (Independente — Método, C1 habilitado)

**Método para orquestração de múltiplos modelos de inteligência artificial heterogêneos em contexto único de máquina virtual**, executado em um único processo computacional, **caracterizado por** compreender:

(a) carregar, na mesma máquina virtual de instruções de 32 bytes fixos, pelo menos um modelo Transformer com KV cache por camada, um modelo de espaço de estados com estado recorrente `h_t` e um codec de áudio com janelas temporais, cada qual com pesos residentes em região persistente `mmap` e estado mutável em região de cache versionada, todos endereçáveis no mesmo espaço virtual de 128 bits;

(b) executar, por comutação de pipeline com fence (`CTX_SWITCH`), sem troca de contexto e sem serialização inter-processos, operações nativas do referido Transformer (`ATTN/FFN/NORM/ROPE/MATVEC`), do referido modelo de espaço de estados (`SSM_SCAN/SSM_RESET`) e do referido codec (`CODEC_ENC/CODEC_DEC/AUDIO_ALIGN`), além de indexação indireta (`GATHER`), distância em lote com top-k (`DISTANCE`) e recorrência matricial CoW (`RANK1_UPDATE`);

(c) capturar, por instrução única de snapshot (`FORK`) com semântica copy-on-write por contagem de referências sob contador de versão monotônico e janela de retenção limitada, o **estado conjunto** compreendendo o referido KV cache, o referido `h_t` e a referida matriz recorrente;

(d) restaurar, por instrução única de rollback (`ABORT`), o referido estado conjunto de forma **atômica por troca de ponteiros**, com descarte de ramificações especulativas e truncamento de KV (`KV_TRUNCATE`), de modo que o prefixo confirmado seja retomado bit-exato no mesmo host/build;

(e) escalonar os referidos modelos por deadline absoluto por contexto (`SET_DEADLINE`) sob prioridade estrita de três níveis com preempção por flag atômica, em que interrupção de áudio (`SENSE USER_INPUT → IF_INTERRUPT`) desvia para o referido rollback e reinjeção de prompt no checkpoint.

## Por que esta redação (mapeamento achado → limitação)

| Achado | Onde foi neutralizado |
|---|---|
| StreamWise: multi-modal sem mesmo-endereço, com fila deadline-aware | (a)+(b): mesmo espaço 128 bits + sem IPC/gRPC; (e): deadline por contexto na VM, não fila de serving |
| DeltaBox: CoW de sandbox (arquivos+processos, overlay/CRIU) | (c)+(d): CoW de **tensores heterogêneos residentes** via ISA, com I-Mono + janela, sem FS/processo |
| vLLM #4907 / TRT-LLM #13453: fork/replay mono-modelo | (c)+(d): **conjunto transversal** KV+`h_t`+`H` em ato único + `KV_TRUNCATE` |
| Triton/Ray: backends isolados, sequence-batcher por correlation ID | (b): `CTX_SWITCH` sem troca de contexto; estado intermediário diretamente acessível entre motores |
| PHAROS/EDF-AoT (risco 3): EDF heterogêneo | (e) deliberadamente **sem EDF composto** (fica na divisional); aqui só deadline absoluto + 3 níveis — matéria habilitada |
| SNN no texto original | Removida do executado; `SPIKE_STEP 0x20` citado como extensão C2 na dependente |

## Dependente sugerida (C2, divisional)

**6.2.** Método conforme 6.1, em que a pluralidade compreende ainda rede SNN (`SPIKE_STEP`), retrieval (`RAG_SEARCH`) e transferência cross-model residente descartável em `ABORT`, e em que deadlines compõem-se em árvore `AND/OR/N_OF_M/CHAIN` com herança cross-device — matéria de pedidos divisionais condicionados a RFC+testes+Lean.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
