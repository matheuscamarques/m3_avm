# QUADRO REIVINDICATÓRIO (minuta — revisar com agente de PI)

> Convenção: **(C1)** = núcleo habilitado (suficiência em `docs/ESPEC.md` +
> código); **(C2)** = modalidade futura (codificação congelada, execução
> pendente de RFC + testes + Lean, cf. `docs/ESPEC-V2.md §14`). O agente deve
> avaliar unidade de invenção e eventual divisão (ver `01_RELATORIO §8.2`).
> Preâmbulos seguem arts. 25 LPI (clareza, concisão, suporte no relatório).

## Reivindicações — Método

### Reivindicação 1 (Independente — Método, C1)

**Método para orquestração de múltiplos modelos de inteligência artificial
heterogêneos em contexto único de máquina virtual**, executado em um único
processo computacional sobre hardware convencional, **caracterizado por**
compreender:

(a) carregar uma pluralidade de modelos, incluindo pelo menos um modelo
Transformer e um modelo de espaço de estados (SSM), em uma máquina virtual de
instruções de **32 bytes fixos** (`opcode 1B | flags 1B | 4 regs 4B | payload
26B little-endian`), com **quatro regiões tipadas** — GLOBAL, TEMPORAL
circular, PERSISTENTE `mmap` e KV_CACHE por camada — em espaço virtual de
128 bits;

(b) manter um contexto computacional com 16 registradores, contador de
programa, prioridade, flags de comparação/interrupto, pipeline e deadline
absoluto, compartilhando as regiões entre todos os modelos carregados;

(c) executar, pelo conjunto de instruções, operações nativas compreendendo
`ATTN`, `FFN`, `NORM`, `ROPE`, `MATVEC`, `MUL`, `SILU` (Transformer),
`SSM_SCAN`, `SSM_RESET` (SSM) e `CODEC_ENC`, `CODEC_DEC`, `AUDIO_ALIGN`
(áudio), comutando entre motores por instrução de comutação com fence
(`CTX_SWITCH`) sem troca de contexto;

(d) capturar, por instrução de snapshot (`FORK`), estado conjunto CoW dos
modelos ativos, compreendendo KV cache de Transformer e estado recorrente de
SSM, sob contador de versão monotônico e janela de retenção limitada;

(e) restaurar, por instrução de rollback (`ABORT`), o estado capturado de
forma atômica por troca de ponteiros;

(f) associar a cada contexto bloqueável um deadline absoluto
(`SET_DEADLINE`/`GET_DEADLINE`) e escalonar por **prioridade estrita
Red > Blue > Green** com preempção por flag atômica.

### Reivindicação 2 (Dependente de 1 — C1)

Método conforme 1, **caracterizado por** a captura usar copy-on-write por
contagem de referências, sem cópia física até escrita, com janela deslizante
(padrão 16, opt-out explícito) e erro limpo em restore de snapshot evictado.

### Reivindicação 3 (Dependente de 1 — C1)

Método conforme 1, **caracterizado por** a comutação entre motores compreender
instrução que ajusta pipeline, prioridade, fence de memória, snapshot e
reavaliação do escalonador em ato único.

### Reivindicação 4 (Dependente de 1 — C1)

Método conforme 1, **caracterizado por** compreender ainda indexação indireta
por tensor (`GATHER` com modos gather/scatter-add/scatter-max e erro explícito
em out-of-bounds), distância em lote com top-k fundido (`DISTANCE`, 4 métricas)
e recorrência matricial por produto externo com semântica CoW
(`RANK1_UPDATE`), e modo top-k de amostragem (`SAMPLE TOPK`) para despacho.

### Reivindicação 5 (Dependente de 1 — C1)

Método conforme 1, **caracterizado por** compreender higiene rolante de KV
(`KV_TRUNCATE` por comprimento com porta de stream) e reinicialização de
estado SSM (`SSM_RESET`), combináveis ao rollback para sincronizar prefixos
após rejeição especulativa.

### Reivindicação 6 (Dependente de 1 — C1)

Método conforme 1, **caracterizado por** a reprodutibilidade compreender RNG
semeado por contexto (`RNG_SEED/NEXT/NORMAL/UNIFORM`), hashing/checksum
(`HASH/CHECKSUM/HMAC` truncado) e telemetria com traps (`CYCLES_COUNT,
TRACE_EVENT, SANITY_CHECK, PREEMPT_CHECK, ASSERT, DUMP, YIELD`),
e imediatos (`LOADI/MOV`) com predicados de comparação estendidos.

### Reivindicação 7 (Dependente de 1 — C1)

Método conforme 1, **caracterizado por** decodificador exigir 32 bytes,
registradores `0..15` ou `0xFF`, inteiros little-endian em offsets fixos,
encodings imutáveis e assembler de 2 passes com labels e alvos validados.

### Reivindicação 8 (Dependente de 1 — C2, modalidade futura)

Método conforme 1, **caracterizado por** compreender ainda composição de
deadlines em árvore (`AND`, `OR`, `N_DE_M`, `CHAIN`) sobre deadlines absolutos,
com propagação de violação e relaxamento/migração, escalonamento
earliest-deadline-first com herança de deadline e timeouts obrigatórios.

### Reivindicação 9 (Dependente de 8 — C2)

Método conforme 8, **caracterizado por** afinidade de dispositivo e NUMA por
contexto, cálculo de prioridade efetiva por (deadline, localidade, base),
herança cross-device com corte anti-deadlock e backpressure de staging a 80%.

### Reivindicação 10 (Dependente de 1 — C2, mecanismo de runtime)

Método conforme 1, **caracterizado por** instrução nativa de transferência de
estado que, a partir de KV/estado fonte **já residente** nas regiões tipadas,
aplica mapeamento pré-calibrado para produzir KV/estado alvo residente, em que
o estado transferido permanece **descartável até commit** e é descartado por
`ABORT` que restaure o snapshot anterior, e em que a transferência alimenta
`ATTN` podada via `DISTANCE` top-k; o método matemático de regressão em si
(ridge/MLP, RoPE-strip, calibração) não é reivindicado isoladamente.

## Reivindicações — Sistema

### Reivindicação 11 (Independente — Sistema, C1)

**Sistema para orquestração de múltiplos modelos de IA heterogêneos em
contexto único**, **caracterizado por** compreender processador e memória
configurados para executar a máquina virtual do método da reivindicação 1,
com as quatro regiões tipadas, o conjunto de instruções de 32 bytes, o
mecanismo `FORK/ABORT` CoW com contador monotônico e o escalonador por
prioridade estrita com deadline absoluto.

### Reivindicação 12 (Dependente de 11 — C2)

Sistema conforme 11, **caracterizado por** compreender ainda módulo de
descoberta de hardware que popula registro de dispositivos (GPUs/CPUs/NPUs,
barramentos, topologia) e formas estendidas de 64 bytes com campos
`LAMPORT/DEADLINE/PAYLOAD_EXT` para endereçar 16 regiões lógicas.

### Reivindicação 13 (Dependente de 11 — C2)

Sistema conforme 11, **caracterizado por** compreender ainda módulo de cluster
com opcodes básicos (`REMOTE_SPAWN/SIGNAL/SEND_TENSOR/BARRIER`), formas-X,
migração com write-ahead log retido até ACK + janela, timeouts e filas de
urgência sobre bulk, e máquina de suspeita de nós.

### Reivindicação 14 (Dependente de 11 — C2)

Sistema conforme 11, **caracterizado por** compreender ainda módulo de segurança
com cookie de comparação constante no handshake e tabela de capacidades com
falha em extensão requerida desconhecida; proteção de transporte forte
(mTLS/ChaCha) como evolução prevista, não afirmada como pronta.

### Reivindicação 15 (Dependente de 11 — C1 parcial + C2)

Sistema conforme 11, **caracterizado por** compreender ainda módulo de
observabilidade com eventos estruturados locais (`TRACE_EVENT`) e contadores,
evoluindo a traços distribuídos (W3C Trace Context/OTLP) na modalidade C2.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
