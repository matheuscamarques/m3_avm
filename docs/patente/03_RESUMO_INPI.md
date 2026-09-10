# RESUMO (padrão INPI — máx. 150 palavras)

> Minuta com 83 palavras no corpo (conferir contagem na versão protocolada).
> Título repetido + essência + uso. Sem jargão de marketing, sem números
> não medidos.

**SISTEMA E MÉTODO PARA ORQUESTRAÇÃO DE MÚLTIPLOS MODELOS DE INTELIGÊNCIA ARTIFICIAL HETEROGÊNEOS EM CONTEXTO ÚNICO DE MÁQUINA VIRTUAL COM ESTADO COMPARTILHADO E ROLLBACK CONJUNTO**

Máquina virtual em processo único executa modelos heterogêneos (Transformer, SSM, áudio, retrieval) no mesmo espaço de endereço, com quatro regiões tipadas e instruções nativas de 32 bytes. Instrução de comutação alterna motores sem trocar contexto. Snapshot copy-on-write captura KV cache e estados recorrentes; rollback restaura atomicamente por troca de ponteiros, com contador monotônico. Cada contexto tem deadline absoluto e prioridade estrita com preempção. Extensão prevê deadlines compostos, escalonamento topológico e transferência de estado entre modelos. Aplicação: pipelines voz-RAG-decisão-voz com interrupção e retomada bit-exata.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
