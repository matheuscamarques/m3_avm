> **ARQUIVO — material de entrada, NÃO normativo.**
> Recebido como "mais dados" em 2026-09-10. A especificação canônica é
> `docs/ESPEC.md` (ISA v1.3 + modelo Σ). Este documento contradiz o
> repositório e, em pontos, o próprio corpus v2.0 (ver revisão na conversa
> de trabalho: conflito `0xB0–0xB6`, `R255`, região `0xF`, `ISA_V2.md`
> inexistente). Nada aqui deve ser implementado sem reconciliação.
> Conteúdo original preservado verbatim abaixo.
>
> **Reconciliação:** ver `docs/ESPEC-V2.md` (R3 ESCAPE `0xB0-0xB6` mantido,
> federated movido p/ `0xB7-0xBF`; R10 regra de farejo legado; R11
> governança adotada; §11 deste RFC sem efeito até os arquivos existirem).
>
> ---

# RFC-0001 — Mecanismo de Extensão da M³-AVM ISA

```
RFC Number  : 0001
Title       : Extension Mechanism for M³-AVM ISA
Status      : PROPOSED
Category    : Standards Track
Author      : Matheus de Camargo Marques
Date        : 2026-09-09
Updates     : ISA v2.0 (doc/ISA_V2.md)
Obsoletes   : None
```

---

## Abstract

Este RFC especifica o **mecanismo formal de extensão** da ISA M³-AVM que permite evolução por **50 anos sem quebra binária**. Define:

1. O opcode `ESCAPE` (`0xB0–0xB6`) como válvula de expansão infinita
2. O **cabeçalho de bytecode versionado** de 32 bytes
3. A **indireção de bancos de registradores e regiões de memória**
4. O **processo formal** de proposta, revisão e ratificação de extensões

Este documento é normativo. Qualquer implementação conforme à ISA v2.0 **deve** respeitar as reservas aqui definidas, mesmo que não implemente os mecanismos ainda.

---

## 1. Motivação

A ISA M³-AVM v2.0 (doc/ISA_V2.md) cobre ~180 opcodes dos 256 disponíveis. Análise crítica identificou **quatro fraturas** que forçariam reescrita em 15–25 anos:

| Fratura | Causa raiz |
|:---|:---|
| F1 | Seletor de região de 4 bits (16 regiões fixas) |
| F2 | `PAYLOAD_CORE` de 32 bytes fixos |
| F3 | `LAMPORT` de 64 bits |
| F4 | Ausência de mecanismo de negociação de versão |
| F5 | Ausência de mecanismo de expansão de formato |

A lição histórica de IBM z/Architecture (60 anos), x86 (47 anos), ARM (40 anos) e BEAM (40 anos) é inequívoca: **ISAs longevos não duram por serem completos, mas por terem mecanismos explícitos de escape**.

Este RFC fornece esses mecanismos.

---

## 2. Convenções e Terminologia

- **MAY / MUST / SHOULD / MUST NOT** seguem RFC 2119
- **Bytecode** = arquivo compilado no formato `.m3bc`
- **Runtime** = implementação da VM que executa bytecode
- **Extensão** = qualquer adição de opcode, formato, registro ou região
- **Conformidade** = implementação que respeita as reservas deste RFC, mesmo sem implementar os mecanismos
- **Feature bit** = bit em `REQUIRED_FEATURES` ou `OPTIONAL_FEATURES` do cabeçalho

---

## 3. Opcodes de ESCAPE (`0xB0–0xB6`)

### 3.1 Alocação

| Opcode | Nome | Formato | Propósito |
|:--:|:---|:---|:---|
| `0xB0` | `ESCAPE` | 64B | Prefixo para extensão de 64B |
| `0xB1` | `ESCAPE_128` | 128B | Extensão de 128B |
| `0xB2` | `ESCAPE_256` | 256B | Extensão de 256B |
| `0xB3` | `ESCAPE_VAR` | variável | Extensão com `ext_len` auto-descritivo |
| `0xB4` | `VERSION` | 64B | Declara versão em runtime |
| `0xB5` | `CAPABILITY_QUERY` | 64B | Pergunta capacidades do runtime |
| `0xB6` | `CAPABILITY_ASSERT` | 64B | Exige capacidade ou falha |

**Estes opcodes são reservados nesta especificação e MUST NOT ser usados para outro propósito.**

### 3.2 Layout de `ESCAPE` (`0xB0`)

```
Byte 0      : 0xB0 (ESCAPE)
Byte 1      : ext_opcode (u8) — novo espaço de 256 opcodes
Bytes 2-3   : ext_flags (u16) — flags específicos da extensão
Bytes 4-7   : ext_version (u16 major, u16 minor) — versão da extensão
Bytes 8-15  : ext_len (u64 LE) — tamanho total da instrução (≥ 64, múltiplo de 8)
Bytes 16-23 : ext_reserved (u64) — reservado, MUST be zero
Bytes 24-31 : ext_checksum (u64) — checksum xxHash64 dos bytes 0..23 + payload
Bytes 32-63 : ext_payload_head (32 B) — primeiros 32B do payload
Bytes 64-…  : ext_payload_tail (ext_len - 64 bytes) — continuação
```

### 3.3 Semântica

- Um runtime que **não** reconhece `ext_opcode` MUST:
  - Se `ext_flags.bit0 = 0` (obrigatório): falhar com `UnsupportedExtension`
  - Se `ext_flags.bit0 = 1` (opcional): pular `ext_len` bytes e continuar
- Um runtime que **reconhece** `ext_opcode` MUST validar `ext_checksum` antes de executar
- `ext_version` permite múltiplas versões do mesmo `ext_opcode` coexistirem

### 3.4 Exemplo: extensão fotônica em 2040

```
0xB0              ; ESCAPE
0x42              ; ext_opcode = PHOTONIC_MATMUL
0x01 0x00         ; ext_flags = bit0=optional
0x01 0x00 0x00 0x00 ; ext_version = 1.0
0x00 0x01 0x00 0x00 0x00 0x00 0x00 0x00 ; ext_len = 256
...               ; payload fotônico
```

Runtime de 2026: pula 256 bytes, continua.  
Runtime de 2040: executa.

---

## 4. Cabeçalho de Bytecode Versionado

### 4.1 Formato

Todo arquivo `.m3bc` MUST começar com um cabeçalho de 32 bytes:

```
Bytes 0-3   : MAGIC "M3BC" (0x4D 0x33 0x42 0x43)
Byte 4      : ISA_MAJOR (u8) — incrementa em quebra binária
Byte 5      : ISA_MINOR (u8) — incrementa em adição de opcode
Bytes 6-7   : ISA_PATCH (u16) — incrementa em correção semântica
Bytes 8-15  : REQUIRED_FEATURES (u64 LE) — bitmask de features obrigatórias
Bytes 16-23 : OPTIONAL_FEATURES (u64 LE) — bitmask de features opcionais
Bytes 24-27 : ENTRY_PC (u32 LE) — offset do ponto de entrada
Bytes 28-31 : CHECKSUM (u32 LE) — CRC32 do resto do arquivo
```

### 4.2 Negociação

Um runtime MUST:

1. Verificar `MAGIC` == `"M3BC"`. Se falhar, erro `InvalidBytecode`.
2. Verificar `ISA_MAJOR` ≤ runtime major. Se maior, erro `IncompatibleMajor`.
3. Verificar `REQUIRED_FEATURES & ~runtime_features == 0`. Se falhar, erro `MissingRequiredFeature`.
4. `OPTIONAL_FEATURES` podem estar ausentes; runtime deve degradar graciosamente.
5. Verificar `CHECKSUM`. Se falhar, erro `CorruptedBytecode`.

### 4.3 Registro de Features

Features são registradas neste RFC (§7). Cada feature tem um bit fixo. Bits 0–31 reservados para features core; bits 32–63 para features de extensão.

### 4.4 Exemplo

```
"M3BC"               ; MAGIC
0x02                 ; ISA_MAJOR = 2
0x00                 ; ISA_MINOR = 0
0x00 0x00            ; ISA_PATCH = 0
0x01 0x00 0x00 0x00 0x00 0x00 0x00 0x00 ; REQUIRED = ESCAPE (bit0)
0x02 0x00 0x00 0x00 0x00 0x00 0x00 0x00 ; OPTIONAL = PHOTONIC (bit1)
0x20 0x00 0x00 0x00  ; ENTRY_PC = 32
0xDE 0xAD 0xBE 0xEF  ; CHECKSUM
```

Runtime de 2026 sem suporte a `PHOTONIC`: executa com degradação (se o bytecode usar `ESCAPE` opcional, pula).  
Runtime de 2040 com suporte: executa acelerado.

---

## 5. Indireção de Registradores e Regiões

### 5.1 Banco de registradores indireto

O registrador `R255` é **reservado** como **Register File Handle (RFH)**:

| Valor de `R255` | Banco ativo |
|:--:|:---|
| `0` | Banco 0 (padrão: GPR0–254) |
| `1` | Banco 1 (reservado para VEC estendido) |
| `2` | Banco 2 (reservado para Tensor Registers) |
| `3` | Banco 3 (reservado para Distributed State) |
| `4–255` | Reservado |

**Runtime MUST** tratar `R255 == 0` como o comportamento atual (compatibilidade v2.0).  
**Runtime MAY** implementar bancos 1–3; se não implementar, acesso a eles MUST falhar com `UnsupportedRegisterBank`.

Quando `R255 != 0`, a interpretação de `R0–R254` muda para o banco ativo. Isso permite **até 255 × 255 = 65.025 registradores** sem mudar o encoding.

### 5.2 Tabela de regiões indireta

A região `0xF` (MMIO) é **reservada** para conter a **Region Table**:

```
Offset 0    : N_REGIONS (u32 LE) — número de regiões ativas
Offset 4    : RESERVED (u32)
Offset 8    : REGIONS[N] — tabela de N entradas de 16 bytes cada
              Cada entrada:
                Bytes 0-7   : BASE (u64)
                Byte 8      : SIZE_LOG2 (u8) — tamanho = 2^SIZE_LOG2
                Byte 9      : FLAGS (u8) — RO/RW/CoW/MMIO
                Bytes 10-11 : REGION_ID_ALIAS (u16)
                Bytes 12-15 : RESERVED (u32)
```

**Runtime MUST** ler a Region Table no boot. Se `N_REGIONS == 0`, usa as 16 regiões padrão (compatibilidade v2.0). Se `N_REGIONS > 0`, a tabela **substitui** as regiões padrão.

Isso permite **até 65.535 regiões** em 2050, sem re-encodar endereços (o seletor de 4 bits passa a apontar para entradas da tabela).

### 5.3 Exemplo: adicionar região fotônica em 2040

```
; No bytecode:
0xF000_0000_0000_0000 + offset 0 : N_REGIONS = 17
0xF000_0000_0000_0008 + 16*16   : BASE = 0x1000_0000_0000_0000
                                   SIZE_LOG2 = 40 (1 TiB)
                                   FLAGS = RW
                                   REGION_ID_ALIAS = 0x10 (nova)
```

Runtime de 2026 sem suporte: ignora; usa as 16 padrão.  
Runtime de 2040: acessa `0x10` como região fotônica.

---

## 6. Processo de Extensão

### 6.1 Tipos de extensão

| Tipo | Mecanismo | Aprovação |
|:---|:---|:---|
| **Opcode novo em faixa livre** | Bump minor (`v2.0 → v2.1`) | RFC Standard Track |
| **Flag nova em opcode existente** | Documentação + bump patch | RFC Informational |
| **Modo novo em payload** | Documentação + bump patch | RFC Informational |
| **ESCAPE opcode novo** | Bump minor | RFC Standard Track |
| **Nova região de memória** | Region Table + bump minor | RFC Standard Track |
| **Nova feature de negociação** | Registro em §7 + bump minor | RFC Standard Track |
| **Mudança de formato de instrução** | Bump **major** (quebra) | RFC Historic |

### 6.2 Ciclo de vida de um RFC

```
DRAFT → PROPOSED → ACCEPTED → IMPLEMENTED → DEPRECATED → OBSOLETED
```

- **DRAFT**: autor propõe; sem compromisso
- **PROPOSED**: revisado por ≥2 membros; sem implementação
- **ACCEPTED**: aprovado; encoding/feature reservado
- **IMPLEMENTED**: código no repo; testes verdes
- **DEPRECATED**: não recomendado para novo código
- **OBSOLETED**: substituído por RFC posterior

### 6.3 Regras invioláveis

1. **Encodings nunca são reutilizados.** Um opcode deprecado permanece reservado.
2. **Features nunca são removidas.** Bit 0 de uma feature permanece para sempre.
3. **Regiões nunca são removidas.** Region Table pode marcar como `DEPRECATED`, mas acesso MUST falhar limpo.
4. **Bits de `REQUIRED_FEATURES` nunca são reutilizados.** Se feature morre, bit vira `RESERVED`.
5. **Mudança de layout de instrução é sempre quebra major.** Se `ISA_MAJOR` muda, runtime antigo MUST rejeitar.

### 6.4 Template de RFC

Todo RFC de extensão MUST conter:

```markdown
# RFC-NNNN — Título
Status      : DRAFT | PROPOSED | ACCEPTED | IMPLEMENTED | DEPRECATED | OBSOLETED
Category    : Standards Track | Informational | Historic
Updates     : (RFC ou ISA)
Obsoletes   : (RFC ou ISA ou None)
Feature Bit : N (se aplicável)
Bump        : MAJOR | MINOR | PATCH

## Abstract
## Motivation
## Specification
## Backwards Compatibility
## Security Considerations
## Reference Implementation (link ao PR)
## Changelog
```

---

## 7. Registro de Features

Features são bitmask de 64 bits. Bits 0–31 core; 32–63 extensão.

### 7.1 Core (bits 0–31)

| Bit | Nome | Descrição | Introduzido |
|:--:|:---|:---|:---|
| 0 | `ESCAPE` | Suporte a `ESCAPE` | v2.0 |
| 1 | `DUAL_MODE` | Suporte a 32B/64B | v2.0 |
| 2 | `LAMPORT` | Suporte a relógio lógico | v2.0 |
| 3 | `DEADLINE` | Suporte a EDF | v2.0 |
| 4 | `WAL_MIGRATE` | Suporte a `MIGRATE` com WAL | v2.0 |
| 5 | `COW_SNAPSHOT` | Suporte a snapshot CoW | v2.0 |
| 6 | `CLUSTER` | Suporte a cluster | v2.0 |
| 7 | `GPU_MMIO` | Suporte a MMIO GPU | v2.0 |
| 8 | `TENSOR_REGS` | Suporte a Tensor Registers | futuro |
| 9 | `REGION_TABLE` | Suporte a Region Table | futuro |
| 10–31 | RESERVED | Reservado | — |

### 7.2 Extensão (bits 32–63)

| Bit | Nome | Descrição | Introduzido |
|:--:|:---|:---|:---|
| 32 | `PHOTONIC` | Acelerador fotônico | futuro |
| 33 | `NEUROMORPHIC` | Acelerador neuromórfico | futuro |
| 34 | `QUANTUM_SIM` | Simulador quântico | futuro |
| 35 | `FEDERATED` | Aprendizado federado | futuro |
| 36 | `CONFIDENTIAL` | TEE/enclave | futuro |
| 37–63 | RESERVED | Reservado | — |

**Features em §7.2 MUST NOT ser implementadas nesta versão da ISA.** São reservas para RFCs futuros.

---

## 8. Compatibilidade

### 8.1 Backwards compatibility

Um runtime v2.0 que implementa este RFC MUST:

- Executar bytecode v1.3 (ISA_MAJOR=1) sem alteração
- Executar bytecode v2.0 (ISA_MAJOR=2) com ou sem `ESCAPE`
- Rejeitar bytecode v3.0+ com `IncompatibleMajor`

### 8.2 Forward compatibility

Um runtime v2.0 que implementa este RFC SHOULD:

- Pular `ESCAPE` opcional desconhecido (bit0=1)
- Falhar limpo em `ESCAPE` obrigatório desconhecido (bit0=0)
- Ignorar bits de `OPTIONAL_FEATURES` que não implementa
- Reportar `CAPABILITY_QUERY` com bitmask correto

### 8.3 Teste de conformidade

Um runtime é **conforme a RFC-0001** se passa em:

1. `test_escape_optional_skip` — pula ESCAPE opcional desconhecido
2. `test_escape_required_fail` — falha em ESCAPE obrigatório desconhecido
3. `test_header_negotiation` — negocia header de bytecode
4. `test_rfh_bank_switch` — troca banco de registradores
5. `test_region_table_override` — usa Region Table customizada
6. `test_feature_bit_reserved` — rejeita feature bit desconhecida obrigatória
7. `test_isa_major_mismatch` — rejeita major maior

---

## 9. Security Considerations

### 9.1 ESCAPE como vetor de ataque

`ESCAPE` permite payload arbitrário. Runtime MUST:

- Validar `ext_len` (≤ 1 MiB; se maior, erro `PayloadTooLarge`)
- Validar `ext_checksum`
- Limitar número de `ESCAPE` encadeados (≤ 1; sem recursão)
- Não executar `ESCAPE` de região `SHARED` sem `FENCE`

### 9.1 Region Table como vetor

Region Table permite mapear regiões arbitrárias. Runtime MUST:

- Validar `N_REGIONS` (≤ 65.535)
- Validar `BASE` alinhado a 4 KiB
- Validar `SIZE_LOG2` (≤ 63)
- Rejeitar `REGION_ID_ALIAS` conflitante
- Rejeitar `FLAGS` inconsistentes (RO + CoW)

### 9.3 RFH como vetor

Troca de banco de registradores pode expor estado de outro contexto. Runtime MUST:

- Isolar bancos por contexto
- Validar `R255` a cada instrução
- Rejeitar acesso cross-context a bancos 1–3
- Logar troca de banco (auditoria)

### 9.4 Header como vetor

Header de bytecode pode ser forjado. Runtime MUST:

- Validar `CHECKSUM` **antes** de executar qualquer instrução
- Validar `MAGIC`
- Nunca executar bytecode sem header válido (bytecode sem header é inválido)

---

## 10. IANA Considerations

Este RFC estabelece três registros:

| Registro | Faixa | RFC que atualiza |
|:---|:---|:---|
| **M³-AVM Opcode Registry** | `0x00–0xFF` | Este RFC + doc/ISA_V2.md |
| **M³-AVM Feature Registry** | 64 bits | §7 deste RFC |
| **M³-AVM Region Registry** | 16 bits (Region Table) | §5.2 deste RFC |

Novas entradas são adicionadas via RFC Standard Track. Entradas nunca são removidas.

---

## 11. Reference Implementation

A implementação de referência está em:

- `src/opcodes.rs::decode_escape` — decodificação de `ESCAPE`
- `src/bytecode.rs::parse_header` — parsing do cabeçalho
- `src/memory.rs::load_region_table` — leitura da Region Table
- `src/vm.rs::execute_escape` — execução de extensões

Testes de conformidade em `tests/rfc0001/`.

---

## 12. Changelog

| Versão | Data | Mudanças |
|:---|:---|:---|
| 0001-00 | 2026-09-09 | Especificação inicial |

---

## 13. Referências

- **ISA v2.0** — `docs/ISA_V2.md`
- **MATH_MVP.md** — `docs/MATH_MVP.md`
- **PLANO_AVM_CLUSTER.md** — `docs/PLANO_AVM_CLUSTER.md`
- **PLANO_ISA_UNIVERSAL.md** — `docs/PLANO_ISA_UNIVERSAL.md`
- **PLANO_MOSHI_NATIVO.md** — `docs/PLANO_MOSHI_NATIVO.md`
- **RFC 2119** — Key words for use in RFCs (IETF)
- **IBM z/Architecture Principles of Operation** — modelo de facilities
- **ARM Architecture Reference Manual** — modelo de encoding spaces
- **Intel 64 and IA-32 Architectures Software Developer's Manual** — modelo de prefixos

---

## Appendix A — Exemplo completo: extensão em 2040

**Cenário:** em 2040, um pesquisador quer adicionar `OP_PHOTONIC_MATMUL` com payload de 512 bytes.

**Passos:**

1. Abre RFC-0042 "Photonic Matmul Extension"
2. Especifica `ext_opcode = 0x42`, `ext_len = 512`, feature bit 32 (`PHOTONIC`)
3. RFC é revisado, aceito, implementado
4. ISA v2.5 é publicada (bump minor de 2.4)

**Bytecode gerado em 2040:**

```
"M3BC"                          ; header
0x02 0x05 0x00 0x00             ; ISA v2.5
REQUIRED_FEATURES = ESCAPE      ; bit0
OPTIONAL_FEATURES = PHOTONIC    ; bit32
...
0xB0 0x42 ...                   ; ESCAPE PHOTONIC_MATMUL
```

**Runtime de 2026:**

- Lê header, vê `ISA_MAJOR=2`, `ISA_MINOR=5`
- Runtime 2.0 suporta minor 5? Sim, minors são compatíveis
- `REQUIRED_FEATURES` = `ESCAPE` (bit0), que runtime 2.0 suporta
- `OPTIONAL_FEATURES` = `PHOTONIC` (bit32), que runtime 2.0 **não** suporta
- Encontra `ESCAPE 0x42` com `bit0=1` (opcional)
- Pula 512 bytes e continua
- Log: "Photonic acceleration skipped; running fallback"

**Runtime de 2040:**

- Lê header, mesma análise
- `PHOTONIC` suportado (bit32 implementado)
- Executa `ESCAPE 0x42` acelerado

**Resultado:** o mesmo bytecode roda em ambos os runtimes. Degradação graciosa. Zero quebra binária.

---

## Appendix B — Checklist de conformidade

Um runtime é conforme a RFC-0001 se:

- [ ] Reserva `0xB0–0xB6` como ESCAPE (mesmo sem implementar)
- [ ] Reserva `R255` como RFH
- [ ] Reserva região `0xF` como Region Table
- [ ] Lê header de bytecode antes de executar
- [ ] Valida `REQUIRED_FEATURES`
- [ ] Degrada graciosamente em `OPTIONAL_FEATURES`
- [ ] Pula `ESCAPE` opcional desconhecido
- [ ] Falha limpo em `ESCAPE` obrigatório desconhecido
- [ ] Reporta `CAPABILITY_QUERY` corretamente
- [ ] Nunca reutiliza encoding deprecado
- [ ] Nunca remove feature bit
- [ ] Nunca remove região

---

**Fim do RFC-0001.**

Este documento é a **válvula de escape** que transforma a ISA M³-AVM de uma especificação de 15 anos em uma especificação de 50 anos. Nada nele precisa ser implementado hoje — apenas respeitado.

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
