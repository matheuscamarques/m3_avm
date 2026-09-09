# M³-AVM — Modelos Matemáticos do MVP e Provas

> **Author: Matheus de Camargo Marques** · https://github.com/matheuscamarques/m3_avm
> Nível: provas matemáticas estilo paper (teorema/demonstração verificáveis à mão).
> **Não** são provas machine-checked (Lean/Coq) — ver [§8](#8-o-que-é-formal-aqui-e-o-que-não-é).

Notação: `d` = hidden (`d_model`), `I` = `d_inner`, `S` = `d_state`,
`N` = nº de tokens gerados, `L` = nº de camadas, `H` = nº de heads.

---

## 1. Recorrência SSM (Mamba-1) — o que o código implementa

Definição (contínua, por canal `o`, estado `i`). Com `A[o,i] < 0` fixo,
entrada `x[o]`, passo `dt[o] > 0`:

```
ḣ = A·h + B[i]·x        (1)
y   = C·h + D[o]·x       (2)
```

O código (`src/ssm.rs`, convenção documentada) implementa Euler explícito:

```
a_d    = exp(dt·A)                       (3)
h ← h·a_d + x·B[i]·dt                    (4)
y = ⟨h, C⟩ + D·x                         (5)
```

**Teorema 1 (estabilidade + consistência do passo Euler).**
Se `A < 0` e `dt > 0`, então:
(a) `0 < a_d < 1` (contração — o estado esquece exponencialmente);
(b) o erro local de um passo Euler vs. a solução exata ZOH é `O(dt²)`.

*Demonstração.* (a) `a_d = exp(dt·A)` com `dt·A < 0` logo `a_d ∈ (0,1). ∎`
(b) A solução exata de (1) com entrada constante no passo é
`h(dt) = e^{Adt}h₀ + (e^{Adt}−1)/A·Bx`. Expandindo,
`(e^{Adt}−1)/A = dt + A·dt²/2 + …`. O código usa `dt` no lugar desse fator;
a diferença é `|A|·dt²/2·|Bx| = O(dt²)`. ∎

*Corolário prático.* O comentário no `src/ssm.rs` ("Euler coincide em `dt`
pequeno") é exatamente o item (b). `dt` aqui vem de `softplus(·)`, logo é
sempre positivo e pequeno por construção — a hipótese do teorema é garantida
pelo próprio pipeline, não assumida.

**Teorema 2 (custo constante por passo — o "O(1)" do diagrama).**
Um passo SSM de uma camada custa `Θ(I·S)` FLOPs e o estado ocupa
`I·S + I·d_conv` escalares, **independente de N**.

*Demonstração.* (4) executa uma multiplicação-adição por par `(o,i)`: `I·S`
pares. (5) é um dot por canal: `I·S` de novo. O estado `MambaState`
(`src/ssm.rs`) guarda só `conv` (`I·d_conv`) + `ssm` (`I·S`). Nada indexado
por `N` aparece. ∎

*Instância numérica (defaults do código: `I = 2d`, `S = 16`, `d_conv = 4`).*
Para `d = 2048`: estado/camada = `2·2048·16 + 2·2048·4 = 81.920` f32 ≈ 320 KiB;
22 camadas ≈ 7 MiB. Cabe em cache — é por isso que o MVP roda em CPU modesta.

---

## 2. Por que o híbrido Mamba + Transformer é viável

**Teorema 3 (cruzamento quadrático → linear).**
Atenção densa custa `Θ(N²·d)` por camada; o scan SSM custa `Θ(N·I·S)`.
Existe `N₀ = Θ(I·S/d)` acima do qual o SSM é assintoticamente mais barato,
e o KV-cache da atenção custa `Θ(L·N·d)` memória contra `Θ(L·I·S)` do SSM.

*Demonstração.* Atenção: matriz de scores `N×N`, cada score um dot de dim
`d/H` em `H` heads → `Θ(N²d)`. SSM: Teorema 2 por passo × `N` passos.
KV-cache guarda K,V (`2·N·d` por camada). Comparando `N²d` vs `N·I·S`,
o quociente é `N·d/(I·S)`, crescente em `N`. ∎

*Instância.* Com `I·S/d = 2·16 = 32`: acima de poucas dezenas de tokens o
termo quadrático domina — exatamente a região onde o MVP faz rollback e
pré-empção valerem a pena. O diagrama põe Mamba e Transformer nos mesmos
blocos de memória **porque** o Teorema 3 diz quando usar cada um.

---

## 3. Correção do rollback (FORK/ABORT) — o coração do MVP

Modelo: o estado da VM é a tupla
`Σ = (global_heap, sparse_heap, kv_cache_layers, kv_heap)`,
versionada por `version ∈ ℕ` (`src/memory.rs`).

**Teorema 4 (rollback exato).**
Se `v = snapshot()` foi chamado no estado `Σ₀`, e nenhuma operação
destrói a entrada `v` dos mapas de snapshot, então `restore(v)` recoloca
a VM exatamente em `Σ₀` e ajusta `version = v`.

*Demonstração.* `snapshot()` insere clones (`Arc`/estruturas) de cada
componente de `Σ₀` sob a chave `v` nos quatro mapas e retorna `v`.
`restore(v)` substitui cada componente atual pelo valor armazenado sob `v`
e só então faz `version = v`; se nenhuma chave existe, retorna erro em vez
de estado parcial. Logo, em caso de sucesso, `Σ = Σ₀`. ∎

*Observações honestas (limites da prova).*
(i) É correção **lógica**, não temporal: nada aqui prova os `~217µs` —
esses são medidos (`README.md` §5).
(ii) `Arc::clone` dá CoW barato **enquanto** os tensores forem tratados como
imutáveis após o snapshot; mutação in-place de um tensor alcançável pelo
snapshot quebraria a hipótese — invariante a manter em `vm.rs`.
(iii) Snapshots nunca são podados no código atual: `T` snapshots custam
`O(T·|Σ|)` referências — o MVP precisa de política de retenção (janela de
`k` versões) antes de sequências longas. Vira teorema análogo com janela
deslizante; fica como trabalho futuro.

---

## 4. Latência de preempção — modelo, não teorema

```
L_abort = t_detect + t_sched + t_restore            (6)
```

- `t_detect`: `SENSE`/`watch::broadcast` — ordem de µs (teste `src/bus.rs`).
- `t_sched`: remoção do contexto + checagem de prioridade — `O(1)`.
- `t_restore`: Teorema 4 — cópia de um punhado de `Arc`s + `HashMap`s, `O(|Σ|)`
  referências, µs para heaps pequenos.

Por (6), `L_abort` **não contém nenhum termo proporcional ao tamanho do
modelo nem a N** — é por isso que preempção por bloco é estruturalmente mais
barata que "matar o kernel e recomeçar". O valor absoluto (`25 ms` medido em
`32×32` sob carga, `README.md` §5) ainda reflete gargalos de implementação
Tokio/emulador, não silício — o diagrama marca `~217µs` como **meta**.

---

## 5. Erro de quantização Q4_K — bound que justifica inferência real no MVP

Cada super-bloco Q4_K (`src/quant.rs`): 256 pesos em 144 B
(`≈ 0.56 B/peso` vs 4 B em f32 → **7.1× menos memória**).
Pesos dequantizados `ŵ = s·(q − z)` com escala `s` de 6 bits por sub-bloco
de 32.

**Teorema 5 (bound do erro de matvec quantizado).**
Seja `e = w − ŵ` o erro de um vetor de `n` pesos com `|eᵢ| ≤ s_max/2`.
Para qualquer ativação `x`, `|⟨e,x⟩| ≤ (s_max/2)·‖x‖₁ ≤ (s_max/2)·√n·‖x‖₂`.

*Demonstração.* Cauchy–Schwarz discreta:
`|Σ eᵢxᵢ| ≤ max|eᵢ|·Σ|xᵢ| = (s_max/2)·‖x‖₁ ≤ (s_max/2)·√n·‖x‖₂`. ∎

*Leitura.* O erro por token cresce no máximo com `√n` e é proporcional à
escala do bloco — blocos de 32 pesos com escala própria (o desenho Q4_K)
mantêm `s_max` pequeno onde os pesos são pequenos. É o argumento formal de
que `matvec_q4k` AVX2 (`src/matvec_quant.rs`) preserva qualidade suficiente
para o MVP sem prova empírica de perplexidade (que continua necessária —
ver §8).

---

## 6. Orçamento tempo-real do áudio (barge-in)

`SENSE_AUDIO_PCM` entrega PCM 24 kHz em janelas de 80 ms = 1920 f32
(`src/opcodes.rs`, `SENSE_AUDIO_PCM`).

**Teorema 6 (condição de tempo-real).**
Se o pipeline `CODEC_ENC → SSM_SCAN → AUDIO_ALIGN` processa uma janela em
tempo `T_proc < 80 ms`, o atraso de detecção de barge-in é limitado por
`80 ms + T_proc`, independente do comprimento da conversa.

*Demonstração.* Janelas chegam a cada 80 ms; com `T_proc < 80 ms` a fila nunca
cresce (argumento clássico de estabilidade de fila determinística:
serviço mais rápido que chegada ⇒ backlog ≤ 1 janela). A detecção usa no
máximo a janela corrente + o processamento dela. ∎

*Instância.* `CODEC_ENC: 1920 f32 → 16 tokens` (fator 120×, `src/mimi.rs`);
o estado SSM por janela é `O(1)` (Teorema 2) — logo `T_proc` é dominado pelo
encoder, e a condição do Teorema 6 vira um requisito testável por bench
(`--bench` por janela de 80 ms).

---

## 7. Esparsidade CSR — economia que o MVP assume

**Proposição 7.** Tensor `r×c` com densidade `ρ`: CSR ocupa
`nnz·4 + (r+1)·4 + nnz·4` bytes (`nnz = ρrc`) contra `rc·4` densos —
razão ≈ `2ρ` para `r` grande. Matvec esparso custa `Θ(nnz)` em vez de
`Θ(rc)`.

*Exemplo do repo:* `64×64` a 5% → `~10× menor` (teste `src/qa.rs:900`).
Limite honesto: `attn_sparse` densifica no softmax (`src/sparse.rs:190`) —
o ganho vale para projeções/FFN, não para a atenção em si (sem
FlashAttention esparsa).

---

## 8. O que é "formal" aqui e o que não é

| Afirmação | Status |
| :--- | :--- |
| Teoremas 1–7 acima | Provados a partir do código citado; verif
...[truncated 911 chars]