# PLANO — PersonaPlex/Moshi Quantizado Nativo na M³-AVM

**Objetivo:** rodar PersonaPlex-7B (base Kyutai Moshi) quantizado em Rust `Q4_0/INT4 ~5GB ~180ms` 100% dentro da M³-AVM, sem sidecar Python.
**Estratégia escolhida:** Nativo M3-AVM completo.
**Data:** 2026-09-09

## 0. Contexto confirmado

* `PersonaPlex == Moshi`: `NVIDIA/personaplex` declara `based on the Moshi architecture and weights`, base `kyutai/moshiko-pytorch-bf16`, `7B`.
* Arquitetura alvo: `Temporal 32 layers dim=4096 32 heads SwiGLU + Depformer 6 layers dim=1024 16 heads + Mimi 16 codebooks 12.5Hz 24kHz`, `17 streams (1 texto + 8 user + 8 agent)`.
* Latência referência: teórico `80ms frame + 80ms delay = 160ms`, prático `~200ms L4`, PersonaPlex `turn 170ms / interrupt 240ms`.
* Referência `~5GB`: `idle-intelligence/personaplex-7b-v1-q4_k-webgpu` = `4.4GB Q4_K shards + 367MB Mimi + ~940MB embeddings = ~4.9GB`. `moshi-backend` oficial Rust só tem `candle-bf16/q8` (`config.json/config-q8.json`), `INT4` oficial só no MLX.

## 1. Estado atual M³-AVM (o que já temos)

* `src/quant.rs:9,38,76,156` — `dequant_q4_0 / q4_k / q6_k / q8_0` prontos.
* `src/matvec_quant.rs:221,361,368,375` — `matvec_q4k AVX2 + q6k + q4_0 + q8_0` fused, sem materializar `~400MB f32`.
* `src/memory.rs:15,506` — `PERSISTENTE 0x20 mmap` zero-copy + `madvise WillNeed`, `KV_CACHE 0x30` por layer.
* `src/inference.rs:37,227,258` — `ModelConfig::from_gguf + LayerNames + RealInference::forward_one` só para `llama/qwen2/mistral/mamba` single-stream texto.
* `src/bus.rs:20,src/reactor.rs:1,src/context.rs:15` — `watch/broadcast`, scheduler `Red>Blue>Green`, `FORK CoW 39µs / ABORT 217µs`, `rollback.rs` por entropia.
* `src/gguf.rs:3` — comentário diz `Q4_* rejeitado` (desatualizado, código já suporta).
* Gaps: sem `src/moshi.rs`, sem `src/mimi.rs` (`src/streaming.rs` vazio, `src/stt.rs:93` é `whisper 16kHz`, não `Mimi 24kHz`), sem `Depformer`, sem `KV 17 streams`, `src/vm.rs:621 SENSE` hoje é `rand` white-noise, `src/memory_wgpu.rs:185` só `ATTN<=64`.

## 2. Fases

### F0 — Modelo + mapa (1-2 dias)
- [ ] Baixar `Codes4Fun/personaplex-7b-v1-q4_k-GGUF` ou `idle-intelligence/personaplex-7b-v1-q4_k-webgpu` + `tokenizer_spm_32k_3.model + config.json + voices/*.pt`.
- [ ] Listar tensores: `n_tensors`, `dtype` por tensor, confirmar `32x temporal + 6x depformer + gating16`.
- [ ] Gerar `docs/MOSHI_MAP.md` com nomes reais vs `blk.{i}.attn_*`.
- Aceite: `cargo run --bin m3_avm -- run examples/minimal.m3asm --model <gguf> --emit-asm /tmp/gen.m3asm` lista `arch/hidden/layers` sem panic.

### F1 — `src/moshi.rs` + loader
- [ ] Criar `src/moshi.rs`: `MoshiConfig{hidden=4096,n_layers=32,n_heads=32,ff=11264,dep_layers=6,dep_dim=1024,dep_q=16,codebooks=16,vocab=32000,frame_hz=12.5,sr=24000}` + `from_gguf() + from_config_json()`.
- [ ] Criar `MoshiLayerNames` + `resolve 1x` espelhando `src/inference.rs:304 resolve_layer_names`.
- [ ] Estender `ModelConfig::from_gguf (src/inference.rs:37)` ou branch `is_moshi()`, corrigir `src/gguf.rs:3`.
- [ ] Exportar em `src/lib.rs:1` + `src/main.rs:14`.
- Aceite: `cargo test --lib moshi::test_parse_personaplex_header`.

### F2 — `src/mimi.rs` (maior risco)
- [ ] Criar `src/mimi.rs`: `SEANet encoder/decoder + transformer + RVQ 16` stub com `ndarray/faer`.
- [ ] MVP offline: `WAV 24kHz mono 80ms (1920 spl) -> codes[16]@12.5Hz -> WAV`.
- [ ] Trocar `SENSE AUDIO (src/vm.rs:621)` por PCM real em `TEMPORAL 0x10 (src/memory.rs:30)`. Fallback temporário: `rustymimi` via `Command` como em `src/stt.rs:72`.
- Aceite: roundtrip `10s WAV` determinístico, `RTF` logado.

### F3 — Temporal + Depformer quantizado
- [ ] Reusar `matvec_q4k/q4_0/q8_0 (src/matvec_quant.rs)` para shapes `4096x4096, 4096x11264, 1024x2816`. `4096%256==0` já entra no `AVX2 (src/matvec_quant.rs:236)`.
- [ ] Implementar `forward_temporal_one() + forward_depformer()` clonando `forward_one (src/inference.rs:527)`: `RMSNorm->QKV+RoPE->KV append 17 streams->softmax->O->residual->SwiGLU`, + `acoustic delay 1 frame`.
- [ ] `KV_CACHE 0x30 (src/memory.rs:43)`: estender de `seq*hidden` texto para `17 streams x N` + `truncate_kv_17()`, `snapshot/restore` para rollback.
- [ ] Condicionamento voz: `voices/*.embeddings.bin + *.cache.json (17x4)`.
- Aceite: paridade 1 frame vs `moshi.pytorch` com `max_diff <1e-3`.

### F4 — Full-duplex na ISA
- [ ] Novo `programs/moshi_loop.m3asm`: `SENSE TOKEN 12.5Hz -> FORK GREEN -> NORM/ATTN NOTIFY_EACH_HEAD/FFN/ADD -> SAMPLE -> STREAM DECODED -> COMPARE/IF_EQUAL + IF_INTERRUPT HANDLE_ABORT`.
- [ ] Ligar `RED=interrupção/barge-in, BLUE=I/O áudio, GREEN=inference` em `src/reactor.rs + src/bus.rs + src/rollback.rs` para 17 streams.
- [ ] CLI: `cargo run --bin m3_avm -- run programs/moshi_loop.m3asm --model <q4_k.gguf> --real --interactive-audio`.
- Aceite: `M3_PROFILE=1` mostra breakdown `temporal/depformer/mimi`, interrupção digitada faz `ABORT+rollback` sem perder `>90%` contexto.

### F5 — INT4 ~5GB + otimização
- [ ] Quantizar como `idle-intelligence`: temporal `Q4_K (4.4GB)`, embeddings `Q4_0`, depformer/mimi `fp16`.
- [ ] `persistent_mib 2048-5120`, `mmap lazy`, `RUSTFLAGS="-C target-cpu=znver1"` (+10% em `README.md:101`).
- [ ] Deixar `GEMV.wgsl Q4_K` (`src/memory_wgpu.rs`, `src/gemv.wgsl`, `src/attn.wgsl`) para depois — hoje ganho `<5%`.
- Aceite: `RAM ~5GB`, `~180ms/frame`, `FullDuplexBench TOR/latência` medidos.

## 3. Ordem e não-objetivos
Ordem: `F0 -> F1 -> F2 -> F3 -> F4 -> F5`. Não pular `F2` — sem Mimi não há speech-to-speech.
Não fazer agora: `BSR/FlashAttention sparse (README.md:82)`, `TUI nova (src/tui.rs)`.

## 4. Comandos úteis
```bash
cargo test --lib moshi mimi matvec_quant -- --nocapture
M3_PROFILE=1 cargo run --bin m3_avm --release -- run programs/moshi_loop.m3asm --model models/personaplex-7b-v1-q4_k.gguf --real --max-steps 3
RUSTFLAGS="-C target-cpu=znver1" cargo run --release -- run --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --real --max-steps 3
```

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
