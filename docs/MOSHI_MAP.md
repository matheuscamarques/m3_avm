# MOSHI_MAP — PersonaPlex/Moshi na M³-AVM (F0)

Gerado a partir de `src/moshi.rs` (`describe_map`). Fonte de verdade em runtime é
`temporal_tensor_names() / depformer_tensor_names() / mimi_tensor_names()`.

## Config referência (PersonaPlex-7B-v1)

`hidden=4096 layers=32 heads=32 ff=11264 (head_dim=128) | dep=6x1024x16 ff=2816 dep_q=16 | codebooks=16 | vocab=32000 | 12.5Hz 24kHz | streams=17 (1 texto + 8 user + 8 agent) | frame=1920 samples (80ms) | delay teórico 160ms`

## Temporal layer 0 (candidatos por kind, primeiro = GGUF moshi.cpp, segundo = safetensors HF)

- q: `blk.0.attn_q.weight` | `temporal.layers.0.self_attn.q_proj.weight`
- k: `blk.0.attn_k.weight` | `temporal.layers.0.self_attn.k_proj.weight`
- v: `blk.0.attn_v.weight` | `temporal.layers.0.self_attn.v_proj.weight`
- o: `blk.0.attn_output.weight` | `temporal.layers.0.self_attn.o_proj.weight`
- gate: `blk.0.ffn_gate.weight` | `temporal.layers.0.mlp.gate_proj.weight`
- up: `blk.0.ffn_up.weight` | `temporal.layers.0.mlp.up_proj.weight`
- down: `blk.0.ffn_down.weight` | `temporal.layers.0.mlp.down_proj.weight`
- attn_norm: `blk.0.attn_norm.weight` | `temporal.layers.0.input_layernorm.weight`
- ffn_norm: `blk.0.ffn_norm.weight` | `temporal.layers.0.post_attention_layernorm.weight`

## Depformer layer 0

- in_proj: `blk_dep.0.in_proj.weight` | `depformer.layers.0.self_attn.in_proj.weight`
- o: `blk_dep.0.out_proj.weight` | `depformer.layers.0.self_attn.out_proj.weight`
- gate: `blk_dep.0.gate_proj.weight` | `depformer.layers.0.mlp.gate_proj.weight`
- up: `blk_dep.0.up_proj.weight` | `depformer.layers.0.mlp.up_proj.weight`
- down: `blk_dep.0.down_proj.weight` | `depformer.layers.0.mlp.down_proj.weight`
- norm: `blk_dep.0.norm.weight` | `depformer.layers.0.input_layernorm.weight`

## Mimi / embeddings

- token_embd: `token_embd.weight` | `embeddings.text.weight` | `model.embed_tokens.weight`
- audio_embd: `audio_embd.weight` | `embeddings.audio.weight`
- output: `output.weight` | `lm_head.weight`
- mimi_encoder: `mimi.encoder.weight` | `tokenizer.encoder.weight`
- mimi_decoder: `mimi.decoder.weight` | `tokenizer.decoder.weight`

## Quantização (~5GB, padrão idle-intelligence)

- Temporal (32x4096, atenção + FFN): `Q4_K` → `~3.5GB` (+ `Q4_0` embeddings).
- Depformer (~50MB) + Mimi (~370MB) + norms `F32`: mantém `fp16/F32` (qualidade áudio).
- Total: `~4.9GB` (`temporal.safetensors Q4 + depformer fp16 + mimi fp16 + embeddings`).
- Kernels: `src/quant.rs dequant_q4_0/q4_k/q6_k/q8_0` + `src/matvec_quant.rs matvec_q4k AVX2 / q4_0 / q8_0`.

## Detecção

- `is_moshi_arch()`: contém `moshi|mimi|personaplex|moshiko|moshika` (case-insensitive).
- `count_moshi_tensors()`: nº de `blk.{l}.attn/ffn` encontrados; `0` em LLM texto puro com outros nomes? Não — LLM texto também usa `blk.*`, então checar `arch` primeiro, contagem depois.
- Moshi original (8 codebooks) vs PersonaPlex (16): checar `moshi.audio.codebooks / mimi.codebooks`.
