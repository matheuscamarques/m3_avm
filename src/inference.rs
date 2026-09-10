//! inference.rs — Real inference minimal para TinyLlama/DeepSeek f32/Q4_K
//! Lê GGUF via `crate::gguf::GgufFile`, faz forward de 1 token com 1-2 layers usando `ndarray`
//! e integra com `M3Tokenizer` e `MemoryManager` (mmap PERSISTENTE).
//! Objetivo: provar pipeline real sem dummy 2x8, mantendo rollback 6µs.

use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

use crate::memory::{DType, MemoryManager};
use crate::tokenizer::M3Tokenizer;
use crate::gguf::GgufFile;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden: usize,          // embedding_length / hidden_size
    pub intermediate: usize,    // feed_forward_length / intermediate_size
    pub n_layers: usize,        // block_count
    pub n_heads: usize,         // attention.head_count
    pub n_kv_heads: usize,      // attention.head_count_kv (GQA/MQA)
    pub vocab: usize,
    pub arch: String,           // general.architecture
    pub context_length: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    // --- Mamba (spike: arch=mamba / falcon_mamba) ---
    // GGUF KV: {arch}.ssm.{conv_kernel,inner_size,state_size,time_step_rank,dt_b_c_rms}
    // Ref: llama.cpp gguf-py/gguf/constants.py (Keys.SSM) + conversion/mamba.py
    pub d_inner: usize,         // ssm.inner_size (default 2*hidden)
    pub d_state: usize,         // ssm.state_size (default 16)
    pub d_conv: usize,          // ssm.conv_kernel (default 4)
    pub dt_rank: usize,         // ssm.time_step_rank (default ceil(hidden/16))
    pub dt_b_c_rms: bool,       // Falcon-Mamba aplica RMS em dt/B/C
}

impl ModelConfig {
    pub fn from_gguf(gg: &GgufFile) -> Self {
        let get = |k: &str| gg.kv.get(k).cloned().unwrap_or_default();
        let get_parse = |keys: &[&str], default: usize| -> usize {
            for k in keys {
                if let Some(v) = gg.kv.get(*k) {
                    if let Ok(n) = v.parse::<usize>() { return n; }
                }
            }
            default
        };
        let get_f32 = |keys: &[&str], default: f32| -> f32 {
            for k in keys {
                if let Some(v) = gg.kv.get(*k) {
                    if let Ok(n) = v.parse::<f32>() { return n; }
                }
            }
            default
        };
        let arch = get("general.architecture");
        // hidden: qwen2.embedding_length vs llama.embedding_length vs general
        // (+mamba/falcon_mamba para o spike SSM)
        let hidden = get_parse(&["qwen2.embedding_length", "llama.embedding_length", "mistral.embedding_length", "mamba.embedding_length", "falcon_mamba.embedding_length", "general.embedding_length", "hidden_size"], 2048);
        let hidden = if hidden == 2048 && arch == "qwen2" { get_parse(&["qwen2.embedding_length"], 1536) } else { hidden };
        // intermediate
        let intermediate = get_parse(&["qwen2.feed_forward_length", "qwen2.intermediate_size", "llama.feed_forward_length", "mistral.feed_forward_length", "intermediate_size"], 5632);
        // layers
        let n_layers = get_parse(&["qwen2.block_count", "llama.block_count", "mistral.block_count", "mamba.block_count", "falcon_mamba.block_count", "general.block_count", "num_hidden_layers"], 22);
        // heads
        let n_heads = get_parse(&["qwen2.attention.head_count", "llama.attention.head_count", "mistral.attention.head_count", "num_attention_heads"], 32);
        // kv heads (GQA)
        let n_kv_heads = get_parse(&["qwen2.attention.head_count_kv", "llama.attention.head_count_kv", "mistral.attention.head_count_kv", "num_key_value_heads"], n_heads);
        // vocab
        let vocab = get_parse(&["qwen2.vocab_size", "llama.vocab_size", "general.vocab_size"], 32000);
        let vocab = if vocab == 32000 {
            // tenta tokenizer.ggml.tokens length fallback via gg.n_tensors? mantém 32000
            if let Some(v) = gg.kv.get("tokenizer.ggml.tokens") {
                if v.starts_with("array[") { v[6..v.len()-1].parse().unwrap_or(32000) } else { 32000 }
            } else { 32000 }
        } else { vocab };
        let context_length = get_parse(&["qwen2.context_length", "llama.context_length", "mamba.context_length", "general.context_length"], 2048);
        let rope_theta = get_f32(&["qwen2.rope.freq_base", "llama.rope.freq_base", "general.rope.freq_base"], 10000.0);
        let norm_eps = get_f32(&["qwen2.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon", "mamba.attention.layer_norm_rms_epsilon", "general.layer_norm_rms_epsilon"], 1e-5);
        // --- Mamba SSM params (só relevantes se arch contém "mamba") ---
        let d_inner = get_parse(&["mamba.ssm.inner_size", "falcon_mamba.ssm.inner_size", "ssm.inner_size"], hidden * 2);
        let d_state = get_parse(&["mamba.ssm.state_size", "falcon_mamba.ssm.state_size", "ssm.state_size"], 16);
        let d_conv = get_parse(&["mamba.ssm.conv_kernel", "falcon_mamba.ssm.conv_kernel", "ssm.conv_kernel"], 4);
        let dt_rank_default = hidden.div_ceil(16).max(1);
        let dt_rank = get_parse(&["mamba.ssm.time_step_rank", "falcon_mamba.ssm.time_step_rank", "ssm.time_step_rank"], dt_rank_default);
        let dt_b_c_rms = gg.kv.get("mamba.ssm.dt_b_c_rms")
            .or_else(|| gg.kv.get("falcon_mamba.ssm.dt_b_c_rms"))
            .map(|v| v == "true" || v == "1")
            .unwrap_or_else(|| arch.contains("falcon"));
        Self { hidden, intermediate, n_layers, n_heads, n_kv_heads, vocab, arch: arch.clone(), context_length, rope_theta, norm_eps, d_inner, d_state, d_conv, dt_rank, dt_b_c_rms }
    }

    pub fn head_dim(&self) -> usize {
        if self.n_heads == 0 { self.hidden } else { self.hidden / self.n_heads }
    }
    pub fn is_gqa(&self) -> bool { self.n_kv_heads < self.n_heads }
    /// Spike Mamba: arch=mamba / falcon_mamba / mamba2 (contém "mamba").
    pub fn is_mamba(&self) -> bool { self.arch.contains("mamba") }
}

/// Abstração para nomes de tensores por arquitetura (llama, qwen2, mistral, deepseek, phi)
pub trait TensorNamer {
    fn token_embd() -> &'static str { "token_embd.weight" }
    fn output_norm() -> &'static str { "output_norm.weight" }
    fn output_weight() -> &'static str { "output.weight" }
    fn attn_norm(layer: usize) -> String;
    fn attn_q(layer: usize) -> String;
    fn attn_k(layer: usize) -> String;
    fn attn_v(layer: usize) -> String;
    fn attn_o(layer: usize) -> String;
    fn ffn_norm(layer: usize) -> String;
    fn ffn_gate(layer: usize) -> String;
    fn ffn_up(layer: usize) -> String;
    fn ffn_down(layer: usize) -> String;
}

pub struct LlamaNamer;
impl TensorNamer for LlamaNamer {
    fn attn_norm(l: usize) -> String { format!("blk.{}.attn_norm.weight", l) }
    fn attn_q(l: usize) -> String { format!("blk.{}.attn_q.weight", l) }
    fn attn_k(l: usize) -> String { format!("blk.{}.attn_k.weight", l) }
    fn attn_v(l: usize) -> String { format!("blk.{}.attn_v.weight", l) }
    fn attn_o(l: usize) -> String { format!("blk.{}.attn_output.weight", l) }
    fn ffn_norm(l: usize) -> String { format!("blk.{}.ffn_norm.weight", l) }
    fn ffn_gate(l: usize) -> String { format!("blk.{}.ffn_gate.weight", l) }
    fn ffn_up(l: usize) -> String { format!("blk.{}.ffn_up.weight", l) }
    fn ffn_down(l: usize) -> String { format!("blk.{}.ffn_down.weight", l) }
}

pub struct Qwen2Namer;
impl TensorNamer for Qwen2Namer {
    // Qwen2 GGUF ainda usa blk.* para compat llama.cpp, mas também suporta model.layers.*
    fn attn_norm(l: usize) -> String { format!("blk.{}.attn_norm.weight", l) }
    fn attn_q(l: usize) -> String { format!("blk.{}.attn_q.weight", l) }
    fn attn_k(l: usize) -> String { format!("blk.{}.attn_k.weight", l) }
    fn attn_v(l: usize) -> String { format!("blk.{}.attn_v.weight", l) }
    fn attn_o(l: usize) -> String { format!("blk.{}.attn_output.weight", l) }
    fn ffn_norm(l: usize) -> String { format!("blk.{}.ffn_norm.weight", l) }
    fn ffn_gate(l: usize) -> String { format!("blk.{}.ffn_gate.weight", l) }
    fn ffn_up(l: usize) -> String { format!("blk.{}.ffn_up.weight", l) }
    fn ffn_down(l: usize) -> String { format!("blk.{}.ffn_down.weight", l) }
    // fallback para HF naming (não usado no GGUF atual, mas coberto via try_alternates)
}

impl ModelConfig {
    /// Tenta múltiplos padrões de nome para compatibilidade HF vs GGUF
    pub fn attn_q_names(&self, layer: usize) -> Vec<String> {
        vec![
            format!("blk.{}.attn_q.weight", layer),
            format!("model.layers.{}.self_attn.q_proj.weight", layer),
            format!("blk.{}.attn.q.weight", layer),
        ]
    }
    /// Nomes Mamba-1 (GGUF `blk.{i}.ssm_*` + fallback HF `mixer.*`).
    /// Ref: llama.cpp `MODEL_TENSOR::{SSM_IN,SSM_CONV1D,SSM_X,SSM_DT,SSM_A,SSM_D,SSM_OUT}`
    /// + `ATTN_NORM` reusada como norm da camada.
    pub fn try_get_mamba_tensor_names(&self, layer: usize, kind: &str) -> Vec<String> {
        match kind {
            "mamba_norm" => vec![
                format!("blk.{}.attn_norm.weight", layer),
                format!("model.layers.{}.input_layernorm.weight", layer),
            ],
            "ssm_in" => vec![
                format!("blk.{}.ssm_in.weight", layer),
                format!("model.layers.{}.mixer.in_proj.weight", layer),
            ],
            "conv_w" => vec![
                format!("blk.{}.ssm_conv1d.weight", layer),
                format!("model.layers.{}.mixer.conv1d.weight", layer),
            ],
            "conv_b" => vec![
                format!("blk.{}.ssm_conv1d.bias", layer),
                format!("model.layers.{}.mixer.conv1d.bias", layer),
            ],
            "ssm_x" => vec![
                format!("blk.{}.ssm_x.weight", layer),
                format!("model.layers.{}.mixer.x_proj.weight", layer),
            ],
            "ssm_dt" => vec![
                format!("blk.{}.ssm_dt.weight", layer),
                format!("model.layers.{}.mixer.dt_proj.weight", layer),
            ],
            "ssm_dt_bias" => vec![
                format!("blk.{}.ssm_dt.bias", layer),
                format!("model.layers.{}.mixer.dt_proj.bias", layer),
            ],
            // ssm_a/d no GGUF não têm sufixo `.weight` (ver jamba.cpp/mamba.py)
            "ssm_a" => vec![
                format!("blk.{}.ssm_a", layer),
                format!("blk.{}.ssm_a.weight", layer),
                format!("model.layers.{}.mixer.A_log", layer),
            ],
            "ssm_d" => vec![
                format!("blk.{}.ssm_d", layer),
                format!("blk.{}.ssm_d.weight", layer),
            ],
            "ssm_out" => vec![
                format!("blk.{}.ssm_out.weight", layer),
                format!("model.layers.{}.mixer.out_proj.weight", layer),
            ],
            _ => vec![],
        }
    }
    pub fn try_get_tensor_names(&self, layer: usize, kind: &str) -> Vec<String> {
        match kind {
            "q" => self.attn_q_names(layer),
            "k" => vec![format!("blk.{}.attn_k.weight", layer), format!("model.layers.{}.self_attn.k_proj.weight", layer)],
            "v" => vec![format!("blk.{}.attn_v.weight", layer), format!("model.layers.{}.self_attn.v_proj.weight", layer)],
            "o" => vec![format!("blk.{}.attn_output.weight", layer), format!("model.layers.{}.self_attn.o_proj.weight", layer)],
            "gate" => vec![format!("blk.{}.ffn_gate.weight", layer), format!("model.layers.{}.mlp.gate_proj.weight", layer)],
            "up" => vec![format!("blk.{}.ffn_up.weight", layer), format!("model.layers.{}.mlp.up_proj.weight", layer)],
            "down" => vec![format!("blk.{}.ffn_down.weight", layer), format!("model.layers.{}.mlp.down_proj.weight", layer)],
            "attn_norm" => vec![format!("blk.{}.attn_norm.weight", layer), format!("model.layers.{}.input_layernorm.weight", layer)],
            "ffn_norm" => vec![format!("blk.{}.ffn_norm.weight", layer), format!("model.layers.{}.post_attention_layernorm.weight", layer)],
            "qbias" => vec![format!("blk.{}.attn_q.bias", layer), format!("model.layers.{}.self_attn.q_proj.bias", layer)],
            "kbias" => vec![format!("blk.{}.attn_k.bias", layer), format!("model.layers.{}.self_attn.k_proj.bias", layer)],
            "vbias" => vec![format!("blk.{}.attn_v.bias", layer), format!("model.layers.{}.self_attn.v_proj.bias", layer)],
            _ => vec![],
        }
    }
}

/// Nomes de tensores de uma camada resolvidos 1× no `new()` (P0.1).
/// `None` = tensor ausente (caminho dummy, como antes). Guardar o índice no
/// `gguf.tensors` elimina `format!` + `Vec<String>` + busca linear (339 nomes)
/// a cada um dos ~154 acessos por token.
#[derive(Debug, Clone, Default)]
pub struct LayerNames {
    pub q: Option<usize>,
    pub k: Option<usize>,
    pub v: Option<usize>,
    pub o: Option<usize>,
    pub gate: Option<usize>,
    pub up: Option<usize>,
    pub down: Option<usize>,
    pub attn_norm: Option<usize>,
    pub ffn_norm: Option<usize>,
    pub qbias: Option<usize>,
    pub kbias: Option<usize>,
    pub vbias: Option<usize>,
}

/// Índices pré-resolvidos dos pesos Mamba-1 por camada (`None` = ausente/dummy).
/// Espelha `LayerNames` do caminho Transformer para o mesmo padrão P0.1.
#[derive(Debug, Clone, Default)]
pub struct MambaLayerNames {
    pub norm: Option<usize>,
    pub ssm_in: Option<usize>,
    pub conv_w: Option<usize>,
    pub conv_b: Option<usize>,
    pub ssm_x: Option<usize>,
    pub ssm_dt: Option<usize>,
    pub ssm_dt_bias: Option<usize>,
    pub ssm_a: Option<usize>,
    pub ssm_d: Option<usize>,
    pub ssm_out: Option<usize>,
}

pub struct RealInference {
    pub config: ModelConfig,
    pub tokenizer: M3Tokenizer,
    pub gguf: GgufFile,
    pub gguf_path: String,
    // KV cache per layer: Vec<(k_cache, v_cache)> onde cada cache é [seq_len * hidden]
    pub kv_cache_k: Vec<Vec<f32>>,
    pub kv_cache_v: Vec<Vec<f32>>,
    // Cache de pesos dequantizados (Fase 2.1) — evita 197 dequants/token
    weight_cache: fxhash::FxHashMap<String, std::sync::Arc<Vec<f32>>>,
    // Histórico de ids gerados (+prompt) para repetition penalty real
    gen_history: Vec<u32>,
    // P0.1: nomes resolvidos por camada (índice em gguf.tensors)
    layer_names: Vec<LayerNames>,
    // Spike Mamba: nomes + estados recorrentes (conv + SSM) por camada.
    // Estado O(d_inner*d_state) constante — sem KV crescente.
    pub mamba_names: Vec<MambaLayerNames>,
    pub ssm_states: Vec<crate::ssm::MambaState>,
    // P0.2: profiler lido 1× (env M3_PROFILE), não por token
    profile: bool,
    // GPU híbrida (llama.cpp-style): offload persistente gate/up/down/head.
    // None = CPU puro (default; também quando M3_GPU=0 ou sem adapter).
    #[cfg(feature = "wgpu")]
    gpu: Option<crate::inference_gpu::GpuOffload>,
}

impl RealInference {
    pub fn new(gguf_path: &str) -> Result<Self> {
        let gg = GgufFile::open(gguf_path)?;
        let cfg = ModelConfig::from_gguf(&gg);
        let tok = M3Tokenizer::from_gguf_or_mock(gguf_path);
        eprintln!("[inference] config {:?} vocab {} tensors {}", cfg, tok.vocab_size(), gg.n_tensors);
        let n_layers = cfg.n_layers;
        let profile = std::env::var("M3_PROFILE").map(|v| v == "1").unwrap_or(false);
        let mut inf = Self { config: cfg.clone(), tokenizer: tok, gguf: gg, gguf_path: gguf_path.to_string(), kv_cache_k: vec![Vec::new(); n_layers], kv_cache_v: vec![Vec::new(); n_layers], weight_cache: fxhash::FxHashMap::default(), gen_history: Vec::new(), layer_names: Vec::new(), mamba_names: Vec::new(), ssm_states: Vec::new(), profile,
            #[cfg(feature = "wgpu")]
            gpu: None };
        inf.resolve_layer_names();
        inf.resolve_mamba_names();
        inf.ensure_ssm_states();
        let _ = cfg;
        Ok(inf)
    }

    /// P0.1: resolve 1× qual candidato existe por (camada, kind). Idempotente;
    /// chamada de novo se `n_layers` mudar (mesmo padrão do resize de KV).
    fn resolve_layer_names(&mut self) {
        let mut out = Vec::with_capacity(self.config.n_layers);
        for blk in 0..self.config.n_layers {
            let mut ln = LayerNames::default();
            let mut resolve = |kind: &str| -> Option<usize> {
                for name in self.config.try_get_tensor_names(blk, kind) {
                    if let Some(pos) = self.gguf.tensors.iter().position(|t| t.name == name) {
                        return Some(pos);
                    }
                }
                None
            };
            ln.q = resolve("q");
            ln.k = resolve("k");
            ln.v = resolve("v");
            ln.o = resolve("o");
            ln.gate = resolve("gate");
            ln.up = resolve("up");
            ln.down = resolve("down");
            ln.attn_norm = resolve("attn_norm");
            ln.ffn_norm = resolve("ffn_norm");
            ln.qbias = resolve("qbias");
            ln.kbias = resolve("kbias");
            ln.vbias = resolve("vbias");
            out.push(ln);
        }
        self.layer_names = out;
    }

    /// Acesso seguro (testes constroem sem `new()`; resolve sob demanda).
    fn layer_name(&mut self, blk: usize, f: impl Fn(&LayerNames) -> Option<usize>) -> Option<usize> {
        if self.layer_names.len() != self.config.n_layers {
            self.resolve_layer_names();
        }
        self.layer_names.get(blk).and_then(f)
    }

    /// Resolve 1× os índices `blk.{i}.ssm_*` (GGUF) com fallback HF `mixer.*`.
    /// Idempotente; `None` = peso ausente → caminho dummy nos testes sem modelo.
    fn resolve_mamba_names(&mut self) {
        let mut out = Vec::with_capacity(self.config.n_layers);
        for blk in 0..self.config.n_layers {
            let mut mn = MambaLayerNames::default();
            let mut resolve = |kind: &str| -> Option<usize> {
                for name in self.config.try_get_mamba_tensor_names(blk, kind) {
                    if let Some(pos) = self.gguf.tensors.iter().position(|t| t.name == name) {
                        return Some(pos);
                    }
                }
                None
            };
            mn.norm = resolve("mamba_norm");
            mn.ssm_in = resolve("ssm_in");
            mn.conv_w = resolve("conv_w");
            mn.conv_b = resolve("conv_b");
            mn.ssm_x = resolve("ssm_x");
            mn.ssm_dt = resolve("ssm_dt");
            mn.ssm_dt_bias = resolve("ssm_dt_bias");
            mn.ssm_a = resolve("ssm_a");
            mn.ssm_d = resolve("ssm_d");
            mn.ssm_out = resolve("ssm_out");
            out.push(mn);
        }
        self.mamba_names = out;
    }

    fn mamba_layer_name(&mut self, blk: usize, f: impl Fn(&MambaLayerNames) -> Option<usize>) -> Option<usize> {
        if self.mamba_names.len() != self.config.n_layers {
            self.resolve_mamba_names();
        }
        self.mamba_names.get(blk).and_then(f)
    }

    /// Garante `ssm_states` compatível com o config (conv + SSM zerados).
    /// Chamada em `new()` e sob demanda no `forward_one_mamba`.
    fn ensure_ssm_states(&mut self) {
        let (di, ds, dc, nl) = (self.config.d_inner, self.config.d_state, self.config.d_conv, self.config.n_layers);
        if self.ssm_states.len() != nl
            || self.ssm_states.iter().any(|s| !s.is_compatible(di, ds, dc))
        {
            self.ssm_states = (0..nl)
                .map(|_| crate::ssm::MambaState::new(di, ds, dc))
                .collect();
        }
    }

    /// Zera estados recorrentes (equivalente a `clear_kv_cache` no Transformer).
    pub fn clear_ssm_states(&mut self) {
        for s in &mut self.ssm_states {
            s.reset();
        }
    }
    /// Histórico p/ repetition penalty: registra prompt + gerados
    pub fn push_gen(&mut self, id: u32) { self.gen_history.push(id); }
    pub fn extend_gen(&mut self, ids: &[u32]) { self.gen_history.extend_from_slice(ids); }
    pub fn clear_gen(&mut self) { self.gen_history.clear(); }
    pub fn truncate_gen(&mut self, n: usize) { self.gen_history.truncate(n); }
    pub fn gen_len(&self) -> usize { self.gen_history.len() }
    pub fn clear_weight_cache(&mut self) { self.weight_cache.clear(); }
    pub fn weight_cache_len(&self) -> usize { self.weight_cache.len() }

    pub fn clear_kv_cache(&mut self) {
        for k in &mut self.kv_cache_k { k.clear(); }
        for v in &mut self.kv_cache_v { v.clear(); }
    }

    pub fn truncate_kv_cache(&mut self, seq_len: usize) {
        let h = self.config.hidden;
        for k in &mut self.kv_cache_k { k.truncate(seq_len * h); }
        for v in &mut self.kv_cache_v { v.truncate(seq_len * h); }
    }

    /// Despacho único CLI/testes: Mamba usa `forward_one_mamba`, resto Transformer.
    /// Evita que `main.rs` precise conhecer a arch.
    pub fn forward_one_auto(&mut self, mem: &crate::vm::MemBackend, token_id: u32) -> Result<Vec<f32>> {
        if self.config.is_mamba() {
            self.forward_one_mamba(mem, token_id)
        } else {
            self.forward_one(mem, token_id)
        }
    }

    /// Limpa ambos os caches (KV do Transformer + estados SSM do Mamba).
    /// Para prompt novo no modo `--real --interactive`.
    pub fn clear_caches(&mut self) {
        self.clear_kv_cache();
        self.clear_ssm_states();
    }

    /// Trunca caches p/ rollback. KV trunca de verdade; SSM é recorrente e
    /// não suporta truncate sem snapshot — no spike, mantém estado e avisa.
    /// Rollback total de Mamba = `clear_caches` + `prefill` de novo (futuro:
    /// snapshot por token dos `MambaState`).
    pub fn truncate_caches(&mut self, seq_len: usize) {
        self.truncate_kv_cache(seq_len);
        if self.config.is_mamba() && seq_len == 0 {
            self.clear_ssm_states();
        } else if self.config.is_mamba() {
            eprintln!("[mamba] truncate_caches({}) sem snapshot: estado SSM mantido (spike)", seq_len);
        }
    }

    /// Repete KV heads para GQA (n_kv_heads < n_heads) — expande kv_hidden para hidden.
    /// P1.3: estende direto no destino (KV cache), sem Vec temporário.
    fn repeat_kv_extend(dst: &mut Vec<f32>, kv_small: &[f32], n_heads: usize, n_kv_heads: usize, head_dim: usize) {
        let hidden = n_heads * head_dim;
        if kv_small.is_empty() {
            dst.resize(dst.len() + hidden, 0.0);
            return;
        }
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            // fallback: repete simples até preencher hidden
            let mut n = 0;
            while n < hidden {
                let take = (hidden - n).min(kv_small.len());
                dst.extend_from_slice(&kv_small[..take]);
                n += take;
            }
            return;
        }
        let groups = n_heads / n_kv_heads;
        for kv_idx in 0..n_kv_heads {
            let src = &kv_small[kv_idx * head_dim..(kv_idx + 1) * head_dim];
            for _ in 0..groups {
                dst.extend_from_slice(src);
            }
        }
    }

    /// Núcleo compartilhado: lê bytes do tensor e dequantiza (Fase 2.1).
    /// Extraído de `read_tensor_f32` para reuso pelo caminho por índice (P0.1).
    fn dequant_by_info(&self, mem: &crate::vm::MemBackend, dtype: u32, n: usize, offset: u64, name: &str) -> Result<Vec<f32>> {
        let file_offset = self.gguf.data_offset + offset;
        let paddr = crate::memory::make_persistent_addr(file_offset as u128);
        // Calcula byte_len via dtype; para Q6_K usa 210 por 256, para Q8_0 34 por 32, etc.
        let raw_len = match DType::from_u32(dtype) {
            DType::F32 => n*4,
            DType::F16 => n*2,
            DType::Q4_K | DType::Q5_K => (n/256)*144,
            DType::Q6_K => (n/256)*210,
            DType::Q8_0 => (n/32)*34,
            DType::Q4_0 => (n/32)*18,
            _ => {
                // fallback: estima via próximo tensor offset se disponível
                let mut next_off = None;
                for t in &self.gguf.tensors {
                    if t.offset > offset {
                        if next_off.is_none() || t.offset < next_off.unwrap() {
                            next_off = Some(t.offset);
                        }
                    }
                }
                if let Some(no) = next_off {
                    (no - offset) as usize
                } else {
                    n*4
                }
            },
        };
        let raw = mem.read(paddr, raw_len)?;
        let mut dst = vec![0.0f32; n];
        let ok = crate::quant::dequantize(&raw, dtype, &mut dst, n);
        if !ok {
            // fallback: tenta interpretar como F32 LE se dequant não suportado
            for i in 0..n.min(raw.len()/4) {
                dst[i] = f32::from_le_bytes([raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]]);
            }
            // se ainda falhar e for quantizado, loga
            if DType::from_u32(dtype).is_quantized() {
                eprintln!("[quant] aviso: dequant fallback F32 para dtype {} tensor {}", dtype, name);
            }
        }
        Ok(dst)
    }

    /// Lê tensor do GGUF via MemBackend (mmap PERSISTENTE) e dequantiza se Q4_K/Q6_K
    /// Fase 2.1: usa weight_cache (FxHashMap<String, Arc<Vec<f32>>>) para evitar 197 dequants/token
    fn read_tensor_f32(&self, mem: &crate::vm::MemBackend, name: &str) -> Result<Vec<f32>> {
        // Checa cache primeiro (sem lock, &self mas interior mutability via cache é &mut na prática)
        // Para manter &self, usamos try: se estiver em cache, clona Arc
        if let Some(cached) = self.weight_cache.get(name) {
            return Ok((**cached).clone());
        }
        let info = self.gguf.find_tensor(name).ok_or_else(|| anyhow!("tensor {} não encontrado", name))?;
        self.dequant_by_info(mem, info.dtype, info.n_elements, info.offset, name)
    }
    /// Versão por nome (embedding fallback, LM head: 1×/token, sem hot loop).
    /// Delega ao caminho por índice.
    fn read_tensor_f32_cached(&mut self, mem: &crate::vm::MemBackend, name: &str) -> Result<std::sync::Arc<Vec<f32>>> {
        let idx = self.gguf.tensors.iter().position(|t| t.name == name)
            .ok_or_else(|| anyhow!("tensor {} não encontrado", name))?;
        self.read_tensor_f32_cached_idx(mem, idx)
    }

    /// Versão por índice pré-resolvido (P0.1): hit sem nenhuma alocação
    /// (chave &str emprestada); miss clona o nome 1× para inserir.
    fn read_tensor_f32_cached_idx(&mut self, mem: &crate::vm::MemBackend, idx: usize) -> Result<std::sync::Arc<Vec<f32>>> {
        let key: &str = self.gguf.tensors.get(idx).map(|t| t.name.as_str()).unwrap_or("");
        if let Some(cached) = self.weight_cache.get(key) {
            return Ok(cached.clone());
        }
        let info = self.gguf.tensors.get(idx).ok_or_else(|| anyhow!("tensor idx {} inválido", idx))?;
        let (dtype, n, offset, name) = (info.dtype, info.n_elements, info.offset, info.name.clone());
        let v = self.dequant_by_info(mem, dtype, n, offset, &name)?;
        let arc = std::sync::Arc::new(v);
        self.weight_cache.insert(name, arc.clone());
        Ok(arc)
    }

    /// Matvec por índice pré-resolvido (P0.1): kernel direto sobre bytes GGUF
    /// quando possível; senão cache f32 + faer. `None` = tensor ausente
    /// (testes sem modelo) → dot sobre dummy 0.5, paridade legada.
    fn matvec_weight_pre(&mut self, mem: &crate::vm::MemBackend, tidx: Option<usize>, x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        if let Some(idx) = tidx {
            if let Some(info) = self.gguf.tensors.get(idx) {
                let (dtype, n, offset) = (info.dtype, info.n_elements, info.offset);
                if n == in_dim * out_dim {
                    if let Some(raw_len) = crate::matvec_quant::quant_raw_len(dtype, n) {
                        let file_offset = self.gguf.data_offset + offset;
                        if let Some(raw) = mem.read_model_raw(file_offset, raw_len) {
                            if let Some(y) = crate::matvec_quant::matvec_quant(x, raw, dtype, in_dim, out_dim) {
                                return y;
                            }
                        } else {
                            let paddr = crate::memory::make_persistent_addr(file_offset as u128);
                            if let Ok(raw) = mem.read(paddr, raw_len) {
                                if let Some(y) = crate::matvec_quant::matvec_quant(x, &raw, dtype, in_dim, out_dim) {
                                    return y;
                                }
                            }
                        }
                    }
                    if let Ok(w) = self.read_tensor_f32_cached_idx(mem, idx) {
                        return crate::matvec::matvec(x, &w[..], in_dim, out_dim);
                    }
                } else if let Ok(w) = self.read_tensor_f32_cached_idx(mem, idx) {
                    return crate::matvec::matvec(x, &w[..], in_dim, out_dim);
                }
            }
        }
        crate::matvec::matvec(x, &vec![0.5; in_dim * out_dim], in_dim, out_dim)
    }
    /// Tenta múltiplos nomes (HF vs GGUF) até achar um tensor (sem cache)
    fn read_tensor_f32_try(&self, mem: &crate::vm::MemBackend, names: &[String]) -> Result<Vec<f32>> {
        let mut last_err = None;
        for n in names {
            match self.read_tensor_f32(mem, n) {
                Ok(v) => return Ok(v),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("tensor {:?} não encontrado", names)))
    }
    // REMOVIDO (P0.1): `read_tensor_f32_try_cached(names)` superseded por
    // `read_tensor_f32_cached_idx` + `LayerNames` (resolve_layer_names).
    /// Atalho P0.1: peso de camada por índice (sem Vec/format/find por token).
    fn matvec_layer(&mut self, mem: &crate::vm::MemBackend, blk: usize, f: impl Fn(&LayerNames) -> Option<usize>, x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let idx = self.layer_name(blk, f);
        self.matvec_weight_pre(mem, idx, x, in_dim, out_dim)
    }

    /// Atalho Mamba: mesmo kernel `matvec_weight_pre`, mas com `MambaLayerNames`.
    fn matvec_mamba_layer(&mut self, mem: &crate::vm::MemBackend, blk: usize, f: impl Fn(&MambaLayerNames) -> Option<usize>, x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let idx = self.mamba_layer_name(blk, f);
        self.matvec_weight_pre(mem, idx, x, in_dim, out_dim)
    }

    /// Vetor Mamba por índice (norms/bias/A/D), com dummy determinístico.
    fn cached_mamba_vec(&mut self, mem: &crate::vm::MemBackend, blk: usize, f: impl Fn(&MambaLayerNames) -> Option<usize>, dummy: Vec<f32>) -> std::sync::Arc<Vec<f32>> {
        if let Some(idx) = self.mamba_layer_name(blk, f) {
            if let Ok(w) = self.read_tensor_f32_cached_idx(mem, idx) {
                return w;
            }
        }
        std::sync::Arc::new(dummy)
    }

    /// Forward de 1 token para `arch=mamba` (spike Mamba-1 + Falcon RMS extra).
    /// Pipeline por camada: RMSNorm → in_proj(2*d_inner) → conv1d+SiLU →
    /// x_proj(dt/B/C) → [RMS] → dt_proj+softplus → scan → gate(SiLU z) →
    /// out_proj + residual. Estado constante em `ssm_states` (sem KV).
    pub fn forward_one_mamba(&mut self, mem: &crate::vm::MemBackend, token_id: u32) -> Result<Vec<f32>> {
        self.ensure_ssm_states();
        let h = self.config.hidden;
        let di = self.config.d_inner.max(1);
        let ds = self.config.d_state.max(1);
        let dt_rank = self.config.dt_rank.max(1);
        let eps = self.config.norm_eps;
        // Embedding (mesmo row-slice do Transformer; dummy se sem modelo)
        let mut hidden_cur = vec![0.0f32; h];
        match self.embedding_row(mem, "token_embd.weight", token_id as usize, h) {
            Ok(r) => hidden_cur.copy_from_slice(&r),
            Err(_) => {
                for i in 0..h {
                    hidden_cur[i] = (token_id as f32 * 0.01 + i as f32 * 0.001).sin();
                }
            }
        }
        let mut norm_buf = vec![0.0f32; h.max(di)];
        for blk in 0..self.config.n_layers {
            // norm
            let gamma = self.cached_mamba_vec(mem, blk, |l| l.norm, vec![1.0; h]);
            rms_norm_into(&hidden_cur, &gamma[..h.min(gamma.len())], &mut norm_buf[..h]);
            let hidden_norm = norm_buf[..h].to_vec();
            // in_proj: h -> 2*di
            let xz = self.matvec_mamba_layer(mem, blk, |l| l.ssm_in, &hidden_norm, h, 2 * di);
            let (x0, z) = xz.split_at(di.min(xz.len()));
            let mut x0 = x0.to_vec();
            let mut z = z.to_vec();
            x0.resize(di, 0.0);
            z.resize(di, 0.0);
            // conv1d depthwise + SiLU (dummy = identidade se peso ausente)
            let conv_out = if let Some(idx) = self.mamba_layer_name(blk, |l| l.conv_w) {
                let w = self.read_tensor_f32_cached_idx(mem, idx)
                    .map(|a| (*a).clone())
                    .unwrap_or_else(|_| vec![0.0f32; di * self.config.d_conv.max(1)]);
                let b = self.mamba_layer_name(blk, |l| l.conv_b)
                    .and_then(|bi| self.read_tensor_f32_cached_idx(mem, bi).ok())
                    .map(|a| (*a).clone())
                    .unwrap_or_else(|| vec![0.0f32; di]);
                let st = &mut self.ssm_states[blk];
                let dc = self.config.d_conv.max(1);
                // tolera shape divergente: cai para identidade
                if w.len() == di * dc && st.is_compatible(di, ds, dc) {
                    let mut y = crate::ssm::conv1d_depthwise_update(&mut st.conv, &x0, &w, &b, di, dc);
                    for v in y.iter_mut() {
                        *v = crate::ssm::silu(*v);
                    }
                    y
                } else {
                    x0.iter().map(|&v| crate::ssm::silu(v)).collect()
                }
            } else {
                x0.iter().map(|&v| crate::ssm::silu(v)).collect()
            };
            // x_proj: di -> dt_rank + 2*ds
            let x_dim = dt_rank + 2 * ds;
            let dxbc = self.matvec_mamba_layer(mem, blk, |l| l.ssm_x, &conv_out, di, x_dim);
            let mut dxbc_r = dxbc;
            dxbc_r.resize(x_dim, 0.0);
            let (dt_raw, rest) = dxbc_r.split_at(dt_rank);
            let (b_raw, c_raw) = rest.split_at(ds.min(rest.len()));
            let mut dt_raw = dt_raw.to_vec();
            let mut b_raw = b_raw.to_vec();
            let mut c_raw = c_raw.to_vec();
            b_raw.resize(ds, 0.0);
            c_raw.resize(ds, 0.01);
            if self.config.dt_b_c_rms {
                dt_raw = crate::ssm::rms_norm_plain(&dt_raw, eps);
                b_raw = crate::ssm::rms_norm_plain(&b_raw, eps);
                c_raw = crate::ssm::rms_norm_plain(&c_raw, eps);
            }
            // dt_proj: dt_rank -> di + bias + softplus
            let mut dt_pre = self.matvec_mamba_layer(mem, blk, |l| l.ssm_dt, &dt_raw, dt_rank, di);
            dt_pre.resize(di, 0.0);
            if let Some(bi) = self.mamba_layer_name(blk, |l| l.ssm_dt_bias) {
                if let Ok(bias) = self.read_tensor_f32_cached_idx(mem, bi) {
                    for (a, bb) in dt_pre.iter_mut().zip(bias.iter()) {
                        *a += *bb;
                    }
                }
            }
            let dt: Vec<f32> = dt_pre.iter().map(|&v| crate::ssm::softplus(v)).collect();
            // A/D (dummy estável se ausente: A=-0.5, D=0.5)
            let a_w = self.cached_mamba_vec(mem, blk, |l| l.ssm_a, vec![-0.5; di * ds]);
            let d_w = self.cached_mamba_vec(mem, blk, |l| l.ssm_d, vec![0.5; di]);
            let mut a_full = (*a_w).clone();
            a_full.resize(di * ds, -0.5);
            let mut d_full = (*d_w).clone();
            d_full.resize(di, 0.5);
            // scan
            let st = &mut self.ssm_states[blk];
            let mut y = crate::ssm::selective_scan_update(
                &mut st.ssm, &conv_out, &dt, &a_full, &b_raw, &c_raw, &d_full, di, ds,
            );
            crate::ssm::apply_gate(&mut y, &z);
            // out_proj: di -> h + residual
            let out = self.matvec_mamba_layer(mem, blk, |l| l.ssm_out, &y, di, h);
            for i in 0..h {
                hidden_cur[i] += out.get(i).cloned().unwrap_or(0.0);
            }
        }
        // head (mesmo do Transformer)
        let out_norm_w = self.read_tensor_f32_cached(mem, "output_norm.weight").unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
        rms_norm_into(&hidden_cur, &out_norm_w[..h.min(out_norm_w.len())], &mut norm_buf[..h]);
        let hidden_norm_final = norm_buf[..h].to_vec();
        let vocab = self.tokenizer.vocab_size();
        Ok(self.logits_from_head_cached(mem, &hidden_norm_final, h, vocab))
    }

    /// LM head via cache f32 (caminho CPU legado): norm final + output.weight.
    fn logits_from_head_cached(
        &mut self,
        mem: &crate::vm::MemBackend,
        hidden_norm_final: &[f32],
        h: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let lm_head = self.read_tensor_f32_cached(mem, "output.weight")
            .or_else(|_| self.read_tensor_f32_cached(mem, "token_embd.weight"))
            .unwrap_or_else(|_| {
                let h_dummy = self.config.hidden;
                let mut dummy = vec![0.0f32; vocab * h_dummy];
                for i in 0..dummy.len() { dummy[i] = ((i as f32 * 0.002).sin() * 0.3); }
                std::sync::Arc::new(dummy)
            }); // tied
        // lm_head shape [vocab, hidden] ou [hidden, vocab] — tenta ambos
        let mut logits = vec![0.0; vocab];
        // Se lm_head len == vocab*h, assume [vocab, hidden]
        if lm_head.len() == vocab * h {
            for i in 0..vocab {
                let mut sum = 0.0;
                for j in 0..h { sum += hidden_norm_final[j] * lm_head[i * h + j]; }
                logits[i] = sum;
            }
        } else {
            // fallback dummy
            for i in 0..vocab.min(16) { logits[i % vocab] = (i as f32 * 0.1).sin(); }
        }
        logits
    }

    /// GPU híbrida: tenta o tensor offloaded; cai no `matvec_layer` (CPU).
    /// Sem feature wgpu = direto CPU (zero custo).
    fn matvec_layer_gpu(&mut self, mem: &crate::vm::MemBackend, blk: usize, f: impl Fn(&LayerNames) -> Option<usize>, x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        #[cfg(feature = "wgpu")]
        {
            if let Some(y) = self.gpu_layer(blk, &f, x, in_dim, out_dim) {
                return y;
            }
        }
        self.matvec_layer(mem, blk, f, x, in_dim, out_dim)
    }

    /// GPU híbrida, 1 chamada por peso (gate/up/down). `None` = CPU.
    #[cfg(feature = "wgpu")]
    fn gpu_layer(&mut self, blk: usize, f: &impl Fn(&LayerNames) -> Option<usize>, x: &[f32], in_dim: usize, out_dim: usize) -> Option<Vec<f32>> {
        let idx = self.layer_name(blk, f)?;
        let gpu = self.gpu.as_mut()?;
        gpu.matvec(idx, x, in_dim, out_dim)
    }

    /// Logits via head offloaded (output.weight Q4K/Q6K). `None` = CPU.
    #[cfg(feature = "wgpu")]
    fn gpu_logits(&mut self, hidden_norm_final: &[f32]) -> Option<Vec<f32>> {
        let h = self.config.hidden;
        let vocab = self.tokenizer.vocab_size();
        let hi = self.gpu.as_ref()?.head?;
        self.gpu.as_mut()?.matvec(hi, hidden_norm_final, h, vocab)
    }

    /// Offload híbrido (llama.cpp-style): gate/up/down + head para buffers
    /// persistentes da GPU, 1×. Retorna nº de tensores offloaded (0 = CPU puro).
    /// `M3_GPU=0` desliga. Idempotente para chamadas repetidas (re-sobe).
    #[cfg(feature = "wgpu")]
    pub fn offload_gpu(&mut self, mem: &crate::vm::MemBackend) -> usize {
        if std::env::var("M3_GPU").map(|v| v == "0").unwrap_or(false) {
            eprintln!("[gpu] desligada via M3_GPU=0");
            return 0;
        }
        if self.layer_names.len() != self.config.n_layers {
            self.resolve_layer_names();
        }
        let h = self.config.hidden;
        let inter = self.config.intermediate;
        let vocab = self.tokenizer.vocab_size();
        let max_in = h.max(inter);
        let max_out = inter.max(h).max(vocab);
        let mut gpu = match crate::inference_gpu::GpuOffload::try_new(max_in, max_out) {
            Some(g) => g,
            None => return 0,
        };
        let mut n = 0;
        // gate/up/down de todas as camadas (só Q4K/Q6K alinhados sobem)
        for blk in 0..self.config.n_layers {
            let ln = &self.layer_names[blk];
            for (tidx, (idim, odim)) in [
                (ln.gate, (h, inter)),
                (ln.up, (h, inter)),
                (ln.down, (inter, h)),
            ] {
                if let Some(idx) = tidx {
                    if self.gguf_offload_one(mem, &mut gpu, idx, idim, odim) {
                        n += 1;
                    }
                }
            }
        }
        // head: output.weight (ou token_embd amarrado); só Q4K/Q6K sobem
        let head_name = if self.gguf.find_tensor("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        if let Some(hi) = self.gguf.tensors.iter().position(|t| t.name == head_name) {
            if self.gguf_offload_one(mem, &mut gpu, hi, h, vocab) {
                gpu.head = Some(hi);
                n += 1;
            }
        }
        eprintln!("[gpu] offload: {} tensores (+head={}) em buffers persistentes", gpu.tensor_count(), gpu.head.is_some());
        self.gpu = Some(gpu);
        n
    }

    /// Sobe 1 tensor (Q4K/Q6K) lendo zero-copy quando possível.
    #[cfg(feature = "wgpu")]
    fn gguf_offload_one(
        &self,
        mem: &crate::vm::MemBackend,
        gpu: &mut crate::inference_gpu::GpuOffload,
        idx: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> bool {
        let info = match self.gguf.tensors.get(idx) {
            Some(t) => t,
            None => return false,
        };
        let n = info.n_elements;
        if n != in_dim * out_dim {
            return false;
        }
        let raw_len = match crate::matvec_quant::quant_raw_len(info.dtype, n) {
            Some(l) => l,
            None => return false,
        };
        let fo = self.gguf.data_offset + info.offset;
        if let Some(raw) = mem.read_model_raw(fo, raw_len) {
            if gpu.upload(idx, info.dtype, raw, in_dim, out_dim) {
                return true;
            }
        }
        let paddr = crate::memory::make_persistent_addr(fo as u128);
        match mem.read(paddr, raw_len) {
            Ok(raw) => gpu.upload(idx, info.dtype, &raw, in_dim, out_dim),
            Err(_) => false,
        }
    }

    /// Atalho P0.1: vetor de camada (norms/bias) por índice, com dummy.
    fn cached_layer_vec(&mut self, mem: &crate::vm::MemBackend, blk: usize, f: impl Fn(&LayerNames) -> Option<usize>, dummy: Vec<f32>) -> std::sync::Arc<Vec<f32>> {
        if let Some(idx) = self.layer_name(blk, f) {
            if let Ok(w) = self.read_tensor_f32_cached_idx(mem, idx) {
                return w;
            }
        }
        std::sync::Arc::new(dummy)
    }

    /// Variante Option (bias: ausente = skip, sem dummy).
    fn cached_layer_opt(&mut self, mem: &crate::vm::MemBackend, blk: usize, f: impl Fn(&LayerNames) -> Option<usize>) -> Option<std::sync::Arc<Vec<f32>>> {
        self.layer_name(blk, f).and_then(|idx| self.read_tensor_f32_cached_idx(mem, idx).ok())
    }

    /// Linha `row` de matriz [rows, hidden] row-major sem dequantizar tudo.
    /// Evita materializar o embedding 151936×1536 (~934MB f32) no cache:
    /// lê+decodifica só os blocos da linha (Q4_K: 6×144B para hidden 1536).
    /// Paridade exata com fatiar a matriz full (mesmos bytes, mesmo dequant).
    fn embedding_row(&self, mem: &crate::vm::MemBackend, name: &str, row: usize, hidden: usize) -> Result<Vec<f32>> {
        let info = self.gguf.find_tensor(name).ok_or_else(|| anyhow!("tensor {} não encontrado", name))?;
        let n = info.n_elements;
        let base = row.checked_mul(hidden).ok_or_else(|| anyhow!("overflow embedding row"))?;
        if hidden == 0 || base + hidden > n {
            return Err(anyhow!("row {} fora de {} (n={})", row, name, n));
        }
        let file_offset = self.gguf.data_offset + info.offset;
        match DType::from_u32(info.dtype) {
            DType::F32 => {
                let paddr = crate::memory::make_persistent_addr(file_offset as u128 + (base * 4) as u128);
                let raw = mem.read(paddr, hidden * 4)?;
                Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
            }
            DType::F16 => {
                let paddr = crate::memory::make_persistent_addr(file_offset as u128 + (base * 2) as u128);
                let raw = mem.read(paddr, hidden * 2)?;
                Ok(raw.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()).collect())
            }
            _ => {
                let (block_elems, block_bytes) = match info.dtype {
                    2 => (32usize, 18usize),    // Q4_0
                    8 => (32usize, 34usize),    // Q8_0
                    12 | 13 => (256usize, 144usize), // Q4_K (Q5_K igual aqui)
                    14 => (256usize, 210usize), // Q6_K
                    _ => return Err(anyhow!("dtype {} sem row-slice", info.dtype)),
                };
                if base % block_elems != 0 || hidden % block_elems != 0 {
                    return Err(anyhow!("row {} não alinhada a bloco {}", row, block_elems));
                }
                let b0 = base / block_elems;
                let nb = hidden / block_elems;
                let paddr = crate::memory::make_persistent_addr(file_offset as u128 + (b0 * block_bytes) as u128);
                let raw = mem.read(paddr, nb * block_bytes)?;
                let mut dst = vec![0.0f32; hidden];
                if !crate::quant::dequantize(&raw, info.dtype, &mut dst, hidden) {
                    return Err(anyhow!("dequant row falhou para {}", name));
                }
                Ok(dst)
            }
        }
    }

    /// Forward de 1 token (id) com 22 layers reais e KV cache per-layer (0x30 região lógica)
    /// Implementa loop 22 camadas com attention sobre cache sequencial:
    ///   - append K/V da camada atual ao cache (seq_len cresce)
    ///   - scores = Q · K_cache^T / sqrt(head_dim) para cada posição i
    ///   - softmax(scores) -> attn_weights
    ///   - attn_out = Σ_i w_i * V_cache[i]
    /// Isso garante O(seq_len) por camada com memória O(layers*seq_len*hidden) = 1GiB lógico
    pub fn forward_one(&mut self, mem: &crate::vm::MemBackend, token_id: u32) -> Result<Vec<f32>> {
        // Profiler opcional (P0.2: flag lida 1× no new(), sem syscall por token)
        let profile = self.profile;
        let mut t_matvec = 0u128;
        let mut t_attn = 0u128;
        let mut t_norm_ffn = 0u128;
        let now = std::time::Instant::now;
        if self.kv_cache_k.len() != self.config.n_layers {
            self.kv_cache_k = vec![Vec::new(); self.config.n_layers];
            self.kv_cache_v = vec![Vec::new(); self.config.n_layers];
        }
        // Embedding: row-slice direto (1 linha, sem materializar 934MB).
        // Fallback legado: matriz full via cache, ou dummy senoidal sem modelo.
        let h = self.config.hidden;
        let mut hidden = vec![0.0f32; h];
        let row = self.embedding_row(mem, "token_embd.weight", token_id as usize, h)
            .or_else(|_| self.embedding_row(mem, "tok_embeddings.weight", token_id as usize, h));
        match row {
            Ok(r) => hidden.copy_from_slice(&r),
            Err(_) => {
                let emb = self.read_tensor_f32_cached(mem, "token_embd.weight")
                    .or_else(|_| self.read_tensor_f32_cached(mem, "tok_embeddings.weight"))
                    .unwrap_or_else(|_| {
                        let vocab = self.tokenizer.vocab_size();
                        let h_dummy = self.config.hidden;
                        let mut dummy = vec![0.0f32; vocab * h_dummy];
                        for i in 0..dummy.len() { dummy[i] = ((i as f32 * 0.001).sin() * 0.5); }
                        std::sync::Arc::new(dummy)
                    });
                let off = token_id as usize * h;
                if off + h <= emb.len() {
                    hidden.copy_from_slice(&emb[off..off+h]);
                } else {
                    for i in 0..h { hidden[i] = (token_id as f32 * 0.01 + i as f32 * 0.001).sin(); }
                }
            }
        }

        // Loop sobre todas as camadas reais (22 para TinyLlama, 28 para DeepSeek) — tese 2.1: KV_CACHE 0x30
        // P0.4: arena reutilizável p/ rms_norm (evita ~57 allocs/token)
        let mut norm_buf = vec![0.0f32; self.config.intermediate.max(h)];
        let mut hidden_cur = hidden;
        let n_heads = self.config.n_heads.max(1);
        let head_dim = if self.config.hidden % n_heads == 0 { self.config.hidden / n_heads } else { self.config.hidden };
        let scale = (head_dim as f32).sqrt().recip();
        // RoPE: posição = seq_len atual (igual em todas as camadas); freqs 1×/token
        let pos = self.kv_cache_k.first().map(|k| k.len() / h.max(1)).unwrap_or(0);
        let rope_hd = self.config.head_dim();
        let rope_theta = self.config.rope_theta;
        let rope_freqs: Vec<(f32, f32)> = (0..rope_hd.max(2) / 2)
            .map(|i| {
                let f = pos as f32 * rope_theta.powf(-2.0 * i as f32 / rope_hd.max(1) as f32);
                (f.cos(), f.sin())
            })
            .collect();
        for blk in 0..self.config.n_layers {
            let t_blk = now();
            // Arc emprestado (sem clone de dados); dummy 1.0 só sem modelo
            let gamma_attn = self.cached_layer_vec(mem, blk, |l| l.attn_norm, vec![1.0; h]);
            rms_norm_into(&hidden_cur, &gamma_attn[..], &mut norm_buf[..h]);
            let hidden_norm = &norm_buf[..h];

            // Suporte GQA: q = h*h, k/v = h*kv_hidden (kv_hidden = n_kv_heads * head_dim)
            let kv_hidden = self.config.n_kv_heads * self.config.head_dim();
            // Kernel int4 direto quando possível (bypass dequant+clone)
            let t0 = now();
            let mut q = self.matvec_layer(mem, blk, |l| l.q, &hidden_norm, h, h);
            let mut k_small = self.matvec_layer(mem, blk, |l| l.k, &hidden_norm, h, kv_hidden);
            let mut v_small = self.matvec_layer(mem, blk, |l| l.v, &hidden_norm, h, kv_hidden);
            let o_w_idx = self.layer_name(blk, |l| l.o);
            if profile { t_matvec += t0.elapsed().as_micros(); }
            // Bias q/k/v (Qwen2/DeepSeek têm attn_*.bias F32; ausente = skip)
            if let Some(b) = self.cached_layer_opt(mem, blk, |l| l.qbias) {
                for (a, bb) in q.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Some(b) = self.cached_layer_opt(mem, blk, |l| l.kbias) {
                for (a, bb) in k_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Some(b) = self.cached_layer_opt(mem, blk, |l| l.vbias) {
                for (a, bb) in v_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            // RoPE em q/k ANTES do KV append e do repeat GQA
            apply_rope(&mut q, self.config.n_heads, rope_hd, &rope_freqs);
            apply_rope(&mut k_small, self.config.n_kv_heads, rope_hd, &rope_freqs);
            // Expande GQA se necessário (kv_hidden < h)
            // P1.3: append direto no KV (GQA ou cópia), sem Vec temporário
            if kv_hidden == h {
                self.kv_cache_k[blk].extend_from_slice(&k_small);
                self.kv_cache_v[blk].extend_from_slice(&v_small);
            } else {
                Self::repeat_kv_extend(&mut self.kv_cache_k[blk], &k_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim());
                Self::repeat_kv_extend(&mut self.kv_cache_v[blk], &v_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim());
            }

            // --- KV Cache append (0x30 região) ---
            // (já estendido acima, com ou sem repeat GQA)
            let seq_len = self.kv_cache_k[blk].len() / h;
            debug_assert_eq!(self.kv_cache_v[blk].len() / h, seq_len);

            // Attention PER-HEAD sobre o cache (MHA/GQA correto: softmax por head,
            // não sobre o vetor 1536 concatenado — a versão conjunta saturava
            // numa posição e congelava o hidden).
            let t_attn0 = now();
            let k_cache = &self.kv_cache_k[blk];
            let v_cache = &self.kv_cache_v[blk];
            let mut attn_agg = vec![0.0f32; h];
            for hh in 0..n_heads {
                let hb = hh * head_dim;
                if hb + head_dim > h || hb + head_dim > q.len() {
                    break; // config degenerada (testes dummy): ignora heads extras
                }
                let qb = &q[hb..hb + head_dim];
                let mut scores = vec![0.0f32; seq_len];
                for i in 0..seq_len {
                    let kb = &k_cache[i * h + hb..i * h + hb + head_dim];
                    let mut dot = 0.0f32;
                    for (a, b) in qb.iter().zip(kb.iter()) { dot += a * b; }
                    scores[i] = dot * scale;
                }
                // causal softmax por head (todo o cache é passado; futuro não existe)
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
                for s in scores.iter_mut() { *s /= sum; }
                for i in 0..seq_len {
                    let vb = &v_cache[i * h + hb..i * h + hb + head_dim];
                    let w = scores[i];
                    for j in 0..head_dim { attn_agg[hb + j] += w * vb[j]; }
                }
            }
            if profile { t_attn += t_attn0.elapsed().as_micros(); }

            let t1 = now();
            let attn_out = self.matvec_weight_pre(mem, o_w_idx, &attn_agg, h, h);
            let mut hidden2 = vec![0.0; h];
            for i in 0..h { hidden2[i] = hidden_cur[i] + attn_out[i]; }

            let gamma_ffn = self.cached_layer_vec(mem, blk, |l| l.ffn_norm, vec![1.0; h]);
            rms_norm_into(&hidden2, &gamma_ffn[..], &mut norm_buf[..h]);
            let hidden2_norm = &norm_buf[..h];
            let inter = self.config.intermediate;
            let gate = self.matvec_layer_gpu(mem, blk, |l| l.gate, &hidden2_norm, h, inter);
            let up = self.matvec_layer_gpu(mem, blk, |l| l.up, &hidden2_norm, h, inter);
            let mut ffn_h = vec![0.0; self.config.intermediate];
            for i in 0..self.config.intermediate {
                let g = gate[i];
                let sig = 1.0/(1.0+(-g).exp());
                ffn_h[i] = g * sig * up[i];
            }
            let ffn_out = self.matvec_layer_gpu(mem, blk, |l| l.down, &ffn_h, inter, h);
            if profile { t_matvec += t1.elapsed().as_micros(); t_norm_ffn += t_blk.elapsed().as_micros(); }
            let mut hidden_next = vec![0.0; h];
            for i in 0..h { hidden_next[i] = hidden2[i] + ffn_out[i]; }
            hidden_cur = hidden_next;
        }

        // Final norm + LM head (cache, Arc emprestado — sem clone de 934MB)
        let t_head0 = now();
        let out_norm_w = self.read_tensor_f32_cached(mem, "output_norm.weight").unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
        rms_norm_into(&hidden_cur, &out_norm_w[..], &mut norm_buf[..h]);
        let hidden_norm_final = &norm_buf[..h];
        // Head: GPU offloaded primeiro; senão caminho f32 legado.
        let vocab = self.tokenizer.vocab_size();
        #[cfg(feature = "wgpu")]
        let logits = match self.gpu_logits(hidden_norm_final) {
            Some(g) => g,
            None => self.logits_from_head_cached(mem, hidden_norm_final, h, vocab),
        };
        #[cfg(not(feature = "wgpu"))]
        let logits = self.logits_from_head_cached(mem, hidden_norm_final, h, vocab);
        if profile {
            // t_norm_ffn inclui t_matvec+t_attn (tempo de bloco); fases em ms
            // + sanidade dos logits: top1/entropia denunciam pico sistemático
            let mut top1 = 0usize;
            let mut topv = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > topv { topv = v; top1 = i; }
            }
            let maxl = topv;
            let mut se = 0.0f32;
            let mut ssum = 0.0f32;
            for &v in logits.iter() {
                let e = (v - maxl).exp();
                ssum += e;
            }
            for &v in logits.iter() {
                let p = (v - maxl).exp() / ssum;
                if p > 1e-12 { se -= p * p.ln(); }
            }
            let top1p = ((topv - maxl).exp() / ssum) * 100.0;
            let hmean: f32 = hidden_norm_final.iter().map(|a| a.abs()).sum::<f32>() / hidden_norm_final.len() as f32;
            eprintln!("[profile tok {}] layersTOTAL={:.0}ms matvec={:.0}ms attn={:.0}ms norm+ffn-resto={:.0}ms head={:.0}ms top1={}({:.1}%) H={:.2} |h|={:.3}",
                token_id, t_norm_ffn as f64 / 1000.0, t_matvec as f64 / 1000.0,
                t_attn as f64 / 1000.0,
                (t_norm_ffn as f64 - t_matvec as f64 - t_attn as f64).max(0.0) / 1000.0,
                t_head0.elapsed().as_micros() as f64 / 1000.0,
                top1, top1p, se, hmean);
        }
        Ok(logits)
    }

    /// Variante que também sincroniza KV cache com a região 0x30 do MemoryManager (MemBackend)
    /// Útil para demonstrar que loop 22 usa memória física 0x30... e snapshots.
    pub fn forward_one_with_mem_kv(&mut self, mem: &mut crate::vm::MemBackend, token_id: u32) -> Result<Vec<f32>> {
        if mem.kv_cache_n_layers() != self.config.n_layers {
            mem.kv_cache_init(self.config.n_layers, self.config.hidden);
        }
        // Reusa lógica de forward_one mas também espelha para mem
        // Embedding via row-slice (sem 934MB); fallback legado abaixo.
        let h = self.config.hidden;
        let mut hidden = vec![0.0f32; h];
        let row = self.embedding_row(mem, "token_embd.weight", token_id as usize, h)
            .or_else(|_| self.embedding_row(mem, "tok_embeddings.weight", token_id as usize, h));
        match row {
            Ok(r) => hidden.copy_from_slice(&r),
            Err(_) => {
                let emb = self.read_tensor_f32_cached(mem, "token_embd.weight")
                    .or_else(|_| self.read_tensor_f32_cached(mem, "tok_embeddings.weight"))
                    .unwrap_or_else(|_| {
                        let vocab = self.tokenizer.vocab_size();
                        let h_dummy = self.config.hidden;
                        let mut dummy = vec![0.0f32; vocab * h_dummy];
                        for i in 0..dummy.len() { dummy[i] = ((i as f32 * 0.001).sin() * 0.5); }
                        std::sync::Arc::new(dummy)
                    });
                let off = token_id as usize * h;
                if off + h <= emb.len() {
                    hidden.copy_from_slice(&emb[off..off+h]);
                } else {
                    for i in 0..h { hidden[i] = (token_id as f32 * 0.01 + i as f32 * 0.001).sin(); }
                }
            }
        }
        let mut hidden_cur = hidden;
        let n_heads = self.config.n_heads.max(1);
        let head_dim = if h % n_heads == 0 { h / n_heads } else { h };
        let scale = (head_dim as f32).sqrt().recip();
        // RoPE: mesma convenção do forward_one (posição = seq_len atual)
        let pos = self.kv_cache_k.first().map(|k| k.len() / h.max(1)).unwrap_or(0);        let rope_hd = self.config.head_dim();
        let rope_theta = self.config.rope_theta;
        let rope_freqs: Vec<(f32, f32)> = (0..rope_hd.max(2) / 2)
            .map(|i| {
                let f = pos as f32 * rope_theta.powf(-2.0 * i as f32 / rope_hd.max(1) as f32);
                (f.cos(), f.sin())
            })
            .collect();
        // P0.4: arena reutilizável p/ rms_norm (igual ao forward_one)
        let mut norm_buf = vec![0.0f32; self.config.intermediate.max(h)];
        for blk in 0..self.config.n_layers {
            let gamma_attn = self.cached_layer_vec(mem, blk, |l| l.attn_norm, vec![1.0; h]);
            rms_norm_into(&hidden_cur, &gamma_attn[..], &mut norm_buf[..h]);
            let hidden_norm = &norm_buf[..h];
            let kv_hidden = self.config.n_kv_heads * self.config.head_dim();
            let mut q = self.matvec_layer(mem, blk, |l| l.q, &hidden_norm, h, h);
            let mut k_small = self.matvec_layer(mem, blk, |l| l.k, &hidden_norm, h, kv_hidden);
            let mut v_small = self.matvec_layer(mem, blk, |l| l.v, &hidden_norm, h, kv_hidden);
            let o_w_idx = self.layer_name(blk, |l| l.o);
            if let Some(b) = self.cached_layer_opt(mem, blk, |l| l.qbias) {
                for (a, bb) in q.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Some(b) = self.cached_layer_opt(mem, blk, |l| l.kbias) {
                for (a, bb) in k_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Some(b) = self.cached_layer_opt(mem, blk, |l| l.vbias) {
                for (a, bb) in v_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            apply_rope(&mut q, self.config.n_heads, rope_hd, &rope_freqs);
            apply_rope(&mut k_small, self.config.n_kv_heads, rope_hd, &rope_freqs);
            // P1.3: append direto no KV (GQA ou cópia), sem Vec temporário
            if kv_hidden == h {
                self.kv_cache_k[blk].extend_from_slice(&k_small);
                self.kv_cache_v[blk].extend_from_slice(&v_small);
            } else {
                Self::repeat_kv_extend(&mut self.kv_cache_k[blk], &k_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim());
                Self::repeat_kv_extend(&mut self.kv_cache_v[blk], &v_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim());
            }
            // internal cache (já estendido acima)
            // mem cache (0x30): espelha a cauda recém-anexada (borrow disjunto)
            let (kk, vv) = (self.kv_cache_k[blk].len(), self.kv_cache_v[blk].len());
            let _ = mem.kv_cache_append(blk, &self.kv_cache_k[blk][kk - h..kk], &self.kv_cache_v[blk][vv - h..vv]);
            let seq_len = self.kv_cache_k[blk].len() / h;
            let k_cache = &self.kv_cache_k[blk];
            let v_cache = &self.kv_cache_v[blk];
            // Attention PER-HEAD (igual ao forward_one: softmax por head)
            let mut attn_agg = vec![0.0f32; h];
            for hh in 0..n_heads {
                let hb = hh * head_dim;
                if hb + head_dim > h || hb + head_dim > q.len() {
                    break;
                }
                let qb = &q[hb..hb + head_dim];
                let mut scores = vec![0.0f32; seq_len];
                for i in 0..seq_len {
                    let kb = &k_cache[i * h + hb..i * h + hb + head_dim];
                    let mut dot = 0.0f32;
                    for (a, b) in qb.iter().zip(kb.iter()) { dot += a * b; }
                    scores[i] = dot * scale;
                }
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
                for s in scores.iter_mut() { *s /= sum; }
                for i in 0..seq_len {
                    let vb = &v_cache[i * h + hb..i * h + hb + head_dim];
                    let w = scores[i];
                    for j in 0..head_dim { attn_agg[hb + j] += w * vb[j]; }
                }
            }
            let attn_out = self.matvec_weight_pre(mem, o_w_idx, &attn_agg, h, h);
            let mut hidden2 = vec![0.0; h];
            for i in 0..h { hidden2[i] = hidden_cur[i] + attn_out[i]; }
            let gamma_ffn = self.cached_layer_vec(mem, blk, |l| l.ffn_norm, vec![1.0; h]);
            rms_norm_into(&hidden2, &gamma_ffn[..], &mut norm_buf[..h]);
            let hidden2_norm = &norm_buf[..h];
            let inter = self.config.intermediate;
            let gate = self.matvec_layer(mem, blk, |l| l.gate, &hidden2_norm, h, inter);
            let up = self.matvec_layer(mem, blk, |l| l.up, &hidden2_norm, h, inter);
            let mut ffn_h = vec![0.0; self.config.intermediate];
            for i in 0..self.config.intermediate {
                let g = gate[i];
                let sig = 1.0/(1.0+(-g).exp());
                ffn_h[i] = g * sig * up[i];
            }
            let ffn_out = self.matvec_layer(mem, blk, |l| l.down, &ffn_h, inter, h);
            let mut hidden_next = vec![0.0; h];
            for i in 0..h { hidden_next[i] = hidden2[i] + ffn_out[i]; }
            hidden_cur = hidden_next;
        }
        let out_norm_w = self.read_tensor_f32_cached(mem, "output_norm.weight").unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
        rms_norm_into(&hidden_cur, &out_norm_w[..], &mut norm_buf[..h]);
        let hidden_norm_final = &norm_buf[..h];
        let lm_head = self.read_tensor_f32_cached(mem, "output.weight")
            .or_else(|_| self.read_tensor_f32_cached(mem, "token_embd.weight"))
            .unwrap_or_else(|_| {
                let vocab = self.tokenizer.vocab_size();
                let h_dummy = self.config.hidden;
                let mut dummy = vec![0.0f32; vocab * h_dummy];
                for i in 0..dummy.len() { dummy[i] = ((i as f32 * 0.002).sin() * 0.3); }
                std::sync::Arc::new(dummy)
            });
        let vocab = self.tokenizer.vocab_size();
        let mut logits = vec![0.0; vocab];
        if lm_head.len() == vocab * h {
            for i in 0..vocab {
                let mut sum=0.0;
                for j in 0..h { sum += hidden_norm_final[j] * lm_head[i*h + j]; }
                logits[i]=sum;
            }
        } else {
            for i in 0..vocab.min(16) { logits[i%vocab] = (i as f32 * 0.1).sin(); }
        }
        Ok(logits)
    }

    pub fn sample(&self, logits: &[f32]) -> u32 {
        self.sample_with_params(logits, 0.7, 0.9, 40, 1.1)
    }

    /// Sampling com temperatura, top_p, top_k e repetition_penalty.
    /// Overrides via env (padrões entre parênteses): M3_TEMP (0.7), M3_TOP_P
    /// (0.9), M3_TOP_K (40), M3_REPPEN (1.1). Ex.: `M3_TEMP=0.3 M3_TOP_K=1`
    /// para saída gulosa/determinística em modelos fracos.
    pub fn sample_with_params(&self, logits: &[f32], temp: f32, top_p: f32, top_k: usize, repeat_penalty: f32) -> u32 {
        if logits.is_empty() { return 0; }
        let env_f = |k: &str, dflt: f32| {
            std::env::var(k).ok().and_then(|v| v.parse::<f32>().ok()).filter(|v| v.is_finite() && *v > 0.0).unwrap_or(dflt)
        };
        let env_k = || {
            std::env::var("M3_TOP_K").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(top_k)
        };
        let (temp, top_p, top_k, repeat_penalty) =
            (env_f("M3_TEMP", temp), env_f("M3_TOP_P", top_p), env_k(), env_f("M3_REPPEN", repeat_penalty));
        // Repetition penalty estilo llama.cpp sobre os últimos 64 ids
        // (prompt + gerados, registrados via push_gen/extend_gen).
        let mut adj_logits = logits.to_vec();
        if repeat_penalty != 1.0 && !self.gen_history.is_empty() {
            let start = self.gen_history.len().saturating_sub(64);
            for &id in &self.gen_history[start..] {
                if (id as usize) < adj_logits.len() {
                    let l = &mut adj_logits[id as usize];
                    if *l < 0.0 { *l *= repeat_penalty; } else { *l /= repeat_penalty; }
                }
            }
        }
        // Temperatura
        let temp = temp.max(0.01);
        for v in adj_logits.iter_mut() { *v /= temp; }
        // Top-k truncamento
        let mut indexed: Vec<(usize, f32)> = adj_logits.iter().enumerate().map(|(i,&v)| (i,v)).collect();
        indexed.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let k = top_k.min(indexed.len()).max(1);
        indexed.truncate(k);
        // Softmax com top_p (nucleus)
        let max = indexed.iter().map(|(_,v)| *v).fold(f32::NEG_INFINITY, f32::max);
        let mut exps: Vec<(usize,f32)> = indexed.into_iter().map(|(i,v)| (i, (v-max).exp())).collect();
        exps.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());
        let sum: f32 = exps.iter().map(|(_,e)| *e).sum();
        // Top-p cutoff
        let mut cum = 0.0;
        let mut cutoff = exps.len();
        for (idx, (_,p)) in exps.iter().enumerate() {
            cum += p / sum;
            if cum >= top_p {
                cutoff = idx+1;
                break;
            }
        }
        exps.truncate(cutoff.max(1));
        let sum2: f32 = exps.iter().map(|(_,e)| *e).sum();
        let mut rng = rand::thread_rng();
        use rand::Rng;
        let mut r: f32 = rng.gen_range(0.0..sum2);
        for (i,e) in exps.iter().cloned() {
            r -= e;
            if r <= 0.0 { return i as u32; }
        }
        exps.first().map(|(i,_)| *i as u32).unwrap_or(0)
    }

    /// Prefill: alimenta KV cache com todos os tokens do prompt sem amostrar
    pub fn prefill(&mut self, mem: &crate::vm::MemBackend, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() { return Ok(()); }
        // Garante cache limpo se for novo prompt
        for &tid in tokens {
            let _ = self.forward_one_auto(mem, tid)?;
            self.push_gen(tid);
        }
        // Remove último token do cache? Não, prefill deixa todos no cache; próximo forward será o próximo token
        // Mas forward_one já adicionou todos; para gerar próximo, basta chamar forward_one com último token
        // Então truncamos o último para não duplicar? Não, mantemos todos.
        // O caller deve usar o último token como current_token para próxima geração
        Ok(())
    }

    /// Prefill otimizado sem gerar logits intermediários desnecessários (só último gera logits)
    pub fn prefill_logits(&mut self, mem: &crate::vm::MemBackend, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() { return Ok(vec![0.0; self.tokenizer.vocab_size()]); }
        let mut last_logits = Vec::new();
        for (idx, &tid) in tokens.iter().enumerate() {
            let logits = self.forward_one_auto(mem, tid)?;
            self.push_gen(tid);
            if idx == tokens.len() - 1 {
                last_logits = logits;
            }
        }
        Ok(last_logits)
    }

    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        // BPE greedy longest-match sobre o vocab GGUF real.
        // Estilo: Qwen/GPT-2 usa Ġ (espaço) e Ċ (\n); SentencePiece usa ▁.
        // O pré-processamento traduz os separadores e o greedy resolve o resto
        // (palavras com pontuação, especiais <｜User｜> etc. via lookup exato —
        //  sem ids hardcoded: eles variam por arquivo, ex. 151643 é EOS aqui).
        let is_gpt2 = self.tokenizer.vocab.iter().any(|v| v.starts_with('Ġ'));
        let pre = if is_gpt2 {
            text.replace(' ', "Ġ").replace('\n', "Ċ")
        } else {
            text.replace(' ', "▁")
        };
        if pre.is_empty() {
            return vec![0];
        }
        let map: HashMap<&str, u32> = self
            .tokenizer
            .vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.as_str(), i as u32))
            .collect();
        let mut ids = Vec::new();
        let mut pos = 0;
        while pos < pre.len() {
            // longest match a partir de pos (peças BPE têm < 64 bytes; cap evita
            // varrer a string toda quando nada casa)
            let mut matched: Option<u32> = None;
            let mut end = pre.len().min(pos + 64);
            if !pre.is_char_boundary(end) {
                // ajusta para fronteira válida mais próxima abaixo
                while end > pos && !pre.is_char_boundary(end) {
                    end -= 1;
                }
            }
            let mut e = end;
            while e > pos {
                if let Some(&id) = map.get(&pre[pos..e]) {
                    matched = Some(id);
                    end = e;
                    break;
                }
                // recua 1 char
                e -= 1;
                while e > pos && !pre.is_char_boundary(e) {
                    e -= 1;
                }
            }
            if let Some(id) = matched {
                ids.push(id);
                pos = end;
                continue;
            }
            // Fallback byte a byte: formas <0xNN> (GGUF), byte cru, id==byte (Qwen)
            let ch = pre[pos..].chars().next().unwrap();
            let mut done = false;
            if ch.len_utf8() == 1 {
                let b = ch as u8;
                for cand in [format!("<0x{:02X}>", b), format!("<0x{:02x}>", b)] {
                    if let Some(&id) = map.get(cand.as_str()) {
                        ids.push(id);
                        done = true;
                        break;
                    }
                }
                if !done {
                    let s = &pre[pos..pos + 1];
                    if let Some(&id) = map.get(s) {
                        ids.push(id);
                        done = true;
                    } else if (b as u32) < self.tokenizer.vocab_size() as u32 && is_gpt2 {
                        // Qwen: bytes 0..255 são os ids 0..255
                        ids.push(b as u32);
                        done = true;
                    }
                }
            } else {
                // char multibyte sem peça: tenta string crua, senão pula
                if let Some(&id) = map.get(&pre[pos..pos + ch.len_utf8()]) {
                    ids.push(id);
                    done = true;
                }
            }
            pos += ch.len_utf8();
            let _ = done;
        }
        if ids.is_empty() {
            ids.push(0);
        }
        ids
    }

    pub fn format_chat(&self, user_text: &str) -> String {
        // Usa template simples por arquitetura; ignora tokenizer.chat_template Jinja (muito grande).
        // ATENÇÃO: literais de especiais variam por arquivo GGUF — aqui usamos os
        // literais que existem NESTE vocab (ex. DeepSeek-R1-Distill-Qwen usa os
        // fullwidth <｜User｜>/<｜Assistant｜>, não <|im_start|> ASCII).
        // Se os marcadores não existirem no vocab (ex. TinyLlama Q4_K_M sem
        // <|user|>), usa formato com palavras normais — emitir marcador
        // fragmentado em peças envenena o prompt.
        let has = |lit: &str| self.tokenizer.vocab.iter().any(|v| v == lit);
        match self.config.arch.as_str() {
            "qwen2" | "qwen" => {
                // DeepSeek-R1-Distill-Qwen: <｜begin▁of▁sentence｜><｜User｜>{user}<｜Assistant｜>
                // (bos=151646; geração termina em <｜end▁of▁sentence｜>=151643)
                format!("<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>", user_text)
            },
            "llama" | "tinyllama" => {
                // BOS (<s>) abre toda sequência Llama; sem ele o contexto sai
                // da distribuição de treino. Marcadores <|user|> só se existirem.
                if has("<|im_start|>") && has("<|im_end|>") {
                    // SmolLM2 e similares: <|im_start|> também é BOS.
                    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_text)
                } else if has("<|user|>") && has("<|assistant|>") {
                    format!("<s><|user|>\n{}\n<|assistant|>\n", user_text)
                } else {
                    format!("<s>User: {}\nAssistant:", user_text)
                }
            },
            _ => {
                // Fallback genérico
                if self.config.arch.contains("qwen") || self.config.arch.contains("deepseek") {
                    format!("<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜>", user_text)
                } else if has("<|user|>") {
                    format!("<s><|user|>\n{}\n<|assistant|>\n", user_text)
                } else {
                    format!("<s>User: {}\nAssistant:", user_text)
                }
            }
        }
    }

    /// EOS deste arquivo: lê `tokenizer.ggml.eos_token_id` do KV, senão procura
    /// literais conhecidos no vocab. `None` = gera até limite.
    pub fn eos_id(&self) -> Option<u32> {
        if let Some(s) = self.gguf.kv.get("tokenizer.ggml.eos_token_id") {
            if let Ok(id) = s.parse::<u32>() {
                return Some(id);
            }
        }
        for lit in ["<｜end▁of▁sentence｜>", "<|im_end|>", "<|endoftext|>", "</s>"] {
            if let Some(pos) = self.tokenizer.vocab.iter().position(|v| v == lit) {
                return Some(pos as u32);
            }
        }
        None
    }
}

/// RoPE estilo HF Llama/Qwen2 (split-half, `rotate_half` — mesma convenção que
/// o modo NEOX do ggml): par (i, i+half) gira pelo ângulo da freq i.
/// `v` tem `n_heads * head_dim` elementos; `freqs` tem `head_dim/2` (cos, sin).
/// `pub` para reuso pelo opcode nativo `OP_ROPE` (`vm.rs::exec_rope`).
pub fn apply_rope(v: &mut [f32], n_heads: usize, head_dim: usize, freqs: &[(f32, f32)]) {
    if head_dim == 0 || head_dim % 2 != 0 || freqs.len() * 2 != head_dim {
        return;
    }
    let half = head_dim / 2;
    for hh in 0..n_heads {
        let base = hh * head_dim;
        if base + head_dim > v.len() {
            break;
        }
        for i in 0..half {
            let (c, s) = freqs[i];
            let a = v[base + i];
            let b = v[base + i + half];
            v[base + i] = a * c - b * s;
            v[base + i + half] = a * s + b * c;
        }
    }
}

/// P0.4: RMSNorm sem alocar — escreve em `out` (arena do chamador).
pub(crate) fn rms_norm_into(x: &[f32], gamma: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let eps = 1e-5;
    let mut sum = 0.0;
    for &v in x { sum += v * v; }
    let mean = sum / x.len() as f32;
    let rms = (mean + eps).sqrt();
    for (i, &v) in x.iter().enumerate() {
        out[i] = v / rms * gamma.get(i).cloned().unwrap_or(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tiny_forward_one_dummy() {
        // Testa com dummy mem (sem GGUF) — deve usar fallback e não panicar (rápido, hidden 32)
        let mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let mem = crate::vm::MemBackend::Cpu(mem_mgr);
        let cfg = ModelConfig { hidden: 32, intermediate: 64, n_layers: 2, n_heads: 4, n_kv_heads: 4, vocab: 32000, arch: "llama".to_string(), context_length: 2048, rope_theta: 10000.0, norm_eps: 1e-5, d_inner: 64, d_state: 16, d_conv: 4, dt_rank: 2, dt_b_c_rms: false };
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: Vec::new(), kv: std::collections::HashMap::new(), data_offset: 0 };
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let mut inf = RealInference { config: cfg, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 2], kv_cache_v: vec![Vec::new(); 2], weight_cache: fxhash::FxHashMap::default(), gen_history: Vec::new(), layer_names: Vec::new(), mamba_names: Vec::new(), ssm_states: Vec::new(), profile: false,
            #[cfg(feature = "wgpu")]
            gpu: None };
        let logits = inf.forward_one(&mem, 0).unwrap();
        println!("logits len {} sample {}", logits.len(), inf.sample(&logits));
        assert_eq!(logits.len(), inf.tokenizer.vocab_size());
        assert!(logits.len() >= 8);
    }
    #[test]
    fn test_tiny_forward_real_config() {
        // Testa que parsing de TinyLlama real retorna 22 layers (sem rodar forward pesado)
        let gg_path = "./models/tinyllama-1.1b-chat-v1.0.fp32.gguf";
        if std::path::Path::new(gg_path).exists() {
            let inf = RealInference::new(gg_path).unwrap();
            assert_eq!(inf.config.n_layers, 22);
            assert_eq!(inf.config.hidden, 2048);
            assert_eq!(inf.tokenizer.vocab_size(), 32000);
        }
    }
    #[test]
    fn test_rope_identity_and_rotation() {
        // pos=0 → identidade
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        let f0: Vec<(f32, f32)> = (0..2)
            .map(|i| {
                let f = 0.0 * 10000f32.powf(-2.0 * i as f32 / 4.0);
                (f.cos(), f.sin())
            })
            .collect();
        apply_rope(&mut v, 1, 4, &f0);
        for (a, b) in v.iter().zip([1.0, 2.0, 3.0, 4.0]) {
            assert!((a - b).abs() < 1e-6);
        }
        // split-half: par (0,2) com ângulo π/2: (1,3) -> (-3,1); par (1,3) parado
        let f1 = vec![(0.0f32, 1.0f32), (1.0f32, 0.0f32)];
        let mut w = vec![1.0f32, 0.0, 3.0, 4.0];
        apply_rope(&mut w, 1, 4, &f1);
        assert!((w[0] - (-3.0)).abs() < 1e-6, "{}", w[0]);
        assert!((w[2] - 1.0).abs() < 1e-6, "{}", w[2]);
        assert!((w[1] - 0.0).abs() < 1e-6 && (w[3] - 4.0).abs() < 1e-6);
        let before: f32 = [1.0, 0.0, 3.0, 4.0].iter().map(|a| a * a).sum();
        let after: f32 = w.iter().map(|a| a * a).sum();
        assert!((before - after).abs() < 1e-5);
    }
    #[test]
    fn test_repeat_penalty_dethrones() {
        // Determinístico com top_k=1: sem histórico vence id 0; com id 0 no
        // histórico e penalty 2.0, id 1 assume o topo.
        let mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let _mem = crate::vm::MemBackend::Cpu(mem_mgr);
        let cfg = ModelConfig { hidden: 32, intermediate: 64, n_layers: 2, n_heads: 4, n_kv_heads: 4, vocab: 32000, arch: "llama".to_string(), context_length: 2048, rope_theta: 10000.0, norm_eps: 1e-5, d_inner: 64, d_state: 16, d_conv: 4, dt_rank: 2, dt_b_c_rms: false };
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: Vec::new(), kv: std::collections::HashMap::new(), data_offset: 0 };
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let mut inf = RealInference { config: cfg, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 2], kv_cache_v: vec![Vec::new(); 2], weight_cache: fxhash::FxHashMap::default(), gen_history: Vec::new(), layer_names: Vec::new(), mamba_names: Vec::new(), ssm_states: Vec::new(), profile: false,
            #[cfg(feature = "wgpu")]
            gpu: None };
        let logits = vec![2.0f32, 1.9];
        assert_eq!(inf.sample_with_params(&logits, 1.0, 1.0, 1, 2.0), 0);
        inf.push_gen(0);
        assert_eq!(inf.sample_with_params(&logits, 1.0, 1.0, 1, 2.0), 1);
    }
    #[test]
    fn test_tokenize_greedy_qwen_roundtrip() {
        let gg_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
        if !std::path::Path::new(gg_path).exists() {
            eprintln!("skip sem modelo");
            return;
        }
        let inf = RealInference::new(gg_path).unwrap();
        let ids = inf.tokenize("Hello, how are you?");
        println!("ids: {:?}", ids);
        // Qwen BPE: poucas peças (não explosão byte-a-byte), todas válidas
        assert!(ids.len() < 20 && ids.len() >= 3, "len {}", ids.len());
        assert!(ids.iter().all(|&i| (i as usize) < inf.tokenizer.vocab_size()));
        let back = inf.tokenizer.decode_stream(&ids);
        println!("roundtrip: {:?}", back);
        assert!(back.contains("Hello"), "sem Hello: {:?}", back);
        assert!(back.contains("you"), "sem you: {:?}", back);
    }
    #[test]
    fn test_chat_template_uses_file_specials() {
        // O template deve usar os especiais DESTE arquivo (fullwidth) e o
        // greedy deve resolvê-los em 1 token cada — <|im_start|> ASCII não existe aqui.
        let gg_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
        if !std::path::Path::new(gg_path).exists() {
            eprintln!("skip sem modelo");
            return;
        }
        let inf = RealInference::new(gg_path).unwrap();
        let prompt = inf.format_chat("Hi");
        println!("template: {:?}", prompt);
        let ids = inf.tokenize(&prompt);
        println!("template ids: {:?}", ids);
        assert_eq!(ids.first().cloned(), Some(151646), "BOS");
        assert!(ids.contains(&151644), "falta <｜User｜>: {:?}", ids);
        assert_eq!(ids.last().cloned(), Some(151645), "Assistant no fim: {:?}", ids);
        assert_eq!(inf.eos_id(), Some(151643));
    }
    #[test]
    fn test_chat_template_plain_without_markers() {
        // Vocab sem <|user|>/<|assistant|> (mock): template não deve emitir
        // marcadores que fragmentariam em peças (caso TinyLlama Q4_K_M).
        let cfg = ModelConfig { hidden: 32, intermediate: 64, n_layers: 2, n_heads: 4, n_kv_heads: 4, vocab: 32000, arch: "llama".to_string(), context_length: 2048, rope_theta: 10000.0, norm_eps: 1e-5, d_inner: 64, d_state: 16, d_conv: 4, dt_rank: 2, dt_b_c_rms: false };
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: Vec::new(), kv: std::collections::HashMap::new(), data_offset: 0 };
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let inf = RealInference { config: cfg, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 2], kv_cache_v: vec![Vec::new(); 2], weight_cache: fxhash::FxHashMap::default(), gen_history: Vec::new(), layer_names: Vec::new(), mamba_names: Vec::new(), ssm_states: Vec::new(), profile: false,
            #[cfg(feature = "wgpu")]
            gpu: None };
        let prompt = inf.format_chat("Hi");
        assert!(!prompt.contains("<|"), "marcador inexistente no template: {:?}", prompt);
        assert!(prompt.contains("Hi"));
    }
    #[test]
    fn test_repeat_kv_extend_gqa_order() {
        // GQA 4 heads <- 2 kv heads, dim 4: interleave [kv0×2, kv1×2]
        let small: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut out = Vec::new();
        RealInference::repeat_kv_extend(&mut out, &small, 4, 2, 4);
        assert_eq!(out, vec![0., 1., 2., 3., 0., 1., 2., 3., 4., 5., 6., 7., 4., 5., 6., 7.]);
    }
    #[test]
    fn test_embedding_row_matches_full_matrix() {
        // Paridade row-slice vs fatiar a matriz full (tinyllama Q4_K_M, hidden 2048)
        let gg_path = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
        if !std::path::Path::new(gg_path).exists() {
            eprintln!("skip: {} ausente", gg_path);
            return;
        }
        let inf = RealInference::new(gg_path).unwrap();
        let mut mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let mem = crate::vm::MemBackend::Cpu({
            mem_mgr.load_gguf_model(gg_path).unwrap();
            mem_mgr
        });
        let h = inf.config.hidden;
        assert_eq!(h, 2048);
        for tok in [0u32, 1, 150] {
            let row = inf.embedding_row(&mem, "token_embd.weight", tok as usize, h).unwrap();
            assert_eq!(row.len(), h);
            let full = inf.read_tensor_f32(&mem, "token_embd.weight").unwrap();
            let off = tok as usize * h;
            let max_diff = row.iter().zip(full[off..off + h].iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_diff == 0.0, "tok {} max_diff {}", tok, max_diff);
            assert!(row.iter().any(|&v| v != 0.0), "row {} zerada?", tok);
        }
    }
    #[test]
    fn test_matvec_weight_matches_cached_path() {
        // Paridade kernel int4 vs caminho cache f32+faer (deepseek q_proj blk 0)
        let gg_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
        if !std::path::Path::new(gg_path).exists() {
            eprintln!("skip: {} ausente", gg_path);
            return;
        }
        let mut inf = RealInference::new(gg_path).unwrap();
        let mut mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let mem = crate::vm::MemBackend::Cpu({
            mem_mgr.load_gguf_model(gg_path).unwrap();
            mem_mgr
        });
        let h = inf.config.hidden;
        let x: Vec<f32> = (0..h).map(|i| ((i as f32 * 0.017).sin() * 0.6)).collect();
        let y_kernel = inf.matvec_layer(&mem, 0, |l| l.q, &x, h, h);
        // Kernel direto não preenche o cache f32 (prova que o bypass foi exercido)
        assert_eq!(inf.weight_cache_len(), 0, "kernel deveria bypassar o cache");
        // Referência: cache f32 + faer (mesmo índice pré-resolvido)
        let idx = inf.layer_name(0, |l| l.q).expect("q resolvido");
        let w = inf.read_tensor_f32_cached_idx(&mem, idx).unwrap();
        let y_ref = crate::matvec::matvec(&x, &w[..], h, h);
        assert_eq!(y_kernel.len(), y_ref.len());
        let max_ref = y_ref.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_diff = y_kernel.iter().zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("q_proj blk0: max_ref {:.4} max_diff {:.6}", max_ref, max_diff);
        assert!(max_ref > 1e-6, "peso zerado?");
        assert!(max_diff / max_ref.max(1e-6) < 1e-3, "divergência relativa {}", max_diff / max_ref.max(1e-6));
    }
    #[test]
    fn test_kv_cache_22_layers_loop() {
        let mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let mem = crate::vm::MemBackend::Cpu(mem_mgr);
        // Usa config artificial 22 layers com hidden 32 para teste rápido (sem 1GB mmap)
        let cfg = ModelConfig { hidden: 32, intermediate: 64, n_layers: 22, n_heads: 4, n_kv_heads: 4, vocab: 32000, arch: "llama".to_string(), context_length: 2048, rope_theta: 10000.0, norm_eps: 1e-5, d_inner: 64, d_state: 16, d_conv: 4, dt_rank: 2, dt_b_c_rms: false };
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: Vec::new(), kv: std::collections::HashMap::new(), data_offset: 0 };
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let mut inf = RealInference { config: cfg, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 22], kv_cache_v: vec![Vec::new(); 22], weight_cache: fxhash::FxHashMap::default(), gen_history: Vec::new(), layer_names: Vec::new(), mamba_names: Vec::new(), ssm_states: Vec::new(), profile: false,
            #[cfg(feature = "wgpu")]
            gpu: None };
        assert_eq!(inf.config.n_layers, 22);
        // Primeiro token
        let logits1 = inf.forward_one(&mem, 1).unwrap();
        assert!(inf.kv_cache_k[0].len() == inf.config.hidden);
        assert_eq!(inf.kv_cache_k.len(), 22);
        // Segundo token seq_len 2
        let _logits2 = inf.forward_one(&mem, 2).unwrap();
        assert_eq!(inf.kv_cache_k[0].len(), inf.config.hidden * 2);
        // Truncate para 1 (rollback)
        inf.truncate_kv_cache(1);
        assert_eq!(inf.kv_cache_k[0].len(), inf.config.hidden);
        // Clear
        inf.clear_kv_cache();
        assert_eq!(inf.kv_cache_k[0].len(), 0);
        assert!(logits1.len() > 0);
    }

    /// GPU híbrida: offload sobe 84 pesos + head; gate via GPU == via CPU.
    /// Pula graciosamente sem modelo, sem feature ou sem adapter.
    #[test]
    #[cfg(feature = "wgpu")]
    fn test_gpu_offload_gate_parity() {        let gg_path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
        if !std::path::Path::new(gg_path).exists() {
            eprintln!("skip sem modelo");
            return;
        }
        // Sem adapter (CI sem GPU) → offload retorna 0, sem falhar
        let mut inf = RealInference::new(gg_path).unwrap();
        let mut mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let mem = crate::vm::MemBackend::Cpu({
            mem_mgr.load_gguf_model(gg_path).unwrap();
            mem_mgr
        });
        let n = inf.offload_gpu(&mem);
        if n == 0 {
            eprintln!("skip sem adapter GPU");
            return;
        }
        // 28 layers × (gate/up/down) + head
        assert_eq!(n, 28 * 3 + 1, "offload count {}", n);
        let h = inf.config.hidden;
        let inter = inf.config.intermediate;
        let x: Vec<f32> = (0..h).map(|i| ((i as f32 * 0.021).sin() * 0.4)).collect();
        let y_gpu = inf.gpu_layer(0, &|l: &LayerNames| l.gate, &x, h, inter).expect("gate na GPU");
        let y_cpu = inf.matvec_layer(&mem, 0, |l| l.gate, &x, h, inter);
        assert_eq!(y_gpu.len(), y_cpu.len());
        let max_ref = y_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_diff = y_gpu.iter().zip(y_cpu.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("gpu gate blk0: max_ref {:.4} max_diff {:.6}", max_ref, max_diff);
        assert!(max_ref > 1e-6);
        assert!(max_diff / max_ref < 1e-2, "divergência {}", max_diff / max_ref);
        // Q6K real (down de um blk Q6_K): exercita scales int8 negativos
        let mut down6: Option<usize> = None;
        for blk in 0..inf.config.n_layers {
            if let Some(idx) = inf.layer_name(blk, |l| l.down) {
                if inf.gguf.tensors[idx].dtype == 14 {
                    down6 = Some(blk);
                    break;
                }
            }
        }
        if let Some(blk) = down6 {
            let h = inf.config.hidden;
            let inter = inf.config.intermediate;
            let xd: Vec<f32> = (0..inter).map(|i| ((i as f32 * 0.043).sin() * 0.3)).collect();
            let yg = inf.gpu_layer(blk, &|l: &LayerNames| l.down, &xd, inter, h).expect("down Q6K na GPU");
            let yc = inf.matvec_layer(&mem, blk, |l| l.down, &xd, inter, h);
            let mr = yc.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let md = yg.iter().zip(yc.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("gpu down-Q6K blk{}: max_ref {:.4} max_diff {:.6}", blk, mr, md);
            assert!(mr > 1e-6);
            assert!(md / mr < 1e-2, "divergência Q6K {}", md / mr);
        } else {
            eprintln!("(sem down Q6K neste modelo — skip)");
        }
    }

    // --- Spike Mamba (130M-scale sintético + real se presente) ---

    fn mock_mamba_cfg() -> ModelConfig {
        ModelConfig { hidden: 16, intermediate: 0, n_layers: 2, n_heads: 0, n_kv_heads: 0, vocab: 32000, arch: "mamba".to_string(), context_length: 2048, rope_theta: 10000.0, norm_eps: 1e-5, d_inner: 32, d_state: 8, d_conv: 4, dt_rank: 2, dt_b_c_rms: false }
    }

    #[test]
    fn test_mamba_config_parses_ssm_kv() {
        let mut kv = std::collections::HashMap::new();
        kv.insert("general.architecture".to_string(), "mamba".to_string());
        kv.insert("mamba.embedding_length".to_string(), "768".to_string());
        kv.insert("mamba.block_count".to_string(), "24".to_string());
        kv.insert("mamba.ssm.inner_size".to_string(), "1536".to_string());
        kv.insert("mamba.ssm.state_size".to_string(), "16".to_string());
        kv.insert("mamba.ssm.conv_kernel".to_string(), "4".to_string());
        kv.insert("mamba.ssm.time_step_rank".to_string(), "48".to_string());
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 7, tensors: Vec::new(), kv, data_offset: 0 };
        let cfg = ModelConfig::from_gguf(&gg);
        assert!(cfg.is_mamba());
        assert_eq!(cfg.hidden, 768);
        assert_eq!(cfg.n_layers, 24);
        assert_eq!(cfg.d_inner, 1536);
        assert_eq!(cfg.d_state, 16);
        assert_eq!(cfg.d_conv, 4);
        assert_eq!(cfg.dt_rank, 48);
        // Falcon liga dt_b_c_rms via arch
        let mut kv2 = std::collections::HashMap::new();
        kv2.insert("general.architecture".to_string(), "falcon_mamba".to_string());
        let gg2 = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 1, tensors: Vec::new(), kv: kv2, data_offset: 0 };
        let cfg2 = ModelConfig::from_gguf(&gg2);
        assert!(cfg2.is_mamba());
        assert!(cfg2.dt_b_c_rms, "falcon_mamba implica RMS em dt/B/C");
        assert_eq!(cfg2.d_inner, cfg2.hidden * 2);
    }

    #[test]
    fn test_mamba_names_resolve_blk_ssm() {
        use crate::gguf::GgufTensorInfo;
        let mk = |name: &str| GgufTensorInfo { name: name.to_string(), dims: vec![8, 8], shape: vec![8, 8], dtype: 0, offset: 0, n_elements: 64 };
        let gg = crate::gguf::GgufFile {
            version: 3, n_tensors: 10, n_kv: 1,
            tensors: vec![
                mk("blk.0.attn_norm.weight"), mk("blk.0.ssm_in.weight"),
                mk("blk.0.ssm_conv1d.weight"), mk("blk.0.ssm_conv1d.bias"),
                mk("blk.0.ssm_x.weight"), mk("blk.0.ssm_dt.weight"),
                mk("blk.0.ssm_dt.bias"), mk("blk.0.ssm_a"),
                mk("blk.0.ssm_d"), mk("blk.0.ssm_out.weight"),
            ],
            kv: std::collections::HashMap::new(), data_offset: 0,
        };
        let cfg = mock_mamba_cfg();
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let mut inf = RealInference { config: ModelConfig { n_layers: 1, ..cfg }, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 1], kv_cache_v: vec![Vec::new(); 1], weight_cache: fxhash::FxHashMap::default(), gen_history: Vec::new(), layer_names: Vec::new(), mamba_names: Vec::new(), ssm_states: Vec::new(), profile: false,
            #[cfg(feature = "wgpu")]
            gpu: None };
        inf.resolve_mamba_names();
        assert_eq!(inf.mamba_names.len(), 1);
        let mn = &inf.mamba_names[0];
        assert!(mn.norm.is_some() && mn.ssm_in.is_some() && mn.conv_w.is_some());
        assert!(mn.conv_b.is_some() && mn.ssm_x.is_some() && mn.ssm_dt.is_some());
        assert!(mn.ssm_dt_bias.is_some() && mn.ssm_a.is_some() && mn.ssm_d.is_some() && mn.ssm_out.is_some());
    }

    #[test]
    fn test_forward_one_mamba_dummy_state_constant() {
        let mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let mem = crate::vm::MemBackend::Cpu(mem_mgr);
        let cfg = mock_mamba_cfg();
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: Vec::new(), kv: std::collections::HashMap::new(), data_offset: 0 };
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let mut inf = RealInference { config: cfg, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 2], kv_cache_v: vec![Vec::new(); 2], weight_cache: fxhash::FxHashMap::default(), gen_history: Vec::new(), layer_names: Vec::new(), mamba_names: Vec::new(), ssm_states: Vec::new(), profile: false,
            #[cfg(feature = "wgpu")]
            gpu: None };
        let l1 = inf.forward_one_mamba(&mem, 1).unwrap();
        assert_eq!(l1.len(), inf.tokenizer.vocab_size());
        assert!(l1.iter().all(|v| v.is_finite()));
        // estado recorrente alocado e constante (não cresce como KV)
        assert_eq!(inf.ssm_states.len(), 2);
        let (c0, s0) = (inf.ssm_states[0].conv.len(), inf.ssm_states[0].ssm.len());
        assert_eq!((c0, s0), (32 * 4, 32 * 8));
        assert!(inf.ssm_states[0].ssm.iter().any(|&v| v != 0.0), "scan deveria sujar o estado");
        let _ = inf.forward_one_mamba(&mem, 2).unwrap();
        assert_eq!(inf.ssm_states[0].conv.len(), c0);
        assert_eq!(inf.ssm_states[0].ssm.len(), s0);
        inf.clear_ssm_states();
        assert!(inf.ssm_states.iter().all(|s| s.ssm.iter().all(|&v| v == 0.0)));
    }

    #[test]
    fn test_mamba_real_file_if_present() {
        // mamba-130M / Falcon-Mamba Q4_K_M se baixado em ./models (skip gracioso).
        let cands = [
            "./models/mamba-130m-hf.Q4_K_M.gguf",
            "./models/mamba-130m-hf-Q4_K_M.gguf",
            "./models/mamba-130m-Q4_K_M.gguf",
            "./models/falcon-mamba-7b-Q4_K_M.gguf",
            "./models/Falcon-Mamba-7B-Q4_K_M.gguf",
        ];
        let path = cands.iter().find(|p| std::path::Path::new(p).exists());
        let Some(gg_path) = path else {
            eprintln!("skip mamba real ausente (baixe um Q4_K_M em ./models)");
            return;
        };
        let inf = RealInference::new(gg_path).unwrap();
        assert!(inf.config.is_mamba(), "arch={}", inf.config.arch);
        assert!(inf.config.d_inner == inf.config.hidden * 2, "d_inner={} hidden={}", inf.config.d_inner, inf.config.hidden);
        assert!(!inf.mamba_names.is_empty());
        let resolved = inf.mamba_names.iter().filter(|m| m.ssm_in.is_some()).count();
        assert!(resolved > 0, "nenhum ssm_in resolvido em {}", gg_path);
        eprintln!("[mamba-real] {} layers={} hidden={} d_inner={} d_state={} d_conv={} dt_rank={} rms={} ssm_in={}/{}",
            gg_path, inf.config.n_layers, inf.config.hidden, inf.config.d_inner,
            inf.config.d_state, inf.config.d_conv, inf.config.dt_rank,
            inf.config.dt_b_c_rms, resolved, inf.config.n_layers);
    }
}
