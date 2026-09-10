//! moshi.rs — Config + nomes de tensores PersonaPlex/Moshi (F1 do PLANO_MOSHI_NATIVO)
//!
//! PersonaPlex-7B é base Kyutai Moshi (`moshiko` weights):
//! Temporal 32 layers dim=4096 32 heads SwiGLU (ff=11264) + Depformer 6 layers
//! dim=1024 16 heads + Mimi 16 codebooks 12.5Hz 24kHz, 17 streams
//! (1 texto + 8 user + 8 agent). Ref: `docs/ESPEC.md` §14.
//!
//! Escopo F1+F3a: config/loader/nomes + forward temporal/depformer (stub com
//! pesos injetáveis; pesos GGUF reais entram em F3b via `matvec_quant`).
//! Codec em `crate::mimi` (F2).

use crate::gguf::GgufFile;

// ---------------------------------------------------------------------------
// Constantes de referência (PersonaPlex-7B-v1 / Moshiko)
// ---------------------------------------------------------------------------

/// Streams simultâneos: 1 texto + 8 user + 8 agent.
pub const MOSHI_N_STREAMS: usize = 17;
/// Codebooks de áudio (Mimi). Moshi original usa 8, PersonaPlex usa 16.
pub const MOSHI_N_CODEBOOKS: usize = 16;
/// Codebooks do Moshi original (8) — mantido para detectar modelo antigo.
pub const MOSHI_N_CODEBOOKS_LEGACY: usize = 8;
/// Taxa de frames do Mimi.
pub const MOSHI_FRAME_HZ: f32 = 12.5;
/// Sample rate do Mimi.
pub const MOSHI_SAMPLE_RATE: u32 = 24_000;
/// Delay acústico teórico: 80ms frame + 80ms delay = 160ms.
pub const MOSHI_ACOUSTIC_DELAY_MS: f32 = 160.0;

pub const MOSHI_HIDDEN: usize = 4096;
pub const MOSHI_LAYERS: usize = 32;
pub const MOSHI_HEADS: usize = 32;
pub const MOSHI_FF: usize = 11_264;
pub const MOSHI_VOCAB: usize = 32_000;
pub const MOSHI_DEP_LAYERS: usize = 6;
pub const MOSHI_DEP_DIM: usize = 1024;
pub const MOSHI_DEP_HEADS: usize = 16;
pub const MOSHI_DEP_FF: usize = 2816;
/// `dep_q` do Depformer (pesos por codebook).
pub const MOSHI_DEP_Q: usize = 16;
/// Janela do KV do depformer (`depformer_context: 8` no config real).
pub const DEPFORMER_CONTEXT: usize = 8;
pub const MOSHI_ROPE_THETA: f32 = 10_000.0;
pub const MOSHI_NORM_EPS: f32 = 1e-5;

/// Samples por frame Mimi: 24000 / 12.5 = 1920.
#[inline]
pub fn samples_per_frame(sample_rate: u32, frame_hz: f32) -> usize {
    ((sample_rate as f32) / frame_hz).round() as usize
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MoshiConfig {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub intermediate: usize,
    pub dep_layers: usize,
    pub dep_dim: usize,
    pub dep_heads: usize,
    pub dep_ff: usize,
    pub codebooks: usize,
    pub text_vocab: usize,
    pub frame_hz: f32,
    pub sample_rate: u32,
    pub context_length: usize,
    pub rope_theta: f32,
    pub norm_eps: f32,
    pub arch: String,
}

impl Default for MoshiConfig {
    fn default() -> Self {
        Self {
            hidden: MOSHI_HIDDEN,
            n_layers: MOSHI_LAYERS,
            n_heads: MOSHI_HEADS,
            intermediate: MOSHI_FF,
            dep_layers: MOSHI_DEP_LAYERS,
            dep_dim: MOSHI_DEP_DIM,
            dep_heads: MOSHI_DEP_HEADS,
            dep_ff: MOSHI_DEP_FF,
            codebooks: MOSHI_N_CODEBOOKS,
            text_vocab: MOSHI_VOCAB,
            frame_hz: MOSHI_FRAME_HZ,
            sample_rate: MOSHI_SAMPLE_RATE,
            context_length: 2048,
            rope_theta: MOSHI_ROPE_THETA,
            norm_eps: MOSHI_NORM_EPS,
            arch: "moshi".to_string(),
        }
    }
}

/// `true` se `arch` é Moshi/PersonaPlex/Mimi (case-insensitive, contém substring).
pub fn is_moshi_arch(arch: &str) -> bool {
    let a = arch.to_lowercase();
    a.contains("moshi") || a.contains("mimi") || a.contains("personaplex") || a.contains("moshiko") || a.contains("moshika")
}

impl MoshiConfig {
    pub fn personaplex() -> Self {
        Self::default()
    }

    pub fn head_dim(&self) -> usize {
        if self.n_heads == 0 { self.hidden } else { self.hidden / self.n_heads }
    }

    pub fn samples_per_frame(&self) -> usize {
        samples_per_frame(self.sample_rate, self.frame_hz)
    }

    /// Constrói a partir do KV do GGUF. Usa chaves `moshi.*` + fallbacks
    /// `llama.*`/`general.*` (conversões `moshi.cpp` mantêm `blk.*`).
    /// Sem chave relevante, volta aos defaults PersonaPlex.
    pub fn from_gguf(gg: &GgufFile) -> Self {
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
        let arch = gg.kv.get("general.architecture").cloned().unwrap_or_else(|| "moshi".to_string());
        let hidden = get_parse(&["moshi.embedding_length", "llama.embedding_length", "general.embedding_length"], MOSHI_HIDDEN);
        let n_layers = get_parse(&["moshi.block_count", "llama.block_count", "general.block_count"], MOSHI_LAYERS);
        let n_heads = get_parse(&["moshi.attention.head_count", "llama.attention.head_count"], MOSHI_HEADS);
        let intermediate = get_parse(&["moshi.feed_forward_length", "llama.feed_forward_length"], MOSHI_FF);
        let text_vocab = get_parse(&["moshi.vocab_size", "llama.vocab_size", "general.vocab_size"], MOSHI_VOCAB);
        let context_length = get_parse(&["moshi.context_length", "llama.context_length", "general.context_length"], 2048);
        let rope_theta = get_f32(&["moshi.rope.freq_base", "llama.rope.freq_base"], MOSHI_ROPE_THETA);
        let norm_eps = get_f32(&["moshi.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon"], MOSHI_NORM_EPS);
        let dep_layers = get_parse(&["moshi.depformer.block_count", "depformer.block_count"], MOSHI_DEP_LAYERS);
        let dep_dim = get_parse(&["moshi.depformer.embedding_length", "depformer.embedding_length"], MOSHI_DEP_DIM);
        let codebooks = get_parse(&["moshi.audio.codebooks", "mimi.codebooks"], MOSHI_N_CODEBOOKS);
        Self { hidden, n_layers, n_heads, intermediate, dep_layers, dep_dim, dep_heads: MOSHI_DEP_HEADS, dep_ff: MOSHI_DEP_FF, codebooks, text_vocab, frame_hz: MOSHI_FRAME_HZ, sample_rate: MOSHI_SAMPLE_RATE, context_length, rope_theta, norm_eps, arch }
    }

    /// Parser mínimo de `config.json` (sts-web / HF) sem nova dependência.
    /// Extrai apenas inteiros das chaves conhecidas; resto = default.
    pub fn from_config_json(json: &str) -> Self {
        let mut cfg = Self::default();
        let get = |keys: &[&str]| -> Option<usize> {
            for k in keys {
                if let Some(v) = extract_json_usize(json, k) { return Some(v); }
            }
            None
        };
        if let Some(v) = get(&["dim", "hidden_size", "d_model"]) { cfg.hidden = v; }
        if let Some(v) = get(&["num_hidden_layers", "n_layers", "num_layers"]) { cfg.n_layers = v; }
        if let Some(v) = get(&["num_attention_heads", "n_heads", "num_heads"]) { cfg.n_heads = v; }
        if let Some(v) = get(&["hidden_dim", "intermediate_size", "ffn_dim"]) { cfg.intermediate = v; }
        if let Some(v) = get(&["depformer_num_layers", "num_depformer_layers"]) { cfg.dep_layers = v; }
        if let Some(v) = get(&["depformer_dim"]) { cfg.dep_dim = v; }
        if let Some(v) = get(&["num_codebooks", "n_codebooks", "codebooks", "n_q", "dep_q"]) { cfg.codebooks = v; }
        if let Some(v) = get(&["vocab_size", "text_vocab_size", "text_card"]) { cfg.text_vocab = v; }
        cfg
    }
}

/// Extrai `"key": 1234` (com ou sem aspas no valor) de um JSON simples.
/// Retorna `None` se não achar inteiro após a chave.
fn extract_json_usize(json: &str, key: &str) -> Option<usize> {
    let quoted = format!("\"{}\"", key);
    let pos = json.find(&quoted)?;
    let after = &json[pos + quoted.len()..];
    let colon = after.find(':')?;
    let mut num = String::new();
    for c in after[colon + 1..].chars() {
        if c.is_ascii_digit() {
            num.push(c);
        } else if num.is_empty() {
            if c.is_whitespace() || c == '"' { continue; } else { break; }
        } else {
            break;
        }
    }
    if num.is_empty() { None } else { num.parse::<usize>().ok() }
}

// ---------------------------------------------------------------------------
// Nomes de tensores
// ---------------------------------------------------------------------------

/// Nomes candidatos do Temporal por (camada, kind).
/// Cobre GGUF `blk.*` (moshi.cpp) + safetensors `temporal.*` (HF original).
pub fn temporal_tensor_names(layer: usize, kind: &str) -> Vec<String> {
    let t = format!("temporal.layers.{}", layer);
    let b = format!("blk.{}", layer);
    match kind {
        "q" => vec![format!("{}.attn_q.weight", b), format!("{}.self_attn.q_proj.weight", t)],
        "k" => vec![format!("{}.attn_k.weight", b), format!("{}.self_attn.k_proj.weight", t)],
        "v" => vec![format!("{}.attn_v.weight", b), format!("{}.self_attn.v_proj.weight", t)],
        "o" => vec![format!("{}.attn_output.weight", b), format!("{}.self_attn.o_proj.weight", t)],
        "gate" => vec![format!("{}.ffn_gate.weight", b), format!("{}.mlp.gate_proj.weight", t)],
        "up" => vec![format!("{}.ffn_up.weight", b), format!("{}.mlp.up_proj.weight", t)],
        "down" => vec![format!("{}.ffn_down.weight", b), format!("{}.mlp.down_proj.weight", t)],
        "attn_norm" => vec![format!("{}.attn_norm.weight", b), format!("{}.input_layernorm.weight", t)],
        "ffn_norm" => vec![format!("{}.ffn_norm.weight", b), format!("{}.post_attention_layernorm.weight", t)],
        _ => vec![],
    }
}

/// Nomes candidatos do Depformer por (camada, kind).
pub fn depformer_tensor_names(layer: usize, kind: &str) -> Vec<String> {
    let d = format!("depformer.layers.{}", layer);
    let b = format!("blk_dep.{}", layer);
    match kind {
        "q" | "in_proj" => vec![format!("{}.in_proj.weight", b), format!("{}.self_attn.in_proj.weight", d)],
        "o" => vec![format!("{}.out_proj.weight", b), format!("{}.self_attn.out_proj.weight", d)],
        "gate" => vec![format!("{}.gate_proj.weight", b), format!("{}.mlp.gate_proj.weight", d)],
        "up" => vec![format!("{}.up_proj.weight", b), format!("{}.mlp.up_proj.weight", d)],
        "down" => vec![format!("{}.down_proj.weight", b), format!("{}.mlp.down_proj.weight", d)],
        "norm" => vec![format!("{}.norm.weight", b), format!("{}.input_layernorm.weight", d)],
        _ => vec![],
    }
}

/// Nomes candidatos do Mimi (codec) e embeddings.
pub fn mimi_tensor_names(kind: &str) -> Vec<String> {
    match kind {
        "token_embd" => vec!["token_embd.weight".to_string(), "embeddings.text.weight".to_string(), "model.embed_tokens.weight".to_string()],
        "audio_embd" => vec!["audio_embd.weight".to_string(), "embeddings.audio.weight".to_string()],
        "output" => vec!["output.weight".to_string(), "lm_head.weight".to_string()],
        "mimi_encoder" => vec!["mimi.encoder.weight".to_string(), "tokenizer.encoder.weight".to_string()],
        "mimi_decoder" => vec!["mimi.decoder.weight".to_string(), "tokenizer.decoder.weight".to_string()],
        _ => vec![],
    }
}

/// Índices pré-resolvidos do Temporal por camada (`None` = ausente/dummy).
#[derive(Debug, Clone, Default)]
pub struct MoshiLayerNames {
    pub q: Option<usize>,
    pub k: Option<usize>,
    pub v: Option<usize>,
    pub o: Option<usize>,
    pub gate: Option<usize>,
    pub up: Option<usize>,
    pub down: Option<usize>,
    pub attn_norm: Option<usize>,
    pub ffn_norm: Option<usize>,
}

/// Índices pré-resolvidos do Depformer por camada.
#[derive(Debug, Clone, Default)]
pub struct DepformerLayerNames {
    pub in_proj: Option<usize>,
    pub o: Option<usize>,
    pub gate: Option<usize>,
    pub up: Option<usize>,
    pub down: Option<usize>,
    pub norm: Option<usize>,
}

fn resolve_in(gg: &GgufFile, candidates: &[String]) -> Option<usize> {
    for name in candidates {
        if let Some(pos) = gg.tensors.iter().position(|t| &t.name == name) {
            return Some(pos);
        }
    }
    None
}

/// Resolve `MoshiLayerNames` para todas as camadas do `cfg`.
pub fn resolve_temporal_names(gg: &GgufFile, cfg: &MoshiConfig) -> Vec<MoshiLayerNames> {
    (0..cfg.n_layers).map(|l| MoshiLayerNames {
        q: resolve_in(gg, &temporal_tensor_names(l, "q")),
        k: resolve_in(gg, &temporal_tensor_names(l, "k")),
        v: resolve_in(gg, &temporal_tensor_names(l, "v")),
        o: resolve_in(gg, &temporal_tensor_names(l, "o")),
        gate: resolve_in(gg, &temporal_tensor_names(l, "gate")),
        up: resolve_in(gg, &temporal_tensor_names(l, "up")),
        down: resolve_in(gg, &temporal_tensor_names(l, "down")),
        attn_norm: resolve_in(gg, &temporal_tensor_names(l, "attn_norm")),
        ffn_norm: resolve_in(gg, &temporal_tensor_names(l, "ffn_norm")),
    }).collect()
}

/// Resolve `DepformerLayerNames` para todas as camadas depformer do `cfg`.
pub fn resolve_depformer_names(gg: &GgufFile, cfg: &MoshiConfig) -> Vec<DepformerLayerNames> {
    (0..cfg.dep_layers).map(|l| DepformerLayerNames {
        in_proj: resolve_in(gg, &depformer_tensor_names(l, "in_proj")),
        o: resolve_in(gg, &depformer_tensor_names(l, "o")),
        gate: resolve_in(gg, &depformer_tensor_names(l, "gate")),
        up: resolve_in(gg, &depformer_tensor_names(l, "up")),
        down: resolve_in(gg, &depformer_tensor_names(l, "down")),
        norm: resolve_in(gg, &depformer_tensor_names(l, "norm")),
    }).collect()
}

/// Conta quantos tensores do GGUF batem com algum nome Moshi conhecido.
/// Útil para detectar se um GGUF é Moshi/PersonaPlex ou LLM texto.
pub fn count_moshi_tensors(gg: &GgufFile, cfg: &MoshiConfig) -> usize {
    let names = resolve_temporal_names(gg, cfg);
    names.iter().map(|l| [l.q, l.k, l.v, l.o, l.gate, l.up, l.down, l.attn_norm, l.ffn_norm].into_iter().flatten().count()).sum()
}

/// Mapa textual para `docs/ESPEC.md` §14 (gerado, não lido em runtime).
pub fn describe_map(cfg: &MoshiConfig) -> String {
    let mut s = String::new();
    s.push_str(&format!("# MOSHI_MAP (gerado): arch={} hidden={} layers={} heads={} ff={} dep={}x{} codebooks={}\n\n", cfg.arch, cfg.hidden, cfg.n_layers, cfg.n_heads, cfg.intermediate, cfg.dep_layers, cfg.dep_dim, cfg.codebooks));
    s.push_str("## Temporal layer 0 (candidatos por kind)\n");
    for kind in ["q", "k", "v", "o", "gate", "up", "down", "attn_norm", "ffn_norm"] {
        s.push_str(&format!("- {}: {}\n", kind, temporal_tensor_names(0, kind).join(" | ")));
    }
    s.push_str("\n## Depformer layer 0\n");
    for kind in ["in_proj", "o", "gate", "up", "down", "norm"] {
        s.push_str(&format!("- {}: {}\n", kind, depformer_tensor_names(0, kind).join(" | ")));
    }
    s.push_str("\n## Mimi/embeddings\n");
    for kind in ["token_embd", "audio_embd", "output", "mimi_encoder", "mimi_decoder"] {
        s.push_str(&format!("- {}: {}\n", kind, mimi_tensor_names(kind).join(" | ")));
    }
    s
}

// ---------------------------------------------------------------------------
// Forward Temporal + Depformer (F3a: stub estrutural com pesos injetáveis)
// ---------------------------------------------------------------------------
// Espelha `RealInference::forward_one` (`src/inference.rs:953`): RMSNorm ->
// QKV matvec -> RoPE -> KV append -> atenção causal per-head -> O proj ->
// residual -> SwiGLU FFN -> residual. Diferenças Moshi (simplificadas, F3b
// refina com pesos GGUF + delay pattern real):
// - Entrada é 1 frame mixado (texto+áudio) em vez de token embedding.
// - Sem GQA (MHA 32 heads no real); fallback 1 head se `hidden % heads != 0`.
// - Depformer de 6x1024 com KV próprio + 16 códigos de saída (dep_q=16).
// - Delay acústico simplificado: entrada efetiva = média do frame atual com
//   o anterior (real: streams com shift temporal + feedback de códigos).

use crate::inference::{apply_rope, rms_norm_into};

/// Qual stack o peso pertence (para `MoshiWeights`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoshiStack {
    Temporal,
    Depformer,
}

/// Fonte de pesos do forward. `DummyMoshiWeights` replica o fallback dummy de
/// `RealInference::matvec_weight_pre` (`src/inference.rs:584`, matriz 0.5);
/// F3b adiciona `GgufMoshiWeights` sobre `matvec_quant` + nomes resolvidos.
pub trait MoshiWeights {
    fn matvec(&mut self, stack: MoshiStack, layer: usize, kind: &str, x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32>;
    fn norm_gamma(&mut self, stack: MoshiStack, layer: usize, kind: &str, dim: usize) -> Vec<f32>;
    /// Início de frame: limpa memo de fused (pesos reais). Default no-op.
    fn begin_frame(&mut self) {}
}

/// Pesos dummy: `y[j] = 0.5*sum(x)`, gamma 1.0.
#[derive(Debug, Clone, Default)]
pub struct DummyMoshiWeights;

impl MoshiWeights for DummyMoshiWeights {
    fn matvec(&mut self, _stack: MoshiStack, _layer: usize, _kind: &str, x: &[f32], _in_dim: usize, out_dim: usize) -> Vec<f32> {
        vec![0.5 * x.iter().sum::<f32>(); out_dim]
    }
    fn norm_gamma(&mut self, _stack: MoshiStack, _layer: usize, _kind: &str, dim: usize) -> Vec<f32> {
        vec![1.0; dim]
    }
}

/// Saída de 1 frame: hidden temporal + hidden depformer + 16 códigos previstos.
#[derive(Debug, Clone)]
pub struct MoshiFrameOut {
    pub hidden: Vec<f32>,
    pub dep_hidden: Vec<f32>,
    pub codes: [u16; MOSHI_N_CODEBOOKS],
}

/// Tempos do último `forward_frame` (ms).
#[derive(Debug, Clone, Default)]
pub struct MoshiProfileMs {
    pub temporal_ms: f64,
    pub depformer_ms: f64,
    pub total_ms: f64,
}

/// Inferência Moshi/PersonaPlex por frame.
pub struct MoshiInference<W: MoshiWeights = DummyMoshiWeights> {
    pub cfg: MoshiConfig,
    weights: W,
    kv_k: Vec<Vec<f32>>,
    kv_v: Vec<Vec<f32>>,
    dep_kv_k: Vec<Vec<f32>>,
    dep_kv_v: Vec<Vec<f32>>,
    delay_prev: Option<Vec<f32>>,
    frames: usize,
    pub profile: bool,
    pub last_ms: MoshiProfileMs,
}

impl<W: MoshiWeights> MoshiInference<W> {
    pub fn new(cfg: MoshiConfig, weights: W) -> Self {
        let kv_k = vec![Vec::new(); cfg.n_layers];
        let kv_v = vec![Vec::new(); cfg.n_layers];
        let dep_kv_k = vec![Vec::new(); cfg.dep_layers];
        let dep_kv_v = vec![Vec::new(); cfg.dep_layers];
        Self { cfg, weights, kv_k, kv_v, dep_kv_k, dep_kv_v, delay_prev: None, frames: 0, profile: false, last_ms: MoshiProfileMs::default() }
    }

    /// Frames processados (= seq_len do KV).
    pub fn seq_len_frames(&self) -> usize {
        self.frames
    }

    /// Limpa KV + delay (prompt novo).
    pub fn clear(&mut self) {
        for k in &mut self.kv_k { k.clear(); }
        for v in &mut self.kv_v { v.clear(); }
        for k in &mut self.dep_kv_k { k.clear(); }
        for v in &mut self.dep_kv_v { v.clear(); }
        self.delay_prev = None;
        self.frames = 0;
    }

    /// Rollback para `n` frames (espelha `truncate_kv_cache`, `src/inference.rs:410`).
    /// Zera o delay (simplificação F3a: continuação truncada difere do run
    /// fresco por 1 frame; `truncate(0)` é idêntico ao fresco).
    pub fn truncate_frames(&mut self, n: usize) {
        let h = self.cfg.hidden;
        for k in &mut self.kv_k { k.truncate(n * h); }
        for v in &mut self.kv_v { v.truncate(n * h); }
        let d = self.cfg.dep_dim;
        for k in &mut self.dep_kv_k { k.truncate(n * d); }
        for v in &mut self.dep_kv_v { v.truncate(n * d); }
        self.delay_prev = None;
        self.frames = n;
    }

    /// 1 frame mixado (`hidden` floats) -> hidden + 16 códigos.
    pub fn forward_frame(&mut self, frame_mix: &[f32]) -> MoshiFrameOut {
        let t_total = std::time::Instant::now();
        let h = self.cfg.hidden;
        assert_eq!(frame_mix.len(), h, "frame_mix deve ter hidden={} floats", h);
        self.weights.begin_frame();

        // Delay acústico simplificado: média com o frame anterior.
        let input: Vec<f32> = delay_mix(self.delay_prev.as_deref(), frame_mix);

        let t0 = std::time::Instant::now();
        let hidden = self.forward_temporal(&input);
        let temporal_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = std::time::Instant::now();
        let (dep_hidden, codes) = self.forward_depformer(&hidden);
        let depformer_ms = t1.elapsed().as_secs_f64() * 1000.0;

        self.delay_prev = Some(frame_mix.to_vec());
        self.frames += 1;
        if self.profile {
            self.last_ms = MoshiProfileMs { temporal_ms, depformer_ms, total_ms: t_total.elapsed().as_secs_f64() * 1000.0 };
        }
        MoshiFrameOut { hidden, dep_hidden, codes }
    }

    /// Stack temporal: `n_layers` blocos transformer com KV (seq de frames).
    fn forward_temporal(&mut self, input: &[f32]) -> Vec<f32> {
        let h = self.cfg.hidden;
        let heads_cfg = self.cfg.n_heads.max(1);
        let (n_heads, head_dim) = if h % heads_cfg == 0 { (heads_cfg, h / heads_cfg) } else { (1, h) };
        let pos = self.frames;
        let rope_hd = head_dim;
        let freqs: Vec<(f32, f32)> = (0..rope_hd.max(2) / 2)
            .map(|i| {
                let f = pos as f32 * self.cfg.rope_theta.powf(-2.0 * i as f32 / rope_hd.max(1) as f32);
                (f.cos(), f.sin())
            })
            .collect();

        let mut hidden_cur = input.to_vec();
        let mut normed = vec![0.0f32; h];
        let inter = self.cfg.intermediate;
        let mut ffn_h = vec![0.0f32; inter];
        for l in 0..self.cfg.n_layers {
            let gamma = self.weights.norm_gamma(MoshiStack::Temporal, l, "attn_norm", h);
            rms_norm_into(&hidden_cur, &gamma, &mut normed);
            let q = self.weights.matvec(MoshiStack::Temporal, l, "q", &normed, h, h);
            let k = self.weights.matvec(MoshiStack::Temporal, l, "k", &normed, h, h);
            let v = self.weights.matvec(MoshiStack::Temporal, l, "v", &normed, h, h);
            let mut q = q;
            let mut k = k;
            apply_rope(&mut q, n_heads, head_dim, &freqs);
            apply_rope(&mut k, n_heads, head_dim, &freqs);
            self.kv_k[l].extend_from_slice(&k);
            self.kv_v[l].extend_from_slice(&v);
            let seq = self.kv_k[l].len() / h;
            let mut agg = vec![0.0f32; h];
            per_head_attn(&q, &self.kv_k[l], &self.kv_v[l], n_heads, head_dim, seq, &mut agg);
            let o = self.weights.matvec(MoshiStack::Temporal, l, "o", &agg, h, h);
            let mut hidden2 = vec![0.0f32; h];
            for i in 0..h { hidden2[i] = hidden_cur[i] + o[i]; }

            let gamma_f = self.weights.norm_gamma(MoshiStack::Temporal, l, "ffn_norm", h);
            rms_norm_into(&hidden2, &gamma_f, &mut normed);
            let gate = self.weights.matvec(MoshiStack::Temporal, l, "gate", &normed, h, inter);
            let up = self.weights.matvec(MoshiStack::Temporal, l, "up", &normed, h, inter);
            for i in 0..inter {
                let g = gate[i];
                ffn_h[i] = g / (1.0 + (-g).exp()) * up[i];
            }
            let down = self.weights.matvec(MoshiStack::Temporal, l, "down", &ffn_h, inter, h);
            let mut next = vec![0.0f32; h];
            for i in 0..h { next[i] = hidden2[i] + down[i]; }
            hidden_cur = next;
        }
        hidden_cur
    }

    /// Depformer: projeta hidden->dep_dim, `dep_layers` blocos com KV próprio
    /// (janela `DEPFORMER_CONTEXT`), 16 códigos por média de chunks quantizada.
    /// Sem RoPE: `depformer_pos_emb none` no config real.
    fn forward_depformer(&mut self, hidden: &[f32]) -> (Vec<f32>, [u16; MOSHI_N_CODEBOOKS]) {
        let h = self.cfg.hidden;
        let d = self.cfg.dep_dim;
        let heads_cfg = self.cfg.dep_heads.max(1);
        let (n_heads, head_dim) = if d % heads_cfg == 0 { (heads_cfg, d / heads_cfg) } else { (1, d) };

        let mut dep = self.weights.matvec(MoshiStack::Depformer, 0, "proj", hidden, h, d);
        let mut normed = vec![0.0f32; d];
        let dep_ff = self.cfg.dep_ff;
        let mut ffn_h = vec![0.0f32; dep_ff];
        for l in 0..self.cfg.dep_layers {
            let gamma = self.weights.norm_gamma(MoshiStack::Depformer, l, "norm", d);
            rms_norm_into(&dep, &gamma, &mut normed);
            let proj = self.weights.matvec(MoshiStack::Depformer, l, "in_proj", &normed, d, d);
            let mut dep2 = vec![0.0f32; d];
            for i in 0..d { dep2[i] = dep[i] + proj[i]; }
            // Atenção do depformer sobre seu KV (1 passo/frame no stub, sem RoPE).
            let q = self.weights.matvec(MoshiStack::Depformer, l, "q", &dep2, d, d);
            let k = self.weights.matvec(MoshiStack::Depformer, l, "k", &dep2, d, d);
            let v = self.weights.matvec(MoshiStack::Depformer, l, "v", &dep2, d, d);
            self.dep_kv_k[l].extend_from_slice(&k);
            self.dep_kv_v[l].extend_from_slice(&v);
            // Janela deslizante (`depformer_context: 8` no config real).
            while self.dep_kv_k[l].len() / d > DEPFORMER_CONTEXT {
                self.dep_kv_k[l].drain(..d);
                self.dep_kv_v[l].drain(..d);
            }
            let seq = self.dep_kv_k[l].len() / d;
            let mut agg = vec![0.0f32; d];
            per_head_attn(&q, &self.dep_kv_k[l], &self.dep_kv_v[l], n_heads, head_dim, seq, &mut agg);
            let o = self.weights.matvec(MoshiStack::Depformer, l, "o", &agg, d, d);
            for i in 0..d { dep2[i] += o[i]; }

            let gamma_f = self.weights.norm_gamma(MoshiStack::Depformer, l, "ffn_norm", d);
            rms_norm_into(&dep2, &gamma_f, &mut normed);
            let gate = self.weights.matvec(MoshiStack::Depformer, l, "gate", &normed, d, dep_ff);
            let up = self.weights.matvec(MoshiStack::Depformer, l, "up", &normed, d, dep_ff);
            for i in 0..dep_ff {
                let g = gate[i];
                ffn_h[i] = g / (1.0 + (-g).exp()) * up[i];
            }
            let down = self.weights.matvec(MoshiStack::Depformer, l, "down", &ffn_h, dep_ff, d);
            let mut next = vec![0.0f32; d];
            for i in 0..d { next[i] = dep2[i] + down[i]; }
            dep = next;
        }
        // 16 códigos: média por chunk quantizada em 10 bits (stub; F3b usa heads reais).
        let mut codes = [0u16; MOSHI_N_CODEBOOKS];
        let chunk = d.div_ceil(MOSHI_N_CODEBOOKS).max(1);
        for c in 0..MOSHI_N_CODEBOOKS {
            let mut sum = 0.0f32;
            let mut n = 0usize;
            for i in 0..chunk {
                if let Some(&v) = dep.get(c * chunk + i) { sum += v; n += 1; }
            }
            let mean = if n > 0 { sum / n as f32 } else { 0.0 };
            let q = ((mean.clamp(-1.0, 1.0) * 0.5 + 0.5) * 1023.0).round() as i32;
            codes[c] = q.clamp(0, 1023) as u16;
        }
        (dep, codes)
    }
}

/// Mistura do delay acústico simplificado: sem anterior usa o frame atual;
/// com anterior, média dos dois (F3b: shift temporal + feedback de códigos).
pub(crate) fn delay_mix(prev: Option<&[f32]>, cur: &[f32]) -> Vec<f32> {
    match prev {
        Some(p) => p.iter().zip(cur.iter()).map(|(a, b)| 0.5 * a + 0.5 * b).collect(),
        None => cur.to_vec(),
    }
}
/// Atenção causal per-head sobre o cache (mesma matemática de `forward_one`,
/// `src/inference.rs:1058`). `out` acumula (`+=`); chamador zera antes.
fn per_head_attn(q: &[f32], k_cache: &[f32], v_cache: &[f32], n_heads: usize, head_dim: usize, seq: usize, out: &mut [f32]) {
    let h = n_heads * head_dim;
    let scale = (head_dim as f32).sqrt().recip();
    for hh in 0..n_heads {
        let hb = hh * head_dim;
        if hb + head_dim > h || hb + head_dim > q.len() {
            break;
        }
        let qb = &q[hb..hb + head_dim];
        let mut scores = vec![0.0f32; seq];
        let mut max = f32::NEG_INFINITY;
        for i in 0..seq {
            let kb = &k_cache[i * h + hb..i * h + hb + head_dim];
            let mut dot = 0.0f32;
            for (a, b) in qb.iter().zip(kb.iter()) { dot += a * b; }
            scores[i] = dot * scale;
            if scores[i] > max { max = scores[i]; }
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum; }
        for i in 0..seq {
            let vb = &v_cache[i * h + hb..i * h + hb + head_dim];
            let w = scores[i];
            for j in 0..head_dim { out[hb + j] += w * vb[j]; }
        }
    }
}

// ---------------------------------------------------------------------------
// Pesos GGUF reais, formato moshi.cpp `lm.*` fused (F3b)
// ---------------------------------------------------------------------------
// Layout medido em `models/personaplex-7b-v1-q4_k.gguf` (655 tensores,
// `n_kv=0`, dtypes Q4_K=545/F32=77/Q4_0=33). Ver `docs/ESPEC.md` §14.
// Temporal por camada: `in_projs.0 [4096,12288]` (QKV packed), `out_projs.0`
// [4096,4096], `gating.linear_in [4096,22528]` (gate+up packed, ff=11264),
// `gating.linear_out [11264,4096]`, `norm1/norm2.alpha [4096]` F32.
// Depformer por camada: `in_projs.0 [1024,3072]`, `out_projs.0 [1024,1024]`,
// `gating.{c}.linear_in [1024,5632]` x16 codebooks (F3b usa `gating.0` p/
// todos; per-codebook em F4), `gating.0.linear_out [2816,1024]`, norms F32.
// Entrada depformer: `depformer_in.0 [4096,1024]` (F3b usa `.0` p/ todos).

/// Nome de tensor temporal moshi.cpp por (camada, kind fused).
pub fn temporal_cpp_name(layer: usize, kind: &str) -> String {
    let b = format!("lm.transformer.layers.{}", layer);
    match kind {
        "qkv" => format!("{}.self_attn.in_projs.0.weight", b),
        "o" => format!("{}.self_attn.out_projs.0.weight", b),
        "fused_gate" => format!("{}.gating.linear_in.weight", b),
        "down" => format!("{}.gating.linear_out.weight", b),
        "norm1" => format!("{}.norm1.alpha", b),
        "norm2" => format!("{}.norm2.alpha", b),
        _ => String::new(),
    }
}

/// Nome de tensor depformer moshi.cpp por (camada, kind fused).
pub fn depformer_cpp_name(layer: usize, kind: &str) -> String {
    let b = format!("lm.depformer.layers.{}", layer);
    match kind {
        "qkv" => format!("{}.self_attn.in_projs.0.weight", b),
        "o" => format!("{}.self_attn.out_projs.0.weight", b),
        "fused_gate" => format!("{}.gating.0.linear_in.weight", b),
        "down" => format!("{}.gating.0.linear_out.weight", b),
        "norm1" => format!("{}.norm1.alpha", b),
        "norm2" => format!("{}.norm2.alpha", b),
        _ => String::new(),
    }
}

/// Índices fused do Temporal por camada (`None` = ausente → dummy).
#[derive(Debug, Clone, Default)]
pub struct MoshiCppTemporal {
    pub qkv: Option<usize>,
    pub o: Option<usize>,
    pub fused_gate: Option<usize>,
    pub down: Option<usize>,
    pub norm1: Option<usize>,
    pub norm2: Option<usize>,
}

/// Índices fused do Depformer por camada.
#[derive(Debug, Clone, Default)]
pub struct MoshiCppDep {
    pub qkv: Option<usize>,
    pub o: Option<usize>,
    pub fused_gate: Option<usize>,
    pub down: Option<usize>,
    pub norm1: Option<usize>,
    pub norm2: Option<usize>,
}

fn resolve_cpp_in(gg: &GgufFile, name: &str) -> Option<usize> {
    if name.is_empty() { return None; }
    gg.tensors.iter().position(|t| t.name == name)
}

/// Sonda camadas `0..` até a primeira sem nenhum tensor conhecido.
/// Retorna (temporal, depformer, depformer_in.0).
pub fn resolve_moshi_cpp(gg: &GgufFile) -> (Vec<MoshiCppTemporal>, Vec<MoshiCppDep>, Option<usize>) {
    let mut t = Vec::new();
    for l in 0..256 {
        let layer = MoshiCppTemporal {
            qkv: resolve_cpp_in(gg, &temporal_cpp_name(l, "qkv")),
            o: resolve_cpp_in(gg, &temporal_cpp_name(l, "o")),
            fused_gate: resolve_cpp_in(gg, &temporal_cpp_name(l, "fused_gate")),
            down: resolve_cpp_in(gg, &temporal_cpp_name(l, "down")),
            norm1: resolve_cpp_in(gg, &temporal_cpp_name(l, "norm1")),
            norm2: resolve_cpp_in(gg, &temporal_cpp_name(l, "norm2")),
        };
        let any = layer.qkv.is_some() || layer.o.is_some() || layer.fused_gate.is_some()
            || layer.down.is_some() || layer.norm1.is_some() || layer.norm2.is_some();
        if !any { break; }
        t.push(layer);
    }
    let mut d = Vec::new();
    for l in 0..64 {
        let layer = MoshiCppDep {
            qkv: resolve_cpp_in(gg, &depformer_cpp_name(l, "qkv")),
            o: resolve_cpp_in(gg, &depformer_cpp_name(l, "o")),
            fused_gate: resolve_cpp_in(gg, &depformer_cpp_name(l, "fused_gate")),
            down: resolve_cpp_in(gg, &depformer_cpp_name(l, "down")),
            norm1: resolve_cpp_in(gg, &depformer_cpp_name(l, "norm1")),
            norm2: resolve_cpp_in(gg, &depformer_cpp_name(l, "norm2")),
        };
        let any = layer.qkv.is_some() || layer.o.is_some() || layer.fused_gate.is_some()
            || layer.down.is_some() || layer.norm1.is_some() || layer.norm2.is_some();
        if !any { break; }
        d.push(layer);
    }
    let proj = resolve_cpp_in(gg, "lm.depformer_in.0.weight");
    (t, d, proj)
}

/// Quantos tensores fused moshi.cpp existem no GGUF (0 = outro formato).
pub fn count_moshi_cpp_tensors(gg: &GgufFile) -> usize {
    let (t, d, proj) = resolve_moshi_cpp(gg);
    let ct: usize = t.iter().map(|l| [l.qkv, l.o, l.fused_gate, l.down, l.norm1, l.norm2].into_iter().flatten().count()).sum();
    let cd: usize = d.iter().map(|l| [l.qkv, l.o, l.fused_gate, l.down, l.norm1, l.norm2].into_iter().flatten().count()).sum();
    ct + cd + proj.map(|_| 1).unwrap_or(0)
}

/// Pesos GGUF reais via `matvec_quant` fused + dequant cache.
/// `q/k/v` computam o fused `qkv` 1×/frame (memo em `begin_frame`) e dividem
/// em terços; `gate/up` idem sobre `fused_gate` (metades). Tensor ausente ou
/// com shape inesperado → dummy 0.5 (mesmo fallback de `DummyMoshiWeights`).
pub struct GgufMoshiWeights<'a> {
    gg: &'a GgufFile,
    mem: &'a crate::vm::MemBackend,
    t: Vec<MoshiCppTemporal>,
    d: Vec<MoshiCppDep>,
    proj_in: Option<usize>,
    dequant_cache: std::collections::HashMap<usize, Vec<f32>>,
    /// Memo fused por frame: (stack 0=temporal/1=dep, layer, which 0=qkv/1=gate).
    fused_memo: std::collections::HashMap<(u8, usize, u8), Vec<f32>>,
}

impl<'a> GgufMoshiWeights<'a> {
    pub fn new(gg: &'a GgufFile, mem: &'a crate::vm::MemBackend) -> Self {
        let (t, d, proj_in) = resolve_moshi_cpp(gg);
        Self { gg, mem, t, d, proj_in, dequant_cache: std::collections::HashMap::new(), fused_memo: std::collections::HashMap::new() }
    }

    /// Camadas temporais resolvidas (para checar cobertura em testes/CLI).
    pub fn n_temporal(&self) -> usize {
        self.t.len()
    }

    /// Camadas depformer resolvidas.
    pub fn n_depformer(&self) -> usize {
        self.d.len()
    }

    fn raw_of(&self, idx: usize, len: usize) -> Option<Vec<u8>> {
        let info = self.gg.tensors.get(idx)?;
        let file_offset = self.gg.data_offset + info.offset;
        if let Some(s) = self.mem.read_model_raw(file_offset, len) {
            return Some(s.to_vec());
        }
        let paddr = crate::memory::make_persistent_addr(file_offset as u128);
        self.mem.read(paddr, len).ok()
    }

    /// Caminho direto: fused `matvec_quant` (Q4_K/Q4_0/Q8_0/Q6_K) ou dequant
    /// cache + `matvec` (F32/F16). `None` = sem peso (caller usa dummy).
    pub(crate) fn fused_or_cached(&mut self, idx: usize, x: &[f32], in_dim: usize, out_dim: usize) -> Option<Vec<f32>> {
        let info = self.gg.tensors.get(idx)?;
        if info.n_elements != in_dim * out_dim {
            return None;
        }
        if let Some(raw_len) = crate::matvec_quant::quant_raw_len(info.dtype, info.n_elements) {
            if let Some(raw) = self.raw_of(idx, raw_len) {
                if let Some(y) = crate::matvec_quant::matvec_quant(x, &raw, info.dtype, in_dim, out_dim) {
                    return Some(y);
                }
            }
        }
        if let Some(w) = self.dequant_cache.get(&idx) {
            return Some(crate::matvec::matvec(x, w, in_dim, out_dim));
        }
        let raw_len = match info.dtype {
            0 => info.n_elements * 4,
            1 => info.n_elements * 2,
            _ => return None,
        };
        let raw = self.raw_of(idx, raw_len)?;
        let mut dst = vec![0.0f32; info.n_elements];
        if !crate::quant::dequantize(&raw, info.dtype, &mut dst, info.n_elements) {
            return None;
        }
        let y = crate::matvec::matvec(x, &dst, in_dim, out_dim);
        self.dequant_cache.insert(idx, dst);
        Some(y)
    }

    /// Fused memoizado (qkv ou gate) + divisão em partes iguais.
    fn fused_part(&mut self, stack: u8, layer: usize, which: u8, idx: usize, x: &[f32], in_dim: usize, parts: usize, part: usize) -> Option<Vec<f32>> {
        let key = (stack, layer, which);
        if let Some(full) = self.fused_memo.get(&key) {
            return Some(split_part(full, parts, part));
        }
        let info = self.gg.tensors.get(idx)?;
        if info.n_elements % in_dim != 0 {
            return None;
        }
        let fused_out = info.n_elements / in_dim;
        let full = self.fused_or_cached(idx, x, in_dim, fused_out)?;
        let out = split_part(&full, parts, part);
        self.fused_memo.insert(key, full);
        Some(out)
    }

    fn layer_idx(&self, stack: MoshiStack, layer: usize, kind: &str) -> Option<(u8, usize, u8, usize, usize, usize)> {
        // Retorna (stack_id, layer, which, idx, parts, part) para fused,
        // ou idx direto via which=255 para tensores não-fused.
        match stack {
            MoshiStack::Temporal => {
                let l = self.t.get(layer)?;
                match kind {
                    "q" => Some((0, layer, 0, l.qkv?, 3, 0)),
                    "k" => Some((0, layer, 0, l.qkv?, 3, 1)),
                    "v" => Some((0, layer, 0, l.qkv?, 3, 2)),
                    "o" => Some((0, layer, 255, l.o?, 1, 0)),
                    "gate" => Some((0, layer, 1, l.fused_gate?, 2, 0)),
                    "up" => Some((0, layer, 1, l.fused_gate?, 2, 1)),
                    "down" => Some((0, layer, 255, l.down?, 1, 0)),
                    _ => None,
                }
            }
            MoshiStack::Depformer => {
                let l = self.d.get(layer)?;
                match kind {
                    "q" => Some((1, layer, 0, l.qkv?, 3, 0)),
                    "k" => Some((1, layer, 0, l.qkv?, 3, 1)),
                    "v" => Some((1, layer, 0, l.qkv?, 3, 2)),
                    "in_proj" => Some((1, layer, 0, l.qkv?, 3, 0)),
                    "o" => Some((1, layer, 255, l.o?, 1, 0)),
                    "gate" => Some((1, layer, 1, l.fused_gate?, 2, 0)),
                    "up" => Some((1, layer, 1, l.fused_gate?, 2, 1)),
                    "down" => Some((1, layer, 255, l.down?, 1, 0)),
                    "proj" => Some((1, layer, 255, self.proj_in?, 1, 0)),
                    _ => None,
                }
            }
        }
    }
}

/// Divide vetor fused em `parts` partes iguais, retornando a `part`.
fn split_part(full: &[f32], parts: usize, part: usize) -> Vec<f32> {
    let n = full.len() / parts;
    full[part * n..(part + 1) * n].to_vec()
}

fn dummy_matvec(x: &[f32], out_dim: usize) -> Vec<f32> {
    vec![0.5 * x.iter().sum::<f32>(); out_dim]
}

impl<'a> MoshiWeights for GgufMoshiWeights<'a> {
    fn begin_frame(&mut self) {
        self.fused_memo.clear();
    }

    fn matvec(&mut self, stack: MoshiStack, layer: usize, kind: &str, x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        // "proj" temporal->dep usa depformer_in.0 (in=hidden, out=dep_dim).
        if let Some((_s, _l, which, idx, parts, part)) = self.layer_idx(stack, layer, kind) {
            if which == 255 {
                if let Some(y) = self.fused_or_cached(idx, x, in_dim, out_dim) {
                    return y;
                }
            } else if let Some(y) = self.fused_part(_s, _l, which, idx, x, in_dim, parts, part) {
                return y;
            }
        }
        dummy_matvec(x, out_dim)
    }

    fn norm_gamma(&mut self, stack: MoshiStack, layer: usize, kind: &str, dim: usize) -> Vec<f32> {
        let idx = match stack {
            MoshiStack::Temporal => self.t.get(layer).and_then(|l| match kind {
                "attn_norm" => l.norm1,
                "ffn_norm" => l.norm2,
                _ => None,
            }),
            MoshiStack::Depformer => self.d.get(layer).and_then(|l| match kind {
                "norm" | "attn_norm" => l.norm1,
                "ffn_norm" => l.norm2,
                _ => None,
            }),
        };
        if let Some(i) = idx {
            if let Some(info) = self.gg.tensors.get(i) {
                if info.n_elements == dim && (info.dtype == 0 || info.dtype == 1) {
                    let raw_len = if info.dtype == 0 { dim * 4 } else { dim * 2 };
                    if let Some(raw) = self.raw_of(i, raw_len) {
                        let mut dst = vec![0.0f32; dim];
                        if crate::quant::dequantize(&raw, info.dtype, &mut dst, dim) {
                            return dst;
                        }
                    }
                }
            }
        }
        vec![1.0; dim]
    }
}

/// KV auxiliar para testes sem arquivo real.
#[cfg(test)]
fn mock_kv(pairs: &[(&str, &str)]) -> std::collections::HashMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::gguf::{GgufFile, GgufTensorInfo};

    fn empty_gguf() -> GgufFile {
        GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: vec![], kv: HashMap::new(), data_offset: 0 }
    }

    #[test]
    fn test_defaults_personaplex() {
        let c = MoshiConfig::personaplex();
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.n_layers, 32);
        assert_eq!(c.n_heads, 32);
        assert_eq!(c.intermediate, 11_264);
        assert_eq!(c.dep_layers, 6);
        assert_eq!(c.dep_dim, 1024);
        assert_eq!(c.codebooks, 16);
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.samples_per_frame(), 1920);
        assert_eq!(MOSHI_N_STREAMS, 17);
    }

    #[test]
    fn test_is_moshi_arch() {
        assert!(is_moshi_arch("moshi"));
        assert!(is_moshi_arch("personaplex-7b-v1"));
        assert!(is_moshi_arch("kyutai/moshiko-pytorch-bf16"));
        assert!(is_moshi_arch("MIMI"));
        assert!(!is_moshi_arch("llama"));
        assert!(!is_moshi_arch("qwen2"));
    }

    #[test]
    fn test_from_gguf_defaults_sem_kv() {
        let gg = empty_gguf();
        let c = MoshiConfig::from_gguf(&gg);
        assert_eq!(c.hidden, MOSHI_HIDDEN);
        assert_eq!(c.n_layers, MOSHI_LAYERS);
    }

    #[test]
    fn test_from_gguf_le_kv() {
        let mut gg = empty_gguf();
        gg.kv = mock_kv(&[("general.architecture", "moshi"), ("llama.embedding_length", "4096"), ("llama.block_count", "32")]);
        let c = MoshiConfig::from_gguf(&gg);
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.n_layers, 32);
        assert!(is_moshi_arch(&c.arch));
    }

    #[test]
    fn test_from_config_json_minimo() {
        let c = MoshiConfig::from_config_json(r#"{"dim": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "hidden_dim": 11264}"#);
        assert_eq!(c.hidden, 4096);
        assert_eq!(c.n_layers, 32);
        assert_eq!(c.intermediate, 11_264);
        // JSON vazio = defaults
        let d = MoshiConfig::from_config_json("{}");
        assert_eq!(d.hidden, MOSHI_HIDDEN);
    }

    #[test]
    fn test_nomes_temporal_depformer_nao_vazios() {
        assert!(!temporal_tensor_names(0, "q").is_empty());
        assert!(!depformer_tensor_names(0, "in_proj").is_empty());
        assert!(!mimi_tensor_names("token_embd").is_empty());
        assert!(temporal_tensor_names(0, "invalido").is_empty());
    }

    #[test]
    fn test_resolve_encontra_blk() {
        let mut gg = empty_gguf();
        gg.tensors.push(GgufTensorInfo { name: "blk.0.attn_q.weight".to_string(), dims: vec![4096, 4096], shape: vec![4096, 4096], dtype: 12, offset: 0, n_elements: 4096 * 4096 });
        gg.tensors.push(GgufTensorInfo { name: "depformer.layers.0.self_attn.in_proj.weight".to_string(), dims: vec![1024, 1024], shape: vec![1024, 1024], dtype: 0, offset: 0, n_elements: 1024 * 1024 });
        let cfg = MoshiConfig::default();
        let t = resolve_temporal_names(&gg, &cfg);
        let d = resolve_depformer_names(&gg, &cfg);
        assert_eq!(t[0].q, Some(0));
        assert_eq!(t[1].q, None);
        assert_eq!(d[0].in_proj, Some(1));
        assert_eq!(count_moshi_tensors(&gg, &cfg), 1);
    }

    #[test]
    fn test_describe_map_contem_camadas() {
        let s = describe_map(&MoshiConfig::default());
        assert!(s.contains("blk.0.attn_q.weight"));
        assert!(s.contains("depformer.layers.0"));
        assert!(s.contains("token_embd.weight"));
    }

    // --- Forward F3a (pesos dummy, dims pequenas) ---

    fn tiny_forward_cfg() -> MoshiConfig {
        MoshiConfig {
            hidden: 32, n_layers: 2, n_heads: 4, intermediate: 64,
            dep_layers: 2, dep_dim: 16, dep_heads: 2, dep_ff: 32,
            codebooks: MOSHI_N_CODEBOOKS, text_vocab: 32000,
            frame_hz: MOSHI_FRAME_HZ, sample_rate: MOSHI_SAMPLE_RATE,
            context_length: 2048, rope_theta: MOSHI_ROPE_THETA,
            norm_eps: MOSHI_NORM_EPS, arch: "moshi".to_string(),
        }
    }

    fn sine_frame(h: usize) -> Vec<f32> {
        (0..h).map(|i| ((i as f32 * 0.3).sin() * 0.5)).collect()
    }

    #[test]
    fn test_forward_dims_e_seq() {
        let mut inf = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        assert_eq!(inf.seq_len_frames(), 0);
        let out = inf.forward_frame(&sine_frame(32));
        assert_eq!(out.hidden.len(), 32);
        assert_eq!(out.dep_hidden.len(), 16);
        assert_eq!(out.codes.len(), 16);
        assert_eq!(inf.seq_len_frames(), 1);
        inf.forward_frame(&sine_frame(32));
        assert_eq!(inf.seq_len_frames(), 2);
        for &v in out.hidden.iter().chain(out.dep_hidden.iter()) {
            assert!(v.is_finite(), "não-finito {}", v);
        }
    }

    #[test]
    fn test_forward_deterministico() {
        let f = sine_frame(32);
        let mut a = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        let mut b = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        let oa = a.forward_frame(&f);
        let ob = b.forward_frame(&f);
        assert_eq!(oa.codes, ob.codes);
        for (x, y) in oa.hidden.iter().zip(ob.hidden.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn test_forward_codes_faixa() {
        let mut inf = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        let out = inf.forward_frame(&vec![0.1; 32]);
        assert!(out.codes.iter().all(|&c| c <= 1023), "{:?}", out.codes);
    }

    #[test]
    fn test_delay_mix_media() {
        assert_eq!(delay_mix(None, &[4.0, 6.0]), vec![4.0, 6.0]);
        assert_eq!(delay_mix(Some(&[2.0, 4.0]), &[4.0, 6.0]), vec![3.0, 5.0]);
    }

    #[test]
    fn test_frames_diferentes_saidas_diferentes() {
        // Pipeline responde à entrada (delay + KV): mixes distintos -> outs distintos.
        let mut inf = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        let o1 = inf.forward_frame(&sine_frame(32));
        let o2 = inf.forward_frame(&vec![0.1; 32]);
        let diff: f32 = o1.hidden.iter().zip(o2.hidden.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-3, "pipeline insensível à entrada? diff {}", diff);
    }

    #[test]
    fn test_truncate_zero_igual_fresco() {
        let f = sine_frame(32);
        let mut a = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        a.forward_frame(&f);
        a.forward_frame(&f);
        a.truncate_frames(0);
        assert_eq!(a.seq_len_frames(), 0);
        let oa = a.forward_frame(&f);
        let mut b = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        let ob = b.forward_frame(&f);
        assert_eq!(oa.codes, ob.codes);
        for (x, y) in oa.hidden.iter().zip(ob.hidden.iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    #[test]
    fn test_truncate_e_clear() {
        let f = sine_frame(32);
        let mut inf = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        inf.forward_frame(&f);
        inf.forward_frame(&f);
        inf.forward_frame(&f);
        inf.truncate_frames(1);
        assert_eq!(inf.seq_len_frames(), 1);
        // KV truncado de verdade: 1 layer temporal guarda 1*h floats
        assert_eq!(inf.kv_k[0].len(), 32);
        inf.clear();
        assert_eq!(inf.seq_len_frames(), 0);
        assert!(inf.kv_k.iter().all(|k| k.is_empty()));
        assert!(inf.dep_kv_k.iter().all(|k| k.is_empty()));
    }

    #[test]
    fn test_forward_profile_ms() {
        let mut inf = MoshiInference::new(tiny_forward_cfg(), DummyMoshiWeights);
        inf.profile = true;
        inf.forward_frame(&sine_frame(32));
        assert!(inf.last_ms.total_ms >= 0.0);
        assert!(inf.last_ms.temporal_ms >= 0.0);
        assert!(inf.last_ms.depformer_ms >= 0.0);
    }

    // --- GGUF real PersonaPlex Q4_K (pula sem o arquivo) ---

    const PP_GGUF: &str = "./models/personaplex-7b-v1-q4_k.gguf";

    fn pp_available() -> bool {
        std::path::Path::new(PP_GGUF).exists()
    }

    fn pp_mem() -> Option<(crate::gguf::GgufFile, crate::vm::MemBackend)> {
        if !pp_available() {
            return None;
        }
        let gg = crate::gguf::GgufFile::open(PP_GGUF).ok()?;
        let mut mgr = crate::memory::MemoryManager::new_in_memory();
        mgr.load_gguf_model(PP_GGUF).ok()?;
        Some((gg, crate::vm::MemBackend::Cpu(mgr)))
    }

    #[test]
    fn test_personaplex_header_real() {
        let Some((gg, _mem)) = pp_mem() else {
            eprintln!("skip sem GGUF PersonaPlex");
            return;
        };
        assert_eq!(gg.n_tensors, 655, "conversão moshi.cpp tem 655 tensores");
        let (t, d, proj) = resolve_moshi_cpp(&gg);
        assert_eq!(t.len(), 32, "temporal 32 layers");
        assert_eq!(d.len(), 6, "depformer 6 layers");
        assert!(proj.is_some(), "depformer_in.0");
        assert!(t.iter().all(|l| l.qkv.is_some() && l.o.is_some() && l.fused_gate.is_some() && l.down.is_some() && l.norm1.is_some() && l.norm2.is_some()));
        assert!(d.iter().all(|l| l.qkv.is_some() && l.o.is_some() && l.fused_gate.is_some() && l.down.is_some()));
        assert_eq!(gg.tensors[t[0].qkv.unwrap()].dtype, 12, "qkv Q4_K");
        assert_eq!(count_moshi_cpp_tensors(&gg), 32 * 6 + 6 * 6 + 1);
        // config json do repo (dim/n_q/text_card)
        let json = std::fs::read_to_string("./models/personaplex-config.json").expect("personaplex-config.json");
        let cfg = MoshiConfig::from_config_json(&json);
        assert_eq!(cfg.hidden, 4096);
        assert_eq!(cfg.n_layers, 32);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.codebooks, 16);
        assert_eq!(cfg.text_vocab, 32000);
        assert_eq!(cfg.dep_layers, 6);
        assert_eq!(cfg.dep_dim, 1024);
    }

    #[test]
    fn test_gguf_matvec_parity_linears() {
        // lm.linears.0 [1024,2048] Q4_K: fused vs dequant+dot (tol igual matvec_quant).
        let Some((gg, mem)) = pp_mem() else {
            eprintln!("skip sem GGUF PersonaPlex");
            return;
        };
        let idx = gg.tensors.iter().position(|t| t.name == "lm.linears.0.weight").expect("lm.linears.0.weight");
        let mut w = GgufMoshiWeights::new(&gg, &mem);
        let x: Vec<f32> = (0..1024).map(|i| ((i as f32 * 0.13).sin() * 0.7)).collect();
        let y_fused = w.fused_or_cached(idx, &x, 1024, 2048).expect("fused linears.0");
        assert_eq!(y_fused.len(), 2048);
        // Referência: dequant full + dot
        let info = &gg.tensors[idx];
        let raw = w.raw_of(idx, (info.n_elements / 256) * 144).expect("raw");
        let mut wfull = vec![0.0f32; info.n_elements];
        assert!(crate::quant::dequantize(&raw, info.dtype, &mut wfull, info.n_elements));
        let y_ref = crate::matvec::matvec_ndarray(&x, &wfull, 1024, 2048);
        let max_diff = y_fused.iter().zip(y_ref.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "max_diff {}", max_diff);
    }

    #[test]
    fn test_gguf_qkv_split_shapes() {
        // 1 fused qkv temporal (4096x12288 Q4_K) dividido em q/k/v de 4096.
        let Some((gg, mem)) = pp_mem() else {
            eprintln!("skip sem GGUF PersonaPlex");
            return;
        };
        let mut w = GgufMoshiWeights::new(&gg, &mem);
        assert_eq!(w.n_temporal(), 32);
        assert_eq!(w.n_depformer(), 6);
        let x: Vec<f32> = (0..4096).map(|i| ((i as f32 * 0.01).sin() * 0.3)).collect();
        let q = MoshiWeights::matvec(&mut w, MoshiStack::Temporal, 0, "q", &x, 4096, 4096);
        let k = MoshiWeights::matvec(&mut w, MoshiStack::Temporal, 0, "k", &x, 4096, 4096);
        let v = MoshiWeights::matvec(&mut w, MoshiStack::Temporal, 0, "v", &x, 4096, 4096);
        assert_eq!(q.len(), 4096);
        assert_ne!(q, k, "q e k devem diferir (terços distintos)");
        assert_ne!(k, v);
        for &val in q.iter().chain(k.iter()).chain(v.iter()) {
            assert!(val.is_finite());
        }
        let gate = MoshiWeights::matvec(&mut w, MoshiStack::Temporal, 0, "gate", &x, 4096, 11264);
        assert_eq!(gate.len(), 11264);
        let gamma = MoshiWeights::norm_gamma(&mut w, MoshiStack::Temporal, 0, "attn_norm", 4096);
        assert_eq!(gamma.len(), 4096);
        assert!(gamma.iter().all(|g| g.is_finite() && *g > 0.0));
    }
}
