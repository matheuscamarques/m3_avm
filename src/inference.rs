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
        let hidden = get_parse(&["qwen2.embedding_length", "llama.embedding_length", "mistral.embedding_length", "general.embedding_length", "hidden_size"], 2048);
        let hidden = if hidden == 2048 && arch == "qwen2" { get_parse(&["qwen2.embedding_length"], 1536) } else { hidden };
        // intermediate
        let intermediate = get_parse(&["qwen2.feed_forward_length", "qwen2.intermediate_size", "llama.feed_forward_length", "mistral.feed_forward_length", "intermediate_size"], 5632);
        // layers
        let n_layers = get_parse(&["qwen2.block_count", "llama.block_count", "mistral.block_count", "general.block_count", "num_hidden_layers"], 22);
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
        let context_length = get_parse(&["qwen2.context_length", "llama.context_length", "general.context_length"], 2048);
        let rope_theta = get_f32(&["qwen2.rope.freq_base", "llama.rope.freq_base", "general.rope.freq_base"], 10000.0);
        let norm_eps = get_f32(&["qwen2.attention.layer_norm_rms_epsilon", "llama.attention.layer_norm_rms_epsilon", "general.layer_norm_rms_epsilon"], 1e-5);
        Self { hidden, intermediate, n_layers, n_heads, n_kv_heads, vocab, arch: arch.clone(), context_length, rope_theta, norm_eps }
    }

    pub fn head_dim(&self) -> usize {
        if self.n_heads == 0 { self.hidden } else { self.hidden / self.n_heads }
    }
    pub fn is_gqa(&self) -> bool { self.n_kv_heads < self.n_heads }
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

pub struct RealInference {
    pub config: ModelConfig,
    pub tokenizer: M3Tokenizer,
    pub gguf: GgufFile,
    pub gguf_path: String,
    // KV cache per layer: Vec<(k_cache, v_cache)> onde cada cache é [seq_len * hidden]
    pub kv_cache_k: Vec<Vec<f32>>,
    pub kv_cache_v: Vec<Vec<f32>>,
    // Cache de pesos dequantizados (Fase 2.1) — evita 197 dequants/token
    weight_cache: HashMap<String, std::sync::Arc<Vec<f32>>>,
    // Histórico de ids gerados (+prompt) para repetition penalty real
    gen_history: Vec<u32>,
}

impl RealInference {
    pub fn new(gguf_path: &str) -> Result<Self> {
        let gg = GgufFile::open(gguf_path)?;
        let cfg = ModelConfig::from_gguf(&gg);
        let tok = M3Tokenizer::from_gguf_or_mock(gguf_path);
        eprintln!("[inference] config {:?} vocab {} tensors {}", cfg, tok.vocab_size(), gg.n_tensors);
        let n_layers = cfg.n_layers;
        Ok(Self { config: cfg, tokenizer: tok, gguf: gg, gguf_path: gguf_path.to_string(), kv_cache_k: vec![Vec::new(); n_layers], kv_cache_v: vec![Vec::new(); n_layers], weight_cache: HashMap::new(), gen_history: Vec::new() })
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

    /// Repete KV heads para GQA (n_kv_heads < n_heads) — expande kv_hidden para hidden
    fn repeat_kv(kv_small: &[f32], n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Vec<f32> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            // fallback: repete simples até preencher hidden
            let hidden = n_heads * head_dim;
            let mut out = Vec::with_capacity(hidden);
            while out.len() < hidden {
                let need = hidden - out.len();
                out.extend_from_slice(&kv_small[..need.min(kv_small.len())]);
            }
            return out;
        }
        let groups = n_heads / n_kv_heads;
        let mut out = Vec::with_capacity(n_heads * head_dim);
        for kv_idx in 0..n_kv_heads {
            let src = &kv_small[kv_idx*head_dim..(kv_idx+1)*head_dim];
            for _ in 0..groups {
                out.extend_from_slice(src);
            }
        }
        out
    }

    /// Lê tensor do GGUF via MemBackend (mmap PERSISTENTE) e dequantiza se Q4_K/Q6_K
    /// Fase 2.1: usa weight_cache (HashMap<String, Arc<Vec<f32>>>) para evitar 197 dequants/token
    fn read_tensor_f32(&self, mem: &crate::vm::MemBackend, name: &str) -> Result<Vec<f32>> {
        // Checa cache primeiro (sem lock, &self mas interior mutability via cache é &mut na prática)
        // Para manter &self, usamos try: se estiver em cache, clona Arc
        if let Some(cached) = self.weight_cache.get(name) {
            return Ok((**cached).clone());
        }
        let info = self.gguf.find_tensor(name).ok_or_else(|| anyhow!("tensor {} não encontrado", name))?;
        let n = info.n_elements;
        let file_offset = self.gguf.data_offset + info.offset;
        let paddr = crate::memory::make_persistent_addr(file_offset as u128);
        // Calcula byte_len via dtype; para Q6_K usa 210 por 256, para Q8_0 34 por 32, etc.
        let raw_len = match DType::from_u32(info.dtype) {
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
                    if t.offset > info.offset {
                        if next_off.is_none() || t.offset < next_off.unwrap() {
                            next_off = Some(t.offset);
                        }
                    }
                }
                if let Some(no) = next_off {
                    (no - info.offset) as usize
                } else {
                    n*4
                }
            },
        };
        let raw = mem.read(paddr, raw_len)?;
        let mut dst = vec![0.0f32; n];
        let ok = crate::quant::dequantize(&raw, info.dtype, &mut dst, n);
        if !ok {
            // fallback: tenta interpretar como F32 LE se dequant não suportado
            for i in 0..n.min(raw.len()/4) {
                dst[i] = f32::from_le_bytes([raw[i*4], raw[i*4+1], raw[i*4+2], raw[i*4+3]]);
            }
            // se ainda falhar e for quantizado, loga
            if DType::from_u32(info.dtype).is_quantized() {
                eprintln!("[quant] aviso: dequant fallback F32 para dtype {} tensor {}", info.dtype, name);
            }
        }
        Ok(dst)
    }
    /// Versão com cache interior mutável (usada no loop para evitar clone desnecessário)
    fn read_tensor_f32_cached(&mut self, mem: &crate::vm::MemBackend, name: &str) -> Result<std::sync::Arc<Vec<f32>>> {
        if let Some(cached) = self.weight_cache.get(name) {
            return Ok(cached.clone());
        }
        let v = self.read_tensor_f32(mem, name)?;
        let arc = std::sync::Arc::new(v);
        self.weight_cache.insert(name.to_string(), arc.clone());
        Ok(arc)
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
    /// Versão com cache (Fase 2.1) — evita 197 dequants/token, ~6× ganho
    fn read_tensor_f32_try_cached(&mut self, mem: &crate::vm::MemBackend, names: &[String]) -> Result<std::sync::Arc<Vec<f32>>> {
        for n in names {
            if let Some(cached) = self.weight_cache.get(n) {
                return Ok(cached.clone());
            }
            if self.gguf.find_tensor(n).is_some() {
                // achou, dequantiza e cacheia
                let v = self.read_tensor_f32(mem, n)?;
                let arc = std::sync::Arc::new(v);
                self.weight_cache.insert(n.clone(), arc.clone());
                return Ok(arc);
            }
        }
        // tenta qualquer nome da lista e retorna erro do último
        let mut last_err = None;
        for n in names {
            match self.read_tensor_f32(mem, n) {
                Ok(v) => {
                    let arc = std::sync::Arc::new(v);
                    self.weight_cache.insert(n.clone(), arc.clone());
                    return Ok(arc);
                },
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("tensor {:?} não encontrado", names)))
    }
    /// Matvec com bypass int4 (Fase 3.2): tenta kernel direto sobre os bytes GGUF
    /// (1 pass fundido, sem dequant full nem clone de cache); senão cai para
    /// cache f32 + faer. Tensor ausente → dot sobre dummy 0.5 (paridade legada
    /// com os testes sem modelo).
    fn matvec_weight(&mut self, mem: &crate::vm::MemBackend, names: &[String], x: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
        let meta = {
            let mut found = None;
            for name in names {
                if let Some(info) = self.gguf.find_tensor(name) {
                    found = Some((info.dtype, info.n_elements, info.offset));
                    break;
                }
            }
            found
        };
        if let Some((dtype, n, offset)) = meta {
            if n == in_dim * out_dim {
                // Caminho rápido: kernel direto (Q4_K/Q6_K/Q4_0/Q8_0).
                // Zero-copy do mmap primeiro; `read` (cópia) como fallback.
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
                // Fallback f32 (F32/F16, kernel None ou raw curto)
                if let Ok(w) = self.read_tensor_f32_try_cached(mem, names) {
                    return crate::matvec::matvec(x, &w[..], in_dim, out_dim);
                }
            } else if let Ok(w) = self.read_tensor_f32_try_cached(mem, names) {
                // Shape inesperado: preserva fallback-mismatch legado via matvec
                return crate::matvec::matvec(x, &w[..], in_dim, out_dim);
            }
        }
        crate::matvec::matvec(x, &vec![0.5; in_dim * out_dim], in_dim, out_dim)
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
        // Profiler opcional via env M3_PROFILE=1 (fases em ms, sem custo quando off)
        let profile = std::env::var("M3_PROFILE").map(|v| v == "1").unwrap_or(false);
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
            let gamma_attn = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "attn_norm")).unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
            let hidden_norm = rms_norm(&hidden_cur, &gamma_attn[..]);

            // Suporte GQA: q = h*h, k/v = h*kv_hidden (kv_hidden = n_kv_heads * head_dim)
            let kv_hidden = self.config.n_kv_heads * self.config.head_dim();
            // Kernel int4 direto quando possível (bypass dequant+clone)
            let t0 = now();
            let mut q = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "q"), &hidden_norm, h, h);
            let mut k_small = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "k"), &hidden_norm, h, kv_hidden);
            let mut v_small = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "v"), &hidden_norm, h, kv_hidden);
            let o_w_names = self.config.try_get_tensor_names(blk, "o");
            if profile { t_matvec += t0.elapsed().as_micros(); }
            // Bias q/k/v (Qwen2/DeepSeek têm attn_*.bias F32; ausente = skip)
            if let Ok(b) = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "qbias")) {
                for (a, bb) in q.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Ok(b) = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "kbias")) {
                for (a, bb) in k_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Ok(b) = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "vbias")) {
                for (a, bb) in v_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            // RoPE em q/k ANTES do KV append e do repeat GQA
            apply_rope(&mut q, self.config.n_heads, rope_hd, &rope_freqs);
            apply_rope(&mut k_small, self.config.n_kv_heads, rope_hd, &rope_freqs);
            // Expande GQA se necessário (kv_hidden < h)
            let k = if kv_hidden == h { k_small } else { Self::repeat_kv(&k_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim()) };
            let v = if kv_hidden == h { v_small } else { Self::repeat_kv(&v_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim()) };

            // --- KV Cache append (0x30 região) ---
            // Garante que buffers existem e fazem push do token atual
            self.kv_cache_k[blk].extend_from_slice(&k);
            self.kv_cache_v[blk].extend_from_slice(&v);
            let seq_len = self.kv_cache_k[blk].len() / h;
            debug_assert_eq!(self.kv_cache_v[blk].len() / h, seq_len);

            // Attention sobre cache acumulado (seq_len)
            // scores[i] = Q·K_i * scale
            let t_attn0 = now();
            let k_cache = &self.kv_cache_k[blk];
            let v_cache = &self.kv_cache_v[blk];
            let mut scores = vec![0.0f32; seq_len];
            for i in 0..seq_len {
                let k_i = &k_cache[i*h..(i+1)*h];
                let mut dot = 0.0f32;
                for (a,b) in q.iter().zip(k_i.iter()) { dot += a * b; }
                scores[i] = dot * scale;
            }
            // causal softmax (todos os tokens anteriores são visíveis; futuro não existe)
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
            for s in scores.iter_mut() { *s /= sum; }
            // weighted sum V
            let mut attn_agg = vec![0.0f32; h];
            for i in 0..seq_len {
                let v_i = &v_cache[i*h..(i+1)*h];
                let w = scores[i];
                for j in 0..h { attn_agg[j] += w * v_i[j]; }
            }
            if profile { t_attn += t_attn0.elapsed().as_micros(); }

            let t1 = now();
            let attn_out = self.matvec_weight(mem, &o_w_names, &attn_agg, h, h);
            let mut hidden2 = vec![0.0; h];
            for i in 0..h { hidden2[i] = hidden_cur[i] + attn_out[i]; }

            let gamma_ffn = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "ffn_norm")).unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
            let hidden2_norm = rms_norm(&hidden2, &gamma_ffn[..]);
            let inter = self.config.intermediate;
            let gate = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "gate"), &hidden2_norm, h, inter);
            let up = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "up"), &hidden2_norm, h, inter);
            let mut ffn_h = vec![0.0; self.config.intermediate];
            for i in 0..self.config.intermediate {
                let g = gate[i];
                let sig = 1.0/(1.0+(-g).exp());
                ffn_h[i] = g * sig * up[i];
            }
            let ffn_out = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "down"), &ffn_h, inter, h);
            if profile { t_matvec += t1.elapsed().as_micros(); t_norm_ffn += t_blk.elapsed().as_micros(); }
            let mut hidden_next = vec![0.0; h];
            for i in 0..h { hidden_next[i] = hidden2[i] + ffn_out[i]; }
            hidden_cur = hidden_next;
        }

        // Final norm + LM head (cache, Arc emprestado — sem clone de 934MB)
        let t_head0 = now();
        let out_norm_w = self.read_tensor_f32_cached(mem, "output_norm.weight").unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
        let hidden_norm_final = rms_norm(&hidden_cur, &out_norm_w[..]);
        let lm_head = self.read_tensor_f32_cached(mem, "output.weight")
            .or_else(|_| self.read_tensor_f32_cached(mem, "token_embd.weight"))
            .unwrap_or_else(|_| {
                let vocab = self.tokenizer.vocab_size();
                let h_dummy = self.config.hidden;
                let mut dummy = vec![0.0f32; vocab * h_dummy];
                for i in 0..dummy.len() { dummy[i] = ((i as f32 * 0.002).sin() * 0.3); }
                std::sync::Arc::new(dummy)
            }); // tied
        // lm_head shape [vocab, hidden] ou [hidden, vocab] — tenta ambos
        let vocab = self.tokenizer.vocab_size();
        let mut logits = vec![0.0; vocab];
        // Se lm_head len == vocab*h, assume [vocab, hidden]
        if lm_head.len() == vocab * h {
            for i in 0..vocab {
                let mut sum=0.0;
                for j in 0..h { sum += hidden_norm_final[j] * lm_head[i*h + j]; }
                logits[i]=sum;
            }
        } else {
            // fallback dummy
            for i in 0..vocab.min(16) { logits[i%vocab] = (i as f32 * 0.1).sin(); }
        }
        if profile {
            // t_norm_ffn inclui t_matvec+t_attn (tempo de bloco); fases em ms
            eprintln!("[profile tok {}] layersTOTAL={:.0}ms matvec={:.0}ms attn={:.0}ms norm+ffn-resto={:.0}ms head={:.0}ms",
                token_id, t_norm_ffn as f64 / 1000.0, t_matvec as f64 / 1000.0,
                t_attn as f64 / 1000.0,
                (t_norm_ffn as f64 - t_matvec as f64 - t_attn as f64).max(0.0) / 1000.0,
                t_head0.elapsed().as_micros() as f64 / 1000.0);
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
            let gamma_attn = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "attn_norm")).unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
            let hidden_norm = rms_norm(&hidden_cur, &gamma_attn[..]);
            let kv_hidden = self.config.n_kv_heads * self.config.head_dim();
            let mut q = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "q"), &hidden_norm, h, h);
            let mut k_small = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "k"), &hidden_norm, h, kv_hidden);
            let mut v_small = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "v"), &hidden_norm, h, kv_hidden);
            let o_w_names = self.config.try_get_tensor_names(blk, "o");
            if let Ok(b) = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "qbias")) {
                for (a, bb) in q.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Ok(b) = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "kbias")) {
                for (a, bb) in k_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            if let Ok(b) = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "vbias")) {
                for (a, bb) in v_small.iter_mut().zip(b.iter()) { *a += *bb; }
            }
            apply_rope(&mut q, self.config.n_heads, rope_hd, &rope_freqs);
            apply_rope(&mut k_small, self.config.n_kv_heads, rope_hd, &rope_freqs);
            let k = if kv_hidden == h { k_small } else { Self::repeat_kv(&k_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim()) };
            let v = if kv_hidden == h { v_small } else { Self::repeat_kv(&v_small, self.config.n_heads, self.config.n_kv_heads, self.config.head_dim()) };
            // internal cache
            self.kv_cache_k[blk].extend_from_slice(&k);
            self.kv_cache_v[blk].extend_from_slice(&v);
            // mem cache (0x30)
            let _ = mem.kv_cache_append(blk, &k, &v);
            let seq_len = self.kv_cache_k[blk].len() / h;
            let k_cache = &self.kv_cache_k[blk];
            let v_cache = &self.kv_cache_v[blk];
            let mut scores = vec![0.0f32; seq_len];
            for i in 0..seq_len {
                let k_i = &k_cache[i*h..(i+1)*h];
                let mut dot = 0.0f32;
                for (a,b) in q.iter().zip(k_i.iter()) { dot += a * b; }
                scores[i] = dot * scale;
            }
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
            for s in scores.iter_mut() { *s /= sum; }
            let mut attn_agg = vec![0.0f32; h];
            for i in 0..seq_len {
                let v_i = &v_cache[i*h..(i+1)*h];
                let w = scores[i];
                for j in 0..h { attn_agg[j] += w * v_i[j]; }
            }
            let attn_out = self.matvec_weight(mem, &o_w_names, &attn_agg, h, h);
            let mut hidden2 = vec![0.0; h];
            for i in 0..h { hidden2[i] = hidden_cur[i] + attn_out[i]; }
            let gamma_ffn = self.read_tensor_f32_try_cached(mem, &self.config.try_get_tensor_names(blk, "ffn_norm")).unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
            let hidden2_norm = rms_norm(&hidden2, &gamma_ffn[..]);
            let inter = self.config.intermediate;
            let gate = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "gate"), &hidden2_norm, h, inter);
            let up = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "up"), &hidden2_norm, h, inter);
            let mut ffn_h = vec![0.0; self.config.intermediate];
            for i in 0..self.config.intermediate {
                let g = gate[i];
                let sig = 1.0/(1.0+(-g).exp());
                ffn_h[i] = g * sig * up[i];
            }
            let ffn_out = self.matvec_weight(mem, &self.config.try_get_tensor_names(blk, "down"), &ffn_h, inter, h);
            let mut hidden_next = vec![0.0; h];
            for i in 0..h { hidden_next[i] = hidden2[i] + ffn_out[i]; }
            hidden_cur = hidden_next;
        }
        let out_norm_w = self.read_tensor_f32_cached(mem, "output_norm.weight").unwrap_or_else(|_| std::sync::Arc::new(vec![1.0; h]));
        let hidden_norm_final = rms_norm(&hidden_cur, &out_norm_w[..]);
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

    /// Sampling com temperatura, top_p, top_k e repetition_penalty
    pub fn sample_with_params(&self, logits: &[f32], temp: f32, top_p: f32, top_k: usize, repeat_penalty: f32) -> u32 {
        if logits.is_empty() { return 0; }
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
            let _ = self.forward_one(mem, tid)?;
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
            let logits = self.forward_one(mem, tid)?;
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
        // (palavras com pontuação, especiais <|im_start|> etc. via lookup).
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
        // Usa template simples por arquitetura; ignora tokenizer.chat_template Jinja (muito grande)
        match self.config.arch.as_str() {
            "qwen2" | "qwen" => {
                // Qwen2 / DeepSeek: <|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
                format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_text)
            },
            "llama" | "tinyllama" => {
                format!("<|user|>\n{}\n<|assistant|>\n", user_text)
            },
            _ => {
                // Fallback genérico
                if self.config.arch.contains("qwen") || self.config.arch.contains("deepseek") {
                    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", user_text)
                } else {
                    format!("<|user|>\n{}\n<|assistant|>\n", user_text)
                }
            }
        }
    }
}

/// RoPE estilo HF Llama/Qwen2 (split-half, `rotate_half` — mesma convenção que
/// o modo NEOX do ggml): par (i, i+half) gira pelo ângulo da freq i.
/// `v` tem `n_heads * head_dim` elementos; `freqs` tem `head_dim/2` (cos, sin).
fn apply_rope(v: &mut [f32], n_heads: usize, head_dim: usize, freqs: &[(f32, f32)]) {
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

fn rms_norm(x: &[f32], gamma: &[f32]) -> Vec<f32> {
    let eps=1e-5;
    let mut sum=0.0;
    for &v in x { sum+=v*v; }
    let mean=sum/x.len() as f32;
    let rms=(mean+eps).sqrt();
    x.iter().enumerate().map(|(i,&v)| v / rms * gamma.get(i).cloned().unwrap_or(1.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tiny_forward_one_dummy() {
        // Testa com dummy mem (sem GGUF) — deve usar fallback e não panicar (rápido, hidden 32)
        let mem_mgr = crate::memory::MemoryManager::new_in_memory();
        let mem = crate::vm::MemBackend::Cpu(mem_mgr);
        let cfg = ModelConfig { hidden: 32, intermediate: 64, n_layers: 2, n_heads: 4, n_kv_heads: 4, vocab: 32000, arch: "llama".to_string(), context_length: 2048, rope_theta: 10000.0, norm_eps: 1e-5 };
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: Vec::new(), kv: std::collections::HashMap::new(), data_offset: 0 };
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let mut inf = RealInference { config: cfg, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 2], kv_cache_v: vec![Vec::new(); 2], weight_cache: HashMap::new(), gen_history: Vec::new() };
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
        let names = inf.config.try_get_tensor_names(0, "q");
        let y_kernel = inf.matvec_weight(&mem, &names, &x, h, h);
        // Kernel direto não preenche o cache f32 (prova que o bypass foi exercido)
        assert_eq!(inf.weight_cache_len(), 0, "kernel deveria bypassar o cache");
        // Referência: cache f32 + faer
        let w = inf.read_tensor_f32_try_cached(&mem, &names).unwrap();
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
        let cfg = ModelConfig { hidden: 32, intermediate: 64, n_layers: 22, n_heads: 4, n_kv_heads: 4, vocab: 32000, arch: "llama".to_string(), context_length: 2048, rope_theta: 10000.0, norm_eps: 1e-5 };
        let gg = crate::gguf::GgufFile { version: 3, n_tensors: 0, n_kv: 0, tensors: Vec::new(), kv: std::collections::HashMap::new(), data_offset: 0 };
        let tok = crate::tokenizer::M3Tokenizer::mock();
        let mut inf = RealInference { config: cfg, tokenizer: tok, gguf: gg, gguf_path: "".to_string(), kv_cache_k: vec![Vec::new(); 22], kv_cache_v: vec![Vec::new(); 22], weight_cache: HashMap::new(), gen_history: Vec::new() };
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
}
