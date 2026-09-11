//! depformer.rs — KV deslizante + atenção causal por head (RFC-0032).
//!
//! Mecânica do passo depformer sem RoPE (paridade com o stub
//! `forward_depformer` em `moshi.rs`): append de K/V com janela,
//! atenção causal per-head sobre a janela, amostragem por codebook
//! vive no `exec_depformer` (precisa do RNG do contexto).
//! Chaveado por (stream, layer) desde o dia um (17 streams: RFC-0033).

use thiserror::Error;

/// Janela default (== `DEPFORMER_CONTEXT` do config real).
pub const DEPFORMER_DEFAULT_CONTEXT: usize = 8;
/// Codebooks default (Mimi/PersonaPlex).
pub const DEPFORMER_DEFAULT_NCB: usize = 16;
/// Heads default.
pub const DEPFORMER_DEFAULT_NHEADS: usize = 16;
/// Níveis por codebook default (10 bits).
pub const DEPFORMER_DEFAULT_LEVELS: usize = 1024;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum DepError {
    #[error("janela 0 (CONTEXT >= 1)")]
    ZeroWindow,
    #[error("dimensão 0")]
    ZeroDim,
}

/// KV de uma (stream, layer): linhas K/V row-major `[rows, d]`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DepKV {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub d: usize,
}

impl DepKV {
    pub fn new(d: usize) -> Self {
        Self { k: Vec::new(), v: Vec::new(), d }
    }

    /// Linhas acumuladas.
    pub fn rows(&self) -> usize {
        if self.d == 0 {
            0
        } else {
            self.k.len() / self.d
        }
    }

    /// Append de uma linha (K, V) + dreno do excedente da janela.
    /// `window == 0` nunca chega aqui (exec mapeia p/ default).
    pub fn push(&mut self, kv: &[f32], vv: &[f32], window: usize) -> Result<(), DepError> {
        if window == 0 {
            return Err(DepError::ZeroWindow);
        }
        if self.d == 0 {
            return Err(DepError::ZeroDim);
        }
        debug_assert_eq!(kv.len(), self.d);
        debug_assert_eq!(vv.len(), self.d);
        self.k.extend_from_slice(kv);
        self.v.extend_from_slice(vv);
        while self.rows() > window {
            self.k.drain(..self.d);
            self.v.drain(..self.d);
        }
        Ok(())
    }
}

/// Atenção causal per-head sobre a janela (sem RoPE — paridade moshi).
/// `q`: [D], `k_all`/`v_all`: [rows*D] row-major; retorna [D].
/// Ordem das somas documentada (tolerância, não bits, vs ndarray).
pub fn dep_attention(q: &[f32], k_all: &[f32], v_all: &[f32], d: usize, nheads: usize, rows: usize) -> Vec<f32> {
    debug_assert!(d % nheads == 0);
    let hd = d / nheads;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; d];
    for h in 0..nheads {
        // Scores + softmax estável do head.
        let mut max = f32::NEG_INFINITY;
        let mut scores = vec![0.0f32; rows];
        for r in 0..rows {
            let mut s = 0.0f32;
            for i in 0..hd {
                s += q[h * hd + i] * k_all[r * d + h * hd + i];
            }
            s *= scale;
            if s > max {
                max = s;
            }
            scores[r] = s;
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            sum += *s;
        }
        for r in 0..rows {
            let w = scores[r] / sum;
            for i in 0..hd {
                out[h * hd + i] += w * v_all[r * d + h * hd + i];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_caps_and_drains_in_order() {
        let mut kv = DepKV::new(2);
        kv.push(&[1.0, 2.0], &[10.0, 20.0], 8).unwrap();
        kv.push(&[3.0, 4.0], &[30.0, 40.0], 2).unwrap();
        kv.push(&[5.0, 6.0], &[50.0, 60.0], 2).unwrap();
        assert_eq!(kv.rows(), 2);
        // Restam as 2 últimas: [3,4] e [5,6].
        assert_eq!(kv.k, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(kv.v, vec![30.0, 40.0, 50.0, 60.0]);
        assert_eq!(kv.push(&[0.0, 0.0], &[0.0, 0.0], 0), Err(DepError::ZeroWindow));
        let mut z = DepKV::new(0);
        assert_eq!(z.push(&[], &[], 8), Err(DepError::ZeroDim));
    }

    #[test]
    fn attention_single_row_is_passthrough() {
        // 1 linha: softmax([s]) = [1] => out = V, qualquer Q.
        let q = vec![0.3f32, -0.7];
        let k = vec![5.0f32, 6.0];
        let v = vec![7.0f32, 8.0];
        let out = dep_attention(&q, &k, &v, 2, 1, 1);
        assert_eq!(out, vec![7.0, 8.0]);
    }

    #[test]
    fn attention_uniform_keys_mixes_evenly() {
        // Q ortogonal por head: cada head vê seu próprio pico.
        // D=4, 2 heads: q=[1,0,0,1]; K rows=[1,0,0,0],[0,0,0,1]; V=I-ish.
        let q = vec![1.0f32, 0.0, 0.0, 1.0];
        let k = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let v = vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let out = dep_attention(&q, &k, &v, 4, 2, 2);
        // Escala 1/√2: scores [0.7071,0] → softmax ≈ [0.6698,0.3302].
        // Head 0: 0.6698*10+0.3302*50 ≈ 23.2095; head 1 simétrico.
        assert!((out[0] - 23.2095).abs() < 1e-3, "{:?}", out);
        assert!((out[3] - 66.7905).abs() < 1e-3, "{:?}", out);
    }
}
