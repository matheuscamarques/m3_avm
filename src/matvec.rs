//! matvec.rs — matvec para inferência LLM (Fase 3.1)
//!
//! Layout GGUF: `w` tem K=`in_dim` contíguo — `y[j] = Σ_i x[i]·w[i + j*in_dim]`
//! (convenção llama.cpp: dims `[K, N]`, K fastest). `x` tem `in_dim`
//! elementos, saída tem `out_dim`.
//!
//! - `matvec_ndarray`: baseline (ndarray + matrixmultiply SIMD).
//! - `matvec_faer`: SIMD via `faer`.
//! - `matvec`: dispatcher padrão (faer, com fallback ndarray em shape mismatch).

use faer::{col as fcol, mat as fmat};
use ndarray::ShapeBuilder;

/// Fallback para shape mismatch (mesma semântica do baseline em inference.rs).
fn fallback_mismatch(x: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; out_dim];
    for i in 0..out_dim.min(x.len()) {
        out[i] = x[i % x.len()] * 0.5;
    }
    out
}

/// Baseline: `ndarray` + matrixmultiply SIMD, convenção GGUF (K fastest).
pub fn matvec_ndarray(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    if w.len() != in_dim * out_dim {
        return fallback_mismatch(x, out_dim);
    }
    use ndarray::{ArrayView1, ArrayView2};
    let x_view = ArrayView1::from(x);
    // w[i + j*in] = F-order (in, out); .t() dá (out, in) p/ dot
    let w_view = ArrayView2::from_shape((in_dim, out_dim).f(), w).expect("shape validado acima");
    w_view.t().dot(&x_view).to_vec()
}

/// Otimizado: `faer` com SIMD. `w` convenção GGUF: `MatRef` column-major
/// `[in_dim, out_dim]` × vetor-coluna `x` → coluna resultado.
pub fn matvec_faer(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    if w.len() != in_dim * out_dim || x.len() != in_dim {
        return fallback_mismatch(x, out_dim);
    }
    let x_col = fcol::from_slice(x);
    let w_mat = fmat::from_column_major_slice::<f32>(w, in_dim, out_dim);
    // y = Wᵀ·x  (W é in×out, y é out)
    let out = &w_mat.transpose() * &x_col;
    out.as_slice().to_vec()
}

/// Dispatcher padrão: faer (rápido), com mesma semântica de fallback do baseline.
pub fn matvec(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    matvec_faer(x, w, in_dim, out_dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(in_dim: usize, out_dim: usize) -> (Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32 * 0.013).sin() * 0.5)).collect();
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| ((i as f32 * 0.002).sin() * 0.3))
            .collect();
        (x, w)
    }

    #[test]
    fn test_faer_matches_ndarray_small() {
        let (x, w) = sample(32, 64);
        let a = matvec_ndarray(&x, &w, 32, 64);
        let b = matvec_faer(&x, &w, 32, 64);
        assert_eq!(a.len(), b.len());
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((va - vb).abs() < 1e-4, "idx {}: {} vs {}", i, va, vb);
        }
    }

    #[test]
    fn test_faer_matches_ndarray_llm_shape() {
        // Shape típico Qwen2/DeepSeek down-proj parcial: 512 -> 256 (rápido em debug)
        let (x, w) = sample(512, 256);
        let a = matvec_ndarray(&x, &w, 512, 256);
        let b = matvec_faer(&x, &w, 512, 256);
        assert_eq!(a.len(), 256);
        let max_diff = a
            .iter()
            .zip(b.iter())
            .map(|(va, vb)| (va - vb).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "max_diff {}", max_diff);
    }

    #[test]
    fn test_mismatch_fallback_parity() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![0.5; 7]; // não é in_dim*out_dim
        let a = matvec_ndarray(&x, &w, 4, 5);
        let b = matvec_faer(&x, &w, 4, 5);
        assert_eq!(a, b);
        assert_eq!(a.len(), 5);
    }

    #[test]
    fn test_gguf_orientation_nonsquare() {
        // Convenção GGUF: w[i + j*in]. W[i,j] = i*10+j, x = ones →
        // y[j] = Σ_i (i*10+j) = 10*(0+1+2) + 3*j = 30 + 3j.
        let (in_dim, out_dim) = (3usize, 4usize);
        let mut w = vec![0.0f32; in_dim * out_dim];
        for j in 0..out_dim {
            for i in 0..in_dim {
                w[i + j * in_dim] = (i * 10 + j) as f32;
            }
        }
        let x = vec![1.0f32; in_dim];
        let expect: Vec<f32> = (0..out_dim).map(|j| 30.0 + 3.0 * j as f32).collect();
        for (name, y) in [("ndarray", matvec_ndarray(&x, &w, in_dim, out_dim)), ("faer", matvec_faer(&x, &w, in_dim, out_dim))] {
            assert_eq!(y.len(), out_dim, "{}", name);
            for (j, (&got, &want)) in y.iter().zip(expect.iter()).enumerate() {
                assert!((got - want).abs() < 1e-4, "{} j{}: {} vs {}", name, j, got, want);
            }
        }
    }
}
