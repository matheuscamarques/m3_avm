//! matvec_quant.rs — Kernel int4 direto, Fase 3.2
//!
//! Motivação: `RealInference` hoje faz `dequantize` (197 tensores → ~400MB
//! f32) + `matvec` em 2 passes. Este módulo funde os 2 passes em 1:
//! decodifica 1 super-bloco por vez num buffer de pilha ([f32; 256]) e
//! acumula direto em `y`, sem nunca materializar o vetor dequantizado.
//! Tráfego de memória cai ~2× e some o `Arc→Vec` clone por token.
//!
//! Propositalmente **isolado**: não é usado por `src/inference.rs` ainda
//! (outra IA está com esse arquivo). Plug futuro, 1 linha por call-site:
//!
//! ```ignore
//! // em inference.rs, onde hoje há read+dequant+matvec:
//! if let Some(y) = crate::matvec_quant::matvec_quant(&x, &raw, dtype, in_dim, out_dim) {
//!     y
//! } else {
//!     crate::matvec::matvec(&x, &dequant_vec, in_dim, out_dim) // fallback
//! }
//! ```
//!
//! Layout: `w` row-major `[in_dim, out_dim]`; `raw` são os bytes GGUF do
//! tensor (super-blocos consecutivos sobre o índice flat
//! `k = i * out_dim + j`). Bit-paridade com `crate::quant`: a decodificação
//! reaproveita `dequant_q*_k` por bloco, então o único delta numérico vs
//! dequant+dot é a ordem de soma (tolerância 1e-3 nos testes).
//!
//! Q4_K tem ainda caminho AVX2 (`x86_64`): unpack de nibbles + AXPY de 8-wide
//! com a **mesma ordem de operações** do escalar (mul, add, mul, add — sem
//! FMA), logo paridade bit a bit por elemento; só a redução entre threads
//! varia (como já variava com rayon).

use rayon::prelude::*;

use crate::quant::{dequant_q4_0, dequant_q4_k, dequant_q6_k};

/// AVX2 presente? (cacheado; Ryzen 3500U tem)
fn cpu_has_avx2() -> bool {
    static AVX2: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVX2.get_or_init(|| {
        #[cfg(target_arch = "x86_64")] {
            std::arch::is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))] {
            false
        }
    })
}

/// Bytes GGUF por `n` elementos, espelhando `quant::dequantize` + `inference.rs`.
/// `None` = dtype sem kernel direto (caller usa fallback dequant+matvec).
pub fn quant_raw_len(dtype: u32, n: usize) -> Option<usize> {
    match dtype {
        2 => {
            // Q4_0: 18 bytes por 32
            if n % 32 != 0 {
                return None;
            }
            Some((n / 32) * 18)
        }
        12 | 13 => {
            // Q4_K (Q5_K tratado como Q4_K, igual ao resto do código)
            if n % 256 != 0 {
                return None;
            }
            Some((n / 256) * 144)
        }
        14 => {
            // Q6_K: 210 bytes por 256
            if n % 256 != 0 {
                return None;
            }
            Some((n / 256) * 210)
        }
        8 => {
            // Q8_0: 34 bytes por 32
            if n % 32 != 0 {
                return None;
            }
            Some((n / 32) * 34)
        }
        _ => None,
    }
}

/// Dispatcher: `Some(y)` se dtype tem kernel direto e shapes batem,
/// `None` para fallback (F32/F16/dtypes não suportados).
pub fn matvec_quant(
    x: &[f32],
    raw: &[u8],
    dtype: u32,
    in_dim: usize,
    out_dim: usize,
) -> Option<Vec<f32>> {
    if x.len() != in_dim || in_dim == 0 || out_dim == 0 {
        return None;
    }
    let n = in_dim.checked_mul(out_dim)?;
    let need = quant_raw_len(dtype, n)?;
    if raw.len() < need {
        return None;
    }
    match dtype {
        12 | 13 => Some(matvec_q4k(x, &raw[..need], in_dim, out_dim)),
        14 => Some(matvec_q6k(x, &raw[..need], in_dim, out_dim)),
        2 => Some(matvec_q4_0(x, &raw[..need], in_dim, out_dim)),
        8 => Some(matvec_q8_0(x, &raw[..need], in_dim, out_dim)),
        _ => None,
    }
}

/// Núcleo genérico, convenção GGUF (`flat = i + j*in_dim`).
/// Decodifica 1 bloco por vez e acumula dots por coluna: `y[j] += x[i..]·buf`.
/// `block_elems`: elementos por bloco; `block_bytes`: bytes por bloco;
/// P1.2: abaixo deste nº de elementos, rayon custa mais que ajuda.
/// Tunável via env M3_PAR_MIN (default 1M: k/v 393k vão serial, q/gate/head paralelo).
fn par_min_elems() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("M3_PAR_MIN").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1_048_576)
    })
}

/// Variante serial do núcleo (ops pequenas): mesmo runs-loop, sem rayon.
fn matvec_blocked_serial(
    x: &[f32],
    raw: &[u8],
    in_dim: usize,
    out_dim: usize,
    block_elems: usize,
    block_bytes: usize,
    decode: impl Fn(&[u8], &mut [f32]),
) -> Vec<f32> {
    let n_blocks = (in_dim * out_dim) / block_elems;
    let mut y = vec![0.0f32; out_dim];
    let mut buf = vec![0.0f32; block_elems];
    for b in 0..n_blocks {
        let blk = &raw[b * block_bytes..(b + 1) * block_bytes];
        decode(blk, &mut buf);
        let base = b * block_elems;
        let mut t = 0;
        let mut j = base / in_dim;
        let mut i = base % in_dim;
        while t < block_elems {
            let run = (in_dim - i).min(block_elems - t);
            let mut s = 0.0f32;
            for (a, &vv) in x[i..i + run].iter().zip(buf[t..t + run].iter()) {
                s += a * vv;
            }
            y[j] += s;
            t += run;
            i += run;
            if i == in_dim {
                i = 0;
                j += 1;
            }
        }
    }
    y
}

/// Núcleo genérico, convenção GGUF (`flat = i + j*in_dim`).
/// Decodifica 1 bloco por vez e acumula dots por coluna: `y[j] += x[i..]·buf`.
/// `block_elems`: elementos por bloco; `block_bytes`: bytes por bloco;
/// `decode`: decodifica `&raw[b*block_bytes..]` em `buf` (len = block_elems).
fn matvec_blocked(
    x: &[f32],
    raw: &[u8],
    in_dim: usize,
    out_dim: usize,
    block_elems: usize,
    block_bytes: usize,
    decode: impl Fn(&[u8], &mut [f32]) + Sync,
) -> Vec<f32> {
    let n_blocks = (in_dim * out_dim) / block_elems;
    // Paralelo por super-bloco (rayon): cada thread acumula num y privado
    // + buf reutilizado, reduce soma no final. Sem races, sem mutex.
    (0..n_blocks)
        .into_par_iter()
        .fold(
            || (vec![0.0f32; out_dim], vec![0.0f32; block_elems]),
            |(mut acc, mut buf), b| {
                let blk = &raw[b * block_bytes..(b + 1) * block_bytes];
                decode(blk, &mut buf);
                // Runs contíguos por coluna: 1 div/mod por bloco
                let base = b * block_elems;
                let mut t = 0;
                let mut j = base / in_dim;
                let mut i = base % in_dim;
                while t < block_elems {
                    let run = (in_dim - i).min(block_elems - t);
                    let mut s = 0.0f32;
                    for (a, &vv) in x[i..i + run].iter().zip(buf[t..t + run].iter()) {
                        s += a * vv;
                    }
                    acc[j] += s;
                    t += run;
                    i += run;
                    if i == in_dim {
                        i = 0;
                        j += 1;
                    }
                }
                (acc, buf)
            },
        )
        .map(|(acc, _)| acc)
        .reduce(
            || vec![0.0f32; out_dim],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Q4_K/Q5_K direto: 144 bytes → 256 f32 por super-bloco.
/// Auto-seleciona AVX2 quando `out_dim % 256 == 0` (todo bloco cabe numa
/// linha, caso de todos os shapes LLM aqui: 1536, 8960, 256).
pub fn matvec_q4k(x: &[f32], raw: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    matvec_q4k_impl(x, raw, in_dim, out_dim, cpu_has_avx2())
}

/// Seletor explícito escalar vs AVX2 (para benchmarks A/B; produção usa
/// [`matvec_q4k`] com auto-detecção). AVX2 exige `in_dim % 256 == 0`
/// (todo super-bloco numa coluna — caso de todos os shapes LLM: 1536, 8960).
pub fn matvec_q4k_impl(x: &[f32], raw: &[u8], in_dim: usize, out_dim: usize, use_avx2: bool) -> Vec<f32> {
    // P1.2: ops pequenas vão serial (rayon não se paga); resto como antes.
    if in_dim * out_dim < par_min_elems() {
        return matvec_blocked_serial(x, raw, in_dim, out_dim, 256, 144, |blk, buf| {
            dequant_q4_k(blk, buf, 256);
        });
    }
    #[cfg(target_arch = "x86_64")]
    if use_avx2 && in_dim % 256 == 0 && (in_dim * out_dim) % 256 == 0 {
        return matvec_q4k_avx2(x, raw, in_dim, out_dim);
    }
    let _ = use_avx2;
    matvec_blocked(x, raw, in_dim, out_dim, 256, 144, |blk, buf| {
        dequant_q4_k(blk, buf, 256);
    })
}

/// Q4_K paralelo com dot AVX2 por super-bloco (só `x86_64`, só `in_dim % 256
/// == 0` — verificado pelo chamador). Semântica ggml exata: `y = q*d - m`
/// com `get_scale_min_k4`, agrupamento em pares de 32B.
#[cfg(target_arch = "x86_64")]
fn matvec_q4k_avx2(x: &[f32], raw: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    let n_blocks = (in_dim * out_dim) / 256;
    (0..n_blocks)
        .into_par_iter()
        .fold(
            || vec![0.0f32; out_dim],
            |mut acc, b| {
                let off = b * 144;
                let d = half::f16::from_bits(u16::from_le_bytes([raw[off], raw[off + 1]])).to_f32();
                let dmin = half::f16::from_bits(u16::from_le_bytes([raw[off + 2], raw[off + 3]])).to_f32();
                let base = b * 256;
                let j0 = base / in_dim;
                let i0 = base % in_dim;
                let partial = unsafe {
                    dot_q4k_superblock(
                        &raw[off + 16..off + 144],
                        &raw[off + 4..off + 16],
                        d,
                        dmin,
                        &x[i0..i0 + 256],
                    )
                };
                acc[j0] += partial;
                acc
            },
        )
        .map(|acc| acc)
        .reduce(
            || vec![0.0f32; out_dim],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

/// Dot de um super-bloco Q4_K (256 elems, 1 coluna) com segmento de `x`.
/// Agrupamento ggml exato: chunk c (32B) -> sub 2c (low nibbles, 32 elems)
/// e sub 2c+1 (high nibbles, 32 elems). Mesma ordem de ops do escalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_q4k_superblock(
    qs: &[u8],
    scales: &[u8],
    d: f32,
    dmin: f32,
    xseg: &[f32],
) -> f32 {
    use std::arch::x86_64::*;
    debug_assert!(qs.len() == 128 && xseg.len() == 256);
    let nib_mask = _mm_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_ps();
    // 4 chunks de 32B; cada um cobre xseg[64c..64c+64]
    for c in 0..4 {
        let mut sc0 = 0u8;
        let mut m0 = 0u8;
        crate::quant::get_scale_min_k4(2 * c, scales, &mut sc0, &mut m0);
        let mut sc1 = 0u8;
        let mut m1 = 0u8;
        crate::quant::get_scale_min_k4(2 * c + 1, scales, &mut sc1, &mut m1);
        let d0 = _mm256_set1_ps(d * sc0 as f32);
        let mv0 = _mm256_set1_ps(dmin * m0 as f32);
        let d1 = _mm256_set1_ps(d * sc1 as f32);
        let mv1 = _mm256_set1_ps(dmin * m1 as f32);
        let qp = qs.as_ptr().add(c * 32);
        let q0 = _mm_loadu_si128(qp as *const __m128i);
        let q1 = _mm_loadu_si128(qp.add(16) as *const __m128i);
        // low nibbles dos 32B -> sub 2c (x+0..32); high -> sub 2c+1 (x+32..64)
        let lo = [_mm_and_si128(q0, nib_mask), _mm_and_si128(q1, nib_mask)];
        let hi = [
            _mm_and_si128(_mm_srli_epi16(q0, 4), nib_mask),
            _mm_and_si128(_mm_srli_epi16(q1, 4), nib_mask),
        ];
        let xb = xseg.as_ptr().add(c * 64);
        // 16 nibbles por reg, em metades de 8
        let jobs = [
            (lo[0], d0, mv0, 0usize),
            (lo[0], d0, mv0, 8usize),
            (lo[1], d0, mv0, 16usize),
            (lo[1], d0, mv0, 24usize),
            (hi[0], d1, mv1, 32usize),
            (hi[0], d1, mv1, 40usize),
            (hi[1], d1, mv1, 48usize),
            (hi[1], d1, mv1, 56usize),
        ];
        for (nib, dv, mv, ko) in jobs {
            // ko<16 (dentro do reg de 16): bytes 0..8 ou 8..16
            let part = if ko % 16 == 0 { nib } else { _mm_srli_si128(nib, 8) };
            let qi = _mm256_cvtepi8_epi32(part);
            // q sem viés (0..15), igual ao escalar: v = q*d - m
            // (nota: cvtepi8 interpreta bit7 como sinal; nibbles são 0..15,
            //  sempre positivos — conversão exata)
            let qf = _mm256_cvtepi32_ps(qi);
            let v = _mm256_sub_ps(_mm256_mul_ps(qf, dv), mv);
            let xv = _mm256_loadu_ps(xb.add(ko));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(v, xv));
        }
    }
    // hsum
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let s4 = _mm_add_ps(lo128, hi128);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x1));
    _mm_cvtss_f32(s1)
}

/// Q6_K direto: 210 bytes → 256 f32 por super-bloco.

/// Q6_K direto: 210 bytes → 256 f32 por super-bloco.
pub fn matvec_q6k(x: &[f32], raw: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    matvec_blocked(x, raw, in_dim, out_dim, 256, 210, |blk, buf| {
        dequant_q6_k(blk, buf, 256);
    })
}

/// Q4_0 direto: 18 bytes → 32 f32 por bloco.
pub fn matvec_q4_0(x: &[f32], raw: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    matvec_blocked(x, raw, in_dim, out_dim, 32, 18, |blk, buf| {
        dequant_q4_0(blk, buf, 32);
    })
}

/// Q8_0 direto: `d: f16` + 32×i8 por bloco de 34 bytes (decode inline, trivial).
pub fn matvec_q8_0(x: &[f32], raw: &[u8], in_dim: usize, out_dim: usize) -> Vec<f32> {
    matvec_blocked(x, raw, in_dim, out_dim, 32, 34, |blk, buf| {
        let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for i in 0..32 {
            buf[i] = blk[2 + i] as i8 as f32 * d;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::dequantize;

    /// Monta tensor Q4_K fake onde todo elemento decodifica ≈ 1.0:
    /// d=1.0, dmin=0, scales que decodificam sc=1,m=0, qs=0x11 (nibble 1 → 1*1-0=1)
    fn raw_q4k_ones(n: usize) -> Vec<u8> {
        assert!(n % 256 == 0);
        let mut raw = vec![0u8; (n / 256) * 144];
        for b in 0..n / 256 {
            let off = b * 144;
            raw[off] = 0x00;
            raw[off + 1] = 0x3c; // f16 1.0
            raw[off + 2] = 0x00;
            raw[off + 3] = 0x00; // dmin 0
            // scales que dão sc=1,m=0 para todos os 8 sub-blocos
            raw[off + 4] = 1; raw[off + 5] = 1; raw[off + 6] = 1; raw[off + 7] = 1;
            raw[off + 8] = 0; raw[off + 9] = 0; raw[off + 10] = 0; raw[off + 11] = 0;
            raw[off + 12] = 1; raw[off + 13] = 1; raw[off + 14] = 1; raw[off + 15] = 1;
            for i in 16..144 {
                raw[off + i] = 0x11; // low 1 high 1 => q=1
            }
        }
        raw
    }

    #[test]
    fn test_q4k_ones_matvec() {
        // W ones 16x16 → y[j] = sum(x) para todo j
        let in_dim = 16;
        let out_dim = 16;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let expect: f32 = x.iter().sum();
        let raw = raw_q4k_ones(in_dim * out_dim);
        let y = matvec_q4k(&x, &raw, in_dim, out_dim);
        assert_eq!(y.len(), 16);
        for (j, &v) in y.iter().enumerate() {
            assert!((v - expect).abs() < 1e-3, "j {}: {} vs {}", j, v, expect);
        }
    }

    #[test]
    fn test_quant_matches_dequant_dot_q4k() {
        // Paridade kernel vs caminho atual (dequant full + dot), pesos pseudo-aleatórios
        let in_dim = 32;
        let out_dim = 16; // 512 elems = 2 super-blocos
        let n = in_dim * out_dim;
        let mut raw = vec![0u8; (n / 256) * 144];
        // d=1.0, dmin=0, scales variados, qs variados determinísticos
        for b in 0..n / 256 {
            let off = b * 144;
            raw[off] = 0x00;
            raw[off + 1] = 0x3c;
            raw[off + 2] = 0x00;
            raw[off + 3] = 0x00;
            for i in 0..12 {
                raw[off + 4 + i] = 0x10u8.wrapping_add((b as u8).wrapping_add(i as u8) % 0x20);
            }
            for i in 0..128 {
                raw[off + 16 + i] = (b as u8).wrapping_mul(37).wrapping_add((i as u8).wrapping_mul(11));
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32 * 0.13).sin() * 0.7)).collect();

        let y_kernel = matvec_q4k(&x, &raw, in_dim, out_dim);

        // Referência: dequant full + dot (caminho atual)
        let mut w = vec![0.0f32; n];
        assert!(dequantize(&raw, 12, &mut w, n));
        let y_ref = crate::matvec::matvec_ndarray(&x, &w, in_dim, out_dim);

        assert_eq!(y_kernel.len(), y_ref.len());
        let max_diff = y_kernel
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-2, "max_diff {}", max_diff);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_matches_scalar_q4k() {
        if !super::cpu_has_avx2() {
            eprintln!("skip sem AVX2");
            return;
        }
        // Shape alinhada p/ AVX2 (in % 256 == 0): 256x16 (4096 elems = 16 blocos)
        let in_dim = 256;
        let out_dim = 16;
        let n = in_dim * out_dim;
        let mut raw = vec![0u8; (n / 256) * 144];
        for b in 0..n / 256 {
            let off = b * 144;
            raw[off] = 0x00;
            raw[off + 1] = 0x3c;
            raw[off + 2] = 0x00;
            raw[off + 3] = 0xbc; // dmin negativo p/ exercício
            for i in 0..12 {
                raw[off + 4 + i] = (0x10u8).wrapping_add((b as u8).wrapping_add(i as u8));
            }
            for i in 0..128 {
                raw[off + 16 + i] = (b as u8).wrapping_mul(53).wrapping_add((i as u8).wrapping_mul(29));
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32 * 0.31).sin() * 1.7)).collect();
        // Compara direto paralelo-escalar vs AVX2 (fora do gate de tamanho)
        let a = super::matvec_blocked(&x, &raw, in_dim, out_dim, 256, 144, |blk, buf| {
            crate::quant::dequant_q4_k(blk, buf, 256);
        });
        let b = super::matvec_q4k_avx2(&x, &raw, in_dim, out_dim);
        assert_eq!(a.len(), b.len());
        let max_ref = a.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let max_diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!("avx2 vs escalar: max_ref {:.4} max_diff {:.6}", max_ref, max_diff);
        assert!(max_ref > 1e-6);
        assert!(max_diff / max_ref < 1e-4, "divergência {}", max_diff / max_ref);
    }
    #[test]
    fn test_serial_matches_parallel_blocked() {
        // P1.2: mesmo runs-loop, sem rayon — paridade exata esperada
        let in_dim = 48;
        let out_dim = 32; // 1536 elems; blocos de 256 cruzam linhas
        let n = in_dim * out_dim;
        assert_eq!(n % 256, 0);
        let mut raw = vec![0u8; (n / 256) * 144];
        for b in 0..n / 256 {
            let off = b * 144;
            raw[off] = 0x00;
            raw[off + 1] = 0x3c;
            raw[off + 2] = 0x00;
            raw[off + 3] = 0x00;
            for i in 0..12 {
                raw[off + 4 + i] = (0x10u8).wrapping_add((b as u8).wrapping_add(i as u8));
            }
            for i in 0..128 {
                raw[off + 16 + i] = (b as u8).wrapping_mul(53).wrapping_add((i as u8).wrapping_mul(29));
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32 * 0.11).sin() * 0.9)).collect();
        let decode = |blk: &[u8], buf: &mut [f32]| crate::quant::dequant_q4_k(blk, buf, 256);
        let a = super::matvec_blocked(&x, &raw, in_dim, out_dim, 256, 144, decode);
        let b = super::matvec_blocked_serial(&x, &raw, in_dim, out_dim, 256, 144, decode);
        assert_eq!(a.len(), b.len());
        for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
            assert!((va - vb).abs() < 1e-6, "idx {}: {} vs {}", i, va, vb);
        }
    }
    #[test]
    fn test_dispatcher_fallbacks() {
        let x = vec![1.0f32; 8];
        // dtype desconhecido → None
        assert!(matvec_quant(&x, &vec![0u8; 64], 0, 8, 2).is_none());
        // raw curto → None
        assert!(matvec_quant(&x, &vec![0u8; 10], 12, 32, 8).is_none());
        // n não múltiplo de 256 p/ Q4_K → None
        assert!(matvec_quant(&x, &vec![0u8; 144], 12, 7, 5).is_none());
        // x com len errado → None
        let raw = raw_q4k_ones(256);
        assert!(matvec_quant(&vec![1.0; 4], &raw, 12, 16, 16).is_none());
        // Q4_K válido → Some, paridade com direto
        let x16 = vec![0.5f32; 16];
        let a = matvec_quant(&x16, &raw, 12, 16, 16).expect("Q4_K suportado");
        let b = matvec_q4k(&x16, &raw, 16, 16);
        assert_eq!(a.len(), b.len());
        for (va, vb) in a.iter().zip(b.iter()) {
            assert!((va - vb).abs() < 1e-6);
        }
    }

    #[test]
    fn test_q8_0_matches_dequant() {
        // Blocos Q8_0 com d=1.0 e q = i-16 → valores conhecidos
        let in_dim = 8;
        let out_dim = 8; // 64 elems = 2 blocos de 32
        let n = in_dim * out_dim;
        let mut raw = vec![0u8; (n / 32) * 34];
        for b in 0..n / 32 {
            let off = b * 34;
            raw[off] = 0x00;
            raw[off + 1] = 0x3c; // d=1.0
            for i in 0..32 {
                raw[off + 2 + i] = (i as i8).wrapping_sub(16) as u8;
            }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.3 + 1.0)).collect();
        let y_kernel = matvec_q8_0(&x, &raw, in_dim, out_dim);
        let mut w = vec![0.0f32; n];
        assert!(dequantize(&raw, 8, &mut w, n));
        let y_ref = crate::matvec::matvec_ndarray(&x, &w, in_dim, out_dim);
        let max_diff = y_kernel
            .iter()
            .zip(y_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "max_diff {}", max_diff);
    }
}
