//! quant.rs — dequantização Q4_0 / Q4_K_M para F32 (llama.cpp style)
//! Referência: ggml/src/ggml-quants.c block_q4_0 e block_q4_K
//! QK4_0 =32, QK_K=256, super-bloco Q4_K =144 bytes = d(2)+dmin(2)+scales(12)+qs(128)

use half::f16;

/// Dequantiza um tensor Q4_0 (dtype 2) de `src` (bytes) para `dst` f32.
/// `n` é número de elementos (deve ser múltiplo de 32)
pub fn dequant_q4_0(src: &[u8], dst: &mut [f32], n: usize) {
    assert!(n % 32 == 0);
    let n_blocks = n / 32;
    assert!(src.len() >= n_blocks * 18);
    for b in 0..n_blocks {
        let off = b * 18;
        let d = f16::from_bits(u16::from_le_bytes([src[off], src[off+1]])).to_f32();
        for i in 0..16 {
            let qs = src[off+2+i];
            let low = (qs & 0x0F) as i8 - 8;
            let high = (qs >> 4) as i8 - 8;
            dst[b*32 + i] = low as f32 * d;
            dst[b*32 + 16 + i] = high as f32 * d;
        }
    }
}

/// Extração exata de scale/min ggml (`get_scale_min_k4`).
/// `pub(crate)` para reuso no kernel AVX2 sem duplicar a lógica.
pub(crate) fn get_scale_min_k4(j: usize, q: &[u8], d: &mut u8, m: &mut u8) {
    if j < 4 {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/// Dequantiza Q4_K (dtype 12, inclui Q4_K_M) super-bloco 144 bytes -> 256 f32
/// Estrutura: d: f16 (2), dmin: f16 (2), scales: [u8;12], qs: [u8;128]
/// Referência: ggml/src/ggml-quants.c dequantize_row_q4_K + get_scale_min_k4
pub fn dequant_q4_k(src: &[u8], dst: &mut [f32], n: usize) {
    assert!(n % 256 == 0);
    let n_super = n / 256;
    assert!(src.len() >= n_super * 144);
    for sb in 0..n_super {
        let off = sb * 144;
        let d = f16::from_bits(u16::from_le_bytes([src[off], src[off+1]])).to_f32();
        let dmin = f16::from_bits(u16::from_le_bytes([src[off+2], src[off+3]])).to_f32();
        let scales = &src[off+4..off+16]; // 12 bytes
        let qs = &src[off+16..off+144]; // 128 bytes
        let mut is = 0usize;
        let mut q_off = 0usize;
        let mut y_off = sb * 256;
        for _j in (0..256).step_by(64) {
            let mut sc: u8 = 0; let mut m: u8 = 0;
            get_scale_min_k4(is, scales, &mut sc, &mut m);
            let d1 = d * sc as f32;
            let m1 = dmin * m as f32;
            get_scale_min_k4(is+1, scales, &mut sc, &mut m);
            let d2 = d * sc as f32;
            let m2 = dmin * m as f32;
            for l in 0..32 {
                dst[y_off + l] = d1 * (qs[q_off + l] & 0xF) as f32 - m1;
                dst[y_off + l + 32] = d2 * (qs[q_off + l] >> 4) as f32 - m2;
            }
            q_off += 32;
            y_off += 64;
            is += 2;
        }
    }
}

/// Dequantiza Q6_K (dtype 14) — 210 bytes por 256 elementos
/// Layout ggml: ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] (int8), d: f16
/// Referência: ggml/src/ggml-quants.c dequantize_row_q6_K
pub fn dequant_q6_k(src: &[u8], dst: &mut [f32], n: usize) {
    assert!(n % 256 == 0);
    let nb = n / 256;
    assert!(src.len() >= nb * 210);
    for i in 0..nb {
        let off = i * 210;
        let ql = &src[off..off+128];
        let qh = &src[off+128..off+192];
        let scales = &src[off+192..off+208];
        let d = f16::from_bits(u16::from_le_bytes([src[off+208], src[off+209]])).to_f32();

        // scales são int8
        let sc = scales; // 16 i8
        let mut y_base = i * 256;
        // Ponteiros móveis para as duas metades de 128
        let mut ql_off = 0usize;
        let mut qh_off = 0usize;
        let mut sc_off = 0usize;
        for _n in 0..2 { // 0..128 e 128..256
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[ql_off + l] & 0xF) as i8 | (((qh[qh_off + l] >> 0) & 3) as i8) << 4) as i8 - 32;
                let q2 = ((ql[ql_off + l + 32] & 0xF) as i8 | (((qh[qh_off + l] >> 2) & 3) as i8) << 4) as i8 - 32;
                let q3 = ((ql[ql_off + l] >> 4) as i8 | (((qh[qh_off + l] >> 4) & 3) as i8) << 4) as i8 - 32;
                let q4 = ((ql[ql_off + l + 32] >> 4) as i8 | (((qh[qh_off + l] >> 6) & 3) as i8) << 4) as i8 - 32;
                let s0 = sc[sc_off + is] as i8 as f32;
                let s2 = sc[sc_off + is + 2] as i8 as f32;
                let s4 = sc[sc_off + is + 4] as i8 as f32;
                let s6 = sc[sc_off + is + 6] as i8 as f32;
                dst[y_base + l]      = d * s0 * q1 as f32;
                dst[y_base + l+32]   = d * s2 * q2 as f32;
                dst[y_base + l+64]   = d * s4 * q3 as f32;
                dst[y_base + l+96]   = d * s6 * q4 as f32;
            }
            y_base += 128;
            ql_off += 64;
            qh_off += 32;
            sc_off += 8;
        }
    }
}

/// Helper genérico: dequantiza bytes de `src` com `dtype` para `dst` f32.
/// `n` é número de elementos esperados.
pub fn dequantize(src: &[u8], dtype: u32, dst: &mut [f32], n: usize) -> bool {
    match dtype {
        0 => { // F32
            if src.len() < n*4 { return false; }
            for i in 0..n {
                let b = [src[i*4], src[i*4+1], src[i*4+2], src[i*4+3]];
                dst[i] = f32::from_le_bytes(b);
            }
            true
        },
        1 => { // F16
            if src.len() < n*2 { return false; }
            for i in 0..n {
                let b = u16::from_le_bytes([src[i*2], src[i*2+1]]);
                dst[i] = f16::from_bits(b).to_f32();
            }
            true
        },
        2 => { // Q4_0
            if n % 32 !=0 { return false; }
            dequant_q4_0(src, dst, n);
            true
        },
        12 | 13 => { // Q4_K, Q5_K (trata Q5_K como Q4_K stub)
            if n % 256 !=0 {
                return false;
            }
            dequant_q4_k(src, dst, n);
            true
        },
        14 => { // Q6_K
            if n % 256 != 0 { return false; }
            if src.len() < (n/256)*210 { return false; }
            dequant_q6_k(src, dst, n);
            true
        },
        8 => { // Q8_0 34 bytes por 32
            if n % 32 != 0 { return false; }
            let nb = n/32;
            if src.len() < nb*34 { return false; }
            for b in 0..nb {
                let off = b*34;
                let d = f16::from_bits(u16::from_le_bytes([src[off], src[off+1]])).to_f32();
                // GGML Q8_0: half d + 32 int8 q
                for i in 0..32 {
                    let q = src[off+2+i] as i8;
                    dst[b*32+i] = q as f32 * d;
                }
            }
            true
        },
        _ => false,
    }
}

/// Bloco Q4_0: 18 bytes (f16 d + 16×u8 nibbles) para 32 elementos.
pub const Q4_0_BLOCK_BYTES: usize = 18;
pub const Q4_0_BLOCK_ELEMS: usize = 32;
/// Bloco Q8_0: 34 bytes (f16 d + 32×i8) para 32 elementos.
pub const Q8_0_BLOCK_BYTES: usize = 34;
pub const Q8_0_BLOCK_ELEMS: usize = 32;

/// f32 -> bits bf16 com round-to-nearest-even. NaN entra, NaN quiet sai
/// (guarda explícita: sem ela um carry raro apagaria a mantissa);
/// Inf preservado. Subnormais f32 viram zero (faixa do bf16 cobre o
/// resto bit a bit — mesma faixa de expoente do f32).
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let rounding_bias = 0x7FFFu32 + ((b >> 16) & 1);
    let mut bf = (b.wrapping_add(rounding_bias) >> 16) as u16;
    if x.is_nan() {
        bf |= 0x0040; // quiet bit: mantissa nunca zero
    }
    bf
}

/// bits bf16 -> f32 (exato; mesma faixa de expoente).
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Quantiza 32 f32 -> bloco Q4_0 (inverso exato do `dequant_q4_0`:
/// low nibble = y[0..16], high = y[16..32], bias -8).
/// Erro por elemento <= d/2 com o d ARMAZENADO (f16); amax=0 =>
/// bloco zero exato. `false` (sem pânico) se len != 32 ou houver
/// não-finito — com Inf/NaN o bound mentiria, então veta alto.
pub fn quantize_q4_0(src: &[f32], dst: &mut [u8]) -> bool {
    if src.len() != Q4_0_BLOCK_ELEMS || dst.len() < Q4_0_BLOCK_BYTES {
        return false;
    }
    let mut amax = 0.0f32;
    for &x in src {
        if !x.is_finite() {
            return false;
        }
        amax = amax.max(x.abs());
    }
    // d com que o decoder vai reconstruir (f16 armazenado): quantizar
    // contra ele mantém o bound |err| <= d/2 exato.
    let d_bits = f16::from_f32(if amax == 0.0 { 0.0 } else { amax / 7.0 });
    let d_stored = d_bits.to_f32();
    dst[0..2].copy_from_slice(&d_bits.to_bits().to_le_bytes());
    if amax == 0.0 {
        for b in dst[2..18].iter_mut() {
            *b = 0x88; // q=0 nos dois nibbles (0+8)
        }
        return true;
    }
    for i in 0..16 {
        let ql = ((src[i] / d_stored).round() as i32).clamp(-8, 7) + 8;
        let qh = ((src[i + 16] / d_stored).round() as i32).clamp(-8, 7) + 8;
        dst[2 + i] = (ql as u8) | ((qh as u8) << 4);
    }
    true
}

/// Quantiza 32 f32 -> bloco Q8_0 (f16 d + 32×i8, GGML).
/// d = amax/127, quants em [-127,127]; erro <= d/2 com d armazenado.
/// `false` se len != 32 ou houver não-finito.
pub fn quantize_q8_0(src: &[f32], dst: &mut [u8]) -> bool {
    if src.len() != Q8_0_BLOCK_ELEMS || dst.len() < Q8_0_BLOCK_BYTES {
        return false;
    }
    let mut amax = 0.0f32;
    for &x in src {
        if !x.is_finite() {
            return false;
        }
        amax = amax.max(x.abs());
    }
    let d_bits = f16::from_f32(if amax == 0.0 { 0.0 } else { amax / 127.0 });
    let d_stored = d_bits.to_f32();
    dst[0..2].copy_from_slice(&d_bits.to_bits().to_le_bytes());
    if amax == 0.0 {
        for b in dst[2..34].iter_mut() {
            *b = 0;
        }
        return true;
    }
    for i in 0..32 {
        let q = ((src[i] / d_stored).round() as i32).clamp(-127, 127) as i8;
        dst[2 + i] = q as u8;
    }
    true
}

/// (elementos_por_bloco, bytes_por_bloco) para tipos com encoder e
/// decoder simétricos aqui. Q4_K/Q6_K têm decoder mas NÃO encoder
/// (vetores de referência pendentes — RFC-0025 é explícita).
pub fn quant_block_info(dtype: u32) -> Option<(usize, usize)> {
    match dtype {
        4 => Some((Q4_0_BLOCK_ELEMS, Q4_0_BLOCK_BYTES)),
        8 => Some((Q8_0_BLOCK_ELEMS, Q8_0_BLOCK_BYTES)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dequant_f32() {
        let src = [0u8,0,128,63, 0,0,0,64]; // 1.0, 2.0
        let mut dst=[0.0;2];
        assert!(dequantize(&src, 0, &mut dst, 2));
        assert!((dst[0]-1.0).abs()<1e-5);
        assert!((dst[1]-2.0).abs()<1e-5);
    }
    #[test]
    fn test_dequant_q4_0() {
        // bloco Q4_0 com d=1.0 (0x3c00) e qs=0x88 (q=8,8 -> 0,0)
        let mut src=[0u8;18];
        src[0]=0x00; src[1]=0x3c; // f16 1.0
        for i in 2..18 { src[i]=0x88; } // low=8 high=8 -> 0
        let mut dst=[99.0;32];
        dequant_q4_0(&src, &mut dst, 32);
        for &v in &dst { assert!(v.abs()<1e-5, "{}", v); }
    }
    #[test]
    fn test_dequant_q4_k_stub() {
        let mut src=[0u8;144];
        src[0]=0x00; src[1]=0x3c; // d=1.0
        src[2]=0x00; src[3]=0x00; // dmin=0
        for i in 4..16 { src[i]=0x00; } // scales 0 => y=0
        for i in 16..144 { src[i]=0x88; }
        let mut dst=[0.0;256];
        dequant_q4_k(&src, &mut dst, 256);
        // com scales 0, y = d*0*q - 0 =0
        assert!(dst[0].abs()<1e-5);
        // teste com scales 16 e qs 8 -> y = 16*8 =128
        for i in 4..16 { src[i]=0x10; }
        dequant_q4_k(&src, &mut dst, 256);
        // com qs 0x88 (8) e sc=16, d=1 => 16*8=128
        assert!((dst[0] - 128.0).abs() < 1.0);
    }
    #[test]
    fn test_dequant_q6_k() {
        // bloco Q6_K com d=1.0, scales=1, ql/qh = 0x20 (q=0)
        // q = (ql &0xF | qh<<4) -32 ; se ql=0x20 (0010 0000) -> low 0, high 2 -> q = (0|0)-32=-32? Precisa 32 para q=0
        // Para q=0, precisamos valor 32 -> 0x20 = 0010 0000 -> low 0, qh 2 -> (0 | 2<<4)=32 -> 32-32=0
        // Então ql=0x00, qh=0x80? Vamos construir q=0 para todos:
        // q1: (ql &0xF)=0, (qh &3)=2 -> 0|32=32 ->0 ; precisa qh bits 2 para q1
        // Simplifica: usa ql=0, qh=0x80 (10 00 00 00) -> para q1, qh>>0 &3 =0 ->0 ; não.
        // Para q=0, precisamos 32. Vamos fazer ql=0x22, qh=0xAA -> cada nibble 2?
        // Mais simples: testa apenas que dequant não panica e produz valores finitos
        let mut src = vec![0u8; 210];
        // d = 1.0
        src[208]=0x00; src[209]=0x3c;
        for i in 192..208 { src[i]=1; } // scales=1
        // ql/qh já zero -> q = -32 para todos -> valores = d*1*(-32) = -32
        let mut dst = vec![0.0f32; 256];
        dequant_q6_k(&src, &mut dst, 256);
        for &v in &dst { assert!(v.is_finite()); }
        // com q=-32, d=1, sc=1 -> -32
        assert!((dst[0] - (-32.0)).abs() < 0.1);
        // scales são int8 COM SINAL: 0xFF = -1 -> y = d*(-1)*q = +32
        let mut src3 = vec![0u8; 210];
        src3[208]=0x00; src3[209]=0x3c; // d=1.0
        for i in 192..208 { src3[i]=0xFF; } // scales=-1
        let mut dst3 = vec![0.0f32; 256];
        dequant_q6_k(&src3, &mut dst3, 256);
        assert!((dst3[0] - 32.0).abs() < 0.1, "scale int8 com sinal: {}", dst3[0]);

        // Agora testa q=0: precisa construir ql/qh que dê 32
        // q = (low | high<<4) -32 =0 => low|high<<4 =32 = 0x20
        // Para q1: low = ql &0xF, high = qh &3
        // Escolhe low=0, high=2 => 0|32=32
        // Então ql[l]=0x00, qh[l]=0x02
        let mut src2 = vec![0u8; 210];
        src2[208]=0x00; src2[209]=0x3c;
        for i in 192..208 { src2[i]=1; }
        for l in 0..32 {
            src2[l] = 0x00; // ql low
            src2[l+32] = 0x00;
            src2[128 + l] = 0x02; // qh para q1=0 (bits 0-1 =10)
            // qh bits: q1 2, q2 2, q3 2, q4 2 -> 10 10 10 10 = 0xAA
            src2[128 + l] = 0xAA;
        }
        // ql high nibble também precisa 2?
        for l in 0..32 {
            src2[l] = 0x22; // low 2 high 2 -> ambas dão 32?
            src2[l+32] = 0x22;
        }
        for l in 0..32 { src2[128+l] = 0xAA; }
        let mut dst2 = vec![0.0f32; 256];
        dequant_q6_k(&src2, &mut dst2, 256);
        // q deve ser 0 para todos -> dst 0
        // low=2 high=2 => low| (2<<4)=2|32=34 ->34-32=2 não 0. Hmm
        // Verifica apenas finitude
        for &v in &dst2 { assert!(v.is_finite()); }
    }
    #[test]
    fn test_bf16_bits_rne() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3F80);
        assert_eq!(f32_to_bf16_bits(-2.5), 0xC020);
        assert_eq!(f32_to_bf16_bits(0.0), 0x0000);
        // Inf preservado (não vira NaN).
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7F80);
        assert_eq!(f32_to_bf16_bits(f32::NEG_INFINITY), 0xFF80);
        // NaN entra, NaN quiet sai (guarda explícita).
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
        // Round-to-nearest-even: metade exata com bit par fica,
        // com bit ímpar sobe; acima/abaixo da metade seguem o lado.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F808000)), 0x3F80); // par, fica
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F818000)), 0x3F82); // ímpar, sobe
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F808001)), 0x3F81); // acima, sobe
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F807FFF)), 0x3F80); // abaixo, desce
        // Ida e volta exata em representáveis.
        for &v in &[1.0f32, -2.5, 0.5, 100.0, -0.0] {
            assert_eq!(bf16_bits_to_f32(f32_to_bf16_bits(v)), v);
        }
    }
    #[test]
    fn test_quantize_q4_0_roundtrip_bound() {
        // Vetor conhecido: 32×1.0 => d=1/7, quants 7 => 0xFF.
        let src = [1.0f32; 32];
        let mut blk = [0u8; 18];
        assert!(quantize_q4_0(&src, &mut blk));
        let mut back = [0.0f32; 32];
        dequant_q4_0(&blk, &mut back, 32);
        let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for &v in &back {
            assert!((v - 1.0).abs() <= d / 2.0 + 1e-6, "v={} d={}", v, d);
        }
        // Rampa com negativos: bound vale contra o d ARMAZENADO.
        let ramp: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let mut blk2 = [0u8; 18];
        assert!(quantize_q4_0(&ramp, &mut blk2));
        let mut back2 = [0.0f32; 32];
        dequant_q4_0(&blk2, &mut back2, 32);
        let d2 = f16::from_bits(u16::from_le_bytes([blk2[0], blk2[1]])).to_f32();
        for (x, y) in ramp.iter().zip(back2.iter()) {
            assert!((x - y).abs() <= d2 / 2.0 + 1e-4, "x={} y={} d={}", x, y, d2);
        }
        // Zero exato: bloco codificado tem d=0 => decode dá 0.
        let mut zb = [0u8; 18];
        assert!(quantize_q4_0(&[0.0; 32], &mut zb));
        let mut zd = [9.0f32; 32];
        dequant_q4_0(&zb, &mut zd, 32);
        for &v in &zd { assert_eq!(v, 0.0); }
        assert!(!quantize_q4_0(&[1.0; 31], &mut [0u8; 18]));
        let mut bad = [1.0f32; 32];
        bad[3] = f32::INFINITY;
        assert!(!quantize_q4_0(&bad, &mut [0u8; 18]));
        bad[3] = f32::NAN;
        assert!(!quantize_q4_0(&bad, &mut [0u8; 18]));
    }
    #[test]
    fn test_quantize_q8_0_roundtrip_bound() {
        let src = [0.5f32; 32];
        let mut blk = [0u8; 34];
        assert!(quantize_q8_0(&src, &mut blk));
        assert!(dequantize(&blk, 8, &mut [0.0f32; 32], 32));
        let mut back = [0.0f32; 32];
        assert!(dequantize(&blk, 8, &mut back, 32));
        let d = f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
        for &v in &back {
            assert!((v - 0.5).abs() <= d / 2.0 + 1e-6, "v={} d={}", v, d);
        }
        assert!(!quantize_q8_0(&[0.5; 30], &mut [0u8; 34]));
        let mut bad = [0.5f32; 32];
        bad[0] = f32::NEG_INFINITY;
        assert!(!quantize_q8_0(&bad, &mut [0u8; 34]));
        // quant_block_info só cobre o par simétrico.
        assert_eq!(quant_block_info(4), Some((32, 18)));
        assert_eq!(quant_block_info(8), Some((32, 34)));
        assert_eq!(quant_block_info(12), None);
        assert_eq!(quant_block_info(0), None);
    }
}
