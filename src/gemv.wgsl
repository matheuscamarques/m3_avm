// gemv.wgsl — GEMV K-first (flat = i + j*in) para benchmark Vega 8.
// Dois kernels: f32 direto e Q4_K fundido (decode exato ggml + dot).

struct Params {
    in_dim: u32,
    out_dim: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read> w_f32: array<f32>;
@group(0) @binding(1) var<storage, read> x_vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_vec: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn gemv_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params.out_dim) { return; }
    let in_dim = params.in_dim;
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < in_dim; i = i + 1u) {
        sum = sum + x_vec[i] * w_f32[i + j * in_dim];
    }
    y_vec[j] = sum;
}

// ---- Q4_K fundido: mesmos bytes do GGUF (144B/bloco, 256 elems) ----
@group(0) @binding(0) var<storage, read> w_u32: array<u32>;
@group(0) @binding(1) var<storage, read> x_q4: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_q4: array<f32>;
@group(0) @binding(3) var<uniform> params_q4: Params;
fn get_u8_at(word_idx: u32, byte_in_word: u32) -> u32 {
    return (w_u32[word_idx] >> (byte_in_word * 8u)) & 0xFFu;
}

// super-bloco b começa no byte b*144 do tensor
fn superblock_base(b: u32) -> u32 {
    return b * 36u; // em u32
}

fn f16_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 15u) & 0x1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    var f: f32;
    if (exp == 0u) {
        f = f32(mant) * 5.960464477539063e-8; // 2^-24 subnormal
    } else if (exp == 31u) {
        f = 3.402823466e+38;
    } else {
        let e = i32(exp) - 15 + 127;
        f = bitcast<f32>((sign << 31u) | (u32(e) << 23u) | (mant << 13u));
    }
    if (sign == 1u && exp != 0u && exp != 31u) {
        f = -f;
    }
    // trata -0/subnormal com sinal de forma aproximada (pesos reais: d>0)
    return f;
}

// byte k dos 12 bytes de scales (words sb+1..sb+3)
fn scale_byte(sb: u32, k: u32) -> u32 {
    return get_u8_at(sb + 1u + k / 4u, k % 4u);
}

// get_scale_min_k4 exato (ggml): j em 0..8, scales = 12 bytes do bloco
fn scale_min(sb: u32, j: u32) -> vec2<u32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        sc = scale_byte(sb, j) & 63u;
        mn = scale_byte(sb, j + 4u) & 63u;
    } else {
        let a = scale_byte(sb, j + 4u);
        let b0 = scale_byte(sb, j - 4u);
        let c = scale_byte(sb, j);
        sc = (a & 0xFu) | ((b0 >> 6u) << 4u);
        mn = (a >> 4u) | ((c >> 6u) << 4u);
    }
    return vec2<u32>(sc, mn);
}

@compute @workgroup_size(64)
fn gemv_q4k(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params_q4.out_dim) { return; }
    let in_dim = params_q4.in_dim;
    // nº de blocos por coluna (256 | in_dim em todos os shapes LLM aqui)
    let blocks_per_col = in_dim / 256u;
    let col_base_elem = j * in_dim; // flat do elemento (0, j)
    let first_block = col_base_elem / 256u;
    var acc: f32 = 0.0;
    for (var bb: u32 = 0u; bb < blocks_per_col; bb = bb + 1u) {
        let b = first_block + bb;
        let sb = superblock_base(b); // u32 idx base do bloco
        let d = f16_to_f32(get_u8_at(sb, 0u) | (get_u8_at(sb, 0u + 1u) << 8u));
        // byte layout: d[0..2], dmin[2..4], scales[4..16], qs[16..144]
        let dmin = f16_to_f32(get_u8_at(sb, 2u) | (get_u8_at(sb, 3u) << 8u));
        // 4 chunks de 32B -> subs 2c/2c+1, 32 elems cada; x local em i0..i0+256
        let i0 = bb * 256u;
        for (var c: u32 = 0u; c < 4u; c = c + 1u) {
            let sm0 = scale_min(sb, 2u * c);
            let sm1 = scale_min(sb, 2u * c + 1u);
            let d0 = d * f32(sm0.x);
            let m0 = dmin * f32(sm0.y);
            let d1 = d * f32(sm1.x);
            let m1 = dmin * f32(sm1.y);
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                // q byte l do chunk (32B): low nibble -> sub 2c, high -> sub 2c+1
                let qbyte = get_u8_at(sb + 4u + c * 8u + l / 4u, l % 4u);
                let qlow = f32(qbyte & 0xFu);
                let qhigh = f32((qbyte >> 4u) & 0xFu);
                let e0 = i0 + c * 64u + l;
                let e1 = i0 + c * 64u + 32u + l;
                acc = acc + (qlow * d0 - m0) * x_q4[e0];
                acc = acc + (qhigh * d1 - m1) * x_q4[e1];
            }
        }
    }
    y_q4[j] = acc;
}

// ---- Q6_K fundido: 210B/bloco (ql[128], qh[64], scales[16] int8, d f16) ----
@group(0) @binding(0) var<storage, read> w6_u32: array<u32>;
@group(0) @binding(1) var<storage, read> x_q6: array<f32>;
@group(0) @binding(2) var<storage, read_write> y_q6: array<f32>;
@group(0) @binding(3) var<uniform> params_q6: Params;

// byte absoluto k do tensor -> valor
fn raw6_byte(base: u32, k: u32) -> u32 {
    let abs = base + k;
    return (w6_u32[abs / 4u] >> ((abs % 4u) * 8u)) & 0xFFu;
}

fn f16_6(bits: u32) -> f32 {
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;
    if (exp == 0u) { return 0.0; }
    if (exp == 31u) { return 3.402823466e+38; }
    let e = i32(exp) - 15 + 127;
    return bitcast<f32>((u32(e) << 23u) | (mant << 13u));
}

fn i8_6(v: u32) -> f32 {
    // scales int8 com sinal
    if (v >= 128u) {
        return f32(i32(v) - 256);
    }
    return f32(v);
}

@compute @workgroup_size(64)
fn gemv_q6k(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= params_q6.out_dim) { return; }
    let in_dim = params_q6.in_dim;
    let blocks_per_col = in_dim / 256u;
    let first_block = (j * in_dim) / 256u;
    let i0col = (j * in_dim) % 256u; // == 0 quando 256 | in_dim
    var acc: f32 = 0.0;
    for (var bb: u32 = 0u; bb < blocks_per_col; bb = bb + 1u) {
        let base = (first_block + bb) * 210u; // byte base do bloco
        let d = f16_6(raw6_byte(base, 208u) | (raw6_byte(base, 209u) << 8u));
        let i0 = i0col + bb * 256u;
        // duas metades de 128
        for (var nn: u32 = 0u; nn < 2u; nn = nn + 1u) {
            let ql0 = base + nn * 64u;   // ql da metade (64B usados: l e l+32)
            let qh0 = base + 128u + nn * 32u;
            let sc0 = base + 192u + nn * 8u;
            for (var l: u32 = 0u; l < 32u; l = l + 1u) {
                let is = l / 16u;
                let ql_a = raw6_byte(0u, ql0 + l);
                let ql_b = raw6_byte(0u, ql0 + l + 32u);
                let qh = raw6_byte(0u, qh0 + l);
                let q1 = f32(i32((ql_a & 0xFu) | (((qh >> 0u) & 3u) << 4u)) - 32);
                let q2 = f32(i32((ql_b & 0xFu) | (((qh >> 2u) & 3u) << 4u)) - 32);
                let q3 = f32(i32((ql_a >> 4u) | (((qh >> 4u) & 3u) << 4u)) - 32);
                let q4 = f32(i32((ql_b >> 4u) | (((qh >> 6u) & 3u) << 4u)) - 32);
                let s0 = i8_6(raw6_byte(0u, sc0 + is));
                let s2 = i8_6(raw6_byte(0u, sc0 + is + 2u));
                let s4 = i8_6(raw6_byte(0u, sc0 + is + 4u));
                let s6 = i8_6(raw6_byte(0u, sc0 + is + 6u));
                let base_e = i0 + nn * 128u;
                acc = acc + d * s0 * q1 * x_q6[base_e + l];
                acc = acc + d * s2 * q2 * x_q6[base_e + l + 32u];
                acc = acc + d * s4 * q3 * x_q6[base_e + l + 64u];
                acc = acc + d * s6 * q4 * x_q6[base_e + l + 96u];
            }
        }
    }
    y_q6[j] = acc;
}
