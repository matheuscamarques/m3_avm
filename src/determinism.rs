//! Determinismo da M³-AVM (RFC-0005): RNG por contexto + hashing.
//!
//! - RNG: splitmix64 por contexto (`Context.rng_state`), semente fixa no
//!   boot frio => streams determinísticas. FORK herda (re-seed p/ divergir).
//! - Uniforme em [a,b): 24 bits superiores => [0,1), mapeamento afim.
//! - Normal: Box-Muller sem laço de rejeição (u1 grampeado em MIN_POSITIVE
//!   para evitar ln(0); determinístico sempre).
//! - HASH = FNV-1a/64 (downgrade honesto do "SipHash" do blueprint: 5 linhas
//!   auditáveis; hash com chave fica p/ RFC futura).
//! - CHECKSUM = CRC32-IEEE (tabela construída por chamada, sem estado global).
//! - HMAC = HMAC-SHA256 truncado em 64 bits (SHA256 compacto, verificado
//!   contra o vetor NIST "abc").

/// Semente padrão do boot frio (golden ratio; determinismo total).
pub const DEFAULT_RNG_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// splitmix64 (Steele et al.): avança `*state`, devolve u64.
pub fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniforme f32 em [a, b). Requer `a < b` finitos (validado pelo chamador).
pub fn uniform_f32(state: &mut u64, a: f32, b: f32) -> f32 {
    let bits24 = (splitmix64(state) >> 40) as f32; // [0, 2^24)
    let t = bits24 / 16_777_216.0; // [0, 1)
    a + t * (b - a)
}

/// Normal(mean, std) via Box-Muller (ramo cos). Requer std > 0 finito.
pub fn normal_f32(state: &mut u64, mean: f32, std: f32) -> f32 {
    let u1 = uniform_f32(state, 0.0, 1.0).max(f32::MIN_POSITIVE);
    let u2 = uniform_f32(state, 0.0, 1.0);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    mean + std * r * theta.cos()
}

/// FNV-1a/64.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h: u64 = 14_695_981_039_346_656_037;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(1_099_511_628_211);
    }
    h
}

/// CRC32-IEEE (polinômio refletido 0xEDB88320), tabela por chamada.
pub fn crc32_ieee(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
        *slot = c;
    }
    let mut crc: u32 = 0xFFFF_FFFF;
    for b in data {
        crc = table[((crc ^ (*b as u32)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

const SHA_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 compacto (FIPS 180-4). Verificado contra vetores NIST em testes.
pub fn sha256(msg: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut padded = msg.to_vec();
    let bitlen = (msg.len() as u64).wrapping_mul(8);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(SHA_K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

/// HMAC-SHA256 (RFC 2104), truncado nos primeiros 8 bytes (BE) => u64.
/// "Truncado" conforme o blueprint; forma de 256 bits fica p/ modo futuro.
pub fn hmac_sha256_trunc64(key: &[u8], msg: &[u8]) -> u64 {
    const BLOCK: usize = 64;
    let mut k = [0u8; BLOCK];
    if key.len() > BLOCK {
        let d = sha256(key);
        k[..32].copy_from_slice(&d);
    } else {
        k[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; BLOCK];
    let mut opad = [0x5cu8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }
    let mut inner = ipad.to_vec();
    inner.extend_from_slice(msg);
    let inner_digest = sha256(&inner);
    let mut outer = opad.to_vec();
    outer.extend_from_slice(&inner_digest);
    let digest = sha256(&outer);
    let mut b = [0u8; 8];
    b.copy_from_slice(&digest[..8]);
    u64::from_be_bytes(b)
}

/// Amostragem seeded de logits (RFC-0009): jitter anti-degeneração +
/// softmax + draw ponderado, tudo do `state` (splitmix64). Mesma semente +
/// mesmos logits => mesmo token, sempre. Vazios => 0 (legado).
pub fn sample_logits_seeded(state: &mut u64, logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    // Jitter: dummy logits constantes (ex: 0.7) não colapsam p/ token 0.
    let jittered: Vec<f32> = logits.iter().map(|&v| v + uniform_f32(state, -0.5, 0.5)).collect();
    let max = jittered.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut exps = Vec::with_capacity(jittered.len());
    for &v in &jittered {
        let e = (v - max).exp();
        exps.push(e);
        sum += e;
    }
    if sum.is_finite() && sum > 0.0 {
        let mut r = uniform_f32(state, 0.0, sum);
        for (i, e) in exps.iter().enumerate() {
            r -= *e;
            if r <= 0.0 {
                return i as u32;
            }
        }
    }
    // Fallback argmax (não-finito ou resíduo).
    let mut best_idx = 0usize;
    let mut best_p = 0.0f32;
    for (i, e) in exps.iter().enumerate() {
        let p = e / sum;
        if p > best_p {
            best_p = p;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Amostragem host/REPL (não-determinística por desenho: humano no loop).
/// Mesma forma do seeded (jitter + softmax + draw), fonte `thread_rng`.
/// Uso restrito a `main.rs`; a Vm NUNCA chama esta função.
pub fn sample_logits_host(logits: &[f32]) -> u32 {
    use rand::Rng;
    if logits.is_empty() {
        return 0;
    }
    let mut rng = rand::thread_rng();
    let jittered: Vec<f32> = logits.iter().map(|&v| v + rng.gen_range(-0.5..0.5)).collect();
    let max = jittered.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    let mut exps = Vec::with_capacity(jittered.len());
    for &v in &jittered {
        let e = (v - max).exp();
        exps.push(e);
        sum += e;
    }
    if sum.is_finite() && sum > 0.0 {
        let mut r: f32 = rng.gen_range(0.0..sum);
        for (i, e) in exps.iter().enumerate() {
            r -= *e;
            if r <= 0.0 {
                return i as u32;
            }
        }
    }
    let mut best_idx = 0usize;
    let mut best_p = 0.0f32;
    for (i, e) in exps.iter().enumerate() {
        let p = e / sum;
        if p > best_p {
            best_p = p;
            best_idx = i;
        }
    }
    best_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_vectors() {
        // Vetor âncora: string vazia = offset basis (certo por definição).
        assert_eq!(fnv1a64(b""), 14_695_981_039_346_656_037);
        // Avalanche sobre entradas vizinhas (propriedade, não vetor externo).
        let mut seen = std::collections::HashSet::new();
        for s in [b"a".as_slice(), b"b", b"aa", b"ab", b"ba", b"abc", b"abcd", b"123456789"] {
            assert!(seen.insert(fnv1a64(s)), "colisão FNV em {:?}", s);
        }
    }

    #[test]
    fn crc32_check_vector() {
        // Vetor padrão ("123456789" => 0xCBF43926).
        assert_eq!(crc32_ieee(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32_ieee(b""), 0x0000_0000);
    }

    #[test]
    fn sha256_nist_vectors() {
        // NIST FIPS 180-4: SHA256("abc").
        let d = sha256(b"abc");
        let hex: String = d.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        // SHA256("").
        let d0 = sha256(b"");
        let hex0: String = d0.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(hex0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn hmac_rfc4231_tc1_truncated() {
        // RFC 4231 TC1: key = 20x 0x0b, data = "Hi There".
        // HMAC-SHA256 = b0344c61d8db38535ca8afceaf0bf12b88...
        // truncado (8B BE) = 0xb0344c61d8db3853.
        let key = [0x0bu8; 20];
        assert_eq!(hmac_sha256_trunc64(&key, b"Hi There"), 0xb034_4c61_d8db_3853);
    }

    #[test]
    fn rng_determinism_and_reseed() {
        let mut s1 = DEFAULT_RNG_SEED;
        let mut s2 = DEFAULT_RNG_SEED;
        let a: Vec<u64> = (0..100).map(|_| splitmix64(&mut s1)).collect();
        let b: Vec<u64> = (0..100).map(|_| splitmix64(&mut s2)).collect();
        assert_eq!(a, b, "mesma semente => mesma sequência");
        let mut s3 = DEFAULT_RNG_SEED.wrapping_add(1);
        let c: Vec<u64> = (0..100).map(|_| splitmix64(&mut s3)).collect();
        assert_ne!(a, c, "sementes distintas divergem");
        // Uniforme respeita [a,b) em 10k amostras.
        let mut s = DEFAULT_RNG_SEED;
        for _ in 0..10_000 {
            let x = uniform_f32(&mut s, -2.0, 5.0);
            assert!((-2.0..5.0).contains(&x), "uniforme fora de [a,b): {}", x);
        }
        // Normal finita e centrada (média frouxa em 20k amostras).
        let mut s = DEFAULT_RNG_SEED;
        let n = 20_000;
        let mut acc = 0.0f64;
        for _ in 0..n {
            let x = normal_f32(&mut s, 0.0, 1.0);
            assert!(x.is_finite());
            acc += x as f64;
        }
        assert!((acc / n as f64).abs() < 0.05, "média normal desviada: {}", acc / n as f64);
    }

    #[test]
    fn sample_logits_seeded_properties() {
        let logits = vec![0.2, 1.5, -0.7, 0.9];
        // Determinismo: mesma seed + mesmos logits => mesmo token.
        let mut s1 = DEFAULT_RNG_SEED;
        let mut s2 = DEFAULT_RNG_SEED;
        let t1: Vec<u32> = (0..20).map(|_| sample_logits_seeded(&mut s1, &logits)).collect();
        let t2: Vec<u32> = (0..20).map(|_| sample_logits_seeded(&mut s2, &logits)).collect();
        assert_eq!(t1, t2);
        for &t in &t1 {
            assert!((t as usize) < logits.len());
        }
        // Reseed replays o prefixo.
        let mut s3 = DEFAULT_RNG_SEED;
        let t3: Vec<u32> = (0..20).map(|_| sample_logits_seeded(&mut s3, &logits)).collect();
        assert_eq!(t1, t3);
        // Vazios => 0 (legado).
        let mut s4 = DEFAULT_RNG_SEED;
        assert_eq!(sample_logits_seeded(&mut s4, &[]), 0);
        // Logits extremos => argmax determinístico.
        let mut s5 = DEFAULT_RNG_SEED;
        assert_eq!(sample_logits_seeded(&mut s5, &[100.0, 0.0, 0.0]), 0);
        // Host sampler: só forma (finito e no range); valores variam por desenho.
        let h = sample_logits_host(&logits);
        assert!((h as usize) < logits.len());
    }
}
