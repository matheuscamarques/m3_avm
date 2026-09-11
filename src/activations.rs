//! activations.rs — não-linearidades elementares + softmax estável (RFC-0028).
//!
//! Funções totais sobre f32 com semântica IEEE documentada (ver RFC para
//! a tabela NaN). Sem `libm`: erf via Abramowitz & Stegun 7.1.26
//! (|err| <= 1.5e-7, suficiente p/ f32); exp/log/tanh são intrínsecos.

/// erf(x) via Abramowitz & Stegun 7.1.26. Total (NaN => NaN).
pub fn erf_approx(x: f32) -> f32 {
    if x.is_nan() {
        return f32::NAN;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    // Horner: ((((a5·t+a4)·t+a3)·t+a2)·t+a1)·t·e^{-x²}
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}

/// GELU exato: 0.5·x·(1+erf(x/√2)).
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_approx(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Sigmoid: 1/(1+e^{-x}). A aritmética inf do f32 torna as caudas exatas
/// (x→-inf dá 0, x→+inf dá 1); NaN propaga.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax estável in-place sobre uma lane: subtrai o máximo, divide
/// pela temperatura, expõe, normaliza. `temp` deve ser > 0 ou +inf
/// (uniforme); o chamador valida (temp NaN/<=0 é Err no exec).
/// NaN na lane envenena a lane (aritmética natural, documentado).
pub fn softmax_lane(vals: &mut [f32], temp: f32) {
    debug_assert!(!temp.is_nan() && (temp > 0.0 || temp == f32::INFINITY));
    let mut max = f32::NEG_INFINITY;
    for &v in vals.iter() {
        if v > max {
            max = v;
        }
    }
    // Nota: max ignora NaN (f32::max seria igual); NaN entra via exp.
    let mut sum = 0.0f32;
    for v in vals.iter_mut() {
        *v = ((*v - max) / temp).exp();
        sum += *v;
    }
    for v in vals.iter_mut() {
        *v /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_vectors() {
        assert!((erf_approx(0.0)).abs() < 1e-7);
        assert!((erf_approx(1.0) - 0.8427007).abs() < 2e-6);
        assert!((erf_approx(-1.0) + 0.8427007).abs() < 2e-6);
        assert!((erf_approx(3.0) - 0.9999779).abs() < 2e-6);
        assert!(erf_approx(f32::NAN).is_nan());
    }

    #[test]
    fn gelu_vectors() {
        // Φ(1) = 0.8413447 (exato); A&S dá ~1e-7 de erro.
        assert!((gelu(1.0) - 0.8413447).abs() < 1e-6);
        assert!((gelu(-1.0) + 0.1586553).abs() < 1e-6);
        assert_eq!(gelu(0.0), 0.0);
        assert!(gelu(f32::NAN).is_nan());
    }

    #[test]
    fn sigmoid_vectors() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!((sigmoid(2.0) - 0.8807971).abs() < 1e-6);
        assert_eq!(sigmoid(1000.0), 1.0);
        assert_eq!(sigmoid(-1000.0), 0.0);
        assert!(sigmoid(f32::NAN).is_nan());
    }

    #[test]
    fn softmax_lanes() {
        // Uniforme.
        let mut u = [0.0f32; 4];
        softmax_lane(&mut u, 1.0);
        assert_eq!(u, [0.25; 4]);
        // Vetor conhecido [1,2,3].
        let mut v = [1.0f32, 2.0, 3.0];
        softmax_lane(&mut v, 1.0);
        assert!((v[0] - 0.0900306).abs() < 1e-5, "{:?}", v);
        assert!((v[1] - 0.2447285).abs() < 1e-5, "{:?}", v);
        assert!((v[2] - 0.6652410).abs() < 1e-5, "{:?}", v);
        // Temperatura alta achata.
        let mut h = [1.0f32, 2.0, 3.0];
        softmax_lane(&mut h, 100.0);
        for x in h {
            assert!((x - 1.0 / 3.0).abs() < 0.01, "{:?}", h);
        }
        // Temperatura baixa afia.
        let mut l = [1.0f32, 2.0, 3.0];
        softmax_lane(&mut l, 0.1);
        assert!(l[2] > 0.999, "{:?}", l);
        // NaN envenena a lane (documentado).
        let mut n = [1.0f32, f32::NAN, 3.0];
        softmax_lane(&mut n, 1.0);
        assert!(n.iter().all(|x| x.is_nan()), "{:?}", n);
    }
}
