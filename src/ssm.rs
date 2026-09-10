//! ssm.rs — Núcleo Mamba-1 (spike) para o M³-AVM.
//!
//! Escopo propositalmente pequeno: primitivas puras + estado recorrente,
//! sem GGUF/matvec. O `inference.rs` alimenta estas funções com pesos
//! dequantizados (ou `matvec_quant` direto) e resolve os nomes
//! `blk.{i}.ssm_*` via `MambaLayerNames`.
//!
//! Convenções (documentadas para paridade futura com llama.cpp):
//! - `d_model` = hidden (`ModelConfig::hidden`)
//! - `d_inner` = ssm.inner_size (default `2*d_model`)
//! - `d_state` = ssm.state_size (default 16)
//! - `d_conv` = ssm.conv_kernel (default 4)
//! - `dt_rank` = ssm.time_step_rank (default `ceil(d_model/16)`)
//!
//! Layouts:
//! - conv1d depthwise: `weight[o*d_conv + k]`, `k=0` oldest, `k=d_conv-1` newest.
//!   `state_conv[o*d_conv + k]` guarda a janela (após update, `[.., x_cur]`).
//! - SSM: `A[o*d_state + i]` (já `-exp(A_log)`, negativo, como o GGUF após
//!   `conversion/mamba.py`), `B/C[i]`, `D[o]`, `x/dt[o]`.
//!
//! Matemática (Euler simplificado, documentado como spike):
//! ```text
//! a_d = exp(dt[o] * A[o,i])
//! h[o,i] = h[o,i]*a_d + x[o]*B[i]*dt[o]
//! y[o] = dot(h[o], C) + D[o]*x[o]
//! ```
//! O ZOH exato usaria `(a_d-1)/A * B`; o Euler coincide em `dt` pequeno e
//! evita divisão por `A≈0`. Paridade bit a bit com `mamba_ssm` NÃO é meta
//! deste spike — meta é estado constante + teste determinístico.

/// SiLU: x * sigmoid(x).
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Softplus: ln(1 + exp(x)), estável para x grande.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// RMSNorm sem peso (Falcon-Mamba aplica em dt/B/C sem parâmetros aprendidos).
pub fn rms_norm_plain(x: &[f32], eps: f32) -> Vec<f32> {
    if x.is_empty() {
        return Vec::new();
    }
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    let rms = ((sum / x.len() as f32) + eps).sqrt().max(1e-12);
    x.iter().map(|&v| v / rms).collect()
}

/// Estado recorrente de UMA camada Mamba: conv causal + SSM.
/// `conv` tem `d_inner*d_conv` (janela por canal), `ssm` tem `d_inner*d_state`.
#[derive(Debug, Clone)]
pub struct MambaState {
    pub conv: Vec<f32>,
    pub ssm: Vec<f32>,
    pub d_inner: usize,
    pub d_state: usize,
    pub d_conv: usize,
}

impl MambaState {
    pub fn new(d_inner: usize, d_state: usize, d_conv: usize) -> Self {
        Self {
            conv: vec![0.0; d_inner * d_conv],
            ssm: vec![0.0; d_inner * d_state],
            d_inner,
            d_state,
            d_conv,
        }
    }

    pub fn reset(&mut self) {
        for v in self.conv.iter_mut() {
            *v = 0.0;
        }
        for v in self.ssm.iter_mut() {
            *v = 0.0;
        }
    }

    pub fn is_compatible(&self, d_inner: usize, d_state: usize, d_conv: usize) -> bool {
        self.d_inner == d_inner && self.d_state == d_state && self.d_conv == d_conv
    }
}

/// Atualiza a janela causal e aplica conv depthwise + bias.
/// Retorna `y[o] = dot(janela_o, peso_o) + bias[o]`.
/// `weight` tem `d_inner*d_conv`, `bias` tem `d_inner` (vazio = zero).
pub fn conv1d_depthwise_update(
    state_conv: &mut [f32],
    x_cur: &[f32],
    weight: &[f32],
    bias: &[f32],
    d_inner: usize,
    d_conv: usize,
) -> Vec<f32> {
    debug_assert_eq!(state_conv.len(), d_inner * d_conv);
    debug_assert_eq!(x_cur.len(), d_inner);
    debug_assert_eq!(weight.len(), d_inner * d_conv);
    let mut y = vec![0.0f32; d_inner];
    for o in 0..d_inner {
        let base = o * d_conv;
        // shift: descarta oldest, desloca [1..] -> [0..]
        if d_conv > 1 {
            state_conv.copy_within(base + 1..base + d_conv, base);
        }
        state_conv[base + d_conv - 1] = x_cur[o];
        let mut acc = if bias.len() == d_inner { bias[o] } else { 0.0 };
        for k in 0..d_conv {
            acc += state_conv[base + k] * weight[o * d_conv + k];
        }
        y[o] = acc;
    }
    y
}

/// Monta o vetor de params do `OP_SSM_SCAN` (`vm.rs::read_ssm_pack`):
/// `dt[d_inner] + A[d_inner*d_state] + B[d_state] + C[d_state] + D[d_inner]`.
/// Panics se os tamanhos não baterem — falhar cedo na montagem do programa.
pub fn pack_params(dt: &[f32], a: &[f32], b: &[f32], c: &[f32], d: &[f32], d_inner: usize, d_state: usize) -> Vec<f32> {
    assert_eq!(dt.len(), d_inner, "dt deve ter d_inner={}", d_inner);
    assert_eq!(a.len(), d_inner * d_state, "A deve ter d_inner*d_state={}", d_inner * d_state);
    assert_eq!(b.len(), d_state, "B deve ter d_state={}", d_state);
    assert_eq!(c.len(), d_state, "C deve ter d_state={}", d_state);
    assert_eq!(d.len(), d_inner, "D deve ter d_inner={}", d_inner);
    let mut out = Vec::with_capacity(d_inner + d_inner * d_state + 2 * d_state + d_inner);
    out.extend_from_slice(dt);
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out.extend_from_slice(c);
    out.extend_from_slice(d);
    out
}

/// Um passo do selective scan (Euler). Atualiza `state_ssm` in-place e retorna `y`.
pub fn selective_scan_update(
    state_ssm: &mut [f32],
    x: &[f32],
    dt: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    d: &[f32],
    d_inner: usize,
    d_state: usize,
) -> Vec<f32> {
    debug_assert_eq!(state_ssm.len(), d_inner * d_state);
    debug_assert_eq!(x.len(), d_inner);
    debug_assert_eq!(dt.len(), d_inner);
    debug_assert_eq!(a.len(), d_inner * d_state);
    debug_assert_eq!(b.len(), d_state);
    debug_assert_eq!(c.len(), d_state);
    let mut y = vec![0.0f32; d_inner];
    for o in 0..d_inner {
        let dt_o = dt[o];
        let x_o = x[o];
        let row = o * d_state;
        let mut acc = 0.0f32;
        for i in 0..d_state {
            let a_oi = a[row + i];
            let a_d = (dt_o * a_oi).exp();
            let h = &mut state_ssm[row + i];
            *h = *h * a_d + x_o * b[i] * dt_o;
            acc += *h * c[i];
        }
        let d_o = if d.len() == d_inner { d[o] } else { 0.0 };
        y[o] = acc + d_o * x_o;
    }
    y
}

/// Gating final do bloco: `y[o] *= silu(z[o])`.
pub fn apply_gate(y: &mut [f32], z: &[f32]) {
    debug_assert_eq!(y.len(), z.len());
    for (yv, &zv) in y.iter_mut().zip(z.iter()) {
        *yv *= silu(zv);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_softplus_sanity() {
        assert!((silu(0.0)).abs() < 1e-6);
        assert!((silu(1.0) - 0.7310586).abs() < 1e-5);
        assert!((softplus(0.0) - 0.6931472).abs() < 1e-5);
        assert!((softplus(30.0) - 30.0).abs() < 1e-4);
        // softplus sempre > 0 (dt positivo)
        for &v in &[-3.0f32, -0.5, 0.7, 5.0] {
            assert!(softplus(v) > 0.0);
        }
    }

    #[test]
    fn test_conv1d_window_and_dot() {
        // d_inner=2, d_conv=3, peso identidade no newest (k=2), bias 0.
        let d_inner = 2;
        let d_conv = 3;
        let mut st = MambaState::new(d_inner, 4, d_conv);
        let w = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let b: Vec<f32> = vec![];
        let y1 = conv1d_depthwise_update(&mut st.conv, &[5.0, 7.0], &w, &b, d_inner, d_conv);
        assert_eq!(y1, vec![5.0, 7.0]);
        // janela agora [0,0,5]/[0,0,7]; próximo passo com peso oldest-only
        let w_old = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let y2 = conv1d_depthwise_update(&mut st.conv, &[1.0, 1.0], &w_old, &b, d_inner, d_conv);
        // janela [0,5,1]/[0,7,1] -> oldest = 0
        assert_eq!(y2, vec![0.0, 0.0]);
    }

    #[test]
    fn test_scan_decay_and_leak() {
        // d_inner=1, d_state=1: h = h*exp(dt*A) + x*B*dt; y = h*C + D*x
        let mut h = vec![0.0f32; 1];
        let a = vec![-1.0f32];
        let b = vec![1.0f32];
        let c = vec![1.0f32];
        let d = vec![0.0f32];
        // passo 1: x=1, dt=1 -> h=1, y=1
        let y1 = selective_scan_update(&mut h, &[1.0], &[1.0], &a, &b, &c, &d, 1, 1);
        assert!((y1[0] - 1.0).abs() < 1e-5, "{}", y1[0]);
        // passo 2: x=0 -> h decai por e^-1
        let y2 = selective_scan_update(&mut h, &[0.0], &[1.0], &a, &b, &c, &d, 1, 1);
        let expect = (-1.0f32).exp();
        assert!((y2[0] - expect).abs() < 1e-5, "{} vs {}", y2[0], expect);
        // D (skip) soma direto
        let mut h2 = vec![0.0f32; 1];
        let d2 = vec![2.0f32];
        let y3 = selective_scan_update(&mut h2, &[3.0], &[0.5], &a, &b, &c, &d2, 1, 1);
        // h=3*1*0.5=1.5, y=1.5*1+2*3=7.5
        assert!((y3[0] - 7.5).abs() < 1e-5, "{}", y3[0]);
    }

    #[test]
    fn test_pack_params_layout_roundtrip() {
        // Layout precisa espelhar `Vm::read_ssm_pack`: dt+A+B+C+D.
        let dt = vec![1.0f32, 2.0];
        let a = vec![-1.0f32; 4];
        let b = vec![0.5f32, 1.5];
        let c = vec![1.0f32, 1.0];
        let d = vec![0.0f32, 0.25];
        let p = pack_params(&dt, &a, &b, &c, &d, 2, 2);
        assert_eq!(p.len(), 2 + 4 + 2 + 2 + 2);
        assert_eq!(&p[0..2], &dt);
        assert_eq!(&p[2..6], &a);
        assert_eq!(&p[6..8], &b);
        assert_eq!(&p[8..10], &c);
        assert_eq!(&p[10..12], &d);
    }

    #[test]
    #[should_panic(expected = "B deve ter")]
    fn test_pack_params_rejeita_tamanho_errado() {
        pack_params(&[1.0], &[-1.0], &[1.0, 2.0], &[1.0], &[0.0], 1, 1);
    }

    #[test]
    fn test_state_reset_and_compat() {
        let mut st = MambaState::new(4, 8, 4);
        assert!(st.is_compatible(4, 8, 4));
        assert!(!st.is_compatible(8, 8, 4));
        st.conv[0] = 1.0;
        st.ssm[0] = 2.0;
        st.reset();
        assert!(st.conv.iter().all(|&v| v == 0.0));
        assert!(st.ssm.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_rms_plain_falcon() {
        // RMS sem peso: normaliza para norma sqrt(n)
        let y = rms_norm_plain(&[3.0, 4.0], 0.0);
        // rms = sqrt((9+16)/2)=sqrt(12.5)
        let rms = (12.5f32).sqrt();
        assert!((y[0] - 3.0 / rms).abs() < 1e-5);
        assert!((y[1] - 4.0 / rms).abs() < 1e-5);
    }
}
