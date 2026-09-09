//! rollback.rs — Algoritmo de Rollback V1: Entropia + Fallback Janela Fixa
//!
//! Tese: preempção cirúrgica preserva 95% do raciocínio. Para achar *onde* voltar,
//! V1 combina:
//!   Algoritmo 4 — pico de entropia dos logits (incerteza do modelo)
//!   Algoritmo 1 — fallback janela fixa 5 tokens (robusto)
//!
//! Entropia: H(p) = - Σ p_i ln p_i , p = softmax(logits)

use std::collections::VecDeque;

/// Rastreia entropia dos logits durante a geração
pub struct EntropyTracker {
    history: VecDeque<EntropySample>,
    max_samples: usize,
}

/// Uma amostra de entropia com seu índice no buffer de saída
#[derive(Debug, Clone, Copy)]
pub struct EntropySample {
    pub token_index: usize,
    pub entropy: f32,
}

impl EntropyTracker {
    pub fn new(max_samples: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Adiciona nova amostra
    pub fn push(&mut self, token_index: usize, entropy: f32) {
        if self.history.len() >= self.max_samples {
            self.history.pop_front();
        }
        self.history.push_back(EntropySample { token_index, entropy });
    }

    /// Encontra índice do pico máximo de entropia nos últimos N.
    /// Retorna target = pico -1 (token anterior ao pico)
    pub fn find_peak_target(&self) -> Option<usize> {
        if self.history.is_empty() {
            return None;
        }
        let peak = self
            .history
            .iter()
            .max_by(|a, b| a.entropy.partial_cmp(&b.entropy).unwrap())?;
        if peak.entropy < 0.5 {
            return None;
        }
        let target = if peak.token_index > 0 { peak.token_index - 1 } else { 0 };
        Some(target)
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }

    pub fn average_entropy(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.history.iter().map(|s| s.entropy).sum();
        sum / self.history.len() as f32
    }

    pub fn len(&self) -> usize { self.history.len() }
    pub fn is_empty(&self) -> bool { self.history.is_empty() }

    /// Para debugging: snapshot do histórico
    pub fn history(&self) -> &VecDeque<EntropySample> { &self.history }
}

/// Algoritmo 1: janela fixa fallback
pub const FALLBACK_OFFSET: usize = 5;

pub fn fallback_target(total_tokens: usize) -> usize {
    if total_tokens >= FALLBACK_OFFSET {
        total_tokens - FALLBACK_OFFSET
    } else {
        0
    }
}

/// Entropia de logits via softmax estável
pub fn compute_entropy(logits: &[f32]) -> f32 {
    if logits.is_empty() { return 0.0; }
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_sum: f32 = logits.iter().map(|&v| (v - max_logit).exp()).sum();
    if exp_sum == 0.0 || !exp_sum.is_finite() { return 0.0; }
    let mut entropy = 0.0;
    for &v in logits {
        let p = (v - max_logit).exp() / exp_sum;
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Função principal V1: tenta entropia (Alg 4), senão fallback (Alg 1)
pub fn determine_target_token_index(
    entropy_tracker: &EntropyTracker,
    total_tokens: usize,
) -> usize {
    if let Some(peak_idx) = entropy_tracker.find_peak_target() {
        eprintln!(
            "[Rollback] Entropia: pico detectado no token {}, voltando para {} (H={:.2} avg={:.2})",
            peak_idx + 1,
            peak_idx,
            entropy_tracker.history.iter().find(|s| s.token_index == peak_idx+1).map(|s| s.entropy).unwrap_or(0.0),
            entropy_tracker.average_entropy()
        );
        return peak_idx;
    }
    let fallback = fallback_target(total_tokens);
    eprintln!(
        "[Rollback] Fallback: voltando {} tokens (últimos {} of {}) -> target {}",
        total_tokens.saturating_sub(fallback),
        FALLBACK_OFFSET,
        total_tokens,
        fallback
    );
    fallback
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_tracker_peak() {
        let mut t = EntropyTracker::new(50);
        t.push(0, 0.2);
        t.push(1, 0.3);
        t.push(2, 1.8); // pico
        t.push(3, 0.4);
        assert_eq!(t.find_peak_target(), Some(1)); // pico 2 -> target 1
    }

    #[test]
    fn test_entropy_ignore_low() {
        let mut t = EntropyTracker::new(10);
        t.push(0, 0.1);
        t.push(1, 0.3);
        assert_eq!(t.find_peak_target(), None);
    }

    #[test]
    fn test_fallback() {
        assert_eq!(fallback_target(10), 5);
        assert_eq!(fallback_target(3), 0);
        assert_eq!(fallback_target(5), 0);
    }

    #[test]
    fn test_determine_uses_entropy_first() {
        let mut t = EntropyTracker::new(10);
        t.push(0, 0.2);
        t.push(5, 2.0);
        assert_eq!(determine_target_token_index(&t, 10), 4);
    }

    #[test]
    fn test_determine_fallback_when_no_peak() {
        let t = EntropyTracker::new(10);
        assert_eq!(determine_target_token_index(&t, 10), 5);
    }

    #[test]
    fn test_compute_entropy_uniform_high() {
        // 4 logits iguais -> p=0.25 cada -> H = -4*0.25*ln0.25 = 1.386
        let e = compute_entropy(&[0.0, 0.0, 0.0, 0.0]);
        assert!((e - 1.386).abs() < 0.01);
    }

    #[test]
    fn test_compute_entropy_peaked_low() {
        // pico forte -> entropia baixa
        let e = compute_entropy(&[10.0, 0.0, 0.0, 0.0]);
        assert!(e < 0.5);
    }

    #[test]
    fn test_compute_entropy_empty() {
        assert_eq!(compute_entropy(&[]), 0.0);
    }
}
