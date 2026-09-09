//! utils.rs — Helpers de timestamp, logs e métricas

use std::time::{SystemTime, UNIX_EPOCH};

/// Timestamp em nanossegundos desde UNIX_EPOCH (útil para TEMPORAL indexing e logs).
pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

pub fn now_ms() -> u64 {
    now_ns() / 1_000_000
}

/// Log estruturado simples (sem dependência de `tracing` para manter minimalismo).
/// Em prod pode ser trocado por `tracing::info!`.
pub fn log_info(target: &str, msg: &str) {
    eprintln!("[{:>12}][INFO][{}] {}", now_ms(), target, msg);
}

pub fn log_warn(target: &str, msg: &str) {
    eprintln!("[{:>12}][WARN][{}] {}", now_ms(), target, msg);
}

pub fn log_debug(target: &str, msg: &str) {
    if std::env::var("M3_DEBUG").is_ok() {
        eprintln!("[{:>12}][DEBUG][{}] {}", now_ms(), target, msg);
    }
}

/// Formata u128 como hex 0x com padding 32 dígitos (128 bits)
pub fn fmt_addr(addr: u128) -> String {
    format!("0x{:032x}", addr)
}

/// Converte 4 bytes LE para f32
pub fn bytes_to_f32_le(bytes: &[u8]) -> f32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[0..4]);
    f32::from_le_bytes(buf)
}

/// Mede throughput (instruções por segundo) com janela deslizante simples
pub struct ThroughputMeter {
    start_ns: u64,
    count: u64,
}

impl ThroughputMeter {
    pub fn new() -> Self {
        Self { start_ns: now_ns(), count: 0 }
    }

    pub fn tick(&mut self, n: u64) {
        self.count += n;
    }

    pub fn ips(&self) -> f64 {
        let elapsed = now_ns().saturating_sub(self.start_ns) as f64 / 1e9;
        if elapsed < 1e-9 {
            0.0
        } else {
            self.count as f64 / elapsed
        }
    }

    pub fn report(&self) -> String {
        format!("{} instr em {:.3}s = {:.0} IPS", self.count, (now_ns() - self.start_ns) as f64 / 1e9, self.ips())
    }
}

impl Default for ThroughputMeter {
    fn default() -> Self {
        Self::new()
    }
}
