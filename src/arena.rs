//! arena.rs — bump allocator por arena (RFC-0023, `0x26`/`0x27`).
//!
//! `Arena` = buffer crescente + cursor. `alloc` alinha o cursor, cresce
//! por dobra e retorna o offset. `reset` zera o cursor em O(1) mantendo
//! a capacidade. Sem free individual por desenho (bump); `ARENA_FREE`
//! exigiria outro alocador (RFC futura).
//!
//! Isolamento FORK/ABORT vive no `Vm` (clone/push/pop do mapa, como
//! `rank1_layers`, RFC-0004); aqui só a mecânica do bump + teto.

use thiserror::Error;

/// Teto por arena (RFC-0023 §Security; ajustável em RFC futura).
pub const ARENA_MAX_BYTES: usize = 1 << 30;
/// Alinhamento default quando `ALIGN=0`.
pub const ARENA_DEFAULT_ALIGN: usize = 16;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ArenaError {
    #[error("ARENA_ALLOC: size 0 (alocação vazia é bug do chamador)")]
    ZeroSize,
    #[error("ARENA_ALLOC: align {0} não é potência de dois")]
    BadAlign(u32),
    #[error("ARENA_ALLOC: teto de {0} bytes por arena excedido")]
    OverCap(usize),
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Arena {
    buf: Vec<u8>,
    cursor: usize,
}

impl Arena {
    pub fn new() -> Self {
        Self::default()
    }

    /// Aloca `size` bytes com `align` (0 = default); retorna o offset.
    /// Crescimento por dobra (amortizado O(1)); nunca fragmenta.
    pub fn alloc(&mut self, size: usize, align: usize) -> Result<usize, ArenaError> {
        if size == 0 {
            return Err(ArenaError::ZeroSize);
        }
        let align = if align == 0 { ARENA_DEFAULT_ALIGN } else { align };
        if !align.is_power_of_two() {
            return Err(ArenaError::BadAlign(align as u32));
        }
        let aligned = self
            .cursor
            .checked_add(align - 1)
            .ok_or(ArenaError::OverCap(ARENA_MAX_BYTES))?
            & !(align - 1);
        let end = aligned
            .checked_add(size)
            .ok_or(ArenaError::OverCap(ARENA_MAX_BYTES))?;
        if end > ARENA_MAX_BYTES {
            return Err(ArenaError::OverCap(ARENA_MAX_BYTES));
        }
        if end > self.buf.len() {
            let mut cap = self.buf.len().max(64);
            while cap < end {
                cap = cap.saturating_mul(2);
            }
            self.buf.resize(cap, 0);
        }
        self.cursor = end;
        Ok(aligned)
    }

    /// RESET O(1): cursor volta a 0, capacidade mantida.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Cursor atual (= bytes em uso).
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Capacidade reservada (não encolhe no RESET).
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Janela alocada `[..cursor]` (inspeção/teste).
    pub fn used(&self) -> &[u8] {
        &self.buf[..self.cursor]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_offsets_and_alignment() {
        let mut a = Arena::new();
        assert_eq!(a.alloc(64, 0).unwrap(), 0);
        assert_eq!(a.alloc(32, 16).unwrap(), 64);
        // cursor=96; align-up de 1 em 16 => 96.
        assert_eq!(a.alloc(1, 16).unwrap(), 96);
        assert_eq!(a.cursor(), 97);
        // align-up real: cursor=97, align 32 => 128.
        assert_eq!(a.alloc(8, 32).unwrap(), 128);
    }

    #[test]
    fn reset_reuses_and_keeps_capacity() {
        let mut a = Arena::new();
        a.alloc(100, 1).unwrap();
        let cap = a.capacity();
        assert!(cap >= 100);
        a.reset();
        assert_eq!(a.cursor(), 0);
        assert_eq!(a.capacity(), cap); // O(1): não desaloca
        assert_eq!(a.alloc(16, 0).unwrap(), 0); // reuso do início
    }

    #[test]
    fn growth_doubles() {
        let mut a = Arena::new();
        a.alloc(100, 1).unwrap();
        assert_eq!(a.capacity(), 128); // 64 -> 128
    }

    #[test]
    fn errors_without_big_allocs() {
        let mut a = Arena::new();
        assert_eq!(a.alloc(0, 16), Err(ArenaError::ZeroSize));
        assert_eq!(a.alloc(8, 3), Err(ArenaError::BadAlign(3)));
        assert_eq!(a.alloc(8, 0), Ok(0)); // ALIGN=0 => default, ok
        // Teto sem alocar 1 GiB: size além do teto falha antes do resize.
        assert_eq!(
            a.alloc(ARENA_MAX_BYTES + 1, 1),
            Err(ArenaError::OverCap(ARENA_MAX_BYTES))
        );
        assert_eq!(a.capacity(), 64); // nada foi reservado no erro
    }
}
