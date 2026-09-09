//! context.rs — Modelo de Contextos e Escalonador da M³-AVM
//!
//! Cada Contexto possui:
//!   - 16 registradores `u128` (endereço ou imediato)
//!   - Program Counter (u128)
//!   - ponteiro para raiz/version da memória persistente
//!   - prioridade Red/Blue/Green
//!   - estado (Ready/Running/Blocked/Terminated)
//!
//! O escalonador obedece prioridades estritas:
//!   RED   (IRQ/SENSE) — preempção absoluta, latência alvo <100µs
//!   BLUE  (Áudio)     — preempção sobre GREEN
//!   GREEN (Batch/LLM) — best-effort
//!
//! Implementado como 3 filas FIFO por prioridade + mapa de contextos.

use std::collections::{HashMap, VecDeque};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// GREEN — Batch / LLM, menor prioridade
    Green = 0,
    /// BLUE — Áudio, média
    Blue = 1,
    /// RED — IRQ / SENSE, máxima, preemptiva
    Red = 2,
}

impl Priority {
    pub fn from_flags(flags: u8) -> Self {
        match flags & 0b11 {
            0b00 => Self::Green,
            0b01 => Self::Blue,
            0b10 => Self::Red,
            _ => Self::Red,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Blue,
            2 => Self::Red,
            _ => Self::Green,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Red => "RED",
            Self::Blue => "BLUE",
            Self::Green => "GREEN",
        }
    }
}

impl std::fmt::Display for Priority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextState {
    Ready,
    Running,
    Blocked,
    Terminated,
}

impl std::fmt::Display for ContextState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Ready => "Ready",
            Self::Running => "Running",
            Self::Blocked => "Blocked",
            Self::Terminated => "Terminated",
        };
        write!(f, "{}", s)
    }
}

/// Pipeline de execução ativo do contexto (CTX_SWITCH 0x18).
/// 0=Mamba/SSM, 1=Transformer (default), 2=Áudio/Depformer.
pub type PipelineId = u8;
pub const PIPE_MAMBA_CTX: PipelineId = 0;
pub const PIPE_TRANSFORMER_CTX: PipelineId = 1;
pub const PIPE_AUDIO_CTX: PipelineId = 2;

#[derive(Debug, Clone)]
pub struct Context {
    pub id: u64,
    pub regs: [u128; 16],
    pub pc: u128,
    pub root_version: u64,
    pub priority: Priority,
    pub state: ContextState,
    /// Timestamp de criação (para TEMPORAL indexing / debug)
    pub created_at_ns: u64,
    /// Flag de igualdade para controle de fluxo (COMPARE / IF_EQUAL)
    pub cmp_equal: bool,
    /// Flag de interrupção específica do contexto
    pub interrupt_flag: bool,
    /// Pipeline ativo (CTX_SWITCH). Default Transformer para compat.
    pub pipeline: PipelineId,
}

impl Context {
    pub fn new(id: u64, priority: Priority, pc: u128, root_version: u64) -> Self {
        Self {
            id,
            regs: [0u128; 16],
            pc,
            root_version,
            priority,
            state: ContextState::Ready,
            created_at_ns: crate::utils::now_ns(),
            cmp_equal: false,
            interrupt_flag: false,
            pipeline: PIPE_TRANSFORMER_CTX,
        }
    }

    /// Clona contexto para FORK — novo ID, copia regs/PC/root, prioridade pode mudar.
    pub fn fork(&self, new_id: u64, new_priority: Priority) -> Self {
        let mut child = self.clone();
        child.id = new_id;
        child.priority = new_priority;
        child.state = ContextState::Ready;
        child.created_at_ns = crate::utils::now_ns();
        // PC do filho avança 1 instrução (32 bytes) para não re-executar o FORK
        child.pc = child.pc.wrapping_add(32);
        child
    }

    pub fn reg(&self, idx: u8) -> Result<u128, ContextError> {
        if idx >= 16 {
            return Err(ContextError::InvalidRegister(idx));
        }
        Ok(self.regs[idx as usize])
    }

    pub fn set_reg(&mut self, idx: u8, val: u128) -> Result<(), ContextError> {
        if idx >= 16 {
            return Err(ContextError::InvalidRegister(idx));
        }
        self.regs[idx as usize] = val;
        Ok(())
    }

    /// Incrementa PC em 32 bytes (tamanho fixo da instrução).
    pub fn advance_pc(&mut self) {
        self.pc = self.pc.wrapping_add(32);
    }
}

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("registrador inválido: r{0} (válido 0..15)")]
    InvalidRegister(u8),
    #[error("contexto {0} não encontrado")]
    NotFound(u64),
    #[error("nenhum contexto pronto para escalonar")]
    NoReadyContext,
}

// ---------------------------------------------------------------------------
// Scheduler — 3 filas estritas
// ---------------------------------------------------------------------------

pub struct Scheduler {
    /// Filas FIFO por prioridade
    red_q: VecDeque<u64>,
    blue_q: VecDeque<u64>,
    green_q: VecDeque<u64>,
    /// Mapa de contextos
    contexts: HashMap<u64, Context>,
    next_id: u64,
    /// Contexto atualmente em execução (para preempção)
    current: Option<u64>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            red_q: VecDeque::new(),
            blue_q: VecDeque::new(),
            green_q: VecDeque::new(),
            contexts: HashMap::new(),
            next_id: 1, // 0 reservado para idle / kernel
            current: None,
        }
    }

    pub fn create_context(&mut self, priority: Priority, pc: u128, root_version: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let ctx = Context::new(id, priority, pc, root_version);
        self.contexts.insert(id, ctx);
        self.enqueue(id);
        id
    }

    pub fn insert_context(&mut self, ctx: Context) {
        let id = ctx.id;
        // Garante que next_id avance
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        self.contexts.insert(id, ctx);
        self.enqueue(id);
    }

    pub fn get(&self, id: u64) -> Option<&Context> {
        self.contexts.get(&id)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Context> {
        self.contexts.get_mut(&id)
    }

    pub fn remove(&mut self, id: u64) -> Option<Context> {
        self.dequeue_id(id);
        if self.current == Some(id) {
            self.current = None;
        }
        self.contexts.remove(&id)
    }

    /// Enfileira contexto conforme prioridade (se Ready/Running)
    pub fn enqueue(&mut self, id: u64) {
        if let Some(ctx) = self.contexts.get(&id) {
            if ctx.state == ContextState::Terminated {
                return;
            }
            match ctx.priority {
                Priority::Red => self.red_q.push_back(id),
                Priority::Blue => self.blue_q.push_back(id),
                Priority::Green => self.green_q.push_back(id),
            }
        }
    }

    fn dequeue_id(&mut self, id: u64) {
        self.red_q.retain(|&x| x != id);
        self.blue_q.retain(|&x| x != id);
        self.green_q.retain(|&x| x != id);
    }

    /// Escalonador estrito: RED > BLUE > GREEN, FIFO dentro da faixa.
    /// Retorna ID do próximo contexto a executar.
    pub fn pick_next(&mut self) -> Result<u64, ContextError> {
        // Se houver RED pronto, sempre escolhe RED (preempção)
        if let Some(&id) = self.red_q.front() {
            // Verifica se ainda está Ready/Running
            if let Some(ctx) = self.contexts.get(&id) {
                if ctx.state == ContextState::Terminated || ctx.state == ContextState::Blocked {
                    self.red_q.pop_front();
                    return self.pick_next();
                }
            }
            let id = self.red_q.pop_front().unwrap();
            self.current = Some(id);
            if let Some(ctx) = self.contexts.get_mut(&id) {
                ctx.state = ContextState::Running;
            }
            return Ok(id);
        }
        if let Some(&id) = self.blue_q.front() {
            if let Some(ctx) = self.contexts.get(&id) {
                if ctx.state == ContextState::Terminated || ctx.state == ContextState::Blocked {
                    self.blue_q.pop_front();
                    return self.pick_next();
                }
            }
            let id = self.blue_q.pop_front().unwrap();
            self.current = Some(id);
            if let Some(ctx) = self.contexts.get_mut(&id) {
                ctx.state = ContextState::Running;
            }
            return Ok(id);
        }
        if let Some(&id) = self.green_q.front() {
            if let Some(ctx) = self.contexts.get(&id) {
                if ctx.state == ContextState::Terminated || ctx.state == ContextState::Blocked {
                    self.green_q.pop_front();
                    return self.pick_next();
                }
            }
            let id = self.green_q.pop_front().unwrap();
            self.current = Some(id);
            if let Some(ctx) = self.contexts.get_mut(&id) {
                ctx.state = ContextState::Running;
            }
            return Ok(id);
        }
        Err(ContextError::NoReadyContext)
    }

    /// Re-enfileira o contexto atual após uma fatia (time-slice) — round-robin dentro da prioridade
    pub fn yield_current(&mut self) {
        if let Some(id) = self.current.take() {
            if let Some(ctx) = self.contexts.get_mut(&id) {
                if ctx.state == ContextState::Running {
                    ctx.state = ContextState::Ready;
                }
                if ctx.state == ContextState::Ready {
                    // Re-enfileira no fim da sua fila
                    match ctx.priority {
                        Priority::Red => self.red_q.push_back(id),
                        Priority::Blue => self.blue_q.push_back(id),
                        Priority::Green => self.green_q.push_back(id),
                    }
                }
            }
        }
    }

    /// Força preempção: se um RED chegou, o atual volta para fila.
    pub fn maybe_preempt(&mut self) {
        if !self.red_q.is_empty() {
            if let Some(cur) = self.current {
                if let Some(ctx) = self.contexts.get(&cur) {
                    if ctx.priority != Priority::Red {
                        self.yield_current();
                    }
                }
            }
        }
    }

    pub fn contexts(&self) -> &HashMap<u64, Context> {
        &self.contexts
    }

    pub fn queue_lengths(&self) -> (usize, usize, usize) {
        (self.red_q.len(), self.blue_q.len(), self.green_q.len())
    }

    pub fn count(&self) -> usize {
        self.contexts.len()
    }

    pub fn active_count(&self) -> usize {
        self.contexts
            .values()
            .filter(|c| c.state != ContextState::Terminated)
            .count()
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priority_strict() {
        let mut sched = Scheduler::new();
        let g1 = sched.create_context(Priority::Green, 0, 0);
        let g2 = sched.create_context(Priority::Green, 32, 0);
        let r1 = sched.create_context(Priority::Red, 64, 0);
        let b1 = sched.create_context(Priority::Blue, 96, 0);

        // Deve escolher RED primeiro, depois BLUE, depois GREEN FIFO
        assert_eq!(sched.pick_next().unwrap(), r1);
        sched.yield_current(); // RED volta? mas vamos consumir
        // Para teste, removemos r1
        sched.remove(r1);
        assert_eq!(sched.pick_next().unwrap(), b1);
        sched.remove(b1);
        assert_eq!(sched.pick_next().unwrap(), g1);
        sched.remove(g1);
        assert_eq!(sched.pick_next().unwrap(), g2);
    }

    #[test]
    fn test_fork_clones_regs() {
        let mut parent = Context::new(1, Priority::Green, 0x1000, 0);
        parent.set_reg(0, 42).unwrap();
        parent.set_reg(1, 0xDEADBEEF).unwrap();
        let child = parent.fork(2, Priority::Blue);
        assert_eq!(child.reg(0).unwrap(), 42);
        assert_eq!(child.priority, Priority::Blue);
        assert_eq!(child.pc, 0x1000 + 32);
    }
}
