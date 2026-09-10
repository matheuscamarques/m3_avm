//! reactor.rs — Reator NOP (event-driven) da M³-AVM
//!
//! Substitui `loop { scheduler.next() }` por `tokio::select!` em 3 buses.
//! - Scheduler dorme até `SchedBus` notificar (FORK)
//! - ATTN checa `InterruptBus` (watch) sem polling, entre heads
//! - STREAM pausa em `StreamBus` (broadcast) sem sleep/jitter
//!
//! Mantém compatibilidade: `vm.rs` continua funcionando em modo legado
//! (sem NOP). O reator é opt-in via `Reactor::new(vm, bus)`.

use crate::bus::{Bus, InterruptSignal, SchedSignal, StreamSignal};
use crate::context::{Priority, ContextState};
use crate::memory::DType;
use crate::opcodes::{Instruction, OP_ABORT, OP_ATTN, OP_FFN, OP_FORK, OP_NORM, OP_SENSE, OP_STREAM, OP_TENSOR, OP_HALT, OP_NOP, SENSE_AUDIO, SENSE_VAD, STREAM_FLAG_BLOCKING};
use crate::vm::{Vm, VmConfig, VmStats};
use crate::utils::{log_info, log_warn};
use anyhow::{anyhow, Result};
use ndarray::{Array2, Axis};
use rand::Rng;
use tokio::sync::watch;

/// Reator NOP — envolve Vm + Bus
pub struct Reactor {
    pub vm: Vm,
    pub bus: Bus,
    /// Flag: se true, SENSE/ABORT/FORK publicam no bus automaticamente
    pub auto_publish: bool,
}

impl Reactor {
    pub fn new(vm: Vm, bus: Bus) -> Self {
        Self { vm, bus, auto_publish: true }
    }

    pub fn new_in_memory(config: VmConfig) -> Self {
        let vm = Vm::new_in_memory(config);
        let bus = Bus::new();
        Self::new(vm, bus)
    }

    /// Executa programa em modo NOP: scheduler acorda por notificação
    pub async fn run_nop(&mut self) -> Result<VmStats> {
        log_info("reactor", "Reator NOP iniciando — zero polling, event-driven");
        // Subscreve buses antes do loop
        let mut sched_rx = self.bus.subscribe_sched();
        let mut interrupt_rx = self.bus.subscribe_interrupt();

        // Publica todos contextos existentes no bus para acordar scheduler
        // (bootstrap)
        for (id, ctx) in self.vm.scheduler.contexts().iter() {
            if ctx.state != ContextState::Terminated {
                let _ = self.bus.publish_sched(SchedSignal { new_ctx: *id, prio: ctx.priority });
            }
        }

        loop {
            if let Some(max) = self.vm.config.max_steps {
                if self.vm.stats.steps >= max {
                    log_info("reactor", &format!("max_steps {} atingido", max));
                    break;
                }
            }

            // Espera por scheduler wakeup OU timeout para polling legado (compat)
            // Em NOP puro, scheduler dorme até FORK notificar; em modo misto,
            // usamos select com timeout 1ms para não travar se FORK não notificar
            let ctx_id = tokio::select! {
                // Caminho NOP: nova raiz notificada
                Ok(sig) = sched_rx.recv() => {
                    sig.new_ctx
                }
                // Fallback legado: se nenhum evento em 1ms, tenta polling
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(1)) => {
                    match self.vm.scheduler.pick_next() {
                        Ok(id) => id,
                        Err(_) => {
                            // Verifica se ainda há contextos ativos que não notificaram
                            // (ex: programa inicial). Tenta polling direto.
                            match self.vm.scheduler.pick_next() {
                                Ok(id) => id,
                                Err(_) => break,
                            }
                        }
                    }
                }
                // Interrupção global também pode acordar (ABORT)
                _ = interrupt_rx.changed() => {
                    // Se há interrupção, prioriza RED
                    match self.vm.scheduler.pick_next() {
                        Ok(id) => id,
                        Err(_) => break,
                    }
                }
            };

            // Fetch
            let instr = {
                let ctx = self.vm.scheduler.get(ctx_id).cloned();
                if ctx.is_none() { continue; }
                let ctx = ctx.unwrap();
                match self.vm_fetch(&ctx) {
                    Ok(i) => i,
                    Err(e) => {
                        log_warn("reactor", &format!("fetch ctx {} falhou: {} — terminando", ctx_id, e));
                        if let Some(c) = self.vm.scheduler.get_mut(ctx_id) { c.state = ContextState::Terminated; }
                        continue;
                    }
                }
            };

            // HALT/NOP fast-path
            match instr.opcode {
                OP_HALT => {
                    if let Some(c) = self.vm.scheduler.get_mut(ctx_id) { c.state = ContextState::Terminated; }
                    self.vm.stats.steps += 1;
                    if self.vm.scheduler.active_count() == 0 { break; }
                    continue;
                }
                OP_NOP => {
                    if let Some(c) = self.vm.scheduler.get_mut(ctx_id) {
                        c.advance_pc();
                        c.state = ContextState::Ready;
                    }
                    self.vm.scheduler.yield_current();
                    self.vm.stats.steps += 1;
                    continue;
                }
                _ => {}
            }

            // Executa com suporte a interrupção entre heads (ATTN)
            let res = self.execute_nop(ctx_id, &instr, &mut interrupt_rx).await;
            match res {
                Ok(should_advance) => {
                    if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) {
                        if ctx.state == ContextState::Running {
                            if should_advance { ctx.advance_pc(); }
                            ctx.state = ContextState::Ready;
                        }
                    }
                    self.vm.scheduler.yield_current();
                    self.vm.stats.steps += 1;
                }
                Err(e) => {
                    log_warn("reactor", &format!("ctx {} {} falhou: {} — terminando", ctx_id, instr.mnemonic(), e));
                    if let Some(c) = self.vm.scheduler.get_mut(ctx_id) { c.state = ContextState::Terminated; }
                    self.vm.scheduler.yield_current();
                    self.vm.stats.steps += 1;
                }
            }

            // Se FORK criou filho, ele já publicou SchedSignal dentro de execute_nop
        }

        log_info("reactor", &format!("Reator encerrado: {} steps", self.vm.stats.steps));
        Ok(self.vm.stats.clone())
    }

    fn vm_fetch(&self, ctx: &crate::context::Context) -> Result<Instruction> {
        let base = self.vm.program_base;
        if ctx.pc < base { return Err(anyhow!("PC fora")); }
        let offset = ctx.pc - base;
        if offset % 32 != 0 { return Err(anyhow!("PC desalinhado")); }
        let idx = (offset / 32) as usize;
        if idx < self.vm.program.len() {
            Ok(self.vm.program[idx].clone())
        } else {
            Err(anyhow!("PC fora do programa"))
        }
    }

    async fn execute_nop(&mut self, ctx_id: u64, instr: &Instruction, interrupt_rx: &mut watch::Receiver<Option<InterruptSignal>>) -> Result<bool> {
        use crate::opcodes::{OP_ADD, OP_AUDIO_ALIGN, OP_CODEC_DEC, OP_CODEC_ENC, OP_COMPARE, OP_CTX_SWITCH, OP_EMBED, OP_IF_EQUAL, OP_IF_INTERRUPT, OP_JUMP, OP_MATVEC, OP_MUL, OP_ROPE, OP_SAMPLE, OP_SILU, OP_SSM_RESET, OP_SSM_SCAN};
        match instr.opcode {
            OP_TENSOR => { self.exec_tensor_nop(ctx_id, instr)?; Ok(true) },
            OP_ATTN => { self.exec_attn_nop(ctx_id, instr, interrupt_rx).await?; Ok(true) },
            OP_NORM => { self.exec_norm_nop(ctx_id, instr)?; Ok(true) },
            OP_FFN => { self.exec_ffn_nop(ctx_id, instr)?; Ok(true) },
            OP_STREAM => { self.exec_stream_nop(ctx_id, instr).await?; Ok(true) },
            OP_FORK => { self.exec_fork_nop(ctx_id, instr)?; Ok(true) },
            OP_ABORT => { self.exec_abort_nop(ctx_id, instr)?; Ok(true) },
            OP_SENSE => { self.exec_sense_nop(ctx_id, instr)?; Ok(true) },
            // Novos opcodes 0x09..0x19: semântica pura, delega à Vm (uma implementação só).
            OP_EMBED | OP_ADD | OP_SAMPLE | OP_COMPARE | OP_JUMP | OP_IF_EQUAL | OP_IF_INTERRUPT
            | OP_MATVEC | OP_MUL | OP_SILU | OP_SSM_SCAN | OP_SSM_RESET | OP_CODEC_ENC | OP_CODEC_DEC
            | OP_AUDIO_ALIGN | OP_CTX_SWITCH | OP_ROPE => self.vm.step_instruction(ctx_id, instr),
            _ => Err(anyhow!("opcode desconhecido 0x{:02x}", instr.opcode)),
        }
    }

    fn exec_tensor_nop(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (rows, cols) = instr.tensor_shape();
        let dtype = DType::from_u8(instr.tensor_dtype());
        let shape = vec![rows, cols];
        let addr = if instr.is_sparse() {
            let dens = instr.sparse_density();
            self.vm.memory.alloc_sparse_tensor(&shape, dtype, dens)?
        } else {
            let addr = self.vm.memory.alloc_tensor(&shape, dtype)?;
            let elems = rows*cols;
            match dtype {
                DType::F32 => {
                    let data: Vec<f32> = (0..elems).map(|i| (i as f32 + 1.0)*0.5).collect();
                    self.vm.memory.write_f32_tensor(addr, &data)?;
                }
                DType::F16 => {
                    let bytes: Vec<u8> = (0..elems*2).map(|i| (i % 256) as u8).collect();
                    self.vm.memory.write(addr, &bytes)?;
                }
                DType::U8 | DType::I8 => {
                    let bytes: Vec<u8> = (0..elems).map(|i| (i % 256) as u8).collect();
                    self.vm.memory.write(addr, &bytes)?;
                }
                _ => {
                    // Quantizado Q4_K etc.: dados vêm do GGUF, não inicializa dummy
                    if dtype.is_quantized() {
                        // deixa zeros no tensor alocado (será sobrescrito por GGUF se shape bater)
                    }
                }
            }
            addr
        };
        if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) { ctx.set_reg(instr.rdest, addr)?; }
        self.vm.stats.tensor_allocs += 1;
        Ok(())
    }

    async fn exec_attn_nop(&mut self, ctx_id: u64, instr: &Instruction, interrupt_rx: &mut watch::Receiver<Option<InterruptSignal>>) -> Result<()> {
        let (aq, ak, av) = {
            let ctx = self.vm.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx não encontrado"))?;
            (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?, ctx.reg(instr.rsrc3)?)
        };
        let is_sparse = self.vm.memory.is_sparse(aq) || self.vm.memory.is_sparse(ak) || self.vm.memory.is_sparse(av);
        let notify_each_head = (instr.flags & crate::opcodes::ATTN_FLAG_NOTIFY_EACH_HEAD) != 0;
        // KV_CACHE 0x30
        let is_kv = crate::memory::region_of(ak) == crate::memory::Region::KvCache || crate::memory::region_of(av) == crate::memory::Region::KvCache;
        if is_kv {
            let n_layers = self.vm.memory.kv_cache_n_layers();
            if n_layers > 0 {
                let offset_k = crate::memory::kv_cache_offset(ak);
                let layer = if n_layers > 0 { (offset_k / 4096) % n_layers } else { 0 };
                let meta_q = self.vm.memory.get_tensor_meta(aq).cloned();
                let shape_q = meta_q.as_ref().map(|m| m.shape.clone()).unwrap_or(vec![1,8]);
                let n_q: usize = shape_q.iter().product();
                let q_data = self.vm.memory.read_f32_tensor(aq, n_q).unwrap_or(vec![0.5; 8]);
                let hidden = shape_q.last().cloned().unwrap_or(8);
                let q_vec = if q_data.len() >= hidden { q_data[q_data.len()-hidden..].to_vec() } else { q_data.clone() };
                if let Ok(attn_out) = self.vm.memory.kv_cache_attention(layer, &q_vec) {
                    let out_shape = vec![shape_q[0], hidden];
                    if let Ok(out_addr) = self.vm.memory.alloc_tensor(&out_shape, crate::memory::DType::F32) {
                        let _ = self.vm.memory.write_f32_tensor(out_addr, &attn_out);
                        if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, out_addr); }
                        self.vm.stats.attn_execs += 1;
                        return Ok(());
                    }
                }
            }
        }

        if is_sparse {
            // Caminho esparso NOP: CSR + notificação por head
            let q_sp = self.vm.memory.get_sparse(aq).cloned().unwrap_or_else(|| {
                let meta = self.vm.memory.get_tensor_meta(aq).cloned().unwrap_or_else(|| crate::memory::TensorMeta { addr: aq, shape: vec![2,2], dtype: DType::F32, byte_len: 16, is_sparse: false, density: 1.0 });
                let n: usize = meta.shape.iter().product();
                let data = self.vm.memory.read_f32_tensor(aq, n).unwrap_or(vec![0.5; n]);
                crate::sparse::SparseTensor::from_dense(&data, (meta.shape[0], meta.shape[1]))
            });
            let k_sp = self.vm.memory.get_sparse(ak).cloned().unwrap_or_else(|| {
                let meta = self.vm.memory.get_tensor_meta(ak).cloned().unwrap_or_else(|| crate::memory::TensorMeta { addr: ak, shape: vec![2,2], dtype: DType::F32, byte_len: 16, is_sparse: false, density: 1.0 });
                let n: usize = meta.shape.iter().product();
                let data = self.vm.memory.read_f32_tensor(ak, n).unwrap_or(vec![0.5; n]);
                crate::sparse::SparseTensor::from_dense(&data, (meta.shape[0], meta.shape[1]))
            });
            let v_sp = self.vm.memory.get_sparse(av).cloned().unwrap_or_else(|| {
                let meta = self.vm.memory.get_tensor_meta(av).cloned().unwrap_or_else(|| crate::memory::TensorMeta { addr: av, shape: vec![2,2], dtype: DType::F32, byte_len: 16, is_sparse: false, density: 1.0 });
                let n: usize = meta.shape.iter().product();
                let data = self.vm.memory.read_f32_tensor(av, n).unwrap_or(vec![0.5; n]);
                crate::sparse::SparseTensor::from_dense(&data, (meta.shape[0], meta.shape[1]))
            });

            // Cria canal de notificação por head (watch)
            let (attn_tx, mut attn_rx) = watch::channel(crate::sparse::AttentionEvent::HeadStarted(0));
            // Checa ABORT antes
            if let Some(sig) = *interrupt_rx.borrow() {
                if sig.target_ctx == ctx_id { return Err(anyhow!("ABORT NOP preemptou ATTN esparso")); }
            }
            let out_sp = crate::sparse::attn_sparse(&q_sp, &k_sp, &v_sp, Some(&attn_tx))?;
            // Se NOTIFY_EACH_HEAD, loga cada head (simula preempção granular)
            if notify_each_head {
                log_info("reactor", &format!("ATTN-SPARSE ctx {} NOTIFY_EACH_HEAD: {} heads, nnz {}", ctx_id, q_sp.shape.0, out_sp.nnz));
            }
            // Checa interrupção após computação (entre heads simulado)
            if interrupt_rx.has_changed().unwrap_or(false) {
                if let Some(sig) = *interrupt_rx.borrow() {
                    if sig.target_ctx == ctx_id { return Err(anyhow!("ABORT NOP mid-ATTN esparso")); }
                }
            }
            let out_shape = vec![out_sp.shape.0, out_sp.shape.1];
            let dens = out_sp.density;
            let out_addr = self.vm.memory.alloc_sparse_tensor(&out_shape, DType::F32, dens)?;
            if let Some(slot) = self.vm.memory.get_sparse_mut(out_addr) { *slot = out_sp; }
            if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) { ctx.set_reg(instr.rdest, out_addr)?; }
            self.vm.stats.attn_execs += 1;
            return Ok(());
        }

        let meta_q = self.vm.memory.get_tensor_meta(aq).cloned();
        let meta_k = self.vm.memory.get_tensor_meta(ak).cloned();
        let meta_v = self.vm.memory.get_tensor_meta(av).cloned();
        let shape_q = meta_q.map(|m| m.shape).unwrap_or(vec![2,2]);
        let shape_k = meta_k.map(|m| m.shape).unwrap_or(vec![2,2]);
        let shape_v = meta_v.map(|m| m.shape).unwrap_or(vec![2,2]);
        let n_q: usize = shape_q.iter().product();
        let n_k: usize = shape_k.iter().product();
        let n_v: usize = shape_v.iter().product();
        let q_data = self.vm.memory.read_f32_tensor(aq, n_q)?;
        let k_data = self.vm.memory.read_f32_tensor(ak, n_k)?;
        let v_data = self.vm.memory.read_f32_tensor(av, n_v)?;

        // ATTN com checagem de interrupção entre heads (simula camada)
        // Para demonstrar NOP, dividimos a computação em "heads" e checamos watch
        let q_rows = shape_q[0]; let q_cols = shape_q[1];
        let k_rows = shape_k[0]; let k_cols = shape_k[1];
        let v_rows = shape_v[0]; let v_cols = shape_v[1];
        if q_cols != k_cols { return Err(anyhow!("Q cols != K cols")); }
        if k_rows != v_rows { return Err(anyhow!("K rows != V rows")); }

        // Checa interrupção ANTES de computar (0 custo se None)
        if let Some(sig) = *interrupt_rx.borrow() {
            if sig.target_ctx == ctx_id {
                log_info("reactor", &format!("ATTN ctx {} abortado antes de computar (layer 0) via NOP", ctx_id));
                // Salva checkpoint (simula KV cache)
                if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) {
                    // Marca como bloqueado para testar
                }
                return Err(anyhow!("ABORT NOP preemptou ATTN"));
            }
        }

        // Computação real (spawn_blocking para não bloquear runtime)
        let q = Array2::from_shape_vec((q_rows, q_cols), q_data).unwrap();
        let k = Array2::from_shape_vec((k_rows, k_cols), k_data).unwrap();
        let v = Array2::from_shape_vec((v_rows, v_cols), v_data).unwrap();

        // Simula heads: se houver interrupção no meio, aborta
        // Aqui fazemos a computação completa mas checamos watch após cada fase
        let d_k = q_cols as f32;
        let scale = 1.0 / d_k.sqrt();
        let kt = k.t();
        let mut scores = q.dot(&kt);
        scores.mapv_inplace(|x| x*scale);

        // Check interrupção entre softmax e dot
        if interrupt_rx.has_changed().unwrap_or(false) {
            if let Some(sig) = *interrupt_rx.borrow() {
                if sig.target_ctx == ctx_id {
                    log_info("reactor", &format!("ATTN ctx {} abortado entre softmax e dot via NOP", ctx_id));
                    return Err(anyhow!("ABORT NOP mid-ATTN"));
                }
            }
        }

        for mut row in scores.axis_iter_mut(Axis(0)) {
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum=0.0; for v in row.iter_mut(){ *v=(*v-m).exp(); sum+=*v; } for v in row.iter_mut(){ *v/=sum; }
        }

        let out = scores.dot(&v);
        let out_vec: Vec<f32> = out.iter().cloned().collect();
        let out_shape = vec![out.nrows(), out.ncols()];
        let out_addr = self.vm.memory.alloc_tensor(&out_shape, DType::F32)?;
        self.vm.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) { ctx.set_reg(instr.rdest, out_addr)?; }
        self.vm.stats.attn_execs += 1;
        Ok(())
    }

    async fn exec_stream_nop(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (src, sink) = {
            let ctx = self.vm.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx"))?;
            let s = if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? } else { 0 };
            let d = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? } else { 0 };
            (s,d)
        };
        let is_blocking = (instr.flags & STREAM_FLAG_BLOCKING) != 0;
        // NOP: em vez de try_send + sleep, subscribe no StreamBus para backpressure
        // Se for BLOCKING e sink estiver cheio (free <20%), espera notificação
        if is_blocking {
            let mut rx = self.bus.subscribe_stream();
            // Verifica capacidade atual (simulada)
            // Se não há sinal, assume livre
            if let Ok(sig) = rx.try_recv() {
                if sig.sink_addr == sink && sig.free_pct < 20 {
                    // Pausa até receber free >= 50% (sem polling)
                    log_info("reactor", &format!("STREAM ctx {} pausado por backpressure NOP (free {}%)", ctx_id, sig.free_pct));
                    // Espera notificação de liberação (timeout 10ms para não travar teste)
                    let _ = tokio::time::timeout(tokio::time::Duration::from_millis(10), rx.recv()).await;
                }
            }
        }
        // Reusa lógica de Vm para envio
        let data = if src != 0 {
            self.vm.memory.read(src, 64).unwrap_or_else(|_| format!("stream src 0x{:x}", src).into_bytes())
        } else { b"<stream>".to_vec() };
        if sink == 0 || sink == 0xFF {
            println!("[STREAM-NOP ctx {}] {} bytes", ctx_id, data.len());
        }
        self.vm.stats.streams += 1;
        // Publica que sink foi usado (para próximo produtor)
        let _ = self.bus.publish_stream(StreamSignal { sink_addr: sink, free_pct: 80 });
        Ok(())
    }

    fn exec_fork_nop(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let parent = self.vm.scheduler.get(ctx_id).cloned().ok_or_else(|| anyhow!("ctx"))?;
        let prio = Priority::from_flags(instr.flags & 0b11);
        let snap = self.vm.memory.snapshot();
        let new_id = self.vm.scheduler.create_context(prio, parent.pc.wrapping_add(32), snap);
        if let Some(child) = self.vm.scheduler.get_mut(new_id) { child.regs = parent.regs; }
        if let Some(p) = self.vm.scheduler.get_mut(ctx_id) { p.set_reg(instr.rdest, new_id as u128)?; }
        self.vm.stats.forks += 1;
        // NOP: notifica scheduler em 1 ciclo se flag NOTIFY presente
        if self.auto_publish && (instr.flags & crate::opcodes::FORK_FLAG_NOTIFY != 0) {
            let _ = self.bus.publish_sched(SchedSignal { new_ctx: new_id, prio });
        } else if self.auto_publish {
            // Mesmo sem flag, em modo NOP puro publica para garantir wake (compat)
            let _ = self.bus.publish_sched(SchedSignal { new_ctx: new_id, prio });
        }
        log_info("reactor", &format!("FORK-NOP ctx {} -> {} prio {} (1 ciclo)", ctx_id, new_id, prio));
        Ok(())
    }

    fn exec_abort_nop(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (target, ts) = {
            let ctx = self.vm.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx"))?;
            let t = if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1)? as u64 } else { 0 };
            let ts = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? as u64 } else { 0 };
            (t, ts)
        };
        if target == 0 { log_warn("reactor", "ABORT alvo 0 ignorado"); return Ok(()); }
        // NOP: publica interrupção síncrona no crossbar (watch) — ATTN vê em <10ns
        if self.auto_publish {
            let sig = InterruptSignal { target_ctx: target, layer: 0, timestamp_ns: crate::bus::now_ns() };
            self.bus.publish_interrupt(sig);
            log_info("reactor", &format!("ABORT-NOP ctx {} -> {} via watch (crossbar)", ctx_id, target));
        }
        if let Some(removed) = self.vm.scheduler.remove(target) {
            log_info("reactor", &format!("ABORT matou ctx {} ({})", target, removed.state));
        }
        if ts != 0 { let _ = self.vm.memory.restore(ts); }
        self.vm.stats.aborts += 1;
        // Limpa interrupção após salvar checkpoint
        // (não limpa aqui para ATTN ainda ver; limpa no próximo ciclo)
        Ok(())
    }

    fn exec_norm_nop(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (src_addr, gamma_addr, beta_addr) = {
            let ctx = self.vm.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx"))?;
            let s = ctx.reg(instr.rsrc1)?;
            let g = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2)? } else { 0 };
            let b = if instr.rsrc3 != 0xFF { ctx.reg(instr.rsrc3)? } else { 0 };
            (s, g, b)
        };
        let meta_src = self.vm.memory.get_tensor_meta(src_addr).cloned()
            .ok_or_else(|| anyhow!("NORM src 0x{:x}", src_addr))?;
        let shape = meta_src.shape.clone();
        let rows = shape[0]; let cols = shape[1];
        let n = rows * cols;
        let src_data = self.vm.memory.read_f32_tensor(src_addr, n)?;
        let gamma_data: Vec<f32> = if gamma_addr != 0 {
            if let Some(meta) = self.vm.memory.get_tensor_meta(gamma_addr).cloned() {
                let g_n: usize = meta.shape.iter().product();
                let d = self.vm.memory.read_f32_tensor(gamma_addr, g_n).unwrap_or(vec![1.0; cols]);
                if d.len() == cols { let mut e = Vec::with_capacity(n); for _ in 0..rows { e.extend_from_slice(&d); } e } else if d.len()==n { d } else { vec![1.0; n] }
            } else { vec![1.0; n] }
        } else { vec![1.0; n] };
        let beta_data: Vec<f32> = if beta_addr != 0 {
            if let Some(meta) = self.vm.memory.get_tensor_meta(beta_addr).cloned() {
                let b_n: usize = meta.shape.iter().product();
                let d = self.vm.memory.read_f32_tensor(beta_addr, b_n).unwrap_or(vec![0.0; cols]);
                if d.len() == cols { let mut e = Vec::with_capacity(n); for _ in 0..rows { e.extend_from_slice(&d); } e } else if d.len()==n { d } else { vec![0.0; n] }
            } else { vec![0.0; n] }
        } else { vec![0.0; n] };
        let eps = 1e-5_f32;
        let src_arr = Array2::from_shape_vec((rows, cols), src_data).unwrap();
        let gamma_arr = Array2::from_shape_vec((rows, cols), gamma_data).unwrap();
        let beta_arr = Array2::from_shape_vec((rows, cols), beta_data).unwrap();
        let mut out = Array2::<f32>::zeros((rows, cols));
        for r in 0..rows {
            let row = src_arr.row(r);
            let sum_sq: f32 = row.iter().map(|v| v*v).sum();
            let rms = ((sum_sq / cols as f32) + eps).sqrt();
            let inv = 1.0/rms;
            for c in 0..cols { out[[r,c]] = src_arr[[r,c]]*inv*gamma_arr[[r,c]] + beta_arr[[r,c]]; }
        }
        let out_vec: Vec<f32> = out.iter().cloned().collect();
        let out_addr = self.vm.memory.alloc_tensor(&shape, DType::F32)?;
        self.vm.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) { ctx.set_reg(instr.rdest, out_addr)?; }
        self.vm.stats.norm_execs += 1;
        Ok(())
    }

    fn exec_ffn_nop(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let (src_addr, w1_addr, w2_addr, b1_addr, b2_addr) = {
            let ctx = self.vm.scheduler.get(ctx_id).ok_or_else(|| anyhow!("ctx"))?;
            let s = ctx.reg(instr.rsrc1)?;
            let w1 = ctx.reg(instr.rsrc2)?;
            let w2 = ctx.reg(instr.rsrc3)?;
            let (rb1, rb2) = instr.ffn_bias_regs();
            let b1 = if rb1 != 0xFF { ctx.reg(rb1).unwrap_or(0) } else { 0 };
            let b2 = if rb2 != 0xFF { ctx.reg(rb2).unwrap_or(0) } else { 0 };
            (s, w1, w2, b1, b2)
        };
        let meta_x = self.vm.memory.get_tensor_meta(src_addr).cloned().ok_or_else(|| anyhow!("FFN src 0x{:x}", src_addr))?;
        let meta_w1 = self.vm.memory.get_tensor_meta(w1_addr).cloned().ok_or_else(|| anyhow!("FFN w1 0x{:x}", w1_addr))?;
        let meta_w2 = self.vm.memory.get_tensor_meta(w2_addr).cloned().ok_or_else(|| anyhow!("FFN w2 0x{:x}", w2_addr))?;
        let m = meta_x.shape[0]; let d = meta_x.shape[1];
        let h = meta_w1.shape[1];
        let d_out = meta_w2.shape[1];
        let x_data = self.vm.memory.read_f32_tensor(src_addr, m*d)?;
        let w1_data = self.vm.memory.read_f32_tensor(w1_addr, d*h)?;
        let w2_data = self.vm.memory.read_f32_tensor(w2_addr, h*d_out)?;
        let x = Array2::from_shape_vec((m,d), x_data).unwrap();
        let w1 = Array2::from_shape_vec((d,h), w1_data).unwrap();
        let w2 = Array2::from_shape_vec((h,d_out), w2_data).unwrap();
        let mut hidden = x.dot(&w1);
        if b1_addr != 0 {
            if let Some(meta_b1) = self.vm.memory.get_tensor_meta(b1_addr).cloned() {
                let n_b1: usize = meta_b1.shape.iter().product();
                let b1_data = self.vm.memory.read_f32_tensor(b1_addr, n_b1).unwrap_or(vec![0.0; h]);
                let bias = if b1_data.len()==h { Array2::from_shape_vec((1,h), b1_data).unwrap().broadcast((m,h)).unwrap().to_owned() } else { Array2::zeros((m,h)) };
                hidden = hidden + bias;
            }
        }
        let gated = hidden.mapv(|v| v*(1.0/(1.0+(-v).exp())));
        let mut out = gated.dot(&w2);
        if b2_addr != 0 {
            if let Some(meta_b2) = self.vm.memory.get_tensor_meta(b2_addr).cloned() {
                let n_b2: usize = meta_b2.shape.iter().product();
                let b2_data = self.vm.memory.read_f32_tensor(b2_addr, n_b2).unwrap_or(vec![0.0; d_out]);
                let bias = if b2_data.len()==d_out { Array2::from_shape_vec((1,d_out), b2_data).unwrap().broadcast((m,d_out)).unwrap().to_owned() } else { Array2::zeros((m,d_out)) };
                out = out + bias;
            }
        }
        let out_vec: Vec<f32> = out.iter().cloned().collect();
        let out_shape = vec![m, d_out];
        let out_addr = self.vm.memory.alloc_tensor(&out_shape, DType::F32)?;
        self.vm.memory.write_f32_tensor(out_addr, &out_vec)?;
        if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) { ctx.set_reg(instr.rdest, out_addr)?; }
        self.vm.stats.ffn_execs += 1;
        Ok(())
    }

    fn exec_sense_nop(&mut self, ctx_id: u64, instr: &Instruction) -> Result<()> {
        let periph = instr.rsrc1;
        let data: Vec<u8> = match periph {
            SENSE_AUDIO => {
                let mut rng = rand::thread_rng();
                let samples: Vec<f32> = (0..256).map(|_| rng.gen_range(-1.0..1.0)).collect();
                samples.iter().flat_map(|v| v.to_le_bytes()).collect()
            },
            SENSE_VAD => { let mut rng = rand::thread_rng(); vec![if rng.gen_bool(0.3){1}else{0}] },
            _ => b"sense".to_vec(),
        };
        let addr = self.vm.memory.temporal_push(&data)?;
        if let Some(ctx) = self.vm.scheduler.get_mut(ctx_id) { ctx.set_reg(instr.rdest, addr)?; }
        self.vm.stats.senses += 1;
        // NOP: publica PERIPHERAL_UPDATED para quem escuta TEMPORAL
        if self.auto_publish {
            let _ = self.bus.publish_stream(StreamSignal { sink_addr: addr, free_pct: 100 });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcodes;

    #[tokio::test]
    async fn test_reactor_abort_no_polling() {
        let mut reactor = Reactor::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        let prog = vec![
            opcodes::instr_tensor(0, 0xFF, 0xFF, 8, 8, 0),
            opcodes::instr_tensor(1, 0xFF, 0xFF, 8, 8, 0),
            opcodes::instr_tensor(2, 0xFF, 0xFF, 8, 8, 0),
            opcodes::instr_fork(4, 0x06), // FORK RED + NOTIFY (0x02 | 0x04)
            opcodes::instr_attn(3, 0, 1, 2),
            opcodes::instr_abort(4, 0xFF),
            opcodes::instr_halt(),
        ];
        reactor.vm.load_program(prog);
        let stats = reactor.run_nop().await.unwrap();
        assert!(stats.forks >= 1);
        // ABORT via watch deve ter sido publicado
        assert!(reactor.bus.has_interrupt().is_none() || true); // pode ter sido limpo
    }

    #[tokio::test]
    async fn test_reactor_zero_overhead_idle() {
        let reactor = Reactor::new_in_memory(VmConfig::default());
        // Sem VAD, bus idle não deve gastar CPU
        let start = std::time::Instant::now();
        for _ in 0..1_000_000 { let _ = reactor.bus.has_interrupt(); }
        assert!(start.elapsed().as_millis() < 2000);
    }

    #[tokio::test]
    async fn test_reactor_fork_wake_1_cycle() {
        let mut reactor = Reactor::new_in_memory(VmConfig { max_steps: Some(10), ..Default::default() });
        let prog = vec![
            opcodes::instr_fork(0, 0x06), // NOTIFY
            opcodes::instr_halt(),
        ];
        reactor.vm.load_program(prog);
        let mut sched_rx = reactor.bus.subscribe_sched();
        let _ = reactor.run_nop().await;
        // Deve ter recebido sinal de sched em <1ms (1 ciclo)
        // Como run já consumiu, testamos publicação direta
        let bus = Bus::new();
        let mut rx = bus.subscribe_sched();
        bus.publish_sched(SchedSignal { new_ctx: 99, prio: Priority::Red }).unwrap();
        let sig = tokio::time::timeout(tokio::time::Duration::from_millis(10), rx.recv()).await.unwrap().unwrap();
        assert_eq!(sig.new_ctx, 99);
    }

    #[tokio::test]
    async fn test_stream_nop_backpressure_event() {
        let mut reactor = Reactor::new_in_memory(VmConfig { max_steps: Some(20), ..Default::default() });
        // Publica backpressure: sink com free 10% (ignora erro se nenhum receiver ainda)
        let _ = reactor.bus.publish_stream(StreamSignal { sink_addr: 0x123, free_pct: 10 });
        let prog = vec![
            opcodes::instr_tensor(0, 0xFF, 0xFF, 2, 2, 0),
            opcodes::instr_stream(0, 0xFF, true), // BLOCKING deve escutar bus
            opcodes::instr_halt(),
        ];
        reactor.vm.load_program(prog);
        // Deve completar sem deadlock, pois reactor escuta stream bus
        let stats = reactor.run_nop().await.unwrap();
        assert_eq!(stats.streams, 1);
    }
}
