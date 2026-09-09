//! main.rs — CLI da M³-AVM
//!
//! Uso:
//!   cargo run -- run examples/test.m3asm         # assembly textual
//!   cargo run -- run program.m3bin               # binário 32 bytes/instr
//!   cargo run -- bench --nops 1000000            # mede IPS
//!   cargo run -- asm --help                      # ajuda do assembler

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

pub mod bus;
pub mod context;
pub mod memory;
pub mod opcodes;
pub mod reactor;
pub mod rollback;
pub mod sparse;
pub mod stt;
pub mod tokenizer;
pub mod gguf;
pub mod quant;
pub mod inference;
pub mod matvec;
pub mod matvec_quant;
pub mod mimi;
pub mod moshi;
pub mod ssm;
#[cfg(feature = "wgpu")]
pub mod inference_gpu;
pub mod tui;
pub mod utils;
pub mod vm;
pub mod memory_wgpu;
pub mod asm_emitter;

use opcodes::{assemble, Instruction, INSTR_SIZE, OP_HALT, OP_NOP};
use vm::{Vm, VmConfig};
use bus::{Bus, InterruptSignal};
use std::sync::mpsc as std_mpsc;

#[derive(Parser)]
#[command(name = "m3_avm", version, about = "M³-AVM — Máquina Abstrata de Matheus de Camargo Marques", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Executa um programa (.m3asm ou .m3bin) — tese: SENSE/FORK/ABORT inseparáveis de ATTN/NORM/FFN
    Run {
        /// Arquivo de entrada (.m3asm ou .m3bin)
        file: PathBuf,
        /// Limite de passos (default: ilimitado até HALT)
        #[arg(long, default_value = "1000000")]
        max_steps: u64,
        /// Tracing por instrução
        #[arg(long)]
        trace: bool,
        /// Caminho para modelo GGUF (mmap zero-copy em PERSISTENTE 0x20)
        #[arg(long, value_name = "FILE")]
        model: Option<PathBuf>,
        /// Modo interativo: qualquer tecla durante geração = ABORT+rollback+injeção (tese)
        #[arg(long)]
        interactive: bool,
        /// Tamanho PERSISTENTE em MiB (só informativo; tamanho fixo 64 MiB em dev)
        #[arg(long, default_value_t = 64)]
        persistent_mib: usize,
        /// Modo real inference: usa pesos f32/Q4_K do GGUF para gerar texto real (lento, CPU)
        #[arg(long)]
        real: bool,
        /// Gera assembly M³ desenrolado a partir do modelo GGUF e escreve em <FILE> (não executa)
        #[arg(long, value_name = "FILE")]
        emit_asm: Option<PathBuf>,
    },
    /// Monta .m3asm -> .m3bin
    Assemble {
        /// Arquivo .m3asm
        input: PathBuf,
        /// Saída .m3bin (default: input com extensão trocada)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Desmonta .m3bin -> texto
    Disassemble {
        file: PathBuf,
    },
    /// Benchmark de throughput (NOPs)
    Bench {
        /// Número de NOPs
        #[arg(long, default_value = "1000000")]
        nops: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Run { file, max_steps, trace, model, interactive, persistent_mib, real, emit_asm } => run_file(file, max_steps, trace, model, interactive, persistent_mib, real, emit_asm).await?,
        Commands::Assemble { input, output } => assemble_file(input, output)?,
        Commands::Disassemble { file } => disassemble_file(file)?,
        Commands::Bench { nops } => bench(nops).await?,
    }

    Ok(())
}

async fn run_file(path: PathBuf, max_steps: u64, trace: bool, model: Option<PathBuf>, interactive: bool, persistent_mib: usize, real: bool, emit_asm: Option<PathBuf>) -> Result<()> {
    // --emit-asm: caminho separado — gera assembly desenrolado e sai, sem tocar VM/interpreter
    if let Some(emit_path) = emit_asm {
        let model_path = model.as_ref().ok_or_else(|| anyhow!("--emit-asm precisa de --model <GGUF>"))?;
        let model_str = model_path.to_str().ok_or_else(|| anyhow!("model path inválido"))?;
        let inf = crate::inference::RealInference::new(model_str)?;
        let emitter = crate::asm_emitter::AsmEmitter::new(&inf.config);
        let text = emitter.emit_to_string();
        if let Some(parent) = emit_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(&emit_path, &text)?;
        println!(";; --emit-asm: {} instruções estimadas -> {} ({} bytes)", emitter.estimate_instr_count(), emit_path.display(), text.len());
        println!(";; Modelo: {} arch={} hidden={} layers={} vocab={}", model_path.display(), inf.config.arch, inf.config.hidden, inf.config.n_layers, inf.config.vocab);
        println!(";; Execute: cargo run -- run {} --model {} {}", emit_path.display(), model_path.display(), if interactive { "--interactive" } else { "" });
        // valida que assembly monta
        let prog = assemble(&text)?;
        println!(";; Validado: {} instruções montáveis", prog.len());
        return Ok(());
    }
    if !path.exists() {
        return Err(anyhow!("arquivo não encontrado: {}", path.display()));
    }

    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let prog: Vec<Instruction> = match ext {
        "m3asm" | "asm" | "s" => {
            let text = fs::read_to_string(&path)?;
            println!(";; Assembling {} ({} bytes texto)", path.display(), text.len());
            let prog = assemble(&text)?;
            println!(";; -> {} instruções", prog.len());
            prog
        }
        "m3bin" | "bin" => {
            let bytes = fs::read(&path)?;
            if bytes.len() % INSTR_SIZE != 0 {
                return Err(anyhow!(".m3bin tamanho inválido: {} bytes", bytes.len()));
            }
            let mut prog = Vec::new();
            for chunk in bytes.chunks_exact(INSTR_SIZE) {
                prog.push(Instruction::decode(chunk)?);
            }
            println!(";; Loaded {} instruções de {}", prog.len(), path.display());
            prog
        }
        _ => {
            // Tenta detectar por conteúdo: se contém texto legível, assume asm
            let bytes = fs::read(&path)?;
            if bytes.len() % INSTR_SIZE == 0 && is_probably_bin(&bytes) {
                let mut prog = Vec::new();
                for chunk in bytes.chunks_exact(INSTR_SIZE) {
                    prog.push(Instruction::decode(chunk)?);
                }
                prog
            } else {
                let text = String::from_utf8(bytes).map_err(|_| anyhow!("arquivo não é utf8 nem binário válido"))?;
                assemble(&text)?
            }
        }
    };

    let persistent_bytes = persistent_mib * 1024 * 1024;
    if persistent_mib != 64 {
        println!(";; persistent_mib={} MiB solicitado", persistent_mib);
    }

    let config = VmConfig {
        max_steps: Some(max_steps),
        enable_tracing: trace,
        ips_target: 1_000_000,
    };

    // Tenta criar VM com persistência; se falhar, cai para in-memory
    let mut vm = match Vm::new_with_persistent_size(config.clone(), persistent_bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("warn: falha ao criar persistência {}MiB ({}), usando memória volátil", persistent_mib, e);
            Vm::new_in_memory(config.clone())
        }
    };

    // Carrega modelo GGUF se solicitado (mmap zero-copy em PERSISTENTE 0x20)
    let model_for_tok = model.clone();
    if let Some(model_path) = model {
        println!("Carregando modelo: {}", model_path.display());
        match vm.memory.load_gguf_model(model_path.to_str().unwrap_or("")) {
            Ok(base) => {
                println!("Modelo mapeado em PERSISTENTE 0x{:032x}", base);
                // TENSORs no programa devem apontar para offsets a partir de base; já simulado via alloc
            }
            Err(e) => {
                eprintln!("erro ao carregar modelo: {} — continuando sem modelo (para teste use arquivo dummy)", e);
                if !interactive {
                    return Err(e);
                }
            }
        }
        // Carrega tokenizer do GGUF (ou mock)
        let tok = crate::tokenizer::M3Tokenizer::from_gguf_or_mock(model_path.to_str().unwrap_or(""));
        println!("[tokenizer] vocab {} tokens", tok.vocab_size());
        vm.tokenizer = Some(tok);
    } else {
        // Sem modelo, usa mock
        vm.tokenizer = Some(crate::tokenizer::M3Tokenizer::mock());
    }
    // Se model_for_tok existe mas vm.tokenizer já set, ok; se não, tenta de novo (para caso de interactive sem model)
    if vm.tokenizer.is_none() {
        if let Some(ref p) = model_for_tok {
            vm.tokenizer = Some(crate::tokenizer::M3Tokenizer::from_gguf_or_mock(p.to_str().unwrap_or("")));
        }
    }

    // Modo interativo da tese: SENSE no topo + FORK a cada N + ABORT rollback (stdin)
    let stats = if interactive {
        if real {
            println!("\n=== M³-AVM REAL INFERENCE (f32/Q4_K) ===");
            println!("Modelo: {:?} | tokenizer {} tokens", model_for_tok.as_ref(), vm.tokenizer.as_ref().map(|t| t.vocab_size()).unwrap_or(0));
            println!("Digite prompt inicial + ENTER. Durante geração, digite novo prompt + ENTER para rollback.\n");
            let gguf_path = model_for_tok.as_ref().map(|p| p.to_str().unwrap_or("").to_string()).unwrap_or_default();
            if gguf_path.is_empty() { return Err(anyhow!("--real precisa de --model GGUF")); }
            #[cfg_attr(not(feature = "wgpu"), allow(unused_mut))]
            let mut real_inf = crate::inference::RealInference::new(&gguf_path)?;
            if real_inf.config.is_mamba() {
                println!("[mamba] arch={} layers={} hidden={} d_inner={} d_state={} d_conv={} dt_rank={} rms={} — forward_one_mamba + estado constante",
                    real_inf.config.arch, real_inf.config.n_layers, real_inf.config.hidden,
                    real_inf.config.d_inner, real_inf.config.d_state, real_inf.config.d_conv,
                    real_inf.config.dt_rank, real_inf.config.dt_b_c_rms);
            }
            // Offload híbrido GPU (só com --features wgpu; M3_GPU=0 desliga)
            #[cfg(feature = "wgpu")]
            {
                let n = real_inf.offload_gpu(&vm.memory);
                println!("[gpu] {n} tensores offloaded (gate/up/down/head); resto no CPU");
            }
            // Real inference não usa o programa m3asm dummy, mas carrega para manter compat
            vm.load_program(prog);
            let (s, v) = run_real_interactive(vm, real_inf, max_steps, trace).await?;
            vm = v;
            s
        } else {
            println!("\n=== M³-AVM em execução (interativo) ===");
            println!("Digite qualquer texto + ENTER durante a geração para interromper e redirecionar.");
            println!("FORK a cada instrução de checkpoint, ABORT restaura CoW 39µs, SENSE escuta supervisor.\n");
            vm.load_program(prog);
            let (s, v) = run_interactive(vm, max_steps, trace).await?;
            vm = v;
            s
        }
    } else {
        let is_transcribe = path.file_name().and_then(|n| n.to_str()).map(|n| n.contains("transcribe")).unwrap_or(false);
        if is_transcribe {
        println!(";; [stt] modo transcribe detectado — NOP + whisper mock (8 opcodes intactos)");
        let bus = bus::Bus::new();
        let mut reactor = reactor::Reactor::new(vm, bus);
        reactor.vm.load_program(prog);
        let stats = reactor.run_nop().await?;
        // Host lê áudio de TEMPORAL (SENSE r0) e transcreve via mock/service
        let audio_pcm = {
            let mem = &reactor.vm.memory;
            let audio_addr = reactor.vm.scheduler.get(1).map(|c| c.regs[0]).unwrap_or(0);
            if audio_addr != 0 && crate::memory::region_of(audio_addr) == crate::memory::Region::Temporal {
                let bytes = mem.read(audio_addr, 16000*2).unwrap_or_default(); // 1s
                stt::WhisperService::pcm_bytes_to_f32(&bytes)
            } else {
                // Fallback: gera 2s de senoide 440Hz como demo
                (0..16000*2).map(|i| (2.0*std::f32::consts::PI*440.0*i as f32/16000.0).sin()*0.5).collect()
            }
        };
        let svc = {
            // Prefere modelo real (--model ou default tiny-q8_0) quando o arquivo existe;
            // senão cai para mock determinístico (sem binário whisper-cli).
            let default_model = stt::MODEL_DEFAULT.to_string();
            let model_path = model_for_tok
                .as_ref()
                .and_then(|p| p.to_str())
                .unwrap_or(&default_model);
            match stt::WhisperService::new(model_path) {
                Ok(s) => s,
                Err(_) => stt::WhisperService::new_mock(),
            }
        };
        let start = std::time::Instant::now();
        let text = svc.transcribe(&audio_pcm).unwrap_or_else(|e| format!("stt error: {}", e));
        let elapsed = start.elapsed();
        let rtf = stt::WhisperService::rtf(elapsed.as_secs_f32(), audio_pcm.len() as f32/16000.0);
        println!("\n=== STT ===");
        println!("Áudio: {} samples ({:.1}s)", audio_pcm.len(), audio_pcm.len() as f32/16000.0);
        println!("Texto: {}", text);
        println!("RTF: {:.2} (elapsed {:?})", rtf, elapsed);
        // Escreve em PERSISTENTE para STREAM posterior (simula host)
        let persist_addr = crate::memory::make_persistent_addr(0x1000);
        let _ = reactor.vm.memory.write(persist_addr, text.as_bytes());
        let _ = reactor.vm.memory.persistent_flush();
        println!("Texto escrito em PERSISTENTE 0x{:x} ({} bytes)", persist_addr, text.len());
        // Move stats e memória de volta para vm para exibição
        vm = reactor.vm;
        stats
    } else {
        vm.load_program(prog);
        vm.run()?
    }
    };

    println!("\n=== M³-AVM Estatísticas ===");
    println!("Steps           : {}", stats.steps);
    println!("TENSOR allocs   : {}", stats.tensor_allocs);
    println!("ATTN execs      : {}", stats.attn_execs);
    println!("NORM execs      : {}", stats.norm_execs);
    println!("FFN execs       : {}", stats.ffn_execs);
    println!("EMBED execs     : {}", stats.embed_execs);
    println!("ADD execs       : {}", stats.add_execs);
    println!("SAMPLE execs    : {}", stats.sample_execs);
    println!("STREAMs         : {}", stats.streams);
    println!("FORKs           : {}", stats.forks);
    println!("ABORTs          : {}", stats.aborts);
    println!("SENSEs          : {}", stats.senses);
    println!("Tempo           : {} ms", stats.elapsed_ms());
    println!("Throughput      : {:.0} IPS",  stats.steps as f64 / (stats.elapsed_ms().max(1) as f64 / 1000.0));
    println!("Memória         : {}", vm.memory.stats());
    let (r, b, g) = vm.scheduler.queue_lengths();
    println!("Filas (R/B/G)   : {}/{}/{}", r, b, g);
    println!("Contextos ativos: {}", vm.scheduler.active_count());

    // Mostra registradores do contexto 1 (se existir)
    if let Some(ctx) = vm.scheduler.get(1) {
        println!("\n-- Contexto 1 regs --");
        for (i, v) in ctx.regs.iter().enumerate() {
            if *v != 0 {
                println!("  r{:02} = 0x{:032x} ({})", i, v, v);
            }
        }
        println!("  PC   = 0x{:032x}", ctx.pc);
        println!("  Prio = {}", ctx.priority);
    }

    Ok(())
}

/// Loop interativo da tese: SENSE no topo, FORK checkpoint, ABORT rollback via stdin.
/// Mantém os 8 opcodes inseparáveis — controle (COMPARE/JUMP) fica no host, não na ISA.
async fn run_interactive(mut vm: vm::Vm, max_steps: u64, trace: bool) -> Result<(vm::VmStats, vm::Vm)> {
    use std::io::{self, Write};
    use crate::opcodes::Instruction;

    // Canal std_mpsc para entrada do usuário (sem poluição de stdout)
    let (tx, rx) = std_mpsc::channel::<String>();
    std::thread::spawn(move || {
        let stdin = io::stdin();
        let mut line = String::new();
        loop {
            line.clear();
            match stdin.read_line(&mut line) {
                Ok(0) => break, // EOF (pipe fechado) -> disconnect canal para encerrar idle
                Ok(_) => {},
                Err(_) => break,
            }
            let trimmed = line.trim().to_string();
            if !trimmed.is_empty() {
                let _ = tx.send(trimmed);
            }
        }
    });

    let bus = Bus::new();
    let mut is_generating = false; // inicia idle: espera prompt inicial do usuário (não gera sozinho)
    let mut output_buffer: Vec<String> = Vec::new();
    #[derive(Clone, Copy, Debug)]
    struct Checkpoint { version: u64, pc: u128, token_idx: usize }
    let mut checkpoints: Vec<Checkpoint> = Vec::new();
    let mut entropy_tracker = rollback::EntropyTracker::new(50);
    let mut steps: u64 = 0;
    let _start_ns = utils::now_ns();

    println!("[Aguardando prompt inicial — digite texto e ENTER para iniciar geração]");
    println!("[Após iniciar, digite novamente + ENTER a qualquer momento para interromper e fazer rollback]");

    loop {
        if let Some(max) = Some(max_steps) {
            if steps >= max { println!("\n[max_steps {} atingido]", max); break; }
        }

        // 1. SENSE implícito: verifica stdin sem bloquear (tese: supervisor) — Algoritmo V1
        // Detecta também disconnect (pipe fechado) para não travar infinito em `echo | m3_avm`
        match rx.try_recv() {
            Ok(new_prompt) => {
                // trata prompt abaixo — fecha match no final do bloco
                let new_prompt = new_prompt;
            if is_generating {
                println!("\n[Interrupção implícita detectada: \"{}\"]", new_prompt);
                if !checkpoints.is_empty() {
                    // V1: Entropia (Alg 4) + Fallback janela fixa (Alg 1) — src/rollback.rs:34
                    let target_idx = rollback::determine_target_token_index(&entropy_tracker, output_buffer.len());
                    // Encontra checkpoint mais próximo <= target_idx
                    let cp = checkpoints.iter().rev().find(|c| c.token_idx <= target_idx).copied()
                        .or_else(|| checkpoints.first().copied());
                    if let Some(Checkpoint { version: ver, pc, token_idx: tok_idx }) = cp {
                        let t0 = utils::now_ns();
                        match vm.memory.restore(ver) {
                            Ok(_) => {
                                let dt = (utils::now_ns() - t0) / 1000;
                                println!("[Rollback CoW {}µs para v{} pc 0x{:x} — truncando {}->{} tokens (target {} via V1)]", dt, ver, pc, output_buffer.len(), tok_idx, target_idx);
                                // Restaura PC
                                if let Some(ctx) = vm.scheduler.get_mut(1) {
                                    ctx.pc = pc;
                                } else if let Some(id) = vm.scheduler.contexts().keys().next().cloned() {
                                    if let Some(ctx) = vm.scheduler.get_mut(id) { ctx.pc = pc; }
                                }
                                output_buffer.truncate(tok_idx);
                                // Limpa histórico de entropia pós-rollback (estado mudou)
                                entropy_tracker.clear();
                                // Trunca checkpoints futuros
                                checkpoints.retain(|c| c.token_idx <= tok_idx);
                                // Injeta novo prompt em TEMPORAL
                                match vm.memory.temporal_push(new_prompt.as_bytes()) {
                                    Ok(addr) => {
                                        println!("[Novo prompt injetado @ TEMPORAL 0x{:032x} ({} bytes): \"{}\"]", addr, new_prompt.len(), new_prompt);
                                        let sig = InterruptSignal { target_ctx: 1, layer: 0, timestamp_ns: utils::now_ns() };
                                        let _ = bus.publish_interrupt(sig);
                                    }
                                    Err(e) => eprintln!("falha ao injetar prompt em TEMPORAL: {}", e),
                                }
                                vm.stats.aborts += 1;
                            }
                            Err(e) => eprintln!("restore falhou v{}: {}", ver, e),
                        }
                    } else {
                        let addr = vm.memory.temporal_push(new_prompt.as_bytes()).unwrap_or(0);
                        println!("[Sem checkpoint <= target {} — prompt injetado @ 0x{:x}]", target_idx, addr);
                    }
                } else {
                    // Sem checkpoint, apenas injeta (fallback robusto V1)
                    let addr = vm.memory.temporal_push(new_prompt.as_bytes()).unwrap_or(0);
                    println!("[Sem checkpoint — prompt injetado @ 0x{:x} (fallback)]", addr);
                }
            } else {
                // Não está gerando, inicia nova geração (host-side)
                println!("\n[Iniciando geração com prompt: \"{}\"]", new_prompt);
                let addr = vm.memory.temporal_push(new_prompt.as_bytes()).unwrap_or(0);
                // Se não há contexto ativo (programa já HALTou), recria contexto GREEN no entry PC
                if vm.scheduler.active_count() == 0 {
                    let ver = vm.memory.current_version();
                    let entry_pc = vm.program_base;
                    let nid = vm.scheduler.create_context(crate::context::Priority::Green, entry_pc, ver);
                    if let Some(ctx) = vm.scheduler.get_mut(nid) {
                        ctx.regs[10] = addr;
                    }
                    println!("[Novo contexto {} criado @ PC 0x{:x} para prompt]", nid, entry_pc);
                } else if let Some(ctx) = vm.scheduler.get_mut(1) {
                    ctx.regs[10] = addr;
                    if ctx.state == crate::context::ContextState::Terminated {
                        ctx.state = crate::context::ContextState::Ready;
                        ctx.pc = vm.program_base;
                    }
                }
                is_generating = true;
                output_buffer.clear();
                // Limpa checkpoints/entropia da geração anterior para nova sessão
                checkpoints.clear();
                entropy_tracker.clear();
            }
            },
            Err(std_mpsc::TryRecvError::Disconnected) => {
                if !is_generating && vm.scheduler.active_count() == 0 {
                    // pipe fechado (echo | ) e sem geração — encerra graciosamente
                    break;
                }
            },
            Err(std_mpsc::TryRecvError::Empty) => {}
        }

        if !is_generating {
            // Pausa curta e continua aguardando prompt — nunca break automático em interativo
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            // max_steps ainda respeitado (se usuário passou --max-steps finito, encerra idle após atingi-lo)
            if max_steps != 0 && steps >= max_steps {
                break;
            }
            continue;
        }

        // 2. Scheduler estrito (RED>BLUE>GREEN)
        let ctx_id = match vm.scheduler.pick_next() {
            Ok(id) => id,
            Err(_) => {
                println!("\n[VM ociosa — sem contexto pronto]");
                is_generating = false;
                continue;
            }
        };

        let instr = {
            let ctx = match vm.scheduler.get(ctx_id) {
                Some(c) => c.clone(),
                None => continue,
            };
            let base = vm.program_base;
            if ctx.pc < base { eprintln!("PC fora"); vm.scheduler.get_mut(ctx_id).unwrap().state = crate::context::ContextState::Terminated; continue; }
            let idx = ((ctx.pc - base)/32) as usize;
            if idx >= vm.program.len() { eprintln!("PC fora do programa"); vm.scheduler.get_mut(ctx_id).unwrap().state = crate::context::ContextState::Terminated; continue; }
            vm.program[idx].clone()
        };

        if trace {
            let ctx = vm.scheduler.get(ctx_id).unwrap();
            eprintln!("[trace][ctx {} pc 0x{:x}] {}", ctx_id, ctx.pc, instr);
        }

        // HALT/NOP fast path
        match instr.opcode {
            OP_HALT => {
                if let Some(c) = vm.scheduler.get_mut(ctx_id) { c.state = crate::context::ContextState::Terminated; }
                steps += 1; vm.stats.steps += 1;
                if vm.scheduler.active_count() == 0 {
                    // Em modo interativo NÃO encerra: aguarda novo prompt (tese: geração contínua)
                    println!("\n[Geração concluída — digite novo prompt ou Ctrl+C para sair]");
                    is_generating = false;
                    // Não break — volta ao topo e fica em idle esperando stdin (loop abaixo)
                }
                vm.scheduler.yield_current();
                continue;
            }
            OP_NOP => {
                if let Some(c) = vm.scheduler.get_mut(ctx_id) { c.advance_pc(); c.state = crate::context::ContextState::Ready; }
                vm.scheduler.yield_current(); steps += 1; vm.stats.steps += 1; continue;
            }
            _ => {}
        }

        // 3. Executa instrução via stepping compartilhado da VM quando a
        // semântica é pura (uma implementação só — ver Vm::step_instruction).
        // ATTN/NORM/FFN/STREAM/FORK/ABORT/SENSE seguem stubs de demo abaixo
        // (instrumentação de preempção/outputs/checkpoints da tese).
        let exec_res = match instr.opcode {
            opcodes::OP_TENSOR
            | opcodes::OP_EMBED
            | opcodes::OP_ADD
            | opcodes::OP_SAMPLE
            | opcodes::OP_COMPARE
            | opcodes::OP_JUMP
            | opcodes::OP_IF_EQUAL
            | opcodes::OP_IF_INTERRUPT => vm.step_instruction(ctx_id, &instr),
            opcodes::OP_ATTN => {
                // Checa interrupção antes (watch 0 custo)
                if let Some(sig) = bus.has_interrupt() { if sig.target_ctx == ctx_id { eprintln!("[ATTN preemptado por ABORT antes]"); } }
                // Reusa Vm attn via chamada direta se possível — simplifica: chama vm.memory attn_sparse/dense inline minimal
                // Para MVP, delega para VM via snapshot de método público: cria instrução ATTN via vm internal?
                // Fallback: executa ATTN denso 2x2 via Vm helper (copia lógica curta)
                // Aqui apenas marca attn_execs e aloca tensor dummy para não travar demo interativo
                // (ATTN real já testado em vm.rs; interativo foca em preempção)
                let rdest = instr.rdest;
                // Tenta execução real se tensores existirem, senão dummy
                let mut try_real = || -> anyhow::Result<()> {
                    let (aq, ak, av) = {
                        let ctx = vm.scheduler.get(ctx_id).ok_or(anyhow::anyhow!("ctx"))?;
                        (ctx.reg(instr.rsrc1)?, ctx.reg(instr.rsrc2)?, ctx.reg(instr.rsrc3)?)
                    };
                    // Se algum esparso, usa sparse path simplificado
                    if vm.memory.is_sparse(aq) || vm.memory.is_sparse(ak) || vm.memory.is_sparse(av) {
                        // dummy allocate 8x8
                        let out = vm.memory.alloc_sparse_tensor(&[8,8], crate::memory::DType::F32, 0.05)?;
                        if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { ctx.set_reg(rdest, out)?; }
                    } else {
                        let out = vm.memory.alloc_tensor(&[2,2], crate::memory::DType::F32)?;
                        let _ = vm.memory.write_f32_tensor(out, &[1.0;4]);
                        if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { ctx.set_reg(rdest, out)?; }
                    }
                    Ok(())
                };
                let _ = try_real();
                vm.stats.attn_execs += 1;
                Ok(true)
            }
            opcodes::OP_NORM => {
                // NORM minimal dummy
                let rdest = instr.rdest;
                let out = vm.memory.alloc_tensor(&[2,4], crate::memory::DType::F32).unwrap();
                let _ = vm.memory.write_f32_tensor(out, &[0.5;8]);
                if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(rdest, out); }
                vm.stats.norm_execs += 1;
                Ok(true)
            }
            opcodes::OP_FFN => {
                let rdest = instr.rdest;
                let out = vm.memory.alloc_tensor(&[2,4], crate::memory::DType::F32).unwrap();
                let _ = vm.memory.write_f32_tensor(out, &[0.7;8]);
                if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(rdest, out); }
                vm.stats.ffn_execs += 1;
                Ok(true)
            }
            opcodes::OP_STREAM => {
                // Periféricos raw (STREAM r, SAMPLE) via instr.rsrc2 sem dereferenciar — evita `return` que sairia da função
                let stream_res: Result<bool> = if instr.rsrc2 == 2 {
                    let src = {
                        let ctx = vm.scheduler.get(ctx_id).unwrap();
                        if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1).unwrap_or(0) } else { 0 }
                    };
                    let logits: Vec<f32> = if src != 0 {
                        if let Some(meta) = vm.memory.get_tensor_meta(src).cloned() {
                            let n: usize = meta.shape.iter().product();
                            vm.memory.read_f32_tensor(src, n).unwrap_or_else(|_| vec![0.5; 4])
                        } else {
                            vm.memory.read(src, 16).ok().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()).unwrap_or(vec![0.5;4])
                        }
                    } else { vec![0.5;4] };
                    let tok = vm.sample_logits(&logits);
                    vm.last_sample = tok;
                    if trace { eprintln!("[SAMPLE tok {} from {} logits]", tok, logits.len()); }
                    let entropy = rollback::compute_entropy(&logits);
                    entropy_tracker.push(output_buffer.len(), entropy);
                    self::utils::log_info("sample", &format!("SAMPLE {} -> token {}", logits.len(), tok));
                    vm.stats.streams += 1;
                    if instr.rdest != 0xFF { if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, tok as u128); } }
                    Ok(true)
                } else if instr.rsrc2 == 4 {
                    let src = {
                        let ctx = vm.scheduler.get(ctx_id).unwrap();
                        if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1).unwrap_or(0) } else { 0 }
                    };
                    let token_id: u32 = if src < 100_000 && src != 0 {
                        src as u32
                    } else if src != 0 {
                        vm.memory.read(src, 4).ok().and_then(|b| if b.len()>=4 { Some(u32::from_le_bytes([b[0],b[1],b[2],b[3]])) } else { None }).unwrap_or(vm.last_sample)
                    } else {
                        vm.last_sample
                    };
                    let text = if let Some(tok) = &vm.tokenizer { tok.decode(token_id) } else { let vocab = ["Olá","mundo","M³","AVM","DeepSeek","Qwen","token","teste"]; vocab[(token_id as usize) % vocab.len()].to_string() };
                    let out = if output_buffer.is_empty() { text.clone() } else { format!(" {}", text) };
                    print!("{}", out);
                    let _ = io::stdout().flush();
                    output_buffer.push(text);
                    vm.stats.streams += 1;
                    if trace { eprintln!("[OUTPUT_DECODED token {}]", token_id); }
                    if instr.rdest != 0xFF { if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, output_buffer.len() as u128); } }
                    Ok(true)
                } else {
                    let (src, sink) = {
                        let ctx = vm.scheduler.get(ctx_id).unwrap();
                        let s = if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1).unwrap_or(0) } else { 0 };
                        let d = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2).unwrap_or(0) } else { 0 };
                        (s,d)
                    };
                // Periféricos especiais via valor dereferenciado (compat)
                if sink == crate::opcodes::STREAM_PERIPHERAL_SAMPLE {
                    // Amostra logits -> token
                    let logits: Vec<f32> = if src != 0 {
                        if let Some(meta) = vm.memory.get_tensor_meta(src).cloned() {
                            let n: usize = meta.shape.iter().product();
                            vm.memory.read_f32_tensor(src, n).unwrap_or_else(|_| vec![0.5; 4])
                        } else {
                            vm.memory.read(src, 16).ok().map(|b| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()).unwrap_or(vec![0.5;4])
                        }
                    } else { vec![0.5;4] };
                    let tok = vm.sample_logits(&logits);
                    vm.last_sample = tok;
                    if trace { eprintln!("[SAMPLE tok {} from {} logits]", tok, logits.len()); }
                    let entropy = rollback::compute_entropy(&logits);
                    entropy_tracker.push(output_buffer.len(), entropy);
                    self::utils::log_info("sample", &format!("SAMPLE {} -> token {}", logits.len(), tok));
                    vm.stats.streams += 1;
                    if instr.rdest != 0xFF { if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, tok as u128); } }
                    Ok(true)
                } else if sink == crate::opcodes::STREAM_PERIPHERAL_OUTPUT_DECODED {
                    let token_id: u32 = if src < 100_000 && src != 0 {
                        src as u32
                    } else if src != 0 {
                        vm.memory.read(src, 4).ok().and_then(|b| if b.len()>=4 { Some(u32::from_le_bytes([b[0],b[1],b[2],b[3]])) } else { None }).unwrap_or(vm.last_sample)
                    } else {
                        vm.last_sample
                    };
                    let text = if let Some(tok) = &vm.tokenizer {
                        tok.decode(token_id)
                    } else {
                        let vocab = ["Olá","mundo","M³","AVM","DeepSeek","Qwen","token","teste"];
                        vocab[(token_id as usize) % vocab.len()].to_string()
                    };
                    let out = if output_buffer.is_empty() { text.clone() } else { format!(" {}", text) };
                    print!("{}", out);
                    let _ = io::stdout().flush();
                    output_buffer.push(text);
                    vm.stats.streams += 1;
                    if trace { eprintln!("[OUTPUT_DECODED token {}]", token_id); }
                    if instr.rdest != 0xFF { if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, output_buffer.len() as u128); } }
                    Ok(true)
                } else if sink == 0xFF || sink == 0 {
                    let txt = if src != 0 {
                        vm.memory.read(src, 16).ok().and_then(|b| String::from_utf8(b).ok()).unwrap_or(format!("tok_{}", output_buffer.len()))
                    } else { format!("tok_{}", output_buffer.len()) };
                    print!("{}", txt);
                    let _ = io::stdout().flush();
                    let token_idx = output_buffer.len();
                    output_buffer.push(txt);
                    let logits: Vec<f32> = if src != 0 {
                        if let Some(meta) = vm.memory.get_tensor_meta(src).cloned() {
                            let n: usize = meta.shape.iter().product();
                            vm.memory.read_f32_tensor(src, n).unwrap_or_else(|_| vec![0.0; 4])
                        } else {
                            vm.memory.read(src, 16).ok().map(|b| b.iter().map(|&x| x as f32 / 255.0).collect()).unwrap_or(vec![0.0;4])
                        }
                    } else { vec![0.5;4] };
                    let entropy = rollback::compute_entropy(&logits);
                    entropy_tracker.push(token_idx, entropy);
                    if trace { eprintln!("[entropy tok {} H={:.2}]", token_idx, entropy); }
                    vm.stats.streams += 1;
                    if instr.rdest != 0xFF { if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, output_buffer.len() as u128); } }
                    Ok(true)
                } else {
                    // STREAM interno (não token)
                    vm.stats.streams += 1;
                    if instr.rdest != 0xFF { if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, output_buffer.len() as u128); } }
                    Ok(true)
                }
                };
                stream_res
            }
            opcodes::OP_FORK => {
                use crate::context::Priority;
                let prio = Priority::from_flags(instr.flags & 0b11);
                let parent = vm.scheduler.get(ctx_id).cloned().unwrap();
                let snap = vm.memory.snapshot();
                // FORK com rótulo: filho começa no alvo (igual à VM real)
                let mut ibb = [0u8; 16];
                ibb.copy_from_slice(&instr.payload[0..16]);
                let label_pc = u128::from_le_bytes(ibb);
                let child_pc = if label_pc != 0 { label_pc } else { parent.pc.wrapping_add(32) };
                let new_id = vm.scheduler.create_context(prio, child_pc, snap);
                if let Some(child) = vm.scheduler.get_mut(new_id) { child.regs = parent.regs; }
                if let Some(p) = vm.scheduler.get_mut(ctx_id) { let _ = p.set_reg(instr.rdest, new_id as u128); }
                vm.stats.forks += 1;
                let _ = bus.publish_sched(crate::bus::SchedSignal{ new_ctx: new_id, prio });
                // Salva checkpoint tese (a cada FORK) — 39µs CoW — V1 usa para rollback por entropia
                let cp_pc = child_pc;
                checkpoints.push(Checkpoint { version: snap, pc: cp_pc, token_idx: output_buffer.len() });
                if trace { eprintln!("[Checkpoint v{} pc 0x{:x} tok {}]", snap, cp_pc, output_buffer.len()); }
                Ok(true)
            }
            opcodes::OP_ABORT => {
                let (target, ts) = {
                    let ctx = vm.scheduler.get(ctx_id).unwrap();
                    let t = if instr.rsrc1 != 0xFF { ctx.reg(instr.rsrc1).unwrap_or(0) as u64 } else { 0 };
                    let ts = if instr.rsrc2 != 0xFF { ctx.reg(instr.rsrc2).unwrap_or(0) as u64 } else { 0 };
                    (t, ts)
                };
                if target != 0 {
                    let sig = InterruptSignal { target_ctx: target, layer: 0, timestamp_ns: utils::now_ns() };
                    let _ = bus.publish_interrupt(sig);
                    if let Some(removed) = vm.scheduler.remove(target) { eprintln!("[ABORT matou ctx {}]", target); let _ = removed; }
                    if ts != 0 { let _ = vm.memory.restore(ts); }
                    vm.stats.aborts += 1;
                }
                Ok(true)
            }
            opcodes::OP_SENSE => {
                let periph = instr.rsrc1;
                if periph == opcodes::SENSE_USER_INPUT {
                    // Consome fila host (push via TUI/pipe); valor 1/0 + flag.
                    let item = vm.user_input.pop_front();
                    let has = item.is_some();
                    if let Some(ctx) = vm.scheduler.get_mut(ctx_id) {
                        let _ = ctx.set_reg(instr.rdest, if has { 1 } else { 0 });
                        ctx.interrupt_flag = has;
                    }
                    vm.stats.senses += 1;
                    if let Some(text) = item {
                        let addr = vm.memory.temporal_push(text.as_bytes()).unwrap_or(0);
                        if trace { eprintln!("[SENSE USER_INPUT {} bytes @ 0x{:x}]", text.len(), addr); }
                    }
                    Ok(true)
                } else {
                let data: Vec<u8> = match periph {
                    opcodes::SENSE_AUDIO => crate::mimi::pcm_to_bytes(&crate::mimi::synth_frame_440hz()),
                    opcodes::SENSE_VAD => vec![if rand::random::<bool>(){1}else{0}],
                    opcodes::SENSE_TOKEN => vm.last_sample.to_le_bytes().to_vec(),
                    _ => b"sense".to_vec(),
                };
                let addr = vm.memory.temporal_push(&data).unwrap();
                if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, addr); }
                // Também escreve last_sample direto no registrador se for TOKEN (facilita assembly que espera token ID direto)
                if periph == opcodes::SENSE_TOKEN {
                    if let Some(ctx) = vm.scheduler.get_mut(ctx_id) { let _ = ctx.set_reg(instr.rdest, vm.last_sample as u128); }
                }
                vm.stats.senses += 1;
                Ok(true)
                }
            }
            _ => Err(anyhow::anyhow!("opcode 0x{:02x}", instr.opcode)),
        };

        match exec_res {
            Ok(should_advance) => {
                if let Some(ctx) = vm.scheduler.get_mut(ctx_id) {
                    if ctx.state == crate::context::ContextState::Running {
                        if should_advance { ctx.advance_pc(); }
                        ctx.state = crate::context::ContextState::Ready;
                    }
                }
                vm.scheduler.yield_current();
                steps += 1; vm.stats.steps += 1;
                // Checa interrupção pós-ATTN
                if bus.has_interrupt().is_some() { bus.clear_interrupt(); }
            }
            Err(e) => {
                eprintln!("exec falhou ctx {} {}: {} — terminando", ctx_id, instr.mnemonic(), e);
                if let Some(c) = vm.scheduler.get_mut(ctx_id) { c.state = crate::context::ContextState::Terminated; }
                vm.scheduler.yield_current(); steps += 1; vm.stats.steps += 1;
            }
        }

        // Pequena pausa para demonstrar streaming token a token (tese: 217µs+39µs, aqui 10ms)
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Se todos terminaram, fica esperando novo prompt interativo (não encerra após 200ms)
        if vm.scheduler.active_count() == 0 {
            if is_generating {
                println!("\n[Geração concluída — digite novo prompt ou Ctrl+C para sair]");
                is_generating = false;
            }
            // Idle: aguarda indefinidamente por stdin sem busy-loop
            // Não dá break automático — usuário decide com Ctrl+C
            // Mas respeita max_steps: se atingiu, encerra
            if max_steps != 0 && steps >= max_steps {
                break;
            }
            // Dá chance ao thread de stdin enviar
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            // Continua no topo onde rx.try_recv() será verificado
            // Se ainda is_generating==false, cai no bloco `if !is_generating` no topo
            continue;
        }
    }
    Ok((vm.stats.clone(), vm))
}

async fn run_real_interactive(mut vm: vm::Vm, mut real: crate::inference::RealInference, max_steps: u64, trace: bool) -> Result<(vm::VmStats, vm::Vm)> {
    // REPL com linha de input fixa (tty) ou passthrough (pipe): output da LLM
    // nunca se mistura com o que está sendo digitado.
    let mut repl = tui::Repl::new()?;
    let bus = Bus::new();
    let mut is_generating = false;
    let mut output_buffer: Vec<String> = Vec::new();
    #[derive(Clone, Copy, Debug)] struct Checkpoint { version: u64, token_idx: usize }
    let mut checkpoints: Vec<Checkpoint> = Vec::new();
    let mut entropy_tracker = crate::rollback::EntropyTracker::new(50);
    let mut steps: u64 = 0;
    let mut current_token: u32 = 0;
    let mut token_queue: Vec<u32> = Vec::new(); // tokens do prompt inicial a serem consumidos

    repl.print_line("[Real] Aguardando prompt inicial — digite e ENTER (Ctrl+C sai)")?;

    loop {
        if max_steps != 0 && steps >= max_steps { repl.print_line("[max_steps atingido]")?; break; }

        // SENSE implícito
        match repl.try_poll() {
            tui::Input::Line(new_prompt) => {
                if is_generating {
                    repl.print_line(&format!("[Real Interrupção: \"{}\"]", new_prompt))?;
                    if !checkpoints.is_empty() {
                        let target_idx = crate::rollback::determine_target_token_index(&entropy_tracker, output_buffer.len());
                        let cp = checkpoints.iter().rev().find(|c| c.token_idx <= target_idx).copied().or_else(|| checkpoints.first().copied());
                        if let Some(Checkpoint{version: ver, token_idx: tok_idx}) = cp {
                            let t0 = crate::utils::now_ns();
                            let _ = vm.memory.restore(ver);
                            let dt = (crate::utils::now_ns() - t0)/1000;
                            repl.print_line(&format!("[Real Rollback CoW {}µs para v{} — truncando {}->{} tok (target {} via V1)]", dt, ver, output_buffer.len(), tok_idx, target_idx))?;
                            output_buffer.truncate(tok_idx);
                            entropy_tracker.clear();
                            checkpoints.retain(|c| c.token_idx <= tok_idx);
                            real.truncate_caches(tok_idx);
                            real.truncate_gen(tok_idx);
                            let _ = vm.memory.temporal_push(new_prompt.as_bytes());
                            let _ = bus.publish_interrupt(crate::bus::InterruptSignal{ target_ctx: 1, layer: 0, timestamp_ns: crate::utils::now_ns() });
                            vm.stats.aborts += 1;
                            // Tokeniza novo prompt e prefill sobre KV truncado
                            let prompt_fmt = if new_prompt.contains("<|im_start|>") || new_prompt.contains("<|user|>") { new_prompt.clone() } else { real.format_chat(&new_prompt) };
                            let prompt_tokens = real.tokenize(&prompt_fmt);
                            repl.print_line(&format!("[Real] Interrupção tokenizada: {} tokens, prefill sobre KV truncado", prompt_tokens.len()))?;
                            match real.prefill_logits(&vm.memory, &prompt_tokens) {
                                Ok(logits) => {
                                    let next = real.sample_with_params(&logits, 0.7, 0.9, 40, 1.1);
                                    current_token = next;
                                    let text = real.tokenizer.decode_human(next).replace('▁', " ");
                                    repl.print_output(&text)?;
                                    output_buffer.push(text.trim().to_string());
                                    let entropy = crate::rollback::compute_entropy(&logits);
                                    entropy_tracker.push(output_buffer.len()-1, entropy);
                                    token_queue.clear();
                                },
                                Err(e) => {
                                    eprintln!("[Real prefill interrupção falhou: {}] — fallback queue", e);
                                    token_queue = prompt_tokens;
                                    current_token = token_queue.first().cloned().unwrap_or(0);
                                    if token_queue.len()>1 { token_queue.remove(0); }
                                }
                            }
                        }
                    } else {
                        let _ = vm.memory.temporal_push(new_prompt.as_bytes());
                        repl.print_line("[Real Sem checkpoint — prompt injetado]")?;
                        let prompt_fmt = if new_prompt.contains("<|im_start|>") || new_prompt.contains("<|user|>") { new_prompt.clone() } else { real.format_chat(&new_prompt) };
                        let prompt_tokens = real.tokenize(&prompt_fmt);
                        real.clear_caches();
                        real.clear_gen();
                        match real.prefill_logits(&vm.memory, &prompt_tokens) {
                            Ok(logits) => {
                                let next = real.sample_with_params(&logits, 0.7, 0.9, 40, 1.1);
                                current_token = next;
                                let text = real.tokenizer.decode_human(next).replace('▁', " ");
                                repl.print_output(&text)?;
                                output_buffer.push(text.trim().to_string());
                                token_queue.clear();
                            },
                            Err(_) => {
                                token_queue = prompt_tokens;
                                current_token = token_queue.first().cloned().unwrap_or(0);
                            }
                        }
                    }
                } else {
                    repl.print_line(&format!("[Real Iniciando com prompt: \"{}\"]", new_prompt))?;
                    let _ = vm.memory.temporal_push(new_prompt.as_bytes());
                    let prompt_fmt = if new_prompt.contains("<|im_start|>") || new_prompt.contains("<|user|>") { new_prompt.clone() } else { real.format_chat(&new_prompt) };
                    let prompt_tokens = real.tokenize(&prompt_fmt);
                    repl.print_line(&format!("[Real] Prompt tokenizado: {} tokens", prompt_tokens.len()))?;
                    is_generating = true;
                    output_buffer.clear();
                    checkpoints.clear();
                    entropy_tracker.clear();
                    real.clear_caches();
                    real.clear_gen();
                    // Prefill: alimenta KV com todos os tokens do prompt sem amostrar
                    if !prompt_tokens.is_empty() {
                        let prefill_start = crate::utils::now_ns();
                        match real.prefill_logits(&vm.memory, &prompt_tokens) {
                            Ok(logits) => {
                                let dt = (crate::utils::now_ns() - prefill_start)/1_000_000;
                                repl.print_line(&format!("[Real] Prefill {} tokens em {}ms", prompt_tokens.len(), dt))?;
                                // Amostra primeiro token após prompt
                                let next = real.sample_with_params(&logits, 0.7, 0.9, 40, 1.1);
                                current_token = next;
                                token_queue.clear();
                                // Decodifica e imprime primeiro token gerado imediatamente
                                let text = real.tokenizer.decode_human(next);
                                let clean = text.replace('▁', " ");
                                repl.print_output(&clean)?;
                                output_buffer.push(clean.trim().to_string());
                                let entropy = crate::rollback::compute_entropy(&logits);
                                entropy_tracker.push(0, entropy);
                            },
                            Err(e) => {
                                eprintln!("[Real prefill falhou: {}] — usando fallback", e);
                                token_queue = prompt_tokens;
                                current_token = token_queue.first().cloned().unwrap_or(0);
                                if token_queue.len()>1 { token_queue.remove(0); }
                            }
                        }
                    } else {
                        token_queue.clear();
                        current_token = 0;
                    }
                    // Checkpoint inicial
                    let ver = vm.memory.snapshot();
                    checkpoints.push(Checkpoint{ version: ver, token_idx: 0 });
                }
            },
            tui::Input::Eof => {
                if !is_generating && output_buffer.is_empty() { break; }
                // Se EOF durante geração, continua até acabar ou max_steps
            },
            tui::Input::Interrupt => {
                repl.print_line("[interrompido via Ctrl+C — encerrando]")?;
                break;
            },
            tui::Input::Empty => {},
        }

        if !is_generating {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            continue;
        }

        // Se há token_queue de fallback (prefill falhou), consome com KV
        let token_to_feed = if !token_queue.is_empty() {
            let t = token_queue.remove(0);
            // Prefill: alimenta KV sem gerar output humano
            let logits = match real.forward_one_auto(&vm.memory, t) {
                Ok(l) => l,
                Err(e) => { eprintln!("[Real prefill fallback falhou: {}]", e); continue; }
            };
            real.push_gen(t);
            if !token_queue.is_empty() {
                if trace { eprintln!("[Real prefill fallback tok {} '{}']", t, real.tokenizer.decode_human(t).replace('▁', " ")); }
                continue;
            }
            // último token do prompt: amostra próximo token diretamente dos logits
            let next = real.sample_with_params(&logits, 0.7, 0.9, 40, 1.1);
            if Some(next) == real.eos_id() {
                repl.print_line("[Real EOS — fim da resposta]")?;
                is_generating = false;
                continue;
            }
            let text = real.tokenizer.decode_human(next).replace('▁', " ");
            if text.starts_with(' ') && !output_buffer.is_empty() && output_buffer.last().map(|s| s.ends_with(' ')).unwrap_or(false) {
                repl.print_output(text.trim_start())?;
            } else {
                repl.print_output(&text)?;
            }
            output_buffer.push(text.trim().to_string());
            let entropy = crate::rollback::compute_entropy(&logits);
            entropy_tracker.push(output_buffer.len()-1, entropy);
            if trace { eprintln!("[Real tok {} H={:.2} id {}]", output_buffer.len()-1, entropy, next); }
            if output_buffer.len() % 1 == 0 {
                let ver = vm.memory.snapshot();
                checkpoints.push(Checkpoint{ version: ver, token_idx: output_buffer.len() });
            }
            current_token = next;
            steps += 1; vm.stats.steps += 1; vm.stats.streams += 1;
            tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
            continue;
        } else {
            current_token
        };

        // Forward real
        let logits = match real.forward_one_auto(&vm.memory, token_to_feed) {
            Ok(l) => l,
            Err(e) => { eprintln!("[Real forward falhou: {}]", e); tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; continue; }
        };
        real.push_gen(token_to_feed);
        let next_token = real.sample_with_params(&logits, 0.7, 0.9, 40, 1.1);
        if Some(next_token) == real.eos_id() {
            repl.print_line("[Real EOS — fim da resposta]")?;
            is_generating = false;
            continue;
        }
        let text_raw = real.tokenizer.decode_human(next_token);
        let text = text_raw.replace('▁', " ");
        // Streaming humano: não adiciona " " manual, ▁ já é espaço; evita duplicar
        repl.print_output(&text)?;
        output_buffer.push(text.trim().to_string());
        let entropy = crate::rollback::compute_entropy(&logits);
        entropy_tracker.push(output_buffer.len()-1, entropy);
        if trace { eprintln!("[Real tok {} H={:.2} id {}]", output_buffer.len()-1, entropy, next_token); }

        // Checkpoint a cada token (ou a cada 5)
        if output_buffer.len() % 1 == 0 {
            let ver = vm.memory.snapshot();
            checkpoints.push(Checkpoint{ version: ver, token_idx: output_buffer.len() });
            if trace { eprintln!("[Real Checkpoint v{} tok {}]", ver, output_buffer.len()); }
        }

        current_token = next_token;
        steps += 1; vm.stats.steps += 1; vm.stats.streams += 1;
        // Simula ~100ms por token para dar tempo de interrupção (real seria ~100-200ms em CPU)
        tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;

        if vm.stats.steps >= max_steps { break; }
        // Se output_buffer muito grande, pausa
        if output_buffer.len() > 200 { repl.print_line("[Real 200 tokens gerados — pausando, digite novo prompt ou Ctrl+C]")?; is_generating = false; }
    }
    Ok((vm.stats.clone(), vm))
}

fn assemble_file(input: PathBuf, output: Option<PathBuf>) -> Result<()> {
    let text = fs::read_to_string(&input)?;
    let prog = assemble(&text)?;
    let out_path = output.unwrap_or_else(|| input.with_extension("m3bin"));
    let mut bytes = Vec::with_capacity(prog.len() * INSTR_SIZE);
    for instr in &prog {
        bytes.extend_from_slice(&instr.encode());
    }
    fs::write(&out_path, &bytes)?;
    println!("OK: {} instruções -> {} ({} bytes)", prog.len(), out_path.display(), bytes.len());
    Ok(())
}

fn disassemble_file(path: PathBuf) -> Result<()> {
    let bytes = fs::read(&path)?;
    if bytes.len() % INSTR_SIZE != 0 {
        return Err(anyhow!("tamanho inválido para .m3bin"));
    }
    for (idx, chunk) in bytes.chunks_exact(INSTR_SIZE).enumerate() {
        let instr = Instruction::decode(chunk)?;
        println!("{:04}: {:032x}  {}", idx, idx as u128 * 32 + 0x1000, instr);
    }
    Ok(())
}

async fn bench(nops: usize) -> Result<()> {
    println!(";; Benchmark {} NOPs (target 1_000_000 IPS)...", nops);
    let mut vm = Vm::new_in_memory(VmConfig {
        max_steps: Some(nops as u64 + 10),
        enable_tracing: false,
        ips_target: 1_000_000,
    });
    let prog = vec![opcodes::instr_nop(); nops];
    // Adiciona HALT ao final
    let mut prog_with_halt = prog;
    prog_with_halt.push(opcodes::instr_halt());
    vm.load_program(prog_with_halt);

    let start = utils::now_ns();
    let stats = vm.run()?;
    let elapsed_ns = utils::now_ns() - start;
    let elapsed_s = elapsed_ns as f64 / 1e9;
    let ips = stats.steps as f64 / elapsed_s;

    println!("\n=== Benchmark ===");
    println!("Instruções : {}", stats.steps);
    println!("Tempo      : {:.3} s", elapsed_s);
    println!("Throughput : {:.0} IPS", ips);
    if ips >= 1_000_000.0 {
        println!("✅ PASS: atingiu target 1M IPS");
    } else {
        println!("⚠️  abaixo do target 1M IPS (mas OK em debug sem otimização)");
        println!("   Dica: rode com --release para +3-5x");
    }
    Ok(())
}

fn is_probably_bin(bytes: &[u8]) -> bool {
    // Heurística: se bytes 0 são opcodes conhecidos e resto parece payload zero
    if bytes.is_empty() {
        return false;
    }
    let opcode = bytes[0];
    matches!(opcode, 0x00 | 0x01 | 0x02 | 0x03 | 0x04 | 0x05 | 0x06 | 0x07 | 0x08 | 0xFF)
}
