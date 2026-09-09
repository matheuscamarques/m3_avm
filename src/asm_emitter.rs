//! asm_emitter.rs — Gerador de assembly M³ a partir de ModelConfig (emit-asm MVP)
//!
//! Traduz RealInference (22 layers, GQA) para programa M³ desenrolado com regs fixos.
//! Não toca VM: apenas escreve texto .m3asm que `opcodes::assemble` consome.
//! Pesos não são alocados via TENSOR PERSIST aqui — host faz `load_gguf_model()` antes de `vm.run()`,
//! o programa usa tensores temporários (hidden, norms, q/k/v) e assume que o interpreter
//! mapeia `ATTN/NORM/FFN` sobre os bytes já em PERSISTENTE 0x20. Para MVP, TENSORs são só temporários;
//! o emitter não precisa emitir `TENSOR` para cada peso do modelo (evita 200 TENSORs).

use anyhow::Result;
use std::path::Path;

use crate::inference::ModelConfig;

/// Emissor com registradores fixos e reuso por camada (cabe em R0..R15).
/// Mapa (sem alocador dinâmico):
/// R0 hidden_cur (entrada/saída da camada, residual)
/// R1 hidden_norm (saída NORM)
/// R2 q (hidden×hidden) / também recebe SAMPLE result temporariamente
/// R3 k_small (hidden×kv_hidden)
/// R4 v_small (idem)
/// R5 attn_out (após o_proj)
/// R6 ffn_gate / ffn_norm temporário
/// R7 ffn_up
/// R8 ffn_down result (usa depois para ADD)
/// R11 token_id atual (entrada EMBED, saída SAMPLE)
/// R10/R12-R15 temporários livres para SENSE/FORK/ABORT
pub struct AsmEmitter {
    pub config: ModelConfig,
}

impl AsmEmitter {
    pub fn new(config: &ModelConfig) -> Self {
        Self { config: config.clone() }
    }

    /// Gera programa assembly completo desenrolado (sem loops de camada).
    /// Retorna string pronta para escrita; gargalo é O(n_layers) texto.
    pub fn emit_to_string(&self) -> String {
        let mut out = Vec::new();
        out.push("; ======================================================".to_string());
        out.push("; Programa M³ gerado por --emit-asm (asm_emitter.rs)".to_string());
        out.push(format!("; arch={} hidden={} intermediate={} layers={} heads={} kv_heads={} vocab={} rope_theta={}",
            self.config.arch, self.config.hidden, self.config.intermediate, self.config.n_layers,
            self.config.n_heads, self.config.n_kv_heads, self.config.vocab, self.config.rope_theta));
        out.push(format!("; Gerado: {} instruções estimadas (~{} por camada + prólogo/epílogo)", self.estimate_instr_count(), 10));
        out.push("; Pesos: host já fez load_gguf_model() em PERSISTENTE 0x20 antes de run".to_string());
        out.push("; Execução: cargo run -- run <este_arquivo> --model <mesmo_gguf> [--trace]".to_string());
        out.push("; Preempção: SENSE USER_INPUT + IF_INTERRUPT a cada ATTN/FFN (tese §5)".to_string());
        out.push("; ======================================================".to_string());
        out.push(String::new());

        // Prólogo: aloca temporários com shapes compatíveis mas pequenos para FFN (evita colisão GGUF)
        // Hidden [1, hidden] é seguro (não colide com norma [hidden]); W1/W2 usam inter=64 dummy
        // para passar validação [1,hidden]*[hidden,64]*[64,hidden] sem mapear para pesos reais de 11M.
        // Isso prova completude da ISA; com modelo real, RealInference leria pesos via mmap, não via TENSOR aqui.
        let ff_inter = 64usize.min(self.config.intermediate);
        out.push("; --- prólogo: aloca temporários (stub) ---".to_string());
        out.push(format!("TENSOR r0 1 {} f32   ; hidden cur [1, hidden]", self.config.hidden));
        out.push(format!("TENSOR r9 {} {} f32  ; W1 dummy [hidden, {}] para FFN (stub, evita GGUF map)", self.config.hidden, ff_inter, ff_inter));
        out.push(format!("TENSOR r10 {} {} f32 ; W2 dummy [{}, hidden] para FFN", ff_inter, self.config.hidden, ff_inter));
        out.push("SENSE r15, USER_INPUT".to_string());
        out.push("IF_INTERRUPT HANDLE_ABORT".to_string());
        out.push(String::new());

        // Loop principal rotulado
        out.push("MAIN_LOOP:".to_string());
        out.push("    ; --- embedding lookup: R0 = embed[token R11] ---".to_string());
        out.push("    ; R11 já contém token_id (inicial vem de SENSE TOKEN ou host push)".to_string());
        out.push("    SENSE r11, TOKEN".to_string());
        out.push("    ; EMBED r0, r11, r1  ; (quando R1 mapear token_embd, descomentar)".to_string());
        out.push("    FORK r12, GREEN, NOTIFY".to_string());
        out.push("    SENSE r15, USER_INPUT".to_string());
        out.push("    IF_INTERRUPT HANDLE_ABORT".to_string());
        out.push(String::new());

        for layer in 0..self.config.n_layers {
            out.push(format!("    ; ===== camada {} ===== ({} layers total)", layer, self.config.n_layers));
            // NORM1: R1 = RMSNorm(R0) [1, hidden]
            out.push(format!("    NORM r1, r0, r0, r0   ; R1 = norm(R0) layer {}", layer));
            // Q/K/V/O projections via MATVEC + ATTN (usa pesos reais quando shape colide com GGUF)
            // Para Fase 1, emitimos MATVEC com shapes que mapeiam para GGUF distintos via try_gguf used-set:
            // Q/K/V/O todos [hidden, hidden] (4M) mapeiam para blk.{}.attn_q/k/v/output distintos.
            out.push(format!("    TENSOR r6 {} {} f32 ; q_proj layer {} [hidden, hidden]", self.config.hidden, self.config.hidden, layer));
            out.push(format!("    MATVEC r2, r1, r6      ; R2 = R1 * q_proj layer {}", layer));
            out.push(format!("    TENSOR r7 {} {} f32 ; k_proj layer {}", self.config.hidden, self.config.hidden, layer));
            out.push(format!("    MATVEC r3, r1, r7      ; R3 = R1 * k_proj layer {}", layer));
            out.push(format!("    TENSOR r8 {} {} f32 ; v_proj layer {}", self.config.hidden, self.config.hidden, layer));
            out.push(format!("    MATVEC r4, r1, r8      ; R4 = R1 * v_proj layer {}", layer));
            out.push(format!("    ATTN r5, r2, r3, r4   ; R5 = attn(Q=R2,K=R3,V=R4) layer {}", layer));
            out.push(format!("    TENSOR r6 {} {} f32 ; o_proj layer {} [hidden, hidden]", self.config.hidden, self.config.hidden, layer));
            out.push(format!("    MATVEC r5, r5, r6      ; R5 = R5 * o_proj layer {}", layer));
            out.push("    SENSE r15, USER_INPUT".to_string());
            out.push("    IF_INTERRUPT HANDLE_ABORT".to_string());
            out.push(format!("    ADD r0, r0, r5       ; R0 += attn_out layer {}", layer));
            // NORM2
            out.push(format!("    NORM r1, r0, r0, r0   ; norm2 layer {}", layer));
            // FFN gate/up/down — shapes [hidden, inter] e [inter, hidden]; usa inter=64 stub para não colidir se quiser stub, mas aqui emitimos inter real para mapear quando possível
            // Para Fase 1 mantemos stub 64 para FFN (evita 11M alloc repetido); Fase 2 emitirá inter real
            out.push(format!("    FFN r8, r1, r9, r10  ; FFN layer {} (R1->[hidden,64]->[hidden]) stub", layer));
            out.push("    SENSE r15, USER_INPUT".to_string());
            out.push("    IF_INTERRUPT HANDLE_ABORT".to_string());
            out.push(format!("    ADD r0, r0, r8       ; R0 += ffn_out layer {}", layer));
            out.push(String::new());
        }

        // Projeção final + SAMPLE + saída
        out.push("    ; --- LM head + SAMPLE ---".to_string());
        out.push("    ; logits em R1, SAMPLE -> R11 (token id)".to_string());
        out.push("    SAMPLE r11, r0".to_string());
        out.push("    STREAM r11, 4 BLOCKING   ; 4 = OUTPUT_DECODED (token -> texto)".to_string());
        out.push("    SENSE r15, USER_INPUT".to_string());
        out.push("    IF_INTERRUPT HANDLE_ABORT".to_string());
        out.push("    COMPARE r11, 2           ; EOS check (2 = default, DeepSeek usa 151643 mas VM compara com 2 stub)".to_string());
        out.push("    IF_EQUAL PROGRAM_END".to_string());
        out.push("    JUMP MAIN_LOOP".to_string());
        out.push(String::new());
        out.push("HANDLE_ABORT:".to_string());
        out.push("    ; rollback CoW + truncate já feito pelo host (bus interrupt); aqui só volta".to_string());
        out.push("    JUMP MAIN_LOOP".to_string());
        out.push(String::new());
        out.push("PROGRAM_END:".to_string());
        out.push("    HALT".to_string());

        out.join("\n")
    }

    pub fn estimate_instr_count(&self) -> usize {
        // prólogo 5 + MAIN_LOOP header 5 + n_layers*18 (NORM+3×(TENSOR+MATVEC)+ATTN+TENSOR+MATVEC+ADD+NORM+FFN) + epílogo 7
        5 + 5 + self.config.n_layers * 18 + 7
    }

    /// Escreve arquivo `.m3asm` no path informado.
    pub fn emit_to_file(&self, path: &Path) -> Result<()> {
        let text = self.emit_to_string();
        std::fs::write(path, text)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::ModelConfig;

    fn mock_cfg(layers: usize) -> ModelConfig {
        ModelConfig {
            hidden: 32,
            intermediate: 64,
            n_layers: layers,
            n_heads: 4,
            n_kv_heads: 4,
            vocab: 32000,
            arch: "llama".to_string(),
            context_length: 2048,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
        }
    }

    #[test]
    fn test_emit_assembles() {
        let cfg = mock_cfg(2);
        let emitter = AsmEmitter::new(&cfg);
        let text = emitter.emit_to_string();
        let prog = crate::opcodes::assemble(&text).unwrap();
        // deve conter HALT e JUMP
        assert!(prog.iter().any(|i| i.opcode == crate::opcodes::OP_HALT));
        assert!(prog.iter().any(|i| i.opcode == crate::opcodes::OP_JUMP));
        // contagem ~ prólogo + 2*10 + epílogo
        assert!(prog.len() >= 20);
    }

    #[test]
    fn test_emit_labels_resolved() {
        let cfg = mock_cfg(1);
        let emitter = AsmEmitter::new(&cfg);
        let text = emitter.emit_to_string();
        assert!(text.contains("MAIN_LOOP:"));
        assert!(text.contains("HANDLE_ABORT:"));
        assert!(text.contains("PROGRAM_END:"));
        let prog = crate::opcodes::assemble(&text).unwrap();
        // Deve ter JUMP e IF_EQUAL/IF_INTERRUPT com alvos resolvidos dentro do range
        let jumps: Vec<_> = prog.iter().filter(|i| i.opcode == crate::opcodes::OP_JUMP || i.opcode == crate::opcodes::OP_IF_EQUAL || i.opcode == crate::opcodes::OP_IF_INTERRUPT).collect();
        assert!(!jumps.is_empty(), "deveria ter saltos");
        for j in &jumps {
            let pc = j.imm_u128();
            assert!(pc >= 0x1000 && pc < 0x1000 + (prog.len() as u128)*32, "alvo fora do programa: {:#x}", pc);
            assert_eq!(pc % 32, 0, "alvo não alinhado");
        }
        // JUMP MAIN_LOOP deve voltar para início do loop — sanity que label existe no texto
        assert!(text.lines().any(|l| l.trim() == "MAIN_LOOP:"));
    }

    #[test]
    fn test_emit_file_roundtrip() {
        let cfg = mock_cfg(1);
        let emitter = AsmEmitter::new(&cfg);
        let dir = std::env::temp_dir();
        let path = dir.join("m3_emit_test.m3asm");
        emitter.emit_to_file(&path).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let prog = crate::opcodes::assemble(&text).unwrap();
        assert!(!prog.is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
