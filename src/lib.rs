//! lib.rs — biblioteca M³-AVM (para testes de integração e NIF)
//! Re-exporta os módulos da VM para uso como crate.

pub mod bus;
pub mod context;
pub mod memory;
pub mod memory_wgpu;
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
pub mod determinism;
#[cfg(feature = "wgpu")]
pub mod inference_gpu;
pub mod tui;
pub mod utils;
pub mod vm;
pub mod asm_emitter;
pub mod m3bc;
pub mod arena;
pub mod activations;

// Módulo de QA — testes de física abstrata (Nível 1-4)
#[cfg(test)]
pub mod qa;
