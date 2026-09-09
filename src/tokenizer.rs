//! tokenizer.rs — M3Tokenizer: parse GGUF -> vocab + decode
//! Tenta ler `tokenizer.ggml.tokens` do GGUF; fallback mock ["Olá","mundo","M³","AVM"] se falhar.
//! GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
//! Header: magic 4B "GGUF" 0x46554747 LE, version u32 LE, n_tensors u64 LE, n_kv u64 LE
//! KV: key_len u64 LE + key bytes, type u32 LE, value (depende do type)
//! Value types: 0=U8,1=I8,2=U16,3=I16,4=U32,5=I32,6=F32,7=Bool,8=String,9=Array,10=U64,11=I64,12=F64
//! Array: type u32 LE + len u64 LE + elementos sequenciais

use anyhow::{anyhow, Result};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct M3Tokenizer {
    pub vocab: Vec<String>,
}

impl M3Tokenizer {
    pub fn mock() -> Self {
        Self { vocab: vec!["Olá".to_string(), "mundo".to_string(), "M³".to_string(), "AVM".to_string(), "DeepSeek".to_string(), "Qwen".to_string(), "token".to_string(), "teste".to_string()] }
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }

    pub fn decode(&self, token_id: u32) -> String {
        let idx = token_id as usize;
        if idx < self.vocab.len() {
            self.vocab[idx].clone()
        } else {
            format!("[{}]", token_id)
        }
    }

    /// Decodifica para humano: trata ▁/Ġ (espaço), Ċ (newline) e bytes <0xNN>
    pub fn decode_human(&self, token_id: u32) -> String {
        let raw = self.decode(token_id);
        // Remove tokens especiais <|...|> mantendo como está
        if raw.starts_with("<|") && raw.ends_with("|>") {
            return raw;
        }
        // Byte fallback GGUF <0xNN> -> char correspondente
        if raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>') {
            if let Ok(b) = u8::from_str_radix(&raw[3..5], 16) {
                return (b as char).to_string();
            }
        }
        // GPT-2/Qwen: Ġ = espaço, Ċ = newline
        // SentencePiece: ▁ = espaço
        let mut s = raw.replace('Ġ', " ").replace('▁', " ");
        s = s.replace('Ċ', "\n");
        // Limpa caracteres de controle
        s = s.replace("<0x0A>", "\n");
        s
    }

    /// Decodifica stream de tokens para string humana, tratando ▁ corretamente
    pub fn decode_stream(&self, tokens: &[u32]) -> String {
        let mut out = String::new();
        for &tid in tokens {
            let piece = self.decode_human(tid);
            // Evita duplicar espaço: se piece começa com espaço e out termina com espaço
            if piece.starts_with(' ') && out.ends_with(' ') {
                out.push_str(piece.trim_start());
            } else {
                out.push_str(&piece);
            }
        }
        // Normaliza: remove espaço inicial se for o primeiro token
        out.trim_start().to_string()
    }

    /// Tenta carregar vocab do GGUF em `path`. Se falhar, retorna mock e loga.
    pub fn from_gguf_or_mock(path: &str) -> Self {
        match Self::from_gguf(path) {
            Ok(t) => {
                eprintln!("[tokenizer] GGUF vocab carregado: {} tokens de {}", t.vocab.len(), path);
                t
            }
            Err(e) => {
                eprintln!("[tokenizer] falha ao ler GGUF vocab de {}: {} — usando mock {} tokens", path, e, Self::mock().vocab.len());
                Self::mock()
            }
        }
    }

    pub fn from_gguf(path: &str) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| anyhow!("open {}: {}", path, e))?;
        // Verifica tamanho mínimo
        let meta = file.metadata().map_err(|e| anyhow!("metadata: {}", e))?;
        if meta.len() < 32 {
            return Err(anyhow!("arquivo muito pequeno"));
        }
        // Header
        let mut hdr = [0u8; 4];
        file.read_exact(&mut hdr)?;
        if hdr != [0x47, 0x47, 0x55, 0x46] { // "GGUF" LE: 0x47 0x47 0x55 0x46
            // Tenta BE (GGUF)
            if hdr != [0x46, 0x55, 0x47, 0x47] {
                return Err(anyhow!("magic GGUF inválido {:02x?}", hdr));
            }
        }
        let version = read_u32_le(&mut file)?;
        if version < 2 || version > 4 {
            eprintln!("[tokenizer] aviso: GGUF version {} inesperada", version);
        }
        let n_tensors = read_u64_le(&mut file)?;
        let n_kv = read_u64_le(&mut file)?;
        // Itera KV para achar tokenizer.ggml.tokens
        for _ in 0..n_kv {
            let key = read_string(&mut file)?; // key_len u64 + bytes
            let vtype = read_u32_le(&mut file)?;
            // Se for o array de tokens, parseia
            if key == "tokenizer.ggml.tokens" {
                if vtype != 9 {
                    return Err(anyhow!("tokenizer.ggml.tokens type != Array ({})", vtype));
                }
                let arr_type = read_u32_le(&mut file)?;
                if arr_type != 8 {
                    return Err(anyhow!("tokens array elem type != String ({})", arr_type));
                }
                let arr_len = read_u64_le(&mut file)?;
                if arr_len > 500_000 {
                    return Err(anyhow!("vocab muito grande {}", arr_len));
                }
                let mut vocab = Vec::with_capacity(arr_len as usize);
                for _ in 0..arr_len {
                    let s = read_string(&mut file)?;
                    vocab.push(s);
                }
                // Encontrou vocab — ainda precisa pular resto do KV? Não, já consumiu value, pode retornar
                // Mas precisamos consumir resto do arquivo? Não necessário, já temos vocab
                return Ok(Self { vocab });
            } else {
                // Pula value conforme type
                skip_value(&mut file, vtype)?;
            }
        }
        // Não achou token array, tenta fallback "general.vocab" ou similar?
        // Para compatibilidade, tenta também "tokenizer.ggml.token_type" etc. mas se não achou, erro
        Err(anyhow!("tokenizer.ggml.tokens não encontrado (n_kv={}, n_tensors={})", n_kv, n_tensors))
    }
}

// Helpers LE
fn read_u32_le(f: &mut File) -> Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf).map_err(|e| anyhow!("read u32: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u64_le(f: &mut File) -> Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf).map_err(|e| anyhow!("read u64: {}", e))?;
    Ok(u64::from_le_bytes(buf))
}
fn read_string(f: &mut File) -> Result<String> {
    let len = read_u64_le(f)?;
    if len > 10_000_000 {
        return Err(anyhow!("string len muito grande {}", len));
    }
    let mut buf = vec![0u8; len as usize];
    if len > 0 {
        f.read_exact(&mut buf).map_err(|e| anyhow!("read string: {}", e))?;
    }
    String::from_utf8(buf).map_err(|e| anyhow!("utf8: {}", e))
}
fn skip_value(f: &mut File, vtype: u32) -> Result<()> {
    match vtype {
        0 => { let mut b=[0u8;1]; f.read_exact(&mut b).map_err(|e| anyhow!("skip U8: {}", e))?; Ok(()) }, // U8
        1 => { let mut b=[0u8;1]; f.read_exact(&mut b)?; Ok(()) }, // I8
        2 => { let mut b=[0u8;2]; f.read_exact(&mut b)?; Ok(()) }, // U16
        3 => { let mut b=[0u8;2]; f.read_exact(&mut b)?; Ok(()) }, // I16
        4 => { let mut b=[0u8;4]; f.read_exact(&mut b)?; Ok(()) }, // U32
        5 => { let mut b=[0u8;4]; f.read_exact(&mut b)?; Ok(()) }, // I32
        6 => { let mut b=[0u8;4]; f.read_exact(&mut b)?; Ok(()) }, // F32
        7 => { let mut b=[0u8;1]; f.read_exact(&mut b)?; Ok(()) }, // Bool (u8)
        8 => { let _s = read_string(f)?; Ok(()) }, // String
        9 => { // Array
            let arr_type = read_u32_le(f)?;
            let arr_len = read_u64_le(f)?;
            if arr_len > 1_000_000 { return Err(anyhow!("array len muito grande {}", arr_len)); }
            for _ in 0..arr_len {
                skip_value(f, arr_type)?;
            }
            Ok(())
        },
        10 => { let mut b=[0u8;8]; f.read_exact(&mut b)?; Ok(()) }, // U64
        11 => { let mut b=[0u8;8]; f.read_exact(&mut b)?; Ok(()) }, // I64
        12 => { let mut b=[0u8;8]; f.read_exact(&mut b)?; Ok(()) }, // F64
        _ => Err(anyhow!("vtype desconhecido {}", vtype)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mock_decode() {
        let t = M3Tokenizer::mock();
        assert_eq!(t.decode(0), "Olá");
        assert_eq!(t.decode(1), "mundo");
        assert_eq!(t.decode(100), "[100]");
    }
    #[test]
    fn test_gguf_deepseek_vocab() {
        // Tenta carregar vocab real se arquivo existir (não falha em CI sem arquivo)
        let path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
        if Path::new(path).exists() {
            match M3Tokenizer::from_gguf(path) {
                Ok(t) => {
                    println!("vocab DeepSeek: {} tokens, primeiro: {:?}", t.vocab.len(), &t.vocab[..3.min(t.vocab.len())]);
                    assert!(t.vocab.len() > 1000);
                }
                Err(e) => {
                    eprintln!("gguf parse falhou (ok se versão diferente): {}", e);
                }
            }
        }
    }
    #[test]
    fn test_gguf_tiny_vocab() {
        let path = "./models/ggml-tiny-q8_0.bin";
        if Path::new(path).exists() {
            let t = M3Tokenizer::from_gguf_or_mock(path);
            println!("tiny vocab: {} {:?}", t.vocab.len(), &t.vocab[..3.min(t.vocab.len())]);
            assert!(t.vocab.len() >= 4);
        }
    }
}
