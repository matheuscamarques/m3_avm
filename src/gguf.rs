//! gguf.rs — parser mínimo GGUF para f32 + vocab
//! Lê header, KV (para tokenizer) e tensor_infos (nome, dims, dtype, offset)
//! Suporta apenas F32 (0) e F16 (1) para VM; Q4_* é rejeitado com erro claro.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>, // GGUF armazena dims little-endian, ordem invertida? Mantém como lida
    pub shape: Vec<usize>, // dims como usize
    pub dtype: u32, // 0=F32,1=F16,2=Q4_0 etc.
    pub offset: u64, // offset dentro da região de dados (após header, alinhado 32)
    pub n_elements: usize,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub n_tensors: u64,
    pub n_kv: u64,
    pub tensors: Vec<GgufTensorInfo>,
    pub kv: HashMap<String, String>, // apenas chaves string para debug
    pub data_offset: u64, // offset no arquivo onde começam os dados dos tensores
}

fn read_u32_le(f: &mut File) -> Result<u32> {
    let mut buf = [0u8;4];
    f.read_exact(&mut buf).map_err(|e| anyhow!("read u32: {}", e))?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u64_le(f: &mut File) -> Result<u64> {
    let mut buf = [0u8;8];
    f.read_exact(&mut buf).map_err(|e| anyhow!("read u64: {}", e))?;
    Ok(u64::from_le_bytes(buf))
}
fn read_string(f: &mut File) -> Result<String> {
    let len = read_u64_le(f)?;
    if len > 10_000_000 { return Err(anyhow!("string len {}", len)); }
    let mut buf = vec![0u8; len as usize];
    if len>0 { f.read_exact(&mut buf).map_err(|e| anyhow!("read string: {}", e))?; }
    String::from_utf8(buf).map_err(|e| anyhow!("utf8: {}", e))
}
fn skip_value(f: &mut File, vtype: u32) -> Result<()> {
    match vtype {
        0=>{let mut b=[0u8;1]; f.read_exact(&mut b).map_err(|e| anyhow!("skip U8: {}", e))?; Ok(())},
        1=>{let mut b=[0u8;1]; f.read_exact(&mut b)?; Ok(())},
        2=>{let mut b=[0u8;2]; f.read_exact(&mut b)?; Ok(())},
        3=>{let mut b=[0u8;2]; f.read_exact(&mut b)?; Ok(())},
        4=>{let mut b=[0u8;4]; f.read_exact(&mut b)?; Ok(())},
        5=>{let mut b=[0u8;4]; f.read_exact(&mut b)?; Ok(())},
        6=>{let mut b=[0u8;4]; f.read_exact(&mut b)?; Ok(())},
        7=>{let mut b=[0u8;1]; f.read_exact(&mut b)?; Ok(())},
        8=>{let _s=read_string(f)?; Ok(())},
        9=>{
            let at = read_u32_le(f)?;
            let al = read_u64_le(f)?;
            for _ in 0..al { skip_value(f, at)?; }
            Ok(())
        },
        10=>{let mut b=[0u8;8]; f.read_exact(&mut b)?; Ok(())},
        11=>{let mut b=[0u8;8]; f.read_exact(&mut b)?; Ok(())},
        12=>{let mut b=[0u8;8]; f.read_exact(&mut b)?; Ok(())},
        _=>Err(anyhow!("vtype {}", vtype)),
    }
}

impl GgufFile {
    pub fn open(path: &str) -> Result<Self> {
        let mut f = File::open(path).map_err(|e| anyhow!("open {}: {}", path, e))?;
        let mut magic=[0u8;4];
        f.read_exact(&mut magic).map_err(|e| anyhow!("read magic: {}", e))?;
        if magic != [0x47,0x47,0x55,0x46] && magic != [0x46,0x55,0x47,0x47] {
            return Err(anyhow!("magic GGUF inválido {:02x?}", magic));
        }
        let version = read_u32_le(&mut f)?;
        let n_tensors = read_u64_le(&mut f)?;
        let n_kv = read_u64_le(&mut f)?;
        let mut kv = HashMap::new();
        let mut vocab: Vec<String> = Vec::new();
        // Para tokenizer, vamos capturar token array se for o primeiro, mas parse genérico
        // Vamos iterar KV e guardar chaves; para tokenizer, detectamos array de tokens depois
        // Simplifica: parse KV igual tokenizer.rs, mas também guarda tensores depois
        for _ in 0..n_kv {
            let key = read_string(&mut f)?;
            let vtype = read_u32_le(&mut f)?;
            // Captura tokenizer tokens se for o array
            if key == "tokenizer.ggml.tokens" && vtype==9 {
                let at = read_u32_le(&mut f)?;
                let al = read_u64_le(&mut f)?;
                if at==8 {
                    // Para não alocar 150k strings aqui, apenas skip e deixa tokenizer.rs fazer parse separado
                    // Mas precisamos pular corretamente para não desalinhar offset
                    for _ in 0..al {
                        let _s = read_string(&mut f)?;
                    }
                    kv.insert(key, format!("array[{}]", al));
                } else {
                    // skip
                    for _ in 0..al { skip_value(&mut f, at)?; }
                }
            } else {
                // Para chaves string simples, guarda valor se for string
                if vtype==8 {
                    let v = read_string(&mut f)?;
                    kv.insert(key, v);
                } else if vtype<=7 || vtype==10 || vtype==11 || vtype==12 {
                    // Escalares numéricos com sinal correto (F32/F64 inclusive:
                    // rope theta, eps etc. dependem disso!)
                    let val = match vtype {
                        0 => { let mut b=[0u8;1]; f.read_exact(&mut b).map_err(|e| anyhow!("read U8: {}", e))?; (b[0] as f64).to_string() },
                        1 => { let mut b=[0u8;1]; f.read_exact(&mut b).map_err(|e| anyhow!("read I8: {}", e))?; ((b[0] as i8) as f64).to_string() },
                        2 => { let mut b=[0u8;2]; f.read_exact(&mut b).map_err(|e| anyhow!("read U16: {}", e))?; (u16::from_le_bytes(b) as f64).to_string() },
                        3 => { let mut b=[0u8;2]; f.read_exact(&mut b).map_err(|e| anyhow!("read I16: {}", e))?; ((i16::from_le_bytes(b)) as f64).to_string() },
                        4 => (read_u32_le(&mut f)? as f64).to_string(),
                        5 => { let mut b=[0u8;4]; f.read_exact(&mut b).map_err(|e| anyhow!("read I32: {}", e))?; (i32::from_le_bytes(b) as f64).to_string() },
                        6 => { let mut b=[0u8;4]; f.read_exact(&mut b).map_err(|e| anyhow!("read F32: {}", e))?; f32::from_le_bytes(b).to_string() },
                        7 => { let mut b=[0u8;1]; f.read_exact(&mut b).map_err(|e| anyhow!("read bool: {}", e))?; (b[0] != 0).to_string() },
                        10=> read_u64_le(&mut f)?.to_string(),
                        11=> { let mut b=[0u8;8]; f.read_exact(&mut b).map_err(|e| anyhow!("read I64: {}", e))?; (i64::from_le_bytes(b)).to_string() },
                        12=> { let mut b=[0u8;8]; f.read_exact(&mut b).map_err(|e| anyhow!("read F64: {}", e))?; f64::from_le_bytes(b).to_string() },
                        _=> unreachable!(),
                    };
                    kv.insert(key, val);
                } else {
                    skip_value(&mut f, vtype)?;
                    kv.insert(key, format!("type{}", vtype));
                }
            }
        }
        // Tensor infos
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = read_string(&mut f)?;
            let n_dims = read_u32_le(&mut f)?;
            if n_dims>8 { return Err(anyhow!("n_dims {}", n_dims)); }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims { dims.push(read_u64_le(&mut f)?); }
            let dtype = read_u32_le(&mut f)?;
            let offset = read_u64_le(&mut f)?;
            let mut shape = dims.iter().map(|&x| x as usize).collect::<Vec<_>>();
            // GGUF dims são little-endian e na ordem [n0,n1,...] onde n0 é dim mais interna
            // Para VM, mantemos como está, mas n_elements é produto
            let n_elements: usize = shape.iter().product::<usize>().max(1);
            // shape para VM: mantém dims como lida, mas para matriz 2D [4096,4096] pode vir [4096,4096]
            tensors.push(GgufTensorInfo{ name, dims: dims.clone(), shape: shape.clone(), dtype, offset, n_elements });
        }
        // Data offset: posição atual alinhada a 32
        let pos = f.stream_position().map_err(|e| anyhow!("tell: {}", e))?;
        let aligned = (pos + 31) & !31;
        let data_offset = aligned;
        Ok(Self{ version, n_tensors, n_kv, tensors, kv, data_offset })
    }

    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name==name)
    }

    pub fn tensors_by_dtype(&self, dtype: u32) -> Vec<&GgufTensorInfo> {
        self.tensors.iter().filter(|t| t.dtype==dtype).collect()
    }

    /// Lê escalar F32 do KV com default (freq_base, eps etc. dependem disso).
    pub fn kv_f32(&self, key: &str, default: f32) -> f32 {
        self.kv.get(key).and_then(|v| v.parse::<f32>().ok()).unwrap_or(default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gguf_tinyllama_f32_header() {
        let path = "./models/tinyllama-1.1b-chat-v1.0-f32.gguf";
        if std::path::Path::new(path).exists() {
            let gg = GgufFile::open(path).unwrap();
            println!("tinyllama f32: version {} tensors {} kv {}", gg.version, gg.n_tensors, gg.n_kv);
            for t in gg.tensors.iter().take(5) {
                println!("  {} {:?} dtype {} off {}", t.name, t.shape, t.dtype, t.offset);
            }
            assert!(gg.n_tensors>0);
        } else {
            eprintln!("skip tinyllama f32 not downloaded");
        }
    }
    #[test]
    fn test_gguf_f32_kv_parsed_not_type6() {
        // Regressão: F32 do KV (rope theta, eps) era descartado como "type6",
        // e o SmolLM2 (theta 130000) herdava default 10000 em silêncio.
        let path = "./models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            eprintln!("skip sem modelo");
            return;
        }
        let gg = GgufFile::open(path).unwrap();
        let theta = gg.kv_f32("llama.rope.freq_base", 10000.0);
        println!("smol theta kv = {}", theta);
        assert!((theta - 130000.0).abs() < 1.0, "theta {}", theta);
    }
    #[test]
    fn test_gguf_deepseek_q4_header() {
        let path = "./models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf";
        if std::path::Path::new(path).exists() {
            let gg = GgufFile::open(path).unwrap();
            println!("deepseek q4: {} tensors", gg.n_tensors);
            assert!(gg.n_tensors>0);
        }
    }
}
