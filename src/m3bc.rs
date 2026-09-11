//! m3bc.rs — container `.m3bc` (ESPEC-V2 §8, W10-loader).
//!
//! Header (34 bytes, tudo LE):
//!
//! ```text
//! [0..4]   MAGIC "M3BC"
//! [4..6]   MAJOR u16
//! [6..8]   MINOR u16
//! [8..10]  PATCH u16
//! [10..18] REQUIRED u64 (bits = capacidades exigidas)
//! [18..26] OPTIONAL u64 (bits = capacidades opcionais, ignorados se ausentes)
//! [26..30] ENTRY_PC u32 (byte offset a partir do início do payload)
//! [30..34] CRC32-IEEE u32 sobre bytes [0..30] ++ payload completo
//! ```
//!
//! **Regra de sniffing (R10):** ausência de MAGIC implica o path legado
//! v1.x `.m3bin` (executa sem negociação; nunca rejeita por falta de
//! header). Negociação aplica-se somente a `.m3bc`.
//!
//! **Versão do container:** acompanha a linha ISA (`1.5.0` = ISA v1.5).
//! Loader estrito: `MAJOR` diferente do corrente é `Err` explícito
//! (relaxar exige RFC — nada de sniffing silencioso de futuro).
//!
//! **Capacidades:** nenhum bit implementado além do baseline v1.3 (que
//! não precisa de bit, cf. ESPEC-V2 §8). Qualquer bit em REQUIRED é
//! `Err(UnsupportedFeature)` nomeando os bits; OPTIONAL é ignorado.
//!
//! **Frames largos:** o payload é fatiado pela size function normativa
//! (`instr_width`, ESPEC-V2 §4.3, exceção `0xFF` sempre 32B). Frames
//! `Fixed32` decodificam para `Instruction`; larguras maiores voltam
//! como `Frame::Raw` — a VM ainda não os executa (W1-remainder: fetch
//! stride 64B pendente, cf. RFC-0002 follow-up). Nada é misdecodificado.

use anyhow::{bail, Context, Result};

use crate::determinism::crc32_ieee;
use crate::opcodes::{fixed_size, instr_width, Instruction, InstrWidth};

/// MAGIC do container `.m3bc`.
pub const M3BC_MAGIC: &[u8; 4] = b"M3BC";
/// Tamanho total do header (MAGIC + campos + CRC32).
pub const M3BC_HEADER_LEN: usize = 34;
/// Versão corrente do container (= linha ISA implementada).
pub const M3BC_VERSION: (u16, u16, u16) = (1, 5, 0);
/// Bits REQUIRED suportados: nenhum além do baseline (que não usa bit).
pub const M3BC_SUPPORTED_REQUIRED: u64 = 0;
/// Bit OPTIONAL 49: seção `.data` no container (V-1b dia 3).
/// NÚMERO CONGELADO, semântica pendente: o container ainda não carrega
/// payload de dados (CLI `assemble` rejeita `.data`; só o path API
/// `assemble_with_data` + `load_assembled` existe). Ligar este bit sem
/// o layout de container seria mentira documentada — wiring quando o
/// layout existir. Teste abaixo trava o valor.
pub const M3BC_OPTIONAL_HAS_DATA_SECTION: u64 = 1 << 49;

/// Formato detectado por sniffing (R10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerFormat {
    /// Header `.m3bc` presente: vale negociação de features.
    M3bc,
    /// Sem MAGIC: legado v1.x `.m3bin`, sem negociação.
    LegacyM3bin,
}

/// Header `.m3bc` já validado (campos, sem o CRC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct M3bcHeader {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
    pub required: u64,
    pub optional: u64,
    pub entry_pc: u32,
}

impl M3bcHeader {
    /// Serializa os 30 bytes cobertos pelo CRC (`[0..30]`).
    pub fn encode_body(&self) -> [u8; 30] {
        let mut out = [0u8; 30];
        out[0..4].copy_from_slice(M3BC_MAGIC);
        out[4..6].copy_from_slice(&self.major.to_le_bytes());
        out[6..8].copy_from_slice(&self.minor.to_le_bytes());
        out[8..10].copy_from_slice(&self.patch.to_le_bytes());
        out[10..18].copy_from_slice(&self.required.to_le_bytes());
        out[18..26].copy_from_slice(&self.optional.to_le_bytes());
        out[26..30].copy_from_slice(&self.entry_pc.to_le_bytes());
        out
    }

    /// Parseia e valida estrutura (MAGIC + tamanhos); CRC e negociação
    /// ficam com o `load` (precisam do payload).
    pub fn parse(data: &[u8]) -> Result<Self> {
        if data.len() < M3BC_HEADER_LEN {
            bail!(
                "header .m3bc incompleto: {} bytes (mínimo {})",
                data.len(),
                M3BC_HEADER_LEN
            );
        }
        if &data[0..4] != M3BC_MAGIC {
            bail!("MAGIC ausente — trate como legado `.m3bin` (R10), não como erro");
        }
        let u16le = |r: std::ops::Range<usize>| u16::from_le_bytes([data[r.start], data[r.start + 1]]);
        let u32le = |r: std::ops::Range<usize>| {
            u32::from_le_bytes([data[r.start], data[r.start + 1], data[r.start + 2], data[r.start + 3]])
        };
        let u64le = |r: std::ops::Range<usize>| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&data[r]);
            u64::from_le_bytes(b)
        };
        Ok(Self {
            major: u16le(4..6),
            minor: u16le(6..8),
            patch: u16le(8..10),
            required: u64le(10..18),
            optional: u64le(18..26),
            entry_pc: u32le(26..30),
        })
    }

    pub fn stored_crc(data: &[u8]) -> u32 {
        let mut b = [0u8; 4];
        b.copy_from_slice(&data[30..34]);
        u32::from_le_bytes(b)
    }
}

/// Um frame do payload: instrução 32B decodificada ou bytes crus largos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    /// Instrução 32B pronta para a VM.
    I32(Instruction),
    /// Largura >32B: bytes crus + largura (VM ainda não executa).
    Raw { width: InstrWidth, bytes: Vec<u8> },
}

impl Frame {
    /// Largura em bytes deste frame.
    pub fn byte_len(&self) -> usize {
        match self {
            Frame::I32(_) => 32,
            Frame::Raw { bytes, .. } => bytes.len(),
        }
    }
}

/// Programa carregado de bytes (qualquer formato).
#[derive(Debug, Clone)]
pub struct LoadedProgram {
    pub format: ContainerFormat,
    /// `Some` apenas no formato `.m3bc`.
    pub header: Option<M3bcHeader>,
    pub entry_pc: u32,
    pub frames: Vec<Frame>,
}

impl LoadedProgram {
    /// Só os frames 32B, na ordem. Falha nomeando o primeiro frame largo
    /// (caminho honesto até o fetch 64B existir).
    pub fn instructions_32(&self) -> Result<Vec<Instruction>> {
        let mut out = Vec::new();
        for (i, f) in self.frames.iter().enumerate() {
            match f {
                Frame::I32(ins) => out.push(*ins),
                Frame::Raw { width, bytes } => bail!(
                    "frame {}: largura {:?} ({}B) exige fetch 64B (W1-remainder, pendente)",
                    i,
                    width,
                    bytes.len()
                ),
            }
        }
        Ok(out)
    }
}

/// Sniffing puro (R10): MAGIC presente => `.m3bc`, senão legado.
pub fn sniff(data: &[u8]) -> ContainerFormat {
    if data.len() >= 4 && &data[0..4] == M3BC_MAGIC {
        ContainerFormat::M3bc
    } else {
        ContainerFormat::LegacyM3bin
    }
}

/// Carrega bytes (qualquer formato) com validação total.
pub fn load(data: &[u8]) -> Result<LoadedProgram> {
    match sniff(data) {
        ContainerFormat::LegacyM3bin => load_legacy(data),
        ContainerFormat::M3bc => load_m3bc(data),
    }
}

/// Path legado: payload cru fatiado em 32B, cada chunk decodificado com
/// o decoder v1.x estrito (opcodes `>= 0x80` => `UnsupportedWidth`,
/// reservados => `ReservedOpcode` — nunca misdecodifica).
fn load_legacy(data: &[u8]) -> Result<LoadedProgram> {
    if data.len() % 32 != 0 {
        bail!(
            "legado `.m3bin` com tamanho não-múltiplo de 32: {} bytes",
            data.len()
        );
    }
    let mut frames = Vec::with_capacity(data.len() / 32);
    for (i, chunk) in data.chunks_exact(32).enumerate() {
        let ins =
            Instruction::decode(chunk).with_context(|| format!("legado: chunk {} inválido", i))?;
        frames.push(Frame::I32(ins));
    }
    Ok(LoadedProgram {
        format: ContainerFormat::LegacyM3bin,
        header: None,
        entry_pc: 0,
        frames,
    })
}

/// Path `.m3bc`: header + CRC + negociação + fatiamento por largura.
fn load_m3bc(data: &[u8]) -> Result<LoadedProgram> {
    let header = M3bcHeader::parse(data)?;
    if header.major != M3BC_VERSION.0 {
        bail!(
            "versão .m3bc não suportada: {}.{}.{} (corrente {}.{}.{})",
            header.major,
            header.minor,
            header.patch,
            M3BC_VERSION.0,
            M3BC_VERSION.1,
            M3BC_VERSION.2
        );
    }
    let unsupported = header.required & !M3BC_SUPPORTED_REQUIRED;
    if unsupported != 0 {
        bail!(
            "features REQUIRED não suportadas: 0x{:016x} (OPTIONAL 0x{:016x} ignorado)",
            unsupported,
            header.optional
        );
    }
    let payload = &data[M3BC_HEADER_LEN..];
    // CRC32 sobre header[0..30] ++ payload.
    let mut crc_input = Vec::with_capacity(30 + payload.len());
    crc_input.extend_from_slice(&data[0..30]);
    crc_input.extend_from_slice(payload);
    let want = crc32_ieee(&crc_input);
    let got = M3bcHeader::stored_crc(data);
    if want != got {
        bail!(
            "CRC32 .m3bc inválido: calculado 0x{:08x}, armazenado 0x{:08x}",
            want,
            got
        );
    }
    let frames = split_frames(payload)?;
    // ENTRY_PC deve apontar para início de frame.
    let mut off = 0usize;
    let mut aligned = header.entry_pc == 0;
    for f in &frames {
        if off as u32 == header.entry_pc {
            aligned = true;
            break;
        }
        off += f.byte_len();
    }
    if !aligned {
        bail!(
            "ENTRY_PC {} não alinha com início de frame (payload {}B, {} frames)",
            header.entry_pc,
            payload.len(),
            frames.len()
        );
    }
    Ok(LoadedProgram {
        format: ContainerFormat::M3bc,
        header: Some(header),
        entry_pc: header.entry_pc,
        frames,
    })
}

/// Fatia o payload pela size function normativa (§4.3).
fn split_frames(payload: &[u8]) -> Result<Vec<Frame>> {
    let mut frames = Vec::new();
    let mut off = 0usize;
    let mut idx = 0usize;
    while off < payload.len() {
        let op = payload[off];
        let width = instr_width(op);
        match width {
            InstrWidth::Fixed32 | InstrWidth::Fixed64 | InstrWidth::Fixed128 | InstrWidth::Fixed256 => {
                let n = fixed_size(width).expect("largura fixa tem tamanho");
                if off + n > payload.len() {
                    bail!(
                        "frame {} (op 0x{:02x}): precisa de {}B, restam {}B",
                        idx,
                        op,
                        n,
                        payload.len() - off
                    );
                }
                let bytes = &payload[off..off + n];
                if width == InstrWidth::Fixed32 {
                    let ins = Instruction::decode(bytes)
                        .with_context(|| format!("frame {} (op 0x{:02x}) inválido", idx, op))?;
                    frames.push(Frame::I32(ins));
                } else {
                    frames.push(Frame::Raw {
                        width,
                        bytes: bytes.to_vec(),
                    });
                }
                off += n;
            }
            InstrWidth::EscapeVar => bail!(
                "frame {} (op 0x{:02x} ESCAPE_VAR): layout ext_len de [R] §3.2 ainda não mapeado — sem chute de tamanho",
                idx,
                op
            ),
            InstrWidth::Reserved => bail!(
                "frame {}: opcode reservado 0x{:02x} (sem tamanho assumido, cf. §4.3)",
                idx,
                op
            ),
        }
        idx += 1;
    }
    Ok(frames)
}

/// Constrói bytes `.m3bc` a partir de header (sem CRC) + payload.
/// Usado por testes e pelo futuro `--emit-m3bc`.
pub fn encode_m3bc(header: &M3bcHeader, payload: &[u8]) -> Vec<u8> {
    let body = header.encode_body();
    let mut crc_input = Vec::with_capacity(30 + payload.len());
    crc_input.extend_from_slice(&body);
    crc_input.extend_from_slice(payload);
    let crc = crc32_ieee(&crc_input);
    let mut out = Vec::with_capacity(M3BC_HEADER_LEN + payload.len());
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcodes::{instr_nop, instr_tensor};

    fn header(required: u64, optional: u64, entry_pc: u32) -> M3bcHeader {
        M3bcHeader {
            major: M3BC_VERSION.0,
            minor: M3BC_VERSION.1,
            patch: M3BC_VERSION.2,
            required,
            optional,
            entry_pc,
        }
    }

    #[test]
    fn sniff_legacy() {
        // Sem MAGIC => legado, nunca erro.
        let data = vec![0x01u8; 32];
        assert_eq!(sniff(&data), ContainerFormat::LegacyM3bin);
        assert_eq!(sniff(&[]), ContainerFormat::LegacyM3bin);
        assert_eq!(sniff(b"M3B"), ContainerFormat::LegacyM3bin);
    }

    #[test]
    fn sniff_m3bc() {
        let mut data = Vec::from(&b"M3BC"[..]);
        data.extend_from_slice(&[0u8; 30]);
        assert_eq!(sniff(&data), ContainerFormat::M3bc);
    }

    #[test]
    fn header_roundtrip() {
        let h = header(0, 0xFF, 64);
        let body = h.encode_body();
        assert_eq!(&body[0..4], b"M3BC");
        let back = M3bcHeader::parse(&[body.to_vec(), 0u32.to_le_bytes().to_vec()].concat()).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn legacy_load_decodes() {
        let mut data = Vec::new();
        data.extend_from_slice(&instr_tensor(0, 0xFF, 0xFF, 2, 2, 0).encode());
        data.extend_from_slice(&instr_nop().encode());
        let prog = load(&data).unwrap();
        assert_eq!(prog.format, ContainerFormat::LegacyM3bin);
        assert_eq!(prog.frames.len(), 2);
        let ins = prog.instructions_32().unwrap();
        assert_eq!(ins.len(), 2);
        assert_eq!(ins[0].opcode, crate::opcodes::OP_TENSOR);
    }

    #[test]
    fn legacy_rejects_bad_len() {
        assert!(load(&[0u8; 33]).is_err());
    }

    #[test]
    fn legacy_rejects_64b_opcode() {
        // Opcode >= 0x80 no path legado => UnsupportedWidth, nunca misdecode.
        let mut data = vec![0u8; 32];
        data[0] = 0x80;
        assert!(load(&data).is_err());
    }

    #[test]
    fn m3bc_roundtrip_ok() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&instr_nop().encode());
        payload.extend_from_slice(&instr_tensor(1, 0xFF, 0xFF, 2, 2, 0).encode());
        let bytes = encode_m3bc(&header(0, 0, 0), &payload);
        let prog = load(&bytes).unwrap();
        assert_eq!(prog.format, ContainerFormat::M3bc);
        assert_eq!(prog.frames.len(), 2);
        assert_eq!(prog.instructions_32().unwrap().len(), 2);
    }

    #[test]
    fn m3bc_required_bits_rejected() {
        let payload = instr_nop().encode();
        let bytes = encode_m3bc(&header(0b101, 0, 0), &payload);
        let err = load(&bytes).unwrap_err().to_string();
        assert!(err.contains("REQUIRED"), "erro deve nomear REQUIRED: {}", err);
    }

    #[test]
    fn m3bc_optional_bits_ignored() {
        let payload = instr_nop().encode();
        let bytes = encode_m3bc(&header(0, 0xDEAD_BEEF, 0), &payload);
        assert!(load(&bytes).is_ok());
    }

    #[test]
    fn m3bc_crc_mismatch_rejected() {
        let payload = instr_nop().encode();
        let mut bytes = encode_m3bc(&header(0, 0, 0), &payload);
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let err = load(&bytes).unwrap_err().to_string();
        assert!(err.contains("CRC32"), "erro deve nomear CRC32: {}", err);
    }

    #[test]
    fn m3bc_wrong_major_rejected() {
        let mut h = header(0, 0, 0);
        h.major = M3BC_VERSION.0 + 1;
        let payload = instr_nop().encode();
        let bytes = encode_m3bc(&h, &payload);
        assert!(load(&bytes).is_err());
    }

    #[test]
    fn m3bc_entry_pc_alignment() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&instr_nop().encode());
        payload.extend_from_slice(&instr_nop().encode());
        // ENTRY_PC=32 alinha com o 2º frame.
        let bytes = encode_m3bc(&header(0, 0, 32), &payload);
        assert!(load(&bytes).is_ok());
        // ENTRY_PC=16 cai no meio do 1º frame => erro.
        let bytes = encode_m3bc(&header(0, 0, 16), &payload);
        assert!(load(&bytes).is_err());
    }

    #[test]
    fn m3bc_mixed_width_frames() {
        // [NOP 32B][op 0x80 + 63 bytes][NOP 32B] => I32, Raw(Fixed64), I32.
        let mut payload = Vec::new();
        payload.extend_from_slice(&instr_nop().encode());
        payload.push(0x80);
        payload.extend_from_slice(&[0u8; 63]);
        payload.extend_from_slice(&instr_nop().encode());
        let bytes = encode_m3bc(&header(0, 0, 0), &payload);
        let prog = load(&bytes).unwrap();
        assert_eq!(prog.frames.len(), 3);
        assert!(matches!(prog.frames[0], Frame::I32(_)));
        assert!(matches!(
            prog.frames[1],
            Frame::Raw {
                width: InstrWidth::Fixed64,
                ..
            }
        ));
        assert!(matches!(prog.frames[2], Frame::I32(_)));
        // instructions_32 deve falhar nomeando o frame largo (honesto).
        assert!(prog.instructions_32().is_err());
    }

    #[test]
    fn has_data_section_bit_frozen() {
        // Número congelado (RFC-0037 0037-04); wiring só com o layout.
        assert_eq!(M3BC_OPTIONAL_HAS_DATA_SECTION, 1 << 49);
        assert_eq!(M3BC_OPTIONAL_HAS_DATA_SECTION & M3BC_SUPPORTED_REQUIRED, 0);
    }

    #[test]
    fn m3bc_reserved_opcode_rejected() {
        // 0xB7 é RESERVED: sem tamanho assumido => erro nomeado.
        let payload = vec![0xB7u8; 32];
        let bytes = encode_m3bc(&header(0, 0, 0), &payload);
        let err = load(&bytes).unwrap_err().to_string();
        assert!(err.contains("reservado"), "erro deve nomear reservado: {}", err);
    }
}
