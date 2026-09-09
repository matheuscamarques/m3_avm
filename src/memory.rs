//! memory.rs — Gerenciador de memória da M³-AVM
//!
//! Espaço de endereçamento 128 bits, 4 regiões fixas:
//!   GLOBAL     (0x00...) — tensores imutáveis / código compartilhado. Heap linear via `Vec<u8>` + CoW (`Arc`).
//!   TEMPORAL   (0x10...) — buffer circular de 1 GB (lógico) para dados sensoriais, indexado por timestamp.
//!   PERSISTENTE(0x20...) — `mmap` em disco, sobrevive entre execuções (simula PCM).
//!   KV_CACHE   (0x30...) — cache de chaves/valores por camada, 1 GiB lógico / 64 MiB físico dev, com append seq.
//!
//! Design modular: basta substituir este arquivo por um driver `cudaMalloc`/`mmap` real.
//! O resto da VM só enxerga `MemoryManager::alloc/read/write`.

use anyhow::{anyhow, Result};
use memmap2::{Mmap, MmapMut};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::sync::Arc;
use crate::sparse::SparseTensor;

// ---------------------------------------------------------------------------
// Constantes de região
// ---------------------------------------------------------------------------

/// Prefixo do byte mais significativo (bits 120..127) para cada região.
pub const REGION_GLOBAL: u8 = 0x00;
pub const REGION_TEMPORAL: u8 = 0x10;
pub const REGION_PERSISTENT: u8 = 0x20;
pub const REGION_KV_CACHE: u8 = 0x30;

/// Tamanho lógico da região TEMPORAL: 1 GiB (spec).
pub const TEMPORAL_LOGICAL_SIZE: usize = 1024 * 1024 * 1024; // 1 GiB

/// Tamanho físico alocado em processo de desenvolvimento.
/// Produção deve ser 1 GiB; em dev usamos 64 MiB para não OOM em CI / notebooks.
pub const TEMPORAL_PHYSICAL_SIZE: usize = 64 * 1024 * 1024; // 64 MiB

/// Tamanho da região persistente (arquivo mmap).
pub const PERSISTENT_SIZE: usize = 64 * 1024 * 1024; // 64 MiB

/// Arquivo padrão para persistência.
pub const PERSISTENT_FILE: &str = "m3_persistent.dat";

/// Tamanho lógico da região KV_CACHE: 1 GiB (spec tese 2.1)
pub const KV_CACHE_LOGICAL_SIZE: usize = 1024 * 1024 * 1024; // 1 GiB lógico
/// Tamanho físico em dev (64 MiB por padrão, 256 MiB se quiser cabe 22 camadas)
pub const KV_CACHE_PHYSICAL_SIZE: usize = 64 * 1024 * 1024; // 64 MiB físico dev (suficiente para testes 22 camadas)
/// Máximo de tokens (seq_len) suportado no cache lógico
pub const KV_CACHE_MAX_SEQ: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Global,
    Temporal,
    Persistent,
    KvCache,
    Unknown,
}

/// Classifica endereço de 128 bits pela região.
pub fn region_of(addr: u128) -> Region {
    let prefix = (addr >> 120) as u8;
    match prefix {
        REGION_GLOBAL => Region::Global,
        REGION_TEMPORAL => Region::Temporal,
        REGION_PERSISTENT => Region::Persistent,
        REGION_KV_CACHE => Region::KvCache,
        _ => Region::Unknown,
    }
}

/// Helpers para construir endereços canônicos de cada região.
pub fn make_global_addr(offset: u128) -> u128 {
    // Garante que top byte é 0x00 e offset não colide
    offset & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128
}
pub fn make_temporal_addr(offset: u128) -> u128 {
    ((REGION_TEMPORAL as u128) << 120) | (offset & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128)
}
pub fn make_persistent_addr(offset: u128) -> u128 {
    ((REGION_PERSISTENT as u128) << 120) | (offset & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128)
}
pub fn make_kv_cache_addr(offset: u128) -> u128 {
    ((REGION_KV_CACHE as u128) << 120) | (offset & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128)
}
/// Helper para decodificar offset dentro de KV_CACHE (bits 0..119)
pub fn kv_cache_offset(addr: u128) -> usize {
    (addr & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128) as usize
}

// ---------------------------------------------------------------------------
// Metadados de tensor (para op ATTN)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    I8 = 2,
    U8 = 3,
    Q4_0 = 4,
    Q4_1 = 5,
    Q5_0 = 6,
    Q8_0 = 8,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
}

impl DType {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::F16,
            2 => Self::I8,
            3 => Self::U8,
            4 => Self::Q4_0,
            5 => Self::Q4_1,
            6 => Self::Q5_0,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            _ => Self::F32,
        }
    }
    pub fn from_u32(v: u32) -> Self { Self::from_u8(v as u8) }
    pub fn byte_width(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I8 => 1,
            Self::U8 => 1,
            Self::Q4_0 => 18, // 32 *0.5 +16 (não usado direto)
            Self::Q4_K => 144/256*4, // ~2.25 bytes por elemento em bloco, mas byte_width não aplicável
            _ => 1,
        }
    }
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Q4_0|Self::Q4_1|Self::Q5_0|Self::Q8_0|Self::Q4_K|Self::Q5_K|Self::Q6_K|Self::Q8_K)
    }
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub addr: u128,
    pub shape: Vec<usize>, // ex: [2,2]
    pub dtype: DType,
    pub byte_len: usize,
    pub is_sparse: bool,
    pub density: f32, // 0.0..1.0, só relevante se is_sparse
}

impl Default for TensorMeta {
    fn default() -> Self {
        Self { addr: 0, shape: vec![], dtype: DType::F32, byte_len: 0, is_sparse: false, density: 1.0 }
    }
}

// ---------------------------------------------------------------------------
// KV Cache — 22 camadas com buffer por layer (tese 2.1, 0x30...)
// ---------------------------------------------------------------------------

/// Cache de K/V por camada (seq_len * hidden floats por cache)
#[derive(Debug, Clone)]
pub struct KvCacheLayer {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub hidden: usize,
    pub seq_len: usize,
}

impl KvCacheLayer {
    pub fn new(hidden: usize) -> Self {
        Self { k: Vec::new(), v: Vec::new(), hidden, seq_len: 0 }
    }
    pub fn clear(&mut self) {
        self.k.clear();
        self.v.clear();
        self.seq_len = 0;
    }
    pub fn truncate(&mut self, new_seq_len: usize) {
        let h = self.hidden;
        self.k.truncate(new_seq_len * h);
        self.v.truncate(new_seq_len * h);
        self.seq_len = new_seq_len.min(self.seq_len);
        // seq_len from len/h after truncate
        self.seq_len = if h > 0 { self.k.len() / h } else { 0 };
    }
    pub fn push(&mut self, k_vec: &[f32], v_vec: &[f32]) {
        debug_assert_eq!(k_vec.len(), self.hidden);
        debug_assert_eq!(v_vec.len(), self.hidden);
        self.k.extend_from_slice(k_vec);
        self.v.extend_from_slice(v_vec);
        self.seq_len += 1;
    }
    pub fn seq_len(&self) -> usize { self.seq_len }
    pub fn is_empty(&self) -> bool { self.seq_len == 0 }
}

// ---------------------------------------------------------------------------
// MemoryManager
// ---------------------------------------------------------------------------

/// Gerenciador de memória com CoW simplificado via `Arc<Vec<u8>>`.
///
/// - `global_heap`: mapa endereço -> bloco Arc (CoW gratuito: clone é barato).
/// - `temporal`: buffer circular + head.
/// - `persistent`: `MmapMut` + HashMap de snapshots para ABORT.
/// - `kv_cache`: 22 camadas, cada com K/V buffers sequenciais + snapshots.
/// - `snapshots`: version -> clone do global_heap (para SENSE/ABORT transactional).
pub struct MemoryManager {
    // GLOBAL
    global_heap: HashMap<u128, Arc<Vec<u8>>>,
    pub(crate) tensor_meta: HashMap<u128, TensorMeta>,
    next_global_offset: u128,
    // SPARSE — CSR storage (PIM: computa onde está, sem mover)
    sparse_heap: HashMap<u128, SparseTensor>,

    // TEMPORAL
    temporal_buffer: Vec<u8>,
    temporal_head: usize,        // offset físico [0..PHYSICAL)
    temporal_logical_head: u128, // offset lógico [0..LOGICAL) — usado para endereçamento
    temporal_base_addr: u128,    // prefixo temporal

    // PERSISTENT
    persistent_mmap: MmapMut,
    #[allow(dead_code)]
    persistent_file: File,
    // GGUF f32 mapping (para TinyLlama/DeepSeek f32) — lazy mmap sem cópia
    pub(crate) gguf: Option<crate::gguf::GgufFile>,
    pub(crate) gguf_data_base: u128, // offset base dos dados no arquivo (para cálculo addr)
    pub(crate) model_mmap: Option<Arc<Mmap>>, // mmap readonly do modelo, zero-copy
    pub(crate) model_path: Option<String>,

    // KV_CACHE (0x30...) — 22 camadas
    kv_cache_layers: Vec<KvCacheLayer>,
    kv_cache_max_seq: usize,
    kv_cache_base_addr: u128,
    // Para generic read/write na região KV_CACHE (byte view)
    kv_heap: HashMap<u128, Arc<Vec<u8>>>,
    kv_next_offset: u128,

    // Snapshot / versão (para ABORT e FORK)
    version: u64,
    snapshots: HashMap<u64, HashMap<u128, Arc<Vec<u8>>>>,
    sparse_snapshots: HashMap<u64, HashMap<u128, SparseTensor>>,
    kv_snapshots: HashMap<u64, Vec<KvCacheLayer>>,
    kv_heap_snapshots: HashMap<u64, HashMap<u128, Arc<Vec<u8>>>>,
}

impl MemoryManager {
    pub fn new() -> Result<Self> {
        Self::with_persistent_file(PERSISTENT_FILE)
    }

    pub fn new_with_size(persistent_bytes: usize) -> Result<Self> {
        Self::with_persistent_file_sized(PERSISTENT_FILE, persistent_bytes)
    }

    pub fn with_persistent_file_sized(path: &str, size: usize) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        file.set_len(size as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(Self {
            global_heap: HashMap::new(),
            tensor_meta: HashMap::new(),
            next_global_offset: 0x1000,
            sparse_heap: HashMap::new(),
            temporal_buffer: vec![0u8; TEMPORAL_PHYSICAL_SIZE],
            temporal_head: 0,
            temporal_logical_head: 0,
            temporal_base_addr: (REGION_TEMPORAL as u128) << 120,
            persistent_mmap: mmap,
            persistent_file: file,
            gguf: None,
            gguf_data_base: 0,
            model_mmap: None,
            model_path: None,
            kv_cache_layers: Vec::new(),
            kv_cache_max_seq: KV_CACHE_MAX_SEQ,
            kv_cache_base_addr: (REGION_KV_CACHE as u128) << 120,
            kv_heap: HashMap::new(),
            kv_next_offset: 0x1000,
            version: 0,
            snapshots: HashMap::new(),
            sparse_snapshots: HashMap::new(),
            kv_snapshots: HashMap::new(),
            kv_heap_snapshots: HashMap::new(),
        })
    }

    pub fn with_persistent_file(path: &str) -> Result<Self> {
        // Abre/cria arquivo persistente
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        file.set_len(PERSISTENT_SIZE as u64)?;
        // SAFETY: MmapMut de arquivo recém-criado
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(Self {
            global_heap: HashMap::new(),
            tensor_meta: HashMap::new(),
            next_global_offset: 0x1000, // reserva primeiros 4 KiB para NULL + cabeçalhos
            sparse_heap: HashMap::new(),
            temporal_buffer: vec![0u8; TEMPORAL_PHYSICAL_SIZE],
            temporal_head: 0,
            temporal_logical_head: 0,
            temporal_base_addr: (REGION_TEMPORAL as u128) << 120,
            persistent_mmap: mmap,
            persistent_file: file,
            gguf: None,
            gguf_data_base: 0,
            model_mmap: None,
            model_path: None,
            kv_cache_layers: Vec::new(),
            kv_cache_max_seq: KV_CACHE_MAX_SEQ,
            kv_cache_base_addr: (REGION_KV_CACHE as u128) << 120,
            kv_heap: HashMap::new(),
            kv_next_offset: 0x1000,
            version: 0,
            snapshots: HashMap::new(),
            sparse_snapshots: HashMap::new(),
            kv_snapshots: HashMap::new(),
            kv_heap_snapshots: HashMap::new(),
        })
    }

    /// Construtor sem persistência (para testes sem I/O).
    pub fn new_in_memory() -> Self {
        // Cria um arquivo anônimo via memmap2? Simula com Vec mas precisamos MmapMut.
        // Truque: cria um Vec e transmuta via mmap anonymous.
        // Mais simples: abre um arquivo temporário em /tmp.
        Self::with_persistent_file("/tmp/m3_persistent_test.dat")
            .unwrap_or_else(|_| {
                // Fallback absoluto sem mmap (para ambientes restritos)
                // Cria mmap anônimo via `memmap2::MmapMut::map_anon`
                let mmap = memmap2::MmapMut::map_anon(PERSISTENT_SIZE).expect("anonymous mmap");
                Self {
                    global_heap: HashMap::new(),
                    tensor_meta: HashMap::new(),
                    next_global_offset: 0x1000,
                    sparse_heap: HashMap::new(),
                    temporal_buffer: vec![0u8; TEMPORAL_PHYSICAL_SIZE],
                    temporal_head: 0,
                    temporal_logical_head: 0,
                    temporal_base_addr: (REGION_TEMPORAL as u128) << 120,
                    persistent_mmap: mmap,
                    persistent_file: File::create("/tmp/m3_persist_dummy").unwrap(),
                    gguf: None,
                    gguf_data_base: 0,
                    model_mmap: None,
                    model_path: None,
                    kv_cache_layers: Vec::new(),
                    kv_cache_max_seq: KV_CACHE_MAX_SEQ,
                    kv_cache_base_addr: (REGION_KV_CACHE as u128) << 120,
                    kv_heap: HashMap::new(),
                    kv_next_offset: 0x1000,
                    version: 0,
                    snapshots: HashMap::new(),
                    sparse_snapshots: HashMap::new(),
                    kv_snapshots: HashMap::new(),
                    kv_heap_snapshots: HashMap::new(),
                }
            })
    }

    // -----------------------------------------------------------------------
    // Alocação GLOBAL (TENSOR)
    // -----------------------------------------------------------------------

    /// Aloca `size` bytes na região GLOBAL, retorna endereço canônico 128-bit.
    /// Alinhado a 64 bytes (cache-line friendly).
    pub fn alloc_global(&mut self, size: usize) -> Result<u128> {
        if size == 0 {
            return Err(anyhow!("alloc_global: size zero"));
        }
        let aligned = (size + 63) & !63;
        let addr = make_global_addr(self.next_global_offset);
        let block = Arc::new(vec![0u8; aligned]);
        self.global_heap.insert(addr, block);
        self.next_global_offset += aligned as u128;
        Ok(addr)
    }

    /// Tenta mapear tensor do GGUF para shape (zero-copy PERSISTENTE)
    /// Se shape [2048,2048] bater com `blk.0.attn_q.weight` (f32 ou Q4_K), retorna addr PERSISTENTE
    /// Suporta F32 (0) e Q4_K (12) — para Q4_K, dtype da meta será Q4_K e leitura fará dequant
    pub fn try_gguf_tensor(&mut self, shape: &[usize], dtype: DType) -> Option<u128> {
        let gg = self.gguf.as_ref()?;
        let elems: usize = shape.iter().product();
        // Para dummy 2x8 (16 elems) não mapear para Q4_K (que exige múltiplo de 256), evita falso positivo
        let is_small = elems < 256;
        for t in &gg.tensors {
            if t.n_elements != elems { continue; }
            // Filtra por dtype: se GGUF é quantizado e request é F32, permite (dequant)
            // Mas se GGUF é F32 e request é F32, ok. Se GGUF é Q4_K e request é Q4_K, ok.
            // Para small shapes, só aceita F32 para evitar Q4_K com 16 elems (não múltiplo 256)
            if is_small && t.dtype != 0 { continue; }
            let gg_dtype = DType::from_u32(t.dtype);
            // Se request é quantizado, só mapeia se gg dtype igual; se request é F32, mapeia tanto F32 quanto Q4_K
            if dtype.is_quantized() && gg_dtype != dtype { continue; }
            // Checa shape exato ou invertida
            let mut shape_match = t.shape.len()==shape.len() && t.shape.iter().zip(shape).all(|(a,b)| *a as usize==*b);
            if !shape_match {
                let rev: Vec<usize> = t.shape.iter().rev().map(|&x| x as usize).collect();
                shape_match = rev == shape;
            }
            if shape_match {
                let file_offset = gg.data_offset + t.offset;
                let addr = make_persistent_addr(file_offset as u128);
                // Evita mapear mesmo tensor duas vezes para shapes iguais (Q/K/V/O todos 2048x2048)
                // — cada TENSOR com mesma shape deve pegar o próximo tensor livre com mesma shape
                if self.tensor_meta.contains_key(&addr) {
                    continue;
                }
                // byte_len depende do dtype real do GGUF
                let byte_len = match gg_dtype {
                    DType::F32 => elems*4,
                    DType::F16 => elems*2,
                    DType::Q4_K | DType::Q5_K | DType::Q6_K => (elems/256)*144,
                    DType::Q4_0 => (elems/32)*18,
                    _ => elems*4,
                };
                let meta = TensorMeta { addr, shape: shape.to_vec(), dtype: gg_dtype, byte_len, is_sparse: false, density: 1.0 };
                self.tensor_meta.insert(addr, meta);
                eprintln!("[gguf] TENSOR {:?} {} mapeado para GGUF '{}' shape {:?} dtype {} @ PERSISTENTE 0x{:x} ({} bytes)", shape, if gg_dtype.is_quantized() {"Q4_K"} else {"f32"}, t.name, t.shape, t.dtype, addr, byte_len);
                return Some(addr);
            }
        }
        None
    }

    /// Variante que registra metadados de tensor (shape + dtype).
    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> Result<u128> {
        // Tenta zero-copy do GGUF f32 antes de alocar dummy
        if let Some(addr) = self.try_gguf_tensor(shape, dtype) {
            return Ok(addr);
        }
        let elems: usize = shape.iter().product();
        let byte_len = elems * dtype.byte_width();
        let addr = self.alloc_global(byte_len)?;
        let meta = TensorMeta {
            addr,
            shape: shape.to_vec(),
            dtype,
            byte_len,
            is_sparse: false,
            density: 1.0,
        };
        self.tensor_meta.insert(addr, meta);
        Ok(addr)
    }

    /// Aloca tensor esparso CSR (PIM: sem mover dados, computa onde está)
    pub fn alloc_sparse_tensor(&mut self, shape: &[usize], dtype: DType, density: f32) -> Result<u128> {
        if shape.len() != 2 { return Err(anyhow!("sparse apenas 2D suportado")); }
        let (rows, cols) = (shape[0], shape[1]);
        let density = density.clamp(0.01, 1.0);
        let elems = rows * cols;
        let nnz_est = (elems as f32 * density) as usize;
        // Reserva espaço proporcional a nnz (CSR: valores + índices)
        let byte_len = nnz_est * dtype.byte_width() + (rows+1)*4 + nnz_est*4;
        let addr = self.alloc_global(byte_len.max(64))?;
        let mut rng = rand::thread_rng();
        let sparse = crate::sparse::SparseTensor::random((rows, cols), density, &mut rng);
        let nnz = sparse.nnz;
        self.sparse_heap.insert(addr, sparse);
        let meta = TensorMeta {
            addr,
            shape: shape.to_vec(),
            dtype,
            byte_len,
            is_sparse: true,
            density,
        };
        self.tensor_meta.insert(addr, meta);
        // Também marca quantidade real
        if let Some(st) = self.sparse_heap.get(&addr) {
            // atualiza com nnz real
            let _ = nnz;
        }
        Ok(addr)
    }

    pub fn is_sparse(&self, addr: u128) -> bool {
        self.tensor_meta.get(&addr).map(|m| m.is_sparse).unwrap_or(false)
            || self.sparse_heap.contains_key(&addr)
    }

    pub fn get_sparse(&self, addr: u128) -> Option<&crate::sparse::SparseTensor> {
        self.sparse_heap.get(&addr)
    }

    pub fn get_sparse_mut(&mut self, addr: u128) -> Option<&mut crate::sparse::SparseTensor> {
        self.sparse_heap.get_mut(&addr)
    }

    pub fn get_tensor_meta(&self, addr: u128) -> Option<&TensorMeta> {
        self.tensor_meta.get(&addr)
    }

    /// Zero-copy: fatia de bytes brutos do modelo direto do mmap (sem `Vec`).
    /// `file_offset` usa a mesma base de `GgufFile.data_offset + tensor.offset`.
    pub fn get_model_raw_slice(&self, file_offset: u64, len: usize) -> Option<&[u8]> {
        let mmap = self.model_mmap.as_ref()?;
        let start = file_offset as usize;
        let end = start.checked_add(len)?;
        if end <= mmap.len() {
            Some(&mmap[start..end])
        } else {
            None
        }
    }

    /// Zero-copy: retorna &[f32] direto do mmap se tensor é F32 e está em model_mmap (sem Vec)
    pub fn get_tensor_f32_slice(&self, addr: u128) -> Option<&[f32]> {
        let meta = self.get_tensor_meta(addr)?;
        if meta.dtype != DType::F32 { return None; }
        let offset = (addr & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128) as usize;
        // Tenta model_mmap (lazy, sem cópia) se offset dentro da região de dados do modelo
        if let Some(mmap) = &self.model_mmap {
            let data_base = self.gguf_data_base as usize;
            if offset >= data_base && offset + meta.byte_len <= mmap.len() {
                let ptr = mmap.as_ptr() as *const u8;
                // f32 requer alinhamento 4
                if (ptr as usize + offset) % 4 == 0 {
                    let fptr = unsafe { ptr.add(offset) as *const f32 };
                    let len = meta.byte_len / 4;
                    unsafe { return Some(std::slice::from_raw_parts(fptr, len)); }
                }
            }
        }
        // Fallback: tenta global_heap (para tensores dummy 2x8)
        if let Some(block) = self.global_heap.get(&addr) {
            if block.len() >= meta.byte_len && (block.as_ptr() as usize) % 4 == 0 {
                let ptr = block.as_ptr() as *const f32;
                let len = meta.byte_len / 4;
                unsafe { return Some(std::slice::from_raw_parts(ptr, len)); }
            }
        }
        None
    }

    // -----------------------------------------------------------------------
    // Read / Write genérico (dispatch por região)
    // -----------------------------------------------------------------------

    pub fn read(&self, addr: u128, len: usize) -> Result<Vec<u8>> {
        match region_of(addr) {
            Region::Global => {
                // Busca exata; para simplicidade não suportamos leitura cross-block.
                // Em prod, haveria paginação.
                if let Some(block) = self.global_heap.get(&addr) {
                    if len > block.len() {
                        return Err(anyhow!(
                            "read GLOBAL out of bounds: addr {:032x} len {} > block {}",
                            addr,
                            len,
                            block.len()
                        ));
                    }
                    Ok(block[..len].to_vec())
                } else {
                    // Tenta buscar bloco que contém addr (addr dentro de bloco)
                    for (base, block) in &self.global_heap {
                        let offset = addr.checked_sub(*base);
                        if let Some(off) = offset {
                            let off = off as usize;
                            if off + len <= block.len() {
                                return Ok(block[off..off + len].to_vec());
                            }
                        }
                    }
                    Err(anyhow!("read GLOBAL: endereço não mapeado {:032x}", addr))
                }
            }
            Region::Temporal => {
                let logical_offset = (addr & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128) as usize;
                let phys_offset = logical_offset % TEMPORAL_PHYSICAL_SIZE;
                if phys_offset + len <= TEMPORAL_PHYSICAL_SIZE {
                    Ok(self.temporal_buffer[phys_offset..phys_offset + len].to_vec())
                } else {
                    // Wrapping read circular
                    let mut out = Vec::with_capacity(len);
                    let first = TEMPORAL_PHYSICAL_SIZE - phys_offset;
                    out.extend_from_slice(&self.temporal_buffer[phys_offset..]);
                    out.extend_from_slice(&self.temporal_buffer[..len - first]);
                    Ok(out)
                }
            }
            Region::Persistent => {
                let offset_raw = (addr & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128) as usize;
                // Se é dado de modelo (mmap lazy), lê direto do model_mmap sem cópia
                if let Some(mmap) = &self.model_mmap {
                    let data_base = self.gguf_data_base as usize;
                    if offset_raw >= data_base && offset_raw + len <= mmap.len() {
                        return Ok(mmap[offset_raw..offset_raw + len].to_vec());
                    }
                    // Também tenta offset % mmap.len() para compat com % psize anterior
                    let off2 = offset_raw % mmap.len();
                    if off2 + len <= mmap.len() && off2 >= data_base {
                        // Heurística: se offset dentro da região de dados do modelo, usa model_mmap
                        // Para evitar conflito com persistent 0x1000, só usa se off2 >= data_base
                        return Ok(mmap[off2..off2 + len].to_vec());
                    }
                }
                let psize = self.persistent_mmap.len();
                let offset = offset_raw % psize;
                if offset + len > psize {
                    return Err(anyhow!("read PERSISTENT out of bounds"));
                }
                Ok(self.persistent_mmap[offset..offset + len].to_vec())
            }
            Region::KvCache => {
                // Tenta kv_heap primeiro (alocações genéricas)
                if let Some(block) = self.kv_heap.get(&addr) {
                    if len > block.len() {
                        return Err(anyhow!("read KV_CACHE out of bounds: len {} > block {}", len, block.len()));
                    }
                    return Ok(block[..len].to_vec());
                }
                // Tenta buscar bloco que contém addr
                for (base, block) in &self.kv_heap {
                    let off = addr.checked_sub(*base);
                    if let Some(off) = off {
                        let off = off as usize;
                        if off + len <= block.len() {
                            return Ok(block[off..off+len].to_vec());
                        }
                    }
                }
                Err(anyhow!("read KV_CACHE: endereço não mapeado {:032x}", addr))
            }
            Region::Unknown => Err(anyhow!("read: região desconhecida para addr {:032x}", addr)),
        }
    }

    pub fn write(&mut self, addr: u128, data: &[u8]) -> Result<()> {
        match region_of(addr) {
            Region::Global => {
                // CoW: se Arc tem múltiplas refs, clona antes de escrever
                if let Some(block) = self.global_heap.get_mut(&addr) {
                    // Arc::make_mut garante CoW
                    let vec = Arc::make_mut(block);
                    if data.len() > vec.len() {
                        return Err(anyhow!("write GLOBAL: data maior que bloco"));
                    }
                    vec[..data.len()].copy_from_slice(data);
                    Ok(())
                } else {
                    // Busca por base
                    let mut found: Option<u128> = None;
                    for (base, block) in &self.global_heap {
                        let off = addr.checked_sub(*base);
                        if let Some(off) = off {
                            let off = off as usize;
                            if off + data.len() <= block.len() {
                                found = Some(*base);
                                break;
                            }
                        }
                    }
                    if let Some(base) = found {
                        let block = self.global_heap.get_mut(&base).unwrap();
                        let off = (addr - base) as usize;
                        let vec = Arc::make_mut(block);
                        vec[off..off + data.len()].copy_from_slice(data);
                        Ok(())
                    } else {
                        Err(anyhow!("write GLOBAL: endereço não mapeado {:032x}", addr))
                    }
                }
            }
            Region::Temporal => {
                let logical_offset =
                    (addr & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFFu128) as usize;
                let phys_offset = logical_offset % TEMPORAL_PHYSICAL_SIZE;
                if phys_offset + data.len() <= TEMPORAL_PHYSICAL_SIZE {
                    self.temporal_buffer[phys_offset..phys_offset + data.len()]
                        .copy_from_slice(data);
                } else {
                    let first = TEMPORAL_PHYSICAL_SIZE - phys_offset;
                    self.temporal_buffer[phys_offset..].copy_from_slice(&data[..first]);
                    self.temporal_buffer[..data.len() - first].copy_from_slice(&data[first..]);
                }
                Ok(())
            }
            Region::Persistent => {
                let psize = self.persistent_mmap.len();
                let offset = (addr & 0x00FF_FFFF_FFFF_FFFF_FFFF_FFFFu128) as usize
                    % psize;
                if offset + data.len() > psize {
                    return Err(anyhow!("write PERSISTENT out of bounds"));
                }
                self.persistent_mmap[offset..offset + data.len()].copy_from_slice(data);
                // Flush é opcional; chamamos flush assíncrono periodicamente
                Ok(())
            }
            Region::KvCache => {
                if let Some(block) = self.kv_heap.get_mut(&addr) {
                    let vec = Arc::make_mut(block);
                    if data.len() > vec.len() {
                        return Err(anyhow!("write KV_CACHE: data maior que bloco"));
                    }
                    vec[..data.len()].copy_from_slice(data);
                    return Ok(());
                }
                let mut found: Option<u128> = None;
                for (base, block) in &self.kv_heap {
                    let off = addr.checked_sub(*base);
                    if let Some(off) = off {
                        let off = off as usize;
                        if off + data.len() <= block.len() {
                            found = Some(*base);
                            break;
                        }
                    }
                }
                if let Some(base) = found {
                    let block = self.kv_heap.get_mut(&base).unwrap();
                    let off = (addr - base) as usize;
                    let vec = Arc::make_mut(block);
                    vec[off..off+data.len()].copy_from_slice(data);
                    Ok(())
                } else {
                    Err(anyhow!("write KV_CACHE: endereço não mapeado {:032x}", addr))
                }
            }
                        Region::Unknown => Err(anyhow!("write: região desconhecida {:032x}", addr)),
        }
    }

    // -----------------------------------------------------------------------
    // TEMPORAL — buffer circular
    // -----------------------------------------------------------------------

    /// Escreve `data` no buffer circular TEMPORAL e retorna endereço lógico (128-bit).
    pub fn temporal_push(&mut self, data: &[u8]) -> Result<u128> {
        let len = data.len();
        if len > TEMPORAL_PHYSICAL_SIZE {
            return Err(anyhow!("temporal_push: dado maior que buffer físico"));
        }
        let logical_addr = self.temporal_base_addr | self.temporal_logical_head;
        // Escrita física com wrap
        if self.temporal_head + len <= TEMPORAL_PHYSICAL_SIZE {
            self.temporal_buffer[self.temporal_head..self.temporal_head + len]
                .copy_from_slice(data);
            self.temporal_head += len;
        } else {
            let first = TEMPORAL_PHYSICAL_SIZE - self.temporal_head;
            self.temporal_buffer[self.temporal_head..].copy_from_slice(&data[..first]);
            self.temporal_buffer[..len - first].copy_from_slice(&data[first..]);
            self.temporal_head = len - first;
        }
        self.temporal_logical_head =
            (self.temporal_logical_head + len as u128) % TEMPORAL_LOGICAL_SIZE as u128;
        if self.temporal_head >= TEMPORAL_PHYSICAL_SIZE {
            self.temporal_head = 0;
        }
        Ok(logical_addr)
    }

    /// Lê `len` bytes a partir de um endereço temporal.
    pub fn temporal_read(&self, addr: u128, len: usize) -> Result<Vec<u8>> {
        self.read(addr, len)
    }

    // -----------------------------------------------------------------------
    // PERSISTENT — flush
    // -----------------------------------------------------------------------

    pub fn persistent_flush(&self) -> Result<()> {
        self.persistent_mmap.flush()?;
        Ok(())
    }

    pub fn persistent_write(&mut self, offset: usize, data: &[u8]) -> Result<u128> {
        let addr = make_persistent_addr(offset as u128);
        self.write(addr, data)?;
        Ok(addr)
    }

    /// Carrega modelo GGUF via mmap zero-copy (lazy, sem cópia para PERSISTENTE)
    /// Usa `Mmap` readonly + `madvise(WillNeed)` e guarda `Arc<Mmap>` em `model_mmap`
    /// Tensores são lidos sob demanda via `get_tensor_f32_slice` (ponteiro direto)
    /// Retorna endereço base (0x2000_0000_0000_0000).
    pub fn load_gguf_model(&mut self, path: &str) -> Result<u128> {
        use std::fs::File;
        use memmap2::Mmap;
        let file = File::open(path).map_err(|e| anyhow!("load_gguf_model open {}: {}", path, e))?;
        let len = file.metadata().map_err(|e| anyhow!("metadata {}: {}", path, e))?.len() as usize;
        if len == 0 {
            return Err(anyhow!("modelo vazio: {}", path));
        }
        // Para lazy mmap, não precisa caber em PERSISTENTE — usa model_mmap separado
        // Mantém check apenas como aviso se len > 8G
        if len > 8*1024*1024*1024 {
            eprintln!("[gguf] aviso: modelo {}MiB muito grande", len/(1024*1024));
        }
        // SAFETY: Mmap readonly do arquivo
        let mmap = unsafe { Mmap::map(&file).map_err(|e| anyhow!("mmap {}: {}", path, e))? };
        // Dica ao OS para pré-carregar páginas em background (WillNeed) — não bloqueia
        #[cfg(unix)]
        {
            unsafe {
                let ptr = mmap.as_ptr() as *mut libc::c_void;
                let len = mmap.len();
                let ret = libc::madvise(ptr, len, libc::MADV_WILLNEED);
                if ret == 0 {
                    eprintln!("[gguf] madvise WillNeed ok {}MiB", len/(1024*1024));
                }
            }
        }
        let mmap_arc = Arc::new(mmap);
        self.model_mmap = Some(mmap_arc.clone());
        self.model_path = Some(path.to_string());
        // Parse GGUF header para mapear tensores (sem copiar dados)
        match crate::gguf::GgufFile::open(path) {
            Ok(gg) => {
                let n_f32 = gg.tensors_by_dtype(0).len();
                let n_q4k = gg.tensors_by_dtype(12).len() + gg.tensors_by_dtype(13).len();
                eprintln!("[gguf] {} tensores, {} F32, {} Q4_K, data_offset 0x{:x} (lazy mmap, sem cópia)", gg.n_tensors, n_f32, n_q4k, gg.data_offset);
                self.gguf_data_base = gg.data_offset as u128;
                self.gguf = Some(gg);
            }
            Err(e) => {
                eprintln!("[gguf] aviso: falha ao parsear GGUF {}: {} — usando dummy", path, e);
                self.gguf = None;
            }
        }
        // Retorna base canônica
        Ok(make_persistent_addr(0))
    }

    /// Variante que carrega bytes já em memória (para testes sem arquivo).
    pub fn load_gguf_bytes(&mut self, bytes: &[u8]) -> Result<u128> {
        let psize = self.persistent_mmap.len();
        if bytes.len() > psize {
            return Err(anyhow!("bytes {} > PERSISTENTE {}", bytes.len(), psize));
        }
        self.persistent_mmap[..bytes.len()].copy_from_slice(bytes);
        let _ = self.persistent_mmap.flush();
        Ok(make_persistent_addr(0))
    }

    // -----------------------------------------------------------------------
    // KV_CACHE — 22 camadas com buffer seq_len * hidden
    // -----------------------------------------------------------------------

    /// Inicializa ou redimensiona o KV cache para `n_layers` com `hidden` por layer.
    /// Deve ser chamado antes do loop de inferência. Reutiliza buffers se já inicializado com mesmo hidden.
    pub fn kv_cache_init(&mut self, n_layers: usize, hidden: usize) {
        if self.kv_cache_layers.len() == n_layers && self.kv_cache_layers.first().map(|l| l.hidden == hidden).unwrap_or(false) {
            // já inicializado, apenas limpa seq
            self.kv_cache_clear();
            return;
        }
        self.kv_cache_layers = (0..n_layers).map(|_| KvCacheLayer::new(hidden)).collect();
        self.kv_cache_max_seq = KV_CACHE_MAX_SEQ;
        self.kv_cache_base_addr = (REGION_KV_CACHE as u128) << 120;
        self.kv_next_offset = 0x1000;
    }

    /// Retorna número de layers inicializados.
    pub fn kv_cache_n_layers(&self) -> usize { self.kv_cache_layers.len() }

    /// Retorna seq_len atual (assume todas layers têm mesmo seq_len).
    pub fn kv_cache_seq_len(&self) -> usize {
        self.kv_cache_layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    /// Retorna hidden configurado.
    pub fn kv_cache_hidden(&self) -> usize {
        self.kv_cache_layers.first().map(|l| l.hidden).unwrap_or(0)
    }

    /// Append de K/V para uma camada específica.
    pub fn kv_cache_append(&mut self, layer: usize, k_vec: &[f32], v_vec: &[f32]) -> Result<()> {
        if layer >= self.kv_cache_layers.len() {
            return Err(anyhow!("kv_cache_append: layer {} >= n_layers {}", layer, self.kv_cache_layers.len()));
        }
        let hidden = self.kv_cache_layers[layer].hidden;
        if k_vec.len() != hidden || v_vec.len() != hidden {
            return Err(anyhow!("kv_cache_append: k/v len {} / {} != hidden {}", k_vec.len(), v_vec.len(), hidden));
        }
        if self.kv_cache_layers[layer].seq_len >= self.kv_cache_max_seq {
            return Err(anyhow!("kv_cache_append: seq_len {} atingiu max {}", self.kv_cache_layers[layer].seq_len, self.kv_cache_max_seq));
        }
        self.kv_cache_layers[layer].push(k_vec, v_vec);
        Ok(())
    }

    /// Lê K cache de uma camada (flat seq_len*hidden).
    pub fn kv_cache_get_k(&self, layer: usize) -> Option<&[f32]> {
        self.kv_cache_layers.get(layer).map(|l| l.k.as_slice())
    }
    pub fn kv_cache_get_v(&self, layer: usize) -> Option<&[f32]> {
        self.kv_cache_layers.get(layer).map(|l| l.v.as_slice())
    }

    /// Trunca todas as camadas para `new_seq_len` (rollback seletivo).
    pub fn kv_cache_truncate(&mut self, new_seq_len: usize) {
        for layer in &mut self.kv_cache_layers {
            layer.truncate(new_seq_len);
        }
    }

    /// Limpa todo o KV cache (início de nova geração).
    pub fn kv_cache_clear(&mut self) {
        for layer in &mut self.kv_cache_layers {
            layer.clear();
        }
    }

    /// Aloca um tensor no KV_CACHE (para debug via endereços 0x30...)
    /// Retorna addr canônico na região KV_CACHE.
    pub fn alloc_kv_tensor(&mut self, size: usize) -> Result<u128> {
        if size == 0 { return Err(anyhow!("alloc_kv: size zero")); }
        let aligned = (size + 63) & !63;
        let addr = make_kv_cache_addr(self.kv_next_offset);
        let block = Arc::new(vec![0u8; aligned]);
        self.kv_heap.insert(addr, block);
        self.kv_next_offset += aligned as u128;
        Ok(addr)
    }

    /// Estatísticas rápidas do KV cache.
    pub fn kv_cache_stats(&self) -> String {
        let n = self.kv_cache_layers.len();
        let seq = self.kv_cache_seq_len();
        let h = self.kv_cache_hidden();
        let mem_mb = if h>0 { (n * seq * h * 2 * 4) as f64 / (1024.0*1024.0) } else { 0.0 };
        format!("KV_CACHE layers={} seq_len={} hidden={} mem={:.2}MiB logical=1GiB", n, seq, h, mem_mb)
    }

    /// Calcula atenção com KV cache para uma camada (single query).
    /// q: [hidden], k_cached: [seq_len*hidden], v_cached: [seq_len*hidden] + k_cur/v_cur já inclusos?
    /// Aqui `q` é query atual, `k_cache`/`v_cache` já contêm o token atual após append.
    /// Retorna attn_out: [hidden] com softmax sobre seq_len.
    pub fn kv_cache_attention(&self, layer: usize, q: &[f32]) -> Result<Vec<f32>> {
        let l = self.kv_cache_layers.get(layer).ok_or_else(|| anyhow!("kv layer {} não existe", layer))?;
        let h = l.hidden;
        let seq = l.seq_len;
        if seq == 0 { return Err(anyhow!("kv_cache_attention: seq_len 0")); }
        if q.len() != h { return Err(anyhow!("q len {} != hidden {}", q.len(), h)); }
        if l.k.len() != seq * h || l.v.len() != seq * h { return Err(anyhow!("kv cache inconsistent")); }
        let scale = 1.0 / (h as f32).sqrt(); // se multi-head, usar head_dim; aqui hidden como single head
        // scores = q·k_i * scale
        let mut scores = vec![0.0f32; seq];
        for i in 0..seq {
            let k_i = &l.k[i*h..(i+1)*h];
            let mut dot = 0.0;
            for (a,b) in q.iter().zip(k_i.iter()) { dot += a * b; }
            scores[i] = dot * scale;
        }
        // softmax
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum; }
        // weighted sum V
        let mut out = vec![0.0f32; h];
        for i in 0..seq {
            let v_i = &l.v[i*h..(i+1)*h];
            let w = scores[i];
            for j in 0..h { out[j] += w * v_i[j]; }
        }
        Ok(out)
    }

    /// Versão com head_dim separado (para n_heads >1)
    pub fn kv_cache_attention_head_dim(&self, layer: usize, q: &[f32], head_dim: usize) -> Result<Vec<f32>> {
        let l = self.kv_cache_layers.get(layer).ok_or_else(|| anyhow!("kv layer {} não existe", layer))?;
        let h = l.hidden;
        let seq = l.seq_len;
        if seq == 0 { return Err(anyhow!("kv_cache_attention: seq_len 0")); }
        if q.len() != h { return Err(anyhow!("q len {} != hidden {}", q.len(), h)); }
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = vec![0.0f32; seq];
        for i in 0..seq {
            let k_i = &l.k[i*h..(i+1)*h];
            let mut dot = 0.0;
            for (a,b) in q.iter().zip(k_i.iter()) { dot += a * b; }
            scores[i] = dot * scale;
        }
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for s in scores.iter_mut() { *s = (*s - max).exp(); sum += *s; }
        for s in scores.iter_mut() { *s /= sum; }
        let mut out = vec![0.0f32; h];
        for i in 0..seq {
            let v_i = &l.v[i*h..(i+1)*h];
            let w = scores[i];
            for j in 0..h { out[j] += w * v_i[j]; }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Snapshots / Versionamento (FORK / ABORT)
    // -----------------------------------------------------------------------

    /// Cria snapshot da heap GLOBAL (CoW: clones de Arc são baratos).
    /// Também snapshot da sparse_heap para rollback de matriz esparsa e KV cache.
    pub fn snapshot(&mut self) -> u64 {
        self.version += 1;
        let snap = self.global_heap.clone(); // Arc clones
        self.snapshots.insert(self.version, snap);
        let sparse_snap = self.sparse_heap.clone();
        self.sparse_snapshots.insert(self.version, sparse_snap);
        // KV cache snapshot (22 camadas)
        let kv_snap = self.kv_cache_layers.clone();
        self.kv_snapshots.insert(self.version, kv_snap);
        let kv_heap_snap = self.kv_heap.clone();
        self.kv_heap_snapshots.insert(self.version, kv_heap_snap);
        self.version
    }

    pub fn restore(&mut self, version: u64) -> Result<()> {
        let mut ok = false;
        if let Some(snap) = self.snapshots.get(&version) {
            self.global_heap = snap.clone();
            ok = true;
        }
        if let Some(sparse_snap) = self.sparse_snapshots.get(&version) {
            self.sparse_heap = sparse_snap.clone();
            ok = true;
        }
        if let Some(kv_snap) = self.kv_snapshots.get(&version) {
            self.kv_cache_layers = kv_snap.clone();
            ok = true;
        }
        if let Some(kv_heap_snap) = self.kv_heap_snapshots.get(&version) {
            self.kv_heap = kv_heap_snap.clone();
            ok = true;
        }
        if ok {
            self.version = version;
            Ok(())
        } else {
            Err(anyhow!("restore: versão {} não encontrada", version))
        }
    }

    pub fn current_version(&self) -> u64 {
        self.version
    }

    // -----------------------------------------------------------------------
    // Helpers para ATTN (leitura/escrita f32)
    // -----------------------------------------------------------------------

    pub fn write_f32_tensor(&mut self, addr: u128, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write(addr, &bytes)
    }

    pub fn read_f32_tensor(&self, addr: u128, count: usize) -> Result<Vec<f32>> {
        // Zero-copy para F32 em model_mmap (sem Vec<u8> intermediário)
        if let Some(slice) = self.get_tensor_f32_slice(addr) {
            if slice.len() >= count {
                return Ok(slice[..count].to_vec());
            }
        }
        // Se tensor é quantizado (Q4_K), dequantiza via kernel SIMD
        if let Some(meta) = self.get_tensor_meta(addr) {
            if meta.dtype.is_quantized() {
                let raw = self.read(addr, meta.byte_len)?;
                let mut dst = vec![0.0f32; count];
                let dtype_u32 = match meta.dtype {
                    DType::Q4_0 => 2,
                    DType::Q4_K => 12,
                    DType::Q5_K => 13,
                    DType::Q6_K => 14,
                    DType::Q8_K => 15,
                    _ => 12,
                };
                if crate::quant::dequantize(&raw, dtype_u32, &mut dst, count) {
                    return Ok(dst);
                } else {
                    eprintln!("[quant] dequant falhou para {:?} dtype {:?} count {}", meta.shape, meta.dtype, count);
                }
            }
        }
        let bytes = self.read(addr, count * 4)?;
        let mut out = Vec::with_capacity(count);
        for chunk in bytes.chunks_exact(4) {
            out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Debug
    // -----------------------------------------------------------------------

    pub fn stats(&self) -> String {
        format!(
            "Memory stats: global_blocks={} next_offset=0x{:x} temporal_head={}/{} version={} persistent_size={}MiB | {} kv_heap={}",
            self.global_heap.len(),
            self.next_global_offset,
            self.temporal_head,
            TEMPORAL_PHYSICAL_SIZE,
            self.version,
            self.persistent_mmap.len() / (1024*1024),
            self.kv_cache_stats(),
            self.kv_heap.len()
        )
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new_in_memory()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_alloc_read_write() {
        let mut mem = MemoryManager::new_in_memory();
        let addr = mem.alloc_global(128).unwrap();
        assert_eq!(region_of(addr), Region::Global);
        mem.write(addr, &[0xAB; 128]).unwrap();
        let out = mem.read(addr, 128).unwrap();
        assert_eq!(out, vec![0xAB; 128]);
    }

    #[test]
    fn test_cow_fork_semantics() {
        let mut mem = MemoryManager::new_in_memory();
        let addr = mem.alloc_global(64).unwrap();
        mem.write(addr, &[1u8; 64]).unwrap();
        let snap = mem.snapshot();
        // Modifica original
        mem.write(addr, &[2u8; 64]).unwrap();
        let out = mem.read(addr, 64).unwrap();
        assert_eq!(out[0], 2);
        // Restore
        mem.restore(snap).unwrap();
        let out2 = mem.read(addr, 64).unwrap();
        assert_eq!(out2[0], 1);
    }

    #[test]
    fn test_temporal_circular() {
        let mut mem = MemoryManager::new_in_memory();
        let data = vec![0x42u8; 1024];
        let addr = mem.temporal_push(&data).unwrap();
        assert_eq!(region_of(addr), Region::Temporal);
        let out = mem.read(addr, 1024).unwrap();
        assert_eq!(out, data);
    }

    #[test]
    fn test_persistent_write_read() {
        let mut mem = MemoryManager::new_in_memory();
        let addr = make_persistent_addr(0x1000);
        mem.write(addr, b"hello persistent").unwrap();
        let out = mem.read(addr, 16).unwrap();
        assert_eq!(&out[..16], b"hello persistent");
    }

    #[test]
    fn test_kv_cache_region_and_ops() {
        let mut mem = MemoryManager::new_in_memory();
        let kv_addr = make_kv_cache_addr(0x1000);
        assert_eq!(region_of(kv_addr), Region::KvCache);
        // Init 22 layers, hidden 8 (minimal)
        mem.kv_cache_init(22, 8);
        assert_eq!(mem.kv_cache_n_layers(), 22);
        assert_eq!(mem.kv_cache_seq_len(), 0);
        let k = vec![1.0f32; 8];
        let v = vec![2.0f32; 8];
        for layer in 0..22 {
            mem.kv_cache_append(layer, &k, &v).unwrap();
        }
        assert_eq!(mem.kv_cache_seq_len(), 1);
        // second token
        let k2 = vec![0.5f32; 8];
        let v2 = vec![1.5f32; 8];
        for layer in 0..22 {
            mem.kv_cache_append(layer, &k2, &v2).unwrap();
        }
        assert_eq!(mem.kv_cache_seq_len(), 2);
        // attention
        let q = vec![0.7f32; 8];
        let out = mem.kv_cache_attention(0, &q).unwrap();
        assert_eq!(out.len(), 8);
        for &x in &out { assert!(x.is_finite()); }
        // truncate rollback
        mem.kv_cache_truncate(1);
        assert_eq!(mem.kv_cache_seq_len(), 1);
        // snapshot / restore
        let snap = mem.snapshot();
        for layer in 0..22 { mem.kv_cache_append(layer, &k2, &v2).unwrap(); }
        assert_eq!(mem.kv_cache_seq_len(), 2);
        mem.restore(snap).unwrap();
        assert_eq!(mem.kv_cache_seq_len(), 1);
        // clear
        mem.kv_cache_clear();
        assert_eq!(mem.kv_cache_seq_len(), 0);
    }

    #[test]
    fn test_kv_heap_alloc_read_write() {
        let mut mem = MemoryManager::new_in_memory();
        let addr = mem.alloc_kv_tensor(128).unwrap();
        assert_eq!(region_of(addr), Region::KvCache);
        mem.write(addr, &[0xCD; 128]).unwrap();
        let out = mem.read(addr, 128).unwrap();
        assert_eq!(out, vec![0xCD; 128]);
        let snap = mem.snapshot();
        mem.write(addr, &[0xEF; 128]).unwrap();
        assert_eq!(mem.read(addr, 128).unwrap()[0], 0xEF);
        mem.restore(snap).unwrap();
        assert_eq!(mem.read(addr, 128).unwrap()[0], 0xCD);
    }
}
