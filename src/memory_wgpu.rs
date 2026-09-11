#![cfg(feature = "wgpu")]
//! memory_wgpu.rs — Driver WGPU para Vega 8 (Raven) — V1 GPU
//!
//! Substitui `memory.rs` `Vec<u8>`/`mmap` por `wgpu::Buffer` quando feature `wgpu` ativa.
//! Mantém mesma trait `alloc/read/write` para `vm.rs` não quebrar.
//! Vega 8: RADV RAVEN, 8 CUs @1200MHz, 1.1 TFLOPS, via Vulkan (`wgpu::Backends::VULKAN`).

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, Device, Queue};

use crate::memory::{DType, TensorMeta, REGION_GLOBAL, REGION_PERSISTENT, REGION_TEMPORAL};
use crate::sparse::SparseTensor;

// ---------------------------------------------------------------------------
// WgpuMemoryManager — espelho de MemoryManager mas com GPU buffers
// ---------------------------------------------------------------------------

pub struct WgpuMemoryManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    // CPU fallback para controle (metadados), GPU para dados
    cpu_fallback: crate::memory::MemoryManager,
    // GPU buffers por endereço GLOBAL 0x00...
    gpu_buffers: HashMap<u128, Buffer>,
    sparse_heap: HashMap<u128, SparseTensor>,
    next_offset: u128,
    // P2.1: pipelines do ATTN criados 1× (criar por chamada custava ~100ms+).
    // attn_shader mantido vivo de propósito (propriedade do módulo).
    #[allow(dead_code)]
    attn_shader: Option<wgpu::ShaderModule>,
    pipe_qkt: Option<wgpu::ComputePipeline>,
    pipe_softmax: Option<wgpu::ComputePipeline>,
    pipe_smv: Option<wgpu::ComputePipeline>,
}

impl WgpuMemoryManager {
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("nenhum adapter wgpu (Vega 8 não encontrada)"))?;

        let info = adapter.get_info();
        eprintln!("[wgpu] adapter: {} ({:?}, backend {:?})", info.name, info.device_type, info.backend);
        if !info.name.contains("Radeon") && !info.name.contains("AMD") {
            eprintln!("[wgpu] aviso: adapter não é Vega 8, mas vai tentar: {:?}", info);
        }

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("m3-wgpu-vega8"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| anyhow!("wgpu device: {:?}", e))?;

        let cpu_fallback = crate::memory::MemoryManager::new_in_memory();

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            cpu_fallback,
            gpu_buffers: HashMap::new(),
            sparse_heap: HashMap::new(),
            next_offset: 0x1000,
            attn_shader: None,
            pipe_qkt: None,
            pipe_softmax: None,
            pipe_smv: None,
        })
    }

    /// Garante shader + 3 pipelines do ATTN (cria 1×, reusa sempre).
    fn ensure_attn_pipelines(&mut self) -> Result<()> {
        if self.pipe_qkt.is_some() {
            return Ok(());
        }
        let device = self.device.clone();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("m3-attn"),
            source: wgpu::ShaderSource::Wgsl(include_str!("attn.wgsl").into()),
        });
        let mk = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: entry,
            })
        };
        self.pipe_qkt = Some(mk("qkt"));
        self.pipe_softmax = Some(mk("softmax"));
        self.pipe_smv = Some(mk("sm_v"));
        self.attn_shader = Some(shader);
        Ok(())
    }

    pub fn new_blocking() -> Result<Self> {
        pollster::block_on(Self::new())
    }

    // ---- API compatível com MemoryManager ----

    pub fn alloc_global(&mut self, size: usize) -> Result<u128> {
        if size == 0 { return Err(anyhow!("size zero")); }
        let aligned = (size + 63) & !63;
        let addr = crate::memory::make_global_addr(self.next_offset);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("m3_global_{:x}", addr)),
            size: aligned as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu_buffers.insert(addr, buffer);
        self.next_offset += aligned as u128;
        // Mantém meta no fallback para is_sparse etc.
        self.cpu_fallback.alloc_global(aligned).ok();
        Ok(addr)
    }

    pub fn global_heap_cursor(&self) -> u128 {
        self.next_offset
    }

    pub fn alloc_tensor(&mut self, shape: &[usize], dtype: DType) -> Result<u128> {
        let elems: usize = shape.iter().product();
        let byte_len = elems * dtype.byte_width();
        let addr = self.alloc_global(byte_len)?;
        let meta = TensorMeta { addr, shape: shape.to_vec(), dtype, byte_len, is_sparse: false, density: 1.0 };
        self.cpu_fallback.tensor_meta.insert(addr, meta);
        Ok(addr)
    }

    pub fn alloc_sparse_tensor(&mut self, shape: &[usize], dtype: DType, density: f32) -> Result<u128> {
        // Para Vega 8, sparse ainda fica em CPU (CSR) — GPU só para denso por enquanto
        // Futuro: BSR em wgpu storage buffer
        let mut rng = rand::thread_rng();
        let st = SparseTensor::random((shape[0], shape[1]), density.clamp(0.01, 1.0), &mut rng);
        let addr = self.alloc_global(64)?; // placeholder GPU buffer para endereço
        self.sparse_heap.insert(addr, st);
        let meta = TensorMeta { addr, shape: shape.to_vec(), dtype, byte_len: 64, is_sparse: true, density };
        self.cpu_fallback.tensor_meta.insert(addr, meta);
        Ok(addr)
    }

    pub fn is_sparse(&self, addr: u128) -> bool {
        self.sparse_heap.contains_key(&addr) || self.cpu_fallback.is_sparse(addr)
    }

    pub fn get_sparse(&self, addr: u128) -> Option<&SparseTensor> {
        self.sparse_heap.get(&addr)
    }

    pub fn get_sparse_mut(&mut self, addr: u128) -> Option<&mut SparseTensor> {
        self.sparse_heap.get_mut(&addr)
    }

    pub fn temporal_push(&mut self, data: &[u8]) -> Result<u128> {
        self.cpu_fallback.temporal_push(data)
    }

    pub fn temporal_read(&self, addr: u128, len: usize) -> Result<Vec<u8>> {
        self.cpu_fallback.temporal_read(addr, len)
    }

    pub fn snapshot(&mut self) -> u64 {
        self.cpu_fallback.snapshot()
    }

    pub fn restore(&mut self, version: u64) -> Result<()> {
        self.cpu_fallback.restore(version)
    }

    pub fn remove_tensor(&mut self, addr: u128) -> bool {
        self.cpu_fallback.remove_tensor(addr)
    }

    pub fn current_version(&self) -> u64 {
        self.cpu_fallback.current_version()
    }

    pub fn persistent_flush(&self) -> Result<()> {
        self.cpu_fallback.persistent_flush()
    }

    /// GGUF via fallback CPU (mmap lazy; GPU usa buffers próprios via upload).
    /// Faltava no dispatch de `MemBackend` e quebrava `--features wgpu`.
    pub fn load_gguf_model(&mut self, path: &str) -> Result<u128> {
        self.cpu_fallback.load_gguf_model(path)
    }

    pub fn load_gguf_bytes(&mut self, bytes: &[u8]) -> Result<u128> {
        self.cpu_fallback.load_gguf_bytes(bytes)
    }

    pub fn write(&mut self, addr: u128, data: &[u8]) -> Result<()> {
        if let Some(buf) = self.gpu_buffers.get(&addr) {
            self.queue.write_buffer(buf, 0, data);
            // Write-through para fallback para read_f32 compat
            let _ = self.cpu_fallback.write(addr, data);
            Ok(())
        } else {
            self.cpu_fallback.write(addr, data)
        }
    }

    pub fn read(&self, addr: u128, len: usize) -> Result<Vec<u8>> {
        // Para GPU, precisaria map_async + polling — por enquanto lê do fallback
        // Vega 8: readback é lento (PCIe), então mantemos shadow em CPU
        self.cpu_fallback.read(addr, len)
    }

    pub fn write_f32_tensor(&mut self, addr: u128, data: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write(addr, &bytes)
    }

    pub fn read_f32_tensor(&self, addr: u128, count: usize) -> Result<Vec<f32>> {
        self.cpu_fallback.read_f32_tensor(addr, count)
    }

    pub fn get_tensor_meta(&self, addr: u128) -> Option<&TensorMeta> {
        self.cpu_fallback.get_tensor_meta(addr)
    }

    /// ATTN na Vega 8 via wgpu compute shader — Q*K^T + softmax + *V em 3 dispatches
    /// Usa `src/attn.wgsl` com workgroups 8x8, suporta até 64x64 (Vega 8: 8 CUs)
    pub fn attn_wgpu(&mut self, q_addr: u128, k_addr: u128, v_addr: u128, q_shape: (usize, usize), k_shape: (usize, usize), v_shape: (usize, usize)) -> Result<(u128, Vec<f32>)> {
        let (m, d) = q_shape;
        let (n, d2) = k_shape;
        let (_, p) = v_shape;
        if d != d2 { return Err(anyhow!("Q cols {} != K cols {}", d, d2)); }
        if n != v_shape.0 { return Err(anyhow!("K rows {} != V rows {}", n, v_shape.0)); }

        // Lê Q,K,V do fallback CPU (shadow) — Vega 8 readback é via CPU shadow por enquanto
        let q_data = self.read_f32_tensor(q_addr, m*d)?;
        let k_data = self.read_f32_tensor(k_addr, n*d)?;
        let v_data = self.read_f32_tensor(v_addr, n*p)?;

        // Pipelines em cache (P2.1); refs compartilhadas (ComputePipeline não é Clone)
        self.ensure_attn_pipelines()?;
        let qkt_pipeline = self.pipe_qkt.as_ref().unwrap();
        let sm_pipeline = self.pipe_softmax.as_ref().unwrap();
        let smv_pipeline = self.pipe_smv.as_ref().unwrap();

        // Helper para criar STORAGE buffer inicializado
        let create_storage = |label: &str, data: &[f32]| {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.len() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue.write_buffer(&buf, 0, &bytes);
            buf
        };

        let q_buf = create_storage("q", &q_data);
        let k_buf = create_storage("k", &k_data);
        let v_buf = create_storage("v", &v_data);
        let scores_bytes = (m*n*4) as u64;
        let scores_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: scores_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_bytes = (m*p*4) as u64;
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out"),
            size: out_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Staging para readback (MAP_READ só com COPY_DST)
        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Params uniform
        let scale = 1.0 / (d as f32).sqrt();
        let params = [m as u32, n as u32, d as u32, u32::from_le_bytes(scale.to_le_bytes())];
        // Na verdade scale é f32, mas WGSL Params.scale é f32 — precisamos pack como bytes
        // Vamos criar buffer com 16 bytes: m,n,d,scale (u32,u32,u32,f32)
        let mut param_bytes = Vec::new();
        param_bytes.extend_from_slice(&(m as u32).to_le_bytes());
        param_bytes.extend_from_slice(&(n as u32).to_le_bytes());
        param_bytes.extend_from_slice(&(d as u32).to_le_bytes());
        param_bytes.extend_from_slice(&scale.to_le_bytes());
        let param_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&param_buf, 0, &param_bytes);

        let param2_bytes = {
            let mut b = Vec::new();
            b.extend_from_slice(&(m as u32).to_le_bytes());
            b.extend_from_slice(&(n as u32).to_le_bytes());
            b.extend_from_slice(&(p as u32).to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes());
            b
        };
        let param2_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params2"),
            size: 16,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&param2_buf, 0, &param2_bytes);

        // Bind groups (baratos; buffers variam por chamada)
        let qkt_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qkt_bind"),
            layout: &qkt_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: scores_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: param_buf.as_entire_binding() },
            ],
        });

        let sm_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sm_bind"),
            layout: &sm_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scores_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: param_buf.as_entire_binding() },
            ],
        });

        let smv_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smv_bind"),
            layout: &smv_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: scores_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: param2_buf.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("m3-attn-encoder") });
        // Dispatch QKT: workgroups (ceil(m/8), ceil(n/8))
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("qkt_pass"), timestamp_writes: None });
            pass.set_pipeline(&qkt_pipeline);
            pass.set_bind_group(0, &qkt_bind, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (n as u32).div_ceil(8), 1);
        }
        // Dispatch softmax: 1 workgroup por linha (m)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sm_pass"), timestamp_writes: None });
            pass.set_pipeline(&sm_pipeline);
            pass.set_bind_group(0, &sm_bind, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(64), 1, 1);
        }
        // Dispatch SM*V
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("smv_pass"), timestamp_writes: None });
            pass.set_pipeline(&smv_pipeline);
            pass.set_bind_group(0, &smv_bind, &[]);
            pass.dispatch_workgroups((m as u32).div_ceil(8), (p as u32).div_ceil(8), 1);
        }
        // Copy out -> staging for MAP_READ
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        // Map staging and read
        let buffer_slice = staging_buf.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| { let _ = sender.send(v); });
        self.device.poll(wgpu::Maintain::Wait);
        let map_result = pollster::block_on(receiver).map_err(|e| anyhow!("oneshot recv: {:?}", e))?;
        map_result.map_err(|e| anyhow!("map failed: {:?}", e))?;
        let data = buffer_slice.get_mapped_range();
        let out_f32: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        drop(data);
        staging_buf.unmap();

        // Aloca tensor de saída em GPU + CPU shadow
        let out_shape = vec![m, p];
        let out_addr = self.alloc_tensor(&out_shape, DType::F32)?;
        self.write_f32_tensor(out_addr, &out_f32)?;
        Ok((out_addr, out_f32))
    }

    pub fn kv_cache_init(&mut self, n_layers: usize, hidden: usize) { self.cpu_fallback.kv_cache_init(n_layers, hidden) }
    pub fn kv_cache_append(&mut self, layer: usize, k: &[f32], v: &[f32]) -> anyhow::Result<()> { self.cpu_fallback.kv_cache_append(layer, k, v) }
    pub fn kv_cache_get_k(&self, layer: usize) -> Option<&[f32]> { self.cpu_fallback.kv_cache_get_k(layer) }
    pub fn kv_cache_get_v(&self, layer: usize) -> Option<&[f32]> { self.cpu_fallback.kv_cache_get_v(layer) }
    pub fn kv_cache_seq_len(&self) -> usize { self.cpu_fallback.kv_cache_seq_len() }
    pub fn kv_cache_n_layers(&self) -> usize { self.cpu_fallback.kv_cache_n_layers() }
    pub fn kv_cache_truncate(&mut self, seq: usize) { self.cpu_fallback.kv_cache_truncate(seq) }
    pub fn kv_cache_truncate_stream(&mut self, sid: u16, seq: usize) { self.cpu_fallback.kv_cache_truncate_stream(sid, seq) }
    pub fn kv_cache_compress_sink_window_stream(&mut self, sid: u16, sink: usize, window: usize) { self.cpu_fallback.kv_cache_compress_sink_window_stream(sid, sink, window) }
    pub fn kv_cache_append_stream(&mut self, sid: u16, layer: usize, k: &[f32], v: &[f32]) -> anyhow::Result<()> { self.cpu_fallback.kv_cache_append_stream(sid, layer, k, v) }
    pub fn kv_cache_seq_len_stream(&self, sid: u16) -> usize { self.cpu_fallback.kv_cache_seq_len_stream(sid) }
    pub fn kv_cache_compress_sink_window(&mut self, sink: usize, window: usize) { self.cpu_fallback.kv_cache_compress_sink_window(sink, window) }
    pub fn kv_cache_clear(&mut self) { self.cpu_fallback.kv_cache_clear() }
    pub fn kv_cache_attention(&self, layer: usize, q: &[f32]) -> anyhow::Result<Vec<f32>> { self.cpu_fallback.kv_cache_attention(layer, q) }
    pub fn kv_cache_stats(&self) -> String { self.cpu_fallback.kv_cache_stats() }
    pub fn stats(&self) -> String {
        format!("WGPU Vega8: gpu_buffers={} cpu_fallback={} next_offset=0x{:x}", self.gpu_buffers.len(), self.cpu_fallback.stats(), self.next_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::DType;

    #[test]
    fn test_wgpu_vega8_available() {
        // Só executa se Vega 8 estiver presente (seu 3500U)
        let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let mgr = rt.block_on(WgpuMemoryManager::new());
        match mgr {
            Ok(m) => {
                println!("✅ Vega 8 wgpu OK: {}", m.stats());
                assert!(m.stats().contains("gpu_buffers"));
            }
            Err(e) => {
                eprintln!("⚠️ wgpu não disponível (ok em CI sem GPU): {}", e);
                // Não falha em CI sem GPU
            }
        }
    }

    #[test]
    fn test_wgpu_alloc_tensor() {
        let mut mgr = match WgpuMemoryManager::new_blocking() {
            Ok(m) => m,
            Err(e) => { eprintln!("skip wgpu test (no GPU): {}", e); return; }
        };
        let addr = mgr.alloc_tensor(&[4,4], DType::F32).unwrap();
        assert_eq!(crate::memory::region_of(addr), crate::memory::Region::Global);
        mgr.write_f32_tensor(addr, &[1.0;16]).unwrap();
        let out = mgr.read_f32_tensor(addr, 16).unwrap();
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn test_wgpu_sparse_fallback() {
        let mut mgr = match WgpuMemoryManager::new_blocking() {
            Ok(m) => m,
            Err(e) => { eprintln!("skip: {}", e); return; }
        };
        let addr = mgr.alloc_sparse_tensor(&[8,8], DType::F32, 0.1).unwrap();
        assert!(mgr.is_sparse(addr));
        let st = mgr.get_sparse(addr).unwrap();
        assert_eq!(st.shape, (8,8));
    }
}
