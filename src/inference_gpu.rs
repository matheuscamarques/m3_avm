#![cfg(feature = "wgpu")]
//! inference_gpu.rs — Offload híbrido seletivo p/ Vega 8 (llama.cpp-style).
//!
//! Modelo: placement POR TENSOR no load (não por op): gate/up/down (Q4K/Q6K)
//! + head (Q4K/Q6K) sobem 1× para buffers persistentes; q/k/v/o, norms,
//! atenção, embedding e amostragem ficam no CPU. Pesos nunca fazem re-upload.
//! Ativações (x tiny, y tiny) trafegam por chamada — medido como irrelevante.
//!
//! Falhas NUNCA propagam: `matvec` retorna `None` e o caller cai no CPU.
//! Sem GPU/adapter → `try_new` retorna `None` → 100% CPU, mesmo binário.

use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Quais tensores vão para a GPU (só dtypes com kernel WGSL).
fn gpu_dtype(dtype: u32) -> Option<Kernel> {
    match dtype {
        12 | 13 => Some(Kernel::Q4K),
        14 => Some(Kernel::Q6K),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    Q4K,
    Q6K,
}

struct GpuTensor {
    // Mantido vivo de propósito: propriedade explícita do buffer do peso.
    #[allow(dead_code)]
    buf: wgpu::Buffer,
    bind: wgpu::BindGroup,
    in_dim: usize,
    out_dim: usize,
    q6k: bool,
}

pub struct GpuOffload {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipe_q4k: wgpu::ComputePipeline,
    pipe_q6k: wgpu::ComputePipeline,
    tensors: HashMap<usize, GpuTensor>,
    x_buf: wgpu::Buffer,
    y_buf: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// nº de matvecs executados na GPU (observabilidade)
    pub calls: u64,
    /// tensor idx do LM head (output.weight), se offloaded
    pub head: Option<usize>,
    max_binding: u64,
}

impl GpuOffload {
    /// Cria device + pipelines (1×). `None` = sem GPU (fallback CPU total).
    pub fn try_new(max_in: usize, max_out: usize) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;
        let info = adapter.get_info();
        eprintln!("[gpu] adapter: {} backend={:?}", info.name, info.backend);
        // Binding único do head tem ~191MB > 128MB downlevel: pede o máximo do device.
        let mut limits = adapter.limits();
        let want_binding = 512u64 * 1024 * 1024;
        limits.max_storage_buffer_binding_size =
            limits.max_storage_buffer_binding_size.min(want_binding as u32);
        let max_binding = limits.max_storage_buffer_binding_size as u64;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("m3-infer"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
            },
            None,
        ))
        .ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gemv"),
            source: wgpu::ShaderSource::Wgsl(include_str!("gemv.wgsl").into()),
        });
        let mk = |device: &wgpu::Device, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &shader,
                entry_point: entry,
            })
        };
        let pipe_q4k = mk(&device, "gemv_q4k");
        let pipe_q6k = mk(&device, "gemv_q6k");
        let x_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gx"),
            size: (max_in * 4).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gy"),
            size: (max_out * 4).max(64) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gstg"),
            size: (max_out * 4).max(64) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Some(Self {
            device,
            queue,
            pipe_q4k,
            pipe_q6k,
            tensors: HashMap::new(),
            x_buf,
            y_buf,
            staging,
            calls: 0,
            head: None,
            max_binding,
        })
    }

    /// Sobe 1 tensor (bytes GGUF crus) para buffer persistente + bind group.
    /// `false` = dtype/shape sem kernel (caller mantém no CPU).
    pub fn upload(&mut self, tidx: usize, dtype: u32, raw: &[u8], in_dim: usize, out_dim: usize) -> bool {
        let kernel = match gpu_dtype(dtype) {
            Some(k) => k,
            None => return false,
        };
        // Kernels exigem coluna alinhada (256 | in_dim); resto fica no CPU.
        if in_dim == 0 || out_dim == 0 || in_dim % 256 != 0 {
            return false;
        }
        let expect = crate::matvec_quant::quant_raw_len(dtype, in_dim * out_dim);
        if expect != Some(raw.len()) {
            return false;
        }
        // Binding único precisa caber no limite do device (head ~191MB).
        if raw.len() as u64 > self.max_binding {
            return false;
        }
        let pipe = match kernel {
            Kernel::Q4K => &self.pipe_q4k,
            Kernel::Q6K => &self.pipe_q6k,
        };
        let w_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gw"),
            contents: raw,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params: [u32; 4] = [in_dim as u32, out_dim as u32, 0, 0];
        let p_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gp"),
            contents: bytemuck_u32(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gb"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
        });
        self.tensors.insert(tidx, GpuTensor { buf: w_buf, bind, in_dim, out_dim, q6k: kernel == Kernel::Q6K });
        // Garante que o upload completou antes de seguir (1×, fora do hot path)
        self.device.poll(wgpu::Maintain::Wait);
        true
    }

    pub fn offloaded(&self, tidx: usize) -> bool {
        self.tensors.contains_key(&tidx)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// GEMV sobre tensor offloaded. `None` = fallback CPU.
    pub fn matvec(&mut self, tidx: usize, x: &[f32], in_dim: usize, out_dim: usize) -> Option<Vec<f32>> {
        let t = self.tensors.get(&tidx)?;
        if t.in_dim != in_dim || t.out_dim != out_dim || x.len() != in_dim {
            eprintln!("[gpu] matvec dims?? tidx={} stored={}x{} call={}x{} xlen={}", tidx, t.in_dim, t.out_dim, in_dim, out_dim, x.len());
            return None;
        }
        if (in_dim * 4) as u64 > self.x_buf.size() || (out_dim * 4) as u64 > self.y_buf.size() {
            eprintln!("[gpu] matvec scratch pequeno");
            return None;
        }
        // x da chamada (tiny, ~6-36KB); pesos já estão na GPU
        self.queue.write_buffer(&self.x_buf, 0, bytemuck_f32(x));
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
            // Pipeline correspondente ao dtype do tensor (layouts idênticos).
            if t.q6k {
                pass.set_pipeline(&self.pipe_q6k);
            } else {
                pass.set_pipeline(&self.pipe_q4k);
            }
            pass.set_bind_group(0, &t.bind, &[]);
            pass.dispatch_workgroups(out_dim.div_ceil(64) as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&self.y_buf, 0, &self.staging, 0, (out_dim * 4) as u64);
        self.queue.submit(Some(enc.finish()));
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = tx.send(v);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let got = rx.recv().ok()?;
        got.ok()?;
        let data = slice.get_mapped_range();
        // Staging é dimensionado p/ max_out; lê só os bytes desta chamada.
        let want = out_dim * 4;
        if data.len() < want {
            eprintln!("[gpu] matvec staging curto");
            return None;
        }
        let out: Vec<f32> = data[..want].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        self.staging.unmap();
        if out.len() != out_dim {
            return None;
        }
        self.calls += 1;
        Some(out)
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

fn bytemuck_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}
