//! bench_gpu_gemv — CPU (faer/AVX2) vs GPU (Vega 8, WGSL) no GEMV real.
//!
//! Uso: `cargo run --release --features wgpu --bin bench_gpu_gemv`
//! Pesos sobem 1× (buffers persistentes = teto do full-GPU); mede mediana de 5
//! com A/B alternado. Também mede 1 upload (piso: re-upload por op).

#[cfg(feature = "wgpu")]
fn main() -> anyhow::Result<()> {
    use wgpu::util::DeviceExt;
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    rt.block_on(async_main())
}

#[cfg(feature = "wgpu")]
fn median(v: &[u128]) -> u128 {
    let mut s = v.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

#[cfg(feature = "wgpu")]
async fn async_main() -> anyhow::Result<()> {
    use wgpu::util::DeviceExt;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .ok_or_else(|| anyhow::anyhow!("sem adapter"))?;
    let info = adapter.get_info();
    println!("adapter: {} backend={:?} type={:?}", info.name, info.backend, info.device_type);
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gemv-bench"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        )
        .await
        .map_err(|e| anyhow::anyhow!("device: {:?}", e))?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("gemv"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../gemv.wgsl").into()),
    });
    let mk_pipeline = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: None,
            module: &shader,
            entry_point: entry,
        })
    };
    let pipe_f32 = mk_pipeline("gemv_f32");
    let pipe_q4k = mk_pipeline("gemv_q4k");

    for (tag, in_dim, out_dim) in [("qproj", 1536usize, 1536usize), ("gate", 1536usize, 8960usize)] {
        println!("=== {} {}x{} ===", tag, in_dim, out_dim);
        let n = in_dim * out_dim;
        // pesos f32 + x
        let w: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.002).sin() * 0.3)).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32 * 0.013).sin() * 0.5)).collect();
        // ---- GPU f32, upload 1x ----
        let t_up = std::time::Instant::now();
        let w_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("w"),
            contents: bytemuck_cast(&w),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let x_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("x"),
            contents: bytemuck_cast(&x),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let y_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y"),
            size: (out_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stg"),
            size: (out_dim * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params: [u32; 4] = [in_dim as u32, out_dim as u32, 0, 0];
        let p_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("p"),
            contents: bytemuck_cast_u32(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("b"),
            layout: &pipe_f32.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
        });
        device.poll(wgpu::Maintain::Wait);
        println!("upload {:.1}MB: {:?}", (w.len() * 4) as f64 / 1e6, t_up.elapsed());

        let run_gpu = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| -> Vec<f32> {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(out_dim.div_ceil(64) as u32, 1, 1);
            }
            enc.copy_buffer_to_buffer(&y_buf, 0, &staging, 0, (out_dim * 4) as u64);
            queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            staging.unmap();
            out
        };

        // warmup + validação vs CPU
        let y_cpu = m3_avm::matvec::matvec(&x, &w, in_dim, out_dim);
        let y_gpu = run_gpu(&pipe_f32, &bind);
        let md = max_rel_diff(&y_cpu, &y_gpu);
        println!("f32 parity max_rel_diff={:.2e} (cpu[0]={:.4} gpu[0]={:.4})", md, y_cpu[0], y_gpu[0]);

        let mut t_cpu = vec![];
        let mut t_gpu = vec![];
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            let _ = m3_avm::matvec::matvec(&x, &w, in_dim, out_dim);
            t_cpu.push(t0.elapsed().as_micros());
            let t1 = std::time::Instant::now();
            let _ = run_gpu(&pipe_f32, &bind);
            t_gpu.push(t1.elapsed().as_micros());
        }
        println!("cpu_f32 med {}us | gpu_f32 med {}us | speedup {:.2}x", median(&t_cpu), median(&t_gpu), median(&t_cpu) as f64 / median(&t_gpu).max(1) as f64);

        // ---- Q4K fundido ----
        let raw = synth_q4k(n);
        let y_cpu_q = m3_avm::matvec_quant::matvec_q4k(&x, &raw, in_dim, out_dim);
        // reusa mesmos buffers de controle; peso como bytes
        let wq_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wq"),
            contents: &raw,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let yq_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yq"),
            size: (out_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_q = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bq"),
            layout: &pipe_q4k.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wq_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: yq_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: p_buf.as_entire_binding() },
            ],
        });
        let run_gpu_q = || -> Vec<f32> {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_pipeline(&pipe_q4k);
                pass.set_bind_group(0, &bind_q, &[]);
                pass.dispatch_workgroups(out_dim.div_ceil(64) as u32, 1, 1);
            }
            enc.copy_buffer_to_buffer(&yq_buf, 0, &staging, 0, (out_dim * 4) as u64);
            queue.submit(Some(enc.finish()));
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |v| { let _ = tx.send(v); });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let out: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            drop(data);
            staging.unmap();
            out
        };
        let y_gpu_q = run_gpu_q();
        let mdq = max_rel_diff(&y_cpu_q, &y_gpu_q);
        println!("q4k parity max_rel_diff={:.2e} (cpu[0]={:.4} gpu[0]={:.4})", mdq, y_cpu_q[0], y_gpu_q[0]);
        let mut tc = vec![];
        let mut tg = vec![];
        for _ in 0..5 {
            let t0 = std::time::Instant::now();
            let _ = m3_avm::matvec_quant::matvec_q4k(&x, &raw, in_dim, out_dim);
            tc.push(t0.elapsed().as_micros());
            let t1 = std::time::Instant::now();
            let _ = run_gpu_q();
            tg.push(t1.elapsed().as_micros());
        }
        println!("cpu_q4k med {}us | gpu_q4k med {}us | speedup {:.2}x", median(&tc), median(&tg), median(&tc) as f64 / median(&tg).max(1) as f64);
    }
    Ok(())
}

/// Q4K sintético com semântica exata (d=1, dmin=0, sc variado, qs variado).
#[cfg(feature = "wgpu")]
fn synth_q4k(n: usize) -> Vec<u8> {
    assert!(n % 256 == 0);
    let mut raw = vec![0u8; (n / 256) * 144];
    for b in 0..n / 256 {
        let off = b * 144;
        raw[off] = 0x00;
        raw[off + 1] = 0x3c;
        raw[off + 2] = 0x00;
        raw[off + 3] = 0x00;
        for i in 0..12 {
            raw[off + 4 + i] = 0x10u8.wrapping_add((b as u8).wrapping_add(i as u8));
        }
        for i in 0..128 {
            raw[off + 16 + i] = (b as u8).wrapping_mul(37).wrapping_add((i as u8).wrapping_mul(11));
        }
    }
    raw
}

#[cfg(feature = "wgpu")]
fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs() / x.abs().max(1e-6)).fold(0.0f32, f32::max)
}

#[cfg(feature = "wgpu")]
fn bytemuck_cast(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[cfg(feature = "wgpu")]
fn bytemuck_cast_u32(v: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }
}

#[cfg(not(feature = "wgpu"))]
fn main() {
    eprintln!("rebuild com --features wgpu");
}
