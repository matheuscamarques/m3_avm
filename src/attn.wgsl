// attn.wgsl — ATTN Q*K^T / sqrt(d) para Vega 8 (RADV RAVEN)
// Workgroup 8x8, suporta até 64x64 com dispatch (m/8, n/8)
// Bindings: 0=Q (m*d), 1=K (n*d), 2=scores (m*n), 3=params (m,n,d,scale)

struct Params {
    m: u32,
    n: u32,
    d: u32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn qkt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = params.m;
    let n = params.n;
    let d = params.d;
    let scale = params.scale;

    let i = gid.x;
    let j = gid.y;
    if (i >= m || j >= n) { return; }

    var sum: f32 = 0.0;
    for (var kk: u32 = 0u; kk < d; kk = kk + 1u) {
        sum = sum + q[i * d + kk] * k[j * d + kk];
    }
    scores[i * n + j] = sum * scale;
}

// Softmax por linha — um workgroup por linha
// Bindings: 0=scores (m*n) read_write, 1=params
@group(0) @binding(0) var<storage, read_write> sm_scores: array<f32>;
@group(0) @binding(1) var<uniform> sm_params: Params;

@compute @workgroup_size(64, 1, 1)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = sm_params.m;
    let n = sm_params.n;
    let row = gid.x;
    if (row >= m) { return; }

    // max
    var maxv: f32 = -3.402823e+38;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let v = sm_scores[row * n + j];
        if (v > maxv) { maxv = v; }
    }
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        let e = exp(sm_scores[row * n + j] - maxv);
        sm_scores[row * n + j] = e;
        sum = sum + e;
    }
    for (var j: u32 = 0u; j < n; j = j + 1u) {
        sm_scores[row * n + j] = sm_scores[row * n + j] / sum;
    }
}

// scores (m*n) * V (n*p) -> out (m*p)
struct Params2 {
    m: u32,
    n: u32,
    p: u32,
};
@group(0) @binding(0) var<storage, read> scores2: array<f32>;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params2: Params2;

@compute @workgroup_size(8, 8, 1)
fn sm_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = params2.m;
    let n = params2.n;
    let p = params2.p;
    let i = gid.x;
    let j = gid.y;
    if (i >= m || j >= p) { return; }
    var sum: f32 = 0.0;
    for (var kk: u32 = 0u; kk < n; kk = kk + 1u) {
        sum = sum + scores2[i * n + kk] * v[kk * p + j];
    }
    out[i * p + j] = sum;
}
