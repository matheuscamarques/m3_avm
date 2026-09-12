//! rag.rs — IndexStore p/ retrieval V-2 (RFC-0038, turno 1: ADD/DEL).
//!
//! Constantes travadas (V-2_SCOPE §9): dim de embedding 1024 (BGE-M3),
//! dtype f32, métrica default cosine. Índice = flat `Vec<f32>` (dim×n) +
//! ids; HNSW/IVF é follow-up explícito, não desta RFC.
//!
//! Concorrência/rollback (opção B, CoW): o dono é `Vm` via
//! `Arc<RwLock<IndexStore>>`; snapshots guardam `Arc::clone` O(1) e a
//! primeira escrita com `strong_count>1` duplica o interior antes de
//! mutar (ver `Vm::rag_store_mut`). `f32` só; `NaN/Inf` recusados na
//! entrada (precedente FOREST/SANITY_CHECK).

use std::collections::HashMap;

use anyhow::{anyhow, Result};

/// Dimensão de embedding travada (BGE-M3).
pub const RAG_EMBED_DIM: usize = 1024;

/// Dtype de embedding (unitário hoje; enum p/ falhar alto no futuro,
/// nunca silencioso).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingDtype {
    F32,
}

/// Métrica de distância (default Cosine; SEARCH no turno 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagMetric {
    Cosine,
    Euclid,
    Dot,
}

impl RagMetric {
    pub fn default() -> Self {
        RagMetric::Cosine
    }

    /// Mapeia o byte de payload (`RAG_METRIC_*`); outro => erro alto.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(RagMetric::Cosine),
            1 => Ok(RagMetric::Euclid),
            2 => Ok(RagMetric::Dot),
            _ => Err(anyhow!("RAG: METRIC {} inválida (0=COSINE,1=EUCLID,2=DOT)", v)),
        }
    }
}

/// Um índice: dimensão fixada na criação + vetores flat + ids.
#[derive(Debug, Clone, PartialEq)]
pub struct RagIndex {
    pub dim: usize,
    pub dtype: EmbeddingDtype,
    pub metric: RagMetric,
    pub vec_ids: Vec<u64>,
    pub vectors: Vec<f32>,
    next_vec: u64,
}

impl RagIndex {
    fn new(dim: usize) -> Self {
        Self { dim, dtype: EmbeddingDtype::F32, metric: RagMetric::default(), vec_ids: Vec::new(), vectors: Vec::new(), next_vec: 0 }
    }

    fn check_vec(&self, v: &[f32]) -> Result<()> {
        if v.len() != self.dim {
            return Err(anyhow!("RAG: vetor com {} elems, índice espera dim {}", v.len(), self.dim));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(anyhow!("RAG: vetor não-finito recusado (NaN/Inf)"));
        }
        Ok(())
    }
}

/// Store de índices: `store_id u64 -> Index`. Id 0 nunca é válido
/// (sentinela "criar" no `rDb`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndexStore {
    next_store: u64,
    stores: HashMap<u64, RagIndex>,
}

impl IndexStore {
    pub fn new() -> Self {
        Self { next_store: 1, stores: HashMap::new() }
    }

    /// Cria índice vazio com `dim`; retorna o id (>0).
    pub fn create(&mut self, dim: usize) -> Result<u64> {
        if dim == 0 {
            return Err(anyhow!("RAG: dim zero"));
        }
        let id = self.next_store;
        self.next_store += 1;
        self.stores.insert(id, RagIndex::new(dim));
        Ok(id)
    }

    fn get_mut(&mut self, store: u64) -> Result<&mut RagIndex> {
        self.stores.get_mut(&store).ok_or_else(|| anyhow!("RAG: índice {} inexistente", store))
    }

    pub fn get(&self, store: u64) -> Result<&RagIndex> {
        self.stores.get(&store).ok_or_else(|| anyhow!("RAG: índice {} inexistente", store))
    }

    /// Anexa vetor; `explicit` = id exigido (colisão => erro) ou
    /// sequencial. Retorna o id do vetor.
    pub fn add(&mut self, store: u64, vec: &[f32], explicit: Option<u64>) -> Result<u64> {
        let idx = self.get_mut(store)?;
        idx.check_vec(vec)?;
        let id = match explicit {
            Some(e) => {
                if idx.vec_ids.contains(&e) {
                    return Err(anyhow!("RAG: id {} já existe no índice {}", e, store));
                }
                e
            }
            None => {
                let id = idx.next_vec;
                idx.next_vec += 1;
                id
            }
        };
        idx.vec_ids.push(id);
        idx.vectors.extend_from_slice(vec);
        Ok(id)
    }

    /// Remove vetor; retorna contagem restante. Ausente => erro alto.
    pub fn del(&mut self, store: u64, id: u64) -> Result<usize> {
        let idx = self.get_mut(store)?;
        let pos = idx.vec_ids.iter().position(|&x| x == id)
            .ok_or_else(|| anyhow!("RAG: id {} ausente no índice {}", id, store))?;
        idx.vec_ids.remove(pos);
        let start = pos * idx.dim;
        idx.vectors.drain(start..start + idx.dim);
        Ok(idx.vec_ids.len())
    }

    pub fn len(&self, store: u64) -> Result<usize> {
        Ok(self.get(store)?.vec_ids.len())
    }

    pub fn dim(&self, store: u64) -> Result<usize> {
        Ok(self.get(store)?.dim)
    }
}

/// Distância query×linha — mesmas fórmulas do `DISTANCE` (EUCLID
/// sqrt-SSE; COSINE 1-cos com zero-norma => 1.0; DOT negado) e mesma
/// regra de não-finito => `f32::MAX` (entradas finitas por construção:
/// ADD recusa `NaN/Inf`, query checada no exec).
pub fn metric_dist(metric: RagMetric, q: &[f32], row: &[f32], qnorm2: f32) -> f32 {
    let dist = match metric {
        RagMetric::Euclid => row.iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt(),
        RagMetric::Dot => -row.iter().zip(q.iter()).map(|(a, b)| a * b).sum::<f32>(),
        RagMetric::Cosine => {
            let rn: f32 = row.iter().map(|x| x * x).sum();
            if qnorm2 == 0.0 || rn == 0.0 {
                1.0
            } else {
                let dot: f32 = row.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
                1.0 - dot / (qnorm2.sqrt() * rn.sqrt())
            }
        }
    };
    if dist.is_finite() { dist } else { f32::MAX }
}

/// Top-k estável: ordena por (dist, id) — empate => menor id primeiro
/// (precedente DISTANCE, que ordena por índice).
pub fn topk_by_id(scored: &[(u64, f32)], k: usize) -> Vec<(u64, f32)> {
    let mut order: Vec<usize> = (0..scored.len()).collect();
    order.sort_by(|&a, &b| {
        scored[a].1.partial_cmp(&scored[b].1).unwrap_or(std::cmp::Ordering::Equal)
            .then(scored[a].0.cmp(&scored[b].0))
    });
    order.truncate(k.min(scored.len()));
    order.iter().map(|&i| scored[i]).collect()
}

// ---------------------------------------------------------------------------
// PQ — product quantizer (turno 3, RFC-0038).
// ---------------------------------------------------------------------------

/// Valida geometria PQ e extrai (subdim, K). Erros altos nunca silenciosos.
pub fn pq_validate(d: usize, nsub: usize, cb_rows: usize, cb_cols: usize) -> Result<(usize, usize)> {
    if nsub == 0 {
        return Err(anyhow!("PQ: NSUB=0"));
    }
    if d == 0 || cb_rows == 0 || cb_cols == 0 {
        return Err(anyhow!("PQ: dimensão zero (D={}, rows={}, cols={})", d, cb_rows, cb_cols));
    }
    if d % nsub != 0 {
        return Err(anyhow!("PQ: D={} não divisível por NSUB={}", d, nsub));
    }
    let subdim = d / nsub;
    if cb_rows % nsub != 0 {
        return Err(anyhow!("PQ: codebook rows={} não divisível por NSUB={}", cb_rows, nsub));
    }
    let k = cb_rows / nsub;
    if k == 0 || k > 65536 {
        return Err(anyhow!("PQ: K={} fora de [1,65536]", k));
    }
    if cb_cols != subdim {
        return Err(anyhow!("PQ: codebook cols={} != subdim {} (D/NSUB)", cb_cols, subdim));
    }
    Ok((subdim, k))
}

/// PQ_ENCODE: para cada subvetor, encontra centróide mais próximo (euclidiana
/// quadrada; empate => menor índice). `vec` len D, `codebook` len rows*cols
/// row-major [rows, cols].
pub fn pq_encode(vec: &[f32], codebook: &[f32], nsub: usize) -> Result<Vec<u16>> {
    if vec.iter().any(|x| !x.is_finite()) {
        return Err(anyhow!("PQ_ENCODE: vetor não-finito recusado (NaN/Inf)"));
    }
    if codebook.iter().any(|x| !x.is_finite()) {
        return Err(anyhow!("PQ_ENCODE: codebook não-finito recusado (NaN/Inf)"));
    }
    // Geometria será validada pelo caller com cb_rows/cb_cols; aqui inferimos
    // subdim via D/nsub.
    if nsub == 0 {
        return Err(anyhow!("PQ_ENCODE: NSUB=0"));
    }
    if vec.len() % nsub != 0 {
        return Err(anyhow!("PQ_ENCODE: D={} não divisível por NSUB={}", vec.len(), nsub));
    }
    let subdim = vec.len() / nsub;
    let rows = codebook.len() / subdim;
    if codebook.len() % subdim != 0 || rows % nsub != 0 {
        return Err(anyhow!("PQ_ENCODE: geometria de codebook inconsistente (rows*cols={})", codebook.len()));
    }
    let k = rows / nsub;
    let mut codes = Vec::with_capacity(nsub);
    for s in 0..nsub {
        let v = &vec[s * subdim..(s + 1) * subdim];
        let mut best = 0usize;
        let mut best_d = f32::MAX;
        for c in 0..k {
            let row = s * k + c;
            let cent = &codebook[row * subdim..(row + 1) * subdim];
            let d: f32 = v.iter().zip(cent.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        codes.push(best as u16);
    }
    Ok(codes)
}

/// PQ_DECODE: reconstrói vetor concatenando centróides indexados por `codes`.
pub fn pq_decode(codes: &[u16], codebook: &[f32], nsub: usize, subdim: usize, k: usize) -> Result<Vec<f32>> {
    if codes.len() != nsub {
        return Err(anyhow!("PQ_DECODE: codes len {} != NSUB {}", codes.len(), nsub));
    }
    if codebook.len() != nsub * k * subdim {
        return Err(anyhow!("PQ_DECODE: codebook len {} != NSUB*K*subdim {}*{}*{}", codebook.len(), nsub, k, subdim));
    }
    for &c in codes {
        if (c as usize) >= k {
            return Err(anyhow!("PQ_DECODE: code {} >= K {}", c, k));
        }
    }
    if codebook.iter().any(|x| !x.is_finite()) {
        return Err(anyhow!("PQ_DECODE: codebook não-finito recusado (NaN/Inf)"));
    }
    let mut out = Vec::with_capacity(nsub * subdim);
    for (s, &code) in codes.iter().enumerate() {
        let row = s * k + code as usize;
        out.extend_from_slice(&codebook[row * subdim..(row + 1) * subdim]);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_validate_golden() {
        // D=4 NSUB=2 => subdim=2 K=2 (rows=4)
        assert_eq!(pq_validate(4, 2, 4, 2).unwrap(), (2, 2));
        assert!(pq_validate(4, 3, 6, 2).is_err()); // D%NSUB
        assert!(pq_validate(4, 2, 3, 2).is_err()); // rows%NSUB
        assert!(pq_validate(4, 2, 4, 3).is_err()); // cols != subdim
        assert!(pq_validate(4, 0, 4, 2).is_err());
        assert!(pq_validate(0, 2, 4, 2).is_err());
    }

    #[test]
    fn pq_roundtrip_small() {
        // NSUB=2 subdim=2 K=2 -> codebook 4x2
        // s0: c0=[0,0] c1=[10,10]; s1: c0=[0,0] c1=[10,10]
        let cb = vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 10.0, 10.0];
        let v = vec![0.1, 0.2, 9.8, 10.1];
        let codes = pq_encode(&v, &cb, 2).unwrap();
        assert_eq!(codes, vec![0, 1]);
        let (subdim, k) = pq_validate(4, 2, 4, 2).unwrap();
        let rec = pq_decode(&codes, &cb, 2, subdim, k).unwrap();
        assert_eq!(rec, vec![0.0, 0.0, 10.0, 10.0]);
        // Empate => menor índice (distância igual a dois centróides equidistantes).
        let cb2 = vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0];
        let v2 = vec![1.0, 0.0, 1.0, 0.0];
        let codes2 = pq_encode(&v2, &cb2, 2).unwrap();
        assert_eq!(codes2, vec![0, 0]); // 1.0 equidistante de 0 e 2 -> 0 vence
        // NaN recusa, geometria inconsistente recusa.
        assert!(pq_encode(&[f32::NAN, 0.0, 0.0, 0.0], &cb, 2).is_err());
        assert!(pq_encode(&v, &vec![f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2).is_err());
        assert!(pq_encode(&[0.0, 0.0, 0.0], &cb, 2).is_err());
        // Decode: code >=K erra.
        assert!(pq_decode(&[2, 0], &cb, 2, 2, 2).is_err());
        assert!(pq_decode(&[0], &cb, 2, 2, 2).is_err());
    }

    #[test]
    fn lifecycle_store() {
        let mut s = IndexStore::new();
        let db = s.create(4).unwrap();
        assert!(db > 0);
        assert_eq!(s.len(db).unwrap(), 0);
        let a = s.add(db, &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        let b = s.add(db, &[0.0, 1.0, 0.0, 0.0], Some(42)).unwrap();
        assert_eq!((a, b), (0, 42));
        assert_eq!(s.len(db).unwrap(), 2);
        assert_eq!(s.del(db, 42).unwrap(), 1);
        assert_eq!(s.len(db).unwrap(), 1);
        // Erros altos: id ausente/duplicado, dim errada, NaN, store inexistente.
        assert!(s.del(db, 42).is_err());
        assert!(s.add(db, &[1.0, 0.0, 0.0, 0.0], Some(0)).is_err());
        assert!(s.add(db, &[1.0, 0.0], None).is_err());
        assert!(s.add(db, &[f32::NAN, 0.0, 0.0, 0.0], None).is_err());
        assert!(s.add(db, &[f32::INFINITY, 0.0, 0.0, 0.0], None).is_err());
        assert!(s.add(999, &[1.0, 0.0, 0.0, 0.0], None).is_err());
        assert!(s.create(0).is_err());
        assert!(s.len(999).is_err());
    }

    #[test]
    fn metric_tie_break() {
        // Empate => menor id; fórmulas espelham DISTANCE.
        let q = [1.0f32, 0.0];
        let scored = vec![(2u64, metric_dist(RagMetric::Euclid, &q, &[1.0, 1.0], 1.0)), (0u64, metric_dist(RagMetric::Euclid, &q, &[1.0, 1.0], 1.0))];
        assert_eq!(scored[0].1, scored[1].1);
        let top = topk_by_id(&scored, 2);
        assert_eq!(top[0].0, 0);
        assert_eq!(top[1].0, 2);
        // Euclid exato no match; cosine exato no match; dot negado.
        assert_eq!(metric_dist(RagMetric::Euclid, &q, &[1.0, 0.0], 1.0), 0.0);
        assert_eq!(metric_dist(RagMetric::Cosine, &q, &[1.0, 0.0], 1.0), 0.0);
        assert_eq!(metric_dist(RagMetric::Dot, &[1.0, 0.0], &[2.0, 0.0], 1.0), -2.0);
        // Métrica inválida erra.
        assert!(RagMetric::from_u8(3).is_err());
    }
}
