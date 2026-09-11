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

#[cfg(test)]
mod tests {
    use super::*;

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
}
