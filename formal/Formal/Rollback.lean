import Mathlib.Tactic

/-!
# M³-AVM — Correção do rollback (snapshot / restore versionado).

Corresponde a `docs/ESPEC.md` §6.3/§8 e ao código `src/memory.rs:1013-1045`.

O payload é abstrato (`σ`): o teorema é sobre a *disciplina de versionamento*.
O modelo espelha o código, inclusive o detalhe de `restore` rebaixar
`version := v` — o que permite demonstrar:
- `snapshot_restore_roundtrip`: restore imediato é exato (Teorema 4);
- `clobber_demo` / `clobber_loses`: restore-then-snapshot pode soterrar um
  snapshot antigo (achado real — ver `FORMAL_FINDINGS` no README do `formal/`);
- variante `restoreFix` (contador monotônico) + `fresh_of_inv`: com o fix,
  todo id novo é fresco e o roundtrip vale incondicionalmente.
-/

namespace M3AVM

/-- Snapshot store: contador + lista de associação `(versão, estado)` + atual. -/
structure Snap (σ : Type) where
  counter : ℕ
  snaps : List (ℕ × σ)
  cur : σ

/-- Lookup newest-first (semântica de `HashMap::get` após `insert`). -/
def findSnap : List (ℕ × σ) → ℕ → Option σ
  | [], _ => none
  | (k, v) :: t, n => if k = n then some v else findSnap t n

/-- `snapshot()`: `version += 1`, insere, retorna o id. (`src/memory.rs:1013`) -/
def snapshot (st : Snap σ) (s : σ) : Snap σ × ℕ :=
  let id := st.counter + 1
  ({ counter := id, snaps := (id, s) :: st.snaps, cur := s }, id)

/-- `restore(v)`: recoloca o estado e faz `version := v`. (`src/memory.rs:1027`) -/
def restore (st : Snap σ) (n : ℕ) : Option (Snap σ) :=
  match findSnap st.snaps n with
  | none => none
  | some s => some { counter := n, snaps := st.snaps, cur := s }

theorem findSnap_cons_hit (t : List (ℕ × σ)) (k : ℕ) (v : σ) :
    findSnap ((k, v) :: t) k = some v := by
  simp [findSnap]

/-- Teorema 4 (roundtrip imediato): restore do snapshot recém-criado é exato. -/
theorem snapshot_restore_roundtrip (st : Snap σ) (s : σ) :
    (restore (snapshot st s).1 (snapshot st s).2).map Snap.cur = some s := by
  simp [snapshot, restore, findSnap]

/-- Traço concreto: snapshot 111 (v1), snapshot 222 (v2), restore v1,
snapshot 333 — o id reutilizado `2` soterra o 222. -/
def clobberTrace : Snap ℕ :=
  let st0 : Snap ℕ := ⟨0, [], 0⟩
  let (st1, _) := snapshot st0 111
  let (st2, _) := snapshot st1 222
  let st3 := (restore st2 1).getD st2
  (snapshot st3 333).1

/-- O lookup do id reutilizado devolve o valor novo, não o antigo. -/
theorem clobber_demo : findSnap clobberTrace.snaps 2 = some 333 := by
  decide

/-- …logo o snapshot antigo (222) fica inacessível: perda real. -/
theorem clobber_loses : findSnap clobberTrace.snaps 2 ≠ some 222 := by
  decide

/-- Variante corrigida: `restore` preserva o contador (monotônico). -/
def restoreFix (st : Snap σ) (n : ℕ) : Option (Snap σ) :=
  match findSnap st.snaps n with
  | none => none
  | some s => some { counter := st.counter, snaps := st.snaps, cur := s }

/-- Invariante: toda chave armazenada é `≤ counter`. -/
def Inv (st : Snap σ) : Prop :=
  ∀ k ∈ (st.snaps.map Prod.fst), k ≤ st.counter

theorem inv_init (s : σ) : Inv (⟨0, [], s⟩ : Snap σ) := by
  intro k hk
  simp at hk

theorem inv_snapshot (st : Snap σ) (h : Inv st) (s : σ) :
    Inv (snapshot st s).1 := by
  intro k hk
  simp only [snapshot] at hk ⊢
  simp at hk
  rcases hk with rfl | ⟨v, hmem⟩
  · exact le_rfl
  · exact Nat.le_trans (h k (List.mem_map.mpr ⟨(k, v), hmem, rfl⟩)) (Nat.le_succ _)

theorem inv_restoreFix (st : Snap σ) (h : Inv st) (n : ℕ) (r : Snap σ)
    (hr : restoreFix st n = some r) : Inv r := by
  have hfind : ∃ s, findSnap st.snaps n = some s := by
    unfold restoreFix at hr
    cases heq : findSnap st.snaps n with
    | none => simp [heq] at hr
    | some s => exact ⟨s, rfl⟩
  obtain ⟨s, hs⟩ := hfind
  unfold restoreFix at hr
  simp [hs] at hr
  subst hr
  exact h

/-- Com o fix, o próximo id é sempre fresco: clobber impossível. -/
theorem fresh_of_inv (st : Snap σ) (h : Inv st) :
    st.counter + 1 ∉ st.snaps.map Prod.fst := by
  intro hm
  have hle := h _ hm
  omega

/-- Roundtrip também vale na variante corrigida. -/
theorem restoreFix_roundtrip (st : Snap σ) (s : σ) :
    (restoreFix (snapshot st s).1 (snapshot st s).2).map Snap.cur = some s := by
  simp [snapshot, restoreFix, findSnap]

/-- Ids de snapshots sucessivos são estritamente crescentes (com o fix). -/
theorem snapshot_counter_succ (st : Snap σ) (s : σ) :
    (snapshot st s).1.counter = st.counter + 1 := rfl

end M3AVM
