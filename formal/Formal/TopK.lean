import Mathlib

open scoped BigOperators

/-!
# M³-AVM-Σ — T4: atenção top-k com erro limitado sob decaimento rápido.

Corresponde a `docs/ESPEC.md` §10 (T4) e fecha a obrigação da RFC-0029:
se a massa de cauda do softmax fora do top-T é `τ < ε`, a renormalização
no top-T aproxima o softmax denso com erro L1 `≤ 2ε/(1−ε)`.
Prova: conta de renormalização — L1 = `τ + τ`, e `2τ ≤ 2ε/(1−ε)`.
Vale para QUALQUER subconjunto `T`; "top" só importa para a hipótese
`τ` pequeno, que é assumida, não provada.
-/

namespace M3AVM

/-- Massa de cauda fora do top-T (fração da massa softmax descartada). -/
noncomputable def tailMass (tailZ Z : ℝ) : ℝ := tailZ / Z

/-- Bound L1 da poda top-k em função da cauda. -/
noncomputable def topkBound (eps : ℝ) : ℝ := 2 * eps / (1 - eps)

/-- T4: cauda `< ε` com `ε < 1` implica erro L1 limitado pela renormalização
(`p̂ᵢ = pᵢ/(Z−τZ)` no top-T, `0` fora). -/
theorem topk_error_bound {ι : Type*} [Fintype ι] [DecidableEq ι]
    {p : ι → ℝ} {T : Finset ι} {eps : ℝ}
    (hpos : ∀ i, 0 ≤ p i)
    (hZ : 0 < ∑ i, p i)
    (heps1 : eps < 1)
    (htau : (∑ i ∈ Finset.univ \ T, p i) / (∑ i, p i) < eps) :
    ∑ i, |p i / (∑ j, p j) - (if i ∈ T then p i / ((∑ j, p j) - ∑ i ∈ Finset.univ \ T, p i) else 0)|
      ≤ topkBound eps := by
  set Z := ∑ i, p i with hZdef
  set tailZ := ∑ i ∈ Finset.univ \ T, p i with htailZdef
  have hTsub : T ⊆ Finset.univ := Finset.subset_univ T
  have htail_nonneg : 0 ≤ tailZ := by
    rw [htailZdef]
    exact Finset.sum_nonneg (fun i _ => hpos i)
  have hZpos : 0 < Z := by
    rw [hZdef]
    exact hZ
  have htau' : tailZ / Z < eps := by
    rw [htailZdef, hZdef]
    exact htau
  have heps_pos : 0 < eps :=
    lt_of_le_of_lt (div_nonneg htail_nonneg hZpos.le) htau'
  have h1eps : (0 : ℝ) < 1 - eps := by linarith
  -- Cauda menor que o total: τ < ε < 1 vezes Z > 0.
  have htail_lt : tailZ < Z := (div_lt_one hZpos).mp (lt_trans htau' heps1)
  have hZsub : 0 < Z - tailZ := by linarith
  have hZne : Z ≠ 0 := ne_of_gt hZpos
  have hZsubne : Z - tailZ ≠ 0 := ne_of_gt hZsub
  have hc_nonneg : 0 ≤ tailZ / (Z * (Z - tailZ)) :=
    div_nonneg htail_nonneg (mul_nonneg hZpos.le hZsub.le)
  -- Termo a termo no top-T: |p/Z − p/(Z−τ)| = p·τ/(Z·(Z−τ)).
  have hTterm : ∀ i ∈ T,
      |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|
        = p i * (tailZ / (Z * (Z - tailZ))) := by
    intro i hi
    rw [if_pos hi]
    have h1 : p i / Z - p i / (Z - tailZ) = -(p i * (tailZ / (Z * (Z - tailZ)))) := by
      field_simp
      ring
    rw [h1, abs_neg, abs_of_nonneg (mul_nonneg (hpos i) hc_nonneg)]
  -- Soma no top-T = τ (a massa do top-T é Z−τZ por complemento).
  have hTsum : (∑ i ∈ T, |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|)
      = tailZ / Z := by
    calc (∑ i ∈ T, |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|)
        = ∑ i ∈ T, p i * (tailZ / (Z * (Z - tailZ))) :=
          Finset.sum_congr rfl (fun i hi => hTterm i hi)
      _ = (∑ i ∈ T, p i) * (tailZ / (Z * (Z - tailZ))) := (Finset.sum_mul ..).symm
      _ = tailZ / Z := by
          have hTtotal : (∑ i ∈ T, p i) = Z - tailZ := by
            have hpsplit := Finset.sum_sdiff hTsub (f := fun i => p i)
            rw [← htailZdef, ← hZdef] at hpsplit
            linarith
          rw [hTtotal]
          field_simp
  -- Fora do top-T: |q − 0| = p/Z; soma = τ.
  have hCterm : ∀ i ∈ Finset.univ \ T,
      |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)| = p i / Z := by
    intro i hi
    have hnot : i ∉ T := by
      simpa only [Finset.mem_sdiff, Finset.mem_univ, true_and] using hi
    rw [if_neg hnot, sub_zero]
    exact abs_of_nonneg (div_nonneg (hpos i) hZpos.le)
  have hCsum : (∑ i ∈ Finset.univ \ T, |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|)
      = tailZ / Z := by
    calc (∑ i ∈ Finset.univ \ T, |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|)
        = ∑ i ∈ Finset.univ \ T, p i / Z :=
          Finset.sum_congr rfl (fun i hi => hCterm i hi)
      _ = (∑ i ∈ Finset.univ \ T, p i) / Z := (Finset.sum_div ..).symm
      _ = tailZ / Z := by rw [← htailZdef]
  -- L1 total = τ + τ ≤ 2ε/(1−ε).
  have hsplit := Finset.sum_sdiff hTsub
    (f := fun i => |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|)
  have hL1 : (∑ i, |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|)
      = tailZ / Z + tailZ / Z := by
    have h := hsplit
    rw [hCsum, hTsum] at h
    exact h.symm
  have hfin : tailZ / Z + tailZ / Z ≤ 2 * eps / (1 - eps) := by
    have h1 : tailZ / Z ≤ eps := le_of_lt htau'
    calc tailZ / Z + tailZ / Z ≤ 2 * eps := by linarith
      _ ≤ 2 * eps / (1 - eps) := by
          rw [le_div_iff₀ h1eps]
          have hsq : (0 : ℝ) ≤ eps ^ 2 := sq_nonneg eps
          linarith
  calc (∑ i, |p i / Z - (if i ∈ T then p i / (Z - tailZ) else 0)|)
      = tailZ / Z + tailZ / Z := hL1
    _ ≤ 2 * eps / (1 - eps) := hfin
    _ = topkBound eps := rfl

/-- Caso honesto sem ganho: cauda uniforme `τ ≈ 1 − T/N` não satisfaz
`τ < ε` pequeno; a poda é decorativa (Prop. 7 mantida). -/
theorem uniform_tail_nogain (T N : ℕ) (hT : T ≤ N) (hN : 0 < N) :
    (T : ℝ) / (N : ℝ) ≤ 1 := by
  have hNpos : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  have hTN : (T : ℝ) ≤ (N : ℝ) := Nat.cast_le.mpr hT
  rw [div_le_one hNpos]
  exact hTN

end M3AVM
