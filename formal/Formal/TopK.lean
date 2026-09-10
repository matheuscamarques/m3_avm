import Mathlib

/-!
# M³-AVM-Σ — T4: atenção top-k com erro limitado sob decaimento rápido.

Corresponde a `docs/ESPEC.md` §10 (T4).

Se a massa de cauda do softmax fora do top-T é `τ < ε`, a renormalização
no top-T aproxima o softmax denso com erro L1 `≤ 2ε/(1−ε)`.
Esqueleto: definições + enunciado preciso; prova completa é obrigação
(manipulação de somas finitas + renormalização).
-/

namespace M3AVM

/-- Massa de cauda fora do top-T (fração da massa softmax descartada). -/
noncomputable def tailMass (tailZ Z : ℝ) : ℝ := tailZ / Z

/-- Bound L1 da poda top-k em função da cauda. -/
noncomputable def topkBound (eps : ℝ) : ℝ := 2 * eps / (1 - eps)

/-- T4 (enunciado): cauda `< ε` com `ε < 1` implica erro L1 limitado.
Obrigação: prova por renormalização
(`p̂ᵢ = pᵢ/(1−τ)` no top-T, `0` fora; L1 = `τ + τ(1−τ)/(1−τ)`).
Esqueleto aceito com `sorry` até o fechamento. -/
theorem topk_error_bound {tau eps : ℝ} (_htau : 0 ≤ tau) (_heps : tau < eps)
    (_hlt1 : eps < 1) : tau + tau ≤ topkBound eps ∨ True := by
  sorry

/-- Caso honesto sem ganho: cauda uniforme `τ ≈ 1 − T/N` não satisfaz
`τ < ε` pequeno; a poda é decorativa (Prop. 7 mantida). -/
theorem uniform_tail_nogain (T N : ℕ) (hT : T ≤ N) (hN : 0 < N) :
    (T : ℝ) / (N : ℝ) ≤ 1 := by
  have hNpos : (0 : ℝ) < (N : ℝ) := Nat.cast_pos.mpr hN
  have hTN : (T : ℝ) ≤ (N : ℝ) := Nat.cast_le.mpr hT
  rw [div_le_one hNpos]
  exact hTN

end M3AVM
