import Mathlib.Analysis.Complex.Exponential
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# M³-AVM — SSM (Mamba-1): estabilidade, erro de Euler, custo constante.

Corresponde a `docs/ESPEC.md` §10 e ao código `src/ssm.rs`.

- Teorema 1a (`ssm_contraction`): `A < 0, dt > 0 → 0 < exp(dt·A) < 1`.
- Teorema 1b (`euler_local_error`): erro local Euler vs. ZOH ≤ `|xB|·(|A|·dt²)`.
- Esquecimento exponencial (`ssm_forget`): recorrência homogênea decai com `aⁿ`.
- Teorema 2 (`scanLoop_length`, `scanCost_eq`): estado limitado, custo linear em N.
-/

namespace M3AVM

/-- Fator de decaimento do passo SSM: `a_d = exp(dt * A)`. (`src/ssm.rs`, eq. 3) -/
noncomputable def ssmDecay (A dt : ℝ) : ℝ := Real.exp (dt * A)

/-- Teorema 1a (contração): com `A < 0` e `dt > 0` o fator está em `(0,1)`. -/
theorem ssm_contraction {A dt : ℝ} (hA : A < 0) (hdt : 0 < dt) :
    0 < ssmDecay A dt ∧ ssmDecay A dt < 1 := by
  have hneg : dt * A < 0 := mul_neg_of_pos_of_neg hdt hA
  constructor
  · exact Real.exp_pos _
  · unfold ssmDecay
    rw [← Real.exp_zero]
    exact (Real.exp_lt_exp).mpr hneg

/-- Passo Euler do código: `h ← h·a + x·B·dt`. (eq. 4) -/
noncomputable def eulerStep (a h xB dt : ℝ) : ℝ := h * a + xB * dt

/-- Passo exato ZOH: `h ← h·a + x·B·(a−1)/A`. -/
noncomputable def zohStep (a h xB A : ℝ) : ℝ := h * a + xB * ((a - 1) / A)

/-- Teorema 1b: o erro local de um passo Euler é `O(dt²)`, com constante explícita.
Hipótese `|A|·dt ≤ 1` = regime de passo pequeno do `softplus` (`src/ssm.rs`). -/
theorem euler_local_error {A dt xB h : ℝ} (hA : A < 0) (hdt : 0 < dt)
    (hsmall : |A| * dt ≤ 1) :
    |eulerStep (ssmDecay A dt) h xB dt - zohStep (ssmDecay A dt) h xB A|
      ≤ |xB| * (|A| * dt ^ 2) := by
  have hA0 : A ≠ 0 := ne_of_lt hA
  have hApos : 0 < |A| := abs_pos.mpr hA0
  have hu1 : |A * dt| ≤ 1 := by
    rw [abs_mul, abs_of_pos hdt]
    exact hsmall
  -- Núcleo analítico: |exp u − 1 − u| ≤ u² para |u| ≤ 1 (Mathlib).
  have key : |Real.exp (A * dt) - 1 - A * dt| ≤ (A * dt) ^ 2 := by
    have h := Real.norm_exp_sub_one_sub_id_le (x := A * dt) (by rwa [Real.norm_eq_abs])
    simp only [Real.norm_eq_abs, sq_abs] at h
    exact h
  -- Álgebra: a diferença dos passos é xB·((A·dt − (exp u − 1))/A).
  have halg : eulerStep (ssmDecay A dt) h xB dt - zohStep (ssmDecay A dt) h xB A
      = xB * ((A * dt - (Real.exp (A * dt) - 1)) / A) := by
    have hcomm : dt * A = A * dt := mul_comm _ _
    unfold eulerStep zohStep ssmDecay
    rw [hcomm]
    field_simp
    ring
  have hneg : A * dt - (Real.exp (A * dt) - 1)
      = -(Real.exp (A * dt) - 1 - A * dt) := by ring
  rw [halg, hneg, abs_mul, abs_div, abs_neg]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
  have h2 : (A * dt) ^ 2 / |A| = |A| * dt ^ 2 := by
    rw [mul_pow, ← sq_abs A]
    field_simp
  calc |Real.exp (A * dt) - 1 - A * dt| / |A|
        ≤ (A * dt) ^ 2 / |A| := (div_le_div_iff_of_pos_right hApos).mpr key
      _ = |A| * dt ^ 2 := h2

/-- Recorrência homogênea `hₙ₊₁ = a·hₙ` (entrada zero). -/
def ssmRun (a h0 : ℝ) : ℕ → ℝ
  | 0 => h0
  | n + 1 => a * ssmRun a h0 n

/-- Esquecimento exponencial: `|hₙ| ≤ aⁿ·|h₀|` para `0 ≤ a`. -/
theorem ssm_forget {a h0 : ℝ} (ha0 : 0 ≤ a) (n : ℕ) :
    |ssmRun a h0 n| ≤ a ^ n * |h0| := by
  induction n with
  | zero => simp [ssmRun]
  | succ k ih =>
    simp only [ssmRun]
    rw [abs_mul, abs_of_nonneg ha0]
    calc a * |ssmRun a h0 k| ≤ a * (a ^ k * |h0|) :=
          mul_le_mul_of_nonneg_left ih ha0
      _ = a ^ (k + 1) * |h0| := by ring

/-- Loop de scan: aplica `step` n vezes. O passo preserva o tamanho do estado. -/
def scanLoop (step : List ℝ → List ℝ) : ℕ → List ℝ → List ℝ
  | 0, s => s
  | n + 1, s => step (scanLoop step n s)

/-- Teorema 2a: o estado nunca cresce com N (custo de memória constante). -/
theorem scanLoop_length (step : List ℝ → List ℝ)
    (hstep : ∀ s, (step s).length = s.length) (n : ℕ) (s : List ℝ) :
    (scanLoop step n s).length = s.length := by
  induction n generalizing s with
  | zero => rfl
  | succ k ih =>
    simp only [scanLoop]
    rw [hstep, ih]

/-- Custo acumulado de n passos a custo unitário c. -/
def scanCost : ℕ → ℕ → ℕ
  | 0, _ => 0
  | n + 1, c => scanCost n c + c

/-- Teorema 2b: custo total é exatamente `n·c` (linear em N, sem termo oculto). -/
theorem scanCost_eq (n c : ℕ) : scanCost n c = n * c := by
  induction n with
  | zero => simp [scanCost]
  | succ k ih => simp [scanCost, ih]; ring

end M3AVM
