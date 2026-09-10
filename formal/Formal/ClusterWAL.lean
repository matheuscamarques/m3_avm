/-!
# M³-AVM-Σ — T2/T5: migração com WAL conserva estado (fail-stop 1 nó).

Corresponde a `docs/ESPEC.md` §12 (T5) e §8 (T2 distribuído).

Modelo mínimo: cada tensor tem presença na origem, no destino e retenção
no WAL da origem. Invariante: `origem ∨ destino` (nunca perdido).
Transições preservam o invariante sob falha de no máximo 1 nó.
`MOVE = COPY + invalidação sem retenção` viola o invariante
(contraexemplo `moveWithoutWal_loses` abaixo).
-/

namespace M3AVM

/-- Presença de um tensor no episódio de migração. -/
structure Mig where
  origin : Bool
  dest : Bool
  wal : Bool
deriving DecidableEq, Repr

/-- Invariante de conservação: o tensor existe em ≥1 nó vivo. -/
def MigInv (m : Mig) : Prop := m.origin = true ∨ m.dest = true

/-- Estado inicial: origem tem, WAL logado, destino ainda não. -/
def migInit : Mig := ⟨true, false, true⟩

theorem migInit_inv : MigInv migInit := by
  simp [MigInv, migInit]

/-- `SEND`: destino recebe cópia (origem + WAL retidos). -/
def migSend (m : Mig) : Mig := ⟨m.origin, true, m.wal⟩

theorem migSend_inv (m : Mig) (_h : MigInv m) : MigInv (migSend m) := by
  simp [MigInv, migSend]

/-- `ACK + marca coletável mas retém por janela`: origem pode liberar o
vivo mas o WAL ainda conta como retenção lógica nesta abstração. -/
def migAckRetain (m : Mig) : Mig := ⟨m.origin, true, true⟩

theorem migAckRetain_inv (m : Mig) (_h : MigInv m) : MigInv (migAckRetain m) := by
  simp [MigInv, migAckRetain]

/-- Falha da origem após ACK com retenção: destino tem a cópia. -/
def migFailOrigin (m : Mig) : Mig := ⟨false, m.dest, m.wal⟩

theorem migFailOrigin_inv (m : Mig) (_h : MigInv (migAckRetain m)) :
    MigInv (migFailOrigin (migAckRetain m)) := by
  simp [MigInv, migAckRetain, migFailOrigin]

/-- Falha do destino: origem/WAL retêm e reenviam. -/
def migFailDest (m : Mig) : Mig := ⟨m.origin, false, m.wal⟩

theorem migFailDest_inv (m : Mig) (_h : MigInv m) (ho : m.origin = true) :
    MigInv (migFailDest m) := by
  simp [MigInv, migFailDest, ho]

/-- Contraexemplo do `MOVE` atual: invalida origem sem WAL e sem ACK do
destino → tensor perdido (nenhum nó tem). -/
def moveWithoutWal : Mig := ⟨false, false, false⟩

theorem moveWithoutWal_loses : ¬ MigInv moveWithoutWal := by
  simp [MigInv, moveWithoutWal]

end M3AVM
