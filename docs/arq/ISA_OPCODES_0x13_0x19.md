# ISA M³-AVM — Opcodes `0x13–0x19` (Mamba / Codec / Híbrido)

Extensão da ISA base (`0x01–0x12`, ver `README.md` §3). Formato fixo 32B
(`src/opcodes.rs:88`): `[op|flags|rdest|rsrc1|rsrc2|rsrc3|payload:26]`.

## 1. Tabela

| Op | Hex | Sintaxe `.m3asm` | Registradores | Payload | Flags |
|---|---|---|---|---|---|
| `SSM_SCAN` | `0x13` | `SSM_SCAN rY,rX,rH,rP [D_INNER=n D_STATE=n LAYER=n]` | `rY=[1,di]` out; `rX=[1,di]`; `rH=[di,ds]` in-place ou `_`=estado na Vm; `rP`=pack `dt+A+B+C+D` (montar com `ssm::pack_params`) ou `_`=defaults | `[0..2]=di u16, [2..4]=ds u16, [4]=layer` | `CONV/GATE` rejeitadas (erro explícito; conv via `MATVEC`, gate via `SILU+MUL`) |
| `SSM_RESET` | `0x14` | `SSM_RESET rH [D_INNER=n D_STATE=n LAYER=n]` | `rH` ou ausente/`_` => `ssm_states[layer]` | mesmo | — |
| `CODEC_ENC` | `0x15` | `CODEC_ENC rD,rS [TENSOR]` | `rS`=PCM `1920xf32` (tensor ou bytes `TEMPORAL`); `rD`=codes 32B `TEMPORAL` ou `[1,16]` tensor | — | `TENSOR`=out tensor |
| `CODEC_DEC` | `0x16` | `CODEC_DEC rD,rS [TENSOR]` | inverso | — | idem |
| `AUDIO_ALIGN` | `0x17` | `AUDIO_ALIGN rD,rU,rA [SR=.. FRAME=.. HZ=.. DELAY=..]` | `rU/rA`=ts (imed./tensor/`TEMPORAL`); `rD=[1,4]` `[t_user,t_ai,delta,frame]` | `[0..4]=sr, [4..8]=spf, [8..12]=hz*100, [12..16]=delay*100` | — |
| `CTX_SWITCH` | `0x18` | `CTX_SWITCH MAMBA\|TRANSFORMER\|AUDIO [,RED\|BLUE\|GREEN]` | pipe 0/1/2 no `payload[0]`+`MAGIC` (`instr_ctx_switch`; `rsrc1` é legado) | `[0]=pipe, [1]=0xA5` | prio `FORK_FLAG_*` |
| `ROPE` | `0x19` | `ROPE rD,rS POS=n HDIM=n NHEADS=n [THETA=n] [INPLACE]` | `rS=[..,nh*hd]`; `rD` novo tensor | `[0..4]=pos u32, [4..6]=hd, [6..8]=nh, [8..12]=theta f32` | `INPLACE` (view: `rD` novo addr) |

`SENSE` estendido: `6=AUDIO_PCM` (7680B PCM), `7=CODEC_FRAME` (32B codes).
`PIPE_*`: `0=MAMBA, 1=TRANSFORMER (default), 2=AUDIO` (`opcodes.rs`, `context.rs`).

## 2. Semântica de referência

* `SSM_SCAN`: `h=h*exp(dt*A)+x*B*dt; y=h·C+D*x` (`ssm::selective_scan_update`).
  Pack: `dt[di]+A[di*ds]+B[ds]+C[ds]+D[di]` f32 LE (montar com `ssm::pack_params`,
  que valida tamanhos); sem pack => `dt=1,A=-1,B=C=1,D=0`
  (golden `ssm.rs::test_scan_decay_and_leak`). `Rh=_` usa `Vm::ssm_states[layer]`
  (híbrido; `ensure` com `d_conv` preservado). `FORK` empilha snapshot, `ABORT` dá pop.
* `CODEC_*`: `mimi::encode_frame/decode_frame` (16cb `12.5Hz/24kHz`, `MIMI_SILENCE_CODE=512`).
  Tensor path guarda codes como `f32` (`0..1023`); bytes path usa `TEMPORAL`.
* `AUDIO_ALIGN`: `delta=t_ai-t_user` (ns, wrapping), `frame=delta_ms/80`.
  `read_ts` aceita imediato, tensor (`f32[0]`) ou 8B LE — mas bytes só de região
  `TEMPORAL` (nunca `GLOBAL`, para não ler peso como timestamp); demais casos = imediato.
* `CTX_SWITCH`: `ctx.pipeline=pipe; ctx.priority=flags; root_version=snapshot(); maybe_preempt()`.
  `rsrc1` é **imediato**, não registrador.
* `ROPE`: `inference::apply_rope` por bloco `[nh*hd]` (batch OK); `pos=0` = identidade;
  `hd` ímpar => erro (igual early-return do host, mas explícito na VM).

## 3. Exemplos

```asm
; Mamba (programs/mamba_scan_demo.m3asm)
TENSOR r0 1 2 f32
TENSOR r1 2 1 f32
CTX_SWITCH MAMBA, GREEN
SSM_SCAN r5, r0, r1, r0 D_INNER=2 D_STATE=1 LAYER=0
ROPE r2, r0 POS=0 HDIM=2 NHEADS=1

; Codec (programs/codec_loop.m3asm)
SENSE r0, AUDIO_PCM
CODEC_ENC r1, r0
CODEC_DEC r2, r1
AUDIO_ALIGN r4, r0, r2

; Full-duplex (programs/moshi_loop_v2.m3asm)
SENSE r10, AUDIO_PCM
CODEC_ENC r11, r10
NORM r1, r0, r0, r0
ROPE r1, r1 POS=0 HDIM=8 NHEADS=1
ATTN r4, r1, r1, r1, NOTIFY_EACH_HEAD
SSM_SCAN r9, r0, r7, r0 D_INNER=2 D_STATE=1 LAYER=0
SAMPLE r2, r0
STREAM r2, 4 BLOCKING
CODEC_DEC r13, r11
```

`--emit-asm` agora emite `ROPE` após `MATVEC Q/K` e `SSM_SCAN+CTX_SWITCH MAMBA`
quando `arch` contém `mamba` (`asm_emitter.rs`).

## 4. Determinismo / preempção (tese §5)

`SENSE USER_INPUT → IF_INTERRUPT → ABORT (+pop SSM) → SSM_RESET` restaura `h_t`
O(1) no checkpoint do `FORK`, espelhando `KV truncate`. `CODEC_*` mantém áudio em
tensor/`TEMPORAL` (sem Python/ALSA no hot path); `ALIGN` dá `t_interrupção` p/ barge-in
(`160ms` = frame 80ms + delay 80ms).

---
*Author: Matheus de Camargo Marques — matheuscamarques@gmail.com — ORCID [0009-0003-4518-2258](https://orcid.org/0009-0003-4518-2258).*
