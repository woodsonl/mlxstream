# basalt `native_qsa.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's `native_qsa.cu` (the weighted RMS norm
`norm<BlockSize>` behind `native_qsa_rms_norm_weighted`, and the sigmoid output gate `gate`), read from the sm_120
SASS (with the PTX where it shows what the front end chose), and the Metal form the port writes at each site. This is
the file's page under invariant 5 of `docs/design/flash-next.md`; `sampler.md` (branch `woodsonl/basalt-sampler`) is
the model for its layout and notation.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/native_qsa.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/native_qsa.cu` with its one local include inlined under `#line` directives
  (`include/basalt/kernels/native_qsa.hpp`), so every PTX and SASS line names its basalt file and line. CUDA's own
  headers come from the compiler's installation. Only basalt's public MIT source was sent.
- **Flags.** `native_qsa.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:140`) and is in
  the `--use_fast_math` list (`:186-202`, the file at `:193`, the option at `:202`). A Release build (`:12-14`)
  passes CMake's `-O3 -DNDEBUG`, `add_compile_definitions` passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one
  architecture is 120 (`:3-4`, `:26-31`). `--use_fast_math` means `-ftz=true -prec-div=false -prec-sqrt=false
  -fmad=true` and the fast intrinsic for each single-precision library call; no other `-fmad`, `-ftz`, `-prec-div`
  or `-Xptxas` flag is set. So every single-precision `FADD`, `FMUL` and `FFMA` below carries `.FTZ`, and `/` is the
  approximate division (`div.full.f32` or `div.approx.f32`, section 3).
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`. Compiler id `nvcc130`: NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48`. Basalt requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`). The
  file was also compiled with `nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) as a cross-check
  (section 3).
- **userArguments**, verbatim (the responses' `compilationOptions` echo them). nvcc 13.0, quoted by the tables:
  `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"' --use_fast_math`; nvcc 13.0 again and nvcc 13.3:
  `-arch=sm_120 -lineinfo -O3 -DNDEBUG --use_fast_math`. The two 13.0 compiles give the same SASS for all three
  kernels instruction for instruction (registers and labels renamed), as they must: nothing in `native_qsa.cu` or
  `include/basalt` names `BASALT_VERSION` (grep). Compiler Explorer adds `-g -lineinfo --keep-device-functions -c
  -keep -Xcompiler=-S` and its host compiler (GCC 14.1); none of them changes a kernel's code (`sampler.md`
  section 1).
- **Date.** Compiled October 16, 2026 (both compilers).
- **Translation unit.** The whole file in one unit; its SASS listing is complete (1,089 lines: `norm<1024>`,
  `norm<256>`, `gate` and CUDA's unsigned 64-bit division subroutine).
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.ftz.f32`) and which division
  it chose; the SASS shows what `ptxas` made of it. A row quotes the SASS of one copy where a site is unrolled; the
  copies were compared and differ only in registers. The instructions after a kernel's last `EXIT` are CUDA's
  out-of-line subroutines, which `-lineinfo` attributes to the line before them; the tables attribute each to the
  `CALL.REL.NOINC` that reaches it.
- **Not float arithmetic.** `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` and `IMAD.HI.U32` at a strided loop's
  head are `ptxas`'s exact integer division; `HFMA2 R, -RZ, RZ, a, b` moves a constant; `IMAD.WIDE` forms addresses.
  The tables leave them out unless they read a stored value.

## 2. Kernels in the file and their paths

`native_qsa.cu` holds two kernels and their host launchers, ported from llama.cpp's `norm.cu`, `common.cuh` and
`unary.cu` (`:1-2`). The engine forces the native QSA set on, `o.native_qsa = o.native_qsa_indexer = o.native_rope =
o.native_ple_postops = true;` (`src/program/generate.cpp:1893`), and applies it with
`basalt::kernels::native_qsa_set_enabled(o.native_qsa);` (`:2399`) before any session exists; the flag is read only on
the host (`:105-106`). `native_qsa_rms_norm_weighted` picks the instance from the width alone, `norm<256>` for
`n_cols < 1024` and `norm<1024>` otherwise (`:114-117`), so a row's bits never depend on how many rows a call
carries.

| Kernel or function | Lines | Path | Evidence |
|---|---|---|---|
| `norm<256>` via `native_qsa_rms_norm_weighted` | `:44-68`, `:108-119` (launch `:114-115`) | ON: decode, the verify window's K norm (2 heads × 256 per row), query norm (24 × 256) and indexer-query norm (4 × 128), in place | `norm_rope` (`src/core/verify.cpp:1340-1345`, the call at `:1341`), called at `verify.cpp:1451`, `:1452` (K), `:1506`, `:1519` (query), `:1510`, `:1525` (indexer query); with fp16 KV the fused `qsa_dec_kv_post` is not taken (`fused_pre` needs `st.kv_int8`, `verify.cpp:1352-1353`) |
| the same | | ON: drafter, its K rows (2 × 256), in place | `norm_rope` (`src/core/mtp.cpp:763-767`, the call at `:764`), called at `mtp.cpp:841`; `kv_fused` needs `st_.kv_int8` (`mtp.cpp:750-751`). The drafter's query norm is the fused `qsa_dec_q_post` (`mtp.cpp:861-864`, `qsa_dec_fused.md`) |
| `norm<1024>` via `native_qsa_rms_norm_weighted` | `:44-68`, `:116-117` | ON: drafter, the hidden-input norm (`T` rows × `HC * N` = 4 × 2560 = 10240) and the embedding-input norm (`T` × 2560), out of place | `mtp.cpp:727` (`record_hidden`, called at `:796` and `:974`) and `mtp.cpp:792`; the widths are `n_embd = 2560` and `hc = 4` (`include/basalt/core/layout.hpp:28`, `:48`). Not a QSA norm, but this file's kernel on the drafter's path |
| `gate` via `native_qsa_gate_apply` | `:69-77`, `:120-126` | OFF as a kernel: decode runs the gate inside `qsa_dec_attn`'s merge, `if (fused_gate) {}   // the merge wrote attn32_` (`verify.cpp:1572`), `fused_gate` holding with fp16 pools (`:1558-1559`); the drafter takes the fused merge at `mtp.cpp:881-883`, so its `:888` does not run | its arithmetic is ON inside the merge as inline PTX, `gate_fast` (`src/kernels/cuda/qsa_dec_attn.cu:308-319`, "sigmoid gate with the fast-math sequence native_qsa_gate_apply compiles to"), whose SASS is `qsa_dec_attn.md`'s; this kernel runs as `qsa_dec_attn_parity`'s reference (`src/kernels/qsa_dec_attn_parity.cpp:131`, `:173`, `:321`) |
| `layer.cpp:906` (norm), `:1036` (gate) in `qsa_layer` | | OFF: the per-token layer forward, which the contract's decode does not run (`docs/design/flash-next.md:291-292`) | `src/core/layer.cpp:903`, from `block_layer_pre` at `:1280` |
| `warp_sum` | `:38-43` | ON, inlined twice into each `norm` instance | `:56`, `:63` |
| `elements`, `valid`, `overlap`, `buffers`, `check_launch` (host) | `:78-102` | ON (host): the launchers' refusals | `:110-113`, `:122-123` |

Every on-path call passes `epsilon = EPS = 1e-6f` (`verify.cpp:116`, `mtp.cpp:63`). Basalt's tests run `norm<256>`
as the reference the fused kernels are held to bitwise (`src/kernels/qsa_dec_fused_parity.cpp:111`, `:123`, `:125`;
`qsa_dec_fused.md`).

## 3. Per-kernel SASS tables

### 3.1 `norm<256>` and `norm<1024>` (`:44-68`)

`norm<256>` has 448 SASS instructions and `norm<1024>` 440, neither with a subroutine call; their float instructions
are the same per source line (30 `FFMA.FTZ`, 10 `FADD.FTZ`, one `I2FP.F32.S32`, one `MUFU.RCP`, one `MUFU.RSQ`, 66
`FMUL.FTZ`), and they differ only at `:62` (below). Launch (`:114-117`): one block of `BlockSize` threads per row,
grid `n_rows`, no dynamic shared memory (the static `sums[32]`, 128 bytes), no cluster. Thread `tid` owns the columns
`tid, tid + BlockSize, ...` of its row in both loops.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:50-54` | `float partial = 0.0f; for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) { const float x = input[col]; partial += x * x; }` | PTX `fma.rn.ftz.f32 %f24, %f23, %f23, %f130`; SASS `FFMA.FTZ R33, R33, R33, R0`, `FFMA.FTZ R33, R32, R32, R33`, ... one dependent chain in column order (the loop unrolled by 16, then 8, then 4, then 1; every copy is a chain through the previous result) | `partial = fma_ftz(x, x, partial)` in the same column order, from `partial = 0.0f` |
| `:56` (`:38-43`) | `partial = warp_sum(partial);` — `for (int offset = 16; offset; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset, 32);` | `SHFL.BFLY PT, R11, R0, 0x10, 0x1f`; `FADD.FTZ R11, R11, R0`; the same pair with `0x8`, `0x4`, `0x2`, `0x1` | `warp_sum_ftz(partial)` (`kernels/common/basalt_fp.metal:87`): `simd_shuffle_xor` on the bits, offsets 16, 8, 4, 2, 1, `add_ftz` (the add is commutative, so the operand order `ptxas` picked does not matter) |
| `:57-58` | `const int lane = tid % 32; if (lane == 0) sums[tid / 32] = partial;` | `LOP3.LUT P0, R14, R2, 0x1f, RZ, 0xc0, !PT`; `@!P0 STS [R11], R16` | `threadgroup float sums[32]`; lane `tid % 32` writes slot `tid / 32` |
| `:61` | `__syncthreads();` | `BAR.SYNC.DEFER_BLOCKING 0x0` | `threadgroup_barrier(mem_flags::mem_threadgroup)`; only `sums` crosses threads (each thread later reads and writes only its own columns, so `input == output` is exact without a device barrier) |
| `:62` | `partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;` | `norm<256>`: `ISETP.GT.U32.AND P1, PT, R14, 0x7, PT`; `MOV R0, RZ`; `@!P1 LDS R0, [R10]` (lanes 8-31 hold +0.0); `norm<1024>`: `LDS R10, [R10]` unpredicated | the same select; +0.0 added to a sum of squares (never −0) leaves it unchanged |
| `:63` | `partial = warp_sum(partial);` | as `:56`, a second inlined copy | `warp_sum_ftz(partial)`: every SIMD-group reduces the same 32 values, so every thread holds the same total |
| `:64` | `const float mean = partial / n_cols;` | `I2FP.F32.S32 R16, R8` (`cvt.rn.f32.s32`; exact, `n_cols` < 2^24 on every path); PTX `div.approx.ftz.f32 %f100, %f98, %f99`, which `ptxas` expands to `MUFU.RCP R0, R16` and a multiply it then fuses into `:65`'s add, so `mean` is never rounded on its own | `rcp_ftz(float(n_cols))` (`basalt_fp.metal:83`): the correctly rounded reciprocal where SASS has `MUFU.RCP` (invariant 5); exact for the powers of two 128 and 256 |
| `:65` | `const float scale = rsqrtf(mean + epsilon);` | PTX `add.ftz.f32 %f101, %f18, %f100`; `rsqrt.approx.ftz.f32 %f17, %f101`; SASS `FFMA.FTZ R0, R11, R0, UR6` = fma(total, rcp(n_cols), epsilon) (`norm<1024>`: `FFMA.FTZ R15, R15, R0, UR4`); `MUFU.RSQ R0, R0` | `scale = fast::rsqrt(fma_ftz(partial, rcp_ftz(float(n_cols)), epsilon))`; the fma's exact value is at least `epsilon` = 1e-6, a normal, so no flush is in play at `rsqrt` |
| `:66-67` | `for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) output[col] = scale * input[col] * gamma[col];` | `LDG.E R19` (input: not `.CONSTANT`, it may alias output), `LDG.E.CONSTANT R18` (gamma, `__restrict__`); `FMUL.FTZ R19, R0, R19`; `FMUL.FTZ R19, R19, R18`; `STG.E`; unrolled by 4 with a remainder loop, 33 pairs in all | `output[col] = cuda_nan(mul_ftz(mul_ftz(scale, x), g))`, left to right |

### 3.2 `gate` (`:69-77`), OFF-PATH as a kernel

77 SASS instructions, then CUDA's `__internal_0_$__cuda_sm20_div_u64` (75). Launch (`:124`): blocks of 256, grid
⌈`n_head * head_dim` / 256⌉, no shared memory. The kernel is `qsa_dec_attn_parity`'s reference
(`src/kernels/qsa_dec_attn_parity.cpp:131`, `:173`, `:321`); decode and the drafter run the same five operations as
the inline PTX `gate_fast` inside `qsa_dec_attn`'s merge (`src/kernels/cuda/qsa_dec_attn.cu:308-319`, recorded in
`qsa_dec_attn.md`), section 2.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:71-73` | `i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n_head * head_dim) return; head = i / head_dim, channel = i % head_dim;` | `@P0 EXIT`; the 64-bit division by `I2F.U32.RP`/`MUFU.RCP`/`F2I.FTZ.U32.TRUNC.NTZ` when both high words are zero, else `CALL.REL.NOINC` to `$__internal_0_$__cuda_sm20_div_u64` | integer |
| `:74` | `const float raw = q_full[head * 2 * head_dim + head_dim + channel];` | `LDG.E.CONSTANT R6` | per head, `q_full` holds `head_dim` query values and then `head_dim` gate values |
| `:75` | `const float sigmoid = 1.0f / (1.0f + expf(-raw));` | PTX `mul.ftz.f32 %f2, %f1, 0fBFB8AA3B` (`expf` is `__expf` under `--use_fast_math`; the negation is folded into −log2 e), `ex2.approx.ftz.f32 %f3, %f2`, `add.ftz.f32 %f4, %f3, 0f3F800000`, `rcp.approx.ftz.f32 %f5, %f4` (`1.0f / x` with `-prec-div=false`); SASS `FMUL.FTZ R0, R6, -1.4426950216293334961`; `MUFU.EX2 R0, R0`; `FADD.FTZ R8, R0, 1`; `MUFU.RCP R8, R8` | `rcp_ftz(add_ftz(expf_ftz(-raw), 1.0f))` (`basalt_fp.metal:107`, `:49`, `:83`): `mul_ftz(-raw, log2 e)` and `fl(raw · −log2 e)` are the same bits (negation is exact, rounding to nearest even is symmetric) |
| `:76` | `output[i] = attn[i] * sigmoid;` | `LDG.E R5`; `FMUL.FTZ R7, R8, R5` (PTX `mul.ftz.f32 %f7, %f6, %f5`, attn · r); `STG.E` | `cuda_nan(mul_ftz(attn, sigmoid))`; `gate_fast`'s `mul.ftz.f32 %0, %1, r` is the same product |

### 3.3 nvcc 13.3

`nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) gives `norm<256>` 464 instructions, `norm<1024>` 456 and
`gate` 78. Compared per source line as opcode multisets with immediates, the float-class instructions (`FFMA`,
`FADD`, `FMUL`, `MUFU`, `I2F`, `I2FP`; the shuffles move bits) are the same on every line of all three kernels (`norm`: `:41`, `:53`,
`:64`, `:65`, `:67`; `gate`: `:73`, `:75`, `:76`). The integer differences are spelling and scheduling only: `IADD3`
becomes `IADD`, `ISETP.NE.S64.AND` becomes `ISETP.NE.U64.AND` (tests against zero), the loop at `:51` gains
`BSSY.RELIABLE`/`BREAK.RELIABLE` convergence markers and `NOP` padding. No stored bit depends on the toolkit version.

## 4. What the record implies for the Metal port

- **No libdevice routine.** The PTX has no `call`, no `__nv_` or `__cudart_` symbol and no `.f64` operation; the
  SASS has no `FCHK` and no `D*` instruction. Under `--use_fast_math` `rsqrtf` is `rsqrt.approx.ftz.f32`
  (`MUFU.RSQ`), `expf` is `__expf` (`ex2.approx.ftz.f32`, `MUFU.EX2`), `/` is `div.approx.ftz.f32` and `1.0f / x`
  `rcp.approx.ftz.f32` (`MUFU.RCP`). The one subroutine is CUDA's integer `__internal_0_$__cuda_sm20_div_u64` in
  `gate` (`:73`). Nothing here falls under the libdevice rule.
- **Every float operation is a `.FTZ` one, in a fixed order the port writes out.** Per thread a sequential `fma` chain
  in column order (`:53`); the two butterflies with offsets 16, 8, 4, 2, 1 (`:56`, `:63`); the slot write and the
  barrier; `fma(total, rcp(float(n_cols)), epsilon)` with no rounding of `mean` (`:64-65`); `rsqrt`; and
  `(scale · x) · gamma` (`:67`). The kernel starts from `#pragma STDC FP_CONTRACT OFF` and writes each `fma` itself.
- **The lane mapping is part of the bits.** The butterflies and `sums[tid / 32]` give basalt's sums only if the
  SIMD-group of 32 lanes holding `tid` is `tid / 32` and its lane `tid % 32`. Main's ports take `tid`
  (`thread_index_in_threadgroup`) for the strided work and `lane`/`sg` (`thread_index_in_simdgroup`,
  `simdgroup_index_in_threadgroup`) for the shuffles and slots (`kernels/common/basalt_q5k_head.metal:146-155`),
  which assumes that identity for a one-dimensional threadgroup. The rule that holds by construction: form the column
  owner as `t = sg * 32 + lane` and use `t` for the strided loops, `lane` for the shuffles, `sg` for the slot.
  The threadgroup is 256 or 1024 threads, the widths the launcher picks from `n_cols` alone (`:114-117`), so a row's
  bits never depend on the number of rows in the call.
- **What equals CUDA bit for bit, and what does not.** The `FFMA`/`FADD`/`FMUL` chain is exact transcription
  (`fma_ftz`, `add_ftz`, `mul_ftz`; every operand here is normal or zero except an adversarial input, which the
  helpers flush as `.FTZ` does). Three approximate instructions sit on the path: `MUFU.RSQ` at every width, mapped to
  `fast::rsqrt` (invariant 5), whose last bits NVIDIA and Apple do not specify; `MUFU.RCP` of `float(n_cols)`, which
  equals the correctly rounded reciprocal for 128 and 256 (powers of two, exact) but is unspecified for the
  drafter's 2560 and 10240, whose correctly rounded reciprocals are `0x39CCCCCD` and `0x38CCCCCD`; and the gate's
  `MUFU.EX2` and `MUFU.RCP`. So "exact" for this file means: the Metal kernels basalt holds bitwise against each
  other stay bitwise against each other (`qsa_dec_fused`'s query and key post against `norm<256>`,
  `qsa_dec_fused_parity.cpp:111`, `:123`, `:125`; `qsa_dec_attn`'s merged gate against `gate`,
  `qsa_dec_attn_parity.cpp:131`), because every site uses the same helper in the same order; and the contract's
  double-precision tests hold: `qsa_parity`'s `if (!(worst <= 1e-6)) {` (`src/kernels/qsa_parity.cpp:611`) for
  `native_qsa_rms_norm_weighted` at all its on-path widths, the drafter's `norm<1024>` at `mtp.cpp:727` and `:792`
  included (`docs/design/flash-next.md:463`), and `rel < 2e-6` for the gate.
- **Stores and NaN.** Every `output` store can carry a NaN (an input NaN, or `0 · inf` once an infinite input makes
  `scale` zero), so it goes through `cuda_nan`. `rsqrt` of +inf and `exp2` of ±inf are not exercised by any basalt
  test; the port keeps `fast::rsqrt` and `fast::exp2` there and does not special-case them.
- **Per-step data.** None: the kernels read only the activations, `gamma` and the shape; `epsilon` (1e-6 on every
  path) and `n_cols` are fixed per program and may be params-record values.
- **`gate` as a kernel.** It is OFF-PATH (section 2). The port needs its arithmetic once, as the merge's gate helper
  (`qsa_dec_attn.md`); a standalone `gate` kernel is only worth porting as the bitwise reference of a
  `qsa_dec_attn_parity` port, which can call the same helper.

**Helper inventory.**

| Helper | Where it lives | Used at |
|---|---|---|
| `fma_ftz` | `kernels/common/basalt_fp.metal:73` (origin/main) | `:53`, `:65` |
| `warp_sum_ftz` (with `shfl_xor`, `add_ftz`) | `basalt_fp.metal:87` (`:85`, `:49`) | `:56`, `:63` |
| `rcp_ftz` | `basalt_fp.metal:83` | `:64`, `:75` |
| `fast::rsqrt` | MSL standard library (invariant 5's mapping for `rsqrtf`); no wrapper needed: its input is at least `epsilon` | `:65` |
| `mul_ftz` | `basalt_fp.metal:80` | `:67`, `:76` (and inside `expf_ftz`) |
| `add_ftz` | `basalt_fp.metal:49` | `:75` |
| `expf_ftz` | `basalt_fp.metal:107` | `:75`, as `expf_ftz(-raw)` |
| `cuda_nan` | `basalt_fp.metal:120` | the stores at `:67`, `:76` |

No new helper is needed for this file. The fused query post (`qsa_dec_fused.md`) and the merged gate
(`qsa_dec_attn.md`) must call these same helpers in this order.
