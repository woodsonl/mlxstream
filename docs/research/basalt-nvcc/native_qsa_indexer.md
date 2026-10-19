# basalt `native_qsa_indexer.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's native QSA indexer append (the indexer key's
RMS norm, its RoPE and the 4-cell block pooling), read from the sm_120 SASS (with the PTX where it shows what the
front end chose), and the Metal form the port writes at each site. This is the indexer append's page under invariant
5 of `docs/design/flash-next.md`; `matvec.md` is the q8_1 page and `sampler.md` (branch `woodsonl/basalt-sampler`)
the model for this page's layout and notation.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/native_qsa_indexer.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/native_qsa_indexer.cu` with basalt's local includes inlined under `#line` directives
  (`include/basalt/kernels/native_qsa_indexer.hpp`, `qsa.hpp`, `qsa_seg.hpp`, `kv_stream.hpp`,
  `qsa_decode_attn.hpp`, `qsa_dec_fused.hpp`, `rope_scaling.hpp`, `mrope.hpp`), so every PTX and SASS line names its
  basalt file and line. CUDA's own headers (`cuda_fp16.h` and the math headers) come from the compiler's
  installation. Only basalt's public MIT source was sent.
- **Flags.** `native_qsa_indexer.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:144`)
  and is one of the sixteen files `set_source_files_properties` gives `COMPILE_OPTIONS "--use_fast_math"`
  (`CMakeLists.txt:186-202`, the file at `:197`). A Release build (`:12-14`) passes CMake's `-O3 -DNDEBUG`,
  `add_compile_definitions` passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one architecture is 120 (`:3-4`,
  `:26-31`). `--use_fast_math` is `-ftz=true -prec-div=false -prec-sqrt=false -fmad=true` plus the replacement of
  the single-precision math library calls by their intrinsics (`expf` by `__expf`, `sinf`/`cosf` by
  `__sinf`/`__cosf`, `logf` by `__logf`, `powf` by `__powf`, and so on). So every float instruction this file's own
  arithmetic produces carries `.FTZ` (CUDA reads a subnormal operand as a zero of its sign and writes a subnormal
  result as zero, like this Mac's GPU), a `/` is the approximate division (`div.full.f32`, or `div.approx.f32` where
  the source writes `__fdividef`), and `sqrtf` and `1.0f / sqrtf` are `MUFU`-based approximations.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`, which returns the PTX and the `SASS (sm_120)`
  listing. Compiler id `nvcc130`: NVCC 13.0.0, version string `Cuda compilation tools, release 13.0, V13.0.48`.
  Basalt requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`).
- **userArguments**, verbatim: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"' --use_fast_math`.
  Compiler Explorer adds `-g -lineinfo --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC
  14.1); none of them changes a kernel's code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026; the request was posted again on October 18, 2026 and returned the same PTX
  (2,896 lines) and SASS (2,387 lines), instruction for instruction and source line for source line
  (`.context/qsa/scripts-nvcc-c/recheck.sh`).
- **Translation unit.** The whole file in one unit; its SASS listing is complete (no `[truncated; too many lines]`).
  It holds eleven kernels: `append<false>` (208 instructions) and `append<true>` (224), `commit_layers<false>` (240)
  and `<true>` (256), `commit_seg<false>` (264) and `<true>` (288), `append_first<false>` (136) and `<true>` (144),
  `append_blocks<false>` (208) and `<true>` (224), `append_tail` (48). The request and response are
  `.context/qsa/nvcc/native_qsa_indexer.{request.json,response_nvcc130.json}` (the source sent,
  `native_qsa_indexer_inlined.cu`); `.context/qsa/scripts-nvcc-c/funcs.py` lists or dumps a response's functions
  (with `float`, only the float instructions, each with its source line), and
  `.context/qsa/nvcc/sass/native_qsa_indexer__<kernel>.sass.txt` are the per-kernel dumps.
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.ftz.f32`), which division
  and square root it chose and which library sequence it inlined; the SASS shows what `ptxas` made of it. A row quotes
  the SASS of one copy where a site is unrolled or inlined twice; the copies were compared and differ only in
  registers.
- **Not float arithmetic.** `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ` and `IMAD.HI.U32` at a strided loop's
  head are `ptxas`'s exact integer division; `HFMA2 R, -RZ, RZ, a, b` moves a constant; `IMAD.WIDE` forms
  addresses. The tables leave them out unless they read a stored value.

## 2. Kernels in the file and their paths

`native_qsa_indexer_set_enabled` is forced on for the engine (`src/program/generate.cpp:1893`, applied at `:2400`)
and the window requires it (`src/core/layer.cpp:435`), so the table indexer `indexer_key_append` (`qsa.cu:721-734`) is
off the decode path (`.context/qsa/spec-kernels-indexer-a.md` section 1.1). Every host entry picks its kernel's
`TAB = true` instance only when `rope_table_for(scaling).cos != nullptr` (`:290`, `:311`, `:330`, `:354`, `:364`), and
`rope_table_for` returns `{}` unconditionally, "the kernels compute the angles (the stored table is not read)"
(`src/kernels/cuda/native_rope.cu:98-99`): no `TAB = true` instance is ever launched at b4c63b5. Each `<true>` instance
differs from its `<false>` twin only in the angle source (section 3.1), and this page records the `<false>` ones.

| Kernel | Lines | Host entry and launch | Path | This page |
|---|---|---|---|---|
| `append<false>` | `:108-118`, the body `append_one` `:46-107` | `native_qsa_indexer_append` (`:274-300`): `<<<1, 256>>>` (`:294-297`) | ON: the window, one launch per row (`src/core/verify.cpp:1478-1483`), and the layer forward (`layer.cpp:970-975`). Tests: `qsa_parity` (`src/kernels/qsa_parity.cpp:1428`, `:1495`), `dec_commit_parity` (`:121`), `qsa_dec_fused_parity` (`:119`) | section 3.2 |
| `commit_layers<false>` | `:120-133`, the body `append_one` | `native_qsa_indexer_commit_layers` (`:301-319`): `<<<n_layers, 256>>>` (`:315-316`) | ON: the window's DEC_COMMIT (`verify.cpp:2134`). Test: `dec_commit_parity` (`:186`, `:210`) against `n_tok` `append` calls | section 3.3 |
| `append_first<false>` | `:190-215` | `native_qsa_indexer_append_batch` (`:339-373`), `p0 == 0` only (`:353`): `<<<1, 256>>>` (`:358-359`) | OFF the decode path: the prompt path (`src/prefill/prefill.cpp:1653`). `qsa_parity`'s comparison partner for the decode append (`qsa_parity.cpp:1430-1437`) | section 3.4 |
| `append_blocks<false>` | `:216-250` | `native_qsa_indexer_append_batch`: `<<<hi - first + 1, 256>>>` (`:368-369`) | OFF: as `append_first` | section 3.4 |
| `append_tail` | `:252-259` | `native_qsa_indexer_append_batch`: `<<<R - 1 = 3, D = 128>>>` (`:370`) | OFF: as `append_first`; it only copies floats | section 3.4 |
| `commit_seg<false>` | `:136-152`, the body `append_one` | `native_qsa_indexer_commit_seg` (`:320-338`) | OFF: the `--parallel` batch window (`verify.cpp:3021`; `spec-kernels-indexer-a.md` section 1.4) | section 3.5 |
| `append<true>`, `commit_layers<true>`, `commit_seg<true>`, `append_first<true>`, `append_blocks<true>` | as their twins | the `rt.cos != nullptr` branches | never launched (above) | section 3.5 |

## 3. Per-kernel SASS tables

### 3.0 The Metal operations this page names

The file builds with `--use_fast_math`, so every scalar float instruction in the listing carries `.FTZ`: CUDA reads
a subnormal operand as zero (sign kept) and writes a subnormal result as zero, which this GPU's arithmetic also does
in `MTLMathModeSafe` except at the boundary main's helpers rebuild. The Metal column uses main's helpers in
`kernels/common/basalt_fp.metal`, with the file's `#pragma STDC FP_CONTRACT OFF` (`:43`): `ftz` (`:45`), `add_ftz`
(`:49`), `fma_ftz` (`:73`), `mul_ftz` (`:80`, which is `fma_ftz(a, b, -0.0f)`), `warp_sum_ftz` (`:87`), `h2f`
(`:110`), `f16_bits` (`:115`) and `cuda_nan` (`:120`); `fast::rsqrt` for `MUFU.RSQ` (invariant 5); and for the
angle `native_rope.md`'s `rope_cs(pos, pair, theta_scale)` and `mul_rz` (that page's section 4: `fast::log2`,
`fast::exp2`, `fast::sinpi`, `fast::cospi` for `MUFU.LG2`, `MUFU.EX2`, `MUFU.SIN`, `MUFU.COS`, "the `metal::fast::`
function of the same operation" in invariant 5's words, with the `FMUL.RZ` reduction to turns transcribed exactly).
No helper from the sampler branch is needed.

Census of the whole response (PTX and SASS of the eleven kernels; `.context/qsa/scripts-nvcc-c/funcs.py` and a
per-body count of the PTX):

- **No library routine, no slow path.** No PTX `call`, no `__nv_` or `__cudart_` symbol; no SASS `CALL`, no `FCHK`,
  no `D*` instruction. An inlined libdevice routine would leave no `call` either, so the evidence is what stands at
  each call site: under `--use_fast_math` `powf` is `__powf` (`lg2.approx.ftz`, `mul.ftz`, `ex2.approx.ftz`),
  `sinf`/`cosf` are `__sinf`/`__cosf` (`sin.approx.ftz`/`cos.approx.ftz`), the YaRN `logf` is `__logf`
  (`lg2.approx.ftz` and the multiply by ln 2), `rsqrtf` is `rsqrt.approx.ftz` and `/` is `div.approx.ftz`: single
  hardware approximations with no library code around them (no range reduction, no `.local` table, no loop), as
  `native_rope.md` section 4 finds for the same calls. Nothing in this file falls under the libdevice rule. The only
  `.f64` PTX (3 instructions) is the body of `std::isfinite(double)` (from `rope_scaling.hpp`), which
  `--keep-device-functions` emits and no kernel calls.
- **Per kernel with a RoPE** (the ten `append`, `commit_layers`, `commit_seg`, `append_first`, `append_blocks`
  instances): PTX `div.approx.ftz.f32` ×2 (the norm's `/ D` and the YaRN ramp's division), `rsqrt.approx.ftz.f32`
  ×1, `lg2.approx.ftz.f32` ×2 (`__powf` and the YaRN `logf`), `ex2.approx.ftz.f32` ×1 (`__powf`),
  `rcp.approx.ftz.f32` ×1 (the YaRN `1.0f / freq_scale`), `sin.approx.ftz.f32` and `cos.approx.ftz.f32` ×1 each.
  SASS: 8 `MUFU` per kernel (`RSQ` 1, `LG2` 2, `EX2` 1, `RCP` 2, `SIN` 1, `COS` 1), 80 in all: the norm's division by
  the constant 128 leaves no `MUFU.RCP` (section 3.1, `:84`), so both `RCP`s and one `LG2` are on the YaRN branch,
  which the path never takes.
- **The uncalled bodies.** The PTX's remaining approximate operations belong to the out-of-line copies
  `--keep-device-functions` emits: `rope_scaled_angle` (`div.approx`, `rcp.approx`, `lg2.approx`, `sin.approx`,
  `cos.approx`, one each) and `rope_yarn_ramp` (`div.approx`). With them the totals are the listing's: `div.approx`
  22, `rsqrt.approx` 10, `lg2.approx` 21, `ex2.approx` 10, `rcp.approx` 11, `sin.approx` and `cos.approx` 11 each.
- **`append_tail`** has no float arithmetic (it copies floats, section 3.4).

### 3.1 `append_one<false>` (`:46-107`), the body of `append`, `commit_layers` and `commit_seg`

`append_one` is `__forceinline__`; the SASS below is its copy in `append<false>` (addresses in that kernel, which is
nothing else, section 3.2). One CTA of 256 threads handles one indexer row: thread `d` < 128 owns dim `d`, threads
128-255 only take part in the norm's reductions. Every row converts its raw key to fp16 and back and parks it in the
layer's 3-row `tail` (slots 0-2); only the fourth row of a 4-cell block (`slot == 3`) and the spare row (`pos == 0`)
go on to pool the block: the mean of its four rows, the RMS norm with `gamma`, the RoPE of dims 0-63, and the store
of the block's key into `pooled`. Kernel arguments as SASS reads them: `raw` `c[0x380]`, `pos_dev` `c[0x388]`,
`pos_base` `c[0x390]`, `gamma` `c[0x398]`, `epsilon` `c[0x3a0]`, `tail` `c[0x3a8]`, `dead` `c[0x3b0]`, `pooled`
`c[0x3b8]`, `block_pos` `c[0x3c0]`, `max_cells` `c[0x3c8]`, `theta_scale` `c[0x3cc]`, `freq_scale` `c[0x3d0]`,
`corr_low` `c[0x3d4]`, `corr_high` `c[0x3d8]`, `ext_factor` `c[0x3dc]`, `mscale` `c[0x3e0]`, `mtab` `c[0x3e8]`.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:54-55` | `const int pos = *pos_dev, d = threadIdx.x; if (pos < 0 \|\| pos >= max_cells) return;` | `LDG.E.CONSTANT R4` (`pos_dev[0]`); `ISETP.GE.AND P0, PT, R4, UR6, PT` (`max_cells`); `ISETP.LT.OR P0, PT, R4, RZ, P0`; `@P0 EXIT` | integer; `pos` from a buffer (invariant 4) |
| `:56` | `const int slot = pos % R;` | `LOP3.LUT R11, R4, 0x3, RZ, 0xc0, !PT` (`pos` ≥ 0 here) | integer |
| `:60` | `incoming = __half2float(__float2half_rn(raw[d]));` (`d` < 128) | `LDG.E.CONSTANT R6`; `F2FP.F16.F32.PACK_AB R13, RZ, R6` (`cuda_fp16.hpp:566`, PTX `cvt.rn.f16.f32` in inline assembly, so `-ftz` does not reach it; round to nearest even, ±inf from 65520 up); `HADD2.F32 R13, -RZ, R13.H0_H0` (`cuda_fp16.hpp:708`, PTX `cvt.f32.f16`, exact) | `incoming = h2f(f16_bits(x))` (`kernels/common/basalt_fp.metal:110`, `:115`): the same rounding, overflow and fp16 subnormals; an fp32 subnormal `x` becomes the zero of its sign on both sides (it lies far below fp16's least subnormal, 2^-24) |
| `:61` | `if (slot < R - 1) tail[slot * D + d] = incoming;` | `ISETP.NE.AND P2, PT, R11, 0x3, PT`; `@P2 STG.E` (address `slot * 0x80 + d`) | `tail[slot * 128 + d] = cuda_nan(incoming)` (`basalt_fp.metal:120`): a NaN `raw` leaves `F2FP` as 0x7fff and `HADD2.F32` as 0x7fffffff (`nan.md`'s rows for both, [H]), where `half()` and `h2f` give 0x7e00 and 0x7fc00000 |
| `:63` | `if (pos != 0 && slot != R - 1) return;` | `@P1 EXIT P3` (`P1`: `slot != 3`, `P3`: `pos != 0`) | integer |
| `:70-73` | `float sum = pos == 0 ? incoming : tail[d]; for (int j = 1; j < R; ++j) sum = __fadd_rn(sum, pos == 0 \|\| j == R - 1 ? incoming : tail[j * D + d]);` | PTX `add.rn.ftz.f32` three times per path (`-ftz=true` keeps `.ftz` on `__fadd_rn`). `pos != 0`: `@P1 LDG.E R0` (`tail[d]`), `@P2 LDG.E R9, [R6.64+0x200]`, `@P2 LDG.E R10, [R6.64+0x400]` (plain loads: `tail` is written by this kernel), `@P2 FADD.FTZ R9, R9, R0`, `@P2 FADD.FTZ R10, R9, R10`, `@P2 FADD.FTZ R0, R10, R13`; `pos == 0`: `MOV R0, R13`, `@!P2 FADD.FTZ R8, R0, R13`, `@!P2 FADD.FTZ R8, R8, R13`, `@!P2 FADD.FTZ R0, R8, R13` | `sum = add_ftz(add_ftz(add_ftz(t0, t1), t2), incoming)` (`basalt_fp.metal:49`) with `t_j = tail[j * 128 + d]`, and `add_ftz(add_ftz(add_ftz(incoming, incoming), incoming), incoming)` for the spare: the three rows in slot order, the new row last |
| `:74` | `mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.` | PTX `fma.rn.ftz.f32 %f104, %f103, 0f3E800000, 0f00000000`; SASS `FFMA.FTZ R0, R0, 0.25, RZ` (`RZ` is +0) | `fma_ftz(sum, 0.25f, 0.0f)` (`basalt_fp.metal:73`), with the +0 addend: a -0 `sum` gives a +0 `mean`. Main's `mul_ftz(sum, 0.25f)` is `fma_ftz(a, b, -0.0f)` (`:80`) and would keep -0, flipping the zero sign that reaches `pooled`. The product itself is exact (a nonzero `sum` of fp16 values is at least 2^-24 in magnitude) |
| `:76-77` | `float square_sum = 0.0f; if (d < D) square_sum += mean * mean;` | PTX `fma.rn.ftz.f32 %f48, %f104, %f104, 0f00000000`, `selp.f32 %f49, %f48, 0f00000000, %p12`; SASS `FFMA.FTZ R6, R0, R0, RZ`; `FSEL R6, R6, RZ, !P3` (`P3`: `d` ≥ 128) | `d < 128 ? fma_ftz(mean, mean, 0.0f) : 0.0f` |
| `:78` (`:37-42`) | `square_sum = warp_sum(square_sum);` — `for (int offset = 16; offset; offset >>= 1) x += __shfl_xor_sync(0xffffffffu, x, offset);` | `SHFL.BFLY PT, R7, R6, 0x10, 0x1f`; `FADD.FTZ R10, R6, R7`; the same pair with `0x8`, `0x4`, `0x2`, `0x1` (PTX `add.ftz.f32`) | `warp_sum_ftz` (`basalt_fp.metal:87`): offsets 16, 8, 4, 2, 1 on the bits, `add_ftz`; every lane ends with the same bits (each step adds the same two values in either order) |
| `:79-80` | `const int lane = d % 32; if (lane == 0) partials[d / 32] = square_sum;` | `LOP3.LUT P1, R6, R2, 0x1f, RZ, 0xc0, !PT`; `@!P1 STS [R15], R18` (slot `d >> 5`) | `threadgroup float partials[8]`; slot `sg`, lane 0 (the lane mapping of `native_qsa.md` section 4: owner `t = sg * 32 + lane`) |
| `:81` | `__syncthreads();` | `BAR.SYNC.DEFER_BLOCKING 0x0` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `:82` | `square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;` | `ISETP.GT.U32.AND P2, PT, R6, 0x7, PT`; `HFMA2 R8, -RZ, RZ, 0, 0` (+0); `@!P2 LDS R8, [R9]` | the same select; slots 4-7 hold +0 (warps of `d` ≥ 128) |
| `:83` | `square_sum = warp_sum(square_sum);` | as `:78`, the second inlined copy | `warp_sum_ftz`. With lanes 8-31 at +0 the butterfly's total is `(p0 + p2) + (p1 + p3)` in slot terms (`p4`-`p7` are +0, and +0 added to a sum of squares changes no bit); the port gets it by running the same butterfly |
| `:84` | `const float scale = rsqrtf(square_sum / D + epsilon);` | PTX `div.approx.ftz.f32 %f71, %f69, 0f43000000` (`-prec-div=false`, `D` = 128), `add.ftz.f32 %f13, %f31, %f71`, `rsqrt.approx.ftz.f32 %f72, %f13`; SASS `FFMA.FTZ R9, R15, 0.0078125, R18` (`R18` = `epsilon`): `ptxas` replaced the approximate division by the constant 128 with the multiply by its exact reciprocal 2^-7 and fused it into the add, so no `MUFU.RCP` occurs (unlike `native_qsa.md`'s `norm`, whose divisor is a variable); `@!P0 MUFU.RSQ R9, R9` (`d` < 128 only) | `scale = fast::rsqrt(fma_ftz(total, 0.0078125f, epsilon))`: no `rcp_ftz` and no division. `total * 2^-7` is exact, so the fusion changes no bit against a separate multiply and add; the fma's value is at least `epsilon`, a normal |
| `:85` | `if (d < D) values[d] = scale * mean * gamma[d];` | PTX `mul.ftz.f32 %f73, %f104, %f72`, `mul.ftz.f32 %f75, %f73, %f74`; SASS `@!P0 FMUL.FTZ R0, R9, R0` (scale · mean); `@!P0 FMUL.FTZ R7, R0, R7` (`R7` = `gamma[d]`, `LDG.E.CONSTANT`); `@!P0 STS [R8], R7` | `values[d] = mul_ftz(mul_ftz(scale, mean), gamma[d])` (`basalt_fp.metal:80`), `threadgroup float values[128]` |
| `:86-87` | `__syncthreads(); if (d >= D) return;` | `BAR.SYNC.DEFER_BLOCKING 0x0`; `@P0 EXIT` | `threadgroup_barrier(mem_flags::mem_threadgroup)`; return |
| `:88-89` | `const int b = pos / R; const int rope_pos = pos == 0 ? 0 : pos_base + R * b;` | `SHF.R.U32.HI R7, RZ, 0x2, R4`; `LOP3.LUT R0, R4, 0x7ffffffc, RZ, 0xc0, !PT`; `IADD3 R0, PT, PT, R0, UR6, RZ` (`pos_base`); the `pos == 0` choice is applied where `rope_pos` is used (`:97`, `:106`) | integer |
| `:90` | `float y = values[d];` | `LDS R13, [R8]` | dims 64-127 store it unrotated |
| `:92`, `:96-97` | `pair = d % (ROT / 2)`; `theta_extrap = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair))` (`__powf` under `--use_fast_math`) | `pos == 0`: `MOV R9, RZ` (the position as +0.0, PTX `mov.f32 %f106, 0f00000000`); else `mrope.hpp:49`'s `ISETP.NE.S64.AND P0, PT, R8, RZ, PT` (`mtab` against null) and its `LDG.E.CONSTANT` of `mtab[rope_pos * 3 + pair % 3]`, then `I2FP.F32.S32 R9, R7` (exact: positions < 2^24). `I2FP.F32.U32 R6, R6` (`float(pair)`, exact); `MUFU.LG2 R13, UR6` (`theta_scale`); `FMUL.FTZ R13, R6, R13`; `MUFU.EX2 R10, R13` (PTX `lg2.approx.ftz.f32`, `mul.ftz.f32`, `ex2.approx.ftz.f32`); `FMUL.FTZ R14, R10, R9` (p · position) | `native_rope.md` section 3.1's chain, instruction for instruction: `p = ftz(fast::exp2(mul_ftz(float(pair), ftz(fast::log2(theta_scale)))))`, `theta_extrap = mul_ftz(float(pos == 0 ? 0 : rope_pos), p)`; the position is `rope_pos` (`mtab` is null without `--vision`, `rope.md` section 2) |
| `rope_scaling.hpp:153`, `:155` | `float theta = freq_scale * theta_extrap; if (ext_factor != 0.0f) {` | `FMUL.FTZ R9, R14, UR6` (`freq_scale`); `UFSETP.NEU.FTZ.AND UP0, UPT, UR7, URZ, UPT` (`ext_factor`); `BRA.U !UP0` | `mul_ftz(theta_extrap, freq_scale)` (the identity at 1, `native_rope.md` section 3.1); the branch is not taken (`ext_factor` 0) |
| `rope_scaling.hpp:137-141`, `:156-159` | the YaRN ramp, mix and `mscale` update | OFF (0x8b0-0xa00): `MUFU.RCP` and `MUFU.LG2` (`1.0f / freq_scale`, `logf`), `FMUL.FTZ` by `0.69314718246459960938`, `FFMA.FTZ R5, R5, R12, 1` (`R12` = 0.1f), `FMUL.FTZ` by `mscale`; `FADD.FTZ` ×2, `FMNMX.FTZ R5, R5, 0.0010000000474974513054, !PT`, `MUFU.RCP`, `FMUL.FTZ`; `FMNMX.NAN R10, R6, 1, PT`, `FSETP.GEU.FTZ.AND`, `FADD.FTZ R10, -R10, 1`, `FSEL`; `FMUL.FTZ` by `ext_factor`, `FADD.FTZ R6, -R10, 1`, `FMUL.FTZ`, `FFMA.FTZ R9, R14, R10, R9` | none: the same opcodes as `native_rope`'s off-contract branch |
| `rope_scaling.hpp:161-162` | `cos_out = cosf(theta) * mscale; sin_out = sinf(theta) * mscale;` (`__cosf`, `__sinf`) | `FMUL.RZ R9, R9, 0.15915493667125701904` (θ · `0x3E22F983`, toward zero, to turns); `MUFU.SIN R10, R9`; `MUFU.COS R6, R9`; `FMUL.FTZ R11, R10, R5`; `FMUL.FTZ R5, R6, R5` (`R5` = `mscale`, `c[0x3e0]`) | `rope_cs`'s turns and `fast::cospi`/`fast::sinpi` (`native_rope.md` sections 3.1 and 4), then `mul_ftz` by `mscale` (the identity at 1) |
| `:100` | `const float a = values[pair], z = values[pair + ROT / 2];` | `LDS R7, [R11]`; `LDS R8, [R11+0x80]` | two threadgroup loads |
| `:101` | `y = d < ROT / 2 ? a * c - z * s : a * s + z * c;` | `d` < 32: PTX `mul.ftz.f32` (c·a), `mul.ftz.f32` (s·z), `sub.ftz.f32`; SASS `@!P0 FMUL.FTZ R6, R11, R8` (s·z, rounded), `@!P0 FFMA.FTZ R13, R5, R7, -R6` (c·a exact: `ptxas` fused the `mul` and the `sub`). `d` in 32-63: PTX `mul.ftz.f32 %f97, %f24, %f27` (c·z), `fma.rn.ftz.f32 %f109, %f26, %f25, %f97`; SASS `@P0 FMUL.FTZ R8, R5, R8` (c·z, rounded), `@P0 FFMA.FTZ R13, R11, R7, R8` (a·s exact) | `d < 32 ? fma_ftz(c, a, -mul_ftz(s, z)) : fma_ftz(s, a, mul_ftz(c, z))`: `native_rope`'s pairing at both outputs (`native_rope.md` section 3.1, `:66-67`) |
| `:103` | `pooled[std::size_t(b) * D + d] = y;` | `STG.E desc[UR4][R8.64], R13` | `pooled[b * 128 + d] = cuda_nan(y)` |
| `:104` | `if (pos == 0) dead[d] = y;` | `@!P1 STG.E desc[UR4][R10.64], R13` | `dead[d] = cuda_nan(y)` |
| `:105` | `else pooled[std::size_t(b + 1) * D + d] = dead[d];` | `@P1 LDG.E R3`; `@P1 STG.E desc[UR4][R8.64+0x200], R3` | a bit copy of the stored spare (already sm_120's NaN) |
| `:106` | `if (d == 0 && pos != 0) *block_pos = rope_pos;` | `ISETP.NE.OR P0, PT, R2, RZ, !P0`; `@P0 EXIT`; `STG.E desc[UR4][R2.64], R0` | integer |

- **The angle and the rotation are `native_rope`'s.** Lines `:96-101` compile to the float instructions of
  `native_rope.cu`'s `apply<false>` in the same order (`MUFU.LG2`, `FMUL.FTZ`, `MUFU.EX2`, the position multiply,
  `freq_scale`, `FMUL.RZ` by `0x3E22F983`, `MUFU.SIN`, `MUFU.COS`, the two `mscale` multiplies, then the rotation
  with a·c and a·s exact), differing only in the position (0 for the spare, `rope_pos` otherwise) and in the operand
  order of the commutative `FMUL.FTZ R14, R10, R9`. So the indexer calls the one `rope_cs(pos, pair, theta_scale)`
  of `native_rope.md` section 4, as that page already plans for "the native indexer's key and query rotations".
- **Zero angles.** The spare (`pos == 0`) and a first block at `rope_pos` 0 rotate by θ = +0, which leaves the key
  unrotated exactly when cos(0) is 1.0 and sin(0) is 0 in the approximations: [H] for `MUFU.COS`/`MUFU.SIN` at +0,
  and a one-line assertion for `rope_cs`'s test on this GPU (`fast::cospi(0.0f) == 1.0f`, `fast::sinpi(0.0f) == 0.0f`).
- **Subnormals.** `incoming` is an fp16 value, so no value up to `scale * mean` is subnormal: `incoming`, `sum`
  and `mean` are zero or at least 2^-26 in magnitude (multiples of 2^-24, then a power-of-two scaling), the squares
  and their totals zero or at least 2^-52, the fma at least `epsilon`, `scale` between 1/65504 (finite fp16 inputs)
  and `rsqrt(epsilon)`, so `scale * mean` is zero or above 2^-43. A subnormal can appear only at `· gamma[d]` and
  after, with an adversarial `gamma` (a model's norm weight is normal), where `mul_ftz` and `fma_ftz` flush as
  `.FTZ` does.
- **NaN at the stores.** A NaN that reaches `tail`, `pooled` or `dead` has passed `HADD2.F32` or an `FMUL`/`FFMA`,
  so each of those stores goes through `cuda_nan`; the spare copy at `:105` moves stored bits.

### 3.2 `append<false>` (`:108-118`)

208 SASS instructions, no loop and no subroutine: the kernel is `append_one<false>` (section 3.1) plus the stack and
descriptor loads (`LDC R1, c[0x0][0x37c]`, `LDCU.64 UR4, c[0x0][0x358]`).

- **Launch.** `native_qsa_indexer_append` (`:274-300`): `append<false><<<1, THREADS = 256, 0, stream>>>` (`:295`),
  no dynamic shared memory (the static `values[128]` and `partials[32]`, 640 bytes), no cluster. The host refuses
  any geometry but `idx_dim` 128, `idx_block` 4 and `n_rot` 64, a `pos_base` that is negative or not a multiple of
  4, a `max_cells` outside `[1, INT32_MAX]`, and an `epsilon` that is not finite and positive (`:277-281`).
- **Host-side values.** `theta_scale = powf((float) scaling.freq_base, -2.0f / ROT)` (`:287`) is `native_rope.cu:118`'s
  expression at the same `n_rot` 64, computed by the host's libm: `0x3F1AB32B` for base 1e7, a settled value
  (`native_rope.md` section 4). `scaling.kernel_args(ROT)` (`:288`) gives `None`'s identity constants
  (`freq_scale` 1, `ext_factor` 0, `attn_factor` = `mscale` 1), and `rope_table_for` returns `{}`, so `<false>` runs
  (section 2).
- **On-path arguments.** Both engine calls pass `pos_base = 0` (the window, `src/core/verify.cpp:1480`; the layer
  forward, `src/core/layer.cpp:972`, with the caller's 0 from `src/core/session.cpp:293` and `:907`) and `epsilon`
  1e-6 (`EPS`, `RMS_EPS`), with
  `gamma = indexer.k_norm.weight` (`.context/qsa/spec-kernels-indexer-a.md` section 1.2's call table). So
  `rope_pos` is `4b` on the path, and none of the kernel arguments varies per step.
- **Per-step data read.** `pos_dev[0]`, the step record's `kStepPos` word (`step_ + t * kStepCount + kStepPos`), and
  the raw key row; the layer's `tail`, `dead` and `pooled` are state. Invariant 4 holds as basalt has it: the
  position is a buffer word, `pos_base`, `epsilon`, `max_cells` and the RoPE constants are per-program values.
- **Metal mapping.** One threadgroup of 256 threads per row; `t = sg * 32 + lane` owns dim `t` (section 3.1). The
  window issues one launch per row in row order (`verify.cpp:1478-1483`), and row order is part of the bits: a row's
  `tail` store is the next rows' input. A port may run the window's rows in one threadgroup in order, as
  `commit_layers` does (section 3.3), with a device-memory barrier between rows; each row is the same `append_one`.

### 3.3 `commit_layers<false>` (`:120-133`)

240 SASS instructions, no subroutine. Launch (`:315-316`): `<<<n_layers, THREADS = 256, 0, stream>>>`, one CTA per
QSA layer (at most `kQsaCommitMaxLayers` = 16, `include/basalt/kernels/native_qsa_indexer.hpp:44`), no dynamic
shared memory, no cluster. The `QsaCommitLayers` table travels by value in the parameter buffer: CTA `l` reads its
layer's pointers at `c[0x0][R0 + 0x380 ...]` with `R0 = blockIdx.x * 0x40` (64 bytes per `QsaCommitLayer`,
`native_qsa_indexer.hpp:45-52`), `a.pos` at `c[0x780]`, `a.n_tok` at `c[0x790]`; `theta_scale` and the RoPE
constants follow (`c[0x798]`-`c[0x7b0]`).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:124-125` | `const QsaCommitLayer& L = a.layer[blockIdx.x]; for (int i = threadIdx.x; i < (R - 1) * D; i += THREADS) L.tail[i] = ((const volatile float*) L.tail_snap)[i];` | `IMAD.SHL.U32 R0, R0, 0x40, RZ`; `ISETP.GT.U32.AND P0, PT, R2, 0x17f, PT`; `LDG.E.STRONG.SYS R9` (the `volatile` read of the host-mapped snapshot) and `STG.E` per word, `i` stepping by 256 below 384 (threads 0-127 copy two words, 128-255 one) | a 32-bit word copy of 384 words, no float operation; the snapshot is an ordinary device buffer on Metal |
| `:126` | `__syncthreads();` | `BAR.SYNC.DEFER_BLOCKING 0x0` | `threadgroup_barrier(mem_flags::mem_device)`: thread `d` then reads `tail[128 + d]`, which thread `128 + d` restored, so this barrier must order device memory (CUDA's `__syncthreads` orders global and shared accesses; Metal's `mem_threadgroup` alone does not) |
| `:127` | `for (int t = 0; t < a.n_tok; ++t) {` | `LDC R23, c[0x0][0x790]`; `ISETP.GE.AND P1, PT, R23, 0x1, PT`; `@!P1 EXIT`; the count kept negated (`IADD3 R23, PT, PT, -R23, RZ, RZ`, then `IADD3 R23, R23, 0x1` and `ISETP.NE.AND P1` per trip) | a loop bounded by the program's `n_tok` (below) |
| `:128-130` | `append_one<TAB>(L.raw + (size_t) t * D, a.pos + t, 0, L.gamma, a.epsilon, L.tail, L.dead, L.pooled, L.block_pos, L.max_cells, ...)` | section 3.1's float sequence, per source line the same opcodes with the same immediates (`fseq.py` on both kernels, repeated with `funcs.py`'s fuller float listing through `fcmp.py`, section 3.5: besides the hoisting below, the `ext_factor` test `UFSETP.NEU.FTZ.AND` is tagged `:97` instead of `rope_scaling.hpp:155`, and one off-path YaRN ramp subtraction is written `FADD.FTZ R, -R, R`, the same IEEE value): the conversion, the sum and its operand order, the mean `FFMA.FTZ R, R, 0.25, RZ`, the square, both butterflies, `FFMA.FTZ R, R, 0.0078125, R` and `MUFU.RSQ`, the two norm products, the angle, `FMUL.RZ`, `MUFU.SIN`/`MUFU.COS`, the rotation and the stores. The position is `LDG.E R29` of `a.pos[t]`, and `pos_base = 0` is folded (`rope_pos = pos & 0x7ffffffc`, `LOP3.LUT R33, R29, 0x7ffffffc, RZ, 0xc0, !PT`) | the same device function as `append` (section 3.1), called with `pos_base` 0 |
| (hoisted) | `float(pair)` and the YaRN constants | `ptxas` moved them ahead of the loop, tagged `:55`: `I2FP.F32.U32 R27, R25` (`float(pair)`), `MUFU.RCP`/`MUFU.LG2`/`FMUL.FTZ` by ln 2/`FFMA.FTZ R28, R8, R9, 1`/`FMUL.FTZ` by `mscale` (the YaRN `mscale`), and the ramp's `FADD.FTZ` ×2, `FMNMX.FTZ R7, R7, 0.0010000000474974513054, !PT`, `MUFU.RCP R7, R7`, `FMUL.FTZ` | none: the same values computed once; the YaRN ones are unused on the path |
| `:131` | `__syncthreads();` | `BAR.SYNC.DEFER_BLOCKING 0x0` | `threadgroup_barrier(mem_flags::mem_threadgroup)` suffices: across rows a thread reads back only its own `tail`, `dead` and `pooled` words, and `values`/`partials` are threadgroup memory |

- **Bitwise against `append`.** The header promises the commit equals "n_tok native_qsa_indexer_append calls with
  pos_base 0 would, bitwise" (`native_qsa_indexer.hpp:41-43`), and the SASS keeps that on CUDA: the row body is
  `append<false>`'s arithmetic instruction for instruction, and a `MUFU` result is a fixed function of its input.
  The Metal port keeps it by construction, one device function for both kernels.
- **Per-step data.** `a.pos[t]` is the commit buffer's position words, `commit_ + 2` (`src/core/verify.cpp:2118`),
  which the host fills per step as `t < n_keep ? last_pos0_ + t : -1` (`verify.cpp:2737`); a row at -1 returns at
  `:55`. `a.n_tok` is `MT = max_t_` (`verify.cpp:2095`, `:2118`), the program's window width, not the step's
  count. So invariant 4 holds as basalt has it: the loop runs the program's width and the step's rows are chosen by
  the buffer, which a Metal port keeps (`n_tok` a params-record value, the loop bounded by it).
- **Tail snapshot.** `tail_snap` is host-mapped memory read with `volatile` (`LDG.E.STRONG.SYS`); the port reads
  the snapshot buffer the window wrote, a copy of bits.

### 3.4 Off the decode path: `append_first<false>` (`:190-215`), `append_blocks<false>` (`:216-250`), `append_tail` (`:252-259`)

The batched append of the prompt path, `native_qsa_indexer_append_batch` (`:339-373`), launches up to three
kernels in stream order: `append_first<false><<<1, THREADS = 256>>>` when `p0 == 0` (`:353-359`), then
`append_blocks<false><<<hi - first + 1, 256>>>`, one CTA per completed block (the blocks whose last cell `4b + 3`
lies in `[p0, p0 + n)`, `:361-369`), then `append_tail<<<R - 1 = 3, D = 128>>>` (`:370`); no dynamic shared memory,
no cluster. The host's checks are `:343-346` (the append's geometry and epsilon, plus `p0 + n <= max_cells` and
`pos_base + max_cells <= INT32_MAX`), and `theta_scale` is `:347`, the expression of `:287`. The per-kernel SASS is in
`.context/qsa/nvcc/sass/native_qsa_indexer__{append_first_false,append_blocks_false,append_tail}.sass.txt`.
Recorded for completeness and for `qsa_parity`, which compares the decode append against this path (section 4.2);
no Metal kernel of the decode path needs them.

| Kernel | Line | basalt | SASS | Metal, if ported |
|---|---|---|---|---|
| `append_first<false>` (136 instructions; `raw` `c[0x380]`, `gamma` `c[0x388]`, `epsilon` `c[0x390]`, `dead` `c[0x398]`, `pooled` `c[0x3a0]`, `theta_scale` `c[0x3a8]`, `freq_scale` `c[0x3ac]`, `ext_factor` `c[0x3b8]`, `mscale` `c[0x3bc]`, `mtab` `c[0x3c0]`) | `:200` | `incoming = __half2float(__float2half_rn(raw[d]))` (`d` < 128) | `@!P0 LDG.E.CONSTANT R4`; `@!P0 F2FP.F16.F32.PACK_AB R0, RZ, R4`; `@!P0 HADD2.F32 R0, -RZ, R0.H0_H0` | `h2f(f16_bits(x))`, section 3.1 `:60` |
| | `:201-203` | `sum = incoming; for (j = 1; j < R; ++j) sum = __fadd_rn(sum, incoming);` | `@!P0 FADD.FTZ R3, R0, R0`; `@!P0 FADD.FTZ R7, R0, R3`; `@!P0 FADD.FTZ R0, R0, R7` | `add_ftz(add_ftz(add_ftz(incoming, incoming), incoming), incoming)`: section 3.1's spare sum (only the operand order of the commutative adds differs) |
| | `:198`, `:204` | `float mean = 0.0f; ... mean = __fmaf_rn(0.25f, sum, 0.0f);` | `HFMA2 R3, -RZ, RZ, 0, 0` (+0 for `d` ≥ 128); `@!P0 FFMA.FTZ R3, R0, 0.25, RZ` | `fma_ftz(sum, 0.25f, 0.0f)`, section 3.1 `:74` |
| | `:180` (`norm_scale`, `:179-188`) | `float square_sum = d < D ? mean * mean : 0.0f;` | `FMUL.FTZ R0, R3, R3`; `FSEL R0, R0, RZ, !P1` (`P1`: `d` ≥ 128) | a plain product where `append_one` has `FFMA.FTZ R6, R0, R0, RZ` (`:77`'s `0.0f + mean * mean`, fused): the same bits for a square (below), so `fma_ftz(mean, mean, 0.0f)` of section 3.1 serves |
| | `:181`, `:186` (`:37-42`) | `warp_sum` twice, around `partials` | `SHFL.BFLY` `0x10`, `0x8`, `0x4`, `0x2`, `0x1`, each followed by `FADD.FTZ` (own + other in the first copy, other + own in the second); `@!P1 STS` (lane 0, slot `d >> 5`); `BAR.SYNC.DEFER_BLOCKING 0x0`; `@!P2 LDS` (lane < 8), else `MOV R6, RZ` | `warp_sum_ftz` twice and the 8-slot `partials`, section 3.1 `:78-83` |
| | `:207` | `scale = rsqrtf(square_sum / D + epsilon);` | `LDC R14, c[0x0][0x390]`; `FFMA.FTZ R7, R7, 0.0078125, R14`; `@!P0 MUFU.RSQ R6, R7` | `fast::rsqrt(fma_ftz(total, 0.0078125f, epsilon))`, section 3.1 `:84` |
| | `:208` | `values[d] = scale * mean * gamma[d];` | `@!P0 FMUL.FTZ R6, R6, R3` (scale · mean); `@!P0 FMUL.FTZ R13, R6, R13` (`gamma[d]`, `LDG.E.CONSTANT`); `@!P0 STS` | `mul_ftz(mul_ftz(scale, mean), gamma[d])` |
| | `:163-175` (`pooled_value<false>`, `zero_pos` true) | `y = values[d]`; for `d` < 64 the angle at position `0` and the rotation | `LDS R9`; `I2FP.F32.U32 R5, R4` (`float(pair)`); `MUFU.LG2 R6, UR6`; `FMUL.FTZ R6, R5, R6`; `MUFU.EX2 R6, R6`; `FMUL.FTZ R4, RZ, R6` (the position: the `int` 0 became the register +0, and the multiply is kept); `FMUL.FTZ R7, R4, UR7` (`freq_scale`); `UFSETP.NEU.FTZ.AND UP0, UPT, UR6, URZ, UPT` (`ext_factor`) and the YaRN block 0x530-0x670 (section 3.1's OFF row, the same opcodes); `FMUL.RZ R7, R7, 0.15915493667125701904`; `MUFU.SIN R6, R7`; `MUFU.COS R4, R7`; `FMUL.FTZ R6, R6, R3`; `FMUL.FTZ R4, R4, R3` (`R3` = `mscale`, `MOV R3, UR8`); `@!P0 FMUL.FTZ R3, R6, R9` (s·z), `@!P0 FFMA.FTZ R9, R4, R0, -R3` (c·a − s·z); `@P0 FMUL.FTZ R5, R4, R9` (c·z), `@P0 FFMA.FTZ R9, R6, R0, R5` (s·a + c·z) | section 3.1's `:90-101` at position 0: `rope_cs(0, pair, theta_scale)` and the same rotation |
| | `:213-214` | `pooled[d] = y; dead[d] = y;` | `STG.E desc[UR4][R4.64], R9` (`pooled`, `c[0x3a0]`); `STG.E desc[UR4][R2.64], R9` (`dead`, `c[0x398]`) | `cuda_nan(y)` at both |
| `append_blocks<false>` (208 instructions; `raw` `c[0x380]`, `n` `c[0x388]`, `p0` `c[0x390]`, `pos_base` `c[0x398]`, `gamma` `c[0x3a0]`, `epsilon` `c[0x3a8]`, `tail` `c[0x3b0]`, `dead` `c[0x3b8]`, `pooled` `c[0x3c0]`, `block_pos` `c[0x3c8]`, `first_block` `c[0x3d0]`, `last_block` `c[0x3d8]`, `theta_scale` `c[0x3e0]`, `freq_scale` `c[0x3e4]`, `ext_factor` `c[0x3f0]`, `mscale` `c[0x3f4]`, `mtab` `c[0x3f8]`) | `:223` | `const int64_t b = first_block + blockIdx.x;` | `S2UR UR4, SR_CTAID.X`; `IADD.64 R2, R2, UR4` | integer |
| | `:229-232` | `key(j) = cell >= p0 ? __half2float(__float2half_rn(raw[(cell - p0) * D + d])) : tail[j * D + d]` with `cell = b * R + j` | per `j` an `ISETP.GE.S64` of `4b + j` against `p0`; `@P LDG.E.CONSTANT` of `raw` (the four rows at `+0x200` steps from `(4b - p0) * 128 + d`), `F2FP.F16.F32.PACK_AB`, `HADD2.F32`; `@!P LDG.E.CONSTANT` of `tail[j * 128 + d]`, no conversion (the tail holds converted keys) | `cell >= p0 ? h2f(f16_bits(raw[...])) : tail[j * 128 + d]` |
| | `:233-235` | `sum = key(0); for (j = 1; j < R; ++j) sum = __fadd_rn(sum, key(j));` | `FADD.FTZ R11, R11, R0` (k1 + k0); `FADD.FTZ R11, R11, R10` (+ k2); `FADD.FTZ R0, R11, R12` (+ k3) | `add_ftz(add_ftz(add_ftz(k0, k1), k2), k3)`: section 3.1's `:70-73` for `pos != 0` |
| | `:227`, `:236` | `float mean = 0.0f; ... mean = __fmaf_rn(0.25f, sum, 0.0f);` | `HFMA2 R0, -RZ, RZ, 0, 0`; `FFMA.FTZ R0, R0, 0.25, RZ` | `fma_ftz(sum, 0.25f, 0.0f)` |
| | `:238-240` | `norm_scale`, `scale`, `values[d]` | `FMUL.FTZ R6, R0, R0`, `FSEL R6, R6, RZ, !P1`; the two butterflies as in `append_first`; `LDC R17, c[0x0][0x3a8]`, `FFMA.FTZ R9, R8, 0.0078125, R17`, `@!P0 MUFU.RSQ R9, R9`; `@!P0 FMUL.FTZ R0, R9, R0`, `@!P0 FMUL.FTZ R7, R0, R7` | as `append_first` |
| | `:243` | `const int rope_pos = pos_base + R * (int) b;` | `IMAD R0, R2, 0x4, R7` (`R7` = `pos_base`); no overflow, `pos_base + max_cells <= INT32_MAX` (`:344`) | integer |
| | `:163-175` (`zero_pos` false) | the angle at `mrope_pos(mtab, rope_pos, pair)` and the rotation | `mrope.hpp:49`'s null test of `mtab` and its load, else `MOV R13, R0` (`rope_pos`); `MUFU.LG2 R14, UR4`; `I2FP.F32.U32 R9, R6`; `FMUL.FTZ R14, R9, R14`; `MUFU.EX2 R15, R14`; `I2FP.F32.S32 R6, R13`; `FMUL.FTZ R16, R6, R15` (position · p); `FMUL.FTZ R6, R16, UR5`; the `ext_factor` test and the YaRN block 0x880-0x9c0; `FMUL.RZ R6, R6, 0.15915493667125701904`; `MUFU.SIN R11, R6`; `MUFU.COS R9, R6`; `FMUL.FTZ R11, R11, R4`; `FMUL.FTZ R9, R9, R4` (`R4` = `mscale`, `MOV R4, UR8`); `@!P0 FMUL.FTZ R4, R11, R8`, `@!P0 FFMA.FTZ R9, R9, R7, -R4`; `@P0 FMUL.FTZ R8, R9, R8`, `@P0 FFMA.FTZ R9, R11, R7, R8` | `rope_cs(rope_pos, pair, theta_scale)` and section 3.1's rotation |
| | `:244` | `pooled[b * D + d] = pooled_value<false>(...)` | `STG.E desc[UR6][R6.64], R9` | `cuda_nan(y)` |
| | `:246-248` | `if (b == last_block) { pooled[(b + 1) * D + d] = dead[d]; if (d == 0) *block_pos = rope_pos; }` | `ISETP.NE.S64.AND P0, PT, R2, UR8, PT`; `@P0 EXIT`; `LDG.E.CONSTANT R3` (`dead[d]`); `STG.E desc[UR6][R4.64+0x200], R3`; `STG.E desc[UR6][R2.64], R0` (`d` == 0) | a bit copy; integer |
| `append_tail` (48 instructions; `raw` `c[0x380]`, `n` `c[0x388]`, `p0` `c[0x390]`, `tail` `c[0x398]`) | `:253-257` | `s = blockIdx.x`; `last = p0 + n - 1; cell = last - ((last % R) - s + R) % R; if (cell < p0) return;` | 64-bit integer code (`IADD.64`, `SHF.R.S32.HI`, `LEA.HI`, `LOP3.LUT`, `ISETP.GE.S64`); the `d >= D` test (`:254`) stays (`ISETP.GT.U32.AND P0, PT, R8, 0x7f, PT`, `@P0 EXIT`), never true at 128 threads | integer |
| | `:258` | `tail[s * D + d] = __half2float(__float2half_rn(raw[(cell - p0) * D + d]));` | `LDG.E.CONSTANT R4`; `F2FP.F16.F32.PACK_AB R0, RZ, R4`; `HADD2.F32 R7, -RZ, R0.H0_H0`; `STG.E` | `cuda_nan(h2f(f16_bits(x)))`, section 3.1 `:60-61` |

- **The square, `FMUL.FTZ` against `FFMA.FTZ x, x, RZ`.** `norm_scale` writes `d < D ? mean * mean : 0.0f`, a
  product `ptxas` leaves unfused; `append_one` writes `square_sum = 0.0f; square_sum += mean * mean`, fused with the
  +0 addend. Both round the same exact product once; a square is +0 or positive, so the +0 addend changes no bit and
  no -0 arises; and no square here is subnormal (section 3.1, "Subnormals"), so neither flush acts. A NaN `mean`
  (a NaN raw key) gives a NaN square either way, and the NaN then reaches every dim and every store through
  `cuda_nan`. So one Metal expression serves the three kernels: `fma_ftz(mean, mean, 0.0f)`, or main's
  `mul_ftz(mean, mean)` (`fma_ftz(a, b, -0.0f)`, `basalt_fp.metal:80`), which gives the same bits for a square
  (+0 plus -0 is +0 in round to nearest).
- **Bitwise the per-row append.** The comment at `:153-157` promises "the single append's arithmetic", and the SASS
  keeps it on CUDA: per source line the opcodes and immediates of section 3.1 (the conversion; the sum, k0 to k3 in
  cell order or the spare's four `incoming`; `FFMA.FTZ R, R, 0.25, RZ`; the norm; `FFMA.FTZ R, R, 0.0078125, epsilon`;
  `MUFU.RSQ`; the two norm products; the angle chain; `FMUL.RZ`; `MUFU.SIN`/`MUFU.COS`; the `mscale` products; the
  rotation with c·a and a·s exact), differing only in the square above and in the operand order of commutative
  `FADD.FTZ`/`FMUL.FTZ`. The positions agree: the spare at +0 (`append_one`'s `MOV R9, RZ`, `append_first`'s `RZ`
  operand), block `b` at `pos_base + 4b`, `append_one`'s `rope_pos` (`:89`).
- **The batch's final state is the per-row path's.** For rows `p0 .. p0 + n - 1` in order, `append_one` leaves
  `tail[s]` holding the last row with `pos % 4 == s` (`s` < 3), `pooled[b]` the key of every block it completes,
  `pooled[b + 1]` the spare copy after the last one, `dead` the spare and `*block_pos` the last block's `rope_pos`
  (no write for the spare, `:106`). The batch writes exactly these: `append_first` the spare (`pooled[0]`, `dead`),
  `append_blocks` every completed block and, for the last only, the spare copy and `block_pos`, `append_tail` each
  slot's last cell (`:256`). The intermediate spare copies the per-row path writes are overwritten there by the next
  block's key, which the batch writes directly. The order across launches is part of this: `append_blocks` reads the
  previous batch's `tail` before `append_tail` replaces it, and the last block reads `dead` after `append_first`
  wrote it.
- **Metal, if ported.** The prompt path is outside v1, and the Metal engine feeds a prompt through the program in
  chunks (`CLAUDE.md`), whose rows take the per-row append; a port of the batch keeps one device function for the
  pooled key of a block (the key selection aside) and three dispatches separated by barriers in this order. `p0`,
  `n`, `first` and `hi` are host values in kernel arguments here; a port inside the program would read them from
  the step's buffers (invariant 4).

### 3.5 `commit_seg<false>` and the `TAB = true` instances

Both OFF the decode path (section 2). The comparisons below take each kernel's float instructions from
`.context/qsa/scripts-nvcc-c/funcs.py <response> SASS <kernel> float`, which includes the uniform-datapath and half
instructions (`HADD2.F32`, `UFSETP`), and compare them per source line, and again ignoring the line, as multisets with
registers, predicates and `.reuse` stripped and `HFMA2` constant moves dropped (`.context/qsa/scripts-nvcc-c/fcmp.py`).
Ignoring the line, the only differences in any pair below are the sum's shape and one off-path YaRN ramp subtraction
written `FADD.FTZ R, -R, R` where its twin writes `FADD.FTZ R, R, -R` (the same IEEE operation).

- **`commit_seg<false>`** (264 instructions; `:136-152`). Launch `native_qsa_indexer_commit_seg` (`:320-338`):
  `<<<dim3(n_layers, n_seg), THREADS = 256, 0, stream>>>` (`:329`, `:334`), no dynamic shared memory, no cluster;
  the host refuses more than `kQsaCommitMaxLayers` layers, any other geometry, a non-finite or non-positive
  `epsilon` and an invalid scaling (`:322-324`), and computes `theta_scale` with `:287`'s expression (`:325`).
  CTA (`l`, `j`) skips an unused table entry (`sg.slot < 0`, `:142`: `@!P0 EXIT` on the sign of the `LDG.E` of
  `a.segs[j].slot`), restores the slot's `tail` from its snapshot (`:144`: `LD.E.STRONG.SYS` and `ST.E` per word,
  generic loads because the slot's pointers come from the `QsaSegSlot` table in device memory), `BAR.SYNC`, then
  appends `sg.len` rows in order, each `append_one<false>` with `pos_base` 0 at `pos[slot * pos_stride + t]`
  (`:146-151`). Against `append<false>` its float instructions per source line are the same with the same
  immediates, but for three placements that change no bit (the `ext_factor` test `UFSETP.NEU.FTZ.AND` is tagged
  `:100` instead of `rope_scaling.hpp:155`, and the two below):
  - `ptxas` hoisted `float(pair)` (`I2FP.F32.U32`) and the YaRN constants (`MUFU.RCP`, `MUFU.LG2`, `FMUL.FTZ` by ln 2,
    `FFMA.FTZ R27, R14, R27, 1`, the `mscale` multiply, the ramp's `FADD.FTZ` ×2 and `FMNMX.FTZ`) ahead of the row
    loop, tagged `:55`, as in `commit_layers` (section 3.3);
  - the sum (`:70-73`): `append<false>` runs two predicated chains of three `FADD.FTZ`, one per `pos == 0` case
    (section 3.1); `commit_seg<false>` selects the operands first (`MOV R0, R29`, `@P4 LD.E R0` for `tail[d]`,
    the same for `tail[128 + d]` and `tail[256 + d]`, `R29` = `incoming`) and runs one chain, `FADD.FTZ R31, R31,
    R0` (t1 + t0), `FADD.FTZ R0, R31, R30` (+ t2), `FADD.FTZ R0, R0, R29` (+ incoming): the same operands in the
    same order.

  So the segment commit is the row append's arithmetic, as the window commit is; the Metal port, if the batch window
  is ever in scope, calls the same device function per (layer, segment) in row order.
- **The `TAB = true` instances.** `append<true>` (224 instructions), `append_first<true>` (144) and
  `append_blocks<true>` (224) have exactly their `<false>` twins' float instructions per source line;
  `commit_layers<true>` (256) and `commit_seg<true>` (288) differ only in placement: `ptxas` hoisted
  `p = powf(theta_scale, float(pair))` (`MUFU.LG2`, `FMUL.FTZ`, `MUFU.EX2`) ahead of the row loop (tagged `:55`)
  where the `<false>` instance computes it per row at `:97` (in `commit_seg<true>` also the YaRN ramp's `MUFU.RCP`
  and its multiply, with the `ext_factor` test tagged `:97` instead of `:100`), and `commit_seg<true>` writes the
  one ramp subtraction of the method note above as `FADD.FTZ R, -R, R`. The table adds no float instruction:
  `rope_tab_cs` (`include/basalt/kernels/mrope.hpp:53-58`) is the bounds test (`ISETP.GE.AND` against `t.max_pos`,
  `ISETP.LT.OR` for `p < 0`, `ISETP.EQ.S64.OR` for a null `t.cos`) and two `__ldg` loads of `cos` and `sin` at
  `p * 32 + pair`, then a branch past the angle; the loaded `c` and `s` enter the rotation as they are ("scaling
  and magnitude included", `mrope.hpp:51`), with no `mscale` multiply. The instance counts (section 1, alignment
  `NOP`s included) differ by 8 to 24: that branch and its argument loads. No `<true>` instance is launched at
  b4c63b5 (`rope_table_for` returns `{}`, `src/kernels/cuda/native_rope.cu:98-99`), so the port has nothing to
  transcribe here.

## 4. Metal transcription rules for this file

### 4.1 Helpers by kernel

Every ported source starts with `#pragma STDC FP_CONTRACT OFF` (main's `kernels/common/basalt_fp.metal:43`) and writes
the fma of `:74`, `:77`, `:84` and `:101` explicitly. "Main" is `origin/main` at e7eb8b4 (its `kernels/` are
unchanged at 50a6152); `rope_cs` and `mul_rz` are on no branch yet (`git grep` of main and
`origin/woodsonl/basalt-sampler`).

| Kernel | Path | Float helpers | Where they live | Subnormals reachable through |
|---|---|---|---|---|
| `append<false>` (3.1, 3.2) | ON: the window's per-row append and the layer forward | `h2f(f16_bits(x))` (the fp16 round trip); `add_ftz` (the sum); `fma_ftz` (the mean with its +0 addend, the square, `total · 2^-7 + epsilon`, the rotation); `warp_sum_ftz` (both butterflies); `fast::rsqrt`; `mul_ftz` (the two norm products, the rotation's rounded products); `rope_cs(pos, pair, theta_scale)` (the angle); `cuda_nan` (the `tail`, `pooled` and `dead` stores) | main `basalt_fp.metal`: `h2f` `:110`, `f16_bits` `:115`, `add_ftz` `:49`, `fma_ftz` `:73`, `mul_ftz` `:80`, `warp_sum_ftz` `:87`, `cuda_nan` `:120` (and `ftz` `:45` inside `rope_cs`); `fast::rsqrt` the MSL library; `rope_cs` and its `mul_rz` new, specified in `native_rope.md` section 4 | `· gamma[d]` and after, with an adversarial `gamma`, and the rotation's operands; no earlier value can be subnormal (section 3.1, "Subnormals") |
| `commit_layers<false>` (3.3) | ON: the window's DEC_COMMIT | the same device function as `append`, called per row with `pos_base` 0; the snapshot restore is a word copy | as `append` | as `append` |
| `commit_seg<false>` (3.5) | OFF: the `--parallel` batch window | as `commit_layers` | as `append` | as `append` |
| `append_first<false>`, `append_blocks<false>` (3.4) | OFF: the prompt path | as `append` (the square through `fma_ftz(mean, mean, 0.0f)` or `mul_ftz(mean, mean)`, the same bits, section 3.4) | as `append` | as `append` |
| `append_tail` (3.4) | OFF: the prompt path | `h2f(f16_bits(x))`, `cuda_nan` | main `basalt_fp.metal:110`, `:115`, `:120` | none (the conversion keeps fp16 subnormals, and an fp32 subnormal input becomes a zero of its sign on both GPUs) |
| the five `<true>` instances (3.5) | never launched at b4c63b5 | none | - | - |

No kernel of this file needs a helper from the sampler branch, an `ieee_*` helper (every float instruction carries
`.FTZ`, section 3.0), `rcp_ftz` (the norm's division by 128 is a multiply by 2^-7, section 3.1 `:84`), an exponential
helper or binary64. The one device function of `append_one` serves `append` and both commits, which is what keeps
`dec_commit_parity`'s bitwise comparison (section 4.2) true on Metal by construction.

### 4.2 What basalt's tests demand

The checks are transcribed in full in `.context/qsa/spec-tests-a.md` (P2, P10, P11), `spec-tests-d.md` (section 0,
V8) and `spec-repo.md` section 5.9; this section records what the SASS above means for each.

- **The commit, bitwise against the per-row append.** `dec_commit_parity` runs `native_qsa_indexer_commit_layers`
  (`src/kernels/dec_commit_parity.cpp:186`) against `qsa_ref`, the snapshot copied into `tail` and then `MT = 7`
  `native_qsa_indexer_append` calls with `pos_base` 0 (`:114-124`), on 3 layers whose `raw`, `tail_snap`, `gamma`,
  `tail`, `dead` and `pooled` are N(0, 1) floats (`dev_rand(rng, n, 1.0f)`, `:42-50`, `:103-108`; `CELLS` 4096,
  `EPS` 1e-6, `:29-31`), for `n_keep` 0 to 7 and `p0` in {0, 1, 5, 6, 1001} (`:163-164`), the positions
  `t < n_keep ? p0 + t : -1` (`:148-152`). The state is compared as bytes (`qsa_snapshot`, `:126-135`: `tail`,
  `dead`, `pooled`, `block_pos` per layer):
  `if (qa != qb) { std::printf("  QSA n_keep=%d p0=%d: indexer buffers differ\n", n_keep, p0); ok = false; }`
  (`:189`), and the commit must change something:
  `if (qa == q0 && n_keep > 0) { std::printf("  QSA n_keep=%d p0=%d: no change (no power)\n", n_keep, p0); ok = false; }`
  (`:190`). On CUDA the first holds by construction (section 3.3: one row body, instruction for instruction). The
  Metal test runs the same fixture with the Metal commit against the snapshot restore and seven Metal row appends,
  the state compared as `uint32` words (`spec-repo.md` section 5.9); it holds on Metal by construction when both
  kernels call one `append_one` device function (section 4.1). The commit's device-memory barrier after the restore
  is required (section 3.3, `:126`), but a missing one is a race this test need not catch.
- **The batch, bitwise against the per-row append.** `qsa_parity`'s P10 appends 8 cells at `pos_base` 100 one by one
  (`src/kernels/qsa_parity.cpp:1424-1429`, the decode append) and as one batch (`:1430`), then requires every word
  equal (`:1437-1440`, `:1444`):
  `for (size_t i = 0; i < pa.size(); ++i) bad += std::memcmp(&pa[i], &pb[i], 4) != 0;` (and the same for `dead`,
  `tail` and `block_pos`), then `if (bad) ++g_bad;`. On CUDA it holds by construction (section 3.4: the same
  opcodes per source line and the same final state). The batch is the prompt path and is not ported; the Metal test
  is `spec-tests-a.md` P10's port plan: the Metal append op at T = 8 against eight T = 1 dispatches, positions 0 to
  7, the scaling `none` only, every pooled row, `dead`, `tail` and `block_pos` as `uint32` words. The `linear 2`
  and `yarn 2` variants and QP-10.2 are off the path (the YaRN branch of section 3.1 is never taken).
- **The spare's factor.** P11's QP-11.4 requires
  `require("  <v>: the native spare carries the same row-0 factor", nworst <= 4e-7, ...)` (`qsa_parity.cpp:1538-1539`)
  of the native append at position 0 (`:1495`); under `none` its reference is the same kernel's output times 1.0f,
  so it is vacuous on the path (`spec-tests-a.md` P11). What the SASS adds: on CUDA the spare's rotated dims are the
  normed key exactly when `MUFU.COS(+0)` is 1.0 and `MUFU.SIN(+0)` is +0 ([H], section 3.1, "Zero angles"); on Metal
  the same holds when `rope_cs(0, pair, theta_scale)` returns exactly (1, +0), which a one-line assertion in
  `rope_cs`'s test settles (`fast::cospi(0.0f) == 1.0f`, `fast::sinpi(0.0f) == 0.0f`). `spec-tests-a.md` P11 moves
  the on-path content into P2's port test (i), with two bitwise assertions on the native spare (`pooled[0]` equal to
  `dead` after cell 0, `:103-104`; `pooled[b + 1]` equal to `dead` after each completed block, `:105`).
- **The pooled keys against double (the contract's new test).** `docs/design/flash-next.md:464` (at e7eb8b4) gives
  the native append "the pooled keys against a double-precision reference with `rope_parity`'s bound for the native
  rotation: `if (!(rel <= 3e-3)) ++agree_bad;` (`src/kernels/rope_parity.cpp:400`)"; `spec-tests-a.md` QP-2.3 (i)
  is its port, `test_basalt_qsa_indexer_append_pooled_vs_double` (qsa-indexer). The record shows why a tolerance is
  the only possible CUDA-side criterion: the norm's `MUFU.RSQ` and the angle's `MUFU.LG2`, `MUFU.EX2`, `MUFU.SIN` and
  `MUFU.COS` are approximations whose bits the Metal `fast::` functions do not reproduce (section 4.3), while every
  other operation is exact transcription. The reference must take the fp16-rounded raw keys (the kernel rounds its
  input first, section 3.1 `:60`), so the bound measures only the fp32 arithmetic and the approximations;
  `native_rope.md` section 4 measured `rope_cs` within 1.7e-7 (`|dc| + |ds|`) of a binary64 model of CUDA's angle
  chain at every path position, far inside 3e-3.
- **`qsa_dec_fused_parity`** calls the native append only on its reference side
  (`src/kernels/qsa_dec_fused_parity.cpp:118-120`), for the int8-KV `qsa_dec_kv_post` checks, which are outside
  the contract (`spec-tests-d.md` section 0 and its F8-F12 row); nothing there checks this file's output on the
  fp16 path.
- **What no criterion sees.** The two bitwise checks compare the port with itself, and the double test sits orders of
  magnitude above an fp32 rounding, so a sequence error common to every path (a different sum order, the mean's
  zero addend, the fused 2^-7 multiply-add, the rotation's pairing) would pass every test; the transcription rules of
  this page carry it. No fixture holds a NaN, an infinity or an fp16 overflow (N(0, 1) and `gauss · 4^(t % 4)` keys),
  so nothing checks the `cuda_nan` stores against CUDA either. The `tail` rows are the one output a host can compute
  exactly (numpy's fp16 round trip of the raw keys, NaN as `0x7fffffff`); a bitwise host check of them is a line in
  the double test (section 4.3).

### 4.3 Points for the contract and the kernel lane

- **No library function.** No libdevice routine is inlined into any kernel of the file (section 3.0): `__powf`,
  `__sinf`, `__cosf`, `__logf`, `rsqrtf` and `/` are single hardware approximations, so the LIBRARY FUNCTIONS rule is
  not triggered, and the only host libm value, `theta_scale` (`:287`), is `native_rope.md` section 4's settled
  `0x3F1AB32B`.
- **The approximate operations are invariant 5's general clause.** "For the other approximate operations basalt
  computes, whether a `--use_fast_math` file produces them or an IEEE-built kernel writes them (the list above), the
  mapping is the `metal::fast::` function of the same operation" (`docs/design/flash-next.md:133-136` at e7eb8b4),
  which names `fast::exp2` and `fast::rsqrt` and covers the rest: `MUFU.RSQ` is `fast::rsqrt`, `MUFU.LG2` and
  `MUFU.EX2` are `fast::log2` and `fast::exp2`, and `MUFU.SIN`/`MUFU.COS` after the exact `FMUL.RZ` reduction to
  turns are `fast::sinpi`/`fast::cospi` of twice the reduced turns (`f = t - rint(t)`, then `fast::cospi(2f)`,
  `fast::sinpi(2f)`, as `rope_cs` computes them), all inside `native_rope.md`'s one `rope_cs`. Open,
  for the contract: its list of where the Metal result can differ from CUDA's last bits (`flash-next.md:139-141`:
  `MUFU.RCP`, "`ex2.approx`, `rsqrtf`", `log1pf`, the library `expf`) names neither `lg2.approx` nor
  `sin.approx`/`cos.approx`. Recommendation: add the RoPE angle's `lg2.approx`, `sin.approx` and `cos.approx` (this
  file and `native_rope.cu`) to that sentence, with the tolerances that accept them (`rope_parity.cpp:400`'s 3e-3 and
  the pooled keys' double test, section 4.2).
- **Open, [H]: the zero angle.** The spare and a block at `rope_pos` 0 are unrotated only if `MUFU.COS(+0)` is
  exactly 1.0 and `MUFU.SIN(+0)` exactly +0 on sm_120 (no sm_120 here). Recommendation: the Metal side is settled by
  a one-line assertion in `rope_cs`'s test (`fast::cospi(0.0f) == 1.0f`, `fast::sinpi(0.0f) == +0.0f`, as bits); the
  CUDA side stays [H], which no basalt criterion can expose (section 4.2, the spare's factor).
- **Open, [H]: the NaN of the fp16 round trip.** A NaN raw key leaves `F2FP.F16.F32.PACK_AB` as 0x7fff and
  `HADD2.F32` as 0x7fffffff (`nan.md:59`, [H]); `cuda_nan` at the `tail`, `pooled` and `dead` stores gives the same
  0x7fffffff. Recommendation: keep `nan.md`'s hypothesis; no fixture reaches it (section 4.2).
- **For the kernel lane: three places a plausible transcription loses a bit.** The mean is
  `fma_ftz(sum, 0.25f, 0.0f)` with the +0 addend, not main's `mul_ftz` (whose -0 addend keeps a -0 mean, section 3.1
  `:74`); the norm's `/ D + epsilon` is one `fma_ftz(total, 0.0078125f, epsilon)` with no reciprocal (section 3.1
  `:84`); and the commit's barrier after the snapshot restore must order device memory (section 3.3 `:126`).
- **One cheap exact check.** The `tail` rows are the only output a host computes exactly (numpy's fp16 round trip of
  the raw keys, a NaN as `0x7fffffff`). Recommendation: assert them as `uint32` words in the pooled keys' double
  test (one line); no other criterion compares this kernel with an independent exact result.
- **Invariants 1 and 4 hold as basalt has them.** A row's outputs depend only on its raw key, its position (a buffer
  word) and the state the earlier rows left, through one device function at every window width; row order is part of
  the bits (a row's `tail` store is the next row's input), so the window's rows run in order (section 3.2). The commit
  loop is bounded by the program's width `n_tok`, and a row at position -1 returns (section 3.3).
- **Order.** Fixed per row: the four cells in slot order with the new row last, the butterfly 16, 8, 4, 2, 1 twice
  around the 8-slot `partials` (slots 4-7 hold +0), then the scale, the two norm products and the rotation's
  pairing. The geometry is otherwise free: any lane mapping with the same butterfly pairs gives the same bits.
