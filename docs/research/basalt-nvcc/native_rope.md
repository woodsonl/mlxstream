# basalt `native_rope.cu` as nvcc compiles it: the per-site record

Status: COMPLETE.

What NVIDIA's compiler emits at every floating-point site of basalt's `native_rope.cu` (the native NEOX rotation,
the kernel template `apply<bool>` behind `native_rope_apply`), read from the sm_120 SASS (with the PTX where it shows
what the front end chose), and the Metal form the port writes at each site. This is the file's page under invariant 5
of `docs/design/flash-next.md`; `sampler.md` (branch `woodsonl/basalt-sampler`) is the model for its layout and
notation, and `rope.md` is the page of the host table and the table rotation.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/native_rope.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/native_rope.cu` with basalt's local includes inlined under `#line` directives
  (`include/basalt/kernels/native_rope.hpp`, `mrope.hpp`, `rope_scaling.hpp`), so every PTX and SASS line names its
  basalt file and line. CUDA's own headers come from the compiler's installation. Only basalt's public MIT source was
  sent.
- **Flags.** `native_rope.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`, the file at `:141`) and is
  in the `--use_fast_math` list (`:186-202`, the file at `:194`, the option at `:202`). A Release build (`:12-14`)
  passes CMake's `-O3 -DNDEBUG`, `add_compile_definitions` passes `-DBASALT_VERSION="0.0.1"` (`:7`), and the one
  architecture is 120 (`:3-4`, `:26-31`). `--use_fast_math` means `-ftz=true -prec-div=false -prec-sqrt=false
  -fmad=true` and the fast intrinsic for each single-precision library call (`sinf` becomes `__sinf`, `cosf`
  becomes `__cosf`, `powf` becomes `__powf`, ...); no other `-fmad`, `-ftz`, `-prec-div` or `-Xptxas` flag is set.
  So every single-precision `FADD`, `FMUL` and `FFMA` below carries `.FTZ`.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`. Compiler id `nvcc130`: NVCC 13.0.0, version string
  `Cuda compilation tools, release 13.0, V13.0.48` (`Build cuda_13.0.r13.0/compiler.36260728_0`). Basalt requires
  CUDA 13.0 or newer (`CMakeLists.txt:32-33`). The file was also compiled with `nvcc133` (`Cuda compilation tools,
  release 13.3, V13.3.33`, `Build cuda_13.3.r13.3/compiler.37862127_0`) as a cross-check (section 3).
- **userArguments**, verbatim, both compilers: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'
  --use_fast_math` (the responses' `compilationOptions` echo it). Compiler Explorer adds `-g -lineinfo
  --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC 14.1); none of them changes a kernel's
  code (`sampler.md` section 1).
- **Date.** Compiled October 16, 2026 (both compilers).
- **Translation unit.** The whole file in one unit; its SASS listing is complete (277 lines: `apply<false>`, 120
  instructions, and `apply<true>`, 144; no out-of-line subroutine).
- **Reading a site.** As in `sampler.md`: the PTX shows what the front end fused (`fma.rn.ftz.f32`) and which
  approximation it chose (`sin.approx.ftz.f32`, `cos.approx.ftz.f32`, `lg2.approx.ftz.f32`, `ex2.approx.ftz.f32`);
  the SASS shows what `ptxas` made of it. A row quotes the SASS of one copy where a site is unrolled; the copies were
  compared and differ only in registers.
- **Not float arithmetic.** `IMAD.WIDE` forms addresses; `HFMA2 R, -RZ, RZ, a, b` moves a constant. The tables leave
  them out.

## 2. Kernels in the file and their paths

`native_rope.cu` holds one kernel template and its host launcher. The engine forces native RoPE on
(`src/program/generate.cpp:1893`, applied at `:2401`), so every rotation on the path is the first branch of
`if (native_rope_enabled()) native_rope_apply(...) else rope_neox_apply(...)`. The launcher always takes
`apply<false>`: `rope_table_for` returns no table (`:98-99`, "the kernels compute the angles (the stored table is not
read)"), so `rt.cos` is null and `:122-127` launches the `<false>` instance. The process scaling is `None` with base
1e7 (`src/program/generate.cpp:2135-2147`, set at `:2158`), so `scaling.kernel_args` returns the identity constants
`freq_scale = 1`, `corr_low = corr_high = 0`, `ext_factor = 0`, `attn_factor = 1`
(`include/basalt/kernels/rope_scaling.hpp:54-60`, `:107-109`).

| Kernel or function | Lines | Path | Evidence |
|---|---|---|---|
| `apply<false>` via `native_rope_apply` | `:45-68`, `:102-130` | ON: decode, the verify window's K, query and indexer-query rows (fp16 KV: `fused_pre` needs `st.kv_int8`, `src/core/verify.cpp:1352-1353`) | `norm_rope` (`verify.cpp:1340-1345`, the call at `:1343`), called at `verify.cpp:1451`, `:1452` (K), `:1506`, `:1519` (query), `:1510`, `:1525` (indexer query) |
| the same | | ON: drafter, its K rows (fp16 KV: `kv_fused` needs `st_.kv_int8`, `src/core/mtp.cpp:750-751`) | `norm_rope` (`mtp.cpp:763-767`, the call at `:765`), called at `mtp.cpp:841`; the drafter's query takes the fused `qsa_dec_q_post` (`mtp.cpp:861-864`, `qsa_dec_fused.md`), and `:873` only when that is unsupported |
| the same, YaRN/Linear body (`rope_scaling.hpp:155-160`) | inlined at `:63` | OFF: `ext_factor` is 0 for `None` | `rope_scaling.hpp:109`; reached only by `rope_parity`'s scaled check (`src/kernels/rope_parity.cpp:385`) |
| the same, tail copy `:53-58` | `:54-57` | OFF as a copy: every engine call is in place (`x == out`, `verify.cpp:1343`, `mtp.cpp:765`), so pairs 32 and up only return | `:54` |
| `apply<true>` | `:45-68` with `TAB` | OFF: never launched, `rt.cos` is always null | `:99`, `:122` |
| `native_rope_apply` (host) | `:102-130` | ON (host) | the launch above; `theta_scale` is computed here (section 3.1) |
| `layer.cpp:908`, `normalize_rotate` in `qsa_layer` | | OFF: the per-token layer forward, which the contract's decode does not run (`docs/design/flash-next.md:291-292`) | `src/core/layer.cpp:903`, from `block_layer_pre` at `:1280` |

Basalt's tests run the kernel at `src/kernels/rope_parity.cpp:385` (YaRN) and `:410` (`None`, against the table
rotation at `:417`), and at `src/kernels/qsa_dec_fused_parity.cpp:112`, `:124` and `:126`, where the separate norm and
rotation are the reference the fused kernels are held to (`qsa_dec_fused.md`).

## 3. Per-kernel SASS tables

### 3.1 `apply<false>` (`:45-68`), with `rope_scaled_angle` (`include/basalt/kernels/rope_scaling.hpp:149-163`)

120 SASS instructions, no subroutine. Launch (`:121`, `:126-127`): block 128 threads, grid
(⌈`head_dim / 2` / 128⌉, `rows`) = (1, `rows`) for `head_dim` 128 or 256, no shared memory, no cluster; one thread per
pair (`pair = blockIdx.x * blockDim.x + threadIdx.x`) of row `blockIdx.y`. Pairs below `n_rot / 2` = 32 rotate
`x[pair]` with `x[pair + 32]` (NEOX halves of the first 64 dims); the others return (`:53-58`). Kernel arguments as
SASS reads them: `theta_scale` `c[0x39c]`, `freq_scale` `c[0x3a0]`, `ext_factor` `c[0x3ac]`, `mscale` `c[0x3b0]`
(the launcher's `k.attn_factor`), `positions` `c[0x3b8]`, `mtab` `c[0x3c0]`.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:118` (host) | `const float theta_scale = powf((float) scaling.freq_base, -2.0f / n_rot);` | none: the host's libm `powf` (glibc on basalt's Linux build), passed as a kernel argument | the constant `0x3F1AB32B` for base 1e7 and `n_rot` 64 (section 4); a params-record value, not per-step |
| `:49-51` | `row = blockIdx.y; pair = blockIdx.x * blockDim.x + threadIdx.x; if (row >= rows \|\| pair >= width / 2) return;` | `S2UR SR_CTAID.Y`, `IMAD R2, R2, UR4, R3`; `ISETP.GE.AND`, `ISETP.LE.OR`; `@P0 EXIT` | integer |
| `:53-58` | `if (pair >= n_rot / 2) { if (x != out) { out[...] = x[...]; ... } return; }` | `ISETP.GE.AND P0, PT, R2, R7, PT`; `ISETP.NE.S64.AND P0, PT, R6, R8, PT` (`x` against `out`); `@!P0 EXIT`; two `LDG.E`/`STG.E` word copies | in place on every path (`verify.cpp:1343`, `mtp.cpp:765`): return; the copy is OFF |
| `:62`, `mrope.hpp:48-50` | `mrope_pos(mtab, positions[row], pair)`: `tab ? __ldg(tab + (size_t) pos * 3 + pair % 3) : pos` | `LDG.E R3` (`positions[row]`); `ISETP.NE.S64.AND P0, PT, R8, RZ, PT` (`mtab` against null); `@P0 LDG.E.CONSTANT R3` | `pos = positions[row]` (a per-step buffer, invariant 4); `mtab` is null without `--vision` (`rope.md` section 2) |
| `:62` | `powf(theta_scale, float(pair))` (`__powf` under `--use_fast_math`) | PTX `lg2.approx.ftz.f32 %f25, %f14`; `mul.ftz.f32 %f26, %f25, %f3`; `ex2.approx.ftz.f32 %f27, %f26`. SASS `MUFU.LG2 R13, UR4`; `I2FP.F32.S32 R6, R2` (`float(pair)`, exact); `FMUL.FTZ R13, R6, R13`; `MUFU.EX2 R10, R13` | `p = ftz(fast::exp2(mul_ftz(float(pair), ftz(fast::log2(theta_scale)))))`: invariant 5's `fast::` function per `MUFU`; `p` ≥ `theta_scale`^31 ≈ 1.6e-7, a normal |
| `:62` | `theta_extrap = mrope_pos(...) * powf(...)` | PTX `cvt.rn.f32.s32 %f24, %r36`; `mul.ftz.f32 %f4, %f27, %f24`. SASS `I2FP.F32.S32 R3, R3` (exact: positions < 2^24); `FMUL.FTZ R12, R3, R10` | `mul_ftz(float(pos), p)` |
| `rope_scaling.hpp:153` | `float theta = freq_scale * theta_extrap;` | `FMUL.FTZ R10, R12, UR4` | `mul_ftz(theta_extrap, freq_scale)`: the identity at `freq_scale` = 1 for the path's zero or normal `theta_extrap` (`rope_scaling.hpp:107-109`), so the shared helper may omit it |
| `rope_scaling.hpp:155` | `if (ext_factor != 0.0f) {` | PTX `setp.eq.ftz.f32 %p13, %f18, 0f00000000`; SASS `UFSETP.NEU.FTZ.AND UP0, UPT, UR5, URZ, UPT`; `BRA.U !UP0` | not taken: `ext_factor` is 0 for `None` |
| `rope_scaling.hpp:137-141`, `:156-159` | the YaRN ramp, mix and `mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale)` | OFF: `FADD.FTZ` ×2, `FMNMX.FTZ R0, R0, 0.0010000000474974513054, !PT`, `MUFU.RCP`, `FMUL.FTZ` (the ramp's `div.approx.ftz.f32`); `FMNMX.NAN R8, R6, 1, PT`, `FSETP.GEU.FTZ.AND`, `FADD.FTZ R8, -R8, 1`, `FSEL`; `FMUL.FTZ` by `ext_factor`; `FADD.FTZ R9, -R8, 1`, `FMUL.FTZ`, `FFMA.FTZ R10, R12, R8, R9`; `MUFU.RCP`, `MUFU.LG2`, `FMUL.FTZ` by `0.69314718246459960938`, `FFMA.FTZ R0, R0, R13, 1` (`R13` = 0.1f), `FMUL.FTZ` by `mscale` | none: off-contract (`rope_parity`'s scaled check only, `src/kernels/rope_parity.cpp:385`) |
| `rope_scaling.hpp:161-162` | `cos_out = cosf(theta) * mscale; sin_out = sinf(theta) * mscale;` (`__cosf`, `__sinf`) | PTX `cos.approx.ftz.f32 %f43, %f52`, `sin.approx.ftz.f32 %f44, %f52`; SASS `FMUL.RZ R12, R10, 0.15915493667125701904` (θ · `0x3E22F983` = `float(1 / 2π)`, rounded toward zero, to turns), `MUFU.SIN R13, R12`, `MUFU.COS R11, R12` (one shared turn count) | `t = mul_rz(theta, as_type<float>(0x3E22F983u))` (exact for the path's `theta` ≥ 0: the rounded product `t`, and one ULP down when `t` > 0 and its remainder `fma(theta, k, -t)` is negative), `f = t - rint(t)` (exact), `c = fast::cospi(2.0f * f)`, `s = fast::sinpi(2.0f * f)`: CUDA's structure (the turns rounded toward zero, then the cosine and sine of the turns), not `fast::cos(theta)` |
| the same | `* mscale` | `FMUL.FTZ R13, R13, R0`; `FMUL.FTZ R11, R11, R0` (`R0` = `mscale`, `c[0x3b0]`) | `mul_ftz(s, mscale)`, `mul_ftz(c, mscale)`: the identity at `mscale` = 1 for the path's zero or normal `c` and `s`, so the shared helper may omit them |
| `:65` | `const float a = x[start + pair], b = x[start + pair + n_rot / 2];` | `LDG.E R4`; `IMAD.WIDE R8, R7, 0x4, R4`; `LDG.E R8` | two loads |
| `:66` | `out[start + pair] = a * c - b * s;` | PTX `mul.ftz.f32 %f47, %f54, %f45` (c·a), `mul.ftz.f32 %f48, %f55, %f46` (s·b), `sub.ftz.f32 %f49, %f47, %f48`; SASS `FMUL.FTZ R0, R13, R8` (s·b, rounded); `FFMA.FTZ R11, R11, R4, -R0`: `ptxas` fused the PTX's `mul` and `sub`, so a·c is the exact product; `STG.E` | `cuda_nan(fma_ftz(c, a, -mul_ftz(s, b)))` |
| `:67` | `out[start + pair + n_rot / 2] = a * s + b * c;` | PTX `mul.ftz.f32 %f50, %f54, %f46` (c·b), `fma.rn.ftz.f32 %f51, %f55, %f45, %f50`; SASS `FMUL.FTZ R10, R11, R8` (c·b, rounded); `FFMA.FTZ R13, R13, R4, R10` (a·s exact); `STG.E` | `cuda_nan(fma_ftz(s, a, mul_ftz(c, b)))`: a·s fused, b·c rounded, the reverse of the table kernel's second output (`rope.md` section 3.1) |

### 3.2 `apply<true>` (`:45-68` with `TAB`), OFF-PATH

144 SASS instructions: the same float opcodes per source line as `apply<false>` (compared as multisets), plus the
table read of `rope_tab_cs` (`mrope.hpp:53-58`: two `LDG.E.CONSTANT` and the bounds test) ahead of them. Never launched: `rt.cos`
is always null (`:99`, `:122`). Not tabulated.

### 3.3 nvcc 13.3

`nvcc133` (`Cuda compilation tools, release 13.3, V13.3.33`) gives `apply<false>` 120 instructions and `apply<true>`
152. Per source line, the float-class opcodes with their immediates are the same in both instances (13 lines each),
and in `apply<false>` the four rotation instructions are register for register the same, `FMUL.FTZ R0, R13, R8`,
`FMUL.FTZ R10, R11, R8`, `FFMA.FTZ R11, R11, R4, -R0`, `FFMA.FTZ R13, R13, R4, R10`, so the fusion choice at `:66`
and `:67` does not depend on the toolkit. The integer differences are spelling only (`IADD3` to `IADD`,
`ISETP.NE.S64.AND` to `ISETP.NE.U64.AND`, `IMAD.SHL.U32`/`IADD.64` address forms to `IMAD.WIDE`, `NOP` padding).

## 4. What the record implies for the Metal port

- **No libdevice routine.** The PTX has no `call` and no `__nv_` or `__cudart_` symbol; its only `.f64` operations
  (`abs.f64`, `setp.ne.f64`) are the body of `std::isfinite(double)` (from `rope_scaling.hpp`), which
  `--keep-device-functions` emits and no kernel calls; the SASS has no `CALL`, no `FCHK` and no `D*` instruction.
  Under `--use_fast_math`, `powf` is `__powf` (`lg2.approx.ftz`, `mul.ftz`, `ex2.approx.ftz`) and `sinf`/`cosf`
  are `__sinf`/`__cosf` (`sin.approx.ftz`/`cos.approx.ftz`): hardware approximations, not library code. Nothing
  here falls under the libdevice rule.
- **One host libm value: `theta_scale` (`:118`).** `powf(1e7f, -0.03125f)` runs on the host; the kernel receives the
  float. The exact value 10^(-7/32) = 0.60429639023813... lies 0.067 ULP above `0x3F1AB32B` (0.6042963862419128)
  [M: computed to 60 digits on the host], so every `powf` whose error is below 0.93 ULP returns `0x3F1AB32B`;
  glibc's `powf` states a 0.82 ULP bound [R], and macOS's `powf` (numpy float32 `power`) returns `0x3F1AB32B` [M].
  The port passes that constant (it depends only on the model's base 1e7 and `n_rot` 64) and a test asserts it
  equals the correctly rounded binary32 of 10^(-7/32). Recorded as an open question only for the record: its bits are
  settled.
- **The rotation is exact transcription.** `out[pair] = fma(c, a, -fl(s·b))` and `out[pair + 32] = fma(s, a,
  fl(c·b))` (section 3.1), every operation `.FTZ`: `fma_ftz` and `mul_ftz` in that pairing. The second output's
  pairing is the reverse of `rope.cu`'s table rotation (`rope.md` section 3.1); the two kernels are not
  interchangeable bit for bit. The only operands that can be subnormal are the activations `a` and `b`, which the
  helpers flush as `.FTZ` does; the angle chain's values are zero or normal (`p` ≥ 1.6e-7, `pos` an integer below
  2^24).
- **The angle is approximate, by invariant 5, and its reduction is not.** `MUFU.LG2`, `MUFU.EX2`, `MUFU.SIN` and
  `MUFU.COS` map to `fast::log2`, `fast::exp2`, `fast::sinpi`, `fast::cospi`; the `FMUL.RZ` to turns is an exactly
  specified rounding and is transcribed exactly (`mul_rz`). The measured consequence [M] (this Mac, M3 Max,
  `MTLMathModeSafe`, every `pos` < 262144 and `pair` < 32: 8,388,608 angles): this form stays within 1.7e-7
  (`|dc| + |ds|`) of a binary64 model of CUDA's chain (the turns rounded toward zero, then an exact cosine and sine of
  the turns) at every position, where `fast::sin(theta)`/`fast::cos(theta)` depart from it by up to 1.5e-3 within
  8192 cells and 4.9e-2 at the trained context. So the Metal rotation's bits differ from CUDA's in the last bits of
  `p`, `c` and `s` only; the tests that hold it are `rope_parity`'s `if (!(rel <= 3e-3)) ++agree_bad;`
  (`src/kernels/rope_parity.cpp:400`) and the contract's double test of the pooled indexer keys
  (`docs/design/flash-next.md:376`, `:464`).
- **One angle helper at every native rotation site.** Basalt holds the fused query post bitwise against this kernel
  after `norm<256>` ("with the arithmetic of the separate kernels bit for bit",
  `include/basalt/kernels/qsa_dec_fused.hpp:1-2`; the reference calls at `src/kernels/qsa_dec_fused_parity.cpp:123-126`;
  `qsa_dec_fused.md`), and invariant 5 gives each operation one helper at every site, so `native_rope`,
  `qsa_dec_q_post` and the native indexer's key and query rotations all call the same
  `rope_cs(pos, pair, theta_scale)`; the Metal comparisons stay bitwise because the helper is deterministic.
  The identity multiplies by `freq_scale` and `mscale` (both 1 for `None`) change no bit on the path and may be left
  out of the helper.
- **Geometry is free.** No value crosses threads: each output depends only on its own pair, `pos` and
  `theta_scale`, so any grid gives the same bits. The port runs in place (every engine call is), so pairs 32 and up
  do nothing; the copy branch (`:54-57`) is OFF.
- **Stores and NaN.** A NaN or infinite activation makes the `FFMA` result NaN, which sm_120 stores as `0x7fffffff`:
  both stores go through `cuda_nan`.
- **Per-step data.** `positions[row]` only (a buffer, invariant 4); `theta_scale`, `n_rot` and the row width are
  per-program constants.

**Helper inventory.**

| Helper | Where it lives | Used at |
|---|---|---|
| `rope_cs(pos, pair, theta_scale)`: `lg = ftz(fast::log2(theta_scale))`, `p = ftz(fast::exp2(mul_ftz(float(pair), lg)))`, `t = mul_rz(mul_ftz(float(pos), p), 0x3E22F983)`, `f = t - rint(t)`, `(fast::cospi(2f), fast::sinpi(2f))` | new; [M] on this Mac its `mul_rz` gives the round-toward-zero product at all 8,388,608 angles of the path, and the helper the chain measured above (section 4) bit for bit | `:62`, `rope_scaling.hpp:153`, `:161-162` |
| `mul_rz` | new, with `rope_cs` (same file) | the `FMUL.RZ` inside `sin.approx`/`cos.approx` |
| `ftz` | `kernels/common/basalt_fp.metal:45` (origin/main) | inside `rope_cs` |
| `mul_ftz` | `basalt_fp.metal:80` | `:62`, `:66`, `:67` |
| `fma_ftz` | `basalt_fp.metal:73` | `:66`, `:67` |
| `cuda_nan` | `basalt_fp.metal:120` | the stores at `:66`, `:67` |
| `fast::log2`, `fast::exp2`, `fast::sinpi`, `fast::cospi`, `rint` | MSL standard library | inside `rope_cs` |
