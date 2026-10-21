# basalt's GDN decode recurrence as nvcc compiles it

## Method

- **Source.** basalt `src/kernels/cuda/dec_gdn_rec.cu` at `b4c63b55754637f4c72ad53a0f791d425f71eb36` (MIT, 405
  lines). Line numbers below are that file's unless a header is named.
- **Flags.** basalt `CMakeLists.txt` at b4c63b5: `:3-4` set `CMAKE_CUDA_ARCHITECTURES 120`; `:6-7`
  `project(basalt VERSION 0.0.1 ...)` and `add_compile_definitions(BASALT_VERSION="${PROJECT_VERSION}")`; `:12-14`
  default the build type to Release, so CMake adds its CUDA Release flags `-O3 -DNDEBUG`; `:118-167` the `kernels`
  static library lists the file at `:162`. No option reaches it per file or per target: no `CMAKE_CUDA_FLAGS`, no
  `CMAKE_CUDA_STANDARD` (nvcc's default dialect), `target_compile_options` only on the CXX `warnings` interface
  (`:44-48`) and on the separate `mmq` target (`:728`), and the one CUDA per-file option, `--use_fast_math` on the
  list at `:186-202`, names neither `dec_gdn_rec.cu` nor `verify_kernels.cu`. The file therefore builds with nvcc's
  IEEE defaults, `-ftz=false -prec-div=true -prec-sqrt=true -fmad=true` (the file's own comment, `:3-5`, says it is
  built with the flags of `verify_kernels.cu`).
- **Compiler.** Compiler Explorer, compiler id `nvcc130` ("NVCC 13.0.0"); its `nvcc --version`:
  `Cuda compilation tools, release 13.0, V13.0.48` / `Build cuda_13.0.r13.0/compiler.36260728_0`.
- **userArguments**, exactly: `-arch=sm_120 -lineinfo -O3 -DNDEBUG -DBASALT_VERSION=\"0.0.1\"` with
  `compilerOptions: {produceDevice: true}`, `filters: {}`. Compiler Explorer adds `-g --keep-device-functions -c -keep
  -Xcompiler=-S`; `-g` is host debug information and does not change device code.
- **Date.** October 18, 2026.
- **Source unit.** One self-contained file per kernel instantiation: `dec_gdn_rec.cu` with every `#include
  "basalt/..."` header (public, MIT) inlined recursively under `#line` markers, so every row of the listing keeps
  basalt's file and line; system and CUDA headers stay includes. The host launchers (`:354-404`) are blanked (kept
  as empty lines) and one appended line takes the address of the instantiation the listing is about, for example
  `void* gdn_nvcc_table[] = {(void*) basalt::kernels::rec_kernel<2>};`. Ten compiles: `rec_kernel<2>`, `<4>`, `<8>`
  (`kDecMaxT = 8`, `include/basalt/kernels/dec_gdn.hpp:16`; the host picks `<2>` for `n_tok <= 2`, `<4>` for
  `n_tok <= 4`, else `<8>`, `:362-364`), the same three of `rec_split_kernel` (`:391-393`) and `rec_seg_kernel`
  (`:377-379`), and `rec_norm_kernel` (`:400`). Every compile returned code 0 with an empty stderr. The units, the
  requests, the raw responses and the per-row listings are outside the repository (the lane's scratch,
  `.context/gdn/nvcc/`).
- **Reading.** SASS is the ground truth, PTX a guide. Every float instruction of every listing was attributed to its
  source line through `-lineinfo` and counted per (line, opcode with modifiers); the sets are compared across the
  instantiations (section 5). Where a library sequence is inlined (`log1pf`), a probe compile of the bare function
  and of `__nv_log1pf` identifies it. The `<8>` listings reach Compiler Explorer's 5000-row limit only after the
  kernel body: the body is complete, the cut falls in the out-of-line slow-path subroutines, which are complete in
  the `<2>` and `<4>` listings.

## Notation

- SASS is quoted with source names for registers: `FFMA d, a, b, c` is `d = a * b + c` rounded once. A modifier not
  shown is absent: "no FTZ" means the instruction has no `.FTZ`, so it reads and writes subnormals (IEEE).
- `ex2n(e)`: the non-ftz `ex2.approx.f32` sequence nvcc emits for `__expf` in this build,
  `FSETP.GEU.AND P0, PT, e, -126, PT; @!P0 FMUL e, e, 0.5; MUFU.EX2 r, e; @!P0 FMUL r, r, r` (an argument below
  -126 is halved and the result squared, so subnormal results survive; `GEU` is true for NaN). `__expf(x)` is
  `FMUL e, x, 1.4426950216293334961` (`0x3FB8AA3B`, no FTZ) then `ex2n(e)`; `__expf(-x)` folds the negation into the
  constant, `FMUL e, x, -1.4426950216293334961` (`0xBFB8AA3B`), the same bits as `-x * 0x3FB8AA3B`.
- `rsqn(x)`: the non-ftz `rsqrt.approx.f32` sequence, `FSETP.GEU.AND P0, PT, |x|, 1.175494350822287508e-38, PT;
  @!P0 FMUL x, x, 16777216; MUFU.RSQ r, x; @!P0 FMUL r, r, 4096` (a subnormal or zero argument is scaled by 2^24
  and the result by 2^12).
- `rcp_rn(d)`: the correctly rounded `rcp.rn.f32` of `1.0f / d`, `IADD3 t, d, 0x1800000; LOP3.LUT t, t, 0x7f800000,
  RZ, 0xc0; ISETP.GT.U32.AND P0, PT, t, 0x1ffffff, PT`, fast path `MUFU.RCP r, d; FFMA e, d, r, -1; FADD.FTZ e, -e,
  -RZ; FFMA r, r, e, r`, otherwise `CALL __cuda_sm20_rcp_rn_f32_slowpath` (out of line; exponent extremes, zero,
  infinity, NaN, subnormal `d`). Result: the correctly rounded IEEE reciprocal.
- `div_rn(n, d)`: the full-range `div.rn.f32` of a `/`, `MUFU.RCP r, d; FCHK P0, n, d; FFMA t, -d, r, 1;
  FFMA r, r, t, r; FFMA q, n, r, RZ; FFMA e, -d, q, n; FFMA q, r, e, q; @P0 CALL
  __cuda_sm3x_div_rn_noftz_f32_slowpath`. Result: the correctly rounded IEEE quotient, subnormals kept.
- Metal names: `kernels/common/basalt_fp.metal` and `kernels/common/basalt_q8_1.metal` on origin/main (`ieee_add`
  `:95`, `ieee_mul` `:108`, `ieee_rcp` `:153`, `ieee_fma` `:167`, `ieee_div` `:181`, `warp_sum_ieee` `:206`,
  `q8_1_store_native` `:52`); on `origin/woodsonl/basalt-sampler` the `ieee_*` helpers move to `basalt_ieee.metal`.
  Helpers this record calls for that do not exist yet are named in section 6.
- **Subnormal reachable**: whether an fp32 subnormal can be an operand or the result at that site for some input the
  kernel accepts, with the reason; "immaterial" marks a site where a subnormal can occur but cannot change any output
  bit.

## Summary

- The file builds with nvcc's IEEE defaults. Every float instruction of the five kernels is unflagged except the two
  explicit `div.approx.ftz` sites (`FMUL.FTZ` by `fl(1/127)`; `MUFU.RCP` then `FMUL.FTZ`) and the `FADD.FTZ` inside
  the correctly rounded reciprocal's refinement. The out-of-line `rcp.rn` and `div.rn` slow paths use `FADD.FTZ` and
  `FSETP.*.FTZ` internally and return the correctly rounded IEEE value (PTX `.rn` semantics). No `MUFU.LG2`,
  `MUFU.SQRT` or `F2F` occurs; the int-to-float conversions are `I2FP.F32.S32` in `log1pf` and `I2F.RP` in the
  integer divisions.
- `__expf` is the non-ftz `ex2.approx` (halving below -126, squaring the result); `rsqrtf` the non-ftz
  `rsqrt.approx` (2^24 prescale, 2^12 postscale); `rsqrtf(128.0f)` is folded to `0x3DB504F3` (6b).
- `log1pf` at `:111` is libdevice's `__nv_log1pf` inlined (6a); on Metal its round-toward-zero add and its
  subnormal pass-through need explicit handling.
- FFMAs the source does not write: the conv sum `:64`, `ss / S + eps` at `:119`, `delta` at `:255` (6d).
- `norm_q8`'s q8_1 has `native_quantize_q8_1`'s divisions; only the unflagged sum can change an output bit (the
  sign of a half zero in `ds.y`; larger sums only through exact ties): `q8_1_store_native` with `warp_sum_ieee` (6c).
- All MAXT instantiations and `rec_seg_kernel` share one per-line float operation set; `rec_split_kernel` plus
  `rec_norm_kernel` reproduces `rec_kernel`'s bits (sections 4, 5).
- Subnormals are material at `:65`, `:109`, `:111-112`, `:120`, `:228-229`, `:248`, `:251-260`, `:265-266` and the
  q8_1 sum; new helpers needed: a non-ftz `ex2` (scaled form) and a `log1pf`; the `ieee_*` helpers cover the rest.

## Port mapping

The SASS columns are the record; the Metal columns are its proposals, written before the port. The port,
`kernels/common/basalt_gdn.metal`, writes the SASS sequence at every site through shared steps that `dec_gdn_rec` and
the per-stage kernels of `verify_gdn.md` both call, with one helper per operation wherever a subnormal can occur. Every
`__expf` is `expf_ieee` (`kernels/common/basalt_ieee.metal`, the `ex2n` helper of 6b), at `expf_neg`'s sites as well,
with `__expf(-x)` as `expf_ieee(sign_flip(x))`; every run-time `rsqrtf` is `rsqrt_ieee` (`rsqn` with its prescale and
postscale, not `fast::rsqrt` alone), and `rsqrtf((float) S)` the literal `0x3DB504F3`; `1.0f / (1.0f + e)` and the
SiLU's quotient are `ieee_div`, not `ieee_rcp`, which is documented only up to 2^126; `log1pf` is `log1pf_nv`
(`kernels/common/basalt_libdevice.metal`: 6a's sequence, its twelve `FFMA` as `fma`); the q8_1 rows are `q8_1_store_rec`
(`kernels/common/basalt_q8_1.metal`: 6c's form, native's flushed max, `mul_ftz` and `rcp_ftz`, the sum by
`warp_sum_ieee`). Every other `FFMA` is `fma_ieee` (`kernels/common/basalt_ieee_f64.metal`), 6d's three contractions
(`:64`, `:119`, `:255`) and the chains SASS starts from `RZ` (on `0.0f`) included, and every other unflagged `FMUL`,
`FADD` and butterfly is `ieee_mul`, `ieee_add` and `warp_sum_ieee`, the sums of squares, `v + dt` and the partial sums
of `oc` and `ss` included. The stored h, gate, beta and y go through `cuda_nan` (`nan.md`).
`tests/kernels/test_basalt_gdn_helpers.py` checks each step bit for bit against `tests/kernels/basalt_emu.py`, which
follows this record's SASS rows.

## 1. Shared device functions (inlined into every kernel)

Every listing inlines these; the instructions are the same in every instantiation (section 5), attributed to the
function's own lines. `conv_silu` and `ab_one` run inside `rec_block` (sections 2, 3, 5), `norm_q8` and `sq32` in
`rec_block` and `rec_norm_kernel` (section 4). Register names are replaced by the values they hold.

| Line | Expression | SASS | Metal | Subnormal reachable |
|---|---|---|---|---|
| `:40-44` | `div_approx_ftz(a, b)`: `asm("div.approx.ftz.f32 %0, %1, %2;" ...)` | divisor the literal 127 (`:128`): `FMUL.FTZ d, amax, 0.0078740157186985015869` (`0x3C010204` = `fl(1/127)`); variable divisor (`:49`): `MUFU.RCP r, d; FMUL.FTZ q, a, r` | `mul_ftz(amax, 0x1.020408p-7f)`; `mul_ftz(a, rcp_ftz(d))` (`basalt_fp.metal:80`, `:83`), the contract's mapping | flushed by the instruction, as in `q8_1_store_native` |
| `:48` | `if (amax == 0.0f) return 0;` | `FSETP.NEU.AND P, PT, amax, RZ, PT` (no FTZ) | the native test, `amax == 0.0f` (Metal reads a subnormal as 0) | yes, immaterial: a subnormal `amax` gives `d = 0` (`:128`) and `F2I(NaN) = 0` codes, the fast-math codes (6c) |
| `:49` | `roundf(div_approx_ftz(xi, d))` | `MUFU.RCP r, d; FMUL.FTZ q, yv, r`; `LOP3.LUT h, 0.5, 0x80000000, q, 0xb8` (copysign); `FADD.RZ q, q, h` (no FTZ); `FRND.TRUNC q, q` | `round(mul_ftz(yv, rcp_ftz(d)))` (Metal `round` breaks ties away from zero like `roundf`) | no: `q` comes out of `FMUL.FTZ` |
| `:50` | `(int8_t) (q > 127.0f ? 127.0f : q < -127.0f ? -127.0f : q)` | `FSETP.GT.AND P, PT, q, 127, PT; FMNMX.NAN q, q, -127, !PT; FSEL q, q, 127, !P; F2I.TRUNC.NTZ` (NaN gives 0) | `q8_1_code` (`basalt_q8_1.metal:37`) | no |
| `:64` | `float sum = v0 * w[0] + v1 * w[1] + v2 * w[2] + x * w[3];` | `FMUL a, w1, v1; FFMA a, w0, v0, a; FFMA a, w2, v2, a; FFMA sum, w3, x, a` (contracted, no FTZ) | `fma(x, w3, fma(v2, w2, fma(v0, w0, v1 * w1)))`; `ieee_mul`, `ieee_fma` where a subnormal can occur | yes: an operand product below `2^-126` (operands near `2^-63`, or a subnormal weight), or a sum that cancels below `2^-126` (`:65`); not in model data |
| `:65` | `return sum / (1.0f + __expf(-sum));` | `FMUL e, sum, -1.4426950216293334961`, `ex2n(e)`, `FADD d, r, 1`, `div_rn(sum, d)` (MUFU.RCP, FCHK, five FFMA, slow path) | `sum / (1.0f + expf_neg(sum))`, correctly rounded; `ieee_div` for a subnormal `sum` or quotient | yes, material: a subnormal `sum` gives a subnormal SiLU (`d >= 1`) that the L2 norm scales back up (`:228-229`); `e` subnormal is immaterial (`1 + e = 1`) |
| `:103-106` | 80 `v = fmaf(__uint_as_float(w.x << 16), xa[q].x, v); v = fmaf(__uint_as_float(w.x & 0xffff0000u), xa[q].y, v); ...` (q = 0..9, eight per `q`: `w.x` lo, hi with `xa.x`, `xa.y`; `w.y` with `xa.z`, `xa.w`; `w.z` with `xb.x`, `xb.y`; `w.w` with `xb.z`, `xb.w`) | `SHF.L.U32 w, w, 0x10` / `LOP3.LUT w, w, 0xffff0000, RZ, 0xc0`; one serial chain of 80 `FFMA v, w, x, v` in source order, the first `FFMA v, w, x, RZ` (no FTZ); the `q` loop is unrolled, the job loop around `ab_one` is not | the same serial `fma` chain from `0.0f`; `ieee_fma` where a subnormal can occur | yes: a bf16 weight below `2^-126` is an fp32 subnormal (Metal's `fma` reads it as 0, CUDA multiplies it into a possibly normal product), or a running sum below `2^-126`; neither occurs in model data |
| `:108` | `for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);` | `SHFL.BFLY r, v, o, 0x1f; FADD v, v, r` for o = 16, 8, 4, 2, 1 (no FTZ) | `warp_sum_ieee` (`basalt_q8_1.metal:206`), or `warp_sum` (`basalt_fp.metal:97`) where no subnormal can reach | as `:103-106` |
| `:109` | `if (is_beta) return 1.0f / (1.0f + __expf(-v));` | `FMUL e, v, -1.4426950216293334961`, `ex2n(e)`, `FADD d, r, 1`, `rcp_rn(d)` | `ieee_rcp(1.0f + expf_neg(v))` (`basalt_q8_1.metal:153`) | yes, material: `v` in about (-88.72, -87.34) gives `d > 2^126` and a subnormal beta (stored, `:239`, and used at `:255`) |
| `:110` | `const float u = v + dt;` | `FADD u, dt, v` | `v + dt` | immaterial: a subnormal `u` needs `v` and `dt` within `2^-126`; `__expf` of it is 1 either way |
| `:111` | `const float sp = u > 20.0f ? u : log1pf(__expf(u));` | `FSETP.GT.AND P0, PT, u, 20, PT; @P0 BRA`; `FMUL e, u, 1.4426950216293334961`, `ex2n(e)`; `__nv_log1pf` inlined (6a) | `u > 20.0f ? u : log1pf_helper(ex2n(u * 0x1.715476p+0f))`, the helper per 6a | yes, material: `u < -87.34` gives a subnormal `__expf(u)` and `sp = __expf(u)` exactly |
| `:112` | `return sp * sa;` | `FMUL g, sa, sp` (no FTZ) | `ieee_mul(sp, sa)` | yes: a subnormal `sp` (`:111`), or a small `sp` times `\|sa\| < 1`; the gate is stored (`:238`) |
| `:119` | `const float scale = rsqrtf(ss / (float) S + a.eps);` | `FFMA x, ss, 0.0078125, eps` (division by 128 made a multiply by `2^-7`, then contracted), `rsqn(x)` | `fast::rsqrt(fma(ss, 0x1p-7f, a.eps))` | `ss` subnormal: immaterial (a sum a subnormal can change is far below `eps`'s half ulp); `x` never subnormal while `eps >= 2^-126` |
| `:120` | `const float yv = oc * scale * gam * (1.0f / (1.0f + __expf(-z)));` | `FMUL t, oc, scale; FMUL t, gam, t`; `FMUL e, z, -1.4426950216293334961`, `ex2n(e)`, `FADD d, r, 1`, `rcp_rn(d)`; `FMUL yv, t, r` (left to right, no FTZ) | `((oc * scale) * gam) * ieee_rcp(1.0f + expf_neg(z))`, each `*` an `ieee_mul` where it can underflow | yes, material: a small `oc` (decayed or zero state, small `q`) or `z < -87.34` (subnormal gate factor); `yv` is stored to `a.y` and quantized |
| `:122-127` | `float amax = fabsf(yv), sum = yv;` then `amax = fmaxf(amax, __shfl_xor_sync(.., amax, o, 32)); sum += __shfl_xor_sync(.., sum, o, 32);` for o = 16..1 | `FADD a, \|yv\|, -RZ` (abs, no FTZ, folded into the first `FMNMX` as `\|yv\|`); per offset `SHFL.BFLY` and `FMNMX a, a, r, !PT` (no FTZ), `SHFL.BFLY` and `FADD s, s, r` (no FTZ), interleaved | `amax`: `warp_max_ftz(abs_ftz(yv))` as `q8_1_store_native` (cannot change a bit, 6c); `sum`: `warp_sum_ieee(yv)` | yes, material for `sum` only: the sign of the half zero `ds.y` (6c) |
| `:128` | `const float d = q8_1_finite(div_approx_ftz(amax, 127.0f));` | `FMUL.FTZ d, amax, 0x3C010204`; `FSETP.GT.AND P, PT, \|d\|, 65504, PT; LOP3.LUT c, 65504, 0x80000000, d, 0xb8; FSEL d, c, d, P` (65504 materialized by `HFMA2`) | `q8_1_finite(mul_ftz(amax, 0x1.020408p-7f))` (`basalt_q8_1.metal:34`) | no: `FMUL.FTZ` |
| `:130` | `blk->qs[lane] = q8_1_quant_fm(yv, d, amax);` | `:48-50` above | the native code path of `q8_1_store_native` | as `:48-50` |
| `:131` | `if (lane == 0) blk->ds = q8_1_ds(d, sum);` | `q8_1_finite` of `d` and `sum` (`q8_1_finite.hpp:19`: `FSETP.GT.AND`, `LOP3.LUT`, `FSEL`), `F2FP.F16.F32.PACK_AB ds, sum, d` (`cuda_fp16.hpp:264`, round to nearest even) | `q8_1_put` (`basalt_q8_1.metal:42`) with the IEEE `sum` | the half of a subnormal `sum` is a zero with `sum`'s sign |
| `:136` | `float q = v * v;` | `FMUL q, v, v` (no FTZ; not contracted: `q` has two uses) | `v * v` | yes, immaterial: `\|v\| < 2^-63`; the sum only enters `+ eps` (`:119`) |
| `:137` | `for (int o = 16; o > 0; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);` | `SHFL.BFLY r, q, o, 0x1f; FADD q, q, r` for o = 16..1 (no FTZ) | `warp_sum` | as `:136` |

`HFMA2 r, -RZ, RZ, c` rows (constants 0.25, 65504, 127) are constant moves, not arithmetic; every `HFMA2` of every
listing has that form, including the one row at `:73` (`HFMA2 R0, -RZ, RZ, 0, 5.9604644775390625e-07`, the integer 10
of the timing stamp's index `blockIdx.x * 10 + i`). A diff of every source line that carries a float opcode in the ten
listings against the lines this record cites leaves no other line.

## 2. `rec_kernel<MAXT>` (MAXT = 2, 4, kDecMaxT)

`:302-307`: `__cluster_dims__(4, 1, 1) __launch_bounds__(S)` (`S = 128`), one cluster of four blocks per value head,
`rec_block<MAXT, false>(a, T, blockIdx.x / 4, blockIdx.x % 4, sm, &cl)`. The rows are `rec_block`'s (`:142-300`);
the device functions it calls are in section 1. The loops over the window's tokens at `:205-220`, `:224-236` and
`:292-296` are unrolled `MAXT` times (`if (t >= T) break;`), so their arithmetic appears once per token; the job loop
at `:182` and the recurrence loop at `:247` are runtime loops, one body each. A shuffle butterfly inside a
conditional (`:108`, `:211-212`, `:137`) has a second copy behind `BRA.DIV` for a diverged warp, with the same `FADD`s
in the same order (the counts in the listing are doubled by it, less one shared final `FADD`).

| Line | Expression | SASS | Metal | Subnormal reachable |
|---|---|---|---|---|
| `:149` | `const int qh = head % a.h_k;` | nvcc's integer division sequence (`IABS`, `I2F.RP`, `MUFU.RCP`, `IADD3 r, r, 0xffffffe`, `F2I.FTZ.U32.TRUNC.NTZ`, `IMAD.HI.U32`, compare-and-subtract fix-ups): an exact integer result | `head % h_k` on integers | n/a |
| `:158`, `:163-167`, `:173` | state `s[r]`, conv weights and history, `dt`, `sa = a.ssm_a[head]`, `gam` | loads | loads | the stored state can hold subnormals (a decaying recurrence) |
| `:182-187` | `for (int job = rg; job < 2 * T; job += RG) { ... ab_one(...) ... }` | `ab_one` (section 1, `:89-113`) once, in a runtime loop | the same job split | section 1 |
| `:207`, `:208`, `:216` | `yq[t] = conv_silu<MAXT>(hq, wq, xq, t);` (`yk`, and `yv` for `rg == 1`) | section 1 `:64-65` per token and channel; the window `[h0, h1, h2, x0, ..]` from `t` (copies `t = 0`, `1` compared) | section 1 | section 1 |
| `:209` | `float sq = yq[t] * yq[t], sk = yk[t] * yk[t];` | `FMUL sq, yq, yq; FMUL sk, yk, yk` (no FTZ) | `yq * yq` | yes, immaterial: `\|yq\| < 2^-63`; the sum only meets `+ eps` (`:228`) |
| `:210-212` | `sq += __shfl_xor_sync(0xffffffffu, sq, o); sk += ...` for o = 16..1 | per offset `SHFL.BFLY` and `FADD sq, sq, r`, then the same for `sk` (no FTZ) | `warp_sum` (`basalt_fp.metal:97`) of each | as `:209` |
| `:226`, `:227` | `ssq = sm.part[0][t][0] + sm.part[0][t][1] + sm.part[0][t][2] + sm.part[0][t][3];` (`ssk` the same) | `FADD a, p0, p1; FADD a, p2, a; FADD a, p3, a` (left to right, no FTZ) | `((p0 + p1) + p2) + p3` | as `:209` |
| `:228`, `:229` | `yq[t] *= rsqrtf(ssq + a.eps);` (`yk` with `ssk`) | `FADD x, ssq, eps` (no contraction: a sum, not a product), `rsqn(x)`, `FMUL yq, r, yq` | `ieee_mul(yq, fast::rsqrt(ssq + eps))` | yes, material: a subnormal SiLU (`:65`) times `rsqrt(eps)` (up to 1000 at `eps = 1e-6`) can be normal; the result goes to `sm.qn`/`sm.kn` and `a.h` |
| `:238-239` | `a.gate[...] = sm.g[tid]; a.beta[...] = sm.b[tid];` | stores | stores | the stored gate and beta keep subnormals (section 1 `:109`, `:112`) |
| `:248` | `const float g = __expf(sm.g[t]);` | `FMUL e, gate, 1.4426950216293334961`, `ex2n(e)` | `ex2n(gate * 0x1.715476p+0f)` (6b) | yes, material: a gate in about (-103.3, -87.34) |
| `:251` | `kv = fmaf(s[r], sm.kn[t][rg * RPG + r], kv);` r = 0..31 | `FFMA kv, kn0, s0, RZ`, then 31 serial `FFMA kv, kn_r, s_r, kv` (no FTZ) | the serial `fma` chain from `0.0f`; `ieee_fma` | yes: subnormal state elements (a decaying state), products of small values |
| `:254` | `kv_col = red_kv[0][lane] + red_kv[1][lane] + red_kv[2][lane] + red_kv[3][lane];` | `FADD a, r0, r1; FADD a, a, r2; FADD a, a, r3` | `((r0 + r1) + r2) + r3`; `ieee_add` | yes: small partial sums |
| `:255` | `const float delta = (sm.v[t][lane] - g * kv_col) * sm.b[t];` | `FFMA t, -g, kv_col, v; FMUL delta, t, b` (contracted, no FTZ) | `ieee_mul(ieee_fma(-g, kv_col, v), b)` | yes, material: a subnormal `g` (`:248`) or `b` (`:109`) |
| `:259` | `s[r] = fmaf(g, s[r], sm.kn[t][rg * RPG + r] * delta);` | `FMUL p, delta, kn_r; FFMA s_r, g, s_r, p` (32 times, no FTZ) | `ieee_fma(g, s[r], ieee_mul(kn, delta))` | yes, material: subnormal `g`, `s[r]`, or a product of a small key element and a small `delta` |
| `:260` | `o = fmaf(s[r], sm.qn[t][rg * RPG + r], o);` | `FFMA o, s_r, qn_r, RZ` (the updated `s_r`), then 31 serial `FFMA o, s_r, qn_r, o` | the serial `fma` chain from `0.0f`; `ieee_fma` | yes, as `:259` |
| `:265-266` | `oc = (red_o[0][lane] + red_o[1][lane] + red_o[2][lane] + red_o[3][lane]) * rsqrtf((float) S);` | `FADD a, o0, o1; FADD a, a, o2; FADD a, a, o3; FMUL oc, a, 0.08838834613561630249` (`rsqrtf(128.0f)` folded to `0x3DB504F3`, no `MUFU.RSQ`) | `ieee_mul(((o0 + o1) + o2) + o3, 0x1.6a09e6p-4f)` | yes, material: a small output row (the multiply by 0.088 can underflow) |
| `:270-272` | `sq_part = sq32(oc); sm.oc[t][lane] = oc; if (lane == t) wsum_t = sq_part;` | section 1 `:136-137`; the last butterfly `FADD` predicated on `lane == t` | `sq32` | section 1 `:136` |
| `:280-286` | cluster exchange of `wsum_t`: `__shfl_sync(.., wsum_t, t)` into every rank's `sm.wsum[t][rank]` | `UCGABAR_WAIT` (the cluster barrier), a generic `ST.E` into the mapped rank's shared memory; no float instruction | no arithmetic: the four ranks' partials reach `:294` unchanged | n/a |
| `:294` | `ss = sm.wsum[t][0] + sm.wsum[t][1] + sm.wsum[t][2] + sm.wsum[t][3];` | `FADD a, w0, w1; FADD a, w2, a; FADD a, w3, a` (left to right) | `((w0 + w1) + w2) + w3` | as `:209` |
| `:295` | `norm_q8(a, t, head, rank, lane, sm.oc[t][lane], ss, gam, zz[t]);` | section 1 `:116-131` per token | section 1 | section 1 |

`zz[t]` (`:198`) is `a.z` of the head's column; `S`, `RPG = 32` and `RG = 4` are compile-time constants.

## 3. `rec_split_kernel<MAXT>`

`:309-313`: `__launch_bounds__(S)`, no cluster: `rec_block<MAXT, true>(a, T, blockIdx.x / 4, blockIdx.x % 4, sm,
nullptr)`. With `SPLIT` the block loads no `z` (`zz[t] = 0`, `:198`), stores `oc` to `a.o` (`:267-268`) in place of
`sq32` and the `sm.oc` store, and returns at `:277`, before the cluster exchange and the norm. For MAXT = 2, 4 and 8
its listing has exactly the (line, opcode with modifiers) set of `rec_kernel<MAXT>` minus the norm's rows (`:42`,
`:48-50`, `:119-126`, `:136-137`, `:294`, `q8_1_finite.hpp:19`, `cuda_fp16.hpp:264`). Every row of section 2 up to
`:265-266` applies unchanged; the one new row:

| Line | Expression | SASS | Metal | Subnormal reachable |
|---|---|---|---|---|
| `:265-268` | `oc = (...) * rsqrtf((float) S);` then `a.o[(size_t) t * value_dim + head * S + col] = oc;` | section 2 `:265-266`, then a global store of `oc` | `ieee_mul(((o0 + o1) + o2) + o3, 0x1.6a09e6p-4f)`, stored | yes, as section 2 `:265-266`; `a.o` carries the subnormal to `rec_norm_kernel` |

## 4. `rec_norm_kernel`

`:334-351`: not a template, one warp per (token, value head), `(n_tok * h_v + 3) / 4` blocks of 128 threads
(`:399-404`); every listing carries the same copy. It reads the split kernel's `a.o` and reproduces the fused
kernel's norm: `q[r]` is rank `r`'s `sq32` over the same 32 columns (an XOR butterfly leaves the same bits in every
lane, so lane `t`'s `wsum_t` in `rec_kernel` equals any lane's `q[r]` here), `ss` adds them in the same order as
`:294`, and `norm_q8` is the same code. `rec_split_kernel` then `rec_norm_kernel` therefore writes the bits of
`rec_kernel` (`a.y`, `a.yq`; `a.h`, `a.gate`, `a.beta` come from the shared `rec_block`).

| Line | Expression | SASS | Metal | Subnormal reachable |
|---|---|---|---|---|
| `:340` | `const int t = w / a.h_v, head = w % a.h_v;` | nvcc's integer division sequence (`MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ`, `IMAD.HI.U32`, fix-ups): exact | integer `/` and `%` | n/a |
| `:344` | `oc[r] = a.o[base + 32 * r];` r = 0..3 | loads | loads | the stored `oc` (section 3) |
| `:346` | `q[r] = sq32(oc[r]);` | section 1 `:136-137`, four times | `sq32` | section 1 `:136` |
| `:347` | `const float ss = q[0] + q[1] + q[2] + q[3];` | `FADD a, q0, q1; FADD a, a, q2; FADD a, a, q3` (left to right, no FTZ); `norm_q8`'s `FFMA x, ss, 0.0078125, eps` and `rsqn(x)` are hoisted out of the `r` loop and attributed here | `((q0 + q1) + q2) + q3`, then section 1 `:119` once | as section 1 `:136` |
| `:349-350` | `norm_q8(a, t, head, r, lane, oc[r], ss, a.gamma[32 * r + lane], a.z[base + 32 * r]);` r = 0..3 | section 1 `:116-131` four times; the loop is unswitched on `a.y` (`BRA.U`): two copies, with and without the `a.y` store, with the same float instructions; the `rcp_rn` slow path's rows are attributed to `:351` | section 1 | section 1 |

## 5. `rec_seg_kernel<MAXT>` and the comparison across instantiations

`:315-332`: `__cluster_dims__(4, 1, 1) __launch_bounds__(S)`. Cluster row `blockIdx.y` reads its `SegRow`, returns
unless `r.j == 0`, points the state at `r.gdn + state_off` and the row arrays at row `r.row0` (integer address
arithmetic), and runs `rec_block<MAXT, false>(a, r.n, ...)`: section 2 with the segment's `T = r.n`. The host
(`:369-383`) picks MAXT from the largest `n`, as `dec_gdn_rec` does from `n_tok`.

| Line | Expression | SASS | Metal | Subnormal reachable |
|---|---|---|---|---|
| `:320-321` | `const SegRow r = rows[blockIdx.y]; if (r.j != 0) return;` | a load and an integer test; no float instruction (none of `:315-332`'s rows is a float instruction in any `rec_seg_kernel` listing) | the same early exit per window row, the rows read from a buffer (invariant 4) | n/a |
| `:325-330` | `a.state = r.gdn + state_off; a.qkv += r0 * a.C; ...` | integer address arithmetic | buffer offsets | n/a |
| `:331` | `rec_block<MAXT, false>(a, r.n, blockIdx.x / 4, blockIdx.x % 4, sm, &cl);` | section 2's rows with `T = r.n` | section 2 | section 2 |

Comparison per listing, per function, per (file:line, opcode with every modifier), the `HFMA2` constant moves
excluded: `rec_kernel<4>`, `rec_kernel<8>` and `rec_seg_kernel<2>`, `<4>`, `<8>` have exactly the set of
`rec_kernel<2>`; `rec_split_kernel<2>`, `<4>`, `<8>` have that set minus the norm's rows (section 3);
`rec_norm_kernel` is the same in every listing. The counts grow with MAXT only in the unrolled per-token loops
(`:64`: 24 rows at MAXT 2, two tokens by three channels by four instructions, 96 at MAXT 8). The instantiation the
host picks for a window changes how many unrolled copies exist, not a token's arithmetic; the conv copies `t = 0` and
`t = 1` of `rec_kernel<2>` and the per-token norm copies were compared row by row, the rest by the set comparison.

## 6. Answers: log1pf, __expf / rsqrtf, the q8_1 against native_quantize_q8_1, FFMAs the source does not write

### 6a. `log1pf` at `:111` is libdevice's `__nv_log1pf`, inlined

`:111` is `const float sp = u > 20.0f ? u : log1pf(__expf(u));`. A probe compile with the same compiler and
userArguments of three kernels, `log1pf(x)`, `extern "C" __nv_log1pf(x)` and the `:111` expression, gives the same
SASS for the first two (same constants, same order), and the third is the `:111` block of every `rec_kernel`,
`rec_split_kernel` and `rec_seg_kernel` listing. So `log1pf` is libdevice's `__nv_log1pf`, inlined; the IEEE build
adds no `.FTZ` and substitutes nothing. Step by step, as `rec_kernel<2>` runs it (the PTX gives the constants in
hex; `x` is `__expf(u)`; every float operation rounds to nearest even and keeps subnormals, except the `.RZ` add):

```
// :111  FSETP.GT.AND P0, PT, u, 20, PT ; @P0 BRA -> sp = u            (u > 20; NaN takes the log1pf path)
// x = __expf(u) = ex2n(fl(u * 0x3FB8AA3B))                              FMUL, then ex2n (Notation)
float t  = x + 1.0f rounded toward zero;                    // FADD.RZ t, x, 1
int   k  = (bits(t) - 0x3F400000) & 0xFF800000;             // IADD3 k, t, -0x3f400000 ; LOP3.LUT k, k, 0xff800000, RZ, 0xc0
float m  = from_bits(bits(x) - k);                          // IADD3 m, x, -k          (integer)
float s  = from_bits(0x40800000 - k);                       // IADD3 s, -k, 0x40800000 (integer)
s = fma(s, 0.25f, -1.0f);                                   // FFMA s, s, 0.25, -1     (0.25 = 0x3E800000, an HFMA2 constant move)
m = m + s;                                                  // FADD m, m, s
float kf = (float) k;                                       // I2FP.F32.S32 (exact: k is a multiple of 2^23)
kf = kf * from_bits(0x34000000);                            // FMUL kf, kf, 1.1920928955078125e-07 (2^-23)
float p = fma(m, from_bits(0xBD39BF78), from_bits(0x3DD80012));  // FFMA p, m, -R(0x3d39bf78), 0.10546888411045074463
p = fma(p, m, from_bits(0xBE0778E0));                       // -0.13229703903198242188
p = fma(p, m, from_bits(0x3E146475));                       //  0.14491446316242218018
p = fma(p, m, from_bits(0xBE2A68DD));                       // -0.16641564667224884033
p = fma(p, m, from_bits(0x3E4CAF9E));                       //  0.19988867640495300293
p = fma(p, m, from_bits(0xBE800042));                       // -0.25000196695327758789
p = fma(p, m, from_bits(0x3EAAAAE6));                       //  0.33333510160446166992
p = fma(p, m, from_bits(0xBF000000));                       // -0.5
p = m * p;                                                  // FMUL p, m, p
float r = fma(m, p, m);                                     // FFMA r, m, p, m
r = fma(kf, from_bits(0x3F317218), r);                      // FFMA r, kf, 0.69314718246459960938, r  (ln 2)
if (bits(x) >= 0x7F800000u) {                               // ISETP.GE.U32 P0: +inf, NaN, sign bit set
    if ((int) bits(x) > (int) 0xBF800000) r = fma(x, +INF, +INF);   // ISETP.GT P2; @P2 FFMA: +inf, NaN, x < -1, -inf
    if (x == 0.0f) r = -0.0f;                               // @P0 FSETP.NEU P1, x, RZ ; @P0 FSEL r, r, -RZ, P1 (IEEE ==)
}
// :112  FMUL g, sa, r   (sp * sa)
```

In this kernel `x` lies in [+0, e^20] or is NaN (`u` NaN): the special block runs only for NaN, where `fma(NaN, inf,
inf)` returns CUDA's canonical NaN `0x7FFFFFFF` (`cuda_nan()` in Metal). For a subnormal `x` (`u` below about
-87.34, which `ex2n` keeps) the sequence returns `x` exactly: `t = 1`, `k = 0`, `m = x + 0 = x`, every Horner step
rounds to its constant, `m * p` is a subnormal, `fma(m, p, m) = m`, `fma(0, ln 2, m) = m`. On Metal `m + s` reads
that subnormal as zero and the result would be `+0`, so the helper must return a subnormal `x` unchanged (a bits
test); for a normal `x` Metal's flush cannot change a bit (`m * p` is subnormal only for `m < 2^-125`, where `fma(m,
p, m) = m` both ways). Metal has no per-instruction rounding mode: for `x >= 0`, `t` is the round-to-nearest sum
stepped down one ulp (`bits - 1`) when the exact error of that sum (Fast2Sum of `max(x, 1)` and `min(x, 1)`, exact
and never subnormal here) is negative. The `.RZ` matters: `k` changes at `t = 0.75 * 2^n`, and a round-to-nearest
sum reaches such a value from below where the truncated one does not. Whether the port transcribes `__nv_log1pf`
(credited, like `__nv_exp` and `__nv_logf` on the sampler branch) or writes its own helper is the plan's choice;
only the transcription gives these bits for every `x` of the kernel's domain.

### 6b. `__expf` and `rsqrtf` in this IEEE build

Every `__expf` is `ex2.approx.f32` without `.ftz` and every `rsqrtf` is `rsqrt.approx.f32` without `.ftz` (PTX);
ptxas emits `ex2n` and `rsqn` (Notation): `MUFU.EX2` with the halving below `-126` and the squaring of the result,
`MUFU.RSQ` with the `2^24` prescale of a subnormal or zero argument and the `2^12` postscale. No `.FTZ` on any of
them, and the scaling multiplies are plain `FMUL`.

| Site | Source | SASS | Metal | Subnormal reachable |
|---|---|---|---|---|
| `:65` | `__expf(-sum)` in `sum / (1.0f + __expf(-sum))` | `FMUL e, sum, -1.4426950216293334961` (`0xBFB8AA3B`), `ex2n(e)` | `expf_neg(sum)` (`basalt_fp.metal:104`) | result subnormal for `sum > 87.34`: immaterial, `1.0f + e` is `1` both ways |
| `:109` | `__expf(-v)` in `1.0f / (1.0f + __expf(-v))` | as `:65` with `v` | `expf_neg(v)` | immaterial, as `:65` |
| `:111` | `__expf(u)` | `FMUL e, u, 1.4426950216293334961` (`0x3FB8AA3B`), `ex2n(e)` | `ex2n` helper: `e < -126 ? ieee_mul(r, r)` with `r = fast::exp2(e * 0.5f)`, else `fast::exp2(e)` | yes, material: `u < -87.34`; the subnormal goes through `log1pf` unchanged into the gate |
| `:120` | `__expf(-z)` | as `:65` with `z` (scheduled early) | `expf_neg(z)` | immaterial, as `:65` |
| `:248` | `__expf(sm.g[t])` | `FMUL e, gate, 1.4426950216293334961`, `ex2n(e)` | `ex2n` helper | yes, material: a gate in about (-103.3, -87.34) gives a subnormal decay `g` |
| `:119` | `rsqrtf(ss / (float) S + a.eps)` | `FFMA x, ss, 0.0078125, eps`, `rsqn(x)` | `fast::rsqrt(fma(ss, 0x1p-7f, eps))` | no while `eps >= 2^-126`: `x >= eps` |
| `:228`, `:229` | `yq[t] *= rsqrtf(ssq + a.eps)`, same for `yk` | `FADD x, ssq, eps`, `rsqn(x)`, `FMUL y, r, y` | `ieee_mul(y, fast::rsqrt(ssq + eps))` (section 2) | no while `eps >= 2^-126` |
| `:266` | `rsqrtf((float) S)` | no MUFU: folded to `0.08838834613561630249` (`0x3DB504F3`, the correctly rounded `1/sqrt(128)`), one `FMUL` | the literal `0x1.6a09e6p-4f`, never `fast::rsqrt(128.0f)` | no |

A subnormal argument of `__expf` (a subnormal `sum`, `v`, `u`, `z` or gate times `log2 e`) gives `ex2 = 1` both
ways, so only subnormal results matter. `rsqn`'s prescale would matter only for `eps < 2^-126` with a tiny sum of
squares.

### 6c. `norm_q8`'s q8_1 against `native_quantize_q8_1`

Not the same instruction sequence; the same divisions.

| Step | `norm_q8`, this IEEE build | `native_quantize_q8_1` (fast-math, matvec.md section 1) |
|---|---|---|
| `fabsf(yv)` `:122` | `FADD a, \|yv\|, -RZ`; the first max reads `\|yv\|` directly | `FADD.FTZ` |
| max butterfly `:125`, offsets 16, 8, 4, 2, 1 | `FMNMX a, a, recv, !PT` x5 | `FMNMX.FTZ` x5 |
| sum butterfly `:126` | `FADD s, s, recv` x5 | `FADD.FTZ` x5 |
| `div_approx_ftz(amax, 127.0f)` `:128`, `:42` | `FMUL.FTZ d, amax, 0.0078740157186985015869` (`0x3C010204`) | the same |
| `q8_1_finite(d)` | `FSETP.GT.AND \|d\|, 65504`, `LOP3.LUT` copysign, `FSEL` | `FSETP.GT.FTZ.AND`, the same |
| `amax == 0.0f` `:48` | `FSETP.NEU.AND` | `FSETP.NEU.FTZ.AND` |
| `div_approx_ftz(xi, d)` `:49`, `:42` | `MUFU.RCP r, d; FMUL.FTZ q, yv, r` | the same |
| `roundf` `:49` | `LOP3.LUT` copysign(0.5, q), `FADD.RZ`, `FRND.TRUNC` | `FADD.FTZ.RZ` |
| clamp, `(int8_t)` `:50` | `FSETP.GT.AND q, 127`, `FMNMX.NAN q, -127, !PT`, `FSEL 127`, `F2I.TRUNC.NTZ` | `.FTZ` on the compare and the `F2I` |
| `q8_1_ds(d, sum)` `:131` | `q8_1_finite` on both, `F2FP.F16.F32.PACK_AB` | the same |

ptxas turns the approximate division by 127 into a multiply by `fl(1/127) = 0x3C010204`, the correctly rounded
reciprocal (`|1/127 - fl| = 2.9e-11 <` half an ulp, `4.7e-10`), with `.FTZ`, in both builds; the per-code division
is `MUFU.RCP` and an `FMUL.FTZ` in both. The unflagged abs, max, zero test, round and conversion cannot change a
bit: a subnormal `amax` gives `d = 0` through the flushing multiply, and its codes are `F2I(NaN) = 0` (a subnormal or
zero `yv` times `rcp(0) = inf`) where the fast-math build tests `amax == 0` true, so codes and `ds.x` agree for
every input; a subnormal `yv` beside a normal `amax` is flushed by `FMUL.FTZ` in both. The unflagged sum is the one
difference. The IEEE and the flushed butterfly part only where a subnormal operand or result occurs, which needs a
partial sum below `2^-102` (subnormal `yv`, or tiny ones that cancel); the half of a sum of magnitude `2^-25` or less
is a signed zero, so there the difference is the sign bit of `ds.y` (`[-2^-140, +0, ...]`: IEEE `-2^-140`, half `-0`;
flushed `-0 + +0 = +0`, half `+0`). A larger sum inherits a difference only through exact rounding ties in the later
butterfly levels and in the half conversion; model data does not produce that. Metal: `q8_1_store_native` (`basalt_q8_1.metal:52`) with its sum taken by `warp_sum_ieee`
(`:206`) in place of `warp_sum_ftz`. `q8_1_store_ieee` (`:211`) is a different operation (correctly rounded
`amax / 127` and `xi / d`) and does not apply.

Invariant 5's integer-key compare is not needed at the two IEEE compares here that can see a subnormal. At `:48`
(`amax == 0.0f`) a subnormal `amax` gives the codes 0 both ways (above). At `:125` (`fmaxf`), Metal's zero-reading
compare can pick a different subnormal among subnormals and zeros, but a normal operand always beats a subnormal one,
so the warp's `amax` is CUDA's whenever CUDA's is normal, and is a subnormal or zero (codes 0, `d = 0`) exactly when
CUDA's is.
NaN operands are `q8_1_store_native`'s case unchanged: `FMNMX` and `FMNMX.FTZ` differ only in the flush. Every other float compare in the file has a normal constant on
one side: `:50` (`±127`, on an `FMUL.FTZ` product), `:111` (`u > 20`), `q8_1_finite`'s `65504`, and `ex2n`'s `-126`.
Reading a subnormal as zero cannot change those outcomes. `__nv_log1pf`'s `x == 0` (6a) sees only NaN in this kernel.

### 6d. FFMAs the source does not write

| Site | Source | SASS | Metal |
|---|---|---|---|
| `:64` | `float sum = v0 * w[0] + v1 * w[1] + v2 * w[2] + x * w[3];` | `FMUL a, w1, v1; FFMA a, w0, v0, a; FFMA a, w2, v2, a; FFMA sum, w3, x, a` | `fma(x, w3, fma(v2, w2, fma(v0, w0, v1 * w1)))`: the product left unfused is the second, `v1 * w[1]` (LLVM fuses the first product of `fadd(fmul, fmul)`) |
| `:119` | `ss / (float) S + a.eps`, `S = 128` | `FFMA x, ss, 0.0078125, eps`: the division by 128 becomes a multiply by `2^-7` (exact for every `ss`, NVVM) and is contracted with the add | `fma(ss, 0x1p-7f, eps)`, one rounding |
| `:255` | `const float delta = (v - g * kv_col) * b;` | `FFMA t, -g, kv_col, v; FMUL delta, t, b` | `fma(-g, kv_col, v) * b` |
| `:65` | `sum / (...)` | `div_rn`: five FFMAs refine `MUFU.RCP` to the correctly rounded quotient; `FCHK` routes the rest to `__cuda_sm3x_div_rn_noftz_f32_slowpath` | the correctly rounded `/`; `ieee_div` where a subnormal can occur |
| `:109`, `:120` | `1.0f / (...)` | `rcp_rn`: two FFMAs refine `MUFU.RCP`; `__cuda_sm20_rcp_rn_f32_slowpath` out of line | the correctly rounded `1.0f / d`; `ieee_rcp` where the result can be subnormal |
| `:111` | `log1pf(...)` | the twelve FFMAs of `__nv_log1pf` (6a) | the 6a sequence |

Every other FFMA is an explicit `fmaf` of the source (`:103-106`, `:251`, `:259`, `:260`); where a chain starts
from `float v = 0.0f` the first one is `FFMA v, a, b, RZ`. nvcc never reassociates a sum: every `FADD` chain below
keeps the source's left-to-right order.
