# basalt's verify-window GDN kernels as nvcc compiles them

What NVIDIA's compiler emits at every float site of the GDN kernels of `src/kernels/cuda/verify_kernels.cu`, read from
the sm_120 PTX and SASS, and the Metal form the port writes at each site: the per-stage kernels that `dec_gdn_rec`
fuses and that basalt's parity tests run as the reference side (`gdn_conv_l2_multi_kernel`, `gdn_ab_multi_kernel`,
`gdn_ab_multi_unrolled_kernel`, `gdn_step_norm_multi_kernel`), the conv-history commit (`gdn_conv_commit_kernel`),
the commit replay that writes the kept tokens into the recurrent state (`gdn_commit_layers_pipe_kernel`, contract
invariant 3) and the PLE history copy (`ple_hist_seg_kernel`). `dec_gdn.md` is the companion page for the fused
kernel; this page is the authority the Metal kernels follow (`docs/design/flash-next.md`, invariant 5).

Date: October 18, 2026. Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b`
means lines a to b there. A bare `:N` is a line of `src/kernels/cuda/verify_kernels.cu`; `rec.cu:N` is a line of
`src/kernels/cuda/dec_gdn_rec.cu`.

## Port mapping

The SASS columns are the record; the Metal columns are this page's proposals, written before the port under section 2's
names. The port writes the per-stage kernels in `kernels/common/basalt_gdn.metal` as `dec_gdn_parity`'s test-only
reference side (`gdn_conv_l2_multi`, one `gdn_ab_multi` for basalt's five instantiations, `gdn_step_norm_multi` in its
verify form), on the shared steps that `dec_gdn_rec` calls too, and settles section 9's four points with one helper per
operation wherever a subnormal can occur. Every `__expf` is `expf_ieee` (`kernels/common/basalt_ieee.metal`), with
`__expf(-x)` as `expf_ieee(sign_flip(x))`, the bits of the folded `0xBFB8AA3B` product; every run-time `rsqrtf` is
`rsqrt_ieee` (`rsqn` with its prescale and postscale, not `metal::fast::rsqrt` alone), and `rsqrtf((float) S)` the
literal `0x3DB504F3`; `1.0f / (1.0f + e)` and the SiLU's quotient are `ieee_div`; `log1pf` is `log1pf_nv`
(`kernels/common/basalt_libdevice.metal`; `log1pf_cuda` here: section 2's sequence, its twelve `FFMA` as `fma`). Every
other `FFMA` is `fma_ieee` (`kernels/common/basalt_ieee_f64.metal`; `ieee_fma_any` here, a helper of its own rather than
the sampler's `ieee_fma_count`), the three contractions (`:45`, `:414`, `:439`) and the chains SASS starts from `RZ` (on
`0.0f`) included, and every other unflagged `FMUL`, `FADD` and butterfly is `ieee_mul`, `ieee_add` and `warp_sum_ieee`,
`a + dt[r]` included. The stored h, gate, beta and y go through `cuda_nan` (`nan.md`). The commit kernels' sections (4
and 7) keep their proposals until the commit is ported.

## 1. Build, compiler and method

- **Sources.** `src/kernels/cuda/verify_kernels.cu` (1114 lines) with the basalt headers it includes
  (`include/basalt/kernels/verify_kernels.hpp` and those it pulls in) inlined under `#line` directives, so every PTX and
  SASS row names its basalt `file:line`. Only basalt's public MIT source was sent.
- **Flags.** `verify_kernels.cu` is a source of `add_library(kernels ...)` (`CMakeLists.txt:161`; `dec_gdn_rec.cu` is
  `:162`) and is not in the `--use_fast_math` list (`CMakeLists.txt:187-202`), so it builds with nvcc's IEEE defaults:
  `-ftz=false -prec-div=true -prec-sqrt=true -fmad=true`. Architecture 120 (`CMakeLists.txt:3-4`), the Release build
  (`:12-13`) and `BASALT_VERSION` (`:6-7`) give the arguments `-arch=sm_120 -lineinfo -O3 -DNDEBUG
  -DBASALT_VERSION="0.0.1"`, identical in both compiler versions.
- **Compilers.** Compiler Explorer's public API, as in `matvec.md`: NVCC 13.0.0 (`nvcc130`) is the record, 13.3.0
  (`nvcc133`) the version check; PTX and the `SASS (sm_120)` listing.
- **Translation units.** The whole file overflows the listing's 5,000 rows, so each unit is the file with the other
  kernels' lines and the host launchers (`:609-1113`) blanked (a blanked line stays empty, so every line keeps its
  number), and a table appended that takes the address of the instantiations the unit emits: `conv`
  (`gdn_conv_l2_multi_kernel`, `gdn_conv_l2_seg_kernel`, `gdn_conv_commit_kernel`, `ple_hist_seg_kernel`), `ab4` and
  `ab8` (`gdn_ab_multi_kernel<4>`, `<8>`), `abu4`, `abu6` and `abu8` (`gdn_ab_multi_unrolled_kernel<4, 10>`, `<6, 10>`,
  `<8, 10>`), `stepnorm` (`gdn_step_norm_multi_kernel`), `stepnormseg` (`gdn_step_norm_seg_kernel`), `stepcommit`
  (`gdn_step_commit_kernel`), `commitpipe` (`gdn_commit_layers_pipe_kernel`) and `commitseg`
  (`gdn_commit_layers_seg_kernel`). These are the instantiations the launchers make: `gdn_ab_multi` (`:873-898`) takes
  `unrolled<4|6|8, 10>` for `n_embd == 2560` (32 lanes x 8 values x 10) and `gdn_ab_multi_kernel<4|8>` for any other
  width; `gdn_step_norm_multi` (`:900-916`) launches `gdn_step_commit_kernel` instead of the step-norm kernel when
  `n_keep != nullptr && t_out_begin >= n_tok`. Every response has code 0 and a complete SASS listing.
- **Where.** The units, requests, raw responses and listings are outside the repository, in the lane's scratch
  (`.context/gdn/nvcc/verify/nvcc130/` and `nvcc133/`; `dec_gdn_rec`'s in `.context/gdn/nvcc/nvcc130/`, as
  `dec_gdn.md`'s Method says). A listing row cited below (`conv.sass.txt` rows 555-812) is a row of that unit's 13.0
  SASS listing as Compiler Explorer returns it.
- **Reading.** Every float instruction of each kernel is mapped to its source line through the `-lineinfo` rows; the
  sequences quoted below are verbatim SASS, with the register's meaning in parentheses. A contraction is settled by
  the SASS, not the PTX: ptxas fuses an uncontracted `mul.f32` with the add or subtract that consumes it
  (`dec_gdn.md`, Method), as at `:248`, `:323` and `:414` below.
- **Version check.** Per kernel and source line, the multiset of float operations with the registers stripped is the
  same in 13.0 and 13.3 for every unit. The differences are moves (`HFMA2` loading a constant, `R2UR`), the placement
  of the integer-division emulation that computes loop bounds and indices, and the datapath of the decay's `ex2`
  sequence (uniform `UFMUL`/`UFSETP` against vector `FMUL`/`FSETP`, in either direction between the versions): the same
  IEEE operations on the same operands, the same bits. The record holds for both versions.

## 2. Shared sequences

These sequences recur at several sites and compile identically at each, registers aside; the per-kernel tables use
their names. The names and sequences are `dec_gdn.md`'s (Notation), the same instructions at the same source text.
Metal helper names follow the team's spec (`.context/gdn/spec-repo.md`): `expf_ieee` and `log1pf_cuda` are the prompt
lane's names, `ieee_fma_any` the general IEEE fma, which origin/main lacks and `origin/woodsonl/basalt-sampler`
has as `ieee_fma_count` (section 9). On origin/main, `ieee_add` (`:95`),
`ieee_mul` (`:108`), `ieee_rcp` (`:153`), `ieee_fma` (`:167`), `ieee_div` (`:181`) and `warp_sum_ieee` (`:206`) are in
`kernels/common/basalt_q8_1.metal`, `warp_sum` (`:97`) and `expf_neg` (`:104`) in `kernels/common/basalt_fp.metal`.

- **`__expf(x)`**: `FMUL y, x, 1.4426950216293334961` (`0x3FB8AA3B`, no FTZ), then `ex2n(y)` =
  `FSETP.GEU.AND P0, PT, y, -126, PT; @!P0 FMUL y, y, 0.5; MUFU.EX2 r, y; @!P0 FMUL r, r, r`. For `__expf(-x)` the
  negation is folded into the constant, `FMUL y, x, -1.4426950216293334961` (`0xBFB8AA3B`), the same bits as
  `(-x) * 0x3FB8AA3B`. The decay of the step-norm kernels (`:407`) and of the pipe commit (`:315`) runs on the
  uniform datapath (`UFMUL`, `UFSETP`, `MUFU.EX2` of a uniform register, `R2UR`, `@!UP1 UFMUL` for the square): the
  same IEEE operations on the same operands.
  - Metal: `expf_ieee(x)`: `y = x * L2E; small = y < -126.0f; if (small) y *= 0.5f; r = metal::fast::exp2(y); if
    (small) r = ieee_mul(r, r);` with `L2E = as_type<float>(0x3FB8AA3Bu)`; for `__expf(-x)`, `y = x *
    as_type<float>(0xBFB8AA3Bu)`. `fast::exp2` stands for `MUFU.EX2` (the contract's mapping); the halving and the
    squaring are CUDA's, and only `ieee_mul` keeps a square in the subnormal range, which Metal's `*` flushes. NaN
    gives `small` false in both (`GEU` is true for NaN). `expf_neg` lacks the halving and gives the same bits only
    where 1 is added to the result; one helper per operation (contract) makes it `expf_ieee` at every site.
  - Subnormal: the product `x * L2E` is subnormal only for |x| < 8.2e-39, and `ex2` of it is 1.0 whether it is
    flushed or not: immaterial. The result is subnormal for y in [-150, -126), x in (-103.97, -87.34): material where
    the result is used as it is (the decay `__expf(gate)`, `:241`, `:315`, `:407`; the softplus's `__expf(v)`,
    `:154`, `:209`), immaterial where `1.0f + e` follows (`:46`, `:76`, `:151`, `:206`, `:441`).
- **`rsqrtf(x)`**: `rsqn(x)` = `FSETP.GEU.AND P0, PT, |x|, 1.175494350822287508e-38, PT; @!P0 FMUL x, x, 16777216;
  MUFU.RSQ r, x; @!P0 FMUL r, r, 4096`: a zero or subnormal argument is scaled by 2^24 and the result by 2^12.
  - Metal: `metal::fast::rsqrt(x)` (the contract's mapping). The scaling acts only on a zero or subnormal argument,
    and every run-time `rsqrtf` here has an argument of at least `eps` (`:53`, `:83`, `:439`): dead for a normal
    positive `eps`.
  - `rsqrtf((float) S)` (`:431`): PTX `mov.f32 %f416, 0f43000000; rsqrt.approx.f32 %f417, %f416;`, folded by ptxas
    into `FMUL oc, x, 0.08838834613561630249` = `0x3DB504F3`, the correctly rounded 2^-3.5 (`rec.cu:266` folds the
    same). Metal writes the constant, `as_type<float>(0x3DB504F3u)`, not `fast::rsqrt(128.0f)`.
- **`1.0f / d`**: `rcp_rn(d)` = `IADD3 t, d, 0x1800000; LOP3.LUT t, t, 0x7f800000, RZ, 0xc0; ISETP.GT.U32.AND P0,
  PT, t, 0x1ffffff, PT`, fast path `MUFU.RCP r, d; FFMA e, d, r, -1; FADD.FTZ e, -e, -RZ; FFMA r, r, e, r`, else
  `CALL __cuda_sm20_rcp_rn_f32_slowpath`: the correctly rounded reciprocal, subnormal results kept (the `.FTZ` acts
  on the residual `d * r - 1`, zero or far above the subnormal range).
  - Metal: `ieee_div(1.0f, d)` (documented for `b >= 1`, infinite or NaN; it keeps a subnormal quotient). At every
    site d = `1.0f + e` with e >= 0. Not `ieee_rcp`, documented for |s| <= 2^126 (d in (2^126, inf) has a subnormal
    reciprocal), and not Metal's `1.0f / d`, which flushes it.
  - Subnormal: the result is subnormal for d > 2^126, a logit below -87.34: material.
- **`n / d`** (`:46`, `:76`): `div_rn(n, d)` = `MUFU.RCP r, d; FCHK P0, n, d; FFMA t, -d, r, 1; FFMA r, r, t, r; FFMA
  q, n, r, RZ; FFMA e, -d, q, n; FFMA q, r, e, q; @P0 CALL __cuda_sm3x_div_rn_noftz_f32_slowpath`: the correctly
  rounded IEEE quotient, subnormals kept. Metal: `ieee_div(n, d)` (d = `1.0f + e` >= 1).
- **`log1pf(e)`** (`:154`, `:209`): libdevice's `__nv_log1pf`, inlined (a probe compile of the bare `log1pf` and of
  `__nv_log1pf` gives the same SASS, `.context/gdn/nvcc/probe/`). `gdn_ab_multi_kernel<4>` (`ab4`), verbatim with
  register names, constants in hex:

  ```
  FADD.RZ t, e, 1
  ISETP.GE.U32.AND P0, PT, e, 0x7f800000, PT          // +inf, NaN or sign set
  IADD3 t, PT, PT, t, -0x3f400000, RZ
  ISETP.GT.AND P2, PT, e, -0x40800000, P0              // P0 and (int) bits(e) > (int) 0xBF800000
  LOP3.LUT k, t, 0xff800000, RZ, 0xc0, !PT             // k = (bits(t) - 0x3F400000) & 0xFF800000
  @P0 FSETP.NEU.AND P1, PT, e, RZ, PT
  IADD3 s, PT, PT, -k, 0x40800000, RZ                  // s = as_float(0x40800000 - k)
  IADD3 m, PT, PT, e, -k, RZ                           // m = as_float(bits(e) - k)
  I2FP.F32.S32 kf, k
  FFMA s, s, 0x3E800000, -1                            // s * 0.25 - 1
  FMUL kf, kf, 0x34000000                              // * 2^-23
  FADD m, m, s
  FFMA p, m, 0xBD39BF78, 0x3DD80012
  FFMA p, m, p, 0xBE0778E0
  FFMA p, m, p, 0x3E146475
  FFMA p, m, p, 0xBE2A68DD
  FFMA p, m, p, 0x3E4CAF9E
  FFMA p, m, p, 0xBE800042
  FFMA p, m, p, 0x3EAAAAE6
  FFMA p, m, p, 0xBF000000
  FMUL p, m, p
  FFMA r, m, p, m
  FFMA r, kf, 0x3F317218, r                            // + k * ln 2
  @P2 FFMA r, e, +INF, +INF
  @P0 FSEL r, r, -RZ, P1
  ```

  (The SASS writes the first coefficient as `-R18` with `MOV R18, 0x3d39bf78`, and 0.25 as `HFMA2 R17, -RZ, RZ,
  1.625, 0`, a constant move.) No FTZ anywhere: `FADD.RZ` rounds toward zero, every other operation to nearest.
  - Metal: `log1pf_cuda(e)`, an exact transcription with three changes of form: (1) when e's exponent field is zero
    (a zero or a subnormal e) return e, which the sequence returns there (`FADD.RZ` gives 1.0, so k = 0, s = 0 and
    `FADD m, m, s` keeps m = e; the Horner chain on a tiny m ends at exactly -0.5; `FMUL` gives -m/2; `FFMA r, m, p,
    m` gives m; adding 0 * ln 2 gives m), where Metal's `FADD` would read e as zero; (2) `add.rz` emulated, Metal's
    `+` rounding to nearest even; (3) the special branch kept: here e = `__expf(v)` with v <= 20 (the select) is in
    [0, 4.86e8] or NaN, so only a NaN reaches it, and `FFMA r, e, +INF, +INF` gives CUDA's canonical NaN
    (`cuda_nan()`). For a normal e the rest is plain Metal arithmetic: the one product that can leave the normal
    range, `FMUL p, m, p` for m below 2^-125, is added to m and cannot change it, flushed or not.
  - Subnormal: e is subnormal for v in (-103.97, -87.34): material (the sequence returns it, and the stored gate is
    `sp * ssm_a`).
- **Warp butterfly** `for (o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o)`: `SHFL.BFLY r, v, o,
  0x1f; FADD v, v, r` for o = 16, 8, 4, 2, 1, no FTZ (some kernels order the operands (r, v): IEEE addition is
  commutative bit for bit). Metal: `warp_sum_ieee` (`x = ieee_add(x, shfl_xor(x, o))`, the same order) at every
  butterfly of these kernels (`:49`, `:79`, `:148`, `:203`, `:434`): a partial can be subnormal at each.
- **Four partial sums** `p0 + p1 + p2 + p3` (`:52`, `:82`, `:247`, `:321-322`, `:413`, `:431`, `:438`): `FADD t, p0,
  p1; FADD t, t, p2; FADD t, t, p3`, no FTZ, the source's left-to-right order. Metal: the same order with `ieee_add`,
  a subnormal partial being reachable at each.
- **Contractions the source does not write.** NVVM contracts a product into the add or subtract that consumes it
  (`-fmad=true`): the conv's first add `v0 * w0 + v1 * w1` (`:45`, `:75`) keeps `v1 * w1` rounded and fuses `v0 *
  w0`; `v - g * kv_col` (`:248`, `:323`, `:414`) is one `FFMA d, kv_col, -g, v`; `ss / (float) S + eps` (`:439`)
  becomes `FFMA d, ss, 0.0078125, eps` (the division by 128 an exact multiply by 2^-7). The Metal source writes each
  as `fma()` with these operands; every other product is rounded on its own (`#pragma STDC FP_CONTRACT OFF`).
- **Not float data.** `I2F.U32.RP`/`MUFU.RCP`/`F2I.FTZ.U32.TRUNC` and `I2F.U64.RP`/`MUFU.RCP`/`F2I.U64.TRUNC` (and
  `__cuda_sm20_div_u64`) are nvcc's unsigned integer division (`head % h_k`, the commit's item index, `ple_hist`'s
  trip count): exact integer results, Metal's integer `/` and `%`. `HFMA2 r, -RZ, RZ, c` loads a constant; `R2UR`
  and `MOV` move bits.

## 3. `gdn_conv_l2_multi_kernel`

`:29-56`, `__launch_bounds__(S)` with S = 128 (`:20`): one thread per channel `c = blockIdx.x * S + threadIdx.x` of
token `t = t_begin + blockIdx.y`; block `blockIdx.x` is one head of 128 channels, the first `qk_heads` blocks being q
and k heads. The window is `win[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c]`, `src = t + j`,
j < 3; `v0`, `v1`, `v2` are `win[0..2]`, `x = qkv[(size_t) t * C + c]` (`:44`), the conv weights `w[c * 4 + 0..3]`
(`w0` to `w3`). Listing: `verify/nvcc130/conv.sass.txt` rows 555-812. `gdn_conv_l2_seg_kernel` (`:59-86`, rows
281-554), the segmented batch form, has the same source text at `:75`, `:76`, `:78`, `:79`, `:82`, `:83` and the same
instruction sequence (div slow path and rsqrt scaling included); only its addressing differs (`SegRow`, `hist`
offset by the row's `gdn` pointer).

| Line | Expression | SASS (verbatim, source names) | Metal | Subnormal reachable |
| --- | --- | --- | --- | --- |
| `:45` | `float sum = v0 * w[c * 4] + v1 * w[c * 4 + 1] + v2 * w[c * 4 + 2] + x * w[c * 4 + 3];` | `FMUL p, w1, v1; FFMA s, w0, v0, p; FFMA s, w2, v2, s; FFMA sum, x, w3, s` | `sum = ieee_fma_any(x, w3, ieee_fma_any(w2, v2, ieee_fma_any(w0, v0, ieee_mul(w1, v1))))` | yes, material: a product of two small values (an activation and a weight below about 1e-19) is subnormal, and it reaches the output through `:46` and `:53` |
| `:46` | `float y = sum / (1.0f + __expf(-sum));` | `FMUL e, sum, -1.4426950216293334961; ex2n(e) -> r; FADD d, r, 1; div_rn(sum, d)` | `y = ieee_div(sum, 1.0f + expf_ieee(-sum))` | yes: a subnormal `sum` gives a subnormal `y` (about `sum / 2`), material; `e` subnormal for `sum` > 87.34, immaterial (1 + e = 1) |
| `:48` | `float sq = y * y;` | `FMUL sq, y, y` | `ieee_mul(y, y)` | yes, material in principle: the square of a |y| below 1.08e-19 is subnormal, kept by CUDA and zero in Metal's `*`; it changes a partial's last bit only while the partial is below 2^-102, and reaching `ss + eps` takes a chain of exact rounding ties through the later additions (constructible, never met by random data); the IEEE helpers cost nine operations per channel and token, so the port keeps CUDA's bits here as well (invariant 5) |
| `:49` | `for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);` | `SHFL.BFLY r, sq, o, 0x1f; FADD sq, sq, r`, o = 16, 8, 4, 2, 1 | `warp_sum_ieee(sq)` | as `:48` |
| `:52` | `const float ss = part[0] + part[1] + part[2] + part[3];` | `LDS.128 p0..p3; FADD a, p0, p1; FADD a, a, p2; FADD ss, a, p3` | `ieee_add(ieee_add(ieee_add(part[0], part[1]), part[2]), part[3])` | as `:48` |
| `:53` | `y *= rsqrtf(ss + eps);` | `FADD x, ss, eps; rsqn(x) -> r; FMUL y, y, r` | `y = ieee_mul(y, fast::rsqrt(ss + eps))` | `ss + eps` >= eps: no (normal eps), so the `rsqn` scaling is dead; `y * r`: yes, material (a subnormal `y` times up to `1 / sqrt(eps)` is a normal output in CUDA and zero in Metal's `*`) |
| `:55` | `h[(size_t) t * C + c] = y;` | `STG` | a store | the stored bits are the value's: a v channel (`blockIdx.x >= qk_heads`) stores `:46`'s subnormal `y` itself |

No FFMA appears in this kernel other than `:45`'s three and the `div_rn` sequence's five (and the slow path's); every
FMUL and FADD is unflagged (IEEE).

## 4. `gdn_conv_commit_kernel`

`:88-111`: `gdn_conv_commit_kernel` (one thread per channel, `c = blockIdx.x * blockDim.x + threadIdx.x`, 256 threads
per block, `:867-871`) calls `gdn_conv_commit_one`, which reads `n = *n_keep` and calls `gdn_conv_commit_n`, the
function the commit replay also runs for its conv items (section 7). Listing: `verify/nvcc130/conv.sass.txt` rows
214-280.

| Line | Expression | SASS | Metal | Subnormal reachable |
| --- | --- | --- | --- | --- |
| `:96` | `seq[j] = src < 3 ? hist[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];` (`src = n + j`, j < 3) | `LDG.E` 32-bit loads; no float instruction (the only one, `HFMA2 R7, -RZ, RZ, 0, 0`, loads a constant) | 32-bit word loads (`uint` or `float`: a move keeps the bits) | yes (any value of the conv input), immaterial: moved, never computed on |
| `:98-100` | `hist[c * 3] = seq[0]; hist[c * 3 + 1] = seq[1]; hist[c * 3 + 2] = seq[2];` | `STG.E` 32-bit stores | 32-bit word stores | as `:96` |

The kernel has no float arithmetic; its bits are the inputs' bits. `n` comes from `*n_keep` unclamped (the replay's
callers clamp it to `kVerifyMaxT`, `:358`, `:366`), so the window's row count bounds the `qkv` reads: the port checks
`n` against its binding.

## 5. `gdn_ab_multi_kernel` and `gdn_ab_multi_unrolled_kernel`

`gdn_ab_multi_kernel<MAX_T>` (`:113-158`, `__launch_bounds__(64)`): two warps per block, one output row per warp,
`row = blockIdx.x * 2 + (threadIdx.x >> 5)`; rows `[0, h_v)` are the alpha rows (`wa`), rows `[h_v, 2 h_v)` the beta
rows (`wb`), each a row of `n` bf16 weights read as `uint4` words (8 weights). Lane `lane` accumulates, per token t
< T, the words `j = lane, lane + 32, ...` below `n / 8` (`:128`), `xa`/`xb` being the 8 activations `x[t * n + 8 j ..
+ 7]` (`:134-135`); the accumulators start at zero (`:127`). `:136-141`, verbatim (the same text with `wv.y`, `wv.z`,
`wv.w` follows):

```
a = fmaf(__uint_as_float(wv.x << 16), xa.x, a); a = fmaf(__uint_as_float(wv.x & 0xffff0000u), xa.y, a);
```

`gdn_ab_multi_unrolled_kernel<MAX_T, NJ>` (`:162-213`) is the same computation for rows of exactly `32 NJ` words
(`NJ = 10`, `n = 2560`): the lane's `NJ` words are loaded before `pdl_wait` (`:175-176`), a row past `2 h_v` loads
row 0 and returns after the wait (`:177`), and the product lines `:191-194` (`j = lane + 32 q`, `q` outer, t inner)
carry the same text. Its per-token order of terms is the generic kernel's (`j` ascending from `lane` in steps of 32),
so for `n = 2560` the two kernels give the same bits; the zeroed accumulator is folded into the first `fma` (PTX
`fma.rn.f32 %f115, %f114, %f106, 0f00000000`, `abu4.ptx.txt:79-102`), the same as an `fma` with a `+0` addend. The
three instantiations the launcher takes agree token for token (`verify/abu_diff.py`, over the 13.0 PTX and SASS): in
`<4, 10>`, `<6, 10>` and `<8, 10>` each token's sum is one chain of 80 `fma.rn.f32` links from `0.0f` in line order
`:191, :191, :192, :192, :193, :193, :194, :194` repeated 10 times, and per source line the count of each float
PTX and SASS opcode is MAX_T times one per-token count, the same in all three (the reciprocal slow path `:213` and the
constant load `:175` once per kernel): MAX_T changes how many tokens a block serves, not a token's arithmetic. Line
numbers below are the generic kernel's, the unrolled kernel's in brackets.

Listings: `verify/nvcc130/ab4.sass.txt` (`<4>`: rows 1-545, the reciprocal slow path rows 546-618; the product chains
rows 62-111, the butterfly rows 129-141, the beta rows 147-169, the softplus rows 182-214, one copy per unrolled
token), `ab8.sass.txt`, `abu4.sass.txt` (the tail `:203-210` rows 693-857), `abu6.sass.txt`, `abu8.sass.txt`. FFMA
counts at the product lines: `<4>` 8 per (j, t), `<8>` 16 per line (`:137-140`, 2 per token), `unrolled<4, 10>` 320
(80 per line `:191-194`), `<6, 10>` 480, `<8, 10>` 640: one FFMA per weight, no other FFMA.

| Line | Expression | SASS (verbatim, source names) | Metal | Subnormal reachable |
| --- | --- | --- | --- | --- |
| `:136-141` [`:191-194`] | the 8 `fmaf` per word and token (above) | `IMAD.U32 wlo, wv.x, 0x10000` (`<< 16`), `LOP3.LUT whi, wv.x, 0xffff0000, RZ, 0xc0`; then per (j, t) the chain in source order `FFMA a, wlo(x), xa.x, a; FFMA a, whi(x), xa.y, a; FFMA a, wlo(y), xa.z, a; FFMA a, whi(y), xa.w, a; FFMA a, wlo(z), xb.x, a; FFMA a, whi(z), xb.y, a; FFMA a, wlo(w), xb.z, a; FFMA a, whi(w), xb.w, a` (`ab4` rows 62-111: `FFMA R21, R35, R12, R26` is `wlo(x) * xa.x + acc[0]`) | `a = ieee_fma_any(as_type<float>(wv.x << 16), xa.x, a)`, then `as_type<float>(wv.x & 0xffff0000u)` with `xa.y`, and so on in the same order; the weight words are moves (a bf16 subnormal stays an fp32 subnormal) | yes, material: a bf16 weight with a zero exponent field is an fp32 subnormal, a product of a small weight and a small activation is subnormal, and a running sum near zero is subnormal; Metal's `fma` reads each as zero (a subnormal weight times a large activation is lost; a subnormal addend under a product at a rounding midpoint changes the rounding) |
| `:148` [`:203`] | `for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);` | `SHFL.BFLY r, a, o, 0x1f; FADD a, r, a`, o = 16, 8, 4, 2, 1 (operands (received, own): commutative, the same bits) | `warp_sum_ieee(a)` | yes: a lane's partial can be subnormal; it changes the sum only when every partial is below about 2^-101, where `a` is lost in `1 + __expf(-a)` (beta = 0.5) and in `a + dt` unless `dt` is as small: immaterial in practice, `warp_sum_ieee` keeps it exact at 5 adds per row |
| `:151` [`:206`] | `beta[(size_t) t * h_v + r] = 1.0f / (1.0f + __expf(-a));` | `FMUL y, a, -1.4426950216293334961; ex2n(y) -> e; FADD d, e, 1; rcp_rn(d)` (`CALL __cuda_sm20_rcp_rn_f32_slowpath` off the fast path) | `ieee_div(1.0f, 1.0f + expf_ieee(-a))` | `e` subnormal (a in (87.34, 103.97)): immaterial (`1 + e` = 1); the reciprocal is subnormal for a < -87.34 (d > 2^126): material, the stored beta (Metal's `1.0f / d` and `ieee_rcp` lose it) |
| `:153` [`:208`] | `const float v = a + dt[r];` | `FADD v, a, dt` | `a + dt[r]` | yes only when `a` and `dt[r]` are both below about 2^-101; then `__expf(v)` = 1 either way: immaterial |
| `:154` [`:209`] | `const float sp = v > 20.0f ? v : log1pf(__expf(v));` | `FSETP.GT.AND P, PT, v, 20, PT` (the select); `FMUL y, v, 1.4426950216293334961; ex2n(y) -> e`; the `log1pf` sequence of section 2 on `e` | `v > 20.0f ? v : log1pf_cuda(expf_ieee(v))` | `e` subnormal for v in (-103.97, -87.34): material, `log1pf` returns e and `sp` = e reaches the stored gate (`:155`); the compare is exact for every v |
| `:155` [`:210`] | `gate[(size_t) t * h_v + r] = sp * ssm_a[r];` | `FMUL g, ssm_a, sp` | `ieee_mul(sp, ssm_a[r])` | yes, material: a subnormal or tiny `sp` times `ssm_a` (an `A`-derived factor of a few units) is stored as it is |

Lane 0 of a row stores one of the two outputs (`:149-150`: the beta rows `:151`, the alpha rows `:153-155`); every
FMUL and FADD is unflagged and the only `.FTZ` is the reciprocal's residual (section 2).

## 6. `gdn_step_norm_multi_kernel`

`gdn_step_norm_multi_kernel` (`:450-461`, `__launch_bounds__(512)`) waits on the previous launch (`pdl_wait`) and runs
`gdn_step_norm_body` (`:378-448`) with `n = n_keep ? *n_keep : T` and `write = n_keep != nullptr`. One block per value
head (`blockIdx.x = head`), 512 threads as (`col` = `threadIdx.x` < 128, `rg` = `threadIdx.y` < 4): thread (rg, col)
holds the state rows `k = 32 rg + r`, r < 32 (`RPG`), of column `col` in registers (`base = state + ((rg * RPG) *
h_v + head) * S + col`, row stride `h_v * S`). Per token t < n the block stages the head's k row (`sk`, L2-normed by
the conv kernel) and, when `t >= t_out_begin`, its q row (`sq`); the v value is read from `hbuf`, the gate and beta
from the `gdn_ab_multi` outputs. Tokens below `t_out_begin` update the state only (`:415-418`); the state is stored
after the loop when `write` (`:444-447`). `n` comes from `*n_keep` unclamped: the port bounds it by the window's T.
`gdn_step_norm_seg_kernel` (`:464-477`, the segmented batch form, `n = r.n`, `write = false`, `t_out_begin = 0`)
has only the output path (`:415-418` removed) and the same instruction sequence at every line below. The launcher
runs `gdn_step_commit_kernel` instead when `n_keep != nullptr && t_out_begin >= n_tok` (`:900-916`; section 7).
Listings: `verify/nvcc130/stepnorm.sass.txt` and `.ptx.txt`; `stepnormseg.sass.txt`.

| Line | Expression | SASS (verbatim, source names) | Metal | Subnormal reachable |
| --- | --- | --- | --- | --- |
| `:407` | `const float g = __expf(gate[(size_t) t * h_v + head]);` | uniform datapath: `R2UR UR6, gate; UFMUL UR6, UR6, 1.4426950216293334961; UFSETP.GEU.AND UP1, UPT, UR6, -126, UPT; @!UP1 UFMUL UR6, UR6, 0.5; MUFU.EX2 R6, UR6; R2UR UR12, R6; @!UP1 UFMUL UR12, UR12, UR12` (= `ex2n`, section 2) | `g = expf_ieee(gate[...])` | yes, material: `gate = sp * ssm_a` <= 0 is unbounded below, and g is subnormal for gate in (-103.97, -87.34); it scales every state entry (`:417`, `:423`), where Metal's flushed square would zero a subnormal decay |
| `:410` | `for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);` (`kv = 0.0f`, `:408`) | `FFMA kv, sk0, s0, RZ`, then 31 `FFMA kv, sk_r, s_r, kv`, r ascending | `kv = ieee_fma_any(s[r], sk[rg * RPG + r], kv)` from `kv = 0.0f`, r ascending | yes, material: the decayed state goes subnormal, and products of small state entries and key components are subnormal |
| `:413` | `const float kv_col = red[0][col] + red[1][col] + red[2][col] + red[3][col];` | `LDS` x4; `FADD t, r0, r1; FADD t, t, r2; FADD kv_col, t, r3` | `ieee_add(ieee_add(ieee_add(r0, r1), r2), r3)` | yes (subnormal partials of a decayed state): material wherever `v` (`:414`) is as small |
| `:414` | `const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];` | `FFMA d, kv_col, -g, v; FMUL delta, d, beta`: `v - g * kv_col` contracted by ptxas into one rounding | `delta = ieee_mul(ieee_fma_any(-g, kv_col, v), beta)` | yes, material: a subnormal g, kv_col, v or beta (section 5, `:151`), and their products |
| `:417`, `:423` | `s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);` | per r: `FMUL p, delta, sk_r; FFMA s_r, s_r, g, p` (the product `sk * delta` rounded first) | `p = ieee_mul(sk[rg * RPG + r], delta); s[r] = ieee_fma_any(g, s[r], p)` | yes, material: the state itself, which decays toward zero |
| `:424` | `o = fmaf(s[r], sq[rg * RPG + r], o);` (`o = 0.0f`, `:420`) | `FFMA o, s0', sq0, RZ`, then 31 `FFMA o, s_r', sq_r, o`, r ascending, each with the updated `s[r]` of `:423` (the two chains interleave) | `o = ieee_fma_any(s[r], sq[rg * RPG + r], o)` after `:423`'s update of the same r | yes, material: as `:410` |
| `:431` | `oc = (red[0][col] + red[1][col] + red[2][col] + red[3][col]) * rsqrtf((float) S);` | `LDS` x4; `FADD` x3 left to right; `FMUL oc, t, 0.08838834613561630249`: PTX `rsqrt.approx.f32` of `0f43000000`, folded by ptxas into `0x3DB504F3` | `oc = ieee_mul(ieee_add(ieee_add(ieee_add(r0, r1), r2), r3), as_type<float>(0x3DB504F3u))` | yes, material: a subnormal or small `oc` is scaled by up to `1 / sqrt(eps)` at `:441` into a normal output |
| `:432` | `sq_part = oc * oc;` | `FMUL sq, oc, oc` (rg 0 only; `HFMA2 sq, -RZ, RZ, 0, 0` gives the other warps `sq_part = 0`) | `ieee_mul(oc, oc)` | yes, material in principle: the square of a |oc| below 1.08e-19 is subnormal, kept by CUDA and zero in Metal's `*`; it changes a partial's last bit only while the partial is below 2^-102, and reaching `ss * 2^-7 + eps` (`:439`) takes a chain of exact rounding ties through the later additions (constructible, never met by random data); the IEEE helpers cost nine operations per column and token, so the port keeps CUDA's bits here as well (invariant 5) |
| `:434` | `for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);` | `SHFL.BFLY r, sq, o, 0x1f; FADD sq, r, sq`, o = 16, 8, 4, 2, 1, every warp (rg > 0 sums zeros) | `warp_sum_ieee(sq_part)` | as `:432` |
| `:438` | `const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];` (rg 0's four warps) | `LDS.128 w0..w3; FADD t, w0, w1; FADD t, t, w2; FADD ss, t, w3` | `ieee_add(ieee_add(ieee_add(w0, w1), w2), w3)` | as `:432` |
| `:439` | `const float scale = rsqrtf(ss / (float) S + eps);` | `FFMA x, ss, 0.0078125, eps` (PTX `fma.rn.f32 %f434, %f433, 0f3C000000, %f509`: NVVM's exact `* 2^-7` for `/ 128`, contracted with `+ eps`); `rsqn(x) -> scale` | `scale = metal::fast::rsqrt(fma(ss, 0.0078125f, eps))` | no: x >= eps (normal), so the `rsqn` scaling is dead; a subnormal `ss` adds nothing to eps |
| `:441` | `y[...] = oc * scale * gamma[col] * (1.0f / (1.0f + __expf(-zz)));` | `FMUL t, scale, oc; FMUL t, t, gamma; FMUL u, zz, -1.4426950216293334961; ex2n(u) -> e; FADD d, e, 1; rcp_rn(d) -> r; FMUL y, t, r` | `ieee_mul(ieee_mul(ieee_mul(oc, scale), gamma[col]), ieee_div(1.0f, 1.0f + expf_ieee(-zz)))` | yes, material: a subnormal `oc` times `scale` is normal in CUDA; the sigmoid is subnormal for zz < -87.34; the products of small values; `e` subnormal (zz in (87.34, 103.97)) is immaterial (`1 + e` = 1) |
| `:446` | `base[r * row_stride] = s[r];` | `STG` | a store | the stored bits are the state's: subnormal entries stay |

The FFMA of this kernel: the 32 of `:410`, the 32 of `:417` or `:423`, the 32 of `:424`, one at `:414` and one at
`:439` (plus the reciprocal's two on its fast path); none other.

## 7. The commit kernels

Three kernels write the kept tokens into the recurrent state; all three run the same per-token recurrence as the
step-norm kernel's state update (`:407-417`), with the same instruction sequence at every line.

- `gdn_commit_layers_pipe_kernel` (`:357-361`, `__launch_bounds__(128)`, 32 blocks of (32, RG) threads, `:859-865`):
  `n = min(*a.n_keep, kVerifyMaxT)` (`kVerifyMaxT = 8`, `verify_kernels.hpp:21`), nothing when `n <= 0`, then
  `commit_layers_body` (`:334-355`): a grid-stride loop over `n_layers * (h_v * (S / 32) + conv_blocks)` items, each
  item either a state block (one value head, 32 columns `col = cblk * 32 + threadIdx.x`, the 128 state rows split over
  the 4 row groups `threadIdx.y`) or a conv-history block (`gdn_conv_commit_n`, moves only, section 4). The next item's
  inputs are loaded (`commit_pre_load`, `:271-294`: the state column `p.s[r]`, the token's k, v, gate and beta into
  `CommitPre`, `:266-269`) before the current item runs (`commit_pre_run`, `:297-331`, the shared `red` buffer
  double-buffered by `t & 1`, one barrier per token). Its t loop is fully unrolled (8 copies): 256 FFMA at `:318`,
  256 FFMA and 256 FMUL at `:325`.
- `gdn_commit_layers_seg_kernel` (`:364-375`), the segmented batch form: `n = min(r.n_keep, kVerifyMaxT)` per row,
  only rows with `r.j == 0`, the same body. Its decay runs on the vector datapath (`FMUL`/`FSETP`, per thread), the
  pipe kernel's on the uniform one: the same operations.
- `gdn_step_commit_kernel` (`:256-263`, body `gdn_step_commit_block`, `:217-254`; grid (h_v, 4) of (32, RG)):
  `n = *n_keep`, not clamped; launched by `gdn_step_norm_multi` when every kept token precedes the first output token
  (section 6). One block per (head, 32-column block), the state in registers, `red[RG][32]` reused per token behind
  the barrier that stages the next token's k.

Line numbers are the pipe kernel's, `gdn_step_commit_block`'s in brackets. Listings:
`verify/nvcc130/commitpipe.sass.txt`, `commitseg.sass.txt`, `stepcommit.sass.txt`.

| Line | Expression | SASS (verbatim, source names) | Metal | Subnormal reachable |
| --- | --- | --- | --- | --- |
| `:315` [`:241`] | `const float gt = __expf(p.g[t]);` [`const float g = __expf(gate[(size_t) t * h_v + head]);`] | pipe: `UFMUL UR7, p.g[t], 1.4426950216293334961; UFSETP.GEU.AND UP1, UPT, UR7, -126, UPT; @!UP1 UFMUL UR7, UR7, 0.5; MUFU.EX2 R40, UR7; R2UR UR7, R40; @!UP1 UFMUL UR7, UR7, UR7`; step commit and seg: `FMUL y, gate, 1.4426950216293334961; FSETP.GEU.AND P0, PT, y, -126, PT; @!P0 FMUL y, y, 0.5; MUFU.EX2 g, y; @!P0 FMUL g, g, g`: `ex2n`, non-FTZ | `gt = expf_ieee(p.g[t])` | yes, material: as `:407` (a subnormal decay for gate in (-103.97, -87.34), which `fast::exp2` of the product alone flushes) |
| `:318` [`:244`] | `for (int r = 0; r < RPG; ++r) kv = fmaf(p.s[r], sk[t][rg * RPG + r], kv);` (`kv = 0.0f`) | `FFMA kv, s0, sk0, RZ`, then 31 `FFMA kv, s_r, sk_r, kv`, r ascending | `kv = ieee_fma_any(s[r], sk[rg * RPG + r], kv)` from `0.0f`, r ascending | yes, material: as `:410` |
| `:321-322` [`:247`] | `kv_col = red[t & 1][0][col_local] + red[t & 1][1][col_local] + red[t & 1][2][col_local] + red[t & 1][3][col_local];` | `LDS` x4; `FADD` x3 left to right | `ieee_add(ieee_add(ieee_add(r0, r1), r2), r3)` | yes: as `:413` |
| `:323` [`:248`] | `const float delta = (p.v[t] - gt * kv_col) * p.b[t];` | pipe: `FFMA d, kv_col, -UR7(gt), v; FMUL delta, b, d`; step commit: `FFMA d, -g, kv_col, v; FMUL delta, d, beta`: one contracted rounding, then the product | `delta = ieee_mul(ieee_fma_any(-gt, kv_col, v), b)` | yes, material: as `:414` |
| `:325` [`:250`] | `p.s[r] = fmaf(gt, p.s[r], sk[t][rg * RPG + r] * delta);` | per r: `FMUL p, delta, sk_r; FFMA s_r, s_r, UR7(gt), p` (step commit: `FFMA s_r, g, s_r, p`) | `s[r] = ieee_fma_any(gt, s[r], ieee_mul(sk[rg * RPG + r], delta))` | yes, material: as `:417` |
| `:327-330` [`:253`] | the state store | `STG` | a store | the stored bits are the state's |

No other float instruction acts on data: the `MUFU.RCP` rows of the seg kernel (`:272`, `:280`, `:300`, `:338`,
`:342`, `:347`) are the unsigned integer divisions of its item and head indices (`I2F.U32.RP`, `MUFU.RCP`,
`F2I.FTZ.U32.TRUNC`). The state the commit writes is therefore the state the step-norm kernel computes over the same
kept tokens (`:417`/`:423` against `:325`, the same decay, `kv` chain, contraction and update): a Metal port that
uses the same helpers in both keeps them bitwise equal, which `dec_commit_parity` checks. The contract lists `:315`
among the approximate operations to map (`flash-next.md`, invariant 5, the commit replay's decay): `expf_ieee` is that
mapping with CUDA's non-FTZ subnormal path (section 2).

## 8. `ple_hist_seg_kernel`

`:515-532`: per segment row `t = blockIdx.y`, `dir == 0` copies the slot's array into work row t, `dir == 1` copies
work row `r.row0 + r.n_keep - 1` into the slot (first rows of a segment with `n_keep > 0` only); the launcher
(`:1092-1101`) uses 256 threads per block. Listing: `verify/nvcc130/conv.sass.txt` rows 1-213.

| Line | Expression | SASS | Metal | Subnormal reachable |
| --- | --- | --- | --- | --- |
| `:530-531` | `for (int64_t i = ...; i < n; i += (int64_t) gridDim.x * blockDim.x) dst[i] = src[i];` | `LDG.E`/`STG.E` 32-bit; the only float-unit instructions are `I2F.U32.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC` (`:530`) and `I2F.U64.RP`, `MUFU.RCP`, `F2I.U64.TRUNC` with `__cuda_sm20_div_u64` (`:532`): the unsigned integer division of the loop's trip count | a bounded grid-stride copy of 32-bit words; integer `/` where a trip count is needed | yes, immaterial: moved, never computed on (subnormals and NaN payloads keep their bits) |

No float arithmetic touches the data.

## 9. The per-stage kernels against `dec_gdn_rec`

`dec_gdn_parity` (`src/kernels/dec_gdn_parity.cpp:1-17`) runs `gdn_conv_l2_multi`, `gdn_ab_multi` and
`gdn_step_norm_multi` as the reference (`:184-188` of that file, `qk_heads = 2 * HK`) and requires every output the
fused path writes (h, gate, beta, the q8_1 of y, the out projection) to match bit for bit (`memcmp`, `:244-263`), at
T = 1..8, each column equal to the same column of an eight-token window, the state untouched (`:297`). Its x and state
are random normal values, but qkv and z, which the kernels read, are `native_mmvq` products of random GGUF bytes with
normal fp16 scales (that file's `:78-93`, `:127-128`, `:181`, `:187`), so they can reach the subnormal paths. [M] On the
port's draws of these distributions (`tests/kernels/test_basalt_dec_gdn.py`'s `gdnf_case` at T = 8, max |qkv| 1.1e3 to
5.3e3 per combo), 120-278 h values per combo are subnormal, all on q and k channels, where the L2 scale (`:53`) takes a
normal SiLU output (2.6e-37 to 1e-34) below 2^-126; and 50-349 y values are subnormal, with 11-138 z per combo in
(-88.72, -87.34), where the sigmoid factor (`:441`) is subnormal (combo 1's z is a Q3_K product where basalt's is
IQ3_S). [H] basalt's own draws, from the same distributions, reach them too. `parity_seg`'s qkv, z and x are random
normal (`dec_gdn_parity.cpp:437-438`). The comparison below
reads `dec_gdn_rec`'s own compile, the same compiler and flags (`dec_gdn_rec.cu` builds with the IEEE defaults too,
`CMakeLists.txt:162`): `.context/gdn/nvcc/nvcc130/rec_kernel_2.sass.txt` and `.ptx.txt` (`rec_kernel<2>` and
`rec_norm_kernel`), and `dec_gdn.md`'s record of that compile (cited as `dec_gdn.md` §n, its section and the
`rec.cu` line of its row); the listing and the record agree at every row. The operands flow between the stages as fp32
(`h`, `gate`, `beta` are `float` buffers, `:31`, `:379`), so the stages' boundaries add no rounding the fused kernel
skips.

| Quantity | Per-stage | `dec_gdn_rec` | Compiled sequence in both | Same |
| --- | --- | --- | --- | --- |
| conv | `:45` (`:75`) | `rec.cu:64` `float sum = v0 * w[0] + v1 * w[1] + v2 * w[2] + x * w[3];` (`dec_gdn.md` §1, §6d) | PTX `mul.f32 (v1, w1); fma.rn.f32 (v0, w0, +); fma.rn.f32 (v2, w2, +); fma.rn.f32 (x, w3, +)` (`rec_kernel_2.ptx.txt:1643-1646`, SASS rows 627-649): `v1 * w1` rounded, `v0 * w0` fused into the first add | yes |
| SiLU | `:46` (`:76`) | `rec.cu:65` `return sum / (1.0f + __expf(-sum));` (`dec_gdn.md` §1, §6b) | `FMUL u, sum, -1.4426950216293334961; ex2n(u); FADD d, e, 1; div_rn(sum, d)` with `__cuda_sm3x_div_rn_noftz_f32_slowpath` (PTX `rec_kernel_2.ptx.txt:1647-1650`) | yes |
| L2 norm of q and k | `:48-53` (`:78-83`) | `rec.cu:209-214`, `:226-229` (`dec_gdn.md` §2) | `FMUL y*y`; butterfly 16..1 over 32 consecutive channels (the same channel-to-lane map: conv block = one head's 128 channels, `c = blockIdx.x * S + tid`; `rec.cu:160` `cq = qh * S + tid`, `ck = qk + qh * S + tid`); `((p0 + p1) + p2) + p3`; `FADD x, ss, eps`; `rsqn(x)`; `FMUL y, y, r` (rows 679-888) | yes |
| v channels | `:46`, `:55` (blocks `>= qk_heads`) | `rec.cu:215-219` (warp `rg == 1`; `dec_gdn.md` §2 `:216`) | the SiLU above, stored unnormalized | yes |
| gate GEMV | `:128-141`, `:182-194` | `rec.cu:89-106` (`ab_one`; `dec_gdn.md` §1 `:103-106`) | per lane `j = lane + 32 q` ascending, the same 8 `fmaf` per word (bf16 low half first), one chain per token; `rec` reads only `n_embd = 2560` (10 words per lane), where `gdn_ab_multi` takes `unrolled<., 10>` and both kernels give the same chain | yes, at `n_embd = 2560` |
| gate sums | `:148`, `:203` | `rec.cu:108` (`dec_gdn.md` §1) | `SHFL.BFLY` + `FADD`, o = 16..1 | yes |
| beta | `:151`, `:206` | `rec.cu:109` `1.0f / (1.0f + __expf(-v))` (`dec_gdn.md` §1, §6b, §6d) | `ex2n` of `v * -log2e`; `FADD d, e, 1`; `rcp_rn(d)` | yes |
| softplus | `:153-154`, `:208-209` | `rec.cu:110-111` `u > 20.0f ? u : log1pf(__expf(u))` (`dec_gdn.md` §1, §6a) | `FADD u, a, dt`; the select; `__expf`; the `log1pf` sequence of section 2 instruction for instruction (`dec_gdn.md` §6a: the same constants in the same order, the same `FADD.RZ` and special block; only register names and the placement of one `MOV` differ) | yes |
| gate | `:155`, `:210` | `rec.cu:112` `sp * sa` (`dec_gdn.md` §1) | `FMUL` | yes |
| decay | `:407` (`:241`, `:315`) | `rec.cu:248` `const float g = __expf(sm.g[t]);` (`dec_gdn.md` §2, §6b) | `FMUL y, gate, 1.4426950216293334961; ex2n(y)`: vector `FMUL`/`FSETP` in `rec` (rows 1010-1027), uniform `UFMUL`/`UFSETP` in the verify kernels: the same IEEE operations on the same operands | yes |
| `kv` | `:410` (`:244`, `:318`) | `rec.cu:251` (`dec_gdn.md` §2) | `FFMA kv, s0, k0, RZ` (row 1004) and 31 chained, r ascending, over the same state partition (thread (rg, col) holds rows `32 rg + r` of column `col`: `rec.cu:154-158` against `:394-397`, `:232-235`, `:290-293`) | yes |
| `kv_col` | `:413` (`:247`, `:321-322`) | `rec.cu:254` (`dec_gdn.md` §2) | `((r0 + r1) + r2) + r3` (rows 1062-1064) | yes |
| `delta` | `:414` (`:248`, `:323`) | `rec.cu:255` `(sm.v[t][lane] - g * kv_col) * sm.b[t]` (`dec_gdn.md` §2, §6d) | `FFMA d, -g, kv_col, v; FMUL delta, d, b` (rows 1065-1066): the same contraction the source does not write | yes |
| state update | `:417`, `:423` (`:250`, `:325`) | `rec.cu:259` (`dec_gdn.md` §2) | `FMUL p, delta, k_r; FFMA s_r, g, s_r, p` | yes |
| `o` | `:424` | `rec.cu:260` (`dec_gdn.md` §2) | `FFMA o, s0', q0, RZ` (row 1075) and 31 chained with the updated `s[r]` | yes |
| `oc` | `:431` | `rec.cu:265-266` (`dec_gdn.md` §2, §6b) | `((r0 + r1) + r2) + r3`, then `FMUL oc, t, 0.08838834613561630249` (`rsqrtf((float) S)` folded to `0x3DB504F3`: `rec_kernel_2.sass.txt:1184`, `rec_split_kernel_2:1166`, `rec_seg_kernel_2:1186`) | yes |
| sum of squares | `:432-438` | `rec.cu:135-139` (`sq32`), `:270-272`, `:280-294` (`dec_gdn.md` §1 `:136-137`, §2 `:294`, §4 `:347`) | `FMUL oc*oc`, butterfly 16..1 over 32 consecutive columns, then `((w0 + w1) + w2) + w3` over the four 32-column groups (verify: rg 0's four warps; `rec`: the four cluster ranks, `col = rank * 32 + lane`, `rec.cu:148`, pushed through the cluster's shared memory) | yes |
| scale | `:439` | `rec.cu:119` `rsqrtf(ss / (float) S + a.eps)` (`dec_gdn.md` §1, §6d) | `FFMA x, ss, 0.0078125, eps; rsqn(x)` (`rec_kernel_2:1278`) | yes |
| gated output | `:441` | `rec.cu:120` `oc * scale * gam * (1.0f / (1.0f + __expf(-z)))` (`dec_gdn.md` §1) | `((oc * scale) * gamma) * rcp_rn(1 + ex2n(z * -log2e))` (`rec_norm_kernel` PTX, `rec_kernel_2.ptx.txt:154-160`; operands commuted only) | yes |

Every quantity the per-stage kernels share with `dec_gdn_rec` compiles to the same instructions, the same operand
pairing in every FFMA, the same reduction order and the same approximate units: no FFMA pairs differently, and the
three contractions the source does not write (`:45`/`rec.cu:64`, `:414`/`rec.cu:255`, `:439`/`rec.cu:119`) fall on
the same operands. The pairing was read in `rec_kernel<2>`'s PTX and SASS; the per-line float-operation counts of
all ten `dec_gdn_rec` instantiations (`rec_kernel<2|4|8>`, `rec_split_kernel<2|4|8>`, `rec_seg_kernel<2|4|8>`,
`rec_norm_kernel`) have the same structure at every row of the table: one FMUL and three FFMA per conv
(`rec.cu:64`), one FFMA and one FMUL and no FADD at `rec.cu:255`, one FFMA per token and no FADD at `rec.cu:119`
and `rec.cu:347` (`rec_norm_kernel`'s scale), the same multiset at `rec.cu:111`, so no instantiation pairs or
reduces differently. In CUDA the fused kernel is therefore the per-stage kernels' arithmetic, which is why
`dec_gdn_parity` holds bit for bit; y's q8_1 rows (`rec.cu:118-131` against `native_quantize_q8_1`) are outside this
page (`dec_gdn.md` §6c). On Metal the two paths stay bitwise equal, the subnormal cases included, when both
use one helper per operation at every site: `expf_ieee`, `log1pf_cuda`, `ieee_div(1.0f, d)`, `ieee_div(n, d)`, the
constant `0x3DB504F3`, `metal::fast::rsqrt`, `ieee_fma_any`, `ieee_mul`, `ieee_add` and `warp_sum_ieee`
as sections 3 to 7 assign them, and `fma()` at exactly the three contractions above.

The two pages agree on every compiled sequence. Their Metal columns differ at four points, which the plan settles
once for both kernels: a Metal port whose fused and per-stage kernels took different helpers at a shared site would
give different bits wherever a subnormal is reached, and `dec_gdn_parity`'s inputs reach one at shared sites (above: h
at the L2 scale, `:53`, and the sigmoid factor of y, `:441`, at point 2's logits in (-88.72, -87.34)).

1. `__expf` where `1.0f + e` follows (`rec.cu:65`, `:109`, `:120` against `:46`, `:76`, `:151`, `:206`, `:441`):
   `dec_gdn.md` §6b writes `expf_neg`, this page `expf_ieee`. The bits are the same there (a subnormal `e` vanishes in
   `1 + e`); the contract's one helper per operation makes it `expf_ieee` at every site, which is `dec_gdn.md`'s `ex2n`
   helper (§6b, at the decay and the softplus) under another name.
2. `1.0f / (1.0f + e)` (`rec.cu:109`, `:120` against `:151`, `:206`, `:441`): `dec_gdn.md` §1 and §6d write
   `ieee_rcp`, which origin/main documents for |s| <= 2^126, an infinity or NaN
   (`kernels/common/basalt_q8_1.metal:150`, `:153`). Above 2^126 it returns Metal's `1.0f / s`, which writes the
   subnormal reciprocal of a d in (2^126, 2^128), a logit in (-88.72, -87.34), as zero where CUDA's `rcp_rn` keeps
   it: the stored beta, the sigmoid factor of y. `ieee_div(1.0f, d)` (`basalt_q8_1.metal:176`, `:181`, documented for b >= 1) rebuilds
   that quotient on the subnormal grid: the rule at both.
3. Sums of possibly subnormal partials. (i) The sums of squares (`rec.cu:136-137`, `:209-212`, `:226-227`, `:294`,
   `:347` against `:48-52`, `:432-438`): `dec_gdn.md` calls a subnormal square immaterial and writes Metal's `*`, `+`
   and `warp_sum`; this page writes `ieee_mul`, `ieee_add` and `warp_sum_ieee`. The pages agree on the facts (a
   subnormal square changes a partial only below 2^-102 and reaches `ss + eps` only through a chain of exact rounding
   ties); invariant 5 reads as the IEEE helpers. (ii) `oc`'s partial sum (`rec.cu:265` against `:431`): both pages
   call the site material (a small `oc` is scaled up at `rec.cu:120`, `:441`), but `dec_gdn.md` §2 writes plain `+`
   there, where its `kv_col` row (`rec.cu:254`) writes `ieee_add`: `ieee_add` at both.
4. The general FFMA chains (`rec.cu:64`, `:103-106`, `:251`, `:255`, `:259`, `:260` against `:45`, `:136-141`,
   `:191-194`, `:410`, `:414`, `:417`, `:423`, `:424`, `:318`, `:323`, `:325`): `dec_gdn.md` writes `ieee_fma` where a
   subnormal can occur; origin/main's `ieee_fma` (`basalt_q8_1.metal:158`, `:167`) is documented for an integer
   multiplier |n| < 2^24 only, so both need the general `ieee_fma_any` (and `ieee_mul` for the conv's unfused
   product).

(The decay's subnormal range differs in description only: `dec_gdn.md`'s "about (-103.3, -87.34)" counts results
from 2^-149 up, while `ex2n`'s squared result rounds up to 2^-149 from just above 2^-150, which gives (-103.97,
-87.34). No rule changes.)

For the contract (`flash-next.md`): its list of approximate operations names `:315` alone for `verify_kernels.cu`;
the per-stage kernels' `__expf` (`:46`, `:76`, `:151`, `:154`, `:206`, `:209`, `:241`, `:407`, `:441`), `log1pf`
(`:154`, `:209`) and run-time `rsqrtf` (`:53`, `:83`, `:439`) are approximate in the same way and belong on it, with
`rsqrtf((float) S)` (`:431`) as the folded constant. Three helpers the rows above need do not exist on origin/main; the unmerged `origin/woodsonl/basalt-sampler` (read at
`2acc918`) settles one of them:

- `ieee_fma_any`: that branch has it as `ieee_fma_count` (`kernels/common/basalt_sampler.metal:77-86`), a correctly
  rounded fma for any operands: the hardware's `fma` when all three exponent fields are nonzero and the result is at
  least 2^-125, otherwise `basalt_f64.metal`'s binary64 fma rounded to odd, then to binary32. The port reuses it under
  one general name in `basalt_ieee.metal` rather than writing a second one.
- `expf_ieee`: not on that branch. Its `expf_approx` (`basalt_sampler.metal:90-93`,
  `fast::exp2(ieee_mul(x, 0x3fb8aa3b))`) is the contract's literal mapping, "without ex2.approx's branch for subnormal
  results"; like `expf_neg`, it writes zero where CUDA's halving and squaring below 2^-126 gives a subnormal, and that
  result is material at the decay (`:241`, `:315`, `:407`) and the softplus (`:154`, `:209`). One helper per operation
  would put `expf_ieee` at every IEEE-built `__expf` site, the sampler's included, which amends the contract's `__expf`
  mapping: a decision for the plan.
- `log1pf_cuda`: not on that branch. Its `basalt_libdevice.metal` (exact transcriptions of libdevice's `__nv_exp` and
  `__nv_logf`, credited in `third_party/NOTICE`) is the precedent and the home for a transcription of section 2's
  sequence.

Divisions need nothing new: every one on this page divides by `1.0f + e` >= 1, inside origin/main's `ieee_div`, so
that branch's general `ieee_div_any` (`basalt_sampler.metal:65-75`) is not needed here.
