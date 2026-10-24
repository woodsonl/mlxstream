# basalt `sampler.cu` as nvcc compiles it: the per-site record

What NVIDIA's compiler emits at every floating-point site of basalt's sampler kernels, read from the sm_120 SASS
(with the PTX where it shows what the front end chose), and the Metal form the port writes at each site. This is the
sampler family's page under invariant 5 of `docs/design/flash-next.md`; `matvec.md` and `dequant.md` (branch
`woodsonl/basalt-mmvq`) are the q8_1 and matrix-vector pages, and this page uses their notation.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of `src/kernels/cuda/sampler.cu`.

## 1. Build, compiler and method

- **Source.** `src/kernels/cuda/sampler.cu` with basalt's three local includes inlined under `#line` directives
  (`include/basalt/kernels/sampler.hpp`, `include/basalt/core/coupled_draft.hpp`, `include/basalt/core/diag.hpp`),
  so every PTX and SASS line names its basalt file and line. CUDA's own headers come from the compiler's
  installation. Only basalt's public MIT source was sent.
- **Flags.** `sampler.cu` builds in the `kernels` library (`CMakeLists.txt:118-167`) with nvcc's IEEE defaults: it is
  not in the `--use_fast_math` list (`:185-202`), and the only other fast-math flag (`:728`) belongs to the `mmq`
  target. A Release build (`:12-14`) passes CMake's `-O3 -DNDEBUG`, `add_compile_definitions` passes
  `-DBASALT_VERSION="0.0.1"` (`:7`), and the one architecture is 120 (`:3-4`, `:26-31`), for which CMake passes
  `--generate-code=arch=compute_120,code=[compute_120,sm_120]`; `-arch=sm_120` names the same target. No CUDA
  standard, `-fmad`, `-ftz`, `-prec-div` or `-Xptxas` flag is set anywhere, so nvcc's defaults hold: `-fmad=true` (a
  multiply and an add may contract), `-ftz=false`, `-prec-div=true`, `-prec-sqrt=true`.
- **Compiler.** Compiler Explorer's public API, `POST https://godbolt.org/api/compiler/nvcc130/compile` with
  `"compilerOptions": {"produceDevice": true}` and `"filters": {}`, which returns the PTX and the `SASS (sm_120)`
  listing. Compiler id `nvcc130`: NVCC 13.0.0, version string `Cuda compilation tools, release 13.0, V13.0.48`.
  Basalt requires CUDA 13.0 or newer (`CMakeLists.txt:32-33`).
- **userArguments**, verbatim: `-arch=sm_120 -lineinfo -O3 -DNDEBUG '-DBASALT_VERSION="0.0.1"'`. Compiler Explorer
  adds `-g -lineinfo --keep-device-functions -c -keep -Xcompiler=-S` and its host compiler (GCC 14.1). `-g` is host
  debug information (`-G` would be the device one), `--keep-device-functions` only keeps device functions no kernel
  calls, and `-lineinfo` only attributes, so none of them changes a kernel's code.
- **Date.** Compiled October 16, 2026.
- **Translation units.** Compiler Explorer cuts a SASS listing at about 5,000 lines (`[truncated; too many lines]`),
  and the whole file lists more: the full compile stops inside `sampler_split_merge_kernel`, the fifth kernel of the
  listing. So each kernel group was also compiled as its own unit: `sampler.cu` with the bodies of the other kernels
  blanked line for line (an empty line keeps every line number; a blanked kernel compiles to `EXIT`). The groups are
  the three greedy kernels; `sampler_one_block_kernel`; `sampler_split_part_kernel`; `sampler_split_merge_kernel`;
  `coupled_stage_kernel`, `coupled_penalize_kernel` and `coupled_part_kernel<8, 16>`; `coupled_merge_kernel`. The cut
  changes nothing in a kept kernel, because every device function it calls is inlined into it: the SASS of each
  kernel the full compile lists complete (`coupled_part_kernel<8, 16>`, `coupled_merge_kernel`,
  `coupled_penalize_kernel`, `coupled_stage_kernel`) and the 1,355 lines of `sampler_split_merge_kernel` before the
  cut are identical to the group units', instruction for instruction and source line for source line, once labels and
  the number in the slow-path subroutine's name are renamed.
- **`sampler_split_part_kernel`** lists more than 5,000 lines on its own: the penalty block (`:621-628`) is unrolled
  32 times, each copy with an inlined `history_count` loop and a division. Its unit lists the prologue and 21
  complete copies of the block before the cut, and its PTX shows the same float operations in all 32 copies
  (`setp.gtu.f32`, `mul.f32`, `div.rn.f32`, `cvt.rn.f32.u32`, `fma.rn.f32`, `sub.f32`, 32 of each). A second unit
  blanks only the penalty block, so the rounds and the merge fit; its PTX for `:630-661`, for `take_first` and
  `warp_first` (`:339-360`) and for `warp_merge_lists` (`:536-575`) is identical to the real unit's (registers and
  labels renamed). Equal PTX does not prove equal SASS, but those lines hold comparisons, selects and integer
  operations only, and section 3.5 records them from the second unit's SASS.
- **Reading a site.** The PTX shows what the front end fused (`fma.rn.f32`), which division it chose (`div.rn.f32`,
  `rcp.rn.f32`, `div.rn.f64`) and which library sequence it inlined; the SASS shows what `ptxas` made of it. A table
  row quotes the SASS of one copy where a site is inlined or unrolled several times; the per-line opcode multisets of
  every copy were compared, and copies differ only in unroll count. The instructions after a kernel's last `EXIT` are
  CUDA's out-of-line slow paths, `__internal_N_$__cuda_sm3x_div_rn_noftz_f32_slowpath` (binary32 division),
  `__internal_N_$__cuda_sm20_rcp_rn_f32_slowpath` (binary32 reciprocal) and `__internal_N_$__cuda_sm20_div_rn_f64_full`
  (binary64 division), which `-lineinfo` attributes to whatever line precedes them. The tables attribute each to the
  `CALL.REL.NOINC` that reaches it; the `.FTZ` instructions inside them are CUDA's own range handling, not basalt
  arithmetic.
- **Not float arithmetic.** At the head of a strided loop (`:115` and so on), `I2F.U32.RP`, `MUFU.RCP`,
  `F2I.FTZ.U32.TRUNC.NTZ` and `IMAD.HI.U32` are `ptxas`'s integer division for the trip count, exact;
  `HFMA2 R, -RZ, RZ, a, b` moves a constant into a register; `IMAD.WIDE` forms addresses. The tables leave them out.

## 2. Shared device functions and library sequences

Every kernel inlines its device functions, so the sequences below appear once per call site; the kernel tables in
section 3 point here. Kernel parameters live in constant bank 0 from `c[0x0][0x380]`; in `sampler_greedy_kernel` the
`SamplerParams` copy starts at `0x3a0`, so `penalty_repeat` is `c[0x0][0x3b8]`, `penalty_freq` `0x3bc` and
`penalty_present` `0x3c0` (`sampler.hpp:15-29`: `top_k`, `top_p`, `min_p`, `temperature`, `min_keep`,
`penalty_last_n`, then the three penalties at byte offsets 24, 28 and 32). Where `SamplerParams` is uniform the
compiler runs the scalar parts on the uniform datapath (`UFFMA`, `UFMUL`, `UIMAD.WIDE.U32`, ...); that changes no
rounding.

### 2.0 The Metal operations this page names

No floating-point instruction that basalt's sampler source produces carries `.FTZ`: the build is IEEE, so CUDA keeps
subnormal operands and results everywhere here, and its ordered comparisons see a subnormal as itself. This GPU
flushes subnormals in arithmetic and compares them as zero (`matvec.md`, Notation). The Metal forms therefore use the
following operations, one helper per operation at every site; the names are proposals for the kernel lane.

- `key32(x)`: the order-preserving key, `b = as_type<uint>(x); if (b == 0x80000000u) b = 0u; return (b >> 31) ? ~b :
  (b | 0x80000000u);`. For non-NaN a and b, a < b exactly when `key32(a) < key32(b)` and a == b exactly when the keys
  are equal: IEEE order with every subnormal distinct, -0 equal to +0. `nan32(x)` is
  `(as_type<uint>(x) & 0x7fffffffu) > 0x7f800000u`.
- Comparisons: `gt32`, `ge32`, `eq32`, `ne32` and `lt32` compare keys and are false when either operand is NaN
  (CUDA's ordered `FSETP.GT`, `.GE`, `.EQ`, `.NE`, `.LT`); the unordered forms `geu32`, `gtu32` and `neu32` are true
  when either operand is NaN (`FSETP.GEU`, `.GTU`, `.NEU`). Each row below names the flavor the SASS uses where it
  matters.
- `fmax32(a, b)`: CUDA's `fmaxf` (`FMNMX` without `.FTZ`): the other operand when one is NaN, else the larger by
  `key32`. Which zero it returns for -0 against +0 is not recorded; no sampler site depends on it (section 2.4).
- `add32`, `mul32`, `fma32`, `div32`, `rcp32`: the correctly rounded IEEE operation with subnormal operands and
  results kept (CUDA's `FADD`, `FMUL`, `FFMA`, `div.rn.f32` and `rcp.rn.f32`). Each is Metal's `+`, `*`, `fma` or `/`
  whenever its operands and its result are normal, zero or infinite; a subnormal operand or result takes the integer
  emulation `matvec.md` section 3 describes. The sites below say whether a subnormal is reachable.
- binary64 (MSL has no `double` on the GPU, so all of it is software on 32-bit words): `dadd`, `dmul`, `dfma` and
  `ddiv`, IEEE binary64 round-to-nearest-even with subnormals (NVIDIA's `DADD`, `DMUL`, `DFMA` and `div.rn.f64` round
  correctly and never flush, so any correct implementation gives their bits); `d_of_f32` (`F2F.F64.F32`, exact);
  `f32_of_d` (`F2F.F32.F64`, `cvt.rn.f32.f64`: round to nearest even, float subnormals kept); the comparisons
  `dge`, `dgt`, `dlt` (ordered, `DSETP`) and `dgeu` (unordered).
- The shipped names (`kernels/common/basalt_sampler.metal`, `basalt_ieee.metal`, `basalt_f64.metal` and
  `basalt_libdevice.metal`, whose compares and conversions take and return bit patterns): `key32`, `nan32`, `gt32` and
  `lt32` are `okey`, `fnan`, `f_gt` and `f_lt`; `eq32`, `ge32`, `ne32` and the unordered forms have no helper, each site
  comparing `okey` keys (`am_merge`, `take_first`) or testing `fnan` (`gtu32(x, 0.0f)` is `fnan(x) || f_gt(x, 0)`);
  `fmax32` is `fmaxf_bits`; `add32`, `mul32`, `div32`, `rcp32` and `fma32` are `ieee_add`, `ieee_mul`,
  `ieee_div_any`, `ieee_div_any(1, x)` and `ieee_fma_count`, except that section 3.3's `rcp32(tot)` is a plain
  `1.0f / tot` (tot is at least 1), and `ex2_approx(mul32(x, 0x1.715476p+0f))` is `expf_approx(x)`; `dadd`, `dmul`, `dfma`, `ddiv`, `d_of_f32` and
  `f32_of_d` are `f64_add` (`dadd(x, -y)` is `f64_sub(x, y)`), `f64_mul`, `f64_fma`, `f64_div`, `f64_from_f32` and
  `f32_from_f64`, `dge` and `dlt` are `f64_ge` and `f64_lt`, `dgt(a, b)` is `f64_lt(b, a)` and `dgeu(a, b)` is
  `!f64_lt(a, b)`, and `hi32` and `lo32` are `hi` and `lo`; `cuda_exp` and `cuda_logf` are `nv_exp` and `nv_logf`.

### 2.1 `apply_penalties` (`:71-77`) and `history_count` (`:65-69`)

`history_count` is integer compares and adds. SASS from `sampler_greedy_kernel`; every other copy
(`sampler_one_block_kernel`, 32 in `sampler_split_part_kernel`, `coupled_penalize_kernel`) has the same instructions.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:72` | `if (count <= 0) return logit;` | `ISETP.NE.AND P1, PT, R17, RZ, PT` (count is never negative) | integer |
| `:73` | `if (logit <= 0.0f) logit *= p.penalty_repeat;` | `FSETP.GTU.AND P1, PT, R2, RZ, PT`; `@!P1 FMUL R12, R2, R13` (R13 = `c[0x0][0x3b8]`, `penalty_repeat`) | `if (!gtu32(logit, 0.0f)) logit = mul32(logit, repeat);` |
| `:74` | `else logit /= p.penalty_repeat;` | PTX `div.rn.f32`. SASS `MUFU.RCP R12, R19`; `FCHK P1, R2, R19`; `FFMA R13, R12, -R19, 1`; `FFMA R13, R12, R13, R12`; `FFMA R12, R2, R13, RZ`; `FFMA R18, R12, -R19, R2`; `FFMA R12, R13, R18, R12`; when `FCHK` flags the operands, `CALL.REL.NOINC` to `__cuda_sm3x_div_rn_noftz_f32_slowpath` | `logit = div32(logit, repeat);` |
| `:75` | `logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;` | PTX `cvt.rn.f32.u32`, `fma.rn.f32`, `sub.f32`. SASS `I2FP.F32.U32 R17, R17`; `FFMA R17, R17, UR4, R2` (UR4 = `c[0x0][0x3bc]`, `penalty_freq`; R2 = `c[0x0][0x3c0]`, `penalty_present`); `FADD R2, -R17, R12` | `t = fma32(float(count), freq, present); logit = add32(logit, -t);` |

- **`:73` branches on IEEE order.** `FSETP.GTU` sends a positive subnormal logit (and NaN) to the division. A Metal
  `logit <= 0.0f` compares the subnormal as zero and would multiply it instead. NaN gives NaN on both branches.
- **`:74` is correctly rounded.** `div.rn.f32` is IEEE division: the reciprocal iteration gives the correctly rounded
  quotient wherever `FCHK` passes, and the slow path covers the rest (zero, infinite, NaN or subnormal operands and
  quotients near the range ends). Metal's `/` gives the same bits for normal operands and a normal quotient.
- **`:75` is one FMA and one subtraction.** The front end folds `(count > 0 ? 1.0f : 0.0f) * p.penalty_present` to
  `p.penalty_present`: `:72` has returned for every count ≤ 0, and 1·x is exact. It then fuses the count's product
  with that add (`fma.rn.f32` in the PTX, so `ptxas` keeps it): t = fma(f(count), freq, present), rounded once, and
  the result is fl(logit − t). `float(count)` is exact (the window is shorter than 2^24).
- **Subnormals.** These lines meet a subnormal only through a logit or a penalty parameter below 2^-126 in
  magnitude, or a result that small; basalt's fixtures have none (`sampler_parity.cpp` uses ±0, ±inf, NaN and normal
  values), but nothing excludes them.

### 2.2 `philox_uniform` (`:37-59`)

| Line | basalt | SASS (`sampler_split_merge_kernel`) | Metal |
|---|---|---|---|
| `:39-42` | `__umulhi(0x9E3779B9u, c0)`, `__umulhi(0xBB67AE85u, c2)`, the two low products | `UIMAD.WIDE.U32 UR10, UR8, -0x61c88647, URZ` (high word: `__umulhi`; `-0x61c88647` is `0x9E3779B9`), `UIMAD UR8, UR8, -0x61c88647, URZ` (low word); likewise with `-0x4498517b` (`0xBB67AE85`). In `sampler_one_block_kernel` the same products as `IMAD.HI.U32` | `mulhi(c0, 0x9E3779B9u)`, `c0 * 0x9E3779B9u` on `uint`: exact |
| `:43-45` | `hi1 ^ c1 ^ k0`, `hi0 ^ c3 ^ k1` | `ULOP3.LUT ..., 0x96` (three-way XOR, the round index i as an immediate) and `0x3c` (two-way, k1 = 0) | `^` |
| `:54-56` | ten rounds | unrolled | unrolled or a bounded loop: the same integers |
| `:58` | `(float) (c0 >> 8) * (1.0f / 16777216.0f)` | `USHF.R.U32.HI UR9, URZ, 0x8, UR9`; `UI2FP.F32.U32 UR9, UR9`; `UFMUL UR9, UR9, 5.9604644775390625e-08` (2^-24, the folded constant) | `float(c0 >> 8) * 0x1p-24f`: exact, an integer below 2^24 scaled by a power of two, never subnormal |

The draw's counter is `p.counter + (uint64_t) t` (`:420`, a 64-bit integer add, `UIADD3.64`).

### 2.3 The selection order (`take_first`, `warp_first`, `am_take`, `am_merge`, the round thresholds)

Every comparison of the selection order is a plain `FSETP` without `.FTZ`, so CUDA orders subnormals exactly and
-0 equals +0. The compiler rewrites each predicate into an equivalent one; the forms seen are:

| basalt | Sites | SASS form | Metal |
|---|---|---|---|
| `if (s > bv)` | `:133`, `:181` (`am_take`), the `s > bv` half of `:491`, `:643-644`, `:759-760` | `FSETP.GT.AND P1, PT, s, bv, PT`; `FSEL`/`SEL` on P1. With bv still -inf the compiler writes `s > -inf` as `FSETP.NE.AND P, PT, s, -INF` (ordered: false for NaN) | `gt32(s, bv)` |
| `ov > bv \|\| (ov == bv && oi < bi)` | `:138`, `:152` (`sampler_greedy_kernel`), `:184` (`am_merge`), `:346` (`take_first`), `:496`, `:507` | either `FSETP.GEU.AND P1, PT, bv, ov`; `@P1 ISETP.GE.AND P2, PT, oi, bi`; `@P1 FSETP.EQ.AND P0, PT, bv, ov, !P2` with P0 preset true, taking on P0; or `ISETP.GE.AND P0, PT, oi, bi`; `FSETP.NEU.OR P0, PT, bv, ov, P0`; `FSETP.GEU.AND P0, PT, bv, ov, P0`, taking on !P0; or that form with the operands swapped, `FSETP.NEU.OR P0, PT, ov, bv, P0`; `FSETP.LEU.AND P0, PT, ov, bv, P0` (`take_first(b0, i0, b1, i1)`, `:646`, `:762`); or branching, `FSETP.GT.AND P0, PT, ov, bv, PT` taking at once, else `ISETP.GE.AND P0, PT, oi, bi`; `FSETP.NEU.OR P0, PT, ov, bv, P0` keeping on P0 (`:557`; the last `warp_first` step of a round tests `FSETP.GEU` first) | `gt32(ov, bv) \|\| (eq32(ov, bv) && oi < bi)` |
| `(s < prev_v \|\| (s == prev_v && v > prev_i)) && s > bv` | `:491`, `:643-644`, `:759-760` | `FSETP.GEU.AND P3, PT, s, prev_v`; `@P3 ISETP.GT.AND P2, PT, v, prev_i`; `@P3 FSETP.EQ.AND P0, PT, s, prev_v, P2` (P0 preset true); `FSETP.GT.AND P0, PT, s, bv, P0`. The even chain of `:643` and `:759` branches instead: `FSETP.GEU.AND P4, PT, s, prev_v, PT`; on !P4, `FSETP.GT.AND P0, PT, s, bv, PT` takes; on P4, `ISETP.GT.AND P0, PT, v, prev_i`; `FSETP.GT.AND P2, PT, s, bv`; `FSETP.EQ.AND P0, PT, s, prev_v, P0`; `@P0 BRA P2` takes | `(lt32(s, prev_v) \|\| (eq32(s, prev_v) && v > prev_i)) && gt32(s, bv)` |

Each SASS form evaluates the source's predicate exactly, NaN included: an unordered `GEU` keeps the candidate out,
so a NaN is never taken, and bv starts at -inf and changes only through a strict `>`. `FSETP.LEU(ov, bv)` is
`FSETP.GEU(bv, ov)`, and the negation of an unordered `LEU` is the ordered `>`, so the swapped form takes on
exactly the source's ordered predicate. In the split rounds the first element of each chain meets bv = -inf, so its
`s > bv` is `FSETP.NE.AND P, PT, s, -INF, PT`; for the odd chain the compiler hoists that test out of the round
loop (it is attributed to `:634` and `:752`) and re-applies it with `@P3 FSETP.NE.AND P0, PT, s, -INF, P2`. The last
butterfly step of `:152` keeps only the index (`FSETP.NEU`, `FSETP.GEU`, `@!P1 VIMNMX.S32` for the tie, `SEL`);
the index is the same.
On Metal the keys are required: `>` and `==` would see two subnormals, or a subnormal and a zero, as equal and break
the tie by id.

### 2.4 `sampled_tail_warp` (`:369-439`)

SASS from `sampler_split_merge_kernel` (`<false>`); `sampler_one_block_kernel` (`<false>`) and `coupled_merge_kernel`
(`<true>`) have the same per-copy instructions at every line, with different unroll counts. `coupled_merge_kernel`
passes no `qprob`, so `:430-437` are absent there, and only it has `:429`. ex holds binary64 values in shared
memory.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:374` | `p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f` | `FSETP.GT.AND P3, PT, R5, RZ, PT`; PTX `rcp.rn.f32`: when the exponent field of T is in 1..252 (`IADD3 R0, PT, PT, R5, 0x1800000, RZ`; `LOP3.LUT R0, R0, 0x7f800000, ...`; `ISETP.GT.U32.AND P0, PT, R0, 0x1ffffff, PT`), `MUFU.RCP R12, R5`; `FFMA R0, R12, R5, -1`; `FADD.FTZ R3, -R0, -RZ`; `FFMA R12, R12, R3, R12`, else `CALL.REL.NOINC` to `__cuda_sm20_rcp_rn_f32_slowpath`; `FSEL R12, R12, RZ, P3` | `gt32(T, 0.0f) ? rcp32(T) : 0.0f` |
| `:376-377` | `mx = sel_logit[0]; mx = fmaxf(mx, sel_logit[i])`, i = 1..k−1 | `FMNMX R, R, R, !PT` (maximum), a serial chain in i | `fmax32` in the same order |
| `:378` | `p.top_p < 1.0f` | `UFSETP.GEU.AND UP0, UPT, UR10, 1, UPT` (skip when not less: NaN skips) | `lt32(top_p, 1.0f)` |
| `:379` | `ex[i] = exp((double) sel_logit[i] - (double) mx)` | `F2F.F64.F32` of both; `DADD R16, -R18, R4`; section 2.6 | `cuda_exp(dadd(d_of_f32(x), -d_of_f32(mx)))` |
| `:383` | `sum += ex[i]` (lane 0, i = 0..k−1, from 0.0) | `DADD` chain in i | `dadd` in i |
| `:384` | `__shfl_sync(kFullMask, sum, 0)` | two `SHFL.IDX` (the double's words) | `simd_shuffle` of both words |
| `:386` | `ex[i] = ex[i] / sum` | PTX `div.rn.f64`. SASS `MUFU.RCP64H R3, R15` (low word 1); `DFMA R4, -R14, R2, 1`; `DFMA R4, R4, R4, R4`; `DFMA R4, R2, R4, R2`; `DFMA R2, -R14, R4, 1`; `DFMA R2, R4, R2, R4`; `DMUL R4, R18, R2`; `DFMA R16, -R14, R4, R18`; `DFMA R2, R2, R16, R4`; range checks `FSETP.GEU.AND P3, PT, \|R19\|, 6.5827683646048100446e-37` (numerator's high word) and `FFMA R4, RZ, R15, R3`, `FSETP.GT.AND P1, PT, \|R4\|, 1.469367938527859385e-39` (quotient's high word, denominator finite); otherwise `CALL.REL.NOINC` to `__cuda_sm20_div_rn_f64_full` | `ddiv(ex[i], sum)` |
| `:389`, `:391` | `(double) p.top_p` | `F2F.F64.F32` (hoisted out of the loop) | `d_of_f32` |
| `:392` | `cum += ex[i]` | `DADD` | `dadd` |
| `:393` | `if (cum >= (double) p.top_p) { cut = i + 1; break; }` | `DSETP.GE.AND P0, PT, R4, R2, PT` (PTX `setp.ltu.f64` to continue): NaN never cuts | `dge(cum, top_p_d)` |
| `:397` | `if (cut < p.min_keep) ...` | integer | integer |
| `:402` | `p.min_p > 0.0f` | `UFSETP.GT.AND UP0, UPT, UR12, URZ, UPT` | `gt32(min_p, 0.0f)` |
| `:403` | `thresh = sel_logit[0] + logf(p.min_p)` | section 2.5, then `FADD R4, R5, UR6` | `add32(sel_logit[0], cuda_logf(min_p))` |
| `:405` | `if (sel_logit[i] < thresh) { n_keep = i; break; }` | `FSETP.GEU.AND P0, PT, R3, R4, PT`; the break on !P0 (ordered less) | `lt32(sel_logit[i], thresh)` |
| `:408` | `smx = sel_logit[0] * inv_t` | `FMUL R5, R12, R5` | `mul32` |
| `:409` | `smx = fmaxf(smx, sel_logit[i] * inv_t)`, i = 1..n_keep−1 | `FMUL R4, R12, R25`; `FMNMX R4, R4, R5, !PT`, serial | `fmax32(smx, mul32(x, inv_t))` in i |
| `:410` | `ex[i] = exp((double) (sel_logit[i] * inv_t) - (double) smx)` | `FMUL R13, R12, R3`; `F2F.F64.F32` of it and of smx; `DADD R14, -R16, R2`; section 2.6 | `cuda_exp(dadd(d_of_f32(mul32(x, inv_t)), -d_of_f32(smx)))` |
| `:414` | `sum += ex[i]` | `DADD` chain | `dadd` |
| `:417` | `ex[i] = ex[i] / sum` | as `:386` | `ddiv` |
| `:420` | `u = philox_uniform(p.seed, p.counter + t)` | section 2.2 | section 2.2 |
| `:425` | `cum += ex[i]` | `DADD R4, R6, R4` | `dadd` |
| `:426` | `if ((double) u < cum)` | `F2F.F64.F32 R2, UR9` (hoisted); `DSETP.GT.AND P0, PT, R4, R2, PT` (PTX `setp.leu.f64` to continue): NaN never picks | `dgt(cum, d_of_f32(u))` |
| `:429` | `*prob_out = n_keep > 0 ? (float) ex[pi] : 1.0f` (`<true>` only) | `F2F.F32.F64` | `f32_of_d` |
| `:434-435` | `pq = (float) ex[i]`; `if (n_keep == 0 && sel_ids[0] == q) pq = 1.0f;` | `F2F.F32.F64 R0, R2`; `@!P0 FSEL R0, R0, 1, P1` | `f32_of_d`; select |

- **Where the sign of zero and NaN can enter.** sel_logit is a selection-order list: finite values, +inf, and the
  -inf sentinel of an empty round, never NaN (no round takes a NaN). So mx is sel_logit[0] up to the sign of zero
  (-0 and +0 tie and the lower id comes first, so a -0 can lead a +0); x − mx is then ±0 for the leading entries,
  and the exponential of either zero is 1 (section 2.6 gives exactly 1.0 for both), so `fmax32`'s choice between
  zeros changes no bit. The same holds for smx. A +inf logit makes x − mx NaN (inf − inf), and NaN then flows through
  `exp`, the sums and the divisions: no cut fires (`:393`) and no draw lands (`:426`), so the pick is
  `sel_ids[n_keep − 1]` (`sampler_parity.cpp:1366-1367`). The binary64 helpers must propagate NaN and keep every
  comparison with it false.
- **`:374` is correctly rounded.** `rcp.rn.f32` is IEEE: the fast path is one Newton step on `MUFU.RCP` for
  exponents 1..252, and the slow path covers zero, subnormal, very large, infinite and NaN T. The `FADD.FTZ` negates
  the step's error term, which is either zero or far above the subnormal range when T's exponent is in range, so it
  flushes nothing. A subnormal temperature is greater than zero to CUDA and has a reciprocal of 2^126 or more
  (infinity below about 2^-128); a T of 2^126 or more has a subnormal reciprocal. Both need `rcp32`'s emulation.
- **`:393`, `:426` order NaN out.** `DSETP.GE` and `DSETP.GT` are ordered; the PTX writes the loop-continue
  conditions as `setp.ltu.f64` and `setp.leu.f64`, the same predicates negated.
- **`:429`, `:434` round once.** `cvt.rn.f32.f64` rounds to nearest even and keeps float subnormals, which are
  reachable: a kept entry with a probability below 2^-126 (about 1.2e-38, a logit gap above about 87 at temperature
  1) yields a subnormal `qprob` or `prob_out`.
- **Subnormals in binary32.** `mul32` at `:408-410` meets one only for a logit or product below 2^-126 in magnitude;
  `fmax32` and the comparisons need the keys for subnormal logits.

### 2.5 CUDA's `logf` (`:403`)

The library `logf` of the IEEE build (`__nv_logf`), inlined; not correctly rounded, so the port transcribes it rather
than calling a Metal logarithm. PTX from `sampler_one_block_kernel`, constants as fp32 bits; the SASS of
`sampler_split_merge_kernel` runs the same operations on the uniform datapath.

```
x  = min_p                                          // positive here: :402 has tested min_p > 0
s  = x < 0x00800000 (FLT_MIN)                       // setp.lt.f32 (FSETP.GEU ... 1.1754943508e-38, negated)
xs = s ? x * 0x4B000000 (2^23) : x                  // mul.f32 (FMUL, IEEE: exact for a subnormal x)
ea = s ? 0xC1B80000 (-23.0) : 0.0                   // selp.f32 (FSEL / UFSEL)
ib = bits(xs) - 0x3f2aaaab;  eb = ib & 0xff800000   // integer
m  = as_float(bits(xs) - eb)                        // m in [2/3, 4/3)
e  = fma(float((int) eb), 0x34000000 (2^-23), ea)   // cvt.rn.f32.s32; fma.rn.f32
f  = m + 0xBF800000 (-1.0)                          // add.f32
p  = fma(f, 0xBE055027, 0x3E1039F6)
p  = fma(p, f, 0xBDF8CDCC)
p  = fma(p, f, 0x3E0F2955)
p  = fma(p, f, 0xBE2AD8B9)
p  = fma(p, f, 0x3E4CED0B)
p  = fma(p, f, 0xBE7FFF22)
p  = fma(p, f, 0x3EAAAA78)
p  = fma(p, f, 0xBF000000 (-0.5))
q  = f * p                                          // mul.f32
q  = fma(q, f, f)
r  = fma(e, 0x3F317218 (ln 2), q)
r  = bits(xs) > 0x7f7fffff (unsigned) ? fma(xs, +inf, +inf) : r    // +inf, NaN, negative
r  = xs == 0 ? -inf : r
```

In the SASS the first FMA is `UFFMA UR6, UR10, -UR6, 0.14084610342979431152` with `UR6 = 0x3e055027`: the negated
operand is the constant, the same product. Metal: this sequence with `fma()`, `*` and `+` (every operation after the
pre-scale sees normal values: |f| is at least 2^-24 or exactly 0), and the pre-scale of a subnormal x done exactly
from its bits, `float(as_type<uint>(x) & 0x7fffffu) * 0x1p-126f`, because Metal's multiply would flush x to zero.
The gate `:402` must then be `gt32`, which sees a positive subnormal min_p as positive. `sampler_parity` keeps its
min_p fixture off the cut's rounding edge (`sampler_parity.cpp:1029-1030`), so it does not itself demand these bits;
the transcription gives them for every min_p at the cost of fourteen float operations.

### 2.6 CUDA's binary64 `exp` (`:379`, `:410`)

The library `exp` (`__nv_exp`), inlined at both lines in every copy, with the same constants (checked in all three
kernels). PTX `fma.rn.f64`, `add.rn.f64`, `sub.f64`, `add.f64`, `mul.f64`; SASS `DFMA`, `DADD`, `DMUL`, `FSETP` and
`DSETP`. Constants as binary64 bits; d is the argument (`DADD R16, -R18, R4`: x − mx).

```
t  = dfma(d, 0x3FF71547652B82FE (1/ln 2), 0x4338000000000000 (1.5 * 2^52))
j  = dadd(t, 0xC338000000000000)                   // = t - 1.5 * 2^52, the nearest integer to d / ln 2
r  = dfma(j, 0xBFE62E42FEFA39EF (-ln2_hi), d)
r  = dfma(j, 0xBC7ABC9E3B39803F (-ln2_lo), r)
p  = dfma(r, 0x3E5ADE1569CE2BDF, 0x3E928AF3FCA213EA)
p  = dfma(p, r, 0x3EC71DEE62401315)
p  = dfma(p, r, 0x3EFA01997C89EB71)
p  = dfma(p, r, 0x3F2A01A014761F65)
p  = dfma(p, r, 0x3F56C16C1852B7AF)
p  = dfma(p, r, 0x3F81111111122322)
p  = dfma(p, r, 0x3FA55555555502A1)
p  = dfma(p, r, 0x3FC5555555555511)
p  = dfma(p, r, 0x3FE000000000000B)
p  = dfma(p, r, 0x3FF0000000000000 (1.0))
e  = dfma(p, r, 0x3FF0000000000000 (1.0))
n  = lo32(t)                                        // j as an int32: t = 1.5 * 2^52 + j
y  = (lo32(e), hi32(e) + (n << 20))                 // IMAD R5, R20, 0x100000, R15: 2^j by the exponent field
if ((hi32(d) & 0x7fffffff) >= 0x4086232B) {         // FSETP.GEU |hi(d) as f32| >= 4.1917929649 (|d| >= 708.396)
    if ((hi32(d) & 0x7fffffff) >= 0x40874800)        // FSETP.GEU |hi(d) as f32| >= 4.2275390625 (|d| >= 745)
        y = dgeu(d, 0.0) ? dadd(d, +inf) : +0.0;     // DSETP.GEU, DADD ... +INF, FSEL: +inf, NaN or +0
    else {
        h = n / 2;                                   // LEA.HI + SHF.R.S32.HI: C division, toward zero
        y = dmul((lo32(e), hi32(e) + (h << 20)), (0, ((n - h) << 20) + 0x3ff00000));   // one rounding
    }
}
```

- The range test reads the high word of d as an fp32 (`abs.f32`, `setp.lt.f32` in the PTX); that is the integer
  comparison above, for every value of the word, including words that are NaN or subnormal as fp32. A NaN d takes
  the slow branch and returns d + inf, a NaN.
- `exp(±0)`: t = 1.5 · 2^52, j = 0, r = ±0, e = fma(p, ±0, 1) = 1.0 exactly.
- The near-overflow and subnormal branch splits 2^n into 2^h · 2^(n−h) so that one `DMUL` rounds the subnormal
  result once. The port follows every step with the binary64 helpers; the result is not correctly rounded, so a
  different exp, even a correctly rounded one, gives different bits.

### 2.7 `warp_merge_lists` (`:536-575`)

Integer bookkeeping, `__int_as_float` moves (`:551`, `:567`) and `take_first` / `warp_first` (section 2.3): an XOR
butterfly over offsets 16, 8, 4, 2, 1 (`SHFL.BFLY ... 0x10`, `0x8`, `0x4`, `0x2`, `0x1`), each step the `:346`
predicate. Metal: `simd_shuffle_xor` on the value's bits in that offset order, then `take_first` with `gt32` and
`eq32`. No arithmetic.

## 3. Per-kernel tables

### 3.1 `sampler_greedy_kernel` (`sampler.cu:91-157`)

Unit `g1_greedy`. The scan loop `:131-134` is unswitched on `p.logit_bias`: one copy reads `l[v]` alone, the other
reads both arrays and adds; each copy inlines `apply_penalties` once.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:132` | `p.logit_bias ? l[v] + p.logit_bias[v] : l[v]` | biased copy: `FADD R18, R12, R17` | `add32(l[v], bias[v])` |
| `:132` | `apply_penalties(..., hit_count(v), p)` | section 2.1, once per loop copy (`:73` at 5232 and 8752) | section 2.1 |
| `:133` | `if (s > bv) { bv = s; best = v; }` | `FSETP.GT.AND P1, PT, R2, R9, PT`; `SEL R8, R15, R8, P1`; `FSEL R9, R2, R9, P1` | `gt32(s, bv)` |
| `:135-138` | `__shfl_down_sync` of bv and best, offsets 16, 8, 4, 2, 1; `ov > bv \|\| (ov == bv && oi < best)` | per step `SHFL.DOWN PT, R0, R9, 0x10, 0x1f` (value), `SHFL.DOWN PT, R3, R8, 0x10, 0x1f` (index), then section 2.3's first form: `PLOP3.LUT P0, PT, PT, PT, PT, 0x80, 0x8` (P0 true); `FSETP.GEU.AND P1, PT, R9, R0, PT`; `@P1 ISETP.GE.AND P2, PT, R3, R8, PT`; `@P1 FSETP.EQ.AND P0, PT, R9, R0, !P2`; `FSEL R0, R0, R9, P0`; `SEL R3, R3, R8, P0` | `simd_shuffle_down` of `as_type<uint>(bv)` and best by 16, 8, 4, 2, 1, then `gt32(ov, bv) \|\| (eq32(ov, bv) && oi < best)` |
| `:147-148` | `lane < nw ? sv[lane] : -inf`, `... : n_vocab` | integer selects | integer |
| `:149-152` | the same tree over the warps' results | four steps as `:138`; the fifth keeps only the index: `FSETP.NEU.AND P1, PT, R2, R3, PT`; `FSETP.GEU.AND P0, PT, R2, R3, PT`; `@!P1 VIMNMX.S32 R0, R0, R5, PT`; `SEL R5, R5, R0, !P0` | as `:138` |
| `:155` | `(wi < n_vocab) ? wi : 0` | `ISETP.GE.AND P0, PT, R5, R12, PT`; `SEL R5, R5, RZ, !P0` | integer |
| `:157` | (none) | the instructions after the `EXIT` at 11296 are `__internal_1_$__cuda_sm3x_div_rn_noftz_f32_slowpath`, called from both copies of `:74` | `div32` |

- **The pick is order-free.** Every element's value is computed on its own (`:132`), and the scan and both trees
  take the first element of the selection order (section 2.3), so the token is a function of the row: the lowest
  index of the largest value above -inf, NaN never taken, and 0 when nothing qualifies (`:155`). The Metal kernel may
  therefore use any threadgroup size and any partition of the row; what it must keep is the key comparison at
  `:133`, `:138` and `:152`, the IEEE add at `:132` and section 2.1's arithmetic.
- **The shuffle's edge lanes do not matter.** `SHFL.DOWN` with clamp `0x1f` returns the caller's own value to the
  top `off` lanes, and an own pair never replaces itself (`ov == bv` and `oi == best`). Only lane 0's result is
  used, and every lane that feeds it at step `off` lies below 32 - `off`, so its source lane is in range; what
  `simd_shuffle_down` returns to the top lanes cannot reach the token.
- **Subnormals.** A subnormal logit, bias or sum is ordered as itself by CUDA. `:132`'s add needs `add32`'s
  emulation only when an operand or the sum is subnormal (a bias of about -x on a logit x, say).

### 3.2 `sampler_greedy_cluster_kernel` (`sampler.cu:188-244`)

Unit `g1_greedy`. The four-load loop `:203-209` is unrolled four times (16 takes per pass) and the tail loop `:210`
unrolled as well; the code after the kernel's last `EXIT` (8992) is the divergent-warp form of the `:212` and `:220`
shuffles (`WARPSYNC.COLLECTIVE`, `SHFL.DOWN` on register operands, `ENDCOLLECTIVE`), with the same merge predicate.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:181` (`:205-210`) | `am_take`: `if (s > bv) { bv = s; best = v; }` | `FSETP.GT.AND P3, PT, R13, R6, PT`; `FSEL R13, R13, R6, P3`, a serial chain over the thread's elements in ascending v (57 copies) | `gt32(s, bv)` |
| `:184` (`:212`, `:220`, `:237`) | `am_merge`: `ov > bv \|\| (ov == bv && oi < best)` | section 2.3's second form: `ISETP.GE.AND P0, PT, R7, R4, PT`; `FSETP.NEU.OR P0, PT, R6, R3, P0`; `FSETP.GEU.AND P0, PT, R6, R3, P0`; `@!P0 MOV R4, R7`; `@!P0 MOV R6, R3`, after `SHFL.DOWN` of value and index by `0x10` ... `0x1` | `gt32(ov, bv) \|\| (eq32(ov, bv) && oi < best)`, shuffles by 16, 8, 4, 2, 1 |
| `:217-218` | `bv = sv[lane]; best = si[lane]` (32 warps: every lane valid) | `LDS` | threadgroup loads |
| `:222-232` | cluster barriers; CTA r's pair stored to CTA 0's `cv[r]`, `ci[r]` through `mapa` | `UCGABAR_ARV`, `UCGABAR_WAIT`, `ST.E` to the mapped window; no float | the partials dispatch writes its pair to a buffer slot [row][r] |
| `:234-235` | `lane < kAmCtas ? cv[lane] : -inf`, `... : n_vocab` | `MOV R0, 0xff800000` and loads under `ISETP.GT.U32 ... 0x7` | the merge dispatch reads slots 0..7 |
| `:238` | `(best < n_vocab) ? best : 0` | `ISETP.GE.AND`; `SEL` | integer |

- **Two dispatches give its bits.** No CTA cluster exists on this GPU: one dispatch of 8 threadgroups of 1,024
  threads per row runs `:198-221` and stores each threadgroup's pair, and a second, one SIMD-group per row, runs
  `:233-238` on the 8 pairs. The token is the first element of the selection order, as in section 3.1, so it equals
  the one-block kernel's and the host's answer for any decomposition, provided every comparison is the key form.
  `decode_cluster_parity` demands exactly that, `if (a[(size_t) t] != b[(size_t) t] || a[(size_t) t] != hb) ++bad;`
  (`src/kernels/decode_cluster_parity.cpp:237`), over fixtures that include ties across CTAs, NaN, ±inf, signed zeros
  and subnormals: `aDenormal` sets `l[v] = (float) (at(rng) % 5) * 1e-42f - 1e-30f * (v & 1)` (`:205`), whose largest
  values are subnormals that `>` on this GPU would see as zero.

### 3.3 `sampler_greedy_top_prob_cluster_kernel` (`sampler.cu:249-325`)

Unit `g1_greedy`. Lines `:262-305` are section 3.2's argmax line for line (57 `FSETP.GT`/`FSEL` pairs at `:181`,
the `:184` merges), ending with `pick` in CTA 0's shared memory instead of `out`. Then CTA 0 alone (the other CTAs
have returned at `:297`) computes the pick's probability with `row_top_prob_kernel`'s arithmetic
(`src/kernels/cuda/verify_kernels.cu:580-596`). This is the only approximate operation in `sampler.cu`, listed in
the contract (`flash-next.md`, approximate operations in IEEE-built kernels).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:309` | `const float m = l[id]` | `LDG.E.CONSTANT R6` | load |
| `:311` | `for (i = threadIdx.x; i < n_vocab; i += blockDim.x) s += __expf(l[i] - m)`, from `s = 0.0f` | PTX `sub.f32`, `mul.f32 ... 0f3FB8AA3B`, `ex2.approx.f32` (no `.ftz`), `add.f32`. SASS `FADD R12, -R6, R9`; `FMUL R12, R12, 1.4426950216293334961`; `FSETP.GEU.AND P0, PT, R12, -126, PT`; `@!P0 FMUL R12, R12, 0.5`; `MUFU.EX2 R13, R12`; `@!P0 FMUL R13, R13, R13`; `FADD R10, R13, R10`, one element per iteration (not unrolled) | `s = s + ex2_approx(mul32(add32(l[i], -m), 0x1.715476p+0f))` with `ex2_approx(y) = metal::fast::exp2(y)`, the thread's elements in ascending i |
| `:312` | `for (o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o)` | `SHFL.BFLY PT, R3, R10, 0x10, 0x1f`; `FADD R3, R3, R10`; then `0x8`, `0x4`, `0x2`, `0x1`, one `FADD` each | `s = s + simd_shuffle_xor(s, o)` for o = 16, 8, 4, 2, 1 (an add is commutative, so both lanes of a pair hold the same bits) |
| `:313` | `part[threadIdx.x >> 5] = s` (lane 0 of each warp) | `STS` | threadgroup store |
| `:317` | `float tot = 0.0f; for (w = 0; w < blockDim.x >> 5; ++w) tot += part[w]` | `MOV R19, RZ`; `LDS.128`; `FADD R24, R24, R19` (the first add, 0 + part[0], is kept); `FADD R25, R24, R25` ... a serial chain in w, unrolled by 16 with 8-, 4- and 1-wide remainders for other block sizes (unused at 1,024 threads: two passes of 16) | `tot = 0.0f; tot = tot + part[w]` for w = 0..31 |
| `:318` | `probs[blockIdx.y] = 1.0f / tot` | PTX `rcp.rn.f32`. SASS as `:374` (section 2.4): `IADD3 R0, PT, PT, R19, 0x1800000, RZ`; `LOP3.LUT R0, R0, 0x7f800000, ...`; `ISETP.GT.U32.AND P0, PT, R0, 0x1ffffff, PT`; `MUFU.RCP R0, R19`; `FFMA R2, R0, R19, -1`; `FADD.FTZ R11, -R2, -RZ`; `FFMA R11, R0, R11, R0`; else `CALL.REL.NOINC` to `__cuda_sm20_rcp_rn_f32_slowpath` (the `sm_30_intrinsics.hpp:397` block after the last `EXIT`) | `1.0f / tot` (correctly rounded: tot is at least 1 or NaN, never subnormal) |
| `:319` | `out[blockIdx.y] = map != nullptr ? map[id] : id` | integer | integer |

- **`ex2.approx` keeps subnormal results on CUDA; this GPU's are [H].** `MUFU.EX2` alone flushes a subnormal
  result, so for the non-ftz `ex2.approx.f32` `ptxas` halves an argument below -126, raises 2 to it and squares the
  result with an IEEE multiply: a term 2^y for y in [-149, -126) comes out subnormal instead of zero. What
  `metal::fast::exp2` returns there on this GPU is [H], and a Metal multiply would flush the square anyway. The
  difference cannot reach `probs` except through exact rounding ties: the pick's own term is `ex2(0) = 1`, so `tot`
  is at least 1, and a term below 2^-126 changes a partial sum only while that partial is below about 2^-102, far
  under half an ulp of anything it later meets on the way to `tot`. The record recommends `fast::exp2` alone,
  without emulating the halve-and-square branch.
- **What basalt tests here is approximate.** `sampler.hpp:57-59` says the kernel equals `sample_greedy_cluster`,
  `row_top_prob` and `map_ids` bit for bit, but no test compares it with `row_top_prob`. The q5k and q6k head tests
  use it as the reference for the fused head's probability at a relative bound, `if (logit_bad || id_bad || max_rel
  > 1e-5) fail = 1;` (`src/kernels/q6k_head_parity.cpp:145`, the same at `src/kernels/q5k_head_parity.cpp:145`), and
  its token goes through section 3.2's bitwise check. The drafter runs this kernel, or `row_top_prob` with the
  argmax when the cluster kernel cannot run (`src/core/mtp.cpp:1036-1042`). One Metal probability kernel, with the
  same `ex2_approx` helper and the same order, can serve both calls, so they stay bitwise equal.
- **Dispatch shape.** The argmax may be split as in section 3.2. The sum is not order-free: it has to run as one
  threadgroup of 1,024 threads per row, thread i summing elements i, i + 1,024, ... in order, then the five
  butterfly steps and the serial finish over 32 partials from +0.
- **Rows without a finite maximum.** For an all-NaN or all -inf row, or a +inf pick, `l[id] - m` is NaN at the
  pick, so CUDA writes a NaN probability, as it does for a NaN logit anywhere in the row. `fast::exp2` returns
  0x7fc00000 for a NaN on this GPU (`nan.md`, measured), and the following `add32` propagates it. The store carries
  sm_120's NaN, 0x7fffffff (`cuda_nan`, invariant 5; `nan.md`).

### 3.4 `sampler_one_block_kernel` (`sampler.cu:446-518`)

Unit `g2_one_block`. As in section 3.1, the scan `:489-492` is unswitched on `p.logit_bias`, each copy with its own
`apply_penalties`. After the kernel's last `EXIT` (40688) come the divergent-warp copy of the `:503-507` tree and
three slow paths: `__cuda_sm20_div_rn_f64_full` (from `:386` and `:417`), `__cuda_sm20_rcp_rn_f32_slowpath` (from
`:374`) and `__cuda_sm3x_div_rn_noftz_f32_slowpath` (from `:74`).

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:490` | `p.logit_bias ? l[v] + p.logit_bias[v] : l[v]` | biased copy: `FADD R2, R18, R23` | `add32` |
| `:490` | `apply_penalties(..., hit_count(v), p)` | section 2.1, once per loop copy | section 2.1 |
| `:491` | `if ((s < prev_v \|\| (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }` | section 2.3's third form: `FSETP.GEU.AND P3, PT, R2, R13, PT`; `@P3 ISETP.GT.AND P2, ...`; `@P3 FSETP.EQ.AND P0, PT, R2, R13, P2`; `FSETP.GT.AND P0, PT, R2, R15, P0` (attributed to `:489`); `FSEL R15, R2, R15, P0` | `(lt32(s, prev_v) \|\| (eq32(s, prev_v) && v > prev_i)) && gt32(s, bv)` |
| `:493-497` | the shuffle-down tree, 16, 8, 4, 2, 1 | section 2.3's first form, five steps | as section 3.1 `:138` |
| `:501-508` | `lane < nw ? sv[lane] : -inf`, then the same tree on warp 0 | five steps in the converged path and again in the divergent copy | as section 3.1 `:138` |
| `:509` | `sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv;` | integer select; `STS` | integer; the value's bits stored unchanged |
| `:513-514` | `prev_v = sel_logit[i]; prev_i = sel_ids[i];` | `LDS` | threadgroup loads |
| `:517` | `sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex, nullptr, query, nq, qprob)` | section 2.4 (`<false>`, `:374-437`), with sections 2.2, 2.5 and 2.6 inlined | section 2.4 |

- **Each round is order-free; the tail is not.** Round i takes the first element of the selection order after round
  i-1's pick, a function of the penalised row alone, so the scan may use any threadgroup size and partition. The
  sentinel after an empty round, (-inf, 0), admits a -inf logit at v > 0 to `:491`'s first clause, but `s > bv` from
  -inf rejects it, so every later round is empty too. The tail's order is fixed by its source: the exponentials and
  quotients are per entry, and the sums and scans run serially in i on lane 0 (section 2.4).
- **The penalty's subnormal cases** are section 2.1's; the round comparisons need the keys for a subnormal logit.

### 3.5 `sampler_split_part_kernel` (`sampler.cu:582-661`)

Units `g3_split_part` (the real kernel, through 21 of its 32 penalty copies; section 1) and `g3b_split_part_tail`
(the penalty block blanked) for `:630-661`. After the tail unit's last `EXIT` (17936) comes only the divergent-warp
copy of the rounds' `warp_first`; the real unit's `:74` slow path lies beyond its listing's cut.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:619` | `s[j] = v < n_vocab ? (p.logit_bias ? l[v] + p.logit_bias[v] : l[v]) : -inf` | biased version: `@!P1 FADD R30, R5, R0` per element, predicated on `v < n_vocab` (32 copies); `MOV ..., 0xff800000` past the vocabulary | `add32(l[v], bias[v])`; -inf as bits |
| `:626` | `s[j] = apply_penalties(s[j], history_count(hrow, hlen, v), p)` on a bitmap hit | section 2.1, unrolled 32 times (`:73` `FSETP.GTU` and `@!P FMUL`; `:74`'s seven-instruction division and its slow-path call; `:75`'s `I2FP.F32.U32`, `FFMA`, `FADD` in each copy) | section 2.1 |
| `:634` | `for (; i < k; ++i)` | `FSETP.NE.AND P1, PT, R68, -INF, PT`: s[1] > -inf, the odd chain's first test hoisted (section 2.3) | `gt32(s[1], -inf)`, inside or outside the loop |
| `:643` | even chain, j = 0, 2, ..., 30: `if ((x0 < prev_v \|\| (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }` | section 2.3's branching form of the third predicate; for j = 0, `FSETP.NE.AND P2, PT, R70, -INF, PT` in place of `x0 > b0` | as `:491` (section 3.4) |
| `:644` | odd chain, j = 1, 3, ..., 31, the same on x1, b1, i1 | section 2.3's predicated form: `FSETP.GEU.AND P3, PT, R64, R5, PT`; `PLOP3.LUT P0, ...` (true); `@P3 ISETP.GT.AND P2, PT, R65, R4, PT`; `@P3 FSETP.EQ.AND P0, PT, R64, R5, P2`; `FSETP.GT.AND P0, PT, R64, R75, P0`; `FSEL R75, R64, R75, P0`; `SEL R72, R65, R72, P0` | as `:491` |
| `:646` | `take_first(b0, i0, b1, i1)` | `ISETP.GE.AND P0, PT, R72, R69, PT`; `FSETP.NEU.OR P0, PT, R4, R71, P0`; `FSETP.LEU.AND P0, PT, R4, R71, P0`; `@!P0 MOV R71, R4`; `@!P0 MOV R69, R72` | `gt32(b1, b0) \|\| (eq32(b1, b0) && i1 < i0)` |
| `:647` | `warp_first(b0, i0)` | `SHFL.BFLY` of index and value by `0x10`, `0x8`, `0x4`, `0x2`, `0x1`, each step section 2.3's second form; the fifth step branches (`FSETP.GEU.AND P0, PT, R71, R74, PT`, then `ISETP.GE` and `FSETP.NEU.OR`) | `simd_shuffle_xor` by 16, 8, 4, 2, 1, then `take_first` |
| `:648-653` | `if (i0 >= n_vocab) break;`, `wl[warp][i] = make_int2(i0, __float_as_int(b0))`, prev, sentinels `(n_vocab, 0xff800000)` | integer; `STS.64` | integer; the value's bits |
| `:657-659` | `warp_merge_lists(&wl[0][0], 4, kSelMax, k, n_vocab, ...)`, `dst[r] = make_int2(id, __float_as_int(v))` | section 2.7, with the head pair's `take_first` at `:557` in the branching form (`FSETP.GT.AND P0, PT, R11, R9, PT`; `ISETP.GE.AND P0, PT, R10, R8, PT`; `FSETP.NEU.OR P0, PT, R11, R9, P0`) and the butterfly in the second form; `STG.E.64` | section 2.7 |

- **Every round is a function of the warp's 1,024 values.** Each chain walks ascending ids with a strict `>`, so it
  keeps its own first element after (prev_v, prev_i), and `take_first` and the butterfly take the first of those:
  round i yields the i-th element of the selection order among the warp's logits. The block list, and the row list
  `sampler_split_merge_kernel` merges from it, are therefore the first k of the row, as in the one-block kernel, and
  every value they carry has the bits section 2.1 gives it. The split itself (32 logits per lane, 4 warps, 4,096 per
  block) is free on Metal; the key comparisons and the bit-exact penalised values are what the port must keep.
- **The penalised values must be the one-block kernel's bits.** `sampler_parity` checks both paths' picks against
  the same host transcriptions (`for (int t = 0; t < n_tokens; ++t) if (got[(size_t) t] != want[(size_t) t]) ++bad;`,
  `src/kernels/sampler_parity.cpp:124`; 70 rows take the one-block kernel, `:393`, `:1610`), so one
  `apply_penalties` helper serves both.

### 3.6 `sampler_split_merge_kernel` (`sampler.cu:665-682`)

Unit `g4_split_merge`; section 2.4's SASS is this kernel's. After its last `EXIT` (39072) come
`__cuda_sm20_div_rn_f64_full` (from `:386` and `:417`) and `__cuda_sm20_rcp_rn_f32_slowpath` (from `:374`),
attributed to `:682`. One warp per row and `SamplerParams` a kernel parameter, so the chain's scalar parts run on
the uniform datapath (`UFSETP` at `:378` and `:402`, `logf` as `UFMUL`/`UFFMA`, Philox as `UIMAD.WIDE.U32`), and no
divergent-warp copies exist.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:675` | `lists[e] = src[e]` | `LDG`/`STS`, integer | copy |
| `:677-679` | `warp_merge_lists(lists, n_blocks, k, k, n_vocab, ...)`; `sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v;` | section 2.7: the head pair's `take_first` (`:557`) branching (`FSETP.GT.AND P0, PT, R13, R5, PT`; `ISETP.GE.AND`; `FSETP.NEU.OR P0, PT, R13, R5, P0`), then five butterfly steps in section 2.3's second form | section 2.7; the value's bits stored unchanged |
| `:681` | `sampled_tail_warp(sel_ids, sel_logit, k, p, t, out, ex, nullptr, query, nq, qprob)` | section 2.4 (`<false>`), with sections 2.2, 2.5 and 2.6 | section 2.4 |

- **The list is the one-block kernel's.** The merge takes the first k of the union of ordered, disjoint lists
  under a strict total order, so `sel_ids` and `sel_logit` equal `sampler_one_block_kernel`'s `:509` entries, value
  bits included, and the tail then computes the same token and `qprob` (section 2.4).

### 3.7 `coupled_stage_kernel` (`sampler.cu:692-700`)

Unit `g5_coupled`. No floating-point instruction: the 18 words of `SamplerParams` (`ISETP.GE.U32.AND P0, PT, R0,
0x12, PT`) and the history base are copied with `LDG.E.STRONG.SYS` (the `volatile` reads of mapped host memory) and
`STG.E`, and `coupled_hist_len` (`coupled_draft.hpp:71-73`) is `ISETP.GT`, `VIMNMX.S32` and `SEL`. The Metal form
copies words; the parameters' float bits pass through unchanged.

### 3.8 `coupled_penalize_kernel` (`sampler.cu:706-726`)

Unit `g5_coupled`. `p` is loaded from `*dp`, so the penalties sit in registers rather than the constant bank. The
history loop `:717-725` is unrolled twice, each copy with its own `apply_penalties`; after the last `EXIT` (8496)
comes `__cuda_sm3x_div_rn_noftz_f32_slowpath`, attributed to `:726`.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:723` | `if (atomicOr(&seen[s >> 5], bit) & bit) continue;` | integer | `atomic_fetch_or_explicit` on threadgroup memory |
| `:724` | `logits[s] = apply_penalties(logits[s], history_count(hrow, h, v), p)` | section 2.1, with register operands: `FSETP.GTU.AND P0, PT, R3, RZ, PT`; `@!P0 FMUL R0, R6, R3`; `MUFU.RCP R15, R6`; `FCHK P0, R3, R6`; `FFMA R0, -R6, R15, 1`; `FFMA R0, R15, R0, R15`; `FFMA R15, R3, R0, RZ`; `FFMA R14, -R6, R15, R3`; `FFMA R0, R0, R14, R15`; `CALL.REL.NOINC` to the slow path; `I2FP.F32.U32 R18, R18`; `FFMA R3, R7, R18, R2`; `FADD R3, -R3, R0` | section 2.1 |

- **The same value as the target's.** `-R6 · R15` and `R15 · -R6` are the same exact product inside an `FFMA`, so
  the division's operand order differs from section 2.1's copy without changing a bit. The draft logit gets the
  target's penalised value only if both use one `apply_penalties` helper. Which thread wins `atomicOr` decides only
  which thread writes; every candidate writer computes the same value.
- **A NaN reaches the row.** A counted NaN logit, or a counted +inf less an infinite frequency or presence term,
  leaves `FADD` as sm_120's 0x7fffffff; the Metal store rewrites this GPU's 0x7fc00000 to it (`cuda_nan`; `nan.md`).

### 3.9 `coupled_part_kernel<8, 16>` (`sampler.cu:733-777`)

Unit `g5_coupled`, the one instantiation the host launches (`:1066`: 16 warps of 8 logits per lane, 4,096 per
block). Section 3.5's rounds with no bias and no penalty (the logits arrive penalised from section 3.8): the same
forms at the corresponding lines, four elements per chain. After the last `EXIT` (5632) comes only the
divergent-warp copy of the rounds' `warp_first`.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:746` | `s[j] = v < n_vocab ? l[v] : -inf` | loads; `MOV R12, 0xff800000` | load; -inf as bits |
| `:752` | `for (; i < k; ++i)` (k from `dp->top_k`) | `FSETP.NE.AND P1, PT, R10, -INF, PT`: s[1] > -inf hoisted | as section 3.5 `:634` |
| `:759` | even chain, j = 0, 2, 4, 6 | as section 3.5 `:643` | as `:491` |
| `:760` | odd chain, j = 1, 3, 5, 7 | as section 3.5 `:644` | as `:491` |
| `:762` | `take_first(b0, i0, b1, i1)` | `ISETP.GE.AND P0, PT, R28, R25, PT`; `FSETP.NEU.OR P0, PT, R4, R27, P0`; `FSETP.LEU.AND P0, PT, R4, R27, P0` | as section 3.5 `:646` |
| `:763` | `warp_first(b0, i0)` | as section 3.5 `:647` (`ISETP.GE.AND P0, PT, R28, R25, PT`; `FSETP.NEU.OR P0, PT, R27, R4, P0`; `FSETP.GEU.AND P0, PT, R27, R4, P0` per step) | as section 3.5 `:647` |
| `:773-775` | `warp_merge_lists(&wl[0][0], 16, kSelMax, k, n_vocab, ...)`, `dst[r] = make_int2(id, __float_as_int(v))` | section 2.7, the head pair's `take_first` branching as at section 3.5 `:657` | section 2.7 |

- **The lists are the split's.** The selection order is a strict total order, so the block's first k is the same
  whatever the per-lane and per-warp split; `coupled_merge_kernel` then sees the lists `sampler_split_part_kernel`
  would build from the same penalised row.

### 3.10 `coupled_merge_kernel` (`sampler.cu:782-818`)

Unit `g6_coupled_merge`. One warp; `p` is a register copy of `*dp`, so the tail's scalar tests are ordinary
`FSETP` (`:378` is `FSETP.GEU.AND P0, PT, R13, 1, PT`) rather than section 3.6's uniform ones. After the last
`EXIT` (33392) come `__cuda_sm20_div_rn_f64_full` and `__cuda_sm20_rcp_rn_f32_slowpath`, attributed to `:818`.

| Line | basalt | SASS | Metal |
|---|---|---|---|
| `:788` | `p.counter = coupled_draft_counter((int64_t) step_rec[0])`, that is `(uint64_t) (cell + 1)` (`coupled_draft.hpp:68`) | `LDG.E.CONSTANT R6`; `SHF.R.S32.HI R7, RZ, 0x1f, R6` (sign extension), a 64-bit add | `ulong(long(step_rec[0]) + 1)`, read from the step-record buffer |
| `:796-798` | `b = e / k, r = e - b * k; lists[b * kSelMax + r] = cand[b * kSelMax + r]` | `ptxas`'s integer division (`I2F.RP`, `MUFU.RCP`, `F2I.FTZ.U32.TRUNC.NTZ`, `IMAD.HI.U32`), exact | integer `/` |
| `:801-803` | `warp_merge_lists(lists, n_blocks, kSelMax, k, nv, ...)`; `sel_ids[i] = id < nv ? id : 0; sel_logit[i] = v;` | section 2.7, as in section 3.6 (`FSETP.GT.AND x1`, `FSETP.NEU.OR x6`, `FSETP.GEU.AND x5` at `:346`) | section 2.7 |
| `:805` | `if (p.greedy \|\| p.temperature <= 0.0f)` | `FSETP.GTU.AND P0, PT, R5, RZ, PT`; `@P0 BRA P1, ...` to the sampled path (P1: not greedy) | `!greedy && !(temperature <= 0)` takes the tail, so a NaN temperature samples: `gtu32(temperature, 0.0f)` |
| `:806` | `pick[0] = sel_ids[0]; prob[0] = 1.0f;` | `HFMA2 R3, -RZ, RZ, 1.875, 0` (the constant 0x3f800000); `STS.64` | integer and the constant 1.0f |
| `:808` | `sampled_tail_warp<true>(sel_ids, sel_logit, k, p, 0, pick, ex, prob)` | section 2.4 with t = 0; `:429`: `MOV R0, 0x3f800000`; `@P0 LDS.64 R2, [R2]`; `@P0 F2F.F32.F64 R0, R2` | section 2.4 |
| `:812-816` | `id = sub_to_id != nullptr ? sub_to_id[s] : s`; `*out_id = id; *out_prob = prob[0]; ring[cap + j] = id;` | integer; `STG` | integer; the probability's bits |

- **A NaN temperature.** `:805` sends it to the tail, where `:374`'s ordered `FSETP.GT` gives `inv_t = 0`: every
  scaled logit is ±0 or NaN, so the draw is uniform over the kept list, or lands nowhere when a kept entry is
  infinite, a -inf sentinel included (inf · 0 is NaN; section 2.4). The Metal gate must be the unordered form to
  match. `sample_tokens_probe` makes the same test on the host (`:991`), where the comparison is IEEE as well. The
  verify window calls it again only for a sampled request (`temperature > 0.0f`, false for a NaN), a penalty window
  or a logit bias (`src/core/verify.cpp:2496-2498`, `:2659`), so the target's window samples a NaN temperature only
  with a window or a bias (`basalt_sampler_mode`).
- **`:429` reaches subnormals.** `*out_prob` is `f32_of_d(ex[pi])`, a subnormal for a kept entry below 2^-126
  (section 2.4).

## 4. What the record implies for the Metal port

### 4.1 Helpers by kernel

"Keys" is section 2.0's comparison set (`key32` and its ordered and unordered forms); "§2.1" is `apply_penalties`'
`gtu32`, `mul32`, `div32`, `fma32` and `add32`; "tail" is section 2.4 with sections 2.2, 2.5 and 2.6.

| Kernel | Keys | binary32 arithmetic | binary64 | Library sequences | Subnormals reachable through |
|---|---|---|---|---|---|
| `sampler_greedy_kernel` (3.1) | yes | `add32` (bias), §2.1 | no | no | logits, bias, penalty parameters, penalised values |
| `sampler_greedy_cluster_kernel` (3.2) | yes | none | no | no | logits (`decode_cluster_parity`'s `aDenormal`) |
| `sampler_greedy_top_prob_cluster_kernel` (3.3) | yes | `add32`, `mul32`, `rcp32` (tot at least 1) | no | `ex2_approx` = `fast::exp2`, the only approximate site | terms below 2^-126 (approximate output) |
| `sampler_one_block_kernel` (3.4) | yes | `add32` (bias), §2.1; tail: `rcp32`, `fmax32`, `mul32`, `add32` | tail | `cuda_logf`, `cuda_exp` | logits, bias, penalties, temperature, min_p, `qprob` |
| `sampler_split_part_kernel` (3.5) | yes | `add32` (bias), §2.1 | no | no | logits, bias, penalties |
| `sampler_split_merge_kernel` (3.6) | yes | tail | tail | `cuda_logf`, `cuda_exp` | logits, temperature, min_p, `qprob` |
| `coupled_stage_kernel` (3.7) | no | none | no | no | none (word copies) |
| `coupled_penalize_kernel` (3.8) | `gtu32` (§2.1) | §2.1 | no | no | logits, penalties |
| `coupled_part_kernel<8, 16>` (3.9) | yes | none | no | no | logits |
| `coupled_merge_kernel` (3.10) | yes, and `gtu32` at `:805` | tail | tail | `cuda_logf`, `cuda_exp` | logits, temperature, min_p, `out_prob` |

`philox_uniform` (section 2.2) is integer arithmetic plus one exact scaling, so Metal's `uint` operations give its
bits.

### 4.2 What basalt's tests demand

- **Tokens, bitwise.** `sampler_parity` compares every pick of both sampled paths and the greedy kernel with host
  transcriptions (`src/kernels/sampler_parity.cpp:124`), and `decode_cluster_parity` compares the cluster argmax
  with the one-block kernel and with the host (`src/kernels/decode_cluster_parity.cpp:237`). The host side keeps
  subnormals and orders them as themselves, so the key comparisons are required wherever a subnormal logit can
  meet another value; `aDenormal` (`:205`) is such a fixture. The host transcriptions compute the tail with the
  host library's `std::exp` and `std::log`, not CUDA's sequences, so their fixtures keep cuts and draws away from
  the last bits (as for min_p, section 2.5). The tests therefore do not by themselves pin `cuda_exp`'s or
  `cuda_logf`'s bits; those are recorded for the port's own exactness (invariant 5's rounding sequence).
- **`qprob`, bitwise across widths and within `1e-6` of double.** `if (std::memcmp(&one, &pr[(size_t) t], 4) != 0
  || pick1 != ref[(size_t) t]) ++width_diff;` (`src/kernels/sampler_parity.cpp:457`) compares a row's probe
  probability alone with the same row in a window, and `max_abs > 1e-6` (`:459`) bounds it against a double
  mirror. Each row's tail depends only on its own list, `p` and `p.counter + t`, so a deterministic Metal tail meets
  the first; the second accepts any faithful binary64 arithmetic.
- **The greedy pick's probability, approximately.** Only `max_rel > 1e-5` against the fused heads
  (`src/kernels/q6k_head_parity.cpp:145`, `src/kernels/q5k_head_parity.cpp:145`); section 3.3.
- **The host mirrors' penalty expression.** `sampler_parity.cpp:168` (and `:230`, `:288`, `:887`) write `:75`'s
  expression in host C++. `CMakeLists.txt` sets no `-march` and gives `-mfma` only to the CPU kernel files
  (`:674-707`), so on an x86-64 host the mirror rounds the product and the sum separately, while the GPU rounds once
  (section 2.1). The two can differ in the last bit of a penalised logit, which changes a pick only where that
  logit meets another within an ulp; that no fixture does is [H]. A Metal test's mirror that is meant to give the
  GPU's bits computes t = count · freq + present with one rounding to binary32.

### 4.3 Points for the contract and the kernel lane

- **No approximate division or reciprocal.** Every division and reciprocal in `sampler.cu` is IEEE (`div.rn.f32`,
  `rcp.rn.f32`, `div.rn.f64`); the `.FTZ` instructions in the listings are CUDA's own (inside `rcp.rn`'s fast path,
  where they flush nothing, and inside the slow-path subroutines). The amended flush-to-zero rule for approximate
  reciprocals therefore has no site here. The one approximate operation is `:311`'s `__expf` (section 3.3).
- **Two library functions are not in invariant 5's list.** The contract maps the library `expf` to
  `metal::precise::exp` and gives `log1pf` a helper; `sampler.cu` calls the library `logf` (`:403`) and the
  binary64 `exp` (`:379`, `:410`). This record transcribes CUDA's sequences for both (sections 2.5 and 2.6), which
  gives CUDA's bits for every argument; the contract should name them.
- **binary64 in software.** MSL has no `double` on the GPU. The tail needs `dadd`, `dmul`, `dfma`, `ddiv`, the two
  conversions and four comparisons, IEEE with subnormals, on integer words. They run on one SIMD-group per row over
  at most 64 entries, mostly on lane 0.
- **Order.** The selection rounds, the greedy scans and every list merge are order-free under the strict total
  order, so their geometry is free and CUDA clusters become a partials dispatch and a merge dispatch (section 3.2).
  The tail (serial sums and scans on lane 0) and the greedy pick's probability (a 1,024-thread threadgroup at
  stride 1,024, the butterfly, the serial finish) have fixed orders that the Metal kernels keep.
- **Per-step values.** `sample_tokens` passes `SamplerParams`, `seed` and `counter` included, as a kernel parameter
  (`:997-1019`). Under invariant 4 the Metal kernels read the counter (and the request's parameters) from a buffer
  or StepState, as the coupled kernels already do from `*dp` and the step record (`:787-788`); the draw for row t
  stays `counter + t`, so a row's token does not depend on the window width (invariant 1).
- **Open choices.** `fmax32`'s result for -0 against +0 is not recorded, and no sampler output depends on it
  (section 2.4). Whether to emulate `ex2.approx`'s subnormal results at `:311` (section 3.3): the record
  recommends not to.
