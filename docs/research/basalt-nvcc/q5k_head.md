# basalt's Q5_K head as nvcc compiles it

What NVIDIA's compiler emits at every rounding site of basalt's Q5_K head (`src/kernels/cuda/q5k_head.cu`) and of the
2-bit proxy it includes (`src/kernels/cuda/head_proxy.cuh`), read from the sm_120 PTX and SASS, and the Metal form the
port writes at each site (`kernels/common/basalt_q5k_head.metal`). `q5k_head.cu` builds with `--use_fast_math`
(`CMakeLists.txt:200`). The format follows `matvec.md`, whose method, notation and helpers this page uses.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b there,
and a bare `:N` is a line of the file named last.

## Method

- **Sources.** `q5k_head.cu` verbatim with the quoted includes inlined (`basalt/kernels/q5k_head.hpp`, `dp4a.hpp`,
  `pdl.hpp`, `head_proxy.cuh`). Only basalt's public MIT source was sent.
- **Compilers and flags.** Compiler Explorer's public API as in `matvec.md`: NVCC 13.0.0 (`nvcc130`) and 13.3.0
  (`nvcc133`), each with basalt's flags for this file: `-O3 -DNDEBUG -DBASALT_VERSION="0.0.1"
  --generate-code=arch=compute_120,code=[compute_120,sm_120] --use_fast_math`, plus `-Xptxas -v` for the register
  counts.
- **Translation units.** `q5k_head.cu:1-252` (includes, block structs, `q5_load`, `q5_apply`, `q5k_row_dev`, `Top`,
  `q5k_rows_10`, `q5k_head_kernel`), for the top-1 unit also `:253-335` (`q5k_head_top1_kernel`) and `:399-404`
  (`mask_unmapped_kernel`), for the proxy units `:436-498` (`q5k_proxy_pack_kernel`, `q5k_proxy_final_kernel`); then a
  table that takes the address of each kernel instantiation the unit emits (`q5k_head_kernel<1, 10>`, `<2, 10>`,
  `<8, 10>`, `<1, 0>`, `<8, 0>`; `q5k_head_top1_kernel`, `mask_unmapped_kernel`; `q5k_proxy_pack_kernel`;
  `q5k_proxy_final_kernel`; `px::score_kernel`, which every unit emits). The host launchers are left out: a kernel's
  code depends only on its template and the inlined device functions. Nine units per compiler, all compiled (code 0);
  only `q5k_head_kernel<8, 0>`'s SASS listing reached Compiler Explorer's 5,000-line cut (its PTX is complete, and
  `<8, 10>` has the same per-(row, column) templates).
- **Reading a site.** As in `matvec.md`: the PTX shows what NVVM fused and each conversion; the SASS what `ptxas` made
  of it. The dataflow tracer of `matvec.md` prints each floating-point SASS instruction as an expression template;
  the templates below are 13.3's, and 13.0's are the same multiset at every site this page records.

## Register and shared-memory use (sm_120, `ptxas -v`, both compilers)

| Kernel | Registers | Shared memory | Barriers |
|---|---|---|---|
| `q5k_head_kernel<1, 10>`, `<2, 10>` | 128 | dynamic (columns) | 1 |
| `q5k_head_kernel<8, 10>` | 156 | dynamic | 1 |
| `q5k_head_kernel<1, 0>` | 63 | dynamic | 1 |
| `q5k_head_kernel<8, 0>` | 128 | dynamic | 1 |
| `q5k_head_top1_kernel` | 126 | 7,777 B | 1 |
| `mask_unmapped_kernel` | 10 | 0 | 0 |
| `q5k_proxy_pack_kernel` | 56 | 0 | 0 |
| `q5k_proxy_final_kernel` | 40 | 8,516 B (`xe`, 7,680 B, and `px::Final`, 836 B) | 1 |
| `px::score_kernel` | 77 | 3,712 B | 1 |

The top-1's grid (`q5k_head.cu:375-386`) is the multiprocessor count times `cudaOccupancyMaxActiveBlocksPerMultiprocessor`
for 256 threads, capped at 4,096. On sm_120 (65,536 registers per multiprocessor, allocation in 8-register units per
thread) 126 registers round to 128: 128 × 256 = 32,768 registers a block, two blocks a multiprocessor; the 7,777 bytes of
shared memory and the 1,536-thread limit allow more. Basalt's measured RTX 5090 has 170 multiprocessors
(`docs/research/basalt-nvcc/dense_tc.md` and the scoping report's decision 4), so its grid is 340 blocks for every row
count above 2,720 and ceil(n_out / 8) below.

## Summary

| Kernel | Rounding sequence nvcc 13.x emits (sm_120) | Metal |
|---|---|---|
| `q5k_head_mmvq` (`q5_apply`, `q5k_head.cu:92-106`; `q5k_row_dev` / `q5k_rows_10`, `:110-223`) | per (row, column) `native_mmvq`'s Q5_K sequence (`matvec.md` section 5): sumf_d = fma(d8_i, f(dot1_i·sc_i), sumf_d), sumf_m = fma(d8_i, f(dot2_i·m_i), sumf_m) from 0; v = fma(dm.x, sumf_d, −fl(dm.y·sumf_m)); p_v = 0 + v, then + v per further block; t = ((p_0 + p_1) + p_2) + p_3; butterfly 16..1; all `.FTZ` | the same, `fma_ftz`, `mul_ftz`, `add_ftz`, `warp_sum_ftz`: `basalt_mmvq.metal`'s Q5_K dot, shared |
| `Top::take` (`:155-164`) | l > m: s = fma(s, e, 1) with e = `MUFU.EX2`(fl((m − l)·fl(log2 e))); else s = s + `MUFU.EX2`(fl((l − m)·fl(log2 e))); all `.FTZ` | `fma_ftz(s, expf_ftz(m - l), 1)`; `add_ftz(s, expf_ftz(l - m))`; `expf_ftz(x)` = `ftz(fast::exp2(mul_ftz(x, fl(log2 e))))` |
| `Top::merge` (`:165-175`) | o.m > m (or empty): s = fma(s, e(m − o.m), o.s); else s = fma(o.s, e(o.m − m), s) | `fma_ftz` with the same `expf_ftz` |
| top-1 output (`:330-332`) | `1.0f / b.s` as `MUFU.RCP` alone | `rcp_ftz(s)` |
| `mask_unmapped_kernel` (`:400-403`) | a store of −inf, no arithmetic | the same |

The `[H]` of the scoping report (section 7.2: whether the head resolves `dm.x * sumf_d - dm.y * sumf_m` as
`native_mmvq` does) is settled: the head's SASS rounds the dmin product and fuses the d product with its negation,
`FMUL.FTZ` then `FFMA.FTZ ..., -R`, exactly `native_mmvq.cu:200`'s form (`matvec.md` section 5, Q4_K and Q5_K), in both
compilers.

## 1. `q5k_head_mmvq`

`q5k_head_kernel<1, 10>` (13.3), one row's five blocks for lane L (`kb = {a, a + 8, a + 2, a + 4, a + 6}`, `vt = {0, 0,
1, 2, 3}`, `q5k_head.cu:183-185`):

| Site | SASS (13.3) | Template | Metal |
|---|---|---|---|
| `sumf_d += d8 * (dot1 * r.sc[i])` (`:102`), i = 0, 1 | `FFMA.FTZ R30, R31, R30, RZ`; `FFMA.FTZ R118, R35, R34, R30` | `FFMA.FTZ(i, in, FFMA.FTZ(i, in, 0))` | `fma_ftz(d8, float(dot1 * sc), sumf_d)` from 0 |
| `sumf_m += d8 * (dot2 * r.m[i])` (`:103`) | `FFMA.FTZ R28, R31, R28, RZ`; `FFMA.FTZ R116, R35, R33, R28` | the same | `fma_ftz(d8, float(dot2 * m), sumf_m)` |
| `r.dm.x * sumf_d - r.dm.y * sumf_m` (`:105`) | `FMUL.FTZ R45, R118, R113`; `FFMA.FTZ R116, R116, R51, -R45` | `FFMA.FTZ(sumf, in, −FMUL.FTZ(sumf, in))` | `fma_ftz(dmx, sumf_d, -mul_ftz(dmy, sumf_m))` |
| `p[vt[c]][j] += q5_apply(...)` (`:206`) | `FADD.FTZ R3, RZ, R116`; `FADD.FTZ R3, R3, R118` (virtual thread 0's two blocks); `FADD.FTZ R6, RZ, R6` (threads 1..3, one block each) | `FADD.FTZ(0, v)`, `FADD.FTZ(p, v)` | `add_ftz(p, v)` from 0 |
| `t = p[0][j]; t += p[v][j]` (`:209-211`) | `FADD.FTZ R3, R3, R6`; `FADD.FTZ R3, R2, R3`; `FADD.FTZ R3, R4, R3` | `((p0 + p1) + p2) + p3` | `add_ftz` ×3 in virtual-thread order |
| `warp_sum(t)` (`:212`) | `SHFL.BFLY` + `FADD.FTZ` with offsets 16, 8, 4, 2, 1 | `FADD.FTZ(in, shfl)` | `warp_sum_ftz` |

That is `native_mmvq`'s Q5_K kernel per value: thread L + 32v's `tmp` starts at +0 and adds its blocks in ascending
kbx (`FADD.FTZ R20, RZ, R0` there too), warp 0 adds warps 1..3's partials in warp order, then the same butterfly
(`matvec.md` section 4). Which of the two `dm` halves is rounded alone is read from the conversions: as in
`native_mmvq`, the `FMUL.FTZ` takes the high half (dmin) and the `FFMA.FTZ` the low half (d).

The column staging (`:229-240`) precomputes per (q8_1 block, position) `{u0, u1, dot2, d8}`: `dot2` is the same
integer `DP4A(0x01010101, u0, DP4A(0x01010101, u1, 0))` as `q5_q8_dot_impl`'s, and `d8` the exact conversion of the
block's fp16 scale (`__low2float`). Staging is not arithmetic.

## 2. `q5k_head_top1`

| Site | SASS (13.3) | Template | Metal |
|---|---|---|---|
| `__expf(m - l)` (`:157`) | `FADD.FTZ R0, R56, -R7`; `FMUL.FTZ R0, R0, 1.4426950216293334961`; `MUFU.EX2 R3, R0` | `MUFU.EX2(FMUL.FTZ(fl(log2 e), FADD.FTZ(−l, m)))` | `expf_ftz(m - l)`: `ftz(fast::exp2(mul_ftz(add_ftz(m, -l), 0x3FB8AA3B)))` |
| `s = s * __expf(m - l) + 1.0f` (`:157`) | `@P1 FFMA.FTZ R8, R8, R3, 1` | `FFMA.FTZ(e, s, 1)` | `fma_ftz(s, e, 1.0f)` |
| `s += __expf(l - m)` (`:161`) | `@!P1 FADD.FTZ R2, -R56, R7`; `FMUL.FTZ`; `MUFU.EX2 R5`; `@!P1 FADD.FTZ R8, R8, R5` | `FADD.FTZ(s, MUFU.EX2(...))` | `add_ftz(s, expf_ftz(l - m))` |
| `s = o.s + s * __expf(m - o.m)` (`:168`) | `FADD.FTZ`; `FMUL.FTZ ..., 1.4426950216293334961`; `MUFU.EX2`; `FFMA.FTZ R5, R2, R5, R6` | `FFMA.FTZ(e, s, o.s)` | `fma_ftz(s, expf_ftz(m - o.m), o.s)` |
| `s += o.s * __expf(o.m - m)` (`:172`) | `FFMA.FTZ R5, R6, R0, R5` | `FFMA.FTZ(e, o.s, s)` | `fma_ftz(o.s, expf_ftz(o.m - m), s)` |
| `*out_prob = 1.0f / b.s` (`:332`) | `MUFU.RCP R13, R6`, stored as is | `MUFU.RCP(s)` | `rcp_ftz(s)` |

The comparisons (`l > m`, `l == m && row < idx`, `o.m > m`, `o.idx < idx`) are exact. The per-block merge of the 8
warps (`:297-299`, thread 0, warp order), the last block's thread-strided merge over the block partials (`:311-317`),
the butterfly over the struct (`:318-324`, all three fields shuffled with the same offset, then `merge`) and the final
warp-order merge (`:327-329`) are the same `merge` sequence at each step. `__expf` is `ex2.approx.ftz.f32` of
`mul.ftz.f32 x, 0f3FB8AA3B` in PTX: `MUFU.EX2` has no documented Metal counterpart, and the contract
(`docs/design/flash-next.md`, invariant 5) maps it to `fast::exp2` of the same product; the tests take its values from
the GPU's own function.

## 3. The cross-threadgroup merge

To be completed in this page: the Metal Shading Language's memory-model text on device-scope atomics and fences, and
why the port runs the top-1's merge as a second dispatch.

## 4. The 2-bit proxy

To be completed in this page: `q5k_proxy_pack_kernel` (`fit_group`, `level`, `pack_group`), `px::score_kernel` and
`q5k_proxy_final_kernel` (`collect`, `pick`).
