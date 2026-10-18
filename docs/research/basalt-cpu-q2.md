# basalt's CPU Q2_0 expert kernels as x86 compilers compile them

What GCC and Clang emit at every rounding site of basalt's two CPU expert kernels, `act_quant_q8_1` and
`q2_0_gguf_rows_multi`, and the ARM NEON form the port writes at each site. basalt computes an expert on the CPU
when the GPU cache misses it, so these two kernels carry the CPU tier's whole arithmetic. This page is the
authority `runtime/src/basalt_cpu.cpp` follows, as `docs/research/basalt-nvcc/matvec.md` is for the GPU kernels.

Basalt paths are relative to commit `b4c63b55754637f4c72ad53a0f791d425f71eb36`; `path:a-b` means lines a to b
there, and a bare `:N` is a line of the file named last. Tags: **[M]** measured here, **[S]** a vendor
specification or a compiler's own header, **Inference:** a conclusion drawn from reading code.

## The sources

`src/kernels/cpu/expert.cpp:143-148` (`unpack64_q2_0`), `:336-370` (`act_quant_q8_1`, the AVX-512 block; the
function returns at `:369`, so the scalar loop at `:371-397` is unreachable) and `:503-556` (`q2g_row_multi`,
`q2g_rows`, `q2_0_gguf_rows_multi`), with `include/basalt/kernels/cpu/expert.hpp:25-61` for `ActQ` and the
geometry. `src/kernels/cpu/expert_layout.cpp:124-133` routes this pack to them wherever AVX-512 is present.

## 1. Which compiler, and the one value that depends on the answer

basalt pins no compiler: `README.md:35-43` asks for "CMake 3.24 or newer and a C++20 compiler" and builds with a
bare `cmake -S . -B build`; `CMakeLists.txt` sets no `CMAKE_CXX_COMPILER` and only `-Wall -Wextra` (`:45-46`).
`_mm512_reduce_add_ps` and `_mm512_reduce_max_ps` are header-defined, so their lane trees come from whichever
compiler builds basalt. Both candidates were read, and both were cross-compiled and their x86 output inspected [M]:

- **GCC** `gcc/config/i386/avx512fintrin.h` (`releases/gcc-14`): the float `__MM512_REDUCE_OP` at `:16297-16306`,
  used by `_mm512_reduce_add_ps` at `:16308-16312`, and the max form at `:16339-16350`, used by
  `_mm512_reduce_max_ps` at `:16359-16362` [S].
- **Clang** `avx512fintrin.h:9448-9450` and `:9579-9581`: `__builtin_ia32_reduce_fadd_ps512(-0.0f, __W)` and
  `__builtin_ia32_reduce_fmax_ps512(__V)` [S], which lower to the same shape.

Both expand to the same four-level pairwise tree, so the **grouping is compiler-independent**:

```
R_add(v[0..15]):  u[j] = v[j] + v[j+8]  (j=0..7);  t[j] = u[j] + u[j+4]  (j=0..3)
                  a = t[0] + t[2];  b = t[1] + t[3];  res = a + b
R_max(v[0..15]):  u[j] = maxps(v[j+8], v[j]);       t[j] = maxps(u[j+4], u[j])
                  p = maxps(t[0], t[2]);  q = maxps(t[1], t[3]);  res = maxps(p, q)
                  where maxps(a, b) = (a > b) ? a : b      [S] Intel SDM, MAXPS Operation
```

The sum is identical under both because the **grouping** is, not because addition commutes: a reduction's bits
follow its association. Clang's `reassoc llvm.vector.reduce.fadd` licenses any association [M], so that agreement
is a measurement to re-check rather than a guarantee. The **only** compiler-dependent value in either kernel is
`amax` when a chunk contains a NaN: GCC's first two levels take `src1 = hi` as written above, Clang's
take `src1 = lo`, and MAXPS returns SRC2 on an unordered compare. The port writes GCC's order, since basalt is
Linux-only (`AGENTS.md:3`, `README.md:33`), so its default `c++` is g++ (**Inference**); `runtime/src/basalt_cpu.cpp`
records both orders, and `tests/kernels/test_basalt_cpu_q2.py` case A6 asserts one while printing the other, so the
choice flips in one line if basalt's build is ever shown to be Clang. Clang's `fmax` reduction carries `nnan`, so
under Clang that value is undefined for a NaN chunk rather than merely differently ordered, and the low-half tree
is the measured realisation of it [M].

## 2. The one output whose bits may differ

`res[t]` can differ where the kernel **generates** a NaN from operands that are not NaN. An infinite activation
element gives its chunk `scale = +inf`, and then `d * scale` with `d = ±0`, the accumulator's fused multiply-add on
a lane whose four codes are all zero, the same on the correction, or `inf - inf` in the final subtraction each
produce one. AArch64 writes its default NaN, `0x7fc00000`; x86 writes the floating-point indefinite, `0xffc00000`,
with the sign bit set ([S] Intel SDM Vol. 1 4.8.3.7) [M, both measured]. `act_quant_q8_1` cannot emit one, since
`amax > 0` is an ordered compare and `s = inf` requires an infinite lane, whose `sum <= -127` makes
`hx = s * sum = -inf` rather than `inf * 0`. The port's comparison is therefore bitwise everywhere and NaN-ness on
`res`, and a recorded-output replay against an AVX-512 machine must compare it the same way.

## 3. The floating-point environment

`grep -rn "setcsr|FLUSH_ZERO|DENORMALS_ZERO|ffast-math|Ofast|ffp-contract|crtfastmath|_MM_SET_"` over basalt's
`CMakeLists.txt`, `src` and `include` returns nothing [M], so basalt runs on the default MXCSR: round to nearest
even, every exception masked, flush-to-zero off, denormals-are-zero off (**Inference** from that absence plus the
x86-64 ABI default). macOS on arm64 leaves FPCR at 0 (round to nearest, flush-to-zero off), so subnormals survive on
both sides and the subnormal-scale case of section 4 is reachable here too.

## 4. The quantizer's reachable edge cases

`s = amax > 0 ? amax / 127 : 0` and `inv = s > 0 ? 1 / s : 0`, then `t = x * inv`, `r = t + copysign(0.5, t)` and
`VCVTTPS2DQ` (`expert.cpp:352-362`). The conversion returns the **integer indefinite** `0x80000000` for a NaN, for
either infinity and for anything outside int32 ([S] Intel SDM Vol. 1, Table 11-1 and D.4.2.2), which the clamp to
[-127, 127] then takes to **−127**, never +127. Three consequences, all reachable from finite input:

- A chunk whose `s` falls at or below `2^-128` has `inv = +inf`, so its zeros give `0 * inf = NaN` and its nonzeros
  `±inf`: **every code is −127**, zeros included.
- An infinite element gives `s = inf`, `inv = 0` and `t = inf * 0 = NaN`: that element is −127 while the finite
  ones, including `−0`, take the `>= 0` branch and give 0.
- `amax` at or below `127 * 2^-149` rounds `s` to zero, so `inv = 0` and every code is 0.

The +127 side of the clamp is inert: wherever `inv` is finite, `|x| <= amax` holds `r` at 127.50003 or below [M],
measured at the exhaustive worst case in case A8. ARM's `FCVTZS` **saturates** instead (NaN gives 0, `+inf` gives
`INT32_MAX`), which after the clamp would give code 0 and +127 where x86 gives −127 for both, so the port selects on
`vcagtq_f32` against `2^31` and substitutes `INT32_MIN` itself.

## 5. Where NEON is not x86

| x86 | NEON | Why they differ |
|---|---|---|
| `_mm512_and_ps(x, 0x7fffffff)` | `vabsq_f32` | identical |
| `_mm512_max_ps`, every node of the trees | `vbslq_f32(vcgtq_f32(a, b), a, b)` | `vmaxq_f32` returns the NaN and `vmaxnmq_f32` the number; MAXPS returns SRC2 on any unordered compare and on `(±0, ∓0)`. The select is MAXPS exactly, and it keeps the operand order of section 1 visible |
| `_mm512_reduce_add_ps`, `_mm512_reduce_max_ps` | the explicit four-level trees | `vaddvq_f32` gives `(t0+t1)+(t2+t3)` where x86 gives `(t0+t2)+(t1+t3)`, and `FMAXV` has the wrong NaN rule |
| `_mm512_mul_ps` then `_mm512_add_ps` | `vmulq_f32` then `vaddq_f32`, built `-ffp-contract=off` | two roundings; a contracted `fma` skips the rounding of `t` and changes codes. Apple clang keeps them apart at its default while they are separate statements and fuses them under `fast` [M], so the flag pins the guarantee |
| `_mm512_fmadd_ps` (`acc`, `corr`) | `vfmaq_f32` | one rounding; fusion is required here, which is why contraction is disabled on the file and not globally |
| `_mm512_cvttps_epi32` | `vbslq_s32(vcagtq_f32(2^31, r), vcvtq_s32_f32(r), INT32_MIN)` | section 4 |
| `_mm512_dpbusd_epi32` | `vdotq_s32` | SDOT is signed by signed, VPDPBUSD unsigned by signed, equal here because codes are 0..3. NEON vector `j` lane `k` is the x86 dword lane `4j + k` |
| `_mm512_multishift_epi64_epi8` | `vqtbl1q_u8`, `vshlq_u8`, `vandq_u8` | the port decodes the documented rule `code m = (qs[m / 4] >> 2 * (m % 4)) & 3` (`ggml/src/ggml-quants.c:220-222`), which is what VPMULTISHIFTQB produces |
| `_mm256_cvtph_ps` | `vcvt_f32_f16` | both exact for every half including subnormals at the defaults of section 2 [M] |
| `_mm512_cvtepi32_epi8`, `_mm512_reduce_add_epi32` | `vmovn_s32` with `vmovn_s16`, `vaddvq_s32` | exact after the clamp; the integer sum is order-independent (`|sum| <= 4064`) |
