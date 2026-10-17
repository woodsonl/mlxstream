# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/s2_expert_grouped_parity.cpp and src/kernels/quantize_act_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their fixtures, plans, sentinels and pass
# criteria.
"""basalt's drafter experts on Metal (kernels/common/basalt_s2_experts.metal): moe_group_resident,
quantize_q8_0_scaled and moe_grouped_s2 (gate/up, SwiGLU, the intermediate's quantizer, down), with the IEEE helpers
they share (kernels/common/basalt_q8_1.metal) and the fp16 conversion (basalt_fp.metal's f16_bits).

Fixtures follow basalt's tests; basalt draws them with std::mt19937 and std:: distributions, whose outputs are
implementation-defined, and these draw the same distributions with numpy. basalt's fixtures never make a subnormal (its
scales keep every product above 1e-6), so planted inputs exercise the IEEE paths, each with a negative control: the
same kernels with the plain Metal operation in place of each IEEE helper must differ on them."""

import numpy as np
import pytest

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_gpu as gpu

F32 = np.float32
H, FF, K = kernels.S2_H, kernels.S2_FF, 10
NC_GU, NC_D = H // 32, FF // 32
XROW = NC_GU * kernels.Q8_0_BLOCK_BYTES                 # one token's activation row of block_q8_0
O_D_CODES = 2 * FF * (H // 4)
O_GU_SCALES = O_D_CODES + H * (FF // 4)
O_D_SCALES = O_GU_SCALES + 2 * FF * (H // 64) * 2


def gpu_eval(body, *arrays, out=1):
    """A one-off kernel over n threads on basalt_q8_1_source's helpers: ``body`` computes ``r`` from ``a``, ``b``, ``c``
    (float inputs at index i); returns the float32 r."""
    names = "abc"[: len(arrays)]
    args = "".join(f"device const float* {v}_ [[buffer({k})]], " for k, v in enumerate(names))
    loads = "".join(f"const float {v} = {v}_[i]; " for v in names)
    src = kernels.basalt_q8_1_source() + f"""
kernel void ev({args}device float* r_ [[buffer({len(names)})]], constant uint& n [[buffer({len(names) + 1})]],
               uint i [[thread_position_in_grid]]) {{
    if (i >= n) return;
    {loads}float r; {body}
    r_[i] = r;
}}
"""
    arrays = [np.ascontiguousarray(a, F32).reshape(-1) for a in arrays]
    n = arrays[0].size
    res = gpu.run(gpu.pipeline(src, "ev"), [a.tobytes() for a in arrays] + [4 * n], np.uint32(n).tobytes(),
                  len(arrays) + 1, -(-n // 256), 256)
    return np.frombuffer(res[len(arrays)], F32)


def same_bits(got, want):
    return got.view(np.uint32) == np.asarray(want, F32).view(np.uint32)


# ---- the fp16 conversion: quantize_act_parity's fp16 half (quantize_act_parity.cpp:280-336) -------------------------

def test_f16_bits_is_f16_from_f32():
    """basalt_fp.metal's f16_bits (Metal's half(), which q8_1_put and the quantizer share) against basalt's vectors,
    quoted: ``if (got != c.want) {`` (quantize_act_parity.cpp:322); h2f(0x0001) == 2^-24, quoted: ``if (!sub_ok) ++bad;``
    (:335); and against the transcription of f16_from_f32 (tests/kernels/basalt_emu.py) on the fp16 subnormal range, the
    65504-65520 boundary, both overflow edges and random float32 patterns (a NaN's payload is the GPU's)."""
    cases = [(1e30, 0x7C00), (-1e30, 0xFC00), (65536.0, 0x7C00), (3.0e38, 0x7C00), (65504.0, 0x7BFF),
             (65519.996, 0x7BFF), (65520.0, 0x7C00), (65536.0, 0x7C00), (5.9604645e-8, 0x0001), (1.0, 0x3C00),
             (0.0, 0x0000)]
    rng = np.random.default_rng(3)
    sweep = np.concatenate([
        np.array([c for c, _ in cases], F32),
        np.arange(0x33000000, 0x38800000, 97, dtype=np.uint32).view(F32),            # 2^-25 .. 2^-14: fp16 subnormals
        np.arange(0x477FE000, 0x47800100, dtype=np.uint32).view(F32),                # 65504 .. 65536
        -np.arange(0x477FE000, 0x47800100, 7, dtype=np.uint32).view(F32),
        np.uint32([1, 0x007FFFFF, 0x00800000, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF]).view(F32),
        rng.integers(0, 2 ** 32, 1 << 20, dtype=np.uint64).astype(np.uint32).view(F32)])
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + """
kernel void f16(device const float* x [[buffer(0)]], device ushort* h [[buffer(1)]], device float* back [[buffer(2)]],
                constant uint& n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    h[i] = f16_bits(x[i]);
    back[i] = h2f(ushort(i));
}
"""
    n = sweep.size
    out = gpu.run(gpu.pipeline(src, "f16"), [sweep.tobytes(), 2 * n, 4 * n], np.uint32(n).tobytes(), 3,
                  -(-n // 256), 256)
    got = np.frombuffer(out[1], np.uint16)
    fp16_bad = sum(int(got[k]) != want for k, (_, want) in enumerate(cases))
    assert fp16_bad == 0, [(c, hex(w), hex(int(got[k]))) for k, (c, w) in enumerate(cases)]   # basalt: if (got != c.want)
    back = np.frombuffer(out[2], F32)[:65536]
    assert back[1] == F32(5.9604645e-8)                    # basalt: sub_ok = (sub == 5.9604645e-8f)
    with np.errstate(invalid="ignore"):
        exact = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(F32)
    assert np.array_equal(back.view(np.uint32)[~np.isnan(exact)], exact.view(np.uint32)[~np.isnan(exact)])
    want = emu.f16_from_f32(sweep)
    nan = np.isnan(sweep)
    assert np.array_equal(got[~nan], want[~nan]), np.flatnonzero(got[~nan] != want[~nan])[:8]
    assert np.all((got[nan] & 0x7C00) == 0x7C00) and np.all(got[nan] & 0x3FF)
    print(f"\nf16_bits: {len(cases)} basalt vectors, {np.count_nonzero(~nan)} values bit for bit with f16_from_f32")


# ---- quantize_q8_0_scaled: the contract's host transcription (flash-next.md) ------------------------------------------

def quantize_distributions():
    """quantize_act_parity's five distributions of 131,072 values (quantize_act_parity.cpp:245-275)."""
    rng = np.random.default_rng(7)
    n = 32 * 4096
    d = ((np.arange(n) % 200) - 100 + 0.5).astype(F32)
    d[0] = 127.0
    e = np.zeros(n, F32)
    e[::32] = 1000.0 + (np.arange(n // 32) % 7)
    return {"random normal": rng.standard_normal(n).astype(F32),
            "small magnitude (1e-6)": (rng.standard_normal(n).astype(F32) * F32(1e-6)).astype(F32),
            "all zero": np.zeros(n, F32), "exact .5 boundaries": d, "one extreme per block": e}


def planted_blocks():
    """Blocks where the GPU's plain operations and IEEE's differ: amax subnormal (s on the subnormal grid, inv from the
    scaled reciprocal), amax at 2^-121 .. 2^-119 (s just normal, inv large), amax below about 2^-127 (inv infinite)
    with zero lanes (t NaN, code 0), subnormal lanes in normal blocks, codes on .5 ties of tiny blocks, and blocks with
    NaN, infinite and signed-zero lanes."""
    rng = np.random.default_rng(11)
    blocks = []
    for lo, hi in ((-149, -127), (-130, -121), (-126, -119), (-140, -100)):
        for _ in range(64):
            b = (rng.choice([-1, 1], 32) * np.exp2(rng.uniform(lo, hi, 32))).astype(F32)
            b[rng.integers(0, 32, 3)] = 0.0
            blocks.append(b)
    for k in range(1, 40):                                                     # amax = k units of 2^-149
        b = np.zeros(32, F32)
        b[: min(k, 32)] = np.uint32(np.arange(1, min(k, 32) + 1)).view(F32)
        b[-1] = np.uint32(k).view(F32)
        blocks.append(b)
    for _ in range(64):                                                        # subnormal lanes in normal blocks
        b = rng.standard_normal(32).astype(F32) * F32(2.0 ** -100)
        b[rng.integers(0, 32, 8)] = rng.integers(1, 1 << 23, 8).astype(np.uint32).view(F32)
        blocks.append(b)
    for _ in range(64):                                                        # ordinary blocks with planted lanes
        b = rng.standard_normal(32).astype(F32)
        b[:5] = [np.nan, np.inf, -0.0, 0.0, -np.inf] if rng.random() < 0.5 else [np.nan, 1.0, -0.0, 0.0, -2.0]
        blocks.append(b)
    blocks.append(np.full(32, np.nan, F32))
    return np.concatenate(blocks).astype(F32)


@pytest.mark.parametrize("dist", ["random normal", "small magnitude (1e-6)", "all zero", "exact .5 boundaries",
                                  "one extreme per block", "planted"])
def test_quantize_q8_0_scaled(dist):
    """basalt_quantize_q8_0_scaled against the host transcription of quantize_q8_0_scaled_warp_kernel
    (tests/kernels/basalt_emu.py's q8_0_scaled): the contract's test for this quantizer (docs/design/flash-next.md, the
    table of kernels without a basalt reference), codes and fp32 scales byte for byte, as quantize_act_parity checks
    quantize_q8_0, quoted: ``if (g_blocks[i] != r_blocks[i]) ++byte_bad;`` (quantize_act_parity.cpp:100). On
    quantize_act_parity's five distributions, and on planted blocks (planted_blocks) that only an IEEE kernel gets
    right."""
    x = planted_blocks() if dist == "planted" else quantize_distributions()[dist]
    blocks, scales = gpu.q8_0_scaled(x)
    want_b, want_s = emu.q8_0_scaled(x)
    byte_bad = np.count_nonzero(blocks != want_b)
    scale_bad = np.count_nonzero(scales.view(np.uint32) != want_s.view(np.uint32))
    assert byte_bad == 0 and scale_bad == 0, (dist, byte_bad, scale_bad)
    if dist == "planted":
        with np.errstate(divide="ignore", over="ignore"):
            sub = np.count_nonzero((want_s > 0) & (want_s < F32(2.0 ** -126)))
            inf_inv = np.count_nonzero((want_s > 0) & np.isinf(F32(1.0) / want_s))
        assert sub > 100 and inf_inv > 10, (sub, inf_inv)
    print(f"\nquantize_q8_0_scaled, {dist}: {x.size // 32} blocks byte for byte")
