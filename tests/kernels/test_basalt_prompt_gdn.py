# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/elementwise_parity.cpp and src/prefill/gdn_rec_parity.cu
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): elementwise_parity's gate check (:57-116),
# its fixture properties (:82-99) and pass criterion (:116); gdn_rec_parity's widths and seeds (:1038-1041).
"""basalt's prompt-path GDN kernels on Metal (kernels/common/basalt_prompt_gdn.metal), each entry one pipeline compiled
once as a dynamic-T program at the bound B = 4352 (17 x 256 >= 4099 + 1) with the model's constants as its handler sets
them, the rows R = t_this_step read from StepState (invariant 4), on the references and fixtures of
tests/basalt_prompt_gdn_ref.py. Every output is B rows pre-filled 0xFF and every run has R < B, so rows R..B-1 must
keep their 0xFF words; non-finite outputs are counted and must be 0; shader validation catches a store outside its
binding.

The emulation's approximate steps take this GPU's values (spec T§4.4): ``exp2`` and ``rsqrt`` probe metal::fast::exp2
and fast::rsqrt, the kernels' stand-ins for MUFU.EX2 and MUFU.RSQ, on the arguments numpy computes, one thread per
value (basalt_gpu.expf_ftz's pattern). The rest of __expf and rsqrtf (the FMUL, the -126 halving and the IEEE square,
the 2^24 prescale) is the emulation's own (tests/kernels/basalt_emu.py), so a flushed square or a missing halving in a
kernel shows as a bit difference.

D-gates, gdn_gates_kernel (kernels.cu:207-216), on D-gates' fixture (gates_inputs: every third head's alpha + dt past
20 and up to about 120, ssm_a negative) at basalt's widths, seed 27 + i (W's) for the float64 test and 11 + i (B-rec's)
for the bitwise ones:
- ``test_prompt_gdn_gates_against_double[T]``: gate and beta against gates64, elementwise_parity's rel_l1 <= 1e-6 each
  (:116), the fixture's coverage asserted on the rows run.
- ``test_prompt_gdn_gates_row_invariance[T]``: the R rows in one dispatch against each row alone, every word.
- ``test_prompt_gdn_gates_emulation_bit_for_bit[T]`` (T <= 257): bit for bit against gates_emu with the exp2 probe.
"""

import numpy as np
import pytest

from monolith import kernels
from monolith.core.step_state import StepStateLayout
from monolith.runtime import _native as nt
from tests.basalt_gdn_ref import C, EPS, F32, HK, HV, n_bad
from tests.basalt_prompt_gdn_ref import SEED_B, SEED_W, TS, U32, diff_words, gates64, gates_emu, gates_inputs, rel_l1
from tests.kernels import basalt_gpu
from tests.kernels.basalt_prompt_gpu import prompt_pipeline, run_rows

B = 4352                                  # every pipeline's ROWS_MAX: 17 x 256 >= 4099 + 1 (spec T Q9)
MACROS = {"GDN_C": str(C), "GDN_HK": str(HK), "GDN_HV": str(HV), "GDN_EPS_BITS": f"{int(EPS.view(U32)):#x}u"}
FILL = 0xFFFFFFFF                         # a word no dispatch wrote
TE = [T for T in TS if T <= 257]          # the bitwise-emulation widths (the plan's numpy budget)


# ---- the probes and the wrappers ------------------------------------------------------------------------------------

PROBE = kernels.PRELUDE + """
kernel void pgdn_probe(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                       constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[2 * i] = fast::exp2(x[i]);
    out[2 * i + 1] = fast::rsqrt(x[i]);
}
"""


def probe(x, k):
    """fast::exp2 (k = 0) or fast::rsqrt (k = 1) of every float32 of x on this GPU, in x's shape."""
    x = np.ascontiguousarray(x, F32)
    out = basalt_gpu.run(basalt_gpu.pipeline(PROBE, "pgdn_probe"), [x.tobytes(), 8 * x.size],
                         np.uint32(x.size).tobytes(), 2, -(-x.size // 256), 256)[1]
    return np.frombuffer(out, F32)[k::2].reshape(x.shape)


def exp2(x):
    """MUFU.EX2's stand-in, the value the kernels' expf_ieee takes for an argument."""
    return probe(x, 0)


def rsqrt(x):
    """MUFU.RSQ's stand-in, the value the kernels' rsqrt_ieee takes for an argument."""
    return probe(x, 1)


def pipeline(name):
    """Entry ``name`` of basalt_prompt_gdn_source() as a dynamic-T program at bound B with the handler's macros."""
    return prompt_pipeline(kernels.basalt_prompt_gdn_source(), name, B, MACROS)


def rows(a, width):
    """The rows of a as B rows of ``width`` float32 words, the rows past len(a) 0xFF (so a read past R that reached a
    store would leave cuda_nan's 0x7fffffff there)."""
    out = np.full((B, width), FILL, U32)
    out[:len(a)] = np.asarray(a, F32).reshape(len(a), width).view(U32)
    return out.tobytes()


def untouched(a, r):
    """Rows r..B-1 of the B-row output a keep the 0xFF fill."""
    return bool((a[r:].view(U32) == FILL).all())


def gates(alpha, logit, dt, ssm_a, alone=False):
    """gdn_gates_kernel at bound B over R = len(alpha) rows: alpha and logit (R, HV) bound as B rows (``rows``), dt and
    ssm_a (HV,) at exactly their bytes, gate and beta as B x HV words of 0xFF, the grid encoded at B (ceil(B HV / 256)
    threadgroups of 256). One dispatch with t_this_step = R, or with ``alone`` R dispatches in one command buffer, row t
    by itself: t_this_step = 1 and the four row buffers bound at row t. Returns gate and beta, (B, HV) float32."""
    R = len(alpha)
    pso = pipeline("gdn_gates_kernel")
    bufs = [rows(alpha, HV), rows(logit, HV), np.asarray(dt, F32).tobytes(), np.asarray(ssm_a, F32).tobytes(),
            4 * B * HV, 4 * B * HV]
    grid = (-(-B * HV // 256), 1, 1)
    if not alone:
        out = run_rows(pso, bufs, None, 0, grid, (256, 1, 1), R, 0, fill=0xFF)[4:6]
    else:
        dev = basalt_gpu.device()
        bufs = [nt.Buffer(dev, b) for b in bufs]
        for b in bufs[4:]:
            b.fill(0xFF)
        st = nt.Buffer(dev, StepStateLayout().pack({"t_this_step": 1, "position": 0, "done": 0}))
        chain = []
        for t in range(R):
            d = nt.Dispatch().pipeline(pso)
            for k, b in enumerate(bufs):
                d = d.buffer(k, b, 0 if k in (2, 3) else 4 * HV * t)
            chain.append(d.buffer(15, st).grid(*grid).threadgroup(256, 1, 1))
        r = nt.Queue(dev).run(chain)
        assert not r.error, r.error
        out = [b.read(0, 4 * B * HV) for b in bufs[4:]]
    return [np.frombuffer(o, F32).reshape(B, HV) for o in out]


# ---- D-gates: gdn_gates_kernel (kernels.cu:207-216) ----------------------------------------------------------------

@pytest.mark.parametrize("T", TS)
def test_prompt_gdn_gates_against_double(T):
    """D-gates, nearest bound; no basalt test launches this kernel. elementwise_parity's check 1 (:57-116) runs the
    decode-era gdn_gate, the same softplus(alpha + dt) * ssm_a, against its double reference (:78-80); here
    gdn_gates_kernel's gate and beta against gates64 (that softplus, and beta's 1 / (1 + exp(-x))), rel_l1 <= 1e-6
    each, on D-gates' fixture at seed 27 + i for basalt's width i. Its coverage, which elementwise_parity asserts
    (:82-99), is asserted on the rows run: alpha + dt (the float sum) past 20, and past exp's overflow at 88, on every
    third head and at most 20 elsewhere, so both sides of the softplus branch run; ssm_a negative."""
    alpha, logit, dt, ssm_a = args = gates_inputs(T, SEED_W + TS.index(T))
    u = alpha + dt
    assert (u > 20).any() and (u > 88).any() and (u <= 20).any(), "`if (!over) ++bad;` (elementwise_parity.cpp:92)"
    assert (ssm_a < 0).all(), "`if (positive) ++bad;` (elementwise_parity.cpp:99)"
    gate, beta = gates(*args)
    assert untouched(gate, T) and untouched(beta, T), "a store at or past row R"
    assert n_bad(gate[:T]) == n_bad(beta[:T]) == 0, "a non-finite gate or beta"
    rg, rb = (rel_l1(want, got[:T]) for want, got in zip(gates64(*args), (gate, beta)))
    assert rg <= 1e-6 and rb <= 1e-6, (r'`if (!(rel <= 1e-6)) { std::printf("    *** over 1e-6 ***\n"); ++bad; }` '
                                       f"(elementwise_parity.cpp:116): gate rel {rg:.3e}, beta rel {rb:.3e}")


@pytest.mark.parametrize("T", TS)
def test_prompt_gdn_gates_row_invariance(T):
    """D-gates' bitwise check (spec K§9.2; nearest bound; no basalt test launches this kernel): a row's gate and beta do
    not depend on the chunk around it. The R rows of D-gates' fixture (seed 11 + i for basalt's width i) in one
    dispatch against each row alone, R dispatches of one row in one command buffer: every word of gate and beta, the
    rows past R included."""
    args = gates_inputs(T, SEED_B + TS.index(T))
    whole, alone = gates(*args), gates(*args, alone=True)
    assert untouched(whole[0], T) and untouched(whole[1], T), "a store at or past row R"
    assert n_bad(whole[0][:T]) == n_bad(whole[1][:T]) == 0, "a non-finite gate or beta"
    assert [diff_words(a, b) for a, b in zip(whole, alone)] == [0, 0], "a row alone differs from the chunk's"


@pytest.mark.parametrize("T", TE)
def test_prompt_gdn_gates_emulation_bit_for_bit(T):
    """D-gates' exact-port check (spec K§9.2; nearest bound; no basalt test launches this kernel): gate and beta bit for
    bit against gates_emu, the SASS's operations in numpy (FADD, u itself above 20, else log1pf_nv of __expf(u), FMUL
    by ssm_a; beta's FMUL by -log2(e), ex2, FADD 1 and the correctly rounded reciprocal) with this GPU's exp2, on
    D-gates' fixture at seed 11 + i for basalt's width i."""
    args = gates_inputs(T, SEED_B + TS.index(T))
    got = gates(*args)
    assert untouched(got[0], T) and untouched(got[1], T), "a store at or past row R"
    assert n_bad(got[0][:T]) == n_bad(got[1][:T]) == 0, "a non-finite gate or beta"
    want = gates_emu(*args, exp2)
    assert [diff_words(g[:T], w) for g, w in zip(got, want)] == [0, 0], "gate or beta is not the emulation's"
