# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/gr_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): its data distributions, the float64
# references' forms with their activation rounding, and the pass criteria.
"""basalt's prompt-path hyper-connection kernels on Metal (kernels/common/basalt_prompt_hc.metal), from
src/prefill/kernels.cu:85-204. No basalt test launches them (gr_parity runs decode's gr.cu kernels), so each gets
gr_parity's bounds as the nearest bound, a bitwise host twin and row invariance; every check runs at B = 256 through
basalt_prompt_gpu.prompt_hc, under shader validation in the verification runs.

Fixture (``fixture``): numpy default_rng, n = 2560, lr = 320, eps = 1e-6f; r, w, w_next, gated, lo, bo and e ~ N(0, 1),
inj ~ 3 N(0, 1) (gr_parity.cpp:675, ``x = gauss(rng) * 3.0f``), B = 256 tokens run with 1, 7 or 256 live rows; the
mix's rs is gr_norm_rs's of r and w (the norm twin's, the kernel's bits). Edge tokens (``fixture(edges=False)`` has
none), each kind seeing the classes on its inputs:
- token 0: E2, 1e-40 (subnormal) and 1e-36 at columns 5 and 6 of r and gated (every stream), lo, bo and e; E7, lo in
  [2^-126, 2^-123) at columns 8-11 (a normal input, a subnormal x / (1 + e)); E8, the NaN payload 0x7fa00001 at e's
  column 7 (a signalling NaN, planted and compared as bits only).
- token 1: E3 a NaN and E4 +inf in r (stream 0 and stream 1, column 3), lo (columns 3 and 4) and inj (streams 0, 1).
- token 2: E6, the tie row (stream 1): r[k] = 2^-70 s, r[k + 256] = (1 + 2^-12) s at k = 0, the rest 0, so thread
  k's first square is subnormal and decides the rounding of its sum (s = ``E6_S`` = 1, for which the IEEE and
  flushing twins' rs differ); inj 0 on stream 0 (E5: 2 sigmoid = 1, gr_parity.cpp:641's plain add).
- token 3: E1, an all-zero r row (stream 0: rs = fast::rsqrt(1e-6)), with E5's inj (-352, -400, 100, -100): -352 / 4
  = -88 has a subnormal sigmoid, -400 / 4 = -100 a zero one; gated -88 and -100 on all four streams of columns 0 and 1.

- HC-3 (``test_hc_twin``): at 1, 7 and 256 live rows every output word of the live rows equals the kind's twin
  (broadcast: e's 32-bit words in each stream; norm: basalt_emu.hc_norm, whose rs is the probe's fast::rsqrt of the
  argument fl(fl(S / n) + eps) the twin computes; silu and mix: basalt_emu.hc_silu and hc_mix, with the probe's
  expf_ieee(sign_flip(x)) of the twin's arguments), the rows past them keep the 0xFF fill, the inputs keep their
  bytes, and a run with done = 1 stores nothing; the mix at MIX_BF 0 (the drafter's KV pass, no BF16 image) stores the
  same mixed and mixed_h and leaves slot 5 at 0xFF. Discrimination, so a flushing kernel fails: broadcast's words
  differ from their flushed values on E2; norm's IEEE and flushing twins differ in token 0's xn16 (E2) and in token
  2's stream-1 rs (E6), mix's in token 0's columns 5-6 (E2) and token 3's column 0 (E5: gated -88 on every stream, a
  subnormal sigmoid), silu's in token 0's columns 8-11 (E7; its E2 words are BF16 zeros either way).
  ``test_hc_probe_specials`` (K1): the probe's fast::rsqrt and expf_ieee(sign_flip(x)) on the specials match
  PTX's documented rsqrt.approx and ex2.approx: rsqrt(+inf) = +0, rsqrt(NaN) NaN, exp_neg(+inf) = +0, exp_neg(-100)
  and exp_neg(-inf) +inf, exp_neg(NaN) NaN.
- HC-4 (``test_hc_rows_invariant``): bitwise, tokens 0, 1 and 255 of the 256-row run equal each token alone at one
  row; a static program's 7 rows (T_STATIC_ROWS) equal the dynamic 7; two runs give the same bytes; four forced 64-row
  slices equal the one-slice run at 256 and 100 live rows; broadcast's every stream equals e's words, E8's payload
  included."""

from functools import partial

import numpy as np
import pytest

from monolith import kernels
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_prompt_gpu as gpu

F32 = np.float32
B, N, LR = 256, 2560, 320
EPS = F32(1e-6)              # the handler's HC_EPS, 0x358637BD (prompt_hc's)
E6_S = 1.0                   # E6's s: twins' rs 0x424a13f8 (IEEE) vs 0x424a13f9 (flush) on the M3 Max; 2^-4 ties them
E8_NAN = 0x7FA00001
WORD = {4: np.uint32, 2: np.uint16}
RSQRT, EXP_NEG = (partial(gpu.hc_probe, fn) for fn in ("rsqrt", "exp_neg"))   # the GPU's approximate operations

# kind: (entry, inputs, outputs as (words per token, bytes per word)), in the kernel's slot order
KINDS = {"broadcast": ("prompt_gr_broadcast", ("e",), ((4 * N, 4),)),
         "norm": ("prompt_gr_norm_rs", ("r", "w"), ((4, 4), (4 * N, 2))),          # rs, xn16
         "silu": ("prompt_gr_silu", ("lo",), ((LR, 2),)),                           # lo16
         "mix": ("prompt_gr_mix_r", ("r", "rs", "w", "gated"), ((N, 4), (N, 2), (N, 2)))}  # mixed, mixed_bf, mixed_h


def fixture(edges=True):
    """The fixture's inputs, float32: r, gated [B, 4n], w, w_next [4n], lo [B, lr], bo, e [B, n], inj [B, 4], and rs
    [B, 4], gr_norm_rs's row scales of r with w (the norm twin's, so the kernel's bits), which the mix reads."""
    rng = np.random.default_rng(20261020)
    g = lambda *shape: rng.standard_normal(shape, dtype=F32)  # noqa: E731
    d = {"r": g(B, 4 * N), "w": g(4 * N), "w_next": g(4 * N), "gated": g(B, 4 * N), "lo": g(B, LR), "bo": g(B, N),
         "e": g(B, N), "inj": g(B, 4) * F32(3)}
    if edges:
        plant(d)
    d["rs"] = emu.hc_norm(d["r"], d["w"], EPS, RSQRT)[0].view(F32)
    return d


def plant(d):
    """E1-E8 (the module docstring's map) in ``d``'s rows, in place."""
    r, gated, lo, bo, e, inj = (d[k] for k in ("r", "gated", "lo", "bo", "e", "inj"))
    for c in range(4):                                                    # token 0: E2 on every stream of r, gated
        r[0, c * N + 5: c * N + 7] = gated[0, c * N + 5: c * N + 7] = (1e-40, 1e-36)
    lo[0, 5:7] = bo[0, 5:7] = e[0, 5:7] = (1e-40, 1e-36)                  # E2
    lo[0, 8:12] = (2.0 ** -126, 2.0 ** -125, 1.5 * 2.0 ** -124, 2.0 ** -123 * (1 - 2.0 ** -24))   # E7
    e.view(np.uint32)[0, 7] = E8_NAN                                      # E8, by its bits
    r[1, 3] = lo[1, 3] = inj[1, 0] = np.nan                               # token 1: E3
    r[1, N + 3] = lo[1, 4] = inj[1, 1] = np.inf                           # E4
    r[2, N: 2 * N] = 0                                                    # token 2: E6 (k = 0)
    r[2, N], r[2, N + 256] = 2.0 ** -70 * E6_S, (1 + 2.0 ** -12) * E6_S
    inj[2, 0] = 0                                                         # E5's inj 0
    r[3, :N] = 0                                                          # token 3: E1
    inj[3] = (-352, -400, 100, -100)                                      # E5
    for c in range(4):
        gated[3, c * N: c * N + 2] = (-88, -100)


def alone(d, t):
    """``d`` with token t's rows (by their bits) at row 0; the weights unchanged."""
    out = {k: v.copy() for k, v in d.items()}
    for k, v in out.items():
        if v.ndim == 2:
            v.view(np.uint32)[0] = d[k].view(np.uint32)[t]
    return out


def run(kind, d, rows, **kw):
    """The kind's kernel on ``d`` at bound B with ``rows`` live (prompt_hc's keywords beside): its outputs' words,
    [B, words per token] each. The inputs must keep their bytes."""
    entry, names, outs = KINDS[kind]
    ins = [np.ascontiguousarray(d[k]).tobytes() for k in names]
    res = gpu.prompt_hc(entry, ins, [B * w * s for w, s in outs], B=B, rows=rows, **kw)
    assert res[: len(ins)] == ins, f"{kind}: an input's bytes changed"
    return [np.frombuffer(b, WORD[s]).reshape(B, w) for b, (w, s) in zip(res[len(ins):], outs)]


def twin(kind, d, flush=False):
    """The kind's outputs' words for every token, from the host: broadcast copies e's 32-bit words into each stream;
    the others are basalt_emu's twins (``flush``: their flushing forms), with this GPU's fast::rsqrt and
    expf_ieee(sign_flip(x)) from hc_probe, so norm's rs is the probe's rsqrt of the argument the twin computes."""
    if kind == "broadcast":
        return [np.tile(d["e"].view(np.uint32), (1, 4))]
    if kind == "norm":
        return list(emu.hc_norm(d["r"], d["w"], EPS, RSQRT, flush))
    if kind == "silu":
        return [emu.hc_silu(d["lo"], EXP_NEG, flush)]
    if kind == "mix":
        return list(emu.hc_mix(d["r"], d["rs"], d["w"], d["gated"], EXP_NEG, flush))
    raise KeyError(kind)


def unwritten(g):
    return bool((g == np.iinfo(g.dtype).max).all())


@pytest.mark.parametrize("kind", list(KINDS))
def test_hc_twin(kind):
    d = fixture()
    want = twin(kind, d)
    for rows in (1, 7, B):
        for k, (g, w) in enumerate(zip(run(kind, d, rows), want)):
            bad = np.argwhere(g[:rows] != w[:rows])
            assert not len(bad), f"{kind} output {k} at {rows} rows: {len(bad)} words differ from the twin, at {bad[0]}"
            assert unwritten(g[rows:]), f"{kind} output {k}: a row at or past {rows} was written"
    assert all(unwritten(g) for g in run(kind, d, B, done=1)), f"{kind}: a run with done = 1 stored"
    if kind == "broadcast":                       # E2's subnormal words are live at every row count
        e = d["e"][:1]
        assert (emu.ftz(e).view(np.uint32) != e.view(np.uint32)).any(), "no live word a flushing copy would change"
    if kind == "norm":                            # a flushing kernel fails on E2 (token 0) and E6 (token 2, stream 1)
        flushed = twin(kind, d, flush=True)
        assert (flushed[1][0] != want[1][0]).any(), "E2: no xn16 word of token 0 differs from a flushing kernel's"
        assert flushed[0][2, 1] != want[0][2, 1], "E6: the flushing twin's rs equals the IEEE twin's (E6_S)"
    if kind == "mix":                             # MIX_BF 0 (the drafter's KV pass) stores no BF16 image
        bf1, bf0 = run(kind, d, B), run(kind, d, B, mix_bf=0)
        assert np.array_equal(bf0[0], bf1[0]) and np.array_equal(bf0[2], bf1[2]), "MIX_BF 0: mixed or mixed_h differ"
        assert unwritten(bf0[1]), "MIX_BF 0 stored into slot 5"
        flushed = twin(kind, d, flush=True)       # a flushing kernel fails on E2 (token 0) and E5 (token 3, column 0)
        assert any((f[0, 5:7] != i[0, 5:7]).any() for f, i in zip(flushed, want)), \
            "E2: no output word of token 0's columns 5-6 differs from a flushing kernel's"
        assert any(f[3, 0] != i[3, 0] for f, i in zip(flushed, want)), \
            "E5: no output word of token 3's column 0 (gated -88) differs from a flushing kernel's"
    if kind == "silu":                            # a flushing kernel fails on E7 (token 0, columns 8-11)
        flushed = twin(kind, d, flush=True)
        assert (flushed[0][0, 8:12] != want[0][0, 8:12]).any(), \
            "E7: no lo16 word of token 0's columns 8-11 differs from a flushing kernel's"


def test_hc_probe_specials():
    rs = gpu.hc_probe("rsqrt", np.array([np.inf, np.nan], F32))
    ex = gpu.hc_probe("exp_neg", np.array([np.inf, -100, -np.inf, np.nan], F32))
    assert rs.view(np.uint32)[0] == 0 and np.isnan(rs[1]), f"rsqrt(+inf), rsqrt(NaN): {rs.view(np.uint32)}"
    assert ex.view(np.uint32)[0] == 0 and ex[1] == ex[2] == np.inf and np.isnan(ex[3]), \
        f"exp_neg(+inf, -100, -inf, NaN): {ex.view(np.uint32)}"


@pytest.mark.parametrize("kind", list(KINDS))
def test_hc_rows_invariant(kind):
    d = fixture()
    full = run(kind, d, B)
    for t in (0, 1, B - 1):
        for k, (f, o) in enumerate(zip(full, run(kind, alone(d, t), 1))):
            assert np.array_equal(o[0], f[t]), f"{kind} output {k}: token {t} alone differs from its row in {B}"
    for k, (s, g) in enumerate(zip(run(kind, d, 7, static=True), run(kind, d, 7))):
        assert np.array_equal(s, g), f"{kind} output {k}: the static 7 rows differ from the dynamic 7"
    for k, (a, g) in enumerate(zip(full, run(kind, d, B))):
        assert np.array_equal(a, g), f"{kind} output {k}: two runs differ"
    slices = kernels.basalt_prompt_hc_slices(B, 64)
    assert slices == [(0, 64), (64, 64), (128, 64), (192, 64)], slices
    for rows in (B, 100):
        for k, (s, g) in enumerate(zip(run(kind, d, rows, slices=slices), run(kind, d, rows))):
            assert np.array_equal(s, g), f"{kind} output {k} at {rows} rows: 64-row slices differ from one slice"
    if kind == "broadcast":
        words = d["e"].view(np.uint32)
        for c in range(4):
            assert np.array_equal(full[0][:, c * N: (c + 1) * N], words), f"stream {c} is not e's words"
            assert full[0][0, c * N + 7] == E8_NAN, f"stream {c}: E8's payload {full[0][0, c * N + 7]:#010x}"
