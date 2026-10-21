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
  flushing twins' rs differ); inj 0 on stream 0 (E5: 2 sigmoid = 1, gr_parity.cpp:641's plain add) and -400 on
  stream 1 (E5's zero sigmoid: sc = +0, so gr_write_norm_rs's R' row is E6's r, bit for bit, and its sum of squares
  meets the tie).
- token 3: E1, an all-zero r row (stream 0: rs = fast::rsqrt(1e-6)), with E5's inj (-352, -400, 100, -100): -352 / 4
  = -88 has a subnormal sigmoid, -400 / 4 = -100 a zero one; gated -88 and -100 on all four streams of columns 0 and 1.

gr_parity's bounds are the nearest bound; no basalt test launches these kernels (gr_parity gates decode's gr_read and
gr_write). Its metrics, rel_diff (normalised L1, the reference first) and rel_terms (per element, against the terms'
magnitudes), are src/kernels/gr_parity.cpp:136-156's, copied here until the hc lane's gr-post adds them to
tests/basalt_hc_ref.py.
Both run on the fixture without edge tokens at 1, 7 and 256 live rows and print each measured value:
- HC-1 (``test_hc_read_double``): gr_norm_rs's rs and xn16, gr_mix_r's mixed, mixed_bf and mixed_h (MIX_BF 1) and
  gr_silu's lo16 against float64 references, rs = 1 / sqrt(mean(r^2) + eps) per stream, xn = r rs w, mixed = sum_c r
  rs w / (1 + exp(-gated)) / 4 (with that rs; the kernel reads the fixture's) and lo16 = (lo / 4) / (1 + exp(-lo / 4));
  a BF16 or FP16 image's reference is that value rounded to float32, then by gr_parity's to_bf16 (:46-53, its
  reference's own activation rounding, :96, :111) or to FP16 to nearest even. Per output rel_diff(want, got) <= 1e-4
  (``const bool ok = rm <= 1e-4 && ri <= 1e-4;``, :566), no output non-finite.
- HC-2 (``test_hc_write_double``): gr_write's and gr_write_norm_rs's R' against want = R + bo w_c, w_c = 2 / (1 +
  exp(-inj_c / 4)) in float64 (:681, :684), worst rel_terms(want, got, R, bo w_c) < 1e-6 (``const bool streams_ok =
  worst_stream < 1e-6;``, :689); then with inj = 0, where w_c = 1 and want = R + bo, worst rel_terms(want, got, R, bo)
  < 1e-6 (``const bool plain = worst_plain < 1e-6;``, :641, :645); gr_write_norm_rs's rs and xn16 under HC-1's measure
  on its own R' with w_next; no output non-finite.

- HC-3 (``test_hc_twin``): at 1, 7 and 256 live rows every output word of the live rows equals the kind's twin
  (broadcast: e's 32-bit words in each stream; norm: basalt_emu.hc_norm, whose rs is the probe's fast::rsqrt of the
  argument fl(fl(S / n) + eps) the twin computes; silu and mix: basalt_emu.hc_silu and hc_mix, with the probe's
  expf_ieee(sign_flip(x)) of the twin's arguments; write and write_norm: basalt_emu.hc_write and hc_write_norm, the
  latter hc_write followed by hc_norm with w_next), the rows past them keep the 0xFF fill, the inputs keep their
  bytes, and a run with done = 1 stores nothing; the mix at MIX_BF 0 (the drafter's KV pass, no BF16 image) stores the
  same mixed and mixed_h and leaves slot 5 at 0xFF. Discrimination, so a flushing kernel fails: broadcast's words
  differ from their flushed values on E2; norm's IEEE and flushing twins differ in token 0's xn16 (E2) and in token
  2's stream-1 rs (E6), mix's in token 0's columns 5-6 (E2) and token 3's column 0 (E5: gated -88 on every stream, a
  subnormal sigmoid), silu's in token 0's columns 8-11 (E7; its E2 words are BF16 zeros either way), write's and
  write_norm's in R' (and write_norm's xn16) at token 0's columns 5-6 (E2) and token 3's stream 0 (E5: inj -352, a
  subnormal sigmoid whose double is normal, on E1's zero row, so R' = fl(bo sc)), and write_norm's in token 2's
  stream-1 rs (E6).
  ``test_hc_probe_specials`` (K1): the probe's fast::rsqrt and expf_ieee(sign_flip(x)) on the specials match
  PTX's documented rsqrt.approx and ex2.approx: rsqrt(+inf) = +0, rsqrt(NaN) NaN, exp_neg(+inf) = +0, exp_neg(-100)
  and exp_neg(-inf) +inf, exp_neg(NaN) NaN.
- HC-4 (``test_hc_rows_invariant``): bitwise, tokens 0, 1 and 255 of the 256-row run equal each token alone at one
  row; a static program's 7 rows (T_STATIC_ROWS) equal the dynamic 7; two runs give the same bytes; four forced 64-row
  slices equal the one-slice run at 256 and 100 live rows; broadcast's every stream equals e's words, E8's payload
  included; write_norm's three outputs equal gr_write followed by gr_norm_rs on its R' with w_next (KE 3.4 (a),
  (b))."""

from functools import partial

import numpy as np
import pytest

from monolith import kernels
from monolith.formats.fp import bf16_to_f32
from tests.kernels import basalt_emu as emu
from tests.kernels import basalt_prompt_gpu as gpu

F32 = np.float32
B, N, LR = 256, 2560, 320
EPS = F32(1e-6)              # the handler's HC_EPS, 0x358637BD (prompt_hc's)
E6_S = 1.0                   # E6's s: twins' rs 0x424a13f8 (IEEE) vs 0x424a13f9 (flush) on the M3 Max; 2^-4 ties them
E8_NAN = 0x7FA00001
WORD = {4: np.uint32, 2: np.uint16}
RSQRT, EXP_NEG = (partial(gpu.hc_probe, fn) for fn in ("rsqrt", "exp_neg"))   # the GPU's approximate operations
NEAREST = "nearest bound; no basalt test launches this kernel: gr_parity.cpp"   # HC-1's and HC-2's gates (ER-D5)
READ_GATE = f"{NEAREST}:566 const bool ok = rm <= 1e-4 && ri <= 1e-4;"
STREAMS_GATE = f"{NEAREST}:689 const bool streams_ok = worst_stream < 1e-6;"
PLAIN_GATE = f"{NEAREST}:645 const bool plain = worst_plain < 1e-6;"

# kind: (entry, inputs, outputs as (words per token, bytes per word)), in the kernel's slot order
KINDS = {"broadcast": ("prompt_gr_broadcast", ("e",), ((4 * N, 4),)),
         "norm": ("prompt_gr_norm_rs", ("r", "w"), ((4, 4), (4 * N, 2))),          # rs, xn16
         "silu": ("prompt_gr_silu", ("lo",), ((LR, 2),)),                           # lo16
         "mix": ("prompt_gr_mix_r", ("r", "rs", "w", "gated"), ((N, 4), (N, 2), (N, 2))),  # mixed, mixed_bf, mixed_h
         "write": ("prompt_gr_write", ("r", "bo", "inj"), ((4 * N, 4),)),                  # R'
         "write_norm": ("prompt_gr_write_norm_rs", ("r", "bo", "inj", "w_next"),
                        ((4 * N, 4), (4, 4), (4 * N, 2)))}                                 # R', rs, xn16


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
    inj[2, :2] = (0, -400)                                                # E5's inj 0; -400 makes write_norm's R' E6's
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
    if kind == "write":
        return [emu.hc_write(d["r"], d["bo"], d["inj"], EXP_NEG, flush)]
    if kind == "write_norm":
        return list(emu.hc_write_norm(d["r"], d["bo"], d["inj"], d["w_next"], EPS, RSQRT, EXP_NEG, flush))
    raise KeyError(kind)


def unwritten(g):
    return bool((g == np.iinfo(g.dtype).max).all())


# gr_parity's metrics, as the hc lane's tests/basalt_hc_ref.py (gr-post) defines them; until it lands here, local copies
def rel_terms(want, got, term_a, term_b):           # gr_parity.cpp:147-156, elementwise
    """``|want - got| / (|term_a| + |term_b|)``, the denominator floored at 1e-30, with every argument converted to
    float64 first, as basalt's ``double`` parameters are: a float32 denominator would round in binary32."""
    want, got, term_a, term_b = (np.asarray(v, np.float64) for v in (want, got, term_a, term_b))
    den = np.abs(term_a) + np.abs(term_b)
    return np.abs(want - got) / np.where(den > 1e-30, den, 1e-30)


def rel_diff(a, b):                                 # gr_parity.cpp:136-145
    """The normalised L1 difference ``sum |a - b| / sum |a|`` in float64, the denominator floored at 1e-30."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d, mag = np.abs(a - b).sum(), np.abs(a).sum()
    return d / (mag if mag > 1e-30 else 1e-30)


def norm_double(r, w):
    """gr_norm_rs in float64 (gr_parity.cpp's reference, :76-86, in double throughout) for the rows ``r`` [T, 4n] and
    the weights ``w`` [4n]: rs [T, 4] = 1 / sqrt(mean(r^2) + eps) per stream (eps the float32 1e-6, widened, as
    ``(double) eps``) and xn [T, 4n] = r rs w."""
    r = np.asarray(r, np.float64).reshape(len(r), 4, N)
    rs = 1 / np.sqrt((r * r).mean(axis=2) + np.float64(EPS))
    return rs, (r * rs[..., None] * np.asarray(w, np.float64).reshape(4, N)).reshape(len(r), 4 * N)


def bf(x):
    """gr_parity's to_bf16 (:46-53, to nearest even; emu.bf16_bits) of ``x`` rounded to float32, as a float32 value."""
    return bf16_to_f32(emu.bf16_bits(np.asarray(x, F32)))


def read_gate(case, want, got):
    """HC-1's measure: rel_diff(want, got) <= 1e-4 with every value of ``got`` finite; prints the value."""
    rel, bad = rel_diff(want, got), np.count_nonzero(~np.isfinite(got))
    print(f"{case}: rel_diff {rel:.3e}")
    assert bad == 0 and rel <= 1e-4, f"{READ_GATE} ({case}): rel_diff {rel:.3e}, {bad} not finite"


@pytest.mark.parametrize("T", (1, 7, B))
def test_hc_read_double(T):
    d = fixture(edges=False)
    rs, xn = norm_double(d["r"], d["w"])
    mixed = (xn / (1 + np.exp(-d["gated"].astype(np.float64)))).reshape(B, 4, N).sum(axis=1) / 4
    q = d["lo"].astype(np.float64) / 4
    lo = q / (1 + np.exp(-q))
    (rs_k, xn_k), (m_k, mbf_k, mh_k), (lo_k,) = (run(kind, d, T) for kind in ("norm", "mix", "silu"))
    for name, want, got in (("norm rs", rs, rs_k.view(F32)), ("norm xn16", bf(xn), bf16_to_f32(xn_k)),
                            ("mix mixed", mixed, m_k.view(F32)), ("mix mixed_bf", bf(mixed), bf16_to_f32(mbf_k)),
                            ("mix mixed_h", emu.h2f(emu.half_bits(mixed.astype(F32))), emu.h2f(mh_k)),
                            ("silu lo16", bf(lo), bf16_to_f32(lo_k))):
        read_gate(f"HC-1 T {T} {name}", want[:T], got[:T])


@pytest.mark.parametrize("T", (1, 7, B))
def test_hc_write_double(T):
    d = fixture(edges=False)
    r, bo = d["r"].astype(np.float64)[:T], d["bo"].astype(np.float64)[:T]
    for case, inj, gate in (("inj", d["inj"], STREAMS_GATE), ("inj 0", np.zeros_like(d["inj"]), PLAIN_GATE)):
        w = 2 / (1 + np.exp(-inj.astype(np.float64)[:T] / 4))     # :681; 1 exactly at inj 0, so want = R + bo (:641)
        bw = (bo[:, None, :] * w[..., None]).reshape(T, 4 * N)
        e = {**d, "inj": inj}
        (rp,), (rp2, rs_k, xn_k) = run("write", e, T), run("write_norm", e, T)
        for name, got in (("write R'", rp), ("write_norm R'", rp2)):
            g = got[:T].view(F32)
            worst, bad = rel_terms(r + bw, g, r, bw).max(), np.count_nonzero(~np.isfinite(g))
            print(f"HC-2 T {T} {case} {name}: worst rel_terms {worst:.3e}")
            assert bad == 0 and worst < 1e-6, f"{gate} ({name}, {case}, {T} rows): worst {worst:.3e}, {bad} not finite"
        rs, xn = norm_double(rp2[:T].view(F32), d["w_next"])      # write_norm's norm half, on its own R'
        read_gate(f"HC-2 T {T} {case} write_norm rs", rs, rs_k[:T].view(F32))
        read_gate(f"HC-2 T {T} {case} write_norm xn16", bf(xn), bf16_to_f32(xn_k[:T]))


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
    if kind in ("write", "write_norm"):           # a flushing kernel fails on E2 (token 0) and E5 (token 3, stream 0)
        flushed = twin(kind, d, flush=True)
        for k in (0,) if kind == "write" else (0, 2):                     # R', and write_norm's xn16
            assert (flushed[k][0, 5:7] != want[k][0, 5:7]).any(), \
                f"E2: no word of output {k} at token 0's columns 5-6 differs from a flushing kernel's"
            assert (flushed[k][3, :N] != want[k][3, :N]).any(), f"E5: no word of output {k} at token 3's stream 0 " \
                "(inj -352 on E1's zero row) differs from a flushing kernel's"
    if kind == "write_norm":                      # and on E6 (token 2, stream 1: sc = +0 leaves E6's r as R')
        assert flushed[1][2, 1] != want[1][2, 1], "E6: the flushing twin's rs equals the IEEE twin's (E6_S)"


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
    if kind == "write_norm":                      # KE 3.4 (a), (b): gr_write, then gr_norm_rs on its R' with w_next
        rp = run("write", d, B)[0]
        for k, (f, g) in enumerate(zip(full, [rp, *run("norm", {"r": rp.view(F32), "w": d["w_next"]}, B)])):
            assert np.array_equal(f, g), f"write_norm output {k} differs from gr_write followed by gr_norm_rs"
