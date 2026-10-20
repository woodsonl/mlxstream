# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) tests/cuda/prefill_cpu_share_test.cpp and tests/cuda/prefill_mmq_kq_test.cpp @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): their fixtures (fill_matrix, make_blob, make_x,
# make_ids, the blobs' allocations), their double reference and their compare, and prefill_mmq_kq_test's MK-4 batches
# and gate (:379-434).
"""The fixture of the prompt path's MoE MMQ tests on Q2_0/Q2_0 expert blobs at 2560/640 (tests/kernels/
test_basalt_moe_mmq.py; the expert-tier lane's CPU-pool test reuses it unchanged). numpy only, no Metal.

basalt runs no Q2_0/Q2_0 pair, so no basalt output exists to match and libstdc++'s distributions are not replayed:
the generator keeps basalt's distributions, parameters and order of draws (the blobs, then x, then the ids) on
``np.random.default_rng(seed)``, with basalt's seed formulas at Q2_0's type id 42 for traceability.

- ``fill_q2_0`` is prefill_cpu_share_test's fill_matrix at Q2_0 (:68-80): every byte uniform, then one fp16 scale per
  block, U(0.004, 0.024) in float32 rounded to nearest even; no second field.
- ``make_x`` (:98-106): N(0, 1) activations; with outliers one value in 200 is a 12-sigma draw; token 5 all zero.
- ``make_ids`` (:108-124): ten distinct experts per token, slots 0-2 skewed to the ``hot`` lowest ids, the top
  ``unused`` ids never drawn.
- The five configurations (``CONFIGS``): ``share`` is prefill_cpu_share_test's geometry (12 tokens, hot 6, the whole
  id table drawn again while an expert has more than MAXT rows, :269-272; one seed for both activation kinds, :255);
  ``kq`` prefill_mmq_kq_test's (256 tokens, hot 6, ids 61-63 unused, :311; seed :299); ``wide`` the port's own, kq at
  512 tokens, whose hot experts take more than 128 rows (column tiles ``ntx > 1``, empty column tiles), under kq-plain's
  seed. ``config`` builds one and asserts what its tests need from it (``_coverage``).
- Placement (cpu_share :259-264): the 64 blobs in one buffer, expert e at ``e * (BYTES + 64) + 2 * (e % 2)`` (its own
  allocation of BYTES + 64 bytes, odd experts 2 bytes in), every other byte 0xFF: an fp16 NaN scale wherever a read
  strays past a blob, which basalt's group buffers never meet.
- ``reference`` is both tests' double reference (cpu_share :332-360, mmq_kq :337-365), ``compare`` their compare
  (cpu_share :211-236; mmq_kq :259-282's rmax, printed and never gated).
- ``mk4`` is mmq_kq's batch-independence check (:379-434) with the chain injected as ``run``: the host emulator
  (basalt_emu.mmq_moe) or the Metal chain, so the bound measured on the one is computed by the code the other runs."""

import functools
from types import SimpleNamespace

import numpy as np

from monolith.kernels import DENSE_TC_SM_COUNT
from tests.kernels import basalt_emu as emu
from tests.kernels.basalt_cpu_emu import unpack64_q2_0

F32 = np.float32
N, FF, K, E = 2560, 640, 10, 64          # cpu_share :36, :253
MAXT = 8                                 # the CPU share's rows per expert (include/basalt/kernels/cpu/expert.hpp:122)
QK, QB = 64, 18                          # Q2_0: 64 values in 18 bytes, the fp16 d then qs[16]
UP_OFF, DOWN_OFF, BYTES = 460800, 921600, 1382400   # Geo (:56-65) at Q2_0: gate (640 x N) | up | down (N x 640)
SLACK = 64                               # each blob's allocation, Dev(geo.bytes + 64) (:261)
PRODUCTS = {"gate_up": (2 * FF // 128, N // QK), "down": (N // 128, FF // QK)}   # (nty, bpne) per MMQ product

# name: (tokens, hot, unused, outlier activations, at most MAXT rows per expert, seed). Seeds: cpu_share's
# 7 + gu * 5 + d (:255) and mmq_kq's 11 + gu * 7 + d (+ 1000 with outliers, :299) at gu = d = 42.
CONFIGS = {"share-plain": (12, 6, 0, False, True, 259), "share-outliers": (12, 6, 0, True, True, 259),
           "kq-plain": (256, 6, 3, False, False, 347), "kq-outliers": (256, 6, 3, True, False, 1347),
           "wide-plain": (512, 6, 3, False, False, 347)}


def fill_q2_0(rng, rows, cols):
    """fill_matrix (cpu_share :68-80) for a Q2_0 matrix [rows, cols]: every byte uniform first, then per block the
    scale ``sc(rng)`` of ``std::uniform_real_distribution<float> sc(lo, 6.0f * lo)``, lo 0.004f, all in float32,
    stored as fp16 (ggml_fp32_to_fp16, nearest even) at bytes 0-1."""
    nb = rows * (cols // QK)
    m = rng.integers(0, 256, nb * QB, dtype=np.uint8).reshape(nb, QB)
    lo = F32(0.004)
    d = lo + (F32(6.0) * lo - lo) * rng.random(nb, dtype=np.float32)
    m[:, :2] = d.astype("<f2").view(np.uint8).reshape(nb, 2)
    return m.reshape(-1)


def make_blob(rng):
    """make_blob (:81-86): gate and up (1280 rows of N), then down (N rows of FF)."""
    return np.concatenate([fill_q2_0(rng, 2 * FF, N), fill_q2_0(rng, N, FF)])


def make_x(rng, T, outliers, zero=5):
    """make_x (:98-106): N(0, 1) float32 activations [T, N], times 12 for one value in 200 with ``outliers``; token
    ``zero`` all zero."""
    x = rng.standard_normal((T, N), dtype=np.float32)
    if outliers:
        x *= np.where(rng.integers(0, 200, (T, N)) == 0, F32(12.0), F32(1.0))
    x[zero] = 0.0
    return x


def make_ids(rng, T, hot, unused):
    """make_ids (:108-124): per token ten distinct experts, ``e = (hot > 0 && k < 3 && rng() % 4 != 0) ? rng() % hot :
    any(rng)`` with ``any`` uniform on [0, E - 1 - unused], drawn again while it repeats an earlier slot (at most 64
    draws a slot here)."""
    ids = np.zeros((T, K), np.int32)
    for t in range(T):
        for k in range(K):
            for _ in range(64):
                skew = hot > 0 and k < 3 and rng.integers(0, 4) != 0
                e = rng.integers(0, hot) if skew else rng.integers(0, E - unused)
                if e not in ids[t, :k]:
                    break
            else:
                raise RuntimeError(f"make_ids: token {t} slot {k} repeated 64 draws")
            ids[t, k] = e
    return ids


def dequant_q2_0(raw, rows, cols):
    """ggml's dequantize_row_q2_0 (basalt src/kernels/cpu/ggml/ggml_quants.c:209-227) of Q2_0 bytes into float32
    [rows, cols], vectorized: ``((int) q - 1) * d``, code q of value m ``(qs[m // 4] >> (2 * (m % 4))) & 3``. Exact: an
    integer in {-1, 0, 1, 2} times an fp16 value."""
    b = np.asarray(raw, np.uint8).reshape(-1, QB)
    d = b[:, :2].copy().view("<f2").astype(F32)
    return ((unpack64_q2_0(b[:, 2:]).astype(F32) - 1) * d).reshape(rows, cols)


def blob(cfg, e):
    """Expert e's BYTES bytes, a view into the placed buffer."""
    return cfg.buf[cfg.offsets[e]: cfg.offsets[e] + BYTES]


def launches(cfg, product, nsm=DENSE_TC_SM_COUNT):
    """basalt's MMQ launches of ``product`` ("gate_up" or "down") for the configuration's plan, at ``nsm``
    multiprocessors (basalt's 170 by default), each with its tiles' expressions (``emu.mmq_walk``) in ``tiles``."""
    nty, bpne = PRODUCTS[product]
    out = emu.mmq_launches(cfg.plan.bounds, nty, nsm)
    for la in out:
        la.tiles = emu.mmq_walk(la.ntiles, bpne, la.grid)
    return out


def _split(la):
    return any(len(emu.mmq_pieces(e)) > 1 for e in la.tiles.values())


def _coverage(cfg):
    """What the configuration's tests need from its draws, at basalt's 170 multiprocessors. ``share``: every expert at
    most MAXT rows, so tile width 8 everywhere, three full groups of 16 (unsplit) and a short last group whose gate/up
    launch splits; its down launch splits too unless it has 8 experts (160 tiles: share-outliers, measured). ``kq``:
    every usable expert routed (61: groups of 16, 16, 16, 13), the hot experts' group at tile width 128, the full groups
    unsplit and the last group split in both products. ``wide``: a hot expert over 128 rows (column tiles ``ntx > 1``)
    and an empty column tile (an expert with no rows at ``jt``)."""
    p, name = cfg.plan, cfg.name
    cnt = np.diff(p.bounds)
    gu, dn = launches(cfg, "gate_up"), launches(cfg, "down")
    unsplit = not any(_split(la) for la in gu[:-1] + dn[:-1])
    if name.startswith("share"):
        assert cnt.max() <= MAXT and {la.J for la in gu} == {8} and 48 < p.n < 64, (name, cnt.max(), p.n)
        assert unsplit and _split(gu[-1]) and (_split(dn[-1]) or dn[-1].ngx == 8), name
    elif name.startswith("kq"):
        assert p.n == E - 3 and [la.ngx for la in gu] == [16, 16, 16, 13] and gu[0].J == 128, (name, p.n, cnt.max())
        assert unsplit and _split(gu[-1]) and _split(dn[-1]), name
    else:
        assert any(la.ntx > 1 for la in gu), (name, cnt.max())
        assert any(jt * la.J >= c for la in gu for c in cnt[la.j0: la.j0 + la.ngx] for jt in range(la.ntx)), name


@functools.lru_cache(maxsize=None)
def config(name):
    """One of ``CONFIGS``, built in basalt's draw order (the blobs, then x, then the ids; run_pair :251-272 and
    :296-311): ``buf`` the placed blobs (poisoned slack), ``offsets`` [E] their byte offsets in it, ``x`` [T, N]
    float32, ``ids`` [T, K] int32 and ``plan`` their ``emu.moe_plan`` (``plan.slot`` is the tests' row_of). Cached:
    its arrays are read-only."""
    T, hot, unused, outliers, maxt, seed = CONFIGS[name]
    rng = np.random.default_rng(seed)
    offsets = np.arange(E, dtype=np.int64) * (BYTES + SLACK) + 2 * (np.arange(E) % 2)
    buf = np.full(E * (BYTES + SLACK), 0xFF, np.uint8)
    for e in range(E):
        buf[offsets[e]: offsets[e] + BYTES] = make_blob(rng)
    x = make_x(rng, T, outliers)
    for _ in range(1000):                                  # cpu_share :269-272: the whole table again
        ids = make_ids(rng, T, hot, unused)
        if not maxt or np.bincount(ids.ravel(), minlength=E).max() <= MAXT:
            break
    else:
        raise RuntimeError(f"{name}: 1000 id tables, each with an expert over {MAXT} rows")
    for a in (buf, offsets, x, ids):
        a.flags.writeable = False
    cfg = SimpleNamespace(name=name, T=T, outliers=outliers, buf=buf, offsets=offsets, x=x, ids=ids,
                          plan=emu.moe_plan(ids, E))
    _coverage(cfg)
    return cfg


@functools.lru_cache(maxsize=None)
def reference(name):
    """The double reference of both basalt tests (cpu_share :332-360, mmq_kq :337-365), float32 [T * K, N] in pair
    order: per routed expert ggml's dequantized weights; per pair the gate and up rows' float64 dot products with the
    token's activations, ``h = (float) (gt / (1.0 + std::exp(-gt)) * up)``, then each down row's float64 dot product
    with h, rounded to float. No q8_1 rounding, routing weight or combine. BLAS sums in another order than basalt's
    sequential loops: a few 1e-16 relative, at most one float rounding moved by an ulp, orders below the gates."""
    cfg = config(name)
    flat = cfg.ids.ravel()
    ref = np.zeros((flat.size, N), F32)
    for e in np.unique(flat):
        pairs = np.flatnonzero(flat == e)
        w = blob(cfg, e)
        gu = dequant_q2_0(w[:DOWN_OFF], 2 * FF, N).astype(np.float64)
        dn = dequant_q2_0(w[DOWN_OFF:], N, FF).astype(np.float64)
        x = cfg.x[pairs // K].astype(np.float64)
        gt, up = x @ gu[:FF].T, x @ gu[FF:].T
        h = (gt / (1.0 + np.exp(-gt)) * up).astype(F32)
        ref[pairs] = (h.astype(np.float64) @ dn.T).astype(F32)
    ref.flags.writeable = False
    return ref


def compare(dm, ref, row_of):
    """compare (cpu_share :211-236) of ``dm`` [rows, N] in sorted-row order against ``ref`` [pairs, N] in pair order,
    row ``row_of[i]`` facing pair i: ``rms`` the relative RMS pooled over every pair, ``worst`` the largest row's
    max |d| / max |ref| (``else if (me > 0) out.worst = 1e30;   // a zero row must stay exactly zero``), and mmq_kq's
    ``rmax`` (:274), max |d| / |ref| over the elements above 1 % of their row's max. Every max is std::max's (a NaN
    never replaces the running value); basalt throws at a non-finite output first (:224), which the tests count (G1)."""
    a = np.asarray(dm, F32)[np.asarray(row_of).ravel()].astype(np.float64)
    b = np.asarray(ref, F32).astype(np.float64)
    d = a - b
    me = np.fmax.reduce(np.abs(d), axis=1, initial=0.0)
    mr = np.fmax.reduce(np.abs(b), axis=1, initial=0.0)
    row = np.where(mr > 0, me / np.where(mr > 0, mr, 1.0), np.where(me > 0, 1e30, 0.0))
    big = np.abs(b) > 0.01 * mr[:, None]
    r2 = float((b * b).sum())
    return SimpleNamespace(rms=float(np.sqrt((d * d).sum() / r2)) if r2 > 0 else 0.0,
                           worst=float(np.fmax.reduce(row, initial=0.0)),
                           rmax=float(np.fmax.reduce(np.abs(d[big]) / np.abs(b[big]), initial=0.0)))


def mk4(cfg, run):
    """MK-4 (prefill_mmq_kq_test.cpp:379-434) on configuration ``cfg`` with ``run(x, ids)`` returning Dm [rows, N] in
    sorted-row order (moe_plan's, whose ``slot`` places a pair's row): the configuration's own run, then for t = 0..7
    token t alone and as the last token of a 3-token batch whose other tokens u = 0, 1 take x rows 100 + u and ten
    experts scanned from E - 1 down, none of token t's, even for u = 0 and odd for u = 1 (:383-404). Returns ``em``
    (the full run's compare against the reference: G3's numbers), ``worst_alone`` (:416-423: per row of token t the
    largest float difference |a - b| over the largest |b|, b its row in the full run; std::max never takes a NaN),
    ``differ`` (:422, rows with any difference) and ``nonfinite`` (the small runs' non-finite values, which MK-4 lets
    through). basalt's gate: ``if (worst_alone > 0.1 * em.worst)`` fails (:433)."""
    dm = run(cfg.x, cfg.ids)
    em = compare(dm, reference(cfg.name), cfg.plan.slot)
    worst, differ, nonfinite = 0.0, 0, 0
    for t in range(8):
        for tn in (1, 3):                                  # variant 0: token t alone; 1: token t last of three
            ids1, x1 = np.zeros((tn, K), np.int32), np.zeros((tn, N), F32)
            ids1[-1], x1[-1] = cfg.ids[t], cfg.x[t]
            for u in range(tn - 1):                        # another token's activations and experts not routed for t
                filled = 0
                for e in range(E - 1, -1, -1):             # for (int e = E - 1; filled < K; --e)
                    if filled < K and e not in cfg.ids[t] and e not in ids1[u, :filled] and (e + u) % 2 == 0:
                        ids1[u, filled] = e
                        filled += 1
                assert filled == K, (t, u)
                x1[u] = cfg.x[100 + u]
            d1 = run(x1, ids1)
            nonfinite += np.count_nonzero(~np.isfinite(d1))
            slot1 = emu.moe_plan(ids1, E).slot
            for k in range(K):
                a, b = d1[slot1[-1, k]], dm[cfg.plan.slot[t, k]]
                with np.errstate(over="ignore", invalid="ignore"):
                    md = np.fmax.reduce(np.abs(a - b).astype(np.float64), initial=0.0)
                mr = np.fmax.reduce(np.abs(b).astype(np.float64), initial=0.0)
                differ += int(md > 0)
                worst = max(worst, md / mr if mr > 0 else md)
    return SimpleNamespace(em=em, worst_alone=float(worst), differ=differ, nonfinite=nonfinite)
