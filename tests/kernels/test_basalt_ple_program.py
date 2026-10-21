# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/ple_win_parity.cpp and src/kernels/ple_parity.cpp
# @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE): ple_win_parity's window against the verify
# window's per-row chain (:71-116: for T = 1..8 from one reset, the result rows, the history and the snapshots compared
# value by value, its criterion and its log line, :96-109), and ple_parity's guard byte 0xa5 (:186-187) as the
# snapshots' sentinel.
"""basalt's n-gram block on the decode path in compiled programs on the GPU: the window op ``basalt_ple_block`` in one
static program per width, T = 1 .. 8 (basalt captures one graph per width), and the commit op ``basalt_ple_commit`` in
a program of its own at t = 8, every program emitted with ``tail=None`` (spec G§3.3: the stock tail moves
``position``), from the hand-built graph of tests/contract/test_basalt_ple_ops.py (``window``, ``commit_program``):
the inputs ``ngram.rows`` and ``ngram.hidden``, the pack's six tensors bound from a BasaltSource over a .basalt file
that holds them under the pack's names and types (``write_file``), the states ``ngram.hist`` and ``ngram.snap``, and
the commit's record ``commit``. The file holds spec T§6.3's fixtures' weights, synthetic and real (tests.basalt_ple_ref
``synthetic`` and ``real``, the real ones cut from $BASALT_PLE_FIXTURE's ranges), and the runs take the fixture's
activations: rows 0 .. 7 of its gathered rows (emb) and hidden rows, and its history H0 in basalt's layout (row r of
channel c at c * 9 + r).

- The engines share by name (plan §3): the T = 8 window's engine is built first and holds every buffer the others
  share, the weight windows, ``ngram.hist``, ``ngram.snap``, the row I/O (``ngram.rows``, ``ngram.hidden``,
  ``ngram.result``, ``ngram.normalized``: the T = 8 program's are the largest), StepState and the ring; every other
  width's engine and the commit's take ``buffers=`` from it, and each name they hold in common is its buffer
  (asserted). They keep their own params records and scratch (the names carry the width) and, the commit's, the
  record. One encoding per program, replayed: between runs only buffers change, and each run's outputs are read
  before the next run.
- A window run (``run_window``) writes all T_MAX rows of rows and hidden and, unless it continues from the history the
  last run left, H0; the snapshots start as 0xa5 bytes, result and normalized as 0xFF bytes. A commit run
  (``run_commit``) writes its record (``commit_record``) first: the record's zero-filled arena would restore slot 0.
- X-WIN ``test_window_rows_equal_one_row_calls`` (spec T§8; W1, ple_win_parity.cpp:96-116): for T = 1..8 the T-row
  window from H0, its input rows at or past T NaN (``sentinel``), gives R_T[t] (result) and N_T[t] (normalized), and
  its commit with each n_keep in 1..T the committed history C_T[n_keep]; the one-row windows from H0, row t at row 0
  and NaN in rows 1..7, each committed with n_keep = 1 and continuing from the history the commit left, give R_1[t],
  N_1[t] and C_1[t], the history after the commit of row t. Criteria: "for every T in 1..8 and every t < T: R_T[t] ==
  R_1[t], N_T[t] == N_1[t] and C_T[t + 1] == C_1[t]", byte for byte, W1's ``fail += d0 + d1 + d2 != 0;`` (:109) with
  its log line; "The outputs of rows below T must equal the clean-input run bit for bit, which catches a kernel that
  reads past T": the window over the fixture's own rows T..7 gives the same result, normalized and snapshot rows below
  T and the same history after it; and X-NF rule 1, "With finite fixtures the count must be 0", over the output rows
  below T and the committed histories.
- X-SNAP ``test_snapshots_are_history_after_each_row`` (spec T§8; W1's snapshot comparison and verify.cpp:1123), on
  the windows of X-WIN, for every T and t < T: "rows 0..7 of snapshot t equal rows 1..8 of snapshot t − 1, byte for
  byte (rows 1..8 of the initial history when t = 0)", "row 8 of snapshot t equals row t's normalized output, byte for
  byte", and "snapshot rows T..T_MAX − 1 still hold their 0xa5 sentinel"."""

import numpy as np
import pytest

from monolith.packs.basalt import BasaltSource
from monolith.runtime import Engine
from tests.basalt_ple_ref import D, HIST, T_MAX, commit_record, nonfinite, real, synthetic
from tests.contract.test_basalt_ple_ops import commit_program, window, write_file
from tests.kernels import basalt_gpu as gpu

F32 = np.float32
WIDTHS = range(1, T_MAX + 1)
HS = HIST * D                                   # a history's words
FIXTURES = {"synthetic": synthetic, "real": real}
WEIGHTS = {  # each tensor of the pack (blk.1.ple_<name>.weight) and the fixture's field that holds its bytes
    "conv1d": "conv1d", "key": "key", "norm_conv": "w_nc", "norm_key": "w_nk", "norm_query": "w_nq", "value": "value"}
SNAP_FILL = 0xA5A5A5A5                          # a snapshot word no run has written
PIPELINES = {}                                  # one pipeline cache for every engine of the module


@pytest.fixture(scope="module", params=list(FIXTURES))
def session(request, tmp_path_factory):
    """The fixture ``request.param`` and the engines of its eight window programs and its commit program, sharing by
    name as the module docstring says: (fx, engines), ``engines[t]`` the T = t window's, ``engines["commit"]``."""
    fx = FIXTURES[request.param]()
    data = {f"blk.1.ple_{name}.weight": getattr(fx, field).tobytes() for name, field in WEIGHTS.items()}
    src = BasaltSource(write_file(tmp_path_factory.mktemp("ple") / "ple.basalt", data))
    progs = {t: window(src, t)[1] for t in WIDTHS}
    progs["commit"] = commit_program(src, T_MAX)
    dev = gpu.device()
    engines = {T_MAX: Engine(progs[T_MAX], dev, pipeline_cache=PIPELINES)}
    shared = engines[T_MAX].buffers
    for k in (*WIDTHS[:-1], "commit"):
        engines[k] = Engine(progs[k], dev, buffers=shared, pipeline_cache=PIPELINES)
    own = [(k, n) for k, e in engines.items() for n in e.buffers
           if n in shared and e.program.buffers[n].role != "params" and e.buffers[n] is not shared[n]]
    common = set(engines["commit"].buffers) & set(shared)
    assert not own and {"ngram.hist", "ngram.snap"} <= common and "commit" not in shared, (
        "plan §3: every engine takes the T = 8 engine's buffer for each name both hold, params records aside; "
        f"(engine, name) {own} hold their own; the commit's program holds {sorted(common)} of them")
    return fx, engines


def words(eng, name, n):
    """``eng``'s buffer ``name``, all of it, as rows of ``n`` uint32 words."""
    return np.frombuffer(eng.read(name), np.uint32).reshape(-1, n)


def run_window(engines, t, rows, hidden, hist=None):
    """One replay of the T = ``t`` window over ``rows`` (T_MAX, N) and ``hidden`` (T_MAX, D), all T_MAX rows of each
    written, from the history ``hist`` (HS float32 values in basalt's layout; None: the history the last run left), the
    snapshots first 0xa5 bytes, result and normalized 0xFF bytes. Returns its ``result`` and ``normalized`` (T_MAX rows
    of D words), its snapshots ``snap`` (T_MAX slots of HS words) and ``hist`` after it (HS words), uint32."""
    eng = engines[t]
    b = eng.buffers
    b["ngram.rows"].write(np.ascontiguousarray(rows, F32).tobytes(), 0)
    b["ngram.hidden"].write(np.ascontiguousarray(hidden, F32).tobytes(), 0)
    if hist is not None:
        b["ngram.hist"].write(np.ascontiguousarray(hist, F32).tobytes(), 0)
    b["ngram.snap"].fill(0xA5)
    b["ngram.result"].fill(0xFF)
    b["ngram.normalized"].fill(0xFF)
    assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
    return {"result": words(eng, "ngram.result", D), "normalized": words(eng, "ngram.normalized", D),
            "snap": words(eng, "ngram.snap", HS), "hist": words(eng, "ngram.hist", HS)[0]}


def run_commit(engines, record):
    """One replay of the commit, its record ``record`` (int32 words) written first. Returns the history after it, HS
    uint32 words."""
    eng = engines["commit"]
    eng.buffers["commit"].write(np.asarray(record, np.int32).tobytes(), 0)
    assert eng.run(1, steps_per_cb=1, in_flight=1).steps == 1
    return words(eng, "ngram.hist", HS)[0]


def sentinel(a, t):
    """The first T_MAX rows of ``a``, float32, with rows ``t`` and on NaN (spec T§8's sentinel input rows)."""
    out = np.array(a[:T_MAX], F32)
    out[t:] = np.nan
    return out


@pytest.fixture(scope="module")
def runs(session):
    """X-WIN's runs, in the module docstring's order, each read before the next: ``wide[T]`` the sentinel window,
    ``committed[T][n_keep - 1]`` its commit's C_T[n_keep], ``clean[T]`` the window over the fixture's rows; ``ones[t]``
    the one-row window of row t and ``chain[t]`` its commit's C_1[t]; ``h0`` the initial history (HS float32)."""
    fx, engines = session
    h0 = np.ascontiguousarray(fx.hist.T, F32).ravel()
    wide, committed, clean = {}, {}, {}
    for t in WIDTHS:
        wide[t] = run_window(engines, t, sentinel(fx.emb, t), sentinel(fx.hidden, t), h0)
        committed[t] = [run_commit(engines, commit_record(n, t, 0)) for n in range(1, t + 1)]
        clean[t] = run_window(engines, t, fx.emb[:T_MAX], fx.hidden[:T_MAX], h0)
    ones, chain = [], []
    for t in range(T_MAX):
        rows, hidden = sentinel(fx.emb, 1), sentinel(fx.hidden, 1)
        rows[0], hidden[0] = fx.emb[t], fx.hidden[t]
        ones.append(run_window(engines, 1, rows, hidden, h0 if t == 0 else None))
        chain.append(run_commit(engines, commit_record(1, 1, t)))
    return {"h0": h0, "wide": wide, "committed": committed, "clean": clean, "ones": ones, "chain": chain}


def test_window_rows_equal_one_row_calls(runs):
    ones, chain = runs["ones"], runs["chain"]
    fail, diffs, past, nf = 0, {}, [], []
    for t in WIDTHS:
        w, c, ct = runs["wide"][t], runs["clean"][t], runs["committed"][t]
        d0 = sum(int(np.count_nonzero(w["result"][r] != ones[r]["result"][0])) for r in range(t))
        d1 = sum(int(np.count_nonzero(w["normalized"][r] != ones[r]["normalized"][0])) for r in range(t))
        d2 = sum(int(np.count_nonzero(ct[r] != chain[r])) for r in range(t))      # C_T[r + 1] against C_1[r]
        print(f"T={t}  result {d0}  normalized {d1}  history {d2}  differing values{'  DIFF' if d0 + d1 + d2 else ''}")
        fail += d0 + d1 + d2 != 0
        if d0 + d1 + d2:
            diffs[t] = (d0, d1, d2)
        past += [(t, k) for k in ("result", "normalized", "snap") if not np.array_equal(w[k][:t], c[k][:t])]
        past += [(t, "hist")] if not np.array_equal(w["hist"], c["hist"]) else []
        for k, a in (("result", w["result"][:t]), ("normalized", w["normalized"][:t]), ("history", np.array(ct))):
            if n := nonfinite(a.view(F32)):
                nf.append((t, k, n))
    assert not fail, (
        "X-WIN (spec T§8), \"for every T in 1..8 and every t < T: R_T[t] == R_1[t], N_T[t] == N_1[t] and C_T[t + 1] == "
        "C_1[t]\", byte for byte, W1's `fail += d0 + d1 + d2 != 0;` (ple_win_parity.cpp:109): differing values "
        f"(result, normalized, history) at T {diffs}")
    assert not past, (
        "X-WIN (spec T§8), \"The outputs of rows below T must equal the clean-input run bit for bit, which catches a "
        "kernel that reads past T\": with NaN in input rows T..7, (T, output) "
        f"{past} differ from the window over the fixture's rows")
    assert not nf, (
        "X-NF rule 1 (spec T§8), \"With finite fixtures the count must be 0\": (T, output, non-finite values) "
        f"{nf} in the output rows below T and the committed histories")


def test_snapshots_are_history_after_each_row(runs):
    h0 = runs["h0"].view(np.uint32).reshape(D, HIST)
    shifted, entered, kept = [], [], []
    for t in WIDTHS:
        w = runs["wide"][t]
        snap = w["snap"].reshape(T_MAX, D, HIST)                # row r of channel c at [slot, c, r]
        for r in range(t):
            prev = h0 if r == 0 else snap[r - 1]
            if not np.array_equal(snap[r][:, :HIST - 1], prev[:, 1:]):
                shifted.append((t, r))
            if not np.array_equal(snap[r][:, HIST - 1], w["normalized"][r]):
                entered.append((t, r))
        kept += [(t, r) for r in range(t, T_MAX) if (w["snap"][r] != SNAP_FILL).any()]
    assert not shifted and not entered and not kept, (
        "X-SNAP (spec T§8): \"rows 0..7 of snapshot t equal rows 1..8 of snapshot t − 1, byte for byte (rows 1..8 of "
        f"the initial history when t = 0)\": (T, t) {shifted} differ; \"row 8 of snapshot t equals row t's normalized "
        f"output, byte for byte\": (T, t) {entered} differ; \"snapshot rows T..T_MAX − 1 still hold their 0xa5 "
        f"sentinel\": (T, slot) {kept} were written")
