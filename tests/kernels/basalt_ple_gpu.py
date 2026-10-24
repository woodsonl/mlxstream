"""Single-kernel runs of the n-gram block's Metal port (kernels/common/basalt_gr_norm.metal and basalt_ple.metal) for
the GPU tests, on tests/kernels/basalt_gpu.py's device, pipelines and runs: one wrapper per entry, every output
starting as 0xFF bytes, and GPU probes of the basalt_fp.metal helpers whose values the numpy emulation
(tests/kernels/basalt_emu.py) takes from the GPU. The history's moves (advance, snapshot, restore) run on words, each
binding between GUARD bytes of 0xA5 (basalt's guards, ple_parity.cpp:178-190). ``chain`` runs the whole block, row by
row, as the window op dispatches it: main's q8_1, mmvq and mmvf, then the block's kernels. The prompt path's five
entries (``*_batch``: the prompt builds of the norm and the gate, and the three batch kernels) compile through
basalt_prompt_gpu.prompt_pipeline at the bound B of their exact-size buffers and run on its run_rows, the chunk's rows
in StepState at slot 15. ``batch`` runs native_ple_postops_batch's seven launches on them, and ``postops_chain``, its
oracle, the decode kernels after the projections, row by row with the history's advance after each."""

import numpy as np

from monolith import kernels
from monolith.runtime import _native as nt
from tests.kernels import basalt_gpu as gpu
from tests.kernels.basalt_prompt_gpu import prompt_pipeline, run_rows

GUARD = 64                                  # bytes of 0xA5 on both sides of a guarded binding (16 floats)
HS = kernels.PLE_HIST * kernels.PLE_D       # a history's words


def gr_norm(x, gamma, n_rows, n_cols, gamma_rows, eps):
    """``basalt_gr_norm`` in basalt's launch (``basalt_gr_norm_macros``): ``n_rows`` rows of ``n_cols`` float32 values
    ``x``, row r with row r % ``gamma_rows`` of ``gamma``, into an output that starts as 0xFF bytes; (n_rows, n_cols)
    float32. The block size fixes the reduction order, so the pipeline must take a threadgroup of that size."""
    macros = kernels.basalt_gr_norm_macros(n_cols)
    pso = gpu.pipeline(kernels.basalt_gr_norm_source(), "basalt_gr_norm", tuple(sorted(macros.items())))
    grid, threads = kernels.basalt_gr_norm_grid(n_rows, macros)
    assert pso.max_threads_per_threadgroup >= threads, (pso.max_threads_per_threadgroup, threads)
    ins = [np.ascontiguousarray(a, np.float32).tobytes() for a in (x, gamma)]
    out = gpu.run(pso, ins + [4 * n_rows * n_cols], kernels.basalt_gr_norm_params(n_cols, gamma_rows, eps), 3, grid,
                  threads, 0xFF)
    return np.frombuffer(out[2], np.float32).reshape(n_rows, n_cols)


def gate(key, query):
    """``basalt_ple_gate`` in basalt's launch (``PLE_GRIDS["gate"]``: one threadgroup of 512 threads per stream) on the
    PLE_H rows of PLE_N normalized key and query values, into gates that start as 0xFF bytes; (PLE_H,) float32. The
    block size fixes the reduction order, so the pipeline must take a threadgroup of that size."""
    pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_gate")
    grid, threads = kernels.PLE_GRIDS["gate"]
    assert pso.max_threads_per_threadgroup >= threads, (pso.max_threads_per_threadgroup, threads)
    ins = [np.ascontiguousarray(a, np.float32).reshape(grid * kernels.PLE_N).tobytes() for a in (key, query)]
    out = gpu.run(pso, ins + [4 * grid], kernels.basalt_ple_gate_params(), 3, grid, threads, 0xFF)
    return np.frombuffer(out[2], np.float32)


def broadcast(value, gate):
    """``basalt_ple_broadcast`` in basalt's launch: the PLE_N values times each of the PLE_H gates, into a PLE_D output
    that starts as 0xFF bytes; float32."""
    pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_broadcast")
    ins = [np.ascontiguousarray(value, np.float32).reshape(kernels.PLE_N).tobytes(),
           np.ascontiguousarray(gate, np.float32).reshape(kernels.PLE_H).tobytes()]
    out = gpu.run(pso, ins + [4 * kernels.PLE_D], None, None, *kernels.PLE_GRIDS["broadcast"], 0xFF)
    return np.frombuffer(out[2], np.float32)


def conv(hist, normalized, taps, hidden, gated):
    """``basalt_ple_conv`` in basalt's launch: ``hist`` the history in basalt's layout (row r of channel c at
    c * 9 + r), ``taps`` the fp16 kernel (float16 or its bits) at [k + 4 c], ``normalized``, ``hidden`` and ``gated``
    PLE_D values each; conv and result start as 0xFF bytes. Returns (conv, result), PLE_D float32 each."""
    pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_conv")
    d = kernels.PLE_D

    def f32(a, n):
        return np.ascontiguousarray(a, np.float32).reshape(n).tobytes()

    ins = [f32(hist, kernels.PLE_HIST * d), f32(normalized, d),
           np.ascontiguousarray(taps).view(np.uint16).reshape(4 * d).tobytes(), f32(hidden, d), f32(gated, d)]
    out = gpu.run(pso, ins + [4 * d, 4 * d], None, None, *kernels.PLE_GRIDS["conv"], 0xFF)
    return np.frombuffer(out[5], np.float32), np.frombuffer(out[6], np.float32)


def _words(a, n):
    """``a``, 4-byte elements (float32, uint32 or int32), as rows of ``n`` uint32 words with its bits."""
    a = np.ascontiguousarray(a)
    assert a.dtype.itemsize == 4, a.dtype
    return a.view(np.uint32).reshape(-1, n)


def _guarded(payload):
    """A buffer holding the bytes ``payload`` between GUARD bytes of 0xA5 on both sides (ple_parity.cpp:183-190)."""
    b = nt.Buffer(gpu.device(), len(payload) + 2 * GUARD)
    assert b.nbytes == len(payload) + 2 * GUARD, (b.nbytes, len(payload))
    b.fill(0xA5)
    b.write(payload, GUARD)
    return b


def _payload(b):
    """A guarded buffer's payload as uint32 words, and whether both guards still hold 0xA5 (ple_parity.cpp:221-227)."""
    raw = b.read(0, b.nbytes)
    return np.frombuffer(raw[GUARD:-GUARD], np.uint32), raw[:GUARD] + raw[-GUARD:] == b"\xa5" * (2 * GUARD)


def _launch(pso, bindings, grid, params=None):
    """A dispatch of ``pso`` in ``grid`` = (threadgroups, threads), binding k at ``bindings[k]`` = (buffer, byte
    offset) and ``params`` (if any) at the slot after them."""
    d = nt.Dispatch().pipeline(pso)
    for k, (b, offset) in enumerate(bindings):
        d = d.buffer(k, b, offset)
    if params is not None:
        d = d.bytes(len(bindings), params)
    return d.grid(grid[0]).threadgroup(grid[1])


def _dispatch(name, bindings, params=None):
    """``basalt_ple_<name>`` in basalt's launch (``PLE_GRIDS[name]``), binding k at ``bindings[k]`` = (buffer, byte
    offset)."""
    return _launch(gpu.pipeline(kernels.basalt_ple_source(), f"basalt_ple_{name}"), bindings, kernels.PLE_GRIDS[name],
                   params)


def _submit(q, *ds):
    """``ds`` in order, in one command buffer."""
    r = q.run(list(ds))
    assert not r.error, r.error


def advance(hist, normalized):
    """``basalt_ple_advance`` on the history ``hist`` (HS words in basalt's layout, row r of channel c at c * 9 + r),
    once per row of ``normalized`` (rows of PLE_D words): one dispatch, built once as basalt captures its advance
    (ple_parity.cpp:195-198) and run once per row after the row is written into its normalized buffer (:206-207).
    Returns the history after each run, (rows, HS) uint32, and whether every guard byte is intact."""
    rows = _words(normalized, kernels.PLE_D)
    bh, bn = _guarded(_words(hist, HS).tobytes()), _guarded(bytes(4 * kernels.PLE_D))
    d, q, out = _dispatch("advance", [(bh, GUARD), (bn, GUARD)]), nt.Queue(gpu.device()), []
    for row in rows:
        bn.write(row.tobytes(), GUARD)
        _submit(q, d)
        out.append(_payload(bh)[0])
    return np.array(out), _payload(bh)[1] and _payload(bn)[1]


def snapshot(hist, slot):
    """``basalt_ple_snapshot``: the history ``hist`` (HS words) into slot ``slot`` of PLE_SNAP_SLOTS snapshots of HS
    words, bound at the slot as the program binds it. The snapshots start as 0xA5 bytes, the slot as 0xFF bytes.
    Returns the snapshots after the run, (PLE_SNAP_SLOTS, HS) uint32, and whether every guard byte is intact."""
    bs = _guarded(b"\xa5" * (4 * HS * kernels.PLE_SNAP_SLOTS))
    bs.write(b"\xff" * (4 * HS), GUARD + 4 * HS * slot)
    bh = _guarded(_words(hist, HS).tobytes())
    _submit(nt.Queue(gpu.device()), _dispatch("snapshot", [(bs, GUARD + 4 * HS * slot), (bh, GUARD)]))
    return _payload(bs)[0].reshape(kernels.PLE_SNAP_SLOTS, HS), _payload(bs)[1] and _payload(bh)[1]


def restore(hist, snap, commit):
    """``basalt_ple_restore``: the history ``hist`` (HS words) from the PLE_SNAP_SLOTS snapshots ``snap`` (HS words
    each) by the commit record ``commit`` (int32 words, of which the kernel reads word 1). Returns the history after
    the run, HS uint32, and whether every guard byte is intact."""
    snap = _words(snap, HS)
    assert snap.shape[0] == kernels.PLE_SNAP_SLOTS, snap.shape
    bh, bs = _guarded(_words(hist, HS).tobytes()), _guarded(snap.tobytes())
    bc = _guarded(np.ascontiguousarray(commit, np.int32).tobytes())
    _submit(nt.Queue(gpu.device()), _dispatch("restore", [(bh, GUARD), (bs, GUARD), (bc, GUARD)]))
    return _payload(bh)[0], _payload(bh)[1] and _payload(bs)[1] and _payload(bc)[1]


SCRATCH = {  # the window op's per-row scratch (plan §3), bytes: q8_1, raw key, key norm, value, gate, gated, conv
    "q8_1": kernels.PLE_N // 32 * kernels.Q8_1_BLOCK_BYTES, "key": 4 * kernels.PLE_D, "key_norm": 4 * kernels.PLE_D,
    "value": 4 * kernels.PLE_N, "gate": 4 * kernels.PLE_H, "gated": 4 * kernels.PLE_D, "conv": 4 * kernels.PLE_D,
}


def chain(rows, hidden, hist, w, n=None, guard=False, split_query=False, inplace=False):
    """basalt's per-row n-gram block (verify.cpp:1105-1124: ple_block's native branch, ple.cu:298-326, then the
    history's advance and snapshot, :1118, :1123) as the window op's eleven dispatches a row (plan §3), on rows 0 to
    ``n`` - 1 (default: all) of the gathered rows ``rows`` (R, PLE_N) and the hidden rows ``hidden`` (R, PLE_D), from
    the history ``hist`` (HS values in basalt's layout, row r of channel c at c * 9 + r), with ``w``'s weights
    (tests.basalt_ple_ref.Fixture's key, value, w_nk, w_nq, w_nc and conv1d). A row is one command buffer:
    native_quantize_q8_1, basalt_mmvq (type 42) and basalt_mmvf at one column, with the pipelines, grids and params
    basalt_gpu.q8_1, mmvq and mmvf launch them with; the key norm; the query norm into normalized row r; the gate; the
    broadcast; the conv norm over normalized row r (basalt binds the two norms to one buffer, ple.cu:320-326); the
    convolution into conv and result row r; the advance of the history; its snapshot into slot r. Rows bind at r times
    their row bytes, as the program binds them, and every row shares the scratch, as basalt shares its workspace
    (ple.cu:282-292). Every output starts as 0xFF bytes, the scratch again before each row. Returns a dict: per row
    run, the scratch after it (``q8_1`` uint8, ``key`` the raw key, ``key_norm``, ``value``, ``gate``, ``gated``,
    ``conv``) and ``hist``, the history after it, (n, ...) each; ``normalized``, ``result`` and ``snap`` (R snapshot
    slots), all R rows (rows from n on as they started). The values are float32.

    ``guard``: every binding (inputs, weights, history, outputs and scratch) lies between GUARD bytes of 0xA5 and binds
    after its head guard (basalt's guards, ple_parity.cpp:183-190), and ``guards`` maps each buffer's name to whether
    both its guards still hold 0xA5. ``split_query``: the query norm writes its own buffer (``query``, scratch), which
    the gate reads, in place of normalized row r (basalt's permitted alias query == normalized undone,
    native_ple_postops.cu:344-351). ``inplace``: result binds onto hidden (basalt's in-place alias, ple_parity.cpp:957,
    verify.cpp:1105-1124), and ``result`` is read back from the hidden buffer."""
    N, D, H = kernels.PLE_N, kernels.PLE_D, kernels.PLE_H
    rows, hidden = _words(rows, N), _words(hidden, D)
    n = len(rows) if n is None else n
    assert len(hidden) == len(rows) and 1 <= n <= len(rows), (rows.shape, hidden.shape, n)
    key = np.ascontiguousarray(w.key).view(np.uint8).reshape(D, -1)
    assert key.shape[1] == N // 64 * 18, key.shape                       # type-42 rows: 40 blocks of 18 B
    mq, mf = kernels.basalt_mmvq_macros(42, N, 1), kernels.basalt_mmvf_macros(N, N, 1)
    nm = kernels.basalt_gr_norm_macros(N)
    q81 = gpu.pipeline(kernels.basalt_q8_1_source(), "native_quantize_q8_1")
    mmvq = gpu.pipeline(kernels.basalt_mmvq_source(), "basalt_mmvq", tuple(sorted(mq.items())))
    mmvf = gpu.pipeline(kernels.basalt_mmvf_source(), "basalt_mmvf", tuple(sorted(mf.items())))
    norm = gpu.pipeline(kernels.basalt_gr_norm_source(), "basalt_gr_norm", tuple(sorted(nm.items())))
    grids = (kernels.basalt_q8_1_grid(N), kernels.basalt_mmvq_grid(mq, D), kernels.basalt_mmvf_grid(mf, N),
             kernels.basalt_gr_norm_grid(H, nm))
    assert grids == ((10, 256), (2560, 128), (2560, 256), (4, 1024)), grids        # plan §3's one-column launches
    gate_pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_gate")
    for pso, threads in ((norm, grids[3][1]), (gate_pso, kernels.PLE_GRIDS["gate"][1])):
        assert pso.max_threads_per_threadgroup >= threads, (pso.max_threads_per_threadgroup, threads)
    r_all, pad = len(rows), GUARD if guard else 0
    scratch = {**SCRATCH, **({"query": 4 * D} if split_query else {})}
    data = {  # each binding's bytes, or its size for one that starts as 0xFF bytes
        "rows": rows.tobytes(), "hidden": hidden.tobytes(), "hist": _words(hist, HS).tobytes(),
        **{k: _words(getattr(w, k), D).tobytes() for k in ("w_nk", "w_nq", "w_nc")}, "w_key": key.tobytes(),
        "w_value": np.ascontiguousarray(w.value).view(np.uint16).reshape(N * N).tobytes(),
        "taps": np.ascontiguousarray(w.conv1d).view(np.uint16).reshape(4 * D).tobytes(),
        "normalized": 4 * r_all * D, "snap": 4 * r_all * HS, **({} if inplace else {"result": 4 * r_all * D}),
        **scratch}
    data = {k: b"\xff" * v if isinstance(v, int) else v for k, v in data.items()}
    b = {k: _guarded(v) if guard else nt.Buffer(gpu.device(), v) for k, v in data.items()}
    if inplace:
        b["result"] = b["hidden"]

    def at(k, offset=0):
        return b[k], pad + offset

    def payload(k):
        return b[k].read(pad, b[k].nbytes - 2 * pad)

    s = {k: at(k) for k in scratch}
    q81_p, mmvq_p, mmvf_p = (kernels.basalt_q8_1_params(N), kernels.basalt_mmvq_params(N, D),
                             kernels.basalt_mmvf_params(N, N, 1))
    norm_p, gate_p = kernels.basalt_gr_norm_params(N, H, kernels.PLE_RMS_EPS), kernels.basalt_ple_gate_params()
    q, out = nt.Queue(gpu.device()), {k: [] for k in (*scratch, "hist")}
    for r in range(n):
        for k in scratch:
            b[k].write(data[k], pad)
        x, h = at("rows", 4 * N * r), at("hidden", 4 * D * r)
        nr, res = at("normalized", 4 * D * r), at("result", 4 * D * r)
        qr = s["query"] if split_query else nr
        _submit(q,
                _launch(q81, [x, s["q8_1"]], grids[0], q81_p),                                    # ple.cu:298-299
                _launch(mmvq, [at("w_key"), s["q8_1"], s["key"]], grids[1], mmvq_p),              # :300-301
                _launch(mmvf, [x, at("w_value"), s["value"]], grids[2], mmvf_p),                  # :312-313
                _launch(norm, [s["key"], at("w_nk"), s["key_norm"]], grids[3], norm_p),           # postops.cu:352
                _launch(norm, [h, at("w_nq"), qr], grids[3], norm_p),                             # :353
                _dispatch("gate", [s["key_norm"], qr, s["gate"]], gate_p),                        # :355
                _dispatch("broadcast", [s["value"], s["gate"], s["gated"]]),                      # :356
                _launch(norm, [s["gated"], at("w_nc"), nr], grids[3], norm_p),                    # :358
                _dispatch("conv", [at("hist"), nr, at("taps"), h, s["gated"], s["conv"], res]),   # :359
                _dispatch("advance", [at("hist"), nr]),                                           # verify.cpp:1118
                _dispatch("snapshot", [at("snap", 4 * HS * r), at("hist")]))                      # :1123
        for k in out:
            out[k].append(payload(k))
    got = {k: np.frombuffer(b"".join(v), np.uint8 if k == "q8_1" else np.float32).reshape(n, -1)
           for k, v in out.items()}
    for k in ("normalized", "result", "snap"):
        got[k] = np.frombuffer(payload(k), np.float32).reshape(r_all, -1)
    if guard:
        got["guards"] = {k: _payload(v)[1] for k, v in b.items()}
    return got


def _helper(name, x):
    """basalt_fp.metal's float helper ``name`` on the GPU, elementwise (basalt_gpu.expf_ftz's pattern)."""
    src = kernels.PRELUDE + kernels.template("basalt_fp.metal") + f"""
kernel void probe(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], constant uint& n [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {{
    if (i < n) y[i] = {name}(x[i]);
}}
"""
    x = np.ascontiguousarray(x, np.float32)
    flat = x.reshape(-1)
    out = gpu.run(gpu.pipeline(src, "probe"), [flat.tobytes(), 4 * flat.size], np.uint32(flat.size).tobytes(), 2,
                  -(-flat.size // 256), 256, 0xFF)
    return np.frombuffer(out[1], np.float32).reshape(x.shape)


def rsqrtf_ftz(x):
    """basalt_fp.metal's rsqrtf_ftz (``rsqrtf`` in a fast-math file) on the GPU, elementwise: the values the norm
    uses."""
    return _helper("rsqrtf_ftz", x)


def sqrtf_ftz(x):
    """basalt_fp.metal's sqrtf_ftz (``sqrtf`` in a fast-math file) on the GPU, elementwise: the values the gate uses."""
    return _helper("sqrtf_ftz", x)


# ---- the prompt path: native_ple_postops_batch's entries (native_ple_postops.cu:382-397), one run each ------------
def _batch(entry, stage, B, buffers, params, rows, done):
    """``entry``'s prompt build at bound ``B`` (prompt_pipeline: the dynamic-T row macros beside PLE_PROMPT_MACROS, and
    basalt_gr_norm_macros(PLE_N) for the norm), its source looked up on monolith.kernels now, run once by run_rows in
    basalt's launch at T = B (``kernels.basalt_ple_batch_grids(B)[stage]``) with ``rows`` (t_this_step) and ``done``
    in StepState: ``buffers`` (bytes, or a byte count for an output, which starts as 0xFF bytes) at 0, 1, ... and
    ``params`` (if any) at 3. Returns the buffers' bytes. The block size fixes the norm's and the gate's reduction
    order, so every pipeline must take its whole block."""
    norm = entry == "basalt_gr_norm"
    source = kernels.basalt_gr_norm_source(prompt=True) if norm else kernels.basalt_ple_source(prompt=True)
    macros = {**kernels.PLE_PROMPT_MACROS, **(kernels.basalt_gr_norm_macros(kernels.PLE_N) if norm else {})}
    pso = prompt_pipeline(source, entry, B, macros)
    grid, threads = kernels.basalt_ple_batch_grids(B)[stage]
    assert pso.max_threads_per_threadgroup >= threads, (entry, pso.max_threads_per_threadgroup, threads)
    return run_rows(pso, buffers, params, 3, (grid, 1, 1), (threads, 1, 1), rows, 0, 0xFF, done)


def gr_norm_batch(x, gamma, rows, done=0):
    """``basalt_gr_norm``'s prompt build as basalt's ``rms_rep_kernel`` (native_ple_postops.cu:97-121, launched at
    :389, :390 and :393) at bound B: the B rows ``x`` (B, PLE_D), each PLE_H rows of PLE_N values, row r of the 4 B
    normalized with row r % PLE_H of ``gamma`` (PLE_D values) and eps = PLE_RMS_EPS, into an output that starts as 0xFF
    bytes; (B, PLE_D) float32."""
    x = _words(x, kernels.PLE_D)
    out = _batch("basalt_gr_norm", "norm", len(x), [x.tobytes(), _words(gamma, kernels.PLE_D).tobytes(), 4 * x.size],
                 kernels.basalt_gr_norm_params(kernels.PLE_N, kernels.PLE_H, kernels.PLE_RMS_EPS), rows, done)
    return np.frombuffer(out[2], np.float32).reshape(x.shape)


def gate_batch(key, query, rows, done=0):
    """``basalt_ple_gate``'s prompt build as basalt's batch launches ``gate_kernel`` (native_ple_postops.cu:391) at
    bound B: one threadgroup of 512 threads per stream, 4 B of them, on the B rows of normalized key and query values
    (B, PLE_D) each, into gates that start as 0xFF bytes; (B, PLE_H) float32."""
    key, query = _words(key, kernels.PLE_D), _words(query, kernels.PLE_D)
    assert key.shape == query.shape, (key.shape, query.shape)
    out = _batch("basalt_ple_gate", "gate", len(key), [key.tobytes(), query.tobytes(), 4 * kernels.PLE_H * len(key)],
                 kernels.basalt_ple_gate_params(), rows, done)
    return np.frombuffer(out[2], np.float32).reshape(len(key), kernels.PLE_H)


def broadcast_batch(value, gate, rows, done=0):
    """``basalt_ple_broadcast_batch`` (``broadcast_batch_kernel``, native_ple_postops.cu:122-127, launched at :392) at
    bound B: row t's PLE_N values ``value`` (B, PLE_N) times each of its PLE_H gates ``gate`` (B, PLE_H), into rows
    that start as 0xFF bytes; (B, PLE_D) float32."""
    value, gate = _words(value, kernels.PLE_N), _words(gate, kernels.PLE_H)
    assert len(value) == len(gate), (value.shape, gate.shape)
    out = _batch("basalt_ple_broadcast_batch", "broadcast", len(value),
                 [value.tobytes(), gate.tobytes(), 4 * kernels.PLE_D * len(value)], None, rows, done)
    return np.frombuffer(out[2], np.float32).reshape(len(value), kernels.PLE_D)


def conv_batch(hist, normalized, taps, hidden, gated, rows, done=0):
    """``basalt_ple_conv_batch`` (``conv_residual_batch_kernel``, native_ple_postops.cu:128-145, launched at :394) at
    bound B: ``hist`` the history (HS values in basalt's layout, row r of channel c at c * 9 + r), ``taps`` the fp16
    kernel (float16 or its bits) at [k + 4 c], and the B rows ``normalized``, ``hidden`` and ``gated`` (B, PLE_D) each,
    into a result that starts as 0xFF bytes, out of place where basalt adds into hidden; (B, PLE_D) float32."""
    normalized, hidden, gated = (_words(a, kernels.PLE_D) for a in (normalized, hidden, gated))
    assert normalized.shape == hidden.shape == gated.shape, (normalized.shape, hidden.shape, gated.shape)
    ins = [_words(hist, HS).tobytes(), normalized.tobytes(),
           np.ascontiguousarray(taps).view(np.uint16).reshape(4 * kernels.PLE_D).tobytes(), hidden.tobytes(),
           gated.tobytes()]
    out = _batch("basalt_ple_conv_batch", "conv", len(normalized), ins + [4 * normalized.size], None, rows, done)
    return np.frombuffer(out[5], np.float32).reshape(normalized.shape)


def history_batch(hist, normalized, rows, done=0):
    """``basalt_ple_history_batch`` (``history_batch_kernel``, native_ple_postops.cu:146-158, launched at :395) at
    bound B, on words: the history ``hist`` (HS words in basalt's layout) updated in place past the chunk's rows of
    the B rows ``normalized`` (B, PLE_D). Returns the history after the run, HS uint32."""
    normalized = _words(normalized, kernels.PLE_D)
    out = _batch("basalt_ple_history_batch", "history", len(normalized),
                 [_words(hist, HS).tobytes(), normalized.tobytes()], None, rows, done)
    return np.frombuffer(out[0], np.uint32)


def batch(key, value, hidden, hist, w, rows, b):
    """native_ple_postops_batch (native_ple_postops.cu:382-397) as the handler dispatches it, on the wrappers above at
    bound ``b`` with ``rows`` (t_this_step) in StepState: basalt's seven launches in order (:389-395), the key norm
    into its own rows where basalt normalizes the key in place (:389), the query norm and then the conv norm each into
    rows of normalized values (basalt's query_norm holds both, :390, :393), the result out of place where basalt adds
    into hidden (:394), then the history (:395). The raw keys ``key`` and the hidden rows ``hidden`` (rows of PLE_D
    values) and the values ``value`` (rows of PLE_N) fill the leading rows of their b-row inputs, whose other rows hold
    0xFF bytes; ``hist`` is the history (HS values in basalt's layout), ``w`` holds the norms and taps
    (tests.basalt_ple_ref.Fixture's w_nk, w_nq, w_nc and conv1d). Returns a dict of float32: ``key_norm``,
    ``normalized`` (the conv norm's), ``gated`` and ``result`` (b, PLE_D), ``gate`` (b, PLE_H), and ``hist``, the
    history after the run (HS)."""
    def inputs(a, n):
        a = _words(a, n)
        assert len(a) <= b, (a.shape, b)
        return np.concatenate([a, np.full((b - len(a), n), 0xFFFFFFFF, np.uint32)])

    key, hidden, value = inputs(key, kernels.PLE_D), inputs(hidden, kernels.PLE_D), inputs(value, kernels.PLE_N)
    key_norm = gr_norm_batch(key, w.w_nk, rows)                                       # :389
    query = gr_norm_batch(hidden, w.w_nq, rows)                                       # :390
    gate = gate_batch(key_norm, query, rows)                                          # :391
    gated = broadcast_batch(value, gate, rows)                                        # :392
    normalized = gr_norm_batch(gated, w.w_nc, rows)                                   # :393
    result = conv_batch(hist, normalized, w.conv1d, hidden, gated, rows)              # :394
    return {"key_norm": key_norm, "gate": gate, "gated": gated, "normalized": normalized, "result": result,
            "hist": history_batch(hist, normalized, rows).view(np.float32)}           # :395


def postops_chain(key, value, hidden, hist, w):
    """The batch's oracle, its header's claim (native_ple_postops.hpp:32-33): "element for element the arithmetic of T
    calls of native_ple_postops with the history advanced after each". native_ple_postops
    (native_ple_postops.cu:352-359) and the history's advance (verify.cpp:1118) on the decode builds (no guard),
    ``chain``'s stages after the projections, on each of the T raw keys ``key`` (T, PLE_D), values ``value``
    (T, PLE_N) and hidden rows ``hidden`` (T, PLE_D) in turn, from the history ``hist`` (HS values in basalt's layout),
    with ``w``'s norms and taps. Row r is one command buffer: the key norm, the query norm into normalized row r, the
    gate, the broadcast, the conv norm over normalized row r, the convolution into conv and result row r, and the
    advance. Every output starts as 0xFF bytes, and row r of every row buffer binds at r times its row bytes. Returns
    a dict of (T, ...) float32: ``key_norm``, ``gate``, ``gated``, ``normalized``, ``conv`` and ``result``, and
    ``hist``, the history after each row."""
    N, D, H = kernels.PLE_N, kernels.PLE_D, kernels.PLE_H
    nm = kernels.basalt_gr_norm_macros(N)
    norm = gpu.pipeline(kernels.basalt_gr_norm_source(), "basalt_gr_norm", tuple(sorted(nm.items())))
    grid = kernels.basalt_gr_norm_grid(H, nm)
    gate_pso = gpu.pipeline(kernels.basalt_ple_source(), "basalt_ple_gate")
    for pso, threads in ((norm, grid[1]), (gate_pso, kernels.PLE_GRIDS["gate"][1])):
        assert pso.max_threads_per_threadgroup >= threads, (pso.max_threads_per_threadgroup, threads)
    rows = {"key": _words(key, D), "value": _words(value, N), "hidden": _words(hidden, D)}
    T = len(rows["key"])
    assert len(rows["value"]) == len(rows["hidden"]) == T, {k: a.shape for k, a in rows.items()}
    outs = {"key_norm": D, "gate": H, "gated": D, "normalized": D, "conv": D, "result": D}     # a row's values
    rows.update({k: np.full((T, n), 0xFFFFFFFF, np.uint32) for k, n in outs.items()})
    data = {**{k: a.tobytes() for k, a in rows.items()}, "hist": _words(hist, HS).tobytes(),
            **{k: _words(getattr(w, k), D).tobytes() for k in ("w_nk", "w_nq", "w_nc")},
            "taps": np.ascontiguousarray(w.conv1d).view(np.uint16).reshape(4 * D).tobytes()}
    b = {k: nt.Buffer(gpu.device(), v) for k, v in data.items()}
    width = {k: a.shape[1] for k, a in rows.items()}

    def at(k, r=0):
        return b[k], 4 * width.get(k, 0) * r

    norm_p, gate_p = kernels.basalt_gr_norm_params(N, H, kernels.PLE_RMS_EPS), kernels.basalt_ple_gate_params()
    q, hists = nt.Queue(gpu.device()), []
    for r in range(T):
        kn, nr, g, gd, h = (at(k, r) for k in ("key_norm", "normalized", "gate", "gated", "hidden"))
        _submit(q,
                _launch(norm, [at("key", r), at("w_nk"), kn], grid, norm_p),                       # postops.cu:352
                _launch(norm, [h, at("w_nq"), nr], grid, norm_p),                                  # :353
                _dispatch("gate", [kn, nr, g], gate_p),                                            # :355
                _dispatch("broadcast", [at("value", r), g, gd]),                                   # :356
                _launch(norm, [gd, at("w_nc"), nr], grid, norm_p),                                 # :358
                _dispatch("conv", [at("hist"), nr, at("taps"), h, gd, at("conv", r), at("result", r)]),  # :359
                _dispatch("advance", [at("hist"), nr]))                                            # verify.cpp:1118
        hists.append(b["hist"].read(0, 4 * HS))
    got = {k: np.frombuffer(b[k].read(0, b[k].nbytes), np.float32).reshape(T, n) for k, n in outs.items()}
    got["hist"] = np.frombuffer(b"".join(hists), np.float32).reshape(T, HS)
    return got
