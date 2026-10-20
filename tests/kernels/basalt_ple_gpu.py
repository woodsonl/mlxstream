"""Single-kernel runs of the n-gram block's Metal port (kernels/common/basalt_gr_norm.metal and basalt_ple.metal) for
the GPU tests, on tests/kernels/basalt_gpu.py's device, pipelines and runs: one wrapper per entry, every output
starting as 0xFF bytes, and GPU probes of the basalt_fp.metal helpers whose values the numpy emulation
(tests/kernels/basalt_emu.py) takes from the GPU. The history's moves (advance, snapshot, restore) run on words, each
binding between GUARD bytes of 0xA5 (basalt's guards, ple_parity.cpp:178-190)."""

import numpy as np

from monolith import kernels
from monolith.runtime import _native as nt
from tests.kernels import basalt_gpu as gpu

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


def _dispatch(name, bindings):
    """``basalt_ple_<name>`` in basalt's launch (``PLE_GRIDS[name]``), binding k at ``bindings[k]`` = (buffer, byte
    offset)."""
    d = nt.Dispatch().pipeline(gpu.pipeline(kernels.basalt_ple_source(), f"basalt_ple_{name}"))
    for k, (b, offset) in enumerate(bindings):
        d = d.buffer(k, b, offset)
    grid, threads = kernels.PLE_GRIDS[name]
    return d.grid(grid).threadgroup(threads)


def _submit(q, d):
    r = q.run([d])
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
