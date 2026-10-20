"""Single-kernel runs of basalt's GDN decode window (kernels/common/basalt_gdn.metal) for the GPU tests, on the geometry
of tests/basalt_gdn_ref.py.

- Every entry point reads one ``BsGdnParams`` record (``params``: ``kernels.basalt_gdn_params``) at buffer 16; buffer
  15, StepState's slot, stays unbound.
- ``launch`` binds each input at exactly its bytes, so shader validation sees a read past one, and each output as its
  rows plus one sentinel row, every byte 0xFF (NaN as a float). A window of width T writes rows t < T only, so every
  output's bytes from row T on must keep their 0xFF: an activation's row T, the window stores' rows T..7 and their row
  8. ``launch`` reports that as ``sentinel_ok``, asserts the run's error empty and returns its result (``gpu_ms``)
  with every buffer's bytes after the run, the inputs' included (the state's: the window must leave it untouched).
"""

import numpy as np

from monolith import kernels
from monolith.runtime import _native as nt
from tests.basalt_gdn_ref import C, EPS, HK, HV, N
from tests.kernels import basalt_gpu as gpu

PARAMS_SLOT = 16
FILL = 0xFF


def params(t):
    """``BsGdnParams`` for a window of width t on the fixture's geometry."""
    return kernels.basalt_gdn_params(t, C, N, HK, HV, float(EPS))


def pipeline(name, macros=None):
    """The entry point ``name`` of ``basalt_gdn_source()`` built with ``macros`` (e.g. {"GDN_Y_BOUND": "1"})."""
    return gpu.pipeline(kernels.basalt_gdn_source(), name, tuple(sorted((macros or {}).items())))


def launch(pso, t, inputs, outputs, grid, threads):
    """One dispatch of ``pso`` for a window of width t: ``grid`` threadgroups (a tuple) of ``threads`` threads.
    ``inputs``: {slot: array}, each bound at exactly its bytes; ``outputs``: {slot: (rows, row_bytes)}, each bound as
    rows + 1 rows of row_bytes, every byte 0xFF. Returns (the RunResult, {slot: the buffer's bytes after the run as
    uint8}, sentinel_ok: every output's bytes from row t on still 0xFF)."""
    assert pso.max_threads_per_threadgroup >= threads, (threads, pso.max_threads_per_threadgroup)
    dev = gpu.device()
    bufs = {k: nt.Buffer(dev, np.ascontiguousarray(a).tobytes()) for k, a in inputs.items()}
    for k, (rows, row_bytes) in outputs.items():
        bufs[k] = nt.Buffer(dev, (rows + 1) * row_bytes)
        bufs[k].fill(FILL)
    d = nt.Dispatch().pipeline(pso)
    for k, b in bufs.items():
        d = d.buffer(k, b)
    r = nt.Queue(dev).run([d.bytes(PARAMS_SLOT, params(t)).grid(*grid).threadgroup(threads)])
    assert not r.error, r.error
    got = {k: np.frombuffer(b.read(0, b.nbytes), np.uint8) for k, b in bufs.items()}
    return r, got, all((got[k][t * row_bytes:] == FILL).all() for k, (_, row_bytes) in outputs.items())
