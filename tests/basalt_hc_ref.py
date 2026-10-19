# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/hc_fused.cu:1171-1197 (hc8_quant_kernel),
# src/core/model_file.cpp:487-513 (hc8_read) and src/kernels/hc8_parity.cpp:69-99 (hc8_parity's matrices) @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""Host references of basalt's hyper-connection unit for the port's tests: numpy only, no Metal, imported by every tier
like ``tests/moe_synth.py``. Each is a transcription of basalt's source written apart from the code it checks, so the
two cannot share a mistake.

* ``hc8_quant_scalar``: ``hc8_quant_kernel``, one group and one value at a time, a numpy float32 or float16 scalar at
  every step (the host keeps binary32 subnormals, as the IEEE-built kernel does); the independent side of
  ``monolith.packs.basalt_hc8.quantize``.
"""

import numpy as np


def hc8_quant_scalar(w, gpr, srow):                  # hc8_quant_kernel, hc_fused.cu:1171-1197, G8 = 32 (:343)
    """The copy of the BF16 bits ``w`` (any shape, flattened row-major; ``gpr`` scales per row, ``srow`` the row
    stride of the scales): ``(q, sc)``, the codes as ``int8 [rows * cols]`` and the scales' f16 bits as
    ``uint16 [rows * srow]``, padding scales 0."""
    w = np.asarray(w, np.uint16).reshape(-1)
    groups = w.size // 32
    q = np.zeros(w.size, np.int8)
    sc = np.zeros(groups // gpr * srow, np.uint16)   # cudaMemset(c.s, 0, rows * srow * 2) (:1223)
    for gi in range(groups):
        x = [np.uint32(int(w[gi * 32 + i]) << 16).view(np.float32) for i in range(32)]  # :1178
        amax = np.float32(0.0)
        for i in range(32):
            amax = np.fmax(amax, np.abs(x[i]))       # :1179 fmaxf(amax, fabsf(x[i]))
        with np.errstate(over="ignore"):             # above f16's range the scale is inf, as cvt.rn.f16.f32 gives
            h = np.float16(amax / np.float32(127.0))  # :1181 __float2half_rn(amax / 127.0f): div.rn, RNE
        s = np.float32(h)                            # :1182 __half2float(h)
        sc[gi // gpr * srow + gi % gpr] = h.view(np.uint16)                            # :1183
        for i in range(32):                          # :1185-1186
            v = np.fmin(np.fmax(np.rint(x[i] / s), np.float32(-127.0)), np.float32(127.0)) if s > 0 else np.float32(0.0)
            q[gi * 32 + i] = np.int8(int(v))
    return q, sc
