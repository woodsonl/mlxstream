# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jesús Gulfo
# adapted from basalt (jesdga95/basalt) src/kernels/cuda/hc_fused.cu:1171-1227 (hc8_quant_kernel, hc8_make),
# include/basalt/core/model_file.hpp:126-149 and src/core/model_file.cpp:438-485 (the BASALTH8 table),
# src/core/verify.cpp:566-581 and src/core/mtp.cpp:215-218, :488-503 (the copied matrices) @
# b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE).
"""Basalt's int8 copies of the hyper-connection matrices, made on the host.

A copy of a BF16 matrix ``w [rows][cols]`` (``cols % 32 == 0``) is one block of bytes, the allocation ``hc8_make``
fills (``src/kernels/cuda/hc_fused.cu:1210-1227``): the codes ``q [rows][cols]`` as int8, row-major, then the scales
``s [rows][srow]`` as f16, ``srow = (cols / 32 + 7) & ~7``. Scale ``j`` of row ``r`` covers ``w[r][32 j .. 32 j + 31]``:
``s = f16_rne(absmax / 127)`` and ``q = clamp(rint(w / s), -127, 127)``, every code of the group 0 where ``s`` is 0, and
the ``srow - cols / 32`` padding scales of each row are 0. ``hc8_quant_kernel`` (``hc_fused.cu:1171-1197``) computes it
in IEEE binary32: subnormals kept, both divisions correctly rounded, both roundings to nearest even. ``quantize`` is
that arithmetic in numpy, operation for operation; ``docs/research/basalt-nvcc/hc_fused.md`` section 1 records what
nvcc compiles it to.

Basalt quantizes every hyper-connection down, up and injection matrix of the target and of the drafter at load, from
the BF16 bytes its model file stores, and its reads then take only the copies (``src/program/generate.cpp:2233-2365``,
``src/core/mtp.cpp:254-282``, ``:488-503``). The port makes the copies on the host, once per ``.basalt`` file:

* Exact without a flush argument: the host CPU computes binary32 with subnormals kept, the kernel's own arithmetic,
  where a Metal pass would flush them.
* No new runtime mechanism: the repository has no run-once setup program, and packs prepared on the host exist already.
* Memory: the GPU binds only the copies, never the target's 1,271,398,400 bytes of BF16 down and up matrices beside the
  decode program.

The copies pack is a :class:`~monolith.packs.packer.PackFile` directory. Its data file ``weights.pack`` is basalt's
``BASALTH8`` table as ``hc8_write`` lays it out (``include/basalt/core/model_file.hpp:126-149``,
``src/core/model_file.cpp:438-485``): a 16-byte header (``b"BASALTH8"``, version 1 and the entry count as ``uint32``),
160-byte entries (the name NUL-padded to 128 bytes; rows, cols, srow and 0 as ``uint32``; offset and bytes as
``uint64``), each copy's bytes at a 256-aligned offset with zero fill between, and the file zero-padded to a multiple
of 16384 bytes, because the runtime maps whole pages only. Its manifest lists two aux rows per copy, under basalt's
table name (``blk.L.hc_attn_down.weight`` and the like; ``mtp.`` then the tensor's name for the drafter's):
``hc8/<name>/q``, ``U8 [rows, cols]`` at the entry's offset, and ``hc8/<name>/s``, ``F16 [rows, srow]`` at that
offset plus ``rows * cols``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

G8 = 32                                           # values per scale group (hc_fused.cu:343)


def quantize(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The copy of the BF16 matrix whose bits are ``w`` (``uint16 [rows, cols]``): ``(q, s)``, ``q`` the codes' int8
    bit patterns as ``uint8 [rows, cols]`` and ``s`` the scales' f16 bits as ``uint16 [rows, srow]``, padding scales
    0, so that ``q.tobytes() + s.tobytes()`` is basalt's allocation. Raises ``ValueError`` unless ``w`` is a 2-D
    ``uint16`` array with ``rows`` and ``cols`` positive, ``cols % 32 == 0`` and every weight finite. A finite weight
    above ``127 * 65504`` is basalt's case too: its group's scale is inf and every code of the group 0."""
    if not (isinstance(w, np.ndarray) and w.dtype == np.uint16 and w.ndim == 2):
        raise ValueError(f"quantize: the weights must be a 2-D uint16 array of BF16 bits, not "
                         f"{type(w).__name__} {getattr(w, 'dtype', '')} of shape {getattr(w, 'shape', '')}")
    rows, cols = w.shape
    if rows == 0 or cols == 0 or cols % G8:
        raise ValueError(f"quantize: a {rows} x {cols} matrix; rows and cols must be positive and cols a multiple of "
                         f"{G8}, so that a group of {G8} never straddles a row")
    bad = (w & 0x7F80) == 0x7F80
    if bad.any():
        r, c = np.argwhere(bad)[0]
        raise ValueError(f"quantize: {int(bad.sum())} non-finite weights, the first at row {r}, column {c} "
                         f"(bits 0x{int(w[r, c]):04X}); a valid pack's hyper-connection matrices have none")
    gpr = cols // G8
    x = (w.astype(np.uint32) << 16).view(np.float32).reshape(rows * gpr, G8)    # :1178 the BF16 bits as fp32
    amax = np.abs(x).max(axis=1)                  # :1179 the fmaxf chain from 0: exact on finite weights
    with np.errstate(over="ignore"):              # above f16's range: inf, basalt's scale too
        h = (amax / np.float32(127)).astype(np.float16)                          # :1181 div.rn.f32, cvt.rn.f16.f32
    s = h.astype(np.float32)                      # :1182 exact
    with np.errstate(divide="ignore", invalid="ignore"):                         # s == 0: the quotients are unused
        v = np.where(s[:, None] > 0, np.clip(np.rint(x / s[:, None]), -127, 127), 0)  # :1185 div.rn, rni, max, min
    q = v.astype(np.int8).view(np.uint8).reshape(rows, cols)                    # :1186 integral, in [-127, 127]
    sc = np.zeros((rows, (gpr + 7) & ~7), np.uint16)                            # hc8_make's memset (:1223)
    sc[:, :gpr] = h.view(np.uint16).reshape(rows, gpr)                          # :1183
    return q, sc
