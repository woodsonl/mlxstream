// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/sampler.cu, include/basalt/kernels/sampler.hpp and
// include/basalt/core/coupled_draft.hpp @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE),
// written in the rounding sequence nvcc compiles for them (docs/research/basalt-nvcc/sampler.md).
//
// basalt's sampler family on Metal. Needs basalt_fp.metal, basalt_ieee.metal, basalt_f64.metal and
// basalt_libdevice.metal ahead of it (monolith/kernels.py's basalt_sampler_source()).
#pragma STDC FP_CONTRACT OFF
