// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/dec_gdn_rec.cu:30-307,
// src/kernels/cuda/verify_kernels.cu:29-57, :113-213 and :377-461 and include/basalt/kernels/dec_gdn.hpp:16-39 @
// b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), written in the rounding sequence nvcc compiles
// for them (docs/research/basalt-nvcc/dec_gdn.md, docs/research/basalt-nvcc/verify_gdn.md).
//
// basalt's GDN decode window on Metal: one function per GDN step, called alike by dec_gdn_rec and by the per-stage
// kernels dec_gdn_parity grades it against (gdn_conv_l2_multi, gdn_ab_multi, gdn_step_norm_multi: a test-only
// reference side), so that its bitwise checks hold here. Needs basalt_fp.metal, basalt_ieee.metal, basalt_f64.metal,
// basalt_ieee_f64.metal, basalt_q8_1.metal and basalt_libdevice.metal ahead of it (monolith/kernels.py's
// basalt_gdn_source()).
#pragma STDC FP_CONTRACT OFF

// Every entry point's record (monolith/kernels.py's basalt_gdn_params): the window's width T (1..8, a program
// constant), the channels C = 2 * 128 * h_k + 128 * h_v (q, k, then v), n_embd (2560), the key and value heads, and
// the norms' eps.
struct BsGdnParams { uint T; uint C; uint n_embd; uint h_k; uint h_v; float eps; uint pad0; uint pad1; };
