// Transcribed from NVIDIA libdevice's __nv_exp and __nv_logf as nvcc 13.0 (V13.0.48, -arch=sm_120 -lineinfo -O3
// -DNDEBUG) inlined them into basalt's sampler.cu (jesdga95/basalt src/kernels/cuda/sampler.cu
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36; third_party/NOTICE); PTX listing and transcription:
// docs/research/basalt-nvcc/sampler.md sections 2.5-2.6.
//
// CUDA's binary64 exp and binary32 logf, which are not correctly rounded, so no Metal function gives their bits: the
// sampled tail's exp((double) x - (double) max) and logf(min_p), operation for operation. Needs basalt_f64.metal ahead
// of it.
#pragma STDC FP_CONTRACT OFF
