// IEEE binary32 operations built on the software binary64, written for this port (the repository's own code; no
// upstream copy).
//
// CUDA's binary32 arithmetic in an IEEE-built kernel keeps subnormal operands and results; this GPU flushes them
// (tests/kernels/test_basalt_platform.py). Where basalt_ieee.metal's integer rebuilds do not reach, the exact result
// comes from basalt_f64.metal's binary64 rounded to odd, then to binary32. Needs basalt_ieee.metal and basalt_f64.metal
// ahead of it (monolith/kernels.py's basalt_gdn_source()).
#pragma STDC FP_CONTRACT OFF
