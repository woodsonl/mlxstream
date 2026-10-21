// Copyright (c) 2023-2026 The ggml authors (MIT)
// Copyright (c) 2026 Jesús Gulfo (MIT)
// Adapted from basalt (jesdga95/basalt) src/kernels/cuda/native_router.cu:33-100 and src/kernels/cuda/hc_fused.cu:816-916
// @ b4c63b55754637f4c72ad53a0f791d425f71eb36 (MIT; third_party/NOTICE), which adapts ggml-org/llama.cpp
// ggml/src/ggml-cuda/{topk-moe.cu,common.cuh} @ 3cf03257f219afbe7334045ff7c6a06ac68c627d (MIT; third_party/NOTICE).
//
// basalt's router top-10 over 512 experts: per token, the softmax of the 512 logits, the ten largest probabilities in
// order (ties to the lower expert id), and their weights renormalized by their sum floored at 2^-14. One kernel for
// native_router.cu's route (native_router_top10, native_router_top10_multi) and the fused read's hc_top10, which give
// the same bits for every input (route's ten argmax rounds and hc_top10's candidate ranking pick the same ten in the
// same order). The float steps are a fast-math build's flush-to-zero instructions: max.ftz, sub.ftz, mul.ftz,
// ex2.approx.ftz, add.ftz, rcp.approx.ftz, mul.ftz, written as basalt_fp.metal's max_ftz, add_ftz of -M, expf_ftz,
// add_ftz, rcp_ftz and mul_ftz (ex2.approx and rcp.approx mapped as docs/design/flash-next.md's invariant 5 states;
// docs/research/basalt-nvcc/hc_fused.md section 8 records the SASS of both). Needs basalt_fp.metal ahead of it.
//
// Grid: T threadgroups of 32 threads (one SIMD-group each); threadgroup k is token k.
#pragma STDC FP_CONTRACT OFF

kernel void basalt_router_top10(device const float* logits [[buffer(0)]], device int* ids [[buffer(1)]],
                                device float* weights [[buffer(2)]], uint k [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    logits += ulong(k) * 512u;
    ids += ulong(k) * 10u;
    weights += ulong(k) * 10u;
    float values[16];
    for (int i = 0; i < 16; ++i) values[i] = logits[lane + 32 * i];
    float maximum = -INFINITY;
    for (int i = 0; i < 16; ++i) maximum = max_ftz(maximum, values[i]);
    maximum = warp_max_ftz(maximum);
    float sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        values[i] = expf_ftz(add_ftz(values[i], -maximum));     // expf(v - M): sub.ftz, mul.ftz by log2(e), ex2.approx.ftz
        sum = add_ftz(sum, values[i]);
    }
    const float reciprocal = rcp_ftz(warp_sum_ftz(sum));
    for (int i = 0; i < 16; ++i) {
        values[i] = mul_ftz(values[i], reciprocal);
        if ((as_type<uint>(values[i]) & 0x7fffffffu) > 0x7f800000u) values[i] = as_type<float>(0xff7fffffu);   // NaN -> -FLT_MAX
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = values[0];
        int expert = int(lane);
        for (int i = 1; i < 16; ++i)
            if (values[i] > best) { best = values[i]; expert = int(lane) + 32 * i; }
        for (ushort mask = 16; mask > 0; mask >>= 1) {
            const float other = shfl_xor(best, mask);
            const int other_id = simd_shuffle_xor(expert, mask);
            if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
        }
        if ((expert & 31) == int(lane)) {                        // the winner's lane: every lane holds the same winner
            for (int i = 0; i < 16; ++i)
                if (i == expert >> 5) values[i] = -INFINITY;
            ids[rank] = expert;
            selected_sum = add_ftz(selected_sum, best);          // by winning expert's lane, in selection order
        }
        if (rank == int(lane)) selected = best;
    }
    selected_sum = max_ftz(warp_sum_ftz(selected_sum), 6.103515625e-5f);
    const float inverse_selected_sum = rcp_ftz(selected_sum);
    if (lane < 10) weights[lane] = mul_ftz(selected, inverse_selected_sum);
}
