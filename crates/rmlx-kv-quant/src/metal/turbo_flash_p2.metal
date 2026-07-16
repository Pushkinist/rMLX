
uint n_blocks = params_p2[0];
uint head_dim = params_p2[1];
uint bh_total = params_p2[2]; // B × n_q_heads

uint tid    = thread_position_in_threadgroup.x;
uint bh_idx = threadgroup_position_in_grid.x;
if (bh_idx >= bh_total)
    return;

// ── Step 1: find global max across blocks (single-thread scan) ────────────
// shared_out is sized to the maximum supported head_dim (256). Currently
// the variable is allocated but unread — TG-cooperative merge is a future
// optimization. The size matches the documented intent and keeps the
// P2 kernel valid for head_dim ∈ {128, 256}.
threadgroup float shared_out[256]; // max head_dim supported = 256
threadgroup float g_max_buf[1];
threadgroup float g_sum_buf[1];

if (tid == 0u) {
    float gmax = -INFINITY;
    for (uint b = 0u; b < n_blocks; b++) {
        float bmax = partial_ms[(bh_idx * n_blocks + b) * 2u + 0u];
        if (bmax > gmax)
            gmax = bmax;
    }
    g_max_buf[0] = gmax;

    float gsum = 0.0f;
    for (uint b = 0u; b < n_blocks; b++) {
        float bmax = partial_ms[(bh_idx * n_blocks + b) * 2u + 0u];
        float bsum = partial_ms[(bh_idx * n_blocks + b) * 2u + 1u];
        gsum += exp(bmax - gmax) * bsum;
    }
    g_sum_buf[0] = gsum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float global_max = g_max_buf[0];
float global_sum = g_sum_buf[0];
float inv_sum    = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

// ── Step 2: merge block partial outputs ───────────────────────────────────
if (tid < head_dim) {
    float accum = 0.0f;
    for (uint b = 0u; b < n_blocks; b++) {
        float bmax       = partial_ms[(bh_idx * n_blocks + b) * 2u + 0u];
        float correction = exp(bmax - global_max);
        float bval       = partial_out[(bh_idx * n_blocks + b) * head_dim + tid];
        accum += correction * bval;
    }
    // Normalize.
    dst[bh_idx * head_dim + tid] = accum * inv_sum;
}
