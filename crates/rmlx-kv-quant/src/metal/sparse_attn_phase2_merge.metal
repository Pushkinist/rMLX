
    uint head_dim = dims_p2[0];
    uint n_tiles  = dims_p2[1];
    uint n_bh     = dims_p2[2];

    uint tid = thread_position_in_threadgroup.x;
    uint bh  = threadgroup_position_in_grid.x;
    if (bh >= n_bh) return;

    threadgroup float g_max_buf[1];
    threadgroup float g_sum_buf[1];

    if (tid == 0u) {
        float gmax = -INFINITY;
        for (uint t = 0u; t < n_tiles; t++) {
            float tmax = tile_lse[(t * n_bh + bh) * 2u + 0u];
            if (tmax > gmax) gmax = tmax;
        }
        g_max_buf[0] = gmax;

        float gsum = 0.0f;
        for (uint t = 0u; t < n_tiles; t++) {
            float tmax = tile_lse[(t * n_bh + bh) * 2u + 0u];
            float tsum = tile_lse[(t * n_bh + bh) * 2u + 1u];
            gsum += exp(tmax - gmax) * tsum;
        }
        g_sum_buf[0] = gsum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = g_max_buf[0];
    float global_sum = g_sum_buf[0];
    float inv_sum    = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;

    if (tid < head_dim) {
        float accum = 0.0f;
        for (uint t = 0u; t < n_tiles; t++) {
            float tmax = tile_lse[(t * n_bh + bh) * 2u + 0u];
            float corr = exp(tmax - global_max);
            float pv   = partial_o[(t * n_bh + bh) * head_dim + tid];
            accum += corr * pv;
        }
        dst[bh * head_dim + tid] = accum * inv_sum;
    }
