
    uint row  = threadgroup_position_in_grid.x;
    uint tid  = thread_position_in_threadgroup.x;

    const uint D_C           = 64u;
    const uint GROUP_SIZE_C  = 64u;
    const uint LOG2_D_C      = 6u;
    const uint GROUPS_ROW    = 1u;
    const uint WORDS_ROW     = 16u;
    const float INV_SQRT_D   = 1.0f / sqrt((float)D_C);

    threadgroup float buf[64];
    buf[tid] = (float)inp[row * D_C + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 0u; s < LOG2_D_C; s++) {
        uint stride = 1u << s;
        if ((tid & stride) == 0u) {
            uint j = tid + stride;
            float a = buf[tid];
            float b = buf[j];
            buf[tid] = a + b;
            buf[j]   = a - b;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float x_rot = buf[tid] * INV_SQRT_D;
    buf[tid] = x_rot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint g    = tid / GROUP_SIZE_C;
    uint lidg = tid % GROUP_SIZE_C;

    threadgroup float grp_max[1];
    threadgroup float grp_min[1];

    if (lidg == 0u) {
        float gmax = buf[g * GROUP_SIZE_C];
        float gmin = buf[g * GROUP_SIZE_C];
        for (uint k = 1u; k < GROUP_SIZE_C; k++) {
            float v = buf[g * GROUP_SIZE_C + k];
            if (v > gmax) gmax = v;
            if (v < gmin) gmin = v;
        }
        grp_max[g] = gmax;
        grp_min[g] = gmin;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gmax  = grp_max[g];
    float gmin  = grp_min[g];
 // Constant group (all elements equal): MLX mx.quantize produces scale≈0
 // and bias=gmin, yielding code=0 for all elements (dequant = bias).
 // Mirror that convention: scale=0.0 makes the guard below emit code=0.
    float scale = (gmax > gmin) ? ((gmax - gmin) / 255.0f) : 0.0f;
    float bias  = gmin;

    float v_norm = (scale > 0.0f) ? ((x_rot - bias) / scale) : 0.0f;
    uint  code   = (uint)clamp(round(v_norm), 0.0f, 255.0f);

    uint word_idx = row * WORDS_ROW + (tid / 4u);
    uint shift    = (tid % 4u) * 8u;
    atomic_fetch_or_explicit((device atomic_uint*)&out_codes[word_idx],
                             (code & 0xFFu) << shift,
                             memory_order_relaxed);

    if (lidg == 0u) {
        uint sg_idx = row * GROUPS_ROW + g;
        out_scales[sg_idx] = scale;
        out_biases[sg_idx] = bias;
    }
