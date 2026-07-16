
    uint group_id = threadgroup_position_in_grid.x;
    uint tid      = thread_position_in_threadgroup.x;  // 0..31

    uint base = group_id * 128u + tid * 4u;

    float x0 = inp[base + 0u];
    float x1 = inp[base + 1u];
    float x2 = inp[base + 2u];
    float x3 = inp[base + 3u];

    float local_max = max(max(abs(x0), abs(x1)), max(abs(x2), abs(x3)));

    threadgroup float partial[32];
    partial[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 16u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            partial[tid] = max(partial[tid], partial[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float abs_max = partial[0];

    float scale = (abs_max > 0.0f) ? (abs_max / 127.0f) : 0.0f;
    if (tid == 0u) {
        scales[group_id] = scale;
    }

    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    int q0 = (int)clamp(rint(x0 * inv_scale), -128.0f, 127.0f);
    int q1 = (int)clamp(rint(x1 * inv_scale), -128.0f, 127.0f);
    int q2 = (int)clamp(rint(x2 * inv_scale), -128.0f, 127.0f);
    int q3 = (int)clamp(rint(x3 * inv_scale), -128.0f, 127.0f);

    uint b0 = (uint)(q0 & 0xFF);
    uint b1 = (uint)(q1 & 0xFF);
    uint b2 = (uint)(q2 & 0xFF);
    uint b3 = (uint)(q3 & 0xFF);

    uint word = b0 | (b1 << 8u) | (b2 << 16u) | (b3 << 24u);
    codes[group_id * 32u + tid] = word;
