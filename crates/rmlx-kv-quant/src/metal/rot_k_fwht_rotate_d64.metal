
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;

    const uint D_C       = 64u;
    const uint LOG2_D_C  = 6u;
    const float INV_SQRT_D = 1.0f / sqrt((float)D_C);

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

    out[row * D_C + tid] = buf[tid] * INV_SQRT_D;
