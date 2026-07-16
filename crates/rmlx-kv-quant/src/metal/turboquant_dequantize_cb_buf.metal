
    uint gid      = thread_position_in_grid.x;
    uint group_id = gid / 4u;
    uint word_off = gid % 4u;
    uint elem_in_grp = thread_position_in_threadgroup.x;

 // Cache the 16-entry codebook in threadgroup memory (4 threads per group:
 // each loads 4 entries to cover [0..16)). Guard on `elem_in_grp < 4u` so
 // any future threadgroup-size bump cannot OOB-write past cb_shared[15]
 // (mirrors the `elem < 16u` guard pattern in QUANTIZE_CB_BUF_SOURCE).
    threadgroup float cb_shared[16];
    if (elem_in_grp < 4u) {
        uint base = elem_in_grp * 4u;
        cb_shared[base + 0u] = cb[base + 0u];
        cb_shared[base + 1u] = cb[base + 1u];
        cb_shared[base + 2u] = cb[base + 2u];
        cb_shared[base + 3u] = cb[base + 3u];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint word  = codes[gid];
    float scale = scales[group_id];

    uint base_out = group_id * 32u + word_off * 8u;

    for (uint byte_idx = 0u; byte_idx < 4u; byte_idx++) {
        uint b   = (word >> (byte_idx * 8u)) & 0xFFu;
        uint lo  = b & 0xFu;
        uint hi  = b >> 4u;
        float v0 = cb_shared[lo] * scale;
        float v1 = cb_shared[hi] * scale;
        out[base_out + byte_idx * 2u    ] = static_cast<OutT>(v0);
        out[base_out + byte_idx * 2u + 1] = static_cast<OutT>(v1);
    }
