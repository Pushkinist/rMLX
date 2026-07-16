
    uint group_id = threadgroup_position_in_grid.x;
    uint elem     = thread_position_in_threadgroup.x;

 // -- Step 1: load group into threadgroup shared memory -----------------
    threadgroup float shared_x[32];
    shared_x[elem] = inp[group_id * 32u + elem];
    threadgroup_barrier(mem_flags::mem_threadgroup);

 // -- Step 2: thread 0 finds max(|x|) and writes scale ------------------
 // Sequential scan by thread 0 mirrors the CPU path exactly (CPU-parity).
    threadgroup float group_scale[1];
    if (elem == 0u) {
        float abs_max = 0.0f;
        for (uint i = 0u; i < 32u; i++) {
            float a = abs(shared_x[i]);
            if (a > abs_max) { abs_max = a; }
        }
 // Lloyd-Max N(0,1) 2-bit max centroid.
        const float cb2_max = 1.51f;
        group_scale[0] = (abs_max > 0.0f) ? (abs_max / cb2_max) : 0.0f;
        scales[group_id] = group_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = group_scale[0];

 // -- Step 3: each thread finds its nearest-centroid index --------------
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    float normalized = shared_x[elem] * inv_scale;

    uint idx = 0u;
    for (uint b = 0u; b < 3u; b++) {
        if (normalized > BOUNDARIES_2[b]) {
            idx++;
        }
    }

 // -- Step 4: pack 32 x 2-bit indices into 2 x u32 words ----------------
 // Element e occupies bits [e*2, e*2+2) of the concatenated 64-bit
 // little-endian stream. Word w (w in 0..2) carries bits [w*32, w*32+32).
 // We use 2 writer threads (elem 0 and elem 16) — each scans the 32
 // element indices and ORs the contribution into a 32-bit accumulator.
 // Unlike the V3 path the 2-bit packing is word-aligned at every 16th
 // element, so the accumulator never needs the cross-word straddle case.
 // We keep the signed-shift accumulator pattern for code-shape uniformity
 // with `k8vturbo3_append_msl.rs`.
    threadgroup uint idx_shared[32];
    idx_shared[elem] = idx & 0x3u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool is_writer = (elem == 0u) || (elem == 16u);
    if (is_writer) {
        uint word_off = (elem == 0u) ? 0u : 1u;

 // Accumulate into a 64-bit local register so the loop body matches
 // the V3 cross-boundary form (even though V2 never straddles).
        ulong acc = 0ul;
        for (uint e = 0u; e < 32u; e++) {
            int shift = (int)(e * 2u) - (int)(word_off * 32u);
 // Element e's 2-bit window is [shift, shift+2) in word coords.
 // It touches this word iff shift in [-1, 31].
            if (shift > -2 && shift < 32) {
                ulong bits2 = (ulong)(idx_shared[e] & 0x3u);
                if (shift >= 0) {
                    acc |= (bits2 << (uint)shift);
                } else {
 // shift == -1: low (2 + shift) = 1 bit of the element spills
 // into this word at bit 0. V2 is word-aligned every 16 elems
 // so this branch is structurally dead but kept for parity
 // with V3.
                    acc |= (bits2 >> (uint)(-shift));
                }
            }
        }
        codes[group_id * 2u + word_off] = (uint)(acc & 0xFFFFFFFFul);
    }
