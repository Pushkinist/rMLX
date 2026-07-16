
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
 // Lloyd-Max N(0,1) 3-bit max centroid.
        const float cb3_max = 2.1519449f;
        group_scale[0] = (abs_max > 0.0f) ? (abs_max / cb3_max) : 0.0f;
        scales[group_id] = group_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = group_scale[0];

 // -- Step 3: each thread finds its nearest-centroid index --------------
    float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    float normalized = shared_x[elem] * inv_scale;

    uint idx = 0u;
    for (uint b = 0u; b < 7u; b++) {
        if (normalized > BOUNDARIES_3[b]) {
            idx++;
        }
    }

 // -- Step 4: pack 32 x 3-bit indices into 3 x u32 words ----------------
 // Element e occupies bits [e*3, e*3+3) of the concatenated 96-bit
 // little-endian stream. Word w (w in 0..3) carries bits [w*32, w*32+32).
 // We use 3 writer threads (elem 0, 11, 22) — each scans the 32 element
 // indices, computes signed bit-shift offsets, and ORs the contribution
 // into a 64-bit accumulator.
    threadgroup uint idx_shared[32];
    idx_shared[elem] = idx & 0x7u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool is_writer = (elem == 0u) || (elem == 11u) || (elem == 22u);
    if (is_writer) {
        uint word_off;
        if (elem == 0u)       { word_off = 0u; }
        else if (elem == 11u) { word_off = 1u; }
        else                  { word_off = 2u; }

 // Accumulate into a 64-bit local register so a 3-bit index that
 // straddles the 32-bit word boundary is captured exactly.
        ulong acc = 0ul;
        for (uint e = 0u; e < 32u; e++) {
            int shift = (int)(e * 3u) - (int)(word_off * 32u);
 // Element e's 3-bit window is [shift, shift+3) in word coords.
 // It touches this word iff that window overlaps [0, 32), i.e.
 // shift in [-2, 31].
            if (shift > -3 && shift < 32) {
                ulong bits3 = (ulong)(idx_shared[e] & 0x7u);
                if (shift >= 0) {
                    acc |= (bits3 << (uint)shift);
                } else {
 // shift in {-1, -2}: the low (3 + shift) bits of the
 // element spill into this word, starting at bit 0.
                    acc |= (bits3 >> (uint)(-shift));
                }
            }
        }
        codes[group_id * 3u + word_off] = (uint)(acc & 0xFFFFFFFFul);
    }
