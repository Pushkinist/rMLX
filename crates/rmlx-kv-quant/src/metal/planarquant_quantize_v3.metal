
    uint pair_id   = thread_position_in_grid.x;
    uint group_id  = pair_id / PAIRS_PER_GROUP_C3;
    uint pair_in_g = pair_id % PAIRS_PER_GROUP_C3;

    float a = inp[pair_id * 2u];
    float b = inp[pair_id * 2u + 1u];

 // ── Try all 16 rotations, pick best ──────────────────────────────────────
    float best_err   = 1.0e38f;
    uint  best_rot   = 0u;
    float best_scale = 0.0f;
    uint  best_idx_a = 0u;
    uint  best_idx_b = 0u;

    for (uint k = 0u; k < 16u; k++) {
        float c     = ROT_CB3[k][0];
        float neg_s = ROT_CB3[k][1];
        float sv    = ROT_CB3[k][2];
        float c2    = ROT_CB3[k][3];

        float ya = c * a + neg_s * b;
        float yb = sv * a + c2 * b;

        float abs_max = max(abs(ya), abs(yb));
        float scale   = (abs_max > 0.0f) ? (abs_max / CB_MAX_3) : 0.0f;
        float inv_s   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        float norm_a = ya * inv_s;
        float norm_b = yb * inv_s;

        uint ia = 0u;
        for (uint bi = 0u; bi < 7u; bi++) { if (norm_a > BOUNDARIES3[bi]) ia++; }
        uint ib = 0u;
        for (uint bi = 0u; bi < 7u; bi++) { if (norm_b > BOUNDARIES3[bi]) ib++; }

        float ra  = CB3[ia] * scale;
        float rb  = CB3[ib] * scale;
        float err = max(abs(ya - ra), abs(yb - rb));

        if (err < best_err) {
            best_err   = err;
            best_rot   = k;
            best_scale = scale;
            best_idx_a = ia;
            best_idx_b = ib;
        }
    }

 // ── Write scale ───────────────────────────────────────────────────────────
    scales[pair_id] = best_scale;

 // ── Write rotation index into rot32 (atomic OR) ───────────────────────────
 // Same layout as V4: 16 pairs/group = 2 uint32 words, 4-bit each.
    {
        uint rot_word  = group_id * 2u + (pair_in_g / 8u);
        uint rot_shift = (pair_in_g % 8u) * 4u;
        atomic_fetch_or_explicit((device atomic_uint*)&rot32[rot_word],
                                 (best_rot & 0xFu) << rot_shift,
                                 memory_order_relaxed);
    }

 // ── Write 3-bit code indices (10 vals/u32, atomic OR) ────────────────────
    {
        uint elem_a  = pair_in_g * 2u;
        uint elem_b  = pair_in_g * 2u + 1u;
        uint word_a  = group_id * 4u + (elem_a / 10u);
        uint word_b  = group_id * 4u + (elem_b / 10u);
        uint shift_a = (elem_a % 10u) * 3u;
        uint shift_b = (elem_b % 10u) * 3u;
        atomic_fetch_or_explicit((device atomic_uint*)&codes[word_a],
                                 (best_idx_a & 0x7u) << shift_a,
                                 memory_order_relaxed);
        atomic_fetch_or_explicit((device atomic_uint*)&codes[word_b],
                                 (best_idx_b & 0x7u) << shift_b,
                                 memory_order_relaxed);
    }
