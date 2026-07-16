
    uint pair_id   = thread_position_in_grid.x;
    uint group_id  = pair_id / PAIRS_PER_GROUP_C;
    uint pair_in_g = pair_id % PAIRS_PER_GROUP_C;

    float a = inp[pair_id * 2u];
    float b = inp[pair_id * 2u + 1u];

 // ── Try all 16 rotations, pick best ──────────────────────────────────────
    float best_err   = 1.0e38f;
    uint  best_rot   = 0u;
    float best_scale = 0.0f;
    uint  best_idx_a = 0u;
    uint  best_idx_b = 0u;

    for (uint k = 0u; k < 16u; k++) {
        float c     = ROT_CB[k][0];
        float neg_s = ROT_CB[k][1];
        float sv    = ROT_CB[k][2];
        float c2    = ROT_CB[k][3];

        float ya = c * a + neg_s * b;
        float yb = sv * a + c2 * b;

        float abs_max = max(abs(ya), abs(yb));
        float scale   = (abs_max > 0.0f) ? (abs_max / CB_MAX) : 0.0f;
        float inv_s   = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

        float norm_a = ya * inv_s;
        float norm_b = yb * inv_s;

        uint ia = 0u;
        for (uint bi = 0u; bi < 15u; bi++) { if (norm_a > BOUNDARIES[bi]) ia++; }
        uint ib = 0u;
        for (uint bi = 0u; bi < 15u; bi++) { if (norm_b > BOUNDARIES[bi]) ib++; }

        float ra  = CB[ia] * scale;
        float rb  = CB[ib] * scale;
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
 // rot32: 16 pairs/group = 64 bits = 2 uint32 words per group.
 // pair_in_g 0..7 → word 0 bits [0:31] (4-bit each).
 // pair_in_g 8..15 → word 1 bits [0:31].
    {
        uint rot_word  = group_id * 2u + (pair_in_g / 8u);
        uint rot_shift = (pair_in_g % 8u) * 4u;
        atomic_fetch_or_explicit((device atomic_uint*)&rot32[rot_word],
                                 (best_rot & 0xFu) << rot_shift,
                                 memory_order_relaxed);
    }

 // ── Write code indices into codes (atomic OR) ─────────────────────────────
 // codes: 32 elements/group = 4 uint32 words (8 4-bit indices each).
 // Element e = pair_in_g*2 or pair_in_g*2+1.
    {
        uint elem_a  = pair_in_g * 2u;
        uint elem_b  = pair_in_g * 2u + 1u;
        uint word_a  = group_id * 4u + (elem_a / 8u);
        uint word_b  = group_id * 4u + (elem_b / 8u);
        uint shift_a = (elem_a % 8u) * 4u;
        uint shift_b = (elem_b % 8u) * 4u;
        atomic_fetch_or_explicit((device atomic_uint*)&codes[word_a],
                                 (best_idx_a & 0xFu) << shift_a,
                                 memory_order_relaxed);
        atomic_fetch_or_explicit((device atomic_uint*)&codes[word_b],
                                 (best_idx_b & 0xFu) << shift_b,
                                 memory_order_relaxed);
    }
