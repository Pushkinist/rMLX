
    // One thread per group (block of GROUP_SIZE=32 elements).
    uint group_id = thread_position_in_grid.x;
    uint base     = group_id * 32u;

    // -- Step 1: load group into registers + compute abs_max -------------
    float xs[32];
    float abs_max = 0.0f;
    for (uint i = 0u; i < 32u; i++) {
        float v = inp[base + i];
        xs[i] = v;
        float a = abs(v);
        if (a > abs_max) { abs_max = a; }
    }
    float scale = (abs_max > 0.0f) ? (abs_max / CB2_MAX_K) : 0.0f;
    scales[group_id] = scale;

    // Zero-scale: emit 2 zero u32 words (all indices = 0).
    if (scale == 0.0f) {
        codes[group_id * 2u + 0u] = 0u;
        codes[group_id * 2u + 1u] = 0u;
        return;
    }

    float inv_scale = 1.0f / scale;

    // -- Step 2: Viterbi forward pass ------------------------------------
    float path_cost[4];
    float next_cost[4];
    uchar back_states[32 * 4];
    uchar back_levels[32 * 4];

    const float INF = as_type<float>(0x7F800000u);  // +inf
    path_cost[0] = 0.0f;
    path_cost[1] = INF;
    path_cost[2] = INF;
    path_cost[3] = INF;

    for (uint t = 0u; t < 32u; t++) {
        float normalised = xs[t] * inv_scale;

        next_cost[0] = INF;
        next_cost[1] = INF;
        next_cost[2] = INF;
        next_cost[3] = INF;

        for (uint s = 0u; s < 4u; s++) {
            float prev = path_cost[s];
            if (!isfinite(prev)) { continue; }
            for (uint lvl = 0u; lvl < 4u; lvl++) {
                uint next_s = ((s << 1) | (lvl & 1u)) & 3u;
                float diff = normalised - CB2[lvl];
                float cand = prev + diff * diff;
                if (cand < next_cost[next_s]) {
                    next_cost[next_s] = cand;
                    back_states[t * 4u + next_s] = (uchar)s;
                    back_levels[t * 4u + next_s] = (uchar)lvl;
                }
            }
        }
        path_cost[0] = next_cost[0];
        path_cost[1] = next_cost[1];
        path_cost[2] = next_cost[2];
        path_cost[3] = next_cost[3];
    }

    // -- Step 3: pick best final state ------------------------------------
    uint best_state = 0u;
    float best_cost = path_cost[0];
    for (uint s = 1u; s < 4u; s++) {
        if (path_cost[s] < best_cost) {
            best_cost = path_cost[s];
            best_state = s;
        }
    }

    // -- Step 4: back-trace, emit indices ----------------------------------
    uchar idxs[32];
    uint cur = best_state;
    for (int ti = 31; ti >= 0; ti--) {
        uint t = (uint)ti;
        uchar lvl = back_levels[t * 4u + cur];
        idxs[t] = lvl;
        cur = (uint)back_states[t * 4u + cur];
    }

    // -- Step 5: pack 32 × 2-bit into 2 × u32, LSB-first ------------------
    // Element e occupies bits [e*2, e*2+2) of the 64-bit per-group stream.
    // 16 elements per u32 word (two words for 32 elements).
    uint w0 = 0u;
    uint w1 = 0u;
    for (uint e = 0u; e < 16u; e++) {
        uint shift = e * 2u;
        w0 |= ((uint)idxs[e] & 0x3u) << shift;
    }
    for (uint e = 0u; e < 16u; e++) {
        uint shift = e * 2u;
        w1 |= ((uint)idxs[16u + e] & 0x3u) << shift;
    }
    codes[group_id * 2u + 0u] = w0;
    codes[group_id * 2u + 1u] = w1;
