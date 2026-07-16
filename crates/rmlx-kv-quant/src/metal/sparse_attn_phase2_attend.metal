
    uint head_dim     = dims[0];
    uint kv_seq       = dims[1];
    uint n_bh         = dims[2];
    uint kv_h         = dims[3];
    uint heads_per_kv = dims[4];
    uint n_tiles      = dims[5];

    uint tile_idx = threadgroup_position_in_grid.x;
    uint bh       = threadgroup_position_in_grid.y;
    uint tid      = thread_position_in_threadgroup.x;

    if (bh >= n_bh)         return;
    if (tile_idx >= n_tiles) return;

    uint n_q_heads = kv_h * heads_per_kv;
    uint b         = bh / n_q_heads;
    uint hq        = bh % n_q_heads;
    uint kv_h_idx  = hq / heads_per_kv;

    uint tile_start = tile_idx * SA2_TILE_SIZE;
    uint tile_end   = tile_start + SA2_TILE_SIZE;
    if (tile_end > kv_seq) tile_end = kv_seq;

    float thr = head_threshold[bh];

    // ── Tile-level early-exit ────────────────────────────────────────────
    // Thread 0 scans, broadcasts via threadgroup memory.
    threadgroup bool tile_has_survivors[1];
    if (tid == 0u) {
        tile_has_survivors[0] = false;
        for (uint t = tile_start; t < tile_end; t++) {
            if (all_scores[bh * kv_seq + t] >= thr) {
                tile_has_survivors[0] = true;
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!tile_has_survivors[0]) {
        // Sentinel LSE + zero partial_o slice.
        uint out_base = (tile_idx * n_bh + bh) * head_dim;
        partial_o[out_base + tid] = 0.0f;
        if (tid == 0u) {
            uint meta = (tile_idx * n_bh + bh) * 2u;
            tile_lse[meta + 0u] = -INFINITY;
            tile_lse[meta + 1u] = 0.0f;
        }
        return;
    }

    // Per-token K layout.
    uint codes_words_per_tok    = (head_dim / 32u) * 4u;
    uint scales_pairs_per_tok   = head_dim / 2u;
    uint rot_words_per_tok      = head_dim / 16u;

    uint group_id_in_head = tid / 32u;
    uint elem_in_group    = tid % 32u;
    uint pair_in_group    = elem_in_group / 2u;
    uint elem_in_pair     = elem_in_group & 1u;

    // ── Load Q ───────────────────────────────────────────────────────────
    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];

    threadgroup float k_pre_rot[256];
    threadgroup float dot_shared[256];

    // Online softmax broadcast slots.
    threadgroup float s_max[1];
    threadgroup float s_sum[1];
    threadgroup float s_corr[1];
    threadgroup float s_expsc[1];
    if (tid == 0u) {
        s_max[0] = -INFINITY;
        s_sum[0] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc_v = 0.0f;

    for (uint t = tile_start; t < tile_end; t++) {
        // Skip below-threshold tokens entirely.
        float pre_score = all_scores[bh * kv_seq + t];
        if (pre_score < thr) {
            continue;
        }

        // ── Decode K[tid] for this (b, kv_h_idx, t) ──────────────────────
        // K packing is SEQUENCE-major (`[B, S, kv_h, D]`): per token all heads
        // are contiguous, matching QuantPlanarK's chunk-append layout. (V below
        // stays head-major — it is the bf16 mirror, not the planar-packed buffer.)
        uint kv_tok          = (b * kv_seq + t) * kv_h + kv_h_idx;
        uint codes_tok_off   = kv_tok * codes_words_per_tok;
        uint scales_tok_off  = kv_tok * scales_pairs_per_tok;
        uint rot_tok_off     = kv_tok * rot_words_per_tok;

        uint code_word_in_group = elem_in_group / 8u;
        uint code_word_abs      = codes_tok_off + group_id_in_head * 4u + code_word_in_group;
        uint cb_idx             = (k_codes[code_word_abs] >> ((elem_in_group & 7u) * 4u)) & 0xFu;

        uint pair_global = group_id_in_head * 16u + pair_in_group;
        float scale_pair = k_scales[scales_tok_off + pair_global];

        uint rot_word_in_group = pair_in_group / 8u;
        uint rot_word_abs      = rot_tok_off + group_id_in_head * 2u + rot_word_in_group;
        uint rot_shift         = (pair_in_group & 7u) * 4u;
        uint rot_idx           = (k_rot32[rot_word_abs] >> rot_shift) & 0xFu;

        k_pre_rot[tid] = SA2_CB[cb_idx] * scale_pair;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint pair_base = tid - elem_in_pair;
        float ya = k_pre_rot[pair_base];
        float yb = k_pre_rot[pair_base + 1u];

        float c     = SA2_ROT_CB[rot_idx][0];
        float neg_s = SA2_ROT_CB[rot_idx][1];
        float sv    = SA2_ROT_CB[rot_idx][2];
        float c2    = SA2_ROT_CB[rot_idx][3];

        float k_val = (elem_in_pair == 0u)
            ? (c * ya + sv * yb)
            : (neg_s * ya + c2 * yb);

        // QK dot product + tree reduction.
        dot_shared[tid] = q_shared[tid] * k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                dot_shared[tid] += dot_shared[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ── Online softmax update (thread 0 broadcasts) ─────────────────
        if (tid == 0u) {
            float score = dot_shared[0] * scale_arr[0];

            float old_max = s_max[0];
            float new_max = (score > old_max) ? score : old_max;
            float corr    = exp(old_max - new_max);
            float es      = exp(score - new_max);

            s_max[0]   = new_max;
            s_sum[0]   = s_sum[0] * corr + es;
            s_corr[0]  = corr;
            s_expsc[0] = es;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── V read + softmax-weighted accumulation ──────────────────────
        float corr = s_corr[0];
        float es   = s_expsc[0];

        uint v_off = ((b * kv_h + kv_h_idx) * kv_seq + t) * head_dim + tid;
        float v_val = v_flat[v_off];

        acc_v = acc_v * corr + es * v_val;
    }

    // ── Write per-tile partials + LSE ────────────────────────────────────
    uint out_base = (tile_idx * n_bh + bh) * head_dim;
    partial_o[out_base + tid] = acc_v;

    if (tid == 0u) {
        uint meta = (tile_idx * n_bh + bh) * 2u;
        tile_lse[meta + 0u] = s_max[0];
        tile_lse[meta + 1u] = s_sum[0];
    }
