
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

    uint tile_start = tile_idx * SA_TILE_SIZE;
    uint tile_end   = tile_start + SA_TILE_SIZE;
    if (tile_end > kv_seq) tile_end = kv_seq;

    uint codes_words_per_tok    = (head_dim / 32u) * 4u;
    uint scales_pairs_per_tok   = head_dim / 2u;
    uint rot_words_per_tok      = head_dim / 16u;

    uint group_id_in_head = tid / 32u;
    uint elem_in_group    = tid % 32u;
    uint pair_in_group    = elem_in_group / 2u;
    uint elem_in_pair     = elem_in_group & 1u;

    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];

    threadgroup float k_pre_rot[256];
    threadgroup float dot_shared[256];

    threadgroup float tops[SA_TOP_PER_TILE];
    threadgroup uint  tops_idx[SA_TOP_PER_TILE];
    if (tid < SA_TOP_PER_TILE) {
        tops[tid]     = -INFINITY;
        tops_idx[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = tile_start; t < tile_end; t++) {
        // K packing is SEQUENCE-major (`[B, S, kv_h, D]`): per token all heads
        // are contiguous, matching QuantPlanarK's chunk-append layout. A
        // head-major base would scramble heads↔seq after a multi-token append.
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

        k_pre_rot[tid] = SA_CB[cb_idx] * scale_pair;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint pair_base = tid - elem_in_pair;
        float ya = k_pre_rot[pair_base];
        float yb = k_pre_rot[pair_base + 1u];

        float c     = SA_ROT_CB[rot_idx][0];
        float neg_s = SA_ROT_CB[rot_idx][1];
        float sv    = SA_ROT_CB[rot_idx][2];
        float c2    = SA_ROT_CB[rot_idx][3];

        float k_val = (elem_in_pair == 0u)
            ? (c * ya + sv * yb)
            : (neg_s * ya + c2 * yb);

        dot_shared[tid] = q_shared[tid] * k_val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {
            if (tid < stride) {
                dot_shared[tid] += dot_shared[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0u) {
            float score = dot_shared[0] * scale_arr[0];

            all_scores[bh * kv_seq + t] = score;

            if (score > tops[SA_TOP_PER_TILE - 1u]) {
                tops[SA_TOP_PER_TILE - 1u]     = score;
                tops_idx[SA_TOP_PER_TILE - 1u] = t;
                for (int i = (int)SA_TOP_PER_TILE - 2; i >= 0; i--) {
                    if (tops[i + 1] > tops[i]) {
                        float tmp_s = tops[i];
                        tops[i]     = tops[i + 1];
                        tops[i + 1] = tmp_s;
                        uint tmp_i  = tops_idx[i];
                        tops_idx[i]     = tops_idx[i + 1];
                        tops_idx[i + 1] = tmp_i;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < SA_TOP_PER_TILE) {
        uint base = (tile_idx * n_bh + bh) * SA_TOP_PER_TILE + tid;
        tile_top_scores[base]  = tops[tid];
        tile_top_indices[base] = tops_idx[tid];
    }
