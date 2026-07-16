
    uint s_kv         = threadgroup_position_in_grid.x;
    uint bh           = threadgroup_position_in_grid.y;
    uint tid          = thread_position_in_threadgroup.x;
    uint head_dim     = dims[0];
    uint kv_seq       = dims[1];
    uint kv_h         = dims[2];
    uint heads_per_kv = dims[3];
    uint has_mask     = dims[4];

    uint n_q_heads = kv_h * heads_per_kv;
    uint b         = bh / n_q_heads;
    uint hq        = bh % n_q_heads;
    uint kv_h_idx  = hq / heads_per_kv;

    uint kv_tok                = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
    uint codes_words_per_tok   = head_dim / 4u;
    uint scales_groups_per_tok = head_dim / 128u;
    uint codes_tok_off         = kv_tok * codes_words_per_tok;
    uint scales_tok_off        = kv_tok * scales_groups_per_tok;

    uint group_id_in_head = tid / 128u;
    uint elem_in_group    = tid % 128u;
    uint word_in_group    = elem_in_group / 4u;
    uint byte_in_word     = elem_in_group & 3u;

    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint code_word_abs = codes_tok_off + group_id_in_head * 32u + word_in_group;
    uint scale_abs     = scales_tok_off + group_id_in_head;

    uint word     = codes[code_word_abs];
    uint raw_byte = (word >> (byte_in_word * 8u)) & 0xFFu;
    int code      = (int)raw_byte;
    if (code & 0x80) { code -= 256; }

    float k_scale = scales[scale_abs];
    float k_val   = k_scale * (float)code;

    threadgroup float dot_shared[256];
    dot_shared[tid] = q_shared[tid] * k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            dot_shared[tid] += dot_shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float result = dot_shared[0] * scale_arr[0];
        if (has_mask != 0u) {
            result += mask[bh * kv_seq + s_kv];
        }
        out[bh * kv_seq + s_kv] = result;
    }
