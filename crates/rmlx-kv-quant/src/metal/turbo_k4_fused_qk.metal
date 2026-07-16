
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

    // Per-token layout: D elements per K row =>
    //   groups_per_tok = D / 32
    //   words_per_tok  = D / 8      (4 u32 per group of 32, 8 nibbles per u32)
    //   scales_per_tok = D / 32     (== groups_per_tok)
    uint kv_tok            = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
    uint groups_per_tok    = head_dim / 32u;
    uint words_per_tok     = head_dim / 8u;
    uint scales_per_tok    = groups_per_tok;
    uint codes_tok_off     = kv_tok * words_per_tok;
    uint scales_tok_off    = kv_tok * scales_per_tok;

    // tid handles one head_dim element. Which group and which lane in group?
    uint group_id_in_head = tid / 32u;
    uint elem_in_group    = tid - group_id_in_head * 32u;

    // Each thread loads one Q element into shared memory.  256-slot SMEM
    // covers head_dim in {128, 256}; trailing slots are unused for 128.
    threadgroup float q_shared[256];
    q_shared[tid] = query[bh * head_dim + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4-bit unpack: locate this thread's nibble within the group's 4-word
    // (128-bit) stream.  Element `e` in group lives at word `e/8`, nibble
    // `e%8` (bits [(e%8)*4 .. (e%8)*4 + 4)).  Group base in codes: 4 words.
    uint group_codes_base = codes_tok_off + group_id_in_head * 4u;
    uint word_off         = elem_in_group / 8u;            // 0..3
    uint nibble_in_word   = elem_in_group - word_off * 8u; // 0..7
    uint word             = codes[group_codes_base + word_off];
    uint idx              = (word >> (nibble_in_word * 4u)) & 0xFu;

    // Codebook lookup x per-group scale.  No WHT / Hadamard — turbo4 K
    // is plain codebook * scale (matches DEQUANTIZE_SOURCE in
    // turboquant_msl.rs: out = CB[idx] * scale).
    float k_scale = scales[scales_tok_off + group_id_in_head];
    float k_val   = CB4[idx] * k_scale;

    // QK partial product + threadgroup tree reduction (in-place).
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
