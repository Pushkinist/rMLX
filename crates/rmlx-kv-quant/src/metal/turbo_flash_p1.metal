
 // ── Decode params ─────────────────────────────────────────────────────────
    uint B             = params_p1[0];
    uint n_q_heads     = params_p1[1];
    uint n_kv_heads    = params_p1[2];
    uint n_repeats     = params_p1[3];
    uint T_active      = params_p1[4];   // iteration bound + mask length
    uint head_dim      = params_p1[5];
    uint n_blocks      = params_p1[6];
    uint has_mask      = params_p1[7];
    uint q8_words_per_tok  = params_p1[8];   // head_dim / 4
    uint tq4_words_per_tok = params_p1[9];   // head_dim / 8
    uint T_stride      = params_p1[10];  // per-head row stride for K/V buffers

 // ── Thread identity ───────────────────────────────────────────────────────
    uint lane     = thread_position_in_threadgroup.x;  // 0..31
    uint bh_idx   = threadgroup_position_in_grid.x;    // flat (b, q_head) index
    uint block_id = threadgroup_position_in_grid.y;    // 0..n_blocks-1

    if (bh_idx >= B * n_q_heads) return;
    if (block_id >= n_blocks)    return;

    uint b      = bh_idx / n_q_heads;
    uint q_head = bh_idx % n_q_heads;
    uint kv_head = q_head / n_repeats;  // GQA: n_q_heads = n_kv_heads * n_repeats

 // ── Token range for this block ────────────────────────────────────────────
    uint t_start = block_id * BLOCK_SIZE;
    uint t_end   = min(t_start + BLOCK_SIZE, T_active);

 // ── Load Q into registers (interleaved: lane handles dims lane, lane+32, ...) ──
 // dims_per_lane ∈ {4, 8} for head_dim ∈ {128, 256}. Register arrays are
 // sized to MAX_DIMS_PER_LANE=8 so the same kernel handles both dims.
    uint dims_per_lane = head_dim / TG_SIZE;  // 128/32=4, 256/32=8
    uint q_base = (b * n_q_heads + q_head) * head_dim;
    float q_vals[8];  // max dims_per_lane = head_dim/32 = 256/32 = 8
    for (uint i = 0; i < dims_per_lane; i++) {
        uint d = lane + i * TG_SIZE;
        q_vals[i] = (d < head_dim) ? q_flat[q_base + d] : 0.0f;
    }

 // ── Online softmax state — registers only ─────────────────────────────────
    float m_state = -INFINITY;
    float l_state = 0.0f;
    float o_state[8];  // same max size as dims_per_lane
    for (uint i = 0; i < dims_per_lane; i++) {
        o_state[i] = 0.0f;
    }

 // ── KV base offsets for this (b, kv_head) ────────────────────────────────
    uint kv_tok_stride_k_codes  = q8_words_per_tok;
    uint kv_tok_stride_k_scales = head_dim / Q8_GROUP;  // scales per token
    uint kv_tok_stride_v_codes  = tq4_words_per_tok;
    uint kv_tok_stride_v_scales = head_dim / TQ4_GROUP;

    uint kv_base_codes    = (b * n_kv_heads + kv_head) * T_stride * kv_tok_stride_k_codes;
    uint kv_base_scales   = (b * n_kv_heads + kv_head) * T_stride * kv_tok_stride_k_scales;
    uint kv_base_v_codes  = (b * n_kv_heads + kv_head) * T_stride * kv_tok_stride_v_codes;
    uint kv_base_v_scales = (b * n_kv_heads + kv_head) * T_stride * kv_tok_stride_v_scales;

 // ── Mask base offset (q_head index, not kv_head — mask is per query) ─────
 // Mask length is T_active (one entry per valid KV token), not T_stride.
    uint mask_base = has_mask ? ((b * n_q_heads + q_head) * T_active) : 0u;

 // ── Process tokens in this block ──────────────────────────────────────────
    for (uint t = t_start; t < t_end; t++) {

 // Check causal mask.
        float mask_val = 0.0f;
        if (has_mask) {
            mask_val = mask_flat[mask_base + t];
            if (mask_val <= -1e9f) continue;  // fully masked — skip
        }

 // ── Dequant K (rMLX q8_0) and compute Q·K ────────────────────────────
 // q8_0 layout: i8 codes packed 4/u32, f32 scale per Q8_GROUP=128 elements.
 // codes_word = k_codes[tok_base + d/4]
 // byte = (codes_word >> ((d%4)*8)) & 0xFF (little-endian i8)
 // val = float(int8_t(byte)) * scale[group]
        float dot_partial = 0.0f;

        uint tok_k_codes_base  = kv_base_codes + t * kv_tok_stride_k_codes;
        uint tok_k_scales_base = kv_base_scales + t * kv_tok_stride_k_scales;

        for (uint i = 0; i < dims_per_lane; i++) {
            uint d = lane + i * TG_SIZE;
            if (d >= head_dim) break;

            uint word_idx = d / 4u;
            uint byte_shift = (d % 4u) * 8u;
            uint raw_byte = (k_codes[tok_k_codes_base + word_idx] >> byte_shift) & 0xFFu;
 // Sign-extend to i8: if bit7 set, the byte is negative.
            int scode = (int)(raw_byte);
            if (scode >= 128) scode -= 256;
            float kval = (float)scode * k_scales[tok_k_scales_base + d / Q8_GROUP];

            dot_partial += q_vals[i] * kval;
        }

 // Reduce across SIMD lanes.
        float score = simd_sum(dot_partial) + mask_val;

 // ── Dequant V (rMLX turbo4 Lloyd-Max) ────────────────────────────────
 // turbo4 layout: 4-bit codes 8/u32, f32 scale per TQ4_GROUP=32 elements.
 // word = v_codes[tok_base + d/8]
 // nibble = (word >> ((d%8)*4)) & 0xF — unsigned 4-bit index
 // val = TURBO_CB[nibble] * scale[d/32]
        uint tok_v_codes_base  = kv_base_v_codes + t * kv_tok_stride_v_codes;
        uint tok_v_scales_base = kv_base_v_scales + t * kv_tok_stride_v_scales;

        float v_decoded[8];  // max dims_per_lane = head_dim/32 = 256/32 = 8
        for (uint i = 0; i < dims_per_lane; i++) {
            uint d = lane + i * TG_SIZE;
            if (d >= head_dim) {
                v_decoded[i] = 0.0f;
                continue;
            }
            uint word_idx = d / 8u;
            uint nib_shift = (d % 8u) * 4u;
            uint nibble = (v_codes[tok_v_codes_base + word_idx] >> nib_shift) & 0xFu;
            v_decoded[i] = TURBO_CB[nibble] * v_scales[tok_v_scales_base + d / TQ4_GROUP];
        }

 // ── Online softmax update + V accumulation ────────────────────────────
        float new_m    = max(m_state, score);
        float exp_diff  = exp(m_state - new_m);
        float exp_score = exp(score - new_m);

        for (uint i = 0; i < dims_per_lane; i++) {
            o_state[i] = o_state[i] * exp_diff + exp_score * v_decoded[i];
        }
        l_state = l_state * exp_diff + exp_score;
        m_state = new_m;
    }

 // ── Write partial results ─────────────────────────────────────────────────
 // partial_out: flat [B × n_q_heads × n_blocks × head_dim]
 // partial_ms: flat [B × n_q_heads × n_blocks × 2]
    uint out_block_base = (bh_idx * n_blocks + block_id) * head_dim;
    for (uint i = 0; i < dims_per_lane; i++) {
        uint d = lane + i * TG_SIZE;
        if (d < head_dim) {
            partial_out[out_block_base + d] = o_state[i];
        }
    }

    uint ms_base = (bh_idx * n_blocks + block_id) * 2u;
    if (lane == 0u) {
        partial_ms[ms_base + 0u] = m_state;
        partial_ms[ms_base + 1u] = l_state;
    }
