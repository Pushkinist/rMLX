
 // ── Decode params ─────────────────────────────────────────────────────────
    uint B          = params[0];
    uint n_kv_heads = params[1];
    uint n_repeats  = params[2];
    uint T_seq      = params[3];
    uint head_dim   = params[4];
    uint group_size = params[5];
    uint bits       = params[6];
    uint el_per_int = params[7];

 // ── Thread -> (b, rep_h, dim_idx) ────────────────────────────────────────
    uint tid     = thread_position_in_grid.x;
    uint dim_idx = tid % head_dim;
    uint rem     = tid / head_dim;
    uint rep_h   = rem % (n_kv_heads * n_repeats);
    uint b       = rem / (n_kv_heads * n_repeats);

 // kv_head index for GQA: V has n_kv_heads, queries have n_kv_heads * n_repeats.
    uint kv_h = rep_h / n_repeats;  // GQA: rep_h = kv_h * n_repeats + repeat_idx

 // Derived strides.
    uint codes_d  = head_dim / el_per_int;
    uint scales_d = head_dim / group_size;

 // Base offsets for this (b, kv_h, rep_h) block.
    uint probs_base = (b * n_kv_heads * n_repeats + rep_h) * T_seq;
    uint v_base     = (b * n_kv_heads + kv_h) * T_seq;

 // Code extraction: which uint32 word and which shift for dim_idx.
    uint code_mask = (1u << bits) - 1u;
    uint d_word    = dim_idx / el_per_int;
    uint d_shift   = (dim_idx % el_per_int) * bits;

 // Scale/bias group for dim_idx.
    uint scale_idx_in_token = dim_idx / group_size;

 // Midpoint for unsigned-to-signed conversion: 2^(bits-1).
    float midpoint = (float)(1u << (bits - 1u));

 // ── Main loop over T_seq tokens ───────────────────────────────────────────
    float acc = 0.0f;

    for (uint t = 0; t < T_seq; t++) {
        float prob = probs[probs_base + t];

 // SPARSE-V SKIP: negligible attention weights contribute nothing.
 // Skip the V dequant + multiply-accumulate entirely to save bandwidth.
        if (prob < SPARSE_V_EPS) continue;

 // Affine dequant for (kv_h, token t, dim_idx).
        uint token_code_base  = (v_base + t) * codes_d;
        uint token_scale_base = (v_base + t) * scales_d;

        uint  raw        = (v_codes[token_code_base + d_word] >> d_shift) & code_mask;
        float code_float = (float)raw - midpoint;
        float scale      = v_scales[token_scale_base + scale_idx_in_token];
        float bias       = v_biases[token_scale_base + scale_idx_in_token];
        float val        = scale * code_float + bias;

        acc += prob * val;
    }

 // ── Write output ──────────────────────────────────────────────────────────
    uint out_idx = (b * n_kv_heads * n_repeats + rep_h) * head_dim + dim_idx;
    out[out_idx] = acc;
