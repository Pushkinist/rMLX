
// ── Thread / threadgroup coordinates ──────────────────────────────────
uint head_dim     = dims[0];
uint kv_seq       = dims[1];
uint n_bh         = dims[2];
uint kv_h         = dims[3];
uint heads_per_kv = dims[4];
uint n_tiles      = dims[5];
uint has_mask     = dims[6];
uint n_groups     = dims[7];

uint tile_idx = threadgroup_position_in_grid.x;
uint bh       = threadgroup_position_in_grid.y;
uint tid      = thread_position_in_threadgroup.x;

if (bh >= n_bh)
    return;
if (tile_idx >= n_tiles)
    return;

uint n_q_heads = kv_h * heads_per_kv;
uint b         = bh / n_q_heads;
uint hq        = bh % n_q_heads;
uint kv_h_idx  = hq / heads_per_kv;

uint tile_start = tile_idx * RF_TILE_SIZE;
uint tile_end   = tile_start + RF_TILE_SIZE;
if (tile_end > kv_seq)
    tile_end = kv_seq;

// ── Load Q once into threadgroup memory ──────────────────────────────
// SMEM ceiling: head_dim <= RF_HEAD_DIM_MAX (dispatcher-enforced). Two
// float[RF_HEAD_DIM_MAX] arrays at the 512 ceiling = 4 KiB, well inside the
// Apple GPU 32 KiB threadgroup-memory budget.
threadgroup float q_shared[RF_HEAD_DIM_MAX];
q_shared[tid] = query[bh * head_dim + tid];

// ── Online softmax broadcast slots ───────────────────────────────────
threadgroup float s_max[1];
threadgroup float s_sum[1];
threadgroup float s_corr[1];
threadgroup float s_expsc[1];

if (tid == 0u) {
    s_max[0] = -INFINITY;
    s_sum[0] = 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Per-thread V accumulator (registers — never spills to threadgroup mem).
float acc_v = 0.0f;

// Shared scratch for the QK dot reduction.
threadgroup float dot_shared[RF_HEAD_DIM_MAX];

for (uint t = tile_start; t < tile_end; t++) {
    // ── Decode K[tid] and V[tid] for this (b, kv_h_idx, t) ───────────
    // BOTH rotor stores are SEQUENCE-major (`[B, S, kv_h, n_groups]`): per
    // token all heads are contiguous, matching the chunk-append layout. Unlike
    // the bf16-V sibling kernel, V here has no head-major mirror — it is read
    // straight from its own packed rotor ring at the same token index as K.
    uint kv_tok = (b * kv_seq + t) * kv_h + kv_h_idx;

    float k_val = rf_decode_k_lane(k_codes, k_scales, k_norms, k_rotors, kv_tok, n_groups, tid);

    // ── QK dot product + tree reduction ─────────────────────────────
    dot_shared[tid] = q_shared[tid] * k_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // REQUIRES head_dim to be a power of two (dispatcher rejects non-pow-2).
    for (uint stride = head_dim >> 1; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            dot_shared[tid] += dot_shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Thread 0: online softmax update + broadcast ──────────────────
    if (tid == 0u) {
        float raw = dot_shared[0] * scale_arr[0];
        // Mask is per (b, q_head, t) — add inside thread 0.
        float mask_val = 0.0f;
        if (has_mask != 0u) {
            mask_val = mask_flat[(b * n_q_heads + hq) * kv_seq + t];
        }
        float score = raw + mask_val;

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

    // ── Quant-V unpack + softmax-weighted accumulation ──────────────
    // The rotor codec is axis-agnostic and its per-lane decode is
    // self-contained: lane `tid` reads only its own group's code word, that
    // group's scale, the group's rotor, and the token's L2 norm. So the same
    // `rf_decode_k_lane` that produced K above produces this lane's V — no
    // cross-lane exchange, no extra barrier, and no bf16 V in device memory.
    float corr = s_corr[0];
    float es   = s_expsc[0];

    float v_val = rf_decode_k_lane(v_codes, v_scales, v_norms, v_rotors, kv_tok, n_groups, tid);

    acc_v = acc_v * corr + es * v_val;
}

// ── Write per-tile partials ──────────────────────────────────────────
uint out_base             = (tile_idx * n_bh + bh) * head_dim;
partial_o[out_base + tid] = acc_v;

if (tid == 0u) {
    uint meta          = tile_idx * n_bh + bh;
    tile_max[meta]     = s_max[0];
    tile_sum_exp[meta] = s_sum[0];
}
