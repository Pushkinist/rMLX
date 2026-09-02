
// ── Thread / threadgroup coordinates ──────────────────────────────────
uint head_dim     = dims[0];
uint kv_seq       = dims[1];
uint n_bh         = dims[2];
uint kv_h         = dims[3];
uint heads_per_kv = dims[4];
uint n_tiles      = dims[5];
uint has_mask     = dims[6];
// V's own sequence extent, which may exceed `kv_seq`: the caller hands over the
// whole bf16 mirror allocation rather than a `..kv_seq` slice of it, so that no
// partial-slice view has to be made row-contiguous before dispatch.
uint v_seq_stride = dims[7];

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

uint tile_start = tile_idx * PF_TILE_SIZE;
uint tile_end   = tile_start + PF_TILE_SIZE;
if (tile_end > kv_seq)
    tile_end = kv_seq;

// Per-token K layout (matches fused-QK storage contract bit-exact).
uint codes_words_per_tok  = (head_dim / 32u) * 4u;
uint scales_pairs_per_tok = head_dim / 2u;
uint rot_words_per_tok    = head_dim / 16u;

// Group bookkeeping (32 elements per group; 16 pairs per group).
uint group_id_in_head = tid / 32u;
uint elem_in_group    = tid % 32u;
uint pair_in_group    = elem_in_group / 2u;
uint elem_in_pair     = elem_in_group & 1u;

// ── Load Q once into threadgroup memory ──────────────────────────────
// SMEM ceiling: head_dim in 128 or 256 (dispatcher-enforced via
// PLANAR_FLASH_HEAD_DIM_MAX); 256 floats * 4 B = 1 KiB per buffer keeps
// three threadgroup arrays at 3 KiB total, well within the Apple GPU
// 32 KiB threadgroup-memory budget and tight enough not to gate
// occupancy on the head_dim=128 hot path.
threadgroup float q_shared[256];
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

// Shared scratch for K decode (pre- and post-Givens) and the dot
// reduction.  Sized to the head_dim ceiling — see q_shared comment above.
threadgroup float k_pre_rot[256];
threadgroup float dot_shared[256];

for (uint t = tile_start; t < tile_end; t++) {
    // ── Decode K[tid] for this (b, kv_h_idx, t) ──────────────────────
    // K packing is SEQUENCE-major (`[B, S, kv_h, D]`): per token all heads
    // are contiguous, matching QuantPlanarK's chunk-append layout. (V below
    // stays head-major — it is the separate bf16 decode mirror, not the
    // planar-packed buffer.)
    uint kv_tok         = (b * kv_seq + t) * kv_h + kv_h_idx;
    uint codes_tok_off  = kv_tok * codes_words_per_tok;
    uint scales_tok_off = kv_tok * scales_pairs_per_tok;
    uint rot_tok_off    = kv_tok * rot_words_per_tok;

    // Code word + nibble.
    uint code_word_in_group = elem_in_group / 8u;
    uint code_word_abs      = codes_tok_off + group_id_in_head * 4u + code_word_in_group;
    uint cb_idx             = (k_codes[code_word_abs] >> ((elem_in_group & 7u) * 4u)) & 0xFu;

    // Per-pair scale.
    uint pair_global = group_id_in_head * 16u + pair_in_group;
    float scale_pair = k_scales[scales_tok_off + pair_global];

    // Per-pair Givens rotation index (4-bit, 8 pairs per u32, 2 words per group).
    uint rot_word_in_group = pair_in_group / 8u;
    uint rot_word_abs      = rot_tok_off + group_id_in_head * 2u + rot_word_in_group;
    uint rot_shift         = (pair_in_group & 7u) * 4u;
    uint rot_idx           = (k_rot32[rot_word_abs] >> rot_shift) & 0xFu;

    // Pre-rotation: centroid x per-pair scale.
    k_pre_rot[tid] = PF_CB[cb_idx] * scale_pair;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Apply R^T to (y_a, y_b) — matches planar_fused_qk exactly.
    uint pair_base = tid - elem_in_pair;
    float ya       = k_pre_rot[pair_base];
    float yb       = k_pre_rot[pair_base + 1u];

    float c     = PF_ROT_CB[rot_idx][0];
    float neg_s = PF_ROT_CB[rot_idx][1];
    float sv    = PF_ROT_CB[rot_idx][2];
    float c2    = PF_ROT_CB[rot_idx][3];

    float k_val = (elem_in_pair == 0u)
                      ? (c * ya + sv * yb)
                      : (neg_s * ya + c2 * yb);

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

    // ── V read + softmax-weighted accumulation ──────────────────────
    // V layout: [B, kv_h, v_seq_stride, head_dim], flat.
    float corr = s_corr[0];
    float es   = s_expsc[0];

    uint v_off  = ((b * kv_h + kv_h_idx) * v_seq_stride + t) * head_dim + tid;
    float v_val = v_flat[v_off];

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
