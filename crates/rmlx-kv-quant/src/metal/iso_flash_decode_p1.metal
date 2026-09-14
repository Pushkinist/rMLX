
// ── Thread / threadgroup coordinates ──────────────────────────────────
uint head_dim     = dims[0];
uint kv_seq       = dims[1];
uint n_bh         = dims[2];
uint kv_h         = dims[3];
uint heads_per_kv = dims[4];
uint n_tiles      = dims[5];
uint has_mask     = dims[6];
uint n_groups     = dims[7];
// Words the dense code plane holds per row. Loop-invariant, and the
// per-lane decode below runs once per KV token, so it is derived here
// rather than inside the decode.
uint row_words = cp_row_words(n_groups);
// V's own sequence extent, which may exceed `kv_seq`: the caller hands over the
// whole bf16 mirror allocation rather than a `..kv_seq` slice of it, so that no
// partial-slice view has to be made row-contiguous before dispatch.
uint v_seq_stride = dims[8];

uint tile_idx = threadgroup_position_in_grid.x;
uint bh       = threadgroup_position_in_grid.y;
uint tid      = thread_position_in_threadgroup.x;

// SIMD-group coordinates (Apple GPU: 32 lanes per simdgroup). The threadgroup
// is exactly `head_dim` threads (power-of-two, dispatcher-enforced). The fold
// below uses the runtime `n_simd = simdgroups_per_threadgroup`, which covers the
// head whether or not the last simdgroup is full.
uint simd_lane = thread_index_in_simdgroup;
uint simd_id   = simdgroup_index_in_threadgroup;
uint n_simd    = simdgroups_per_threadgroup;

if (bh >= n_bh)
    return;
if (tile_idx >= n_tiles)
    return;

uint n_q_heads = kv_h * heads_per_kv;
uint b         = bh / n_q_heads;
uint hq        = bh % n_q_heads;
uint kv_h_idx  = hq / heads_per_kv;

uint tile_start = tile_idx * IF_TILE_SIZE;
uint tile_end   = tile_start + IF_TILE_SIZE;
if (tile_end > kv_seq)
    tile_end = kv_seq;

// ── Load this lane's Q into a register ────────────────────────────────
// Each lane only ever multiplies its own head-dim slot, so Q lives in a
// register rather than threadgroup memory.
float q_lane = query[bh * head_dim + tid];

// ── Online-softmax broadcast slots ───────────────────────────────────
// Thread 0 owns the authoritative running (max, sum) in registers across the
// tile and broadcasts the per-token correction / exp-score so every lane can
// rescale its V accumulator.
threadgroup float s_corr[1];
threadgroup float s_expsc[1];

float run_max = -INFINITY;
float run_sum = 0.0f;

// Per-thread V accumulator (registers — never spills to threadgroup mem).
float acc_v = 0.0f;

// Per-simdgroup QK-dot partials (folded by thread 0).
threadgroup float dot_partials[IF_HEAD_DIM_MAX / 32];

for (uint t = tile_start; t < tile_end; t++) {
    // ── Decode K[tid] for this (b, kv_h_idx, t) ──────────────────────
    // The iso K ring is SEQUENCE-major (`[B, S, kv_h, n_groups]`): per token
    // all heads are contiguous, matching the chunk-append layout. (V below
    // stays head-major — it is the separate bf16 mirror, not an iso-packed
    // buffer.) The iso decode is self-contained per lane (one quaternion
    // block fits one u32), so it stays in registers with no threadgroup stage.
    uint kv_tok = (b * kv_seq + t) * kv_h + kv_h_idx;

    float k_val =
        if_decode_k_lane(codes, scales, norms, kv_tok, n_groups, row_words, tid);

    // ── QK dot via simdgroup reduction ───────────────────────────────
    // simd_sum folds each simdgroup's 32 lanes with no threadgroup barrier and
    // no idle-lane tree; one partial per simdgroup then folds on thread 0.
    float prod     = q_lane * k_val;
    float lane_sum = simd_sum(prod);
    if (simd_lane == 0u) {
        dot_partials[simd_id] = lane_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Thread 0: fold partials + online-softmax update + broadcast ───
    if (tid == 0u) {
        float raw = 0.0f;
        for (uint i = 0u; i < n_simd; i++) {
            raw += dot_partials[i];
        }
        raw *= scale_arr[0];
        // Mask is per (b, q_head, t) — add inside thread 0.
        float mask_val = 0.0f;
        if (has_mask != 0u) {
            mask_val = mask_flat[(b * n_q_heads + hq) * kv_seq + t];
        }
        float score = raw + mask_val;

        float old_max = run_max;
        float new_max = (score > old_max) ? score : old_max;
        float corr    = exp(old_max - new_max);
        float es      = exp(score - new_max);

        run_max    = new_max;
        run_sum    = run_sum * corr + es;
        s_corr[0]  = corr;
        s_expsc[0] = es;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── V read + softmax-weighted accumulation ──────────────────────
    // V layout: [B, kv_h, v_seq_stride, head_dim], flat. Read in V's native dtype
    // (bf16 / f16 / f32) — the pointer is auto-typed by mlx-c from the array
    // dtype and MSL promotes to float at the read site, halving V bandwidth
    // vs. an f32 astype upcast.
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
    tile_max[meta]     = run_max;
    tile_sum_exp[meta] = run_sum;
}
