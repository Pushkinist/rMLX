
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

// SIMD-group coordinates (Apple GPU: 32 lanes per simdgroup). The threadgroup
// is exactly `head_dim` threads and head_dim is a power of two >= 128, so every
// simdgroup is full and `n_simd = head_dim / simd_width` covers the head.
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

uint tile_start = tile_idx * RF_TILE_SIZE;
uint tile_end   = tile_start + RF_TILE_SIZE;
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

// One decode per group: the group leader stages all its lanes' K here so the
// ~64-FMA inverse sandwich runs once per Cl(3,0) block instead of once per lane.
threadgroup float k_shared[RF_HEAD_DIM_MAX];
// Per-simdgroup QK-dot partials (folded by thread 0).
threadgroup float dot_partials[RF_HEAD_DIM_MAX / 32];

// This lane's Cl(3,0) block (each owns RF_GROUP_SIZE consecutive head-dim slots).
uint group_id_in_head = tid / RF_GROUP_SIZE;
uint lane_in_group    = tid - group_id_in_head * RF_GROUP_SIZE;

for (uint t = tile_start; t < tile_end; t++) {
    // ── One decode per group ─────────────────────────────────────────
    // The rotor K store is SEQUENCE-major (`[B, S, kv_h, n_groups]`): per
    // token all heads are contiguous, matching the chunk-append layout.
    // (V below stays head-major — it is the separate bf16 mirror, not a
    // rotor-packed buffer.) Only lane 0 of each group runs the sandwich, then
    // scatters the block's RF_GROUP_SIZE grade-1 lanes into k_shared.
    uint kv_tok = (b * kv_seq + t) * kv_h + kv_h_idx;
    if (lane_in_group == 0u) {
        float kg[RF_GROUP_SIZE];
        rf_decode_k_group(codes, scales, norms, rotors, kv_tok, n_groups,
                          group_id_in_head, kg);
        for (uint e = 0u; e < RF_GROUP_SIZE; ++e) {
            uint slot = tid + e;
            if (slot < head_dim) {
                k_shared[slot] = kg[e];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── QK dot via simdgroup reduction ───────────────────────────────
    // simd_sum folds each simdgroup's 32 lanes with no threadgroup barrier and
    // no idle-lane tree; one partial per simdgroup then folds on thread 0.
    float prod     = q_lane * k_shared[tid];
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
    // V layout: [B, kv_h, kv_seq, head_dim], flat. Read in V's native dtype
    // (bf16 / f16 / f32) — the pointer is auto-typed by mlx-c from the array
    // dtype and MSL promotes to float at the read site, halving V bandwidth
    // vs. an f32 astype upcast.
    float corr = s_corr[0];
    float es   = s_expsc[0];

    uint v_off  = ((b * kv_h + kv_h_idx) * kv_seq + t) * head_dim + tid;
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
