
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
//   n_groups_tok    = D / 4  (one quaternion block per group)
//   codes_per_tok   = n_groups_tok (one u32 word per group of 4)
//   scales_per_tok  = n_groups_tok (one f32 per group)
uint kv_tok         = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
uint n_groups_tok   = head_dim / 4u;
uint codes_tok_off  = kv_tok * n_groups_tok;
uint scales_tok_off = kv_tok * n_groups_tok;

// tid handles one head_dim element.  Which group and which lane in group?
uint group_id_in_head = tid / 4u;
uint elem_in_group    = tid - group_id_in_head * 4u;

// Phase 1: load Q, unpack idx, codebook lookup x scale -> rots SMEM
threadgroup float q_shared[256];
threadgroup float rots_shared[256];
q_shared[tid] = query[bh * head_dim + tid];

uint word  = codes[codes_tok_off + group_id_in_head];
uint shift = elem_in_group * 3u;
uint idx   = (word >> shift) & 0x7u;

float k_scale    = scales[scales_tok_off + group_id_in_head];
rots_shared[tid] = ISO_CB[idx] * k_scale;
threadgroup_barrier(mem_flags::mem_threadgroup);

// Phase 2: inverse Hamilton product r' = qbar * r, then x per-token norm.
//
// qbar = (ISO_QW, ISO_QCX, ISO_QCY, ISO_QCZ).  Each thread is one lane of
// a 4-element group; read all 4 rotated centroids from SMEM and compute
// the inverse-rotated lane belonging to this thread.
uint grp_base = group_id_in_head * 4u;
float rw      = rots_shared[grp_base + 0u];
float rx      = rots_shared[grp_base + 1u];
float ry      = rots_shared[grp_base + 2u];
float rz      = rots_shared[grp_base + 3u];

// qbar * r — Hamilton product in [w, x, y, z] convention.
// Mirrors crate::isoquant_msl::DEQUANTIZE_SOURCE_ISO3 (and iso_decode_fast).
float v_for_lane;
if (elem_in_group == 0u) {
    v_for_lane = ISO_QW * rw - ISO_QCX * rx - ISO_QCY * ry - ISO_QCZ * rz;
} else if (elem_in_group == 1u) {
    v_for_lane = ISO_QW * rx + ISO_QCX * rw + ISO_QCY * rz - ISO_QCZ * ry;
} else if (elem_in_group == 2u) {
    v_for_lane = ISO_QW * ry - ISO_QCX * rz + ISO_QCY * rw + ISO_QCZ * rx;
} else {
    v_for_lane = ISO_QW * rz + ISO_QCX * ry - ISO_QCY * rx + ISO_QCZ * rw;
}
float norm  = norms[kv_tok];
float k_val = v_for_lane * norm;

// Phase 3: QK partial product + threadgroup tree reduction.
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
