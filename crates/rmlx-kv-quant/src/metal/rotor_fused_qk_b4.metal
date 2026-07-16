
uint s_kv         = threadgroup_position_in_grid.x;
uint bh           = threadgroup_position_in_grid.y;
uint tid          = thread_position_in_threadgroup.x;
uint head_dim     = dims[0];
uint kv_seq       = dims[1];
uint kv_h         = dims[2];
uint heads_per_kv = dims[3];
uint has_mask     = dims[4];
uint n_groups     = dims[5];

uint n_q_heads = kv_h * heads_per_kv;
uint b         = bh / n_q_heads;
uint hq        = bh % n_q_heads;
uint kv_h_idx  = hq / heads_per_kv;

// Per-token layout:
//   codes_per_tok  = n_groups (1 u32 word per group of 8 mv components)
//   scales_per_tok = n_groups (1 f32 per group)
uint kv_tok         = (b * kv_h + kv_h_idx) * kv_seq + s_kv;
uint codes_tok_off  = kv_tok * n_groups;
uint scales_tok_off = kv_tok * n_groups;

// tid handles one head_dim element. Which group and which grade-1 lane?
uint group_id_in_head = tid / 3u;
uint lane_in_group    = tid - group_id_in_head * 3u; // 0, 1, or 2 → e_{lane+1}

// Phase 1: load Q.
threadgroup float q_shared[256];
q_shared[tid] = query[bh * head_dim + tid];

// Phase 2: per-thread rotor decode for its own head-dim slot.
// Each thread independently redoes the inverse sandwich for its group
// (acceptable triplication: 3 lanes per group all decode the same 8-component
// mv → very small kernel, no SMEM staging needed, see module docs).

// Read packed code word + scale + rotor for this group.
uint word     = codes[codes_tok_off + group_id_in_head];
float k_scale = scales[scales_tok_off + group_id_in_head];

uint rotor_base = group_id_in_head * 4u;
float rs        = rotors[rotor_base + 0u];
float rb12      = rotors[rotor_base + 1u];
float rb13      = rotors[rotor_base + 2u];
float rb23      = rotors[rotor_base + 3u];

// Unpack 8 BITS-bit codes → centroid × scale → mv_q[0..8].
float mv_q[8];
for (uint e = 0u; e < 8u; ++e) {
    uint shift = e * 4u;
    uint idx   = (word >> shift) & 0xFu;
    mv_q[e]    = ROTOR_CB[idx] * k_scale;
}

// Phase 3: inverse Cl(3,0) sandwich  restored = R̃ * mv_q * R.
//
// Rotor compact form r = (rs, rb12, rb13, rb23) → dense MV at basis
// positions [0, 4, 5, 6]. Clifford reverse R̃ flips the 3 bivector
// signs: R̃ → (rs, -rb12, -rb13, -rb23).
//
// gp(a, b)[k] = sum over (i, j) of a[i] * b[j] * MUL_S[i*8+j]
//               where MUL_T[i*8+j] == k.
//
// Step A: tmp = R̃ * mv_q. R̃ has 4 non-zero entries (positions 0, 4, 5, 6).
float rbar[8] = {rs, 0.0f, 0.0f, 0.0f, -rb12, -rb13, -rb23, 0.0f};

float tmp[8];
for (uint k = 0u; k < 8u; ++k) {
    tmp[k] = 0.0f;
}
uint sparse_i[4] = {0u, 4u, 5u, 6u};
for (uint a = 0u; a < 4u; ++a) {
    uint i   = sparse_i[a];
    float ri = rbar[i];
    for (uint j = 0u; j < 8u; ++j) {
        uint t    = MUL_T[i * 8u + j];
        float sgn = MUL_S[i * 8u + j];
        tmp[t] += ri * mv_q[j] * sgn;
    }
}

// Step B: restored = tmp * R. R has 4 non-zero entries (positions 0, 4, 5, 6).
float r_dense[8] = {rs, 0.0f, 0.0f, 0.0f, rb12, rb13, rb23, 0.0f};

float restored[8];
for (uint k = 0u; k < 8u; ++k) {
    restored[k] = 0.0f;
}
uint sparse_j[4] = {0u, 4u, 5u, 6u};
for (uint i = 0u; i < 8u; ++i) {
    float ti = tmp[i];
    for (uint c = 0u; c < 4u; ++c) {
        uint j    = sparse_j[c];
        float rj  = r_dense[j];
        uint t    = MUL_T[i * 8u + j];
        float sgn = MUL_S[i * 8u + j];
        restored[t] += ti * rj * sgn;
    }
}

// Phase 4: extract grade-1 lane for this thread and multiply by per-token L2.
float restored_lane = restored[lane_in_group + 1u];
float norm_tok      = norms[kv_tok];
float k_val         = restored_lane * norm_tok;

// Phase 5: QK partial product + threadgroup tree reduction.
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
