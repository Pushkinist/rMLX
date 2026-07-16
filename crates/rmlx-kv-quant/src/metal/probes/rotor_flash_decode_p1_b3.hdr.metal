
// Rotor flash-decode header — unpack params + Lloyd-Max N(0,1)
// codebook + Cl(3,0) MUL_TABLE + the shared per-lane K decode.
// BITS = 3 (codebook = 8 entries, mask = 0x7).
// Bit-exact with crate::clifford::MUL_TABLE + lloyd_gaussian_codebook(3).

#define RF_BITS 3u
#define RF_MASK 0x7u

constant float RF_CB[8] = {
    as_type<float>(0xC009B977u),
    as_type<float>(0xBFAC0532u),
    as_type<float>(0xBF418987u),
    as_type<float>(0xBE7AF9EBu),
    as_type<float>(0x3E7AF9EBu),
    as_type<float>(0x3F418987u),
    as_type<float>(0x3FAC0532u),
    as_type<float>(0x4009B977u)
};

// Cl(3,0) MUL_TABLE — target basis index + sign for e_I * e_J.
// Indexed row-major: MUL_T[i*8+j], MUL_S[i*8+j].
// Bit-exact with crate::clifford::MUL_TABLE (re-derived from BASIS_BITS).
constant uint RF_MUL_T[64] = {
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 
    1u, 0u, 4u, 5u, 2u, 3u, 7u, 6u, 
    2u, 4u, 0u, 6u, 1u, 7u, 3u, 5u, 
    3u, 5u, 6u, 0u, 7u, 1u, 2u, 4u, 
    4u, 2u, 1u, 7u, 0u, 6u, 5u, 3u, 
    5u, 3u, 7u, 1u, 6u, 0u, 4u, 2u, 
    6u, 7u, 3u, 2u, 5u, 4u, 0u, 1u, 
    7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u
};

constant float RF_MUL_S[64] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 
    1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 
    1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f
};

// Decode one head-dim lane of a rotor-quantized token.
//
// `tok_idx` indexes the flat sequence-major token stream
// (`(b * kv_seq + t) * kv_h + kv_h_idx`); `lane` is the head-dim slot
// in [0, head_dim). Bit-exact with the CPU rotor{3,4}_decode.
//
// Shared surface: a quantized-V flash kernel calls this unchanged.
inline float rf_decode_k_lane(
    device const uint*  codes,
    device const float* scales,
    device const float* norms,
    device const float* rotors,
    uint                tok_idx,
    uint                n_groups,
    uint                lane) {
    // Each group of 3 head-dim slots is one Cl(3,0) rotor block.
    uint group_id_in_head = lane / 3u;
    uint lane_in_group    = lane - group_id_in_head * 3u;

    uint  word    = codes[tok_idx * n_groups + group_id_in_head];
    float k_scale = scales[tok_idx * n_groups + group_id_in_head];

    uint  rotor_base = group_id_in_head * 4u;
    float rs         = rotors[rotor_base + 0u];
    float rb12       = rotors[rotor_base + 1u];
    float rb13       = rotors[rotor_base + 2u];
    float rb23       = rotors[rotor_base + 3u];

    // Unpack 8 RF_BITS-bit codes -> centroid x scale -> mv_q[0..8].
    float mv_q[8];
    for (uint e = 0u; e < 8u; ++e) {
        uint idx = (word >> (e * RF_BITS)) & RF_MASK;
        mv_q[e]  = RF_CB[idx] * k_scale;
    }

    // Inverse sandwich: restored = R~ * mv_q * R.
    //
    // Rotor compact form r = (rs, rb12, rb13, rb23) sits at dense MV
    // positions [0, 4, 5, 6]. The Clifford reverse R~ flips the three
    // bivector signs. gp(a, b)[k] = sum over (i, j) of
    // a[i] * b[j] * RF_MUL_S[i*8+j] where RF_MUL_T[i*8+j] == k.
    //
    // Step A: tmp = R~ * mv_q (R~ has 4 non-zero entries).
    float rbar[8] = {rs, 0.0f, 0.0f, 0.0f, -rb12, -rb13, -rb23, 0.0f};
    float tmp[8];
    for (uint k = 0u; k < 8u; ++k) {
        tmp[k] = 0.0f;
    }
    uint sparse_i[4] = {0u, 4u, 5u, 6u};
    for (uint a = 0u; a < 4u; ++a) {
        uint  i  = sparse_i[a];
        float ri = rbar[i];
        for (uint j = 0u; j < 8u; ++j) {
            tmp[RF_MUL_T[i * 8u + j]] += ri * mv_q[j] * RF_MUL_S[i * 8u + j];
        }
    }

    // Step B: restored = tmp * R (R has 4 non-zero entries).
    float r_dense[8] = {rs, 0.0f, 0.0f, 0.0f, rb12, rb13, rb23, 0.0f};
    float restored[8];
    for (uint k = 0u; k < 8u; ++k) {
        restored[k] = 0.0f;
    }
    uint sparse_j[4] = {0u, 4u, 5u, 6u};
    for (uint i = 0u; i < 8u; ++i) {
        float ti = tmp[i];
        for (uint c = 0u; c < 4u; ++c) {
            uint j = sparse_j[c];
            restored[RF_MUL_T[i * 8u + j]] += ti * r_dense[j] * RF_MUL_S[i * 8u + j];
        }
    }

    // Grade-1 lives at MV indices 1..=3; rescale by the per-token L2.
    return restored[lane_in_group + 1u] * norms[tok_idx];
}

#define RF_TILE_SIZE 64u
#define RF_HEAD_DIM_MAX 512
