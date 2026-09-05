
// Rotor flash-decode header — unpack params + Lloyd-Max N(0,1)
// codebook + Cl(3,0) MUL_TABLE + the shared per-lane K decode.
// BITS = 4 (codebook = 16 entries, mask = 0xF).
// Bit-exact with crate::clifford::MUL_TABLE + lloyd_gaussian_codebook(4).

#define RF_BITS 4u
#define RF_MASK 0xFu
#define RF_GROUP_SIZE 3u

constant float RF_CB[16] = {
    as_type<float>(0xC02DEE42u),
    as_type<float>(0xC003563Bu),
    as_type<float>(0xBFCCE718u),
    as_type<float>(0xBF9EB6FAu),
    as_type<float>(0xBF6DA172u),
    as_type<float>(0xBF255816u),
    as_type<float>(0xBEC329CBu),
    as_type<float>(0xBE011273u),
    as_type<float>(0x3E011273u),
    as_type<float>(0x3EC329CBu),
    as_type<float>(0x3F255816u),
    as_type<float>(0x3F6DA172u),
    as_type<float>(0x3F9EB6FAu),
    as_type<float>(0x3FCCE718u),
    as_type<float>(0x4003563Bu),
    as_type<float>(0x402DEE42u)
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

// Dense code plane — codes packed LSB-first across a row's groups, the row
// padded to a whole u32. Mirrors crate::code_plane.
#define CP_BITS 4u
#define CP_MASK 0xFu
#define CP_CODES_PER_GROUP 3u

// u32 words one row of `n_groups` groups occupies.
inline uint cp_row_words(uint n_groups) {
    return (n_groups * CP_CODES_PER_GROUP * CP_BITS + 31u) / 32u;
}

// One group's codes, returned by value: a vector lives in registers where a
// `thread uint[]` out-parameter is addressable and can be spilled to thread
// memory, which on the decode path costs more than the loads it saves.
typedef uint3 cp_group_t;

// Read all CP_CODES_PER_GROUP codes of group `group_id`.
//
// Templated over the address space because MLX binds a small input buffer as
// `constant` and a large one as `device`, and MSL will not convert between the
// two: a single-address-space reader compiles for one dispatch shape and fails
// the other at JIT time.
//
// A whole group is CP_CODES_PER_GROUP * CP_BITS bits — at most 32, and never
// spanning more than a word pair — so this is a word pair, where a code at a
// time would be one load per code. That matters: every decode
// lane needs its group's whole code set (the quaternion product and the rotor
// sandwich both mix all of them), so a per-code reader multiplies the loads on
// the hottest path in the kernel by the group size.
template <typename P>
inline cp_group_t cp_read_group(P codes, uint row_base, uint group_id) {
    uint span = CP_CODES_PER_GROUP * CP_BITS;
    uint bit  = group_id * span;
    uint word = row_base + (bit >> 5u);
    uint off  = bit & 31u;
    // A group's span never exceeds 32 bits, so its bits fit one word once
    // shifted down — no 64-bit arithmetic, which Apple GPUs emulate.
    //
    // The straddle is a select, not a branch. Whether a group crosses a word
    // boundary depends on its index, so within one simdgroup some lanes cross
    // and some do not: a branch there is executed by every lane anyway, and
    // costs the divergence on top. The second word is addressed with a select
    // (`word` when there is nothing to fetch, so never out of bounds) and its
    // contribution is masked to zero the same way.
    bool crosses = off + span > 32u;
    uint v       = codes[word] >> off;
    uint hi      = codes[word + (crosses ? 1u : 0u)];
    v |= crosses ? (hi << (32u - off)) : 0u;
    cp_group_t out;
    out.x = (v >> 0u * CP_BITS) & CP_MASK;
    out.y = (v >> 1u * CP_BITS) & CP_MASK;
    out.z = (v >> 2u * CP_BITS) & CP_MASK;

    return out;
}

// OR code `idx` into the row whose first word is `codes[row_base]`. The plane
// is zero-initialised at dispatch, so an OR is a write; a code that straddles
// two words ORs into both.
inline void cp_write_code(device uint* codes, uint row_base, uint idx, uint code) {
    uint bit  = idx * CP_BITS;
    uint word = row_base + (bit >> 5u);
    uint off  = bit & 31u;
    uint v    = code & CP_MASK;
    atomic_fetch_or_explicit((device atomic_uint *)&codes[word], v << off,
                             memory_order_relaxed);
    if (off + CP_BITS > 32u) {
        atomic_fetch_or_explicit((device atomic_uint *)&codes[word + 1u],
                                 v >> (32u - off), memory_order_relaxed);
    }
}

// Decode a whole Cl(3,0) block of a rotor-quantized token.
//
// `tok_idx` indexes the flat sequence-major token stream
// (`(b * kv_seq + t) * kv_h + kv_h_idx`); `group_id` is the block index
// in [0, n_groups). Writes the block's RF_GROUP_SIZE grade-1 lanes into
// `out`. Bit-exact with the CPU rotor{3,4}_decode.
//
// One decode per group: the sandwich runs once here rather than once
// per head-dim lane.
inline void rf_decode_k_group(
    device const uint*   codes,
    device const bfloat* scales,
    device const bfloat* norms,
    device const float*  rotors,
    uint                tok_idx,
    uint                n_groups,
    uint                row_words,
    uint                group_id,
    thread float*       out) {
    float k_scale = scales[tok_idx * n_groups + group_id];

    uint  rotor_base = group_id * 4u;
    float rs         = rotors[rotor_base + 0u];
    float rb12       = rotors[rotor_base + 1u];
    float rb13       = rotors[rotor_base + 2u];
    float rb23       = rotors[rotor_base + 3u];

    // Read the group's RF_GROUP_SIZE stored codes out of the dense
    // plane -> centroid x scale -> grade-1 slots of mv_q. The other
    // five components are algebraically zero: the sandwich preserves
    // grade, so nothing was stored for them.
    float mv_q[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint  row_base = tok_idx * row_words;
    cp_group_t g = cp_read_group(codes, row_base, group_id);
    mv_q[1] = RF_CB[g.x] * k_scale;
    mv_q[2] = RF_CB[g.y] * k_scale;
    mv_q[3] = RF_CB[g.z] * k_scale;

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

    // Grade-1 lives at MV indices 1..=RF_GROUP_SIZE; rescale by L2.
    float k_norm = norms[tok_idx];
    for (uint e = 0u; e < RF_GROUP_SIZE; ++e) {
        out[e] = restored[e + 1u] * k_norm;
    }
}

// Decode one head-dim lane of a rotor-quantized token.
//
// Thin wrapper over rf_decode_k_group: resolves the lane's block, then
// returns that lane's grade-1 component. `lane` is the head-dim slot in
// [0, head_dim). Shared surface: a quantized-V flash kernel calls this
// unchanged.
inline float rf_decode_k_lane(
    device const uint*   codes,
    device const bfloat* scales,
    device const bfloat* norms,
    device const float*  rotors,
    uint                tok_idx,
    uint                n_groups,
    uint                row_words,
    uint                lane) {
    // Each group of RF_GROUP_SIZE head-dim slots is one Cl(3,0) block.
    uint group_id_in_head = lane / RF_GROUP_SIZE;
    uint lane_in_group    = lane - group_id_in_head * RF_GROUP_SIZE;
    float g[RF_GROUP_SIZE];
    rf_decode_k_group(codes, scales, norms, rotors, tok_idx, n_groups,
                      row_words, group_id_in_head, g);
    return g[lane_in_group];
}

#define RF_TILE_SIZE 64u
#define RF_HEAD_DIM_MAX 512
