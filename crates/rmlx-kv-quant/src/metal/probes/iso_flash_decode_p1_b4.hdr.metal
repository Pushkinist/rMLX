
// Iso flash-decode header — unpack params + conjugated golden-ratio
// fixed quaternion + Lloyd-Max N(0,1) codebook + the shared per-lane
// K decode.
// BITS = 4 (codebook = 16 entries, mask = 0xF).
// Bit-exact with crate::isoquant::FIXED_QUAT + lloyd_gaussian_codebook(4).

#define IF_BITS 4u
#define IF_MASK 0xFu

constant float ISO_QW  = as_type<float>(0x3EE4F930u);
constant float ISO_QCX = as_type<float>(0xBF393E4Cu);
constant float ISO_QCY = as_type<float>(0xBE8D8368u);
constant float ISO_QCZ = as_type<float>(0xBEE4F930u);

constant float ISO_CB[16] = {
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

// Dense code plane — codes packed LSB-first across a row's groups, the row
// padded to a whole u32. Mirrors crate::code_plane.
#define CP_BITS 4u
#define CP_MASK 0xFu
#define CP_CODES_PER_GROUP 4u

// u32 words one row of `n_groups` groups occupies.
inline uint cp_row_words(uint n_groups) {
    return (n_groups * CP_CODES_PER_GROUP * CP_BITS + 31u) / 32u;
}

// Read code `idx` of the row whose first word is `codes[row_base]`.
//
// Templated over the address space because MLX binds a small input buffer as
// `constant` and a large one as `device`, and MSL will not convert between the
// two: a single-address-space reader compiles for one dispatch shape and fails
// the other at JIT time.
template <typename P>
inline uint cp_read_code(P codes, uint row_base, uint idx) {
    uint bit  = idx * CP_BITS;
    uint word = row_base + (bit >> 5u);
    uint off  = bit & 31u;
    uint v    = codes[word] >> off;
    if (off + CP_BITS > 32u) {
        v |= codes[word + 1u] << (32u - off);
    }
    return v & CP_MASK;
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

// Decode one head-dim lane of an iso-quantized token.
//
// `tok_idx` indexes the flat sequence-major token stream
// (`(b * kv_seq + t) * kv_h + kv_h_idx`); `lane` is the head-dim slot
// in [0, head_dim). Bit-exact with the CPU iso_decode_fast.
//
// Self-contained per lane: the group's four codes are four reads of
// the row's code plane, so the Hamilton product runs in registers
// with no threadgroup staging and no barrier — callable from any
// kernel body.
//
// Shared surface: a quantized-V flash kernel calls this unchanged,
// passing the V store's (codes, scales, norms) instead of K's.
inline float if_decode_k_lane(
    device const uint*   codes,
    device const bfloat* scales,
    device const bfloat* norms,
    uint                tok_idx,
    uint                n_groups,
    uint                lane) {
    // Each group of 4 head-dim slots is one quaternion block.
    uint group_id_in_head = lane / 4u;
    uint lane_in_group    = lane - group_id_in_head * 4u;

    float k_scale = scales[tok_idx * n_groups + group_id_in_head];

    // Read the group's 4 codes out of the row's dense plane ->
    // centroid x scale.
    uint row_base  = tok_idx * cp_row_words(n_groups);
    uint code_base = group_id_in_head * CP_CODES_PER_GROUP;
    float rw = ISO_CB[cp_read_code(codes, row_base, code_base + 0u)] * k_scale;
    float rx = ISO_CB[cp_read_code(codes, row_base, code_base + 1u)] * k_scale;
    float ry = ISO_CB[cp_read_code(codes, row_base, code_base + 2u)] * k_scale;
    float rz = ISO_CB[cp_read_code(codes, row_base, code_base + 3u)] * k_scale;

    // Inverse rotation: r' = qbar * r, Hamilton product in the
    // [w, x, y, z] convention. qbar = (ISO_QW, ISO_QCX, ISO_QCY,
    // ISO_QCZ) is already conjugated by the header.
    float v_for_lane;
    if (lane_in_group == 0u) {
        v_for_lane = ISO_QW * rw - ISO_QCX * rx - ISO_QCY * ry - ISO_QCZ * rz;
    } else if (lane_in_group == 1u) {
        v_for_lane = ISO_QW * rx + ISO_QCX * rw + ISO_QCY * rz - ISO_QCZ * ry;
    } else if (lane_in_group == 2u) {
        v_for_lane = ISO_QW * ry - ISO_QCX * rz + ISO_QCY * rw + ISO_QCZ * rx;
    } else {
        v_for_lane = ISO_QW * rz + ISO_QCX * ry - ISO_QCY * rx + ISO_QCZ * rw;
    }

    return v_for_lane * norms[tok_idx];
}

#define IF_TILE_SIZE 64u
#define IF_HEAD_DIM_MAX 512
