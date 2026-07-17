
// Iso flash-decode header — unpack params + conjugated golden-ratio
// fixed quaternion + Lloyd-Max N(0,1) codebook + the shared per-lane
// K decode.
// BITS = 3 (codebook = 8 entries, mask = 0x7).
// Bit-exact with crate::isoquant::FIXED_QUAT + lloyd_gaussian_codebook(3).

#define IF_BITS 3u
#define IF_MASK 0x7u

constant float ISO_QW  = as_type<float>(0x3EE4F930u);
constant float ISO_QCX = as_type<float>(0xBF393E4Cu);
constant float ISO_QCY = as_type<float>(0xBE8D8368u);
constant float ISO_QCZ = as_type<float>(0xBEE4F930u);

constant float ISO_CB[8] = {
    as_type<float>(0xC009B977u),
    as_type<float>(0xBFAC0532u),
    as_type<float>(0xBF418987u),
    as_type<float>(0xBE7AF9EBu),
    as_type<float>(0x3E7AF9EBu),
    as_type<float>(0x3F418987u),
    as_type<float>(0x3FAC0532u),
    as_type<float>(0x4009B977u)
};

// Decode one head-dim lane of an iso-quantized token.
//
// `tok_idx` indexes the flat sequence-major token stream
// (`(b * kv_seq + t) * kv_h + kv_h_idx`); `lane` is the head-dim slot
// in [0, head_dim). Bit-exact with the CPU iso_decode_fast.
//
// Self-contained per lane: the group's four codes all live in one u32,
// so the Hamilton product runs in registers with no threadgroup
// staging and no barrier — callable from any kernel body.
//
// Shared surface: a quantized-V flash kernel calls this unchanged,
// passing the V store's (codes, scales, norms) instead of K's.
inline float if_decode_k_lane(
    device const uint*  codes,
    device const float* scales,
    device const float* norms,
    uint                tok_idx,
    uint                n_groups,
    uint                lane) {
    // Each group of 4 head-dim slots is one quaternion block.
    uint group_id_in_head = lane / 4u;
    uint lane_in_group    = lane - group_id_in_head * 4u;

    uint  word    = codes[tok_idx * n_groups + group_id_in_head];
    float k_scale = scales[tok_idx * n_groups + group_id_in_head];

    // Unpack the group's 4 IF_BITS-bit codes -> centroid x scale.
    float rw = ISO_CB[(word >> (0u * IF_BITS)) & IF_MASK] * k_scale;
    float rx = ISO_CB[(word >> (1u * IF_BITS)) & IF_MASK] * k_scale;
    float ry = ISO_CB[(word >> (2u * IF_BITS)) & IF_MASK] * k_scale;
    float rz = ISO_CB[(word >> (3u * IF_BITS)) & IF_MASK] * k_scale;

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
