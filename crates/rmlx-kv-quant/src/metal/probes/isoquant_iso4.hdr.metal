
// iso4 fixed quaternion q_L = [w, x, y, z] (golden-ratio unit quat).
// Source: multi_turboquant/methods/isoquant.py; bit-exact with isoquant.rs::FIXED_QUAT.
constant float ISO4_QW = as_type<float>(0x3EE4F930u);
constant float ISO4_QX = as_type<float>(0x3F393E4Cu);
constant float ISO4_QY = as_type<float>(0x3E8D8368u);
constant float ISO4_QZ = as_type<float>(0x3EE4F930u);
// Conjugate q̄_L = (w, -x, -y, -z) for dequantize (inverse rotation).
constant float ISO4_CX = as_type<float>(0xBF393E4Cu);
constant float ISO4_CY = as_type<float>(0xBE8D8368u);
constant float ISO4_CZ = as_type<float>(0xBEE4F930u);

// 4-bit Lloyd-Max N(0,1) codebook — 16 entries.
// Bit patterns match turboquant.rs::CODEBOOK_4BIT.
constant float ISO4_CB[16] = {
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

// 15 midpoint decision boundaries for the 4-bit codebook.
constant float ISO4_BOUNDS[15] = {
    as_type<float>(0xC018A23Eu),
    as_type<float>(0xBFE9C9C7u),
    as_type<float>(0xBFB5CF09u),
    as_type<float>(0xBF8AC3DAu),
    as_type<float>(0xBF497CC4u),
    as_type<float>(0xBF03767Eu),
    as_type<float>(0xBE81D982u),
    as_type<float>(0x00000000u),
    as_type<float>(0x3E81D982u),
    as_type<float>(0x3F03767Eu),
    as_type<float>(0x3F497CC4u),
    as_type<float>(0x3F8AC3DAu),
    as_type<float>(0x3FB5CF09u),
    as_type<float>(0x3FE9C9C7u),
    as_type<float>(0x4018A23Eu)
};

constant float ISO4_CB_MAX = as_type<float>(0x402DEE42u);
// Quaternion-block group size (4 elements per group).
constant uint  ISO4_GS = 4u;

// Dense code plane — codes packed LSB-first across a row's groups, the row
// padded to a whole u32. Mirrors crate::code_plane.
#define CP_BITS 4u
#define CP_MASK 0xFu
#define CP_CODES_PER_GROUP 4u

// u32 words one row of `n_groups` groups occupies.
inline uint cp_row_words(uint n_groups) {
    return (n_groups * CP_CODES_PER_GROUP * CP_BITS + 31u) / 32u;
}

// One group's codes, returned by value: a vector lives in registers where a
// `thread uint[]` out-parameter is addressable and can be spilled to thread
// memory, which on the decode path costs more than the loads it saves.
typedef uint4 cp_group_t;

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
    out.w = (v >> 3u * CP_BITS) & CP_MASK;

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
