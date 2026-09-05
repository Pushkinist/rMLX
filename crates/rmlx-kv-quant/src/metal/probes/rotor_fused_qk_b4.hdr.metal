
// Rotor fused-QK header — Cl(3,0) MUL_TABLE + Lloyd-Max N(0,1) codebook.
// BITS = 4 (codebook = 16 entries, mask = 0xF).
// Bit-exact with crate::clifford::MUL_TABLE + lloyd_gaussian_codebook(4).

#define RF_BITS 4u
#define RF_MASK 0xFu

constant float ROTOR_CB[16] = {
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
constant uint MUL_T[64] = {
    0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 
    1u, 0u, 4u, 5u, 2u, 3u, 7u, 6u, 
    2u, 4u, 0u, 6u, 1u, 7u, 3u, 5u, 
    3u, 5u, 6u, 0u, 7u, 1u, 2u, 4u, 
    4u, 2u, 1u, 7u, 0u, 6u, 5u, 3u, 
    5u, 3u, 7u, 1u, 6u, 0u, 4u, 2u, 
    6u, 7u, 3u, 2u, 5u, 4u, 0u, 1u, 
    7u, 6u, 5u, 4u, 3u, 2u, 1u, 0u
};

constant float MUL_S[64] = {
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
