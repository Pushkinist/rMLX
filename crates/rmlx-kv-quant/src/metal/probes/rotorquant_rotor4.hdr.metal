
// 4-bit Lloyd-Max N(0,1) codebook — 16 entries.
// Bit patterns match turboquant.rs::CODEBOOK_4BIT.
constant float ROTOR4_CB[16] = {
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
constant float ROTOR4_BOUNDS[15] = {
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

constant float ROTOR4_CB_MAX = as_type<float>(0x402DEE42u);
// Multivector component count (Cl(3,0) basis size).
constant uint  ROTOR4_MV = 8u;
// Grade-1 group size (3 grade-1 components per group; one rotor).
constant uint  ROTOR4_GS = 3u;

// Dense code plane — codes packed LSB-first across a row's groups, the row
// padded to a whole u32. Mirrors crate::code_plane.
#define CP_BITS 4u
#define CP_MASK 0xFu
#define CP_CODES_PER_GROUP 3u

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
