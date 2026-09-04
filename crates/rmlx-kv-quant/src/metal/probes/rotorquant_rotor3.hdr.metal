
// 3-bit Lloyd-Max N(0,1) codebook — 8 entries.
// Bit patterns match turboquant.rs::CODEBOOK_3BIT.
constant float ROTOR3_CB[8] = {
    as_type<float>(0xC009B977u),
    as_type<float>(0xBFAC0532u),
    as_type<float>(0xBF418987u),
    as_type<float>(0xBE7AF9EBu),
    as_type<float>(0x3E7AF9EBu),
    as_type<float>(0x3F418987u),
    as_type<float>(0x3FAC0532u),
    as_type<float>(0x4009B977u)
};

// 7 midpoint decision boundaries for the 3-bit codebook.
constant float ROTOR3_BOUNDS[7] = {
    as_type<float>(0xBFDFBC10u),
    as_type<float>(0xBF8664FBu),
    as_type<float>(0xBF002401u),
    as_type<float>(0x00000000u),
    as_type<float>(0x3F002401u),
    as_type<float>(0x3F8664FBu),
    as_type<float>(0x3FDFBC10u)
};

constant float ROTOR3_CB_MAX = as_type<float>(0x4009B977u);
// Multivector component count (Cl(3,0) basis size).
constant uint  ROTOR3_MV = 8u;
// Grade-1 group size (3 grade-1 components per group; one rotor).
constant uint  ROTOR3_GS = 3u;

// Dense code plane — codes packed LSB-first across a row's groups, the row
// padded to a whole u32. Mirrors crate::code_plane.
#define CP_BITS 3u
#define CP_MASK 0x7u
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
