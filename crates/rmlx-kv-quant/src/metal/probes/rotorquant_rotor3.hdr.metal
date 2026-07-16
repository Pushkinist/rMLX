
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
// 3-bit values per u32 word.
constant uint  ROTOR3_VPW = 10u;
// u32 words per group = 1 for both rotor3 (24/30 bits) and rotor4 (32/32 dense).
constant uint  ROTOR3_WPG = 1u;
