
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
// 4-bit values per u32 word.
constant uint  ROTOR4_VPW = 8u;
// u32 words per group = 1 for both rotor3 (24/30 bits) and rotor4 (32/32 dense).
constant uint  ROTOR4_WPG = 1u;
