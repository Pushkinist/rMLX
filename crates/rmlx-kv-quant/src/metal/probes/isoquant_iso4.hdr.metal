
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
// 4-bit values per u32 word (8 vals, 32 bits used — dense pack).
constant uint  ISO4_VPW = 8u;
// u32 words per group = ceil(ISO4_GS / ISO4_VPW) = 1 for GS=4.
constant uint  ISO4_WPG = 1u;
