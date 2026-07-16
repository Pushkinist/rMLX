
// iso3 fixed quaternion q_L = [w, x, y, z] (golden-ratio unit quat).
// Source: multi_turboquant/methods/isoquant.py; bit-exact with isoquant.rs::FIXED_QUAT.
constant float ISO3_QW = as_type<float>(0x3EE4F930u);
constant float ISO3_QX = as_type<float>(0x3F393E4Cu);
constant float ISO3_QY = as_type<float>(0x3E8D8368u);
constant float ISO3_QZ = as_type<float>(0x3EE4F930u);
// Conjugate q̄_L = (w, -x, -y, -z) for dequantize (inverse rotation).
// q̄_L.w == q_L.w, so ISO3_QW is reused.
constant float ISO3_CX = as_type<float>(0xBF393E4Cu);
constant float ISO3_CY = as_type<float>(0xBE8D8368u);
constant float ISO3_CZ = as_type<float>(0xBEE4F930u);

// 3-bit Lloyd-Max N(0,1) codebook — 8 entries.
// Bit patterns match turboquant.rs::CODEBOOK_3BIT.
constant float ISO3_CB[8] = {
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
constant float ISO3_BOUNDS[7] = {
    as_type<float>(0xBFDFBC10u),
    as_type<float>(0xBF8664FBu),
    as_type<float>(0xBF002401u),
    as_type<float>(0x00000000u),
    as_type<float>(0x3F002401u),
    as_type<float>(0x3F8664FBu),
    as_type<float>(0x3FDFBC10u)
};

constant float ISO3_CB_MAX = as_type<float>(0x4009B977u);
// Quaternion-block group size (4 elements per group).
constant uint  ISO3_GS = 4u;
// 3-bit values per u32 word (10 vals, 30 bits used).
constant uint  ISO3_VPW = 10u;
// u32 words per group = ceil(ISO3_GS / ISO3_VPW) = 1 for GS=4.
constant uint  ISO3_WPG = 1u;
