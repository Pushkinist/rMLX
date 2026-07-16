
// Iso fused-QK header — golden-ratio fixed quaternion + Lloyd-Max N(0,1) codebook.
// BITS = 3 (codebook = 8 entries).
// Bit-exact with crate::isoquant::FIXED_QUAT + lloyd_gaussian_codebook(3).
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
