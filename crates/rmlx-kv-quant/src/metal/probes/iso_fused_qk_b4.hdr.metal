
// Iso fused-QK header — golden-ratio fixed quaternion + Lloyd-Max N(0,1) codebook.
// BITS = 4 (codebook = 16 entries).
// Bit-exact with crate::isoquant::FIXED_QUAT + lloyd_gaussian_codebook(4).
constant float ISO_QW  = as_type<float>(0x3EE4F930u);
constant float ISO_QCX = as_type<float>(0xBF393E4Cu);
constant float ISO_QCY = as_type<float>(0xBE8D8368u);
constant float ISO_QCZ = as_type<float>(0xBEE4F930u);

constant float ISO_CB[16] = {
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
