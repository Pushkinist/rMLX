
// Rotor fused-QK header — Cl(3,0) MUL_TABLE + Lloyd-Max N(0,1) codebook.
// BITS = 3 (codebook = 8 entries).
// Bit-exact with crate::clifford::MUL_TABLE + lloyd_gaussian_codebook(3).

constant float ROTOR_CB[8] = {
    as_type<float>(0xC009B977u),
    as_type<float>(0xBFAC0532u),
    as_type<float>(0xBF418987u),
    as_type<float>(0xBE7AF9EBu),
    as_type<float>(0x3E7AF9EBu),
    as_type<float>(0x3F418987u),
    as_type<float>(0x3FAC0532u),
    as_type<float>(0x4009B977u)
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
