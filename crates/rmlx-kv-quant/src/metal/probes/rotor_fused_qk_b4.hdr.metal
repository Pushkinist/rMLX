
// Rotor fused-QK header — Cl(3,0) MUL_TABLE + Lloyd-Max N(0,1) codebook.
// BITS = 4 (codebook = 16 entries).
// Bit-exact with crate::clifford::MUL_TABLE + lloyd_gaussian_codebook(4).

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
