
// PlanarQuant 16-entry Givens rotation codebook (bit-exact with CPU).
constant float PF_ROT_CB[16][4] = {
    {as_type<float>(0x3F800000u), as_type<float>(0x80000000u), as_type<float>(0x00000000u), as_type<float>(0x3F800000u)},
    {as_type<float>(0x3F7B14BEu), as_type<float>(0xBE47C5C2u), as_type<float>(0x3E47C5C2u), as_type<float>(0x3F7B14BEu)},
    {as_type<float>(0x3F6C835Eu), as_type<float>(0xBEC3EF16u), as_type<float>(0x3EC3EF16u), as_type<float>(0x3F6C835Eu)},
    {as_type<float>(0x3F54DB31u), as_type<float>(0xBF0E39DAu), as_type<float>(0x3F0E39DAu), as_type<float>(0x3F54DB31u)},
    {as_type<float>(0x3F3504F3u), as_type<float>(0xBF3504F3u), as_type<float>(0x3F3504F3u), as_type<float>(0x3F3504F3u)},
    {as_type<float>(0x3F0E39D9u), as_type<float>(0xBF54DB32u), as_type<float>(0x3F54DB32u), as_type<float>(0x3F0E39D9u)},
    {as_type<float>(0x3EC3EF15u), as_type<float>(0xBF6C835Eu), as_type<float>(0x3F6C835Eu), as_type<float>(0x3EC3EF15u)},
    {as_type<float>(0x3E47C5BCu), as_type<float>(0xBF7B14BFu), as_type<float>(0x3F7B14BFu), as_type<float>(0x3E47C5BCu)},
    {as_type<float>(0xB33BBD2Eu), as_type<float>(0xBF800000u), as_type<float>(0x3F800000u), as_type<float>(0xB33BBD2Eu)},
    {as_type<float>(0xBE47C5C2u), as_type<float>(0xBF7B14BEu), as_type<float>(0x3F7B14BEu), as_type<float>(0xBE47C5C2u)},
    {as_type<float>(0xBEC3EF18u), as_type<float>(0xBF6C835Eu), as_type<float>(0x3F6C835Eu), as_type<float>(0xBEC3EF18u)},
    {as_type<float>(0xBF0E39DCu), as_type<float>(0xBF54DB30u), as_type<float>(0x3F54DB30u), as_type<float>(0xBF0E39DCu)},
    {as_type<float>(0xBF3504F3u), as_type<float>(0xBF3504F3u), as_type<float>(0x3F3504F3u), as_type<float>(0xBF3504F3u)},
    {as_type<float>(0xBF54DB32u), as_type<float>(0xBF0E39D9u), as_type<float>(0x3F0E39D9u), as_type<float>(0xBF54DB32u)},
    {as_type<float>(0xBF6C8360u), as_type<float>(0xBEC3EF10u), as_type<float>(0x3EC3EF10u), as_type<float>(0xBF6C8360u)},
    {as_type<float>(0xBF7B14BFu), as_type<float>(0xBE47C5C1u), as_type<float>(0x3E47C5C1u), as_type<float>(0xBF7B14BFu)}
};

// Lloyd-Max N(0,1) centroid codebook (4-bit, 16 entries).
constant float PF_CB[16] = {
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

#define PF_TILE_SIZE 64u
