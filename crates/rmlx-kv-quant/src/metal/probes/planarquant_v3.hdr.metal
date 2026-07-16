
// PlanarQuant 3-bit — rotation codebook (same 16 Givens entries as V4).
constant float ROT_CB3[16][4] = {
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

// 3-bit Lloyd-Max N(0,1) codebook — 8 entries.
constant float CB3[8] = {
    as_type<float>(0xC009B977u),
    as_type<float>(0xBFAC0532u),
    as_type<float>(0xBF418987u),
    as_type<float>(0xBE7AF9EBu),
    as_type<float>(0x3E7AF9EBu),
    as_type<float>(0x3F418987u),
    as_type<float>(0x3FAC0532u),
    as_type<float>(0x4009B977u)
};

// 7 midpoint decision boundaries for 3-bit codebook.
constant float BOUNDARIES3[7] = {
    as_type<float>(0xBFDFBC10u),
    as_type<float>(0xBF8664FBu),
    as_type<float>(0xBF002401u),
    as_type<float>(0x00000000u),
    as_type<float>(0x3F002401u),
    as_type<float>(0x3F8664FBu),
    as_type<float>(0x3FDFBC10u)
};

constant float CB_MAX_3 = as_type<float>(0x4009B977u);
constant uint  PAIRS_PER_GROUP_C3 = 16u;
