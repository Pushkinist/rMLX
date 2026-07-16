
// 3-bit TurboQuant codebook (Lloyd-Max N(0,1)). Bit-exact with
// crate::turboquant::CODEBOOK_3BIT.
constant float CB3[8] = {
    as_type<float>(0xC009B977u),  // -2.1519449
    as_type<float>(0xBFAC0532u),  // -1.3439085
    as_type<float>(0xBF418987u),  // -0.7560048
    as_type<float>(0xBE7AF9EBu),  // -0.2450940
    as_type<float>(0x3E7AF9EBu),  //  0.2450940
    as_type<float>(0x3F418987u),  //  0.7560048
    as_type<float>(0x3FAC0532u),  //  1.3439085
    as_type<float>(0x4009B977u)   //  2.1519449
};

constant float CB3_MAX_K = as_type<float>(0x4009B977u);  // 2.1519449

// Viterbi trellis: 4 states, 8 levels. Transition formula:
//   next = ((state << 1) | (level & 1)) & 3
// Computed in-kernel rather than embedded as a constant table; the latter
// would still require a runtime modulo.
