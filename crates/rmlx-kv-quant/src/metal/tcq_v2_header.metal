
// 2-bit TurboQuant codebook (Lloyd-Max N(0,1)). Bit-exact with
// crate::turboquant::CODEBOOK_2BIT.
constant float CB2[4] = {
    as_type<float>(0xBFC147AEu),  // -1.51
    as_type<float>(0xBEE7AE14u),  // -0.453
    as_type<float>(0x3EE7AE14u),  //  0.453
    as_type<float>(0x3FC147AEu)   //  1.51
};

constant float CB2_MAX_K = as_type<float>(0x3FC147AEu);  // 1.51

// Viterbi trellis: 4 states, 4 levels. Transition formula:
//   next = ((state << 1) | (level & 1)) & 3
// Computed in-kernel rather than embedded as a constant table.
