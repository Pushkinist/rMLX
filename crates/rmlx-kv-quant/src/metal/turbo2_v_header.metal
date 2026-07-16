
// 2-bit TurboQuant codebook: 4 Lloyd-Max optimal N(0,1) centroids.
// Bit-exact with crate::turboquant::CODEBOOK_2BIT.
constant float CB2[4] = {
    as_type<float>(0xBFC147AEu), // -1.51
    as_type<float>(0xBEE7EF9Eu), // -0.453
    as_type<float>(0x3EE7EF9Eu), //  0.453
    as_type<float>(0x3FC147AEu)  //  1.51
};

// 3 decision boundaries: midpoints between consecutive centroids,
// computed as (CB2[i] + CB2[i+1]) * 0.5f in single precision.
constant float BOUNDARIES_2[3] = {
    as_type<float>(0xBF7B4396u), // -0.9815
    as_type<float>(0x00000000u), //  0.0
    as_type<float>(0x3F7B4396u)  //  0.9815
};
