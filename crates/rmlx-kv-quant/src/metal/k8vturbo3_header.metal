
// 3-bit TurboQuant codebook: 8 Lloyd-Max optimal N(0,1) centroids.
// Bit-exact with crate::turboquant::CODEBOOK_3BIT.
constant float CB3[8] = {
    as_type<float>(0xC009B977u), // -2.1519449
    as_type<float>(0xBFAC0532u), // -1.3439085
    as_type<float>(0xBF418987u), // -0.7560048
    as_type<float>(0xBE7AF9EBu), // -0.2450940
    as_type<float>(0x3E7AF9EBu), //  0.2450940
    as_type<float>(0x3F418987u), //  0.7560048
    as_type<float>(0x3FAC0532u), //  1.3439085
    as_type<float>(0x4009B977u)  //  2.1519449
};

// 7 decision boundaries: midpoints between consecutive centroids,
// computed as (CB3[i] + CB3[i+1]) * 0.5f in single precision.
constant float BOUNDARIES_3[7] = {
    as_type<float>(0xBFDFBC10u), // -1.7479267
    as_type<float>(0xBF8664FBu), // -1.0499567
    as_type<float>(0xBF002401u), // -0.5005494
    as_type<float>(0x00000000u), //  0.0000000
    as_type<float>(0x3F002401u), //  0.5005494
    as_type<float>(0x3F8664FBu), //  1.0499567
    as_type<float>(0x3FDFBC10u)  //  1.7479267
};
