
// 4-bit TurboQuant codebook: 16 Lloyd-Max optimal N(0,1) centroids.
// Derived by turboquant_plus/turboquant/codebook.py::_lloyds_gaussian(sigma=1.0, n_iter=100).
// Replaces prior N(0,1) quantile centroids.
// Bit patterns match turboquant.rs::CODEBOOK_4BIT.
constant float CB[16] = {
    as_type<float>(0xC02DEE42u),  // -2.7176671
    as_type<float>(0xC003563Bu),  // -2.0521381
    as_type<float>(0xBFCCE718u),  // -1.6008024
    as_type<float>(0xBF9EB6FAu),  // -1.2399590
    as_type<float>(0xBF6DA172u),  // -0.9282447
    as_type<float>(0xBF255816u),  // -0.6458753
    as_type<float>(0xBEC329CBu),  // -0.3811782
    as_type<float>(0xBE011273u),  // -0.1260469
    as_type<float>(0x3E011273u),  //  0.1260469
    as_type<float>(0x3EC329CBu),  //  0.3811782
    as_type<float>(0x3F255816u),  //  0.6458753
    as_type<float>(0x3F6DA172u),  //  0.9282447
    as_type<float>(0x3F9EB6FAu),  //  1.2399590
    as_type<float>(0x3FCCE718u),  //  1.6008024
    as_type<float>(0x4003563Bu),  //  2.0521381
    as_type<float>(0x402DEE42u)   //  2.7176671
};

// 15 decision boundaries: midpoints between consecutive Lloyd-Max centroids,
// computed as (CB[i] + CB[i+1]) * 0.5f in single precision.
// Bit patterns match what the CPU turboquant.rs::nearest_centroid computes
// at runtime using the same formula.
constant float BOUNDARIES[15] = {
    as_type<float>(0xC018A23Eu),  // -2.3849025
    as_type<float>(0xBFE9C9C7u),  // -1.8264703
    as_type<float>(0xBFB5CF09u),  // -1.4203807
    as_type<float>(0xBF8AC3DAu),  // -1.0841019
    as_type<float>(0xBF497CC4u),  // -0.7870600
    as_type<float>(0xBF03767Eu),  // -0.5135268
    as_type<float>(0xBE81D982u),  // -0.2536126
    as_type<float>(0x00000000u),  //  0.0000000
    as_type<float>(0x3E81D982u),  //  0.2536126
    as_type<float>(0x3F03767Eu),  //  0.5135268
    as_type<float>(0x3F497CC4u),  //  0.7870600
    as_type<float>(0x3F8AC3DAu),  //  1.0841019
    as_type<float>(0x3FB5CF09u),  //  1.4203807
    as_type<float>(0x3FE9C9C7u),  //  1.8264703
    as_type<float>(0x4018A23Eu)   //  2.3849025
};
