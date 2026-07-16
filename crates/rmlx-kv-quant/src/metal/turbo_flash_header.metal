
// TurboFlash — rMLX split-K FlashAttention.
//
// History: TheTom's original TurboFlash is default-OFF on Apple10 (M5+) due
// to corruption (commit `67f076f2e`). rMLX's adaptation uses different
// dequantization arithmetic but the same split-K pattern. The initial B1
// validation (2026-05) reproduced a SIGSEGV at head_dim=256, 32k on M5 Max;
// a 2026-06 re-validation on M5 Max showed the failure did not reproduce.
// CLI default: --turbo-flash auto resolves ON Apple10+.
// Env override: RMLX_TURBO_FLASH=0 hard-disables.
//
// K format: rMLX q8_0 (group_size=128, f32 scale, i8 codes packed 4/u32).
// V format: rMLX turbo4 (group_size=32, f32 scale, 4-bit Lloyd-Max 8/u32).
// No WHT rotation: rMLX turbo4 is a scalar codebook, not WHT-rotated.

// 4-bit TurboQuant codebook: 16 Lloyd-Max N(0,1) centroids.
constant float TURBO_CB[16] = {
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

#define BLOCK_SIZE 64u
#define TG_SIZE    32u
#define Q8_GROUP   128u
#define TQ4_GROUP  32u
