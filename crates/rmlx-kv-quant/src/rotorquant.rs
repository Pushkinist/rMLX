//! rotor3 / rotor4 KV-cache codec — Cl(3,0) Clifford rotor sandwich + 3-bit or 4-bit codebook.
//!
//! # Algorithm
//!
//! For each token's `head_dim`-element row:
//!
//!   1. L2-normalise the vector; store the norm separately.
//!   2. Reshape into `n_groups = ceil(head_dim / 3)` groups of 3 elements.
//!      The last group is **zero-padded** if `head_dim % 3 != 0`; decode
//!      masks the padded slots back out.
//!   3. Embed each 3-vector as the grade-1 part of a Cl(3,0) multivector:
//!      `mv = [0, v1, v2, v3, 0, 0, 0, 0]`.
//!   4. Apply per-group rotor sandwich `T(mv) = R_g * mv * R_g̃` (see
//!      [`crate::clifford`]). Rotors are **static per (layer, head)** — the
//!      caller supplies the rotor table; the codec does not regenerate it
//!      per token.
//!   5. Per-group, find `max|r_i|` over all 8 multivector components and
//!      set `scale = max / max_centroid`.
//!   6. Quantise each of the 8 components against
//!      [`lloyd_gaussian_codebook`]`(3)` (8 centroids).
//!   7. Pack 3-bit codes at **10 vals per u32** (planar3 / iso3 convention —
//!      30 bits used, 2 wasted). Per group: 8 codes = 24 bits → 1 u32.
//!
//! Decode reverses:
//!   1. Unpack 8 codes per group → centroid lookup → multiply by per-group scale.
//!   2. Apply inverse sandwich `R_g̃ * mv_q * R_g` to undo the rotation.
//!   3. Extract grade-1 components `(e1, e2, e3)` back to the 3-vector.
//!   4. Rescale by the stored L2 norm.
//!   5. Strip tail-pad slots from the last group.
//!
//! # Storage layout
//!
//! Encode output (per call, flat `Vec`s):
//!
//! | Buffer       | Shape                                  | Notes |
//! |--------------|----------------------------------------|-------|
//! | `codes`      | `n_tokens * n_groups * 1` u32          | 8 codes × 3 bits per group, packed 10 vals/u32 |
//! | `scales`     | `n_tokens * n_groups` f32              | per-group scale |
//! | `norms`      | `n_tokens` f32                         | per-token L2 norm |
//!
//! The static `rotors` `[n_groups, 4]` is **not** returned by the codec —
//! the caller (storage struct) owns it for the lifetime of the layer.
//!
//! # Pack convention
//!
//! `vals_per_word = 10`, `mask = 0x7`. Element `e` within a group of `8`
//! codes maps to `word = 0`, `shift = e * 3`. For `head_dim` not a multiple
//! of 3 the padded tail elements still consume code slots — decode applies
//! the mask **after** extracting the grade-1 components.
//!
//! # Single-codebook simplification
//!
//! The Python reference (`rotorquant/turboquant/rotorquant.py`) uses two
//! codebooks (`vector` grade and `trivector` grade) with different bit
//! budgets. This implementation ships a **single 8-centroid codebook** for
//! all 8 components to keep the storage flat. The grade-aware variant is
//! tracked as a follow-up (deferred); the cosine gate is measured empirically.
//!
//! # Effective bpe
//!
//! The single-codebook simplification ships at **~8 bpe pre-scale** — each
//! group quantises 3 real grade-1 elements but stores 8 three-bit codes
//! (24 bits) for the full 8-component multivector, plus one per-group scale and
//! one per-token norm, both at
//! [`crate::storage::KV_SIDEBAND_DTYPE`]. The grade-aware variant — which would
//! drop the bit budget for the high grades and bring the storage closer to
//! the 3.25 bpe target reported by the Python `rotorquant` reference — is
//! gated on the follow-up (deferred).
//!
//! **"Pre-scale" is doing a lot of work in that number, and the delivered
//! figure is what matters.** At `head_dim = 128` there are
//! `ceil(128 / 3) = 43` groups per row, and the store spends, per input value:
//!
//! | Component | Layout | Bits / value |
//! |---|---|---|
//! | codes | 43 `u32` per 128 values | **10.75** |
//! | scales | 43 `bf16` per 128 values | **5.375** |
//! | norm | 1 `bf16` per 128 values | **0.125** |
//! | **total** | | **16.25** (bf16 is 16.0) |
//!
//! `rotor_rate_splits_into_documented_code_scale_and_norm_bits` measures that
//! split off a real encode rather than restating it, and
//! [`crate::storage::ring_bits_per_value`] is where the total comes from.
//!
//! It was 21.75 while the scale and norm planes were `f32`. Halving them was a
//! 25% saving that still does not reach bf16, which is the point of the split
//! above: a sideband change cannot fix a code cadence. Two independent
//! overruns remain, with different fixes:
//!
//! 1. **Dead components.** Only 3 of the 8 quantised components carry
//!    information. A rotor sandwich is grade-preserving, so for the grade-1
//!    input this codec embeds, the scalar, bivector and pseudoscalar slots are
//!    algebraically zero on encode, and on decode the inverse sandwich keeps
//!    every non-grade-1 part out of the reconstructed vector. 15 of the 24
//!    code bits per group are therefore dead budget, not distortion — pinned by
//!    `clifford_tests::sandwich_of_grade1_in_3d_stays_grade1` and
//!    `clifford_tests::inverse_sandwich_of_non_grade1_leaks_nothing_into_grade1`.
//! 2. **Scale cadence.** One `bf16` per 3 values is 5.375 bits per value on its
//!    own — a third of the total, and the dominant term after the codes. A
//!    group of 3 sharing a whole scale is not a viable rate point at any code
//!    width, at any sideband width.
//!
//! rotor4 fills all 32 code bits instead of 24 and therefore occupies
//! *byte-identical* storage. No rotor codec is smaller than bf16 at any
//! `head_dim`; see `docs/KV_QUANT.md` § "Memory truth" and the crate-wide
//! stored-rate ceiling in `kv_rate_tests.rs`, where this family's overrun is a
//! written exemption that `exempt_families_actually_exceed_the_floor` will
//! reject the day it stops being one.
//!
//! # No QJL residual
//!
//! V-side codec only — the 1-bit QJL residual stage from
//! [`crate::clifford`]'s `RotorQuantProd` Python reference is K-only and
//! out of scope here.
//!
//! # References
//!
//! * Python source: `rotorquant/turboquant/rotorquant.py` (`RotorQuantMSE`).
//! * MSL reference: `rotorquant/turboquant/rotor_fused.metal`.
//! * Paper: `rotorquant/paper/rotorquant.pdf`.

use thiserror::Error;

use crate::clifford::{rotor_reverse, rotor_sandwich, Rotor, MV_DIM};
use crate::storage::bf16_round;
use crate::turboquant::lloyd_gaussian_codebook;

/// Bit-width of the rotor3 codebook (fixed at 3-bit Lloyd-Max).
pub const ROTOR3_BITS: u8 = 3;

/// Bit-width of the rotor4 codebook (fixed at 4-bit Lloyd-Max).
pub const ROTOR4_BITS: u8 = 4;

/// Multivector group size: 3 grade-1 components per group (one rotor).
pub const ROTOR3_GROUP_SIZE: usize = 3;

/// Group size for the rotor4 codec — identical to rotor3 (3 grade-1 components
/// per group, one rotor per group).
pub const ROTOR4_GROUP_SIZE: usize = ROTOR3_GROUP_SIZE;

/// Number of multivector components per group (Cl(3,0) basis size).
pub const ROTOR3_MV_COMPONENTS: usize = MV_DIM;

/// 3-bit values per u32 word (planar3 / iso3 convention — 30 bits used).
pub const ROTOR3_VALS_PER_WORD: usize = 10;

/// 4-bit values per u32 word — dense pack (32 bits used, 0 wasted — iso4
/// convention).
pub const ROTOR4_VALS_PER_WORD: usize = 8;

/// Words per group for rotor3: 8 codes × 3 bits = 24 bits ≤ 30, fits in 1 u32.
pub const ROTOR3_WORDS_PER_GROUP: usize = ROTOR3_MV_COMPONENTS.div_ceil(ROTOR3_VALS_PER_WORD);

/// Words per group for rotor4: 8 codes × 4 bits = 32 bits, fits in 1 u32
/// (dense pack — same u32-per-group count as rotor3, just denser).
pub const ROTOR4_WORDS_PER_GROUP: usize = ROTOR3_MV_COMPONENTS.div_ceil(ROTOR4_VALS_PER_WORD);

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors from the rotor3 codec.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed codec-internal error enum; new validation guards may add variants but this is not a public extension point"
)]
#[derive(Debug, Error)]
pub enum RotorQuantError {
    /// `head_dim` is zero — the codec requires at least one rotor group.
    #[error("rotor3: head_dim must be > 0")]
    HeadDimZero,
    /// The rotor table shape `[n_groups, 4]` does not match the head_dim's
    /// `ceil(head_dim / 3)` group count.
    #[error(
        "rotor3: rotor table length {got} does not equal expected {expected} \
         (n_groups={n_groups} * 4 per rotor)"
    )]
    RotorTableLen {
        /// Actual rotor flat length.
        got: usize,
        /// Expected `n_groups * 4`.
        expected: usize,
        /// `n_groups = ceil(head_dim / 3)`.
        n_groups: usize,
    },
    /// Input slice length is not a multiple of `head_dim`.
    #[error("rotor3: v.len()={len} is not a multiple of head_dim={head_dim}")]
    LenNotMultipleOfHeadDim {
        /// The actual slice length.
        len: usize,
        /// The expected head dimension divisor.
        head_dim: usize,
    },
    /// The underlying Lloyd-Max codebook lookup failed (should be unreachable
    /// for `ROTOR3_BITS=3`; surfaced as defence in depth).
    #[error("rotor3: codebook error: {0}")]
    Codebook(String),
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Number of multivector groups for a given `head_dim`.
///
/// `n_groups = ceil(head_dim / 3)`. The last group has `head_dim % 3` real
/// elements and `3 - (head_dim % 3)` zero-padded slots.
#[inline]
#[must_use]
pub fn n_groups_for(head_dim: usize) -> usize {
    head_dim.div_ceil(ROTOR3_GROUP_SIZE)
}

/// Number of u32 words needed to pack the codes for `n_tokens * n_groups`
/// groups (`ROTOR3_WORDS_PER_GROUP` words per group, currently `1`).
#[inline]
#[must_use]
pub fn n_code_words_for(n_tokens: usize, head_dim: usize) -> usize {
    n_tokens * n_groups_for(head_dim) * ROTOR3_WORDS_PER_GROUP
}

/// Pack 8 codes for one group into 1 u32 (planar3 / iso3 convention).
///
/// `mask = 0x7`; element `e` lands at bits `[e*3 .. e*3+3]`. With 8 elements
/// the highest bit used is `7 * 3 + 2 = 23` — well within a single u32.
#[inline]
fn pack_group(codes: [u8; ROTOR3_MV_COMPONENTS]) -> u32 {
    let mut w: u32 = 0;
    for (e, &c) in codes.iter().enumerate() {
        w |= (u32::from(c) & 0x7) << (e * 3);
    }
    w
}

/// Unpack 1 u32 into 8 codes (inverse of [`pack_group`]).
#[inline]
fn unpack_group(word: u32) -> [u8; ROTOR3_MV_COMPONENTS] {
    let mut out = [0_u8; ROTOR3_MV_COMPONENTS];
    for (e, slot) in out.iter_mut().enumerate() {
        *slot = ((word >> (e * 3)) & 0x7) as u8;
    }
    out
}

// ── Encode ────────────────────────────────────────────────────────────────────

/// Encode a V tensor with the rotor3 codec.
///
/// # Arguments
///
/// - `v` — flat f32 slice of shape `[n_tokens, head_dim]`.
/// - `rotors` — static per-(layer, head) rotor table, flat
///   `[n_groups * 4]` f32 in `[s, b12, b13, b23]` order.
/// - `head_dim` — must be `> 0`.
///
/// # Returns
///
/// `(codes_packed, scales, norms)`:
///
/// - `codes_packed` — 3-bit indices packed at 10 per u32. Length
///   `n_tokens * n_groups * ROTOR3_WORDS_PER_GROUP`.
/// - `scales` — per-group f32 scale. Length `n_tokens * n_groups`.
/// - `norms` — per-token L2 norm. Length `n_tokens`.
///
/// # Errors
///
/// Returns [`RotorQuantError`] for invalid inputs (zero `head_dim`,
/// `v.len()` not a multiple of `head_dim`, rotor table size mismatch, or
/// a codebook fault).
pub fn rotor3_encode(
    v: &[f32],
    rotors: &[f32],
    head_dim: usize,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>), RotorQuantError> {
    if head_dim == 0 {
        return Err(RotorQuantError::HeadDimZero);
    }
    if !v.len().is_multiple_of(head_dim) {
        return Err(RotorQuantError::LenNotMultipleOfHeadDim {
            len: v.len(),
            head_dim,
        });
    }
    let n_groups = n_groups_for(head_dim);
    let expected = n_groups * 4;
    if rotors.len() != expected {
        return Err(RotorQuantError::RotorTableLen {
            got: rotors.len(),
            expected,
            n_groups,
        });
    }

    let codebook = lloyd_gaussian_codebook(ROTOR3_BITS)
        .map_err(|e| RotorQuantError::Codebook(e.to_string()))?;
    let max_centroid = codebook
        .iter()
        .copied()
        .fold(0.0_f32, |acc, c| acc.max(c.abs()));

    let n_tokens = v.len() / head_dim;
    let words_per_group = ROTOR3_WORDS_PER_GROUP;
    let total_groups = n_tokens * n_groups;

    let mut codes_all: Vec<u32> = Vec::with_capacity(total_groups * words_per_group);
    let mut scales: Vec<f32> = Vec::with_capacity(total_groups);
    let mut norms: Vec<f32> = Vec::with_capacity(n_tokens);

    for tok in 0..n_tokens {
        // tok < n_tokens = v.len() / head_dim guarantees the row slice fits.
        #[allow(
            clippy::indexing_slicing,
            reason = "tok < n_tokens = v.len()/head_dim; row slice is in-bounds"
        )]
        let row = &v[tok * head_dim..(tok + 1) * head_dim];
        // Rounded to the stored sideband precision before use — see
        // `crate::isoquant`: the decode multiplies by the *stored* norm.
        let norm = {
            let sq: f32 = row.iter().map(|&x| x * x).sum();
            bf16_round(sq.sqrt().max(1e-8))
        };
        norms.push(norm);

        for grp in 0..n_groups {
            // Load the per-group rotor: `[s, b12, b13, b23]`.
            let r_base = grp * 4;
            // r_base + 3 < rotors.len() == n_groups * 4 by validation above.
            #[allow(
                clippy::indexing_slicing,
                reason = "r_base+3 < n_groups*4 = rotors.len() validated above"
            )]
            let r: Rotor = [
                rotors[r_base],
                rotors[r_base + 1],
                rotors[r_base + 2],
                rotors[r_base + 3],
            ];

            // Load 3 grade-1 components, zero-padding the tail.
            let grp_start = grp * ROTOR3_GROUP_SIZE;
            let mut mv = [0.0_f32; ROTOR3_MV_COMPONENTS];
            for e in 0..ROTOR3_GROUP_SIZE {
                let idx = grp_start + e;
                if idx < head_dim {
                    // idx < head_dim ≤ row.len() by construction.
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "idx < head_dim = row.len() by construction"
                    )]
                    {
                        mv[e + 1] = row[idx] / norm; // grade-1: e1, e2, e3 (indices 1, 2, 3)
                    }
                }
            }

            // Forward sandwich: rotated = R * mv * R̃.
            //
            // Earlier versions called `gp_rotor_mv` twice (the second call put
            // the reverse on the left), which is `R̃ * (R * mv) = mv` — a
            // silent no-op. The dense [`rotor_sandwich`] helper applies the
            // correct `R * mv * R̃`.
            let rotated = rotor_sandwich(r, &mv);

            // Per-group max-abs over all 8 components → scale.
            let max_abs = rotated
                .iter()
                .copied()
                .fold(0.0_f32, |acc, x| acc.max(x.abs()));
            // Rounded to the stored sideband precision before the codes are
            // chosen against it.
            let scale = bf16_round(if max_abs < 1e-12 {
                1e-12
            } else {
                max_abs / max_centroid
            });
            scales.push(scale);

            // Quantise each component → nearest centroid (linear scan; 8 entries).
            let mut group_codes = [0_u8; ROTOR3_MV_COMPONENTS];
            for (e, &rv) in rotated.iter().enumerate() {
                let normalised = rv / scale;
                let idx = codebook
                    .iter()
                    .enumerate()
                    .min_by(|(_, &a), (_, &b)| {
                        (a - normalised).abs().total_cmp(&(b - normalised).abs())
                    })
                    .map_or(0, |(i, _)| i as u8);
                // idx < 8 by construction (codebook has 8 entries); fits 3 bits.
                #[allow(
                    clippy::indexing_slicing,
                    reason = "e < ROTOR3_MV_COMPONENTS = group_codes.len() by loop bound"
                )]
                {
                    group_codes[e] = idx;
                }
            }

            // Pack 8 codes into 1 u32 (planar3 / iso3 convention).
            codes_all.push(pack_group(group_codes));
        }
    }

    Ok((codes_all, scales, norms))
}

// ── Decode ────────────────────────────────────────────────────────────────────

/// Decode a rotor3-compressed V tensor.
///
/// Reverses [`rotor3_encode`]. All parameters must match the encode call;
/// in particular, `rotors` must be the **same static table** the encoder
/// saw — otherwise the inverse sandwich rotates into the wrong basis.
///
/// # Arguments
///
/// - `codes_packed` — packed 3-bit indices (10 per u32).
/// - `scales` — per-group f32 scale, length `n_tokens * n_groups`.
/// - `norms` — per-token L2 norm, length `n_tokens`.
/// - `rotors` — static per-(layer, head) rotor table (`n_groups * 4` f32).
/// - `head_dim` — must be `> 0` and match the encode call.
///
/// # Returns
///
/// Dequantised f32 values of length `norms.len() * head_dim`.
///
/// # Errors
///
/// Returns [`RotorQuantError`] for shape/length mismatches.
pub fn rotor3_decode(
    codes_packed: &[u32],
    scales: &[f32],
    norms: &[f32],
    rotors: &[f32],
    head_dim: usize,
) -> Result<Vec<f32>, RotorQuantError> {
    if head_dim == 0 {
        return Err(RotorQuantError::HeadDimZero);
    }
    let n_groups = n_groups_for(head_dim);
    let expected = n_groups * 4;
    if rotors.len() != expected {
        return Err(RotorQuantError::RotorTableLen {
            got: rotors.len(),
            expected,
            n_groups,
        });
    }

    let codebook = lloyd_gaussian_codebook(ROTOR3_BITS)
        .map_err(|e| RotorQuantError::Codebook(e.to_string()))?;

    let n_tokens = norms.len();
    let words_per_group = ROTOR3_WORDS_PER_GROUP;
    let mut out = vec![0.0_f32; n_tokens * head_dim];

    for (tok, &norm) in norms.iter().enumerate() {
        for grp in 0..n_groups {
            let flat_grp = tok * n_groups + grp;
            // flat_grp < n_tokens * n_groups = scales.len() by encode contract.
            #[allow(
                clippy::indexing_slicing,
                reason = "flat_grp < n_tokens*n_groups = scales.len() by encode contract"
            )]
            let scale = scales[flat_grp];

            // Load rotor.
            let r_base = grp * 4;
            #[allow(
                clippy::indexing_slicing,
                reason = "r_base+3 < n_groups*4 = rotors.len() validated above"
            )]
            let r: Rotor = [
                rotors[r_base],
                rotors[r_base + 1],
                rotors[r_base + 2],
                rotors[r_base + 3],
            ];

            // Unpack 8 codes from 1 u32.
            let word_base = flat_grp * words_per_group;
            #[allow(
                clippy::indexing_slicing,
                reason = "word_base < codes_packed.len() by encode contract (total_groups * 1)"
            )]
            let word = codes_packed[word_base];
            let codes = unpack_group(word);

            // Dequantise: code → centroid → multiply by per-group scale.
            let mut mv_q = [0.0_f32; ROTOR3_MV_COMPONENTS];
            for (slot, &c) in mv_q.iter_mut().zip(codes.iter()) {
                // c < 8 = codebook.len() by construction (pack mask 0x7).
                #[allow(
                    clippy::indexing_slicing,
                    reason = "c < 2^3 = codebook.len() by pack mask (0x7)"
                )]
                let centroid = codebook[c as usize];
                *slot = centroid * scale;
            }

            // Inverse sandwich: R̃ * mv_q * R.
            //
            // Earlier versions called `gp_rotor_mv` twice (the second call put
            // R on the left of the intermediate), which collapsed to
            // `R * (R̃ * mv_q) = mv_q` — a silent no-op. The dense
            // [`rotor_sandwich`] helper applies the correct `R̃ * mv_q * R`
            // when fed the reversed rotor.
            let restored = rotor_sandwich(rotor_reverse(r), &mv_q);

            // Extract grade-1 components → original 3-vector slot.
            let out_row = tok * head_dim;
            let grp_start = grp * ROTOR3_GROUP_SIZE;
            for e in 0..ROTOR3_GROUP_SIZE {
                let idx = grp_start + e;
                if idx < head_dim {
                    // idx < head_dim ≤ out.len() - tok*head_dim by construction.
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "out_row + idx < n_tokens*head_dim = out.len() by construction"
                    )]
                    {
                        // Grade-1 sits at MV indices 1..=3.
                        out[out_row + idx] = restored[e + 1] * norm;
                    }
                }
            }
        }
    }

    Ok(out)
}

// ── rotor4 pack / unpack ───────────────────────────────────────────────────────

/// Pack 8 codes for one group into 1 u32 (rotor4 / iso4 dense convention).
///
/// `mask = 0xF`; element `e` lands at bits `[e*4 .. e*4+4]`. With 8 elements
/// the highest bit used is `7 * 4 + 3 = 31` — fills the u32 exactly.
#[inline]
fn pack_group_4bit(codes: [u8; ROTOR3_MV_COMPONENTS]) -> u32 {
    debug_assert!(
        codes.iter().all(|&c| c < 16),
        "pack_group_4bit: code out of range [0,15]"
    );
    let mut w: u32 = 0;
    for (e, &c) in codes.iter().enumerate() {
        w |= (u32::from(c) & 0xF) << (e * 4);
    }
    w
}

/// Unpack 1 u32 into 8 codes (inverse of [`pack_group_4bit`]).
#[inline]
fn unpack_group_4bit(word: u32) -> [u8; ROTOR3_MV_COMPONENTS] {
    let mut out = [0_u8; ROTOR3_MV_COMPONENTS];
    for (e, slot) in out.iter_mut().enumerate() {
        *slot = ((word >> (e * 4)) & 0xF) as u8;
    }
    out
}

// ── rotor4 encode ─────────────────────────────────────────────────────────────

/// Encode a V tensor with the rotor4 codec.
///
/// Identical algorithm to [`rotor3_encode`] except:
/// - Uses [`lloyd_gaussian_codebook`]`(4)` (16 centroids).
/// - Packs 8 codes per u32 at 4 bits each (dense iso4 convention — `mask = 0xF`).
///
/// # Arguments
///
/// - `v` — flat f32 slice of shape `[n_tokens, head_dim]`.
/// - `rotors` — static per-(layer, head) rotor table, flat `[n_groups * 4]` f32.
/// - `head_dim` — must be `> 0`.
///
/// # Returns
///
/// `(codes_packed, scales, norms)`:
///
/// - `codes_packed` — 4-bit indices packed at 8 per u32. Length
///   `n_tokens * n_groups * ROTOR4_WORDS_PER_GROUP` (= `n_tokens * n_groups` u32s).
/// - `scales` — per-group f32 scale. Length `n_tokens * n_groups`.
/// - `norms` — per-token L2 norm. Length `n_tokens`.
///
/// # Errors
///
/// Returns [`RotorQuantError`] for invalid inputs.
pub fn rotor4_encode(
    v: &[f32],
    rotors: &[f32],
    head_dim: usize,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>), RotorQuantError> {
    if head_dim == 0 {
        return Err(RotorQuantError::HeadDimZero);
    }
    if !v.len().is_multiple_of(head_dim) {
        return Err(RotorQuantError::LenNotMultipleOfHeadDim {
            len: v.len(),
            head_dim,
        });
    }
    let n_groups = n_groups_for(head_dim);
    let expected = n_groups * 4;
    if rotors.len() != expected {
        return Err(RotorQuantError::RotorTableLen {
            got: rotors.len(),
            expected,
            n_groups,
        });
    }

    let codebook = lloyd_gaussian_codebook(ROTOR4_BITS)
        .map_err(|e| RotorQuantError::Codebook(e.to_string()))?;
    let max_centroid = codebook
        .iter()
        .copied()
        .fold(0.0_f32, |acc, c| acc.max(c.abs()));

    let n_tokens = v.len() / head_dim;
    let words_per_group = ROTOR4_WORDS_PER_GROUP;
    let total_groups = n_tokens * n_groups;

    let mut codes_all: Vec<u32> = Vec::with_capacity(total_groups * words_per_group);
    let mut scales: Vec<f32> = Vec::with_capacity(total_groups);
    let mut norms: Vec<f32> = Vec::with_capacity(n_tokens);

    for tok in 0..n_tokens {
        // tok < n_tokens = v.len() / head_dim guarantees the row slice fits.
        #[allow(
            clippy::indexing_slicing,
            reason = "tok < n_tokens = v.len()/head_dim; row slice is in-bounds"
        )]
        let row = &v[tok * head_dim..(tok + 1) * head_dim];
        // Rounded to the stored sideband precision before use — see
        // `crate::isoquant`: the decode multiplies by the *stored* norm.
        let norm = {
            let sq: f32 = row.iter().map(|&x| x * x).sum();
            bf16_round(sq.sqrt().max(1e-8))
        };
        norms.push(norm);

        for grp in 0..n_groups {
            let r_base = grp * 4;
            // r_base + 3 < rotors.len() == n_groups * 4 by validation above.
            #[allow(
                clippy::indexing_slicing,
                reason = "r_base+3 < n_groups*4 = rotors.len() validated above"
            )]
            let r: Rotor = [
                rotors[r_base],
                rotors[r_base + 1],
                rotors[r_base + 2],
                rotors[r_base + 3],
            ];

            let grp_start = grp * ROTOR4_GROUP_SIZE;
            let mut mv = [0.0_f32; ROTOR3_MV_COMPONENTS];
            for e in 0..ROTOR4_GROUP_SIZE {
                let idx = grp_start + e;
                if idx < head_dim {
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "idx < head_dim = row.len() by construction"
                    )]
                    {
                        mv[e + 1] = row[idx] / norm;
                    }
                }
            }

            // Forward sandwich: R * mv * R̃.
            let rotated = rotor_sandwich(r, &mv);

            let max_abs = rotated
                .iter()
                .copied()
                .fold(0.0_f32, |acc, x| acc.max(x.abs()));
            // Rounded to the stored sideband precision before the codes are
            // chosen against it.
            let scale = bf16_round(if max_abs < 1e-12 {
                1e-12
            } else {
                max_abs / max_centroid
            });
            scales.push(scale);

            let mut group_codes = [0_u8; ROTOR3_MV_COMPONENTS];
            for (e, &rv) in rotated.iter().enumerate() {
                let normalised = rv / scale;
                let idx = codebook
                    .iter()
                    .enumerate()
                    .min_by(|(_, &a), (_, &b)| {
                        (a - normalised).abs().total_cmp(&(b - normalised).abs())
                    })
                    .map_or(0, |(i, _)| i as u8);
                // idx < 16 by construction (codebook has 16 entries); fits 4 bits.
                #[allow(
                    clippy::indexing_slicing,
                    reason = "e < ROTOR3_MV_COMPONENTS = group_codes.len() by loop bound"
                )]
                {
                    group_codes[e] = idx;
                }
            }

            // Pack 8 codes into 1 u32 (dense 4-bit / iso4 convention).
            codes_all.push(pack_group_4bit(group_codes));
        }
    }

    Ok((codes_all, scales, norms))
}

// ── rotor4 decode ─────────────────────────────────────────────────────────────

/// Decode a rotor4-compressed V tensor.
///
/// Reverses [`rotor4_encode`]. All parameters must match the encode call.
///
/// # Arguments
///
/// - `codes_packed` — packed 4-bit indices (8 per u32).
/// - `scales` — per-group f32 scale, length `n_tokens * n_groups`.
/// - `norms` — per-token L2 norm, length `n_tokens`.
/// - `rotors` — static per-(layer, head) rotor table (`n_groups * 4` f32).
/// - `head_dim` — must be `> 0` and match the encode call.
///
/// # Returns
///
/// Dequantised f32 values of length `norms.len() * head_dim`.
///
/// # Errors
///
/// Returns [`RotorQuantError`] for shape/length mismatches.
pub fn rotor4_decode(
    codes_packed: &[u32],
    scales: &[f32],
    norms: &[f32],
    rotors: &[f32],
    head_dim: usize,
) -> Result<Vec<f32>, RotorQuantError> {
    if head_dim == 0 {
        return Err(RotorQuantError::HeadDimZero);
    }
    let n_groups = n_groups_for(head_dim);
    let expected = n_groups * 4;
    if rotors.len() != expected {
        return Err(RotorQuantError::RotorTableLen {
            got: rotors.len(),
            expected,
            n_groups,
        });
    }

    let codebook = lloyd_gaussian_codebook(ROTOR4_BITS)
        .map_err(|e| RotorQuantError::Codebook(e.to_string()))?;

    let n_tokens = norms.len();
    let words_per_group = ROTOR4_WORDS_PER_GROUP;
    let mut out = vec![0.0_f32; n_tokens * head_dim];

    for (tok, &norm) in norms.iter().enumerate() {
        for grp in 0..n_groups {
            let flat_grp = tok * n_groups + grp;
            #[allow(
                clippy::indexing_slicing,
                reason = "flat_grp < n_tokens*n_groups = scales.len() by encode contract"
            )]
            let scale = scales[flat_grp];

            let r_base = grp * 4;
            #[allow(
                clippy::indexing_slicing,
                reason = "r_base+3 < n_groups*4 = rotors.len() validated above"
            )]
            let r: Rotor = [
                rotors[r_base],
                rotors[r_base + 1],
                rotors[r_base + 2],
                rotors[r_base + 3],
            ];

            let word_base = flat_grp * words_per_group;
            #[allow(
                clippy::indexing_slicing,
                reason = "word_base < codes_packed.len() by encode contract"
            )]
            let word = codes_packed[word_base];
            let codes = unpack_group_4bit(word);

            let mut mv_q = [0.0_f32; ROTOR3_MV_COMPONENTS];
            for (slot, &c) in mv_q.iter_mut().zip(codes.iter()) {
                // c < 16 = codebook.len() by construction (pack mask 0xF).
                #[allow(
                    clippy::indexing_slicing,
                    reason = "c < 2^4 = codebook.len() by pack mask (0xF)"
                )]
                let centroid = codebook[c as usize];
                *slot = centroid * scale;
            }

            // Inverse sandwich: R̃ * mv_q * R.
            let restored = rotor_sandwich(rotor_reverse(r), &mv_q);

            let out_row = tok * head_dim;
            let grp_start = grp * ROTOR4_GROUP_SIZE;
            for e in 0..ROTOR4_GROUP_SIZE {
                let idx = grp_start + e;
                if idx < head_dim {
                    #[allow(
                        clippy::indexing_slicing,
                        reason = "out_row + idx < n_tokens*head_dim = out.len() by construction"
                    )]
                    {
                        out[out_row + idx] = restored[e + 1] * norm;
                    }
                }
            }
        }
    }

    Ok(out)
}

// ── K-side rotor encoders with optional QJL residual ─────────────────────────

/// Deterministic seed for the QJL projection matrix (Python reference uses
/// `seed + 1` where the rotor MSE seed is `seed`; rMLX uses a single global
/// QJL seed since the rotor table seed is per-(layer, head) via
/// `crate::clifford::make_rotor_table`).
pub const ROTOR_QJL_PROJECTION_SEED: u64 = 0x52_4f_54_4f_52_5f_51_4a; // "ROTOR_QJ"

/// Build the QJL projection matrix `S` of shape `[qjl_dim, head_dim]` seeded
/// deterministically. Mirrors the Python `RotorQuantProd.__init__` Gaussian
/// matrix at `gen.manual_seed(seed + 1)` — but uses an xorshift64-based
/// `randn`-equivalent for cross-process determinism without bringing in a
/// stdlib RNG dependency.
///
/// `qjl_dim` defaults to `head_dim` (matches Python reference). The returned
/// flat layout is row-major: row `i` of length `head_dim` starts at index
/// `i * head_dim`.
#[must_use]
pub fn make_qjl_projection(head_dim: usize) -> Vec<f32> {
    let qjl_dim = head_dim;
    let mut out = vec![0.0_f32; qjl_dim * head_dim];
    let mut state: u64 = ROTOR_QJL_PROJECTION_SEED;
    for slot in &mut out {
        // Box-Muller via two xorshift64-derived uniforms — gives N(0, 1).
        let u1 = next_uniform(&mut state).max(f32::MIN_POSITIVE);
        let u2 = next_uniform(&mut state);
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = 2.0_f32 * std::f32::consts::PI * u2;
        *slot = r * theta.cos();
    }
    out
}

/// xorshift64* — small deterministic PRNG used to seed the QJL projection.
#[inline]
fn next_uniform(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    // top 24 bits → [0, 1)
    ((r >> 40) as f32) / ((1u32 << 24) as f32)
}

/// Pack 1-bit sign values into a u8 row-major buffer.
///
/// Bit order per Python `rotorquant.py:233-234` — `torch.sign(projected)`
/// returns ±1 floats per element. We pack **8 elements per byte, LSB =
/// element 0**: element `e` lands at `bytes[i] >> (e & 7) & 1`. `1` = positive,
/// `0` = negative. Zero residuals are clamped to positive (matches Python
/// `qjl_signs[qjl_signs == 0] = 1.0`).
///
/// Output length per token = `ceil(qjl_dim / 8)`.
#[must_use]
pub fn pack_qjl_signs(signs: &[f32]) -> Vec<u8> {
    let bytes = signs.len().div_ceil(8);
    let mut out = vec![0_u8; bytes];
    for (i, &s) in signs.iter().enumerate() {
        let bit = u8::from(s >= 0.0);
        let byte = i / 8;
        let shift = i % 8;
        // byte < out.len() by div_ceil construction.
        #[allow(
            clippy::indexing_slicing,
            reason = "byte = i/8 < signs.len().div_ceil(8) = out.len() by construction"
        )]
        {
            out[byte] |= bit << shift;
        }
    }
    out
}

/// Unpack a u8 sign buffer back to f32 ±1 values.
///
/// Inverse of [`pack_qjl_signs`]. The `len` argument carries the exact
/// `qjl_dim` (= `head_dim`) so trailing pad bits in the last byte are
/// discarded.
#[must_use]
pub fn unpack_qjl_signs(packed: &[u8], len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let byte = i / 8;
        let shift = i % 8;
        let bit = packed.get(byte).map_or(1, |b| (b >> shift) & 1);
        out.push(if bit == 1 { 1.0_f32 } else { -1.0_f32 });
    }
    out
}

/// Compute the QJL projection of one token's `[head_dim]` residual vector
/// against the projection matrix `S` of shape `[qjl_dim, head_dim]` (flat
/// row-major) and return the packed sign bytes.
///
/// The projection step is `signs = sign(S @ residual)`. This is one matrix-
/// vector multiply per token — `O(qjl_dim * head_dim)` cost per encode call.
#[must_use]
pub fn qjl_encode_residual(residual: &[f32], s_matrix: &[f32], head_dim: usize) -> Vec<u8> {
    let qjl_dim = s_matrix.len() / head_dim;
    let mut signs = Vec::with_capacity(qjl_dim);
    for i in 0..qjl_dim {
        let row_base = i * head_dim;
        let mut acc = 0.0_f32;
        for j in 0..head_dim {
            // row_base+j < s_matrix.len() = qjl_dim * head_dim by construction.
            #[allow(
                clippy::indexing_slicing,
                reason = "row_base+j < qjl_dim*head_dim = s_matrix.len() by loop bound"
            )]
            {
                acc += s_matrix[row_base + j] * residual.get(j).copied().unwrap_or(0.0);
            }
        }
        signs.push(if acc >= 0.0 { 1.0_f32 } else { -1.0_f32 });
    }
    pack_qjl_signs(&signs)
}

/// Encode a K tensor with the rotor3 codec and an optional 1-bit
/// QJL residual stage.
///
/// Identical algorithm to [`rotor3_encode`] for the rotor MSE stage. When
/// `qjl_s_matrix` is `Some(&S)`, a per-token JL projection `sign(S @ (x - x_hat))`
/// is captured as 1 bit per `qjl_dim` element and returned alongside the
/// rotor codes. The residual L2 norm is also returned per token (separate from
/// the rotor-MSE per-token norm — that one is the pre-normalisation L2 of `x`).
///
/// Returns `(codes_packed, scales, norms, qjl_packed, qjl_norms)`:
/// - `codes_packed` / `scales` / `norms` — identical layout to
///   [`rotor3_encode`].
/// - `qjl_packed` — packed 1-bit signs, length `n_tokens * ceil(qjl_dim / 8)`
///   bytes; empty when `qjl_s_matrix` is `None`.
/// - `qjl_norms` — per-token residual L2 norm (length `n_tokens`); empty when
///   `qjl_s_matrix` is `None`.
///
/// # Errors
///
/// Returns [`RotorQuantError`] for invalid inputs (zero `head_dim`,
/// length mismatch, rotor table size mismatch, codebook fault).
#[allow(clippy::type_complexity)]
pub fn rotor3_k_encode(
    k: &[f32],
    rotors: &[f32],
    head_dim: usize,
    qjl_s_matrix: Option<&[f32]>,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<f32>), RotorQuantError> {
    rotor_k_encode_inner(k, rotors, head_dim, qjl_s_matrix, ROTOR3_BITS)
}

/// Encode a K tensor with the rotor4 codec and an optional 1-bit QJL residual
/// stage. Mirror of [`rotor3_k_encode`] with `bits=4`.
#[allow(clippy::type_complexity)]
pub fn rotor4_k_encode(
    k: &[f32],
    rotors: &[f32],
    head_dim: usize,
    qjl_s_matrix: Option<&[f32]>,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<f32>), RotorQuantError> {
    rotor_k_encode_inner(k, rotors, head_dim, qjl_s_matrix, ROTOR4_BITS)
}

/// Decode a rotor3-K compressed tensor with optional QJL residual.
///
/// When `qjl_packed` is non-empty and `qjl_s_matrix` is `Some(&S)`, the per-
/// token QJL correction is added to the rotor3-reconstructed values; otherwise
/// behaves identically to [`rotor3_decode`].
pub fn rotor3_k_decode(
    codes_packed: &[u32],
    scales: &[f32],
    norms: &[f32],
    rotors: &[f32],
    head_dim: usize,
    qjl_packed: &[u8],
    qjl_norms: &[f32],
    qjl_s_matrix: Option<&[f32]>,
) -> Result<Vec<f32>, RotorQuantError> {
    let mut out = rotor3_decode(codes_packed, scales, norms, rotors, head_dim)?;
    apply_qjl_correction(&mut out, head_dim, qjl_packed, qjl_norms, qjl_s_matrix);
    Ok(out)
}

/// Decode a rotor4-K compressed tensor with optional QJL residual.
pub fn rotor4_k_decode(
    codes_packed: &[u32],
    scales: &[f32],
    norms: &[f32],
    rotors: &[f32],
    head_dim: usize,
    qjl_packed: &[u8],
    qjl_norms: &[f32],
    qjl_s_matrix: Option<&[f32]>,
) -> Result<Vec<f32>, RotorQuantError> {
    let mut out = rotor4_decode(codes_packed, scales, norms, rotors, head_dim)?;
    apply_qjl_correction(&mut out, head_dim, qjl_packed, qjl_norms, qjl_s_matrix);
    Ok(out)
}

/// Internal common path for rotor-K encode: runs the rotor MSE forward, then
/// (when `qjl_s_matrix` is `Some`) captures the residual = original − recon
/// and projects to qjl signs.
#[allow(clippy::type_complexity)]
fn rotor_k_encode_inner(
    k: &[f32],
    rotors: &[f32],
    head_dim: usize,
    qjl_s_matrix: Option<&[f32]>,
    bits: u8,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<u8>, Vec<f32>), RotorQuantError> {
    let (codes, scales, norms) = if bits == ROTOR3_BITS {
        rotor3_encode(k, rotors, head_dim)?
    } else {
        rotor4_encode(k, rotors, head_dim)?
    };

    let mut qjl_packed: Vec<u8> = Vec::new();
    let mut qjl_norms: Vec<f32> = Vec::new();
    if let Some(s_matrix) = qjl_s_matrix {
        let recon = if bits == ROTOR3_BITS {
            rotor3_decode(&codes, &scales, &norms, rotors, head_dim)?
        } else {
            rotor4_decode(&codes, &scales, &norms, rotors, head_dim)?
        };
        let n_tokens = norms.len();
        let qjl_bytes_per_tok = head_dim.div_ceil(8);
        qjl_packed.reserve(n_tokens * qjl_bytes_per_tok);
        qjl_norms.reserve(n_tokens);
        for tok in 0..n_tokens {
            let row_start = tok * head_dim;
            let row_end = row_start + head_dim;
            // row_end ≤ k.len() because k.len() = n_tokens*head_dim by encode contract.
            #[allow(
                clippy::indexing_slicing,
                reason = "row_start+head_dim ≤ k.len() by encode contract validated above"
            )]
            let orig = &k[row_start..row_end];
            #[allow(
                clippy::indexing_slicing,
                reason = "same: recon.len() == k.len() by rotor decode contract"
            )]
            let recon_row = &recon[row_start..row_end];
            let mut residual = vec![0.0_f32; head_dim];
            let mut norm_sq = 0.0_f32;
            for j in 0..head_dim {
                // bounds OK by row_start+j < row_end ≤ k.len(); residual[j] j<head_dim.
                #[allow(
                    clippy::indexing_slicing,
                    reason = "j < head_dim = orig.len() = residual.len() by construction"
                )]
                {
                    residual[j] = orig[j] - recon_row[j];
                    norm_sq += residual[j] * residual[j];
                }
            }
            qjl_norms.push(norm_sq.sqrt());
            let packed = qjl_encode_residual(&residual, s_matrix, head_dim);
            qjl_packed.extend_from_slice(&packed);
        }
    }

    Ok((codes, scales, norms, qjl_packed, qjl_norms))
}

/// Apply the QJL residual correction to an already-rotor-decoded f32 buffer
/// in place — one per-token correction vector per row.
///
/// **Score-time correction via dequant-side residual-add:**
///
/// The Python `RotorQuantProd.inner_product` computes a score-time correction
///
/// ```text
///   term2 = ||r|| · (sqrt(pi/2) / m) · (y @ S.T · qjl_signs).sum()
/// ```
///
/// which is linear in `y`. We exploit that linearity to express the same
/// scalar as a **per-token K-side residual add**:
///
/// ```text
///   delta_k[t, j] = ||r_t|| · (sqrt(pi/2) / m) · sum_i ( S[i, j] · signs[t, i] )
///   Q · (K_dequant + delta_k) = Q · K_dequant + ||r_t|| · scale · (Q · S.T · signs[t])
/// ```
///
/// The right-hand side equals the Python score-time `term1 + term2` token-by-
/// token at full precision. This lets the correction live entirely inside the
/// codec — the engine continues to call `update()` + `scaled_dot_product_attention`
/// unchanged. The boundary contract `rmlx-kv-quant → rmlx-models` (one-way)
/// is preserved.
///
/// # Per-K cosine vs per-score correctness
///
/// Prior analysis flagged that this exact dequant-side residual-add
/// **lowers per-K cosine** by ~0.002 on the synthetic LCG fixture at
/// head_dim=128 (correction-on adds variance noise that exceeds the small
/// per-element bias correction). That observation is real but the wrong gate
/// for SDPA quality: per-K cosine measures `cos(K_corrected, K_true)`, while
/// the score-time quantity that drives attention is `Q · K`. The JL sketch is
/// **unbiased** for the inner product (sketch: see Indyk-Motwani 1998, QJL
/// arxiv.org/abs/2406.03482) so adding the per-token residual estimate to K
/// gives the same *expected* score as the Python score-time formula, with
/// variance shrinking as `O(||r||^2 / m)`. Cosine on the final SDPA output
/// (output-logit-lift on real models) is the meaningful gate; documented in
/// `docs/KV_QUANT.md`.
///
/// # Layout contract
///
/// * `out` — length `n_tokens * head_dim`, row-major. Already populated with
///   the rotor MSE reconstruction by [`rotor3_decode`] / [`rotor4_decode`].
/// * `qjl_packed` — `n_tokens * ceil(head_dim / 8)` bytes, packed by
///   [`pack_qjl_signs`].
/// * `qjl_norms` — `n_tokens` f32, per-token residual L2 norm from encode.
/// * `qjl_s_matrix` — `head_dim * head_dim` f32, JL projection from
///   [`make_qjl_projection`].
///
/// A `None` projection matrix or empty `qjl_packed` is a no-op (the QJL
/// sideband was disabled at first `append`).
fn apply_qjl_correction(
    out: &mut [f32],
    head_dim: usize,
    qjl_packed: &[u8],
    qjl_norms: &[f32],
    qjl_s_matrix: Option<&[f32]>,
) {
    let Some(s_matrix) = qjl_s_matrix else {
        return;
    };
    if qjl_packed.is_empty() || qjl_norms.is_empty() || head_dim == 0 {
        return;
    }
    let qjl_dim = s_matrix.len() / head_dim;
    if qjl_dim == 0 {
        return;
    }
    let bytes_per_tok = qjl_dim.div_ceil(8);
    if bytes_per_tok == 0 {
        return;
    }
    let n_tokens = qjl_norms.len();
    debug_assert_eq!(
        qjl_packed.len(),
        n_tokens * bytes_per_tok,
        "QJL packed bytes ({}) != n_tokens ({}) * bytes_per_tok ({})",
        qjl_packed.len(),
        n_tokens,
        bytes_per_tok,
    );
    debug_assert_eq!(
        out.len(),
        n_tokens * head_dim,
        "QJL out buffer length ({}) != n_tokens ({}) * head_dim ({})",
        out.len(),
        n_tokens,
        head_dim,
    );

    let correction_scale = (std::f32::consts::PI / 2.0_f32).sqrt() / (qjl_dim as f32);

    // Same back-projection math as `qjl_decode_correction` — keep in sync.
    // Kept inline here (vs delegating per token) because per-token Vec
    // allocations on the hot decode path are not acceptable.
    let mut delta = vec![0.0_f32; head_dim];
    for tok in 0..n_tokens {
        let norm = qjl_norms.get(tok).copied().unwrap_or(0.0);
        if norm == 0.0 {
            continue;
        }
        let packed_start = tok * bytes_per_tok;
        let packed_end = packed_start + bytes_per_tok;
        if packed_end > qjl_packed.len() {
            tracing::warn!(
                tok,
                n_tokens,
                packed_len = qjl_packed.len(),
                "apply_qjl_correction: packed range out-of-bounds; aborting partial correction"
            );
            return;
        }
        // Bounds verified above; .unwrap_or(&[]) is the unreachable defensive
        // fallback.
        let packed_row = qjl_packed.get(packed_start..packed_end).unwrap_or(&[]);
        let signs = unpack_qjl_signs(packed_row, qjl_dim);

        let row_base = tok * head_dim;
        let coeff = norm * correction_scale;

        // Cache-friendly accumulation: outer `i`, inner `j` walks each
        // `s_matrix` row contiguously and avoids a head_dim-stride per inner
        // step. Bit-equivalent to the prior `j`-outer form modulo f32
        // reorder (verified by `qjl_residual_add_matches_score_time_correction`).
        delta.fill(0.0);
        for i in 0..qjl_dim {
            // i < qjl_dim = signs.len() by `unpack_qjl_signs` contract.
            #[allow(
                clippy::indexing_slicing,
                reason = "i < qjl_dim = signs.len() by unpack_qjl_signs contract"
            )]
            let s_i = signs[i];
            // i*head_dim+(head_dim-1) < qjl_dim*head_dim = s_matrix.len() by loop bound.
            #[allow(
                clippy::indexing_slicing,
                reason = "(i+1)*head_dim ≤ qjl_dim*head_dim = s_matrix.len() by loop bound"
            )]
            let s_row = &s_matrix[i * head_dim..(i + 1) * head_dim];
            for j in 0..head_dim {
                // j < head_dim = delta.len(), j < head_dim = s_row.len().
                #[allow(
                    clippy::indexing_slicing,
                    reason = "j < head_dim = delta.len() = s_row.len() by loop bound"
                )]
                {
                    delta[j] += s_row[j] * s_i;
                }
            }
        }
        for j in 0..head_dim {
            // row_base + j < (tok+1)*head_dim ≤ n_tokens*head_dim = out.len()
            // by `tok < n_tokens` and debug-assert above.
            #[allow(
                clippy::indexing_slicing,
                reason = "row_base+j < n_tokens*head_dim = out.len() by debug_assert above"
            )]
            {
                out[row_base + j] += coeff * delta[j];
            }
        }
    }
}

#[cfg(test)]
#[path = "rotorquant_tests.rs"]
mod rotorquant_tests;
