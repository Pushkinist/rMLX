//! iso3 / iso4 KV-cache codec — Quaternion SO(4) 4D block rotation + 3- or 4-bit quantization.
//!
//! # Algorithm overview
//!
//! IsoQuant applies an **isoclinic SO(4) rotation** (fast mode: `T(v) = q_L * v`)
//! to groups of 4 elements before Lloyd-Max quantization.
//!
//! iso4 parameterizes the existing iso3 encoder/decoder over
//! `bits ∈ {3, 4}`. The 4-bit path (iso4) is a trivial follow-on: it uses
//! [`lloyd_gaussian_codebook(4)`] (16 centroids) and the dense 8-vals-per-u32
//! pack (32 bits used, 0 wasted) instead of iso3's 10-vals-per-u32 pack (30
//! bits used, 2 wasted). All other algorithmic steps — rotation, scale, group
//! layout, quaternion / norm storage — are identical.
//!
//! Pipeline per token:
//!   1. L2-normalise the vector; store the norm separately.
//!   2. Reshape into `head_dim / group_size` groups of `group_size` elements.
//!      Each group of 4 elements is treated as a quaternion `v = (w, x, y, z)`.
//!   3. Apply Hamilton product `r = q_L * v` where `q_L` is the fixed golden-ratio
//!      unit quaternion (see [`FIXED_QUAT`]).
//!   4. Within each group, compute a per-group scale (`max|r_i| / max_centroid`)
//!      and quantize all elements against the `bits`-bit Lloyd-Max N(0,1) codebook.
//!   5. Pack `32/bits` codes per u32 (`vals_per_word(bits)`) — same convention as
//!      Planar3 pack convention to enable future kernel reuse.
//!
//! Dequantize reverses steps 5→1:
//!   1. Unpack codes → centroid lookup → rescale by per-group scale.
//!   2. Apply inverse rotation `q̄_L * r` (Hamilton product with conjugate).
//!   3. Rescale by the stored norm.
//!
//! # Codebook divergence — rMLX uses Gaussian Lloyd, Python uses Beta Lloyd
//!
//! The Python reference implementations (`rotorquant/turboquant/lloyd_max.py`)
//! solve Lloyd-Max for the **Beta distribution** that arises from randomly
//! rotating a d-dimensional unit vector. rMLX reuses the existing
//! `lloyd_gaussian_codebook(3)` (Lloyd-Max for N(0,1)) to avoid introducing a
//! new codebook solver and to stay consistent with TurboQuant and PlanarQuant.
//!
//! For `head_dim = 128`, the Beta(d=128) distribution is extremely well
//! approximated by N(0, 1/d), and after the per-group scale step the effective
//! range seen by the codebook is N(0,1) regardless. The quality gap versus
//! Beta-codebook Python is therefore negligible in practice. The Python mtq
//! benchmark reports cosine ≈ 0.9783 on realistic KV vectors; rMLX measurands
//! on LCG fixture are documented in `isoquant_tests.rs`.
//!
//! # Fixed quaternion
//!
//! We use the golden-ratio-based unit quaternion from `multi-turboquant`
//! (`multi_turboquant/methods/isoquant.py`):
//!   `q = (1, φ, φ−1, 1) / ||(1, φ, φ−1, 1)||`
//! where `φ = (1 + √5) / 2`. This quaternion maximises channel decorrelation
//! without calibration or per-group fitting. The same quaternion is applied to
//! every group (fast-path; T11b will add per-group optimised quaternions).
//!
//! # Bit-packing
//!
//! Generic packer: `vals_per_word = 32 / bits`. Element `e` within a group maps
//! to `word = e / vals_per_word`, `shift = (e % vals_per_word) * bits`.
//!
//! - `bits=3`: 10 vals/u32, 30 bits used, 2 wasted — Planar3 pack convention.
//! - `bits=4`: 8 vals/u32, 32 bits used, 0 wasted — dense packing for iso4.
//!
//! # Wire-up status
//!
//! - **iso3** — CPU codec + storage variant + SDPA fallthrough + SSD spill
//!   + MSL kernel hook (CPU primary; GPU `#[ignore]`-gated).
//! - **iso4** — parameterizes `iso_encode_fast`/`iso_decode_fast` over
//!   `bits ∈ {3,4}`. iso4 ships **CPU-only** (no MSL kernel) — the existing
//!   MSL is hard-coded for `bits=3`; an iso4 MSL variant is deferred.

use crate::storage::bf16_round;
use crate::turboquant::lloyd_gaussian_codebook;
use thiserror::Error;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Golden-ratio-based unit quaternion `[w, x, y, z]`.
///
/// Derived from `q_raw = (1, φ, φ−1, 1)` where `φ = (1+√5)/2 ≈ 1.6180339887`.
/// Normalised to unit length. Same quaternion used by `multi_turboquant/methods/isoquant.py`.
///
/// Applied as fast-mode left isoclinic rotation: `T(v) = q_L * v`.
pub const FIXED_QUAT: [f32; 4] = {
    // φ = (1 + √5) / 2.  We need compile-time f32 constants.
    // Pre-computed:
    //   q_raw = [1.0, 1.6180339887, 0.6180339887, 1.0]
    //   norm  = sqrt(1 + φ² + (φ−1)² + 1)
    //         = sqrt(1 + 2.6180339887 + 0.3819660113 + 1)
    //         = sqrt(5.0) ≈ 2.2360679775
    //   q     = q_raw / norm
    // Values rounded to nearest f32:
    const PHI: f32 = 1.618_033_9;
    const PHI_M1: f32 = 0.618_033_9;
    const NORM: f32 = 2.236_067_8; // sqrt(5)
    [1.0 / NORM, PHI / NORM, PHI_M1 / NORM, 1.0 / NORM]
};

/// Number of values packed per u32 word for `bits`.
///
/// - `bits=3` → 10 vals/u32 (30 bits used, 2 wasted — Planar3 convention).
/// - `bits=4` → 8 vals/u32 (32 bits used, 0 wasted — dense iso4 packing).
///
/// Defined as `32 / bits`; only 3 and 4 are validated codec values.
#[inline]
fn vals_per_word(bits: u8) -> usize {
    debug_assert!(bits == 3 || bits == 4, "vals_per_word: bits must be 3 or 4");
    (32 / bits) as usize
}

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors from the iso3 / iso4 codec.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed codec-internal error enum; T11b/c/d may add variants but this is not a public extension point"
)]
#[derive(Debug, Error)]
pub enum IsoQuantError {
    /// `head_dim` is not a multiple of 4 (quaternion block alignment).
    ///
    /// iso3 processes vectors in groups of 4 elements (quaternion representation).
    /// Any `head_dim` not divisible by 4 cannot be partitioned into clean groups.
    #[error("isoquant: head_dim={head_dim} is not a multiple of 4 (quaternion alignment)")]
    HeadDimNotMultipleOf4 {
        /// The offending head dimension.
        head_dim: usize,
    },

    /// `group_size` is not a multiple of 4 or not positive.
    #[error("isoquant: group_size={group_size} must be a positive multiple of 4")]
    InvalidGroupSize {
        /// The offending group size.
        group_size: usize,
    },

    /// `head_dim` is not a multiple of `group_size`.
    ///
    /// iso3 partitions `head_dim` elements into `head_dim / group_size` groups.
    /// If `group_size` does not divide `head_dim` evenly, the last group would be
    /// incomplete and `n_groups` would be zero or truncated.
    #[error(
        "isoquant: head_dim={head_dim} is not a multiple of group_size={group_size} \
         (head_dim % group_size = {rem})",
        rem = head_dim % group_size
    )]
    GroupSizeNotDivisorOfHeadDim {
        /// The group size that does not divide `head_dim`.
        group_size: usize,
        /// The head dimension.
        head_dim: usize,
    },

    /// `bits` is not supported (only 3 and 4 are supported).
    #[error(
        "isoquant: bits={bits} not supported; only bits=3 (iso3) and bits=4 (iso4) are supported"
    )]
    UnsupportedBits {
        /// The offending bit width.
        bits: u8,
    },

    /// The underlying Lloyd-Max codebook lookup failed.
    #[error("isoquant: codebook error: {0}")]
    Codebook(String),

    /// Input slice length is not a multiple of `head_dim`.
    #[error("isoquant: v.len()={len} is not a multiple of head_dim={head_dim}")]
    LenNotMultipleOfHeadDim {
        /// The actual slice length.
        len: usize,
        /// The expected head dimension divisor.
        head_dim: usize,
    },
}

// ── Hamilton product ──────────────────────────────────────────────────────────

/// Pure-scalar Hamilton product of two quaternions `a * b`.
///
/// Both quaternions use the `[w, x, y, z]` convention.
///
/// ```text
/// rw = aw*bw − ax*bx − ay*by − az*bz
/// rx = aw*bx + ax*bw + ay*bz − az*by
/// ry = aw*by − ax*bz + ay*bw + az*bx
/// rz = aw*bz + ax*by − ay*bx + az*bw
/// ```
///
/// Exact translation of `quat_multiply` from
/// `rotorquant/turboquant/isoquant.py` (16 multiplies + 12 adds).
/// Used as the correctness reference for the bit-exact Python fixture tests
/// in `isoquant_tests.rs`.
#[inline]
pub fn quat_multiply(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw.mul_add(bw, ax.mul_add(-bx, ay.mul_add(-by, -az * bz))),
        aw.mul_add(bx, ax.mul_add(bw, ay.mul_add(bz, -az * by))),
        aw.mul_add(by, ax.mul_add(-bz, ay.mul_add(bw, az * bx))),
        aw.mul_add(bz, ax.mul_add(by, ay.mul_add(-bx, az * bw))),
    ]
}

/// Quaternion conjugate: `(w, x, y, z) → (w, −x, −y, −z)`.
#[inline]
pub fn quat_conjugate(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

// ── Bit packing helpers ───────────────────────────────────────────────────────

/// Pack `bits`-wide indices into u32 words at `32 / bits` values per word.
///
/// Element `e` within a flat sequence of `n_codes` indices maps to:
///   `word  = e / vals_per_word(bits)`
///   `shift = (e % vals_per_word(bits)) * bits`
///
/// `bits=3` matches the Planar3 pack convention; `bits=4` is the dense iso4
/// pack. Higher bits are not used by this codec — caller must pass 3 or 4.
fn pack_bits(indices: &[u8], bits: u8) -> Vec<u32> {
    let vpw = vals_per_word(bits);
    let mask: u32 = (1u32 << bits) - 1;
    let n_words = indices.len().div_ceil(vpw);
    let mut words = vec![0u32; n_words];
    for (e, &idx) in indices.iter().enumerate() {
        let word = e / vpw;
        let shift = (e % vpw) * (bits as usize);
        // word < n_words by construction: word = e / vpw < div_ceil(len, vpw) = n_words.
        #[allow(
            clippy::indexing_slicing,
            reason = "word = e / vpw < div_ceil(indices.len(), vpw) = n_words = words.len()"
        )]
        {
            words[word] |= (u32::from(idx) & mask) << shift;
        }
    }
    words
}

/// Unpack `bits`-wide indices from u32 words (`32 / bits` values per word).
fn unpack_bits(words: &[u32], n_codes: usize, bits: u8) -> Vec<u8> {
    let vpw = vals_per_word(bits);
    let mask: u32 = (1u32 << bits) - 1;
    let mut indices = Vec::with_capacity(n_codes);
    for e in 0..n_codes {
        let word = e / vpw;
        let shift = (e % vpw) * (bits as usize);
        // word < words.len() by construction: caller ensures words.len() == div_ceil(n_codes, vpw).
        #[allow(
            clippy::indexing_slicing,
            reason = "word = e / vpw < div_ceil(n_codes, vpw) = words.len()"
        )]
        let idx = ((words[word] >> shift) & mask) as u8;
        indices.push(idx);
    }
    indices
}

// ── Encode ────────────────────────────────────────────────────────────────────

/// Encode V tensor with iso3 / iso4: quaternion rotation + `bits`-bit Lloyd-Max
/// quantization.
///
/// # Arguments
///
/// - `v` — flat f32 slice of shape `[n_tokens, head_dim]`.
/// - `head_dim` — must be a positive multiple of 4.
/// - `group_size` — elements per quantization group; must be a positive multiple of 4
///   and divide `head_dim` evenly. Typically 4 (one quaternion block per group).
/// - `bits` — `3` for iso3 (10 vals/u32, Planar3 pack) or `4` for iso4
///   (8 vals/u32, dense pack — iso4).
///
/// # Returns
///
/// `(codes_packed, scales, quaternions, norms)` where:
/// - `codes_packed` — `bits`-bit indices, `32 / bits` per u32.
///   Length: `n_tokens * n_groups * ceil(group_size * bits / 32)` words.
/// - `scales` — per-group f32 scale, one per `(token, group)`.
///   Length: `n_tokens * n_groups`.
/// - `quaternions` — per-group unit quaternion `[w, x, y, z]` f32.
///   Length: `n_tokens * n_groups * 4`.
///   In the current fixed-rotation implementation all entries are [`FIXED_QUAT`].
/// - `norms` — per-token L2 norm, f32.
///   Length: `n_tokens`.
///
/// # Errors
///
/// Returns [`IsoQuantError`] for invalid `head_dim`, `group_size`, or `bits`.
pub fn iso_encode_fast(
    v: &[f32],
    head_dim: usize,
    group_size: usize,
    bits: u8,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>), IsoQuantError> {
    // ── Validate inputs ───────────────────────────────────────────────────────
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(IsoQuantError::HeadDimNotMultipleOf4 { head_dim });
    }
    if group_size == 0 || !group_size.is_multiple_of(4) {
        return Err(IsoQuantError::InvalidGroupSize { group_size });
    }
    if !head_dim.is_multiple_of(group_size) {
        return Err(IsoQuantError::GroupSizeNotDivisorOfHeadDim {
            group_size,
            head_dim,
        });
    }
    if bits != 3 && bits != 4 {
        return Err(IsoQuantError::UnsupportedBits { bits });
    }
    if !v.len().is_multiple_of(head_dim) {
        return Err(IsoQuantError::LenNotMultipleOfHeadDim {
            len: v.len(),
            head_dim,
        });
    }

    let codebook =
        lloyd_gaussian_codebook(bits).map_err(|e| IsoQuantError::Codebook(e.to_string()))?;
    let max_centroid = codebook
        .iter()
        .copied()
        .fold(0.0_f32, |acc, c| acc.max(c.abs()));

    let n_tokens = v.len() / head_dim;
    let n_groups = head_dim / group_size;
    let words_per_group = group_size.div_ceil(vals_per_word(bits));

    let total_groups = n_tokens * n_groups;
    let total_words = total_groups * words_per_group;

    let mut codes_all = Vec::with_capacity(total_words);
    let mut scales = Vec::with_capacity(total_groups);
    let mut quaternions = Vec::with_capacity(total_groups * 4);
    let mut norms = Vec::with_capacity(n_tokens);

    let q_l = FIXED_QUAT;

    for tok in 0..n_tokens {
        // ── Normalise by L2 norm ──────────────────────────────────────────────
        // tok < n_tokens = v.len() / head_dim; so tok*head_dim .. (tok+1)*head_dim is in bounds.
        #[allow(
            clippy::indexing_slicing,
            reason = "tok < n_tokens = v.len()/head_dim; row slice is always in-bounds"
        )]
        let row = &v[tok * head_dim..(tok + 1) * head_dim];
        // Rounded to the stored sideband precision before it is used: the
        // decode multiplies by the *stored* norm, so quantizing against a
        // finer one would bake in an error the store cannot represent.
        let norm = {
            let sq: f32 = row.iter().map(|&x| x * x).sum();
            bf16_round(sq.sqrt().max(1e-8))
        };
        norms.push(norm);

        for grp in 0..n_groups {
            let grp_start = grp * group_size;

            // Push quaternion (same fixed quat for every group in this step).
            quaternions.extend_from_slice(&q_l);

            // Process in quaternion-blocks of 4.
            let mut group_indices = Vec::with_capacity(group_size);
            let mut group_scale = 0.0_f32;

            // Pass 1: rotate and find per-group max-abs for scale.
            let mut rotated_buf = Vec::with_capacity(group_size);
            for qblk in 0..group_size / 4 {
                let base = grp_start + qblk * 4;
                // base + 3 = grp_start + qblk*4 + 3 < grp_start + group_size <= head_dim == row.len().
                #[allow(
                    clippy::indexing_slicing,
                    reason = "base+3 < grp_start+group_size <= head_dim = row.len()"
                )]
                let unit_block = [
                    row[base] / norm,
                    row[base + 1] / norm,
                    row[base + 2] / norm,
                    row[base + 3] / norm,
                ];
                let r = quat_multiply(q_l, unit_block);
                rotated_buf.extend_from_slice(&r);
                for &rv in &r {
                    let abs = rv.abs();
                    if abs > group_scale {
                        group_scale = abs;
                    }
                }
            }

            // Scale = max_abs / max_centroid (same convention as
            // TurboQuant/PlanarQuant), rounded to the stored sideband
            // precision before the codes are chosen against it.
            let scale = bf16_round(if group_scale < 1e-12 {
                1e-12
            } else {
                group_scale / max_centroid
            });
            scales.push(scale);

            // Pass 2: quantize.
            for &rv in &rotated_buf {
                let normalized = rv / scale;
                // Nearest centroid lookup. Use total_cmp so NaN in codebook or
                // normalized (from ±Inf overflow) yields a defined ordering rather
                // than silently picking an arbitrary centroid.
                let idx = codebook
                    .iter()
                    .enumerate()
                    .min_by(|(_, &a), (_, &b)| {
                        (a - normalized).abs().total_cmp(&(b - normalized).abs())
                    })
                    .map_or(0, |(i, _)| i as u8);
                group_indices.push(idx);
            }

            // Pack `bits`-bit indices.
            let packed = pack_bits(&group_indices, bits);
            codes_all.extend_from_slice(&packed);
        }
    }

    Ok((codes_all, scales, quaternions, norms))
}

// ── Decode ────────────────────────────────────────────────────────────────────

/// Decode iso3 / iso4-compressed V tensor.
///
/// Reverses [`iso_encode_fast`]. All parameters must match the encode call.
///
/// # Arguments
///
/// - `codes_packed` — `bits`-bit indices packed at `32 / bits` per u32.
/// - `scales` — per-group f32 scale. Length: `n_tokens * n_groups`.
/// - `quaternions` — per-group unit quaternion `[w, x, y, z]`. Length: `n_tokens * n_groups * 4`.
/// - `norms` — per-token L2 norm. Length: `n_tokens`.
/// - `head_dim` — must be a positive multiple of 4.
/// - `group_size` — must be a positive multiple of 4 and divide `head_dim`.
/// - `bits` — `3` (iso3) or `4` (iso4).
///
/// # Returns
///
/// Dequantized f32 values of length `norms.len() * head_dim`.
///
/// # Errors
///
/// Returns [`IsoQuantError`] for invalid `head_dim`, `group_size`, or `bits`.
pub fn iso_decode_fast(
    codes_packed: &[u32],
    scales: &[f32],
    quaternions: &[f32],
    norms: &[f32],
    head_dim: usize,
    group_size: usize,
    bits: u8,
) -> Result<Vec<f32>, IsoQuantError> {
    // ── Validate inputs ───────────────────────────────────────────────────────
    if head_dim == 0 || !head_dim.is_multiple_of(4) {
        return Err(IsoQuantError::HeadDimNotMultipleOf4 { head_dim });
    }
    if group_size == 0 || !group_size.is_multiple_of(4) {
        return Err(IsoQuantError::InvalidGroupSize { group_size });
    }
    if !head_dim.is_multiple_of(group_size) {
        return Err(IsoQuantError::GroupSizeNotDivisorOfHeadDim {
            group_size,
            head_dim,
        });
    }
    if bits != 3 && bits != 4 {
        return Err(IsoQuantError::UnsupportedBits { bits });
    }

    let codebook =
        lloyd_gaussian_codebook(bits).map_err(|e| IsoQuantError::Codebook(e.to_string()))?;

    let n_tokens = norms.len();
    let n_groups = head_dim / group_size;
    let words_per_group = group_size.div_ceil(vals_per_word(bits));

    let mut out = vec![0.0_f32; n_tokens * head_dim];

    for (tok, &norm) in norms.iter().enumerate() {
        for grp in 0..n_groups {
            let flat_grp = tok * n_groups + grp;

            // flat_grp < n_tokens * n_groups = scales.len() (caller contract).
            #[allow(
                clippy::indexing_slicing,
                reason = "flat_grp < n_tokens*n_groups; scales.len() == n_tokens*n_groups by encode contract"
            )]
            let scale = scales[flat_grp];

            // Per-group quaternion (stored as 4 consecutive f32s).
            let quat_base = flat_grp * 4;
            // quat_base + 3 < quaternions.len() == n_tokens*n_groups*4 by encode contract.
            #[allow(
                clippy::indexing_slicing,
                reason = "quat_base+3 < quaternions.len() == n_tokens*n_groups*4 by encode contract"
            )]
            let q_l = [
                quaternions[quat_base],
                quaternions[quat_base + 1],
                quaternions[quat_base + 2],
                quaternions[quat_base + 3],
            ];
            let q_l_conj = quat_conjugate(q_l);

            // Unpack `bits`-bit codes for this group.
            let code_base = flat_grp * words_per_group;
            // code_base + words_per_group <= codes_packed.len() by encode contract.
            #[allow(
                clippy::indexing_slicing,
                reason = "code_base+words_per_group <= codes_packed.len() by encode contract"
            )]
            let grp_words = &codes_packed[code_base..code_base + words_per_group];
            let indices = unpack_bits(grp_words, group_size, bits);

            // Dequantize: index → centroid → rescale.
            let mut dequant_vals = Vec::with_capacity(group_size);
            for &idx in &indices {
                // idx is `bits`-wide (0..2^bits); codebook has 2^bits entries.
                #[allow(
                    clippy::indexing_slicing,
                    reason = "idx < 2^bits; lloyd_gaussian_codebook returns 2^bits entries"
                )]
                let centroid = codebook[idx as usize];
                dequant_vals.push(centroid * scale);
            }

            // Inverse rotate in quaternion blocks of 4.
            let out_row_start = tok * head_dim;
            let grp_start = grp * group_size;
            for qblk in 0..group_size / 4 {
                let base = qblk * 4;
                // base + 3 < group_size <= dequant_vals.len() (capacity = group_size, all pushed).
                #[allow(
                    clippy::indexing_slicing,
                    reason = "base+3 < group_size = dequant_vals.len()"
                )]
                let rotated = [
                    dequant_vals[base],
                    dequant_vals[base + 1],
                    dequant_vals[base + 2],
                    dequant_vals[base + 3],
                ];
                // T^{-1}(r) = q̄_L * r
                let restored = quat_multiply(q_l_conj, rotated);
                let out_base = out_row_start + grp_start + base;
                // out_base + 3 < n_tokens * head_dim = out.len() (tok < n_tokens, grp_start + base + 3 < head_dim).
                #[allow(
                    clippy::indexing_slicing,
                    reason = "out_base+3 < n_tokens*head_dim = out.len(): tok < n_tokens; grp_start+base+3 < head_dim"
                )]
                {
                    out[out_base] = restored[0] * norm;
                    out[out_base + 1] = restored[1] * norm;
                    out[out_base + 2] = restored[2] * norm;
                    out[out_base + 3] = restored[3] * norm;
                }
            }
        }
    }

    Ok(out)
}

// ── Module tests reference ────────────────────────────────────────────────────

#[cfg(test)]
#[path = "isoquant_tests.rs"]
mod isoquant_tests;
