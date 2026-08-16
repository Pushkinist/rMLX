//! Rotation-quality gates for the KV codecs that apply an orthogonal transform.
//!
//! # What the existing cosine gates do not measure
//!
//! Every per-codec cosine gate in this crate runs on [`lcg_data`] — i.i.d.
//! uniform on `[-1, 1]`. That distribution is already close to maximally
//! incoherent (mean `mu` ≈ 1.72 at `head_dim = 128`, against a minimum of 1),
//! so no rotation can improve it; a Hadamard actually pushes it *up*, toward
//! the i.i.d. Gaussian value of ≈ 2.87, by the central limit theorem. The
//! cosine gates therefore measure round-trip quantizer fidelity and nothing
//! about rotation quality — a rotation replaced by the identity passes them
//! all.
//!
//! Rotation-based KV codecs exist to neutralise **outlier channels**, so the
//! gates here run on [`outlier_fixture`] and measure the incoherence parameter
//!
//! ```text
//! mu(x) = sqrt(d) · max_i |x_i| / ||x||_2
//! ```
//!
//! directly, before and after each codec's transform. See
//! [`crate::test_utils::outlier_channel_data`] for what real K-cache data looks
//! like and the citations behind the fixture's shape.
//!
//! # The block-size ceiling
//!
//! Only one codec family applies a full-dimension orthogonal transform. The
//! rest rotate inside small blocks, and a block-`b` orthogonal transform can
//! reduce `mu` by at most `sqrt(b)`:
//!
//! > Let `P = max_j |x_j|`, attained in block `B`. A block-diagonal orthogonal
//! > map preserves `||y_B||_2 = ||x_B||_2 >= P`, and any `b`-vector has
//! > `max_j |y_j| >= ||y_B||_2 / sqrt(b)`, so the new peak is at least
//! > `P / sqrt(b)`. `||y||_2 = ||x||_2`, therefore
//! > `mu_after >= mu_before / sqrt(b)`.
//!
//! | Codec family | Transform | Block | `mu` ceiling |
//! |---|---|---|---|
//! | `rot_k` / `RotK` | Walsh-Hadamard, full `head_dim` | `head_dim` | `sqrt(head_dim)` |
//! | `iso3` / `iso4` | isoclinic SO(4), fixed quaternion | 4 | 2.00 |
//! | `rotor3` / `rotor4` | Cl(3,0) rotor sandwich, static per (layer, head) | 3 | 1.73 |
//! | `planar3` / `planar4` | Givens, 16-entry codebook, per-pair search | 2 | 1.41 |
//!
//! This is not a defect in the block-local codecs — they buy packing
//! efficiency, a different axis. The gates below pin what each family actually
//! delivers so the naming stops implying a capability only one family has.
//!
//! # Deterministic Hadamard
//!
//! `rot_k` uses a plain Sylvester Hadamard with no random sign flips. The
//! incoherence guarantee from the QuIP line of work is stated for a
//! *randomized* Hadamard, so a deterministic `H` has adversarial inputs in
//! principle. [`hadamard_incoherence_ratio_beats_every_block_local_rotation`]
//! measures what it delivers on this fixture rather than assuming the
//! guarantee transfers.

use crate::clifford::{make_rotor_table, rotor_sandwich, Rotor, MV_DIM};
use crate::isoquant::{
    iso_decode_fast, iso_encode_fast, quat_conjugate, quat_multiply, FIXED_QUAT,
};
use crate::planarquant::{planar_dequantize, planar_quantize, planar_rotation_codebook};
use crate::rotorquant::{
    n_groups_for, rotor3_decode, rotor3_encode, rotor4_decode, rotor4_encode, ROTOR3_GROUP_SIZE,
};
use crate::test_utils::{
    cosine_similarity_per_row, fwht_normalize, incoherence_per_row, lcg_data, outlier_fixture,
    sqnr_db, CosineStats, DB_PER_BIT, OUTLIER_HEAD_DIM, OUTLIER_ROWS, TEST_SEED,
};
use crate::turboquant::GROUP_SIZE;

// ── Row-wise rotations, expressed with each codec's own primitives ───────────

/// `rot_k`: normalized Walsh-Hadamard over the whole `head_dim`.
///
/// This is the same [`fwht_normalize`] call the existing `rot_k` cosine gate
/// uses as the CPU reference for `K_rot = K @ R`.
fn rotate_hadamard(buf: &mut [f32]) {
    fwht_normalize(buf, OUTLIER_HEAD_DIM);
}

/// The same Walsh-Hadamard truncated to blocks of 4 — the plausible way `rot_k`
/// stops being a full-dimension transform, as opposed to being deleted.
///
/// Still orthogonal, still self-inverse, still genuinely decorrelating; it just
/// works over 4 coordinates instead of 128. The block-`b` ceiling therefore caps
/// it at `sqrt(4) = 2.00x` of `mu` reduction and at `log2(sqrt(4)) = 1.0` bits
/// of affine gain, both below the gates — so its rejection is a theorem, not an
/// empirical hope.
fn rotate_hadamard_block4(buf: &mut [f32]) {
    fwht_normalize(buf, 4);
}

/// `iso3` / `iso4`: left isoclinic quaternion product on each block of 4.
///
/// `iso_encode_fast` computes `quat_multiply(FIXED_QUAT, row_block / norm)`.
/// The per-row L2 normalisation is a positive scalar and `mu` is scale
/// invariant, so dropping it here changes nothing the statistic can see. The
/// quaternion and the product are the codec's own.
fn rotate_iso(buf: &mut [f32]) {
    for block in buf.chunks_exact_mut(4) {
        let rotated = quat_multiply(FIXED_QUAT, [block[0], block[1], block[2], block[3]]);
        block.copy_from_slice(&rotated);
    }
}

/// `rotor3` / `rotor4`: Cl(3,0) rotor sandwich on each block of 3.
///
/// Mirrors `rotor3_encode`: embed the 3-vector as the grade-1 part of a
/// multivector, apply `R · mv · R̃`, read the grade-1 part back. The sandwich
/// preserves grade, so grades 0/2/3 stay zero and the codec's per-group
/// `max|r_i|` over all 8 components is in fact a max over these 3 — the reason
/// the effective sample count for the scale is 3, not 8.
///
/// The rotor table is `make_rotor_table(layer, head, n_groups)`, the same
/// deterministic per-(layer, head, group) table production uses.
///
/// The draw is a parameter because it matters. Unlike iso (one fixed constant
/// quaternion) and planar (searched per pair, so it follows the data), a rotor
/// table is drawn per (layer, head) and only the handful of groups holding
/// outlier channels affect `mu` — at `head_dim = 128` with 4 outliers that is
/// four rotors out of 43. A single draw is a four-sample estimate, so
/// [`rotor_block_rotation_incoherence_gate`] sweeps draws rather than pinning
/// one.
fn rotate_rotor_with(buf: &mut [f32], layer: u32, head: u32) {
    let n_groups = n_groups_for(OUTLIER_HEAD_DIM);
    let rotors = make_rotor_table(layer, head, n_groups);

    for row in buf.chunks_exact_mut(OUTLIER_HEAD_DIM) {
        for group in 0..n_groups {
            let base = group * 4;
            let rotor: Rotor = [
                rotors[base],
                rotors[base + 1],
                rotors[base + 2],
                rotors[base + 3],
            ];

            let start = group * ROTOR3_GROUP_SIZE;
            let mut mv = [0.0f32; MV_DIM];
            for e in 0..ROTOR3_GROUP_SIZE {
                if start + e < OUTLIER_HEAD_DIM {
                    mv[e + 1] = row[start + e];
                }
            }
            let rotated = rotor_sandwich(rotor, &mv);
            for e in 0..ROTOR3_GROUP_SIZE {
                if start + e < OUTLIER_HEAD_DIM {
                    row[start + e] = rotated[e + 1];
                }
            }
        }
    }
}

/// (layer, head) draws swept by the rotor gates, so a pin describes the family
/// rather than one table.
const ROTOR_DRAWS: [(u32, u32); 8] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (3, 2),
    (7, 5),
    (11, 0),
    (17, 9),
    (31, 13),
];

/// The `(0, 0)` draw — used where a single representative table is all the
/// comparison needs (the cross-family table, the substitute guards).
fn rotate_rotor(buf: &mut [f32]) {
    rotate_rotor_with(buf, 0, 0);
}

/// `planar3` / `planar4`: the Givens rotation `planar_quantize` actually chose
/// for each pair, replayed out of the encoded block's `rotations` field.
///
/// Reading the encoder's decision back rather than re-running the search keeps
/// this measurement tied to the shipped selection rule (minimise max-abs
/// reconstruction error under the `bits`-wide codebook), which is why the
/// rotation — and therefore `mu` — depends on `bits` for this family and not
/// for the other two.
fn rotate_planar(buf: &mut [f32], bits: u8) {
    let shape = [1, 1, OUTLIER_ROWS as i32, OUTLIER_HEAD_DIM as i32];
    let blocks = planar_quantize(buf, GROUP_SIZE, bits, &shape).expect("planar_quantize");
    let codebook = planar_rotation_codebook();

    for (pair, values) in buf.chunks_exact_mut(2).enumerate() {
        // 4-bit index per pair, 2 per byte, low nibble first.
        let index = usize::from((blocks.rotations[pair / 2] >> ((pair % 2) * 4)) & 0xF);
        let entry = codebook[index];
        let ya = entry[0].mul_add(values[0], entry[1] * values[1]);
        let yb = entry[2].mul_add(values[0], entry[3] * values[1]);
        values[0] = ya;
        values[1] = yb;
    }
}

// ── The gate, as one function both the gates and the guards call ─────────────

/// Mean-`mu` reduction factor `mu_before / mu_after` delivered by `rotate` on
/// the canonical outlier fixture.
///
/// `> 1` means the transform reduced incoherence; `1.0` means it did nothing.
/// Every gate and every mutation guard below goes through this one function, so
/// a guard cannot drift away from the gate it is guarding.
fn mean_mu_reduction(rotate: impl FnOnce(&mut [f32])) -> f64 {
    let data = outlier_fixture();
    let before = incoherence_per_row(&data, OUTLIER_HEAD_DIM);
    let mut rotated = data;
    rotate(&mut rotated);
    let after = incoherence_per_row(&rotated, OUTLIER_HEAD_DIM);
    before.mean / after.mean
}

/// Minimum `mu` reduction demanded of a full-dimension orthogonal transform.
///
/// Justification, and it is a separation argument rather than a prediction of
/// the measured value. The block-`b` ceiling in the module docs says a
/// transform that reduces `mu` by a factor `R` must have `b >= R²`. Demanding
/// `R >= 3.0` therefore demands an effective block of **at least 9**, which no
/// degenerate can supply: block-4 iso caps at 2.00x, block-3 rotor at 1.73x,
/// block-2 planar at 1.41x, a block-4 truncated Hadamard at 2.00x, and the
/// identity sits at 1.00x. The only transform in the crate that can clear it is
/// one that mixes the whole `head_dim`. 3.0 is thus the smallest round number
/// that separates "full-dimension" from every alternative, chosen from the
/// ceilings rather than fitted to the codec.
///
/// Do not try to predict the measured 3.89x from the fixture parameters. A
/// natural back-of-envelope — outlier mass spread evenly, `mu` lands near the
/// i.i.d. Gaussian value of 2.87 — gives `8.37 / 2.87 = 2.9x` and would
/// wrongly predict this gate goes red. It is wrong because the Hadamard image
/// of 4 outlier channels takes only `2^4 = 16` distinct values across the 128
/// coordinates, so the post-rotation peak is a max over ~16 effective draws
/// rather than 128, and `mu_after` (2.15) lands *below* the i.i.d. value.
///
/// [`identity_rotation_excluded_by_the_hadamard_incoherence_threshold`] and
/// [`non_full_dimension_rotations_fail_the_hadamard_incoherence_gate`] assert
/// the separation.
const HADAMARD_MIN_MU_REDUCTION: f64 = 3.0;

/// The full-dimension Hadamard clears the reduction gate, and no block-local
/// rotation comes close.
///
/// Reported alongside is each block-local family's `sqrt(block)` ceiling, so
/// the table in the module docs is checked rather than asserted in prose.
#[test]
fn hadamard_incoherence_ratio_beats_every_block_local_rotation() {
    let hadamard = mean_mu_reduction(rotate_hadamard);
    let iso = mean_mu_reduction(rotate_iso);
    let rotor = mean_mu_reduction(rotate_rotor);
    let planar3 = mean_mu_reduction(|buf| rotate_planar(buf, 3));
    let planar4 = mean_mu_reduction(|buf| rotate_planar(buf, 4));

    println!(
        "mu reduction on the outlier fixture (head_dim={OUTLIER_HEAD_DIM}, rows={OUTLIER_ROWS}):\n\
         \x20 hadamard(full) {hadamard:.4}x  ceiling {:.2}x\n\
         \x20 iso(block 4)   {iso:.4}x  ceiling 2.00x\n\
         \x20 rotor(block 3) {rotor:.4}x  ceiling 1.73x\n\
         \x20 planar3(pair)  {planar3:.4}x  ceiling 1.41x\n\
         \x20 planar4(pair)  {planar4:.4}x  ceiling 1.41x",
        (OUTLIER_HEAD_DIM as f64).sqrt(),
    );

    assert!(
        hadamard >= HADAMARD_MIN_MU_REDUCTION,
        "rot_k Hadamard reduced mean mu by only {hadamard:.4}x, below the \
         {HADAMARD_MIN_MU_REDUCTION:.1}x a full-dimension orthogonal transform must deliver \
         on this fixture"
    );

    for (name, reduction, ceiling) in [
        ("iso", iso, 4.0f64),
        ("rotor", rotor, 3.0),
        ("planar3", planar3, 2.0),
        ("planar4", planar4, 2.0),
    ] {
        assert!(
            reduction <= ceiling.sqrt() + 1e-3,
            "{name} reported a {reduction:.4}x mu reduction, above the sqrt({ceiling}) = \
             {:.4}x a block-{ceiling} orthogonal transform can deliver — the measurement or \
             the declared block size is wrong",
            ceiling.sqrt(),
        );
        assert!(
            hadamard > reduction,
            "{name} ({reduction:.4}x) matched or beat the full-dimension Hadamard \
             ({hadamard:.4}x) — the block-size table in the module docs is wrong"
        );
    }
}

/// The threshold excludes the identity.
///
/// Named for what it is: `mean_mu_reduction(|_| {})` divides `before.mean` by
/// itself, so the `1.0` is exact by construction and this pins the *constant
/// comparison* `1.0 < 3.0`, not the behaviour of the statistic. The guard that
/// mutates a real transform is
/// [`non_full_dimension_rotations_fail_the_hadamard_incoherence_gate`].
#[test]
fn identity_rotation_excluded_by_the_hadamard_incoherence_threshold() {
    let reduction = mean_mu_reduction(|_| {});
    assert!(
        (reduction - 1.0).abs() < 1e-9,
        "the identity must leave mu untouched, got {reduction:.6}x"
    );
    assert!(
        reduction < HADAMARD_MIN_MU_REDUCTION,
        "the identity passed a gate that demands {HADAMARD_MIN_MU_REDUCTION:.1}x — \
         the gate cannot fail and is worthless"
    );
}

/// Mutation guard: every rotation that is orthogonal but not full-dimension
/// must fail the gate.
///
/// These are the plausible degradations, not the crude one — genuine orthogonal
/// transforms that genuinely decorrelate, over too few dimensions. The first is
/// the likeliest real defect: `rot_k`'s own FWHT called with the wrong width.
#[test]
fn non_full_dimension_rotations_fail_the_hadamard_incoherence_gate() {
    for (name, reduction) in [
        (
            "hadamard truncated to block-4",
            mean_mu_reduction(rotate_hadamard_block4),
        ),
        ("iso block-4", mean_mu_reduction(rotate_iso)),
        ("rotor block-3", mean_mu_reduction(rotate_rotor)),
        (
            "planar block-2",
            mean_mu_reduction(|buf| rotate_planar(buf, 4)),
        ),
    ] {
        println!("{name}: {reduction:.4}x against a {HADAMARD_MIN_MU_REDUCTION:.1}x gate");
        assert!(
            reduction < HADAMARD_MIN_MU_REDUCTION,
            "{name} reached {reduction:.4}x and passed a gate meant to require a \
             full-dimension transform"
        );
    }
}

// ── Per-family pinned reductions ─────────────────────────────────────────────

/// Slack under a pinned block-local `mu` reduction, in reduction factor.
///
/// A family may lose 0.05 of its measured reduction *factor* before the floor
/// bites — 3.6% of iso's measurement, 4.6% of rotor's. The fixture is seeded and
/// the transforms are deterministic, so nothing varies run to run; the slack
/// covers f32 and codegen drift. It is small enough that the identity — the
/// limit case at exactly `1.00x` — stays outside every floor, by 0.031 at the
/// tightest (rotor, whose floor is pinned to the weakest of several draws).
const BLOCK_REDUCTION_SLACK: f64 = 0.05;

/// Assert a block-local family's `mu` reduction sits between its `sqrt(block)`
/// ceiling and its pinned measurement.
///
/// Two-sided on purpose. The ceiling is a theorem (see the module docs) and
/// catches a measurement that claims more decorrelation than the block size
/// allows. The floor, `measured - BLOCK_REDUCTION_SLACK`, catches the
/// regression that matters — a rotation quietly becoming weaker.
fn assert_block_local_reduction(name: &str, block: usize, reduction: f64, measured: f64) {
    let ceiling = (block as f64).sqrt();
    let floor = measured - BLOCK_REDUCTION_SLACK;
    println!("{name}: mu reduction {reduction:.4}x (block {block}, ceiling {ceiling:.4}x, floor {floor:.4}x)");
    assert!(
        reduction <= ceiling + 1e-3,
        "{name} reduction {reduction:.4}x exceeds the block-{block} ceiling {ceiling:.4}x"
    );
    assert!(
        reduction >= floor,
        "{name} reduction {reduction:.4}x fell below the pinned floor {floor:.4}x"
    );
}

/// `iso3` / `iso4` — block 4, ceiling `sqrt(4) = 2.00x`.
///
/// The fixed golden-ratio quaternion spreads a lone large coordinate over the
/// block with weights `(1, phi, phi-1, 1)/sqrt(5)`, whose largest entry is
/// `phi/sqrt(5) = 0.7236`. A single dominant coordinate therefore keeps 72% of
/// its peak, i.e. a 1.38x reduction — well short of the 2.00x an SO(4) rotation
/// could in principle deliver, because the quaternion is fixed rather than
/// fitted to the data.
///
/// The rotation does not depend on `bits`, so iso3 and iso4 share this gate.
#[test]
fn iso_block_rotation_incoherence_gate() {
    assert_block_local_reduction("iso3/iso4", 4, mean_mu_reduction(rotate_iso), 1.3846);
}

/// `rotor3` / `rotor4` — block 3, ceiling `sqrt(3) = 1.73x`.
///
/// The rotor is a random SO(3) element drawn from a per-(layer, head, group)
/// seed, not fitted to the data, so on average it spreads a dominant coordinate
/// over the block only partially. The rotation does not depend on `bits`, so
/// rotor3 and rotor4 share this gate.
///
/// **Swept over draws, not pinned to one.** Only the groups holding outlier
/// channels move `mu`, which at `head_dim = 128` with 4 outliers is four rotors
/// of 43; a single `(layer, head)` table is therefore a four-sample estimate.
/// Across [`ROTOR_DRAWS`] the reduction spans 1.0815x–1.2089x, so a pin taken
/// from the `(0, 0)` draw alone (1.1815x) would sit above what half the other
/// draws deliver and would go red on a different layer. The floor is pinned to
/// the **weakest** draw instead, so it describes the family.
#[test]
fn rotor_block_rotation_incoherence_gate() {
    let reductions: Vec<(u32, u32, f64)> = ROTOR_DRAWS
        .iter()
        .map(|&(layer, head)| {
            (
                layer,
                head,
                mean_mu_reduction(|b| rotate_rotor_with(b, layer, head)),
            )
        })
        .collect();

    let weakest = reductions
        .iter()
        .map(|&(_, _, r)| r)
        .fold(f64::INFINITY, f64::min);
    let strongest = reductions
        .iter()
        .map(|&(_, _, r)| r)
        .fold(f64::NEG_INFINITY, f64::max);

    println!(
        "rotor mu reduction across {} (layer, head) draws: weakest {weakest:.4}x, \
         strongest {strongest:.4}x — {}",
        reductions.len(),
        reductions
            .iter()
            .map(|(l, h, r)| format!("({l},{h}) {r:.4}"))
            .collect::<Vec<_>>()
            .join(" "),
    );

    // Every draw must clear the ceiling; the floor is pinned to the weakest.
    for &(layer, head, reduction) in &reductions {
        assert!(
            reduction <= (3.0f64).sqrt() + 1e-3,
            "rotor draw ({layer},{head}) reduction {reduction:.4}x exceeds the block-3 ceiling"
        );
    }
    assert_block_local_reduction("rotor3/rotor4 (weakest draw)", 3, weakest, 1.0815);
}

/// `planar3` — block 2, ceiling `sqrt(2) = 1.41x`.
///
/// Unlike iso and rotor, the Givens angle is *searched* per pair, so the
/// rotation depends on the codebook and therefore on `bits`. The search
/// minimises reconstruction error, not incoherence, which is why the delivered
/// reduction is far below the ceiling.
#[test]
fn planar3_pair_rotation_incoherence_gate() {
    assert_block_local_reduction(
        "planar3",
        2,
        mean_mu_reduction(|b| rotate_planar(b, 3)),
        1.1910,
    );
}

/// `planar4` — block 2, ceiling `sqrt(2) = 1.41x`. See
/// [`planar3_pair_rotation_incoherence_gate`].
#[test]
fn planar4_pair_rotation_incoherence_gate() {
    assert_block_local_reduction(
        "planar4",
        2,
        mean_mu_reduction(|b| rotate_planar(b, 4)),
        1.1518,
    );
}

// ── Does the rotation earn its keep? ─────────────────────────────────────────

/// The i.i.d. uniform fixture the existing per-codec cosine gates use, at the
/// outlier fixture's shape so the two are directly comparable.
fn lcg_fixture() -> Vec<f32> {
    lcg_data(OUTLIER_ROWS * OUTLIER_HEAD_DIM, TEST_SEED)
}

/// Symmetric 8-bit affine at `group_size = 64` — `rot_k`'s quantizer, with no
/// rotation.
fn affine_q8_g64(data: &[f32]) -> Vec<f32> {
    const GROUP: usize = 64;
    let mut decoded = vec![0.0f32; data.len()];
    for (group, out) in data
        .chunks_exact(GROUP)
        .zip(decoded.chunks_exact_mut(GROUP))
    {
        let abs_max = group.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (&v, slot) in group.iter().zip(out.iter_mut()) {
            let code = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            *slot = scale * f32::from(code);
        }
    }
    decoded
}

/// Run `rotate` over each `head_dim` row, quantize with [`affine_q8_g64`], then
/// undo `rotate` — the `rot_k` pipeline, parameterised by the transform.
///
/// Every rotation here is orthogonal and involutive-by-inverse, so `unrotate`
/// is supplied explicitly rather than assumed.
fn rotated_affine_roundtrip(
    data: &[f32],
    rotate: impl Fn(&mut [f32]),
    unrotate: impl Fn(&mut [f32]),
) -> Vec<f32> {
    let mut buf = data.to_vec();
    rotate(&mut buf);
    let mut decoded = affine_q8_g64(&buf);
    unrotate(&mut decoded);
    decoded
}

/// `rot_k`: FWHT, [`affine_q8_g64`], FWHT back (the normalized Hadamard is
/// self-inverse). Same protocol as `rot_k_hadamard_8bit_cosine_gate`, at the
/// outlier fixture's `head_dim = 128`.
fn rot_k_roundtrip(data: &[f32]) -> Vec<f32> {
    rotated_affine_roundtrip(data, rotate_hadamard, rotate_hadamard)
}

/// Inverse of [`rotate_iso`]: left-multiply by the conjugate quaternion, which
/// is what `iso_decode_fast` does.
fn unrotate_iso(buf: &mut [f32]) {
    let conjugate = quat_conjugate(FIXED_QUAT);
    for block in buf.chunks_exact_mut(4) {
        let restored = quat_multiply(conjugate, [block[0], block[1], block[2], block[3]]);
        block.copy_from_slice(&restored);
    }
}

/// Bits of SQNR `rotate` buys over the same quantizer with no transform at all.
///
/// The gate and both mutation guards call this one function, so a guard cannot
/// drift away from the gate it guards.
fn rotation_gain_bits(
    data: &[f32],
    rotate: impl Fn(&mut [f32]),
    unrotate: impl Fn(&mut [f32]),
) -> f64 {
    let rotated = sqnr_db(data, &rotated_affine_roundtrip(data, rotate, unrotate));
    let plain = sqnr_db(data, &affine_q8_g64(data));
    (rotated - plain) / DB_PER_BIT
}

/// Minimum SQNR the Hadamard must buy over the identical unrotated quantizer,
/// on outlier data, in bits.
///
/// The model, stated so it can be checked. `rot_k`'s K-side quantizer is a
/// symmetric 8-bit affine over a group of 64 channels, so its step is set by
/// that group's largest magnitude and the SQNR gain from any transform is
/// exactly `log2(peak_plain / peak_rotated)` for the group. That is the same
/// quantity the block-`b` ceiling bounds: `peak` can fall by at most `sqrt(b)`,
/// so a block-`b` transform can buy at most `log2(sqrt(b)) = 0.5·log2(b)` bits
/// — 1.00 for block-4, 0.79 for block-3, 0.50 for block-2, 0 for the identity.
///
/// **Demanding 1.5 bits therefore demands an effective block of at least 8**,
/// which is the separation the gate is for. It is not a fraction of the outlier
/// ratio: a tempting model says one channel at `r` inflates the step by `r` and
/// a rotation recovers `log2(20) ≈ 4.3` bits, but that is wrong — the transform
/// spreads the outlier's energy into every coordinate, so the RMS that sets the
/// *new* peak rises too. The measured fall is 4.06x in the group peak, i.e.
/// 2.02 bits, not 4.3.
///
/// 1.5 sits between what a full-dimension transform delivers here and the
/// strongest substitute: measured 1.81 bits against 0.91 for a block-4
/// truncation of the same Hadamard. Both bounds are asserted —
/// [`identity_rotation_excluded_by_the_rot_k_gain_threshold`] and
/// [`non_full_dimension_rotations_fail_the_rot_k_gain_gate`].
const ROT_K_MIN_OUTLIER_GAIN_BITS: f64 = 1.5;

/// The decisive `rot_k` measurement: the Hadamard is worth bits on outlier
/// data, and is a **net loss** on the i.i.d. fixture the older gate uses.
///
/// This is the property the codec exists for and the one no LCG-fixture gate
/// can see. See [`identity_rotation_fails_the_rot_k_gain_gate`] and
/// [`block_local_rotation_fails_the_rot_k_gain_gate`] for the guards.
#[test]
fn rot_k_hadamard_buys_bits_on_outlier_data_and_costs_them_on_iid_data() {
    let outlier = outlier_fixture();
    let uniform = lcg_fixture();

    let outlier_gain = rotation_gain_bits(&outlier, rotate_hadamard, rotate_hadamard);
    let uniform_gain = rotation_gain_bits(&uniform, rotate_hadamard, rotate_hadamard);

    println!(
        "rot_k Hadamard vs the same affine-q8-g64 quantizer without it:\n\
         \x20 outlier fixture: {:.3} dB rotated vs {:.3} dB plain = {outlier_gain:+.3} bits\n\
         \x20 i.i.d. uniform : {:.3} dB rotated vs {:.3} dB plain = {uniform_gain:+.3} bits",
        sqnr_db(&outlier, &rot_k_roundtrip(&outlier)),
        sqnr_db(&outlier, &affine_q8_g64(&outlier)),
        sqnr_db(&uniform, &rot_k_roundtrip(&uniform)),
        sqnr_db(&uniform, &affine_q8_g64(&uniform)),
    );

    assert!(
        outlier_gain >= ROT_K_MIN_OUTLIER_GAIN_BITS,
        "the Hadamard bought only {outlier_gain:+.3} bits on outlier data, below the \
         {ROT_K_MIN_OUTLIER_GAIN_BITS:.1} bits a full-dimension decorrelating transform must \
         deliver against an outlier-inflated group scale"
    );
    assert!(
        uniform_gain < 0.0,
        "the Hadamard gained {uniform_gain:+.3} bits on i.i.d. uniform data. It is supposed \
         to *lose* there — the transform pushes uniform toward Gaussian, raising the \
         peak-to-RMS ratio the quantizer step is set by. If this passes, the i.i.d. fixture \
         has changed and the premise that it cannot measure rotation quality needs re-checking"
    );
}

/// The threshold excludes the identity.
///
/// Named for what it is: `rotation_gain_bits` with no transform subtracts a
/// value from itself, so the `0.0` is exact by construction and this pins the
/// constant comparison `0.0 < 1.5`. The guards that mutate a real transform are
/// in [`non_full_dimension_rotations_fail_the_rot_k_gain_gate`].
#[test]
fn identity_rotation_excluded_by_the_rot_k_gain_threshold() {
    let gain = rotation_gain_bits(&outlier_fixture(), |_| {}, |_| {});
    assert!(
        gain.abs() < 1e-9,
        "the identity must buy exactly nothing, got {gain:+.6} bits"
    );
    assert!(
        gain < ROT_K_MIN_OUTLIER_GAIN_BITS,
        "a codec that does not rotate reported {gain:+.3} bits of rotation gain — \
         the gate cannot fail"
    );
}

/// Mutation guard: orthogonal transforms that are not full-dimension must fail
/// the gain gate.
///
/// The plausible degradations. `0.5·log2(b)` caps a block-`b` transform at 1.00
/// bit (block-4) and 0.79 (block-3), both under the 1.5-bit gate, so these
/// rejections are theorems rather than empirical hopes. The truncated FWHT is
/// the likeliest real defect — `rot_k`'s own transform called with the wrong
/// width.
#[test]
fn non_full_dimension_rotations_fail_the_rot_k_gain_gate() {
    let data = outlier_fixture();
    for (name, gain) in [
        (
            "hadamard truncated to block-4",
            rotation_gain_bits(&data, rotate_hadamard_block4, rotate_hadamard_block4),
        ),
        (
            "iso block-4 isoclinic",
            rotation_gain_bits(&data, rotate_iso, unrotate_iso),
        ),
    ] {
        println!("{name} buys {gain:+.3} bits against a {ROT_K_MIN_OUTLIER_GAIN_BITS:.1}-bit gate");
        assert!(
            gain < ROT_K_MIN_OUTLIER_GAIN_BITS,
            "{name} reported {gain:+.3} bits and passed a gate meant to require a \
             full-dimension transform"
        );
    }
}

// ── Outlier-fixture cosine gates ─────────────────────────────────────────────

/// Tolerance on a pinned cosine measurement, as a multiple of the reconstruction
/// error `1 - cos`.
///
/// The crate's older cosine gates pin at `measured - 0.001`. That convention
/// cannot work here: `rot_k` scores 0.999991 on this fixture and 0.999881 with
/// the Hadamard deleted outright, so an absolute 0.001 slack is **fifty times
/// wider than the entire effect** and the removal sails through. Sizing the
/// tolerance to the error instead — a codec may double `1 - cos` before the
/// floor bites — makes one rule work across five orders of magnitude of codec
/// quality.
///
/// The fixture is seeded and every codec here is deterministic, so nothing
/// varies run to run; the tolerance covers f32 and codegen drift, which moves
/// these cosines by ~1e-7 against a `rot_k` budget of 9e-6.
const COSINE_ERROR_TOLERANCE: f32 = 2.0;

/// Floor derived from a pinned outlier-fixture cosine measurement.
///
/// Shared by the gates and by
/// [`lossier_codecs_fail_the_outlier_cosine_floors`], so a guard cannot drift
/// away from the floor it is guarding.
fn outlier_cosine_floor(measured: f32) -> f32 {
    COSINE_ERROR_TOLERANCE.mul_add(measured - 1.0, 1.0)
}

/// Pinned outlier-fixture cosine measurements, `(mean, min)`.
///
/// One definition each, referenced by the gate and by the mutation guard.
const ROT_K_PIN: (f32, f32) = (0.999_989, 0.999_980);
const ISO3_PIN: (f32, f32) = (0.996_769, 0.992_033);
const ISO4_PIN: (f32, f32) = (0.998_764, 0.997_551);
const ROTOR3_PIN: (f32, f32) = (0.995_003, 0.989_004);
const ROTOR4_PIN: (f32, f32) = (0.998_991, 0.997_015);
const PLANAR3_PIN: (f32, f32) = (0.999_960, 0.999_871);
const PLANAR4_PIN: (f32, f32) = (0.999_932, 0.999_615);

/// Assert a codec's outlier-fixture cosine against its pinned measurement, and
/// report its i.i.d. number beside it.
///
/// `measured_mean` / `measured_min` are the observed values; the floor is
/// derived as `1 - COSINE_ERROR_TOLERANCE * (1 - measured)`.
///
/// These are regression floors. They are **not** on their own evidence that a
/// rotation works: cosine is dominated by a row's largest components, and on
/// this fixture those are the outlier channels, which every codec here
/// reconstructs well because its scale is a per-group *maximum* and therefore
/// already adapts to them. The reported i.i.d. column makes that visible —
/// several codecs score *higher* on the outlier fixture than on the uniform
/// one. The rotation question is answered by the incoherence gates above and by
/// [`rot_k_hadamard_buys_bits_on_outlier_data_and_costs_them_on_iid_data`].
fn assert_outlier_cosine(
    name: &str,
    roundtrip: impl Fn(&[f32]) -> Vec<f32>,
    measured_mean: f32,
    measured_min: f32,
) {
    let floor_mean = outlier_cosine_floor(measured_mean);
    let floor_min = outlier_cosine_floor(measured_min);

    let outlier = outlier_fixture();
    let outlier_stats: CosineStats =
        cosine_similarity_per_row(&outlier, &roundtrip(&outlier), OUTLIER_HEAD_DIM);

    let uniform = lcg_fixture();
    let uniform_stats = cosine_similarity_per_row(&uniform, &roundtrip(&uniform), OUTLIER_HEAD_DIM);

    println!(
        "{name} cosine: outlier mean={:.6} min={:.6} | i.i.d. mean={:.6} \
         (floors {floor_mean:.6} / {floor_min:.6})",
        outlier_stats.mean, outlier_stats.min, uniform_stats.mean,
    );

    assert!(
        outlier_stats.mean >= floor_mean,
        "{name} outlier-fixture mean cosine {:.6} fell below the floor {floor_mean:.6} \
         (pinned measurement {measured_mean:.6}, error allowed to double)",
        outlier_stats.mean,
    );
    assert!(
        outlier_stats.min >= floor_min,
        "{name} outlier-fixture min cosine {:.6} fell below the floor {floor_min:.6} \
         (pinned measurement {measured_min:.6}, error allowed to double)",
        outlier_stats.min,
    );
}

fn iso_roundtrip(data: &[f32], bits: u8) -> Vec<f32> {
    let (codes, scales, quats, norms) =
        iso_encode_fast(data, OUTLIER_HEAD_DIM, 4, bits).expect("iso_encode_fast");
    iso_decode_fast(&codes, &scales, &quats, &norms, OUTLIER_HEAD_DIM, 4, bits)
        .expect("iso_decode_fast")
}

fn rotor_roundtrip(data: &[f32], bits: u8) -> Vec<f32> {
    let rotors = make_rotor_table(0, 0, n_groups_for(OUTLIER_HEAD_DIM));
    if bits == 3 {
        let (codes, scales, norms) =
            rotor3_encode(data, &rotors, OUTLIER_HEAD_DIM).expect("rotor3_encode");
        rotor3_decode(&codes, &scales, &norms, &rotors, OUTLIER_HEAD_DIM).expect("rotor3_decode")
    } else {
        let (codes, scales, norms) =
            rotor4_encode(data, &rotors, OUTLIER_HEAD_DIM).expect("rotor4_encode");
        rotor4_decode(&codes, &scales, &norms, &rotors, OUTLIER_HEAD_DIM).expect("rotor4_decode")
    }
}

fn planar_roundtrip(data: &[f32], bits: u8) -> Vec<f32> {
    let n_rows = data.len() / OUTLIER_HEAD_DIM;
    let shape = [1, 1, n_rows as i32, OUTLIER_HEAD_DIM as i32];
    let blocks = planar_quantize(data, GROUP_SIZE, bits, &shape).expect("planar_quantize");
    planar_dequantize(&blocks).expect("planar_dequantize")
}

/// `rot_k` Hadamard + 8-bit affine, outlier fixture.
#[test]
fn rot_k_outlier_cosine_gate() {
    assert_outlier_cosine("rot_k h8", rot_k_roundtrip, ROT_K_PIN.0, ROT_K_PIN.1);
}

/// `iso3`, outlier fixture.
#[test]
fn iso3_outlier_cosine_gate() {
    assert_outlier_cosine("iso3", |d| iso_roundtrip(d, 3), ISO3_PIN.0, ISO3_PIN.1);
}

/// `iso4`, outlier fixture.
#[test]
fn iso4_outlier_cosine_gate() {
    assert_outlier_cosine("iso4", |d| iso_roundtrip(d, 4), ISO4_PIN.0, ISO4_PIN.1);
}

/// `rotor3`, outlier fixture.
#[test]
fn rotor3_outlier_cosine_gate() {
    assert_outlier_cosine(
        "rotor3",
        |d| rotor_roundtrip(d, 3),
        ROTOR3_PIN.0,
        ROTOR3_PIN.1,
    );
}

/// `rotor4`, outlier fixture.
#[test]
fn rotor4_outlier_cosine_gate() {
    assert_outlier_cosine(
        "rotor4",
        |d| rotor_roundtrip(d, 4),
        ROTOR4_PIN.0,
        ROTOR4_PIN.1,
    );
}

/// `planar3`, outlier fixture.
#[test]
fn planar3_outlier_cosine_gate() {
    assert_outlier_cosine(
        "planar3",
        |d| planar_roundtrip(d, 3),
        PLANAR3_PIN.0,
        PLANAR3_PIN.1,
    );
}

/// `planar4`, outlier fixture.
#[test]
fn planar4_outlier_cosine_gate() {
    assert_outlier_cosine(
        "planar4",
        |d| planar_roundtrip(d, 4),
        PLANAR4_PIN.0,
        PLANAR4_PIN.1,
    );
}

/// Within `iso` and `rotor`, the wider codebook must score higher on the
/// fixture that can tell them apart.
///
/// A per-codec floor cannot catch a bit-width plumbing fault that makes iso4
/// behave like iso3 — both floors would still hold. This ordering can.
///
/// `planar` is deliberately absent: it **inverts**, and does so on the i.i.d.
/// fixture too. See `byte_identical_bit_widths_leave_one_width_dominated` in
/// `rate_distortion_tests.rs`, which pins the inversion with dB numbers rather
/// than hiding it here.
#[test]
fn wider_codebooks_score_higher_on_the_outlier_fixture() {
    let data = outlier_fixture();
    for (family, narrow, wide) in [
        ("iso", iso_roundtrip(&data, 3), iso_roundtrip(&data, 4)),
        (
            "rotor",
            rotor_roundtrip(&data, 3),
            rotor_roundtrip(&data, 4),
        ),
    ] {
        let narrow_cos = cosine_similarity_per_row(&data, &narrow, OUTLIER_HEAD_DIM).mean;
        let wide_cos = cosine_similarity_per_row(&data, &wide, OUTLIER_HEAD_DIM).mean;
        assert!(
            wide_cos > narrow_cos,
            "{family}4 cosine {wide_cos:.6} did not beat {family}3 {narrow_cos:.6}"
        );
    }
}

/// Mutation guard: every one of the seven cosine floors must reject a codec
/// that genuinely loses more information than the one it was pinned from.
///
/// A floor nothing in the tree can cross proves only that the tree is
/// unchanged. Each stand-in below is a real codec run on the real fixture,
/// judged by [`outlier_cosine_floor`] — the same function the gates use — so
/// this cannot drift away from what it guards:
///
/// * `rot_k` — the identical `affine q8 g64` quantizer with the Hadamard
///   deleted. This is the case that motivated the error-relative tolerance: at
///   the crate's usual absolute `measured - 0.001` the deletion passes.
/// * `iso4`, `rotor4` — the same family one bit narrower.
/// * `planar3` — `planar4`, which is measurably worse at identical storage.
/// * `iso3`, `rotor3`, `planar4` — `turbo` at the same nominal width, i.e. one
///   scale per 32 values instead of the family's per-2/3/4 scale.
///
/// The predicate is `mean` **or** `min` below its floor, mirroring the gate,
/// which asserts both. `planar3` is caught on `min` alone: `planar4`'s mean sits
/// 1.2e-5 above the mean floor while its min sits 1.3e-4 below the min floor.
/// That is the tightest case here and the reason both statistics are pinned.
#[test]
fn lossier_codecs_fail_the_outlier_cosine_floors() {
    let data = outlier_fixture();

    let turbo = |bits: u8| {
        let rows = data.len() / OUTLIER_HEAD_DIM;
        let shape = [1, 1, rows as i32, OUTLIER_HEAD_DIM as i32];
        let blocks =
            crate::turboquant::turbo_quantize_v(&data, bits, &shape).expect("turbo_quantize_v");
        crate::turboquant::turbo_dequantize(&blocks).expect("turbo_dequantize")
    };

    let cases: [(&str, (f32, f32), &str, Vec<f32>); 7] = [
        (
            "rot_k h8",
            ROT_K_PIN,
            "the same quantizer with the Hadamard deleted",
            affine_q8_g64(&data),
        ),
        ("iso4", ISO4_PIN, "iso3", iso_roundtrip(&data, 3)),
        ("rotor4", ROTOR4_PIN, "rotor3", rotor_roundtrip(&data, 3)),
        (
            "planar3",
            PLANAR3_PIN,
            "planar4",
            planar_roundtrip(&data, 4),
        ),
        ("iso3", ISO3_PIN, "turbo3", turbo(3)),
        ("rotor3", ROTOR3_PIN, "turbo3", turbo(3)),
        ("planar4", PLANAR4_PIN, "turbo4", turbo(4)),
    ];

    for (owner, pin, substitute, decoded) in cases {
        let stats = cosine_similarity_per_row(&data, &decoded, OUTLIER_HEAD_DIM);
        let floor_mean = outlier_cosine_floor(pin.0);
        let floor_min = outlier_cosine_floor(pin.1);

        println!(
            "{owner} floor ({floor_mean:.6} / {floor_min:.6}) vs {substitute}: \
             mean {:.6} min {:.6}",
            stats.mean, stats.min,
        );
        assert!(
            stats.mean < floor_mean || stats.min < floor_min,
            "{substitute} cleared the {owner} floor (mean {:.6} >= {floor_mean:.6} and \
             min {:.6} >= {floor_min:.6}) — that floor cannot detect a genuinely lossier \
             codec and is decoration",
            stats.mean,
            stats.min,
        );
    }
}
