//! Does MLX's `mxfp8` buy fidelity over MLX affine 8-bit at the same rate?
//!
//! # Why this is the deciding cell
//!
//! An fp8 KV store is reachable today — mlx-c exposes `mlx_to_fp8` /
//! `mlx_from_fp8`, and `mlx_quantize` accepts `mode="mxfp8"`, which rMLX
//! already passes on the weight side. What it does **not** buy is rate:
//! `mxfp8` spends 8 code bits plus one E8M0 byte per group of 32, and MLX
//! affine 8-bit at group 128 spends 8 code bits plus a bf16 scale and a bf16
//! bias per group of 128. Both land on 8.250 bits per value — measured from
//! MLX's own output buffers by
//! [`mxfp8_and_affine_g128_spend_the_same_bits_per_value`], not asserted.
//!
//! At equal rate the only axis left is fidelity, and the two formats fail
//! differently. Affine's step is `range/255` over a group, so its error is
//! roughly **absolute** and a single large channel in the group coarsens every
//! other value in it. E4M3's error is roughly **relative** — a fixed 3-bit
//! mantissa at whatever exponent the value needs — so a large channel costs its
//! neighbours nothing, but every small value pays ~2^-4 of itself. Which one
//! wins is a property of the data's dynamic range, so this file measures it as
//! a *function* of dynamic range rather than at one fixture point.
//!
//! # Criterion, declared before any fidelity number was measured
//!
//! Metric: **relative Frobenius error** `‖x − x̂‖₂ / ‖x‖₂` against the bf16
//! tensor actually handed to the encoder, plus max-abs error as a secondary.
//! Both arms see the identical bf16 input, so bf16 rounding is common-mode.
//!
//! * **PASS** (the negative verdict on fp8 KV stands): `mxfp8` relative
//!   Frobenius error `>=` affine-g128 relative Frobenius error in **every**
//!   cell of the sweep.
//! * **FAIL / falsifier**: `mxfp8` error `<= 0.85 ×` affine-g128 error — a
//!   >=15% fidelity gain at identical size — across the discriminating
//!   (outlier-channel) fixture at both head dims.
//! * **INCONCLUSIVE**: anything in between, i.e. `mxfp8` ahead somewhere but
//!   never materially.
//!
//! Deterministic: fixed seeds, `Device::Cpu`, no GPU and no model serving.
//!
//! A second fp8 arm was added once MLX's encoder turned out to clip (see
//! [`the_mlx_mxfp8_encoder_clips_group_maxima_and_the_reference_encoder_does_not`]):
//! the same E4M3 codes under a non-clipping E8M0 scale, which no shipped
//! encoder reaches. A best case cannot be held to the PASS bar without
//! asserting more than the question asks, so it carries the **falsifier**
//! instead — it may tie affine, and does at one corner, but it must never be
//! materially better.
//!
//! # The fixture is the trap
//!
//! Three fixtures live in [`crate::test_utils`] and they do not measure the
//! same thing. [`lcg_data`](crate::test_utils::lcg_data) is i.i.d. uniform on
//! `[-1, 1]` — it has **no dynamic range to speak of**, which is precisely the
//! property fp8 exists to exploit, so a format comparison run on it returns "no
//! difference" for reasons that have nothing to do with the formats.
//! [`outlier_channel_data`](crate::test_utils::outlier_channel_data) is the
//! literature-shaped K-cache model: i.i.d. Gaussian with a few fixed channels
//! scaled up. That is the one with a dynamic-range axis, and
//! [`the_outlier_fixture_separates_group_sizes_and_the_uniform_fixture_does_not`]
//! demonstrates its power rather than assuming it — on two codecs already known
//! to differ (affine at group 32 vs group 128, same codec, differing only in
//! the adaptivity under test).
//!
//! # Why not real KV tensors
//!
//! A real K/V capture is the more honest fixture and is **not reachable in this
//! cell**: nothing in the tree ships a captured KV tensor, and producing one
//! means a forward pass over a real checkpoint, which is a GPU/serving cell with
//! its own window. What is done instead is to sweep the one parameter that
//! stands in for the real thing — the outlier magnitude ratio, from 1.0 (pure
//! Gaussian, no outliers) to 100.0 — so the answer is a curve over dynamic
//! range and not a point estimate at somebody's chosen constant.

use crate::test_utils::{
    gaussian_data, lcg_data, outlier_channel_data, sqnr_db, stored_bits_per_value,
    OUTLIER_CHANNELS, TEST_SEED,
};
use rmlx_mlx::{dequantize, quantize_mode, Array, Device, Dtype};

/// Bits both arms are held to. The comparison is only meaningful at equal rate.
const BITS: i32 = 8;

/// MLX affine group size that ties `mxfp8`'s rate exactly (8.250 bits/value).
const AFFINE_GROUP: i32 = 128;

/// `mxfp8`'s only group size — the OCP microscaling block, and the only one
/// MLX ships a kernel for.
const MXFP8_GROUP: i32 = 32;

/// Rate both arms must land on, in bits per stored value.
const EQUAL_RATE_BITS_PER_VALUE: f64 = 8.25;

/// The falsifier's threshold: `mxfp8` counts as *materially* better only if its
/// relative Frobenius error is at most this fraction of affine-g128's.
const MATERIAL_GAIN: f64 = 0.85;

/// Rows in every fixture here. Enough that the Frobenius ratio is stable across
/// seeds; see the per-seed spread the sweep prints.
const ROWS: usize = 256;

/// Head dims under test: 128 is Bonsai / Qwen3, 256 is gemma-4 and Qwen3.6.
/// Both are exact multiples of both group sizes, so neither arm gets a ragged
/// final group the other does not.
const HEAD_DIMS: [usize; 2] = [128, 256];

/// Outlier magnitude ratios swept. 1.0 is the no-outlier control (the fixture
/// degenerates to i.i.d. Gaussian); 20.0 is the canonical
/// [`crate::test_utils::OUTLIER_RATIO`]; 100.0 is well past anything reported
/// for emergent outlier features.
const RATIOS: [f32; 5] = [1.0, 5.0, 20.0, 50.0, 100.0];

/// Seeds per cell, so no number here rests on one draw.
const SEEDS: [u64; 3] = [TEST_SEED, TEST_SEED ^ 0x5DEE_CE66, TEST_SEED ^ 0x0BAD_C0DE];

/// One quantizer round trip: what it cost in fidelity and what it cost in rate.
struct RoundTrip {
    /// `‖x − x̂‖₂ / ‖x‖₂` against the bf16 input the encoder actually saw.
    rel_frob: f64,
    /// `max |x − x̂|`, same reference.
    max_abs: f64,
    /// Bits per input value across every buffer the encoder emitted.
    bits_per_value: f64,
}

#[allow(
    clippy::expect_used,
    reason = "test helper: an MLX failure here is a test bug, not a result"
)]
fn to_f32_vec(a: &Array, device: Device) -> Vec<f32> {
    let wide = a.astype(Dtype::F32, device).expect("astype f32");
    wide.eval().expect("eval");
    wide.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunks_exact(4) yields 4 bytes")))
        .collect()
}

/// Bytes an MLX array occupies, from its own shape and dtype.
#[allow(
    clippy::expect_used,
    reason = "test helper: an MLX failure here is a test bug, not a result"
)]
fn array_bytes(a: &Array) -> u64 {
    a.eval().expect("eval");
    let elems: u64 = a.shape().iter().map(|&d| d as u64).product();
    elems * a.dtype().itemsize() as u64
}

/// The values every arm is scored against: `data` rounded to bf16.
///
/// Not the f32 source. Every arm quantizes the same bf16 array, so bf16's own
/// rounding is common-mode and cannot favour either. bf16 is also what the KV
/// ring actually holds, and it is what makes affine's sideband 32 bits per
/// group rather than 64.
#[allow(
    clippy::expect_used,
    reason = "test fixture setup in this fn; a failure here is a test bug"
)]
fn bf16_reference(data: &[f32], head_dim: usize) -> Vec<f32> {
    let device = Device::Cpu;
    let rows = data.len() / head_dim;
    assert_eq!(
        rows * head_dim,
        data.len(),
        "bf16_reference: ragged fixture"
    );
    let stored = Array::from_f32_slice(data, &[rows as i32, head_dim as i32])
        .expect("from_f32_slice")
        .astype(Dtype::Bf16, device)
        .expect("astype bf16");
    to_f32_vec(&stored, device)
}

/// Quantize `reference` with MLX's `mode` at `group` and dequantize it back,
/// returning the decoded values and the bytes the store occupies.
#[allow(
    clippy::expect_used,
    reason = "test fixture setup in this fn; a failure here is a test bug"
)]
fn mlx_round_trip(reference: &[f32], head_dim: usize, mode: &str, group: i32) -> (Vec<f32>, u64) {
    let device = Device::Cpu;
    let rows = reference.len() / head_dim;
    let stored = Array::from_f32_slice(reference, &[rows as i32, head_dim as i32])
        .expect("from_f32_slice")
        .astype(Dtype::Bf16, device)
        .expect("astype bf16");

    let (codes, scales, biases) =
        quantize_mode(&stored, group, BITS, mode, device).expect("quantize");
    // The fp codecs emit no bias plane; affine does and needs it back.
    let biases_ref = (mode == "affine").then_some(&biases);
    let decoded =
        dequantize(&codes, &scales, biases_ref, group, BITS, mode, device).expect("dequantize");
    let bytes = array_bytes(&codes) + array_bytes(&scales) + biases_ref.map_or(0, array_bytes);
    (to_f32_vec(&decoded, device), bytes)
}

/// Decoded values only, for callers that do not need the rate.
fn mlx_round_trip_values(reference: &[f32], head_dim: usize, mode: &str, group: i32) -> Vec<f32> {
    mlx_round_trip(reference, head_dim, mode, group).0
}

/// Score a decode against the reference it was made from.
fn score(reference: &[f32], decoded: &[f32], stored_bytes: u64) -> RoundTrip {
    RoundTrip {
        // ‖x − x̂‖₂/‖x‖₂ is the SQNR ratio under a square root, so the existing
        // accumulator is the same measurement in another unit.
        rel_frob: 10.0_f64.powf(-sqnr_db(reference, decoded) / 20.0),
        max_abs: reference
            .iter()
            .zip(decoded.iter())
            .map(|(&r, &d)| f64::from((r - d).abs()))
            .fold(0.0_f64, f64::max),
        bits_per_value: stored_bits_per_value(stored_bytes, reference.len()),
    }
}

/// Round-trip `data` (`[rows, head_dim]`) through MLX's `mode` at `group` and
/// score it.
fn round_trip(data: &[f32], head_dim: usize, mode: &str, group: i32) -> RoundTrip {
    let reference = bf16_reference(data, head_dim);
    let (decoded, bytes) = mlx_round_trip(&reference, head_dim, mode, group);
    score(&reference, &decoded, bytes)
}

/// The fixtures under test, as `(name, generator)`.
///
/// `ratio == 1.0` makes [`outlier_channel_data`] the plain Gaussian fixture, so
/// the sweep's first rung is its own no-outlier control.
fn fixture(name: &str, head_dim: usize, seed: u64, ratio: f32) -> Vec<f32> {
    match name {
        "uniform-lcg" => lcg_data(ROWS * head_dim, seed),
        "gaussian" => gaussian_data(ROWS * head_dim, seed),
        "outlier" => outlier_channel_data(ROWS, head_dim, OUTLIER_CHANNELS, ratio, seed),
        other => panic!("fixture: unknown fixture {other}"),
    }
}

/// Precondition of the whole comparison: the two arms are byte-identical in
/// size, so any fidelity difference is bought with nothing.
///
/// Measured from the arrays MLX returns — shape times dtype width — not from a
/// constant restating the claim. Affine at group 32 is included as the
/// direction check: it is *not* rate-tied, and the assertion fails if it
/// silently were.
#[test]
fn mxfp8_and_affine_g128_spend_the_same_bits_per_value() {
    for head_dim in HEAD_DIMS {
        let data = fixture("outlier", head_dim, TEST_SEED, 20.0);
        let mxfp8 = round_trip(&data, head_dim, "mxfp8", MXFP8_GROUP);
        let affine = round_trip(&data, head_dim, "affine", AFFINE_GROUP);
        let affine_g32 = round_trip(&data, head_dim, "affine", MXFP8_GROUP);

        assert!(
            (mxfp8.bits_per_value - EQUAL_RATE_BITS_PER_VALUE).abs() < 1e-9,
            "head_dim={head_dim}: mxfp8 stored {} bits/value, expected {EQUAL_RATE_BITS_PER_VALUE}",
            mxfp8.bits_per_value,
        );
        assert!(
            (affine.bits_per_value - EQUAL_RATE_BITS_PER_VALUE).abs() < 1e-9,
            "head_dim={head_dim}: affine-g{AFFINE_GROUP} stored {} bits/value, expected \
             {EQUAL_RATE_BITS_PER_VALUE}",
            affine.bits_per_value,
        );
        assert!(
            affine_g32.bits_per_value > EQUAL_RATE_BITS_PER_VALUE,
            "head_dim={head_dim}: affine-g{MXFP8_GROUP} stored {} bits/value; it pays a bf16 \
             scale and bias every 32 values and cannot tie mxfp8's rate",
            affine_g32.bits_per_value,
        );
    }
}

/// The fixture's power, demonstrated rather than assumed.
///
/// Affine at group 32 and group 128 are the same codec differing only in how
/// often the scale re-adapts — the exact axis `mxfp8`'s per-32 E8M0 scale is
/// supposed to buy something on. On the outlier fixture the narrower group must
/// be clearly better; on the uniform LCG fixture, which has no dynamic range to
/// adapt to, it must be nearly a tie. A comparison run on the uniform fixture
/// therefore cannot see this axis at all, and neither could it see fp8's.
#[test]
fn the_outlier_fixture_separates_group_sizes_and_the_uniform_fixture_does_not() {
    for head_dim in HEAD_DIMS {
        let outlier = fixture("outlier", head_dim, TEST_SEED, 20.0);
        let sharp = round_trip(&outlier, head_dim, "affine", MXFP8_GROUP).rel_frob
            / round_trip(&outlier, head_dim, "affine", AFFINE_GROUP).rel_frob;

        let uniform = fixture("uniform-lcg", head_dim, TEST_SEED, 1.0);
        let blunt = round_trip(&uniform, head_dim, "affine", MXFP8_GROUP).rel_frob
            / round_trip(&uniform, head_dim, "affine", AFFINE_GROUP).rel_frob;

        println!("d={head_dim:3} affine-g32/affine-g128: outlier {sharp:.3}x, uniform {blunt:.3}x");
        assert!(
            sharp < 0.7,
            "head_dim={head_dim}: the outlier fixture put affine-g32 at {sharp:.3}x affine-g128; \
             a fixture that cannot separate two group sizes cannot separate two formats",
        );
        assert!(
            blunt > 0.9,
            "head_dim={head_dim}: the uniform fixture separated affine-g32 from affine-g128 at \
             {blunt:.3}x; it was expected to be blind to group size, so the negative control \
             for this comparison is not the control it is documented to be",
        );
    }
}

/// The R7 measurement: 8-bit float against 8-bit affine at identical rate, over
/// three fixtures, two head dims, five outlier ratios and three seeds.
///
/// Two fp8 arms are scored, and they are held to different bars because they
/// answer different questions. `mxfp8` is what MLX would actually store, so it
/// carries the declared PASS criterion: it must never be more faithful than
/// affine-g128. `ref-fp8` is the same format under a non-clipping E8M0 scale —
/// a best case no shipped encoder reaches — so it carries the declared
/// *falsifier* instead: it must never be materially better, [`MATERIAL_GAIN`]
/// or below. Holding the best case to the PASS bar would assert something
/// stronger than the question asked, and at `head_dim = 256` with extreme
/// outliers it is a tie rather than a loss.
///
/// Prints every cell so the verdict can be read off the transcript rather than
/// taken from the assertion.
#[test]
fn no_eight_bit_float_arm_is_more_faithful_than_affine_eight_bit_at_equal_rate() {
    let mut closest_shipped = f64::INFINITY;
    let mut closest_shipped_cell = String::new();
    let mut closest_best_case = f64::INFINITY;
    let mut closest_best_case_cell = String::new();

    for head_dim in HEAD_DIMS {
        for (name, ratios) in [
            ("uniform-lcg", &RATIOS[..1]),
            ("gaussian", &RATIOS[..1]),
            ("outlier", &RATIOS[..]),
        ] {
            for &ratio in ratios {
                for seed in SEEDS {
                    let data = fixture(name, head_dim, seed, ratio);
                    let reference = bf16_reference(&data, head_dim);

                    let (mxfp8_values, mxfp8_bytes) =
                        mlx_round_trip(&reference, head_dim, "mxfp8", MXFP8_GROUP);
                    let mxfp8 = score(&reference, &mxfp8_values, mxfp8_bytes);
                    let ideal = score(
                        &reference,
                        &ideal_mxfp8_round_trip(&reference, MXFP8_GROUP as usize),
                        mxfp8_bytes,
                    );
                    let (affine_values, affine_bytes) =
                        mlx_round_trip(&reference, head_dim, "affine", AFFINE_GROUP);
                    let affine = score(&reference, &affine_values, affine_bytes);

                    let mxfp8_vs = mxfp8.rel_frob / affine.rel_frob;
                    let ideal_vs = ideal.rel_frob / affine.rel_frob;

                    println!(
                        "d={head_dim:3} {name:11} r={ratio:5.1} seed={seed:#018x}  \
                         mxfp8 relF={:.5} max={:8.4}  ref-fp8 relF={:.5} max={:8.4}  \
                         affine-g128 relF={:.5} max={:6.4}  mxfp8/affine={mxfp8_vs:6.3}x  \
                         ref-fp8/affine={ideal_vs:6.3}x",
                        mxfp8.rel_frob,
                        mxfp8.max_abs,
                        ideal.rel_frob,
                        ideal.max_abs,
                        affine.rel_frob,
                        affine.max_abs,
                    );

                    let cell = format!("d={head_dim} {name} r={ratio} seed={seed:#x}");
                    if mxfp8_vs < closest_shipped {
                        closest_shipped = mxfp8_vs;
                        closest_shipped_cell = cell.clone();
                    }
                    if ideal_vs < closest_best_case {
                        closest_best_case = ideal_vs;
                        closest_best_case_cell = cell;
                    }
                }
            }
        }
    }

    println!(
        "closest cells: mxfp8 {closest_shipped_cell} at {closest_shipped:.3}x, \
         ref-fp8 {closest_best_case_cell} at {closest_best_case:.3}x affine-g128"
    );
    assert!(
        closest_shipped >= 1.0,
        "MLX's mxfp8 beat affine-g128 at equal rate in {closest_shipped_cell} \
         ({closest_shipped:.3}x), contradicting the recorded verdict that fp8 buys no fidelity \
         over affine at the same bits/value",
    );
    assert!(
        closest_best_case > MATERIAL_GAIN,
        "a non-clipping best-case E4M3 encoder reached {closest_best_case:.3}x affine-g128 in \
         {closest_best_case_cell}, at or past the {MATERIAL_GAIN}x declared falsifier. The \
         encoding buys real fidelity at equal size after all and route B is worth a throughput \
         cell",
    );
}

// ── An independent E4M3 / E8M0 reference ─────────────────────────────────────
//
// MLX's encoder is one implementation of the OCP microscaling format, and a
// verdict that rests on it alone is a verdict about MLX rather than about fp8.
// The reference below is written from the format definition — sign, 4-bit
// exponent at bias 7, 3-bit mantissa, no infinities, saturating at 448 — and is
// used two ways: to check that MLX's store really is that format, and to
// supply a *best-case* fp8 arm that MLX's implementation cannot be blamed for.

/// Largest finite E4M3 magnitude: `1.75 · 2^8`.
const E4M3_MAX: f32 = 448.0;

/// One bf16 unit in the last place, relative: bf16 carries 8 significand bits,
/// so this is the widest legitimate disagreement between two decodes of the
/// same bf16-valued store.
const BF16_ULP: f32 = 1.0 / 256.0;

/// Value of the E4M3 code `bits`, from the field definition.
///
/// Exponent bias is **7**, against `half`'s 15 — the whole reason a widen to
/// `half` is not a bitcast. `0x7F` / `0xFF` is the format's NaN pattern; MLX's
/// widener maps it to a finite 480 and the encoder never emits it (it saturates
/// at `0x7E`), so it is decoded here the same way rather than special-cased.
fn e4m3_value(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 3) & 0x0F);
    let mantissa = f32::from(bits & 0x07) / 8.0;
    let magnitude = if exponent == 0 {
        mantissa * 2.0_f32.powi(-6)
    } else {
        (1.0 + mantissa) * 2.0_f32.powi(exponent - 7)
    };
    sign * magnitude
}

/// Value of the E8M0 shared-scale code `bits`: a bare power of two at bias 127.
fn e8m0_value(bits: u8) -> f32 {
    2.0_f32.powi(i32::from(bits) - 127)
}

/// Value of the IEEE `half` whose bit pattern is `bits` — the thing MLX's
/// widener actually bitcasts to before it multiplies.
fn half_value(bits: u16) -> f32 {
    let sign = if bits & 0x8000 == 0 { 1.0 } else { -1.0 };
    let exponent = i32::from((bits >> 10) & 0x1F);
    let mantissa = f32::from(bits & 0x03FF) / 1024.0;
    let magnitude = if exponent == 0 {
        mantissa * 2.0_f32.powi(-14)
    } else {
        (1.0 + mantissa) * 2.0_f32.powi(exponent - 15)
    };
    sign * magnitude
}

/// The 128 non-negative E4M3 magnitudes, ascending — the reconstruction grid.
fn e4m3_magnitudes() -> Vec<f32> {
    let mut m: Vec<f32> = (0u8..128).map(e4m3_value).collect();
    m.sort_by(f32::total_cmp);
    m
}

/// Round `v` to the nearest E4M3 magnitude, ties to the even code, saturating.
///
/// Written as a search over the reconstruction grid rather than as bit
/// manipulation: this is the oracle, and an oracle that reuses the encoder's
/// own trick proves nothing about it.
fn e4m3_round(v: f32, grid: &[f32]) -> f32 {
    let a = v.abs().min(E4M3_MAX);
    let upper = grid.partition_point(|&g| g < a);
    let sign = if v < 0.0 { -1.0 } else { 1.0 };
    if upper == 0 {
        return sign * grid[0];
    }
    if upper >= grid.len() {
        return sign * E4M3_MAX;
    }
    let (lo, hi) = (grid[upper - 1], grid[upper]);
    let pick = match (a - lo).partial_cmp(&(hi - a)) {
        Some(std::cmp::Ordering::Less) => lo,
        Some(std::cmp::Ordering::Greater) => hi,
        // Tie: the even code. Within a binade the grid step is constant, so the
        // lower of the pair is the even one exactly when its index is even.
        _ if (upper - 1) % 2 == 0 => lo,
        _ => hi,
    };
    sign * pick
}

/// Best-case mxfp8 round trip: per-group E8M0 scale chosen as the smallest
/// power of two that puts the group maximum at or below `E4M3_MAX`, so no value
/// is ever clipped, then round-to-nearest E4M3.
///
/// Deliberately more generous than any shipped encoder — it is here so the
/// verdict cannot be answered with "MLX's mxfp8 encoder is the problem".
fn ideal_mxfp8_round_trip(reference: &[f32], group: usize) -> Vec<f32> {
    let grid = e4m3_magnitudes();
    let mut out = Vec::with_capacity(reference.len());
    for chunk in reference.chunks(group) {
        let max_abs = chunk.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
        if max_abs == 0.0 {
            out.extend(chunk.iter().map(|_| 0.0));
            continue;
        }
        let exponent = (max_abs / E4M3_MAX).log2().ceil().clamp(-127.0, 127.0) as i32;
        let scale = 2.0_f32.powi(exponent);
        out.extend(chunk.iter().map(|&v| e4m3_round(v / scale, &grid) * scale));
    }
    out
}

/// The decisive premise of the recorded verdict, checked against the format
/// itself: reinterpreting an E4M3 code's bits as a `half` leaves the value low
/// by exactly `2^8`, because the two formats' exponent biases are 7 and 15.
///
/// Checked over all 256 codes, so it covers subnormals and the NaN pattern as
/// well as normals. This is why an fp8 "widen" contains a floating-point
/// multiply and is not a bitcast.
#[test]
fn widening_an_e4m3_code_to_half_is_low_by_exactly_two_to_the_eighth() {
    for bits in 0u8..=255 {
        let reinterpreted = half_value(u16::from(bits & 127) << 7);
        let corrected = reinterpreted * 256.0;
        let want = e4m3_value(bits).abs();
        assert_eq!(
            corrected, want,
            "code {bits:#04x}: bitcast-to-half gives {reinterpreted}, x256 gives {corrected}, \
             the E4M3 field decode gives {want}",
        );
    }
}

/// MLX's `mxfp8` store really is OCP E4M3 codes times an E8M0 power of two.
///
/// Decodes the raw `(codes, scales)` MLX emits with the field-definition
/// reference above and compares against MLX's own `dequantize`. Agreement to
/// bf16's own resolution means the numbers measured here describe the format,
/// not an MLX-private layout — and it is what lets the CPU device stand in for
/// the Metal one, since both read the same store.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test fixture setup in this fn; a failure here is a test bug"
)]
fn the_mlx_mxfp8_store_decodes_as_e4m3_codes_times_an_e8m0_scale() {
    let device = Device::Cpu;
    let head_dim = 128;
    let data = fixture("outlier", head_dim, TEST_SEED, 20.0);
    let stored = Array::from_f32_slice(&data, &[ROWS as i32, head_dim as i32])
        .expect("from_f32_slice")
        .astype(Dtype::Bf16, device)
        .expect("astype bf16");

    let (codes, scales, _) =
        quantize_mode(&stored, MXFP8_GROUP, BITS, "mxfp8", device).expect("quantize");
    let mlx = to_f32_vec(
        &dequantize(&codes, &scales, None, MXFP8_GROUP, BITS, "mxfp8", device).expect("dequantize"),
        device,
    );

    codes.eval().expect("eval");
    scales.eval().expect("eval");
    let code_bytes = codes.to_bytes().expect("codes to_bytes");
    let scale_bytes = scales.to_bytes().expect("scales to_bytes");
    assert_eq!(code_bytes.len(), data.len(), "one code byte per value");
    assert_eq!(
        scale_bytes.len(),
        data.len() / MXFP8_GROUP as usize,
        "one E8M0 scale byte per group",
    );

    for (i, (&code, &got)) in code_bytes.iter().zip(mlx.iter()).enumerate() {
        let want = e4m3_value(code) * e8m0_value(scale_bytes[i / MXFP8_GROUP as usize]);
        assert!(
            (want - got).abs() <= BF16_ULP * want.abs(),
            "value {i}: code {code:#04x} x scale {:#04x} decodes to {want} by the field \
             definition, MLX returned {got}",
            scale_bytes[i / MXFP8_GROUP as usize],
        );
    }
}

/// MLX's `mxfp8` encoder clips: its E8M0 scale is `round(log2(max/448))`, and
/// when that rounds *down* the group maximum lands above E4M3's 448 ceiling and
/// saturates, losing up to half its magnitude.
///
/// Measured as the fraction of groups whose reconstructed maximum falls below
/// 90% of the true one. The non-clipping reference arm must show none, which is
/// what makes this a statement about the encoder rather than about the format —
/// and is why the sweep carries the reference arm at all.
#[test]
fn the_mlx_mxfp8_encoder_clips_group_maxima_and_the_reference_encoder_does_not() {
    let head_dim = 128;
    let group = MXFP8_GROUP as usize;
    let data = fixture("gaussian", head_dim, TEST_SEED, 1.0);
    let reference = bf16_reference(&data, head_dim);
    let mlx = mlx_round_trip_values(&reference, head_dim, "mxfp8", MXFP8_GROUP);
    let ideal = ideal_mxfp8_round_trip(&reference, group);

    let clipped = |decoded: &[f32]| -> f64 {
        let groups = reference.len() / group;
        let hit = reference
            .chunks(group)
            .zip(decoded.chunks(group))
            .filter(|(src, dec)| {
                let want = src.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                let got = dec.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                want > 0.0 && got < 0.9 * want
            })
            .count();
        hit as f64 / groups as f64
    };

    let mlx_clipped = clipped(&mlx);
    let ideal_clipped = clipped(&ideal);
    println!(
        "groups clipped: mlx mxfp8 {mlx_clipped:.3}, non-clipping reference {ideal_clipped:.3}"
    );

    assert!(
        mlx_clipped > 0.25,
        "MLX's mxfp8 encoder clipped only {mlx_clipped:.3} of groups; the round-to-nearest E8M0 \
         scale rule this documents would clip far more, so either the rule changed or this \
         measurement stopped measuring it",
    );
    assert_eq!(
        ideal_clipped, 0.0,
        "the reference encoder clipped {ideal_clipped:.3} of groups; it picks the smallest \
         non-clipping power of two by construction and must clip none",
    );
}
