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

/// Round-trip `data` (`[rows, head_dim]`) through `mode` at `group`.
///
/// The reference is the **bf16** array the encoder is handed, not the f32
/// source: both arms round to bf16 first, so that rounding is common-mode and
/// cannot favour either. bf16 is also what the KV ring actually holds, and it
/// is what makes affine's sideband 32 bits per group rather than 64.
#[allow(
    clippy::expect_used,
    reason = "test fixture setup in this fn; a failure here is a test bug"
)]
fn round_trip(data: &[f32], head_dim: usize, mode: &str, group: i32) -> RoundTrip {
    let device = Device::Cpu;
    let rows = data.len() / head_dim;
    assert_eq!(rows * head_dim, data.len(), "round_trip: ragged fixture");

    let stored = Array::from_f32_slice(data, &[rows as i32, head_dim as i32])
        .expect("from_f32_slice")
        .astype(Dtype::Bf16, device)
        .expect("astype bf16");
    let reference = to_f32_vec(&stored, device);

    let (codes, scales, biases) =
        quantize_mode(&stored, group, BITS, mode, device).expect("quantize");
    // The fp codecs emit no bias plane; affine does and needs it back.
    let biases_ref = (mode == "affine").then_some(&biases);
    let decoded_arr =
        dequantize(&codes, &scales, biases_ref, group, BITS, mode, device).expect("dequantize");
    let decoded = to_f32_vec(&decoded_arr, device);

    let stored_bytes =
        array_bytes(&codes) + array_bytes(&scales) + biases_ref.map_or(0, array_bytes);

    let max_abs = reference
        .iter()
        .zip(decoded.iter())
        .map(|(&r, &d)| f64::from((r - d).abs()))
        .fold(0.0_f64, f64::max);

    RoundTrip {
        // ‖x − x̂‖₂/‖x‖₂ is the SQNR ratio under a square root, so the existing
        // accumulator is the same measurement in another unit.
        rel_frob: 10.0_f64.powf(-sqnr_db(&reference, &decoded) / 20.0),
        max_abs,
        bits_per_value: stored_bits_per_value(stored_bytes, data.len()),
    }
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

        assert!(
            sharp < 0.5,
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

/// The R7 measurement: `mxfp8` against affine 8-bit at identical rate, over
/// three fixtures, two head dims, five outlier ratios and three seeds.
///
/// Prints every cell so the verdict can be read off the transcript rather than
/// taken from the assertion.
#[test]
fn mxfp8_is_never_more_faithful_than_affine_eight_bit_at_equal_rate() {
    let mut worst_ratio = f64::INFINITY;
    let mut worst_cell = String::new();

    for head_dim in HEAD_DIMS {
        for (name, ratios) in [
            ("uniform-lcg", &RATIOS[..1]),
            ("gaussian", &RATIOS[..1]),
            ("outlier", &RATIOS[..]),
        ] {
            for &ratio in ratios {
                for seed in SEEDS {
                    let data = fixture(name, head_dim, seed, ratio);
                    let mxfp8 = round_trip(&data, head_dim, "mxfp8", MXFP8_GROUP);
                    let affine = round_trip(&data, head_dim, "affine", AFFINE_GROUP);
                    let ratio_of_errors = mxfp8.rel_frob / affine.rel_frob;

                    println!(
                        "d={head_dim:3} {name:11} ratio={ratio:5.1} seed={seed:#018x}  \
                         mxfp8 relF={:.5} maxabs={:.4}  affine-g128 relF={:.5} maxabs={:.4}  \
                         mxfp8/affine={ratio_of_errors:.3}x",
                        mxfp8.rel_frob, mxfp8.max_abs, affine.rel_frob, affine.max_abs,
                    );

                    if ratio_of_errors < worst_ratio {
                        worst_ratio = ratio_of_errors;
                        worst_cell = format!("d={head_dim} {name} ratio={ratio} seed={seed:#x}");
                    }
                }
            }
        }
    }

    println!("closest cell: {worst_cell} at {worst_ratio:.3}x");
    assert!(
        worst_ratio >= 1.0,
        "mxfp8 beat affine-g128 at equal rate in {worst_cell} ({worst_ratio:.3}x). The declared \
         falsifier is {MATERIAL_GAIN}x or better; anything below 1.0x already contradicts the \
         recorded verdict that mxfp8 buys no fidelity over affine at the same bits/value",
    );
}
