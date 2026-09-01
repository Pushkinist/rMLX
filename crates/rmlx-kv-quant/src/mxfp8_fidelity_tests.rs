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
//! neighbours nothing, but every small value pays ~2^-4 of itself.
//!
//! # Criterion, declared before any fidelity number was measured
//!
//! Metric: **relative Frobenius error** `‖x − x̂‖₂ / ‖x‖₂` against the bf16
//! tensor actually handed to the encoder, plus max-abs error as a secondary.
//! Both arms see the identical bf16 input, so bf16 rounding is common-mode.
//!
//! * **PASS** (the negative verdict on fp8 KV stands): fp8 relative Frobenius
//!   error `>=` affine-g128 relative Frobenius error in **every** cell.
//! * **FAIL / falsifier**: fp8 error `<= 0.85 ×` affine-g128 error — a >=15%
//!   fidelity gain at identical size.
//! * **INCONCLUSIVE**: anything in between, i.e. fp8 ahead somewhere but never
//!   materially.
//!
//! Deterministic: fixed seeds, `Device::Cpu`, no GPU and no model serving.
//!
//! # Two fp8 arms, and which one the criterion belongs on
//!
//! `mxfp8` is what MLX would actually store. `ref-fp8` is the same E4M3 codes
//! under a **non-clipping** E8M0 scale, which no shipped encoder reaches: MLX's
//! `fp_quantize` rounds its shared exponent to nearest and saturates the group
//! maximum whenever that rounds down
//! ([`the_mlx_mxfp8_encoder_clips_group_maxima_and_the_reference_encoder_does_not`]).
//!
//! That defect, not the encoding, is most of `mxfp8`'s measured loss. So the
//! **declared criterion is carried by `ref-fp8`**, the arm that isolates the
//! format. `mxfp8` carries two weaker duties: it must not beat affine, and its
//! margin over `ref-fp8` is pinned, so an upstream fix to the E8M0 rounding
//! turns this file red instead of passing silently on a premise that no longer
//! holds.
//!
//! # What the sweep answers, on each arm
//!
//! The two arms land in **different buckets of the declared criterion**, and
//! collapsing them into one verdict would overstate the result.
//!
//! * **MLX's shipped `mxfp8`: PASS.** Never below `4.04x` affine-g128 in any of
//!   the 96 cells. Most of that margin is the clipping defect, not the format.
//! * **The format itself (`ref-fp8`): INCONCLUSIVE.** It is ahead of affine in
//!   10 of 96 cells, all of them low outlier density at high magnitude, by at
//!   most 6.1% — and the per-cell seed ranges straddle 1.0, so those cells are
//!   ties rather than wins. The global minimum is `0.939x`, nowhere near the
//!   `0.85x` the falsifier requires. "Ahead somewhere, never materially" is
//!   exactly the middle bucket declared in advance.
//!
//! So E4M3 at equal rate is a **wash** against affine, not a loss, and the
//! recorded verdict's practical conclusion survives on the cost and coverage
//! arguments rather than on fidelity.
//!
//! # The fixture is the trap
//!
//! Three fixtures live in [`crate::test_utils`] and they do not measure the
//! same thing. [`lcg_data`](crate::test_utils::lcg_data) is i.i.d. uniform on
//! `[-1, 1]` — it has **no dynamic range to speak of**, which is the property
//! fp8 exists to exploit, so a format comparison run on it returns "no
//! difference" for reasons that have nothing to do with the formats.
//! [`outlier_channel_data`](crate::test_utils::outlier_channel_data) is the
//! literature-shaped K-cache model.
//!
//! Its **density** is load-bearing and is swept here as a first-class axis.
//! [`crate::test_utils::OUTLIER_CHANNELS`] is 4, a constant chosen for the
//! rotation gates so that every affine group of *64* holds an outlier. At
//! `head_dim = 128` those four channels land in four distinct groups of 32 —
//! one in **every** group — which cancels the per-group dynamic range that
//! E8M0 exists to capture before the measurement starts. A verdict taken at
//! that density alone would be a statement about channel placement: it is the
//! density sweep, not the magnitude sweep, that moves `ref-fp8` from a clean
//! loss to a tie. The sweep therefore runs 1, 2, 4 and 16 channels, prints the
//! fraction of groups that actually contain one, and includes a
//! literature-faithful ~0.8% cell where most groups of 32 are clean.
//!
//! # Positive control
//!
//! Showing that a fixture separates affine-g32 from affine-g128 is not enough:
//! both are absolute-step codecs, so that only establishes power on the group
//! size axis. [`the_log_uniform_control_separates_the_formats_on_relative_precision`]
//! adds the control that matters — log-uniform magnitudes spanning decades
//! inside every group, where E4M3's relative precision must win if the setup
//! can see the format axis at all. It does win there, on a scale-invariant
//! metric, and still loses on relative Frobenius. That is not a fixture defect;
//! it is what an energy-weighted metric is, and the module's verdict is stated
//! on the metric the question was asked in.
//!
//! # Why not real KV tensors
//!
//! A real K/V capture is the more honest fixture and is **not reachable in this
//! cell**: nothing in the tree ships a captured KV tensor, and producing one
//! means a forward pass over a real checkpoint, which is a GPU/serving cell with
//! its own window.

use crate::test_utils::{
    gaussian_data, lcg_data, outlier_channel_data, outlier_channels, sqnr_db,
    stored_bits_per_value, TEST_SEED,
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

/// The falsifier's threshold: fp8 counts as *materially* better only if its
/// relative Frobenius error is at most this fraction of affine-g128's.
const MATERIAL_GAIN: f64 = 0.85;

/// Rows in every fixture here.
const ROWS: usize = 256;

/// Head dims under test: 128 is Bonsai / Qwen3, 256 is gemma-4 and Qwen3.6.
/// Both are exact multiples of both group sizes, so neither arm gets a ragged
/// final group the other does not.
const HEAD_DIMS: [usize; 2] = [128, 256];

/// Outlier channel counts swept. 1 is the literature-faithful density (0.8% of
/// a 128-wide head, 0.4% of a 256-wide one) and leaves most groups of 32 clean;
/// 16 saturates them. 4 is [`crate::test_utils::OUTLIER_CHANNELS`], carried so
/// the rotation gates' constant is one rung of a curve rather than the whole
/// measurement.
const OUTLIER_DENSITIES: [usize; 4] = [1, 2, 4, 16];

/// Outlier magnitude ratios swept, spanning the reported order of magnitude for
/// emergent outlier features and well past it.
const OUTLIER_RATIOS: [f32; 3] = [5.0, 20.0, 100.0];

/// Decade spans of the log-uniform positive control.
const LOG_UNIFORM_DECADES: [f32; 2] = [3.0, 6.0];

/// Seeds per cell, so no number here rests on one draw.
const SEEDS: [u64; 3] = [TEST_SEED, TEST_SEED ^ 0x5DEE_CE66, TEST_SEED ^ 0x0BAD_C0DE];

/// Cells the sweep must visit: two head dims x three seeds x (uniform +
/// gaussian + 4 densities x 3 ratios + 2 decade spans) = 96.
///
/// A **literal**, deliberately not `OUTLIER_DENSITIES.len() * ...`. Derived
/// from the axes it would move with them, so shrinking an axis would still
/// match and every assertion below would pass on a sweep that no longer covers
/// what the module documents. Update it by hand when an axis changes, which is
/// the point.
const EXPECTED_CELLS: usize = 96;

// ── Metrics ──────────────────────────────────────────────────────────────────

/// One quantizer round trip: what it cost in fidelity and what it cost in rate.
struct RoundTrip {
    /// `‖x − x̂‖₂ / ‖x‖₂` against the bf16 input the encoder actually saw.
    /// Energy-weighted, so it is dominated by the largest values.
    rel_frob: f64,
    /// Median per-element `|x − x̂| / |x|`. Scale-invariant: unlike `rel_frob`
    /// it weights a small value as heavily as a large one, which is the axis
    /// E4M3's fixed mantissa is built for.
    median_rel: f64,
    /// `max |x − x̂|`, same reference.
    max_abs: f64,
    /// Bits per input value across every buffer the encoder emitted.
    bits_per_value: f64,
}

/// Score a decode against the reference it was made from.
fn score(reference: &[f32], decoded: &[f32], stored_bytes: u64) -> RoundTrip {
    assert_eq!(
        reference.len(),
        decoded.len(),
        "score: decode has {} values against {} in the reference",
        decoded.len(),
        reference.len(),
    );

    let mut relatives: Vec<f64> = reference
        .iter()
        .zip(decoded.iter())
        .filter(|(&r, _)| r != 0.0)
        .map(|(&r, &d)| f64::from((r - d).abs()) / f64::from(r.abs()))
        .collect();
    relatives.sort_by(f64::total_cmp);

    RoundTrip {
        // ‖x − x̂‖₂/‖x‖₂ is the SQNR ratio under a square root, so the existing
        // accumulator is the same measurement in another unit.
        rel_frob: 10.0_f64.powf(-sqnr_db(reference, decoded) / 20.0),
        median_rel: if relatives.is_empty() {
            0.0
        } else {
            relatives[relatives.len() / 2]
        },
        max_abs: reference
            .iter()
            .zip(decoded.iter())
            .map(|(&r, &d)| f64::from((r - d).abs()))
            .fold(0.0_f64, f64::max),
        bits_per_value: stored_bits_per_value(stored_bytes, reference.len()),
    }
}

// ── MLX round trips ──────────────────────────────────────────────────────────

#[allow(
    clippy::expect_used,
    reason = "test helper: an MLX failure here is a test bug, not a result"
)]
fn to_f32_vec(a: &Array, device: Device) -> Vec<f32> {
    a.astype(Dtype::F32, device)
        .expect("astype f32")
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("chunks_exact(4) yields 4 bytes")))
        .collect()
}

/// Bytes an MLX array occupies, from `mlx_array_nbytes`.
///
/// Not `shape() x dtype().itemsize()`: both of those fail open — `dtype()`
/// falls back to `U8` for anything it does not model and `shape()` returns
/// empty on a null handle, whose product is 1 — so a store could under-report
/// by 4x and the rate assertion would still read 8.250 and pass.
#[allow(
    clippy::expect_used,
    reason = "test helper: an MLX failure here is a test bug, not a result"
)]
fn array_bytes(a: &Array) -> u64 {
    a.to_bytes().expect("to_bytes").len() as u64
}

/// What an MLX quantize/dequantize round trip produced.
struct Decoded {
    values: Vec<f32>,
    bytes: u64,
    dtype: Dtype,
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

/// Quantize `reference` with MLX's `mode` at `group` and dequantize it back.
#[allow(
    clippy::expect_used,
    reason = "test fixture setup in this fn; a failure here is a test bug"
)]
fn mlx_round_trip(reference: &[f32], head_dim: usize, mode: &str, group: i32) -> Decoded {
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
    Decoded {
        dtype: decoded.dtype(),
        values: to_f32_vec(&decoded, device),
        bytes: array_bytes(&codes) + array_bytes(&scales) + biases_ref.map_or(0, array_bytes),
    }
}

/// Round-trip `data` (`[rows, head_dim]`) through MLX's `mode` at `group` and
/// score it.
fn round_trip(data: &[f32], head_dim: usize, mode: &str, group: i32) -> RoundTrip {
    let reference = bf16_reference(data, head_dim);
    let decoded = mlx_round_trip(&reference, head_dim, mode, group);
    score(&reference, &decoded.values, decoded.bytes)
}

// ── Fixtures ─────────────────────────────────────────────────────────────────

/// Which fixture a cell draws from, and the parameter that shapes it.
#[derive(Clone, Copy)]
enum Fixture {
    /// i.i.d. uniform on `[-1, 1]`: the blind control.
    UniformLcg,
    /// i.i.d. standard normal.
    Gaussian,
    /// Gaussian base with `channels` persistent channels scaled by `ratio`.
    Outlier { channels: usize, ratio: f32 },
    /// Magnitudes log-uniform over `decades`, random signs: the positive
    /// control for the format axis.
    LogUniform { decades: f32 },
}

impl Fixture {
    fn label(self) -> String {
        match self {
            Fixture::UniformLcg => "uniform-lcg".to_string(),
            Fixture::Gaussian => "gaussian".to_string(),
            Fixture::Outlier { channels, ratio } => format!("outlier c={channels} r={ratio:.0}"),
            Fixture::LogUniform { decades } => format!("log-uniform d={decades:.0}"),
        }
    }

    fn draw(self, head_dim: usize, seed: u64) -> Vec<f32> {
        match self {
            Fixture::UniformLcg => lcg_data(ROWS * head_dim, seed),
            Fixture::Gaussian => gaussian_data(ROWS * head_dim, seed),
            Fixture::Outlier { channels, ratio } => {
                outlier_channel_data(ROWS, head_dim, channels, ratio, seed)
            }
            Fixture::LogUniform { decades } => log_uniform_data(ROWS * head_dim, seed, decades),
        }
    }

    /// Fraction of `mxfp8` groups that contain at least one outlier channel —
    /// the diagnostic that makes channel/group alignment visible instead of
    /// silently deciding the result. 1.0 means every group holds one and the
    /// per-group dynamic range E8M0 exists to capture has been cancelled.
    fn groups_touched(self, head_dim: usize) -> f64 {
        let group = MXFP8_GROUP as usize;
        let groups = head_dim / group;
        match self {
            Fixture::Outlier { channels, .. } => {
                let mut touched: Vec<usize> = outlier_channels(head_dim, channels)
                    .into_iter()
                    .map(|c| c / group)
                    .collect();
                touched.sort_unstable();
                touched.dedup();
                touched.len() as f64 / groups as f64
            }
            // Every other fixture is i.i.d. across channels: no channel is
            // structurally larger than another, so every group is alike.
            Fixture::UniformLcg | Fixture::Gaussian | Fixture::LogUniform { .. } => 1.0,
        }
    }
}

/// Magnitudes log-uniform over `decades`, signs from the same pinned LCG.
///
/// Every group of 32 spans the full range, so the dynamic range is *inside* the
/// group rather than concentrated in a few channels — the shape under which
/// E4M3's fixed relative precision has the most to offer against a per-group
/// absolute step.
fn log_uniform_data(n: usize, seed: u64, decades: f32) -> Vec<f32> {
    lcg_data(2 * n, seed)
        .chunks_exact(2)
        .map(|pair| {
            // lcg_data is uniform on [-1, 1]; fold to [0, 1) for the exponent.
            let unit = (pair[0] + 1.0) / 2.0;
            let magnitude = 10.0_f32.powf(-decades * unit);
            if pair[1] < 0.0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect()
}

/// Every cell of the sweep, in visit order.
fn cells() -> Vec<Fixture> {
    let mut out = vec![Fixture::UniformLcg, Fixture::Gaussian];
    for channels in OUTLIER_DENSITIES {
        for ratio in OUTLIER_RATIOS {
            out.push(Fixture::Outlier { channels, ratio });
        }
    }
    for decades in LOG_UNIFORM_DECADES {
        out.push(Fixture::LogUniform { decades });
    }
    out
}

// ── An independent E4M3 / E8M0 reference ─────────────────────────────────────
//
// MLX's encoder is one implementation of the OCP microscaling format, and a
// verdict that rests on it alone is a verdict about MLX rather than about fp8.
// The reference below is written from the format definition — sign, 4-bit
// exponent at bias 7, 3-bit mantissa, no infinities, saturating at 448 — and is
// used two ways: to check that MLX's store really is that format, and to supply
// a best-case fp8 arm that MLX's implementation cannot be blamed for.

/// Largest finite E4M3 magnitude: `1.75 · 2^8`.
const E4M3_MAX: f32 = 448.0;

/// One bf16 unit in the last place, relative: bf16 carries 8 significand bits,
/// so this is the widest legitimate disagreement between two decodes of the
/// same bf16-valued store.
const BF16_ULP: f32 = 1.0 / 256.0;

/// Value of the E4M3 code `bits`, from the field definition.
///
/// Exponent bias is **7**, against `half`'s 15 — the reason a widen to `half`
/// is not a bitcast.
///
/// `0x7F` / `0xFF` is the format's NaN pattern and is decoded as NaN, matching
/// `rmlx_quant::fp8::e4m3_decode` on the weight side. **MLX's own widener
/// disagrees**: it reconstructs by bit-shifting into a `half`, which maps that
/// pattern to a finite 480.0.
/// [`mlx_widens_e4m3_codes_by_bitcasting_into_half_and_multiplying_by_256`]
/// pins what MLX does, and
/// [`the_mlx_mxfp8_store_decodes_as_e4m3_codes_times_an_e8m0_scale`] pins that
/// MLX's encoder never emits the pattern, which is what keeps the divergence
/// inert rather than a live disagreement between two decoders.
fn e4m3_value(bits: u8) -> f32 {
    if bits & 0x7F == 0x7F {
        return f32::NAN;
    }
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
/// widener bitcasts to before it multiplies.
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

/// The 127 finite non-negative E4M3 magnitudes, ascending — the reconstruction
/// grid. `0x7F` is NaN and is not a reconstruction point.
fn e4m3_magnitudes() -> Vec<f32> {
    let mut m: Vec<f32> = (0u8..127).map(e4m3_value).collect();
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
///
/// # Panics
///
/// Panics (test-only) unless `head_dim` is a whole number of groups. MLX groups
/// within a row; chunking the flat buffer only agrees with that while the row
/// divides evenly, and a ragged head dim would silently make this arm
/// incomparable to the MLX ones.
fn ideal_mxfp8_round_trip(reference: &[f32], head_dim: usize) -> Vec<f32> {
    let group = MXFP8_GROUP as usize;
    assert!(
        head_dim.is_multiple_of(group),
        "ideal_mxfp8_round_trip: head_dim {head_dim} is not a whole number of groups of {group}; \
         flat chunking would straddle rows and stop matching MLX's per-row grouping",
    );
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

/// Bytes the reference encoder's own store would occupy: one code byte per
/// value plus one E8M0 byte per group.
fn ideal_store_bytes(values: usize) -> u64 {
    values as u64 + (values / MXFP8_GROUP as usize) as u64
}

// ── Rate ─────────────────────────────────────────────────────────────────────

/// Precondition of the whole comparison: the two arms are byte-identical in
/// size, so any fidelity difference is bought with nothing.
///
/// Measured from `mlx_array_nbytes` on the arrays MLX returns, not from a
/// constant restating the claim. Affine at group 32 is included as the
/// direction check: it is *not* rate-tied, and the assertion fails if it
/// silently were. The reference encoder's own store is checked against the same
/// figure, since the sweep scores that arm at this rate.
#[test]
fn mxfp8_and_affine_g128_spend_the_same_bits_per_value() {
    for head_dim in HEAD_DIMS {
        let data = Fixture::Outlier {
            channels: 4,
            ratio: 20.0,
        }
        .draw(head_dim, TEST_SEED);
        let mxfp8 = round_trip(&data, head_dim, "mxfp8", MXFP8_GROUP);
        let affine = round_trip(&data, head_dim, "affine", AFFINE_GROUP);
        let affine_g32 = round_trip(&data, head_dim, "affine", MXFP8_GROUP);
        let ideal = stored_bits_per_value(ideal_store_bytes(data.len()), data.len());

        for (name, rate) in [
            ("mxfp8", mxfp8.bits_per_value),
            ("affine-g128", affine.bits_per_value),
            ("reference fp8", ideal),
        ] {
            assert!(
                (rate - EQUAL_RATE_BITS_PER_VALUE).abs() < 1e-9,
                "head_dim={head_dim}: {name} stored {rate} bits/value, expected \
                 {EQUAL_RATE_BITS_PER_VALUE}",
            );
        }
        assert!(
            affine_g32.bits_per_value > EQUAL_RATE_BITS_PER_VALUE,
            "head_dim={head_dim}: affine-g{MXFP8_GROUP} stored {} bits/value; it pays a bf16 \
             scale and bias every 32 values and cannot tie mxfp8's rate",
            affine_g32.bits_per_value,
        );
    }
}

// ── Fixture power ────────────────────────────────────────────────────────────

/// Power on the group-size axis: affine at group 32 and group 128 are the same
/// codec differing only in how often the scale re-adapts.
///
/// This establishes that the outlier fixture has per-group dynamic range for a
/// scale to adapt to, and that the uniform fixture has none. It does **not**
/// establish power on the format axis — both arms here are absolute-step
/// codecs — which is what
/// [`the_log_uniform_control_separates_the_formats_on_relative_precision`] is
/// for.
#[test]
fn the_outlier_fixture_separates_group_sizes_and_the_uniform_fixture_does_not() {
    for head_dim in HEAD_DIMS {
        let outlier = Fixture::Outlier {
            channels: 4,
            ratio: 20.0,
        }
        .draw(head_dim, TEST_SEED);
        let sharp = round_trip(&outlier, head_dim, "affine", MXFP8_GROUP).rel_frob
            / round_trip(&outlier, head_dim, "affine", AFFINE_GROUP).rel_frob;

        let uniform = Fixture::UniformLcg.draw(head_dim, TEST_SEED);
        let blunt = round_trip(&uniform, head_dim, "affine", MXFP8_GROUP).rel_frob
            / round_trip(&uniform, head_dim, "affine", AFFINE_GROUP).rel_frob;

        println!("d={head_dim:3} affine-g32/affine-g128: outlier {sharp:.3}x, uniform {blunt:.3}x");
        assert!(
            sharp < 0.7,
            "head_dim={head_dim}: the outlier fixture put affine-g32 at {sharp:.3}x affine-g128; \
             a fixture that cannot separate two group sizes cannot separate two formats",
        );
        assert!(
            (0.9..1.1).contains(&blunt),
            "head_dim={head_dim}: the uniform fixture put affine-g32 at {blunt:.3}x affine-g128; \
             it was expected to be blind to group size in either direction, so the negative \
             control for this comparison is not the control it is documented to be",
        );
    }
}

/// Power on the **format** axis: log-uniform magnitudes spanning decades inside
/// every group, where E4M3's fixed relative precision must win if the setup can
/// separate a relative-precision format from an absolute-step one at all.
///
/// It wins, decisively, on the scale-invariant metric — and still loses on
/// relative Frobenius, by a wide margin. Both facts are asserted, because the
/// second is the load-bearing one: relative Frobenius is energy-weighted, and a
/// uniform step over a group's range spends its precision exactly where the
/// energy is. That is why no fixture inverts the verdict on that metric, and it
/// is a property of the metric rather than of any fixture. The metric is the one
/// the question was asked in, and it is the right one for KV, where the cache is
/// consumed by inner products.
#[test]
fn the_log_uniform_control_separates_the_formats_on_relative_precision() {
    for head_dim in HEAD_DIMS {
        for decades in LOG_UNIFORM_DECADES {
            let data = Fixture::LogUniform { decades }.draw(head_dim, TEST_SEED);
            let reference = bf16_reference(&data, head_dim);
            let fp8 = score(
                &reference,
                &ideal_mxfp8_round_trip(&reference, head_dim),
                ideal_store_bytes(reference.len()),
            );
            let affine = mlx_round_trip(&reference, head_dim, "affine", AFFINE_GROUP);
            let affine = score(&reference, &affine.values, affine.bytes);

            let by_median = fp8.median_rel / affine.median_rel;
            let by_energy = fp8.rel_frob / affine.rel_frob;
            println!(
                "d={head_dim:3} log-uniform {decades:.0} decades: fp8/affine \
                 median-relative={by_median:.3}x  relative-Frobenius={by_energy:.3}x"
            );

            assert!(
                by_median < 0.5,
                "head_dim={head_dim}, {decades} decades: E4M3 reached only {by_median:.3}x \
                 affine-g128 on median relative error. The control exists to prove the setup can \
                 see the format axis; if E4M3 cannot win here it cannot win anywhere, and the \
                 sweep is measuring the fixture rather than the format",
            );
            assert!(
                by_energy > 1.0,
                "head_dim={head_dim}, {decades} decades: E4M3 reached {by_energy:.3}x \
                 affine-g128 on relative Frobenius. The module's verdict rests on affine winning \
                 the energy-weighted metric even where relative precision is most favoured",
            );
        }
    }
}

// ── The measurement ──────────────────────────────────────────────────────────

/// 8-bit float against 8-bit affine at identical rate, over both head dims,
/// every fixture, every outlier density and magnitude, three seeds.
///
/// `ref-fp8` — the arm that isolates the encoding from MLX's clipping — carries
/// the declared criterion, and is held in **both** directions: it must not
/// reach the falsifier, and it must not stop tying either, because the recorded
/// answer for the format is a tie and a drift to a clean win for either side
/// has to be re-derived rather than absorbed. `mxfp8` carries the weaker duty of
/// not beating affine, plus a pinned margin over `ref-fp8`: that margin *is*
/// MLX's clipping defect, so if upstream changes its E8M0 rounding the margin
/// collapses and this assertion goes red rather than passing on a premise that
/// has gone.
///
/// Prints every cell so the verdict can be read off the transcript rather than
/// taken from the assertion.
#[test]
fn no_eight_bit_float_arm_is_more_faithful_than_affine_eight_bit_at_equal_rate() {
    let mut visited = 0usize;
    let mut closest_shipped = f64::INFINITY;
    let mut closest_shipped_cell = String::new();
    let mut closest_best_case = f64::INFINITY;
    let mut closest_best_case_cell = String::new();
    let mut narrowest_clipping_margin = f64::INFINITY;
    let mut narrowest_clipping_cell = String::new();

    for head_dim in HEAD_DIMS {
        for fixture in cells() {
            for seed in SEEDS {
                visited += 1;
                let data = fixture.draw(head_dim, seed);
                let reference = bf16_reference(&data, head_dim);

                let shipped = mlx_round_trip(&reference, head_dim, "mxfp8", MXFP8_GROUP);
                let affine = mlx_round_trip(&reference, head_dim, "affine", AFFINE_GROUP);
                assert_eq!(
                    shipped.dtype, affine.dtype,
                    "the two arms decoded to different dtypes ({:?} vs {:?}); one would carry a \
                     rounding the other does not",
                    shipped.dtype, affine.dtype,
                );

                let ideal = score(
                    &reference,
                    &ideal_mxfp8_round_trip(&reference, head_dim),
                    ideal_store_bytes(reference.len()),
                );
                let mxfp8 = score(&reference, &shipped.values, shipped.bytes);
                let affine = score(&reference, &affine.values, affine.bytes);

                let mxfp8_vs = mxfp8.rel_frob / affine.rel_frob;
                let ideal_vs = ideal.rel_frob / affine.rel_frob;
                let clipping_margin = mxfp8.rel_frob / ideal.rel_frob;

                let label = fixture.label();
                println!(
                    "d={head_dim:3} {label:18} groups-touched={:.2} seed={seed:#018x}  \
                     mxfp8 relF={:.5} max={:8.4}  ref-fp8 relF={:.5} max={:8.4}  \
                     affine-g128 relF={:.5} max={:6.4}  mxfp8/affine={mxfp8_vs:6.3}x  \
                     ref-fp8/affine={ideal_vs:6.3}x  mxfp8/ref-fp8={clipping_margin:5.2}x",
                    fixture.groups_touched(head_dim),
                    mxfp8.rel_frob,
                    mxfp8.max_abs,
                    ideal.rel_frob,
                    ideal.max_abs,
                    affine.rel_frob,
                    affine.max_abs,
                );

                let cell = format!("d={head_dim} {label} seed={seed:#x}");
                if mxfp8_vs < closest_shipped {
                    closest_shipped = mxfp8_vs;
                    closest_shipped_cell = cell.clone();
                }
                if ideal_vs < closest_best_case {
                    closest_best_case = ideal_vs;
                    closest_best_case_cell = cell.clone();
                }
                if clipping_margin < narrowest_clipping_margin {
                    narrowest_clipping_margin = clipping_margin;
                    narrowest_clipping_cell = cell;
                }
            }
        }
    }

    println!(
        "{visited} cells. closest: mxfp8 {closest_shipped_cell} at {closest_shipped:.3}x, \
         ref-fp8 {closest_best_case_cell} at {closest_best_case:.3}x affine-g128. \
         narrowest clipping margin: {narrowest_clipping_cell} at \
         {narrowest_clipping_margin:.2}x"
    );

    assert_eq!(
        visited, EXPECTED_CELLS,
        "the sweep visited {visited} cells against {EXPECTED_CELLS} declared; every assertion \
         below is satisfied by an accumulator that never moved, so an emptied axis must fail here",
    );
    assert!(
        closest_best_case > MATERIAL_GAIN,
        "a non-clipping best-case E4M3 encoder reached {closest_best_case:.3}x affine-g128 in \
         {closest_best_case_cell}, at or past the {MATERIAL_GAIN}x declared falsifier. The \
         encoding buys real fidelity at equal size after all",
    );
    assert!(
        closest_best_case < 1.0,
        "a non-clipping best-case E4M3 encoder never got ahead of affine-g128 anywhere; its \
         closest cell was {closest_best_case_cell} at {closest_best_case:.3}x. The recorded \
         answer for the format is a tie, not a loss, and it is held here in both directions so \
         that a clean win for either side has to be re-derived rather than absorbed",
    );
    assert!(
        closest_shipped >= 1.0,
        "MLX's mxfp8 beat affine-g128 at equal rate in {closest_shipped_cell} \
         ({closest_shipped:.3}x), contradicting the recorded verdict that fp8 buys no fidelity \
         over affine at the same bits/value",
    );
    assert!(
        narrowest_clipping_margin >= 1.5,
        "MLX's mxfp8 came within {narrowest_clipping_margin:.2}x of the non-clipping reference \
         in {narrowest_clipping_cell}. The shipped arm's margin over affine is largely its \
         clipping defect; if that has been fixed upstream, the shipped arm no longer stands in \
         for the format and the verdict must be re-derived from the reference arm alone",
    );
}

// ── What MLX's store and widener actually do ─────────────────────────────────

/// MLX's `mxfp8` store really is OCP E4M3 codes times an E8M0 power of two, and
/// its encoder never emits the NaN pattern.
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
    let data = Fixture::Outlier {
        channels: 4,
        ratio: 20.0,
    }
    .draw(head_dim, TEST_SEED);
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

    let code_bytes = codes.to_bytes().expect("codes to_bytes");
    let scale_bytes = scales.to_bytes().expect("scales to_bytes");
    assert_eq!(code_bytes.len(), data.len(), "one code byte per value");
    assert_eq!(
        mlx.len(),
        data.len(),
        "MLX decoded {} values from a {}-value store; a zip would have compared only the shorter \
         side and reported a whole-store result",
        mlx.len(),
        data.len(),
    );
    assert_eq!(
        scale_bytes.len(),
        data.len() / MXFP8_GROUP as usize,
        "one E8M0 scale byte per group",
    );
    assert!(
        !code_bytes.iter().any(|&c| c & 0x7F == 0x7F),
        "MLX's encoder emitted the E4M3 NaN pattern; the reference decoder treats it as NaN and \
         MLX's widener as a finite 480, and the two only stay compatible while no encoder \
         produces it",
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

/// MLX widens an E4M3 code by bitcasting its low 7 bits into a `half` and
/// multiplying by 256 — the exponent-bias correction, measured on MLX rather
/// than derived here.
///
/// Every one of the 256 codes is put through `mlx_dequantize` at a unit scale
/// and compared against `half`'s value of the shifted bit pattern, times 256,
/// with the sign bit reapplied. Both signs of every magnitude are covered, and
/// so is the NaN pattern, where MLX's finite 480 diverges from the format
/// definition (see [`e4m3_value`]). This is why an fp8 dequant contains a
/// floating-point multiply and is not a bitcast: E4M3's exponent bias is 7 and
/// `half`'s is 15.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test fixture setup in this fn; a failure here is a test bug"
)]
fn mlx_widens_e4m3_codes_by_bitcasting_into_half_and_multiplying_by_256() {
    let device = Device::Cpu;
    let codes: Vec<u8> = (0u8..=255).collect();
    let head_dim = codes.len() as i32;

    // MLX packs 8-bit codes four to a u32 word, so the same bytes reinterpreted
    // as U32 are the store it expects.
    let code_array = Array::from_bytes(&codes, &[1, head_dim / 4], Dtype::U32).expect("codes");
    // 0x7F is the E8M0 code for 2^0: a unit scale, so the decode is the code's
    // own value.
    let unit_scales = vec![127u8; codes.len() / MXFP8_GROUP as usize];
    let scale_array =
        Array::from_bytes(&unit_scales, &[1, unit_scales.len() as i32], Dtype::U8).expect("scales");

    let decoded = to_f32_vec(
        &dequantize(
            &code_array,
            &scale_array,
            None,
            MXFP8_GROUP,
            BITS,
            "mxfp8",
            device,
        )
        .expect("dequantize"),
        device,
    );
    assert_eq!(decoded.len(), codes.len(), "one decoded value per code");

    for (&code, &got) in codes.iter().zip(decoded.iter()) {
        let reinterpreted = half_value(u16::from(code & 0x7F) << 7);
        let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
        let want = sign * reinterpreted * 256.0;
        assert!(
            (want - got).abs() <= BF16_ULP * want.abs(),
            "code {code:#04x}: bitcast of the low 7 bits into half gives {reinterpreted}, \
             x256 and signed gives {want}, MLX returned {got}",
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
    let data = Fixture::Gaussian.draw(head_dim, TEST_SEED);
    let reference = bf16_reference(&data, head_dim);
    let mlx = mlx_round_trip(&reference, head_dim, "mxfp8", MXFP8_GROUP).values;
    let ideal = ideal_mxfp8_round_trip(&reference, head_dim);

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
