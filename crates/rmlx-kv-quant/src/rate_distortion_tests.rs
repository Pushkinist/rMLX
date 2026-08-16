//! Rate-distortion reference for the scalar-codebook KV codecs.
//!
//! # What the cosine gates cannot tell you
//!
//! A cosine floor answers "did this regress". It cannot answer "is this as good
//! as the bit width allows", because it has no reference. This module supplies
//! one: for a fixed-rate Lloyd-Max quantizer on the standard normal, achievable
//! SQNR is a known constant per bit width — see
//! [`crate::test_utils::LLOYD_MAX_GAUSSIAN_SQNR_DB`]. Measuring a codec against
//! it converts "cosine 0.994" into "0.8 bits ahead of / behind a matched scalar
//! quantizer at the same nominal rate", which is directly actionable.
//!
//! The fixture is i.i.d. standard normal on purpose: the anchor is defined for
//! that source, and comparing a codec on non-Gaussian data against a Gaussian
//! anchor would compare two different things. Outlier-channel behaviour is a
//! separate axis, measured in `rotation_fidelity_tests.rs`.
//!
//! # The anchor is not free
//!
//! The anchor assumes the quantizer knows sigma and spends no rate saying so.
//! Every codec here instead derives its scale from a per-group **maximum** and
//! stores it, so it buys conditioning the anchor does not have and can
//! legitimately land above it. That is why the report prints stored bits per
//! value beside the dB: a codec 5 dB ahead of the 3-bit anchor while spending
//! 11 bits per value has not beaten scalar quantization, it has changed the
//! rate. The rate column is **reported, not gated** — sizing the codecs is
//! tracked separately.
//!
//! The rate column is also **path-specific**: `run_iso` reports the CPU
//! `IsoBlocks` figure, which carries a per-group quaternion sideband the GPU
//! ring does not. See its doc comment before comparing iso against rotor.
//!
//! # What the measurement found
//!
//! The suspicion that motivated this reference was that mapping a *small*-group
//! maximum onto the outermost centroid presents the codebook with data whose
//! standard deviation is ≈ 1.47 rather than 1, and that this would cost several
//! dB. It does not. Small groups come out **ahead** of the anchor — the group
//! maximum is a strong conditioning statistic, it is reconstructed near-exactly
//! by construction, and the remaining 2–3 elements are then known to be smaller
//! than it. The shortfall appears at the other end: large groups at low bit
//! widths, where one f32 scale per 32 values buys little and the outermost
//! Lloyd-Max cell is very wide.

use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::planarquant::{planar_dequantize, planar_quantize};
use crate::rotorquant::{n_groups_for, rotor3_decode, rotor3_encode, rotor4_decode, rotor4_encode};
use crate::tcq::{tcq_quantize_v2, tcq_quantize_v3};
use crate::test_utils::{
    gaussian_data, lloyd_max_anchor_db, sqnr_db, wasted_bits, DB_PER_BIT, TEST_SEED,
};
use crate::turboquant::{
    lloyd_gaussian_codebook, turbo_dequantize, turbo_dequantize_with_codebook, turbo_quantize_v,
    turbo_quantize_v_with_codebook, GROUP_SIZE,
};

// ── Fixture ──────────────────────────────────────────────────────────────────

const RD_ROWS: usize = 256;
const RD_HEAD_DIM: usize = 128;
const RD_SHAPE: [i32; 4] = [1, 1, RD_ROWS as i32, RD_HEAD_DIM as i32];
const RD_VALUES: usize = RD_ROWS * RD_HEAD_DIM;

/// i.i.d. standard normal, pinned seed — the source the anchors are defined for.
fn gaussian_fixture() -> Vec<f32> {
    gaussian_data(RD_VALUES, TEST_SEED)
}

/// Bits per value implied by a heap footprint over [`RD_VALUES`] values.
fn bits_per_value(bytes: u64) -> f64 {
    (bytes * 8) as f64 / RD_VALUES as f64
}

// ── Codec cells ──────────────────────────────────────────────────────────────

/// A codec's encode+decode at a chosen bit width: decoded values plus the bits
/// its own encode output occupies per input value.
type CodecRun = fn(&[f32], u8) -> (Vec<f32>, f64);

/// One `(codec, bit width)` cell: decoded values plus the bits its own encode
/// output occupies per input value.
struct Cell {
    name: &'static str,
    bits: u8,
    run: fn(&[f32]) -> (Vec<f32>, f64),
    /// Pinned ceiling on this cell's shortfall against its anchor, in bits.
    /// Set at the measured value plus [`PIN_SLACK_BITS`].
    budget: f64,
}

fn run_turbo(data: &[f32], bits: u8) -> (Vec<f32>, f64) {
    let blocks = turbo_quantize_v(data, bits, &RD_SHAPE).expect("turbo_quantize_v");
    let rate = bits_per_value(blocks.byte_size());
    (turbo_dequantize(&blocks).expect("turbo_dequantize"), rate)
}

fn run_tcq(data: &[f32], bits: u8) -> (Vec<f32>, f64) {
    let blocks = if bits == 2 {
        tcq_quantize_v2(data, &RD_SHAPE).expect("tcq_quantize_v2")
    } else {
        tcq_quantize_v3(data, &RD_SHAPE).expect("tcq_quantize_v3")
    };
    let rate = bits_per_value(blocks.byte_size());
    (turbo_dequantize(&blocks).expect("turbo_dequantize"), rate)
}

fn run_planar(data: &[f32], bits: u8) -> (Vec<f32>, f64) {
    let blocks = planar_quantize(data, GROUP_SIZE, bits, &RD_SHAPE).expect("planar_quantize");
    let rate = bits_per_value(blocks.byte_size());
    (planar_dequantize(&blocks).expect("planar_dequantize"), rate)
}

/// iso, at the **CPU `IsoBlocks` rate**, not the ring-resident one.
///
/// The byte count includes the per-group quaternion array, so it reports
/// ≈48.25 bits/value at `head_dim = 128`. That is the honest figure for the
/// V-only `iso3` / `iso4` stores, which keep `IsoBlocks`. It is **not** the
/// figure for `k_iso3/4` and `iso3_sym/4_sym`: the GPU ring those decode from
/// does not carry the quaternion — it is the constant `FIXED_QUAT` replicated
/// per group, not data — so they sit at ≈16.25 bits/value. See
/// `docs/KV_QUANT.md` § iso3 "Memory truth". Quoting 48.25 against `rotor`'s
/// 21.75 without that distinction inverts the comparison: on the ring path iso
/// is the cheaper of the two.
///
/// The distortion is identical on both paths — the quaternion is a sideband,
/// not an input to reconstruction — so only the rate column is affected.
fn run_iso(data: &[f32], bits: u8) -> (Vec<f32>, f64) {
    let (codes, scales, quats, norms) =
        iso_encode_fast(data, RD_HEAD_DIM, 4, bits).expect("iso_encode_fast");
    let bytes = 4 * (codes.len() + scales.len() + quats.len() + norms.len()) as u64;
    let decoded = iso_decode_fast(&codes, &scales, &quats, &norms, RD_HEAD_DIM, 4, bits)
        .expect("iso_decode_fast");
    (decoded, bits_per_value(bytes))
}

fn run_rotor(data: &[f32], bits: u8) -> (Vec<f32>, f64) {
    let rotors = crate::clifford::make_rotor_table(0, 0, n_groups_for(RD_HEAD_DIM));
    let (codes, scales, norms) = if bits == 3 {
        rotor3_encode(data, &rotors, RD_HEAD_DIM).expect("rotor3_encode")
    } else {
        rotor4_encode(data, &rotors, RD_HEAD_DIM).expect("rotor4_encode")
    };
    let bytes = 4 * (codes.len() + scales.len() + norms.len()) as u64;
    let decoded = if bits == 3 {
        rotor3_decode(&codes, &scales, &norms, &rotors, RD_HEAD_DIM).expect("rotor3_decode")
    } else {
        rotor4_decode(&codes, &scales, &norms, &rotors, RD_HEAD_DIM).expect("rotor4_decode")
    };
    (decoded, bits_per_value(bytes))
}

/// Every scalar-codebook codec at every bit width a shipped `KvQuant` variant
/// can reach.
///
/// * `turbo` 2 / 3 / 4 — `K8VTurbo2`, `K8VTurbo3` + `TurboSym3`, `K8V4` +
///   `TurboSym4` + `RotKTq4V`.
/// * `tcq` 2 / 3 — `K8VTurbo2Tcq`, `K8VTurbo3Tcq`.
/// * `planar` 3 / 4 — `Planar3`, `Planar` + `PlanarK`.
/// * `iso` 3 / 4 — `Iso3` family, `Iso4` family.
/// * `rotor` 3 / 4 — `Rotor3` family, `Rotor4` family.
///
/// `bits = 1` is tabulated in the anchors but no storage variant reaches it, so
/// it is not measured here.
const CELLS: &[Cell] = &[
    Cell {
        name: "turbo",
        bits: 2,
        run: |d| run_turbo(d, 2),
        budget: 0.44,
    },
    Cell {
        name: "turbo",
        bits: 3,
        run: |d| run_turbo(d, 3),
        budget: 0.04,
    },
    Cell {
        name: "turbo",
        bits: 4,
        run: |d| run_turbo(d, 4),
        budget: -0.13,
    },
    Cell {
        name: "tcq",
        bits: 2,
        run: |d| run_tcq(d, 2),
        budget: 0.44,
    },
    Cell {
        name: "tcq",
        bits: 3,
        run: |d| run_tcq(d, 3),
        budget: 0.04,
    },
    Cell {
        name: "planar",
        bits: 3,
        run: |d| run_planar(d, 3),
        budget: -4.22,
    },
    Cell {
        name: "planar",
        bits: 4,
        run: |d| run_planar(d, 4),
        budget: -2.64,
    },
    Cell {
        name: "iso",
        bits: 3,
        run: |d| run_iso(d, 3),
        budget: -0.68,
    },
    Cell {
        name: "iso",
        bits: 4,
        run: |d| run_iso(d, 4),
        budget: -0.76,
    },
    Cell {
        name: "rotor",
        bits: 3,
        run: |d| run_rotor(d, 3),
        budget: -0.87,
    },
    Cell {
        name: "rotor",
        bits: 4,
        run: |d| run_rotor(d, 4),
        budget: -0.95,
    },
];

// ── The gate ─────────────────────────────────────────────────────────────────

/// Escalation line: a codec more than this many bits short of its anchor gets a
/// filed follow-up with the measured number, not a relaxed threshold.
///
/// **This is a message channel, not an independent gate.** Every budget in
/// [`CELLS`] is at most +0.44, so `wasted > 1.0` implies `wasted > cell.budget`
/// for every cell and the absolute check can never be the only one that fires;
/// it exists to label *why* a failure matters. The dependence runs one way only
/// — the budget catches things this does not, which
/// [`one_bit_short_codec_fails_the_rate_distortion_gate`] demonstrates at
/// `bits = 4`.
///
/// Justification, stated before the measurements. The anchor is a *matched*
/// fixed-rate Lloyd-Max quantizer on N(0,1): it knows sigma and pays no rate to
/// say so. Every codec here pays extra rate — at minimum one f32 scale per
/// group — so it starts with an advantage the anchor does not have and should
/// meet or beat it. A full bit below the anchor means the nominal bit width is
/// not delivering what the same number of bits delivers in the textbook
/// construction.
const MAX_WASTED_BITS: f64 = 1.0;

/// Slack on each cell's pinned budget, in bits.
///
/// The absolute line above cannot be the only gate. Codecs sit at very
/// different offsets from the anchor — `turbo4` is 0.23 bits *ahead* of it — so
/// a codec that silently loses a full bit can still land inside a 1-bit
/// absolute budget. [`one_bit_short_codec_fails_the_rate_distortion_gate`]
/// demonstrates exactly that at `bits = 4`. Each cell therefore also carries
/// its own pinned ceiling.
///
/// 0.10 bits = 0.60 dB. The fixture is seeded and every codec here is
/// deterministic, so there is no run-to-run variation for the slack to absorb;
/// it covers f32 and codegen drift only, and is an order of magnitude below the
/// ~1.1 bits the mutation guard moves.
const PIN_SLACK_BITS: f64 = 0.10;

/// Measure every cell, print the table, and fail listing **all** cells that
/// cross either their own pinned budget or the absolute escalation line.
///
/// Accumulating rather than asserting per cell keeps a single regression from
/// hiding the rest of the table.
#[test]
fn scalar_codebook_rate_distortion_report() {
    let data = gaussian_fixture();

    println!(
        "Rate-distortion vs fixed-rate Lloyd-Max N(0,1), \
         i.i.d. Gaussian fixture ({RD_ROWS} x {RD_HEAD_DIM}):\n\
         \x20 codec   bits   measured    anchor    wasted   budget   stored bits/value"
    );

    let mut over_budget: Vec<String> = Vec::new();
    for cell in CELLS {
        let (decoded, rate) = (cell.run)(&data);
        let measured = sqnr_db(&data, &decoded);
        let anchor = lloyd_max_anchor_db(cell.bits);
        let wasted = wasted_bits(measured, anchor);

        println!(
            "\x20 {:7} {:4}   {measured:8.3} dB {anchor:7.3} dB  {wasted:+7.3}  {:+7.2}   {rate:6.2}",
            cell.name, cell.bits, cell.budget,
        );

        if wasted > cell.budget {
            over_budget.push(format!(
                "{}{} regressed to {wasted:+.2} wasted bits, past its pinned {:+.2} \
                 ({measured:.2} dB vs anchor {anchor:.2} dB) at {rate:.2} stored bits/value. \
                 If the rate moved too this is a deliberate rate-for-distortion trade and the \
                 budget needs re-pinning, not the codec reverting",
                cell.name, cell.bits, cell.budget,
            ));
        }
        if wasted > MAX_WASTED_BITS {
            over_budget.push(format!(
                "{}{} is {wasted:.2} bits short of its anchor, past the {MAX_WASTED_BITS:.1}-bit \
                 escalation line — file a follow-up with this number",
                cell.name, cell.bits,
            ));
        }
    }

    assert!(
        over_budget.is_empty(),
        "rate-distortion gate: {}",
        over_budget.join("; "),
    );
}

/// Every pinned budget sits at its cell's measured shortfall plus exactly
/// [`PIN_SLACK_BITS`], and no cell has been quietly given more room than that.
///
/// A pinned floor is only meaningful if the pin is where it claims to be. This
/// is the test that would go red if someone widened a budget to make a
/// regression pass instead of investigating it.
#[test]
fn pinned_budgets_sit_one_slack_above_the_measurement() {
    let data = gaussian_fixture();
    for cell in CELLS {
        let (decoded, _) = (cell.run)(&data);
        let wasted = wasted_bits(sqnr_db(&data, &decoded), lloyd_max_anchor_db(cell.bits));
        let headroom = cell.budget - wasted;
        // Budgets are recorded to two decimal places, so allow half a unit in
        // the last place on top of the slack itself.
        assert!(
            headroom >= 0.0,
            "{}{} sits {:+.3} bits past its own budget — the report test is the one to read",
            cell.name,
            cell.bits,
            -headroom,
        );
        assert!(
            headroom <= PIN_SLACK_BITS + 0.005,
            "{}{} has {headroom:+.3} bits of headroom against its budget, more than the \
             {PIN_SLACK_BITS:.2} the pin convention allows. Either the budget was widened \
             instead of investigating a regression, or the codec **improved** by \
             {:.3} bits — if it improved, re-pin the budget to the new measurement",
            cell.name,
            cell.bits,
            headroom - PIN_SLACK_BITS,
        );
    }
}

// ── Mutation guard ───────────────────────────────────────────────────────────

/// A codebook with `2^bits` entries but only `2^(bits-1)` distinct
/// reconstruction levels — a codec that is exactly one bit short of its label.
///
/// Each coarse centroid is emitted twice, the second copy nudged by 1e-4 so the
/// encoder's strictly-ascending validation passes while the reconstruction is
/// unchanged to five decimal places. This is the degenerate the reference
/// exists to catch: nominal rate intact, delivered rate halved.
fn one_bit_short_codebook(bits: u8) -> Vec<f32> {
    let coarse = lloyd_gaussian_codebook(bits - 1).expect("coarse codebook");
    let mut out = Vec::with_capacity(1usize << bits);
    for &centroid in coarse {
        out.push(centroid);
        out.push(centroid + 1e-4);
    }
    out
}

/// Mutation guard: a codec one bit short of its label must fail the gate.
///
/// Without this the report is decoration — a threshold nothing in the tree can
/// cross proves only that the tree is unchanged.
///
/// Note which half of the gate catches it. At `bits = 3` the mutation crosses
/// both the pinned budget and the absolute escalation line. At `bits = 4` it
/// crosses **only** the pinned budget: honest `turbo4` sits 0.23 bits *ahead*
/// of its anchor, so losing a full bit still lands inside the 1-bit absolute
/// line. That is the concrete reason [`PIN_SLACK_BITS`] exists — an absolute
/// anchor budget alone would have let this through.
#[test]
fn one_bit_short_codec_fails_the_rate_distortion_gate() {
    let data = gaussian_fixture();

    for bits in [3u8, 4] {
        let codebook = one_bit_short_codebook(bits);
        assert_eq!(
            codebook.len(),
            1usize << bits,
            "codebook must be full width"
        );

        let blocks = turbo_quantize_v_with_codebook(&data, bits, &RD_SHAPE, Some(&codebook))
            .expect("turbo_quantize_v_with_codebook");
        let decoded = turbo_dequantize_with_codebook(&blocks, Some(&codebook))
            .expect("turbo_dequantize_with_codebook");

        let measured = sqnr_db(&data, &decoded);
        let anchor = lloyd_max_anchor_db(bits);
        let wasted = wasted_bits(measured, anchor);

        // The honest codec at the same nominal width, and the budget the report
        // would have judged it against.
        let (honest_decoded, _) = run_turbo(&data, bits);
        let honest = sqnr_db(&data, &honest_decoded);
        let budget = CELLS
            .iter()
            .find(|c| c.name == "turbo" && c.bits == bits)
            .map(|c| c.budget)
            .expect("turbo cell for this bit width");

        println!(
            "one-bit-short turbo{bits}: {measured:.2} dB ({wasted:+.2} bits) vs honest \
             turbo{bits} {honest:.2} dB ({:+.2} bits); budget {budget:+.2}, escalation line \
             {MAX_WASTED_BITS:+.2}",
            wasted_bits(honest, anchor),
        );

        assert!(
            wasted > budget,
            "a turbo{bits} with only {} distinct levels reported {wasted:+.2} wasted bits and \
             passed its pinned budget of {budget:+.2} — the gate cannot fail",
            1usize << (bits - 1),
        );
        assert!(
            honest - measured > 0.5 * DB_PER_BIT,
            "halving the level count cost only {:.2} dB at bits={bits}; the mutation is not \
             lossy enough to be a guard",
            honest - measured,
        );
    }
}

// ── Measured anomaly, pinned ─────────────────────────────────────────────────

/// The 3-bit and 4-bit widths of iso, rotor and planar cost **byte-identical**
/// storage, so in each family one width is strictly dominated.
///
/// All three pack codes into u32 words under the shared `32 / bits`
/// vals-per-word convention, and at every shipped group size the word count
/// comes out the same for `bits = 3` and `bits = 4`: iso and rotor put one
/// group in one word either way, and planar's 32-value block takes
/// `ceil(32/10) = ceil(32/8) = 4` words. The 3-bit variants therefore burn the
/// spare bits rather than saving anything, and the comparison between the two
/// widths is pure quality at fixed rate.
///
/// * `iso` and `rotor` — the 4-bit width wins, so the 3-bit width is dominated.
/// * `planar` — the **3-bit** width wins, by ~3.9 dB, so `planar4` is
///   dominated. Per pair the larger element is pinned to the outermost
///   centroid, leaving the smaller one on the grid `centroid / max_centroid`,
///   whose outermost gap is `(2.152 - 1.344)/2.152 = 0.375` at 3 bits and
///   `(2.718 - 2.052)/2.718 = 0.245` at 4 bits — only 1.5x finer, while the
///   16-angle Givens search that has to land *both* elements on centroids gets
///   no larger. The extra bit does not pay for itself.
///
/// The test pins what is measured rather than the ordering everyone expects, so
/// the day a family's packing or codebook is fixed this goes red and gets
/// re-pointed at the new contract instead of quietly agreeing with whatever the
/// code does.
#[test]
fn byte_identical_bit_widths_leave_one_width_dominated() {
    let data = gaussian_fixture();

    // (family, encoder, which width wins)
    let families: [(&str, CodecRun, u8); 3] = [
        ("iso", run_iso, 4),
        ("rotor", run_rotor, 4),
        ("planar", run_planar, 3),
    ];
    for (family, run, winner) in families {
        let (three, three_rate) = run(&data, 3);
        let (four, four_rate) = run(&data, 4);
        let three_db = sqnr_db(&data, &three);
        let four_db = sqnr_db(&data, &four);

        println!(
            "{family}: 3-bit {three_db:.2} dB @ {three_rate:.2} bits/value | \
             4-bit {four_db:.2} dB @ {four_rate:.2} bits/value | \
             winner {winner}-bit by {:.2} dB",
            (three_db - four_db).abs(),
        );

        assert!(
            (three_rate - four_rate).abs() < 1e-9,
            "{family} 3-bit and 4-bit are supposed to occupy byte-identical storage: \
             {three_rate:.2} vs {four_rate:.2} bits per value. If a packing change made them \
             differ, this comparison is no longer at fixed rate — re-point the test"
        );

        let (winner_db, loser_db, loser) = if winner == 3 {
            (three_db, four_db, 4)
        } else {
            (four_db, three_db, 3)
        };
        assert!(
            winner_db > loser_db,
            "{family}{loser} {loser_db:.2} dB now beats {family}{winner} {winner_db:.2} dB at \
             the same rate. The documented dominance has flipped — update docs/KV_QUANT.md \
             and this test rather than deleting it"
        );
    }
}

/// The Viterbi trellis buys **0.000 dB** over plain nearest-centroid, at both
/// shipped widths, on structured and unstructured data alike.
///
/// The trellis degeneracy itself is already documented — the per-step cost
/// `dist(value, codebook[level])` does not depend on the state, and
/// `build_transition_table` makes every level reachable from every state, so
/// the admissible level sequences are the full product set and the
/// minimum-cost path is the per-step greedy minimum. What was missing is the
/// number: TCQ is the one codec here whose stated purpose is to claw back part
/// of the gap to the rate-distortion bound, and the claw-back measures as
/// exactly zero.
///
/// On i.i.d. data the codes come out byte-identical. On a structured sweep the
/// codes *differ* — the forward pass breaks ties differently — but the
/// distortion does not move, which is the stronger statement and the one the
/// existing non-strict `tcq >= turbo` cosine gate cannot make.
///
/// Pinned as an equality so that giving the trellis a real constraint turns
/// this red and gets it re-pointed at the new contract.
#[test]
fn trellis_coded_quantization_claws_back_nothing() {
    let gaussian = gaussian_fixture();
    // A sweep along the dim axis: strongly inter-element-correlated, the
    // structure a trellis is supposed to exploit.
    let sweep: Vec<f32> = (0..RD_VALUES)
        .map(|i| (((i % RD_HEAD_DIM) as f32) * 0.05).sin() * 2.0)
        .collect();

    for bits in [2u8, 3] {
        for (label, data) in [("i.i.d. gaussian", &gaussian), ("dim-axis sweep", &sweep)] {
            let trellis = if bits == 2 {
                tcq_quantize_v2(data, &RD_SHAPE).expect("tcq_quantize_v2")
            } else {
                tcq_quantize_v3(data, &RD_SHAPE).expect("tcq_quantize_v3")
            };
            let greedy = turbo_quantize_v(data, bits, &RD_SHAPE).expect("turbo_quantize_v");

            let trellis_db = sqnr_db(data, &turbo_dequantize(&trellis).expect("dequant"));
            let greedy_db = sqnr_db(data, &turbo_dequantize(&greedy).expect("dequant"));
            let gain = trellis_db - greedy_db;

            println!(
                "tcq{bits} on {label}: {trellis_db:.4} dB vs turbo{bits} {greedy_db:.4} dB \
                 = {gain:+.4} dB ({} bits); codes identical: {}",
                gain / DB_PER_BIT,
                trellis.codes == greedy.codes,
            );

            assert!(
                gain.abs() < 1e-3,
                "the trellis moved distortion by {gain:+.4} dB at bits={bits} on {label}. \
                 It is documented as degenerate and measured at 0.000 dB — if it now does \
                 something, update docs/KV_QUANT.md and re-point this test"
            );
        }
    }
}
