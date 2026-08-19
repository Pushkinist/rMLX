//! Stored-rate ceiling for every shipped KV codec.
//!
//! # What this gate is for
//!
//! A KV codec advertises a codebook width. What matters for memory is what
//! reaches the store: codes **plus** every per-group and per-token sideband.
//! The two can differ by a lot — the rotor family quantizes to 3 bits and
//! stores 21.75 bits per value, above the bf16 baseline it is supposed to
//! compress — and nothing in the tree observed that until it was found by hand.
//!
//! So each codec family measures its rate here from the **actual bytes its own
//! encoder produced** over a shared fixture, via
//! [`crate::test_utils::stored_bits_per_value`], and is required to land at or
//! below [`crate::test_utils::BF16_BITS_PER_VALUE`] unless it carries an
//! explicit written exemption.
//!
//! # The exemption is not a mute button
//!
//! An exempt family must *actually* measure above the floor
//! ([`exempt_families_actually_exceed_the_floor`]). A family that gets fixed and
//! drops under 16 turns its own exemption red and has to have it removed. That
//! is the difference between an exemption list and a suppression list.
//!
//! # Completeness
//!
//! [`every_kv_quant_variant_names_its_store_families`] matches exhaustively on
//! [`KvQuant`], so a new variant does not compile until someone states which
//! families its K and V axes store in. That forces the **declaration** only.
//! Measurement runs over a hand-maintained representative list, and nothing
//! makes that list grow: a variant that declares a family and never gets a
//! representative is unmeasured and this gate stays green. See that test's own
//! doc — closing it mechanically needs enum iteration, which is a dependency
//! decision, so until then it is review's job.

use crate::isoquant::iso_encode_fast;
use crate::planarquant::planar_quantize;
use crate::q8::{q8_quantize, Q8_GROUP_SIZE};
use crate::quant::KvQuant;
use crate::rotorquant::{n_groups_for, rotor3_encode, rotor4_encode};
use crate::tcq::{tcq_quantize_v2, tcq_quantize_v3};
use crate::test_utils::{gaussian_data, stored_bits_per_value, BF16_BITS_PER_VALUE, TEST_SEED};
use crate::turboquant::{turbo_quantize_v, GROUP_SIZE};

// ── Fixture ──────────────────────────────────────────────────────────────────

const ROWS: usize = 64;
const HEAD_DIM: usize = 128;
const SHAPE: [i32; 4] = [1, 1, ROWS as i32, HEAD_DIM as i32];
const VALUES: usize = ROWS * HEAD_DIM;

/// `head_dim = 128` on purpose: it is the shipped test-target head dimension and
/// the one every published rate figure quotes. The rate of a group-bound store
/// depends on `head_dim` (a group that does not divide it costs a pad group), so
/// a rate quoted without one is not a number.
fn fixture() -> Vec<f32> {
    gaussian_data(VALUES, TEST_SEED)
}

// ── Family table ─────────────────────────────────────────────────────────────

/// How a family's stored rate is established.
enum Rate {
    /// Summed heap bytes the family's own CPU encoder produced for the fixture.
    Measured(fn(&[f32]) -> u64),
    /// bf16 — two bytes per value by definition; there is no encoder to run.
    Bf16,
    /// MLX affine (`scale` + `bias` f32 per group). This crate has no CPU
    /// encoder for it, so the rate is the layout itself: `bits` code bits per
    /// value plus 64 sideband bits per group. Evaluated at the widest cadence
    /// any shipped `KvQuant` reaches, so it is an upper bound over the whole
    /// affine grid rather than one point of it.
    AffineLayout { bits: u32, group: u32 },
}

/// A codec family's verdict against the bf16 floor.
enum Verdict {
    /// Stores at or below 16 bits per value — a real compression format.
    UnderBf16,
    /// Stores above 16 bits per value. The text is the documented reason, and
    /// it is only allowed to stand while the family measures above the floor.
    Exempt(&'static str),
}

struct Family {
    name: &'static str,
    rate: Rate,
    verdict: Verdict,
}

fn measure_q8(data: &[f32]) -> u64 {
    let (codes, scales) = q8_quantize(data);
    (codes.len() + 4 * scales.len()) as u64
}

fn measure_turbo(data: &[f32], bits: u8) -> u64 {
    turbo_quantize_v(data, bits, &SHAPE)
        .expect("turbo_quantize_v")
        .byte_size()
}

fn measure_tcq(data: &[f32], bits: u8) -> u64 {
    if bits == 2 {
        tcq_quantize_v2(data, &SHAPE).expect("tcq_quantize_v2")
    } else {
        tcq_quantize_v3(data, &SHAPE).expect("tcq_quantize_v3")
    }
    .byte_size()
}

fn measure_planar(data: &[f32], bits: u8) -> u64 {
    planar_quantize(data, GROUP_SIZE, bits, &SHAPE)
        .expect("planar_quantize")
        .byte_size()
}

fn measure_iso(data: &[f32], bits: u8) -> u64 {
    let (codes, scales, quats, norms) =
        iso_encode_fast(data, HEAD_DIM, 4, bits).expect("iso_encode_fast");
    4 * (codes.len() + scales.len() + quats.len() + norms.len()) as u64
}

fn measure_rotor(data: &[f32], bits: u8) -> u64 {
    let rotors = crate::clifford::make_rotor_table(0, 0, n_groups_for(HEAD_DIM));
    let (codes, scales, norms) = if bits == 3 {
        rotor3_encode(data, &rotors, HEAD_DIM).expect("rotor3_encode")
    } else {
        rotor4_encode(data, &rotors, HEAD_DIM).expect("rotor4_encode")
    };
    4 * (codes.len() + scales.len() + norms.len()) as u64
}

/// Every store family a shipped `KvQuant` variant can put an axis into.
///
/// The two exemptions are the families the accounting was written to surface.
/// Both spend one whole `u32` code word **and** one `f32` scale per group — 4
/// head-dim slots for iso, 3 for rotor — so the nominal 3-bit and 4-bit members
/// occupy byte-identical storage and neither is a compression format at any
/// `head_dim`.
const FAMILIES: &[Family] = &[
    Family {
        name: "bf16",
        rate: Rate::Bf16,
        verdict: Verdict::UnderBf16,
    },
    Family {
        name: "q8",
        rate: Rate::Measured(measure_q8),
        verdict: Verdict::UnderBf16,
    },
    Family {
        // The widest cadence any **enumerable** affine cache type reaches:
        // `CacheType::Q8G32` (8-bit, group 32) at 10.0 bits per value. This is
        // NOT a bound over every parseable affine config — the `mixed_*`
        // grammar takes an unvalidated group size, so rates above the floor are
        // expressible and this gate cannot see them. See
        // `mixed_grammar_admits_affine_rates_above_the_floor`.
        name: "affine",
        rate: Rate::AffineLayout { bits: 8, group: 32 },
        verdict: Verdict::UnderBf16,
    },
    Family {
        name: "turbo2",
        rate: Rate::Measured(|d| measure_turbo(d, 2)),
        verdict: Verdict::UnderBf16,
    },
    Family {
        name: "turbo3",
        rate: Rate::Measured(|d| measure_turbo(d, 3)),
        verdict: Verdict::UnderBf16,
    },
    Family {
        name: "turbo4",
        rate: Rate::Measured(|d| measure_turbo(d, 4)),
        verdict: Verdict::UnderBf16,
    },
    Family {
        name: "tcq2",
        rate: Rate::Measured(|d| measure_tcq(d, 2)),
        verdict: Verdict::UnderBf16,
    },
    Family {
        name: "tcq3",
        rate: Rate::Measured(|d| measure_tcq(d, 3)),
        verdict: Verdict::UnderBf16,
    },
    Family {
        name: "planar3",
        rate: Rate::Measured(|d| measure_planar(d, 3)),
        verdict: Verdict::Exempt(
            "measured overrun, not a design intent: the scale is per **pair**, one f32 per \
             2 elements, which is 16 bits per value before a single code bit. With 4 bits \
             of codes and 2 of rotation index the store is 22 bits per value at every \
             head_dim and both bit widths. Recorded here rather than fixed — a scale \
             cadence change is a format change",
        ),
    },
    Family {
        name: "planar4",
        rate: Rate::Measured(|d| measure_planar(d, 4)),
        verdict: Verdict::Exempt("same layout as planar3 — byte-identical at every head_dim"),
    },
    Family {
        name: "iso3",
        rate: Rate::Measured(|d| measure_iso(d, 3)),
        verdict: Verdict::Exempt(
            "fidelity experiment, not a compression format: one u32 code word + one f32 \
             scale + one f32 quaternion per 4-element group, plus one f32 norm per token",
        ),
    },
    Family {
        name: "iso4",
        rate: Rate::Measured(|d| measure_iso(d, 4)),
        verdict: Verdict::Exempt("same layout as iso3 — byte-identical at every head_dim"),
    },
    Family {
        name: "rotor3",
        rate: Rate::Measured(|d| measure_rotor(d, 3)),
        verdict: Verdict::Exempt(
            "fidelity experiment, not a compression format: one u32 code word + one f32 \
             scale per 3-element group, plus one f32 norm per token. Five of the eight \
             codes in each word are structurally zero (see crate::clifford)",
        ),
    },
    Family {
        name: "rotor4",
        rate: Rate::Measured(|d| measure_rotor(d, 4)),
        verdict: Verdict::Exempt("same layout as rotor3 — byte-identical at every head_dim"),
    },
];

fn family(name: &str) -> &'static Family {
    FAMILIES
        .iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("no rate entry for store family '{name}'"))
}

fn family_rate(f: &Family, data: &[f32]) -> f64 {
    match f.rate {
        Rate::Measured(m) => stored_bits_per_value(m(data), VALUES),
        Rate::Bf16 => stored_bits_per_value(2 * VALUES as u64, VALUES),
        Rate::AffineLayout { bits, group } => f64::from(bits) + 64.0 / f64::from(group),
    }
}

// ── The gate ─────────────────────────────────────────────────────────────────

/// No codec family stores above the bf16 floor without a written exemption.
#[test]
fn every_store_family_is_at_or_below_the_bf16_floor_or_exempt() {
    let data = fixture();
    println!("Stored rate per KV store family (head_dim = {HEAD_DIM}):");

    let mut over: Vec<String> = Vec::new();
    for f in FAMILIES {
        let rate = family_rate(f, &data);
        let tag = match f.verdict {
            Verdict::UnderBf16 => "",
            Verdict::Exempt(_) => "  [exempt]",
        };
        println!("\x20 {:8} {rate:7.2} bits/value{tag}", f.name);

        if matches!(f.verdict, Verdict::UnderBf16) && rate > BF16_BITS_PER_VALUE {
            over.push(format!(
                "{} stores {rate:.2} bits per value, above bf16's {BF16_BITS_PER_VALUE:.1}. \
                 Either the store layout regressed, or this family belongs in the exempt \
                 list with a written reason — not silently over the floor",
                f.name,
            ));
        }
    }
    assert!(over.is_empty(), "stored-rate gate: {}", over.join("; "));
}

/// An exemption must describe a real overrun, and say why.
///
/// Without this the exempt list is a suppression list: a family could be fixed,
/// or mis-listed in the first place, and nothing would notice. Turning the
/// exemption red when the family drops under the floor forces the list to be
/// deleted from rather than only added to.
#[test]
fn exempt_families_actually_exceed_the_floor() {
    let data = fixture();
    for f in FAMILIES {
        let Verdict::Exempt(reason) = f.verdict else {
            continue;
        };
        assert!(
            !reason.trim().is_empty(),
            "{} carries an empty exemption reason",
            f.name
        );
        let rate = family_rate(f, &data);
        assert!(
            rate > BF16_BITS_PER_VALUE,
            "{} is exempt from the bf16 floor but measures {rate:.2} bits per value, at or \
             below it. If the store was fixed, delete the exemption; if the exemption was \
             wrong, it was hiding nothing and should go either way",
            f.name,
        );
    }
}

/// Mutation guard: the gate can fail.
///
/// A ceiling nothing in the tree crosses proves only that the tree is unchanged.
/// This runs the gate's own predicate against a store inflated past the floor
/// and requires a rejection — and against the honest rotor bytes to show the
/// number the predicate sees is the one the store actually holds.
#[test]
fn the_floor_rejects_a_store_that_grew_past_bf16() {
    let data = fixture();

    // A family that claims to be under the floor while storing one f32 per
    // value — the shape of the defect (a sideband nobody counted).
    let inflated = stored_bits_per_value(4 * VALUES as u64, VALUES);
    assert!(
        inflated > BF16_BITS_PER_VALUE,
        "an f32-per-value store must be rejected by the floor, got {inflated:.2}"
    );

    // And the predicate reads real encoder bytes, not a constant: turbo4 passes
    // it and rotor3 does not, on the same fixture.
    let turbo = family_rate(family("turbo4"), &data);
    let rotor = family_rate(family("rotor3"), &data);
    assert!(
        turbo <= BF16_BITS_PER_VALUE,
        "turbo4 measured {turbo:.2} bits per value — above the floor, so the gate's pass \
         side is not exercised by anything"
    );
    assert!(
        rotor > BF16_BITS_PER_VALUE,
        "rotor3 measured {rotor:.2} bits per value — at or below the floor, so the gate's \
         fail side is not exercised by anything"
    );
}

/// The rotor rate splits into code bits, scale bits and norm bits, and the split
/// is what the codec documentation quotes.
///
/// Pinned separately from the total because the two halves have different fixes:
/// the code half is addressed by storing only the three grade-1 components, the
/// scale half by a coarser scale cadence. A total alone hides which one moved.
#[test]
fn rotor_rate_splits_into_documented_code_scale_and_norm_bits() {
    let rotors = crate::clifford::make_rotor_table(0, 0, n_groups_for(HEAD_DIM));
    let data = fixture();
    let (codes, scales, norms) = rotor3_encode(&data, &rotors, HEAD_DIM).expect("rotor3_encode");

    let code_bits = stored_bits_per_value(4 * codes.len() as u64, VALUES);
    let scale_bits = stored_bits_per_value(4 * scales.len() as u64, VALUES);
    let norm_bits = stored_bits_per_value(4 * norms.len() as u64, VALUES);
    let total = code_bits + scale_bits + norm_bits;
    println!(
        "rotor3 @ head_dim={HEAD_DIM}: codes {code_bits:.2} + scales {scale_bits:.2} + \
         norms {norm_bits:.2} = {total:.2} bits/value"
    );

    // 43 groups per row of 128 values: 43 * 32 / 128 = 10.75 on each of codes
    // and scales, and one f32 norm per row is 32 / 128 = 0.25.
    assert!((code_bits - 10.75).abs() < 1e-9, "code rate {code_bits}");
    assert!((scale_bits - 10.75).abs() < 1e-9, "scale rate {scale_bits}");
    assert!((norm_bits - 0.25).abs() < 1e-9, "norm rate {norm_bits}");
    assert!((total - 21.75).abs() < 1e-9, "total rate {total}");

    // The scale sideband alone costs two thirds of a bf16 value. Stated as an
    // assertion so a scale-cadence change has to come back through here.
    assert!(
        scale_bits > 0.5 * BF16_BITS_PER_VALUE,
        "one f32 scale per 3 values is supposed to be the dominant term ({scale_bits:.2})"
    );
}

/// Every `KvQuant` variant declares which store family each of its axes uses.
///
/// The `axes` match is exhaustive, so a new variant does not **compile** until
/// someone writes down where its bytes go. That is the whole of the mechanical
/// guarantee, and it is a guarantee about the *declaration*, not the
/// measurement.
///
/// **What is not guaranteed.** The `variants` list below is hand-maintained and
/// nothing forces it to grow: a new variant that gets its `axes` arm but no
/// representative is declared and never measured, and this test still passes.
/// The same hole runs the other way — **deleting** a representative while its
/// `axes` arm stays leaves that variant unmeasured just as quietly.
/// Two earlier shapes here claimed otherwise and neither held — a
/// `KV_QUANT_VARIANT_COUNT` compared against `variants.len()` (both sides
/// hand-kept), then an arm-index lookup whose out-of-bounds panic was supposed
/// to force a table to grow but is only ever reached *through* `variants`, so it
/// never fires for the missing case. Closing this mechanically needs enum
/// iteration (a `strum`-style derive), which is a dependency decision. Until
/// then it is review's job, and saying so plainly is worth more than a third
/// mechanism that does not work.
///
/// `Mixed` / `RotK` / the asym variants carry runtime bit and group fields and
/// map to the `affine` family — whose entry bounds the *enumerable* cache types
/// only, not every parseable config (see
/// [`mixed_grammar_admits_affine_rates_above_the_floor`]).
#[test]
fn every_kv_quant_variant_names_its_store_families() {
    let data = fixture();

    #[allow(
        clippy::match_same_arms,
        reason = "one arm per variant even when two share a family — merging them hides which \
                  codecs were considered, and this match exists to be read variant by variant"
    )]
    let axes = |q: &KvQuant| -> [&'static str; 2] {
        match q {
            KvQuant::None => ["bf16", "bf16"],
            KvQuant::K8V8 => ["q8", "q8"],
            KvQuant::K8V4 => ["q8", "turbo4"],
            KvQuant::RotKTq4V => ["q8", "turbo4"],
            KvQuant::Planar => ["q8", "planar4"],
            KvQuant::Planar3 => ["q8", "planar3"],
            KvQuant::PlanarK => ["planar4", "bf16"],
            KvQuant::Mixed { .. } => ["affine", "affine"],
            KvQuant::RotK { .. } => ["affine", "affine"],
            KvQuant::K8VTurbo3 => ["q8", "turbo3"],
            KvQuant::K8VTurbo3Tcq => ["q8", "tcq3"],
            KvQuant::K8VTurbo2 => ["q8", "turbo2"],
            KvQuant::K8VTurbo2Tcq => ["q8", "tcq2"],
            KvQuant::TurboSym3 => ["turbo3", "turbo3"],
            KvQuant::TurboSym4 => ["turbo4", "turbo4"],
            KvQuant::Iso3 => ["q8", "iso3"],
            KvQuant::Iso4 => ["q8", "iso4"],
            KvQuant::Iso3Sym => ["iso3", "iso3"],
            KvQuant::Iso4Sym => ["iso4", "iso4"],
            KvQuant::IsoKOnly3 => ["iso3", "bf16"],
            KvQuant::IsoKOnly4 => ["iso4", "bf16"],
            KvQuant::Rotor3 => ["q8", "rotor3"],
            KvQuant::Rotor4 => ["q8", "rotor4"],
            KvQuant::Rotor3Sym => ["rotor3", "rotor3"],
            KvQuant::Rotor4Sym => ["rotor4", "rotor4"],
            KvQuant::RotorKOnly3 => ["rotor3", "bf16"],
            KvQuant::RotorKOnly4 => ["rotor4", "bf16"],
            KvQuant::RotorK3Asym { .. } => ["rotor3", "affine"],
            KvQuant::RotorK4Asym { .. } => ["rotor4", "affine"],
        }
    };

    // One representative per variant shape; the field values do not change which
    // family an axis stores in.
    let variants = [
        KvQuant::None,
        KvQuant::K8V8,
        KvQuant::K8V4,
        KvQuant::RotKTq4V,
        KvQuant::Planar,
        KvQuant::Planar3,
        KvQuant::PlanarK,
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 128,
            v_group_size: 64,
        },
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        },
        KvQuant::K8VTurbo3,
        KvQuant::K8VTurbo3Tcq,
        KvQuant::K8VTurbo2,
        KvQuant::K8VTurbo2Tcq,
        KvQuant::TurboSym3,
        KvQuant::TurboSym4,
        KvQuant::Iso3,
        KvQuant::Iso4,
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
        KvQuant::Rotor3,
        KvQuant::Rotor4,
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
        KvQuant::RotorK3Asym {
            v_bits: 8,
            v_group_size: 128,
        },
        KvQuant::RotorK4Asym {
            v_bits: 8,
            v_group_size: 128,
        },
    ];

    for q in &variants {
        for name in axes(q) {
            let f = family(name);
            let rate = family_rate(f, &data);
            assert!(
                rate.is_finite(),
                "{q}: family '{name}' has no finite stored rate"
            );
        }
    }

    // Every table entry is reachable from some variant — an entry nothing maps
    // to is a rate nobody pays, and quietly stops being a gate.
    for f in FAMILIES {
        assert!(
            variants.iter().any(|q| axes(q).contains(&f.name)),
            "no KvQuant variant stores in family '{}' — delete the entry or map it",
            f.name
        );
    }
}

/// The `mixed_*` grammar no longer admits affine rates above the bf16 floor.
///
/// This test used to pin the opposite, and said so: `parse_kv_side` read the
/// group size as a bare `u16` with no whitelist, so `mixed_k8g4_v8g4` parsed
/// and stored `8 + 64/4 = 24` bits per value on each axis — a rate no
/// enum-driven table can see, because it is a property of a runtime field with
/// an unbounded domain rather than of the variant. Its own failure message
/// named the disposition: *"if this now fails, the parser grew a floor and this
/// test should assert that instead"*. It did, so this does.
///
/// The parser now validates both sides against MLX's affine grid
/// (`validate_mixed_side`), which bounds the domain to widths the codec can
/// actually store. The ceiling gate's coverage claim therefore extends over the
/// whole `mixed_*` grammar: every parseable spelling has a rate the table
/// enumerates.
#[test]
fn mixed_grammar_no_longer_admits_unbounded_affine_rates() {
    // The former witness: parses no more.
    let unbounded = "mixed_k8g4_v8g4".parse::<KvQuant>();
    assert!(
        unbounded.is_err(),
        "a group size outside the affine grid must be a parse error, got {unbounded:?}"
    );

    // The accepted grid is finite, so its worst rate is a number this test can
    // state: 8 bits at group 32 = 8 + 64/32 = 10.00 bits per value per axis.
    // Before the floor, the grid had no worst case at all.
    let mut worst = 0.0_f64;
    for group in [32u16, 64, 128] {
        for bits in [2u8, 3, 4, 5, 6, 8] {
            let spec = format!("mixed_k{bits}g{group}_v{bits}g{group}");
            let parsed = spec.parse::<KvQuant>();
            assert!(parsed.is_ok(), "{spec} must parse, got {parsed:?}");
            worst = worst.max(f64::from(bits) + 64.0 / f64::from(group));
        }
    }
    assert!(
        (worst - 10.0).abs() < 1e-9,
        "worst parseable mixed rate is {worst:.2} bits per value, expected 10.00 — \
         the accepted grid moved, so the ceiling gate's coverage claim moved with it"
    );
    assert!(
        worst < BF16_BITS_PER_VALUE,
        "the whole point of the floor is that no parseable mixed spec stores above bf16 \
         ({BF16_BITS_PER_VALUE:.2}); {worst:.2} does"
    );
}

#[test]
fn q8_family_rate_matches_its_shipped_group_size() {
    let data = fixture();
    let measured = stored_bits_per_value(measure_q8(&data), VALUES);
    let expected = 8.0 + 32.0 / Q8_GROUP_SIZE as f64;
    assert!(
        (measured - expected).abs() < 1e-9,
        "q8 measured {measured:.4} bits per value, layout says {expected:.4} — one f32 \
         scale per {Q8_GROUP_SIZE} codes"
    );
}
