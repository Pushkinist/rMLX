//! Display↔FromStr round-trip tests for all `KvQuant` variants.
//!
//! Every value that `Display` emits must parse back to the identical variant
//! via `FromStr`. This file is the guard that catches any future asymmetry.
//!
//! Deliberately-excluded variants (intentionally Display-only / not a
//! standalone CLI preset):
//! — none currently; all variants are expected to round-trip.

use std::str::FromStr;

use super::KvQuant;

/// Construct one representative instance of every `KvQuant` variant and assert
/// `KvQuant::from_str(&q.to_string()) == Ok(q)`.
///
/// Parametric variants (`Mixed`, `RotK`, `RotorK3Asym`, `RotorK4Asym`) are
/// exercised with multiple sample parameter sets to guard against partial
/// parse regressions (e.g. "parses 4-bit but not 8-bit").
#[test]
fn all_variants_display_fromstr_roundtrip() {
    let cases: &[KvQuant] = &[
        // ── simple unit variants ─────────────────────────────────────────────
        KvQuant::None,
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::Planar,
        KvQuant::Planar3,
        KvQuant::PlanarK,
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
        KvQuant::RotKTq4V,
        // ── Mixed — multiple param sets ──────────────────────────────────────
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 8,
            k_group_size: 128,
            v_group_size: 128,
        },
        KvQuant::Mixed {
            k_bits: 4,
            v_bits: 2,
            k_group_size: 32,
            v_group_size: 32,
        },
        // ── RotK — multiple param sets ───────────────────────────────────────
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        },
        KvQuant::RotK {
            v_bits: 8,
            v_group_size: 128,
        },
        KvQuant::RotK {
            v_bits: 2,
            v_group_size: 32,
        },
        // ── RotorK3Asym — multiple valid V codecs ────────────────────────────
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        KvQuant::RotorK3Asym {
            v_bits: 3,
            v_group_size: 64,
        },
        KvQuant::RotorK3Asym {
            v_bits: 2,
            v_group_size: 64,
        },
        // ── RotorK4Asym — multiple valid V codecs ────────────────────────────
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 32,
        },
        KvQuant::RotorK4Asym {
            v_bits: 3,
            v_group_size: 64,
        },
        KvQuant::RotorK4Asym {
            v_bits: 2,
            v_group_size: 64,
        },
    ];

    for &q in cases {
        let displayed = q.to_string();
        let parsed = KvQuant::from_str(&displayed).unwrap_or_else(|e| {
            panic!("{q:?} Display='{displayed}' failed to parse back: {e}");
        });
        assert_eq!(
            parsed, q,
            "{q:?} Display='{displayed}' parsed to {parsed:?}, not the original"
        );
    }
}

/// Spot-check a few known aliases that should parse to their canonical variant
/// but may Display differently (one-way only, not a round-trip failure).
///
/// These are inputs that are valid CLI shortcuts but not the canonical Display
/// form — they do not violate the round-trip invariant.
#[test]
fn aliases_parse_correctly() {
    // "bf16" and "f16" are accepted synonyms for None; Display emits "none".
    assert_eq!(KvQuant::from_str("bf16").unwrap(), KvQuant::None);
    assert_eq!(KvQuant::from_str("f16").unwrap(), KvQuant::None);
    // "rotor_v_3" / "rotor_v_4" are alternate names for Rotor3 / Rotor4.
    assert_eq!(KvQuant::from_str("rotor_v_3").unwrap(), KvQuant::Rotor3);
    assert_eq!(KvQuant::from_str("rotor_v_4").unwrap(), KvQuant::Rotor4);
}

/// Confirm that the previously-broken RotK `Display`/`FromStr` form now
/// parses correctly end-to-end.
#[test]
fn rotk_cli_form_parses() {
    // This is the exact form that --kv-quant rot_k_v4g64 would need to accept.
    let q = KvQuant::from_str("rot_k_v4g64").expect("rot_k_v4g64 must parse");
    assert_eq!(
        q,
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        }
    );
    // And the Display round-trip.
    assert_eq!(q.to_string(), "rot_k_v4g64");
}

/// Issue #26: `cache_key_salt` must be collision-free across distinct codecs so
/// the codec-partitioned prompt-cache key never conflates two codecs. Two
/// distinct `KvQuant` values (including payload-bearing variants that differ
/// only by bit-width / group size) must produce distinct salts; the same value
/// must produce the same salt (determinism).
#[test]
fn cache_key_salt_is_unique_and_deterministic() {
    // when adding a KvQuant variant, add it here
    let cases: &[KvQuant] = &[
        KvQuant::None,
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::Planar,
        KvQuant::K8VTurbo3,
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 8,
            k_group_size: 128,
            v_group_size: 128,
        },
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 64,
        },
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
    ];
    // Determinism: same value → same salt.
    for &q in cases {
        assert_eq!(
            q.cache_key_salt(),
            q.cache_key_salt(),
            "{q:?} cache_key_salt must be deterministic"
        );
    }
    // Uniqueness: every distinct pair must differ.
    for (i, &a) in cases.iter().enumerate() {
        for &b in &cases[i + 1..] {
            assert_ne!(
                a.cache_key_salt(),
                b.cache_key_salt(),
                "{a:?} and {b:?} must have distinct cache_key_salts (codec partitioning)"
            );
        }
    }
    // The two Mixed variants above differ only by bit-width/group size — their
    // salts must still diverge (payload is part of the codec identity).
    let m1 = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 4,
        k_group_size: 64,
        v_group_size: 64,
    };
    let m2 = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 8,
        k_group_size: 128,
        v_group_size: 128,
    };
    assert_ne!(m1.cache_key_salt(), m2.cache_key_salt());
}

// ── Issue #34: per-layer net-benefit estimator ────────────────────────────────

/// A windowed layer always runs the bf16 rotating ring regardless of the codec
/// flag, so the estimated net saving is exactly 0 (codec is a no-op there).
#[test]
fn windowed_layer_net_saving_is_zero_for_any_codec() {
    // Gemma4 e2b geometry: head_dim=256, kv_heads=1, window=512.
    for q in [KvQuant::K8V4, KvQuant::K8V8, KvQuant::Planar, KvQuant::None] {
        let saving = q.estimated_net_saving_per_layer(
            512,  // seq (== window)
            256,  // head_dim
            1,    // kv_heads
            true, // is_windowed
        );
        assert_eq!(
            saving, 0,
            "{q:?} windowed-layer net saving must be 0 (bf16 ring no-op)"
        );
    }
}

/// On a **global** layer at small context the scratch-heavy codec is
/// net-NEGATIVE: it keeps a full bf16 decode seed (warm-TTFT) plus packed codes
/// and per-group scales, so it is strictly larger than plain bf16. K8V4 on the
/// Gemma4 e2b global geometry must report a negative saving.
#[test]
fn global_layer_scratch_heavy_codec_is_net_negative_at_small_ctx() {
    // e2b global layer: global_head_dim=256, num_global_key_value_heads=1.
    let saving = KvQuant::K8V4.estimated_net_saving_per_layer(
        4096,  // seq
        256,   // head_dim
        1,     // kv_heads
        false, // global layer
    );
    assert!(
        saving < 0,
        "K8V4 global layer must be net-negative (bf16 seed + scales > bytes saved); got {saving}"
    );
    // Bonsai-like geometry (head_dim=128, 8 kv heads) is also net-negative for
    // any K8V* codec that retains the bf16 seed.
    let saving_bonsai = KvQuant::K8V8.estimated_net_saving_per_layer(2048, 128, 8, false);
    assert!(
        saving_bonsai < 0,
        "K8V8 global layer with retained bf16 seed must be net-negative; got {saving_bonsai}"
    );
}

/// A seed-free K-only codec that carries a sideband-heavy family codec on K
/// (iso quaternions) is net-NEGATIVE on memory: the per 4-element-group code +
/// scale + 4×f32 quaternion, plus one f32 norm per token, exceeds the bf16 K it
/// replaces — even though K's nominal code width is only 4 bits. This is the
/// operator truth the resolve-time net-negative warn relies on; a naive
/// bits-only model wrongly predicts a saving here. The only seed-free K-only
/// codecs in the matrix (`IsoKOnly*`/`RotorKOnly*`) all carry such a sideband,
/// so none of them actually save on a global layer.
#[test]
fn k_only_iso_codec_is_net_negative_from_sidebands() {
    let saving = KvQuant::IsoKOnly4.estimated_net_saving_per_layer(
        16_384, // seq (large)
        128,    // head_dim
        8,      // kv_heads
        false,  // global
    );
    assert!(
        saving < 0,
        "IsoKOnly4 K carries the iso quaternion sideband (larger than bf16 K) → net-negative; got {saving}"
    );
}

/// A both-seed-retaining codec (K8V4 keeps the bf16 K *and* V decode seed on
/// top of its codes) has a constant per-element overhead, so its net saving is
/// negative and scales **more** negative with context — there is no crossover
/// while both seeds are retained. This is the core issue #34 finding: the warm
/// seed is the dominant term, not the window size.
#[test]
fn both_seed_codec_gets_more_negative_with_context() {
    let small = KvQuant::K8V4.estimated_net_saving_per_layer(512, 256, 1, false);
    let large = KvQuant::K8V4.estimated_net_saving_per_layer(8192, 256, 1, false);
    assert!(
        small < 0 && large < 0,
        "both must be net-negative: small={small} large={large}"
    );
    assert!(
        large < small,
        "K8V4 retains both bf16 seeds → overhead scales with context (more negative): small={small} large={large}"
    );
}

/// The iso K-only codec's net cost scales with context: more tokens → more
/// quaternion/norm sideband, so the saving grows MORE negative. There is no
/// crossover — the sideband is a per-token overhead, not a fixed one, so it can
/// never amortize away.
#[test]
fn iso_k_only_codec_gets_more_negative_with_context() {
    let small = KvQuant::IsoKOnly4.estimated_net_saving_per_layer(64, 128, 8, false);
    let large = KvQuant::IsoKOnly4.estimated_net_saving_per_layer(16_384, 128, 8, false);
    assert!(
        small < 0 && large < 0,
        "iso K-only is net-negative at both contexts: small={small} large={large}"
    );
    assert!(
        large < small,
        "iso quaternion sideband is a per-token overhead → more negative with context: small={small} large={large}"
    );
}

/// bf16 (`None`) reports exactly the two bf16 buffers and zero saving vs itself.
#[test]
fn none_codec_zero_saving_vs_itself() {
    let bytes = KvQuant::None.estimated_resident_bytes_per_layer(1024, 128, 8);
    // 2 buffers × 1024 × 128 × 8 × 2 bytes.
    assert_eq!(bytes, 2 * 1024 * 128 * 8 * 2);
    let saving = KvQuant::None.estimated_net_saving_per_layer(1024, 128, 8, false);
    assert_eq!(saving, 0, "None vs bf16 baseline must net to 0");
}

/// The resident-bytes estimator must track the codec's ACTUAL stored bytes
/// (codes + scales + sidebands), not just nominal code bits — otherwise the
/// net-negative warn lies for sideband-heavy families (iso quaternions,
/// rotor group-3 scale cadence).
#[test]
fn estimator_matches_actual_iso_rotor_encode_bytes() {
    let head_dim = 128usize;
    let seq = 64u64;
    let kv_heads = 4u64;
    let n_tokens = (seq * kv_heads) as usize;
    let v: Vec<f32> = (0..n_tokens * head_dim)
        .map(|i| ((i % 251) as f32) / 251.0 - 0.5)
        .collect();

    // iso3: actual stored bytes per side (codes + scales + quaternions + norms).
    let (codes, scales, quats, norms) =
        crate::isoquant::iso_encode_fast(&v, head_dim, 4, 3).unwrap();
    let iso_side_actual = 4 * (codes.len() + scales.len() + quats.len() + norms.len()) as u64;

    // Iso3Sym quantizes BOTH sides with the iso codec and retains bf16 seeds
    // for both (warm-TTFT): estimate = 2*(side + seed).
    let elems = seq * head_dim as u64 * kv_heads;
    let seed = elems * 2;
    let est = KvQuant::Iso3Sym.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads);
    let expected = 2 * (iso_side_actual + seed);
    let tol = expected / 10; // ±10%
    assert!(
        est.abs_diff(expected) <= tol,
        "Iso3Sym estimate {est} not within 10% of actual {expected}"
    );

    // rotor3: per-token = n_groups*(code u32 + scale f32) + norm f32.
    let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
    let rotors = vec![0.5f32; n_groups * 4];
    let (r_codes, r_scales, r_norms) =
        crate::rotorquant::rotor3_encode(&v, &rotors, head_dim).unwrap();
    let rotor_side_actual = 4 * (r_codes.len() + r_scales.len() + r_norms.len()) as u64;
    let est_r =
        KvQuant::Rotor3Sym.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads);
    // Rotor3Sym quantizes BOTH sides and — unlike Iso3Sym — retains **no** bf16
    // seed on either: its decode is a flash kernel over the two packed rings, so
    // estimate = 2*side with no seed term. This is the codec's whole memory
    // claim; if a seed ever creeps back the estimate and this test move together.
    let expected_r = 2 * rotor_side_actual;
    let tol_r = expected_r / 10;
    assert!(
        est_r.abs_diff(expected_r) <= tol_r,
        "Rotor3Sym estimate {est_r} not within 10% of actual {expected_r}"
    );
    // The seedless estimate must be strictly below the seeded sibling's — the
    // point of the fused path. Same codec shape, same bit width, so the gap is
    // exactly the two mirrors.
    assert!(
        est_r < 2 * (rotor_side_actual + seed),
        "Rotor3Sym must not carry a bf16 mirror: {est_r} should be below the seeded {}",
        2 * (rotor_side_actual + seed)
    );

    // Net-saving must now be NEGATIVE for iso at head_dim=128.
    let saving =
        KvQuant::Iso3Sym.estimated_net_saving_per_layer(seq, head_dim as u64, kv_heads, false);
    assert!(
        saving < 0,
        "iso3 must report net-negative at head_dim=128, got {saving}"
    );

    // V-only variants keep an 8-bit AFFINE K — their K side must take the
    // generic codes+scales path, not the quaternion formula. Sym estimate
    // (iso K) must be strictly larger than the V-only estimate (affine K).
    let est_v_only =
        KvQuant::Iso3.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads);
    assert!(
        est_v_only < est,
        "Iso3 (affine K) estimate {est_v_only} must be below Iso3Sym (iso K) {est}"
    );
}
