//! Display↔FromStr round-trip tests for all `KvQuant` variants.
//!
//! Every value that `Display` emits must parse back to the identical variant
//! via `FromStr`. This file is the guard that catches any future asymmetry.
//!
//! Deliberately-excluded variants (intentionally Display-only / not a
//! standalone CLI preset):
//! — none currently; all variants are expected to round-trip.

use std::str::FromStr;

use super::{KvQuant, ALL_KV_QUANTS};

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

/// On a **global** layer a codec that keeps a packed store **and** both bf16
/// mirrors is net-NEGATIVE: the codes and per-group scales are pure addition on
/// top of a mirror pair that is already exactly bf16-sized. `Mixed` is that
/// shape — its decode reads the packed 3-tuples, so the store is real, and it
/// still hands both mirrors to a cross-layer-KV consumer.
#[test]
fn global_layer_store_plus_mirror_codec_is_net_negative() {
    let mixed = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 8,
        k_group_size: 64,
        v_group_size: 64,
    };
    // e2b global layer: global_head_dim=256, num_global_key_value_heads=1.
    let saving = mixed.estimated_net_saving_per_layer(
        4096,  // seq
        256,   // head_dim
        1,     // kv_heads
        false, // global layer
    );
    assert!(
        saving < 0,
        "a store + both-mirror codec must be net-negative (codes + scales on top of \
         bf16-sized mirrors); got {saving}"
    );
    // Bonsai-like geometry (head_dim=128, 8 kv heads) reports the same sign.
    let saving_bonsai = mixed.estimated_net_saving_per_layer(2048, 128, 8, false);
    assert!(
        saving_bonsai < 0,
        "store + both-mirror codec must be net-negative on the dense geometry too; \
         got {saving_bonsai}"
    );
}

/// A codec whose decode reads only the bf16 mirrors builds no packed store, so
/// its estimate is exactly the two mirrors — the same bytes as `None`, at every
/// context and every geometry. Break-even, never negative: this is the estimate
/// side of the `exit_prefill` skip, and it would go negative again the moment a
/// store is materialised for a codec that reads none.
#[test]
fn mirror_only_codec_is_exactly_break_even_with_bf16() {
    for q in [
        KvQuant::K8V4,
        KvQuant::K8V8,
        KvQuant::Planar,
        KvQuant::Planar3,
        KvQuant::PlanarK,
        KvQuant::TurboSym4,
        KvQuant::Rotor3,
    ] {
        assert!(
            !q.materialises_packed_store(),
            "{q:?} is expected to be a mirror-only codec"
        );
        for (seq, head_dim, kv_heads) in [(512u64, 256u64, 1u64), (8192, 128, 8), (131_072, 128, 8)]
        {
            let saving = q.estimated_net_saving_per_layer(seq, head_dim, kv_heads, false);
            assert_eq!(
                saving, 0,
                "{q:?} at seq={seq} head_dim={head_dim} kv_heads={kv_heads}: a mirror-only \
                 codec holds exactly the bf16 bytes, so the saving is 0"
            );
        }
    }
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

/// A codec that keeps both bf16 mirrors **and** a packed store has a constant
/// per-element overhead, so its net saving is negative and scales **more**
/// negative with context — there is no crossover while both are retained. The
/// warm mirror is the dominant term, not the window size.
#[test]
fn store_plus_mirror_codec_gets_more_negative_with_context() {
    let mixed = KvQuant::Mixed {
        k_bits: 8,
        v_bits: 8,
        k_group_size: 64,
        v_group_size: 64,
    };
    let small = mixed.estimated_net_saving_per_layer(512, 256, 1, false);
    let large = mixed.estimated_net_saving_per_layer(8192, 256, 1, false);
    assert!(
        small < 0 && large < 0,
        "both must be net-negative: small={small} large={large}"
    );
    assert!(
        large < small,
        "retaining both mirrors alongside the store → overhead scales with context \
         (more negative): small={small} large={large}"
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

    // Iso3Sym quantizes BOTH sides with the iso codec and — like Rotor3Sym —
    // retains **no** bf16 seed on either: its decode is the quant-V flash kernel
    // over both packed iso rings, so estimate = 2*side with no seed term. (The
    // estimator still counts the quaternion sideband, matching the CPU encode
    // bytes; the resident ring drops it, so real residency is below this — the
    // estimate is a conservative upper bound, not a census.)
    let elems = seq * head_dim as u64 * kv_heads;
    let seed = elems * 2;
    let est = KvQuant::Iso3Sym.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads);
    let expected = 2 * iso_side_actual;
    let tol = expected / 10; // ±10%
    assert!(
        est.abs_diff(expected) <= tol,
        "Iso3Sym estimate {est} not within 10% of actual {expected}"
    );
    // The seedless estimate must be strictly below the seeded sibling's — the
    // point of the fused path.
    assert!(
        est < 2 * (iso_side_actual + seed),
        "Iso3Sym must not carry a bf16 mirror: {est} should be below the seeded {}",
        2 * (iso_side_actual + seed)
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

/// None of the eight iso / rotor codecs that quantize K can be a memory win —
/// at any context, any `head_dim`, any KV-head count.
///
/// Both families spend one whole `u32` code word **and** one `f32` scale per
/// group — 4 head-dim slots for iso, 3 for rotor — so a packed side costs at
/// least 2 B per value (iso) or 2.67 B per value (rotor) before its per-token
/// norm, while bf16 costs exactly 2. The nominal 3-bit / 4-bit codebook width
/// never reaches the store, which is why the 3-bit and 4-bit member of each
/// family occupy byte-identical storage. No shape amortizes that away: the
/// overhead is per token, not fixed.
///
/// Pinned as a sweep so a future store layout that genuinely does buy
/// compression has to update this test deliberately, rather than quietly
/// re-introducing a saving claim the format never delivered.
#[test]
fn iso_and_rotor_k_codecs_are_never_a_memory_win() {
    let codecs = [
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
    ];
    for q in codecs {
        for head_dim in [64_u64, 128, 256, 512] {
            for kv_heads in [1_u64, 8] {
                for seq in [256_u64, 4096, 65_536] {
                    let saving = q.estimated_net_saving_per_layer(seq, head_dim, kv_heads, false);
                    assert!(
                        saving < 0,
                        "{q} at seq={seq} head_dim={head_dim} kv_heads={kv_heads} reports a \
                         {saving} B saving vs bf16 — the group-bound store cannot be smaller \
                         than bf16, so a non-negative number here means the layout model has \
                         drifted from the store"
                    );
                }
            }
        }
    }
}

/// A `mixed_*` spec whose widths the codec cannot store is a parse error.
///
/// `mixed_k16g64_v16g64` used to parse. Its `approx_code_bits()` is then
/// `(16, 16)` — the value that means "this side is kept at model dtype" — so
/// every check keyed on that property read it as a codec that quantizes
/// nothing, while at runtime it still packed affine codes. The width set is
/// MLX's affine `quantize` set; anything else is rejected with the valid
/// widths named.
#[test]
fn mixed_rejects_widths_the_codec_cannot_store() {
    for spec in [
        "mixed_k16g64_v16g64",
        "mixed_k8g64_v16g64",
        "mixed_k16g64_v4g64",
        "mixed_k0g64_v4g64",
        "mixed_k8g64_v7g64",
        "mixed_k8g100_v4g64",
    ] {
        let parsed = KvQuant::from_str(spec);
        assert!(
            parsed.is_err(),
            "{spec} must not parse — it names a width the Mixed codec cannot store, got {parsed:?}"
        );
    }
}

/// Every `mixed_*` spec that DOES parse quantizes at least one side.
///
/// This is the invariant the boundary-layer quality floor keys on: a codec is
/// exempt from the promotion only when it keeps both sides at model dtype, and
/// no parametric spelling of `Mixed` may reach that state.
#[test]
fn every_parseable_mixed_quantizes_a_side() {
    for bits in 0u8..=17 {
        for group in [32u16, 64, 128] {
            let spec = format!("mixed_k{bits}g{group}_v{bits}g{group}");
            let Ok(q) = KvQuant::from_str(&spec) else {
                continue;
            };
            let (k, v) = q.approx_code_bits();
            assert!(
                k < 16 || v < 16,
                "{spec} parsed but reports model-dtype width on both sides ({k}, {v}) — \
                 it would read as a codec that quantizes nothing"
            );
        }
    }
}

// ── The codec surface is swept exhaustively, by construction ─────────────────

/// [`ALL_KV_QUANTS`] names every variant exactly once.
///
/// The oracle is `variant_index`, whose `match` the compiler checks: a variant
/// added to the enum and not to the list leaves a hole in the index set here,
/// and a variant added to neither fails to compile in `quant.rs`. Every sweep
/// test below inherits its exhaustiveness from this one.
#[test]
fn all_kv_quants_names_every_variant_once() {
    let mut seen: Vec<usize> = ALL_KV_QUANTS.iter().map(KvQuant::variant_index).collect();
    seen.sort_unstable();
    let n = seen.len();
    seen.dedup();
    assert_eq!(
        seen.len(),
        n,
        "ALL_KV_QUANTS lists a variant twice: {ALL_KV_QUANTS:?}"
    );
    assert_eq!(
        seen,
        (0..n).collect::<Vec<_>>(),
        "ALL_KV_QUANTS must cover every discriminant `variant_index` can return \
         — a gap means a variant was added to the enum but not to the list"
    );
}

/// A codec that builds no packed store must have a storage variant that can
/// **report** that it holds none.
///
/// The two predicates live on different enums and nothing else couples them.
/// `KvQuant::materialises_packed_store` decides whether `exit_prefill` builds a
/// payload; `KvStorage::geometry_only_max_seq` is what the spill writer asks
/// before it stamps a codec geometry. A codec classified `false` whose storage
/// sits in the "payload is not an `Option`" arm (`Mixed | Paged`)
/// compiles cleanly and makes the writer emit a codec tag with no tensors
/// behind it — the reader then fails on `missing tensor 'lN.k.codes'`.
///
/// The storage is built through the same `KvStorage::new` the cache uses, so
/// this is the real pairing and not a restatement of either predicate.
#[test]
fn a_storeless_codec_always_has_a_geometry_only_storage() {
    for &q in ALL_KV_QUANTS {
        if q.materialises_packed_store() {
            continue;
        }
        let storage = crate::storage::KvStorage::new(q, 4096);
        assert!(
            storage.geometry_only_max_seq().is_some(),
            "{q:?} builds no packed store, but its storage cannot report itself \
             geometry-only — the spill writer would stamp a codec geometry with \
             no tensors behind it"
        );
    }
}

/// A codec with no packed store reads a bf16 mirror on **both** axes.
///
/// `materialises_packed_store` is defined as
/// `decode_reads_packed_store() || !feeds_bf16_k || !feeds_bf16_v`, so `false`
/// implies both mirrors — which is what makes the byte estimate's
/// "return the two mirrors" branch total. Stated over every variant here
/// instead of as a `debug_assert` inside that branch, where it is both
/// unreachable and compiled out under `release-perf`.
#[test]
fn a_storeless_codec_mirrors_both_axes() {
    for &q in ALL_KV_QUANTS {
        if q.materialises_packed_store() {
            continue;
        }
        assert!(
            q.feeds_bf16_k_at_decode() && q.feeds_bf16_v_at_decode(),
            "{q:?} has no packed store and no mirror on one axis — that axis has \
             nowhere to decode from"
        );
    }
}

// ── codec disposition ────────────────────────────────────────────────────────

/// What selecting a codec actually does to a served request.
///
/// The three classes are what a disposition has to distinguish, because they
/// warrant different outcomes: a codec that is *beaten* by the bf16 baseline on
/// every axis is not in the same position as one that is merely not selected by
/// default, and neither is in the same position as the baseline itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disposition {
    /// Unquantised bf16. The reference every other row is measured against and
    /// the resolved `auto` default.
    Baseline,
    /// Decode reads the bf16 mirror on both axes, so `exit_prefill` builds no
    /// packed store. Resident KV and generated token ids are identical to
    /// `Baseline`; the only difference is the cost of carrying a quantised
    /// layer type. Beaten by `none` on every axis a caller can observe.
    InertMirrorFed,
    /// Decode reads this codec's own packed store. These are the only codecs
    /// that quantize anything a served request touches — the ones a fused
    /// decode kernel would make pay, and the ones whose residency actually
    /// differs from bf16 (today, upward).
    ReadsItsOwnStore,
}

/// Derive the disposition from the classifiers the runtime itself dispatches
/// on, so the table below cannot claim something the code does not do.
fn disposition_of(q: KvQuant) -> Disposition {
    if q == KvQuant::None {
        Disposition::Baseline
    } else if q.materialises_packed_store() {
        Disposition::ReadsItsOwnStore
    } else {
        Disposition::InertMirrorFed
    }
}

/// Every codec the tree can spell, with its disposition written out by name.
///
/// This exists so "nobody picks it" can never be an answer: a variant added to
/// the enum reaches [`ALL_KV_QUANTS`] (pinned by `variants_are_exhaustive`) and
/// then has to be classified here or the sweep below fails on it. Writing the
/// class by hand rather than deriving it is the point — the derivation is what
/// is being checked.
///
/// Parameterised families are listed at the same representative parameters
/// `ALL_KV_QUANTS` uses; their disposition does not vary with the parameters,
/// which `disposition_is_a_property_of_the_family_not_its_parameters` pins.
const DISPOSITIONS: &[(KvQuant, Disposition)] = &[
    (KvQuant::None, Disposition::Baseline),
    (KvQuant::K8V4, Disposition::InertMirrorFed),
    (KvQuant::K8V8, Disposition::InertMirrorFed),
    (KvQuant::Planar, Disposition::InertMirrorFed),
    (KvQuant::Planar3, Disposition::InertMirrorFed),
    (KvQuant::PlanarK, Disposition::InertMirrorFed),
    (
        KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        },
        Disposition::ReadsItsOwnStore,
    ),
    (
        KvQuant::RotK {
            v_bits: 8,
            v_group_size: 64,
        },
        Disposition::ReadsItsOwnStore,
    ),
    (KvQuant::K8VTurbo3, Disposition::InertMirrorFed),
    (KvQuant::K8VTurbo3Tcq, Disposition::InertMirrorFed),
    (KvQuant::K8VTurbo2, Disposition::InertMirrorFed),
    (KvQuant::K8VTurbo2Tcq, Disposition::InertMirrorFed),
    (KvQuant::TurboSym3, Disposition::InertMirrorFed),
    (KvQuant::TurboSym4, Disposition::InertMirrorFed),
    (KvQuant::Iso3, Disposition::InertMirrorFed),
    (KvQuant::Iso4, Disposition::InertMirrorFed),
    (KvQuant::Iso3Sym, Disposition::ReadsItsOwnStore),
    (KvQuant::Iso4Sym, Disposition::ReadsItsOwnStore),
    (KvQuant::IsoKOnly3, Disposition::ReadsItsOwnStore),
    (KvQuant::IsoKOnly4, Disposition::ReadsItsOwnStore),
    (KvQuant::Rotor3, Disposition::InertMirrorFed),
    (KvQuant::Rotor4, Disposition::InertMirrorFed),
    (KvQuant::Rotor3Sym, Disposition::ReadsItsOwnStore),
    (KvQuant::Rotor4Sym, Disposition::ReadsItsOwnStore),
    (KvQuant::RotorKOnly3, Disposition::ReadsItsOwnStore),
    (KvQuant::RotorKOnly4, Disposition::ReadsItsOwnStore),
    (
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        Disposition::InertMirrorFed,
    ),
    (
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 64,
        },
        Disposition::InertMirrorFed,
    ),
];

/// The hand-written table and the runtime classifiers agree, on every variant.
#[test]
fn every_codec_carries_a_disposition() {
    for &(q, want) in DISPOSITIONS {
        assert_eq!(
            disposition_of(q),
            want,
            "{q} is recorded as {want:?} but the runtime classifiers say \
             {:?} — one of the two moved without the other",
            disposition_of(q)
        );
    }
}

/// The table covers `ALL_KV_QUANTS` exactly — no variant unclassified, none
/// listed twice, none left behind after a retirement.
///
/// Driven off the const beside the enum rather than a count kept here: a
/// literal expected length is satisfied by a swap as well as by coverage.
#[test]
fn disposition_table_covers_every_variant_once() {
    for &q in ALL_KV_QUANTS {
        let hits = DISPOSITIONS.iter().filter(|(k, _)| *k == q).count();
        assert_eq!(
            hits, 1,
            "{q} appears {hits} times in the disposition table — every codec \
             needs exactly one, or a reader cannot tell which class it is in"
        );
    }
    assert_eq!(
        DISPOSITIONS.len(),
        ALL_KV_QUANTS.len(),
        "the disposition table names a codec that is not in ALL_KV_QUANTS — a \
         retired variant left a row behind"
    );
}

/// The bf16 baseline is the only codec in its class.
///
/// If a second variant ever classifies as `Baseline` the comparison every other
/// row is measured against stops being a single thing, and "identical to
/// `none`" stops naming one number.
#[test]
fn exactly_one_codec_is_the_baseline() {
    let n = DISPOSITIONS
        .iter()
        .filter(|(_, d)| *d == Disposition::Baseline)
        .count();
    assert_eq!(n, 1, "expected exactly one Baseline codec, found {n}");
}

/// A codec's disposition is a property of its family, not of the bits and
/// group sizes a caller spells.
///
/// The four parameterised families appear once each in the table above, at one
/// representative parameter set. That is only a legitimate stand-in for the
/// family if the classification cannot move with the parameters — otherwise the
/// table would be silent about every point it does not name.
#[test]
fn disposition_is_a_property_of_the_family_not_its_parameters() {
    let mixed: &[KvQuant] = &[
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
            k_bits: 8,
            v_bits: 2,
            k_group_size: 64,
            v_group_size: 32,
        },
    ];
    for &q in mixed {
        assert_eq!(disposition_of(q), Disposition::ReadsItsOwnStore, "{q}");
    }

    let rot_k: &[KvQuant] = &[
        KvQuant::RotK {
            v_bits: 8,
            v_group_size: 64,
        },
        KvQuant::RotK {
            v_bits: 4,
            v_group_size: 128,
        },
    ];
    for &q in rot_k {
        assert_eq!(disposition_of(q), Disposition::ReadsItsOwnStore, "{q}");
    }

    // `validate_rotor_k_asym_v` accepts (4, 128|64|32) and (3|2, 64).
    let rotor_asym: &[KvQuant] = &[
        KvQuant::RotorK3Asym {
            v_bits: 4,
            v_group_size: 128,
        },
        KvQuant::RotorK3Asym {
            v_bits: 2,
            v_group_size: 64,
        },
        KvQuant::RotorK4Asym {
            v_bits: 4,
            v_group_size: 32,
        },
        KvQuant::RotorK4Asym {
            v_bits: 3,
            v_group_size: 64,
        },
    ];
    for &q in rotor_asym {
        assert_eq!(disposition_of(q), Disposition::InertMirrorFed, "{q}");
    }
}

/// An inert codec is inert on both axes and reads nothing packed.
///
/// This is what makes "identical resident KV and identical token ids to `none`"
/// a derivation rather than a coincidence of the two cells it was measured at:
/// with no store built and both axes fed from bf16, there is no path by which a
/// served request can differ.
#[test]
fn an_inert_codec_has_no_quantised_read_path() {
    for &(q, d) in DISPOSITIONS {
        if d != Disposition::InertMirrorFed {
            continue;
        }
        assert!(
            !q.decode_reads_packed_store(),
            "{q} is recorded inert but decode reads its packed store"
        );
        assert!(
            q.feeds_bf16_k_at_decode() && q.feeds_bf16_v_at_decode(),
            "{q} is recorded inert but one axis is not fed from the bf16 mirror"
        );
        assert!(
            !q.materialises_packed_store(),
            "{q} is recorded inert but exit_prefill still builds its store"
        );
    }
}
