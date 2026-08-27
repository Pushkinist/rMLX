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
use crate::mixed_quant::MixedTuple;
use rmlx_mlx::{quantize, Array, Device, Dtype};

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

/// `cache_key_salt` must be collision-free across distinct codecs so
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

// ── Per-layer net-benefit estimator ───────────────────────────────────────────

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

/// The two seed-free K-only families split on the bf16 floor, and the split is
/// the sideband width — not the code width, which is 4 bits in both cases.
///
/// `IsoKOnly4` packs 4 head-dim slots into one `u32` and spends one scale on
/// them, plus one norm per token: 8 bits of codes per value and, at the ring's
/// stored sideband dtype, 4.125 of sideband — under bf16's 16. `RotorKOnly4`
/// packs 3 slots into the same `u32`, so its codes alone are 10.67 bits per
/// value and its scale cadence is a third denser; the same narrowing leaves it
/// above the floor. Both are asserted here so neither result reads as a
/// property of "K-only codecs" — it is a property of each one's group geometry.
///
/// This is the operator truth the resolve-time net-benefit `warn!` relies on: a
/// naive bits-only model reads both as a 4× saving.
#[test]
fn k_only_iso_and_rotor_codecs_split_on_the_bf16_floor() {
    let (seq, head_dim, kv_heads) = (16_384_u64, 128_u64, 8_u64);
    let iso = KvQuant::IsoKOnly4.estimated_net_saving_per_layer(seq, head_dim, kv_heads, false);
    assert!(
        iso > 0,
        "IsoKOnly4's ring is 12.125 bits per value against bf16's 16 → net-positive; got {iso}"
    );
    let rotor = KvQuant::RotorKOnly4.estimated_net_saving_per_layer(seq, head_dim, kv_heads, false);
    assert!(
        rotor < 0,
        "RotorKOnly4's ring is 16.25 bits per value, still above bf16's 16 → net-negative; \
         got {rotor}"
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

/// Both K-only families' net figure scales linearly with context, in opposite
/// directions, because every term in the store is per-token: iso saves more the
/// longer the context, rotor costs more. Neither crosses over — there is no
/// fixed overhead to amortize away, so the sign is set by the geometry alone
/// and holds at every context.
#[test]
fn k_only_codec_net_figures_scale_with_context() {
    let iso_small = KvQuant::IsoKOnly4.estimated_net_saving_per_layer(64, 128, 8, false);
    let iso_large = KvQuant::IsoKOnly4.estimated_net_saving_per_layer(16_384, 128, 8, false);
    assert!(
        iso_small > 0 && iso_large > iso_small,
        "iso K-only saves at every context and saves more with it: \
         small={iso_small} large={iso_large}"
    );
    let rot_small = KvQuant::RotorKOnly4.estimated_net_saving_per_layer(64, 128, 8, false);
    let rot_large = KvQuant::RotorKOnly4.estimated_net_saving_per_layer(16_384, 128, 8, false);
    assert!(
        rot_small < 0 && rot_large < rot_small,
        "rotor K-only's overrun is per-token, so it grows with context: \
         small={rot_small} large={rot_large}"
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
///
/// The iso anchor is the **ring** payload, not the CPU encode's full output.
/// Both forms exist, and which one is resident is not a matter of taste: the
/// four iso codecs that materialise a store (`k_iso3/4`, `iso3_sym/4_sym`) all
/// decode through a fused kernel that reads the GPU ring, and their append
/// drops the CPU blocks the moment the ring is live
/// (`drop_blocks_when_ring_live_iso_*`). The quaternion the CPU blocks carry is
/// the constant `FIXED_QUAT` replicated per group and never reaches the ring.
/// So the ring is what a served request holds, and the block form — asserted
/// below at its measured 3.98x — is what the same store holds only between
/// `exit_prefill` and the first fused decode step.
#[test]
fn estimator_matches_actual_iso_rotor_encode_bytes() {
    let head_dim = 128usize;
    let seq = 64u64;
    let kv_heads = 4u64;
    let n_tokens = (seq * kv_heads) as usize;
    let v: Vec<f32> = (0..n_tokens * head_dim)
        .map(|i| ((i % 251) as f32) / 251.0 - 0.5)
        .collect();

    // iso3: the ring holds codes (u32) + scales (f32) + one norm (f32) per
    // token. `iso_encode_fast` also returns the per-group quaternion, which
    // `QuantKGpuRing::alloc` has no buffer for.
    let (codes, scales, quaternions, norms) =
        crate::isoquant::iso_encode_fast(&v, head_dim, 4, 3).unwrap();
    let iso_ring_actual = ring_side_bytes(
        &codes,
        &scales,
        &norms,
        kv_heads,
        head_dim / crate::storage::ISO3_GROUP_SIZE,
        seq,
    );
    let iso_blocks_actual = crate::storage::IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    }
    .byte_size();

    // The two forms differ by the quaternion sideband **and** by the width the
    // scale and norm planes are stored at: `f32` in the host `Vec`s, the ring's
    // stored sideband dtype on the GPU. Pinned so the choice above stays a
    // choice a reader can check rather than a claim.
    let block_ratio = iso_blocks_actual as f64 / iso_ring_actual as f64;
    assert!(
        (block_ratio - 3.979).abs() < 0.01,
        "iso CPU blocks must be 3.98x the ring at head_dim=128, got {block_ratio}"
    );

    // Iso3Sym quantizes BOTH sides with the iso codec and — like Rotor3Sym —
    // retains **no** bf16 seed on either: its decode is the quant-V flash kernel
    // over both packed iso rings, so estimate = 2*side with no seed term.
    let elems = seq * head_dim as u64 * kv_heads;
    let seed = elems * 2;
    let est = KvQuant::Iso3Sym.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads);
    let expected = 2 * iso_ring_actual;
    assert_eq!(
        est, expected,
        "Iso3Sym estimate {est} must be exactly the two ring payloads {expected}"
    );
    // The seedless estimate must be strictly below the seeded sibling's — the
    // point of the fused path.
    assert!(
        est < 2 * (iso_ring_actual + seed),
        "Iso3Sym must not carry a bf16 mirror: {est} should be below the seeded {}",
        2 * (iso_ring_actual + seed)
    );

    // rotor3: per-token = n_groups*(code u32 + scale) + one norm, the two
    // sideband planes at the ring's stored dtype. Rotor has no quaternion
    // analogue, so the ring and the CPU blocks carry the same *payload*; they
    // differ only in the width that payload's sideband is stored at.
    let n_groups = head_dim.div_ceil(crate::rotorquant::ROTOR3_GROUP_SIZE);
    let rotors = vec![0.5f32; n_groups * 4];
    let (r_codes, r_scales, r_norms) =
        crate::rotorquant::rotor3_encode(&v, &rotors, head_dim).unwrap();
    let rotor_side_actual = ring_side_bytes(&r_codes, &r_scales, &r_norms, kv_heads, n_groups, seq);
    let est_r =
        KvQuant::Rotor3Sym.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads);
    // Rotor3Sym quantizes BOTH sides and — like Iso3Sym — retains **no** bf16
    // seed on either: its decode is a flash kernel over the two packed rings, so
    // estimate = 2*side with no seed term. This is the codec's whole memory
    // claim; if a seed ever creeps back the estimate and this test move together.
    let expected_r = 2 * rotor_side_actual;
    assert_eq!(
        est_r, expected_r,
        "Rotor3Sym estimate {est_r} must be exactly the two ring payloads {expected_r}"
    );
    // The seedless estimate must be strictly below the seeded sibling's — the
    // point of the fused path. Same codec shape, same bit width, so the gap is
    // exactly the two mirrors.
    assert!(
        est_r < 2 * (rotor_side_actual + seed),
        "Rotor3Sym must not carry a bf16 mirror: {est_r} should be below the seeded {}",
        2 * (rotor_side_actual + seed)
    );

    // Net-saving is POSITIVE for iso at head_dim=128: the ring is 12.125 bits
    // per value against bf16's 16.0. Dropping the quaternion got it to 16.25 —
    // still above the floor — and narrowing the scale and norm planes to the
    // stored sideband dtype is what took it under. Anchored against the two
    // measured ring payloads so the number moves with the store, not with a
    // restated constant.
    let saving =
        KvQuant::Iso3Sym.estimated_net_saving_per_layer(seq, head_dim as u64, kv_heads, false);
    assert_eq!(
        saving,
        2 * (elems * 2) as i64 - 2 * iso_ring_actual as i64,
        "iso3's saving is exactly bf16's two buffers less the two ring payloads"
    );
    assert!(
        saving > 0,
        "iso3 must report a net saving at head_dim=128, got {saving}"
    );

    // Rotor does NOT clear the floor at the same geometry, and that is the
    // point of measuring the two families side by side: the same narrowing
    // takes rotor from 21.75 to 16.25 bits per value, a 25% cut that still
    // leaves it above bf16, because rotor spends a whole u32 code word per 3
    // head-dim slots — 10.67 bits per value before any sideband.
    let saving_r =
        KvQuant::Rotor3Sym.estimated_net_saving_per_layer(seq, head_dim as u64, kv_heads, false);
    assert!(
        saving_r < 0,
        "rotor3 must still report net-negative at head_dim=128, got {saving_r}"
    );

    // The V-only variant is a different codec, not a cheaper one. `Iso3`
    // decodes off the bf16 mirrors and builds no store at all
    // (`materialises_packed_store` is false), so its estimate is exactly bf16's
    // two buffers — which is now *larger* than the fused sibling's, the reverse
    // of the ordering that held while the ring's sideband was `f32`. Asserted
    // against `None` rather than as an inequality, because "equals bf16" is the
    // property, and an inequality would have kept passing in either direction.
    let est_v_only =
        KvQuant::Iso3.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads);
    assert_eq!(
        est_v_only,
        KvQuant::None.estimated_resident_bytes_per_layer(seq, head_dim as u64, kv_heads),
        "Iso3 builds no packed store, so it holds exactly the two bf16 mirrors"
    );
    assert!(
        est < est_v_only,
        "Iso3Sym {est} must be below the mirror-only Iso3 {est_v_only}"
    );
}

/// The eight iso / rotor codecs that quantize K split by family on the bf16
/// floor, at every context, `head_dim` and KV-head count — and the split does
/// not depend on any of those.
///
/// Both families spend one whole `u32` code word per group, plus one scale per
/// group and one norm per token, the two sideband planes at the ring's stored
/// dtype. The group is 4 head-dim slots for iso and 3 for rotor, so the codes
/// alone are 8 bits per value for iso and 10.67 for rotor. With a `bf16`
/// sideband that puts iso at `12 + 16/head_dim` and rotor at
/// `(48 * ceil(head_dim/3) + 16) / head_dim` — under and over bf16's 16
/// respectively, with no `head_dim` in between: iso's sideband term shrinks
/// with `head_dim` and rotor's code term never does. The nominal 3-bit / 4-bit
/// codebook width never reaches the store, which is why the 3-bit and 4-bit
/// member of each family occupy byte-identical storage.
///
/// Pinned as a sweep in both directions: a change that pushes iso back over the
/// floor, or that appears to pull rotor under it without changing rotor's code
/// cadence, has to come back through here.
#[test]
fn iso_k_codecs_win_and_rotor_k_codecs_do_not_at_every_geometry() {
    let iso = [
        KvQuant::IsoKOnly3,
        KvQuant::IsoKOnly4,
        KvQuant::Iso3Sym,
        KvQuant::Iso4Sym,
    ];
    let rotor = [
        KvQuant::RotorKOnly3,
        KvQuant::RotorKOnly4,
        KvQuant::Rotor3Sym,
        KvQuant::Rotor4Sym,
    ];
    for head_dim in [64_u64, 128, 256, 512] {
        for kv_heads in [1_u64, 8] {
            for seq in [256_u64, 4096, 65_536] {
                for q in iso {
                    let saving = q.estimated_net_saving_per_layer(seq, head_dim, kv_heads, false);
                    assert!(
                        saving > 0,
                        "{q} at seq={seq} head_dim={head_dim} kv_heads={kv_heads} reports \
                         {saving} B — the iso ring is 12 bits of codes plus a per-token \
                         sideband that shrinks with head_dim, so it is under bf16 at every \
                         geometry"
                    );
                }
                for q in rotor {
                    let saving = q.estimated_net_saving_per_layer(seq, head_dim, kv_heads, false);
                    assert!(
                        saving < 0,
                        "{q} at seq={seq} head_dim={head_dim} kv_heads={kv_heads} reports a \
                         {saving} B saving vs bf16 — rotor spends a u32 per 3 head-dim slots, \
                         so it cannot be smaller than bf16 whatever its sideband costs, and a \
                         non-negative number here means the layout model has drifted from the \
                         store"
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

/// [`ALL_KV_QUANTS`] indexes densely from zero and repeats no variant.
///
/// Pairs with `variant_index_has_one_arm_per_listed_codec`, which supplies the
/// count this test cannot: both sides of the comparison here are derived from
/// the list, so this one sees a duplicate or a re-used index and nothing else.
#[test]
fn all_kv_quants_indexes_densely_with_no_repeats() {
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
        "ALL_KV_QUANTS's indices must run 0..{n} with no gap — a hole means two \
         entries claim indices that skip one"
    );
}

/// [`ALL_KV_QUANTS`] names every variant — including one no value in the test
/// binary can construct.
///
/// The oracle is `variant_index`, whose `match` the compiler checks, but the
/// coupling has to be read out of the *source*: a variant wired into that match
/// and forgotten in the list produces no value anywhere in this crate, so no
/// test that sweeps the list can observe it. `ALL_KV_QUANTS.len()` and
/// `(0..ALL_KV_QUANTS.len())` are the same number twice. Counting the arms of
/// the match is the one reading that moves when the list does not.
///
/// The count is `=>` occurrences inside the fn body, so a rustfmt-wrapped arm
/// still counts once. Anything the scan cannot read back — a renamed fn, a
/// comment carrying `=>` — fails loudly rather than passing.
#[test]
fn variant_index_has_one_arm_per_listed_codec() {
    const SRC: &str = include_str!("quant.rs");
    const OPEN: &str = "pub fn variant_index(&self) -> usize {";

    let Some((_, after_open)) = SRC.split_once(OPEN) else {
        panic!("quant.rs no longer declares `{OPEN}` — this test reads that fn's arms")
    };
    // The fn body ends at the first line that closes an item at impl-block
    // indentation; everything before it is the `match self { ... }` arms.
    let Some((body, _)) = after_open.split_once("\n    }\n") else {
        panic!("could not find the end of `variant_index` in quant.rs")
    };
    let arms = body.lines().filter(|line| line.contains("=>")).count();

    assert_eq!(
        arms,
        ALL_KV_QUANTS.len(),
        "`variant_index` has {arms} arms but ALL_KV_QUANTS lists {} codecs. \
         A variant added to the enum reaches the match by compiler error; it \
         reaches the list, every sweep below, and the disposition manifest only \
         if someone adds it there too.",
        ALL_KV_QUANTS.len()
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
    /// packed store and prefill encodes nothing.
    ///
    /// **Equivalent to `Baseline`, not beaten by it.** Resident KV and greedy
    /// token ids measure identical (4 cells, 2 architectures); decode
    /// throughput against it is INCONCLUSIVE at all five recorded ABBA cells.
    /// An earlier draft called this class dominated, on the strength of a
    /// per-layer dispatch cost that `docs/KV_QUANT.md` § "`--kv-quant none` is
    /// a bf16 control" records as no longer reproducing. There is no axis left
    /// with a measured difference in either direction.
    InertMirrorFed,
    /// Decode reads this codec's own packed store. These are the only codecs
    /// that quantize anything a served request touches — the ones a fused
    /// decode kernel would make pay, and the ones whose residency actually
    /// differs from bf16 (today, upward).
    ReadsItsOwnStore,
}

/// Derive the disposition from the classifiers the runtime itself dispatches
/// on, so the table below cannot claim something the code does not do.
///
/// The store-reading class keys off [`KvQuant::decode_reads_packed_store`] —
/// the predicate the class name asserts — and **not** off
/// `materialises_packed_store`, which is the strictly weaker "`exit_prefill`
/// builds a store". The two agree on every variant today, which is exactly why
/// the weaker one must not be used: the table would then be right by accident,
/// and `an_inert_codec_has_no_quantised_read_path` would reduce to restating
/// the single predicate it was derived from. Keyed this way, that test asserts
/// three independent facts, and it fires the day a half-mirrored codec
/// (one axis quantised, the other bf16) makes the two predicates diverge —
/// which is the case `a_storeless_codec_mirrors_both_axes` exists to describe
/// and that nothing else would catch.
fn disposition_of(q: KvQuant) -> Disposition {
    if q == KvQuant::None {
        Disposition::Baseline
    } else if q.decode_reads_packed_store() {
        Disposition::ReadsItsOwnStore
    } else {
        Disposition::InertMirrorFed
    }
}

/// Every codec the tree can spell, with its disposition written out by name.
///
/// This exists so "nobody picks it" can never be an answer: a variant added to
/// the enum reaches [`ALL_KV_QUANTS`] (pinned by
/// `variant_index_has_one_arm_per_listed_codec`) and then has to be classified here
/// or the sweep below fails on it. Writing the
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

/// An inert codec is inert on both axes and allocates nothing packed.
///
/// This is what makes "identical resident KV and identical token ids to `none`"
/// a derivation rather than a coincidence of the four cells it was measured at:
/// with no store built and both axes fed from bf16, there is no path by which a
/// served request can differ.
///
/// [`disposition_of`] keys the class on `decode_reads_packed_store`, so that
/// predicate is the *premise* here and is not re-asserted — restating it would
/// be a tautology dressed as a check.
///
/// The two claims below are independent of the premise and of each other, each
/// against a different failure. Mutation-checked, both ways:
///
/// * A codec that stops mirroring one axis becomes store-*allocating* without
///   becoming store-*reading* — `materialises_packed_store` is
///   `decode_reads || !feeds_bf16_k || !feeds_bf16_v`. The first assertion
///   catches it, and it is caught **here** rather than absorbed into a class
///   change: keyed off the weaker predicate instead, the codec's derived class
///   would move, and relabelling its row would turn the suite green while the
///   table claimed a codec reads a store its own predicate denies.
/// * A change to `materialises_packed_store`'s definition that leaves per-codec
///   mirroring intact passes the first assertion and fails the second.
///
/// Deleting the two mirror disjuncts from `materialises_packed_store` remains
/// behaviour-preserving *today* — no shipped codec is half-mirrored — so that
/// mutation is a null in both. The first bullet is what makes the disjuncts a
/// live claim rather than only a documented intention.
#[test]
fn an_inert_codec_has_no_quantised_read_path() {
    for &(q, d) in DISPOSITIONS {
        if d != Disposition::InertMirrorFed {
            continue;
        }
        assert!(
            q.feeds_bf16_k_at_decode() && q.feeds_bf16_v_at_decode(),
            "{q} is recorded inert but one axis is not fed from the bf16 mirror — \
             that axis has to decode from somewhere, and it is not the mirror"
        );
        assert!(
            !q.materialises_packed_store(),
            "{q} is recorded inert (decode reads no packed store) but exit_prefill \
             still builds one — an O(context) allocation per layer that nothing \
             reads, which is what makes this class cost the same as `none`"
        );
    }
}

// ── The parametric families all validate their payload ───────────────────────

/// `rot_k_v<vb>g<vg>` accepts only V codecs its store can actually hold.
///
/// `RotK` builds `KvStorage::Mixed` via `MixedKvState::new_rotated`, so its V
/// side is the same MLX affine quantizer the `mixed_*` family hands its
/// `(bits, group_size)` to — and therefore the same accepted set. Before this
/// check the arm did no validation at all: `rot_k_v99g7` parsed into a codec
/// that would have asked MLX for a 99-bit affine quantize at its first encode.
///
/// The malformed spellings are here for the same reason. The shape already
/// matched by the time they fail, so they are bad `rot_k_*` tags and must say
/// so; reporting them as `Unknown` printed all 28 codec names and never named
/// the component that failed. Asserting the variant and not just "the message
/// contains the input" is what separates the two — `Unknown`'s message
/// contains the input as well.
#[test]
fn rot_k_rejects_a_v_codec_the_store_cannot_hold() {
    for tag in [
        // Shape is well-formed, the (bits, group) tuple is not.
        "rot_k_v99g7",
        "rot_k_v16g64",
        "rot_k_v4g17",
        "rot_k_v0g0",
        // Shape matched, a numeric component did not parse.
        "rot_k_vXg64",
        "rot_k_v4gY",
        "rot_k_v4",
    ] {
        let err = KvQuant::from_str(tag).unwrap_err();
        assert!(
            matches!(err, super::KvQuantParseError::InvalidRotK { .. }),
            "'{tag}' matched the rot_k_ shape, so its rejection must be \
             InvalidRotK and not a generic unknown-codec error: {err:?}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(tag),
            "the rejection should name the input: {msg}"
        );
    }
}

/// The accepted `rot_k_*` set is exactly the accepted `mixed_*` V set.
///
/// Stated as an equivalence rather than a second table, because a second table
/// is how the two drift: `RotK`'s V slot *is* the `Mixed` V slot.
#[test]
fn rot_k_and_mixed_accept_the_same_v_side() {
    for v_bits in 0u8..=16 {
        for v_group_size in [0u16, 1, 17, 31, 32, 63, 64, 100, 128, 256] {
            let rot_k = KvQuant::from_str(&format!("rot_k_v{v_bits}g{v_group_size}")).is_ok();
            let mixed = KvQuant::from_str(&format!("mixed_k8g64_v{v_bits}g{v_group_size}")).is_ok();
            assert_eq!(
                rot_k, mixed,
                "rot_k_v{v_bits}g{v_group_size} accepted={rot_k} but the same V side \
                 under mixed_ accepted={mixed}"
            );
        }
    }
}

// ── The disposition manifest the user-facing surfaces are checked against ────

/// The token a user-facing surface (CLI help, `docs/KV_QUANT.md`) can be
/// searched for to find this codec.
///
/// For a non-parametric codec that is its `Display` form. For the four
/// parametric families it is the fixed prefix their `Display` starts with,
/// because prose spells them `rot_k_v<vb>g<vg>`, not at one sample parameter.
///
/// Exhaustive on purpose: a new parametric family has to declare its prefix
/// here or this stops compiling, and the gate would otherwise search the
/// surfaces for a name that never appears in them.
fn surface_stem(q: KvQuant) -> String {
    match q {
        KvQuant::Mixed { .. } => "mixed_".to_string(),
        KvQuant::RotK { .. } => "rot_k_v".to_string(),
        KvQuant::RotorK3Asym { .. } => "rotor_k_3_asym_".to_string(),
        KvQuant::RotorK4Asym { .. } => "rotor_k_4_asym_".to_string(),
        KvQuant::None
        | KvQuant::K8V4
        | KvQuant::K8V8
        | KvQuant::Planar
        | KvQuant::Planar3
        | KvQuant::PlanarK
        | KvQuant::K8VTurbo3
        | KvQuant::K8VTurbo3Tcq
        | KvQuant::K8VTurbo2
        | KvQuant::K8VTurbo2Tcq
        | KvQuant::TurboSym3
        | KvQuant::TurboSym4
        | KvQuant::Iso3
        | KvQuant::Iso4
        | KvQuant::Iso3Sym
        | KvQuant::Iso4Sym
        | KvQuant::IsoKOnly3
        | KvQuant::IsoKOnly4
        | KvQuant::Rotor3
        | KvQuant::Rotor4
        | KvQuant::Rotor3Sym
        | KvQuant::Rotor4Sym
        | KvQuant::RotorKOnly3
        | KvQuant::RotorKOnly4 => q.to_string(),
    }
}

/// Print one line per codec, classified by the runtime's own predicates, for
/// `scripts/check_kv_codec_disposition.sh` to check the CLI help and
/// `docs/KV_QUANT.md` against.
///
/// The sweep is [`ALL_KV_QUANTS`], whose completeness
/// `variant_index_has_one_arm_per_listed_codec` pins against the
/// compiler-checked `variant_index` — so a codec cannot reach the CLI without
/// reaching this manifest, and the gate cannot go stale by omission.
///
/// `INERT` is [`KvQuant::materialises_packed_store`] returning false, which is
/// the disjunction of the three classifiers printed beside it and the exact
/// condition `exit_prefill` skips the encode on. The three are printed so a
/// reader of the gate's output can see which one moved. `KvQuant::None` also
/// builds no store — it has none to build — and is labelled `BASELINE` instead.
///
/// The predicate here is deliberately not `disposition_of`'s. That one keys on
/// `decode_reads_packed_store` — "this codec's decode reads its own store" —
/// because the class it names asserts exactly that, and the file argues above
/// why the weaker predicate must not stand in for it. This manifest is checked
/// against surfaces that promise an operator a codec *does something*, and the
/// thing `exit_prefill` gates on is `materialises_packed_store`. The two agree
/// on every variant today; they are still different questions, and a
/// half-mirrored codec would answer them differently.
///
/// Emits sentinels around the block: a run that reaches `BEGIN` and stops has
/// failed an assertion here, which is a violation; a run with no `BEGIN` at all
/// did not get far enough to have an opinion, which is an environment error.
/// The gate reports those two differently.
#[test]
fn emit_kv_codec_disposition_manifest() {
    println!("KVQUANT-DISPOSITION-BEGIN");
    let mut stems: Vec<String> = Vec::new();
    for &q in ALL_KV_QUANTS {
        let stem = surface_stem(q);
        assert!(!stem.is_empty(), "{q} has an empty surface stem");
        let mode = if stem == q.to_string() {
            "EXACT"
        } else {
            "PREFIX"
        };
        println!(
            "KVQUANT-DISPOSITION\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            q.variant_index(),
            q,
            stem,
            mode,
            if q == KvQuant::None {
                // The unquantised baseline has no packed store to skip. It
                // shares `materialises_packed_store() == false` with the inert
                // class and nothing else about it, so it gets its own label —
                // a surface must not describe bf16 as a codec that does nothing.
                "BASELINE"
            } else if q.materialises_packed_store() {
                "LIVE"
            } else {
                "INERT"
            },
            u8::from(q.decode_reads_packed_store()),
            u8::from(q.feeds_bf16_k_at_decode()),
            u8::from(q.feeds_bf16_v_at_decode()),
        );
        stems.push(stem);
    }
    let n = stems.len();
    stems.sort();
    stems.dedup();
    assert_eq!(
        stems.len(),
        n,
        "two codecs share a surface stem — a gate searching for one would find \
         the other"
    );
    println!("KVQUANT-DISPOSITION-END\t{n}");
}

// ── Store-cadence gate: the byte model against the real allocation ────────────
//
// `KvQuant::estimated_resident_bytes_per_layer` is the instrument the
// resolve-time net-benefit warn reads, and `scripts/perf_ceiling.py` mirrors it.
// A cadence that drifts from the store it models does not fail anything on its
// own — it just reports a saving the store never delivers. These tests close
// that by measuring each store's bytes from its own encoder over one shared
// fixture, at three head dimensions, and holding the model to the result.

/// One fixture geometry: `kv_heads * seq` rows of `head_dim`.
#[derive(Clone, Copy, Debug)]
struct Cadence {
    head_dim: u64,
    kv_heads: u64,
    seq: u64,
}

impl Cadence {
    fn rows(self) -> u64 {
        self.seq * self.kv_heads
    }
    fn values(self) -> u64 {
        self.rows() * self.head_dim
    }
    fn shape(self) -> [i32; 4] {
        [1, 1, self.rows() as i32, self.head_dim as i32]
    }
    fn data(self) -> Vec<f32> {
        (0..self.values())
            .map(|i| ((i % 251) as f32) / 251.0 - 0.5)
            .collect()
    }
}

/// The geometries every cadence assertion runs at.
///
/// `head_dim = 128` is the shipped test-target head dimension and the one every
/// published rate figure quotes; 256 is what a served gemma-4 layer uses; 64 is
/// the low end. A group-bound store's rate depends on `head_dim`, so a rate
/// quoted without one is not a number — and a proof at one `head_dim` is not a
/// proof: `SideStore::Rotor`'s `head_dim.div_ceil(3)` is the only non-linear
/// term in the whole model, and the group-bound arms divide `elems` by a group
/// size that a short `head_dim` could leave a remainder against.
const CADENCE_GEOMETRIES: &[Cadence] = &[
    Cadence {
        head_dim: 64,
        kv_heads: 4,
        seq: 64,
    },
    Cadence {
        head_dim: 128,
        kv_heads: 4,
        seq: 64,
    },
    Cadence {
        head_dim: 256,
        kv_heads: 4,
        seq: 64,
    },
];

/// The store one axis of a codec actually writes.
///
/// Named per codec by [`codec_side_layouts`] and measured by
/// [`measured_side_bytes`] from that store's own encoder — never from a table
/// of expected rates, which is a number that agrees with itself.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum StoreLayout {
    /// No packed store on this axis: one buffer at model dtype.
    Bf16,
    /// `QuantK` — affine q8_0, group 128.
    Q8,
    /// `QuantV` — TurboQuant Lloyd-Max at `bits`, group 32.
    ///
    /// This is also what `RotorK{3,4}Asym` puts its V in. The constructor it
    /// reaches is named `QuantV::new_affine_decode` and the storage field is
    /// documented as "MLX-affine", but the codec behind both is the TurboQuant
    /// N(0,1) Lloyd-Max one at a fixed 32-element group, and the codec's
    /// `v_group_size` never reaches it (`validate_rotor_k_asym_v`).
    Turbo(u8),
    /// `QuantV` with Viterbi assignment; decoder and layout identical to
    /// [`StoreLayout::Turbo`] at the same width.
    Tcq(u8),
    /// `QuantPlanarK` / `QuantPlanarV` — Givens rotation, per-pair scales.
    Planar(u8),
    /// IsoQuant on `QuantKGpuRing`: codes, scales, per-token norms. The
    /// resident form for every iso codec that materialises a store.
    IsoRing(u8),
    /// IsoQuant in CPU `IsoBlocks`: the ring plus a per-group quaternion.
    IsoBlocks(u8),
    /// RotorQuant — one code word and one scale per 3-element group. Ring and
    /// CPU blocks carry the same payload, so one layout covers both.
    Rotor(u8),
    /// MLX affine 3-tuple (`Mixed` / `RotK`).
    ///
    /// Measured like every other row, from `mx.quantize` itself: this crate has
    /// no CPU affine encoder, but MLX's is reachable and its output *is* the
    /// store — `exit_prefill` hands the 3-tuple straight to `MixedTuple`
    /// (`bulk_init_from_fp16`), and the incremental path's `init_quant`
    /// pre-allocates the same three arrays at the same dtypes (it takes
    /// `scales_dtype` from the same `keys.dtype()`). So the bytes below come
    /// from the allocation, not from a formula restating it.
    Affine { bits: u32, group: u32 },
}

/// The fixture as the bf16 KV stream a store sees.
///
/// bf16 and not f32 on purpose: the sideband width of the affine 3-tuple is the
/// input dtype, and `cast_store_bf16` floors the prefill buffer that
/// `exit_prefill` quantizes at bf16 on every arch. Measuring an f32 array would
/// measure a stream no cache holds.
fn cadence_bf16_array(data: &[f32], geom: Cadence) -> Array {
    Array::from_f32_slice(data, &geom.shape())
        .expect("fixture array")
        .astype(Dtype::Bf16, Device::Cpu)
        .expect("bf16 cast")
}

/// The MLX affine 3-tuple for one side of the fixture, as the store holds it.
fn affine_tuple(data: &[f32], geom: Cadence, bits: u32, group: u32) -> MixedTuple {
    let arr = cadence_bf16_array(data, geom);
    let (codes, scales, biases) = quantize(&arr, group as i32, bits as i32, Device::Cpu)
        .expect("mx.quantize(mode = affine) on the fixture");
    MixedTuple {
        codes,
        scales,
        biases,
    }
}

/// Bytes this store holds for the fixture.
#[allow(
    clippy::unwrap_used,
    reason = "every encoder here is called on the fixture with a shape and width it validates; a failure is a broken encoder and the panic names it"
)]
/// Bytes the shipped GPU ring holds for one already-encoded side.
///
/// Built and measured, not restated: `QuantKGpuRing::byte_size` reads each
/// buffer's own shape and dtype, so this tracks the stored sideband width
/// instead of asserting one — a `4 *` written here would have gone on reporting
/// the old width after the planes narrowed, and the cadence gate below it would
/// have gone green against a store it no longer describes.
///
/// `max_seq == seq` keeps `page_round` from rounding the allocation up to a
/// whole `KV_PAGE_SIZE`, so what comes back is the payload and not a page
/// ceiling.
fn ring_side_bytes(
    codes: &[u32],
    scales: &[f32],
    norms: &[f32],
    kv_heads: u64,
    n_groups: usize,
    seq: u64,
) -> u64 {
    let mut ring = crate::storage::QuantKGpuRing::default();
    ring.seed_from_cpu(
        codes,
        scales,
        norms,
        kv_heads as i32,
        n_groups as i32,
        seq as i32,
        seq as i32,
        Device::Cpu,
    )
    .expect("seed the ring from the encoder output");
    ring.byte_size()
}

fn measured_side_bytes(layout: StoreLayout, data: &[f32], geom: Cadence) -> u64 {
    let elems = geom.values();
    let head_dim = geom.head_dim as usize;
    let shape = geom.shape();
    match layout {
        StoreLayout::Bf16 => elems * 2,
        StoreLayout::Q8 => {
            let (codes, scales) = crate::q8::q8_quantize(data);
            (codes.len() + 4 * scales.len()) as u64
        }
        StoreLayout::Turbo(bits) => crate::turboquant::turbo_quantize_v(data, bits, &shape)
            .unwrap()
            .byte_size(),
        StoreLayout::Tcq(bits) => if bits == 2 {
            crate::tcq::tcq_quantize_v2(data, &shape).unwrap()
        } else {
            crate::tcq::tcq_quantize_v3(data, &shape).unwrap()
        }
        .byte_size(),
        StoreLayout::Planar(bits) => {
            crate::planarquant::planar_quantize(data, crate::turboquant::GROUP_SIZE, bits, &shape)
                .unwrap()
                .byte_size()
        }
        StoreLayout::IsoRing(bits) | StoreLayout::IsoBlocks(bits) => {
            let (codes, scales, quaternions, norms) =
                crate::isoquant::iso_encode_fast(data, head_dim, 4, bits).unwrap();
            if matches!(layout, StoreLayout::IsoBlocks(_)) {
                // The host `Vec` form, measured by the block's own `byte_size`.
                crate::storage::IsoBlocks {
                    codes,
                    scales,
                    quaternions,
                    norms,
                    n_tokens: geom.rows() as usize,
                }
                .byte_size()
            } else {
                ring_side_bytes(
                    &codes,
                    &scales,
                    &norms,
                    geom.kv_heads,
                    head_dim / crate::storage::ISO3_GROUP_SIZE,
                    geom.seq,
                )
            }
        }
        StoreLayout::Rotor(bits) => {
            let n_groups = crate::rotorquant::n_groups_for(head_dim);
            let rotors = crate::clifford::make_rotor_table(0, 0, n_groups);
            let (codes, scales, norms) = if bits == 3 {
                crate::rotorquant::rotor3_encode(data, &rotors, head_dim).unwrap()
            } else {
                crate::rotorquant::rotor4_encode(data, &rotors, head_dim).unwrap()
            };
            ring_side_bytes(&codes, &scales, &norms, geom.kv_heads, n_groups, geom.seq)
        }
        StoreLayout::Affine { bits, group } => affine_tuple(data, geom, bits, group).byte_size(),
    }
}

/// The estimator's layout for this store, or `None` for an unquantised axis.
fn model_side_store(layout: StoreLayout) -> Option<super::SideStore> {
    match layout {
        StoreLayout::Bf16 => None,
        StoreLayout::Q8 => Some(super::SideStore::Q8),
        StoreLayout::Turbo(_) | StoreLayout::Tcq(_) => Some(super::SideStore::Turbo),
        StoreLayout::Affine { group, .. } => Some(super::SideStore::Affine { group }),
        StoreLayout::Planar(_) => Some(super::SideStore::Planar),
        StoreLayout::IsoRing(_) => Some(super::SideStore::IsoRing),
        StoreLayout::IsoBlocks(_) => Some(super::SideStore::IsoBlocks),
        StoreLayout::Rotor(_) => Some(super::SideStore::Rotor),
    }
}

/// The store each axis of each codec writes, as `[K, V]`.
///
/// One arm per variant, no grouping: the arm count is what
/// [`codec_side_layouts_has_one_arm_per_listed_codec`] compares against
/// [`ALL_KV_QUANTS`], and a grouped arm would let a variant ride in on
/// another's line. Deliberately no `=>` in any comment inside the body for the
/// same reason.
#[allow(
    clippy::match_same_arms,
    reason = "one arm per variant even when two write the same pair of stores — merging them hides which codecs were considered, and this match exists to be read variant by variant"
)]
fn codec_side_layouts(q: KvQuant) -> [StoreLayout; 2] {
    match q {
        KvQuant::None => [StoreLayout::Bf16, StoreLayout::Bf16],
        KvQuant::K8V4 => [StoreLayout::Q8, StoreLayout::Turbo(4)],
        KvQuant::K8V8 => [StoreLayout::Q8, StoreLayout::Q8],
        KvQuant::Planar => [StoreLayout::Q8, StoreLayout::Planar(4)],
        KvQuant::Planar3 => [StoreLayout::Q8, StoreLayout::Planar(3)],
        KvQuant::PlanarK => [StoreLayout::Planar(4), StoreLayout::Bf16],
        KvQuant::Mixed {
            k_bits,
            v_bits,
            k_group_size,
            v_group_size,
        } => [
            StoreLayout::Affine {
                bits: u32::from(k_bits),
                group: u32::from(k_group_size),
            },
            StoreLayout::Affine {
                bits: u32::from(v_bits),
                group: u32::from(v_group_size),
            },
        ],
        KvQuant::RotK {
            v_bits,
            v_group_size,
        } => [
            StoreLayout::Affine { bits: 8, group: 64 },
            StoreLayout::Affine {
                bits: u32::from(v_bits),
                group: u32::from(v_group_size),
            },
        ],
        KvQuant::K8VTurbo3 => [StoreLayout::Q8, StoreLayout::Turbo(3)],
        KvQuant::K8VTurbo3Tcq => [StoreLayout::Q8, StoreLayout::Tcq(3)],
        KvQuant::K8VTurbo2 => [StoreLayout::Q8, StoreLayout::Turbo(2)],
        KvQuant::K8VTurbo2Tcq => [StoreLayout::Q8, StoreLayout::Tcq(2)],
        KvQuant::TurboSym3 => [StoreLayout::Turbo(3), StoreLayout::Turbo(3)],
        KvQuant::TurboSym4 => [StoreLayout::Turbo(4), StoreLayout::Turbo(4)],
        KvQuant::Iso3 => [StoreLayout::Q8, StoreLayout::IsoBlocks(3)],
        KvQuant::Iso4 => [StoreLayout::Q8, StoreLayout::IsoBlocks(4)],
        KvQuant::Iso3Sym => [StoreLayout::IsoRing(3), StoreLayout::IsoRing(3)],
        KvQuant::Iso4Sym => [StoreLayout::IsoRing(4), StoreLayout::IsoRing(4)],
        KvQuant::IsoKOnly3 => [StoreLayout::IsoRing(3), StoreLayout::Bf16],
        KvQuant::IsoKOnly4 => [StoreLayout::IsoRing(4), StoreLayout::Bf16],
        KvQuant::Rotor3 => [StoreLayout::Q8, StoreLayout::Rotor(3)],
        KvQuant::Rotor4 => [StoreLayout::Q8, StoreLayout::Rotor(4)],
        KvQuant::Rotor3Sym => [StoreLayout::Rotor(3), StoreLayout::Rotor(3)],
        KvQuant::Rotor4Sym => [StoreLayout::Rotor(4), StoreLayout::Rotor(4)],
        KvQuant::RotorKOnly3 => [StoreLayout::Rotor(3), StoreLayout::Bf16],
        KvQuant::RotorKOnly4 => [StoreLayout::Rotor(4), StoreLayout::Bf16],
        // The V axis is TurboQuant, not affine. `v_group_size` is a layout-key
        // tag the V encoder never reads, so it names no store parameter here.
        KvQuant::RotorK3Asym {
            v_bits,
            v_group_size: _,
        } => [StoreLayout::Rotor(3), StoreLayout::Turbo(v_bits)],
        KvQuant::RotorK4Asym {
            v_bits,
            v_group_size: _,
        } => [StoreLayout::Rotor(4), StoreLayout::Turbo(v_bits)],
    }
}

/// The affine sideband is a scale **and** a bias at the KV stream's dtype, and
/// the KV stream is bf16 — so 32 bits per group, not 64.
///
/// The figure was stated at both values in this crate and neither was read off
/// an allocation. It is load-bearing: `validate_mixed_side` accepts
/// `group_size == 32`, where the two readings differ by a whole bit per value —
/// `bits + 2.0` against `bits + 1.0`. The estimator now charges the measured
/// one, so the number below is the one `SideStore::Affine` spends and a change
/// to either has to come back through here.
///
/// Read from `mx.quantize`'s own output over a bf16 input, which is what
/// `exit_prefill` stores (`bulk_init_from_fp16`).
#[test]
fn affine_sideband_is_thirty_two_bits_per_group() {
    let geom = Cadence {
        head_dim: 128,
        kv_heads: 4,
        seq: 64,
    };
    let data = geom.data();
    let elems = geom.values();

    for (bits, group) in [(8u32, 32u32), (8, 64), (8, 128), (4, 64), (2, 32)] {
        let tuple = affine_tuple(&data, geom, bits, group);
        assert_eq!(
            tuple.scales.dtype(),
            Dtype::Bf16,
            "affine scales follow the input dtype; a bf16 KV stream must not \
             produce an f32 scale"
        );
        assert_eq!(tuple.biases.dtype(), Dtype::Bf16, "same for the bias");

        let codes = elems * u64::from(bits) / 8;
        let sideband = tuple.byte_size() - codes;
        let groups = elems / u64::from(group);
        assert_eq!(
            sideband * 8 / groups,
            32,
            "bits={bits} group={group}: the sideband is a scale and a bias at \
             the input dtype — 32 bits per group, not 64 (f32) and not 16 (one \
             scalar). Measured {sideband} B over {groups} groups."
        );
    }
}

/// Every codec's byte model is checked against the store it names, per axis and
/// then whole, at every geometry in [`CADENCE_GEOMETRIES`].
///
/// Three assertions per codec per geometry, and they fail for different reasons
/// on purpose:
///
/// 1. **The layout the estimator picked is the layout the codec writes.**
///    `KvQuant::side_stores` against [`codec_side_layouts`]. A codec whose store
///    changes family fails here first.
/// 2. **The cadence is right.** The estimator's per-side bytes against the bytes
///    that store's own encoder produced — byte for byte, no declared ratio and
///    no tolerance. Reaches every layout including the four no live codec
///    materialises today (`Q8`, `Turbo`, `Planar`, `IsoBlocks`), which are only
///    reachable this way, and a cadence nothing can call is a gate that cannot
///    fail.
/// 3. **The gating is right.** The whole-codec estimate against the per-side
///    model re-assembled through `materialises_packed_store` and the two
///    `feeds_bf16_*` predicates. Independent of (2): this one moves when a
///    codec's mirror or store disposition changes, not when a cadence does.
#[test]
fn every_codec_byte_model_matches_the_store_it_writes() {
    for &geom in CADENCE_GEOMETRIES {
        let data = geom.data();
        let elems = geom.values();
        let n_tokens = geom.rows();

        for &q in ALL_KV_QUANTS {
            let layouts = codec_side_layouts(q);
            let (k_bits, v_bits) = q.approx_code_bits();
            let bits = [k_bits, v_bits];
            let (k_store, v_store) = q.side_stores();
            let stores = [k_store, v_store];
            let packs = q.materialises_packed_store();
            let feeds = [q.feeds_bf16_k_at_decode(), q.feeds_bf16_v_at_decode()];

            let mut expected_total = 0u64;
            for axis in 0..2 {
                let (layout, store, side_bits) = (layouts[axis], stores[axis], bits[axis]);
                let side = if axis == 0 { "K" } else { "V" };

                // (1) same layout.
                assert_eq!(
                    store,
                    model_side_store(layout),
                    "{q} {side}: the estimator sizes this side from {store:?} but the codec \
                     writes {layout:?}"
                );

                // (2) same cadence, byte for byte.
                let actual = measured_side_bytes(layout, &data, geom);
                expected_total += match store {
                    None => {
                        assert_eq!(
                            actual,
                            elems * 2,
                            "{q} {side}: an unquantised axis is two bytes per value"
                        );
                        elems * 2
                    }
                    Some(store) => {
                        let modelled = super::packed_side_bytes(
                            store,
                            side_bits,
                            elems,
                            geom.head_dim,
                            n_tokens,
                        );
                        assert_eq!(
                            modelled, actual,
                            "{q} {side} at head_dim={}: the model holds {modelled} B for a \
                             {layout:?} store of {actual} B. The cadence drifted from the \
                             store it models.",
                            geom.head_dim
                        );
                        // (3) gating: a codec that builds no store holds only mirrors.
                        if packs {
                            modelled + if feeds[axis] { elems * 2 } else { 0 }
                        } else {
                            elems * 2
                        }
                    }
                };
            }

            assert_eq!(
                q.estimated_resident_bytes_per_layer(geom.seq, geom.head_dim, geom.kv_heads),
                expected_total,
                "{q} at head_dim={}: whole-codec estimate disagrees with its own per-side \
                 model assembled through materialises_packed_store + feeds_bf16_k/v",
                geom.head_dim
            );
        }
    }
}

/// Three of the seven [`SideStore`] variants are reachable from the estimator;
/// four are latent, and named only by codecs that build no store.
///
/// The type's own doc says which, and a doc is not a gate. The estimator sizes
/// a side only when `materialises_packed_store` holds, so a layout named
/// exclusively by mirror-family codecs is never evaluated in production — its
/// cadence is reached only by
/// [`every_codec_byte_model_matches_the_store_it_writes`], which is why that
/// test calls `packed_side_bytes` directly instead of going through the
/// estimator. If a latent layout becomes live (a codec grows a decode path over
/// its store), this fails and the doc has to be rewritten in the same change.
///
/// Latent is not dead: every one of the four is the true layout of the side that
/// names it, and this asserts each is named.
#[test]
fn the_estimator_reaches_three_of_the_seven_store_layouts() {
    fn tag(s: super::SideStore) -> &'static str {
        match s {
            super::SideStore::Q8 => "Q8",
            super::SideStore::Turbo => "Turbo",
            super::SideStore::Affine { .. } => "Affine",
            super::SideStore::Planar => "Planar",
            super::SideStore::IsoRing => "IsoRing",
            super::SideStore::IsoBlocks => "IsoBlocks",
            super::SideStore::Rotor => "Rotor",
        }
    }

    let mut live: Vec<&'static str> = Vec::new();
    let mut named: Vec<&'static str> = Vec::new();
    for &q in ALL_KV_QUANTS {
        let (k, v) = q.side_stores();
        for store in [k, v].into_iter().flatten() {
            named.push(tag(store));
            if q.materialises_packed_store() {
                live.push(tag(store));
            }
        }
    }
    live.sort_unstable();
    live.dedup();
    named.sort_unstable();
    named.dedup();

    assert_eq!(
        live,
        ["Affine", "IsoRing", "Rotor"],
        "the estimator evaluates these store layouts in production; the SideStore doc and \
         the net-negative warn's wording both name that set"
    );
    assert_eq!(
        named,
        [
            "Affine",
            "IsoBlocks",
            "IsoRing",
            "Planar",
            "Q8",
            "Rotor",
            "Turbo"
        ],
        "every SideStore variant must be the declared layout of some codec's side — one \
         that no codec names is a cadence modelling nothing"
    );
}

/// A side that reports 16 bits from `approx_code_bits` has no packed store, and
/// a side that has one reports fewer.
///
/// The two are separate matches over the same enum and nothing in the type
/// system couples them. `approx_code_bits`'s "a side kept at model dtype reports
/// 16" is the property several callers key off; `side_stores`'s `None` is what
/// the byte model branches on. A variant that gains a store on a side still
/// reporting 16 would be sized as bf16 and cost the estimate its whole store
/// term, silently.
#[test]
fn side_stores_agree_with_approx_code_bits() {
    for &q in ALL_KV_QUANTS {
        let (k_bits, v_bits) = q.approx_code_bits();
        let (k_store, v_store) = q.side_stores();
        for (side, bits, store) in [("K", k_bits, k_store), ("V", v_bits, v_store)] {
            assert_eq!(
                bits >= 16,
                store.is_none(),
                "{q} {side}: approx_code_bits says {bits} but side_stores says {store:?} — \
                 one of the two matches has a variant the other does not"
            );
        }
    }
}

/// [`codec_side_layouts`] names every variant, one arm each.
///
/// Same oracle and same reason as `variant_index_has_one_arm_per_listed_codec`:
/// the `match` is exhaustive so a new variant cannot be forgotten, but a variant
/// folded into a neighbour's arm with a `|` would be swept by
/// [`every_codec_byte_model_matches_the_store_it_writes`] under the neighbour's
/// layouts and never on its own. Counting the arms out of the source is the one
/// reading that moves when the grouping does. This repo has shipped a sweep that
/// iterated a literal list and ran 21 variants behind while reporting full
/// coverage; the count is what stops that.
///
/// The count alone does not: exhaustiveness plus an equal arm count forces a
/// bijection only while no arm can cover two variants and none can cover a
/// variant the compiler would otherwise demand. A `_ =>` catch-all added
/// alongside one `|`-grouped arm restores both the count and the
/// exhaustiveness, and defeats the reading. So the shape of the arms is
/// asserted too, not just how many there are.
#[test]
fn codec_side_layouts_has_one_arm_per_listed_codec() {
    const SRC: &str = include_str!("quant_tests.rs");
    const OPEN: &str = "fn codec_side_layouts(q: KvQuant) -> [StoreLayout; 2] {";

    let Some((_, after_open)) = SRC.split_once(OPEN) else {
        panic!("quant_tests.rs no longer declares `{OPEN}` — this test reads that fn's arms")
    };
    // A free fn's body ends at the first line that closes an item at column 0;
    // the inner `match` closes one level in.
    let Some((body, _)) = after_open.split_once("\n}\n") else {
        panic!("could not find the end of `codec_side_layouts` in quant_tests.rs")
    };
    for line in body.lines() {
        let t = line.trim();
        assert!(
            !t.starts_with("_ =>") && !t.starts_with("_=>") && !t.starts_with("_ if"),
            "`codec_side_layouts` has a catch-all arm (`{t}`). With one, the count below \
             stops implying a bijection: a `|`-grouped arm can hide a variant and the \
             catch-all restores the count the compiler would otherwise have broken."
        );
        assert!(
            !t.starts_with('|') && !t.contains(" | "),
            "`codec_side_layouts` groups patterns with `|` (`{t}`). One arm per variant — \
             a grouped arm is swept under its neighbour's layouts and never on its own."
        );
    }
    let arms = body.lines().filter(|line| line.contains("=>")).count();

    assert_eq!(
        arms,
        ALL_KV_QUANTS.len(),
        "`codec_side_layouts` has {arms} arms but ALL_KV_QUANTS lists {} codecs. One arm \
         per variant, no `|` grouping — a grouped arm hides a codec inside another's \
         layouts.",
        ALL_KV_QUANTS.len()
    );
}
