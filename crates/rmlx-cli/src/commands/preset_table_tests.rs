use super::*;
use rmlx_kv_quant::KvQuant;

// ── lookup_preset: happy-path — every starter preset resolves ────────────────

#[test]
fn preset_fp16_resolves_to_kv_quant_none() {
    // KvQuant::None = bf16 both sides (the variant named None, not Option::None).
    let spec = lookup_preset("fp16").expect("fp16 must resolve");
    assert_eq!(spec.kv_quant, KvQuant::None);
    assert!(!spec.requires_calibration);
}

#[test]
fn preset_q8_resolves_to_k8v8() {
    let spec = lookup_preset("q8").expect("q8 must resolve");
    assert_eq!(spec.kv_quant, KvQuant::K8V8);
    assert!(!spec.requires_calibration);
}

#[test]
fn preset_speed_resolves_to_tsym3() {
    // speed promoted from K8VTurbo3 to TurboSym3 (symmetric WHT-3 K+V,
    // matching mtq's `speed` preset definition exactly).
    let spec = lookup_preset("speed").expect("speed must resolve");
    assert_eq!(spec.kv_quant, KvQuant::TurboSym3);
    assert!(!spec.requires_calibration);
}

#[test]
fn preset_quality_resolves_to_tsym4() {
    // `quality` resolves to symmetric WHT-4 K + tq4 V, matching
    // mtq's `quality` definition byte-for-byte.
    let spec = lookup_preset("quality").expect("quality must resolve");
    assert_eq!(spec.kv_quant, KvQuant::TurboSym4);
    assert!(!spec.requires_calibration);
}

#[test]
fn preset_planar_resolves_to_planar() {
    let spec = lookup_preset("planar").expect("planar must resolve");
    assert_eq!(spec.kv_quant, KvQuant::Planar);
    assert!(!spec.requires_calibration);
}

#[test]
fn preset_k_only_planar_resolves_to_planar_k() {
    // `k_only_planar` resolves to KvQuant::PlanarK (K-axis 4-bit;
    // V stays bf16). Mirrors mtq's `k_only_planar` preset.
    let spec = lookup_preset("k_only_planar").expect("k_only_planar must resolve");
    assert_eq!(spec.kv_quant, KvQuant::PlanarK);
    assert!(!spec.requires_calibration);
}

// ── planar3 preset ────────────────────────────────────────────────────────────

#[test]
fn preset_planar3_resolves_to_planar3() {
    // `planar3` resolves to KvQuant::Planar3 (K=q8_0, V=3-bit PlanarQuant).
    let spec = lookup_preset("planar3").expect("planar3 must resolve");
    assert_eq!(spec.kv_quant, KvQuant::Planar3);
    assert!(!spec.requires_calibration);
}

// ── lookup_preset: starter table has exactly 7 entries ──────────────────────

#[test]
fn preset_table_has_seven_entries() {
    assert_eq!(
        PRESETS.len(),
        7,
        "starter table must have exactly 7 preset rows (5 starter + k_only_planar + planar3)"
    );
}

// ── lookup_preset: error paths ───────────────────────────────────────────────

#[test]
fn auto_is_reserved() {
    assert_eq!(lookup_preset("auto"), Err(PresetError::Reserved));
}

#[test]
fn unknown_name_returns_unknown_error() {
    assert_eq!(lookup_preset("turbo9000"), Err(PresetError::Unknown));
    assert_eq!(lookup_preset(""), Err(PresetError::Unknown));
    assert_eq!(
        lookup_preset("SPEED"),
        Err(PresetError::Unknown),
        "lookup is case-sensitive"
    );
}

// ── available_names constant covers all starter presets ─────────────────────

#[test]
fn available_names_contains_all_starter_presets() {
    // Split on ", " and check exact membership — substring match (e.g.
    // "q".contains in "q8") would cause spurious passes for short names.
    let listed: Vec<&str> = AVAILABLE_NAMES.split(", ").collect();
    for (name, _) in PRESETS {
        assert!(
            listed.contains(name),
            "AVAILABLE_NAMES must list '{name}' as a distinct entry"
        );
    }
}

// ── regression — reserved-name behaviour unchanged ───────────────────────────

/// `lookup_preset("auto")` still returns `Err(Reserved)` — parse.rs
/// intercepts "auto" before calling lookup_preset.
#[test]
fn auto_still_reserved_in_lookup() {
    assert_eq!(lookup_preset("auto"), Err(PresetError::Reserved));
}

// ── what a preset actually buys ──────────────────────────────────────────────

/// No preset in the table reduces resident KV, and the table says so.
///
/// Every non-`fp16` row resolves to a codec whose decode reads the bf16 mirror,
/// so `exit_prefill` never builds its packed store and a served request holds
/// exactly the bytes `fp16` holds. Measured on gemma-4-e2b (`kv_h == 1`,
/// shared-KV) and Ternary-Bonsai-8B (`kv_h == 8`, dense) at two contexts:
/// identical `kv_cache_bytes` and identical greedy token ids in every cell.
///
/// This is a **pin on a claim the docs make**, not a rule against progress. A
/// preset whose codec grows a decode path over its own store is exactly the
/// outcome the codec work is aiming at — when one lands, this test fails, and
/// the fix is to say so in `docs/KV_QUANT.md` and the `--kv-preset` long help
/// rather than to silence the test.
#[test]
fn no_preset_is_a_memory_lever() {
    for (name, spec) in PRESETS {
        assert!(
            !spec.kv_quant.materialises_packed_store(),
            "preset '{name}' resolves to {}, which now builds a packed store — \
             the claim that no preset changes resident KV is stale. Update \
             docs/KV_QUANT.md and the --kv-preset long help, then this pin.",
            spec.kv_quant
        );
    }
}

/// `--kv-preset auto` and `--kv-quant auto` resolve to the same codec, from the
/// strings an operator actually types.
///
/// Driven through both real parsers rather than through `KvPresetArg::Auto` and
/// the constant: comparing `DEFAULT_KV_QUANT` against itself would hold no
/// matter what `parse_kv_preset` did with the word `auto`, and the defect this
/// pins was a second resolver sitting behind exactly that word. `parse_kv_quant`
/// returns `Ok(None)` for `auto` — the "use the engine default" sentinel — so
/// the two sides are compared after each has been resolved the way its own
/// command path resolves it.
#[test]
fn preset_auto_is_the_same_default_as_kv_quant_auto() {
    use crate::commands::parse::{parse_kv_preset, parse_kv_quant, resolve_preset_arg};
    let via_preset = resolve_preset_arg(parse_kv_preset("auto").expect("`auto` must parse"));
    let via_kv_quant = parse_kv_quant("auto")
        .expect("`auto` must parse")
        .unwrap_or(rmlx_models::kv_cache::DEFAULT_KV_QUANT);
    assert_eq!(
        via_preset, via_kv_quant,
        "--kv-preset auto and --kv-quant auto must resolve to one codec"
    );
    assert_eq!(
        via_preset,
        rmlx_models::kv_cache::DEFAULT_KV_QUANT,
        "--kv-preset auto must read DEFAULT_KV_QUANT, not a second resolver"
    );
}

/// Every preset name resolves to its own codec, from the string inward.
///
/// `parse_kv_preset` → `resolve_preset_arg` is the whole path a `--kv-preset`
/// argument takes. Building a `KvPresetArg::Resolved` from the table and
/// asserting it comes back unchanged would exercise neither the name lookup nor
/// the parser, so a name→codec regression would pass it.
#[test]
fn named_presets_still_resolve_to_their_own_codec() {
    use crate::commands::parse::{parse_kv_preset, resolve_preset_arg};
    for (name, spec) in PRESETS {
        let parsed = parse_kv_preset(name).unwrap_or_else(|e| panic!("preset '{name}': {e}"));
        assert_eq!(
            resolve_preset_arg(parsed),
            spec.kv_quant,
            "preset '{name}' must resolve to its own codec"
        );
    }
}
