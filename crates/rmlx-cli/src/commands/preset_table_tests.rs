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

// ── recommend_preset decision tree ───────────────────────────────────────────
//
// Coverage: all 5 decision-tree branches (fp16, q8, 4-bit, 2-bit, fallback)
// × current starter preset set.  Each fixture documents the expected outcome
// under the current starter set (fp16, q8, quality, speed, planar, planar3,
// k_only_planar).  Comments note what would change after balanced and
// max_compression land.

/// Branch 1: total bf16 fits — should return "fp16".
/// (8B model, 8k ctx, 192 GB VRAM → easily fits bf16)
#[test]
fn recommend_fp16_when_everything_fits() {
    let preset = recommend_preset(8.0, 8_192, 192.0);
    assert_eq!(preset, "fp16", "8B@8k on 192GB should choose fp16");
}

/// Branch 1 edge: tiny model, short context, reasonable VRAM.
/// (1B model, 4k ctx, 16 GB VRAM)
#[test]
fn recommend_fp16_small_model_16gb() {
    let preset = recommend_preset(1.0, 4_096, 16.0);
    assert_eq!(preset, "fp16", "1B@4k on 16GB should choose fp16");
}

/// Branch 2: model + kv/2 fits → q8.
/// (8B model, 32k ctx, 32 GB VRAM)
/// model_bytes  = 8 * 2e9 = 16 GB
/// kv_bf16      = 8 * 32768 * 1e6 / 1e9 ≈ 262 GB
/// total_bf16   ≈ 278 GB  > budget(32*0.7=22.4 GB) → not fp16
/// model + kv/2 ≈ 16 + 131 = 147 GB > budget → not q8 either
/// Actually: need model + kv/2 < 22.4 → 16 + kv/2 < 22.4 → kv/2 < 6.4 → kv < 12.8
/// kv = 8 * 32768 * 1e6 / 1e9 ≈ 262. That won't hit q8.
/// Use shorter context: 8B, 2k, 32GB
/// model=16 GB, kv=8*2048*1e6/1e9=16.4 GB, total=32.4 > 22.4
/// model + kv/2 = 16 + 8.2 = 24.2 > 22.4 → also doesn't fit q8
/// Use 8B, 1k, 64GB:
/// model=16, kv=8*1024*1e6/1e9=8.2, total=24.2; budget=64*0.7=44.8
/// total < budget → fp16. Need a case where total > budget but model+kv/2 < budget.
/// 8B, 2k, 32GB: total=32.4 > 22.4; model+kv/2=24.2 > 22.4 → no
/// 8B, 1k, 32GB: total=24.2 > 22.4; model+kv/2=16+4.1=20.1 < 22.4 → q8 ✓
#[test]
fn recommend_q8_when_half_kv_fits() {
    // 8B model, 1024 ctx, 32GB VRAM
    let preset = recommend_preset(8.0, 1_024, 32.0);
    assert_eq!(preset, "q8", "8B@1k on 32GB should choose q8");
}

/// Branch 2 alt: same q8 scenario with explicit numbers check.
/// (70B model, 4k ctx, 96GB VRAM)
/// model=140GB, kv=70*4096*1e6/1e9=286GB, total=426
/// budget=96*0.7=67.2 → not fp16
/// model+kv/2=140+143=283 > 67.2 → not q8
/// 70B doesn't fit in 96GB at 4k — use (7B, 512, 48GB)
/// model=14, kv=7*512*1e6/1e9=3.584, total=17.584; budget=33.6
/// total < budget → fp16.
/// (7B, 4k, 24GB): model=14, kv=7*4096*1e6/1e9=28.7, total=42.7; budget=16.8
/// model+kv/2=14+14.3=28.3 > 16.8 → not q8
/// (7B, 2k, 24GB): model=14, kv=14.3, total=28.3 > 16.8; model+kv/2=14+7.2=21.2 > 16.8
/// (7B, 1k, 24GB): model=14, kv=7.2, total=21.2 > 16.8; model+kv/2=14+3.6=17.6 > 16.8
/// (7B, 512, 24GB): model=14, kv=3.6, total=17.6 > 16.8; model+kv/2=14+1.8=15.8 < 16.8 ✓ q8
#[test]
fn recommend_q8_7b_512ctx_24gb() {
    let preset = recommend_preset(7.0, 512, 24.0);
    assert_eq!(preset, "q8", "7B@512ctx on 24GB should choose q8");
}

/// Branch 3: model + kv/4 fits → preferred_4bit() = "quality" (current starter set).
/// Need: model + kv/2 >= budget but model + kv/4 < budget.
/// (7B, 1k, 24GB): budget=16.8, model=14, kv=7.168
/// model+kv/2=14+3.584=17.584 > 16.8 → not q8
/// model+kv/4=14+1.792=15.792 < 16.8 → 4-bit ✓
#[test]
fn recommend_4bit_when_quarter_kv_fits() {
    let preset = recommend_preset(7.0, 1_024, 24.0);
    // current starter set: preferred_4bit() returns "quality"
    assert_eq!(
        preset, "quality",
        "7B@1k on 24GB should choose quality (4-bit)"
    );
}

/// Branch 3 alt: (13B, 2k, 48GB)
/// model=26, kv=13*2048*1e6/1e9=26.6, budget=33.6
/// total=52.6 > 33.6 → not fp16
/// model+kv/2=26+13.3=39.3 > 33.6 → not q8
/// model+kv/4=26+6.65=32.65 < 33.6 → 4-bit ✓
#[test]
fn recommend_4bit_13b_2k_48gb() {
    let preset = recommend_preset(13.0, 2_048, 48.0);
    assert_eq!(
        preset, "quality",
        "13B@2k on 48GB should choose quality (4-bit)"
    );
}

/// Branch 4: model + kv/8 fits → preferred_2bit().
/// Current starter set: preferred_2bit() tries max_compression → balanced → quality → q8.
/// max_compression and balanced are not yet present, so it falls to "quality".
/// Need: model + kv/4 >= budget but model + kv/8 < budget.
/// (7B, 2k, 24GB): model=14, kv=14.336, budget=16.8
/// model+kv/4=14+3.584=17.584 > 16.8 → not 4-bit
/// model+kv/8=14+1.792=15.792 < 16.8 → 2-bit ✓
#[test]
fn recommend_2bit_when_eighth_kv_fits() {
    let preset = recommend_preset(7.0, 2_048, 24.0);
    // current starter set: preferred_2bit() falls through to "quality" (max_compression/balanced missing)
    assert_eq!(
        preset, "quality",
        "7B@2k on 24GB should choose quality (preferred_2bit fallback to quality)"
    );
}

/// Branch 4 alt: (13B, 4k, 48GB)
/// model=26, kv=53.2, budget=33.6
/// model+kv/4=26+13.3=39.3 > 33.6 → not 4-bit
/// model+kv/8=26+6.65=32.65 < 33.6 → 2-bit ✓
#[test]
fn recommend_2bit_13b_4k_48gb() {
    let preset = recommend_preset(13.0, 4_096, 48.0);
    assert_eq!(
        preset, "quality",
        "13B@4k on 48GB should choose quality (preferred_2bit)"
    );
}

/// Branch 5: nothing fits → "max_compression_fallback".
/// (70B model, 32k ctx, 24GB VRAM)
/// model=140, kv=70*32768*1e6/1e9=2293, budget=16.8
/// model alone exceeds budget → fallback
#[test]
fn recommend_max_compression_fallback_when_nothing_fits() {
    let preset = recommend_preset(70.0, 32_768, 24.0);
    assert_eq!(
        preset, "max_compression_fallback",
        "70B@32k on 24GB should return max_compression_fallback"
    );
}

/// Branch 5 alt: very large model, small VRAM.
/// (405B model, 8k ctx, 8GB VRAM)
#[test]
fn recommend_max_compression_fallback_405b_8gb() {
    let preset = recommend_preset(405.0, 8_192, 8.0);
    assert_eq!(
        preset, "max_compression_fallback",
        "405B on 8GB should return max_compression_fallback"
    );
}

// ── preferred_4bit / preferred_2bit fallback chain ───────────────────────────

/// Under current starter set, preferred_4bit() returns "quality" (present).
#[test]
fn preferred_4bit_returns_quality_under_t07() {
    assert_eq!(preferred_4bit(), "quality");
}

/// Under current starter set, preferred_2bit() falls through max_compression
/// (absent) → balanced (absent) → quality (present) → returns "quality".
#[test]
fn preferred_2bit_falls_back_to_quality_under_t07() {
    assert_eq!(preferred_2bit(), "quality");
}

/// preset_exists returns true for all starter presets.
#[test]
fn preset_exists_true_for_starter_presets() {
    for (name, _) in PRESETS {
        assert!(
            preset_exists(name),
            "preset_exists must be true for '{name}'"
        );
    }
}

/// preset_exists returns false for unknown names.
#[test]
fn preset_exists_false_for_unknown() {
    assert!(
        !preset_exists("max_compression"),
        "max_compression not yet in table"
    );
    assert!(!preset_exists("balanced"), "balanced not yet in table");
    assert!(!preset_exists("nonexistent_preset_xyz"), "random name");
}

// ── regression — reserved-name behaviour unchanged ───────────────────────────

/// `lookup_preset("auto")` still returns `Err(Reserved)` — parse.rs
/// intercepts "auto" before calling lookup_preset.
#[test]
fn auto_still_reserved_in_lookup() {
    assert_eq!(lookup_preset("auto"), Err(PresetError::Reserved));
}

// ── decision-tree boundary math ───────────────────────────────────────────────

/// Boundary between fp16 and q8: at exactly 70% utilisation total should
/// switch from fp16 to q8.
///
/// (1B model, 100k ctx, 64GB VRAM):
/// model=2e9, kv=1*100000*1e6=1e11, total≈102e9
/// budget=64e9*0.7=44.8e9
/// total > budget → not fp16; model+kv/2=2e9+50e9=52e9 > 44.8e9 → not q8
/// → 4-bit branch: model+kv/4=2e9+25e9=27e9 < 44.8e9 → 4-bit ✓
#[test]
fn decision_tree_boundary_4bit_1b_100k_64gb() {
    let preset = recommend_preset(1.0, 100_000, 64.0);
    assert_eq!(
        preset, "quality",
        "1B@100k on 64GB: bf16 KV doesn't fit but 4-bit does"
    );
}

/// Verify that zero VRAM (degenerate input) always returns fallback.
#[test]
fn recommend_zero_vram_returns_fallback() {
    let preset = recommend_preset(7.0, 4_096, 0.0);
    assert_eq!(
        preset, "max_compression_fallback",
        "zero VRAM must return max_compression_fallback"
    );
}
