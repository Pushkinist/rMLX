use super::*;
use serde_json::Value;
use std::collections::BTreeMap;

fn make_idx(entries: &[(&str, &str)]) -> ShardIndex {
    let weight_map = entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect::<BTreeMap<_, _>>();
    ShardIndex {
        metadata: Value::Null,
        weight_map,
    }
}

#[test]
fn plain_only() {
    let idx = make_idx(&[
        ("norm.weight", "s.safetensors"),
        ("embed.weight", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    assert_eq!(resolved.len(), 2);
    assert!(resolved.iter().all(|t| t.kind == TensorKind::Plain));
}

#[test]
fn affine_triplet() {
    let idx = make_idx(&[
        ("layers.0.mlp.weight", "s.safetensors"),
        ("layers.0.mlp.scales", "s.safetensors"),
        ("layers.0.mlp.biases", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    assert_eq!(resolved.len(), 1);
    let t = &resolved[0];
    assert_eq!(t.base_name, "layers.0.mlp");
    assert_eq!(t.kind, TensorKind::Affine);
    assert!(t.scales_shard.is_some());
    assert!(t.biases_shard.is_some());
}

#[test]
fn mxfp_pair() {
    let idx = make_idx(&[
        ("layers.0.mlp.weight", "s.safetensors"),
        ("layers.0.mlp.scales", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    assert_eq!(resolved.len(), 1);
    let t = &resolved[0];
    assert_eq!(t.base_name, "layers.0.mlp");
    assert_eq!(t.kind, TensorKind::Mxfp);
    assert!(t.scales_shard.is_some());
    assert!(t.biases_shard.is_none());
}

#[test]
fn mixed_map() {
    let idx = make_idx(&[
        // affine
        ("mlp.gate.weight", "s1.safetensors"),
        ("mlp.gate.scales", "s1.safetensors"),
        ("mlp.gate.biases", "s1.safetensors"),
        // mxfp
        ("mlp.down.weight", "s1.safetensors"),
        ("mlp.down.scales", "s1.safetensors"),
        // plain .weight
        ("embed_tokens.weight", "s1.safetensors"),
        // plain non-.weight
        ("norm.weight", "s2.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    let by_kind = |k: &TensorKind| resolved.iter().filter(|t| &t.kind == k).count();
    assert_eq!(by_kind(&TensorKind::Affine), 1, "affine");
    assert_eq!(by_kind(&TensorKind::Mxfp), 1, "mxfp");
    assert_eq!(by_kind(&TensorKind::Plain), 2, "plain");
    assert_eq!(resolved.len(), 4);
}

#[test]
fn orphan_scales_is_err() {
    // .scales without a .weight or plain peer.
    let idx = make_idx(&[("dangling.scales", "s.safetensors")]);
    assert!(resolve(&idx).is_err());
}

#[test]
fn sorted_output() {
    let idx = make_idx(&[
        ("z.weight", "s.safetensors"),
        ("a.weight", "s.safetensors"),
        ("m.weight", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    let names: Vec<&str> = resolved.iter().map(|t| t.base_name.as_str()).collect();
    assert_eq!(names, ["a", "m", "z"]);
}

#[test]
fn plain_non_weight_suffix() {
    // Entries that don't end in .weight/.scales/.biases (like input_max, input_min)
    // should be treated as plain base names.
    let idx = make_idx(&[
        ("audio_tower.layers.0.input_max", "s.safetensors"),
        ("audio_tower.layers.0.input_min", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    assert_eq!(resolved.len(), 2);
    assert!(resolved.iter().all(|t| t.kind == TensorKind::Plain));
}

/// ParoQuant layer: qweight + scales + qzeros + pairs + theta + channel_scales.
/// The base name should appear once, classified as ParoQuant.
#[test]
fn paroquant_layer_resolves_as_paroquant() {
    let idx = make_idx(&[
        // PARO siblings
        ("layers.0.mlp.down_proj.qweight", "s.safetensors"),
        ("layers.0.mlp.down_proj.scales", "s.safetensors"),
        ("layers.0.mlp.down_proj.qzeros", "s.safetensors"),
        ("layers.0.mlp.down_proj.pairs", "s.safetensors"),
        ("layers.0.mlp.down_proj.theta", "s.safetensors"),
        ("layers.0.mlp.down_proj.channel_scales", "s.safetensors"),
        // Plain weight alongside
        ("embed_tokens.weight", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    // Expect: 1 ParoQuant entry + 1 Plain entry
    let paro: Vec<_> = resolved
        .iter()
        .filter(|t| t.kind == TensorKind::ParoQuant)
        .collect();
    assert_eq!(paro.len(), 1, "expected 1 ParoQuant entry");
    assert_eq!(paro[0].base_name, "layers.0.mlp.down_proj");
    let plain_count = resolved
        .iter()
        .filter(|t| t.kind == TensorKind::Plain)
        .count();
    assert_eq!(plain_count, 1, "expected 1 plain entry");
    // Total: no extra phantom entries from the sibling keys
    assert_eq!(resolved.len(), 2, "total resolved entries");
}

/// Vanilla (non-PARO) safetensors must not trigger ParoQuant — no `.pairs` sibling.
#[test]
fn vanilla_affine_does_not_trigger_paroquant() {
    let idx = make_idx(&[
        ("layers.0.mlp.weight", "s.safetensors"),
        ("layers.0.mlp.scales", "s.safetensors"),
        ("layers.0.mlp.biases", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    assert_eq!(resolved.len(), 1);
    assert_eq!(resolved[0].kind, TensorKind::Affine);
}

/// Mixed checkpoint: some PARO layers and some plain/affine layers.
#[test]
fn mixed_paro_and_plain() {
    let idx = make_idx(&[
        // PARO linear layer
        ("model.layers.0.mlp.gate.qweight", "s.safetensors"),
        ("model.layers.0.mlp.gate.scales", "s.safetensors"),
        ("model.layers.0.mlp.gate.qzeros", "s.safetensors"),
        ("model.layers.0.mlp.gate.pairs", "s.safetensors"),
        ("model.layers.0.mlp.gate.theta", "s.safetensors"),
        ("model.layers.0.mlp.gate.channel_scales", "s.safetensors"),
        // Plain norm weight
        ("model.norm.weight", "s.safetensors"),
    ]);
    let resolved = resolve(&idx).unwrap();
    let n_paro = resolved
        .iter()
        .filter(|t| t.kind == TensorKind::ParoQuant)
        .count();
    let n_plain = resolved
        .iter()
        .filter(|t| t.kind == TensorKind::Plain)
        .count();
    assert_eq!(n_paro, 1);
    assert_eq!(n_plain, 1);
    assert_eq!(resolved.len(), 2);
}

// ----- try_exact_then_suffix tests ---------------------------------------

/// exact match returns the key unchanged (fast path).
#[test]
fn suffix_match_exact_hit() {
    let idx = make_idx(&[
        ("model.layers.0.self_attn.q_proj.weight", "s.safetensors"),
        ("model.norm.weight", "s.safetensors"),
    ]);
    let result = try_exact_then_suffix(&idx, "model.layers.0.self_attn.q_proj.weight", 5);
    assert_eq!(
        result,
        Some("model.layers.0.self_attn.q_proj.weight"),
        "exact match should return the key as-is"
    );
}

/// suffix fallback resolves a tensor when the checkpoint uses a
/// different prefix (`transformer.h.N.*` vs requested `model.layers.N.*`).
#[test]
fn suffix_match_different_prefix() {
    let idx = make_idx(&[
        ("transformer.h.0.attn.q_proj.weight", "model.safetensors"),
        ("transformer.wte.weight", "model.safetensors"),
    ]);
    // Caller knows the sub-path but not the outer prefix.
    let result = try_exact_then_suffix(&idx, "model.layers.0.attn.q_proj.weight", 4);
    assert_eq!(
        result,
        Some("transformer.h.0.attn.q_proj.weight"),
        "suffix match should find the key under the different prefix"
    );
}

/// suffix fallback with bare prefix (`layers.N.*`).
#[test]
fn suffix_match_bare_layers_prefix() {
    let idx = make_idx(&[
        ("layers.0.mlp.down_proj.weight", "s.safetensors"),
        ("norm.weight", "s.safetensors"),
    ]);
    let result = try_exact_then_suffix(&idx, "model.layers.0.mlp.down_proj.weight", 4);
    assert_eq!(
        result,
        Some("layers.0.mlp.down_proj.weight"),
        "bare-prefix tensor should be found via suffix match"
    );
}

/// no match returns None — caller can handle with its own error.
#[test]
fn suffix_match_no_match_returns_none() {
    let idx = make_idx(&[("model.norm.weight", "s.safetensors")]);
    let result = try_exact_then_suffix(&idx, "model.layers.0.mlp.gate.weight", 4);
    assert!(result.is_none(), "absent tensor should yield None");
}

/// suffix_segments=0 disables Phase 2; a name absent from the map returns None.
#[test]
fn suffix_match_zero_segments_no_phase2_fallback() {
    let idx = make_idx(&[("model.norm.weight", "s.safetensors")]);
    // "other.norm.weight" is not in the map and suffix search is disabled.
    let result = try_exact_then_suffix(&idx, "other.norm.weight", 0);
    assert!(
        result.is_none(),
        "absent name with suffix_segments=0 must yield None"
    );
}

/// suffix_segments=0 disables Phase 2 but Phase 1 (exact) still fires.
#[test]
fn suffix_match_zero_segments_phase1_still_fires() {
    let idx = make_idx(&[("model.norm.weight", "s.safetensors")]);
    // Exact hit — Phase 1 finds it even though suffix_segments=0.
    let result = try_exact_then_suffix(&idx, "model.norm.weight", 0);
    assert_eq!(
        result,
        Some("model.norm.weight"),
        "exact hit must succeed with suffix_segments=0"
    );
}

/// suffix matching requires a dot boundary — "xnorm.weight" must not
/// match a requested suffix of "norm.weight" because 'x' precedes 'norm' with
/// no intervening dot.
#[test]
fn suffix_match_no_false_positive_on_partial_word() {
    let idx = make_idx(&[("model.xnorm.weight", "s.safetensors")]);
    // "norm.weight" (suffix_segments=2) must NOT match "xnorm.weight" —
    // the character before 'n' in "xnorm" is 'x', not '.'.
    let result = try_exact_then_suffix(&idx, "model.norm.weight", 2);
    assert!(
        result.is_none(),
        "'xnorm.weight' must not match suffix 'norm.weight'"
    );
}

// Boundary tests for the safe indexing refactor.

/// Key that IS the entire suffix (prefix_len == 0) must match — the
/// split_at_checked path must treat a zero-length prefix as an automatic
/// boundary hit, not a false negative.
#[test]
fn suffix_match_key_equals_suffix_matches() {
    let idx = make_idx(&[("norm.weight", "s.safetensors")]);
    // suffix_segments=2 → suffix is "norm.weight"; key IS "norm.weight" (prefix_len=0).
    let result = try_exact_then_suffix(&idx, "norm.weight", 2);
    assert_eq!(
        result,
        Some("norm.weight"),
        "key equal to the full suffix must match (prefix_len == 0 boundary)"
    );
}

/// Verify that resolve_paro returns an empty state (not a panic) when the
/// weight_map contains no `.pairs` entries.
#[test]
fn resolve_paro_empty_paro_bases_returns_empty_state() {
    let idx = make_idx(&[
        ("layers.0.mlp.weight", "s.safetensors"),
        ("layers.0.mlp.scales", "s.safetensors"),
    ]);
    // krot_hint=Some(8) so no I/O; paro_bases will be empty → early return.
    let state = resolve_paro(&idx, std::path::Path::new("/nonexistent"), Some(8)).unwrap();
    assert_eq!(state.layer_count(), 0, "no .pairs → empty state");
}
