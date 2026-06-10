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

// ----- view / view_discriminated tests -----------------------------------
//
// These exercise the discriminated tensor lookup against real on-disk shards.
// A valid shard is written with `safetensors::serialize`; a corrupt shard is a
// truncated/garbage header. The discrimination contract is:
//   Ok(Found)       — tensor located
//   Ok(NotInIndex)  — name absent from weight_map           → safe to fall back
//   Ok(WrongShard)  — index points at a shard lacking it    → safe to fall back
//   Err(...)        — shard header failed to parse (CORRUPT) → MUST propagate

/// Write a one-tensor `.safetensors` shard into `dir` and return its filename.
/// The tensor is a tiny F32 scalar — content is irrelevant, only the header.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O and serialization failures should abort the test loudly"
)]
fn write_valid_shard(dir: &std::path::Path, filename: &str, tensor_name: &str) {
    let data: [u8; 4] = 1.0f32.to_le_bytes();
    let tv = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &data).unwrap();
    let bytes = safetensors::serialize([(tensor_name.to_owned(), tv)], None).unwrap();
    std::fs::write(dir.join(filename), bytes).unwrap();
}

/// Write a corrupt `.safetensors` shard (bogus header) into `dir`.
/// The 8-byte little-endian length prefix claims a 16-byte header, but the
/// "header" bytes are not valid JSON — `SafeTensors::deserialize` must fail.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O failures should abort the test loudly"
)]
fn write_corrupt_shard(dir: &std::path::Path, filename: &str) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&16u64.to_le_bytes()); // header_len = 16
    bytes.extend_from_slice(b"not valid json!!"); // 16 bytes of garbage
    std::fs::write(dir.join(filename), bytes).unwrap();
}

/// Contract test: a corrupt shard header must propagate as `Err`, NOT be
/// reported as a not-found/fall-back signal. This is the discrimination the
/// split exists to prevent: a fallback caller that treats every lookup miss as
/// "scan elsewhere" would silently mask shard corruption.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn corrupt_header_propagates_as_err_not_notfound() {
    let dir = tempfile::tempdir().unwrap();
    write_corrupt_shard(dir.path(), "corrupt.safetensors");
    // Index claims the tensor lives in the corrupt shard.
    let idx = make_idx(&[("model.norm.weight", "corrupt.safetensors")]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();

    // view_discriminated: the corrupt header must surface as Err, never as
    // Ok(NotInIndex) / Ok(WrongShard).
    let disc = view_discriminated(&shards, &idx, "model.norm.weight");
    assert!(
        disc.is_err(),
        "corrupt header must be Err, got {:?} (would silently mask corruption on fall-back)",
        disc.as_ref().map(|_| "Ok")
    );
    let msg = disc.unwrap_err().to_string();
    assert!(
        msg.contains("parse") || msg.contains("header"),
        "error should name a header-parse failure: {msg}"
    );

    // view() convenience wrapper must likewise hard-error, not swallow.
    assert!(
        view(&shards, &idx, "model.norm.weight").is_err(),
        "view() must propagate the corrupt-header error"
    );
}

/// A name absent from `weight_map` discriminates as `NotInIndex` (safe to fall
/// back). The shard on disk is valid; the index simply has no entry.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn not_in_index_discriminated() {
    let dir = tempfile::tempdir().unwrap();
    write_valid_shard(dir.path(), "s.safetensors", "present.weight");
    let idx = make_idx(&[("present.weight", "s.safetensors")]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();

    match view_discriminated(&shards, &idx, "absent.weight").unwrap() {
        TensorLookup::NotInIndex => {}
        other @ (TensorLookup::Found(_) | TensorLookup::WrongShard) => {
            panic!("expected NotInIndex for an unmapped name, got {other:?}")
        }
    }
    // The convenience wrapper collapses NotInIndex into an Err.
    assert!(view(&shards, &idx, "absent.weight").is_err());
}

/// A name whose `weight_map` shard does not actually contain it discriminates
/// as `WrongShard` (the "index lies" / medgemma class — safe to fall back).
/// Two valid shards exist; the index points the tensor at the wrong one.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn wrong_shard_discriminated() {
    let dir = tempfile::tempdir().unwrap();
    // The tensor physically lives in s2, but the index claims it is in s1.
    write_valid_shard(dir.path(), "s1.safetensors", "decoy.weight");
    write_valid_shard(dir.path(), "s2.safetensors", "model.norm.weight");
    let idx = make_idx(&[
        ("decoy.weight", "s1.safetensors"),
        ("model.norm.weight", "s1.safetensors"), // lies: it is really in s2
    ]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();

    match view_discriminated(&shards, &idx, "model.norm.weight").unwrap() {
        TensorLookup::WrongShard => {}
        other @ (TensorLookup::Found(_) | TensorLookup::NotInIndex) => {
            panic!("expected WrongShard for an index-lies entry, got {other:?}")
        }
    }
    assert!(view(&shards, &idx, "model.norm.weight").is_err());
}

/// A correctly-indexed tensor discriminates as `Found` with intact shape/dtype.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn found_discriminated() {
    let dir = tempfile::tempdir().unwrap();
    write_valid_shard(dir.path(), "s.safetensors", "model.norm.weight");
    let idx = make_idx(&[("model.norm.weight", "s.safetensors")]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();

    match view_discriminated(&shards, &idx, "model.norm.weight").unwrap() {
        TensorLookup::Found(tv) => {
            assert_eq!(tv.name, "model.norm.weight");
            assert_eq!(tv.dtype, safetensors::Dtype::F32);
            assert_eq!(tv.shape, vec![1]);
            assert_eq!(tv.bytes.len(), 4);
        }
        other @ (TensorLookup::NotInIndex | TensorLookup::WrongShard) => {
            panic!("expected Found, got {other:?}")
        }
    }
    // The convenience wrapper returns the same view.
    let tv = view(&shards, &idx, "model.norm.weight").unwrap();
    assert_eq!(tv.shape, vec![1]);
}

/// An index entry that names a shard which is not open also discriminates as
/// `WrongShard` (the named handle is absent from the open set).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn unopened_shard_discriminated_as_wrong_shard() {
    let dir = tempfile::tempdir().unwrap();
    write_valid_shard(dir.path(), "s.safetensors", "present.weight");
    // Open the ShardSet over only the real shard...
    let open_idx = make_idx(&[("present.weight", "s.safetensors")]);
    let shards = ShardSet::open(dir.path(), &open_idx).unwrap();
    // ...but query against an index that points at a never-opened shard.
    let lying_idx = make_idx(&[("ghost.weight", "missing.safetensors")]);

    match view_discriminated(&shards, &lying_idx, "ghost.weight").unwrap() {
        TensorLookup::WrongShard => {}
        other @ (TensorLookup::Found(_) | TensorLookup::NotInIndex) => {
            panic!("expected WrongShard for an unopened-shard entry, got {other:?}")
        }
    }
}
