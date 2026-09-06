//! DFlash 2 config-parsing unit tests.
//!
//! The positive case is the published `z-lab/Qwen3.8-27B-DFlash2` config, kept
//! here verbatim so the negative cases below are one edit away from a config
//! that is known to work. `tests/dflash2_loader.rs` runs the same assertions
//! against the file on disk, which is what keeps this copy honest.

use super::*;

/// The published DFlash 2 drafter config, field for field.
const DFLASH2_CONFIG_JSON: &str = r#"{
  "architectures": ["DFlash2DraftModel"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": null,
  "is_causal": false,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248070,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61]
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 17408,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 262144,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 64,
  "pad_token_id": 248044,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000000,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.15.0",
  "use_cache": true,
  "use_sliding_window": true,
  "vocab_size": 248320
}"#;

/// The verifier this drafter is published against.
const VERIFIER_HIDDEN: usize = 5120;

#[allow(
    clippy::expect_used,
    reason = "test fixture: a JSON literal in this file that stops parsing is the assertion failing"
)]
fn model_config(json: &str) -> rmlx_loader::ModelConfig {
    serde_json::from_str(json).expect("fixture config.json parses")
}

/// Edit one key of the fixture and re-parse. `path` is `""` for a top-level
/// key or `"dflash_config"` for one inside that block; a `None` value removes
/// the key.
#[allow(
    clippy::expect_used,
    reason = "test fixture: the fixture is a JSON object with the block this helper edits"
)]
fn parsed_with(path: &str, key: &str, value: Option<serde_json::Value>) -> Result<DFlash2Config> {
    let mut root: serde_json::Value =
        serde_json::from_str(DFLASH2_CONFIG_JSON).expect("fixture parses");
    let obj = if path.is_empty() {
        root.as_object_mut().expect("root is an object")
    } else {
        root.get_mut(path)
            .and_then(serde_json::Value::as_object_mut)
            .expect("named block is an object")
    };
    match value {
        Some(v) => {
            obj.insert(key.to_owned(), v);
        }
        None => {
            obj.remove(key);
        }
    }
    parse_config(&model_config(&root.to_string()), VERIFIER_HIDDEN)
}

/// The published config parses to the values the checkpoint carries — every one
/// of them read, none defaulted.
///
/// `block_size` and `rope_theta` are the two this pins hardest: DFlash 1 keeps
/// both at the top level and DFlash 2 keeps them under `dflash_config` and
/// `rope_parameters`, so a loader reading DFlash 1's places off this file gets
/// no error, just whatever its defaults are.
#[test]
#[allow(
    clippy::expect_used,
    clippy::float_cmp,
    reason = "test assertions: the fixture is the published config and an Err or a moved value is the assertion failing"
)]
fn the_published_config_parses_to_the_checkpoints_own_values() {
    let cfg = parse_config(&model_config(DFLASH2_CONFIG_JSON), VERIFIER_HIDDEN)
        .expect("the published config must parse");
    assert_eq!(cfg.block_size, 8, "block_size is 8, not DFlash 1's 16");
    assert_eq!(cfg.rope_theta, 1.0e7);
    assert_eq!(cfg.conv_group_size, 16);
    assert_eq!(cfg.conv_kernel_size, 2);
    assert_eq!(cfg.selector_rank, 256);
    assert_eq!(cfg.selector_top_k, 16);
    assert_eq!(cfg.mask_token_id, 248_070);
    assert_eq!(cfg.target_layer_ids, vec![5, 19, 33, 47, 61]);
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.num_hidden_layers, 5);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.intermediate_size, 17408);
    assert_eq!(cfg.vocab_size, 248_320);
    assert_eq!(cfg.sliding_window, 2048);
    assert!(!cfg.is_causal, "the block is drafted bidirectionally");
}

/// A key the drafter needs and the checkpoint does not carry is refused by
/// name. None of these may be defaulted: a default is indistinguishable from
/// the checkpoint's own value once the run is recorded.
///
/// `block_size` heads the list because it is the one a DFlash 1 loader reads
/// from the top level and silently defaults to 16 on this file.
#[test]
fn a_missing_key_is_refused_by_name_and_never_defaulted() {
    let cases: &[(&str, &str)] = &[
        ("dflash_config", "block_size"),
        ("dflash_config", "conv_group_size"),
        ("dflash_config", "conv_kernel_size"),
        ("dflash_config", "selector_rank"),
        ("dflash_config", "selector_top_k"),
        ("dflash_config", "mask_token_id"),
        ("dflash_config", "target_layer_ids"),
        ("", "hidden_size"),
        ("", "num_hidden_layers"),
        ("", "num_attention_heads"),
        ("", "num_key_value_heads"),
        ("", "head_dim"),
        ("", "intermediate_size"),
        ("", "vocab_size"),
        ("", "rms_norm_eps"),
        ("", "sliding_window"),
        ("", "is_causal"),
        ("", "layer_types"),
    ];
    for (path, key) in cases {
        let err = match parsed_with(path, key, None) {
            Ok(cfg) => panic!("dropping {path}.{key} must refuse, parsed to {cfg:?}"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(key),
            "the refusal for a missing {path}.{key} must name it: {err}"
        );
    }
}

/// The RoPE base lives under `rope_parameters`, not at the top level, and a
/// top-level `rope_theta` is not a substitute: reading the wrong one is how a
/// drafter ends up positioned differently from the checkpoint it was trained
/// as, with no error anywhere.
#[test]
fn the_rope_base_is_read_from_rope_parameters_only() {
    let mut root: serde_json::Value = match serde_json::from_str(DFLASH2_CONFIG_JSON) {
        Ok(v) => v,
        Err(e) => panic!("fixture parses: {e}"),
    };
    // Only the base moves: `rope_parameters` stays, with its `rope_type`, so
    // nothing but the base's own read decides this.
    let Some(rope) = root
        .get_mut("rope_parameters")
        .and_then(serde_json::Value::as_object_mut)
    else {
        panic!("rope_parameters is an object")
    };
    rope.remove("rope_theta");
    let Some(obj) = root.as_object_mut() else {
        panic!("root is an object")
    };
    obj.insert("rope_theta".to_owned(), serde_json::json!(1.0e7));
    let err = match parse_config(&model_config(&root.to_string()), VERIFIER_HIDDEN) {
        Ok(cfg) => panic!("a top-level rope_theta must not stand in, parsed to {cfg:?}"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("rope_parameters.rope_theta"),
        "the refusal must name where the base is read from: {err}"
    );
}

/// A scaled RoPE is refused rather than applied as a plain one.
#[test]
fn a_scaled_rope_is_refused() {
    let err = match parsed_with(
        "rope_parameters",
        "rope_type",
        Some(serde_json::json!("yarn")),
    ) {
        Ok(cfg) => panic!("a yarn rope_type must refuse, parsed to {cfg:?}"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("yarn") && err.contains("rope_type"),
        "the refusal must name the scaling it will not apply: {err}"
    );
}

/// A snapshot that is not a DFlash 2 drafter is refused before any key is read,
/// naming what it does declare.
#[test]
fn another_architecture_is_refused_by_name() {
    for arch in ["DFlashDraftModel", "Qwen3ForCausalLM", ""] {
        let json = DFLASH2_CONFIG_JSON.replace("DFlash2DraftModel", arch);
        let err = match parse_config(&model_config(&json), VERIFIER_HIDDEN) {
            Ok(cfg) => panic!("{arch:?} must refuse, parsed to {cfg:?}"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(arch) || arch.is_empty(),
            "the refusal must name the architecture it was given: {err}"
        );
        assert!(
            err.contains("DFlash2DraftModel"),
            "the refusal must name the architecture it builds: {err}"
        );
    }
}

/// A config the loader can parse but the drafter cannot run is refused with the
/// property that fails, not with a tensor-shape error three hundred weights
/// later.
#[test]
fn a_config_the_forward_could_not_honour_is_refused() {
    let cases: &[(&str, &str, serde_json::Value, &str)] = &[
        // A three-tap kernel convolved with two taps drops a tap in silence.
        (
            "dflash_config",
            "conv_kernel_size",
            serde_json::json!(3),
            "conv_kernel_size",
        ),
        // A group size that does not divide the channels leaves a partial group
        // with no correction.
        (
            "dflash_config",
            "conv_group_size",
            serde_json::json!(48),
            "conv_group_size",
        ),
        (
            "dflash_config",
            "conv_group_size",
            serde_json::json!(0),
            "conv_group_size",
        ),
        // One candidate is no choice.
        (
            "dflash_config",
            "selector_top_k",
            serde_json::json!(1),
            "selector_top_k",
        ),
        // A block of one is the seed alone.
        (
            "dflash_config",
            "block_size",
            serde_json::json!(1),
            "block_size",
        ),
        // An empty target-layer list projects nothing.
        (
            "dflash_config",
            "target_layer_ids",
            serde_json::json!([]),
            "target_layer_ids",
        ),
        // A full-attention layer would be given the sliding mask.
        (
            "",
            "layer_types",
            serde_json::json!([
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "sliding_attention"
            ]),
            "layer_types",
        ),
        // A layer_types shorter than the stack describes a different model.
        (
            "",
            "layer_types",
            serde_json::json!(["sliding_attention"]),
            "layer_types",
        ),
    ];
    for (path, key, value, want) in cases {
        let err = match parsed_with(path, key, Some(value.clone())) {
            Ok(cfg) => panic!("{path}.{key} = {value} must refuse, parsed to {cfg:?}"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(want),
            "the refusal for {path}.{key} = {value} must name {want}: {err}"
        );
    }
}

/// A drafter of a different width than the verifier is refused: its `fc` reads
/// the verifier's hidden states directly.
#[test]
fn a_width_the_verifier_does_not_share_is_refused() {
    let err = match parse_config(&model_config(DFLASH2_CONFIG_JSON), 2048) {
        Ok(cfg) => panic!("a 2048-wide verifier must refuse, parsed to {cfg:?}"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("5120") && err.contains("2048"),
        "the refusal must name both widths: {err}"
    );
}
