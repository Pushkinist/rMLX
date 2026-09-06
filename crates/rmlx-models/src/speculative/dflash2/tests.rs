//! DFlash 2 config-parsing and loader unit tests.
//!
//! The positive case is the published `z-lab/Qwen3.8-27B-DFlash2` config, kept
//! here verbatim so the negative cases below are one edit away from a config
//! that is known to work. `tests/dflash2_loader.rs` runs the same assertions
//! against the file on disk, which is what keeps this copy honest.
//!
//! The loader half runs on a scale-model snapshot written to a temp dir — the
//! same tensor names at a width small enough to write. It exists for the one
//! property the real checkpoint cannot show: the real checkpoint carries no
//! tensor the loader fails to read, so on it the unread-tensor refusal returns
//! the same answer whether it is wired or deleted.

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
        // More candidates than the vocabulary holds: the partition the selector
        // takes them with has no such rank to keep.
        (
            "dflash_config",
            "selector_top_k",
            serde_json::json!(248_321),
            "selector_top_k",
        ),
        // A KV head count that does not divide the query heads cannot be
        // repeated a whole number of times, and no tensor shape shows it.
        (
            "",
            "num_key_value_heads",
            serde_json::json!(7),
            "num_key_value_heads",
        ),
        (
            "",
            "num_key_value_heads",
            serde_json::json!(0),
            "num_key_value_heads",
        ),
        // A block of one is the seed alone.
        (
            "dflash_config",
            "block_size",
            serde_json::json!(1),
            "block_size",
        ),
        // A block bigger than one verify forward can score sizes the round's
        // token buffer, the verify input and the drafter's mask before anything
        // can refuse it, and the pass it describes would time the GPU out.
        (
            "dflash_config",
            "block_size",
            serde_json::json!(u64::from(u32::MAX)),
            "block_size",
        ),
        // A mask token outside the vocabulary is embedded from a clamped row of
        // the verifier's table, and every drafted position but the seed is it.
        (
            "dflash_config",
            "mask_token_id",
            serde_json::json!(248_320),
            "mask_token_id",
        ),
        // A window that reaches back past no conditioning row at all leaves the
        // forward's window arithmetic below zero.
        ("", "sliding_window", serde_json::json!(1), "sliding_window"),
        ("", "sliding_window", serde_json::json!(0), "sliding_window"),
        // A window wider than an array axis wraps negative in the conditioning
        // trim, which then slices from past its own end and returns nothing.
        (
            "",
            "sliding_window",
            serde_json::json!(u64::from(u32::MAX)),
            "sliding_window",
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

/// Each size bound accepts the last value it is meant to and refuses the first
/// it is not.
///
/// Every refusal above asserts only that a bad value is rejected, which a guard
/// that rejects *everything* also satisfies — and a size ceiling is exactly the
/// shape that goes one-sided, because the value that proves it is the one nobody
/// writes in a config by hand. So each bound is walked from both sides at its own
/// boundary.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a refusal of a value the drafter can run is the assertion failing"
)]
fn each_size_bound_accepts_its_boundary_and_refuses_one_past_it() {
    let axis = u64::try_from(i32::MAX).expect("i32::MAX is a u64");
    let block = round::MAX_BLOCK_SIZE as u64;
    let cases: &[(&str, &str, u64, u64)] = &[
        // (path, key, largest accepted, first refused)
        ("dflash_config", "block_size", block, block + 1),
        ("", "sliding_window", axis, axis + 1),
        // A token id is an index, so the last one the vocabulary holds is the
        // last one accepted.
        ("dflash_config", "mask_token_id", 248_319, 248_320),
        ("dflash_config", "selector_top_k", 248_320, 248_321),
    ];
    for &(path, key, accepted, refused) in cases {
        parsed_with(path, key, Some(serde_json::json!(accepted)))
            .unwrap_or_else(|e| panic!("{path}.{key} = {accepted} must still parse: {e}"));
        let err = match parsed_with(path, key, Some(serde_json::json!(refused))) {
            Ok(cfg) => panic!("{path}.{key} = {refused} must refuse, parsed to {cfg:?}"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(key),
            "the refusal for {path}.{key} = {refused} must name the key: {err}"
        );
    }
}

/// A causal checkpoint is refused, and a bidirectional one is not.
///
/// The forward's mask is unconditionally bidirectional. The reference branches
/// on this flag — it ands the block term with `key <= query` when it is set — so
/// a causal checkpoint run through this drafter would be denoised the wrong way
/// round and still produce fluent proposals, at an accept rate nothing
/// downstream can attribute. Refusing it is the whole of this drafter's answer
/// to that flag, which is why the flag has to be read for something.
///
/// Both directions are asserted. Refusing the true case alone is satisfied by a
/// guard of the wrong polarity, which would refuse the published checkpoint.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a refusal of the published checkpoint's own value is the assertion failing"
)]
fn a_causal_checkpoint_is_refused_and_a_bidirectional_one_is_not() {
    let err = match parsed_with("", "is_causal", Some(serde_json::json!(true))) {
        Ok(cfg) => panic!("a causal checkpoint must refuse, parsed to {cfg:?}"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("is_causal"), "names the key: {err}");
    assert!(
        err.contains("bidirectionally"),
        "names the direction the forward actually denoises in, so a refusal for \
         some other reason cannot satisfy this: {err}"
    );

    let cfg = parsed_with("", "is_causal", Some(serde_json::json!(false)))
        .expect("the published checkpoint's own value is the one this drafter runs");
    assert!(!cfg.is_causal);
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

/// The three scalars the reference applies to the drafter's logit path are
/// refused when a checkpoint moves one of them off the reference's default.
///
/// Every other refusal in this file fires on a key that is **absent**. These
/// three are the opposite failure: the published checkpoint declares none of
/// them, so a loader that ignores them is indistinguishable from one that
/// applies them, and a later checkpoint that sets one would be drafted through a
/// differently scaled head with nothing in the run to say so.
#[test]
fn a_logit_path_scalar_off_its_reference_default_is_refused() {
    let cases: &[(&str, &str, serde_json::Value)] = &[
        (
            "dflash_config",
            "input_embedding_scale",
            serde_json::json!(11.3137),
        ),
        ("dflash_config", "output_multiplier", serde_json::json!(2.5)),
        (
            "dflash_config",
            "final_logit_softcapping",
            serde_json::json!(30.0),
        ),
        // The reference falls back to the top level for the cap alone, so a
        // guard reading only `dflash_config` would miss this one.
        ("", "final_logit_softcapping", serde_json::json!(30.0)),
    ];
    for (path, key, value) in cases {
        let err = match parsed_with(path, key, Some(value.clone())) {
            Ok(cfg) => panic!("{path}.{key} = {value} must refuse, parsed to {cfg:?}"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(key),
            "the refusal for {path}.{key} = {value} must name the key: {err}"
        );
    }
}

/// A scalar declared **at** the reference's default is not a refusal: the
/// drafter's path is that default, so the two agree.
///
/// Without this the guard could be "refuse whenever the key is present", which
/// would turn a checkpoint that merely spells out its defaults into a load
/// failure.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a refusal of a config that agrees with the port is the assertion failing"
)]
fn a_logit_path_scalar_at_its_reference_default_is_accepted() {
    for (path, key, value) in [
        (
            "dflash_config",
            "input_embedding_scale",
            serde_json::json!(1.0),
        ),
        ("dflash_config", "output_multiplier", serde_json::json!(1.0)),
        (
            "dflash_config",
            "final_logit_softcapping",
            serde_json::Value::Null,
        ),
        ("", "final_logit_softcapping", serde_json::Value::Null),
    ] {
        parsed_with(path, key, Some(value.clone()))
            .unwrap_or_else(|e| panic!("{path}.{key} = {value} is the port's own path: {e}"));
    }
}

// --- the loader, on a scale model of the snapshot ---

/// A DFlash 2 config at a width small enough to write a whole snapshot for.
/// Same keys, same relationships: `head_dim * num_attention_heads` is the
/// hidden size, `conv_group_size` divides it, `layer_types` matches the stack.
const SCALE_CONFIG_JSON: &str = r#"{
  "architectures": ["DFlash2DraftModel"],
  "is_causal": false,
  "dflash_config": {
    "block_size": 4,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 39,
    "selector_rank": 8,
    "selector_top_k": 4,
    "target_layer_ids": [0, 1]
  },
  "head_dim": 16,
  "hidden_size": 64,
  "intermediate_size": 32,
  "layer_types": ["sliding_attention"],
  "model_type": "qwen3",
  "num_attention_heads": 4,
  "num_hidden_layers": 1,
  "num_key_value_heads": 2,
  "rms_norm_eps": 1e-06,
  "rope_parameters": { "rope_theta": 10000000, "rope_type": "default" },
  "sliding_window": 64,
  "vocab_size": 40
}"#;

const SCALE_HIDDEN: usize = 64;

/// Every tensor the scale-model snapshot ships, at the shape the config above
/// predicts.
fn scale_tensors() -> Vec<(String, Vec<usize>)> {
    let mut t: Vec<(String, Vec<usize>)> = vec![
        ("fc.weight".to_owned(), vec![64, 128]),
        ("hidden_norm.weight".to_owned(), vec![64]),
        ("norm.weight".to_owned(), vec![64]),
        (
            "candidate_selector.hidden_projection.weight".to_owned(),
            vec![8, 64],
        ),
        (
            "candidate_selector.predecessor_codebook".to_owned(),
            vec![40, 8],
        ),
        (
            "candidate_selector.successor_codebook".to_owned(),
            vec![40, 8],
        ),
    ];
    for (name, shape) in [
        ("input_layernorm.weight", vec![64]),
        ("post_attention_layernorm.weight", vec![64]),
        ("self_attn.q_proj.weight", vec![64, 64]),
        ("self_attn.k_proj.weight", vec![32, 64]),
        ("self_attn.v_proj.weight", vec![32, 64]),
        ("self_attn.o_proj.weight", vec![64, 64]),
        ("self_attn.q_norm.weight", vec![16]),
        ("self_attn.k_norm.weight", vec![16]),
        ("mlp.gate_proj.weight", vec![32, 64]),
        ("mlp.up_proj.weight", vec![32, 64]),
        ("mlp.down_proj.weight", vec![64, 32]),
        ("attention_conv.base_kernel", vec![2, 2, 64]),
        ("attention_conv.kernel_projection.weight", vec![16, 64]),
        ("mlp_conv.base_kernel", vec![2, 2, 64]),
        ("mlp_conv.kernel_projection.weight", vec![16, 64]),
    ] {
        t.push((format!("layers.0.{name}"), shape));
    }
    t
}

/// Serialise a zero-filled bf16 safetensors file. Offsets are assigned in the
/// order the header serialises in, which the format requires to be contiguous.
fn safetensors_bytes(tensors: &[(String, Vec<usize>)]) -> Vec<u8> {
    let mut sorted: Vec<&(String, Vec<usize>)> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape) in sorted {
        let bytes = shape.iter().product::<usize>() * 2;
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + bytes],
            }),
        );
        offset += bytes;
    }
    let hdr = serde_json::Value::Object(header).to_string();

    let mut out = Vec::with_capacity(8 + hdr.len() + offset);
    out.extend_from_slice(&(hdr.len() as u64).to_le_bytes());
    out.extend_from_slice(hdr.as_bytes());
    out.resize(out.len() + offset, 0);
    out
}

/// Write a snapshot directory holding the given tensors.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a temp dir that cannot be written is the harness failing, not the code under test"
)]
fn write_snapshot(tensors: &[(String, Vec<usize>)]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), SCALE_CONFIG_JSON).expect("config.json");
    std::fs::write(
        dir.path().join("model.safetensors"),
        safetensors_bytes(tensors),
    )
    .expect("model.safetensors");
    dir
}

/// The loader binds every tensor of a whole snapshot and reports the config it
/// read. This is the control for the refusal below: without it, a loader that
/// refused everything would pass that test.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertions: the fixture is a whole snapshot and an Err here is the assertion failing"
)]
fn a_whole_snapshot_loads_and_reports_what_it_read() {
    let dir = write_snapshot(&scale_tensors());
    let drafter = DFlash2Drafter::load(dir.path(), SCALE_HIDDEN, Device::Cpu)
        .expect("a snapshot carrying every tensor must load");
    assert_eq!(drafter.cfg.block_size, 4);
    assert_eq!(drafter.cfg.selector_rank, 8);
    assert_eq!(drafter.layers.len(), 1);
    assert_eq!(
        drafter.selector.predecessor_codebook.shape(),
        vec![40, 8],
        "the codebook binds under its bare name"
    );
    let layer = drafter.layers.first().expect("one layer");
    assert_eq!(layer.attention_conv.base_kernel.shape(), vec![2, 2, 64]);
    assert_eq!(layer.mlp_conv.base_kernel.shape(), vec![2, 2, 64]);
}

/// A snapshot carrying a tensor this loader does not read is refused, naming
/// it.
///
/// The published checkpoint cannot show this: the loader reads all 81 of its
/// tensors, so the refusal answers the same whether it is wired or deleted.
/// A DFlash generation past this one is exactly the case it exists for.
#[test]
fn a_snapshot_with_a_tensor_this_loader_cannot_build_is_refused() {
    let mut tensors = scale_tensors();
    tensors.push((
        "layers.0.self_attn.gate_proj.weight".to_owned(),
        vec![32, 64],
    ));
    let dir = write_snapshot(&tensors);

    let err = match DFlash2Drafter::load(dir.path(), SCALE_HIDDEN, Device::Cpu) {
        Ok(_) => panic!("a snapshot carrying an unread tensor must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("DFlash2Drafter"),
        "the refusal names the loader that issued it: {err}"
    );
    assert!(
        err.contains("layers.0.self_attn.gate_proj.weight"),
        "the refusal names the tensor it cannot build: {err}"
    );
}

/// A tensor missing from the snapshot is refused by name rather than skipped.
#[test]
fn a_missing_tensor_is_refused_by_name() {
    let tensors: Vec<(String, Vec<usize>)> = scale_tensors()
        .into_iter()
        .filter(|(n, _)| n != "candidate_selector.successor_codebook")
        .collect();
    let dir = write_snapshot(&tensors);

    let err = match DFlash2Drafter::load(dir.path(), SCALE_HIDDEN, Device::Cpu) {
        Ok(_) => panic!("a snapshot missing a tensor must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("candidate_selector.successor_codebook"),
        "the refusal names the tensor it could not find: {err}"
    );
}

/// A tensor present under the right name at the wrong shape is refused, naming
/// both shapes.
#[test]
fn a_tensor_at_the_wrong_shape_is_refused_naming_both() {
    let tensors: Vec<(String, Vec<usize>)> = scale_tensors()
        .into_iter()
        .map(|(n, s)| {
            if n == "layers.0.attention_conv.kernel_projection.weight" {
                (n, vec![8, 64])
            } else {
                (n, s)
            }
        })
        .collect();
    let dir = write_snapshot(&tensors);

    let err = match DFlash2Drafter::load(dir.path(), SCALE_HIDDEN, Device::Cpu) {
        Ok(_) => panic!("a tensor at the wrong shape must be refused"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("[8, 64]") && err.contains("[16, 64]"),
        "the refusal names the shape it found and the shape it predicted: {err}"
    );
}
