use super::*;

/// Parse a minimal `config.json` body into `ModelConfig` for the pre-flight
/// tests below. Only the `quantization` block matters for
/// `preflight_weight_quant`; every other field takes its serde default.
#[allow(clippy::unwrap_used)]
fn cfg_with_quant(json: &str) -> rmlx_loader::ModelConfig {
    serde_json::from_str(json).unwrap()
}

/// Same join logic `check_affine_bits` uses for the error message's
/// `supported: ...` list — derived from `SUPPORTED_BITS` rather than
/// hardcoded, so the assertion tracks the constant instead of duplicating it.
fn supported_bits_csv() -> String {
    rmlx_quant::affine::SUPPORTED_BITS
        .iter()
        .map(u8::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// Concrete regression case: `prism-ml__Bonsai-27B-mlx-1bit`
/// (`quantization.bits=1`, no `mode` -> defaults to affine). No dequant
/// kernel exists for 1-bit affine in this build's mlx-c; the pre-flight must
/// reject it with the exact, actionable message.
#[test]
#[allow(clippy::expect_used)]
fn rejects_1bit_affine_default_mode() {
    let cfg = cfg_with_quant(r#"{"quantization":{"group_size":128,"bits":1}}"#);
    let err = preflight_weight_quant(&cfg, "Qwen3_5ForConditionalGeneration")
        .expect_err("bits=1 affine must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("bits=1"),
        "error should name the offending bits: {msg}"
    );
    assert!(
        msg.contains("(affine)"),
        "error should name the quant mode: {msg}"
    );
    assert!(
        msg.contains(&format!("supported: {}", supported_bits_csv())),
        "error should list the supported set: {msg}"
    );
    assert!(
        msg.contains("mlx#3161"),
        "1-bit case should point at the tracking issue: {msg}"
    );
}

/// `prism-ml/Ternary-Bonsai-8B-mlx-2bit`-shaped config: `bits=2`, affine
/// default mode. Must load — this is a real, working production snapshot.
#[test]
fn accepts_2bit_affine() {
    let cfg = cfg_with_quant(r#"{"quantization":{"group_size":128,"bits":2}}"#);
    assert!(preflight_weight_quant(&cfg, "Qwen3ForCausalLM").is_ok());
}

/// Every affine bit-width this build's codec supports must pass.
#[test]
fn accepts_all_supported_affine_bits() {
    for bits in rmlx_quant::affine::SUPPORTED_BITS {
        let json =
            format!(r#"{{"quantization":{{"group_size":64,"bits":{bits},"mode":"affine"}}}}"#);
        let cfg = cfg_with_quant(&json);
        assert!(
            preflight_weight_quant(&cfg, "test").is_ok(),
            "bits={bits} is in SUPPORTED_BITS and must not be rejected"
        );
    }
}

/// An unsupported bits value is only checked under the affine mode — a
/// non-affine mode (mxfp8/mxfp4/nvfp4) has its own fixed-format kernel and
/// must not be false-rejected by the affine bit-width set.
#[test]
fn non_affine_mode_skips_the_affine_bits_check() {
    let cfg = cfg_with_quant(r#"{"quantization":{"group_size":32,"bits":7,"mode":"mxfp4"}}"#);
    assert!(preflight_weight_quant(&cfg, "Gemma4ForConditionalGeneration").is_ok());
}

/// `gemma-4-e4b-it-mxfp8`-shaped config: real production mxfp8 checkpoint.
/// Must load unchanged (no-regression guard for the concrete positive-proof
/// model list).
#[test]
fn accepts_real_mxfp8_shape() {
    let cfg = cfg_with_quant(r#"{"quantization":{"group_size":32,"bits":8,"mode":"mxfp8"}}"#);
    assert!(preflight_weight_quant(&cfg, "Gemma4ForConditionalGeneration").is_ok());
}

/// A model with no `quantization` block at all (bf16 / unquantized, or a
/// PARO checkpoint that only carries `quantization_config`) must not be
/// touched by this check.
#[test]
fn no_quantization_block_is_a_no_op() {
    let cfg = cfg_with_quant(r#"{"architectures":["Qwen3ForCausalLM"]}"#);
    assert!(preflight_weight_quant(&cfg, "Qwen3ForCausalLM").is_ok());
}

/// An unsupported bits value that is not the concrete 1-bit case gets the
/// same rejection shape, minus the 1-bit-specific hint.
#[test]
#[allow(clippy::expect_used)]
fn rejects_other_unsupported_bits_without_the_1bit_hint() {
    let cfg = cfg_with_quant(r#"{"quantization":{"group_size":64,"bits":7,"mode":"affine"}}"#);
    let err = preflight_weight_quant(&cfg, "test").expect_err("bits=7 affine must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("bits=7"), "{msg}");
    assert!(!msg.contains("mlx#3161"), "hint is 1-bit specific: {msg}");
}

/// Real-shaped regression case: `mlx-community__Qwen3.6-35B-A3B-8bit`-style
/// config carries a supported global `bits` (per-arch `resolve_quant` inherits
/// the global mode when an override omits its own `mode`, so a bare
/// `{"group_size":..,"bits":..}` override entry — the exact shape real
/// snapshots use — is affine by inheritance here too) alongside a
/// `tensor_overrides` entry with an unsupported affine `bits`. A global-only
/// check would false-accept this and defer the same load-then-die failure to
/// that tensor's first prefill; the override map must be scanned too.
#[test]
#[allow(clippy::expect_used)]
fn rejects_unsupported_affine_tensor_override_despite_supported_global_bits() {
    let cfg = cfg_with_quant(
        r#"{"quantization":{"group_size":64,"bits":4,"mode":"affine",
        "tensor_overrides":{"language_model.model.layers.0.mlp.gate":{"group_size":64,"bits":1}}}}"#,
    );
    let err = preflight_weight_quant(&cfg, "test")
        .expect_err("unsupported affine bits in a tensor_overrides entry must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("bits=1"), "{msg}");
    assert!(
        msg.contains("language_model.model.layers.0.mlp.gate"),
        "error should name the offending tensor: {msg}"
    );
}

/// A supported global bit-width with a *supported* tensor_overrides entry
/// must still load — the override scan must not false-reject a valid config
/// (no-regression guard for the override-walk added by finding 1).
#[test]
fn accepts_supported_affine_tensor_override() {
    let cfg = cfg_with_quant(
        r#"{"quantization":{"group_size":64,"bits":8,"mode":"affine",
        "tensor_overrides":{"language_model.model.layers.0.mlp.gate":{"group_size":64,"bits":8}}}}"#,
    );
    assert!(preflight_weight_quant(&cfg, "test").is_ok());
}

/// An unrecognized/future `mode` string paired with an out-of-affine-set
/// `bits` must be skipped (accepted), not resolved to `"affine"` and
/// false-rejected. Exercises the exact-string gate from finding 2 —
/// `QuantMode::from`'s "unknown -> Affine" fallback would wrongly reject
/// this if it were still used for the gate.
#[test]
fn unknown_mode_with_out_of_set_bits_is_not_false_rejected() {
    let cfg = cfg_with_quant(
        r#"{"quantization":{"group_size":32,"bits":7,"mode":"some_future_format"}}"#,
    );
    assert!(preflight_weight_quant(&cfg, "test").is_ok());
}
