use super::*;

/// Parse a minimal `config.json` body into `ModelConfig` for the pre-flight
/// tests below. Only the `quantization` block matters for
/// `preflight_weight_quant`; every other field takes its serde default.
#[allow(clippy::unwrap_used)]
fn cfg_with_quant(json: &str) -> rmlx_loader::ModelConfig {
    serde_json::from_str(json).unwrap()
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
        msg.contains("supported: 2,3,4,5,6,8"),
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
