use super::*;

// --- split_model_path ---

#[test]
fn split_model_path_mlx_community_no_trailing_slash() {
    let (ns, model) =
        split_model_path("/opt/open-models/mlx-community__gemma-4-e2b-it-mxfp8").unwrap();
    assert_eq!(ns, "mlx-community");
    assert_eq!(model, "gemma-4-e2b-it-mxfp8");
}

#[test]
fn split_model_path_mlx_community_trailing_slash() {
    let (ns, model) =
        split_model_path("/opt/open-models/mlx-community__Qwen3.6-35B-A3B-8bit/").unwrap();
    assert_eq!(ns, "mlx-community");
    assert_eq!(model, "Qwen3.6-35B-A3B-8bit");
}

#[test]
fn split_model_path_z_lab() {
    let (ns, model) = split_model_path("/opt/open-models/z-lab__Qwen3.6-27B-PARO").unwrap();
    assert_eq!(ns, "z-lab");
    assert_eq!(model, "Qwen3.6-27B-PARO");
}

#[test]
fn split_model_path_prism_ml() {
    let (ns, model) =
        split_model_path("/opt/open-models/prism-ml__Ternary-Bonsai-8B-mlx-2bit").unwrap();
    assert_eq!(ns, "prism-ml");
    assert_eq!(model, "Ternary-Bonsai-8B-mlx-2bit");
}

#[test]
fn split_model_path_ollama_tag() {
    let (ns, model) = split_model_path("llama3.2:3b").unwrap();
    assert_eq!(ns, "ollama");
    assert_eq!(model, "llama3.2:3b");
}

#[test]
fn split_model_path_hf_id() {
    let (ns, model) = split_model_path("meta-llama/Llama-3.2-3B-Instruct").unwrap();
    assert_eq!(ns, "hf");
    assert_eq!(model, "meta-llama/Llama-3.2-3B-Instruct");
}

#[test]
fn split_model_path_local_finetune() {
    let (ns, model) = split_model_path("/opt/open-models/my-finetune-v1").unwrap();
    assert_eq!(ns, "local");
    assert_eq!(model, "my-finetune-v1");
}

#[test]
fn split_model_path_unknown_namespace_errors() {
    let err = split_model_path("/x/foo__bar").unwrap_err();
    match err {
        Error::IdentityNotInWhitelist { field, value, .. } => {
            assert_eq!(field, "model_namespace");
            assert_eq!(value, "foo");
        }
        Error::MissingBackendVersion { .. }
        | Error::Sqlite(_)
        | Error::Io(_)
        | Error::Schema(_)
        | Error::IdentityModelPath(_)
        | Error::UnknownMetric(_)
        | Error::UnknownDirection(_)
        | Error::InvalidTimestamp(_)
        | Error::InvalidPrompt(_)
        | Error::NoMeasurements
        | Error::InvalidIngestField { .. }
        | Error::Recorder(_)
        | Error::Query(_)
        | Error::Scope(_) => panic!("unexpected error: {err}"),
    }
}

// --- canonicalize ---

#[test]
fn canonicalize_backend_lowercase() {
    let result = canonicalize("backend", "RMLX", BACKEND_WHITELIST).unwrap();
    assert_eq!(result, "rmlx");
}

#[test]
fn canonicalize_backend_unknown() {
    let err = canonicalize("backend", "pytorch", BACKEND_WHITELIST).unwrap_err();
    match err {
        Error::IdentityNotInWhitelist { field, value, .. } => {
            assert_eq!(field, "backend");
            assert_eq!(value, "pytorch");
        }
        Error::MissingBackendVersion { .. }
        | Error::Sqlite(_)
        | Error::Io(_)
        | Error::Schema(_)
        | Error::IdentityModelPath(_)
        | Error::UnknownMetric(_)
        | Error::UnknownDirection(_)
        | Error::InvalidTimestamp(_)
        | Error::InvalidPrompt(_)
        | Error::NoMeasurements
        | Error::InvalidIngestField { .. }
        | Error::Recorder(_)
        | Error::Query(_)
        | Error::Scope(_) => panic!("unexpected error: {err}"),
    }
}

#[test]
fn canonicalize_backend_llama_cpp_canonical() {
    let result = canonicalize("backend", "llama_cpp", BACKEND_WHITELIST).unwrap();
    assert_eq!(result, "llama_cpp");
}

#[test]
fn canonicalize_backend_llama_cpp_dot_alias() {
    let result = canonicalize("backend", "llama.cpp", BACKEND_WHITELIST).unwrap();
    assert_eq!(result, "llama_cpp");
}

#[test]
fn canonicalize_backend_llama_cpp_dash_alias() {
    let result = canonicalize("backend", "llama-cpp", BACKEND_WHITELIST).unwrap();
    assert_eq!(result, "llama_cpp");
}

#[test]
fn canonicalize_backend_llamacpp_alias() {
    let result = canonicalize("backend", "llamacpp", BACKEND_WHITELIST).unwrap();
    assert_eq!(result, "llama_cpp");
}

#[test]
fn alias_normalization_non_backend_field_untouched() {
    // "llama.cpp" is NOT a valid weight_quant — alias only fires for backend field.
    let err = canonicalize("weight_quant", "llama.cpp", WEIGHT_QUANT_WHITELIST).unwrap_err();
    match err {
        Error::IdentityNotInWhitelist { field, .. } => assert_eq!(field, "weight_quant"),
        Error::MissingBackendVersion { .. }
        | Error::Sqlite(_)
        | Error::Io(_)
        | Error::Schema(_)
        | Error::IdentityModelPath(_)
        | Error::UnknownMetric(_)
        | Error::UnknownDirection(_)
        | Error::InvalidTimestamp(_)
        | Error::InvalidPrompt(_)
        | Error::NoMeasurements
        | Error::InvalidIngestField { .. }
        | Error::Recorder(_)
        | Error::Query(_)
        | Error::Scope(_) => panic!("unexpected error: {err}"),
    }
}

#[test]
fn canonicalize_weight_quant_passthrough_lowercase() {
    let result = canonicalize("weight_quant", "MXFP8", WEIGHT_QUANT_WHITELIST).unwrap();
    assert_eq!(result, "mxfp8");
}

// --- canonicalize_kv_quant (permissive: free-form label, no whitelist) ---

#[test]
fn canonicalize_kv_quant_none_passes_through() {
    assert_eq!(canonicalize_kv_quant("none").unwrap(), "none");
}

#[test]
fn canonicalize_kv_quant_bf16_alias_canonicalizes_to_none() {
    assert_eq!(canonicalize_kv_quant("bf16").unwrap(), "none");
    assert_eq!(canonicalize_kv_quant("f16").unwrap(), "none");
}

#[test]
fn canonicalize_kv_quant_rotor_v_alias_canonicalizes() {
    assert_eq!(canonicalize_kv_quant("rotor_v_3").unwrap(), "rotor3");
    assert_eq!(canonicalize_kv_quant("rotor_v_4").unwrap(), "rotor4");
}

#[test]
fn canonicalize_kv_quant_k8v4_k8v8_planar() {
    assert_eq!(canonicalize_kv_quant("k8v4").unwrap(), "k8v4");
    assert_eq!(canonicalize_kv_quant("k8v8").unwrap(), "k8v8");
    assert_eq!(canonicalize_kv_quant("planar").unwrap(), "planar");
}

#[test]
fn canonicalize_kv_quant_legacy_turbo() {
    assert_eq!(canonicalize_kv_quant("turbo4").unwrap(), "turbo4");
    assert_eq!(canonicalize_kv_quant("turbo8").unwrap(), "turbo8");
}

/// A couple of real rotation-family codec names — the exact class the
/// old hand-maintained allow-list dropped — record fine, verbatim.
#[test]
fn canonicalize_kv_quant_real_rotation_codec_names_pass_through() {
    assert_eq!(canonicalize_kv_quant("rotor4_sym").unwrap(), "rotor4_sym");
    assert_eq!(canonicalize_kv_quant("k_rotor3").unwrap(), "k_rotor3");
}

/// The core of the fix: an arbitrary token this binary has never heard of
/// — a codec that does not exist yet, a typo, anything — is recorded
/// verbatim (lowercased) instead of rejected. No allow-list, no drift.
#[test]
fn canonicalize_kv_quant_unknown_token_records_verbatim() {
    assert_eq!(
        canonicalize_kv_quant("some_future_codec_v9").unwrap(),
        "some_future_codec_v9"
    );
    assert_eq!(
        canonicalize_kv_quant("Some_Future_Codec_V9").unwrap(),
        "some_future_codec_v9"
    );
    assert_eq!(canonicalize_kv_quant("kx99").unwrap(), "kx99");
}

#[test]
fn canonicalize_kv_quant_lowercases_and_trims() {
    assert_eq!(canonicalize_kv_quant("  K8V4  ").unwrap(), "k8v4");
}

#[test]
fn canonicalize_kv_quant_mixed_long_form_passes_through() {
    assert_eq!(
        canonicalize_kv_quant("mixed_k8g128_v4g64").unwrap(),
        "mixed_k8g128_v4g64"
    );
    assert_eq!(
        canonicalize_kv_quant("MIXED_K8G64_V8G64").unwrap(),
        "mixed_k8g64_v8g64"
    );
}

// --- mlx_nax ---
//
// Asserted against `mlx_nax_or_unknown()` directly, not `RunIdentity::get()`:
// `IDENTITY` is a process-wide `OnceLock` shared with every other test in
// this binary, so whichever test runs first decides its cached value for
// the rest of the run. `MLX_NAX` has no such contention — nothing else in
// this crate ever calls `set_mlx_nax`, so this test is its only writer
// regardless of execution order.

#[test]
fn mlx_nax_defaults_to_unknown_then_takes_the_first_set_value() {
    assert_eq!(
        mlx_nax_or_unknown(),
        "unknown",
        "no caller has set it yet in this process"
    );
    set_mlx_nax("present");
    assert_eq!(mlx_nax_or_unknown(), "present");
    // First writer wins — a later call must not override it.
    set_mlx_nax("absent");
    assert_eq!(mlx_nax_or_unknown(), "present");
}

/// Previously-"malformed" `mixed_*` shapes no longer error — they are just
/// another free-form label now, recorded verbatim.
#[test]
fn canonicalize_kv_quant_malformed_mixed_records_verbatim() {
    assert_eq!(
        canonicalize_kv_quant("mixed_garbage").unwrap(),
        "mixed_garbage"
    );
    assert_eq!(canonicalize_kv_quant("mixed_k8_v4").unwrap(), "mixed_k8_v4");
}
