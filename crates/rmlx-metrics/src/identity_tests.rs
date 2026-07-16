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

#[test]
fn canonicalize_kv_quant_unknown_errors() {
    let err = canonicalize_kv_quant("kx99").unwrap_err();
    match err {
        Error::IdentityNotInWhitelist { field, value, .. } => {
            assert_eq!(field, "kv_quant");
            assert_eq!(value, "kx99");
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

// --- canonicalize_kv_quant ---

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

#[test]
fn canonicalize_kv_quant_mixed_long_form() {
    assert_eq!(
        canonicalize_kv_quant("mixed_k8g128_v4g64").unwrap(),
        "mixed_k8g128_v4g64"
    );
    assert_eq!(
        canonicalize_kv_quant("MIXED_K8G64_V8G64").unwrap(),
        "mixed_k8g64_v8g64"
    );
}

#[test]
fn canonicalize_kv_quant_malformed_mixed_errors() {
    assert!(canonicalize_kv_quant("mixed_garbage").is_err());
    assert!(canonicalize_kv_quant("mixed_k8_v4").is_err());
    assert!(canonicalize_kv_quant("mixed_x8g64_v4g64").is_err());
}

/// Every unit-variant `<KvQuant as Display>` token accepted without
/// rejection (issue: the drainer allow-list was a stale hand-maintained
/// mirror missing the rotation/sym/planar/turbo families). One entry per
/// unit variant declared in `crates/rmlx-kv-quant/src/quant.rs`, excluding
/// the four payload-bearing variants (`Mixed`, `RotK`, `RotorK3Asym`,
/// `RotorK4Asym`) covered separately below.
#[test]
fn canonicalize_kv_quant_accepts_all_unit_variant_tokens() {
    let tokens = [
        "none",
        "k8v4",
        "k8v8",
        "planar",
        "planar3",
        "planar_k",
        "k8vturbo3",
        "k8vturbo3tcq",
        "k8vturbo2",
        "k8vturbo2tcq",
        "tsym3",
        "tsym4",
        "iso3",
        "iso4",
        "iso3_sym",
        "iso4_sym",
        "k_iso3",
        "k_iso4",
        "rotor3",
        "rotor4",
        "rotor3_sym",
        "rotor4_sym",
        "k_rotor3",
        "k_rotor4",
        "rot_k_tq4v",
    ];
    assert_eq!(
        tokens.len(),
        25,
        "keep in sync with the 25 unit KvQuant variants"
    );
    for token in tokens {
        assert_eq!(
            canonicalize_kv_quant(token).unwrap_or_else(|e| panic!("{token} rejected: {e}")),
            token,
            "token {token} did not canonicalize to itself"
        );
    }
}

/// The four payload-bearing `KvQuant` Display forms parse structurally
/// (bits/group digits), mirroring `RotK` / `RotorK3Asym` / `RotorK4Asym`
/// which the pre-fix allow-list did not accept at all.
#[test]
fn canonicalize_kv_quant_accepts_payload_variant_samples() {
    for token in [
        "mixed_k8g128_v4g64",
        "rot_k_v4g64",
        "rot_k_v8g128",
        "rotor_k_3_asym_v4_g128",
        "rotor_k_4_asym_v3_g64",
    ] {
        assert_eq!(
            canonicalize_kv_quant(token).unwrap_or_else(|e| panic!("{token} rejected: {e}")),
            token
        );
    }
}

#[test]
fn canonicalize_kv_quant_rotor_v_alias_canonicalizes() {
    assert_eq!(canonicalize_kv_quant("rotor_v_3").unwrap(), "rotor3");
    assert_eq!(canonicalize_kv_quant("rotor_v_4").unwrap(), "rotor4");
}

#[test]
fn canonicalize_kv_quant_malformed_rot_k_and_rotor_k_asym_errors() {
    assert!(canonicalize_kv_quant("rot_k_v4").is_err());
    assert!(canonicalize_kv_quant("rot_k_vxg64").is_err());
    assert!(canonicalize_kv_quant("rotor_k_5_asym_v4_g128").is_err());
    assert!(canonicalize_kv_quant("rotor_k_3_asym_v4g128").is_err());
}
