use super::*;
use std::fs;
use tempfile::TempDir;

/// Build a minimal config.json in a temp dir with the given architectures array.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn write_config(dir: &TempDir, architectures: &[&str]) -> std::path::PathBuf {
    let archs: Vec<String> = architectures.iter().map(|s| format!("\"{s}\"")).collect();
    let json = format!(
        r#"{{
            "architectures": [{}],
            "dtype": "bfloat16"
        }}"#,
        archs.join(", ")
    );
    let path = dir.path().join("config.json");
    fs::write(&path, json).expect("write config.json");
    dir.path().to_path_buf()
}

// -- is_arch_supported predicate tests ------------------------------------

/// Every registered architecture is classified as generative or not, here,
/// against the registry as a set — so a new entry fails this test until
/// someone says which it is, rather than being inferred a full draft model
/// and dying inside `load_model`.
#[test]
fn every_registered_architecture_is_classified_as_generative_or_not() {
    let expected: &[(&str, bool)] = &[
        ("Gemma4ForConditionalGeneration", true),
        ("Gemma4UnifiedForConditionalGeneration", true),
        ("Gemma3ForConditionalGeneration", true),
        ("Qwen2ForCausalLM", true),
        ("Qwen3ForCausalLM", true),
        ("LagunaForCausalLM", true),
        ("Qwen3_5MoeForConditionalGeneration", true),
        ("Qwen3_5ForConditionalGeneration", true),
        ("Qwen3VLMoeForConditionalGeneration", true),
        ("BitNetForCausalLM", true),
        ("JinaEmbeddingsV4Model", false),
    ];
    let mut covered: Vec<&str> = expected.iter().map(|(class, _)| *class).collect();
    covered.sort_unstable();
    let mut registered: Vec<&str> = KNOWN_ARCHS.to_vec();
    registered.sort_unstable();
    assert_eq!(
        covered, registered,
        "this table and the architecture registry describe different sets"
    );
    for (class, generative) in expected {
        assert_eq!(
            is_generative_arch(class),
            *generative,
            "{class} is classified the wrong way"
        );
    }
    assert!(
        !is_generative_arch("NotRegisteredForCausalLM"),
        "an unregistered architecture is not generative either"
    );
}

#[test]
fn is_arch_supported_returns_true_for_known_archs() {
    // Spot-check a representative subset so any KNOWN_ARCHS typo is caught.
    let known = [
        "Gemma4ForConditionalGeneration",
        "Gemma3ForConditionalGeneration",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        "LagunaForCausalLM",
        "JinaEmbeddingsV4Model",
    ];
    for arch in &known {
        assert!(
            is_arch_supported(arch),
            "expected is_arch_supported({arch}) == true"
        );
    }
}

#[test]
fn is_arch_supported_returns_false_for_unknown_arch() {
    // XLMRobertaModel is the canonical negative test case (jina-embeddings-v3).
    assert!(
        !is_arch_supported("XLMRobertaModel"),
        "XLMRobertaModel must not be supported"
    );
    assert!(
        !is_arch_supported("FakeArchThatDoesNotExist"),
        "fictional arch must not be supported"
    );
    assert!(
        !is_arch_supported("(empty)"),
        "'(empty)' sentinel must not be supported"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn load_model_rejects_unknown_architecture() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = write_config(&dir, &["FakeArchXyz"]);

    let result = load_model(&model_path, Device::Cpu, &LoadOpts::default());
    match result {
        Err(Error::Model(msg)) => {
            assert!(
                msg.contains("FakeArchXyz"),
                "error message should mention the arch: {msg}"
            );
            assert!(
                msg.contains("not yet supported"),
                "error message should say not yet supported: {msg}"
            );
        }
        other => panic!("expected Err(Model(..)), got {other:?}"),
    }
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn load_model_rejects_empty_architectures() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = write_config(&dir, &[]);

    let result = load_model(&model_path, Device::Cpu, &LoadOpts::default());
    // Empty arch array -> arch_str = "(empty)" -> Model error.
    assert!(
        matches!(result, Err(Error::Model(_))),
        "expected Err(Model(..)) for empty arch, got {result:?}"
    );
}

/// A Prism Hadamard pack declares `hadamard_config`. Its weights need an
/// input-side Hadamard transform that no loader applies, so `load_model`
/// refuses it before the architecture check, whatever `architectures` says.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test fixture setup; a failed write is a broken test, not a result"
)]
fn load_model_refuses_a_hadamard_config_whatever_the_architecture() {
    for architectures in [r#"["Qwen3_5ForConditionalGeneration"]"#, "[]"] {
        let dir = TempDir::new().expect("tempdir");
        let json = format!(
            r#"{{"architectures": {architectures}, "hadamard_config": "hadamard.json",
                "quantization": {{"group_size": 128, "bits": 2, "mode": "affine"}}}}"#
        );
        fs::write(dir.path().join("config.json"), json).expect("write config.json");

        let msg = load_model(dir.path(), Device::Cpu, &LoadOpts::default())
            .err()
            .map_or_else(String::new, |e| e.to_string());
        assert!(msg.contains("hadamard_config"), "names the fact: {msg}");
        assert!(
            msg.contains("Hadamard transform"),
            "names the reason: {msg}"
        );
    }
}

/// Write `model.safetensors` holding one tensor, `model.embed_tokens.signs`.
#[allow(
    clippy::expect_used,
    reason = "test fixture setup; a failed write is a broken test, not a result"
)]
fn write_signs_shard(dir: &TempDir) {
    let signs = [0u8; 16];
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![4], &signs)
        .expect("tensor view");
    let bytes =
        safetensors::serialize([("model.embed_tokens.signs", view)], None).expect("serialize");
    fs::write(dir.path().join("model.safetensors"), bytes).expect("write shard");
}

/// `*.safetensors` files that cannot be parsed: an AppleDouble side file and a
/// truncated stray. Neither feeds weights, so neither changes the outcome.
#[allow(
    clippy::expect_used,
    reason = "test fixture setup; a failed write is a broken test, not a result"
)]
fn write_unparseable_strays(dir: &TempDir) {
    fs::write(dir.path().join("._model.safetensors"), [0u8, 5, 22, 7]).expect("write");
    fs::write(dir.path().join("stray.safetensors"), b"not a shard").expect("write");
}

/// Unparseable stray files beside a `.signs` shard: the pack is still refused
/// for its `.signs` tensor.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test fixture setup; a failed write is a broken test, not a result"
)]
fn a_signs_shard_is_refused_beside_unparseable_strays() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = write_config(&dir, &["Qwen3ForCausalLM"]);
    write_signs_shard(&dir);
    write_unparseable_strays(&dir);

    let msg = load_model(&model_path, Device::Cpu, &LoadOpts::default())
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(msg.contains("model.embed_tokens.signs"), "{msg}");
}

/// Unparseable stray files in a snapshot with no Hadamard fact do not fail the
/// check: the load reaches the architecture's own config error, as it did
/// before the check existed.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test fixture setup; a failed write is a broken test, not a result"
)]
fn unparseable_strays_do_not_fail_the_hadamard_check() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = write_config(&dir, &["Qwen3ForCausalLM"]);
    write_unparseable_strays(&dir);

    let msg = load_model(&model_path, Device::Cpu, &LoadOpts::default())
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(msg.contains("missing hidden_size"), "{msg}");
}

/// A pack whose config says nothing is still refused when a shard carries a
/// `.signs` tensor, the per-channel signs of the same Hadamard transform.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test fixture setup; a failed write is a broken test, not a result"
)]
fn load_model_refuses_a_signs_tensor_in_a_shard() {
    let dir = TempDir::new().expect("tempdir");
    let model_path = write_config(&dir, &["Qwen3ForCausalLM"]);
    write_signs_shard(&dir);

    let msg = load_model(&model_path, Device::Cpu, &LoadOpts::default())
        .err()
        .map_or_else(String::new, |e| e.to_string());
    assert!(
        msg.contains("model.embed_tokens.signs"),
        "names the tensor: {msg}"
    );
    assert!(
        msg.contains("Hadamard transform"),
        "names the reason: {msg}"
    );
}

/// LoadPhases struct must have all 5 fields initialised and the sum of
/// sub-phases must not exceed total_load_ms (modulo warmup rounding).
///
/// This test constructs a LoadPhases directly (no real model load required)
/// and asserts that the struct is correctly defined and that the invariant
/// `mmap_ms + dequant_ms + gpu_residency_ms + first_kernel_ready_ms <= total_load_ms`
/// holds when constructed with consistent values.
#[test]
fn load_phases_struct_fields_populated_and_sum_invariant() {
    let phases = LoadPhases {
        mmap_ms: 100,
        dequant_ms: 800,
        gpu_residency_ms: 0,
        first_kernel_ready_ms: 50,
        total_load_ms: 1000,
    };

    // All 5 fields readable.
    assert_eq!(phases.mmap_ms, 100);
    assert_eq!(phases.dequant_ms, 800);
    assert_eq!(phases.gpu_residency_ms, 0);
    assert_eq!(phases.first_kernel_ready_ms, 50);
    assert_eq!(phases.total_load_ms, 1000);

    // Sum-of-phases <= total (5% slack for unaccounted gap).
    let sub_total =
        phases.mmap_ms + phases.dequant_ms + phases.gpu_residency_ms + phases.first_kernel_ready_ms;
    let slack = phases.total_load_ms / 20; // 5%
    assert!(
        sub_total <= phases.total_load_ms + slack,
        "sum of sub-phases ({sub_total}) exceeds total_load_ms ({}) beyond 5% slack",
        phases.total_load_ms
    );
}

/// `read_load_phases` returns `None` before any successful load.
///
/// Note: this test relies on `LAST_LOAD_PHASES` being zero-initialised at
/// process start. Running it after a model load in the same process may
/// see `Some(...)` -- that is correct behaviour. The test is authored for
/// the cold-start case (first test run, or when no real model load precedes).
#[test]
fn read_load_phases_returns_none_when_all_zero() {
    // If a prior test in the same process loaded a real model, skip.
    let guard = phases::LAST_LOAD_PHASES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.total_load_ms > 0 {
        // A real model was loaded before this test -- skip to avoid false failure.
        return;
    }
    drop(guard);
    // No model loaded yet: should be None.
    assert!(
        read_load_phases().is_none(),
        "read_load_phases() should return None when total_load_ms == 0"
    );
}
