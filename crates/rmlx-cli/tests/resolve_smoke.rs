//! Integration smoke test for the sibling-tensor resolver + ShardHandle mmap.
//!
//! Skips gracefully if the primary test snapshot is absent — never fails CI
//! on a developer who doesn't have the model locally.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp
)]

use rmlx_loader::{load_config, load_shard_index, resolve, ShardHandle, ShardSet, TensorKind};

fn primary_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

// ── resolve smoke ─────────────────────────────────────────────────────────────

#[test]
fn resolve_smoke_gemma4_mxfp8() {
    let Some(model_path_buf) = primary_model_dir() else {
        eprintln!("SKIP resolve_smoke: skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("SKIP resolve_smoke: primary test model absent at {model_path:?}");
        return;
    }

    let _cfg = load_config(model_path).expect("load_config");
    let idx = load_shard_index(model_path).expect("load_shard_index");

    let resolved = resolve(&idx).expect("resolve must succeed for well-formed snapshot");

    // Largest bucket should be Mxfp (Gemma4-mxfp8 has hundreds of quantized weights).
    let n_mxfp = resolved
        .iter()
        .filter(|t| t.kind == TensorKind::Mxfp)
        .count();
    let n_plain = resolved
        .iter()
        .filter(|t| t.kind == TensorKind::Plain)
        .count();

    assert!(
        n_mxfp > 0,
        "expected Mxfp tensors in mxfp8 snapshot, got {n_mxfp}"
    );
    assert!(
        n_plain > 0,
        "expected at least one plain tensor (embedding / RMSNorm), got {n_plain}"
    );
    // Gemma4 is multimodal: the audio/vision towers add many plain tensors.
    // The text-path quantized layers are the Mxfp bucket. Verify a meaningful
    // count rather than strict ordering.
    assert!(
        n_mxfp >= 100,
        "expected at least 100 Mxfp tensors for the quantized text layers, got {n_mxfp}"
    );

    // Every Mxfp tensor must have scales_shard set and biases_shard absent.
    for t in resolved.iter().filter(|t| t.kind == TensorKind::Mxfp) {
        assert!(
            t.scales_shard.is_some(),
            "Mxfp tensor '{}' missing scales_shard",
            t.base_name
        );
        assert!(
            t.biases_shard.is_none(),
            "Mxfp tensor '{}' should not have biases_shard",
            t.base_name
        );
    }
}

// ── mmap / ShardHandle smoke ──────────────────────────────────────────────────

#[test]
fn shard_handle_mmap_smoke() {
    let Some(model_path_buf) = primary_model_dir() else {
        eprintln!("SKIP shard_handle_mmap_smoke: skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("SKIP shard_handle_mmap_smoke: primary test model absent at {model_path:?}");
        return;
    }

    let idx = load_shard_index(model_path).expect("load_shard_index");

    // Pick the first shard filename alphabetically.
    let first_shard = idx
        .weight_map
        .values()
        .next()
        .expect("weight_map must not be empty");

    let handle = ShardHandle::open(model_path, first_shard)
        .expect("ShardHandle::open must succeed for existing shard");

    // Parse safetensors header — O(KB), not the full 4+ GB shard.
    let st = handle
        .safetensors()
        .expect("safetensors() must parse successfully");

    let tensor_names = st.names();
    assert!(
        !tensor_names.is_empty(),
        "shard must contain at least one tensor, got 0"
    );

    eprintln!(
        "shard '{}' contains {} tensors (mmap len = {} bytes)",
        first_shard,
        tensor_names.len(),
        handle.as_bytes().len()
    );
}

// ── ShardSet open smoke ───────────────────────────────────────────────────────

#[test]
fn shard_set_open_smoke() {
    let Some(model_path_buf) = primary_model_dir() else {
        eprintln!("SKIP shard_set_open_smoke: skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("SKIP shard_set_open_smoke: primary test model absent at {model_path:?}");
        return;
    }

    let idx = load_shard_index(model_path).expect("load_shard_index");
    let shard_set = ShardSet::open(model_path, &idx).expect("ShardSet::open must succeed");

    // Gemma4 mxfp8 has 2 shards.
    assert_eq!(shard_set.len(), 2, "expected 2 shards for Gemma4-mxfp8");
    assert!(!shard_set.is_empty());

    for (filename, handle) in shard_set.iter() {
        assert!(
            !handle.as_bytes().is_empty(),
            "shard '{filename}' mmap is empty"
        );
    }
}
