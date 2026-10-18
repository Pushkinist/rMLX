//! Per-arch golden-token decode gate for the Prism Hadamard Qwen3.5 pack.
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Covers the dense
//! GatedDeltaNet + full-attention backbone behind a Hadamard-rotated ternary
//! weight pack, whose `config.json` names no architecture: the harness
//! identifies the snapshot by its `model_type`, `prism_hadamard_qwen35`.
//!
//! Model: `prism-ml__Ternary-Bonsai-2-27B-mlx-2bit`.
//! KV quant: K8V8, pinned by the test (`auto` resolves to bf16).
//!
//! The snapshot resolves from `RMLX_O_MODELS_ROOT` by slug, so no per-run
//! variable is needed on a machine holding it (see `tests/common/mod.rs`).
//! A missing fixture fails the test when it runs.
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 cargo test -p rmlx-models --test bonsai2_golden_tokens -- --ignored
//! Then gate:
//! cargo test -p rmlx-models --test bonsai2_golden_tokens -- --ignored

#![allow(
    clippy::ignore_without_reason,
    reason = "a golden test is ignored because it needs a GPU and a snapshot; the module doc says so"
)]

mod common;

use rmlx_kv_quant::KvQuant;

/// The snapshot these tests cover, and the identity they were recorded against.
const MODEL: common::GoldenModel = common::GoldenModel {
    slug: "prism-ml__Ternary-Bonsai-2-27B-mlx-2bit",
    archs: &["prism_hadamard_qwen35"],
};

#[ignore]
#[test]
fn bonsai2_golden_tokens_k8v8() {
    let Some(model_path) = common::model_for(&MODEL, "bonsai2_golden_tokens_k8v8") else {
        return;
    };
    common::run_golden_test(
        "bonsai2_golden_tokens_k8v8",
        "bonsai2_27b_k8v8",
        KvQuant::K8V8,
        &model_path,
    );
}
