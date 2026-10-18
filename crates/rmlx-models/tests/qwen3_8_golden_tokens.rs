//! Per-arch golden-token decode gate for the dense Qwen3.5 hybrid path.
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Covers the dense
//! GatedDeltaNet + full-attention backbone (`Qwen3_5ForConditionalGeneration`).
//!
//! Model: `mlx-community__Qwen3.8-27B-4bit`.
//! KV quant: K8V8, pinned by the test (`auto` resolves to bf16).
//!
//! The snapshot resolves from `RMLX_O_MODELS_ROOT` by slug, so no per-run
//! variable is needed on a machine holding it (see `tests/common/mod.rs`).
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 cargo test -p rmlx-models --test qwen3_8_golden_tokens -- --ignored
//! Then gate:
//! cargo test -p rmlx-models --test qwen3_8_golden_tokens -- --ignored

#![allow(
    clippy::ignore_without_reason,
    reason = "a golden test is ignored because it needs a GPU and a snapshot; the module doc says so"
)]

mod common;

use rmlx_kv_quant::KvQuant;

/// The snapshot these tests cover, and the architecture they were recorded
/// against.
const MODEL: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3.8-27B-4bit",
    archs: &["Qwen3_5ForConditionalGeneration"],
};

#[ignore]
#[test]
fn qwen3_8_golden_tokens_k8v8() {
    let Some(model_path) = common::model_for(&MODEL, "qwen3_8_golden_tokens_k8v8") else {
        return;
    };
    common::run_golden_test(
        "qwen3_8_golden_tokens_k8v8",
        "qwen3_8_27b_4bit_k8v8",
        KvQuant::K8V8,
        &model_path,
    );
}
