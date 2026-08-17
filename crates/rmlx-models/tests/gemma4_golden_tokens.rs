//! Per-arch golden-token decode gate for Gemma4 (e4b).
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Catches genuine decode
//! regressions without server/metrics noise.
//!
//! Model: `mlx-community__gemma-4-e4b-it-mxfp8` (Gemma4ForConditionalGeneration).
//! KV quant: K8V8 (the resolver default for small non-MoE Gemma4).
//!
//! The snapshot resolves from `RMLX_O_MODELS_ROOT` by slug, so no per-run
//! variable is needed on a machine holding it (see `tests/common/mod.rs`).
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=/path/to/gemma-4-e4b-it-mxfp8 \
//! cargo test -p rmlx-models --test gemma4_golden_tokens -- --ignored
//! Then gate:
//! cargo test -p rmlx-models --test gemma4_golden_tokens -- --ignored

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
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

mod common;

use rmlx_kv_quant::KvQuant;

/// The snapshot this golden covers, and the architectures it was recorded
/// against.
const MODEL: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__gemma-4-e4b-it-mxfp8",
    archs: &["Gemma4ForConditionalGeneration"],
};

#[ignore]
#[test]
fn gemma4_golden_tokens_k8v8() {
    let Some(model_path) = common::model_for(&MODEL, "gemma4_golden_tokens_k8v8") else {
        return;
    };
    common::run_golden_test("gemma4_e4b_k8v8", KvQuant::K8V8, &model_path);
}
