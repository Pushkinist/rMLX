//! Per-arch golden-token decode gate for Gemma4 (e4b).
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Catches genuine decode
//! regressions without server/metrics noise.
//!
//! Model: `mlx-community__gemma-4-e4b-it-mxfp8` (Gemma4ForConditionalGeneration).
//! KV quant: K8V8 (the resolver default for small non-MoE Gemma4).
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=/path/to/gemma-4-e4b-it-mxfp8 \
//! cargo test -p rmlx-models --test gemma4_golden_tokens -- --ignored
//! Then gate:
//! RMLX_KV_TEST_MODEL=/path/to/gemma-4-e4b-it-mxfp8 \
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

/// Architectures this golden was recorded against. Any other arch is skipped.
const EXPECTED_ARCHS: &[&str] = &["Gemma4ForConditionalGeneration"];

#[ignore]
#[test]
fn gemma4_golden_tokens_k8v8() {
    let Some(model_path) = common::model_path_from_env() else {
        return;
    };
    if common::skip_if_arch_mismatch(&model_path, "gemma4_golden_tokens_k8v8", EXPECTED_ARCHS) {
        return;
    }
    common::run_golden_test("gemma4_e4b_k8v8", KvQuant::K8V8);
}
