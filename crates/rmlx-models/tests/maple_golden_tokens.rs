//! Per-arch golden-token decode gate for Maple (DeepGrove 20B-A1B ternary MoE).
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Covers `MapleForCausalLM`
//! (hybrid SWA/NoPE, affine 2-bit `row_alpha`, clamped-SwiGLU MoE).
//!
//! Model: `maple-2bit-mlx`.
//! KV quant: K8V8 (the advertised Maple default; `auto` is unquantised bf16).
//!
//! The snapshot resolves from `RMLX_O_MODELS_ROOT` by slug, so no per-run
//! variable is needed on a machine holding it (see `tests/common/mod.rs`).
//! A local checkout that is not under the models root can still arm the
//! gate with `RMLX_KV_TEST_MODEL`.
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=/path/to/maple-2bit-mlx \
//! cargo test -p rmlx-models --test maple_golden_tokens -- --ignored
//! Then gate:
//! cargo test -p rmlx-models --test maple_golden_tokens -- --ignored

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
    slug: "maple-2bit-mlx",
    archs: &["MapleForCausalLM"],
};

#[ignore]
#[test]
fn maple_golden_tokens_k8v8() {
    let Some(model_path) = common::model_for(&MODEL, "maple_golden_tokens_k8v8") else {
        return;
    };
    common::run_golden_test("maple_2bit_k8v8", KvQuant::K8V8, &model_path);
}
