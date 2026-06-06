//! Per-arch golden-token decode gate for BitNet (b1.58 2B-4T).
//!
//! Server-free temp=0 greedy decode of a fixed prompt must reproduce the
//! committed golden token-id sequence exactly. Covers the ternary
//! `BitNetForCausalLM` backbone (int2-packed weights dequantized to BF16 at
//! load time) — guards the trit unpack convention (value = raw-1, strided
//! row interleave) against regression.
//!
//! Model: `mlx-community__bitnet-b1.58-2B-4T` (BitNetForCausalLM).
//! KV quant: K8V8 (the resolver default for this small dense backbone).
//!
//! Record once:
//! RMLX_REGEN_GOLDENS=1 RMLX_KV_TEST_MODEL=/path/to/bitnet-b1.58-2B-4T \
//! cargo test -p rmlx-models --test bitnet_golden_tokens -- --ignored
//! Then gate:
//! RMLX_KV_TEST_MODEL=/path/to/bitnet-b1.58-2B-4T \
//! cargo test -p rmlx-models --test bitnet_golden_tokens -- --ignored

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
const EXPECTED_ARCHS: &[&str] = &["BitNetForCausalLM"];

#[ignore]
#[test]
fn bitnet_golden_tokens_k8v8() {
    let Some(model_path) = common::model_path_from_env() else {
        return;
    };
    if common::skip_if_arch_mismatch(&model_path, "bitnet_golden_tokens_k8v8", EXPECTED_ARCHS) {
        return;
    }
    common::run_golden_test("bitnet_2b_k8v8", KvQuant::K8V8);
}
