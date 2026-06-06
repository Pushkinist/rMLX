//! Nested-config `eos_token_id` resolution.
//!
//! Qwen3-VL-MoE sets the top-level `eos_token_id` to `null` and carries the
//! real id (151645) inside `text_config.eos_token_id`. `eos_token_ids()` must
//! fall back to the nested value. Gated on the on-disk snapshot.

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
    // disallowed_methods is a separate lint from unwrap_used;
    // integration test code (bucket-B) is already exempted for unwrap_used.
    clippy::disallowed_methods,
)]

use rmlx_loader::load_config;

fn qwen3_vl_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_QWEN3_VL_30B").map(std::path::PathBuf::from)
}

#[test]
fn nested_text_config_eos_resolves() {
    let Some(model_path) = qwen3_vl_model_dir() else {
        eprintln!("SKIP: RMLX_TEST_MODEL_QWEN3_VL_30B not set");
        return;
    };
    let p = model_path.as_path();
    if !p.exists() {
        eprintln!("SKIP: Qwen3-VL snapshot absent at {p:?}");
        return;
    }
    let cfg = load_config(p).unwrap();
    let eos = cfg.eos_token_ids();
    assert!(
        eos.contains(&151645),
        "expected nested text_config eos 151645 in {eos:?}"
    );
}
