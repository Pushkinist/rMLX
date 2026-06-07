//! Integration tests for `--cache-type-k` / `--cache-type-v` fail-fast semantics.
//!
//! Each test subprocesses the built `rmlx` binary (via `CARGO_BIN_EXE_rmlx`) and
//! asserts the exit code and stderr content.
//!
//! ## Two tiers
//!
//! **Env-free** (no model file required — always run):
//! - `clap_collision_kv_quant_and_cache_type_k` — exit 2, clap conflict error.
//! - `unknown_cache_type_tag` — exit 1, parse error ("unknown").
//! - `q8_0_returns_not_implemented_hint` — exit 1, parse error ("llama.cpp legacy" +
//!   substitute tag).
//!
//! **Env-gated** (require real snapshots — skip when env unset):
//! - `tq4_on_k_side_rejected` — `RMLX_TEST_MODEL_BONSAI`, exit 78.
//! - `planar4_on_k_side_rejected` — `RMLX_TEST_MODEL_BONSAI`, exit 78.
//! - `qwen_moe_low_k_bits_rejected` — `RMLX_TEST_MODEL_QWEN36`, exit 78.
//! - `asymmetric_auto_with_tq4_rejected_on_bonsai` — `RMLX_TEST_MODEL_BONSAI`, exit 78.
//!   (Bonsai's auto K-side decomposes to `q8_g64`, which is not `q8_g128`;
//!   the asymmetric coercion guard rejects rather than silently promoting.)

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
    clippy::manual_let_else
)]

use std::process::Command;

// ── helpers ───────────────────────────────────────────────────────────────────

/// Locate the cargo-built `rmlx` binary.
///
/// `CARGO_BIN_EXE_rmlx` is injected by Cargo's integration-test runner.
fn rmlx_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_rmlx"))
}

struct RunResult {
    exit_code: i32,
    stderr: String,
}

/// Run `rmlx` with the given args, isolated in a per-invocation temp dir so
/// parallel tests do not contend on the same SQLite metrics DB.
fn run(args: &[&str]) -> RunResult {
    // Each call gets a unique directory under /tmp so concurrent test threads
    // don't contend on the same SQLite `runs.db`.
    let rmlx_home = std::env::temp_dir().join(format!(
        "rmlx_ctv_{}_{}",
        std::process::id(),
        // Use a monotonic counter to differentiate calls within the same process.
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            static CTR: AtomicU64 = AtomicU64::new(0);
            CTR.fetch_add(1, Ordering::Relaxed)
        }
    ));

    let out = Command::new(rmlx_bin())
        // Suppress log noise; tests assert on plain `error:` lines written via `eprintln!`.
        .env("RUST_LOG", "off")
        // Isolate filesystem side-effects from the workspace .rmlx/ directory.
        .env("RMLX_HOME", &rmlx_home)
        .args(args)
        .output()
        .expect("failed to spawn rmlx subprocess");

    let exit_code = out.status.code().unwrap_or(-1);
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    RunResult { exit_code, stderr }
}

// ── env-free tests (always run) ───────────────────────────────────────────────

/// clap rejects `--kv-quant` and `--cache-type-k` together (exit 2).
///
/// stderr: "cannot be used with"
#[test]
fn clap_collision_kv_quant_and_cache_type_k() {
    let r = run(&[
        "info",
        "--model",
        "SOMETHING",
        "--kv-quant",
        "k8v4",
        "--cache-type-k",
        "q8_g128",
    ]);
    assert_eq!(
        r.exit_code, 2,
        "expected exit 2 (clap conflict), got {}; stderr was: {}",
        r.exit_code, r.stderr
    );
    assert!(
        r.stderr.contains("cannot be used with"),
        "expected 'cannot be used with' in stderr; got: {}",
        r.stderr
    );
}

/// Unknown tag is rejected with exit 1 at parse time (before model load).
///
/// stderr: "unknown cache type 'garbage'"
///
/// Exit is 1 (anyhow error path), not 78 — `build_cache_type_spec` fails
/// before `load_config` is ever called.
#[test]
fn unknown_cache_type_tag() {
    let r = run(&["info", "--model", "SOMETHING", "--cache-type-k", "garbage"]);
    assert_ne!(
        r.exit_code, 0,
        "expected non-zero exit for unknown tag; stderr was: {}",
        r.stderr
    );
    assert!(
        r.stderr.to_lowercase().contains("unknown"),
        "expected 'unknown' in stderr; got: {}",
        r.stderr
    );
}

/// `q8_0` (llama.cpp legacy block-32 codec) is rejected with exit 1 at parse
/// time. stderr must contain "llama.cpp legacy" AND a substitute hint tag.
///
/// Exit is 1 (anyhow error path), not 78 — `parse_cache_type` fails before
/// `load_config` is called.
#[test]
fn q8_0_returns_not_implemented_hint() {
    let r = run(&["info", "--model", "SOMETHING", "--cache-type-k", "q8_0"]);
    assert_ne!(
        r.exit_code, 0,
        "expected non-zero exit for q8_0; stderr was: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("llama.cpp legacy"),
        "expected 'llama.cpp legacy' in stderr; got: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("q8_g32") || r.stderr.contains("q8_g128"),
        "expected a substitute hint tag ('q8_g32' or 'q8_g128') in stderr; got: {}",
        r.stderr
    );
}

// ── env-gated tests (require real model snapshots) ────────────────────────────

/// `--cache-type-k tq4` rejected with exit 78 on Bonsai (K-side rotation
/// codec not allowed).
///
/// stderr: "K-side rotation codec 'tq4' not implemented"
///
/// Requires: `RMLX_TEST_MODEL_BONSAI=/path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit`
#[test]
fn tq4_on_k_side_rejected() {
    let bonsai = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_BONSAI") {
        p
    } else {
        eprintln!("skipping tq4_on_k_side_rejected: RMLX_TEST_MODEL_BONSAI not set");
        return;
    };
    let r = run(&["info", "--model", &bonsai, "--cache-type-k", "tq4"]);
    assert_eq!(
        r.exit_code, 78,
        "expected exit 78 (EX_CONFIG); stderr was: {}",
        r.stderr
    );
    assert!(
        r.stderr
            .contains("K-side rotation codec 'tq4' not implemented"),
        "expected rotation-codec rejection message in stderr; got: {}",
        r.stderr
    );
}

/// `--cache-type-k planar4` rejected with exit 78 on Bonsai (K-side rotation
/// codec not allowed).
///
/// stderr: "K-side rotation codec 'planar4' not implemented"
///
/// Requires: `RMLX_TEST_MODEL_BONSAI=/path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit`
#[test]
fn planar4_on_k_side_rejected() {
    let bonsai = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_BONSAI") {
        p
    } else {
        eprintln!("skipping planar4_on_k_side_rejected: RMLX_TEST_MODEL_BONSAI not set");
        return;
    };
    let r = run(&["info", "--model", &bonsai, "--cache-type-k", "planar4"]);
    assert_eq!(
        r.exit_code, 78,
        "expected exit 78 (EX_CONFIG); stderr was: {}",
        r.stderr
    );
    assert!(
        r.stderr
            .contains("K-side rotation codec 'planar4' not implemented"),
        "expected rotation-codec rejection message in stderr; got: {}",
        r.stderr
    );
}

/// `--cache-type-k q4_g64 --cache-type-v q4_g64` rejected with exit 78 on
/// Qwen3.6-35B-A3B (Qwen MoE requires K-side bits >= 8).
///
/// stderr: "Qwen MoE family requires K-side bits >= 8"
///
/// Requires: `RMLX_TEST_MODEL_QWEN36=/path/to/mlx-community__Qwen3.6-35B-A3B-8bit`
#[test]
fn qwen_moe_low_k_bits_rejected() {
    let qwen = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_QWEN36") {
        p
    } else {
        eprintln!("skipping qwen_moe_low_k_bits_rejected: RMLX_TEST_MODEL_QWEN36 not set");
        return;
    };
    let r = run(&[
        "info",
        "--model",
        &qwen,
        "--cache-type-k",
        "q4_g64",
        "--cache-type-v",
        "q4_g64",
    ]);
    assert_eq!(
        r.exit_code, 78,
        "expected exit 78 (EX_CONFIG); stderr was: {}",
        r.stderr
    );
    assert!(
        r.stderr
            .contains("Qwen MoE family requires K-side bits >= 8"),
        "expected Qwen MoE K-bits rejection in stderr; got: {}",
        r.stderr
    );
}

/// `--cache-type-k q8_g128 --cache-type-v q4_g64` on Gemma4 is now
/// ACCEPTED (the shared-KV path dequantises before share). `info` does no
/// inference, so it must exit 0 with the Mixed combo resolved.
///
/// Requires: `RMLX_TEST_MODEL_GEMMA4_E4B=/path/to/mlx-community__gemma-4-e4b-it-mxfp8`
#[test]
fn gemma4_mixed_cache_type_spec_accepted() {
    let gemma4 = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_GEMMA4_E4B") {
        p
    } else {
        eprintln!(
            "skipping gemma4_mixed_cache_type_spec_accepted: RMLX_TEST_MODEL_GEMMA4_E4B not set"
        );
        return;
    };
    let r = run(&[
        "info",
        "--model",
        &gemma4,
        "--cache-type-k",
        "q8_g128",
        "--cache-type-v",
        "q4_g64",
    ]);
    assert_eq!(
        r.exit_code, 0,
        "expected exit 0 (Gemma4+Mixed now accepted); stderr was: {}",
        r.stderr
    );
}

/// `--kv-quant mixed_k8g128_v4g64` on Gemma4 is now accepted at startup
/// (validate_resolved no longer rejects shared-KV + Mixed).
///
/// Requires: `RMLX_TEST_MODEL_GEMMA4_E4B=/path/to/mlx-community__gemma-4-e4b-it-mxfp8`
#[test]
fn gemma4_mixed_kv_quant_preset_accepted() {
    let gemma4 = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_GEMMA4_E4B") {
        p
    } else {
        eprintln!(
            "skipping gemma4_mixed_kv_quant_preset_accepted: RMLX_TEST_MODEL_GEMMA4_E4B not set"
        );
        return;
    };
    let r = run(&[
        "info",
        "--model",
        &gemma4,
        "--kv-quant",
        "mixed_k8g128_v4g64",
    ]);
    assert_eq!(
        r.exit_code, 0,
        "expected exit 0 (Gemma4+Mixed preset now accepted); stderr was: {}",
        r.stderr
    );
}

/// `--kv-quant k8v8` on Gemma4 must succeed with exit 0.
///
/// Requires: `RMLX_TEST_MODEL_GEMMA4_E4B=/path/to/mlx-community__gemma-4-e4b-it-mxfp8`
#[test]
fn gemma4_k8v8_accepted() {
    let gemma4 = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_GEMMA4_E4B") {
        p
    } else {
        eprintln!("skipping gemma4_k8v8_accepted: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let r = run(&["info", "--model", &gemma4, "--kv-quant", "k8v8"]);
    assert_eq!(
        r.exit_code, 0,
        "expected exit 0 for k8v8 on Gemma4; stderr was: {}",
        r.stderr
    );
}

/// Mixed on Bonsai must NOT be rejected — Bonsai is not a shared-KV arch.
///
/// Requires: `RMLX_TEST_MODEL_BONSAI=/path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit`
#[test]
fn bonsai_mixed_not_rejected() {
    let bonsai = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_BONSAI") {
        p
    } else {
        eprintln!("skipping bonsai_mixed_not_rejected: RMLX_TEST_MODEL_BONSAI not set");
        return;
    };
    let r = run(&[
        "info",
        "--model",
        &bonsai,
        "--kv-quant",
        "mixed_k8g64_v4g64",
    ]);
    assert_eq!(
        r.exit_code, 0,
        "expected exit 0 for Mixed on Bonsai (not a shared-KV arch); stderr was: {}",
        r.stderr
    );
}

/// `--cache-type-k auto --cache-type-v tq4` on Bonsai is rejected with exit 78.
///
/// Bonsai's `auto` resolves to `Mixed{k_bits:8, k_group_size:64, ...}`, which
/// decomposes to K=`q8_g64`. The asymmetric coercion guard in `combo_to_kv_quant`
/// only coerces `(q8_g128, tq4) → K8V4`; it never silently promotes a different
/// K codec. `(q8_g64, tq4)` is therefore an `UnsupportedCombo`.
///
/// This test documents and pins the resolver's no-silent-coercion invariant for
/// Bonsai-style mixed auto defaults.
///
/// stderr: contains both "q8_g64" and "tq4"
///
/// Requires: `RMLX_TEST_MODEL_BONSAI=/path/to/prism-ml__Ternary-Bonsai-8B-mlx-2bit`
#[test]
fn asymmetric_auto_with_tq4_rejected_on_bonsai() {
    let bonsai = if let Ok(p) = std::env::var("RMLX_TEST_MODEL_BONSAI") {
        p
    } else {
        eprintln!(
            "skipping asymmetric_auto_with_tq4_rejected_on_bonsai: RMLX_TEST_MODEL_BONSAI not set"
        );
        return;
    };
    let r = run(&[
        "info",
        "--model",
        &bonsai,
        "--cache-type-k",
        "auto",
        "--cache-type-v",
        "tq4",
    ]);
    assert_eq!(
        r.exit_code, 78,
        "expected exit 78 (EX_CONFIG — Bonsai auto K=q8_g64 not coerced to q8_g128); \
         stderr was: {}",
        r.stderr
    );
    // The UnsupportedCombo message names both the K codec and the V codec.
    assert!(
        r.stderr.contains("q8_g64"),
        "expected K codec 'q8_g64' named in stderr; got: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("tq4"),
        "expected V codec 'tq4' named in stderr; got: {}",
        r.stderr
    );
}
