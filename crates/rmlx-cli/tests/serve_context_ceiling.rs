//! `rmlx serve` refuses to start when `--max-ctx` is above what the checkpoint
//! can address.
//!
//! An unsatisfiable `--max-ctx` is operator input, not a transient load
//! failure: no retry fixes it, and every request would 503 for the life of the
//! process. Coming up healthy on a port and failing every request is worse than
//! not coming up, so the eager preload aborts startup before the port is bound.
//!
//! Env-gated on `RMLX_TEST_MODEL_GEMMA4_E2B`; the run loads the model (and so
//! enters Metal) before the refusal, which is why it is `#[ignore]`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "integration-test scaffolding: panics surface assertion failures and stderr is the artefact under test"
)]

use std::process::Command;

/// Locate the cargo-built `rmlx` binary.
fn rmlx_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_rmlx"))
}

/// A port no other test in this file uses, high enough to be free on a dev box.
const PORT: &str = "8231";

/// `--max-ctx` past the checkpoint's positional capacity aborts startup with
/// the context resolver's message, and leaves nothing listening.
// gpu-test-gate: metal-unscanned  the child process loads the model on Metal.
#[test]
#[ignore = "GPU Metal: RMLX_TEST_MODEL_GEMMA4_E2B=... cargo test -p rmlx-cli --test serve_context_ceiling -- --ignored"]
fn serve_refuses_to_start_above_the_positional_capacity() {
    let Some(model) = std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E2B") else {
        eprintln!("RMLX_TEST_MODEL_GEMMA4_E2B not set — skipping");
        return;
    };
    let rmlx_home = std::env::temp_dir().join(format!("rmlx_serve_ctx_{}", std::process::id()));

    let out = Command::new(rmlx_bin())
        .env("RMLX_HOME", &rmlx_home)
        .arg("serve")
        .arg("--model")
        .arg(&model)
        .args(["--port", PORT, "--kv-quant", "none", "--max-ctx", "200000"])
        .args(["--metrics", "off"])
        .output()
        .expect("failed to spawn rmlx serve");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_ne!(
        out.status.code(),
        Some(0),
        "serve must not start; stderr: {stderr}"
    );
    assert!(
        stderr.contains("200000") && stderr.contains("positional capacity"),
        "the refusal must name the request and the capacity; stderr: {stderr}"
    );
    assert!(
        std::net::TcpStream::connect(("127.0.0.1", PORT.parse::<u16>().expect("port parses")))
            .is_err(),
        "nothing may be listening on {PORT} after a refused startup"
    );

    std::fs::remove_dir_all(&rmlx_home).ok();
}
