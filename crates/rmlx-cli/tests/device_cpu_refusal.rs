//! `--device cpu` refuses a KV codec that carries MSL, and a GPU capture, up
//! front: each command exits non-zero with the refusal before it loads a
//! model.
//!
//! Each test subprocesses the built `rmlx` binary over a temp model directory
//! holding only `config.json`, so no weights are ever read and no claim is
//! taken (`--device cpu` takes none).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use std::path::Path;
use std::process::Command;

const CONFIG: &str = r#"{
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 8,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_hidden_layers": 1,
    "intermediate_size": 8,
    "vocab_size": 8,
    "head_dim": 4,
    "max_position_embeddings": 64
}"#;

const CODEC_REFUSAL: &str = "KV codec 'k8v8' runs Metal kernels and cannot run on --device cpu";

struct Run {
    code: Option<i32>,
    stderr: String,
}

fn rmlx(home: &Path, args: &[&str]) -> Run {
    rmlx_logging(home, args, "off")
}

fn rmlx_logging(home: &Path, args: &[&str], rust_log: &str) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_rmlx"))
        .env("RUST_LOG", rust_log)
        .env("RMLX_HOME", home)
        .args(args)
        .output()
        .expect("rmlx runs");
    Run {
        code: out.status.code(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Run `args` (with `{model}` replaced by a temp model directory) and assert
/// the process exits non-zero naming `refusal`.
fn assert_refused(args: &[&str], refusal: &str) {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    std::fs::create_dir(&model).unwrap();
    std::fs::write(model.join("config.json"), CONFIG).unwrap();
    std::fs::write(dir.path().join("corpus.txt"), "hello").unwrap();
    let model = model.to_str().unwrap();
    let text = dir.path().join("corpus.txt");
    let text = text.to_str().unwrap();
    let args: Vec<&str> = args
        .iter()
        .map(|&a| match a {
            "{model}" => model,
            "{text}" => text,
            other => other,
        })
        .collect();
    let run = rmlx(&dir.path().join("home"), &args);
    assert_ne!(run.code, Some(0), "{args:?} must be refused");
    assert!(
        run.stderr.contains(refusal),
        "{args:?}: expected {refusal:?} in stderr:\n{}",
        run.stderr
    );
}

#[test]
fn serve_refuses_an_msl_codec_on_cpu_before_the_registry_read() {
    assert_refused(
        &[
            "serve",
            "--registry",
            "/nonexistent/registry.toml",
            "--port",
            "0",
            "--device",
            "cpu",
            "--kv-quant",
            "k8v8",
        ],
        CODEC_REFUSAL,
    );
}

/// `serve --device cpu` holds no claim, so it skips the wired limit rather
/// than calling Metal. The missing registry then ends the run, after the
/// wired-limit decision.
#[test]
fn serve_on_cpu_skips_the_wired_limit() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let run = rmlx_logging(
        &home,
        &[
            "serve",
            "--registry",
            "/nonexistent/registry.toml",
            "--port",
            "0",
            "--device",
            "cpu",
        ],
        "info",
    );
    assert_ne!(run.code, Some(0));
    assert!(run.stderr.contains("registry file"), "{}", run.stderr);
    let logs: String = std::fs::read_dir(home.join("logs"))
        .unwrap()
        .map(|entry| std::fs::read_to_string(entry.unwrap().path()).unwrap())
        .collect();
    assert!(
        logs.contains("device is cpu; skipping set_wired_limit"),
        "{logs}"
    );
    assert!(!logs.contains("set_wired_limit failed"), "{logs}");
}

#[test]
fn chat_refuses_an_msl_codec_on_cpu() {
    assert_refused(
        &[
            "chat",
            "--model",
            "{model}",
            "--device",
            "cpu",
            "--kv-quant",
            "k8v8",
        ],
        CODEC_REFUSAL,
    );
}

#[test]
fn info_probe_refuses_an_msl_codec_on_cpu() {
    assert_refused(
        &[
            "info",
            "--model",
            "{model}",
            "--probe-forward",
            "--device",
            "cpu",
            "--kv-quant",
            "k8v8",
        ],
        CODEC_REFUSAL,
    );
}

#[test]
fn baseline_refuses_an_msl_codec_on_cpu() {
    assert_refused(
        &[
            "baseline",
            "--model",
            "{model}",
            "--device",
            "cpu",
            "--kv-quant",
            "k8v8",
        ],
        CODEC_REFUSAL,
    );
}

#[test]
fn bench_refuses_an_msl_codec_on_cpu() {
    assert_refused(
        &[
            "bench",
            "--model",
            "{model}",
            "--device",
            "cpu",
            "--kv-quant",
            "k8v8",
        ],
        CODEC_REFUSAL,
    );
}

#[cfg(feature = "metal-capture")]
#[test]
fn gpu_capture_refused_on_cpu_before_the_model_load() {
    assert_refused(
        &[
            "baseline",
            "--model",
            "{model}",
            "--device",
            "cpu",
            "--gpu-capture",
            "unused.gputrace",
        ],
        "--gpu-capture records a Metal trace and needs --device gpu",
    );
}

#[test]
fn ppl_refuses_an_msl_codec_on_cpu() {
    assert_refused(
        &[
            "eval",
            "ppl",
            "--model",
            "{model}",
            "--text-file",
            "{text}",
            "--device",
            "cpu",
            "--kv-quant",
            "k8v8",
        ],
        CODEC_REFUSAL,
    );
}
