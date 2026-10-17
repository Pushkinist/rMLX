//! The CLI readers of a snapshot's dispatch class, driven as an operator
//! drives them: `serve --device cpu` (no claim, and no model load before its
//! startup check) and `info` without a probe (no MLX). Each test runs the
//! built `rmlx` binary over a temp model directory that holds a synthetic
//! `config.json` and, for `info`, a one-tensor `model.safetensors`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The `config.json` of a Prism Hadamard pack on a Qwen3.5 backbone: no
/// `architectures`. The tests write no contract file beside it.
const PACK_CONFIG: &str = r#"{
    "model_type": "prism_hadamard_qwen35",
    "base_model_type": "qwen3_5",
    "hadamard_config": "hadamard.json",
    "quantization": {"group_size": 128, "bits": 2, "mode": "affine"}
}"#;

/// A `serve` that passes its startup check goes on to listen; this bound
/// ends such a run, which then reads as no exit code.
const RUN_BOUND: Duration = Duration::from_secs(60);

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn rmlx(home: &Path, args: &[&str]) -> Run {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rmlx"))
        .env("RUST_LOG", "off")
        .env("RMLX_HOME", home)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("rmlx runs");
    let deadline = Instant::now() + RUN_BOUND;
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break Some(status);
        }
        if Instant::now() > deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    Run {
        code: status.and_then(|s| s.code()),
        stdout,
        stderr,
    }
}

/// A model directory under `dir` holding `config`.
fn model_dir(dir: &Path, config: &str) -> std::path::PathBuf {
    let model = dir.join("model");
    std::fs::create_dir(&model).unwrap();
    std::fs::write(model.join("config.json"), config).unwrap();
    model
}

fn serve(model: &Path, home: &Path) -> Run {
    rmlx(
        home,
        &[
            "serve",
            "--model",
            model.to_str().unwrap(),
            "--port",
            "0",
            "--device",
            "cpu",
        ],
    )
}

/// `serve` refuses a Hadamard pack whose contract does not hold at its
/// startup check, exit 1, by the fact: here the contract file is missing.
/// Every load path refuses such a pack, so a server that started would
/// refuse every request.
#[test]
fn serve_refuses_a_hadamard_pack_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let model = model_dir(dir.path(), PACK_CONFIG);
    let run = serve(&model, &dir.path().join("home"));
    assert_eq!(run.code, Some(1), "stderr:\n{}", run.stderr);
    assert!(
        run.stderr.contains("is a Hadamard pack") && run.stderr.contains("hadamard.json"),
        "{}",
        run.stderr
    );
}

/// `serve --draft-model <pack>` refuses a pack whose contract does not hold
/// at its startup check, exit 1, as it does for `--model`: a server that
/// started would refuse every request.
#[test]
fn serve_refuses_a_hadamard_pack_as_the_draft_model_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let model = model_dir(dir.path(), r#"{"architectures": ["Qwen3ForCausalLM"]}"#);
    let draft = dir.path().join("draft");
    std::fs::create_dir(&draft).unwrap();
    std::fs::write(draft.join("config.json"), PACK_CONFIG).unwrap();
    let run = rmlx(
        &dir.path().join("home"),
        &[
            "serve",
            "--model",
            model.to_str().unwrap(),
            "--draft-model",
            draft.to_str().unwrap(),
            "--port",
            "0",
            "--device",
            "cpu",
        ],
    );
    assert_eq!(run.code, Some(1), "stderr:\n{}", run.stderr);
    assert!(run.stderr.contains("is a Hadamard pack"), "{}", run.stderr);
}

/// The startup check names the dispatch class it refuses.
#[test]
fn serve_names_the_dispatch_class_it_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let model = model_dir(dir.path(), r#"{"architectures": ["FooForCausalLM"]}"#);
    let run = serve(&model, &dir.path().join("home"));
    assert_eq!(run.code, Some(1), "stderr:\n{}", run.stderr);
    assert!(
        run.stderr
            .contains("architecture 'FooForCausalLM' not yet supported"),
        "{}",
        run.stderr
    );
}

/// `info` prints the class the config dispatches on: for a Hadamard pack,
/// its backbone's class.
#[test]
fn info_prints_the_dispatch_class_of_a_hadamard_pack() {
    let dir = tempfile::tempdir().unwrap();
    let model = model_dir(dir.path(), PACK_CONFIG);
    let header = br#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut shard = (header.len() as u64).to_le_bytes().to_vec();
    shard.extend_from_slice(header);
    shard.extend_from_slice(&1.0f32.to_le_bytes());
    std::fs::write(model.join("model.safetensors"), shard).unwrap();

    let run = rmlx(
        &dir.path().join("home"),
        &["info", "--model", model.to_str().unwrap()],
    );
    assert_eq!(run.code, Some(0), "stderr:\n{}", run.stderr);
    assert!(
        run.stdout
            .lines()
            .any(|l| l == "arch:  Qwen3_5ForConditionalGeneration"),
        "{}",
        run.stdout
    );
}

/// A pack whose `base_model_type` has no class is refused at startup by that
/// fact, not as an empty architecture.
#[test]
fn serve_names_the_base_model_type_of_a_pack_with_no_class() {
    let dir = tempfile::tempdir().unwrap();
    let model = model_dir(dir.path(), &PACK_CONFIG.replace("\"qwen3_5\"", "\"llama\""));
    let run = serve(&model, &dir.path().join("home"));
    assert_eq!(run.code, Some(1), "stderr:\n{}", run.stderr);
    assert!(
        run.stderr.contains("base_model_type") && run.stderr.contains("\"llama\""),
        "{}",
        run.stderr
    );
    assert!(!run.stderr.contains("(empty)"), "{}", run.stderr);
}

/// `serve --registry` runs the pack check on each model the registry file
/// names: a pack the check refuses stops the startup, exit 1, by its fact. A
/// registry of models the check admits passes it: the run goes on to bind a
/// port the test holds, and fails there.
#[test]
fn serve_refuses_a_registry_that_names_a_pack_it_cannot_serve() {
    let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = held.local_addr().unwrap().port().to_string();
    let dir = tempfile::tempdir().unwrap();
    let plain = model_dir(dir.path(), r#"{"architectures": ["Qwen3ForCausalLM"]}"#);
    let pack = dir.path().join("pack");
    std::fs::create_dir(&pack).unwrap();
    std::fs::write(pack.join("config.json"), PACK_CONFIG).unwrap();
    for (models, refused) in [(vec![&plain], false), (vec![&plain, &pack], true)] {
        let entries: Vec<_> = models
            .iter()
            .map(|p| serde_json::json!({"path": p}))
            .collect();
        let registry = dir.path().join("registry.json");
        std::fs::write(
            &registry,
            serde_json::json!({ "models": entries }).to_string(),
        )
        .unwrap();
        let run = rmlx(
            &dir.path().join("home"),
            &[
                "serve",
                "--registry",
                registry.to_str().unwrap(),
                "--port",
                &port,
                "--device",
                "cpu",
            ],
        );
        if refused {
            assert_eq!(run.code, Some(1), "stderr:\n{}", run.stderr);
            assert!(
                run.stderr.contains("is a Hadamard pack") && run.stderr.contains("hadamard.json"),
                "{}",
                run.stderr
            );
        } else {
            assert!(
                run.code.is_some_and(|c| c != 0) && !run.stderr.contains("Hadamard"),
                "a registry the check admits fails on the held port: {:?}\n{}",
                run.code,
                run.stderr
            );
        }
    }
    drop(held);
}

/// `serve --model <pack> --draft-model <EAGLE-3 drafter>` exits 1 at startup
/// when the drafter has a reduced vocabulary: its restricted-vocabulary head
/// would gather folded output rows of the pack, so every request would be
/// refused. The check reads the two configs only. Controls: a drafter with
/// the full vocabulary passes it and the run stops at the pack's own check
/// (here its contract file is missing); the reduced-vocabulary drafter over a
/// plain `--model` passes it and the run fails only on a port the test holds.
#[test]
fn serve_refuses_an_eagle3_drafter_with_hot_ids_over_a_pack_at_startup() {
    let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = held.local_addr().unwrap().port().to_string();
    let dir = tempfile::tempdir().unwrap();
    let pack = model_dir(dir.path(), PACK_CONFIG);
    let plain = dir.path().join("plain");
    std::fs::create_dir(&plain).unwrap();
    std::fs::write(
        plain.join("config.json"),
        r#"{"architectures": ["Qwen3ForCausalLM"]}"#,
    )
    .unwrap();
    for (model, draft_vocab, refused) in [(&pack, 4, true), (&pack, 8, false), (&plain, 4, false)] {
        let draft = dir.path().join(format!("draft-{draft_vocab}"));
        std::fs::create_dir_all(&draft).unwrap();
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLMEagle3"],
            "hidden_size": 16,
            "draft_vocab_size": draft_vocab, "vocab_size": 8
        });
        std::fs::write(draft.join("config.json"), config.to_string()).unwrap();
        let run = rmlx(
            &dir.path().join("home"),
            &[
                "serve",
                "--model",
                model.to_str().unwrap(),
                "--draft-model",
                draft.to_str().unwrap(),
                "--port",
                &port,
                "--device",
                "cpu",
            ],
        );
        assert!(
            run.code.is_some_and(|c| c != 0),
            "{}: {:?}\n{}",
            model.display(),
            run.code,
            run.stderr
        );
        assert_eq!(
            run.stderr.contains("restricted-vocab head"),
            refused,
            "{} with draft vocabulary {draft_vocab}: {}",
            model.display(),
            run.stderr
        );
    }
    drop(held);
}
