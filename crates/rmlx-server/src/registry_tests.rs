use super::*;
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;

fn make_snapshot(dir: &std::path::Path, arch: &str) {
    let config = serde_json::json!({
        "architectures": [arch],
        "dtype": "bfloat16"
    });
    let mut f = std::fs::File::create(dir.join("config.json")).unwrap();
    f.write_all(config.to_string().as_bytes()).unwrap();
}

#[test]
fn from_paths_skips_bad_paths() {
    let bad = PathBuf::from("/tmp/rmlx-does-not-exist-xyz");
    let reg = ModelRegistry::from_paths(&[bad]);
    assert!(reg.list().is_empty(), "bad path should be skipped");
}

#[test]
fn from_paths_loads_valid_snapshot() {
    let tmp = tempdir().unwrap();
    let snap = tmp.path().join("MyModel");
    std::fs::create_dir_all(&snap).unwrap();
    make_snapshot(&snap, "LlamaForCausalLM");

    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    assert_eq!(reg.list().len(), 1);

    let entry = reg.get("MyModel").unwrap();
    assert_eq!(entry.arch, "LlamaForCausalLM");
}

#[test]
fn list_is_alphabetical() {
    let tmp = tempdir().unwrap();
    for name in ["ZModel", "AModel", "MModel"] {
        let snap = tmp.path().join(name);
        std::fs::create_dir_all(&snap).unwrap();
        make_snapshot(&snap, "Arch");
    }
    let paths: Vec<PathBuf> = ["ZModel", "AModel", "MModel"]
        .iter()
        .map(|n| tmp.path().join(n))
        .collect();
    let reg = ModelRegistry::from_paths(&paths);
    let ids: Vec<&str> = reg.list().iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, vec!["AModel", "MModel", "ZModel"]);
}

#[test]
fn get_returns_none_for_unknown() {
    let reg = ModelRegistry::default();
    assert!(reg.get("nope").is_none());
}

/// A snapshot with only `config.json` (no chat_template.jinja, no tokenizer.json)
/// must still produce a valid entry — chat_template and tokenizer are None.
#[test]
fn missing_chat_template_gives_none_not_error() {
    let tmp = tempdir().unwrap();
    let snap = tmp.path().join("NoTemplateModel");
    std::fs::create_dir_all(&snap).unwrap();
    make_snapshot(&snap, "LlamaForCausalLM");
    // Explicitly do NOT create chat_template.jinja or tokenizer.json.

    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    assert_eq!(
        reg.list().len(),
        1,
        "entry must be present despite missing template"
    );

    let entry = reg.get("NoTemplateModel").unwrap();
    assert!(
        entry.chat_template.is_none(),
        "chat_template must be None when chat_template.jinja is absent"
    );
    assert!(
        entry.tokenizer.is_none(),
        "tokenizer must be None when tokenizer.json is absent"
    );
}

// ── A9: tools_supported probe ──────────────────────────────────────────────

/// A template with no `{% if tools %}` branch that errors when `tools` is
/// passed must yield `tools_supported = false`.
#[test]
fn tools_supported_false_for_template_that_errors_on_tools() {
    use super::probe_tools_supported;
    use crate::chat_template::ChatTemplate;

    // Template that raises an exception unconditionally if `tools` is defined
    // and non-empty — simulates a template with no tools branch.
    let src = r"{% if tools %}{{ tools | raise_exception }}{% endif %}Hello".to_owned();
    let tpl = ChatTemplate::new(src).expect("compile");
    assert!(
        !probe_tools_supported(&tpl),
        "template that errors on tools must yield tools_supported=false"
    );
}

/// A template that has a proper `{% if tools %}` branch must yield
/// `tools_supported = true`.
#[test]
fn tools_supported_true_for_template_with_tools_branch() {
    use super::probe_tools_supported;
    use crate::chat_template::ChatTemplate;

    // Minimal template with an `{% if tools %}` branch — mirrors the Qwen3 pattern.
    let src = r"{% if tools %}TOOLS{% else %}NO_TOOLS{% endif %}".to_owned();
    let tpl = ChatTemplate::new(src).expect("compile");
    assert!(
        probe_tools_supported(&tpl),
        "template with tools branch must yield tools_supported=true"
    );
}

/// A snapshot loaded from disk with a Qwen3.6 template must have
/// `tools_supported = true`. Skips if the snapshot is absent.
#[test]
fn qwen36_snapshot_tools_supported() {
    let Some(snap_buf) = std::env::var_os("RMLX_TEST_MODEL_QWEN36").map(PathBuf::from) else {
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap.to_path_buf()));
    let entry = reg.get("mlx-community__Qwen3.6-35B-A3B-8bit").unwrap();
    assert!(entry.tools_supported, "Qwen3.6 template must support tools");
}

/// the medgemma snapshot must register as a `Gemma3ForConditionalGeneration`
/// entry (text path). Snapshot-gated `#[ignore]`; run with
/// `cargo test -p rmlx-server medgemma_registers_as_gemma3 -- --ignored`.
#[test]
#[ignore]
fn medgemma_registers_as_gemma3() {
    let Some(snap_buf) = std::env::var_os("RMLX_TEST_MODEL_MEDGEMMA").map(PathBuf::from) else {
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap.to_path_buf()));
    let entry = reg
        .get("mlx-community__medgemma-1.5-4b-it-8bit")
        .expect("medgemma entry present");
    assert_eq!(
        entry.arch, "Gemma3ForConditionalGeneration",
        "medgemma must dispatch to the Gemma3 text path, not Gemma4"
    );
    assert!(
        entry.tokenizer.is_some(),
        "medgemma tokenizer.json must load"
    );
}

/// A snapshot whose template has no tools branch must have
/// `tools_supported = false` after registry build.
#[test]
fn no_tools_branch_snapshot_tools_supported_false() {
    use std::io::Write as _;

    let tmp = tempdir().unwrap();
    let snap = tmp.path().join("NoToolsModel");
    std::fs::create_dir_all(&snap).unwrap();
    make_snapshot(&snap, "FakeArch");

    // Write a template that errors when tools is non-empty.
    let mut f = std::fs::File::create(snap.join("chat_template.jinja")).unwrap();
    f.write_all(b"{% if tools %}{{ tools | raise_exception }}{% endif %}hello")
        .unwrap();

    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let entry = reg.get("NoToolsModel").unwrap();
    assert!(
        !entry.tools_supported,
        "template that errors on tools must yield tools_supported=false in registry"
    );
}
