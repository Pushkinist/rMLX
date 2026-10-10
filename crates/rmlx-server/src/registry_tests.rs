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

// ── tools_supported probe ──────────────────────────────────────────────

/// A template with no `{% if tools %}` branch that errors when `tools` is
/// passed must yield `tools_supported = false`.
#[test]
fn tools_supported_false_for_template_that_errors_on_tools() {
    use super::probe_tools_supported;
    use crate::chat_template::ChatTemplate;

    // Template that raises an exception unconditionally if `tools` is defined
    // and non-empty — simulates a template with no tools branch.
    let src = r"{% if tools %}{{ raise_exception('no tools') }}{% endif %}Hello".to_owned();
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
/// entry (text path). Returns without asserting when `RMLX_TEST_MODEL_MEDGEMMA`
/// names no snapshot; registration reads only the config and tokenizer.
#[test]
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
    f.write_all(b"{% if tools %}{{ raise_exception('no tools') }}{% endif %}hello")
        .unwrap();

    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let entry = reg.get("NoToolsModel").unwrap();
    assert!(
        !entry.tools_supported,
        "template that errors on tools must yield tools_supported=false in registry"
    );
}

/// The slug of the Prism Hadamard pack on a Qwen3.5 backbone.
const PRISM_PACK: &str = "prism-ml__Ternary-Bonsai-2-27B-mlx-2bit";

/// A Prism Hadamard pack names no `architectures`. It registers under the
/// class of its `base_model_type`, the class its first-generation sibling
/// declares, and takes that class's prefill-chunk row. The pack's own
/// `config.json` is read too when `RMLX_O_MODELS_ROOT` holds it.
#[test]
fn a_hadamard_pack_registers_under_its_backbone_class() {
    let tmp = tempdir().unwrap();
    let mut configs = vec![(
        "synthetic",
        serde_json::json!({
            "model_type": "prism_hadamard_qwen35",
            "base_model_type": "qwen3_5",
            "hadamard_config": "hadamard.json",
        })
        .to_string()
        .into_bytes(),
    )];
    let real = std::env::var("RMLX_O_MODELS_ROOT")
        .ok()
        .map(|root| PathBuf::from(root).join(PRISM_PACK).join("config.json"))
        .filter(|p| p.is_file());
    match real {
        Some(p) => configs.push(("real", std::fs::read(p).unwrap())),
        None => {
            println!("note: RMLX_O_MODELS_ROOT does not hold {PRISM_PACK}; synthetic config only");
        }
    }
    for (id, config) in configs {
        let snap = tmp.path().join(id);
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("config.json"), config).unwrap();
        let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
        let entry = reg.get(id).unwrap();
        assert_eq!(entry.arch, "Qwen3_5ForConditionalGeneration", "{id}");
        assert_eq!(
            rmlx_models::prefill_chunk::module_key_for_class(&entry.arch),
            "qwen3_5_moe",
            "{id}"
        );
    }
}

/// A config that names no class and is no Hadamard pack registers as
/// `unknown`.
#[test]
fn a_config_with_no_class_registers_as_unknown() {
    let tmp = tempdir().unwrap();
    let snap = tmp.path().join("sidecar");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(snap.join("config.json"), r#"{"model_type": "qwen3_5_mtp"}"#).unwrap();
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    assert_eq!(reg.get("sidecar").unwrap().arch, "unknown");
}

/// `RMLX_O_MODELS_ROOT` when it names a directory, else `None` after a
/// `SKIP <test>` line.
fn models_root(test: &str) -> Option<PathBuf> {
    let root = std::env::var("RMLX_O_MODELS_ROOT")
        .ok()
        .filter(|r| !r.is_empty())
        .map(PathBuf::from)
        .filter(|r| r.is_dir());
    if root.is_none() {
        println!("SKIP {test}: RMLX_O_MODELS_ROOT names no directory");
    }
    root
}

/// Every local snapshot the server can chat with stops at the token its
/// tokenizer config names as `eos_token`: the id `tokenizer.json` gives that
/// token is in the stop set the registry holds, which is the set the model
/// load reads. Outside the census: a
/// snapshot with no dispatch class (a drafter sidecar), no chat template, or
/// no tokenizer that knows the token.
#[test]
fn every_chat_snapshot_stops_at_the_eos_token_its_tokenizer_config_names() {
    const TEST: &str = "every_chat_snapshot_stops_at_the_eos_token_its_tokenizer_config_names";
    let Some(root) = models_root(TEST) else {
        return;
    };
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|dir| dir.join("config.json").is_file())
        .collect();
    dirs.sort();
    let mut checked = Vec::new();
    let mut wrong = Vec::new();
    for dir in &dirs {
        let reg = ModelRegistry::from_paths(std::slice::from_ref(dir));
        let Some(entry) = reg.list().into_iter().next() else {
            continue;
        };
        let (Some(_), Some(tokenizer), Some(eos)) =
            (&entry.chat_template, &entry.tokenizer, &entry.eos_token)
        else {
            continue;
        };
        let Some(eos_id) = tokenizer.token_to_id(eos) else {
            continue;
        };
        if entry.arch == "unknown" {
            continue;
        }
        let stop = entry.stop_token_ids.clone();
        let loaded = rmlx_loader::stop_token_ids(&entry.abs_path).unwrap();
        if loaded != stop {
            wrong.push(format!(
                "{}: registry {stop:?}, model load {loaded:?}",
                entry.id
            ));
        }
        checked.push(format!("{} {eos}={eos_id} {stop:?}", entry.id));
        if !stop.contains(&eos_id) {
            wrong.push(format!(
                "{}: {eos} is {eos_id}, stop set {stop:?}",
                entry.id
            ));
        }
    }
    println!(
        "{TEST}: {} of {} snapshots:\n  {}",
        checked.len(),
        dirs.len(),
        checked.join("\n  ")
    );
    assert!(
        !checked.is_empty(),
        "{} holds no chat snapshot",
        root.display()
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The Ternary-Bonsai-2 pack, served as the registry builds it from its
/// files: its template asks for the Qwen3 XML tool-call form and renders a
/// tool, and the stop set holds `<|im_end|>`, the end of a chat turn.
#[test]
fn the_bonsai_2_pack_asks_for_xml_tool_calls_and_stops_at_im_end() {
    const TEST: &str = "the_bonsai_2_pack_asks_for_xml_tool_calls_and_stops_at_im_end";
    let Some(root) = models_root(TEST) else {
        return;
    };
    let dir = root.join(PRISM_PACK);
    if !dir.is_dir() {
        println!("SKIP {TEST}: RMLX_O_MODELS_ROOT does not hold {PRISM_PACK}");
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&dir));
    let entry = reg.get(PRISM_PACK).unwrap();
    assert_eq!(
        crate::tool_parser::detect_tool_call_format(
            entry.chat_template_src.as_deref(),
            &entry.arch
        ),
        Some(crate::tool_parser::ToolCallFormat::Qwen3XmlFunction)
    );
    assert!(entry.tools_supported, "the template must render a tool");

    let im_end = entry
        .tokenizer
        .as_ref()
        .unwrap()
        .token_to_id("<|im_end|>")
        .unwrap();
    let stop = entry.stop_token_ids.clone();
    assert_eq!(stop, rmlx_loader::stop_token_ids(&entry.abs_path).unwrap());
    println!("{TEST}: <|im_end|> is {im_end}, stop set {stop:?}");
    assert!(
        stop.contains(&im_end),
        "stop set {stop:?} lacks <|im_end|> {im_end}"
    );
}

/// The registry reads `generation_config.json` with the one strict reader the
/// model load uses: a file that does not parse leaves the snapshot out, as a
/// `config.json` that does not parse does, instead of a registered model that
/// every load then refuses.
#[test]
fn a_snapshot_whose_generation_config_does_not_parse_is_not_registered() {
    let tmp = tempdir().unwrap();
    let good = tmp.path().join("good");
    let bad = tmp.path().join("bad");
    for snap in [&good, &bad] {
        std::fs::create_dir_all(snap).unwrap();
        make_snapshot(snap, "LlamaForCausalLM");
    }
    std::fs::write(
        good.join("generation_config.json"),
        r#"{"temperature": 0.6}"#,
    )
    .unwrap();
    std::fs::write(bad.join("generation_config.json"), "{ not json").unwrap();
    let reg = ModelRegistry::from_paths(&[good, bad]);
    assert!(
        reg.get("good").is_some(),
        "control: a parsed file registers"
    );
    assert!(
        reg.get("bad").is_none(),
        "a malformed generation_config.json registers"
    );
}
