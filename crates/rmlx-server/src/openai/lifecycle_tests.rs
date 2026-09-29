//! Unit tests for the `GET /v1/models` entry builder.

use std::collections::HashMap;
use std::path::PathBuf;

use super::{model_entries, ResidentInfo};
use crate::registry::ModelRegistry;

/// A registry of two snapshots, `gen` and `embed`, each only a `config.json`.
#[allow(
    clippy::expect_used,
    reason = "test fixture setup; a failed write is a broken test, not a result"
)]
fn two_model_registry() -> (ModelRegistry, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("tempdir");
    let mut paths: Vec<PathBuf> = Vec::new();
    for (name, arch) in [
        ("gen", "Qwen3ForCausalLM"),
        ("embed", "JinaEmbeddingsV4Model"),
    ] {
        let dir = root.path().join(name);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let cfg = serde_json::json!({ "architectures": [arch] });
        std::fs::write(dir.join("config.json"), cfg.to_string()).expect("write config.json");
        paths.push(dir);
    }
    (ModelRegistry::from_paths(&paths), root)
}

fn loaded_by_id(entries: &[serde_json::Value]) -> HashMap<String, bool> {
    entries
        .iter()
        .filter_map(|e| Some((e["id"].as_str()?.to_owned(), e["loaded"].as_bool()?)))
        .collect()
}

/// A model resident in the embedding slot is reported as loaded, with no
/// slot timestamps, and the idle generation model is not.
#[test]
fn a_resident_embedding_model_is_loaded() {
    let (registry, _root) = two_model_registry();
    let entries = model_entries(&registry, &HashMap::new(), Some("embed"));
    let loaded = loaded_by_id(&entries);
    assert_eq!(loaded.get("embed"), Some(&true), "{entries:?}");
    assert_eq!(loaded.get("gen"), Some(&false), "{entries:?}");
    assert!(
        entries.iter().all(|e| e.get("loaded_at").is_none()),
        "{entries:?}"
    );
}

/// A model in a generation slot is loaded and carries its timestamps; with
/// the embedding slot empty, the embedding model is not loaded.
#[test]
fn a_generation_slot_is_loaded_and_an_empty_embedding_slot_is_not() {
    let (registry, _root) = two_model_registry();
    let resident = HashMap::from([(
        "gen".to_owned(),
        ResidentInfo {
            loaded_at: 10,
            last_used: 20,
            context: None,
        },
    )]);
    let entries = model_entries(&registry, &resident, None);
    let loaded = loaded_by_id(&entries);
    assert_eq!(loaded.get("gen"), Some(&true), "{entries:?}");
    assert_eq!(loaded.get("embed"), Some(&false), "{entries:?}");
    let gen = entries.iter().find(|e| e["id"] == "gen");
    assert_eq!(gen.map(|e| e["last_used"].clone()), Some(20.into()));
}
