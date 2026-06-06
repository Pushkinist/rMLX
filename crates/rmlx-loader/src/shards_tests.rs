use super::*;
use serde_json::json;

fn make_index_json(weight_map: &[(&str, &str)]) -> String {
    let wm: serde_json::Map<String, Value> = weight_map
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    serde_json::to_string(&json!({
        "metadata": {"format": "pt", "total_size": 1234},
        "weight_map": wm
    }))
    .unwrap()
}

/// Build a JSON index with `n` distinct tensor entries, all pointing to one shard.
fn make_large_index_json(n: usize) -> String {
    let wm: serde_json::Map<String, Value> = (0..n)
        .map(|i| (format!("tensor_{i}"), json!("model.safetensors")))
        .collect();
    serde_json::to_string(&json!({
        "metadata": {},
        "weight_map": wm
    }))
    .unwrap()
}

/// Build a JSON index with entries spanning `n` distinct shard filenames.
fn make_many_shards_index_json(n: usize) -> String {
    let wm: serde_json::Map<String, Value> = (0..n)
        .map(|i| {
            (
                format!("tensor_{i}"),
                json!(format!("shard_{i:05}.safetensors")),
            )
        })
        .collect();
    serde_json::to_string(&json!({
        "metadata": {},
        "weight_map": wm
    }))
    .unwrap()
}

// Oversized weight_map must be rejected.
#[test]
fn too_many_tensors_returns_err() {
    use std::io::Write;
    let n = MAX_TENSORS + 1;
    let json_str = make_large_index_json(n);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors.index.json");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(json_str.as_bytes())
        .unwrap();
    let result = load_shard_index(dir.path());
    assert!(result.is_err(), "expected Err for {n} tensors");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("MAX_TENSORS") || msg.contains(&MAX_TENSORS.to_string()),
        "error message should mention the limit: {msg}"
    );
}

// weight_map within bound must parse successfully.
#[test]
fn normal_weight_map_parses_ok() {
    use std::io::Write;
    let json_str = make_large_index_json(10);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors.index.json");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(json_str.as_bytes())
        .unwrap();
    let result = load_shard_index(dir.path());
    assert!(
        result.is_ok(),
        "expected Ok for 10 tensors: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().weight_map.len(), 10);
}

// Oversized shard count must be rejected.
#[test]
fn too_many_shards_returns_err() {
    use std::io::Write;
    let n = MAX_SHARDS + 1;
    let json_str = make_many_shards_index_json(n);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors.index.json");
    std::fs::File::create(&path)
        .unwrap()
        .write_all(json_str.as_bytes())
        .unwrap();
    let result = load_shard_index(dir.path());
    assert!(result.is_err(), "expected Err for {n} shards");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("MAX_SHARDS") || msg.contains(&MAX_SHARDS.to_string()),
        "error message should mention the limit: {msg}"
    );
}

#[test]
fn parses_weight_map() {
    let json_str = make_index_json(&[
        (
            "model.layers.0.attn.q_proj.weight",
            "model-00001-of-00002.safetensors",
        ),
        (
            "model.layers.0.attn.k_proj.weight",
            "model-00001-of-00002.safetensors",
        ),
        (
            "model.layers.1.attn.q_proj.weight",
            "model-00002-of-00002.safetensors",
        ),
    ]);
    let data = json_str.as_bytes().to_vec();

    // Parse inline via internal helper — replicate logic without fs.
    let mut root: serde_json::Map<String, Value> = serde_json::from_slice(&data).unwrap();
    let metadata = root.remove("metadata").unwrap_or(Value::Null);
    let raw_map = root.remove("weight_map").unwrap();
    let weight_map: BTreeMap<String, String> = raw_map
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
        .collect();

    let idx = ShardIndex {
        metadata,
        weight_map,
    };
    assert_eq!(idx.weight_map.len(), 3);

    let counts = count_tensors_per_shard(&idx);
    assert_eq!(counts["model-00001-of-00002.safetensors"], 2);
    assert_eq!(counts["model-00002-of-00002.safetensors"], 1);
}

#[test]
fn count_tensors_per_shard_basic() {
    let mut weight_map = BTreeMap::new();
    weight_map.insert("a".to_owned(), "shard1.safetensors".to_owned());
    weight_map.insert("b".to_owned(), "shard1.safetensors".to_owned());
    weight_map.insert("c".to_owned(), "shard2.safetensors".to_owned());
    let idx = ShardIndex {
        metadata: Value::Null,
        weight_map,
    };
    let counts = count_tensors_per_shard(&idx);
    assert_eq!(counts["shard1.safetensors"], 2);
    assert_eq!(counts["shard2.safetensors"], 1);
}
