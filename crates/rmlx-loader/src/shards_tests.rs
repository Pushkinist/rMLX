use super::*;
use rmlx_core::mapped_pages::file_pages_in_cache;
use serde_json::json;

/// Serializes the tests that map shards or read the page cache, so one test's
/// large mapping cannot change what another measures.
static SHARD_TESTS: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
    SHARD_TESTS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Reads every byte, so the pages behind `bytes` are resident.
fn touch(bytes: &[u8]) {
    std::hint::black_box(bytes.iter().fold(0u64, |sum, &b| sum + u64::from(b)));
}

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
    File::create(&path)
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
    File::create(&path)
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
    File::create(&path)
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

/// Write a one-tensor `.safetensors` shard into `dir`. The tensor is a tiny
/// F32 scalar — only the header matters for shard discovery.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O and serialization failures should abort the test loudly"
)]
fn write_shard(dir: &Path, filename: &str, tensor_name: &str) {
    let data: [u8; 4] = 1.0f32.to_le_bytes();
    let tv = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &data).unwrap();
    let bytes = safetensors::serialize([(tensor_name.to_owned(), tv)], None).unwrap();
    std::fs::write(dir.join(filename), bytes).unwrap();
}

/// `open_dir` discovers every `*.safetensors` shard by glob, ignoring any index
/// and any non-`.safetensors` file in the directory.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / open_dir failures should abort the test loudly"
)]
fn open_dir_discovers_all_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(dir.path(), "a.safetensors", "tensor.a");
    write_shard(dir.path(), "b.safetensors", "tensor.b");
    // Decoy non-shard files that must be ignored.
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    std::fs::write(dir.path().join("notes.txt"), b"hi").unwrap();

    let shards = ShardSet::open_dir(dir.path()).unwrap();
    assert_eq!(shards.len(), 2, "exactly the two .safetensors shards");
    assert!(shards.get("a.safetensors").is_some());
    assert!(shards.get("b.safetensors").is_some());
    assert!(
        shards.get("config.json").is_none(),
        "non-shard files must not be opened"
    );
}

/// `open_dir` on a directory with no `*.safetensors` shard is an error, not an
/// empty set — an index-less load against an empty dir must fail loudly.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir creation failure should abort the test loudly"
)]
fn open_dir_empty_is_err() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
    let result = ShardSet::open_dir(dir.path());
    assert!(result.is_err(), "no shards present must yield Err");
}

/// With no index, every tensor of `model.safetensors` maps to that file, and
/// a `model.safetensors` whose header does not parse is an error.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / index failures should abort the test loudly"
)]
fn a_single_shard_without_index_maps_each_tensor_to_it() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(dir.path(), "model.safetensors", "tensor.a");
    write_shard(dir.path(), "other.safetensors", "tensor.b");
    let idx = load_shard_index(dir.path()).unwrap();
    assert_eq!(
        idx.weight_map.into_iter().collect::<Vec<_>>(),
        vec![("tensor.a".to_owned(), "model.safetensors".to_owned())]
    );
    assert_eq!(idx.metadata, Value::Null);

    std::fs::write(dir.path().join("model.safetensors"), b"not a header").unwrap();
    assert!(load_shard_index(dir.path()).is_err());
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

/// An index names each shard by a plain `.safetensors` file name in the
/// snapshot directory. A value with a directory part (a trailing `/`
/// included), an absolute path, `.`, `..`, the empty name, a NUL byte or
/// another extension is refused before any loader opens it.
#[test]
fn an_index_names_its_shards_by_plain_file_name() {
    let cases = [
        ("model-00001-of-00002.safetensors", true),
        ("../x.safetensors", false),
        ("/tmp/x.safetensors", false),
        ("sub/x.safetensors", false),
        ("./x.safetensors", false),
        ("x.bin", false),
        ("", false),
        (".", false),
        ("..", false),
        ("x.safetensors/", false),
        ("x.safetensors/.", false),
        ("x\0y.safetensors", false),
    ];
    for (shard, accepted) in cases {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            make_index_json(&[("t.weight", shard)]),
        )
        .unwrap();
        let r = load_shard_index(dir.path());
        assert_eq!(r.is_ok(), accepted, "{shard}: {r:?}");
        if let Err(e) = r {
            assert!(
                e.to_string().contains(&format!("{shard:?}")),
                "{shard:?}: {e}"
            );
        }
    }
}

/// Every shard index under `RMLX_O_MODELS_ROOT` is accepted.
#[test]
fn every_local_snapshot_index_is_accepted() {
    const TEST: &str = "every_local_snapshot_index_is_accepted";
    let Some(root) = std::env::var("RMLX_O_MODELS_ROOT")
        .ok()
        .filter(|r| !r.is_empty())
        .map(PathBuf::from)
        .filter(|r| r.is_dir())
    else {
        println!("SKIP {TEST}: RMLX_O_MODELS_ROOT names no directory");
        return;
    };
    let mut indexed = 0;
    let mut wrong = Vec::new();
    for entry in std::fs::read_dir(&root).unwrap() {
        let dir = entry.unwrap().path();
        if !dir.join("model.safetensors.index.json").is_file() {
            continue;
        }
        indexed += 1;
        if let Err(e) = load_shard_index(&dir) {
            wrong.push(format!("{}: {e}", dir.display()));
        }
    }
    println!("{TEST}: {indexed} indexed snapshots");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

fn pattern(len: usize, seed: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 31 + seed) % 251) as u8).collect()
}

/// A one-shard snapshot holding `tensors` as `U8` vectors, flushed so its
/// page-cache pages are clean.
fn write_u8_snapshot(dir: &Path, tensors: &[(&str, &[u8])]) {
    let views: Vec<(&str, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .map(|&(name, data)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::U8,
                vec![data.len()],
                data,
            )
            .unwrap();
            (name, view)
        })
        .collect();
    let path = dir.join(SINGLE_SHARD);
    safetensors::serialize_to_file(views, None, &path).unwrap();
    File::open(&path).unwrap().sync_all().unwrap();
}

#[test]
fn release_mapped_keeps_every_tensor_readable() {
    let _serial = serial();
    let page = page_size().unwrap();
    let big = pattern(5 * page + 123, 1);
    let small = pattern(100, 2);
    let straddle = pattern(page + 200, 3);
    let dir = tempfile::tempdir().unwrap();
    write_u8_snapshot(
        dir.path(),
        &[
            ("a_big", &big),
            ("b_small", &small),
            ("c_straddle", &straddle),
        ],
    );
    let idx = load_shard_index(dir.path()).unwrap();
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let big_view = crate::tensors::view(&shards, &idx, "a_big").unwrap();
    let small_view = crate::tensors::view(&shards, &idx, "b_small").unwrap();
    let straddle_view = crate::tensors::view(&shards, &idx, "c_straddle").unwrap();

    let big_copy = big_view.bytes.to_vec();
    let released = release_mapped(big_view.bytes);
    assert!(released > 0, "a tensor of five pages released nothing");
    assert_eq!(released % page, 0);
    assert!(released <= big.len());
    assert!(released > big.len() - 2 * page);
    assert_eq!(big_copy, big);

    let straddle_released = release_mapped(straddle_view.bytes);
    assert_eq!(straddle_released % page, 0);
    assert!(straddle_released <= straddle.len());
    assert_eq!(release_mapped(small_view.bytes), 0);

    assert_eq!(big_view.bytes, big.as_slice());
    assert_eq!(small_view.bytes, small.as_slice());
    assert_eq!(straddle_view.bytes, straddle.as_slice());
    let fresh = crate::tensors::view(&shards, &idx, "a_big").unwrap();
    assert_eq!(fresh.bytes, big.as_slice());
}

#[test]
fn release_mapped_twice_rereads_the_same_bytes() {
    let _serial = serial();
    let page = page_size().unwrap();
    let big = pattern(4 * page, 7);
    let dir = tempfile::tempdir().unwrap();
    write_u8_snapshot(dir.path(), &[("w", &big)]);
    let idx = load_shard_index(dir.path()).unwrap();
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let view = crate::tensors::view(&shards, &idx, "w").unwrap();

    let first = release_mapped(view.bytes);
    assert!(first > 0);
    assert_eq!(view.bytes, big.as_slice());
    assert_eq!(release_mapped(view.bytes), first);
    assert_eq!(view.bytes, big.as_slice());
}

/// A synthetic registry: A = [0x1000, 0x5000) and B = [0x5000, 0x9000).
fn two_adjacent_mappings() -> BTreeMap<usize, LiveMapping> {
    [(0x1000, 0x4000), (0x5000, 0x4000)]
        .into_iter()
        .map(|(base, len)| {
            (
                base,
                LiveMapping {
                    len,
                    path: PathBuf::new(),
                    warned: false,
                },
            )
        })
        .collect()
}

#[test]
fn containing_mapping_accepts_a_slice_ending_at_the_mapping_end() {
    let map = two_adjacent_mappings();
    assert_eq!(containing_mapping(&map, 0x2000, 0x5000), Some(0x1000));
    assert_eq!(containing_mapping(&map, 0x1000, 0x5000), Some(0x1000));
}

#[test]
fn containing_mapping_refuses_a_slice_one_byte_past_the_mapping_end() {
    let map = two_adjacent_mappings();
    assert_eq!(containing_mapping(&map, 0x2000, 0x5001), None);
}

#[test]
fn containing_mapping_refuses_a_slice_below_every_mapping() {
    let map = two_adjacent_mappings();
    assert_eq!(containing_mapping(&map, 0x500, 0x800), None);
}

#[test]
fn containing_mapping_refuses_a_slice_across_two_mappings() {
    let map = two_adjacent_mappings();
    assert_eq!(containing_mapping(&map, 0x4000, 0x6000), None);
}

#[test]
fn containing_mapping_finds_the_second_mapping() {
    let map = two_adjacent_mappings();
    assert_eq!(containing_mapping(&map, 0x5000, 0x9000), Some(0x5000));
    assert_eq!(containing_mapping(&map, 0x6000, 0x7000), Some(0x5000));
}

#[test]
fn release_mapped_ignores_bytes_outside_a_shard_mapping() {
    let _serial = serial();
    let page = page_size().unwrap();
    let heap = pattern(4 * page, 9);
    assert_eq!(release_mapped(&heap), 0);
    assert_eq!(heap, pattern(4 * page, 9));

    let dir = tempfile::tempdir().unwrap();
    let other_path = dir.path().join("other.bin");
    let other_data = pattern(8 * page, 11);
    std::fs::write(&other_path, &other_data).unwrap();
    // SAFETY: the file is private to this test and not modified while mapped.
    let other = unsafe { memmap2::Mmap::map(&File::open(&other_path).unwrap()) }.unwrap();
    assert_eq!(release_mapped(&other), 0);
    assert_eq!(&other[..], other_data.as_slice());
}

#[test]
fn release_mapped_evicts_the_pages_from_the_page_cache() {
    const LEN: usize = 64 << 20;
    let _serial = serial();
    let page = page_size().unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_u8_snapshot(dir.path(), &[("w", &pattern(LEN, 5))]);
    let path = dir.path().join(SINGLE_SHARD);
    let idx = load_shard_index(dir.path()).unwrap();
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let base = shards.get(SINGLE_SHARD).unwrap().as_bytes().as_ptr() as usize;
    let view = crate::tensors::view(&shards, &idx, "w").unwrap();
    assert_eq!(view.bytes, pattern(LEN, 5).as_slice());

    let start = view.bytes.as_ptr() as usize;
    let pages = inner_page_range(start, start + LEN, page).unwrap();
    let file_pages = (pages.start - base) / page..(pages.end - base) / page;
    touch(view.bytes);
    let cached = file_pages_in_cache(&path).unwrap();
    assert!(
        cached[file_pages.clone()].iter().all(|&c| c),
        "the tensor's pages are not in the page cache before the release"
    );

    touch(view.bytes);
    let touched = rmlx_core::mach_mem::read_proc_mem().unwrap().external_bytes;
    let released = release_mapped(view.bytes);
    let after = rmlx_core::mach_mem::read_proc_mem().unwrap().external_bytes;
    assert_eq!(released, pages.len());

    let cached = file_pages_in_cache(&path).unwrap();
    let still_cached = cached[file_pages.clone()].iter().filter(|&&c| c).count();
    assert_eq!(
        still_cached,
        0,
        "{still_cached} of {} released pages are still in the page cache",
        file_pages.len()
    );
    let dropped = touched.saturating_sub(after);
    assert!(
        dropped >= (LEN as u64) * 3 / 4,
        "file-backed memory fell by {dropped} bytes after releasing {released}"
    );

    assert_eq!(view.bytes, pattern(LEN, 5).as_slice());
    let again = crate::tensors::view(&shards, &idx, "w").unwrap();
    assert_eq!(again.bytes, pattern(LEN, 5).as_slice());
}
