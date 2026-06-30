//! Unit tests for the `Weights` tensor-fetch helper.
//!
//! Fixtures are real on-disk `.safetensors` shards built with
//! `safetensors::serialize` + tempfile (mirroring
//! `rmlx-loader/src/tensors_tests.rs`). They exercise the index-first /
//! header-scan-fallback contract end to end:
//!   - index hit                 → `TensorLookup::Found`, no scan
//!   - name absent from index     → `NotInIndex` → header-scan fallback succeeds
//!   - index points at wrong shard→ `WrongShard` → header-scan fallback succeeds
//!   - corrupt shard header        → `Err` propagates, never masked as "absent"
//!   - `has()` reads headers, finding siblings the index omits (medgemma rule)
//!   - `scan_only` (no index) loads via header scan over `open_dir` shards

use std::path::Path;

use rmlx_loader::{load_shard_index, ShardIndex, ShardSet};
use serde_json::{json, Value};

use super::Weights;

/// Build a `ShardIndex` from explicit `(tensor, shard)` pairs — lets a test
/// model "lie" (omit a sibling, point at the wrong shard) independently of the
/// shards actually written to disk.
fn make_idx(entries: &[(&str, &str)]) -> ShardIndex {
    let weight_map = entries
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    ShardIndex {
        metadata: Value::Null,
        weight_map,
    }
}

/// Write a multi-tensor F32 `.safetensors` shard into `dir`.
///
/// Each entry is `(tensor_name, n_elems)`; values are `1.0f32` repeated. F32 is
/// host-side in `Array::from_bytes` (no Metal claim), so `array()` is safe to
/// call from a unit test.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O and serialization failures should abort the test loudly"
)]
fn write_shard(dir: &Path, filename: &str, tensors: &[(&str, usize)]) {
    // Owned backing buffers must outlive the TensorView borrows passed to serialize.
    let buffers: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, n)| {
            let mut bytes = Vec::with_capacity(n * 4);
            for _ in 0..n {
                bytes.extend_from_slice(&1.0f32.to_le_bytes());
            }
            bytes
        })
        .collect();

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .zip(buffers.iter())
        .map(|(&(name, n), buf)| {
            let tv = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![n], buf)
                .unwrap();
            (name.to_owned(), tv)
        })
        .collect();

    let bytes = safetensors::serialize(views, None).unwrap();
    std::fs::write(dir.join(filename), bytes).unwrap();
}

/// Write a corrupt `.safetensors` shard (valid 8-byte length prefix, garbage
/// header) so `SafeTensors::deserialize` fails.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O failures should abort the test loudly"
)]
fn write_corrupt_shard(dir: &Path, filename: &str) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&16u64.to_le_bytes()); // header_len = 16
    bytes.extend_from_slice(b"not valid json!!"); // 16 bytes of garbage
    std::fs::write(dir.join(filename), bytes).unwrap();
}

/// Write a real `model.safetensors.index.json` so `load_shard_index` produces a
/// truthful index (used by the index-hit path).
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O / serialization failures should abort the test loudly"
)]
fn write_index_json(dir: &Path, entries: &[(&str, &str)]) {
    let wm: serde_json::Map<String, Value> = entries
        .iter()
        .map(|(k, v)| ((*k).to_owned(), json!(v)))
        .collect();
    let body = serde_json::to_string(&json!({ "metadata": {}, "weight_map": wm })).unwrap();
    std::fs::write(dir.join("model.safetensors.index.json"), body).unwrap();
}

/// Index hit: a truthful index resolves the tensor via the index view (no scan).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn index_hit() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(dir.path(), "s.safetensors", &[("model.norm.weight", 3)]);
    write_index_json(dir.path(), &[("model.norm.weight", "s.safetensors")]);

    let idx = load_shard_index(dir.path()).unwrap();
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let w = Weights::new(&shards, &idx);

    let arr = w.array("model.norm.weight").unwrap();
    assert_eq!(arr.shape(), vec![3]);
}

/// Name absent from the index discriminates as `NotInIndex`; `array` falls back
/// to a header scan and still loads the tensor from the shard that holds it.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn not_in_index_fallback() {
    let dir = tempfile::tempdir().unwrap();
    // The tensor is physically present, but the index omits it entirely.
    write_shard(dir.path(), "s.safetensors", &[("omitted.weight", 4)]);
    // Index references some other (also present) tensor.
    write_shard(dir.path(), "other.safetensors", &[("known.weight", 2)]);
    let idx = make_idx(&[("known.weight", "other.safetensors")]);
    // Open ALL shards by dir (the medgemma-class loader pattern): the index is
    // used for speed, but the fallback can only scan shards that are open. An
    // index-driven `ShardSet::open` would not open the shard holding the omitted
    // tensor, so the helper is paired with `open_dir` for untrustworthy indexes.
    let shards = ShardSet::open_dir(dir.path()).unwrap();
    let w = Weights::new(&shards, &idx);

    // `omitted.weight` is NotInIndex → header scan finds it in s.safetensors.
    let arr = w.array("omitted.weight").unwrap();
    assert_eq!(arr.shape(), vec![4]);
}

/// Index points at the wrong shard (`WrongShard`); `array` falls back to a
/// header scan and loads from the shard that actually holds the tensor. The
/// warning is logged at the lookup source (`view_discriminated`).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn wrong_shard_fallback_warns() {
    let dir = tempfile::tempdir().unwrap();
    // The tensor physically lives in s2, but the index claims it is in s1.
    write_shard(dir.path(), "s1.safetensors", &[("decoy.weight", 1)]);
    write_shard(dir.path(), "s2.safetensors", &[("model.norm.weight", 5)]);
    let idx = make_idx(&[
        ("decoy.weight", "s1.safetensors"),
        ("model.norm.weight", "s1.safetensors"), // lie: it is really in s2
    ]);
    // Open all shards by dir so the WrongShard fallback can reach s2 (see
    // `not_in_index_fallback` — the index-driven open would not open s2).
    let shards = ShardSet::open_dir(dir.path()).unwrap();
    let w = Weights::new(&shards, &idx);

    let arr = w.array("model.norm.weight").unwrap();
    assert_eq!(arr.shape(), vec![5]);
}

/// A corrupt shard header must propagate as `Err` — never be masked by the
/// header-scan fallback as "tensor absent". This is the discrimination the
/// `view_discriminated` split exists to guarantee.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn corrupt_propagates() {
    let dir = tempfile::tempdir().unwrap();
    write_corrupt_shard(dir.path(), "corrupt.safetensors");
    // Index points the tensor at the corrupt shard → view_discriminated Errs.
    let idx = make_idx(&[("model.norm.weight", "corrupt.safetensors")]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let w = Weights::new(&shards, &idx);

    let res = w.array("model.norm.weight");
    assert!(
        res.is_err(),
        "corrupt header must propagate as Err, not fall back to a not-found scan"
    );
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("parse") || msg.contains("header"),
        "error should name a header-parse failure: {msg}"
    );
}

/// Header-SCAN-branch corruption (no index): the scan visits a corrupt shard
/// before the healthy one that holds the tensor. The parse failure must
/// propagate as `Err`, NOT be skipped to find the tensor in the later healthy
/// shard. `BTreeMap` order scans `a.safetensors` (corrupt) before
/// `b.safetensors` (healthy `t.weight`), so a `continue`-on-parse-error
/// regression would still resolve the tensor and turn this green.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / open_dir failures should abort the test loudly"
)]
fn corrupt_propagates_on_header_scan() {
    let dir = tempfile::tempdir().unwrap();
    // a < b in BTreeMap order, so the scan touches the corrupt shard first.
    write_corrupt_shard(dir.path(), "a.safetensors");
    write_shard(dir.path(), "b.safetensors", &[("t.weight", 3)]);

    // No index → scan_only goes straight to the header scan (no index branch).
    let shards = ShardSet::open_dir(dir.path()).unwrap();
    let w = Weights::scan_only(&shards);

    // `array` scans a (corrupt) before b — must Err on the parse failure even
    // though `t.weight` physically exists in the later healthy shard b.
    let arr = w.array("t.weight");
    assert!(
        arr.is_err(),
        "corrupt header on the scan branch must propagate as Err, not skip to the later healthy shard"
    );
    let msg = arr.unwrap_err().to_string();
    assert!(
        msg.contains("parse") || msg.contains("header"),
        "error should name a header-parse failure: {msg}"
    );

    // The same propagation must hold for the existence path.
    assert!(
        w.has("t.weight").is_err(),
        "has() must propagate the corrupt-header parse failure, not mask it as absent"
    );
}

/// `has()` is header-based and finds `.scales`/`.biases` siblings the index
/// omits (the medgemma rule). The index lists only `.weight`; the shard header
/// carries the siblings.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn has_finds_index_omitted_siblings() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "s.safetensors",
        &[
            ("layers.0.mlp.weight", 8),
            ("layers.0.mlp.scales", 1),
            ("layers.0.mlp.biases", 1),
        ],
    );
    // The index lists ONLY the weight — siblings are omitted, as on medgemma.
    let idx = make_idx(&[("layers.0.mlp.weight", "s.safetensors")]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let w = Weights::new(&shards, &idx);

    assert!(
        w.has("layers.0.mlp.weight").unwrap(),
        "weight present in header"
    );
    assert!(
        w.has("layers.0.mlp.scales").unwrap(),
        "scales sibling found by header scan even though the index omits it"
    );
    assert!(
        w.has("layers.0.mlp.biases").unwrap(),
        "biases sibling found by header scan even though the index omits it"
    );
    assert!(
        !w.has("layers.0.mlp.absent").unwrap(),
        "truly-absent name is false"
    );
}

/// `scan_only` (no index) loads tensors via `ShardSet::open_dir` + a pure header
/// scan — the index-less path (qwen3_vl_moe / gemma3 class).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / open_dir failures should abort the test loudly"
)]
fn scan_only_works_without_index() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(dir.path(), "a.safetensors", &[("attn.q_proj.weight", 6)]);
    write_shard(dir.path(), "b.safetensors", &[("attn.k_proj.weight", 6)]);
    // No model.safetensors.index.json written — open_dir globs the files.
    let shards = ShardSet::open_dir(dir.path()).unwrap();
    let w = Weights::scan_only(&shards);

    assert_eq!(w.array("attn.q_proj.weight").unwrap().shape(), vec![6]);
    assert_eq!(w.array("attn.k_proj.weight").unwrap().shape(), vec![6]);
    assert!(w.has("attn.k_proj.weight").unwrap());
    assert!(!w.has("attn.v_proj.weight").unwrap());
    // A truly-absent tensor is a hard error (not a silent miss).
    assert!(w.array("attn.v_proj.weight").is_err());
}

/// `linear()` maps onto the shared `layers::Linear`: a `.scales`-less base is
/// `Plain`; a base with `.scales` + `.biases` is `Quantized { biases: Some }`;
/// `.scales` without `.biases` is `Quantized { biases: None }`. Sibling
/// detection is header-based (the index lists only the `.weight` keys here).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn linear_maps_to_shared_linear() {
    use crate::layers::{Linear, QuantParams};

    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "s.safetensors",
        &[
            // Plain: weight only.
            ("plain.weight", 4),
            // Affine: weight + scales + biases.
            ("affine.weight", 8),
            ("affine.scales", 1),
            ("affine.biases", 1),
            // Mxfp-style: weight + scales, no biases.
            ("mxfp.weight", 8),
            ("mxfp.scales", 1),
        ],
    );
    // Index lists only the `.weight` keys; siblings resolve via header scan.
    let idx = make_idx(&[
        ("plain.weight", "s.safetensors"),
        ("affine.weight", "s.safetensors"),
        ("mxfp.weight", "s.safetensors"),
    ]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let w = Weights::new(&shards, &idx);

    // qp closure receives header-detected has_biases.
    let qp_affine = |_has_biases: bool| Ok(QuantParams::global(32, 8, "affine"));

    match w.linear("plain", qp_affine).unwrap() {
        Linear::Plain { weight } => assert_eq!(weight.shape(), vec![4]),
        Linear::Quantized { .. } | Linear::Paro { .. } => {
            panic!("expected Plain for scales-less base")
        }
    }

    match w.linear("affine", qp_affine).unwrap() {
        Linear::Quantized { biases, bits, .. } => {
            assert!(biases.is_some(), "affine base has a .biases sibling");
            assert_eq!(bits, 8);
        }
        Linear::Plain { .. } | Linear::Paro { .. } => panic!("expected Quantized for affine base"),
    }

    match w.linear("mxfp", qp_affine).unwrap() {
        Linear::Quantized { biases, .. } => {
            assert!(biases.is_none(), "mxfp base has no .biases sibling");
        }
        Linear::Plain { .. } | Linear::Paro { .. } => panic!("expected Quantized for mxfp base"),
    }
}

/// The `qp` closure may hard-error (config/data contradiction); `linear`
/// propagates that error rather than building a `Linear`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn linear_propagates_qp_error() {
    use rmlx_core::error::Error;

    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "s.safetensors",
        &[("q.weight", 8), ("q.scales", 1), ("q.biases", 1)],
    );
    let idx = make_idx(&[("q.weight", "s.safetensors")]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let w = Weights::new(&shards, &idx);

    let qp_err =
        |_has_biases: bool| Err(Error::Loader("config contradicts tensor data".to_owned()));
    let res = w.linear("q", qp_err);
    assert!(res.is_err(), "qp hard-error must propagate out of linear()");
}

/// `classify_load_oom` promotes an allocation-failure MLX error to `Error::Oom`
/// and leaves non-alloc errors untouched.
#[test]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn classify_load_oom_alloc_string_becomes_oom() {
    use rmlx_core::{Error, OomPhase};
    // The exact shape MLX throws on a Metal allocation failure.
    let e = Error::Mlx("Array::eval: [malloc_or_wait] Unable to allocate 9000000000 bytes".into());
    match super::classify_load_oom(e) {
        Error::Oom { phase, .. } => assert_eq!(phase, OomPhase::LoadWeights),
        other => panic!("expected Error::Oom, got {other:?}"),
    }
}

#[test]
fn classify_load_oom_shape_error_stays_mlx() {
    use rmlx_core::Error;
    // A non-OOM MLX failure must NOT be misclassified as OOM.
    let e = Error::Mlx("reshape: total size mismatch 12 vs 16".into());
    assert!(
        matches!(super::classify_load_oom(e), Error::Mlx(_)),
        "shape error must stay Error::Mlx"
    );
}

#[test]
fn classify_load_oom_non_mlx_untouched() {
    use rmlx_core::Error;
    let e = Error::Loader("missing config.json".into());
    assert!(matches!(super::classify_load_oom(e), Error::Loader(_)));
}

/// `embedding()` maps onto `Embedding`: a `.scales`-less base is `Plain`; a base
/// with `.scales` + `.biases` is `Quantized { biases: Some }` (affine); `.scales`
/// without `.biases` is `Quantized { biases: None }` (mxfp8-style). Sibling
/// detection is header-based — index lists only `.weight` keys.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn embedding_maps_to_shared_embedding() {
    use crate::layers::{Embedding, QuantParams};

    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "s.safetensors",
        &[
            // Plain: weight only.
            ("plain.weight", 4),
            // Affine: weight + scales + biases.
            ("affine.weight", 8),
            ("affine.scales", 1),
            ("affine.biases", 1),
            // Mxfp8-style: weight + scales, no biases.
            ("mxfp.weight", 8),
            ("mxfp.scales", 1),
        ],
    );
    // Index lists only the `.weight` keys; siblings resolve via header scan.
    let idx = make_idx(&[
        ("plain.weight", "s.safetensors"),
        ("affine.weight", "s.safetensors"),
        ("mxfp.weight", "s.safetensors"),
    ]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let w = Weights::new(&shards, &idx);

    let qp = |_has_biases: bool| Ok(QuantParams::global(32, 8, "affine"));

    // Plain fallback: no .scales sibling.
    match w.embedding("plain", qp).unwrap() {
        Embedding::Plain { weight } => assert_eq!(weight.shape(), vec![4]),
        Embedding::Quantized { .. } => panic!("expected Plain for scales-less base"),
    }

    // Affine: .scales + .biases present → biases: Some.
    match w.embedding("affine", qp).unwrap() {
        Embedding::Quantized { biases, bits, .. } => {
            assert!(biases.is_some(), "affine base has a .biases sibling");
            assert_eq!(bits, 8);
        }
        Embedding::Plain { .. } => panic!("expected Quantized for affine base"),
    }

    // Mxfp8-style: .scales present, .biases absent → biases: None.
    match w.embedding("mxfp", qp).unwrap() {
        Embedding::Quantized { biases, .. } => {
            assert!(biases.is_none(), "mxfp base has no .biases sibling");
        }
        Embedding::Plain { .. } => panic!("expected Quantized for mxfp base"),
    }
}

/// `bf16_scales` must leave a uint8 E8M0 scale tensor (mxfp8 / mxfp4) untouched:
/// MLX's `dequantize` rejects any scale dtype other than uint8 for those modes,
/// so casting the shared per-block exponent to bf16 corrupts it and crashes the
/// kernel at first prefill. The dtype gate is the general rule — it follows the
/// per-tensor checkpoint fact, not the arch string.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: Array construction failures should abort the test loudly"
)]
fn bf16_scales_keeps_uint8_e8m0_intact() {
    use rmlx_mlx::{Array, Dtype};

    // A uint8 E8M0 scale block as mxfp8 ships it: one shared exponent per group.
    let u8_scales = Array::from_bytes(&[127u8, 128, 129, 130], &[4], Dtype::U8).unwrap();
    let out = super::bf16_scales(u8_scales).unwrap();
    assert_eq!(
        out.dtype(),
        Dtype::U8,
        "mxfp8/mxfp4 uint8 E8M0 scales must survive the loader as uint8"
    );
}

/// `bf16_scales` must still apply the float-uniformity cast for affine float
/// scales: an fp16/f32 scale would otherwise promote `quantized_matmul` output
/// to f32 and leak into the KV cache. Only non-float scales are exempt.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: Array construction failures should abort the test loudly"
)]
fn bf16_scales_casts_float_scales_to_bf16() {
    use rmlx_mlx::{Array, Dtype};

    // An f32 affine scale tensor: must be lifted to bf16 (uniformity discipline).
    let f32_scales = Array::from_f32_slice(&[0.5, 0.25, 0.125, 1.0], &[4]).unwrap();
    let out = super::bf16_scales(f32_scales).unwrap();
    assert_eq!(
        out.dtype(),
        Dtype::Bf16,
        "affine float scales keep the bf16 uniformity cast"
    );
}

/// The `qp` closure may hard-error; `embedding` propagates the error rather
/// than building an `Embedding`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open failures should abort the test loudly"
)]
fn embedding_propagates_qp_error() {
    use rmlx_core::error::Error;

    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "s.safetensors",
        &[("emb.weight", 8), ("emb.scales", 1), ("emb.biases", 1)],
    );
    let idx = make_idx(&[("emb.weight", "s.safetensors")]);
    let shards = ShardSet::open(dir.path(), &idx).unwrap();
    let w = Weights::new(&shards, &idx);

    let qp_err =
        |_has_biases: bool| Err(Error::Loader("config contradicts tensor data".to_owned()));
    let res = w.embedding("emb", qp_err);
    assert!(
        res.is_err(),
        "qp hard-error must propagate out of embedding()"
    );
}
