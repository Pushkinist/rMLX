#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "these codec tests assert malformed-input behavior with concise panics"
)]

use super::*;
use std::collections::HashMap;
use tempfile::TempDir;

#[test]
fn v2_identity_round_trips_full_prompt_and_replay() {
    let tmp = TempDir::new().expect("temp dir");
    let path = tmp.path().join("identity.kvb");
    let layers = [KvStorage::None { max_seq: 4096 }];
    let refs: Vec<&KvStorage> = layers.iter().collect();
    let prompt_ids = vec![11, 22, 33, 44, 55, 66, 77];
    let replay = ExactReplayMetadata {
        id: 101,
        piece: " first".into(),
    };
    let identity = BlockIdentity {
        prompt_ids,
        exact_replay: Some(replay),
    };
    serialize_block_refs_timed(
        &path,
        Device::Cpu,
        "test/model",
        KvQuant::None,
        &refs,
        &[],
        &[],
        &[],
        Some(&identity),
    )
    .expect("serialize identity block");

    let reader = KvBlockReader::open(&path).expect("open identity block");
    assert_eq!(
        KvBlockReader::format_version(&reader.header().expect("header"))
            .expect("supported version"),
        2
    );
    assert_eq!(
        reader.block_identity().expect("read identity"),
        Some(identity)
    );
    assert_eq!(reader.seq_len().expect("seq len"), 0);
}

#[test]
fn unknown_format_version_fails_closed() {
    let mut meta = HashMap::new();
    meta.insert(META_FORMAT_VERSION.to_string(), "99".to_string());
    let tensor = OwnedTensor::from_u32(&[1]);
    let bytes = safetensors::serialize([("__prompt_ids".to_string(), &tensor)], Some(meta))
        .expect("serialize future block");
    let reader = KvBlockReader { bytes };
    let header = reader.header().expect("header");
    let err = KvBlockReader::format_version(&header).expect_err("future version must fail");
    assert!(err.to_string().contains("unsupported format_version 99"));
}

#[test]
fn missing_or_previous_format_version_fails_closed() {
    let tensor = OwnedTensor::from_u32(&[1]);
    for (version, expected) in [
        (None, "missing format_version"),
        (Some("1"), "unsupported format_version 1"),
    ] {
        let mut meta = HashMap::new();
        if let Some(version) = version {
            meta.insert(META_FORMAT_VERSION.to_string(), version.to_string());
        }
        let bytes = safetensors::serialize([("__prompt_ids".to_string(), &tensor)], Some(meta))
            .expect("serialize incompatible block");
        let reader = KvBlockReader { bytes };
        let header = reader.header().expect("header");
        let err = KvBlockReader::format_version(&header).expect_err("non-v2 version must fail");
        assert!(err.to_string().contains(expected));
    }
}

#[test]
fn partial_replay_metadata_fails_closed() {
    let mut meta = HashMap::new();
    meta.insert(META_FORMAT_VERSION.to_string(), FORMAT_VERSION.to_string());
    meta.insert(
        META_PROMPT_IDS_TENSOR.to_string(),
        "__prompt_ids".to_string(),
    );
    meta.insert(META_EXACT_REPLAY_ID.to_string(), "7".to_string());
    let tensor = OwnedTensor::from_u32(&[1, 2]);
    let bytes = safetensors::serialize([("__prompt_ids".to_string(), &tensor)], Some(meta))
        .expect("serialize malformed block");
    let reader = KvBlockReader { bytes };
    let err = reader
        .block_identity()
        .expect_err("partial replay must fail");
    assert!(err
        .to_string()
        .contains("exact replay id and piece must be present together"));
}

#[test]
fn empty_rotating_ring_round_trips_without_payload_tensors() {
    let tmp = TempDir::new().expect("temp dir");
    let path = tmp.path().join("empty-ring.kvb");
    let layers = [KvStorage::None { max_seq: 512 }];
    let refs: Vec<&KvStorage> = layers.iter().collect();
    let ring = RotatingStateSnapshot {
        keys: None,
        values: None,
        offset: 0,
        max_size: 512,
        keep: 0,
        valid_len: 0,
        idx: 0,
    };
    let identity = BlockIdentity {
        prompt_ids: vec![1, 2, 3, 4],
        exact_replay: None,
    };
    serialize_block_refs_timed(
        &path,
        Device::Cpu,
        "test/model",
        KvQuant::None,
        &refs,
        &[],
        &[Some(ring.clone())],
        &[],
        Some(&identity),
    )
    .expect("serialize empty ring");
    let reader = KvBlockReader::open(&path).expect("open empty ring");
    let snapshots = reader.rotating_snapshots().expect("read empty ring");
    assert_eq!(snapshots, vec![Some(ring)]);
    let (caches, _, _, _, _, _, _) = read_caches_timed_with_identity(
        &path,
        Device::Cpu,
        "test/model",
        KvQuant::None,
        DispatchPolicy::default(),
    )
    .expect("read rotating block")
    .expect("block exists");
    assert_eq!(caches.len(), 1);
    assert!(caches[0].is_rotating());
    assert_eq!(caches[0].offset(), 0);
}
