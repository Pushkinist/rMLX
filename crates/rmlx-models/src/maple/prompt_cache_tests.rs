//! CPU-only regression coverage for Maple's SSD/prompt-cache gates.

use super::*;
use rmlx_kv_quant::{KvCache, KvQuant};

#[allow(clippy::unwrap_used, reason = "CPU-only fixture construction")]
fn filled_rotating(window: i32, seq: i32) -> KvCache {
    use rmlx_mlx::{Array, Device, Dtype};

    let mut cache = KvCache::with_quant_max_seq_window(KvQuant::K8V8, 4096, Some(window));
    cache.enter_prefill();
    let bytes = vec![0u8; (seq * 8 * 2) as usize];
    let k = Array::from_bytes(&bytes, &[1, 1, seq, 8], Dtype::Bf16).unwrap();
    let v = Array::from_bytes(&bytes, &[1, 1, seq, 8], Dtype::Bf16).unwrap();
    cache.update(&k, &v, Device::Cpu).unwrap();
    cache.exit_prefill(Device::Cpu).unwrap();
    cache
}

fn entry(prompt: &[u32], hydrated: bool, caches: Vec<KvCache>) -> MapleEntry {
    MapleEntry {
        prompt_token_ids: prompt.to_vec(),
        block_hashes: vec![0xfeed_u64],
        kv_caches: caches,
        first_id: if hydrated { 0 } else { 42 },
        first_piece: if hydrated { String::new() } else { "x".into() },
        exact_replay: if hydrated {
            None
        } else {
            Some(ExactReplayMetadata {
                id: 42,
                piece: "x".into(),
            })
        },
        kv_quant: Some(KvQuant::K8V8),
        is_ssd_hydrated: hydrated,
    }
}

#[test]
fn maple_is_exact_only() {
    assert_eq!(PROMPT_CACHE.policy(), ReusePolicy::ExactOnly);
}

#[test]
fn exact_reuse_requires_full_prompt_and_complete_entry() {
    let e = entry(&[1, 2, 3], false, vec![KvCache::with_quant(KvQuant::K8V8)]);
    assert!(e.is_reusable_prefix_of(&[1, 2, 3, 4], false, 1).is_none());
    assert!(!e.is_ssd_hydrated());
    assert!(e.is_hydrate_complete());
    assert_eq!(e.exact_replay().map(|r| r.id), Some(42));
}

#[test]
fn hydrated_empty_or_legacy_payload_fails_closed() {
    let empty = entry(&[1, 2, 3], true, vec![]);
    assert!(!empty.is_hydrate_complete());
    assert!(empty.exact_replay().is_none());
    assert!(empty
        .is_reusable_prefix_of(&[1, 2, 3, 4], true, 1)
        .is_none());

    // A legacy/incomplete block has geometry but no actual payload. It must
    // not become reusable merely because its metadata and hash are present.
    let legacy = entry(&[1, 2, 3], true, vec![KvCache::with_quant(KvQuant::None)]);
    assert!(!legacy.is_hydrate_complete());
}

#[test]
fn hydrated_empty_codec_storage_is_not_fake_complete() {
    let hydrated = entry(
        &[1, 2, 3, 4],
        true,
        vec![
            KvCache::with_quant(KvQuant::K8V8),
            KvCache::with_quant(KvQuant::K8V8),
        ],
    );
    assert!(!hydrated.is_hydrate_complete());
    // ExactOnly declines prefix reuse regardless; this prevents a complete
    // block-aligned hydrate from being mistaken for an exact prompt snapshot.
    assert!(hydrated
        .is_reusable_prefix_of(&[1, 2, 3, 4, 5], true, 1)
        .is_none());
}

#[test]
fn hydrated_rotating_payload_counts_as_complete() {
    let hydrated = entry(&[1, 2, 3, 4], true, vec![filled_rotating(512, 4)]);
    assert!(hydrated.is_hydrate_complete());
}

#[test]
fn exact_replay_metadata_is_preserved_by_clone() {
    let e = entry(&[1, 2, 3], false, vec![KvCache::with_quant(KvQuant::K8V8)]);
    let Ok(cloned) = e.deep_clone() else {
        panic!("empty cache clone failed");
    };
    assert_eq!(cloned.exact_replay(), e.exact_replay());
}
