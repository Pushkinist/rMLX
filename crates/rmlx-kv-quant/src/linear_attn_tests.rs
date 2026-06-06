use super::*;
use rmlx_mlx::{Array, Dtype};

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn arr(vals: &[f32]) -> Array {
    let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    Array::from_bytes(&bytes, &[vals.len() as i32], Dtype::F32).expect("from_bytes")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn read(a: &Array) -> Vec<f32> {
    Array::eval(a).expect("materialise");
    a.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// snapshot() deep-clones the recurrent state; mutating the live cache
/// afterwards must NOT change the snapshot, and restore_snapshot() must
/// bring the live cache back to the snapshotted values exactly. This is
/// the spec-rollback invariant for the GDN recurrent state.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn snapshot_restore_round_trip() {
    let mut cache = LinearAttnCache::new();
    cache.conv_state = Some(arr(&[1.0, 2.0, 3.0]));
    cache.delta_state = Some(arr(&[10.0, 20.0]));

    let snap = cache.snapshot().expect("snapshot");

    // Advance the live cache (simulating a speculative draft round).
    cache.conv_state = Some(arr(&[7.0, 8.0, 9.0]));
    cache.delta_state = Some(arr(&[70.0, 80.0]));

    // Snapshot must be independent of the post-advance mutation.
    assert_eq!(read(snap.conv_state.as_ref().unwrap()), vec![1.0, 2.0, 3.0]);
    assert_eq!(read(snap.delta_state.as_ref().unwrap()), vec![10.0, 20.0]);

    // Roll back: live cache returns to the snapshotted values.
    cache.restore_snapshot(snap);
    assert_eq!(
        read(cache.conv_state.as_ref().unwrap()),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(read(cache.delta_state.as_ref().unwrap()), vec![10.0, 20.0]);
}

/// Snapshot/restore must survive `None` fields (pre-first-forward state).
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn snapshot_restore_empty() {
    let cache = LinearAttnCache::new();
    let snap = cache.snapshot().expect("snapshot empty");
    assert!(snap.conv_state.is_none());
    assert!(snap.delta_state.is_none());

    let mut live = LinearAttnCache::new();
    live.conv_state = Some(arr(&[1.0]));
    live.restore_snapshot(snap);
    assert!(live.conv_state.is_none());
    assert!(live.delta_state.is_none());
}
