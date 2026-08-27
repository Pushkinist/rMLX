use super::*;
use rmlx_mlx::Dtype;

/// Deterministic `[B=1, kv_h=1, S, D=2]` f32 tensor; cell value
/// `base + position(+0.5)` so two tensors are trivially comparable.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn kv(s: i32, base: f32) -> Array {
    let mut data: Vec<f32> = Vec::with_capacity((s * 2) as usize);
    for p in 0..s {
        data.push(base + p as f32);
        data.push(base + p as f32 + 0.5);
    }
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, &[1, 1, s, 2], Dtype::F32).unwrap()
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn host(a: &Array) -> Vec<f32> {
    let e = a.try_clone().unwrap();
    e.eval().unwrap();
    e.to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Drive a ring past its window (wrapped: `offset >= max_size`), snapshot,
/// then prove (a) snapshot buffers+meta == source, (b) an identical decode
/// step after `restore` into a fresh (deliberately wrong-sized) ring
/// yields byte-identical K/V and meta to the un-snapshotted ring.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn snapshot_restore_roundtrip_wrapped_exact() {
    let device = Device::Cpu;
    let max_size = 4;

    let mut a = RotatingState::new(max_size);
    a.update_and_fetch(&kv(6, 0.0), &kv(6, 100.0), device)
        .unwrap();
    for step in 0..3 {
        let p = (6 + step) as f32;
        a.update_and_fetch(&kv(1, p), &kv(1, 100.0 + p), device)
            .unwrap();
    }
    assert!(a.offset >= a.max_size, "ring must be wrapped for this test");

    let snap = a.snapshot().unwrap();
    // (a) snapshot is an exact copy of the source.
    assert_eq!(
        host(snap.keys.as_ref().unwrap()),
        host(a.keys.as_ref().unwrap())
    );
    assert_eq!(
        host(snap.values.as_ref().unwrap()),
        host(a.values.as_ref().unwrap())
    );
    assert_eq!(snap.offset, a.offset);
    assert_eq!(snap.idx, a.idx);
    assert_eq!(snap.max_size, a.max_size);
    assert_eq!(snap.keep, a.keep);

    // Reference: one more decode step on the source ring.
    let (rk, rv) = a
        .update_and_fetch(&kv(1, 42.0), &kv(1, 142.0), device)
        .unwrap();
    let (rk, rv) = (host(&rk), host(&rv));
    let (a_off, a_idx) = (a.offset, a.idx);

    // Restore into a fresh, deliberately wrong-sized ring; restore must
    // overwrite max_size too. Drive the SAME step.
    let mut b = RotatingState::new(999);
    b.restore(&snap).unwrap();
    assert_eq!(b.max_size, max_size, "restore must overwrite max_size");
    assert_eq!(b.offset, snap.offset);
    assert_eq!(b.idx, snap.idx);
    let (bk, bv) = b
        .update_and_fetch(&kv(1, 42.0), &kv(1, 142.0), device)
        .unwrap();
    assert_eq!(host(&bk), rk, "K after restore+decode must equal reference");
    assert_eq!(host(&bv), rv, "V after restore+decode must equal reference");
    assert_eq!(b.offset, a_off, "offset must match reference after step");
    assert_eq!(b.idx, a_idx, "idx must match reference after step");
}

/// Non-wrapped round-trip: prefill below the window, snapshot, restore,
/// continue — decode-after byte-identical.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn snapshot_restore_roundtrip_non_wrapped() {
    let device = Device::Cpu;
    let max_size = 16;

    let mut a = RotatingState::new(max_size);
    a.update_and_fetch(&kv(5, 0.0), &kv(5, 50.0), device)
        .unwrap();
    assert!(a.offset < a.max_size, "must be non-wrapped");

    let snap = a.snapshot().unwrap();
    let (rk, _) = a
        .update_and_fetch(&kv(1, 7.0), &kv(1, 57.0), device)
        .unwrap();
    let rk = host(&rk);

    let mut b = RotatingState::new(max_size);
    b.restore(&snap).unwrap();
    assert_eq!(b.offset, 5);
    let (bk, _) = b
        .update_and_fetch(&kv(1, 7.0), &kv(1, 57.0), device)
        .unwrap();
    assert_eq!(host(&bk), rk);
}

/// A multi-token tail after restore (the B1 prefix path forwards the new
/// suffix in one `_update_concat` call, not one token at a time) must
/// equal the same multi-token append on the un-snapshotted ring.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn snapshot_restore_multitoken_tail_wrapped() {
    let device = Device::Cpu;
    let max_size = 4;

    let mut a = RotatingState::new(max_size);
    a.update_and_fetch(&kv(7, 0.0), &kv(7, 70.0), device)
        .unwrap();
    assert!(a.offset >= a.max_size);

    let snap = a.snapshot().unwrap();
    // Reference: append a 3-token tail in one concat call.
    let (rk, rv) = a
        .update_and_fetch(&kv(3, 200.0), &kv(3, 270.0), device)
        .unwrap();
    let (rk, rv) = (host(&rk), host(&rv));

    let mut b = RotatingState::new(999);
    b.restore(&snap).unwrap();
    let (bk, bv) = b
        .update_and_fetch(&kv(3, 200.0), &kv(3, 270.0), device)
        .unwrap();
    assert_eq!(host(&bk), rk, "multi-token tail K must equal reference");
    assert_eq!(host(&bv), rv, "multi-token tail V must equal reference");
    assert_eq!(b.offset, a.offset);
    assert_eq!(b.idx, a.idx);
}

#[test]
fn persistent_snapshot_roundtrip_and_next_append_wrapped() {
    let device = Device::Cpu;
    let mut source = RotatingState::new(4);
    source
        .update_and_fetch(&kv(7, 0.0), &kv(7, 70.0), device)
        .unwrap();
    let persisted = source.snapshot_persistent().unwrap();
    assert_eq!(persisted.valid_len, 4);
    assert_eq!(persisted.offset, 7);
    assert_eq!(persisted.idx, source.idx);

    let expected = source
        .update_and_fetch(&kv(1, 9.0), &kv(1, 79.0), device)
        .map(|(k, v)| (host(&k), host(&v), source.offset, source.idx))
        .unwrap();
    let mut restored = RotatingState::new(99);
    restored.restore_persistent(&persisted, device).unwrap();
    let actual = restored
        .update_and_fetch(&kv(1, 9.0), &kv(1, 79.0), device)
        .map(|(k, v)| (host(&k), host(&v), restored.offset, restored.idx))
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn persistent_snapshot_roundtrip_and_next_append_unwrapped() {
    let device = Device::Cpu;
    let mut source = RotatingState::new(16);
    source
        .update_and_fetch(&kv(5, 0.0), &kv(5, 50.0), device)
        .unwrap();
    let persisted = source.snapshot_persistent().unwrap();
    assert_eq!(persisted.valid_len, 5);
    assert_eq!(persisted.offset, 5);

    let expected = source
        .update_and_fetch(&kv(1, 8.0), &kv(1, 58.0), device)
        .map(|(k, v)| (host(&k), host(&v), source.offset, source.idx))
        .unwrap();
    let mut restored = RotatingState::new(16);
    restored.restore_persistent(&persisted, device).unwrap();
    let actual = restored
        .update_and_fetch(&kv(1, 8.0), &kv(1, 58.0), device)
        .map(|(k, v)| (host(&k), host(&v), restored.offset, restored.idx))
        .unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn persistent_restore_rejects_missing_payload_for_nonempty_ring() {
    let snapshot = RotatingStateSnapshot {
        keys: None,
        values: None,
        offset: 1,
        max_size: 4,
        keep: 0,
        valid_len: 1,
        idx: 1,
    };
    let mut state = RotatingState::new(4);
    assert!(state.restore_persistent(&snapshot, Device::Cpu).is_err());
}
