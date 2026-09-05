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

/// A block write leaves the ring holding its window plus the block, and rolling
/// the rejected tail back off it leaves the window the shorter block would have
/// left — which is what a speculative round's partial acceptance needs.
///
/// The keys carry their own absolute position (`kv` writes `p, p+0.5`), so this
/// fails on content as well as on bookkeeping: a rollback that moved `offset`
/// without dropping the rejected keys reads the same offset and the wrong keys.
///
/// The last case is the boundary: a rollback of the *whole* block would leave
/// the ring one position short of its window, and is refused. The assistant
/// round loop never asks for it — the bonus token is always kept — so the
/// guarantee it needs is exactly "any tail up to `block - 1`". A **recurrent**
/// round loop does ask for the whole block, because it replays from the
/// pre-round offset its state snapshot was taken at, and would be refused here;
/// no architecture wired today pairs recurrent state with a windowed KV layer,
/// and `rollback_round_caches` is where the first one that does will say so.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn a_wrapped_ring_rolls_a_rejected_block_tail_back_off() {
    let device = Device::Cpu;
    let max_size = 4;
    let block = 4i32;
    let prompt = 6i32;

    for kept in 1..=block {
        let mut st = RotatingState::new(max_size);
        st.update_and_fetch(&kv(prompt, 0.0), &kv(prompt, 100.0), device)
            .unwrap();
        st.update_and_fetch(
            &kv(block, prompt as f32),
            &kv(block, 100.0 + prompt as f32),
            device,
        )
        .unwrap();
        assert!(
            st.offset >= st.max_size,
            "the ring must be wrapped or this proves nothing"
        );
        assert!(
            st.roll_back(block - kept).unwrap(),
            "kept={kept}: a ring a block write left in temporal order must roll back"
        );
        assert_eq!(st.offset, prompt + kept, "kept={kept}: rolled-back offset");

        // The window the rolled-back offset needs, read off the tail of the
        // buffer, is the run of absolute positions ending at that offset.
        let keys = host(st.keys.as_ref().unwrap());
        let window = max_size.min(st.offset) as usize;
        let tail = &keys[keys.len() - window * 2..];
        let want: Vec<f32> = (st.offset - window as i32..st.offset)
            .flat_map(|p| [p as f32, p as f32 + 0.5])
            .collect();
        assert_eq!(
            tail, want,
            "kept={kept}: the rolled-back ring must hold the window ending at its offset"
        );

        // And the ring is still writable, on both paths that write it. A
        // post-wrap rollback leaves a sliced view behind; a decode step reaches
        // that view through `slice_update`, which no-ops silently when its
        // bounds are wrong, and the next block write reaches it through
        // `temporal_order` / `trim`. Reading the buffer straight after the
        // rollback exercises neither.
        let before_decode = st.offset;
        let (k1, _v1) = st
            .update_and_fetch(
                &kv(1, before_decode as f32),
                &kv(1, 100.0 + before_decode as f32),
                device,
            )
            .unwrap();
        assert_eq!(
            st.offset,
            before_decode + 1,
            "kept={kept}: offset after a decode step"
        );
        // Past the wrap a decode step writes in place, so the window is present
        // but not in order — which positions the ring holds is the assertion.
        let mut held: Vec<f32> = host(&k1).chunks_exact(2).map(|c| c[0]).collect();
        held.sort_by(f32::total_cmp);
        let live = max_size.min(st.offset);
        let want_held: Vec<f32> = ((st.offset - live)..st.offset).map(|p| p as f32).collect();
        assert_eq!(
            held, want_held,
            "kept={kept}: a decode step into a rolled-back ring must leave it holding \
             the window ending at the new offset"
        );

        let before_block = st.offset;
        let (k2, _v2) = st
            .update_and_fetch(
                &kv(2, before_block as f32),
                &kv(2, 100.0 + before_block as f32),
                device,
            )
            .unwrap();
        assert_eq!(
            st.offset,
            before_block + 2,
            "kept={kept}: offset after a block write"
        );
        let after = host(&k2);
        let live = max_size.min(st.offset);
        let want_after: Vec<f32> = ((st.offset - live)..st.offset)
            .flat_map(|p| [p as f32, p as f32 + 0.5])
            .collect();
        assert_eq!(
            &after[after.len() - (live as usize) * 2..],
            want_after.as_slice(),
            "kept={kept}: a block write into a rolled-back ring must continue the window \
             it kept"
        );
    }

    let mut whole = RotatingState::new(max_size);
    whole
        .update_and_fetch(&kv(prompt, 0.0), &kv(prompt, 100.0), device)
        .unwrap();
    whole
        .update_and_fetch(
            &kv(block, prompt as f32),
            &kv(block, 100.0 + prompt as f32),
            device,
        )
        .unwrap();
    assert!(
        !whole.can_trim(block),
        "rolling the whole block back would leave the ring inside its own window"
    );
}

/// The refusal side. A ring past its wrap that a single-token write left in
/// rotated order cannot give a position back: the newest slots hold what they
/// overwrote. `can_trim` says so and `roll_back` leaves the ring untouched.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn a_rotated_ring_refuses_a_rollback_instead_of_losing_keys() {
    let device = Device::Cpu;
    let mut st = RotatingState::new(4);
    st.update_and_fetch(&kv(6, 0.0), &kv(6, 100.0), device)
        .unwrap();
    for step in 0..3 {
        let p = (6 + step) as f32;
        st.update_and_fetch(&kv(1, p), &kv(1, 100.0 + p), device)
            .unwrap();
    }
    assert!(st.offset >= st.max_size, "the ring must be wrapped");
    assert!(st.idx < st.keys.as_ref().unwrap().shape()[2], "and rotated");

    let before = (st.offset, st.idx, host(st.keys.as_ref().unwrap()));
    assert!(
        !st.can_trim(1),
        "a rotated ring cannot give a position back"
    );
    assert!(
        !st.roll_back(1).unwrap(),
        "and roll_back must say so rather than moving the offset"
    );
    assert_eq!(
        (st.offset, st.idx, host(st.keys.as_ref().unwrap())),
        before,
        "a refused rollback leaves the ring exactly as it was"
    );
}

/// Before the first wrap every position is still at its own slot, so any
/// rollback is a move of the write pointer — and the ring reads back as the
/// shorter prefix.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn an_unwrapped_ring_rolls_back_to_any_prefix() {
    let device = Device::Cpu;
    let mut st = RotatingState::new(64);
    st.update_and_fetch(&kv(6, 0.0), &kv(6, 100.0), device)
        .unwrap();
    assert!(st.can_trim(4));
    assert!(st.roll_back(4).unwrap());
    assert_eq!(st.offset, 2);
    assert_eq!(st.idx, 2);

    let (k, _v) = st
        .update_and_fetch(&kv(1, 2.0), &kv(1, 102.0), device)
        .unwrap();
    assert_eq!(
        host(&k),
        vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "the rolled-back ring continues from the prefix it kept"
    );
}

/// What `can_trim` promises is what `roll_back` leaves, at every rollback depth
/// a ring in either regime can be asked for. The callers gate on the predicate,
/// so a predicate that drifted from the operation is worse than none.
///
/// Asserting only that the two *agree* would be a tautology — `roll_back`'s
/// first statement consults `can_trim`, so no mutation short of deleting that
/// line could fail it. Each depth therefore asserts the ring's resulting state:
/// a rollback that happened holds the window ending at its new offset, one that
/// was refused left the ring byte-identical, and a rollback of nothing is a
/// no-op whichever regime the ring is in.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "the window slice is bounded by the buffer length it is taken from"
)]
fn what_can_trim_promises_is_what_roll_back_leaves() {
    let device = Device::Cpu;
    for rotated in [false, true] {
        for depth in 0..10i32 {
            let mut st = RotatingState::new(4);
            st.update_and_fetch(&kv(6, 0.0), &kv(6, 100.0), device)
                .unwrap();
            if rotated {
                st.update_and_fetch(&kv(1, 6.0), &kv(1, 106.0), device)
                    .unwrap();
            } else {
                st.update_and_fetch(&kv(4, 6.0), &kv(4, 106.0), device)
                    .unwrap();
            }
            let before = (st.offset, st.idx, host(st.keys.as_ref().unwrap()));
            let predicted = st.can_trim(depth);
            let done = st.roll_back(depth).unwrap();
            assert_eq!(
                predicted, done,
                "rotated={rotated} depth={depth}: can_trim promised {predicted} and \
                 roll_back did {done}"
            );
            let now = (st.offset, st.idx, host(st.keys.as_ref().unwrap()));
            if !done || depth == 0 {
                assert_eq!(
                    now, before,
                    "rotated={rotated} depth={depth}: a refused or empty rollback must \
                     leave the ring exactly as it was"
                );
                continue;
            }
            assert_eq!(
                st.offset,
                before.0 - depth,
                "rotated={rotated} depth={depth}: rolled-back offset"
            );
            let window = st.max_size.min(st.offset) as usize;
            let want: Vec<f32> = (st.offset - window as i32..st.offset)
                .flat_map(|p| [p as f32, p as f32 + 0.5])
                .collect();
            assert_eq!(
                &now.2[now.2.len() - window * 2..],
                want.as_slice(),
                "rotated={rotated} depth={depth}: the ring must hold the window ending \
                 at the offset it rolled back to"
            );
        }
    }
}
