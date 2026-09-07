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

fn seg(len: usize) -> GdnTapeSegment {
    GdnTapeSegment {
        q: arr(&[1.0]),
        k: arr(&[2.0]),
        v: arr(&[3.0]),
        g: arr(&[4.0]),
        beta: arr(&[5.0]),
        conv_input: arr(&[6.0]),
        len,
    }
}

/// A tape records the state the round started from once, at its first segment,
/// and keeps counting positions as later forwards append to it. That first
/// state is what an accepted prefix is refolded from, so a later segment
/// overwriting it would silently refold from the wrong place.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn tape_keeps_the_first_state_and_counts_every_segment() {
    let mut tape = GdnTape::new();
    assert_eq!(tape.positions(), 0);
    assert!(tape.state_in().is_none());

    tape.push(&arr(&[10.0, 20.0]), seg(2)).expect("first push");
    tape.push(&arr(&[70.0, 80.0]), seg(1)).expect("second push");

    assert_eq!(tape.positions(), 3);
    assert_eq!(tape.segments().len(), 2);
    assert_eq!(read(tape.state_in().unwrap()), vec![10.0, 20.0]);
}

/// Arming clears whatever the previous round left, and taking the tape
/// disarms the cache: a forward after that records nothing.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn arm_clears_and_take_disarms() {
    let mut cache = LinearAttnCache::new();
    assert!(cache.tape.is_none());

    cache.arm_tape();
    cache
        .tape
        .as_mut()
        .expect("armed")
        .push(&arr(&[1.0]), seg(4))
        .expect("push");
    assert_eq!(cache.tape.as_ref().expect("armed").positions(), 4);

    cache.arm_tape();
    assert_eq!(cache.tape.as_ref().expect("re-armed").positions(), 0);

    cache.arm_tape();
    assert!(cache.take_tape().is_some());
    assert!(cache.tape.is_none());
    assert!(cache.take_tape().is_none());
}

/// The tape is resident memory for as long as a round is open, and
/// `resident_bytes` is the sum every K/V total is built from.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn resident_bytes_counts_an_armed_tape() {
    let mut cache = LinearAttnCache::new();
    cache.conv_state = Some(arr(&[1.0, 2.0, 3.0]));
    cache.delta_state = Some(arr(&[10.0, 20.0]));
    let bare = cache.resident_bytes();
    assert_eq!(bare, 5 * 4);

    cache.arm_tape();
    assert_eq!(cache.resident_bytes(), bare);

    cache
        .tape
        .as_mut()
        .expect("armed")
        .push(&arr(&[0.0, 0.0]), seg(1))
        .expect("push");
    // One f32 in each of the six segment buffers, plus the two-element
    // pre-round state the tape kept.
    assert_eq!(cache.resident_bytes(), bare + (6 + 2) * 4);
}

/// A prompt-cache clone is not a live round: it carries the recurrent state and
/// no tape.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn deep_clone_drops_the_tape() {
    let mut cache = LinearAttnCache::new();
    cache.conv_state = Some(arr(&[1.0, 2.0, 3.0]));
    cache.delta_state = Some(arr(&[10.0, 20.0]));
    cache.arm_tape();
    cache
        .tape
        .as_mut()
        .expect("armed")
        .push(&arr(&[1.0]), seg(1))
        .expect("push");

    let clone = cache.try_deep_clone().expect("deep clone");
    assert!(clone.tape.is_none());
    assert_eq!(
        read(clone.conv_state.as_ref().unwrap()),
        vec![1.0, 2.0, 3.0]
    );
    assert_eq!(read(clone.delta_state.as_ref().unwrap()), vec![10.0, 20.0]);
}
