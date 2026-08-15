use super::*;

/// Store → sample round-trip. Mirrors the per-request `/metrics/cache` wire path.
#[test]
fn counter_round_trip() {
    let counter = KvBytesCounter::default();
    let post = crate::decode_loop::PostDecode::for_test();
    assert_eq!(counter.sample().bytes, 0, "fresh counter reports zero");
    counter.store(424_242, post);
    assert_eq!(counter.sample().bytes, 424_242);
    counter.store(0, post);
    assert_eq!(counter.sample().bytes, 0);
}

/// The store sequence separates "no generation has reported a byte count" from
/// "a generation reported zero" — two states the bare byte value collapses into
/// the same `0`. A caller that records the figure as a measurement reads the
/// sequence, not the value, to decide whether it has one.
#[test]
fn seq_distinguishes_unreported_from_zero() {
    let counter = KvBytesCounter::default();
    let post = crate::decode_loop::PostDecode::for_test();

    assert_eq!(
        counter.sample().seq,
        0,
        "no store yet — sequence must still be zero"
    );

    // A genuine store of zero bytes is NOT the same state as "never stored",
    // even though both report `bytes == 0`.
    counter.store(0, post);
    let reported_zero = counter.sample();
    assert_eq!(reported_zero.bytes, 0);
    assert_eq!(
        reported_zero.seq, 1,
        "a store of zero must still advance the sequence — otherwise it is \
         indistinguishable from never having stored"
    );

    // Every store advances it, so a caller can tell "this generation reported"
    // from "I am reading the previous generation's figure".
    counter.store(4096, post);
    assert_eq!(
        counter.sample(),
        KvBytesSample {
            bytes: 4096,
            seq: 2
        }
    );
}

/// Two counters are two counters: a store on one must be invisible to the
/// other, including through the sequence a recording caller brackets with.
///
/// This is the unit-level shape of the cross-attribution defect. With one shared
/// counter behind both models, `b.store(...)` advances the sequence `a`'s
/// bracket is watching, `classify_kv_bytes` returns `Reported(b's bytes)`, and
/// the wrong figure is written to an append-only table under `a`'s name. The
/// wiring that keeps these separate in production — one counter per model
/// instance, never one per arch — is proven on real models by
/// `tests/kv_bytes_sample_point.rs::kv_bytes_counter_is_per_model_instance`.
#[test]
fn two_counters_do_not_cross_attribute() {
    let a = KvBytesCounter::default();
    let b = KvBytesCounter::default();
    let post = crate::decode_loop::PostDecode::for_test();

    a.store(1_000, post);
    let a_before = a.sample();

    // A whole generation on `b`, while nothing runs on `a`.
    b.store(9_999, post);

    let a_after = a.sample();
    assert_eq!(
        a_after, a_before,
        "a generation on another model must not move this model's sample"
    );
    assert_eq!(
        classify_kv_bytes(a_before, a_after),
        KvBytesVerdict::Unreported,
        "with no generation of its own, `a` must have nothing to record — not \
         `b`'s byte count wearing `a`'s label"
    );
}

/// `classify_kv_bytes` decides on the sequence first, and only then reads the
/// value. Collapsing the two is what records one run's number under another
/// run's label.
#[test]
fn classify_separates_detection_from_value() {
    let s = |bytes, seq| KvBytesSample { bytes, seq };

    // Sequence advanced, non-zero value → usable.
    assert_eq!(
        classify_kv_bytes(s(0, 4), s(8_192, 5)),
        KvBytesVerdict::Reported(8_192)
    );
    // Sequence advanced, zero value → the plumbing worked, the answer did not.
    assert_eq!(
        classify_kv_bytes(s(8_192, 4), s(0, 5)),
        KvBytesVerdict::ReportedZero
    );
    // Sequence unchanged → the readable non-zero value is an earlier
    // generation's, however plausible it looks.
    assert_eq!(
        classify_kv_bytes(s(8_192, 5), s(8_192, 5)),
        KvBytesVerdict::Unreported
    );
    // Never written at all.
    assert_eq!(
        classify_kv_bytes(s(0, 0), s(0, 0)),
        KvBytesVerdict::Unreported
    );
}
