use super::*;
use std::time::Duration;

#[test]
fn prefill_chunk_constant() {
    // Metal watchdog safety margin: changing this requires re-validating all
    // six archs at 8k context.
    assert_eq!(PREFILL_CHUNK, 64);
}

#[test]
fn decode_profile_records_steps() {
    let mut p = DecodeProfile::default();
    p.record_step(100, 50, 200);
    p.record_step(120, 60, 220);
    assert_eq!(p.decode_steps, 2);
    assert_eq!(p.forward_total_ns, 220);
    assert_eq!(p.eval_total_ns, 110);
    assert_eq!(p.step_total_ns, 420);
}

#[test]
fn decode_profile_records_prefill() {
    let mut p = DecodeProfile::default();
    let t0 = DecodeProfile::start_prefill();
    std::thread::sleep(Duration::from_millis(2));
    p.record_prefill(t0);
    // Avoid asserting exact ns — just sanity-check it advanced.
    assert!(
        p.prefill_total_ns > 1_000_000,
        "prefill_total_ns must be > 1ms"
    );
}

#[test]
fn smoke_verdict_equality() {
    assert_eq!(SmokeVerdict::Ok, SmokeVerdict::Ok);
    let a = SmokeVerdict::BrokenNan { at_step: 3 };
    let b = SmokeVerdict::BrokenNan { at_step: 3 };
    assert_eq!(a, b);
}
