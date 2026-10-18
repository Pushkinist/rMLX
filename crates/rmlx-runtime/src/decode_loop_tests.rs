use super::*;

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
fn smoke_verdict_equality() {
    assert_eq!(SmokeVerdict::Ok, SmokeVerdict::Ok);
    let a = SmokeVerdict::BrokenNan { at_step: 3 };
    let b = SmokeVerdict::BrokenNan { at_step: 3 };
    assert_eq!(a, b);
}
