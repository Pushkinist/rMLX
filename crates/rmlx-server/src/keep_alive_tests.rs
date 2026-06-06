//! keep_alive module tests.

use super::*;

// ── parse_duration_spec ──────────────────────────────────────────────────────

#[test]
fn parse_int_seconds() {
    assert_eq!(
        parse_duration_spec("30").unwrap(),
        KeepAlivePolicy::Idle(Duration::from_secs(30))
    );
    assert_eq!(
        parse_duration_spec("900").unwrap(),
        KeepAlivePolicy::Idle(Duration::from_secs(900))
    );
}

#[test]
fn parse_seconds_suffix() {
    assert_eq!(
        parse_duration_spec("30s").unwrap(),
        KeepAlivePolicy::Idle(Duration::from_secs(30))
    );
}

#[test]
fn parse_minutes_suffix() {
    assert_eq!(
        parse_duration_spec("15m").unwrap(),
        KeepAlivePolicy::Idle(Duration::from_secs(15 * 60))
    );
}

#[test]
fn parse_hours_suffix() {
    assert_eq!(
        parse_duration_spec("2h").unwrap(),
        KeepAlivePolicy::Idle(Duration::from_secs(2 * 3600))
    );
    assert_eq!(
        parse_duration_spec("24h").unwrap(),
        KeepAlivePolicy::Idle(Duration::from_secs(24 * 3600))
    );
}

#[test]
fn parse_zero_unloads_after() {
    assert_eq!(
        parse_duration_spec("0").unwrap(),
        KeepAlivePolicy::UnloadAfter
    );
    assert_eq!(
        parse_duration_spec("0s").unwrap(),
        KeepAlivePolicy::UnloadAfter
    );
}

#[test]
fn parse_negative_pins() {
    assert_eq!(parse_duration_spec("-1").unwrap(), KeepAlivePolicy::Pin);
    assert_eq!(parse_duration_spec("-1s").unwrap(), KeepAlivePolicy::Pin);
    assert_eq!(parse_duration_spec("-30m").unwrap(), KeepAlivePolicy::Pin);
}

#[test]
fn parse_whitespace_tolerant() {
    assert_eq!(
        parse_duration_spec("  15m  ").unwrap(),
        KeepAlivePolicy::Idle(Duration::from_secs(900))
    );
}

#[test]
fn parse_rejects_empty() {
    assert!(parse_duration_spec("").is_err());
    assert!(parse_duration_spec("   ").is_err());
}

#[test]
fn parse_rejects_bad_unit() {
    assert!(parse_duration_spec("5d").is_err());
    assert!(parse_duration_spec("5x").is_err());
}

#[test]
fn parse_rejects_bad_number() {
    assert!(parse_duration_spec("abc").is_err());
    assert!(parse_duration_spec("12.5m").is_err()); // fractional not supported
}

#[test]
fn parse_rejects_missing_number() {
    assert!(parse_duration_spec("m").is_err());
    assert!(parse_duration_spec("-m").is_err());
    assert!(parse_duration_spec("-").is_err());
}

// ── KeepAlivePolicy::resolve ─────────────────────────────────────────────────

#[test]
fn resolve_request_wins_over_env_and_flag() {
    let r = KeepAlivePolicy::resolve(
        Some(KeepAlivePolicy::Pin),
        Some(KeepAlivePolicy::Idle(Duration::from_secs(60))),
        Some(KeepAlivePolicy::Idle(Duration::from_secs(120))),
    );
    assert_eq!(r, KeepAlivePolicy::Pin);
}

#[test]
fn resolve_env_wins_over_flag() {
    let r = KeepAlivePolicy::resolve(
        None,
        Some(KeepAlivePolicy::Idle(Duration::from_secs(60))),
        Some(KeepAlivePolicy::Idle(Duration::from_secs(120))),
    );
    assert_eq!(r, KeepAlivePolicy::Idle(Duration::from_secs(60)));
}

#[test]
fn resolve_flag_wins_when_no_request_or_env() {
    let r = KeepAlivePolicy::resolve(
        None,
        None,
        Some(KeepAlivePolicy::Idle(Duration::from_secs(120))),
    );
    assert_eq!(r, KeepAlivePolicy::Idle(Duration::from_secs(120)));
}

#[test]
fn resolve_default_15m_when_nothing_set() {
    let r = KeepAlivePolicy::resolve(None, None, None);
    assert_eq!(
        r,
        KeepAlivePolicy::Idle(Duration::from_secs(KeepAlivePolicy::DEFAULT_TTL_SECS))
    );
}

// ── ttl() ────────────────────────────────────────────────────────────────────

#[test]
fn ttl_pin_is_none() {
    assert_eq!(KeepAlivePolicy::Pin.ttl(), None);
}

#[test]
fn ttl_unload_after_is_near_zero() {
    let t = KeepAlivePolicy::UnloadAfter.ttl().unwrap();
    assert!(t.as_millis() <= 10);
}

#[test]
fn ttl_idle_is_duration() {
    assert_eq!(
        KeepAlivePolicy::Idle(Duration::from_secs(123)).ttl(),
        Some(Duration::from_secs(123))
    );
}

// ── policy_from_request_field ────────────────────────────────────────────────

#[test]
fn request_field_negative_pins() {
    assert_eq!(
        policy_from_request_field(Some(-1)),
        Some(KeepAlivePolicy::Pin)
    );
}

#[test]
fn request_field_zero_unloads_after() {
    assert_eq!(
        policy_from_request_field(Some(0)),
        Some(KeepAlivePolicy::UnloadAfter)
    );
}

#[test]
fn request_field_positive_idle() {
    assert_eq!(
        policy_from_request_field(Some(60)),
        Some(KeepAlivePolicy::Idle(Duration::from_secs(60)))
    );
}

#[test]
fn request_field_absent_is_none() {
    assert_eq!(policy_from_request_field(None), None);
}

// ── Decode lease guard ───────────────────────────────────────────────────────

#[test]
fn decode_lease_increments_on_acquire() {
    let lease: DecodeLease = Arc::new(AtomicUsize::new(0));
    let g = DecodeLeaseGuard::acquire(Arc::clone(&lease));
    assert_eq!(g.count(), 1);
    drop(g);
    assert_eq!(lease.load(Ordering::Acquire), 0);
}

#[test]
fn decode_lease_nests() {
    let lease: DecodeLease = Arc::new(AtomicUsize::new(0));
    let g1 = DecodeLeaseGuard::acquire(Arc::clone(&lease));
    let g2 = DecodeLeaseGuard::acquire(Arc::clone(&lease));
    assert_eq!(g1.count(), 2);
    drop(g1);
    assert_eq!(g2.count(), 1);
    drop(g2);
    assert_eq!(lease.load(Ordering::Acquire), 0);
}

#[test]
fn decode_lease_drop_on_panic_path() {
    // Simulate spawn_blocking panic: guard drops via stack unwind.
    let lease: DecodeLease = Arc::new(AtomicUsize::new(0));
    let lease_clone = Arc::clone(&lease);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = DecodeLeaseGuard::acquire(lease_clone);
        panic!("simulated decode panic");
    }));
    assert!(r.is_err());
    assert_eq!(lease.load(Ordering::Acquire), 0);
}
