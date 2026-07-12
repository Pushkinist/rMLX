use super::*;

#[test]
fn parse_round_trip() {
    for m in [MetricsMode::Off, MetricsMode::Events, MetricsMode::Full] {
        assert_eq!(MetricsMode::parse(m.as_str()), Some(m));
    }
}

#[test]
fn parse_rejects_junk() {
    assert_eq!(MetricsMode::parse("on"), None);
    assert_eq!(MetricsMode::parse(""), None);
    assert_eq!(MetricsMode::parse("FULL"), None);
}

#[test]
fn default_is_full() {
    // Library/test use never calls init(); historical behaviour must be preserved.
    assert_eq!(MetricsMode::default(), MetricsMode::Full);
}

#[test]
fn gates_follow_the_mode() {
    // Asserted against `MetricsMode`'s own predicate methods — the exact code
    // the free functions `events_enabled()` / `observations_enabled()`
    // delegate to (see `current().events_enabled()` etc. above), not a
    // reimplemented closure that could drift from the shipped logic while
    // still passing. Deliberately does NOT touch the process-global
    // `OnceLock` (`init` / `current`) — that state is shared across every
    // test in this binary, and asserting a specific value read through it
    // would race any other test that also sets it.
    let gates = |m: MetricsMode| (m.events_enabled(), m.observations_enabled());

    assert_eq!(gates(MetricsMode::Off), (false, false));
    assert_eq!(gates(MetricsMode::Events), (true, false));
    assert_eq!(gates(MetricsMode::Full), (true, true));
}
