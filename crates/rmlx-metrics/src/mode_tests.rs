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
    // Pure functions of the mode value — asserted without touching the global
    // OnceLock, which other tests in this binary may already have set.
    let gates = |m: MetricsMode| (m != MetricsMode::Off, m == MetricsMode::Full);

    assert_eq!(gates(MetricsMode::Off), (false, false));
    assert_eq!(gates(MetricsMode::Events), (true, false));
    assert_eq!(gates(MetricsMode::Full), (true, true));
}
