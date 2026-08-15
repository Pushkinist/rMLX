use super::*;

fn key(model: &str, session: &str) -> SessionKey {
    SessionKey {
        model_id: model.to_owned(),
        session_id: session.to_owned(),
    }
}

#[test]
fn new_session_is_miss() {
    let mut cache = SessionCache::new(4);
    let hit = cache.touch(key("m", "s1"), 10);
    assert!(!hit, "first touch must be a miss");
}

#[test]
fn returning_session_is_hit() {
    let mut cache = SessionCache::new(4);
    cache.touch(key("m", "s1"), 10);
    let hit = cache.touch(key("m", "s1"), 20);
    assert!(hit, "second touch with same key must be a hit");
}

#[test]
fn different_sessions_are_isolated() {
    let mut cache = SessionCache::new(4);
    cache.touch(key("m", "s1"), 10);
    let hit = cache.touch(key("m", "s2"), 10);
    assert!(!hit, "different session id must be a miss");
}

#[test]
fn cross_model_no_collision() {
    let mut cache = SessionCache::new(4);
    cache.touch(key("model-a", "sess"), 10);
    let hit = cache.touch(key("model-b", "sess"), 10);
    assert!(!hit, "same session id but different model must be a miss");
}

#[test]
fn lru_eviction_when_at_capacity() {
    let mut cache = SessionCache::new(2);
    // Fill to capacity.
    cache.touch(key("m", "s1"), 10);
    cache.touch(key("m", "s2"), 10);
    assert_eq!(cache.active_count(), 2);

    // s3 triggers eviction of oldest (s1 or s2 — whichever has min last_used).
    cache.touch(key("m", "s3"), 10);
    assert_eq!(
        cache.active_count(),
        2,
        "capacity must stay at max_sessions"
    );
}

#[test]
fn active_count_reflects_unique_sessions() {
    let mut cache = SessionCache::new(10);
    cache.touch(key("m", "s1"), 1);
    cache.touch(key("m", "s2"), 1);
    cache.touch(key("m", "s1"), 2); // re-touch, should not increment count
    assert_eq!(cache.active_count(), 2);
}

#[test]
fn remove_model_clears_that_model_only() {
    let mut cache = SessionCache::new(10);
    cache.touch(key("ma", "s1"), 1);
    cache.touch(key("mb", "s1"), 1);
    cache.remove_model("ma");
    assert_eq!(cache.active_count(), 1);
    // mb still present.
    let hit = cache.touch(key("mb", "s1"), 2);
    assert!(hit);
}

#[test]
fn max_sessions_enforced_at_zero() {
    // max_sessions=0 is clamped to 1.
    let mut cache = SessionCache::new(0);
    cache.touch(key("m", "s1"), 1);
    cache.touch(key("m", "s2"), 1);
    assert_eq!(cache.active_count(), 1, "capacity must be at least 1");
}

/// A zero-slot server stays zero-slot for a session-bearing request.
///
/// `--prompt-cache-slots 0` disables the prompt cache. The session path used to
/// widen a hard-coded 4, so an `X-Session-Id` header re-enabled a cache the
/// operator had switched off — snapshots stored and Exact hits served on a
/// server configured to store none — and alternating between 0 and 4+n rebuilt
/// the cache on every request.
#[test]
fn zero_base_is_not_re_enabled_by_a_session() {
    assert_eq!(effective_prompt_cache_slots(0, 0), None);
    assert_eq!(
        effective_prompt_cache_slots(0, 7),
        None,
        "no number of active sessions may switch a disabled cache back on"
    );
}

/// A non-zero base is the operator's number, widened — never replaced by a
/// literal. Widening a hard-coded 4 gives someone who configured 8 a smaller
/// cache than they asked for.
#[test]
fn non_zero_base_is_widened_not_replaced() {
    assert_eq!(effective_prompt_cache_slots(1, 0), Some(1));
    assert_eq!(effective_prompt_cache_slots(4, 3), Some(7));
    assert_eq!(
        effective_prompt_cache_slots(8, 2),
        Some(10),
        "a base of 8 must not collapse to 4 + active"
    );
}
