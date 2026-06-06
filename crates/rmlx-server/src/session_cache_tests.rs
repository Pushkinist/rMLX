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
