use super::*;

#[test]
fn noop_mask_is_all_true() {
    let mut c = NoOpConstraint::new();
    let m = c.step_mask(8);
    assert_eq!(m.len(), 8);
    assert!(m.iter().all(|&b| b), "NoOp mask must be all-true");
}

#[test]
fn noop_advance_is_idempotent_and_never_finishes() {
    let mut c = NoOpConstraint::new();
    let _ = c.step_mask(4);
    c.advance(0);
    c.advance(3);
    assert!(!c.finished());
    let m2 = c.step_mask(4);
    assert!(m2.iter().all(|&b| b));
}

#[test]
fn engagement_gives_the_bytes_of_each_token_that_are_before_the_value() {
    let e = Engagement::default();
    assert_eq!(e.bytes_before(1, "ab"), 2, "not engaged: all before");
    e.report(3, 1);
    assert_eq!(e.bytes_before(2, "ab"), 2, "a token before the place");
    assert_eq!(e.bytes_before(3, "\n{"), 1, "the token of the place");
    assert_eq!(e.bytes_before(3, " \n{"), 2, "counted from the end");
    assert_eq!(e.bytes_before(3, ""), 0, "a text shorter than the tail");
    assert_eq!(e.bytes_before(4, "ab"), 0, "a token after the place");
    e.report(3, 2);
    assert_eq!(e.bytes_before(3, "é{"), 0, "not inside a character");
    e.report_token_start(5);
    assert_eq!(e.bytes_before(4, "ab"), 2);
    assert_eq!(e.bytes_before(5, "ab"), 0);
}

#[test]
fn noop_reports_no_engagement() {
    assert!(NoOpConstraint::new().engagement().is_none());
}

#[test]
fn noop_handles_vocab_size_change() {
    let mut c = NoOpConstraint::new();
    let m1_len = c.step_mask(4).len();
    let m2_len = c.step_mask(7).len();
    assert_eq!(m1_len, 4);
    assert_eq!(m2_len, 7);
}

// -------- round-trip test using a deliberately-restrictive impl --------

/// Test-only constraint that only allows token ids in a fixed allowed-set.
/// Used to verify the round-trip of step_mask / advance against a real
/// argmax in `apply_mask_argmax` (see sampler tests).
#[derive(Debug)]
pub(crate) struct AllowedSetConstraint {
    allowed: std::collections::HashSet<u32>,
    mask: Vec<bool>,
    sampled: Vec<u32>,
}

impl AllowedSetConstraint {
    pub(crate) fn new(allowed: impl IntoIterator<Item = u32>) -> Self {
        Self {
            allowed: allowed.into_iter().collect(),
            mask: Vec::new(),
            sampled: Vec::new(),
        }
    }
    pub(crate) fn sampled(&self) -> &[u32] {
        &self.sampled
    }
}

impl ConstraintEngine for AllowedSetConstraint {
    fn step_mask(&mut self, vocab_size: usize) -> &[bool] {
        if self.mask.len() != vocab_size {
            self.mask = (0..vocab_size as u32)
                .map(|i| self.allowed.contains(&i))
                .collect();
        }
        &self.mask
    }
    fn advance(&mut self, token_id: u32) {
        self.sampled.push(token_id);
    }
    fn finished(&self) -> bool {
        false
    }
}

#[test]
fn allowed_set_step_mask_only_marks_allowed() {
    let mut c = AllowedSetConstraint::new([1u32, 4, 7]);
    let m = c.step_mask(10);
    for (i, &b) in m.iter().enumerate() {
        assert_eq!(
            b,
            matches!(i, 1 | 4 | 7),
            "mask[{i}] expected {} got {}",
            matches!(i, 1 | 4 | 7),
            b
        );
    }
}
