use super::*;

fn capture(target: Option<usize>) -> AnchorCapture {
    AnchorCapture::new(target, KvCache::eval_prefill_state)
}

#[test]
fn an_entry_nothing_can_keep_captures_nothing() {
    assert!(entry_is_storable(4, false));
    assert!(!entry_is_storable(0, false), "no slots");
    assert!(
        !entry_is_storable(4, true),
        "paged KV refuses a cache holding pages"
    );
    assert_eq!(anchor_target(700, false, 0), None);
    assert_eq!(anchor_target(700, true, 0), Some(512));
    let none = capture(anchor_target(700, false, 0));
    assert_eq!(none.cut_in(0, 2048), None, "and no split of the chunk");
}

#[test]
fn a_resumed_request_captures_only_at_or_past_its_resume_point() {
    assert_eq!(
        anchor_target(740, true, 700),
        None,
        "the anchor 512 is behind it"
    );
    assert_eq!(anchor_target(1300, true, 700), Some(1280));
    assert_eq!(
        anchor_target(769, true, 768),
        Some(768),
        "at the resume point"
    );
}

#[test]
fn a_chunk_is_cut_where_the_target_falls_inside_it() {
    let capture = capture(Some(700));
    assert_eq!(capture.cut_in(0, 2048), Some(700));
    assert_eq!(capture.cut_in(512, 2048), Some(188));
}

#[test]
fn a_target_on_a_chunk_edge_cuts_nothing_off_the_chunk() {
    let capture = capture(Some(2048));
    assert_eq!(
        capture.cut_in(0, 2048),
        Some(2048),
        "taken after the chunk that ends on it"
    );
    assert_eq!(
        capture.cut_in(2048, 2048),
        Some(0),
        "taken before the chunk that starts on it"
    );
}

#[test]
fn a_chunk_the_target_is_not_in_is_not_cut() {
    let capture = capture(Some(700));
    assert_eq!(capture.cut_in(0, 512), None);
    assert_eq!(capture.cut_in(2048, 2048), None);
    assert_eq!(self::capture(None).cut_in(0, 2048), None);
}

#[test]
fn the_state_is_taken_once() {
    let mut capture = capture(Some(700));
    assert!(capture.capture(700, &[]).is_ok());
    assert_eq!(capture.cut_in(0, 2048), None);
    assert_eq!(capture.into_anchor().map(|a| a.pos), Some(700));
}
