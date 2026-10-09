//! Unit tests for `JsonReply`: the reply is cut where the constraint reported.

use std::sync::Arc;

use rmlx_models::Engagement;

use super::JsonReply;

fn reply_with(report: impl FnOnce(&Engagement)) -> JsonReply {
    let engagement = Arc::new(Engagement::default());
    report(&engagement);
    JsonReply::new(engagement)
}

#[test]
fn text_is_released_from_the_reported_byte_piece_by_piece() {
    let mut reply = reply_with(|e| e.report(2, 1));
    let mut out = String::new();
    for (token, piece) in [(1, "```"), (2, "\n{"), (3, "\"a\""), (4, "}")] {
        reply.receive(token, piece);
        out.push_str(&piece[reply.release(piece)..]);
    }
    assert_eq!(out, "{\"a\"}");
    assert_eq!(reply.start(), Some(4));
    assert!(reply.engaged());
}

#[test]
fn the_whole_answer_released_at_once_gives_the_same_reply() {
    let mut reply = reply_with(|e| e.report(2, 1));
    for (token, piece) in [(1, "```"), (2, "\n{"), (3, "\"a\""), (4, "}")] {
        reply.receive(token, piece);
    }
    assert_eq!(reply.release("```\n{\"a\"}"), 4);
    assert!(reply.engaged());
}

#[test]
fn text_released_in_other_pieces_than_it_arrived_gives_the_same_reply() {
    let mut reply = reply_with(|e| e.report(2, 1));
    reply.receive(1, "```");
    reply.receive(2, "\n{");
    assert_eq!(reply.release("``"), 2);
    assert_eq!(reply.release("`\n{"), 2);
    reply.receive(3, "}");
    assert_eq!(reply.release("}"), 0);
}

#[test]
fn text_that_ends_before_the_reported_byte_is_not_engaged() {
    let mut reply = reply_with(|e| e.report(2, 1));
    reply.receive(1, "```");
    reply.receive(2, "\n{");
    assert_eq!(reply.release("```\n"), 4);
    assert!(!reply.engaged(), "the text stops at the byte");
}

#[test]
fn no_report_gives_no_reply() {
    let mut reply = reply_with(|_| {});
    reply.receive(1, "{}");
    assert_eq!(reply.release("{}"), 2);
    assert_eq!(reply.start(), None);
    assert!(!reply.engaged());
}

#[test]
fn a_report_for_the_start_of_the_next_token_skips_an_empty_piece() {
    let mut reply = reply_with(|e| e.report_token_start(2));
    reply.receive(1, "Here");
    reply.receive(2, "");
    reply.receive(3, "true");
    assert_eq!(reply.release("Heretrue"), 4);
}
