//! How the speculative test files read their own source.
//!
//! Several facts about this module family are wirings rather than arithmetics —
//! an order, a branch, a call's owner — and none of them can be driven without
//! loading a model. What holds them is a text reading of the drafter's own
//! source, and the reading is the same one every time, so it is here once
//! instead of once per drafter.
//!
//! **It reads text and is blind past that**, which every caller restates for its
//! own needles: a statement at the position a test names, inside a branch that
//! never runs, reads identical, and so does one whose arguments are wrong.
//!
//! It holds no `#[test]`, which is why it is not a `*_tests.rs` file: that
//! suffix is what the file-hygiene gates read as "this file is a test body", and
//! a helper wearing it would be scanned for tests it does not have.

/// Whether a line is code rather than a whole-line comment.
///
/// A needle written into the sentence that explains it would otherwise read as
/// the statement it describes — the same reading `scripts/check_spec_charge.sh`
/// takes, and for the same reason. A trailing comment on a line of code is not
/// stripped: that needs a quote-aware scan, and no needle here is one a caller
/// would write at the end of a statement.
pub(crate) fn is_code(line: &str) -> bool {
    !line.trim_start().starts_with("//")
}

/// Every code line of `src` carrying `needle`, each with the name of the `fn` it
/// sits in.
///
/// The owner is half of what these readings hold: a call moved from `verify` to
/// `condition`, or from `rollback` to `propose`, runs at a different point of
/// the round with the line itself unchanged.
pub(crate) fn lines_in_fns<'a>(src: &'a str, needle: &str) -> Vec<(&'a str, String)> {
    let mut current = String::new();
    let mut found = Vec::new();
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("fn ") {
            current = rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default()
                .to_owned();
        }
        if is_code(line) && line.contains(needle) {
            found.push((line.trim(), current.clone()));
        }
    }
    found
}
