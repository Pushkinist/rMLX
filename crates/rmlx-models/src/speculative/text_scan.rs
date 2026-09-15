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

/// The name this line declares a `fn` under, whatever qualifiers stand before
/// the keyword.
///
/// Read as tokens off the head of the line rather than off `fn ` alone: a
/// `pub(crate) fn`, a `const fn` or an `unsafe fn` whose name a reading skipped
/// would hand that reading the *previous* declaration's name, so a statement
/// inside it reads as one inside whatever was declared above — a wrong owner
/// rather than a missing one. `pub`, `pub(...)`, `async`, `unsafe`, `const` and
/// `extern "…"` are skipped in any order, which is every order the language
/// permits and a few it does not.
///
/// **Two shapes it does not read, and both are silent.** A `fn` nested inside
/// another never restores the outer name, so a statement after it is attributed
/// to the inner one; and an attribute on the same line as the declaration —
/// `#[inline] fn f()` — is not a declaration this reader sees at all. Neither
/// exists in the sources these readings scan, and a caller that grows one gets a
/// wrong owner rather than a refusal.
fn declared_fn(line: &str) -> Option<&str> {
    if !is_code(line) {
        return None;
    }
    let mut head = line.trim_start();
    loop {
        let rest = if let Some(rest) = head.strip_prefix("pub") {
            match rest.strip_prefix('(') {
                Some(scope) => scope.split_once(')').map_or(scope, |(_, tail)| tail),
                None => rest,
            }
        } else if let Some(rest) = head.strip_prefix("extern") {
            // `extern "C"`, and a bare `extern` too.
            rest.trim_start().strip_prefix('"').map_or(rest, |abi| {
                abi.split_once('"').map_or(abi, |(_, tail)| tail)
            })
        } else if let Some(rest) = head
            .strip_prefix("async ")
            .or_else(|| head.strip_prefix("unsafe "))
            .or_else(|| head.strip_prefix("const "))
        {
            rest
        } else {
            break;
        };
        head = rest.trim_start();
    }
    head.strip_prefix("fn ").map(|rest| {
        rest.split(|c: char| !c.is_alphanumeric() && c != '_')
            .next()
            .unwrap_or_default()
    })
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
        if let Some(name) = declared_fn(line) {
            current = name.to_owned();
        }
        if is_code(line) && line.contains(needle) {
            found.push((line.trim(), current.clone()));
        }
    }
    found
}
