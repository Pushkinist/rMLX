//! The first-token logprob wiring of the prompt cache, derived from the entry
//! types in the source tree and not from a list of architectures.
//!
//! An entry type is a struct that holds `prompt_token_ids`, `first_id` and
//! `first_piece`. One that emits logprobs holds `first_logprobs` too, and every
//! file that builds it stores a record at each store site and replays it on an
//! exact hit. A new entry type fails here until it does, or is named in
//! [`NO_LOGPROBS_ENTRIES`].

#![allow(
    clippy::indexing_slicing,
    reason = "test assertions index the fixed one-element list they just checked"
)]

use crate::stop_scan::sources_under;

/// Entry types of the arches that emit no logprobs on any path, a miss included
/// (every `ProbeStep` they build carries `logprobs: None`), so an entry has no
/// record to store or replay.
const NO_LOGPROBS_ENTRIES: &[&str] = &["LagunaEntry", "Qwen2Entry", "Qwen3VlMoeEntry"];

type Sources = [(String, String)];

struct EntryType {
    name: String,
    has_first_logprobs: bool,
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Byte offsets of `needle` in `code` where it is a whole word.
fn word_offsets(code: &str, needle: &str) -> Vec<usize> {
    code.match_indices(needle)
        .map(|(at, _)| at)
        .filter(|&at| {
            let before = code[..at].chars().next_back();
            let after = code[at + needle.len()..].chars().next();
            !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
        })
        .collect()
}

/// The text between the brace that opens at `open` and its match.
fn braced(code: &str, open: usize) -> &str {
    let mut depth = 0usize;
    for (at, c) in code[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &code[open + 1..open + at];
                }
            }
            _ => {}
        }
    }
    ""
}

fn entry_types(sources: &Sources) -> Vec<EntryType> {
    let mut found = Vec::new();
    for (_, code) in sources {
        for at in word_offsets(code, "struct") {
            let rest = code[at + "struct".len()..].trim_start();
            let name: String = rest.chars().take_while(|&c| is_ident(c)).collect();
            let after = rest[name.len()..].trim_start();
            if name.is_empty() || !after.starts_with('{') {
                continue;
            }
            let open = code.len() - after.len();
            let body = braced(code, open);
            if ["prompt_token_ids:", "first_id:", "first_piece:"]
                .iter()
                .all(|field| body.contains(field))
            {
                found.push(EntryType {
                    name,
                    has_first_logprobs: body.contains("first_logprobs:"),
                });
            }
        }
    }
    found
}

/// Offsets of the struct literals of `name` in `code`: not its definition, an
/// `impl` header, a pattern or a return type.
fn literal_offsets(code: &str, name: &str) -> Vec<usize> {
    word_offsets(code, name)
        .into_iter()
        .filter(|&at| code[at + name.len()..].trim_start().starts_with('{'))
        .filter(|&at| {
            let before = code[..at].trim_end();
            let previous = before.rsplit(char::is_whitespace).next().unwrap_or("");
            !["struct", "for", "impl", "let", "->"].contains(&previous)
        })
        .collect()
}

/// Calls of the fn `name` in `code`, its definition not counted.
fn calls_of(code: &str, name: &str) -> usize {
    word_offsets(code, name)
        .into_iter()
        .filter(|&at| code[at + name.len()..].starts_with('('))
        .filter(|&at| !code[..at].trim_end().ends_with("fn"))
        .count()
}

/// The name of the fn whose body holds offset `at`.
fn enclosing_fn(code: &str, at: usize) -> String {
    let head = &code[..at];
    word_offsets(head, "fn")
        .last()
        .map(|&fn_at| {
            head[fn_at + "fn".len()..]
                .trim_start()
                .chars()
                .take_while(|&c| is_ident(c))
                .collect()
        })
        .unwrap_or_default()
}

/// Every way the tree's wiring departs from the rule, one line each.
fn violations(sources: &Sources) -> Vec<String> {
    let entries = entry_types(sources);
    let mut out = Vec::new();

    for exempt in NO_LOGPROBS_ENTRIES {
        match entries.iter().find(|e| e.name == *exempt) {
            None => out.push(format!(
                "{exempt}: named in NO_LOGPROBS_ENTRIES but no such entry type exists"
            )),
            Some(e) if e.has_first_logprobs => out.push(format!(
                "{exempt}: holds first_logprobs, so it no longer belongs in NO_LOGPROBS_ENTRIES"
            )),
            Some(_) => {}
        }
    }

    let wired: Vec<&EntryType> = entries
        .iter()
        .filter(|e| !NO_LOGPROBS_ENTRIES.contains(&e.name.as_str()))
        .collect();
    for e in &wired {
        if !e.has_first_logprobs {
            out.push(format!(
                "{}: an entry type with no first_logprobs field, and not in NO_LOGPROBS_ENTRIES",
                e.name
            ));
        }
    }

    for (file, code) in sources {
        let mut stores = 0;
        for e in wired.iter().filter(|e| e.has_first_logprobs) {
            for at in literal_offsets(code, &e.name) {
                let helper = calls_of(code, &enclosing_fn(code, at));
                stores += helper.max(1);
            }
        }
        if stores == 0 {
            continue;
        }
        let captures = calls_of(code, "capture_first_logprobs");
        if captures != stores {
            out.push(format!(
                "{file}: {stores} store site(s) but {captures} call(s) of capture_first_logprobs"
            ));
        }
        if calls_of(code, "replay_first_logprobs") == 0 {
            out.push(format!(
                "{file}: builds an entry but never calls replay_first_logprobs"
            ));
        }
    }
    out
}

fn crate_sources() -> Vec<(String, String)> {
    sources_under(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"))
}

/// Every entry type of the crate emits logprobs through the shared helpers, or
/// is named as an arch that emits none.
#[test]
fn every_entry_type_stores_and_replays_its_first_logprobs() {
    let sources = crate_sources();
    assert!(
        entry_types(&sources).len() > NO_LOGPROBS_ENTRIES.len(),
        "the scan found no entry type that emits logprobs"
    );
    assert_eq!(violations(&sources), Vec::<String>::new());
}

/// Write `files` under a temporary `src` and scan them as the crate is scanned,
/// comments and string bodies removed by the shared reader.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O failures should abort the test loudly"
)]
fn violations_of(files: &[(&str, &str)]) -> Vec<String> {
    let dir = tempfile::tempdir().unwrap();
    for (name, text) in files {
        std::fs::write(dir.path().join(name), text).unwrap();
    }
    violations(&sources_under(dir.path()))
}

const EXEMPT_STRUCTS: &str = "
struct LagunaEntry { prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String }
struct Qwen2Entry { prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String }
struct Qwen3VlMoeEntry { prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String }
";

const WIRED_ENTRY: &str = "struct WiredEntry {
    prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String,
    first_logprobs: Option<TokenLogprobs> }";

const WIRED_GENERATE: &str = "fn generate_greedy() {
    if let Consumed::Exact(c) = consume() { emit(replay_first_logprobs(c.first_logprobs, k)); }
    let e = WiredEntry { prompt_token_ids, first_id, first_piece,
        first_logprobs: capture_first_logprobs(&l, &t) };
}";

/// The planted tree the cases below edit one line of.
fn wired_tree() -> Vec<(&'static str, &'static str)> {
    vec![
        ("exempt.rs", EXEMPT_STRUCTS),
        ("entry.rs", WIRED_ENTRY),
        ("generate.rs", WIRED_GENERATE),
    ]
}

#[test]
fn the_planted_tree_is_clean() {
    assert_eq!(violations_of(&wired_tree()), Vec::<String>::new());
}

#[test]
fn a_new_entry_type_without_the_field_fails() {
    let mut tree = wired_tree();
    tree.push((
        "newarch.rs",
        "struct NewArchEntry { prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String }",
    ));
    let found = violations_of(&tree);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].starts_with("NewArchEntry:"), "{found:?}");
}

#[test]
fn an_exempt_entry_that_gains_the_field_fails() {
    let mut tree = wired_tree();
    tree[0] = (
        "exempt.rs",
        "struct LagunaEntry { prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String,
            first_logprobs: Option<TokenLogprobs> }
         struct Qwen2Entry { prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String }
         struct Qwen3VlMoeEntry { prompt_token_ids: Vec<u32>, first_id: u32, first_piece: String }",
    );
    let found = violations_of(&tree);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].starts_with("LagunaEntry:"), "{found:?}");
}

#[test]
fn a_capture_named_only_in_a_comment_fails() {
    let mut tree = wired_tree();
    tree[2] = (
        "generate.rs",
        "fn generate_greedy() {
            // capture_first_logprobs(&l, &t)
            if let Consumed::Exact(c) = consume() { emit(replay_first_logprobs(c.first_logprobs, k)); }
            let e = WiredEntry { prompt_token_ids, first_id, first_piece, first_logprobs: None };
        }",
    );
    let found = violations_of(&tree);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("0 call(s) of capture_first_logprobs"),
        "{found:?}"
    );
}

#[test]
fn a_replay_named_only_in_a_comment_fails() {
    let mut tree = wired_tree();
    tree[2] = (
        "generate.rs",
        "fn generate_greedy() {
            // replay_first_logprobs(c.first_logprobs, k)
            let e = WiredEntry { prompt_token_ids, first_id, first_piece,
                first_logprobs: capture_first_logprobs(&l, &t) };
        }",
    );
    let found = violations_of(&tree);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("never calls replay_first_logprobs"),
        "{found:?}"
    );
}

/// A store helper called from two paths is two store sites: dropping the
/// capture from one of them leaves the other satisfying a bare presence check.
#[test]
fn a_store_helper_called_twice_needs_a_capture_at_each_call() {
    let helper = "fn store_snapshot(first_logprobs: Option<TokenLogprobs>) {
        let e = WiredEntry { prompt_token_ids, first_id, first_piece, first_logprobs };
    }
    fn generate_greedy() {
        emit(replay_first_logprobs(hit.first_logprobs, k));";
    let both = format!(
        "{helper}
        let a = capture_first_logprobs(&l, &t); store_snapshot(a);
        let b = capture_first_logprobs(&l, &t); store_snapshot(b);
    }}"
    );
    let one = format!(
        "{helper}
        let a = capture_first_logprobs(&l, &t); store_snapshot(a);
        store_snapshot(None);
    }}"
    );
    let scan = |generate: &str| {
        violations_of(&[
            ("exempt.rs", EXEMPT_STRUCTS),
            ("entry.rs", WIRED_ENTRY),
            ("generate.rs", generate),
        ])
    };
    assert_eq!(scan(&both), Vec::<String>::new());
    let found = scan(&one);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("2 store site(s) but 1 call(s)"),
        "{found:?}"
    );
}
