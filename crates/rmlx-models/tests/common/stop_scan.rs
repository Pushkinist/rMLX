//! Source readers for the rules on the stop of a request. The tests of each
//! crate that holds a loop or a blocking job of a request include this file.
//!
//! The rules read text. They hold that the one stop helper is the first
//! statement of each loop that a request can keep running, with `?`, and that
//! each such loop is found from the work it does and not from a list of names.
//! They cannot hold that the flag a job hands over is the flag its request
//! sets.
//!
//! What the scan does not read: a loop by recursion, and work behind a name it
//! does not know, such as an evaluation inside a wrapper fn. The fn reader here
//! is a second one beside `speculative/text_scan.rs`, which is private to one
//! crate and cannot be loaded twice in it.

#![allow(
    dead_code,
    unreachable_pub,
    clippy::expect_used,
    reason = "test support that several crates include: each uses a part of it, and a source \
              tree that cannot be read is a test failure"
)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The one function a loop of a request calls to read the stop. It returns
/// the stop outcome as an error, and the loop hands that error on with `?`.
pub const STOP_HELPER: &str = "ensure_not_cancelled";

/// The error variant of a stop, as `Debug` opens it.
pub const STOP_OUTCOME: &str = "Cancelled";

/// The code of a Rust source file: `//` comments removed and the body of each
/// string literal blanked, by the readers of `scripts/lib/awk_text.sh`.
pub fn code_of(path: &Path) -> String {
    let readers = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../scripts/lib/awk_text.sh");
    let out = std::process::Command::new("bash")
        .arg("-c")
        .arg(r#". "$1"; awk "${AWK_TEXT_FNS}"'{ print blank_strings(decomment($0)) }' "$2""#)
        .arg("code_of")
        .arg(&readers)
        .arg(path)
        .output()
        .expect("bash must run");
    assert!(out.status.success(), "{}: {out:?}", path.display());
    String::from_utf8(out.stdout).expect("the source is UTF-8")
}

/// The source files under `dir` that are not test files, sorted.
pub fn source_files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut dirs = vec![dir.to_path_buf()];
    while let Some(next) = dirs.pop() {
        for entry in std::fs::read_dir(&next).expect("a source directory") {
            let path = entry.expect("a directory entry").path();
            let is_test = path
                .file_stem()
                .is_some_and(|stem| stem.to_string_lossy().ends_with("tests"));
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") && !is_test {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

/// The source files under `src`, each with its path from `src` and its code.
pub fn sources_under(src: &Path) -> Vec<(String, String)> {
    source_files(src)
        .into_iter()
        .map(|path| {
            let file = path
                .strip_prefix(src)
                .expect("a file under the source directory")
                .to_string_lossy()
                .into_owned();
            (file, code_of(&path))
        })
        .collect()
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The offsets at which `code` names `word` as a whole identifier.
fn word_offsets(code: &str, word: &str) -> Vec<usize> {
    code.match_indices(word)
        .map(|(at, _)| at)
        .filter(|at| {
            let before = code.get(..*at).and_then(|s| s.chars().next_back());
            let after = code.get(at + word.len()..).and_then(|s| s.chars().next());
            !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
        })
        .collect()
}

/// `true` when `code` names `word` as a whole identifier.
pub fn names(code: &str, word: &str) -> bool {
    !word_offsets(code, word).is_empty()
}

/// One call in a source text.
#[derive(Clone, Copy, Debug)]
pub struct Call<'c> {
    /// Where the name of the callee starts in the text.
    pub at: usize,
    /// The argument list.
    pub args: &'c str,
    /// The text after the closing parenthesis.
    pub after: &'c str,
}

impl Call<'_> {
    /// `true` when `offset` is inside the argument list of this call.
    pub fn holds(&self, callee: &str, offset: usize) -> bool {
        let open = self.at + callee.len() + 1;
        (open..open + self.args.len()).contains(&offset)
    }
}

/// Each call of `callee` in `code`. A fn of a longer name, and the declaration
/// of `callee`, are not calls of it.
pub fn calls_of<'c>(code: &'c str, callee: &str) -> Vec<Call<'c>> {
    let open = format!("{callee}(");
    code.match_indices(&open)
        .filter(|(at, _)| {
            let before = code.get(..*at).unwrap_or_default();
            !before.ends_with(is_ident) && !before.ends_with("fn ")
        })
        .filter_map(|(at, _)| {
            let args = code.get(at + open.len()..)?;
            let mut depth = 1usize;
            let end = args.char_indices().find_map(|(i, c)| {
                match c {
                    '(' => depth += 1,
                    ')' => depth -= 1,
                    _ => {}
                }
                (depth == 0).then_some(i)
            })?;
            Some(Call {
                at,
                args: args.get(..end)?,
                after: args.get(end + 1..)?,
            })
        })
        .collect()
}

/// The calls of the stop helper in `code` that hand its error on with `?`,
/// and the calls that do anything else with it.
pub fn stop_reads(code: &str) -> (usize, usize) {
    let calls = calls_of(code, STOP_HELPER);
    let handed_on = calls
        .iter()
        .filter(|call| call.after.trim_start().starts_with('?'))
        .count();
    (handed_on, calls.len() - handed_on)
}

/// `true` when the first statement of the body of `loop_code` is the stop
/// helper with `?`, by its bare name or by a path. A call after the work, in a
/// condition, or after a `continue` is not the first statement.
pub fn opens_with_the_stop_read(loop_code: &str) -> bool {
    let body = match loop_code.find("{\n") {
        Some(brace) => loop_code.get(brace + 1..).unwrap_or_default(),
        None => loop_code.split_once('\n').map_or("", |(_, body)| body),
    };
    let body = body.trim_start();
    calls_of(body, STOP_HELPER).first().is_some_and(|call| {
        let path = body.get(..call.at).unwrap_or_default();
        path.chars().all(|c| is_ident(c) || c == ':') && call.after.trim_start().starts_with('?')
    })
}

/// `true` when `loop_code` also reads for itself the flag that it hands to
/// the stop helper.
pub fn reads_the_flag_for_itself(loop_code: &str) -> bool {
    calls_of(loop_code, STOP_HELPER).iter().any(|call| {
        let flag = call.args.trim().trim_start_matches('&');
        !flag.is_empty() && loop_code.contains(&format!("{flag}.load("))
    })
}

/// `true` when `loop_code` reads the stop by the one rule: the helper is its
/// first statement, it is the only call of the helper, and the loop reads the
/// flag in no other way.
pub fn stops_by_the_rule(loop_code: &str) -> bool {
    stop_reads(loop_code) == (1, 0)
        && opens_with_the_stop_read(loop_code)
        && !reads_the_flag_for_itself(loop_code)
}

/// What a loop does in each pass that can keep a request on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Work {
    /// A call of a fn named `forward` or `forward_*`.
    Forward,
    /// An evaluation: `eval` or `async_eval`.
    Eval,
    /// The evaluation of the state of a cache during a prefill.
    StateEval,
    /// A token handed to the step callback, or the tokens of a round.
    Emit,
    /// A call of a closure that the fn was handed.
    Callback,
    /// One attention over a slice of its queries.
    Attention,
    /// The load of one image source: a fetch of up to 20 s, a decode, a copy.
    Load,
}

/// One loop of a source file.
#[derive(Clone, Debug)]
pub struct Loop {
    /// The fn that holds the loop.
    pub owner: String,
    /// The header line, trimmed.
    pub header: String,
    /// The header and the body.
    pub code: String,
    /// What each pass of the loop does for a request.
    pub work: BTreeSet<Work>,
}

impl Loop {
    /// The loop as a list names it: its file, its fn and its header.
    pub fn key(&self, file: &str) -> String {
        format!("{file}::{}: {}", self.owner, self.header)
    }
}

/// The name of the fn that `line` opens.
fn fn_name(line: &str) -> Option<&str> {
    let mut rest = line.trim_start();
    for prefix in [
        "pub(crate) ",
        "pub(super) ",
        "pub ",
        "const ",
        "async ",
        "unsafe ",
    ] {
        rest = rest.strip_prefix(prefix).unwrap_or(rest);
    }
    rest.strip_prefix("fn ")?.split(['(', '<']).next()
}

/// Methods that take a closure and run it at most once. Every other method
/// that takes a closure is read as a loop over the items of an iterator, so an
/// adaptor that this list does not know is a loop and not a blind spot.
const RUNS_ITS_CLOSURE_ONCE: [&str; 22] = [
    "and_then",
    "expect_err",
    "get_or_insert_with",
    "inspect_err",
    "is_err_and",
    "is_none_or",
    "is_ok_and",
    "is_some_and",
    "map_err",
    "map_or",
    "map_or_else",
    "ok_or_else",
    "or_else",
    "or_insert_with",
    "then",
    "unwrap_or_else",
    "with_default",
    "with_inner",
    "with_inner_mut",
    "spawn",
    "spawn_blocking",
    "block_on",
];

/// `code` without a leading loop label (`'name: `).
fn without_label(code: &str) -> &str {
    code.strip_prefix('\'')
        .and_then(|labelled| labelled.split_once(": "))
        .map_or(code, |(_, after)| after)
}

/// `true` when `line` opens a loop: `for`, `while` or `loop`, with a label or
/// without one, a `loop` bound to a name (`let x = loop {`), or a closure that
/// a method runs for each item.
pub fn opens_a_loop(line: &str) -> bool {
    let code = without_label(line.trim_start());
    (code.starts_with("for ") && code.contains(" in "))
        || code.starts_with("while ")
        || code.starts_with("loop {")
        || code
            .split_once(" = ")
            .is_some_and(|(_, bound)| without_label(bound).starts_with("loop {"))
        || closure_takers(line)
            .iter()
            .any(|method| !RUNS_ITS_CLOSURE_ONCE.contains(method))
}

/// `true` when one of the arguments that `args` opens with is a closure.
fn takes_a_closure(args: &str) -> bool {
    let mut depth = 0usize;
    let mut argument_starts = true;
    for (at, c) in args.char_indices() {
        if argument_starts && depth == 0 && !c.is_whitespace() {
            let rest = args.get(at..).unwrap_or_default();
            if rest.strip_prefix("move ").unwrap_or(rest).starts_with('|') {
                return true;
            }
            argument_starts = false;
        }
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' if depth == 0 => return false,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => argument_starts = true,
            _ => {}
        }
    }
    false
}

/// The methods on `line` that are called with a closure.
fn closure_takers(line: &str) -> Vec<&str> {
    line.match_indices('(')
        .filter(|(at, _)| takes_a_closure(line.get(at + 1..).unwrap_or_default()))
        .filter_map(|(at, _)| {
            let before = line.get(..at)?;
            let start = before
                .char_indices()
                .rev()
                .take_while(|(_, c)| is_ident(*c))
                .last()
                .map(|(i, _)| i)?;
            before
                .get(..start)?
                .ends_with('.')
                .then(|| before.get(start..))?
        })
        .collect()
}

/// The closures that the fn opened by `signature` was handed, by parameter
/// name: each `name: impl FnMut(`, `name: &mut impl FnMut(` or
/// `name: &mut dyn FnMut(`.
fn callbacks(signature: &[&str]) -> Vec<String> {
    let signature = signature.join(" ");
    signature
        .match_indices("FnMut(")
        .filter_map(|(at, _)| {
            let before = signature.get(..at)?;
            let colon = before.rfind(':')?;
            let kind = before.get(colon + 1..)?;
            if kind.contains([',', ')', '>']) {
                return None;
            }
            let named = before.get(..colon)?.trim_end();
            let start = named
                .char_indices()
                .rev()
                .take_while(|(_, c)| is_ident(*c))
                .last()
                .map(|(i, _)| i)?;
            named.get(start..).map(str::to_owned)
        })
        .collect()
}

/// The work `code` does, read from the name before each `(`.
fn work_of(code: &str, callbacks: &[String]) -> BTreeSet<Work> {
    code.match_indices('(')
        .filter_map(|(at, _)| {
            let before = code.get(..at)?.trim_end_matches(')');
            let start = before
                .char_indices()
                .rev()
                .take_while(|(_, c)| is_ident(*c))
                .last()
                .map(|(i, _)| i)?;
            let name = before.get(start..)?;
            let declared = before.get(..start)?.trim_end().ends_with("fn");
            match name {
                _ if declared => None,
                "eval" | "async_eval" => Some(Work::Eval),
                "eval_prefill_state" | "eval_gpu_state" => Some(Work::StateEval),
                "step_fn" | "emit_round_tokens" => Some(Work::Emit),
                "scaled_dot_product_attention" => Some(Work::Attention),
                "load_image" | "rmlx_server_load_image" => Some(Work::Load),
                _ if name == "forward" || name.starts_with("forward_") => Some(Work::Forward),
                _ if callbacks.iter().any(|c| c == name) => Some(Work::Callback),
                _ => None,
            }
        })
        .collect()
}

/// The fns of `code`, each as its name and its lines, from its declaration to
/// the next one.
fn fn_lines(code: &str) -> Vec<(String, Vec<&str>)> {
    let mut fns: Vec<(String, Vec<&str>)> = Vec::new();
    for line in code.lines() {
        if let Some(name) = fn_name(line) {
            fns.push((name.to_owned(), Vec::new()));
        }
        if let Some((_, lines)) = fns.last_mut() {
            lines.push(line);
        }
    }
    fns
}

/// The fns of `code`, each as its name and its text.
pub fn fns_of(code: &str) -> Vec<(String, String)> {
    fn_lines(code)
        .into_iter()
        .map(|(name, lines)| (name, lines.join("\n")))
        .collect()
}

/// The loops of `code` that do work for a request. The body of a loop is the
/// lines that follow its header at a deeper indent, and the brace that closes
/// a header wrapped over several lines.
pub fn loops_of(code: &str) -> Vec<Loop> {
    let indent = |line: &str| line.len() - line.trim_start().len();
    let mut found = Vec::new();
    for (owner, lines) in fn_lines(code) {
        let body_from = lines
            .iter()
            .position(|line| line.trim_end().ends_with('{'))
            .unwrap_or(0);
        let callbacks = callbacks(lines.get(..=body_from).unwrap_or_default());
        for (at, header) in lines.iter().enumerate().skip(body_from + 1) {
            if !opens_a_loop(header) {
                continue;
            }
            let body = lines.iter().skip(at + 1).take_while(|line| {
                line.trim().is_empty() || line.trim() == "{" || indent(line) > indent(header)
            });
            let code = std::iter::once(header)
                .chain(body)
                .copied()
                .collect::<Vec<_>>()
                .join("\n");
            let work = work_of(&code, &callbacks);
            if !work.is_empty() {
                found.push(Loop {
                    owner: owner.clone(),
                    header: header.trim().to_owned(),
                    code,
                    work,
                });
            }
        }
    }
    found
}

/// The loops of every source file under `src` that do work for a request,
/// each with its file as a path from `src`.
pub fn loops_under(src: &Path) -> Vec<(String, Loop)> {
    sources_under(src)
        .into_iter()
        .flat_map(|(file, code)| loops_of(&code).into_iter().map(move |l| (file.clone(), l)))
        .collect()
}

/// `true` for a loop that is bounded by its shape. It walks the layers or the
/// blocks of one forward and does nothing but that forward, so it builds one
/// graph; or it walks the caches of the layers and only evaluates their
/// prefill state. A loop over `blocks` that evaluates, emits or calls back is
/// not bounded by its shape.
pub fn bounded_by_shape(l: &Loop) -> bool {
    let only = |work: Work| l.work.iter().all(|w| *w == work);
    let over_layers = ["layers", "blocks"].iter().any(|w| l.header.contains(w));
    let over_caches = l.header.contains("aches");
    (over_layers && only(Work::Forward)) || (over_caches && only(Work::StateEval))
}

/// The loops that no list names, and the listed loops that the scan did not
/// find, each by its key. A loop that its shape bounds needs no list.
pub fn unlisted_and_lost(found: &[(String, Loop)], listed: &[&str]) -> (Vec<String>, Vec<String>) {
    let mut found: Vec<String> = found
        .iter()
        .filter(|(_, l)| !bounded_by_shape(l))
        .map(|(file, l)| l.key(file))
        .collect();
    let mut lost = Vec::new();
    for key in listed {
        match found.iter().position(|f| f == key) {
            Some(at) => drop(found.swap_remove(at)),
            None => lost.push((*key).to_owned()),
        }
    }
    found.sort();
    (found, lost)
}

/// The loops of `found` that `listed` names and that do not stop by the rule,
/// each by its key. A listed loop that the scan did not find is one of them.
pub fn not_stopping(found: &[(String, Loop)], listed: &[&str]) -> Vec<String> {
    listed
        .iter()
        .filter(|key| {
            !found
                .iter()
                .any(|(file, l)| l.key(file) == **key && stops_by_the_rule(&l.code))
        })
        .map(|key| (*key).to_owned())
        .collect()
}

/// The calls of the stop helper in `sources` that are in none of the `listed`
/// loops, in any form.
pub fn stop_reads_past(
    sources: &[(String, String)],
    found: &[(String, Loop)],
    listed: &[&str],
) -> usize {
    let all: usize = sources
        .iter()
        .map(|(_, code)| calls_of(code, STOP_HELPER).len())
        .sum();
    let in_loops: usize = found
        .iter()
        .filter(|(file, l)| listed.contains(&l.key(file).as_str()))
        .map(|(_, l)| calls_of(&l.code, STOP_HELPER).len())
        .sum();
    all.saturating_sub(in_loops)
}

// ── The flag of a request ───────────────────────────────────────────────────
//
// The flag that a loop reads is the one flag of its request, made by the job
// runner of the server. A loop stops for nobody when a caller hands it another
// flag, or when a fn keeps the flag under another name. So the rule is on the
// use of the flag. A flag parameter is a fn parameter of type `&AtomicBool`,
// whatever its name. The job closure of the runner binds the flag to `cancel`.
// Each of them is used as the argument of the stop helper or as the bare
// argument of a call, in no other way, and never in a pattern.

/// The name that the job closure of the runner gives to its stop flag.
pub const FLAG: &str = "cancel";

/// The fn that starts a job of the server and hands it the flag of its request.
const RUNNER_FN: &str = "run_admitted";

/// The shape of a fn as a call must match it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FnShape {
    pub name: String,
    /// The parameters of the fn, `self` not counted.
    pub params: usize,
    /// `true` when the fn takes `self`, `&self` or `&mut self`.
    pub receiver: bool,
    /// The place of the first parameter of type `&AtomicBool` among the
    /// parameters, and its name, if the fn has one.
    pub flag: Option<(usize, String)>,
}

/// The offset of the delimiter that closes the group which the first char of
/// `s` opens. With `angles`, `<` opens a group too, as in a signature.
fn group_end(s: &str, angles: bool) -> Option<usize> {
    let mut depth = 0usize;
    let mut previous = ' ';
    for (at, c) in s.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            '<' if angles => depth += 1,
            ')' | ']' | '}' => depth = depth.checked_sub(1)?,
            '>' if angles && previous != '-' => depth = depth.checked_sub(1)?,
            _ => {}
        }
        if depth == 0 {
            return Some(at);
        }
        previous = c;
    }
    None
}

/// `true` for the receiver of a method: `self`, `&self`, `&mut self`.
fn is_receiver(param: &str) -> bool {
    let head = param.split(':').next().unwrap_or_default();
    head.trim()
        .trim_start_matches('&')
        .split_whitespace()
        .last()
        == Some("self")
}

/// The parameters of the fn named `name` that `text` declares, the receiver
/// left out, and whether it had one.
fn parameters(text: &str, name: &str) -> Option<(Vec<String>, bool)> {
    let (_, rest) = text.split_once(&format!("fn {name}"))?;
    let mut rest = rest.trim_start();
    if rest.starts_with('<') {
        rest = rest.get(group_end(rest, true)? + 1..)?.trim_start();
    }
    if !rest.starts_with('(') {
        return None;
    }
    let inner = rest.get(1..group_end(rest, true)?)?;
    let mut found = Vec::new();
    let (mut depth, mut start, mut previous) = (0usize, 0usize, ' ');
    for (at, c) in inner.char_indices() {
        match c {
            '(' | '[' | '{' | '<' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            '>' if previous != '-' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                found.push(inner.get(start..at)?.trim().to_owned());
                start = at + 1;
            }
            _ => {}
        }
        previous = c;
    }
    found.push(inner.get(start..)?.trim().to_owned());
    found.retain(|param| !param.is_empty());
    let receiver = found.iter().any(|param| is_receiver(param));
    found.retain(|param| !is_receiver(param));
    Some((found, receiver))
}

/// `true` when `ty` is `&AtomicBool`, with a path or a lifetime.
fn is_flag_type(ty: &str) -> bool {
    let rest = ty.trim().strip_prefix('&').unwrap_or("").trim_start();
    let rest = if rest.starts_with('\'') {
        rest.split_once(char::is_whitespace).map_or("", |(_, r)| r)
    } else {
        rest
    };
    let rest = rest.trim();
    rest == "AtomicBool" || rest.ends_with("::AtomicBool")
}

/// The name of `param` when it is a flag parameter: `name: &AtomicBool`.
fn flag_param_name(param: &str) -> Option<&str> {
    let (name, ty) = param.split_once(':')?;
    is_flag_type(ty).then_some(name.trim())
}

/// The shape of every fn of `code`.
pub fn fn_shapes(code: &str) -> Vec<FnShape> {
    fns_of(code)
        .into_iter()
        .filter_map(|(name, text)| {
            let (params, receiver) = parameters(&text, &name)?;
            let flag = params
                .iter()
                .enumerate()
                .find_map(|(at, p)| flag_param_name(p).map(|n| (at, n.to_owned())));
            Some(FnShape {
                params: params.len(),
                name,
                receiver,
                flag,
            })
        })
        .collect()
}

/// `true` when `shape` takes a flag.
pub fn takes_the_flag(shape: &FnShape) -> bool {
    shape.flag.is_some()
}

/// The arguments of a call at its top level: nested calls, closures with a
/// braced body, index brackets and the parameter list `|a, b|` of a closure
/// stay in one piece, with its `&`, `&mut` or `move` in front. A trailing comma
/// gives no empty last argument.
pub fn top_level_args(args: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let (mut depth, mut start, mut in_closure_params) = (0usize, 0usize, false);
    for (at, c) in args.char_indices() {
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            '|' if depth == 0 && in_closure_params => in_closure_params = false,
            '|' if depth == 0 => {
                let piece = args.get(start..at).unwrap_or_default();
                in_closure_params = piece
                    .split(|c: char| c == '&' || c.is_whitespace())
                    .all(|word| matches!(word, "" | "mut" | "move"));
            }
            ',' if depth == 0 && !in_closure_params => {
                found.push(args.get(start..at).unwrap_or_default());
                start = at + 1;
            }
            _ => {}
        }
    }
    let last = args.get(start..).unwrap_or_default();
    if !last.trim().is_empty() {
        found.push(last);
    }
    found
}

/// The innermost bracket around `at`: its offset and its char.
fn enclosing_bracket(code: &str, at: usize) -> Option<(usize, char)> {
    let mut depth = 0usize;
    for (i, c) in code.get(..at)?.char_indices().rev() {
        match c {
            ')' | ']' | '}' => depth += 1,
            '(' | '[' | '{' if depth == 0 => return Some((i, c)),
            '(' | '[' | '{' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// `true` when the parenthesis at `open` opens the parameters of a fn.
fn opens_a_signature(code: &str, open: usize) -> bool {
    let before = code.get(..open).unwrap_or_default();
    before
        .rfind("fn ")
        .and_then(|at| before.get(at + 3..))
        .is_some_and(|between| !between.contains([';', '{', '}']))
}

/// `true` when the parenthesis at `open` opens the arguments of a call.
fn opens_a_call(code: &str, open: usize) -> bool {
    code.get(..open)
        .and_then(|before| before.chars().next_back())
        .is_some_and(|c| is_ident(c) || c == '>')
}

/// `true` when the parenthesis at `open` opens the arguments of the runner.
fn opens_the_runner(code: &str, open: usize) -> bool {
    code.get(..open)
        .is_some_and(|before| before.ends_with(RUNNER_FN))
}

/// `true` when the call whose arguments open at `open` is a pattern: it is
/// followed by `=` (a `let`, an `if let`, a `while let`, an arm), by `in`, by
/// `|` or by a guard.
fn is_a_pattern(code: &str, open: usize) -> bool {
    let Some(close) = code.get(open..).and_then(|s| group_end(s, false)) else {
        return false;
    };
    let after = code
        .get(open + close + 1..)
        .unwrap_or_default()
        .trim_start();
    let word = |w: &str| {
        after
            .strip_prefix(w)
            .is_some_and(|rest| !rest.starts_with(is_ident))
    };
    (after.starts_with('=') && !after.starts_with("=="))
        || (after.starts_with('|') && !after.starts_with("||"))
        || word("in")
        || word("if")
}

/// `true` when the name `name` at `at` is a use that the rule allows: the
/// parameter of a fn, the flag parameter of a job closure of the runner, a
/// segment of a path, or a bare argument of a call that is not a pattern.
fn is_an_allowed_use(code: &str, at: usize, name: &str) -> bool {
    let before = code.get(..at).unwrap_or_default().trim_end();
    let after = code.get(at + name.len()..).unwrap_or_default().trim_start();
    let inside_parens = enclosing_bracket(code, at).filter(|(_, c)| *c == '(');
    let after_an_opener = before.ends_with('(') || before.ends_with(',');
    if after.starts_with("::") {
        return true;
    }
    if before.ends_with('|') && after.starts_with('|') {
        return name == FLAG && inside_parens.is_some_and(|(open, _)| opens_the_runner(code, open));
    }
    if after.starts_with(':') && after_an_opener {
        let ty = after.get(1..).unwrap_or_default().trim_start();
        let ty = ty.split([',', ')']).next().unwrap_or_default();
        return is_flag_type(ty)
            && inside_parens.is_some_and(|(open, _)| opens_a_signature(code, open));
    }
    after_an_opener
        && (after.starts_with(')') || after.starts_with(','))
        && inside_parens
            .is_some_and(|(open, _)| opens_a_call(code, open) && !is_a_pattern(code, open))
}

/// The line of `code` that holds the offset `at`, trimmed.
fn line_at(code: &str, at: usize) -> String {
    let head = code.get(..at).unwrap_or_default();
    let from = head.rfind('\n').map_or(0, |n| n + 1);
    let rest = code.get(from..).unwrap_or_default();
    rest.lines().next().unwrap_or_default().trim().to_owned()
}

/// The names that stand for the flag in `code`: `cancel`, and the name of each
/// flag parameter.
fn flag_names(code: &str) -> Vec<String> {
    let mut names = vec![FLAG.to_owned()];
    names.extend(
        fn_shapes(code)
            .into_iter()
            .filter_map(|s| s.flag.map(|f| f.1)),
    );
    names.sort();
    names.dedup();
    names
}

/// The uses of the flag in `code` that the rule does not allow, each as its
/// line: a binding to another name, a read of the flag, a field, a `static`, a
/// flag made in place, a pattern, a closure parameter outside the runner.
///
/// The name `cancel` is read in the whole file. The name of any other flag
/// parameter is read in the fn that declares it.
pub fn flag_uses_off_the_rule(code: &str) -> Vec<String> {
    let mut found: Vec<String> = word_offsets(code, FLAG)
        .into_iter()
        .filter(|at| !is_an_allowed_use(code, *at, FLAG))
        .map(|at| line_at(code, at))
        .collect();
    for (name, text) in fns_of(code) {
        let Some((params, _)) = parameters(&text, &name) else {
            continue;
        };
        for param in params.iter().filter_map(|p| flag_param_name(p)) {
            if param == FLAG {
                continue;
            }
            found.extend(
                word_offsets(&text, param)
                    .into_iter()
                    .filter(|at| !is_an_allowed_use(&text, *at, param))
                    .map(|at| line_at(&text, at)),
            );
        }
    }
    found
}

/// A call of a fn that takes the flag, with the place of the flag among the
/// arguments of the call.
struct FlagCall<'c> {
    at: usize,
    args: &'c str,
    flag_at: usize,
}

/// `true` when the call of `name` at `at` is a call of a closure that the fn
/// holding the call binds with `let name = |..|`, `let mut name = |..|` or
/// `let name = move |..|`. It is no call of a fn of that name. A call after
/// `.` or `::` is a call of a method or a path, never of the closure.
fn is_a_local_closure(code: &str, name: &str, at: usize) -> bool {
    let head = code.get(..at).unwrap_or_default();
    if head.ends_with('.') || head.ends_with("::") {
        return false;
    }
    let (mut fn_start, mut offset) = (0, 0);
    for line in head.split_inclusive('\n') {
        if fn_name(line).is_some() {
            fn_start = offset;
        }
        offset += line.len();
    }
    let body = head.get(fn_start..).unwrap_or_default();
    body.match_indices(name).any(|(i, _)| {
        let before = body.get(..i).unwrap_or_default();
        let binds = !before.ends_with(is_ident)
            && (ends_with_word(before.trim_end(), "let")
                || ends_with_word(before.trim_end().trim_end_matches("mut").trim_end(), "let"));
        let after = body.get(i + name.len()..).unwrap_or_default().trim_start();
        let closure = after
            .strip_prefix('=')
            .filter(|rest| !rest.starts_with('='))
            .map(str::trim_start)
            .map(|rest| rest.strip_prefix("move").map_or(rest, str::trim_start))
            .is_some_and(|rest| rest.starts_with('|'));
        binds && closure
    })
}

/// The fns of `shapes` named `name` that a call with `count` arguments can be
/// a call of, each with the shift of the flag place: a path call
/// `Type::method(receiver, ..)` of a method has one argument more.
fn fns_a_call_can_be<'s>(
    shapes: &'s [FnShape],
    name: &str,
    count: usize,
    by_path: bool,
) -> Vec<(&'s FnShape, usize)> {
    shapes
        .iter()
        .filter(|s| s.name == name)
        .filter_map(|s| {
            let shift = if s.params == count {
                0
            } else if by_path && s.receiver && s.params + 1 == count {
                1
            } else {
                return None;
            };
            Some((s, shift))
        })
        .collect()
}

/// The names of the fns in `shapes` that take the flag.
fn flag_fn_names(shapes: &[FnShape]) -> Vec<&str> {
    let mut names: Vec<&str> = shapes
        .iter()
        .filter(|s| takes_the_flag(s))
        .map(|s| s.name.as_str())
        .collect();
    names.sort_unstable();
    names.dedup();
    names
}

/// The calls in `code` of a fn that takes the flag.
///
/// A call is matched to a fn by name and count of arguments. When another fn
/// of `shapes` matches and takes no flag, the call is not read: the text
/// cannot tell which fn it calls. A call of a local closure is read by neither
/// function. A call that matches no fn is reported by
/// [`flag_calls_that_match_no_fn`].
fn flag_calls<'c>(code: &'c str, shapes: &[FnShape]) -> Vec<FlagCall<'c>> {
    let mut found = Vec::new();
    for name in flag_fn_names(shapes) {
        for call in calls_of(code, name) {
            if is_a_local_closure(code, name, call.at) {
                continue;
            }
            let count = top_level_args(call.args).len();
            let by_path = code.get(..call.at).is_some_and(|b| b.ends_with("::"));
            let matched = fns_a_call_can_be(shapes, name, count, by_path);
            let flags: Vec<usize> = matched
                .iter()
                .filter_map(|(s, shift)| s.flag.as_ref().map(|f| f.0 + shift))
                .collect();
            if !matched.is_empty() && flags.len() == matched.len() {
                found.push(FlagCall {
                    at: call.at,
                    args: call.args,
                    flag_at: flags.first().copied().unwrap_or_default(),
                });
            }
        }
    }
    found
}

/// The calls in `code` of a name that a fn taking the flag has, whose count of
/// arguments matches no fn of that name, each as its line number, the name,
/// the counts and its line. The rule cannot find the place of the flag in such
/// a call, so it is a failure and not a pass.
fn flag_calls_that_match_no_fn(code: &str, shapes: &[FnShape]) -> Vec<String> {
    let mut found = Vec::new();
    for name in flag_fn_names(shapes) {
        for call in calls_of(code, name) {
            if is_a_local_closure(code, name, call.at) {
                continue;
            }
            let count = top_level_args(call.args).len();
            let by_path = code.get(..call.at).is_some_and(|b| b.ends_with("::"));
            if !fns_a_call_can_be(shapes, name, count, by_path).is_empty() {
                continue;
            }
            let mut takes: Vec<usize> = shapes
                .iter()
                .filter(|s| s.name == name)
                .map(|s| s.params)
                .collect();
            takes.sort_unstable();
            takes.dedup();
            let line = code
                .get(..call.at)
                .unwrap_or_default()
                .matches('\n')
                .count()
                + 1;
            found.push(format!(
                "line {line}: `{name}` called with {count} arguments, the fns of that name take {takes:?}: {}",
                line_at(code, call.at)
            ));
        }
    }
    found
}

/// The calls in `code` that do not hand the flag of their caller on, each as
/// its line. A call of a fn that takes the flag hands a bare flag of its caller
/// at the place of the flag, and a call of the stop helper reads one. A call
/// whose count of arguments matches no fn of its name is one of them.
pub fn calls_off_the_rule(code: &str, shapes: &[FnShape]) -> Vec<String> {
    let names = flag_names(code);
    let is_a_flag = |arg: &str| names.iter().any(|n| n == arg.trim());
    let mut found = Vec::new();
    for call in flag_calls(code, shapes) {
        let args = top_level_args(call.args);
        if !args.get(call.flag_at).is_some_and(|arg| is_a_flag(arg)) {
            found.push(line_at(code, call.at));
        }
    }
    found.extend(flag_calls_that_match_no_fn(code, shapes));
    for call in calls_of(code, STOP_HELPER) {
        if !is_a_flag(call.args) {
            found.push(line_at(code, call.at));
        }
    }
    found
}

/// The pairs of fns in `shapes` with one name, one count of parameters and the
/// flag at two places. A call cannot be matched to either, so the pair is
/// refused.
pub fn flag_fns_that_disagree(shapes: &[FnShape]) -> Vec<String> {
    let mut found = Vec::new();
    for (i, a) in shapes.iter().enumerate() {
        for b in shapes.iter().skip(i + 1) {
            let (Some(fa), Some(fb)) = (&a.flag, &b.flag) else {
                continue;
            };
            if a.name == b.name && a.params == b.params && a.receiver == b.receiver && fa.0 != fb.0
            {
                found.push(format!("{}: the flag at {} and at {}", a.name, fa.0, fb.0));
            }
        }
    }
    found
}

/// `true` when `code` calls a fn that takes the flag, read as
/// [`calls_off_the_rule`] reads a call.
pub fn calls_a_flag_fn(code: &str, shapes: &[FnShape]) -> bool {
    !flag_calls(code, shapes).is_empty()
}

/// `true` when `before` ends with the word `word`.
fn ends_with_word(before: &str, word: &str) -> bool {
    before
        .strip_suffix(word)
        .is_some_and(|rest| !rest.ends_with(is_ident))
}

/// The lines of `code` that give the stop helper another meaning: a rename on
/// import, or a binding of the name by `let`, `const` or `static`.
pub fn helper_shadowed(code: &str) -> Vec<String> {
    word_offsets(code, STOP_HELPER)
        .into_iter()
        .filter(|at| {
            let before = code.get(..*at).unwrap_or_default().trim_end();
            ["as", "let", "mut", "const", "static"]
                .iter()
                .any(|w| ends_with_word(before, w))
        })
        .map(|at| line_at(code, at))
        .collect()
}

// ── The arms that log a failure ─────────────────────────────────────────────
//
// A result that can be the stop must not be logged as a failure: the request
// is gone, and the stop is a normal event. So a `match` arm `Err(e)` that logs
// `e` at `warn` or `error` has an arm for the stop before it.

/// The `Err(name) => ..` arms in `code` that log `name` at `warn` or `error`
/// and have no `Err(Error::Cancelled)` arm before them in their `match`, each
/// as its line.
pub fn failure_arms_without_a_stop_arm(code: &str) -> Vec<String> {
    code.match_indices("Err(")
        .filter_map(|(at, _)| {
            let rest = code.get(at + 4..)?;
            let close = rest.find(')')?;
            let name = rest.get(..close)?;
            if name.is_empty() || !name.chars().all(is_ident) {
                return None;
            }
            let after = rest.get(close + 1..)?.trim_start().strip_prefix("=>")?;
            let after = after.trim_start();
            let body = if after.starts_with('{') {
                after.get(..=group_end(after, false)?)?
            } else {
                after.split([',', '\n']).next()?
            };
            let logs = ["warn!", "error!"].iter().any(|m| body.contains(m));
            if !logs || !names(body, name) {
                return None;
            }
            let (open, _) = enclosing_bracket(code, at)?;
            let before = code.get(open..at)?;
            let stop_arm = ["Err(Error::Cancelled)", "Err(rmlx_core::Error::Cancelled)"]
                .iter()
                .any(|a| before.contains(a));
            (!stop_arm).then(|| line_at(code, at))
        })
        .collect()
}

// ── The sweep that ends a prefill ───────────────────────────────────────────
//
// Every cache that enters its prefill state must leave it, whatever the chunks
// did. A `?` or a `return` between the entry and the sweep skips the sweep on
// the first error and on a stop. The sweep is a loop over the same caches as
// the entry loop, in the block of the entry or in plain blocks inside it, with
// no `if`, `match` or adapter around it or on it, and each entry has its own.

/// `text` with each char literal replaced by spaces of the same length.
fn without_char_literals(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut kept = String::with_capacity(text.len());
    let mut i = 0;
    while let Some(&c) = chars.get(i) {
        let closing = if c != '\'' {
            None
        } else if chars.get(i + 1) == Some(&'\\') {
            Some(i + 3).filter(|end| chars.get(*end) == Some(&'\''))
        } else {
            Some(i + 2).filter(|end| chars.get(*end) == Some(&'\''))
        };
        if let Some(end) = closing {
            for gone in chars.get(i..=end).unwrap_or_default() {
                kept.push_str(&" ".repeat(gone.len_utf8()));
            }
            i = end + 1;
        } else {
            kept.push(c);
            i += 1;
        }
    }
    kept
}

/// The offset after the braced body of the closure that opens with the `|` at
/// `bar`, if it opens one.
fn closure_body_end(text: &str, bar: usize) -> Option<usize> {
    let before = text.get(..bar)?.trim_end();
    if !before.ends_with(['=', '(', ',']) && !before.ends_with("move") {
        return None;
    }
    let params_end = bar + 1 + text.get(bar + 1..)?.find('|')?;
    if text.get(bar + 1..params_end)?.contains([';', '{', '}']) {
        return None;
    }
    let after = text.get(params_end + 1..)?;
    let brace = after.find('{')?;
    let head = after.get(..brace)?.trim();
    if !(head.is_empty() || head.starts_with("->")) || head.contains(';') {
        return None;
    }
    let body = params_end + 1 + brace;
    Some(body + group_end(text.get(body..)?, false)? + 1)
}

/// `text` with the braced body of each closure left out: a `?` or a `return` in
/// it ends the closure and not the fn around it.
fn without_closure_bodies(text: &str) -> String {
    let mut kept = String::new();
    let mut rest = text;
    while let Some(bar) = rest.find('|') {
        if let Some(end) = closure_body_end(rest, bar) {
            kept.push_str(rest.get(..bar).unwrap_or_default());
            rest = rest.get(end..).unwrap_or_default();
        } else {
            kept.push_str(rest.get(..=bar).unwrap_or_default());
            rest = rest.get(bar + 1..).unwrap_or_default();
        }
    }
    kept.push_str(rest);
    kept
}

/// A `for` loop that holds an offset directly: where its statement starts and
/// the iterated expression.
struct ForLoop {
    stmt: usize,
    iterated: String,
}

/// The `for` loop whose body holds `at` directly, if the innermost brace
/// around `at` is the body of one.
fn for_loop_around(code: &str, at: usize) -> Option<ForLoop> {
    let (open, c) = enclosing_bracket(code, at)?;
    if c != '{' {
        return None;
    }
    let before = code.get(..open)?;
    let stmt = before.rfind([';', '{', '}']).map_or(0, |p| p + 1);
    let header = before.get(stmt..)?.trim();
    let (_, iterated) = header.strip_prefix("for ")?.split_once(" in ")?;
    Some(ForLoop {
        stmt,
        iterated: iterated.split_whitespace().collect::<Vec<_>>().join(" "),
    })
}

/// `true` when the brace at `open` opens a block that runs once, in order: a
/// block expression, not the body of an `if`, a `match`, a loop or a closure.
fn is_a_plain_block(code: &str, open: usize) -> bool {
    code.get(..open)
        .map(str::trim_end)
        .is_some_and(|b| b.ends_with(['=', ';', '{', '}']))
}

/// For each call of `enter_prefill` in `fn_text`, the stretch from that call to
/// its sweep, the bodies of closures left out; or the reason that the sweep is
/// not the one the rule asks for.
pub fn prefill_sweeps(fn_text: &str) -> Vec<Result<String, &'static str>> {
    let code = without_char_literals(fn_text);
    let entries: Vec<usize> = calls_of(&code, "enter_prefill")
        .iter()
        .map(|c| c.at)
        .collect();
    let sweeps: Vec<usize> = calls_of(&code, "exit_prefill")
        .iter()
        .map(|c| c.at)
        .collect();
    entries
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let sweep = *sweeps
                .iter()
                .find(|at| *at > entry)
                .ok_or("no sweep after the entry")?;
            if entries.get(i + 1).is_some_and(|next| *next < sweep) {
                return Err("another entry comes before the sweep");
            }
            let entry_loop = for_loop_around(&code, *entry).ok_or("the entry is not in a loop")?;
            let sweep_loop =
                for_loop_around(&code, sweep).ok_or("the sweep is not directly in a loop")?;
            let adapter = [".take(", ".skip(", ".filter(", ".step_by(", ".rev("]
                .iter()
                .any(|a| entry_loop.iterated.contains(a));
            if adapter || entry_loop.iterated != sweep_loop.iterated {
                return Err("the sweep loop does not iterate the caches of the entry loop");
            }
            let (entry_block, _) =
                enclosing_bracket(&code, entry_loop.stmt).ok_or("the entry loop is in no block")?;
            let mut at = sweep_loop.stmt;
            loop {
                let (open, c) = enclosing_bracket(&code, at)
                    .ok_or("the sweep is not in the block of the entry")?;
                if open == entry_block {
                    break;
                }
                if c != '{' || !is_a_plain_block(&code, open) {
                    return Err("the sweep is inside a branch or a loop");
                }
                at = open;
            }
            Ok(without_closure_bodies(
                code.get(*entry..sweep).unwrap_or_default(),
            ))
        })
        .collect()
}

/// `true` when `stretch` can leave its fn: a `?`, or a `return`.
pub fn can_leave(stretch: &str) -> bool {
    stretch.contains('?') || names(stretch, "return")
}

fn count_events(at_most: tracing::Level, f: impl FnOnce()) -> usize {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct Count(Arc<AtomicUsize>, tracing::Level);
    impl tracing::Subscriber for Count {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            if *event.metadata().level() <= self.1 {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    let seen = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::with_default(Count(Arc::clone(&seen), at_most), f);
    seen.load(Ordering::Relaxed)
}

/// The number of tracing events at `ERROR` that `f` emits on this thread.
pub fn error_events(f: impl FnOnce()) -> usize {
    count_events(tracing::Level::ERROR, f)
}

/// The number of tracing events at `WARN` or `ERROR` that `f` emits on this
/// thread.
pub fn loud_events(f: impl FnOnce()) -> usize {
    count_events(tracing::Level::WARN, f)
}

/// The number of tracing events at `INFO` or above that `f` emits on this
/// thread.
pub fn events_at_info_or_above(f: impl FnOnce()) -> usize {
    count_events(tracing::Level::INFO, f)
}
