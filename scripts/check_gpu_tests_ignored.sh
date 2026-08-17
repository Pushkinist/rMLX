#!/usr/bin/env bash
# scripts/check_gpu_tests_ignored.sh — CI gate: fail if a GPU-touching test in
# ANY workspace member crate is missing the `#[ignore]` Metal-context attribute.
#
# WHY
#   A shared Metal context cannot be driven from parallel test threads. A GPU
#   test left un-ignored runs under the default `cargo test` alongside its
#   siblings and aborts the whole binary ("Rust cannot catch foreign
#   exceptions"), taking every other test in that binary down with it. Such
#   tests pass when run in isolation, so a PR's own targeted run looks green.
#   The abort is load-dependent — a couple of parallel GPU tests can pass for a
#   long time before enough of them tip the binary over — so "it passes today"
#   is not evidence of compliance. That is why this is a static gate.
#
# SCOPE
#   Every member crate of the workspace (read from `Cargo.toml [workspace]
#   members`, never hard-coded), across both of its test roots:
#     * unit tests under `src/` — files named `<name>_tests.rs` OR bare
#       `tests.rs` (the file-size convention in CLAUDE.md allows both; a glob
#       of `*_tests.rs` alone silently misses every `tests.rs`).
#     * integration binaries under `tests/*.rs`.
#   `cargo test -p <crate> --lib` does not build `tests/` at all, so a check
#   scoped to `src/` alone would mirror exactly the blind spot it exists to
#   catch.
#
# WHAT COUNTS AS A TEST
#   `#[test]`, `#[tokio::test]`, and `#[tokio::test(flavor = ..)]`. An async
#   test attribute is a test attribute; matching only the bare spelling left
#   ~107 of them in this tree unclassified in BOTH directions — never flagged,
#   never listed — which is the same hole macro-generated tests were in.
#
# WHAT COUNTS AS GPU-TOUCHING (shape, not message)
#   A test fn that reaches `Device::Gpu`, directly in its own body or
#   transitively through a helper it calls (including generic helpers called
#   with a turbofish, and methods on an inherent impl). `Device::Gpu` bound to
#   a module-scope `const NAME: Device = Device::Gpu;` counts too — a body that
#   uses NAME is GPU-touching. The check keys on that shape alone — never on
#   the ignore *reason* text, which varies across the tree and would make the
#   gate green while the class stayed live.
#
# MACRO-GENERATED TESTS
#   A `macro_rules!` body that emits `#[test] fn $name() { .. }` declares a test
#   whose fn line names a `$metavar`, and whose invocation sites emit no `fn`
#   line at all. A classifier that only accepts `fn <ident>` sees such a test in
#   NEITHER direction: it is never flagged however much Metal it dispatches, and
#   it never reaches the compliant list either — so the rule is upheld by author
#   discipline alone and deleting the attribute from the macro body is caught by
#   nothing.
#
#   So the macro BODY is classified as one synthetic test, at its definition
#   site. That is where the attribute under enforcement actually lives, and one
#   body governs every cell it generates: a `#[ignore]` deleted there trips the
#   gate no matter how many invocations exist. Its reachability is traced from
#   the body exactly like a normal fn's, so a body calling a GPU helper is
#   GPU-touching. A test-generating macro that is never invoked is still held to
#   the rule — the body is the unit, so it is a violation with no live cells.
#   That is deliberate: `unused_macros` under `-D warnings` deletes dead bodies
#   anyway, so the case should not arise, and fail-closed is the right side to
#   err on if it does.
#
#   Fail-closed companions — both report `U` and both exist because a shape this
#   parser cannot read is indistinguishable from compliance:
#     * a body that contains `#[test]` but from which NO test item could be
#       extracted (a name assembled by `paste!` / `concat_idents!`, or a
#       `#[test]` sharing its `fn`'s line);
#     * a `macro_rules!` written entirely on ONE line whose body declares
#       `#[test]` — no `fn` line ever follows it, so nothing is classifiable.
#   The first is also what stops the original blindness from recurring quietly:
#   narrowing the fn-name recognition again does not hide macro cells, it makes
#   the declared-vs-readable counts disagree and the gate goes red.
#
#   Macro-generated tests are enforced but deliberately NOT emitted to `--list`;
#   see the LIST vs ENFORCE note below.
#
# REACHABILITY IS WHOLE-CRATE, NOT PER-FILE
#   The fn call-graph is built across ALL of a crate's scanned files together
#   (not one file at a time). A `#[test]` that reaches Metal ONLY through a
#   helper defined in a DIFFERENT scanned file — the canonical case being a
#   `tests/common/mod.rs` helper such as `run_golden_test` that binds
#   `let device = Device::Gpu;` and is called as `common::run_golden_test(..)`
#   from each `tests/<arch>_golden_tokens.rs` binary — is now seen. Call
#   resolution is collision-safe (two files may define same-named helpers):
#     * an UNqualified call `helper(..)` binds to a same-file definition only;
#     * a QUALIFIED call `module::helper(..)` binds to a helper defined in the
#       file whose module name is `module` (the directory name for a `mod.rs`,
#       otherwise the file stem).
#   So a CPU-only `helper()` in file A is never tainted by a same-named GPU
#   `helper()` in unrelated file B — the unqualified call binds to A's own.
#
# RESIDUAL (documented honestly — the gate narrows the blind spot, it does not
# claim to erase every path):
#   * A helper defined in a NON-scanned regular source file (e.g.
#     `crates/rmlx-models/src/paroquant_msl.rs`) reached from its sibling
#     `*_tests.rs` via `use super::*` is not traced — only the crate's SCANNED
#     test roots are in the graph. (The present instance, `paro_rotate_kernel()`,
#     builds a `MetalKernel` and never names `Device::Gpu`, so no gate shape
#     matches it regardless of reachability.)
#   * An UNqualified cross-file call resolved through a glob import
#     (`use module::*; helper()`) is not traced across files — cross-file
#     binding requires the `module::` qualifier. This is the price of
#     collision-safety and it fails toward MISSING a path, never toward a
#     false positive.
#   * A test generated by anything other than a `macro_rules!` body in a
#     SCANNED file — a proc-macro attribute (`#[rstest]`, `#[test_case]`), or a
#     `macro_rules!` defined in a non-scanned source file and invoked here — is
#     still invisible. Nothing in the tree has that shape today; adding one
#     needs this classifier extended, not a review note.
#   * A `#[test]` written on the SAME line as its `fn` is missed for plain fns
#     too (the attribute line is consumed whole). Inside a macro body the
#     fail-closed `U` cases above catch it; at top level they do not, and
#     rustfmt's own layout is what keeps the shape out of the tree. Note that
#     rustfmt does NOT normalise a `macro_rules!` body containing a `$(..)*`
#     repetition, which is why the macro side is fail-closed rather than
#     relying on `make fmt-check`.
#   * A `macro_rules!` with a NON-brace delimiter (`macro_rules! m ( .. );`) has
#     its name captured for labelling, but its body extent is not tracked, so
#     the readability counters above do not apply to it. Items inside it are
#     still classified — a `$metavar` fn is macro-generated by its own shape —
#     so recall is unaffected; only the fail-closed net is.
#
# THE FIX FOR A VIOLATION — pick the one that is true:
#   * The test really drives the GPU -> add the attribute:
#       #[ignore = "GPU Metal context — run in isolation: \
#                   cargo test <filter> -- --ignored --test-threads=1"]
#     Wrapping the reason across lines with a trailing `\` is safe: the
#     attribute capture below closes on the line whose last SIGNIFICANT
#     character is `]`, which is the `"]` such a continuation ends in. It was
#     not always so — a close-test keyed on the wrapped-`#[cfg(..)]` shape `)]`
#     alone left this exact spelling latched for the rest of the file, so the
#     documented way to comply was the way to blind the gate.
#   * The test only exercises a guard that returns before any GPU work ->
#     pass `Device::Cpu`, and leave it un-ignored so it keeps running in the
#     default gate.
#
# THE CONVERSE IS FATAL TOO — an `#[ignore]` whose reason claims a Metal
# context on a test from which no `Device::Gpu` is reachable. Every such test is
# skipped by `make test` (it is ignored) AND skipped by `make gpu-test` (it is
# not classified), so it runs nowhere while reading, at both gates, exactly like
# a test that is covered.
#
# It was a non-fatal warning, on the reasoning that some ignores are legitimately
# non-GPU and that a helper in a non-scanned source file makes a real Metal test
# look GPU-free here. Both are true and neither is a reason to leave the finding
# advisory: an advisory channel with two valid outcomes and no way to record
# which one applies is a channel nothing ever closes. It sat open over six such
# tests. So the finding now demands a disposition, and there are three:
#
#   * the test drives Metal by a route this scanner cannot follow -> declare it
#     with the `metal-unscanned` marker below;
#   * the test does not touch the GPU -> drop the `#[ignore]` (and pass
#     `Device::Cpu`), so it runs in the default gate again;
#   * the test is ignored for some other reason -> say that reason in the
#     `#[ignore]` text instead of claiming Metal.
#
# The third is a real escape hatch and is stated rather than hidden: this check
# keys on the ignore reason's WORDING, which is the one place in this gate that
# does. It has to — there is no shape to key on, that being the whole problem —
# and the consequence is that a Metal-driving test whose ignore text never says
# "Metal" or "GPU" is invisible to it.
#
# DECLARED METAL ROUTE (`// gpu-test-gate: metal-unscanned`)
#   The inverse of the exemption below: a line-leading marker in the fn's own
#   attribute block declaring that the test DOES drive Metal, by a route no
#   shape in the scanned sources can express. Two such routes exist here:
#
#     * an HTTP boundary — the test drives an in-process axum router and the
#       device is chosen inside `crates/rmlx-server/src/embeddings.rs`, which is
#       not a scanned file and could not be linked to the test even if it were
#       (the call goes through a routing table, not a call graph);
#     * a process boundary — the test spawns `rmlx serve` and the Metal context
#       belongs to the child.
#
#   Effect: the test counts as GPU-touching, so the `#[ignore]` rule bites on it
#   — deleting the attribute is a violation, which before the marker it was not.
#   It is NOT emitted to `--list`; see LIST vs ENFORCE. A marker on a test the
#   scanner CAN see through, and a marker paired with the exemption, are both
#   hard failures: the first is stale, the second says the test does and does not
#   drive Metal, and guessing which half to believe is how a marker rots.
#
#   Like the exemption, it is opt-in — but not opt-in the way that word usually
#   means "and therefore forgettable". The fatal rule above is what asks for it:
#   an `#[ignore]` claiming Metal with nothing to back it fails the gate until
#   one of the three dispositions is recorded.
#
# PER-TEST EXEMPTION (device-as-value false positives)
#   Detection is by shape: naming `Device::Gpu`. In compute crates that always
#   means a Metal dispatch. In higher crates `Device` is a plain config enum —
#   a test may pass `Device::Gpu` to a PURE function that only branches on it
#   (e.g. a CLI truncation-policy resolver) and never touches Metal. No local
#   shape separates `pure_fn(.., Device::Gpu, ..)` from `mlx_op(.., Device::Gpu)`,
#   and the seed must stay broad (the real GPU tests bind `let d = Device::Gpu;`,
#   so any narrower match would miss them). Such a test opts out with a
#   line-leading marker inside its OWN attribute/comment block:
#       // gpu-test-gate: exempt
#   The exemption is scoped to that one `#[test]` fn — NOT the whole file — so a
#   Metal-driving test added to the same file still trips the gate. The marker
#   must lead its line and sit in the fn's attribute block; a copy inside a fn
#   body does not exempt. A reviewer audits each marker; use it ONLY for a test
#   that passes `Device::Gpu` as a value and never dispatches Metal.
#
#   INSIDE A `macro_rules!` BODY THE BLAST RADIUS IS EVERY CELL. The body is one
#   synthetic test, so one marker line exempts every test that macro generates —
#   for the shape in this tree, roughly thirty cells from a single comment.
#   "Scoped to one `#[test]`" is still literally true and badly understates it:
#   review a marker inside a macro body against every invocation, not one.
#
# LIST vs ENFORCE (they are not the same population, on purpose)
#   `--list` feeds `scripts/run_gpu_tests.sh`, which turns each name into a
#   libtest substring filter. A macro-generated test has no name until the
#   compiler expands it — the classifier holds the macro's name, not the cells'
#   — so listing it would emit a filter that selects nothing and trip that
#   runner's "executed fewer than classified" check. Macro-generated tests are
#   therefore ENFORCED here and excluded from `--list`; the enforcing run prints
#   the excluded set so the divergence is stated rather than silent.
#
#   A `metal-unscanned` test is excluded for a different reason, and it is a
#   property of the RUNNER, not of the classifier. `run_gpu_tests.sh` asserts,
#   per crate, that Metal's shader-validation banner appeared — a crate that
#   created no Metal device proved nothing and is reported as having run
#   uninstrumented. Every `metal-unscanned` test in this tree is snapshot-gated
#   or drives a child process, so a machine without that snapshot would execute
#   them, watch them all return early, produce no banner, and fail the suite for
#   a missing model rather than a defect. They are enforced here and listed
#   nowhere; the excluded set is printed every run alongside the macro one.
#
#   That divergence is a real cost and is accepted deliberately for the one
#   population that has it today, the NIAH long-context cells: they are
#   snapshot-gated, run 8k–32k-token prefills, and already have a purpose-built
#   driver (`scripts/release_e2e/stage6_perf/niah_long_context.sh`, wired into
#   `make smoke-codec-matrix`). Folding ~60 model loads into the pre-merge GPU
#   suite would make it depend on model snapshots and cost hours. What the
#   present change buys is the half that was genuinely unenforced: the
#   `#[ignore]` attribute on the macro body.
#
# PARSING
#   Relies on rustfmt layout (already enforced by `make fmt-check`): a `fn`
#   item's closing brace sits at the same indent as its `fn` keyword. That
#   makes brace/string depth tracking unnecessary. rustfmt does not reformat a
#   `macro_rules!` body, so the same layout there is a convention — which is why
#   an unreadable body fails closed (`U`) instead of being skipped.
#
#   Two consequences are handled explicitly rather than assumed away:
#     * An item that CLOSES on its own line is self-contained, and latching the
#       multi-line capture on it would wait for a close that never comes —
#       swallowing every later item in the file into its body and silently
#       ending classification. Such a line is captured whole and never latches.
#       "Closes on its own line" is decided from the line's LAST significant
#       character (`}` ends an inline body, `;` ends a signature-only
#       declaration), never from a brace COUNT: `{` / `}` inside a string, a
#       char literal or a comment are not block delimiters, and counting them
#       breaks the decision in both directions — a false "balanced" drops the
#       item's body so a `Device::Gpu` in it is never seen, a false "open"
#       swallows the rest of the file.
#
#       KNOWN OPEN BLIND SPOT. The `;` arm covers a signature that fits on ONE
#       line. A `where` clause pushes the `;` onto a later line, so such a
#       declaration latches — and the latch is then closed by the first later
#       line that bares to the fn's indent, which in a Rust test file is `    }`,
#       the close of any nested block. When it closes that way NO `U` is
#       emitted: the swallowed `#[test]` was never registered, so nothing looks
#       unterminated, and the gate reports OK at exit 0 over an un-ignored
#       `Device::Gpu` test. This is fail-OPEN, not fail-closed. It is
#       unreachable in the tree today (no scanned file has a where-split
#       signature) and the class is tracked for reconciliation against the
#       compiled `cargo test -- --list`. The `trait_where_signature` fixture
#       pins only the sub-case where nothing closes the latch;
#       `trait_where_signature_open_hole` pins the open answer.
#     * A capture that is still open at a file boundary means the parser lost
#       the file, so it reports `U` rather than letting the remainder go
#       unclassified. This holds for all THREE captures — fn item, macro body
#       and attribute block.
#
#   ATTRIBUTES ARE A THIRD CAPTURE, with the same latch hazard. An attribute
#   that does not close on its own line latches until one does, and while
#   latched it consumes every line — including every `fn` — so a latch that
#   never ends silently unclassifies the rest of the file. The close-test is
#   therefore the same "last SIGNIFICANT character" rule the fn arm uses, not a
#   match against one wrapped spelling: reading only `)]` (how a wrapped
#   `#[cfg(..)]` ends) missed the wrapped STRING form `"]`, and reading the raw
#   line missed any attribute with a trailing comment. The string state is
#   carried across the line break (`bare()`'s `q0` / `bare_q`), so a `//` in the
#   continued payload of a wrapped string is not mistaken for a comment.
#
#   `bare()` finds the trailing comment with a string-aware scan, so neither a
#   `//` inside a string literal nor a char literal of any payload form (`b'"'`,
#   `'\x1b'`, `'\u{FFFD}'`, `'é'`) derails it. Remaining known parse hazards,
#   which apply to the attribute close-test exactly as they do to the fn one:
#   raw strings (the `\` in `r"a\"` is not an escape, and `r#"…"#` hashes are
#   not tracked) and block comments — a `/* … */` spanning an item's opening
#   line is read literally. An attribute whose closing `]` is hidden behind one
#   of those latches, and what happens next splits two ways — state both rather
#   than claim the good one:
#     * nothing later in the file bares to `]` -> the file boundary reports `U`.
#       Fail-closed.
#     * something later does — a subsequent `#[test]` is exactly that shape ->
#       the latch ends there, classification resumes correctly, and the items
#       swallowed in between are gone with NO report. Fail-OPEN, the same shape
#       as the `where`-split hole above, and not detectable from here;
#       reconciliation against the compiled `cargo test -- --list` is what would
#       catch it.
#   What the close-test buys is not immunity but reachability: getting into that
#   state now requires one of the two `bare()` hazards, where before it needed
#   only the ordinary wrapped-string spelling.
#
#   PORTABLE AWK ONLY — this runs on the developer's BSD awk and on whatever
#   the Linux CI image provides. Two constructs are out of bounds because they
#   do not mean the same thing everywhere, and getting them wrong takes the gate
#   from "narrow" to "does not execute at all":
#     * a bracket RANGE whose endpoints are escapes (`[\300-\337]`) — a hard
#       syntax error in gawk, silently accepted by BSD awk and mawk;
#     * the POSIX classes as a stand-in for "is this byte ASCII" — BSD awk and
#       mawk call a UTF-8 continuation byte `[[:print:]]` under a UTF-8 locale
#       and not under C, and gawk indexes characters where they index bytes.
#   Prefer `index()` / `substr()` arithmetic, which counts in the same units as
#   `length()` and `RLENGTH` in every implementation.
#   `scripts/check_gpu_tests_ignored_fixtures.sh` re-runs its whole corpus under
#   every awk installed, so a violation of this surfaces there.
#
# Exit 0 = clean. Exit 1 = violation (or a scan that found nothing to check).

set -euo pipefail

# --list: print the COMPLIANT set (GPU-touching tests that carry #[ignore]) as
# `<crate><TAB><fn>` and exit, instead of enforcing. `scripts/run_gpu_tests.sh`
# consumes this so the rule and the runner cannot cover different populations —
# a GPU test this gate mandates is by construction a GPU test that gets run.
#   (Macro-generated tests are the stated exception; see LIST vs ENFORCE above.)
#
# --root <dir>: scan <dir> instead of the repository. Used by
# `scripts/check_gpu_tests_ignored_fixtures.sh` to drive the classifier over
# synthetic workspaces whose expected exit code is known, so a change that loses
# recall fails there rather than passing silently on the real tree.
LIST_MODE=0
ROOT_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --list)
            LIST_MODE=1
            shift
            ;;
        --root)
            ROOT_OVERRIDE="${2:-}"
            if [ -z "${ROOT_OVERRIDE}" ]; then
                echo "ERROR: --root needs a directory argument." >&2
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument '$1' (expected --list and/or --root <dir>)." >&2
            exit 1
            ;;
    esac
done

if [ -n "${ROOT_OVERRIDE}" ]; then
    if [ ! -d "${ROOT_OVERRIDE}" ]; then
        echo "ERROR: --root '${ROOT_OVERRIDE}' is not a directory." >&2
        exit 1
    fi
    REPO_ROOT="$(cd "${ROOT_OVERRIDE}" && pwd)"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
CARGO_TOML="${REPO_ROOT}/Cargo.toml"

# Fail closed: the workspace manifest is the source of the member list. If it
# is gone the gate is scanning nothing.
if [ ! -f "${CARGO_TOML}" ]; then
    echo "ERROR: ${CARGO_TOML} not found — cannot resolve workspace members." >&2
    exit 1
fi

# Read `[workspace] members = [ ... ]` — the single source of the crate list.
# Never hard-code a crate root: a moved/renamed crate must surface as a
# fail-closed error below, not as a silently narrowed scan.
#
# NOTE: this parser assumes the one-member-per-line array layout. `make
# fmt-check` (rustfmt) does NOT format `Cargo.toml`, so that layout is a
# convention, not an enforced invariant — a reformat to a single-line array or
# an inline comment could silently narrow the scan. The plausibility floor
# below (parsed members vs. crate dirs on disk) guards against exactly that.
members=()
while IFS= read -r m; do
    [ -n "$m" ] && members+=("$m")
done < <(awk '
    /^[[:space:]]*members[[:space:]]*=[[:space:]]*\[/ { inm = 1; next }
    inm {
        line = $0
        if (line ~ /\]/) { inm = 0 }
        gsub(/[",]/, "", line)
        gsub(/[[:space:]]/, "", line)
        gsub(/\[/, "", line)
        gsub(/\]/, "", line)
        if (line != "" && line !~ /^#/) { print line }
    }
' "${CARGO_TOML}")

if [ ${#members[@]} -eq 0 ]; then
    echo "ERROR: parsed 0 workspace members from ${CARGO_TOML}." >&2
    echo "A gate that scans nothing passes everything; refusing to report OK." >&2
    exit 1
fi

# Plausibility floor: every crate under crates/ (a dir with its own Cargo.toml)
# must be a parsed member. If the parse yields fewer, the members array layout
# changed and the scan silently narrowed — the exact scope-loss class this gate
# guards against. Fail closed. (A crate deliberately excluded from the workspace
# would trip this; that is a reviewable event, not a silent narrowing.)
disk_crates=0
if [ -d "${REPO_ROOT}/crates" ]; then
    disk_crates=$(find "${REPO_ROOT}/crates" -mindepth 2 -maxdepth 2 \
        -name Cargo.toml -not -path "*/target/*" | wc -l | tr -d ' ')
fi
if [ "${#members[@]}" -lt "${disk_crates}" ]; then
    echo "ERROR: parsed ${#members[@]} workspace members but ${disk_crates} crate dirs exist under crates/." >&2
    echo "The members array in ${CARGO_TOML} likely changed layout and narrowed the scan." >&2
    echo "Restore the one-member-per-line layout, or reconcile the members list." >&2
    exit 1
fi

# The awk detector, shared across every crate's scan. Reads all of a crate's
# scanned files in one pass (FILENAME distinguishes them) and builds a
# whole-crate fn call-graph, so a cross-file helper is reachable. Emits
# `V  <file>: <fn>` for a violation and `W  <file>: <fn>` for the converse
# warning. Kept in a variable so the per-crate loop invokes one identical
# program over each crate's file list.
read -r -d '' AWK_DETECT <<'AWK' || true
    # The line's significant text: trailing line comment and trailing blanks
    # removed. Used both to compare a closing brace against its marker (`}
    # // end of macro` must still close) and to read the line's LAST
    # significant character, which is what decides whether an item is
    # self-contained.
    #
    # The `//` scan is string-aware. A `//` inside a string literal — a URL is
    # the everyday case — is not a comment, and cutting there would end the
    # line in mid-string and flip every decision that reads its last
    # character.
    #
    # `q0` seeds that string state, and `bare_q` reports where the scan ended,
    # so a caller walking a MULTI-LINE construct can carry the state across the
    # line break. A string continued with a trailing `\` is still open on the
    # next line; scanning that line from q=0 would read its payload as code and
    # cut at the first `//` in it. Callers that read one self-contained line
    # pass neither and get the old q=0 behaviour exactly.
    function bare(s, q0,   i, n, q, c, rest, w, ascii) {
        n = length(s)
        q = q0 + 0
        for (i = 1; i <= n; i++) {
            c = substr(s, i, 1)
            if (c == "\\" && q) { i++; continue }
            if (c == "\"") { q = !q; continue }
            if (!q && c == "'") {
                # Step over the WHOLE char literal. A partially-stepped one
                # leaves its closing quote to be re-read as an opener, and that
                # stray `'` matches the one-char shape below and swallows a real
                # `"` — putting the scan back inside a string for the rest of
                # the line. A lifetime (`'a`, `'static`) matches no shape here
                # and is left alone.
                rest = substr(s, i + 1)
                if (match(rest, /^\\u\{[0-9a-fA-F][0-9a-fA-F]*\}'/) \
                 || match(rest, /^\\x[0-9a-fA-F][0-9a-fA-F]'/) \
                 || match(rest, /^\\.'/) \
                 || match(rest, /^[^'\\]'/)) {
                    i += RLENGTH
                    continue
                }
                # A non-ASCII payload (`'é'`) is ONE character but 2-4 bytes, so
                # the one-unit shape above does not match it where substr walks
                # bytes. Locate the closing quote by INDEX in a bounded window:
                # index/substr/length/RLENGTH all count in the same units in
                # every awk, so the offset lands correctly whether the
                # implementation walks bytes or characters.
                #
                # Deliberately not a byte class. `[\300-\337]` is a hard syntax
                # error in gawk (the gate then does not run AT ALL, which is
                # worse than the blind spot it closes), and `[[:print:]]` is no
                # better: BSD awk and mawk both classify a UTF-8 continuation
                # byte as printable under a UTF-8 locale but not under C. An
                # index() against a literal ASCII set is locale-independent, and
                # it is also what stops a lifetime tick with a nearby quote
                # (`<'a>'x'`) from being stepped as though it were a literal.
                ascii = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
                w = index(substr(rest, 1, 5), "'")
                if (w >= 3 && index(ascii, substr(rest, 1, 1)) == 0 \
                 && index(substr(rest, 1, w - 1), "\\") == 0) {
                    i += w
                }
                continue
            }
            if (!q && c == "/" && substr(s, i + 1, 1) == "/") {
                s = substr(s, 1, i - 1)
                break
            }
        }
        bare_q = q
        sub(/[[:space:]]+$/, "", s)
        return s
    }

    # Tear down a fn item whose closing brace was never found. The capture is
    # greedy — every later line joins that fn's body — so silence here means
    # the rest of the file went unclassified. That is the same blind spot the
    # gate exists to prevent, so it fails closed rather than reporting a scan.
    function flush_fn() {
        if (in_fn) {
            printf "U  %s: %s (fn never closed — parser lost the file from here)\n",
                file_of[curid], label_of[curid]
        }
        in_fn = 0
        curid = 0
        close_marker = ""
    }

    # Tear down the open `macro_rules!` body, reporting it when it declared
    # more `#[test]`s than this parser could read back as items. Called at the
    # body's closing brace, at a file boundary, and at END — an unterminated
    # body must not swallow the check.
    function flush_macro() {
        if (in_macro && macro_test_n > macro_fn_n) {
            printf "U  %s: %s! (%d #[test] declared, %d readable)\n",
                macro_file, macro_name, macro_test_n, macro_fn_n
        }
        in_macro = 0
        macro_name = ""
        macro_close = ""
        macro_file = ""
        macro_test_n = 0
        macro_fn_n = 0
    }

    # Tear down an attribute capture whose closing `]` was never found. The
    # capture is greedy — every later line, including every `fn`, joins the
    # attribute block — so silence here means the rest of the file was never
    # classified while the gate reported a clean scan. That is fail-OPEN and it
    # is the one direction this gate must never take, so an attribute the
    # close-test could not read is a hard failure, exactly like an unterminated
    # fn or macro body.
    function flush_attr() {
        if (in_attr) {
            printf "U  %s: %s (attribute never closed — parser lost the file from here)\n",
                attr_file, attr_label
        }
        in_attr = 0
        attr_q = 0
        attr_file = ""
        attr_label = ""
    }

    # New file: reset the per-file parse state and derive this file's module
    # name (directory name for a mod.rs, otherwise the file stem). The module
    # name is how a QUALIFIED cross-file call `module::helper(..)` binds.
    FNR == 1 {
        flush_fn()
        flush_macro()
        flush_attr()
        attrs = ""; last_macro_name = ""
        nseg = split(FILENAME, seg, "/")
        base = seg[nseg]
        if (base == "mod.rs" && nseg >= 2) {
            curmod = seg[nseg - 1]
        } else {
            curmod = base
            sub(/\.rs$/, "", curmod)
        }
    }

    # ── Multi-line attribute continuation ────────────────────────────────
    # Closed by the line's SIGNIFICANT text ending in `]`, carrying the string
    # state across the line break. Keying this on the wrapped-`#[cfg(..)]`
    # shape `)]` alone reads only one of the two ways an attribute wraps: a
    # wrapped string argument ends `"]`, never `)]`, so such an attribute never
    # closed and the capture swallowed the rest of the FILE — every later item
    # unclassified, and a clean OK printed over them. The `)]` arm is kept
    # ahead of the general one: it needs no string scan, so it still closes a
    # wrapped list whose last line the scan below would misread.
    in_attr {
        attrs = attrs " " $0
        sig = bare($0, attr_q)
        attr_q = bare_q
        if ($0 ~ /^[[:space:]]*\)\]/ || sig ~ /\]$/) { in_attr = 0 }
        next
    }
    # ── Attribute (any indent) ──────────────────────────────────────────
    /^[[:space:]]*#\[/ {
        attrs = attrs " " $0
        # Count `#[test]`s declared inside a macro body, so the END check can
        # compare them against the test items actually extracted from it.
        if (in_macro && !in_fn && $0 ~ /#\[(tokio::)?test[](]/) { macro_test_n++ }
        # Self-contained iff the SIGNIFICANT text closes it. Read from the raw
        # line, a trailing comment (`#[test] // why`) hides the closing `]` and
        # latches the capture over the rest of the file — the same fail-open as
        # the continuation rule above, reached by an easier-to-write shape.
        if (bare($0) !~ /\]$/) {
            in_attr = 1
            attr_q = bare_q
            attr_file = FILENAME
            attr_label = $0
            sub(/^[[:space:]]+/, "", attr_label)
        }
        next
    }
    # ── Per-test exemption marker (line-leading, in the attr block) ─────
    # Folds into the pending attribute block so it attaches to the NEXT fn
    # only. Guarded by !in_fn so a copy inside a body cannot exempt.
    !in_fn && /^[[:space:]]*\/\/[[:space:]]*gpu-test-gate:[[:space:]]*exempt([[:space:]]|$)/ {
        attrs = attrs " GATE_EXEMPT"
        next
    }
    # ── Declared Metal route the scanner cannot follow ──────────────────
    # The inverse marker: the test DOES drive Metal, across an HTTP or a
    # process boundary no source shape here can express. Folds into the
    # pending attribute block on the same terms as the exemption — attaches
    # to the NEXT fn only, and a copy inside a body declares nothing.
    !in_fn && /^[[:space:]]*\/\/[[:space:]]*gpu-test-gate:[[:space:]]*metal-unscanned([[:space:]]|$)/ {
        attrs = attrs " GATE_UNSCANNED"
        next
    }
    # ── Comments / doc comments keep the pending attribute block ────────
    /^[[:space:]]*\/\// { next }

    # ── Module-scope `const NAME: Device = Device::Gpu;` ────────────────
    # A test that uses NAME (rather than the Device::Gpu literal) is still
    # GPU-touching; record the alias, scoped to THIS file, so a same-named
    # const in another file does not leak across the crate.
    !in_fn && /^[[:space:]]*const[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:[[:space:]]*Device[[:space:]]*=[[:space:]]*Device::Gpu/ {
        cname = $0
        sub(/^[[:space:]]*const[[:space:]]+/, "", cname)
        sub(/[^A-Za-z0-9_].*$/, "", cname)
        gpu_const[FILENAME SUBSEP cname] = 1
        attrs = ""
        next
    }

    # ── `macro_rules! NAME` — a body that may declare tests ─────────────
    # Tracked so a `fn $metavar` inside it is recognised as a macro-generated
    # item, and so a body holding a `#[test]` we could NOT read fails closed.
    # The name is captured for ANY delimiter so a finding is greppable; only a
    # brace-delimited body is tracked to its end (see the non-brace note in
    # RESIDUAL).
    !in_fn && !in_macro && /^[[:space:]]*macro_rules![[:space:]]*[A-Za-z_][A-Za-z0-9_]*/ {
        macro_name = $0
        sub(/^[[:space:]]*macro_rules![[:space:]]*/, "", macro_name)
        sub(/[^A-Za-z0-9_].*$/, "", macro_name)
        last_macro_name = macro_name
        if ($0 !~ /^[[:space:]]*macro_rules![[:space:]]*[A-Za-z_][A-Za-z0-9_]*[[:space:]]*\{/) {
            # Non-brace delimiter: the body's extent is not tracked. Items in
            # it are still classified (a `$metavar` fn is macro-generated by
            # its own shape) and now carry this name.
            macro_name = ""
            attrs = ""
            next
        }
        if (bare($0) ~ /\}$/) {
            # The whole body is on this one line, so no `fn` line will ever
            # follow. If it declares a test, that test is unreadable — fail
            # closed. If it declares none, the line is simply not interesting.
            if ($0 ~ /#\[(tokio::)?test[](]/) {
                printf "U  %s: %s! (one-line body declaring #[test])\n",
                    FILENAME, macro_name
            }
            macro_name = ""
            attrs = ""
            next
        }
        in_macro = 1
        macro_close = $0
        sub(/[^[:space:]].*$/, "", macro_close)   # leading whitespace only
        macro_close = macro_close "}"
        macro_file = FILENAME
        macro_fn_n = 0
        # A `#[test]` sharing the opener line is never seen by the attribute
        # rule below (this rule consumes the line), so count it here.
        macro_test_n = ($0 ~ /#\[(tokio::)?test[](]/) ? 1 : 0
        attrs = ""
        next
    }
    # ── Close of the macro body: `}` at the `macro_rules!` indent ───────
    # Every `#[test]` the body declared must have yielded a test item. A
    # shortfall means a name this parser cannot read (assembled by `paste!` /
    # `concat_idents!`, or an attribute sharing its `fn`'s line) — the test
    # exists and is unclassifiable, which is a gate blind spot, not a clean
    # scan. flush_macro() reports it.
    in_macro && !in_fn && bare($0) == macro_close {
        flush_macro()
        attrs = ""
        next
    }

    # ── fn item at any indent (top-level, inside an inherent impl, or
    #    inside a `macro_rules!` body where the name is a `$metavar`) ────
    !in_fn && /^[[:space:]]*(pub[^ ]* )?(async )?fn [$a-zA-Z0-9_]/ {
        line = $0
        indent = line
        sub(/[^[:space:]].*$/, "", indent)   # leading whitespace only
        name = line
        sub(/^[[:space:]]*(pub[^ ]* )?(async )?fn /, "", name)
        sub(/[^$a-zA-Z0-9_].*$/, "", name)
        G++
        order[++ord_n] = G
        name_of[G] = name
        file_of[G] = FILENAME
        mod_of[G] = curmod
        attrs_of[G] = attrs
        body_of[G] = line
        # A `$metavar` fn name is macro-generated by construction: its real
        # names exist only after expansion. Recorded even when the enclosing
        # `macro_rules!` line was not matched (a non-brace macro delimiter), so
        # such an item can never leak into `--list` as an unusable filter.
        if (substr(name, 1, 1) == "$") {
            gen_of[G] = (macro_name != "" ? macro_name \
                : (last_macro_name != "" ? last_macro_name : "<macro>"))
            label_of[G] = gen_of[G] "!{" name "}"
        } else {
            gen_of[G] = ""
            label_of[G] = name
        }
        if (in_macro && attrs ~ /#\[(tokio::)?test[](]/) { macro_fn_n++ }
        curid = G
        # Self-contained iff the line's last significant character closes it:
        # `}` ends an inline body, `;` ends a signature-only declaration (a
        # trait method, an extern block). Anything else — `{`, or a `(` from a
        # multi-line signature — continues onto later lines and must latch.
        #
        # A brace COUNT cannot make this call: `{` / `}` inside a string, a
        # char literal or a trailing comment are not block delimiters, and
        # counting them breaks the decision in BOTH directions. A false
        # "balanced" drops the fn's body, so a `Device::Gpu` inside it never
        # reaches body_of and the test is silently unclassified; a false "open"
        # latches and swallows every later item in the file.
        if (bare(line) ~ /(\}|;)$/) {
            in_fn = 0
            curid = 0
            close_marker = ""
        } else {
            in_fn = 1
            close_marker = indent "}"
        }
        attrs = ""
        next
    }
    # ── Close of the captured fn: `}` at the fn keyword indent ──────────
    in_fn && bare($0) == close_marker {
        in_fn = 0
        curid = 0
        attrs = ""
        next
    }
    # Body lines, with line comments stripped: prose that names a GPU
    # helper or Device::Gpu must not make the fn look GPU-touching.
    in_fn {
        line = $0
        sub(/\/\/.*$/, "", line)
        body_of[curid] = body_of[curid] " " line
        next
    }
    # Any other line drops a dangling attribute block.
    /^[[:space:]]*[^[:space:]]/ { attrs = "" }

    END {
        # An item left open by the last file still gets its check — otherwise
        # an unterminated fn, macro body or attribute is a free pass.
        flush_fn()
        flush_macro()
        flush_attr()

        # ── Seed: fns naming Device::Gpu (literal) or a file-scoped
        #    `const … = Device::Gpu` alias in their own body. Seeds feed a
        #    worklist; the GPU set is small, so we pop each GPU fn once and
        #    taint its callers rather than sweep every pair every round.
        qh = 0; qt = 0
        for (g = 1; g <= G; g++) {
            b = body_of[g]
            isg = (b ~ /Device::Gpu/) ? 1 : 0
            if (!isg) {
                f = file_of[g]
                for (key in gpu_const) {
                    si = index(key, SUBSEP)
                    kf = substr(key, 1, si - 1)
                    kc = substr(key, si + 1)
                    if (kf == f && b ~ ("(^|[^A-Za-z0-9_])" kc "([^A-Za-z0-9_]|$)")) {
                        isg = 1
                        break
                    }
                }
            }
            if (isg) { gpu[g] = 1; queue[++qt] = g }
        }

        # ── Fixed point via worklist. Pop a GPU fn h; any not-yet-GPU fn
        #    that CALLS h becomes GPU. Resolution is collision-safe:
        #      * same file  -> unqualified/method call `h(` (any qualifier
        #        char admitted, as a file has no two free fns of one name);
        #      * other file -> QUALIFIED call `mod_of[h] :: h (`.
        #    A cheap index() substring prune avoids compiling the anchored
        #    regex for the vast majority of (caller, callee) pairs.
        #    A macro-generated item is a leaf: its name is a `$metavar`, which
        #    nothing calls and which would be read as an anchor inside the
        #    regexes below, so it is never used as a callee.
        while (qh < qt) {
            h = queue[++qh]
            if (gen_of[h] != "") { continue }
            nm = name_of[h]
            hf = file_of[h]
            hq = mod_of[h]
            same_re = "(^|[^a-zA-Z0-9_])" nm "([[:space:]]*::[[:space:]]*<[^;{]*>)?[[:space:]]*\\("
            qual_re = "(^|[^A-Za-z0-9_])" hq "[[:space:]]*::[[:space:]]*" nm "([[:space:]]*::[[:space:]]*<[^;{]*>)?[[:space:]]*\\("
            for (c = 1; c <= G; c++) {
                if (gpu[c]) { continue }
                if (index(body_of[c], nm) == 0) { continue }
                matched = 0
                if (file_of[c] == hf) {
                    if (body_of[c] ~ same_re) { matched = 1 }
                }
                if (!matched && body_of[c] ~ qual_re) { matched = 1 }
                if (matched) {
                    gpu[c] = 1
                    queue[++qt] = c
                }
            }
        }

        # ── Report, in source order across the crate's files ────────────
        # `#[tokio::test]` (with or without arguments) is a test attribute
        # exactly as much as `#[test]` is, and this crate tree has ~107 of
        # them. Matching only the bare spelling left every one of them
        # unclassified in BOTH directions — never flagged, never listed — which
        # is the same hole macro-generated tests were in.
        for (i = 1; i <= ord_n; i++) {
            g = order[i]
            if (attrs_of[g] !~ /#\[(tokio::)?test[](]/) { continue }
            has_ignore = (attrs_of[g] ~ /#\[ignore/)
            exempt = (attrs_of[g] ~ /GATE_EXEMPT/)
            declared = (attrs_of[g] ~ /GATE_UNSCANNED/)

            # Two marker states have no right answer, so both fail closed
            # rather than this picking one. A marker on a test the reachability
            # pass CAN see through is stale — its claim is now checkable and
            # keeping it would hold the test out of `--list` forever. A marker
            # beside the exemption asserts that the test both does and does not
            # drive Metal.
            if (declared && exempt) {
                printf "U  %s: %s (marked metal-unscanned AND exempt — pick one)\n",
                    file_of[g], label_of[g]
                continue
            }
            if (declared && gpu[g]) {
                printf "U  %s: %s (marked metal-unscanned, but Device::Gpu is reachable here)\n",
                    file_of[g], label_of[g]
                continue
            }

            # A declared route counts as GPU-touching for the attribute rule:
            # deleting the `#[ignore]` from such a test is a violation, which
            # before the marker existed it was not.
            if ((gpu[g] || declared) && !has_ignore && !exempt) {
                printf "V  %s: %s\n", file_of[g], label_of[g]
            }
            # The compliant set: GPU-touching AND ignored. This is exactly the
            # population `scripts/run_gpu_tests.sh` must execute, so it is
            # derived from the same classifier rather than from a second,
            # drifting list. Exempt fns are device-as-value, not Metal.
            #
            # A macro-generated item goes to S instead: the runner turns each
            # listed name into a libtest filter, and a `$metavar` matches no
            # test, so listing it would under-match and trip the runner's own
            # coverage check. S is reported, not dropped.
            if (gpu[g] && has_ignore && !exempt) {
                if (gen_of[g] != "") {
                    printf "S  %s: %s\n", file_of[g], label_of[g]
                } else {
                    printf "T  %s: %s\n", file_of[g], name_of[g]
                }
            }
            # A declared route is enforced above but never listed: the runner
            # asserts a per-crate Metal validation banner, and every test with
            # this marker is snapshot-gated or drives a child process, so
            # listing it would fail that assertion on any machine without the
            # snapshot. Reported, not dropped.
            if (declared && has_ignore) {
                printf "N  %s: %s\n", file_of[g], label_of[g]
            }
            # Converse: an ignore claiming a Metal context on a test that
            # never reaches one and never declared why. It is skipped by the
            # default gate (it is ignored) and by the GPU gate (it is not
            # classified), so it runs nowhere while looking covered at both.
            if (!gpu[g] && !declared && has_ignore \
             && attrs_of[g] ~ /#\[ignore[^]]*([Mm]etal|GPU)/) {
                printf "W  %s: %s\n", file_of[g], label_of[g]
            }
        }
    }
AWK

# Fail closed per member. A member listed in Cargo.toml whose dir or `src/` is
# missing means the crate moved or was renamed — break loudly rather than let
# the gate report OK over a narrowed scan (this workspace has extracted crates
# before; see the dep graph in CLAUDE.md).
total_files=0
violations=""
warnings=""
gpu_tests=""
unreadable=""
macro_generated=""
declared_unscanned=""

for crate in "${members[@]}"; do
    crate_dir="${REPO_ROOT}/${crate}"
    src_dir="${crate_dir}/src"
    tests_dir="${crate_dir}/tests"

    if [ ! -d "${crate_dir}" ]; then
        echo "ERROR: member crate '${crate}' from Cargo.toml has no directory at ${crate_dir}." >&2
        echo "The crate moved or was renamed; fix the members list or the path." >&2
        exit 1
    fi
    if [ ! -d "${src_dir}" ]; then
        echo "ERROR: ${src_dir} does not exist — this gate is scanning nothing for '${crate}'." >&2
        echo "The crate moved or was renamed; update Cargo.toml members." >&2
        exit 1
    fi

    # The cargo package name, for `--list` consumers that feed it to
    # `cargo test -p`. Read from the manifest rather than assumed equal to the
    # directory basename: the two happen to match for every member today, but a
    # mismatch would surface as an opaque "package not found" from cargo instead
    # of failing at this gate, which is the fail-closed discipline the rest of
    # this script follows.
    pkg_name="$(awk -F'"' '/^name[[:space:]]*=/{print $2; exit}' "${crate_dir}/Cargo.toml" 2>/dev/null)"
    if [ -z "${pkg_name}" ]; then
        echo "ERROR: could not read [package] name from ${crate_dir}/Cargo.toml." >&2
        exit 1
    fi

    # Gather this crate's scanned files. Whole-crate reachability needs every
    # scanned file of the crate in ONE awk pass, so collect them per crate.
    crate_files=()
    # Unit tests: match both `<name>_tests.rs` and bare `tests.rs`.
    while IFS= read -r -d '' f; do crate_files+=("$f"); done < <(
        find "${src_dir}" \( -name "*_tests.rs" -o -name "tests.rs" \) \
            -not -path "*/target/*" -print0
    )
    # Integration binaries + their shared modules (optional per crate).
    if [ -d "${tests_dir}" ]; then
        while IFS= read -r -d '' f; do crate_files+=("$f"); done < <(
            find "${tests_dir}" -name "*.rs" -not -path "*/target/*" -print0
        )
    fi

    [ ${#crate_files[@]} -eq 0 ] && continue
    total_files=$((total_files + ${#crate_files[@]}))

    out=$(awk "$AWK_DETECT" "${crate_files[@]}")
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        case "$line" in
            "V  "*) violations="${violations}  ${line#V  }"$'\n' ;;
            "W  "*) warnings="${warnings}  ${line#W  }"$'\n' ;;
            "U  "*) unreadable="${unreadable}  ${line#U  }"$'\n' ;;
            "S  "*) macro_generated="${macro_generated}  ${line#S  }"$'\n' ;;
            "N  "*) declared_unscanned="${declared_unscanned}  ${line#N  }"$'\n' ;;
            "T  "*) gpu_tests="${gpu_tests}${pkg_name}"$'\t'"${line##*: }"$'\n' ;;
        esac
    done <<< "$out"
done

if [ "${total_files}" -eq 0 ]; then
    echo "ERROR: matched 0 test files across ${#members[@]} workspace members." >&2
    echo "A gate that scans nothing passes everything; refusing to report OK." >&2
    exit 1
fi

# An unreadable test-generating macro fails BOTH modes. The classification is
# incomplete either way: enforcing over a test it could not read reports OK on
# an unchecked test, and listing over it hands the runner a population that is
# missing one. This is the same fail-closed rule as "scanned 0 files".
if [ -n "$unreadable" ]; then
    echo "ERROR: this gate could not classify part of a scanned file" >&2
    echo "       (an unreadable macro_rules! body, or an item or attribute that" >&2
    echo "        never closed, ending classification mid-file):" >&2
    printf '%s' "$unreadable" >&2
    echo >&2
    echo "The gate classifies a test-generating macro from its body: it needs an" >&2
    echo "attribute block followed by a line-leading \`fn \$<metavar>(\` (or a literal" >&2
    echo "fn name). A name assembled by paste!/concat_idents!, a #[test] sharing" >&2
    echo "its fn's line, and a whole macro_rules! on one line are all invisible —" >&2
    echo "and an invisible GPU test is exactly what this gate exists to prevent." >&2
    echo >&2
    echo "A 'fn never closed' or 'attribute never closed' finding means the parser" >&2
    echo "lost the file at that point, so everything after it went unclassified. All" >&2
    echo "are reported rather than skipped, because a shape this gate cannot read" >&2
    echo "looks exactly like a compliant one." >&2
    echo >&2
    echo "Write the generated fn as \`fn \$name()\` on its own line with its" >&2
    echo "attributes above it, close the attribute on a line whose last significant" >&2
    echo "character is \`]\`, or extend the classifier to read the new shape." >&2
    exit 1
fi

# An `#[ignore]` claiming a Metal context with no reachable `Device::Gpu` and no
# declared route fails BOTH modes, for the same reason the unreadable case does:
# such a test is skipped by `make test` because it is ignored and skipped by
# `make gpu-test` because it is not classified, so it runs nowhere while reading
# as covered at both gates. This was advisory for a long time and six tests sat
# in it; an advisory channel with two valid outcomes and no way to record which
# one applies is a channel nothing ever closes.
if [ -n "$warnings" ]; then
    echo "ERROR: #[ignore] claims a Metal context but no Device::Gpu is reachable" >&2
    echo "       in the scanned roots, and no route was declared:" >&2
    printf '%s' "$warnings" >&2
    echo >&2
    echo "Each of these runs under NO gate: \`make test\` skips it (it is ignored)" >&2
    echo "and \`make gpu-test\` skips it (it is not classified). Pick the true one:" >&2
    echo >&2
    echo "  * it drives Metal through a route this scanner cannot follow — an HTTP" >&2
    echo "    handler in production source, or a child process — then declare it" >&2
    echo "    with a line-leading marker in the fn's own attribute block:" >&2
    echo "      // gpu-test-gate: metal-unscanned  <why the scanner cannot see it>" >&2
    echo "    It then counts as GPU-touching (the #[ignore] rule bites on it) and" >&2
    echo "    is deliberately NOT listed for scripts/run_gpu_tests.sh." >&2
    echo "  * it does not touch the GPU — drop the #[ignore] and pass Device::Cpu," >&2
    echo "    so it runs in the default gate again." >&2
    echo "  * it is ignored for some other reason — say THAT reason in the #[ignore]" >&2
    echo "    text instead of claiming Metal." >&2
    echo >&2
    echo "See docs/TESTING.md." >&2
    exit 1
fi

# Macro-generated tests are enforced above but excluded from `--list`, so state
# the divergence every run rather than letting it be a property only the source
# comments record. Printed BEFORE the `--list` early exit and on stderr: the one
# surface where "these are not in this run" matters is `make gpu-test`, which
# calls `--list` and nothing else, so a note that only the enforcing run prints
# is invisible to the operator who needs it. stderr keeps the machine-readable
# stdout listing clean.
if [ -n "$macro_generated" ]; then
    echo "NOTE: macro-generated GPU tests — the #[ignore] rule is ENFORCED on the macro" >&2
    echo "body, but the cells are not listed for scripts/run_gpu_tests.sh (their fn names" >&2
    echo "exist only after expansion, so a filter cannot be derived):" >&2
    printf '%s' "$macro_generated" >&2
    echo "  -> These run under their own driver; see docs/TESTING.md." >&2
    echo >&2
fi

# The other enforced-but-unlisted population, stated on the same terms and for
# the same reason: `make gpu-test` calls only `--list`, so an operator reading
# that run must be told what it does not cover.
if [ -n "$declared_unscanned" ]; then
    echo "NOTE: declared-route GPU tests — the #[ignore] rule is ENFORCED on them, but" >&2
    echo "they are not listed for scripts/run_gpu_tests.sh (that runner asserts a Metal" >&2
    echo "validation banner per crate, and these are snapshot-gated or drive a child" >&2
    echo "process, so listing them would fail the suite for a missing model):" >&2
    printf '%s' "$declared_unscanned" >&2
    echo "  -> No gate executes these. See docs/TESTING.md for each one's coverage." >&2
    echo >&2
fi

if [ "${LIST_MODE}" -eq 1 ]; then
    if [ -z "${gpu_tests}" ]; then
        echo "ERROR: classified 0 GPU-touching #[ignore] tests across ${total_files} files." >&2
        echo "The detector found nothing to run; refusing to emit an empty list." >&2
        exit 1
    fi
    printf '%s' "${gpu_tests}"
    exit 0
fi

if [ -n "$violations" ]; then
    echo "ERROR: GPU-touching tests missing the #[ignore] Metal-context attribute:" >&2
    printf '%s' "$violations" >&2
    echo >&2
    echo "A shared Metal context cannot be driven from parallel test threads: an" >&2
    echo "un-ignored GPU test aborts the whole test binary under a plain" >&2
    echo "\`cargo test\`, killing every other test in it." >&2
    echo >&2
    echo "Add to each test that really drives the GPU:" >&2
    echo "  #[ignore = \"GPU Metal context — run in isolation: cargo test <filter> -- --ignored --test-threads=1\"]" >&2
    echo >&2
    echo "If the test only exercises a guard that returns before any GPU work," >&2
    echo "pass Device::Cpu instead and leave it un-ignored." >&2
    exit 1
fi

echo "OK: every GPU-touching test carries #[ignore] (${total_files} files across ${#members[@]} workspace members)."
