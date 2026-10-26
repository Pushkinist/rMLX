#!/usr/bin/env bash
# CI gate: the classified GPU tests carry #[ignore] and name their stand-downs; one tool lists them.
#
# One classifier of the workspace's GPU tests, and the rules and the listing that read its
# population, in three modes:
#
#   ignored       every test that reaches Device::Gpu carries #[ignore], and no #[ignore] hides a
#                 test that no gate runs.
#   skip-notices  every classified test that announces its own stand-down names itself in the
#                 notice, and none stands down silently.
#   list          the compliant set, `<crate><TAB><fn>` per line; with --files a third column
#                 names the declaring file, which is `scripts/gpu_test_halves.sh`'s and
#                 `scripts/gpu_test_threadgroup.sh`'s input. `scripts/run_gpu_tests.sh` turns
#                 the two columns into the tests it runs.
#
# The population is classified once, by the code below the awk program. `ignored` and `list` are
# its two ends, and `skip-notices` reads its listing with the declaring file, so the three cannot
# name different populations. The comments that follow state each rule where it is enforced; the
# first block is the classifier and the `ignored` rule, the second the `skip-notices` rule.
#
# ---- the classifier and the `ignored` rule ------------------------------------------------
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
#     * integration binaries under `tests/**/*.rs`.
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
#   Macro-generated tests are enforced but deliberately NOT emitted to `list`;
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
# A REASONLESS `#[ignore]` ON AN ENVIRONMENT-GATED TEST is fatal for the same
# reason: a test that reads an environment variable — `env::var`, `env::var_os`,
# or the snapshot resolver `test_snapshot::snapshot` — reaches no `Device::Gpu`
# and carries a bare `#[ignore]` runs under no gate, and the rule above reads
# the reason text, which it does not have. Its dispositions: pass
# `Device::Gpu` so the GPU suite lists it, drop the `#[ignore]` so the default
# gate runs it, or write the reason. A bare `#[ignore]` on a test that reads no
# variable, or reads one only through another helper, is outside it.
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
#   It is NOT emitted to `list`; see LIST vs ENFORCE. A marker on a test the
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
#   `list` feeds `scripts/run_gpu_tests.sh`, which turns each name into a
#   libtest substring filter. A macro-generated test has no name until the
#   compiler expands it — the classifier holds the macro's name, not the cells'
#   — so listing it would emit a filter that selects nothing and trip that
#   runner's "executed fewer than classified" check. Macro-generated tests are
#   therefore ENFORCED here and excluded from `list`; the enforcing run prints
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
#   driver (`scripts/release_e2e/stage6_perf/niah_long_context.sh`, run by hand;
#   `make smoke-codec-matrix` runs the same cells through its matrix runner).
#   Folding ~75 model loads into the pre-merge GPU
#   suite would make it depend on model snapshots and cost hours. What the
#   present change buys is the half that was genuinely unenforced: the
#   `#[ignore]` attribute on the macro body.
#
# PARSING
#   Relies on rustfmt layout (already enforced by `make fmt-check`): a `fn`
#   item's closing brace sits at the same indent as its `fn` keyword. That
#   ends a BODY without tracking its braces. A signature has no such marker, so
#   it is read by bracket depth (below). rustfmt does not reformat a
#   `macro_rules!` body, so the same layout there is a convention — which is why
#   an unreadable body fails closed (`U`) instead of being skipped.
#
#   Two consequences are handled explicitly rather than assumed away:
#     * An item that CLOSES on its own line is self-contained, and latching the
#       multi-line capture on it would wait for a close that never comes —
#       swallowing every later item in the file into its body and silently
#       ending classification. Such a line is captured whole and never latches.
#       The signature is read by one reader, from the header line on, over the
#       line's CODE (comments, strings and char literals blanked by `bare()`): it
#       keeps the depths of `(`, `[`, `{` and the generic `<` across the
#       signature's lines, and carries the depth of an open `/* … */` comment
#       across them. A `{` still open at the end of a line is the body, which
#       latches until the `}` at the fn's indent. While a `(`, `[` or `<` is open
#       the signature goes on, whatever the line ends in; a `;` with no `(` or `[`
#       open ends a declaration however many lines the signature ran over, since
#       no generic list holds one; with no generic `<` open either, a `{` opened
#       and closed on the line, ending in `}`, is an inline body and the item is
#       complete, and anything else is a signature that goes on. A `<`
#       or `>` counts as a generic bracket only outside every `(`, `[` and `{`,
#       and the `>` of `->` and `=>` never does.
#       Counting over the raw line instead would break the decision in both
#       directions: a `{` or `}` in a string, a char literal or a comment is not
#       a block delimiter — a false "balanced" drops the item's body so a
#       `Device::Gpu` in it is never seen, a false "open" swallows the rest of
#       the file — and a `;` in `[u8; 4]` or a comment is not a terminator.
#       The body carries the string and comment state of the line that opened it
#       from line to line, so a `}` that starts a line inside a string or a comment
#       closes nothing, and a `//` inside one cuts nothing. Outside a fn the same
#       state is carried too, and across the lines of an attribute.
#     * A capture that is still open at a file boundary means the parser lost
#       the file, so it reports `U` rather than letting the remainder go
#       unclassified. This holds for all FOUR captures — fn item, macro body,
#       attribute block and the string or comment state outside a fn.
#
#   ATTRIBUTES ARE A THIRD CAPTURE, with the same latch hazard. An attribute
#   that does not close on its own line latches until one does, and while
#   latched it consumes every line — including every `fn` — so a latch that
#   never ends silently unclassifies the rest of the file. It closes at its own
#   `]`, found by bracket depth over the code of each line, with the string and
#   block-comment state (`attr_q`, `attr_c`) carried across the line break. An
#   attribute never closed at the file boundary is reported `U`.
#
#   An item after the `]` on its line (`#[test] fn x() {}`, `)] fn x() {`,
#   `macro_rules!`, a `const NAME: Device = Device::Gpu` alias) is refused, since
#   the reader would not follow it. Any other tail (`T,`, `pub u8,`, `mod m {`)
#   feeds no rule and ends the pending attribute block.
#
#   `bare()` finds the trailing comment with a string-aware scan, so neither a
#   `//` inside a string literal nor a char literal of any payload form (`b'"'`,
#   `'\x1b'`, `'\u{FFFD}'`, `'é'`) derails it, a non-ASCII lifetime (`<'é, '"'>`)
#   is no char literal and swallows nothing after it, and a block comment is read
#   before a `//` or a quote in it is looked at. A raw string that does not close
#   on its line is carried like a plain one (the `\` in `r"a\"` is no escape).
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
#   `scripts/check/rust_gates_fixtures.sh` re-runs its whole corpus under
#   every awk installed, so a violation of this surfaces there.
#
# ---- the `skip-notices` rule --------------------------------------------------------------
# WHY
#   `scripts/run_gpu_tests.sh` reads the tests' own words to tell a cell that
#   held from one that was never asked. The notice it reads is
#   `SKIP <test>: <why>`, and the NAME is the whole of the attribution: it is
#   what puts the cell on the stand-down list an operator reads, and what lets
#   the shader-validation census drop that cell's pinned hit count instead of
#   expecting a count the run could not produce.
#
#   A notice spelled `SKIP: <why>` passes the runner's stand-down pattern and
#   names nothing. The runner counts it — it will not attribute it to whichever
#   test was nearby — so the run is marked INCOMPLETE with a number and no list,
#   and the reader is told that something stood down without being told what.
#   No number of snapshots fixes that: the notice cannot be attributed at any
#   point after it is printed.
#
#   So the rule is enforced where it can be fixed — at the source line. This
#   scan is the recall half of the runner's report: the runner says how many
#   notices could not be attributed, this says which lines they come from.
#
# WHAT IS SCANNED
#   The classified GPU tests, from the classifier's `list` — the
#   same population the runner selects, derived from the same classifier rather
#   than listed again here. A notice in a helper, or in a test the classifier
#   does not name, is out of scope: the runner never selects it, so it never
#   reaches a report.
#
# WHAT IS ACCEPTED
#   A notice whose name is the enclosing test fn, either written out or built
#   from the `{test}` placeholder — the spelling a test that passes its own name
#   uses. That placeholder is accepted by its exact text and no other: `{other}`
#   reads identically here and names a cell that ran.
#
#   A notice naming a DIFFERENT test is refused for the same reason: the runner
#   would list it under a name no libtest filter reaches.
#
# WHAT IS ALSO REFUSED: A GUARD THAT SAYS NOTHING AT ALL
#   The rule above reads notices. The second rule reads their ABSENCE, which is
#   the shape the first one cannot see: a line carrying no SKIP token is not a
#   notice to it, libtest reports the cell as `ok`, and the runner's harvest
#   never finds it — so a cell that stood down is counted as a pass by every
#   gate in the tree, and no number of snapshots changes that.
#
#   The shape is a block opened by a line reading an environment variable and
#   closed by a `return` that CARRIES NO VALUE, with no notice inside it.
#   `return;`, a bare `return`, `return None;` and `return Ok(());` stand a cell
#   down; `return Some(p);` or `return v.parse().unwrap_or(4096);` is a result,
#   and a guard whose every exit carries one has answered rather than skipped —
#   the two-env-key resolvers in this tree are that shape. The needle is not
#   anchored at line start either, or a one-line guard body
#   (`let Ok(p) = std::env::var("X") else { return; };`) reads as a block with
#   no return in it at all.
#
#   `RMLX_SKIP_GPU` is out of it: `scripts/run_gpu_tests.sh` refuses to start
#   with that variable set, so a guard on it cannot stand a cell down in this
#   suite.
#
#   Its population is the DECLARING FILES of the classified GPU tests, not the
#   test fns: such a guard is routinely in a file-local helper, and a rule
#   scoped to test bodies reads a helper's silent return as a clean scan.
#
#   The line that reads the variable and the line that opens the block can be
#   two lines: `let Some(p) =` / `std::env::var_os("X")` / `else {`. The guard
#   starts at the block, whichever line opens it.
#
# WHAT IS ALSO REFUSED: A STAND-DOWN IN ANOTHER FORM
#   A classified GPU test that prints a line and then returns with no value, in
#   one block, has announced a stand-down. When the block holds no `SKIP`
#   notice, the announcement is in a form the runner does not harvest:
#   `<test>: skipping: <why>` reads as prose there, libtest prints `ok`, and
#   the cell is counted as passed. So the notice has one form, and this rule
#   refuses a block that announces in any other. Whatever guards the block —
#   a variable, a missing file, a counter that did not move — is not read.
#
#   Its population is every fn of the declaring files, as for the guard rule:
#   a helper that serves several tests stands each of them down. A return with
#   no value is the guard rule's: `return;`, a bare `return`, `return None;`
#   and `return Ok(());`. A print is `println!`, `eprintln!`, `print!` or
#   `eprint!`.
#
#   WHAT THIS RULE CANNOT READ. A block that returns and prints nothing: a
#   test that returns after a helper announced the stand-down by the test's
#   name is that shape. A line written through `tracing` or another macro. A
#   print with no return after it: a test that falls through to its end, or
#   that takes the `else` body. A print in an outer block with the return in
#   an inner one. A block after a nested `fn` item, because the scan restarts
#   its depth at every `fn`. A block at the top level of a fn body.
#
#   WHAT IS OUTSIDE THE POPULATION. A helper module that classified tests call
#   and that declares no classified test is read by neither this rule nor the
#   guard rule. The three that hold a stand-down today are
#   `crates/rmlx-models/tests/common/mod.rs`,
#   `crates/rmlx-server/tests/thread_boundary/mod.rs` and
#   `crates/rmlx-models/src/test_snapshot.rs`. Review holds their stand-downs
#   to `SKIP {test}:`.
#
# WHAT IS ALSO REFUSED: A WAIVER THAT NAMES NO TEST
#   `WAIVED <test>: <why>` is the notice of a test that ran and did not make one
#   of its checks. In every fn of the declaring files it names the enclosing fn
#   or the caller through the exact `{test}` placeholder; any other waiver is
#   one the runner counts and lists nowhere.
#
#   The notice's SHAPE is not defined here. `scripts/lib/skip_notice_patterns.sh`
#   holds it, and `scripts/run_gpu_tests.sh` reads the same file — a source gate
#   that accepted `SKIP  foo:` while the runner counted it as nameless would pass
#   CI and leave every run INCOMPLETE with a number and no name.
#
#   skip-notices passes when every notice in a classified GPU test names its own test, every
#   environment guard in those tests' files announces the stand-down it takes, and no classified
#   test announces a stand-down in another form. It names each one that does not, with its file,
#   line and reason.
#
#   A notice is matched to its test by the declaring file and the fn, so a package whose name
#   differs from its directory is attributed like any other.
#
# ---- a scan that cannot read is a refusal -------------------------------------------------
#   A directory or file the scan cannot read is not a place with no findings. Every mode refuses,
#   with a line naming it, when a find fails, a file in a population is unreadable, or an awk
#   fails: `ignored` and `list` exit 1 as for any classification they cannot trust, `skip-notices`
#   exits 2.
#
# LOC-exempt: one classifier shared by two rules and the listing; the rule text is the spec
# Usage:  bash scripts/check/gpu_tests.sh <ignored|skip-notices|list> [--files] [--root DIR]
#         --files    list only: add the declaring file as a third column
#         --root     scan DIR instead of the repository
# Exit:   0 clean, or the listing printed; 1 a rule fired, or (ignored, list) the classification
#         cannot be trusted: no member, no test file, a directory or file it could not read, a
#         body or attribute it could not parse, an #[ignore] no gate runs; 2 a usage error, or
#         (skip-notices) the scan cannot be trusted: no classification, no test sources, a
#         directory or file it could not read
# Not a mode of: check_gpu_device_census.sh — that gate reads the rmlx binary's own source for
#                its one Device::Gpu site; this one reads the workspace's test sources for the
#                tests that reach it

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/awk_text.sh
. "$REPO_ROOT/scripts/lib/awk_text.sh"
# shellcheck source=scripts/lib/skip_notice_patterns.sh
. "$REPO_ROOT/scripts/lib/skip_notice_patterns.sh"
# shellcheck source=scripts/lib/workspace_members.sh
. "$REPO_ROOT/scripts/lib/workspace_members.sh"

usage() {
    echo "gpu_tests: $1" >&2
    echo "usage: bash scripts/check/gpu_tests.sh <ignored|skip-notices|list> [--files] [--root DIR]" >&2
    exit 2
}

MODE="${1:-}"
case "$MODE" in
ignored | list) FAIL_RC=1 ;;
skip-notices) FAIL_RC=2 ;;
"") usage "no mode given" ;;
*) usage "unknown mode: $MODE" ;;
esac
shift

FILES=0
ROOT_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
    --files)
        [ "$MODE" = list ] || usage "--files belongs to list"
        FILES=1
        shift
        ;;
    --root)
        [ $# -ge 2 ] || usage "--root needs a directory"
        [ -n "$2" ] || usage "--root is empty"
        [ -z "$ROOT_ARG" ] || usage "--root given twice"
        ROOT_ARG="$2"
        shift 2
        ;;
    *) usage "unknown option: $1" ;;
    esac
done

refuse() {
    echo "ERROR: $1" >&2
    exit "$FAIL_RC"
}

SCAN_ROOT="${ROOT_ARG:-$REPO_ROOT}"
if [ -d "$SCAN_ROOT" ]; then
    SCAN_ROOT="$(cd "$SCAN_ROOT" && pwd)" || refuse "cannot enter the scan root ${SCAN_ROOT}."
fi

SCAN_LIST="$(mktemp "${TMPDIR:-/tmp}/gpu_tests.XXXXXX")" || refuse "cannot make a scratch file."
trap 'rm -f "$SCAN_LIST"' EXIT

# list_into_scan <what> <find args>...: run find into $SCAN_LIST, and refuse when it fails. A
# find that fails has listed part of what it was asked for, and a part reads as the whole.
list_into_scan() {
    local what="$1"
    shift
    find "$@" >"$SCAN_LIST" || refuse "could not list ${what}."
}

# The awk detector, shared across every crate's scan. Reads all of a crate's
# scanned files in one pass (FILENAME distinguishes them) and builds a
# whole-crate fn call-graph, so a cross-file helper is reachable. Emits
# `V  <file>: <fn>` for a violation and `W  <file>: <fn>` for the converse
# warning. Kept in a variable so the per-crate loop invokes one identical
# program over each crate's file list.
read -r -d '' AWK_DETECT <<'AWK' || true
    # An fn item header: visibility, qualifiers, an optional extern ABI, then `fn` and
    # a space. A name starts with `$`, `_` or a character that is neither white space
    # nor ASCII punctuation, which admits a non-ASCII letter.
    BEGIN {
        FN_HEAD = "^[[:space:]]*(pub([[:space:]]*\\([^)]*\\))?[[:space:]]+)?((const|async|unsafe)[[:space:]]+)*(extern[[:space:]]+(\"[^\"]*\"[[:space:]]+)?)?fn[[:space:]]+"
        FN_NAME_START = "([$_]|[^[:space:][:punct:]])"
        FN_MATCHER = "\\$[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:"
        QUALIFIER = "(pub([[:space:]]*\\([^)]*\\))?|const|async|unsafe|extern([[:space:]]+\"[^\"]*\")?)"
        QUALIFIER_LINE = "^[[:space:]]*" QUALIFIER "([[:space:]]+" QUALIFIER ")*[[:space:]]*$"
    }

    # The name at the start of s: cut at the first ASCII character that cannot be in
    # one, so a non-ASCII character stays in the name.
    function fn_name(s,   i, c) {
        for (i = 1; i <= length(s); i++) {
            c = substr(s, i, 1)
            if (c != "$" && c != "_" && index(" \t!\"#%&'()*+,-./:;<=>?@[\\]^`{|}~", c) > 0) {
                return substr(s, 1, i - 1)
            }
        }
        return s
    }

    # The line's significant text: trailing line comment and trailing blanks
    # removed. Used both to compare a closing brace against its marker (`}
    # // end of macro` must still close) and to read the line's LAST
    # significant character, which is what decides whether an item is
    # self-contained.
    #
    # This is the one place that decides what a comment, a string and a char
    # literal are. A `/* ... */` comment is read first, with its nesting, and its
    # body is skipped before any `//`, `"` or `'` in it is looked at: a URL or a
    # quote inside a block comment starts no line comment and opens no string.
    #
    # The `//` scan is string-aware. A `//` inside a string literal — a URL is
    # the everyday case — is not a comment, and cutting there would end the
    # line in mid-string and flip every decision that reads its last
    # character. A raw string (`r"..."`, `br#"..."#`, `cr"..."`, with the hashes
    # matched) is a string too.
    #
    # `q0` and `c0` seed the string state and the block-comment depth, and
    # `bare_q` and `bare_c` report where the scan ended, so a caller walking a
    # MULTI-LINE construct can carry both across the line break. The string state
    # is 0 outside a string, 1 inside a plain string (continued with a trailing
    # `\`, or by a quote left open) and 2 plus the hash count inside a raw string
    # that did not close on its line; the line that closes a raw string is cut at
    # the `"` and the same hashes, and a `\` in it is no escape. Scanning such a
    # line from q=0 would read its payload as code and cut at the first `//` in it.
    # A caller that reads one self-contained line passes neither, and a caller that
    # carries the state hands back what it was given.
    #
    # With `want` set, `bare_code` is the same text with every comment body,
    # string body and char literal body blanked (the quotes stay), one character
    # for each character of the line, for a reader that counts brackets. Only
    # the callers that read it ask for it.
    function blanks(k,   r) {
        r = ""
        while (k-- > 0) r = r " "
        return r
    }
    function bare(s, q0, c0, want,   i, n, q, d, c, nx, rest, w, ascii, st, out, j, h, e) {
        ascii = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        n = length(s)
        q = q0 + 0
        d = c0 + 0
        out = ""
        i = 1
        if (q >= 2) {
            h = ""
            for (j = 2; j < q; j++) h = h "#"
            e = index(s, "\"" h)
            if (e == 0) {
                i = n + 1
            } else {
                i = e + length(h) + 1
                q = 0
            }
            if (want) out = blanks(i - 1 > n ? n : i - 1)
        }
        for (; i <= n; i++) {
            st = i
            c = substr(s, i, 1)
            nx = substr(s, i + 1, 1)
            if (d) {
                if (c == "*" && nx == "/") { d--; i++ }
                else if (c == "/" && nx == "*") { d++; i++ }
                if (want) out = out blanks(i - st + 1)
                continue
            }
            if (!q && c == "/" && nx == "*") { d = 1; i++; if (want) out = out "  "; continue }
            if (c == "\\" && q) { i++; if (want) out = out blanks(i > n ? 1 : 2); continue }
            if (!q && (c == "r" || ((c == "b" || c == "c") && nx == "r")) \
             && (i == 1 || substr(s, i - 1, 1) !~ /[A-Za-z0-9_]/)) {
                # A raw string: the `r` (after a `b` or a `c`), the hashes, a quote, and
                # the same hashes after the closing quote. One that does not close on
                # this line is carried to the next, as a state of its own.
                j = i + (c == "r" ? 1 : 2)
                h = ""
                while (substr(s, j, 1) == "#") { h = h "#"; j++ }
                if (substr(s, j, 1) == "\"") {
                    e = index(substr(s, j + 1), "\"" h)
                    if (e > 0) {
                        i = j + e + length(h)
                        if (want) out = out blanks(i - st + 1)
                        continue
                    }
                    q = 2 + length(h)
                    if (want) out = out blanks(n - st + 1)
                    break
                }
            }
            if (c == "\"") { q = !q; if (want) out = out c; continue }
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
                    if (want) out = out blanks(i - st + 1)
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
                # A non-ASCII literal holds no ASCII character, so the `,`, space
                # or `>` after a non-ASCII lifetime (`<'é, '"'>`) rules the window out.
                w = index(substr(rest, 1, 5), "'")
                if (w >= 3) {
                    for (j = 1; j < w; j++) if (index(ascii, substr(rest, j, 1))) break
                    if (j == w) i += w
                }
                if (want) out = out blanks(i - st + 1)
                continue
            }
            if (!q && c == "/" && nx == "/") {
                s = substr(s, 1, i - 1)
                break
            }
            if (want) out = out (q || index(ascii, c) == 0 ? " " : c)
        }
        bare_q = q
        bare_c = d
        bare_code = out
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
        in_sig = 0
        curid = 0
        close_marker = ""
        body_q = 0
        body_c = 0
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

    # One reader of a fn item's signature, from its header line on, over the line's
    # code (`bare_code`: comments, strings and char literals blanked). It keeps the
    # depths of `(`, `[`, `{` and the generic `<` across the signature's lines, and
    # leaves, for the line it read, the number of `{` it opened (`sig_opened`), its
    # last significant character (`sig_last`) and whether a `)`, `]` or `}` met
    # no opener of its own kind (`sig_stray`). What a comment is, `bare()`
    # decides: a caller carries the depth of an open `/* ... */` from line to line
    # in `sig_bc` and the state of an open string in `sig_q`.
    #
    # A `<` or `>` is a generic bracket only when no `(`, `[` or `{` is open: inside
    # them it is an operator (`[u8; 1 << 4]`, `{ N > 4 }`) and counting it would end or
    # extend the item wrongly. The `>` of `->` and `=>` is never a bracket.
    function sig_reset() {
        sig_par = 0; sig_sq = 0; sig_br = 0; sig_ang = 0; sig_bc = 0; sig_q = 0
    }
    function sig_scan(m,   i, n, c, p) {
        n = length(m)
        sig_opened = 0
        sig_stray = 0
        sig_last = ""
        for (i = 1; i <= n; i++) {
            c = substr(m, i, 1)
            if (c == "<" && sig_br == 0 && sig_par == 0 && sig_sq == 0) sig_ang++
            else if (c == ">" && sig_br == 0 && sig_par == 0 && sig_sq == 0) {
                p = (i > 1) ? substr(m, i - 1, 1) : ""
                if (p != "-" && p != "=" && sig_ang > 0) sig_ang--
            }
            else if (c == "(") sig_par++
            else if (c == ")") { if (sig_par > 0) sig_par--; else sig_stray = 1 }
            else if (c == "[") sig_sq++
            else if (c == "]") { if (sig_sq > 0) sig_sq--; else sig_stray = 1 }
            else if (c == "{") { sig_br++; sig_opened++ }
            else if (c == "}") { if (sig_br > 0) sig_br--; else sig_stray = 1 }
            if (c !~ /[[:space:]]/) sig_last = c
        }
    }
    # What the line just read decides: "body" (a `{` is open: the body starts here),
    # "more" (a `(` or `[` is still open: the signature goes on, whatever the line ends
    # in), "decl" (a `;` with none open: a declaration with no body, which no generic
    # list can contain), "more" again (a generic `<` is still open) and "done" (a `{`
    # opened and closed on this line: the item is complete).
    function sig_verdict() {
        if (sig_br > 0) return "body"
        if (sig_par > 0 || sig_sq > 0) return "more"
        if (sig_last == ";") return "decl"
        if (sig_ang > 0) return "more"
        if (sig_opened > 0 && sig_last == "}") return "done"
        return "more"
    }

    # A signature that closes a bracket it never opened was not a signature: the
    # header line was the middle of something else (a macro matcher), and the
    # braces after it are not this fn's.
    function sig_stray_refuse(   l) {
        l = $0
        sub(/^[[:space:]]+/, "", l)
        printf "U  %s:%d: %s (a signature closes a bracket it never opened — parser lost the item)\n",
            FILENAME, FNR, l
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

    # A string or block comment still open when its file ends. Valid Rust never
    # ends that way, so the reader lost the file at some point and everything
    # after it was read as text: a refusal, not a clean scan.
    function flush_top() {
        if (top_q || top_c) {
            printf "U  %s: end-of-file (a string or block comment is open at the end of the file)\n", top_file
        }
        top_q = 0
        top_c = 0
    }

    # Count `[` and `]` over the code `m` from column `from`, carrying `attr_depth`
    # across lines. Returns the column of the `]` that closes the attribute, or 0.
    function attr_scan(m, from,   i, n, c) {
        n = length(m)
        for (i = from; i <= n; i++) {
            c = substr(m, i, 1)
            if (c == "[") attr_depth++
            else if (c == "]" && --attr_depth == 0) return i
        }
        return 0
    }

    # What follows an attribute's closing `]` on its line. A tail holding an fn
    # (`#[test] fn x() {}`, `impl S { fn h() {} }`), a second attribute, `macro_rules!`
    # or `Device::Gpu` is refused, not skipped; a fn pointer type is `fn(`. Any other
    # tail (`T,`, `pub u8,`, `mod m {`) feeds no rule and ends the pending attribute block.
    function attr_tail_refuse(m, pos,   t, l) {
        t = substr(m, pos + 1)
        sub(/^[[:space:]]+/, "", t)
        if (t == "") return
        if ((t ~ /(^|[^A-Za-z0-9_])fn([[:space:]]|$)/ && t !~ (FN_HEAD FN_MATCHER)) \
            || index(t, "#[") || index(t, "macro_rules!") || index(t, "Device::Gpu")) {
            l = $0
            sub(/^[[:space:]]+/, "", l)
            printf "U  %s:%d: %s (an item after an attribute's closing bracket on its line — parser lost the item)\n",
                FILENAME, FNR, l
            return
        }
        attrs = ""
    }

    # New file: reset the per-file parse state and derive this file's module
    # name (directory name for a mod.rs, otherwise the file stem). The module
    # name is how a QUALIFIED cross-file call `module::helper(..)` binds.
    FNR == 1 {
        flush_fn()
        flush_macro()
        flush_attr()
        flush_top()
        top_file = FILENAME
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
    # Closed by the attribute's own `]`, found by counting brackets over the
    # code of each line and carrying the string and block-comment state across
    # the line break.
    in_attr {
        attrs = attrs " " $0
        bare($0, attr_q, attr_c, 1)
        attr_q = bare_q
        attr_c = bare_c
        attr_end = attr_scan(bare_code, 1)
        if (attr_end > 0) {
            in_attr = 0
            top_q = attr_q
            top_c = attr_c
            attr_tail_refuse(bare_code, attr_end)
        }
        next
    }
    # ── Strings and block comments outside a fn ─────────────────────────
    # `top_q` and `top_c` carry the state as `body_q` and `body_c` do inside a fn.
    # A line that starts inside a string or a comment is its content, which no
    # rule below may read as an item, an attribute or a close. A fn header or an
    # attribute that continues hands the state to the reader that owns its lines.
    !in_fn && !in_attr {
        top_open = (top_q || top_c)
        if (index($0, "\"") || index($0, "/")) {
            bare($0, top_q, top_c)
            top_q = bare_q
            top_c = bare_c
        }
        if (top_open) { next }
    }
    # ── Attribute (any indent, outside a fn) ────────────────────────────
    # Only the attribute block that heads the next item is captured. Inside a fn
    # an attribute line is a line of the signature or the body.
    !in_fn && /^[[:space:]]*#\[/ {
        attrs = attrs " " $0
        # Count `#[test]`s declared inside a macro body, so the END check can
        # compare them against the test items actually extracted from it.
        if (in_macro && $0 ~ /#\[(tokio::)?test[](]/) { macro_test_n++ }
        bare($0, 0, 0, 1)
        attr_q = bare_q
        attr_c = bare_c
        attr_depth = 0
        attr_end = attr_scan(bare_code, index(bare_code, "#["))
        if (attr_end > 0) {
            attr_tail_refuse(bare_code, attr_end)
            next
        }
        in_attr = 1
        attr_file = FILENAME
        attr_label = $0
        sub(/^[[:space:]]+/, "", attr_label)
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
    !(in_fn && (body_q || body_c)) && /^[[:space:]]*\/\// { next }

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
    # A header whose name is a metavariable followed by `:` is a macro matcher
    # (`fn $name:ident where ...`), the pattern a macro invocation is matched
    # against, and never an item.
    !in_fn && $0 ~ (FN_HEAD FN_NAME_START) && $0 !~ (FN_HEAD FN_MATCHER) {
        line = $0
        indent = line
        sub(/[^[:space:]].*$/, "", indent)   # leading whitespace only
        name = line
        sub(FN_HEAD, "", name)
        name = fn_name(name)
        G++
        order[++ord_n] = G
        name_of[G] = name
        file_of[G] = FILENAME
        mod_of[G] = curmod
        attrs_of[G] = attrs
        # A `$metavar` fn name is macro-generated by construction: its real
        # names exist only after expansion. Recorded even when the enclosing
        # `macro_rules!` line was not matched (a non-brace macro delimiter), so
        # such an item can never leak into `list` as an unusable filter.
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
        # The signature is read from this line on (sig_scan). It is a body when a `{`
        # is still open at the end of a line, which is latched until the `}` at the
        # fn's indent. While a `(`, `[` or generic `<` is open it continues onto
        # later lines, whatever the line ends in. With none open, the item is
        # complete when a `{` opened and closed on a line (an inline body) or a `;`
        # ends a declaration (a trait method, an extern block); anything else is a
        # signature that continues.
        #
        # The counts run over the line's CODE, not the raw line: `{` / `}` inside
        # a string, a char literal or a comment are not block delimiters, and
        # counting them raw breaks the decision in BOTH directions. A false
        # "balanced" drops the fn's body, so a `Device::Gpu` inside it never
        # reaches body_of and the test is silently unclassified; a false "open"
        # latches and swallows every later item in the file.
        #
        # A string or a `/* … */` comment that stays open at the end of a line is
        # carried to the next line of the signature, and ends with the signature.
        body_of[G] = bare(line, 0, 0, 1)
        sig_reset()
        sig_bc = bare_c
        sig_q = bare_q
        sig_scan(bare_code)
        sig_state = sig_verdict()
        if (sig_stray && sig_state != "decl") sig_stray_refuse()
        if (sig_state == "done" || sig_state == "decl") {
            in_fn = 0
            curid = 0
            close_marker = ""
        } else {
            in_fn = 1
            top_q = 0
            top_c = 0
            close_marker = indent "}"
            in_sig = (sig_state == "more")
            if (sig_state == "body") {
                body_q = sig_q
                body_c = sig_bc
            }
        }
        attrs = ""
        next
    }
    # ── A `where` clause can put a declaration's `;` on a later line ────
    # The line that opens the body is the last the signature reads: the body takes
    # it, and the state it ends in.
    in_fn && in_sig {
        line = bare($0, sig_q, sig_bc, 1)
        sig_bc = bare_c
        sig_q = bare_q
        sig_scan(bare_code)
        sig_state = sig_verdict()
        if (sig_stray && sig_state != "decl") sig_stray_refuse()
        if (sig_state == "body") {
            in_sig = 0
            body_q = sig_q
            body_c = sig_bc
            body_of[curid] = body_of[curid] " " line
            next
        }
        if (sig_state == "done" || sig_state == "decl") {
            if (sig_state == "done") {
                body_of[curid] = body_of[curid] " " line
            }
            in_fn = 0
            in_sig = 0
            curid = 0
            close_marker = ""
            top_q = sig_q
            top_c = sig_bc
            attrs = ""
            next
        }
    }
    # ── Close of the captured fn: `}` at the fn keyword indent ──────────
    # A `}` that starts a line inside a string or a block comment of the body closes
    # nothing, so the test waits for both states to be closed. The cut line is a
    # prefix of the line, so only a line that starts with the marker is worth cutting;
    # every other body line skips the scan.
    in_fn && !body_q && !body_c && substr($0, 1, length(close_marker)) == close_marker && bare($0) == close_marker {
        in_fn = 0
        curid = 0
        attrs = ""
        next
    }
    # Body lines, with line comments stripped: prose that names a GPU
    # helper or Device::Gpu must not make the fn look GPU-touching. The string and
    # comment state of the body is carried from line to line (`body_q`, `body_c`),
    # starting from the line that opened the body, so a `//` or a `}` inside a string
    # or a comment that runs over several lines is not read as code. A line with no
    # quote and no slash cannot change the state, and skips the call. The rules
    # above that would take such a line for an attribute or a comment are gated on
    # the same state.
    in_fn {
        if (index($0, "\"") || index($0, "/")) {
            line = bare($0, body_q, body_c)
            body_q = bare_q
            body_c = bare_c
        } else {
            line = $0
        }
        body_of[curid] = body_of[curid] " " line
        next
    }
    # Qualifiers alone on a line while an attribute block is pending: the fn that
    # follows on the next line would be read without its attributes.
    attrs != "" && $0 ~ /^[[:space:]]*(pub|const|async|unsafe|extern)/ && bare($0) ~ QUALIFIER_LINE {
        l = $0
        sub(/^[[:space:]]+/, "", l)
        printf "U  %s:%d: %s (a qualifier alone on a line after an attribute — parser lost the item)\n",
            FILENAME, FNR, l
    }
    # Any other line drops a dangling attribute block.
    /^[[:space:]]*[^[:space:]]/ { attrs = "" }

    END {
        # An item left open by the last file still gets its check — otherwise
        # an unterminated fn, macro body or attribute is a free pass.
        flush_fn()
        flush_macro()
        flush_attr()
        flush_top()

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
            # keeping it would hold the test out of `list` forever. A marker
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
            # The same place with no words on it: a reasonless `#[ignore]` on
            # an environment-gated test no device is reachable from, gated
            # directly or through the snapshot resolver. The rule above reads
            # the reason, so an empty one is invisible to it.
            if (!gpu[g] && !declared && attrs_of[g] ~ /#\[ignore\]/ \
             && body_of[g] ~ /(env::var(_os)?|test_snapshot::snapshot)[[:space:]]*\(/) {
                printf "R  %s: %s\n", file_of[g], label_of[g]
            }
        }
    }
AWK

# classify_root: classify the GPU tests under SCAN_ROOT into the globals the modes read, and exit
# FAIL_RC when the classification cannot be trusted.
classify_root() {
    if [ ! -d "${SCAN_ROOT}" ]; then
        echo "ERROR: --root '${ROOT_ARG}' is not a directory." >&2
        exit "$FAIL_RC"
    fi
    CARGO_TOML="${SCAN_ROOT}/Cargo.toml"

    # Fail closed: the workspace manifest is the source of the member list. If it
    # is gone the gate is scanning nothing.
    if [ ! -f "${CARGO_TOML}" ]; then
        echo "ERROR: ${CARGO_TOML} not found — cannot resolve workspace members." >&2
        exit "$FAIL_RC"
    fi
    [ -r "${CARGO_TOML}" ] || refuse "cannot read ${CARGO_TOML}."

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
        members+=("$m")
    done < <(workspace_members "${CARGO_TOML}")

    if [ ${#members[@]} -eq 0 ]; then
        echo "ERROR: parsed 0 workspace members from ${CARGO_TOML}." >&2
        echo "A gate that scans nothing passes everything; refusing to report OK." >&2
        exit "$FAIL_RC"
    fi

    # Plausibility floor: every crate under crates/ (a dir with its own Cargo.toml)
    # must be a parsed member. If the parse yields fewer, the members array layout
    # changed and the scan silently narrowed — the exact scope-loss class this gate
    # guards against. Fail closed. (A crate deliberately excluded from the workspace
    # would trip this; that is a reviewable event, not a silent narrowing.)
    disk_crates=0
    if [ -d "${SCAN_ROOT}/crates" ]; then
        list_into_scan "the member manifests under ${SCAN_ROOT}/crates" "${SCAN_ROOT}/crates" \
            -mindepth 2 -maxdepth 2 -name Cargo.toml -not -path "*/target/*" -print
        disk_crates=$(wc -l <"${SCAN_LIST}" | tr -d ' ') ||
            refuse "could not count the member manifests listed in ${SCAN_LIST}."
    fi
    if [ "${#members[@]}" -lt "${disk_crates}" ]; then
        echo "ERROR: parsed ${#members[@]} workspace members but ${disk_crates} crate dirs exist under crates/." >&2
        echo "The members array in ${CARGO_TOML} likely changed layout and narrowed the scan." >&2
        echo "Restore the one-member-per-line layout, or reconcile the members list." >&2
        exit "$FAIL_RC"
    fi

    # Fail closed per member. A member listed in Cargo.toml whose dir or `src/` is
    # missing means the crate moved or was renamed — break loudly rather than let
    # the gate report OK over a narrowed scan (this workspace has extracted crates
    # before; see the dep graph in CLAUDE.md).
    total_files=0
    violations=""
    warnings=""
    unreasoned=""
    gpu_tests=""
    unreadable=""
    macro_generated=""
    declared_unscanned=""

    for crate in "${members[@]}"; do
        crate_dir="${SCAN_ROOT}/${crate}"
        [ "${crate}" != . ] || crate_dir="${SCAN_ROOT}"
        src_dir="${crate_dir}/src"
        tests_dir="${crate_dir}/tests"

        if [ ! -d "${crate_dir}" ]; then
            echo "ERROR: member crate '${crate}' from Cargo.toml has no directory at ${crate_dir}." >&2
            echo "The crate moved or was renamed; fix the members list or the path." >&2
            exit "$FAIL_RC"
        fi
        if [ ! -d "${src_dir}" ]; then
            echo "ERROR: ${src_dir} does not exist — this gate is scanning nothing for '${crate}'." >&2
            echo "The crate moved or was renamed; update Cargo.toml members." >&2
            exit "$FAIL_RC"
        fi

        # The cargo package name, for `list` consumers that feed it to
        # `cargo test -p`. Read from the manifest rather than assumed equal to the
        # directory basename: the two happen to match for every member today, but a
        # mismatch would surface as an opaque "package not found" from cargo instead
        # of failing at this gate, which is the fail-closed discipline the rest of
        # this script follows.
        pkg_name="$(awk -F'"' '/^name[[:space:]]*=/{print $2; exit}' "${crate_dir}/Cargo.toml" 2>/dev/null)" || pkg_name=""
        if [ -z "${pkg_name}" ]; then
            refuse "could not read [package] name from ${crate_dir}/Cargo.toml."
        fi

        # Gather this crate's scanned files. Whole-crate reachability needs every
        # scanned file of the crate in ONE awk pass, so collect them per crate.
        crate_files=()
        # Unit tests: match both `<name>_tests.rs` and bare `tests.rs`.
        list_into_scan "the test files under ${src_dir}" "${src_dir}" \
            \( -name "*_tests.rs" -o -name "tests.rs" \) -not -path "*/target/*" -print0
        while IFS= read -r -d '' f; do crate_files+=("$f"); done <"${SCAN_LIST}"
        # Integration binaries + their shared modules (optional per crate).
        if [ -d "${tests_dir}" ]; then
            list_into_scan "the test files under ${tests_dir}" "${tests_dir}" \
                -name "*.rs" -not -path "*/target/*" -print0
            while IFS= read -r -d '' f; do crate_files+=("$f"); done <"${SCAN_LIST}"
        fi

        [ ${#crate_files[@]} -eq 0 ] && continue
        total_files=$((total_files + ${#crate_files[@]}))

        for f in "${crate_files[@]}"; do
            [ -r "$f" ] || refuse "cannot read ${f}."
        done
        out=$(awk "$AWK_DETECT" "${crate_files[@]}") || refuse "the detector failed over the test files of ${crate_dir}."
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            case "$line" in
                "V  "*) violations="${violations}  ${line#V  }"$'\n' ;;
                "W  "*) warnings="${warnings}  ${line#W  }"$'\n' ;;
                "R  "*) unreasoned="${unreasoned}  ${line#R  }"$'\n' ;;
                "U  "*) unreadable="${unreadable}  ${line#U  }"$'\n' ;;
                "S  "*) macro_generated="${macro_generated}  ${line#S  }"$'\n' ;;
                "N  "*) declared_unscanned="${declared_unscanned}  ${line#N  }"$'\n' ;;
                "T  "*) t_rec="${line#T  }"
                        gpu_tests="${gpu_tests}${pkg_name}"$'\t'"${t_rec##*: }"$'\t'"${t_rec%%: *}"$'\n' ;;
            esac
        done <<< "$out"
    done

    if [ "${total_files}" -eq 0 ]; then
        echo "ERROR: matched 0 test files across ${#members[@]} workspace members." >&2
        echo "A gate that scans nothing passes everything; refusing to report OK." >&2
        exit "$FAIL_RC"
    fi

    # An unreadable test-generating macro fails BOTH modes. The classification is
    # incomplete either way: enforcing over a test it could not read reports OK on
    # an unchecked test, and listing over it hands the runner a population that is
    # missing one. This is the same fail-closed rule as "scanned 0 files".
    if [ -n "$unreadable" ]; then
        echo "ERROR: this gate could not classify part of a scanned file" >&2
        echo "       (an unreadable macro_rules! body, an item or attribute that never" >&2
        echo "        closed, an item after an attribute's closing bracket on its line, a" >&2
        echo "        qualifier alone on a line after an attribute, a signature that" >&2
        echo "        closes a bracket it never opened, or a string or comment open at" >&2
        echo "        the end of the file, ending" >&2
        echo "        classification mid-file):" >&2
        printf '%s' "$unreadable" >&2
        echo >&2
        echo "The gate classifies a test-generating macro from its body: it needs an" >&2
        echo "attribute block followed by a line-leading \`fn \$<metavar>(\` (or a literal" >&2
        echo "fn name). A name assembled by paste!/concat_idents!, a #[test] sharing" >&2
        echo "its fn's line, and a whole macro_rules! on one line are all invisible —" >&2
        echo "and an invisible GPU test is exactly what this gate exists to prevent." >&2
        echo >&2
        echo "A 'fn never closed', 'attribute never closed' or 'open at the end of the" >&2
        echo "file' finding means the parser" >&2
        echo "lost the file at that point, so everything after it went unclassified. All" >&2
        echo "are reported rather than skipped, because a shape this gate cannot read" >&2
        echo "looks exactly like a compliant one." >&2
        echo >&2
        echo "Write the generated fn as \`fn \$name()\` on its own line with its" >&2
        echo "attributes above it, put nothing after an attribute's closing \`]\` on its" >&2
        echo "line, or extend the classifier to read the new shape." >&2
        exit "$FAIL_RC"
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
        echo "See docs/GPU_TESTS.md." >&2
        exit "$FAIL_RC"
    fi

    # The reasonless twin of the rule above, fatal in both modes for the same
    # reason: a bare `#[ignore]` on a test that reads an environment variable and
    # reaches no `Device::Gpu` is skipped by `make test` and never listed for
    # `make gpu-test`, and with no reason text the rule above cannot see it.
    if [ -n "$unreasoned" ]; then
        echo "ERROR: #[ignore] with no reason on an environment-gated test that reaches" >&2
        echo "       no Device::Gpu:" >&2
        printf '%s' "$unreasoned" >&2
        echo >&2
        echo "Each of these runs under NO gate: \`make test\` skips it (it is ignored)" >&2
        echo "and \`make gpu-test\` skips it (it is not classified). Pick the true one:" >&2
        echo >&2
        echo "  * it drives Metal — pass Device::Gpu, so the GPU suite lists and runs it." >&2
        echo "  * it stands down on its own when the variable is unset — drop the" >&2
        echo "    #[ignore], so it runs in the default gate." >&2
        echo "  * it is ignored for a reason — write that reason in the #[ignore]." >&2
        echo >&2
        echo "See docs/GPU_TESTS.md." >&2
        exit "$FAIL_RC"
    fi
}

# print_notes: state, on stderr, the two enforced populations that `list` leaves out.
print_notes() {
    # Macro-generated tests are enforced above but excluded from `list`, so state
    # the divergence every run rather than letting it be a property only the source
    # comments record. Printed BEFORE the `list` early exit and on stderr: the one
    # surface where "these are not in this run" matters is `make gpu-test`, which
    # calls `list` and nothing else, so a note that only the enforcing run prints
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
    # the same reason: `make gpu-test` calls only `list`, so an operator reading
    # that run must be told what it does not cover.
    if [ -n "$declared_unscanned" ]; then
        echo "NOTE: declared-route GPU tests — the #[ignore] rule is ENFORCED on them, but" >&2
        echo "they are not listed for scripts/run_gpu_tests.sh (that runner asserts a Metal" >&2
        echo "validation banner per crate, and these are snapshot-gated or drive a child" >&2
        echo "process, so listing them would fail the suite for a missing model):" >&2
        printf '%s' "$declared_unscanned" >&2
        echo "  -> No gate executes these. See docs/GPU_TESTS.md for each one's coverage." >&2
        echo >&2
    fi
}

# list_rows <files|pairs>: the classified tests as `<crate><TAB><fn>[<TAB><file>]` per line.
list_rows() {
    if [ "$1" = files ]; then
        printf '%s' "${gpu_tests}"
    else
        printf '%s' "${gpu_tests}" | cut -f1,2
    fi
}

mode_list() {
    classify_root
    print_notes
    if [ -z "${gpu_tests}" ]; then
        echo "ERROR: classified 0 GPU-touching #[ignore] tests across ${total_files} files." >&2
        echo "The detector found nothing to run; refusing to emit an empty list." >&2
        exit 1
    fi
    if [ "$FILES" -eq 1 ]; then list_rows files; else list_rows pairs; fi
}

mode_ignored() {
    classify_root
    print_notes
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
}

# scan_failed <file>: an awk that fails has read none or part of its file, and what it did not
# read is not a clean file.
scan_failed() {
    refuse "the scan of ${1} failed."
}

# package_of <file>: the package that declares the classified tests of <file>.
package_of() {
    local c f
    while IFS=$'\t' read -r c _ f; do
        if [ "$f" = "$1" ]; then
            printf '%s' "$c"
            return 0
        fi
    done <<<"${classification}"
}

# report_path <file>: the file as a report names it: relative to crates/ for a member
# under it, relative to the root for any other.
report_path() {
    case "$1" in
    "${SCAN_ROOT}"/crates/*) printf '%s' "${1#"${SCAN_ROOT}"/crates/}" ;;
    *) printf '%s' "${1#"${SCAN_ROOT}"/}" ;;
    esac
}

mode_skip_notices() {
    local gpu_files notices records f out
    local violations n silent n_silent other n_other waivers n_waivers red rel

    classify_root
    classification="$(list_rows files)"
    if [ -z "${classification}" ]; then
        echo "ERROR: the classification produced no GPU tests — the scan" >&2
        echo "would pass by having nothing to look at." >&2
        exit 2
    fi
    # The files that declare a classified test, whichever member holds them. A notice
    # in any other file is in a fn the classifier cannot have named.
    gpu_files="$(printf '%s\n' "${classification}" | cut -f3 | sort -u)"

    # One `<file>\t<line>\t<enclosing fn>\t<verdict>\t<text>` record per stand-down
    # notice, where the verdict is what the notice names:
    #
    #   self   — the enclosing fn, written out or as a format placeholder
    #   none   — nothing; `SKIP: <why>`
    #   other  — a different identifier
    #
    # The enclosing fn is the nearest preceding `fn`, which is what a notice inside
    # a closure or a nested block belongs to as well. A notice in a helper resolves
    # to the helper, which the attribution step below then drops.
    notices=""
    while IFS= read -r f; do
        [ -n "${f}" ] || continue
        out="$(awk -v FILE="${f}" \
            -v ANY="${ANY_SKIP}" -v NAMED="${NAMED_SKIP}" -v PLACEHOLDER="${NAMED_SKIP_PLACEHOLDER}" '
            match($0, /^[[:space:]]*(pub[[:space:]]+(\([^)]*\)[[:space:]]*)?)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
                head = substr($0, RSTART, RLENGTH)
                sub(/^.*fn[[:space:]]+/, "", head)
                cur = head
            }
            # A stand-down notice is a quoted literal carrying SKIP as a whole word.
            # Both halves are required: the bare word appears in prose comments and
            # in variable names, and a quote alone is every other string in the file.
            $0 ~ /"/ && $0 ~ ANY {
                verdict = "none"
                if (match($0, PLACEHOLDER)) {
                    verdict = "self"
                } else if (match($0, NAMED)) {
                    named = substr($0, RSTART, RLENGTH)
                    sub(/^SKIP /, "", named)
                    sub(/:$/, "", named)
                    verdict = (named == cur) ? "self" : "other:" named
                }
                text = $0
                sub(/^[[:space:]]+/, "", text)
                printf "%s\t%d\t%s\t%s\t%s\n", FILE, NR, cur, verdict, text
            }
        ' "${f}")" || scan_failed "${f}"
        notices="${notices}${out}"$'\n'
    done <<<"${gpu_files}"

    violations=""
    n=0
    while IFS=$'\t' read -r v_file v_line v_fn v_verdict v_text; do
        [ -n "${v_file:-}" ] || continue
        case "${v_verdict}" in self) continue ;; esac
        # Keep only what the classifier named, by the file that declares it and its fn.
        case $'\n'"${classification}"$'\n' in
            *$'\t'"${v_fn}"$'\t'"${v_file}"$'\n'*) ;;
            *) continue ;;
        esac
        rel="$(report_path "${v_file}")"
        crate="$(package_of "${v_file}")"
        case "${v_verdict}" in
            none) why="names no test" ;;
            other:*) why="names ${v_verdict#other:}, not itself" ;;
            *) why="unclassified notice" ;;
        esac
        violations="${violations}    ${rel}:${v_line} — ${crate} ${v_fn} ${why}"$'\n'
        violations="${violations}        ${v_text}"$'\n'
        n=$((n + 1))
    done <<< "${notices}"

    # The guards that stand a cell down and announce nothing. One record per guard
    # block that returns with no notice in it, `<file>\t<line>\t<fn>\t<text>`.
    #
    # The block is followed by brace depth over the line's CODE — comments removed
    # and string bodies blanked, so a literal carrying a brace does not close a
    # guard early — while the notice and the guarded variable are read from the
    # line as written, because both of those ARE literals.
    silent=""
    n_silent=0
    records=""
    while IFS= read -r f; do
        [ -n "${f}" ] || continue
        out="$(awk -v FILE="${f}" -v ANY="${ANY_SKIP}" "${AWK_TEXT_FNS}"'
            {
                raw = $0
                code = blank_strings(decomment(raw))
                bare = decomment(raw)
            }
            match(code, /^[[:space:]]*(pub[[:space:]]+(\([^)]*\)[[:space:]]*)?)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
                head = substr(code, RSTART, RLENGTH)
                sub(/^.*fn[[:space:]]+/, "", head)
                cur = head
            }
            # The process-wide GPU off switch is not a missing-model guard: the
            # runner refuses to start with it set, so a cell behind it never
            # stands down in that suite.
            !g && code ~ /env::var/ && bare !~ /RMLX_SKIP_GPU/ && index(code, "{") > 0 {
                g = 1; gl = NR; gfn = cur; gtext = raw
                sub(/^[[:space:]]+/, "", gtext)
                depth = 0; notice = 0; ret = 0; pend = 0
            }
            # The variable is read on one line and the block opens on a later one.
            # The statement that ends first, with no block, is not a guard.
            !g && !pend && code ~ /env::var/ && bare !~ /RMLX_SKIP_GPU/ && code !~ /;[[:space:]]*$/ {
                pend = 1; gl = NR; gfn = cur; gtext = raw
                sub(/^[[:space:]]+/, "", gtext)
                next
            }
            pend && index(code, "{") > 0 { g = 1; pend = 0; depth = 0; notice = 0; ret = 0 }
            pend && code ~ /;[[:space:]]*$/ { pend = 0 }
            g {
                opens = gsub(/\{/, "{", code)
                closes = gsub(/\}/, "}", code)
                depth += opens - closes
                if (raw ~ /"/ && raw ~ ANY) notice = 1
                if (code ~ /(^|[^A-Za-z0-9_])return([[:space:]]*(;|$)|[[:space:]]+(None|Ok\(\(\)\))[[:space:]]*;)/) ret = 1
                if (depth <= 0) {
                    if (ret && !notice)
                        printf "%s\t%d\t%s\t%s\n", FILE, gl, gfn, gtext
                    g = 0
                }
            }
        ' "${f}")" || scan_failed "${f}"
        records="${records}${out}"$'\n'
    done <<<"${gpu_files}"
    while IFS=$'\t' read -r s_file s_line s_fn s_text; do
        [ -n "${s_file:-}" ] || continue
        rel="$(report_path "${s_file}")"
        silent="${silent}    ${rel}:${s_line} — ${s_fn} returns from an environment guard with no stand-down notice"$'\n'
        silent="${silent}        ${s_text}"$'\n'
        n_silent=$((n_silent + 1))
    done <<< "${records}"

    # The blocks of a classified test that print, return with no value and hold no
    # stand-down notice. One record per block, `<file>\t<line>\t<fn>\t<text>`, at
    # the line of the `return`.
    #
    # Blocks are followed by brace depth over the line's CODE, as above. A print
    # and a notice belong to the innermost block open on their line.
    other=""
    n_other=0
    records=""
    while IFS= read -r f; do
        [ -n "${f}" ] || continue
        out="$(awk -v FILE="${f}" -v ANY="${ANY_SKIP}" "${AWK_TEXT_FNS}"'
            {
                raw = $0
                code = blank_strings(decomment(raw))
            }
            match(code, /^[[:space:]]*(pub[[:space:]]+(\([^)]*\)[[:space:]]*)?)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
                head = substr(code, RSTART, RLENGTH)
                sub(/^.*fn[[:space:]]+/, "", head)
                cur = head; sp = 0
            }
            {
                n = length(code)
                for (i = 1; i <= n; i++) {
                    ch = substr(code, i, 1)
                    if (ch == "{") { sp++; printed[sp] = ""; noticed[sp] = 0 }
                    else if (ch == "}" && sp > 0) sp--
                }
                if (sp < 2) next
                if (code ~ /(^|[^A-Za-z0-9_])e?print(ln)?!/) {
                    printed[sp] = raw
                    sub(/^[[:space:]]+/, "", printed[sp])
                }
                if (raw ~ /"/ && raw ~ ANY) noticed[sp] = 1
                if (code ~ /(^|[^A-Za-z0-9_])return([[:space:]]*(;|$)|[[:space:]]+(None|Ok\(\(\)\))[[:space:]]*;)/ && printed[sp] != "" && !noticed[sp])
                    printf "%s\t%d\t%s\t%s\n", FILE, NR, cur, printed[sp]
            }
        ' "${f}")" || scan_failed "${f}"
        records="${records}${out}"$'\n'
    done <<<"${gpu_files}"
    while IFS=$'\t' read -r o_file o_line o_fn o_text; do
        [ -n "${o_file:-}" ] || continue
        rel="$(report_path "${o_file}")"
        other="${other}    ${rel}:${o_line} — ${o_fn} prints and returns with no value, and the block holds no stand-down notice"$'\n'
        other="${other}        ${o_text}"$'\n'
        n_other=$((n_other + 1))
    done <<< "${records}"

    # Every waiver, `WAIVED <test>: <why>`, in every fn of the classified tests'
    # files. It names the caller through the exact `{test}` placeholder or names
    # its enclosing fn; anything else is one record `<file>\t<line>\t<fn>\t<why>\t<text>`.
    waivers=""
    n_waivers=0
    records=""
    while IFS= read -r f; do
        [ -n "${f}" ] || continue
        out="$(awk -v FILE="${f}" -v ANY="${ANY_WAIVED}" -v NAMED="${NAMED_WAIVED}" \
            -v PLACEHOLDER="${NAMED_WAIVED_PLACEHOLDER}" '
            match($0, /^[[:space:]]*(pub[[:space:]]+(\([^)]*\)[[:space:]]*)?)?(async[[:space:]]+)?fn[[:space:]]+[A-Za-z_][A-Za-z0-9_]*/) {
                head = substr($0, RSTART, RLENGTH)
                sub(/^.*fn[[:space:]]+/, "", head)
                cur = head
            }
            $0 ~ /"/ && $0 ~ ANY && !match($0, PLACEHOLDER) {
                why = "names no test"
                if (match($0, NAMED)) {
                    named = substr($0, RSTART, RLENGTH)
                    sub(/^WAIVED /, "", named)
                    sub(/:$/, "", named)
                    if (named == cur) next
                    why = "names " named ", not itself"
                }
                text = $0
                sub(/^[[:space:]]+/, "", text)
                printf "%s\t%d\t%s\t%s\t%s\n", FILE, NR, cur, why, text
            }
        ' "${f}")" || scan_failed "${f}"
        records="${records}${out}"$'\n'
    done <<<"${gpu_files}"
    while IFS=$'\t' read -r w_file w_line w_fn w_why w_text; do
        [ -n "${w_file:-}" ] || continue
        rel="$(report_path "${w_file}")"
        waivers="${waivers}    ${rel}:${w_line} — $(package_of "${w_file}") ${w_fn}: the waiver ${w_why}"$'\n'
        waivers="${waivers}        ${w_text}"$'\n'
        n_waivers=$((n_waivers + 1))
    done <<< "${records}"

    # Every kind is printed before the single exit at the end. They co-occur — a
    # suite half-converted has one of each — and a gate that exits inside the first
    # block sends the reader back for the second one run later.
    red=0

    if [ "${n_other}" -gt 0 ]; then
        echo "ERROR: ${n_other} block(s) in classified GPU tests' files announce a stand-down in a form" >&2
        echo "       the runner does not harvest:" >&2
        printf '%s' "${other}" >&2
        echo >&2
        echo "A fn that prints and returns with no value did not run to its end. libtest" >&2
        echo "reports it as \`ok\`, and scripts/run_gpu_tests.sh lists it only when the line" >&2
        echo "it printed is" >&2
        echo "  SKIP <this test fn>: <why>" >&2
        echo "A helper takes the caller's test name and prints \`SKIP {test}: <why>\`." >&2
        echo "See docs/GPU_TESTS.md." >&2
        echo >&2
        red=1
    fi

    if [ "${n_silent}" -gt 0 ]; then
        echo "ERROR: ${n_silent} environment guard(s) in classified GPU tests' files stand a" >&2
        echo "       cell down and announce nothing:" >&2
        printf '%s' "${silent}" >&2
        echo >&2
        echo "libtest reports such a cell as \`ok\` and scripts/run_gpu_tests.sh never sees" >&2
        echo "it, so a cell that could not run is counted as one that passed. Announce it:" >&2
        echo "  SKIP <the test fn>: <why>" >&2
        echo "A helper takes the caller's test name as an argument and prints" >&2
        echo "\`SKIP {test}: <why>\`; it cannot name itself, because no libtest filter" >&2
        echo "reaches a helper. See docs/GPU_TESTS.md." >&2
        red=1
    fi

    if [ "${n}" -gt 0 ]; then
        echo "ERROR: ${n} stand-down notice(s) in classified GPU tests do not name their test:" >&2
        printf '%s' "${violations}" >&2
        echo >&2
        echo "scripts/run_gpu_tests.sh counts such a notice and lists it nowhere, so the run" >&2
        echo "is INCOMPLETE with a number and no name. Write it as" >&2
        echo "  SKIP <this test fn>: <why>" >&2
        echo "See docs/GPU_TESTS.md." >&2
        red=1
    fi

    if [ "${n_waivers}" -gt 0 ]; then
        echo "ERROR: ${n_waivers} waiver(s) in classified GPU tests' files do not name their test:" >&2
        printf '%s' "${waivers}" >&2
        echo >&2
        echo "scripts/run_gpu_tests.sh counts such a waiver and lists it nowhere. Write it as" >&2
        echo "  WAIVED <this test fn>: <why>" >&2
        echo "or, in a helper, \`WAIVED {test}: <why>\`. See docs/GPU_TESTS.md." >&2
        red=1
    fi

    if [ "${red}" = "1" ]; then
        exit 1
    fi

    echo "OK: every stand-down notice in a classified GPU test names its own test,
every environment guard in those files announces the stand-down it takes, and
no classified test announces a stand-down in another form."
}

case "$MODE" in
ignored) mode_ignored ;;
skip-notices) mode_skip_notices ;;
list) mode_list ;;
esac
