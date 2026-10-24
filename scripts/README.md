# scripts/ — rules

`scripts/` holds the shell and Python that the `Makefile`, the hosted CI
(through `make` only) and people run against this tree. `INDEX.md` lists every
file with its caller and purpose; `make scripts-index` generates it from the
files' own headers and `make check-scripts` fails when it differs, so never
edit it by hand. A rule marked [gate] is enforced by a named `make` gate
(`make check-scripts` unless the line names another); [review] is the
reviewer's.

## 1. Add a mode, not a script

A new scenario changes an existing script: a flag, a subcommand, a case.
Several scripts that do one job for their caller and differ in a constant, a
model or codec list, a target, an output sink or a mode are one tool with
arguments (CLAUDE.md simplicity rule 6).

A new file is allowed only when no existing tool owns its job. Its header
carries:

    # Not a mode of: <closest existing tool> — <why it cannot be a mode there>

[gate: on every entry point added since the base, the line exists and gives a
reason (`not-a-mode`); on every file, the file it names must be another entry
point, not the file itself, `INDEX.md`, `README.md` or a directory
(`not-a-mode-stale`). review: the reason is true]

Two kinds of file are not tools and need no such line: a recall test, whose
`Recall test for: <file>` names another file of the tree, and a module, whose
`Module of: <file>` does. A name that ends in `_selftest.sh` proves nothing, and
a `Module of:` or `Recall test for:` that names nothing leaves the file an
entry point under every rule. A file is new when it is absent at the base; move
a file with `git mv`, because a plain `mv` that is not staged reads as a new
file.

Two files of one language that say the same lines are one tool in two files.
The figure of a pair is `matched_lines` (`report/debt_report.py
--matched-lines`): the larger of the two directions of `difflib` over the
significant lines of both files, digit-folded. A significant line is not blank,
not a comment and not a bare closer (`fi`, `done`, `}`). A pair is a twin at 40
matched lines, or at half of the smaller file's significant lines (rounded up)
when that file has at least 20 of them; below 20 a header and a few lines of boilerplate would
make every small pair one. A shell file and a Python file are never compared. A
pinned pair whose figure grows fails `twin-pin`. [gate: `twin`, `twin-pin`]

A shell function defined in a `lib/` file and defined again in a file outside
`lib/` fails (`lib-shadow`): the file stops using the one the others share. A
recall test is not exempt; none redefines one today. [gate: `lib-shadow`]

A committed file is a **repeated** operation with a **named caller**: a make
target, another script, or a person who runs it again. A one-off campaign
harness is scratch under `<RMLX_HOME>/tmp/` and is not committed. [review]

## 2. Where a file goes

| Directory | Meaning |
|---|---|
| `check/` | `make ci` gates that read the tree, and their recall tests |
| `report/` | advisory reports and structural figures (`debt_report.py` and its modules) |
| `bench/` | harnesses and probes that run a model and measure |
| `published/` | the published-protocol pipeline |
| `ingest/` | promote a harness result into §8.5 records |
| `suite/` | drive cargo test suites beyond `cargo test` |
| `profile/` | capture, read and bound performance profiles |
| `host/` | act on this machine's toolchain or disk |
| `release/` | release mechanics |
| `spec/` | speculative-decoding equivalence evidence |
| `gen/` | one-off generators of checked-in assets |
| `lib/` | sourced or imported only |
| `fixtures/<gate>/` | planted trees and case tables read by one recall test |

Nothing else sits at the top level but this file and `INDEX.md`. [gate:
`class`] The files that predate the layout, at the top level and in older
directories, are pinned (section 11) until they move. `fixtures/` is not
scanned. A new directory needs three files that share its meaning and a row
here. [review]

* A recall test lives beside its subject as `<subject>_selftest.sh`. [review]
* A module used by one tool lives beside it, has no shebang and no exec bit,
  and its header says `Module of: <class>/<tool>`. A file with that line is not
  an entry point. The tool must exist, not be the file itself, and reach it: run,
  source or import it, or do so through modules that do. [gate: `module-of`;
  `shebang` and `exec-bit` for the bits]
* `lib/` holds a file only when two or more different files use it: scripts that
  source, import, run or read it, or `crates/` sources that name it in a string
  literal. A recall test counts only for `lib/selftest_support.sh`. A file with
  one user carries `Module of: <that user>` and is not counted. [gate:
  `lib-consumers`]
* `lib/` is sourced or imported, never run: no recipe, script, workflow or
  crate runs a file in it. A crate runs one when its path is the argument of
  `Command::new(`; any other string literal that names it only reads it. [gate:
  `lib-run`]
* A `.sh` or `.py` file outside `lib/` with no shebang that nothing runs,
  sources or imports is an orphan: no other rule sees it. [gate: `orphan`]
* A name does not repeat its directory (`check/eval_lock.sh`, not
  `check/check_eval_lock.sh`). [review]

## 3. Header

Every file starts with this block (Python: the module docstring; a data file
or an `.awk` file: a `#` comment; an `.entitlements` file: its first XML
comment):

    #!/usr/bin/env bash
    # <purpose: one complete line, at most 100 characters>
    #
    # Usage:  <invocation>
    # Exit:   0 <meaning>; 1 <meaning>; 2 <meaning>
    # Set:    -uo — <why a failing step must not stop the script>
    # Caller: by hand              (only when no make target or script runs it)

* The purpose line is the `INDEX.md` row: the line right after the shebang,
  not wrapped onto the next one. [gate: `purpose`]
* `Usage:` on every entry point, and it says something. An entry point is a
  `.sh` or `.py` file that is run: it has a shebang and sits outside `lib/`, or
  a recipe or a script runs it (sourcing and importing are not running),
  wherever it lives; a file that is a module of another file is not one.
  [gate: `usage`]
* `Exit:` names every literal exit code the code uses (`exit N`,
  `sys.exit(N)`, `raise SystemExit(N)`). A number counts only at a code
  position, the start of a line or after `;`, `,`, `|`, `/` or `(`, alone or as
  a range `3-10`; a number in prose does not. A status passed through from
  another program is `passthrough of <program>`. A `return N` from a Python
  `main()` is not read. [gate: `exit-undeclared`]
* `Set:` only on a shell entry point that runs under `set -uo pipefail`
  (section 5), with the reason after the dash. [gate: `set-mode`]
* The caller is derived: a make target whose recipe runs the file, or another
  script that runs, sources or imports it (a mention in a comment or an `echo`
  is not a run). A `VAR=path` assignment counts only where the same file later
  puts `$VAR` in command position, as the command, behind a wrapper such as
  `bash`, `env` or `timeout`, as the first argument of `st_run`, or after the
  `--` of `st_case`. A Python file runs a file through a command vector passed
  to a call, or through a name bound to the path (a string, or `Path(__file__)`
  arithmetic) and passed as the program of a `subprocess` call; a name that is
  only printed is no caller. It imports a module with `import`. A hosted-CI
  `run:` step and a pre-commit `entry:` are callers, shown as `ci.yml` and
  `pre-commit`, and so is a `crates/` source whose string literal is the whole
  path (`concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/x.sh")`), shown
  as the crate path; a path in a comment or inside a longer string is not.
  Write `Caller: by hand` only when nothing runs the file; the gate fails the
  line when something does. [gate: `caller`, entry points]
* A recall test names its subject: `Recall test for: <tool> [<mode> ...]`, the
  tool a path relative to `scripts/`, one line per tool. A mode is the first
  argument of the line that runs the tool when that is a bare flag or word
  (`--library`, `verify`), else `default`; a test that covers several modes lists
  them. The tool must exist and not be the file itself, and each mode token must
  occur in the tool's text. [gate: `recall-stale`]
* Explanation may follow the block. It states why, not what, and names no
  ticket, plan or campaign (CLAUDE.md "Comments and identifiers"). [review]

## 4. Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | the checked thing failed: a violation, a regression, a refused measurement |
| 2 | could not run: usage error, missing input, unreadable tree ("unavailable") |
| 3–10 | specific to one script; each declared in `Exit:` |
| 11 | passthrough: `rmlx` refused the Metal claim (`docs/CLI.md` § "Claim file") |
| 78 | passthrough: `rmlx` refused the configuration (EX_CONFIG) |
| 125 | a measurement precondition failed; `git bisect` skips |

## 5. Shell and Python

* Python for parsing, arithmetic, JSON, SQLite, HTTP and source scanning;
  standard library only, runs on Python 3.9 (the host's `python3`). The gate
  parses every `.py` with `ast.parse(..., feature_version=(3, 9))`, whatever
  interpreter runs it, and fails a `|` union in an annotation Python evaluates
  when the definition runs (an argument or return annotation, a variable
  annotation outside a function) in a file with no `from __future__ import
  annotations`. A library call newer than 3.9 is not seen (section 12). [gate:
  `py-floor`, never pinned]
* Bash for running other programs in order — `cargo`, `make`, `xcrun`,
  `rmlx serve`, `rmlx claim run`, a PID it started — and for every recall test.
  It runs on bash 3.2 (the Makefile's `SHELL`) and on GNU tools in CI: no
  associative arrays, no `mapfile`, guard `"${arr[@]}"` under `set -u`.
  [review]
* No Python heredoc or `python3 -c` body longer than 20 lines in a `.sh`
  file; it becomes a `.py` file. A body is the lines between the opener of a
  heredoc on a command line that holds `python` or `python3` and its terminator,
  or the lines of a `-c` string. The heredoc of another command is skipped whole,
  so Python held in the body of a `cat` heredoc is text. A file with a longer
  body is pinned with the length of each; a longer body, or one more, fails
  `heredoc-pin`. [gate: `heredoc`, `heredoc-pin`]
* Shebang `#!/usr/bin/env bash` or `#!/usr/bin/env python3` matching the
  extension; exec bit on every entry point, never on a module or `lib/` file,
  and no shebang on those. A `lib/` file that is run is judged as a `lib/` file.
  The exec bit is the one git's index records; an untracked file's is the
  file system's. [gate: `shebang`, `exec-bit`]
* `set -euo pipefail` in every shell entry point: the options `e`, `u` and
  `pipefail` set by `set` commands in any spelling (`-Eeuo pipefail`, `-eu -o
  pipefail`). A recall test or matrix driver that must go on after a failing
  case uses `set -uo pipefail` and gives the reason in a `Set: -uo — <reason>`
  header field; the field is stale on a file that runs under the strict mode.
  [gate: `set-mode`; review: the reason is true]

## 6. Use what `lib/` has

| Need | Use |
|---|---|
| run identity for a §8.5 record | `lib/identity.sh` (`rmlx_export_identity`) |
| start, wait for and attribute one `rmlx serve` | `lib/bench_server.sh` |
| host CPU interference | `lib/cpu_snapshot.sh` + `lib/busiest_between.awk` |
| a recall-test case runner | `lib/selftest_support.sh` (`st_init`, `st_case`, `st_run`, `st_check`, `st_verdict`, `st_bounded`, `st_done`) |
| the planted `scripts/` trees of the gate's recall tests | `lib/selftest_support.sh` (`sg_init`, `sg_fresh`, `sg_put`, `sg_gate`, ...) |
| comment and literal blanking in a Rust scanner | `lib/awk_text.sh` |
| a snapshot's identity columns | `lib/snapshot_identity.py` |
| the gated `.metal` directories | `metal_dirs.sh` (to move into `lib/`) |

A script that defines one of these again fails the gate (`lib-shadow`, section
1). A path of the form `target/<profile>/rmlx`, the profile a name or a
variable, written in a file other than `lib/identity.sh` fails it too: a
comment is not read, a recall test (a file with a valid `Recall test for:`) is
exempt because it plants stub binaries, and a file merely named
`*_selftest.sh` is not one. The files that write it today are pinned with their
site count; one more site, or a larger count, fails `binary-literal-pin`. [gate:
`binary-literal`, `binary-literal-pin`]

## 7. Knobs and names

* Knobs are arguments. A file reads no environment variable it did not read
  when the rule landed. A read is `${NAME:-…}`, `${NAME-…}`, `${NAME:=…}`,
  `${NAME:?…}` (and `=`, `?` without the colon) in shell, or `os.environ[...]`,
  `os.environ.get(...)` and `os.getenv(...)` with a literal name in Python, of
  an upper-case name that the file has not assigned on an earlier line (`NAME=`,
  `local NAME`, `for NAME in`), and that is not a system name (`HOME`, `PATH`,
  `TMPDIR`, `PWD`, `USER`, `SHELL`, `TERM`, `LANG`, `LC_ALL`, `TZ`, `CI`,
  `NO_COLOR` and the bash variables such as `BASH_SOURCE`). A file pins the set
  of names it reads; a name outside the set fails, and `pin-new` refuses a name
  added to a pin. `RMLX_BIN` (in `perf-iter/bench_decode_tps.sh`) and
  `RMLX_BINARY` (in `bench/codec_inertness_probe.sh` and `bench_codec_cell.sh`)
  are pinned where they are read today, and `RMLX` is read by no file; the
  `--binary` sweep removes the first two, and no other file can add any of the
  three. [gate: `env-knob`]
* File names use underscores. [gate: `name`] The repository root variable is
  `REPO_ROOT`, derived from the file's own location. A variable is a root
  variable when its value is, in shell, `dirname` of `$0` or `${BASH_SOURCE[0]}`
  followed by one `/..` for each part of the file's path under `scripts/`
  (`check/x.sh` has two), or `git rev-parse --show-toplevel`; in Python, an
  expression of `__file__` that climbs as many `.parent`, `dirname` and
  `.parents[k]` steps as that count plus one and goes no deeper (a `/ "docs"`
  makes it a path below the root, not the root). Any other name for it fails
  `name`; the files that use one today are pinned with the names. [gate: `name`]
* No absolute machine path in a checked-in file; model snapshots resolve
  under `RMLX_O_MODELS_ROOT`, which the Makefile exports.
  [gate: `make check-personal-data`]
* A `scripts/<path>` token in a script, the Makefile, a workflow,
  `.pre-commit-config.yaml`, `CLAUDE.md`, `CONTRIBUTING.md`, the top-level
  `README.md`, this file, a doc under `docs/` or a `crates/` string literal names
  a file or directory that git lists. A recall test and the case runner
  (`lib/selftest_support.sh`) are not read: they plant trees. A token holding
  `*`, `<`, `{` or `$`, one followed by `(`, `[`, `?`, `\`, `+` or `|` (a
  pattern), and a path git ignores (a produced file) are not checked.
  `CHANGELOG.md` is not read. [gate: `dead-path`; `make
  check-doc-source-citations` for `crates/` paths in docs]

## 8. Processes and the Metal claim

A harness stops only the process it started, by its PID, and passes the claim
refusal (exit 11) through. It never stops a process by name and never touches
the claim file. [gate: `make check-claim-bypass`, tree-wide]

## 9. Gates and the Makefile

* Every script CI needs is reached through a make target; the hosted CI calls
  `make` only. Target names are a contract: a merge keeps the target and
  changes its recipe. [review]
* Every mode of a tool that `make ci` runs has a recall test declared with
  `Recall test for:`, run by `make ci`, asserting the exit code and the reason.
  What `make ci` runs is its recipe, its prerequisites and every `$(MAKE)
  <target>` line it or they hold, recursively. A tool that is itself a recall
  test asks for none. A recipe line that ends in `|| true` throws the failure
  away and asks for none when the tool's purpose line starts with `Advisory`;
  the three such lines `make ci` holds run `debt_report.py` (twice) and
  `target_size_report.sh`, both advisory. On any other tool the line is a gate
  someone silenced and is reported. [gate: `recall-missing`]
* A recall test never sources its gate, and a gate never sources the case
  runner. [review]

## 10. Size and lint

* Soft limit 1000 lines; a longer file carries `LOC-exempt: <why>` in its
  header, with a reason. A file over the limit without one is pinned with its
  line count; a pinned file that grew fails `size-pin`. [gate: `size`,
  `size-pin`]
* `make shellcheck-report` prints the `shellcheck` findings of every shell file
  in `scripts/` (`debt_report.py --shellcheck`), or `skipped: shellcheck not on
  PATH`. It runs at the end of `make ci`, gates nothing and exits 0.

## 11. Pins

Every [gate] rule started with the files that broke it pinned in
`report/scripts_gate_pins.txt`, one `<rule> <path> [<members>]` per line. A pin
accepts every violation of that rule in that file and can only shrink. [gate]

* Fixing a file without deleting its line fails (`pin-stale`); so does a pin
  for a file that is gone or a rule that cannot be pinned. A line written twice
  fails (`pin-dup`).
* `class`, `purpose`, `usage`, `exit-undeclared`, `caller`, `recall-missing`,
  `lib-consumers`, `lib-run`, `shebang`, `exec-bit`, `dead-path`, `set-mode`,
  `name`, `size`, `heredoc`, `twin`, `lib-shadow`, `binary-literal` and
  `env-knob` can be pinned. The line `# pinnable: <rule> ...` lists them and
  must equal the gate's set (`pin-header`); `index-stale`, `not-a-mode`,
  `not-a-mode-stale`, `recall-stale`, `module-of`, `orphan`, `py-floor` and
  every `pin-*` and `*-pin` rule never can.
* A rule whose violation is a set pins the set: `exit-undeclared <path>
  <codes>`, `recall-missing <path> <modes>`, `lib-run <path> <sites>` (a site is
  `make:<target>`, a script, `ci.yml`, `pre-commit` or a crate path), `dead-path
  <path> <tokens>` (the token is the part after `scripts/`; a path outside
  `scripts/` is written `../<path>`), `name <path> <items>` (`hyphen` and the
  root variable names), `lib-shadow <path> <functions>` and `env-knob <path>
  <variables>`. A member outside the set fails the rule; a member of the set
  that is fixed or gone fails `pin-stale`.
* A rule whose violation is a measure pins the measure (a pin with no number
  fails `pin-figure`): `size <path> <lines>`,
  `heredoc <path> <n>,<n>,...` (the lengths of the over-limit bodies, longest
  first), `twin <a>+<b> <matched lines>` (the paths in sorted order) and
  `binary-literal <path> <sites>`. The measure must equal its pin. A larger
  one fails the rule's own `-pin` rule (`size-pin`, `heredoc-pin`, `twin-pin`,
  `binary-literal-pin`); a smaller one fails `pin-stale` until the pin is
  lowered.
* A pin the base does not have, for a rule the base's `# pinnable:` line lists,
  fails (`pin-new`); so does a set member the base's pin for that file lacks and
  a measure above the base's. A moved file keeps its pin when the pin line moves
  with it, and a twin pin follows each of its two files. A rule the base does not list is new and may seed pins.
* While the base has no pin list, the run prints `pins: <base> has no pin list;
  growth not checked` and `pin-new` has nothing to compare with. Until the pin
  list lands on `main`, the recall tests hold the per-rule counts as literals,
  so a chunk that shrinks a pin lowers its literal and growth shows in the diff.

## 12. What the gate cannot see

* That a purpose line is true, that a `Not a mode of:` reason is, that a
  `Caller: by hand` file is repeated work, that a `Set:` reason is true, and
  that a recall test asserts the reason as well as the exit code.
* An exit code that is not a literal: `return N` from a Python `main()`, and a
  helper whose code is a positional parameter (`fail() { exit "$1"; }` then
  `fail 9`). No file in the tree uses the second form today; a new one hides
  its exits from `exit-undeclared`.
* A script run through a path the gate cannot read: an interpreter held in a
  variable, `python3 -m`, a Rust path built from `.join("scripts").join(..)`
  pieces or passed through a variable, a shell path assembled from fragments.
  A file only such a path runs reads as having no caller, which fails closed.
* A mode that is not the first argument of its line, a first argument that is a
  bare word and no mode (`bash run.sh all` reads as mode `all`), and a tool
  whose recall test covers a mode by running it without naming it.
* Python newer than 3.9 that parses: a library call (`zip(strict=True)`,
  `int.bit_count`, `dataclass(slots=True)`), and a `|` union in an expression
  that runs (`isinstance(x, int | str)`, a type alias assignment).
* A Python body whose interpreter is held in a variable or called by another
  name than `python` or `python3` (`heredoc`).
* `set` options given any other way than a `set` command in the file: on the
  shebang line, by a sourced file, or by `bash -e` in the caller.
* An environment read written `$NAME` or `${NAME}`, one through `env` or
  `printenv`, an `os.environ` held under another name, and a name that is not a
  literal. A shell string is read as code: `${NAME:-x}` inside a quoted string
  that no shell expands counts as a read, and a name assigned later in the file
  or in a sourced file counts as read.
* A root variable derived through another variable (`REPO_ROOT="$HERE/.."`), a
  `dirname` written without quotes, and a Python root built with `os.path.join`.
* A binary path assembled from pieces, and the same text in two files that
  differs enough to stay under the twin threshold: the rule measures text, not
  whether two files do one job.
* A function defined in `lib/` and redefined outside it by `eval`, or a Python
  function of the same name.
