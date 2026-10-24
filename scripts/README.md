# scripts/ — rules

`scripts/` holds the shell and Python that the `Makefile`, the hosted CI
(through `make` only) and people run against this tree. `INDEX.md` lists every
file with its caller and purpose; `make scripts-index` generates it from the
files' own headers and `make check-scripts` fails when it differs, so never
edit it by hand. A rule marked [gate] is enforced by `make check-scripts`;
[gate, next] is meant to be and is not yet; [review] is the reviewer's.

## 1. Add a mode, not a script

A new scenario changes an existing script: a flag, a subcommand, a case.
Several scripts that do one job for their caller and differ in a constant, a
model or codec list, a target, an output sink or a mode are one tool with
arguments (CLAUDE.md simplicity rule 6).

A new file is allowed only when no existing tool owns its job. Its header
carries:

    # Not a mode of: <closest existing tool> — <why it cannot be a mode there>

[gate: on every entry point added since the base that is not a recall test or a
module, the line exists, names an existing file and gives a reason; on every
file, a named file that no longer exists fails (`not-a-mode`,
`not-a-mode-stale`). review: the reason is true]

A file whose body matches another script past a twin threshold, and a `lib/`
function defined again outside `lib/`, fail too. [gate, next]

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

* A recall test lives beside its subject as `<subject>_selftest.sh`.
  [gate, next]
* A module used by one tool lives beside it, has no shebang and no exec bit,
  and its header says `Module of: <class>/<tool>`. A file with that line is not
  an entry point. [gate, next: the tool exists and imports or runs it]
* `lib/` holds a file only when two or more files use it, or one script and a
  Rust test; a helper of one `lib/` file sits beside it with
  `Module of: lib/<file>`. No make recipe runs a `lib/` file. [gate, next]
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
    # Caller: by hand              (only when no make target or script runs it)

* The purpose line is the `INDEX.md` row: the line right after the shebang,
  not wrapped onto the next one. [gate: `purpose`]
* `Usage:` on every entry point. An entry point is a `.sh` or `.py` file with a
  shebang, no `Module of:` line, outside `lib/`. [gate: `usage`]
* `Exit:` names every literal exit code the code uses (`exit N`,
  `sys.exit(N)`, `raise SystemExit(N)`); a range `3-10` counts. A status passed
  through from another program is `passthrough of <program>`. A `return N`
  from a Python `main()` is not read. [gate: `exit-undeclared`]
* The caller is derived: a make target whose recipe runs the file, or another
  script that runs, sources or imports it (a mention in a comment or an `echo`
  is not a run). Write `Caller: by hand` only when nothing runs the file; the
  gate fails the line when something does. [gate: `caller`, entry points]
* A recall test names its subject: `Recall test for: <class>/<tool> [<mode>]`.
  [gate, next]
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
  standard library only, runs on Python 3.9 (the host's `python3`).
  [gate, next: compiles; no construct newer than 3.9]
* Bash for running other programs in order — `cargo`, `make`, `xcrun`,
  `rmlx serve`, `rmlx claim run`, a PID it started — and for every recall test.
  It runs on bash 3.2 (the Makefile's `SHELL`) and on GNU tools in CI: no
  associative arrays, no `mapfile`, guard `"${arr[@]}"` under `set -u`.
  [review]
* No Python heredoc or `python3 -c` body longer than 20 lines in a `.sh`
  file; it becomes a `.py` file. [gate, next]
* Shebang `#!/usr/bin/env bash` or `#!/usr/bin/env python3` matching the
  extension; exec bit on every entry point, never on a module or `lib/` file.
  [gate, next]
* `set -euo pipefail`. A recall test or matrix driver that must go on after a
  failing case uses `set -uo pipefail` and says so in its header. [gate, next:
  mode; review: the reason]

## 6. Use what `lib/` has

| Need | Use |
|---|---|
| run identity for a §8.5 record | `lib/identity.sh` (`rmlx_export_identity`) |
| start, wait for and attribute one `rmlx serve` | `lib/bench_server.sh` |
| host CPU interference | `lib/cpu_snapshot.sh` + `lib/busiest_between.awk` |
| a recall-test case runner | `lib/selftest_support.sh` (`st_init`, `st_case`, `st_run`, `st_check`, `st_verdict`, `st_done`) |
| comment and literal blanking in a Rust scanner | `lib/awk_text.sh` |
| a snapshot's identity columns | `lib/snapshot_identity.py` |
| the gated `.metal` directories | `metal_dirs.sh` (to move into `lib/`) |

A script that defines one of these again fails the gate (`lib-shadow`), and
`target/<profile>/rmlx` written outside `lib/identity.sh` fails it too
(`binary-literal`; recall tests that plant a stub binary are exempt).
[gate, next]

## 7. Knobs and names

* Knobs are arguments. No script reads a new environment variable; the ones
  read today are pinned and the pin only shrinks. [gate, next]
* File names use underscores; the repository root variable is `REPO_ROOT`,
  derived from the file's own location. [gate, next, pinned]
* No absolute machine path in a checked-in file; model snapshots resolve
  under `RMLX_O_MODELS_ROOT`, which the Makefile exports.
  [gate: `make check-personal-data`]
* A `scripts/<path>` written in a script, the Makefile, a crate's
  `concat!`/`join` path or a doc names a file that exists. [gate, next here;
  `make check-doc-source-citations` for `crates/` paths in docs]

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
  [gate, next]
* A recall test never sources its gate, and a gate never sources the case
  runner. [review]

## 10. Size

* Soft limit 1000 lines; a longer file carries `LOC-exempt: <why>` in its
  header. [gate, next]

## 11. Pins

Every [gate] rule started with the files that broke it pinned in
`report/scripts_gate_pins.txt`, one `<rule> <path>` per line. A pin accepts
every violation of that rule in that file and can only shrink: fixing a file
without deleting its line fails (`pin-stale`), and a pin for a file that is
gone or a rule that does not exist fails the same way. A new file is never
pinned. [review]
