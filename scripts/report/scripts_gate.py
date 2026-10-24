"""The rules for scripts/ and the generated scripts/INDEX.md.

Module of: report/debt_report.py

Run as `debt_report.py --scripts [--root DIR] [--base REF] [--write-index]`.
The population is what git lists under `<root>/scripts/` (tracked, or untracked
and not ignored) without `fixtures/`, `INDEX.md` and `README.md`. A violation
prints as `<rule>: <path>:<line>: <detail>`, the path relative to `scripts/`.

Each rule is a function over the population that yields violations. A pin is a
`<rule> <path>` line in `report/scripts_gate_pins.txt` and accepts every
violation of that rule in that file. A rule whose violation is a set, listed in
`SET_NOUNS`, pins the set (`exit-undeclared <path> <codes>`); a rule whose
violation is a measure, listed in `FIGURE_RULES`, pins the measure (`size <path>
<lines>`) and fails its own `-pin` rule when the measure grows.
Pins only shrink: a pin whose file now passes is a violation (`pin-stale`), and
a pin the base lacks, for a rule the base's `# pinnable:` line lists, is one too
(`pin-new`). Only the rules in `PINNABLE` can be pinned.

`debt_report.py --shellcheck` is the advisory mode: it prints what `shellcheck`
finds in the shell files of the population, or that it is not on `PATH`, and
exits 0 either way.

Definitions the rules share:

* An entry point is a `.sh` or `.py` file that is run: it starts with a shebang
  outside `lib/`, or a recipe or a script runs it (not sources or imports it).
  A file whose `Module of:` names another file of the tree is not one.
* The run of `make ci` is the closure of the `ci` target: its recipe lines, its
  prerequisites and every `$(MAKE) <target>` line, recursively. A mode of a tool
  is the first option or bare word of its recipe line that is not the value of
  an option before it (`--x=v` is a mode as written), else `default`.
* The header is the leading comment block (Python: the module docstring;
  `.entitlements`: the first XML comment). Its first line, directly after the
  shebang, is the purpose; `Usage:`, `Exit:`, `Caller:`, `Recall test for:`,
  `Module of:`, `Not a mode of:`, `LOC-exempt:` and `Set:` start a field that
  runs to the next blank line or field.
* A caller is a Makefile target whose recipe, or another script that, runs or
  sources the file; a Python import of the module counts, and so does a name
  bound to its path and passed as the program of a `subprocess` call. A hosted-CI
  run step, a pre-commit entry and a `crates/` string literal that is the whole
  path are callers too. A data file has a reader instead, and any non-comment
  mention counts.
* A new file is one absent at the base, a moved file (`git mv`) excepted.

Not checked, by design: that a purpose line is true, that a `Not a mode of:`
reason is, a `return N` from a Python `main()` (only `sys.exit(N)`, `exit(N)`
and `SystemExit(N)` are literal exits), and an exit through a helper that takes
its code as a parameter. `scripts/README.md` section 12 lists the rest.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scripts_gate_callers import Resolver, derive_callers, index_violations, render_index
from scripts_gate_content import CONTENT_RULES
from scripts_gate_external import load_external
from scripts_gate_files import INDEX, Unavailable, Violation, git, git_ok, load_population
from scripts_gate_recall import RECALL_RULES
from scripts_gate_rules import RULE_FUNCTIONS, Context, entry_points
from scripts_gate_structure import STRUCTURE_RULES
from scripts_gate_twin import rule_twin

PINS = "report/scripts_gate_pins.txt"
PINNABLE = (
    "class",
    "purpose",
    "usage",
    "exit-undeclared",
    "caller",
    "recall-missing",
    "silenced",
    "lib-consumers",
    "lib-run",
    "shebang",
    "exec-bit",
    "dead-path",
    "set-mode",
    "name",
    "size",
    "heredoc",
    "twin",
    "lib-shadow",
    "binary-literal",
    "env-knob",
)
SET_NOUNS = {
    "exit-undeclared": "code",
    "recall-missing": "mode",
    "silenced": "mode",
    "lib-run": "site",
    "dead-path": "token",
    "name": "item",
    "lib-shadow": "function",
    "env-knob": "variable",
}


def total_figure(violations: list[Violation]) -> tuple[int, ...]:
    return (sum(v.figure for v in violations),)


def sorted_figures(violations: list[Violation]) -> tuple[int, ...]:
    return tuple(sorted((v.figure for v in violations), reverse=True))


# A rule whose violation is a measure pins the measure: `<rule> <path> <n>[,<n>...]`.
# The rule that fails when the measure grows, and how the violations reduce to it.
FIGURE_RULES: dict[str, tuple[str, Callable[[list[Violation]], tuple[int, ...]]]] = {
    "size": ("size-pin", total_figure),
    "twin": ("twin-pin", total_figure),
    "binary-literal": ("binary-literal-pin", total_figure),
    "heredoc": ("heredoc-pin", sorted_figures),
}
PINS_HEADER = "pinnable:"


def resolve_base(root: Path, base: str | None) -> tuple[str, str]:
    """(name, commit) of the base that decides which files are new."""
    if base is not None:
        sha = git_ok(root, "rev-parse", "--verify", "--quiet", f"{base}^{{commit}}")
        if not sha:
            raise Unavailable(f"--base {base} is not a commit")
        return base, sha.strip()
    if not git_ok(root, "rev-parse", "--verify", "--quiet", "refs/remotes/origin/main"):
        raise Unavailable("no origin/main ref; fetch, or pass --base")
    merge_base = git(root, "merge-base", "HEAD", "origin/main").strip()
    head = git(root, "rev-parse", "HEAD").strip()
    if merge_base == head:
        parent = git_ok(root, "rev-parse", "--verify", "--quiet", "HEAD^")
        return "HEAD^", (parent or head).strip()
    return "origin/main", merge_base


def changes_since(root: Path, sha: str) -> tuple[set[str], dict[str, str]]:
    """(files added since `sha`, new path -> old path of each file moved)."""
    tracked = git(root, "diff", "-M", "--name-status", sha, "--", "scripts")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", "scripts")
    added = {p for p in untracked.split("\0") if p}
    renames: dict[str, str] = {}
    for line in tracked.splitlines():
        parts = line.split("\t")
        if parts[0] == "A":
            added.add(parts[1])
        elif parts[0].startswith("R"):
            renames[parts[2].removeprefix("scripts/")] = parts[1].removeprefix("scripts/")
    return {p.removeprefix("scripts/") for p in added if p.startswith("scripts/")}, renames


@dataclass
class Pin:
    rule: str
    path: str
    codes: frozenset[str]
    line: int
    figure: tuple[int, ...] = ()


@dataclass
class PinList:
    exists: bool = False
    pins: dict[tuple[str, str], Pin] = field(default_factory=dict)
    pinnable: set[str] | None = None
    header_line: int = 1
    problems: list[Violation] = field(default_factory=list)


def parse_pins(text: str) -> PinList:
    """`<rule> <path>` per line; a rule in `SET_NOUNS` adds the members it
    accepts as `<a,b>`; `# pinnable: <rule> ...` names the rules a pin may be
    added for."""
    out = PinList(exists=True)
    for number, raw in enumerate(text.split("\n"), 1):
        line = raw.strip()
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if body.startswith(PINS_HEADER):
                out.pinnable = set(body[len(PINS_HEADER) :].split())
                out.header_line = number
            continue
        if not line:
            continue
        parts = line.split()
        rule, path = parts[0], parts[1] if len(parts) > 1 else ""
        key = (rule, path)
        if key in out.pins:
            detail = f"{rule} {path} is already pinned at line {out.pins[key].line}"
            out.problems.append(Violation("pin-dup", PINS, number, detail))
            continue
        if rule in FIGURE_RULES:
            try:
                figure = tuple(int(n) for n in parts[2].split(","))
            except (IndexError, ValueError):
                detail = f"{rule} {path} needs a figure: `{rule} {path} <n>`"
                out.problems.append(Violation("pin-figure", PINS, number, detail))
                continue
            out.pins[key] = Pin(rule, path, frozenset(), number, figure)
            continue
        codes = frozenset(parts[2].split(",")) if len(parts) > 2 else frozenset()
        out.pins[key] = Pin(rule, path, codes, number)
    return out


def grows(now: tuple[int, ...], pinned: tuple[int, ...]) -> bool:
    """A figure grew when it has more members or a member above its pin's."""
    return len(now) > len(pinned) or any(n > p for n, p in zip(now, pinned))


def show(figure: tuple[int, ...]) -> str:
    return ",".join(str(n) for n in figure)


def figure_problems(key: tuple[str, str], violations: list[Violation], pin: Pin) -> list[Violation]:
    """Held when the figure equals its pin; a larger one fails the rule's `-pin`
    rule, a smaller one asks for the pin to be lowered."""
    pin_rule, reduce = FIGURE_RULES[key[0]]
    now = reduce(violations)
    if grows(now, pin.figure):
        detail = f"{key[0]} figure is {show(now)}, pinned {show(pin.figure)}; fix the file, do not raise the pin"
        return [Violation(pin_rule, violations[0].path, violations[0].line, detail)]
    if now != pin.figure:
        detail = f"{key[0]} {key[1]}: figure is {show(now)}, pinned {show(pin.figure)}; lower the pin"
        return [Violation("pin-stale", PINS, pin.line, detail)]
    return []


def load_pins(root: Path) -> PinList:
    path = root / "scripts" / PINS
    if not path.is_file():
        return PinList()
    pins = parse_pins(path.read_text(encoding="utf-8"))
    if pins.pinnable != set(PINNABLE):
        detail = f"the `# {PINS_HEADER}` line must list exactly: {' '.join(PINNABLE)}"
        pins.problems.append(Violation("pin-header", PINS, pins.header_line, detail))
    return pins


def apply_pins(found: list[Violation], pins: PinList) -> tuple[list[Violation], dict[tuple[str, str], Pin]]:
    """The violations no pin accepts, and the pins that held."""
    active = list(pins.problems)
    held: dict[tuple[str, str], Pin] = {}
    groups: dict[tuple[str, str], list[Violation]] = defaultdict(list)
    for violation in found:
        groups[(violation.rule, violation.pin_path or violation.path)].append(violation)
    for key, violations in groups.items():
        pin = pins.pins.get(key)
        if pin is None or key[0] not in PINNABLE:
            active += violations
            continue
        if key[0] in FIGURE_RULES:
            active += figure_problems(key, violations, pin)
        elif key[0] in SET_NOUNS:
            active += [v for v in violations if v.key not in pin.codes]
            for member in sorted(pin.codes - {v.key for v in violations}):
                detail = f"{key[0]} {key[1]}: {SET_NOUNS[key[0]]} {member} is fixed or gone; delete it from the pin"
                active.append(Violation("pin-stale", PINS, pin.line, detail))
        held[key] = pin
    for key, pin in pins.pins.items():
        if key[0] not in PINNABLE:
            active.append(Violation("pin-stale", PINS, pin.line, f"{key[0]} {key[1]}: {key[0]} cannot be pinned"))
        elif key not in groups:
            detail = f"{key[0]} {key[1]}: no longer violated; delete the pin"
            halves = key[1].split("+")
            if key[0] == "twin" and halves != sorted(halves):
                detail = f"{key[0]} {key[1]}: write the two paths of a pair in sorted order ({'+'.join(sorted(halves))})"
            active.append(Violation("pin-stale", PINS, pin.line, detail))
    return active, held


def base_path(rule: str, path: str, renames: dict[str, str]) -> str:
    """The path a pin had at the base: a moved file's old path; for a twin pair
    `<a>+<b>`, the old path of each half, in sorted order."""
    if rule == "twin":
        return "+".join(sorted(renames.get(half, half) for half in path.split("+")))
    return renames.get(path, path)


def pin_growth(
    root: Path, sha: str, name: str, pins: PinList, renames: dict[str, str]
) -> tuple[list[Violation], str]:
    """Pins the base does not have, for a rule the base already pinned, and the
    note to print when the base has no pin list to compare with."""
    if not pins.exists:
        return [], ""
    text = git_ok(root, "show", f"{sha}:scripts/{PINS}")
    if text is None:
        return [], f"pins: {name} has no pin list; growth not checked"
    before = parse_pins(text)
    out = []
    for (rule, path), pin in pins.pins.items():
        if rule not in (before.pinnable or ()):
            continue
        old = before.pins.get((rule, base_path(rule, path, renames)))
        if old is None:
            out.append(Violation("pin-new", PINS, pin.line, f"{rule} {path} is not pinned at the base"))
            continue
        if rule in FIGURE_RULES and grows(pin.figure, old.figure):
            detail = f"{rule} {path}: figure {show(pin.figure)} is above the base's {show(old.figure)}"
            out.append(Violation("pin-new", PINS, pin.line, detail))
        for member in sorted(pin.codes - old.codes):
            detail = f"{rule} {path}: {SET_NOUNS.get(rule, 'member')} {member} is not pinned at the base"
            out.append(Violation("pin-new", PINS, pin.line, detail))
    return out, ""


def run(root: Path, base: str | None, write_index: bool) -> int:
    try:
        return check(root, base, write_index)
    except Unavailable as exc:
        print(f"check-scripts: unavailable ({exc})", file=sys.stderr)
        return 2


def run_shellcheck(root: Path) -> int:
    """Print the `shellcheck` findings of the shell files of the population.
    Advisory: exits 0 whatever it finds, whether `shellcheck` is absent or the
    tree cannot be read."""
    exe = shutil.which("shellcheck")
    if exe is None:
        print("shellcheck-report: skipped: shellcheck not on PATH")
        return 0
    try:
        require_work_tree(root)
        paths = [f"scripts/{f.rel}" for f in load_population(root) if f.rel.endswith(".sh")]
    except Unavailable as exc:
        print(f"shellcheck-report: unavailable ({exc})")
        return 0
    proc = subprocess.run([exe, "--format=gcc", "--", *paths], cwd=root, capture_output=True, text=True)
    findings = [line for line in proc.stdout.split("\n") if line.strip()]
    for line in findings:
        print(line)
    print(f"shellcheck-report: {len(findings)} finding(s) in {len(paths)} shell file(s)")
    return 0


def require_work_tree(root: Path) -> None:
    if git_ok(root, "rev-parse", "--is-inside-work-tree") is None:
        raise Unavailable(f"{root} is not inside a git work tree")


def check(root: Path, base: str | None, write_index: bool) -> int:
    require_work_tree(root)
    files = load_population(root)
    name, sha = resolve_base(root, base)
    makefile = root / "Makefile"
    makefile_text = makefile.read_text(encoding="utf-8") if makefile.is_file() else ""
    external = load_external(root)
    resolver = Resolver(files)
    callers = derive_callers(files, resolver, makefile_text, external)
    expected = render_index(files, callers)
    if write_index:
        (root / "scripts" / INDEX).write_text(expected, encoding="utf-8")
        print(f"scripts-index: wrote scripts/{INDEX} ({len(files)} rows)")
    print(f"check-scripts: base {name} ({sha[:12]})")
    added, renames = changes_since(root, sha)
    ctx = Context(root, files, callers, added, entry_points(files, callers), makefile_text, external, resolver)
    found: list[Violation] = []
    for _name, rule in (*RULE_FUNCTIONS, *RECALL_RULES, *STRUCTURE_RULES, *CONTENT_RULES, ("twin", rule_twin)):
        found += sorted(rule(ctx), key=lambda v: (v.path, v.line))
    pins = load_pins(root)
    active, held = apply_pins(found, pins)
    growth, note = pin_growth(root, sha, name, pins, renames)
    active += growth
    if not write_index:
        active += index_violations(root, expected)
    if note:
        print(note)
    for violation in active:
        print(violation.text())
    per_rule: dict[str, int] = defaultdict(int)
    for rule, _path in held:
        per_rule[rule] += 1
    print("pinned: " + (", ".join(f"{r} {per_rule[r]}" for r in PINNABLE if per_rule[r]) or "none"))
    if active:
        print(f"check-scripts: FAIL: {len(active)} violation(s)")
        return 1
    print(f"check-scripts: ok ({len(files)} files, {len(held)} pinned)")
    return 0
