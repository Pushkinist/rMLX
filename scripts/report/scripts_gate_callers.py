"""The derived caller of each file and the INDEX.md built from it.

Module of: report/debt_report.py

A caller is a Makefile target or another script that runs, sources or imports
a file. The index lists each file with its derived caller and its purpose.
"""

from __future__ import annotations

import itertools
import os
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from scripts_gate_external import PRE_COMMIT_LABEL, External
from scripts_gate_files import INDEX, Script, Violation
from scripts_gate_text import Refs, clean_shell, make_recipes, make_refs, python_refs, shell_refs, starts_a_token

CALLERS_SHOWN = 3
MAKE_SITE = "make:"


@dataclass
class Callers:
    """Who uses a file. `run_by` holds the sites that run it, as `make:<target>`,
    a script's path or the label of a non-script caller."""

    targets: set[str] = field(default_factory=set)
    scripts: set[str] = field(default_factory=set)
    others: set[str] = field(default_factory=set)
    run_by: set[str] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.targets or self.scripts or self.others)

    @property
    def runs(self) -> bool:
        return bool(self.run_by)


EXPANSION_RE = re.compile(r"\$(?:\{[^}]*\}|\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*)")
SCRIPTS_DIR_RE = re.compile(r"(?:^|/)scripts/(.*)$")


class Resolver:
    """The file of the tree a word names. Only the part after the word's last
    expansion is read. A part that goes through a `scripts/` directory must name
    the file's path under it, and that directory must be the repository's (the
    test `dead-path` applies: a word start or a repository-root expression in
    front), else the word names a file of another tree and no file here; any other
    part names the file by its base name when that is unique, else by its path."""

    def __init__(self, files: list[Script]) -> None:
        self.by_base: dict[str, list[Script]] = defaultdict(list)
        for f in files:
            self.by_base[os.path.basename(f.rel)].append(f)

    def find(self, word: str, code: bool) -> list[Script]:
        if not word or any(ch.isspace() for ch in word):
            return []
        literal = EXPANSION_RE.split(word)[-1]
        candidates = self.by_base.get(os.path.basename(literal), [])
        below_scripts = SCRIPTS_DIR_RE.search(literal)
        if below_scripts:
            start = len(word) - len(literal) + below_scripts.start()
            if word[start] == "/":
                start += 1
            if not starts_a_token(word, start):
                return []
        found = []
        for f in candidates:
            if f.is_code != code:
                continue
            if below_scripts:
                named = below_scripts.group(1) == f.rel
            else:
                named = len(candidates) == 1 or word == f.rel or word.endswith("/" + f.rel)
            if named:
                found.append(f)
        return found


def derive_callers(
    files: list[Script], resolver: Resolver, makefile: str, external: External | None = None
) -> dict[str, Callers]:
    callers = {f.rel: Callers() for f in files}

    def credit(refs: Refs, who: str, kind: str) -> None:
        for words, want_code, runs in (
            (refs.run, True, True),
            (refs.source, True, False),
            (refs.anywhere, False, False),
        ):
            for word in words:
                for f in resolver.find(word, want_code):
                    if f.rel != who:
                        mark(callers[f.rel], who, kind, runs)

    for f in files:
        if f.rel.endswith(".sh"):
            credit(shell_refs(clean_shell(f.text, False, True)), f.rel, "script")
        elif f.rel.endswith(".py"):
            credit(python_refs(f.text), f.rel, "script")
    for target, recipe in make_recipes(makefile).items():
        credit(make_refs(recipe), target, "target")
    if external is not None:
        credit_external(external, callers, credit)
    return callers


def mark(callers: Callers, who: str, kind: str, runs: bool) -> None:
    if kind == "target":
        callers.targets.add(who)
    elif kind == "script":
        callers.scripts.add(who)
    else:
        callers.others.add(who)
    if runs:
        callers.run_by.add(MAKE_SITE + who if kind == "target" else who)


def credit_external(external: External, callers: dict[str, Callers], credit: Callable[[Refs, str, str], None]) -> None:
    """Workflow run steps and pre-commit entries are shell; a crate credits the
    scripts/ path a string literal of its sources names."""
    for label, lines in external.workflows.items():
        credit(shell_refs(clean_shell("\n".join(lines), False, True)), label, "other")
    credit(shell_refs(clean_shell("\n".join(external.pre_commit), False, True)), PRE_COMMIT_LABEL, "other")
    for path, literals in external.crates.items():
        for literal in literals:
            if literal.rel in callers:
                mark(callers[literal.rel], path, "other", literal.runs)


def caller_cell(f: Script, callers: Callers) -> str:
    parts = [f"make {t}" for t in sorted(callers.targets)]
    parts += sorted(callers.others)
    scripts = sorted(callers.scripts)
    parts += scripts[:CALLERS_SHOWN]
    if len(scripts) > CALLERS_SHOWN:
        parts.append(f"+{len(scripts) - CALLERS_SHOWN} more scripts")
    if parts:
        return ", ".join(parts)
    declared = f.fields.get("Caller")
    return "by hand" if declared and declared[1].startswith("by hand") else "none"


# ---- index ---------------------------------------------------------------

INDEX_PREAMBLE = (
    "# scripts/ — index\n"
    "\n"
    "Generated by `make scripts-index` from each file's header and the callers the\n"
    "Makefile and the other scripts give it; `make check-scripts` fails when it\n"
    "differs. Never edit it by hand. The rules are in [README.md](README.md).\n"
)
ROW_RE = re.compile(r"^\| `([^`]+)` \|")


def render_index(files: list[Script], callers: dict[str, Callers]) -> str:
    groups: dict[str, list[Script]] = defaultdict(list)
    for f in files:
        groups[os.path.dirname(f.rel)].append(f)
    out = [INDEX_PREAMBLE]
    for directory in sorted(groups):
        out.append(f"## scripts/{directory}/" if directory else "## scripts/")
        out.append("")
        out.append("| path | caller (derived) | purpose (header line 1) |")
        out.append("|---|---|---|")
        for f in sorted(groups[directory], key=lambda s: s.rel):
            purpose = (f.purpose or "(no purpose line)").replace("|", "\\|")
            cell = caller_cell(f, callers[f.rel]).replace("|", "\\|")
            out.append(f"| `{f.rel}` | {cell} | {purpose} |")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


def index_rows(lines: list[str]) -> dict[str, tuple[int, str]]:
    rows = {}
    for number, line in enumerate(lines, 1):
        found = ROW_RE.match(line)
        if found:
            rows[found.group(1)] = (number, line)
    return rows


def index_violations(root: Path, expected: str) -> list[Violation]:
    path = root / "scripts" / INDEX
    if not path.is_file():
        return [Violation("index-stale", INDEX, 1, "missing; run `make scripts-index`")]
    have_lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    want_lines = expected.split("\n")
    if have_lines == want_lines:
        return []
    have, want = index_rows(have_lines), index_rows(want_lines)
    out = []
    for rel, (number, line) in have.items():
        if rel not in want:
            out.append(Violation("index-stale", INDEX, number, f"row for {rel} names no file"))
        elif want[rel][1] != line:
            out.append(Violation("index-stale", INDEX, number, f"row for {rel} differs from its header"))
    last = 0
    for rel, (_number, _line) in want.items():
        if rel in have:
            last = have[rel][0]
        else:
            out.append(Violation("index-stale", INDEX, last + 1, f"missing row for {rel}"))
    if not out:
        pairs = itertools.zip_longest(have_lines, want_lines)
        first = next(i for i, (a, b) in enumerate(pairs, 1) if a != b)
        out.append(Violation("index-stale", INDEX, first, "differs from the generated text"))
    return sorted(out, key=lambda v: (v.line, v.detail))
