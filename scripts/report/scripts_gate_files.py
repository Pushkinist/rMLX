"""The population of scripts/ and the header each file carries.

Module of: report/debt_report.py

Reads what git lists under `<root>/scripts/` into `Script` records: the header
fields, the purpose line and the text. Also the violation record every rule
yields and the git helpers the gate shares.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

INDEX = "INDEX.md"
README = "README.md"
FIELD_NAMES = (
    "Usage",
    "Exit",
    "Caller",
    "Recall test for",
    "Module of",
    "Not a mode of",
    "LOC-exempt",
)
FIELD_RE = re.compile(r"^(" + "|".join(FIELD_NAMES) + r"):\s*(.*)$")
# A line after the purpose that begins a field or a marker is not a wrapped purpose.
NOT_WRAP_RE = re.compile(r"^(?:" + "|".join(FIELD_NAMES) + r"|doc-refs):")


class Unavailable(Exception):
    """The tree cannot be measured: the run exits 2, never with a pass."""


@dataclass(frozen=True)
class Violation:
    rule: str
    path: str
    line: int
    detail: str
    key: str = ""

    def text(self) -> str:
        return f"{self.rule}: {self.path}:{self.line}: {self.detail}"


@dataclass
class Script:
    rel: str
    text: str
    shebang: bool
    purpose_line: int
    purpose: str
    wrap_line: int
    fields: dict[str, tuple[int, str]]
    executable: bool = False

    @property
    def is_code(self) -> bool:
        return self.rel.endswith((".sh", ".py"))

    @property
    def shebang_line(self) -> str:
        return self.text.split("\n", 1)[0] if self.shebang else ""


def first_token(text: str) -> str:
    parts = text.strip().split()
    return parts[0].strip("`").removeprefix("scripts/") if parts else ""


def named_file(f: Script, label: str, rels: set[str]) -> str:
    """The file a header field names, when it is another file of the tree."""
    declared = f.fields.get(label)
    token = first_token(declared[1]) if declared else ""
    return token if token in rels and token != f.rel else ""



def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise Unavailable(f"git {args[0]} exit {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def git_ok(root: Path, *args: str) -> str | None:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return proc.stdout if proc.returncode == 0 else None


def header_lines(rel: str, text: str) -> tuple[bool, list[tuple[int, str]], int]:
    """(has shebang, header lines as (file line, text), file line of a missing purpose)."""
    lines = text.split("\n")
    shebang = bool(lines) and lines[0].startswith("#!")
    start = 1 if shebang else 0
    if rel.endswith(".py"):
        try:
            body = ast.parse(text).body
        except SyntaxError:
            return shebang, [], start + 1
        first = body[0] if body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            doc = first.value.value.split("\n")
            return shebang, [(first.lineno + k, t.rstrip()) for k, t in enumerate(doc)], first.lineno
        return shebang, [], start + 1
    if rel.endswith(".entitlements"):
        return shebang, xml_comment(lines), 1
    out: list[tuple[int, str]] = []
    for number in range(start, len(lines)):
        stripped = lines[number].strip()
        if stripped.startswith("#"):
            body_text = stripped[1:]
            out.append((number + 1, body_text[1:] if body_text.startswith(" ") else body_text))
        elif stripped == "":
            out.append((number + 1, ""))
        else:
            break
    if out and out[0][0] != start + 1:
        out = []
    return shebang, [(n, t.rstrip()) for n, t in out], start + 1


def xml_comment(lines: list[str]) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    inside = False
    for number, raw in enumerate(lines, 1):
        text = raw.strip()
        if not inside:
            if "<!--" not in text:
                continue
            inside = True
            text = text.split("<!--", 1)[1].strip()
            if text:
                out.append((number, text))
            continue
        if "-->" in text:
            text = text.split("-->", 1)[0].strip()
            if text:
                out.append((number, text))
            break
        out.append((number, text))
    while out and out[0][1] == "":
        out.pop(0)
    return out


def parse_fields(hlines: list[tuple[int, str]]) -> dict[str, tuple[int, str]]:
    fields: dict[str, tuple[int, str]] = {}
    i = 0
    while i < len(hlines):
        number, text = hlines[i]
        found = FIELD_RE.match(text)
        if not found:
            i += 1
            continue
        parts = [found.group(2)]
        j = i + 1
        while j < len(hlines) and hlines[j][1].strip() and not FIELD_RE.match(hlines[j][1]):
            parts.append(hlines[j][1].strip())
            j += 1
        fields.setdefault(found.group(1), (number, "\n".join(parts).strip()))
        i = j
    return fields


def load_script(root: Path, rel: str, executable: bool) -> Script:
    path = root / "scripts" / rel
    text = path.read_text(encoding="utf-8", errors="replace")
    shebang, hlines, missing_line = header_lines(rel, text)
    purpose_line, purpose, wrap_line = missing_line, "", 0
    if hlines:
        purpose_line, purpose = hlines[0]
        if len(hlines) > 1:
            number, nxt = hlines[1]
            if nxt.strip() and not NOT_WRAP_RE.match(nxt):
                wrap_line = number
    return Script(
        rel, text, shebang, purpose_line, purpose.strip(), wrap_line, parse_fields(hlines), executable
    )


def index_modes(root: Path) -> dict[str, bool]:
    """Path -> whether git's index records the executable bit, for tracked files."""
    modes = {}
    for entry in git(root, "ls-files", "--stage", "-z", "--", "scripts").split("\0"):
        if "\t" in entry:
            info, path = entry.split("\t", 1)
            modes[path] = info.startswith("100755")
    return modes


def scripts_tree(root: Path) -> set[str]:
    """Every file under `<root>/scripts/` git lists, relative to it: fixtures,
    INDEX.md and README.md included."""
    listing = git(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "scripts"
    )
    rels = set()
    for entry in listing.split("\0"):
        if entry.startswith("scripts/") and (root / entry).is_file():
            rels.add(entry[len("scripts/") :])
    return rels


def git_ignored(root: Path, paths: list[str]) -> set[str]:
    """The paths among `paths` that git ignores."""
    if not paths:
        return set()
    proc = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-z", "--stdin"],
        input="\0".join(paths),
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        raise Unavailable(f"git check-ignore exit {proc.returncode}: {proc.stderr.strip()}")
    return {p for p in proc.stdout.split("\0") if p}


def load_population(root: Path) -> list[Script]:
    rels = {
        rel
        for rel in scripts_tree(root)
        if not rel.startswith("fixtures/") and rel not in (INDEX, README)
    }
    if not rels:
        raise Unavailable("no file under scripts/ to check")
    modes = index_modes(root)
    return [
        load_script(root, rel, modes.get(f"scripts/{rel}", os.access(root / "scripts" / rel, os.X_OK)))
        for rel in sorted(rels)
    ]
