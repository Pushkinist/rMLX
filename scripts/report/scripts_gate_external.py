"""Callers of scripts/ that are not scripts: hosted CI steps, pre-commit hooks, Rust sources.

Module of: report/debt_report.py

A workflow run step and a pre-commit `entry:` are shell command lines. A Rust
source names a script in a string literal that is the whole path, as in
`concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/x.sh")`; comments and
text that merely mentions a path do not count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from scripts_gate_files import git

PRE_COMMIT = ".pre-commit-config.yaml"
PRE_COMMIT_LABEL = "pre-commit"
RUN_KEY_RE = re.compile(r"^(\s*)(-\s+)?run:\s*(.*)$")
ENTRY_KEY_RE = re.compile(r"^\s*(?:-\s+)?entry:\s*(.*)$")
SCRIPT_PATH_RE = re.compile(r"[\w./-]*?(?:^|/)scripts/([\w./-]+)")
RAW_START_RE = re.compile(r'b?r(#*)"')
COMMAND_NEW = "Command::new("


@dataclass
class RustPath:
    """A string literal of a Rust file that is a path into scripts/."""

    rel: str
    line: int
    runs: bool


@dataclass
class External:
    workflows: dict[str, list[str]] = field(default_factory=dict)
    pre_commit: list[str] = field(default_factory=list)
    crates: dict[str, list[RustPath]] = field(default_factory=dict)


def workflow_run_lines(text: str) -> list[str]:
    """The shell lines of every `run:` step: the inline value, or the indented
    block under a `|` or `>` scalar."""
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        found = RUN_KEY_RE.match(lines[i])
        i += 1
        if not found:
            continue
        key_column = len(found.group(1)) + (len(found.group(2)) if found.group(2) else 0)
        value = found.group(3).strip()
        if value and value[0] in "|>":
            while i < len(lines):
                body = lines[i]
                indent = len(body) - len(body.lstrip())
                if body.strip() and indent <= key_column:
                    break
                if body.strip():
                    out.append(body.strip())
                i += 1
        elif value:
            out.append(value)
    return out


def pre_commit_entries(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        found = ENTRY_KEY_RE.match(line)
        if found and found.group(1).strip():
            out.append(found.group(1).strip())
    return out


def blank(code: list[str], start: int, end: int) -> None:
    for k in range(start, end):
        if code[k] != "\n":
            code[k] = " "


def block_comment_end(text: str, start: int) -> int:
    depth, j = 1, start + 2
    while j < len(text) and depth:
        if text.startswith("/*", j):
            depth, j = depth + 1, j + 2
        elif text.startswith("*/", j):
            depth, j = depth - 1, j + 2
        else:
            j += 1
    return j


def take_string(text: str, quote: int, code: list[str], literals: list[tuple[int, str]]) -> int:
    """Record the literal whose opening quote is at `quote`; the index past it."""
    j = quote + 1
    while j < len(text) and text[j] != '"':
        j += 2 if text[j] == "\\" else 1
    j = min(j, len(text))
    literals.append((quote + 1, text[quote + 1 : j]))
    blank(code, quote + 1, j)
    return min(j + 1, len(text))


def take_raw_string(text: str, found: re.Match[str], code: list[str], literals: list[tuple[int, str]]) -> int:
    close = '"' + found.group(1)
    end = text.find(close, found.end())
    end = len(text) if end < 0 else end
    literals.append((found.end(), text[found.end() : end]))
    blank(code, found.end(), end)
    return min(end + len(close), len(text))


def skip_char_literal(text: str, i: int) -> int:
    """Past a char literal such as `'"'`; past the quote alone for a lifetime."""
    if text[i + 1 : i + 2] == "\\":
        end = text.find("'", i + 3)
        return i + 1 if end < 0 else end + 1
    if text[i + 2 : i + 3] == "'":
        return i + 3
    return i + 1


def rust_code_and_literals(text: str) -> tuple[str, list[tuple[int, str]]]:
    """(the source with comments and string bodies blanked, every string literal
    as (offset of its body, body)). The blanked source keeps the length and the
    newlines of the original, so an offset indexes both."""
    code = list(text)
    literals: list[tuple[int, str]] = []
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            end = text.find("\n", i)
            end = n if end < 0 else end
            blank(code, i, end)
            i = end
        elif text.startswith("/*", i):
            end = block_comment_end(text, i)
            blank(code, i, end)
            i = end
        elif c == '"':
            i = take_string(text, i, code, literals)
        elif c == "'":
            i = skip_char_literal(text, i)
        elif c in "rb" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            raw = RAW_START_RE.match(text, i)
            i = take_raw_string(text, raw, code, literals) if raw else i + 1
        else:
            i += 1
    return "".join(code), literals


def rust_script_paths(text: str) -> list[RustPath]:
    """Every string literal that is a path into scripts/, and whether it sits
    inside a `Command::new(` call."""
    if "scripts/" not in text:
        return []
    code, literals = rust_code_and_literals(text)
    out = []
    for offset, body in literals:
        found = SCRIPT_PATH_RE.fullmatch(body)
        if found:
            out.append(RustPath(found.group(1), text.count("\n", 0, offset) + 1, inside_command_new(code, offset)))
    return out


def inside_command_new(code: str, offset: int) -> bool:
    start = code.rfind(COMMAND_NEW, 0, offset)
    if start < 0:
        return False
    depth = 1
    for ch in code[start + len(COMMAND_NEW) : offset]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return False
    return True


def load_external(root: Path) -> External:
    out = External()
    listing = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "crates", ".github")
    for entry in sorted(listing.split("\0")):
        path = root / entry
        if not entry or not path.is_file():
            continue
        if entry.startswith(".github/workflows/") and entry.endswith((".yml", ".yaml")):
            out.workflows[entry.rsplit("/", 1)[1]] = workflow_run_lines(path.read_text(encoding="utf-8", errors="replace"))
        elif entry.startswith("crates/") and entry.endswith(".rs"):
            paths = rust_script_paths(path.read_text(encoding="utf-8", errors="replace"))
            if paths:
                out.crates[entry] = paths
    pre_commit = root / PRE_COMMIT
    if pre_commit.is_file():
        out.pre_commit = pre_commit_entries(pre_commit.read_text(encoding="utf-8", errors="replace"))
    return out
