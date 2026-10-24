"""The rules for scripts/ and the generated scripts/INDEX.md.

Module of: report/debt_report.py

Run as `debt_report.py --scripts [--root DIR] [--base REF] [--write-index]`.
The population is what git lists under `<root>/scripts/` (tracked, or untracked
and not ignored) without `fixtures/`, `INDEX.md` and `README.md`. A violation
prints as `<rule>: <path>:<line>: <detail>`, the path relative to `scripts/`.

Each rule is a function over the population that yields violations. A pin is a
`<rule> <path>` line in `report/scripts_gate_pins.txt` and accepts every
violation of that rule in that file (an `exit-undeclared` pin lists the codes).
Pins only shrink: a pin whose file now passes is a violation (`pin-stale`), and
a pin the base lacks, for a rule the base's `# pinnable:` line lists, is one too
(`pin-new`). Only the rules in `PINNABLE` can be pinned.

Definitions the rules share:

* An entry point is a `.sh` or `.py` file that is run: it starts with a shebang
  outside `lib/`, or a recipe or a script runs it (not sources or imports it).
  A file whose `Module of:` names another file of the tree is not one.
* The header is the leading comment block (Python: the module docstring;
  `.entitlements`: the first XML comment). Its first line, directly after the
  shebang, is the purpose; `Usage:`, `Exit:`, `Caller:`, `Recall test for:`,
  `Module of:`, `Not a mode of:` and `LOC-exempt:` start a field that runs to
  the next blank line or field.
* A caller is a Makefile target whose recipe, or another script that, runs or
  sources the file; a Python import of the module counts. A data file has a
  reader instead, and any non-comment mention counts.
* A new file is one absent at the base, a moved file (`git mv`) excepted.

Not checked, by design: that a purpose line is true, that a `Not a mode of:`
reason is, and a `return N` from a Python `main()` (only `sys.exit(N)`,
`exit(N)` and `SystemExit(N)` are literal exits).
"""

from __future__ import annotations

import ast
import itertools
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CLASSES = (
    "check",
    "report",
    "bench",
    "published",
    "ingest",
    "suite",
    "profile",
    "host",
    "release",
    "spec",
    "gen",
    "lib",
)
INDEX = "INDEX.md"
README = "README.md"
PINS = "report/scripts_gate_pins.txt"
PURPOSE_MAX = 100
CALLERS_SHOWN = 3
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
HEREDOC_RE = re.compile(r"<<(-?)\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
EXIT_RE = re.compile(r"(?:^|[;&|(){!)]|\bthen\b|\belse\b|\bdo\b|\belif\b)\s*exit\s+([0-9]+)\b")
ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
VAR_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")
EXIT_CODE_RE = re.compile(r"(?:^|[;,|/(])[ \t]*(\d+)(?:[ \t]*[-\u2013][ \t]*(\d+))?\b", re.M)
SUBST_RE = re.compile(r"\$\(([^()]*)\)")
RULE_RE = re.compile(r"^([A-Za-z0-9_./ -]+?)\s*:(?![=:])")
KEYWORDS = frozenset({"if", "then", "else", "elif", "do", "while", "until", "!", "{", "}", "time"})
WRAPPERS = frozenset(
    {"exec", "env", "command", "nohup", "nice", "timeout", "sudo", "builtin", "st_run"}
)
INTERPRETERS = frozenset(
    {"bash", "sh", "zsh", "dash", "python", "python3", "awk", "gawk", "perl", "source", ".", "uv"}
)
SOURCING = frozenset({"source", "."})
OPTIONS_WITH_VALUE = frozenset({"-u", "-C", "-s"})
PROSE_COMMANDS = frozenset({"echo", "printf"})
PINNABLE = ("class", "purpose", "usage", "exit-undeclared", "caller")
PINS_HEADER = "pinnable:"


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

    @property
    def is_code(self) -> bool:
        return self.rel.endswith((".sh", ".py"))


def first_token(text: str) -> str:
    parts = text.strip().split()
    return parts[0].strip("`").removeprefix("scripts/") if parts else ""


def named_file(f: Script, label: str, rels: set[str]) -> str:
    """The file a header field names, when it is another file of the tree."""
    declared = f.fields.get(label)
    token = first_token(declared[1]) if declared else ""
    return token if token in rels and token != f.rel else ""


@dataclass
class Callers:
    targets: set[str] = field(default_factory=set)
    scripts: set[str] = field(default_factory=set)
    runs: bool = False

    def __bool__(self) -> bool:
        return bool(self.targets or self.scripts)


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


def load_script(root: Path, rel: str) -> Script:
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
        rel, text, shebang, purpose_line, purpose.strip(), wrap_line, parse_fields(hlines)
    )


def load_population(root: Path) -> list[Script]:
    listing = git(
        root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "scripts"
    )
    rels = set()
    for entry in listing.split("\0"):
        if not entry.startswith("scripts/"):
            continue
        rel = entry[len("scripts/") :]
        if rel.startswith("fixtures/") or rel in (INDEX, README):
            continue
        if (root / entry).is_file():
            rels.add(rel)
    if not rels:
        raise Unavailable("no file under scripts/ to check")
    return [load_script(root, rel) for rel in sorted(rels)]


def clean_shell(text: str, blank_strings: bool, join: bool = False) -> list[str]:
    """Shell source without comments and heredoc bodies; string bodies are
    blanked when asked. One entry per physical line, so line numbers hold; with
    `join`, one entry per command line, a string or a trailing backslash
    carrying a command over several physical lines."""
    lines: list[str] = []
    out: list[str] = []
    i, n = 0, len(text)
    quote = ""
    prev = "\n"
    pending: list[tuple[str, bool]] = []
    while i < n:
        c = text[i]
        if c == "\n":
            if quote and join:
                out.append(" ")
                i += 1
                continue
            lines.append("".join(out))
            out = []
            i += 1
            prev = "\n"
            if not quote and pending:
                i = skip_heredocs(text, i, pending, [] if join else lines)
                pending = []
            continue
        if c == "\\" and join and text[i + 1 : i + 2] == "\n":
            out.append(" ")
            i += 2
            continue
        if quote == '"' and c == "$" and text[i + 1 : i + 2] == "(":
            end = skip_substitution(text, i)
            segment = text[i:end].replace("\\\n", " ") if join else text[i:end]
            for ch in segment:
                if ch != "\n":
                    out.append(" " if blank_strings else ch)
                elif join:
                    out.append(" ")
                else:
                    lines.append("".join(out))
                    out = []
            i = end
            continue
        if quote:
            if quote == '"' and c == "\\" and i + 1 < n and text[i + 1] != "\n":
                out.append("  " if blank_strings else text[i : i + 2])
                i += 2
            elif c == quote:
                out.append(c)
                quote = ""
                i += 1
            else:
                out.append(" " if blank_strings else c)
                i += 1
            continue
        if c == "\\" and i + 1 < n and text[i + 1] != "\n":
            out.append(text[i : i + 2])
            i += 2
            prev = "x"
            continue
        if c in "'\"":
            quote = c
            out.append(c)
            i += 1
            prev = c
            continue
        if c == "#" and prev in " \t\n;&|()":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if c == "<" and text.startswith("<<", i) and not text.startswith("<<<", i):
            found = HEREDOC_RE.match(text, i)
            if found:
                pending.append((found.group(3), found.group(1) == "-"))
                out.append(found.group(0))
                i = found.end()
                prev = "x"
                continue
        out.append(c)
        prev = c
        i += 1
    lines.append("".join(out))
    return lines


def skip_substitution(text: str, i: int) -> int:
    """Index just past the `$( ... )` that starts at `i`, quotes inside it
    balanced on their own."""
    depth = 0
    j = i + 1
    n = len(text)
    while j < n:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "'":
            j = text.find("'", j + 1)
            if j < 0:
                return n
        elif c == '"':
            j += 1
            while j < n and text[j] != '"':
                if text[j] == "\\":
                    j += 1
                elif text[j] == "$" and text[j + 1 : j + 2] == "(":
                    j = skip_substitution(text, j) - 1
                j += 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return n


def skip_heredocs(text: str, i: int, pending: list[tuple[str, bool]], lines: list[str]) -> int:
    for tag, strip_tabs in pending:
        while i < len(text):
            end = text.find("\n", i)
            end = len(text) if end < 0 else end
            body = text[i:end]
            i = min(end + 1, len(text))
            lines.append("")
            if (body.lstrip("\t") if strip_tabs else body).rstrip() == tag:
                break
    return i


def shell_literal_exits(text: str) -> list[tuple[int, int]]:
    found = []
    for number, line in enumerate(clean_shell(text, True), 1):
        for match in EXIT_RE.finditer(line):
            found.append((number, int(match.group(1))))
    return found


def python_literal_exits(text: str) -> list[tuple[int, int]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        arg = node.args[0]
        if not (isinstance(arg, ast.Constant) and type(arg.value) is int):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else ""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            name = f"{func.value.id}.{func.attr}"
        if name in ("sys.exit", "exit", "quit", "os._exit", "SystemExit"):
            found.append((node.lineno, arg.value))
    return sorted(found)


def declared_exit_codes(text: str) -> set[int]:
    """The codes an `Exit:` field names: a number at a code position (the start
    of a line or after `;`, `,`, `|`, `/` or `(`), alone or as a range."""
    codes: set[int] = set()
    for low, high in EXIT_CODE_RE.findall(text):
        codes.update(range(int(low), int(high or low) + 1))
    return codes


# ---- callers -------------------------------------------------------------


def peel_substitutions(line: str) -> list[str]:
    """The line with each command substitution replaced by `$SUB`, then the
    substitution bodies, so a script run inside `$(...)` is still seen."""
    parts = []
    while True:
        found = SUBST_RE.search(line)
        if not found:
            break
        parts.append(found.group(1))
        line = line[: found.start()] + "$SUB" + line[found.end() :]
    return [line, *parts]


def split_commands(line: str) -> list[list[str]]:
    commands: list[list[str]] = [[]]
    word: list[str] | None = None
    quote = ""
    i, n = 0, len(line)

    def flush() -> None:
        nonlocal word
        if word is not None:
            commands[-1].append("".join(word))
            word = None

    def split() -> None:
        flush()
        if commands[-1]:
            commands.append([])

    while i < n:
        c = line[i]
        if quote:
            if c == quote:
                quote = ""
            elif c == "\\" and quote == '"' and i + 1 < n:
                word.append(line[i + 1])  # type: ignore[union-attr]
                i += 1
            else:
                word.append(c)  # type: ignore[union-attr]
        elif c in "'\"":
            quote = c
            word = word if word is not None else []
        elif c == "\\" and i + 1 < n:
            word = word if word is not None else []
            word.append(line[i + 1])
            i += 1
        elif c.isspace():
            flush()
        elif c == "$" and line[i + 1 : i + 2] == "{":
            end = line.find("}", i)
            end = n - 1 if end < 0 else end
            word = word if word is not None else []
            word.append(line[i : end + 1])
            i = end
        elif c in ";&|()`":
            split()
        else:
            word = word if word is not None else []
            word.append(c)
        i += 1
    flush()
    commands = [c for c in commands if c]
    out: list[list[str]] = []
    for command in commands:
        current: list[str] = []
        for w in command:
            if w == "--":
                if current:
                    out.append(current)
                current = []
            else:
                current.append(w)
        if current:
            out.append(current)
    return out


def interpreter_target(words: list[str], interpreter: str) -> list[str]:
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("-v", "-W", "-X", "-o"):
            i += 2
            continue
        if w.startswith("-") or (interpreter == "uv" and w == "run"):
            i += 1
            continue
        return [w]
    return []


def command_words(words: list[str]) -> tuple[list[str], list[str], list[str], list[tuple[str, str]]]:
    """(words naming what the command runs, words it sources, every word,
    the assignments in front of it)."""
    i = 0
    assigned: list[tuple[str, str]] = []
    while i < len(words):
        w = words[i]
        if w in KEYWORDS:
            i += 1
        elif ASSIGN_RE.match(w):
            name, value = w.split("=", 1)
            assigned.append((name, value))
            i += 1
        else:
            break
    if i < len(words) and os.path.basename(words[i]) in PROSE_COMMANDS:
        return [], [], [], assigned
    run: list[str] = []
    source: list[str] = []
    while i < len(words):
        w = words[i]
        base = os.path.basename(w)
        following = os.path.basename(words[i + 1]) if i + 1 < len(words) else ""
        carries = following in (WRAPPERS | INTERPRETERS) - SOURCING
        if base in WRAPPERS or (base not in INTERPRETERS and carries):
            i += 1
            while i < len(words) and (
                words[i].startswith("-") or ASSIGN_RE.match(words[i]) or words[i].isdigit()
            ):
                i += 2 if words[i] in OPTIONS_WITH_VALUE else 1
            continue
        if base in INTERPRETERS:
            (source if base in SOURCING else run).extend(interpreter_target(words[i + 1 :], base))
        else:
            run.append(w)
        break
    return run, source, list(words), assigned


@dataclass
class Refs:
    run: set[str] = field(default_factory=set)
    source: set[str] = field(default_factory=set)
    anywhere: set[str] = field(default_factory=set)


def shell_refs(lines: list[str]) -> Refs:
    """What a shell file runs, sources and mentions. A `VAR=path` assignment
    credits the path only where the file later puts `$VAR` where a command
    goes."""
    refs = Refs()
    values: dict[str, list[str]] = defaultdict(list)

    def put(target: set[str], word: str, seen: frozenset[str] = frozenset()) -> None:
        target.add(word)
        found = VAR_RE.match(word)
        if found and found.group(1) not in seen:
            for value in values.get(found.group(1), ()):
                put(target, value, seen | {found.group(1)})

    for logical in lines:
        for piece in peel_substitutions(logical):
            for command in split_commands(piece):
                run, source, every, assigned = command_words(command)
                for name, value in assigned:
                    values[name].append(value)
                for word in run:
                    put(refs.run, word)
                for word in source:
                    put(refs.source, word)
                refs.anywhere.update(every)
    return refs


def python_refs(text: str) -> Refs:
    """What a Python file imports, runs (a command vector passed to a call) and
    mentions (any string constant)."""
    refs = Refs()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return refs
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in [*node.args, *(keyword.value for keyword in node.keywords)]:
                if isinstance(arg, (ast.List, ast.Tuple)):
                    refs.run.update(
                        e.value for e in arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
        elif isinstance(node, ast.Import):
            refs.source.update(alias.name.split(".")[0] + ".py" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            refs.source.add(node.module.split(".")[0] + ".py")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            refs.anywhere.add(node.value)
    return refs


def make_recipes(text: str) -> dict[str, list[str]]:
    """Target name -> recipe lines, comments and continuations resolved."""
    logical: list[tuple[bool, str]] = []
    held = ""
    for raw in text.split("\n"):
        if raw.endswith("\\"):
            held += raw[:-1] + " "
            continue
        line = held + raw
        held = ""
        logical.append((line.startswith("\t"), line))
    recipes: dict[str, list[str]] = defaultdict(list)
    current: list[str] = []
    for is_recipe, line in logical:
        if is_recipe:
            for target in current:
                recipes[target].append(line.strip())
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        found = RULE_RE.match(line)
        if found and "$(" not in found.group(1):
            current = [t for t in found.group(1).split() if not t.startswith(".") and "%" not in t]
        else:
            current = []
    return recipes


def make_refs(lines: list[str]) -> Refs:
    recipe = "\n".join(line.lstrip("@-+ \t") for line in lines)
    return shell_refs(clean_shell(recipe, False, True))


def derive_callers(files: list[Script], makefile: str) -> dict[str, Callers]:
    by_base: dict[str, list[Script]] = defaultdict(list)
    for f in files:
        by_base[os.path.basename(f.rel)].append(f)
    callers = {f.rel: Callers() for f in files}

    def credit(refs: Refs, who: str, is_target: bool) -> None:
        for words, want_code, runs in (
            (refs.run, True, True),
            (refs.source, True, False),
            (refs.anywhere, False, False),
        ):
            for word in words:
                if not word or any(ch.isspace() for ch in word):
                    continue
                candidates = by_base.get(os.path.basename(word), [])
                for f in candidates:
                    if f.is_code != want_code or f.rel == who:
                        continue
                    if len(candidates) == 1 or word == f.rel or word.endswith("/" + f.rel):
                        (callers[f.rel].targets if is_target else callers[f.rel].scripts).add(who)
                        callers[f.rel].runs = callers[f.rel].runs or runs

    for f in files:
        if f.rel.endswith(".sh"):
            credit(shell_refs(clean_shell(f.text, False, True)), f.rel, False)
        elif f.rel.endswith(".py"):
            credit(python_refs(f.text), f.rel, False)
    for target, recipe in make_recipes(makefile).items():
        credit(make_refs(recipe), target, True)
    return callers


def caller_cell(f: Script, callers: Callers) -> str:
    parts = [f"make {t}" for t in sorted(callers.targets)]
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


# ---- rules ---------------------------------------------------------------


def entry_points(files: list[Script], callers: dict[str, Callers]) -> set[str]:
    """Shell and Python files that are run: they start with a shebang outside
    lib/, or something runs them. A file that names another file in `Module of:`
    belongs to that file and is not one."""
    rels = {f.rel for f in files}
    return {
        f.rel
        for f in files
        if f.is_code
        and not named_file(f, "Module of", rels)
        and ((f.shebang and not f.rel.startswith("lib/")) or callers[f.rel].runs)
    }


@dataclass
class Context:
    root: Path
    files: list[Script]
    callers: dict[str, Callers]
    added: set[str]
    entries: set[str]

    @property
    def rels(self) -> set[str]:
        return {f.rel for f in self.files}


def rule_class(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        top = f.rel.split("/")[0]
        if "/" not in f.rel:
            out.append(Violation("class", f.rel, 1, "a top-level file; it belongs in a class directory"))
        elif top not in CLASSES:
            out.append(Violation("class", f.rel, 1, f"{top}/ is not a class directory"))
    return out


def rule_purpose(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if not f.purpose:
            out.append(Violation("purpose", f.rel, f.purpose_line, "no purpose line at the top of the header"))
        elif len(f.purpose) > PURPOSE_MAX:
            out.append(
                Violation("purpose", f.rel, f.purpose_line, f"purpose is {len(f.purpose)} characters, over {PURPOSE_MAX}")
            )
        elif f.wrap_line:
            out.append(Violation("purpose", f.rel, f.wrap_line, "the purpose line is wrapped onto this line"))
    return out


def rule_usage(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if f.rel not in ctx.entries:
            continue
        declared = f.fields.get("Usage")
        if not declared:
            out.append(Violation("usage", f.rel, 1, "entry point has no `Usage:` line in its header"))
        elif not declared[1]:
            out.append(Violation("usage", f.rel, declared[0], "`Usage:` says nothing"))
    return out


def rule_exit_undeclared(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if f.rel not in ctx.entries:
            continue
        exits = shell_literal_exits(f.text) if f.rel.endswith(".sh") else python_literal_exits(f.text)
        declared = declared_exit_codes(f.fields.get("Exit", (0, ""))[1])
        seen = set()
        for number, code in exits:
            if code not in declared and code not in seen:
                seen.add(code)
                detail = f"{code} is not declared in `Exit:`"
                out.append(Violation("exit-undeclared", f.rel, number, detail, key=str(code)))
    return out


def rule_caller(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if f.rel not in ctx.entries:
            continue
        derived = ctx.callers[f.rel]
        declared = f.fields.get("Caller")
        by_hand = bool(declared) and declared[1].startswith("by hand")
        if by_hand and derived:
            out.append(Violation("caller", f.rel, declared[0], "declares `Caller: by hand` but something runs it"))
        elif not by_hand and not derived:
            out.append(Violation("caller", f.rel, 1, "nothing runs it and the header has no `Caller: by hand`"))
    return out


def not_a_mode_target(text: str) -> tuple[str, str]:
    for separator in (" — ", " -- ", " - "):
        if separator in text:
            tool, reason = text.split(separator, 1)
            break
    else:
        tool, reason = text, ""
    tool = tool.strip().strip("`")
    return tool.removeprefix("scripts/"), reason.strip()


def rule_not_a_mode(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if f.rel not in ctx.added or f.rel not in ctx.entries:
            continue
        if named_file(f, "Recall test for", ctx.rels):
            continue
        declared = f.fields.get("Not a mode of")
        if not declared:
            out.append(Violation("not-a-mode", f.rel, 1, "added since the base without `Not a mode of: <tool> — <reason>`"))
        elif not not_a_mode_target(declared[1])[1]:
            out.append(Violation("not-a-mode", f.rel, declared[0], "`Not a mode of:` gives no reason"))
    return out


def rule_not_a_mode_stale(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        declared = f.fields.get("Not a mode of")
        if declared:
            tool = not_a_mode_target(declared[1])[0]
            if tool not in ctx.entries or tool == f.rel:
                out.append(
                    Violation("not-a-mode-stale", f.rel, declared[0], f"names {tool or 'nothing'}, which is not another entry point")
                )
    return out


RULE_FUNCTIONS = (
    ("class", rule_class),
    ("purpose", rule_purpose),
    ("usage", rule_usage),
    ("exit-undeclared", rule_exit_undeclared),
    ("caller", rule_caller),
    ("not-a-mode", rule_not_a_mode),
    ("not-a-mode-stale", rule_not_a_mode_stale),
)


# ---- base, pins, run -----------------------------------------------------


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


@dataclass
class PinList:
    exists: bool = False
    pins: dict[tuple[str, str], Pin] = field(default_factory=dict)
    pinnable: set[str] | None = None
    header_line: int = 1
    problems: list[Violation] = field(default_factory=list)


def parse_pins(text: str) -> PinList:
    """`<rule> <path>` per line; `exit-undeclared` adds its undeclared codes as
    `<c1,c2>`; `# pinnable: <rule> ...` names the rules a pin may be added for."""
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
        codes = frozenset(parts[2].split(",")) if len(parts) > 2 else frozenset()
        out.pins[key] = Pin(rule, path, codes, number)
    return out


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
        groups[(violation.rule, violation.path)].append(violation)
    for key, violations in groups.items():
        pin = pins.pins.get(key)
        if pin is None or key[0] not in PINNABLE:
            active += violations
            continue
        if key[0] == "exit-undeclared":
            active += [v for v in violations if v.key not in pin.codes]
            for code in sorted(pin.codes - {v.key for v in violations}):
                detail = f"{key[0]} {key[1]}: code {code} is declared or gone; delete it from the pin"
                active.append(Violation("pin-stale", PINS, pin.line, detail))
        held[key] = pin
    for key, pin in pins.pins.items():
        if key[0] not in PINNABLE:
            active.append(Violation("pin-stale", PINS, pin.line, f"{key[0]} {key[1]}: {key[0]} cannot be pinned"))
        elif key not in groups:
            active.append(Violation("pin-stale", PINS, pin.line, f"{key[0]} {key[1]}: no longer violated; delete the pin"))
    return active, held


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
        old = before.pins.get((rule, renames.get(path, path)))
        if old is None:
            out.append(Violation("pin-new", PINS, pin.line, f"{rule} {path} is not pinned at the base"))
            continue
        for code in sorted(pin.codes - old.codes):
            out.append(Violation("pin-new", PINS, pin.line, f"{rule} {path}: code {code} is not pinned at the base"))
    return out, ""


def run(root: Path, base: str | None, write_index: bool) -> int:
    try:
        return check(root, base, write_index)
    except Unavailable as exc:
        print(f"check-scripts: unavailable ({exc})", file=sys.stderr)
        return 2


def check(root: Path, base: str | None, write_index: bool) -> int:
    if git_ok(root, "rev-parse", "--is-inside-work-tree") is None:
        raise Unavailable(f"{root} is not inside a git work tree")
    files = load_population(root)
    name, sha = resolve_base(root, base)
    makefile = root / "Makefile"
    callers = derive_callers(files, makefile.read_text(encoding="utf-8") if makefile.is_file() else "")
    expected = render_index(files, callers)
    if write_index:
        (root / "scripts" / INDEX).write_text(expected, encoding="utf-8")
        print(f"scripts-index: wrote scripts/{INDEX} ({len(files)} rows)")
    print(f"check-scripts: base {name} ({sha[:12]})")
    added, renames = changes_since(root, sha)
    ctx = Context(root, files, callers, added, entry_points(files, callers))
    found: list[Violation] = []
    for _name, rule in RULE_FUNCTIONS:
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
