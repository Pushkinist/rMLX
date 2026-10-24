"""Readers of script text: shell, Python and Makefile source.

Module of: report/debt_report.py

Blanks comments, heredoc bodies and strings out of shell, finds the literal
exit codes of a shell or Python file, and reads which files a shell file, a
Python file or a Makefile recipe runs, sources or mentions.
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

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
SUBPROCESS_CALLS = frozenset({"run", "check_output", "check_call", "call", "Popen"})


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


def bound_strings(tree: ast.AST) -> dict[str, set[str]]:
    """Name -> the string constants of every value assigned to it, the names
    inside a value resolved through the assignments of those names."""
    direct: dict[str, set[str]] = defaultdict(set)
    uses: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            targets = [node.target]
        else:
            continue
        for target in targets:
            for sub in ast.walk(node.value):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    direct[target.id].add(sub.value)
                elif isinstance(sub, ast.Name):
                    uses[target.id].add(sub.id)
    changed = True
    while changed:
        changed = False
        for name, used in uses.items():
            for other in used:
                if not direct[other] <= direct[name]:
                    direct[name] |= direct[other]
                    changed = True
    return direct


def subprocess_names(tree: ast.AST) -> set[str]:
    """The local names `from subprocess import ...` binds to a process call."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "subprocess" and node.level == 0:
            names.update(a.asname or a.name for a in node.names if a.name in SUBPROCESS_CALLS)
    return names


def python_refs(text: str) -> Refs:
    """What a Python file imports, runs (a command vector passed to a call, or a
    name bound to a path and passed as the program of a `subprocess` call) and
    mentions (any string constant)."""
    refs = Refs()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return refs
    bound = bound_strings(tree)
    local_calls = subprocess_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in [*node.args, *(keyword.value for keyword in node.keywords)]:
                if isinstance(arg, (ast.List, ast.Tuple)):
                    refs.run.update(
                        e.value for e in arg.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
            program = node.args[0] if node.args else None
            if program is not None and is_subprocess_call(node.func, local_calls):
                for sub in ast.walk(program):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        refs.run.add(sub.value)
                    elif isinstance(sub, ast.Name):
                        refs.run.update(bound.get(sub.id, ()))
        elif isinstance(node, ast.Import):
            refs.source.update(alias.name.split(".")[0] + ".py" for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            refs.source.add(node.module.split(".")[0] + ".py")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            refs.anywhere.add(node.value)
    return refs


def is_subprocess_call(func: ast.expr, local_calls: set[str]) -> bool:
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id == "subprocess" and func.attr in SUBPROCESS_CALLS
    return isinstance(func, ast.Name) and func.id in local_calls


def make_rules(text: str) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(target -> recipe lines, target -> prerequisites). Comments and
    continuations are resolved; a prerequisite that holds a variable or a
    pattern is dropped."""
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
    prerequisites: dict[str, list[str]] = defaultdict(list)
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
            after = line[found.end() :].split("#", 1)[0]
            needs = [w for w in after.split() if w != "|" and not any(ch in w for ch in "$%=")]
            for target in current:
                prerequisites[target].extend(needs)
        else:
            current = []
    return recipes, prerequisites


def make_recipes(text: str) -> dict[str, list[str]]:
    """Target name -> recipe lines, comments and continuations resolved."""
    return make_rules(text)[0]


def make_refs(lines: list[str]) -> Refs:
    recipe = "\n".join(line.lstrip("@-+ \t") for line in lines)
    return shell_refs(clean_shell(recipe, False, True))
