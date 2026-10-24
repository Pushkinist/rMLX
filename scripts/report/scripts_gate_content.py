"""Per-file content rules of the scripts gate: shell mode, Python floor, names, size, heredocs, knobs.

Module of: report/debt_report.py

Each rule is a function over the `Context` that yields violations; the
registry is `CONTENT_RULES`. A rule here reads what is written in one file
(its code, its header, its length) and not how files relate to each other.
"""

from __future__ import annotations

import ast
import os
import re
from collections.abc import Iterator

from scripts_gate_files import Script, Violation
from scripts_gate_rules import Context, not_a_mode_target
from scripts_gate_structure import LIB_PREFIX, is_recall_test
from scripts_gate_text import HEREDOC_RE, KEYWORDS, clean_shell, split_commands

PY_FLOOR = (3, 9)
SIZE_LIMIT = 1000
HEREDOC_LIMIT = 20
ROOT_VARIABLE = "REPO_ROOT"
IDENTITY_FILE = "lib/identity.sh"
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
ASSIGN_RE = re.compile(r"(?:^|[\s;&|({])(?:(?:local|export|readonly|declare|typeset)\s+(?:-\w+\s+)*)?([A-Za-z_]\w*)\+?=")
OWN_DIR_UP_RE = re.compile(r'dirname\s+"(?:\$0|\$\{BASH_SOURCE\[0\]\})"\)((?:/\.\.)*)')
TOPLEVEL_RE = re.compile(r"rev-parse\s+--show-toplevel")
PYTHON_WORD_RE = re.compile(r"\bpython3?\b")
PYTHON_C_RE = re.compile(r"""\bpython3?\b[^\n]*?\s-c\s+(['"])""")
BINARY_RE = re.compile(r"target/[^/\s\"']+/rmlx(?![\w-])")
FUNCTION_RE = re.compile(r"^\s*(?:function\s+([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*\(\s*\))")
ENV_READ_RE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::?[-=?])")
ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
DECLARING = frozenset({"local", "declare", "typeset", "export", "readonly", "read"})
SYSTEM_ENV = frozenset(
    {
        "HOME", "PATH", "TMPDIR", "PWD", "OLDPWD", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL", "TZ", "CI",
        "NO_COLOR", "BASH_SOURCE", "BASH_VERSION", "BASH_REMATCH", "FUNCNAME", "PIPESTATUS", "LINENO", "RANDOM",
        "SECONDS", "HOSTNAME", "UID", "EUID", "OSTYPE", "IFS", "REPLY", "OPTARG", "OPTIND", "PPID", "SHLVL",
    }
)  # fmt: skip


# ---- set-mode ------------------------------------------------------------


def set_options(text: str) -> tuple[set[str], set[str]]:
    """(option letters, `-o` names) of every `set` command in the shell text."""
    letters: set[str] = set()
    names: set[str] = set()
    for line in clean_shell(text, True, True):
        for command in split_commands(line):
            while command and command[0] in KEYWORDS:
                command = command[1:]
            if not command or command[0] != "set":
                continue
            i = 1
            while i < len(command) and re.fullmatch(r"-[A-Za-z]+", command[i]):
                cluster = command[i]
                letters.update(cluster[1:])
                if cluster.endswith("o") and i + 1 < len(command):
                    names.add(command[i + 1])
                    i += 1
                i += 1
    return letters, names


def rule_set_mode(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        if not f.rel.endswith(".sh") or f.rel not in ctx.entries:
            continue
        letters, names = set_options(f.text)
        pipefail = "pipefail" in names
        declared = f.fields.get("Set")
        if {"e", "u"} <= letters and pipefail:
            if declared:
                yield Violation("set-mode", f.rel, declared[0], "`Set:` is declared but the file runs under `set -euo pipefail`")
        elif "u" in letters and pipefail:
            if not declared or not not_a_mode_target(declared[1])[1]:
                yield Violation("set-mode", f.rel, 1, "`set -uo pipefail` without `Set: -uo — <reason>` in the header")
        else:
            yield Violation("set-mode", f.rel, 1, "no `set -euo pipefail`")


# ---- py-floor ------------------------------------------------------------


def has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def evaluated_annotations(node: ast.AST, in_function: bool) -> Iterator[ast.expr]:
    """The annotations Python evaluates when the definition runs: every
    argument and return annotation, and a variable annotation outside a
    function."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = node.args
        every = [*args.posonlyargs, *args.args, *args.kwonlyargs, *(a for a in (args.vararg, args.kwarg) if a)]
        yield from (a.annotation for a in every if a.annotation)
        if node.returns:
            yield node.returns
        in_function = True
    elif isinstance(node, ast.AnnAssign) and not in_function:
        yield node.annotation
    for child in ast.iter_child_nodes(node):
        yield from evaluated_annotations(child, in_function)


def rule_py_floor(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        if not f.rel.endswith(".py"):
            continue
        try:
            tree = ast.parse(f.text, feature_version=PY_FLOOR)
        except SyntaxError as exc:
            yield Violation("py-floor", f.rel, exc.lineno or 1, f"does not parse as Python 3.9: {exc.msg}")
            continue
        if has_future_annotations(tree):
            continue
        for annotation in evaluated_annotations(tree, False):
            if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitOr) for n in ast.walk(annotation)):
                detail = "a `|` union in an annotation needs `from __future__ import annotations` on Python 3.9"
                yield Violation("py-floor", f.rel, annotation.lineno, detail)


# ---- name ----------------------------------------------------------------


def shell_root_variables(f: Script) -> Iterator[tuple[int, str]]:
    """Variables assigned the directory above the file's own: `dirname` of
    `$0` or `${BASH_SOURCE[0]}` followed by one `/..` for each path part of the
    file under scripts/, or the work tree's top level."""
    depth = len(f.rel.split("/"))
    for number, line in enumerate(clean_shell(f.text, False), 1):
        found = ASSIGN_RE.search(line)
        value = line[found.end() :] if found else ""
        if not found or value.startswith("'"):
            continue
        up = OWN_DIR_UP_RE.search(value)
        if TOPLEVEL_RE.search(value) or (up and up.group(1).count("/..") == depth):
            yield number, found.group(1)


def levels_above_file(value: ast.expr) -> int:
    """How many directories a Python expression built from `__file__` climbs:
    one for the file's own directory, then one for each `.parent`, each
    `dirname` and `k + 1` for `.parents[k]`. -1 when it is not built from
    `__file__` or when it goes on into a path below the directory it reached."""
    if not any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(value)):
        return -1
    if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in ast.walk(value)):
        return -1
    levels = 0
    for node in ast.walk(value):
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            levels += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dirname":
            levels += 1
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            index = node.slice
            index = getattr(index, "value", index) if not isinstance(index, ast.Constant) else index
            levels += (index.value + 1) if isinstance(index, ast.Constant) and isinstance(index.value, int) else 0
    return levels


def python_root_variables(f: Script) -> Iterator[tuple[int, str]]:
    try:
        tree = ast.parse(f.text)
    except SyntaxError:
        return
    depth = len(f.rel.split("/")) + 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            names = [node.target.id]
        else:
            continue
        if names and levels_above_file(node.value) == depth:  # type: ignore[arg-type]
            yield node.lineno, names[0]


def rule_name(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        if "-" in os.path.basename(f.rel):
            yield Violation("name", f.rel, 1, "a hyphen in the file name; use an underscore", key="hyphen")
        roots = shell_root_variables(f) if f.rel.endswith(".sh") else python_root_variables(f) if f.rel.endswith(".py") else ()
        for number, name in roots:
            if name != ROOT_VARIABLE:
                yield Violation("name", f.rel, number, f"the repository root variable is {name}; name it {ROOT_VARIABLE}", key=name)


# ---- size ----------------------------------------------------------------


def rule_size(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        lines = len(f.text.splitlines())
        exempt = f.fields.get("LOC-exempt")
        if lines > SIZE_LIMIT and not (exempt and exempt[1]):
            detail = f"{lines} lines, over {SIZE_LIMIT}, with no `LOC-exempt: <why>` in the header"
            yield Violation("size", f.rel, 1, detail, figure=lines)


# ---- heredoc -------------------------------------------------------------


def closing_quote(text: str, opening: int) -> int:
    """Index of the quote that closes the one at `opening`; the end of the text
    when it never closes. Inside double quotes a backslash escapes."""
    quote = text[opening]
    i = opening + 1
    while i < len(text) and text[i] != quote:
        i += 2 if quote == '"' and text[i] == "\\" else 1
    return min(i, len(text))


def python_bodies(text: str) -> Iterator[tuple[int, int]]:
    """(line, length in lines) of every Python body in shell text: the lines
    between a `python` command's heredoc opener and its terminator, or the
    lines of a `python -c` string. A heredoc of another command is skipped
    whole, so a planted file that holds Python is not read as Python. Openers
    are looked for in the text with its comments, heredoc bodies and string
    bodies blanked, so one inside a string is not an opener."""
    lines = text.split("\n")
    blanked = clean_shell(text, True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line) + 1)
    i = 0
    while i < len(lines):
        line = blanked[i]
        opener = next((m for m in HEREDOC_RE.finditer(line) if line[m.start() - 1 : m.start()] != "<"), None)
        if opener:
            end = i + 1
            while end < len(lines):
                body = lines[end].lstrip("\t") if opener.group(1) else lines[end]
                if body.rstrip() == opener.group(3):
                    break
                end += 1
            if PYTHON_WORD_RE.search(line):
                yield i + 1, end - i - 1
            i = end + 1
            continue
        command = PYTHON_C_RE.search(line)
        if command:
            opening = starts[i] + command.end() - 1
            closing = closing_quote(text, opening)
            body = text[opening + 1 : closing].split("\n")
            body = body[1:] if body and not body[0].strip() else body
            body = body[:-1] if body and not body[-1].strip() else body
            yield i + 1, len(body)
            i = text.count("\n", 0, closing) + 1
            continue
        i += 1


def rule_heredoc(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        if not f.rel.endswith(".sh"):
            continue
        for number, length in python_bodies(f.text):
            if length > HEREDOC_LIMIT:
                detail = f"a Python body of {length} lines, over {HEREDOC_LIMIT}; make it a .py file"
                yield Violation("heredoc", f.rel, number, detail, figure=length)


# ---- lib-shadow ----------------------------------------------------------


def defined_functions(text: str) -> Iterator[tuple[int, str]]:
    for number, line in enumerate(clean_shell(text, False), 1):
        found = FUNCTION_RE.match(line)
        if found:
            yield number, found.group(1) or found.group(2)


def rule_lib_shadow(ctx: Context) -> Iterator[Violation]:
    shell = [f for f in ctx.files if f.rel.endswith(".sh")]
    in_lib = {name: f.rel for f in shell if f.rel.startswith(LIB_PREFIX) for _n, name in defined_functions(f.text)}
    for f in shell:
        if f.rel.startswith(LIB_PREFIX):
            continue
        for number, name in defined_functions(f.text):
            if name in in_lib:
                yield Violation("lib-shadow", f.rel, number, f"{name}() is defined in {in_lib[name]}", key=name)


# ---- binary-literal ------------------------------------------------------


def rule_binary_literal(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        if f.rel == IDENTITY_FILE or is_recall_test(f, ctx):
            continue
        for number, line in enumerate(f.text.split("\n"), 1):
            if line.lstrip().startswith("#"):
                continue
            for found in BINARY_RE.finditer(line):
                detail = f"`{found.group(0)}` is written out; take the binary from lib/identity.sh"
                yield Violation("binary-literal", f.rel, number, detail, figure=1)


# ---- env-knob ------------------------------------------------------------


def assigned_names(line: str) -> set[str]:
    names = set(ASSIGN_RE.findall(line))
    names.update(re.findall(r"\bfor\s+([A-Za-z_]\w*)\s+in\b", line))
    words = line.split()
    if words and words[0] in DECLARING:
        names.update(w for w in words[1:] if IDENTIFIER_RE.fullmatch(w))
    return names


def shell_env_reads(text: str) -> Iterator[tuple[int, str]]:
    """(line, name) of each `${NAME:-…}`-style read of an upper-case name the
    file has not assigned on an earlier line."""
    assigned: set[str] = set()
    for number, line in enumerate(clean_shell(text, False), 1):
        for found in ENV_READ_RE.finditer(line):
            if found.group(1) not in assigned:
                yield number, found.group(1)
        assigned |= assigned_names(line)


def is_environ(node: ast.expr) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "environ") or (isinstance(node, ast.Name) and node.id == "environ")


def python_env_reads(text: str) -> Iterator[tuple[int, str]]:
    """(line, name) of `os.environ[...]`, `os.environ.get(...)` and
    `os.getenv(...)` with a literal name."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    for node in ast.walk(tree):
        key: ast.expr | None = None
        if isinstance(node, ast.Subscript) and is_environ(node.value) and isinstance(node.ctx, ast.Load):
            key = node.slice if isinstance(node.slice, ast.Constant) else getattr(node.slice, "value", None)
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "get" and is_environ(func.value)) or (
                isinstance(func, ast.Attribute) and func.attr == "getenv" and isinstance(func.value, ast.Name)
            ) or (isinstance(func, ast.Name) and func.id == "getenv"):
                key = node.args[0]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            yield node.lineno, key.value


def rule_env_knob(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        if f.rel.endswith(".sh"):
            reads = shell_env_reads(f.text)
        elif f.rel.endswith(".py"):
            reads = python_env_reads(f.text)
        else:
            continue
        for number, name in reads:
            if ENV_NAME_RE.fullmatch(name) and name not in SYSTEM_ENV:
                yield Violation("env-knob", f.rel, number, f"reads the environment variable {name}", key=name)


CONTENT_RULES = (
    ("set-mode", rule_set_mode),
    ("py-floor", rule_py_floor),
    ("name", rule_name),
    ("size", rule_size),
    ("heredoc", rule_heredoc),
    ("lib-shadow", rule_lib_shadow),
    ("binary-literal", rule_binary_literal),
    ("env-knob", rule_env_knob),
)
