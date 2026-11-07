"""Environment reads of the scripts gate: the knobs a file takes from the environment.

Module of: report/debt_report.py

A file reads the environment when it expands a name that nothing the file runs
under assigns: in shell `${NAME}`, `${NAME:-x}` and the other `${NAME` forms,
`${#NAME}`, bare `$NAME`, and `printenv NAME`; in Python `os.environ[...]`, its
`get`, `setdefault` and `pop`, `"NAME" in os.environ`, and `os.getenv(...)`.
`scripts/README.md` section 7 defines them.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator

from scripts_gate_files import Script, Violation
from scripts_gate_rules import Context
from scripts_gate_text import KEYWORDS, clean_shell, peel_substitutions, split_commands

ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")
IDENTIFIER_RE = re.compile(r"[A-Za-z_]\w*")
BRACED_RE = re.compile(r"(?<!\\)\$\{([#!]?)([A-Za-z_]\w*)(.?)")
BARE_RE = re.compile(r"(?<!\\)\$([A-Za-z_]\w*)")
ASSIGN_WORD_RE = re.compile(r"^([A-Za-z_]\w*)\+?=")
DECLARING = frozenset({"local", "declare", "typeset", "export", "readonly", "read"})
UNKNOWN_NAME = "?"
SYSTEM_ENV = frozenset(
    {
        "HOME", "PATH", "TMPDIR", "PWD", "OLDPWD", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LC_ALL", "TZ",
        "NO_COLOR", "BASH_SOURCE", "BASH_VERSION", "BASH_REMATCH", "FUNCNAME", "PIPESTATUS", "LINENO", "RANDOM",
        "SECONDS", "HOSTNAME", "UID", "EUID", "OSTYPE", "IFS", "REPLY", "OPTARG", "OPTIND", "PPID", "SHLVL",
    }
)  # fmt: skip


# ---- shell ---------------------------------------------------------------


def shell_commands(line: str) -> Iterator[list[str]]:
    for piece in peel_substitutions(line):
        yield from split_commands(piece)


def assigned_by(command: list[str]) -> set[str]:
    """The names a command assigns: leading `NAME=value` words, the names a
    declaring builtin takes with a value (`export NAME=v`; `read NAME` takes its
    value from input), and the variable of a `for` loop. A bare `export NAME`,
    `declare -x NAME`, `readonly NAME` or `local NAME` assigns nothing. An
    assignment whose value expands the name itself reads it instead
    (`X="${X:-d}"`) and assigns nothing."""
    out: set[str] = set()
    i = 0
    while i < len(command) and (command[i] in KEYWORDS or ASSIGN_WORD_RE.match(command[i])):
        found = ASSIGN_WORD_RE.match(command[i])
        if found and not references(command[i][found.end() :], found.group(1)):
            out.add(found.group(1))
        i += 1
    rest = command[i:]
    if rest[:1] and rest[0] in DECLARING:
        for word in rest[1:]:
            name = word.split("=", 1)[0]
            valued = "=" in word or rest[0] == "read"
            if valued and not word.startswith("-") and IDENTIFIER_RE.fullmatch(name) and not references(word[len(name) :], name):
                out.add(name)
    elif len(rest) > 2 and rest[0] == "for" and rest[2] == "in" and IDENTIFIER_RE.fullmatch(rest[1]):
        out.add(rest[1])
    elif len(rest) > 2 and rest[0] == "getopts" and IDENTIFIER_RE.fullmatch(rest[2]):
        out.add(rest[2])
    elif len(rest) > 2 and rest[0] == "printf" and rest[1] == "-v" and IDENTIFIER_RE.fullmatch(rest[2]):
        out.add(rest[2])
    elif len(rest) > 1 and rest[0] in ("mapfile", "readarray") and IDENTIFIER_RE.fullmatch(rest[-1]):
        out.add(rest[-1])
    return out


def references(value: str, name: str) -> bool:
    return bool(re.search(r"\$\{?[#!]?" + re.escape(name) + r"(?![A-Za-z0-9_])", value))


def shell_assignments(lines: list[str]) -> list[set[str]]:
    """The names each line assigns."""
    out = []
    for line in lines:
        names: set[str] = set()
        for command in shell_commands(line):
            names |= assigned_by(command)
        out.append(names)
    return out


def shell_reads(line: str) -> Iterator[tuple[str, bool]]:
    """(name, bare) of each expansion in a line. A bare read is `${NAME}`,
    `${#NAME}`, `${!NAME}` or `$NAME`; the others carry an operator
    (`${NAME:-x}`, `${NAME+x}`, `${NAME%/}`, ...). An array (`${NAME[@]}`) is not
    an environment variable."""
    for found in BRACED_RE.finditer(line):
        after = found.group(3)
        if after == "[" or after == "":
            continue
        yield found.group(2), after == "}"
    for found in BARE_RE.finditer(line):
        yield found.group(1), True


def printenv_reads(line: str) -> Iterator[str]:
    """The names `printenv` is asked for; `?` for a bare `printenv`, which
    prints them all."""
    for command in shell_commands(line):
        if command[:1] == ["printenv"]:
            names = [w for w in command[1:] if not w.startswith("-")]
            yield from (names or [UNKNOWN_NAME])


class ShellAssignments:
    """Which names each shell file assigns, and where its names come from."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.lines: dict[str, list[str]] = {}
        self.per_line: dict[str, list[set[str]]] = {}
        self.sourced: dict[str, set[str]] = {f.rel: set() for f in ctx.files if f.rel.endswith(".sh")}
        for f in ctx.files:
            if f.rel.endswith(".sh"):
                self.lines[f.rel] = clean_shell(f.text, False, False, True)
                self.per_line[f.rel] = shell_assignments(self.lines[f.rel])
        for rel, callers in ctx.callers.items():
            if rel in self.sourced:
                for caller in callers.scripts - callers.run_by:
                    if caller in self.sourced:
                        self.sourced[caller].add(rel)
        self.assigned_cache: dict[str, set[str]] = {}

    def assigned_anywhere(self, rel: str) -> set[str]:
        return set().union(*self.per_line[rel]) if self.per_line[rel] else set()

    def inherited(self, rel: str, seen: frozenset[str] = frozenset()) -> set[str]:
        """The names the files `rel` sources, transitively, assign."""
        names: set[str] = set()
        for other in self.sourced[rel] - seen:
            names |= self.assigned_anywhere(other) | self.inherited(other, seen | {rel, other})
        return names

    def consumer_names(self, rel: str) -> set[str]:
        """The names every file that sources `rel` assigns, itself or through
        the other files it sources: a file sourced by others reads what they all
        set, and a name some consumer leaves to the environment is a knob."""
        callers = self.ctx.callers[rel]
        sets = [
            self.assigned_anywhere(c) | self.inherited(c, frozenset({rel}))
            for c in callers.scripts - callers.run_by
            if c in self.sourced
        ]
        return set.intersection(*sets) if sets else set()


def shell_env_reads(f: Script, state: ShellAssignments) -> Iterator[tuple[int, str]]:
    """(line, name) of each read of an upper-case name nothing assigns. An
    operator expansion reads a name unless an earlier line assigns it; a bare
    one unless the file assigns it anywhere. The names a sourced file assigns,
    and the ones every file that sources this one assigns, are assigned."""
    given = state.inherited(f.rel) | state.consumer_names(f.rel)
    everywhere = state.assigned_anywhere(f.rel) | given
    before = set(given)
    for number, line in enumerate(state.lines[f.rel], 1):
        for name, bare in shell_reads(line):
            if ENV_NAME_RE.fullmatch(name) and name not in (everywhere if bare else before):
                yield number, name
        for name in printenv_reads(line):
            yield number, name
        before |= state.per_line[f.rel][number - 1]


# ---- python --------------------------------------------------------------


def environ_names(tree: ast.Module) -> set[str]:
    """The names bound to `os.environ`: `from os import environ [as x]` and
    `x = os.environ`."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            names.update(a.asname or a.name for a in node.names if a.name == "environ")
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def getenv_names(tree: ast.Module) -> set[str]:
    """The names `getenv` is called by: itself, and `from os import getenv as x`."""
    names = {"getenv"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "os":
            names.update(a.asname or a.name for a in node.names if a.name == "getenv")
    return names


def call_key(node: ast.Call) -> ast.expr | None:
    """The key a call looks up: its first argument, else its `key=` keyword."""
    if node.args:
        return node.args[0]
    return next((k.value for k in node.keywords if k.arg == "key"), None)


def literal_key(key: ast.expr | None) -> str:
    return key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else UNKNOWN_NAME


def python_env_reads(text: str) -> Iterator[tuple[int, str]]:
    """(line, name) of each read of the environment by name: a subscript, `get`,
    `setdefault`, `pop`, `in` and `getenv`. A key that is not a string literal is
    a read of `?`."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return
    aliases = environ_names(tree)
    getenvs = getenv_names(tree)

    def is_environ(node: ast.expr) -> bool:
        return (isinstance(node, ast.Attribute) and node.attr == "environ") or (
            isinstance(node, ast.Name) and node.id in aliases
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and is_environ(node.value) and isinstance(node.ctx, ast.Load):
            yield node.lineno, literal_key(node.slice)
        elif isinstance(node, ast.Call) and call_key(node) is not None:
            func = node.func
            on_environ = isinstance(func, ast.Attribute) and func.attr in ("get", "setdefault", "pop") and is_environ(func.value)
            getenv = (isinstance(func, ast.Attribute) and func.attr == "getenv") or (
                isinstance(func, ast.Name) and func.id in getenvs
            )
            if on_environ or getenv:
                yield node.lineno, literal_key(call_key(node))
        elif isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.In, ast.NotIn)) and is_environ(node.comparators[0]):
            yield node.lineno, literal_key(node.left)


# ---- rule ----------------------------------------------------------------


def rule_env_knob(ctx: Context) -> Iterator[Violation]:
    state = ShellAssignments(ctx)
    for f in ctx.files:
        if f.rel.endswith(".sh"):
            reads = shell_env_reads(f, state)
        elif f.rel.endswith(".py"):
            reads = python_env_reads(f.text)
        else:
            continue
        for number, name in reads:
            if name == UNKNOWN_NAME or (ENV_NAME_RE.fullmatch(name) and name not in SYSTEM_ENV):
                yield Violation("env-knob", f.rel, number, f"reads the environment variable {name}", key=name)
