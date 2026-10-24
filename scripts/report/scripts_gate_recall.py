"""Recall tests of the scripts gate: what `make ci` runs, who covers it, what a line silences.

Module of: report/debt_report.py

Each rule is a function over the `Context` that yields violations; the
registry is `RECALL_RULES`. The run of `make ci` is the closure of the `ci`
target (section 9 of scripts/README.md); a claim in a `Recall test for:` field
holds only when the test runs the tool and the mode token it names occurs in the
tool and in the test's own code.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Iterator
from typing import TYPE_CHECKING

from scripts_gate_files import Script, Unavailable, Violation
from scripts_gate_text import (
    clean_shell,
    command_words,
    make_rules,
    make_variables,
    peel_substitutions,
    split_commands,
)

if TYPE_CHECKING:
    from scripts_gate_rules import Context

CI_TARGET = "ci"
DEFAULT_MODE = "default"
OPTION_RE = re.compile(r"--?[A-Za-z][\w-]*(?:=\S*)?")
WORD_RE = re.compile(r"[a-z][a-z0-9_-]*")
MAKE_CALL_RE = re.compile(r"\$[({]MAKE[)}]([^;&|]*)")
VARIABLE_REFERENCE_RE = re.compile(r"\$[({]([A-Za-z_][A-Za-z0-9_]*)[)}]")
DISCARDS_EXIT_RE = re.compile(r"\|\|\s*(?:true|:|exit\s+0|echo|printf)(?![\w-])|;\s*(?:true|:|exit\s+0)\s*$")
MAKE_IGNORE_FLAG = "-"
MAKE_RECIPE_FLAGS = "@-+ \t"


def is_option(word: str) -> bool:
    return word.startswith("-")


def line_mode(args: list[str]) -> str:
    """The mode of a tool's command line: the first argument that is an option
    or a bare word. An option followed by a word that is not an option takes it
    as its value, so neither is a mode and the scan goes on after the value;
    `--x=v` is a mode as written. The first argument that is neither (a path, a
    number, a substitution) leaves the default mode."""
    i = 0
    while i < len(args):
        word = args[i]
        if is_option(word):
            if not OPTION_RE.fullmatch(word):
                return DEFAULT_MODE
            if "=" not in word and i + 1 < len(args) and not is_option(args[i + 1]):
                i += 2
                continue
            return word
        return word if WORD_RE.fullmatch(word) else DEFAULT_MODE
    return DEFAULT_MODE


def is_mode_token(word: str) -> bool:
    return bool(OPTION_RE.fullmatch(word) or WORD_RE.fullmatch(word))


def discards_exit(line: str) -> bool:
    """True when failure of the recipe line is thrown away: make ignores it
    (`-` in front of the command), or the shell does (`|| true`, `|| :`,
    `|| exit 0`, `|| echo ...`, `|| printf ...` anywhere on the line, or a
    trailing `; true`, `; :`, `; exit 0`)."""
    command = line.lstrip(MAKE_RECIPE_FLAGS)
    if MAKE_IGNORE_FLAG in line[: len(line) - len(command)]:
        return True
    return bool(DISCARDS_EXIT_RE.search(" ".join(clean_shell(command, True, True))))


def expand_words(words: list[str], variables: dict[str, str | None], where: str) -> list[str]:
    """`words` with each `$(NAME)` / `${NAME}` word replaced by the words of the
    variable's value. Any other make expansion cannot be followed."""
    out: list[str] = []
    for word in words:
        if "$" not in word:
            out.append(word)
            continue
        found = VARIABLE_REFERENCE_RE.fullmatch(word)
        value = variables.get(found.group(1)) if found else None
        if value is None:
            raise Unavailable(
                f"`{word}` in {where} is a make expansion the gate cannot follow; "
                "name the target, or define the variable once as `NAME := words`"
            )
        out.extend(value.split())
    return out


def make_closure(ctx: Context) -> tuple[dict[str, list[str]], set[str]]:
    """(target -> recipe lines for every target `make ci` reaches, through its
    prerequisites and through `$(MAKE) <target>` lines, recursively; the targets
    reached only through a line that discards its exit code)."""
    recipes, prerequisites = make_rules(ctx.makefile)
    variables = make_variables(ctx.makefile)
    out: dict[str, list[str]] = {}
    seen: set[tuple[str, bool]] = set()
    pending = [(CI_TARGET, False)] if CI_TARGET in recipes or CI_TARGET in prerequisites else []
    while pending:
        target, quiet = pending.pop()
        if (target, quiet) in seen:
            continue
        seen.add((target, quiet))
        out[target] = recipes.get(target, [])
        needs = expand_words(prerequisites.get(target, []), variables, f"the prerequisites of `{target}`")
        pending += [(need, quiet) for need in needs]
        for line in out[target]:
            line_quiet = quiet or discards_exit(line)
            for found in MAKE_CALL_RE.finditer(line):
                words = [w for w in found.group(1).split() if not is_option(w) and "=" not in w]
                called = expand_words(words, variables, f"a `$(MAKE)` line of `{target}`")
                pending += [(name, line_quiet) for name in called]
    loud = {target for target, quiet in seen if not quiet}
    return out, {target for target, quiet in seen if quiet} - loud


def recipe_runs(ctx: Context, line: str) -> list[tuple[str, str]]:
    """(file, mode) for each scripts/ file a recipe line runs."""
    out = []
    for logical in clean_shell(line.lstrip(MAKE_RECIPE_FLAGS), False, True):
        for piece in peel_substitutions(logical):
            for command in split_commands(piece):
                run, _source, every, _assigned = command_words(command)
                for word in run:
                    for f in ctx.resolver.find(word, True):
                        args = every[every.index(word) + 1 :] if word in every else []
                        out.append((f.rel, line_mode(args)))
    return out


def closure_runs(ctx: Context) -> tuple[dict[str, dict[str, set[str]]], set[tuple[str, str]]]:
    """(file -> mode -> the make targets of the `make ci` closure that run it,
    the (file, mode) pairs run on a line that discards its exit code)."""
    runs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    silenced: set[tuple[str, str]] = set()
    lines_by_target, quiet_targets = make_closure(ctx)
    for target, lines in lines_by_target.items():
        for line in lines:
            line_quiet = target in quiet_targets or discards_exit(line)
            for rel, mode in recipe_runs(ctx, line):
                runs[rel][mode].add(target)
                if line_quiet:
                    silenced.add((rel, mode))
    return runs, silenced


def reachable(ctx: Context, start: str) -> set[str]:
    """The files `start` runs, sources or imports, and what those do in turn."""
    callees: dict[str, list[str]] = defaultdict(list)
    for rel, callers in ctx.callers.items():
        for caller in callers.scripts:
            callees[caller].append(rel)
    seen: set[str] = set()
    pending = [start]
    while pending:
        for rel in callees.get(pending.pop(), []):
            if rel not in seen:
                seen.add(rel)
                pending.append(rel)
    return seen


# ---- claims --------------------------------------------------------------


def recall_claims(f: Script) -> list[tuple[int, str, list[str]]]:
    """(line, tool, mode tokens) per line of the `Recall test for:` field; a
    tool with no mode token claims the default mode."""
    declared = f.fields.get("Recall test for")
    if not declared:
        return []
    out = []
    for offset, text in enumerate(declared[1].split("\n")):
        words = text.replace(",", " ").split()
        if words:
            tool = words[0].strip("`").removeprefix("scripts/")
            out.append((declared[0] + offset, tool, words[1:] or [DEFAULT_MODE]))
    return out


def code_text(f: Script) -> str:
    """The code of a file: no comments, no header; string bodies kept. Heredoc
    bodies of a shell file and the module docstring of a Python file are not
    code."""
    if f.rel.endswith(".sh"):
        return "\n".join(clean_shell(f.text, False))
    if f.rel.endswith(".py"):
        try:
            tree = ast.parse(f.text)
        except SyntaxError:
            return ""
        docstring = tree.body[0].value if tree.body and isinstance(tree.body[0], ast.Expr) else None
        return "\n".join(
            n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n is not docstring
        )
    return "\n".join(line for line in f.text.split("\n") if not line.lstrip().startswith("#"))


def mode_problem(tool: Script, test: Script, mode: str) -> str:
    """Why `mode` is not a mode of `tool` that `test` exercises: its option name
    must occur in the tool, and the token as written in the test's code."""
    if mode == DEFAULT_MODE:
        return ""
    if not is_mode_token(mode):
        return f"`{mode}` is not a mode token"
    if mode.split("=", 1)[0] not in tool.text:
        return f"`{mode}` occurs nowhere in {tool.rel}"
    if mode not in code_text(test):
        return f"`{mode}` occurs nowhere in the code of {test.rel}"
    return ""


def judge_claims(ctx: Context, f: Script) -> tuple[list[tuple[str, str]], list[tuple[int, str]]]:
    """((tool, mode) pairs the test's claims hold for, (line, problem) of each
    claim that does not hold). A claim holds when the tool is another file of
    the tree that the test runs, sources or imports, and the mode is one the test
    exercises."""
    cached = ctx.cache.get(("claims", f.rel))
    if cached is not None:
        return cached
    by_rel = ctx.by_rel
    holds: list[tuple[str, str]] = []
    problems: list[tuple[int, str]] = []
    for number, tool, modes in recall_claims(f):
        if tool == f.rel:
            problems.append((number, "`Recall test for:` names the file itself"))
        elif tool not in by_rel:
            problems.append((number, f"names {tool}, which is no file of the tree"))
        elif tool not in reachable(ctx, f.rel):
            problems.append((number, f"names {tool}, which this file does not run, source or import"))
        else:
            for mode in modes:
                problem = mode_problem(by_rel[tool], f, mode)
                if problem:
                    problems.append((number, f"{tool}: {problem}"))
                else:
                    holds.append((tool, mode))
    ctx.cache[("claims", f.rel)] = (holds, problems)
    return holds, problems


def is_recall_test(f: Script, ctx: Context) -> bool:
    """A file with at least one claim that holds. A claim that does not hold
    exempts the file from nothing."""
    return bool(judge_claims(ctx, f)[0])


def covering_tests(ctx: Context) -> dict[tuple[str, str], list[str]]:
    """(tool, mode) -> the recall tests with a claim that holds for it."""
    covered: dict[tuple[str, str], list[str]] = defaultdict(list)
    for f in ctx.files:
        for pair in judge_claims(ctx, f)[0]:
            covered[pair].append(f.rel)
    return covered


# ---- rules ---------------------------------------------------------------


def rule_recall_stale(ctx: Context) -> Iterator[Violation]:
    for f in ctx.files:
        for number, problem in judge_claims(ctx, f)[1]:
            yield Violation("recall-stale", f.rel, number, problem)


def rule_recall_missing(ctx: Context) -> Iterator[Violation]:
    runs, _silenced = closure_runs(ctx)
    covered = covering_tests(ctx)
    by_rel = ctx.by_rel
    for tool in sorted(runs):
        if is_recall_test(by_rel[tool], ctx):
            continue
        for mode in sorted(runs[tool]):
            tests = covered.get((tool, mode), [])
            if any(t in runs for t in tests):
                continue
            if tests:
                detail = f"mode {mode}: its recall test {tests[0]} is not run by `make {CI_TARGET}`"
            else:
                detail = f"mode {mode} is run by `make {CI_TARGET}` and no `Recall test for:` names it"
            yield Violation("recall-missing", tool, 1, detail, key=mode)


def rule_silenced(ctx: Context) -> Iterator[Violation]:
    _runs, silenced = closure_runs(ctx)
    for tool, mode in sorted(silenced):
        detail = f"mode {mode} is run by `make {CI_TARGET}` on a line that throws its exit code away"
        yield Violation("silenced", tool, 1, detail, key=mode)


RECALL_RULES = (
    ("recall-missing", rule_recall_missing),
    ("recall-stale", rule_recall_stale),
    ("silenced", rule_silenced),
)
