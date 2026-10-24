"""Structural rules of the scripts gate: recall tests, lib/, modules, shebang, exec bit, dead paths.

Module of: report/debt_report.py

Each rule is a function over the `Context` that yields violations; the
registry is `STRUCTURE_RULES`. A rule here reads how files relate to one
another (who runs whom, which recall test covers which mode) and not what one
file's header says.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

from scripts_gate_files import Script, Violation, first_token, git, git_ignored, named_file, scripts_tree
from scripts_gate_rules import Context
from scripts_gate_text import clean_shell, command_words, make_rules, peel_substitutions, split_commands

CI_TARGET = "ci"
DEFAULT_MODE = "default"
MODE_RE = re.compile(r"--?[A-Za-z][\w-]*|[a-z][a-z0-9_-]*")
MAKE_CALL_RE = re.compile(r"\$[({]MAKE[)}]([^;&|]*)")
ADVISORY_RE = re.compile(r"\|\|\s*true\s*$")
SHEBANGS = {".sh": "#!/usr/bin/env bash", ".py": "#!/usr/bin/env python3"}
LIB_PREFIX = "lib/"
LIB_CONSUMERS_MIN = 2
SELFTEST_SUPPORT = "lib/selftest_support.sh"
DEAD_TOKEN_RE = re.compile(r"(?<![\w./-])scripts/([\w./-]*)")
PATTERN_FOLLOWERS = frozenset("*<{$([?\\+|")
DOC_ROOTS = ("CLAUDE.md", "CONTRIBUTING.md", "README.md", "Makefile", ".pre-commit-config.yaml")
OUTSIDE = "../"


def is_recall_test(f: Script, ctx: Context) -> bool:
    return bool(named_file(f, "Recall test for", ctx.rels))


def make_closure(ctx: Context) -> dict[str, list[str]]:
    """Target -> recipe lines, for every target `make ci` reaches: through its
    prerequisites and through `$(MAKE) <target>` lines, recursively."""
    recipes, prerequisites = make_rules(ctx.makefile)
    out: dict[str, list[str]] = {}
    pending = [CI_TARGET] if CI_TARGET in recipes or CI_TARGET in prerequisites else []
    while pending:
        target = pending.pop()
        if target in out:
            continue
        out[target] = recipes.get(target, [])
        pending += prerequisites.get(target, [])
        for line in out[target]:
            for found in MAKE_CALL_RE.finditer(line):
                pending += [w for w in found.group(1).split() if not w.startswith("-") and "=" not in w]
    return out


def recipe_runs(ctx: Context, line: str) -> list[tuple[str, str]]:
    """(file, mode) for each scripts/ file a recipe line runs. The mode is the
    first argument when it is a bare flag or word, else the default; a line
    that ends in `|| true` gates nothing and runs nothing here."""
    text = line.lstrip("@-+ \t")
    if ADVISORY_RE.search(text):
        return []
    out = []
    for logical in clean_shell(text, False, True):
        for piece in peel_substitutions(logical):
            for command in split_commands(piece):
                run, _source, every, _assigned = command_words(command)
                for word in run:
                    for f in ctx.resolver.find(word, True):
                        args = every[every.index(word) + 1 :] if word in every else []
                        mode = args[0] if args and MODE_RE.fullmatch(args[0]) else DEFAULT_MODE
                        out.append((f.rel, mode))
    return out


def closure_runs(ctx: Context) -> dict[str, dict[str, set[str]]]:
    """File -> mode -> the make targets of the `make ci` closure that run it."""
    runs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for target, lines in make_closure(ctx).items():
        for line in lines:
            for rel, mode in recipe_runs(ctx, line):
                runs[rel][mode].add(target)
    return runs


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


def mode_problem(tool: Script, mode: str) -> str:
    if mode == DEFAULT_MODE:
        return ""
    if not MODE_RE.fullmatch(mode):
        return f"`{mode}` is not a mode token"
    return "" if mode in tool.text else f"`{mode}` occurs nowhere in {tool.rel}"


def rule_recall_stale(ctx: Context) -> list[Violation]:
    by_rel = ctx.by_rel
    out = []
    for f in ctx.files:
        for number, tool, modes in recall_claims(f):
            if tool == f.rel:
                out.append(Violation("recall-stale", f.rel, number, "`Recall test for:` names the file itself"))
            elif tool not in by_rel:
                out.append(Violation("recall-stale", f.rel, number, f"names {tool}, which is no file of the tree"))
            else:
                for mode in modes:
                    problem = mode_problem(by_rel[tool], mode)
                    if problem:
                        out.append(Violation("recall-stale", f.rel, number, f"{tool}: {problem}"))
    return out


def covering_tests(ctx: Context) -> dict[tuple[str, str], list[str]]:
    """(tool, mode) -> the recall tests whose valid claim names it."""
    by_rel = ctx.by_rel
    covered: dict[tuple[str, str], list[str]] = defaultdict(list)
    for f in ctx.files:
        for _number, tool, modes in recall_claims(f):
            if tool in by_rel and tool != f.rel:
                for mode in modes:
                    if not mode_problem(by_rel[tool], mode):
                        covered[(tool, mode)].append(f.rel)
    return covered


def rule_recall_missing(ctx: Context) -> list[Violation]:
    runs = closure_runs(ctx)
    covered = covering_tests(ctx)
    by_rel = ctx.by_rel
    out = []
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
            out.append(Violation("recall-missing", tool, 1, detail, key=mode))
    return out


def rule_lib_consumers(ctx: Context) -> list[Violation]:
    out = []
    by_rel = ctx.by_rel
    for f in ctx.files:
        if not f.rel.startswith(LIB_PREFIX) or named_file(f, "Module of", ctx.rels):
            continue
        callers = ctx.callers[f.rel]
        users = {
            s
            for s in callers.scripts
            if f.rel == SELFTEST_SUPPORT or not is_recall_test(by_rel[s], ctx)
        }
        users |= {o for o in callers.others if o.startswith("crates/")}
        if len(users) < LIB_CONSUMERS_MIN:
            detail = f"{len(users)} consumer(s), needs {LIB_CONSUMERS_MIN} or a `Module of:` line"
            out.append(Violation("lib-consumers", f.rel, 1, detail))
    return out


def rule_lib_run(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if f.rel.startswith(LIB_PREFIX):
            for site in sorted(ctx.callers[f.rel].run_by):
                out.append(Violation("lib-run", f.rel, 1, f"lib/ is sourced or imported, never run; run by {site}", key=site))
    return out


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


def rule_module_of(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        declared = f.fields.get("Module of")
        if not declared:
            continue
        tool = first_token(declared[1])
        if tool == f.rel:
            out.append(Violation("module-of", f.rel, declared[0], "`Module of:` names the file itself"))
        elif tool not in ctx.rels:
            out.append(Violation("module-of", f.rel, declared[0], f"names {tool or 'nothing'}, which is no file of the tree"))
        elif f.rel not in reachable(ctx, tool):
            out.append(Violation("module-of", f.rel, declared[0], f"{tool} does not run, source or import it"))
    return out


def must_be_executable(f: Script, ctx: Context) -> bool:
    return f.is_code and f.rel in ctx.entries and not f.rel.startswith(LIB_PREFIX)


def rule_shebang(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if not f.is_code:
            continue
        if must_be_executable(f, ctx):
            want = SHEBANGS[os.path.splitext(f.rel)[1]]
            if f.shebang_line != want:
                have = f.shebang_line or "no shebang"
                out.append(Violation("shebang", f.rel, 1, f"entry point has {have}, expected {want}"))
        elif f.shebang:
            out.append(Violation("shebang", f.rel, 1, "a module or lib/ file carries a shebang"))
    return out


def rule_exec_bit(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        if not f.is_code:
            continue
        if must_be_executable(f, ctx) and not f.executable:
            out.append(Violation("exec-bit", f.rel, 1, "entry point without the exec bit"))
        elif f.executable and not must_be_executable(f, ctx):
            out.append(Violation("exec-bit", f.rel, 1, "a module or lib/ file with the exec bit"))
    return out


def rule_orphan(ctx: Context) -> list[Violation]:
    out = []
    for f in ctx.files:
        unused = not ctx.callers[f.rel] and not f.shebang and not named_file(f, "Module of", ctx.rels)
        if f.is_code and unused and not f.rel.startswith(LIB_PREFIX):
            out.append(Violation("orphan", f.rel, 1, "no shebang and nothing runs, sources or imports it"))
    return out


def dead_tokens(text: str, files: set[str], dirs: set[str]) -> list[tuple[int, str]]:
    """The `scripts/<path>` tokens of `text` that name no file or directory of
    the tree. A token followed by a pattern character is a template."""
    out = []
    for number, line in enumerate(text.split("\n"), 1):
        for found in DEAD_TOKEN_RE.finditer(line):
            token = found.group(1).rstrip(".")
            follower = line[found.end() : found.end() + 1]
            if not token or (follower and follower in PATTERN_FOLLOWERS):
                continue
            if token not in files and token.rstrip("/") not in dirs:
                out.append((number, token))
    return out


def dead_path_sources(ctx: Context) -> list[tuple[str, str]]:
    """(path as printed, text): the scripts that are not recall tests, then the
    Makefile, the pre-commit config, the guides, the workflows and the docs."""
    out = [(f.rel, f.text) for f in ctx.files if not is_recall_test(f, ctx)]
    listing = git(ctx.root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "docs", ".github")
    names = [n for n in listing.split("\0") if n.endswith((".md", ".yml", ".yaml"))]
    for name in (*DOC_ROOTS, *names):
        path = ctx.root / name
        if path.is_file():
            out.append((OUTSIDE + name, path.read_text(encoding="utf-8", errors="replace")))
    return out


def rule_dead_path(ctx: Context) -> list[Violation]:
    files = scripts_tree(ctx.root)
    dirs = {"/".join(p.split("/")[:k]) for p in files for k in range(1, p.count("/") + 1)}
    found: list[tuple[str, int, str]] = []
    for path, text in dead_path_sources(ctx):
        found += [(path, number, token) for number, token in dead_tokens(text, files, dirs)]
    for crate, literals in ctx.external.crates.items():
        found += [
            (OUTSIDE + crate, lit.line, lit.rel)
            for lit in literals
            if lit.rel not in files and lit.rel.rstrip("/") not in dirs
        ]
    ignored = git_ignored(ctx.root, [f"scripts/{token}" for _p, _n, token in found])
    return [
        Violation("dead-path", path, number, f"scripts/{token} names no file", key=token)
        for path, number, token in found
        if f"scripts/{token}" not in ignored
    ]


STRUCTURE_RULES = (
    ("recall-missing", rule_recall_missing),
    ("recall-stale", rule_recall_stale),
    ("lib-consumers", rule_lib_consumers),
    ("lib-run", rule_lib_run),
    ("module-of", rule_module_of),
    ("shebang", rule_shebang),
    ("exec-bit", rule_exec_bit),
    ("orphan", rule_orphan),
    ("dead-path", rule_dead_path),
)
