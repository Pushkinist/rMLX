"""Structural rules of the scripts gate: lib/, modules, shebang, exec bit, orphans, dead paths.

Module of: report/debt_report.py

Each rule is a function over the `Context` that yields violations; the
registry is `STRUCTURE_RULES`. A rule here reads how files relate to one
another (who runs, sources or imports whom) and not what one file's header says.
"""

from __future__ import annotations

import os
import re

from scripts_gate_files import Script, Violation, first_token, git, git_ignored, named_file, scripts_tree
from scripts_gate_recall import is_recall_test, reachable
from scripts_gate_rules import Context
from scripts_gate_text import starts_a_token

SHEBANGS = {".sh": "#!/usr/bin/env bash", ".py": "#!/usr/bin/env python3"}
LIB_PREFIX = "lib/"
LIB_CONSUMERS_MIN = 2
SELFTEST_SUPPORT = "lib/selftest_support.sh"
DEAD_TOKEN_RE = re.compile(r"scripts/([\w./-]*)")
PATTERN_FOLLOWERS = frozenset("*<{$([?\\+|")
DOC_ROOTS = ("CLAUDE.md", "CONTRIBUTING.md", "README.md", "Makefile", ".pre-commit-config.yaml", "scripts/README.md")
OUTSIDE = "../"


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
            if not starts_a_token(line, found.start()):
                continue
            token = found.group(1).rstrip(".")
            follower = line[found.end() : found.end() + 1]
            if not token or (follower and follower in PATTERN_FOLLOWERS):
                continue
            if token not in files and token.rstrip("/") not in dirs:
                out.append((number, token))
    return out


def dead_path_sources(ctx: Context) -> list[tuple[str, str]]:
    """(path as printed, text): the scripts that are not recall tests or the
    case runner (both plant trees), then the Makefile, the pre-commit config,
    the guides, the README of scripts/, the workflows and the docs."""
    out = [(f.rel, f.text) for f in ctx.files if not is_recall_test(f, ctx) and f.rel != SELFTEST_SUPPORT]
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
    ("lib-consumers", rule_lib_consumers),
    ("lib-run", rule_lib_run),
    ("module-of", rule_module_of),
    ("shebang", rule_shebang),
    ("exec-bit", rule_exec_bit),
    ("orphan", rule_orphan),
    ("dead-path", rule_dead_path),
)
