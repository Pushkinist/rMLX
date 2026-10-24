"""The per-file rules of the scripts gate and their registry.

Module of: report/debt_report.py

Each rule is a function over the `Context` that yields violations. A rule
function is registered in `RULE_FUNCTIONS`; the gate runs them in that order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts_gate_callers import Callers, Resolver
from scripts_gate_external import External
from scripts_gate_files import Script, Violation, named_file
from scripts_gate_text import declared_exit_codes, python_literal_exits, shell_literal_exits

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
PURPOSE_MAX = 100


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
    makefile: str
    external: External
    resolver: Resolver

    @property
    def rels(self) -> set[str]:
        return {f.rel for f in self.files}

    @property
    def by_rel(self) -> dict[str, Script]:
        return {f.rel: f for f in self.files}


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
