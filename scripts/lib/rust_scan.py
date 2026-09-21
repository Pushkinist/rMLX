"""Enum and `match` reader for the KV structural-metric producers.

It blanks comments and literals, finds an `enum` body, and walks `match`
expressions arm by arm. It is not a Rust parser. It reads the shapes this tree
writes and refuses, loudly, on a shape it cannot read back — a scan that
silently drops a site reports a smaller number than the truth, which is the
failure mode every producer here exists to avoid.

`fn` bodies are **not** read here. `lib/debt_report.py` owns that scan, and
both producers of the update-body figure call it, so the two cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


class ScanError(Exception):
    """The reader reached a shape it cannot read back."""


def blank_text(src: str) -> str:
    """Return `src` with every comment and literal replaced by spaces of the
    same length, so offsets and line numbers are preserved.

    A lifetime (`'a`) is left alone: it has no closing quote and is not a
    literal. A char literal is blanked.
    """
    out = list(src)
    i = 0
    n = len(src)

    def blank(start: int, end: int) -> None:
        for j in range(start, min(end, n)):
            if out[j] != "\n":
                out[j] = " "

    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif c == "/" and i + 1 < n and src[i + 1] == "*":
            depth = 1
            j = i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            if depth:
                raise ScanError("unterminated block comment")
            blank(i, j)
            i = j
        elif c == "r" and i + 1 < n and src[i + 1] in '#"':
            m = re.match(r'r(#*)"', src[i:])
            if not m:
                i += 1
                continue
            close = '"' + m.group(1)
            j = src.find(close, i + m.end())
            if j < 0:
                raise ScanError("unterminated raw string")
            j += len(close)
            blank(i + m.end(), j - len(close))
            i = j
        elif c == '"':
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    break
                j += 1
            if j >= n:
                raise ScanError("unterminated string")
            blank(i + 1, j)
            i = j + 1
        elif c == "'":
            m = re.match(r"'(\\.|[^\\'])'", src[i:])
            if m:
                blank(i + 1, i + m.end() - 1)
                i += m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def line_of(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def block_end(blanked: str, open_brace: int) -> int:
    """Index just past the `}` closing the `{` at `open_brace`."""
    depth = 0
    for i in range(open_brace, len(blanked)):
        c = blanked[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ScanError(f"unterminated block opened at line {line_of(blanked, open_brace)}")


def enum_variants(blanked: str, name: str) -> list[tuple[str, list[str]]]:
    """`(variant, field names)` for every variant of `enum <name>`.

    A tuple variant reports an empty field list — no enum this reads has one,
    and inventing positional names would put a shape in the census that the
    source does not carry.
    """
    m = re.search(rf"\benum\s+{re.escape(name)}\b[^{{]*{{", blanked)
    if not m:
        raise ScanError(f"enum {name} not found")
    start = m.end() - 1
    end = block_end(blanked, start)
    body = blanked[start + 1 : end - 1]
    out: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(body):
        vm = re.compile(r"[A-Z][A-Za-z0-9_]*").search(body, i)
        if not vm:
            break
        after = body[vm.end() :]
        lead = re.match(r"\s*", after).end()
        head = after[lead : lead + 1]
        if head == "{":
            open_at = vm.end() + lead
            close = block_end(body, open_at)
            fields = re.findall(r"(?:^|,)\s*(?:#\[[^\]]*\]\s*)*([a-z_][A-Za-z0-9_]*)\s*:", body[open_at + 1 : close - 1])
            out.append((vm.group(0), fields))
            i = close
        elif head == "(":
            out.append((vm.group(0), []))
            i = vm.end() + lead
        elif head in (",", "}", ""):
            out.append((vm.group(0), []))
            i = vm.end() + lead
        else:
            i = vm.end()
    return out


@dataclass
class MatchArm:
    pattern: str
    line: int


@dataclass
class MatchSite:
    line: int
    arms: list[MatchArm] = field(default_factory=list)


def match_sites(blanked: str) -> list[MatchSite]:
    """Every `match` expression in the file, with its arm patterns.

    Nested matches are reported too: each is its own site, and an inner one is
    a site a new variant can force a touch on just like an outer one.
    """
    sites: list[MatchSite] = []
    for m in re.finditer(r"\bmatch\b", blanked):
        brace = blanked.find("{", m.end())
        if brace < 0:
            raise ScanError(f"match at line {line_of(blanked, m.start())} opens no block")
        try:
            end = block_end(blanked, brace)
        except ScanError as exc:
            raise ScanError(f"match at line {line_of(blanked, m.start())}: {exc}") from exc
        site = MatchSite(line=line_of(blanked, m.start()))
        site.arms = _arms(blanked, brace + 1, end - 1)
        sites.append(site)
    return sites


def _arms(blanked: str, start: int, stop: int) -> list[MatchArm]:
    arms: list[MatchArm] = []
    i = start
    pat_start = start
    depth = 0
    while i < stop:
        c = blanked[i]
        if c in "{([":
            depth += 1
            i += 1
            continue
        if c in "})]":
            depth -= 1
            i += 1
            continue
        if depth == 0 and blanked.startswith("=>", i):
            pattern = blanked[pat_start:i]
            arms.append(MatchArm(pattern=pattern.strip(), line=line_of(blanked, pat_start)))
            i += 2
            while i < stop and blanked[i].isspace():
                i += 1
            if i < stop and blanked[i] == "{":
                i = block_end(blanked, i)
            else:
                d = 0
                while i < stop:
                    if blanked[i] in "{([":
                        d += 1
                    elif blanked[i] in "})]":
                        d -= 1
                    elif blanked[i] == "," and d == 0:
                        break
                    i += 1
            while i < stop and blanked[i] in ", \t\r\n":
                i += 1
            pat_start = i
            continue
        i += 1
    return arms
