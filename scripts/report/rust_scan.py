"""Rust source reader for the report tools: enums, `match` arms and `fn` bodies.

Module of: report/debt_report.py

It blanks comments and literals, finds an `enum` body, walks `match`
expressions arm by arm, and reads the pattern of each `matches!` call and
each `let`. It is not a Rust parser. It reads the shapes this tree writes and
refuses, loudly, on a shape it cannot read back — a scan that silently drops
a site reports a smaller number than the truth, which is the failure mode
every producer here exists to avoid.

`fn` bodies are read by `extract_fns` below, the one scan both producers of
the update-body figure call, so the two cannot drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


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


def block_end(blanked: str, open_at: int) -> int:
    """Index just past the bracket closing the `{`, `(` or `[` at `open_at`."""
    depth = 0
    for i in range(open_at, len(blanked)):
        c = blanked[i]
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ScanError(f"unterminated block opened at line {line_of(blanked, open_at)}")


def enum_body(blanked: str, name: str) -> str:
    """The text between the braces of `enum <name>`."""
    m = re.search(rf"\benum\s+{re.escape(name)}\b[^{{]*{{", blanked)
    if not m:
        raise ScanError(f"enum {name} not found")
    start = m.end() - 1
    end = block_end(blanked, start)
    return blanked[start + 1 : end - 1]


def enum_variants(blanked: str, name: str) -> list[tuple[str, list[str]]]:
    """`(variant, field names)` for every variant of `enum <name>`.

    A tuple variant reports an empty field list — no enum this reads has one,
    and inventing positional names would put a shape in the census that the
    source does not carry.
    """
    body = enum_body(blanked, name)
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
    body: str = ""


@dataclass
class MatchSite:
    line: int
    start: int
    arms: list[MatchArm] = field(default_factory=list)


def match_sites(blanked: str) -> list[MatchSite]:
    """Every `match` expression in the file, with its arm patterns.

    Nested matches are reported too: each is its own site, and an inner one is
    a site a new variant can force a touch on just like an outer one.
    """
    sites: list[MatchSite] = []
    for m in re.finditer(r"\bmatch\b", blanked):
        try:
            brace, end = _arm_block(blanked, m.end())
        except ScanError as exc:
            raise ScanError(f"match at line {line_of(blanked, m.start())}: {exc}") from exc
        site = MatchSite(line=line_of(blanked, m.start()), start=m.start())
        site.arms = _arms(blanked, brace + 1, end - 1)
        sites.append(site)
    return sites


def _top_level_arrow(text: str) -> bool:
    depth = 0
    for i, c in enumerate(text):
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        elif depth == 0 and text.startswith("=>", i):
            return True
    return False


def _arm_block(blanked: str, start: int) -> tuple[int, int]:
    """`(open, end)` of the block that holds the arms of a `match`.

    The scrutinee can hold a block too (a closure body, `unsafe { .. }`), so
    the arm block is the first block outside brackets that holds a top-level
    `=>` or is empty.
    """
    depth = 0
    i = start
    while i < len(blanked):
        c = blanked[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == "{":
            end = block_end(blanked, i)
            inner = blanked[i + 1 : end - 1]
            if depth == 0 and (not inner.strip() or _top_level_arrow(inner)):
                return i, end
            i = end
            continue
        elif c == ";" and depth == 0:
            break
        i += 1
    raise ScanError("no block holds its arms")


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
            arm = MatchArm(pattern=blanked[pat_start:i].strip(), line=line_of(blanked, pat_start))
            arms.append(arm)
            i += 2
            while i < stop and blanked[i].isspace():
                i += 1
            body_start = i
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
            arm.body = blanked[body_start:i]
            while i < stop and blanked[i] in ", \t\r\n":
                i += 1
            pat_start = i
            continue
        i += 1
    return arms


def matches_macros(blanked: str) -> list[MatchSite]:
    """Every `matches!(expr, pattern)` call, as a site whose one arm is the
    pattern, guard included."""
    out: list[MatchSite] = []
    for m in re.finditer(r"\bmatches!\s*[\(\[\{]", blanked):
        open_at = m.end() - 1
        close = block_end(blanked, open_at) - 1
        depth = 0
        for i in range(open_at + 1, close):
            c = blanked[i]
            if c in "{([":
                depth += 1
            elif c in "})]":
                depth -= 1
            elif c == "," and depth == 0:
                pattern = blanked[i + 1 : close].strip().rstrip(",").strip()
                line = line_of(blanked, m.start())
                out.append(MatchSite(line=line, start=m.start(), arms=[MatchArm(pattern=pattern, line=line)]))
                break
        else:
            raise ScanError(f"matches! at line {line_of(blanked, m.start())} has no pattern")
    return out


def let_patterns(blanked: str) -> list[MatchSite]:
    """The pattern of every `let` — plain, `if let`, `while let` and
    `let … else` — as a site whose one arm is the pattern.

    A `let` with no initialiser has no `=` before its `;` and is left out.
    """
    out: list[MatchSite] = []
    for m in re.finditer(r"\blet\b", blanked):
        depth = 0
        for i in range(m.end(), len(blanked)):
            c = blanked[i]
            if c in "{([":
                depth += 1
            elif c in "})]":
                depth -= 1
                if depth < 0:
                    break
            elif depth == 0 and c == ";":
                break
            elif depth == 0 and c == "=" and blanked[i + 1 : i + 2] not in ("=", ">") and blanked[i - 1] not in "=!<>.":
                line = line_of(blanked, m.start())
                pattern = blanked[m.end() : i].strip()
                out.append(MatchSite(line=line, start=m.start(), arms=[MatchArm(pattern=pattern, line=line)]))
                break
    return out


# `r"..."`, `r#"..."#`, `r##"..."##`, and the byte-string forms `br"..."` /
# `br#"..."#`. A raw string does not process `\` as an escape, so the plain
# `"..."` scanner below would desync on one containing a literal backslash,
# and does not stop at the first bare `"` when hashes are in play.
_RAW_STRING_OPEN = re.compile(r'b?r(?P<hashes>#*)"')


_FN_SIG = re.compile(
    r"^[ \t]*(?:pub(?:\([\w:]+\))?\s+)?"
    r"(?:default\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?"
    r"(?:extern\s+\"[^\"]*\"\s+)?fn\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)


def is_test_path(path: Path) -> bool:
    parts = path.parts
    if "tests" in parts:
        return True
    name = path.name
    return name == "tests.rs" or name.endswith("_tests.rs")


def glob_source_files(root: Path, directory: str, glob: str) -> list[Path]:
    """The non-test files of `root/directory` matching `glob`, sorted. Raises
    RuntimeError naming what is missing: the directory, or any file in it. A
    directory that is not there and one whose glob finds nothing are different
    answers."""
    base = root / directory
    if not base.is_dir():
        raise RuntimeError(f"{directory} is not a directory")
    paths = [p for p in sorted(base.glob(glob)) if not is_test_path(p.relative_to(root))]
    if not paths:
        raise RuntimeError(f"{directory}/{glob} matches no file")
    return paths


# ---- fn extraction ----------------------------------------------------------
#
# A single low-level scanner (`_skip_token`) knows how to step over anything
# that is not plain code — a line or block comment, a "..." or raw string, a
# char literal — and both the signature scan and the body scan share it. A
# bug fixed here (a char literal that closes on a brace or a quote, a raw
# string with hashes) is fixed in both places at once, instead of twice or
# once.


def _skip_token(text: str, i: int) -> int:
    """If position `i` opens a comment, string, char, or raw-string literal,
    return the index just past it. Otherwise return `i` unchanged — the
    caller treats `text[i]` as one plain code character."""
    n = len(text)
    if text.startswith("//", i):
        j = text.find("\n", i)
        return n if j == -1 else j
    if text.startswith("/*", i):
        j = text.find("*/", i + 2)
        return n if j == -1 else j + 2
    m = _RAW_STRING_OPEN.match(text, i)
    if m:
        close = '"' + m.group("hashes")
        j = text.find(close, m.end())
        return n if j == -1 else j + len(close)
    c = text[i]
    if c == '"':
        j = i + 1
        while j < n and text[j] != '"':
            j += 2 if text[j] == "\\" else 1
        return min(j + 1, n)
    if c == "'":
        # A char literal ('a', '\n', '{', '"', '\u{2581}') always closes on a
        # `'` at a fixed offset once its own escape is accounted for; a
        # lifetime ('a, 'static) never does. Only consume it as a literal if
        # the expected closing quote is actually there.
        if i + 1 < n and text[i + 1] == "\\":
            if text[i + 2 : i + 4] == "u{":
                close_brace = text.find("}", i + 4)
                j = close_brace + 1 if close_brace != -1 else i + 3
            else:
                j = i + 3
        else:
            j = i + 2
        if j < n and text[j] == "'":
            return j + 1
        return i + 1
    return i


@dataclass
class FnInfo:
    name: str
    line: int
    signature: str
    body: str
    file: str = ""


def scan_after_name(text: str, start: int) -> tuple[str, tuple[int, int] | None]:
    """From just past an `fn NAME`, scan the parameter list, return type and
    where clause. A top-level `;` (paren/bracket depth 0) is a body-less
    declaration — a trait method or an `extern` block signature — and is
    refused rather than let the scan run on to steal the next `fn`'s body. A
    top-level `{` opens the body. Depth tracking is what keeps an array
    type's `[u8; 4]` from reading as the declaration's own terminator."""
    i, n = start, len(text)
    depth = 0
    while i < n:
        j = _skip_token(text, i)
        if j != i:
            i = j
            continue
        c = text[i]
        if c in "([":
            depth += 1
        elif c in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and c == ";":
            return text[start:i], None
        elif depth == 0 and c == "{":
            return text[start:i], (i, _scan_body(text, i))
        i += 1
    return text[start:n], None


def _scan_body(text: str, start: int) -> int:
    """End (exclusive) of the balanced `{ ... }` opened at `start`."""
    i, n = start, len(text)
    depth = 0
    while i < n:
        j = _skip_token(text, i)
        if j != i:
            i = j
            continue
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def extract_fns(text: str) -> tuple[list[FnInfo], int]:
    """Every `fn` in `text` that has a body — keyed by (name, line), so two
    declarations sharing a name (an inherent method and a trait impl of the
    same name, say) are both kept rather than one silently overwriting the
    other. Returns `(fns, skipped)`: `skipped` counts declarations this
    scanner deliberately did not extract (body-less — a trait method or an
    `extern` signature) — a caller-visible count, not a silent drop."""
    fns: list[FnInfo] = []
    skipped = 0
    for m in _FN_SIG.finditer(text):
        sig, body_span = scan_after_name(text, m.end())
        if body_span is None:
            skipped += 1
            continue
        start, end = body_span
        line = text.count("\n", 0, m.start()) + 1
        fns.append(FnInfo(name=m.group("name"), line=line, signature=sig, body=text[start:end]))
    return fns, skipped
