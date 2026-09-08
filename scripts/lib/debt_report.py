"""scripts/lib/debt_report.py — advisory technical-debt report.

Four sections, in order: sibling-file/fn similarity ("twins"), debt counters,
the add/remove line ratio since the last tag, and oversized docs. Reads only
files under ``--root`` (default: the repo this file lives in) plus ``git log``
against that same working tree — no network, no other machine path.

Deterministic for a given tree: every collection is name-sorted before it is
printed, and the "twin" measure is the normalised-diff idea that found the
rotor/iso/turbo pairs by hand, generalised from those specific spellings to
any digit run touching a letter, underscore or hyphen.

Advisory: this module never raises for anything short of a caller error (a
``--root`` that does not exist). ``scripts/debt_report.sh`` is the wrapper
that also guarantees exit 0 for `make debt-report`.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SIM_THRESHOLD = 0.60
DOC_SIZE_THRESHOLD_KB = 200
LOC_THRESHOLD = 1000

SIBLING_DIRS = ("crates/rmlx-kv-quant", "crates/rmlx-models")
SPEC_DIR = "crates/rmlx-models/src/speculative"
WORKSPACE_SOURCE_DIR = "crates"

# The same rule scripts/check_spec_sampling.sh uses to find a round-loop
# driver — generalised from "pub fn" to any visibility, which is what
# surfaces the private cached loops a pub-only list misses. Not a name list:
# a driver is whatever currently carries this parameter, in the tree, today.
DRIVER_SIGNATURE_MARKER = "step_fn: &mut dyn FnMut(&ProbeStep)"

DEBT_COMMENT_RE = re.compile(
    r"\b(inert|dormant|deferred|kept for|future-reference|no longer)\b",
    re.IGNORECASE,
)

# A digit run is folded to a single 'N' whenever it touches an identifier
# character on either side — `rotor3` / `rotor4` -> `rotorN`, `K3` -> `KN`,
# `3u8` -> `Nu8`, `4-bit` -> `N-bit`. A bare number with no adjacent letter
# (a line count in a comment, say) is left alone.
_DIGIT_RUN = re.compile(r"(?<=[A-Za-z_])\d+|\d+(?=[A-Za-z_-])")

_FN_SIG = re.compile(
    r"^[ \t]*(?:pub(?:\([\w:]+\))?\s+)?"
    r"(?:default\s+)?(?:async\s+)?(?:const\s+)?(?:unsafe\s+)?"
    r"(?:extern\s+\"[^\"]*\"\s+)?fn\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)

# `r"..."`, `r#"..."#`, `r##"..."##`, and the byte-string forms `br"..."` /
# `br#"..."#`. A raw string does not process `\` as an escape, so the plain
# `"..."` scanner below would desync on one containing a literal backslash,
# and does not stop at the first bare `"` when hashes are in play.
_RAW_STRING_OPEN = re.compile(r'b?r(?P<hashes>#*)"')


def normalize(text: str) -> str:
    return _DIGIT_RUN.sub("N", text)


def is_test_path(path: Path) -> bool:
    parts = path.parts
    if "tests" in parts:
        return True
    name = path.name
    return name == "tests.rs" or name.endswith("_tests.rs")


def rust_files(root: Path, subdir: str, include_tests: bool) -> list[Path]:
    base = root / subdir
    if not base.is_dir():
        return []
    out = [p for p in base.rglob("*.rs")]
    if not include_tests:
        out = [p for p in out if not is_test_path(p.relative_to(root))]
    return sorted(out)


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


def _scan_after_name(text: str, start: int) -> tuple[str, tuple[int, int] | None]:
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
        sig, body_span = _scan_after_name(text, m.end())
        if body_span is None:
            skipped += 1
            continue
        start, end = body_span
        line = text.count("\n", 0, m.start()) + 1
        fns.append(FnInfo(name=m.group("name"), line=line, signature=sig, body=text[start:end]))
    return fns, skipped


def fns_in_file(root: Path, path: Path) -> tuple[list[FnInfo], int]:
    text = path.read_text(errors="ignore")
    fns, skipped = extract_fns(text)
    rel = str(path.relative_to(root))
    for fn in fns:
        fn.file = rel
    return fns, skipped


# ---- section 1: sibling similarity ("twins") --------------------------------


def file_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Group non-test .rs files under SIBLING_DIRS by digit-stripped stem
    (directory + filename with every digit removed); every pair sharing a
    stem is a candidate twin. Stripping rather than folding to a placeholder
    also pairs a bare name against its explicitly-numbered sibling — e.g.
    `quant_iso_v.rs` (the 3-bit variant, unmarked) against `quant_iso_v4.rs`
    — which a same-shape digit placeholder would miss."""
    groups: dict[tuple[str, str], list[Path]] = {}
    for subdir in SIBLING_DIRS:
        for f in rust_files(root, subdir, include_tests=False):
            key = (str(f.parent), re.sub(r"\d+", "", f.stem))
            groups.setdefault(key, []).append(f)
    pairs = []
    for files in groups.values():
        if len(files) < 2:
            continue
        pairs.extend(itertools.combinations(sorted(files), 2))
    return sorted(pairs)


def similarity(a: str, b: str) -> float:
    """Line-level `difflib.SequenceMatcher` ratio over digit-folded text — no
    other normalisation (whitespace, comments and identifiers otherwise
    count). `autojunk=False`: the default heuristic discounts a line that
    recurs often in `a`, which for source text is common lines like `}` or
    `Ok(())`, not noise — with it on, two files can share a run of those and
    still under-measure."""
    a_lines = normalize(a).splitlines()
    b_lines = normalize(b).splitlines()
    return difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False).ratio()


@dataclass
class DriverGroup:
    drivers: list[FnInfo] = field(default_factory=list)
    skipped: int = 0


def discover_drivers(root: Path) -> DriverGroup:
    group = DriverGroup()
    for f in rust_files(root, SPEC_DIR, include_tests=False):
        fns, skipped = fns_in_file(root, f)
        group.skipped += skipped
        group.drivers.extend(fn for fn in fns if DRIVER_SIGNATURE_MARKER in fn.signature)
    group.drivers.sort(key=lambda fn: (fn.file, fn.line))
    return group


def report_sibling_similarity(root: Path, lines: list[str]) -> None:
    lines.append("=== sibling similarity (twins) ===")
    lines.append(
        "  measure: line-level difflib.SequenceMatcher ratio over digit-folded "
        "text (autojunk off); no other normalisation"
    )

    # Named group: whichever functions under SPEC_DIR currently carry the
    # round-loop driver signature. Always printed, not gated on the
    # similarity threshold — this is the standing violation the twin rule
    # names, not a candidate. Discovered, not a literal name list: a name
    # list is itself a second producer of the same fact
    # scripts/check_spec_sampling.sh already derives from the tree.
    lines.append("")
    lines.append(f"--- round-loop drivers ({SPEC_DIR}) ---")
    lines.append(f"  a driver is any fn whose signature contains `{DRIVER_SIGNATURE_MARKER}`")
    group = discover_drivers(root)
    lines.append(f"  {len(group.drivers)} driver(s) found")
    if group.skipped:
        lines.append(f"  ({group.skipped} fn declaration(s) skipped: body-less or unscannable)")
    if len(group.drivers) < 2:
        lines.append("  no round-loop drivers group (fewer than two found)")
    else:
        for fn in group.drivers:
            lines.append(f"  {fn.name:<32} {fn.file}:{fn.line}")
        lines.append("  pairwise similarity:")
        for a, b in itertools.combinations(group.drivers, 2):
            pct = similarity(a.body, b.body) * 100
            lines.append(
                f"    {a.name}@{a.file}:{a.line} <-> {b.name}@{b.file}:{b.line}: {pct:.1f}%"
            )

    # General file-level twins across the two crates, name-paired by digit
    # normalisation, filtered to >= SIM_THRESHOLD.
    lines.append("")
    lines.append(
        f"--- file pairs >= {int(SIM_THRESHOLD * 100)}% shared "
        f"({', '.join(SIBLING_DIRS)}) ---"
    )
    reported = False
    for a, b in file_pairs(root):
        a_text = a.read_text(errors="ignore")
        b_text = b.read_text(errors="ignore")
        pct = similarity(a_text, b_text) * 100
        if pct / 100 < SIM_THRESHOLD:
            continue
        reported = True
        lines.append(f"  {a.relative_to(root)} <-> {b.relative_to(root)}: {pct:.1f}% shared")
        a_fns, a_skipped = fns_in_file(root, a)
        b_fns, b_skipped = fns_in_file(root, b)
        a_by_norm: dict[str, list[FnInfo]] = {}
        for fn in a_fns:
            a_by_norm.setdefault(normalize(fn.name), []).append(fn)
        b_by_norm: dict[str, list[FnInfo]] = {}
        for fn in b_fns:
            b_by_norm.setdefault(normalize(fn.name), []).append(fn)
        for norm_name in sorted(set(a_by_norm) & set(b_by_norm)):
            for fa in a_by_norm[norm_name]:
                for fb in b_by_norm[norm_name]:
                    fpct = similarity(fa.body, fb.body) * 100
                    if fpct / 100 < SIM_THRESHOLD:
                        continue
                    label = fa.name if fa.name == fb.name else f"{fa.name}/{fb.name}"
                    lines.append(
                        f"    fn {label} ({fa.file}:{fa.line} <-> {fb.file}:{fb.line}): "
                        f"{fpct:.1f}% shared"
                    )
        if a_skipped or b_skipped:
            lines.append(
                f"    ({a_skipped + b_skipped} fn declaration(s) skipped in this pair)"
            )
    if not reported:
        lines.append(f"  no pairs at or above the {int(SIM_THRESHOLD * 100)}% threshold")


# ---- section 2: debt counters ----------------------------------------------


def report_debt_counters(root: Path, lines: list[str]) -> None:
    lines.append("")
    lines.append("=== debt counters ===")

    allow_sites = 0
    for f in rust_files(root, WORKSPACE_SOURCE_DIR, include_tests=False):
        allow_sites += f.read_text(errors="ignore").count("#[allow(")
    lines.append(f"#[allow(...)] sites (non-test source only): {allow_sites}")

    debt_comments = 0
    for f in rust_files(root, WORKSPACE_SOURCE_DIR, include_tests=True):
        for line in f.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("//") and DEBT_COMMENT_RE.search(stripped):
                debt_comments += 1
    lines.append(
        "debt-marker comments (inert|dormant|deferred|kept for|future-reference|"
        f"no longer; source + tests): {debt_comments}"
    )

    makefile = root / "Makefile"
    check_targets = 0
    if makefile.is_file():
        text = makefile.read_text(errors="ignore")
        check_targets = len(re.findall(r"^check-[\w-]+:", text, re.MULTILINE))
    lines.append(f"check-* Make targets: {check_targets}")

    oversized = 0
    for f in rust_files(root, WORKSPACE_SOURCE_DIR, include_tests=False):
        text = f.read_text(errors="ignore")
        if len(text.splitlines()) > LOC_THRESHOLD and "LOC-exempt" not in text:
            oversized += 1
    lines.append(
        f"files >{LOC_THRESHOLD} LOC without a LOC-exempt marker (non-test source only): "
        f"{oversized}"
    )

    lines.append(
        "dead-path (zero non-test callers): not attempted — a text-grep call "
        "count cannot tell a real zero-caller function from one reached "
        "through a trait object, a macro, or FFI, and a wrong answer here is "
        "worse than none; that needs a compiler-level call graph, which is "
        "out of scope for a shell+stdlib-python script"
    )


# ---- section 3: churn since the last tag ------------------------------------


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def last_tag(root: Path) -> str | None:
    try:
        return git(root, "describe", "--tags", "--abbrev=0").strip() or None
    except subprocess.CalledProcessError:
        return None


def shortstat_totals(
    root: Path, rev_range: str, pathspecs: list[str]
) -> tuple[int, int] | str:
    """`(insertions, deletions)` summed over every commit in `rev_range` that
    touched `pathspecs` — a line touched twice counts twice; this is churn,
    not the net diff between the two endpoints, and the two numbers disagree
    on a tree with any reverted or re-edited line. On any git failure (no
    such ref, not a repository, …) returns the first line of stderr instead
    of a silent `(0, 0)` — the caller must not print a made-up zero."""
    proc = subprocess.run(
        ["git", "-C", str(root), "log", "--shortstat", "--pretty=format:", rev_range, "--", *pathspecs],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err_lines = proc.stderr.strip().splitlines()
        return err_lines[0] if err_lines else f"git log exited {proc.returncode}"
    ins = dels = 0
    for line in proc.stdout.splitlines():
        m = re.search(r"(\d+) insertion", line)
        if m:
            ins += int(m.group(1))
        m = re.search(r"(\d+) deletion", line)
        if m:
            dels += int(m.group(1))
    return ins, dels


def format_ratio(result: tuple[int, int] | str) -> str:
    if isinstance(result, str):
        return f"unavailable ({result})"
    ins, dels = result
    if dels == 0:
        ratio = "no removals" if ins else "no changes"
    else:
        ratio = f"{ins / dels:.2f}x"
    return f"+{ins} / -{dels}  (ratio {ratio})"


def report_add_remove_ratio(root: Path, since: str | None, lines: list[str]) -> None:
    ref = since or last_tag(root)
    lines.append("")
    if ref is None:
        lines.append("=== churn (summed over commits): no tag found, showing full history ===")
        rev_range = "HEAD"
    else:
        lines.append(f"=== churn (summed over commits) since {ref} ===")
        rev_range = f"{ref}..HEAD"
    lines.append(f"source (crates/): {format_ratio(shortstat_totals(root, rev_range, ['crates']))}")
    # One call with both pathspecs, not two summed — a doc under docs/ also
    # matches the bare "*.md" pathspec (git's wildcard crosses "/"), and
    # querying them separately would double-count it.
    lines.append(
        f"docs (docs/, *.md): {format_ratio(shortstat_totals(root, rev_range, ['docs', '*.md']))}"
    )


# ---- section 4: doc sizes ---------------------------------------------------


def report_doc_sizes(root: Path, lines: list[str]) -> None:
    lines.append("")
    lines.append(f"=== docs over {DOC_SIZE_THRESHOLD_KB} KB ===")
    docs_dir = root / "docs"
    over = []
    if docs_dir.is_dir():
        for f in sorted(docs_dir.glob("*.md")):
            size_kb = f.stat().st_size / 1024
            if size_kb > DOC_SIZE_THRESHOLD_KB:
                over.append((f.name, size_kb))
    if not over:
        lines.append(f"  no docs/*.md file exceeds the {DOC_SIZE_THRESHOLD_KB} KB threshold")
    for name, size_kb in over:
        lines.append(f"  docs/{name}  {size_kb:.1f} KB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root to scan (default: the repo this file lives in)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="git ref for the churn section (default: last tag)",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"debt-report: --root {root} is not a directory", file=sys.stderr)
        return 1

    lines: list[str] = []
    report_sibling_similarity(root, lines)
    report_debt_counters(root, lines)
    report_add_remove_ratio(root, args.since, lines)
    report_doc_sizes(root, lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
