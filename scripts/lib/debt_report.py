"""scripts/lib/debt_report.py — advisory technical-debt report.

Four sections, in order: sibling-file/fn similarity ("twins"), debt counters,
the add/remove line ratio since the last tag, and oversized docs. Reads only
files under ``--root`` (default: the repo this file lives in) plus ``git log``
against that same working tree — no network, no other machine path.

``--matched-lines <population>`` is a second, non-advisory mode: prints the
summed pairwise ``difflib`` matched-line count for one named population and
exits, instead of the four-section report. Unlike the report, it exits 1 (not
0) when the population it needs is unavailable — a measurement with no figure
behind it must not read as a passing one, and a population that resolves to
zero members is unavailable, never a ``0`` indistinguishable from a real one.

The populations, each carrying its own root and its own pairing rule (see
``MATCHED_LINES_POPULATIONS``):

* ``drivers`` / ``impls`` — the speculative round-loop drivers and the
  ``impl RoundDrafter`` bodies, one family, so every item pairs with every
  other.
* ``rotor-storage`` — the non-test ``quant_rotor_*.rs`` files under
  ``crates/rmlx-kv-quant/src/storage``, paired inside a group sharing the
  filename stem with every digit run removed (``quant_rotor_v3`` and
  ``quant_rotor_v4`` -> ``quant_rotor_v``): same axis, different width. A
  group of one contributes an item and no pair, so a collapsed axis reads 0
  matched lines with the population still found.
* ``rotor-updates`` — the ``update_rotor*`` fns of
  ``crates/rmlx-kv-quant/src/kvcache/update.rs``, paired by the same
  digit-stripped-name rule (``update_rotor_k_only_3`` and ``_4`` ->
  ``update_rotor_k_only_``).

Neither rotor population is a literal file or fn list: both are a glob plus a
name rule, so the same command measures a tree that still carries the twins
and one that does not.

Deterministic for a given tree: every collection is name-sorted before it is
printed, and the "twin" measure is the normalised-diff idea that found the
rotor/iso/turbo pairs by hand, generalised from those specific spellings to
any digit run touching a letter, underscore or hyphen.

Advisory (the four-section report only): this module never raises for
anything short of a caller error (a ``--root`` that does not exist) — an
unavailable round-loop-driver scan prints ``unavailable (...)`` in its
section rather than aborting the rest of the report. ``scripts/debt_report.sh``
is the wrapper that also guarantees exit 0 for `make debt-report`, and
forwards this module's own exit code for ``--matched-lines``.
"""

from __future__ import annotations

import argparse
import difflib
import itertools
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

SIM_THRESHOLD = 0.60
DOC_SIZE_THRESHOLD_KB = 200
LOC_THRESHOLD = 1000

SIBLING_DIRS = ("crates/rmlx-kv-quant", "crates/rmlx-models")
SPEC_DIR = "crates/rmlx-models/src/speculative"
ROTOR_STORAGE_DIR = "crates/rmlx-kv-quant/src/storage"
ROTOR_STORAGE_GLOB = "quant_rotor_*.rs"
ROTOR_UPDATE_FILE = "crates/rmlx-kv-quant/src/kvcache/update.rs"
ROTOR_UPDATE_FN_PREFIX = "update_rotor"
WORKSPACE_SOURCE_DIR = "crates"
CHECK_SPEC_CHARGE_SCRIPT = Path(__file__).resolve().parents[1] / "check_spec_charge.sh"

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


_IMPL_ROUND_DRAFTER = re.compile(
    r"^impl(?:<[^>{]*>)?\s+RoundDrafter\s+for\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)


def extract_round_drafter_impls(text: str) -> list[FnInfo]:
    """Every `impl RoundDrafter for <Type>` block in `text`, whole-block body
    (every method it implements, not one at a time) — the unit the migration
    chunks measured duplication over once a drafter's round loop collapsed
    into the shared one and its own logic moved into this impl. Reuses
    `_scan_after_name`, the same balanced-brace scanner `extract_fns` uses:
    an impl header carries no `(` or `[` before its opening `{`, so the scan
    that finds a fn's body finds an impl block's just as well."""
    out: list[FnInfo] = []
    for m in _IMPL_ROUND_DRAFTER.finditer(text):
        _sig, body_span = _scan_after_name(text, m.end())
        if body_span is None:
            continue
        start, end = body_span
        line = text.count("\n", 0, m.start()) + 1
        out.append(FnInfo(name=m.group("name"), line=line, signature="", body=text[start:end]))
    return out


def round_drafter_impls(root: Path) -> list[FnInfo]:
    items: list[FnInfo] = []
    for f in rust_files(root, SPEC_DIR, include_tests=False):
        text = f.read_text(errors="ignore")
        rel = str(f.relative_to(root))
        for impl in extract_round_drafter_impls(text):
            impl.file = rel
            items.append(impl)
    items.sort(key=lambda fn: (fn.file, fn.line))
    return items


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


def matched_lines(a: str, b: str) -> int:
    """Summed size of every matching block `difflib.SequenceMatcher` finds
    between `a` and `b`, over the same digit-folded, `autojunk=False` lines
    `similarity()` compares — a line count, not a ratio. This is the figure
    the round-loop migration chunks reported by hand; `--matched-lines` is
    its one producer now."""
    a_lines = normalize(a).splitlines()
    b_lines = normalize(b).splitlines()
    sm = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    return sum(block.size for block in sm.get_matching_blocks())


@dataclass
class DriverGroup:
    drivers: list[FnInfo] = field(default_factory=list)
    skipped: int = 0
    # Gate-listed (file, fn, line) triples this file's own fn scan of SPEC_DIR
    # could not resolve — dropped silently before this field existed. A file
    # the gate can reach and this scan cannot (moved under a `tests/`
    # subdirectory, say — `rust_files()` excludes any `tests` path component,
    # the gate excludes only by filename) lands here, not out of the report.
    unresolved: list[tuple[str, str, int]] = field(default_factory=list)


def list_driver_population(root: Path) -> list[tuple[str, str, int]]:
    """``(file, fn, line)`` triples for population (a) — round loops —
    exactly as ``check_spec_charge.sh``'s ``check-spec-charge`` gate derives
    them: the driver signature plus a constructed ``RoundTotals``. Runs the
    real gate script in ``--list-drivers`` mode with ``--root`` pointed at
    ``root`` — the same flag its own fixture recall test uses to aim the gate
    at a synthetic tree — so this is the population's one producer rather
    than a second copy of its rule. The line number is part of the join key:
    file and fn name alone collide when two same-named fns live in one file.

    Raises ``RuntimeError`` rather than returning an empty list for every
    failure mode, including "no speculative source directory" (the tree
    genuinely has none) and "found no round loop" (the gate scanned and the
    population came back empty) — both read as ``unavailable`` upstream, not
    as a silent ``0 driver(s) found`` indistinguishable from a real answer."""
    if not CHECK_SPEC_CHARGE_SCRIPT.is_file():
        raise RuntimeError(f"check_spec_charge.sh not found at {CHECK_SPEC_CHARGE_SCRIPT}")
    proc = subprocess.run(
        ["bash", str(CHECK_SPEC_CHARGE_SCRIPT), "--list-drivers", "--root", str(root)],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 2 and "no speculative source directory" in proc.stderr:
        raise RuntimeError("no speculative source directory")
    if proc.returncode != 0:
        raise RuntimeError(
            f"check_spec_charge.sh --list-drivers failed (exit {proc.returncode}): "
            f"{proc.stderr.strip()}"
        )
    triples: list[tuple[str, str, int]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        rel_file, fn_name, fn_line = line.split("\t")
        triples.append((rel_file, fn_name, int(fn_line)))
    return triples


def discover_drivers(root: Path) -> DriverGroup:
    group = DriverGroup()
    fns_by_key: dict[tuple[str, str, int], FnInfo] = {}
    for f in rust_files(root, SPEC_DIR, include_tests=False):
        fns, skipped = fns_in_file(root, f)
        group.skipped += skipped
        for fn in fns:
            fns_by_key[(fn.file, fn.name, fn.line)] = fn
    for rel_file, fn_name, fn_line in list_driver_population(root):
        match = fns_by_key.get((rel_file, fn_name, fn_line))
        if match is None:
            group.unresolved.append((rel_file, fn_name, fn_line))
        else:
            group.drivers.append(match)
    group.drivers.sort(key=lambda fn: (fn.file, fn.line))
    group.unresolved.sort()
    return group


def _driver_items(root: Path) -> list[FnInfo]:
    return discover_drivers(root).drivers


def width_pair_key(item: FnInfo) -> str:
    """Pairing key for the rotor populations: the item's own name with every
    digit run removed, so `quant_rotor_v3` and `quant_rotor_v4` land in the
    group `quant_rotor_v`, and `quant_rotor_k3` in a different one. Stripping
    rather than folding to a placeholder also groups a bare name against its
    explicitly-numbered sibling — the same rule `file_pairs()` uses."""
    return re.sub(r"\d+", "", item.name)


def rotor_storage_items(root: Path) -> list[FnInfo]:
    """Every non-test file matching ROTOR_STORAGE_GLOB under
    ROTOR_STORAGE_DIR, as one item whose "body" is the whole file — a glob
    and a name rule, never a file list, so this reads a tree that carries the
    width twins and one that has collapsed them."""
    base = root / ROTOR_STORAGE_DIR
    if not base.is_dir():
        raise RuntimeError(f"{ROTOR_STORAGE_DIR} is not a directory")
    items: list[FnInfo] = []
    for path in sorted(base.glob(ROTOR_STORAGE_GLOB)):
        rel = path.relative_to(root)
        if is_test_path(rel):
            continue
        items.append(
            FnInfo(
                name=path.stem,
                line=1,
                signature="",
                body=path.read_text(errors="ignore"),
                file=str(rel),
            )
        )
    return items


def rotor_update_items(root: Path) -> list[FnInfo]:
    """Every fn of ROTOR_UPDATE_FILE whose name starts with
    ROTOR_UPDATE_FN_PREFIX, body only — the same `extract_fns` scan the twin
    section uses, so the body is brace to brace and the signature lines above
    it are not counted."""
    path = root / ROTOR_UPDATE_FILE
    if not path.is_file():
        raise RuntimeError(f"{ROTOR_UPDATE_FILE} is not a file")
    fns, _skipped = fns_in_file(root, path)
    return [fn for fn in fns if fn.name.startswith(ROTOR_UPDATE_FN_PREFIX)]


@dataclass(frozen=True)
class Population:
    label: str
    # The population's own root, printed beside its label. Not SPEC_DIR: a
    # figure measured over one directory must not print another's path.
    root: str
    collect: Callable[[Path], list[FnInfo]]
    # None — one family: every item pairs with every other. Otherwise items
    # pair only inside a group sharing this key.
    pair_key: Callable[[FnInfo], str] | None = None


MATCHED_LINES_POPULATIONS = {
    "drivers": Population("round-loop drivers", SPEC_DIR, _driver_items),
    "impls": Population("impl RoundDrafter bodies", SPEC_DIR, round_drafter_impls),
    "rotor-storage": Population(
        "rotor storage twins", ROTOR_STORAGE_DIR, rotor_storage_items, width_pair_key
    ),
    "rotor-updates": Population(
        "rotor update twins", ROTOR_UPDATE_FILE, rotor_update_items, width_pair_key
    ),
}


def population_pairs(
    items: list[FnInfo], pair_key: Callable[[FnInfo], str] | None
) -> list[tuple[FnInfo, FnInfo]]:
    if pair_key is None:
        return list(itertools.combinations(items, 2))
    groups: dict[str, list[FnInfo]] = {}
    for item in items:
        groups.setdefault(pair_key(item), []).append(item)
    pairs: list[tuple[FnInfo, FnInfo]] = []
    for key in sorted(groups):
        pairs.extend(itertools.combinations(groups[key], 2))
    return pairs


def matched_lines_report(root: Path, population: str) -> str:
    """`--matched-lines <population>` — the summed pairwise `matched_lines()`
    over one named population, the way the round-loop migration chunks and
    the rotor collapse reported their duplication figure by hand:
    `<label> (<root>): <matched> matched lines over <body-lines> body lines
    (<n> item(s), <pairs> pair(s))`.

    A population that resolves to zero members raises rather than printing a
    `0` — an empty population and a collapsed one are different answers, and
    only the second is a measurement."""
    pop = MATCHED_LINES_POPULATIONS[population]
    items = pop.collect(root)
    if not items:
        raise RuntimeError(f"{pop.root}: population is empty")
    pairs = population_pairs(items, pop.pair_key)
    total_matched = sum(matched_lines(a.body, b.body) for a, b in pairs)
    total_body_lines = sum(len(item.body.splitlines()) for item in items)
    return (
        f"{pop.label} ({pop.root}): {total_matched} matched lines over {total_body_lines} "
        f"body lines ({len(items)} item(s), {len(pairs)} pair(s))"
    )


def report_sibling_similarity(root: Path, lines: list[str]) -> None:
    lines.append("=== sibling similarity (twins) ===")
    lines.append(
        "  measure: line-level difflib.SequenceMatcher ratio over digit-folded "
        "text (autojunk off); no other normalisation"
    )

    # Named group: population (a) of check-spec-charge — the round loops —
    # read from check_spec_charge.sh --list-drivers, not re-derived here.
    # Always printed, not gated on the similarity threshold — this is the
    # standing violation the twin rule names, not a candidate.
    lines.append("")
    lines.append(f"--- round-loop drivers ({SPEC_DIR}) ---")
    lines.append(
        "  a driver is population (a) of check-spec-charge "
        "(`scripts/check_spec_charge.sh --list-drivers`)"
    )
    try:
        group = discover_drivers(root)
    except RuntimeError as exc:
        # No count line follows: an unavailable scan prints exactly one line
        # and nothing else, so it can never be mistaken for a real "0".
        lines.append(f"  unavailable ({exc})")
    else:
        if group.unresolved:
            listed = len(group.drivers) + len(group.unresolved)
            lines.append(f"  {listed} listed, {len(group.drivers)} resolved")
            for rel_file, fn_name, fn_line in group.unresolved:
                lines.append(f"    unresolved: {fn_name} {rel_file}:{fn_line}")
        else:
            lines.append(f"  {len(group.drivers)} driver(s) found")
        if group.skipped:
            lines.append(f"  ({group.skipped} fn declaration(s) skipped: body-less or unscannable)")
        if not group.drivers:
            lines.append("  no round-loop drivers found")
        else:
            for fn in group.drivers:
                lines.append(f"  {fn.name:<32} {fn.file}:{fn.line}")
            if len(group.drivers) < 2:
                lines.append("  no pairwise similarity (fewer than two drivers)")
            else:
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
    parser.add_argument(
        "--matched-lines",
        choices=sorted(MATCHED_LINES_POPULATIONS),
        default=None,
        help="print the summed pairwise matched-line count for one named "
        "population, over that population's own root, and exit, instead of "
        "the full report",
    )
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not root.is_dir():
        print(f"debt-report: --root {root} is not a directory", file=sys.stderr)
        return 1

    if args.matched_lines:
        try:
            print(matched_lines_report(root, args.matched_lines))
        except RuntimeError as exc:
            print(
                f"debt-report --matched-lines {args.matched_lines}: unavailable ({exc})",
                file=sys.stderr,
            )
            return 1
        return 0

    lines: list[str] = []
    report_sibling_similarity(root, lines)
    report_debt_counters(root, lines)
    report_add_remove_ratio(root, args.since, lines)
    report_doc_sizes(root, lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
