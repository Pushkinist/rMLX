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
* ``rotor-updates`` — every fn of the update files
  (``crates/rmlx-kv-quant/src/kvcache/update*.rs``: the dispatch file plus one
  file per codec family) carrying the ``rotor`` token as a whole segment,
  paired by the same digit-stripped-name rule (``update_rotor_k_only_3`` and
  ``_4`` -> ``update_rotor_k_only_``).
* ``turbo-storage`` — the non-test ``quant_k_turbo*.rs`` files under the same
  storage directory, same digit-stripped-stem pairing.
* ``turbo-updates`` — the fns of the update files carrying the symmetric turbo
  token, spelled ``tsym`` on the decode side and ``turbo_sym`` on the prefill
  side, same pairing. Only the symmetric ones: the same family carries two
  other turbo width pairs that belong to a different collapse.
* ``turbo-ssd`` — the turbo helper fns of
  ``crates/rmlx-kv-ssd/src/block_io.rs``, same pairing. A population of its
  own because its root is a file in another crate.
* ``ssd-hydrate`` — the non-test fns under ``crates/rmlx-models/src`` named
  exactly ``hydrate`` or exactly ``from_hydrated``, one family. Both names
  are needed because one name spans only one tree: before the SSD-hydrate
  collapse the population is the per-arch ``hydrate`` bodies, after it the
  short ``from_hydrated`` constructors. ``hydrate_from_ssd`` is a third name
  and stays out.
* ``update-bodies`` — every ``update_``-prefixed fn of the update files, one
  family. The prefix is the whole rule, so the shared entries the dispatch
  reaches (``update_and_sdpa_*``, ``update_decode_fp16*``,
  ``update_prefill_raw``) are counted beside the per-variant bodies; the label
  says what the prefix finds rather than claiming a narrower population.

No population is a literal file or fn list: each is a glob plus a name rule,
so the same command measures a tree that still carries the twins and one that
does not, and a tree that held every family's update path in one file and one
that split them out. ``ssd-hydrate`` is a glob plus a name rule for the same
reason.

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
import functools
import itertools
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

SIM_THRESHOLD = 0.60
DOC_SIZE_THRESHOLD_KB = 40
LOC_THRESHOLD = 1000

SIBLING_DIRS = ("crates/rmlx-kv-quant", "crates/rmlx-models")
SPEC_DIR = "crates/rmlx-models/src/speculative"
KV_STORAGE_DIR = "crates/rmlx-kv-quant/src/storage"
ROTOR_STORAGE_GLOB = "quant_rotor_*.rs"
ISO_STORAGE_GLOB = "quant_iso_*.rs"
TURBO_STORAGE_GLOB = "quant_k_turbo*.rs"
KV_UPDATE_DIR = "crates/rmlx-kv-quant/src/kvcache"
# The update path is one dispatch file plus one file per codec family. A glob,
# never a file list, so one command measures the tree that held every family in
# `update.rs` and the tree that split them out.
KV_UPDATE_GLOB = "update*.rs"
KV_UPDATE_ROOT = f"{KV_UPDATE_DIR}/{KV_UPDATE_GLOB}"
KV_SSD_BLOCK_IO_FILE = "crates/rmlx-kv-ssd/src/block_io.rs"
# Name patterns, not prefixes: a family is a shape, and a prefix reads only the
# fns that happen to lead with it. Each pattern is the codec's own token,
# matched as a whole segment wherever it sits in the name, so one rule reaches
# a family's entries (`update_rotor_*`), the bodies they enter (`rotor_*_update`,
# `iso_k_only_k_side`) and its prefill bulk-encode bodies (`exit_prefill_rotor*`)
# alike. The anchored `^update_rotor` this replaces could not: it found four fns
# and read a measured 0 whatever the eight rotor prefill bodies beside them
# held, so deleting one of an exact pair moved no figure.
ROTOR_UPDATE_FN_PATTERN = r"(^|_)rotor(\d|_|$)"
ISO_UPDATE_FN_PATTERN = r"(^|_)iso(\d|_|$)"
# The symmetric turbo entries, the bodies they enter and their prefill
# bulk-encode bodies, and nothing else. The same file carries two further width
# pairs — `update_k8vturbo3` / `update_k8vturbo2` and their TCQ siblings — that
# a bare `turbo` pattern would fold in; those are a different family's twins
# and are not what the turbo K-storage collapse removes, so this population
# names only the symmetric token. It has two spellings: the decode side writes
# it `tsym` and the prefill bodies write it `turbo_sym`, so `urbo_` is optional
# and the whole token is matched as a segment rather than as a prefix. The
# collapsed entry (`update_tsym`) and the width-parametric body it enters
# (`tsym_update`) spell it on opposite sides of the name, so an anchored
# `^update_tsym` would see the entry and not the body, and a re-split of that
# body into two width bodies would then be invisible to this counter.
TURBO_UPDATE_FN_PATTERN = r"(^|_)t(urbo_)?sym(\d|_|$)"
# No digit in the pattern: after the collapse the SSD helpers lose their width
# suffix, and a digit-bearing pattern would find nothing and report the
# population unavailable rather than a measured 0.
TURBO_SSD_FN_PATTERN = r"(turbo|tsym)"
# Every `update_`-prefixed fn of the update files, whatever family it belongs
# to. The three family patterns above each read one codec's twins; this one
# reads the whole prefixed population, which is what the "one update body per
# store shape" step has to shrink. The prefix admits the shared entries the
# dispatch reaches as well as the per-variant bodies — `update_and_sdpa_*`,
# `update_decode_fp16*`, `update_prefill_raw` — and they are counted, because
# any rule that dropped them would be a hand-drawn boundary over which body is
# "per-variant" enough, and the label says what the prefix finds.
UPDATE_FN_PATTERN = r"^update_"
MODELS_SOURCE_DIR = "crates/rmlx-models/src"
SSD_HYDRATE_FN_NAMES = ("from_hydrated", "hydrate")
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
    its one producer now.

    Measured both ways round and reported as the larger. `SequenceMatcher` is
    not symmetric: it anchors on the longest match it finds in `a` and can
    reach a smaller total with the arguments swapped. A one-directional
    measure would therefore move when a body changed file, or was renamed, or
    when a population gained an item that re-ordered it — with no line of any
    body changing. The larger of the two is the count of lines the pair
    genuinely shares, and it depends on the pair alone."""
    a_lines = normalize(a).splitlines()
    b_lines = normalize(b).splitlines()
    forward = difflib.SequenceMatcher(None, a_lines, b_lines, autojunk=False)
    reverse = difflib.SequenceMatcher(None, b_lines, a_lines, autojunk=False)
    return max(
        sum(block.size for block in forward.get_matching_blocks()),
        sum(block.size for block in reverse.get_matching_blocks()),
    )


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
    explicitly-numbered sibling — the same rule `file_pairs()` uses.

    A run of `_` or `-` left behind by the stripping collapses to one, so a
    width spelled as its own segment joins the group it belongs to:
    `update_rotor_5_sym` -> `update_rotor__sym` -> `update_rotor_sym`, the
    same group as `update_rotor3_sym`. Without that step the separator the
    width carried would be the only thing keeping the two apart, and the pair
    would go unmeasured with nothing saying so."""
    return re.sub(r"[_-]{2,}", "_", re.sub(r"\d+", "", item.name))


def storage_file_items(root: Path, *, glob: str) -> list[FnInfo]:
    """Every non-test file matching `glob` under KV_STORAGE_DIR, as one item
    whose "body" is the whole file — a glob and a name rule, never a file
    list, so this reads a tree that carries the width twins and one that has
    collapsed them.

    `glob` arrives from the registration site rather than from a field on
    `Population`: the other populations read no glob, and a field none of them
    uses is the shape this module exists to discourage."""
    base = root / KV_STORAGE_DIR
    if not base.is_dir():
        raise RuntimeError(f"{KV_STORAGE_DIR} is not a directory")
    items: list[FnInfo] = []
    for path in sorted(base.glob(glob)):
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


def file_fn_items(root: Path, *, file: str, pattern: str) -> list[FnInfo]:
    """Every fn of `file` whose name matches `pattern` (`re.search`), body only
    — the same `extract_fns` scan the twin section uses, so the body is brace
    to brace and the signature lines above it are not counted.

    `file` and `pattern` arrive from the registration site, for the reason
    [`storage_file_items`] gives for its glob. A pattern rather than a prefix
    because a family is a shape: anchor it (`^update_rotor`) to get prefix
    behaviour, leave it unanchored (`iso`) to name a family whose entries and
    bodies do not share one."""
    path = root / file
    if not path.is_file():
        raise RuntimeError(f"{file} is not a file")
    matches = re.compile(pattern).search
    fns, _skipped = fns_in_file(root, path)
    return [fn for fn in fns if matches(fn.name)]


def glob_fn_items(root: Path, *, directory: str, glob: str, pattern: str) -> list[FnInfo]:
    """Every fn matching `pattern` in every non-test file of `directory` that
    matches `glob`, body only — [`file_fn_items`] widened from one file to a
    glob over a directory.

    `directory`, `glob` and `pattern` arrive from the registration site, for
    the reason [`storage_file_items`] gives for its glob. A glob that matches
    no file raises its own reason rather than an empty population: a root that
    is not there and a root whose files hold no matching fn are different
    answers."""
    base = root / directory
    if not base.is_dir():
        raise RuntimeError(f"{directory} is not a directory")
    paths = [p for p in sorted(base.glob(glob)) if not is_test_path(p.relative_to(root))]
    if not paths:
        raise RuntimeError(f"{directory}/{glob} matches no file")
    matches = re.compile(pattern).search
    items: list[FnInfo] = []
    for path in paths:
        fns, _skipped = fns_in_file(root, path)
        items.extend(fn for fn in fns if matches(fn.name))
    return items


def ssd_hydrate_items(root: Path) -> list[FnInfo]:
    """Every non-test fn under MODELS_SOURCE_DIR whose name is one of
    SSD_HYDRATE_FN_NAMES, body only — a glob plus a name rule, never a file
    list, so one command measures the tree that carries the per-arch hydrate
    bodies and the tree that has collapsed them onto one blanket impl."""
    base = root / MODELS_SOURCE_DIR
    if not base.is_dir():
        raise RuntimeError(f"{MODELS_SOURCE_DIR} is not a directory")
    items: list[FnInfo] = []
    for path in rust_files(root, MODELS_SOURCE_DIR, include_tests=False):
        fns, _skipped = fns_in_file(root, path)
        items.extend(fn for fn in fns if fn.name in SSD_HYDRATE_FN_NAMES)
    return items


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
    "iso-storage": Population(
        "iso storage twins",
        KV_STORAGE_DIR,
        functools.partial(storage_file_items, glob=ISO_STORAGE_GLOB),
        width_pair_key,
    ),
    # Every pair, not `width_pair_key`: the iso update family's live
    # duplication is between same-width bodies of different entries, and a
    # width key puts those in two groups and compares them never — a counter
    # that cannot move off 0 whatever the files hold. The width-twin question
    # for this family is `iso-storage`'s.
    "iso-updates": Population(
        "iso update fns",
        KV_UPDATE_ROOT,
        functools.partial(
            glob_fn_items,
            directory=KV_UPDATE_DIR,
            glob=KV_UPDATE_GLOB,
            pattern=ISO_UPDATE_FN_PATTERN,
        ),
    ),
    "turbo-storage": Population(
        "turbo storage twins",
        KV_STORAGE_DIR,
        functools.partial(storage_file_items, glob=TURBO_STORAGE_GLOB),
        width_pair_key,
    ),
    "turbo-updates": Population(
        "turbo update twins",
        KV_UPDATE_ROOT,
        functools.partial(
            glob_fn_items,
            directory=KV_UPDATE_DIR,
            glob=KV_UPDATE_GLOB,
            pattern=TURBO_UPDATE_FN_PATTERN,
        ),
        width_pair_key,
    ),
    # Its own population rather than a widened `turbo-storage` glob, for two
    # reasons. `storage_file_items` reads whole files under one directory of
    # one crate and these are fns inside one file of another, so neither glob
    # reaches them. And a figure prints the root it was measured over, so
    # folding them in would print the kv-quant storage directory beside a
    # number measured partly in kv-ssd.
    "turbo-ssd": Population(
        "turbo ssd helper twins",
        KV_SSD_BLOCK_IO_FILE,
        functools.partial(
            file_fn_items, file=KV_SSD_BLOCK_IO_FILE, pattern=TURBO_SSD_FN_PATTERN
        ),
        width_pair_key,
    ),
    "rotor-storage": Population(
        "rotor storage twins",
        KV_STORAGE_DIR,
        functools.partial(storage_file_items, glob=ROTOR_STORAGE_GLOB),
        width_pair_key,
    ),
    "rotor-updates": Population(
        "rotor update twins",
        KV_UPDATE_ROOT,
        functools.partial(
            glob_fn_items,
            directory=KV_UPDATE_DIR,
            glob=KV_UPDATE_GLOB,
            pattern=ROTOR_UPDATE_FN_PATTERN,
        ),
        width_pair_key,
    ),
    "ssd-hydrate": Population("ssd hydrate twins", MODELS_SOURCE_DIR, ssd_hydrate_items),
    # Every pair, not `width_pair_key`: the claim this figure measures is that
    # the update bodies share one sequence whatever codec they belong to, so a
    # key that only compares two widths of one family would report a 0 the
    # moment the widths collapsed and say nothing about the shape the
    # restructure writes once.
    "update-bodies": Population(
        "update_-prefixed fns of the update files",
        KV_UPDATE_ROOT,
        functools.partial(
            glob_fn_items,
            directory=KV_UPDATE_DIR,
            glob=KV_UPDATE_GLOB,
            pattern=UPDATE_FN_PATTERN,
        ),
    ),
}


def population_pairs(
    items: list[FnInfo], pair_key: Callable[[FnInfo], str] | None
) -> list[tuple[FnInfo, FnInfo]]:
    """Every pair a population contributes, in one order whatever order its
    collector walked the tree in.

    One sort here rather than one per collector: a collector's walk order is
    the file layout, and three of them (the glob over the update files, the
    hydrate scan over the model tree, the `impl RoundDrafter` scan) return a
    population whose order a move or a rename changes. `matched_lines` is
    measured both ways round, so no figure depends on this order any more;
    what the sort buys is that the pair list is a function of the population
    and not of the filesystem walk."""
    items = sorted(items, key=lambda fn: (fn.name, fn.file, fn.line))
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
