#!/usr/bin/env python3
"""Structural figures for the KV update-path restructure.

Four modes, each printing one figure the restructure is judged on:

* `variants` — every `KvStorage` variant with its field shape, and the census
  of how many carry the store-slot shape the restructure writes one body for.
* `match-sites` — every `match` over `KvStorage` or `KvQuant` that enumerates
  the codec surface, per file, and the count. This is the "match sites a new
  codec must touch" figure.
* `update-bodies` — every `update_`-prefixed fn of the update files, with the
  file it sits in and the lines its body holds.
* `refs` — `KvStorage::` / `KvQuant::` variant references in one file.

The figures are derived from the tree on every run. None of them is a hand
list, so one command reads the tree before the restructure and the tree after
it.

Every mode exits 2 and prints `unavailable: <reason>` when it cannot measure —
a missing directory, an enum it cannot find, a source shape it cannot read
back. It never prints a `0` that a reader could mistake for a measured answer.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from debt_report import extract_fns, is_test_path  # noqa: E402
from rust_scan import (  # noqa: E402
    ScanError,
    blank_text,
    enum_variants,
    match_sites,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

STORAGE_ENUM_FILE = "crates/rmlx-kv-quant/src/storage/kv_storage.rs"
QUANT_ENUM_FILE = "crates/rmlx-kv-quant/src/quant.rs"
UPDATE_DIR = "crates/rmlx-kv-quant/src/kvcache"
#: The update path is one dispatch file plus one file per codec family. A glob,
#: never a file list, so one command reads the tree that held every family in
#: `update.rs` and the tree that split them out.
UPDATE_GLOB = "update*.rs"
CRATES_DIR = "crates"

#: Every `update_`-prefixed fn of the update files. Not only the per-variant
#: bodies: the shared entries the dispatch reaches (`update_and_sdpa_*`,
#: `update_decode_fp16*`, `update_prefill_raw`) carry the prefix and are
#: counted with them. The prefix is the rule, so no hand-drawn boundary
#: decides which body is "per-variant" enough to count.
UPDATE_FN_PATTERN = re.compile(r"^update_")


def fail(reason: str) -> None:
    print(f"unavailable: {reason}", file=sys.stderr)
    raise SystemExit(2)


def read_blanked(path: Path) -> tuple[str, str]:
    src = path.read_text(errors="ignore")
    try:
        return src, blank_text(src)
    except ScanError as exc:
        fail(f"{path}: {exc}")
        raise


def variant_shapes(root: Path) -> list[tuple[str, list[str]]]:
    path = root / STORAGE_ENUM_FILE
    if not path.is_file():
        fail(f"{STORAGE_ENUM_FILE} is not a file under {root}")
    _src, blanked = read_blanked(path)
    try:
        variants = enum_variants(blanked, "KvStorage")
    except ScanError as exc:
        fail(f"{STORAGE_ENUM_FILE}: {exc}")
        raise
    if not variants:
        fail("enum KvStorage holds no variant")
    return variants


#: Scalar knobs a variant may carry beside its two store slots. A field
#: outside this set holds state, not a setting, so the variant leaves the
#: shared shape and is reported as `other` rather than folded in quietly.
SCALAR_KNOBS = frozenset({"bits", "v_bits", "v_group_size"})


def classify(fields: list[str]) -> str:
    """Which update body a variant's field shape can share.

    `kv_slots` is the shape the restructure writes once: a K slot, a V slot and
    `max_seq`. `kv_slots_plus` is that shape with scalar knobs beside it.
    `k_only` is the same shape with the V side living on the parent cache as
    bf16. Everything else holds its state somewhere the shape cannot reach, and
    keeps its own body.
    """
    has = set(fields)
    if has == {"k", "v", "max_seq"}:
        return "kv_slots"
    if {"k", "v", "max_seq"} <= has and has - {"k", "v", "max_seq"} <= SCALAR_KNOBS:
        return "kv_slots_plus"
    if has == {"k", "max_seq"}:
        return "k_only"
    return "other"


def mode_variants(root: Path) -> None:
    variants = variant_shapes(root)
    counts: dict[str, int] = {}
    for name, fields in variants:
        kind = classify(fields)
        counts[kind] = counts.get(kind, 0) + 1
        print(f"variant {name} fields={','.join(fields) or '-'} shape={kind}")
    print(f"variants {len(variants)}")
    for kind in ("kv_slots", "kv_slots_plus", "k_only", "other"):
        print(f"shape {kind} {counts.get(kind, 0)}")
    print(f"shape store_slots_total {sum(counts.get(k, 0) for k in ('kv_slots', 'kv_slots_plus', 'k_only'))}")


def enum_variant_count(root: Path, rel: str, name: str) -> int:
    path = root / rel
    if not path.is_file():
        fail(f"{rel} is not a file under {root}")
    _src, blanked = read_blanked(path)
    try:
        variants = enum_variants(blanked, name)
    except ScanError as exc:
        fail(f"{rel}: {exc}")
        raise
    if not variants:
        fail(f"enum {name} holds no variant")
    return len(variants)


def candidate_files(root: Path, include_tests: bool) -> list[Path]:
    base = root / CRATES_DIR
    if not base.is_dir():
        fail(f"{CRATES_DIR} is not a directory under {root}")
    out: list[Path] = []
    for path in sorted(base.rglob("*.rs")):
        rel = path.relative_to(root)
        if "target" in rel.parts:
            continue
        if not include_tests and is_test_path(rel):
            continue
        text = path.read_text(errors="ignore")
        if "KvStorage::" in text or "KvQuant::" in text:
            out.append(path)
    return out


CATCH_ALL = re.compile(r"^(?:_|[a-z_][A-Za-z0-9_]*)$")


def arm_variants(pattern: str) -> tuple[set[str], set[str], bool]:
    """`(KvStorage names, KvQuant names, is catch-all)` for one arm pattern."""
    guard = pattern.split(" if ", 1)[0].strip()
    storage = set(re.findall(r"\bKvStorage::([A-Za-z0-9_]+)", guard))
    quant = set(re.findall(r"\bKvQuant::([A-Za-z0-9_]+)", guard))
    catch_all = False
    for alt in guard.split("|"):
        alt = alt.strip().rstrip("@").strip()
        if CATCH_ALL.match(alt):
            catch_all = True
    return storage, quant, catch_all


def mode_match_sites(root: Path, threshold: int | None, include_tests: bool) -> None:
    storage_total = enum_variant_count(root, STORAGE_ENUM_FILE, "KvStorage")
    quant_total = enum_variant_count(root, QUANT_ENUM_FILE, "KvQuant")
    files = candidate_files(root, include_tests)
    if not files:
        fail(f"no source file under {CRATES_DIR} names a KvStorage or KvQuant variant")

    # A site naming fewer than half the enum's variants is a case analysis over
    # a subset; at half or more it enumerates the codec surface, which is the
    # population the restructure has to shrink. Derived from the enum rather
    # than fixed, so a codec added or retired moves the bar with it.
    bar_storage = threshold if threshold is not None else math.ceil(storage_total / 2)
    bar_quant = threshold if threshold is not None else math.ceil(quant_total / 2)
    print(f"enum KvStorage variants={storage_total} threshold={bar_storage}")
    print(f"enum KvQuant variants={quant_total} threshold={bar_quant}")

    per_file: dict[str, int] = {}
    forcing = 0
    total = 0
    for path in files:
        rel = str(path.relative_to(root))
        try:
            _src, blanked = read_blanked(path)
            sites = match_sites(blanked)
        except ScanError as exc:
            fail(f"{rel}: {exc}")
        for site in sites:
            storage: set[str] = set()
            quant: set[str] = set()
            catch_all = False
            for arm in site.arms:
                s, q, c = arm_variants(arm.pattern)
                storage |= s
                quant |= q
                catch_all = catch_all or c
            over = storage if len(storage) >= len(quant) else quant
            which = "KvStorage" if over is storage else "KvQuant"
            bar = bar_storage if which == "KvStorage" else bar_quant
            if len(over) < bar:
                continue
            total += 1
            per_file[rel] = per_file.get(rel, 0) + 1
            if not catch_all:
                forcing += 1
            print(
                f"site {rel}:{site.line} enum={which} variants={len(over)} "
                f"arms={len(site.arms)} catch_all={'yes' if catch_all else 'no'}"
            )
    for rel in sorted(per_file):
        print(f"file {rel} {per_file[rel]}")
    print(f"match-sites {total}")
    print(f"forcing-sites {forcing}")


def update_files(root: Path) -> list[Path]:
    base = root / UPDATE_DIR
    if not base.is_dir():
        fail(f"{UPDATE_DIR} is not a directory under {root}")
    paths = [p for p in sorted(base.glob(UPDATE_GLOB)) if not is_test_path(p.relative_to(root))]
    if not paths:
        fail(f"{UPDATE_DIR}/{UPDATE_GLOB} matches no file under {root}")
    return paths


def mode_update_bodies(root: Path) -> None:
    bodies: list[tuple[str, str, int, int]] = []
    file_lines = 0
    for path in update_files(root):
        src, _blanked = read_blanked(path)
        file_lines += len(src.splitlines())
        rel = str(path.relative_to(root))
        fns, _skipped = extract_fns(src)
        bodies.extend(
            (rel, fn.name, fn.line, fn.body.count("\n") + 1)
            for fn in fns
            if UPDATE_FN_PATTERN.search(fn.name)
        )
    if not bodies:
        fail(f"{UPDATE_DIR}/{UPDATE_GLOB} holds no update_* fn")
    for rel, name, line, lines in bodies:
        print(f"body {name} file={rel} line={line} lines={lines}")
    print(f"file-lines {file_lines}")
    print(f"update-bodies {len(bodies)}")
    print(f"update-body-lines {sum(lines for _r, _n, _l, lines in bodies)}")


def mode_refs(root: Path, rel: str) -> None:
    path = root / rel
    if not path.is_file():
        fail(f"{rel} is not a file under {root}")
    _src, blanked = read_blanked(path)
    storage = re.findall(r"\bKvStorage::([A-Za-z0-9_]+)", blanked)
    quant = re.findall(r"\bKvQuant::([A-Za-z0-9_]+)", blanked)
    if not storage and not quant:
        fail(f"{rel} names no KvStorage or KvQuant variant")
    print(f"refs {rel} KvStorage={len(storage)} distinct={len(set(storage))}")
    print(f"refs {rel} KvQuant={len(quant)} distinct={len(set(quant))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["variants", "match-sites", "update-bodies", "refs", "all"])
    ap.add_argument("--root", default=str(REPO_ROOT), help="tree to measure")
    ap.add_argument("--threshold", type=int, default=None, help="override the derived arm bar")
    ap.add_argument("--file", default=None, help="file for the refs mode")
    ap.add_argument("--include-tests", action="store_true", help="count test files too")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        fail(f"{root} is not a directory")
    if args.mode == "variants":
        mode_variants(root)
    elif args.mode == "match-sites":
        mode_match_sites(root, args.threshold, args.include_tests)
    elif args.mode == "update-bodies":
        mode_update_bodies(root)
    elif args.mode == "refs":
        if not args.file:
            fail("refs mode needs --file")
        mode_refs(root, args.file)
    else:
        mode_variants(root)
        mode_match_sites(root, args.threshold, args.include_tests)
        mode_update_bodies(root)


if __name__ == "__main__":
    main()
