#!/usr/bin/env python3
"""Structural figures for the KV update-path restructure.

Four modes, each printing one figure the restructure is judged on:

* `variants` — every `KvStorage` variant with its field shape, and the census
  of how many carry the store-slot shape the restructure writes one body for.
* `match-sites` — every `match` over a codec enum that enumerates the codec
  surface, per file, and the count. This is the "match sites a new codec must
  touch" figure. The codec enums are `KvStorage`, `KvQuant` and every enum a
  `KvStorage` field holds. A variant is found through its enum's path, an
  alias of it, `Self` inside an `impl` of it, or a glob import of it. Two more
  figures count what a new codec does not break at compile time:
  `subset-sites` (a `matches!` naming a codec variant, and a `match` under the
  bar with a catch-all arm) and `table-sites` (a `match` keyed by string
  literals or constants whose arm bodies name at least half of one codec
  enum).
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
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

from debt_report import extract_fns, is_test_path  # noqa: E402
from rust_scan import (  # noqa: E402
    MatchArm,
    ScanError,
    blank_text,
    block_end,
    enum_body,
    enum_variants,
    match_sites,
    matches_macros,
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


BASE_ENUMS = (("KvStorage", STORAGE_ENUM_FILE), ("KvQuant", QUANT_ENUM_FILE))
TYPE_NAME = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
ENUM_DEF = re.compile(r"\benum\s+([A-Z][A-Za-z0-9_]*)\b")
USE_STMT = re.compile(r"\buse\b[^;]*;")
#: A capitalised name that is not a path segment: a variant, if a glob import
#: brought an enum holding it into scope.
BARE_NAME = re.compile(r"(?<![\w:])([A-Z][A-Za-z0-9_]*)\b(?!\s*::)")
SOME_ARM = re.compile(r"\bSome\s*\(")
CATCH_ALL = re.compile(r"^(?:_|[a-z_][A-Za-z0-9_]*)$")
#: A table key: a string literal (blanked to spaces) or a `SCREAMING_CASE`
#: constant, bare or behind a path.
TABLE_KEY = re.compile(r'^(?:b?"[^"]*"|(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Z][A-Z0-9_]*)$')


def source_files(root: Path, include_tests: bool) -> dict[str, str]:
    """Blanked text of every Rust file under `crates`, keyed by its path."""
    base = root / CRATES_DIR
    if not base.is_dir():
        fail(f"{CRATES_DIR} is not a directory under {root}")
    out: dict[str, str] = {}
    for path in sorted(base.rglob("*.rs")):
        rel = path.relative_to(root)
        if "target" in rel.parts:
            continue
        if not include_tests and is_test_path(rel):
            continue
        _src, out[str(rel)] = read_blanked(path)
    return out


def enum_in(rel: str, blanked: str, name: str) -> tuple[list[str], set[str]]:
    """Variant names of `enum <name>`, and the type names its fields hold."""
    try:
        variants = [v for v, _fields in enum_variants(blanked, name)]
        body = enum_body(blanked, name)
    except ScanError as exc:
        fail(f"{rel}: {exc}")
        raise
    if not variants:
        fail(f"enum {name} holds no variant")
    return variants, set(TYPE_NAME.findall(body)) - set(variants)


def codec_enums(root: Path, sources: dict[str, str]) -> dict[str, list[str]]:
    """Variant names of `KvStorage`, `KvQuant`, and every enum a `KvStorage`
    field holds.

    The held set is closed transitively, so a dispatch moved into an enum
    below `KvStorage` stays a site and does not read as a reduction.
    """
    enums: dict[str, list[str]] = {}
    held: list[str] = []
    for name, rel in BASE_ENUMS:
        if rel not in sources:
            fail(f"{rel} is not a file under {root}")
        enums[name], types = enum_in(rel, sources[rel], name)
        if name == "KvStorage":
            held = sorted(types)
    defs: dict[str, list[str]] = {}
    for rel, blanked in sources.items():
        for m in ENUM_DEF.finditer(blanked):
            defs.setdefault(m.group(1), []).append(rel)
    while held:
        name = held.pop(0)
        if name in enums or name not in defs:
            continue
        if len(defs[name]) > 1:
            fail(f"enum {name}, held by a KvStorage field, is defined in {len(defs[name])} files")
        enums[name], types = enum_in(defs[name][0], sources[defs[name][0]], name)
        held.extend(sorted(types))
    return enums


def impl_self_type(header: str) -> str:
    """The last path segment of the type an `impl` header is for."""
    head = re.split(r"\bwhere\b", header)[0]
    if re.search(r"\bfor\b", head):
        head = re.split(r"\bfor\b", head)[-1]
    else:
        head = head.strip()
        if head.startswith("<"):
            depth = 0
            for i, c in enumerate(head):
                depth += (c == "<") - (c == ">")
                if depth == 0:
                    head = head[i + 1 :]
                    break
    m = re.match(r"\s*([A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)", head)
    return re.split(r"\s*::\s*", m.group(1))[-1] if m else ""


@dataclass
class Scope:
    """How one file spells the codec enums: path prefixes (each enum's name
    and its `use … as` aliases), glob-imported enums, and the spans of the
    `impl` blocks where `Self` is one of them."""

    prefixes: dict[str, str]
    globs: list[str]
    impls: list[tuple[int, int, str]]

    def self_at(self, pos: int) -> str | None:
        inner = [(start, name) for start, end, name in self.impls if start <= pos < end]
        return max(inner)[1] if inner else None

    def names_a_variant(self, blanked: str) -> bool:
        return bool(self.globs or self.impls) or any(
            re.search(rf"\b{re.escape(p)}\s*::", blanked) for p in self.prefixes
        )


def file_scope(blanked: str, enums: dict[str, list[str]]) -> Scope:
    prefixes = {name: name for name in enums}
    globs: list[str] = []
    for stmt in USE_STMT.finditer(blanked):
        for name in enums:
            for alias in re.findall(rf"\b{name}\s+as\s+([A-Za-z_]\w*)", stmt.group(0)):
                prefixes[alias] = name
            if re.search(rf"\b{name}\s*::\s*\*", stmt.group(0)) and name not in globs:
                globs.append(name)
    impls: list[tuple[int, int, str]] = []
    for m in re.finditer(r"\bimpl\b", blanked):
        brace = blanked.find("{", m.end())
        semi = blanked.find(";", m.end())
        if brace < 0 or 0 <= semi < brace:
            continue
        name = prefixes.get(impl_self_type(blanked[m.end() : brace]))
        if name:
            impls.append((brace, block_end(blanked, brace), name))
    return Scope(prefixes, globs, impls)


def named(text: str, pos: int, scope: Scope, enums: dict[str, list[str]]) -> dict[str, set[str]]:
    """The codec variants `text` names, per enum. `pos` places `Self`.

    Where `text` also holds a `Some(..)`, a bare `None` is `Option`'s, not a
    glob-imported variant.
    """
    out: dict[str, set[str]] = {name: set() for name in enums}
    prefixes = dict(scope.prefixes)
    self_ty = scope.self_at(pos)
    if self_ty:
        prefixes["Self"] = self_ty
    for prefix, name in prefixes.items():
        for v in re.findall(rf"\b{re.escape(prefix)}\s*::\s*([A-Za-z_]\w*)", text):
            if v in enums[name]:
                out[name].add(v)
    option = SOME_ARM.search(text) is not None
    for v in BARE_NAME.findall(text):
        if option and v == "None":
            continue
        for name in scope.globs:
            if v in enums[name]:
                out[name].add(v)
    return out


def widest(counts: dict[str, set[str]], bars: dict[str, int]) -> tuple[str, int] | None:
    """The enum naming the most variants among those named at or over their bar."""
    best: tuple[str, int] | None = None
    for name, bar in bars.items():
        n = len(counts[name])
        if n >= max(bar, 1) and (best is None or n > best[1]):
            best = (name, n)
    return best


def most_named(counts: dict[str, set[str]]) -> tuple[str, int]:
    name = max(counts, key=lambda k: len(counts[k]))
    return name, len(counts[name])


def alternatives(pattern: str) -> list[str]:
    return [alt.strip().rstrip("@").strip() for alt in pattern.split(" if ", 1)[0].split("|")]


def is_table(arms: list[MatchArm]) -> bool:
    """Every arm is keyed by a string literal or a constant, or is a catch-all."""
    keyed = False
    for arm in arms:
        for alt in alternatives(arm.pattern):
            if CATCH_ALL.match(alt):
                continue
            if not TABLE_KEY.match(alt):
                return False
            keyed = True
    return keyed


def mode_match_sites(root: Path, threshold: int | None, include_tests: bool) -> None:
    sources = source_files(root, include_tests)
    enums = codec_enums(root, sources)

    # A site naming fewer than half the enum's variants is a case analysis over
    # a subset; at half or more it enumerates the codec surface, which is the
    # population the restructure has to shrink. Derived from the enum rather
    # than fixed, so a codec added or retired moves the bar with it.
    bars = {name: threshold if threshold is not None else math.ceil(len(v) / 2) for name, v in enums.items()}
    for name, variants in enums.items():
        print(f"enum {name} variants={len(variants)} threshold={bars[name]}")

    per_file: dict[str, int] = {}
    total = forcing = subsets = tables = candidates = 0
    for rel, blanked in sources.items():
        scope = file_scope(blanked, enums)
        if not scope.names_a_variant(blanked):
            continue
        candidates += 1
        try:
            sites = match_sites(blanked)
            macros = matches_macros(blanked)
        except ScanError as exc:
            fail(f"{rel}: {exc}")
            raise
        for site in sites:
            counts = named(" | ".join(arm.pattern for arm in site.arms), site.start, scope, enums)
            catch_all = any(CATCH_ALL.match(alt) for arm in site.arms for alt in alternatives(arm.pattern))
            hit = widest(counts, bars)
            if hit:
                total += 1
                per_file[rel] = per_file.get(rel, 0) + 1
                forcing += not catch_all
                print(
                    f"site {rel}:{site.line} enum={hit[0]} variants={hit[1]} "
                    f"arms={len(site.arms)} catch_all={'yes' if catch_all else 'no'}"
                )
                continue
            name, n = most_named(counts)
            if n and catch_all:
                subsets += 1
                print(f"subset {rel}:{site.line} kind=wildcard enum={name} variants={n}")
                continue
            if n or not is_table(site.arms):
                continue
            hit = widest(named(" ".join(arm.body for arm in site.arms), site.start, scope, enums), bars)
            if hit:
                tables += 1
                print(f"table {rel}:{site.line} enum={hit[0]} variants={hit[1]} arms={len(site.arms)}")
        for macro in macros:
            name, n = most_named(named(macro.arms[0].pattern, macro.start, scope, enums))
            if n:
                subsets += 1
                print(f"subset {rel}:{macro.line} kind=matches enum={name} variants={n}")
    if not candidates:
        fail(f"no source file under {CRATES_DIR} names a KvStorage or KvQuant variant")
    for rel in sorted(per_file):
        print(f"file {rel} {per_file[rel]}")
    print(f"match-sites {total}")
    print(f"forcing-sites {forcing}")
    print(f"subset-sites {subsets}")
    print(f"table-sites {tables}")


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
