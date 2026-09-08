#!/usr/bin/env python3
"""Compare two captures of the speculative per-round event stream, or hold one
to the checked-in baseline manifest.

WHY
    The equivalence pairs read the answer and the accept counters read the
    aggregate; neither sees which round moved. The per-round stream does, and it
    is what an extraction of the round loops is judged against — so a capture
    taken before a change and one taken after have to be compared field by
    field, not eyeballed.

WHAT IS COMPARED
    The `*.stable.jsonl` files the run writes: the same round objects with every
    wall-clock `*_ms` field dropped. Those move on every run of the same engine,
    so a digest over the full line compares two machines' load. The engine is
    the single producer of that form; this script checks that it dropped what it
    says it drops and hashes what it wrote.

USAGE
    compare  <dir-a> <dir-b>   field-by-field over every cell in both
    manifest <dir>             emit a MANIFEST.sha256 for one capture
    verify   <dir> [manifest]  hold one capture to a manifest

EXIT
    0 agree, 1 a difference, 2 the comparison could not be made. A `verify` over
    a capture holding only some of the pinned cells — one pair is six of
    thirty-six — names the ones that did not run and says INCOMPLETE, because a
    cell that could not run is not a cell that agreed.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys

DEFAULT_MANIFEST = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256"
)


def cells(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    return {p.name: p for p in sorted(directory.glob("*.stable.jsonl"))}


def rounds(path: pathlib.Path) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        timing = [k for k in obj if k.endswith("_ms")]
        if timing:
            raise SystemExit(
                f"{path.name}:{n} carries the wall-clock field(s) {timing}. The engine "
                f"writes this file with those dropped, so either it stopped or this is "
                f"not that file."
            )
        out.append(obj)
    return out


def compare(a_dir: pathlib.Path, b_dir: pathlib.Path) -> int:
    a, b = cells(a_dir), cells(b_dir)
    if not a or not b:
        print(f"no `*.stable.jsonl` cell in {a_dir if not a else b_dir}", file=sys.stderr)
        return 2
    if a.keys() != b.keys():
        print(f"cell sets differ: only in A {sorted(a.keys() - b.keys())}, "
              f"only in B {sorted(b.keys() - a.keys())}", file=sys.stderr)
        return 1
    total = 0
    for name in a:
        ra, rb = rounds(a[name]), rounds(b[name])
        if len(ra) != len(rb):
            print(f"{name}: A ran {len(ra)} rounds and B ran {len(rb)}", file=sys.stderr)
            return 1
        for i, (x, y) in enumerate(zip(ra, rb)):
            if x.keys() != y.keys():
                print(f"{name} round-line {i}: field sets differ — "
                      f"A only {sorted(x.keys() - y.keys())}, "
                      f"B only {sorted(y.keys() - x.keys())}", file=sys.stderr)
                return 1
            for k in sorted(x):
                if x[k] != y[k]:
                    print(f"{name} round-line {i} field {k}: A={x[k]!r} B={y[k]!r}",
                          file=sys.stderr)
                    return 1
        total += len(ra)
    print(f"{len(a)} cells, {total} round lines, every field identical")
    return 0


def manifest_lines(directory: pathlib.Path) -> list[str]:
    out = []
    for name, path in cells(directory).items():
        n = len(rounds(path))
        out.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {n:>5}  {name}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    mode = argv[1]
    directory = pathlib.Path(argv[2])
    if not directory.is_dir():
        print(f"{directory} is not a directory", file=sys.stderr)
        return 2

    if mode == "compare":
        if len(argv) != 4:
            print(__doc__, file=sys.stderr)
            return 2
        other = pathlib.Path(argv[3])
        if not other.is_dir():
            print(f"{other} is not a directory", file=sys.stderr)
            return 2
        return compare(directory, other)

    if mode == "manifest":
        lines = manifest_lines(directory)
        if not lines:
            print(f"no `*.stable.jsonl` cell in {directory}", file=sys.stderr)
            return 2
        print("\n".join(lines))
        return 0

    if mode == "verify":
        path = pathlib.Path(argv[3]) if len(argv) > 3 else DEFAULT_MANIFEST
        if not path.is_file():
            print(f"no manifest at {path}", file=sys.stderr)
            return 2
        want = {
            line.split()[2]: (line.split()[0], line.split()[1])
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        have = {
            line.split()[2]: (line.split()[0], line.split()[1])
            for line in manifest_lines(directory)
        }
        if not have:
            print(f"no `*.stable.jsonl` cell in {directory}", file=sys.stderr)
            return 2
        bad = 0
        matched = 0
        not_run = []
        for name in sorted(want.keys() | have.keys()):
            if name not in have:
                # One pair at a time is the normal way to run these, so a cell
                # the capture does not hold is a cell that did not run. It is
                # named and counted rather than passed over: a cell that could
                # not run is not a cell that agreed.
                not_run.append(name)
            elif name not in want:
                print(f"{name}: captured and the manifest pins no such cell",
                      file=sys.stderr)
                bad = 1
            elif want[name] != have[name]:
                print(f"{name}: manifest {want[name][0]} over {want[name][1]} rounds, "
                      f"capture {have[name][0]} over {have[name][1]}", file=sys.stderr)
                bad = 1
            else:
                matched += 1
        if bad:
            return 1
        if not matched:
            print(f"no cell in {directory} is pinned by {path}", file=sys.stderr)
            return 2
        if not_run:
            for name in not_run:
                print(f"  not in this capture: {name}", file=sys.stderr)
            print(f"INCOMPLETE: {matched} of {len(want)} pinned cells matched, "
                  f"{len(not_run)} did not run")
            return 0
        print(f"{matched} cells match {path}")
        return 0

    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
