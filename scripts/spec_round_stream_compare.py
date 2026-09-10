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
    compare  <dir-a> <dir-b> [--fields f[|alias],...]
                               field-by-field over every cell in both, or over
                               the named fields only
    manifest <dir>             emit a MANIFEST.sha256 for one capture
    verify   <dir> [manifest]  hold one capture to a manifest

--fields
    Two captures either side of a change that renamed, added or dropped a field
    are not comparable line for line, and the fields that carry the round's
    arithmetic still are. `--fields` names those: each entry is a field name, or
    several under `|` when the two sides spell the same fact differently, and
    the first alias present in a line is the one read. A field present on one
    side and not the other is a difference; a field on neither is not a common
    field of that cell and is skipped. Nothing else about the comparison
    changes — the cell sets, the round counts and the completeness statement are
    read the same way.

EXIT
    0 agree, 1 a difference, 2 the comparison could not be made, 3 INCOMPLETE —
    a `verify` or a `compare` over captures holding only some of the pinned
    cells, which one pair always is. It names the ones that did not run and does
    not report 0, because a cell that could not run is not a cell that agreed.

    The manifest is held to the same three facts as the capture, at both ends: a
    hand-shrunk manifest is exit 2, not a smaller pass.
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

# What a manifest is a manifest OF. Regenerating one from whatever happens to be
# in a directory would re-bless a shrunken capture as the baseline — the six
# cells of a single pair, or five prompts because one stood down, would render
# as a complete manifest and read as one. These are the facts a regeneration
# cannot supply for itself, so they are stated here and the emitter is held to
# them. A change to the baseline that legitimately moves one is a change to this
# file, in the same commit, with the reason in the message.
PAIRS = (
    "the_adaptive_round_loop_reproduces_plain_greedy",
    "the_assistant_round_loop_reproduces_plain_greedy",
    "the_block_round_loop_reproduces_plain_greedy",
    "the_recurrent_round_loop_reproduces_plain_greedy",
    "the_restricted_vocab_round_loop_reproduces_plain_greedy",
    "the_two_model_round_loop_reproduces_plain_greedy",
)
PROMPTS = (
    "database-isolation",
    "hash-map-collisions",
    "longctx-4k",
    "photosynthesis",
    "tcp-congestion",
    "virtual-memory",
)
EXPECTED_CELLS = {f"{p}.{q}.stable.jsonl" for p in PAIRS for q in PROMPTS}
TOTAL_ROUNDS = 4423

# What every round event carries.
# Mirrors `ROUND_EVENT_FIELDS` in `crates/rmlx-models/tests/common/round_stream.rs`,
# which is the filter the engine writes these files through — stated again here
# because this side has to be able to refuse a file of well-formed JSON that is
# not a round stream.
ROUND_EVENT_FIELDS = ("round", "accept", "num_draft")


def cells(directory: pathlib.Path) -> dict[str, pathlib.Path]:
    return {p.name: p for p in sorted(directory.glob("*.stable.jsonl"))}


def rounds(path: pathlib.Path) -> list[dict]:
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        absent = [f for f in ROUND_EVENT_FIELDS if f not in obj]
        if absent:
            print(
                f"{path.name}:{n} carries no {absent}. A round event carries the round's "
                f"index, what it accepted and how many proposals it accepted them from, "
                f"so this line is not one and this file is not a round stream.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        timing = [k for k in obj if k.endswith("_ms")]
        if timing:
            print(
                f"{path.name}:{n} carries the wall-clock field(s) {timing}. The engine "
                f"writes this file with those dropped, so either it stopped or this is "
                f"not that file.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        out.append(obj)
    return out


def incompleteness(names: set[str]) -> tuple[list[str], list[str]]:
    """What a capture is missing, and what it holds that the baseline is not over."""
    return sorted(EXPECTED_CELLS - names), sorted(names - EXPECTED_CELLS)


def resolve(obj: dict, aliases: list[str]) -> tuple[str, object] | None:
    """The first of `aliases` this line carries, under the name it carries it."""
    for name in aliases:
        if name in obj:
            return name, obj[name]
    return None


def compare_fields(name: str, i: int, x: dict, y: dict, fields: list[list[str]]) -> bool:
    """Whether two round lines agree on every named field. Prints the first
    disagreement, a field one side has and the other does not included."""
    for aliases in fields:
        got_a, got_b = resolve(x, aliases), resolve(y, aliases)
        if got_a is None and got_b is None:
            continue
        if got_a is None or got_b is None:
            side = "B" if got_a is None else "A"
            print(f"{name} round-line {i} field {'|'.join(aliases)}: only {side} carries it",
                  file=sys.stderr)
            return False
        if got_a[1] != got_b[1]:
            print(f"{name} round-line {i} field {got_a[0]}/{got_b[0]}: "
                  f"A={got_a[1]!r} B={got_b[1]!r}", file=sys.stderr)
            return False
    return True


def compare(a_dir: pathlib.Path, b_dir: pathlib.Path,
            fields: list[list[str]] | None = None) -> int:
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
            if fields is not None:
                if not compare_fields(name, i, x, y, fields):
                    return 1
                continue
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
    over = "every field" if fields is None else \
        f"every one of {', '.join('|'.join(f) for f in fields)}"
    missing, extra = incompleteness(set(a))
    if extra:
        print(f"these captures hold {len(extra)} cell(s) the baseline is not over, "
              f"first {extra[0]}", file=sys.stderr)
        return 1
    if missing:
        for name in missing:
            print(f"  in neither capture: {name}", file=sys.stderr)
        print(f"INCOMPLETE: {len(a)} of {len(EXPECTED_CELLS)} cells agreed on "
              f"{total} round lines over {over}, {len(missing)} did not run")
        return 3
    print(f"{len(a)} cells, {total} round lines, {over} identical")
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
        rest = argv[4:]
        fields = None
        if rest[:1] == ["--fields"]:
            if len(rest) != 2 or not rest[1]:
                print("--fields takes one comma-separated list", file=sys.stderr)
                return 2
            fields = [alias.split("|") for alias in rest[1].split(",")]
            rest = []
        if len(argv) < 4 or rest:
            print(__doc__, file=sys.stderr)
            return 2
        other = pathlib.Path(argv[3])
        if not other.is_dir():
            print(f"{other} is not a directory", file=sys.stderr)
            return 2
        return compare(directory, other, fields)

    if mode == "manifest":
        lines = manifest_lines(directory)
        if not lines:
            print(f"no `*.stable.jsonl` cell in {directory}", file=sys.stderr)
            return 2
        names = {line.split()[2] for line in lines}
        missing = sorted(EXPECTED_CELLS - names)
        extra = sorted(names - EXPECTED_CELLS)
        if missing or extra:
            if missing:
                print(f"this capture is missing {len(missing)} of {len(EXPECTED_CELLS)} "
                      f"cells, first {missing[0]}", file=sys.stderr)
            if extra:
                print(f"this capture holds {len(extra)} cell(s) the baseline is not "
                      f"over, first {extra[0]}", file=sys.stderr)
            print("A manifest is over all six pairs and all six prompts. Emitting one "
                  "from a partial capture would re-bless it as the baseline.",
                  file=sys.stderr)
            return 2
        total = sum(int(line.split()[1]) for line in lines)
        if total != TOTAL_ROUNDS:
            print(f"this capture ran {total} rounds and the baseline is {TOTAL_ROUNDS}. "
                  f"A round count that moved is a round loop that changed: say which in "
                  f"the commit that moves TOTAL_ROUNDS.", file=sys.stderr)
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
        # The emitter is held to these three facts; so is the consumer. A
        # hand-shrunk manifest and a capture shrunk to match it would otherwise
        # agree with each other and report a pass over two cells.
        missing, extra = incompleteness(set(want))
        want_rounds = sum(int(v[1]) for v in want.values())
        if missing or extra or want_rounds != TOTAL_ROUNDS:
            print(f"{path} pins {len(want)} of {len(EXPECTED_CELLS)} cells over "
                  f"{want_rounds} rounds, and the baseline is {len(EXPECTED_CELLS)} "
                  f"over {TOTAL_ROUNDS}.", file=sys.stderr)
            if missing:
                print(f"  it pins no {missing[0]}" +
                      (f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""),
                      file=sys.stderr)
            if extra:
                print(f"  it pins {extra[0]}, which the baseline is not over" +
                      (f" (and {len(extra) - 1} more)" if len(extra) > 1 else ""),
                      file=sys.stderr)
            print("A manifest that is not over the whole baseline cannot say a capture "
                  "agreed with it.", file=sys.stderr)
            return 2
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
            return 3
        print(f"{matched} cells match {path}")
        return 0

    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
