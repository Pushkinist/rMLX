#!/usr/bin/env bash
# scripts/spec_round_stream_compare_selftest.sh — recall test for
# `scripts/spec_round_stream_compare.py verify`, the one reader of the pinned
# round-stream manifest.
#
# Each case builds a synthetic capture of the 36 pinned cells, whose round
# counts sum to the script's own TOTAL_ROUNDS, and asserts the literal exit code
# and the line naming why. The last case holds the checked-in manifest to the
# script's three literals. No GPU, no model.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$REPO_ROOT/scripts/spec_round_stream_compare.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILED=0
PASSED=0

PAIR=the_block_round_loop_reproduces_plain_greedy
OTHER=the_assistant_round_loop_reproduces_plain_greedy

# capture <dir>: one synthetic round stream per pinned cell.
capture() {
    python3 - "$TOOL" "$1" <<'EOF'
import importlib.util, json, pathlib, sys
spec = importlib.util.spec_from_file_location("compare", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True)
names = sorted(m.EXPECTED_CELLS)
base, extra = divmod(m.TOTAL_ROUNDS, len(names))
for i, name in enumerate(names):
    rounds = base + (1 if i < extra else 0)
    lines = [json.dumps({"target": "rmlx::spec::phase", "message": "speculative round",
                         "round": str(r), "accept": str(r % 3), "num_draft": "4",
                         "emitted_total": str(3 * r + 1)}) for r in range(rounds)]
    (out / name).write_text("\n".join(lines) + "\n")
EOF
}

# pinned <dir>: a manifest of that capture, as `manifest` emits it.
pinned() {
    python3 "$TOOL" manifest "$1" >"$1.manifest" || exit 1
    printf '%s' "$1.manifest"
}

# edit <file>: the same rounds, one accepted count moved on the first line.
edit() {
    python3 - "$1" <<'EOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
first = json.loads(lines[0])
first["accept"] = str(int(first["accept"]) + 1)
lines[0] = json.dumps(first)
p.write_text("\n".join(lines) + "\n")
EOF
}

# case_run <name> <what> <want-exit> <needle|-> <args...>
case_run() {
    local name="$1" what="$2" want="$3" needle="$4"
    shift 4
    local out status
    out=$(python3 "$TOOL" "$@" 2>&1)
    status=$?
    if [ "$status" != "$want" ]; then
        FAILED=$((FAILED + 1))
        printf '  FAIL %-36s (want exit %s, got %s) — %s\n%s\n' "$name" "$want" "$status" "$what" "$out"
    elif [ "$needle" != "-" ] && ! grep -qF -- "$needle" <<<"$out"; then
        FAILED=$((FAILED + 1))
        printf '  FAIL %-36s (missing %q) — %s\n%s\n' "$name" "$needle" "$what" "$out"
    else
        PASSED=$((PASSED + 1))
        printf '  ok   %-36s — %s\n' "$name" "$what"
    fi
}

dir="$WORK/true"; capture "$dir"; man=$(pinned "$dir")
case_run true_pin "a capture held to its own manifest matches" \
    0 "36 cells match" verify "$dir" "$man"
case_run true_pin_one_pair "one pair of it matches on its six cells" \
    0 "6 cells match" verify "$dir" "$man" --pair "$PAIR"

dir="$WORK/stale"; capture "$dir"; man=$(pinned "$dir")
edit "$dir/$PAIR.photosynthesis.stable.jsonl"
case_run stale_pin "one round's accepted count moved: the pin is stale" \
    1 "$PAIR.photosynthesis.stable.jsonl: manifest" verify "$dir" "$man"
case_run stale_pin_one_pair "the same moved cell under --pair" \
    1 "$PAIR.photosynthesis.stable.jsonl: manifest" verify "$dir" "$man" --pair "$PAIR"
case_run stale_other_pair "a moved cell of another pair is not read" \
    0 "6 cells match" verify "$dir" "$man" --pair "$OTHER"

dir="$WORK/short"; capture "$dir"; man=$(pinned "$dir")
short="$dir/$PAIR.longctx-4k.stable.jsonl"
python3 -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text("".join(p.read_text().splitlines(True)[:-1]))' \
    "$short"
case_run stale_round_count "one round fewer is a difference, and the capture's count is named" \
    1 "capture $(shasum -a 256 "$short" | cut -c1-64) over $(grep -c '' "$short")" \
    verify "$dir" "$man" --pair "$PAIR"

dir="$WORK/missing"; capture "$dir"; man=$(pinned "$dir")
rm -f "$dir/$PAIR.virtual-memory.stable.jsonl"
case_run one_pair_incomplete "a pair with a cell that did not run is INCOMPLETE, not a pass" \
    3 "INCOMPLETE: 5 of 6 pinned cells matched" verify "$dir" "$man" --pair "$PAIR"

dir="$WORK/foreign"; capture "$dir"; man=$(pinned "$dir")
printf 'not json\n' >"$dir/$OTHER.tcp-congestion.stable.jsonl"
printf 'not json\n' >"$dir/${PAIR}_at_the_declared_block.tcp-congestion.stable.jsonl"
case_run foreign_cells_not_read "an unreadable cell of another pair, or of a test whose name extends this one, is not read" \
    0 "6 cells match" verify "$dir" "$man" --pair "$PAIR"
dir="$WORK/unpinned"; capture "$dir"; man=$(pinned "$dir")
printf '{"round": "0", "accept": "0", "num_draft": "4", "emitted_total": "1"}\n' \
    >"$dir/${PAIR}_at_the_declared_block.tcp-congestion.stable.jsonl"
case_run unpinned_test_not_verified "a cell of a test the manifest is not over is not read, and is named" \
    0 "not read: the cells of 1 test(s) the manifest is not over (${PAIR}_at_the_declared_block)" \
    verify "$dir" "$man"
case_run unpinned_test_matches_the_rest "with that cell beside them, the pinned cells still match" \
    0 "36 cells match" verify "$dir" "$man"
case_run unpinned_test_not_in_manifest "a manifest from that directory is over the pinned cells alone" \
    0 "not read: the cells of 1 test(s)" manifest "$dir"

dir="$WORK/shrunk"; capture "$dir"; man=$(pinned "$dir")
python3 -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text("".join(p.read_text().splitlines(True)[1:]))' "$man"
case_run manifest_line_dropped "a hand-shrunk manifest is not a smaller pass" \
    2 "pins 35 of 36 cells" verify "$dir" "$man"
dir="$WORK/recount"; capture "$dir"; man=$(pinned "$dir")
python3 - "$man" <<'EOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
digest, rounds, name = lines[0].split()
lines[0] = f"{digest}  {int(rounds) + 1:>5}  {name}"
p.write_text("\n".join(lines) + "\n")
EOF
case_run manifest_count_edited "a manifest whose counts do not sum to TOTAL_ROUNDS is refused" \
    2 "and the baseline is 36 over" verify "$dir" "$man" --pair "$PAIR"

dir="$WORK/none"; capture "$dir"; man=$(pinned "$dir")
rm -f "$dir/$PAIR".*
case_run pair_not_run "a pair with no cell at all is not a comparison" \
    2 "no \`*.stable.jsonl\` cell of $PAIR" verify "$dir" "$man" --pair "$PAIR"
case_run pair_not_pinned "a test the manifest does not pin is refused" \
    2 "not one of the 6 pinned pairs" verify "$dir" "$man" --pair "${PAIR}_at_the_declared_block"
case_run pair_without_name "--pair with no name is refused" \
    2 "--pair takes one pinned test name" verify "$dir" "$man" --pair

dir="$WORK/real"; mkdir -p "$dir"
printf '{"round": "0", "accept": "0", "num_draft": "4", "emitted_total": "1"}\n' \
    >"$dir/$PAIR.photosynthesis.stable.jsonl"
case_run checked_in_manifest "the checked-in manifest is over the 36 cells and TOTAL_ROUNDS, so a capture is compared" \
    1 "$PAIR.photosynthesis.stable.jsonl: manifest" verify "$dir" --pair "$PAIR"

printf '\n%d passed, %d failed\n' "$PASSED" "$FAILED"
[ "$FAILED" = 0 ]
