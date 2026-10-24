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
# Exit 0 = every case held. Exit 1 = at least one did not. Exit 2 = a capture
# could not be built.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$REPO_ROOT/scripts/spec_round_stream_compare.py"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init spec-round-stream-compare-selftest

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

dir="$ST_WORK/true"; capture "$dir"; man=$(pinned "$dir")
st_case true_pin 0 --what "a capture held to its own manifest matches" \
    --has "36 cells match" -- python3 "$TOOL" verify "$dir" "$man"
st_case true_pin_one_pair 0 --what "one pair of it matches on its six cells" \
    --has "6 cells match" -- python3 "$TOOL" verify "$dir" "$man" --pair "$PAIR"

dir="$ST_WORK/stale"; capture "$dir"; man=$(pinned "$dir")
edit "$dir/$PAIR.photosynthesis.stable.jsonl"
st_case stale_pin 1 --what "one round's accepted count moved: the pin is stale" \
    --has "$PAIR.photosynthesis.stable.jsonl: manifest" -- python3 "$TOOL" verify "$dir" "$man"
st_case stale_pin_one_pair 1 --what "the same moved cell under --pair" \
    --has "$PAIR.photosynthesis.stable.jsonl: manifest" -- python3 "$TOOL" verify "$dir" "$man" --pair "$PAIR"
st_case stale_other_pair 0 --what "a moved cell of another pair is not read" \
    --has "6 cells match" -- python3 "$TOOL" verify "$dir" "$man" --pair "$OTHER"

dir="$ST_WORK/short"; capture "$dir"; man=$(pinned "$dir")
short="$dir/$PAIR.longctx-4k.stable.jsonl"
python3 -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text("".join(p.read_text().splitlines(True)[:-1]))' \
    "$short"
st_case stale_round_count 1 --what "one round fewer is a difference, and the capture's count is named" \
    --has "capture $(shasum -a 256 "$short" | cut -c1-64) over $(grep -c '' "$short")" \
    -- python3 "$TOOL" verify "$dir" "$man" --pair "$PAIR"

dir="$ST_WORK/missing"; capture "$dir"; man=$(pinned "$dir")
rm -f "$dir/$PAIR.virtual-memory.stable.jsonl"
st_case one_pair_incomplete 3 --what "a pair with a cell that did not run is INCOMPLETE, not a pass" \
    --has "INCOMPLETE: 5 of 6 pinned cells matched" -- python3 "$TOOL" verify "$dir" "$man" --pair "$PAIR"

dir="$ST_WORK/foreign"; capture "$dir"; man=$(pinned "$dir")
printf 'not json\n' >"$dir/$OTHER.tcp-congestion.stable.jsonl"
printf 'not json\n' >"$dir/${PAIR}_at_the_declared_block.tcp-congestion.stable.jsonl"
st_case foreign_cells_not_read 0 --what "an unreadable cell of another pair, or of a test whose name extends this one, is not read" \
    --has "6 cells match" -- python3 "$TOOL" verify "$dir" "$man" --pair "$PAIR"
dir="$ST_WORK/unpinned"; capture "$dir"; man=$(pinned "$dir")
printf '{"round": "0", "accept": "0", "num_draft": "4", "emitted_total": "1"}\n' \
    >"$dir/${PAIR}_at_the_declared_block.tcp-congestion.stable.jsonl"
st_case unpinned_test_not_verified 0 --what "a cell of a test the manifest is not over is not read, and is named" \
    --has "not read: the cells of 1 test(s) the manifest is not over (${PAIR}_at_the_declared_block)" -- python3 "$TOOL" verify "$dir" "$man"
st_case unpinned_test_matches_the_rest 0 --what "with that cell beside them, the pinned cells still match" \
    --has "36 cells match" -- python3 "$TOOL" verify "$dir" "$man"
st_case unpinned_test_not_in_manifest 0 --what "a manifest from that directory is over the pinned cells alone" \
    --has "not read: the cells of 1 test(s)" -- python3 "$TOOL" manifest "$dir"

dir="$ST_WORK/shrunk"; capture "$dir"; man=$(pinned "$dir")
python3 -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); p.write_text("".join(p.read_text().splitlines(True)[1:]))' "$man"
st_case manifest_line_dropped 2 --what "a hand-shrunk manifest is not a smaller pass" \
    --has "pins 35 of 36 cells" -- python3 "$TOOL" verify "$dir" "$man"
dir="$ST_WORK/recount"; capture "$dir"; man=$(pinned "$dir")
python3 - "$man" <<'EOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
digest, rounds, name = lines[0].split()
lines[0] = f"{digest}  {int(rounds) + 1:>5}  {name}"
p.write_text("\n".join(lines) + "\n")
EOF
st_case manifest_count_edited 2 --what "a manifest whose counts do not sum to TOTAL_ROUNDS is refused" \
    --has "and the baseline is 36 over" -- python3 "$TOOL" verify "$dir" "$man" --pair "$PAIR"

dir="$ST_WORK/none"; capture "$dir"; man=$(pinned "$dir")
rm -f "$dir/$PAIR".*
st_case pair_not_run 2 --what "a pair with no cell at all is not a comparison" \
    --has "no \`*.stable.jsonl\` cell of $PAIR" -- python3 "$TOOL" verify "$dir" "$man" --pair "$PAIR"
st_case pair_not_pinned 2 --what "a test the manifest does not pin is refused" \
    --has "not one of the 6 pinned pairs" -- python3 "$TOOL" verify "$dir" "$man" --pair "${PAIR}_at_the_declared_block"
st_case pair_without_name 2 --what "--pair with no name is refused" \
    --has "--pair takes one pinned test name" -- python3 "$TOOL" verify "$dir" "$man" --pair

dir="$ST_WORK/real"; mkdir -p "$dir"
printf '{"round": "0", "accept": "0", "num_draft": "4", "emitted_total": "1"}\n' \
    >"$dir/$PAIR.photosynthesis.stable.jsonl"
st_case checked_in_manifest 1 --what "the checked-in manifest is over the 36 cells and TOTAL_ROUNDS, so a capture is compared" \
    --has "$PAIR.photosynthesis.stable.jsonl: manifest" -- python3 "$TOOL" verify "$dir" --pair "$PAIR"

st_done
