#!/usr/bin/env bash
# Recall test for spec_round_stream_compare.py verify: synthetic captures, asserting the exit code.
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

# retouch <file> <python statement>: run the statement on the first round line
# of the file, named `first`, and write the file back.
retouch() {
    python3 - "$1" "$2" <<'EOF'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
first = json.loads(lines[0])
exec(sys.argv[2])
lines[0] = json.dumps(first)
p.write_text("\n".join(lines) + "\n")
EOF
}

# edit <file>: the same rounds, one accepted count moved on the first line.
edit() {
    retouch "$1" 'first["accept"] = str(int(first["accept"]) + 1)'
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

dir="$ST_WORK/timing"; capture "$dir"; man=$(pinned "$dir")
retouch "$dir/$PAIR.photosynthesis.stable.jsonl" 'first["elapsed_ms"] = "3"'
st_case timing_field_refused 2 --what "a cell that carries a wall-clock field is not a stable stream" \
    --has "carries the wall-clock field(s)" -- python3 "$TOOL" verify "$dir" "$man"

dir="$ST_WORK/absent"; capture "$dir"; man=$(pinned "$dir")
retouch "$dir/$PAIR.photosynthesis.stable.jsonl" 'del first["emitted_total"]'
st_case round_field_absent_refused 2 --what "a line without a round event's field is not a round stream" \
    --has "carries no ['emitted_total']" -- python3 "$TOOL" verify "$dir" "$man"

dir="$ST_WORK/blank"; capture "$dir"
python3 -c 'import pathlib, sys; p = pathlib.Path(sys.argv[1]); t = p.read_text().splitlines(True); p.write_text(t[0] + "\n" + "".join(t[1:]) + "\n")' \
    "$dir/$PAIR.photosynthesis.stable.jsonl"
man=$(pinned "$dir")
st_case blank_lines_skipped 0 --what "a blank line in a cell is not a round" \
    --has "36 cells match" -- python3 "$TOOL" verify "$dir" "$man"

dir="$ST_WORK/more_rounds"; capture "$dir"
printf '%s\n' '{"round": "9999", "accept": "0", "num_draft": "4", "emitted_total": "1"}' \
    >>"$dir/$PAIR.photosynthesis.stable.jsonl"
st_case manifest_of_another_round_count_refused 2 --what "a manifest is not emitted from a capture whose round count moved" \
    --has "A round count that moved is a round loop that changed" -- python3 "$TOOL" manifest "$dir"

dir="$ST_WORK/extra_line"; capture "$dir"; man=$(pinned "$dir")
printf '%064d      0  bogus.photosynthesis.stable.jsonl\n' 0 >>"$man"
st_case manifest_extra_cell_refused 2 --what "a manifest that pins a cell the baseline is not over is refused" \
    --has "it pins bogus.photosynthesis.stable.jsonl, which the baseline is not over" -- python3 "$TOOL" verify "$dir" "$man"

dir="$ST_WORK/compensated"; capture "$dir"; man=$(pinned "$dir")
python3 - "$man" <<'EOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
lines = p.read_text().splitlines()
_, rounds0, _ = lines[0].split()
digest, rounds1, name = lines[1].split()
lines[1] = f"{digest}  {int(rounds1) + int(rounds0):>5}  {name}"
p.write_text("\n".join(lines[1:]) + "\n")
EOF
st_case manifest_missing_cell_refused 2 --what "a manifest that lost a cell is refused though its round count still sums" \
    --has "pins 35 of 36 cells" -- python3 "$TOOL" verify "$dir" "$man"

dir="$ST_WORK/unpinned_cell"; capture "$dir"; man=$(pinned "$dir")
cp "$dir/$PAIR.photosynthesis.stable.jsonl" "$dir/$PAIR.zzz-unpinned.stable.jsonl"
st_case captured_cell_not_pinned 1 --what "a cell of a pinned pair that the manifest does not pin is a difference" \
    --has "captured and the manifest pins no such cell" -- python3 "$TOOL" verify "$dir" "$man"

dir="$ST_WORK/two_manifests"; capture "$dir"; man=$(pinned "$dir")
st_case verify_takes_one_manifest 2 --what "a second manifest argument is refused, not ignored" \
    --has "USAGE" -- python3 "$TOOL" verify "$dir" "$man" "$man"

dir="$ST_WORK/real"; mkdir -p "$dir"
printf '{"round": "0", "accept": "0", "num_draft": "4", "emitted_total": "1"}\n' \
    >"$dir/$PAIR.photosynthesis.stable.jsonl"
st_case checked_in_manifest 1 --what "the checked-in manifest is over the 36 cells and TOTAL_ROUNDS, so a capture is compared" \
    --has "$PAIR.photosynthesis.stable.jsonl: manifest" -- python3 "$TOOL" verify "$dir" --pair "$PAIR"

st_done
