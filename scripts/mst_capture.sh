#!/usr/bin/env bash
# Record a Metal System Trace of a live rmlx run, export the GPU-interval table
# and summarise it.
#
# WHY THIS AND NOT A .gputrace
#   A `.gputrace` names the kernels a decode window ran but carries no timing of
#   any kind, and only Xcode's GUI Profile replay would add any — against the
#   replay's schedule, not the run's. Host round-trips (a blocking per-layer
#   Array::eval, a per-step restage) therefore cannot appear in one at all.
#   Metal System Trace records the live process and does, headlessly, with
#   nanosecond resolution. Its `metal-gpu-intervals` table gives per GPU
#   submission `start`, `duration`, and `start-latency` — the CPU->GPU gap, the
#   signal nothing else on this hardware exposes.
#
#   Use it WITH `scripts/gputrace_kernels.sh`, not instead: the export carries no
#   pipeline or function names, so "which kernel" still comes from a capture.
#
# THREE TRAPS, ALL HANDLED HERE
#   1. `--attach <pid>` does not work for this template. It reports "No
#      configuration information received, will have to guess" and exports zero
#      rows: the Metal instrumentation has to be present at launch. This script
#      uses `--launch --`, which is also why the recording necessarily starts at
#      process launch. Weight load submits no GPU work so it leaves no rows, but
#      prefill does: --skip-ms drops the first N ms of THIS PROCESS's GPU work to
#      leave a decode-only window.
#   2. The export XML is positional with `id`/`ref` back-references and
#      `<sentinel/>` for NULL, and a naive reader silently misaligns columns —
#      producing plausible wrong numbers rather than an error. Parsing is done
#      by rmlx_mlx::xctrace, which refuses a layout it cannot align.
#   3. Volume. An 8 s trace is a ~145 MB bundle and tens of MB of XML for one
#      table. --time-limit bounds the recording and the newest --keep bundles
#      are retained; the rest are removed after a successful run.
#
# WHAT IT CANNOT GIVE — do not plan around these; they are device ceilings.
#   No per-dispatch kernel timing (supportsCounterSampling(atDispatchBoundary)
#   is false), no occupancy / limiter / bandwidth counters (one counter set,
#   GPUTimestamp), and no pipeline or function names in the export. The driver
#   also coalesces consecutive compute encoders into one GPU kick, so one row
#   can cover several encoders.
#
# USAGE
#   bash scripts/mst_capture.sh --model /path/to/snapshot
#   bash scripts/mst_capture.sh --model ... --kv-quant k8v8 --time-limit 12 \
#       --prompt-tokens 4096 --max-tokens 400 --skip-ms 4000
#
# Exit 0 = a bundle, an export and a summary. Anything else is an error; an
# empty or unparseable table is never reported as a run with no GPU work.

set -uo pipefail

MODEL=""
KV_QUANT="none"
# baseline resolves this to a canonical prompt fixture (prompts/longctx_<n>.json),
# so only the sizes shipped there are valid: 4096, 8192, 16384, 32768, 65536, 131072.
PROMPT_TOKENS=4096
MAX_TOKENS=400
TIME_LIMIT=12
# Empty means "not supplied" — the run's own measured prefill is used instead.
SKIP_MS=""
BIN="target/release/rmlx"
KEEP=5
OUT_DIR=""

usage() {
	cat <<'USAGE'
usage: mst_capture.sh --model <snapshot-abs-path>
         [--kv-quant <codec>]     KV codec (default none)
         [--prompt-tokens N]      4096|8192|16384|32768|65536|131072 (default 4096)
         [--max-tokens N]         tokens to decode (default 400)
         [--time-limit S]         seconds to record (default 12)
         [--skip-ms N]            drop the first N ms of THIS PROCESS's GPU work
                                  from the summary. DEFAULT: the prefill_ms this
                                  very run reported, read back from its own log,
                                  so the decode window needs no guessing. Weight
                                  load submits nothing and is absent already.
         [--binary PATH]          rmlx to run (default target/release/rmlx)
         [--keep N]               .trace bundles to retain (default 5)
         [--out-dir DIR]          default <RMLX_HOME>/traces/mst
USAGE
}

while [ $# -gt 0 ]; do
	case "$1" in
	--model) MODEL="${2:?--model needs a value}"; shift 2 ;;
	--kv-quant) KV_QUANT="${2:?--kv-quant needs a value}"; shift 2 ;;
	--prompt-tokens) PROMPT_TOKENS="${2:?--prompt-tokens needs a value}"; shift 2 ;;
	--max-tokens) MAX_TOKENS="${2:?--max-tokens needs a value}"; shift 2 ;;
	--time-limit) TIME_LIMIT="${2:?--time-limit needs a value}"; shift 2 ;;
	--skip-ms) SKIP_MS="${2:?--skip-ms needs a value}"; shift 2 ;;
	--binary) BIN="${2:?--binary needs a value}"; shift 2 ;;
	--keep) KEEP="${2:?--keep needs a value}"; shift 2 ;;
	--out-dir) OUT_DIR="${2:?--out-dir needs a value}"; shift 2 ;;
	-h | --help) usage; exit 0 ;;
	*) echo "ERROR: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
	esac
done

if [ -z "$MODEL" ]; then
	usage >&2
	exit 2
fi

# Unvalidated numbers reach xctrace and rmlx as nonsense that fails much later
# and much less clearly — `--time-limit abc` becomes the literal `abcs`.
require_uint() { # name value
	case "$2" in
	'' | *[!0-9]*)
		echo "ERROR: $1 must be a non-negative integer, got '$2'" >&2
		exit 2
		;;
	esac
}
require_uint --time-limit "$TIME_LIMIT"
require_uint --max-tokens "$MAX_TOKENS"
require_uint --keep "$KEEP"
require_uint --prompt-tokens "$PROMPT_TOKENS"
[ -n "$SKIP_MS" ] && require_uint --skip-ms "$SKIP_MS"

# baseline resolves --prompt-tokens to a checked-in fixture, so an unlisted size
# is rejected after the model has loaded. Catch it here instead.
case "$PROMPT_TOKENS" in
4096 | 8192 | 16384 | 32768 | 65536 | 131072) ;;
*)
	echo "ERROR: --prompt-tokens $PROMPT_TOKENS has no prompt fixture." >&2
	echo "  Valid: 4096, 8192, 16384, 32768, 65536, 131072 (prompts/longctx_*.json)" >&2
	exit 2
	;;
esac

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

RMLX_HOME_DIR="${RMLX_HOME:-$PWD/.rmlx}"
[ -n "$OUT_DIR" ] || OUT_DIR="$RMLX_HOME_DIR/traces/mst"
mkdir -p "$OUT_DIR" || exit 1

# Retention runs on EVERY exit, not only the happy one. The failure paths below
# — bad arguments, a run that produced no rows — are the common ones while
# debugging, which is exactly when bundles pile up at ~300-400 MB each, and a
# bound that only applies to successful runs is not a bound.
#
# The bundle just written is excluded BY NAME. Guarding by ordinal position
# alone deletes it whenever KEEP is 0, which `make profile-mst KEEP=0` reaches:
# GNU make treats the string "0" as true, so `$(if 0,...)` passes it straight
# through. KEEP is validated as an integer above, so `[` cannot fall through to
# the delete on a non-numeric value either.
prune_traces() {
	rc=$?
	# The bundle just written occupies one of the KEEP slots — but only if this
	# run got far enough to write one. Reserving it unconditionally would prune
	# one bundle too many on every preflight failure.
	limit="$KEEP"
	if [ -n "${trace:-}" ] && [ -e "${trace:-}" ]; then
		limit=$((KEEP > 0 ? KEEP - 1 : 0))
	fi
	kept=0
	while IFS= read -r old; do
		[ -n "$old" ] || continue
		[ "$old" = "${trace:-}" ] && continue
		kept=$((kept + 1))
		[ "$kept" -le "$limit" ] && continue
		old_base="${old%.trace}"
		size=$(du -sh "$old" 2>/dev/null | cut -f1)
		rm -rf "$old" "${old_base}.gpu-intervals.xml" "${old_base}.channels.csv"
		echo "retention: removed $(basename "$old") (${size:-?}, beyond the newest $KEEP)"
	done < <(ls -1dt "$OUT_DIR"/*.trace 2>/dev/null)
	return "$rc"
}
trap prune_traces EXIT

if [ ! -x "$BIN" ]; then
	echo "ERROR: $BIN not found or not executable." >&2
	echo "  Build it with: make build       # cargo build --workspace --release" >&2
	exit 1
fi
if [ ! -d "$MODEL" ]; then
	echo "ERROR: model snapshot not found: $MODEL" >&2
	exit 1
fi
if ! command -v xctrace >/dev/null 2>&1 && ! xcrun -f xctrace >/dev/null 2>&1; then
	echo "ERROR: xctrace not available. Select full Xcode:" >&2
	echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
	exit 1
fi

# CLAUDE.md hard rule 8 — a single MLX process per Mac. A co-resident server
# also submits to the GPU, and Metal System Trace records the whole system, so
# its rows would land in this table under a different process and any timing
# read here would be unattributable. Refuse rather than pkill: killing a process
# this script does not own is not its call.
if pgrep -f 'rmlx serve|mlx_lm|paroquant|omlx' >/dev/null 2>&1; then
	echo "ERROR: another MLX process is live — this trace needs the GPU to itself." >&2
	pgrep -fl 'rmlx serve|mlx_lm|paroquant|omlx' >&2 || true
	echo "Stop it first: pkill -f 'rmlx serve'; pkill -f mlx_lm; rm -f /tmp/rmlx.*.claim" >&2
	exit 1
fi


stamp=$(date +%Y%m%d-%H%M%S)
model_tag=$(basename "${MODEL%/}")
base="$OUT_DIR/${model_tag}-${KV_QUANT}-${PROMPT_TOKENS}tok-${stamp}"
trace="${base}.trace"
xml="${base}.gpu-intervals.xml"
csv="${base}.channels.csv"

# Room for the prompt plus the generation: baseline defaults --max-ctx to 4096
# and would otherwise reject the prefill and report a zero decode rate.
max_ctx=$((PROMPT_TOKENS + MAX_TOKENS + 512))

run_cmd=("$BIN" --metrics off baseline
	--model "$MODEL"
	--kv-quant "$KV_QUANT"
	--prompt-tokens "$PROMPT_TOKENS"
	--max-tokens "$MAX_TOKENS"
	--max-ctx "$max_ctx")

echo "recording:  model=$model_tag codec=$KV_QUANT prompt=$PROMPT_TOKENS gen=$MAX_TOKENS limit=${TIME_LIMIT}s"
echo "trace:      $trace"
# xctrace --launch takes the child's stdout and stderr with it, so a run that
# refuses its arguments fails invisibly and shows up only as a table with no
# rows for this process. Print the command so it can be re-run by hand.
echo "running:    ${run_cmd[*]}"

# --launch, not --attach (trap 1). --metrics off because a traced run's numbers
# are perturbed and must never reach runs.db.
xcrun xctrace record \
	--template 'Metal System Trace' \
	--no-prompt \
	--output "$trace" \
	--time-limit "${TIME_LIMIT}s" \
	--launch -- "${run_cmd[@]}"
rc=$?
# A bounded recording ends by killing the launched process, so xctrace reports
# the child's termination as a non-zero exit on the ordinary, successful path.
# The exit code therefore cannot be the gate; the bundle and the exported table
# are. A run that genuinely failed leaves no rows for this process, and the
# summariser refuses that rather than printing zeros.
if [ $rc -ne 0 ]; then
	echo "note: xctrace record exited $rc — expected when --time-limit ends a" >&2
	echo "  still-running process. The export below is what decides." >&2
fi
if [ ! -e "$trace" ]; then
	echo "ERROR: no bundle at $trace — the recording did not start." >&2
	exit 1
fi

echo ""
echo "exporting:  metal-gpu-intervals -> $xml"
xcrun xctrace export \
	--input "$trace" \
	--xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
	--output "$xml"
rc=$?
if [ $rc -ne 0 ] || [ ! -s "$xml" ]; then
	echo "ERROR: export produced no table. The recording holds no GPU intervals," >&2
	echo "  which is what --attach produces; this script uses --launch, so check" >&2
	echo "  that the run actually reached the GPU before the time limit." >&2
	exit 1
fi

# Nothing in the GPU-interval table marks where prefill ends — weight load
# submits no work, so the process's first row is already prefill and the
# boundary is invisible. The run itself knows: `decode_profile{prefill_ms}` is a
# plain info! event, so it is in the run's own log at the default level even
# though `xctrace --launch` swallowed the child's stdout.
if [ -z "$SKIP_MS" ]; then
	log_file=$(ls -1t "$RMLX_HOME_DIR"/logs/*.jsonl 2>/dev/null | head -1)
	if [ -n "$log_file" ]; then
		SKIP_MS=$(python3 "$REPO_ROOT/scripts/lib/prefill_ms.py" "$log_file")
	fi
	if [ -n "$SKIP_MS" ]; then
		echo "skip:       ${SKIP_MS} ms — measured prefill_ms, read back from this run's log"
	else
		SKIP_MS=0
		echo "WARNING: no decode_profile{prefill_ms} in ${log_file:-<no log>}; the" >&2
		echo "  summary below INCLUDES prefill. Pass --skip-ms explicitly." >&2
	fi
fi

# --release, not the dev default: a 100 MB-scale scan, run twice below, must not
# make the harness the slowest step in the loop it measures.
summarise() { # skip_ms [csv_path]
	if [ -n "${2:-}" ]; then
		cargo run -q --release -p rmlx-mlx --features metal-capture --example gpu_timeline -- \
			--input "$xml" --process "$(basename "$BIN")" --skip-ms "$1" --csv "$2"
	else
		cargo run -q --release -p rmlx-mlx --features metal-capture --example gpu_timeline -- \
			--input "$xml" --process "$(basename "$BIN")" --skip-ms "$1"
	fi
}

# Both windows are printed. The full one is what the within-run cross-check
# needs (its span should equal prefill_ms + step_total_ms from the same log);
# the decode-only one is what kernel questions are asked of.
echo ""
echo "== full window (prefill included) =============================="
summarise 0
rc=$?
if [ $rc -eq 0 ] && [ "$SKIP_MS" != "0" ]; then
	echo ""
	echo "== decode-only window (first ${SKIP_MS} ms skipped) ============"
	summarise "$SKIP_MS" "$csv"
	rc=$?
elif [ $rc -eq 0 ]; then
	summarise 0 "$csv" >/dev/null
	rc=$?
fi
# The parser refuses a misaligned or empty table rather than printing zeros, so
# its exit code is load-bearing here.
if [ $rc -ne 0 ]; then
	echo "ERROR: summarising $xml failed (exit $rc)" >&2
	echo "  If it reports no rows for this process, the run itself failed and" >&2
	echo "  xctrace swallowed its output. Re-run it directly to see why:" >&2
	echo "    ${run_cmd[*]}" >&2
	exit $rc
fi

echo ""
echo "bundle: $trace"
echo "table:  $xml"
echo "csv:    $csv"
echo ""
echo "This table has no kernel names — pair it with a capture when you need to"
echo "know WHICH kernel: bash scripts/gputrace_kernels.sh <bundle>.gputrace"
