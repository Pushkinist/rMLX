#!/usr/bin/env bash
# Launch rmlx under a GPU instrument: a Metal capture, a Metal System Trace, or the host preflight
#
# Usage:  bash scripts/profile/capture.sh gputrace --kv-quant <codec> --model <snapshot-abs-path>
#             [--prompt-tokens N] [--skip N] [--steps N] [--gen N] [--out-dir DIR] [--keep-all]
#         bash scripts/profile/capture.sh mst --model <snapshot-abs-path> [--kv-quant <codec>]
#             [--prompt-tokens N] [--max-tokens N] [--time-limit S] [--skip-ms N] [--binary PATH]
#             [--keep N] [--out-dir DIR]
#         bash scripts/profile/capture.sh preflight [--binary PATH]
# Exit:   0 done; 1 refused or failed; 2 usage error (an unknown argument, a flag with no value or
#         with an empty one); 75 the run is kept and its diagnostic record did not land;
#         passthrough of rmlx (gputrace) and of the summariser (mst)
# Set:    -uo — a failed engine run, export or summary is a status the code reads and reports
# Not a mode of: profile/gputrace.sh — that one reads a finished .gputrace bundle; this one launches the engine
#
# gputrace  a Metal capture of a bounded window of steady-state decode, for replay in Xcode. The
#           engine opens the scope after --skip decode steps and closes it --steps later; keep
#           --steps at 8 or more, because the decode loop is pipelined and a shorter window's
#           kernel set is a strict subset (it misses the gather_front* embedding lookups). Binary:
#           target/release-debug/rmlx built with the metal-capture feature (`make build-capture`).
#           After a successful run host/gc.sh bounds the directory; --keep-all skips that.
# mst       a Metal System Trace of a live run, its GPU-interval table exported and summarised.
#           Binary: target/release/rmlx. The bundle holds timing and the CPU->GPU gap, a .gputrace
#           holds kernel names and no timing, so use the two together (profile/gputrace.sh).
#           The newest --keep bundles are retained, on every exit.
# preflight the host prerequisites for opening a capture in Apple's GPU tools: full Xcode selected,
#           developer mode on, and the binary signed with com.apple.security.get-task-allow
#           (profile/rmlx-capture.entitlements). It does not promise per-dispatch timings.
#
# Why mst launches rather than attaches: `xctrace --attach` reports "No configuration information
# received" and exports zero rows, because the Metal instrumentation has to be present at launch.
# Launching means prefill is in the table, so --skip-ms drops the first N ms of this process's GPU
# work; by default it is the prefill_ms the run reported in its own log (profile/prefill_ms.py).
# The export XML is positional with id/ref back-references and <sentinel/> for NULL; parsing is
# done by rmlx_mlx::xctrace, which refuses a layout it cannot align. A recording stopped by
# --time-limit makes xctrace exit non-zero on the ordinary path, so the bundle and the exported
# table decide, never the exit code. Granularity is one row per encoder. See docs/PROFILING.md.
#
# gputrace sets MTL_CAPTURE_ENABLED=1 for the engine alone: Metal inserts the capture layer at
# launch and cannot do so afterwards. Both runs are recorded as diagnostic, so no champion and no
# gate reads their perturbed timings.

set -uo pipefail

usage() {
	cat <<'USAGE'
usage: capture.sh gputrace --kv-quant <codec> --model <snapshot-abs-path>
         [--prompt-tokens N]      prompt size (default 4096)
         [--skip N]               decode steps before the window opens (default 4)
         [--steps N]              decode steps in the window (default 8)
         [--gen N]                tokens to decode (default skip + steps + 6)
         [--out-dir DIR]          default <RMLX_HOME>/traces
         [--keep-all]             do not enforce the trace cap
       capture.sh mst --model <snapshot-abs-path>
         [--kv-quant <codec>]     KV codec (default none)
         [--prompt-tokens N]      4096|8192|16384|32768|65536|131072 (default 4096)
         [--max-tokens N]         tokens to decode (default 400)
         [--time-limit S]         seconds to record (default 12)
         [--skip-ms N]            drop the first N ms of THIS PROCESS's GPU work from the summary.
                                  DEFAULT: the prefill_ms this very run reported in its
                                  `generate: prefill` event, read back from its own log
         [--binary PATH]          rmlx to run (default: the release build)
         [--keep N]               .trace bundles to retain (default 5)
         [--out-dir DIR]          default <RMLX_HOME>/traces/mst
       capture.sh preflight [--binary PATH]
                                  binary to check (default: the release-debug build)
USAGE
}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)" || exit 1
# shellcheck source=scripts/lib/identity.sh
. "$REPO_ROOT/scripts/lib/identity.sh"
# shellcheck source=scripts/lib/rmlx_home.sh
. "$REPO_ROOT/scripts/lib/rmlx_home.sh"

SUBCOMMAND="${1:-}"
BIN="$(rmlx_default_binary "$REPO_ROOT" release-debug)"
KV_QUANT=""
case "$SUBCOMMAND" in
gputrace) FLAGS=" --kv-quant --model --prompt-tokens --skip --steps --gen --out-dir --keep-all " ;;
mst)
	FLAGS=" --model --kv-quant --prompt-tokens --max-tokens --time-limit --skip-ms --binary --keep --out-dir "
	KV_QUANT="none"
	BIN="$(rmlx_default_binary "$REPO_ROOT" release)"
	;;
preflight) FLAGS=" --binary " ;;
-h | --help)
	usage
	exit 0
	;;
*)
	usage >&2
	exit 2
	;;
esac
shift

MODEL=""
# baseline resolves --prompt-tokens to a canonical prompt fixture (prompts/longctx_<n>.json), so
# only the sizes shipped there are valid for mst: 4096, 8192, 16384, 32768, 65536, 131072.
PROMPT_TOKENS=4096
SKIP=4
STEPS=8
GEN=""
MAX_TOKENS=400
TIME_LIMIT=12
SKIP_MS=""
KEEP=5
OUT_DIR=""
KEEP_ALL=0

while [ $# -gt 0 ]; do
	case "$1" in
	-h | --help)
		usage
		exit 0
		;;
	esac
	case "$FLAGS" in
	*" $1 "*) ;;
	*)
		echo "unknown argument: $1" >&2
		usage >&2
		exit 2
		;;
	esac
	if [ "$1" = --keep-all ]; then
		KEEP_ALL=1
		shift
		continue
	fi
	if [ $# -lt 2 ] || [ -z "$2" ]; then
		echo "ERROR: $1 needs a value" >&2
		exit 2
	fi
	case "$1" in
	--kv-quant) KV_QUANT="$2" ;;
	--model) MODEL="$2" ;;
	--prompt-tokens) PROMPT_TOKENS="$2" ;;
	--skip) SKIP="$2" ;;
	--steps) STEPS="$2" ;;
	--gen) GEN="$2" ;;
	--max-tokens) MAX_TOKENS="$2" ;;
	--time-limit) TIME_LIMIT="$2" ;;
	--skip-ms) SKIP_MS="$2" ;;
	--binary) BIN="$2" ;;
	--keep) KEEP="$2" ;;
	--out-dir) OUT_DIR="$2" ;;
	esac
	shift 2
done

cd "$REPO_ROOT" || exit 1

RMLX_HOME="$(rmlx_home "$REPO_ROOT")" || exit 1
TRACES_DIR="$RMLX_HOME/traces"

# The model goes in the name: bundles are multi-GB and land side by side, and without it two runs
# of the same codec and prompt size are indistinguishable short of reading kernel names out of
# the archive. Sets stamp and base.
name_trace() {
	stamp=$(date +%Y%m%d-%H%M%S)
	base="$OUT_DIR/$(basename "${MODEL%/}")-${KV_QUANT}-${PROMPT_TOKENS}tok-${stamp}"
}

require_binary() { # <path> <how to build it>
	if [ ! -x "$1" ]; then
		echo "ERROR: $1 not found or not executable." >&2
		echo "  Build it with: $2" >&2
		exit 1
	fi
}

pass() { echo "  ok    $1"; }
warn() { echo "  warn  $1"; }
bad() {
	echo "  FAIL  $1" >&2
	shift
	for line in "$@"; do echo "        $line" >&2; done
	fail=1
}

# Every failure names its exact fix. Returns 1 when a prerequisite is unmet; warnings do not fail.
host_preflight() {
	fail=0
	echo "gputrace preflight (binary: $BIN)"

	# Command Line Tools ship no GPU tools, so a trace cannot be opened at all.
	dev_dir=$(xcode-select -p 2>/dev/null)
	if [ -z "$dev_dir" ]; then
		bad "xcode-select has no developer directory selected." \
			"sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
	elif [ "${dev_dir##*/}" = "CommandLineTools" ]; then
		bad "xcode-select points at Command Line Tools, not Xcode ($dev_dir)." \
			"sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
	else
		pass "Xcode selected: $dev_dir"
	fi

	# Without developer mode the GPU tools cannot attach. An output this cannot read is reported
	# as unreadable, never taken for "disabled" or "fine".
	dts=$(DevToolsSecurity -status 2>&1)
	case "$dts" in
	*"currently enabled"*) pass "developer mode enabled" ;;
	*"currently disabled"*)
		bad "developer mode is disabled — Apple's GPU tools cannot attach at all." \
			"sudo DevToolsSecurity -enable"
		;;
	*)
		bad "could not read developer-mode status; DevToolsSecurity said: ${dts:-<no output>}" \
			"Check it by hand:  DevToolsSecurity -status" \
			"Enable it with:    sudo DevToolsSecurity -enable"
		;;
	esac

	# Cargo's linker-signed ad-hoc signature carries no entitlements, so a fresh build fails this
	# until `make build-capture` re-signs it.
	SIGN_HINT_A="make build-capture     # builds and re-signs with the entitlement"
	SIGN_HINT_B="or: codesign --force --sign - --entitlements scripts/profile/rmlx-capture.entitlements $BIN"
	if [ ! -x "$BIN" ]; then
		bad "capture binary not found at $BIN." "$SIGN_HINT_A"
	else
		ents=$(codesign -d --entitlements - --xml "$BIN" 2>/dev/null)
		cs_rc=$?
		# The key is dotted and plutil reads dots as key-path separators, so it has to be escaped:
		# unescaped, plutil reports "no value at that key path" for a binary that is entitled.
		gta_key='com\.apple\.security\.get-task-allow'
		if [ $cs_rc -ne 0 ]; then
			bad "codesign could not read $BIN (exit $cs_rc) — is it a signed Mach-O?" "$SIGN_HINT_A"
		elif [ -z "$ents" ]; then
			bad "$BIN carries no entitlements at all (cargo's linker-signed ad-hoc signature)." \
				"Apple's GPU tools may not attach to a process that is not marked" \
				"debuggable, so a capture from it is not usable in them. Re-sign it:" \
				"$SIGN_HINT_A" \
				"$SIGN_HINT_B"
		elif ! printf '%s' "$ents" | plutil -lint - >/dev/null 2>&1; then
			bad "codesign printed an entitlement blob that is not a readable plist." \
				"Inspect it by hand:  codesign -d --entitlements - $BIN"
		else
			gta=$(printf '%s' "$ents" | plutil -extract "$gta_key" raw - 2>/dev/null)
			if [ "$gta" = "true" ]; then
				pass "get-task-allow entitlement present"
			elif [ -n "$gta" ]; then
				bad "$BIN has com.apple.security.get-task-allow set to '$gta', not true." \
					"$SIGN_HINT_A"
			else
				bad "$BIN carries no com.apple.security.get-task-allow entitlement." \
					"Apple's GPU tools may not attach to a process that is not marked" \
					"debuggable, so a capture from it is not usable in them. Re-sign it:" \
					"$SIGN_HINT_A" \
					"$SIGN_HINT_B"
			fi
		fi
	fi

	# Advisory: anything that recompiles the captured pipelines needs it, but capture does not.
	if xcrun -sdk macosx metal --version >/dev/null 2>&1; then
		pass "Metal toolchain present: $(xcrun -sdk macosx metal --version 2>&1 | sed -n 1p)"
	else
		warn "Metal toolchain not installed — anything recompiling captured shaders will fail."
		warn "Install it with:  xcodebuild -downloadComponent MetalToolchain"
	fi

	if [ "$fail" -ne 0 ]; then
		echo "preflight FAILED — fix the items above before capturing." >&2
		return 1
	fi
	echo "preflight ok — Apple's GPU tools can attach to this binary on this host"
}

run_gputrace() {
	if [ -z "$KV_QUANT" ] || [ -z "$MODEL" ]; then
		usage >&2
		exit 2
	fi

	# The engine needs skip + steps + 2 decode steps to open, fill and close the window; a couple
	# more keeps a short EOS from truncating it.
	min_gen=$((SKIP + STEPS + 2))
	if [ -z "$GEN" ]; then
		GEN=$((min_gen + 4))
	elif [ "$GEN" -lt "$min_gen" ]; then
		echo "ERROR: --gen $GEN cannot hold a $SKIP-skip / $STEPS-step window." >&2
		echo "  Raise it to at least $min_gen, or shrink --skip / --steps." >&2
		exit 2
	fi

	# The --gpu-capture flags are compiled out without the feature, so their absence from --help
	# is the authoritative check.
	require_binary "$BIN" "make build-capture     # cargo build --profile release-debug --features rmlx-cli/metal-capture, then sign it"
	if ! "$BIN" baseline --help 2>/dev/null | grep -- '--gpu-capture' >/dev/null; then
		echo "ERROR: $BIN was built WITHOUT the metal-capture feature." >&2
		echo "  It has no --gpu-capture flag and cannot write a trace. Rebuild with:" >&2
		echo "    make build-capture     # cargo build --profile release-debug --features rmlx-cli/metal-capture, then sign it" >&2
		exit 1
	fi

	# Before the run: a capture from an unentitled process or a host without developer mode still
	# writes several GB and only shows the problem as an empty Xcode timeline.
	host_preflight || exit 1

	[ -n "$OUT_DIR" ] || OUT_DIR="$TRACES_DIR"
	mkdir -p "$OUT_DIR" || exit 1
	name_trace
	trace="$base.gputrace"

	# Room for the prompt plus the generation, so the run is not rejected for context before it
	# ever decodes.
	max_ctx=$((PROMPT_TOKENS + GEN + 512))

	echo "capturing: codec=$KV_QUANT prompt=$PROMPT_TOKENS skip=$SKIP steps=$STEPS gen=$GEN"
	echo "trace:     $trace"

	# MTL_CAPTURE_ENABLED is Apple's, read by Metal at launch; it is not an rMLX knob: the trace
	# path and the window come from CLI flags. Capture makes every timing false, hence diagnostic.
	MTL_CAPTURE_ENABLED=1 \
		"$BIN" baseline \
		--record --run-class diagnostic --label "gpu_capture:${stamp}" \
		--model "$MODEL" \
		--kv-quant "$KV_QUANT" \
		--prompt-tokens "$PROMPT_TOKENS" \
		--max-tokens "$GEN" \
		--max-ctx "$max_ctx" \
		--max-prompt-tokens "$((PROMPT_TOKENS + 64))" \
		--gpu-capture "$trace" \
		--gpu-capture-skip "$SKIP" \
		--gpu-capture-steps "$STEPS"
	rc=$?

	# shellcheck source=scripts/lib/record_pending.sh
	. scripts/lib/record_pending.sh
	# A diagnostic record that did not land does not undo the capture: the trace is kept, the
	# retention runs, and the script ends 75 at the end.
	record_pending=0
	if [ $rc -eq "$RECORD_PENDING_EXIT" ]; then
		echo "WARNING: the capture succeeded, but its diagnostic record did not land (buffer file" >&2
		echo "  named above); replay it with \`rmlx metrics record --replay-pending\`" >&2
		record_pending=1
		rc=0
	fi

	# A zero exit means the bundle exists; check anyway, because a missing bundle must never read
	# as success.
	if [ $rc -ne 0 ]; then
		echo "capture run failed (exit $rc)" >&2
		exit $rc
	fi
	if [ ! -e "$trace" ]; then
		echo "ERROR: run reported success but no trace exists at $trace" >&2
		exit 1
	fi

	echo ""
	echo "done: $trace"
	echo "open with:  open '$trace'"
	echo ""

	# Bundles are ~6 GB and a session of A/B captures fills a disk long before anything ages out,
	# so the cap is enforced here rather than left to a later run. Oldest first, never the new
	# bundle, every removal printed.
	if [ "$KEEP_ALL" = "1" ]; then
		echo "retention: --keep-all, cap not enforced ($(ls -d "$OUT_DIR"/*.gputrace 2>/dev/null | wc -l | tr -d ' ') bundles in $OUT_DIR)"
	else
		bash scripts/host/gc.sh traces --apply --dir "$OUT_DIR"
	fi
	echo ""
	echo "Offline, no Xcode:"
	echo "  bash scripts/profile/gputrace.sh summary '$trace'"
	echo "  bash scripts/profile/gputrace.sh kernels '$trace'"
	echo "  bash scripts/profile/gputrace.sh diff <other.gputrace> '$trace'"
	echo ""
	echo "What this bundle holds, and what it does not:"
	echo "  - kernel identity is in the bundle already: which pipelines the window"
	echo "    referenced, and which of them were actually used. No Xcode needed."
	echo "  - no timing of any kind. Only Xcode's GUI Profile replay writes a"
	echo "    .gpuprofiler_raw, and the per-dispatch counters people want are not"
	echo "    supported on this GPU. An empty Xcode timeline is expected here."
	echo "  - for wall-clock GPU time and CPU->GPU gaps, record the live process:"
	echo "      bash scripts/profile/capture.sh mst --model <snapshot> --kv-quant $KV_QUANT"
	echo "    See docs/PROFILING.md §5 for the export and its parsing traps."
	if [ "$record_pending" = 1 ]; then
		echo "INCOMPLETE: the trace stands, its diagnostic record is pending (named above)" >&2
		exit "$RECORD_PENDING_EXIT"
	fi
}

# Unvalidated numbers reach xctrace and rmlx as nonsense that fails much later and much less
# clearly: `--time-limit abc` becomes the literal `abcs`.
require_uint() { # name value
	case "$2" in
	'' | *[!0-9]*)
		echo "ERROR: $1 must be a non-negative integer, got '$2'" >&2
		exit 2
		;;
	esac
}

# Runs on every exit, not only the happy one: the failure paths are the common ones while
# debugging, which is when bundles pile up at ~300-400 MB each. The bundle just written is excluded
# by name, never by position, so KEEP=0 cannot delete it; it occupies one of the KEEP slots only
# when this run got far enough to write one.
prune_traces() {
	rc=$?
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

# --release, not the dev default: a 100 MB-scale scan, run twice, must not make the harness the
# slowest step in the loop it measures.
summarise() { # skip_ms [csv_path]
	if [ -n "${2:-}" ]; then
		cargo run -q --release -p rmlx-mlx --features metal-capture --example gpu_timeline -- \
			--input "$xml" --process "$(basename "$BIN")" --skip-ms "$1" --csv "$2"
	else
		cargo run -q --release -p rmlx-mlx --features metal-capture --example gpu_timeline -- \
			--input "$xml" --process "$(basename "$BIN")" --skip-ms "$1"
	fi
}

run_mst() {
	if [ -z "$MODEL" ]; then
		usage >&2
		exit 2
	fi
	require_uint --time-limit "$TIME_LIMIT"
	require_uint --max-tokens "$MAX_TOKENS"
	require_uint --keep "$KEEP"
	require_uint --prompt-tokens "$PROMPT_TOKENS"
	[ -n "$SKIP_MS" ] && require_uint --skip-ms "$SKIP_MS"

	# An unlisted size is rejected by baseline after the model has loaded; catch it here.
	case "$PROMPT_TOKENS" in
	4096 | 8192 | 16384 | 32768 | 65536 | 131072) ;;
	*)
		echo "ERROR: --prompt-tokens $PROMPT_TOKENS has no prompt fixture." >&2
		echo "  Valid: 4096, 8192, 16384, 32768, 65536, 131072 (prompts/longctx_*.json)" >&2
		exit 2
		;;
	esac

	[ -n "$OUT_DIR" ] || OUT_DIR="$TRACES_DIR/mst"
	mkdir -p "$OUT_DIR" || exit 1
	trace=""
	trap prune_traces EXIT

	require_binary "$BIN" "make build       # cargo build --workspace --release"
	if [ ! -d "$MODEL" ]; then
		echo "ERROR: model snapshot not found: $MODEL" >&2
		exit 1
	fi
	if ! command -v xctrace >/dev/null 2>&1 && ! xcrun -f xctrace >/dev/null 2>&1; then
		echo "ERROR: xctrace not available. Select full Xcode:" >&2
		echo "  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
		exit 1
	fi

	# A single MLX process per Mac. A co-resident server also submits to the GPU, and Metal System
	# Trace records the whole system, so its rows would land in this table under another process.
	# Refuse and stop nothing: a process this script did not start is not its to stop. The probe
	# takes the Metal claim and releases it at once, or exits 11 and names the holder.
	if ! "$BIN" claim run -- true; then
		echo "ERROR: another process holds the Metal claim — this trace needs the GPU to itself." >&2
		echo "Stop it first, by the PID the refusal above names." >&2
		exit 1
	fi

	name_trace
	trace="${base}.trace"
	xml="${base}.gpu-intervals.xml"
	csv="${base}.channels.csv"

	# Room for the prompt plus the generation: baseline defaults --max-ctx to 4096 and would
	# otherwise reject the prefill and report a zero decode rate.
	max_ctx=$((PROMPT_TOKENS + MAX_TOKENS + 512))

	# `baseline` records at exit, so a run that --time-limit stops before it finishes leaves no row.
	label="mst_capture:${stamp}"
	run_cmd=("$BIN" baseline
		--record --run-class diagnostic --label "$label"
		--model "$MODEL"
		--kv-quant "$KV_QUANT"
		--prompt-tokens "$PROMPT_TOKENS"
		--max-tokens "$MAX_TOKENS"
		--max-ctx "$max_ctx")
	run_text="${run_cmd[*]+"${run_cmd[*]}"}"

	echo "recording:  model=$(basename "${MODEL%/}") codec=$KV_QUANT prompt=$PROMPT_TOKENS gen=$MAX_TOKENS limit=${TIME_LIMIT}s"
	echo "trace:      $trace"
	# xctrace --launch takes the child's stdout and stderr with it, so a run that refuses its
	# arguments fails invisibly and shows up only as a table with no rows for this process.
	echo "running:    $run_text"

	# xctrace takes the child's exit code with its output, so a record that did not land
	# (`baseline` exit 75) is found as the buffer file in the pending queue whose record carries
	# this run's label. The label holds the run's start second, so no earlier run can carry it.
	pending_dir="$RMLX_HOME/metrics/buffer/pending"

	# RMLX_HOME is passed explicitly so the child writes its log and its buffer file where this
	# script reads them.
	xcrun xctrace record \
		--template 'Metal System Trace' \
		--no-prompt \
		--output "$trace" \
		--time-limit "${TIME_LIMIT}s" \
		--env "RMLX_HOME=$RMLX_HOME" \
		--launch -- ${run_cmd[@]+"${run_cmd[@]}"}
	rc=$?

	pending_buffer=""
	for f in "$pending_dir"/*.json; do
		[ -e "$f" ] || continue
		python3 -c 'import json, sys; notes = json.load(open(sys.argv[1])).get("notes") or ""; sys.exit(notes.split("; ")[0] != "label=" + sys.argv[2])' \
			"$f" "$label" 2>/dev/null && pending_buffer="${pending_buffer:+$pending_buffer }$f"
	done
	if [ -n "$pending_buffer" ]; then
		echo "record pending: $pending_buffer" >&2
		echo "  the run's diagnostic record did not land; replay it with" >&2
		echo "  \`rmlx metrics record --replay-pending\` once the cause is fixed" >&2
	fi
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

	# Nothing in the table marks where prefill ends. The run knows: `generate: prefill` is an info
	# event in its own log. The log is the newest one whose `rmlx start` event names this run's
	# label, never merely the newest: every rmlx process writes a log.
	if [ -z "$SKIP_MS" ]; then
		log_file=""
		for f in "$RMLX_HOME"/logs/*.jsonl; do
			[ -e "$f" ] && grep -qF "$label" "$f" || continue
			[ -z "$log_file" ] || [ "$f" -nt "$log_file" ] && log_file="$f"
		done
		if [ -n "$log_file" ]; then
			SKIP_MS=$(python3 "$REPO_ROOT/scripts/profile/prefill_ms.py" "$log_file")
		fi
		if [ -n "$SKIP_MS" ]; then
			echo "skip:       ${SKIP_MS} ms — measured prefill_ms, read back from this run's log"
		else
			SKIP_MS=0
			echo "WARNING: no generate: prefill event in ${log_file:-<no log>}; the" >&2
			echo "  summary below INCLUDES prefill. Pass --skip-ms explicitly." >&2
		fi
	fi

	# The full window is what the within-run cross-check needs (its span should be close to
	# prefill_ms + step_total_ms from the same log); the decode-only one is what kernel questions
	# are asked of.
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
	# The parser refuses a misaligned or empty table rather than printing zeros, so its exit code
	# is load-bearing here.
	if [ $rc -ne 0 ]; then
		echo "ERROR: summarising $xml failed (exit $rc)" >&2
		echo "  Read WHICH refusal it printed — they mean different things:" >&2
		echo "  * 'contains no rows'  -> the recording captured nothing from any" >&2
		echo "    process. The recording failed; the workload is not implicated." >&2
		echo "  * 'holds N rows but none for a process matching ...' -> the" >&2
		echo "    recording ran and this process was not in it. It lists the" >&2
		echo "    processes it did see. Re-run the workload directly to check it" >&2
		echo "    reaches the GPU at all:" >&2
		echo "      $run_text" >&2
		echo "  * 'the N ms skip removed every one of the M rows ...' -> the" >&2
		echo "    process IS in the recording and --skip-ms cut past its work." >&2
		echo "    Lower --skip-ms or raise --time-limit; do not re-run to debug" >&2
		echo "    the workload." >&2
		exit $rc
	fi

	echo ""
	echo "bundle: $trace"
	echo "table:  $xml"
	echo "csv:    $csv"
	echo ""
	echo "This table has no kernel names — pair it with a capture when you need to"
	echo "know WHICH kernel: bash scripts/profile/gputrace.sh kernels <bundle>.gputrace"
	if [ -n "$pending_buffer" ]; then
		echo "" >&2
		echo "INCOMPLETE: the trace stands, its diagnostic record is pending: $pending_buffer" >&2
		exit 75
	fi
}

case "$SUBCOMMAND" in
gputrace) run_gputrace ;;
mst) run_mst ;;
preflight)
	host_preflight
	exit $?
	;;
esac
