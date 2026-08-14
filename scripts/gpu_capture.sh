#!/usr/bin/env bash
# Capture a Metal GPU trace of a bounded window of steady-state decode, for
# replay in Xcode (Instruments -> Metal System Trace / GPU counter statistics).
#
# Why this exists: on M5 the Neural Accelerator is part of the GPU, so profiling
# nax needs no special tooling — the ordinary Metal capture path covers it
# (ml-explore/mlx#3182). For MSL kernel questions we need GPU-side counters
# (occupancy, ALU vs memory bound, achieved bandwidth, dispatch gaps), which
# host stack sampling (samply) cannot show.
#
# The capture window matters: a whole run is unusably large and dominated by
# load + prefill, which is not what we are studying. This drives a short
# generation so the trace is mostly steady-state decode.
#
# Prerequisites (checked below):
#   - Xcode (not just Command Line Tools) selected via xcode-select
#   - a binary built with the metal-capture feature
#
# Usage:
#   bash scripts/gpu_capture.sh --kv-quant iso3_sym --model /path/to/snapshot
#   bash scripts/gpu_capture.sh --kv-quant none --model ... --prompt-tokens 4096 --gen 16
#
# Output: a .gputrace bundle under .rmlx/traces/, ready to open in Xcode.

set -uo pipefail

KV_QUANT=""
MODEL=""
PROMPT_TOKENS=4096
GEN=16
OUT_DIR=".rmlx/traces"

while [ $# -gt 0 ]; do
	case "$1" in
	--kv-quant)
		KV_QUANT="$2"
		shift 2
		;;
	--model)
		MODEL="$2"
		shift 2
		;;
	--prompt-tokens)
		PROMPT_TOKENS="$2"
		shift 2
		;;
	--gen)
		GEN="$2"
		shift 2
		;;
	--out-dir)
		OUT_DIR="$2"
		shift 2
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done

if [ -z "$KV_QUANT" ] || [ -z "$MODEL" ]; then
	echo "usage: $0 --kv-quant <codec> --model <snapshot-abs-path> [--prompt-tokens N] [--gen N]" >&2
	exit 2
fi

cd "$(dirname "$0")/.." || exit 1

# --- 1. Xcode must be selected; Command Line Tools alone cannot replay traces --
dev_dir=$(xcode-select -p 2>/dev/null)
if [ "${dev_dir##*/}" = "CommandLineTools" ]; then
	echo "ERROR: xcode-select points at Command Line Tools, not Xcode." >&2
	echo "  GPU traces need the full Xcode toolchain. Select it with:" >&2
	echo "    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
	exit 1
fi

# --- 2. toolchain sanity: same gate the bench path uses ---------------------
bash scripts/mlx_preflight.sh || exit 1

# --- 3. binary must carry the capture feature ------------------------------
# CaptureScope is behind `metal-capture` so release builds pay nothing; a binary
# built without it silently produces no trace.
BIN="target/release-debug/rmlx"
if [ ! -x "$BIN" ]; then
	echo "ERROR: $BIN not found." >&2
	echo "  Build it with the capture feature and full debug info:" >&2
	echo "    cargo build --profile release-debug --features rmlx-mlx/metal-capture" >&2
	exit 1
fi

mkdir -p "$OUT_DIR"
stamp=$(date +%Y%m%d-%H%M%S)
trace="$OUT_DIR/${KV_QUANT}-${PROMPT_TOKENS}tok-${stamp}.gputrace"

echo "capturing: codec=$KV_QUANT prompt=$PROMPT_TOKENS gen=$GEN"
echo "trace:     $trace"

# RMLX_METAL_CAPTURE_PATH is read by the engine's capture hook; the window is
# opened after prefill so the trace is steady-state decode, not model load.
RMLX_METAL_CAPTURE_PATH="$trace" \
	"$BIN" baseline \
	--model "$MODEL" \
	--kv-quant "$KV_QUANT" \
	--prompt-tokens "$PROMPT_TOKENS" \
	--max-tokens "$GEN" \
	--max-ctx 65536 \
	--max-prompt-tokens 65528
rc=$?

if [ $rc -ne 0 ]; then
	echo "capture run failed (exit $rc)" >&2
	exit $rc
fi

if [ ! -e "$trace" ]; then
	echo "ERROR: no trace written to $trace" >&2
	echo "  Was the binary built with --features rmlx-mlx/metal-capture?" >&2
	exit 1
fi

echo ""
echo "done: $trace"
echo "open with:  open '$trace'"
echo ""
echo "In Xcode, the questions this trace should answer:"
echo "  - per-dispatch kernel time for the flash-decode kernel vs the gaps between dispatches"
echo "    (a large gap = host round-trip, e.g. the k_iso*/k_rotor* prefix restage)"
echo "  - limiter counters: ALU vs memory bound, occupancy, achieved bandwidth"
echo "    (tells whether a quant decode kernel is compute- or bandwidth-limited)"
