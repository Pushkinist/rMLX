#!/usr/bin/env bash
# Capture a Metal GPU trace of a bounded window of steady-state decode, for
# replay in Xcode (Instruments -> Metal System Trace / GPU counter statistics).
#
# Why this exists: on M5 the Neural Accelerator is part of the GPU, so profiling
# nax needs no special tooling — the ordinary Metal capture path covers it
# (ml-explore/mlx#3182). MSL kernel questions are answered GPU-side, which host
# stack sampling (samply) cannot show: the bundle names the pipelines the window
# ran, and replaying it in Xcode with Profile produces the counters.
#
# The capture window matters: a whole run is unusably large and dominated by
# load + prefill, which is not what we are studying. The engine opens the scope
# after --skip decode steps and closes it --steps later, so the trace holds
# steady-state decode and nothing else.
#
# Keep --steps at 8 or more. The decode loop is pipelined, so a step's work
# straddles the window boundary: a 1-step window's kernel set is a strict subset
# of an 8-step one (it misses the gather_front* embedding lookups), and reading
# it as "the kernels decode runs" is wrong.
#
# Prerequisites (checked below, each with the fix):
#   - Xcode (not just Command Line Tools) selected via xcode-select
#   - a binary built with the metal-capture feature (the flags do not exist
#     otherwise — a release build cannot capture at all)
#   - MTL_CAPTURE_ENABLED=1 in the child environment; Metal inserts the capture
#     layer at launch and cannot do so afterwards. This script sets it.
#
# Usage:
#   bash scripts/gpu_capture.sh --kv-quant iso3_sym --model /path/to/snapshot
#   bash scripts/gpu_capture.sh --kv-quant none --model ... --prompt-tokens 4096 \
#       --skip 4 --steps 8
#
# Output: a .gputrace bundle under .rmlx/traces/, ready to open in Xcode.

set -uo pipefail

KV_QUANT=""
MODEL=""
PROMPT_TOKENS=4096
SKIP=4
STEPS=8
GEN=""
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
	--skip)
		SKIP="$2"
		shift 2
		;;
	--steps)
		STEPS="$2"
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
	echo "usage: $0 --kv-quant <codec> --model <snapshot-abs-path>" >&2
	echo "       [--prompt-tokens N] [--skip N] [--steps N] [--gen N] [--out-dir DIR]" >&2
	exit 2
fi

cd "$(dirname "$0")/.." || exit 1

# The engine needs skip + steps + 2 decode steps to open, fill and close the
# window; a couple more keeps a short EOS from truncating it.
min_gen=$((SKIP + STEPS + 2))
if [ -z "$GEN" ]; then
	GEN=$((min_gen + 4))
elif [ "$GEN" -lt "$min_gen" ]; then
	echo "ERROR: --gen $GEN cannot hold a $SKIP-skip / $STEPS-step window." >&2
	echo "  Raise it to at least $min_gen, or shrink --skip / --steps." >&2
	exit 2
fi

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
# The --gpu-capture flags are compiled out without it, so their absence from
# --help is the authoritative check: a release binary cannot capture at all.
BIN="target/release-debug/rmlx"
BUILD_HINT="cargo build --profile release-debug --features rmlx-cli/metal-capture"
if [ ! -x "$BIN" ]; then
	echo "ERROR: $BIN not found." >&2
	echo "  Build it with the capture feature and full debug info:" >&2
	echo "    $BUILD_HINT" >&2
	exit 1
fi
if ! "$BIN" baseline --help 2>/dev/null | grep -q -- '--gpu-capture'; then
	echo "ERROR: $BIN was built WITHOUT the metal-capture feature." >&2
	echo "  It has no --gpu-capture flag and cannot write a trace. Rebuild with:" >&2
	echo "    $BUILD_HINT" >&2
	exit 1
fi

mkdir -p "$OUT_DIR"
stamp=$(date +%Y%m%d-%H%M%S)
# The model goes in the name: bundles are multi-GB and land side by side, and
# without it two runs of the same codec and prompt size are indistinguishable
# short of reverse-engineering kernel names out of the archive.
model_tag=$(basename "${MODEL%/}")
trace="$OUT_DIR/${model_tag}-${KV_QUANT}-${PROMPT_TOKENS}tok-${stamp}.gputrace"

# Give the KV ring room for the prompt plus the generation, so the run is not
# rejected for context before it ever decodes.
max_ctx=$((PROMPT_TOKENS + GEN + 512))

echo "capturing: codec=$KV_QUANT prompt=$PROMPT_TOKENS skip=$SKIP steps=$STEPS gen=$GEN"
echo "trace:     $trace"

# MTL_CAPTURE_ENABLED is Apple's — Metal reads it at launch to insert the
# capture layer, and there is no in-process way to add it later. It is not an
# rMLX configuration knob: the trace path and the window come from CLI flags.
# --metrics off keeps a capture-distorted run out of runs.db.
MTL_CAPTURE_ENABLED=1 \
	"$BIN" --metrics off baseline \
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

# The engine already fails loudly on every way a capture can not happen, so a
# zero exit here means the bundle exists. Re-check anyway: this script is the
# thing an operator runs, and a missing bundle must never read as success.
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
echo "What this bundle holds, and what it does not:"
echo "  - kernel identity is in the bundle already: which pipelines the window"
echo "    referenced, and which of them were actually used. No Xcode needed."
echo "  - per-dispatch time, ALU-vs-memory limiter, occupancy and achieved"
echo "    bandwidth are NOT in it. Open it in Xcode and press Profile: that"
echo "    replays the capture on-device with counters enabled and writes them."
echo "  - the gaps between dispatches as they happened in THIS run are not"
echo "    recoverable at all — a replay has the replay's schedule. For host"
echo "    round-trips use:  xcrun xctrace record --template 'Metal System Trace'"
