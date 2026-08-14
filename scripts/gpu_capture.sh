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
#   - a binary built with the metal-capture feature (the flags do not exist
#     otherwise — a release build cannot capture at all)
#   - the replay prerequisites, via scripts/gputrace_preflight.sh: Xcode (not
#     just Command Line Tools) selected, developer mode enabled, and the binary
#     signed with com.apple.security.get-task-allow. A capture written without
#     those is several GB that Xcode opens and shows nothing for.
#   - MTL_CAPTURE_ENABLED=1 in the child environment; Metal inserts the capture
#     layer at launch and cannot do so afterwards. This script sets it.
#
# Usage:
#   bash scripts/gpu_capture.sh --kv-quant iso3_sym --model /path/to/snapshot
#   bash scripts/gpu_capture.sh --kv-quant none --model ... --prompt-tokens 4096 \
#       --skip 4 --steps 8
#   bash scripts/gpu_capture.sh ... --keep-all     # do not enforce the trace cap
#
# Output: a .gputrace bundle under .rmlx/traces/, ready to open in Xcode. After
# a successful capture the trace directory is bounded by scripts/traces_gc.sh
# (oldest-first, never the new bundle, every removal printed); --keep-all skips
# that for a session that wants to keep more than the cap.

set -uo pipefail

KV_QUANT=""
MODEL=""
PROMPT_TOKENS=4096
SKIP=4
STEPS=8
GEN=""
OUT_DIR=".rmlx/traces"
KEEP_ALL=0

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
	--keep-all)
		KEEP_ALL=1
		shift
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done

if [ -z "$KV_QUANT" ] || [ -z "$MODEL" ]; then
	echo "usage: $0 --kv-quant <codec> --model <snapshot-abs-path>" >&2
	echo "       [--prompt-tokens N] [--skip N] [--steps N] [--gen N] [--out-dir DIR] [--keep-all]" >&2
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

# --- 1. binary must carry the capture feature ------------------------------
# The --gpu-capture flags are compiled out without it, so their absence from
# --help is the authoritative check: a release binary cannot capture at all.
BIN="target/release-debug/rmlx"
BUILD_HINT="make build-capture     # cargo build --profile release-debug --features rmlx-cli/metal-capture, then sign it"
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

# --- 2. host must be able to REPLAY what we are about to write --------------
# Xcode selected, developer mode on, and the binary signed with
# com.apple.security.get-task-allow. Checked here, before the run, because a
# capture from an unentitled process or a host without developer mode still
# writes several GB and only reveals the problem as an empty Xcode timeline.
bash scripts/gputrace_preflight.sh --binary "$BIN" || exit 1

# --- 3. toolchain sanity: same gate the bench path uses ---------------------
bash scripts/mlx_preflight.sh || exit 1

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

# Bound the collection, right after a successful capture. Bundles are ~6 GB and
# a session of A/B captures fills a disk long before anything "ages out", so the
# cap is enforced here rather than left as an advisory the operator runs later —
# by the time a disk is full the run in flight has already failed. Eviction is
# oldest-first, never the bundle just written, and every removal is printed.
# --keep-all skips it for a session that genuinely wants more than the cap.
if [ "$KEEP_ALL" = "1" ]; then
	echo "retention: --keep-all, cap not enforced ($(ls -d "$OUT_DIR"/*.gputrace 2>/dev/null | wc -l | tr -d ' ') bundles in $OUT_DIR)"
else
	bash scripts/traces_gc.sh --apply --dir "$OUT_DIR"
fi
echo ""
echo "Offline, no Xcode:"
echo "  bash scripts/gputrace_summary.sh '$trace'"
echo "  bash scripts/gputrace_kernels.sh '$trace'"
echo "  bash scripts/gputrace_diff.sh <other.gputrace> '$trace'"
echo ""
echo "What this bundle holds, and what it does not:"
echo "  - kernel identity is in the bundle already: which pipelines the window"
echo "    referenced, and which of them were actually used. No Xcode needed."
echo "  - no timing of any kind. Only Xcode's GUI Profile replay writes a"
echo "    .gpuprofiler_raw, and the per-dispatch counters people want are not"
echo "    supported on this GPU. An empty Xcode timeline is expected here."
echo "  - for wall-clock GPU time and CPU->GPU gaps, record the live process:"
echo "      xcrun xctrace record --template 'Metal System Trace' --no-prompt \\"
echo "        --output run.trace --time-limit 8s --launch -- <rmlx ...>"
echo "    See docs/PROFILING.md §5 for the export and its parsing traps."
