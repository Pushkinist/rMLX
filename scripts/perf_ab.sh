#!/usr/bin/env bash
# perf_ab.sh — ABBA-interleaved A/B comparison of two `rmlx baseline` arms.
#
# Reached as `bash scripts/perf_canary.sh --ab ...`; runnable directly.
#
# WHY THIS EXISTS
#
# The plain canary runs every measured iteration of one arm, then every
# iteration of the other. Ordering and thermal drift are then confounded with
# the effect: if the host warms up, or another process wakes up halfway
# through, the arm that ran second wears the difference. That has twice
# produced a "regression" here that was a busy host.
#
# This harness removes both confounds and refuses to hide a third:
#
#   * Slots alternate in a balanced ABBA / BAAB / ABBA pattern, so a monotone
#     drift across the run contributes equally to both arms.
#   * Foreign CPU use is measured across every slot and across the comparison
#     as a whole, from cumulative CPU time. Anything running at the start of a
#     window, at its end, or throughout taints the result instead of tilting
#     it. A process that both starts and exits inside one window is invisible
#     to this — it appears in neither snapshot — so what is caught is sustained
#     contention, which is the profile that actually skews a decode benchmark.
#     A window that could not be sampled at all is reported as `unmeasured` and
#     taints too; not knowing is never folded into "nothing was there".
#   * The two arms must be provably distinguishable — same binary digest AND
#     same arguments is refused, because "A vs B" where both are the same
#     build is the failure mode that looks most like a real result.
#   * The exact generated token-id sequence is compared across arms in the
#     same invocation that measures speed, so an arm that is fast and wrong
#     fails there rather than in a later correctness pass.
#
# WHAT IT DOES NOT DO
#
# The arms run as separate processes, one per slot. Alternating two kernel
# dispatch paths *inside* one process needs a threaded dispatch-policy value;
# the five kernel selections are latched in `OnceLock` at first read, so a
# process can only ever exercise one of them. Until that lands, an arm is a
# (binary, arguments) pair. Nothing in this harness changes when it does: an
# in-process policy arm is just another argument.
#
# NEVER WRITES TO runs.db. An A/B run is an experiment, not a recorded
# baseline; the append-only metrics store must not accumulate rows from arms
# that were built to be thrown away, and a wrong row there cannot be taken
# back out. Every slot runs with `--metrics off`, which never opens the file,
# and `--metrics` is refused in arm arguments — it is a global flag, so an
# occurrence after the subcommand would win and re-enable recording.
#
# It does write elsewhere: the result lands in
# `$RMLX_HOME/bench/perf_ab/<timestamp>.json`, and every slot is a full `rmlx`
# process that writes its own `$RMLX_HOME/logs/<run-id>.jsonl` and runs the log
# size-cap rotation. A default run is 42 of those. Point RMLX_HOME at a scratch
# directory when that matters.
#
# Exit codes:
#   0   — ran cleanly; the verdict is on stdout
#   1   — correctness failure: the arms produced different token ids
#   125 — the comparison is not usable: a measurement precondition failed
#         (busy host, indistinguishable arms, missing binary/model, unparseable
#         output), or the run completed but was TAINTED by interference.

# No `pipefail`: several parses here legitimately end in `| head -1`, which
# SIGPIPEs its producer once it has what it needs. Under pipefail that reads as
# a failure and aborts the run. Every field this script parses is checked for
# emptiness immediately afterwards and every subprocess exit status is tested,
# so nothing is being waved through here.
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
AWK_BUSIEST="${REPO_ROOT}/scripts/lib/busiest_between.awk"

# ---- defaults ----------------------------------------------------------------

DEFAULT_BINARY="${REPO_ROOT}/target/release-perf/rmlx"
BIN_A=""
BIN_B=""
ARGS_A=""
ARGS_B=""
LABEL_A="A"
LABEL_B="B"
SLOTS=12
INVERT=false
ALLOW_NULL_ARMS=false
ALLOW_BUSY_HOST=false
ALLOW_TOKEN_DIVERGENCE=false
BUSY_PCT=25
MODELS=()

PROMPT_TOKENS=4096
MAX_TOKENS=100
MAX_CTX=8192

usage() {
	cat <<'USAGE'
usage: scripts/perf_ab.sh [options]

Arms (at least one of the two must differ):
  --binary-a PATH        binary for arm A (default: target/release-perf/rmlx)
  --binary-b PATH        binary for arm B (default: same as --binary-a)
  --arm-a "ARGS"         extra `rmlx baseline` arguments for arm A
  --arm-b "ARGS"         extra `rmlx baseline` arguments for arm B
  --label-a NAME         name for arm A in the report (default: A)
  --label-b NAME         name for arm B in the report (default: B)

Protocol:
  --model PATH           model snapshot to compare on; repeatable.
                         Default: the three canary models.
  --slots N              interleaved slots per model (default 12). Must be a
                         multiple of 4, and large enough that the SEPARATED
                         verdict's null probability stays at or under 0.05 --
                         so 8 or more. At 4 it would be 1 in 3.
  --invert               swap the arm roles in the pattern (cancels any residual
                         positional bias when paired with a non-inverted run)
  --prompt-tokens N      default 4096
  --max-tokens N         default 100. Every slot must generate this many tokens;
                         a short generation is refused, not averaged in.
  --max-ctx N            default 8192

Arm arguments may not contain --metrics: it is a global flag, so an occurrence
after the subcommand overrides the --metrics off every slot runs with, and the
slot would write to the append-only runs.db.

Escape hatches (each one weakens a guard; each is reported in the output):
  --allow-null-arms          permit two identical arms (the null calibration)
  --allow-busy-host          start on a busy host and print the numbers anyway.
                             A contaminated run still exits 125 -- this only
                             buys the right to look at it.
  --allow-token-divergence   permit arms that generate different tokens
                             (e.g. two different KV codecs)
  --busy-pct PCT             foreign-process CPU%% that counts as busy (default 25)
USAGE
}

# ---- argument parsing --------------------------------------------------------

need_value() {
	if [[ $2 -lt 2 ]]; then
		echo "ERROR: $1 requires a value" >&2
		exit 125
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--binary-a) need_value "$1" $#; BIN_A="$2"; shift 2 ;;
	--binary-b) need_value "$1" $#; BIN_B="$2"; shift 2 ;;
	--arm-a) need_value "$1" $#; ARGS_A="$2"; shift 2 ;;
	--arm-b) need_value "$1" $#; ARGS_B="$2"; shift 2 ;;
	--label-a) need_value "$1" $#; LABEL_A="$2"; shift 2 ;;
	--label-b) need_value "$1" $#; LABEL_B="$2"; shift 2 ;;
	--model) need_value "$1" $#; MODELS+=("$2"); shift 2 ;;
	--slots) need_value "$1" $#; SLOTS="$2"; shift 2 ;;
	--prompt-tokens) need_value "$1" $#; PROMPT_TOKENS="$2"; shift 2 ;;
	--max-tokens) need_value "$1" $#; MAX_TOKENS="$2"; shift 2 ;;
	--max-ctx) need_value "$1" $#; MAX_CTX="$2"; shift 2 ;;
	--busy-pct) need_value "$1" $#; BUSY_PCT="$2"; shift 2 ;;
	--invert) INVERT=true; shift ;;
	--allow-null-arms) ALLOW_NULL_ARMS=true; shift ;;
	--allow-busy-host) ALLOW_BUSY_HOST=true; shift ;;
	--allow-token-divergence) ALLOW_TOKEN_DIVERGENCE=true; shift ;;
	-h | --help) usage; exit 0 ;;
	*) echo "unknown flag: $1" >&2; usage >&2; exit 125 ;;
	esac
done

BIN_A="${BIN_A:-$DEFAULT_BINARY}"
BIN_B="${BIN_B:-$BIN_A}"

# Numeric options are validated before anything uses them. Two ways an
# unvalidated value goes wrong here, both silent: `$((SLOTS % 4))` on a
# non-numeric string is a fatal `unbound variable` under `set -u`, which exits
# 1 -- the code this script reserves for "the arms produced different tokens",
# so a wrapper reads a typo as a correctness regression. And awk compares a
# strnum against a non-numeric threshold lexically, so `--busy-pct abc` makes
# `80.0 >= abc` false and the interference gate never fires again.
require_number() {
	case "$2" in
	'' | *[!0-9.]* | *.*.*)
		echo "ERROR: $1 must be a number, got '$2'" >&2
		exit 125
		;;
	esac
}
require_number --slots "$SLOTS"
require_number --busy-pct "$BUSY_PCT"
require_number --prompt-tokens "$PROMPT_TOKENS"
require_number --max-tokens "$MAX_TOKENS"
require_number --max-ctx "$MAX_CTX"

if [[ $((SLOTS % 4)) -ne 0 || $SLOTS -lt 4 ]]; then
	echo "ERROR: --slots must be a multiple of 4 and at least 4 (got $SLOTS)" >&2
	echo "  The pattern is built from ABBA/BAAB blocks; a partial block would" >&2
	echo "  give the arms different mean positions and re-introduce the drift" >&2
	echo "  confound this harness exists to remove." >&2
	exit 125
fi

# The null probability of the SEPARATED verdict is 2/C(slots, slots/2). At 4
# slots that is 1 in 3. The word printed for a one-in-three coin flip would be
# the same word printed for a one-in-462 result, and the word is what ends up
# pasted into a report.
NULL_P="$(awk -v n="$SLOTS" -v k="$((SLOTS / 2))" 'BEGIN {
	r = 1; for (i = 1; i <= k; i++) r = r * (n - k + i) / i; printf "%.5f", 2 / r }')"
if awk -v p="$NULL_P" 'BEGIN { exit !(p > 0.05) }'; then
	echo "ERROR: --slots $SLOTS gives the SEPARATED verdict a null probability of $NULL_P." >&2
	echo "  Above 0.05 the verdict is not worth the word: it would carry the same" >&2
	echo "  authority as the same word at --slots 12, where it means 0.00216." >&2
	echo "  Use --slots 8 (0.02857) or more." >&2
	exit 125
fi

# The relative standard error of a sample stddev is ~1/sqrt(2(n-1)) -- 71% at
# n=2, 41% at n=4, 32% at n=6. Computed, not asserted: a hardcoded figure beside
# an interpolated n is an instrument stating a false uncertainty as if derived.
SD_RSE_PCT="$(awk -v n="$((SLOTS / 2))" 'BEGIN { printf "%.0f", 100 / sqrt(2 * (n - 1)) }')"

# Family-wise false-SEPARATED rate: the null probability is per comparison, and
# a run emits one independent verdict per model.
FAMILY_P="$(awk -v p="$NULL_P" -v m="${#MODELS[@]}" 'BEGIN { printf "%.5f", 1 - (1 - p) ^ m }')"

# `--metrics` may not appear in arm arguments. It is declared `global = true`,
# so a subcommand-level occurrence overrides the harness's leading
# `--metrics off` and the slot opens the real append-only runs.db -- verified,
# including on a failure path where the model never loaded. This is refused
# rather than overridden, because silently ignoring a flag the caller passed is
# its own defect.
case " $ARGS_A $ARGS_B " in
*" --metrics "* | *" --metrics="*)
	echo "ERROR: --metrics may not appear in arm arguments." >&2
	echo "  It is a global flag, so an occurrence after the subcommand wins over" >&2
	echo "  the --metrics off this harness passes, and the slot would write to the" >&2
	echo "  append-only runs.db. A row from a discarded arm cannot be removed." >&2
	echo "  arm A args: '$ARGS_A'" >&2
	echo "  arm B args: '$ARGS_B'" >&2
	exit 125
	;;
esac

if [[ ${#MODELS[@]} -eq 0 ]]; then
	: "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT, or pass --model - see .env.example}"
	MODELS=(
		"${RMLX_O_MODELS_ROOT}/prism-ml__Ternary-Bonsai-8B-mlx-2bit"
		"${RMLX_O_MODELS_ROOT}/mlx-community__gemma-4-e4b-it-mxfp8"
		"${RMLX_O_MODELS_ROOT}/mlx-community__Qwen3.6-35B-A3B-8bit"
	)
fi

# ---- preconditions -----------------------------------------------------------

for bin in "$BIN_A" "$BIN_B"; do
	if [[ ! -x "$bin" ]]; then
		echo "ERROR: not an executable binary: $bin" >&2
		echo "  Build it first: make build-perf" >&2
		exit 125
	fi
done

# An empty digest is the one value this must not tolerate: two empty digests
# compare equal, so a failing `shasum` would make the distinguishability guard
# pass for any pair of arms and record "" as each arm's provenance.
digest() {
	local d
	d="$(shasum -a 256 "$1" | cut -c1-16)"
	if [[ -z "$d" ]]; then
		echo "ERROR: could not digest $1 — arm provenance is unverifiable" >&2
		exit 125
	fi
	printf '%s' "$d"
}
SHA_A="$(digest "$BIN_A")"
SHA_B="$(digest "$BIN_B")"

# The arms must be distinguishable. Two binaries that differ only in path --
# the residue of a build that silently did not rebuild -- produce a beautifully
# clean "no difference" that is really a comparison of one build with itself.
if [[ "$SHA_A" == "$SHA_B" && "$ARGS_A" == "$ARGS_B" ]]; then
	if ! $ALLOW_NULL_ARMS; then
		echo "ERROR: arm A and arm B are indistinguishable." >&2
		echo "  binary A: $BIN_A ($SHA_A)" >&2
		echo "  binary B: $BIN_B ($SHA_B)" >&2
		echo "  args   A: '$ARGS_A'" >&2
		echo "  args   B: '$ARGS_B'" >&2
		echo "  Same digest and same arguments: whatever this measured, it was not" >&2
		echo "  two arms. If a rebuild was meant to have happened, it did not." >&2
		echo "  Pass --allow-null-arms to run this deliberately as a null calibration." >&2
		exit 125
	fi
	echo "NOTE: --allow-null-arms: arms are identical by construction (null calibration)." >&2
fi

TS="$(date -u +"%Y%m%dT%H%M%SZ")"
OUT_DIR="${RMLX_HOME}/bench/perf_ab"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_perf_ab.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$OUT_DIR"
RESULT_JSON="${OUT_DIR}/${TS}.json"

# ---- host quiescence ---------------------------------------------------------
#
# Two things that look like they measure host load here do not:
#
#   * `ps -o pcpu` on macOS is a stale decayed figure. A process pinning a core
#     for ten seconds reads back as single digits, and the number does not move
#     while it runs. A gate built on it never fires.
#   * Load average counts blocked work and sits at 3-5 on an idle developer
#     desktop, so it has no usable threshold.
#
# What is exact is the change in a process's cumulative CPU time across a known
# wall-clock window: (delta cpu seconds) / (window seconds) is that process's
# true average utilisation over the window. Taking the snapshots either side of
# a measured slot makes the window the slot itself, so the report answers the
# question that matters -- was anything else running *while this was timed* --
# at no extra wall-clock cost.
#
# Load averages are recorded alongside as context, never as the criterion.

# `cpu_snapshot` comes from scripts/lib/cpu_snapshot.sh. The arm binaries are
# the measurement, not interference, so they are excluded by their actual
# basenames.
CPU_SNAPSHOT_SKIP="$(basename "$BIN_A") $(basename "$BIN_B")"
export CPU_SNAPSHOT_SKIP
# shellcheck source=scripts/lib/cpu_snapshot.sh
. "${REPO_ROOT}/scripts/lib/cpu_snapshot.sh"

load_averages() {
	uptime | sed -e 's/.*load averages*: *//' -e 's/,/ /g' | awk '{printf "%s %s %s", $1, $2, $3}'
}

# Take a snapshot into $1, recording whether it succeeded. A snapshot that
# could not be taken must never read back as an empty-but-valid one.
snapshot_ok() {
	if cpu_snapshot "$1"; then
		return 0
	fi
	: >"$1.failed"
	return 1
}

# "<state> <pct> <comm>" for the window between two snapshots $3 seconds apart.
# state is one of busy | quiet | unmeasured.
#
# `unmeasured` stays distinct from `quiet` all the way into the report, and it
# covers three different ways of not knowing: the window was too short to divide
# by, a snapshot was empty, or `ps` failed outright. Folding any of them into
# "nothing was running" is how an interference gate quietly stops gating.
classify_window() {
	local raw pct
	if [[ -e "$1.failed" || -e "$2.failed" ]]; then
		echo "unmeasured - -"
		return
	fi
	raw="$(awk -v window="$3" -f "$AWK_BUSIEST" "$1" "$2")"
	case "${raw%% *}" in
	unmeasured)
		echo "unmeasured - -"
		;;
	idle)
		echo "quiet 0.0 -"
		;;
	*)
		pct="$(echo "$raw" | awk '{print $2}')"
		if awk -v p="$pct" -v t="$BUSY_PCT" 'BEGIN { exit !(p >= t) }'; then
			echo "busy ${raw#* }"
		else
			echo "quiet ${raw#* }"
		fi
		;;
	esac
}

# Escape a free-form string for a JSON string literal. Process names, model
# paths and arm arguments all reach the result file; one stray quote in any of
# them would emit a file that no reader can parse, and nothing downstream would
# say so.
json_str() {
	printf '%s' "$1" |
		sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/	/\\t/g' |
		awk 'NR > 1 { printf "\\n" } { printf "%s", $0 } END { printf "\n" }' |
		sed -e 's/\r/\\r/g'
}

# Render a "<state> <pct> <comm>" triple for a human.
host_detail() {
	local rest="${1#* }"
	if [[ "${1%% *}" == "unmeasured" ]]; then
		echo "unmeasured (window too short to sample)"
	else
		echo "${rest%% *}% ${rest#* }"
	fi
}

# Sample a dedicated one-second window. Used before anything is measured.
probe_host() {
	local a b
	a="${WORK_DIR}/probe_a"
	b="${WORK_DIR}/probe_b"
	snapshot_ok "$a" || true
	sleep 1
	snapshot_ok "$b" || true
	classify_window "$a" "$b" 1
}

# ---- slot pattern ------------------------------------------------------------
#
# Balanced blocks: ABBA, then BAAB, then ABBA... For 12 slots this is exactly
# the pattern h3.c uses. Each block gives both arms the same mean position, so
# any drift that is monotone across a block cancels; alternating the block
# polarity extends that cancellation across the whole run.

build_pattern() {
	local n="$1" invert="$2" i block out=""
	for ((i = 0; i < n; i++)); do
		block=$((i / 4))
		local within=$((i % 4))
		local arm
		# ABBA inside an even block, BAAB inside an odd one.
		case "$within" in
		0 | 3) arm=0 ;;
		*) arm=1 ;;
		esac
		if [[ $((block % 2)) -eq 1 ]]; then
			arm=$((1 - arm))
		fi
		if [[ "$invert" == "true" ]]; then
			arm=$((1 - arm))
		fi
		out+="$arm"
	done
	echo "$out"
}

pattern_pretty() {
	echo "$1" | sed 's/0/A/g; s/1/B/g' | fold -w4 | paste -sd' ' -
}

PATTERN="$(build_pattern "$SLOTS" "$INVERT")"

# ---- one slot ----------------------------------------------------------------
#
# Runs one `rmlx baseline` and writes the parsed fields to $WORK_DIR. Every
# field is required: a run whose decode_tps or token_ids could not be read is a
# failed measurement, not a measurement with a missing column. Collapsing those
# two would let a broken arm quietly contribute nothing and still be summarised.

SLOT_TPS=""
SLOT_MEM=""
SLOT_IDS_FILE=""
SLOT_HOST=""

run_slot() {
	local bin="$1" model="$2" extra="$3" tag="$4"
	local raw="${WORK_DIR}/${tag}.out"
	local ids="${WORK_DIR}/${tag}.ids"
	local snap_a="${WORK_DIR}/${tag}.cpu_a"
	local snap_b="${WORK_DIR}/${tag}.cpu_b"
	local elapsed

	snapshot_ok "$snap_a" || true
	# The window is timed with bash's `time` builtin, not `date +%s`. BSD date
	# has whole-second resolution, which rounds a short slot's window to 0 and
	# makes the interference reading unmeasurable for reasons that have nothing
	# to do with the host. TIMEFORMAT=%R gives milliseconds and forks nothing.
	#
	# `--metrics off` is not a preference. An A/B run exercises arms that were
	# built to be thrown away, and `runs.db` is append-only: a row written from
	# a discarded arm cannot be taken back out. `off` never opens the file.
	TIMEFORMAT='%R'
	# shellcheck disable=SC2086  # extra args are deliberately word-split
	if ! elapsed="$( { time {
		RMLX_HOME="$RMLX_HOME" "$bin" --metrics off baseline \
			--model "$model" \
			--prompt-tokens "$PROMPT_TOKENS" \
			--max-tokens "$MAX_TOKENS" \
			--max-ctx "$MAX_CTX" \
			--emit-token-ids \
			$extra \
			>"$raw" 2>"${raw}.err"
	}; } 2>&1 )"; then
		echo "ERROR: slot $tag failed to run" >&2
		echo "  cmd: $bin baseline --model $model ... $extra" >&2
		sed 's/^/    | /' "${raw}.err" | tail -20 >&2
		exit 125
	fi
	snapshot_ok "$snap_b" || true
	SLOT_HOST="$(classify_window "$snap_a" "$snap_b" "$elapsed")"
	# The model-level window is the sum of its slots, kept at the same
	# millisecond resolution rather than re-derived from a whole-second clock.
	MODEL_ELAPSED="$(awk -v a="$MODEL_ELAPSED" -v b="$elapsed" 'BEGIN { printf "%.3f", a + b }')"

	SLOT_TPS="$(sed -n 's/.*decode_tps=\([0-9.]*\).*/\1/p' "$raw" | head -1)"
	SLOT_MEM="$(sed -n 's/.*metal_gen_alloc_mb=\([0-9.]*\).*/\1/p' "$raw" | head -1)"
	local slot_peak
	slot_peak="$(sed -n 's/.*metal_peak_mb=\([0-9.]*\).*/\1/p' "$raw" | head -1)"
	sed -n 's/^baseline: token_ids=//p' "$raw" | head -1 | tr ',' '\n' >"$ids"
	SLOT_IDS_FILE="$ids"
	local has_ids_line=0
	grep -q '^baseline: token_ids=' "$raw" && has_ids_line=1

	if [[ -z "$SLOT_TPS" ]]; then
		echo "ERROR: slot $tag produced no decode_tps. Output: $raw" >&2
		exit 125
	fi
	if [[ -z "$SLOT_MEM" || -z "$slot_peak" ]]; then
		echo "ERROR: slot $tag produced no Metal memory reading. Output: $raw" >&2
		exit 125
	fi
	# Presence is not the same as a measurement. `metal_peak_mb=0.0` is what the
	# bracket emits when the allocator is absent, the reset failed, or nothing
	# was materialised -- and it prints downstream as `A median=0.0 B median=0.0
	# delta=+0.0 MB`, which reads exactly like "both arms allocate identically".
	if awk -v v="$slot_peak" 'BEGIN { exit !(v + 0 <= 0) }'; then
		echo "ERROR: slot $tag reported metal_peak_mb=0 — the peak bracket measured" >&2
		echo "  nothing, so its memory column would be a zero that looks like a" >&2
		echo "  result. Output: $raw" >&2
		exit 125
	fi

	if [[ "$has_ids_line" -eq 0 ]]; then
		echo "ERROR: slot $tag emitted no token_ids line at all — the correctness" >&2
		echo "  comparison would have nothing to compare. Output: $raw" >&2
		exit 125
	fi

	# A token list that is present but degenerate compares 'identical' for the
	# wrong reason. An arm that early-stops at one token, or emits an empty
	# list, would pass a bare presence check while its TPS means nothing.
	local n_ids
	n_ids="$(grep -c '^[0-9][0-9]*$' "$ids" || true)"
	if [[ "${n_ids:-0}" -lt "$MAX_TOKENS" ]]; then
		echo "ERROR: slot $tag emitted $n_ids token ids, expected $MAX_TOKENS." >&2
		echo "  A short generation makes both the timing and the cross-arm token" >&2
		echo "  comparison meaningless. Output: $raw" >&2
		exit 125
	fi
}

# ---- statistics --------------------------------------------------------------

# median sd min max n, over space-separated values.
# Callers guarantee at least two values -- see the arm-count check below.
summarise() {
	echo "$*" | tr ' ' '\n' | sort -n | awk '
	{ a[NR] = $1; s += $1; ss += $1 * $1 }
	END {
		n = NR
		med = (n % 2 == 1) ? a[(n + 1) / 2] : (a[n / 2] + a[n / 2 + 1]) / 2
		sd = 0
		if (n >= 2) {
			mean = s / n
			v = (ss - n * mean * mean) / (n - 1)
			if (v < 0) v = 0
			sd = sqrt(v)
		}
		printf "%.4f %.4f %.4f %.4f %d\n", med, sd, a[1], a[n], n
	}'
}

# ---- run ---------------------------------------------------------------------

cat <<HEADER
========================================================================
rMLX A/B — ABBA-interleaved, $SLOTS slots per model ($((SLOTS / 2)) per arm)
------------------------------------------------------------------------
arm A  "$LABEL_A"   binary=$BIN_A  sha256:$SHA_A
       args: ${ARGS_A:-(none)}
arm B  "$LABEL_B"   binary=$BIN_B  sha256:$SHA_B
       args: ${ARGS_B:-(none)}
pattern: $(pattern_pretty "$PATTERN")$( $INVERT && echo "  (inverted)" )
shape:   --prompt-tokens $PROMPT_TOKENS --max-tokens $MAX_TOKENS --max-ctx $MAX_CTX

CRITERION, declared before any measurement:
  The arms are SEPARATED if and only if their per-slot decode_tps ranges are
  disjoint -- every slot of one arm faster than every slot of the other.
  Under the null hypothesis that the arms are exchangeable (which the ABBA
  pattern is what makes plausible), the probability of that happening by
  chance is exactly 2 / C($SLOTS, $((SLOTS / 2))) = $NULL_P per comparison.
  Anything else is INCONCLUSIVE. An INCONCLUSIVE ratio is not a small effect;
  it is no measured effect.

  FAMILY SIZE: this run makes ${#MODELS[@]} independent comparisons, one per model.
  The chance that AT LEAST ONE comes back SEPARATED under the null is
  1-(1-$NULL_P)^${#MODELS[@]} = $FAMILY_P. Read a single SEPARATED against that number,
  not against the per-comparison one. Pairing a run with --invert doubles the
  family again.

WHAT THE NUMBERS LICENSE:
  n=$((SLOTS / 2)) per arm. The median is a point estimate. The relative standard error
  of a sample stddev over $((SLOTS / 2)) values is ~1/sqrt(2(n-1)) = ${SD_RSE_PCT}%, so the
  spread figures describe this run and nothing more. No confidence interval and
  no p-value beyond the pre-declared rank test above are computed, and none
  should be read into the ratio.

INTERFERENCE GATE: a foreign process at or above ${BUSY_PCT}% of one core taints the
  run.$( awk -v t="$BUSY_PCT" 'BEGIN { exit !(t > 25) }' && echo "  RAISED from the 25% default -- the gate is correspondingly weaker." )
  It is measured from cumulative CPU time across each slot and across the
  comparison, so it sees anything running at the start, the end, or throughout.
  It does NOT see a process that both starts and exits inside one window.

Every slot runs with --metrics off, so runs.db is never opened. This run writes
its result to $RESULT_JSON. Each slot is a full rmlx process
and still writes its own \$RMLX_HOME/logs/<run-id>.jsonl, and each launch runs
the log size-cap rotation -- point RMLX_HOME at a scratch directory if that
matters.
========================================================================
HEADER

INITIAL_HOST="$(probe_host)"
INITIAL_LOAD="$(load_averages)"
echo "host at start: load(1,5,15)=$INITIAL_LOAD  busiest foreign process over a 1s window: $(host_detail "$INITIAL_HOST")"

if [[ "${INITIAL_HOST%% *}" == "busy" ]]; then
	if ! $ALLOW_BUSY_HOST; then
		echo "ERROR: host is not quiescent - $(host_detail "$INITIAL_HOST") is at or above ${BUSY_PCT}% CPU." >&2
		echo "  Measuring now would attribute that process's interference to whichever" >&2
		echo "  arm it lands on. Quiesce the host, or pass --allow-busy-host to see the" >&2
		echo "  numbers anyway -- they will still be reported as TAINTED and still exit 125." >&2
		exit 125
	fi
	echo "WARNING: --allow-busy-host: starting on a busy host; every number below is suspect." >&2
fi

# Single MLX process per Mac. A running server would contend for the Metal
# context throughout, which is exactly the confound this harness exists to
# exclude -- so it is reported, not killed. Killing it here would destroy
# someone else's work to make our number look better.
if pgrep -f "rmlx serve" >/dev/null 2>&1; then
	echo "ERROR: an 'rmlx serve' process is running and holds the Metal context." >&2
	echo "  Stop it before measuring:  pkill -f 'rmlx serve'" >&2
	exit 125
fi

OVERALL_EXIT=0
JSON_MODELS=""

for model in "${MODELS[@]}"; do
	short="$(basename "$model")"
	if [[ ! -d "$model" ]]; then
		echo "ERROR: model path not found: $model" >&2
		exit 125
	fi

	echo ""
	echo "==> $short"

	# One CPU window spanning this model's whole comparison. It is the
	# authoritative interference gate: the per-slot windows below are finer
	# attribution, but a slot can be shorter than the clock's resolution
	# whereas the comparison as a whole never is.
	snapshot_ok "${WORK_DIR}/model_cpu_a" || true
	MODEL_ELAPSED=0

	# Warmup: one untimed run per arm. It pays the page-cache and
	# shader-compile costs that would otherwise land entirely on slot 1, and
	# it establishes each arm's correctness reference.
	echo "  warmup A..." >&2
	run_slot "$BIN_A" "$model" "$ARGS_A" "warm_a"
	cp "$SLOT_IDS_FILE" "${WORK_DIR}/ref_a"
	echo "  warmup B..." >&2
	run_slot "$BIN_B" "$model" "$ARGS_B" "warm_b"
	cp "$SLOT_IDS_FILE" "${WORK_DIR}/ref_b"

	REF_TOKENS="$(wc -l <"${WORK_DIR}/ref_a" | tr -d ' ')"
	TOKENS_VERDICT="identical"
	if ! cmp -s "${WORK_DIR}/ref_a" "${WORK_DIR}/ref_b"; then
		# `cmp` (without -s) names the first differing byte offset and line;
		# it exits non-zero by design here, which is not a script failure.
		first_diff="$(cmp "${WORK_DIR}/ref_a" "${WORK_DIR}/ref_b" 2>&1 || true)"
		TOKENS_VERDICT="DIVERGED between arms ($first_diff)"
		if ! $ALLOW_TOKEN_DIVERGENCE; then
			echo "" >&2
			echo "FAIL [$short]: the arms generate different tokens." >&2
			echo "  $TOKENS_VERDICT" >&2
			echo "  Two arms that compute different things are not comparable on speed." >&2
			echo "  If the difference is intended (two KV codecs, say), re-run with" >&2
			echo "  --allow-token-divergence and say so when reporting the ratio." >&2
			exit 1
		fi
		echo "  NOTE: --allow-token-divergence: arms differ in output; the ratio below" >&2
		echo "        compares two different computations." >&2
	fi

	A_TPS=(); B_TPS=(); A_MEM=(); B_MEM=()
	# A run that had to be waived past the entry gate carries that fact into
	# every verdict it produces. Otherwise a comparison that started on a busy
	# host reads as clean the moment no individual slot happens to trip.
	TAINTED=""
	BUSY_SLOTS=0
	UNMEASURED_SLOTS=0
	WORST_SLOT=""
	WORST_PCT=0
	if [[ "${INITIAL_HOST%% *}" == "busy" ]]; then
		TAINTED="busy at the entry gate ($(host_detail "$INITIAL_HOST")); "
	fi

	for ((i = 0; i < SLOTS; i++)); do
		arm="${PATTERN:i:1}"
		if [[ "$arm" == "0" ]]; then
			bin="$BIN_A"; extra="$ARGS_A"; name="$LABEL_A"
		else
			bin="$BIN_B"; extra="$ARGS_B"; name="$LABEL_B"
		fi

		run_slot "$bin" "$model" "$extra" "slot_$i"

		# Every measured slot must reproduce its arm's reference. A slot that
		# drifts mid-run is a correctness failure even if the arms agreed at
		# warmup.
		ref="${WORK_DIR}/ref_a"
		[[ "$arm" == "1" ]] && ref="${WORK_DIR}/ref_b"
		if ! cmp -s "$SLOT_IDS_FILE" "$ref"; then
			echo "" >&2
			echo "FAIL [$short]: slot $i (arm $name) did not reproduce its own arm's" >&2
			echo "  warmup token ids. The arm is not deterministic, so neither its" >&2
			echo "  timing nor the comparison means anything." >&2
			exit 1
		fi

		if [[ "$arm" == "0" ]]; then
			A_TPS+=("$SLOT_TPS"); A_MEM+=("$SLOT_MEM")
		else
			B_TPS+=("$SLOT_TPS"); B_MEM+=("$SLOT_MEM")
		fi

		case "${SLOT_HOST%% *}" in
		busy)
			BUSY_SLOTS=$((BUSY_SLOTS + 1))
			slot_pct="$(echo "$SLOT_HOST" | awk '{print $2}')"
			if awk -v a="$slot_pct" -v b="$WORST_PCT" 'BEGIN { exit !(a > b) }'; then
				WORST_PCT="$slot_pct"
				WORST_SLOT="$(host_detail "$SLOT_HOST")"
			fi
			;;
		unmeasured)
			# Same rule at slot level as at model level: not knowing whether a
			# slot was interfered with is not the same as knowing it was not.
			UNMEASURED_SLOTS=$((UNMEASURED_SLOTS + 1))
			;;
		esac
		printf "  slot %2d  %-14s decode_tps=%8s  metal_gen_alloc_mb=%7s  busiest_foreign=%s\n" \
			"$i" "$name" "$SLOT_TPS" "$SLOT_MEM" "$(host_detail "$SLOT_HOST")"
	done

	if [[ "$BUSY_SLOTS" -gt 0 ]]; then
		TAINTED="${TAINTED}${BUSY_SLOTS} of ${SLOTS} slots ran alongside a foreign process (worst ${WORST_SLOT}); "
	fi
	if [[ "$UNMEASURED_SLOTS" -gt 0 ]]; then
		TAINTED="${TAINTED}${UNMEASURED_SLOTS} of ${SLOTS} slots could not be sampled for interference; "
	fi

	snapshot_ok "${WORK_DIR}/model_cpu_b" || true
	MODEL_HOST="$(classify_window "${WORK_DIR}/model_cpu_a" "${WORK_DIR}/model_cpu_b" \
		"$MODEL_ELAPSED")"
	case "${MODEL_HOST%% *}" in
	busy)
		TAINTED="${TAINTED}the comparison window as a whole ($(host_detail "$MODEL_HOST")); "
		;;
	unmeasured)
		# The comparison itself was too short to sample. Whatever else this run
		# is, it is not a measurement of a model, and "we did not look" must
		# never present as "nothing was there".
		TAINTED="${TAINTED}interference could not be sampled over the comparison; "
		;;
	esac

	# The pattern is built from whole ABBA blocks, so both arms always fill
	# equally. Check it anyway: summarising an under-filled arm would produce a
	# median and a stddev that look like measurements.
	if [[ ${#A_TPS[@]} -lt 2 || ${#B_TPS[@]} -ne ${#A_TPS[@]} ]]; then
		echo "ERROR: arms collected ${#A_TPS[@]} and ${#B_TPS[@]} samples from a ${SLOTS}-slot" >&2
		echo "  pattern '$PATTERN'. The schedule did not fill both arms; nothing here is" >&2
		echo "  comparable." >&2
		exit 125
	fi

	read -r A_MED A_SD A_MIN A_MAX A_N <<<"$(summarise "${A_TPS[@]}")"
	read -r B_MED B_SD B_MIN B_MAX B_N <<<"$(summarise "${B_TPS[@]}")"
	read -r AM_MED _ _ _ _ <<<"$(summarise "${A_MEM[@]}")"
	read -r BM_MED _ _ _ _ <<<"$(summarise "${B_MEM[@]}")"

	VERDICT="$(awk -v amin="$A_MIN" -v amax="$A_MAX" -v bmin="$B_MIN" -v bmax="$B_MAX" \
		'BEGIN { print (amax < bmin || bmax < amin) ? "SEPARATED" : "INCONCLUSIVE" }')"
	RATIO="$(awk -v a="$A_MED" -v b="$B_MED" 'BEGIN { printf "%.4f", (a > 0) ? b / a : 0 }')"
	DELTA_PCT="$(awk -v a="$A_MED" -v b="$B_MED" 'BEGIN { printf "%+.2f", (a > 0) ? (b / a - 1) * 100 : 0 }')"
	MEM_DELTA="$(awk -v a="$AM_MED" -v b="$BM_MED" 'BEGIN { printf "%+.1f", b - a }')"

	echo ""
	printf "  arm A  %-14s decode_tps  median=%8s  sd=%7s  min=%8s  max=%8s  n=%s\n" \
		"$LABEL_A" "$A_MED" "$A_SD" "$A_MIN" "$A_MAX" "$A_N"
	printf "  arm B  %-14s decode_tps  median=%8s  sd=%7s  min=%8s  max=%8s  n=%s\n" \
		"$LABEL_B" "$B_MED" "$B_SD" "$B_MIN" "$B_MAX" "$B_N"
	printf "  ratio B/A = %s  (%s%%)\n" "$RATIO" "$DELTA_PCT"
	printf "  metal_gen_alloc_mb  A median=%s  B median=%s  delta=%s MB\n" \
		"$AM_MED" "$BM_MED" "$MEM_DELTA"
	printf "  tokens: %s (%s ids per run, every slot re-checked)\n" "$TOKENS_VERDICT" "$REF_TOKENS"

	printf "  host during the comparison: %s\n" "$(host_detail "$MODEL_HOST")"

	if [[ -n "$TAINTED" ]]; then
		echo "  VERDICT: TAINTED — ${TAINTED%; }"
		echo "           The interference did not fall evenly on the arms, so the ratio"
		echo "           above is not usable. Re-run on a quiet host."
		# A tainted run exits non-zero even under --allow-busy-host. That flag
		# buys the right to SEE the numbers on a noisy host; it cannot make a
		# contaminated comparison count as a clean one, which is the exact
		# mistake this harness exists to stop.
		OVERALL_EXIT=125
	else
		printf "  VERDICT: %s\n" "$VERDICT"
		if [[ "$VERDICT" == "INCONCLUSIVE" ]]; then
			echo "           The arms' slot ranges overlap. The ${DELTA_PCT}% above is the"
			echo "           difference between two point estimates drawn from overlapping"
			echo "           spreads; it is not evidence that the arms differ."
		fi
	fi

	JSON_MODELS="${JSON_MODELS}$(
		cat <<JSON
    {"model": "$(json_str "$short")",
     "arm_a": {"label": "$(json_str "$LABEL_A")", "median_tps": $A_MED, "sd_tps": $A_SD, "min_tps": $A_MIN, "max_tps": $A_MAX, "n": $A_N, "median_gen_alloc_mb": $AM_MED, "tps": [$(
			IFS=,
			echo "${A_TPS[*]}"
		)]},
     "arm_b": {"label": "$(json_str "$LABEL_B")", "median_tps": $B_MED, "sd_tps": $B_SD, "min_tps": $B_MIN, "max_tps": $B_MAX, "n": $B_N, "median_gen_alloc_mb": $BM_MED, "tps": [$(
			IFS=,
			echo "${B_TPS[*]}"
		)]},
     "ratio_b_over_a": $RATIO,
     "verdict": "$( [[ -n "$TAINTED" ]] && echo "TAINTED" || echo "$VERDICT" )",
     "tokens": "$( [[ "$TOKENS_VERDICT" == identical ]] && echo identical || echo diverged )",
     "tokens_per_run": $REF_TOKENS,
     "taint": "$(json_str "${TAINTED%; }")"},
JSON
	)"
done

FINAL_HOST="$(probe_host)"
cat >"$RESULT_JSON" <<JSON
{
  "ts_utc": "$TS",
  "pattern": "$(pattern_pretty "$PATTERN")",
  "slots_per_model": $SLOTS,
  "shape": {"prompt_tokens": $PROMPT_TOKENS, "max_tokens": $MAX_TOKENS, "max_ctx": $MAX_CTX},
  "arm_a": {"label": "$(json_str "$LABEL_A")", "binary": "$(json_str "$BIN_A")", "sha256_16": "$SHA_A", "args": "$(json_str "$ARGS_A")"},
  "arm_b": {"label": "$(json_str "$LABEL_B")", "binary": "$(json_str "$BIN_B")", "sha256_16": "$SHA_B", "args": "$(json_str "$ARGS_B")"},
  "host": {"load_at_start": "$INITIAL_LOAD", "busiest_at_start": "$(json_str "$(host_detail "$INITIAL_HOST")")", "busiest_at_end": "$(json_str "$(host_detail "$FINAL_HOST")")", "busy_pct_threshold": $BUSY_PCT},
  "statistics": {"null_p_per_comparison": $NULL_P, "comparisons": ${#MODELS[@]}, "null_p_family": $FAMILY_P, "stddev_rel_std_err_pct": $SD_RSE_PCT},
  "waivers": {"null_arms": $ALLOW_NULL_ARMS, "busy_host": $ALLOW_BUSY_HOST, "token_divergence": $ALLOW_TOKEN_DIVERGENCE, "busy_pct_raised": $(awk -v t="$BUSY_PCT" 'BEGIN { print (t > 25) ? "true" : "false" }')},
  "results": [
${JSON_MODELS%,}
  ]
}
JSON

echo ""
echo "host at end:   busiest foreign process over a 1s window: $(host_detail "$FINAL_HOST")"
echo "result: $RESULT_JSON"
exit "$OVERALL_EXIT"
