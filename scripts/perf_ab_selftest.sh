#!/usr/bin/env bash
# perf_ab_selftest.sh — mutation check for `scripts/perf_ab.sh`.
#
# An A/B harness that cannot detect a planted difference is not an instrument,
# and one that reports a difference between two identical arms is worse than
# none. Both directions are checked here against stub binaries that emit a
# known decode_tps sequence, so every expectation below is an exact number
# rather than a tolerance.
#
# The stubs are shell scripts standing in for `rmlx baseline`. That is the
# point: the harness's job is to schedule, gate and summarise, and this file
# tests exactly that, with no GPU, no model, and no metrics database.
#
# Exit 0 = every case produced its expected exit code and output.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB="$ROOT/scripts/perf_ab.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_ab_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

STATE="$WORK/state"
MODEL="$WORK/model"
mkdir -p "$STATE" "$MODEL"

TOKENS_MAIN="11,22,33,44,55"
TOKENS_ALT="11,22,99,44,55"

# make_stub <name> <tps-csv> <gen_alloc_mb> <tokens> [drift_at_call] [omit]
#
# The stub cycles through <tps-csv> across successive calls, so a run sees a
# spread rather than a constant and the disjoint-range criterion is exercised
# on real (if synthetic) variation. `drift_at_call` makes the Nth call emit
# TOKENS_ALT instead. `omit` drops a required field from the output.
make_stub() {
	local name="$1" tps="$2" mem="$3" tokens="$4" drift="${5:-}" omit="${6:-}"
	local path="$WORK/$name"
	cat >"$path" <<STUB
#!/usr/bin/env bash
# stub rmlx: $name
set -eu
# Imitate the real binary's metrics behaviour: without \`--metrics off\` the
# EventRecorder opens \$RMLX_HOME/metrics/runs.db. That is what makes the
# "no runs.db written" assertion at the end of the suite a real guard rather
# than a statement about stubs not doing very much.
case " \$* " in
*" --metrics off "*) : ;;
*) mkdir -p "\$RMLX_HOME/metrics" && : >"\$RMLX_HOME/metrics/runs.db" ;;
esac
TPS=($(echo "$tps" | tr ',' ' '))
CNT="$STATE/$name.cnt"
n=\$(cat "\$CNT" 2>/dev/null || echo 0)
echo \$((n + 1)) >"\$CNT"
v="\${TPS[\$((n % \${#TPS[@]}))]}"
tok="$tokens"
# A real baseline slot runs for seconds. The stub sleeps so the harness's
# interference window has something to divide by -- an instant stub would
# exercise the "too short to sample" path in every case instead of the
# behaviour under test.
sleep 0.25
if [ -n "$drift" ] && [ "\$n" = "$drift" ]; then tok="$TOKENS_ALT"; fi
if [ "$omit" != "tps" ]; then
  printf 'baseline: model=stub  load=1ms  ttft_ms=1  decode_tps=%s  overall_tps=%s  prefill_tps=1.0  prompt_tokens=4096  peak_rss=1.0MB  metal_peak_mb=100.0  metal_gen_alloc_mb=%s\n' "\$v" "\$v" "$mem"
else
  printf 'baseline: model=stub  load=1ms  ttft_ms=1  prompt_tokens=4096  metal_gen_alloc_mb=%s\n' "$mem"
fi
if [ "$omit" != "ids" ]; then printf 'baseline: token_ids=%s\n' "\$tok"; fi
STUB
	chmod +x "$path"
	echo "$path"
}

reset_state() { rm -f "$STATE"/*.cnt; }

PASSED=0
FAILED=0

# check <name> <expected-exit> <what-it-proves> -- <perf_ab.sh args...>
# Optional trailing `GREP:<pattern>` entries assert on the captured output.
check() {
	local name="$1" want="$2" what="$3"
	shift 3
	local args=() greps=()
	for a in "$@"; do
		case "$a" in
		GREP:*) greps+=("${a#GREP:}") ;;
		NOGREP:*) greps+=("!${a#NOGREP:}") ;;
		*) args+=("$a") ;;
		esac
	done

	reset_state
	local out="$WORK/$name.log"
	RMLX_HOME="$WORK/home" bash "$AB" "${args[@]}" >"$out" 2>&1
	local got=$?

	local ok=1
	[[ "$got" -eq "$want" ]] || ok=0
	local failed_pattern=""
	for g in "${greps[@]}"; do
		if [[ "$g" == !* ]]; then
			grep -qE "${g#!}" "$out" && { ok=0; failed_pattern="unexpected: ${g#!}"; }
		else
			grep -qE "$g" "$out" || { ok=0; failed_pattern="missing: $g"; }
		fi
	done

	if [[ "$ok" -eq 1 ]]; then
		PASSED=$((PASSED + 1))
		printf '  ok   %-26s exit=%s — %s\n' "$name" "$got" "$what"
	else
		FAILED=$((FAILED + 1))
		printf '  FAIL %-26s exit=%s (want %s) %s — %s\n' "$name" "$got" "$want" "$failed_pattern" "$what"
		sed 's/^/       | /' "$out" | tail -30
	fi
}

# ---- stubs -------------------------------------------------------------------

# Cycle 100.0/100.5/101.0. Over 7 calls (1 warmup + 6 slots) the 6 measured
# values are 100.5,101.0,100.0,100.5,101.0,100.0 -> median 100.5, min 100.0,
# max 101.0.
SLOW="$(make_stub slow "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN")"
# Same shape shifted up by exactly 10.0 -> median 110.5, ratio 110.5/100.5.
FAST="$(make_stub fast "110.0,110.5,111.0" 40.0 "$TOKENS_MAIN")"
# Byte-different file, identical numbers: the null arm.
SLOW_TWIN="$(make_stub slow_twin "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN")"
# Medians differ (101.0 vs 102.0) but the ranges overlap.
OVERLAP="$(make_stub overlap "101.0,102.0,103.0" 40.0 "$TOKENS_MAIN")"
# Same speed, more memory: proves the peak bracket reaches the report.
HUNGRY="$(make_stub hungry "100.0,100.5,101.0" 55.0 "$TOKENS_MAIN")"
# Same speed, different tokens.
WRONG="$(make_stub wrong "100.0,100.5,101.0" 40.0 "$TOKENS_ALT")"
# Reproduces its warmup for 3 calls, then drifts on the 4th (0-based call 3).
DRIFTER="$(make_stub drifter "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN" 3)"
NO_TPS="$(make_stub no_tps "100.0" 40.0 "$TOKENS_MAIN" "" tps)"
NO_IDS="$(make_stub no_ids "100.0" 40.0 "$TOKENS_MAIN" "" ids)"

# Nothing on this host counts as busy unless a case says otherwise; the CPU
# gate has its own cases below and must not make the others flaky.
QUIET=(--busy-pct 100000)
COMMON=(--model "$MODEL" --slots 12)

echo "perf_ab selftest: mutation checks"

# ---- can it see a difference that is there? ----------------------------------

check planted_10pct 0 \
	"a planted +9.95% arm is reported as +9.95% and SEPARATED" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:ratio B/A = 1\.0995" \
	"GREP:median= *100\.5000" \
	"GREP:median= *110\.5000" \
	"GREP:VERDICT: SEPARATED"

check planted_inverted 0 \
	"--invert swaps the pattern and still reports the same ratio" \
	--binary-a "$SLOW" --binary-b "$FAST" --invert "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:ratio B/A = 1\.0995" \
	"GREP:VERDICT: SEPARATED" \
	"GREP:pattern: BAAB ABBA BAAB"

check planted_memory 0 \
	"a planted +15 MB allocation shows up in the peak-memory column" \
	--binary-a "$SLOW" --binary-b "$HUNGRY" --allow-null-arms "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:A median=40\.0000  B median=55\.0000  delta=\+15\.0 MB" \
	"GREP:ratio B/A = 1\.0000"

# ---- does it stay quiet when there is nothing there? -------------------------

check null_arms 0 \
	"two arms with identical numbers report ratio 1.0000 and INCONCLUSIVE" \
	--binary-a "$SLOW" --binary-b "$SLOW_TWIN" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:ratio B/A = 1\.0000" \
	"GREP:VERDICT: INCONCLUSIVE" \
	"NOGREP:VERDICT: SEPARATED"

check overlapping_ranges 0 \
	"a 1% median gap whose ranges overlap is INCONCLUSIVE, not a result" \
	--binary-a "$SLOW" --binary-b "$OVERLAP" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:ratio B/A = 1\.0149" \
	"GREP:VERDICT: INCONCLUSIVE" \
	"GREP:is not evidence that the arms differ"

# ---- guards ------------------------------------------------------------------

check same_binary_refused 125 \
	"the same binary with the same args on both arms is refused" \
	--binary-a "$SLOW" --binary-b "$SLOW" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:indistinguishable"

check same_binary_waived 0 \
	"--allow-null-arms permits it deliberately and says so" \
	--binary-a "$SLOW" --binary-b "$SLOW" --allow-null-arms "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:null calibration" \
	"GREP:VERDICT: INCONCLUSIVE"

check token_divergence 1 \
	"arms that generate different tokens fail instead of being timed" \
	--binary-a "$SLOW" --binary-b "$WRONG" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:the arms generate different tokens"

check token_divergence_waived 0 \
	"--allow-token-divergence permits it and labels the ratio" \
	--binary-a "$SLOW" --binary-b "$WRONG" --allow-token-divergence "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:DIVERGED between arms"

check token_drift_within_arm 1 \
	"a slot that stops reproducing its own arm's tokens fails" \
	--binary-a "$DRIFTER" --binary-b "$FAST" --allow-token-divergence "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:did not reproduce its own arm"

check missing_tps 125 \
	"an arm that emits no decode_tps is a failed measurement, not a zero" \
	--binary-a "$NO_TPS" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:produced no decode_tps"

check missing_token_ids 125 \
	"an arm that emits no token_ids is refused rather than timed blind" \
	--binary-a "$NO_IDS" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:produced no token_ids"

check unbalanced_slots 125 \
	"a slot count that cannot form whole ABBA blocks is refused" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$MODEL" --slots 6 "${QUIET[@]}" \
	"GREP:multiple of 4"

check missing_model 125 \
	"a model path that does not exist stops the run" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$WORK/nope" --slots 4 "${QUIET[@]}" \
	"GREP:model path not found"

check busy_host_refused 125 \
	"a host over the CPU threshold is refused before anything is measured" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" --busy-pct 0 \
	"GREP:host is not quiescent"

check busy_host_taints 125 \
	"--allow-busy-host prints the numbers but a tainted run is still not clean" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" --busy-pct 0 --allow-busy-host \
	"GREP:VERDICT: TAINTED" \
	"NOGREP:VERDICT: SEPARATED"

# ---- the interference detector itself ----------------------------------------
#
# The host gate above can only be exercised against whatever this machine
# happens to be doing. The arithmetic underneath it is checked directly, with
# hand-written CPU snapshots, so "it detects a busy process" is a fact rather
# than a hope about the test host.

BUSIEST="$ROOT/scripts/lib/busiest_between.awk"

cpu_case() {
	local name="$1" want="$2" what="$3" window="$4" before="$5" after="$6"
	printf '%s\n' "$before" >"$WORK/cpu_before"
	printf '%s\n' "$after" >"$WORK/cpu_after"
	local got
	got="$(awk -v window="$window" -f "$BUSIEST" "$WORK/cpu_before" "$WORK/cpu_after")"
	if [[ "$got" == "$want" ]]; then
		PASSED=$((PASSED + 1))
		printf '  ok   %-26s        — %s\n' "$name" "$what"
	else
		FAILED=$((FAILED + 1))
		printf '  FAIL %-26s        — %s\n       | got  "%s"\n       | want "%s"\n' \
			"$name" "$what" "$got" "$want"
	fi
}

cpu_case cpu_detects_hog "busy-candidate 80.0 npm" \
	"a process that burns 4.0 CPU seconds in a 5s window reads as 80%" \
	5 $'100 10.00 npm\n200 5.00 Finder' $'100 14.00 npm\n200 5.00 Finder'

cpu_case cpu_idle_is_idle "idle 0.0 -" \
	"nothing burning CPU reads as idle, not as a small number" \
	5 $'100 10.00 npm' $'100 10.00 npm'

cpu_case cpu_new_pid_charged "busy-candidate 50.0 node" \
	"a process that started inside the window is charged all of its CPU time" \
	4 $'100 10.00 npm' $'100 10.00 npm\n900 2.00 node'

cpu_case cpu_short_window_unmeasured "unmeasured - -" \
	"a window too short to divide by reports unmeasured, never idle" \
	0 $'100 10.00 npm' $'100 14.00 npm'

cpu_case cpu_picks_the_biggest "busy-candidate 60.0 npm" \
	"the busiest process wins, not the first one seen" \
	5 $'100 0.00 Finder\n200 0.00 npm' $'100 1.00 Finder\n200 3.00 npm'

# `cpu_snapshot` is checked against a `ps` shim rather than against whatever
# this machine happens to be running, so both the exclusion list and the
# CPU-time parsing are pinned to exact expected output.
# shellcheck source=scripts/lib/cpu_snapshot.sh
. "$ROOT/scripts/lib/cpu_snapshot.sh"

mkdir -p "$WORK/fakebin"
cat >"$WORK/fakebin/ps" <<'FAKEPS'
#!/usr/bin/env bash
# Stand-in for `ps -Ao pid=,time=,comm=`, covering all three time renderings
# macOS produces plus the two binaries an A/B run must not treat as foreign.
cat <<'ROWS'
  101      1:30.50 /usr/bin/some-other-tool
  102   2:03:04.00 /Applications/Browser.app/Contents/MacOS/Browser
  103  1-00:00:01.00 /usr/libexec/ancient
  104      0:10.00 /tmp/build/rmlx.main
  105      0:11.00 /tmp/build/rmlx
ROWS
FAKEPS
chmod +x "$WORK/fakebin/ps"

CPU_SNAPSHOT_SKIP="rmlx.main rmlx" PATH="$WORK/fakebin:$PATH" cpu_snapshot "$WORK/snap_excl"

# 1:30.50 -> 90.50 | 2:03:04 -> 7384.00 | 1-00:00:01 -> 86401.00
WANT_SNAP='101 90.50 /usr/bin/some-other-tool
102 7384.00 /Applications/Browser.app/Contents/MacOS/Browser
103 86401.00 /usr/libexec/ancient'

if [[ "$(cat "$WORK/snap_excl")" == "$WANT_SNAP" ]]; then
	PASSED=$((PASSED + 1))
	printf '  ok   %-26s        — arm binaries excluded, and MM:SS / HH:MM:SS / D-HH:MM:SS all convert to seconds\n' \
		"cpu_snapshot_shape"
else
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — snapshot does not match\n       | got\n%s\n       | want\n%s\n' \
		"cpu_snapshot_shape" \
		"$(sed 's/^/       | /' "$WORK/snap_excl")" \
		"$(printf '%s\n' "$WANT_SNAP" | sed 's/^/       | /')"
fi

# ---- the pattern itself ------------------------------------------------------

check pattern_is_balanced 0 \
	"the default 12-slot pattern is the balanced ABBA BAAB ABBA schedule" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:pattern: ABBA BAAB ABBA" \
	"GREP:n=6"

# The mean slot position must be identical for the two arms, or the pattern
# leaks drift into the comparison. A/B positions are 0,3,4,7,8,11 and
# 1,2,5,6,9,10 -- both sum to 33.
POS_A=0
POS_B=0
PAT="$(grep -m1 '^pattern: ' "$WORK/pattern_is_balanced.log" | sed 's/^pattern: //; s/ //g')"
for ((i = 0; i < ${#PAT}; i++)); do
	if [[ "${PAT:i:1}" == "A" ]]; then POS_A=$((POS_A + i)); else POS_B=$((POS_B + i)); fi
done
if [[ "$POS_A" -eq "$POS_B" && ${#PAT} -eq 12 ]]; then
	PASSED=$((PASSED + 1))
	printf '  ok   %-26s        — arms share a mean slot position (%s == %s), so a monotone drift cancels\n' \
		"pattern_mean_position" "$POS_A" "$POS_B"
else
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — arm slot-position sums differ: A=%s B=%s (pattern %s)\n' \
		"pattern_mean_position" "$POS_A" "$POS_B" "$PAT"
fi

# ---- the result file must be readable ----------------------------------------
#
# Arm labels, arm arguments, model paths and process names all reach the result
# file verbatim. A quote in any of them would emit JSON that no reader can
# parse, and nothing downstream would say so.

check json_result_parses 0 \
	"the result file is valid JSON even with quotes and backslashes in a label" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	--label-a 'he said "fast"' --label-b 'back\slash' \
	"GREP:result: "

JSON_PATH="$(sed -n 's/^result: //p' "$WORK/json_result_parses.log" | tail -1)"
if [[ -n "$JSON_PATH" ]] && python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$JSON_PATH" 2>/dev/null; then
	PASSED=$((PASSED + 1))
	printf '  ok   %-26s        — %s parses as JSON\n' "json_result_parses_body" "$(basename "$JSON_PATH")"
else
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — result file is not valid JSON: %s\n' \
		"json_result_parses_body" "${JSON_PATH:-<no result line>}"
fi

# ---- it must not touch the metrics database ---------------------------------
#
# The stubs create runs.db unless they are handed `--metrics off`, so this is a
# check on what the harness passes, not on stubs being inert. Every case above
# has run by now; if any of them dropped the flag, the file exists.

if [[ -e "$WORK/home/metrics/runs.db" ]]; then
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — a slot ran without --metrics off; runs.db appeared at %s\n' \
		"no_metrics_db_written" "$WORK/home/metrics/runs.db"
else
	PASSED=$((PASSED + 1))
	printf '  ok   %-26s        — every slot ran with --metrics off; no runs.db was created\n' \
		"no_metrics_db_written"
fi

echo ""
if [[ "$FAILED" -ne 0 ]]; then
	echo "perf_ab selftest: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
	exit 1
fi
echo "perf_ab selftest: ok ($PASSED cases)"
