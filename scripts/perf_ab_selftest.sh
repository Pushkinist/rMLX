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

# Sourced up front, before any case calls it. A missing definition exits
# non-zero exactly like a failing `ps`, so the cases below would pass for the
# wrong reason if this arrived later in the file.
# shellcheck source=scripts/lib/cpu_snapshot.sh
. "$ROOT/scripts/lib/cpu_snapshot.sh"
if ! type cpu_snapshot >/dev/null 2>&1; then
	echo "selftest bug: cpu_snapshot is not defined" >&2
	exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_ab_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

STATE="$WORK/state"
MODEL="$WORK/model"
MODEL2="$WORK/model2"
mkdir -p "$STATE" "$MODEL" "$MODEL2"

TOKENS_MAIN="11,22,33,44,55"
TOKENS_ALT="11,22,99,44,55"

# make_stub <name> <tps-csv> <gen_alloc_mb> <tokens> [drift_at_call] [omit] [kv_bytes]
#
# The stub cycles through <tps-csv> across successive calls, so a run sees a
# spread rather than a constant and the disjoint-range criterion is exercised
# on real (if synthetic) variation. `drift_at_call` makes the Nth call emit
# TOKENS_ALT instead. `omit` drops a required field from the output.
# `kv_bytes` is the resident-KV column: a byte count, the literal `n/a` (the
# baseline command's own refusal), or empty to drop the field entirely, which
# is what an older binary looks like to this harness.
make_stub() {
	local name="$1" tps="$2" mem="$3" tokens="$4" drift="${5:-}" omit="${6:-}" kv="${7:-}"
	local path="$WORK/$name"
	cat >"$path" <<STUB
#!/usr/bin/env bash
# stub rmlx: $name
set -eu
# Imitate the real binary's metrics behaviour, including clap's last-occurrence
# -wins resolution of a \`global = true\` flag: \`--metrics off ... --metrics full\`
# records. Without that, the "no runs.db written" assertion at the end of the
# suite would be a statement about stubs not doing very much, instead of a
# guard on the one escape that reaches the real append-only store.
mode=full
prev=""
for a in "\$@"; do
  case "\$a" in
  --metrics=*) mode="\${a#--metrics=}" ;;
  *) [ "\$prev" = "--metrics" ] && mode="\$a" ;;
  esac
  prev="\$a"
done
if [ "\$mode" != "off" ]; then
  mkdir -p "\$RMLX_HOME/metrics" && : >"\$RMLX_HOME/metrics/runs.db"
fi
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
peak=100.0
if [ "$omit" = "zeropeak" ]; then peak=0.0; fi
kvcol=""
kvval="$kv"
# "partial" is the any-vs-all case: this slot refuses on odd-numbered calls and
# reports on even ones, so an arm ends up with a mixture. (No backticks: this
# heredoc is unquoted, so they would run as command substitution right here.)
if [ "$kv" = "partial" ]; then
  if [ \$((n % 2)) -eq 1 ]; then kvval="n/a"; else kvval=1000000123; fi
fi
if [ -n "$kv" ]; then kvcol="  kv_cache_bytes=\$kvval"; fi
if [ "$omit" != "tps" ]; then
  printf 'baseline: model=stub  load=1ms  ttft_ms=1  decode_tps=%s  overall_tps=%s  prefill_tps=1.0  prompt_tokens=4096  peak_rss=1.0MB  metal_peak_mb=%s  metal_gen_alloc_mb=%s%s\n' "\$v" "\$v" "\$peak" "$mem" "\$kvcol"
else
  printf 'baseline: model=stub  load=1ms  ttft_ms=1  prompt_tokens=4096  metal_peak_mb=%s  metal_gen_alloc_mb=%s%s\n' "\$peak" "$mem" "\$kvcol"
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
#
# <expected-exit> may be the sentinel RC_REPORT instead of a number. Use it for
# any case whose subject is what the harness REPORTED, rather than a refusal it
# chose. For a run that reaches its report the exit code is 0 on a quiet host
# and 125 on a tainted one, so a literal `0` there is an assertion about the
# machine: it holds until the interference sampler happens to fire and then
# fails while the behaviour under test is still correct. RC_REPORT asserts
# instead that the code agrees with the harness's OWN verdict line -- `VERDICT:
# TAINTED` means 125, anything else means 0 -- which still catches a harness
# that stops signalling taint, without importing the host into the expectation.
#
# A refusal case keeps its literal number: there the code is the behaviour.
check() {
	local name="$1" want="$2" what="$3"
	shift 3
	local args=() greps=() path_prefix=""
	for a in "$@"; do
		case "$a" in
		GREP:*) greps+=("${a#GREP:}") ;;
		NOGREP:*) greps+=("!${a#NOGREP:}") ;;
		PATHPRE:*) path_prefix="${a#PATHPRE:}" ;;
		*) args+=("$a") ;;
		esac
	done

	reset_state
	local out="$WORK/$name.log"
	# bash 3.2 (the system bash here) treats "${arr[@]}" on an empty array as an
	# unbound variable under `set -u` and dies. Every current call site supplies
	# both, but the guard keeps a future one from failing as a harness crash.
	RMLX_HOME="$WORK/home" PATH="${path_prefix:+$path_prefix:}$PATH" \
		bash "$AB" ${args[@]+"${args[@]}"} >"$out" 2>&1
	local got=$?

	# RC_REPORT: the expected code is whatever the harness's own verdict line
	# implies, resolved after the run rather than assumed before it.
	if [[ "$want" == "RC_REPORT" ]]; then
		if grep -q '^  TAINTED:' "$out"; then want=125; else want=0; fi
	fi

	local ok=1
	[[ "$got" -eq "$want" ]] || ok=0
	local failed_pattern=""
	for g in ${greps[@]+"${greps[@]}"}; do
		if [[ "$g" == !* ]]; then
			grep -qE -- "${g#!}" "$out" && { ok=0; failed_pattern="unexpected: ${g#!}"; }
		else
			grep -qE -- "$g" "$out" || { ok=0; failed_pattern="missing: $g"; }
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
# Resident-KV column. KV_SMALL/KV_BIG carry byte counts a factor 2 apart;
# KV_NA emits the baseline command's own `n/a` refusal; KV_ABSENT drops the
# field, which is what a binary predating the column looks like.
KV_SMALL="$(make_stub kv_small "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN" "" "" 1000000123)"
KV_BIG="$(make_stub kv_big "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN" "" "" 2000000000)"
KV_NA="$(make_stub kv_na "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN" "" "" "n/a")"
# Reports on some calls and refuses on others -- the case the "ANY slot refused"
# invariant is actually about.
KV_PARTIAL="$(make_stub kv_partial "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN" "" "" "partial")"
# Neither a byte count nor the `n/a` refusal.
KV_NAN="$(make_stub kv_nan "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN" "" "" "nan")"
KV_ABSENT="$(make_stub kv_absent "100.0,100.5,101.0" 40.0 "$TOKENS_MAIN")"

# Nothing on this host counts as busy unless a case says otherwise; the CPU
# gate has its own cases below and must not make the others flaky.
QUIET=(--busy-pct 100000)
# The stubs emit 5 token ids, so --max-tokens matches: the harness refuses a
# slot whose generation is shorter than asked for.
COMMON=(--model "$MODEL" --slots 12 --max-tokens 5)

echo "perf_ab selftest: mutation checks"

# ---- can it see a difference that is there? ----------------------------------

check planted_10pct RC_REPORT \
	"a planted +9.95% arm is reported as +9.95% and SEPARATED" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:ratio B/A = 1\.0995" \
	"GREP:median= *100\.5000" \
	"GREP:median= *110\.5000" \
	"GREP:VERDICT: SEPARATED"

check planted_inverted RC_REPORT \
	"--invert swaps the pattern and still reports the same ratio" \
	--binary-a "$SLOW" --binary-b "$FAST" --invert "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:ratio B/A = 1\.0995" \
	"GREP:VERDICT: SEPARATED" \
	"GREP:pattern: BAAB ABBA BAAB"

check planted_memory RC_REPORT \
	"a planted +15 MB allocation shows up in the peak-memory column" \
	--binary-a "$SLOW" --binary-b "$HUNGRY" --allow-null-arms "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:A median=40\.0000  B median=55\.0000  delta=\+15\.0 MB" \
	"GREP:ratio B/A = 1\.0000"

check planted_kv_residency RC_REPORT \
	"a 2x resident-KV arm is reported as ratio 2.0000, in MB" \
	--binary-a "$KV_SMALL" --binary-b "$KV_BIG" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:kv_cache_bytes      A median=1000\.0 MB  B median=2000\.0 MB  ratio B/A=2\.0000" \
	"GREP:result: "

# The printed table rounds to 0.1 MB. The result file is what gets promoted into
# the append-only metrics store, so it has to carry the byte count the slot
# actually reported: KV_SMALL's 1 000 000 123 B prints as 1000.0 MB either way,
# and only the JSON can tell an exact reading from a round-tripped one.
JSON_KV="$(sed -n 's/^result: //p' "$WORK/planted_kv_residency.log" | tail -1)"
KV_A="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['results'][0]['arm_a']['median_kv_cache_bytes'])" "$JSON_KV" 2>/dev/null || echo unreadable)"
if [[ "$KV_A" == "1000000123" ]]; then
	PASSED=$((PASSED + 1))
	printf '  ok   %-26s        — result file carries exact bytes (%s), not the rounded MB\n' \
		"kv_residency_json_exact" "$KV_A"
else
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — arm A median_kv_cache_bytes is %s, want 1000000123\n' \
		"kv_residency_json_exact" "$KV_A"
fi

# The two ways this column can be absent must both read as "not measured".
# A 0 here would divide into a residency ratio as a cache of no bytes, which
# is the shape of every silent-fallback defect this repo has had to unpick.
check kv_residency_refusal_is_not_zero RC_REPORT \
	"a slot whose KV accounting refused reports n/a, never 0" \
	--binary-a "$KV_NA" --binary-b "$KV_BIG" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:kv_cache_bytes      A median=n/a MB  B median=2000\.0 MB  ratio B/A=n/a" \
	"NOGREP:A median=0\.0 MB"

check kv_residency_absent_column_is_not_zero RC_REPORT \
	"a binary that emits no KV column reports n/a, never 0" \
	--binary-a "$KV_ABSENT" --binary-b "$KV_BIG" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:kv_cache_bytes      A median=n/a MB  B median=2000\.0 MB  ratio B/A=n/a"

# The invariant is "n/a if ANY slot refused", and an all-refuse stub cannot tell
# that apart from "n/a if EVERY slot refused" -- both rules agree on it. Only a
# mixture separates them.
check kv_residency_any_refusal_taints_the_arm RC_REPORT \
	"one refusing slot makes the whole arm n/a, not a median of the rest" \
	--binary-a "$KV_PARTIAL" --binary-b "$KV_BIG" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:kv_cache_bytes      A median=n/a MB  B median=2000\.0 MB  ratio B/A=n/a" \
	"NOGREP:A median=1000\.0 MB"

# awk reads a non-numeric token as 0, so an unrecognised spelling would become a
# median of 0 bytes and record as the smallest cache ever measured.
check kv_residency_non_numeric_refused 125 \
	"a kv_cache_bytes token that is neither a number nor n/a is refused" \
	--binary-a "$KV_NAN" --binary-b "$KV_BIG" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:reported kv_cache_bytes=nan"

# ---- does it stay quiet when there is nothing there? -------------------------

check null_arms RC_REPORT \
	"two arms with identical numbers report ratio 1.0000 and INCONCLUSIVE" \
	--binary-a "$SLOW" --binary-b "$SLOW_TWIN" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:ratio B/A = 1\.0000" \
	"GREP:VERDICT: INCONCLUSIVE" \
	"NOGREP:VERDICT: SEPARATED"

check overlapping_ranges RC_REPORT \
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

check same_binary_waived RC_REPORT \
	"--allow-null-arms permits it deliberately and says so" \
	--binary-a "$SLOW" --binary-b "$SLOW" --allow-null-arms "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:null calibration" \
	"GREP:VERDICT: INCONCLUSIVE"

check token_divergence 1 \
	"arms that generate different tokens fail instead of being timed" \
	--binary-a "$SLOW" --binary-b "$WRONG" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:the arms generate different tokens"

check token_divergence_waived RC_REPORT \
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
	"GREP:emitted no token_ids line"

check unbalanced_slots 125 \
	"a slot count that cannot form whole ABBA blocks is refused" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$MODEL" --slots 6 --max-tokens 5 "${QUIET[@]}" \
	"GREP:multiple of 4"

check missing_model 125 \
	"a model path that does not exist stops the run" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$WORK/nope" --slots 8 --max-tokens 5 "${QUIET[@]}" \
	"GREP:model path not found"

# The refusal is the central guard, so its end-to-end coverage must not depend
# on what this machine happens to be doing. A `ps` shim reports a process whose
# cumulative CPU time advances on every call, which is a busy host by
# construction; the earlier form used `--busy-pct 0` and would have gone QUIET
# on an idle CI runner, because `classify_window` short-circuits `idle` before
# the threshold is applied.
mkdir -p "$WORK/busybin"
cat >"$WORK/busybin/ps" <<PSHOG
#!/usr/bin/env bash
n=\$(cat "$STATE/pshog.cnt" 2>/dev/null || echo 0)
echo \$((n + 1)) >"$STATE/pshog.cnt"
printf '%6d %12s %s\n' 4242 "0:\$((n * 100)).00" /usr/local/bin/hog
for i in \$(seq 1 40); do printf '%6d %12s %s\n' \$((5000 + i)) "0:00.10" "/usr/sbin/idle\$i"; done
PSHOG
chmod +x "$WORK/busybin/ps"

check busy_host_refused 125 \
	"a host over the CPU threshold is refused before anything is measured" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" \
	"PATHPRE:$WORK/busybin" \
	"GREP:host is not quiescent" \
	"GREP:hog"

# These two assert that taint is REPORTED and that it does not erase the rank
# test's answer. They used to assert the answer was absent, which pinned the
# old format (taint replacing the verdict) rather than the behaviour.
check busy_host_taints 125 \
	"--allow-busy-host prints the numbers but a tainted run is still not clean" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" --allow-busy-host \
	"PATHPRE:$WORK/busybin" \
	"GREP:^  TAINTED:" \
	"GREP:VERDICT: SEPARATED"

# ---- the runs.db escape ------------------------------------------------------
#
# `--metrics` is `global = true`, so an occurrence after the subcommand beats
# the harness's leading `--metrics off` and the slot opens the real append-only
# runs.db. Verified against the built binary, including on a failure path where
# the model never loaded.

check metrics_flag_in_arm_refused 125 \
	"--metrics in an arm's arguments is refused before anything runs" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	--arm-b "--metrics full" \
	"GREP:--metrics may not appear in arm arguments" \
	"GREP:append-only"

check metrics_flag_equals_form_refused 125 \
	"the --metrics=VALUE spelling is refused too" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	--arm-a "--metrics=full" \
	"GREP:--metrics may not appear in arm arguments"

# ---- degenerate measurements -------------------------------------------------

ZEROPEAK="$(make_stub zeropeak "100.0,100.5,101.0" 0.0 "$TOKENS_MAIN" "" zeropeak)"
check vacuous_memory_refused 125 \
	"a slot whose peak bracket measured nothing is refused, not averaged as 0" \
	--binary-a "$ZEROPEAK" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:metal_peak_mb=0"

SHORT="$(make_stub short "100.0,100.5,101.0" 40.0 "1,2")"
check short_generation_refused 125 \
	"a slot that generated fewer tokens than asked for is refused" \
	--binary-a "$SHORT" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"GREP:emitted 2 token ids, expected 5"

# ---- option validation -------------------------------------------------------

check non_numeric_slots 125 \
	"--slots abc exits 125, not the 1 reserved for token divergence" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$MODEL" --slots abc "${QUIET[@]}" \
	"GREP:--slots must be a number"

check non_numeric_busy_pct 125 \
	"a non-numeric --busy-pct is refused rather than silently disabling the gate" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" --busy-pct abc \
	"GREP:--busy-pct must be a number"

check slots_too_few_for_a_verdict 125 \
	"--slots 4 is refused: SEPARATED would carry a 1-in-3 null probability" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$MODEL" --slots 4 --max-tokens 5 "${QUIET[@]}" \
	"GREP:null probability of 0\.33333"

# ---- the reported statistics must be computed, not asserted ------------------

check stddev_uncertainty_tracks_n RC_REPORT \
	"the stddev's relative standard error is computed from n, not a fixed 30%" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$MODEL" --slots 8 --max-tokens 5 "${QUIET[@]}" \
	"GREP:over 4 values is ~1/sqrt\(2\(n-1\)\) = 41%" \
	"NOGREP:= 32%"

check family_size_is_stated RC_REPORT \
	"the family-wise rate is stated for a run that makes two comparisons" \
	--binary-a "$SLOW" --binary-b "$FAST" --model "$MODEL" --model "$MODEL2" \
	--slots 12 --max-tokens 5 "${QUIET[@]}" \
	"GREP:this run makes 2 independent comparisons" \
	"GREP:1-\(1-0\.00216\)\^2 = 0\.00432"

# ---- the interference detector itself ----------------------------------------
#
# The host gate above can only be exercised against whatever this machine
# happens to be doing. The arithmetic underneath it is checked directly, with
# hand-written CPU snapshots, so "it detects a busy process" is a fact rather
# than a hope about the test host.

BUSIEST="$ROOT/scripts/lib/busiest_between.awk"

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

cpu_case() {
	local name="$1" want="$2" what="$3" window="$4" before="$5" after="$6"
	# An empty fixture must produce a genuinely empty file, not a blank line.
	if [[ -n "$before" ]]; then printf '%s\n' "$before" >"$WORK/cpu_before"; else : >"$WORK/cpu_before"; fi
	if [[ -n "$after" ]]; then printf '%s\n' "$after" >"$WORK/cpu_after"; else : >"$WORK/cpu_after"; fi
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
	"a window below the CPU counter's resolution reports unmeasured, never idle" \
	0.02 $'100 10.00 npm' $'100 14.00 npm'

cpu_case cpu_picks_the_biggest "busy-candidate 60.0 npm" \
	"the busiest process wins, not the first one seen" \
	5 $'100 0.00 Finder\n200 0.00 npm' $'100 1.00 Finder\n200 3.00 npm'

# An empty snapshot is a snapshot that was not taken -- `ps` failed, was
# blocked, or was sandboxed. Reporting that as `idle` maps to `quiet` and
# disables the entry refusal and the taint gate at once, and the run exits 0.
cpu_case cpu_empty_before_is_unmeasured "unmeasured - -" \
	"an empty BEFORE snapshot is unmeasured, not idle, even with a hog in AFTER" \
	5 '' $'100 989.00 /usr/local/bin/hog'

cpu_case cpu_empty_after_is_unmeasured "unmeasured - -" \
	"an empty AFTER snapshot is unmeasured, not idle" \
	5 $'100 0.00 /usr/local/bin/hog' ''

cpu_case cpu_both_empty_is_unmeasured "unmeasured - -" \
	"two empty snapshots answer nothing rather than answering 'quiet'" \
	5 '' ''

# `cpu_snapshot` must report failure rather than leave an empty file behind: the
# pipeline's status is awk's, which is always 0, so a failing `ps` is invisible
# to `set -e` and to every caller that does not check.
mkdir -p "$WORK/failbin"
printf '#!/bin/sh\nexit 1\n' >"$WORK/failbin/ps"
chmod +x "$WORK/failbin/ps"
if PATH="$WORK/failbin:$PATH" cpu_snapshot "$WORK/snap_fail" 2>/dev/null; then
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — cpu_snapshot returned success for a failing ps\n' "cpu_snapshot_reports_failure"
else
	PASSED=$((PASSED + 1))
	printf '  ok   %-26s        — cpu_snapshot returns non-zero when ps fails\n' "cpu_snapshot_reports_failure"
fi

# A truncated process table is not hypothetical: a restricted host returns a
# small fraction of it. Comparing two such snapshots would report a quiet host
# on almost no evidence.
if PATH="$WORK/fakebin:$PATH" cpu_snapshot "$WORK/snap_thin" 2>/dev/null; then
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — a 5-row process table passed the row floor\n' "cpu_snapshot_row_floor"
elif CPU_SNAPSHOT_MIN_ROWS=3 PATH="$WORK/fakebin:$PATH" cpu_snapshot "$WORK/snap_thin" 2>/dev/null; then
	PASSED=$((PASSED + 1))
	printf '  ok   %-26s        — a thin process table is refused, and the floor is what refuses it\n' "cpu_snapshot_row_floor"
else
	FAILED=$((FAILED + 1))
	printf '  FAIL %-26s        — the row floor rejects even when lowered below the row count\n' "cpu_snapshot_row_floor"
fi

# End to end: a failing `ps` must taint the run rather than let it report clean.
check failing_ps_taints 125 \
	"a slot whose interference could not be sampled taints the comparison" \
	--binary-a "$SLOW" --binary-b "$FAST" "${COMMON[@]}" "${QUIET[@]}" \
	"PATHPRE:$WORK/failbin" \
	"GREP:^  TAINTED:" \
	"GREP:could not be sampled" \
	"GREP:VERDICT: SEPARATED"


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

check pattern_is_balanced RC_REPORT \
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

check json_result_parses RC_REPORT \
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
