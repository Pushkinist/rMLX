#!/usr/bin/env bash
# perf_ab_host_gate_fixtures.sh — recall test for the boundary between what
# `scripts/perf_ab.sh` measures and what it reads off the machine.
#
# WHY THIS EXISTS
#
# `perf_ab.sh` refuses to measure on a host it does not have to itself: a
# foreign process over the CPU threshold, or an `rmlx serve` holding the Metal
# context, and the run stops. That is right for a measurement and wrong for
# `perf_ab_selftest.sh`, which drives the same script against stub binaries to
# check its arithmetic and its refusals. A logic check whose result depends on
# what else the machine is doing is not a check; it teaches everyone to re-run
# until green, which is how a real failure gets waved through.
#
# `--synthetic-arms` is the boundary: it declares that the arms are stubs, so
# the run is a logic exercise and not a measurement, and the machine is not
# consulted at all. This file pins both directions of that boundary, because
# either one alone is satisfiable by a broken script:
#
#   * the host gates still FIRE on a hostile host without the flag (cases 1-2).
#     Without these, a `--synthetic-arms` that had accidentally become
#     unconditional would pass everything below.
#   * with the flag, a hostile host and a quiet host produce the SAME verdict
#     (cases 3-5) — host independence proven by equality, not by "it passed
#     once".
#   * the flag waives host state and nothing else (case 6). A mode that skips
#     every guard would also make the selftest green.
#   * a run that took the flag is marked as one in its own result file (case 7),
#     so nothing downstream can promote a stub run as a measurement.
#
# The host is supplied entirely by `ps` and `pgrep` shims on PATH, so no case
# here reads this machine and the file is deterministic under any load.
#
# EXIT CODES
#   0  every fixture produced the expected exit code and reason
#   1  a fixture did not
#   2  the fixtures themselves could not be built

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB="${REPO_ROOT}/scripts/perf_ab.sh"

[ -r "${AB}" ] || {
	echo "ERROR: missing ${AB}" >&2
	exit 2
}

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_ab_hostgate.XXXXXX")" || {
	echo "ERROR: could not create a scratch directory" >&2
	exit 2
}
trap 'rm -rf "${WORK}"' EXIT

MODEL="${WORK}/model"
mkdir -p "${MODEL}"

# ---- arms --------------------------------------------------------------------
#
# Constant decode_tps per arm, so the per-slot ranges are disjoint by
# construction and the verdict is SEPARATED in every case that reaches one.
# Nothing here sleeps: with `--synthetic-arms` no window is sampled, and the
# two cases that do sample are refused at the entry gate before a slot runs.
make_arm() { # make_arm NAME TPS
	local path="${WORK}/$1"
	cat >"${path}" <<STUB
#!/usr/bin/env bash
printf 'baseline: model=stub  load=1ms  ttft_ms=1  decode_tps=$2  overall_tps=$2  prefill_tps=1.0  prompt_tokens=4096  peak_rss=1.0MB  metal_peak_mb=100.0  metal_gen_alloc_mb=40.0\n'
printf 'baseline: token_ids=11,22,33,44,55\n'
STUB
	chmod +x "${path}"
	printf '%s' "${path}"
}
SLOW="$(make_arm slow 100.0)"
FAST="$(make_arm fast 110.0)"
[ -x "${SLOW}" ] && [ -x "${FAST}" ] || {
	echo "ERROR: could not build the stub arms" >&2
	exit 2
}

# ---- hosts -------------------------------------------------------------------
#
# Three synthetic machines. `cpu_snapshot` refuses a process table under 20
# rows, so every `ps` shim emits 40 idle rows on top of whatever it is
# demonstrating.
idle_rows() { # idle_rows -> 40 rows that burn nothing
	printf 'for i in $(seq 1 40); do printf "%%6d %%12s %%s\\n" $((5000 + i)) "0:00.10" "/usr/sbin/idle$i"; done\n'
}

mkdir -p "${WORK}/quiet" "${WORK}/hostile" "${WORK}/metalheld"

# A quiet machine: cumulative CPU times never move, no Metal holder.
{
	echo '#!/usr/bin/env bash'
	idle_rows
} >"${WORK}/quiet/ps"
printf '#!/bin/sh\nexit 1\n' >"${WORK}/quiet/pgrep"

# A hostile machine: one process whose cumulative CPU advances by 100 s on
# every call, which is 10000%% of a core over the harness's 1 s entry window.
{
	echo '#!/usr/bin/env bash'
	echo "n=\$(cat '${WORK}/hog.cnt' 2>/dev/null || echo 0)"
	echo "echo \$((n + 1)) >'${WORK}/hog.cnt'"
	echo 'printf "%6d %12s %s\n" 4242 "0:$((n * 100)).00" /usr/local/bin/hog'
	idle_rows
} >"${WORK}/hostile/ps"
printf '#!/bin/sh\nexit 1\n' >"${WORK}/hostile/pgrep"

# A quiet machine that is nonetheless not ours: something holds the Metal
# context.
cp "${WORK}/quiet/ps" "${WORK}/metalheld/ps"
printf '#!/bin/sh\necho 4243\nexit 0\n' >"${WORK}/metalheld/pgrep"

chmod +x "${WORK}"/quiet/* "${WORK}"/hostile/* "${WORK}"/metalheld/* || {
	echo "ERROR: could not build the host shims" >&2
	exit 2
}
for shim in quiet/ps quiet/pgrep hostile/ps hostile/pgrep metalheld/ps metalheld/pgrep; do
	[ -x "${WORK}/${shim}" ] || {
		echo "ERROR: host shim ${shim} is missing or not executable" >&2
		exit 2
	}
done

# ---- cases -------------------------------------------------------------------

failures=0
LAST_LOG=""
RESULT_PATHS=()

# Each case gets its own RMLX_HOME. The result file is named from a
# whole-second UTC stamp, so under one shared home two cases that finish inside
# the same second write the same path and the later silently replaces the
# earlier -- an assertion on "the result file" would then be reading a file some
# other case produced.
RUN_SEQ=0
run_ab() { # run_ab HOST_DIR [args...] -> exit code, output in $LAST_LOG
	local host="$1"
	shift
	RUN_SEQ=$((RUN_SEQ + 1))
	LAST_LOG="${WORK}/last.log"
	rm -f "${WORK}/hog.cnt"
	PATH="${host}:${PATH}" RMLX_HOME="${WORK}/home/run${RUN_SEQ}" \
		bash "${AB}" --model "${MODEL}" --slots 8 --max-tokens 5 "$@" \
		>"${LAST_LOG}" 2>&1
}

check() { # check LABEL HOST_DIR WANT_EXIT WANT_PATTERN [-- args...]
	local label="$1" host="$2" want_exit="$3" want_pat="$4"
	shift 4
	[ "${1:-}" = "--" ] && shift
	run_ab "${host}" "$@"
	local rc=$?
	local produced
	produced="$(sed -n 's/^result: //p' "${LAST_LOG}" | tail -1)"
	[ -n "${produced}" ] && RESULT_PATHS+=("${produced}")
	if [ "${rc}" -ne "${want_exit}" ]; then
		echo "FAIL  ${label}: exit ${rc}, expected ${want_exit}" >&2
		sed 's/^/      | /' "${LAST_LOG}" | tail -12 >&2
		failures=$((failures + 1))
		return
	fi
	if ! grep -qE -- "${want_pat}" "${LAST_LOG}"; then
		echo "FAIL  ${label}: exit ${rc} as expected but no line matching /${want_pat}/" >&2
		sed 's/^/      | /' "${LAST_LOG}" | tail -12 >&2
		failures=$((failures + 1))
		return
	fi
	echo "ok    ${label}  (exit ${rc}, reason matched)"
}

# 1 — the CPU gate still fires. A `--synthetic-arms` that had become
# unconditional would turn every case below green and this one red.
check "busy host is refused" "${WORK}/hostile" 125 "host is not quiescent" \
	-- --binary-a "${SLOW}" --binary-b "${FAST}"

# 2 — the exclusivity gate still fires. This is the one that made `make ci`
# depend on whether a server happened to be up.
check "a held Metal context is refused" "${WORK}/metalheld" 125 "holds the Metal context" \
	-- --binary-a "${SLOW}" --binary-b "${FAST}"

# 3 — same hostile host, arms declared synthetic: the machine is not consulted,
# so the run reaches its verdict and is not tainted.
check "synthetic arms ignore a busy host" "${WORK}/hostile" 0 "VERDICT: SEPARATED" \
	-- --synthetic-arms --binary-a "${SLOW}" --binary-b "${FAST}"
grep -q '^  TAINTED:' "${LAST_LOG}" && {
	echo "FAIL  synthetic arms ignore a busy host: the run reported TAINTED" >&2
	failures=$((failures + 1))
}
HOSTILE_VERDICT="$(grep -E '^  (ratio B/A|VERDICT:)' "${LAST_LOG}")"
HOSTILE_JSON="$(sed -n 's/^result: //p' "${LAST_LOG}" | tail -1)"

# 4 — and the same for a held Metal context: a stub arm dispatches no Metal, so
# whoever owns the context is irrelevant to it.
check "synthetic arms ignore a held Metal context" "${WORK}/metalheld" 0 "VERDICT: SEPARATED" \
	-- --synthetic-arms --binary-a "${SLOW}" --binary-b "${FAST}"

# 5 — the point of the whole file: the verdict is a function of the arms, not of
# the machine. Compared as text against case 3's, not merely asserted green.
check "synthetic arms are host-independent" "${WORK}/quiet" 0 "VERDICT: SEPARATED" \
	-- --synthetic-arms --binary-a "${SLOW}" --binary-b "${FAST}"
QUIET_VERDICT="$(grep -E '^  (ratio B/A|VERDICT:)' "${LAST_LOG}")"
if [ "${HOSTILE_VERDICT}" = "${QUIET_VERDICT}" ] && [ -n "${QUIET_VERDICT}" ]; then
	echo "ok    a hostile and a quiet host give the same answer  (${QUIET_VERDICT//$'\n'/ | })"
else
	echo "FAIL  a hostile and a quiet host give the same answer:" >&2
	echo "      | hostile: ${HOSTILE_VERDICT:-<none>}" >&2
	echo "      | quiet:   ${QUIET_VERDICT:-<none>}" >&2
	failures=$((failures + 1))
fi

# 6 — the flag waives host state and nothing else. A guard that reads only the
# arms must still refuse, or `--synthetic-arms` is a blanket skip and the
# selftest it exists to serve would be green for the wrong reason.
check "synthetic arms do not waive an arm guard" "${WORK}/quiet" 125 "indistinguishable" \
	-- --synthetic-arms --binary-a "${SLOW}" --binary-b "${SLOW}"

# 7 — the result file says so. `ingest/perf_ab_ingest.py` refuses to promote a
# run carrying this flag; that refusal is only enforceable if the flag is
# recorded.
if [ -n "${HOSTILE_JSON}" ] && [ -r "${HOSTILE_JSON}" ] &&
	grep -q '"synthetic_arms": true' "${HOSTILE_JSON}"; then
	echo "ok    the result file records the synthetic-arms waiver"
else
	echo "FAIL  the result file does not record the synthetic-arms waiver: ${HOSTILE_JSON:-<no result line>}" >&2
	failures=$((failures + 1))
fi

# 8 — the run carries no reading of this machine, not even a contextual one. A
# load average recorded beside a "not sampled" host line is still a number taken
# off this host, and it is the field a reader reaches for when the other one
# says nothing.
if [ -n "${HOSTILE_JSON}" ] && [ -r "${HOSTILE_JSON}" ] &&
	! grep -qE '"load_at_start": "[0-9]' "${HOSTILE_JSON}"; then
	echo "ok    the result file records no host load reading"
else
	echo "FAIL  the result file carries a load average taken from this machine:" >&2
	grep -o '"load_at_start": "[^"]*"' "${HOSTILE_JSON}" 2>/dev/null | sed 's/^/      | /' >&2
	failures=$((failures + 1))
fi

# 9 — every case above must have written its own result file. The name is a
# whole-second UTC stamp, so two runs that finish inside one second land on the
# same path and the later one silently replaces the earlier: case 7 would then
# be asserting on a file some other case produced.
distinct=$(printf '%s\n' "${RESULT_PATHS[@]}" | sort -u | grep -c .)
if [ "${distinct}" -eq "${#RESULT_PATHS[@]}" ] && [ "${distinct}" -gt 1 ]; then
	echo "ok    each case wrote its own result file (${distinct} of ${#RESULT_PATHS[@]})"
else
	echo "FAIL  result files collided: ${distinct} distinct path(s) for ${#RESULT_PATHS[@]} runs -- a case asserted on another case's file" >&2
	printf '      | %s\n' "${RESULT_PATHS[@]}" >&2
	failures=$((failures + 1))
fi

if [ "${failures}" -gt 0 ]; then
	echo >&2
	echo "ERROR: ${failures} fixture(s) did not reproduce the expected behaviour." >&2
	exit 1
fi
echo "OK: host gating fires on a hostile host, and --synthetic-arms removes the host from the answer."
