# Per-process cumulative CPU seconds, for interference gates; source it.
#
# Defines `cpu_snapshot`, `snapshot_ok`, `window_not_sampled` and `classify_window`, which
# every A/B and sweep harness shares. Source it; do not execute it.
#
#     source "$REPO_ROOT/scripts/lib/cpu_snapshot.sh"
#     cpu_snapshot /tmp/before
#     ... the thing being measured ...
#     cpu_snapshot /tmp/after
#     awk -v window=<seconds> -f scripts/lib/busiest_between.awk /tmp/before /tmp/after
#
# Output: `<pid> <cumulative_cpu_seconds> <command>` per line.
#
# Why cumulative CPU time rather than `ps -o pcpu`: on macOS pcpu is a stale
# decayed figure. A process pinning a core reads back as single digits and the
# number does not move while it runs, so a gate built on it never fires. The
# difference of cumulative CPU seconds across a known wall-clock window is
# exact.
#
# `ps -o time` renders as `MM:SS.ss`, `HH:MM:SS.ss` or `D-HH:MM:SS.ss`; all
# three are converted here. Feeding the raw string to a numeric comparison
# instead would silently truncate at the first colon and report every delta as
# zero.
#
# Exclusions, via the environment:
#   CPU_SNAPSHOT_SKIP  space-separated basenames to omit -- the binaries under
#                      measurement, which are the measurement and not
#                      interference. Set it to the actual names in use; a
#                      hard-coded "rmlx" would let an arm named `rmlx.main` be
#                      reported as a foreign process competing with itself.
#   CPU_SNAPSHOT_MIN_ROWS  row floor below which the snapshot is refused
#                      (default 20). Lower it only in a fixture.
# The calling shell and `ps` itself are always omitted.

# Returns non-zero when the snapshot could not be taken. A caller that ignores
# that gets an empty file, and an empty file compares as "nothing was running"
# -- which is how an interference gate silently stops gating. `ps` failing,
# being blocked, or returning a fraction of the process table on a restricted
# host are all real; `2>/dev/null` on the pipeline hides the first two, and the
# pipeline's own status is awk's, which is always 0.
cpu_snapshot() {
	ps -Aww -o pid=,time=,comm= 2>/dev/null |
		awk -v self="$$" -v skip="${CPU_SNAPSHOT_SKIP:-}" '
	BEGIN { n = split(skip, s, " "); for (i = 1; i <= n; i++) drop[s[i]] = 1; drop["ps"] = 1 }
	function tsec(str,   d, p, m, r, i) {
		d = 0
		if (index(str, "-") > 0) {
			d = substr(str, 1, index(str, "-") - 1)
			str = substr(str, index(str, "-") + 1)
		}
		m = split(str, p, ":")
		r = 0
		for (i = 1; i <= m; i++) r = r * 60 + p[i]
		return r + d * 86400
	}
	{
		pid = $1; t = $2; comm = $0
		sub(/^[[:space:]]*[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]+/, "", comm)
		base = comm; sub(/.*\//, "", base)
		if (pid == self) next
		if (base in drop) next
		printf "%s %.2f %s\n", pid, tsec(t), comm
	}' >"$1"

	# A real process table has hundreds of rows. A handful means `ps` was
	# blocked, sandboxed, or truncated, and comparing two such snapshots would
	# report a quiet host on no evidence.
	if [ "$(wc -l <"$1" | tr -d ' ')" -lt "${CPU_SNAPSHOT_MIN_ROWS:-20}" ]; then
		return 1
	fi
	return 0
}

# Take a snapshot into $1 and record WHY it holds nothing when it holds nothing.
# The three outcomes are different facts and a caller that cannot tell them
# apart is how an interference gate stops gating:
#
#   taken            the file has the process table in it
#   $1.failed        `ps` failed, was blocked or was truncated -- not knowing
#   $1.not-sampled   nobody looked, because the caller set SYNTHETIC_ARMS
#
# `SYNTHETIC_ARMS` is the shared boundary between an A/B harness that MEASURES
# and one exercising its own logic against stub arms. A stub arm is not a
# workload, so no fact about this machine belongs in that run's answer -- and a
# gate whose result depends on what else the machine is doing teaches everyone
# to re-run it until it goes green. Both harnesses (`bench/ab.sh`,
# `bench_llama_ab.sh`) set it from their own `--synthetic-arms` flag; it is read
# here and in `classify_window`, via `window_not_sampled`.
snapshot_ok() {
	if ${SYNTHETIC_ARMS:-false}; then
		: >"$1.not-sampled"
		return 0
	fi
	if cpu_snapshot "$1"; then
		return 0
	fi
	: >"$1.failed"
	return 1
}

# True when the window between snapshots $1 and $2 was deliberately not sampled.
window_not_sampled() {
	[ -e "$1.not-sampled" ] || [ -e "$2.not-sampled" ]
}

# "<state> <pct> <comm>" for the window between snapshots $1 and $2, $3 seconds apart.
# state is one of busy | quiet | unmeasured | not-sampled. The caller sets AWK_BUSIEST (the path of
# busiest_between.awk) and BUSY_PCT (the share of one core, in percent, at which a foreign process is busy).
#
# `unmeasured` stays distinct from `quiet` all the way into the report, and it
# covers three different ways of not knowing: the window was too short to divide
# by, a snapshot was empty, or `ps` failed outright. Folding any of them into
# "nothing was running" is how an interference gate quietly stops gating.
classify_window() {
	local raw pct
	if window_not_sampled "$1" "$2"; then
		echo "not-sampled - -"
		return
	fi
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
