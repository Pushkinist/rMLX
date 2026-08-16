#!/usr/bin/env bash
# cpu_snapshot.sh — per-process cumulative CPU seconds, for interference gates.
#
# Defines one function. Source it; do not execute it.
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
