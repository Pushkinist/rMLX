# busiest_between.awk — which process burned the most CPU between two snapshots.
#
# Usage:  awk -v window=<wall_seconds> -f busiest_between.awk <before> <after>
#
# Each snapshot is `<pid> <cumulative_cpu_seconds> <command>` per line, as
# produced by `cpu_snapshot` in scripts/perf_ab.sh.
#
# Prints one line: `<state> <pct> <command>` where state is
#
#   busy-candidate  a process burned CPU during the window; <pct> is its share
#                   of one core over the window. The caller applies the
#                   threshold -- this file does not know what counts as busy.
#   idle            nothing measurably burned CPU.
#   unmeasured      the window was too short to divide by. NOT the same as
#                   idle: it means the question was not answered. A caller that
#                   treats it as idle re-creates the bug this whole file exists
#                   to prevent, so it is a distinct token.
#
# Why cumulative CPU time and not `ps -o pcpu`: on macOS pcpu is a stale decayed
# figure that does not move while a process pins a core. A difference of
# cumulative CPU seconds over a known wall-clock window is exact.

NR == FNR {
	before[$1] = $2
	next
}

{
	pid = $1
	now = $2
	comm = $0
	sub(/^[[:space:]]*[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+/, "", comm)
	# A pid missing from the first snapshot started inside the window, so all
	# of its CPU time was spent in there.
	prev = (pid in before) ? before[pid] : 0
	d = now - prev
	if (d > best) {
		best = d
		who = comm
	}
}

END {
	if (window + 0 < 1) {
		print "unmeasured - -"
		exit
	}
	if (best <= 0) {
		print "idle 0.0 -"
		exit
	}
	printf "busy-candidate %.1f %s\n", 100 * best / window, who
}
