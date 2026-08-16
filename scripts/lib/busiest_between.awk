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
#   unmeasured      the question was not answered: the window was too short to
#                   divide by, or a snapshot was empty. NOT the same as idle. A
#                   caller that treats it as idle re-creates the bug this whole
#                   file exists to prevent, so it is a distinct token.
#
# Known blind spot: only processes present in the AFTER snapshot are scored. A
# process that both started and exited inside the window appears in neither
# snapshot and contributes nothing. One that started inside and is still alive
# IS caught (it is charged all of its CPU time); one alive throughout is caught
# by the difference. Sustained contention -- the profile that actually skews a
# decode benchmark -- is therefore visible; a short-lived burst that fits
# entirely inside one slot is not.
#
# Why cumulative CPU time and not `ps -o pcpu`: on macOS pcpu is a stale decayed
# figure that does not move while a process pins a core. A difference of
# cumulative CPU seconds over a known wall-clock window is exact.

# File split by FILENAME, not by `NR == FNR`. That idiom is true for the SECOND
# file's records whenever the first file has zero records, so an empty `before`
# would route the whole `after` file into before[] and score nothing -- printing
# `idle` for a window in which a process burned CPU for the entire duration.
FILENAME == ARGV[1] {
	before[$1] = $2
	nbefore++
	next
}

{
	nafter++
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
	# `ps -o time` has 10 ms resolution, so a window shorter than ~0.1 s cannot
	# resolve a percentage: a single tick would read as 10% or more of a core
	# purely from quantisation. Callers time the window with millisecond
	# resolution, so this floor is about the CPU counter, not the clock.
	if (window + 0 < 0.1) {
		print "unmeasured - -"
		exit
	}
	# An empty snapshot on either side is a snapshot that was not taken -- a
	# failed, blocked or restricted `ps`. There is nothing to compare, and
	# calling that `idle` would disable every gate built on this file.
	if (nbefore == 0 || nafter == 0) {
		print "unmeasured - -"
		exit
	}
	if (best <= 0) {
		print "idle 0.0 -"
		exit
	}
	printf "busy-candidate %.1f %s\n", 100 * best / window, who
}
