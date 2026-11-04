#!/usr/bin/env bash
# Bound a regenerable directory: prune stale target/ profiles, or evict old .rmlx/traces bundles.
#
# Usage:  bash scripts/host/gc.sh target [--apply] [--all] [--max-age-days N] [--warn-gb N]
#         bash scripts/host/gc.sh traces [--apply] [--all] [--max-age-days N] [--max-count N]
#                                        [--max-total-gb N] [--dir DIR]
#         Both modes report and delete nothing unless --apply is given. Every number is a whole
#         number of at most 9 digits; anything else is refused before anything is listed.
# Exit:   0 reported, pruned or nothing to do; 1 the repository root cannot be entered; 2 usage error
# Set:    -uo — a failed `du` or `stat` is read as an empty size or a skipped entry, not a reason to stop
# Not a mode of: profile/gputrace.sh — that one reads a finished .gputrace bundle; this one deletes them
#
# target: everything under target/ is regenerable, so profiles are pruned by staleness (default 7
# days, newest file inside). release-perf holds the bench binary and is kept unless --all is given.
# `target --warn-gb N` without --apply is the size report: the size, and a hint at or past N GB.
#
# traces: a trace is not cheap to regenerate (a model load plus a capture run), so the collection is
# bounded instead of expired. In the order applied, the newest bundle never being a candidate unless
# --all is given:
#   1. --max-count N     keep the N newest bundles           (default 6)
#   2. --max-total-gb N  then evict oldest-first until the directory fits (default 40)
#   3. --max-age-days N  optional extra rule; off unless the flag is given
# Every eviction is printed with its reason: a capture that vanishes silently costs a re-run.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODE=""
APPLY=0
ALL=0
MAX_AGE_DAYS=""
MAX_COUNT=6
MAX_TOTAL_GB=40
WARN_GB=""
DIR=".rmlx/traces"

usage_text() {
	echo "usage: $0 {target|traces} [--apply] [--all] [--max-age-days N] [--max-count N] [--max-total-gb N] [--warn-gb N] [--dir DIR]"
}

refuse() {
	echo "$1" >&2
	usage_text >&2
	exit 2
}

need_arg() {
	[ "$2" -ge 2 ] || refuse "$1 needs a value"
}

need_number() {
	case "$2" in
	"" | *[!0-9]*) refuse "$1 must be a whole number (got '$2')" ;;
	esac
	[ "${#2}" -le 9 ] || refuse "$1 is too large (got '$2')"
}

only_in() {
	[ "$MODE" = "$1" ] || refuse "$2 does not apply to the $MODE mode"
}

MODE="${1:-}"
case "$MODE" in
target | traces) shift ;;
-h | --help)
	usage_text
	exit 0
	;;
*) refuse "needs a mode, target or traces (got '$MODE')" ;;
esac

while [ $# -gt 0 ]; do
	case "$1" in
	--apply)
		APPLY=1
		shift
		;;
	--all)
		ALL=1
		shift
		;;
	--max-age-days)
		need_arg "$1" $#
		need_number "$1" "$2"
		MAX_AGE_DAYS=$((10#$2))
		shift 2
		;;
	--max-count)
		only_in traces "$1"
		need_arg "$1" $#
		need_number "$1" "$2"
		MAX_COUNT=$((10#$2))
		shift 2
		;;
	--max-total-gb)
		only_in traces "$1"
		need_arg "$1" $#
		need_number "$1" "$2"
		MAX_TOTAL_GB=$((10#$2))
		shift 2
		;;
	--dir)
		only_in traces "$1"
		need_arg "$1" $#
		DIR="$2"
		shift 2
		;;
	--warn-gb)
		only_in target "$1"
		need_arg "$1" $#
		need_number "$1" "$2"
		WARN_GB=$((10#$2))
		shift 2
		;;
	-h | --help)
		usage_text
		exit 0
		;;
	*) refuse "unknown argument: $1" ;;
	esac
done

if [ -n "$WARN_GB" ] && { [ "$APPLY" = "1" ] || [ "$ALL" = "1" ]; }; then
	refuse "--warn-gb is the size report and takes neither --apply nor --all"
fi

[ -n "$REPO_ROOT" ] && cd "$REPO_ROOT" || exit 1

size_report() {
	if [ ! -d target ]; then
		echo "target/ absent — nothing to report"
		exit 0
	fi

	size_kb=$(du -sk target 2>/dev/null | awk '{print $1}')
	size_kb=${size_kb:-0}
	size_h=$(du -sh target 2>/dev/null | awk '{print $1}')
	size_gb=$((size_kb / 1024 / 1024))

	echo "target/: ${size_h:-unknown}"

	if [ "$size_gb" -ge "$WARN_GB" ]; then
		echo "target/ is ${size_gb}G (>= ${WARN_GB}G advisory threshold) — run 'make target-gc' to see what's stale, 'make target-gc APPLY=1' to prune"
	fi
}

gc_target() {
	[ -n "$MAX_AGE_DAYS" ] || MAX_AGE_DAYS=7
	if [ ! -d target ]; then
		echo "no target/ directory — nothing to do"
		exit 0
	fi

	protected="release-perf"
	[ "$ALL" = "1" ] && protected=""

	total_before=$(du -sk target 2>/dev/null | awk '{print $1}')
	reclaimed=0

	for dir in target/*/; do
		profile=$(basename "$dir")
		case "$profile" in
		doc | criterion | package | CACHEDIR.TAG) continue ;;
		esac

		size_k=$(du -sk "$dir" 2>/dev/null | awk '{print $1}')
		size_h=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
		newest=$(find "$dir" -type f -newermt "-${MAX_AGE_DAYS} days" -print -quit 2>/dev/null)

		if [ "$profile" = "$protected" ]; then
			echo "keep    $profile ($size_h) — protected (bench binary; use --all to override)"
			continue
		fi
		if [ -n "$newest" ]; then
			echo "keep    $profile ($size_h) — touched within ${MAX_AGE_DAYS}d"
			continue
		fi

		if [ "$APPLY" = "1" ]; then
			echo "prune   $profile ($size_h) — stale >${MAX_AGE_DAYS}d"
			rm -rf "${dir:?}"
			reclaimed=$((reclaimed + size_k))
		else
			echo "would   $profile ($size_h) — stale >${MAX_AGE_DAYS}d"
		fi
	done

	if [ "$APPLY" = "1" ]; then
		total_after=$(du -sk target 2>/dev/null | awk '{print $1}')
		echo ""
		echo "target: $((total_before / 1024 / 1024))G -> $((total_after / 1024 / 1024))G (reclaimed $((reclaimed / 1024 / 1024))G)"
	else
		echo ""
		echo "report only — re-run with --apply to prune. Current target: $((total_before / 1024 / 1024))G"
	fi
}

gc_traces() {
	if [ ! -d "$DIR" ]; then
		echo "no $DIR — nothing to do"
		exit 0
	fi

	# Oldest first, so eviction order is simply list order and "newest" is the tail.
	listing=$(
		for b in "$DIR"/*.gputrace; do
			[ -d "$b" ] || continue
			m=$(stat -f %m "$b" 2>/dev/null) || continue
			[ -n "$m" ] && printf '%s\t%s\n' "$m" "$b"
		done | sort -n
	)
	if [ -z "$listing" ]; then
		echo "$DIR holds no .gputrace bundles — nothing to do"
		exit 0
	fi

	bundles=()
	while IFS=$'\t' read -r _ path; do
		[ -n "$path" ] && bundles+=("$path")
	done <<<"$listing"

	n=${#bundles[@]}

	kb_of() { du -sk "$1" 2>/dev/null | awk '{print $1}'; }
	h_of() { du -sh "$1" 2>/dev/null | awk '{print $1}'; }

	# The plan is printed whether or not --apply is set, so a dry run shows what an apply removes.
	declare -a reason
	running_total=0
	sizes=()
	for i in $(seq 0 $((n - 1))); do
		k=$(kb_of "${bundles[$i]}")
		k=${k:-0}
		sizes+=("$k")
		running_total=$((running_total + k))
		reason[i]=""
	done
	# Only the bundles this tool can evict count: .rmlx/traces/mst/ holds Metal System Trace
	# bundles with their own retention in scripts/mst_capture.sh.
	total_before=$running_total

	max_total_kb=$((MAX_TOTAL_GB * 1024 * 1024))

	if [ -n "$MAX_AGE_DAYS" ]; then
		for i in $(seq 0 $((n - 1))); do
			fresh=$(find "${bundles[$i]}" -maxdepth 0 -newermt "-${MAX_AGE_DAYS} days" -print -quit 2>/dev/null)
			[ -z "$fresh" ] && reason[i]="stale >${MAX_AGE_DAYS}d"
		done
	fi

	if [ "$n" -gt "$MAX_COUNT" ]; then
		over=$((n - MAX_COUNT))
		for i in $(seq 0 $((over - 1))); do
			[ -z "${reason[$i]}" ] && reason[i]="beyond the newest $MAX_COUNT"
		done
	fi

	projected=0
	for i in $(seq 0 $((n - 1))); do
		[ -z "${reason[$i]}" ] && projected=$((projected + sizes[i]))
	done
	if [ "$projected" -gt "$max_total_kb" ]; then
		for i in $(seq 0 $((n - 1))); do
			[ "$projected" -le "$max_total_kb" ] && break
			[ -n "${reason[$i]}" ] && continue
			reason[i]="over the ${MAX_TOTAL_GB}G cap"
			projected=$((projected - sizes[i]))
		done
	fi

	if [ "$ALL" != "1" ]; then
		reason[$((n - 1))]=""
	fi

	reclaimed=0
	evicted=0
	for i in $(seq 0 $((n - 1))); do
		name=$(basename "${bundles[$i]}")
		size_h=$(h_of "${bundles[$i]}")
		if [ -z "${reason[$i]}" ]; then
			if [ "$i" -eq $((n - 1)) ] && [ "$ALL" != "1" ]; then
				echo "keep    $name ($size_h) — newest bundle (use --all to override)"
			else
				echo "keep    $name ($size_h)"
			fi
			continue
		fi
		if [ "$APPLY" = "1" ]; then
			echo "prune   $name ($size_h) — ${reason[$i]}"
			rm -rf "${bundles[$i]:?}"
			reclaimed=$((reclaimed + sizes[i]))
		else
			echo "would   $name ($size_h) — ${reason[$i]}"
		fi
		evicted=$((evicted + 1))
	done

	gb() { echo $(($1 / 1024 / 1024)); }

	echo ""
	if [ "$APPLY" = "1" ]; then
		total_after=$((total_before - reclaimed))
		echo "$DIR: $(gb "$total_before")G -> $(gb "$total_after")G, $evicted pruned (reclaimed $(gb "$reclaimed")G); caps: $MAX_COUNT bundles / ${MAX_TOTAL_GB}G"
	else
		if [ "$evicted" -eq 0 ]; then
			echo "$DIR: $n bundles, $(gb "$total_before")G — within caps ($MAX_COUNT bundles / ${MAX_TOTAL_GB}G), nothing to prune"
		else
			echo "report only — re-run with --apply to prune $evicted bundle(s). $DIR: $n bundles, $(gb "$total_before")G (caps: $MAX_COUNT bundles / ${MAX_TOTAL_GB}G)"
		fi
	fi
}

if [ "$MODE" = "traces" ]; then
	gc_traces
elif [ -n "$WARN_GB" ]; then
	size_report
else
	gc_target
fi

exit 0
