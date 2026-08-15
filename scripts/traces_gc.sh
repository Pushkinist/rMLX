#!/usr/bin/env bash
# Bound how much disk .rmlx/traces holds, and say exactly what was removed.
#
# A .gputrace is roughly the model's resident footprint — ~6 GB for an e2b-class
# model — because capture snapshots every resident GPU buffer. Four of them
# arrive in one afternoon of A/B work. Unlike target/, a trace is *not* cheap to
# regenerate: each one costs a model load plus a capture run, so the policy is
# to bound the collection, not to expire individual bundles on a timer.
#
# Retention, in the order applied. The newest bundle is never a candidate (it is
# the one being looked at right now) unless --all is passed:
#
#   1. --max-count N     keep the N newest bundles           (default 6)
#   2. --max-total-gb N  then evict oldest-first until the directory fits (40)
#   3. --max-age-days N  optional extra rule; off unless the flag is given
#
# Eviction is always oldest-first and always printed with its reason and the
# space reclaimed. A capture that disappears without a line of output is worse
# than a full disk — the operator re-runs a 6 GB capture wondering where it went.
#
# Deliberately flag-driven, not env-driven: the project treats a new environment
# variable as a support burden (see CLAUDE.md, simplicity rules).
#
# Usage:
#   bash scripts/traces_gc.sh                        # report only, nothing deleted
#   bash scripts/traces_gc.sh --apply                # enforce the caps
#   bash scripts/traces_gc.sh --apply --max-count 12 --max-total-gb 80
#   bash scripts/traces_gc.sh --apply --max-age-days 7
#   bash scripts/traces_gc.sh --apply --all          # newest is evictable too
#   bash scripts/traces_gc.sh --dir /path/to/traces
#
# Exit: 0 on every advisory path (nothing to do, report only, caps enforced),
#       2 on usage.

set -uo pipefail

MAX_COUNT=6
MAX_TOTAL_GB=40
MAX_AGE_DAYS="" # optional; empty = no age rule
APPLY=0
ALL=0
DIR=".rmlx/traces"

need_arg() {
	if [ "$2" -lt 2 ]; then
		echo "$1 needs a value" >&2
		exit 2
	fi
}
need_number() {
	case "$2" in
	"" | *[!0-9]*)
		echo "$1 must be a whole number (got '$2')" >&2
		exit 2
		;;
	esac
}

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
	--max-count)
		need_arg "$1" $#
		need_number "$1" "$2"
		MAX_COUNT="$2"
		shift 2
		;;
	--max-total-gb)
		need_arg "$1" $#
		need_number "$1" "$2"
		MAX_TOTAL_GB="$2"
		shift 2
		;;
	--max-age-days)
		need_arg "$1" $#
		need_number "$1" "$2"
		MAX_AGE_DAYS="$2"
		shift 2
		;;
	--dir)
		need_arg "$1" $#
		DIR="$2"
		shift 2
		;;
	-h | --help)
		echo "usage: $0 [--apply] [--all] [--max-count N] [--max-total-gb N] [--max-age-days N] [--dir DIR]"
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		echo "usage: $0 [--apply] [--all] [--max-count N] [--max-total-gb N] [--max-age-days N] [--dir DIR]" >&2
		exit 2
		;;
	esac
done

cd "$(dirname "$0")/.." || exit 1

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

# Decide first, act second: the plan is printed whether or not --apply is set,
# so a dry run shows exactly what an apply would remove.
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
# Scoped to the bundles this script can actually evict, NOT `du` over the whole
# directory: `.rmlx/traces/mst/` holds Metal System Trace bundles, which have
# their own retention in scripts/mst_capture.sh and are invisible to the glob
# above. Counting them here would report a total against a cap that can never
# act on it.
total_before=$running_total

max_total_kb=$((MAX_TOTAL_GB * 1024 * 1024))

# Rule 3 (optional): age.
if [ -n "$MAX_AGE_DAYS" ]; then
	for i in $(seq 0 $((n - 1))); do
		fresh=$(find "${bundles[$i]}" -maxdepth 0 -newermt "-${MAX_AGE_DAYS} days" -print -quit 2>/dev/null)
		[ -z "$fresh" ] && reason[i]="stale >${MAX_AGE_DAYS}d"
	done
fi

# Rule 1: count. Everything but the newest MAX_COUNT goes, oldest first.
if [ "$n" -gt "$MAX_COUNT" ]; then
	over=$((n - MAX_COUNT))
	for i in $(seq 0 $((over - 1))); do
		[ -z "${reason[$i]}" ] && reason[i]="beyond the newest $MAX_COUNT"
	done
fi

# Rule 2: total size. Oldest first until what remains fits.
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

# The newest bundle is what an operator is looking at right now.
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

exit 0
