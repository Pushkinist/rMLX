#!/usr/bin/env bash
# Reclaim space in ./target by pruning build profiles that have gone stale.
#
# There is no built-in cap on `target/` — the size cap in this project is
# RMLX_LOG_CAP_MB, which governs <RMLX_HOME>/logs, not build artifacts. A
# workspace this size with several profiles (dev, release, release-perf,
# release-debug) plus a full `cargo test --workspace` accumulates hundreds of
# GB, dominated by `target/debug`.
#
# Everything under target/ is regenerable; the only cost of pruning is rebuild
# time. This script therefore prunes by *staleness*, and by default protects the
# profile holding the current bench binary.
#
# Usage:
#   bash scripts/target_gc.sh                # report only, no deletion
#   bash scripts/target_gc.sh --apply        # prune profiles older than MAX_AGE_DAYS
#   bash scripts/target_gc.sh --apply --all  # also prune release-perf
#
# Env:
#   MAX_AGE_DAYS  profile is stale if untouched this long (default 7)

set -uo pipefail

MAX_AGE_DAYS="${MAX_AGE_DAYS:-7}"
APPLY=0
ALL=0
for arg in "$@"; do
	case "$arg" in
	--apply) APPLY=1 ;;
	--all) ALL=1 ;;
	*)
		echo "unknown argument: $arg" >&2
		exit 2
		;;
	esac
done

cd "$(dirname "$0")/.." || exit 1
[ -d target ] || {
	echo "no target/ directory — nothing to do"
	exit 0
}

# release-perf holds the binary used for benching and the perf canary; losing it
# mid-investigation costs a full rebuild before the next measurement.
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
	# Age of the most recently touched artifact in the profile.
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
