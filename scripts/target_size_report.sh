#!/usr/bin/env bash
# scripts/target_size_report.sh — advisory target/ size report for CI.
#
# `target/` has no size cap (see scripts/target_gc.sh) and a workspace this
# size with several profiles in regular use (dev, release, release-perf,
# release-debug) plus repeated `cargo test --workspace` runs can grow into the
# hundreds of GB. This prints the current size and, past a threshold, a hint
# pointing at `make target-gc`. Non-failing — advisory only, mirrors
# file_size_report.sh.
#
# Usage:
#   bash scripts/target_size_report.sh              # threshold=50 (GB)
#   bash scripts/target_size_report.sh 100           # custom threshold

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
threshold_gb=${1:-50}
target_dir="${REPO_ROOT}/target"

if [ ! -d "$target_dir" ]; then
	echo "target/ absent — nothing to report"
	exit 0
fi

size_kb=$(du -sk "$target_dir" 2>/dev/null | awk '{print $1}')
size_kb=${size_kb:-0}
size_h=$(du -sh "$target_dir" 2>/dev/null | awk '{print $1}')
size_gb=$((size_kb / 1024 / 1024))

echo "target/: ${size_h:-unknown}"

if [ "$size_gb" -ge "$threshold_gb" ]; then
	echo "target/ is ${size_gb}G (>= ${threshold_gb}G advisory threshold) — run 'make target-gc' to see what's stale, 'make target-gc APPLY=1' to prune"
fi

exit 0
