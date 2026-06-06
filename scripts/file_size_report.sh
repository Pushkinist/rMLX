#!/usr/bin/env bash
# scripts/file_size_report.sh — advisory LOC report for source files.
#
# Prints all non-test Rust source files above a threshold (default 1000 LOC),
# sorted by size descending. Non-failing; intended for informational use in CI.
#
# Usage:
#   bash scripts/file_size_report.sh [threshold]
#
# Examples:
#   bash scripts/file_size_report.sh          # threshold=1000
#   bash scripts/file_size_report.sh 500      # lower threshold

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
threshold=${1:-1000}

echo "Files >${threshold} LOC (excluding tests/, tests.rs, *_tests.rs):"
echo

find "${REPO_ROOT}/crates" -name "*.rs" \
    -not -path "*/target/*" \
    -not -path "*/tests/*" \
    -not -name "tests.rs" \
    -not -name "*_tests.rs" \
    -print0 \
  | xargs -0 wc -l 2>/dev/null \
  | awk -v t="$threshold" '$1 > t && $2 != "total" { printf "%6d  %s\n", $1, $2 }' \
  | sort -rn

total=$(
  find "${REPO_ROOT}/crates" -name "*.rs" \
    -not -path "*/target/*" \
    -not -path "*/tests/*" \
    -not -name "tests.rs" \
    -not -name "*_tests.rs" \
    -print0 \
  | xargs -0 wc -l 2>/dev/null \
  | tail -1 \
  | awk '{print $1}'
)

echo
echo "Workspace total (excluding test files): ${total} LOC"
