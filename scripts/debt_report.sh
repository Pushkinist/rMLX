#!/usr/bin/env bash
# scripts/debt_report.sh — advisory technical-debt report (non-failing).
#
# Prints sibling-file/fn similarity ("twins"), debt counters, the add/remove
# line ratio since the last tag, and oversized docs. The scan and the
# normalisation live in scripts/lib/debt_report.py; this wrapper resolves the
# repo root and guarantees exit 0 so it can sit at the end of `make ci`
# alongside file_size_report.sh / target_size_report.sh. That guarantee is
# for the advisory report only: `--matched-lines` is a measurement, not a
# report, so its exit code is forwarded rather than swallowed — a broken
# producer must fail loud, not print no figure and exit 0.
#
# The --matched-lines populations, their roots and their pairing rules are
# documented in scripts/lib/debt_report.py's header; --help lists the names.
#
# Usage:
#   bash scripts/debt_report.sh [--root DIR] [--since REF]
#   bash scripts/debt_report.sh --matched-lines \
#       {drivers,impls,iso-storage,iso-updates,rotor-storage,rotor-updates,ssd-hydrate}

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

matched_lines=0
for arg in "$@"; do
  [ "$arg" = "--matched-lines" ] && matched_lines=1
done

python3 "${REPO_ROOT}/scripts/lib/debt_report.py" --root "${REPO_ROOT}" "$@"
status=$?

if [ "$matched_lines" = "1" ]; then
  exit "$status"
fi
exit 0
