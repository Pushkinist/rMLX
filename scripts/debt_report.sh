#!/usr/bin/env bash
# scripts/debt_report.sh — advisory technical-debt report (non-failing).
#
# Prints sibling-file/fn similarity ("twins"), debt counters, the add/remove
# line ratio since the last tag, and oversized docs. The scan and the
# normalisation live in scripts/lib/debt_report.py; this wrapper resolves the
# repo root and guarantees exit 0 so it can sit at the end of `make ci`
# alongside file_size_report.sh / target_size_report.sh.
#
# Usage:
#   bash scripts/debt_report.sh [--root DIR] [--since REF]

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "${REPO_ROOT}/scripts/lib/debt_report.py" --root "${REPO_ROOT}" "$@"
exit 0
