#!/usr/bin/env bash
# regression_gate.sh — compare a committed baseline against the latest canary row.
#
# LEGACY: This script reads the CSV-based canary flow. The authoritative
# regression gate is now `make canary-gate SHA=<sha>` which calls
# `rmlx metrics deltas` against runs.db. This script is preserved as a fallback
# for one release; prefer `make canary-gate` for new workflows.
#
# Usage:
#   scripts/regression_gate.sh <model> <baseline_tps> <baseline_stddev> [--tolerance PCT]
#
# Exit codes:
#   0   — within tolerance (ok)
#   1   — regression detected (1-124 = bad for git bisect)
#   125 — measurement precondition failure (binary absent, model path missing,
#           no canary row) — tells git bisect to SKIP this commit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${REPO_ROOT}/target/release-perf/rmlx"
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
CSV="${RMLX_HOME}/bench/perf_canary.csv"

DEFAULT_TOLERANCE=3

# ---- Argument parsing --------------------------------------------------------
if [[ $# -lt 3 ]]; then
    echo "usage: $0 <model> <baseline_tps> <baseline_stddev> [--tolerance PCT]" >&2
    exit 125
fi

MODEL="$1"
BASELINE_TPS="$2"
BASELINE_STDDEV="$3"
shift 3

TOLERANCE="${DEFAULT_TOLERANCE}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tolerance)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tolerance requires a value" >&2
                exit 125
            fi
            TOLERANCE="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 125
            ;;
    esac
done

# ---- Precondition checks (exit 125 on any failure) ---------------------------

# Binary must exist
if [[ ! -x "${BINARY}" ]]; then
    echo "skip: binary not found at ${BINARY}" >&2
    exit 125
fi

# CSV must exist
if [[ ! -f "${CSV}" ]]; then
    echo "skip: canary CSV not found at ${CSV}" >&2
    exit 125
fi

# Find the LATEST row for this model (last matching line in the append-only CSV)
# CSV columns: ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile
LATEST_ROW="$(grep ",${MODEL}," "${CSV}" 2>/dev/null | tail -1 || true)"

if [[ -z "${LATEST_ROW}" ]]; then
    echo "skip: no canary row found for model '${MODEL}' in ${CSV}" >&2
    exit 125
fi

# Parse current_tps (column 6) and current_stddev (column 7) from the CSV row
# All float math is done in awk to avoid bash integer truncation.
CURRENT_TPS="$(echo "${LATEST_ROW}" | awk -F',' '{print $6}')"
CURRENT_STDDEV="$(echo "${LATEST_ROW}" | awk -F',' '{print $7}')"

if [[ -z "${CURRENT_TPS}" ]] || [[ -z "${CURRENT_STDDEV}" ]]; then
    echo "skip: could not parse tps/stddev from CSV row: ${LATEST_ROW}" >&2
    exit 125
fi

# ---- All float comparisons via awk -------------------------------------------
# awk returns: "ok|<delta_pct>" or "regression|<delta_pct>" or "widened_ok|<delta_pct>"
VERDICT="$(awk -v baseline="${BASELINE_TPS}" \
               -v current="${CURRENT_TPS}" \
               -v cur_sd="${CURRENT_STDDEV}" \
               -v baseline_sd="${BASELINE_STDDEV}" \
               -v tol="${TOLERANCE}" \
'BEGIN {
    delta_pct = (baseline - current) / baseline * 100

    # Strict check
    if (delta_pct <= tol) {
        printf "ok|%.4f\n", delta_pct
        exit
    }

    # Check if stddev justifies widening: current_stddev > 0.5 * baseline_tps * tolerance/100
    sd_threshold = 0.5 * baseline * tol / 100
    if (cur_sd > sd_threshold) {
        # Widen tolerance to 5%
        widened_tol = 5
        if (delta_pct <= widened_tol) {
            printf "widened_ok|%.4f|%.4f\n", delta_pct, widened_tol
        } else {
            printf "widened_regression|%.4f|%.4f\n", delta_pct, widened_tol
        }
    } else {
        printf "regression|%.4f\n", delta_pct
    }
}')"

VERDICT_TYPE="${VERDICT%%|*}"
DELTA_PCT="$(echo "${VERDICT}" | awk -F'|' '{print $2}')"

case "${VERDICT_TYPE}" in
    ok)
        echo "ok: model=${MODEL} delta=${DELTA_PCT}% (within ±${TOLERANCE}%)"
        exit 0
        ;;
    widened_ok)
        WIDENED_TOL="$(echo "${VERDICT}" | awk -F'|' '{print $3}')"
        echo "strict: regression: model=${MODEL} baseline=${BASELINE_TPS} current=${CURRENT_TPS} delta=${DELTA_PCT}% tolerance=${TOLERANCE}%"
        echo "widened: ok: model=${MODEL} delta=${DELTA_PCT}% (stddev justified widening to ±${WIDENED_TOL}%)"
        exit 0
        ;;
    widened_regression)
        WIDENED_TOL="$(echo "${VERDICT}" | awk -F'|' '{print $3}')"
        echo "strict: regression: model=${MODEL} baseline=${BASELINE_TPS} current=${CURRENT_TPS} delta=${DELTA_PCT}% tolerance=${TOLERANCE}%"
        echo "widened: regression: model=${MODEL} baseline=${BASELINE_TPS} current=${CURRENT_TPS} delta=${DELTA_PCT}% tolerance=${WIDENED_TOL}%"
        exit 1
        ;;
    regression)
        echo "regression: model=${MODEL} baseline=${BASELINE_TPS} current=${CURRENT_TPS} delta=${DELTA_PCT}% tolerance=${TOLERANCE}%"
        exit 1
        ;;
    *)
        echo "ERROR: unexpected awk verdict: ${VERDICT}" >&2
        exit 125
        ;;
esac
