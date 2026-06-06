#!/usr/bin/env bash
# diff_baseline.sh — compare two perf-iter JSONL files and emit per-cell deltas.
#
# Usage:
#   ./scripts/perf-iter/diff_baseline.sh <baseline.jsonl> <current.jsonl>
#   ./scripts/perf-iter/diff_baseline.sh --threshold 5 <baseline.jsonl> <current.jsonl>
#
# Each JSONL must follow the metrics/perf-iter schema (produced by bench_decode_tps.sh):
#   { "model_path": "...", "kv_quant": "...", "decode_tps_mean": 106.44,
#     "decode_tps_stddev": 2.08, "git_sha": "a6e7b9d", ... }
#
# For each (model_basename × kv_quant) pair that exists in BOTH files, prints:
#   PASS/FAIL  model_basename  kv_quant  baseline_tps  current_tps  delta%
#
# Exit code: 0 if all cells pass, 1 if any cell regresses beyond --threshold.
#
# Flags:
#   --threshold N  — regression threshold in percent (default: 5)
#                    A cell FAILS if current_tps < baseline_tps * (1 - N/100).
#                    Improvements are always PASS.
#
# Notes:
#   - When multiple rows share the same (model_basename, kv_quant) key, the
#     LAST row wins for each file (most recent run is authoritative).
#   - System noise typically drifts absolute numbers ±3-4% between sessions even
#     with no code change (thermal, scheduler, DRAM refresh). The default 5%
#     threshold sits just above that band.  For within-session pre/post
#     comparisons (same boot, same thermal state), 2% is safe.
#   - Use this script for commit-level comparisons, not cross-day baselines.
#     See metrics/perf-iter/README.md for the full methodology.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────

THRESHOLD=5
BASELINE_FILE=""
CURRENT_FILE=""

# ── Argument parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold)
            [[ $# -ge 2 ]] || { echo "ERROR: --threshold requires a numeric argument" >&2; exit 1; }
            THRESHOLD="$2"
            shift 2
            ;;
        --threshold=*)
            THRESHOLD="${1#*=}"
            shift
            ;;
        -*)
            echo "ERROR: unknown flag: $1" >&2
            echo "Usage: $0 [--threshold N] <baseline.jsonl> <current.jsonl>" >&2
            exit 1
            ;;
        *)
            if [[ -z "${BASELINE_FILE}" ]]; then
                BASELINE_FILE="$1"
            elif [[ -z "${CURRENT_FILE}" ]]; then
                CURRENT_FILE="$1"
            else
                echo "ERROR: unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

[[ -n "${BASELINE_FILE}" && -n "${CURRENT_FILE}" ]] || {
    echo "Usage: $0 [--threshold N] <baseline.jsonl> <current.jsonl>" >&2
    exit 1
}
[[ -f "${BASELINE_FILE}" ]] || { echo "ERROR: baseline file not found: ${BASELINE_FILE}" >&2; exit 1; }
[[ -f "${CURRENT_FILE}" ]]  || { echo "ERROR: current file not found: ${CURRENT_FILE}" >&2; exit 1; }

command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 required" >&2; exit 1; }

# ── Comparison logic (Python inline) ─────────────────────────────────────────

python3 - "${BASELINE_FILE}" "${CURRENT_FILE}" "${THRESHOLD}" <<'PYEOF'
import json
import sys
import os

baseline_path = sys.argv[1]
current_path  = sys.argv[2]
threshold_pct = float(sys.argv[3])

def load_jsonl(path):
    """Return dict keyed by (model_basename, kv_quant) -> record (last row wins)."""
    rows = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN: {path}:{lineno}: skipping malformed line ({e})", file=sys.stderr)
                continue
            model_path = rec.get("model_path", "")
            model_base = os.path.basename(model_path.rstrip("/"))
            kv_quant   = rec.get("kv_quant", "unknown")
            key = (model_base, kv_quant)
            rows[key] = rec
    return rows

baseline = load_jsonl(baseline_path)
current  = load_jsonl(current_path)

common_keys = sorted(set(baseline) & set(current))

if not common_keys:
    print("WARN: no (model_basename, kv_quant) pairs in common between the two files.")
    print(f"  baseline keys: {sorted(baseline)}")
    print(f"  current  keys: {sorted(current)}")
    sys.exit(0)

# Header
print(f"\nthreshold: ±{threshold_pct}%   baseline: {baseline_path}   current: {current_path}\n")
col_model = max((len(k[0]) for k in common_keys), default=12)
header = (
    f"{'RESULT':<7}  "
    f"{'MODEL':<{col_model}}  "
    f"{'KV':<8}  "
    f"{'BASELINE_TPS':>12}  "
    f"{'CURRENT_TPS':>11}  "
    f"{'DELTA%':>7}  "
    f"{'STD_B':>7}  "
    f"{'STD_C':>7}"
)
print(header)
print("-" * len(header))

any_fail = False

for (model_base, kv_quant) in common_keys:
    b = baseline[(model_base, kv_quant)]
    c = current[(model_base, kv_quant)]

    b_tps = b.get("decode_tps_mean", 0.0)
    c_tps = c.get("decode_tps_mean", 0.0)
    b_std = b.get("decode_tps_stddev", 0.0)
    c_std = c.get("decode_tps_stddev", 0.0)

    if b_tps <= 0.0:
        print(f"SKIP     {model_base:<{col_model}}  {kv_quant:<8}  baseline_tps=0, skipping")
        continue

    delta_pct = (c_tps - b_tps) / b_tps * 100.0
    fail = delta_pct < -threshold_pct
    result = "FAIL" if fail else "PASS"
    if fail:
        any_fail = True

    print(
        f"{result:<7}  "
        f"{model_base:<{col_model}}  "
        f"{kv_quant:<8}  "
        f"{b_tps:>12.2f}  "
        f"{c_tps:>11.2f}  "
        f"{delta_pct:>+7.2f}%  "
        f"{b_std:>7.2f}  "
        f"{c_std:>7.2f}"
    )

# Keys only in baseline or only in current (informational)
only_baseline = sorted(set(baseline) - set(current))
only_current  = sorted(set(current)  - set(baseline))
if only_baseline:
    print(f"\nIn baseline only (no current measurement): {only_baseline}")
if only_current:
    print(f"In current only  (new measurement):        {only_current}")

# Footer
print()
if any_fail:
    print(f"RESULT: FAIL — one or more cells regressed >{threshold_pct}%")
    sys.exit(1)
else:
    print("RESULT: PASS — all cells within threshold")
    sys.exit(0)
PYEOF
