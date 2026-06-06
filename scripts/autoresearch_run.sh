#!/usr/bin/env bash
# autoresearch_run.sh — single experiment run.
#
# Builds rMLX from current branch, kills competing MLX/Ollama, starts
# rMLX serve, runs Cross-Backend-Bench, parses median of target metric
# from last 3 summary.csv rows, appends to results.tsv, prints
# RESULT_<metric>=<value> on the last line.
#
# Exit 0 = bench succeeded (regardless of whether target improved).
# Exit 1 = build/bench/server crashed.
#
# Usage:
#   bash scripts/autoresearch_run.sh
#
# Env overrides:
#   PORT=62265 RUNS=3 MAX_TOKENS=4096 RMLX_DIR=/path/to/rmlx ...

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
RMLX_DIR="${RMLX_DIR:-${RMLX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
BENCH_DIR=${BENCH_DIR:-${CROSS_BENCH_ROOT:-../Cross-Backend-Bench}}
PORT=${PORT:-62265}                        # avoid colliding with bench-prod 62264
MODEL_PATH=${MODEL_PATH:-${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit}
MODEL_ID=$(basename "$MODEL_PATH")
MODEL_DISK_GB=${MODEL_DISK_GB:-36.6}
QUANT_SIG=${QUANT_SIG:-affine g64 b8}
RUNS=${RUNS:-3}
MAX_TOKENS=${MAX_TOKENS:-4096}
LOG_FILE=/tmp/autoresearch_serve.log

cd "$RMLX_DIR"

# ── Branch → target metric ──────────────────────────────────────────────────
BRANCH=$(git rev-parse --abbrev-ref HEAD)
case "$BRANCH" in
  autoresearch/rmlx-tps*)  TARGET_METRIC=decode_tps         TARGET_DIR=high ;;
  autoresearch/rmlx-ttft*) TARGET_METRIC=ttft_ms            TARGET_DIR=low  ;;
  autoresearch/rmlx-rss*)  TARGET_METRIC=peak_rss_mb        TARGET_DIR=low  ;;
  *) echo "ERROR: branch '$BRANCH' is not an autoresearch/* branch" >&2
     exit 1 ;;
esac

VERSION=$(git rev-parse --short HEAD)
COMMIT_SUBJECT=$(git log -1 --pretty=%s)
echo "── autoresearch run: branch=$BRANCH target=$TARGET_METRIC commit=$VERSION ──"
echo "── subject: $COMMIT_SUBJECT"

# ── Build ───────────────────────────────────────────────────────────────────
BUILD_START=$(date +%s)
if ! cargo build --release -p rmlx-cli 2>&1 | tail -40; then
  echo "BUILD_FAILED"
  exit 1
fi
BUILD_SECS=$(( $(date +%s) - BUILD_START ))
echo "build_secs=$BUILD_SECS"

# ── Kill competing MLX ──────────────────────────────────────────────────────
KILL_OLLAMA=1 bash "$BENCH_DIR/scripts/_kill_mlx.sh" || true
sleep 2

# ── Start rmlx serve ────────────────────────────────────────────────────────
# cwd=rmlx repo so rmlx writes its metrics/ here, not the bench dir.
( cd "$RMLX_DIR" && ./target/release/rmlx serve \
    --model "$MODEL_PATH" \
    --port "$PORT" \
    --device gpu ) >"$LOG_FILE" 2>&1 &
RMLX_PID=$!

cleanup() {
  kill "$RMLX_PID" 2>/dev/null || true
  sleep 1
  kill -9 "$RMLX_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for /v1/models (up to 120s)
READY=0
for _ in {1..120}; do
  sleep 1
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
done
if [[ $READY -ne 1 ]]; then
  echo "ERROR: rmlx serve did not become ready on :$PORT in 120s"
  tail -50 "$LOG_FILE" >&2
  exit 1
fi
echo "rmlx ready on :$PORT"

# Warmup — triggers model load.
curl -fsS -X POST -H 'content-type: application/json' \
  "http://127.0.0.1:$PORT/v1/chat/completions" \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" \
  >/dev/null || true

# ── Snapshot summary.csv line count BEFORE bench ────────────────────────────
PRE_LINES=$(wc -l <"$BENCH_DIR/metrics/summary.csv" 2>/dev/null || echo 0)

# ── Run bench ───────────────────────────────────────────────────────────────
cd "$BENCH_DIR"
RUN_QUANT_SIG="autoresearch ${BRANCH#autoresearch/} ${VERSION} ${QUANT_SIG}"
if ! uv run python -m runners.run_one \
    --backend rmlx \
    --backend-version "$VERSION" \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL_ID" \
    --model-path "$MODEL_PATH" \
    --quant "$RUN_QUANT_SIG" \
    --device gpu \
    --max-tokens "$MAX_TOKENS" \
    --runs "$RUNS" \
    --backend-pid "$RMLX_PID" \
    --model-disk-gb "$MODEL_DISK_GB"; then
  echo "BENCH_FAILED"
  exit 1
fi

# ── Parse: median of LAST $RUNS rows for target metric ──────────────────────
PARSED=$(python3 - <<PY
import csv, statistics, sys
path = "$BENCH_DIR/metrics/summary.csv"
runs = $RUNS
target = "$TARGET_METRIC"
rows = list(csv.DictReader(open(path)))
last = rows[-runs:]
def fnum(s):
    try: return float(s)
    except: return float("nan")
ttfts  = [fnum(r["ttft_ms"])     for r in last]
tpss   = [fnum(r["decode_tps"])  for r in last]
rsss   = [fnum(r["peak_rss_mb"]) for r in last]
def to_succ(s):
    s = str(s).strip().lower()
    return 1 if s in ("true","1","yes") else 0
succs  = [to_succ(r["success"])  for r in last]
out64  = last[-1]["output_first_64"][:200].replace("\n"," ").replace("\t"," ")
import math
def med(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.median(xs) if xs else 0.0
ttft_p50 = med(ttfts)
tps_p50  = med(tpss)
rss_p50  = med(rsss)
success_pct = 100.0 * sum(succs) / max(len(succs),1)
target_val = {"ttft_ms": ttft_p50, "decode_tps": tps_p50, "peak_rss_mb": rss_p50}[target]
print(f"TTFT_P50={ttft_p50:.4f}")
print(f"TPS_P50={tps_p50:.4f}")
print(f"RSS_P50={rss_p50:.4f}")
print(f"SUCCESS_PCT={success_pct:.1f}")
print(f"OUTPUT_FIRST_64={out64}")
print(f"RESULT_{target}={target_val:.4f}")
PY
)

if [[ -z "$PARSED" ]]; then
  echo "PARSE_FAILED"
  exit 1
fi

echo "── parsed metrics ──"
echo "$PARSED"

# ── Append to cross_history.csv (full-metric history, target-agnostic) ──────
CROSS_CSV="$BENCH_DIR/metrics/cross_history.csv"
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TTFT=$(echo "$PARSED" | awk -F= '/^TTFT_P50=/{print $2}')
TPS=$(echo  "$PARSED" | awk -F= '/^TPS_P50=/{print  $2}')
RSS=$(echo  "$PARSED" | awk -F= '/^RSS_P50=/{print  $2}')
SUCC=$(echo "$PARSED" | awk -F= '/^SUCCESS_PCT=/{print $2}')
OUT64=$(echo "$PARSED" | awk -F= '/^OUTPUT_FIRST_64=/{$1=""; sub(/^=/,""); print substr($0,2)}')

if [[ ! -f "$CROSS_CSV" ]]; then
  echo "timestamp_utc,branch,commit,target_metric,ttft_p50_ms,decode_tps_p50,peak_rss_mb_p50,success_pct,build_secs,description,output_first_64" > "$CROSS_CSV"
fi
DESC_CLEAN=$(echo "$COMMIT_SUBJECT" | tr ',\t' '  ' | cut -c1-120)
OUT64_CLEAN=$(echo "$OUT64" | tr ',\t' '  ' | cut -c1-200)
echo "$TS,$BRANCH,$VERSION,$TARGET_METRIC,$TTFT,$TPS,$RSS,$SUCC,$BUILD_SECS,\"$DESC_CLEAN\",\"$OUT64_CLEAN\"" >> "$CROSS_CSV"

# ── Final RESULT line (last) ────────────────────────────────────────────────
RESULT_LINE=$(echo "$PARSED" | grep '^RESULT_')
echo "$RESULT_LINE"
