#!/usr/bin/env bash
# bench-records-sweep.sh — 5-model × 4-KV-quant BENCHMARK_CHAMPIONS regression sweep.
#
# Methodology (BENCHMARK_CHAMPIONS.md protocol):
#   Prompt   : Cross-Backend-Bench/prompts/longctx_4k.json (4096-token)
#   Server   : --max-ctx 8192, port 62265
#   Decode   : max_tokens=32
#   Runs     : 2 per cell (r0=cold, r1=warm)
#   Metrics  : decode_tps_warm, prefill_tps, ttft_cold, ttft_warm, peak_rss
#   Output   : rows appended to Cross-Backend-Bench/metrics/summary.csv
#
# Teardown between every cell (CLAUDE.md single-MLX-process rule):
#   pkill -f "rmlx serve"; pkill -f mlx_lm; pkill -f paroquant; pkill -f omlx; sleep 5; rm -f /tmp/rmlx.62265.claim
#
# Usage:
#   cd ${RMLX_ROOT}
#   bash scripts/bench-records-sweep.sh [MODEL_FILTER] [KV_FILTER]
#
#   MODEL_FILTER: substring of model dirname; if set, only matching models run
#   KV_FILTER   : one of bf16/k8v4/k8v8/planar; if set, only that KV runs
set -euo pipefail

RMLX_DIR="${RMLX_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CBB_DIR=${CROSS_BENCH_ROOT:-../Cross-Backend-Bench}
RMLX_BIN="$RMLX_DIR/target/release/rmlx"
PORT=62265
MAX_CTX=8192
MAX_TOKENS=32
RUNS=2
PROMPT_FILE="$CBB_DIR/prompts/longctx_4k.json"
PROMPT_TOKENS=4096

MODEL_FILTER="${1:-}"
KV_FILTER="${2:-}"

VERSION=$(cd "$RMLX_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)
LOG_DIR="$RMLX_DIR/logs/bench-records-sweep"
mkdir -p "$LOG_DIR"
SWEEP_LOG="$LOG_DIR/sweep_$(date -u +%Y%m%d-%H%M%S)_${VERSION}.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$SWEEP_LOG"; }
log "=== bench-records-sweep START  rmlx=$VERSION ==="
log "Port=$PORT  max-ctx=$MAX_CTX  max-tokens=$MAX_TOKENS  runs=$RUNS"

# ── Verify prerequisites ──────────────────────────────────────────────────────
[[ -x "$RMLX_BIN" ]] || { log "ERROR: rmlx binary missing at $RMLX_BIN"; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { log "ERROR: prompt file missing: $PROMPT_FILE"; exit 1; }

# ── Model table ───────────────────────────────────────────────────────────────
# Format: "MODEL_PATH|QUANT_SIG|DISK_GB"
MODELS=(
  "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit|affine g64 b8|35.0"
  "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/z-lab__Qwen3.6-27B-PARO|paroquant int4|18.0"
  "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/prism-ml__Ternary-Bonsai-8B-mlx-2bit|2-bit ternary|2.2"
  "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-e2b-it-mxfp8|mxfp8 g32|5.4"
  "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__medgemma-1.5-4b-it-8bit|affine g64 b8|5.3"
)

KV_MODES=(bf16 k8v4 k8v8 planar)

# ── Teardown helper ───────────────────────────────────────────────────────────
teardown() {
  log "  [teardown] pkill rmlx serve / mlx_lm / paroquant / omlx..."
  pkill -f "rmlx serve"  >/dev/null 2>&1 || true
  pkill -f "mlx_lm"      >/dev/null 2>&1 || true
  pkill -f "paroquant"   >/dev/null 2>&1 || true
  pkill -f "omlx"        >/dev/null 2>&1 || true
  sleep 5
  rm -f /tmp/rmlx.${PORT}.claim 2>/dev/null || true
  log "  [teardown] done"
}

# ── Sanity-check output ───────────────────────────────────────────────────────
check_output() {
  # Returns 0 if output looks coherent, 1 if it looks like garbage.
  local first64="$1"
  # Fail if it's all the same character repeated
  if echo "$first64" | python3 -c "
import sys
s = sys.stdin.read().strip()
if len(s) < 8:
    sys.exit(0)  # too short to judge
if len(set(s)) <= 2:
    print('GIBBERISH: very low unique chars', file=sys.stderr)
    sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
    return 0
  else
    return 1
  fi
}

# ── Per-cell bench ────────────────────────────────────────────────────────────
run_cell() {
  local MODEL_PATH="$1"
  local QUANT_SIG="$2"
  local DISK_GB="$3"
  local KV="$4"

  local MODEL_ID
  MODEL_ID=$(basename "$MODEL_PATH")

  log "══ CELL: $MODEL_ID  kv=$KV ══"

  teardown

  # Start rmlx serve
  local SERVER_LOG="$LOG_DIR/${MODEL_ID}_kv${KV}_$(date -u +%Y%m%d-%H%M%S).log"
  (
    cd "$RMLX_DIR"
    "$RMLX_BIN" serve \
      --model "$MODEL_PATH" \
      --port "$PORT" \
      --device gpu \
      --kv-quant "$KV" \
      --max-ctx "$MAX_CTX"
  ) >"$SERVER_LOG" 2>&1 &
  local RMLX_PID=$!
  log "  rmlx pid=$RMLX_PID  log=$SERVER_LOG"

  # Wait for /v1/models (up to 180s for large models)
  local READY=0
  for _ in {1..180}; do
    sleep 1
    if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      READY=1; break
    fi
    # Check if process died
    if ! kill -0 "$RMLX_PID" 2>/dev/null; then
      log "  ERROR: rmlx process died — tail of log:"
      tail -20 "$SERVER_LOG" | tee -a "$SWEEP_LOG" || true
      log "  SKIP: $MODEL_ID kv=$KV — server died at startup"
      return
    fi
  done

  if [[ "$READY" -eq 0 ]]; then
    log "  ERROR: server not ready in 180s — skipping"
    kill "$RMLX_PID" 2>/dev/null || true
    tail -20 "$SERVER_LOG" | tee -a "$SWEEP_LOG" || true
    return
  fi

  log "  server ready"

  # Run 2 requests (r0=cold, r1=warm)
  (
    cd "$CBB_DIR"
    uv run python -m runners.run_one \
      --backend rmlx \
      --backend-version "$VERSION" \
      --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL_ID" \
      --model-path "$MODEL_PATH" \
      --quant "${QUANT_SIG} + kv-${KV}" \
      --device gpu \
      --max-tokens "$MAX_TOKENS" \
      --runs "$RUNS" \
      --backend-pid "$RMLX_PID" \
      --model-disk-gb "$DISK_GB" \
      --prompt-file "$PROMPT_FILE" \
      --prompt-tokens "$PROMPT_TOKENS" 2>&1
  ) | tee -a "$SWEEP_LOG"

  local RC=${PIPESTATUS[0]}
  if [[ "$RC" -ne 0 ]]; then
    log "  WARNING: run_one exited with RC=$RC (may still have written partial records)"
  fi

  log "  DONE cell $MODEL_ID kv=$KV"
  kill "$RMLX_PID" 2>/dev/null || true
  sleep 3
}

# ── Main sweep loop ───────────────────────────────────────────────────────────
CELL_COUNT=0
START_TS=$SECONDS

for entry in "${MODELS[@]}"; do
  IFS='|' read -r MPATH QSIG DGIB <<< "$entry"
  local_id=$(basename "$MPATH")

  # Apply model filter
  if [[ -n "$MODEL_FILTER" ]] && [[ "$local_id" != *"$MODEL_FILTER"* ]]; then
    log "SKIP model $local_id (filter=$MODEL_FILTER)"
    continue
  fi

  for KV in "${KV_MODES[@]}"; do
    # Apply KV filter
    if [[ -n "$KV_FILTER" ]] && [[ "$KV" != "$KV_FILTER" ]]; then
      continue
    fi

    run_cell "$MPATH" "$QSIG" "$DGIB" "$KV"
    CELL_COUNT=$((CELL_COUNT + 1))
  done
done

# Final teardown
teardown

ELAPSED=$(( SECONDS - START_TS ))
log "=== bench-records-sweep DONE  cells=$CELL_COUNT  elapsed=${ELAPSED}s ==="
log "Sweep log: $SWEEP_LOG"
log "CBB summary.csv: $CBB_DIR/metrics/summary.csv"
