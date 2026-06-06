#!/usr/bin/env bash
# env_var_test.sh — Test env-var toggles on Qwen35B planar 8K full-ctx
# Tests: SPARSE_V_KERNEL OFF, SPARSE_V_THRESHOLD OFF, both OFF, SPARSE_V_THRESHOLD high ON
# Each: 1 warmup + 3 measure. Compare to 84.34 baseline (both ON).
set -euo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
QWEN=${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit
MODEL_ID=mlx-community__Qwen3.6-35B-A3B-8bit
PROMPT=${RMLX_ROOT}/prompts/longctx_8k.json

GIT_SHA="$(git -C ${RMLX_ROOT} rev-parse --short HEAD 2>/dev/null || echo unknown)"
LOG=${RMLX_ROOT}/logs/env_var_test_$(date -u +%Y%m%d-%H%M%S).log

mkdir -p "$(dirname "$LOG")"

log() { echo "[env_test] $*" | tee -a "$LOG" >&2; }

preflight() {
    log "Preflight: kill stale..."
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm 2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
    log "Preflight done."
}

wait_health() {
    local elapsed=0
    until curl -s --max-time 3 "http://127.0.0.1:${PORT}/health" | grep -q '"ok"'; do
        kill -0 "$SERVER_PID" 2>/dev/null || { log "Server crashed"; return 1; }
        sleep 2; elapsed=$((elapsed + 2))
        [[ $elapsed -gt 180 ]] && log "Health timeout" && kill "$SERVER_PID" 2>/dev/null && return 1
    done
    log "Ready in ${elapsed}s"
}

make_payload() {
    local max_tok="$1"
    python3 -c "
import json
with open('$PROMPT') as f:
    pf = json.load(f)
print(json.dumps({'model': '$MODEL_ID', 'messages': pf['messages'], 'max_tokens': $max_tok, 'temperature': 0.0, 'stream': False}))
"
}

run_test() {
    local label="$1"
    shift
    local env_vars=("$@")

    log ""
    log "=== TEST: $label ==="
    preflight

    local serve_log
    serve_log="${RMLX_ROOT}/logs/env_test_${label// /_}_$(date -u +%Y%m%d-%H%M%S).log"

    # Start server with env vars
    env "${env_vars[@]}" "$RMLX_BIN" serve \
        --model "$QWEN" --port $PORT --host 127.0.0.1 --device gpu \
        --kv-quant planar --max-ctx 8192 \
        > "$serve_log" 2>&1 &
    SERVER_PID=$!

    wait_health || { echo "RESULT label=$label tps=ERROR"; return 1; }

    # Warmup with full-ctx prompt
    local wu_payload
    wu_payload="$(make_payload 10)"
    log "Warmup..."
    curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$wu_payload" > /dev/null
    log "Warmup done"

    local m_payload
    m_payload="$(make_payload 30)"
    declare -a tps_vals=()
    for run in 1 2 3; do
        local t0 t1 elapsed_ms resp toks tps
        t0="$(python3 -c 'import time; print(int(time.time()*1000))')"
        resp="$(curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "$m_payload")"
        t1="$(python3 -c 'import time; print(int(time.time()*1000))')"
        elapsed_ms=$((t1 - t0))
        toks="$(echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)"
        tps="$(python3 -c "ms=$elapsed_ms; n=$toks; print(f'{n/(ms/1000):.2f}' if n>0 and ms>0 else '0.0')")"
        tps_vals+=("$tps")
        log "  run $run: tps=$tps (${elapsed_ms}ms ${toks}tok)"
    done

    local tps_array="${tps_vals[*]}"
    read -r tps_mean tps_stddev <<< "$(python3 -c "
import math
vals = [float(x) for x in '${tps_array}'.split()]
mean = sum(vals)/len(vals)
stddev = math.sqrt(sum((v-mean)**2 for v in vals)/(len(vals)-1)) if len(vals) > 1 else 0.0
print(f'{mean:.2f} {stddev:.2f}')
")"

    local delta
    delta="$(python3 -c "
baseline=84.34; curr=float('${tps_mean}')
print(f'{(curr-baseline)/baseline*100:+.1f}%')
")"

    log "RESULT $label: tps=${tps_mean} stddev=${tps_stddev} delta_vs_baseline=${delta}"
    echo "RESULT label='${label}' tps=${tps_mean} stddev=${tps_stddev} delta=${delta}"

    kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null
    rm -f "/tmp/rmlx.${PORT}.claim"
}

log "Env-var test bench. Git SHA=${GIT_SHA}"
log "Baseline (SPARSE_V_KERNEL+SPARSE_V_THRESHOLD ON, default): 84.34 TPS"
log "Prior champion (full-ctx baseline): 97.39 TPS"

# Test 1: SPARSE_V_KERNEL OFF
run_test "SPARSE_V_KERNEL_OFF" "RMLX_SPARSE_V_KERNEL=0"

# Test 2: SPARSE_V_THRESHOLD OFF
run_test "SPARSE_V_THRESHOLD_OFF" "RMLX_SPARSE_V_THRESHOLD=0"

# Test 3: BOTH OFF
run_test "BOTH_OFF" "RMLX_SPARSE_V_KERNEL=0" "RMLX_SPARSE_V_THRESHOLD=0"

log "All tests done."
