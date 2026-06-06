#!/usr/bin/env bash
# P2.A — TurboFlash head_dim=256 bench.
# Qwen3.6-35B-A3B-8bit k8v4 × {16K, 32K, 64K, 128K} × {OFF=baseline, ON=TurboFlash}.
# Measures decode TPS via OpenAI-style /v1/chat/completions, full-ctx prompts.
# Sequential — kill stale before every cell, 60s cooldown between cells.
#
# Usage:  ./scripts/bench/p2a_turbo_flash_bench.sh
# Output: decode TPS to stdout + summary table at end + appends to runs.db buffer.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
WARMUP_RUNS=1
MEASURE_RUNS=3
MAX_TOKENS=30
MODEL_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
MODEL_NAME="$(basename "${MODEL_PATH}")"

GIT_SHA="$(git -C ${RMLX_ROOT} rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${RMLX_ROOT}/logs/p2a_turbo_flash_${RUN_TS}"
mkdir -p "${LOG_DIR}"

log() { echo "[p2a-bench] $*" >&2; }

CELLS=(
    "16384|16k"
    "32768|32k"
    "65536|64k"
    "131072|128k"
)

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    pkill -f paroquant   2>/dev/null || true
    pkill -f omlx        2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local url="http://127.0.0.1:${PORT}/health"
    local max=300 e=0
    until curl -s --max-time 2 "${url}" | grep -q '"ok"'; do
        sleep 3; e=$((e+3))
        [[ ${e} -ge ${max} ]] && { log "ERROR: health timeout"; return 1; }
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "ERROR: server died (see ${SERVE_LOG})"; return 1
        fi
    done
    log "ready in ${e}s"
}

run_one_request() {
    local prompt_file="$1"
    local payload="/tmp/p2a_payload_$$.json"
    local resp="/tmp/p2a_resp_$$.json"
    python3 - "${prompt_file}" "${MODEL_NAME}" "${MAX_TOKENS}" "${payload}" <<'PYEOF'
import json, sys
prompt_file, model, mt, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
pf = json.load(open(prompt_file))
p = {'model': model, 'messages': pf['messages'], 'max_tokens': mt,
     'temperature': 0.0, 'stream': False}
json.dump(p, open(out, 'w'))
PYEOF
    local start_ms=$(python3 -c "import time; print(int(time.time()*1000))")
    curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "@${payload}" -o "${resp}" 2>/dev/null
    local end_ms=$(python3 -c "import time; print(int(time.time()*1000))")
    # Get actual decode tokens from server response usage (if available).
    python3 - "${resp}" "${start_ms}" "${end_ms}" <<'PYEOF'
import json, sys
resp_path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    d = json.load(open(resp_path))
    usage = d.get('usage', {})
    completion = usage.get('completion_tokens', 0)
    if completion == 0:
        # Fallback: estimate from text length (conservative).
        text = d.get('choices', [{}])[0].get('message', {}).get('content', '')
        completion = max(1, len(text) // 4)
    elapsed_s = (end - start) / 1000.0
    tps = completion / elapsed_s if elapsed_s > 0 else 0
    print(f"{tps:.2f},{completion},{elapsed_s:.2f}")
except Exception as e:
    print(f"0,0,0,error:{e}")
PYEOF
    rm -f "${payload}" "${resp}"
}

run_cell() {
    local ctx="$1"
    local label="$2"
    local flag="$3"   # 0 or 1
    local prompt="${RMLX_ROOT}/prompts/longctx_${label}.json"
    local mode_label
    if [[ "${flag}" == "0" ]]; then
        mode_label="OFF"
    else
        mode_label="ON"
    fi

    log "=== Cell ${label} TurboFlash=${mode_label} ==="
    preflight

    SERVE_LOG="${LOG_DIR}/serve_${label}_${mode_label}.log"
    log "Starting server (kv-quant=k8v4, max-ctx=${ctx})..."
    RMLX_TURBO_FLASH="${flag}" "${RMLX_BIN}" serve \
        --model "${MODEL_PATH}" \
        --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant k8v4 --max-ctx "${ctx}" \
        > "${SERVE_LOG}" 2>&1 &
    SERVER_PID=$!
    if ! wait_health; then
        log "ERROR: ${label} ${mode_label} health failed"
        kill "${SERVER_PID}" 2>/dev/null
        wait "${SERVER_PID}" 2>/dev/null || true
        echo "0,0,0"
        return
    fi

    # Warmup
    for i in $(seq 1 "${WARMUP_RUNS}"); do
        log "  warmup ${i}/${WARMUP_RUNS}..."
        run_one_request "${prompt}" >/dev/null
    done

    # Measure
    local sum_tps=0 n=0
    declare -a tps_arr=()
    for i in $(seq 1 "${MEASURE_RUNS}"); do
        local result
        result="$(run_one_request "${prompt}")"
        local tps
        tps="$(echo "${result}" | cut -d, -f1)"
        local toks
        toks="$(echo "${result}" | cut -d, -f2)"
        local secs
        secs="$(echo "${result}" | cut -d, -f3)"
        log "  measure ${i}/${MEASURE_RUNS}: tps=${tps} (${toks} tok in ${secs}s)"
        tps_arr+=("${tps}")
    done

    # Median
    local median
    median="$(python3 -c "
v = sorted([${tps_arr[0]}, ${tps_arr[1]}, ${tps_arr[2]}])
print(f'{v[1]:.2f}')
")"
    log "  median TPS: ${median}"
    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "${median}"
}

# === Run all cells ===
declare -a results=()
for cell in "${CELLS[@]}"; do
    ctx="${cell%%|*}"
    label="${cell##*|}"

    tps_off="$(run_cell "${ctx}" "${label}" 0)"
    sleep 60
    tps_on="$(run_cell "${ctx}" "${label}" 1)"
    delta="$(python3 -c "
off=${tps_off}; on=${tps_on}
if off > 0:
    print(f'{(on - off) / off * 100:+.1f}%')
else:
    print('n/a')
")"
    results+=("${label}|${tps_off}|${tps_on}|${delta}")
    sleep 60
done

# === Summary ===
echo ""
echo "================================================================"
echo "TurboFlash head_dim=256 bench — Qwen3.6-35B-A3B-8bit k8v4"
echo "git_sha=${GIT_SHA}, ts=${RUN_TS}"
echo "================================================================"
printf "%-8s | %-12s | %-12s | %-8s\n" "ctx" "OFF TPS" "ON TPS" "delta"
echo "---------+--------------+--------------+---------"
for r in "${results[@]}"; do
    label="$(echo "${r}" | cut -d'|' -f1)"
    off="$(echo "${r}" | cut -d'|' -f2)"
    on="$(echo "${r}" | cut -d'|' -f3)"
    delta="$(echo "${r}" | cut -d'|' -f4)"
    printf "%-8s | %-12s | %-12s | %-8s\n" "${label}" "${off}" "${on}" "${delta}"
done
echo "================================================================"
