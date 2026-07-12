#!/usr/bin/env bash
# fullctx_regression_bench.sh — Step-1 regression check: full-ctx prompts
# Qwen3.6-35B-A3B-8bit × {planar@8K, bf16@16K, k8v8@32K, planar@64K}
# Each cell: kill stale, start server, 1 warmup + 3 measure, record to DB.
# Usage: ./scripts/bench/fullctx_regression_bench.sh [cell_idx]
#   cell_idx 0-3 selects single cell; omit for all 4.

set -euo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"

# Run identity (backend / version / git sha / build profile / hardware tag)
# comes from the measured binary — never hard-coded here.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/identity.sh"
rmlx_export_identity "${RMLX_BIN}"
PORT=62265
WARMUP_RUNS=1
MEASURE_RUNS=3
MAX_TOKENS=30
METRICS_DB="${RMLX_ROOT}/metrics/runs.db"
QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
PROMPTS_DIR="${RMLX_ROOT}/prompts"

GIT_SHA="$(git -C ${RMLX_ROOT} rev-parse --short HEAD 2>/dev/null || echo unknown)"
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
REPORT_LOG="${RMLX_ROOT}/logs/fullctx_regression_$(date -u +%Y%m%d-%H%M%S).log"

mkdir -p "$(dirname "${REPORT_LOG}")"
mkdir -p ${RMLX_ROOT}/metrics/buffer/pending
mkdir -p ${RMLX_ROOT}/metrics/buffer/failed

log() { echo "[fullctx] $*" | tee -a "${REPORT_LOG}" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Cells: MODEL_PATH|KV_QUANT|MAX_CTX|PROMPT_FILE|PROMPT_TOKENS|PRIOR_CHAMPION
CELLS=(
    "${QWEN_PATH}|planar|8192|${PROMPTS_DIR}/longctx_8k.json|8192|97.39"
    "${QWEN_PATH}|bf16|16384|${PROMPTS_DIR}/longctx_16k.json|16381|91.86"
    "${QWEN_PATH}|k8v8|32768|${PROMPTS_DIR}/longctx_32k.json|32764|85.41"
    "${QWEN_PATH}|planar|65536|${PROMPTS_DIR}/longctx_64k.json|65528|71.53"
)

preflight() {
    log "Pre-flight: kill stale processes..."
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm       2>/dev/null || true
    pkill -f paroquant    2>/dev/null || true
    pkill -f omlx         2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
    log "Pre-flight done."
}

wait_health() {
    local url="http://127.0.0.1:${PORT}/health"
    local max_wait=300
    local elapsed=0
    until curl -s --max-time 3 "${url}" | grep -q '"ok"'; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "Server crashed. Tail:"
            tail -20 "${SERVE_LOG}" >&2
            return 1
        fi
        if [[ ${elapsed} -ge ${max_wait} ]]; then
            log "Health timeout. Tail:"
            tail -20 "${SERVE_LOG}" >&2
            kill "${SERVER_PID}" 2>/dev/null || true
            return 1
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    log "Server ready in ${elapsed}s."
}

completion_request() {
    local max_tokens="$1"
    local prompt_file="$2"
    local payload
    payload="$(python3 -c "
import json
with open('${prompt_file}') as f:
    pf = json.load(f)
print(json.dumps({
    'model': '${MODEL_ID}',
    'messages': pf['messages'],
    'max_tokens': ${max_tokens},
    'temperature': 0.0,
    'stream': False,
}))
")"
    local t_start
    t_start="$(python3 -c 'import time; print(int(time.time()*1000))')"
    local response
    response="$(curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "${payload}" 2>/dev/null)"
    local t_end
    t_end="$(python3 -c 'import time; print(int(time.time()*1000))')"
    local elapsed_ms=$(( t_end - t_start ))
    local completion_tokens
    completion_tokens="$(echo "${response}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('usage', {}).get('completion_tokens', 0))
" 2>/dev/null || echo 0)"
    local generated_text
    generated_text="$(echo "${response}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
choices = d.get('choices', [])
print(choices[0].get('message', {}).get('content', '') if choices else '')
" 2>/dev/null || echo '')"
    echo "elapsed_ms=${elapsed_ms} tokens=${completion_tokens} text=${generated_text}"
}

bench_cell() {
    local idx="$1"
    local model_path="$2"
    local kv_quant="$3"
    local max_ctx="$4"
    local prompt_file="$5"
    local prompt_tokens="$6"
    local prior_champion="$7"

    MODEL_ID="$(basename "${model_path}")"
    log "=== Cell ${idx}: kv=${kv_quant} ctx=${max_ctx} prompt=${prompt_tokens}tok prior=${prior_champion} ==="

    preflight

    local run_id
    run_id="$(date -u +%Y%m%d-%H%M%S)-${GIT_SHA}"
    SERVE_LOG="${RMLX_ROOT}/logs/fullctx_cell${idx}_${run_id}.log"

    log "Starting server (kv=${kv_quant} ctx=${max_ctx})..."
    "${RMLX_BIN}" serve \
        --model "${model_path}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        --device gpu \
        --kv-quant "${kv_quant}" \
        --max-ctx "${max_ctx}" \
        > "${SERVE_LOG}" 2>&1 &
    SERVER_PID=$!

    if ! wait_health; then
        log "ERROR: server not ready for cell ${idx}"
        kill "${SERVER_PID}" 2>/dev/null || true
        echo "CELL_RESULT idx=${idx} kv=${kv_quant} ctx=${max_ctx} tps=ERROR"
        return 1
    fi

    # Warmup — use the full-ctx prompt to warm both compile cache AND KV cache
    log "Warmup with full-ctx prompt (${prompt_tokens} tok)..."
    completion_request 10 "${prompt_file}" > /dev/null || true

    # Measure with full-ctx prompt
    declare -a tps_vals=()
    local first_text=""
    for i in $(seq 1 "${MEASURE_RUNS}"); do
        log "  Measure ${i}/${MEASURE_RUNS} (full-ctx ${prompt_tokens} tok)..."
        local result
        result="$(completion_request "${MAX_TOKENS}" "${prompt_file}")"
        local elapsed_ms n_tokens
        elapsed_ms="$(echo "${result}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
        n_tokens="$(echo "${result}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
        local tps
        tps="$(python3 -c "
ms=${elapsed_ms}; n=${n_tokens}
print(f'{n/(ms/1000):.2f}' if n > 0 and ms > 0 else '0.0')
")"
        tps_vals+=("${tps}")
        if [[ ${i} -eq 1 ]]; then
            first_text="$(echo "${result}" | sed 's/elapsed_ms=[0-9]* tokens=[0-9]* text=//')"
        fi
        log "    tps=${tps} (elapsed=${elapsed_ms}ms tokens=${n_tokens})"
    done

    local tps_array="${tps_vals[*]}"
    read -r tps_mean tps_stddev <<< "$(python3 -c "
import math
vals = [float(x) for x in '${tps_array}'.split()]
mean = sum(vals)/len(vals)
stddev = math.sqrt(sum((v-mean)**2 for v in vals)/(len(vals)-1)) if len(vals) > 1 else 0.0
print(f'{mean:.2f} {stddev:.2f}')
")"

    # Delta vs prior champion
    local delta_pct
    delta_pct="$(python3 -c "
prior=${prior_champion}; curr=float('${tps_mean}')
delta=(curr-prior)/prior*100
print(f'{delta:+.1f}')
")"

    log "Cell ${idx}: tps_mean=${tps_mean} stddev=${tps_stddev} prior=${prior_champion} delta=${delta_pct}%"
    log "First output: ${first_text:0:80}"

    # §8.5 buffer record
    local model_dir; model_dir="$(basename "${model_path}")"
    local ns mdl
    if [[ "${model_dir}" == *"__"* ]]; then
        ns="${model_dir%%__*}"
        mdl="${model_dir#*__}"
    else
        ns="local"
        mdl="${model_dir}"
    fi

    local prompt_name; prompt_name="$(basename "${prompt_file}" .json)"

    local buf_ts; buf_ts="$(date -u +%Y%m%d%H%M%S)"
    local buf_uuid; buf_uuid="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
    local record_path="${RMLX_ROOT}/metrics/buffer/pending/${buf_ts}-${buf_uuid}.json"

    python3 - <<PYEOF > "${record_path}" 2>/dev/null || { log "WARN: failed to build record"; rm -f "${record_path}"; }
import json, os
with open('${prompt_file}') as f:
    pf = json.load(f)
first_64 = '${first_text}'[:64] if '${first_text}' else ''
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    'model_namespace': '${ns}',
    'model':           '${mdl}',
    'weight_quant':    'q8_0',
    'kv_quant':        '${kv_quant}',
    'ctx_max':         int('${max_ctx}'),
    'prompt': {
        'name':          '${prompt_name}',
        'body':          pf['messages'],
        'tokens_approx': int('${prompt_tokens}'),
    },
    'ts_utc':          '${TS_UTC}',
    'prompt_tokens':   int('${prompt_tokens}'),
    'max_tokens':      ${MAX_TOKENS},
    'temperature':     0.0,
    'seed':            0,
    'n_warmups':       ${WARMUP_RUNS},
    'n_measure':       ${MEASURE_RUNS},
    'output_first_64': first_64,
    'notes':           'fullctx-regression-bench',
    'description':     'Step1: full-ctx prompt regression check vs prior champion',
    'metrics': [
        {'name': 'decode_tps_warm', 'value': float('${tps_mean}'), 'stddev': float('${tps_stddev}')},
    ],
}
print(json.dumps(rec))
PYEOF

    if [[ -f "${record_path}" ]]; then
        if "${RMLX_BIN}" metrics --db "${METRICS_DB}" record --file "${record_path}" 2>>"${REPORT_LOG}"; then
            rm -f "${record_path}"
            log "§8.5 record ingested."
        else
            mv "${record_path}" ${RMLX_ROOT}/metrics/buffer/failed/ 2>/dev/null || true
            log "WARN: recorder failed. Results still valid."
        fi
    fi

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    rm -f "/tmp/rmlx.${PORT}.claim"

    echo "CELL_RESULT idx=${idx} kv=${kv_quant} ctx=${max_ctx} tps=${tps_mean} stddev=${tps_stddev} prior=${prior_champion} delta=${delta_pct}%"
}

log "Full-ctx regression bench starting. Git SHA=${GIT_SHA}"
log "Total cells: ${#CELLS[@]}"

START_IDX="${1:-0}"
END_IDX="${2:-$((${#CELLS[@]} - 1))}"

for i in $(seq "${START_IDX}" "${END_IDX}"); do
    cell="${CELLS[$i]}"
    IFS='|' read -r model_path kv_quant max_ctx prompt_file prompt_tokens prior_champion <<< "${cell}"
    bench_cell "${i}" "${model_path}" "${kv_quant}" "${max_ctx}" "${prompt_file}" "${prompt_tokens}" "${prior_champion}"
done

log "All cells done."
