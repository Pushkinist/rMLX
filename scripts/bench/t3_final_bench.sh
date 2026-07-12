#!/usr/bin/env bash
# t3_final_bench.sh — final regression bench
# Models: Qwen3.6-35B-A3B-8bit (k8v4, k8v8, planar) x (8K, 16K, 32K, 64K)   = 12 cells
#         gemma-4-26b-a4b-it-mxfp8 (planar, k8v8)    x (8K, 16K, 32K, 64K)   = 8 cells
# Total: 20 cells.
#
# Usage: ./scripts/bench/t3_final_bench.sh [cell_index]
#   If cell_index given, runs only that cell (0-based, 0–19).
#   If omitted, runs all cells in sequence.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"

# Run identity (backend / version / git sha / build profile / hardware tag)
# comes from the measured binary — never hard-coded here.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/identity.sh"
rmlx_export_identity "${RMLX_BIN}"
PORT=62265
WARMUP_RUNS=1
MEASURE_RUNS=3
MAX_TOKENS=30
CELL_TIMEOUT=900        # seconds; kill server + mark TIMEOUT if exceeded
PROMPT_FILE="${RMLX_ROOT}/prompts/longctx_4k.json"
METRICS_OUT="${RMLX_ROOT}/metrics/perf-iter/t3_final.jsonl"
REPORT_LOG="${RMLX_ROOT}/logs/t3_final_bench_$(date -u +%Y%m%d-%H%M%S).log"
QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
GEMMA_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"

GIT_SHA="$(git -C ${RMLX_ROOT} rev-parse --short HEAD 2>/dev/null || echo unknown)"
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

mkdir -p "$(dirname "${REPORT_LOG}")"
mkdir -p "$(dirname "${METRICS_OUT}")"
mkdir -p ${RMLX_ROOT}/metrics/buffer/pending
mkdir -p ${RMLX_ROOT}/metrics/buffer/failed

log() { echo "[t3_final] $*" | tee -a "${REPORT_LOG}" >&2; }
die() { log "ERROR: $*"; exit 1; }

CELLS=(
    "${QWEN_PATH}|k8v4|8192|4096"
    "${QWEN_PATH}|k8v4|16384|4096"
    "${QWEN_PATH}|k8v4|32768|4096"
    "${QWEN_PATH}|k8v4|65536|4096"
    "${QWEN_PATH}|k8v8|8192|4096"
    "${QWEN_PATH}|k8v8|16384|4096"
    "${QWEN_PATH}|k8v8|32768|4096"
    "${QWEN_PATH}|k8v8|65536|4096"
    "${QWEN_PATH}|planar|8192|4096"
    "${QWEN_PATH}|planar|16384|4096"
    "${QWEN_PATH}|planar|32768|4096"
    "${QWEN_PATH}|planar|65536|4096"
    "${GEMMA_PATH}|planar|8192|4096"
    "${GEMMA_PATH}|planar|16384|4096"
    "${GEMMA_PATH}|planar|32768|4096"
    "${GEMMA_PATH}|planar|65536|4096"
    "${GEMMA_PATH}|k8v8|8192|4096"
    "${GEMMA_PATH}|k8v8|16384|4096"
    "${GEMMA_PATH}|k8v8|32768|4096"
    "${GEMMA_PATH}|k8v8|65536|4096"
)

preflight() {
    log "Pre-flight: kill stale processes..."
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    pkill -f paroquant   2>/dev/null || true
    pkill -f omlx        2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
    log "Pre-flight done."
}

wait_health() {
    local url="http://127.0.0.1:${PORT}/health"
    local max_wait=300
    local elapsed=0
    until curl -s --max-time 2 "${url}" | grep -q '"ok"'; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "Server crashed. Tail of log:"
            tail -20 "${SERVE_LOG}" >&2
            return 1
        fi
        if [[ ${elapsed} -ge ${max_wait} ]]; then
            log "Health timeout ${max_wait}s."
            kill "${SERVER_PID}" 2>/dev/null || true
            return 1
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    log "Server ready in ${elapsed}s."
    return 0
}

_CR_PAYLOAD_TMP="/tmp/rmlx_t3_payload_$$.json"
_CR_RESP_TMP="/tmp/rmlx_t3_resp_$$.json"

completion_request() {
    local max_tokens="$1"
    # Build payload into tmp file to avoid shell-quoting issues with large messages
    python3 - "${PROMPT_FILE}" "${MODEL_ID}" "${max_tokens}" "${_CR_PAYLOAD_TMP}" <<'PYEOF'
import json, sys
prompt_file, model_id, max_tokens, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(prompt_file) as f:
    pf = json.load(f)
payload = {
    'model': model_id,
    'messages': pf['messages'],
    'max_tokens': max_tokens,
    'temperature': 0.0,
    'stream': False,
}
with open(out_path, 'w') as f:
    json.dump(payload, f)
PYEOF
    local t_start t_end
    t_start="$(python3 -c 'import time; print(int(time.time()*1000))')"
    curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "@${_CR_PAYLOAD_TMP}" \
        -o "${_CR_RESP_TMP}" 2>/dev/null
    t_end="$(python3 -c 'import time; print(int(time.time()*1000))')"
    local elapsed_ms=$(( t_end - t_start ))
    local completion_tokens generated_text
    completion_tokens="$(python3 -c "
import json
d = json.load(open('${_CR_RESP_TMP}'))
print(d.get('usage', {}).get('completion_tokens', 0))
" 2>/dev/null || echo 0)"
    generated_text="$(python3 -c "
import json
d = json.load(open('${_CR_RESP_TMP}'))
choices = d.get('choices', [])
if choices:
    print(choices[0].get('message', {}).get('content', ''))
" 2>/dev/null || echo '')"
    echo "elapsed_ms=${elapsed_ms} tokens=${completion_tokens} text=${generated_text}"
}

bench_cell() {
    local idx="$1"
    local model_path="$2"
    local kv_quant="$3"
    local max_ctx="$4"
    local prompt_tokens="$5"

    MODEL_ID="$(basename "${model_path}")"
    log "=== Cell ${idx}: model=${MODEL_ID} kv=${kv_quant} ctx=${max_ctx} ==="

    local cell_start; cell_start="$(date +%s)"

    check_cell_timeout() {
        local now; now="$(date +%s)"
        local elapsed=$(( now - cell_start ))
        if [[ ${elapsed} -ge ${CELL_TIMEOUT} ]]; then
            log "TIMEOUT: cell ${idx} exceeded ${CELL_TIMEOUT}s wall-clock. Killing server."
            kill "${SERVER_PID}" 2>/dev/null || true
            wait "${SERVER_PID}" 2>/dev/null || true
            echo "CELL_RESULT idx=${idx} model=${MODEL_ID} kv=${kv_quant} ctx=${max_ctx} tps=TIMEOUT"
            return 1
        fi
        return 0
    }

    preflight

    local run_id
    run_id="$(date -u +%Y%m%d-%H%M%S)-${GIT_SHA}"
    SERVE_LOG="${RMLX_ROOT}/logs/t3_final_cell${idx}_${run_id}.log"

    log "Starting server..."
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
        echo "CELL_RESULT idx=${idx} model=${MODEL_ID} kv=${kv_quant} ctx=${max_ctx} tps=ERROR"
        return 1
    fi

    # Warmup
    log "Warmup..."
    check_cell_timeout || return 0
    completion_request 10 > /dev/null || true

    # Measure
    declare -a tps_vals=()
    local first_text=""
    for i in $(seq 1 "${MEASURE_RUNS}"); do
        check_cell_timeout || return 0
        log "  Measure ${i}/${MEASURE_RUNS}..."
        local result
        result="$(completion_request "${MAX_TOKENS}")" || result="elapsed_ms=0 tokens=0 text="
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

    local first_32_words
    first_32_words="$(echo "${first_text}" | python3 -c "
import sys, json
words = sys.stdin.read().split()[:32]
print(json.dumps(words))
")"

    log "Cell ${idx} result: tps_mean=${tps_mean} stddev=${tps_stddev}"
    log "First output: ${first_text:0:80}"

    echo "CELL_RESULT idx=${idx} model=${MODEL_ID} kv=${kv_quant} ctx=${max_ctx} tps=${tps_mean} stddev=${tps_stddev}"

    # Write legacy JSONL
    python3 -c "
import json, os
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    'run_id': '${run_id}',
    'ts_utc': '${TS_UTC}',
    'model_path': '${model_path}',
    'kv_quant': '${kv_quant}',
    'max_ctx': int('${max_ctx}'),
    'decode_tps_mean': float('${tps_mean}'),
    'decode_tps_stddev': float('${tps_stddev}'),
    'git_sha': '${GIT_SHA}',
    'notes': 'final-bench',
}
print(json.dumps(rec))
" >> "${METRICS_OUT}"

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

    local weight_quant
    local model_dir_lower
    model_dir_lower="$(echo "${model_dir}" | tr '[:upper:]' '[:lower:]')"
    case "${model_dir_lower}" in
        *mxfp8*) weight_quant="mxfp8" ;;
        *8bit*)   weight_quant="q8_0" ;;
        *4bit*)   weight_quant="q4_k_m" ;;
        *2bit*)   weight_quant="2bit" ;;
        *)        weight_quant="unknown" ;;
    esac

    local buf_ts; buf_ts="$(date -u +%Y%m%d%H%M%S)"
    local buf_uuid; buf_uuid="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
    local record_path="${RMLX_ROOT}/metrics/buffer/pending/${buf_ts}-${buf_uuid}.json"

    python3 -c "
import json, os
with open('${PROMPT_FILE}') as f:
    pf = json.load(f)
prompt_body = pf['messages']
words = ${first_32_words}
output_first_64 = ' '.join(str(w) for w in words)[:64]
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    'git_sha':         '${GIT_SHA}',
    'model_namespace': '${ns}',
    'model':           '${mdl}',
    'weight_quant':    '${weight_quant}',
    'kv_quant':        '${kv_quant}',
    'ctx_max':         int('${max_ctx}'),
    'prompt': {
        'name':          'longctx_4k',
        'body':          prompt_body,
        'tokens_approx': int('${prompt_tokens}'),
    },
    'ts_utc':          '${TS_UTC}',
    'prompt_tokens':   int('${prompt_tokens}'),
    'max_tokens':      int('${MAX_TOKENS}'),
    'temperature':     0.0,
    'seed':            0,
    'n_warmups':       int('${WARMUP_RUNS}'),
    'n_measure':       int('${MEASURE_RUNS}'),
    'output_first_64': output_first_64,
    'notes':           'T3-final-bench',
    'description':     None,
    'metrics': [
        {'name': 'decode_tps_warm', 'value': float('${tps_mean}'), 'stddev': float('${tps_stddev}')},
    ],
}
with open('${record_path}', 'w') as f:
    json.dump(rec, f)
print(f'buffer: ${record_path}')
" 2>/dev/null || { log "WARN: failed to build §8.5 record"; rm -f "${record_path}"; }

    if [[ -f "${record_path}" ]]; then
        if "${RMLX_BIN}" metrics record --file "${record_path}" 2>/dev/null; then
            rm -f "${record_path}"
            log "§8.5 record ingested."
        else
            mv "${record_path}" ${RMLX_ROOT}/metrics/buffer/failed/
            log "WARN: recorder failed. Results still valid."
        fi
    fi

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}

# ── Main ──────────────────────────────────────────────────────────────────────

if [[ ${#} -eq 1 ]]; then
    # Single cell mode
    idx="${1}"
    cell="${CELLS[${idx}]}"
    IFS='|' read -r model_path kv_quant max_ctx prompt_tokens <<< "${cell}"
    bench_cell "${idx}" "${model_path}" "${kv_quant}" "${max_ctx}" "${prompt_tokens}" || true
else
    # All cells, with 60s cooldown between model changes
    prev_model=""
    for idx in "${!CELLS[@]}"; do
        cell="${CELLS[${idx}]}"
        IFS='|' read -r model_path kv_quant max_ctx prompt_tokens <<< "${cell}"
        curr_model="$(basename "${model_path}")"
        if [[ "${curr_model}" != "${prev_model}" && -n "${prev_model}" ]]; then
            log "--- 60s inter-model cooldown ---"
            sleep 60
        fi
        prev_model="${curr_model}"
        bench_cell "${idx}" "${model_path}" "${kv_quant}" "${max_ctx}" "${prompt_tokens}" || true
    done
fi

log "All cells complete."
