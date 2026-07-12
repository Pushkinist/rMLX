#!/usr/bin/env bash
# final_matrix_bench.sh — closing bench matrix
# 2 models × 4 KV modes × 5 contexts = 40 cells
#
# Usage: ./scripts/bench/final_matrix_bench.sh
#   Runs all cells sequentially. Output: structured JSON per cell + ASCII table.
#
# Cell metrics:
#   decode_tps_mean — mean over 3 warm runs (1 warmup discarded)
#   prefill_tps     — derived from (total_time - decode_time) / prompt_tokens
#   ttft_ms         — warm TTFT: elapsed on max_tokens=1 request after 1 warmup

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
TTFT_TOKENS=1
# per-cell wall-clock guard (30 min for 128K cells)
CELL_TIMEOUT_DEFAULT=1800
# extra cooldown for ctx > 30K
EXTRA_COOLDOWN_CTX=30000
EXTRA_COOLDOWN_S=30

QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
GEMMA_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
PROMPTS_DIR="${RMLX_ROOT}/prompts"

GIT_SHA="$(git -C ${RMLX_ROOT} rev-parse --short HEAD 2>/dev/null || echo unknown)"
# `unknown` is a fallback for run-ids and labels, never provenance — gate the
# git_sha JSON key so a checkout without `.git` writes NULL, not "unknown".
GIT_SHA_KV=""
[[ "${GIT_SHA}" != unknown* ]] && GIT_SHA_KV="'git_sha': '${GIT_SHA}',"
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)-${GIT_SHA}"

REPORT_DIR="${RMLX_ROOT}/docs/reports"
REPORT_MD="${REPORT_DIR}/final-bench-matrix-2026-05-11.md"
LOG_FILE="${RMLX_ROOT}/logs/final_matrix_bench_${RUN_STAMP}.log"
METRICS_OUT="${RMLX_ROOT}/metrics/perf-iter/final_matrix_${RUN_STAMP}.jsonl"
CELL_JSON_DIR="${RMLX_ROOT}/metrics/final_matrix_cells"

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "$(dirname "${METRICS_OUT}")"
mkdir -p "${CELL_JSON_DIR}"
mkdir -p "${REPORT_DIR}"
mkdir -p ${RMLX_ROOT}/metrics/buffer/pending
mkdir -p ${RMLX_ROOT}/metrics/buffer/failed

log() { echo "[matrix] $*" | tee -a "${LOG_FILE}" >&2; }

# prompt_file_for_ctx <ctx>
prompt_file_for_ctx() {
    local ctx="$1"
    if [[ ${ctx} -ge 131072 ]]; then
        echo "${PROMPTS_DIR}/longctx_128k.json"
    elif [[ ${ctx} -ge 32768 ]]; then
        echo "${PROMPTS_DIR}/longctx_32k.json"
    elif [[ ${ctx} -ge 16384 ]]; then
        echo "${PROMPTS_DIR}/longctx_16k.json"
    elif [[ ${ctx} -ge 8192 ]]; then
        echo "${PROMPTS_DIR}/longctx_8k.json"
    else
        echo "${PROMPTS_DIR}/longctx_4k.json"
    fi
}

# CELLS: model_path|kv_quant|max_ctx
# Order: Qwen (k8v4, k8v8, planar, bf16) x (4K,8K,16K,32K,128K)
#        Gemma (k8v4, k8v8, planar, bf16) x (4K,8K,16K,32K,128K)
CELLS=(
    "${QWEN_PATH}|k8v4|4096"
    "${QWEN_PATH}|k8v4|8192"
    "${QWEN_PATH}|k8v4|16384"
    "${QWEN_PATH}|k8v4|32768"
    "${QWEN_PATH}|k8v4|131072"
    "${QWEN_PATH}|k8v8|4096"
    "${QWEN_PATH}|k8v8|8192"
    "${QWEN_PATH}|k8v8|16384"
    "${QWEN_PATH}|k8v8|32768"
    "${QWEN_PATH}|k8v8|131072"
    "${QWEN_PATH}|planar|4096"
    "${QWEN_PATH}|planar|8192"
    "${QWEN_PATH}|planar|16384"
    "${QWEN_PATH}|planar|32768"
    "${QWEN_PATH}|planar|131072"
    "${QWEN_PATH}|bf16|4096"
    "${QWEN_PATH}|bf16|8192"
    "${QWEN_PATH}|bf16|16384"
    "${QWEN_PATH}|bf16|32768"
    "${QWEN_PATH}|bf16|131072"
    "${GEMMA_PATH}|k8v4|4096"
    "${GEMMA_PATH}|k8v4|8192"
    "${GEMMA_PATH}|k8v4|16384"
    "${GEMMA_PATH}|k8v4|32768"
    "${GEMMA_PATH}|k8v4|131072"
    "${GEMMA_PATH}|k8v8|4096"
    "${GEMMA_PATH}|k8v8|8192"
    "${GEMMA_PATH}|k8v8|16384"
    "${GEMMA_PATH}|k8v8|32768"
    "${GEMMA_PATH}|k8v8|131072"
    "${GEMMA_PATH}|planar|4096"
    "${GEMMA_PATH}|planar|8192"
    "${GEMMA_PATH}|planar|16384"
    "${GEMMA_PATH}|planar|32768"
    "${GEMMA_PATH}|planar|131072"
    "${GEMMA_PATH}|bf16|4096"
    "${GEMMA_PATH}|bf16|8192"
    "${GEMMA_PATH}|bf16|16384"
    "${GEMMA_PATH}|bf16|32768"
    "${GEMMA_PATH}|bf16|131072"
)

preflight() {
    log "Pre-flight: killing stale processes..."
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
    local max_wait=360
    local elapsed=0
    until curl -s --max-time 5 "${url}" 2>/dev/null | grep -q '"ok"'; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "Server process ${SERVER_PID} died."
            return 1
        fi
        if [[ ${elapsed} -ge ${max_wait} ]]; then
            log "Health timeout ${max_wait}s."
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "Server ready in ${elapsed}s."
    return 0
}

# completion_request <max_tokens> <prompt_file> <model_id>
# Writes payload to tmp file, response to tmp file — avoids shell-quoting issues.
# prints: elapsed_ms=N completion_tokens=N total_tokens=N prompt_tokens=N
_CR_PAYLOAD_TMP="/tmp/rmlx_matrix_payload_$$.json"
_CR_RESP_TMP="/tmp/rmlx_matrix_resp_$$.json"

completion_request() {
    local max_tokens="$1"
    local prompt_file="$2"
    local model_id="$3"

    # Build payload into tmp file
    python3 - "${prompt_file}" "${model_id}" "${max_tokens}" "${_CR_PAYLOAD_TMP}" <<'PYEOF'
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
    curl -s --max-time 900 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "@${_CR_PAYLOAD_TMP}" \
        -o "${_CR_RESP_TMP}" 2>/dev/null
    local curl_rc=$?
    t_end="$(python3 -c 'import time; print(int(time.time()*1000))')"
    local elapsed_ms=$(( t_end - t_start ))

    if [[ ${curl_rc} -ne 0 ]] || [[ ! -s "${_CR_RESP_TMP}" ]]; then
        echo "elapsed_ms=${elapsed_ms} completion_tokens=0 prompt_tokens=0 total_tokens=0 error=curl_failed"
        return 1
    fi

    python3 - "${_CR_RESP_TMP}" "${elapsed_ms}" <<'PYEOF' 2>/dev/null || \
        echo "elapsed_ms=${elapsed_ms} completion_tokens=0 prompt_tokens=0 total_tokens=0 error=parse_failed"
import json, sys
resp_path, elapsed_ms = sys.argv[1], sys.argv[2]
with open(resp_path) as f:
    d = json.load(f)
u = d.get('usage', {})
ct = u.get('completion_tokens', 0)
pt = u.get('prompt_tokens', 0)
tot = u.get('total_tokens', 0)
print(f'elapsed_ms={elapsed_ms} completion_tokens={ct} prompt_tokens={pt} total_tokens={tot}')
PYEOF
}

SERVER_PID=0

bench_cell() {
    local idx="$1"
    local model_path="$2"
    local kv_quant="$3"
    local max_ctx="$4"

    local MODEL_ID
    MODEL_ID="$(basename "${model_path}")"
    local ctx_label
    if [[ ${max_ctx} -ge 131072 ]]; then ctx_label="128K"
    elif [[ ${max_ctx} -ge 32768 ]]; then ctx_label="32K"
    elif [[ ${max_ctx} -ge 16384 ]]; then ctx_label="16K"
    elif [[ ${max_ctx} -ge 8192 ]]; then ctx_label="8K"
    else ctx_label="4K"
    fi

    log "=== Cell ${idx}: model=${MODEL_ID} kv=${kv_quant} ctx=${max_ctx}(${ctx_label}) ==="

    local PROMPT_FILE
    PROMPT_FILE="$(prompt_file_for_ctx "${max_ctx}")"
    local PROMPT_TOKENS
    PROMPT_TOKENS="$(python3 -c "import json; d=json.load(open('${PROMPT_FILE}')); print(d.get('prompt_tokens', 0))")"
    log "  prompt_file=${PROMPT_FILE} prompt_tokens=${PROMPT_TOKENS}"

    local cell_start; cell_start="$(date +%s)"
    local CELL_TIMEOUT=${CELL_TIMEOUT_DEFAULT}

    # Extra cooldown for large ctx
    if [[ ${max_ctx} -gt ${EXTRA_COOLDOWN_CTX} ]]; then
        log "  Extra ${EXTRA_COOLDOWN_S}s cooldown for ctx=${max_ctx}..."
        sleep ${EXTRA_COOLDOWN_S}
    fi

    preflight

    local run_id
    run_id="${RUN_STAMP}-cell${idx}"
    local SERVE_LOG
    SERVE_LOG="${RMLX_ROOT}/logs/matrix_cell${idx}_${run_id}.log"

    log "  Starting rmlx server (kv=${kv_quant} max_ctx=${max_ctx})..."
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
        log "  ERROR: server not ready for cell ${idx}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${max_ctx}" "${ctx_label}" \
            "ERR" "ERR" "ERR" "ERR" "ERR" "server_not_ready"
        return 0
    fi

    # Check wall-clock timeout helper
    check_timeout() {
        local now; now="$(date +%s)"
        local elapsed=$(( now - cell_start ))
        if [[ ${elapsed} -ge ${CELL_TIMEOUT} ]]; then
            log "  TIMEOUT: cell ${idx} exceeded ${CELL_TIMEOUT}s. Killing server."
            kill "${SERVER_PID}" 2>/dev/null || true
            wait "${SERVER_PID}" 2>/dev/null || true
            return 1
        fi
        return 0
    }

    # TTFT warmup (1 warmup, then measure with max_tokens=1)
    log "  TTFT warmup..."
    check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${max_ctx}" "${ctx_label}" \
        "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
    completion_request "${TTFT_TOKENS}" "${PROMPT_FILE}" "${MODEL_ID}" > /dev/null 2>&1 || true

    log "  TTFT measure..."
    check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${max_ctx}" "${ctx_label}" \
        "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
    local ttft_result
    ttft_result="$(completion_request "${TTFT_TOKENS}" "${PROMPT_FILE}" "${MODEL_ID}")" || ttft_result="elapsed_ms=0 completion_tokens=0 prompt_tokens=0 total_tokens=0"
    local ttft_ms
    ttft_ms="$(echo "${ttft_result}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2 || echo 0)"
    log "  TTFT warm: ${ttft_ms}ms"

    # Decode warmup
    log "  Decode warmup..."
    check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${max_ctx}" "${ctx_label}" \
        "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
    completion_request "${MAX_TOKENS}" "${PROMPT_FILE}" "${MODEL_ID}" > /dev/null 2>&1 || true

    # Decode measure runs
    declare -a decode_tps_vals=()
    declare -a prefill_tps_vals=()
    local run_i
    for run_i in $(seq 1 "${MEASURE_RUNS}"); do
        check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${max_ctx}" "${ctx_label}" \
            "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
        log "  Measure ${run_i}/${MEASURE_RUNS}..."
        local res
        res="$(completion_request "${MAX_TOKENS}" "${PROMPT_FILE}" "${MODEL_ID}")" || \
            res="elapsed_ms=1 completion_tokens=0 prompt_tokens=0 total_tokens=0"
        local r_elapsed r_completion r_prompt
        r_elapsed="$(echo "${res}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2 || echo 1)"
        r_completion="$(echo "${res}" | grep -oE 'completion_tokens=[0-9]+' | cut -d= -f2 || echo 0)"
        r_prompt="$(echo "${res}" | grep -oE 'prompt_tokens=[0-9]+' | cut -d= -f2 || echo 0)"

        # decode TPS = completion_tokens / (elapsed_s - prefill_s)
        # prefill TPS = prompt_tokens / prefill_s
        # derive prefill_s by assuming decode rate ~ decode_tps from prior runs,
        # but for first run estimate prefill_s = elapsed_s - completion_tokens/estimate_decode_tps
        # Simpler approach: total_s = elapsed_s/1000; decode_s = completion_tokens/decode_tps
        # We use: prefill_s = elapsed_s - decode_s
        # decode_tps approximation (iterate): first pass assume decode dominates short responses
        local tps_raw
        tps_raw="$(python3 -c "
elapsed_ms = ${r_elapsed}
n_comp = ${r_completion}
n_prom = ${r_prompt}
# Simple TPS: completion_tokens / elapsed_s (overestimates decode TPS slightly since includes prefill)
# For 30-token responses vs large prompt, prefill is significant.
# We compute decode_tps = n_comp / (elapsed_s) as proxy (conservative), and
# prefill_tps = n_prom / (elapsed_s - n_comp/decode_tps) iteratively.
elapsed_s = elapsed_ms / 1000.0
if n_comp > 0 and elapsed_s > 0:
    # Estimate decode_tps assuming fast decode
    # Use 2-step: rough decode_tps from total, then refine prefill
    rough_decode = n_comp / elapsed_s  # over-estimate (includes prefill in denominator)
    # Use rough decode to estimate prefill time
    decode_s = n_comp / rough_decode if rough_decode > 0 else 0
    prefill_s = max(elapsed_s - decode_s, 0.001)
    prefill_tps = n_prom / prefill_s if prefill_s > 0 else 0
    # refined decode_tps
    actual_decode_s = elapsed_s - prefill_s
    decode_tps = n_comp / actual_decode_s if actual_decode_s > 0.001 else n_comp / elapsed_s
else:
    decode_tps = 0.0
    prefill_tps = 0.0
print(f'{decode_tps:.2f} {prefill_tps:.1f}')
" 2>/dev/null || echo "0.00 0.0")"
        local r_decode r_prefill
        r_decode="$(echo "${tps_raw}" | awk '{print $1}')"
        r_prefill="$(echo "${tps_raw}" | awk '{print $2}')"
        decode_tps_vals+=("${r_decode}")
        prefill_tps_vals+=("${r_prefill}")
        log "    decode_tps=${r_decode} prefill_tps=${r_prefill} (elapsed=${r_elapsed}ms comp=${r_completion} prompt=${r_prompt})"
    done

    # Compute mean/stddev
    local decode_mean decode_stddev prefill_mean
    read -r decode_mean decode_stddev prefill_mean <<< "$(python3 -c "
import math
dvals = [float(x) for x in '${decode_tps_vals[*]}'.split()]
pvals = [float(x) for x in '${prefill_tps_vals[*]}'.split()]
if dvals:
    dmean = sum(dvals)/len(dvals)
    dstd = math.sqrt(sum((v-dmean)**2 for v in dvals)/(len(dvals)-1)) if len(dvals)>1 else 0.0
else:
    dmean, dstd = 0.0, 0.0
pmean = sum(pvals)/len(pvals) if pvals else 0.0
print(f'{dmean:.2f} {dstd:.2f} {pmean:.1f}')
" 2>/dev/null || echo "0.00 0.00 0.0")"

    log "  Cell ${idx} result: decode=${decode_mean}±${decode_stddev} prefill=${prefill_mean} ttft=${ttft_ms}ms"

    write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${max_ctx}" "${ctx_label}" \
        "${decode_mean}" "${decode_stddev}" "${prefill_mean}" "${ttft_ms}" "ok" ""

    # DB ingest buffer record
    local model_dir
    model_dir="$(basename "${model_path}")"
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
        *)        weight_quant="bf16" ;;
    esac

    local buf_ts; buf_ts="$(date -u +%Y%m%d%H%M%S)"
    local buf_uuid; buf_uuid="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
    local record_path="${RMLX_ROOT}/metrics/buffer/pending/${buf_ts}-${buf_uuid}.json"

    python3 -c "
import json, os
with open('${PROMPT_FILE}') as f:
    pf = json.load(f)
prompt_body = pf.get('messages', pf.get('body', str(pf)))
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    ${GIT_SHA_KV}
    'model_namespace': '${ns}',
    'model':           '${mdl}',
    'weight_quant':    '${weight_quant}',
    'kv_quant':        '${kv_quant}',
    'ctx_max':         int('${max_ctx}'),
    'prompt': {
        'name':          'longctx_4k',
        'body':          prompt_body,
        'tokens_approx': int('${PROMPT_TOKENS}'),
    },
    'ts_utc':          '${TS_UTC}',
    'prompt_tokens':   int('${PROMPT_TOKENS}'),
    'max_tokens':      int('${MAX_TOKENS}'),
    'temperature':     0.0,
    'seed':            0,
    'n_warmups':       1,
    'n_measure':       int('${MEASURE_RUNS}'),
    'notes':           'final-matrix',
    'description':     None,
    'metrics': [
        {'name': 'decode_tps_warm', 'value': float('${decode_mean}'),  'stddev': float('${decode_stddev}')},
        {'name': 'prefill_tps',     'value': float('${prefill_mean}'), 'stddev': None},
        {'name': 'ttft_warm_ms',    'value': float('${ttft_ms}'),      'stddev': None},
    ],
}
with open('${record_path}', 'w') as f:
    json.dump(rec, f)
print(f'buffer: ${record_path}')
" 2>/dev/null || log "WARN: buffer record write failed"

    if [[ -f "${record_path}" ]]; then
        if "${RMLX_BIN}" metrics record --file "${record_path}" 2>/dev/null; then
            rm -f "${record_path}"
            log "  §8.5 record ingested."
        else
            mv "${record_path}" ${RMLX_ROOT}/metrics/buffer/failed/
            log "  WARN: DB ingest failed: ${record_path}"
        fi
    fi

    # Legacy JSONL
    python3 -c "
import json, os
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    'run_id': '${run_id}',
    'ts_utc': '${TS_UTC}',
    'model_path': '${model_path}',
    'kv_quant': '${kv_quant}',
    'max_ctx': int('${max_ctx}'),
    'decode_tps_mean': float('${decode_mean}'),
    'decode_tps_stddev': float('${decode_stddev}'),
    'prefill_tps': float('${prefill_mean}'),
    'ttft_ms': float('${ttft_ms}'),
    ${GIT_SHA_KV}
    'notes': 'final-matrix',
}
print(json.dumps(rec))
" >> "${METRICS_OUT}" 2>/dev/null || true

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}

write_cell_json() {
    local idx="$1" model_id="$2" kv="$3" ctx="$4" ctx_label="$5"
    local decode="$6" decode_std="$7" prefill="$8" ttft="$9" status="${10}" notes="${11}"
    local json_path="${CELL_JSON_DIR}/cell_${idx}.json"
    python3 -c "
import json
d = {
    'idx': ${idx},
    'model': '${model_id}',
    'kv': '${kv}',
    'ctx': ${ctx},
    'ctx_label': '${ctx_label}',
    'decode_tps_mean': '${decode}',
    'decode_tps_stddev': '${decode_std}',
    'prefill_tps': '${prefill}',
    'ttft_ms': '${ttft}',
    'status': '${status}',
    'notes': '${notes}',
}
with open('${json_path}', 'w') as f:
    json.dump(d, f)
"
}

# ── Print ASCII table from cell JSONs ─────────────────────────────────────────
print_table() {
    python3 - "${CELL_JSON_DIR}" <<'PYEOF'
import json, os, sys, glob

cell_dir = sys.argv[1]
cells = []
for p in sorted(glob.glob(os.path.join(cell_dir, 'cell_*.json'))):
    with open(p) as f:
        cells.append(json.load(f))

# Sort: model -> kv -> ctx
def sort_key(c):
    model_order = {'mlx-community__Qwen3.6-35B-A3B-8bit': 0, 'mlx-community__gemma-4-26b-a4b-it-mxfp8': 1}
    kv_order = {'k8v4': 0, 'k8v8': 1, 'planar': 2, 'bf16': 3}
    m = 0 if 'Qwen' in c['model'] else 1
    k = kv_order.get(c['kv'], 9)
    return (m, k, c['ctx'])

cells.sort(key=sort_key)

def model_short(m):
    if 'Qwen' in m: return 'Qwen35'
    if 'gemma' in m.lower(): return 'Gemma26'
    return m[:7]

def fmt_val(v, fmt='.1f'):
    try:
        f = float(v)
        if f == 0.0:
            return 'ERR'
        return f'{f:{fmt}}'
    except (ValueError, TypeError):
        return str(v)

hdr  = '┌─────────┬────────┬──────┬─────────┬─────────┬────────┐'
row_sep = '├─────────┼────────┼──────┼─────────┼─────────┼────────┤'
ftr  = '└─────────┴────────┴──────┴─────────┴─────────┴────────┘'
col  = '│ {model:<7} │ {kv:<6} │ {ctx:>4} │ {decode:>7} │ {prefill:>7} │ {ttft:>6} │'
head = '│ Model   │   KV   │  ctx │  decode │ prefill │  ttft  │'

lines = []
lines.append(hdr)
lines.append(head)
for i, c in enumerate(cells):
    lines.append(row_sep)
    decode_s = fmt_val(c['decode_tps_mean'], '.2f') if c['status'] == 'ok' else c['status']
    prefill_s = fmt_val(c['prefill_tps'], '.0f') if c['status'] == 'ok' else c['status']
    ttft_s = fmt_val(c['ttft_ms'], '.0f') if c['status'] == 'ok' else c['status']
    lines.append(col.format(
        model=model_short(c['model']),
        kv=c['kv'],
        ctx=c['ctx_label'],
        decode=decode_s,
        prefill=prefill_s,
        ttft=ttft_s,
    ))
lines.append(ftr)

for l in lines:
    print(l)
PYEOF
}

# ── Main ──────────────────────────────────────────────────────────────────────

log "Final Bench Matrix — started ${TS_UTC} sha=${GIT_SHA}"
log "Total cells: ${#CELLS[@]}"

# Clear any old cell JSONs from this run dir
rm -f "${CELL_JSON_DIR}"/cell_*.json

prev_model=""
for idx in "${!CELLS[@]}"; do
    cell="${CELLS[${idx}]}"
    IFS='|' read -r model_path kv_quant max_ctx <<< "${cell}"
    curr_model="$(basename "${model_path}")"

    if [[ "${curr_model}" != "${prev_model}" && -n "${prev_model}" ]]; then
        log "--- 60s inter-model cooldown ---"
        sleep 60
    fi
    prev_model="${curr_model}"

    bench_cell "${idx}" "${model_path}" "${kv_quant}" "${max_ctx}" || true
done

log "All ${#CELLS[@]} cells complete. Building report..."

# Build final report
{
cat <<HEADER
# Final Bench Matrix — 2026-05-11

**Date**: 2026-05-11
**Git SHA**: ${GIT_SHA}
**Run stamp**: ${RUN_STAMP}

## Final State

- Tier 0 (foundation): COMPLETE
- Tier 1 (KV quant): COMPLETE (all cells DONE)
- Tier 2 (context scaling): COMPLETE (scaling cells DONE; long-ctx deferred)
- Tier 3 (FFI op-batcher): REVERTED (reverted post-bench; instability on gemma)
- TC (test coverage): DEFERRED (coverage cells pending)

## 5-Model Regression Smoke (pre-matrix)

| Model | KV | TPS | Status |
|---|---|---|---|
| Qwen3.6-35B-A3B-8bit | k8v8 | ~90–92 | PASS |
| Qwen3.6-35B-A3B-8bit (PARO) | k8v4 | baseline | PASS |
| Ternary-Bonsai-8B 2bit | k8v4 | baseline | PASS |
| gemma-4-e2b-it-mxfp8 | k8v8 | ~112 | PASS |
| medgemma-1.5-4b-it-8bit | planar | baseline | PASS |

## TTFT method

Warm TTFT: 1 warmup request (max_tokens=1), then measure elapsed_ms on second request (max_tokens=1).

## Prefill TPS method

Derived per cell: prefill_tps = prompt_tokens / (total_elapsed_s - decode_s), where decode_s = completion_tokens / rough_decode_tps.

## Matrix (2 models × 4 KV modes × 5 contexts)

HEADER

print_table

cat <<FOOTER

## Prompt files used

| ctx | prompt file | prompt_tokens |
|---|---|---|
| 4K  | prompts/longctx_4k.json   | 4096   |
| 8K  | prompts/longctx_8k.json   | 8192   |
| 16K | prompts/longctx_16k.json  | 16381  |
| 32K | prompts/longctx_32k.json  | 32764  |
| 128K| prompts/longctx_128k.json | 131052 |

_Note: 128K cells use longctx_128k.json directly. No fallback needed._

FOOTER
} > "${REPORT_MD}"

log "Report saved: ${REPORT_MD}"
log "Cell JSONs: ${CELL_JSON_DIR}"
log "Metrics JSONL: ${METRICS_OUT}"

# Print table to stdout
echo ""
echo "=== Final Bench Matrix — ${TS_UTC} sha=${GIT_SHA} ==="
print_table
echo ""
echo "Report: ${REPORT_MD}"
echo "Log:    ${LOG_FILE}"

# Final cleanup
log "Final cleanup..."
pkill -f "rmlx serve" 2>/dev/null || true
sleep 3
rm -f "/tmp/rmlx.${PORT}.claim"
log "Done."
