#!/usr/bin/env bash
# gemma_matrix_bench.sh — Gemma-only cells (20-39)
# gemma-4-26b-a4b-it-mxfp8 × 4 KV modes × 5 contexts = 20 cells
#
# max_ctx corrected for Gemma tokenizer (produces ~6% more tokens than Qwen for same text)
# Observed: 4K prompt → 4121 Gemma tokens, 16K prompt → 17152 Gemma tokens
# Use padded max_ctx: actual_gemma_tokens + 200 headroom, rounded to 512-multiple

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
CELL_TIMEOUT_DEFAULT=1800
EXTRA_COOLDOWN_CTX=30000
EXTRA_COOLDOWN_S=30

GEMMA_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
PROMPTS_DIR="${RMLX_ROOT}/prompts"

GIT_SHA="$(git -C ${RMLX_ROOT} rev-parse --short HEAD 2>/dev/null || echo unknown)"
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_STAMP="$(date -u +%Y%m%d-%H%M%S)-${GIT_SHA}"

LOG_FILE="${RMLX_ROOT}/logs/gemma_matrix_bench_${RUN_STAMP}.log"
METRICS_OUT="${RMLX_ROOT}/metrics/perf-iter/gemma_matrix_${RUN_STAMP}.jsonl"
CELL_JSON_DIR="${RMLX_ROOT}/metrics/final_matrix_cells"

mkdir -p "$(dirname "${LOG_FILE}")"
mkdir -p "$(dirname "${METRICS_OUT}")"
mkdir -p "${CELL_JSON_DIR}"
mkdir -p ${RMLX_ROOT}/metrics/buffer/pending
mkdir -p ${RMLX_ROOT}/metrics/buffer/failed

log() { echo "[gemma_matrix] $*" | tee -a "${LOG_FILE}" >&2; }

# CELLS: kv_quant|ctx_label|max_ctx|prompt_file|idx
# max_ctx is padded for Gemma tokenizer overhead (~6-7% more tokens than Qwen)
# Observed token counts: 4K→4121, 16K→17152 (extrapolated: 8K→8465, 32K→34527, 128K→138775)
# max_ctx = ceil(gemma_tokens + 200, 512)
CELLS=(
    "k8v4|4K|4608|${PROMPTS_DIR}/longctx_4k.json|20"
    "k8v4|8K|8704|${PROMPTS_DIR}/longctx_8k.json|21"
    "k8v4|16K|17408|${PROMPTS_DIR}/longctx_16k.json|22"
    "k8v4|32K|34816|${PROMPTS_DIR}/longctx_32k.json|23"
    "k8v4|128K|139264|${PROMPTS_DIR}/longctx_128k.json|24"
    "k8v8|4K|4608|${PROMPTS_DIR}/longctx_4k.json|25"
    "k8v8|8K|8704|${PROMPTS_DIR}/longctx_8k.json|26"
    "k8v8|16K|17408|${PROMPTS_DIR}/longctx_16k.json|27"
    "k8v8|32K|34816|${PROMPTS_DIR}/longctx_32k.json|28"
    "k8v8|128K|139264|${PROMPTS_DIR}/longctx_128k.json|29"
    "planar|4K|4608|${PROMPTS_DIR}/longctx_4k.json|30"
    "planar|8K|8704|${PROMPTS_DIR}/longctx_8k.json|31"
    "planar|16K|17408|${PROMPTS_DIR}/longctx_16k.json|32"
    "planar|32K|34816|${PROMPTS_DIR}/longctx_32k.json|33"
    "planar|128K|139264|${PROMPTS_DIR}/longctx_128k.json|34"
    "bf16|4K|4608|${PROMPTS_DIR}/longctx_4k.json|35"
    "bf16|8K|8704|${PROMPTS_DIR}/longctx_8k.json|36"
    "bf16|16K|17408|${PROMPTS_DIR}/longctx_16k.json|37"
    "bf16|32K|34816|${PROMPTS_DIR}/longctx_32k.json|38"
    "bf16|128K|139264|${PROMPTS_DIR}/longctx_128k.json|39"
)

_CR_PAYLOAD_TMP="/tmp/rmlx_gemma_payload_$$.json"
_CR_RESP_TMP="/tmp/rmlx_gemma_resp_$$.json"

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

completion_request() {
    local max_tokens="$1"
    local prompt_file="$2"
    local model_id="$3"

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

write_cell_json() {
    local idx="$1" model_id="$2" kv="$3" ctx_label="$4" max_ctx="$5"
    local decode="$6" decode_std="$7" prefill_tps="$8" ttft="$9" status="${10}" notes="${11}"
    local json_path="${CELL_JSON_DIR}/cell_${idx}.json"
    python3 -c "
import json
d = {
    'idx': ${idx},
    'model': '${model_id}',
    'kv': '${kv}',
    'ctx': ${max_ctx},
    'ctx_label': '${ctx_label}',
    'decode_tps_mean': '${decode}',
    'decode_tps_stddev': '${decode_std}',
    'prefill_tps': '${prefill_tps}',
    'ttft_ms': '${ttft}',
    'status': '${status}',
    'notes': '${notes}',
}
with open('${json_path}', 'w') as f:
    json.dump(d, f)
"
}

bench_cell() {
    local kv_quant="$1"
    local ctx_label="$2"
    local max_ctx="$3"
    local prompt_file="$4"
    local idx="$5"

    local MODEL_ID
    MODEL_ID="$(basename "${GEMMA_PATH}")"
    log "=== Cell ${idx}: model=${MODEL_ID} kv=${kv_quant} ctx=${ctx_label} max_ctx=${max_ctx} ==="

    local PROMPT_TOKENS
    PROMPT_TOKENS="$(python3 -c "import json; d=json.load(open('${prompt_file}')); print(d.get('prompt_tokens', 0))")"
    log "  prompt_file=${prompt_file} prompt_tokens=${PROMPT_TOKENS}"

    local cell_start; cell_start="$(date +%s)"

    if [[ ${max_ctx} -gt ${EXTRA_COOLDOWN_CTX} ]]; then
        log "  Extra ${EXTRA_COOLDOWN_S}s cooldown for max_ctx=${max_ctx}..."
        sleep ${EXTRA_COOLDOWN_S}
    fi

    preflight

    local run_id="${RUN_STAMP}-cell${idx}"
    local SERVE_LOG="${RMLX_ROOT}/logs/gemma_matrix_cell${idx}_${run_id}.log"

    log "  Starting rmlx server (kv=${kv_quant} max_ctx=${max_ctx})..."
    "${RMLX_BIN}" serve \
        --model "${GEMMA_PATH}" \
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
        write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${ctx_label}" "${max_ctx}" \
            "ERR" "ERR" "ERR" "ERR" "ERR" "server_not_ready"
        return 0
    fi

    check_timeout() {
        local now; now="$(date +%s)"
        local elapsed=$(( now - cell_start ))
        if [[ ${elapsed} -ge ${CELL_TIMEOUT_DEFAULT} ]]; then
            log "  TIMEOUT: cell ${idx} exceeded ${CELL_TIMEOUT_DEFAULT}s. Killing server."
            kill "${SERVER_PID}" 2>/dev/null || true
            wait "${SERVER_PID}" 2>/dev/null || true
            return 1
        fi
        return 0
    }

    # TTFT cold: first request = cold prefill (no cache). Captures real prefill latency.
    log "  TTFT cold (first request)..."
    check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${ctx_label}" "${max_ctx}" \
        "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
    local cold_ttft_result
    cold_ttft_result="$(completion_request "${TTFT_TOKENS}" "${prompt_file}" "${MODEL_ID}")" || \
        cold_ttft_result="elapsed_ms=0 completion_tokens=0 prompt_tokens=0 total_tokens=0"
    local cold_ttft_ms
    cold_ttft_ms="$(echo "${cold_ttft_result}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2 || echo 0)"
    log "  TTFT cold: ${cold_ttft_ms}ms"

    # TTFT warm: second request = prompt cache hit. Captures repeated-query latency.
    log "  TTFT warm (second request, cache hit)..."
    check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${ctx_label}" "${max_ctx}" \
        "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
    local ttft_result
    ttft_result="$(completion_request "${TTFT_TOKENS}" "${prompt_file}" "${MODEL_ID}")" || \
        ttft_result="elapsed_ms=0 completion_tokens=0 prompt_tokens=0 total_tokens=0"
    local ttft_ms
    ttft_ms="$(echo "${ttft_result}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2 || echo 0)"
    log "  TTFT warm: ${ttft_ms}ms"

    # Prefill TPS from cold TTFT: prefill_tps = prompt_tokens / (cold_ttft_ms / 1000)
    local prefill_tps
    prefill_tps="$(python3 -c "
pt = ${PROMPT_TOKENS}
ttft_s = ${cold_ttft_ms} / 1000.0
print(f'{pt/ttft_s:.1f}' if ttft_s > 0 else '0.0')
" 2>/dev/null || echo "0.0")"
    log "  Prefill TPS (from cold TTFT): ${prefill_tps}"

    # Decode warmup
    log "  Decode warmup..."
    check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${ctx_label}" "${max_ctx}" \
        "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
    completion_request "${MAX_TOKENS}" "${prompt_file}" "${MODEL_ID}" > /dev/null 2>&1 || true

    # Measure runs
    declare -a decode_tps_vals=()
    local run_i
    for run_i in $(seq 1 "${MEASURE_RUNS}"); do
        check_timeout || { write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${ctx_label}" "${max_ctx}" \
            "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "TIMEOUT" "cell_timeout"; return 0; }
        log "  Measure ${run_i}/${MEASURE_RUNS}..."
        local res
        res="$(completion_request "${MAX_TOKENS}" "${prompt_file}" "${MODEL_ID}")" || \
            res="elapsed_ms=1 completion_tokens=0 prompt_tokens=0 total_tokens=0"
        local r_elapsed r_completion r_prompt
        r_elapsed="$(echo "${res}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2 || echo 1)"
        r_completion="$(echo "${res}" | grep -oE 'completion_tokens=[0-9]+' | cut -d= -f2 || echo 0)"
        r_prompt="$(echo "${res}" | grep -oE 'prompt_tokens=[0-9]+' | cut -d= -f2 || echo 0)"

        # Decode TPS: total_elapsed - cold_ttft (prefill) = decode time
        local decode_tps
        decode_tps="$(python3 -c "
elapsed_ms = ${r_elapsed}
n_comp = ${r_completion}
cold_ttft_ms = ${cold_ttft_ms}
elapsed_s = elapsed_ms / 1000.0
# For decode measure runs, prompt cache is warm (hit), so minimal prefill overhead.
# Use warm ttft (cache hit time) as the prefill subtraction.
warm_ttft_ms = ${ttft_ms}
prefill_s = warm_ttft_ms / 1000.0
decode_s = max(elapsed_s - prefill_s, 0.001)
if n_comp > 0 and decode_s > 0:
    tps = n_comp / decode_s
else:
    tps = 0.0
print(f'{tps:.2f}')
" 2>/dev/null || echo "0.00")"
        decode_tps_vals+=("${decode_tps}")
        log "    decode_tps=${decode_tps} (elapsed=${r_elapsed}ms comp=${r_completion} ttft_sub=${ttft_ms}ms)"
    done

    local decode_mean decode_stddev
    read -r decode_mean decode_stddev <<< "$(python3 -c "
import math
dvals = [float(x) for x in '${decode_tps_vals[*]}'.split()]
if dvals:
    dmean = sum(dvals)/len(dvals)
    dstd = math.sqrt(sum((v-dmean)**2 for v in dvals)/(len(dvals)-1)) if len(dvals)>1 else 0.0
else:
    dmean, dstd = 0.0, 0.0
print(f'{dmean:.2f} {dstd:.2f}')
" 2>/dev/null || echo "0.00 0.00")"

    log "  Cell ${idx} result: decode=${decode_mean}±${decode_stddev} prefill=${prefill_tps} ttft_cold=${cold_ttft_ms}ms ttft_warm=${ttft_ms}ms"

    write_cell_json "${idx}" "${MODEL_ID}" "${kv_quant}" "${ctx_label}" "${max_ctx}" \
        "${decode_mean}" "${decode_stddev}" "${prefill_tps}" "${cold_ttft_ms}" "ok" "ttft_warm=${ttft_ms}"

    # Buffer record
    local ns="mlx-community"
    local mdl="gemma-4-26b-a4b-it-mxfp8"
    local weight_quant="mxfp8"
    local buf_ts; buf_ts="$(date -u +%Y%m%d%H%M%S)"
    local buf_uuid; buf_uuid="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
    local record_path="${RMLX_ROOT}/metrics/buffer/pending/${buf_ts}-${buf_uuid}.json"

    python3 -c "
import json, os
with open('${prompt_file}') as f:
    pf = json.load(f)
prompt_body = pf.get('messages', pf.get('body', str(pf)))
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
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
    'notes':           'gemma-matrix-bench',
    'description':     None,
    'metrics': [
        {'name': 'decode_tps_warm', 'value': float('${decode_mean}'),  'stddev': float('${decode_stddev}')},
        {'name': 'prefill_tps',     'value': float('${prefill_tps}'),  'stddev': None},
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

    python3 -c "
import json, os
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    'run_id': '${run_id}',
    'ts_utc': '${TS_UTC}',
    'model_path': '${GEMMA_PATH}',
    'kv_quant': '${kv_quant}',
    'max_ctx': int('${max_ctx}'),
    'ctx_label': '${ctx_label}',
    'decode_tps_mean': float('${decode_mean}'),
    'decode_tps_stddev': float('${decode_stddev}'),
    'prefill_tps': float('${prefill_tps}'),
    'ttft_ms': float('${ttft_ms}'),
    'notes': 'gemma-matrix-bench',
}
print(json.dumps(rec))
" >> "${METRICS_OUT}" 2>/dev/null || true

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}

# ── Main ──────────────────────────────────────────────────────────────────────
log "Gemma Matrix Bench — started ${TS_UTC} sha=${GIT_SHA}"
log "Total cells: ${#CELLS[@]}"

for cell_spec in "${CELLS[@]}"; do
    IFS='|' read -r kv_quant ctx_label max_ctx prompt_file idx <<< "${cell_spec}"
    bench_cell "${kv_quant}" "${ctx_label}" "${max_ctx}" "${prompt_file}" "${idx}" || true
done

log "All Gemma cells complete."

# Final cleanup
pkill -f "rmlx serve" 2>/dev/null || true
sleep 3
rm -f "/tmp/rmlx.${PORT}.claim"
log "Done."
