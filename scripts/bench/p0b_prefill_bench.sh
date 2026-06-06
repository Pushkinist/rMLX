#!/usr/bin/env bash
# p0b_prefill_bench.sh — pre/post PREFILL_CHUNK tuning bench.
#
# Compares cold TTFT + decode TPS across context lengths for two priority
# models. Each cell runs twice:
#   PRE  = RMLX_PREFILL_CHUNK=64 (pin to legacy default)
#   POST = unset (pick up the per-arch new default)
#
# Models:
#   - mlx-community/Qwen3.6-35B-A3B-8bit  → qwen3_5_moe (default 256 vs pinned 64)
#   - mlx-community/gemma-4-26b-a4b-it-mxfp8 → gemma4 (default 512 vs pinned 64)
#
# Contexts: 8K, 16K, 32K, 64K
#
# Output:
#   - logs/p0b_prefill_bench_<ts>.log
#   - one row per (model, ctx, mode) appended to logs as
#     `BENCH_RESULT model=… ctx=… mode=… ttft_ms=… decode_tps=…`
#   - results parsed by the executor for the report.
#
# Usage: ./scripts/bench/p0b_prefill_bench.sh [model_filter]
#   model_filter: optional `qwen` or `gemma` to limit run.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
WARMUP_RUNS=1
MEASURE_RUNS=1
MAX_TOKENS=20
HEALTH_TIMEOUT=600
QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
GEMMA_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
PROMPT_DIR="${RMLX_ROOT}/prompts"
TS="$(date -u +%Y%m%d-%H%M%S)"
REPORT_LOG="${RMLX_ROOT}/logs/p0b_prefill_bench_${TS}.log"

mkdir -p "$(dirname "${REPORT_LOG}")"

log() { echo "[p0b] $*" | tee -a "${REPORT_LOG}" >&2; }
die() { log "ERROR: $*"; exit 1; }

FILTER="${1:-}"

CELLS=()
add_cells() {
    local label="$1"
    local model_path="$2"
    local kv_quant="$3"
    for ctx in 8192 16384 32768 65536; do
        CELLS+=("${label}|${model_path}|${kv_quant}|${ctx}")
    done
}

if [[ -z "${FILTER}" || "${FILTER}" == "qwen" ]]; then
    add_cells qwen "${QWEN_PATH}" "k8v8"
fi
if [[ -z "${FILTER}" || "${FILTER}" == "gemma" ]]; then
    add_cells gemma "${GEMMA_PATH}" "planar"
fi

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    pkill -f paroquant   2>/dev/null || true
    pkill -f omlx        2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local elapsed=0
    until curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "Server crashed. Tail of log:"
            tail -30 "${SERVE_LOG}" >&2
            return 1
        fi
        if [[ ${elapsed} -ge ${HEALTH_TIMEOUT} ]]; then
            log "Health timeout ${HEALTH_TIMEOUT}s."
            kill "${SERVER_PID}" 2>/dev/null || true
            return 1
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    log "  health ready in ${elapsed}s"
    return 0
}

ctx_to_prompt_file() {
    local ctx="$1"
    case "${ctx}" in
        8192)  echo "${PROMPT_DIR}/longctx_8k.json"   ;;
        16384) echo "${PROMPT_DIR}/longctx_16k.json"  ;;
        32768) echo "${PROMPT_DIR}/longctx_32k.json"  ;;
        65536) echo "${PROMPT_DIR}/longctx_64k.json"  ;;
        *)     echo ""                                 ;;
    esac
}

PAYLOAD_TMP="/tmp/p0b_payload_$$.json"
RESP_TMP="/tmp/p0b_resp_$$.json"

# Returns "elapsed_ms=… completion_tokens=…"
single_request() {
    local prompt_file="$1"
    local max_tokens="$2"
    python3 - "${prompt_file}" "${MODEL_ID}" "${max_tokens}" "${PAYLOAD_TMP}" <<'PYEOF'
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
        -d "@${PAYLOAD_TMP}" \
        -o "${RESP_TMP}" 2>/dev/null
    t_end="$(python3 -c 'import time; print(int(time.time()*1000))')"
    local elapsed_ms=$(( t_end - t_start ))
    local completion_tokens
    completion_tokens="$(python3 -c "
import json
try:
    d = json.load(open('${RESP_TMP}'))
    print(d.get('usage', {}).get('completion_tokens', 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0)"
    echo "elapsed_ms=${elapsed_ms} completion_tokens=${completion_tokens}"
}

bench_cell() {
    local model_path="$1"
    local kv_quant="$2"
    local max_ctx="$3"
    local mode="$4"  # PRE or POST

    MODEL_ID="$(basename "${model_path}")"
    local prompt_file
    prompt_file="$(ctx_to_prompt_file "${max_ctx}")"
    [[ -z "${prompt_file}" ]] && { log "no prompt for ctx=${max_ctx}"; return 1; }

    log "=== ${MODEL_ID} kv=${kv_quant} ctx=${max_ctx} mode=${mode} ==="

    preflight

    SERVE_LOG="${RMLX_ROOT}/logs/p0b_${MODEL_ID}_${max_ctx}_${mode}_${TS}.log"
    log "  starting server (mode=${mode})..."

    local env_pin=""
    if [[ "${mode}" == "PRE" ]]; then
        env_pin="RMLX_PREFILL_CHUNK=64"
    fi

    # shellcheck disable=SC2086
    env ${env_pin} "${RMLX_BIN}" serve \
        --model "${model_path}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        --device gpu \
        --kv-quant "${kv_quant}" \
        --max-ctx "${max_ctx}" \
        > "${SERVE_LOG}" 2>&1 &
    SERVER_PID=$!

    if ! wait_health; then
        log "  ERROR: server not ready"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        echo "BENCH_RESULT model=${MODEL_ID} ctx=${max_ctx} mode=${mode} ttft_ms=ERROR decode_tps=ERROR" \
            | tee -a "${REPORT_LOG}"
        return 1
    fi

    # COLD ttft = 1 request with max_tokens=1 to isolate prefill
    log "  cold TTFT (max_tokens=1)..."
    local cold_result
    cold_result="$(single_request "${prompt_file}" 1)" || cold_result="elapsed_ms=0 completion_tokens=0"
    local cold_ms
    cold_ms="$(echo "${cold_result}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    log "    cold ttft_ms=${cold_ms}"

    # Warm decode TPS measurement
    log "  warm decode (max_tokens=${MAX_TOKENS})..."
    local warm_result
    warm_result="$(single_request "${prompt_file}" "${MAX_TOKENS}")" || warm_result="elapsed_ms=0 completion_tokens=0"
    local warm_ms warm_tokens
    warm_ms="$(echo "${warm_result}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    warm_tokens="$(echo "${warm_result}" | grep -oE 'completion_tokens=[0-9]+' | cut -d= -f2)"

    # Decode TPS = completion_tokens / (warm_ms - cold_ms) (warm prompt is cached;
    # second run with N tokens has ~zero TTFT, so decode = warm_ms / N approx).
    # Approximate decode TPS:
    local decode_tps
    decode_tps="$(python3 -c "
ms=${warm_ms}; n=${warm_tokens}
print(f'{n/(ms/1000):.2f}' if n > 0 and ms > 0 else '0.0')
")"

    log "    warm ms=${warm_ms} tokens=${warm_tokens} decode_tps≈${decode_tps}"

    echo "BENCH_RESULT model=${MODEL_ID} ctx=${max_ctx} mode=${mode} ttft_ms=${cold_ms} decode_tps=${decode_tps}" \
        | tee -a "${REPORT_LOG}"

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}

# ── Main loop ────────────────────────────────────────────────────────────────
prev_model=""
for cell in "${CELLS[@]}"; do
    IFS='|' read -r label model_path kv_quant max_ctx <<< "${cell}"
    curr_model="$(basename "${model_path}")"
    if [[ "${curr_model}" != "${prev_model}" && -n "${prev_model}" ]]; then
        log "--- 60s inter-model cooldown ---"
        sleep 60
    fi
    prev_model="${curr_model}"

    # PRE then POST, with 30s cooldown between modes
    bench_cell "${model_path}" "${kv_quant}" "${max_ctx}" PRE  || true
    sleep 30
    bench_cell "${model_path}" "${kv_quant}" "${max_ctx}" POST || true
    sleep 30
done

log "All cells complete. Results in ${REPORT_LOG}"
