#!/usr/bin/env bash
# p0b_ttft_only.sh — focused cold-TTFT pre/post bench.
#
# For each (model, ctx) cell: start server with PRE (RMLX_PREFILL_CHUNK=64)
# then POST (per-arch new default), measure cold TTFT (single max_tokens=1
# request after fresh start) and warm decode TPS (subsequent max_tokens=20
# request hitting prompt cache).
#
# All cold prompts use the same prompt file at the right ctx.
#
# Output: BENCH_RESULT one-liners on stdout + log.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
GEMMA_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
PROMPT_DIR="${RMLX_ROOT}/prompts"
TS="$(date -u +%Y%m%d-%H%M%S)"
REPORT_LOG="${RMLX_ROOT}/logs/p0b_ttft_${TS}.log"
mkdir -p "$(dirname "${REPORT_LOG}")"

log() { echo "[p0b-ttft] $*" | tee -a "${REPORT_LOG}" >&2; }

PAYLOAD_TMP="/tmp/p0b_ttft_payload_$$.json"
RESP_TMP="/tmp/p0b_ttft_resp_$$.json"

ctx_to_prompt() {
    case "$1" in
        4096)  echo "${PROMPT_DIR}/longctx_4k.json"   ;;
        8192)  echo "${PROMPT_DIR}/longctx_8k.json"   ;;
        16384) echo "${PROMPT_DIR}/longctx_16k.json"  ;;
        32768) echo "${PROMPT_DIR}/longctx_32k.json"  ;;
        65536) echo "${PROMPT_DIR}/longctx_64k.json"  ;;
    esac
}

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local e=0
    until curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; do
        sleep 5; e=$((e+5))
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then return 1; fi
        [[ ${e} -ge 600 ]] && return 1
    done
    log "    health ok in ${e}s"
}

# Build payload once per (prompt_file, model_id, max_tokens)
build_payload() {
    local prompt_file="$1" model_id="$2" max_tokens="$3"
    python3 - "${prompt_file}" "${model_id}" "${max_tokens}" "${PAYLOAD_TMP}" <<'PYEOF'
import json, sys
prompt_file, model_id, max_tokens, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
with open(prompt_file) as f:
    pf = json.load(f)
payload = {'model': model_id, 'messages': pf['messages'],
           'max_tokens': max_tokens, 'temperature': 0.0, 'stream': False}
with open(out_path, 'w') as f:
    json.dump(payload, f)
PYEOF
}

# fire request, return "elapsed_ms=… tokens=…"
fire() {
    local timeout_s="$1"
    local t0; t0="$(python3 -c 'import time;print(int(time.time()*1000))')"
    curl -s --max-time "${timeout_s}" -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "@${PAYLOAD_TMP}" -o "${RESP_TMP}" 2>/dev/null
    local t1; t1="$(python3 -c 'import time;print(int(time.time()*1000))')"
    local elapsed_ms=$((t1 - t0))
    local tok
    tok="$(python3 -c "
import json
try:
    d = json.load(open('${RESP_TMP}'))
    print(d.get('usage', {}).get('completion_tokens', 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0)"
    echo "elapsed_ms=${elapsed_ms} tokens=${tok}"
}

bench_cell() {
    local model_path="$1" kv="$2" ctx="$3" mode="$4"
    local model_id; model_id="$(basename "${model_path}")"
    local prompt_file; prompt_file="$(ctx_to_prompt "${ctx}")"
    [[ -z "${prompt_file}" ]] && { log "no prompt for ctx=${ctx}"; return; }

    log "=== ${model_id} kv=${kv} ctx=${ctx} mode=${mode} ==="
    preflight

    SERVE_LOG="${RMLX_ROOT}/logs/p0b_${model_id}_${ctx}_${mode}_${TS}.log"
    local env_pin=""
    if [[ "${mode}" == "PRE" ]]; then env_pin="RMLX_PREFILL_CHUNK=64"; fi

    log "  starting server..."
    # shellcheck disable=SC2086
    env ${env_pin} "${RMLX_BIN}" serve \
        --model "${model_path}" --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant "${kv}" --max-ctx "${ctx}" \
        > "${SERVE_LOG}" 2>&1 &
    SERVER_PID=$!
    if ! wait_health; then
        log "  ERROR: server didn't start"
        echo "BENCH_RESULT model=${model_id} ctx=${ctx} mode=${mode} ttft_ms=ERROR decode_tps=ERROR" \
            | tee -a "${REPORT_LOG}"
        kill "${SERVER_PID}" 2>/dev/null; return
    fi

    # COLD: max_tokens=1, captures TTFT-prefill. Generous timeout (1200s) for 64K cells.
    log "  cold TTFT request..."
    build_payload "${prompt_file}" "${model_id}" 1
    local cold; cold="$(fire 1200)"
    local cold_ms; cold_ms="$(echo "${cold}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local cold_tok; cold_tok="$(echo "${cold}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
    log "    cold ms=${cold_ms} tokens=${cold_tok}"

    # WARM: max_tokens=20, prompt cache should hit so prefill ~0; isolate decode TPS
    log "  warm decode request..."
    build_payload "${prompt_file}" "${model_id}" 20
    local warm; warm="$(fire 600)"
    local warm_ms; warm_ms="$(echo "${warm}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local warm_tok; warm_tok="$(echo "${warm}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
    local decode_tps
    decode_tps="$(python3 -c "
ms=${warm_ms}; n=${warm_tok}
print(f'{n/(ms/1000):.2f}' if n > 0 and ms > 0 else '0.0')
")"
    log "    warm ms=${warm_ms} tokens=${warm_tok} decode_tps≈${decode_tps}"

    echo "BENCH_RESULT model=${model_id} ctx=${ctx} mode=${mode} ttft_ms=${cold_ms} decode_tps=${decode_tps}" \
        | tee -a "${REPORT_LOG}"

    kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null || true
}

# Cells: (model, kv, ctx)
# Ordered by model so we minimize model swaps.
QWEN_CTXS=(8192 32768)
GEMMA_CTXS=(8192 32768)

for ctx in "${QWEN_CTXS[@]}"; do
    bench_cell "${QWEN_PATH}" "k8v8" "${ctx}" PRE
    sleep 30
    bench_cell "${QWEN_PATH}" "k8v8" "${ctx}" POST
    sleep 30
done

log "--- 60s inter-model cooldown ---"
sleep 60

for ctx in "${GEMMA_CTXS[@]}"; do
    bench_cell "${GEMMA_PATH}" "planar" "${ctx}" PRE
    sleep 30
    bench_cell "${GEMMA_PATH}" "planar" "${ctx}" POST
    sleep 30
done

log "DONE. Report at ${REPORT_LOG}"
