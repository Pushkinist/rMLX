#!/usr/bin/env bash
# d8-phase1-measure.sh — D8 Phase 1: quantify first-dispatch MSL-compile tax.
#
# Method: cold process -> req1 (small, max_tokens=8, short prompt) -> req2
# (identical). delta(req1-req2) with prefill subtracted via an n1/n8 pair
# (same prefix, max_tokens=1 vs 8) is the first-dispatch MSL-compile estimate.
#
# Short prompt (few tokens) minimises prefill so the req1-req2 gap is
# dominated by kernel JIT + Metal pipeline-state creation, not prompt work.
#
# 2 models:
#   - mlx-community__Qwen3.6-35B-A3B-8bit   kv=k8v8 (GDN-warmed)
#   - prism-ml__Ternary-Bonsai-8B-mlx-2bit  kv=planar
#
# Output: D8P1 lines on stdout + log.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
TS="$(date -u +%Y%m%d-%H%M%S)"
LOG="${RMLX_ROOT}/metrics/d8-phase1-${TS}.log"
mkdir -p "$(dirname "${LOG}")"
RESP_TMP="/tmp/d8p1_resp_$$.json"
PAYLOAD_TMP="/tmp/d8p1_payload_$$.json"

log() { echo "[d8p1] $*" | tee -a "${LOG}" >&2; }

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm 2>/dev/null || true
    pkill -f paroquant 2>/dev/null || true
    pkill -f omlx 2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local e=0
    until curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; do
        sleep 3; e=$((e+3))
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then return 1; fi
        [[ ${e} -ge 600 ]] && return 1
    done
    log "    health ok in ${e}s"
}

build_payload() {
    local model_id="$1" max_tokens="$2"
    python3 - "${model_id}" "${max_tokens}" "${PAYLOAD_TMP}" <<'PYEOF'
import json, sys
model_id, max_tokens, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
payload = {'model': model_id,
           'messages': [{'role': 'user', 'content': 'Say hello.'}],
           'max_tokens': max_tokens, 'temperature': 0.0, 'stream': False}
with open(out_path, 'w') as f:
    json.dump(payload, f)
PYEOF
}

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

measure() {
    local model_path="$1" kv="$2" ctx="$3"
    local model_id; model_id="$(basename "${model_path}")"

    log "=== ${model_id} kv=${kv} ctx=${ctx} ==="
    preflight

    SERVE_LOG="${RMLX_ROOT}/logs/d8p1_${model_id}_${TS}.log"
    log "  starting server..."
    "${RMLX_BIN}" serve \
        --model "${model_path}" --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant "${kv}" --max-ctx "${ctx}" \
        > "${SERVE_LOG}" 2>&1 &
    SERVER_PID=$!
    if ! wait_health; then
        log "  ERROR: server didn't start (see ${SERVE_LOG})"
        kill "${SERVER_PID}" 2>/dev/null
        return
    fi

    # LOAD BARRIER: POST /v1/models/{id}/load is synchronous (ensure_loaded
    # blocks until the model is fully loaded + dequanted + GDN-warmed).
    # It does NOT run a real inference, so the 8 custom MSL kernels are still
    # uncompiled after this returns. This isolates model-load cost OUT of the
    # cold-req1 measurement, leaving first-dispatch MSL compile + decode.
    log "  loading model (sync barrier)..."
    local lt0; lt0="$(python3 -c 'import time;print(int(time.time()*1000))')"
    curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/models/${model_id}/load" \
        -o /tmp/d8p1_load_$$.json 2>/dev/null
    local lt1; lt1="$(python3 -c 'import time;print(int(time.time()*1000))')"
    local load_ms=$((lt1 - lt0))
    log "    load barrier returned in ${load_ms} ms ($(cat /tmp/d8p1_load_$$.json 2>/dev/null))"
    # total_load_ms as the server itself measured it.
    local srv_load_ms
    srv_load_ms="$(grep -oE 'total_load_ms[^0-9]+[0-9]+' "${SERVE_LOG}" 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || echo NA)"

    # req1: COLD first request (max_tokens=8, tiny prompt). Model already
    # resident → this is first-dispatch MSL compile + decode, NOT load.
    build_payload "${model_id}" 8
    local rc1; rc1="$(fire 600)"
    local cold_ms; cold_ms="$(echo "${rc1}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local cold_n;  cold_n="$(echo "${rc1}"  | grep -oE 'tokens=[0-9]+'    | cut -d= -f2)"
    log "  req1 (cold, max=8): elapsed_ms=${cold_ms} tokens=${cold_n}"

    # req2: identical, now warm.
    build_payload "${model_id}" 8
    local rw2; rw2="$(fire 600)"
    local warm_ms; warm_ms="$(echo "${rw2}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local warm_n;  warm_n="$(echo "${rw2}"  | grep -oE 'tokens=[0-9]+'    | cut -d= -f2)"
    log "  req2 (warm, max=8): elapsed_ms=${warm_ms} tokens=${warm_n}"

    # req3: identical again, to confirm warm is stable (rule out req2 still
    # paying residual compile).
    build_payload "${model_id}" 8
    local rw3; rw3="$(fire 600)"
    local warm3_ms; warm3_ms="$(echo "${rw3}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    log "  req3 (warm, max=8): elapsed_ms=${warm3_ms}"

    # Prefill-subtraction control: warm n1 vs warm n8 (same tiny prompt)
    # isolates per-token decode cost so we can confirm the cold-warm gap is
    # NOT explained by token-count differences (it is not — n is identical).
    build_payload "${model_id}" 1
    local rn1; rn1="$(fire 600)"
    local n1_ms; n1_ms="$(echo "${rn1}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    build_payload "${model_id}" 8
    local rn8; rn8="$(fire 600)"
    local n8_ms; n8_ms="$(echo "${rn8}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"

    local tax
    tax="$(python3 -c "
cold=${cold_ms}; warm=${warm_ms}; warm3=${warm3_ms}
print(f'{cold-warm} (cold-req2)  {cold-warm3} (cold-req3)')
")"
    log "  >>> MSL-compile tax estimate (post load-barrier): ${tax} ms"
    echo "D8P1 model=${model_id} kv=${kv} load_barrier_ms=${load_ms} srv_total_load_ms=${srv_load_ms} cold_ms=${cold_ms}(n=${cold_n}) warm2_ms=${warm_ms}(n=${warm_n}) warm3_ms=${warm3_ms} n1_ms=${n1_ms} n8_ms=${n8_ms} tax_cold_minus_warm2=$((cold_ms-warm_ms)) tax_cold_minus_warm3=$((cold_ms-warm3_ms))" \
        | tee -a "${LOG}"

    # Grep serve log for any first-dispatch / compile timing already traced.
    log "  serve-log compile/warmup traces:"
    grep -iE 'compile|warmup|first_kernel|prewarm|pre-warm|gdn|T0\.4|kernel.*ready|jit' "${SERVE_LOG}" 2>/dev/null \
        | head -40 | tee -a "${LOG}" >&2 || log "    (none)"

    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null || true
    sleep 30
}

measure "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"  "k8v8"   4096
measure "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/prism-ml__Ternary-Bonsai-8B-mlx-2bit" "planar" 4096

log "DONE. Log at ${LOG}"
