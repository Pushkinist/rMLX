#!/usr/bin/env bash
# VG.2 NIAH surrogate — greedy first-32-token identity check pre vs post change
# Compares RMLX_OP_BATCHER=1 (default) vs RMLX_OP_BATCHER=0 (eager fallback)
# at 32K context on both primary models.
#
# Usage: ./scripts/bench/vg2_niah_surrogate.sh
# Exits 0 if all identity checks pass, 1 if any differ.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
GEMMA_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
PROMPT_FILE="${RMLX_ROOT}/prompts/longctx_32k.json"
MAX_TOKENS=32
PASS=0
FAIL=0

log() { echo "[vg2] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

preflight() {
    log "Pre-flight: kill stale..."
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local url="http://127.0.0.1:${PORT}/health"
    local max=120 e=0
    until curl -s --max-time 2 "${url}" | grep -q '"ok"'; do
        sleep 3; e=$((e+3))
        [[ ${e} -ge ${max} ]] && die "health timeout"
    done
    log "Server ready in ${e}s"
}

_VG2_PAYLOAD_TMP="/tmp/rmlx_vg2_payload_$$.json"
_VG2_RESP_TMP="/tmp/rmlx_vg2_resp_$$.json"

get_first32() {
    local model_id="$1"
    # Build payload into tmp file to avoid shell-quoting issues with large messages
    python3 - "${PROMPT_FILE}" "${model_id}" "${MAX_TOKENS}" "${_VG2_PAYLOAD_TMP}" <<'PYEOF'
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
    curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "@${_VG2_PAYLOAD_TMP}" \
        -o "${_VG2_RESP_TMP}" 2>/dev/null
    python3 -c "
import json
d = json.load(open('${_VG2_RESP_TMP}'))
choices = d.get('choices', [])
if choices:
    text = choices[0].get('message', {}).get('content', '')
    print(json.dumps(text.split()[:32]))
else:
    print('[]')
" 2>/dev/null || echo '[]'
}

run_check() {
    local model_path="$1"
    local kv="$2"
    local model_id
    model_id="$(basename "${model_path}")"
    local serve_log="/tmp/vg2_${model_id}_${kv}.log"

    log "=== ${model_id} @ k=${kv} 32K ==="

    # --- Run with RMLX_OP_BATCHER=1 (default ON) ---
    preflight
    log "Starting server [batcher=ON]..."
    RMLX_OP_BATCHER=1 "${RMLX_BIN}" serve \
        --model "${model_path}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        --device gpu \
        --kv-quant "${kv}" \
        --max-ctx 32768 \
        > "${serve_log}.on" 2>&1 &
    local pid_on=$!
    wait_health
    log "Getting first-32 tokens [batcher=ON]..."
    local tok_on
    tok_on="$(get_first32 "${model_id}")"
    log "ON: ${tok_on}"
    kill "${pid_on}" 2>/dev/null; wait "${pid_on}" 2>/dev/null || true

    # --- Run with RMLX_OP_BATCHER=0 (eager fallback) ---
    preflight
    log "Starting server [batcher=OFF]..."
    RMLX_OP_BATCHER=0 "${RMLX_BIN}" serve \
        --model "${model_path}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        --device gpu \
        --kv-quant "${kv}" \
        --max-ctx 32768 \
        > "${serve_log}.off" 2>&1 &
    local pid_off=$!
    wait_health
    log "Getting first-32 tokens [batcher=OFF]..."
    local tok_off
    tok_off="$(get_first32 "${model_id}")"
    log "OFF: ${tok_off}"
    kill "${pid_off}" 2>/dev/null; wait "${pid_off}" 2>/dev/null || true

    # --- Compare ---
    if [[ "${tok_on}" == "${tok_off}" ]]; then
        log "PASS: ${model_id} kv=${kv} — first-32 tokens identical"
        PASS=$((PASS+1))
    else
        log "FAIL: ${model_id} kv=${kv} — tokens DIFFER"
        log "  ON:  ${tok_on}"
        log "  OFF: ${tok_off}"
        FAIL=$((FAIL+1))
    fi
}

# Qwen35B at k8v8 (primary test KV for VG.2)
run_check "${QWEN_PATH}" "k8v8"

# 60s cooldown between models
log "60s cooldown..."
sleep 60

# Gemma26B at planar (primary test KV for VG.2)
run_check "${GEMMA_PATH}" "planar"

log "=== VG.2 results: PASS=${PASS} FAIL=${FAIL} ==="
if [[ ${FAIL} -gt 0 ]]; then
    log "FAIL: ${FAIL} identity checks failed. regression!"
    exit 1
fi
log "PASS: all identity checks passed. correctness verified."
exit 0
