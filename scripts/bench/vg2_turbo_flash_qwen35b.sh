#!/usr/bin/env bash
# VG.2 NIAH surrogate — TurboFlash head_dim=256 correctness gate.
# Compares RMLX_TURBO_FLASH=1 vs RMLX_TURBO_FLASH=0 on Qwen3.6-35B-A3B-8bit
# k8v4 @ 32K context. Greedy first-32-token identity.
#
# Exits 0 if identity passes (kernel correct on this machine), 1 on divergence
# (corruption or numerical drift — DEFER to P2.B).

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
PROMPT_FILE="${RMLX_ROOT}/prompts/longctx_32k.json"
MAX_TOKENS=32

log() { echo "[vg2-tf] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

preflight() {
    log "Pre-flight: kill stale..."
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    pkill -f paroquant   2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local url="http://127.0.0.1:${PORT}/health"
    local max=300 e=0
    until curl -s --max-time 2 "${url}" | grep -q '"ok"'; do
        sleep 3; e=$((e+3))
        [[ ${e} -ge ${max} ]] && die "health timeout"
    done
    log "Server ready in ${e}s"
}

_PAY="/tmp/rmlx_vg2tf_payload_$$.json"
_RES="/tmp/rmlx_vg2tf_resp_$$.json"

get_first32() {
    local model_id="$1"
    python3 - "${PROMPT_FILE}" "${model_id}" "${MAX_TOKENS}" "${_PAY}" <<'PYEOF'
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
        -d "@${_PAY}" \
        -o "${_RES}" 2>/dev/null
    python3 -c "
import json
d = json.load(open('${_RES}'))
choices = d.get('choices', [])
if choices:
    text = choices[0].get('message', {}).get('content', '')
    print(json.dumps(text.split()[:32]))
else:
    print('[]')
" 2>/dev/null || echo '[]'
}

run_one() {
    local flag="$1"
    local model_id; model_id="$(basename "${QWEN_PATH}")"
    local serve_log="/tmp/vg2_tf_${model_id}_${flag}.log"

    preflight
    log "Starting server [TURBO_FLASH=${flag}]..."
    RMLX_TURBO_FLASH="${flag}" "${RMLX_BIN}" serve \
        --model "${QWEN_PATH}" \
        --port "${PORT}" \
        --host 127.0.0.1 \
        --device gpu \
        --kv-quant k8v4 \
        --max-ctx 32768 \
        > "${serve_log}" 2>&1 &
    local pid=$!
    wait_health
    log "Getting first-32 tokens [TURBO_FLASH=${flag}]..."
    local tok
    tok="$(get_first32 "${model_id}")"
    log "TURBO_FLASH=${flag}: ${tok}"
    kill "${pid}" 2>/dev/null
    wait "${pid}" 2>/dev/null || true
    echo "${tok}"
}

log "=== Qwen3.6-35B-A3B-8bit k8v4 32K — TurboFlash head_dim=256 VG.2 ==="

tok_off="$(run_one 0)"
sleep 60
tok_on="$(run_one 1)"

log "OFF: ${tok_off}"
log "ON:  ${tok_on}"

if [[ "${tok_off}" == "${tok_on}" ]]; then
    log "PASS: first-32 tokens identical. Kernel correct on this machine."
    exit 0
else
    log "FAIL: tokens DIVERGE. STOP — defer to P2.B."
    exit 1
fi
