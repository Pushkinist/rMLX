#!/usr/bin/env bash
# p0b_vg2_niah.sh — VG.2 NIAH-style identity check across PREFILL_CHUNK values.
#
# Asserts that greedy first-32 tokens at 32K are IDENTICAL between
# RMLX_PREFILL_CHUNK=64 (legacy) and RMLX_PREFILL_CHUNK=<new-default>
# (post-tuning). Determinism gate — chunk size MUST NOT affect output.
#
# Models:
#   - Qwen3.6-35B-A3B-8bit  (qwen3_5_moe, k8v8): 64 vs 256
#   - gemma-4-26b-a4b-it-mxfp8 (gemma4, planar): 64 vs 512
#
# Exits 0 on PASS-all, 1 on any FAIL.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
GEMMA_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
PROMPT_FILE="${RMLX_ROOT}/prompts/longctx_32k.json"
MAX_TOKENS=32
PASS=0
FAIL=0

log() { echo "[p0b-vg2] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local max=300 e=0
    until curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; do
        sleep 3; e=$((e+3))
        [[ ${e} -ge ${max} ]] && return 1
    done
    log "  health ok in ${e}s"
}

PAYLOAD_TMP="/tmp/p0b_vg2_payload_$$.json"
RESP_TMP="/tmp/p0b_vg2_resp_$$.json"

get_first32() {
    local model_id="$1"
    python3 - "${PROMPT_FILE}" "${model_id}" "${MAX_TOKENS}" "${PAYLOAD_TMP}" <<'PYEOF'
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
    curl -s --max-time 900 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "@${PAYLOAD_TMP}" \
        -o "${RESP_TMP}" 2>/dev/null
    python3 -c "
import json
try:
    d = json.load(open('${RESP_TMP}'))
    choices = d.get('choices', [])
    if choices:
        text = choices[0].get('message', {}).get('content', '')
        print(json.dumps(text.split()[:32]))
    else:
        print('[]')
except Exception:
    print('[]')
" 2>/dev/null || echo '[]'
}

run_check() {
    local model_path="$1"
    local kv="$2"
    local pre_chunk="$3"   # legacy
    local post_chunk="$4"  # tuned default

    local model_id; model_id="$(basename "${model_path}")"
    local serve_log_pre  serve_log_post
    serve_log_pre="/tmp/p0b_vg2_${model_id}_${kv}_pre.log"
    serve_log_post="/tmp/p0b_vg2_${model_id}_${kv}_post.log"

    log "=== ${model_id} kv=${kv} chunk ${pre_chunk} vs ${post_chunk} @ 32K ==="

    # PRE
    preflight
    log "  starting [chunk=${pre_chunk}]..."
    RMLX_PREFILL_CHUNK="${pre_chunk}" "${RMLX_BIN}" serve \
        --model "${model_path}" --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant "${kv}" --max-ctx 32768 \
        > "${serve_log_pre}" 2>&1 &
    local pid_pre=$!
    if ! wait_health; then
        log "  FAIL: server didn't start (PRE)"
        kill "${pid_pre}" 2>/dev/null
        FAIL=$((FAIL+1))
        return
    fi
    log "  fetching first-32 [chunk=${pre_chunk}]..."
    local tok_pre; tok_pre="$(get_first32 "${model_id}")"
    log "  PRE:  ${tok_pre}"
    kill "${pid_pre}" 2>/dev/null; wait "${pid_pre}" 2>/dev/null || true

    sleep 30

    # POST
    preflight
    log "  starting [chunk=${post_chunk}]..."
    RMLX_PREFILL_CHUNK="${post_chunk}" "${RMLX_BIN}" serve \
        --model "${model_path}" --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant "${kv}" --max-ctx 32768 \
        > "${serve_log_post}" 2>&1 &
    local pid_post=$!
    if ! wait_health; then
        log "  FAIL: server didn't start (POST)"
        kill "${pid_post}" 2>/dev/null
        FAIL=$((FAIL+1))
        return
    fi
    log "  fetching first-32 [chunk=${post_chunk}]..."
    local tok_post; tok_post="$(get_first32 "${model_id}")"
    log "  POST: ${tok_post}"
    kill "${pid_post}" 2>/dev/null; wait "${pid_post}" 2>/dev/null || true

    if [[ "${tok_pre}" == "${tok_post}" ]]; then
        log "  PASS: identity preserved"
        PASS=$((PASS+1))
    else
        log "  FAIL: tokens differ"
        FAIL=$((FAIL+1))
    fi
}

# Qwen3.5MoE: 64 → 256
run_check "${QWEN_PATH}" "k8v8" 64 256

# 60s inter-model cooldown
log "60s cooldown..."
sleep 60

# Gemma4-26B: 64 → 512
run_check "${GEMMA_PATH}" "planar" 64 512

log "=== p0b-vg2 results: PASS=${PASS} FAIL=${FAIL} ==="
[[ ${FAIL} -gt 0 ]] && exit 1
exit 0
