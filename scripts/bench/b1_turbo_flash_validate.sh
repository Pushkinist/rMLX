#!/usr/bin/env bash
# B1 — TurboFlash M5 validation + default-ON decision (2026-05-18).
#
# NOT a bisect. TheTom commit 67f076f2e is a 4-line default-flip, not a kernel
# fix — no upstream TurboFlash corruption fix exists. B1 empirically validates
# rMLX's DISTINCT kernel (q8_0 K + 4-bit Lloyd-Max V, no WHT) on THIS M5 Max,
# then makes a data-driven default-ON-or-OFF decision.
#
# For each of {Qwen3.6-35B-A3B-8bit (head_dim 256), Bonsai-8B-2bit (head_dim
# 128)} at --kv-quant k8v4 --max-ctx 32768, prompt prompts/longctx_32k.json,
# greedy (temperature 0):
#   1. Capture first-32 tokens with RMLX_TURBO_FLASH=0 (baseline =
#      mixed_quantized_sdpa) vs RMLX_TURBO_FLASH=1 (TurboFlash). Compare
#      bit-exact. Grep serve log for trip-wire warn.
#   2. If bit-exact, measure prefill-subtracted decode TPS at 32k ctx for
#      both TF on/off:  decode_tps = (n65 - n1) / ((ms65 - ms1) / 1000).
#
# Verdict per cell: CORRUPT (diverge / trip-wire / garbage) or CLEAN, plus a
# TPS delta when CLEAN. The Rust gate / module docs are updated by the
# executor only if >=+5% bit-exact gain is observed on at least one cell.
#
# Output: B1_RESULT lines on stdout + /tmp/b1_tf_*.log serve logs.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
PROMPT_FILE="${RMLX_ROOT}/prompts/longctx_32k.json"
TRIPWIRE='TurboFlash: corruption detected'
KERNERR='TurboFlash: kernel error'

QWEN_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
BONSAI_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/prism-ml__Ternary-Bonsai-8B-mlx-2bit"

log() { echo "[b1-tf] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    pkill -f paroquant   2>/dev/null || true
    pkill -f omlx        2>/dev/null || true
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

_PAY="/tmp/b1_tf_payload_$$.json"
_RES="/tmp/b1_tf_resp_$$.json"

build_payload() {
    local model_id="$1" max_tokens="$2"
    python3 - "${PROMPT_FILE}" "${model_id}" "${max_tokens}" "${_PAY}" <<'PYEOF'
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

# Fire one request; echo "elapsed_ms=<n> tokens=<n>".
fire() {
    local timeout_s="$1"
    local t0; t0="$(python3 -c 'import time;print(int(time.time()*1000))')"
    curl -s --max-time "${timeout_s}" -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "@${_PAY}" -o "${_RES}" 2>/dev/null
    local t1; t1="$(python3 -c 'import time;print(int(time.time()*1000))')"
    local tok
    tok="$(python3 -c "
import json
try:
    d = json.load(open('${_RES}'))
    print(d.get('usage', {}).get('completion_tokens', 0))
except Exception:
    print(0)
" 2>/dev/null || echo 0)"
    echo "elapsed_ms=$((t1 - t0)) tokens=${tok}"
}

# Extract the first 32 whitespace-split tokens of the completion text.
# Reasoning models (Qwen3.5 MoE) emit greedy output into reasoning_content
# with content="" — the bit-exact identity must compare whichever field
# carries the decoded stream, so concatenate reasoning_content + content.
first32() {
    python3 -c "
import json
try:
    d = json.load(open('${_RES}'))
    c = d.get('choices', [])
    m = c[0].get('message', {}) if c else {}
    t = (m.get('reasoning_content') or '') + (m.get('content') or '')
    print(json.dumps(t.split()[:32]))
except Exception:
    print('[]')
" 2>/dev/null || echo '[]'
}

# Run one (model, mode) cell. Globals set: CELL_TOK, CELL_DTPS, CELL_TRIP.
run_cell() {
    local model_path="$1" mode="$2"   # mode = OFF | ON
    local model_id; model_id="$(basename "${model_path}")"
    local serve_log="/tmp/b1_tf_${model_id}_${mode}.log"
    local tf
    if [[ "${mode}" == "OFF" ]]; then tf=0; else tf=1; fi

    preflight
    log "=== ${model_id} k8v4 32k  TF=${tf} (${mode}) ==="
    RMLX_TURBO_FLASH=${tf} "${RMLX_BIN}" serve \
        --model "${model_path}" --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant k8v4 --max-ctx 32768 \
        > "${serve_log}" 2>&1 &
    SERVER_PID=$!
    if ! wait_health; then
        die "${model_id} ${mode}: server didn't start (see ${serve_log})"
    fi

    # Warmup (discard) — primes Metal pipeline + graph compile + flash buffers.
    build_payload "${model_id}" 1
    fire 900 >/dev/null

    # first-32 identity capture (max_tokens=32, greedy).
    build_payload "${model_id}" 32
    fire 900 >/dev/null
    CELL_TOK="$(first32)"

    # Prefill-subtracted decode TPS: n1 vs n65.
    build_payload "${model_id}" 1
    local r1; r1="$(fire 900)"
    local ms1; ms1="$(echo "${r1}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local n1;  n1="$(echo "${r1}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
    build_payload "${model_id}" 65
    local r65; r65="$(fire 900)"
    local ms65; ms65="$(echo "${r65}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local n65;  n65="$(echo "${r65}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
    CELL_DTPS="$(python3 -c "
ms1=${ms1}; n1=${n1}; ms65=${ms65}; n65=${n65}
dn=n65-n1; dms=ms65-ms1
print(f'{dn/(dms/1000):.2f}' if dn > 0 and dms > 0 else '0.0')
")"

    if grep -qF "${TRIPWIRE}" "${serve_log}"; then
        CELL_TRIP="YES"
    else
        CELL_TRIP="no"
    fi
    if grep -qF "${KERNERR}" "${serve_log}"; then
        CELL_KERNERR="YES"
    else
        CELL_KERNERR="no"
    fi

    log "    ${mode}: tok=${CELL_TOK}"
    log "    ${mode}: ms1=${ms1}(n=${n1}) ms65=${ms65}(n=${n65}) decode_tps=${CELL_DTPS} tripwire=${CELL_TRIP} kernel_error=${CELL_KERNERR}"

    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null || true
    sleep 10
}

run_model() {
    local model_path="$1"
    local model_id; model_id="$(basename "${model_path}")"

    run_cell "${model_path}" OFF
    local off_tok="${CELL_TOK}" off_dtps="${CELL_DTPS}" off_trip="${CELL_TRIP}"
    run_cell "${model_path}" ON
    local on_tok="${CELL_TOK}" on_dtps="${CELL_DTPS}" on_trip="${CELL_TRIP}" on_kernerr="${CELL_KERNERR}"

    local bitexact verdict gain_pct
    if [[ "${off_tok}" == "${on_tok}" ]]; then
        bitexact="BIT_EXACT"
    else
        bitexact="DIVERGE"
    fi
    gain_pct="$(python3 -c "
off=${off_dtps:-0}; on=${on_dtps:-0}
print(f'{(on-off)/off*100:.2f}' if off > 0 else 'NA')
")"
    if [[ "${bitexact}" == "DIVERGE" || "${on_trip}" == "YES" ]]; then
        verdict="CORRUPT"
    else
        verdict="CLEAN"
    fi

    echo "B1_RESULT model=${model_id} kv=k8v4 ctx=32768 ${bitexact} verdict=${verdict} tripwire=${on_trip} kernel_error=${on_kernerr} off_dtps=${off_dtps} on_dtps=${on_dtps} gain_pct=${gain_pct}"
    echo "B1_TOK_OFF model=${model_id} ${off_tok}"
    echo "B1_TOK_ON  model=${model_id} ${on_tok}"
}

log "########## B1 TurboFlash M5 validation (k8v4 @ 32k) ##########"
run_model "${QWEN_PATH}"
sleep 30
run_model "${BONSAI_PATH}"
preflight
log "########## B1 DONE ##########"
