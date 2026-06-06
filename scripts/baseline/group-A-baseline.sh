#!/usr/bin/env bash
# group-A-baseline.sh — measure rMLX baseline TPS for Group-A regression gate.
#
# 3 models, each at BENCHMARK_CHAMPIONS "rMLX best KV", ctx=4k, max_tokens=20:
#   - Qwen3.6-35B-A3B-8bit   kv=k8v8  (record: ~99.7 TPS decode)
#   - gemma-4-26b-a4b-it-mxfp8 kv=k8v4 (record: ~76.0 TPS decode)
#   - Ternary-Bonsai-8B-mlx-2bit kv=planar (record: ~106.6 TPS decode)
#
# Output: BENCH_RESULT lines on stdout + metrics/group-A-baseline-<TS>.log

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
PROMPT_DIR="${RMLX_ROOT}/prompts"
PROMPT_FILE="${PROMPT_DIR}/longctx_4k.json"
TS="$(date -u +%Y%m%d-%H%M%S)"
LOG="${RMLX_ROOT}/metrics/group-A-baseline-${TS}.log"
mkdir -p "$(dirname "${LOG}")"

log() { echo "[baseline] $*" | tee -a "${LOG}" >&2; }

PAYLOAD_TMP="/tmp/groupA_payload_$$.json"
RESP_TMP="/tmp/groupA_resp_$$.json"

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
        sleep 5; e=$((e+5))
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then return 1; fi
        [[ ${e} -ge 600 ]] && return 1
    done
    log "    health ok in ${e}s"
}

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

bench() {
    local model_path="$1" kv="$2" ctx="$3"
    local model_id; model_id="$(basename "${model_path}")"

    log "=== ${model_id} kv=${kv} ctx=${ctx} ==="
    preflight

    SERVE_LOG="${RMLX_ROOT}/logs/baseline_${model_id}_${TS}.log"
    log "  starting server..."
    "${RMLX_BIN}" serve \
        --model "${model_path}" --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant "${kv}" --max-ctx "${ctx}" \
        > "${SERVE_LOG}" 2>&1 &
    SERVER_PID=$!
    if ! wait_health; then
        log "  ERROR: server didn't start (see ${SERVE_LOG})"
        echo "BENCH_RESULT model=${model_id} kv=${kv} ttft_ms=ERROR decode_tps=ERROR" | tee -a "${LOG}"
        kill "${SERVER_PID}" 2>/dev/null
        return
    fi

    # Prefill-subtracted decode TPS. The naive warm_tok/warm_ms formula is
    # garbage for models whose KV path misses the prompt cache (e.g. Bonsai
    # planar): warm_ms then includes full prefill, diluting decode 9x. K10.
    # Robust method: time max_tokens=1 (n1) and max_tokens=65 (n65); the
    # delta cancels prefill regardless of prompt-cache behaviour.
    #   decode_tps = (n65 - n1) / ((ms65 - ms1) / 1000)
    # ttft is the cold 1-token request (still prefill-dominated => TTFT).

    # Warmup (discard) — primes Metal pipeline + graph compile.
    build_payload "${PROMPT_FILE}" "${model_id}" 1
    fire 600 >/dev/null

    log "  ref decode (max_tokens=1)..."
    build_payload "${PROMPT_FILE}" "${model_id}" 1
    local r1; r1="$(fire 600)"
    local ms1; ms1="$(echo "${r1}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local n1;  n1="$(echo "${r1}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
    local cold_ms="${ms1}"

    # N=257 (not 65): for cache-miss models (Bonsai planar) ms1 contains a
    # multi-second prefill, so (nN-n1)/(msN-ms1) divides by a difference of
    # two large noisy numbers. A 256-token decode sample (~2s+) makes clean
    # decode dominate the ±100ms prefill-timing noise. Models that early-stop
    # (Gemma on this prompt ~20 tok) are inherently small-sample => allow ±8%
    # for Gemma in the regression gate; Qwen/Bonsai are stable at N=257.
    log "  warm decode (max_tokens=257)..."
    build_payload "${PROMPT_FILE}" "${model_id}" 257
    local r65; r65="$(fire 600)"
    local ms65; ms65="$(echo "${r65}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    local n65;  n65="$(echo "${r65}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
    local decode_tps
    decode_tps="$(python3 -c "
ms1=${ms1}; n1=${n1}; ms65=${ms65}; n65=${n65}
dn=n65-n1; dms=ms65-ms1
print(f'{dn/(dms/1000):.2f}' if dn > 0 and dms > 0 else '0.0')
")"
    log "    ms1=${ms1}(n=${n1}) msN=${ms65}(n=${n65}) decode_tps=${decode_tps}"
    echo "BENCH_RESULT model=${model_id} kv=${kv} ctx=${ctx} ttft_ms=${cold_ms} decode_tps=${decode_tps}" \
        | tee -a "${LOG}"

    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null || true
    # Inter-model cooldown. 10s was insufficient: a 35B bench leaves the
    # Metal cache hot, contaminating the next model's first run (observed
    # Qwen 71 vs 97 isolated across B2/B4/B5b — every subtask needed an
    # isolated rerun). 45s lets the GPU/Metal context settle so the
    # first gate run is trustworthy (K10 thermal residual).
    sleep 45
}

# Gemma4-26b uses --max-ctx 8192 because its chat template + 262144-vocab
# tokenizer renders longctx_4k.json to 4121 tokens, overflowing a 4096 cache
# (silent 0-tokens response — see Group-A debugger report 2026-05-13).
# Qwen/Bonsai tokenize the same prompt to 3854/3770 and fit cleanly.
bench "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"   "k8v8"   4096
bench "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8" "k8v4"  8192
bench "${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/prism-ml__Ternary-Bonsai-8B-mlx-2bit"  "planar" 4096

log "DONE. Log at ${LOG}"
