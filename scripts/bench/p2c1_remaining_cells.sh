#!/usr/bin/env bash
# p2c1 continuation — remaining 7 cells:
# 64k_k8v8_K4, 128k_planar_{baseline,K2,K4}, 128k_k8v8_{baseline,K2,K4}
#
# Bash 3.2 compatible.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
PORT=62265
WARMUP_RUNS=1
MEASURE_RUNS=3
MAX_TOKENS=32

VERIFIER_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
DRAFT_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-e2b-it-mxfp8"
VERIFIER_NAME="mlx-community__gemma-4-26b-a4b-it-mxfp8"

PROMPT_64K="${RMLX_ROOT}/prompts/longctx_64k.json"
PROMPT_128K="${RMLX_ROOT}/prompts/longctx_128k.json"

GIT_SHA="$(git -C ${RMLX_ROOT} rev-parse --short HEAD 2>/dev/null || echo unknown)"
RUN_TS="$(date -u +%Y%m%d-%H%M%S)"
LOG_DIR="${RMLX_ROOT}/logs/p2c1_remaining_${RUN_TS}"
mkdir -p "${LOG_DIR}"

log() { echo "[p2c1 $(date -u +%H:%M:%S)] $*" >&2; }

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm      2>/dev/null || true
    pkill -f paroquant   2>/dev/null || true
    pkill -f omlx        2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

wait_health() {
    local max=600 e=0
    until curl -s --max-time 3 "http://127.0.0.1:${PORT}/health" | grep -q '"ok"'; do
        sleep 3; e=$((e+3))
        [[ ${e} -ge ${max} ]] && { log "ERROR: health timeout"; return 1; }
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            log "ERROR: server died"; return 1
        fi
    done
    log "server ready in ${e}s"
}

run_streaming_request() {
    local prompt_file="$1"
    local payload="/tmp/p2c1r_payload_$$.json"
    local sse_out="/tmp/p2c1r_sse_$$.txt"

    python3 - "${prompt_file}" "${VERIFIER_NAME}" "${MAX_TOKENS}" "${payload}" <<'PYEOF'
import json, sys
prompt_file, model, mt, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
pf = json.load(open(prompt_file))
p = {'model': model, 'messages': pf['messages'], 'max_tokens': mt,
     'temperature': 0.0, 'stream': True}
json.dump(p, open(out, 'w'))
PYEOF

    python3 - "${payload}" "${PORT}" "${sse_out}" <<'PYEOF'
import json, sys, time, urllib.request
payload_file, port, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
with open(payload_file, 'rb') as f:
    payload_bytes = f.read()
req = urllib.request.Request(url, data=payload_bytes,
    headers={'Content-Type': 'application/json'})
records = []
try:
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw_line in resp:
            t_now = time.time()
            line = raw_line.decode('utf-8', errors='replace').rstrip('\n\r')
            records.append(f"{t_now:.6f} {line}")
except Exception as e:
    records.append(f"{time.time():.6f} ERROR: {e}")
with open(out_file, 'w') as f:
    f.write('\n'.join(records) + '\n')
PYEOF

    python3 - "${sse_out}" <<'PYEOF'
import json, sys, re
sse_file = sys.argv[1]
lines = open(sse_file).readlines()
token_times = []
total_tokens = 0
for raw in lines:
    raw = raw.rstrip('\n')
    if not raw.strip():
        continue
    m = re.match(r'^([0-9.]+) data: (.*)$', raw)
    if not m:
        continue
    ts = float(m.group(1))
    data_str = m.group(2).strip()
    if data_str == '[DONE]':
        token_times.append(ts)
        break
    try:
        chunk = json.loads(data_str)
        delta = chunk.get('choices', [{}])[0].get('delta', {})
        content = delta.get('content', '')
        if content:
            token_times.append(ts)
            total_tokens += 1
    except Exception:
        pass
if len(token_times) < 2:
    print(f"0,0")
else:
    t_first = token_times[0]
    t_last = token_times[-1]
    decode_elapsed = t_last - t_first
    if decode_elapsed > 0 and total_tokens > 1:
        decode_tps = (total_tokens - 1) / decode_elapsed
    else:
        decode_tps = 1.0
    print(f"{decode_tps:.2f},{total_tokens}")
PYEOF
    rm -f "${payload}" "${sse_out}"
}

extract_accept_rate_from_log() {
    local logfile="$1"
    python3 - "${logfile}" <<'PYEOF'
import re, json, sys
logfile = sys.argv[1]
# Extract all round data: accept=N num_draft=N
rounds = []
try:
    with open(logfile) as f:
        for line in f:
            m = re.search(r'accept=([0-9]+).*?num_draft=([0-9]+)', line)
            if m:
                rounds.append((int(m.group(1)), int(m.group(2))))
except Exception:
    pass
if rounds:
    total_accept = sum(r[0] for r in rounds)
    total_draft = sum(r[1] for r in rounds)
    if total_draft > 0:
        print(f"{total_accept/total_draft*100:.1f}%")
    else:
        print("n/a")
else:
    print("n/a")
PYEOF
}

run_cell() {
    local ctx_label="$1"
    local kv_quant="$2"
    local max_ctx="$3"
    local spec_k="$4"
    local prompt_file="$5"

    local mode_label
    if [[ "${spec_k}" == "0" ]]; then
        mode_label="baseline"
    else
        mode_label="K${spec_k}"
    fi
    local cell_id="${ctx_label}_${kv_quant}_${mode_label}"
    log "=== Cell ${cell_id} ==="

    preflight

    SERVE_LOG="${LOG_DIR}/serve_${cell_id}.log"

    if [[ "${spec_k}" == "0" ]]; then
        "${RMLX_BIN}" serve \
            --model "${VERIFIER_PATH}" \
            --port "${PORT}" --host 127.0.0.1 \
            --device gpu --kv-quant "${kv_quant}" --max-ctx "${max_ctx}" \
            > "${SERVE_LOG}" 2>&1 &
    else
        RMLX_SPEC_K="${spec_k}" "${RMLX_BIN}" serve \
            --model "${VERIFIER_PATH}" \
            --draft-model "${DRAFT_PATH}" \
            --port "${PORT}" --host 127.0.0.1 \
            --device gpu --kv-quant "${kv_quant}" --max-ctx "${max_ctx}" \
            > "${SERVE_LOG}" 2>&1 &
    fi
    SERVER_PID=$!

    if ! wait_health; then
        log "ERROR: ${cell_id} health failed"
        kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null || true
        echo "TIMEOUT,n/a"
        return
    fi

    log "  warmup 1/1..."
    run_streaming_request "${prompt_file}" >/dev/null

    local tps1="0" tps2="0" tps3="0"
    for i in $(seq 1 "${MEASURE_RUNS}"); do
        local result
        result="$(run_streaming_request "${prompt_file}")"
        local tps toks
        tps="$(echo "${result}" | cut -d, -f1)"
        toks="$(echo "${result}" | cut -d, -f2)"
        log "  measure ${i}/${MEASURE_RUNS}: decode_tps=${tps} (${toks} tokens)"
        case ${i} in 1) tps1="${tps}" ;; 2) tps2="${tps}" ;; 3) tps3="${tps}" ;; esac
    done

    local median
    median="$(python3 -c "
v = sorted([${tps1}, ${tps2}, ${tps3}])
print(f'{v[1]:.2f}')
")"
    log "  median decode TPS: ${median}"

    local accept_rate="n/a"
    if [[ "${spec_k}" != "0" ]]; then
        accept_rate="$(extract_accept_rate_from_log "${SERVE_LOG}")"
    fi
    log "  accept_rate: ${accept_rate}"

    kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null || true
    echo "${median},${accept_rate}"
}

# ========================
# Cell list (remaining 7)
# ========================

echo "[p2c1-remaining] Starting remaining 7 cells" >&2

# Cell 1: 64k_k8v8_K4
log "=== REMAINING CELL: 64k_k8v8_K4 ==="
r="$(run_cell 64k k8v8 131072 4 "${PROMPT_64K}")"
tps_64k_k8v8_K4="$(echo "${r}" | cut -d, -f1)"
ar_64k_k8v8_K4="$(echo "${r}" | cut -d, -f2)"
log "64k_k8v8_K4 = ${tps_64k_k8v8_K4} TPS, accept=${ar_64k_k8v8_K4}"
sleep 60

# Cell 2: 128k_planar_baseline
log "=== REMAINING CELL: 128k_planar_baseline ==="
r="$(run_cell 128k planar 196608 0 "${PROMPT_128K}")"
tps_128k_planar_base="$(echo "${r}" | cut -d, -f1)"
log "128k_planar_baseline = ${tps_128k_planar_base} TPS"
sleep 60

# Cell 3: 128k_planar_K2
log "=== REMAINING CELL: 128k_planar_K2 ==="
r="$(run_cell 128k planar 196608 2 "${PROMPT_128K}")"
tps_128k_planar_K2="$(echo "${r}" | cut -d, -f1)"
ar_128k_planar_K2="$(echo "${r}" | cut -d, -f2)"
log "128k_planar_K2 = ${tps_128k_planar_K2} TPS, accept=${ar_128k_planar_K2}"
sleep 60

# Cell 4: 128k_planar_K4
log "=== REMAINING CELL: 128k_planar_K4 ==="
r="$(run_cell 128k planar 196608 4 "${PROMPT_128K}")"
tps_128k_planar_K4="$(echo "${r}" | cut -d, -f1)"
ar_128k_planar_K4="$(echo "${r}" | cut -d, -f2)"
log "128k_planar_K4 = ${tps_128k_planar_K4} TPS, accept=${ar_128k_planar_K4}"
sleep 60

# Cell 5: 128k_k8v8_baseline
log "=== REMAINING CELL: 128k_k8v8_baseline ==="
r="$(run_cell 128k k8v8 196608 0 "${PROMPT_128K}")"
tps_128k_k8v8_base="$(echo "${r}" | cut -d, -f1)"
log "128k_k8v8_baseline = ${tps_128k_k8v8_base} TPS"
sleep 60

# Cell 6: 128k_k8v8_K2
log "=== REMAINING CELL: 128k_k8v8_K2 ==="
r="$(run_cell 128k k8v8 196608 2 "${PROMPT_128K}")"
tps_128k_k8v8_K2="$(echo "${r}" | cut -d, -f1)"
ar_128k_k8v8_K2="$(echo "${r}" | cut -d, -f2)"
log "128k_k8v8_K2 = ${tps_128k_k8v8_K2} TPS, accept=${ar_128k_k8v8_K2}"
sleep 60

# Cell 7: 128k_k8v8_K4
log "=== REMAINING CELL: 128k_k8v8_K4 ==="
r="$(run_cell 128k k8v8 196608 4 "${PROMPT_128K}")"
tps_128k_k8v8_K4="$(echo "${r}" | cut -d, -f1)"
ar_128k_k8v8_K4="$(echo "${r}" | cut -d, -f2)"
log "128k_k8v8_K4 = ${tps_128k_k8v8_K4} TPS, accept=${ar_128k_k8v8_K4}"

# ========================
# Print remaining results
# ========================
echo ""
echo "=== REMAINING CELLS SUMMARY ==="
echo "64k_k8v8_K4: ${tps_64k_k8v8_K4} TPS, accept=${ar_64k_k8v8_K4}"
echo "128k_planar_baseline: ${tps_128k_planar_base} TPS"
echo "128k_planar_K2: ${tps_128k_planar_K2} TPS, accept=${ar_128k_planar_K2}"
echo "128k_planar_K4: ${tps_128k_planar_K4} TPS, accept=${ar_128k_planar_K4}"
echo "128k_k8v8_baseline: ${tps_128k_k8v8_base} TPS"
echo "128k_k8v8_K2: ${tps_128k_k8v8_K2} TPS, accept=${ar_128k_k8v8_K2}"
echo "128k_k8v8_K4: ${tps_128k_k8v8_K4} TPS, accept=${ar_128k_k8v8_K4}"
echo "Log dir: ${LOG_DIR}"
