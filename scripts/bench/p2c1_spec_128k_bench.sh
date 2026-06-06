#!/usr/bin/env bash
# p2c1 — Speculative decoding at 64K + 128K ctx bench.
# Verifier: mlx-community__gemma-4-26b-a4b-it-mxfp8
# Draft:    mlx-community__gemma-4-e2b-it-mxfp8
#
# 12 cells: {baseline, K=2, K=4} x {64K, 128K} x {planar, k8v8}
# Sequential, 1 warmup + 3 measure per cell, 60s cooldown between K sweeps.
#
# Context depths:
#   64K:  longctx_64k.json  (~68,902 tokens after template), max_ctx=131072
#   128K: longctx_128k.json (~137,924 tokens after template), max_ctx=196608
#
# Decode TPS measured via streaming SSE: time between first and last token.
# Spec path does NOT use prompt cache, so every request pays full prefill.
# Streaming is the only way to isolate decode TPS from prefill.
#
# K config: RMLX_SPEC_K env var (2 or 4), default 4.
#
# Usage: ./scripts/bench/p2c1_spec_128k_bench.sh
# Bash 3.2 compatible (no declare -A).

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
LOG_DIR="${RMLX_ROOT}/logs/p2c1_spec_128k_${RUN_TS}"
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

# run_streaming_request <prompt_file>
# Measures decode TPS via SSE streaming: time from first SSE data to last.
# Prints: decode_tps,completion_tokens,accept_rate_from_log
run_streaming_request() {
    local prompt_file="$1"
    local payload="/tmp/p2c1_payload_$$.json"
    local sse_out="/tmp/p2c1_sse_$$.txt"

    python3 - "${prompt_file}" "${VERIFIER_NAME}" "${MAX_TOKENS}" "${payload}" <<'PYEOF'
import json, sys
prompt_file, model, mt, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
pf = json.load(open(prompt_file))
p = {'model': model, 'messages': pf['messages'], 'max_tokens': mt,
     'temperature': 0.0, 'stream': True}
json.dump(p, open(out, 'w'))
PYEOF

    # Stream the response, timestamp each SSE data line.
    # Format: "<unix_ms_float> data: ..." or just the SSE lines.
    # We use python for streaming to handle SSE properly.
    python3 - "${payload}" "${PORT}" "${sse_out}" <<'PYEOF'
import json, sys, time, urllib.request, urllib.error

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

    # Parse SSE output to get decode TPS.
    python3 - "${sse_out}" <<'PYEOF'
import json, sys, re

sse_file = sys.argv[1]
lines = open(sse_file).readlines()

token_times = []   # timestamps of each data line with content
total_tokens = 0
error_seen = False

for raw in lines:
    raw = raw.rstrip('\n')
    if not raw.strip():
        continue
    # Format: "<ts> data: ..."
    m = re.match(r'^([0-9.]+) data: (.*)$', raw)
    if not m:
        continue
    ts = float(m.group(1))
    data_str = m.group(2).strip()
    if data_str == '[DONE]':
        token_times.append(ts)  # final boundary
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
    elif total_tokens == 1:
        # Single token — use rough estimate
        decode_tps = 1.0
    else:
        decode_tps = 0
    print(f"{decode_tps:.2f},{total_tokens}")
PYEOF

    rm -f "${payload}" "${sse_out}"
}

# Extract last accept_rate from server log (tracing text or JSON).
extract_accept_rate() {
    local logfile="$1"
    python3 - "${logfile}" <<'PYEOF'
import re, json, sys
logfile = sys.argv[1]
rate = None
try:
    with open(logfile) as f:
        for line in f:
            line = line.strip()
            try:
                obj = json.loads(line)
                fields = obj.get('fields', {})
                if 'accept_rate' in fields:
                    rate = float(fields['accept_rate'])
            except Exception:
                pass
            m = re.search(r'accept_rate=([0-9.]+)', line)
            if m:
                rate = float(m.group(1))
except Exception:
    pass
if rate is not None:
    print(f"{rate*100:.1f}%")
else:
    print("n/a")
PYEOF
}

# run_cell <ctx_label> <kv_quant> <max_ctx> <spec_k> <prompt_file>
# Prints: median_decode_tps,accept_rate
run_cell() {
    local ctx_label="$1"
    local kv_quant="$2"
    local max_ctx="$3"
    local spec_k="$4"  # 0=baseline, 2, or 4
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
        log "Starting baseline server (kv=${kv_quant}, max-ctx=${max_ctx})..."
        "${RMLX_BIN}" serve \
            --model "${VERIFIER_PATH}" \
            --port "${PORT}" --host 127.0.0.1 \
            --device gpu --kv-quant "${kv_quant}" --max-ctx "${max_ctx}" \
            > "${SERVE_LOG}" 2>&1 &
    else
        log "Starting spec server (kv=${kv_quant}, max-ctx=${max_ctx}, K=${spec_k})..."
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

    # Warmup (spec has no prompt cache; warmup warms Metal compile caches).
    log "  warmup 1/${WARMUP_RUNS}..."
    run_streaming_request "${prompt_file}" >/dev/null

    # Measure runs.
    local tps_sum=0
    local tps1="0" tps2="0" tps3="0"
    for i in $(seq 1 "${MEASURE_RUNS}"); do
        local result
        result="$(run_streaming_request "${prompt_file}")"
        local tps
        tps="$(echo "${result}" | cut -d, -f1)"
        local toks
        toks="$(echo "${result}" | cut -d, -f2)"
        log "  measure ${i}/${MEASURE_RUNS}: decode_tps=${tps} (${toks} tokens)"
        case ${i} in 1) tps1="${tps}" ;; 2) tps2="${tps}" ;; 3) tps3="${tps}" ;; esac
    done

    # Median.
    local median
    median="$(python3 -c "
v = sorted([${tps1}, ${tps2}, ${tps3}])
print(f'{v[1]:.2f}')
")"
    log "  median decode TPS: ${median}"

    # Acceptance rate (spec only).
    local accept_rate="n/a"
    if [[ "${spec_k}" != "0" ]]; then
        accept_rate="$(extract_accept_rate "${SERVE_LOG}")"
    fi
    log "  accept_rate: ${accept_rate}"

    kill "${SERVER_PID}" 2>/dev/null; wait "${SERVER_PID}" 2>/dev/null || true
    echo "${median},${accept_rate}"
}

# ========================
# Main sweep
# ========================
# Cells: ctx_label|kv_quant|max_ctx|prompt_file
CTX_PAIRS=(
    "64k|planar|131072|${PROMPT_64K}"
    "64k|k8v8|131072|${PROMPT_64K}"
    "128k|planar|196608|${PROMPT_128K}"
    "128k|k8v8|196608|${PROMPT_128K}"
)

# Store results as flat indexed arrays (bash 3.2 compatible).
RESULT_KEYS=()
RESULT_BASE=()
RESULT_K2=()
RESULT_K4=()
RESULT_ACCEPT_K2=()
RESULT_ACCEPT_K4=()

for pair in "${CTX_PAIRS[@]}"; do
    ctx_label="$(echo "${pair}" | cut -d'|' -f1)"
    kv_quant="$(echo "${pair}" | cut -d'|' -f2)"
    max_ctx="$(echo "${pair}" | cut -d'|' -f3)"
    prompt_file="$(echo "${pair}" | cut -d'|' -f4)"
    cell_key="${ctx_label}_${kv_quant}"

    log "===== Sweep: ctx=${ctx_label} kv=${kv_quant} max_ctx=${max_ctx} ====="

    r_base="$(run_cell "${ctx_label}" "${kv_quant}" "${max_ctx}" 0 "${prompt_file}")"
    tps_base="$(echo "${r_base}" | cut -d, -f1)"
    log "  BASELINE: ${tps_base} TPS"
    sleep 60

    r_k2="$(run_cell "${ctx_label}" "${kv_quant}" "${max_ctx}" 2 "${prompt_file}")"
    tps_k2="$(echo "${r_k2}" | cut -d, -f1)"
    ar_k2="$(echo "${r_k2}" | cut -d, -f2)"
    log "  K=2: ${tps_k2} TPS, accept=${ar_k2}"
    sleep 60

    r_k4="$(run_cell "${ctx_label}" "${kv_quant}" "${max_ctx}" 4 "${prompt_file}")"
    tps_k4="$(echo "${r_k4}" | cut -d, -f1)"
    ar_k4="$(echo "${r_k4}" | cut -d, -f2)"
    log "  K=4: ${tps_k4} TPS, accept=${ar_k4}"
    sleep 60

    RESULT_KEYS+=("${cell_key}")
    RESULT_BASE+=("${tps_base}")
    RESULT_K2+=("${tps_k2}")
    RESULT_K4+=("${tps_k4}")
    RESULT_ACCEPT_K2+=("${ar_k2}")
    RESULT_ACCEPT_K4+=("${ar_k4}")
done

# ========================
# Summary table
# ========================
echo ""
echo "======================================================================"
echo "P2.C Spec decode 64K + 128K bench — gemma-4-26b-a4b-it-mxfp8"
echo "git_sha=${GIT_SHA}, ts=${RUN_TS}"
echo "Draft: gemma-4-e2b-it-mxfp8 | K=2, K=4"
echo "Metric: decode TPS (stream first-to-last token, excludes prefill)"
echo "======================================================================"
printf "%-18s | %-9s | %-9s | %-9s | %-9s | %-8s | %-8s\n" \
    "cell" "base TPS" "K2 TPS" "K4 TPS" "K2 delta" "K2 acc" "K4 acc"
echo "-------------------+-----------+-----------+-----------+-----------+----------+----------"

n="${#RESULT_KEYS[@]}"
for i in $(seq 0 $((n - 1))); do
    key="${RESULT_KEYS[$i]}"
    base="${RESULT_BASE[$i]}"
    k2="${RESULT_K2[$i]}"
    k4="${RESULT_K4[$i]}"
    ar2="${RESULT_ACCEPT_K2[$i]}"
    ar4="${RESULT_ACCEPT_K4[$i]}"

    delta_k2="$(python3 -c "
base=float('${base}') if '${base}' not in ('TIMEOUT','0') else 0
k2=float('${k2}') if '${k2}' not in ('TIMEOUT','0') else 0
if base > 0 and k2 > 0:
    print(f'{(k2 - base) / base * 100:+.1f}%')
else:
    print('n/a')
" 2>/dev/null || echo n/a)"

    printf "%-18s | %-9s | %-9s | %-9s | %-9s | %-8s | %-8s\n" \
        "${key}" "${base}" "${k2}" "${k4}" "${delta_k2}" "${ar2}" "${ar4}"
done

echo "======================================================================"
# Verdict
any_win=0
for i in $(seq 0 $((n - 1))); do
    key="${RESULT_KEYS[$i]}"
    base="${RESULT_BASE[$i]}"
    k2="${RESULT_K2[$i]}"
    if echo "${key}" | grep -q "128k"; then
        win="$(python3 -c "
base=float('${base}') if '${base}' not in ('TIMEOUT','0') else 0
k2=float('${k2}') if '${k2}' not in ('TIMEOUT','0') else 0
if base > 0 and k2 > 0 and (k2 - base) / base * 100 > 5:
    print('YES')
else:
    print('NO')
" 2>/dev/null || echo NO)"
        if [[ "${win}" == "YES" ]]; then
            any_win=1
            log "WIN: ${key} K=2 > baseline +5%"
        fi
    fi
done
if [[ ${any_win} -eq 1 ]]; then
    echo "VERDICT: SPEC WIN at 128K — K=2 exceeds baseline +5% threshold."
else
    echo "VERDICT: NO WIN — spec does not exceed baseline +5% at 128K ctx."
fi
echo "======================================================================"
echo "Log dir: ${LOG_DIR}"
