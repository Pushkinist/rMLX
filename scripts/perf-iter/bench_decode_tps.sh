#!/usr/bin/env bash
# bench_decode_tps.sh — per-iteration regression bench for the perf-fix campaign.
#
# Usage:
#   MODEL_PATH=/path/to/model KV_QUANT=k8v8 ./scripts/perf-iter/bench_decode_tps.sh
#   MODEL_PATH=... KV_QUANT=k8v4 PORT=62265 ./scripts/perf-iter/bench_decode_tps.sh
#   # Override repeat count via flag (takes precedence over MEASURE_RUNS env var):
#   MODEL_PATH=... KV_QUANT=k8v8 ./scripts/perf-iter/bench_decode_tps.sh --repeat 5
#
# Env vars:
#   MODEL_PATH   — absolute path to an MLX model snapshot directory (required)
#   KV_QUANT     — kv-quant flag for rmlx serve (required; e.g. k8v8, k8v4, planar)
#   PORT         — TCP port for rmlx serve (default: 62265)
#   MAX_CTX      — max context tokens (default: 8192)
#   WARMUP_RUNS  — warmup completions before measurement (default: 1)
#   MEASURE_RUNS — completions to measure (default: 3); overridden by --repeat N
#   RMLX_BIN    — path to rmlx binary (default: ./target/release/rmlx)
#   METRICS_OUT  — JSONL file to append to
#                  (default: <RMLX_HOME>/metrics/perf-iter/baseline.jsonl)
#
# Flags:
#   --repeat N   — override MEASURE_RUNS (useful for stable per-finding numbers;
#                  perf-book ch 2 recommends N>=5 before committing a finding)
#
# CLAUDE.md mandatory pre-flight (single-MLX-process rule):
#   pkill -f "rmlx serve"; pkill -f mlx_lm; pkill -f paroquant; pkill -f omlx;
#   sleep 5; rm -f /tmp/rmlx.<port>.claim
#
# Measurement basis:
#   decode_tps is the rate the server measured for each request over that
#   request's own inter-token gaps — first token to last, prefill excluded —
#   read back through scripts/lib/server_decode_tps.py. Dividing the completion
#   tokens by the whole request would count the prefill and the connection and
#   is a different quantity (overall_tps), not this one.
#
# Output:
#   Prints decode_tps_mean / decode_tps_stddev / ttft_ms per run to stdout.
#   Appends one JSONL line per measurement call to $METRICS_OUT.
#   Also writes a per-session summary line to METRICS_OUT after all runs.

set -euo pipefail

# ── Flag parsing (before env-var defaults) ────────────────────────────────────
# Only --repeat N is supported as a CLI flag; everything else is env-var only.

_REPEAT_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --repeat)
            [[ $# -ge 2 ]] || { echo "ERROR: --repeat requires a numeric argument" >&2; exit 1; }
            _REPEAT_OVERRIDE="$2"
            shift 2
            ;;
        --repeat=*)
            _REPEAT_OVERRIDE="${1#*=}"
            shift
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH="${MODEL_PATH:?MODEL_PATH env var is required}"
KV_QUANT="${KV_QUANT:?KV_QUANT env var is required (e.g. k8v8, k8v4, planar)}"
PORT="${PORT:-62265}"
# All on-disk state lives under one root (CLAUDE.md, "Runtime data root"), so
# nothing here names `metrics` relative to the working directory — that is how
# a run from a sub-directory leaves a stray tree behind.
RMLX_HOME="${RMLX_HOME:-$PWD/.rmlx}"
RMLX_METRICS_DIR="${RMLX_HOME}/metrics"
MAX_CTX="${MAX_CTX:-8192}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURE_RUNS="${MEASURE_RUNS:-3}"
# --repeat N flag overrides MEASURE_RUNS when provided.
[[ -n "${_REPEAT_OVERRIDE}" ]] && MEASURE_RUNS="${_REPEAT_OVERRIDE}"
METRICS_OUT="${METRICS_OUT:-${RMLX_METRICS_DIR}/perf-iter/baseline.jsonl}"

HEALTH_URL="http://127.0.0.1:${PORT}/health"
COMPLETIONS_URL="http://127.0.0.1:${PORT}/v1/chat/completions"
MODEL_ID="$(basename "${MODEL_PATH}")"

# ── Buffer dirs (§8.4) ────────────────────────────────────────────────────────
mkdir -p "${RMLX_METRICS_DIR}/buffer/pending" "${RMLX_METRICS_DIR}/buffer/failed"

# ── Prompt fixture ─────────────────────────────────────────────────────────────
# 32-token-ish prompt that is short enough to give clean decode TPS numbers.
PROMPT_TEXT="Summarize the history of the Roman Empire in detail, covering the key emperors, military campaigns, cultural achievements, and the eventual fall."

# max_tokens for measurement: 30 decode steps.
MAX_TOKENS_MEASURE=30
# Warmup can use fewer tokens.
MAX_TOKENS_WARMUP=10

# ── Git metadata ──────────────────────────────────────────────────────────────

_RMLX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# `git_sha` is a commit this run is attributed to, so a working tree with
# uncommitted edits has none to give: `-dirty` is not a commit and nothing can
# look the row's code up by it. Backend, version, build profile and hardware tag
# all come from the measured binary via lib/identity.sh, never from constants
# here — see docs/METRICS_DB.md on caller-supplied identity.
GIT_SHA="$(git -C "${_RMLX_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if ! git -C "${_RMLX_ROOT}" diff --quiet 2>/dev/null ||
    ! git -C "${_RMLX_ROOT}" diff --cached --quiet 2>/dev/null; then
    GIT_SHA="unknown"
fi
TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_ID="$(date -u +%Y%m%d-%H%M%S)-${GIT_SHA}"
# `unknown` is a fallback for run-ids and labels, never provenance — gate the
# git_sha JSON key so a checkout without `.git`, or one with uncommitted edits,
# writes NULL into observations.git_sha instead of a string nothing resolves.
GIT_SHA_KV=""
if [[ "${GIT_SHA}" != unknown* ]]; then
    GIT_SHA_KV="'git_sha': '${GIT_SHA}',"
fi

# ── Helpers ────────────────────────────────────────────────────────────────────

log() { echo "[bench_decode_tps] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Read one `key=value` out of a block, or the empty string when absent.
field_of() { echo "$1" | sed -n "s/^$2=//p" | tail -1; }

# How many ITL samples the server has recorded so far.
ring_len() {
    field_of "$(python3 "${_RMLX_ROOT}/scripts/lib/server_decode_tps.py" \
        "http://127.0.0.1:${PORT}")" ring_len
}

# The rate the server measured for the request that just finished, given the
# ring length observed before it. Empty when it could not attribute one.
server_rate_after() {
    field_of "$(python3 "${_RMLX_ROOT}/scripts/lib/server_decode_tps.py" \
        "http://127.0.0.1:${PORT}" --after "$1")" decode_tps
}

# Check required tools.
for tool in curl python3 awk; do
    command -v "$tool" >/dev/null 2>&1 || die "required tool '$tool' not found"
done

# ── Locate rmlx binary ────────────────────────────────────────────────────────
# Allow override via RMLX_BIN; otherwise try release then debug build.
if [[ -z "${RMLX_BIN:-}" ]]; then
    if [[ -f "./target/release/rmlx" ]]; then
        RMLX_BIN="./target/release/rmlx"
    elif [[ -f "./target/debug/rmlx" ]]; then
        RMLX_BIN="./target/debug/rmlx"
    else
        die "rmlx binary not found; build first with 'make build'"
    fi
fi

# ── Pre-flight (CLAUDE.md mandatory) ─────────────────────────────────────────

log "Pre-flight: killing any existing rmlx/mlx_lm/paroquant/omlx processes..."
pkill -f "rmlx serve" 2>/dev/null || true
pkill -f mlx_lm      2>/dev/null || true
pkill -f paroquant   2>/dev/null || true
pkill -f omlx        2>/dev/null || true
sleep 5
rm -f "/tmp/rmlx.${PORT}.claim"
log "Pre-flight done."

# ── Verify binary + model ─────────────────────────────────────────────────────

[[ -f "${RMLX_BIN}" ]] || die "rmlx binary not found at ${RMLX_BIN}. Run: make build"

# Run identity (backend / version / git sha / build profile / hardware tag)
# comes from the measured binary — never hard-coded here.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/identity.sh"
rmlx_export_identity "${RMLX_BIN}"
[[ -d "${MODEL_PATH}" ]] || die "model directory not found: ${MODEL_PATH}"

# ── Start server ──────────────────────────────────────────────────────────────

log "Starting rmlx serve: model=${MODEL_ID} port=${PORT} kv_quant=${KV_QUANT} max_ctx=${MAX_CTX}"

SERVE_LOG="logs/bench_decode_tps_${RUN_ID}.log"
mkdir -p logs
"${RMLX_BIN}" serve \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --host 127.0.0.1 \
    --device gpu \
    --kv-quant "${KV_QUANT}" \
    --max-ctx "${MAX_CTX}" \
    > "${SERVE_LOG}" 2>&1 &
SERVER_PID=$!

# ── Wait for readiness (poll /health, no fixed sleep) ────────────────────────

log "Waiting for server readiness on ${HEALTH_URL} ..."
MAX_WAIT=120   # seconds
POLL_INTERVAL=2
elapsed=0
until curl -s --max-time 2 "${HEALTH_URL}" | grep -q '"ok"'; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "Server process exited unexpectedly. Tail of log:"
        tail -20 "${SERVE_LOG}" >&2
        die "Server crashed during startup."
    fi
    if [[ ${elapsed} -ge ${MAX_WAIT} ]]; then
        log "Server did not become ready within ${MAX_WAIT}s. Tail of log:"
        tail -20 "${SERVE_LOG}" >&2
        kill "${SERVER_PID}" 2>/dev/null || true
        die "Readiness timeout."
    fi
    sleep "${POLL_INTERVAL}"
    elapsed=$((elapsed + POLL_INTERVAL))
done
log "Server ready after ${elapsed}s."

# ── Completion helper ─────────────────────────────────────────────────────────

# completion_request <max_tokens>
# Sends one chat completion and prints:
#   elapsed_ms <elapsed_ms> tokens <completion_tokens> text <generated_text>
# to stdout.  Returns non-zero on HTTP error.
completion_request() {
    local max_tokens="$1"
    local payload
    payload="$(python3 -c "
import json, sys
print(json.dumps({
    'model': '${MODEL_ID}',
    'messages': [{'role': 'user', 'content': '${PROMPT_TEXT}'}],
    'max_tokens': ${max_tokens},
    'temperature': 0.0,
    'stream': False,
}))
")"

    local ring_before
    ring_before="$(ring_len)"

    local t_start t_end elapsed_ms
    t_start="$(python3 -c 'import time; print(int(time.time() * 1000))')"
    local response
    response="$(curl -s -X POST "${COMPLETIONS_URL}" \
        -H 'Content-Type: application/json' \
        -d "${payload}")"
    t_end="$(python3 -c 'import time; print(int(time.time() * 1000))')"
    elapsed_ms=$(( t_end - t_start ))

    local decode_tps
    decode_tps="$(server_rate_after "${ring_before:-0}")" || decode_tps=""

    # Parse completion_tokens from usage field.
    local completion_tokens generated_text
    completion_tokens="$(echo "${response}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(d.get('usage', {}).get('completion_tokens', 0))
" 2>/dev/null || echo 0)"

    generated_text="$(echo "${response}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
choices = d.get('choices', [])
if choices:
    print(choices[0].get('message', {}).get('content', ''))
" 2>/dev/null || echo '')"

    echo "elapsed_ms=${elapsed_ms} decode_tps=${decode_tps} tokens=${completion_tokens} text=${generated_text}"
}

# ── Warmup ────────────────────────────────────────────────────────────────────

log "Running ${WARMUP_RUNS} warmup completion(s)..."
for i in $(seq 1 "${WARMUP_RUNS}"); do
    log "  Warmup ${i}/${WARMUP_RUNS}..."
    completion_request "${MAX_TOKENS_WARMUP}" > /dev/null
done
log "Warmup done."

# ── Measurement runs ─────────────────────────────────────────────────────────

log "Running ${MEASURE_RUNS} measurement completion(s)..."

declare -a tps_values=()
declare -a ttft_values=()
first_text=""

for i in $(seq 1 "${MEASURE_RUNS}"); do
    log "  Measurement ${i}/${MEASURE_RUNS}..."
    result="$(completion_request "${MAX_TOKENS_MEASURE}")"

    elapsed_ms="$(echo "${result}" | grep -oE 'elapsed_ms=[0-9]+' | cut -d= -f2)"
    tps="$(echo "${result}" | grep -oE 'decode_tps=[0-9.]+' | cut -d= -f2)"
    n_tokens="$(echo "${result}" | grep -oE 'tokens=[0-9]+' | cut -d= -f2)"
    text="$(echo "${result}" | sed 's/^elapsed_ms=[0-9]* decode_tps=[0-9.]* tokens=[0-9]* text=//')"

    [[ -n "${tps}" ]] || die "run ${i}: the server attributed no decode rate to it"

    # The mean inter-token gap is the reciprocal of that rate, so it is the same
    # measurement rather than a second one derived from the wall clock.
    step_ms="$(LC_ALL=C python3 -c "print(f'{1000.0 / ${tps}:.1f}')")"

    tps_values+=("${tps}")
    ttft_values+=("${step_ms}")

    if [[ ${i} -eq 1 ]]; then
        first_text="${text}"
    fi

    log "    elapsed_ms=${elapsed_ms} tokens=${n_tokens} decode_tps=${tps} step_ms=${step_ms}"
done

if [[ ${#tps_values[@]} -ne ${MEASURE_RUNS} ]]; then
    die "${#tps_values[@]} decode rates for ${MEASURE_RUNS} measured runs"
fi

# ── Statistics ────────────────────────────────────────────────────────────────

tps_array="${tps_values[*]}"
step_ms_array="${ttft_values[*]}"

read -r tps_mean tps_stddev <<< "$(python3 -c "
import math
vals = [float(x) for x in '${tps_array}'.split()]
if not vals:
    print('0.0 0.0')
else:
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        variance = sum((v - mean)**2 for v in vals) / (len(vals) - 1)
        stddev = math.sqrt(variance)
    else:
        stddev = 0.0
    print(f'{mean:.2f} {stddev:.2f}')
")"

read -r ttft_mean <<< "$(python3 -c "
vals = [float(x) for x in '${step_ms_array}'.split()]
print(f'{sum(vals)/len(vals):.1f}' if vals else '0.0')
")"

# ── Print summary ─────────────────────────────────────────────────────────────

echo ""
echo "=== bench_decode_tps results ==="
echo "  model:           ${MODEL_ID}"
echo "  kv_quant:        ${KV_QUANT}"
echo "  decode_tps_mean: ${tps_mean}"
echo "  decode_tps_std:  ${tps_stddev}"
echo "  step_ms_mean:    ${ttft_mean}  (mean inter-token gap, from the server)"
echo "  first_text:      ${first_text:0:120}"

# ── Write JSONL metrics ────────────────────────────────────────────────────────

mkdir -p "$(dirname "${METRICS_OUT}")"

# Extract first 32 tokens from first_text (word-level approximation; exact token
# IDs are not accessible from the non-streaming OpenAI endpoint — see NOTES below).
first_32_words="$(echo "${first_text}" | python3 -c "
import sys, json
words = sys.stdin.read().split()[:32]
print(json.dumps(words))
")"

python3 -c "
import json, sys
from datetime import datetime, timezone

record = {
    'run_id':             '${RUN_ID}',
    'ts_utc':             '${TS_UTC}',
    'model_path':         '${MODEL_PATH}',
    'kv_quant':           '${KV_QUANT}',
    'decode_tps_mean':    ${tps_mean},
    'decode_tps_stddev':  ${tps_stddev},
    'step_ms_mean':       ${ttft_mean},
    'first_32_words':     ${first_32_words},
    ${GIT_SHA_KV}
    'notes':              'decode_window=engine_itl; first_32_words from temp=0 decode',
}
print(json.dumps(record))
" >> "${METRICS_OUT}"

log "Appended JSONL record to ${METRICS_OUT}"

# ── §8.5 buffer record + recorder invocation ──────────────────────────────────
# Build a §8.5 RunRecord JSON and hand it to 'rmlx metrics record --file'.
# On success the recorder deletes the file; on failure we move to failed/ and warn.
# This runs AFTER the legacy JSONL write so a recorder failure never blocks the bench.

_PROMPT_FILE="prompts/longctx_4k.json"
_METRICS_DB="${METRICS_DB:-${RMLX_METRICS_DIR}/runs.db}"
_BUF_TS="$(date -u +%Y%m%d%H%M%S)"

# Generate a short random hex suffix (lowercase 8 chars) for the buffer filename.
if command -v uuidgen >/dev/null 2>&1; then
    _BUF_UUID="$(uuidgen | tr 'A-Z' 'a-z' | tr -d '-' | head -c 8)"
else
    _BUF_UUID="$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
fi
_RECORD_PATH="${RMLX_METRICS_DIR}/buffer/pending/${_BUF_TS}-${_BUF_UUID}.json"

# Build the §8.5 JSON using Python (jq falls back to python3 — avoids hard dep).
# model_namespace + model: split on '__' separator per §5.1.
#   /path/to/ns__model  → ns + model
#   /path/to/model      → "local" + model
python3 -c "
import json, os, re, sys

model_path = '${MODEL_PATH}'.rstrip('/')
model_dir  = os.path.basename(model_path)
if '__' in model_dir:
    ns, mdl = model_dir.split('__', 1)
else:
    ns  = 'local'
    mdl = model_dir

# weight_quant: infer from model name suffix keywords.
def infer_weight_quant(name):
    n = name.lower()
    for tok, q in [('mxfp8','mxfp8'),('mxfp4','mxfp4'),('nvfp4','nvfp4'),
                   ('8bit','q8_0'),('4bit','q4_k_m'),('2bit','2bit'),
                   ('paro','paro'),('bf16','bf16'),('fp16','fp16')]:
        if tok in n:
            return q
    return 'unknown'

weight_quant = infer_weight_quant(mdl)

# Load prompt body from file.
prompt_body = None
prompt_file = '${_PROMPT_FILE}'
if os.path.isfile(prompt_file):
    with open(prompt_file) as f:
        pf = json.load(f)
    prompt_body = pf['messages']
else:
    prompt_body = [{'role': 'user', 'content': '${PROMPT_TEXT}'}]

# first_32_words joined, truncated to 64 chars.
words = ${first_32_words}
output_first_64 = ' '.join(words)[:64]

rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    ${GIT_SHA_KV}
    'model_namespace': ns,
    'model':           mdl,
    'weight_quant':    weight_quant,
    'kv_quant':        '${KV_QUANT}',
    'ctx_max':         int('${MAX_CTX}'),
    'prompt': {
        'name':          'longctx_4k',
        'body':          prompt_body,
        'tokens_approx': 4096,
    },
    'ts_utc':          '${TS_UTC}',
    'prompt_tokens':   4096,
    'max_tokens':      int('${MAX_TOKENS_MEASURE}'),
    'temperature':     0.0,
    'seed':            0,
    'n_warmups':       int('${WARMUP_RUNS}'),
    'n_measure':       int('${MEASURE_RUNS}'),
    'output_first_64': output_first_64,
    'notes':           'decode_window=engine_itl; perf-iter bench script',
    'description':     None,
    'metrics': [
        {'name': 'decode_tps_warm', 'value': float('${tps_mean}'),   'stddev': float('${tps_stddev}')},
        {'name': 'step_ms_mean',    'value': float('${ttft_mean}')},
    ],
}
print(json.dumps(rec))
" > "${_RECORD_PATH}" 2>/dev/null || {
    log "WARN: failed to build §8.5 record JSON; skipping recorder"
    rm -f "${_RECORD_PATH}"
}

if [[ -f "${_RECORD_PATH}" ]]; then
    if "${RMLX_BIN}" metrics --db "${_METRICS_DB}" record --file "${_RECORD_PATH}"; then
        # Recorder deletes on success; defensive rm in case it didn't.
        rm -f "${_RECORD_PATH}"
        log "§8.5 record ingested into ${_METRICS_DB}"
    else
        mv "${_RECORD_PATH}" "${RMLX_METRICS_DIR}/buffer/failed/"
        log "WARN: recorder rejected the record; see metrics/buffer/failed/${_BUF_TS}-${_BUF_UUID}.json"
        log "WARN: bench results are still valid; only DB recording failed."
    fi
fi

# ── Teardown ───────────────────────────────────────────────────────────────────

log "Stopping server (PID ${SERVER_PID})..."
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
rm -f "/tmp/rmlx.${PORT}.claim"
log "Done."
