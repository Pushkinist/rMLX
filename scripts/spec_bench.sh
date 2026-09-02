#!/usr/bin/env bash
# spec_bench.sh — bench gemma-4-e2b-it-mxfp8 in normal vs MTP speculative-decode mode.
#
# Usage:
#   bash scripts/spec_bench.sh [--port N] [--dry-run]
#
# Requires:
#   - Built binary at target/release-perf/rmlx  (run: make build-perf)
#   - Model paths set via env or hard-coded below:
#       VERIFIER_MODEL  — mxfp8 verifier snapshot dir
#       DRAFTER_MODEL   — bf16 assistant drafter snapshot dir
#
# Output:
#   - Two RunRecord JSON files written + ingested into runs.db
#   - Final comparison table printed to stdout
#
# Measurement basis (docs/SPECULATIVE.md):
#   Both arms report decode throughput over the window from the first emitted
#   token to the last, prefill excluded, so the two rows mean the same thing and
#   the delta between them is a decode-rate delta. The speculative arm takes the
#   rate the round loop measured and logged; the no-drafter arm has no such
#   engine-side figure and is timed client-side over the streamed tokens.
#
# Hard constraints honoured:
#   - Preflight (pkill + claim-file delete) before each server start
#   - Single server process at a time; killed explicitly between phases
#   - 1 warmup + 3 measured requests per config; 5 s sleep between requests
#   - All inserts go through `rmlx metrics record --file` (no direct sqlite writes)

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${REPO_ROOT}/target/release-perf/rmlx"

# Run identity (backend / version / git sha / build profile / hardware tag)
# comes from the measured binary — never hard-coded here.
source "$(dirname "${BASH_SOURCE[0]}")/lib/identity.sh"
rmlx_export_identity "${BINARY}"
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
LOG_DIR="${RMLX_HOME}/logs"
BUFFER_DIR="${RMLX_HOME}/metrics/buffer/pending"
DB_PATH="${RMLX_HOME}/metrics/runs.db"
SCRATCH_DIR="${RMLX_HOME}/tmp"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

# Model paths — must be set via env. Concrete absolute paths live in LOCAL.md
# (gitignored, never committed). See LOCAL.md §"Test-target model snapshots".
VERIFIER_MODEL="${VERIFIER_MODEL:-}"
DRAFTER_MODEL="${DRAFTER_MODEL:-}"

if [[ -z "${VERIFIER_MODEL}" || -z "${DRAFTER_MODEL}" ]]; then
    cat >&2 <<USAGE
ERROR: VERIFIER_MODEL and DRAFTER_MODEL env vars are required.

Resolve concrete absolute paths from LOCAL.md (gitignored). Example:

  VERIFIER_MODEL=\$O_MODELS/mlx-community__gemma-4-e2b-it-mxfp8 \\
  DRAFTER_MODEL=\$O_MODELS/mlx-community__gemma-4-E2B-it-assistant-bf16 \\
    bash scripts/spec_bench.sh

USAGE
    exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────

PORT="${PORT:-8090}"
WARMUP_RUNS=1
MEASURED_RUNS=3
MAX_TOKENS=128
TEMPERATURE=0
SEED=42
DRAFT_KIND="mtp"
DRAFT_BLOCK_SIZE=5
HARDWARE_TAG="${RMLX_HARDWARE_TAG:-m5_max_128gb}"

DRY_RUN=false
BENCH_TAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --port=*) PORT="${1#--port=}"; shift ;;
        --port) shift; PORT="${1:?--port requires a value}"; shift ;;
        --tag=*) BENCH_TAG="${1#--tag=}"; shift ;;
        --tag) shift; BENCH_TAG="${1:?--tag requires a value}"; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── Prompt body ───────────────────────────────────────────────────────────────
#
# Priority: BENCH_PROMPT_FILE (canonical JSON) > BENCH_PROMPT (inline string) >
# default prose prompt. Canonical JSON files in prompts/spec_bench/ carry
# `name` + `messages[0].content` and are the preferred source.
# BENCH_PROMPT_NAME overrides the registered prompt row name; if unset and
# BENCH_PROMPT_FILE is used, the JSON's `name` field is honoured.

if [[ -n "${BENCH_PROMPT_FILE:-}" ]]; then
    if [[ ! -f "${BENCH_PROMPT_FILE}" ]]; then
        echo "ERROR: BENCH_PROMPT_FILE not found: ${BENCH_PROMPT_FILE}" >&2
        exit 1
    fi
    PROMPT_CONTENT="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['messages'][0]['content'])" "${BENCH_PROMPT_FILE}")"
    PROMPT_NAME="${BENCH_PROMPT_NAME:-$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['name'])" "${BENCH_PROMPT_FILE}")}"
else
    PROMPT_CONTENT="${BENCH_PROMPT:-List 5 important features of the Rust programming language.}"
    PROMPT_NAME="${BENCH_PROMPT_NAME:-spec-bench-rust-5-features}"
fi

# The server registers models by their full snapshot directory basename
# (namespace__model), so the "model" field in OpenAI requests must use that.
MODEL_ID="mlx-community__gemma-4-e2b-it-mxfp8"

# JSON-escape PROMPT_CONTENT via python3 so quotes/backslashes in canonical
# prompt files don't corrupt the curl payload.
CURL_PAYLOAD=$(
    MODEL_ID="${MODEL_ID}" PROMPT_CONTENT="${PROMPT_CONTENT}" \
    MAX_TOKENS="${MAX_TOKENS}" TEMPERATURE="${TEMPERATURE}" SEED="${SEED}" \
    python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL_ID"],
    "messages": [{"role": "user", "content": os.environ["PROMPT_CONTENT"]}],
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "seed": int(os.environ["SEED"]),
    "stream": True,
}))
'
)

# ── Sanity checks ─────────────────────────────────────────────────────────────

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}. Run: make build-perf" >&2
    exit 125
fi

if [[ ! -d "${VERIFIER_MODEL}" ]]; then
    echo "ERROR: verifier model not found: ${VERIFIER_MODEL}" >&2
    exit 1
fi

if [[ ! -d "${DRAFTER_MODEL}" ]]; then
    echo "ERROR: drafter model not found: ${DRAFTER_MODEL}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${BUFFER_DIR}" "${SCRATCH_DIR}"

echo "==> spec_bench.sh"
echo "    verifier : ${VERIFIER_MODEL}"
echo "    drafter  : ${DRAFTER_MODEL}"
echo "    port     : ${PORT}"
echo "    git_sha  : ${GIT_SHA}"
echo "    dry_run  : ${DRY_RUN}"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────

preflight() {
    echo "  [preflight] killing competing MLX processes..." >&2
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm 2>/dev/null || true
    pkill -f paroquant 2>/dev/null || true
    pkill -f omlx 2>/dev/null || true
    sleep 5
    rm -f /tmp/rmlx.*.claim 2>/dev/null || true
    echo "  [preflight] done." >&2
}

# Wait for server to be ready (polls /v1/models).
wait_for_server() {
    local url="http://127.0.0.1:${PORT}/v1/models"
    local attempts=0
    local max_attempts=60
    echo "  [wait] polling ${url} ..." >&2
    while true; do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo "  [wait] server ready." >&2
            return 0
        fi
        attempts=$((attempts + 1))
        if [[ ${attempts} -ge ${max_attempts} ]]; then
            echo "ERROR: server did not start within $((max_attempts * 2))s" >&2
            return 1
        fi
        sleep 2
    done
}

# Fire one measured chat-completions request and report what the client saw.
#
# Prints the `key=value` block of scripts/lib/sse_decode_window.py: `tokens`,
# `preview`, and `decode_tps` over the first-content-token to last-content-token
# window. A rate over the whole request instead would count the prefill and the
# curl spawn and read low, and would not be comparable with the rate the engine
# logs for the speculative arm.
measured_request() {
    local raw_file="$1"
    curl -s \
        -H "Content-Type: application/json" \
        -d "${CURL_PAYLOAD}" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --no-buffer 2>/dev/null \
        | python3 "${REPO_ROOT}/scripts/lib/sse_decode_window.py" --raw "${raw_file}"
}

# Read one `key=value` out of a block, or the empty string when absent.
field_of() {
    local block="$1" key="$2"
    echo "${block}" | sed -n "s/^${key}=//p" | tail -1
}

# Compute median of space-separated values.
median() {
    echo "$@" | tr ' ' '\n' | sort -n | awk '
    { a[NR]=$1 }
    END {
        n=NR
        if (n % 2 == 1) print a[(n+1)/2]
        else printf "%.6f\n", (a[n/2] + a[n/2+1]) / 2
    }'
}

# Compute sample stddev.
stddev() {
    echo "$@" | tr ' ' '\n' | awk '
    NR==1 { first=$1 }
    { sum+=$1; sumsq+=$1*$1; n++ }
    END {
        if (n < 2) { print "0.000000"; exit }
        mean = sum/n
        variance = (sumsq - n*mean*mean) / (n-1)
        if (variance < 0) variance = 0
        printf "%.6f\n", sqrt(variance)
    }'
}

# Emit a §8.5 RunRecord JSON and ingest it.
# Args: config_name decode_tps stddev accept_rate draft_tokens_total
#        accept_tokens_total draft_rounds_total accepted_per_step output_preview kv_quant
emit_and_ingest() {
    local config="$1"
    local decode_tps="$2"
    local decode_tps_stddev="$3"
    local accept_rate="$4"
    local draft_tokens_total="$5"
    local accept_tokens_total="$6"
    local draft_rounds_total="$7"
    local accepted_per_step="$8"
    local preview="$9"
    local kv_quant="${10:-k8v8}"

    local ts_utc
    ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    # Build the RunRecord JSON using Python, passing all values via env vars to
    # avoid shell quoting issues with numeric and text fields.
    local record_json
    record_json=$(
        BENCH_CONFIG="${config}" \
        BENCH_DECODE_TPS="${decode_tps}" \
        BENCH_DECODE_TPS_STDDEV="${decode_tps_stddev}" \
        BENCH_ACCEPT_RATE="${accept_rate}" \
        BENCH_DRAFT_TOKENS_TOTAL="${draft_tokens_total}" \
        BENCH_ACCEPT_TOKENS_TOTAL="${accept_tokens_total}" \
        BENCH_DRAFT_ROUNDS_TOTAL="${draft_rounds_total}" \
        BENCH_ACCEPTED_PER_STEP="${accepted_per_step}" \
        BENCH_PREVIEW="${preview}" \
        BENCH_KV_QUANT="${kv_quant}" \
        BENCH_TS_UTC="${ts_utc}" \
        BENCH_GIT_SHA="${GIT_SHA}" \
        BENCH_HARDWARE_TAG="${HARDWARE_TAG}" \
        BENCH_MAX_TOKENS="${MAX_TOKENS}" \
        BENCH_WARMUP_RUNS="${WARMUP_RUNS}" \
        BENCH_MEASURED_RUNS="${MEASURED_RUNS}" \
        BENCH_TEMPERATURE="${TEMPERATURE}" \
        BENCH_SEED="${SEED}" \
        BENCH_DRAFT_KIND="${DRAFT_KIND}" \
        BENCH_DRAFT_BLOCK_SIZE="${DRAFT_BLOCK_SIZE}" \
        BENCH_TAG="${BENCH_TAG}" \
        BENCH_PROMPT_CONTENT="${PROMPT_CONTENT}" \
        BENCH_PROMPT_NAME="${PROMPT_NAME}" \
        python3 - <<'PYEOF'
import json, os

config = os.environ["BENCH_CONFIG"]
decode_tps = float(os.environ["BENCH_DECODE_TPS"])
decode_tps_stddev = float(os.environ["BENCH_DECODE_TPS_STDDEV"])
accept_rate = float(os.environ["BENCH_ACCEPT_RATE"])
draft_tokens_total = int(os.environ["BENCH_DRAFT_TOKENS_TOTAL"])
accept_tokens_total = int(os.environ["BENCH_ACCEPT_TOKENS_TOTAL"])
draft_rounds_total = int(os.environ["BENCH_DRAFT_ROUNDS_TOTAL"])
accepted_per_step = float(os.environ["BENCH_ACCEPTED_PER_STEP"])
preview = os.environ["BENCH_PREVIEW"]
kv_quant = os.environ["BENCH_KV_QUANT"]
ts_utc = os.environ["BENCH_TS_UTC"]
git_sha = os.environ["BENCH_GIT_SHA"]
hardware_tag = os.environ["BENCH_HARDWARE_TAG"]
max_tokens = int(os.environ["BENCH_MAX_TOKENS"])
warmup_runs = int(os.environ["BENCH_WARMUP_RUNS"])
measured_runs = int(os.environ["BENCH_MEASURED_RUNS"])
temperature = float(os.environ["BENCH_TEMPERATURE"])
seed = int(os.environ["BENCH_SEED"])
draft_kind = os.environ["BENCH_DRAFT_KIND"]
draft_block_size = int(os.environ["BENCH_DRAFT_BLOCK_SIZE"])
bench_tag = os.environ.get("BENCH_TAG", "")
tag_suffix = f" tag={bench_tag}" if bench_tag else ""

prompt_content = os.environ["BENCH_PROMPT_CONTENT"]
prompt_name = os.environ["BENCH_PROMPT_NAME"]
prompt_body = [{"role": "user", "content": prompt_content}]

if config == "normal":
    metrics = [
        {"name": "decode_tps_warm", "value": decode_tps, "stddev": decode_tps_stddev},
        {"name": "draft_tokens_total", "value": 0},
    ]
else:
    metrics = [
        {"name": "decode_tps_warm", "value": decode_tps, "stddev": decode_tps_stddev},
        {"name": "accept_rate", "value": accept_rate},
        {"name": "draft_tokens_total", "value": draft_tokens_total},
        {"name": "accept_tokens_total", "value": accept_tokens_total},
        {"name": "draft_rounds_total", "value": draft_rounds_total},
        {"name": "accepted_per_step", "value": accepted_per_step},
    ]

obj = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
    # "unknown" is a fallback for the description label below, never
    # provenance — a checkout without .git must not stamp git_sha at all.
    **({"git_sha": git_sha} if not git_sha.startswith("unknown") else {}),
    "model_namespace": "mlx-community",
    "model": "gemma-4-e2b-it-mxfp8",
    "weight_quant": "mxfp8",
    "kv_quant": kv_quant,
    "ctx_max": 8192,
    "prompt": {
        "name": prompt_name,
        "body": prompt_body,
    },
    "ts_utc": ts_utc,
    "prompt_tokens": 14,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "seed": seed,
    "n_warmups": warmup_runs,
    "n_measure": measured_runs,
    "output_first_64": preview,
    # `decode_window` names where this row's decode_tps_warm came from, so rows
    # written while the script derived a prefill-inclusive rate of its own stay
    # tellable apart from rows that carry the measured window
    # (docs/METRICS_DB.md, "Known-bad rows already in the DB").
    "notes": (
        f"config={config} draft_kind=none{tag_suffix} decode_window=client_sse"
        if config == "normal"
        else f"config={config} draft_kind={draft_kind} "
        f"block_size={draft_block_size}{tag_suffix} decode_window=engine"
    ),
    "description": f"spec_bench {config} sha={git_sha}",
    "metrics": metrics,
}
print(json.dumps(obj))
PYEOF
    )

    local ts_tag
    ts_tag="$(date -u +"%Y%m%dT%H%M%S%3N")"
    local pid=$$
    local buf_file="${BUFFER_DIR}/${ts_tag}-${pid}-${config}.json"

    if $DRY_RUN; then
        echo "  [dry-run] would write buffer: ${buf_file}" >&2
        echo "  [dry-run] record: $(echo "${record_json}" | python3 -c "import sys,json; d=json.load(sys.stdin); d['metrics']=d.get('metrics',[]); print('model='+d['model'], 'kv_quant='+d['kv_quant'], 'metrics_count='+str(len(d['metrics'])))")" >&2
        return 0
    fi

    if [[ -z "${record_json}" ]]; then
        echo "  ERROR: record_json is empty; skipping ingest for config=${config}" >&2
        return 1
    fi

    echo "${record_json}" > "${buf_file}"
    echo "  [ingest] buffer: ${buf_file}" >&2

    RMLX_HOME="${RMLX_HOME}" "${BINARY}" metrics record --file "${buf_file}" >&2
    echo "  [ingest] done: ${buf_file}" >&2
    echo "${buf_file}"
}

# ── Phase 1: normal decode ─────────────────────────────────────────────────────

echo "==> Phase 1: normal decode (no drafter)"
echo ""

preflight

# Timestamp reference file for finding the new log.
touch "${SCRATCH_DIR}/ts_ref"
sleep 1

echo "  [server] starting..." >&2
RMLX_HOME="${RMLX_HOME}" \
RMLX_LOG_CAP_MB=200 \
    "${BINARY}" serve \
        --model "${VERIFIER_MODEL}" \
        --port "${PORT}" \
        --log info \
        > "${SCRATCH_DIR}/normal_stdout.txt" 2>&1 &

SERVER_PID=$!
echo "  [server] pid=${SERVER_PID}" >&2

wait_for_server

# Update ref timestamp after server is up.
touch "${SCRATCH_DIR}/ts_ref"

echo "  [normal] warmup..." >&2
for i in $(seq 1 ${WARMUP_RUNS}); do
    measured_request "${SCRATCH_DIR}/warmup_resp.txt" > /dev/null || true
    echo "  [normal] warmup ${i} done" >&2
    sleep 5
done

echo "  [normal] measured runs..." >&2
NORMAL_TPS_VALUES=()
NORMAL_PREVIEW=""

for i in $(seq 1 ${MEASURED_RUNS}); do
    echo "  [normal] measured run ${i}/${MEASURED_RUNS}..." >&2

    RUN_BLOCK="$(measured_request "${SCRATCH_DIR}/normal_resp.txt")" || RUN_BLOCK=""
    N_TOKENS="$(field_of "${RUN_BLOCK}" tokens)"
    TPS="$(field_of "${RUN_BLOCK}" decode_tps)"
    PREVIEW="$(field_of "${RUN_BLOCK}" preview)"

    if [[ -z "${TPS}" ]]; then
        echo "ERROR: normal run ${i} produced no measurable decode window" \
             "(tokens=${N_TOKENS:-0}); response head:" >&2
        head -c 200 "${SCRATCH_DIR}/normal_resp.txt" >&2 || true
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi

    echo "  [normal] run ${i}: tokens=${N_TOKENS} decode_tps=${TPS}" >&2

    NORMAL_TPS_VALUES+=("${TPS}")
    [[ -n "${PREVIEW}" ]] && NORMAL_PREVIEW="${PREVIEW}"
    sleep 5
done

echo "  [server] killing pid=${SERVER_PID}" >&2
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
sleep 3

NORMAL_MEDIAN_TPS=$(median "${NORMAL_TPS_VALUES[@]}")
NORMAL_STDDEV_TPS=$(stddev "${NORMAL_TPS_VALUES[@]}")

echo "  [normal] median_tps=${NORMAL_MEDIAN_TPS} stddev=${NORMAL_STDDEV_TPS}" >&2

# Ingest normal record.
NORMAL_BUF_PATH=$(emit_and_ingest \
    "normal" \
    "${NORMAL_MEDIAN_TPS}" \
    "${NORMAL_STDDEV_TPS}" \
    "0.0" \
    "0" \
    "0" \
    "0" \
    "0.0" \
    "${NORMAL_PREVIEW}" \
    "k8v8")

echo ""
echo "==> Phase 1 complete. Median decode TPS: ${NORMAL_MEDIAN_TPS}"
echo ""

# ── Phase 2: MTP speculative decode ─────────────────────────────────────────

echo "==> Phase 2: MTP speculative decode (draft_kind=${DRAFT_KIND} block_size=${DRAFT_BLOCK_SIZE})"
echo ""

preflight

touch "${SCRATCH_DIR}/ts_ref"
sleep 1

echo "  [server] starting MTP server..." >&2
RMLX_HOME="${RMLX_HOME}" \
RMLX_LOG_CAP_MB=200 \
    "${BINARY}" serve \
        --model "${VERIFIER_MODEL}" \
        --draft-model "${DRAFTER_MODEL}" \
        --draft-kind "${DRAFT_KIND}" \
        --draft-block-size "${DRAFT_BLOCK_SIZE}" \
        --port "${PORT}" \
        --log info \
        > "${SCRATCH_DIR}/mtp_stdout.txt" 2>&1 &

SERVER_PID=$!
echo "  [server] pid=${SERVER_PID}" >&2

wait_for_server

touch "${SCRATCH_DIR}/ts_ref"
sleep 1

echo "  [mtp] warmup..." >&2
for i in $(seq 1 ${WARMUP_RUNS}); do
    measured_request "${SCRATCH_DIR}/warmup_resp.txt" > /dev/null || true
    echo "  [mtp] warmup ${i} done" >&2
    sleep 5
done

# Reset ref AFTER warmup so log parsing only sees measured requests.
touch "${SCRATCH_DIR}/ts_ref"
sleep 1

echo "  [mtp] measured runs..." >&2
MTP_CLIENT_TPS_VALUES=()
MTP_PREVIEW=""

for i in $(seq 1 ${MEASURED_RUNS}); do
    echo "  [mtp] measured run ${i}/${MEASURED_RUNS}..." >&2

    RUN_BLOCK="$(measured_request "${SCRATCH_DIR}/mtp_resp.txt")" || RUN_BLOCK=""
    N_TOKENS="$(field_of "${RUN_BLOCK}" tokens)"
    CLIENT_TPS="$(field_of "${RUN_BLOCK}" decode_tps)"
    PREVIEW="$(field_of "${RUN_BLOCK}" preview)"

    if [[ -z "${CLIENT_TPS}" ]]; then
        echo "ERROR: mtp run ${i} produced no measurable decode window" \
             "(tokens=${N_TOKENS:-0}); response head:" >&2
        head -c 200 "${SCRATCH_DIR}/mtp_resp.txt" >&2 || true
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi

    echo "  [mtp] run ${i}: tokens=${N_TOKENS} client_decode_tps=${CLIENT_TPS}" >&2

    MTP_CLIENT_TPS_VALUES+=("${CLIENT_TPS}")
    [[ -n "${PREVIEW}" ]] && MTP_PREVIEW="${PREVIEW}"
    sleep 5
done

echo "  [server] killing MTP server pid=${SERVER_PID}" >&2
kill "${SERVER_PID}" 2>/dev/null || true
wait "${SERVER_PID}" 2>/dev/null || true
sleep 3

# Find the MTP server's JSONL log (the most recent one).
sleep 2  # let log flush
MTP_LOG=$(find "${LOG_DIR}" -name "*.jsonl" -newer "${SCRATCH_DIR}/ts_ref" \
    2>/dev/null | sort | tail -1)

if [[ -z "${MTP_LOG}" ]]; then
    echo "  WARN: could not find MTP log file after timestamp; trying latest overall" >&2
    MTP_LOG=$(ls -t "${LOG_DIR}"/*.jsonl 2>/dev/null | head -1)
fi

echo "  [mtp] parsing spec metrics from: ${MTP_LOG}" >&2

# Round counts, draft/accept totals and the engine's own decode rate all come
# off the round-loop `done` line, and scripts/lib/spec_round_log.py is the only
# thing that reads it. Skips the warmup by taking the last MEASURED_RUNS events.
if ! MTP_SPEC_DATA=$(python3 "${REPO_ROOT}/scripts/lib/spec_round_log.py" \
        "${MTP_LOG}" --last "${MEASURED_RUNS}"); then
    echo "ERROR: no usable speculative round-loop record in ${MTP_LOG}" >&2
    exit 1
fi

echo "  [mtp] spec metrics: $(echo "${MTP_SPEC_DATA}" | tr '\n' ' ')" >&2

MTP_ROUNDS_TOTAL="$(field_of "${MTP_SPEC_DATA}" rounds_total)"
MTP_DRAFT_TOTAL="$(field_of "${MTP_SPEC_DATA}" draft_tokens_total)"
MTP_ACCEPT_TOTAL="$(field_of "${MTP_SPEC_DATA}" accept_tokens_total)"
MTP_ACCEPT_RATE="$(field_of "${MTP_SPEC_DATA}" accept_rate)"
MTP_ACCEPTED_PER_STEP="$(field_of "${MTP_SPEC_DATA}" accepted_per_step)"

# The engine measures the speculative decode rate over the window from the first
# emitted token to the last and reports it per request; the script's job is to
# aggregate those, not to derive a rate of its own. An `emitted / elapsed_ms`
# off the same line would count the prefill and read low.
MTP_ENGINE_TPS_VALUES=()
while IFS= read -r rate; do
    MTP_ENGINE_TPS_VALUES+=("${rate}")
done < <(echo "${MTP_SPEC_DATA}" | sed -n 's/^decode_tps=//p')

if [[ ${#MTP_ENGINE_TPS_VALUES[@]} -eq 0 ]]; then
    echo "ERROR: the round loop reported no measurable decode rate in ${MTP_LOG}" >&2
    exit 1
fi

MTP_MEDIAN_TPS=$(median "${MTP_ENGINE_TPS_VALUES[@]}")
MTP_STDDEV_TPS=$(stddev "${MTP_ENGINE_TPS_VALUES[@]}")
MTP_CLIENT_MEDIAN_TPS=$(median "${MTP_CLIENT_TPS_VALUES[@]}")

echo "  [mtp] median_tps=${MTP_MEDIAN_TPS} stddev=${MTP_STDDEV_TPS}" \
     "(client-observed median: ${MTP_CLIENT_MEDIAN_TPS})" >&2

MTP_BUF_PATH=$(emit_and_ingest \
    "mtp" \
    "${MTP_MEDIAN_TPS}" \
    "${MTP_STDDEV_TPS}" \
    "${MTP_ACCEPT_RATE}" \
    "${MTP_DRAFT_TOTAL}" \
    "${MTP_ACCEPT_TOTAL}" \
    "${MTP_ROUNDS_TOTAL}" \
    "${MTP_ACCEPTED_PER_STEP}" \
    "${MTP_PREVIEW}" \
    "k8v8")

echo ""
echo "==> Phase 2 complete. Median decode TPS: ${MTP_MEDIAN_TPS}"
echo ""

# ── Final SQL verify ──────────────────────────────────────────────────────────

if ! $DRY_RUN; then
    echo "==> DB verify (rows from last 30 minutes)"
    sqlite3 "${DB_PATH}" \
        "SELECT backend, model, kv_quant, metric, ROUND(value,3) as value
         FROM observations
         WHERE model='gemma-4-e2b-it-mxfp8'
           AND ts_utc >= datetime('now','-30 minutes')
         ORDER BY metric, ts_utc;" \
        2>/dev/null || echo "  (sqlite3 not available or DB empty)"
    echo ""
fi

# ── Final comparison table ────────────────────────────────────────────────────

TPS_DELTA=$(python3 -c "
n=${NORMAL_MEDIAN_TPS}; m=${MTP_MEDIAN_TPS}
if n > 0:
    d = (m - n) / n * 100
    sign = '+' if d >= 0 else ''
    print(f'{sign}{d:.1f}%')
else:
    print('N/A')
" 2>/dev/null || echo "N/A")

NORMAL_SD_FMT=$(python3 -c "print(f'{float(\"${NORMAL_STDDEV_TPS}\"):.2f}')" 2>/dev/null || echo "${NORMAL_STDDEV_TPS}")
MTP_SD_FMT=$(python3 -c "print(f'{float(\"${MTP_STDDEV_TPS}\"):.2f}')" 2>/dev/null || echo "${MTP_STDDEV_TPS}")

echo "============================================================"
echo "  SPEC BENCH RESULTS — gemma-4-e2b-it-mxfp8"
echo "============================================================"
printf "%-10s  %-16s  %-13s  %-20s  %-22s  %s\n" \
    "Config" "decode_tps_warm" "accept_rate" "accepted_per_step" "draft_tokens_total" "notes"
printf "%-10s  %-16s  %-13s  %-20s  %-22s  %s\n" \
    "------" "---------------" "-----------" "-----------------" "------------------" "-----"
printf "%-10s  %-16.2f  %-13s  %-20s  %-22s  %s\n" \
    "normal" \
    "${NORMAL_MEDIAN_TPS}" \
    "N/A" \
    "N/A" \
    "0" \
    "baseline (±${NORMAL_SD_FMT})"
printf "%-10s  %-16.2f  %-13.4f  %-20.4f  %-22s  %s\n" \
    "mtp" \
    "${MTP_MEDIAN_TPS}" \
    "${MTP_ACCEPT_RATE}" \
    "${MTP_ACCEPTED_PER_STEP}" \
    "${MTP_DRAFT_TOTAL}" \
    "${TPS_DELTA} vs normal (±${MTP_SD_FMT})"
echo "============================================================"
echo ""
echo "Buffer files:"
[[ -n "${NORMAL_BUF_PATH}" ]] && echo "  normal : ${NORMAL_BUF_PATH}" || echo "  normal : (dry-run or failed)"
[[ -n "${MTP_BUF_PATH}" ]] && echo "  mtp    : ${MTP_BUF_PATH}" || echo "  mtp    : (dry-run or failed)"
echo ""
echo "Logs:"
echo "  normal server : ${SCRATCH_DIR}/normal_stdout.txt"
echo "  mtp server    : ${SCRATCH_DIR}/mtp_stdout.txt"
[[ -n "${MTP_LOG}" ]] && echo "  mtp spec log  : ${MTP_LOG}"
echo ""
echo "DB: ${DB_PATH}"
echo "Done."
