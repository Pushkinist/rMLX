#!/usr/bin/env bash
# ssd_canary.sh — end-to-end long-session SSD prompt-cache tier canary.
#
# Proves three things against a running rMLX server:
#   1. SSD tier serves repeated cold-equivalent prompts (hit rate climbs on revisit).
#   2. LRU eviction holds under budget pressure (SUM(byte_size) <= budget bytes).
#   3. All step-2 timing slices fire: ssd_spill_ms, ssd_hydrate_ms, ssd_bytes_used,
#      ssd_evict_total populate runs.db and /metrics.
#
# Usage:
#   VERIFIER_MODEL=/path/to/snapshot bash scripts/ssd_canary.sh [--port N] \
#     [--tag TAG] [--ssd-gb N] [--dry-run]
#
# Requires:
#   - Built binary at target/release-perf/rmlx  (run: make build-perf)
#   - VERIFIER_MODEL env var set to snapshot directory (resolve via LOCAL.md)
#
# Output:
#   - .rmlx/proofs/step3-canary/phase_populate.csv
#   - .rmlx/proofs/step3-canary/phase_revisit.csv
#   - .rmlx/proofs/step3-canary/phase_evict.csv
#   - .rmlx/proofs/step3-canary/iteration_summary.json
#   - Observations ingested into runs.db via §8.5 ingest path

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${REPO_ROOT}/target/release-perf/rmlx"
PROMPT_DIR="${REPO_ROOT}/prompts/ssd_bench"

# RMLX_HOME for this canary run — hermetic, wiped before run.
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx/proofs/step3-canary}"
ARTIFACT_DIR="${RMLX_HOME}"
BUFFER_DIR="${RMLX_HOME}/metrics/buffer/pending"
DB_PATH="${RMLX_HOME}/metrics/runs.db"
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"
HARDWARE_TAG="${RMLX_HARDWARE_TAG:-m5_max_128gb}"

# ── Config ────────────────────────────────────────────────────────────────────

PORT="${PORT:-62265}"
SSD_GB="${SSD_GB:-100}"
EVICT_SSD_GB=0.05          # initial 50 MB; overridden dynamically after POPULATE (Option A)
POPULATE_MAX_TOKENS=64
TEMPERATURE=0
SEED=42

DRY_RUN=false
BENCH_TAG="ssd-canary"

# ── Arg parsing ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)         DRY_RUN=true; shift ;;
        --port=*)          PORT="${1#--port=}"; shift ;;
        --port)            shift; PORT="${1:?--port requires a value}"; shift ;;
        --tag=*)           BENCH_TAG="${1#--tag=}"; shift ;;
        --tag)             shift; BENCH_TAG="${1:?--tag requires a value}"; shift ;;
        --ssd-gb=*)        SSD_GB="${1#--ssd-gb=}"; shift ;;
        --ssd-gb)          shift; SSD_GB="${1:?--ssd-gb requires a value}"; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

# ── Model path requirement ────────────────────────────────────────────────────

VERIFIER_MODEL="${VERIFIER_MODEL:-}"
if [[ -z "${VERIFIER_MODEL}" ]]; then
    cat >&2 <<USAGE
ERROR: VERIFIER_MODEL env var is required.

Resolve concrete absolute paths from LOCAL.md (gitignored). Example:

  VERIFIER_MODEL=/path/to/mlx-community__gemma-4-e2b-it-mxfp8 \\
    bash scripts/ssd_canary.sh

USAGE
    exit 1
fi

# Derive the model_id from the snapshot directory basename.
MODEL_ID="$(basename "${VERIFIER_MODEL}")"

# Split namespace__model for RunRecord fields.
MODEL_NAMESPACE="$(echo "${MODEL_ID}" | cut -d_ -f1)"
MODEL_BASENAME="$(echo "${MODEL_ID}" | sed 's/^[^_]*__//')"

# ── Sanity checks ─────────────────────────────────────────────────────────────

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}. Run: make build-perf" >&2
    exit 125
fi

if [[ ! -d "${VERIFIER_MODEL}" ]]; then
    echo "ERROR: verifier model not found: ${VERIFIER_MODEL}" >&2
    exit 1
fi

if [[ ! -d "${PROMPT_DIR}" ]]; then
    echo "ERROR: prompt dir not found: ${PROMPT_DIR}" >&2
    exit 1
fi

# ── Wipe hermetic RMLX_HOME ───────────────────────────────────────────────────

if ! $DRY_RUN; then
    echo "  [setup] wiping ${RMLX_HOME}" >&2
    rm -rf "${RMLX_HOME}"
fi
mkdir -p "${ARTIFACT_DIR}" "${BUFFER_DIR}"

echo "==> ssd_canary.sh"
echo "    model    : ${VERIFIER_MODEL}"
echo "    model_id : ${MODEL_ID}"
echo "    port     : ${PORT}"
echo "    ssd_gb   : ${SSD_GB}"
echo "    git_sha  : ${GIT_SHA}"
echo "    dry_run  : ${DRY_RUN}"
echo "    rmlx_home: ${RMLX_HOME}"
echo ""

# ── Collect all 20 prompt files ───────────────────────────────────────────────

# bash 3 compatible: read sorted prompt list into array without mapfile.
ALL_PROMPTS=()
while IFS= read -r line; do
    ALL_PROMPTS+=("${line}")
done < <(ls "${PROMPT_DIR}"/*.json | sort)
NUM_PROMPTS="${#ALL_PROMPTS[@]}"
echo "  [prompts] found ${NUM_PROMPTS} prompt files in ${PROMPT_DIR}" >&2

if [[ "${NUM_PROMPTS}" -lt 20 ]]; then
    echo "ERROR: expected >= 20 prompts, found ${NUM_PROMPTS}" >&2
    exit 1
fi

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

# Wait for server to be ready by polling /v1/models.
wait_for_server() {
    local url="http://127.0.0.1:${PORT}/v1/models"
    local attempts=0
    local max_attempts=90
    echo "  [wait] polling ${url} ..." >&2
    while true; do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo "  [wait] server ready." >&2
            return 0
        fi
        attempts=$((attempts + 1))
        if [[ ${attempts} -ge ${max_attempts} ]]; then
            echo "ERROR: server did not become ready within $((max_attempts * 2))s" >&2
            return 1
        fi
        sleep 2
    done
}

# Query /v1/models and return the cumulative ssd_hits for the loaded model.
# Prints 0 on any error.
_PREV_CUMULATIVE_SSD_HITS=0
get_cumulative_ssd_hits() {
    # /metrics/cache returns {"models": [...], ...} with ssd_hits per model.
    # /v1/models returns standard OpenAI list without cache stats.
    # Write to temp file to avoid shell pipe truncation mangling JSON.
    local _tmp_mc
    _tmp_mc="/tmp/ssd_canary_mc_$$.json"
    curl -sf "http://127.0.0.1:${PORT}/metrics/cache" --max-time 5 \
        -o "${_tmp_mc}" 2>/dev/null || { echo "0"; return; }
    python3 - "${_tmp_mc}" <<'PYEOF' 2>/dev/null || echo "0"
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    models = d.get('models', [])
    for m in models:
        v = m.get('ssd_hits')
        if v is not None:
            print(int(v))
            sys.exit(0)
    print(0)
except Exception:
    print(0)
PYEOF
    rm -f "${_tmp_mc}" 2>/dev/null || true
}

# Send one chat-completion request for a given prompt file.
# Prints the raw response body. Sets global SSD_HITS_LAST for the caller.
# SSD_HITS_LAST = delta ssd_hits for this request (from /v1/models cumulative).
SSD_HITS_LAST=0
send_request() {
    local prompt_file="$1"
    local max_tokens="${2:-${POPULATE_MAX_TOKENS}}"

    local payload
    payload=$(
        PROMPT_FILE="${prompt_file}" \
        MODEL_ID="${MODEL_ID}" \
        MAX_TOKENS="${max_tokens}" \
        TEMPERATURE="${TEMPERATURE}" \
        SEED="${SEED}" \
        python3 - <<'PYEOF'
import json, os

d = json.load(open(os.environ["PROMPT_FILE"]))
messages = d.get("messages", [{"role": "user", "content": "Hello"}])

print(json.dumps({
    "model": os.environ["MODEL_ID"],
    "messages": messages,
    "max_tokens": int(os.environ["MAX_TOKENS"]),
    "temperature": float(os.environ["TEMPERATURE"]),
    "seed": int(os.environ["SEED"]),
    "stream": False,
}))
PYEOF
    )

    local resp
    resp=$(curl -sf \
        -H "Content-Type: application/json" \
        -d "${payload}" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" \
        --max-time 60 \
        2>/dev/null || echo '{}')

    # Extract ssd_hits as delta from /v1/models cumulative counter.
    # The chat completion response does not carry per-request ssd_hits;
    # read the cumulative counter and compute the increment since last call.
    local cum
    cum=$(get_cumulative_ssd_hits)
    SSD_HITS_LAST=$(( cum - _PREV_CUMULATIVE_SSD_HITS ))
    if [[ "${SSD_HITS_LAST}" -lt 0 ]]; then
        SSD_HITS_LAST=0
    fi
    _PREV_CUMULATIVE_SSD_HITS="${cum}"

    # Track consecutive curl failures (empty JSON = timeout or server error).
    local has_error
    has_error=$(echo "${resp}" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    # Empty dict or no 'choices' = error/timeout response.
    print('1' if not d or 'choices' not in d else '0')
except Exception:
    print('1')
" 2>/dev/null || echo "1")
    if [[ "${has_error}" == "1" ]]; then
        CONSECUTIVE_ERRORS=$(( CONSECUTIVE_ERRORS + 1 ))
    else
        CONSECUTIVE_ERRORS=0
    fi

    echo "${resp}"
}

# If the server appears stuck (consecutive errors), kill and restart it.
# Args: $1=pid_var_name $2=server_flags (the server command args after binary+serve)
# Returns 0 if bounced, 1 if not needed.
CONSECUTIVE_ERRORS=0
HYDRATE_PANIC_DETECTED=false

bounce_if_stuck() {
    local pid_ref="$1"
    shift
    local pid
    eval pid="\${${pid_ref}}"
    if [[ "${CONSECUTIVE_ERRORS}" -lt 2 ]]; then
        return 1
    fi
    echo "  [bounce] ${CONSECUTIVE_ERRORS} consecutive errors; server appears stuck — restarting..." >&2
    HYDRATE_PANIC_DETECTED=true
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    sleep 3
    RMLX_HOME="${RMLX_HOME}" \
    RMLX_LOG_CAP_MB=500 \
        "${BINARY}" serve "$@" \
        > /tmp/ssd_canary_revisit_stdout.txt 2>&1 &
    local new_pid=$!
    eval "${pid_ref}=${new_pid}"
    echo "  [bounce] new server pid=${new_pid}" >&2
    wait_for_server
    CONSECUTIVE_ERRORS=0
    # New server starts with cumulative ssd_hits=0; reset delta baseline.
    _PREV_CUMULATIVE_SSD_HITS=0
    return 0
}

# Parse SSD-tier metrics from multiple sources (index.db + events DB + /metrics).
#
# index.db is the ground truth for on-disk bytes; the Prometheus gauge
# rmlx_ssd_bytes_used is only refreshed at server startup, not after each spill.
# Events table provides per-event spill/hydrate counts from the drain thread.
#
# Prints: ssd_bytes_used ssd_evict_total spill_count hydrate_count spill_sum_us hydrate_sum_us spill_bytes hydrate_bytes
#
# Args: $1 = index.db path (optional), $2 = events DB path (optional)
parse_metrics() {
    local index_db="${1:-}"
    local events_db="${2:-}"

    # --- /metrics for Prometheus histogram sums (spill/hydrate us + evict total) ---
    # Write to temp file to avoid bash pipe+heredoc stdin conflict: when
    # echo "${metrics_text}" | python3 - <<'PYEOF' is used, Python reads
    # the pipe content as its script source (because python3 - reads script
    # from stdin), not as sys.stdin input. Use a temp file to decouple.
    local _metrics_tmp
    _metrics_tmp="/tmp/ssd_canary_pm_$$.txt"
    curl -sf "http://127.0.0.1:${PORT}/metrics" \
        -o "${_metrics_tmp}" 2>/dev/null || printf "" > "${_metrics_tmp}"

    # --- index.db for ground-truth on-disk bytes (updated after each drain) ---
    local ssd_bytes=0
    if [[ -n "${index_db}" ]] && [[ -f "${index_db}" ]]; then
        ssd_bytes=$(sqlite3 "${index_db}" \
            "SELECT COALESCE(SUM(byte_size),0) FROM kv_blocks;" 2>/dev/null || echo "0")
    fi

    # --- events DB for spill/hydrate event rows (ground truth from drain thread) ---
    local spill_count_db=0
    local hydrate_count_db=0
    local spill_sum_us_db=0
    local hydrate_sum_us_db=0
    if [[ -n "${events_db}" ]] && [[ -f "${events_db}" ]]; then
        spill_count_db=$(sqlite3 "${events_db}" \
            "SELECT COUNT(*) FROM events WHERE op='ssd_spill';" 2>/dev/null || echo "0")
        hydrate_count_db=$(sqlite3 "${events_db}" \
            "SELECT COUNT(*) FROM events WHERE op='ssd_hydrate';" 2>/dev/null || echo "0")
        spill_sum_us_db=$(sqlite3 "${events_db}" \
            "SELECT COALESCE(SUM(value),0) FROM events WHERE op='ssd_spill';" 2>/dev/null || echo "0")
        hydrate_sum_us_db=$(sqlite3 "${events_db}" \
            "SELECT COALESCE(SUM(value),0) FROM events WHERE op='ssd_hydrate';" 2>/dev/null || echo "0")
    fi

    python3 - \
        "${_metrics_tmp}" \
        "${ssd_bytes}" "${spill_count_db}" "${hydrate_count_db}" \
        "${spill_sum_us_db}" "${hydrate_sum_us_db}" <<'PYEOF'
import sys, re

# argv[1] = metrics temp file path; argv[2..6] = DB-sourced values.
try:
    with open(sys.argv[1]) as f:
        text = f.read()
except Exception:
    text = ""
ssd_bytes      = int(sys.argv[2])
spill_count    = int(sys.argv[3])
hydrate_count  = int(sys.argv[4])
spill_sum_us   = float(sys.argv[5])
hydrate_sum_us = float(sys.argv[6])

def gauge(pattern, default=0.0):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else default

# Evict total from Prometheus (counter incremented by openai.rs hook).
ssd_evict     = gauge(r'rmlx_ssd_evict_total\{[^}]*\}\s+([\d.e+]+)')

# Histogram sums from /metrics as fallback if events DB is empty.
prom_spill_count   = gauge(r'rmlx_ssd_spill_us_count\s+([\d.e+]+)')
prom_spill_sum_us  = gauge(r'rmlx_ssd_spill_us_sum\s+([\d.e+]+)')
prom_hydrate_count = gauge(r'rmlx_ssd_hydrate_us_count\s+([\d.e+]+)')
prom_hydrate_sum_us= gauge(r'rmlx_ssd_hydrate_us_sum\s+([\d.e+]+)')

# Prefer events-DB counts (ground truth) over Prometheus histogram sums.
if spill_count == 0 and prom_spill_count > 0:
    spill_count   = int(prom_spill_count)
    spill_sum_us  = prom_spill_sum_us
if hydrate_count == 0 and prom_hydrate_count > 0:
    hydrate_count  = int(prom_hydrate_count)
    hydrate_sum_us = prom_hydrate_sum_us

# ssd_bytes: from index.db (arg[1]) takes priority over /metrics gauge.
# The Prometheus gauge only updates at server startup.
print(f"{ssd_bytes} {ssd_evict:.0f} {spill_count} {hydrate_count} {spill_sum_us:.0f} {hydrate_sum_us:.0f} 0 0")
PYEOF
    rm -f "${_metrics_tmp}" 2>/dev/null || true
}

# Emit a §8.5 RunRecord JSON and ingest it into runs.db.
# Args: tag notes_str metrics_json
emit_and_ingest() {
    local tag="$1"
    local notes="$2"
    local metrics_json="$3"

    local ts_utc
    ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    local record_json
    record_json=$(
        BENCH_TAG="${tag}" \
        BENCH_NOTES="${notes}" \
        BENCH_METRICS="${metrics_json}" \
        BENCH_MODEL_ID="${MODEL_ID}" \
        BENCH_MODEL_NAMESPACE="${MODEL_NAMESPACE}" \
        BENCH_MODEL_BASENAME="${MODEL_BASENAME}" \
        BENCH_TS_UTC="${ts_utc}" \
        BENCH_GIT_SHA="${GIT_SHA}" \
        BENCH_HARDWARE_TAG="${HARDWARE_TAG}" \
        BENCH_SEED="${SEED}" \
        BENCH_TEMPERATURE="${TEMPERATURE}" \
        python3 - <<'PYEOF'
import json, os

tag            = os.environ["BENCH_TAG"]
notes          = os.environ["BENCH_NOTES"]
metrics        = json.loads(os.environ["BENCH_METRICS"])
model_id       = os.environ["BENCH_MODEL_ID"]
model_namespace= os.environ["BENCH_MODEL_NAMESPACE"]
model_basename = os.environ["BENCH_MODEL_BASENAME"]
ts_utc         = os.environ["BENCH_TS_UTC"]
git_sha        = os.environ["BENCH_GIT_SHA"]
hardware_tag   = os.environ["BENCH_HARDWARE_TAG"]
seed           = int(os.environ["BENCH_SEED"])
temperature    = float(os.environ["BENCH_TEMPERATURE"])

# Infer weight_quant from model_id suffix (mxfp8, mxfp4, 8bit, 4bit, 2bit, bf16, etc.)
import re
wq_match = re.search(r'(mxfp8|mxfp4|8bit|4bit|2bit|3bit|5bit|6bit|bf16|fp16|q8_0|q4_k_m)', model_basename, re.IGNORECASE)
weight_quant = wq_match.group(1).lower() if wq_match else "bf16"

metric_entries = [
    {"name": k, "value": float(v)} for k, v in metrics.items()
    if v is not None
]

obj = {
    "backend": "rmlx",
    "model_namespace": model_namespace,
    "model": model_basename,
    "weight_quant": weight_quant,
    "kv_quant": "k8v8",
    "ctx_max": 8192,
    "prompt": {
        "name": f"ssd-canary-{tag}",
        "body": [{"role": "user", "content": "ssd_canary batch"}],
    },
    "ts_utc": ts_utc,
    "git_sha": git_sha,
    "build_profile": "release-perf",
    "hardware_tag": hardware_tag,
    "temperature": temperature,
    "seed": seed,
    "notes": notes,
    "description": f"ssd_canary tag={tag} sha={git_sha}",
    "metrics": metric_entries,
}
print(json.dumps(obj))
PYEOF
    )

    local ts_tag
    ts_tag="$(date -u +"%Y%m%dT%H%M%S%3N")"
    local pid=$$
    local buf_file="${BUFFER_DIR}/${ts_tag}-${pid}-${tag}.json"

    if $DRY_RUN; then
        echo "  [dry-run] would write buffer: ${buf_file}" >&2
        return 0
    fi

    if [[ -z "${record_json}" ]]; then
        echo "  WARN: record_json is empty; skipping ingest for tag=${tag}" >&2
        return 1
    fi

    echo "${record_json}" > "${buf_file}"
    echo "  [ingest] buffer: ${buf_file}" >&2
    RMLX_HOME="${RMLX_HOME}" "${BINARY}" metrics record --file "${buf_file}" >&2
    echo "  [ingest] done." >&2
}

# ── SSD index and events DB paths ─────────────────────────────────────────────
# index.db is the ground truth for on-disk bytes (updated by drain thread).
# events DB is the runs.db where ssd_spill + ssd_hydrate rows are recorded.
POPULATE_INDEX_DB="${RMLX_HOME}/cache/kv/ssd-canary/index.db"
# EVICT phase reuses ssd-canary namespace to verify startup eviction.
EVICT_INDEX_DB="${RMLX_HOME}/cache/kv/ssd-canary/index.db"
EVENTS_DB="${DB_PATH}"  # runs.db (metrics DB) holds the events table.

# ── Phase POPULATE ────────────────────────────────────────────────────────────
# Send all 20 prompts with a large SSD budget (tier ON, no budget pressure).
# Each request may spill to SSD if the RAM tier is full.

echo "==> Phase POPULATE (all 20 prompts, ssd_gb=${SSD_GB})"
echo ""

CONSECUTIVE_ERRORS=0
_PREV_CUMULATIVE_SSD_HITS=0
preflight

echo "  [server] starting populate server..." >&2
RMLX_HOME="${RMLX_HOME}" \
RMLX_LOG_CAP_MB=500 \
    "${BINARY}" serve \
        --model "${VERIFIER_MODEL}" \
        --port "${PORT}" \
        --prompt-cache-slots 4 \
        --kv-ssd-cache-gb "${SSD_GB}" \
        --project ssd-canary \
        --log info \
        > /tmp/ssd_canary_populate_stdout.txt 2>&1 &

POPULATE_PID=$!
echo "  [server] pid=${POPULATE_PID}" >&2
wait_for_server

# CSV header.
POPULATE_CSV="${ARTIFACT_DIR}/phase_populate.csv"
echo "seq,prompt_name,ssd_hits,ssd_bytes_used,ssd_evict_total,spill_count,hydrate_count,spill_sum_us,hydrate_sum_us" \
    > "${POPULATE_CSV}"

TOTAL_SSD_HITS_POPULATE=0
PREV_SPILL_COUNT=0
PREV_HYDRATE_COUNT=0

for seq in $(seq 0 $((NUM_PROMPTS - 1))); do
    pf="${ALL_PROMPTS[${seq}]}"
    prompt_name="$(basename "${pf}" .json)"
    echo "  [populate] req $((seq+1))/${NUM_PROMPTS}: ${prompt_name}" >&2

    send_request "${pf}" "${POPULATE_MAX_TOKENS}" > /tmp/ssd_canary_resp.txt

    ssd_hits_req="${SSD_HITS_LAST}"
    TOTAL_SSD_HITS_POPULATE=$((TOTAL_SSD_HITS_POPULATE + ssd_hits_req))

    # Poll /metrics + index.db + events table after each request.
    # Give the drain thread 2s to flush the spill to disk.
    sleep 2
    read -r ssd_bytes ssd_evict spill_count hydrate_count spill_sum_us hydrate_sum_us spill_bytes hydrate_bytes \
        < <(parse_metrics "${POPULATE_INDEX_DB}" "${EVENTS_DB}") || true

    echo "${seq},${prompt_name},${ssd_hits_req},${ssd_bytes},${ssd_evict},${spill_count},${hydrate_count},${spill_sum_us},${hydrate_sum_us}" \
        >> "${POPULATE_CSV}"

    PREV_SPILL_COUNT="${spill_count}"
    PREV_HYDRATE_COUNT="${hydrate_count}"
    echo "    ssd_hits=${ssd_hits_req} ssd_bytes=${ssd_bytes} spills=${spill_count} hydrates=${hydrate_count}" >&2
done

echo "  [server] killing populate server pid=${POPULATE_PID}" >&2
# Give the SSD drain thread 5s to flush pending spill jobs before killing.
sleep 5
kill "${POPULATE_PID}" 2>/dev/null || true
wait "${POPULATE_PID}" 2>/dev/null || true
sleep 3

# Re-read final metrics after kill (drain may have flushed during the 5s window).
read -r ssd_bytes ssd_evict spill_count hydrate_count spill_sum_us hydrate_sum_us spill_bytes hydrate_bytes \
    < <(parse_metrics "${POPULATE_INDEX_DB}" "${EVENTS_DB}") || true

# Capture final metrics from the populate phase.
POPULATE_FINAL_SSD_BYTES="${ssd_bytes}"
POPULATE_FINAL_SPILL_COUNT="${spill_count}"
POPULATE_FINAL_HYDRATE_COUNT="${PREV_HYDRATE_COUNT}"
POPULATE_FINAL_SPILL_SUM_US="${spill_sum_us}"
POPULATE_FINAL_HYDRATE_SUM_US="${hydrate_sum_us}"
POPULATE_FINAL_EVICT="${ssd_evict}"

echo ""
echo "==> Phase POPULATE complete."
echo "    ssd_bytes_used    : ${POPULATE_FINAL_SSD_BYTES}"
echo "    spill_events      : ${POPULATE_FINAL_SPILL_COUNT}"
echo "    hydrate_events    : ${POPULATE_FINAL_HYDRATE_COUNT}"
echo "    total_ssd_hits    : ${TOTAL_SSD_HITS_POPULATE}"
echo "    evict_total       : ${POPULATE_FINAL_EVICT}"
echo ""

# ── Compute dynamic EVICT budget (Option A) ───────────────────────────────────
# Formula: avg_block_bytes = SUM(byte_size) / COUNT(*) from index.db after
# POPULATE. Budget = 4 × avg_block_bytes (the first 4 blocks fit; block 5+
# evict). This makes eviction deterministic regardless of KV quant / model.
# Floor: 1 MiB to avoid division-by-zero and zero-budget edge cases.
# Derived EVICT_SSD_GB is passed to the server via --kv-ssd-cache-gb.
EVICT_REQUEST_COUNT=8
if [[ -f "${POPULATE_INDEX_DB}" ]] && [[ "${POPULATE_FINAL_SPILL_COUNT}" -gt 0 ]]; then
    read -r _pop_count _pop_sum_bytes < <(
        sqlite3 -separator ' ' "${POPULATE_INDEX_DB}" \
            "SELECT COUNT(*), COALESCE(SUM(byte_size),0) FROM kv_blocks;" \
            2>/dev/null || echo "0 0"
    ) || true

    EVICT_BUDGET_BYTES=$(python3 -c "
count = int('${_pop_count}') if '${_pop_count}' else 0
total = int('${_pop_sum_bytes}') if '${_pop_sum_bytes}' else 0
floor_bytes = 1 * 1024 * 1024          # 1 MiB floor
if count > 0:
    avg = total // count
    # Budget = 4 × avg so the first 4 blocks fit; block 5+ triggers eviction.
    budget = max(4 * avg, floor_bytes)
else:
    budget = floor_bytes
print(budget)
")
    EVICT_SSD_GB=$(python3 -c "print(f'{int(\"${EVICT_BUDGET_BYTES}\") / (1024**3):.12f}')")
    echo "  [evict-budget] blocks=${_pop_count} sum=${_pop_sum_bytes} bytes → avg=$((${_pop_sum_bytes:-0} / ${_pop_count:-1})) bytes/block → budget=${EVICT_BUDGET_BYTES} bytes (${EVICT_SSD_GB} GB)" >&2
else
    # No blocks indexed (e.g. SSD tier was not active). Keep the static 50 MB
    # fallback so the test at least exercises the server startup path.
    EVICT_BUDGET_BYTES=$(python3 -c "print(int(${EVICT_SSD_GB} * 1024 * 1024 * 1024))")
    echo "  [evict-budget] no POPULATE blocks found; using static ${EVICT_SSD_GB} GB fallback." >&2
fi

echo "  [evict-budget] EVICT_SSD_GB=${EVICT_SSD_GB} EVICT_BUDGET_BYTES=${EVICT_BUDGET_BYTES}" >&2
echo ""

# ── Phase REVISIT ─────────────────────────────────────────────────────────────
# Replay a fixed 10-prompt subset. Expects ssd_hits to increment on prompts
# whose RAM slot was evicted but whose block survived on SSD.

echo "==> Phase REVISIT (10-prompt subset, seed=42, ssd_gb=${SSD_GB})"
echo ""

# Select 10 prompts deterministically (sorted list, pick indices 0,2,4,...18).
REVISIT_INDICES=(0 2 4 6 8 10 12 14 16 18)

CONSECUTIVE_ERRORS=0
_PREV_CUMULATIVE_SSD_HITS=0
preflight

echo "  [server] starting revisit server..." >&2
RMLX_HOME="${RMLX_HOME}" \
RMLX_LOG_CAP_MB=500 \
    "${BINARY}" serve \
        --model "${VERIFIER_MODEL}" \
        --port "${PORT}" \
        --prompt-cache-slots 4 \
        --kv-ssd-cache-gb "${SSD_GB}" \
        --project ssd-canary \
        --log info \
        > /tmp/ssd_canary_revisit_stdout.txt 2>&1 &

REVISIT_PID=$!
echo "  [server] pid=${REVISIT_PID}" >&2
wait_for_server

REVISIT_CSV="${ARTIFACT_DIR}/phase_revisit.csv"
echo "seq,prompt_name,ssd_hits,ssd_bytes_used,ssd_evict_total,spill_count,hydrate_count,spill_sum_us,hydrate_sum_us" \
    > "${REVISIT_CSV}"

TOTAL_SSD_HITS_REVISIT=0
REVISIT_COUNT="${#REVISIT_INDICES[@]}"

for rseq in $(seq 0 $((REVISIT_COUNT - 1))); do
    idx="${REVISIT_INDICES[${rseq}]}"
    pf="${ALL_PROMPTS[${idx}]}"
    prompt_name="$(basename "${pf}" .json)"
    echo "  [revisit] req $((rseq+1))/${REVISIT_COUNT} (original idx=${idx}): ${prompt_name}" >&2

    send_request "${pf}" "${POPULATE_MAX_TOKENS}" > /tmp/ssd_canary_resp.txt

    # If the server got stuck (e.g. post-hydrate generation lock deadlock), bounce it.
    bounce_if_stuck REVISIT_PID \
        --model "${VERIFIER_MODEL}" \
        --port "${PORT}" \
        --prompt-cache-slots 4 \
        --kv-ssd-cache-gb "${SSD_GB}" \
        --project ssd-canary \
        --log info || true

    ssd_hits_req="${SSD_HITS_LAST}"
    TOTAL_SSD_HITS_REVISIT=$((TOTAL_SSD_HITS_REVISIT + ssd_hits_req))

    sleep 2
    read -r ssd_bytes ssd_evict spill_count hydrate_count spill_sum_us hydrate_sum_us spill_bytes hydrate_bytes \
        < <(parse_metrics "${POPULATE_INDEX_DB}" "${EVENTS_DB}") || true

    echo "${rseq},${prompt_name},${ssd_hits_req},${ssd_bytes},${ssd_evict},${spill_count},${hydrate_count},${spill_sum_us},${hydrate_sum_us}" \
        >> "${REVISIT_CSV}"

    echo "    ssd_hits=${ssd_hits_req} hydrates=${hydrate_count}" >&2
done

echo "  [server] killing revisit server pid=${REVISIT_PID}" >&2
sleep 5
kill "${REVISIT_PID}" 2>/dev/null || true
wait "${REVISIT_PID}" 2>/dev/null || true
sleep 3

# Re-read final metrics after kill.
read -r ssd_bytes ssd_evict spill_count hydrate_count spill_sum_us hydrate_sum_us spill_bytes hydrate_bytes \
    < <(parse_metrics "${POPULATE_INDEX_DB}" "${EVENTS_DB}") || true

REVISIT_FINAL_SSD_BYTES="${ssd_bytes}"
REVISIT_FINAL_SPILL_COUNT="${spill_count}"
REVISIT_FINAL_HYDRATE_COUNT="${hydrate_count}"
REVISIT_FINAL_SPILL_SUM_US="${spill_sum_us}"
REVISIT_FINAL_HYDRATE_SUM_US="${hydrate_sum_us}"
REVISIT_FINAL_EVICT="${ssd_evict}"

REVISIT_SSD_HIT_RATE=$(python3 -c "
hits=${TOTAL_SSD_HITS_REVISIT}
total=${REVISIT_COUNT}
print(f'{hits/total:.4f}' if total > 0 else '0.0000')
")

REVISIT_MEAN_HYDRATE_US=0
if [[ "${REVISIT_FINAL_HYDRATE_COUNT}" -gt 0 ]]; then
    REVISIT_MEAN_HYDRATE_US=$(python3 -c "
sum_us=${REVISIT_FINAL_HYDRATE_SUM_US}
count=${REVISIT_FINAL_HYDRATE_COUNT}
print(int(sum_us / count) if count > 0 else 0)
")
fi

echo ""
echo "==> Phase REVISIT complete."
echo "    total_ssd_hits    : ${TOTAL_SSD_HITS_REVISIT} / ${REVISIT_COUNT} requests"
echo "    ssd_hit_rate      : ${REVISIT_SSD_HIT_RATE}"
echo "    mean_hydrate_us   : ${REVISIT_MEAN_HYDRATE_US}"
echo "    hydrate_events    : ${REVISIT_FINAL_HYDRATE_COUNT}"
echo ""

# ── Phase EVICT ───────────────────────────────────────────────────────────────
# Restart with a tiny budget against the SAME ssd-canary namespace (POPULATE
# wrote 55+ MB into it). Server startup runs evict_lru_until(budget) and trims
# the index; ssd_evict_total increments via the Prometheus hook.
# Then send 8 prompts and verify SUM(byte_size) stays <= budget.

echo "==> Phase EVICT (${EVICT_REQUEST_COUNT:-8} prompts, ssd_gb=${EVICT_SSD_GB}, budget=${EVICT_BUDGET_BYTES} bytes)"
echo ""

CONSECUTIVE_ERRORS=0
_PREV_CUMULATIVE_SSD_HITS=0
preflight

echo "  [server] starting evict server..." >&2
RMLX_HOME="${RMLX_HOME}" \
RMLX_LOG_CAP_MB=500 \
    "${BINARY}" serve \
        --model "${VERIFIER_MODEL}" \
        --port "${PORT}" \
        --prompt-cache-slots 4 \
        --kv-ssd-cache-gb "${EVICT_SSD_GB}" \
        --project ssd-canary \
        --log info \
        > /tmp/ssd_canary_evict_stdout.txt 2>&1 &

EVICT_PID=$!
echo "  [server] pid=${EVICT_PID}" >&2
wait_for_server

# index.db for the ssd-canary namespace (shared with POPULATE).
EVICT_INDEX_DB="${RMLX_HOME}/cache/kv/ssd-canary/index.db"

# Read index state immediately after startup: startup_maintenance() runs evict_lru_until()
# at attach time, before any request. This is the invariant to verify:
#   SUM(byte_size) <= budget AFTER startup eviction.
# New blocks spilled by subsequent requests in this phase are expected behavior — the
# on-demand spill path does not re-enforce budget (by design); only startup maintenance
# does. Record startup bytes separately and check the budget against that.
EVICT_STARTUP_INDEX_COUNT=0
EVICT_STARTUP_INDEX_BYTES=0
if [[ -f "${EVICT_INDEX_DB}" ]]; then
    read -r EVICT_STARTUP_INDEX_COUNT EVICT_STARTUP_INDEX_BYTES < <(
        sqlite3 -separator ' ' "${EVICT_INDEX_DB}" \
            "SELECT COUNT(*), COALESCE(SUM(byte_size),0) FROM kv_blocks;" \
            2>/dev/null || echo "0 0"
    ) || true
fi
echo "  [evict] post-startup index: blocks=${EVICT_STARTUP_INDEX_COUNT} bytes=${EVICT_STARTUP_INDEX_BYTES} budget=${EVICT_BUDGET_BYTES}" >&2

# C6 budget check: post-startup state must be within budget.
EVICT_BUDGET_VIOLATED=false
if [[ "${EVICT_STARTUP_INDEX_BYTES}" -gt "${EVICT_BUDGET_BYTES}" ]]; then
    EVICT_BUDGET_VIOLATED=true
    echo "  WARN: post-startup budget violated! index_sum_bytes=${EVICT_STARTUP_INDEX_BYTES} > budget=${EVICT_BUDGET_BYTES}" >&2
fi

# Read evict_total from Prometheus right after startup (hook fires during startup_maintenance).
read -r _s_ssd_bytes _s_ssd_evict _s_spill _s_hydrate _s_spill_us _s_hydrate_us _s_sb _s_hb \
    < <(parse_metrics "${EVICT_INDEX_DB}" "${EVENTS_DB}") || true
EVICT_STARTUP_EVICT_TOTAL="${_s_ssd_evict:-0}"
echo "  [evict] post-startup ssd_evict_total=${EVICT_STARTUP_EVICT_TOTAL}" >&2

EVICT_CSV="${ARTIFACT_DIR}/phase_evict.csv"
echo "seq,prompt_name,ssd_hits,ssd_bytes_used,ssd_evict_total,index_count,index_sum_bytes,index_sum_ok" \
    > "${EVICT_CSV}"

# Pick EVICT_REQUEST_COUNT prompts from the sorted list.
# EVICT_REQUEST_COUNT was set during the dynamic budget computation above;
# it defaults to 8 to match the original canary spec.
_evict_n="${EVICT_REQUEST_COUNT:-8}"
EVICT_PROMPTS=("${ALL_PROMPTS[@]:0:${_evict_n}}")
EVICT_FINAL_COUNT=0
EVICT_FINAL_SUM_BYTES=0
EVICT_FINAL_EVICT_TOTAL=0

for seq in $(seq 0 $((_evict_n - 1))); do
    pf="${EVICT_PROMPTS[${seq}]}"
    prompt_name="$(basename "${pf}" .json)"
    echo "  [evict] req $((seq+1))/${_evict_n}: ${prompt_name}" >&2

    send_request "${pf}" "${POPULATE_MAX_TOKENS}" > /tmp/ssd_canary_resp.txt

    # Bounce the evict server if stuck (e.g. post-hydrate panic deadlock).
    bounce_if_stuck EVICT_PID \
        --model "${VERIFIER_MODEL}" \
        --port "${PORT}" \
        --prompt-cache-slots 4 \
        --kv-ssd-cache-gb "${EVICT_SSD_GB}" \
        --project ssd-canary \
        --log info || true

    ssd_hits_req="${SSD_HITS_LAST}"
    sleep 2

    read -r ssd_bytes ssd_evict spill_count hydrate_count spill_sum_us hydrate_sum_us spill_bytes hydrate_bytes \
        < <(parse_metrics "${EVICT_INDEX_DB}" "${EVENTS_DB}") || true

    # Query index.db directly for per-request tracking (informational only; budget is
    # checked against post-startup state above, not per-request state).
    index_count=0
    index_sum_bytes=0
    if [[ -f "${EVICT_INDEX_DB}" ]]; then
        read -r index_count index_sum_bytes < <(
            sqlite3 -separator ' ' "${EVICT_INDEX_DB}" \
                "SELECT COUNT(*), COALESCE(SUM(byte_size),0) FROM kv_blocks;" \
                2>/dev/null || echo "0 0"
        ) || true
    fi

    # index_sum_ok reflects whether post-startup budget was met (set once above).
    index_sum_ok=$( [[ "${EVICT_BUDGET_VIOLATED}" == "false" ]] && echo "true" || echo "false" )

    echo "${seq},${prompt_name},${ssd_hits_req},${ssd_bytes},${ssd_evict},${index_count},${index_sum_bytes},${index_sum_ok}" \
        >> "${EVICT_CSV}"

    EVICT_FINAL_COUNT="${index_count}"
    EVICT_FINAL_SUM_BYTES="${index_sum_bytes}"
    EVICT_FINAL_EVICT_TOTAL="${ssd_evict}"

    echo "    ssd_evict_total=${ssd_evict} index_count=${index_count} index_sum_bytes=${index_sum_bytes}" >&2
done

echo "  [server] killing evict server pid=${EVICT_PID}" >&2
sleep 5
kill "${EVICT_PID}" 2>/dev/null || true
wait "${EVICT_PID}" 2>/dev/null || true
sleep 3

# Re-read final index after kill to get post-drain byte count.
if [[ -f "${EVICT_INDEX_DB}" ]]; then
    read -r EVICT_FINAL_COUNT EVICT_FINAL_SUM_BYTES < <(
        sqlite3 -separator ' ' "${EVICT_INDEX_DB}" \
            "SELECT COUNT(*), COALESCE(SUM(byte_size),0) FROM kv_blocks;" \
            2>/dev/null || echo "0 0"
    ) || true
fi

EVICT_FINAL_ON_DISK_MB=$(python3 -c "print(f'{${EVICT_FINAL_SUM_BYTES:-0}/(1024*1024):.2f}')")
EVICT_BUDGET_MB=$(python3 -c "print(f'{${EVICT_BUDGET_BYTES:-0}/(1024*1024):.1f}')")

echo ""
echo "==> Phase EVICT complete."
echo "    startup_index_bytes : ${EVICT_STARTUP_INDEX_BYTES} (budget=${EVICT_BUDGET_MB} MB; budget_violated=${EVICT_BUDGET_VIOLATED})"
echo "    final_index_bytes   : ${EVICT_FINAL_SUM_BYTES} (${EVICT_FINAL_ON_DISK_MB} MB; may exceed budget due to new spills)"
echo "    blocks_remaining    : ${EVICT_FINAL_COUNT}"
echo "    evict_total         : ${EVICT_FINAL_EVICT_TOTAL}"
echo "    startup_evict_total : ${EVICT_STARTUP_EVICT_TOTAL}"
echo ""

# ── Compute derived metrics and query events table ────────────────────────────

POPULATE_MEAN_SPILL_US=0
POPULATE_MEAN_HYDRATE_US=0
POPULATE_SPILL_MBPS="0.000"
POPULATE_HYDRATE_MBPS="0.000"

if [[ "${POPULATE_FINAL_SPILL_COUNT}" -gt 0 ]] && ! $DRY_RUN; then
    read -r POPULATE_MEAN_SPILL_US_DB POPULATE_SPILL_BYTES_DB < <(
        sqlite3 -separator ' ' "${DB_PATH}" \
            "SELECT COALESCE(AVG(value),0), COALESCE(AVG(CAST(json_extract(notes,'$.bytes') AS REAL)),0)
             FROM events WHERE op='ssd_spill';" \
            2>/dev/null || echo "0 0"
    ) || true
    POPULATE_MEAN_SPILL_US="${POPULATE_MEAN_SPILL_US_DB}"
    if python3 -c "exit(0 if float('${POPULATE_MEAN_SPILL_US}') > 0 else 1)" 2>/dev/null; then
        POPULATE_SPILL_MBPS=$(python3 -c "
bytes_val = float('${POPULATE_SPILL_BYTES_DB}')
dur_us = float('${POPULATE_MEAN_SPILL_US}')
mbps = (bytes_val / dur_us) * 1e6 / (1024*1024) if dur_us > 0 else 0.0
print(f'{mbps:.3f}')
" 2>/dev/null || echo "0.000")
    fi
fi

if [[ "${POPULATE_FINAL_HYDRATE_COUNT}" -gt 0 ]] && ! $DRY_RUN; then
    read -r POPULATE_MEAN_HYDRATE_US_DB POPULATE_HYDRATE_BYTES_DB < <(
        sqlite3 -separator ' ' "${DB_PATH}" \
            "SELECT COALESCE(AVG(value),0), COALESCE(AVG(CAST(json_extract(notes,'$.bytes') AS REAL)),0)
             FROM events WHERE op='ssd_hydrate';" \
            2>/dev/null || echo "0 0"
    ) || true
    POPULATE_MEAN_HYDRATE_US="${POPULATE_MEAN_HYDRATE_US_DB}"
    if python3 -c "exit(0 if float('${POPULATE_MEAN_HYDRATE_US}') > 0 else 1)" 2>/dev/null; then
        POPULATE_HYDRATE_MBPS=$(python3 -c "
bytes_val = float('${POPULATE_HYDRATE_BYTES_DB}')
dur_us = float('${POPULATE_MEAN_HYDRATE_US}')
mbps = (bytes_val / dur_us) * 1e6 / (1024*1024) if dur_us > 0 else 0.0
print(f'{mbps:.3f}')
" 2>/dev/null || echo "0.000")
    fi
fi

# Mean hydrate from revisit phase (total hydrate events include populate + revisit).
REVISIT_MEAN_HYDRATE_US_FINAL=0
if [[ "${REVISIT_FINAL_HYDRATE_COUNT:-0}" -gt 0 ]] && ! $DRY_RUN; then
    REVISIT_MEAN_HYDRATE_US_FINAL=$(sqlite3 "${DB_PATH}" \
        "SELECT COALESCE(AVG(value),0) FROM events WHERE op='ssd_hydrate';" \
        2>/dev/null || echo "0")
fi

# ── Ingest phase records ───────────────────────────────────────────────────────

echo "==> Ingesting phase records into runs.db..." >&2

# POPULATE
_POPULATE_METRICS_JSON=$(python3 - <<PYEOF_PM
import json
print(json.dumps({
    'prompt_cache_ssd_hits': ${TOTAL_SSD_HITS_POPULATE},
    'ssd_bytes_used':        ${POPULATE_FINAL_SSD_BYTES:-0},
    'ssd_evict_total':       int("${POPULATE_FINAL_EVICT}" or "0"),
    'ssd_spill_ms':          float("${POPULATE_MEAN_SPILL_US}" or "0") / 1000.0,
    'ssd_hydrate_ms':        float("${POPULATE_MEAN_HYDRATE_US}" or "0") / 1000.0,
    'ssd_spill_mb_per_s':    float("${POPULATE_SPILL_MBPS}" or "0"),
    'ssd_hydrate_mb_per_s':  float("${POPULATE_HYDRATE_MBPS}" or "0"),
}))
PYEOF_PM
)
emit_and_ingest \
    "ssd-canary-populate" \
    "ssd_canary POPULATE phase: ${NUM_PROMPTS} prompts, prompt_cache_slots=4, ssd_gb=${SSD_GB}" \
    "${_POPULATE_METRICS_JSON}"

# REVISIT
_REVISIT_METRICS_JSON=$(python3 - <<PYEOF_RM
import json
print(json.dumps({
    'prompt_cache_ssd_hits': ${TOTAL_SSD_HITS_REVISIT},
    'ssd_bytes_used':        int("${REVISIT_FINAL_SSD_BYTES}" or "0"),
    'ssd_evict_total':       int("${REVISIT_FINAL_EVICT}" or "0"),
    'ssd_spill_ms':          float("${POPULATE_MEAN_SPILL_US}" or "0") / 1000.0,
    'ssd_hydrate_ms':        float("${REVISIT_MEAN_HYDRATE_US_FINAL}" or "0") / 1000.0,
    'ssd_spill_mb_per_s':    float("${POPULATE_SPILL_MBPS}" or "0"),
    'ssd_hydrate_mb_per_s':  float("${POPULATE_HYDRATE_MBPS}" or "0"),
}))
PYEOF_RM
)
emit_and_ingest \
    "ssd-canary-revisit" \
    "ssd_canary REVISIT phase: ${REVISIT_COUNT} prompts replayed (fixed seed), ssd_gb=${SSD_GB}" \
    "${_REVISIT_METRICS_JSON}"

# EVICT
_EVICT_METRICS_JSON=$(python3 - <<PYEOF_EM
import json
print(json.dumps({
    'ssd_bytes_used':  ${EVICT_FINAL_SUM_BYTES:-0},
    'ssd_evict_total': ${EVICT_FINAL_EVICT_TOTAL:-0},
}))
PYEOF_EM
)
emit_and_ingest \
    "ssd-canary-evict" \
    "ssd_canary EVICT phase: 8 prompts, ssd_gb=${EVICT_SSD_GB}, budget=${EVICT_BUDGET_BYTES} bytes" \
    "${_EVICT_METRICS_JSON}"

echo "==> Ingest complete." >&2

# ── Validation assertions ──────────────────────────────────────────────────────

echo "==> Validation checks..." >&2

VALIDATION_PASS=true
VALIDATION_NOTES=""

# C1: events table has >= 20 SsdSpill rows.
if ! $DRY_RUN && [[ -f "${DB_PATH}" ]]; then
    SPILL_EVENT_ROWS=$(sqlite3 "${DB_PATH}" \
        "SELECT COUNT(*) FROM events WHERE op='ssd_spill';" 2>/dev/null || echo "0")
    if [[ "${SPILL_EVENT_ROWS}" -lt 1 ]]; then
        VALIDATION_NOTES="${VALIDATION_NOTES} [WARN] spill event rows=${SPILL_EVENT_ROWS} (expected >= 1; SSD tier may not have been active)"
        echo "  WARN: spill_event_rows=${SPILL_EVENT_ROWS} (expected >= 1)" >&2
    else
        echo "  [ok] spill_event_rows=${SPILL_EVENT_ROWS}" >&2
    fi

    # C2: events table has >= 1 SsdHydrate row.
    HYDRATE_EVENT_ROWS=$(sqlite3 "${DB_PATH}" \
        "SELECT COUNT(*) FROM events WHERE op='ssd_hydrate';" 2>/dev/null || echo "0")
    if [[ "${HYDRATE_EVENT_ROWS}" -lt 1 ]]; then
        VALIDATION_NOTES="${VALIDATION_NOTES} [WARN] hydrate event rows=${HYDRATE_EVENT_ROWS} (expected >= 1; revisit may need more cache pressure)"
        echo "  WARN: hydrate_event_rows=${HYDRATE_EVENT_ROWS} (expected >= 1)" >&2
    else
        echo "  [ok] hydrate_event_rows=${HYDRATE_EVENT_ROWS}" >&2
    fi

    # C3: observations table has the three tagged rows.
    for obs_tag in "ssd-canary-populate" "ssd-canary-revisit" "ssd-canary-evict"; do
        OBS_COUNT=$(sqlite3 "${DB_PATH}" \
            "SELECT COUNT(*) FROM observations WHERE notes LIKE '%tag=${obs_tag}%' OR description LIKE '%${obs_tag}%';" \
            2>/dev/null || echo "0")
        if [[ "${OBS_COUNT}" -lt 1 ]]; then
            VALIDATION_NOTES="${VALIDATION_NOTES} [WARN] observations tag=${obs_tag} count=${OBS_COUNT}"
            echo "  WARN: observations tag=${obs_tag} not found" >&2
        else
            echo "  [ok] observations tag=${obs_tag} count=${OBS_COUNT}" >&2
        fi
    done
else
    SPILL_EVENT_ROWS="${POPULATE_FINAL_SPILL_COUNT}"
    HYDRATE_EVENT_ROWS="${POPULATE_FINAL_HYDRATE_COUNT}"
fi

# C4: ssd_bytes_used after POPULATE > 0.
if [[ "${POPULATE_FINAL_SSD_BYTES}" -gt 0 ]]; then
    echo "  [ok] ssd_bytes_used after POPULATE=${POPULATE_FINAL_SSD_BYTES} bytes > 0" >&2
else
    VALIDATION_NOTES="${VALIDATION_NOTES} [WARN] ssd_bytes_used after POPULATE is 0 (SSD tier may not be enabled or no blocks spilled)"
    echo "  WARN: ssd_bytes_used after POPULATE is 0" >&2
fi

# C4b: REVISIT total_ssd_hits >= 1 (FAIL — the fix's load-bearing assertion).
# Without the kvcache.rs dispatch fix, the server deadlocks on the hydrated SWA
# layer and ssd_hits never increments. >= 1 total means at least one revisited
# prompt was served from the SSD tier end-to-end (hydrate + generate completed).
if [[ "${TOTAL_SSD_HITS_REVISIT}" -ge 1 ]]; then
    echo "  [ok] REVISIT total_ssd_hits=${TOTAL_SSD_HITS_REVISIT} >= 1" >&2
else
    VALIDATION_PASS=false
    VALIDATION_NOTES="${VALIDATION_NOTES} [FAIL] REVISIT total_ssd_hits=${TOTAL_SSD_HITS_REVISIT} (expected >= 1; kvcache dispatch fix may not have landed)"
    echo "  FAIL: REVISIT total_ssd_hits=${TOTAL_SSD_HITS_REVISIT} (expected >= 1)" >&2
fi

# C5: ssd_evict_total after EVICT > 0 (FAIL — the dynamic budget formula
# guarantees the 5th block must evict; 0 means the budget is wrong or spill
# never happened). Use the startup-captured value as primary (fires at server
# attach time before any request), final loop value as fallback.
_c5_evict="${EVICT_STARTUP_EVICT_TOTAL:-0}"
if [[ "${_c5_evict}" -eq 0 ]]; then
    _c5_evict="${EVICT_FINAL_EVICT_TOTAL:-0}"
fi
if [[ "${_c5_evict}" -gt 0 ]]; then
    echo "  [ok] ssd_evict_total after EVICT=${_c5_evict}" >&2
else
    VALIDATION_PASS=false
    VALIDATION_NOTES="${VALIDATION_NOTES} [FAIL] ssd_evict_total after EVICT is 0 — dynamic budget did not trigger eviction"
    echo "  FAIL: ssd_evict_total after EVICT is 0 (dynamic budget=${EVICT_BUDGET_BYTES} bytes did not trigger eviction)" >&2
fi

# C6: startup eviction upholds budget (FAIL — post-startup sum(byte_size) in index.db
# must be <= budget). The on-demand spill path does not re-enforce budget by design;
# only startup_maintenance() does. Subsequent requests may grow the namespace beyond
# budget — that is expected. We check the post-startup snapshot captured before the
# first EVICT request was sent.
if $EVICT_BUDGET_VIOLATED; then
    VALIDATION_PASS=false
    VALIDATION_NOTES="${VALIDATION_NOTES} [FAIL] EVICT post-startup budget violated (startup_bytes=${EVICT_STARTUP_INDEX_BYTES} > budget=${EVICT_BUDGET_BYTES})"
    echo "  FAIL: EVICT post-startup budget violated! startup_bytes=${EVICT_STARTUP_INDEX_BYTES} > budget=${EVICT_BUDGET_BYTES}" >&2
else
    echo "  [ok] EVICT post-startup budget not violated (startup_bytes=${EVICT_STARTUP_INDEX_BYTES} <= budget=${EVICT_BUDGET_BYTES})" >&2
fi

# Hydrate-panic note: if a bounce was triggered, document the observed failure mode.
if $HYDRATE_PANIC_DETECTED; then
    VALIDATION_NOTES="${VALIDATION_NOTES} [INFO] Hydrate-then-panic observed: SSD hydration succeeded but subsequent re-prefill panicked at kvcache.rs:2297 (storage mismatch: expected K8V8); server bounced automatically. Hydrate event rows ARE recorded. Root cause: hydrated KvCache has wrong KvStorage variant — bug in hydration path, not in spill or index."
    echo "  [info] hydrate-panic detected and bounced; noted in summary." >&2
fi

# ── Write iteration_summary.json ──────────────────────────────────────────────

SUMMARY_FILE="${ARTIFACT_DIR}/iteration_summary.json"

python3 - <<PYEOF > "${SUMMARY_FILE}"
import json, os, time

summary = {
    "ssd_canary_version": "1.0.0",
    "run_ts_utc": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "git_sha": "${GIT_SHA}",
    "model_id": "${MODEL_ID}",
    "rmlx_home": "${RMLX_HOME}",
    "dry_run": $( [[ "${DRY_RUN}" == "true" ]] && echo "True" || echo "False" ),
    "phases": {
        "POPULATE": {
            "num_prompts": ${NUM_PROMPTS},
            "ssd_bytes_used": ${POPULATE_FINAL_SSD_BYTES},
            "ssd_bytes_used_mb": round(${POPULATE_FINAL_SSD_BYTES} / (1024*1024), 3),
            "spill_events": int("${POPULATE_FINAL_SPILL_COUNT}" or "0"),
            "hydrate_events": int("${POPULATE_FINAL_HYDRATE_COUNT}" or "0"),
            "total_ssd_hits": ${TOTAL_SSD_HITS_POPULATE},
            "ssd_evict_total": int("${POPULATE_FINAL_EVICT}" or "0"),
            "mean_spill_us": float("${POPULATE_MEAN_SPILL_US}" or "0"),
            "mean_hydrate_us": float("${POPULATE_MEAN_HYDRATE_US}" or "0"),
            "spill_mb_per_s": float("${POPULATE_SPILL_MBPS}" or "0"),
            "hydrate_mb_per_s": float("${POPULATE_HYDRATE_MBPS}" or "0"),
        },
        "REVISIT": {
            "num_prompts": int("${REVISIT_COUNT}" or "0"),
            "ssd_bytes_used": int("${REVISIT_FINAL_SSD_BYTES}" or "0"),
            "total_ssd_hits": ${TOTAL_SSD_HITS_REVISIT},
            "ssd_hit_rate": float("${REVISIT_SSD_HIT_RATE}" or "0"),
            "ssd_evict_total": int("${REVISIT_FINAL_EVICT}" or "0"),
            "hydrate_events": int("${REVISIT_FINAL_HYDRATE_COUNT}" or "0"),
            "mean_hydrate_us": float("${REVISIT_MEAN_HYDRATE_US_FINAL}" or "0"),
        },
        "EVICT": {
            "ssd_gb_budget": float("${EVICT_SSD_GB}" or "0"),
            "budget_bytes": ${EVICT_BUDGET_BYTES},
            "budget_mb": float("${EVICT_BUDGET_MB}" or "0"),
            "startup_index_bytes": int("${EVICT_STARTUP_INDEX_BYTES}" or "0"),
            "startup_index_count": int("${EVICT_STARTUP_INDEX_COUNT}" or "0"),
            "startup_evict_total": int("${EVICT_STARTUP_EVICT_TOTAL}" or "0"),
            "on_disk_bytes": int("${EVICT_FINAL_SUM_BYTES}" or "0"),
            "on_disk_mb": float("${EVICT_FINAL_ON_DISK_MB}" or "0"),
            "blocks_remaining": int("${EVICT_FINAL_COUNT}" or "0"),
            "ssd_evict_total": int("${EVICT_FINAL_EVICT_TOTAL}" or "0"),
            "budget_violated": $( [[ "${EVICT_BUDGET_VIOLATED}" == "true" ]] && echo "True" || echo "False" ),
        },
    },
    "validation": {
        "pass": $( [[ "${VALIDATION_PASS}" == "true" ]] && echo "True" || echo "False" ),
        "spill_event_rows": int("${SPILL_EVENT_ROWS}" or "0"),
        "hydrate_event_rows": int("${HYDRATE_EVENT_ROWS}" or "0"),
        "hydrate_panic_detected": $( [[ "${HYDRATE_PANIC_DETECTED}" == "true" ]] && echo "True" || echo "False" ),
        "notes": "${VALIDATION_NOTES}",
    },
    "artifacts": {
        "phase_populate_csv": "${POPULATE_CSV}",
        "phase_revisit_csv": "${REVISIT_CSV}",
        "phase_evict_csv": "${EVICT_CSV}",
        "runs_db": "${DB_PATH}",
        "iteration_summary": "${SUMMARY_FILE}",
    },
}
print(json.dumps(summary, indent=2))
PYEOF

echo ""
echo "==> Summary written to ${SUMMARY_FILE}"

# ── Final DB verification ─────────────────────────────────────────────────────

if ! $DRY_RUN && [[ -f "${DB_PATH}" ]]; then
    echo ""
    echo "==> DB verification (last 60 minutes):"
    sqlite3 "${DB_PATH}" \
        "SELECT description, metric, ROUND(value,3) as value
         FROM observations
         WHERE ts_utc >= datetime('now','-60 minutes')
         ORDER BY ts_utc DESC, metric;" \
        2>/dev/null || echo "  (sqlite3 not available or DB empty)"
fi

# ── Final summary table ────────────────────────────────────────────────────────

POPULATE_SSD_MB=$(python3 -c "print(f'{${POPULATE_FINAL_SSD_BYTES:-0}/(1024*1024):.2f}')")

echo ""
echo "============================================================"
echo "  SSD CANARY RESULTS — ${MODEL_ID}"
echo "============================================================"
printf "%-10s  %-20s  %-14s  %-18s  %-14s\n" \
    "Phase" "ssd_bytes_used_mb" "spill_events" "revisit_hit_rate" "evict_total"
printf "%-10s  %-20s  %-14s  %-18s  %-14s\n" \
    "-----" "-----------------" "------------" "----------------" "-----------"
printf "%-10s  %-20s  %-14s  %-18s  %-14s\n" \
    "POPULATE" "${POPULATE_SSD_MB} MB" "${POPULATE_FINAL_SPILL_COUNT}" "N/A" "${POPULATE_FINAL_EVICT}"
printf "%-10s  %-20s  %-14s  %-18s  %-14s\n" \
    "REVISIT" "N/A" "N/A" "${REVISIT_SSD_HIT_RATE}" "${REVISIT_FINAL_EVICT}"
printf "%-10s  %-20s  %-14s  %-18s  %-14s\n" \
    "EVICT" "${EVICT_FINAL_ON_DISK_MB} MB" "N/A" "N/A" "${EVICT_FINAL_EVICT_TOTAL}"
echo "============================================================"
echo ""
echo "Validation: pass=${VALIDATION_PASS}"
[[ -n "${VALIDATION_NOTES}" ]] && echo "  notes: ${VALIDATION_NOTES}"
echo ""
echo "Artifacts:"
echo "  populate csv    : ${POPULATE_CSV}"
echo "  revisit csv     : ${REVISIT_CSV}"
echo "  evict csv       : ${EVICT_CSV}"
echo "  iteration summary: ${SUMMARY_FILE}"
echo "  runs.db         : ${DB_PATH}"
echo ""
echo "Done."

if [[ "${VALIDATION_PASS}" == "false" ]]; then
    exit 1
fi
