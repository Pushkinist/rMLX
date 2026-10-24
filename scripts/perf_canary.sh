#!/usr/bin/env bash
# perf_canary.sh — fast decode-TPS canary for the three standard test-target models.
# Usage: bash scripts/perf_canary.sh [--include-26b]
#        bash scripts/perf_canary.sh --ab [perf_ab.sh options]
#
# Runs 1 warmup + 3 measured baseline calls per model, prints median decode_tps
# ± sample stddev, appends one CSV row per model to
# <RMLX_HOME>/bench/perf_canary.csv, and records one further run into runs.db
# via `rmlx baseline --record --run-class gate`. Each model also gets an
# explicit `k8vturbo3` arm, which appends its own CSV row after the `auto` row.
# The warmup, the measured calls and the `k8vturbo3` arm are recorded too, as
# `diagnostic`, labelled `canary sha=<sha> <phase>`: only the one further run
# is the gate's. The one-token run that reads the codec name off the log is not
# recorded: a one-token generation has no decode rate, and the recorder refuses
# a record that carries none.
#
# Every one of those records must land, so a snapshot whose identity cannot be
# recorded (`lib/snapshot_identity.py`, which labels a snapshot as `rmlx
# baseline --record` does) stops the canary before its first run, and a run
# whose record did not land (`baseline --record` then exits non-zero) stops it
# too.
#
# The default mode measures ONE build. All of its measured runs for a model
# happen together, which is fine for tracking one build over time but not for
# comparing two: whichever arm ran second wears any drift. `--ab` hands off to
# `scripts/perf_ab.sh`, which interleaves the arms, gates on host quiescence and
# compares generated token ids across arms. Use it for every two-arm question.
#
# Column order (CSV): ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile
#
# The source of truth for canary TPS is runs.db. `make canary-gate SHA=<sha>`
# (which calls `rmlx metrics deltas`) gates regressions from the DB, and
# `scripts/regression_gate.sh` reads the CSV.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# `--ab` must be the first flag, and is dispatched before anything below runs.
# Everything after it belongs to perf_ab.sh: re-parsing those options here would
# create a second place for the two scripts to disagree about what they mean,
# and resolving the canary model paths first would make `--ab --model <path>`
# demand RMLX_O_MODELS_ROOT for models it is not going to touch.
if [[ "${1:-}" == "--ab" ]]; then
    shift
    exec bash "${REPO_ROOT}/scripts/perf_ab.sh" "$@"
fi

BINARY="${REPO_ROOT}/target/release-perf/rmlx"
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
CSV_DIR="${RMLX_HOME}/bench"
CSV="${CSV_DIR}/perf_canary.csv"
LOG_DIR="${RMLX_HOME}/logs"
SCRATCH_DIR="${RMLX_HOME}/tmp"
source "${REPO_ROOT}/scripts/lib/bench_server.sh"

PROMPT_TOKENS=4096
MAX_TOKENS=100
MAX_CTX=8192
WARMUP_RUNS=1
MEASURED_RUNS=3
BUILD_PROFILE="release-perf"

# Model definitions: "short_name|absolute_path"
GEMMA4E4B_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-e4b-it-mxfp8"
QWEN38_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.8-27B-4bit"
BONSAI2_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/prism-ml__Ternary-Bonsai-2-27B-mlx-2bit"
GEMMA4_26B_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"

INCLUDE_26B=false

# Parse flags
for arg in "$@"; do
    case "$arg" in
        --include-26b) INCLUDE_26B=true ;;
        --ab) echo "--ab must be the first argument" >&2; exit 1 ;;
        *) echo "unknown flag: $arg" >&2; exit 1 ;;
    esac
done

# Build model list
MODELS=(
    "mlx-community__gemma-4-e4b-it-mxfp8|${GEMMA4E4B_PATH}"
    "mlx-community__Qwen3.8-27B-4bit|${QWEN38_PATH}"
    "prism-ml__Ternary-Bonsai-2-27B-mlx-2bit|${BONSAI2_PATH}"
)
if $INCLUDE_26B; then
    MODELS+=("mlx-community__gemma-4-26b-a4b-it-mxfp8|${GEMMA4_26B_PATH}")
fi

# Verify binary present
if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}. Run: make build-perf" >&2
    exit 125
fi

# The measured calls hide rmlx's stderr, so a Metal claim another process holds
# is found here: the probe takes the claim and releases it, or exits 11 naming
# the holder, and the run stops with that status.
"${BINARY}" claim run -- true

# Ensure CSV dir and header
mkdir -p "${CSV_DIR}"
if [[ ! -f "${CSV}" ]]; then
    echo "ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile" > "${CSV}"
fi

source "${REPO_ROOT}/scripts/lib/identity.sh"
rmlx_export_identity "${BINARY}"
GIT_SHA="$(rmlx_git_sha "${REPO_ROOT}")"

# "unknown" is a fallback for the --label text, never provenance — a checkout
# without .git must not pass --git-sha at all.
git_sha_args=()
if [[ "${GIT_SHA}" != unknown* ]]; then
    git_sha_args=(--git-sha "${GIT_SHA}")
fi

# Helper: run one baseline, recorded as diagnostic, and parse decode_tps from stdout.
# `baseline --record` exits non-zero when its record did not land, and that
# stops the canary.
# Usage: run_once <phase> <model_path> [--kv-quant <tag>]
run_once() {
    local phase="$1"
    local model_path="$2"
    local extra_args=("${@:3}")
    local out
    if ! out="$(RMLX_HOME="${RMLX_HOME}" "${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens "${MAX_TOKENS}" \
        --max-ctx "${MAX_CTX}" \
        --label "canary sha=${GIT_SHA} ${phase}" \
        ${git_sha_args[@]+"${git_sha_args[@]}"} \
        --record --run-class diagnostic \
        ${extra_args[@]+"${extra_args[@]}"} \
        2>/dev/null)"; then
        echo "ERROR: ${model_path##*/}: the ${phase} run failed, or its record did not land (see ${LOG_DIR})" >&2
        exit 1
    fi
    printf '%s\n' "${out}" | grep -o 'decode_tps=[0-9.]*' | cut -d= -f2
}

# Which KV codec a one-token run resolves, read out of that run's own log.
#
# Read from the log rather than scraped off stderr, and through
# `lib/server_kv_quant.py` rather than a private parser: the engine states the
# codec once, in one field, and one reader owns it. The old scrape also carried
# a chain of `sed` rules translating a Rust `Debug` rendering (`K8V8`,
# `Mixed { k_bits: 8, … }`) into a tag — the field is written through `Display`
# now, so every one of those branches was dead and any that still fired would
# have written a spelling nothing else uses.
resolve_kv_quant() {
    local model_path="$1"
    mkdir -p "${LOG_DIR}" "${SCRATCH_DIR}"
    snapshot_logs

    RMLX_HOME="${RMLX_HOME}" "${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens 1 \
        --max-ctx "${MAX_CTX}" \
        > /dev/null 2>&1 &
    local probe_pid=$!
    wait "${probe_pid}" || true

    local log
    log="$(phase_log "${probe_pid}")" || return 1

    python3 "${REPO_ROOT}/scripts/lib/server_kv_quant.py" "${log}" \
        | sed -n 's/^kv_quant=//p'
}

# Compute median of space-separated values (sort via sort(1), compute via awk)
median() {
    echo "$@" | tr ' ' '\n' | sort -n | awk '
    { a[NR]=$1 }
    END {
        n=NR
        if (n % 2 == 1) print a[(n+1)/2]
        else printf "%.4f\n", (a[n/2] + a[n/2+1]) / 2
    }'
}

# Compute sample stddev of space-separated values via awk
stddev() {
    echo "$@" | tr ' ' '\n' | awk '
    NR==1 { first=$1 }
    { sum+=$1; sumsq+=$1*$1; n++ }
    END {
        if (n < 2) { print "0.0000"; exit }
        mean = sum/n
        variance = (sumsq - n*mean*mean) / (n-1)
        if (variance < 0) variance = 0
        printf "%.4f\n", sqrt(variance)
    }'
}

# Every run below records, so an identity the DB would refuse stops the canary
# here, before any model is loaded, and not at its first measured run.
for entry in "${MODELS[@]}"; do
    model_path="${entry##*|}"
    [[ -d "${model_path}" ]] || continue
    if ! python3 "${REPO_ROOT}/scripts/lib/snapshot_identity.py" "${model_path}" > /dev/null; then
        echo "ERROR: ${model_path}: its identity cannot be recorded (above); nothing was measured" >&2
        exit 1
    fi
done

# Main loop
for entry in "${MODELS[@]}"; do
    short_name="${entry%%|*}"
    model_path="${entry##*|}"

    if [[ ! -d "${model_path}" ]]; then
        echo "SKIP ${short_name}: model path not found at ${model_path}" >&2
        continue
    fi

    echo "==> ${short_name}" >&2

    # Warmup (discarded)
    for i in $(seq 1 ${WARMUP_RUNS}); do
        echo "  warmup ${i}/${WARMUP_RUNS}..." >&2
        run_once warmup "${model_path}" > /dev/null
    done

    # Measured runs
    tps_values=()
    for i in $(seq 1 ${MEASURED_RUNS}); do
        echo "  measured run ${i}/${MEASURED_RUNS}..." >&2
        val="$(run_once measured "${model_path}")"
        if [[ -z "${val}" ]]; then
            echo "ERROR: failed to parse decode_tps for ${short_name} run ${i}" >&2
            exit 1
        fi
        tps_values+=("${val}")
    done

    med="$(median "${tps_values[@]}")"
    sd="$(stddev "${tps_values[@]}")"

    # Resolve kv_quant (fast single run, max-tokens 1). The engine's own name
    # for the codec goes into the row verbatim.
    kv_quant_tag="$(resolve_kv_quant "${model_path}" 2>/dev/null || true)"
    [[ -z "${kv_quant_tag}" ]] && kv_quant_tag="auto"

    ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

    # LEGACY: Append CSV row (CSV is now secondary; runs.db is authoritative)
    echo "${ts_utc},${GIT_SHA},${short_name},${kv_quant_tag},${PROMPT_TOKENS},${med},${sd},${BUILD_PROFILE}" >> "${CSV}"

    # Record into runs.db via `rmlx baseline --record` (single representative run,
    # same protocol: prompt-tokens/max-tokens/max-ctx as the measured runs, kv_quant=auto).
    # This is the authoritative canary record; use `make canary-gate` to gate regressions.
    echo "  recording into runs.db..." >&2
    if RMLX_HOME="${RMLX_HOME}" "${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens "${MAX_TOKENS}" \
        --max-ctx "${MAX_CTX}" \
        --label "canary sha=${GIT_SHA}" \
        ${git_sha_args[@]+"${git_sha_args[@]}"} \
        --record --run-class gate \
        > /dev/null 2>&1; then
        echo "  runs.db: ok" >&2
    else
        echo "ERROR: ${short_name}: the gate record failed; the canary has no row to gate on" >&2
        exit 1
    fi

    # Print result
    printf "%s  decode_tps=%.2f ± %.2f\n" "${short_name}" "${med}" "${sd}"

    # K8VTurbo3 informational column (1 warmup + 3 measured, explicit --kv-quant)
    # for every model; `auto` resolves to bf16.
    echo "  [k8vturbo3] warmup..." >&2
    run_once k8vturbo3-warmup "${model_path}" --kv-quant k8vturbo3 > /dev/null
    turbo3_values=()
    for i in $(seq 1 ${MEASURED_RUNS}); do
        echo "  [k8vturbo3] measured run ${i}/${MEASURED_RUNS}..." >&2
        val="$(run_once k8vturbo3 "${model_path}" --kv-quant k8vturbo3)"
        if [[ -z "${val}" ]]; then
            echo "  WARN: k8vturbo3 run ${i} failed to parse decode_tps — skipping" >&2
            continue
        fi
        turbo3_values+=("${val}")
    done
    if [[ ${#turbo3_values[@]} -ge 1 ]]; then
        turbo3_med="$(median "${turbo3_values[@]}")"
        turbo3_sd="$(stddev "${turbo3_values[@]}")"
        echo "${ts_utc},${GIT_SHA},${short_name},k8vturbo3,${PROMPT_TOKENS},${turbo3_med},${turbo3_sd},${BUILD_PROFILE}" >> "${CSV}"
        printf "%s  [k8vturbo3] decode_tps=%.2f ± %.2f\n" "${short_name}" "${turbo3_med}" "${turbo3_sd}"
    fi
done

echo "" >&2
echo "LEGACY CSV: ${CSV}" >&2
echo "DB: ${RMLX_HOME}/metrics/runs.db (use 'make canary-gate SHA=<sha>' to gate)" >&2
