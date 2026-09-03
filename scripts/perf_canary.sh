#!/usr/bin/env bash
# perf_canary.sh — fast decode-TPS canary for the three standard test-target models.
# Usage: bash scripts/perf_canary.sh [--include-26b]
#        bash scripts/perf_canary.sh --ab [perf_ab.sh options]
#
# Runs 1 warmup + 3 measured baseline calls per model, prints median decode_tps
# ± sample stddev, appends one CSV row per model to .rmlx/bench/perf_canary.csv,
# and records the median run into runs.db via `rmlx baseline --record`.
#
# The default mode measures ONE build. All of its measured runs for a model
# happen together, which is fine for tracking one build over time but not for
# comparing two: whichever arm ran second wears any drift. `--ab` hands off to
# `scripts/perf_ab.sh`, which interleaves the arms, gates on host quiescence and
# compares generated token ids across arms. Use it for every two-arm question.
#
# Column order (CSV): ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile
#
# MIGRATION NOTE: The CSV (.rmlx/bench/perf_canary.csv) is now a LEGACY
# fallback. The authoritative source-of-truth for canary TPS is runs.db (via
# `rmlx baseline --record`). Use `make canary-gate SHA=<sha>` (which calls
# `rmlx metrics deltas`) to gate regressions from the DB. The CSV append below
# is preserved for one release as a compatibility aid; it will be removed once
# `make canary-gate` is the primary regression gate.

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

PROMPT_TOKENS=4096
MAX_TOKENS=100
MAX_CTX=8192
WARMUP_RUNS=1
MEASURED_RUNS=3
BUILD_PROFILE="release-perf"

# Model definitions: "short_name|absolute_path"
BONSAI_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/prism-ml__Ternary-Bonsai-8B-mlx-2bit"
GEMMA4E4B_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-e4b-it-mxfp8"
QWEN36_PATH="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__Qwen3.6-35B-A3B-8bit"
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
    "prism-ml__Ternary-Bonsai-8B-mlx-2bit|${BONSAI_PATH}"
    "mlx-community__gemma-4-e4b-it-mxfp8|${GEMMA4E4B_PATH}"
    "mlx-community__Qwen3.6-35B-A3B-8bit|${QWEN36_PATH}"
)
if $INCLUDE_26B; then
    MODELS+=("mlx-community__gemma-4-26b-a4b-it-mxfp8|${GEMMA4_26B_PATH}")
fi

# Pre-flight: ensure no competing MLX process holds the GPU
pkill -f "rmlx serve" || true
rm -f /tmp/rmlx.*.claim 2>/dev/null || true

# Verify binary present
if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}. Run: make build-perf" >&2
    exit 125
fi

# Ensure CSV dir and header
mkdir -p "${CSV_DIR}"
if [[ ! -f "${CSV}" ]]; then
    echo "ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile" > "${CSV}"
fi

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo "unknown")"

# Helper: run one baseline, parse decode_tps from stdout
# Usage: run_once <model_path> [--kv-quant <tag>]
run_once() {
    local model_path="$1"
    local kv_flag="${2:-}"
    local kv_val="${3:-}"
    local extra_args=()
    if [[ -n "${kv_flag}" ]]; then
        extra_args=("${kv_flag}" "${kv_val}")
    fi
    RMLX_HOME="${RMLX_HOME}" "${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens "${MAX_TOKENS}" \
        --max-ctx "${MAX_CTX}" \
        ${extra_args[@]+"${extra_args[@]}"} \
        2>/dev/null \
        | grep -o 'decode_tps=[0-9.]*' \
        | cut -d= -f2
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
    { ls -1 "${LOG_DIR}"/*.jsonl 2>/dev/null || true; } | sort \
        > "${SCRATCH_DIR}/canary_logs_before"

    RMLX_HOME="${RMLX_HOME}" "${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens 1 \
        --max-ctx "${MAX_CTX}" \
        > /dev/null 2>&1 &
    local probe_pid=$!
    wait "${probe_pid}" || true

    local log
    log="$({ ls -1 "${LOG_DIR}"/*.jsonl 2>/dev/null || true; } | sort \
        | comm -13 "${SCRATCH_DIR}/canary_logs_before" - \
        | python3 "${REPO_ROOT}/scripts/lib/run_log_for_pid.py" --pid "${probe_pid}")" ||
        return 1

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
        run_once "${model_path}" > /dev/null
    done

    # Measured runs
    tps_values=()
    for i in $(seq 1 ${MEASURED_RUNS}); do
        echo "  measured run ${i}/${MEASURED_RUNS}..." >&2
        val="$(run_once "${model_path}" "" "")"
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
    # "unknown" is a fallback for the --label text above, never provenance —
    # a checkout without .git must not pass --git-sha at all.
    # (top-level loop body, not a function — `local` is not valid here.)
    git_sha_args=()
    if [[ "${GIT_SHA}" != unknown* ]]; then
        git_sha_args=(--git-sha "${GIT_SHA}")
    fi
    if RMLX_HOME="${RMLX_HOME}" "${BINARY}" baseline \
        --model "${model_path}" \
        --prompt-tokens "${PROMPT_TOKENS}" \
        --max-tokens "${MAX_TOKENS}" \
        --max-ctx "${MAX_CTX}" \
        --label "canary sha=${GIT_SHA}" \
        ${git_sha_args[@]+"${git_sha_args[@]}"} \
        --record \
        > /dev/null 2>&1; then
        echo "  runs.db: ok" >&2
    else
        echo "  WARN: runs.db record failed (non-fatal; CSV row still written)" >&2
    fi

    # Print result
    printf "%s  decode_tps=%.2f ± %.2f\n" "${short_name}" "${med}" "${sd}"

    # K8VTurbo3 informational column (1 warmup + 3 measured, explicit --kv-quant).
    # Gemma4 small uses K8VTurbo3 as auto default; this column tracks it explicitly for all models.
    echo "  [k8vturbo3] warmup..." >&2
    run_once "${model_path}" --kv-quant k8vturbo3 > /dev/null
    turbo3_values=()
    for i in $(seq 1 ${MEASURED_RUNS}); do
        echo "  [k8vturbo3] measured run ${i}/${MEASURED_RUNS}..." >&2
        val="$(run_once "${model_path}" --kv-quant k8vturbo3)"
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
