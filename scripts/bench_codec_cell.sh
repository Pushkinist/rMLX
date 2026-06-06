#!/usr/bin/env bash
# bench_codec_cell.sh — single-codec × single-model bench runner.
#
# Usage:
#   scripts/bench_codec_cell.sh --kv-quant <codec> --model <snapshot-abs-path> \
#                               [--max-tokens 100] [--prompt-len 4096]
#
# Behavior:
#   - Hard-rule-8 preflight: kills competing MLX processes + removes claim file.
#   - 1 warmup run (discarded), 3 measured runs.
#   - Each run: target/release/rmlx baseline --model ... --kv-quant ... --prompt-tokens ... --max-tokens ...
#   - Parses decode_tps + prefill_tps from stdout.
#   - Appends 3 rows to .rmlx/bench/codec_cells.csv (one per run_idx 1/2/3).
#   - Prints summary: <codec> × <model>: mean decode_tps=X.X (±stddev), prefill_tps=X.X
#
# CSV schema (codec_cells.csv):
#   timestamp,codec,model,prompt_len,max_tokens,run_idx,decode_tps,prefill_tps,git_sha
#
# Notes:
#   - Uses target/release/rmlx (not release-perf) — matches the accessible binary.
#     Override: RMLX_BINARY=<path> to point at release-perf if available.
#   - Binary path resolves to REPO_ROOT/target/release/rmlx by default; for parity
#     with perf_canary.sh, set RMLX_BINARY=$REPO_ROOT/target/release-perf/rmlx.
#   - The 3% regression gate threshold is enforced by regression_gate.sh / make canary-gate.
#     This script only records; it does not gate.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Binary: prefer release-perf if present; fall back to release.
if [[ -x "${RMLX_BINARY:-}" ]]; then
    BINARY="${RMLX_BINARY}"
elif [[ -x "${REPO_ROOT}/target/release-perf/rmlx" ]]; then
    BINARY="${REPO_ROOT}/target/release-perf/rmlx"
elif [[ -x "${REPO_ROOT}/target/release/rmlx" ]]; then
    BINARY="${REPO_ROOT}/target/release/rmlx"
else
    BINARY="${REPO_ROOT}/target/release/rmlx"
fi

RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
CSV_DIR="${RMLX_HOME}/bench"
CSV="${CSV_DIR}/codec_cells.csv"

# Defaults
MAX_TOKENS=100
PROMPT_LEN=4096
KV_QUANT=""
MODEL_PATH=""

WARMUP_RUNS=1
MEASURED_RUNS=3

# ---- Argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv-quant)
            if [[ $# -lt 2 ]]; then echo "ERROR: --kv-quant requires a value" >&2; exit 1; fi
            KV_QUANT="$2"; shift 2 ;;
        --model)
            if [[ $# -lt 2 ]]; then echo "ERROR: --model requires a value" >&2; exit 1; fi
            MODEL_PATH="$2"; shift 2 ;;
        --max-tokens)
            if [[ $# -lt 2 ]]; then echo "ERROR: --max-tokens requires a value" >&2; exit 1; fi
            MAX_TOKENS="$2"; shift 2 ;;
        --prompt-len)
            if [[ $# -lt 2 ]]; then echo "ERROR: --prompt-len requires a value" >&2; exit 1; fi
            PROMPT_LEN="$2"; shift 2 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: $0 --kv-quant <codec> --model <path> [--max-tokens 100] [--prompt-len 4096]" >&2
            exit 1 ;;
    esac
done

# ---- Validate required args --------------------------------------------------
if [[ -z "${KV_QUANT}" ]]; then
    echo "ERROR: --kv-quant is required" >&2
    exit 1
fi
if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

MODEL_BASENAME="$(basename "${MODEL_PATH}")"

# ---- Preflight: Hard rule 8 —— kill competing MLX processes -----------------
pkill -f "rmlx serve" || true
pkill -f mlx_lm || true
pkill -f paroquant || true
pkill -f omlx || true
sleep 5
rm -f /tmp/rmlx.62265.claim 2>/dev/null || true

# ---- Verify binary -----------------------------------------------------------
if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}. Run: make build or make build-perf" >&2
    exit 1
fi

# ---- Verify model path -------------------------------------------------------
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: model path not found: ${MODEL_PATH}" >&2
    exit 1
fi

# ---- Ensure CSV dir and header -----------------------------------------------
mkdir -p "${CSV_DIR}"
if [[ ! -f "${CSV}" ]]; then
    echo "timestamp,codec,model,prompt_len,max_tokens,run_idx,decode_tps,prefill_tps,git_sha" > "${CSV}"
fi

GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null | cut -c1-12 || echo "unknown")"

# ---- Helper: run one baseline, parse decode_tps and prefill_tps --------------
# Outputs a single colon-separated string: "<decode_tps>:<prefill_tps>"
run_once() {
    local output
    output="$(RMLX_HOME="${RMLX_HOME}" "${BINARY}" baseline \
        --model "${MODEL_PATH}" \
        --kv-quant "${KV_QUANT}" \
        --prompt-tokens "${PROMPT_LEN}" \
        --max-tokens "${MAX_TOKENS}" \
        2>/dev/null)"
    local dtps
    dtps="$(echo "${output}" | grep -o 'decode_tps=[0-9.]*' | cut -d= -f2 || true)"
    local ptps
    ptps="$(echo "${output}" | grep -o 'prefill_tps=[0-9.]*' | cut -d= -f2 || true)"
    echo "${dtps:-0}:${ptps:-0}"
}

# ---- Helper: compute mean of space-separated values via awk -----------------
mean() {
    echo "$@" | tr ' ' '\n' | awk 'NR==1{s=$1;n=1;next}{s+=$1;n++}END{if(n>0) printf "%.4f\n", s/n; else print "0"}'
}

# ---- Helper: compute sample stddev of space-separated values via awk --------
stddev() {
    echo "$@" | tr ' ' '\n' | awk '
    { sum+=$1; sumsq+=$1*$1; n++ }
    END {
        if (n < 2) { print "0.0000"; exit }
        mean = sum/n
        variance = (sumsq - n*mean*mean) / (n-1)
        if (variance < 0) variance = 0
        printf "%.4f\n", sqrt(variance)
    }'
}

# ---- Warmup runs (discarded) -------------------------------------------------
echo "==> ${KV_QUANT} × ${MODEL_BASENAME}" >&2
for i in $(seq 1 "${WARMUP_RUNS}"); do
    echo "  warmup ${i}/${WARMUP_RUNS}..." >&2
    run_once > /dev/null
done

# ---- Measured runs -----------------------------------------------------------
decode_values=()
prefill_values=()
run_records=()  # "decode_tps:prefill_tps" per run, for CSV append

for i in $(seq 1 "${MEASURED_RUNS}"); do
    echo "  measured run ${i}/${MEASURED_RUNS}..." >&2
    result="$(run_once)"
    dtps="${result%%:*}"
    ptps="${result##*:}"
    # Reject zero or empty TPS — indicates a silent model failure (e.g. SWA chunked-prefill bug)
    ok="$(awk -v v="${dtps}" 'BEGIN { print (v+0 > 0.001) ? "yes" : "no" }')"
    if [[ -z "${dtps}" ]] || [[ "${ok}" != "yes" ]]; then
        echo "ERROR: decode_tps=${dtps} for run ${i} — model returned 0 TPS (silent failure). Check stderr for WARN messages." >&2
        exit 1
    fi
    decode_values+=("${dtps}")
    prefill_values+=("${ptps}")
    run_records+=("${dtps}:${ptps}")
done

mean_dtps="$(mean "${decode_values[@]}")"
sd_dtps="$(stddev "${decode_values[@]}")"
mean_ptps="$(mean "${prefill_values[@]}")"

ts_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# ---- Append 3 CSV rows (one per run_idx) ------------------------------------
for i in $(seq 1 "${MEASURED_RUNS}"); do
    record="${run_records[$((i-1))]}"
    dtps="${record%%:*}"
    ptps="${record##*:}"
    echo "${ts_utc},${KV_QUANT},${MODEL_BASENAME},${PROMPT_LEN},${MAX_TOKENS},${i},${dtps},${ptps},${GIT_SHA}" >> "${CSV}"
done

# ---- Print summary -----------------------------------------------------------
printf "%s × %s: mean decode_tps=%.1f (±%.2f), prefill_tps=%.1f\n" \
    "${KV_QUANT}" "${MODEL_BASENAME}" "${mean_dtps}" "${sd_dtps}" "${mean_ptps}"

echo "" >&2
echo "CSV: ${CSV} (3 rows appended)" >&2
