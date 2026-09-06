#!/usr/bin/env bash
# spec_bench_published.sh — bench one verifier under the published on-device
# speculative-decoding protocol.
#
# Usage:
#   bash scripts/spec_bench_published.sh <verifier-snapshot-dir> [options]
#
#     --draft-model DIR        serve with this drafter (absent = plain decode)
#     --draft-kind KIND        mtp | dflash | eagle3 | two_model | ...
#     --draft-block-size N     block to ask the drafter for
#     --kv-quant Q             KV codec to ask the engine for
#     --max-ctx N              context the server is started with
#     --port N
#     --samples-root DIR       measure an unpinned copy instead; not recordable
#     --allow-busy-host        measure anyway, and taint the result
#     --synthetic-arms         the server is a stub, so this run measures nothing
#
# WHAT IT MEASURES
#
# The published protocol reports output speed as a macro-average over MT-Bench,
# MATH-500 and HumanEval subsets, each also reported per dataset, as the mean of
# three consecutive runs. This drives one chat request per checked-in sample,
# three times, and reports the per-dataset and macro means with their
# run-to-run range.
#
# The decode rate of every request is the engine's own reading of the window
# from the first emitted token to the last, prefill excluded — off the
# round-loop `done` line on the speculative arm and off the per-request ITL
# aggregate on the plain one — and each is cross-checked against the same
# window timed at the client. The two rings at GET /metrics/cache hold twenty
# entries, so neither can carry a 128-sample dataset; the readings come out of
# the run log, which holds one event per request.
#
# WHAT THE PROTOCOL DOES NOT STATE, AND WE THEREFORE CHOOSE AND PRINT
#
#   sampling      the model's own defaults. The request carries no temperature,
#                 top_p, top_k or seed, so there is one copy of them and it is
#                 the checkpoint's. A snapshot stating none of the three the
#                 protocol names is refused before anything is served, because
#                 the engine's fallbacks — temperature 1.0, top_p 1.0, top_k 0
#                 (disabled) — would then be published as that checkpoint's
#                 defaults. `repetition_penalty` is not required: the protocol
#                 does not name it and its fallback is 1.0, the neutral element.
#                 What is PRINTED and recorded is not that file but the sampling
#                 the engine says it resolved, read back per request from its
#                 own log. The seed is part of it: the request sends none and
#                 the engine substitutes a fixed default, so the three passes
#                 replay one RNG stream rather than sampling independently.
#   thinking      on, and its tokens count as output — they are generated
#                 tokens and the server counts them in `completion_tokens`.
#   max output    1024 tokens, and MATH-500 a second time at 4096: a reasoning
#                 answer truncated at 1024 is a different workload.
#   warmup        one untimed request per pass.
#   passes        three, as the protocol says. Not a flag: a mean of any other
#                 count is a different figure.
#   macro         the mean of the three datasets at the 1024-token budget. The
#                 MATH-500 4096 cell is a column beside that headline, not a
#                 fourth dataset — averaging it in would give MATH-500 twice the
#                 weight of the others.
#   seed          left at the engine's default in every pass. The three passes
#                 therefore replay one RNG stream, and the run-to-run range is a
#                 reading of machine variance with the sampling held still —
#                 which is the tighter estimator and the one that makes the
#                 range band a statement about measurement stability. Varying it
#                 per pass would fold sampling variance into a figure the
#                 protocol presents as a stability check, and three passes
#                 cannot separate the two. The claim is checked, not asserted:
#                 `divergent_samples` counts the samples that did not generate
#                 the same length in all three passes.
#   fixed prompt  the protocol's second figure — autoregressive output speed,
#                 input speed and resident memory on one prompt of a stated
#                 length. Its token count belongs to the tokenizer as much as to
#                 the bytes, so the body is FITTED against this checkpoint from
#                 a checked-in corpus and a stated cut rule, on a preparation
#                 server that measures nothing, and a target the corpus cannot
#                 reach exactly is refused rather than rounded to. It is sent
#                 once per pass, before the cells, so its prefill is cold. It is
#                 not measured on a speculative arm at all: a rate a drafter
#                 produced is not the autoregressive one.
#   input speed   prompt tokens over TTFT, on that cold prompt cache.
#   memory        the peak of `rmlx_process_phys_footprint_bytes` (the counter
#                 docs/PROFILING.md §9 names) sampled at a fixed interval while
#                 the fixed-prompt request is in flight, reported with the
#                 interval — it is a gauge, so a sampled peak is a lower bound.
#
# WHAT IS RECORDED ALONGSIDE THE NUMBERS
#
#   binary        the file's sha256 AND the log-message literals the readings
#                 are read off, checked before the first server starts. A digest
#                 alone does not separate a build from the stale one a
#                 stash-build-unstash cycle left in `target/` — both hash the
#                 same because both are the same file.
#   thermal       three readings per pass, from `pmset -g therm`. A throttled or
#                 unreadable state taints the run, in the same taint field the
#                 host-interference gate writes to. `powermetrics` is the
#                 instantaneous counter and needs sudo, so it is not reachable
#                 from a non-interactive run; what is used is stated in the
#                 result rather than implied.
#
# THE SAMPLE SETS
#
# The published number comes from `prompts/published/`, and that root is held to
# `published_samples.py verify` before anything is served — no flag turns that
# off. `--samples-root` measures some other copy: it is an operator override for
# working on the harness, it is NOT verified, and the result it writes carries
# `unverified_samples: true` so nothing downstream can promote it as a published
# measurement.
#
# MEASUREMENT VS LOGIC
#
# `--synthetic-arms` declares that the server is a stub, so the run exercises
# this script's own scheduling, guards and arithmetic and measures nothing. The
# machine is then not consulted at all — no preflight, no quiescence gate, no
# per-pass interference sampling — and the run says so on stdout and in
# `synthetic_arms` in the result file. Every guard that reads the run instead of
# the machine is untouched by it.
#
# Exit codes:
#   0   — three passes ran and every mean is within the range band
#   1   — a precondition failed, or a reading could not be attributed
#   3   — the run is readable and at least one mean is refused for its range
#   125 — the host is not quiescent (pass --allow-busy-host to measure anyway)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="${REPO_ROOT}/target/release-perf/rmlx"
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
LOG_DIR="${RMLX_HOME}/logs"
SCRATCH_DIR="${RMLX_HOME}/tmp"
RESULT_DIR="${RMLX_HOME}/bench/spec_bench_published"
AWK_BUSIEST="${REPO_ROOT}/scripts/lib/busiest_between.awk"

# ── Pinned protocol constants ────────────────────────────────────────────────

PASSES=3
WARMUPS_PER_PASS=1
RANGE_REFUSAL_PCT=5
# One network hop on loopback, not a measurement difference. Past this the two
# readings of one decode window are a finding and the run stops.
CROSS_CHECK_BAND_PCT=10
BUSY_PCT=25
# `<dataset>:<max output tokens>`, one measured cell each.
CELLS=(mt_bench:1024 math_500:1024 humaneval:1024 math_500:4096)
# The budget the macro average is taken at — one cell per dataset. Every other
# cell is a column beside the headline.
MACRO_MAX_TOKENS=1024
# The fixed-length prompt the protocol reports autoregressive output speed,
# input speed and resident memory on. The count is a property of the tokenizer
# as well as the bytes, so the body is fitted against the server rather than
# checked in — see lib/published_fixed_prompt.py.
FIXED_PROMPT_TOKENS=1355
FIXED_PROMPT_CORPUS="${REPO_ROOT}/prompts/longctx_4k.json"
# Resident memory is a gauge, so a peak can only be sampled. This interval is
# recorded with the figure: what comes out is a lower bound on the true peak.
MEMORY_POLL_MS=250

# ── Flags ────────────────────────────────────────────────────────────────────

VERIFIER_MODEL=""
DRAFTER_MODEL=""
DRAFT_KIND=""
DRAFT_BLOCK_SIZE=""
KV_QUANT=""
MAX_CTX=8192
PORT=8090
PUBLISHED_ROOT="${REPO_ROOT}/prompts/published"
SAMPLES_ROOT=""
ALLOW_BUSY_HOST=false
SYNTHETIC_ARMS=false

need_value() { [[ -n "${2:-}" ]] || { echo "ERROR: $1 requires a value" >&2; exit 1; }; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --draft-model) need_value "$1" "${2:-}"; DRAFTER_MODEL="$2"; shift 2 ;;
        --draft-kind) need_value "$1" "${2:-}"; DRAFT_KIND="$2"; shift 2 ;;
        --draft-block-size) need_value "$1" "${2:-}"; DRAFT_BLOCK_SIZE="$2"; shift 2 ;;
        --kv-quant) need_value "$1" "${2:-}"; KV_QUANT="$2"; shift 2 ;;
        --max-ctx) need_value "$1" "${2:-}"; MAX_CTX="$2"; shift 2 ;;
        --port) need_value "$1" "${2:-}"; PORT="$2"; shift 2 ;;
        --samples-root) need_value "$1" "${2:-}"; SAMPLES_ROOT="$2"; shift 2 ;;
        --allow-busy-host) ALLOW_BUSY_HOST=true; shift ;;
        --synthetic-arms) SYNTHETIC_ARMS=true; shift ;;
        -*) echo "ERROR: unknown flag: $1" >&2; exit 1 ;;
        *)
            [[ -z "${VERIFIER_MODEL}" ]] || {
                echo "ERROR: one verifier snapshot, got '${VERIFIER_MODEL}' and '$1'" >&2
                exit 1
            }
            VERIFIER_MODEL="$1"; shift ;;
    esac
done

if [[ -z "${VERIFIER_MODEL}" ]]; then
    echo "ERROR: a verifier snapshot directory is required." >&2
    echo "  Resolve a concrete path from LOCAL.md (gitignored):" >&2
    echo "  bash scripts/spec_bench_published.sh \$O_MODELS/<snapshot>" >&2
    exit 1
fi

# The kind reaches a directory name and the result file before the engine sees
# it, so a value with a `/` or a space in it writes outside the scratch tree.
# That is the property this script needs and the only one it checks: which
# kinds exist is the engine's to say, in its own message.
if [[ -n "${DRAFT_KIND}" ]] && ! [[ "${DRAFT_KIND}" =~ ^[a-z0-9_]+$ ]]; then
    echo "ERROR: --draft-kind '${DRAFT_KIND}' is not a bare lower-case name;" \
         "it reaches a path and the result file before the engine sees it" >&2
    exit 1
fi

if [[ -n "${DRAFT_BLOCK_SIZE}" ]]; then
    if ! [[ "${DRAFT_BLOCK_SIZE}" =~ ^[0-9]+$ ]] || (( DRAFT_BLOCK_SIZE < 2 )); then
        echo "ERROR: --draft-block-size '${DRAFT_BLOCK_SIZE}' must be an integer >= 2:" \
             "a block of 1 leaves no room for a draft token" >&2
        exit 1
    fi
fi

if [[ -n "${DRAFT_KIND}${DRAFT_BLOCK_SIZE}" && -z "${DRAFTER_MODEL}" ]]; then
    echo "ERROR: --draft-kind / --draft-block-size describe a drafter this run" \
         "was not given; pass --draft-model or drop them" >&2
    exit 1
fi

ARM="plain"
[[ -n "${DRAFTER_MODEL}" ]] && ARM="speculative"

# ── Preconditions ────────────────────────────────────────────────────────────

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: binary not found at ${BINARY}. Run: make build-perf" >&2
    exit 1
fi

# Which binary this is, and whether it can write the events the readings come
# off. A digest alone would not have caught the stash-build-unstash cycle that
# once had an A/B compare a build against itself: both files hashed the same
# because both were the same file.
BINARY_IDENTITY="$(python3 "${REPO_ROOT}/scripts/lib/binary_identity.py" \
    "${BINARY}" --arm "${ARM}")" || exit 1
BINARY_SHA256="$(python3 -c 'import json, sys; print(json.loads(sys.argv[1])["sha256"])' \
    "${BINARY_IDENTITY}")"
if [[ ! -d "${VERIFIER_MODEL}" ]]; then
    echo "ERROR: verifier snapshot not found: ${VERIFIER_MODEL}" >&2
    exit 1
fi
if [[ -n "${DRAFTER_MODEL}" && ! -d "${DRAFTER_MODEL}" ]]; then
    echo "ERROR: drafter snapshot not found: ${DRAFTER_MODEL}" >&2
    exit 1
fi

# The sample sets are the measurement's inputs. A number traced back to a file
# that no longer re-derives from what the gate pins is traced back to nothing,
# so the published root is verified and no flag turns that off. `--samples-root`
# is a different path entirely: an unverified operator copy, marked as such in
# the result so nothing downstream promotes it.
UNVERIFIED_SAMPLES=false
if [[ -z "${SAMPLES_ROOT}" ]]; then
    SAMPLES_ROOT="${PUBLISHED_ROOT}"
    # `--root` names the tree about to be measured. Without it the gate resolves
    # its own checkout from its own path, which is the same tree in a normal
    # run and a different one whenever this script is driven from elsewhere —
    # and verifying one root while measuring another proves nothing about the
    # number. The anchor it checks against lives in the gate's source either
    # way, so naming the root weakens nothing.
    if ! python3 "${REPO_ROOT}/scripts/published_samples.py" verify \
            --root "${PUBLISHED_ROOT}"; then
        echo "ERROR: the published sample sets do not re-derive from what" \
             "published_samples.py pins; refusing to measure against them" >&2
        exit 1
    fi
else
    UNVERIFIED_SAMPLES=true
    cat >&2 <<'OVERRIDE'
UNVERIFIED SAMPLES: --samples-root names a copy that is not the published one,
  so it is not held to published_samples.py and this run is not a published
  measurement. The result carries `unverified_samples: true`.
OVERRIDE
fi
SAMPLES_ROOT="$(cd "${SAMPLES_ROOT}" 2>/dev/null && pwd)" || {
    echo "ERROR: sample root not found" >&2
    exit 1
}

SNAPSHOT_ID="$(python3 "${REPO_ROOT}/scripts/lib/snapshot_identity.py" "${VERIFIER_MODEL}")" || {
    echo "ERROR: cannot read the identity of ${VERIFIER_MODEL}" >&2
    exit 1
}

# Server lifecycle, log attribution and the resolved-codec reader.
# shellcheck source=scripts/lib/bench_server.sh
. "${REPO_ROOT}/scripts/lib/bench_server.sh"

# Backend, version, build profile and hardware tag come from the binary that is
# actually being measured, never from a constant here.
# shellcheck source=scripts/lib/identity.sh
. "${REPO_ROOT}/scripts/lib/identity.sh"
rmlx_export_identity "${BINARY}"

MODEL_NAMESPACE="$(field_of "${SNAPSHOT_ID}" model_namespace)"
MODEL_NAME="$(field_of "${SNAPSHOT_ID}" model)"
WEIGHT_QUANT="$(field_of "${SNAPSHOT_ID}" weight_quant)"
MODEL_ID="$(basename "${VERIFIER_MODEL%/}")"

# A pre-flight refusal, not the published figure: the checkpoint has to state
# the sampling this run will be measured under, or the engine substitutes a
# fallback nobody chose. What it actually resolved to is read back later.
SNAPSHOT_SAMPLING="$(python3 "${REPO_ROOT}/scripts/lib/snapshot_sampling.py" \
    "${VERIFIER_MODEL}")" || exit 1
SAMPLED="$(field_of "${SNAPSHOT_SAMPLING}" sampled)"

mkdir -p "${LOG_DIR}" "${SCRATCH_DIR}" "${RESULT_DIR}"
WORK="$(mktemp -d "${SCRATCH_DIR}/published.XXXXXX")"

# Three servers run per invocation. A SIGTERM to this script — a CI timeout, an
# operator, a parent tearing down — would otherwise leave one alive holding
# /tmp/rmlx.*.claim, and the next run's preflight `pkill -f "rmlx serve"` does
# not match a snapshotted binary.
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
    rm -rf "${WORK}"
}
trap cleanup EXIT INT TERM

# ── Payloads ─────────────────────────────────────────────────────────────────

CELL_INDEX="${WORK}/cells.tsv"
CELL_ARGS=()
for spec in "${CELLS[@]}"; do CELL_ARGS+=(--cell "${spec}"); done
python3 "${REPO_ROOT}/scripts/lib/published_payloads.py" \
    --samples-root "${SAMPLES_ROOT}" \
    --out "${WORK}/payloads" \
    --model-id "${MODEL_ID}" \
    --index "${CELL_INDEX}" \
    "${CELL_ARGS[@]}" || exit 1

REQUESTS_PER_PASS="$(wc -l < "${CELL_INDEX}" | tr -d ' ')"
# Written beside the cells and outside the index: the warmup is a request, not a
# measurement.
WARMUP_PAYLOAD="${WORK}/payloads/warmup.json"

# The fixed-length-prompt request is one more request in every pass, and it is
# only sent on the plain arm — see the fitting block below for why.
FIXED_REQUESTS=0
[[ "${ARM}" == "plain" ]] && FIXED_REQUESTS=1
REQUESTS_TOTAL=$((REQUESTS_PER_PASS + WARMUPS_PER_PASS + FIXED_REQUESTS))

# One sampler-resolution event per request the engine actually sampled. A greedy
# checkpoint resolves no sampler and writes none, so the reader is told to
# expect none rather than deciding for itself.
if [[ "${SAMPLED}" == "true" ]]; then
    SAMPLER_EVENTS=${REQUESTS_TOTAL}
else
    SAMPLER_EVENTS=0
fi

# ── Host ─────────────────────────────────────────────────────────────────────

CPU_SNAPSHOT_SKIP="$(basename "${BINARY}")"
export CPU_SNAPSHOT_SKIP
# `snapshot_ok` and `window_not_sampled` read SYNTHETIC_ARMS, which is why this
# is sourced after the flags are parsed.
# shellcheck source=scripts/lib/cpu_snapshot.sh
. "${REPO_ROOT}/scripts/lib/cpu_snapshot.sh"

TAINT=""
note_taint() { TAINT="${TAINT}$1; "; }

# "<state> <pct> <comm>" for the window between two snapshots $3 seconds apart.
# `unmeasured` stays distinct from `quiet` all the way into the report: folding
# "nobody could look" into "nothing was running" is how a gate stops gating.
host_window() {
    local raw
    if window_not_sampled "$1" "$2"; then echo "not-sampled - -"; return; fi
    if [[ -e "$1.failed" || -e "$2.failed" ]]; then echo "unmeasured - -"; return; fi
    raw="$(awk -v window="$3" -f "${AWK_BUSIEST}" "$1" "$2")"
    case "${raw%% *}" in
        unmeasured) echo "unmeasured - -" ;;
        idle) echo "quiet 0.0 -" ;;
        *)
            if awk -v p="$(echo "${raw}" | awk '{print $2}')" -v t="${BUSY_PCT}" \
                'BEGIN { exit !(p >= t) }'; then
                echo "busy ${raw#* }"
            else
                echo "quiet ${raw#* }"
            fi
            ;;
    esac
}

# One thermal reading, as a bare state word.
#
# `sudo powermetrics` is the instantaneous counter and it needs a password, so
# it cannot be sampled from a non-interactive run; `pmset -g therm` is what is
# left and it is a different thing — the last thermal-pressure level the system
# posted, which is a notification history and not a reading of this instant.
# That distinction is the reason for four states rather than two:
#
#   nominal      a level was posted and the CPU is not being held back
#   throttled=N  a level was posted and the CPU is capped at N%
#   unrecorded   no level has been posted since boot. Nothing has throttled,
#                which is not the same as having looked and seen nothing, so it
#                is named separately and printed — it just does not taint, or
#                every run on a healthy Mac would carry a taint and the field
#                would stop meaning anything.
#   unreadable   the tool is absent or failed. Nobody looked.
thermal_sample() {
    local raw limit
    raw="$(pmset -g therm 2>/dev/null)" || { echo "unreadable"; return; }
    limit="$(printf '%s\n' "${raw}" | awk -F'= *' '/CPU_Speed_Limit/ { print $2; exit }')"
    if [[ -z "${limit}" ]]; then
        echo "unrecorded"
    elif (( limit < 100 )); then
        echo "throttled=${limit}"
    else
        echo "nominal"
    fi
}

THERMAL_READINGS=()

# note_thermal <where> — sample, record, and taint on a state that says the
# numbers taken around it were taken on a machine that was not at full speed,
# or on nobody having looked.
note_thermal() {
    local state
    state="$(thermal_sample)"
    THERMAL_READINGS+=("$1 ${state}")
    case "${state}" in
    throttled=* | unreadable) note_taint "thermal $1: ${state}" ;;
    esac
}

SAMPLES_LABEL="${SAMPLES_ROOT}"
if $UNVERIFIED_SAMPLES; then
    SAMPLES_LABEL="${SAMPLES_LABEL} (UNVERIFIED, not recordable)"
fi

echo "==> spec_bench_published.sh"
echo "    verifier   : ${MODEL_NAMESPACE}/${MODEL_NAME} (${WEIGHT_QUANT})"
echo "    arm        : ${ARM}${DRAFTER_MODEL:+ (${DRAFT_KIND:-engine default} block ${DRAFT_BLOCK_SIZE:-engine default})}"
echo "    samples    : ${SAMPLES_LABEL} — ${REQUESTS_PER_PASS} requests per pass"
echo "    thinking   : on, counted as output"
echo "    passes     : ${PASSES}, ${WARMUPS_PER_PASS} untimed warmup each"
echo "    range band : ${RANGE_REFUSAL_PCT}% of the mean"
echo "    sampling   : the checkpoint's own; what it resolved to is read back"
echo "                 from the engine and printed with the results"
echo "    seed       : the request sends none, so all three passes replay one"
echo "                 RNG stream and the range is machine variance"
echo "    binary     : sha256:${BINARY_SHA256}"
echo "    thermal    : pmset -g therm (powermetrics needs sudo, so it is not"
echo "                 reachable from a non-interactive run)"
echo ""

if $SYNTHETIC_ARMS; then
    cat >&2 <<'BANNER'
INTERFERENCE GATE: OFF — --synthetic-arms. The server is a stub, so this run
  exercises this script's scheduling, guards and arithmetic and measures
  nothing. The machine is not consulted: no preflight, no entry quiescence
  gate, no per-pass interference sampling. No number below describes this host.
BANNER
else
    snapshot_ok "${WORK}/entry_a" || true
    sleep 5
    snapshot_ok "${WORK}/entry_b" || true
    ENTRY="$(host_window "${WORK}/entry_a" "${WORK}/entry_b" 5)"
    if [[ "${ENTRY%% *}" != "quiet" ]]; then
        echo "host is not quiescent: ${ENTRY}" >&2
        if ! $ALLOW_BUSY_HOST; then
            echo "  Quiesce the host, or pass --allow-busy-host to measure anyway." >&2
            exit 125
        fi
        echo "  --allow-busy-host: every number below is suspect." >&2
        note_taint "entry gate: ${ENTRY}"
    fi
fi

# ── One request ──────────────────────────────────────────────────────────────

# send <payload-file> <kv-out> — one chat request, timed at the client.
send() {
    curl -s -H "Content-Type: application/json" --data-binary @"$1" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" --no-buffer \
        | python3 "${REPO_ROOT}/scripts/lib/sse_decode_window.py" > "$2"
}

# ── Resident memory ──────────────────────────────────────────────────────────
#
# `phys_footprint` is the counter the OOM killer and Activity Monitor use
# (docs/PROFILING.md §9), and the server publishes it as a gauge. A gauge has no
# peak, so the peak is sampled: what comes out is the largest value seen at the
# poll interval, which is a lower bound on the true peak and is recorded with
# the interval so it reads as one.
poll_memory() {
    local out="$1"
    : > "${out}"
    while :; do
        curl -s --max-time 5 "http://127.0.0.1:${PORT}/metrics" \
            | awk '/^rmlx_process_phys_footprint_bytes /{ print "phys", $2 }
                   /^rmlx_process_rss_bytes /{ print "rss", $2 }' >> "${out}"
        sleep "$(awk -v ms="${MEMORY_POLL_MS}" 'BEGIN { print ms / 1000 }')"
    done
}

# ── Passes ───────────────────────────────────────────────────────────────────

PASS_FILES=()
# start_server <tag> — one verifier server, ready to serve, as SERVER_PID.
#
# Two callers: the preparation server the fixed prompt is fitted against, and
# each pass. They must be the same server or the fit describes a different
# engine than the one measured on it.
start_server() {
    local tag="$1"
    local args=(serve --model "${VERIFIER_MODEL}" --max-ctx "${MAX_CTX}" --port "${PORT}")
    [[ -n "${KV_QUANT}" ]] && args+=(--kv-quant "${KV_QUANT}")
    [[ -n "${DRAFTER_MODEL}" ]] && args+=(--draft-model "${DRAFTER_MODEL}")
    [[ -n "${DRAFT_KIND}" ]] && args+=(--draft-kind "${DRAFT_KIND}")
    [[ -n "${DRAFT_BLOCK_SIZE}" ]] && args+=(--draft-block-size "${DRAFT_BLOCK_SIZE}")

    # The plain arm's per-request decode rate is the ITL aggregate the engine
    # writes at the end of each request, and that event is a debug one. The
    # filter names its module rather than raising the whole preset: `debug`
    # across the workspace reaches the per-layer KV events inside the decode
    # loop, which would be measuring a differently instrumented engine.
    RMLX_HOME="${RMLX_HOME}" \
    RMLX_LOG_CAP_MB=400 \
    RUST_LOG="info,rmlx_server::engine::arch_generator=debug" \
        "${BINARY}" "${args[@]}" \
        > "${WORK}/server_${tag}.txt" 2>&1 &
    SERVER_PID=$!
    echo "  [server] pid=${SERVER_PID}" >&2

    if ! wait_for_server; then
        kill "${SERVER_PID}" 2>/dev/null || true
        tail -20 "${WORK}/server_${tag}.txt" >&2 || true
        return 1
    fi
}

# ── The fixed-length prompt ──────────────────────────────────────────────────
#
# The protocol's second figure is autoregressive output speed, input speed and
# resident memory on one prompt of a stated length, so this block does not run
# on a speculative arm at all: a rate produced by a drafter is not the
# autoregressive one, and publishing it under that name is the comparison the
# whole harness exists to make honest.
#
# The body is FITTED, on a server of its own, before the passes. The fit probes
# prefixes of the prompt it is converging on, so every one of them would leave
# that prefix in the prompt cache — and the protocol's input speed is over a
# cold one. A preparation server measures nothing and is thrown away.
FIXED_PAYLOAD=""
FIXED_RECORD=""
if [[ "${ARM}" == "plain" ]]; then
    echo "==> fitting the ${FIXED_PROMPT_TOKENS}-token prompt (preparation server)"
    $SYNTHETIC_ARMS || preflight
    if ! start_server fit; then exit 1; fi
    FIXED_RECORD="${WORK}/fixed_fit.json"
    FIXED_PAYLOAD="${WORK}/fixed_payload.json"
    if ! python3 "${REPO_ROOT}/scripts/lib/published_fixed_prompt.py" \
            --corpus "${FIXED_PROMPT_CORPUS}" \
            --target "${FIXED_PROMPT_TOKENS}" \
            --port "${PORT}" \
            --model-id "${MODEL_ID}" \
            --max-tokens "${MACRO_MAX_TOKENS}" \
            --out "${FIXED_RECORD}" \
            --payload "${FIXED_PAYLOAD}"; then
        kill "${SERVER_PID}" 2>/dev/null || true
        exit 1
    fi
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    SERVER_PID=""
    sleep 3
    echo ""
else
    echo "==> the fixed-prompt block is not measured on a speculative arm:" \
         "its figure is the autoregressive one"
    echo ""
fi

HOST_WINDOWS=()
FIXED_FILES=()

for (( pass = 1; pass <= PASSES; pass++ )); do
    echo "==> pass ${pass}/${PASSES}"
    $SYNTHETIC_ARMS || preflight
    snapshot_logs

    if ! start_server "${pass}"; then exit 1; fi

    snapshot_ok "${WORK}/pass${pass}_a" || true
    $SYNTHETIC_ARMS || note_thermal "pass ${pass} start"
    PASS_STARTED="$(date +%s)"

    for (( w = 0; w < WARMUPS_PER_PASS; w++ )); do
        if ! send "${WARMUP_PAYLOAD}" "${WORK}/warmup.kv"; then
            echo "ERROR: pass ${pass}: the warmup request failed" >&2
            kill "${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
        echo "  [warmup] done" >&2
    done

    PASS_DIR="${WORK}/pass${pass}"
    mkdir -p "${PASS_DIR}"

    # The fixed-length prompt, once per pass and before the cells, so its
    # prefix is not in the prompt cache — the protocol's input speed is over a
    # cold one, and the warmup's prompt is in no sample set and shares no prefix
    # with this.
    if [[ -n "${FIXED_PAYLOAD}" ]]; then
        poll_memory "${PASS_DIR}/memory.txt" &
        MEMORY_PID=$!
        if ! send "${FIXED_PAYLOAD}" "${PASS_DIR}/fixed.kv"; then
            kill "${MEMORY_PID}" 2>/dev/null || true
            echo "ERROR: pass ${pass}: the fixed-prompt request failed" >&2
            kill "${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
        kill "${MEMORY_PID}" 2>/dev/null || true
        wait "${MEMORY_PID}" 2>/dev/null || true
        echo "  [fixed] done" >&2
    fi

    PASS_INDEX="${PASS_DIR}/index.tsv"
    : > "${PASS_INDEX}"
    n=0
    while IFS=$'\t' read -r cell dataset max_tokens sample_id body_sha payload; do
        kv="${PASS_DIR}/$(printf '%05d' "${n}").kv"
        if ! send "${payload}" "${kv}"; then
            echo "ERROR: pass ${pass} ${cell}/${sample_id}: the request failed or its" \
                 "response could not be read" >&2
            kill "${SERVER_PID}" 2>/dev/null || true
            exit 1
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${cell}" "${dataset}" "${max_tokens}" "${sample_id}" "${body_sha}" "${kv}" \
            >> "${PASS_INDEX}"
        n=$((n + 1))
        if (( n == REQUESTS_PER_PASS / 2 )); then
            $SYNTHETIC_ARMS || note_thermal "pass ${pass} mid"
        fi
        if (( n % 25 == 0 )); then
            echo "  [pass ${pass}] ${n}/${REQUESTS_PER_PASS}" >&2
        fi
    done < "${CELL_INDEX}"

    $SYNTHETIC_ARMS || note_thermal "pass ${pass} end"
    PASS_SECONDS=$(( $(date +%s) - PASS_STARTED ))
    snapshot_ok "${WORK}/pass${pass}_b" || true
    WINDOW="$(host_window "${WORK}/pass${pass}_a" "${WORK}/pass${pass}_b" "${PASS_SECONDS}")"
    HOST_WINDOWS+=("${WINDOW}")
    # Three outcomes, not two. `quiet` is nothing was running; `unmeasured` is a
    # snapshot nobody could take, and folding that into `quiet` is how an
    # interference gate stops gating; `not-sampled` is nobody looked, which a
    # run that declared it would consult nothing cannot be faulted for.
    if [[ "${WINDOW%% *}" != "quiet" && "${WINDOW%% *}" != "not-sampled" ]]; then
        note_taint "pass ${pass}: ${WINDOW}"
    fi

    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    sleep 3  # let the run log flush before it is read

    PASS_LOG="$(phase_log "${SERVER_PID}")" || PASS_LOG=""
    if [[ -z "${PASS_LOG}" ]]; then
        echo "ERROR: no run log in ${LOG_DIR} is attributable to the pass-${pass}" \
             "server (pid ${SERVER_PID})" >&2
        exit 1
    fi
    PASS_KV_QUANT="$(log_kv_quant "${PASS_LOG}")" || PASS_KV_QUANT=""
    if [[ -z "${PASS_KV_QUANT}" ]]; then
        echo "ERROR: pass ${pass}: the run never said which KV codec it resolved" >&2
        exit 1
    fi
    if [[ -n "${KV_QUANT_SEEN:-}" && "${KV_QUANT_SEEN}" != "${PASS_KV_QUANT}" ]]; then
        echo "ERROR: pass ${pass} resolved ${PASS_KV_QUANT} where an earlier pass" \
             "resolved ${KV_QUANT_SEEN}; the passes are not repetitions of one" \
             "measurement" >&2
        exit 1
    fi
    KV_QUANT_SEEN="${PASS_KV_QUANT}"

    if ! python3 "${REPO_ROOT}/scripts/lib/published_run_log.py" "${PASS_LOG}" \
            --arm "${ARM}" \
            --expect-total "${REQUESTS_TOTAL}" \
            --last "${REQUESTS_PER_PASS}" \
            --expect-sampler-events "${SAMPLER_EVENTS}" \
            > "${PASS_DIR}/engine.json"; then
        echo "ERROR: pass ${pass}: no usable per-request record in ${PASS_LOG}" >&2
        exit 1
    fi

    # The fixed-prompt request is the one between the warmups and the cells, so
    # it is the first row of a read that keeps one more than the cells. The same
    # reader, over the same log, with the same checks — a second, looser parse
    # of one log is how two numbers taken from it stop agreeing.
    if [[ -n "${FIXED_PAYLOAD}" ]]; then
        if ! python3 "${REPO_ROOT}/scripts/lib/published_run_log.py" "${PASS_LOG}" \
                --arm "${ARM}" \
                --expect-total "${REQUESTS_TOTAL}" \
                --last "$((REQUESTS_PER_PASS + FIXED_REQUESTS))" \
                --expect-sampler-events "${SAMPLER_EVENTS}" \
                > "${PASS_DIR}/engine_with_fixed.json"; then
            echo "ERROR: pass ${pass}: the fixed-prompt request has no record in" \
                 "${PASS_LOG}" >&2
            exit 1
        fi
        if ! python3 "${REPO_ROOT}/scripts/lib/published_fixed_run.py" \
                --engine "${PASS_DIR}/engine_with_fixed.json" \
                --client "${PASS_DIR}/fixed.kv" \
                --memory "${PASS_DIR}/memory.txt" \
                --fit "${FIXED_RECORD}" \
                --pass-number "${pass}" \
                --memory-poll-ms "${MEMORY_POLL_MS}" \
                --cross-check-pct "${CROSS_CHECK_BAND_PCT}" \
                > "${PASS_DIR}/fixed.json"; then
            exit 1
        fi
        FIXED_FILES+=("${PASS_DIR}/fixed.json")
    fi

    if ! python3 "${REPO_ROOT}/scripts/lib/published_aggregate.py" pass \
            --index "${PASS_INDEX}" \
            --engine "${PASS_DIR}/engine.json" \
            --pass-number "${pass}" \
            --cross-check-pct "${CROSS_CHECK_BAND_PCT}" > "${PASS_DIR}/pass.json"; then
        exit 1
    fi
    PASS_FILES+=("${PASS_DIR}/pass.json")

    # What the run actually sampled under, off the engine's own event. The
    # checkpoint's file said what should happen; this says what did.
    PASS_SAMPLING="$(python3 -c 'import json, sys
print(json.dumps(json.load(open(sys.argv[1]))["sampling"], sort_keys=True))' \
        "${PASS_DIR}/engine.json")"
    if [[ -n "${SAMPLING_SEEN:-}" && "${SAMPLING_SEEN}" != "${PASS_SAMPLING}" ]]; then
        echo "ERROR: pass ${pass} sampled under ${PASS_SAMPLING} where an earlier" \
             "pass sampled under ${SAMPLING_SEEN}; the passes are not repetitions" \
             "of one measurement" >&2
        exit 1
    fi
    SAMPLING_SEEN="${PASS_SAMPLING}"

    echo "  [pass ${pass}] host window: ${WINDOW}"
    echo ""
done

# ── Report ───────────────────────────────────────────────────────────────────

META="${WORK}/meta.json"
SYNTHETIC_ARMS="${SYNTHETIC_ARMS}" ARM="${ARM}" \
MODEL_NAMESPACE="${MODEL_NAMESPACE}" MODEL_NAME="${MODEL_NAME}" \
WEIGHT_QUANT="${WEIGHT_QUANT}" KV_QUANT_SEEN="${KV_QUANT_SEEN}" \
MAX_CTX="${MAX_CTX}" SAMPLING_SEEN="${SAMPLING_SEEN}" \
WARMUPS_PER_PASS="${WARMUPS_PER_PASS}" PASSES="${PASSES}" \
MACRO_MAX_TOKENS="${MACRO_MAX_TOKENS}" \
UNVERIFIED_SAMPLES="${UNVERIFIED_SAMPLES}" \
SAMPLES_ROOT="${SAMPLES_ROOT}" TAINT="${TAINT}" \
BINARY_IDENTITY="${BINARY_IDENTITY}" \
HOST_WINDOWS="$(printf '%s\n' "${HOST_WINDOWS[@]}")" \
THERMAL_READINGS="$(printf '%s\n' ${THERMAL_READINGS[@]+"${THERMAL_READINGS[@]}"})" \
python3 - > "${META}" <<'PY'
import json, os

synthetic = os.environ["SYNTHETIC_ARMS"] == "true"
thermal = [r for r in os.environ["THERMAL_READINGS"].split("\n") if r]

print(json.dumps({
    **json.loads(os.environ["RMLX_IDENTITY_JSON"]),
    "synthetic_arms": synthetic,
    "arm": os.environ["ARM"],
    "model_namespace": os.environ["MODEL_NAMESPACE"],
    "model": os.environ["MODEL_NAME"],
    "weight_quant": os.environ["WEIGHT_QUANT"],
    "kv_quant": os.environ["KV_QUANT_SEEN"],
    "ctx_max": int(os.environ["MAX_CTX"]),
    "samples_root": os.environ["SAMPLES_ROOT"],
    # Not a published measurement: the samples it ran on are not the pinned ones.
    "unverified_samples": os.environ["UNVERIFIED_SAMPLES"] == "true",
    "protocol": {
        "passes": int(os.environ["PASSES"]),
        "warmups_per_pass": int(os.environ["WARMUPS_PER_PASS"]),
        "macro_max_tokens": int(os.environ["MACRO_MAX_TOKENS"]),
        # Read back from the engine, not re-derived from the checkpoint file.
        # null means the checkpoint is greedy and resolved no sampler.
        "sampling_resolved": json.loads(os.environ["SAMPLING_SEEN"]),
        "thinking": "on, counted as output",
        # Stated because the protocol does not, and because it is what the
        # run-to-run range means: the request sends no seed, the engine
        # substitutes its fixed default and seeds one RNG per request from it,
        # so the three passes replay one stream rather than sampling
        # independently. Holding the sampling still is the tighter estimator —
        # varying it would fold sampling variance into a figure presented as a
        # stability check. `divergent_samples` per cell is the measured check on
        # that claim.
        "seed_policy": "engine default, identical in all three passes",
    },
    # Which binary produced the numbers, and that it can write the events they
    # were read off. Re-checked at ingest: a rebuild between the run and the
    # record would leave the identity describing a different binary.
    "binary": json.loads(os.environ["BINARY_IDENTITY"]),
    # A run that consulted nothing files no reading taken off this machine.
    "host": {
        "pass_windows": [] if synthetic else os.environ["HOST_WINDOWS"].split("\n"),
        # Three points per pass — entry, midway and exit. `pmset -g therm` is
        # the last level the system posted, not a reading of the instant;
        # `powermetrics` is the instantaneous one and needs sudo, which a
        # non-interactive run does not have.
        "thermal": thermal,
        "thermal_source": "pmset -g therm",
        "taint": os.environ["TAINT"],
    },
}))
PY

RESULT="${RESULT_DIR}/$(date -u +%Y%m%dT%H%M%SZ)-${MODEL_NAME}-${ARM}.json"
set +e
python3 "${REPO_ROOT}/scripts/lib/published_aggregate.py" report \
    "${PASS_FILES[@]}" \
    --range-pct "${RANGE_REFUSAL_PCT}" \
    --macro-max-tokens "${MACRO_MAX_TOKENS}" \
    --meta "${META}" \
    ${FIXED_FILES[0]+--fixed "${FIXED_FILES[@]}"} \
    --json "${RESULT}"
REPORT_STATUS=$?
set -e

echo ""
echo "sampling: ${SAMPLING_SEEN}  (read back from the engine)"
echo "result: ${RESULT}"
if $UNVERIFIED_SAMPLES; then
    echo "UNVERIFIED SAMPLES: not a published measurement, not recordable."
fi
if [[ -n "${TAINT}" ]]; then
    echo "TAINTED: ${TAINT}"
fi
exit "${REPORT_STATUS}"
