#!/usr/bin/env bash
# bench_server.sh — start, wait for and identify one rmlx server, for bench harnesses.
#
# Source it; do not execute it. The caller must have set:
#
#     REPO_ROOT    the checkout root (the lib/ readers are resolved from it)
#     PORT         the port the server was told to listen on
#     LOG_DIR      the run-log directory the server writes into
#     SCRATCH_DIR  a writable scratch directory
#
# The functions here answer three questions a bench harness has to answer the
# same way every time: is the machine free of competing MLX processes, is the
# server up, and which run log belongs to the server this phase started.

# Kill competing MLX processes and drop the Metal claim, so this run has the
# GPU to itself (CLAUDE.md hard rule 8).
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

# Read one `key=value` out of a block, or the empty string when absent.
field_of() {
    local block="$1" key="$2"
    echo "${block}" | sed -n "s/^${key}=//p" | tail -1
}

# Which run logs exist right now. Called before a phase starts its server.
snapshot_logs() {
    { ls -1 "${LOG_DIR}"/*.jsonl 2>/dev/null || true; } | sort \
        > "${SCRATCH_DIR}/logs_before"
}

# The run log a given pid wrote, among those that appeared since snapshot_logs.
#
# Identity, not order: "the newest" and "the last new one" both answer a
# different question, and any other rmlx process writing to this directory
# supplies a candidate. The server states its own pid in its `rmlx start`
# event, so the phase reads the log that names the server it started or none at
# all — reading metrics out of somebody else's log leaves no trace in the
# output.
phase_log() {
    local pid="$1"
    { ls -1 "${LOG_DIR}"/*.jsonl 2>/dev/null || true; } | sort \
        > "${SCRATCH_DIR}/logs_after"
    comm -13 "${SCRATCH_DIR}/logs_before" "${SCRATCH_DIR}/logs_after" \
        | python3 "${REPO_ROOT}/scripts/lib/run_log_for_pid.py" --pid "${pid}"
}

# The KV codec that log says the run resolved. Empty when it does not say.
log_kv_quant() {
    field_of "$(python3 "${REPO_ROOT}/scripts/lib/server_kv_quant.py" "$1")" kv_quant
}
