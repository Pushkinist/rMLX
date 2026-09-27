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
# The functions here answer two questions a bench harness has to answer the
# same way every time: is the server up, and which run log belongs to the server
# this phase started.
#
# A harness stops only the server it started, by its PID: `kill "$pid"; wait
# "$pid"`. When `wait` returns the server's Metal claim is free. A server some
# other process started is not the harness's to stop: `rmlx serve` refuses with
# exit 11 and names the holder, and the harness stops with that status.

# Wait for the server with PID $1 to be ready (polls /v1/models). A server that
# exits first returns its own exit status, so a refused claim (exit 11) stops
# the harness with the status the server gave.
wait_for_server() {
    local pid="$1"
    local url="http://127.0.0.1:${PORT}/v1/models"
    local attempts=0
    local max_attempts=60
    echo "  [wait] polling ${url} ..." >&2
    while true; do
        if curl -sf "${url}" > /dev/null 2>&1; then
            echo "  [wait] server ready." >&2
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            local status=0
            wait "${pid}" || status=$?
            echo "ERROR: server (pid ${pid}) exited with status ${status} before it was ready" >&2
            [[ ${status} -eq 0 ]] && status=1
            return "${status}"
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

# Where this process keeps its before/after log listings. Per-pid: two harnesses
# sharing an RMLX_HOME would otherwise overwrite each other's listing and
# attribute each other's run logs.
log_listing() { echo "${SCRATCH_DIR}/logs_$1.$$"; }

# Which run logs exist right now. Called before a phase starts its server.
snapshot_logs() {
    { ls -1 "${LOG_DIR}"/*.jsonl 2>/dev/null || true; } | sort \
        > "$(log_listing before)"
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
    local pid="$1" before after
    before="$(log_listing before)"
    after="$(log_listing after)"
    { ls -1 "${LOG_DIR}"/*.jsonl 2>/dev/null || true; } | sort > "${after}"
    comm -13 "${before}" "${after}" \
        | python3 "${REPO_ROOT}/scripts/lib/run_log_for_pid.py" --pid "${pid}"
    rm -f "${after}"
}

# The KV codec that log says the run resolved. Empty when it does not say.
log_kv_quant() {
    field_of "$(python3 "${REPO_ROOT}/scripts/lib/server_kv_quant.py" "$1")" kv_quant
}
