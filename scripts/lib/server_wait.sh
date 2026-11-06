# Wait for one rmlx server to answer /v1/models, for a time the caller gives; source it.
#
# Source it; do not execute it. The caller must have set PORT, the port the
# server was told to listen on.
#
# The caller starts the server in its own shell, so the wait can read the exit
# status of a server that died. A server some other process started is not the
# caller's to stop: `rmlx serve` refuses with exit 11 and names the holder, and
# the caller stops with that status.

# Wait for the server with PID $1 to be ready (polls /v1/models) for at most $2
# seconds of real time (default 120). A server that exits first returns its own
# exit status, so a refused claim (exit 11) stops the harness with the status the
# server gave. The limit is read from the shell's own clock, so a `sleep` that a
# self-test shortens does not shorten it.
wait_for_server() {
    local pid="$1" limit="${2:-120}"
    local url="http://127.0.0.1:${PORT}/v1/models"
    local started=${SECONDS}
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
        if (( SECONDS - started >= limit )); then
            echo "ERROR: server did not start within ${limit}s" >&2
            return 1
        fi
        sleep 2
    done
}
