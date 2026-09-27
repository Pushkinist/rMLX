#!/usr/bin/env bash
# ssd_canary_selftest.sh — checks scripts/ssd_canary.sh and its two Makefile
# targets against a stub `rmlx` binary and a stub HTTP server. No GPU, no model,
# no metrics DB outside a throwaway directory.
#
# Each run case builds a fixture tree (a copy of the script, its identity
# helper, the real `prompts/ssd_bench/` and the Makefile, plus a stub binary at
# `target/release-perf/rmlx`), seeds a data root, runs the canary (`make
# ssd-canary`, or the script by hand from outside the tree), then `make
# ssd-canary-gate`, and asserts:
#   survive  every file in the data root before the run is byte-identical after
#   run-dir  every phase server ran in one directory that did not exist before,
#            and a second run gets another one
#   rm       every path the script removed lies inside that directory, and none
#            names a claim
#   ingest   the three phase records went to the DB `--print-db` names, which
#            is an exported RMLX_METRICS_DB when there is one
#   gate     the gate read that same DB
# A last case holds that `--tag`, which was parsed and never read, is refused.
#
# The stub's `metrics record` logs the DB it was pointed at and creates the file
# when absent (as the real one does), but never writes into an existing DB, so
# `survive` covers the DB as well. It refuses a DB outside the fixture.
#
# Usage: ssd_canary_selftest.sh [--script <path>] [--makefile <path>]
#   (defaults: the tree's own scripts/ssd_canary.sh and Makefile)
# Exit 0 — every case behaved; 1 — at least one did not; 2 — cannot run.

set -uo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ssd_canary.sh"
MAKEFILE="$REPO_ROOT/Makefile"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --script)   SCRIPT="${2:?--script requires a path}"; shift 2 ;;
        --makefile) MAKEFILE="${2:?--makefile requires a path}"; shift 2 ;;
        *) echo "ssd-canary-selftest: unknown argument: $1" >&2; exit 2 ;;
    esac
done

for tool in python3 sqlite3 curl make shasum; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "ssd-canary-selftest: unavailable: $tool not found" >&2
        exit 2
    }
done
for f in "$SCRIPT" "$MAKEFILE" "$REPO_ROOT/scripts/lib/identity.sh"; do
    [[ -f "$f" ]] || { echo "ssd-canary-selftest: unavailable: $f not found" >&2; exit 2; }
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_ssd_canary_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PASSED=0
FAILED=0
pass() { PASSED=$((PASSED + 1)); echo "  ok   $1"; }
fail() { FAILED=$((FAILED + 1)); echo "  FAIL $1" >&2; }

# ── Shims: fast `sleep`, logging `rm`, `cargo` that builds nothing ───────────

SHIM="$WORK/shim"
mkdir -p "$SHIM"
cat >"$SHIM/sleep" <<'EOF'
#!/bin/sh
exec /bin/sleep 0.05
EOF
cat >"$SHIM/rm" <<'EOF'
#!/bin/sh
printf '%s\n' "$*" >>"$SELFTEST_LOG/rm.log"
exec /bin/rm "$@"
EOF
cat >"$SHIM/cargo" <<'EOF'
#!/bin/sh
printf 'RMLX_METRICS_DB=%s ARGS=%s\n' "${RMLX_METRICS_DB-UNSET}" "$*" >>"$SELFTEST_LOG/cargo.log"
EOF
chmod +x "$SHIM"/*

cat >"$WORK/stub_server.py" <<'EOF'
import http.server, json, sys

hits = [0]

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, ctype="application/json"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self.send(json.dumps({"object": "list", "data": []}))
        elif self.path == "/metrics/cache":
            self.send(json.dumps({"models": [{"ssd_hits": hits[0]}]}))
        elif self.path == "/metrics":
            self.send('rmlx_ssd_evict_total{namespace="ssd-canary"} 5\n', "text/plain")
        else:
            self.send_error(404)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        hits[0] += 1
        self.send(json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}}]}))

http.server.HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
EOF

# ── Fixture tree ──────────────────────────────────────────────────────────────

# make_tree <dir> — the script, its helper, the prompts, the Makefile, a stub
# binary and an empty Cargo.lock (so the binary's workspace walk-up stops here).
make_tree() {
    local t="$1"
    mkdir -p "$t/scripts/lib" "$t/prompts" "$t/target/release-perf" "$t/models/stub-ns__stub-model-8bit"
    cp "$SCRIPT" "$t/scripts/ssd_canary.sh"
    cp "$REPO_ROOT/scripts/lib/identity.sh" "$t/scripts/lib/identity.sh"
    cp -R "$REPO_ROOT/prompts/ssd_bench" "$t/prompts/ssd_bench"
    cp "$MAKEFILE" "$t/Makefile"
    : >"$t/Cargo.lock"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'WORK=%q\n' "$WORK"
        cat <<'EOF'
LOG="$SELFTEST_LOG"
if [[ "${1:-} ${2:-}" == "metrics identity" ]]; then
    echo '{"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"stub"}'
    exit 0
fi
if [[ "${1:-}" == "serve" ]]; then
    port=""
    while [[ $# -gt 0 ]]; do
        case "$1" in --port) port="$2"; shift 2 ;; *) shift ;; esac
    done
    printf 'home=%s\n' "${RMLX_HOME-UNSET}" >>"$LOG/serve.log"
    exec python3 "$WORK/stub_server.py" "$port"
fi
if [[ "${1:-}" == "metrics" ]]; then
    shift
    db="" file="" sub=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --db) db="$2"; shift 2 ;;
            --db=*) db="${1#--db=}"; shift ;;
            --file) file="$2"; shift 2 ;;
            --file=*) file="${1#--file=}"; shift ;;
            record) sub=record; shift ;;
            *) shift ;;
        esac
    done
    [[ "$sub" == record && -n "$file" ]] || { echo "stub rmlx: unsupported metrics call" >&2; exit 2; }
    if [[ -z "$db" ]]; then
        if [[ -n "${RMLX_METRICS_DB:-}" ]]; then
            db="$RMLX_METRICS_DB"
        elif [[ -n "${RMLX_HOME:-}" ]]; then
            db="$RMLX_HOME/metrics/runs.db"
        else
            d="$PWD"
            while [[ "$d" != / && ! -f "$d/Cargo.lock" ]]; do d="$(dirname "$d")"; done
            if [[ -f "$d/Cargo.lock" ]]; then db="$d/.rmlx/metrics/runs.db"; else db="$HOME/.rmlx/metrics/runs.db"; fi
        fi
    fi
    case "$db" in
        "$WORK"/*) ;;
        *) printf 'outside-fixture db=%s\n' "$db" >>"$LOG/record.log"; exit 3 ;;
    esac
    mkdir -p "$(dirname "$db")"
    [[ -e "$db" ]] || : >"$db"
    printf 'db=%s\n' "$db" >>"$LOG/record.log"
    cat "$file" >>"$LOG/records.jsonl"
    echo >>"$LOG/records.jsonl"
    /bin/rm -f "$file"
    exit 0
fi
echo "stub rmlx: unsupported call: $*" >&2
exit 2
EOF
    } >"$t/target/release-perf/rmlx"
    chmod +x "$t/target/release-perf/rmlx"
}

# seed_root <dir> — a data root holding what a real one holds.
seed_root() {
    local r="$1"
    mkdir -p "$r/metrics/backups" "$r/metrics/buffer/pending" "$r/cache/kv/other" "$r/logs"
    sqlite3 "$r/metrics/runs.db" "CREATE TABLE observations(x); INSERT INTO observations VALUES (42);"
    printf 'backup\n' >"$r/metrics/backups/runs-old.db"
    printf '{"pending":true}\n' >"$r/metrics/buffer/pending/queued.json"
    printf 'index\n' >"$r/cache/kv/other/index.db"
    printf '{"log":1}\n' >"$r/logs/old.jsonl"
}

free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'
}

# run_env <case-dir> <cwd> <extra VAR=value...> -- <command...>
# Runs a command in a clean environment: no inherited RMLX_*, no make flags.
run_env() {
    local cdir="$1" cwd="$2"
    shift 2
    local extra=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do extra+=("$1"); shift; done
    shift
    (cd "$cwd" && env -i \
        HOME="$cdir/home" \
        PATH="$SHIM:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin" \
        TMPDIR="$cdir/tmp" \
        SELFTEST_LOG="$cdir/log" \
        ${extra[@]+"${extra[@]}"} \
        "$@")
}

# ── Run cases ─────────────────────────────────────────────────────────────────

# run_case <name> <root-mode> — root-mode is `exported` (RMLX_HOME points at a
# seeded temp root), `unset` (the tree's own .rmlx is the seeded root) or
# `metrics-db` (as `exported`, plus RMLX_METRICS_DB naming a DB elsewhere, which
# the binary prefers over the data root's, so the records must go there).
run_case() {
    local name="$1" mode="$2"
    local cdir="$WORK/$name"
    local tree="$cdir/tree"
    mkdir -p "$cdir/home" "$cdir/tmp" "$cdir/log"
    make_tree "$tree"
    local root home_env=() expected_db=""
    case "$mode" in
        exported)
            root="$cdir/data-root"
            home_env=("RMLX_HOME=$root") ;;
        metrics-db)
            root="$cdir/data-root"
            expected_db="$cdir/elsewhere/runs.db"
            home_env=("RMLX_HOME=$root" "RMLX_METRICS_DB=$expected_db") ;;
        *)
            root="$tree/.rmlx" ;;
    esac
    seed_root "$root"
    (cd "$root" && find . -type f -print0 | xargs -0 shasum -a 256) >"$cdir/before.sha"
    find "$root" -type d | sort >"$cdir/dirs_before"

    # With RMLX_HOME unset the script is run by hand from outside the tree, where
    # the binary's own DB resolution (a walk up from the working directory) lands
    # elsewhere: only the DB the script names keeps the records where the gate looks.
    local port run_how rc
    port="$(free_port)"
    if [[ "$mode" == unset ]]; then
        run_how="ssd_canary.sh run from outside the tree"
        run_env "$cdir" "$cdir" "VERIFIER_MODEL=$tree/models/stub-ns__stub-model-8bit" "PORT=$port" -- \
            bash "$tree/scripts/ssd_canary.sh" --ssd-gb 100 >"$cdir/run.out" 2>&1
        rc=$?
    else
        run_how="make ssd-canary"
        run_env "$cdir" "$tree" ${home_env[@]+"${home_env[@]}"} \
            "VERIFIER_MODEL=$tree/models/stub-ns__stub-model-8bit" "PORT=$port" -- \
            make --no-print-directory -f "$tree/Makefile" ssd-canary >"$cdir/run.out" 2>&1
        rc=$?
    fi
    if [[ $rc -eq 0 ]]; then
        pass "$name: $run_how exits 0"
    else
        fail "$name: $run_how exited $rc (tail: $(tail -3 "$cdir/run.out" | tr '\n' ' '))"
    fi

    # survive
    if (cd "$root" && shasum -a 256 -c --quiet "$cdir/before.sha") >"$cdir/survive.out" 2>&1; then
        pass "$name: every file of the data root survives byte-identical"
    else
        fail "$name: data root changed: $(tr '\n' ' ' <"$cdir/survive.out" | cut -c1-300)"
    fi

    # run-dir
    local homes run_dir=""
    homes="$(sort -u "$cdir/log/serve.log" 2>/dev/null | sed 's/^home=//')"
    if [[ -n "$homes" && "$(printf '%s\n' "$homes" | wc -l | tr -d ' ')" == 1 ]]; then
        run_dir="$homes"
    fi
    if [[ -z "$run_dir" ]]; then
        fail "$name: phase servers did not share one data root: [$(tr '\n' ' ' <"$cdir/log/serve.log" 2>/dev/null)]"
    elif grep -qxF "$run_dir" "$cdir/dirs_before"; then
        fail "$name: phase servers ran in $run_dir, which existed before the run"
    elif [[ ! -d "$run_dir" ]]; then
        fail "$name: phase servers ran in $run_dir, which is not a directory after the run"
    else
        pass "$name: phase servers ran in a directory the run created"
    fi

    # rm
    local bad_rm=""
    if [[ -f "$cdir/log/rm.log" ]]; then
        while IFS= read -r line; do
            for arg in $line; do
                case "$arg" in -*) continue ;; esac
                if [[ -z "$run_dir" || "$arg" != "$run_dir"/* ]] || [[ "$arg" == *claim* ]]; then
                    bad_rm="$bad_rm [$arg]"
                fi
            done
        done <"$cdir/log/rm.log"
    fi
    if [[ -z "$bad_rm" ]]; then
        pass "$name: every removal lies inside the run's own directory"
    else
        fail "$name: removed outside the run's own directory:$(printf '%s\n' $bad_rm | sort -u | head -3 | tr '\n' ' ')"
    fi

    # ingest
    local printed ingest_dbs n_records
    printed="$(run_env "$cdir" "$tree" ${home_env[@]+"${home_env[@]}"} -- \
        bash "$tree/scripts/ssd_canary.sh" --print-db 2>"$cdir/print.err")"
    ingest_dbs="$(sed -n 's/^db=//p' "$cdir/log/record.log" 2>/dev/null | sort -u)"
    n_records="$(grep -c '^db=' "$cdir/log/record.log" 2>/dev/null)"
    if grep -q '^outside-fixture' "$cdir/log/record.log" 2>/dev/null; then
        fail "$name: a record was pointed outside the fixture: $(grep '^outside-fixture' "$cdir/log/record.log" | head -1)"
    elif [[ -z "$printed" ]]; then
        fail "$name: --print-db printed nothing ($(head -1 "$cdir/print.err"))"
    elif [[ "$n_records" != 3 || "$ingest_dbs" != "$printed" ]]; then
        fail "$name: $n_records record(s) into [$ingest_dbs], --print-db names [$printed]"
    elif [[ -n "$expected_db" && "$printed" != "$expected_db" ]]; then
        fail "$name: records went to [$printed], not the RMLX_METRICS_DB [$expected_db]"
    else
        pass "$name: three records went to the DB --print-db names"
    fi
    local tag missing=""
    for tag in ssd-canary-populate ssd-canary-revisit ssd-canary-evict; do
        grep -q "tag=$tag " "$cdir/log/records.jsonl" 2>/dev/null || missing="$missing $tag"
    done
    if [[ -z "$missing" ]]; then
        pass "$name: one record per phase tag"
    else
        fail "$name: no record for:$missing"
    fi

    # gate
    run_env "$cdir" "$tree" ${home_env[@]+"${home_env[@]}"} -- \
        make --no-print-directory -f "$tree/Makefile" ssd-canary-gate SHA=abc1234 >"$cdir/gate.out" 2>&1
    rc=$?
    local gate_db
    gate_db="$(sed -n 's/^RMLX_METRICS_DB=\(.*\) ARGS=run .*metrics deltas.*/\1/p' "$cdir/log/cargo.log" 2>/dev/null | tail -1)"
    if [[ $rc -ne 0 ]]; then
        fail "$name: make ssd-canary-gate exited $rc ($(tail -1 "$cdir/gate.out"))"
    elif [[ -z "$ingest_dbs" || "$gate_db" != "$ingest_dbs" ]]; then
        fail "$name: the gate read [$gate_db], the run wrote [$ingest_dbs]"
    else
        pass "$name: the gate read the DB the run wrote"
    fi

    # A second run must not reuse the first run's directory: a reused one holds
    # the first run's SSD blocks, so POPULATE would not start cold.
    if [[ "$mode" == exported ]]; then
        find "$root" -type d | sort >"$cdir/dirs_before"
        : >"$cdir/log/serve.log"
        port="$(free_port)"
        run_env "$cdir" "$tree" ${home_env[@]+"${home_env[@]}"} \
            "VERIFIER_MODEL=$tree/models/stub-ns__stub-model-8bit" "PORT=$port" -- \
            make --no-print-directory -f "$tree/Makefile" ssd-canary >"$cdir/run2.out" 2>&1
        local second
        second="$(sort -u "$cdir/log/serve.log" | sed 's/^home=//')"
        if [[ -z "$second" || "$second" == "$run_dir" ]] || grep -qxF "$second" "$cdir/dirs_before"; then
            fail "$name: the second run's servers ran in [$second], not a new directory"
        else
            pass "$name: a second run gets a new directory"
        fi
    fi
}

echo "==> ssd-canary-selftest"
run_case exported-home exported
run_case unset-home unset
run_case metrics-db-env metrics-db

# --tag was parsed and never read; it is refused rather than silently dropped.
tag_dir="$WORK/tag"
mkdir -p "$tag_dir/home" "$tag_dir/tmp" "$tag_dir/log"
make_tree "$tag_dir/tree"
run_env "$tag_dir" "$tag_dir/tree" -- bash "$tag_dir/tree/scripts/ssd_canary.sh" --tag x \
    >"$tag_dir/out" 2>&1
rc=$?
if [[ $rc -eq 1 ]] && grep -q 'Unknown flag: --tag' "$tag_dir/out"; then
    pass "tag: --tag is refused as an unknown flag"
else
    fail "tag: --tag exited $rc: $(head -1 "$tag_dir/out")"
fi

echo "ssd-canary-selftest: $PASSED passed, $FAILED failed"
[[ $FAILED -eq 0 ]]
