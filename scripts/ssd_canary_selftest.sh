#!/usr/bin/env bash
# ssd_canary_selftest.sh — checks scripts/ssd_canary.sh and its two Makefile
# targets against a stub `rmlx` binary and a stub HTTP server. No GPU, no model,
# no metrics DB outside a throwaway directory.
#
# Each run case builds a fixture tree (a copy of the script, the helpers it
# reads, the real `prompts/ssd_bench/` and the Makefile, plus a stub binary at
# `target/release-perf/rmlx`), seeds a data root, runs the canary (`make
# ssd-canary`, or the script by hand from outside the tree), then `make
# ssd-canary-gate`, and asserts:
#   survive  every file in the data root before the run is byte-identical after
#   run-dir  every phase server was handed one absolute directory that did not
#            exist before, and a second run gets another one
#   cleanup  that directory's SSD blocks are gone at exit, its summary and
#            events DB are kept
#   rm       every path the script removed lies inside that directory, and none
#            names a claim; the script deletes through `rm` alone
#   measure  the summary and the POPULATE record carry what the stub server
#            wrote into its own data root (one events row per op, one index row)
#   label    the records carry the KV codec the server logged and the context
#            ceiling the servers were given
#   ingest   the three phase records went to the DB the binary resolves
#   gate     the gate read that same DB, scoped to the canary's prompts
# Single cases: a missing binary exits 125, and `--tag` is refused.
#
# The stub resolves its data root and DB the way the binary does (an absolute
# RMLX_HOME, else the checkout found by walking up for Cargo.lock, else
# $HOME/.rmlx; RMLX_METRICS_DB or `--db` for the DB). Its `metrics record` logs
# the DB it was pointed at and creates the file when absent, but never writes
# into an existing DB, so `survive` covers the DB as well. It refuses any path
# outside the fixture.
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
for f in "$SCRIPT" "$MAKEFILE" "$REPO_ROOT/scripts/lib/identity.sh" "$REPO_ROOT/scripts/lib/server_kv_quant.py"; do
    [[ -f "$f" ]] || { echo "ssd-canary-selftest: unavailable: $f not found" >&2; exit 2; }
done

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_ssd_canary_selftest.XXXXXX")"
WORK="$(cd "$WORK" && pwd -P)"
trap 'rm -rf "$WORK"' EXIT

PASSED=0
FAILED=0
pass() { PASSED=$((PASSED + 1)); echo "  ok   $1"; }
fail() { FAILED=$((FAILED + 1)); echo "  FAIL $1" >&2; }

# The stub server's own writes, which the summary and the records must carry.
STUB_KV_QUANT=k8v4
STUB_SPILL_US=2000
STUB_BLOCK_BYTES=4096

# ── Shims: fast `sleep`, logging `rm`, `cargo` that runs the stub ────────────

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
# `cargo build …` builds nothing; `cargo run … -- <args>` runs the tree's stub.
cat >"$SHIM/cargo" <<'EOF'
#!/bin/sh
printf 'ARGS=%s\n' "$*" >>"$SELFTEST_LOG/cargo.log"
while [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done
[ $# -gt 0 ] || exit 0
shift
exec ./target/release-perf/rmlx "$@"
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

# make_tree <dir> — the script, its helpers, the prompts, the Makefile, a stub
# binary and an empty Cargo.lock (so a workspace walk-up stops here).
make_tree() {
    local t="$1"
    mkdir -p "$t/scripts/lib" "$t/prompts" "$t/target/release-perf" "$t/models/stub-ns__stub-model-8bit"
    cp "$SCRIPT" "$t/scripts/ssd_canary.sh"
    cp "$REPO_ROOT/scripts/lib/identity.sh" "$REPO_ROOT/scripts/lib/server_kv_quant.py" "$t/scripts/lib/"
    cp -R "$REPO_ROOT/prompts/ssd_bench" "$t/prompts/ssd_bench"
    cp "$MAKEFILE" "$t/Makefile"
    : >"$t/Cargo.lock"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'WORK=%q\nKV=%q\nSPILL_US=%q\nBLOCK=%q\n' "$WORK" "$STUB_KV_QUANT" "$STUB_SPILL_US" "$STUB_BLOCK_BYTES"
        cat <<'EOF'
LOG="$SELFTEST_LOG"
inside() {
    case "$1" in
        "$WORK"/*) return 0 ;;
        *) printf 'outside-fixture %s\n' "$1" >>"$LOG/outside.log"; exit 3 ;;
    esac
}
resolve_home() {
    if [[ "${RMLX_HOME:-}" == /* ]]; then printf '%s\n' "$RMLX_HOME"; return; fi
    local d="$PWD"
    while [[ "$d" != / && ! -f "$d/Cargo.lock" ]]; do d="$(dirname "$d")"; done
    if [[ -f "$d/Cargo.lock" ]]; then printf '%s\n' "$d/.rmlx"; else printf '%s\n' "$HOME/.rmlx"; fi
}
if [[ "${1:-} ${2:-}" == "metrics identity" ]]; then
    echo '{"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"stub"}'
    exit 0
fi
if [[ "${1:-}" == "serve" ]]; then
    raw="${RMLX_HOME-UNSET}"
    home="$(resolve_home)"
    printf 'raw=%s home=%s args=%s\n' "$raw" "$home" "$*" >>"$LOG/serve.log"
    inside "$home"
    port=""
    while [[ $# -gt 0 ]]; do
        case "$1" in --port) port="$2"; shift 2 ;; *) shift ;; esac
    done
    mkdir -p "$home/logs" "$home/metrics" "$home/cache/kv/ssd-canary/blocks"
    printf '{"fields":{"message":"cache-type resolved","kv_quant":"%s"}}\n' "$KV" >"$home/logs/stub-$$.jsonl"
    sqlite3 "$home/metrics/runs.db" \
        "CREATE TABLE IF NOT EXISTS events(op TEXT, value REAL, notes TEXT);
         INSERT INTO events VALUES ('ssd_spill', $SPILL_US, '{\"bytes\":1048576}');
         INSERT INTO events VALUES ('ssd_hydrate', 1000, '{\"bytes\":524288}');"
    sqlite3 "$home/cache/kv/ssd-canary/index.db" \
        "CREATE TABLE IF NOT EXISTS kv_blocks(byte_size INTEGER);
         INSERT INTO kv_blocks VALUES ($BLOCK);"
    head -c "$BLOCK" /dev/zero >"$home/cache/kv/ssd-canary/blocks/b$$.bin"
    exec python3 "$WORK/stub_server.py" "$port"
fi
if [[ "${1:-}" == "metrics" ]]; then
    shift
    db="" file="" sub="" want_home=false rest=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --db) db="$2"; shift 2 ;;
            --db=*) db="${1#--db=}"; shift ;;
            --file) file="$2"; shift 2 ;;
            --file=*) file="${1#--file=}"; shift ;;
            --home) want_home=true; shift ;;
            path|record|deltas|query) sub="$1"; shift ;;
            *) rest+=("$1"); shift ;;
        esac
    done
    home="$(resolve_home)"
    [[ -n "$db" ]] || db="${RMLX_METRICS_DB:-$home/metrics/runs.db}"
    case "$sub" in
        path)
            if $want_home; then
                inside "$home"; mkdir -p "$home"; printf '%s\n' "$home"
            else
                inside "$db"; mkdir -p "$(dirname "$db")"; printf '%s\n' "$db"
            fi
            exit 0 ;;
        record)
            inside "$db"
            [[ -n "$file" ]] || { echo "stub rmlx: record needs --file" >&2; exit 2; }
            mkdir -p "$(dirname "$db")"
            [[ -e "$db" ]] || : >"$db"
            printf 'db=%s\n' "$db" >>"$LOG/record.log"
            cat "$file" >>"$LOG/records.jsonl"
            echo >>"$LOG/records.jsonl"
            /bin/rm -f "$file"
            exit 0 ;;
        deltas)
            printf 'db=%s args=%s\n' "$db" "${rest[*]-}" >>"$LOG/deltas.log"
            exit 0 ;;
        query)
            inside "$db"
            sql="${rest[*]-}"
            printf 'db=%s\n' "$db" >>"$LOG/query.log"
            echo "COUNT(*)"
            tag="$(printf '%s' "$sql" | sed -n 's/.*tag=\([a-z-]*\) .*/\1/p')"
            if [[ -n "$tag" ]]; then grep -c "tag=$tag " "$LOG/records.jsonl" 2>/dev/null || echo 0; fi
            exit 0 ;;
    esac
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

# summary_populate <iteration_summary.json> — the POPULATE figures the stub
# server's writes determine, and the events DB the run read them from.
summary_populate() {
    python3 - "$1" <<'PY' 2>/dev/null || echo MISSING
import json, sys
d = json.load(open(sys.argv[1]))
p = d["phases"]["POPULATE"]
print((p["mean_spill_us"], p["ssd_bytes_used"], p["spill_mb_per_s"], d["artifacts"]["events_db"]))
PY
}

# populate_record <records.jsonl> — the POPULATE record's measurement and labels.
populate_record() {
    python3 - "$1" <<'PY' 2>/dev/null || echo MISSING
import json, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line and "tag=ssd-canary-populate " in line:
        r = json.loads(line)
        m = {e["name"]: e["value"] for e in r["metrics"]}
        print((m["ssd_spill_ms"], m["ssd_bytes_used"], r["kv_quant"], r["ctx_max"]))
        break
else:
    print("MISSING")
PY
}

# ── Run cases ─────────────────────────────────────────────────────────────────

# run_case <name> <root-mode>
#   exported    RMLX_HOME points at a seeded temp root
#   unset       the tree's own .rmlx is the seeded root, and the script is run
#               by hand from outside the tree
#   metrics-db  as `exported`, plus RMLX_METRICS_DB naming a DB elsewhere
#   relative    RMLX_HOME is a relative path, which the binary ignores, so the
#               tree's own .rmlx is the data root
run_case() {
    local name="$1" mode="$2"
    local cdir="$WORK/$name"
    local tree="$cdir/tree"
    mkdir -p "$cdir/home" "$cdir/tmp" "$cdir/log"
    make_tree "$tree"
    local root home_env=() expected_db
    case "$mode" in
        exported)
            root="$cdir/data-root"
            home_env=("RMLX_HOME=$root") ;;
        metrics-db)
            root="$cdir/data-root"
            home_env=("RMLX_HOME=$root" "RMLX_METRICS_DB=$cdir/elsewhere/runs.db") ;;
        relative)
            root="$tree/.rmlx"
            home_env=("RMLX_HOME=rel-root") ;;
        *)
            root="$tree/.rmlx" ;;
    esac
    if [[ "$mode" == metrics-db ]]; then
        expected_db="$cdir/elsewhere/runs.db"
    else
        expected_db="$root/metrics/runs.db"
    fi
    seed_root "$root"
    (cd "$root" && find . -type f -print0 | xargs -0 shasum -a 256) >"$cdir/before.sha"
    find "$root" -type d | sort >"$cdir/dirs_before"

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
    if [[ -s "$cdir/log/outside.log" ]]; then
        fail "$name: the binary was pointed outside the fixture: $(head -1 "$cdir/log/outside.log")"
    fi

    # survive
    if (cd "$root" && shasum -a 256 -c --quiet "$cdir/before.sha") >"$cdir/survive.out" 2>&1; then
        pass "$name: every file of the data root survives byte-identical"
    else
        fail "$name: data root changed: $(tr '\n' ' ' <"$cdir/survive.out" | cut -c1-300)"
    fi

    # run-dir
    local raws homes run_dir=""
    raws="$(sed -n 's/^raw=\([^ ]*\) home=.*/\1/p' "$cdir/log/serve.log" 2>/dev/null | sort -u)"
    homes="$(sed -n 's/^raw=[^ ]* home=\([^ ]*\) .*/\1/p' "$cdir/log/serve.log" 2>/dev/null | sort -u)"
    if [[ -z "$raws" || "$(printf '%s\n' "$raws" | wc -l | tr -d ' ')" != 1 ]]; then
        fail "$name: phase servers were not handed one data root: [$(tr '\n' ' ' <"$cdir/log/serve.log" 2>/dev/null)]"
    elif [[ "$raws" != "$homes" ]]; then
        fail "$name: phase servers were handed [$raws], which the binary reads as [$homes]"
    elif grep -qxF "$raws" "$cdir/dirs_before"; then
        fail "$name: phase servers ran in $raws, which existed before the run"
    elif [[ ! -d "$raws" ]]; then
        fail "$name: phase servers ran in $raws, which is not a directory after the run"
    else
        run_dir="$raws"
        pass "$name: phase servers ran in an absolute directory the run created"
    fi
    if [[ "$mode" == relative && -e "$tree/rel-root" ]]; then
        fail "$name: the relative RMLX_HOME was used as a path"
    fi

    # cleanup
    if [[ -n "$run_dir" ]]; then
        if [[ -e "$run_dir/cache/kv" ]]; then
            fail "$name: the run's SSD blocks were left at $run_dir/cache/kv"
        elif [[ ! -f "$run_dir/iteration_summary.json" || ! -f "$run_dir/metrics/runs.db" ]]; then
            fail "$name: the run's summary or events DB was removed"
        else
            pass "$name: SSD blocks removed at exit, summary and events DB kept"
        fi
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

    # measure
    local summary="$run_dir/iteration_summary.json" got
    got="$(summary_populate "$summary")"
    if [[ "$got" == "(${STUB_SPILL_US}.0, ${STUB_BLOCK_BYTES}, 500.0, '$run_dir/metrics/runs.db')" ]]; then
        pass "$name: the summary carries the stub server's events and index"
    else
        fail "$name: summary POPULATE (mean_spill_us, ssd_bytes_used, spill_mb_per_s, events_db) = $got"
    fi
    got="$(populate_record "$cdir/log/records.jsonl")"
    if [[ "$got" == "(2.0, ${STUB_BLOCK_BYTES}.0, '${STUB_KV_QUANT}', 8192)" ]]; then
        pass "$name: the POPULATE record carries the measurement, the logged codec and the context ceiling"
    else
        fail "$name: POPULATE record (ssd_spill_ms, ssd_bytes_used, kv_quant, ctx_max) = $got"
    fi
    if grep -q -- '--max-ctx 8192' "$cdir/log/serve.log" && ! grep -v -- '--max-ctx 8192' "$cdir/log/serve.log" | grep -q .; then
        pass "$name: every phase server was given --max-ctx 8192"
    else
        fail "$name: a phase server ran without --max-ctx 8192"
    fi

    # ingest
    local ingest_dbs n_records
    ingest_dbs="$(sed -n 's/^db=//p' "$cdir/log/record.log" 2>/dev/null | sort -u)"
    n_records="$(grep -c '^db=' "$cdir/log/record.log" 2>/dev/null)"
    if [[ "$n_records" == 3 && "$ingest_dbs" == "$expected_db" ]]; then
        pass "$name: three records went to the DB the binary resolves"
    else
        fail "$name: $n_records record(s) into [$ingest_dbs], expected [$expected_db]"
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

    # Every record must fall under the prefix the gate scopes itself to, and
    # each phase must be its own prompt (prompts are content-addressed).
    local prefix prompts_ok
    prefix="$(sed -n 's/.*--prompt-prefix \([^ ]*\).*/\1/p' "$tree/Makefile" | head -1)"
    prompts_ok="$(python3 - "$cdir/log/records.jsonl" "$prefix" <<'PY' 2>&1
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
names = [r["prompt"]["name"] for r in recs]
bodies = {json.dumps(r["prompt"]["body"], sort_keys=True) for r in recs}
if not sys.argv[2]:
    print("the gate names no --prompt-prefix")
elif not all(n.startswith(sys.argv[2]) for n in names):
    print(f"prompt names {names} do not all start with {sys.argv[2]!r}")
elif len(bodies) != len(recs):
    print(f"{len(recs)} records share {len(bodies)} prompt bodies")
else:
    print("ok")
PY
)"
    if [[ "$prompts_ok" == ok ]]; then
        pass "$name: each phase has its own prompt, under the gate's prefix"
    else
        fail "$name: $prompts_ok"
    fi

    # gate
    run_env "$cdir" "$tree" ${home_env[@]+"${home_env[@]}"} -- \
        make --no-print-directory -f "$tree/Makefile" ssd-canary-gate SHA=abc1234 >"$cdir/gate.out" 2>&1
    rc=$?
    local gate_line gate_db
    gate_line="$(tail -1 "$cdir/log/deltas.log" 2>/dev/null)"
    gate_db="$(printf '%s' "$gate_line" | sed -n 's/^db=\([^ ]*\) args=.*/\1/p')"
    if [[ $rc -ne 0 ]]; then
        fail "$name: make ssd-canary-gate exited $rc ($(tail -1 "$cdir/gate.out"))"
    elif [[ "$gate_db" != "$expected_db" ]]; then
        fail "$name: the gate read [$gate_db], the run wrote [$expected_db]"
    elif [[ "$gate_line" != *"--prompt-prefix ssd-canary- "* ]]; then
        fail "$name: the gate is not scoped to the canary's prompts: $gate_line"
    else
        pass "$name: the gate read the DB the run wrote, scoped to the canary's prompts"
    fi

    # A second run must not reuse the first run's directory: a reused one holds
    # the first run's events, so the next run would read them as its own.
    if [[ "$mode" == exported ]]; then
        find "$root" -type d | sort >"$cdir/dirs_before"
        : >"$cdir/log/serve.log"
        port="$(free_port)"
        run_env "$cdir" "$tree" ${home_env[@]+"${home_env[@]}"} \
            "VERIFIER_MODEL=$tree/models/stub-ns__stub-model-8bit" "PORT=$port" -- \
            make --no-print-directory -f "$tree/Makefile" ssd-canary >"$cdir/run2.out" 2>&1
        local second
        second="$(sed -n 's/^raw=\([^ ]*\) .*/\1/p' "$cdir/log/serve.log" | sort -u)"
        if [[ -z "$second" || "$second" == "$run_dir" ]] || grep -qxF "$second" "$cdir/dirs_before"; then
            fail "$name: the second run's servers ran in [$second], not a new directory"
        else
            pass "$name: a second run gets a new directory"
        fi
    fi
}

echo "==> ssd-canary-selftest"

# Every deletion must go through `rm`, the one the shim can see.
if grep -nE '(/bin/rm|/usr/bin/rm|command rm|\\rm |unlink|rmdir|-delete|shred)' "$SCRIPT" >"$WORK/other_deletes"; then
    fail "static: the script deletes through something other than rm: $(head -1 "$WORK/other_deletes")"
else
    pass "static: the script deletes through rm alone"
fi

run_case exported-home exported
run_case unset-home unset
run_case metrics-db-env metrics-db
run_case relative-home relative

# A missing binary exits 125, as documented, before anything else asks it.
nobin="$WORK/no-binary"
mkdir -p "$nobin/home" "$nobin/tmp" "$nobin/log"
make_tree "$nobin/tree"
/bin/rm -f "$nobin/tree/target/release-perf/rmlx"
run_env "$nobin" "$nobin/tree" "VERIFIER_MODEL=$nobin/tree/models/stub-ns__stub-model-8bit" -- \
    bash "$nobin/tree/scripts/ssd_canary.sh" >"$nobin/out" 2>&1
rc=$?
if [[ $rc -eq 125 ]] && grep -q 'binary not found' "$nobin/out"; then
    pass "no-binary: exits 125 naming the missing binary"
else
    fail "no-binary: exited $rc: $(head -1 "$nobin/out")"
fi

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
