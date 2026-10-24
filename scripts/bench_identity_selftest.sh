#!/usr/bin/env bash
# bench_identity_selftest.sh — mutation check for the commit and run-log sources
# of the bench scripts that spec_bench_selftest.sh does not drive.
#
# A bench row is filed under the commit its run identity names, and the run log
# a phase reads is the one its own server wrote. Each case drives the real script
# against a stub `rmlx` (and, for bench_decode_tps.sh, a stub server) and asserts
# what the script filed or passed on, not that it ran.
#
#   - lib/identity.sh: the identity's git_sha, its refusals, its normalisation.
#   - perf-iter/bench_decode_tps.sh: a git_sha is data, never source.
#   - perf_canary.sh: the filed sha is the identity's, the codec is read from the
#     run log the probe wrote, also when that log is shared with another process.
#
# No GPU, no model, no DB.
#
# Exit codes: 0 — every case behaved; 1 — at least one did not.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_bench_identity_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PASSED=0
FAILED=0

# ── Fake repo root ────────────────────────────────────────────────────────────
FAKE_ROOT="$WORK/repo"
mkdir -p "$FAKE_ROOT/target/release-perf" "$FAKE_ROOT/target/release"
ln -s "$REPO_ROOT/scripts" "$FAKE_ROOT/scripts"
git -C "$FAKE_ROOT" init -q
git -C "$FAKE_ROOT" -c user.name=selftest -c user.email=selftest@example.invalid \
	commit -q --allow-empty -m stub
HEAD_SHA="$(git -C "$FAKE_ROOT" rev-parse --short HEAD)"
HEAD_FULL="$(git -C "$FAKE_ROOT" rev-parse HEAD)"

BASE_IDENTITY='"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb"'

verdict() { # verdict <name> <what-it-proves> <bad-reason-or-empty>
	if [ -z "$3" ]; then
		printf 'ok    %-34s %s\n' "$1" "$2"
		PASSED=$((PASSED + 1))
	else
		printf 'FAIL  %-34s %s\n' "$1" "$2"
		printf '        %s\n' "$3"
		FAILED=$((FAILED + 1))
	fi
}

# ── lib/identity.sh ───────────────────────────────────────────────────────────

# identity_sha <identity-json> — rmlx_git_sha against the fake root; prints
# `<status>|<stdout>|<stderr>`.
identity_sha() {
	local out err status=0
	err="$WORK/identity.err"
	out="$(RMLX_IDENTITY_JSON="$1" bash -c '. "$1/scripts/lib/identity.sh"; rmlx_git_sha "$2"' \
		_ "$REPO_ROOT" "$FAKE_ROOT" 2>"$err")" || status=$?
	printf '%s|%s|%s' "$status" "$out" "$(cat "$err")"
}

WANT_REFUSAL='1||rmlx_identity_git_sha: RMLX_IDENTITY_JSON is not a JSON object'
got="$(identity_sha 'not json')"
verdict identity_refuses_invalid_json "invalid JSON is refused in one line, no traceback" \
	"$([ "$got" = "$WANT_REFUSAL" ] || echo "got [$got]")"
got="$(identity_sha '[1]')"
verdict identity_refuses_non_object "a JSON array is refused in one line, no traceback" \
	"$([ "$got" = "$WANT_REFUSAL" ] || echo "got [$got]")"

got="$(identity_sha "{$BASE_IDENTITY,\"git_sha\":\"unknown-dirty\"}")"
verdict identity_unknown_is_absent "an 'unknown...' git_sha is not a commit; HEAD is the fallback" \
	"$([ "$got" = "0|$HEAD_SHA|" ] || echo "got $got (want 0|$HEAD_SHA|)")"

got="$(identity_sha "{$BASE_IDENTITY,\"git_sha\":\"$HEAD_FULL\"}")"
verdict identity_full_sha_is_shortened "a resolvable sha is filed in the short form HEAD is filed in" \
	"$([ "$got" = "0|$HEAD_SHA|" ] || echo "got $got (want 0|$HEAD_SHA|)")"

got="$(identity_sha "{$BASE_IDENTITY,\"git_sha\":\"deadbeefcafe\"}")"
verdict identity_unresolvable_is_verbatim "a hex sha that resolves nowhere is kept as given" \
	"$([ "$got" = "0|deadbeefcafe|" ] || echo "got $got (want 0|deadbeefcafe|)")"

got="$(identity_sha "{$BASE_IDENTITY}")"
verdict identity_without_sha_uses_head "an identity with no git_sha leaves HEAD" \
	"$([ "$got" = "0|$HEAD_SHA|" ] || echo "got $got (want 0|$HEAD_SHA|)")"

# ── A port and a stub server ──────────────────────────────────────────────────

PORT="$(python3 -c '
import random, socket, sys
for _ in range(200):
    port = random.randint(18000, 19999)
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    except OSError:
        continue
    finally:
        probe.close()
    print(port)
    sys.exit(0)
sys.exit("no free port in 18000-19999")
')"

SERVER_PY="$WORK/stub_server.py"
cat >"$SERVER_PY" <<'PYEOF'
"""Canned server for bench_decode_tps.sh: /health, one ITL sample per request."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ring = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _json(self, body):
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.startswith("/metrics/cache"):
            self._json({"itl": ring})
        else:
            self._json({"status": "ok"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        ring.append({"step_mean_ms": 50.0, "step_count": 30})
        self._json(
            {
                "usage": {"completion_tokens": 30},
                "choices": [{"message": {"content": "hello world"}}],
            }
        )


HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
PYEOF

# ── perf-iter/bench_decode_tps.sh ─────────────────────────────────────────────

DECODE_STUB="$FAKE_ROOT/target/release/rmlx"
cat >"$DECODE_STUB" <<STUBEOF
#!/usr/bin/env bash
set -eu
case "\$1" in
metrics)
	case "\$2" in
	identity) echo '{$BASE_IDENTITY}' ;;
	--db)
		cp "\$6" "\$RMLX_HOME/recorded.json"
		rm -f "\$6"
		;;
	esac
	;;
serve)
	while [ \$# -gt 0 ]; do
		[ "\$1" = --port ] && port="\$2"
		shift
	done
	echo \$\$ >"\$RMLX_HOME/server.pid"
	exec python3 "$SERVER_PY" "\$port"
	;;
esac
STUBEOF
chmod +x "$DECODE_STUB"

# A value that is valid inside a Python dict literal and runs a command when it
# is spliced into source; as data it is only an odd string.
HOSTILE_SHA="a', 'b': __import__('os').system('touch $WORK/pwned'), 'c': 'd"

CASE="$WORK/decode_sha_is_data"
mkdir -p "$CASE/home" "$CASE/model/ns__m"
rm -f "$WORK/pwned"
set +e
(
	cd "$CASE"
	env -i \
		PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
		HOME="$WORK" \
		RMLX_HOME="$CASE/home" \
		MODEL_PATH="$CASE/model/ns__m" \
		KV_QUANT=k8v8 \
		PORT="$PORT" \
		WARMUP_RUNS=1 \
		MEASURE_RUNS=2 \
		RMLX_BIN="$DECODE_STUB" \
		RMLX_IDENTITY_JSON="{$BASE_IDENTITY,\"git_sha\":\"$HOSTILE_SHA\"}" \
		bash "$FAKE_ROOT/scripts/perf-iter/bench_decode_tps.sh" >"$CASE/out.log" 2>&1
)
decode_status=$?
set -e
# The script stops its own server; a script that died first may not have.
kill "$(cat "$CASE/home/server.pid" 2>/dev/null)" 2>/dev/null || true

bad=""
[ "$decode_status" -eq 0 ] || bad="exit=$decode_status"
[ ! -e "$WORK/pwned" ] || bad="$bad; the git_sha ran as code"
filed="$(python3 -c 'import json, sys
print(json.load(open(sys.argv[1])).get("git_sha"))' "$CASE/home/recorded.json" 2>/dev/null || echo "no record")"
[ "$filed" = "$HOSTILE_SHA" ] || bad="$bad; filed git_sha=$filed"
jsonl="$(python3 -c 'import json, sys
print(json.loads(open(sys.argv[1]).readline()).get("git_sha"))' "$CASE/home/metrics/perf-iter/baseline.jsonl" 2>/dev/null || echo "no line")"
[ "$jsonl" = "$HOSTILE_SHA" ] || bad="$bad; jsonl git_sha=$jsonl"
case "$(ls "$CASE/logs" 2>/dev/null)" in
*"'"* | *" "*) bad="$bad; the serve log name carries the raw sha" ;;
esac
tail -3 "$CASE/out.log" | sed 's/^/        | /' >"$CASE/tail.txt"
verdict decode_sha_is_data "a git_sha is data: never run, filed as given, not in file names" "${bad#; }"
[ -z "$bad" ] || cat "$CASE/tail.txt"

# ── perf_canary.sh ────────────────────────────────────────────────────────────

CANARY_STUB="$FAKE_ROOT/target/release-perf/rmlx"
cat >"$CANARY_STUB" <<STUBEOF
#!/usr/bin/env bash
set -eu
case "\$1" in
claim) exit 0 ;;
metrics) echo '{$BASE_IDENTITY}' ;;
baseline)
	printf '%s\n' "\$*" >>"\$RMLX_HOME/baseline_argv"
	case " \$* " in *" --run-class gate "*) [ -z "\${STUB_FAIL_GATE:-}" ] || exit 1 ;; esac
	if [ -n "\${STUB_SHARED_LOG:-}" ]; then
		log="\$RMLX_HOME/logs/\$STUB_SHARED_LOG"
	else
		log="\$RMLX_HOME/logs/\$(date +%s)-\$\$.jsonl"
	fi
	printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"rmlx start\",\"version\":\"9.9.9\",\"run_id\":\"stub\",\"pid\":\$\$}}" >>"\$log"
	printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"cache-type resolved\",\"arch\":\"Stub\",\"kv_quant\":\"stubq\"}}" >>"\$log"
	echo "decode_tps=12.5"
	;;
esac
STUBEOF
chmod +x "$CANARY_STUB"
CANARY_MODEL="$WORK/models/prism-ml__Ternary-Bonsai-2-27B-mlx-2bit"
mkdir -p "$CANARY_MODEL"
printf '{"quantization": {"group_size": 64, "bits": 2}}\n' >"$CANARY_MODEL/config.json"

# canary_case <name> <identity-json-or-empty> <shared-log-name-or-empty>
canary_case() {
	local name="$1" identity="$2" shared="$3" home
	home="$WORK/home_$name"
	mkdir -p "$home/logs"
	local env_extra=()
	[ -z "$identity" ] || env_extra+=("RMLX_IDENTITY_JSON=$identity")
	[ -z "${STUB_FAIL_GATE:-}" ] || env_extra+=("STUB_FAIL_GATE=1")
	if [ -n "$shared" ]; then
		printf '%s\n' '{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"rmlx start","version":"9.9.9","run_id":"stub","pid":4242}}' \
			'{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"cache-type resolved","arch":"Stub","kv_quant":"none"}}' \
			>"$home/logs/$shared"
		env_extra+=("STUB_SHARED_LOG=$shared")
	fi
	set +e
	env -i \
		PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
		HOME="$WORK" \
		RMLX_HOME="$home" \
		RMLX_O_MODELS_ROOT="$WORK/models" \
		${env_extra[@]+"${env_extra[@]}"} \
		bash "$FAKE_ROOT/scripts/perf_canary.sh" >"$home/out.log" 2>&1
	CANARY_STATUS=$?
	set -e
	CANARY_HOME="$home"
}

# csv_field <home> <column 1-based> — that column of the plain (non-turbo) row.
csv_field() {
	awk -F, -v col="$2" 'NR > 1 && $4 != "k8vturbo3" { print $col; exit }' "$1/bench/perf_canary.csv"
}

canary_case canary_identity_sha \
	"{$BASE_IDENTITY,\"git_sha\":\"feedbee\"}" ""
bad=""
[ "$CANARY_STATUS" -eq 0 ] || bad="exit=$CANARY_STATUS"
[ "$(csv_field "$CANARY_HOME" 2)" = feedbee ] || bad="$bad; csv git_sha=$(csv_field "$CANARY_HOME" 2)"
/usr/bin/grep -q -- '--git-sha feedbee' "$CANARY_HOME/baseline_argv" ||
	bad="$bad; the recorded baseline was not given --git-sha feedbee"
/usr/bin/grep -q -- '--label canary sha=feedbee' "$CANARY_HOME/baseline_argv" ||
	bad="$bad; the label does not carry the identity's sha"
[ "$(csv_field "$CANARY_HOME" 4)" = stubq ] || bad="$bad; csv kv_quant=$(csv_field "$CANARY_HOME" 4) (want stubq)"
diag="$(/usr/bin/grep -c -- '--record --run-class diagnostic' "$CANARY_HOME/baseline_argv" || true)"
gate="$(/usr/bin/grep -c -- '--record --run-class gate' "$CANARY_HOME/baseline_argv" || true)"
[ "$diag" = 8 ] || bad="$bad; $diag diagnostic records (want 8: 1+3 auto, 1+3 k8vturbo3)"
[ "$gate" = 1 ] || bad="$bad; $gate gate records (want 1)"
verdict canary_identity_sha "the canary files the identity's commit, reads the codec from the probe's log, and records 8 diagnostic runs and 1 gate run" "${bad#; }"

canary_case canary_same_second_log "" 20260903-000000-9.9.9.jsonl
bad=""
[ "$CANARY_STATUS" -eq 0 ] || bad="exit=$CANARY_STATUS"
[ "$(csv_field "$CANARY_HOME" 2)" = "$HEAD_SHA" ] || bad="$bad; csv git_sha=$(csv_field "$CANARY_HOME" 2) (want $HEAD_SHA)"
[ "$(csv_field "$CANARY_HOME" 4)" = stubq ] ||
	bad="$bad; csv kv_quant=$(csv_field "$CANARY_HOME" 4) (want stubq, not the earlier process's none)"
verdict canary_same_second_log "a probe that appends to an earlier process's log is attributed by pid" "${bad#; }"

STUB_FAIL_GATE=1 canary_case canary_gate_record_fails "" ""
bad=""
[ "$CANARY_STATUS" -eq 1 ] || bad="exit=$CANARY_STATUS (want 1)"
/usr/bin/grep -q 'the gate record failed' "$CANARY_HOME/out.log" || bad="$bad; no error naming the gate record"
verdict canary_gate_record_fails "a gate record that fails stops the canary" "${bad#; }"

printf '{}\n' >"$CANARY_MODEL/config.json"
canary_case canary_unrecordable_identity "" ""
bad=""
[ "$CANARY_STATUS" -eq 1 ] || bad="exit=$CANARY_STATUS (want 1)"
[ ! -s "$CANARY_HOME/baseline_argv" ] || bad="$bad; $(wc -l <"$CANARY_HOME/baseline_argv" | tr -d ' ') run(s) before the refusal"
/usr/bin/grep -q 'its identity cannot be recorded' "$CANARY_HOME/out.log" || bad="$bad; no error naming the identity"
verdict canary_unrecordable_identity "a snapshot whose identity cannot be recorded stops the canary before any run" "${bad#; }"

echo
echo "passed=$PASSED failed=$FAILED"
[ "$FAILED" -eq 0 ] || exit 1
