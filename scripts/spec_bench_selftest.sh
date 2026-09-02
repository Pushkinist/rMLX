#!/usr/bin/env bash
# spec_bench_selftest.sh — mutation check for scripts/spec_bench.sh.
#
# WHY THIS EXISTS
#
# spec_bench.sh writes permanent rows into the append-only metrics store, and
# the number it puts in `decode_tps_warm` is a throughput nothing downstream can
# sanity-check: 91 and 208 tok/s are both inside that metric's plausible-value
# bound, so a wrong-but-plausible value wins a `bests` cell and stays there. The
# one defence is that the value is the rate the engine measured, over the window
# the engine measured it in — and that is only true while somebody watches it be
# true.
#
# Each case drives the real script against a stub `rmlx` whose server streams a
# canned response and writes a canned round-loop `done` line, with the engine's
# reported rate and the prefill-contaminated `emitted / elapsed_ms` on that same
# line deliberately far apart. The suite asserts the ingested value, and for
# every refusal it asserts the reason as well — a guard that refuses for the
# wrong reason stops refusing when that reason moves.
#
# No GPU, no model, no DB: the stub answers `metrics record` without writing
# anything, and the assertions read the §8.5 buffer file the script emits.
#
# Exit codes: 0 — every case behaved; 1 — at least one did not.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_spec_bench_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PASSED=0
FAILED=0

# ── Fake repo root ────────────────────────────────────────────────────────────
#
# spec_bench.sh resolves both its binary and its lib/ helpers from `$0/..`, so a
# root holding the real scripts/ and a stub binary is how the stub gets used.
FAKE_ROOT="$WORK/repo"
mkdir -p "$FAKE_ROOT/target/release-perf"
ln -s "$REPO_ROOT/scripts" "$FAKE_ROOT/scripts"

# ── Shims ─────────────────────────────────────────────────────────────────────
#
# These keep the suite off this host. `pkill` would reach real processes; the
# script's inter-request sleeps are 5 s of nothing when the server is a stub;
# and its preflight deletes the Metal claim files of whatever is running here,
# which is the one thing a test must never do to a machine.
SHIM_DIR="$WORK/shims"
mkdir -p "$SHIM_DIR"
printf '#!/bin/sh\nexit 0\n' >"$SHIM_DIR/pkill"
printf '#!/bin/sh\nexit 0\n' >"$SHIM_DIR/sleep"
cat >"$SHIM_DIR/rm" <<'RMEOF'
#!/bin/sh
for a in "$@"; do
	case "$a" in /tmp/rmlx.*.claim) exit 0 ;; esac
done
exec /bin/rm "$@"
RMEOF
chmod +x "$SHIM_DIR/pkill" "$SHIM_DIR/sleep" "$SHIM_DIR/rm"

# ── Stub server ───────────────────────────────────────────────────────────────

SERVER_PY="$WORK/stub_server.py"
cat >"$SERVER_PY" <<'PYEOF'
"""Canned OpenAI-compatible server for the spec_bench selftest.

Streams STUB_TOKENS content chunks STUB_GAP_S apart, after a STUB_PREFILL_S
pause that stands in for prompt prefill. When the serve argv carried a drafter
it also appends one round-loop `done` line per request to the run log, taking
the `decode_tps` field from the next entry of STUB_DECODE_TPS_SEQ — a
comma-separated list of raw JSON values, the last one repeating, empty for a
line that carries no such field at all.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

TOKENS = int(os.environ["STUB_TOKENS"])
GAP_S = float(os.environ["STUB_GAP_S"])
PREFILL_S = float(os.environ["STUB_PREFILL_S"])
EMITTED = int(os.environ["STUB_EMITTED"])
ELAPSED_MS = float(os.environ["STUB_ELAPSED_MS"])
LOG_PATH = os.environ.get("STUB_LOG", "")
SPECULATIVE = os.environ.get("STUB_SPECULATIVE", "") == "1"
SEQ = [s for s in os.environ.get("STUB_DECODE_TPS_SEQ", "").split(",") if s]

served = 0


def done_line():
    """One round-loop `done` record, as tracing's JSON layer renders it."""
    global served
    raw = SEQ[min(served, len(SEQ) - 1)] if SEQ else ""
    served += 1
    fields = {
        "message": "mtp_generate_greedy: done",
        "rounds": 30,
        "emitted": EMITTED,
        "total_draft": 150,
        "total_accept": 98,
        "accept_rate": 98 / 150,
        "elapsed_ms": ELAPSED_MS,
        "block_size": 5,
    }
    if raw:
        fields["decode_tps"] = json.loads(raw)
    return json.dumps(
        {"timestamp": "2026-09-03T00:00:00Z", "level": "INFO", "fields": fields}
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(delta):
            payload = json.dumps({"choices": [{"delta": delta, "index": 0}]})
            self.wfile.write(f"data: {payload}\n\n".encode())
            self.wfile.flush()

        chunk({"role": "assistant"})
        time.sleep(PREFILL_S)
        for i in range(TOKENS):
            if i:
                time.sleep(GAP_S)
            chunk({"content": f"t{i} "})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

        if SPECULATIVE and LOG_PATH:
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(done_line() + "\n")


HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
PYEOF

# ── Stub binary ───────────────────────────────────────────────────────────────

STUB="$FAKE_ROOT/target/release-perf/rmlx"
cat >"$STUB" <<STUBEOF
#!/usr/bin/env bash
set -eu
case "\$1" in
metrics)
	case "\$2" in
	identity)
		echo '{"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb"}'
		;;
	record)
		echo "stub: recorded \$4" >&2
		;;
	esac
	;;
serve)
	port=8090
	speculative=0
	while [ \$# -gt 0 ]; do
		case "\$1" in
		--port) port="\$2" ;;
		--draft-model) speculative=1 ;;
		esac
		shift
	done
	log="\$RMLX_HOME/logs/\$(date +%s)-\$\$.jsonl"
	: >"\$log"
	export STUB_LOG="\$log" STUB_SPECULATIVE="\$speculative"
	exec python3 "$SERVER_PY" "\$port"
	;;
esac
STUBEOF
chmod +x "$STUB"

mkdir -p "$WORK/verifier" "$WORK/drafter"

# A port nothing else on this host is likely to hold.
PORT=$((18000 + RANDOM % 2000))

# ── Case driver ───────────────────────────────────────────────────────────────

# run_case <name> <want-exit> <what-it-proves> [KEY=VALUE ...] [GREP:pat ...]
#
# Stub defaults describe a run whose engine-reported rate (200) and whose
# emitted / elapsed_ms on the same line (128 / 2.56 s = 50) cannot be confused.
run_case() {
	CASE_NAME="$1"
	local want="$2"
	CASE_WHAT="$3"
	shift 3

	CASE_HOME="$WORK/home_$CASE_NAME"
	mkdir -p "$CASE_HOME/logs" "$CASE_HOME/metrics/buffer/pending"

	local env_pairs=() greps=() a
	for a in "$@"; do
		case "$a" in
		GREP:*) greps+=("${a#GREP:}") ;;
		*) env_pairs+=("$a") ;;
		esac
	done

	CASE_OUT="$WORK/$CASE_NAME.log"
	local got=0
	set +e
	env -i \
		PATH="$SHIM_DIR:/usr/bin:/bin:/usr/sbin:/sbin" \
		HOME="$WORK" \
		RMLX_HOME="$CASE_HOME" \
		VERIFIER_MODEL="$WORK/verifier" \
		DRAFTER_MODEL="$WORK/drafter" \
		STUB_TOKENS=6 \
		STUB_GAP_S=0.01 \
		STUB_PREFILL_S=0.05 \
		STUB_EMITTED=128 \
		STUB_ELAPSED_MS=2560 \
		STUB_DECODE_TPS_SEQ='"Some(200.0)"' \
		${env_pairs[@]+"${env_pairs[@]}"} \
		bash "$FAKE_ROOT/scripts/spec_bench.sh" --port "$PORT" >"$CASE_OUT" 2>&1
	got=$?
	set -e
	pkill -f "$SERVER_PY" 2>/dev/null || true

	CASE_BAD=""
	[ "$got" -ne "$want" ] && CASE_BAD="exit=$got (want $want)"
	local g
	for g in ${greps[@]+"${greps[@]}"}; do
		grep -qE "$g" "$CASE_OUT" || note_bad "missing /$g/"
	done
}

# Add a failure reason to the case being judged.
note_bad() {
	if [ -z "$CASE_BAD" ]; then CASE_BAD="$1"; else CASE_BAD="$CASE_BAD; $1"; fi
}

# Print the verdict for the case just run, after its extra assertions.
verdict() {
	if [ -z "$CASE_BAD" ]; then
		printf 'ok    %-30s %s\n' "$CASE_NAME" "$CASE_WHAT"
		PASSED=$((PASSED + 1))
	else
		printf 'FAIL  %-30s %s\n' "$CASE_NAME" "$CASE_WHAT"
		printf '        %s\n' "$CASE_BAD"
		tail -3 "$CASE_OUT" | sed 's/^/        | /'
		FAILED=$((FAILED + 1))
	fi
}

# record_of <config> — the buffer path this case ingested, or the empty string.
record_of() {
	local hits
	hits=$(echo "$CASE_HOME"/metrics/buffer/pending/*-"$1".json)
	case "$hits" in
	*'*'*) echo "" ;;
	*) echo "${hits%% *}" ;;
	esac
}

# metric_of <config> <field> — `value` or `stddev` of decode_tps_warm.
metric_of() {
	local path
	path="$(record_of "$1")"
	[ -z "$path" ] && return 0
	python3 -c 'import json, sys
rec = json.load(open(sys.argv[1]))
for m in rec.get("metrics", []):
    if m.get("name") == "decode_tps_warm":
        print(m.get(sys.argv[2], ""))
        break' "$path" "$2"
}

# notes_of <config>
notes_of() {
	local path
	path="$(record_of "$1")"
	[ -z "$path" ] && return 0
	python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get("notes", ""))' \
		"$path"
}

# close_to <got> <want> <tolerance-fraction>
close_to() {
	python3 -c 'import sys
got, want, tol = sys.argv[1:4]
try:
    g, w = float(got), float(want)
except ValueError:
    sys.exit(1)
sys.exit(0 if abs(g - w) <= float(tol) * abs(w) else 1)' "$1" "$2" "$3"
}

# no_row <config> — nothing was ingested for that arm.
no_row() {
	[ -n "$(record_of "$1")" ] && note_bad "a $1 row was ingested anyway"
	return 0
}

echo "spec_bench_selftest: stub server on 127.0.0.1:$PORT"
echo

# ── Cases ─────────────────────────────────────────────────────────────────────

# The engine measured 200 tok/s over its decode window; emitted / elapsed_ms on
# the same line is 50. Ingesting 50 is the defect.
run_case engine_rate_is_ingested 0 \
	"mtp row carries the engine's rate, not emitted/elapsed_ms"
got="$(metric_of mtp value)"
close_to "$got" 200.0 0.001 || note_bad "mtp decode_tps_warm=$got (want 200.0)"
verdict

run_case window_is_declared 0 \
	"each row says which window its rate was measured over"
case "$(notes_of mtp)" in
*decode_window=engine*) ;;
*) note_bad "mtp notes=$(notes_of mtp)" ;;
esac
case "$(notes_of normal)" in
*decode_window=client_sse*) ;;
*) note_bad "normal notes=$(notes_of normal)" ;;
esac
verdict

# Three measured events at 190 / 200 / 210 have a median of 200 and a sample
# stddev of 10. A hard 0.0 in that column claims a spread never measured.
run_case stddev_is_measured 0 \
	"the mtp spread comes from the measured runs" \
	'STUB_DECODE_TPS_SEQ="Some(1.0)","Some(190.0)","Some(200.0)","Some(210.0)"'
got="$(metric_of mtp value)"
close_to "$got" 200.0 0.001 || note_bad "mtp decode_tps_warm=$got (want median 200.0)"
got="$(metric_of mtp stddev)"
close_to "$got" 10.0 0.001 || note_bad "mtp stddev=$got (want 10.0)"
verdict

# The first done event is the warmup and must not reach the ingested value. Its
# rate sits outside the measured three on purpose: a warmup that landed inside
# them could not move the median it is supposed to be kept out of.
run_case warmup_event_excluded 0 \
	"the warmup request's rate is not aggregated" \
	'STUB_DECODE_TPS_SEQ="Some(300.0)","Some(190.0)","Some(200.0)","Some(210.0)"'
got="$(metric_of mtp value)"
close_to "$got" 200.0 0.001 || note_bad "mtp decode_tps_warm=$got (want 200.0, not 205.0)"
verdict

# A binary older than the corrected field logs decode_tps as a bare number, and
# that number is the prefill-inclusive one — the defect wearing the corrected
# field's name.
run_case bare_number_refused 1 \
	"a bare-number decode_tps is refused, not read" \
	'STUB_DECODE_TPS_SEQ=200.0' \
	'GREP:bare number' \
	'GREP:still counted prefill'
no_row mtp
verdict

run_case missing_field_refused 1 \
	"a done line with no decode_tps is refused" \
	'STUB_DECODE_TPS_SEQ=' \
	'GREP:carries no decode_tps field'
no_row mtp
verdict

# `None` is the engine saying it has no measurable rate. Substituting a number
# for it — a zero, or a wall clock — is a fabricated measurement.
run_case none_rate_refused 1 \
	"an unmeasurable rate is refused, not substituted" \
	'STUB_DECODE_TPS_SEQ="None"' \
	'GREP:no measurable decode rate'
no_row mtp
verdict

# Eight tokens 0.1 s apart behind a 1.5 s prefill: roughly 9.3 tok/s over the
# decode window against 3.5 over the whole request. The no-drafter arm has to
# report the window, or its row is not comparable with the speculative one it is
# subtracted from. The band is wide because each chunk carries a few ms of HTTP
# framing; it is nowhere near the whole-request figure.
run_case normal_arm_excludes_prefill 0 \
	"the no-drafter arm times the decode window only" \
	'STUB_TOKENS=8' 'STUB_GAP_S=0.1' 'STUB_PREFILL_S=1.5'
got="$(metric_of normal value)"
close_to "$got" 9.3 0.3 || note_bad "normal decode_tps_warm=$got (want ~9.3, not ~3.5)"
verdict

echo
echo "passed=$PASSED failed=$FAILED"
[ "$FAILED" -eq 0 ] || exit 1
