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
pause that stands in for prompt prefill, then a usage block. It times its own
sends and publishes the aggregate at /metrics/cache in the shape the real
server's ITL ring has, so the engine-side figure the bench reads and the client
window it cross-checks against describe the same gaps. STUB_ITL_MEAN_MS
overrides that aggregate, which is how a disagreement is staged;
STUB_ITL_SUPPRESS leaves the ring untouched, as the real server does for a
response too short to have an interval.

When the serve argv carried a drafter it also appends a round-loop `done` line
per request to the run log — written before the final flush, so a reader that
sees `[DONE]` sees the line too. `decode_tps` comes from the next entry of
STUB_DECODE_TPS_SEQ, a comma-separated list of raw JSON values with the last one
repeating, empty for a line carrying no such field. STUB_DONE_LINES caps how
many requests get a line at all.
"""

import json
import os
import statistics
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
DONE_LINES = int(os.environ.get("STUB_DONE_LINES", "-1"))
ITL_MEAN_OVERRIDE = os.environ.get("STUB_ITL_MEAN_MS", "")
ITL_SUPPRESS = os.environ.get("STUB_ITL_SUPPRESS", "") == "1"
ITL_SUPPRESS_AFTER = int(os.environ.get("STUB_ITL_SUPPRESS_AFTER", "-1"))
USAGE_TOKENS = int(os.environ.get("STUB_USAGE_TOKENS", "-1"))
PROMPT_TOKENS = int(os.environ.get("STUB_PROMPT_TOKENS", "-1"))
BOUND_FLAG = os.environ.get("STUB_BOUND_FLAG", "")

served = 0
itl_ring = []


ROUNDS = 30
# Pairwise distinct per-round figures (10 / 30 / 20 ms): with a round loop of
# 1500 the draft and loop residual were both 10, and a swap of the two
# derivations passed every assertion in this suite.
DRAFT_MS = 300.0
VERIFY_MS = 900.0
ROUND_MS = 1800.0
SEED_EMITTED = int(os.environ.get("STUB_SEED_EMITTED", "1"))
# 98 accepted over 30 rounds could produce 128; the rounds emitted 127. The stub
# deliberately sits one token UNDER that budget, because the check this exercises
# is the three counts adding up rather than the inequality it replaced — which
# only bit at the budget, and on real requests bit on one of four loops.
TOTAL_ACCEPT = int(os.environ.get("STUB_TOTAL_ACCEPT", "98"))
DECODE_CONFIG = os.environ.get("STUB_DECODE_CONFIG", "mtp/block=5")
# The engine composes both from one block, so the stub cannot make them
# disagree by accident.
BLOCK_SIZE = int(
    next(t for t in DECODE_CONFIG.split(",") if t.split("=")[0].endswith("/block")).split("=")[1]
)
DERIVED_OVERRIDE = os.environ.get("STUB_DERIVED_OVERRIDE", "")
DROP_FIELDS = [f for f in os.environ.get("STUB_DROP_FIELDS", "").split(",") if f]


def done_line():
    """One round-loop `done` record, as tracing's JSON layer renders it.

    Carries the raw counters and the per-round figures the engine derives from
    them, consistent with each other. STUB_DERIVED_OVERRIDE (`name=value`)
    breaks one of them, which is how the reader's cross-check is staged.
    """
    raw = SEQ[min(served, len(SEQ) - 1)] if SEQ else ""
    # The engine counts this at its own emit site, independently of the seed —
    # which is the whole point: a drifting seed must break the sum rather than
    # be absorbed by it. So the default is the healthy round count, not
    # `EMITTED - SEED_EMITTED`.
    round_emitted = int(os.environ.get("STUB_EMITTED_IN_ROUNDS", EMITTED - 1))
    fields = {
        "message": "mtp_generate_greedy: done",
        "rounds": ROUNDS,
        "emitted": EMITTED,
        "seed_emitted": SEED_EMITTED,
        "emitted_in_rounds": round_emitted,
        "total_draft": 150,
        "total_accept": TOTAL_ACCEPT,
        "accept_rate": TOTAL_ACCEPT / 150,
        "accepted_per_step": TOTAL_ACCEPT / ROUNDS,
        "tokens_per_round": round_emitted / ROUNDS,
        "elapsed_ms": ELAPSED_MS,
        "prefill_ms": 100.0,
        "round_ms": ROUND_MS,
        "draft_ms": DRAFT_MS,
        "verifier_ms": VERIFY_MS,
        "draft_ms_per_round": DRAFT_MS / ROUNDS,
        "verify_ms_per_round": VERIFY_MS / ROUNDS,
        "loop_ms_per_round": (ROUND_MS - DRAFT_MS - VERIFY_MS) / ROUNDS,
        "block_size": BLOCK_SIZE,
        "decode_config": DECODE_CONFIG,
    }
    if DERIVED_OVERRIDE:
        name, _, value = DERIVED_OVERRIDE.partition("=")
        fields[name] = float(value)
    for name in DROP_FIELDS:
        fields.pop(name, None)
    if raw:
        fields["decode_tps"] = json.loads(raw)
    return json.dumps(
        {"timestamp": "2026-09-03T00:00:00Z", "level": "INFO", "fields": fields}
    )


def push_itl(sends):
    """Append this request's ITL aggregate, as the engine's ring holds it."""
    if ITL_SUPPRESS or len(sends) < 2:
        return
    if 0 <= ITL_SUPPRESS_AFTER <= served:
        return
    gaps = [(b - a) * 1000.0 for a, b in zip(sends, sends[1:])]
    mean_ms = float(ITL_MEAN_OVERRIDE) if ITL_MEAN_OVERRIDE else statistics.fmean(gaps)
    itl_ring.append(
        {
            "model_id": "stub",
            "p50_ms": statistics.median(gaps),
            "p95_ms": max(gaps),
            "step_mean_ms": mean_ms,
            "step_count": len(sends),
        }
    )


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
            self._json({"models": [], "ttft": [], "itl": itl_ring})
        else:
            self._json({"object": "list", "data": []})

    def do_POST(self):
        global served
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(payload):
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        sends = []
        chunk({"choices": [{"delta": {"role": "assistant"}, "index": 0}]})
        time.sleep(PREFILL_S)
        # Send against a fixed schedule rather than sleeping between chunks:
        # the per-chunk framing cost then falls inside the gap instead of being
        # added to it, so the stub really does stream at 1/GAP_S and the rate it
        # reports is the rate the client can observe.
        start = time.monotonic()
        for i in range(TOKENS):
            due = start + i * GAP_S
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            chunk({"choices": [{"delta": {"content": f"t{i} "}, "index": 0}]})
            sends.append(time.monotonic())
        reported = USAGE_TOKENS if USAGE_TOKENS >= 0 else TOKENS
        usage = {"completion_tokens": reported}
        if PROMPT_TOKENS >= 0:
            usage["prompt_tokens"] = PROMPT_TOKENS
        chunk({"choices": [], "usage": usage})

        push_itl(sends)
        if SPECULATIVE and LOG_PATH and (DONE_LINES < 0 or served < DONE_LINES):
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(done_line() + "\n")
        served += 1

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


server = HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler)
if BOUND_FLAG:
    with open(BOUND_FLAG, "a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
server.serve_forever()
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
	if [ -z "\${STUB_PID_SUPPRESS:-}" ]; then
		printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"rmlx start\",\"version\":\"9.9.9\",\"run_id\":\"stub\",\"pid\":\$\$}}" >>"\$log"
	fi
	if [ -n "\${STUB_DECOY_LOG:-}" ]; then
		printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"rmlx start\",\"version\":\"9.9.9\",\"run_id\":\"decoy\",\"pid\":999999}}" \
			>"\$RMLX_HOME/logs/zzz-decoy-\$\$.jsonl"
	fi
	if [ -z "\${STUB_KV_QUANT_SUPPRESS:-}" ]; then
		printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"cache-type resolved\",\"arch\":\"Stub\",\"kv_quant\":\"\${STUB_KV_QUANT:-mixed_k8g64_v4g64}\"}}" >>"\$log"
	fi
	export STUB_LOG="\$log" STUB_SPECULATIVE="\$speculative"
	exec python3 "$SERVER_PY" "\$port"
	;;
esac
STUBEOF
chmod +x "$STUB"

# A snapshot the identity reader can actually read: `ns__model` plus the
# `config.json` the weight-quant label comes from.
VERIFIER_DIR="$WORK/models/stub-ns__stub-model"
mkdir -p "$VERIFIER_DIR" "$WORK/drafter"
printf '%s\n' '{"quantization": {"mode": "mxfp8", "bits": 8, "group_size": 32}}' \
	>"$VERIFIER_DIR/config.json"

# A second snapshot with no config.json at all, for the refusal case.
BARE_DIR="$WORK/models/bare-ns__bare-model"
mkdir -p "$BARE_DIR"

# A port this host is not already using. Probed rather than assumed: a foreign
# listener would answer the script's readiness poll and the suite would measure
# it instead of the stub. The stub also records the pid that bound the port, so
# a case whose server never came up is a failure and not a silent substitution.
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

# ── Case driver ───────────────────────────────────────────────────────────────

# run_case <name> <want-exit> <what-it-proves> [KEY=VALUE ...] [ARGS:flag ...]
#          [GREP:pat ...]
#
# Stub defaults describe a run streaming eight tokens 50 ms apart, so the rate
# the engine reports (20) is what the wire actually carried and the client
# cross-check passes on its own. The emitted / elapsed_ms on the same line
# (128 / 2.56 s = 50) is 2.5x away from it and cannot be confused with it.
run_case() {
	CASE_NAME="$1"
	local want="$2"
	CASE_WHAT="$3"
	shift 3

	CASE_HOME="$WORK/home_$CASE_NAME"
	mkdir -p "$CASE_HOME/logs" "$CASE_HOME/metrics/buffer/pending"

	local env_pairs=() greps=() extra_args=() a
	for a in "$@"; do
		case "$a" in
		GREP:*) greps+=("${a#GREP:}") ;;
		ARGS:*) extra_args+=("${a#ARGS:}") ;;
		*) env_pairs+=("$a") ;;
		esac
	done

	CASE_OUT="$WORK/$CASE_NAME.log"
	local got=0
	set +e
	env -i \
		PATH="$SHIM_DIR:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
		HOME="$WORK" \
		RMLX_HOME="$CASE_HOME" \
		VERIFIER_MODEL="$VERIFIER_DIR" \
		DRAFTER_MODEL="$WORK/drafter" \
		STUB_TOKENS=8 \
		STUB_GAP_S=0.05 \
		STUB_PREFILL_S=0.05 \
		STUB_EMITTED=128 \
		STUB_ELAPSED_MS=2560 \
		STUB_DECODE_TPS_SEQ='"Some(20.0)"' \
		STUB_PROMPT_TOKENS=1234 \
		STUB_BOUND_FLAG="$CASE_HOME/stub_bound" \
		${env_pairs[@]+"${env_pairs[@]}"} \
		bash "$FAKE_ROOT/scripts/spec_bench.sh" --port "$PORT" \
		${extra_args[@]+"${extra_args[@]}"} >"$CASE_OUT" 2>&1
	got=$?
	set -e
	pkill -f "$SERVER_PY" 2>/dev/null || true

	CASE_BAD=""
	[ "$got" -ne "$want" ] && CASE_BAD="exit=$got (want $want)"
	# Only meaningful for a case that got as far as starting a server: one
	# that refuses before then has no port to have bound.
	if grep -q '\[server\] starting' "$CASE_OUT"; then
		[ -s "$CASE_HOME/stub_bound" ] ||
			note_bad "the stub never bound port $PORT — something else answered"
	fi
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

# field_of_record <config> <top-level field> — "" when absent, "null" for JSON null.
field_of_record() {
	local path
	path="$(record_of "$1")"
	[ -z "$path" ] && return 0
	python3 -c 'import json, sys
rec = json.load(open(sys.argv[1]))
if sys.argv[2] not in rec:
    print("")
else:
    v = rec[sys.argv[2]]
    print("null" if v is None else v)' "$path" "$2"
}

# notes_of <config>
notes_of() { field_of_record "$1" notes; }

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

# The engine measured 20 tok/s over its decode window; emitted / elapsed_ms on
# the same line is 50. Ingesting 50 is the defect.
run_case engine_rate_is_ingested 0 \
	"mtp row carries the engine's rate, not emitted/elapsed_ms"
got="$(metric_of mtp value)"
close_to "$got" 20.0 0.001 || note_bad "mtp decode_tps_warm=$got (want 20.0)"
verdict

run_case window_is_declared 0 \
	"each row says which engine surface its rate came from"
case "$(notes_of mtp)" in
*decode_window=engine_round_loop*) ;;
*) note_bad "mtp notes=$(notes_of mtp)" ;;
esac
case "$(notes_of normal)" in
*decode_window=engine_itl*) ;;
*) note_bad "normal notes=$(notes_of normal)" ;;
esac
verdict

# Three measured events at 190 / 200 / 210 have a median of 200 and a sample
# stddev of 10. A hard 0.0 in that column claims a spread never measured.
run_case stddev_is_measured 0 \
	"the mtp spread comes from the measured runs" \
	'STUB_DECODE_TPS_SEQ="Some(2.0)","Some(19.0)","Some(20.0)","Some(21.0)"'
got="$(metric_of mtp value)"
close_to "$got" 20.0 0.001 || note_bad "mtp decode_tps_warm=$got (want median 20.0)"
got="$(metric_of mtp stddev)"
close_to "$got" 1.0 0.001 || note_bad "mtp stddev=$got (want 1.0)"
verdict

# The first done event is the warmup and must not reach the ingested value. Its
# rate sits outside the measured three on purpose: a warmup that landed inside
# them could not move the median it is supposed to be kept out of.
run_case warmup_event_excluded 0 \
	"the warmup request's rate is not aggregated" \
	'STUB_DECODE_TPS_SEQ="Some(30.0)","Some(19.0)","Some(20.0)","Some(21.0)"'
got="$(metric_of mtp value)"
close_to "$got" 20.0 0.001 || note_bad "mtp decode_tps_warm=$got (want 20.0, not 20.5)"
verdict

# One done event where four requests ran. Taking the last three of one event
# leaves a single reading, whose sample stddev is the 0.0 that the rows this
# whole change is about are identified by — with n_measure=3 recorded beside it.
run_case truncated_log_refused 1 \
	"a log holding fewer events than requests served is refused" \
	'STUB_DONE_LINES=1' \
	'GREP:holds 1 round-loop' \
	'GREP:expected 4'
no_row mtp
verdict

# Three events where four requests ran: the count check is what catches this,
# because the three that survive are the warmup and the first two measured runs
# and nothing in the line distinguishes them.
run_case dropped_last_event_refused 1 \
	"a log missing the last request's event is refused" \
	'STUB_DONE_LINES=3' \
	'GREP:holds 3 round-loop' \
	'GREP:expected 4'
no_row mtp
verdict

# A binary older than the corrected field logs decode_tps as a bare number, and
# that number is the prefill-inclusive one — the defect wearing the corrected
# field's name.
run_case bare_number_refused 1 \
	"a bare-number decode_tps is refused, not read" \
	'STUB_DECODE_TPS_SEQ=20.0' \
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
	'GREP:decode rates for 3 measured runs'
no_row mtp
verdict

# The no-drafter arm has no round-loop record. Its rate is the one the server
# derived from that request's own inter-token gaps, which the stub reports at
# 1000/50 = 20 tok/s.
run_case normal_arm_uses_server_figure 0 \
	"the no-drafter row carries the server's rate" \
	'STUB_ITL_MEAN_MS=50' \
	'GREP:.normal. cross-check ok'
got="$(metric_of normal value)"
close_to "$got" 20.0 0.0001 || note_bad "normal decode_tps_warm=$got (want 20.0)"
verdict

# Eight tokens 50 ms apart behind a 1.5 s prefill: 20 tok/s over the
# decode window against 4.3 over the whole request. What pins the window's
# (n-1)/span convention is not this band but the cross-check above: over eight
# tokens, n/span is 1/7 = 14% out and the 10% band refuses it.
run_case normal_arm_excludes_prefill 0 \
	"the no-drafter arm times the decode window only" \
	'STUB_PREFILL_S=1.5' \
	'GREP:.normal. cross-check ok'
got="$(metric_of normal value)"
close_to "$got" 20.0 0.15 || note_bad "normal decode_tps_warm=$got (want ~20, not ~4.3)"
verdict

# Two readings of one window that disagree are a finding, not a choice. The stub
# reports half the gap it actually sent, so the engine claims twice the rate.
run_case cross_check_refuses_disagreement 1 \
	"engine and client readings that disagree stop the run" \
	'STUB_ITL_MEAN_MS=25' \
	'GREP:client-observed rate' \
	'GREP:past the 10% band'
no_row normal
verdict

# The record's kv_quant has to be the codec the run resolved. Passing no
# --kv-quant does not make it unknown: the engine resolves one and says so.
run_case kv_quant_is_the_resolved_codec 0 \
	"both rows carry the codec the run resolved" \
	'STUB_KV_QUANT=mixed_k8g64_v4g64'
[ "$(field_of_record mtp kv_quant)" = "mixed_k8g64_v4g64" ] ||
	note_bad "mtp kv_quant=$(field_of_record mtp kv_quant)"
[ "$(field_of_record normal kv_quant)" = "mixed_k8g64_v4g64" ] ||
	note_bad "normal kv_quant=$(field_of_record normal kv_quant)"
verdict

run_case unreported_kv_quant_refused 1 \
	"a run that never said which codec it used is not recorded" \
	'STUB_KV_QUANT_SUPPRESS=1' \
	'GREP:no .cache-type resolved. event'
no_row normal
no_row mtp
verdict

# Namespace, model and weight quant describe the checkpoint being served, and
# the caller chooses that. A constant here files every run under whatever the
# script was written against.
run_case snapshot_identity_is_read 0 \
	"the row describes the snapshot that was served"
for f in model_namespace:stub-ns model:stub-model weight_quant:mxfp8; do
	key="${f%%:*}"
	want="${f#*:}"
	[ "$(field_of_record mtp "$key")" = "$want" ] ||
		note_bad "mtp $key=$(field_of_record mtp "$key") (want $want)"
done
verdict

run_case unreadable_snapshot_refused 1 \
	"a snapshot whose identity cannot be read is not benched" \
	"VERIFIER_MODEL=$BARE_DIR" \
	'GREP:cannot read the identity'
no_row normal
no_row mtp
verdict

# ctx_max is what the server was started with, so the script passes it rather
# than recording the value it was written against.
run_case ctx_max_is_the_served_value 0 \
	"the row carries the context the server was given" \
	'ARGS:--max-ctx=4096'
[ "$(field_of_record mtp ctx_max)" = "4096" ] ||
	note_bad "mtp ctx_max=$(field_of_record mtp ctx_max)"
verdict

# A log the phase cannot attribute to its own server is not this run's log.
run_case unattributable_log_refused 1 \
	"a run log with no pid is not read as this server's" \
	'STUB_PID_SUPPRESS=1' \
	"GREP:no run log in .* is attributable"
no_row normal
no_row mtp
verdict

# Another rmlx process writing to the same directory supplies a candidate that
# sorts last. Selection is by pid, so it is not the one that gets read.
run_case decoy_log_ignored 0 \
	"a log another process wrote is not mistaken for this server's" \
	'STUB_DECOY_LOG=1'
got="$(metric_of mtp value)"
close_to "$got" 20.0 0.001 || note_bad "mtp decode_tps_warm=$got (want 20.0)"
verdict

# `KvQuant` renders `None` / `K8V8` / `Mixed { .. }` under Debug and
# `none` / `k8v8` / `mixed_...` under Display. Only the second is a name the
# flag accepts and the DB records, so a log written the other way is refused
# rather than filed under a cell nothing else reaches.
run_case debug_spelled_kv_quant_refused 1 \
	"a Debug-rendered codec name is refused" \
	'STUB_KV_QUANT=K8V8' \
	'GREP:Debug rendering'
no_row normal
no_row mtp
verdict

# The prompt length is what the server counted, not a constant in the script:
# the same script runs three different prompt files.
run_case prompt_tokens_is_measured 0 \
	"both rows carry the prompt length the server counted" \
	'STUB_PROMPT_TOKENS=1234'
[ "$(field_of_record mtp prompt_tokens)" = "1234" ] ||
	note_bad "mtp prompt_tokens=$(field_of_record mtp prompt_tokens)"
[ "$(field_of_record normal prompt_tokens)" = "1234" ] ||
	note_bad "normal prompt_tokens=$(field_of_record normal prompt_tokens)"
verdict

run_case missing_prompt_tokens_refused 1 \
	"a response with no usage.prompt_tokens is not recorded" \
	'STUB_PROMPT_TOKENS=-1' \
	'GREP:no usage.prompt_tokens'
no_row normal
no_row mtp
verdict

# The bests cell key partitions on this, so the speculative arm has to say it
# is one and the plain arm has to say it is not.
run_case decode_config_names_the_arm 0 \
	"the speculative row declares its arm and the plain one does not"
[ "$(field_of_record mtp decode_config)" = "mtp/block=5" ] ||
	note_bad "mtp decode_config=$(field_of_record mtp decode_config)"
[ "$(field_of_record normal decode_config)" = "null" ] ||
	note_bad "normal decode_config=$(field_of_record normal decode_config)"
verdict

# A round-loop record that reports no rate still counts as an event, so the
# totals line up while one measured run has no reading. Aggregating whatever is
# left would publish two runs' median under n_measure=3.
run_case partial_rates_refused 1 \
	"a measured run with no reading is not aggregated around" \
	'STUB_DECODE_TPS_SEQ="Some(20.0)","Some(20.0)","None","Some(20.0)"' \
	'GREP:reported 2 measurable'
no_row mtp
verdict

# A response the server could not time leaves the ring where it was, and the
# newest entry is then the previous request's. Reading it would file that
# request's rate under this one's name.
run_case stale_ring_entry_refused 1 \
	"a rate belonging to an earlier request is refused" \
	'STUB_ITL_SUPPRESS_AFTER=2' \
	'GREP:ITL ring went from'
no_row normal
verdict

# More content chunks than the completion has tokens: a chunk cannot carry less
# than a token, so the two counts are not describing the same stream.
run_case impossible_chunk_count_refused 1 \
	"more content chunks than tokens is refused" \
	'STUB_USAGE_TOKENS=4' \
	'GREP:not describing the same stream'
no_row normal
verdict

# The other direction is ordinary: a stop token is counted and carries no
# content, so a real completion has more tokens than content chunks. Refusing
# that would refuse every real run.
run_case uncounted_content_tokens_accepted 0 \
	"a completion with more tokens than content chunks is measured, not refused" \
	'STUB_USAGE_TOKENS=10'
got="$(metric_of normal value)"
close_to "$got" 20.0 0.15 || note_bad "normal decode_tps_warm=$got (want ~20)"
verdict

# A response too short for the server to time leaves the ITL ring untouched, so
# the newest entry belongs to an earlier request. Reading it would report that
# request's rate under this one's name.
run_case server_rate_unattributable_refused 1 \
	"a rate the server cannot attribute to this request is refused" \
	'STUB_ITL_SUPPRESS=1' \
	'GREP:the server attributed no decode rate'
no_row normal
verdict

# ── The round-loop figures ────────────────────────────────────────────────────

# metric_value <config> <metric name> — that metric's value, or "" when the row
# carries no such metric.
metric_value() {
	local path
	path="$(record_of "$1")"
	[ -z "$path" ] && return 0
	python3 -c 'import json, sys
rec = json.load(open(sys.argv[1]))
for m in rec.get("metrics", []):
    if m.get("name") == sys.argv[2]:
        print(m.get("value", ""))
        break' "$path" "$2"
}

# The whole point of the change: a speculative row carries what the round loop
# counted, not only its accept rate. 30 rounds emitting 128 tokens of which one
# is the pre-round seed is 127/30 = 4.2333 tokens per round, and
# 1800 - 300 - 900 ms of loop over 30 rounds is 20 ms.
run_case round_loop_figures_recorded 0 \
	"the speculative row carries the per-round split, not only the accept rate"
for pair in \
	"tokens_per_round 4.233333" \
	"accepted_per_step 3.266667" \
	"draft_ms_per_round 10.0" \
	"verify_ms_per_round 30.0" \
	"loop_ms_per_round 20.0"; do
	set -- $pair
	got="$(metric_value mtp "$1")"
	close_to "$got" "$2" 0.001 || note_bad "mtp $1=$got (want $2)"
done
# The no-drafter arm has no round loop, so it must carry no per-round figure —
# a zero there would rank as a measured one.
for name in tokens_per_round accepted_per_step loop_ms_per_round; do
	[ -z "$(metric_value normal "$name")" ] ||
		note_bad "normal carries $name=$(metric_value normal "$name")"
done
verdict

# The engine derives the per-round figures too, and this reader derives them
# again. A drift between the two would file a number no run produced, so an
# event whose own counters contradict its derived field is refused — for every
# one of them, not for the one that happened to be tested. `loop_ms_per_round`
# is the term list that can go wrong quietly: it is the only difference of
# three counters.
for derived in accept_rate accepted_per_step tokens_per_round \
	draft_ms_per_round verify_ms_per_round loop_ms_per_round; do
	run_case "derived_${derived}_contradiction_refused" 1 \
		"a done line whose ${derived} disagrees with its counters is refused" \
		"STUB_DERIVED_OVERRIDE=${derived}=9.0" \
		'GREP:do not agree on the formula'
	no_row mtp
	verdict
done

# A counter a figure is derived from, dropped together with that figure, would
# otherwise aggregate to a zero nobody measured — and the caller cannot see it,
# because this reader always prints the key.
run_case dropped_counter_refused 1 \
	"a counter a figure is derived from is required, not defaulted to zero" \
	'STUB_DROP_FIELDS=emitted,tokens_per_round' \
	'GREP:carries no emitted'
no_row mtp
verdict

run_case dropped_round_span_refused 1 \
	"the round-loop span is required too, so its residual cannot read zero" \
	'STUB_DROP_FIELDS=round_ms,loop_ms_per_round' \
	'GREP:carries no round_ms'
no_row mtp
verdict

# The seed count and the emitted count are read at different points in the loop,
# so a loop that stopped emitting its pre-round token disagrees with itself here
# rather than shifting tokens_per_round by 1/rounds in an append-only table.
# The drift itself, against the stub's own healthy counters: the seed captured
# before the pre-round emit_step, so it reports 0 where the loop emitted 1. The
# stub's line sits one token under the emission budget — where the inequality
# this replaced was blind and where three of the four reachable loops live — so
# only the three counts adding up can see it.
run_case seed_taken_before_the_pre_round_emission_refused 1 \
	"the drift is refused on a request that does not saturate the round budget" \
	'STUB_SEED_EMITTED=0' \
	'GREP:accounts for'
no_row mtp
verdict

# The same inconsistency from the other side: a round loop that counted more
# than it emitted.
run_case round_count_contradicting_emitted_refused 1 \
	"a round count that disagrees with the emitted total is refused" \
	'STUB_EMITTED_IN_ROUNDS=120' \
	'GREP:accounts for'
no_row mtp
verdict

# And the emission budget, which is a different invariant on the same counters.
run_case round_count_over_the_emission_budget_refused 1 \
	"more tokens credited to the rounds than they could have produced is refused" \
	'STUB_EMITTED_IN_ROUNDS=200' 'STUB_EMITTED=201' \
	'GREP:could have produced'
no_row mtp
verdict

# The cell a row belongs to is the one the round loop named. A log that does not
# name it leaves the script to guess from its own flags, which is how a row is
# filed under a configuration the run did not use.
run_case unnamed_cell_refused 1 \
	"a log that does not name its cell is refused rather than guessed at" \
	'STUB_DROP_FIELDS=decode_config' \
	'GREP:carries no decode_config field'
no_row mtp
verdict

# The script asked for block 9; the engine ran block 3 and said so. The row must
# be the engine's cell, or a sidecar that caps the block silently files every
# request under a block it never ran.
run_case engines_cell_beats_the_flag 0 \
	"the recorded cell is the one the engine named, not the one asked for" \
	'STUB_DECODE_CONFIG=mtp/block=3' \
	ARGS:--draft-block-size ARGS:9
[ "$(field_of_record mtp decode_config)" = "mtp/block=3" ] ||
	note_bad "decode_config=$(field_of_record mtp decode_config)"
case "$(notes_of mtp)" in
*"block_size=3"*) ;;
*) note_bad "notes claim a block the engine did not run: $(notes_of mtp)" ;;
esac
verdict

# A drafter that resizes its block names a cell of its own. Recording it as the
# fixed arm at the same ceiling would rank two configurations as one.
run_case adaptive_cell_passes_through 0 \
	"an adaptive drafter's cell reaches the row intact" \
	'STUB_DECODE_CONFIG=dflash/block=16,dflash/depth=accept_rate' \
	ARGS:--draft-kind ARGS:dflash
[ "$(field_of_record dflash decode_config)" = "dflash/block=16,dflash/depth=accept_rate" ] ||
	note_bad "decode_config=$(field_of_record dflash decode_config)"
verdict

# The kind becomes a component of the buffer filename and of `notes` before the
# engine sees it, so a value the engine would reject must not get that far.
run_case hostile_draft_kind_refused 1 \
	"a drafter kind that would escape the buffer directory is refused at parse" \
	ARGS:--draft-kind ARGS:../../etc/mtp \
	'GREP:is not a bare lower-case name'
no_row mtp
[ -z "$(ls "$CASE_HOME"/metrics/buffer/pending/* 2>/dev/null)" ] ||
	note_bad "a buffer file was written for a refused drafter kind"
verdict

run_case unusable_block_size_refused 1 \
	"a block size with no room for a draft token is refused" \
	ARGS:--draft-block-size ARGS:1 \
	'GREP:must be an integer >= 2'
verdict

echo
echo "passed=$PASSED failed=$FAILED"
[ "$FAILED" -eq 0 ] || exit 1
