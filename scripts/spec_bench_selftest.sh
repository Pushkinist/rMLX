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
# `eval spec-answer` is the exception that proves the oracle is the real one: the
# stub binary hands it to the dev build of `rmlx`, which this suite builds first.
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
# A checkout whose HEAD is a commit the identity under test does not name.
git -C "$FAKE_ROOT" init -q
git -C "$FAKE_ROOT" -c user.name=selftest -c user.email=selftest@example.invalid \
	commit -q --allow-empty -m stub
HEAD_SHA="$(git -C "$FAKE_ROOT" rev-parse --short HEAD)"

# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"

# ── Shims ─────────────────────────────────────────────────────────────────────
#
# The script's inter-request sleeps are 5 s of nothing when the server is a stub.
# Each one is noted in $SLEEP_LOG, when a case sets it, so a case can read which
# pauses the script asked for.
SHIM_DIR="$WORK/shims"
mkdir -p "$SHIM_DIR"
printf '#!/bin/sh\necho "$1" >>"${SLEEP_LOG:-/dev/null}"\nexit 0\n' >"$SHIM_DIR/sleep"
chmod +x "$SHIM_DIR/sleep"

# ── Stub server ───────────────────────────────────────────────────────────────

SERVER_PY="$WORK/stub_server.py"
cat >"$SERVER_PY" <<'PYEOF'
"""Canned OpenAI-compatible server for the spec_bench selftest.

Streams STUB_TOKENS content chunks STUB_GAP_S apart, starting STUB_PREFILL_S
after the role chunk, then a usage block. It does not wait between chunks: each
content chunk is preceded by `: sent-at=<seconds>`, the time the chunk is due on
that schedule, and the client reads the stamp instead of its own clock, so no
scheduling delay can move the window. It publishes the aggregate of the same
schedule at /metrics/cache in the shape the real server's ITL ring has, so the
engine-side figure the bench reads and the client window it cross-checks against
describe the same gaps. STUB_ITL_MEAN_MS overrides that aggregate, which is how
a disagreement is staged;
STUB_ITL_SUPPRESS leaves the ring untouched, as the real server does for a
response too short to have an interval.

When the serve argv carried a drafter it also appends a round-loop `done` line
per request to the run log — written before the final flush, so a reader that
sees `[DONE]` sees the line too. `decode_tps` comes from the next entry of
STUB_DECODE_TPS_SEQ, a comma-separated list of raw JSON values with the last one
repeating, empty for a line carrying no such field. STUB_DONE_LINES caps how
many requests get a line at all.

What it answers is canned too. STUB_SPEC_DIVERGE_FROM makes the speculative
server change the completion from that token on, standing in for a round loop
that changed the answer; STUB_VARY_PER_REQUEST makes every request of both
servers answer differently. STUB_SAMPLER appends the engine's per-request
`host categorical sampler active` event, and STUB_SAMPLER_AFTER stops appending
it partway through, which is the run whose requests did not share one sampling
setup.

Every request of a server must carry the messages, max_tokens, temperature and
seed of the first one, or it is answered 400: a request that is not the measured
one is not a pass over the measured answer. A request with `"stream": false` gets
the whole answer back with one logprob record per token when it asks for
`logprobs`, listing `min(top_logprobs, STUB_TOP_LOGPROBS)` candidates, the pieces being the ones the streamed request sends, and the
same per-request events the streamed one leaves. It is not counted in `served`:
the knobs indexed by that count keep addressing the streamed requests. Its
margins are 1.0 apart everywhere but where STUB_MARGIN_AT (`position=gap,...`)
says otherwise; the speculative server reads STUB_SPEC_MARGIN_AT instead.
STUB_TOP_LOGPROBS is how many candidates each record lists (2),
STUB_RECORDS_SHORT drops that many records from the end, STUB_NO_LOGPROBS
leaves the field out, STUB_UNTIMED_DIFFERS (`plain` or `spec`) makes that
server's unstreamed answer a different one from its streamed answer, and
STUB_SAME_PIECES makes the records name the pieces of the undiverged answer
whatever the text says, and STUB_UNSTREAMED_STATUS is the HTTP status it answers
with and STUB_UNSTREAMED_BODY=junk makes it answer 200 with something that is not a
chat completion. STUB_SPEC_EXTRA_TOKENS gives the speculative server that many more tokens
than STUB_TOKENS (fewer when negative), and STUB_SPEC_VARY makes only its streamed
answers differ from request to request.

Arguments after the port. The stub binary reads them from the file `stub_args` in
the case's RMLX_HOME, so a case sets them and no environment variable carries
them:

  --marks DIR          before each request, note in DIR/seen whether the
                       client that sent it had already left DIR/mark.N (N counts
                       the requests over every server of the case)
  --hold PREFIX        after its role chunk, the first streamed request creates
                       PREFIX.held and waits for PREFIX.release, so a case can
                       act while a request is in flight
  --hold-wait SECONDS  how long that wait lasts at most (300)
  --requests FILE      append `<arm> stream` or `<arm> unstreamed` to FILE for each
                       request, `arm` being `spec` on the speculative server
"""

import argparse
import json
import os
import statistics
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("port", type=int)
parser.add_argument("--marks", default="")
parser.add_argument("--hold", default="")
parser.add_argument("--hold-wait", type=float, default=300.0)
parser.add_argument("--requests", default="")
ARGS = parser.parse_args()

SPECULATIVE = os.environ.get("STUB_SPECULATIVE", "") == "1"
TOKENS = int(os.environ["STUB_TOKENS"]) + (
    int(os.environ.get("STUB_SPEC_EXTRA_TOKENS", "0")) if SPECULATIVE else 0
)
GAP_S = float(os.environ["STUB_GAP_S"])
PREFILL_S = float(os.environ["STUB_PREFILL_S"])
EMITTED = int(os.environ["STUB_EMITTED"])
ELAPSED_MS = float(os.environ["STUB_ELAPSED_MS"])
LOG_PATH = os.environ.get("STUB_LOG", "")
SEQ = [s for s in os.environ.get("STUB_DECODE_TPS_SEQ", "").split(",") if s]
DONE_LINES = int(os.environ.get("STUB_DONE_LINES", "-1"))
ITL_MEAN_OVERRIDE = os.environ.get("STUB_ITL_MEAN_MS", "")
ITL_SUPPRESS = os.environ.get("STUB_ITL_SUPPRESS", "") == "1"
ITL_SUPPRESS_AFTER = int(os.environ.get("STUB_ITL_SUPPRESS_AFTER", "-1"))
USAGE_TOKENS = int(os.environ.get("STUB_USAGE_TOKENS", "-1"))
PROMPT_TOKENS = int(os.environ.get("STUB_PROMPT_TOKENS", "-1"))
BOUND_FLAG = os.environ.get("STUB_BOUND_FLAG", "")
DIVERGE_FROM = int(os.environ.get("STUB_SPEC_DIVERGE_FROM", "-1"))
VARY_PER_REQUEST = os.environ.get("STUB_VARY_PER_REQUEST", "") == "1"
SAMPLER = os.environ.get("STUB_SAMPLER", "") == "1"
SAMPLER_AFTER = int(os.environ.get("STUB_SAMPLER_AFTER", "-1"))


def margin_table(name):
    """`position=gap` pairs of an environment variable as a dict."""
    pairs = [p.split("=") for p in os.environ.get(name, "").split(",") if p]
    return {int(position): float(gap) for position, gap in pairs}


MARGIN_AT = margin_table("STUB_SPEC_MARGIN_AT" if SPECULATIVE else "STUB_MARGIN_AT")
TOP_LOGPROBS = int(os.environ.get("STUB_TOP_LOGPROBS", "2"))
RECORDS_SHORT = int(os.environ.get("STUB_RECORDS_SHORT", "0"))
NO_LOGPROBS = os.environ.get("STUB_NO_LOGPROBS", "") == "1"
UNTIMED_DIFFERS = os.environ.get("STUB_UNTIMED_DIFFERS", "")
SAME_PIECES = os.environ.get("STUB_SAME_PIECES", "") == "1"
UNSTREAMED_STATUS = int(os.environ.get("STUB_UNSTREAMED_STATUS", "200"))
UNSTREAMED_BODY = os.environ.get("STUB_UNSTREAMED_BODY", "")
CONTRACT = {}
SPEC_VARY = os.environ.get("STUB_SPEC_VARY", "") == "1"

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
# Whether the engine charged its phases. A JSON boolean on the real done line,
# so the stub emits one too — the reader refuses a string, which is the point.
CHARGED = json.loads(os.environ.get("STUB_CHARGED", "false"))
DERIVED_OVERRIDE = os.environ.get("STUB_DERIVED_OVERRIDE", "")
DROP_FIELDS = [f for f in os.environ.get("STUB_DROP_FIELDS", "").split(",") if f]


def done_line(raw=None):
    """One round-loop `done` record, as tracing's JSON layer renders it.

    Carries the raw counters and the per-round figures the engine derives from
    them, consistent with each other. STUB_DERIVED_OVERRIDE (`name=value`)
    breaks one of them, which is how the reader's cross-check is staged. `raw`
    is the line's `decode_tps` as a JSON value; left out, the next entry of
    STUB_DECODE_TPS_SEQ.
    """
    if raw is None:
        raw = SEQ[min(served, len(SEQ) - 1)] if SEQ else ""
    # The engine counts this at its own emit site, independently of the seed —
    # which is the whole point: a drifting seed must break the sum rather than
    # be absorbed by it. So the default is the healthy round count, not
    # `EMITTED - SEED_EMITTED`.
    round_emitted = int(os.environ.get("STUB_EMITTED_IN_ROUNDS", EMITTED - 1))
    fields = {
        "message": "mtp_generate: done",
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
        "round_ms": ROUND_MS,
        "draft_ms": DRAFT_MS,
        "verifier_ms": VERIFY_MS,
        "draft_ms_per_round": DRAFT_MS / ROUNDS,
        "verify_ms_per_round": VERIFY_MS / ROUNDS,
        "loop_ms_per_round": (ROUND_MS - DRAFT_MS - VERIFY_MS) / ROUNDS,
        "block_size": BLOCK_SIZE,
        "decode_config": DECODE_CONFIG,
        "charged": CHARGED,
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


def token_text(index, diverged=True):
    """The index-th content piece this request streams.

    Every piece is the same length whatever it says, so a diverging arm changes
    the answer without changing the token count or the window it was timed over.
    """
    stem = "x" if diverged and SPECULATIVE and 0 <= DIVERGE_FROM <= index else "t"
    request = f"r{served}-" if VARY_PER_REQUEST or (SPEC_VARY and SPECULATIVE) else ""
    return f"{request}{stem}{index} "


def sampler_line():
    """The engine's per-request sampler-resolution event."""
    return json.dumps(
        {
            "timestamp": "2026-09-03T00:00:00Z",
            "level": "INFO",
            "fields": {
                "message": "generate: host categorical sampler active",
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "seed": 42,
            },
        }
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


def write_request_events(raw_rate=None):
    """The events the engine leaves for one served request: its sampler line
    and, on the speculative server, its round-loop `done` line."""
    if SAMPLER and LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(sampler_line() + "\n")
    if SPECULATIVE and LOG_PATH:
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(done_line(raw_rate) + "\n")


def note_client_mark():
    """Note in the marks directory whether the client that sent this request
    was already running."""
    seen = os.path.join(ARGS.marks, "seen")
    number = 1
    if os.path.exists(seen):
        with open(seen, encoding="utf-8") as handle:
            number += len(handle.readlines())
    up = os.path.exists(os.path.join(ARGS.marks, f"mark.{number}"))
    with open(seen, "a", encoding="utf-8") as handle:
        handle.write("up\n" if up else "down\n")


def same_request_as_before(request):
    """Whether this request asks for the answer the first one asked for."""
    fields = {k: request.get(k) for k in ("messages", "max_tokens", "temperature", "seed")}
    if not CONTRACT:
        CONTRACT.update(fields)
    return fields == CONTRACT


def completion_body(request):
    """The whole answer of one unstreamed request, with its logprob records."""
    pieces = [token_text(i) for i in range(TOKENS)]
    if UNTIMED_DIFFERS == ("spec" if SPECULATIVE else "plain"):
        pieces = [f"u{piece}" for piece in pieces]
    named = [token_text(i, diverged=not SAME_PIECES) for i in range(TOKENS)]
    records = []
    listed = min(request.get("top_logprobs") or 0, TOP_LOGPROBS)
    for i, piece in enumerate(named[: TOKENS - RECORDS_SHORT]):
        candidates = [
            {"token": piece, "logprob": -0.5},
            {"token": "other", "logprob": -0.5 - MARGIN_AT.get(i, 1.0)},
        ]
        records.append({"token": piece, "logprob": -0.5, "top_logprobs": candidates[:listed]})
    choice = {
        "index": 0,
        "message": {"role": "assistant", "content": "".join(pieces)},
        "finish_reason": "length",
    }
    if request.get("logprobs") and not NO_LOGPROBS:
        choice["logprobs"] = {"content": records}
    usage = {"completion_tokens": TOKENS}
    if PROMPT_TOKENS >= 0:
        usage["prompt_tokens"] = PROMPT_TOKENS
    return {"object": "chat.completion", "choices": [choice], "usage": usage}


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
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        if ARGS.marks:
            note_client_mark()
        if not same_request_as_before(request):
            self.send_error(400)
            return
        if ARGS.requests:
            kind = "stream" if request.get("stream", True) else "unstreamed"
            with open(ARGS.requests, "a", encoding="utf-8") as handle:
                handle.write(f"{'spec' if SPECULATIVE else 'plain'} {kind}\n")
        if not request.get("stream", True):
            if UNSTREAMED_STATUS != 200:
                self.send_error(UNSTREAMED_STATUS)
                return
            write_request_events('"Some(20.0)"')
            self._json({"detail": "not a completion"} if UNSTREAMED_BODY == "junk" else completion_body(request))
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(payload, sent_at=None):
            if sent_at is not None:
                self.wfile.write(f": sent-at={sent_at}\n".encode())
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        chunk({"choices": [{"delta": {"role": "assistant"}, "index": 0}]}, 0.0)
        if ARGS.hold:
            open(ARGS.hold + ".held", "w").close()
            # A poll for the file the case creates; the limit ends a stub whose
            # case has died.
            deadline = time.monotonic() + ARGS.hold_wait
            while not os.path.exists(ARGS.hold + ".release") and time.monotonic() < deadline:
                time.sleep(0.05)
        sends = []
        for i in range(TOKENS):
            due = PREFILL_S + i * GAP_S
            chunk({"choices": [{"delta": {"content": token_text(i)}, "index": 0}]}, due)
            sends.append(due)
        reported = USAGE_TOKENS if USAGE_TOKENS >= 0 else TOKENS
        usage = {"completion_tokens": reported}
        if PROMPT_TOKENS >= 0:
            usage["prompt_tokens"] = PROMPT_TOKENS
        chunk({"choices": [], "usage": usage})

        push_itl(sends)
        if SAMPLER and LOG_PATH and (SAMPLER_AFTER < 0 or served < SAMPLER_AFTER):
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(sampler_line() + "\n")
        if SPECULATIVE and LOG_PATH and (DONE_LINES < 0 or served < DONE_LINES):
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(done_line() + "\n")
        served += 1

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


server = HTTPServer(("127.0.0.1", ARGS.port), Handler)
if BOUND_FLAG:
    with open(BOUND_FLAG, "a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
server.serve_forever()
PYEOF

# stop_stubs <bound-file> — stop every stub server a case started, by the PID
# each wrote when it bound. The harness stops its own server; this keeps a
# harness defect from leaving a stub on the next case's port.
stop_stubs() {
	[ -f "$1" ] || return 0
	local pid
	while read -r pid; do
		kill "$pid" 2>/dev/null || true
	done <"$1"
}

# ── The real judge ────────────────────────────────────────────────────────────
#
# `eval spec-answer` is the one subcommand the stub binary hands to the real dev
# binary, so the cases exercise the oracle itself on the wire shape the stub
# serves. In `make ci` the build is already done by `test`.
DEV_RMLX="$REPO_ROOT/target/debug/rmlx"
if ! (cd "$REPO_ROOT" && cargo build -q -p rmlx-cli --bin rmlx) || [ ! -x "$DEV_RMLX" ]; then
	echo "spec_bench_selftest: cannot build the dev rmlx at $DEV_RMLX" >&2
	exit 2
fi

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
eval)
	if [ "\$2" = spec-answer ]; then
		printf '%s\n' "\$@" >"\$RMLX_HOME/spec_answer_argv"
		prev=""
		for arg in "\$@"; do
			case "\$prev" in
			--plain) /bin/cp -f "\$arg" "\$RMLX_HOME/spec_answer_plain.json" ;;
			--spec) /bin/cp -f "\$arg" "\$RMLX_HOME/spec_answer_spec.json" ;;
			esac
			prev="\$arg"
		done
		case "\${STUB_JUDGE:-}" in
		silent)
			echo "error: unrecognized subcommand 'spec-answer'" >&2
			exit 2
			;;
		status1)
			"$DEV_RMLX" "\$@" || true
			exit 1
			;;
		esac
		exec "$DEV_RMLX" "\$@"
	fi
	;;
serve)
	if [ -n "\${STUB_ARGV:-}" ]; then
		printf '%s\n' "\$@" >"\$STUB_ARGV"
	fi
	port=8090
	speculative=0
	while [ \$# -gt 0 ]; do
		case "\$1" in
		--port) port="\$2" ;;
		--draft-model) speculative=1 ;;
		esac
		shift
	done
	if [ -n "\${STUB_SHARED_LOG:-}" ]; then
		log="\$RMLX_HOME/logs/\$STUB_SHARED_LOG"
		if [ -n "\${STUB_FOREIGN_START:-}" ]; then
			printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"rmlx start\",\"version\":\"9.9.9\",\"run_id\":\"stub\",\"pid\":777}}" >>"\$log"
		fi
	else
		log="\$RMLX_HOME/logs/\$(date +%s)-\$\$.jsonl"
		: >"\$log"
	fi
	if [ -n "\${STUB_REWRITE_PID_IN:-}" ]; then
		stale="\$RMLX_HOME/logs/\$STUB_REWRITE_PID_IN"
		sed "s/PIDPLACEHOLDER/\$(printf '%14d' \$\$)/" "\$stale" >"\$stale.tmp"
		cat "\$stale.tmp" >"\$stale"
		rm -f "\$stale.tmp"
	fi
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
	# The case's own arguments for the stub server, one line, if it left any.
	exec python3 "$SERVER_PY" "\$port" \$(cat "\$RMLX_HOME/stub_args" 2>/dev/null)
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
#          [CLIENT_SHIM:N] [NO_STAMPS:1] [READER_CLOCK:<step>]
#          [STUB:<stub server arguments>] [PRELOG:<run log name>]
#          [STALE:<run log name>]
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

	local env_pairs=() greps=() extra_args=() a delayed=0 stamps=true stub_args="" step="" prelog="" stale=""
	for a in "$@"; do
		case "$a" in
		GREP:*) greps+=("${a#GREP:}") ;;
		ARGS:*) extra_args+=("${a#ARGS:}") ;;
		CLIENT_SHIM:*) delayed="${a#CLIENT_SHIM:}" ;;
		NO_STAMPS:*) stamps=false ;;
		READER_CLOCK:*) step="${a#READER_CLOCK:}" ;;
		STUB:*) stub_args="$stub_args ${a#STUB:}" ;;
		PRELOG:*) prelog="${a#PRELOG:}" ;;
		STALE:*) stale="${a#STALE:}" ;;
		*) env_pairs+=("$a") ;;
		esac
	done
	$stamps && extra_args+=(--server-stamps)
	write_client_shim "$CASE_HOME/client" "$delayed" "$step"
	# A run log an earlier rmlx process left in this second: the servers of the
	# case append to it instead of opening their own. It carries that process's
	# own events, none of which belong to the run being measured.
	if [ -n "$prelog" ]; then
		cat >"$CASE_HOME/logs/$prelog" <<'LOGEOF'
{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"rmlx start","version":"9.9.9","run_id":"stub","pid":4242}}
{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"cache-type resolved","arch":"Stub","kv_quant":"none"}}
{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"generate: host categorical sampler active","temperature":0.7,"top_p":0.95,"top_k":20,"min_p":0.0,"seed":42}}
{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"mtp_generate: done","rounds":30,"emitted":128,"decode_tps":"Some(99.0)"}}
LOGEOF
		env_pairs+=("STUB_SHARED_LOG=$prelog")
	fi
	# A run log whose `rmlx start` is the server's own pid, written before the
	# server exists: the stub fills the pid in without changing the file's size.
	if [ -n "$stale" ]; then
		printf '%s\n' '{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"rmlx start","version":"9.9.9","run_id":"stub","pid":PIDPLACEHOLDER}}' \
			>"$CASE_HOME/logs/$stale"
		env_pairs+=("STUB_REWRITE_PID_IN=$stale")
	fi
	printf '%s\n' "--marks $CASE_HOME/client --requests $CASE_HOME/requests$stub_args" >"$CASE_HOME/stub_args"

	CASE_OUT="$WORK/$CASE_NAME.log"
	local got=0
	set +e
	env -i \
		PATH="$CASE_HOME/client/bin:$SHIM_DIR:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
		HOME="$WORK" \
		RMLX_HOME="$CASE_HOME" \
		SLEEP_LOG="$CASE_HOME/sleeps" \
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
		STUB_ARGV="$CASE_HOME/serve_argv" \
		${env_pairs[@]+"${env_pairs[@]}"} \
		bash "$FAKE_ROOT/scripts/spec_bench.sh" --port "$PORT" \
		${extra_args[@]+"${extra_args[@]}"} >"$CASE_OUT" 2>&1
	got=$?
	set -e

	CASE_BAD=""
	[ "$want" != "-" ] && [ "$got" -ne "$want" ] && CASE_BAD="exit=$got (want $want)"
	# However the run ended, it stopped the server it started: a server left
	# alive holds the Metal claim after the harness is gone.
	note_survivors "$CASE_HOME/stub_bound"
	stop_stubs "$CASE_HOME/stub_bound"
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

# note_survivors <bound-file> — a failure reason per stub server still alive.
note_survivors() {
	[ -f "$1" ] || return 0
	local pid
	while read -r pid; do
		kill -0 "$pid" 2>/dev/null && note_bad "server $pid outlived the harness"
	done <"$1"
	return 0
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
	local f
	for f in "$CASE_HOME"/tmp/stamped_records/*-"$1".json "$CASE_HOME"/metrics/buffer/pending/*-"$1".json; do
		if [ -e "$f" ]; then
			echo "$f"
			return 0
		fi
	done
	echo ""
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

# no_judge — `eval spec-answer` was never run.
no_judge() {
	[ -e "$CASE_HOME/spec_answer_argv" ] && note_bad "the judge ran on a run that needed none"
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

# Two done events where five requests ran: the warmup's and the logprob pass's,
# which the stub always writes. Taking the last three of two events leaves two
# readings, and a single reading's sample stddev is the 0.0 that the rows this
# whole change is about are identified by — with n_measure=3 recorded beside it.
run_case truncated_log_refused 1 \
	"a log holding fewer events than requests served is refused" \
	'STUB_DONE_LINES=1' \
	'GREP:holds 2 round-loop' \
	'GREP:expected 5'
no_row mtp
verdict

# Four events where five requests ran: the count check is what catches this,
# because the four that survive are the warmup, the logprob pass and the first two
# measured runs and nothing in the line distinguishes them.
run_case dropped_last_event_refused 1 \
	"a log missing the last request's event is refused" \
	'STUB_DONE_LINES=3' \
	'GREP:holds 4 round-loop' \
	'GREP:expected 5'
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

# The client sends the request itself. A client fed by a separate `curl` could
# start after the stream had begun and stamp the chunks it had missed all at its
# own start, so the window shrank and the rate read high. The client shim here
# starts the third reader a second late, the first measured run, and every request
# of both servers must still find its own client already running.
run_case every_request_comes_from_a_running_client 0 \
	"a client that starts late has not missed any of the stream" \
	CLIENT_SHIM:3 \
	'GREP:.normal. cross-check ok'
[ "$(wc -l <"$CASE_HOME/client/seen" 2>/dev/null | tr -d ' ')" = "10" ] ||
	note_bad "the stub saw $(wc -l <"$CASE_HOME/client/seen" 2>/dev/null | tr -d ' ') requests (want 10, five on each of two servers)"
[ "$(sort -u "$CASE_HOME/client/seen" 2>/dev/null)" = "up" ] ||
	note_bad "a request reached a server before its client existed: $(sort -u "$CASE_HOME/client/seen" 2>/dev/null | tr '\n' ' ')"
[ "$(grep -c -- '--server-stamps' "$CASE_HOME/client/reader_argv" 2>/dev/null)" = "8" ] ||
	note_bad "not every streamed request's client was asked to read the stub's stamps"
verdict

# The harness asks the client for the stub's stamps only when it is asked to.
# Without the argument the stamps are comments, the client times the chunks, and
# the run is whatever this host makes of an instant stub, so the exit code is not
# asserted. What is asserted is what the client was told.
run_case stamps_are_not_used_unless_asked - \
	"the harness does not pass the stamps on by itself" \
	NO_STAMPS:1
[ -s "$CASE_HOME/client/reader_argv" ] || note_bad "no client ran"
if grep -q -- '--server-stamps' "$CASE_HOME/client/reader_argv" 2>/dev/null; then
	note_bad "a client was asked for stamps although the harness was not"
fi
verdict

# A row made from the stub's stamps is not a measurement. The run keeps its rows
# out of the queue `rmlx metrics record` reads and never calls it.
run_case stamped_rows_are_not_filed 0 \
	"a run timed by a stub's stamps files no row" \
	'GREP:not filed'
[ -n "$(record_of normal)" ] || note_bad "the run wrote no row to look at"
if [ -n "$(ls -A "$CASE_HOME/metrics/buffer/pending" 2>/dev/null)" ]; then
	note_bad "a stamped row is in the ingest queue: $(ls "$CASE_HOME/metrics/buffer/pending" | tr '\n' ' ')"
fi
if grep -q 'stub: recorded' "$CASE_OUT"; then
	note_bad "a stamped run called the recorder"
fi
verdict

# The other direction: a run on the client clock files its rows. The stub still
# sends its stamp comments, which the client reads as lines, so each chunk is
# three lines. The client clock here moves a third of the stub's 50 ms gap with
# each line, so the client rate is exactly the engine's 20 tok/s and the
# cross-check cannot refuse the run. The verdict is where the rows go, not a
# rate, and no host clock is read.
run_case unstamped_rows_go_to_the_queue_and_the_recorder 0 \
	"a run timed by the client clock files its rows through the recorder" \
	NO_STAMPS:1 READER_CLOCK:0.0166666667 \
	'GREP:.normal. cross-check ok'
for config in normal mtp; do
	case "$(record_of "$config")" in
	"$CASE_HOME"/metrics/buffer/pending/*) ;;
	*) note_bad "the $config row is not in the ingest queue: $(record_of "$config")" ;;
	esac
done
[ "$(grep -c 'stub: recorded' "$CASE_OUT")" = "2" ] ||
	note_bad "the recorder was called $(grep -c 'stub: recorded' "$CASE_OUT") times (want once for each arm)"
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

# The run id has a granularity of one second, so a server that starts in the
# second an earlier rmlx process started in appends to that process's file. The
# file existed before the snapshot; the pid in its `rmlx start` events is what
# attributes it.
run_case same_second_log_attributed 0 \
	"a server appending to a log that predates the snapshot is attributed by pid" \
	'PRELOG:20260903-000000-9.9.9.jsonl'
got="$(metric_of mtp value)"
close_to "$got" 20.0 0.001 || note_bad "mtp decode_tps_warm=$got (want 20.0)"
got="$(metric_of normal value)"
close_to "$got" 20.0 0.001 || note_bad "normal decode_tps_warm=$got (want 20.0)"
verdict

# The binary reports no commit; a caller that built it elsewhere than the
# checkout's HEAD says which one in the identity, and the row carries that one.
IDENTITY_SHA=feedbee
run_case identity_sha_beats_checkout_head 0 \
	"a git_sha the identity carries is the one filed, not the checkout's HEAD" \
	'RMLX_IDENTITY_JSON={"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb","git_sha":"'$IDENTITY_SHA'"}'
for arm in normal mtp; do
	[ "$(field_of_record "$arm" git_sha)" = "$IDENTITY_SHA" ] ||
		note_bad "$arm git_sha=$(field_of_record "$arm" git_sha) (want $IDENTITY_SHA)"
	case "$(field_of_record "$arm" description)" in
	*"sha=$IDENTITY_SHA") ;;
	*) note_bad "$arm description=$(field_of_record "$arm" description) (want sha=$IDENTITY_SHA)" ;;
	esac
done
verdict

run_case checkout_head_is_the_fallback_sha 0 \
	"an identity with no git_sha leaves the checkout's HEAD as the filed commit"
for arm in normal mtp; do
	[ "$(field_of_record "$arm" git_sha)" = "$HEAD_SHA" ] ||
		note_bad "$arm git_sha=$(field_of_record "$arm" git_sha) (want $HEAD_SHA)"
done
verdict

# A log that existed before the snapshot and did not grow is nobody's phase,
# even when the pid in it is the server's: it is the earlier process's file.
run_case unchanged_log_is_not_a_candidate 1 \
	"a pre-existing log that did not grow is not read as this server's" \
	'STUB_PID_SUPPRESS=1' \
	'STALE:20260903-000001-9.9.9.jsonl' \
	"GREP:no run log in .* is attributable"
no_row normal
no_row mtp
verdict

# Two processes started in one second and both wrote to the phase: their
# lines cannot be told apart, so the phase is refused rather than guessed at.
run_case shared_phase_log_refused 1 \
	"a grown log whose phase carries another pid's start is refused" \
	'PRELOG:20260903-000002-9.9.9.jsonl' \
	'STUB_FOREIGN_START=1' \
	"GREP:cannot be told apart"
no_row normal
no_row mtp
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

# The block a run is measured at is the engine's unless the caller named one, and
# the only place that is visible is the argv the server was launched with. The
# recorded `decode_config` cannot see it: it comes from the stub's own
# STUB_DECODE_CONFIG, so a script that hard-coded a block and always passed it
# would file rows reading `mtp/block=5` either way.
run_case draft_block_left_to_the_engine 0 \
	"the block flag is absent when the caller named none"
if [ -s "$CASE_HOME/serve_argv" ]; then
	grep -qx -- '--draft-block-size' "$CASE_HOME/serve_argv" &&
		note_bad "the serve argv carried --draft-block-size with no caller asking"
else
	note_bad "the stub recorded no serve argv"
fi
verdict

run_case draft_block_passed_when_asked 0 \
	"the block flag is passed with the value the caller named" \
	'ARGS:--draft-block-size=7'
if [ -s "$CASE_HOME/serve_argv" ]; then
	grep -qx -- '--draft-block-size' "$CASE_HOME/serve_argv" ||
		note_bad "the serve argv dropped --draft-block-size the caller asked for"
	grep -qx -- '7' "$CASE_HOME/serve_argv" ||
		note_bad "the serve argv did not carry the block the caller named"
else
	note_bad "the stub recorded no serve argv"
fi
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

# `charged` says the round loop drained its pipeline at every phase boundary, so
# its decode rate measures a different engine. `--log info` does not prevent it:
# RUST_LOG takes precedence over the preset and this script does not clear it.
run_case charged_schedule_is_recorded 0 \
	"a normal run says on its row that it was not charged"
case "$(notes_of mtp)" in
*"charged=false"*) ;;
*) note_bad "notes do not say which schedule produced the row: $(notes_of mtp)" ;;
esac
verdict

run_case charged_row_is_refused 1 \
	"a charged run is refused rather than filed" \
	'STUB_CHARGED=true' \
	'GREP:charged=true'
no_row mtp
verdict

run_case charged_must_be_a_boolean 1 \
	"a charged value that is neither true nor false is refused, not coerced" \
	'STUB_CHARGED="\"false\""' \
	'GREP:not a boolean'
no_row mtp
verdict

run_case missing_charged_refused 1 \
	"a log that does not say which schedule produced it files no row" \
	'STUB_DROP_FIELDS=charged' \
	'GREP:carries no charged field'
no_row mtp
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

# ── The two arms' answers ─────────────────────────────────────────────────────

# answer_of <config> — the `answer=` field of that row's notes.
answer_of() {
	notes_of "$1" | tr ' ' '\n' | sed -n 's/^answer=//p'
}

# answer_check_of <config> — the `answer_check=` field of that row's notes.
answer_check_of() {
	notes_of "$1" | tr ' ' '\n' | sed -n 's/^answer_check=//p'
}

# The guard's other side, and the one a one-sided guard fails: two arms that
# answered the same thing are measured and recorded, and both rows say the
# comparison was made rather than leaving a reader to assume it.
run_case agreeing_arms_are_recorded 0 \
	"two arms that answered alike are recorded, and the rows say so"
[ "$(answer_check_of normal)" = "plain_repeatable" ] ||
	note_bad "normal answer_check=$(answer_check_of normal)"
[ "$(answer_check_of mtp)" = "matches_plain" ] ||
	note_bad "mtp answer_check=$(answer_check_of mtp)"
[ -n "$(answer_of mtp)" ] && [ "$(answer_of mtp)" = "$(answer_of normal)" ] ||
	note_bad "rows carry answers $(answer_of normal) and $(answer_of mtp)"
no_judge
verdict

# The defect this exists for: a drafter that changed the answer at temperature 0
# has had a throughput row filed for it. The arms stream the same token count at
# the same rate, so nothing but the text separates them.
run_case divergent_answer_refused 1 \
	"a speculative arm that answered differently, at a decision the verifier was sure of, files no row" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_MARGIN_AT=3=5.0' \
	'GREP:the difference is not a near tie' \
	'GREP:first differs from the no-drafter arm at token 3' \
	'GREP:surer than 0\.[0-9]+ of its own decisions'
no_row mtp
verdict

# The preview a row already carried is 64 characters, and these arms agree for
# the first 18 tokens. Comparing previews would call this one answer.
run_case divergence_past_the_preview_refused 1 \
	"a divergence past the preview's 64 characters is still refused" \
	'STUB_TOKENS=40' 'STUB_SPEC_DIVERGE_FROM=30' 'STUB_MARGIN_AT=30=5.0' \
	'GREP:first differs from the no-drafter arm at token 30'
no_row mtp
verdict


# ── A near-tie flip ───────────────────────────────────────────────────────────

# judged_pair <plain-token> <spec-token> — the recorded argv handed the judge the
# no-drafter pass as --plain and the speculative pass as --spec: the plain body
# holds the token the plain arm wrote at the divergence and the speculative one
# the token the loop wrote there.
judged_pair() {
	local plain="$CASE_HOME/spec_answer_plain.json" spec="$CASE_HOME/spec_answer_spec.json"
	if [ ! -s "$plain" ] || [ ! -s "$spec" ]; then
		note_bad "the judge was not handed two bodies"
		return 0
	fi
	grep -q "\"token\": \"$1 \"" "$plain" || note_bad "--plain does not hold the no-drafter pass ($1 missing)"
	grep -q "\"token\": \"$2 \"" "$spec" || note_bad "--spec does not hold the speculative pass ($2 missing)"
	grep -q "\"token\": \"$2 \"" "$plain" && note_bad "--plain holds a speculative token ($2)"
	return 0
}

# The pass sits between the warmup and the measured runs of each phase, so it is
# in the state the measured runs are in, and its read-back is paused off before
# they start. The stub notes each request and the sleep shim each pause.
run_case logprob_pass_follows_the_warmup_and_is_paused 0 \
	"each phase sends warmup, pass, measured runs, with a 15 s pause after the pass"
want="plain stream,plain unstreamed,plain stream,plain stream,plain stream"
want="$want,spec stream,spec unstreamed,spec stream,spec stream,spec stream"
got="$(paste -sd, "$CASE_HOME/requests" 2>/dev/null)"
[ "$got" = "$want" ] || note_bad "requests were '$got' (want '$want')"
awk '{ l[NR] = $0 }
	END {
		for (i = 1; i <= NR; i++)
			if (l[i] == "15") { n++; if (l[i - 1] != "5" || l[i + 1] != "5") bad = 1 }
		exit !(n == 2 && !bad)
	}' "$CASE_HOME/sleeps" ||
	note_bad "the sleeps were '$(paste -sd' ' "$CASE_HOME/sleeps")': want one 15 s pause after each pass, between 5 s sleeps"
verdict

# A judge that is not the one this script was written for: a binary that predates
# the subcommand prints nothing, and a status that disagrees with the verdict is a
# judge nobody has read.
run_case judge_without_a_verdict_refused 1 \
	"a binary that predates the subcommand cannot be taken for a verdict" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_MARGIN_AT=3=0.0' 'STUB_JUDGE=silent' \
	'GREP:exited 2 with no verdict'
no_row mtp
verdict

run_case judge_status_must_match_its_verdict 1 \
	"an exit status that disagrees with the verdict is refused by name" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_MARGIN_AT=3=0.0' 'STUB_JUDGE=status1' \
	'GREP:exited 1 but said verdict=near_tie'
no_row mtp
verdict

# The point of the change: the speculative arm parts from the plain one at token 3,
# where the plain arm's top two candidates are tied. The row is filed, says where
# and by how much, and the plain row stays as it was.
run_case near_tie_flip_is_filed 0 \
	"a divergence at a tied decision files a row that names it" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_MARGIN_AT=3=0.0'
[ "$(answer_check_of mtp)" = "near_tie@3" ] ||
	note_bad "mtp answer_check=$(answer_check_of mtp) (want near_tie@3)"
[ "$(answer_check_of normal)" = "plain_repeatable" ] ||
	note_bad "normal answer_check=$(answer_check_of normal)"
case "$(notes_of mtp)" in
*" divergence_rank=0.0000 divergence_margin=0.0000 "*) ;;
*) note_bad "mtp notes do not carry the rank and margin: $(notes_of mtp)" ;;
esac
case "$(notes_of normal)" in
*divergence_*) note_bad "the plain row carries a divergence figure: $(notes_of normal)" ;;
esac
[ -n "$(answer_of mtp)" ] && [ "$(answer_of mtp)" != "$(answer_of normal)" ] ||
	note_bad "rows carry answers $(answer_of normal) and $(answer_of mtp)"
judged_pair t3 x3
verdict

# The same divergence with the plain arm's margins read from the right file: the
# plain arm is tied at token 3 and the speculative arm's own margin there is a
# confident one. A script that handed the judge the arms the wrong way round reads
# the confident margin and refuses.
run_case judge_reads_the_plain_arms_margins 0 \
	"the margins come from the no-drafter pass, not the speculative one" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_MARGIN_AT=3=0.0' 'STUB_SPEC_MARGIN_AT=3=5.0'
[ "$(answer_check_of mtp)" = "near_tie@3" ] ||
	note_bad "mtp answer_check=$(answer_check_of mtp) (want near_tie@3)"
judged_pair t3 x3
verdict

# The judge cannot read what the pass does not carry. Each refusal names the
# input it could not read, and files no speculative row.
run_case one_candidate_per_record_refused 1 \
	"a pass listing one candidate per position has no margin to judge by" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_TOP_LOGPROBS=1' \
	'GREP:cannot be judged' \
	'GREP:normal_logprobs.json: record 0 carries fewer than two top_logprobs'
no_row mtp
verdict

run_case missing_record_refused 1 \
	"a pass with fewer records than tokens is refused by name" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_RECORDS_SHORT=1' \
	'GREP:cannot be judged' \
	'GREP:normal_logprobs.json: 7 logprob records for 8 completion tokens, so a record index is not a token position'
no_row mtp
verdict

run_case no_logprobs_refused 1 \
	"a response with no logprobs at all is refused" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_NO_LOGPROBS=1' \
	'GREP:cannot be judged' \
	'GREP:no choices.0..logprobs.content'
no_row mtp
verdict

# A pass whose answer is not the one its arm's measured runs gave carries the
# margins of some other answer. The normal row is already filed: it is valid.
run_case unbound_plain_pass_refused 1 \
	"a plain pass that answered differently from its measured runs cannot judge" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_MARGIN_AT=3=0.0' 'STUB_UNTIMED_DIFFERS=plain' \
	'GREP:normal: the logprob pass of the no-drafter arm answered' \
	'GREP:its margins are not those of the answer'
no_row mtp
[ -n "$(record_of normal)" ] || note_bad "the valid normal row was not filed"
verdict

# And when the arms agree the pass is never read, so the same stray pass costs
# nothing.
run_case unbound_pass_is_not_read_when_arms_agree 0 \
	"a plain pass that answered differently does not matter when the arms agree" \
	'STUB_UNTIMED_DIFFERS=plain'
[ "$(answer_check_of mtp)" = "matches_plain" ] ||
	note_bad "mtp answer_check=$(answer_check_of mtp)"
no_judge
verdict

run_case unbound_speculative_pass_refused 1 \
	"a speculative pass that answered differently from its measured runs is refused" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_MARGIN_AT=3=0.0' 'STUB_UNTIMED_DIFFERS=spec' \
	'GREP:mtp: the logprob pass of the speculative arm answered'
no_row mtp
verdict

# A server that does not answer the pass leaves nothing to bind or judge by, and
# the run stops before it files the plain row.
run_case unanswered_pass_refused 1 \
	"a server that does not answer the logprob pass stops the run" \
	'STUB_UNSTREAMED_STATUS=500' \
	'GREP:normal: the server did not answer the logprob pass'
no_row normal
no_row mtp
verdict

# The speculative arm wrote the whole reference answer and went on. Nothing
# differs before the reference ends, so no rank can excuse it.
run_case ran_past_the_reference_refused 1 \
	"an arm that reproduces the reference and runs on is refused" \
	'STUB_SPEC_EXTRA_TOKENS=2' \
	'GREP:reproduced the whole reference answer' \
	'GREP:did not stop where the verifier stopped'
no_row mtp
verdict

# A tie at the point the shorter arm stopped would pass the rank; the length
# ratio is what refuses it.
run_case short_arm_refused 1 \
	"an arm far shorter than the reference is refused on its length" \
	'STUB_SPEC_EXTRA_TOKENS=-5' 'STUB_MARGIN_AT=3=0.0' \
	'GREP:a ratio of 0\.3750'
no_row mtp
verdict

run_case speculative_runs_disagreeing_refused 1 \
	"measured speculative runs that differ from each other have no answer to judge" \
	'STUB_SPEC_VARY=1' \
	'GREP:mtp: the 3 measured runs answered differently from each other' \
	'GREP:no one speculative answer to judge'
no_row mtp
no_judge
verdict

run_case indistinguishable_streams_refused 1 \
	"two answers whose pieces read alike cannot be told apart by the oracle" \
	'STUB_SPEC_DIVERGE_FROM=3' 'STUB_SAME_PIECES=1' \
	'GREP:cannot be judged' \
	'GREP:same token pieces'
no_row mtp
verdict

# Under a sampler two arms are two draws, so a difference between them is not a
# finding. The row is filed and states that it was not compared — a silent skip
# would read exactly like a row that agreed.
run_case sampled_run_is_not_compared 0 \
	"a sampled run records that its arms were not compared" \
	'STUB_SAMPLER=1' 'STUB_SPEC_DIVERGE_FROM=3' \
	'GREP:two arms are two draws and need not agree'
[ "$(answer_check_of mtp)" = "sampled" ] ||
	note_bad "mtp answer_check=$(answer_check_of mtp)"
[ "$(answer_check_of normal)" = "sampled" ] ||
	note_bad "normal answer_check=$(answer_check_of normal)"
no_judge
verdict

# The plain arm is the reference. One that answered three different things
# without a sampler to explain it has none to offer, and the run stops before
# either row.
run_case unrepeatable_plain_arm_refused 1 \
	"a no-drafter arm that did not repeat itself is not a reference" \
	'STUB_VARY_PER_REQUEST=1' \
	'GREP:answered differently from each other'
no_row normal
no_row mtp
verdict

# Sampler events covering three of five requests: some ran sampled and some did
# not, so the run has no one disposition and neither answer_check would be true
# of it.
run_case partial_sampler_events_refused 1 \
	"a run whose requests did not share one sampling setup is refused" \
	'STUB_SAMPLER=1' 'STUB_SAMPLER_AFTER=2' \
	'GREP:covered some of them and not the others' \
	'GREP:does not say whether the engine ran a sampler'
no_row normal
no_row mtp
verdict

# A run stopped from outside while its server is up — a CI timeout, an operator
# — stops that server on the way out rather than leaving it on the claim.
sigterm_case() {
	CASE_NAME="sigterm_stops_the_server"
	CASE_WHAT="a killed run takes its server with it"
	CASE_HOME="$WORK/home_$CASE_NAME"
	CASE_OUT="$WORK/$CASE_NAME.log"
	CASE_BAD=""
	mkdir -p "$CASE_HOME/logs" "$CASE_HOME/metrics/buffer/pending"
	printf '%s\n' "--hold $CASE_HOME/hold" >"$CASE_HOME/stub_args"

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
		bash "$FAKE_ROOT/scripts/spec_bench.sh" --port "$PORT" --server-stamps >"$CASE_OUT" 2>&1 &
	local harness=$!

	request_held_or_harness_gone() {
		[ -e "$CASE_HOME/hold.held" ] || ! kill -0 "$harness" 2>/dev/null
	}
	harness_gone() { ! kill -0 "$harness" 2>/dev/null; }

	if ! wait_until 300 request_held_or_harness_gone || [ ! -e "$CASE_HOME/hold.held" ]; then
		note_bad "no request was held within 300 s, or the harness ended before it sent one; nothing was stopped"
		kill "$harness" 2>/dev/null || true
		wait "$harness" 2>/dev/null || true
		stop_stubs "$CASE_HOME/stub_bound"
		return 0
	fi

	kill -TERM "$harness" 2>/dev/null || true
	: >"$CASE_HOME/hold.release"
	if ! wait_until 300 harness_gone; then
		note_bad "the harness was still running 300 s after SIGTERM"
		kill -9 "$harness" 2>/dev/null || true
	fi
	wait "$harness" 2>/dev/null || true
	note_survivors "$CASE_HOME/stub_bound"
	stop_stubs "$CASE_HOME/stub_bound"
}
sigterm_case
verdict

# ── The unstreamed pass and the judge ─────────────────────────────────────────

# stub_answers — the stub server on $PORT accepts connections.
stub_answers() {
	python3 -c "import socket; socket.create_connection(('127.0.0.1', $PORT), 1).close()" 2>/dev/null
}

# start_stub <home> <plain|spec> [KEY=VALUE ...] — one stub server on $PORT, as
# the harness starts it. Sets STUB_SERVER_PID.
start_stub() {
	local home="$1" arm="$2" extra=()
	shift 2
	[ "$arm" = spec ] && extra=(--draft-model "$WORK/drafter")
	mkdir -p "$home/logs"
	: >"$home/stub_args"
	env -i \
		PATH="$SHIM_DIR:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
		HOME="$WORK" \
		RMLX_HOME="$home" \
		STUB_TOKENS=8 \
		STUB_GAP_S=0.05 \
		STUB_PREFILL_S=0.05 \
		STUB_EMITTED=128 \
		STUB_ELAPSED_MS=2560 \
		STUB_DECODE_TPS_SEQ='"Some(20.0)"' \
		STUB_PROMPT_TOKENS=1234 \
		STUB_BOUND_FLAG="$home/stub_bound" \
		"$@" \
		"$STUB" serve --port "$PORT" ${extra[@]+"${extra[@]}"} >/dev/null 2>&1 &
	STUB_SERVER_PID=$!
	wait_until 30 stub_answers
}

stop_stub() {
	kill "$STUB_SERVER_PID" 2>/dev/null || true
	wait "$STUB_SERVER_PID" 2>/dev/null || true
}

# The untimed pass reads the same answer the streamed run digests, and the
# judge reads the two passes: this is the contract the bench builds on. The arms
# part at token 3, where the plain arm's margin is tied.
unstreamed_pass_smoke() {
	CASE_NAME="unstreamed_pass_binds_and_judges"
	CASE_WHAT="an unstreamed pass has its stream's digest, and the judge reads the pair"
	CASE_HOME="$WORK/home_$CASE_NAME"
	CASE_OUT="$WORK/$CASE_NAME.log"
	CASE_BAD=""
	mkdir -p "$CASE_HOME"
	: >"$CASE_OUT"
	local reader="$REPO_ROOT/scripts/lib/sse_decode_window.py"
	local url="http://127.0.0.1:$PORT/v1/chat/completions"
	local messages='[{"role": "user", "content": "hi"}]'
	printf '{"model": "stub", "messages": %s, "stream": true}\n' "$messages" >"$CASE_HOME/streamed.json"
	printf '{"model": "stub", "messages": %s, "stream": false, "logprobs": true, "top_logprobs": 2}\n' \
		"$messages" >"$CASE_HOME/unstreamed.json"

	local arm streamed unstreamed digests=()
	for arm in plain spec; do
		if [ "$arm" = spec ]; then
			start_stub "$CASE_HOME/$arm" spec STUB_SPEC_DIVERGE_FROM=3
		else
			start_stub "$CASE_HOME/$arm" plain STUB_MARGIN_AT=3=0.0
		fi
		streamed="$(python3 "$reader" --url "$url" --body-file "$CASE_HOME/streamed.json" --server-stamps 2>>"$CASE_OUT" |
			sed -n 's/^answer_sha256=//p')"
		unstreamed="$(python3 "$reader" --url "$url" --body-file "$CASE_HOME/unstreamed.json" \
			--complete "$CASE_HOME/$arm.json" 2>>"$CASE_OUT" | sed -n 's/^answer_sha256=//p')"
		stop_stub
		note_survivors "$CASE_HOME/$arm/stub_bound"
		stop_stubs "$CASE_HOME/$arm/stub_bound"
		[ -n "$streamed" ] || note_bad "$arm: the streamed reader printed no digest"
		[ "$streamed" = "$unstreamed" ] ||
			note_bad "$arm: unstreamed digest '$unstreamed' is not the streamed '$streamed'"
		digests+=("$streamed")
	done
	[ "${digests[0]}" != "${digests[1]}" ] || note_bad "the arms answered alike, so nothing parted them"

	local got=0 out
	set +e
	out="$(env -i PATH="$SHIM_DIR:/usr/bin:/bin" HOME="$WORK" RMLX_HOME="$CASE_HOME" \
		"$STUB" eval spec-answer --plain "$CASE_HOME/plain.json" --spec "$CASE_HOME/spec.json" 2>>"$CASE_OUT")"
	got=$?
	set -e
	[ "$got" -eq 0 ] || note_bad "spec-answer exit=$got (want 0): $out"
	printf '%s\n' "$out" | grep -qx 'verdict=near_tie' || note_bad "no near_tie verdict: $out"
	printf '%s\n' "$out" | grep -qx 'position=3' || note_bad "not parted at token 3: $out"
	[ "$(sed -n '4p' "$CASE_HOME/spec_answer_argv" 2>/dev/null)" = "$CASE_HOME/plain.json" ] ||
		note_bad "the stub binary was not handed the plain body as --plain"
	cmp -s "$CASE_HOME/spec_answer_plain.json" "$CASE_HOME/plain.json" ||
		note_bad "the plain body the stub binary kept is not the plain pass's"
	cmp -s "$CASE_HOME/spec_answer_spec.json" "$CASE_HOME/spec.json" ||
		note_bad "the spec body the stub binary kept is not the speculative pass's"
}
unstreamed_pass_smoke
verdict

# unstreamed_status <label> <body-file> [KEY=VALUE ...] — one unstreamed request
# to a fresh plain stub, after the streamed one; prints the reader's exit status
# and leaves its response in $CASE_HOME/<label>.json.
unstreamed_status() {
	local label="$1" body="$2" got=0
	shift 2
	start_stub "$CASE_HOME/$label" plain "$@"
	python3 "$REPO_ROOT/scripts/lib/sse_decode_window.py" --url "http://127.0.0.1:$PORT/v1/chat/completions" \
		--body-file "$CASE_HOME/streamed.json" --server-stamps >/dev/null 2>&1 || true
	python3 "$REPO_ROOT/scripts/lib/sse_decode_window.py" --url "http://127.0.0.1:$PORT/v1/chat/completions" \
		--body-file "$body" --complete "${OUT_PATH:-$CASE_HOME/$label.json}" >/dev/null 2>&1 || got=$?
	stop_stub
	stop_stubs "$CASE_HOME/$label/stub_bound"
	echo "$got"
}

# The stub holds the unstreamed request to the streamed one, so a pass body that
# strays from it is an answer, and the script's pass is only as good as that. The
# reader's own exit codes for a response it cannot use are pinned here too: the
# bench reads every one of them as "the server did not answer".
unstreamed_request_contract() {
	CASE_NAME="unstreamed_request_contract"
	CASE_WHAT="the stub answers the pass the body asks for, and the reader names each failure"
	CASE_HOME="$WORK/home_$CASE_NAME"
	CASE_OUT="$WORK/$CASE_NAME.log"
	CASE_BAD=""
	mkdir -p "$CASE_HOME"
	: >"$CASE_OUT"
	python3 - "$CASE_HOME" <<'PYBODIES'
import json, sys

base = {"model": "stub", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8, "temperature": 0, "seed": 42}
unstreamed = {"stream": False, "logprobs": True, "top_logprobs": 2}
bodies = {
    "streamed": {"stream": True, "stream_options": {"include_usage": True}},
    "pass": unstreamed,
    "top1": {**unstreamed, "top_logprobs": 1},
    "nologprobs": {"stream": False},
    "longer": {**unstreamed, "max_tokens": 64},
    "colder": {**unstreamed, "temperature": 0.7},
    "otherseed": {**unstreamed, "seed": 7},
    "otherprompt": {**unstreamed, "messages": [{"role": "user", "content": "ho"}]},
}
for name, extra in bodies.items():
    with open(f"{sys.argv[1]}/{name}.json", "w") as handle:
        json.dump({**base, **extra}, handle)
PYBODIES

	# candidates_of <response> — how many candidates the first record lists, or
	# `none` when the response carries no logprobs.
	candidates_of() {
		python3 -c 'import json, sys
choice = json.load(open(sys.argv[1]))["choices"][0]
print(len(choice["logprobs"]["content"][0]["top_logprobs"]) if "logprobs" in choice else "none")' "$1" 2>/dev/null
	}
	local label status
	for label in pass top1 nologprobs; do
		status="$(unstreamed_status "ok_$label" "$CASE_HOME/$label.json")"
		[ "$status" = "0" ] || note_bad "$label: the reader exited $status (want 0)"
	done
	[ "$(candidates_of "$CASE_HOME/ok_pass.json")" = "2" ] || note_bad "top_logprobs 2 listed $(candidates_of "$CASE_HOME/ok_pass.json") candidates"
	[ "$(candidates_of "$CASE_HOME/ok_top1.json")" = "1" ] || note_bad "top_logprobs 1 listed $(candidates_of "$CASE_HOME/ok_top1.json") candidates"
	[ "$(candidates_of "$CASE_HOME/ok_nologprobs.json")" = "none" ] || note_bad "a request without logprobs got them"
	for label in longer colder otherseed otherprompt; do
		status="$(unstreamed_status "bad_$label" "$CASE_HOME/$label.json")"
		[ "$status" = "4" ] || note_bad "$label: a pass that is not the measured request exited $status (want 4, the stub's 400)"
	done
	status="$(unstreamed_status status500 "$CASE_HOME/pass.json" STUB_UNSTREAMED_STATUS=500)"
	[ "$status" = "4" ] || note_bad "an answer of 500 exited $status (want 4)"
	status="$(unstreamed_status junk "$CASE_HOME/pass.json" STUB_UNSTREAMED_BODY=junk)"
	[ "$status" = "6" ] || note_bad "a response that is not a chat completion exited $status (want 6)"
	status="$(OUT_PATH="$CASE_HOME/no/such/dir/out.json" unstreamed_status unwritable "$CASE_HOME/pass.json")"
	[ "$status" = "2" ] || note_bad "a response that cannot be written exited $status (want 2)"
}
unstreamed_request_contract
verdict

echo
echo "passed=$PASSED failed=$FAILED"
[ "$FAILED" -eq 0 ] || exit 1
