#!/usr/bin/env bash
# spec_bench_published_selftest.sh — mutation check for
# scripts/spec_bench_published.sh.
#
# WHY THIS EXISTS
#
# The published harness exists to produce a number that goes next to somebody
# else's number in a post. Nothing downstream can sanity-check it: 91 and 208
# tok/s are both plausible, a mean over "the samples that worked" looks exactly
# like a mean over the sample set, and three passes that disagree by 9% average
# to a figure none of them produced. The defences are all refusals, and a
# refusal is only real while something watches it fire.
#
# Each case drives the real script against a stub `rmlx` whose server streams a
# canned response and writes canned TTFT / ITL / round-loop events, over a
# shrunken copy of the checked-in sample sets. Every case asserts the literal
# exit code, and every refusal asserts the reason as well: a guard that refuses
# for the wrong reason stops refusing when that reason moves.
#
# The host is supplied by a `ps` shim and the clock by a `date` shim, so no case
# reads this machine's process table or its time and no verdict here depends on
# what else it is doing.
# `--synthetic-arms` is the boundary between a run that measures and a run that
# exercises this logic, and it is pinned in both directions: the host gates
# still fire without it, and with it a hostile host and a quiet host reach the
# same verdict.
#
# NO VERDICT HERE READS THIS HOST'S CLOCK. The harness cross-checks the
# engine's decode rate against the same window read at the client. The stub
# stamps every content chunk with the time it is due (`: sent-at=`) and every
# case passes `--server-stamps`, so the client reads those stamps and the window
# is exact whatever else the machine is doing. A stamped result is not a
# measurement: it says `client_clock: server_stamps`, and the ingest refuses it.
# The path with no stamps, which is the one a real server takes, is held by
# cases that feed the reader a stream whose clock moves only as lines are
# delivered, and by a case that shows the client sends the request itself.
#
# No GPU, no model, no DB.
#
# Exit codes: 0 — every case behaved; 1 — at least one did not; 2 — the
# fixtures themselves could not be built.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/spec_bench_published.sh"
AGGREGATE="${REPO_ROOT}/scripts/lib/published_aggregate.py"
PUBLISHED="${REPO_ROOT}/prompts/published"

[ -r "${HARNESS}" ] || { echo "ERROR: missing ${HARNESS}" >&2; exit 2; }

# shellcheck source=scripts/lib/selftest_support.sh
. "${REPO_ROOT}/scripts/lib/selftest_support.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_published_selftest.XXXXXX")" || exit 2
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0

# ── Fake repo root ────────────────────────────────────────────────────────────
#
# The harness resolves its binary and its lib/ readers from `$0/..`, so a root
# holding the real scripts/ and prompts/ plus a stub binary is how the stub gets
# used.
FAKE_ROOT="${WORK}/repo"
mkdir -p "${FAKE_ROOT}/target/release-perf"
ln -s "${REPO_ROOT}/scripts" "${FAKE_ROOT}/scripts"
ln -s "${REPO_ROOT}/prompts" "${FAKE_ROOT}/prompts"

# ── Sample sets ───────────────────────────────────────────────────────────────
#
# A shrunken copy of the checked-in sets: the first few samples of each dataset.
# It is NOT held to `published_samples.py`, and nothing here asks it to be — that
# gate anchors the published sets against constants in its own source, so a
# shrunken copy cannot satisfy it and should not. The harness reaches this root
# only through `--samples-root`, which is the unverified operator override, and
# a case below pins that the default path does verify.
#
# Different sizes on purpose — 1, 2 and 3 — so the macro average, a mean over
# the cells, a mean over the datasets-including-the-second-budget, and a pooled
# mean over the rows are four different numbers.
SAMPLES="${WORK}/samples"
mkdir -p "${SAMPLES}"
shrink() { # shrink <src-root> <dst-root>
	python3 - "$1" "$2" <<'PY' || exit 2
import hashlib, json, pathlib, sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
KEEP = {"mt_bench": 1, "math_500": 2, "humaneval": 3}

# The manifest keeps its shape, so a run that reaches this root as the
# published one is refused for what differs — the sample counts and the file
# digests — and not for a manifest that was never built.
man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
for entry in man["datasets"]:
    keep = KEEP[entry["key"]]
    doc = json.loads((src / entry["file"]).read_text(encoding="utf-8"))
    doc["samples"] = doc["samples"][:keep]
    assert len(doc["samples"]) == keep, entry["key"]
    blob = (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (dst / entry["file"]).write_bytes(blob)
    entry["count"] = keep
    entry["selected_ids"] = entry["selected_ids"][:keep]
    entry["file_bytes"] = len(blob)
    entry["file_sha256"] = hashlib.sha256(blob).hexdigest()
(dst / "manifest.json").write_text(
    json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
}
shrink "${PUBLISHED}" "${SAMPLES}"

# 1 + 2 + 3 + 2 (MATH-500 again at 4096) requests per pass.
REQUESTS_PER_PASS=8
# One TTFT, one ITL and one sampler event per request: the untimed warmup, the
# fixed-length prompt, and the eight cell requests.
EVENTS_PER_PASS=10

# A second fake root whose `prompts/published` IS the shrunken copy. A run that
# passes no --samples-root reaches it as the published root, and the gate that
# anchors the published sets refuses it — which is how the default path is shown
# to verify without benching all 336 samples.
CANON_ROOT="${WORK}/repo_canonical"
mkdir -p "${CANON_ROOT}/target/release-perf" "${CANON_ROOT}/prompts"
ln -s "${REPO_ROOT}/scripts" "${CANON_ROOT}/scripts"
ln -s "${SAMPLES}" "${CANON_ROOT}/prompts/published"

# ── Snapshots ─────────────────────────────────────────────────────────────────

VERIFIER="${WORK}/models/stub-ns__stub-model"
mkdir -p "${VERIFIER}" "${WORK}/drafter"
printf '%s\n' '{"quantization": {"mode": "mxfp8", "bits": 8, "group_size": 32}}' \
    >"${VERIFIER}/config.json"
printf '%s\n' '{"temperature": 0.6, "top_p": 0.95, "top_k": 20}' \
    >"${VERIFIER}/generation_config.json"

# The same snapshot with no sampling defaults at all, and one that states only
# half of them.
NO_DEFAULTS="${WORK}/models/bare-ns__bare-model"
mkdir -p "${NO_DEFAULTS}"
cp "${VERIFIER}/config.json" "${NO_DEFAULTS}/config.json"

HALF_DEFAULTS="${WORK}/models/half-ns__half-model"
mkdir -p "${HALF_DEFAULTS}"
cp "${VERIFIER}/config.json" "${HALF_DEFAULTS}/config.json"
printf '%s\n' '{"temperature": 0.6, "top_p": 0.95}' \
	>"${HALF_DEFAULTS}/generation_config.json"

# A checkpoint whose own default is greedy. It resolves no sampler, so the
# engine writes no sampler event and the harness must expect none — the other
# arm of a branch whose first arm every sampled case exercises.
GREEDY="${WORK}/models/greedy-ns__greedy-model"
mkdir -p "${GREEDY}"
cp "${VERIFIER}/config.json" "${GREEDY}/config.json"
printf '%s\n' '{"temperature": 0.0, "top_p": 1.0, "top_k": 0}' \
	>"${GREEDY}/generation_config.json"

# ── Shims ─────────────────────────────────────────────────────────────────────
#
# `sleep` shortens the harness's waits to 50 ms. Nothing reads their length: the
# host windows come from the `ps` shim and the pass length from the `date` shim.
SHIMS="${WORK}/shims"
mkdir -p "${SHIMS}"
printf '#!/bin/sh\nexec /bin/sleep 0.05\n' >"${SHIMS}/sleep"
# The thermal reading is a machine reading, so it is shimmed for the same reason
# the process table is: no verdict here may depend on what this Mac's thermal
# history happens to hold. The default says what an unthrottled Mac says — no
# level has ever been posted — which is the state that must NOT taint.
cat >"${SHIMS}/pmset" <<'PMSETEOF'
#!/bin/sh
echo "Note: No thermal warning level has been recorded"
PMSETEOF
# The harness measures a pass in whole seconds with `date +%s`. Against a stub
# that answers at once a pass takes under a second and its host window would be
# empty, so `+%s` reads a counter that moves one second per call, kept in a file
# beside the shim. Any other format is the real date.
cat >"${SHIMS}/date" <<'DATEEOF'
#!/bin/sh
if [ "$1" = "+%s" ]; then
	n=$(cat "$0.count" 2>/dev/null || echo 1000000)
	echo $((n + 1)) >"$0.count"
	echo "$n"
else
	exec /bin/date "$@"
fi
DATEEOF
chmod +x "${SHIMS}/sleep" "${SHIMS}/pmset" "${SHIMS}/date"

# A harness started in the same second as an earlier one files its run under the
# same name. The run's start stamp is the only thing that varies it, so a case
# that needs the collision pins the stamp and leaves every other `date` call to
# the shim above.
mkdir -p "${WORK}/fixed_ts"
cat >"${WORK}/fixed_ts/date" <<DATEEOF
#!/bin/sh
case "\$*" in
*%Y-%m-%dT%H:%M:%SZ*) echo 2026-09-06T00:00:00Z ;;
*) exec "${SHIMS}/date" "\$@" ;;
esac
DATEEOF
chmod +x "${WORK}/fixed_ts/date"

# A Mac that posted a level and is at full speed, one that is being held back,
# and one where the tool is not there to ask.
for spec in nominal:100 throttled:62; do
	dir="${WORK}/therm_${spec%%:*}"
	mkdir -p "${dir}"
	{
		echo '#!/bin/sh'
		echo 'echo "Note: Thermal pressure recorded"'
		echo "echo 'CPU_Speed_Limit \t= ${spec#*:}'"
	} >"${dir}/pmset"
	chmod +x "${dir}/pmset"
done
mkdir -p "${WORK}/therm_unreadable"
printf '#!/bin/sh\nexit 1\n' >"${WORK}/therm_unreadable/pmset"
chmod +x "${WORK}/therm_unreadable/pmset"

# Two synthetic machines, on the shape `cpu_snapshot` reads. It refuses a
# process table under 20 rows, so both emit 40 idle ones.
mkdir -p "${WORK}/quiet" "${WORK}/hostile" "${WORK}/flaky"
{
    echo '#!/usr/bin/env bash'
    echo 'for i in $(seq 1 40); do printf "%6d %12s %s\n" $((5000 + i)) "0:00.10" "/usr/sbin/idle$i"; done'
} >"${WORK}/quiet/ps"
{
    echo '#!/usr/bin/env bash'
    echo "n=\$(cat '${WORK}/hog.cnt' 2>/dev/null || echo 0)"
    echo "echo \$((n + 1)) >'${WORK}/hog.cnt'"
    echo 'printf "%6d %12s %s\n" 4242 "0:$((n * 100)).00" /usr/local/bin/hog'
    echo 'for i in $(seq 1 40); do printf "%6d %12s %s\n" $((5000 + i)) "0:00.10" "/usr/sbin/idle$i"; done'
} >"${WORK}/hostile/ps"
# A machine that answers the entry gate and then stops answering: a full table
# for the first two calls, a truncated one after. `cpu_snapshot` refuses a table
# under 20 rows, which is `unmeasured` — not knowing, which is not `quiet`.
{
    echo '#!/usr/bin/env bash'
    echo "n=\$(cat '${WORK}/flaky.cnt' 2>/dev/null || echo 0)"
    echo "echo \$((n + 1)) >'${WORK}/flaky.cnt'"
    echo 'if [ "$n" -lt 2 ]; then rows=40; else rows=3; fi'
    echo 'for i in $(seq 1 $rows); do printf "%6d %12s %s\n" $((5000 + i)) "0:00.10" "/usr/sbin/idle$i"; done'
} >"${WORK}/flaky/ps"
chmod +x "${WORK}/quiet/ps" "${WORK}/hostile/ps" "${WORK}/flaky/ps"

# ── Stub server ───────────────────────────────────────────────────────────────

SERVER_PY="${WORK}/stub_server.py"
cat >"${SERVER_PY}" <<'PYEOF'
"""Canned OpenAI-compatible server for the published-harness selftest.

Streams STUB_TOKENS content chunks a nominal gap apart, then a usage block.
STUB_GAP_MS is a gap matrix: `;` separates passes, `,` separates requests within
a pass, and the last entry of each list repeats. The stub does not wait between
chunks. Each content chunk is preceded by `: sent-at=<seconds>`, the time the
chunk is due on a fixed schedule, and the client reads that stamp instead of its
own clock. The wire really does carry 1000/gap tok/s, which is what the client
cross-check compares against, and no scheduling delay can move it.

Arguments after the port. The stub binary reads them from the file `stub_args`
in the case's RMLX_HOME, so a case sets them and no environment variable carries
them:

  --marks DIR          before each streamed request, note in DIR/seen whether the
                       client that sent it had already left DIR/mark.N (N counts
                       the streamed requests over every server of the case)
  --hold PREFIX        after its role chunk, the first streamed request creates
                       PREFIX.held and waits for PREFIX.release, so a case can
                       act while a request is in flight
  --hold-wait SECONDS  how long that wait lasts at most (300)
  --sample-wait SECONDS  how long the fixed-prompt request waits for a /metrics
                       reading that arrives while it is in flight (120)
  --inflight-log FILE  note each /metrics reading served while a request is in
                       flight

`/metrics` answers PHYS_BYTES while a streamed request is in flight and half of
it otherwise, so a resident peak read outside the request is a different number.
The server takes requests on threads, as the real one does. On the plain arm the
second streamed request of a server is the fixed prompt, and it waits for one
in-flight /metrics reading: the harness's poller always has a sample to file.

Per request it appends to the run log the events the harness reads back:

  generate_streaming: TTFT               unless capped by STUB_TTFT_LINES
  generate: ITL stats                    unless capped by STUB_ITL_LINES
  generate: metal peak alloc             plain arm only, unless capped by
                                         STUB_PEAK_LINES; rises 100 MiB per
                                         request unless STUB_METAL_PEAK_MB
  generate: prefill                      unless capped by STUB_PREFILL_LINES;
                                         the request's prompt count, STUB_REUSED
                                         of it from a cache (`all` for all of it),
                                         STUB_PREFILL_MS
                                         long (one value per pass, `;` apart)
  generate: host categorical sampler active
                                         unless STUB_SAMPLED=0, capped by
                                         STUB_SAMPLER_LINES
  <kind>_generate: done                  when the serve argv carried a drafter,
                                         or STUB_FORCE_DONE is set

An unstreamed request that asks for logprobs is a greedy-pass request: it is
answered with a completion whose text is a function of the prompt, so the plain
and the speculative stub agree on every sample, plus the word STUB_GREEDY_MARKS
names for the request's index (`i:WORD,...`) on a speculative stub. The oracle
stub reads that word from the speculative answer to stage its verdict.

`mean_ms` on the ITL event and `decode_tps` on the done line both report the
nominal gap, so the engine figure is exact and the client's reading of the same
window agrees with it. STUB_ITL_MEAN_MS overrides the first, which is how a
disagreement is staged.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("port", type=int)
parser.add_argument("--marks", default="")
parser.add_argument("--hold", default="")
parser.add_argument("--hold-wait", type=float, default=300.0)
parser.add_argument("--sample-wait", type=float, default=120.0)
parser.add_argument("--inflight-log", default="")
ARGS = parser.parse_args()


def gap_matrix(raw):
    return [
        [float(v) for v in row.split(",") if v != ""]
        for row in raw.split(";")
        if row != ""
    ]


TOKENS = int(os.environ.get("STUB_TOKENS", "6"))
GAP_MS = gap_matrix(os.environ.get("STUB_GAP_MS", "") or "40")
PASS = int(os.environ.get("STUB_PASS", "1"))
# The first content chunk is stamped this long after the role chunk, so a client
# that timed from the role chunk would read a rate lower than the wire's.
PREFILL_S = 0.15
PROMPT_TOKENS = int(os.environ.get("STUB_PROMPT_TOKENS", "1234"))
USAGE_TOKENS = int(os.environ.get("STUB_USAGE_TOKENS", "-1"))
# A pass that generates a different length than the others. Every request
# sends no seed, so three passes are meant to be one generation replayed;
# this is how a run where that did not hold is staged.
USAGE_TOKENS_PASS2 = os.environ.get("STUB_USAGE_TOKENS_PASS2", "")
TTFT_MS = float(os.environ.get("STUB_TTFT_MS", "42"))
TTFT_LINES = int(os.environ.get("STUB_TTFT_LINES", "-1"))
ITL_LINES = int(os.environ.get("STUB_ITL_LINES", "-1"))
ITL_MEAN_MS = os.environ.get("STUB_ITL_MEAN_MS", "")
# Which served request the ITL override starts at. The fixed-length prompt is
# request 1 of every pass, so this is how a disagreement can be staged on the
# cells without the fixed prompt tripping the same guard first.
ITL_MEAN_FROM = int(os.environ.get("STUB_ITL_MEAN_FROM", "0"))
PEAK_LINES = int(os.environ.get("STUB_PEAK_LINES", "-1"))
# Within one server the allocator's peak never falls, so the stub's rises by 100
# MiB per served request: the warmup, the fixed prompt and each cell read
# differently, and a reader that took the wrong request's event gets another
# number. Each pass is a new server, and these offsets make the three fixed-prompt
# readings 28100, 30100 and 28600, so their maximum is neither their mean, their
# minimum nor the last one.
PEAK_PASS_OFFSET_MB = {1: 0, 2: 2000, 3: 500}
PEAK_BASE_MB = 28000
PEAK_FIXED_MB = os.environ.get("STUB_METAL_PEAK_MB", "")
# The engine's account of each prefill: how many prompt tokens a cache supplied,
# how long the prefill took in each pass, and how many events are written.
# Prompts above this many characters are the long-context ones. They count
# STUB_LONG_PROMPT_TOKENS, and STUB_REUSED_LONG of them come from a cache.
LONG_CHARS = 20000
LONG_PROMPT_TOKENS = int(os.environ.get("STUB_LONG_PROMPT_TOKENS", "0"))
REUSED_LONG = int(os.environ.get("STUB_REUSED_LONG", "0"))
GREEDY_MARKS = dict(
    (int(i), word)
    for i, _, word in (m.partition(":") for m in os.environ.get("STUB_GREEDY_MARKS", "").split(",") if m)
)
REUSED_ALL = os.environ.get("STUB_REUSED", "0") == "all"
REUSED = 0 if REUSED_ALL else int(os.environ.get("STUB_REUSED", "0"))
PREFILL_MS = [float(v) for v in os.environ.get("STUB_PREFILL_MS", "250").split(";") if v != ""]
PREFILL_LINES = int(os.environ.get("STUB_PREFILL_LINES", "-1"))
# A prompt count that is not the request's, which is how an event paired with
# the wrong request is staged.
PREFILL_PROMPT = os.environ.get("STUB_PREFILL_PROMPT", "")
# Which served request the override starts at: the fixed-length prompt is request 1
# of every plain server, so a mismatch staged on the cells alone starts at 2.
PREFILL_PROMPT_FROM = int(os.environ.get("STUB_PREFILL_PROMPT_FROM", "0"))
# The greedy pass: with a prompt cache on, STUB_GREEDY_CACHE_HIT makes the engine
# serve every answer from it; STUB_GREEDY_REUSED reuses that many tokens whatever
# the cache is set to; STUB_GREEDY_NO_EVENT writes no prefill event for it.
CACHE_SLOTS = os.environ.get("STUB_CACHE_SLOTS", "4")
GREEDY_CACHE_HIT = os.environ.get("STUB_GREEDY_CACHE_HIT", "") == "1"
GREEDY_REUSED = int(os.environ.get("STUB_GREEDY_REUSED", "0"))
GREEDY_NO_EVENT = os.environ.get("STUB_GREEDY_NO_EVENT", "") == "1"
SAMPLED = os.environ.get("STUB_SAMPLED", "1") == "1"
SAMPLER_LINES = int(os.environ.get("STUB_SAMPLER_LINES", "-1"))
SAMPLER_TOP_K = int(os.environ.get("STUB_SAMPLER_TOP_K", "20"))
SAMPLER_TOP_K_PASS2 = os.environ.get("STUB_SAMPLER_TOP_K_PASS2", "")
SAMPLER_TOP_K_AFTER = os.environ.get("STUB_SAMPLER_TOP_K_AFTER", "")
OMIT_COMPLETION_TOKENS = os.environ.get("STUB_OMIT_COMPLETION_TOKENS", "") == "1"
# A prompt length that is a function of the body, which is what the fixed-prompt
# fit searches over. Four characters per token, so one " the" costs exactly one
# and the fit can land on any target the corpus reaches.
TOKEN_BASE = os.environ.get("STUB_TOKEN_BASE", "")
PHYS_BYTES = int(os.environ.get("STUB_PHYS_BYTES", "31000000000"))
RSS_BYTES = int(os.environ.get("STUB_RSS_BYTES", "29000000000"))
MEM_SUPPRESS = os.environ.get("STUB_MEM_SUPPRESS", "") == "1"
MEM_ZERO = os.environ.get("STUB_MEM_ZERO", "") == "1"
LOG_PATH = os.environ.get("STUB_LOG", "")
SPECULATIVE = os.environ.get("STUB_SPECULATIVE", "") == "1"
FORCE_DONE = os.environ.get("STUB_FORCE_DONE", "") == "1"
CHARGED = json.loads(os.environ.get("STUB_CHARGED", "false"))
DROP_FIELDS = [f for f in os.environ.get("STUB_DROP_FIELDS", "").split(",") if f]
BODIES = os.environ.get("STUB_BODIES", "")
BOUND_FLAG = os.environ.get("STUB_BOUND_FLAG", "")

served = 0
greedy_served = 0
FLIGHT = threading.Condition()
in_flight = 0
metrics_in_flight = 0


def pick(values, index):
    return values[min(index, len(values) - 1)]


def gap_ms():
    return pick(pick(GAP_MS, PASS - 1), served)


def event(fields):
    return json.dumps(
        {"timestamp": "2026-09-06T00:00:00Z", "level": "INFO", "fields": fields}
    )


def done_line(rate):
    rounds = 30
    seed_emitted = 1
    emitted = TOKENS
    in_rounds = emitted - seed_emitted
    total_draft = 150
    total_accept = 98
    draft_ms, verify_ms, round_ms = 300.0, 900.0, 1800.0
    fields = {
        "message": "mtp_generate: done",
        "rounds": rounds,
        "emitted": emitted,
        "seed_emitted": seed_emitted,
        "emitted_in_rounds": in_rounds,
        "total_draft": total_draft,
        "total_accept": total_accept,
        "accept_rate": total_accept / total_draft,
        "accepted_per_step": total_accept / rounds,
        "tokens_per_round": in_rounds / rounds,
        "elapsed_ms": 2560.0,
        "round_ms": round_ms,
        "draft_ms": draft_ms,
        "verifier_ms": verify_ms,
        "draft_ms_per_round": draft_ms / rounds,
        "verify_ms_per_round": verify_ms / rounds,
        "loop_ms_per_round": (round_ms - draft_ms - verify_ms) / rounds,
        "block_size": 5,
        "decode_config": os.environ.get("STUB_DECODE_CONFIG", "mtp/block=5"),
        "charged": CHARGED,
        "decode_tps": f"Some({rate})",
    }
    for name in DROP_FIELDS:
        fields.pop(name, None)
    return event(fields)


def prompt_tokens_of(request):
    """What this server says the request's prompt counted.

    With STUB_TOKEN_BASE it is a function of the body — which is what makes the
    fit a search rather than a fixed answer. Without it, the fixed figure the
    other cases were written against.
    """
    content = "".join(m.get("content", "") for m in request.get("messages", []))
    if LONG_PROMPT_TOKENS and len(content) > LONG_CHARS:
        return LONG_PROMPT_TOKENS
    if TOKEN_BASE:
        return int(TOKEN_BASE) + len(content) // 4
    return PROMPT_TOKENS


def is_long(request):
    content = "".join(m.get("content", "") for m in request.get("messages", []))
    return len(content) > LONG_CHARS


def log(line):
    if not LOG_PATH:
        return
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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

    def _text(self, body):
        raw = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        global metrics_in_flight
        if self.path == "/metrics":
            with FLIGHT:
                flying = in_flight > 0
                if flying:
                    metrics_in_flight += 1
                    FLIGHT.notify_all()
            if flying and ARGS.inflight_log:
                with open(ARGS.inflight_log, "a", encoding="utf-8") as handle:
                    handle.write("in-flight reading\n")
            if MEM_SUPPRESS:
                self._text("rmlx_uptime_seconds 1\n")
                return
            phys, rss = (PHYS_BYTES, RSS_BYTES) if flying else (PHYS_BYTES // 2, RSS_BYTES // 2)
            if MEM_ZERO:
                phys, rss = 0, 0
            self._text(
                f"rmlx_process_rss_bytes {rss}\n"
                f"rmlx_process_phys_footprint_bytes {phys}\n"
            )
            return
        self._json({"object": "list", "data": []})

    def do_POST(self):
        global in_flight
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if BODIES:
            with open(BODIES, "ab") as handle:
                handle.write(body + b"\n")
        request = json.loads(body or b"{}")

        # A fixed-prompt fit probe: it asks the server for a token count and
        # nothing else, so it streams nothing and leaves no event behind. A
        # probe that logged would be counted as a served request by the reader.
        if request.get("logprobs") and not request.get("stream"):
            self._greedy(request)
            return
        if not request.get("stream"):
            self._json({
                "choices": [{"message": {"role": "assistant", "content": "."}}],
                "usage": {"prompt_tokens": prompt_tokens_of(request),
                          "completion_tokens": 1},
            })
            return

        with FLIGHT:
            in_flight += 1
            readings_before = metrics_in_flight
        try:
            self._stream(request, readings_before)
        finally:
            with FLIGHT:
                in_flight -= 1

    def _greedy(self, request):
        """A greedy-pass answer: the same text for the same prompt on either arm."""
        global greedy_served
        index = greedy_served
        greedy_served += 1
        content = "".join(m.get("content", "") for m in request.get("messages", []))
        text = "answer " + hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        if SPECULATIVE and index in GREEDY_MARKS:
            text += " " + GREEDY_MARKS[index]
        tokens = len(text.split())
        if not GREEDY_NO_EVENT:
            prompt = prompt_tokens_of(request)
            reused = prompt if (GREEDY_CACHE_HIT and CACHE_SLOTS != "0") else GREEDY_REUSED
            log(event({"message": "generate: prefill", "arch": "Stub",
                       "prompt_tokens": prompt, "reused_tokens": reused,
                       "forwarded_tokens": max(prompt - reused, 0),
                       "prefill_ms": 0.0 if reused >= prompt else 5.0,
                       "draft_prefill_ms": 0.0}))
        record = {"token": "t", "logprob": -0.1,
                  "top_logprobs": [{"token": "t", "logprob": -0.1},
                                   {"token": "u", "logprob": -3.0}]}
        self._json({
            "choices": [{
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "logprobs": {"content": [record] * tokens},
            }],
            "usage": {"prompt_tokens": prompt_tokens_of(request), "completion_tokens": tokens},
        })

    def _stream(self, request, readings_before):
        global served
        this_gap_ms = gap_ms()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(payload, sent_at=None):
            if sent_at is not None:
                self.wfile.write(f": sent-at={sent_at}\n".encode())
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        if ARGS.marks:
            seen = os.path.join(ARGS.marks, "seen")
            number = 1
            if os.path.exists(seen):
                with open(seen, encoding="utf-8") as handle:
                    number += len(handle.readlines())
            up = os.path.exists(os.path.join(ARGS.marks, f"mark.{number}"))
            with open(seen, "a", encoding="utf-8") as handle:
                handle.write("up\n" if up else "down\n")
        chunk({"choices": [{"delta": {"role": "assistant"}, "index": 0}]}, 0.0)
        if ARGS.hold:
            open(ARGS.hold + ".held", "w").close()
            # A poll for the file the case creates; the limit ends a stub whose
            # case has died.
            deadline = time.monotonic() + ARGS.hold_wait
            while not os.path.exists(ARGS.hold + ".release") and time.monotonic() < deadline:
                time.sleep(0.05)
        if not SPECULATIVE and served == 1:
            with FLIGHT:
                FLIGHT.wait_for(lambda: metrics_in_flight > readings_before,
                                timeout=ARGS.sample_wait)
        for i in range(TOKENS):
            chunk({"choices": [{"delta": {"content": f"t{i} "}, "index": 0}]},
                  PREFILL_S + i * this_gap_ms / 1000.0)
        usage = {}
        if not OMIT_COMPLETION_TOKENS:
            count = USAGE_TOKENS if USAGE_TOKENS >= 0 else TOKENS
            if USAGE_TOKENS_PASS2 and PASS >= 2:
                count = int(USAGE_TOKENS_PASS2)
            usage["completion_tokens"] = count
        if PROMPT_TOKENS >= 0:
            usage["prompt_tokens"] = prompt_tokens_of(request)
        chunk({"choices": [], "usage": usage})

        # A request at temperature 0 resolves no sampler, as the engine's does not.
        if SAMPLED and request.get("temperature") != 0 and (SAMPLER_LINES < 0 or served < SAMPLER_LINES):
            top_k = SAMPLER_TOP_K
            if SAMPLER_TOP_K_PASS2 and PASS >= 2:
                top_k = int(SAMPLER_TOP_K_PASS2)
            if SAMPLER_TOP_K_AFTER and served >= 2:
                top_k = int(SAMPLER_TOP_K_AFTER)
            log(event({"message": "generate: host categorical sampler active",
                       "model_id": "stub", "temperature": 0.6, "top_p": 0.95,
                       "top_k": top_k, "min_p": 0.0, "seed": 42919}))
        if TTFT_LINES < 0 or served < TTFT_LINES:
            log(event({"message": "generate_streaming: TTFT",
                       "model_id": "stub", "ttft_ms": TTFT_MS}))
        if ITL_LINES < 0 or served < ITL_LINES:
            mean = this_gap_ms
            if ITL_MEAN_MS and served >= ITL_MEAN_FROM:
                mean = float(ITL_MEAN_MS)
            log(event({"message": "generate: ITL stats", "model_id": "stub",
                       "step_count": TOKENS, "p50_ms": mean, "p95_ms": mean,
                       "p99_ms": mean, "mean_ms": mean, "itl_spikes": 0}))
        if PREFILL_LINES < 0 or served < PREFILL_LINES:
            prompt = (
                int(PREFILL_PROMPT)
                if PREFILL_PROMPT and served >= PREFILL_PROMPT_FROM
                else prompt_tokens_of(request)
            )
            reused = prompt if REUSED_ALL else REUSED
            if is_long(request):
                reused = REUSED_LONG
            log(event({"message": "generate: prefill", "arch": "Stub",
                       "prompt_tokens": prompt, "reused_tokens": reused,
                       "forwarded_tokens": max(prompt - reused, 0),
                       "prefill_ms": pick(PREFILL_MS, PASS - 1),
                       "draft_prefill_ms": 0.0}))
        if not SPECULATIVE and (PEAK_LINES < 0 or served < PEAK_LINES):
            peak = int(PEAK_FIXED_MB) if PEAK_FIXED_MB else PEAK_BASE_MB + PEAK_PASS_OFFSET_MB.get(PASS, 0) + 100 * served
            log(event({"message": "generate: metal peak alloc",
                       "model_id": "stub", "metal_peak_alloc_mb": peak}))
        if SPECULATIVE or FORCE_DONE:
            log(done_line(round(1000.0 / this_gap_ms, 6)))
        served += 1

        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


server = ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler)
server.daemon_threads = True
if BOUND_FLAG:
    with open(BOUND_FLAG, "a", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
server.serve_forever()
PYEOF

# stop_stubs <bound-file> — stop every stub server a case started, by the PID
# each wrote when it bound. The harness stops its own servers; this keeps a
# harness defect from leaving a stub on the next case's port.
stop_stubs() {
    [ -f "$1" ] || return 0
    local pid
    while read -r pid; do
        kill "${pid}" 2>/dev/null || true
    done <"$1"
}

# ── Stub binary ───────────────────────────────────────────────────────────────

STUB="${FAKE_ROOT}/target/release-perf/rmlx"
cat >"${STUB}" <<STUBEOF
#!/usr/bin/env bash
set -eu
# The log-message literals \`lib/binary_identity.py\` requires a binary to
# contain before a run is started against it. A real rmlx carries them because
# it writes them; this stub carries them because it fakes writing them, and a
# case below removes one to show the check is not decorative.
# markers: generate_streaming: TTFT | generate: ITL stats
# markers: generate: host categorical sampler active | cache-type resolved
# markers: mtp_generate: done
# markers: generate: metal peak alloc
# markers: generate: prefill
case "\$1" in
metrics)
	case "\$2" in
	identity)
		echo '{"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb"}'
		;;
	esac
	;;
serve)
	printf '%s\n' "\$*" >>"\$RMLX_HOME/serve_args.log"
	port=8090
	speculative=0
	cache_slots=4
	while [ \$# -gt 0 ]; do
		case "\$1" in
		--port) port="\$2" ;;
		--draft-model) speculative=1 ;;
		--prompt-cache-slots) cache_slots="\$2" ;;
		esac
		shift
	done
	# Which pass this is. The harness starts one server per pass and the stub
	# has to vary its rate across them, so the count lives beside the logs.
	# On the plain arm the FIRST server is the preparation one the fixed-length
	# prompt is fitted against — it measures nothing, so it is pass 0 and the
	# per-pass fixtures still line up with the passes they were written for.
	started=\$(( \$(cat "\$RMLX_HOME/pass.cnt" 2>/dev/null || echo 0) + 1 ))
	echo "\$started" >"\$RMLX_HOME/pass.cnt"
	pass=\$started
	[ "\$speculative" = 0 ] && pass=\$((started - 1))
	if [ -n "\${STUB_SHARED_LOG:-}" ]; then
		log="\$RMLX_HOME/logs/\$STUB_SHARED_LOG"
	else
		log="\$RMLX_HOME/logs/stub-\$\$-p\$pass.jsonl"
		: >"\$log"
	fi
	if [ -z "\${STUB_PID_SUPPRESS:-}" ]; then
		printf '%s\n' "{\"timestamp\":\"2026-09-06T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"rmlx start\",\"version\":\"9.9.9\",\"run_id\":\"stub\",\"pid\":\$\$}}" >>"\$log"
	fi
	if [ -z "\${STUB_KV_QUANT_SUPPRESS:-}" ]; then
		kv="\${STUB_KV_QUANT:-mixed_k8g64_v4g64}"
		if [ -n "\${STUB_KV_QUANT_PASS2:-}" ] && [ "\$pass" -ge 2 ]; then
			kv="\$STUB_KV_QUANT_PASS2"
		fi
		printf '%s\n' "{\"timestamp\":\"2026-09-06T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"cache-type resolved\",\"arch\":\"Stub\",\"kv_quant\":\"\$kv\"}}" >>"\$log"
	fi
	export STUB_LOG="\$log" STUB_SPECULATIVE="\$speculative" STUB_PASS="\$pass" STUB_CACHE_SLOTS="\$cache_slots"
	# The case's own arguments for the stub server, one line, if it left any.
	exec python3 "${SERVER_PY}" "\$port" \$(cat "\$RMLX_HOME/stub_args" 2>/dev/null)
	;;
eval)
	# eval spec-answer --plain P --spec S. The verdict is staged by a word in the
	# speculative answer, and its exit code is the verdict's other spelling.
	spec="\$6"
	if grep -q BADEXIT "\$spec"; then echo verdict=near_tie; echo position=3; exit 1; fi
	if grep -q NOVERDICT "\$spec"; then exit 0; fi
	if grep -q NEAR "\$spec"; then
		printf 'verdict=near_tie\nposition=3\nmargin=0.0100\nrank=0.0200\nreason=a staged near tie\n'
		exit 0
	fi
	if grep -q UNJUDGE "\$spec"; then
		printf 'verdict=unjudgeable\nreason=a staged pair the oracle cannot read\n'
		exit 2
	fi
	printf 'verdict=refused\nposition=2\nrank=0.9000\nreason=a staged divergence\n'
	exit 1
	;;
esac
STUBEOF
chmod +x "${STUB}"
ln -s "${STUB}" "${CANON_ROOT}/target/release-perf/rmlx"

# A third root whose binary is the same stub with one marker literal removed. A
# binary that cannot write the event a reading is read off cannot produce that
# reading, and the run must stop before it starts rather than after three passes
# produced nothing.
MARKERLESS_ROOT="${WORK}/repo_markerless"
mkdir -p "${MARKERLESS_ROOT}/target/release-perf"
ln -s "${REPO_ROOT}/scripts" "${MARKERLESS_ROOT}/scripts"
ln -s "${REPO_ROOT}/prompts" "${MARKERLESS_ROOT}/prompts"
grep -v 'generate: ITL stats' "${STUB}" >"${MARKERLESS_ROOT}/target/release-perf/rmlx"
chmod +x "${MARKERLESS_ROOT}/target/release-perf/rmlx"

# The same, losing the Metal peak literal instead.
NOPEAK_ROOT="${WORK}/repo_nopeak"
mkdir -p "${NOPEAK_ROOT}/target/release-perf"
ln -s "${REPO_ROOT}/scripts" "${NOPEAK_ROOT}/scripts"
ln -s "${REPO_ROOT}/prompts" "${NOPEAK_ROOT}/prompts"
grep -v 'generate: metal peak alloc' "${STUB}" >"${NOPEAK_ROOT}/target/release-perf/rmlx"
chmod +x "${NOPEAK_ROOT}/target/release-perf/rmlx"

# And losing the prefill literal.
NOPREFILL_ROOT="${WORK}/repo_noprefill"
mkdir -p "${NOPREFILL_ROOT}/target/release-perf"
ln -s "${REPO_ROOT}/scripts" "${NOPREFILL_ROOT}/scripts"
ln -s "${REPO_ROOT}/prompts" "${NOPREFILL_ROOT}/prompts"
grep -v 'generate: prefill' "${STUB}" >"${NOPREFILL_ROOT}/target/release-perf/rmlx"
chmod +x "${NOPREFILL_ROOT}/target/release-perf/rmlx"

# A port this host is not already using. Probed rather than assumed: a foreign
# listener would answer the readiness poll and the suite would measure it.
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
')" || exit 2

# ── Case driver ───────────────────────────────────────────────────────────────

# run_case <name> <want-exit> <what-it-proves> [KEY=VALUE ...] [ARGS:flag ...]
#          [HOST:quiet|hostile|flaky] [THERM:nominal|throttled|unreadable]
#          [VERIFIER:dir] [MEASURED:1] [CANONICAL:1] [MARKERLESS:itl|peak|prefill]
#          [LABEL:name | NOLABEL:1] [FIXED_TS:1] [PREDIR:path-under-home ...]
#          [CLIENT_SHIM:N] [NO_STAMPS:1] [STUB:<stub server arguments>]
#          [GREP:pat ...] [NOGREP:pat ...]
#
# <want-exit> `-` accepts any exit code.
#
# Defaults: a quiet host, --synthetic-arms, the healthy verifier snapshot, and
# the shrunken sample sets reached through --samples-root — the unverified
# operator override, which is the only way a root that is not the published one
# can be measured at all. CANONICAL:1 drops the flag and runs out of a repo root
# whose prompts/published IS the shrunken copy, which is how the default path is
# shown to verify.
#
# The stub streams six tokens 40 ms apart, stamped behind a 150 ms prefill — 25
# tok/s on the wire and 25 reported by the engine, at any token count.
# Its prompt count is four characters per token above a base of 100, so the
# fixed-length prompt's fit is a real search over the corpus rather than a
# fixed answer handed back.
run_case() {
    CASE_NAME="$1"
    local want="$2"
    CASE_WHAT="$3"
    shift 3

    CASE_HOME="${WORK}/home_${CASE_NAME}"
    mkdir -p "${CASE_HOME}/logs"

    local env_pairs=() greps=() nogreps=() extra_args=() host="quiet" a
    local synthetic=true verifier="${VERIFIER}" canonical=false
    local therm="${SHIMS}" markerless="" delayed=0 stamps=true stub_args="" prelog=""
    local label="selftest-campaign" fixed_ts="" predirs=()
    for a in "$@"; do
        case "$a" in
        LABEL:*) label="${a#LABEL:}" ;;
        NOLABEL:*) label="" ;;
        FIXED_TS:*) fixed_ts="${WORK}/fixed_ts:" ;;
        PREDIR:*) predirs+=("${a#PREDIR:}") ;;
        GREP:*) greps+=("${a#GREP:}") ;;
        NOGREP:*) nogreps+=("${a#NOGREP:}") ;;
        ARGS:*) extra_args+=("${a#ARGS:}") ;;
        HOST:*) host="${a#HOST:}" ;;
        MEASURED:*) synthetic=false ;;
        CANONICAL:*) canonical=true ;;
        MARKERLESS:*) markerless="${a#MARKERLESS:}" ;;
        CLIENT_SHIM:*) delayed="${a#CLIENT_SHIM:}" ;;
        NO_STAMPS:*) stamps=false ;;
        STUB:*) stub_args="${stub_args} ${a#STUB:}" ;;
        PRELOG:*) prelog="${a#PRELOG:}" ;;
        THERM:*) therm="${WORK}/therm_${a#THERM:}" ;;
        VERIFIER:*) verifier="${a#VERIFIER:}" ;;
        *) env_pairs+=("$a") ;;
        esac
    done
    $synthetic && extra_args+=(--synthetic-arms)
    $stamps && extra_args+=(--server-stamps)
    [ -n "${label}" ] && extra_args+=(--label "${label}")
    for a in ${predirs[@]+"${predirs[@]}"}; do mkdir -p "${CASE_HOME}/${a}"; done
    write_client_shim "${CASE_HOME}/client" "${delayed}"
    printf '%s\n' "--marks ${CASE_HOME}/client${stub_args}" >"${CASE_HOME}/stub_args"
    # A run log an earlier rmlx process left in this second: every server of the
    # case appends to it, and none of its events belong to the run.
    if [ -n "${prelog}" ]; then
        cat >"${CASE_HOME}/logs/${prelog}" <<'LOGEOF'
{"timestamp":"2026-09-06T00:00:00Z","level":"INFO","fields":{"message":"rmlx start","version":"9.9.9","run_id":"stub","pid":4242}}
{"timestamp":"2026-09-06T00:00:00Z","level":"INFO","fields":{"message":"cache-type resolved","arch":"Stub","kv_quant":"none"}}
{"timestamp":"2026-09-06T00:00:00Z","level":"INFO","fields":{"message":"generate_streaming: TTFT","model_id":"stub","ttft_ms":999.0}}
{"timestamp":"2026-09-06T00:00:00Z","level":"INFO","fields":{"message":"generate: ITL stats","model_id":"stub","step_count":8,"p50_ms":99.0,"p95_ms":99.0,"p99_ms":99.0,"mean_ms":99.0,"itl_spikes":0}}
LOGEOF
        env_pairs+=("STUB_SHARED_LOG=${prelog}")
    fi
    local root="${FAKE_ROOT}"
    if $canonical; then
        root="${CANON_ROOT}"
    else
        case "${markerless}" in
        itl) root="${MARKERLESS_ROOT}" ;;
        peak) root="${NOPEAK_ROOT}" ;;
        prefill) root="${NOPREFILL_ROOT}" ;;
        esac
        extra_args+=(--samples-root "${SAMPLES}")
    fi

    CASE_OUT="${WORK}/${CASE_NAME}.log"
    local got=0
    env -i \
        PATH="${CASE_HOME}/client/bin:${fixed_ts}${WORK}/${host}:${therm}:${SHIMS}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
        HOME="${WORK}" \
        RMLX_HOME="${CASE_HOME}" \
        STUB_BOUND_FLAG="${CASE_HOME}/stub_bound" \
        STUB_BODIES="${CASE_HOME}/bodies.jsonl" \
        STUB_TOKEN_BASE=100 \
        ${env_pairs[@]+"${env_pairs[@]}"} \
        bash "${root}/scripts/spec_bench_published.sh" \
        "${verifier}" --port "${PORT}" \
        ${extra_args[@]+"${extra_args[@]}"} >"${CASE_OUT}" 2>&1
    got=$?
    stop_stubs "${CASE_HOME}/stub_bound"

    CASE_BAD=""
    [ "${want}" != "-" ] && [ "${got}" -ne "${want}" ] && CASE_BAD="exit=${got} (want ${want})"
    if grep -q '\[server\] pid=' "${CASE_OUT}"; then
        [ -s "${CASE_HOME}/stub_bound" ] ||
            note_bad "the stub never bound port ${PORT} — something else answered"
    fi
    local g
    for g in ${greps[@]+"${greps[@]}"}; do
        grep -qE "$g" "${CASE_OUT}" || note_bad "missing /$g/"
    done
    for g in ${nogreps[@]+"${nogreps[@]}"}; do
        grep -qE "$g" "${CASE_OUT}" && note_bad "unexpected /$g/"
    done
}

note_bad() {
    if [ -z "${CASE_BAD}" ]; then CASE_BAD="$1"; else CASE_BAD="${CASE_BAD}; $1"; fi
}

verdict() {
    if [ -z "${CASE_BAD}" ]; then
        printf 'ok    %-42s %s\n' "${CASE_NAME}" "${CASE_WHAT}"
        PASSED=$((PASSED + 1))
    else
        printf 'FAIL  %-42s %s\n' "${CASE_NAME}" "${CASE_WHAT}"
        printf '        %s\n' "${CASE_BAD}"
        tail -4 "${CASE_OUT}" | sed 's/^/        | /'
        FAILED=$((FAILED + 1))
    fi
}

# The result file this case wrote, or the empty string.
result_of() {
    local hits
    hits=$(echo "${CASE_HOME}"/bench/published/*/*/result.json)
    case "${hits}" in
    *'*'*) echo "" ;;
    *) echo "${hits%% *}" ;;
    esac
}

# jq_of <python-expression over `r`> — read one value out of the result file.
jq_of() {
    local path
    path="$(result_of)"
    [ -z "${path}" ] && return 0
    python3 -c 'import json, sys
r = json.load(open(sys.argv[1]))
print(eval(sys.argv[2]))' "${path}" "$1" 2>/dev/null
}

no_result() {
    [ -n "$(result_of)" ] && note_bad "a result file was written anyway"
    return 0
}

close_to() {
    python3 -c 'import sys
got, want, tol = sys.argv[1:4]
try:
    g, w = float(got), float(want)
except ValueError:
    sys.exit(1)
sys.exit(0 if abs(g - w) <= float(tol) * abs(w) else 1)' "$1" "$2" "$3"
}

echo "spec_bench_published_selftest: stub server on 127.0.0.1:${PORT}"
echo "  ${REQUESTS_PER_PASS} requests per pass over four cells"
echo
# ── The measured number ───────────────────────────────────────────────────────

# The wire carries 25 tok/s and the engine reports 25. Every cell's mean is that
# rate; anything else is the harness deriving a number of its own.
run_case mean_is_the_engine_reading 0 \
    "every cell's mean is the rate the engine reported"
for cell in mt_bench@1024 math_500@1024 humaneval@1024 math_500@4096; do
    got="$(jq_of "r['cells']['${cell}']['mean']")"
    close_to "${got}" 25.0 0.0001 || note_bad "${cell} mean=${got} (want 25.0)"
done
[ "$(jq_of "r['cells']['mt_bench@1024']['samples']")" = "1" ] ||
    note_bad "mt_bench@1024 counted $(jq_of "r['cells']['mt_bench@1024']['samples']") samples"
[ "$(jq_of "r['cells']['humaneval@1024']['samples']")" = "3" ] ||
    note_bad "humaneval@1024 counted $(jq_of "r['cells']['humaneval@1024']['samples']") samples"
verdict

# Three passes are what the protocol asks for and what the result has to hold.
run_case three_passes_of_every_sample 0 \
    "the result holds three passes of every sample and nothing else"
[ "$(jq_of "len(r['samples'])")" = "24" ] ||
    note_bad "the result holds $(jq_of "len(r['samples'])") rows (want 3 x 8)"
# Every request of the stub streams six tokens, plain arm or speculative; the
# round-loop fixtures are written against that count.
[ "$(jq_of "sorted({s['completion_tokens'] for s in r['samples']})")" = "[6]" ] ||
    note_bad "completion_tokens=$(jq_of "sorted({s['completion_tokens'] for s in r['samples']})") (want [6])"
[ "$(jq_of "sorted({s['pass'] for s in r['samples']})")" = "[1, 2, 3]" ] ||
    note_bad "passes=$(jq_of "sorted({s['pass'] for s in r['samples']})")"
verdict

# ── What the macro average is over ────────────────────────────────────────────
#
# Four wrong answers and one right one, all reachable from the same fixture, so
# this case cannot be satisfied by the weighting it is meant to catch:
#
#   cell means         mt_bench 5, math_500@1024 10, humaneval 20, math_500@4096 40
#   over four CELLS                    (5+10+20+40)/4      = 18.75
#   over three datasets, cells averaged (5 + (10+40)/2 + 20)/3 = 16.667
#   pooled over the 8 rows        (1*5+2*10+3*20+2*40)/8   = 20.625
#   OVER THREE DATASETS AT 1024        (5+10+20)/3         = 11.667  <- correct
#
# Request order is warmup, mt_bench, math_500@1024 x2, humaneval x3,
# math_500@4096 x2; gaps 200/100/50/25 ms give 5/10/20/40 tok/s.
run_case macro_is_over_the_datasets_at_the_headline_budget 0 \
    "the macro is one cell per dataset at 1024, not one per cell" \
    'STUB_GAP_MS=40,40,200,100,100,50,50,50,25,25' \
    'GREP:MACRO covers .* at 1024 output tokens'
close_to "$(jq_of "r['cells']['mt_bench@1024']['mean']")" 5.0 0.01 ||
    note_bad "mt_bench mean=$(jq_of "r['cells']['mt_bench@1024']['mean']") (want 5)"
close_to "$(jq_of "r['cells']['math_500@4096']['mean']")" 40.0 0.01 ||
    note_bad "math_500@4096 mean=$(jq_of "r['cells']['math_500@4096']['mean']") (want 40)"
got="$(jq_of "r['macro']['mean']")"
close_to "${got}" 11.6667 0.01 ||
    note_bad "macro=${got} (want 11.667; four cells give 18.75, datasets-with-both-budgets 16.667, pooled 20.625)"
[ "$(jq_of "sorted(r['macro']['cells'])")" = "['humaneval@1024', 'math_500@1024', 'mt_bench@1024']" ] ||
    note_bad "macro covers $(jq_of "sorted(r['macro']['cells'])")"
verdict

# ── The run-to-run range refusal ──────────────────────────────────────────────
#
# Both sides of the band on one fixture: the per-pass gaps put the three pass
# rates at 24.3875 / 25 / 25.6125 (range 4.9% of the mean) and at
# 24.36 / 25 / 25.64 (5.12%). A gate proven to fire but never proven to hold its
# tongue is a gate that could be firing on everything.
UNDER_BAND="$(python3 -c 'print(";".join(f"{1000/r:.9f}" for r in (24.3875, 25, 25.6125)))')"
OVER_BAND="$(python3 -c 'print(";".join(f"{1000/r:.9f}" for r in (24.36, 25, 25.64)))')"

run_case range_just_under_the_band_is_a_clean_mean 0 \
    "a spread just inside the band is published as a mean" \
    "STUB_GAP_MS=${UNDER_BAND}" \
    'NOGREP:RANGE REFUSAL' \
    'NOGREP:UNSTABLE'
close_to "$(jq_of "r['cells']['mt_bench@1024']['range_pct']")" 4.9 0.02 ||
    note_bad "range=$(jq_of "r['cells']['mt_bench@1024']['range_pct']") (want ~4.9)"
close_to "$(jq_of "r['cells']['mt_bench@1024']['mean']")" 25.0 0.001 ||
    note_bad "mean=$(jq_of "r['cells']['mt_bench@1024']['mean']")"
[ "$(jq_of "all(c['stable'] for c in r['cells'].values())")" = "True" ] ||
    note_bad "a cell was marked unstable"
verdict

run_case range_over_the_band_is_refused 3 \
    "a spread past the band is not printed as a mean" \
    "STUB_GAP_MS=${OVER_BAND}" \
    'GREP:RANGE REFUSAL' \
    'GREP:UNSTABLE' \
    'GREP:mt_bench@1024'
close_to "$(jq_of "r['cells']['mt_bench@1024']['range_pct']")" 5.12 0.02 ||
    note_bad "range=$(jq_of "r['cells']['mt_bench@1024']['range_pct']") (want ~5.12)"
[ "$(jq_of "r['cells']['mt_bench@1024']['stable']")" = "False" ] ||
    note_bad "the cell is marked stable at a range past the band"
# The mean is still in the file — withheld from the column a reader copies, not
# from the record a later pass reads.
close_to "$(jq_of "r['cells']['mt_bench@1024']['mean']")" 25.0 0.001 ||
    note_bad "mean=$(jq_of "r['cells']['mt_bench@1024']['mean']")"
verdict

# ONE cell destabilised, and only one. mt_bench is the single-sample dataset, so
# request index 2 is all of it — index 0 is the warmup and index 1 the
# fixed-length prompt — and 23.5 / 25 / 26.5 tok/s across the passes is a 12%
# range there that leaves every other cell flat.
#
# This is also the empirical half of why the macro has no refusal of its own.
# The macro's OWN range here is 4% — inside the band — while the dataset it is
# built from is refused at 12%. A macro range test would have passed this run
# and published the headline. Withholding it because a dataset was refused is
# what makes the MACRO row say UNSTABLE.
ONE_CELL="$(python3 -c 'g = lambda r: f"{1000/r:.9f}"
print(";".join(["40,40," + g(23.5) + ",40,40,40,40,40,40,40",
                "40",
                "40,40," + g(26.5) + ",40,40,40,40,40,40,40"]))')"

run_case one_unstable_dataset_withholds_only_it_and_the_macro 3 \
    "one refused dataset does not refuse the others, and does withhold the macro" \
    "STUB_GAP_MS=${ONE_CELL}" \
    'GREP:RANGE REFUSAL: mt_bench@1024 —' \
    'GREP:MACRO is withheld too'
[ "$(jq_of "r['cells']['mt_bench@1024']['stable']")" = "False" ] ||
    note_bad "the destabilised cell is marked stable"
for cell in math_500@1024 humaneval@1024 math_500@4096; do
    [ "$(jq_of "r['cells']['${cell}']['stable']")" = "True" ] ||
        note_bad "${cell} was refused along with mt_bench"
    close_to "$(jq_of "r['cells']['${cell}']['mean']")" 25.0 0.001 ||
        note_bad "${cell} mean=$(jq_of "r['cells']['${cell}']['mean']")"
done
[ "$(jq_of "r['macro']['stable']")" = "False" ] ||
    note_bad "the macro was published while a dataset it covers was refused"
got="$(jq_of "r['macro']['range_pct']")"
close_to "${got}" 4.0 0.05 ||
    note_bad "macro range=${got} (want ~4.0 — inside the band, which is the point)"
# The pass-mean range cannot see a sample that moves and moves back, so the
# widest single-sample spread is reported beside it — and is not a refusal.
got="$(jq_of "r['cells']['mt_bench@1024']['sample_range_pct_max']")"
close_to "${got}" 12.0 0.05 ||
    note_bad "sample_range_pct_max=${got} (want ~12)"
[ "$(jq_of "r['cells']['math_500@1024']['sample_range_pct_max']")" = "0.0" ] ||
    note_bad "a flat cell reported a spread"
verdict

# ── What is sent ──────────────────────────────────────────────────────────────

# bodies_check <python-expression over `bodies` and `digests`>
bodies_check() {
    python3 - "${CASE_HOME}/bodies.jsonl" "${SAMPLES}" "$1" "${REPO_ROOT}" <<'PY'
import hashlib, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(sys.argv[4]) / "scripts" / "lib"))
sys.path.insert(0, str(pathlib.Path(sys.argv[4]) / "scripts"))
from published_fixed_prompt import INSTRUCTION

bodies = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
root = pathlib.Path(sys.argv[2])
digests = set()
for entry in json.loads((root / "manifest.json").read_text())["datasets"]:
    for sample in json.loads((root / entry["file"]).read_text())["samples"]:
        digests.add(sample["body_sha256"])


def address(messages):
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# Four kinds of request reach the server and only one of them is a measured
# sample. A fit probe streams nothing; the warmup carries its own budget; the
# fixed-length prompt carries the instruction that module pins, imported rather
# than spelled again here.
probes = [b for b in bodies if not b.get("stream")]
streamed = [b for b in bodies if b.get("stream")]
warmups = [b for b in streamed if b["max_tokens"] == 64]
fixed = [
    b
    for b in streamed
    if b["max_tokens"] != 64 and b["messages"][0]["content"].startswith(INSTRUCTION)
]
measured = [b for b in streamed if b not in warmups and b not in fixed]

print(eval(sys.argv[3]))
PY
}

# The published sampling is the checkpoint's own, which is what sending no
# sampling field at all means. A temperature spelled here would be a second
# copy of a fact the snapshot owns, and the one that got published.
run_case sampling_fields_are_not_sent 0 \
    "no request carries a sampling parameter of this script's"
for field in temperature top_p top_k seed repetition_penalty; do
    [ "$(bodies_check "any('${field}' in b for b in bodies)")" = "False" ] ||
        note_bad "a request carried ${field}"
done
[ "$(bodies_check "all(b['enable_thinking'] is True for b in bodies)")" = "True" ] ||
    note_bad "a request did not ask for thinking"
# The warmup is a 64-token request on a prompt outside the sample sets and a
# fixed-prompt fit probe asks for one token, so the budgets sent are those two
# plus the two the protocol pins.
[ "$(bodies_check "sorted({b['max_tokens'] for b in bodies})")" = "[1, 64, 1024, 4096]" ] ||
    note_bad "max_tokens sent: $(bodies_check "sorted({b['max_tokens'] for b in bodies})")"
[ "$(bodies_check "all(b['max_tokens'] == 1 for b in probes)")" = "True" ] ||
    note_bad "a fit probe asked for more than the one token it needs to be counted"
verdict

# What the run sampled under is the engine's to say. Re-reading the checkpoint
# file would print what should have happened, which is a different claim.
run_case sampling_is_read_back_from_the_engine 0 \
    "the published sampling is the one the engine says it resolved" \
    'STUB_SAMPLER_TOP_K=20' \
    'GREP:sampling: .*"top_k": 20'
[ "$(jq_of "r['protocol']['sampling_resolved']['top_k']")" = "20" ] ||
    note_bad "top_k=$(jq_of "r['protocol']['sampling_resolved']['top_k']")"
[ "$(jq_of "r['protocol']['sampling_resolved']['temperature']")" = "0.6" ] ||
    note_bad "temperature=$(jq_of "r['protocol']['sampling_resolved']['temperature']")"
# The request sends no seed, so the engine substitutes one and three passes
# replay a single RNG stream. That belongs in the record, not in a footnote.
[ "$(jq_of "r['protocol']['sampling_resolved']['seed']")" = "42919" ] ||
    note_bad "seed=$(jq_of "r['protocol']['sampling_resolved']['seed']")"
verdict

# A busy host was seen stretching the client's reading of a decode window past
# the engine's, on the fixed-length-prompt request specifically — the request
# sent between the warmups and the cells, before a plain run has read back its
# own sampling. `STUB_ITL_MEAN_FROM=0` stages that same disagreement without a
# busy host: this pins that a fixed-prompt cross-check failure aborts the run
# before sampling is ever read, which is why that field came back empty on the
# host that hit it rather than the checkpoint being unreadable.
run_case fixed_prompt_disagreement_precedes_sampling_read 1 \
    "a fixed-prompt cross-check failure is the fixed-prompt band, not a lost sampling read" \
    'STUB_ITL_MEAN_MS=20' 'STUB_ITL_MEAN_FROM=0' \
    'GREP:published_fixed_run: pass 1: .*past the 10% band' \
    'GREP:over the fixed prompt.s decode window' \
    'NOGREP:sampling: ' \
    'NOGREP:published_aggregate'
no_result
verdict

# The read this run's sampling comes back on is the same request whose body
# already closed: do_POST writes the sampler event, then TTFT and ITL, then
# `served += 1`, then the final SSE chunk — all before returning, so nothing
# downstream of the response can observe the log mid-write. The harness never
# reads that event until well after the request stream is done, either: it
# kills the server, `wait`s for the process to exit, sleeps another 3 s to let
# any buffered write flush, and only then reads the log. A read reachable only
# behind a confirmed process exit cannot race a write that happened before
# that process's own last response, for any delay on the write — the ordering
# holds by construction, not by a margin a runtime case could lose.

# Two guards, two cases: one pass whose requests did not share a setting, and
# three passes that did not share one with each other.
run_case sampler_drift_within_a_pass_refused 1 \
    "requests of one pass that sampled under different settings are refused" \
    'STUB_SAMPLER_TOP_K=20' 'STUB_SAMPLER_TOP_K_AFTER=40' \
    'GREP:did not share one sampling setup' \
    'GREP:2 values for top_k'
no_result
verdict

run_case sampler_drift_between_passes_refused 1 \
    "passes that sampled under different settings are not averaged" \
    'STUB_SAMPLER_TOP_K=20' 'STUB_SAMPLER_TOP_K_PASS2=40' \
    'GREP:not repetitions of one measurement'
no_result
verdict

run_case unreported_sampling_refused 1 \
    "a run whose sampling cannot be read back files nothing" \
    'STUB_SAMPLER_LINES=0' \
    'GREP:holds no sampler event'
no_result
verdict

# The other arm of that branch: a greedy checkpoint resolves no sampler, the
# engine writes no such event, and the harness must expect none rather than
# refuse a run for lacking one.
run_case greedy_checkpoint_resolves_no_sampler 0 \
    "a checkpoint whose own default is greedy is measured, not refused" \
    "VERIFIER:${GREEDY}" 'STUB_SAMPLED=0'
[ "$(jq_of "r['protocol']['sampling_resolved']")" = "None" ] ||
    note_bad "sampling_resolved=$(jq_of "r['protocol']['sampling_resolved']")"
verdict

run_case sampler_event_on_a_greedy_run_refused 1 \
    "a greedy run whose engine resolved a sampler anyway is refused" \
    "VERIFIER:${GREEDY}" 'STUB_SAMPLED=1' \
    'GREP:resolved a sampler this run was not supposed to have'
no_result
verdict

# The workspace builds serde_json with preserve_order, so a message re-emitted
# as {content, role} has a different content address than the {role, content}
# that was checked in — and a later join on it splits with nothing saying so.
run_case messages_keep_their_content_address 0 \
    "every measured request's messages still hash to the checked-in address"
# `measured` excludes the warmup, the fixed-length prompt and the fit probes:
# none of the three is a sample, and each is meant to be in no sample set. That
# the warmup's prompt is outside them is the point of it — an untimed request on
# a measured sample would leave that one sample facing a warm prompt cache.
[ "$(bodies_check "all(address(b['messages']) in digests for b in measured)")" = "True" ] ||
    note_bad "a measured request's messages no longer hash to any checked-in body_sha256"
[ "$(bodies_check "all(list(m) == ['role', 'content'] for b in bodies for m in b['messages'])")" = "True" ] ||
    note_bad "a message was re-emitted with its keys in another order"
[ "$(bodies_check "any(address(b['messages']) in digests for b in warmups)")" = "False" ] ||
    note_bad "the warmup re-used a measured sample's prompt"
[ "$(bodies_check "len(warmups)")" = "3" ] ||
    note_bad "$(bodies_check "len(warmups)") warmups for three passes"
# The fixed-length prompt is sent once per pass and is in no sample set either.
[ "$(bodies_check "len(fixed)")" = "3" ] ||
    note_bad "$(bodies_check "len(fixed)") fixed-prompt requests for three passes"
[ "$(bodies_check "any(address(b['messages']) in digests for b in fixed)")" = "False" ] ||
    note_bad "the fixed prompt re-used a measured sample's prompt"
verdict

# Thinking tokens are generated tokens and the server counts them; the harness
# records the count the server reported rather than the chunks it saw.
run_case completion_tokens_are_the_servers_count 0 \
    "the recorded output length is the count the server reported" \
    'STUB_USAGE_TOKENS=97'
[ "$(jq_of "sorted({s['completion_tokens'] for s in r['samples']})")" = "[97]" ] ||
    note_bad "completion_tokens=$(jq_of "sorted({s['completion_tokens'] for s in r['samples']})")"
# ...and prompt_tokens is the server's own count of that sample's body, which
# the stub derives from the body, so a harness substituting a constant is seen.
WANT_PROMPT_TOKENS="$(python3 - "${SAMPLES}" <<'TOKENS'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
counts = set()
for entry in json.loads((root / "manifest.json").read_text())["datasets"]:
    for sample in json.loads((root / entry["file"]).read_text())["samples"]:
        content = "".join(m["content"] for m in sample["messages"])
        counts.add(100 + len(content) // 4)
print(sorted(counts))
TOKENS
)"
[ "$(jq_of "sorted({s['prompt_tokens'] for s in r['samples']})")" = "${WANT_PROMPT_TOKENS}" ] ||
    note_bad "prompt_tokens=$(jq_of "sorted({s['prompt_tokens'] for s in r['samples']})") (want ${WANT_PROMPT_TOKENS})"
verdict

run_case missing_prompt_tokens_refused 1 \
    "a response with no usage.prompt_tokens files no row" \
    'STUB_PROMPT_TOKENS=-1' \
    'GREP:carries no prompt_tokens'
no_result
verdict

# The client-side reader falls back to a count of content chunks when the usage
# chunk carried no completion count, and content chunks miss every token whose
# visible piece is empty. Publishing that under an engine field's name reads low
# with nothing saying so, so the row is refused instead. The only server in play
# builds one Usage struct carrying both counts, so this input is not reachable
# from it today — which is exactly why it is staged here rather than left to the
# prompt_tokens requirement to catch by accident.
run_case usage_without_completion_tokens_refused 1 \
    "a count of content chunks is not filed as the engine's completion count" \
    'STUB_OMIT_COMPLETION_TOKENS=1' \
    'GREP:carries no completion_tokens'
no_result
verdict

# TTFT is the engine's, off its own event, per request.
run_case ttft_is_the_engines_reading 0 \
    "each row carries the TTFT the engine reported" \
    'STUB_TTFT_MS=137'
close_to "$(jq_of "r['cells']['mt_bench@1024']['ttft_ms_mean']")" 137.0 0.0001 ||
    note_bad "ttft=$(jq_of "r['cells']['mt_bench@1024']['ttft_ms_mean']")"
verdict
# ── Attribution ───────────────────────────────────────────────────────────────

# A log holding fewer events than requests served does not line up with the runs
# being measured, and the events that survive belong to other samples.
run_case truncated_ttft_log_refused 1 \
    "a log missing a request's TTFT event is refused" \
    'STUB_TTFT_LINES=4' \
    "GREP:TTFT events, expected ${EVENTS_PER_PASS}"
no_result
verdict

run_case missing_itl_events_refused 1 \
    "a plain run whose rate the engine never reported is refused" \
    'STUB_ITL_LINES=0' \
    'GREP:holds no ITL stats event'
no_result
verdict

run_case truncated_itl_log_refused 1 \
    "a plain log missing a request's ITL event is refused" \
    'STUB_ITL_LINES=5' \
    "GREP:ITL stats events, expected ${EVENTS_PER_PASS}"
no_result
verdict

# A round-loop record on a run that was given no drafter means the log belongs
# to a different server, and every number taken from it is somebody else's.
run_case round_loop_events_on_a_plain_run_refused 1 \
    "a plain run whose log holds speculative records is refused" \
    'STUB_FORCE_DONE=1' \
    'GREP:given no drafter'
no_result
verdict

run_case unattributable_log_refused 1 \
    "a run log with no pid is not read as this pass's" \
    'STUB_PID_SUPPRESS=1' \
    'GREP:is attributable to the pass-1 server'
no_result
verdict

run_case same_second_log_attributed 0 \
    "passes that append to a log an earlier process began read only their own events" \
    'PRELOG:20260906-000000-9.9.9.jsonl'
verdict

run_case unreported_kv_quant_refused 1 \
    "a pass that never said which codec it used is not recorded" \
    'STUB_KV_QUANT_SUPPRESS=1' \
    'GREP:never said which KV codec'
no_result
verdict

# Three passes that resolved different codecs are three different measurements,
# and their mean belongs to no cell.
run_case kv_quant_drift_between_passes_refused 1 \
    "passes that resolved different codecs are not averaged" \
    'STUB_KV_QUANT=k8v8' 'STUB_KV_QUANT_PASS2=k8v4' \
    'GREP:not repetitions of one' \
    'GREP:resolved k8v4 where an earlier pass resolved k8v8'
no_result
verdict

run_case kv_quant_is_the_resolved_codec 0 \
    "the result names the codec the run resolved" \
    'STUB_KV_QUANT=mixed_k8g64_v4g64'
[ "$(jq_of "r['kv_quant']")" = "mixed_k8g64_v4g64" ] ||
    note_bad "kv_quant=$(jq_of "r['kv_quant']")"
verdict

# Two readings of one window that disagree are a finding, not a choice. The stub
# reports half the gap it sent, so the engine claims twice the rate.
run_case cross_check_refuses_disagreement 1 \
    "an engine and a client reading that disagree stop the run" \
    'STUB_ITL_MEAN_MS=20' 'STUB_ITL_MEAN_FROM=2' \
    'GREP:past the 10% band' \
    'GREP:mt_bench@1024/mt_bench'
no_result
verdict

# ── The speculative arm ───────────────────────────────────────────────────────

run_case speculative_rows_carry_the_round_figures 0 \
    "a speculative cell carries the per-round split, not only its rate" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" ARGS:--draft-kind ARGS:mtp
[ "$(jq_of "r['arm']")" = "speculative" ] || note_bad "arm=$(jq_of "r['arm']")"
[ "$(jq_of "r['decode_config']")" = "mtp/block=5" ] ||
    note_bad "decode_config=$(jq_of "r['decode_config']")"
# 30 rounds emitting six tokens of which one is the pre-round seed.
close_to "$(jq_of "r['cells']['mt_bench@1024']['tokens_per_round']")" 0.1666667 0.001 ||
    note_bad "tokens_per_round=$(jq_of "r['cells']['mt_bench@1024']['tokens_per_round']")"
close_to "$(jq_of "r['cells']['mt_bench@1024']['accepted_per_step']")" 3.2666667 0.001 ||
    note_bad "accepted_per_step=$(jq_of "r['cells']['mt_bench@1024']['accepted_per_step']")"
close_to "$(jq_of "r['cells']['mt_bench@1024']['mean']")" 25.0 0.001 ||
    note_bad "mean=$(jq_of "r['cells']['mt_bench@1024']['mean']")"
verdict

# The plain arm has no round loop, so it must carry no per-round figure: a zero
# there would read as a measured one.
run_case plain_rows_carry_no_round_figures 0 \
    "the plain arm reports no per-round figure at all"
for figure in tokens_per_round accepted_per_step accept_rate; do
    [ "$(jq_of "'${figure}' in r['cells']['mt_bench@1024']")" = "False" ] ||
        note_bad "the plain arm carries ${figure}"
done
[ "$(jq_of "'decode_config' in r")" = "False" ] ||
    note_bad "the plain arm named a decode_config"
verdict

# `charged` says the round loop drained its pipeline at every phase boundary, so
# its rate describes a slower, differently scheduled engine. It is reachable
# from an ambient RUST_LOG the harness's own filter does not clear.
run_case charged_run_refused 1 \
    "a charged speculative run is refused rather than published" \
    'STUB_CHARGED=true' \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    'GREP:charged'
no_result
verdict

run_case missing_decode_tps_refused 1 \
    "a done line with no decode_tps is refused, not read around" \
    'STUB_DROP_FIELDS=decode_tps' \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    'GREP:carries no decode_tps field'
no_result
verdict

run_case dropped_round_counter_refused 1 \
    "a counter a per-round figure is derived from is required" \
    'STUB_DROP_FIELDS=emitted,tokens_per_round' \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    'GREP:carries no emitted'
no_result
verdict

# ── The inputs ────────────────────────────────────────────────────────────────

# The published root is held to `published_samples.py` and no flag turns that
# off. This runs out of a repo whose prompts/published IS the shrunken copy and
# passes no --samples-root, so the harness reaches it as the published root and
# the anchor refuses it. Every other case here measures that same copy through
# --samples-root, which is the whole difference between the two paths.
run_case canonical_root_is_verified 1 \
    "the published root is verified, and nothing measured against it can skip that" \
    CANONICAL:1 \
    'GREP:manifest count is 1 and disagrees with published_samples.py' \
    'GREP:do not re-derive from what'
no_result
verdict

run_case override_root_is_marked_unverified 0 \
    "an overridden sample root is measured but is not a published measurement" \
    'GREP:UNVERIFIED SAMPLES' \
    'GREP:not a published measurement'
[ "$(jq_of "r['unverified_samples']")" = "True" ] ||
    note_bad "the result does not declare itself unverified"
verdict

# The cell table and the sample sets can drift apart two ways, and neither is
# reachable by editing a manifest: `published_samples.py verify` pins the
# manifest's dataset keys to its own source list first. The drift is a cell
# table that stopped matching a rebuilt sample root, so these two drive the
# renderer against roots that hold what such a rebuild would produce.

payload_case() { # payload_case <name> <want-exit> <what-it-proves> <grep> <cells...>
    CASE_NAME="$1"
    local want="$2"
    CASE_WHAT="$3"
    local pattern="$4"
    shift 4
    local dir="${WORK}/payload_${CASE_NAME}"
    CASE_OUT="${WORK}/payload_${CASE_NAME}.log"
    mkdir -p "${dir}"
    local cell_args=() c
    for c in "$@"; do cell_args+=(--cell "$c"); done
    python3 "${REPO_ROOT}/scripts/lib/published_payloads.py" \
        --samples-root "${SAMPLES}" --out "${dir}" --model-id stub \
        --index "${dir}/index.tsv" "${cell_args[@]}" >"${CASE_OUT}" 2>&1
    local got=$?
    CASE_BAD=""
    [ "${got}" -ne "${want}" ] && CASE_BAD="exit=${got} (want ${want})"
    grep -qE "${pattern}" "${CASE_OUT}" || note_bad "missing /${pattern}/"
}

payload_case cell_names_a_dataset_the_root_lacks 1 \
    "a cell naming a dataset the root does not hold is refused" \
    "holds no dataset gsm8k" \
    mt_bench:1024 math_500:1024 humaneval:1024 gsm8k:1024
verdict

payload_case dataset_no_cell_measures_is_refused 1 \
    "a checked-in dataset no cell measures is refused, not skipped" \
    "which no cell measures" \
    mt_bench:1024 math_500:1024
verdict

payload_case every_dataset_covered_is_accepted 0 \
    "a cell table that covers the root renders every request" \
    "8 requests over 4 cells" \
    mt_bench:1024 math_500:1024 humaneval:1024 math_500:4096
verdict

run_case snapshot_without_sampling_defaults_refused 1 \
    "a checkpoint that states no sampling defaults is not measured" \
    "VERIFIER:${NO_DEFAULTS}" \
    'GREP:generation_config.json does not exist' \
    'GREP:top-k disabled'
no_result
verdict

# Each of the three has a fallback that is not this checkpoint's, and top_k's is
# the quiet one: absent means top-k disabled, while the protocol names top-k 20.
run_case snapshot_without_top_k_refused 1 \
    "a checkpoint stating temperature and top_p but no top_k is refused" \
    "VERIFIER:${HALF_DEFAULTS}" \
    'GREP:states no top_k'
no_result
verdict

run_case hostile_draft_kind_refused 1 \
    "a drafter kind that would escape the scratch tree is refused at parse" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    ARGS:--draft-kind ARGS:../../etc/mtp \
    'GREP:is not a bare lower-case name'
no_result
verdict

run_case unusable_block_size_refused 1 \
    "a block with no room for a draft token is refused" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" ARGS:--draft-block-size ARGS:1 \
    'GREP:must be an integer >= 2'
no_result
verdict

run_case drafter_flags_without_a_drafter_refused 1 \
    "a drafter flag on a run with no drafter is refused, not ignored" \
    ARGS:--draft-kind ARGS:mtp \
    'GREP:describe a drafter this run was not given'
no_result
verdict
# ── The host boundary ─────────────────────────────────────────────────────────
#
# Both directions, because either alone is satisfiable by a broken script: the
# gate still fires without the flag, and with the flag a hostile host and a
# quiet host reach the same verdict.

run_case hostile_host_stops_a_measured_run 125 \
    "the quiescence gate fires on a hostile host" \
    MEASURED:1 HOST:hostile \
    'GREP:host is not quiescent' \
    'GREP:pass --allow-busy-host'
no_result
verdict

run_case hostile_host_measured_with_the_waiver_is_tainted 0 \
    "--allow-busy-host measures and says every number is suspect" \
    MEASURED:1 HOST:hostile ARGS:--allow-busy-host \
    'GREP:every number below is suspect' \
    'GREP:TAINTED'
[ "$(jq_of "'entry gate' in r['host']['taint']")" = "True" ] ||
    note_bad "the result does not record the taint: $(jq_of "r['host']['taint']")"
verdict

# `unmeasured` is a snapshot nobody could take. Folding it into "nothing was
# running" is the exact failure the entry gate is careful to avoid, and the
# per-pass window has to be equally careful. This host answers the entry gate
# and then stops answering.
run_case unmeasured_pass_window_taints 0 \
    "a pass window nobody could sample taints the run" \
    MEASURED:1 HOST:flaky \
    'GREP:TAINTED' \
    'GREP:unmeasured'
[ "$(jq_of "'unmeasured' in r['host']['taint']")" = "True" ] ||
    note_bad "the result does not record it: $(jq_of "r['host']['taint']")"
verdict

run_case quiet_host_measured 0 \
    "a quiet host measures without a taint" \
    MEASURED:1 HOST:quiet \
    'NOGREP:TAINTED'
[ "$(jq_of "r['synthetic_arms']")" = "False" ] ||
    note_bad "a measured run declared itself synthetic"
[ "$(jq_of "len(r['host']['pass_windows'])")" = "3" ] ||
    note_bad "a measured run recorded $(jq_of "len(r['host']['pass_windows'])") pass windows"
verdict

run_case synthetic_arms_waives_the_host_gate 0 \
    "the flag takes the machine out of the run" \
    HOST:hostile \
    'GREP:INTERFERENCE GATE: OFF' \
    'NOGREP:host is not quiescent' \
    'NOGREP:TAINTED'
SYNTHETIC_HOSTILE_MEAN="$(jq_of "r['macro']['mean']")"
[ "$(jq_of "r['synthetic_arms']")" = "True" ] ||
    note_bad "the run does not declare itself synthetic"
[ "$(jq_of "r['host']['pass_windows']")" = "[]" ] ||
    note_bad "a run that consulted nothing filed a reading: $(jq_of "r['host']['pass_windows']")"
verdict

run_case synthetic_arms_verdict_is_host_independent 0 \
    "a hostile and a quiet host give the flag the same answer" \
    HOST:quiet
SYNTHETIC_QUIET_MEAN="$(jq_of "r['macro']['mean']")"
[ "${SYNTHETIC_HOSTILE_MEAN}" = "${SYNTHETIC_QUIET_MEAN}" ] ||
    note_bad "hostile gave ${SYNTHETIC_HOSTILE_MEAN}, quiet gave ${SYNTHETIC_QUIET_MEAN}"
verdict

# The flag waives host state and nothing else. A mode that skipped every guard
# would make this whole suite green. The guard used here is deliberately not the
# one any other case breaks: sharing one would mean a single mutation kills both
# and this case proves nothing the other did not.
run_case synthetic_arms_waives_no_arm_reading_guard 1 \
    "the flag does not waive a guard that reads the run" \
    HOST:hostile 'STUB_KV_QUANT_SUPPRESS=1' \
    'GREP:never said which KV codec'
no_result
verdict

# ── The binary that produced the numbers ──────────────────────────────────────
#
# A digest says whether the file changed. It does not say whether this is the
# build the numbers are being attributed to — a stash-build-unstash cycle once
# left the previous binary in `target/` and an A/B compared it against itself,
# and both arms' digests agreed because both were one file. What separates them
# is whether the binary contains the thing the run depends on, which here is the
# log-message literal each reading is read off.

run_case binary_identity_is_recorded 0 \
    "the result names the binary it measured, and what it can write" \
    'GREP:binary     : sha256:'
WANT_SHA="$(shasum -a 256 "${STUB}" | awk '{print $1}')"
[ "$(jq_of "r['binary']['sha256']")" = "${WANT_SHA}" ] ||
    note_bad "binary sha256=$(jq_of "r['binary']['sha256']") (want ${WANT_SHA})"
[ "$(jq_of "r['binary']['arm']")" = "plain" ] ||
    note_bad "binary arm=$(jq_of "r['binary']['arm']")"
[ "$(jq_of "all(v > 0 for v in r['binary']['markers'].values())")" = "True" ] ||
    note_bad "a marker was recorded absent: $(jq_of "r['binary']['markers']")"
verdict

run_case binary_missing_a_marker_refused 1 \
    "a binary that cannot write a reading's event is refused before it serves" \
    MARKERLESS:itl \
    'GREP:contains none of .generate: ITL stats' \
    'GREP:cannot write the event this run reads its decode_rate from' \
    'NOGREP:\[server\] pid='
no_result
verdict

# ── The campaign ──────────────────────────────────────────────────────────────
#
# One directory per label, one subdirectory per run, the raw material of every
# pass beside the result. The label is what finds a campaign's rows in runs.db
# afterwards, so it is required and it is held to the characters a directory
# name and a LIKE pattern both accept.

run_case label_is_required 1 \
    "a run no campaign owns would leave rows nobody could find as one set" \
    NOLABEL:1 \
    'GREP:label NAME is required' \
    'NOGREP:\[server\] pid='
no_result
verdict

label_n=0
for bad in Upper under_score a/b -lead 'a b'; do
    label_n=$((label_n + 1))
    run_case "label_${label_n}_is_refused" 1 \
        "label '${bad}' names a directory and enters a LIKE pattern, so it is refused" \
        "LABEL:${bad}" \
        'GREP:must be 1-64 characters of lower-case' \
        'NOGREP:\[server\] pid='
    no_result
    verdict
done

run_case campaign_layout 0 \
    "a run is filed under its label, with every pass's raw material beside the result" \
    FIXED_TS:1
RUN="${CASE_HOME}/bench/published/selftest-campaign/20260906T000000Z-stub-model-plain"
[ -f "${RUN}/result.json" ] || note_bad "no result.json at ${RUN#${CASE_HOME}/}"
[ "$(jq_of "r['label']")" = "selftest-campaign" ] ||
    note_bad "label=$(jq_of "r['label']")"
for n in 1 2 3; do
    for f in engine.json engine_with_fixed.json fixed.json fixed.kv index.tsv memory.txt \
        pass.json serve.jsonl; do
        [ -s "${RUN}/passes/${n}/${f}" ] || note_bad "passes/${n}/${f} is missing or empty"
    done
    # The stub server prints nothing, so the file is there and may be empty.
    [ -e "${RUN}/passes/${n}/server_stdout.txt" ] || note_bad "passes/${n}/server_stdout.txt is missing"

done
[ -e "${CASE_HOME}/bench/spec_bench_published" ] &&
    note_bad "a result was also filed at the old location"
# The pass files the result was built from are the filed ones.
[ "$(python3 -c 'import json, sys
r = json.load(open(sys.argv[1]))
p = json.load(open(sys.argv[2]))
print(len(r["samples"]) == 3 * len(p["samples"]))' "${RUN}/result.json" "${RUN}/passes/1/pass.json")" = "True" ] ||
    note_bad "the filed pass.json is not one of the three the result was built from"
verdict

run_case a_run_does_not_overwrite_another 1 \
    "a run directory that exists is an error, never reused" \
    FIXED_TS:1 \
    PREDIR:bench/published/selftest-campaign/20260906T000000Z-stub-model-plain \
    'GREP:already exists; a run does not overwrite another' \
    'NOGREP:\[server\] pid='
no_result
verdict

run_case another_campaigns_run_is_no_collision 0 \
    "the same start stamp under another label is another run" \
    FIXED_TS:1 \
    LABEL:another-campaign \
    PREDIR:bench/published/selftest-campaign/20260906T000000Z-stub-model-plain
verdict

# ── Thermal state ─────────────────────────────────────────────────────────────
#
# Recorded beside the numbers and joined to the same taint field the
# interference gate writes to, not left as a footnote. All three cases drive a
# `pmset` shim: a verdict here must not depend on what this Mac's own thermal
# history holds. The state that must NOT taint is the default one every other
# measured case above already runs under — an unthrottled Mac has posted no
# level at all, and tainting on that would make the field meaningless.

run_case thermal_throttle_taints 0 \
    "a machine held below full speed taints the numbers taken on it" \
    MEASURED:1 THERM:throttled \
    'GREP:TAINTED' \
    'GREP:throttled=62'
[ "$(jq_of "'throttled=62' in r['host']['taint']")" = "True" ] ||
    note_bad "the result does not record it: $(jq_of "r['host']['taint']")"
[ "$(jq_of "len(r['host']['thermal'])")" = "9" ] ||
    note_bad "$(jq_of "len(r['host']['thermal'])") thermal readings (want 3 per pass)"
[ "$(jq_of "r['host']['thermal_source']")" = "pmset -g therm" ] ||
    note_bad "thermal_source=$(jq_of "r['host']['thermal_source']")"
verdict

run_case thermal_unreadable_taints 0 \
    "nobody having looked is not the same as having looked and seen nothing" \
    MEASURED:1 THERM:unreadable \
    'GREP:TAINTED' \
    'GREP:unreadable'
[ "$(jq_of "'unreadable' in r['host']['taint']")" = "True" ] ||
    note_bad "the result does not record it: $(jq_of "r['host']['taint']")"
verdict

run_case thermal_at_full_speed_does_not_taint 0 \
    "a posted level at full speed is recorded and taints nothing" \
    MEASURED:1 THERM:nominal \
    'NOGREP:TAINTED'
[ "$(jq_of "sorted({t.split()[-1] for t in r['host']['thermal']})")" = "['nominal']" ] ||
    note_bad "thermal=$(jq_of "r['host']['thermal']")"
verdict

# A run that declared it would consult nothing files no thermal reading either,
# for the same reason it files no host window: there is nothing on this machine
# it is entitled to describe.
run_case synthetic_arms_consults_no_thermometer 0 \
    "the flag takes the thermometer out of the run with the rest of the machine" \
    THERM:throttled \
    'NOGREP:TAINTED'
[ "$(jq_of "r['host']['thermal']")" = "[]" ] ||
    note_bad "a run that consulted nothing filed a reading: $(jq_of "r['host']['thermal']")"
verdict

# ── The client ────────────────────────────────────────────────────────────────
#
# The client sends the request itself. A client fed by a separate `curl` could
# start after the stream had begun and stamp the chunks it had missed all at its
# own start, so the window shrank and the rate read high. The client shim here
# starts the second client a second late, and every request of every server of
# the case must still find its own client already running.
run_case every_request_comes_from_a_running_client 0 \
    "a client that starts late has not missed any of the stream" \
    CLIENT_SHIM:2
[ "$(wc -l <"${CASE_HOME}/client/seen" 2>/dev/null | tr -d ' ')" = "30" ] ||
    note_bad "the stub saw $(wc -l <"${CASE_HOME}/client/seen" 2>/dev/null | tr -d ' ') streamed requests (want 30, ten on each of three servers)"
[ "$(sort -u "${CASE_HOME}/client/seen" 2>/dev/null)" = "up" ] ||
    note_bad "a request reached a server before its client existed: $(sort -u "${CASE_HOME}/client/seen" 2>/dev/null | tr '\n' ' ')"
[ "$(grep -c -- '--server-stamps' "${CASE_HOME}/client/reader_argv" 2>/dev/null)" = "30" ] ||
    note_bad "not every client was asked to read the stub's stamps"
[ "$(jq_of "r['protocol']['client_clock']")" = "server_stamps" ] ||
    note_bad "the result does not say its clock: $(jq_of "r['protocol']['client_clock']")"
verdict

# The harness asks the client for the stub's stamps only when it is asked to.
# Without the argument the stamps are comments, the client times the chunks, and
# the run is whatever this host makes of an instant stub, so the exit code is
# not asserted. What is asserted is what the client was told.
run_case stamps_are_not_used_unless_asked - \
    "the harness does not pass the stamps on by itself" \
    NO_STAMPS:1
[ -s "${CASE_HOME}/client/reader_argv" ] || note_bad "no client ran"
if grep -q -- '--server-stamps' "${CASE_HOME}/client/reader_argv" 2>/dev/null; then
    note_bad "a client was asked for stamps although the harness was not"
fi
verdict

# The reader on the path a real server takes: no stamps, so every chunk is timed
# by the reader's own clock when it is read. These cases give it a response whose
# clock moves only as lines are handed over, so they read no host clock. A reader
# that took the whole response and then split it would stamp every chunk with one
# instant and have no window.
READER_CASES="${WORK}/reader_cases.py"
cat >"${READER_CASES}" <<'PYEOF'
import contextlib
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading

READER = sys.argv[2]
spec = importlib.util.spec_from_file_location("sse", READER)
sse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sse)

CHUNK = b'data: {"choices": [{"delta": {"content": "t"}}]}\n'
BLANK = b"\n"
DONE = b"data: [DONE]\n"


def stamp(seconds):
    return f": sent-at={seconds}\n".encode()


class Delivery:
    """A response whose clock moves half a second for each line handed over."""

    chunked = False
    length = None

    def __init__(self, lines):
        self.lines = list(lines)
        self.now = 10.0

    def readline(self):
        if not self.lines:
            return b""
        self.now += 0.5
        return self.lines.pop(0)

    def read(self):
        self.now += 0.5
        rest = b"".join(self.lines)
        self.lines = []
        return rest


def run(lines, *flags):
    delivery = Delivery(lines)
    sse.post = lambda url, body: delivery
    sse.time.monotonic = lambda: delivery.now
    with tempfile.NamedTemporaryFile(delete=False) as body:
        body.write(b"{}")
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = sse.main(["--url", "http://x/y", "--body-file", body.name, *flags])
    finally:
        os.unlink(body.name)
    return code, out.getvalue(), err.getvalue()


def expect(what, got, want):
    assert got == want, f"{what}: got {got!r}, want {want!r}"


def case_delivery():
    plain = [CHUNK, BLANK, CHUNK, BLANK, CHUNK, BLANK, DONE]
    code, out, _ = run(plain)
    expect("exit", code, 0)
    assert "clock=client\n" in out and "decode_tps=1.000000\n" in out, out

    stamped = [stamp(0.0), CHUNK, BLANK, stamp(0.1), CHUNK, BLANK, stamp(0.2), CHUNK, BLANK, DONE]
    code, out, _ = run(stamped, "--server-stamps")
    expect("exit with stamps", code, 0)
    assert "clock=server_stamps\n" in out and "decode_tps=10.000000\n" in out, out

    code, out, _ = run(stamped)
    expect("exit without the argument", code, 0)
    assert "clock=client\n" in out and "decode_tps=0.666667\n" in out, out


def case_stamps():
    def stamped(*seconds):
        lines = []
        for value in seconds:
            lines += [] if value is None else [stamp(value)]
            lines += [CHUNK, BLANK]
        return lines + [DONE]

    for what, seconds, named in (
        ("inf", (0.0, "inf", 0.3), "sent-at=inf is not a finite time"),
        ("nan", (0.0, "nan", 0.3), "sent-at=nan is not a finite time"),
        ("not a number", (0.0, "soon", 0.3), "not a number"),
        ("equal", (0.0, 0.1, 0.1), "content chunk 3: sent-at=0.1 does not come after chunk 2"),
        ("decreasing", (0.0, 0.2, 0.1), "content chunk 3: sent-at=0.1 does not come after chunk 2"),
        ("missing on one chunk", (0.0, None, 0.2), "content chunk 2 has no sent-at stamp"),
    ):
        code, out, err = run(stamped(*seconds), "--server-stamps")
        expect(f"{what}: exit", code, 5)
        expect(f"{what}: output", out, "")
        assert named in err, f"{what}: the error does not say {named!r}: {err!r}"


def event(i):
    return ("data: " + json.dumps({"choices": [{"delta": {"content": f"t{i} "}}]}) + "\n\n").encode()


def chunked(body):
    return f"{len(body):x}\r\n".encode() + body + b"\r\n"


CHUNKED_HEAD = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
EVENTS = [chunked(event(i)) for i in range(3)]
DONE_CHUNK = chunked(b"data: [DONE]\n\n")

# name: (what the server sends, exit code the client has to give)
SERVERS = {
    "chunked, complete": (CHUNKED_HEAD + b"".join(EVENTS) + DONE_CHUNK + b"0\r\n\r\n", 0),
    "chunked, stops at a chunk boundary": (CHUNKED_HEAD + b"".join(EVENTS), 4),
    "chunked, stops after [DONE] with no last chunk": (CHUNKED_HEAD + b"".join(EVENTS) + DONE_CHUNK, 4),
    "chunked, stops inside a chunk": (CHUNKED_HEAD + b"".join(EVENTS) + b'40\r\ndata: {"cho', 4),
    "length, body short": (
        b"HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n" + event(0) + event(1),
        4,
    ),
    "close-delimited, complete": (
        b"HTTP/1.0 200 OK\r\n\r\n" + event(0) + event(1) + b"data: [DONE]\n\n",
        0,
    ),
}


def serve_once(reply):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def answer():
        connection, _ = listener.accept()
        with connection:
            # Read the whole request, body included, before the reply: a close
            # with request bytes unread resets the connection and the client
            # loses the reply.
            seen = b""
            while b"\r\n\r\n" not in seen:
                seen += connection.recv(4096)
            head, _, body = seen.partition(b"\r\n\r\n")
            owed = int(head.lower().split(b"content-length:")[1].split(b"\r\n")[0])
            while len(body) < owed:
                body += connection.recv(4096)
            connection.sendall(reply)
        listener.close()

    threading.Thread(target=answer, daemon=True).start()
    return listener.getsockname()[1]


# The client runs under a launcher that holds the body back 50 ms after the
# headers, so the server always has to deal with a request that arrives in two
# parts.
SPLIT_REQUEST = (
    "import http.client, runpy, sys, time\n"
    "send = http.client.HTTPConnection.send\n"
    "def late(self, data):\n"
    "    if not bytes(data).startswith(b'POST'):\n"
    "        time.sleep(0.05)\n"
    "    return send(self, data)\n"
    "http.client.HTTPConnection.send = late\n"
    "sys.argv = sys.argv[1:]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def read_response(port, raw_path):
    with tempfile.NamedTemporaryFile(delete=False) as body:
        body.write(b"{}")
    try:
        done = subprocess.run(
            [sys.executable, "-c", SPLIT_REQUEST, READER,
             "--url", f"http://127.0.0.1:{port}/v1/x",
             "--body-file", body.name, "--raw", raw_path],
            capture_output=True, text=True, timeout=120,
        )
    finally:
        os.unlink(body.name)
    return done


def case_cut():
    for name, (reply, want) in SERVERS.items():
        with tempfile.TemporaryDirectory() as scratch:
            raw = os.path.join(scratch, "raw.txt")
            with open(raw, "w") as old:
                old.write("0.000001 data: ANOTHER REQUEST\n")
            done = read_response(serve_once(reply), raw)
            expect(f"{name}: exit ({done.stderr.strip()})", done.returncode, want)
            if want:
                assert "did not end as a complete response" in done.stderr, (name, done.stderr)
            with open(raw) as handle:
                kept = handle.read()
            assert "ANOTHER REQUEST" not in kept, f"{name}: --raw still holds an earlier request"
            assert kept.count("data:") >= 1, f"{name}: --raw holds none of this response: {kept!r}"

    # A request that never reaches a server leaves no data of an earlier one.
    refused = socket.socket()
    refused.bind(("127.0.0.1", 0))
    port = refused.getsockname()[1]
    refused.close()
    with tempfile.TemporaryDirectory() as scratch:
        raw = os.path.join(scratch, "raw.txt")
        with open(raw, "w") as old:
            old.write("0.000001 data: ANOTHER REQUEST\n")
        done = read_response(port, raw)
        expect("refused: exit", done.returncode, 4)
        with open(raw) as handle:
            expect("refused: --raw", handle.read(), "")


globals()["case_" + sys.argv[1]]()
PYEOF
reader_case() { # reader_case <name> <what> <case-fn>
    CASE_NAME="$1"
    CASE_WHAT="$2"
    CASE_OUT="${WORK}/${CASE_NAME}.log"
    python3 "${READER_CASES}" "$3" "${REPO_ROOT}/scripts/lib/sse_decode_window.py" >"${CASE_OUT}" 2>&1
    local got=$?
    CASE_BAD=""
    [ "${got}" -ne 0 ] && CASE_BAD="exit=${got} (want 0)"
}
reader_case reader_times_chunks_as_they_are_delivered \
    "the client's own clock times each chunk as it arrives; stamps count only when asked" delivery
verdict
reader_case reader_refuses_unusable_stamps \
    "a stamp that is not finite, not increasing or missing is an error that names it" stamps
verdict
reader_case reader_fails_on_a_response_cut_short \
    "a stream that does not end as a complete response fails, and --raw is this request's alone" cut
verdict

# ── The fixed-length prompt ───────────────────────────────────────────────────
#
# The stub charges one token per four characters above a base of 100, so the fit
# is a real search over the corpus and lands on the target exactly or not at all.

run_case fixed_prompt_is_fitted_to_the_target 0 \
    "the fixed prompt is cut to the token count the protocol names, not near it" \
    'GREP:fitted 1355 tokens' \
    'GREP:FIXED PROMPT — 1355 tokens'
[ "$(jq_of "r['fixed_prompt']['prompt_tokens']")" = "1355" ] ||
    note_bad "prompt_tokens=$(jq_of "r['fixed_prompt']['prompt_tokens']")"
[ "$(jq_of "r['fixed_prompt']['target_tokens']")" = "1355" ] ||
    note_bad "target_tokens=$(jq_of "r['fixed_prompt']['target_tokens']")"
[ "$(jq_of "r['fixed_prompt']['corpus']")" = "longctx_4k.json" ] ||
    note_bad "corpus=$(jq_of "r['fixed_prompt']['corpus']")"
# The body travels with the run and hashes to the address recorded for it: the
# fitted body belongs to this tokenizer and is checked in nowhere.
[ "$(jq_of "__import__('hashlib').sha256(__import__('json').dumps(r['fixed_prompt']['messages'], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == r['fixed_prompt']['body_sha256']")" = "True" ] ||
    note_bad "the recorded body does not hash to the recorded address"
[ "$(jq_of "r['fixed_prompt']['memory_poll_ms']")" = "250" ] ||
    note_bad "memory_poll_ms=$(jq_of "r['fixed_prompt']['memory_poll_ms']")"
verdict

run_case fixed_prompt_rates_are_the_engine_reading 0 \
    "output speed is the engine's decode window and input speed is tokens over TTFT"
close_to "$(jq_of "r['fixed_prompt']['decode_tps']['mean']")" 25.0 0.0001 ||
    note_bad "decode_tps=$(jq_of "r['fixed_prompt']['decode_tps']['mean']")"
# 1355 prompt tokens over the 42 ms TTFT the engine reported.
close_to "$(jq_of "r['fixed_prompt']['prefill_tps']['mean']")" 32261.9 0.0001 ||
    note_bad "prefill_tps=$(jq_of "r['fixed_prompt']['prefill_tps']['mean']")"
[ "$(jq_of "r['fixed_prompt']['phys_footprint_bytes']['max']")" = "31000000000" ] ||
    note_bad "phys peak=$(jq_of "r['fixed_prompt']['phys_footprint_bytes']['max']")"
[ "$(jq_of "r['fixed_prompt']['rss_bytes']['max']")" = "29000000000" ] ||
    note_bad "rss peak=$(jq_of "r['fixed_prompt']['rss_bytes']['max']")"
# The stub's allocator peak rises 100 MiB per served request from 28000 plus a
# per-pass offset, and the fixed prompt is the second request of each server: its
# own event reads 28100 in pass 1, not the warmup's 28000 or the first cell's
# 28200. The three runs are 28100, 30100 and 28600, and the block keeps their
# maximum, not their mean, their minimum or the last.
[ "$(jq_of "r['fixed_prompt']['metal_peak_alloc_mb']['run_values']")" = "[28100, 30100, 28600]" ] ||
    note_bad "metal peak runs=$(jq_of "r['fixed_prompt']['metal_peak_alloc_mb']['run_values']")"
[ "$(jq_of "r['fixed_prompt']['metal_peak_alloc_mb']['max']")" = "30100" ] ||
    note_bad "metal peak max=$(jq_of "r['fixed_prompt']['metal_peak_alloc_mb']['max']")"
[ "$(jq_of "r['fixed_prompt']['metal_peak_alloc_mb']['min']")" = "28100" ] ||
    note_bad "metal peak min=$(jq_of "r['fixed_prompt']['metal_peak_alloc_mb']['min']")"
verdict

run_case metal_peak_event_missing_refused 1 \
    "a plain log missing a request's Metal peak event is refused" \
    'STUB_PEAK_LINES=5' \
    "GREP:Metal peak events, expected ${EVENTS_PER_PASS}"
no_result
verdict

run_case metal_peak_of_zero_refused 1 \
    "a Metal peak that reads zero is not a peak and is refused" \
    'STUB_METAL_PEAK_MB=0' \
    'GREP:not a peak allocation'
no_result
verdict

run_case binary_missing_metal_peak_marker_refused 1 \
    "a binary that cannot write the Metal peak event is refused before it serves" \
    MARKERLESS:peak \
    'GREP:contains none of .generate: metal peak alloc' \
    'GREP:cannot write the event this run reads its metal_peak from' \
    'NOGREP:\[server\] pid='
no_result
verdict

# Input speed has two definitions. The stub's TTFT is 42 ms and its prefill 250
# ms, so the client-visible rate is 32261.9 and the engine's own 5420.0: a
# harness that read one clock for both rows would print the same number twice.
run_case engine_input_speed_is_forwarded_tokens_over_the_prefill 0 \
    "the engine's input speed is the tokens its prefill ran over its own prefill time" \
    'GREP:tok/s eng'
close_to "$(jq_of "r['fixed_prompt']['prefill_engine_tps']['mean']")" 5420.0 0.0001 ||
    note_bad "prefill_engine_tps=$(jq_of "r['fixed_prompt']['prefill_engine_tps']['mean']")"
close_to "$(jq_of "r['fixed_prompt']['prefill_tps']['mean']")" 32261.9 0.0001 ||
    note_bad "prefill_tps=$(jq_of "r['fixed_prompt']['prefill_tps']['mean']")"
close_to "$(jq_of "r['fixed_prompt']['prefill_ms']['mean']")" 250.0 0.0001 ||
    note_bad "prefill_ms=$(jq_of "r['fixed_prompt']['prefill_ms']['mean']")"
verdict

# Each pass is a server of its own and reads its own prefill time. The three
# runs are 1355 tokens over 200, 204 and 208 ms; a block that kept the last run,
# or the first, would not report their mean.
run_case engine_input_speed_is_the_mean_of_the_three_passes 0 \
    "the engine rate of the fixed prompt is a mean over the passes, with its spread" \
    'STUB_PREFILL_MS=200;204;208'
close_to "$(jq_of "r['fixed_prompt']['prefill_engine_tps']['mean']")" 6643.8601 0.0001 ||
    note_bad "mean=$(jq_of "r['fixed_prompt']['prefill_engine_tps']['mean']")"
close_to "$(jq_of "r['fixed_prompt']['prefill_engine_tps']['pass_means'][1]")" 6642.1569 0.0001 ||
    note_bad "pass 2=$(jq_of "r['fixed_prompt']['prefill_engine_tps']['pass_means'][1]")"
close_to "$(jq_of "r['fixed_prompt']['prefill_engine_tps']['range_pct']")" 3.9231 0.001 ||
    note_bad "range=$(jq_of "r['fixed_prompt']['prefill_engine_tps']['range_pct']")"
[ "$(jq_of "r['fixed_prompt']['prefill_engine_tps']['stable']")" = "True" ] ||
    note_bad "a 3.9% spread was refused"
verdict

# Past the 5% band the mean is withheld, as it is for the other published rates.
run_case unstable_engine_input_speed_is_refused 3 \
    "three prefills that disagree by more than the band publish no engine input speed" \
    'STUB_PREFILL_MS=200;212;200' \
    'GREP:RANGE REFUSAL: .*fixed_prompt/prefill_engine_tps'
[ "$(jq_of "r['fixed_prompt']['prefill_engine_tps']['stable']")" = "False" ] ||
    note_bad "the unstable engine rate was filed as stable"
[ "$(jq_of "r['fixed_prompt']['prefill_tps']['stable']")" = "True" ] ||
    note_bad "the TTFT rate was refused for the engine rate's spread"
verdict

# The protocol's input speed is over a cold cache, and the engine can now say
# whether it was. A prompt it took any token of from a cache is not cold.
run_case a_fixed_prompt_served_from_a_cache_is_refused 1 \
    "an input speed over a prompt the engine took from a cache is not a cold one" \
    'STUB_REUSED=100' \
    'GREP:took 100 of the fixed prompt.s tokens from a prompt cache' \
    'GREP:not over a cold prompt'
no_result
verdict

run_case an_engine_prefill_of_no_time_is_refused 1 \
    "a prefill that took no time has no rate, and is refused rather than divided by" \
    'STUB_PREFILL_MS=0' \
    'GREP:reported prefill_ms=0.0, which is not a prefill duration'
no_result
verdict

run_case a_negative_prefill_time_is_refused 1 \
    "a prefill time below zero is not a time" \
    'STUB_PREFILL_MS=-5' \
    'GREP:reports prefill_ms=-5.0, which is not a time'
no_result
verdict

run_case a_prefill_that_ran_no_token_but_took_time_is_refused 1 \
    "an event that reuses the whole prompt and still reports a prefill time is incoherent" \
    'STUB_REUSED=all' \
    'GREP:reports 250.0 ms for a prefill that ran no token'
no_result
verdict

run_case a_prefill_event_that_counts_more_than_the_prompt_is_refused 1 \
    "a cache supplying more tokens than the prompt holds is not one prefill" \
    'STUB_REUSED=99999' \
    'GREP:together they exceed it'
no_result
verdict

run_case prefill_event_missing_refused 1 \
    "a log missing a request's prefill event is refused, not zipped short" \
    'STUB_PREFILL_LINES=5' \
    "GREP:prefill events, expected ${EVENTS_PER_PASS}"
no_result
verdict

# The reader pairs the event with its request by order, so it checks the one
# thing it can see: the prompt the event counted is the prompt the server counted.
run_case a_prefill_event_of_another_request_is_refused 1 \
    "an event whose prompt count is not the request's belongs to another request" \
    'STUB_PREFILL_PROMPT=999' \
    'GREP:counts 999 prompt tokens where the server counted 1355' \
    'GREP:belongs to another request'
no_result
verdict

run_case a_cell_paired_with_another_requests_prefill_event_is_refused 1 \
    "each request's event is checked against its own prompt count, not only the fixed prompt's" \
    'STUB_PREFILL_PROMPT=999' \
    'STUB_PREFILL_PROMPT_FROM=2' \
    "GREP:mt_bench@1024/.*the engine's prefill event counts 999 prompt tokens" \
    'GREP:belongs to another request'
no_result
verdict

run_case a_speculative_cell_paired_with_another_requests_event_is_refused 1 \
    "the same check on the speculative arm, where the order is the only pairing there is" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" ARGS:--draft-kind ARGS:mtp \
    'STUB_PREFILL_PROMPT=999' \
    'STUB_PREFILL_PROMPT_FROM=2' \
    "GREP:the engine's prefill event counts 999 prompt tokens"
no_result
verdict

run_case binary_missing_prefill_marker_refused 1 \
    "a binary that cannot write the prefill event is refused before it serves" \
    MARKERLESS:prefill \
    'GREP:contains none of .generate: prefill' \
    'GREP:cannot write the event this run reads its prefill from' \
    'NOGREP:\[server\] pid='
no_result
verdict

# The round loop writes the same event, so a speculative run carries the engine's
# prefill on every row. The stub's seed is drawn by the prefill, so it ran the
# whole prompt, and each request counts its own prompt.
run_case speculative_rows_carry_the_engine_prefill 0 \
    "a speculative request has its prefill event read like a plain one" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" ARGS:--draft-kind ARGS:mtp
ENGINE_JSON="$(dirname "$(result_of)")/passes/1/engine.json"
[ "$(python3 -c 'import json, sys
rows = json.load(open(sys.argv[1]))["requests"]
print(len(rows), all(r["forwarded_tokens"] == r["prompt_tokens"] and r["reused_tokens"] == 0
                     and r["prefill_ms"] == 250.0 and r["prompt_tokens"] > 100 for r in rows))' "${ENGINE_JSON}" 2>&1)" = "${REQUESTS_PER_PASS} True" ] ||
    note_bad "engine rows: $(python3 -c 'import json, sys
print(json.load(open(sys.argv[1]))["requests"][0])' "${ENGINE_JSON}" 2>&1 | cut -c1-200)"
verdict

run_case speculative_prefill_event_missing_refused 1 \
    "a speculative log missing a request's prefill event is refused" \
    'STUB_PREFILL_LINES=5' \
    ARGS:--draft-model "ARGS:${WORK}/drafter" ARGS:--draft-kind ARGS:mtp \
    'GREP:prefill events, expected'
no_result
verdict

# ── The greedy pass ───────────────────────────────────────────────────────────
#
# The speed passes sample under the checkpoint's own defaults, so no two arms answer
# one question there. The greedy pass answers every sample at the macro budget once,
# at temperature 0, unstreamed and with logprobs, on a server of its own. The stub
# answers such a request with a text that is a function of the prompt, so the plain
# and the speculative stub agree on every sample unless a case marks one: index 0 is
# the MT-Bench sample, 1 and 2 are MATH-500's, 3 to 5 are HumanEval's.

run_case a_run_without_the_flag_makes_no_greedy_pass 0 \
    "the greedy pass is asked for, not a default: the speed runs stay as long as they were" \
    'NOGREP:greedy pass'
[ ! -e "$(dirname "$(result_of)")/greedy" ] || note_bad "a greedy directory was written"
[ "$(jq_of "'greedy_pass' in r or 'greedy_match' in r")" = "False" ] ||
    note_bad "the result carries a greedy figure"
verdict

run_case greedy_pass_keeps_and_seals_its_answers 0 \
    "a plain run's greedy pass leaves its answers, an index of their digests and a sealed meta" \
    ARGS:--greedy-pass \
    'GREP:greedy pass: temperature 0'
PLAIN_RUN="$(dirname "$(result_of)")"
GREEDY="${PLAIN_RUN}/greedy"
[ "$(wc -l <"${GREEDY}/index.tsv" | tr -d ' ')" = "6" ] ||
    note_bad "index rows=$(wc -l <"${GREEDY}/index.tsv" | tr -d ' ')"
[ "$(find "${GREEDY}/answers" -type f | wc -l | tr -d ' ')" = "6" ] ||
    note_bad "answers=$(find "${GREEDY}/answers" -type f | wc -l | tr -d ' ')"
[ "$(jq_of "r['greedy_pass']['samples']")" = "6" ] || note_bad "greedy_pass samples"
[ "$(jq_of "r['greedy_pass']['arm']")" = "plain" ] || note_bad "greedy_pass arm"
[ "$(jq_of "r['greedy_pass']['kv_quant']")" = "mixed_k8g64_v4g64" ] || note_bad "greedy_pass kv_quant"
# What was sent: the greedy requests are the only ones that ask for logprobs, they
# are unstreamed at temperature 0 and the macro budget, and no speed request
# overrides the checkpoint's sampling.
[ "$(python3 -c 'import json, sys
bodies = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
greedy = [b for b in bodies if b.get("logprobs")]
print(len(greedy),
      all(b["temperature"] == 0 and b["stream"] is False and b["top_logprobs"] == 2
          and b["max_tokens"] == 1024 for b in greedy),
      all("temperature" not in b for b in bodies if not b.get("logprobs")))' "${CASE_HOME}/bodies.jsonl")" = "6 True True" ] ||
    note_bad "greedy bodies: $(python3 -c 'import json, sys
print([ (b.get("temperature"), b.get("stream"), b.get("logprobs")) for b in map(json.loads, open(sys.argv[1]))][-8:])' "${CASE_HOME}/bodies.jsonl")"
verdict

run_case greedy_match_is_judged_against_the_plain_run 0 \
    "each differing pair goes to the oracle and the result files its verdicts by dataset" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" ARGS:--draft-kind ARGS:mtp \
    ARGS:--greedy-against "ARGS:${PLAIN_RUN}" \
    'STUB_GREEDY_MARKS=1:NEAR,2:REFUSED,4:UNJUDGE' \
    'GREP:GREEDY MATCH against plain decode'
SPEC_RUN="$(dirname "$(result_of)")"
[ "$(jq_of "[ (k, e['samples'], e['identical'], e['near_tie'], e['refused'], e['unjudgeable']) for k, e in sorted(r['greedy_match']['datasets'].items()) ]")" = \
  "[('humaneval', 3, 2, 0, 0, 1), ('math_500', 2, 0, 1, 1, 0), ('mt_bench', 1, 1, 0, 0, 0)]" ] ||
    note_bad "counts=$(jq_of "sorted(r['greedy_match']['datasets'].items())")"
close_to "$(jq_of "r['greedy_match']['identical_rate']")" 0.5 0.0001 ||
    note_bad "identical_rate=$(jq_of "r['greedy_match']['identical_rate']")"
close_to "$(jq_of "r['greedy_match']['identical_or_near_tie_rate']")" 0.6666667 0.0001 ||
    note_bad "identical_or_near_tie_rate=$(jq_of "r['greedy_match']['identical_or_near_tie_rate']")"
[ "$(jq_of "[ d['verdict'] for d in r['greedy_match']['differing'] ]")" = "['near_tie', 'refused', 'unjudgeable']" ] ||
    note_bad "differing=$(jq_of "[ d['verdict'] for d in r['greedy_match']['differing'] ]")"
[ "$(jq_of "r['greedy_match']['differing'][0]['position']")" = "3" ] ||
    note_bad "the near tie's position was not carried"
[ "$(jq_of "r['greedy_pass']['arm'] + ' ' + r['greedy_pass']['decode_config']")" = "speculative mtp/block=5" ] ||
    note_bad "greedy_pass=$(jq_of "r['greedy_pass']")"
verdict

# A plain sample resumed from a cached prefix is a reference the speculative loop,
# which prefills the whole prompt every time, did not answer from. The greedy server
# runs with the prompt cache off, and the engine's events hold it to that.
run_case greedy_pass_runs_with_the_prompt_cache_off 0 \
    "the greedy server is started without a prompt cache, and no other server is" \
    ARGS:--greedy-pass \
    'STUB_GREEDY_CACHE_HIT=1' \
    'GREP:all from a full prefill'
[ "$(grep -c -e '--prompt-cache-slots 0' "${CASE_HOME}/serve_args.log")" = "1" ] ||
    note_bad "servers with the cache off: $(grep -c -e '--prompt-cache-slots 0' "${CASE_HOME}/serve_args.log")"
verdict

run_case a_greedy_answer_taken_from_a_cache_is_refused 1 \
    "a greedy answer decoded from a resumed prefill is not a reference, and the pass says so" \
    ARGS:--greedy-pass \
    'STUB_GREEDY_REUSED=50' \
    'GREP:took 50 tokens of greedy request 1 from a prompt cache'
no_result
verdict

run_case a_greedy_pass_missing_its_prefill_events_is_refused 1 \
    "a greedy pass whose log cannot be attributed answer by answer is refused" \
    ARGS:--greedy-pass \
    'STUB_GREEDY_NO_EVENT=1' \
    'GREP:holds 0 prefill events for 6 requests'
no_result
verdict

run_case a_plain_run_is_not_judged_against_plain 1 \
    "--greedy-against names a plain run, so a run with no drafter would be judged against itself" \
    ARGS:--greedy-against "ARGS:${PLAIN_RUN}" \
    'GREP:judges a speculative run against a plain one' \
    'NOGREP:\[server\] pid='
no_result
verdict

run_case greedy_against_a_directory_with_no_pass_is_refused 1 \
    "a directory that holds no sealed greedy pass is not a plain run to judge against" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    ARGS:--greedy-against "ARGS:${WORK}" \
    'GREP:holds no greedy/meta.json' \
    'NOGREP:\[server\] pid='
no_result
verdict

run_case greedy_against_a_speculative_pass_is_refused 1 \
    "the plain side must be a plain pass: a speculative one beside it compares an arm with itself" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    ARGS:--greedy-against "ARGS:${SPEC_RUN}" \
    'GREP:the plain pass is of the speculative arm, not the plain one'
no_result
verdict

run_case greedy_passes_of_two_codecs_are_not_compared 1 \
    "a speculative pass at one KV codec is not judged against a plain pass at another" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    ARGS:--greedy-against "ARGS:${PLAIN_RUN}" \
    'STUB_KV_QUANT=none' \
    'GREP:differ in kv_quant'
no_result
verdict

# The set of samples is part of what the plain pass was sealed over. An index edited
# afterwards no longer matches its meta, so the pass is not the one that was run.
TAMPERED="${WORK}/tampered_plain"
mkdir -p "${TAMPERED}"
/bin/cp -R "${PLAIN_RUN}/greedy" "${TAMPERED}/greedy"
python3 - "${TAMPERED}/greedy/index.tsv" <<'PY'
import pathlib, sys

path = pathlib.Path(sys.argv[1])
rows = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines()]
rows[0][4] = "0" * 64
path.write_text("".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
PY
run_case greedy_against_a_pass_edited_after_sealing_is_refused 1 \
    "an index that no longer matches the sample set its meta was sealed over is refused" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    ARGS:--greedy-against "ARGS:${TAMPERED}" \
    'GREP:no longer matches the sample set it was sealed over'
no_result
verdict

run_case an_oracle_whose_exit_code_disagrees_with_its_verdict_is_refused 1 \
    "the exit code and the verdict are one answer spelled twice; a pair that disagree is no judgement" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    ARGS:--greedy-against "ARGS:${PLAIN_RUN}" \
    'STUB_GREEDY_MARKS=1:BADEXIT' \
    'GREP:are one answer spelled twice'
no_result
verdict

run_case an_oracle_that_names_no_verdict_is_refused 1 \
    "an oracle that prints no verdict has not judged the pair, and the run does not guess" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    ARGS:--greedy-against "ARGS:${PLAIN_RUN}" \
    'STUB_GREEDY_MARKS=1:NOVERDICT' \
    "GREP:verdict None, which is not a judgement"
no_result
verdict

# ── The long-context row ──────────────────────────────────────────────────────
#
# `prompts/longctx_32k.json` is the real checked-in file, 100 kB of it. The stub
# counts any prompt over 20 000 characters at STUB_LONG_PROMPT_TOKENS, and the
# long-context servers are the fifth to seventh the stub starts on the plain arm
# (the fit, three passes, then these), so the fourth to sixth entries of
# STUB_PREFILL_MS are theirs.

run_case longctx_row_is_three_cold_runs_each_on_a_server_of_its_own 0 \
    "the 32k row is three runs, each on a server started at a ceiling its prompt fits" \
    ARGS:--longctx ARGS:32k \
    'STUB_LONG_PROMPT_TOKENS=32764' \
    'STUB_PREFILL_MS=250;250;250;200;204;208' \
    'GREP:LONG CONTEXT longctx_32k'
[ "$(jq_of "r['longctx']['longctx_32k']['prompt_tokens']")" = "32764" ] ||
    note_bad "prompt_tokens=$(jq_of "r['longctx']['longctx_32k']['prompt_tokens']")"
[ "$(jq_of "r['longctx']['longctx_32k']['target_tokens']")" = "32768" ] ||
    note_bad "target_tokens=$(jq_of "r['longctx']['longctx_32k']['target_tokens']")"
# 32768 plus a quarter, plus the 200-token budget.
[ "$(jq_of "r['longctx']['longctx_32k']['ctx_max']")" = "41160" ] ||
    note_bad "ctx_max=$(jq_of "r['longctx']['longctx_32k']['ctx_max']")"
[ "$(grep -c -e '--max-ctx 41160' "${CASE_HOME}/serve_args.log")" = "3" ] ||
    note_bad "servers at the long ceiling: $(grep -c -e '--max-ctx 41160' "${CASE_HOME}/serve_args.log")"
[ "$(grep -c -e '--max-ctx 8192' "${CASE_HOME}/serve_args.log")" = "4" ] ||
    note_bad "servers at the harness ceiling: $(grep -c -e '--max-ctx 8192' "${CASE_HOME}/serve_args.log")"
# 32764 tokens over 200, 204 and 208 ms of prefill; the client-visible rate is the
# same tokens over the 42 ms TTFT the stub reports.
[ "$(jq_of "[round(v, 1) for v in r['longctx']['longctx_32k']['prefill_engine_tps']['pass_means']]")" = "[163820.0, 160607.8, 157519.2]" ] ||
    note_bad "engine rates=$(jq_of "r['longctx']['longctx_32k']['prefill_engine_tps']['pass_means']")"
close_to "$(jq_of "r['longctx']['longctx_32k']['prefill_tps']['mean']")" 780095.2 0.0001 ||
    note_bad "prefill_tps=$(jq_of "r['longctx']['longctx_32k']['prefill_tps']['mean']")"
close_to "$(jq_of "r['longctx']['longctx_32k']['decode_tps']['mean']")" 25.0 0.0001 ||
    note_bad "decode_tps=$(jq_of "r['longctx']['longctx_32k']['decode_tps']['mean']")"
[ "$(jq_of "r['longctx']['longctx_32k']['prefill_engine_tps']['stable']")" = "True" ] ||
    note_bad "a 3.9% spread was refused"
# The request that was sent is the file's own messages at temperature 0 with a
# 200-token budget; the speed requests are untouched.
[ "$(python3 - "${CASE_HOME}/bodies.jsonl" "${REPO_ROOT}/prompts/longctx_32k.json" <<'PY'
import json, sys

bodies = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
messages = json.load(open(sys.argv[2]))["messages"]
long = [b for b in bodies if b.get("max_tokens") == 200]
print(len(long), all(b["messages"] == messages and b["temperature"] == 0 and b["stream"] for b in long))
PY
)" = "3 True" ] || note_bad "the long-context requests are not the file's messages at temperature 0"
verdict

run_case longctx_size_without_a_prompt_file_is_refused 1 \
    "a size with no checked-in prompt is refused before anything is served" \
    ARGS:--longctx ARGS:99k \
    'GREP:names no checked-in prompt' \
    'NOGREP:\[server\] pid='
no_result
verdict

run_case longctx_on_a_speculative_arm_is_refused 1 \
    "the long-context row is autoregressive, and a drafter's rate is not" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" ARGS:--longctx ARGS:32k \
    'GREP:measured on the plain arm only' \
    'NOGREP:\[server\] pid='
no_result
verdict

run_case a_longctx_prompt_served_from_a_cache_is_refused 1 \
    "an input speed over a long prompt the engine took from a cache is not a cold one" \
    ARGS:--longctx ARGS:32k \
    'STUB_LONG_PROMPT_TOKENS=32764' \
    'STUB_REUSED_LONG=100' \
    "GREP:took 100 of the longctx_32k prompt.s tokens from a prompt cache"
no_result
verdict

run_case a_longctx_prompt_of_another_size_is_refused 1 \
    "a prompt the server counts far from the size in its name is not filed under it" \
    ARGS:--longctx ARGS:32k \
    'STUB_LONG_PROMPT_TOKENS=20000' \
    'GREP:away from the 32768 it was built for'
no_result
verdict

run_case an_unstable_longctx_rate_is_refused 3 \
    "three long-context prefills that disagree past the band publish no engine input speed" \
    ARGS:--longctx ARGS:32k \
    'STUB_LONG_PROMPT_TOKENS=32764' \
    'STUB_PREFILL_MS=250;250;250;200;212;200' \
    'GREP:RANGE REFUSAL: .*longctx/longctx_32k/prefill_engine_tps'
[ "$(jq_of "r['longctx']['longctx_32k']['prefill_engine_tps']['stable']")" = "False" ] ||
    note_bad "the unstable engine rate was filed as stable"
verdict

# The protocol's fixed-prompt figure is the AUTOREGRESSIVE one. A rate a drafter
# produced is not that, and publishing it under that name is the comparison this
# whole harness exists to make honest.
# The stub answers /metrics with the full figure only while a streamed request is
# in flight, and with half of it otherwise, and the fixed-prompt request waits for
# one such reading. A harness that stops polling gets no in-flight reading and
# the request waits out --sample-wait before the run is refused; one that reads
# the gauge before or after the request files half the peak.
run_case memory_is_read_while_the_request_is_in_flight 0 \
    "the fixed prompt's resident peak has a reading taken during the request" \
    "STUB:--inflight-log ${WORK}/home_memory_is_read_while_the_request_is_in_flight/inflight"
[ "$(wc -l <"${CASE_HOME}/inflight" 2>/dev/null | tr -d ' ')" -ge 3 ] 2>/dev/null ||
    note_bad "fewer than one in-flight reading for each of the three passes: $(wc -l <"${CASE_HOME}/inflight" 2>/dev/null | tr -d ' ')"
verdict
run_case resident_peak_is_the_in_flight_reading 0 \
    "the resident peak is the figure read during the request, not the idle one"
[ "$(jq_of "r['fixed_prompt']['phys_footprint_bytes']['min']")" = "31000000000" ] ||
    note_bad "phys lowest peak=$(jq_of "r['fixed_prompt']['phys_footprint_bytes']['min']") (idle reads 15500000000)"
[ "$(jq_of "r['fixed_prompt']['rss_bytes']['min']")" = "29000000000" ] ||
    note_bad "rss lowest peak=$(jq_of "r['fixed_prompt']['rss_bytes']['min']") (idle reads 14500000000)"
verdict
run_case fixed_prompt_not_measured_on_a_speculative_arm 0 \
    "a drafter's rate is not filed as the autoregressive one" \
    ARGS:--draft-model "ARGS:${WORK}/drafter" \
    'GREP:not measured on a speculative arm'
[ "$(jq_of "'fixed_prompt' in r")" = "False" ] ||
    note_bad "a speculative run filed a fixed-prompt block"
verdict

run_case unreachable_fixed_target_refused 1 \
    "a target the corpus cannot reach on this tokenizer is refused, not rounded to" \
    'STUB_TOKEN_BASE=2000' \
    'GREP:a one-word prefix already counts more than 1355'
no_result
verdict

run_case memory_gauge_absent_refused 1 \
    "a resident peak over no sample is not a small peak" \
    'STUB_MEM_SUPPRESS=1' \
    'GREP:published no process-memory gauge'
no_result
verdict

run_case memory_gauge_zero_refused 1 \
    "a gauge that is published but carries nothing is refused" \
    'STUB_MEM_ZERO=1' \
    'GREP:is not a resident figure'
no_result
verdict

# The fixed prompt is request 1 of every pass, so its rate can be moved without
# touching a cell. Both directions on one fixture: the cells stay inside the
# band and are published while the fixed prompt's own spread is refused.
FIXED_UNSTABLE="$(python3 -c 'g = lambda r: f"{1000/r:.9f}"
print(";".join(["40," + g(23.5) + ",40",
                "40",
                "40," + g(26.5) + ",40"]))')"
run_case fixed_prompt_range_over_the_band_refused 3 \
    "the fixed prompt's own spread is refused without refusing the cells" \
    "STUB_GAP_MS=${FIXED_UNSTABLE}" \
    'GREP:RANGE REFUSAL: fixed_prompt/decode_tps'
[ "$(jq_of "r['fixed_prompt']['decode_tps']['stable']")" = "False" ] ||
    note_bad "the fixed prompt is marked stable at a 12% spread"
[ "$(jq_of "all(c['stable'] for c in r['cells'].values())")" = "True" ] ||
    note_bad "a cell was refused along with the fixed prompt"
verdict

# ── What the run-to-run range is a range of ───────────────────────────────────
#
# The request sends no seed, so the engine seeds one RNG per request from its
# fixed default and the three passes are one generation replayed. That is what
# makes the range a reading of machine variance — and it is a claim about the
# run, so it is measured rather than asserted.

run_case passes_are_replicas_when_the_lengths_agree 0 \
    "three passes that generated the same length are reported as replicas" \
    'GREP:so the range is machine variance and not sampling variance'
[ "$(jq_of "sum(c['divergent_samples'] for c in r['cells'].values())")" = "0" ] ||
    note_bad "divergent=$(jq_of "sum(c['divergent_samples'] for c in r['cells'].values())")"
[ "$(jq_of "r['protocol']['seed_policy']")" = "engine default, identical in all three passes" ] ||
    note_bad "seed_policy=$(jq_of "r['protocol']['seed_policy']")"
verdict

run_case divergent_lengths_are_reported_not_hidden 0 \
    "a pass that generated a different length is counted and said out loud" \
    'STUB_USAGE_TOKENS=97' 'STUB_USAGE_TOKENS_PASS2=131' \
    'GREP:samples generated a different length across'
[ "$(jq_of "sum(c['divergent_samples'] for c in r['cells'].values())")" = "8" ] ||
    note_bad "divergent=$(jq_of "sum(c['divergent_samples'] for c in r['cells'].values())") (want all 8)"
# Reported, never refused: the run is readable and its cells are clean.
[ "$(jq_of "all(c['stable'] for c in r['cells'].values())")" = "True" ] ||
    note_bad "a cell was refused for a length that moved"
verdict

# ── An aborted run ────────────────────────────────────────────────────────────
#
# Three servers run per invocation, and a SIGTERM to the harness — a CI timeout,
# an operator, a parent tearing down — must not leave one alive holding the
# Metal claim: no later run may stop a server it did not start.
sigterm_case() {
    CASE_NAME="sigterm_leaves_no_server_running"
    CASE_WHAT="a killed run takes its server with it"
    CASE_HOME="${WORK}/home_${CASE_NAME}"
    CASE_OUT="${WORK}/${CASE_NAME}.log"
    CASE_BAD=""
    mkdir -p "${CASE_HOME}/logs"
    printf '%s\n' "--hold ${CASE_HOME}/hold" >"${CASE_HOME}/stub_args"

    env -i \
        PATH="${WORK}/quiet:${SHIMS}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
        HOME="${WORK}" \
        RMLX_HOME="${CASE_HOME}" \
        STUB_BOUND_FLAG="${CASE_HOME}/stub_bound" \
        STUB_TOKEN_BASE=100 \
        bash "${FAKE_ROOT}/scripts/spec_bench_published.sh" \
        "${VERIFIER}" --port "${PORT}" --samples-root "${SAMPLES}" \
        --label selftest-campaign --synthetic-arms --server-stamps >"${CASE_OUT}" 2>&1 &
    local harness=$!

    request_held_or_harness_gone() {
        [ -e "${CASE_HOME}/hold.held" ] || ! kill -0 "${harness}" 2>/dev/null
    }
    harness_gone() { ! kill -0 "${harness}" 2>/dev/null; }

    if ! wait_until 300 request_held_or_harness_gone || [ ! -e "${CASE_HOME}/hold.held" ]; then
        note_bad "no request was held within 300 s, or the harness ended before it sent one; nothing was killed"
        kill "${harness}" 2>/dev/null
        wait "${harness}" 2>/dev/null
        stop_stubs "${CASE_HOME}/stub_bound"
        return
    fi

    kill -TERM "${harness}" 2>/dev/null
    : >"${CASE_HOME}/hold.release"
    if ! wait_until 300 harness_gone; then
        note_bad "the harness was still running 300 s after SIGTERM"
        kill -9 "${harness}" 2>/dev/null
    fi
    wait "${harness}" 2>/dev/null

    local server
    while read -r server; do
        if kill -0 "${server}" 2>/dev/null; then
            note_bad "server ${server} outlived the harness"
            kill -9 "${server}" 2>/dev/null
        fi
    done <"${CASE_HOME}/stub_bound"
}
sigterm_case
verdict

# ── The aggregate's own inputs ────────────────────────────────────────────────
#
# Two refusals the harness cannot stage from a stub server, because it builds
# one request index and reuses it for every pass. They are what stops a later
# caller assembling a report out of passes that measured different things.

agg_case() { # agg_case <name> <want-exit> <what-it-proves> <grep> <mutation-python>
    CASE_NAME="$1"
    local want="$2"
    CASE_WHAT="$3"
    local pattern="$4" mutation="$5"
    local dir="${WORK}/agg_${CASE_NAME}"
    mkdir -p "${dir}"
    CASE_OUT="${WORK}/agg_${CASE_NAME}.log"
    python3 - "${dir}" "${mutation}" <<'PY'
import json, pathlib, sys

out, mutation = pathlib.Path(sys.argv[1]), sys.argv[2]
passes = []
for number in (1, 2, 3):
    rows = [
        {"pass": number, "cell": "a@1024", "dataset": "a", "sample_id": f"a/{i}",
         "body_sha256": "x", "max_tokens": 1024, "prompt_tokens": 10,
         "completion_tokens": 20, "ttft_ms": 5.0, "decode_tps": 25.0,
         "client_decode_tps": 25.0}
        for i in range(3)
    ]
    passes.append({"pass": number, "arm": "plain", "samples": rows})
exec(mutation)
for obj in passes:
    (out / f"pass{obj['pass']}.json").write_text(json.dumps(obj))
PY
    python3 "${AGGREGATE}" report "${dir}"/pass*.json --range-pct 5 \
        --macro-max-tokens 1024 >"${CASE_OUT}" 2>&1
    local got=$?
    CASE_BAD=""
    [ "${got}" -ne "${want}" ] && CASE_BAD="exit=${got} (want ${want})"
    grep -qE "${pattern}" "${CASE_OUT}" || note_bad "missing /${pattern}/"
}

agg_case sample_dropped_from_a_pass 1 \
    "passes that measured different samples are not averaged" \
    "their ids differ" \
    'passes[1]["samples"].pop()'
verdict

agg_case cell_dropped_from_a_pass 1 \
    "passes that measured different cells are not averaged" \
    "a mean over two sample sets" \
    'passes[2]["samples"] = [dict(r, cell="b@1024") for r in passes[2]["samples"]]'
verdict

# 24.375 / 25 / 25.625 have a range of exactly 5.000% of their mean, and all
# three are exact in binary so the comparison is not decided by rounding. The
# band is `<=`, so this publishes. No fixture driven through the wire can land
# on the boundary; this one can.
agg_case range_exactly_at_the_band_publishes 0 \
    "the band is inclusive, and the boundary is where that is decided" \
    "MACRO" \
    'for n, tps in ((0, 24.375), (1, 25.0), (2, 25.625)):
    for row in passes[n]["samples"]:
        row["decode_tps"] = tps
        row["client_decode_tps"] = tps'
verdict

agg_case macro_needs_one_cell_per_dataset 1 \
    "a dataset with two cells at the macro budget is refused, not counted twice" \
    "cells at the macro budget" \
    'passes[:] = [dict(p, samples=p["samples"] + [dict(r, cell="a@1024#2",
        sample_id=r["sample_id"] + "b") for r in p["samples"]]) for p in passes]'
verdict

agg_case two_passes_is_not_three 1 \
    "a mean of any count but three is a different figure" \
    "mean of three consecutive runs" \
    'passes.pop()'
verdict

echo
echo "passed=${PASSED} failed=${FAILED}"
[ "${FAILED}" -eq 0 ] || exit 1
