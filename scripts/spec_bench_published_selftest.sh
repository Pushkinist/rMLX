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
# The host is supplied by a `ps` shim, so no case reads this machine's process
# table and no verdict here depends on what else it is doing.
# `--synthetic-arms` is the boundary between a run that measures and a run that
# exercises this logic, and it is pinned in both directions: the host gates
# still fire without it, and with it a hostile host and a quiet host reach the
# same verdict.
#
# ONE THING HERE IS STILL WALL-CLOCK SENSITIVE, and it is bounded rather than
# claimed away. The harness cross-checks the engine's decode rate against the
# same window timed at the client, and the client is `curl | python3`, so a cold
# interpreter start can slide the first content chunk's stamp forward and shrink
# the window. The stub therefore holds its first content chunk behind a prefill
# pause an order of magnitude longer than an interpreter start; the pause is
# outside the measured window by construction, so it costs wall time and biases
# nothing. Under two concurrent suites the observed slip was ~30 ms against a
# 150 ms pause.
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
# One TTFT, one ITL and one sampler event per request, warmup included.
EVENTS_PER_PASS=9

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
# `pkill` would reach real processes and the preflight deletes the Metal claim
# files of whatever is running here, which is the one thing a test must never do
# to a machine. `sleep` turns the harness's waits into nothing.
SHIMS="${WORK}/shims"
mkdir -p "${SHIMS}"
printf '#!/bin/sh\nexit 0\n' >"${SHIMS}/pkill"
# Not a no-op: the harness starts a fresh server on the same port every pass and
# its own 3 s settle is what keeps the next bind off the previous listener. 50 ms
# keeps that ordering while costing the suite seconds rather than minutes.
printf '#!/bin/sh\nexec /bin/sleep 0.05\n' >"${SHIMS}/sleep"
cat >"${SHIMS}/rm" <<'RMEOF'
#!/bin/sh
for a in "$@"; do
	case "$a" in /tmp/rmlx.*.claim) exit 0 ;; esac
done
exec /bin/rm "$@"
RMEOF
chmod +x "${SHIMS}/pkill" "${SHIMS}/sleep" "${SHIMS}/rm"

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

Streams STUB_TOKENS content chunks a nominal gap apart, after a prefill pause,
then a usage block. STUB_GAP_MS is a gap matrix: `;` separates passes, `,`
separates requests within a pass, and the last entry of each list repeats. The
chunks go out on a fixed schedule so the per-chunk framing cost falls inside the
gap rather than being added to it — the wire really does carry 1000/gap tok/s,
which is what the client cross-check compares against.

STUB_PREFILL_S is long on purpose. The client is `curl | python3`, and the
window is timed from the moment PYTHON stamps the first content chunk; a cold
interpreter start would slide that forward and shrink the window. The pause is
outside the window by construction, so holding the first content chunk behind
one an order of magnitude longer than an interpreter start costs wall time and
biases nothing.

Per request it appends to the run log the events the harness reads back:

  generate_streaming: TTFT (L6)          unless capped by STUB_TTFT_LINES
  generate: ITL stats (M30)              unless capped by STUB_ITL_LINES
  generate: host categorical sampler active (A7.2)
                                         unless STUB_SAMPLED=0, capped by
                                         STUB_SAMPLER_LINES
  <kind>_generate_greedy: done           when the serve argv carried a drafter,
                                         or STUB_FORCE_DONE is set

`mean_ms` on the ITL event and `decode_tps` on the done line both report the
nominal gap, so the engine figure is exact and the client's reading of the same
window agrees with it. STUB_ITL_MEAN_MS overrides the first, which is how a
disagreement is staged.
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def gap_matrix(raw):
    return [
        [float(v) for v in row.split(",") if v != ""]
        for row in raw.split(";")
        if row != ""
    ]


TOKENS = int(os.environ.get("STUB_TOKENS", "6"))
GAP_MS = gap_matrix(os.environ.get("STUB_GAP_MS", "") or "40")
PASS = int(os.environ.get("STUB_PASS", "1"))
PREFILL_S = float(os.environ.get("STUB_PREFILL_S", "0.15"))
PROMPT_TOKENS = int(os.environ.get("STUB_PROMPT_TOKENS", "1234"))
USAGE_TOKENS = int(os.environ.get("STUB_USAGE_TOKENS", "-1"))
TTFT_MS = float(os.environ.get("STUB_TTFT_MS", "42"))
TTFT_LINES = int(os.environ.get("STUB_TTFT_LINES", "-1"))
ITL_LINES = int(os.environ.get("STUB_ITL_LINES", "-1"))
ITL_MEAN_MS = os.environ.get("STUB_ITL_MEAN_MS", "")
SAMPLED = os.environ.get("STUB_SAMPLED", "1") == "1"
SAMPLER_LINES = int(os.environ.get("STUB_SAMPLER_LINES", "-1"))
SAMPLER_TOP_K = int(os.environ.get("STUB_SAMPLER_TOP_K", "20"))
SAMPLER_TOP_K_PASS2 = os.environ.get("STUB_SAMPLER_TOP_K_PASS2", "")
SAMPLER_TOP_K_AFTER = os.environ.get("STUB_SAMPLER_TOP_K_AFTER", "")
OMIT_COMPLETION_TOKENS = os.environ.get("STUB_OMIT_COMPLETION_TOKENS", "") == "1"
LOG_PATH = os.environ.get("STUB_LOG", "")
SPECULATIVE = os.environ.get("STUB_SPECULATIVE", "") == "1"
FORCE_DONE = os.environ.get("STUB_FORCE_DONE", "") == "1"
CHARGED = json.loads(os.environ.get("STUB_CHARGED", "false"))
DROP_FIELDS = [f for f in os.environ.get("STUB_DROP_FIELDS", "").split(",") if f]
BODIES = os.environ.get("STUB_BODIES", "")
BOUND_FLAG = os.environ.get("STUB_BOUND_FLAG", "")

served = 0


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
        "message": "mtp_generate_greedy: done",
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
        "prefill_ms": 100.0,
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

    def do_GET(self):
        self._json({"object": "list", "data": []})

    def do_POST(self):
        global served
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if BODIES:
            with open(BODIES, "ab") as handle:
                handle.write(body + b"\n")

        this_gap_ms = gap_ms()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(payload):
            self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.flush()

        chunk({"choices": [{"delta": {"role": "assistant"}, "index": 0}]})
        time.sleep(PREFILL_S)
        start = time.monotonic()
        for i in range(TOKENS):
            due = start + i * this_gap_ms / 1000.0
            remaining = due - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            chunk({"choices": [{"delta": {"content": f"t{i} "}, "index": 0}]})
        usage = {}
        if not OMIT_COMPLETION_TOKENS:
            usage["completion_tokens"] = USAGE_TOKENS if USAGE_TOKENS >= 0 else TOKENS
        if PROMPT_TOKENS >= 0:
            usage["prompt_tokens"] = PROMPT_TOKENS
        chunk({"choices": [], "usage": usage})

        if SAMPLED and (SAMPLER_LINES < 0 or served < SAMPLER_LINES):
            top_k = SAMPLER_TOP_K
            if SAMPLER_TOP_K_PASS2 and PASS >= 2:
                top_k = int(SAMPLER_TOP_K_PASS2)
            if SAMPLER_TOP_K_AFTER and served >= 2:
                top_k = int(SAMPLER_TOP_K_AFTER)
            log(event({"message": "generate: host categorical sampler active (A7.2)",
                       "model_id": "stub", "temperature": 0.6, "top_p": 0.95,
                       "top_k": top_k, "min_p": 0.0, "seed": 42919}))
        if TTFT_LINES < 0 or served < TTFT_LINES:
            log(event({"message": "generate_streaming: TTFT (L6)",
                       "model_id": "stub", "ttft_ms": TTFT_MS}))
        if ITL_LINES < 0 or served < ITL_LINES:
            mean = float(ITL_MEAN_MS) if ITL_MEAN_MS else this_gap_ms
            log(event({"message": "generate: ITL stats (M30)", "model_id": "stub",
                       "step_count": TOKENS, "p50_ms": mean, "p95_ms": mean,
                       "p99_ms": mean, "mean_ms": mean, "itl_spikes": 0}))
        if SPECULATIVE or FORCE_DONE:
            log(done_line(round(1000.0 / this_gap_ms, 6)))
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

STUB="${FAKE_ROOT}/target/release-perf/rmlx"
cat >"${STUB}" <<STUBEOF
#!/usr/bin/env bash
set -eu
case "\$1" in
metrics)
	case "\$2" in
	identity)
		echo '{"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb"}'
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
	# Which pass this is. The harness starts one server per pass and the stub
	# has to vary its rate across them, so the count lives beside the logs.
	pass=\$(( \$(cat "\$RMLX_HOME/pass.cnt" 2>/dev/null || echo 0) + 1 ))
	echo "\$pass" >"\$RMLX_HOME/pass.cnt"
	log="\$RMLX_HOME/logs/\$(date +%s)-\$\$-p\$pass.jsonl"
	: >"\$log"
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
	export STUB_LOG="\$log" STUB_SPECULATIVE="\$speculative" STUB_PASS="\$pass"
	exec python3 "${SERVER_PY}" "\$port"
	;;
esac
STUBEOF
chmod +x "${STUB}"
ln -s "${STUB}" "${CANON_ROOT}/target/release-perf/rmlx"

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
#          [HOST:quiet|hostile|flaky] [VERIFIER:dir] [MEASURED:1] [CANONICAL:1]
#          [GREP:pat ...] [NOGREP:pat ...]
#
# Defaults: a quiet host, --synthetic-arms, the healthy verifier snapshot, and
# the shrunken sample sets reached through --samples-root — the unverified
# operator override, which is the only way a root that is not the published one
# can be measured at all. CANONICAL:1 drops the flag and runs out of a repo root
# whose prompts/published IS the shrunken copy, which is how the default path is
# shown to verify.
#
# The stub streams six tokens 40 ms apart behind a 150 ms prefill — 25 tok/s on
# the wire and 25 reported by the engine.
run_case() {
    CASE_NAME="$1"
    local want="$2"
    CASE_WHAT="$3"
    shift 3

    CASE_HOME="${WORK}/home_${CASE_NAME}"
    mkdir -p "${CASE_HOME}/logs"

    local env_pairs=() greps=() nogreps=() extra_args=() host="quiet" a
    local synthetic=true verifier="${VERIFIER}" canonical=false
    for a in "$@"; do
        case "$a" in
        GREP:*) greps+=("${a#GREP:}") ;;
        NOGREP:*) nogreps+=("${a#NOGREP:}") ;;
        ARGS:*) extra_args+=("${a#ARGS:}") ;;
        HOST:*) host="${a#HOST:}" ;;
        MEASURED:*) synthetic=false ;;
        CANONICAL:*) canonical=true ;;
        VERIFIER:*) verifier="${a#VERIFIER:}" ;;
        *) env_pairs+=("$a") ;;
        esac
    done
    $synthetic && extra_args+=(--synthetic-arms)
    local root="${FAKE_ROOT}"
    if $canonical; then
        root="${CANON_ROOT}"
    else
        extra_args+=(--samples-root "${SAMPLES}")
    fi

    CASE_OUT="${WORK}/${CASE_NAME}.log"
    local got=0
    env -i \
        PATH="${WORK}/${host}:${SHIMS}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
        HOME="${WORK}" \
        RMLX_HOME="${CASE_HOME}" \
        STUB_BOUND_FLAG="${CASE_HOME}/stub_bound" \
        STUB_BODIES="${CASE_HOME}/bodies.jsonl" \
        ${env_pairs[@]+"${env_pairs[@]}"} \
        bash "${root}/scripts/spec_bench_published.sh" \
        "${verifier}" --port "${PORT}" \
        ${extra_args[@]+"${extra_args[@]}"} >"${CASE_OUT}" 2>&1
    got=$?
    pkill -f "${SERVER_PY}" 2>/dev/null || true

    CASE_BAD=""
    [ "${got}" -ne "${want}" ] && CASE_BAD="exit=${got} (want ${want})"
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
    hits=$(echo "${CASE_HOME}"/bench/spec_bench_published/*.json)
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
    'STUB_GAP_MS=40,200,100,100,50,50,50,25,25' \
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
# request index 1 is all of it: 23.5 / 25 / 26.5 tok/s across the passes is a
# 12% range there and leaves every other cell flat.
#
# This is also the empirical half of why the macro has no refusal of its own.
# The macro's OWN range here is 4% — inside the band — while the dataset it is
# built from is refused at 12%. A macro range test would have passed this run
# and published the headline. Withholding it because a dataset was refused is
# what makes the MACRO row say UNSTABLE.
ONE_CELL="$(python3 -c 'g = lambda r: f"{1000/r:.9f}"
print(";".join(["40," + g(23.5) + ",40,40,40,40,40,40,40",
                "40",
                "40," + g(26.5) + ",40,40,40,40,40,40,40"]))')"

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
    python3 - "${CASE_HOME}/bodies.jsonl" "${SAMPLES}" "$1" <<'PY'
import hashlib, json, pathlib, sys

bodies = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
root = pathlib.Path(sys.argv[2])
digests = set()
for entry in json.loads((root / "manifest.json").read_text())["datasets"]:
    for sample in json.loads((root / entry["file"]).read_text())["samples"]:
        digests.add(sample["body_sha256"])


def address(messages):
    raw = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# The warmup carries its own token budget, so the two are told apart by it.
warmups = [b for b in bodies if b["max_tokens"] == 64]
measured = [b for b in bodies if b["max_tokens"] != 64]

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
# The warmup is a 64-token request on a prompt outside the sample sets, so the
# budgets sent are its own plus the two the protocol pins.
[ "$(bodies_check "sorted({b['max_tokens'] for b in bodies})")" = "[64, 1024, 4096]" ] ||
    note_bad "max_tokens sent: $(bodies_check "sorted({b['max_tokens'] for b in bodies})")"
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
# `measured` excludes the warmup, whose 64-token budget is its own. That the
# warmup's prompt is in no sample set is the point of it: an untimed request on
# a measured sample would leave that one sample facing a warm prompt cache.
[ "$(bodies_check "all(address(b['messages']) in digests for b in measured)")" = "True" ] ||
    note_bad "a measured request's messages no longer hash to any checked-in body_sha256"
[ "$(bodies_check "all(list(m) == ['role', 'content'] for b in bodies for m in b['messages'])")" = "True" ] ||
    note_bad "a message was re-emitted with its keys in another order"
[ "$(bodies_check "any(address(b['messages']) in digests for b in warmups)")" = "False" ] ||
    note_bad "the warmup re-used a measured sample's prompt"
[ "$(bodies_check "len(warmups)")" = "3" ] ||
    note_bad "$(bodies_check "len(warmups)") warmups for three passes"
verdict

# Thinking tokens are generated tokens and the server counts them; the harness
# records the count the server reported rather than the chunks it saw.
run_case completion_tokens_are_the_servers_count 0 \
    "the recorded output length is the count the server reported" \
    'STUB_USAGE_TOKENS=97'
[ "$(jq_of "sorted({s['completion_tokens'] for s in r['samples']})")" = "[97]" ] ||
    note_bad "completion_tokens=$(jq_of "sorted({s['completion_tokens'] for s in r['samples']})")"
[ "$(jq_of "sorted({s['prompt_tokens'] for s in r['samples']})")" = "[1234]" ] ||
    note_bad "prompt_tokens=$(jq_of "sorted({s['prompt_tokens'] for s in r['samples']})")"
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
    'STUB_ITL_MEAN_MS=20' \
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

# ── An aborted run ────────────────────────────────────────────────────────────
#
# Three servers run per invocation, and a SIGTERM to the harness — a CI timeout,
# an operator, a parent tearing down — must not leave one alive holding the
# Metal claim. The next run's preflight would not find it: `pkill -f "rmlx
# serve"` does not match a snapshotted binary.
sigterm_case() {
    CASE_NAME="sigterm_leaves_no_server_running"
    CASE_WHAT="a killed run takes its server with it"
    CASE_HOME="${WORK}/home_${CASE_NAME}"
    CASE_OUT="${WORK}/${CASE_NAME}.log"
    CASE_BAD=""
    mkdir -p "${CASE_HOME}/logs"

    env -i \
        PATH="${WORK}/quiet:${SHIMS}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
        HOME="${WORK}" \
        RMLX_HOME="${CASE_HOME}" \
        STUB_BOUND_FLAG="${CASE_HOME}/stub_bound" \
        STUB_PREFILL_S=5 \
        bash "${FAKE_ROOT}/scripts/spec_bench_published.sh" \
        "${VERIFIER}" --port "${PORT}" --samples-root "${SAMPLES}" \
        --synthetic-arms >"${CASE_OUT}" 2>&1 &
    local harness=$!

    local waited=0
    while [ ! -s "${CASE_HOME}/stub_bound" ] && [ "${waited}" -lt 200 ]; do
        /bin/sleep 0.1
        waited=$((waited + 1))
    done
    if [ ! -s "${CASE_HOME}/stub_bound" ]; then
        note_bad "the stub never bound; nothing was killed"
        kill "${harness}" 2>/dev/null
        return
    fi
    local server
    server="$(head -1 "${CASE_HOME}/stub_bound")"

    kill -TERM "${harness}" 2>/dev/null
    wait "${harness}" 2>/dev/null

    waited=0
    while kill -0 "${server}" 2>/dev/null && [ "${waited}" -lt 50 ]; do
        /bin/sleep 0.1
        waited=$((waited + 1))
    done
    if kill -0 "${server}" 2>/dev/null; then
        note_bad "server ${server} outlived the harness"
        kill -9 "${server}" 2>/dev/null
    fi
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
