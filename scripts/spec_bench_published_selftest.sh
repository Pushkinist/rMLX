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
# The host is supplied by a `ps` shim, so no case reads this machine and the
# file is deterministic under any load. `--synthetic-arms` is the boundary
# between a run that measures and a run that exercises this logic, and it is
# pinned in both directions: the host gates still fire without it, and with it a
# hostile host and a quiet host reach the same verdict.
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
# A shrunken copy of the checked-in sets: the first few samples of each dataset,
# with the manifest re-derived around them so `published_samples.py verify`
# still passes. Different sizes on purpose — 2, 3 and 4 — so the macro average
# and a pooled mean over the same rows are different numbers.
SAMPLES="${WORK}/samples"
mkdir -p "${SAMPLES}"
python3 - "${PUBLISHED}" "${SAMPLES}" <<'PY' || exit 2
import hashlib, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(sys.argv[0]).resolve().parent))
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
KEEP = {"mt_bench": 2, "math_500": 3, "humaneval": 4}

man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
for entry in man["datasets"]:
    keep = KEEP[entry["key"]]
    doc = json.loads((src / entry["file"]).read_text(encoding="utf-8"))
    entry["selected_ids"] = entry["selected_ids"][:keep]
    entry["count"] = keep
    if entry["sampling"]["mode"] == "all":
        # The whole pool is the selection, so the pool has to shrink with it.
        entry["pool_ids"] = entry["pool_ids"][:keep]
        entry["pool_size"] = keep
    chosen = set(entry["selected_ids"])
    doc["samples"] = [s for s in doc["samples"] if s["source_id"] in chosen]
    assert len(doc["samples"]) == keep, entry["key"]
    blob = (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (dst / entry["file"]).write_bytes(blob)
    entry["file_bytes"] = len(blob)
    entry["file_sha256"] = hashlib.sha256(blob).hexdigest()
(dst / "manifest.json").write_text(
    json.dumps(man, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
python3 "${REPO_ROOT}/scripts/published_samples.py" verify --root "${SAMPLES}" \
    >/dev/null || { echo "ERROR: the shrunken sample sets do not verify" >&2; exit 2; }

# 2 + 3 + 4 + 3 (MATH-500 again at 4096) requests per pass.
REQUESTS_PER_PASS=12

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
printf '%s\n' '{"temperature": 0.6}' >"${HALF_DEFAULTS}/generation_config.json"

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
mkdir -p "${WORK}/quiet" "${WORK}/hostile"
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
chmod +x "${WORK}/quiet/ps" "${WORK}/hostile/ps"

# ── Stub server ───────────────────────────────────────────────────────────────

SERVER_PY="${WORK}/stub_server.py"
cat >"${SERVER_PY}" <<'PYEOF'
"""Canned OpenAI-compatible server for the published-harness selftest.

Streams STUB_TOKENS content chunks a nominal gap apart, after a prefill pause,
then a usage block. The gap for request `i` of pass `p` is
`GAP_MS_SEQ[i] * PASS_SCALE[p - 1]`, both comma-separated with the last entry
repeating, and the chunks go out on a fixed schedule so the per-chunk framing
cost falls inside the gap rather than being added to it — the wire really does
carry 1000/gap tok/s, which is what the client cross-check compares against.

Per request it appends to the run log the events the harness reads back:

  generate_streaming: TTFT (L6)   always, unless capped by STUB_TTFT_LINES
  generate: ITL stats (M30)       always, unless capped by STUB_ITL_LINES
  <kind>_generate_greedy: done    when the serve argv carried a drafter, or
                                  STUB_FORCE_DONE is set

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


def seq(name, default):
    raw = os.environ.get(name, "") or default
    return [float(v) for v in raw.split(",") if v != ""]


TOKENS = int(os.environ.get("STUB_TOKENS", "6"))
GAP_MS_SEQ = seq("STUB_GAP_MS_SEQ", "40")
PASS_SCALE = seq("STUB_PASS_SCALE", "1")
PASS = int(os.environ.get("STUB_PASS", "1"))
PREFILL_S = float(os.environ.get("STUB_PREFILL_S", "0.02"))
PROMPT_TOKENS = int(os.environ.get("STUB_PROMPT_TOKENS", "1234"))
USAGE_TOKENS = int(os.environ.get("STUB_USAGE_TOKENS", "-1"))
TTFT_MS = float(os.environ.get("STUB_TTFT_MS", "42"))
TTFT_LINES = int(os.environ.get("STUB_TTFT_LINES", "-1"))
ITL_LINES = int(os.environ.get("STUB_ITL_LINES", "-1"))
ITL_MEAN_MS = os.environ.get("STUB_ITL_MEAN_MS", "")
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
    return pick(GAP_MS_SEQ, served) * pick(PASS_SCALE, PASS - 1)


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
        usage = {"completion_tokens": USAGE_TOKENS if USAGE_TOKENS >= 0 else TOKENS}
        if PROMPT_TOKENS >= 0:
            usage["prompt_tokens"] = PROMPT_TOKENS
        chunk({"choices": [], "usage": usage})

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
#          [HOST:quiet|hostile] [GREP:pat ...] [NOGREP:pat ...]
#
# Defaults: a quiet host, --synthetic-arms, the shrunken sample sets and the
# healthy verifier snapshot, streaming six tokens 40 ms apart — 25 tok/s on
# the wire and 25 reported by the engine.
run_case() {
    CASE_NAME="$1"
    local want="$2"
    CASE_WHAT="$3"
    shift 3

    CASE_HOME="${WORK}/home_${CASE_NAME}"
    mkdir -p "${CASE_HOME}/logs"

    local env_pairs=() greps=() nogreps=() extra_args=() host="quiet" a
    local synthetic=true verifier="${VERIFIER}"
    for a in "$@"; do
        case "$a" in
        GREP:*) greps+=("${a#GREP:}") ;;
        NOGREP:*) nogreps+=("${a#NOGREP:}") ;;
        ARGS:*) extra_args+=("${a#ARGS:}") ;;
        HOST:*) host="${a#HOST:}" ;;
        MEASURED:*) synthetic=false ;;
        VERIFIER:*) verifier="${a#VERIFIER:}" ;;
        *) env_pairs+=("$a") ;;
        esac
    done
    $synthetic && extra_args+=(--synthetic-arms)

    CASE_OUT="${WORK}/${CASE_NAME}.log"
    local got=0
    env -i \
        PATH="${WORK}/${host}:${SHIMS}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
        HOME="${WORK}" \
        RMLX_HOME="${CASE_HOME}" \
        STUB_BOUND_FLAG="${CASE_HOME}/stub_bound" \
        STUB_BODIES="${CASE_HOME}/bodies.jsonl" \
        ${env_pairs[@]+"${env_pairs[@]}"} \
        bash "${FAKE_ROOT}/scripts/spec_bench_published.sh" \
        "${verifier}" --port "${PORT}" --samples-root "${SAMPLES}" \
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
[ "$(jq_of "r['cells']['mt_bench@1024']['samples']")" = "2" ] ||
    note_bad "mt_bench@1024 counted $(jq_of "r['cells']['mt_bench@1024']['samples']") samples"
[ "$(jq_of "r['cells']['humaneval@1024']['samples']")" = "4" ] ||
    note_bad "humaneval@1024 counted $(jq_of "r['cells']['humaneval@1024']['samples']") samples"
verdict
# Three passes are what the protocol asks for and what the result has to hold.
run_case three_passes_of_every_sample 0 \
    "the result holds three passes of every sample and nothing else"
[ "$(jq_of "len(r['samples'])")" = "36" ] ||
    note_bad "the result holds $(jq_of "len(r['samples'])") rows (want 3 x 12)"
[ "$(jq_of "sorted({s['pass'] for s in r['samples']})")" = "[1, 2, 3]" ] ||
    note_bad "passes=$(jq_of "sorted({s['pass'] for s in r['samples']})")"
verdict

# The macro average is the mean of the dataset means. Pooling the rows instead
# weights MT-Bench's two samples below HumanEval's four, and here the two
# answers are 32.5 and 35.
run_case macro_is_the_mean_of_the_dataset_means 0 \
    "the macro average does not weight a dataset by its size" \
    'STUB_GAP_MS_SEQ=40,100,100,25,25,25,25,25,25,25,25,25,25'
close_to "$(jq_of "r['cells']['mt_bench@1024']['mean']")" 10.0 0.001 ||
    note_bad "mt_bench mean=$(jq_of "r['cells']['mt_bench@1024']['mean']") (want 10)"
close_to "$(jq_of "r['cells']['humaneval@1024']['mean']")" 40.0 0.001 ||
    note_bad "humaneval mean=$(jq_of "r['cells']['humaneval@1024']['mean']") (want 40)"
got="$(jq_of "r['macro']['mean']")"
close_to "${got}" 32.5 0.001 ||
    note_bad "macro=${got} (want 32.5; a pooled mean would be 35.0)"
verdict

# ── The run-to-run range refusal ──────────────────────────────────────────────
#
# Both sides of the band, on the same fixture: the scales put the three pass
# rates at 24.3875 / 25 / 25.6125 (range 4.9% of the mean) and at
# 24.36 / 25 / 25.64 (5.12%). A gate proven to fire but never proven to hold
# its tongue is a gate that could be firing on everything.
UNDER_BAND="$(python3 -c 'print(",".join(f"{25/r:.9f}" for r in (24.3875, 25, 25.6125)))')"
OVER_BAND="$(python3 -c 'print(",".join(f"{25/r:.9f}" for r in (24.36, 25, 25.64)))')"

run_case range_just_under_the_band_is_a_clean_mean 0 \
    "a spread just inside the band is published as a mean" \
    "STUB_PASS_SCALE=${UNDER_BAND}" \
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
    "STUB_PASS_SCALE=${OVER_BAND}" \
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


print(eval(sys.argv[3]))
PY
}

# The published sampling is the checkpoint's own, which is what sending no
# sampling field at all means. A temperature spelled here would be a second
# copy of a fact the snapshot owns, and the one that got published.
run_case sampling_fields_are_not_sent 0 \
    "no request carries a sampling parameter of this script's" \
    'GREP:sampling   : model defaults'
for field in temperature top_p top_k seed repetition_penalty; do
    [ "$(bodies_check "any('${field}' in b for b in bodies)")" = "False" ] ||
        note_bad "a request carried ${field}"
done
[ "$(bodies_check "all(b['enable_thinking'] is True for b in bodies)")" = "True" ] ||
    note_bad "a request did not ask for thinking"
[ "$(bodies_check "sorted({b['max_tokens'] for b in bodies})")" = "[1024, 4096]" ] ||
    note_bad "max_tokens sent: $(bodies_check "sorted({b['max_tokens'] for b in bodies})")"
verdict

# The workspace builds serde_json with preserve_order, so a message re-emitted
# as {content, role} has a different content address than the {role, content}
# that was checked in — and a later join on it splits with nothing saying so.
run_case messages_keep_their_content_address 0 \
    "every request's messages still hash to the checked-in address"
[ "$(bodies_check "all(address(b['messages']) in digests for b in bodies)")" = "True" ] ||
    note_bad "a request's messages no longer hash to any checked-in body_sha256"
[ "$(bodies_check "all(list(m) == ['role', 'content'] for b in bodies for m in b['messages'])")" = "True" ] ||
    note_bad "a message was re-emitted with its keys in another order"
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
    'STUB_TTFT_LINES=5' \
    'GREP:TTFT events, expected 13'
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
    'STUB_ITL_LINES=7' \
    'GREP:ITL stats events, expected 13'
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

TAMPERED="${WORK}/tampered"
cp -R "${SAMPLES}" "${TAMPERED}"
python3 - "${TAMPERED}" <<'PY' || exit 2
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
path = root / "mt_bench.json"
doc = json.loads(path.read_text(encoding="utf-8"))
doc["samples"][0]["messages"][0]["content"] += " (edited)"
path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

run_case tampered_sample_set_refused 1 \
    "a sample file that no longer re-derives is not measured against" \
    "ARGS:--samples-root" "ARGS:${TAMPERED}" \
    'GREP:do not re-derive from'
no_result
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
    "12 requests over 4 cells" \
    mt_bench:1024 math_500:1024 humaneval:1024 math_500:4096
verdict

run_case snapshot_without_sampling_defaults_refused 1 \
    "a checkpoint that states no sampling defaults is not measured" \
    "VERIFIER:${NO_DEFAULTS}" \
    'GREP:hard-coded temperature 1.0'
no_result
verdict

run_case snapshot_with_half_the_defaults_refused 1 \
    "a checkpoint that states only some of them is refused too" \
    "VERIFIER:${HALF_DEFAULTS}" \
    'GREP:states no top_p'
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
# would make this whole suite green.
run_case synthetic_arms_waives_no_arm_reading_guard 1 \
    "the flag does not waive a guard that reads the run" \
    HOST:hostile 'STUB_TTFT_LINES=5' \
    'GREP:TTFT events, expected 13'
no_result
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
        >"${CASE_OUT}" 2>&1
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

agg_case two_passes_is_not_three 1 \
    "a mean of any count but three is a different figure" \
    "mean of three consecutive runs" \
    'passes.pop()'
verdict

echo
echo "passed=${PASSED} failed=${FAILED}"
[ "${FAILED}" -eq 0 ] || exit 1
