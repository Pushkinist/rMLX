#!/usr/bin/env bash
# schema_constraint_canary.sh — real-model proof for the `json_schema` constraint path.
#
# Two probes on two architectures, verdict fixed before the run.
#
#   Probe A — "plain key". Object root, one required enum property named `unit`.
#             The shape the issue was filed with. Kept as a no-regression check.
#   Probe B — "spaced key". Same, but the property is named `unit label`.
#             The space is key CONTENT. A grammar that treats whitespace inside
#             a key string as an insignificant separator parks the key trie on
#             the space it is still expecting; every candidate token carrying
#             the rest of the name is then rejected at its second byte, while a
#             whitespace-only token is accepted as a no-op. The mask offers
#             whitespace and nothing else, forever, and withholds EOS because
#             the value is incomplete. That is forced by the mask, so unlike
#             Probe A it does not depend on the model preferring whitespace.
#
# ─────────────────────────────────────────────────────────────────────────────
# DECISION RULE — read this before running, not after.
# ─────────────────────────────────────────────────────────────────────────────
#
# Per (model, probe) cell, the probe PASSES iff ALL FIVE hold:
#
#   R1  HTTP status is 200.
#   R2  finish_reason == "stop".  A degenerate run cannot stop on EOS — the
#       mask withholds EOS until the value is complete — so it always ends at
#       max_tokens with "length".
#   R3  The payload (content, else reasoning_content) parses as JSON after
#       stripping an optional ```json fence, and VALIDATES against that probe's
#       schema: object, exactly the one expected key, value in the enum.
#   R4  The emitted token-id stream is not degenerate. Degenerate := the tail
#       is periodic with period <= 4 for >= 16 tokens (a constant stream is the
#       period-1 case). Ids come from the run's own log, filtered to this
#       probe's `X-Request-Id`.
#         * Fewer than 16 emitted tokens -> reported "n/a", counts as PASS: a
#           stream that short cannot be degenerate, and R2 already catches the
#           long-run case. It is printed as n/a so nobody reads it as evidence.
#         * Log records present but no parsable token_id, or no step_fn records
#           at all -> HARNESS ERROR, counts as FAIL. A rule that cannot fire
#           must never report PASS.
#   R5  The run log contains a `building SchemaConstraint` line AND a
#       `SchemaConstraint: ... engaging` line, and does NOT contain
#       `constraint never engaged` — all filtered to this probe's request id.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THE BASELINE MUST PRODUCE  (`--expect baseline`)
# ─────────────────────────────────────────────────────────────────────────────
#
# Not "everything fails". The expectation is per cell, and is what was actually
# observed at the pre-fix commit — a cell expected to PASS on the baseline is
# recorded here so that a harness which starts failing it says so.
#
#   bonsai / A   MUST FAIL — the template prefills a CLOSED think block, the
#                splitter starts open, `is_thinking` latches, and the engage
#                gate never fires. Observed: R3 (`unit='Celsius'`, outside the
#                enum) and R5 (`engaging_line=False`).
#   bonsai / B   MUST FAIL — same gate, and the key-string defect underneath it.
#   gemma / A    MUST PASS — **the reported whitespace loop does not reproduce
#                on this probe.** Observed at the pre-fix commit: byte-identical
#                to the fixed arm, `{\n  "unit": "celsius"\n}`, finish_reason
#                stop, 16 tokens, R4 clean. gemma is not thinking-capable so the
#                engage gate is not involved, and with a single-word key the
#                grammar never corners the decoder. Whether it loops on Probe A
#                is then a matter of which token the model prefers at a
#                structural position, which this model does not.
#   gemma / B    MUST FAIL — the mask corners the decoder regardless of
#                preference.
#
#   `--expect fixed`: all four cells MUST PASS.
#
# A fix that does nothing fails `--expect fixed`. A harness too weak to see the
# defect fails `--expect baseline`. A defect that starts reproducing where it
# did not, or stops where it did, also fails `--expect baseline` — the table is
# a claim about the world, not a fudge factor.
#
# Note R4 is deliberately not "N consecutive identical ids": a 2-periodic
# stream has a maximum consecutive-identical run of 1 and sails through that
# weaker check.
#
# ─────────────────────────────────────────────────────────────────────────────
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#
#   make build-perf
#   bash scripts/schema_constraint_canary.sh                  # --expect fixed
#   bash scripts/schema_constraint_canary.sh --expect baseline
#   bash scripts/schema_constraint_canary.sh --model bonsai   # one model only
#
# Requires:
#   - target/release-perf/rmlx  (make build-perf)
#   - RMLX_O_MODELS_ROOT pointing at the snapshot root (resolve via LOCAL.md),
#     or explicit BONSAI_MODEL / GEMMA_E2B_MODEL paths.
#   - Exclusive GPU: one MLX process per Mac. The script preflights strays.
#
# Writes nothing to the real metrics DB: hermetic RMLX_HOME under
# .rmlx/proofs/schema-constraint, `--metrics off` on every server. One server
# per model serves both probes; they are separated in the log by the
# `X-Request-Id` each request carries.
#
# Artifacts per (model, probe), under that hermetic root:
#   <probe>.request.json / .response.json / .status.txt / .verdict.txt
#   serve.log, logs/*.jsonl   (shared per model)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="${REPO_ROOT}/target/release-perf/rmlx"
PORT="${PORT:-62265}"
ROOT_DIR="${REPO_ROOT}/.rmlx/proofs/schema-constraint"

O_MODELS_ROOT="${RMLX_O_MODELS_ROOT:-}"
BONSAI_MODEL="${BONSAI_MODEL:-${O_MODELS_ROOT:+${O_MODELS_ROOT}/prism-ml__Ternary-Bonsai-8B-mlx-2bit}}"
GEMMA_E2B_MODEL="${GEMMA_E2B_MODEL:-${O_MODELS_ROOT:+${O_MODELS_ROOT}/mlx-community__gemma-4-e2b-it-mxfp8}}"

EXPECT="fixed"
ONLY_MODEL="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expect) EXPECT="$2"; shift 2 ;;
        --model)  ONLY_MODEL="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        -h|--help) sed -n '1,105p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
case "${EXPECT}" in
    fixed|baseline) ;;
    *) echo "--expect must be 'fixed' or 'baseline'" >&2; exit 2 ;;
esac

# ── Baseline expectation table (see the header) ──────────────────────────────
# Cells listed here are expected to PASS even on the baseline arm.
baseline_expects_pass() {
    [[ "$1" == "gemma4-e2b" && "$2" == "A" ]]
}

# ── The probe requests ───────────────────────────────────────────────────────
# Object root (engage policy = ValueStarter, the shape that failed), one
# required enum property, additionalProperties:false, strict, temperature 0.
# max_tokens is generous on purpose: a bounded grammar must stop well short of
# it, and an unbounded one must be given room to prove that it does not.

probe_key() {
    case "$1" in
        A) printf 'unit' ;;
        B) printf 'unit label' ;;
    esac
}

probe_body() {
    # $1 = probe id, $2 = model id
    local key
    key="$(probe_key "$1")"
    python3 - "$key" "$2" <<'PY'
import json, sys
key, model = sys.argv[1], sys.argv[2]
print(json.dumps({
    "model": model,
    "messages": [{
        "role": "user",
        "content": "What unit is 25 degrees Celsius measured in? "
                   "Answer with the JSON object only.",
    }],
    "temperature": 0,
    "seed": 0,
    "max_tokens": 256,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "unit_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {key: {"type": "string",
                                     "enum": ["celsius", "fahrenheit"]}},
                "required": [key],
                "additionalProperties": False,
            },
        },
    },
}, indent=2))
PY
}

# ── Preflight ────────────────────────────────────────────────────────────────

preflight() {
    pkill -f "rmlx serve"      2>/dev/null || true
    pkill -f "rmlx_main serve" 2>/dev/null || true
    pkill -f mlx_lm            2>/dev/null || true
    pkill -f paroquant         2>/dev/null || true
    pkill -f omlx              2>/dev/null || true
    sleep 3
    rm -f /tmp/rmlx.*.claim 2>/dev/null || true
}

wait_for_server() {
    local pid="$1" attempts=0
    while (( attempts < 180 )); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "ERROR: server pid ${pid} exited during startup" >&2
            return 1
        fi
        if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        attempts=$((attempts + 1))
        sleep 2
    done
    echo "ERROR: server not ready after 360s" >&2
    return 1
}

# ── Verdict evaluation ───────────────────────────────────────────────────────

evaluate() {
    # $1 = artifact dir, $2 = probe id, $3 = expected key, $4 = request id
    python3 - "$1" "$2" "$3" "$4" <<'PY'
import json, os, re, sys, glob

d, probe, want_key, rid = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

def read(p):
    try:
        with open(p) as f: return f.read()
    except OSError:
        return ""

body_raw = read(os.path.join(d, f"{probe}.response.json"))
status = read(os.path.join(d, f"{probe}.status.txt")).strip()

results = []   # (name, state, detail)  state in {PASS, FAIL, N/A}

# R1 — HTTP 200
try:
    body = json.loads(body_raw)
    results.append(("R1_http_200", "PASS" if status == "200" else "FAIL",
                    f"status={status or 'none'}"))
except Exception as e:
    body = None
    results.append(("R1_http_200", "FAIL",
                    f"status={status or 'none'} body_unparseable: {e}"))

msg, finish = {}, None
if isinstance(body, dict):
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    finish = ch.get("finish_reason")

# R2 — finish_reason
results.append(("R2_finish_stop", "PASS" if finish == "stop" else "FAIL",
                f"finish_reason={finish!r}"))

# R3 — payload validates against this probe's schema
payload = msg.get("content") or msg.get("reasoning_content") or ""
stripped = re.sub(r"\s*```$", "", re.sub(r"^```(?:json)?\s*", "", payload.strip()))
ok, why = False, ""
try:
    v = json.loads(stripped)
    if not isinstance(v, dict):
        why = f"payload is {type(v).__name__}, not an object"
    elif set(v.keys()) != {want_key}:
        why = f"keys={sorted(v.keys())} (schema allows exactly [{want_key!r}])"
    elif v[want_key] not in ("celsius", "fahrenheit"):
        why = f"{want_key}={v[want_key]!r} not in enum"
    else:
        ok, why = True, f"payload={v}"
except Exception as e:
    why = f"not JSON: {e}; payload[:120]={stripped[:120]!r}"
results.append(("R3_schema_valid", "PASS" if ok else "FAIL", why))

# ── Log slice for THIS request ───────────────────────────────────────────────
# The route's `request_id` span is carried across `spawn_blocking`, so every
# decode-thread record for this probe carries the id we sent. A raw substring
# match keeps the filter independent of the JSON layout.
lines, records_seen = [], 0
for lf in sorted(glob.glob(os.path.join(d, "logs", "*.jsonl"))):
    with open(lf) as f:
        for line in f:
            if rid in line:
                lines.append(line)

# R4 — degeneracy over emitted token ids
ids, step_records = [], 0
unparsable = 0
for line in lines:
    if "step_fn sending token" not in line:
        continue
    step_records += 1
    try:
        rec = json.loads(line)
    except Exception:
        unparsable += 1
        continue
    tid = (rec.get("fields") or rec).get("token_id")
    if tid is None:
        unparsable += 1
    else:
        ids.append(int(tid))

def degenerate_tail(seq, max_period=4, min_len=16):
    if len(seq) < min_len:
        return None
    for p in range(1, max_period + 1):
        n, i = 0, len(seq) - 1
        while i - p >= 0 and seq[i] == seq[i - p]:
            n += 1
            i -= 1
        if n + p >= min_len:
            return (p, n + p)
    return None

if step_records == 0:
    results.append(("R4_not_degenerate", "FAIL",
                    "HARNESS ERROR: no `step_fn sending token` records for this "
                    "request id — rerun the server with `--log verbose`"))
elif not ids:
    results.append(("R4_not_degenerate", "FAIL",
                    f"HARNESS ERROR: {step_records} step_fn records found but "
                    f"none carried a parsable token_id ({unparsable} unparsable) "
                    "— the log's JSON shape does not match this extractor"))
else:
    deg = degenerate_tail(ids)
    if deg:
        p, ln = deg
        results.append(("R4_not_degenerate", "FAIL",
                        f"tail is {p}-periodic for {ln} tokens; last12={ids[-12:]}"))
    elif len(ids) < 16:
        results.append(("R4_not_degenerate", "N/A",
                        f"only {len(ids)} tokens — too short for the rule to "
                        f"fire; ids={ids}"))
    else:
        results.append(("R4_not_degenerate", "PASS",
                        f"{len(ids)} tokens, last12={ids[-12:]}"))

# R5 — constraint built AND engaged, no non-enforcement warn
blob = "".join(lines)
built = "building SchemaConstraint" in blob
engaged = re.search(r"SchemaConstraint:[^\"]*engaging", blob) is not None
warned = "constraint never engaged" in blob
results.append(("R5_engaged", "PASS" if (built and engaged and not warned) else "FAIL",
                f"built={built} engaging_line={engaged} never_engaged_warn={warned}"))

out = "\n".join(f"{st:<4}  {name:<20} {why}" for name, st, why in results)
print(out)
with open(os.path.join(d, f"{probe}.verdict.txt"), "w") as f:
    f.write(out + "\n")
sys.exit(0 if all(st != "FAIL" for _, st, _ in results) else 1)
PY
}

# ── Per-model run: one server, both probes ───────────────────────────────────

declare -a CELL_NAMES=() CELL_RESULTS=()

run_model() {
    local label="$1" model_path="$2"
    local art="${ROOT_DIR}/${label}"

    if [[ -z "${model_path}" || ! -d "${model_path}" ]]; then
        echo "SKIP ${label}: snapshot not found (${model_path:-unset}); set RMLX_O_MODELS_ROOT" >&2
        for probe in A B; do
            CELL_NAMES+=("${label}/${probe}")
            CELL_RESULTS+=("skipped")
        done
        return 2
    fi

    rm -rf "${art}"
    mkdir -p "${art}"
    preflight

    echo "==> ${label}: starting server (kv-quant none, max-ctx 4096, metrics off)" >&2
    RMLX_HOME="${art}" \
        "${BINARY}" \
            --log verbose \
            --metrics off \
        serve \
            --model "${model_path}" \
            --port "${PORT}" \
            --kv-quant none \
            --max-ctx 4096 \
        > "${art}/serve.log" 2>&1 &
    local pid=$!
    # shellcheck disable=SC2064
    trap "kill ${pid} 2>/dev/null || true" RETURN

    if ! wait_for_server "${pid}"; then
        tail -40 "${art}/serve.log" >&2
        for probe in A B; do
            CELL_NAMES+=("${label}/${probe}")
            CELL_RESULTS+=("probe-fail")
        done
        return 1
    fi

    # Resolve the registry id from the server rather than guessing it from the
    # snapshot path — the id is registry-assigned and a wrong guess returns 404,
    # which would look like a probe failure instead of a harness failure.
    local model_id
    model_id="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
    if [[ -z "${model_id}" ]]; then
        echo "ERROR: could not read a model id from /v1/models" >&2
        for probe in A B; do
            CELL_NAMES+=("${label}/${probe}")
            CELL_RESULTS+=("probe-fail")
        done
        return 1
    fi
    echo "==> ${label}: registry model id = ${model_id}" >&2

    for probe in A B; do
        local rid="canary-${label}-${probe}-$$"
        local body
        body="$(probe_body "${probe}" "${model_id}")"
        printf '%s\n' "${body}" > "${art}/${probe}.request.json"

        echo "==> ${label}/${probe}: POST /v1/chat/completions (key=$(probe_key "${probe}"))" >&2
        curl -s -o "${art}/${probe}.response.json" -w '%{http_code}' \
            --max-time 600 \
            -H 'Content-Type: application/json' \
            -H "X-Request-Id: ${rid}" \
            -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -d "${body}" > "${art}/${probe}.status.txt" || true

        # Let the appender flush this request's records before reading them.
        sleep 1
        echo "--- ${label}/${probe} ---"
        if evaluate "${art}" "${probe}" "$(probe_key "${probe}")" "${rid}"; then
            CELL_NAMES+=("${label}/${probe}")
            CELL_RESULTS+=("probe-pass")
        else
            CELL_NAMES+=("${label}/${probe}")
            CELL_RESULTS+=("probe-fail")
        fi
    done

    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    sleep 2
    return 0
}

# ── Main ─────────────────────────────────────────────────────────────────────

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: ${BINARY} not found — run: make build-perf" >&2
    exit 2
fi

mkdir -p "${ROOT_DIR}"
overall=0

for spec in "bonsai:${BONSAI_MODEL}" "gemma4-e2b:${GEMMA_E2B_MODEL}"; do
    label="${spec%%:*}"
    path="${spec#*:}"
    [[ "${ONLY_MODEL}" == "all" || "${ONLY_MODEL}" == "${label}" ]] || continue
    run_model "${label}" "${path}"
done

echo
echo "════════ verdict (--expect ${EXPECT}) ════════"
for i in "${!CELL_NAMES[@]}"; do
    cell="${CELL_NAMES[$i]}"
    result="${CELL_RESULTS[$i]}"
    model="${cell%%/*}"
    probe="${cell##*/}"

    if [[ "${result}" == "skipped" ]]; then
        echo "  ${cell}: SKIPPED (snapshot absent)"
        overall=2
        continue
    fi

    want="fail"
    if [[ "${EXPECT}" == "fixed" ]] || baseline_expects_pass "${model}" "${probe}"; then
        want="pass"
    fi

    case "${want}:${result}" in
        pass:probe-pass) echo "  ${cell}: PASS  (expected pass, got pass)" ;;
        fail:probe-fail) echo "  ${cell}: PASS  (expected defect, reproduced)" ;;
        pass:probe-fail)
            echo "  ${cell}: FAIL  (expected pass, got fail — see ${ROOT_DIR}/${model}/${probe}.verdict.txt)"
            overall=1 ;;
        fail:probe-pass)
            echo "  ${cell}: FAIL  (expected the defect to reproduce, it did not — the table in this script is now wrong, or the harness cannot see it)"
            overall=1 ;;
    esac
done

if (( ${#CELL_NAMES[@]} == 0 )); then
    echo "  no models ran"
    overall=2
fi
exit "${overall}"
