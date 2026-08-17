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
# ONE SERVER PER PROBE — why this costs four model loads on purpose
# ─────────────────────────────────────────────────────────────────────────────
#
# An earlier revision ran one server per model and separated the two probes by
# the `X-Request-Id` each request carried. That premise — that the route's
# `request_id` span reaches the decode thread — is itself one of the fixes under
# test. It holds on the fixed arm and not on the baseline, so the filter matched
# every decode record on one arm and none on the other, and the harness reported
# product verdicts for measurements it had not made.
#
# A comparison harness may not depend on the behaviour it is comparing. Each
# probe therefore gets its own server process and its own `RMLX_HOME`, so its
# log contains its records and nothing else, and no filtering is needed on
# either arm. The request id is still sent, for a human reading the log.
#
# ─────────────────────────────────────────────────────────────────────────────
# DECISION RULE — read this before running, not after.
# ─────────────────────────────────────────────────────────────────────────────
#
# Each rule reports PASS, FAIL, N/A, or HARNESS.
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
#       period-1 case).
#         * Fewer than 16 emitted tokens -> N/A, with the count printed. A
#           stream that short cannot be degenerate, and R2 already catches the
#           long-run case; it is not reported as PASS because the rule did not
#           get to fire.
#         * No `step_fn` records, or records with no parsable `token_id`
#           -> HARNESS.
#   R5  The log shows the constraint was built AND engaged, and carries no
#       `constraint never engaged` warn.
#         * No `building SchemaConstraint` record at all -> HARNESS. Every probe
#           requests `response_format`, so a server that never built one is not
#           a product defect, it is a run that measured nothing.
#         * Built but never engaged -> FAIL. That is the product defect.
#
# Cell verdict: any HARNESS -> harness-error. Else any FAIL -> probe-fail. Else
# probe-pass.
#
# **A harness-error fails the run in BOTH arms.** It must never be mistaken for
# the defect `--expect baseline` is looking for — a check that silently matched
# zero records is the same class of defect as the vacuous gates this proof
# exists to eliminate.
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
#                enum) and R5 (no engaging record).
#   bonsai / B   MUST FAIL — same gate, and the key-string defect underneath it.
#   gemma / A    MUST PASS — **the reported whitespace loop does not reproduce
#                on this probe.** Observed at the pre-fix commit: byte-identical
#                to the fixed arm, `{\n  "unit": "celsius"\n}`, finish_reason
#                stop, 16 tokens, no periodic tail. gemma is not thinking-capable
#                so the engage gate is not involved, and with a single-word key
#                the grammar never corners the decoder. Whether it loops on
#                Probe A is then a matter of which token the model prefers at a
#                structural position, which this model does not.
#   gemma / B    MUST FAIL — the mask corners the decoder regardless of
#                preference. Observed at the pre-fix commit: 256 tokens,
#                finish_reason `length`, payload truncated at `{\n  "unit`.
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
# Writes nothing to the real metrics DB: hermetic RMLX_HOME per probe under
# .rmlx/proofs/schema-constraint, `--metrics off` on every server.
#
# Artifacts per (model, probe), under that hermetic root:
#   <model>/<probe>/{request,response}.json, status.txt, verdict.txt,
#                   serve.log, logs/*.jsonl

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
        -h|--help) sed -n '1,125p' "$0"; exit 0 ;;
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
# Exit 0 = probe-pass, 1 = probe-fail, 3 = harness-error.

evaluate() {
    # $1 = probe artifact dir, $2 = expected key
    python3 - "$1" "$2" <<'PY'
import json, os, re, sys, glob

d, want_key = sys.argv[1], sys.argv[2]

def read(p):
    try:
        with open(p) as f: return f.read()
    except OSError:
        return ""

body_raw = read(os.path.join(d, "response.json"))
status = read(os.path.join(d, "status.txt")).strip()

results = []   # (name, state, detail); state in {PASS, FAIL, N/A, HARNESS}

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

# ── This probe's log ─────────────────────────────────────────────────────────
# One server per probe, so every record in this RMLX_HOME belongs to this
# request. No filtering — a filter whose premise is one of the fixes under test
# cannot be used to compare the two arms.
lines = []
for lf in sorted(glob.glob(os.path.join(d, "logs", "*.jsonl"))):
    with open(lf) as f:
        lines.extend(f.readlines())

# R4 — degeneracy over emitted token ids
ids, step_records, unparsable = [], 0, 0
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
    results.append(("R4_not_degenerate", "HARNESS",
                    "no `step_fn sending token` records in this probe's log — "
                    "rerun the server with `--log verbose`"))
elif not ids:
    results.append(("R4_not_degenerate", "HARNESS",
                    f"{step_records} step_fn records found but none carried a "
                    f"parsable token_id ({unparsable} unparsable) — the log's "
                    "JSON shape does not match this extractor"))
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

# R5 — constraint built AND engaged, no non-enforcement warn.
# Every probe asks for `response_format`, so "never built" is a run that
# measured nothing, not a product verdict. "Built but never engaged" is the
# product defect this proof exists for.
blob = "".join(lines)
built = "building SchemaConstraint" in blob
engaged = re.search(r"SchemaConstraint:[^\"]*engaging", blob) is not None
warned = "constraint never engaged" in blob
if not built:
    results.append(("R5_engaged", "HARNESS",
                    "no `building SchemaConstraint` record — every probe requests "
                    "response_format, so the constraint path was never reached "
                    "and this cell measured nothing"))
else:
    results.append(("R5_engaged", "PASS" if (engaged and not warned) else "FAIL",
                    f"built=True engaging_line={engaged} never_engaged_warn={warned}"))

out = "\n".join(f"{st:<7}  {name:<20} {why}" for name, st, why in results)
print(out)
with open(os.path.join(d, "verdict.txt"), "w") as f:
    f.write(out + "\n")

states = [st for _, st, _ in results]
if "HARNESS" in states:
    sys.exit(3)
sys.exit(1 if "FAIL" in states else 0)
PY
}

# ── Per-probe run: its own server, its own RMLX_HOME, its own log ────────────

declare -a CELL_NAMES=() CELL_RESULTS=()

record_cell() {
    CELL_NAMES+=("$1")
    CELL_RESULTS+=("$2")
}

run_probe() {
    local label="$1" model_path="$2" probe="$3"
    local cell="${label}/${probe}"
    local art="${ROOT_DIR}/${label}/${probe}"

    rm -rf "${art}"
    mkdir -p "${art}"
    preflight

    echo "==> ${cell}: starting server (kv-quant none, max-ctx 4096, metrics off)" >&2
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

    if ! wait_for_server "${pid}"; then
        tail -40 "${art}/serve.log" >&2
        kill "${pid}" 2>/dev/null || true
        record_cell "${cell}" "harness-error"
        return
    fi

    # Resolve the registry id from the server rather than guessing it from the
    # snapshot path — the id is registry-assigned and a wrong guess returns 404,
    # which would look like a probe failure instead of a harness failure.
    local model_id
    model_id="$(curl -sf "http://127.0.0.1:${PORT}/v1/models" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)"
    if [[ -z "${model_id}" ]]; then
        echo "ERROR: could not read a model id from /v1/models" >&2
        kill "${pid}" 2>/dev/null || true
        record_cell "${cell}" "harness-error"
        return
    fi

    local body
    body="$(probe_body "${probe}" "${model_id}")"
    printf '%s\n' "${body}" > "${art}/request.json"

    echo "==> ${cell}: POST /v1/chat/completions (model=${model_id} key=$(probe_key "${probe}"))" >&2
    curl -s -o "${art}/response.json" -w '%{http_code}' \
        --max-time 600 \
        -H 'Content-Type: application/json' \
        -H "X-Request-Id: canary-${label}-${probe}" \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -d "${body}" > "${art}/status.txt" || true

    # Stop the server before reading its log so the appender has flushed.
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    sleep 2

    echo "--- ${cell} ---"
    evaluate "${art}" "$(probe_key "${probe}")"
    case $? in
        0) record_cell "${cell}" "probe-pass" ;;
        1) record_cell "${cell}" "probe-fail" ;;
        *) record_cell "${cell}" "harness-error" ;;
    esac
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

    if [[ -z "${path}" || ! -d "${path}" ]]; then
        echo "SKIP ${label}: snapshot not found (${path:-unset}); set RMLX_O_MODELS_ROOT" >&2
        record_cell "${label}/A" "skipped"
        record_cell "${label}/B" "skipped"
        continue
    fi
    for probe in A B; do
        run_probe "${label}" "${path}" "${probe}"
    done
done

echo
echo "════════ verdict (--expect ${EXPECT}) ════════"
for i in "${!CELL_NAMES[@]}"; do
    cell="${CELL_NAMES[$i]}"
    result="${CELL_RESULTS[$i]}"
    model="${cell%%/*}"
    probe="${cell##*/}"

    case "${result}" in
        skipped)
            echo "  ${cell}: SKIPPED (snapshot absent)"
            overall=2
            continue ;;
        harness-error)
            # Never a product verdict, in either arm: this cell measured nothing.
            echo "  ${cell}: HARNESS ERROR — measured nothing (see ${ROOT_DIR}/${model}/${probe}/verdict.txt)"
            overall=1
            continue ;;
    esac

    want="fail"
    if [[ "${EXPECT}" == "fixed" ]] || baseline_expects_pass "${model}" "${probe}"; then
        want="pass"
    fi

    case "${want}:${result}" in
        pass:probe-pass) echo "  ${cell}: PASS  (expected pass, got pass)" ;;
        fail:probe-fail) echo "  ${cell}: PASS  (expected defect, reproduced)" ;;
        pass:probe-fail)
            echo "  ${cell}: FAIL  (expected pass, got fail — see ${ROOT_DIR}/${model}/${probe}/verdict.txt)"
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
