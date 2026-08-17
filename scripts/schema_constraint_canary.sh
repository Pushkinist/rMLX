#!/usr/bin/env bash
# schema_constraint_canary.sh — real-model proof for the `json_schema` constraint path.
#
# Answers two questions, on two architectures, with the verdict fixed before
# the run:
#
#   1. Can a `json_schema` request return success on a non-terminating run of
#      grammar-permitted whitespace?           (must be: no)
#   2. Can a `SchemaConstraint` be built and then never engage, with the caller
#      told nothing?                           (must be: no)
#
# ─────────────────────────────────────────────────────────────────────────────
# DECISION RULE — read this before running, not after.
# ─────────────────────────────────────────────────────────────────────────────
#
# Per model, the probe PASSES iff ALL FIVE hold:
#
#   R1  HTTP status is 200.
#   R2  finish_reason == "stop".  A degenerate run cannot stop on EOS — the
#       mask withholds EOS until the value is complete — so it always ends at
#       max_tokens with "length".
#   R3  The payload (content, else reasoning_content) parses as JSON after
#       stripping an optional ```json fence, and VALIDATES against the probe
#       schema: object, exactly the key "unit", value in {"celsius",
#       "fahrenheit"}, no other keys.
#   R4  The emitted token-id stream is not degenerate. Degenerate := the tail
#       is periodic with period <= 4 for >= 16 tokens (a constant stream is the
#       period-1 case). Ids come from the run's own log.
#   R5  The run log contains a `building SchemaConstraint` line AND a
#       `SchemaConstraint: ... engaging` line, and does NOT contain
#       `constraint never engaged`.
#
# The probe FAILS if any of R1..R5 is false. There is no partial credit and no
# "looks fine" branch.
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT THE BASELINE (origin/main, before the fix) MUST PRODUCE
# ─────────────────────────────────────────────────────────────────────────────
#
# Run with `--expect baseline` and the exit code inverts: the script then
# REQUIRES the documented defect and fails if it does not reproduce. A fix that
# changes nothing is caught by `--expect fixed` failing; a proof harness too
# weak to see the defect is caught by `--expect baseline` failing.
#
#   gemma-4-e2b   R2 false (finish_reason "length"), R3 false (payload is
#                 ```json\n{\n  " plus an endless \n / two-space alternation),
#                 R4 false (ids 107,138,107,138,… — period 2). R5 true: the
#                 constraint DID engage on token 236782 (`{`).
#   Bonsai-8B     R3 false (payload is a different object, e.g. an "error" key
#                 the schema forbids). R5 false: the log has
#                 `building SchemaConstraint` and NO engaging line. `constraint
#                 never engaged` is also absent — on origin/main that warn does
#                 not exist, which is the second half of the defect.
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
# .rmlx/proofs/schema-constraint, `--metrics off` on every server.
#
# Artifacts (per model, under that hermetic root):
#   response.json   raw HTTP body
#   serve.log       server stdout/stderr
#   logs/*.jsonl    structured run log (the R4/R5 evidence)
#   verdict.txt     R1..R5 with the observed value of each

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
        -h|--help) sed -n '1,84p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done
case "${EXPECT}" in
    fixed|baseline) ;;
    *) echo "--expect must be 'fixed' or 'baseline'" >&2; exit 2 ;;
esac

# ── The probe request ─────────────────────────────────────────────────────────
# Object root (so the engage policy is ValueStarter — the shape that failed),
# one required enum property, additionalProperties:false, strict, temperature 0.
# max_tokens is generous on purpose: a bounded grammar must stop well short of
# it, and an unbounded one must be given room to prove it does not.
read -r -d '' PROBE_BODY_TMPL <<'JSON' || true
{
  "model": "MODEL_ID",
  "messages": [
    {"role": "user", "content": "What unit is 25 degrees Celsius measured in? Answer with the JSON object only."}
  ],
  "temperature": 0,
  "seed": 0,
  "max_tokens": 256,
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "unit_answer",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["unit"],
        "additionalProperties": false
      }
    }
  }
}
JSON

# ── Preflight ─────────────────────────────────────────────────────────────────

preflight() {
    pkill -f "rmlx serve"   2>/dev/null || true
    pkill -f "rmlx_main serve" 2>/dev/null || true
    pkill -f mlx_lm         2>/dev/null || true
    pkill -f paroquant      2>/dev/null || true
    pkill -f omlx           2>/dev/null || true
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

# ── Verdict evaluation (python: JSON + the periodicity rule) ──────────────────

evaluate() {
    # $1 = artifact dir
    python3 - "$1" <<'PY'
import json, os, re, sys, glob

d = sys.argv[1]
def read(p):
    try:
        with open(p) as f: return f.read()
    except OSError:
        return ""

body_raw = read(os.path.join(d, "response.json"))
status = read(os.path.join(d, "status.txt")).strip()

results = {}

# R1 — HTTP 200
results["R1_http_200"] = (status == "200", f"status={status or 'none'}")

# Parse the response envelope.
try:
    body = json.loads(body_raw)
except Exception as e:
    body = None
    results["R1_http_200"] = (False, f"status={status} body_unparseable: {e}")

msg = {}
finish = None
if isinstance(body, dict):
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    finish = ch.get("finish_reason")

# R2 — finish_reason
results["R2_finish_stop"] = (finish == "stop", f"finish_reason={finish!r}")

# R3 — payload validates against the probe schema
payload = msg.get("content") or msg.get("reasoning_content") or ""
stripped = payload.strip()
stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
stripped = re.sub(r"\s*```$", "", stripped)
ok_r3, why_r3 = False, ""
try:
    v = json.loads(stripped)
    if not isinstance(v, dict):
        why_r3 = f"payload is {type(v).__name__}, not an object"
    elif set(v.keys()) != {"unit"}:
        why_r3 = f"keys={sorted(v.keys())} (schema allows exactly ['unit'])"
    elif v["unit"] not in ("celsius", "fahrenheit"):
        why_r3 = f"unit={v['unit']!r} not in enum"
    else:
        ok_r3, why_r3 = True, f"payload={v}"
except Exception as e:
    why_r3 = f"not JSON: {e}; payload[:120]={stripped[:120]!r}"
results["R3_schema_valid"] = (ok_r3, why_r3)

# ── R4 — degeneracy over emitted token ids ───────────────────────────────────
# Degenerate := the TAIL of the id sequence is periodic with period p <= 4 for
# >= 16 tokens. A constant stream is the p == 1 case.
ids = []
for lf in sorted(glob.glob(os.path.join(d, "logs", "*.jsonl"))):
    with open(lf) as f:
        for line in f:
            if "step_fn sending token" not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            fields = rec.get("fields", rec)
            tid = fields.get("token_id")
            if tid is not None:
                ids.append(int(tid))

def degenerate_tail(seq, max_period=4, min_len=16):
    if len(seq) < min_len:
        return None
    for p in range(1, max_period + 1):
        n = 0
        i = len(seq) - 1
        while i - p >= 0 and seq[i] == seq[i - p]:
            n += 1
            i -= 1
        if n + p >= min_len:
            return (p, n + p)
    return None

if not ids:
    results["R4_not_degenerate"] = (
        False,
        "no token_id records in the run log — rerun the server with `--log verbose`",
    )
else:
    deg = degenerate_tail(ids)
    if deg:
        p, ln = deg
        results["R4_not_degenerate"] = (
            False, f"tail is {p}-periodic for {ln} tokens; last12={ids[-12:]}")
    else:
        results["R4_not_degenerate"] = (True, f"{len(ids)} tokens, last12={ids[-12:]}")

# ── R5 — the constraint was built AND engaged, with no non-enforcement warn ──
log_text = ""
for lf in sorted(glob.glob(os.path.join(d, "logs", "*.jsonl"))):
    log_text += read(lf)
built    = "building SchemaConstraint" in log_text
engaged  = re.search(r"SchemaConstraint:[^\"]*engaging", log_text) is not None
warned   = "constraint never engaged" in log_text
results["R5_engaged"] = (
    built and engaged and not warned,
    f"built={built} engaging_line={engaged} never_engaged_warn={warned}",
)

lines = []
allpass = True
for k in ("R1_http_200", "R2_finish_stop", "R3_schema_valid", "R4_not_degenerate", "R5_engaged"):
    ok, why = results[k]
    allpass &= ok
    lines.append(f"{'PASS' if ok else 'FAIL'}  {k:<20} {why}")
out = "\n".join(lines)
print(out)
with open(os.path.join(d, "verdict.txt"), "w") as f:
    f.write(out + "\n")
sys.exit(0 if allpass else 1)
PY
}

# ── Per-model run ─────────────────────────────────────────────────────────────

run_model() {
    local label="$1" model_path="$2"
    local art="${ROOT_DIR}/${label}"

    if [[ -z "${model_path}" || ! -d "${model_path}" ]]; then
        echo "SKIP ${label}: snapshot not found (${model_path:-unset}); set RMLX_O_MODELS_ROOT" >&2
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
        return 1
    fi
    echo "==> ${label}: registry model id = ${model_id}" >&2
    local body="${PROBE_BODY_TMPL/MODEL_ID/${model_id}}"
    printf '%s\n' "${body}" > "${art}/request.json"

    echo "==> ${label}: POST /v1/chat/completions" >&2
    curl -s -o "${art}/response.json" -w '%{http_code}' \
        --max-time 600 \
        -H 'Content-Type: application/json' \
        -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -d "${body}" > "${art}/status.txt" || true

    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    sleep 2

    echo "--- ${label} ---"
    evaluate "${art}"
    return $?
}

# ── Main ──────────────────────────────────────────────────────────────────────

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: ${BINARY} not found — run: make build-perf" >&2
    exit 2
fi

mkdir -p "${ROOT_DIR}"
declare -a NAMES=() VERDICTS=()
overall=0

for spec in "bonsai:${BONSAI_MODEL}" "gemma4-e2b:${GEMMA_E2B_MODEL}"; do
    label="${spec%%:*}"
    path="${spec#*:}"
    [[ "${ONLY_MODEL}" == "all" || "${ONLY_MODEL}" == "${label}" ]] || continue

    run_model "${label}" "${path}"
    rc=$?
    NAMES+=("${label}")
    case "${rc}" in
        0) VERDICTS+=("probe-pass") ;;
        1) VERDICTS+=("probe-fail") ;;
        *) VERDICTS+=("skipped");   overall=2 ;;
    esac
done

echo
echo "════════ verdict (--expect ${EXPECT}) ════════"
for i in "${!NAMES[@]}"; do
    n="${NAMES[$i]}"; v="${VERDICTS[$i]}"
    case "${EXPECT}:${v}" in
        fixed:probe-pass)    echo "  ${n}: PASS  (constraint enforced, stream not degenerate)" ;;
        fixed:probe-fail)    echo "  ${n}: FAIL  (see ${ROOT_DIR}/${n}/verdict.txt)"; overall=1 ;;
        baseline:probe-fail) echo "  ${n}: PASS  (defect reproduced, as the baseline must)" ;;
        baseline:probe-pass) echo "  ${n}: FAIL  (baseline did NOT reproduce the defect — this harness cannot see it)"; overall=1 ;;
        *:skipped)           echo "  ${n}: SKIPPED (snapshot absent)" ;;
    esac
done

if (( ${#NAMES[@]} == 0 )); then
    echo "  no models ran"
    overall=2
fi
exit "${overall}"
