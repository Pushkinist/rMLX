#!/usr/bin/env bash
# c1-gemma4-cold-equal.sh — C1 acceptance (C): gemma4 partial-prefix reuse
# must be token-for-token cold-equal.
#
# gemma-4-26b-a4b-it-mxfp8 has sliding_window=1024. The production
# CacheLookup::Prefix path only fires when the cached snapshot's SWA ring
# has NOT wrapped (cached prompt <= ~1024 tokens) — otherwise it correctly
# falls back to Miss (full re-prefill). So this test uses a SHORT base
# prompt (well under the SWA window, >=1 full 256-token block) so a genuine
# PREFIX HIT is taken.
#
# Procedure:
#   1. Fresh server. Fire P_cold once -> COLD completion (cache empty: Miss).
#   2. Fresh server (cache cleared). Fire BASE prompt (populates cache),
#      then P_cold -> must be a PREFIX HIT (shares the long block-aligned
#      prefix, diverges in a short suffix). -> WARM completion.
#   3. PASS iff COLD completion token-ids == WARM completion token-ids.
#   4. Exact-path intact: fire one prompt 3x identical -> equal tokens.
#
# Output: C1_RESULT lines on stdout.

set -uo pipefail

RMLX_BIN="${RMLX_BIN:-${RMLX_ROOT}/target/release/rmlx}"
MODEL="${RMLX_O_MODELS_ROOT:?Set RMLX_O_MODELS_ROOT - see .env.example}/mlx-community__gemma-4-26b-a4b-it-mxfp8"
MODEL_ID="$(basename "${MODEL}")"
PORT=62265
TS="$(date -u +%Y%m%d-%H%M%S)"
RESP="/tmp/c1_resp_$$.json"
PAYLOAD="/tmp/c1_payload_$$.json"

preflight() {
    pkill -f "rmlx serve" 2>/dev/null || true
    pkill -f mlx_lm 2>/dev/null || true
    sleep 5
    rm -f "/tmp/rmlx.${PORT}.claim"
}

start_server() {
    SERVE_LOG="${RMLX_ROOT}/logs/c1_${MODEL_ID}_${TS}_$1.log"
    "${RMLX_BIN}" serve \
        --model "${MODEL}" --port "${PORT}" --host 127.0.0.1 \
        --device gpu --kv-quant k8v4 --max-ctx 8192 \
        > "${SERVE_LOG}" 2>&1 &
    SERVER_PID=$!
    local e=0
    until curl -s --max-time 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q '"ok"'; do
        sleep 5; e=$((e+5))
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "SERVER FAILED, see ${SERVE_LOG}" >&2; return 1
        fi
        [[ ${e} -ge 600 ]] && { echo "server timeout" >&2; return 1; }
    done
    echo "  server up in ${e}s (pid ${SERVER_PID})" >&2
}

stop_server() {
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
}

# Build a chat payload. $1 = "base" | "cold" ; writes $PAYLOAD.
# BASE: a ~700-word system+user prompt (tokenizes well over 256, under the
#       1024 SWA window for this model's tokenizer).
# COLD: BASE user content + a short distinct appended question (shares the
#       long block-aligned prefix, then diverges) -> PREFIX HIT vs BASE.
build_payload() {
    python3 - "$1" "${MODEL_ID}" "${PAYLOAD}" <<'PYEOF'
import json, sys
kind, model_id, out = sys.argv[1], sys.argv[2], sys.argv[3]

# Deterministic body sized so the user turn spans >=2 full 256-token blocks
# but stays well under the 1024-token SWA window after chat templating
# (otherwise the SWA RotatingKvCache wraps and the production path correctly
# falls back to Miss instead of taking the PREFIX HIT we want to test).
para = ("The cache subsystem stores post-prefill key and value tensors so that "
        "a later request sharing a leading block of tokens can skip recomputation. "
        "Block-aligned matching keeps the addressing key stable across requests. ")
body = (para * 22).strip()

system = "You are a careful technical assistant. Answer concisely with no preamble."
base_user = "Background:\n" + body + "\n\nSummarize the caching idea in one sentence."

if kind == "base":
    user = base_user
else:  # cold: shares the long prefix, diverges in the trailing question
    user = "Background:\n" + body + "\n\nInstead, list two risks of stale cache reuse."

payload = {
    "model": model_id,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": False,
}
with open(out, "w") as f:
    json.dump(payload, f)
PYEOF
}

# Fire current $PAYLOAD, print the completion token-id list (space-joined).
fire_tokens() {
    curl -s --max-time 600 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "@${PAYLOAD}" -o "${RESP}" 2>/dev/null
    python3 - "${RESP}" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    # rMLX echoes per-token ids in usage or choices? Fall back to text.
    ch = d["choices"][0]
    txt = ch.get("message", {}).get("content", "")
    # token-id list if the server exposes it; else hash the text.
    toks = d.get("usage", {}).get("completion_token_ids")
    if toks is None:
        toks = ch.get("token_ids")
    if toks is None:
        # No id list exposed: use the exact completion string as the
        # equality key (byte-identical text <=> byte-identical greedy ids
        # at temp=0 for the same tokenizer).
        print("TEXT:" + txt)
    else:
        print("IDS:" + " ".join(str(t) for t in toks))
except Exception as e:
    print("ERR:" + repr(e))
PYEOF
}

echo "=== C1 gemma4 cold-equality (${MODEL_ID}) ===" >&2

# ---- Phase 1: COLD (fresh server, cache empty -> Miss) ----
preflight
start_server cold || exit 1
build_payload cold
COLD="$(fire_tokens)"
echo "  COLD = ${COLD:0:120}" >&2
stop_server

# ---- Phase 2: WARM (fresh server; base populates cache, then P_cold) ----
preflight
start_server warm || exit 1
build_payload base
BASE_OUT="$(fire_tokens)"          # populate cache with the base prompt
echo "  BASE primed (${BASE_OUT:0:60})" >&2
build_payload cold
WARM="$(fire_tokens)"              # must be a PREFIX HIT vs base
echo "  WARM = ${WARM:0:120}" >&2

# Confirm the WARM request actually took the production PREFIX HIT path
# (not a Miss — a Miss would trivially equal COLD and give a false PASS).
if grep -q "prompt cache PREFIX HIT" "${SERVE_LOG}" 2>/dev/null; then
    PREFIX_TAKEN=yes
else
    PREFIX_TAKEN=no
fi
echo "  PREFIX_HIT_taken=${PREFIX_TAKEN}" >&2

# ---- Phase 3: Exact-path intact (3x identical -> equal) ----
build_payload base
E1="$(fire_tokens)"; E2="$(fire_tokens)"; E3="$(fire_tokens)"
stop_server

echo "C1_RESULT prefix_hit_taken=${PREFIX_TAKEN}"
if [[ "${PREFIX_TAKEN}" == "yes" && "${COLD}" == "${WARM}" && "${COLD}" != ERR:* && "${COLD}" != "" ]]; then
    echo "C1_RESULT cold_equal=PASS"
else
    echo "C1_RESULT cold_equal=FAIL"
fi
if [[ "${E1}" == "${E2}" && "${E2}" == "${E3}" && "${E1}" != ERR:* && "${E1}" != "" ]]; then
    echo "C1_RESULT exact_path=PASS"
else
    echo "C1_RESULT exact_path=FAIL"
fi
echo "C1_COLD=${COLD}"
echo "C1_WARM=${WARM}"
