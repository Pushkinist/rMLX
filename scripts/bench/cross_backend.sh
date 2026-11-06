#!/usr/bin/env bash
# Drive one cross-backend bench cell through the Cross-Backend-Bench harness
#
# Usage:  bash scripts/bench/cross_backend.sh [--binary PATH] TAG MODEL_PATH BACKEND KV_QUANT
#             BUDGET_MIN MODEL_DISK_GB WEIGHT_QUANT
#         --binary PATH  the rmlx binary: the server of an rmlx cell, the Metal claim holder of every
#                        other cell, and the source of the run identity (default: the release build
#                        of this checkout); first, before TAG
#         TAG            cell label filed in the progress line (e2b, e4b, 26b, 31b, 31b-paro)
#         MODEL_PATH     absolute path of the snapshot directory; the .gguf file for the llama.cpp arms
#         BACKEND        rmlx | mlx-lm | mlx-lm-turboquant | omlx | paroquant | llama.cpp
#                        | llama-cpp-turboquant
#         KV_QUANT       rmlx: any `rmlx serve --kv-quant` value; mlx-lm: none | auto | bf16 | f16;
#                        llama.cpp arms: a ggml cache type (none and auto mean f16); the other arms
#                        serve their own codec (mlx-lm-turboquant 8,4) and only file this name.
#                        Filed as: rmlx: the codec its server's run log states; the other arms: mixed ->
#                        k8v4, turboquant -> turbo4, native -> none, else as given
#         BUDGET_MIN     whole minutes for the server to come up and the harness to finish
#         MODEL_DISK_GB  model size on disk in GB, for the harness's tps-per-gb
#         WEIGHT_QUANT   weight format label filed with the cell (mxfp8, paro, ...)
# Exit:   0 the cell ran (its success and coherence are in the progress line); 1 unknown backend,
#         a missing argument or an rmlx binary that is not an executable file; 2 the server did not become ready; 3 the budget ran out before the
#         harness started; 4 no codec name to file (an rmlx server's run log states none); 5 no
#         llama-server binary; 6 bare mlx-lm given a quantised KV_QUANT
# Not a mode of: bench/baseline_cells.sh — that one runs `rmlx baseline` in process; this one starts
#         a backend's own server and hands it to the Cross-Backend-Bench harness
#
# One line per cell is appended to <RMLX_HOME>/bench/cross_backend_progress.md. The server and
# harness logs and the oMLX models directory are in <RMLX_HOME>/tmp, where the transient files are.
# The harness checkout is $CROSS_BENCH_ROOT, default ../Cross-Backend-Bench. An rmlx cell's records
# go into <RMLX_HOME>/metrics/runs.db through RMLX_METRICS_DB.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${REPO_ROOT}/scripts/lib/identity.sh"
RMLX_BIN="$(rmlx_default_binary "$REPO_ROOT" release)"
if [ "${1:-}" = --binary ]; then
  [ -n "${2:-}" ] || { echo "ERROR: --binary needs a path" >&2; exit 1; }
  case "$2" in
    /*) RMLX_BIN="$2" ;;
    *) RMLX_BIN="$PWD/$2" ;;
  esac
  shift 2
fi
[ -x "$RMLX_BIN" ] || { echo "ERROR: binary not found / not executable: $RMLX_BIN" >&2; exit 1; }

TAG="$1"
MODEL_PATH="$2"
BACKEND="$3"
KV_QUANT="$4"
BUDGET_MIN="$5"
MODEL_DISK_GB="$6"
WEIGHT_QUANT="$7"

CBB=${CROSS_BENCH_ROOT:-../Cross-Backend-Bench}
# Read by the cross-backend harness to find this checkout's binary and metrics buffer.
export RMLX_REPO_ROOT="${REPO_ROOT}"

source "${REPO_ROOT}/scripts/lib/bench_server.sh"
# Run identity (backend / version / git sha / build profile / hardware tag)
# comes from the measured binary — never hard-coded here.
rmlx_export_identity "$RMLX_BIN"
PORT=62265
BUDGET_S=$((BUDGET_MIN * 60))

# Runtime state lives under one root (CLAUDE.md): `RMLX_HOME`, else the
# checkout's `.rmlx/`. The CBB harness reads the DB path from RMLX_METRICS_DB.
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
export RMLX_HOME
export RMLX_METRICS_DB="${RMLX_HOME}/metrics/runs.db"
PROGRESS="${RMLX_HOME}/bench/cross_backend_progress.md"
LOG_DIR="${RMLX_HOME}/logs"
SCRATCH_DIR="${RMLX_HOME}/tmp"
mkdir -p "$(dirname "$PROGRESS")" "$LOG_DIR" "$SCRATCH_DIR"

  _PROMPT_ID_EFF="longctx_8k"
  _CTX_MAX_EFF=16384

export CBB_CTX_MAX="${_CTX_MAX_EFF}"
export CBB_PROMPT_NAME="${_PROMPT_ID_EFF}"

MODEL_ID=$(basename "$MODEL_PATH")
TS_START=$(date +%s)
TS_HMS=$(date +%H:%M:%S)
LOG="${SCRATCH_DIR}/bench_${TAG}_${BACKEND}_${KV_QUANT}.log"
HARNESS_LOG="${SCRATCH_DIR}/bench_harness_${TAG}_${BACKEND}_${KV_QUANT}.log"

echo "[$(date +%T)] CELL START tag=$TAG backend=$BACKEND kv=$KV_QUANT model_id=$MODEL_ID" >&2

# The Metal claim is the one gate on the GPU: an rmlx arm takes it itself, and
# every other backend runs under `rmlx claim run`. A claim another process
# holds ends the cell with exit 2 and the server's 11 in the message; this script
# stops only the server it started.

# --- Launch backend ---
SERVER_PID=0

case "$BACKEND" in
  rmlx)
    snapshot_logs
    ( cd "$REPO_ROOT" && exec "$RMLX_BIN" serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --device gpu \
        --max-ctx 16384 \
        --kv-quant "$KV_QUANT" ) >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    HARNESS_MODEL="$MODEL_ID"
    ;;
  mlx-lm-turboquant)
    "$RMLX_BIN" claim run -- "${MLX_LM_TURBOQUANT_ROOT:-../mlx-lm-turboquant}/.venv/bin/mlx_lm.server" \
        --model "$MODEL_PATH" \
        --kv-cache-quantization 8,4 \
        --quantized-kv-start 0 \
        --port "$PORT" \
        --max-tokens 8192 \
        --log-level WARNING >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    HARNESS_MODEL="$MODEL_PATH"
    ;;
  omlx)
    OMLX_MODELS="${SCRATCH_DIR}/omlx_models"
    mkdir -p "$OMLX_MODELS"
    find "$OMLX_MODELS" -mindepth 1 -maxdepth 1 -delete 2>/dev/null
    OMLX_MODEL_NAME="gemma4-${TAG}"
    ln -sfn "$MODEL_PATH" "$OMLX_MODELS/$OMLX_MODEL_NAME"
    "$RMLX_BIN" claim run -- "${OMLX_ROOT:-../oMLX}/.venv/bin/omlx" serve \
        --model-dir "$OMLX_MODELS" \
        --port "$PORT" \
        --api-key 1234 >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY="1234"
    HARNESS_MODEL="$OMLX_MODEL_NAME"
    ;;
  paroquant)
    "$RMLX_BIN" claim run -- ${PAROQUANT_ROOT:-../paroquant}/.venv/bin/python -m paroquant.cli.serve \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --log-level WARNING >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    HARNESS_MODEL="$MODEL_PATH"
    ;;
  mlx-lm)
    # Bare Apple stock mlx-lm: no --kv-cache-quantization, so this arm is the
    # unquantised same-framework reference for rMLX, not another codec.
    #
    # Because the launcher below cannot express a codec, a quantised KV_QUANT
    # would be REPORTED and RECORDED while an unquantised run actually happened
    # -- a row labelled `k8v4` produced with no quantisation at all. Refuse it;
    # `mlx-lm-turboquant` is the arm that takes a codec.
    case "$KV_QUANT" in
      none|auto|bf16|f16) ;;
      *)
        echo "[$(date +%T)] CELL FAIL kv-quant-unsupported tag=$TAG backend=mlx-lm kv=$KV_QUANT" >&2
        echo "  bare mlx-lm serves unquantised KV only; use backend 'mlx-lm-turboquant' for a codec." >&2
        echo "[$TS_HMS] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | KV_QUANT_UNSUPPORTED" >>"$PROGRESS"
        exit 6
        ;;
    esac
    "$RMLX_BIN" claim run -- "${MLX_LM_ROOT:-../mlx-lm}/.venv/bin/mlx_lm.server" \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --max-tokens 8192 \
        --log-level WARNING >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    HARNESS_MODEL="$MODEL_PATH"
    ;;
  llama.cpp|llama-cpp-turboquant)
    # GGUF arms. MODEL_PATH is a .gguf file here, not an MLX snapshot dir --
    # no file exists that both families can load, so a cell against an MLX
    # backend rests on a stated quant-equivalence assumption (see
    # docs/PERF_BASELINE.md § "Cross-backend cells"). The fork-vs-upstream
    # pair does not.
    #
    # KV_QUANT is the ggml cache type for both sides; `none` means f16, which
    # is what upstream calls unquantised KV. `turbo2|turbo3|turbo4` exist only
    # in the fork, so upstream refuses them at startup rather than silently
    # running something else.
    if [ "$BACKEND" = "llama.cpp" ]; then
      _LCPP_BIN="${LLAMA_CPP_ROOT:-../llama.cpp}/build/bin/llama-server"
    else
      _LCPP_BIN="${LLAMA_CPP_TURBOQUANT_ROOT:-../llama-cpp-turboquant}/build/bin/llama-server"
    fi
    if [ ! -x "$_LCPP_BIN" ]; then
      echo "[$(date +%T)] CELL FAIL no-llama-server tag=$TAG backend=$BACKEND path=$_LCPP_BIN" >&2
      echo "[$TS_HMS] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | NO_LLAMA_SERVER" >>"$PROGRESS"
      exit 5
    fi
    case "$KV_QUANT" in
      none|auto) _LCPP_CT="f16" ;;
      *)         _LCPP_CT="$KV_QUANT" ;;
    esac
    "$RMLX_BIN" claim run -- "$_LCPP_BIN" \
        --model "$MODEL_PATH" \
        --port "$PORT" \
        --host 127.0.0.1 \
        -c "$CBB_CTX_MAX" \
        -np 1 \
        -fa on \
        --no-webui \
        -ctk "$_LCPP_CT" -ctv "$_LCPP_CT" >"$LOG" 2>&1 &
    SERVER_PID=$!
    BASE_URL="http://127.0.0.1:$PORT"
    API_KEY=""
    # llama-server serves whatever model it was started with and ignores the
    # `model` field, so this is a label, not a lookup key. Strip the `.gguf`
    # extension: the CBB harness derives the recorded `model` column from it,
    # and `Qwen3-8B-Q8_0.gguf` would become a second champion for the cell that
    # `scripts/ingest/ingest.py ab --harness llama` already records as `Qwen3-8B-Q8_0`.
    HARNESS_MODEL="$(basename "$MODEL_PATH" .gguf)"
    # The measured binary's own commit, not the literal "head" the other arms
    # pass. A version column that says "head" for every run of every build
    # cannot distinguish two builds, which is the whole point of recording it.
    BACKEND_VERSION_OVERRIDE="$(git -C "$(dirname "$_LCPP_BIN")/../.." rev-parse --short HEAD 2>/dev/null || echo unknown)"
    # NOTE: the CBB legacy path still derives `model_namespace` itself, from the
    # `--model-path` basename, so these cells land under `local` while the A/B
    # ingest path records them under `hf`. That split lives in the sibling
    # harness, not here. `bench/llama_ab.sh` + `ingest/ingest.py` is the
    # canonical recorder for GGUF cells; this arm is a smoke/coverage path.
    ;;
  *)
    echo "unknown backend $BACKEND" >&2
    echo "[$TS_HMS] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | UNKNOWN_BACKEND" >>"$PROGRESS"
    exit 1
    ;;
esac

# The process to measure. A non-rmlx server is `claim run`'s child, and the
# harness reads the server's memory, not the wrapper's.
BACKEND_PID="$SERVER_PID"
if [ "$BACKEND" != "rmlx" ]; then
  BACKEND_PID=""
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    BACKEND_PID="$(pgrep -P "$SERVER_PID" | sed -n 1p)"
    [ -n "$BACKEND_PID" ] && break
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.5
  done
fi

# stop_server — stop the server this cell started and wait for it, so its
# claim is free when this returns. Its exit status is left in SERVER_RC.
stop_server() {
  kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_RC=$?
}

# --- Wait for /v1/models within budget ---
READY=0
DEADLINE=$((TS_START + BUDGET_S))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  sleep 2
  # Check process alive.
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  if [ -n "$API_KEY" ]; then
    curl -fsS -m 3 -H "Authorization: Bearer $API_KEY" "$BASE_URL/v1/models" >/dev/null 2>&1 && READY=1 && break
  else
    curl -fsS -m 3 "$BASE_URL/v1/models" >/dev/null 2>&1 && READY=1 && break
  fi
done

if [ "$READY" -ne 1 ] || [ -z "$BACKEND_PID" ]; then
  stop_server
  echo "[$(date +%T)] CELL FAIL server-not-ready tag=$TAG backend=$BACKEND kv=$KV_QUANT server_exit=$SERVER_RC (11: the Metal claim is held; its holder is in $LOG)" >&2
  echo "[$(date +%T)] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | SERVER_NOT_READY" >>"$PROGRESS"
  exit 2
fi

# --- Name the codec the cell is filed under ---
KV_FILED=""
if [ "$BACKEND" = rmlx ]; then
  RUN_LOG="$(phase_log "$SERVER_PID")" || RUN_LOG=""
  [ -z "$RUN_LOG" ] || KV_FILED="$(log_kv_quant "$RUN_LOG")"
else
  case "$KV_QUANT" in
    mixed)      KV_FILED=k8v4 ;;
    turboquant) KV_FILED=turbo4 ;;
    native)     KV_FILED=none ;;
    *)          KV_FILED="$KV_QUANT" ;;
  esac
fi
if [ -z "$KV_FILED" ]; then
  stop_server
  echo "ERROR: cell $TAG $BACKEND $KV_QUANT: no KV codec name to file it under (an rmlx server's run log states none)" >&2
  echo "[$(date +%T)] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | NO_KV_NAME" >>"$PROGRESS"
  exit 4
fi
QUANT_SIG="${WEIGHT_QUANT} / ${KV_FILED}"

# --- Warmup (1 token) ---
if [ -n "$API_KEY" ]; then
  curl -fsS -m 30 -X POST -H "Authorization: Bearer $API_KEY" -H 'content-type: application/json' \
    "$BASE_URL/v1/chat/completions" \
    -d "{\"model\":\"$HARNESS_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" \
    >/dev/null 2>&1 || true
else
  curl -fsS -m 30 -X POST -H 'content-type: application/json' \
    "$BASE_URL/v1/chat/completions" \
    -d "{\"model\":\"$HARNESS_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1,\"stream\":false}" \
    >/dev/null 2>&1 || true
fi

# --- Run harness ---
REMAINING=$((DEADLINE - $(date +%s)))
if [ "$REMAINING" -lt 60 ]; then
  echo "[$(date +%T)] CELL FAIL pre-bench-out-of-budget tag=$TAG" >&2
  echo "[$(date +%T)] $TAG | $BACKEND | $KV_QUANT | success=false | coh=false | OUT_OF_BUDGET" >>"$PROGRESS"
  stop_server
  exit 3
fi

cd "$CBB"
HARNESS_ARGS=(
  --backend "$BACKEND"
  --backend-version "${BACKEND_VERSION_OVERRIDE:-head}"
  --base-url "$BASE_URL"
  --model "$HARNESS_MODEL"
  --model-path "$MODEL_PATH"
  --quant "$QUANT_SIG"
  --device gpu
  --max-tokens 8192
  --runs 2
  --backend-pid "$BACKEND_PID"
  --model-disk-gb "$MODEL_DISK_GB"
  --prompt-file "$REPO_ROOT/prompts/longctx_8k.json"
  --request-timeout 1800
)
if [ -n "$API_KEY" ]; then
  HARNESS_ARGS+=(--api-key "$API_KEY")
fi

timeout "${REMAINING}s" uv run python -m runners.run_one "${HARNESS_ARGS[@]}" >"$HARNESS_LOG" 2>&1
HRC=$?

# --- Parse harness output ---
COLD_MS=0; WARM_MS=0; DECODE_TPS=0; SUCCESS=true; COH=true
# Two lines from harness: [1/2] ttft=Xms decode=Y.YY ...
LINES=$(grep -E "^\[[0-9]+/[0-9]+\]" "$HARNESS_LOG" || true)
LINE_COUNT=$(echo "$LINES" | grep -c "^\[" || true)
if [ "$LINE_COUNT" -lt 1 ]; then
  SUCCESS=false; COH=false
fi
COLD_LINE=$(echo "$LINES" | sed -n '1p')
WARM_LINE=$(echo "$LINES" | sed -n '2p')
if [ -n "$COLD_LINE" ]; then
  COLD_MS=$(echo "$COLD_LINE" | grep -oE "ttft=[0-9.]+ms" | sed -nE '1s/ttft=([0-9.]+)ms/\1/p')
fi
if [ -n "$WARM_LINE" ]; then
  WARM_MS=$(echo "$WARM_LINE" | grep -oE "ttft=[0-9.]+ms" | sed -nE '1s/ttft=([0-9.]+)ms/\1/p')
  DECODE_TPS=$(echo "$WARM_LINE" | grep -oE "decode=[0-9.]+tps" | sed -nE '1s/decode=([0-9.]+)tps/\1/p')
fi
# Prefill TPS is deliberately NOT computed here. The formula this line used to
# carry -- 8192000 / warm_ttft_ms -- was fabricated twice over: the 8192 was
# hard-coded regardless of the real prompt length, and `warm_ttft_ms` is a
# prompt-cache HIT, so the quotient described a prefill that never ran. It
# produced ~109 000 and ~88 000 tok/s on the llama.cpp arms. A backend's own
# `timings.prompt_per_second` is the real figure and is what
# `scripts/bench/llama_ab.sh` records; nothing downstream of this progress line
# needs a number here.
PREFILL_TPS="n/a"

# Success check per harness output.
if grep -q "success=False" <<<"$LINES"; then SUCCESS=false; fi
if grep -q "err=other" <<<"$LINES"; then SUCCESS=false; fi

# Coherence: harness output_first_64 is in jsonl, examine.
LATEST_JSONL=$(ls -t "$CBB"/metrics/runs/${BACKEND}_*.jsonl 2>/dev/null | sed -n 1p)
if [ -n "$LATEST_JSONL" ]; then
  FIRST64=$(python3 -c "import json; lines=open('$LATEST_JSONL').readlines(); rec=json.loads(lines[-1]); print((rec.get('output_first_64') or '')[:64])" 2>/dev/null)
  if [ -z "$FIRST64" ]; then COH=false; fi
fi

if [ "$HRC" -ne 0 ]; then SUCCESS=false; fi
# Zero-decode = backend bug (e.g. Mixed cache unsupported on this arch).
if [ "$(echo "$DECODE_TPS" | grep -c '^0\(\.0*\)\?$')" -eq 1 ]; then
  SUCCESS=false; COH=false
fi

# --- Append progress line ---
echo "[$(date +%T)] $TAG | $BACKEND | $KV_QUANT | success=$SUCCESS | coh=$COH | cold=${COLD_MS} warm=${WARM_MS} prefill=${PREFILL_TPS} decode=${DECODE_TPS}" >>"$PROGRESS"

# --- Release ---
stop_server

echo "[$(date +%T)] CELL DONE tag=$TAG backend=$BACKEND kv=$KV_QUANT success=$SUCCESS coh=$COH decode=$DECODE_TPS" >&2
exit 0
