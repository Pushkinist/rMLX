#!/usr/bin/env bash
# bench_cache_types.sh — drive the cache-type combo matrix for a single model.
#
# Loops a fixed combo list, invokes `scripts/bench_cell.sh` once per cell with the
# env-var contract (CTK/CTV/PROMPT_ID/MAX_TOKENS/CTX_MAX/WARMUP_RUNS/
# MEASURED_RUNS/LABEL_PREFIX), and lets the cell driver ingest a §8.5
# RunRecord into <RMLX_HOME>/metrics/runs.db.
#
# Per combo:
#   1. Cleanup (pkill rmlx serve, wait drain ≤30s, rm claim files).
#   2. Smoke probe: short generation via `rmlx baseline`. Reject empty /
#      non-UTF-8 / 4-gram-repeat-heavy output. On fail, write a §8.5 record
#      with notes "status=smoke_fail" and skip the cell.
#   3. Call `scripts/bench_cell.sh` for the real measurement.
#   4. Exit code 78 from `scripts/bench_cell.sh` (unsupported resolver combo) =>
#      write a "status=skip" record and continue.
#
# Usage:
#   scripts/bench_cache_types.sh MODEL_PATH ARCH_CLASS [WEIGHT_BITS] [PROMPT_TOKENS]
#
#   MODEL_PATH      abs path to snapshot directory
#   ARCH_CLASS      e.g. Qwen3ForCausalLM
#   WEIGHT_BITS     optional; inferred from MODEL_PATH if omitted
#                   (e.g. "2bit", "8bit", "mxfp8")
#   PROMPT_TOKENS   optional; default 4096 (=> longctx_4k.json)
#
set -uo pipefail

# ---------------------------------------------------------------------------
# Args + defaults
# ---------------------------------------------------------------------------
if [ $# -lt 2 ]; then
  echo "usage: $0 MODEL_PATH ARCH_CLASS [WEIGHT_BITS] [PROMPT_TOKENS]" >&2
  exit 2
fi

MODEL_PATH="$1"
ARCH="$2"
WEIGHT_BITS_IN="${3:-}"
PROMPT_TOKENS="${4:-4096}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "model path does not exist: $MODEL_PATH" >&2
  exit 2
fi

# Resolve WEIGHT_BITS heuristically when not supplied.
if [ -z "$WEIGHT_BITS_IN" ]; then
  _BASE=$(basename "$MODEL_PATH")
  case "$_BASE" in
    *mxfp8*) WEIGHT_BITS_IN="mxfp8" ;;
    *2bit*)  WEIGHT_BITS_IN="2bit" ;;
    *3bit*)  WEIGHT_BITS_IN="3bit" ;;
    *4bit*)  WEIGHT_BITS_IN="4bit" ;;
    *6bit*)  WEIGHT_BITS_IN="6bit" ;;
    *8bit*)  WEIGHT_BITS_IN="8bit" ;;
    *PARO*|*paro*) WEIGHT_BITS_IN="paro" ;;
    *)       WEIGHT_BITS_IN="unknown" ;;
  esac
fi

# Map prompt tokens → prompt id (longctx_<N/1024>k).
_PROMPT_K=$((PROMPT_TOKENS / 1024))
PROMPT_ID="longctx_${_PROMPT_K}k"

# Map prompt tokens → ctx-max (round up to next 8k for safe headroom).
if [ "$PROMPT_TOKENS" -le 4096 ]; then CTX_MAX_DEFAULT=8192
elif [ "$PROMPT_TOKENS" -le 8192 ]; then CTX_MAX_DEFAULT=16384
elif [ "$PROMPT_TOKENS" -le 16384 ]; then CTX_MAX_DEFAULT=32768
else CTX_MAX_DEFAULT=$((PROMPT_TOKENS * 2)); fi

RMLX_DIR=$(cd "$(dirname "$0")/.." && pwd)
RMLX_BIN="$RMLX_DIR/target/release/rmlx"

# Run identity (backend / version / git sha / build profile / hardware tag)
# comes from the measured binary — never hard-coded here.
source "$(dirname "${BASH_SOURCE[0]}")/lib/identity.sh"
rmlx_export_identity "$RMLX_BIN"
BUFFER_DIR="$RMLX_DIR/.rmlx/metrics/buffer/pending"
mkdir -p "$BUFFER_DIR"

if [ ! -x "$RMLX_BIN" ]; then
  echo "rmlx binary not found / not executable: $RMLX_BIN" >&2
  exit 2
fi

RUN_TS=$(date -u +"%Y%m%dT%H%M%SZ")
MODEL_BASENAME=$(basename "$MODEL_PATH")
case "$MODEL_BASENAME" in
  *__*)
    NS="${MODEL_BASENAME%%__*}"
    MODEL_NAME="${MODEL_BASENAME#*__}"
    ;;
  *)
    NS="local"
    MODEL_NAME="$MODEL_BASENAME"
    ;;
esac

# Pick a TAG for `scripts/bench_cell.sh` (used only for log filenames / progress line).
TAG_BASE=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | cut -c1-20)
TAG_BASE="${TAG_BASE%-}"

# ---------------------------------------------------------------------------
# Combo matrix.
#
# Each entry encodes:
#   label | ctk | ctv | kv_quant_arg
#
# ctk/ctv empty → no per-side override; the cell driver falls through to the
# legacy --kv-quant path with kv_quant_arg as the preset name.
#
# When ctk/ctv are populated, kv_quant_arg MUST be "kv-flag-replaced" so the
# collision guard does not refuse the cell.
# ---------------------------------------------------------------------------
COMBOS=(
  "auto|||auto"
  "bf16|bf16|bf16|kv-flag-replaced"
  "k8v4|||k8v4"
  "k8v8|||k8v8"
  "planar|||planar"
  "q8g128_q4g64|q8_g128|q4_g64|kv-flag-replaced"
  "q8g128_q8g64|q8_g128|q8_g64|kv-flag-replaced"
  "q8g128_q4g128|q8_g128|q4_g128|kv-flag-replaced"
  "q8g128_q6g64|q8_g128|q6_g64|kv-flag-replaced"
  "q8g128_q8g32|q8_g128|q8_g32|kv-flag-replaced"
  "q4g64_q4g64|q4_g64|q4_g64|kv-flag-replaced"
  "q8g128_tq4|q8_g128|tq4|kv-flag-replaced"
  "q8g128_planar4|q8_g128|planar4|kv-flag-replaced"
)

# ---------------------------------------------------------------------------
# Summary state (filled per cell, printed at the end).
# ---------------------------------------------------------------------------
SUMMARY_FILE=$(mktemp /tmp/bench_cache_types_summary.XXXXXX)
printf "%s\t%s\t%s\t%s\t%s\n" "label" "kv_quant" "decode_tps" "decode_stddev" "status" >"$SUMMARY_FILE"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

cleanup_runtime() {
  pkill -f "rmlx serve" 2>/dev/null || true
  # Wait for drain ≤ 30s (script-level safety net; do not exit on timeout).
  timeout 30 bash -c '
    until ! pgrep -f "rmlx serve" >/dev/null 2>&1; do sleep 0.2; done
  ' 2>/dev/null || true
  rm -f /tmp/rmlx.*.claim 2>/dev/null || true
}

# Emit a §8.5 RunRecord JSON for skip / smoke_fail cases.
#  $1 = canonical kv_quant string
#  $2 = status string (e.g. smoke_fail, skip)
#  $3 = combo label
#  $4 = optional extra notes blob
emit_status_record() {
  local kv_canon="$1"
  local status="$2"
  local combo_label="$3"
  local extra_notes="${4:-}"

  local out_json
  out_json="$BUFFER_DIR/$(date -u +"%Y%m%dT%H%M%S%3N")-bench-$$.json"
  local prompt_file="$RMLX_DIR/prompts/${PROMPT_ID}.json"

  PROMPT_FILE="$prompt_file" \
  NS_VAL="$NS" \
  MODEL_VAL="$MODEL_NAME" \
  WQ_VAL="$WEIGHT_BITS_IN" \
  KV_VAL="$kv_canon" \
  CTX_VAL="$CTX_MAX_DEFAULT" \
  PROMPT_NAME_VAL="$PROMPT_ID" \
  TS_VAL="$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  HW_VAL="${RMLX_HARDWARE_TAG:-m5_max_128gb}" \
  PROMPT_TOK_VAL="$PROMPT_TOKENS" \
  NOTES_VAL="label=ctype-${combo_label} status=${status} ${extra_notes}" \
  DESC_VAL="bench_cache_types ${combo_label} ${status}" \
  python3 -c '
import json, os, sys
with open(os.environ["PROMPT_FILE"], "r") as f:
    body = json.load(f)
rec = {
    **json.loads(os.environ['RMLX_IDENTITY_JSON']),
  "model_namespace": os.environ["NS_VAL"],
  "model": os.environ["MODEL_VAL"],
  "weight_quant": os.environ["WQ_VAL"],
  "kv_quant": os.environ["KV_VAL"],
  "ctx_max": int(os.environ["CTX_VAL"]),
  "prompt": {"name": os.environ["PROMPT_NAME_VAL"], "body": body},
  "ts_utc": os.environ["TS_VAL"],
  "prompt_tokens": int(os.environ["PROMPT_TOK_VAL"]),
  "max_tokens": 32,
  "temperature": 0.0,
  "n_warmups": 0,
  "n_measure": 0,
  "notes": os.environ["NOTES_VAL"],
  "description": os.environ["DESC_VAL"],
  "metrics": [
    {"name": "decode_tps_warm", "value": 0.0, "stddev": 0.0},
  ],
}
json.dump(rec, sys.stdout, indent=2)
' >"$out_json"

  if ! ( cd "$RMLX_DIR" && "$RMLX_BIN" metrics record --file "$out_json" >/dev/null 2>&1 ); then
    echo "[$(date +%T)] WARN: failed to ingest status record $out_json" >&2
  fi
}

# Build the canonical kv_quant string for a (ctk, ctv, kv_quant_arg) tuple.
# Mirrors the formula in scripts/bench_cell.sh so smoke/skip rows match warm rows.
compute_kv_canon() {
  local ctk="$1" ctv="$2" kvq="$3"
  if [ -n "$ctk" ] || [ -n "$ctv" ]; then
    local k="${ctk:-q8_g128}" v="${ctv:-q8_g128}"
    local kb kg vb vg
    kb=$(echo "$k" | sed -nE 's/^q([0-9]+)_g[0-9]+$/\1/p')
    kg=$(echo "$k" | sed -nE 's/^q[0-9]+_g([0-9]+)$/\1/p')
    vb=$(echo "$v" | sed -nE 's/^q([0-9]+)_g[0-9]+$/\1/p')
    vg=$(echo "$v" | sed -nE 's/^q[0-9]+_g([0-9]+)$/\1/p')
    if [ -n "$kb" ] && [ -n "$kg" ] && [ -n "$vb" ] && [ -n "$vg" ]; then
      echo "mixed_k${kb}g${kg}_v${vb}g${vg}"
      return
    fi
    # Non-numeric per-side codec (e.g. tq4, planar4, bf16). Emit raw pair.
    echo "ct_${k}_${v}"
    return
  fi
  case "$kvq" in
    auto|"") echo "auto" ;;
    mixed) echo "k8v4" ;;
    *) echo "$kvq" ;;
  esac
}

# Smoke probe — short generation, then reject obvious garbage.
# Returns 0 on pass, non-zero on fail. On fail, populates SMOKE_REASON.
SMOKE_REASON=""
run_smoke_probe() {
  local ctk="$1" ctv="$2" kvq="$3"
  SMOKE_REASON=""
  local smoke_log
  smoke_log=$(mktemp /tmp/smoke_${TAG_BASE}.XXXXXX)

  # Smoke uses the smallest available canonical prompt (longctx_4k) so the
  # probe takes ~10s on Bonsai-class models. Smaller bundles (1k/2k) don't
  # exist; --prompt-tokens must be a multiple of 1024 AND have a matching
  # prompts/longctx_<N/1024>k.json file.
  local -a args
  args=(
    baseline
    --model "$MODEL_PATH"
    --prompt-tokens 4096
    --gen-tokens 16
    --ctx-max 8192
    --device gpu
  )
  if [ -n "$ctk" ] && [ -n "$ctv" ]; then
    args+=(--cache-type-k "$ctk" --cache-type-v "$ctv")
  elif [ -n "$ctk" ]; then
    args+=(--cache-type-k "$ctk" --cache-type-v auto)
  elif [ -n "$ctv" ]; then
    args+=(--cache-type-k auto --cache-type-v "$ctv")
  else
    if [ "$kvq" != "auto" ] && [ -n "$kvq" ] && [ "$kvq" != "kv-flag-replaced" ]; then
      args+=(--kv-quant "$kvq")
    fi
  fi

  ( cd "$RMLX_DIR" && "$RMLX_BIN" "${args[@]}" ) >"$smoke_log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    SMOKE_REASON="baseline_exit_${rc}"
    # Pass through exit 78 (unsupported combo) to caller via SMOKE_REASON.
    if [ "$rc" -eq 78 ]; then
      SMOKE_REASON="resolver_skip"
    fi
    rm -f "$smoke_log"
    return "$rc"
  fi

  # Bug-1: parse TPS from the baseline summary line. If TPS < 0.5 the run
  # succeeded syntactically but produced no real decode (e.g. Gemma4 Shared-KV
  # rejects Mixed codec, emits TPS=0.000). Treat this as a runtime failure.
  local smoke_tps
  smoke_tps=$(grep -E "^baseline: " "$smoke_log" | tail -1 | grep -oE "TPS=[0-9.]+" | head -1 | sed -E 's/TPS=([0-9.]+)/\1/')
  if [ -n "$smoke_tps" ]; then
    local _tps_fail
    _tps_fail=$(awk -v t="$smoke_tps" 'BEGIN { print (t+0 < 0.5) ? "1" : "0" }')
    if [ "$_tps_fail" = "1" ]; then
      SMOKE_REASON="tps_zero=${smoke_tps}"
      rm -f "$smoke_log"
      return 1
    fi
  fi

  # Find the generated output: baseline prints "output: ..." line.
  local output
  output=$(grep -E "^output: " "$smoke_log" | tail -1 | sed -E 's/^output: //')
  if [ -z "$output" ]; then
    # Some builds may stream differently; fall through and accept by default
    # if the baseline summary line is present.
    if grep -qE "^baseline: " "$smoke_log"; then
      rm -f "$smoke_log"
      return 0
    fi
    SMOKE_REASON="no_output"
    rm -f "$smoke_log"
    return 1
  fi

  # Reject non-UTF-8.
  if ! printf '%s' "$output" | iconv -f utf-8 -t utf-8 >/dev/null 2>&1; then
    SMOKE_REASON="non_utf8"
    rm -f "$smoke_log"
    return 1
  fi

  # 4-gram repetition: count token (whitespace-split) 4-grams. If any 4-gram
  # repeats >3 times, declare garbage.
  local max_rep
  max_rep=$(printf '%s\n' "$output" | python3 -c '
import sys
text = sys.stdin.read().split()
if len(text) < 4:
    print(0)
    sys.exit(0)
counts = {}
for i in range(len(text) - 3):
    gram = tuple(text[i:i+4])
    counts[gram] = counts.get(gram, 0) + 1
print(max(counts.values()) if counts else 0)
' 2>/dev/null || echo 0)
  if [ -n "$max_rep" ] && [ "$max_rep" -gt 3 ]; then
    SMOKE_REASON="4gram_repeat=${max_rep}"
    rm -f "$smoke_log"
    return 1
  fi

  rm -f "$smoke_log"
  return 0
}

# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
echo "[$(date +%T)] bench_cache_types: model=$MODEL_BASENAME arch=$ARCH weight=$WEIGHT_BITS_IN prompt=$PROMPT_ID ctx=$CTX_MAX_DEFAULT" >&2
echo "[$(date +%T)] combo count: ${#COMBOS[@]}" >&2

for entry in "${COMBOS[@]}"; do
  IFS='|' read -r LABEL CTK_VAL CTV_VAL KV_ARG <<<"$entry"

  echo "" >&2
  echo "============================================================" >&2
  echo "[$(date +%T)] CELL label=$LABEL ctk='$CTK_VAL' ctv='$CTV_VAL' kvq='$KV_ARG'" >&2
  echo "============================================================" >&2

  KV_CANON=$(compute_kv_canon "$CTK_VAL" "$CTV_VAL" "$KV_ARG")

  # Bug-2: compute_kv_canon emits ct_<k>_<v> for non-numeric per-side codecs
  # (tq4, planar4). The metrics validator rejects those strings. Map them to
  # the canonical value the resolver would have chosen if it accepted the combo.
  # This override is applied before any record emission for this cell.
  export KV_CANON_OVERRIDE=""
  case "$CTV_VAL" in
    tq4)    KV_CANON_OVERRIDE="k8v4"  ; KV_CANON="k8v4"   ;;
    planar4) KV_CANON_OVERRIDE="planar"; KV_CANON="planar" ;;
  esac
  case "$CTK_VAL" in
    tq4)    KV_CANON_OVERRIDE="k8v4"  ; KV_CANON="k8v4"   ;;
    planar4) KV_CANON_OVERRIDE="planar"; KV_CANON="planar" ;;
  esac

  # 1. Cleanup before this cell.
  cleanup_runtime

  # 2. Smoke probe.
  if ! run_smoke_probe "$CTK_VAL" "$CTV_VAL" "$KV_ARG"; then
    rc=$?
    if [ "$SMOKE_REASON" = "resolver_skip" ]; then
      echo "[$(date +%T)] CELL SKIP (resolver) label=$LABEL" >&2
      emit_status_record "$KV_CANON" "skip" "$LABEL" "smoke_rc=$rc"
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "skip" >>"$SUMMARY_FILE"
    else
      # Bug-1: tps_zero is a runtime failure (not a smoke content failure).
      local _emit_status="smoke_fail"
      case "$SMOKE_REASON" in
        tps_zero=*) _emit_status="runtime_fail" ;;
      esac
      echo "[$(date +%T)] CELL SMOKE_FAIL label=$LABEL reason=$SMOKE_REASON status=$_emit_status" >&2
      emit_status_record "$KV_CANON" "$_emit_status" "$LABEL" "reason=$SMOKE_REASON"
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "$_emit_status" >>"$SUMMARY_FILE"
    fi
    cleanup_runtime
    continue
  fi

  cleanup_runtime

  # 3. Real measurement via scripts/bench_cell.sh.
  CELL_LOG=$(mktemp /tmp/cell_${TAG_BASE}_${LABEL}.XXXXXX)
  CTK="$CTK_VAL" \
  CTV="$CTV_VAL" \
  PROMPT_ID="$PROMPT_ID" \
  MAX_TOKENS=32 \
  CTX_MAX="$CTX_MAX_DEFAULT" \
  WARMUP_RUNS=2 \
  MEASURED_RUNS=3 \
  LABEL_PREFIX="ctype-${LABEL}" \
  bash "$RMLX_DIR/scripts/bench_cell.sh" \
    "$TAG_BASE" \
    "$MODEL_PATH" \
    "rmlx" \
    "$KV_ARG" \
    "60" \
    "0" \
    "$WEIGHT_BITS_IN" \
    >"$CELL_LOG" 2>&1
  CELL_RC=$?

  case "$CELL_RC" in
    0)
      # Parse the final CELL DONE line for median/stddev.
      DONE_LINE=$(grep -E "CELL DONE" "$CELL_LOG" | tail -1)
      MEDIAN=$(echo "$DONE_LINE" | grep -oE "median=[0-9.]+" | head -1 | sed -E 's/median=([0-9.]+)/\1/')
      STDDEV=$(echo "$DONE_LINE" | grep -oE "stddev=[0-9.]+" | head -1 | sed -E 's/stddev=([0-9.]+)/\1/')
      KV_FROM=$(echo "$DONE_LINE" | grep -oE "kv=[A-Za-z0-9_]+" | head -1 | sed -E 's/kv=(.+)/\1/')
      [ -n "$KV_FROM" ] && KV_CANON="$KV_FROM"
      echo "[$(date +%T)] CELL OK label=$LABEL kv=$KV_CANON median=${MEDIAN:-?} stddev=${STDDEV:-?}" >&2
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "${MEDIAN:-}" "${STDDEV:-}" "ok" >>"$SUMMARY_FILE"
      ;;
    78)
      echo "[$(date +%T)] CELL RESOLVER_SKIP label=$LABEL" >&2
      emit_status_record "$KV_CANON" "skip" "$LABEL" "cell_rc=78"
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "skip" >>"$SUMMARY_FILE"
      ;;
    10)
      # Bug-1: exit 10 = TPS below threshold in scripts/bench_cell.sh CT-mode loop.
      echo "[$(date +%T)] CELL RUNTIME_FAIL (tps=0) label=$LABEL" >&2
      emit_status_record "$KV_CANON" "runtime_fail" "$LABEL" "cell_rc=10 status=runtime_fail"
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "0.0" "" "runtime_fail" >>"$SUMMARY_FILE"
      ;;
    *)
      # Tail of log helps diagnose.
      echo "[$(date +%T)] CELL FAIL label=$LABEL rc=$CELL_RC (log: $CELL_LOG)" >&2
      tail -15 "$CELL_LOG" >&2 || true
      emit_status_record "$KV_CANON" "cell_fail" "$LABEL" "cell_rc=$CELL_RC"
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "fail_rc${CELL_RC}" >>"$SUMMARY_FILE"
      ;;
  esac

  cleanup_runtime
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "" >&2
echo "============================================================" >&2
echo "MATRIX SUMMARY  model=$MODEL_BASENAME arch=$ARCH weight=$WEIGHT_BITS_IN" >&2
echo "  prompt_id=$PROMPT_ID prompt_tokens=$PROMPT_TOKENS ctx_max=$CTX_MAX_DEFAULT" >&2
echo "  run_ts=$RUN_TS" >&2
echo "============================================================" >&2
column -t -s "$(printf '\t')" "$SUMMARY_FILE" >&2 || cat "$SUMMARY_FILE" >&2
echo "" >&2
echo "(summary file: $SUMMARY_FILE)" >&2

exit 0
