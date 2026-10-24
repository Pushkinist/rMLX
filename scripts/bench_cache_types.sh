#!/usr/bin/env bash
# bench_cache_types.sh — drive the cache-type combo matrix for a single model.
#
# Loops a fixed combo list, invokes `scripts/bench_cell.sh` once per cell with the
# env-var contract (CTK/CTV/PROMPT_ID/MAX_TOKENS/CTX_MAX/WARMUP_RUNS/
# MEASURED_RUNS/LABEL_PREFIX), and lets the cell driver ingest a §8.5
# RunRecord into <RMLX_HOME>/metrics/runs.db.
#
# Per combo:
#   1. Smoke probe: short generation via `rmlx baseline`. Reject empty /
#      non-UTF-8 / 4-gram-repeat-heavy output. On fail, log it in the summary
#      and skip the cell.
#   2. Call `scripts/bench_cell.sh` for the real measurement.
#   3. Exit code 78 from `scripts/bench_cell.sh` (unsupported resolver combo) =>
#      log a skip in the summary and continue.
#
# A cell that fails or is skipped measured nothing, so it writes no runs.db
# record: the recorder refuses a record with no measurement.
#
# Every step is a one-shot `rmlx` command that holds the Metal claim for its own
# run, so nothing here stops a process. A claim another process holds fails the
# step with exit 11 and names the holder.
#
# Usage:
#   scripts/bench_cache_types.sh MODEL_PATH ARCH_CLASS [PROMPT_TOKENS]
#
#   MODEL_PATH      abs path to snapshot directory
#   ARCH_CLASS      e.g. Qwen3ForCausalLM
#   PROMPT_TOKENS   optional; default 4096 (=> longctx_4k.json)
#
# The weight label is read from the snapshot's config.json through
# scripts/lib/snapshot_identity.py, the one shell-side producer of it; there is
# no argument to override it.
#
set -uo pipefail

# ---------------------------------------------------------------------------
# Args + defaults
# ---------------------------------------------------------------------------
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "usage: $0 MODEL_PATH ARCH_CLASS [PROMPT_TOKENS]" >&2
  echo "  (the weight label comes from the snapshot's config.json; it is not an argument)" >&2
  exit 2
fi

MODEL_PATH="$1"
ARCH="$2"
PROMPT_TOKENS="${3:-4096}"
case "$PROMPT_TOKENS" in
  '' | *[!0-9]*)
    echo "PROMPT_TOKENS must be a token count, got '$PROMPT_TOKENS'" >&2
    exit 2
    ;;
esac

if [ ! -d "$MODEL_PATH" ]; then
  echo "model path does not exist: $MODEL_PATH" >&2
  exit 2
fi

# The weight label is the one the snapshot's own config describes, read
# through the one shell-side producer of it.
if ! _IDENTITY="$(python3 "$(dirname "$0")/lib/snapshot_identity.py" "$MODEL_PATH")"; then
  echo "cannot label the snapshot's weights (above)" >&2
  exit 2
fi
WEIGHT_BITS_IN="$(printf '%s\n' "$_IDENTITY" | sed -n 's/^weight_quant=//p')"

# Map prompt tokens → prompt id (longctx_<N/1024>k).
_PROMPT_K=$((PROMPT_TOKENS / 1024))
PROMPT_ID="longctx_${_PROMPT_K}k"

# Map prompt tokens → ctx-max (round up to next 8k for safe headroom).
if [ "$PROMPT_TOKENS" -le 4096 ]; then CTX_MAX_DEFAULT=8192
elif [ "$PROMPT_TOKENS" -le 8192 ]; then CTX_MAX_DEFAULT=16384
elif [ "$PROMPT_TOKENS" -le 16384 ]; then CTX_MAX_DEFAULT=32768
else CTX_MAX_DEFAULT=$((PROMPT_TOKENS * 2)); fi

RMLX_DIR=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=scripts/lib/record_pending.sh
. "$RMLX_DIR/scripts/lib/record_pending.sh"
# Smoke probes that measured and whose diagnostic record did not land.
SMOKE_RECORDS_PENDING=0
RMLX_BIN="$RMLX_DIR/target/release/rmlx"

if [ ! -x "$RMLX_BIN" ]; then
  echo "rmlx binary not found / not executable: $RMLX_BIN" >&2
  exit 2
fi

RUN_TS=$(date -u +"%Y%m%dT%H%M%SZ")
MODEL_BASENAME=$(basename "$MODEL_PATH")
MODEL_NAME="${MODEL_BASENAME#*__}"

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

# Build the canonical kv_quant string for a (ctk, ctv, kv_quant_arg) tuple.
# Mirrors the formula in scripts/bench_cell.sh, so the summary names a cell as
# its record does.
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
    --record --run-class diagnostic --label "ctype-smoke-${LABEL}"
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
  # A probe that measured and whose record did not land is judged on its
  # output like any other; the buffer file is named and the run ends non-zero.
  if [ "$rc" -eq "$RECORD_PENDING_EXIT" ]; then
    record_pending_note "$smoke_log"
    SMOKE_RECORDS_PENDING=$((SMOKE_RECORDS_PENDING + 1))
    rc=0
  fi
  if [ "$rc" -ne 0 ]; then
    SMOKE_REASON="baseline_exit_${rc}"
    # Pass through exit 78 (unsupported combo) to caller via SMOKE_REASON.
    if [ "$rc" -eq 78 ]; then
      SMOKE_REASON="resolver_skip"
    fi
    rm -f "$smoke_log"
    return "$rc"
  fi

  # Parse TPS from the baseline summary line. If TPS < 0.5 the run
  # succeeded syntactically but produced no real decode (e.g. Gemma4 Shared-KV
  # rejects Mixed codec, emits TPS=0.000). Treat this as a runtime failure.
  local smoke_tps
  smoke_tps=$(grep -E "^baseline: " "$smoke_log" | tail -1 | grep -oE "TPS=[0-9.]+" | sed -nE '1s/TPS=([0-9.]+)/\1/p')
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

  # compute_kv_canon emits ct_<k>_<v> for non-numeric per-side codecs
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

  # 1. Smoke probe.
  if ! run_smoke_probe "$CTK_VAL" "$CTV_VAL" "$KV_ARG"; then
    rc=$?
    if [ "$SMOKE_REASON" = "resolver_skip" ]; then
      echo "[$(date +%T)] CELL SKIP (resolver) label=$LABEL" >&2
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "skip" >>"$SUMMARY_FILE"
    else
      # tps_zero is a runtime failure (not a smoke content failure).
      local _status="smoke_fail"
      case "$SMOKE_REASON" in
        tps_zero=*) _status="runtime_fail" ;;
      esac
      echo "[$(date +%T)] CELL SMOKE_FAIL label=$LABEL reason=$SMOKE_REASON status=$_status" >&2
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "$_status" >>"$SUMMARY_FILE"
    fi
    continue
  fi

  # 2. Real measurement via scripts/bench_cell.sh.
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
      MEDIAN=$(echo "$DONE_LINE" | grep -oE "median=[0-9.]+" | sed -nE '1s/median=([0-9.]+)/\1/p')
      STDDEV=$(echo "$DONE_LINE" | grep -oE "stddev=[0-9.]+" | sed -nE '1s/stddev=([0-9.]+)/\1/p')
      KV_FROM=$(echo "$DONE_LINE" | grep -oE "kv=[A-Za-z0-9_]+" | sed -nE '1s/kv=(.+)/\1/p')
      [ -n "$KV_FROM" ] && KV_CANON="$KV_FROM"
      echo "[$(date +%T)] CELL OK label=$LABEL kv=$KV_CANON median=${MEDIAN:-?} stddev=${STDDEV:-?}" >&2
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "${MEDIAN:-}" "${STDDEV:-}" "ok" >>"$SUMMARY_FILE"
      ;;
    78)
      echo "[$(date +%T)] CELL RESOLVER_SKIP label=$LABEL" >&2
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "skip" >>"$SUMMARY_FILE"
      ;;
    10)
      # exit 10 = TPS below threshold in scripts/bench_cell.sh CT-mode loop.
      echo "[$(date +%T)] CELL RUNTIME_FAIL (tps=0) label=$LABEL" >&2
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "0.0" "" "runtime_fail" >>"$SUMMARY_FILE"
      ;;
    *)
      # Tail of log helps diagnose.
      echo "[$(date +%T)] CELL FAIL label=$LABEL rc=$CELL_RC (log: $CELL_LOG)" >&2
      tail -15 "$CELL_LOG" >&2 || true
      printf "%s\t%s\t%s\t%s\t%s\n" "$LABEL" "$KV_CANON" "" "" "fail_rc${CELL_RC}" >>"$SUMMARY_FILE"
      ;;
  esac
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

if [ "$SMOKE_RECORDS_PENDING" -gt 0 ]; then
  echo "INCOMPLETE: $SMOKE_RECORDS_PENDING smoke record(s) pending (named above); the cells stand" >&2
  exit "$RECORD_PENDING_EXIT"
fi
exit 0
