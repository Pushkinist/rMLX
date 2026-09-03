#!/usr/bin/env bash
# prefill_chunk_sweep.sh — Latin-square sweep of one architecture's prefill
# chunk over a set of levels and prompt lengths, recorded to the metrics DB.
#
# WHY THIS EXISTS
#
# `arch_default` in `crates/rmlx-models/src/prefill_chunk.rs` carries one chunk
# size per architecture, and each row is supposed to be backed by a measurement
# on real models. This is the measurement. It exists as a script rather than as
# a transcript in an issue so the next architecture's row costs a command, and
# so the cells land in `runs.db` where a program can read them back instead of
# being retyped into prose.
#
# WHY A LATIN SQUARE AND NOT ABBA
#
# A slot's position in the run affects what it measures, and on this host that
# positional term is not linear: one slot position runs slow for whichever
# level occupies it. An ABBA block cancels only a monotone drift, so it leaves
# that penalty sitting on one arm — which is how two confident wrong calls have
# been made here. This runs a cyclic Latin square instead: level index
# `(position + row) % n_levels`, so across `n_levels` rows every level occupies
# every slot position exactly once and the positional term cancels by
# construction. See `docs/PROFILING.md` §"Ordering on this host".
#
# The verdict it prints is the PAIRED statistic: each level against the
# baseline level *within a row*, median over rows. A pooled median across slots
# would re-admit the positional term this design removes.
#
# THE ENV VAR IS DERIVED, NOT PASSED
#
# `--arch-key qwen3` selects `RMLX_PREFILL_CHUNK_QWEN3`. Taking the variable
# name as an argument would let the tool sweep an unrelated knob while still
# filing its cells under `decode_config = 'prefill_chunk=<n>'` — a wrong cell
# label, which in an append-only store cannot be taken back out.
#
# WHAT IT WRITES
#
# Raw per-slot bench JSON and the records under `--out` (default
# `$RMLX_HOME/bench/prefill_chunk_sweep/<timestamp>`), and with `--record`, one
# §8.5 record per (model, prompt length, level) through
# `rmlx metrics record`. Each slot is a full `rmlx` process that writes its own
# `$RMLX_HOME/logs/<run-id>.jsonl`; point RMLX_HOME at a scratch directory when
# that matters. Slots run with `--metrics off` so the only rows that reach the
# DB are the aggregated ones, after the gates have passed.
#
# Exit codes:
#   0   — ran cleanly; the verdict is on stdout
#   1   — correctness failure: two levels of one cell produced different tokens
#   125 — not usable: busy host, missing binary/model/prompt, bad arguments

# No `pipefail`: several parses here end in `| head -1`, which SIGPIPEs its
# producer once it has what it needs. Every value parsed is checked for
# emptiness immediately after and every subprocess status is tested.
set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
AWK_BUSIEST="${REPO_ROOT}/scripts/lib/busiest_between.awk"

# shellcheck source=scripts/lib/cpu_snapshot.sh
source "${REPO_ROOT}/scripts/lib/cpu_snapshot.sh"
# shellcheck source=scripts/lib/identity.sh
source "${REPO_ROOT}/scripts/lib/identity.sh"

BINARY="${REPO_ROOT}/target/release-perf/rmlx"
MODEL=""
ARCH_KEY=""
LEVELS="256 512 1024 2048 4096"
PROMPT_TOKENS="4096 32768"
ROWS=0
GEN_TOKENS=100
KV_QUANT="none"
RUNS=2
BUSY_PCT=25
OUT_DIR=""
DB_PATH=""
RECORD=false

usage() {
	cat <<'USAGE'
usage: scripts/prefill_chunk_sweep.sh --model <snapshot> --arch-key <key> [options]

Required:
  --model PATH           model snapshot directory
  --arch-key KEY         module-style architecture key as spelled in
                         `arch_default` (qwen3, gemma4, qwen3_5_moe, ...). It
                         selects RMLX_PREFILL_CHUNK_<KEY upper-cased>.

Options:
  --levels "A B C"       chunk levels to sweep      (default: 256 512 1024 2048 4096)
  --prompt-tokens "N M"  prompt lengths, each a multiple of 1024 with a
                         canonical prompts/longctx_<k>k.json
                                                    (default: 4096 32768)
  --rows N               Latin-square rows          (default: one per level)
  --runs N               measured runs per slot     (default: 2)
  --gen-tokens N         tokens generated per run   (default: 100)
  --kv-quant NAME        codec for every slot       (default: none)
  --binary PATH          rmlx binary                (default: target/release-perf/rmlx)
  --busy-pct N           foreign-CPU taint threshold, % of one core (default: 25)
  --out DIR              output directory
  --record               ingest the cells into runs.db through the §8.5 path
  --db PATH              metrics DB for --record    (default: the resolved one)
USAGE
}

need_value() {
	if [[ $2 -lt 2 ]]; then
		echo "ERROR: $1 requires a value" >&2
		exit 125
	fi
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--model) need_value "$1" $#; MODEL="$2"; shift 2 ;;
	--arch-key) need_value "$1" $#; ARCH_KEY="$2"; shift 2 ;;
	--levels) need_value "$1" $#; LEVELS="$2"; shift 2 ;;
	--prompt-tokens) need_value "$1" $#; PROMPT_TOKENS="$2"; shift 2 ;;
	--rows) need_value "$1" $#; ROWS="$2"; shift 2 ;;
	--runs) need_value "$1" $#; RUNS="$2"; shift 2 ;;
	--gen-tokens) need_value "$1" $#; GEN_TOKENS="$2"; shift 2 ;;
	--kv-quant) need_value "$1" $#; KV_QUANT="$2"; shift 2 ;;
	--binary) need_value "$1" $#; BINARY="$2"; shift 2 ;;
	--busy-pct) need_value "$1" $#; BUSY_PCT="$2"; shift 2 ;;
	--out) need_value "$1" $#; OUT_DIR="$2"; shift 2 ;;
	--db) need_value "$1" $#; DB_PATH="$2"; shift 2 ;;
	--record) RECORD=true; shift ;;
	-h | --help) usage; exit 0 ;;
	*) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 125 ;;
	esac
done

[[ -n "$MODEL" ]] || { echo "ERROR: --model is required" >&2; exit 125; }
[[ -n "$ARCH_KEY" ]] || { echo "ERROR: --arch-key is required" >&2; exit 125; }
[[ -d "$MODEL" ]] || { echo "ERROR: model path not found: $MODEL" >&2; exit 125; }
[[ -x "$BINARY" ]] || { echo "ERROR: binary not found / not executable: $BINARY" >&2; exit 125; }

read -r -a LEVEL_LIST <<<"$LEVELS"
read -r -a PTOK_LIST <<<"$PROMPT_TOKENS"
N_LEVELS=${#LEVEL_LIST[@]}
[[ $N_LEVELS -ge 2 ]] || { echo "ERROR: --levels needs at least two levels" >&2; exit 125; }
[[ ${#PTOK_LIST[@]} -ge 1 ]] || { echo "ERROR: --prompt-tokens needs at least one length" >&2; exit 125; }
[[ $ROWS -gt 0 ]] || ROWS=$N_LEVELS

for ptok in "${PTOK_LIST[@]}"; do
	k=$((ptok / 1024))
	if [[ $((k * 1024)) -ne $ptok || $k -lt 1 ]]; then
		echo "ERROR: --prompt-tokens $ptok is not a positive multiple of 1024" >&2
		exit 125
	fi
	if [[ ! -f "${REPO_ROOT}/prompts/longctx_${k}k.json" ]]; then
		echo "ERROR: no canonical prompt for $ptok tokens (prompts/longctx_${k}k.json)" >&2
		exit 125
	fi
done

ENV_VAR="RMLX_PREFILL_CHUNK_$(printf '%s' "$ARCH_KEY" | tr '[:lower:]' '[:upper:]')"
BASELINE_LEVEL="${LEVEL_LIST[0]}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-${RMLX_HOME}/bench/prefill_chunk_sweep/${TS}}"
mkdir -p "$OUT_DIR/raw"

# The measured binary is the measurement, not interference; anything else
# burning a core during a slot is.
export CPU_SNAPSHOT_SKIP="$(basename "$BINARY") rmlx rmlx_main"

# An `rmlx serve` holds the Metal context for the whole run, which is exactly
# the confound this design removes for CPU. It is reported, not killed —
# killing it would destroy someone else's work to make this number look better.
if pgrep -f "rmlx serve" >/dev/null 2>&1; then
	echo "ERROR: an 'rmlx serve' process is running and holds the Metal context." >&2
	echo "  Stop it before measuring:  pkill -f 'rmlx serve'" >&2
	exit 125
fi

rmlx_export_identity "$BINARY"

# Classify one CPU window. `unmeasured` is not `quiet`: not knowing must never
# be folded into "nothing was there", or the gate silently stops gating.
classify_window() {
	local before="$1" after="$2" seconds="$3" raw pct
	if [[ -e "$before.failed" || -e "$after.failed" ]]; then
		echo "unmeasured - -"
		return
	fi
	raw="$(awk -v window="$seconds" -f "$AWK_BUSIEST" "$before" "$after")"
	case "${raw%% *}" in
	unmeasured) echo "unmeasured - -" ;;
	idle) echo "quiet 0.0 -" ;;
	*)
		pct="$(echo "$raw" | awk '{print $2}')"
		if awk -v p="$pct" -v t="$BUSY_PCT" 'BEGIN { exit !(p >= t) }'; then
			echo "busy ${raw#* }"
		else
			echo "quiet ${raw#* }"
		fi
		;;
	esac
}

SHORT="$(basename "$MODEL")"
cat <<HEADER
========================================================================
rMLX prefill-chunk sweep — cyclic Latin square, ${ROWS} rows x ${N_LEVELS} levels
  model      $SHORT
  arch key   $ARCH_KEY  ->  $ENV_VAR
  levels     ${LEVEL_LIST[*]}   (baseline $BASELINE_LEVEL)
  prompts    ${PTOK_LIST[*]} tokens
  kv-quant   $KV_QUANT   gen-tokens $GEN_TOKENS   runs/slot $RUNS
  binary     $BINARY
  out        $OUT_DIR

INTERFERENCE GATE: a foreign process at or above ${BUSY_PCT}% of one core over
  any slot, or over the sweep as a whole, refuses the result. It is measured
  from cumulative CPU time, so it sees anything running at the start of a
  window, at its end, or throughout — but not a process that both starts and
  exits inside one window.
========================================================================
HEADER

TAINTED=""
SWEEP_BEFORE="$OUT_DIR/cpu_sweep_before"
SWEEP_AFTER="$OUT_DIR/cpu_sweep_after"
snapshot_ok "$SWEEP_BEFORE" || true
SWEEP_START="$(date +%s)"

for ptok in "${PTOK_LIST[@]}"; do
	max_ctx=$((ptok + GEN_TOKENS + 4096))
	echo "$max_ctx" >"$OUT_DIR/raw/p${ptok}.max_ctx"
	echo ""
	echo "==> prompt_tokens=$ptok  max_ctx=$max_ctx"
	for ((row = 0; row < ROWS; row++)); do
		for ((pos = 0; pos < N_LEVELS; pos++)); do
			level="${LEVEL_LIST[$(((pos + row) % N_LEVELS))]}"
			out="$OUT_DIR/raw/p${ptok}_c${level}_r${row}_s${pos}.json"
			before="$OUT_DIR/raw/cpu_p${ptok}_c${level}_r${row}_s${pos}_before"
			after="${before%_before}_after"

			snapshot_ok "$before" || true
			slot_start="$(date +%s)"
			set +e
			env "$ENV_VAR=$level" "$BINARY" bench --metrics off \
				--model "$MODEL" --prompt-tokens "$ptok" --kv-quant "$KV_QUANT" \
				--max-ctx "$max_ctx" --gen-tokens "$GEN_TOKENS" \
				--warmup 1 --runs "$RUNS" --json \
				>"$out" 2>"${out%.json}.err"
			rc=$?
			set -e
			slot_seconds=$(($(date +%s) - slot_start))
			snapshot_ok "$after" || true

			if [[ $rc -ne 0 ]]; then
				echo "ERROR: slot row=$row pos=$pos level=$level failed (rc=$rc); see ${out%.json}.err" >&2
				exit 125
			fi

			verdict="$(classify_window "$before" "$after" "$slot_seconds")"
			case "${verdict%% *}" in
			busy) TAINTED="${TAINTED}slot p${ptok} r${row} s${pos} level ${level}: ${verdict}"$'\n' ;;
			unmeasured) TAINTED="${TAINTED}slot p${ptok} r${row} s${pos} level ${level}: window not measurable"$'\n' ;;
			esac
			printf '  row %d slot %d level %5s  %s\n' "$row" "$pos" "$level" "$verdict"
		done
	done
done

SWEEP_SECONDS=$(($(date +%s) - SWEEP_START))
snapshot_ok "$SWEEP_AFTER" || true
SWEEP_VERDICT="$(classify_window "$SWEEP_BEFORE" "$SWEEP_AFTER" "$SWEEP_SECONDS")"
case "${SWEEP_VERDICT%% *}" in
busy) TAINTED="${TAINTED}whole sweep: ${SWEEP_VERDICT}"$'\n' ;;
unmeasured) TAINTED="${TAINTED}whole sweep: window not measurable"$'\n' ;;
esac

if [[ -n "$TAINTED" ]]; then
	echo ""
	echo "TAINTED — the host was not quiet, so these slots are not comparable:" >&2
	printf '%s' "$TAINTED" >&2
	echo "Nothing was recorded. Re-run on a quiet machine." >&2
	exit 125
fi

# ---- verdict + §8.5 records --------------------------------------------------
#
# Python from here: the paired median over rows and the record bodies are
# arithmetic, and awk floats plus a hand-rolled JSON writer is how a sweep ends
# up disagreeing with the DB it wrote.

RMLX_SWEEP_OUT="$OUT_DIR" \
RMLX_SWEEP_MODEL="$MODEL" \
RMLX_SWEEP_LEVELS="$LEVELS" \
RMLX_SWEEP_PTOKS="$PROMPT_TOKENS" \
RMLX_SWEEP_ARCH_KEY="$ARCH_KEY" \
RMLX_SWEEP_KV_QUANT="$KV_QUANT" \
RMLX_SWEEP_REPO="$REPO_ROOT" \
	python3 - <<'PY'
import datetime
import glob
import json
import os
import statistics as st
import subprocess
import sys
from collections import defaultdict

out = os.environ["RMLX_SWEEP_OUT"]
repo = os.environ["RMLX_SWEEP_REPO"]
levels = [int(x) for x in os.environ["RMLX_SWEEP_LEVELS"].split()]
ptoks = [int(x) for x in os.environ["RMLX_SWEEP_PTOKS"].split()]
baseline = levels[0]
# `rmlx bench --json` reports no wall-clock, so the record's timestamp is the
# one this process observes, not one read back out of a slot.
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

sid_out = subprocess.run(
    ["python3", f"{repo}/scripts/lib/snapshot_identity.py", os.environ["RMLX_SWEEP_MODEL"]],
    capture_output=True, text=True, check=True).stdout
sid = dict(line.split("=", 1) for line in sid_out.strip().splitlines())
identity = json.loads(os.environ["RMLX_IDENTITY_JSON"])

METRICS = {
    "ttft_warm_ms": ("ttft_ms", "lower"),
    "prefill_tps": ("prefill_tps", "higher"),
    "decode_tps_warm": ("decode_tps", "higher"),
    "kv_cache_bytes": ("kv_cache_bytes", None),
}

records = []
divergent = []
for ptok in ptoks:
    # runs[(level, row)][metric] = the run values of that slot
    runs = defaultdict(lambda: defaultdict(list))
    digests = defaultdict(set)
    meta = {}
    for path in sorted(glob.glob(f"{out}/raw/p{ptok}_c*_r*_s*.json")):
        stem = os.path.basename(path)[:-5].split("_")
        level, row = int(stem[1][1:]), int(stem[2][1:])
        run = json.load(open(path))
        meta = run
        digests[level].add(run["token_digest"])
        for detail in run["runs_detail"]:
            for name, (field, _) in METRICS.items():
                value = detail.get(field)
                if value:
                    runs[(level, row)][name].append(float(value))
    if not meta:
        continue

    all_digests = set().union(*digests.values())
    if len(all_digests) > 1:
        divergent.append((ptok, {lvl: sorted(d) for lvl, d in sorted(digests.items())}))

    rows = sorted({row for (_lvl, row) in runs})
    print(f"\n### prompt_tokens={meta['prompt_tokens']}  arch={meta['arch']}  "
          f"rows={len(rows)}  token_digest={sorted(all_digests)}")
    for name, (_field, better) in METRICS.items():
        if better is None:
            continue
        pooled = {lvl: [v for row in rows for v in runs.get((lvl, row), {}).get(name, [])]
                  for lvl in levels}
        pooled = {lvl: vs for lvl, vs in pooled.items() if vs}
        if baseline not in pooled:
            continue
        print(f"  -- {name} ({better} is better) --")
        base_lo, base_hi = min(pooled[baseline]), max(pooled[baseline])
        for lvl in levels:
            if lvl not in pooled:
                continue
            vs = pooled[lvl]
            line = (f"    level {lvl:>6}: median {st.median(vs):10.2f} "
                    f"[{min(vs):9.2f}, {max(vs):9.2f}] n={len(vs)}")
            if lvl != baseline:
                paired = [st.median(runs[(lvl, row)][name]) / st.median(runs[(baseline, row)][name])
                          for row in rows
                          if runs.get((lvl, row), {}).get(name)
                          and runs.get((baseline, row), {}).get(name)]
                if paired:
                    disjoint = (max(vs) < base_lo) if better == "lower" else (min(vs) > base_hi)
                    agree = (all(r < 1.0 for r in paired) if better == "lower"
                             else all(r > 1.0 for r in paired))
                    line += (f"  paired vs {baseline}: {st.median(paired):.4f} "
                             f"[{min(paired):.4f}, {max(paired):.4f}] "
                             f"ranges-disjoint={disjoint} every-row-agrees={agree}")
            print(line)

    body = json.load(open(f"{repo}/prompts/longctx_{ptok // 1024}k.json"))
    for lvl in levels:
        pooled = defaultdict(list)
        for row in rows:
            for name, values in runs.get((lvl, row), {}).items():
                pooled[name].extend(values)
        if not pooled:
            continue
        measurements = [
            {"name": name,
             "value": st.median(values),
             "stddev": st.stdev(values) if len(values) > 1 else None}
            for name, values in sorted(pooled.items())
        ]
        records.append({
            **identity,
            "schema_version": 1,
            "model_namespace": sid["model_namespace"],
            "model": sid["model"],
            "weight_quant": sid["weight_quant"],
            "kv_quant": os.environ["RMLX_SWEEP_KV_QUANT"],
            "ctx_max": int(open(f"{out}/raw/p{ptok}.max_ctx").read().strip()),
            "decode_config": f"prefill_chunk={lvl}",
            "prompt": {"name": meta["prompt"], "body": body,
                       "notes": "prefill-chunk Latin-square sweep"},
            "ts_utc": now,
            "prompt_tokens": meta["prompt_tokens"],
            "max_tokens": meta["gen_tokens"],
            "temperature": 0.0,
            "seed": 0,
            "n_warmups": len(rows),
            "n_measure": len(pooled["ttft_warm_ms"]),
            "notes": (f"prefill_chunk sweep; {os.environ['RMLX_SWEEP_ARCH_KEY']} level {lvl}; "
                      f"cyclic Latin square, {len(rows)} rows x {len(levels)} levels, "
                      f"rmlx bench per slot; token_digest={sorted(all_digests)[0]}"),
            "metrics": measurements,
        })

if divergent:
    print("\nCORRECTNESS FAILURE: chunking changed the generated tokens.", file=sys.stderr)
    for ptok, per_level in divergent:
        print(f"  prompt_tokens={ptok}: {per_level}", file=sys.stderr)
    print("  Nothing was recorded.", file=sys.stderr)
    sys.exit(1)

paths = []
os.makedirs(f"{out}/records", exist_ok=True)
for i, record in enumerate(records):
    path = f"{out}/records/rec_{i:03d}.json"
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    paths.append(path)
with open(f"{out}/records/manifest.txt", "w") as fh:
    fh.write("\n".join(paths) + "\n")
print(f"\nbuilt {len(paths)} records under {out}/records")
PY

if $RECORD; then
	echo ""
	echo "==> recording cells through the §8.5 ingest path"
	while read -r record; do
		[[ -n "$record" ]] || continue
		if [[ -n "$DB_PATH" ]]; then
			"$BINARY" metrics record --file "$record" --db "$DB_PATH"
		else
			"$BINARY" metrics record --file "$record"
		fi
	done <"$OUT_DIR/records/manifest.txt"
else
	echo ""
	echo "Not recorded. Pass --record to ingest these cells into runs.db."
fi

echo ""
echo "sweep output: $OUT_DIR"
