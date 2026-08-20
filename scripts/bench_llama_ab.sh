#!/usr/bin/env bash
# bench_llama_ab.sh — ABBA-interleaved A/B comparison of two `llama-server` arms.
#
# WHY THIS EXISTS
#
# `scripts/perf_ab.sh` is the ABBA harness for two `rmlx baseline` arms: it
# parses that binary's stdout, forbids its `--metrics` flag and compares its
# token-id line. None of that exists for `llama-server`, whose measurement
# surface is an HTTP `/completion` response. The measurement *discipline* is
# the same and is not duplicated here — host quiescence comes from the shared
# `scripts/lib/cpu_snapshot.sh` + `scripts/lib/busiest_between.awk` pair, on
# the same cumulative-CPU-seconds criterion and the same default threshold.
#
# What is genuinely different, and why this cannot be a flag on perf_ab.sh:
#
#   * an arm is a (binary, server flags) pair launched as a daemon, probed for
#     readiness and shut down again, not a one-shot CLI invocation;
#   * decode-only throughput is read from the server's own `timings` block,
#     not from a wall clock the harness holds;
#   * resident cost is two numbers, not one — the KV buffer size the backend
#     reports at load, and sampled peak process RSS — because a codec arm that
#     is faster with the same footprint has not demonstrated anything.
#
# DESIGN
#
#   * Slots alternate ABBA / BAAB / ABBA …, so monotone drift over the run
#     contributes equally to both arms. `--pairs N` gives N slots per arm.
#   * Every slot is a fresh server process. Two servers alive at once would
#     make each arm's residency the other's memory pressure.
#   * Foreign CPU use is sampled across each measured window. Anything at or
#     above `--busy-pct` of one core taints the whole comparison rather than
#     tilting one arm; a window that could not be sampled taints as well.
#   * The arms must be distinguishable: identical binary digest AND identical
#     flags is refused. "A vs B" where both are the same build is the failure
#     mode that looks most like a result.
#   * Spread is reported, never hidden. When the two arms' [min,max] ranges
#     overlap the verdict is INCONCLUSIVE, whatever the medians say.
#
# NEVER WRITES TO runs.db. An A/B run is an experiment. Feed an accepted cell
# to `rmlx metrics record --file` yourself, from the emitted JSON.
#
# Exit codes:
#   0   — ran cleanly; verdict on stdout
#   125 — not usable: busy host, indistinguishable arms, missing binary or
#         model, a slot that produced no parseable timing, or TAINTED.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RMLX_HOME="${RMLX_HOME:-${REPO_ROOT}/.rmlx}"
AWK_BUSIEST="${REPO_ROOT}/scripts/lib/busiest_between.awk"

BIN_A=""; BIN_B=""; ARGS_A=""; ARGS_B=""
ENV_A=""; ENV_B=""
LABEL_A="A"; LABEL_B="B"
MODEL=""; PROMPT_FILE=""
N_CTX=8192; N_PREDICT=128; PAIRS=2
PORT=8199; BUSY_PCT=25; ALLOW_BUSY_HOST=false
READY_TIMEOUT=600
OUT_DIR="${RMLX_HOME}/bench/llama_ab"

usage() {
	cat <<'USAGE'
bench_llama_ab.sh --bin-a P --bin-b P --model GGUF --prompt-file F [options]

  --bin-a/--bin-b PATH     llama-server binary per arm (required)
  --args-a/--args-b STR    extra server flags per arm, e.g. "-ctk turbo3 -ctv turbo3"
  --env-a/--env-b STR      extra environment per arm, e.g. "TURBO_FLASH=1". Some
                           backends gate a kernel on an env knob rather than a
                           flag; without this such an arm is unreachable.
  --label-a/--label-b STR  arm label used in the report (default A / B)
  --model PATH             GGUF weights, identical for both arms (required)
  --prompt-file PATH       UTF-8 prompt body sent verbatim to /completion (required)
  --n-ctx N                server context, identical for both arms (default 8192)
  --n-predict N            decode budget per measured request (default 128)
  --pairs N                slots per arm; total slots = 2N (default 2, use >=4)
  --port N                 loopback port for the server (default 8199)
  --busy-pct N             foreign-CPU taint threshold, % of one core (default 25)
  --allow-busy-host        start anyway on a busy host; output is marked TAINTED
  --ready-timeout S        seconds to wait for /health per slot (default 600)
  --out-dir PATH           JSON result directory (default $RMLX_HOME/bench/llama_ab)
USAGE
}

need_value() { [ "$2" -ge 2 ] || { echo "$1 needs a value" >&2; exit 125; }; }

while [ $# -gt 0 ]; do
	case "$1" in
	--bin-a) need_value "$1" $#; BIN_A="$2"; shift 2 ;;
	--bin-b) need_value "$1" $#; BIN_B="$2"; shift 2 ;;
	--args-a) need_value "$1" $#; ARGS_A="$2"; shift 2 ;;
	--args-b) need_value "$1" $#; ARGS_B="$2"; shift 2 ;;
	--env-a) need_value "$1" $#; ENV_A="$2"; shift 2 ;;
	--env-b) need_value "$1" $#; ENV_B="$2"; shift 2 ;;
	--label-a) need_value "$1" $#; LABEL_A="$2"; shift 2 ;;
	--label-b) need_value "$1" $#; LABEL_B="$2"; shift 2 ;;
	--model) need_value "$1" $#; MODEL="$2"; shift 2 ;;
	--prompt-file) need_value "$1" $#; PROMPT_FILE="$2"; shift 2 ;;
	--n-ctx) need_value "$1" $#; N_CTX="$2"; shift 2 ;;
	--n-predict) need_value "$1" $#; N_PREDICT="$2"; shift 2 ;;
	--pairs) need_value "$1" $#; PAIRS="$2"; shift 2 ;;
	--port) need_value "$1" $#; PORT="$2"; shift 2 ;;
	--busy-pct) need_value "$1" $#; BUSY_PCT="$2"; shift 2 ;;
	--allow-busy-host) ALLOW_BUSY_HOST=true; shift ;;
	--ready-timeout) need_value "$1" $#; READY_TIMEOUT="$2"; shift 2 ;;
	--out-dir) need_value "$1" $#; OUT_DIR="$2"; shift 2 ;;
	-h | --help) usage; exit 0 ;;
	*) echo "unknown argument: $1" >&2; usage >&2; exit 125 ;;
	esac
done

[ -n "$BIN_A" ] || { echo "missing required --bin-a" >&2; exit 125; }
[ -n "$BIN_B" ] || { echo "missing required --bin-b" >&2; exit 125; }
[ -n "$MODEL" ] || { echo "missing required --model" >&2; exit 125; }
[ -n "$PROMPT_FILE" ] || { echo "missing required --prompt-file" >&2; exit 125; }
for f in "$BIN_A" "$BIN_B" "$MODEL" "$PROMPT_FILE"; do
	[ -e "$f" ] || { echo "not found: $f" >&2; exit 125; }
done
# `--pairs 1` gives one slot per arm, so both "ranges" are single points and the
# overlap test below cannot fire unless the two floats are exactly equal: the
# verdict would read SEPARATED at the least evidence the harness can collect.
# Two pairs is the minimum at which a range exists on both sides.
if [ "$PAIRS" -lt 2 ]; then
	echo "--pairs must be >= 2: at 1 slot per arm each range is a single point," >&2
	echo "  so the overlap test cannot fire and every run reads SEPARATED." >&2
	exit 125
fi

# ---- arm distinguishability --------------------------------------------------
# Two arms that are the same build with the same flags produce a difference of
# pure noise and a report that reads exactly like a real one. A stash / build /
# unstash sequence that left the old binary in place is how that happens.
SHA_A="$(shasum -a 256 "$BIN_A" | awk '{print $1}')"
SHA_B="$(shasum -a 256 "$BIN_B" | awk '{print $1}')"
if [ "$SHA_A" = "$SHA_B" ] && [ "$ARGS_A" = "$ARGS_B" ] && [ "$ENV_A" = "$ENV_B" ]; then
	echo "arms are indistinguishable: same binary digest, same flags, same env." >&2
	echo "  digest: $SHA_A" >&2
	exit 125
fi

# ---- arm args must not move what the harness pins -----------------------------
# `$extra` is appended after the fixed flags, so a later occurrence wins in
# llama-server's parser. `--args-b "-c 2048"` would run arm B at another context
# while the report, the result JSON's `n_ctx` and every ingested row still claim
# `$N_CTX` -- two arms silently become a different experiment, and a wrong
# `ctx_max` lands in an append-only table. `perf_ab.sh` refuses `--metrics` in
# arm arguments for exactly this reason.
PINNED_FLAGS="--model -m --port --host -c --ctx-size -np --parallel -fa --flash-attn -ngl --n-gpu-layers -t --threads"
refuse_pinned() { # <label> <args>
	local label="$1" tok
	for tok in $2; do
		case " $PINNED_FLAGS " in
		*" $tok "*)
			echo "$label sets '$tok', which this harness pins for both arms." >&2
			echo "  Arm args are appended last, so it would silently win and the" >&2
			echo "  reported n_ctx / model / dispatch would describe a run that did" >&2
			echo "  not happen. Use the harness flag instead." >&2
			exit 125
			;;
		esac
	done
}
refuse_pinned --args-a "$ARGS_A"
refuse_pinned --args-b "$ARGS_B"

CPU_SNAPSHOT_SKIP="$(basename "$BIN_A") $(basename "$BIN_B") llama-server"
export CPU_SNAPSHOT_SKIP
# shellcheck source=scripts/lib/cpu_snapshot.sh
. "${REPO_ROOT}/scripts/lib/cpu_snapshot.sh"

# ---- finding: a stranger already on the port ---------------------------------
# `llama-server` binds only after it has loaded the model, so a foreign or
# leftover server on $PORT answers /health immediately while our process is
# still loading. The measured request would go to the stranger, our process
# would die on the bind, and BOTH arms would silently measure the same third
# binary. Refuse the port up front rather than discovering it in the numbers.
if curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
	echo "port ${PORT} already answers /health -- something is listening there." >&2
	echo "  llama-server binds after loading, so this run would measure that" >&2
	echo "  process instead of its own, on both arms. Free the port or pass" >&2
	echo "  --port." >&2
	exit 125
fi

TMP="$(mktemp -d)"
cleanup() {
	pkill -f "llama-server .*--port ${PORT}" 2>/dev/null || true
	rm -rf "$TMP"
}
trap cleanup EXIT

host_busiest() { # <before> <after> <window_s> -> "<state> <pct> <comm>"
	local raw
	if [ -e "$1.failed" ] || [ -e "$2.failed" ]; then echo "unmeasured - -"; return; fi
	raw="$(awk -v window="$3" -f "$AWK_BUSIEST" "$1" "$2")"
	case "${raw%% *}" in
	unmeasured) echo "unmeasured - -" ;;
	idle) echo "quiet 0.0 -" ;;
	*)
		if awk -v p="$(echo "$raw" | awk '{print $2}')" -v t="$BUSY_PCT" 'BEGIN { exit !(p >= t) }'; then
			echo "busy ${raw#* }"
		else
			echo "quiet ${raw#* }"
		fi
		;;
	esac
}

snapshot_ok() { cpu_snapshot "$1" && return 0; : >"$1.failed"; return 1; }

# ---- entry gate --------------------------------------------------------------
snapshot_ok "$TMP/entry_a" || true
sleep 5
snapshot_ok "$TMP/entry_b" || true
ENTRY="$(host_busiest "$TMP/entry_a" "$TMP/entry_b" 5)"
ENTRY_STATE="${ENTRY%% *}"
if [ "$ENTRY_STATE" != "quiet" ]; then
	echo "host is not quiescent: ${ENTRY}" >&2
	if ! $ALLOW_BUSY_HOST; then
		echo "  Quiesce the host, or pass --allow-busy-host to see the numbers anyway." >&2
		exit 125
	fi
	echo "  --allow-busy-host: every number below is suspect." >&2
fi

TAINT=""
note_taint() { TAINT="${TAINT}$1; "; }
[ "$ENTRY_STATE" != "quiet" ] && note_taint "entry gate: $ENTRY"

# ---- one slot ----------------------------------------------------------------
# Emits "<decode_tps> <prompt_n> <predicted_n> <prompt_tps> <kv_mib> <peak_rss_mb>".
run_slot() { # <bin> <args> <env> <slotdir>
	local bin="$1" extra="$2" armenv="$3" dir="$4"
	mkdir -p "$dir"
	# shellcheck disable=SC2086  # armenv and extra are deliberate word lists
	env $armenv "$bin" --model "$MODEL" --port "$PORT" --host 127.0.0.1 \
		-c "$N_CTX" -np 1 -fa on --no-webui \
		$extra >"$dir/server.log" 2>&1 &
	local pid=$!

	local deadline=$(( $(date +%s) + READY_TIMEOUT ))
	local ready=0
	while [ "$(date +%s)" -lt "$deadline" ]; do
		sleep 1
		kill -0 "$pid" 2>/dev/null || break
		if curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
	done
	if [ "$ready" -ne 1 ]; then
		kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
		echo "SLOT_FAIL server-not-ready" >&2
		return 1
	fi

	# Readiness alone does not say WHOSE server answered. Confirm the process
	# that responds is serving the model this slot launched.
	local served
	served="$(curl -fsS -m 5 "http://127.0.0.1:${PORT}/props" 2>/dev/null |
		python3 -c 'import json,sys; print(json.load(sys.stdin).get("model_path",""))' 2>/dev/null || true)"
	if [ "$served" != "$MODEL" ]; then
		kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
		echo "SLOT_FAIL wrong-server: /props reports model '$served', expected '$MODEL'" >&2
		return 1
	fi

	# Warmup: a short generation over the same prompt, discarded. It pays the
	# first-dispatch pipeline compile and the prompt-cache fill so the measured
	# request times steady-state decode, not Metal's first-use costs.
	post_completion "$dir/warmup.json" 8 >/dev/null 2>&1 || true

	local rss_peak=0 rss
	snapshot_ok "$dir/cpu_before" || true
	local t0 t1
	t0="$(python3 -c 'import time; print(time.time())')"
	post_completion "$dir/measure.json" "$N_PREDICT" >/dev/null 2>&1 &
	local reqpid=$!
	while kill -0 "$reqpid" 2>/dev/null; do
		rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')"
		[ -n "$rss" ] && [ "$rss" -gt "$rss_peak" ] && rss_peak="$rss"
		sleep 0.5
	done
	wait "$reqpid" 2>/dev/null || true
	t1="$(python3 -c 'import time; print(time.time())')"
	snapshot_ok "$dir/cpu_after" || true
	local window; window="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.3f", b - a }')"
	host_busiest "$dir/cpu_before" "$dir/cpu_after" "$window" >"$dir/window"

	kill "$pid" 2>/dev/null || true
	wait "$pid" 2>/dev/null || true

	# The backend's own KV figure. `llama.cpp` prints one line per KV buffer;
	# summing them is the whole cache, and a codec arm that does not move this
	# number has not changed what it stores.
	# `kv_mib` is half the verdict this harness exists to produce, so a log this
	# parser cannot read must FAIL the slot. Emitting `0.00` on no match -- a
	# build that logs GiB, or `2048.00MiB` without the space -- would put a
	# fabricated zero in the report, the result JSON and `kv_cache_bytes`, where
	# the §4.1 plausible-value bounds admit 0 as a real gauge reading.
	local kv_mib kv_hits
	kv_hits="$(grep -c "KV buffer size" "$dir/server.log" || true)"
	kv_mib="$(awk '/KV buffer size/ { for (i = 1; i <= NF; i++) if ($i == "MiB") s += $(i-1); n++ } END { if (n == 0 || s <= 0) exit 1; printf "%.2f", s }' "$dir/server.log")" || {
		echo "SLOT_FAIL kv-buffer-unparsed: ${kv_hits} 'KV buffer size' line(s), no MiB total" >&2
		return 1
	}

	# Same rule for resident memory: `ps` returning nothing for every sample
	# leaves rss_peak at its 0 initialiser, which is not a measurement.
	if [ "$rss_peak" -le 0 ]; then
		echo "SLOT_FAIL rss-unsampled: no ps reading for pid $pid during the measured window" >&2
		return 1
	fi

	# The generated text goes into the report too. Cross-build output is not
	# expected to be bit-identical, but a codec that has silently corrupted the
	# cache produces visible garbage, and a throughput table alone hides it.
	python3 - "$dir/measure.json" "$kv_mib" "$rss_peak" "$dir/first64" "$N_PREDICT" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    t = d["timings"]
    want = int(sys.argv[5])
    # An early EOS or a truncated generation still parses, and its
    # `predicted_per_second` still enters the median and moves it. A decode
    # budget that was not spent is not the cell that was asked for.
    if t["predicted_n"] != want:
        raise ValueError("decode budget not spent: predicted_n=%s, wanted %d"
                         % (t["predicted_n"], want))
    # Zero tokens per second is `tokens / seconds` with a zero numerator --
    # nothing was measured. It is outside the §4.1 rate window for the same
    # reason and must not reach the median.
    if not t["predicted_per_second"] > 0:
        raise ValueError("predicted_per_second=%s is not a measurement"
                         % t["predicted_per_second"])
    text = (d.get("content") or "")[:64]
    # An empty capture cannot be told apart from "the model emitted only
    # characters the report strips", and a coherence claim must not rest on
    # a field that silently reads empty.
    if not text.strip():
        raise ValueError("empty generation capture")
    open(sys.argv[4], "w").write(text)
    print("%.4f %d %d %.4f %s %.1f" % (
        t["predicted_per_second"], t["prompt_n"], t["predicted_n"],
        t["prompt_per_second"], sys.argv[2], float(sys.argv[3]) / 1024.0))
except Exception as exc:  # noqa: BLE001 - any parse failure fails the slot
    print("PARSE_FAIL %s" % exc, file=sys.stderr)
    sys.exit(1)
PY
}

post_completion() { # <outfile> <n_predict>
	python3 - "$1" "$2" "$PROMPT_FILE" "$PORT" <<'PY'
import json, sys, urllib.request
out, npred, pfile, port = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
body = json.dumps({
    "prompt": open(pfile, encoding="utf-8").read(),
    "n_predict": npred,
    "temperature": 0.0,
    "top_k": 1,
    "seed": 0,
    "cache_prompt": False,
    "stream": False,
}).encode()
req = urllib.request.Request("http://127.0.0.1:%s/completion" % port, data=body,
                             headers={"content-type": "application/json"})
with urllib.request.urlopen(req, timeout=3600) as r:
    open(out, "wb").write(r.read())
PY
}

# ---- ABBA --------------------------------------------------------------------
ORDER=""
for i in $(seq 1 "$PAIRS"); do
	if [ $((i % 2)) -eq 1 ]; then ORDER="$ORDER A B"; else ORDER="$ORDER B A"; fi
done

TPS_A=""; TPS_B=""; ROWS=""
TOTAL=$((PAIRS * 2))
SLOT=0
for arm in $ORDER; do
	SLOT=$((SLOT + 1))
	if [ "$arm" = "A" ]; then bin="$BIN_A"; extra="$ARGS_A"; armenv="$ENV_A"; label="$LABEL_A"; else bin="$BIN_B"; extra="$ARGS_B"; armenv="$ENV_B"; label="$LABEL_B"; fi
	dir="$TMP/slot$SLOT"
	echo "[slot $SLOT/$TOTAL] $label" >&2
	if ! out="$(run_slot "$bin" "$extra" "$armenv" "$dir")"; then
		echo "slot $SLOT ($label) produced no measurement" >&2
		exit 125
	fi
	set -- $out
	tps="$1"; pn="$2"; dn="$3"; ptps="$4"; kv="$5"; rss="$6"
	win="$(cat "$dir/window")"
	# The python above refuses to write this file unless the capture was
	# non-empty, so a missing or blank one here means the contract broke.
	first64="$(tr -d '|\n' <"$dir/first64" 2>/dev/null || true)"
	if [ -z "$first64" ]; then
		echo "slot $SLOT ($label): generation capture is empty" >&2
		exit 125
	fi
	[ "${win%% *}" = "quiet" ] || note_taint "slot $SLOT ($label): $win"
	echo "  decode ${tps} tok/s | prompt_n ${pn} | predicted ${dn} | kv ${kv} MiB | peak_rss ${rss} MB | host ${win}" >&2
	echo "  out: ${first64}" >&2
	ROWS="${ROWS}${arm}|${label}|${tps}|${pn}|${dn}|${ptps}|${kv}|${rss}|${win}|${first64}
"
	if [ "$arm" = "A" ]; then TPS_A="$TPS_A $tps"; else TPS_B="$TPS_B $tps"; fi
done

stats() { # <values...> -> "min median max mean"
	python3 -c '
import statistics, sys
v = sorted(float(x) for x in sys.argv[1:])
print("%.3f %.3f %.3f %.3f" % (v[0], statistics.median(v), v[-1], statistics.fmean(v)))
' $1
}
# shellcheck disable=SC2086  # word splitting is the intent
SA="$(stats "$TPS_A")"
# shellcheck disable=SC2086
SB="$(stats "$TPS_B")"

VERDICT="$(python3 -c '
import sys
amin, amed, amax, _ = (float(x) for x in sys.argv[1].split())
bmin, bmed, bmax, _ = (float(x) for x in sys.argv[2].split())
n = int(sys.argv[3])
ratio = bmed / amed
if amax >= bmin and bmax >= amin:
    print("INCONCLUSIVE ranges-overlap %.4f" % ratio)
elif n < 3:
    # Disjoint ranges built from two points each are disjoint on very little.
    # Say so rather than letting the strongest word in the vocabulary rest on
    # the weakest evidence the harness will accept.
    print("SEPARATED-WEAK n=%d-per-arm %.4f" % (n, ratio))
else:
    print("SEPARATED %.4f" % ratio)
' "$SA" "$SB" "$PAIRS")"

export ROWS
mkdir -p "$OUT_DIR"
# Two forms on purpose: the file name wants a compact stamp, and `ts_utc` in a
# §8.5 RunRecord must be ISO-8601 UTC. Emitting only the compact one put a
# string the ingest contract does not accept into every result file.
TS_FILE="$(date -u +%Y%m%dT%H%M%SZ)"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RESULT="${OUT_DIR}/${TS_FILE}.json"
python3 - "$RESULT" "$LABEL_A" "$LABEL_B" "$SHA_A" "$SHA_B" "$ARGS_A" "$ARGS_B" \
	"$MODEL" "$PROMPT_FILE" "$N_CTX" "$N_PREDICT" "$SA" "$SB" "$VERDICT" "${TAINT%; }" "$TS" \
	"$ENV_A" "$ENV_B" <<'PY'
import json, os, sys
(out, la, lb, sa, sb, aa, ab, model, prompt, nctx, npred, statsa, statsb,
 verdict, taint, ts, ea, eb) = sys.argv[1:19]
rows = []
for line in os.environ.get("ROWS", "").splitlines():
    if not line.strip():
        continue
    f = line.split("|")
    rows.append({"arm": f[0], "label": f[1], "decode_tps": float(f[2]),
                 "prompt_n": int(f[3]), "predicted_n": int(f[4]),
                 "prompt_tps": float(f[5]), "kv_mib": float(f[6]),
                 "peak_rss_mb": float(f[7]), "host_window": f[8],
                 "output_first_64": f[9] if len(f) > 9 else ""})
def st(s):
    v = [float(x) for x in s.split()]
    return {"min": v[0], "median": v[1], "max": v[2], "mean": v[3]}
json.dump({
    "ts_utc": ts, "model": model, "prompt_file": prompt,
    "n_ctx": int(nctx), "n_predict": int(npred),
    "arms": {"A": {"label": la, "sha256": sa, "args": aa, "env": ea, "decode_tps": st(statsa)},
             "B": {"label": lb, "sha256": sb, "args": ab, "env": eb, "decode_tps": st(statsb)}},
    "slots": rows, "verdict": verdict, "tainted": taint or None,
}, open(out, "w"), indent=2)
PY

echo
echo "model      $MODEL"
echo "n_ctx      $N_CTX   n_predict $N_PREDICT   pairs $PAIRS"
echo "arm A      $LABEL_A  [${SHA_A:0:12}]  ${ARGS_A:-<no extra flags>}  ${ENV_A:+env: $ENV_A}"
echo "arm B      $LABEL_B  [${SHA_B:0:12}]  ${ARGS_B:-<no extra flags>}  ${ENV_B:+env: $ENV_B}"
printf 'decode A   min %s  median %s  max %s  mean %s\n' $SA
printf 'decode B   min %s  median %s  max %s  mean %s\n' $SB
echo "verdict    $VERDICT   (B/A on medians)"
echo "result     $RESULT"
if [ -n "$TAINT" ]; then
	echo "TAINTED    ${TAINT%; }"
	exit 125
fi
