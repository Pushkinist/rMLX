#!/usr/bin/env bash
# bench_llama_ab_selftest.sh — mutation check for `scripts/bench_llama_ab.sh`.
#
# Every guard in that harness exists because a specific wrong answer is
# reachable without it. A guard nobody has watched fail is a comment. This file
# drives each one to its failure, against a stub `llama-server` — no GPU, no
# GGUF, no metrics database — and asserts the exit code AND the reason text, so
# a guard that starts failing for a different reason does not read as passing.
#
# Its sibling `perf_ab_selftest.sh` does the same for `perf_ab.sh`.
#
# Exit 0 = every case produced its expected exit code and message.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AB="$ROOT/scripts/bench_llama_ab.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_llama_ab_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

MODEL="$WORK/model.gguf"
PROMPT="$WORK/prompt.txt"
printf 'weights\n' >"$MODEL"
printf 'the quick brown fox\n' >"$PROMPT"

# A free port, so a real service on the default never makes a case pass or fail
# for the wrong reason.
PORT="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"

PASS=0
FAIL=0

# make_stub <name> <tps-csv> <kv-line> <content> [predicted_n] [prompt_n] [props_model]
#
# Stands in for `llama-server`: serves /health, /props and /completion over a
# real socket, prints the KV line the harness parses out of the log, and cycles
# <tps-csv> across successive processes via a shared counter file so an arm
# shows a spread rather than a constant.
make_stub() {
	local name="$1" tps="$2" kvline="$3" content="$4"
	local predicted="${5:-4}" promptn="${6:-9}" props="${7:-}"
	local path="$WORK/$name"
	cat >"$path" <<STUB
#!/usr/bin/env bash
# stub llama-server: $name
set -eu
port=""; model=""
while [ \$# -gt 0 ]; do
  case "\$1" in
    --port) port="\$2"; shift 2 ;;
    --model) model="\$2"; shift 2 ;;
    *) shift ;;
  esac
done
# The KV line the harness greps out of the server log.
printf '%s\n' "$kvline"
n=0
[ -f "$WORK/${name}.n" ] && n="\$(cat "$WORK/${name}.n")"
echo \$((n + 1)) >"$WORK/${name}.n"
props="$props"
[ -n "\$props" ] || props="\$model"
export STUB_TPS_CSV="$tps" STUB_N="\$n" STUB_MODEL="\$props"
export STUB_CONTENT="$content" STUB_PRED="$predicted" STUB_PROMPTN="$promptn"
exec python3 "$WORK/serve.py" "\$port"
STUB
	chmod +x "$path"
	echo "$path"
}

cat >"$WORK/serve.py" <<'SERVE'
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

TPS = [float(x) for x in os.environ["STUB_TPS_CSV"].split(",")]
N = int(os.environ["STUB_N"])
TPSV = TPS[N % len(TPS)]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path.startswith("/health"):
            self._send({"status": "ok"})
        elif self.path.startswith("/props"):
            self._send({"model_path": os.environ["STUB_MODEL"]})
        else:
            self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        self._send({
            "content": os.environ["STUB_CONTENT"],
            "timings": {
                "predicted_per_second": TPSV,
                "prompt_per_second": 100.0,
                "prompt_n": int(os.environ["STUB_PROMPTN"]),
                "predicted_n": int(os.environ["STUB_PRED"]),
            },
        })

HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
SERVE

KV_OK='llama_kv_cache:       MTL0 KV buffer size =   512.00 MiB'

# WHY EVERY CASE DECLARES ITS ARMS SYNTHETIC
#
# `bench_llama_ab.sh` refuses to measure on a host it does not have to itself,
# and taints a comparison that ran alongside a foreign process. That is right
# for a measurement and wrong for this file, whose arms are stub HTTP servers
# emitting canned timings and whose every expectation is a fact about the
# harness's logic. Inheriting a runtime precondition for a logic property makes
# the outcome a function of the machine: a completed comparison exits 0 on a
# quiet host and 125 on a busy one, so the exit code these cases assert used to
# be resolved from the run's own output rather than stated in advance -- an
# expectation that agrees with whatever happened cannot catch anything.
#
# With `--synthetic-arms` the machine is not consulted, so the code is a literal
# again. The cases whose subject IS the taint path say REALHOST and supply the
# machine as a `ps` shim on PATH; that is enforced below, not trusted. (This
# harness has no exclusivity gate, so `ps` is the whole host surface here --
# unlike `perf_ab_selftest.sh`, which must shim `pgrep` as well.)
#
# FORCE_UNMEASURED_ROWS drives the harness down its TAINTED path on demand and
# without an interferer. `cpu_snapshot` refuses a process table shorter than
# CPU_SNAPSHOT_MIN_ROWS, a refused snapshot classifies its window `unmeasured`,
# and an unmeasured window taints — so an absurd floor taints every run,
# deterministically, whatever the host is doing.
SYNTHETIC_CASES=0
REAL_HOST_CASES=0
HOST_READING_CASES=0
RUN_SEQ=0

run_ab() { # [REALHOST:<shim-dir>] <extra args...> -> stdout+stderr in $OUT, code in $CODE
	local host_shim="" boundary=(--synthetic-arms)
	if [ "${1#REALHOST:}" != "$1" ]; then
		host_shim="${1#REALHOST:}"
		shift
		boundary=()
		if [ ! -x "$host_shim/ps" ]; then
			HOST_READING_CASES=$((HOST_READING_CASES + 1))
			OUT="selftest bug: REALHOST without a ps shim at $host_shim"
			CODE=-1
			return
		fi
		REAL_HOST_CASES=$((REAL_HOST_CASES + 1))
	else
		SYNTHETIC_CASES=$((SYNTHETIC_CASES + 1))
	fi
	RUN_SEQ=$((RUN_SEQ + 1))
	OUT="$(PATH="${host_shim:+$host_shim:}$PATH" \
		RMLX_HOME="$WORK/home/run$RUN_SEQ" \
		CPU_SNAPSHOT_MIN_ROWS="${FORCE_UNMEASURED_ROWS:-20}" \
		bash "$AB" \
		--model "$MODEL" --prompt-file "$PROMPT" \
		--port "$PORT" --n-predict 4 --n-ctx 64 \
		--out-dir "$WORK/home/run$RUN_SEQ/bench" --ready-timeout 25 \
		${boundary[@]+"${boundary[@]}"} \
		"$@" 2>&1)"
	CODE=$?
}

# A quiet machine, 40 rows of nothing: `cpu_snapshot` refuses a process table
# under 20 rows, so a shim has to look like a real one before it can look like
# anything else. Used by the REALHOST cases, whose subject is the taint path and
# not this host.
# `ps` answers two different questions in this harness and only one of them is
# about the host: `ps -Aww -o pid=,time=,comm=` samples foreign CPU use, while
# `ps -o rss= -p <pid>` is the harness measuring its own arm. A shim that
# answered both would replace a measurement, so the per-pid form is handed to
# the real ps.
write_ps_shim() { # write_ps_shim <dir> <body-line...>
	local dir="$1"
	shift
	mkdir -p "$dir"
	{
		echo '#!/usr/bin/env bash'
		echo 'case " $* " in'
		echo '*" -p "*) exec /bin/ps "$@" ;;'
		echo 'esac'
		printf '%s\n' "$@"
	} >"$dir/ps"
	chmod +x "$dir/ps"
}
IDLE_ROWS='for i in $(seq 1 40); do printf "%6d %12s %s\n" $((5000 + i)) "0:00.10" "/usr/sbin/idle$i"; done'
write_ps_shim "$WORK/quietbin" "$IDLE_ROWS"

# check <label> <expected-code> <expected-substring>
# check_verdict <label> <expected-code> <verdict-text>
#
# Both take the exit code as a literal. For a comparison that ran to completion
# that is 0 with synthetic arms and 125 when the taint path was forced, and
# neither depends on what this machine is doing.
check_verdict() {
	local label="$1" want="$2" needle="$3"
	local verdict_ok=0
	printf '%s' "$OUT" | grep -q "^verdict    $needle" && verdict_ok=1
	if [ "$verdict_ok" -eq 1 ] && [ "$CODE" -eq "$want" ]; then
		echo "  PASS  $label (verdict matched, exit $CODE)"
		PASS=$((PASS + 1))
	else
		if [ "$verdict_ok" -ne 1 ]; then
			echo "  FAIL  $label (wrong verdict; wanted: $needle)"
		else
			echo "  FAIL  $label (verdict correct, but exit $CODE, wanted $want)"
		fi
		printf '%s\n' "$OUT" | sed 's/^/        /' | tail -12
		FAIL=$((FAIL + 1))
	fi
}

check() {
	local label="$1" want="$2" needle="$3"
	if [ "$CODE" -eq "$want" ] && printf '%s' "$OUT" | grep -q -- "$needle"; then
		echo "  PASS  $label"
		PASS=$((PASS + 1))
	else
		echo "  FAIL  $label (exit $CODE, wanted $want; looking for: $needle)"
		printf '%s\n' "$OUT" | sed 's/^/        /' | tail -12
		FAIL=$((FAIL + 1))
	fi
}

echo "bench_llama_ab selftest"

# --- guard: indistinguishable arms -------------------------------------------
STUB_A="$(make_stub a "50,52" "$KV_OK" "hello world")"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_A" --pairs 2 --allow-busy-host
check "identical binary + flags + env is refused" 125 "indistinguishable"

# --- guard: --pairs 1 cannot produce a range ---------------------------------
STUB_B="$(make_stub b "30,31" "$KV_OK" "hello world")"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 1 --allow-busy-host
check "--pairs 1 is refused (overlap test cannot fire)" 125 "pairs must be >= 2"

# --- guard: arm args must not move a harness-pinned flag ---------------------
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 2 --allow-busy-host \
	--args-b "-c 2048"
check "arm args cannot override a pinned flag" 125 "which this harness pins"

# --- guard: a busy port would make both arms measure a stranger --------------
python3 - "$PORT" <<'SQUAT' &
import sys, json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        b = json.dumps({"status": "ok"}).encode()
        self.send_response(200); self.send_header("content-length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
SQUAT
SQUATTER=$!
sleep 1
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 2 --allow-busy-host
check "an occupied port is refused up front" 125 "already answers /health"
kill "$SQUATTER" 2>/dev/null
wait "$SQUATTER" 2>/dev/null

# --- guard: a KV line the parser cannot read must fail, not read 0.00 --------
STUB_NOKV="$(make_stub nokv "30,31" 'llama_kv_cache: KV buffer size = 0.50 GiB' "hello world")"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_NOKV" --pairs 2 --allow-busy-host
check "unparseable KV buffer size fails the slot" 125 "kv-buffer-unparsed"

# --- guard: a decode budget that was not spent -------------------------------
STUB_SHORT="$(make_stub short "30,31" "$KV_OK" "hello world" 2)"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_SHORT" --pairs 2 --allow-busy-host
check "truncated generation fails the slot" 125 "decode budget not spent"

# --- guard: zero decode TPS is not a measurement -----------------------------
STUB_ZERO="$(make_stub zero "0,0" "$KV_OK" "hello world")"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_ZERO" --pairs 2 --allow-busy-host
check "predicted_per_second=0 fails the slot" 125 "is not a measurement"

# --- guard: an empty generation capture --------------------------------------
STUB_EMPTY="$(make_stub empty "30,31" "$KV_OK" "")"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_EMPTY" --pairs 2 --allow-busy-host
check "empty generation capture fails the slot" 125 "empty generation capture"

# --- guard: the server that answered must be ours ----------------------------
STUB_WRONG="$(make_stub wrong "30,31" "$KV_OK" "hello world" 4 9 "/some/other/model.gguf")"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_WRONG" --pairs 2 --allow-busy-host
check "a server serving another model fails the slot" 125 "wrong-server"

# --- behaviour: overlapping ranges must read INCONCLUSIVE --------------------
# Both arms emit the SAME tps sequence from two distinguishable binaries, so
# the ranges coincide exactly. If this reads SEPARATED the overlap test is
# inert and every one of this campaign's verdicts is worthless.
STUB_C="$(make_stub c "50,52" "$KV_OK" "hello world")"
run_ab --bin-a "$STUB_A" --bin-b "$STUB_C" --pairs 2 --allow-busy-host
check_verdict "identical timings read INCONCLUSIVE" 0 "INCONCLUSIVE ranges-overlap"

# --- behaviour: disjoint ranges at n=2 must be flagged weak ------------------
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 2 --allow-busy-host
check_verdict "disjoint ranges at n=2 read SEPARATED-WEAK" 0 "SEPARATED-WEAK n=2-per-arm"

# --- behaviour: disjoint ranges at n=3 read SEPARATED ------------------------
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 3 --allow-busy-host
check_verdict "disjoint ranges at n=3 read SEPARATED" 0 "SEPARATED 0."

# --- the same three verdicts, with the taint path forced --------------------
#
# The point of the three cases above is that the verdict is a property of the
# measurements, not of the machine. These re-run the same comparisons against a
# shimmed host with every interference window forced `unmeasured`, so the
# harness takes its TAINTED branch and exits 125, and require the verdict to be
# character-for-character the same. The 125 is a literal here because the taint
# is manufactured, not observed.
FORCE_UNMEASURED_ROWS=999999
HOST="REALHOST:$WORK/quietbin"

run_ab "$HOST" --bin-a "$STUB_A" --bin-b "$STUB_C" --pairs 2 --allow-busy-host
check_verdict "INCONCLUSIVE survives a tainted host" 125 "INCONCLUSIVE ranges-overlap"
check "...and a forced taint really did taint" 125 "TAINTED"

run_ab "$HOST" --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 2 --allow-busy-host
check_verdict "SEPARATED-WEAK survives a tainted host" 125 "SEPARATED-WEAK n=2-per-arm"

run_ab "$HOST" --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 3 --allow-busy-host
check_verdict "SEPARATED survives a tainted host" 125 "SEPARATED 0."

unset FORCE_UNMEASURED_ROWS

# --- the entry quiescence gate still fires -----------------------------------
#
# Without this the whole file is satisfied by a --synthetic-arms that became
# unconditional: every case above would stay green while the gate it exists to
# protect had stopped existing.
write_ps_shim "$WORK/hostilebin" \
	"n=\$(cat '$WORK/hog.cnt' 2>/dev/null || echo 0)" \
	"echo \$((n + 1)) >'$WORK/hog.cnt'" \
	'printf "%6d %12s %s\n" 4242 "0:$((n * 100)).00" /usr/local/bin/hog' \
	"$IDLE_ROWS"

run_ab "REALHOST:$WORK/hostilebin" --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 2
check "a busy host is refused before anything is measured" 125 "host is not quiescent"

# --- a stub-armed result must not be promotable ------------------------------
#
# The flag is only worth recording if something refuses on it. This takes the
# result file the last synthetic comparison actually wrote -- not a hand-built
# one -- so the field name in the harness and the field name in the promoter
# cannot drift apart without a failure here.
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 2
SYNTH_RESULT="$(printf '%s' "$OUT" | sed -n 's/^result     //p' | tail -1)"
if [ -n "$SYNTH_RESULT" ] && [ -r "$SYNTH_RESULT" ]; then
	INGEST_OUT="$(python3 "$ROOT/scripts/ingest/llama_ab_ingest.py" --dry-run "$SYNTH_RESULT" \
		--prompt-file "$PROMPT" --prompt-name selftest --model stub \
		--weight-quant mxfp8 --arm-a-backend llama.cpp --arm-b-backend llama.cpp \
		--arm-a-kv-quant none --arm-b-kv-quant none 2>&1)"
	INGEST_CODE=$?
	if [ "$INGEST_CODE" -eq 2 ] && printf '%s' "$INGEST_OUT" | grep -q "no waiver for this"; then
		echo "  PASS  a --synthetic-arms result is refused by the promoter, with no waiver"
		PASS=$((PASS + 1))
	else
		echo "  FAIL  the promoter accepted a stub-armed result (exit $INGEST_CODE)"
		printf '%s\n' "$INGEST_OUT" | sed 's/^/        /' | tail -6
		FAIL=$((FAIL + 1))
	fi
else
	echo "  FAIL  the synthetic comparison wrote no result file to promote"
	FAIL=$((FAIL + 1))
fi

# --- the harness must never write runs.db ------------------------------------
if find "$WORK/home" -name 'runs.db*' -print -quit 2>/dev/null | grep -q .; then
	echo "  FAIL  harness wrote a metrics DB"
	FAIL=$((FAIL + 1))
else
	echo "  PASS  no metrics DB written"
	PASS=$((PASS + 1))
fi

LOCAL=$((PASS + FAIL - SYNTHETIC_CASES - REAL_HOST_CASES - HOST_READING_CASES))
printf 'host inputs: %d synthetic-arm cases (the machine is not consulted); %d against a shimmed ps; %d ran no comparison; %d read this machine.\n' \
	"$SYNTHETIC_CASES" "$REAL_HOST_CASES" "$LOCAL" "$HOST_READING_CASES"
echo "bench_llama_ab selftest: $PASS passed, $FAIL failed"
[ "$HOST_READING_CASES" -eq 0 ] || {
	echo "  FAIL  $HOST_READING_CASES case(s) can read this machine, so this suite's answer is not a property of the code" >&2
	exit 1
}
[ "$FAIL" -eq 0 ]
