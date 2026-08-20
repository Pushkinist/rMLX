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

run_ab() { # <extra args...> -> stdout+stderr in $OUT, code in $CODE
	OUT="$(RMLX_HOME="$WORK/home" bash "$AB" \
		--model "$MODEL" --prompt-file "$PROMPT" \
		--port "$PORT" --n-predict 4 --n-ctx 64 \
		--out-dir "$WORK/home/bench" --ready-timeout 25 \
		"$@" 2>&1)"
	CODE=$?
}

# check <label> <expected-code> <expected-substring>
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
check "identical timings read INCONCLUSIVE" 125 "INCONCLUSIVE ranges-overlap"

# --- behaviour: disjoint ranges at n=2 must be flagged weak ------------------
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 2 --allow-busy-host
check "disjoint ranges at n=2 read SEPARATED-WEAK" 125 "SEPARATED-WEAK n=2-per-arm"

# --- behaviour: disjoint ranges at n=3 read SEPARATED ------------------------
run_ab --bin-a "$STUB_A" --bin-b "$STUB_B" --pairs 3 --allow-busy-host
check "disjoint ranges at n=3 read SEPARATED" 125 "SEPARATED 0."

# --- the harness must never write runs.db ------------------------------------
if find "$WORK/home" -name 'runs.db*' -print -quit 2>/dev/null | grep -q .; then
	echo "  FAIL  harness wrote a metrics DB"
	FAIL=$((FAIL + 1))
else
	echo "  PASS  no metrics DB written"
	PASS=$((PASS + 1))
fi

echo "bench_llama_ab selftest: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
