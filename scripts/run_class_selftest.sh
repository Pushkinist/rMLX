#!/usr/bin/env bash
# run_class_selftest.sh — the producers that run `rmlx` record their runs under
# a stated class, and the ones that record a run twice do not.
#
# Each case drives the real script against a stub `rmlx` that logs the argv it
# was handed, and asserts what the script asked for or wrote:
#
#   - gpu_capture.sh and mst_capture.sh run `baseline --record --run-class
#     diagnostic`.
#   - bench/codec_inertness_probe.sh records each run as diagnostic and marks a
#     row `recorded_run_class` only when the run says it ingested its record.
#   - ingest/codec_inertness_ingest.py skips a row the probe already recorded,
#     and refuses a recorder built before the run class before writing anything.
#   - ingest/llama_bench_ingest.py refuses that recorder before writing anything.
#
# No GPU, no model, no DB.
#
# Exit codes: 0 — every case behaved; 1 — at least one did not.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_run_class_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PASSED=0
FAILED=0

verdict() { # verdict <name> <what-it-proves> <bad-reason-or-empty>
	if [ -z "$3" ]; then
		printf 'ok    %-40s %s\n' "$1" "$2"
		PASSED=$((PASSED + 1))
	else
		printf 'FAIL  %-40s %s\n' "$1" "$2"
		printf '        %s\n' "$3"
		FAILED=$((FAILED + 1))
	fi
}

MODEL="$WORK/models/host__m"
mkdir -p "$MODEL"

# make_rmlx <path> — a stub `rmlx` that logs each `baseline` argv to
# $ARGV_LOG, answers `claim run -- <cmd>` by running the command, prints the
# summary line a baseline prints, and says it ingested its record unless the
# argv names the codec in $STUB_NO_INGEST.
make_rmlx() {
	cat >"$1" <<'STUB'
#!/usr/bin/env bash
set -eu
case "$1" in
claim) shift 3; exec "$@" ;;
baseline)
	if [ "${2:-}" = --help ]; then echo "  --gpu-capture <PATH>  --record  --run-class <CLASS>"; exit 0; fi
	printf '%s\n' "$*" >>"$ARGV_LOG"
	echo "baseline: model=stub  ttft_ms=1  decode_tps=10.0  prefill_tps=1.0  prompt_tokens=4200  kv_cache_bytes=1000"
	echo "baseline: token_ids=1,2,3"
	case " $* " in
	*" --kv-quant ${STUB_NO_INGEST:-none-matches} "*) ;;
	*) echo "INFO rmlx_cli::commands::baseline: baseline: ingested record into runs.db" >&2 ;;
	esac
	prev=""
	for a in "$@"; do
		[ "$prev" = --gpu-capture ] && mkdir -p "$a"
		prev="$a"
	done
	;;
esac
STUB
	chmod +x "$1"
}

# ── gpu_capture.sh ────────────────────────────────────────────────────────────
#
# The script runs the binary at a fixed path under its own repo root, so it is
# driven from a fake root whose scripts are the real ones, with the two host
# preflights replaced by stubs that pass.
FAKE="$WORK/repo"
mkdir -p "$FAKE/scripts" "$FAKE/target/release-debug"
for f in "$REPO_ROOT"/scripts/*; do
	ln -s "$f" "$FAKE/scripts/$(basename "$f")"
done
for f in gputrace_preflight.sh mlx_preflight.sh; do
	rm "$FAKE/scripts/$f"
	printf '#!/usr/bin/env bash\nexit 0\n' >"$FAKE/scripts/$f"
done
make_rmlx "$FAKE/target/release-debug/rmlx"

ARGV_LOG="$WORK/gpu_capture.argv"
: >"$ARGV_LOG"
set +e
ARGV_LOG="$ARGV_LOG" bash "$FAKE/scripts/gpu_capture.sh" --kv-quant none --model "$MODEL" \
	--out-dir "$WORK/traces" --keep-all >"$WORK/gpu_capture.out" 2>&1
status=$?
set -e
bad=""
[ "$status" -eq 0 ] || bad="exit=$status: $(tail -2 "$WORK/gpu_capture.out" | tr '\n' ' ')"
grep -q -- '--record --run-class diagnostic --label gpu_capture:' "$ARGV_LOG" ||
	bad="${bad:+$bad; }baseline argv: $(cat "$ARGV_LOG")"
verdict gpu_capture_records_diagnostic "the capture run is recorded as diagnostic" "$bad"

# ── mst_capture.sh ────────────────────────────────────────────────────────────
#
# `xcrun` is a shim: `-f xctrace` answers, and `xctrace record --launch -- cmd`
# logs the launched argv. No trace is written, so the script stops after the
# recording; the case reads only what it launched.
mkdir -p "$WORK/xcbin"
cat >"$WORK/xcbin/xcrun" <<'SHIM'
#!/usr/bin/env bash
[ "$1" = -f ] && exit 0
if [ "$1 $2" = "xctrace record" ]; then
	while [ $# -gt 0 ] && [ "$1" != -- ]; do shift; done
	shift
	printf '%s\n' "$*" >>"$XCRUN_LOG"
fi
exit 0
SHIM
chmod +x "$WORK/xcbin/xcrun"
MST_BIN="$WORK/rmlx_mst"
make_rmlx "$MST_BIN"
XCRUN_LOG="$WORK/mst.launch"
: >"$XCRUN_LOG"
set +e
PATH="$WORK/xcbin:$PATH" XCRUN_LOG="$XCRUN_LOG" ARGV_LOG="$WORK/mst.argv" RMLX_HOME="$WORK/home" \
	bash "$REPO_ROOT/scripts/mst_capture.sh" --model "$MODEL" --binary "$MST_BIN" \
	>"$WORK/mst.out" 2>&1
set -e
bad=""
grep -q -- 'baseline --record --run-class diagnostic --label mst_capture:' "$XCRUN_LOG" ||
	bad="launched argv: $(cat "$XCRUN_LOG"); $(tail -2 "$WORK/mst.out" | tr '\n' ' ')"
verdict mst_capture_records_diagnostic "the traced run is launched recording as diagnostic" "$bad"

# ── codec_inertness_probe.sh ──────────────────────────────────────────────────
PROBE_BIN="$WORK/rmlx_probe"
make_rmlx "$PROBE_BIN"
CSV="$WORK/inertness.csv"
ARGV_LOG="$WORK/probe.argv"
: >"$ARGV_LOG"
set +e
ARGV_LOG="$ARGV_LOG" STUB_NO_INGEST=k8v8 RMLX_BINARY="$PROBE_BIN" RMLX_HOME="$WORK/home" \
	bash "$REPO_ROOT/scripts/bench/codec_inertness_probe.sh" --model "$MODEL" \
	--codecs "none k8v8" --out "$CSV" >"$WORK/probe.out" 2>&1
status=$?
set -e
bad=""
[ "$status" -eq 0 ] || bad="exit=$status: $(tail -2 "$WORK/probe.out" | tr '\n' ' ')"
[ "$(grep -c -- '--record --run-class diagnostic --label codec_inertness:' "$ARGV_LOG" || true)" = 2 ] ||
	bad="${bad:+$bad; }not every run asked to record as diagnostic: $(cat "$ARGV_LOG")"
column="$(python3 -c '
import csv, sys
rows = {r["codec"]: r.get("recorded_run_class") for r in csv.DictReader(open(sys.argv[1]))}
print(rows.get("none"), rows.get("k8v8"))' "$CSV" 2>&1)"
[ "$column" = "diagnostic " ] ||
	bad="${bad:+$bad; }recorded_run_class none/k8v8 = '$column' (want 'diagnostic' and empty)"
verdict probe_marks_only_ingested_runs "each run records as diagnostic; only an ingested one is marked" "$bad"

# ── codec_inertness_ingest.py ─────────────────────────────────────────────────
INGEST="$REPO_ROOT/scripts/ingest/codec_inertness_ingest.py"
set +e
out="$(RMLX_HOME="$WORK/home" python3 "$INGEST" --dry-run "$CSV" \
	--backend-version 9.9.9 --build-profile release-perf 2>&1)"
status=$?
set -e
bad=""
[ "$status" -eq 0 ] || bad="exit=$status: $out"
records="$(grep -c '"schema_version"' <<<"$out" || true)"
[ "$records" = 1 ] || bad="${bad:+$bad; }$records record(s), want 1 (the row the probe did not record)"
grep -q '"kv_quant": "k8v8"' <<<"$out" || bad="${bad:+$bad; }the unrecorded k8v8 row is not the one written"
grep -q '"run_class": "diagnostic"' <<<"$out" || bad="${bad:+$bad; }the record is not diagnostic"
verdict ingest_skips_recorded_rows "a row the probe recorded is skipped; the other is written once" "$bad"

OLD_RECORDER="$WORK/rmlx_old"
printf '#!/usr/bin/env bash\necho "      --file <FILE>"\n' >"$OLD_RECORDER"
chmod +x "$OLD_RECORDER"
set +e
out="$(RMLX_HOME="$WORK/home" python3 "$INGEST" --record "$CSV" --rmlx-bin "$OLD_RECORDER" \
	--backend-version 9.9.9 --build-profile release-perf 2>&1)"
status=$?
set -e
bad=""
[ "$status" -ne 0 ] || bad="exit 0: a classless recorder was accepted"
grep -q "rmlx_old has no \`metrics record --run-class\`" <<<"$out" || bad="${bad:+$bad; }no refusal naming the binary: $out"
[ ! -d "$WORK/home/bench/codec_inertness/records" ] || bad="${bad:+$bad; }records were staged before the refusal"
verdict ingest_refuses_a_classless_recorder "a recorder without --run-class is refused before anything is written" "$bad"

# ── llama_bench_ingest.py ─────────────────────────────────────────────────────
#
# It finds its recorder as `rmlx` on PATH first, so the old recorder goes there.
mkdir -p "$WORK/oldbin" "$WORK/llama_root"
cp "$OLD_RECORDER" "$WORK/oldbin/rmlx"
printf '[]\n' >"$WORK/llama_bench.json"
set +e
out="$(PATH="$WORK/oldbin:$PATH" RMLX_REPO_ROOT="$WORK/llama_root" \
	python3 "$REPO_ROOT/scripts/ingest/llama_bench_ingest.py" "$WORK/llama_bench.json" --record 2>&1)"
status=$?
set -e
bad=""
[ "$status" -ne 0 ] || bad="exit 0: a classless recorder was accepted"
grep -q "oldbin/rmlx has no \`metrics record --run-class\`" <<<"$out" || bad="${bad:+$bad; }no refusal naming the binary: $out"
[ ! -d "$WORK/llama_root/metrics" ] || bad="${bad:+$bad; }a record was written before the refusal"
verdict llama_bench_refuses_a_classless_recorder "a recorder without --run-class is refused before anything is written" "$bad"

echo
echo "passed=$PASSED failed=$FAILED"
[ "$FAILED" -eq 0 ] || exit 1
