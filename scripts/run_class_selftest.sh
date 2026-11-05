#!/usr/bin/env bash
# run_class_selftest.sh — the producers that run `rmlx` record their runs under
# a stated class, and the ones that record a run twice do not.
#
# Each case drives the real script against a stub `rmlx` that logs the argv it
# was handed, and asserts what the script asked for or wrote:
#
#   - bench/baseline_cells.sh --measure inertness records each run as diagnostic,
#     and keeps a measured run's status apart from its record's (`landed` or
#     `pending`).
#   - eval/wikitext2_ppl.py keeps a reading whose record did not land (exit 75).
#   - ingest/ingest.py codec-inertness skips a row the sweep already recorded,
#     refuses a snapshot with no recordable label, and refuses a recorder built
#     before the run class before writing anything.
#   - ingest/ingest.py llama-bench refuses that recorder before writing anything,
#     queues its records under $RMLX_HOME, and records through the checkout's
#     build rather than an `rmlx` on PATH.
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
# float16 on purpose: a label read off the name `m` could only guess bf16.
printf '{"torch_dtype": "float16"}\n' >"$MODEL/config.json"

# make_rmlx <path> — a stub `rmlx` that logs each `baseline` argv to
# $ARGV_LOG, answers `claim run -- <cmd>` by running the command, prints the
# summary line a baseline prints, and, like the real binary, exits 75 after it,
# naming a buffer file, when the argv names the codec in $STUB_NO_INGEST, whose
# record did not land. A codec in $STUB_NO_MEASURE fails before it measures.
make_rmlx() {
	cat >"$1" <<'STUB'
#!/usr/bin/env bash
set -eu
case "$1" in
claim) shift 3; exec "$@" ;;
baseline)
	printf '%s\n' "$*" >>"$ARGV_LOG"
	case " $* " in *" --kv-quant ${STUB_NO_MEASURE:-none-matches} "*) echo "error: load failed" >&2; exit 1 ;; esac
	echo "baseline: model=stub  ttft_ms=1  decode_tps=10.0  prefill_tps=1.0  prompt_tokens=4200  kv_cache_bytes=1000"
	echo "baseline: token_ids=1,2,3"
	case " $* " in *" --kv-quant ${STUB_NO_INGEST:-none-matches} "*)
		echo "error: baseline: measured, but the record did not reach runs.db; it is kept at /x/pending/stub.json for \`rmlx metrics record --replay-pending\`" >&2
		echo "rMLX exits with code 75." >&2
		exit 75 ;;
	esac
	;;
esac
STUB
	chmod +x "$1"
}

# ── baseline_cells.sh --measure inertness ─────────────────────────────────────
PROBE_BIN="$WORK/rmlx_probe"
make_rmlx "$PROBE_BIN"
CSV="$WORK/inertness.csv"
ARGV_LOG="$WORK/probe.argv"
: >"$ARGV_LOG"
set +e
ARGV_LOG="$ARGV_LOG" STUB_NO_INGEST=k8v8 STUB_NO_MEASURE=iso3 RMLX_HOME="$WORK/home" \
	bash "$REPO_ROOT/scripts/bench/baseline_cells.sh" --measure inertness --model "$MODEL" \
	--kv-quant none --kv-quant k8v8 --kv-quant iso3 --csv "$CSV" --binary "$PROBE_BIN" >"$WORK/probe.out" 2>&1
status=$?
set -e
bad=""
[ "$status" -eq 75 ] || bad="exit=$status (want 75, a record pending): $(tail -2 "$WORK/probe.out" | tr '\n' ' ')"
[ "$(grep -- '--record --run-class diagnostic' "$ARGV_LOG" | grep -c -- '--label codec_inertness:' || true)" = 3 ] ||
	bad="${bad:+$bad; }not every run asked to record as diagnostic: $(cat "$ARGV_LOG")"
grep -q 'record pending: /x/pending/stub.json' "$WORK/probe.out" || bad="${bad:+$bad; }the pending buffer file is not named"
columns="$(python3 -c '
import csv, sys
rows = {r["codec"]: r for r in csv.DictReader(open(sys.argv[1]))}
print(" ".join("%s:%s/%s/%s" % (c, rows[c]["exit_code"], rows[c]["recorded_run_class"], rows[c]["record_status"])
               for c in ("none", "k8v8", "iso3")))' "$CSV" 2>&1)"
[ "$columns" = "none:0/diagnostic/landed k8v8:0/diagnostic/pending iso3:1//" ] ||
	bad="${bad:+$bad; }exit_code/recorded_run_class/record_status = '$columns'"
verdict probe_keeps_measured_runs_apart_from_their_records "a measured run keeps exit_code 0 whatever its record did; a pending record is named and ends the sweep 75" "$bad"

# ── ingest.py codec-inertness ─────────────────────────────────────────────────
#
# Over the sweep's own CSV: the landed row and the pending one are both
# recorded (the replay writes the pending one), and the run that did not
# measure has nothing to write, so nothing is promoted.
INGEST="$REPO_ROOT/scripts/ingest/ingest.py"
set +e
out="$(RMLX_HOME="$WORK/home" python3 "$INGEST" codec-inertness --dry-run "$CSV" --models-root "$WORK/models" \
	--backend-version 9.9.9 --build-profile release-perf 2>&1)"
status=$?
set -e
bad=""
[ "$status" -eq 2 ] || bad="exit=$status (want 2, no usable rows): $out"
grep -q 'skipping 2 row(s) the sweep already recorded, or whose record is' <<<"$out" || bad="${bad:+$bad; }the landed and the pending row are not both skipped"
! grep -q '"schema_version"' <<<"$out" || bad="${bad:+$bad; }a row was promoted"
verdict ingest_promotes_no_recorded_or_pending_row "the ingest writes neither a landed nor a pending row of the sweep's CSV" "$bad"

# A sweep from before the tool recorded its runs has no record columns, so
# each measured row is the ingest's to write, once, labelled from its config.
LEGACY="$WORK/inertness_legacy.csv"
python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cols = [c for c in rows[0] if c not in ("recorded_run_class", "record_status")]
w = csv.DictWriter(open(sys.argv[2], "w", newline=""), fieldnames=cols, extrasaction="ignore")
w.writeheader()
w.writerows(rows)' "$CSV" "$LEGACY"
set +e
out="$(RMLX_HOME="$WORK/home" python3 "$INGEST" codec-inertness --dry-run "$LEGACY" --models-root "$WORK/models" \
	--backend-version 9.9.9 --build-profile release-perf 2>&1)"
status=$?
set -e
bad=""
[ "$status" -eq 0 ] || bad="exit=$status: $out"
records="$(grep -c '"schema_version"' <<<"$out" || true)"
[ "$records" = 2 ] || bad="${bad:+$bad; }$records record(s), want 2 (the two measured rows)"
grep -q '"run_class": "diagnostic"' <<<"$out" || bad="${bad:+$bad; }the record is not diagnostic"
grep -q '"weight_quant": "fp16"' <<<"$out" || bad="${bad:+$bad; }the weight label is not the one the snapshot's config describes"
verdict ingest_writes_a_legacy_sweep_once "a sweep with no record columns has each measured row written once" "$bad"

# A snapshot whose config names a mode the metrics DB has no label for is
# refused by the ingest itself, through the one label producer, before a record
# carrying that mode is built.
mkdir -p "$WORK/models_mxfp6"
for snap in $(python3 -c 'import csv,sys; print(" ".join(sorted({r["model"] for r in csv.DictReader(open(sys.argv[1]))})))' "$LEGACY"); do
	mkdir -p "$WORK/models_mxfp6/$snap"
	printf '{"quantization": {"mode": "mxfp6", "bits": 6, "group_size": 32}}\n' >"$WORK/models_mxfp6/$snap/config.json"
done
set +e
out="$(RMLX_HOME="$WORK/home" python3 "$INGEST" codec-inertness --dry-run "$LEGACY" --models-root "$WORK/models_mxfp6" \
	--backend-version 9.9.9 --build-profile release-perf 2>&1)"
status=$?
set -e
bad=""
[ "$status" -ne 0 ] || bad="exit 0: a snapshot with no recordable label was ingested"
grep -q "describes weight quantization 'mxfp6', which is not one of the labels the metrics DB records" <<<"$out" || bad="${bad:+$bad; }no refusal naming the mode: $out"
! grep -q '"schema_version"' <<<"$out" || bad="${bad:+$bad; }a record was built"
verdict ingest_refuses_a_snapshot_with_no_recordable_label "the ingest refuses a weight mode the metrics DB has no label for" "$bad"

OLD_RECORDER="$WORK/rmlx_old"
printf '#!/usr/bin/env bash\necho "      --file <FILE>"\n' >"$OLD_RECORDER"
chmod +x "$OLD_RECORDER"
set +e
out="$(RMLX_HOME="$WORK/home" python3 "$INGEST" codec-inertness --record "$LEGACY" --rmlx-bin "$OLD_RECORDER" --models-root "$WORK/models" \
	--backend-version 9.9.9 --build-profile release-perf 2>&1)"
status=$?
set -e
bad=""
[ "$status" -ne 0 ] || bad="exit 0: a classless recorder was accepted"
grep -q "rmlx_old has no \`metrics record --run-class\`" <<<"$out" || bad="${bad:+$bad; }no refusal naming the binary: $out"
[ ! -d "$WORK/home/bench/codec_inertness/records" ] || bad="${bad:+$bad; }records were staged before the refusal"
verdict ingest_refuses_a_classless_recorder "a recorder without --run-class is refused before anything is written" "$bad"

# ── ingest.py llama-bench ─────────────────────────────────────────────────────
#
mkdir -p "$WORK/oldbin" "$WORK/llama_root"
cp -R "$REPO_ROOT/scripts" "$WORK/llama_root/scripts"
cp "$OLD_RECORDER" "$WORK/oldbin/rmlx"
printf '[]\n' >"$WORK/llama_bench.json"
set +e
out="$(RMLX_HOME="$WORK/llama_home0" \
	python3 "$REPO_ROOT/scripts/ingest/ingest.py" llama-bench "$WORK/llama_bench.json" --record \
	--rmlx-bin "$WORK/oldbin/rmlx" 2>&1)"
status=$?
set -e
bad=""
[ "$status" -ne 0 ] || bad="exit 0: a classless recorder was accepted"
grep -q "oldbin/rmlx has no \`metrics record --run-class\`" <<<"$out" || bad="${bad:+$bad; }no refusal naming the binary: $out"
[ ! -d "$WORK/llama_home0/metrics" ] || bad="${bad:+$bad; }a record was written before the refusal"
verdict llama_bench_refuses_a_classless_recorder "a recorder without --run-class is refused before anything is written" "$bad"

cat >"$WORK/llama_row.json" <<'ROW'
[{"model_filename": "/models/hf__Qwen3-8B-Q8_0.gguf", "test_time": "2026-10-01T00:00:00Z",
  "type_k": "f16", "type_v": "f16", "n_prompt": 0, "n_gen": 128, "avg_ts": 50.0,
  "stddev_ts": 0.5, "build_commit": "abc1234", "n_gpu_layers": 99}]
ROW
set +e
RMLX_HOME="$WORK/llama_home" \
	python3 "$WORK/llama_root/scripts/ingest/ingest.py" llama-bench "$WORK/llama_row.json" \
	>"$WORK/llama_queue.out" 2>&1
status=$?
set -e
bad=""
[ "$status" -eq 0 ] || bad="exit=$status: $(tail -2 "$WORK/llama_queue.out" | tr '\n' ' ')"
queued="$(ls "$WORK/llama_home/metrics/buffer/pending/"*.json 2>/dev/null | wc -l | tr -d ' ')"
[ "$queued" = 1 ] || bad="${bad:+$bad; }$queued file(s) under \$RMLX_HOME/metrics/buffer/pending (want 1)"
[ ! -d "$WORK/llama_root/metrics" ] || bad="${bad:+$bad; }a record was queued under the repo root"
verdict llama_bench_queues_under_rmlx_home "a record is queued under \$RMLX_HOME, not the repo root" "$bad"

# A recorder that has the flag and logs which copy of it ran.
make_logging_recorder() { # <path> <name>
	mkdir -p "$(dirname "$1")"
	cat >"$1" <<STUB
#!/usr/bin/env bash
if [ "\$*" = "metrics record --help" ]; then echo "      --run-class <CLASS>"; exit 0; fi
echo "$2" >>"$WORK/llama_recorder.log"
STUB
	chmod +x "$1"
}
make_logging_recorder "$WORK/pathbin/rmlx" path
make_logging_recorder "$WORK/llama_root/target/release/rmlx" checkout
: >"$WORK/llama_recorder.log"
set +e
PATH="$WORK/pathbin:$PATH" RMLX_HOME="$WORK/llama_home2" \
	python3 "$WORK/llama_root/scripts/ingest/ingest.py" llama-bench "$WORK/llama_row.json" --record \
	>"$WORK/llama_record.out" 2>&1
set -e
bad=""
used="$(sort -u "$WORK/llama_recorder.log" | tr '\n' ' ')"
[ "$used" = "checkout " ] || bad="the recorder that ran: ${used:-none} (want the checkout's build only); $(tail -2 "$WORK/llama_record.out" | tr '\n' ' ')"
verdict llama_bench_records_through_the_checkout "an rmlx on PATH is not chosen over the checkout's build" "$bad"

# ── codec_smoke_runner.sh ────────────────────────────────────────────────────
#
# The codec smoke runner reaches `rmlx baseline` through `cargo run`; a stub
# `cargo` logs what it was handed and builds nothing.
SMOKE_ROOT="$WORK/smoke_models"
SMOKE_SLUG="mlx-community__gemma-4-e4b-it-mxfp8"
mkdir -p "$SMOKE_ROOT/$SMOKE_SLUG" "$WORK/cargobin"
for f in config.json tokenizer.json model.safetensors; do : >"$SMOKE_ROOT/$SMOKE_SLUG/$f"; done
printf '#!/bin/sh\n[ "$1" = run ] && printf "%%s\\n" "$*" >>"$ARGV_LOG"\nexit 0\n' >"$WORK/cargobin/cargo"
chmod +x "$WORK/cargobin/cargo"
cat >"$WORK/smoke.toml" <<'TOML'
[[entries]]
codec_name = "k8v4"
model = "gemma4-e4b"
context_length = 32768
expected_retrieval_pct = 1.0
smoke_probe_prompts = ["coherence"]
skip_reason = ""
cli_args = "--kv-quant k8v4"
niah_filter = "niah_k8v4"
TOML
ARGV_LOG="$WORK/smoke.argv"
: >"$ARGV_LOG"
set +e
PATH="$WORK/cargobin:$PATH" ARGV_LOG="$ARGV_LOG" RMLX_O_MODELS_ROOT="$SMOKE_ROOT" RMLX_HOME="$WORK/home" \
	bash "$REPO_ROOT/scripts/release_e2e/stage6_perf/codec_smoke_runner.sh" \
	--manifest "$WORK/smoke.toml" --out "$WORK/smoke.json" >"$WORK/smoke.out" 2>&1
set -e
bad=""
grep -q -- 'baseline .*--prompt-label coherence .*--record --run-class diagnostic --label codec_smoke:k8v4:coherence' "$ARGV_LOG" ||
	bad="baseline argv: $(grep baseline "$ARGV_LOG" | head -1); $(tail -2 "$WORK/smoke.out" | tr '\n' ' ')"
verdict codec_smoke_records_diagnostic "each smoke prompt's baseline records as diagnostic" "$bad"

# A recorder whose help fails is reported as failing, not as an old binary.
BROKEN_RECORDER="$WORK/rmlx_broken"
printf '#!/usr/bin/env bash\necho "dyld: library not loaded" >&2\nexit 2\n' >"$BROKEN_RECORDER"
chmod +x "$BROKEN_RECORDER"
set +e
out="$(RMLX_HOME="$WORK/llama_home3" \
	python3 "$REPO_ROOT/scripts/ingest/ingest.py" llama-bench "$WORK/llama_bench.json" --record \
	--rmlx-bin "$BROKEN_RECORDER" 2>&1)"
set -e
bad=""
grep -q 'metrics record --help` failed (exit 2): dyld: library not loaded' <<<"$out" ||
	bad="the refusal does not say the help failed: $out"
! grep -q 'built before the run class' <<<"$out" || bad="${bad:+$bad; }a failing recorder is called an old one"
verdict recorder_whose_help_fails_is_reported_as_failing "a recorder whose --help exits non-zero is reported as failing" "$bad"

# ── eval/wikitext2_ppl.py ─────────────────────────────────────────────────────
#
# A PPL run that measured and whose record did not land keeps its number,
# names the buffer file and ends 75; one that failed to measure ends 1.
mkdir -p "$WORK/wikitext_cache"
printf 'some text\n' >"$WORK/wikitext_cache/wiki.test.raw.txt"
cat >"$WORK/rmlx_ppl" <<'STUB'
#!/usr/bin/env bash
[ -n "${STUB_PPL_CRASH:-}" ] && exit 1
echo '{"ppl": 7.5, "scored_tokens": 10, "windows": 1, "ctx_window": 4096, "stride": 2048}'
echo "error: ppl: measured, but the record did not reach runs.db; it is kept at /x/pending/ppl.json for \`rmlx metrics record --replay-pending\`" >&2
echo "rMLX exits with code 75." >&2
exit 75
STUB
chmod +x "$WORK/rmlx_ppl"
ppl_case() { # <crash>
	STUB_PPL_CRASH="$1" python3 "$REPO_ROOT/scripts/eval/wikitext2_ppl.py" --model "$MODEL" \
		--run-class diagnostic --rmlx-binary "$WORK/rmlx_ppl" --cache-dir "$WORK/wikitext_cache" \
		>"$WORK/ppl.out" 2>"$WORK/ppl.err"
}
set +e
ppl_case ""
status=$?
set -e
bad=""
[ "$status" -eq 75 ] || bad="exit=$status (want 75)"
grep -q '"ppl": 7.5' "$WORK/ppl.out" || bad="${bad:+$bad; }the PPL line was not kept"
grep -q 'kept at /x/pending/ppl.json' "$WORK/ppl.err" || bad="${bad:+$bad; }the buffer file is not named"
verdict ppl_with_a_pending_record_keeps_its_number "a PPL whose record did not land is printed, its file named, and the run ends 75" "$bad"
set +e
ppl_case 1
status=$?
set -e
[ "$status" -eq 1 ] && bad="" || bad="exit=$status (want 1)"
verdict ppl_that_did_not_measure_fails "a PPL run that failed to measure ends 1" "$bad"

# ── snapshot_identity.py against the shared label table ──────────────────────
#
# The Rust producer, `ModelConfig::weight_quant_label`, is held to the same
# table by `weight_quant_label_matches_the_shared_fixture_table`.
bad="$(python3 - "$REPO_ROOT" <<'LABELS'
import json, sys
root = sys.argv[1]
sys.path.insert(0, f"{root}/scripts/lib")
from snapshot_identity import weight_quant
table = json.load(open(f"{root}/crates/rmlx-loader/tests/fixtures/weight_quant_labels.json"))
wrong = [f"{c['name']}: {weight_quant(c['config'])!r} != {c['label']!r}"
         for c in table if weight_quant(c["config"]) != c["label"]]
print("; ".join(wrong) if len(table) >= 17 else f"the table lost cases: {len(table)}")
LABELS
)"
verdict python_label_matches_the_shared_table "snapshot_identity.py labels every config in the shared table" "$bad"

echo
echo "passed=$PASSED failed=$FAILED"
[ "$FAILED" -eq 0 ] || exit 1
