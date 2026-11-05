#!/usr/bin/env bash
# Recall test for the ingest entry: record fields, refusals and the stage-and-record tail.
#
# Recall test for: ingest/ingest.py ab codec-inertness llama-bench published
#
# Each case drives the tool in one mode over a synthetic result or CSV and a stub `rmlx` that only
# files what it is handed, then asserts the literal exit code, the reason text and what is left in
# staging and in the pending queue. `ab_selftest.sh`, `bench/llama_ab_selftest.sh` and
# `published_selftest.sh` hold the refusals of their own sources; this file holds the
# record fields of codec-inertness and llama-bench, which nothing else watches, and the tail the
# modes share: dry run, staged but not ingested, ingested, and a recorder that refuses.
#
# Usage:  bash scripts/ingest/ingest_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INGEST="$REPO_ROOT/scripts/ingest/ingest.py"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init ingest-selftest

FILED="$ST_WORK/filed"

ingest() { python3 "$INGEST" "$@"; }

# ingest_in <home> <args...> — the tool with $RMLX_HOME at <home>.
ingest_in() {
    export RMLX_HOME="$1"
    shift
    ingest "$@"
}

# ── stub recorders ────────────────────────────────────────────────────────────
#
# All three answer `metrics record --help` as a binary built with the run class. `rmlx_ok` files
# each record it is handed into $FILED and deletes it, as the real recorder does on success;
# `rmlx_fail` refuses every record with exit 3 and leaves the file; `rmlx_half` refuses the records
# that name `llama_cpp_tq` and files the rest. `rmlx_ok` also answers `metrics identity --json`.
mkdir -p "$FILED"
write_stub() { # write_stub <path> <record-branch>
    cat >"$1" <<STUB
#!/usr/bin/env bash
case "\$*" in
"metrics record --help") echo "      --file <FILE>"; echo "      --run-class <CLASS>"; exit 0 ;;
"metrics identity --json") echo '{"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb"}'; exit 0 ;;
"metrics record --file "*) $2 ;;
esac
STUB
    chmod +x "$1"
}
write_stub "$ST_WORK/rmlx_ok" "cp \"\$4\" \"$FILED/\$(basename \"\$4\")\" && rm -f \"\$4\""
write_stub "$ST_WORK/rmlx_fail" 'echo "stub: rejected" >&2; exit 3'
write_stub "$ST_WORK/rmlx_half" "if grep -q llama_cpp_tq \"\$4\"; then echo 'stub: rejected' >&2; exit 3; fi; cp \"\$4\" \"$FILED/\$(basename \"\$4\")\" && rm -f \"\$4\""
OK="$ST_WORK/rmlx_ok"
FAIL="$ST_WORK/rmlx_fail"
HALF="$ST_WORK/rmlx_half"
OLD_RECORDER="$ST_WORK/rmlx_old"
printf '#!/usr/bin/env bash\necho "      --file <FILE>"\n' >"$OLD_RECORDER"
chmod +x "$OLD_RECORDER"

fresh_home() { # fresh_home <name> — an empty $RMLX_HOME, and an empty $FILED
    rm -rf "$ST_WORK/home_$1" "$FILED"
    mkdir -p "$ST_WORK/home_$1" "$FILED"
    printf '%s' "$ST_WORK/home_$1"
}

count() { find "$1" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' '; }

# ── codec-inertness ───────────────────────────────────────────────────────────

MODELS="$ST_WORK/models"
mkdir -p "$MODELS/host__m" "$ST_WORK/prompts"
printf '{"torch_dtype": "float16"}\n' >"$MODELS/host__m/config.json"
printf '{"messages":[{"role":"user","content":"hi"}]}' >"$ST_WORK/prompts/longctx_4k.json"

CSV_HEADER="timestamp,model,prompt_tokens,prompt_tokens_measured,max_ctx,max_tokens,codec,exit_code,kv_cache_bytes,ids_sha,store_skipped,ttft_ms,decode_tps,prefill_tps,binary_sha,kv_boundary,recorded_run_class,record_status"
CSV_ROW="2026-10-01T00:00:00Z,host__m,4096,4200,8192,32,k8v8,0,1000,abc123,false,1,10.0,1.0,bin456"
write_csv() { # write_csv <name> <row>... — an inertness CSV with the given rows
    local name="$1"
    shift
    {
        echo "$CSV_HEADER"
        printf '%s\n' "$@"
    } >"$ST_WORK/$name.csv"
}
write_csv one "$CSV_ROW,,,"
write_csv bounded "$CSV_ROW,\"7,9\",,"
write_csv default_bound "$CSV_ROW,\"2,8\",,"
write_csv bad_bound "$CSV_ROW,seven,,"
write_csv recorded "$CSV_ROW,,diagnostic,landed"
write_csv failed "2026-10-01T00:00:00Z,host__m,4096,,8192,32,iso3,1,,,,,,,bin456,,,"
write_csv no_measured "2026-10-01T00:00:00Z,host__m,4096,,8192,32,k8v8,0,1000,abc123,false,1,10.0,1.0,bin456,,,"
write_csv no_ctx "2026-10-01T00:00:00Z,host__m,4096,4200,,32,k8v8,0,1000,abc123,false,1,10.0,1.0,bin456,,,"
write_csv no_fixture "2026-10-01T00:00:00Z,host__m,16384,4200,8192,32,k8v8,0,1000,abc123,false,1,10.0,1.0,bin456,,,"
write_csv wide "$CSV_ROW,,,,extra"
printf '%s\n%s\n' "timestamp,model,prompt_tokens,prompt_tokens_measured,max_ctx,max_tokens,codec,exit_code,kv_cache_bytes,ids_sha,store_skipped,binary_sha" \
    "2026-10-01T00:00:00Z,host__m,4096,4200,8192,32,k8v8,0,1000,abc123,false,bin456" >"$ST_WORK/legacy.csv"

CODEC_FLAGS=(--backend-version 9.9.9 --build-profile release-perf --models-root "$MODELS" --prompts-dir "$ST_WORK/prompts")
codec() { # codec <home> <csv-name> <args...> — codec-inertness over one CSV
    local home="$1" csv="$2"
    shift 2
    ingest_in "$home" codec-inertness "$ST_WORK/$csv.csv" "${CODEC_FLAGS[@]}" "$@"
}

home="$(fresh_home codec_fields)"
st_case codec_record_fields 0 --what "one record carries the identity the CSV and the snapshot give" \
    --has '"backend": "rmlx"' --has '"backend_version": "9.9.9"' --has '"model_namespace": "host"' \
    --has '"model": "m"' --has '"weight_quant": "fp16"' --has '"kv_quant": "k8v8"' \
    --has '"decode_config": null' --has '"ctx_max": 8192' --has '"prompt_tokens": 4200' \
    --has '"max_tokens": 32' --has '"ts_utc": "2026-10-01T00:00:00Z"' --has '"n_warmups": 0' \
    --has '"n_measure": 1' --has '"run_class": "diagnostic"' --has '"build_profile": "release-perf"' \
    --has '"hardware_tag": "m5_max_128gb"' --has '"name": "longctx_4k"' --has '<45 chars>' \
    --has 'binary_sha=bin456; token_ids_sha256_16=abc123; any_layer_skipped_store=false' \
    --has '"name": "kv_cache_bytes"' --has '"value": 1000.0' -- codec "$home" one --dry-run
st_run codec "$home" one --dry-run
why="$(st_check 0)"
[ ! -d "$home/bench" ] || why="${why:+$why
}a dry run staged a record"
st_verdict codec_dry_run_writes_nothing "$why" "a dry run stages nothing"

st_case codec_non_default_boundary 0 --what "a non-default boundary is a cell of its own" \
    --has '"decode_config": "kv_boundary/head=7,kv_boundary/tail=9"' -- codec "$home" bounded --dry-run
st_case codec_default_boundary 0 --what "the engine's default boundary is the default cell" \
    --has '"decode_config": null' -- codec "$home" default_bound --dry-run
st_case codec_unparseable_boundary 1 --what "an unreadable boundary column is refused, not filed as the default" \
    --has "unparseable kv_boundary column 'seven'" -- codec "$home" bad_bound --dry-run
st_case codec_legacy_csv_notes_the_missing_column 0 --what "a CSV from before the column is recorded at the default boundary" \
    --has 'has no `kv_boundary` column' --has '"decode_config": null' -- codec "$home" legacy --dry-run
st_case codec_missing_measured_tokens 1 --what "no measured token count is refused, not replaced by the fixture name" \
    --has 'no `prompt_tokens_measured` column' -- codec "$home" no_measured --dry-run
st_case codec_missing_ctx 1 --what "no recorded ceiling is refused, not re-derived" \
    --has 'no `max_ctx` column' -- codec "$home" no_ctx --dry-run
st_case codec_missing_fixture 1 --what "a fixture that is not checked in is refused" \
    --has 'prompt fixture not found' -- codec "$home" no_fixture --dry-run
st_case codec_wide_row 1 --what "a row wider than the header is refused" \
    --has 'carry more fields than the header names' -- codec "$home" wide --dry-run
st_case codec_recorded_row_skipped 2 --what "a row the sweep recorded is not written again" \
    --has 'skipping 1 row(s) the sweep already recorded' --has 'no usable rows' -- codec "$home" recorded --dry-run
st_case codec_failed_row_skipped 2 --what "a run that did not measure has nothing to write" \
    --has 'skipping 1 row(s) with a non-zero exit code' --has 'no usable rows' -- codec "$home" failed --dry-run
st_run codec "$home" one --record --rmlx-bin "$OLD_RECORDER"
why="$(st_check 1 --has 'rmlx_old has no `metrics record --run-class`')"
[ ! -d "$home/bench" ] || why="${why:+$why
}a record was staged before the refusal"
st_verdict codec_classless_recorder "$why" "a recorder without --run-class is refused before staging"

home="$(fresh_home codec_staged)"
st_run codec "$home" one --rmlx-bin "$OK"
why="$(st_check 0 --has 'staged 1 record(s) in' --has 'not ingested (pass --record)' --has 'so no --replay-pending sweep can claim them.')"
[ "$(count "$home/bench/codec_inertness/records")" = 1 ] || why="${why:+$why
}the record is not in staging"
[ ! -d "$home/metrics/buffer/pending" ] || why="${why:+$why
}something is in the pending queue"
st_verdict codec_staged_is_outside_the_queue "$why" "without --record the record stays in staging, outside pending"

home="$(fresh_home codec_ingested)"
st_run codec "$home" one --record --rmlx-bin "$OK"
why="$(st_check 0 --has 'staged 1 record(s) in' --has 'ingested 1 of 1')"
[ "$(count "$FILED")" = 1 ] || why="${why:+$why
}the recorder was not handed the record"
grep -q '"kv_quant": "k8v8"' "$FILED"/*.json || why="${why:+$why
}the filed record is not the one built"
[ "$(count "$home/bench/codec_inertness/records")" = 0 ] || why="${why:+$why
}the filed record is still in staging"
[ "$(count "$home/metrics/buffer/pending")" = 0 ] || why="${why:+$why
}the filed record is still in pending"
st_verdict codec_ingested_leaves_nothing_behind "$why" "a record the recorder took is neither in staging nor in pending"

home="$(fresh_home codec_refused)"
st_run codec "$home" one --record --rmlx-bin "$FAIL"
why="$(st_check 1 --has 'ingest FAILED for' --has '(exit 3)' --has 'record kept in staging, not in the pending queue' --has 'ingested 0 of 1')"
[ "$(count "$home/bench/codec_inertness/records")" = 1 ] || why="${why:+$why
}the refused record was not kept in staging"
[ "$(count "$home/metrics/buffer/pending")" = 0 ] || why="${why:+$why
}the refused record was left in pending"
st_verdict codec_refused_record_stays_in_staging "$why" "a recorder that refuses ends 1 and the record goes back to staging"

# ── llama-bench ───────────────────────────────────────────────────────────────

cat >"$ST_WORK/llama_rows.json" <<'JSON'
[{"model_filename": "/models/hf__Qwen3-8B-Q8_0.gguf", "test_time": "2026-10-01T00:00:00Z",
  "type_k": "q8_0", "type_v": "q4_0", "n_prompt": 512, "n_gen": 0, "avg_ts": 900.0,
  "stddev_ts": 9.0, "build_commit": "abc1234", "n_gpu_layers": 99},
 {"model_filename": "/models/hf__Qwen3-8B-Q8_0.gguf", "test_time": "2026-10-01T00:00:00Z",
  "type_k": "q8_0", "type_v": "q4_0", "n_prompt": 0, "n_gen": 128, "avg_ts": 50.0,
  "stddev_ts": 0.5, "build_commit": "abc1234", "n_gpu_layers": 99}]
JSON
cat >"$ST_WORK/llama_odd.json" <<'JSON'
[{"model_filename": "meta-llama/Llama-3.2-3B-Instruct", "test_time": "2026-10-01T00:00:00",
  "type_k": "iq4_nl", "type_v": "f16", "n_prompt": 0, "n_gen": 64, "avg_ts": 20.0, "n_ctx": 2048},
 "not a row",
 {"model_filename": "meta-llama/Llama-3.2-3B-Instruct", "type_k": "iq4_nl", "type_v": "f16",
  "n_prompt": 32, "n_gen": 16, "avg_ts": 0, "n_ctx": 4096}]
JSON
printf '[]\n' >"$ST_WORK/llama_empty.json"
printf '{not json' >"$ST_WORK/llama_bad.json"

home="$(fresh_home llama_dry)"
st_case llama_bench_merges_prefill_and_decode 0 --what "a prefill row and a decode row of one configuration are one record" \
    --has '"backend": "llama_cpp"' --has '"model_namespace": "hf"' --has '"model": "Qwen3-8B-Q8_0"' \
    --has '"weight_quant": "q8_0"' --has '"kv_quant": "k8v4"' --has '"prompt_tokens": 512' \
    --has '"max_tokens": 128' --has '"ctx_max": 512' --has '"git_sha": "abc1234"' \
    --has '"run_class": "gate"' --has '"hardware_tag": "m5_max_128gb"' --has '"n_measure": 2' \
    --has '"name": "prefill_tps"' --has '"name": "decode_tps_warm"' --has '"name": "step_ms_mean"' \
    --has '"value": 20.0' --has '"notes": "llama-bench n_gpu_layers=99"' \
    --has '"name": "llama_bench_synthetic"' --has 'done: 1 record(s) processed' \
    -- ingest_in "$home" llama-bench --dry-run "$ST_WORK/llama_rows.json"
st_case llama_bench_hardware_tag 0 --what "--hardware-tag overrides the default" \
    --has '"hardware_tag": "m4_max_64gb"' -- ingest_in "$home" llama-bench --dry-run --hardware-tag m4_max_64gb "$ST_WORK/llama_rows.json"
st_case llama_bench_odd_rows 0 --what "an unmapped cache type is kept verbatim; a non-object row and a row with no rate are skipped" \
    --has '"kv_quant": "iq4_nl_f16"' --has '"model_namespace": "hf"' --has '"ctx_max": 2048' \
    --has '"ts_utc": "2026-10-01T00:00:00Z"' --has 'skipping non-object row' \
    --has 'skipping group' --has 'no usable metrics' --has 'done: 1 record(s) processed' \
    -- ingest_in "$home" llama-bench --dry-run "$ST_WORK/llama_odd.json"
st_case llama_bench_empty_input 0 --what "an empty array warns and writes nothing" \
    --has 'WARN: empty input' -- ingest_in "$home" llama-bench --dry-run "$ST_WORK/llama_empty.json"
st_case llama_bench_invalid_json 1 --what "input that is not JSON ends 1" \
    --has 'ERROR: invalid JSON' -- ingest_in "$home" llama-bench --dry-run "$ST_WORK/llama_bad.json"
st_case llama_bench_missing_file 1 --what "a missing input file ends 1" \
    --has 'ERROR: input file not found' -- ingest_in "$home" llama-bench --dry-run "$ST_WORK/no_such.json"
st_case llama_bench_dry_run_excludes_record 2 --what "--dry-run and --record together are a usage error" \
    --has 'mutually exclusive' -- ingest_in "$home" llama-bench --dry-run --record "$ST_WORK/llama_rows.json"
st_run ingest_in "$home" llama-bench --record --rmlx-bin "$OLD_RECORDER" "$ST_WORK/llama_rows.json"
why="$(st_check 1 --has 'rmlx_old has no `metrics record --run-class`')"
[ ! -d "$home/metrics" ] || why="${why:+$why
}a record was written before the refusal"
st_verdict llama_bench_classless_recorder "$why" "a recorder without --run-class is refused before anything is written"

home="$(fresh_home llama_queued)"
st_run ingest_in "$home" llama-bench "$ST_WORK/llama_rows.json"
why="$(st_check 0 --has ' wrote ' --has 'done: 1 record(s) processed')"
[ "$(count "$home/metrics/buffer/pending")" = 1 ] || why="${why:+$why
}the record is not queued under pending"
st_verdict llama_bench_queues_without_record "$why" "without --record the record is queued for a later replay"

home="$(fresh_home llama_stdin)"
st_case llama_bench_reads_stdin 0 --what "'-' reads the result from stdin" \
    --has '"model": "Qwen3-8B-Q8_0"' -- ingest_in "$home" llama-bench --dry-run - <"$ST_WORK/llama_rows.json"

home="$(fresh_home llama_recorded)"
st_run ingest_in "$home" llama-bench --record --rmlx-bin "$OK" "$ST_WORK/llama_rows.json"
why="$(st_check 0 --has 'recorded OK' --has 'done: 1 record(s) processed')"
[ "$(count "$FILED")" = 1 ] || why="${why:+$why
}the recorder was not handed the record"
[ "$(count "$home/metrics/buffer/pending")" = 0 ] || why="${why:+$why
}the filed record is still in pending"
st_verdict llama_bench_records_through_the_recorder "$why" "a record the recorder took is filed once"

home="$(fresh_home llama_rejected)"
st_run ingest_in "$home" llama-bench --record --rmlx-bin "$FAIL" "$ST_WORK/llama_rows.json"
why="$(st_check 0 --has 'WARN: recorder rejected record' --has 'stderr: stub: rejected' --lacks 'recorded OK')"
[ "$(count "$home/metrics/buffer/failed")" = 1 ] || why="${why:+$why
}the rejected record is not under failed"
[ "$(count "$home/metrics/buffer/pending")" = 0 ] || why="${why:+$why
}the rejected record is still in pending"
st_verdict llama_bench_rejected_record_moves_to_failed "$why" "a rejected record leaves the queue for failed/"

# ── ab: a bench/ab.sh result (--harness rmlx, the default) ─────────────────────

printf '{"messages":[{"role":"user","content":"hi"}]}' >"$ST_WORK/prompt.json"
OK_SHA16="$(shasum -a 256 "$OK" | cut -c1-16)"
cat >"$ST_WORK/ab.json" <<JSON
{"ts_utc": "20260820T090151Z", "pattern": "ABBA BAAB",
 "shape": {"prompt_tokens": 4096, "max_tokens": 100, "max_ctx": 8192},
 "arm_a": {"label": "none", "binary": "$OK", "sha256_16": "$OK_SHA16", "args": "--kv-quant none"},
 "arm_b": {"label": "mixed", "binary": "$OK", "sha256_16": "$OK_SHA16", "args": "--kv-quant mixed_k8g64_v4g64"},
 "waivers": {"synthetic_arms": false, "busy_pct_raised": false},
 "results": [{"prompt_tokens": 3770, "verdict": "INCONCLUSIVE", "taint": "",
  "arm_a": {"median_tps": 100.0, "sd_tps": 1.0, "min_tps": 99.0, "max_tps": 101.0, "n": 4, "median_kv_cache_bytes": 1000000123},
  "arm_b": {"median_tps": 110.0, "sd_tps": 1.0, "min_tps": 109.0, "max_tps": 111.0, "n": 4, "median_kv_cache_bytes": 500000123}}]}
JSON
AB_FLAGS=(--model snap --weight-quant mxfp8 --arm-a-kv-quant none --arm-b-kv-quant mixed_k8g64_v4g64 --prompt-file "$ST_WORK/prompt.json" --keep-arm B)
ab() { # ab <home> <args...> — ab over the ab result
    local home="$1"
    shift
    ingest_in "$home" ab "$ST_WORK/ab.json" "${AB_FLAGS[@]}" "$@"
}

home="$(fresh_home ab_fields)"
st_case ab_record_fields 0 --what "the kept arm is one gate record, its identity asked of the measured binary" \
    --has '"backend_version": "9.9.9"' --has '"kv_quant": "mixed_k8g64_v4g64"' --has '"run_class": "gate"' \
    --has '"model_namespace": "mlx-community"' --has '"weight_quant": "mxfp8"' --has '"ctx_max": 8192' \
    --has '"prompt_tokens": 3770' --has '"max_tokens": 100' --has '"ts_utc": "2026-08-20T09:01:51Z"' \
    --has '"name": "decode_tps_warm"' --has '"value": 110.0' --has '"value": 500000123' \
    --has '"decode_config": null' --has 'bench/ab.sh ABBA BAAB n=4/arm' --has '<45 chars>' \
    -- ab "$home" --dry-run
st_case ab_harness_rmlx_is_the_default 0 --what "--harness rmlx reads a bench/ab.sh result as the default does" \
    --has '"kv_quant": "mixed_k8g64_v4g64"' \
    -- ingest_in "$home" ab --harness rmlx "$ST_WORK/ab.json" "${AB_FLAGS[@]}" --dry-run
st_case ab_unknown_harness 2 --what "a harness that is neither rmlx nor llama is a usage error" \
    --has 'invalid choice' \
    -- ingest_in "$home" ab --harness cuda "$ST_WORK/ab.json" "${AB_FLAGS[@]}" --dry-run
st_run ab "$home" --record --rmlx-bin "$OLD_RECORDER"
why="$(st_check 1 --has 'rmlx_old has no `metrics record --run-class`')"
[ ! -d "$home/bench" ] || why="${why:+$why
}a record was staged before the refusal"
st_verdict ab_classless_recorder "$why" "a recorder without --run-class is refused before staging"

home="$(fresh_home ab_staged)"
st_run ab "$home" --rmlx-bin "$OK"
why="$(st_check 0 --has 'staged ' --has 'not ingested (pass --record)' --has 'Inspect, then re-run')"
[ "$(count "$home/bench/perf_ab/records")" = 1 ] || why="${why:+$why
}the record is not in staging"
[ ! -d "$home/metrics/buffer/pending" ] || why="${why:+$why
}something is in the pending queue"
st_verdict ab_staged_is_outside_the_queue "$why" "without --record the record stays in staging, outside pending"

home="$(fresh_home ab_ingested)"
st_run ab "$home" --record --rmlx-bin "$OK"
why="$(st_check 0 --has 'staged ')"
[ "$(count "$FILED")" = 1 ] || why="${why:+$why
}the recorder was not handed the record"
[ "$(count "$home/bench/perf_ab/records")" = 0 ] || why="${why:+$why
}the filed record is still in staging"
st_verdict ab_ingested_leaves_nothing_behind "$why" "a record the recorder took is filed once"

home="$(fresh_home ab_refused)"
st_run ab "$home" --record --rmlx-bin "$FAIL"
why="$(st_check 1 --has 'ingest FAILED for' --has '(exit 3); record kept in staging, not in the pending queue')"
[ "$(count "$home/bench/perf_ab/records")" = 1 ] || why="${why:+$why
}the refused record was not kept in staging"
[ "$(count "$home/metrics/buffer/pending")" = 0 ] || why="${why:+$why
}the refused record was left in pending"
st_verdict ab_refused_record_stays_in_staging "$why" "a recorder that refuses ends 1 and the record goes back to staging"

# ── ab --harness llama: a bench/llama_ab.sh result ────────────────────────────

cat >"$ST_WORK/llama_ab.json" <<'JSON'
{"ts_utc": "2026-10-01T00:00:00Z", "n_ctx": 8192, "n_predict": 64, "verdict": "SEPARATED",
 "tainted": "", "synthetic_arms": false,
 "slots": [
  {"arm": "A", "decode_tps": 50.0, "prompt_tps": 900.0, "kv_mib": 100, "prompt_n": 1000, "predicted_n": 64, "peak_rss_mb": 4000, "output_first_64": "hi"},
  {"arm": "B", "decode_tps": 60.0, "prompt_tps": 800.0, "kv_mib": 50, "prompt_n": 1000, "predicted_n": 64, "peak_rss_mb": 3500, "output_first_64": "hi"},
  {"arm": "A", "decode_tps": 52.0, "prompt_tps": 910.0, "kv_mib": 100, "prompt_n": 1000, "predicted_n": 64, "peak_rss_mb": 4010, "output_first_64": "hi"},
  {"arm": "B", "decode_tps": 62.0, "prompt_tps": 810.0, "kv_mib": 50, "prompt_n": 1000, "predicted_n": 64, "peak_rss_mb": 3510, "output_first_64": "hi"}]}
JSON
sed 's/"kv_mib": 50, "prompt_n": 1000, "predicted_n": 64, "peak_rss_mb": 3510/"kv_mib": 51, "prompt_n": 1000, "predicted_n": 64, "peak_rss_mb": 3510/' "$ST_WORK/llama_ab.json" >"$ST_WORK/llama_ab_split.json"
sed 's/"tainted": ""/"tainted": "WindowServer at 60%"/' "$ST_WORK/llama_ab.json" >"$ST_WORK/llama_ab_tainted.json"
LLAMA_FLAGS=(--prompt-file "$ST_WORK/prompt.json" --prompt-name selftest --model stub --weight-quant q8_0
    --arm-a-backend llama_cpp --arm-b-backend llama_cpp_tq --arm-a-kv-quant none --arm-b-kv-quant turbo3)
llama_ab() { # llama_ab <home> <result-name> <args...> — ab --harness llama over one result
    local home="$1" result="$2"
    shift 2
    ingest_in "$home" ab --harness llama "$ST_WORK/$result.json" "${LLAMA_FLAGS[@]}" "$@"
}

home="$(fresh_home llama_ab_fields)"
st_run llama_ab "$home" llama_ab --dry-run
why="$(st_check 0 --has '"backend": "llama_cpp"' --has '"backend": "llama_cpp_tq"' --has '"run_class": "ab_arm"' \
    --has '"kv_quant": "turbo3"' --has '"model_namespace": "hf"' --has '"ctx_max": 8192' --has '"max_tokens": 64' \
    --has '"prompt_tokens": 1000' --has '"n_measure": 2' --has '"name": "prefill_tps"' --has '"value": 905.0' \
    --has '"value": 104857600' --has '"name": "peak_rss_mb"' --has '"value": 4005.0' --has '"value": 51.0' \
    --has 'bench/llama_ab.sh ABBA n=2/arm' --has '<45 chars>')"
[ "$(printf '%s\n' "$ST_OUT" | grep -c '"schema_version"')" = 2 ] || why="${why:+$why
}not one record for each arm"
st_verdict llama_ab_record_fields "$why" "both arms are written as ab_arm, one record each, with their own backend"
st_case llama_ab_keep_arm 0 --what "--keep-arm writes one gate record, under that arm's backend" \
    --has '"backend": "llama_cpp_tq"' --has '"run_class": "gate"' --lacks '"backend": "llama_cpp"' \
    -- llama_ab "$home" llama_ab --dry-run --keep-arm B
st_case llama_ab_slots_disagree 1 --what "slots of one arm that disagree on the buffer are not one configuration" \
    --has 'slots disagree on kv_mib' -- llama_ab "$home" llama_ab_split --dry-run
st_case llama_ab_tainted_refused 2 --what "a TAINTED run is refused unless waived" \
    --has 'refusing: run is TAINTED' -- llama_ab "$home" llama_ab_tainted --dry-run
st_case llama_ab_tainted_accepted 0 --what "--accept-tainted carries the taint into notes" \
    --has 'TAINTED: WindowServer at 60%' -- llama_ab "$home" llama_ab_tainted --dry-run --accept-tainted
st_case llama_ab_check_args 0 --what "--check-args checks the recorder and reads no result" \
    -- ingest_in "$home" ab --harness llama "${LLAMA_FLAGS[@]}" --check-args --record --rmlx-bin "$OK"
st_case llama_ab_check_args_classless 1 --what "--check-args refuses a recorder without --run-class" \
    --has 'rmlx_old has no `metrics record --run-class`' \
    -- ingest_in "$home" ab --harness llama "${LLAMA_FLAGS[@]}" --check-args --record --rmlx-bin "$OLD_RECORDER"
st_case llama_ab_result_required 2 --what "without --check-args the result is required" \
    --has 'the following arguments are required: result' \
    -- ingest_in "$home" ab --harness llama "${LLAMA_FLAGS[@]}"

home="$(fresh_home llama_ab_staged)"
st_run llama_ab "$home" llama_ab --rmlx-bin "$OK"
why="$(st_check 0 --has 'staged ' --has 'not ingested (pass --record)' --has 'Inspect, then re-run')"
[ "$(count "$home/bench/llama_ab/records")" = 2 ] || why="${why:+$why
}the two records are not in staging"
st_verdict llama_ab_staged_is_outside_the_queue "$why" "without --record both arms stay in staging"

home="$(fresh_home llama_ab_filed)"
st_run llama_ab "$home" llama_ab --record --rmlx-bin "$OK"
why="$(st_check 0 --has 'staged ' --lacks 'not recorded')"
[ "$(count "$FILED")" = 2 ] || why="${why:+$why
}the recorder was not handed both records"
[ "$(count "$home/bench/llama_ab/records")" = 0 ] || why="${why:+$why
}a filed record is still in staging"
st_verdict llama_ab_filed_leaves_nothing_behind "$why" "both records the recorder took are filed once"

home="$(fresh_home llama_ab_half)"
st_run llama_ab "$home" llama_ab --record --rmlx-bin "$HALF"
why="$(st_check 1 --has 'ingest FAILED for' --has 'not recorded: ' --has 'llama_cpp_tq' \
    --has 'do not re-run this ingest, which would write the 1 record(s) that landed again')"
[ "$(count "$FILED")" = 1 ] || why="${why:+$why
}the arm the recorder took was not filed"
[ "$(count "$home/bench/llama_ab/records")" = 1 ] || why="${why:+$why
}the arm the recorder refused was not kept in staging"
st_verdict llama_ab_half_recorded_names_the_one_left "$why" "a half-recorded run names the record that did not land and says not to re-run"

# ── published and the mode dispatch ───────────────────────────────────────────

printf '{}\n' >"$ST_WORK/published_empty.json"
st_case published_unlabelled_result 2 --what "a result with no label is refused, not recorded" \
    --has 'refusing: the result carries no valid label' -- ingest_in "$ST_WORK/home_published" published "$ST_WORK/published_empty.json"
st_case no_mode 2 --what "no mode is a usage error" --has 'usage:' --lacks 'Traceback' -- ingest
st_case unknown_mode 2 --what "an unknown mode is a usage error" --has 'usage:' --lacks 'Traceback' -- ingest convert

st_done
