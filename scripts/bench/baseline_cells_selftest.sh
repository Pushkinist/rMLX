#!/usr/bin/env bash
# Recall test for the baseline cells tool: canary rows, argv, records and exit codes under a stub rmlx
#
# Recall test for: bench/baseline_cells.sh --include-26b --measure inertness
#
# Each case drives the tool against a stub `rmlx` that logs the argv of every `baseline` call and
# answers each call with the reading its call number names, then asserts the argv, the CSV rows,
# the summary lines and the exit code: median and stddev of known readings, the gate record, a
# record that did not land, a run that failed, a claim another process holds, a missing binary,
# and, for the inertness sweep, the columns of each row, the default codec list and the record argv.
#
# Usage:  bash scripts/bench/baseline_cells_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init baseline-cells-selftest

BASE_IDENTITY='"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb"'
BONSAI=prism-ml__Ternary-Bonsai-2-27B-mlx-2bit
GEMMA26=mlx-community__gemma-4-26b-a4b-it-mxfp8

# make_stub <path> — a stub `rmlx`. Controls live in files under $RMLX_HOME:
#   readings     lines `<call number> <decode_tps> <prefill_tps>`; any other call reads 10.0 1.0
#   crash_on     the run whose argv holds this text exits 1 without a reading
#   silent_on    the run whose argv holds this text exits 0 without a reading
#   pending_on   the recorded run whose argv holds this text prints its reading, then exits 75
#   shared_log   name of a run log another process already wrote, which the run appends to
#   claim_held   `claim run` exits 11
#   inertness    lines `<call number> <kv_cache_bytes> <token ids> <store skipped 0|1>`: the residency the
#                run prints, its `token_ids=` line and, for 1, the packed-store-skipped line on stderr
make_stub() {
	cat >"$1" <<'STUB'
#!/usr/bin/env bash
set -eu
echo "$0" >>"$RMLX_HOME/ran"
args=" $* "
holds() {
	[ -e "$RMLX_HOME/$1" ] || return 1
	case "$args" in *"$(cat "$RMLX_HOME/$1")"*) return 0 ;; esac
	return 1
}
case "$1" in
claim)
	[ ! -e "$RMLX_HOME/claim_held" ] || exit 11
	exit 0
	;;
metrics)
	case "$2" in
	path) echo "$RMLX_HOME" ;;
	identity) cat "$RMLX_HOME/identity.json" ;;
	esac
	exit 0
	;;
baseline)
	n=$(($(cat "$RMLX_HOME/calls" 2>/dev/null || echo 0) + 1))
	echo "$n" >"$RMLX_HOME/calls"
	printf '%s\n' "$*" >>"$RMLX_HOME/baseline_argv"
	if [ -e "$RMLX_HOME/shared_log" ]; then
		log="$RMLX_HOME/logs/$(cat "$RMLX_HOME/shared_log")"
	else
		log="$RMLX_HOME/logs/$(date +%s)-$$.jsonl"
	fi
	printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"rmlx start\",\"version\":\"9.9.9\",\"run_id\":\"stub\",\"pid\":$$}}" >>"$log"
	printf '%s\n' "{\"timestamp\":\"2026-09-03T00:00:00Z\",\"level\":\"INFO\",\"fields\":{\"message\":\"cache-type resolved\",\"arch\":\"Stub\",\"kv_quant\":\"stubq\"}}" >>"$log"
	if holds crash_on; then exit 1; fi
	if holds silent_on; then exit 0; fi
	row="$(awk -v n="$n" '$1 == n { print $2, $3 }' "$RMLX_HOME/readings" 2>/dev/null || true)"
	set -- ${row:-10.0 1.0}
	bytes=1000 ids="" skipped=0
	irow="$(awk -v n="$n" '$1 == n { print $2, $3, $4 }' "$RMLX_HOME/inertness" 2>/dev/null || true)"
	[ -z "$irow" ] || read -r bytes ids skipped <<<"$irow"
	echo "baseline: model=stub  ttft_ms=1  decode_tps=$1  prefill_tps=$2  prompt_tokens=4200  kv_cache_bytes=$bytes"
	[ -z "$ids" ] || echo "baseline: token_ids=$ids"
	[ "$skipped" = 0 ] || echo "exit_prefill: packed store skipped - decode reads the bf16 mirror only" >&2
	case "$args" in *" --record "*)
		if holds pending_on; then
			echo "error: baseline: measured, but the record did not reach runs.db; it is kept at $RMLX_HOME/metrics/buffer/pending/stub.json for \`rmlx metrics record --replay-pending\`" >&2
			exit 75
		fi
		;;
	esac
	;;
esac
STUB
	chmod +x "$1"
}

# fake_root <name> <profile>... — a git repo whose scripts are the real ones and whose
# target/<profile>/rmlx is the stub.
fake_root() {
	local root name="$1" profile
	shift
	root="$(st_git_root "$name")"
	git -C "$root" -c user.name=selftest -c user.email=selftest@example.invalid \
		commit -q --allow-empty -m stub
	ln -s "$REPO_ROOT/scripts" "$root/scripts"
	for profile in "$@"; do
		mkdir -p "$root/target/$profile"
		make_stub "$root/target/$profile/rmlx"
	done
	printf '%s' "$root"
}

FAKE="$(fake_root repo release-perf)"
FAKE_RELEASE="$(fake_root repo_release release)"
FAKE_NOBIN="$(fake_root repo_nobin)"
HEAD_SHA="$(git -C "$FAKE" rev-parse --short HEAD)"

# models <name> <dir>... — a models root holding these snapshots, each a 2-bit affine checkpoint.
models() {
	local root="$ST_WORK/models_$1" dir
	shift
	for dir in "$@"; do
		mkdir -p "$root/$dir"
		printf '{"quantization": {"group_size": 64, "bits": 2}}\n' >"$root/$dir/config.json"
	done
	printf '%s' "$root"
}
MODELS="$(models one "$BONSAI")"
MODELS_TWO="$(models two "$BONSAI" "$GEMMA26")"
MODELS_BAD="$(models bad "$BONSAI")"
printf '{}\n' >"$MODELS_BAD/$BONSAI/config.json"
CELL_MODEL="$ST_WORK/models_cell/host__m"
mkdir -p "$CELL_MODEL"
printf '{"torch_dtype": "float16"}\n' >"$CELL_MODEL/config.json"

new_home() { # <name> — a fresh $RMLX_HOME holding the default identity
	local home="$ST_WORK/home_$1"
	mkdir -p "$home/logs"
	printf '%s\n' "{$BASE_IDENTITY}" >"$home/identity.json"
	printf '%s' "$home"
}

control() { # <home> <file> <text>
	printf '%s' "$3" >"$1/$2"
}

# run_tool <home> <root> <args...> — the tool under <root>, alone in an empty environment.
run_tool() {
	local home="$1" root="$2"
	shift 2
	st_run env -i PATH="/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$ST_WORK" \
		RMLX_HOME="$home" bash "$root/scripts/bench/baseline_cells.sh" "$@"
}

why=""
want() { # <what> <got> <expected> — note a difference
	[ "$2" = "$3" ] || why="${why:+$why; }$1: got [$2], want [$3]"
}

# argv_lines_with <argv-file> <text>... — how many logged runs hold every text.
argv_lines_with() {
	local file="$1" texts
	shift
	texts="$(printf '%s|' "$@")"
	awk -v texts="$texts" 'BEGIN { n = split(texts, w, "|"); n-- }
		{ ok = 1; for (i = 1; i <= n; i++) if (index($0, w[i]) == 0) ok = 0; c += ok }
		END { print c + 0 }' "$file"
}

# csv_rows <home> — the canary CSV without its header and its timestamp column.
csv_rows() {
	sed 1d "$1/bench/perf_canary.csv" | cut -d, -f2-
}

# ── the canary preset ─────────────────────────────────────────────────────────

# Call numbers of one model: 1 warmup, 2-4 measured, 5 codec probe, 6 gate record,
# 7 k8vturbo3 warmup, 8-10 k8vturbo3 measured.
home="$(new_home canary_rows_and_argv)"
control "$home" readings "$(printf '%s\n' '2 10.0 1.0' '3 14.0 1.0' '4 12.0 1.0' '8 20.0 1.0' '9 30.0 1.0' '10 25.0 1.0')"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 0 \
	--has "$BONSAI  decode_tps=12.00 ± 2.00" \
	--has "$BONSAI  [k8vturbo3] decode_tps=25.00 ± 5.00" \
	--has "SKIP mlx-community__gemma-4-e4b-it-mxfp8: model path not found" \
	--has "SKIP mlx-community__Qwen3.8-27B-4bit: model path not found" \
	--has "LEGACY CSV: $home/bench/perf_canary.csv")"
model="$MODELS/$BONSAI"
run="baseline --model $model --prompt-tokens 4096 --max-tokens 100 --max-ctx 8192"
diag="--git-sha $HEAD_SHA --record --run-class diagnostic"
label="--label canary sha=$HEAD_SHA"
want "baseline argv" "$(cat "$home/baseline_argv")" "$(printf '%s\n' \
	"$run $label warmup $diag" \
	"$run $label measured $diag" "$run $label measured $diag" "$run $label measured $diag" \
	"baseline --model $model --prompt-tokens 4096 --max-tokens 1 --max-ctx 8192" \
	"$run $label --git-sha $HEAD_SHA --record --run-class gate" \
	"$run $label k8vturbo3-warmup $diag --kv-quant k8vturbo3" \
	"$run $label k8vturbo3 $diag --kv-quant k8vturbo3" "$run $label k8vturbo3 $diag --kv-quant k8vturbo3" \
	"$run $label k8vturbo3 $diag --kv-quant k8vturbo3")"
want "csv rows" "$(csv_rows "$home")" "$(printf '%s\n' \
	"$HEAD_SHA,$BONSAI,stubq,4096,12.0,2.0000,release-perf" \
	"$HEAD_SHA,$BONSAI,k8vturbo3,4096,25.0,5.0000,release-perf")"
want "csv header" "$(head -1 "$home/bench/perf_canary.csv")" \
	"ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile"
st_verdict canary_rows_and_argv "$why" "median and stddev of the measured runs, the argv of every run, the gate record and the CSV rows"

home="$(new_home canary_identity_sha)"
control "$home" identity.json "{$BASE_IDENTITY,\"git_sha\":\"feedbee\"}"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 0)"
want "csv git_sha" "$(csv_rows "$home" | head -1 | cut -d, -f1)" feedbee
want "runs given --git-sha feedbee" "$(argv_lines_with "$home/baseline_argv" "--git-sha feedbee")" 9
want "runs labelled with it" "$(argv_lines_with "$home/baseline_argv" "--label canary sha=feedbee")" 9
want "diagnostic records" "$(argv_lines_with "$home/baseline_argv" "--record --run-class diagnostic")" 8
want "gate records" "$(argv_lines_with "$home/baseline_argv" "--record --run-class gate")" 1
st_verdict canary_identity_sha "$why" "the canary files the identity's commit and records 8 diagnostic runs and 1 gate run"

home="$(new_home canary_same_second_log)"
control "$home" shared_log 20260903-000000-9.9.9.jsonl
printf '%s\n' '{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"rmlx start","version":"9.9.9","run_id":"stub","pid":4242}}' \
	'{"timestamp":"2026-09-03T00:00:00Z","level":"INFO","fields":{"message":"cache-type resolved","arch":"Stub","kv_quant":"none"}}' \
	>"$home/logs/20260903-000000-9.9.9.jsonl"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 0)"
want "csv kv_quant" "$(csv_rows "$home" | head -1 | cut -d, -f3)" stubq
st_verdict canary_same_second_log "$why" "a probe that appends to an earlier process's log is attributed by pid"

home="$(new_home canary_gate_record_fails)"
control "$home" pending_on "--run-class gate"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 1 \
	--has "the gate record failed (exit 75)" \
	--has "metrics/buffer/pending/stub.json" \
	--has "replaying it adds the gate row")"
st_verdict canary_gate_record_fails "$why" "a gate record that does not land (exit 75) stops the canary and names its buffer file"

home="$(new_home canary_diagnostic_record_pending)"
control "$home" pending_on " measured "
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 75 \
	--has "record pending: $home/metrics/buffer/pending/stub.json" \
	--has "canary INCOMPLETE: 3 diagnostic record(s) did not land")"
want "the reading was kept" "$(csv_rows "$home" | head -1 | cut -d, -f5)" 10.0
st_verdict canary_diagnostic_record_pending "$why" "a run whose record does not land keeps its reading, names the file, and the canary ends 75"

home="$(new_home canary_measured_run_crashes)"
control "$home" crash_on " measured "
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 1 --has "the measured run failed (exit 1")"
st_verdict canary_measured_run_crashes "$why" "a measured run that fails without measuring stops the canary"

home="$(new_home canary_turbo_arm_without_a_reading)"
control "$home" silent_on " --kv-quant k8vturbo3 "
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 0)"
want "k8vturbo3 warnings" "$(grep -c 'WARN: k8vturbo3 run [0-9] failed to parse decode_tps' <<<"$ST_OUT" || true)" 3
want "csv rows" "$(csv_rows "$home" | wc -l | tr -d ' ')" 1
st_verdict canary_turbo_arm_without_a_reading "$why" "a k8vturbo3 run with no reading is warned about and skipped, not fatal"

home="$(new_home canary_unrecordable_identity)"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS_BAD"
why="$(st_check 1 --has "its identity cannot be recorded")"
want "runs before the refusal" "$(cat "$home/baseline_argv" 2>/dev/null || true)" ""
st_verdict canary_unrecordable_identity "$why" "a snapshot whose identity cannot be recorded stops the canary before any run"

home="$(new_home canary_claim_held)"
control "$home" claim_held 1
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS"
why="$(st_check 11)"
want "runs while the claim is held" "$(cat "$home/baseline_argv" 2>/dev/null || true)" ""
st_verdict canary_claim_held "$why" "a Metal claim another process holds ends the canary with exit 11, before any run"

home="$(new_home canary_binary_missing)"
run_tool "$home" "$FAKE_NOBIN" --preset canary --models-root "$MODELS"
why="$(st_check 125 --has "ERROR: binary not found at $FAKE_NOBIN/target/release-perf/rmlx")"
st_verdict canary_binary_missing "$why" "a canary with no release-perf binary ends 125"

home="$(new_home canary_takes_no_other_build)"
run_tool "$home" "$FAKE_RELEASE" --preset canary --models-root "$MODELS"
why="$(st_check 125 --has "binary not found at $FAKE_RELEASE/target/release-perf/rmlx")"
st_verdict canary_takes_no_other_build "$why" "the canary does not fall back to the release build"

home="$(new_home canary_without_26b)"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS_TWO"
why="$(st_check 0 --lacks "$GEMMA26")"
want "default-codec rows" "$(csv_rows "$home" | grep -c ',stubq,' || true)" 1
st_verdict canary_without_26b "$why" "the 26b model is not measured without --include-26b"

home="$(new_home canary_include_26b)"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS_TWO" --include-26b
why="$(st_check 0 --has "$GEMMA26  decode_tps=10.00 ± 0.00")"
want "default-codec rows" "$(csv_rows "$home" | grep -c ',stubq,' || true)" 2
want "gate records" "$(argv_lines_with "$home/baseline_argv" "--record --run-class gate")" 2
st_verdict canary_include_26b "$why" "--include-26b adds the 26b model to the canary"

# ── one cell ──────────────────────────────────────────────────────────────────

# Call numbers: 1 warmup, 2-4 measured.
home="$(new_home cell_argv_median_stddev)"
control "$home" readings "$(printf '%s\n' '2 10.0 2.0' '3 14.0 4.0' '4 12.0 3.0')"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic
why="$(st_check 0 --has "k8v8 × host__m: mean decode_tps=12.0 (±2.00), prefill_tps=3.0")"
want "runs" "$(wc -l <"$home/baseline_argv" | tr -d ' ')" 4
want "runs with the cell's options" "$(argv_lines_with "$home/baseline_argv" "--model $CELL_MODEL" "--kv-quant k8v8" "--prompt-tokens 4096" "--max-tokens 100" "--record --run-class diagnostic")" 4
want "runs given --max-ctx or --git-sha" "$(argv_lines_with "$home/baseline_argv" "--max-ctx")$(argv_lines_with "$home/baseline_argv" "--git-sha")" 00
st_verdict cell_argv_median_stddev "$why" "one warmup and three measured runs; their mean decode and prefill rate and sample stddev"

home="$(new_home cell_prompt_and_max_tokens)"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant iso3 --prompt-tokens 2048 --max-tokens 50 --record diagnostic
why="$(st_check 0)"
want "runs with the sizes" "$(argv_lines_with "$home/baseline_argv" "--prompt-tokens 2048" "--max-tokens 50" "--kv-quant iso3")" 4
st_verdict cell_prompt_and_max_tokens "$why" "--prompt-tokens and --max-tokens reach every run"

home="$(new_home codec_cell_records_diagnostic)"
run_tool "$home" "$FAKE" --kv-quant k8v8 --model "$CELL_MODEL" --record diagnostic
why="$(st_check 0)"
labelled="$(awk '/--record --run-class diagnostic/ && /--label codec_cell:[0-9]{8}T[0-9]{6}Z:k8v8( |$)/ { c++ } END { print c + 0 }' "$home/baseline_argv")"
want "runs asked to record as diagnostic" "$labelled" 4
st_verdict codec_cell_records_diagnostic "$why" "the warmup and the three measured runs record as diagnostic"

home="$(new_home codec_cell_keeps_readings_whose_record_is_pending)"
control "$home" pending_on "--record"
run_tool "$home" "$FAKE" --kv-quant k8v8 --model "$CELL_MODEL" --record diagnostic
why="$(st_check 75 \
	--has "record pending: $home/metrics/buffer/pending/stub.json" \
	--has "mean decode_tps=10.0")"
st_verdict codec_cell_keeps_readings_whose_record_is_pending "$why" "runs whose record did not land keep their readings; the cell names the files and ends 75"

home="$(new_home cell_run_fails)"
control "$home" crash_on "--record"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic
why="$(st_check 1 --has "run failed (exit 1")"
st_verdict cell_run_fails "$why" "a run that fails without measuring ends the cell with exit 1"

home="$(new_home cell_zero_reading_fails)"
control "$home" readings "3 0.0 1.0"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic
why="$(st_check 1 --has "host__m [k8v8] run 2 read 0.0 tok/s")"
want "runs" "$(wc -l <"$home/baseline_argv" | tr -d ' ')" 3
st_verdict cell_zero_reading_fails "$why" "a measured run that reads 0 tok/s ends the cell with exit 1 and no later run"

home="$(new_home cell_claim_held)"
control "$home" claim_held 1
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic
why="$(st_check 11)"
st_verdict cell_claim_held "$why" "a Metal claim another process holds ends the cell with exit 11"

home="$(new_home cell_binary_flag)"
make_stub "$ST_WORK/rmlx_named"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic --binary "$ST_WORK/rmlx_named"
why="$(st_check 0)"
want "binary that ran" "$(sort -u "$home/ran")" "$ST_WORK/rmlx_named"
st_verdict cell_binary_flag "$why" "--binary names the rmlx that runs, over the checkout's build"

home="$(new_home cell_default_binary_is_release_perf)"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic
why="$(st_check 0)"
want "binary that ran" "$(sort -u "$home/ran")" "$FAKE/target/release-perf/rmlx"
st_verdict cell_default_binary_is_release_perf "$why" "with no --binary the cell runs the checkout's release-perf build"

home="$(new_home cell_default_binary_falls_back_to_release)"
run_tool "$home" "$FAKE_RELEASE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic
why="$(st_check 0)"
want "binary that ran" "$(sort -u "$home/ran")" "$FAKE_RELEASE/target/release/rmlx"
st_verdict cell_default_binary_falls_back_to_release "$why" "with no release-perf build the cell runs the release build"

# ---- flags the cell gained and the refusals of the unified command line

home="$(new_home cell_binary_missing)"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic --binary "$ST_WORK/no_such_rmlx"
why="$(st_check 125 --has "ERROR: binary not found at $ST_WORK/no_such_rmlx")"
st_verdict cell_binary_missing "$why" "a --binary that is not there ends 125"

home="$(new_home canary_zero_reading_stops)"
control "$home" readings "3 0.0 1.0"
run_tool "$home" "$FAKE" --preset canary --models-root "$MODELS_TWO" --include-26b
why="$(st_check 1 --has "$BONSAI [default codec] run 2 read 0.0 tok/s")"
want "csv rows" "$(csv_rows "$home" 2>/dev/null || true)" ""
want "gate records" "$(argv_lines_with "$home/baseline_argv" "--run-class gate")" 0
want "runs of the later model" "$(argv_lines_with "$home/baseline_argv" "$GEMMA26")" 0
st_verdict canary_zero_reading_stops "$why" "a measured run that reads 0 tok/s stops the canary before a row, a gate record or the next model"

home="$(new_home cell_warmup_and_runs)"
control "$home" readings "$(printf '%s\n' '1 10.0 1.0' '2 20.0 3.0')"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --warmup 0 --runs 2 --record diagnostic
why="$(st_check 0 --has "mean decode_tps=15.0 (±7.07), prefill_tps=2.0")"
want "runs" "$(wc -l <"$home/baseline_argv" | tr -d ' ')" 2
st_verdict cell_warmup_and_runs "$why" "--warmup and --runs set how many runs are made and averaged"

home="$(new_home cell_label)"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8 --record diagnostic --label my-cell
why="$(st_check 0)"
want "runs labelled my-cell" "$(argv_lines_with "$home/baseline_argv" "--label my-cell --record")" 4
st_verdict cell_label "$why" "--label names every run of the cell"

home="$(new_home cell_without_record)"
run_tool "$home" "$FAKE" --model "$CELL_MODEL" --kv-quant k8v8
why="$(st_check 0)"
want "runs asked to record" "$(argv_lines_with "$home/baseline_argv" "--record")" 0
want "files under the data root" "$(ls "$home/bench" 2>/dev/null || true)" ""
st_verdict cell_without_record "$why" "a cell given no --record asks for no record and writes no file"

# ── the inertness measure ─────────────────────────────────────────────────────

INERTNESS_HEADER="timestamp,model,prompt_tokens,prompt_tokens_measured,max_ctx,max_tokens,codec,exit_code,kv_cache_bytes,ids_sha,store_skipped,ttft_ms,decode_tps,prefill_tps,binary_sha,kv_boundary,recorded_run_class,record_status"
DEFAULT_CODECS="none k8v8 k8v4 planar planar3 planar_k k8vturbo3 k8vturbo3tcq k8vturbo2 k8vturbo2tcq tsym3 tsym4 iso3 iso4 rotor3 rotor4 rotor_k_4_asym_v4_g64 mixed_k8g64_v4g64 rot_k_v8g64 iso3_sym iso4_sym k_iso3 k_iso4 rotor3_sym rotor4_sym k_rotor3 k_rotor4"
STUB_SHA="$(shasum -a 256 "$FAKE/target/release-perf/rmlx" | cut -c1-12)"
IDS_123_SHA="$(printf '1,2,3\n' | shasum -a 256 | cut -c1-16)"
IDS_456_SHA="$(printf '4,5,6\n' | shasum -a 256 | cut -c1-16)"

# inertness_rows <home> — the inertness CSV without its header and its timestamp column.
inertness_rows() {
	sed 1d "$1/bench/codec_inertness.csv" | cut -d, -f2-
}

# argv_codecs <argv-file> — the --kv-quant values of the logged runs, in order, on one line.
argv_codecs() {
	sed -n 's/.* --kv-quant \([^ ]*\).*/\1/p' "$1" | tr '\n' ' ' | sed 's/ $//'
}

home="$(new_home inertness_rows_and_columns)"
control "$home" inertness "$(printf '%s\n' '1 5000 1,2,3 0' '2 4000 1,2,3 1' '3 3000 4,5,6 0')"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none --kv-quant k8v8 --kv-quant iso3 \
	--prompt-tokens 2048 --max-tokens 50
why="$(st_check 0)"
want "csv header" "$(head -1 "$home/bench/codec_inertness.csv")" "$INERTNESS_HEADER"
want "csv rows" "$(inertness_rows "$home")" "$(printf '%s\n' \
	"host__m,2048,4200,4658,50,none,0,5000,$IDS_123_SHA,0,1,10.0,1.0,$STUB_SHA,\"\",diagnostic,landed" \
	"host__m,2048,4200,4658,50,k8v8,0,4000,$IDS_123_SHA,1,1,10.0,1.0,$STUB_SHA,\"\",diagnostic,landed" \
	"host__m,2048,4200,4658,50,iso3,0,3000,$IDS_456_SHA,0,1,10.0,1.0,$STUB_SHA,\"\",diagnostic,landed")"
want "timestamps" "$(sed 1d "$home/bench/codec_inertness.csv" | cut -d, -f1 | grep -cE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$' || true)" 3
st_verdict inertness_rows_and_columns "$why" "one row per codec: residency, the digest of the greedy ids, whether the packed store was skipped, and the run's context"

home="$(new_home inertness_default_codecs)"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL"
why="$(st_check 0)"
want "codecs run" "$(argv_codecs "$home/baseline_argv")" "$DEFAULT_CODECS"
want "csv codecs" "$(inertness_rows "$home" | cut -d, -f6 | tr '\n' ' ' | sed 's/ $//')" "$DEFAULT_CODECS"
want "runs with the default sizes" "$(argv_lines_with "$home/baseline_argv" "--prompt-tokens 4096" "--max-tokens 100" "--max-ctx 7268")" 27
st_verdict inertness_default_codecs "$why" "no --kv-quant runs the default codec list, at 4096 prompt tokens, 100 generated and room in the context"

home="$(new_home inertness_record_argv)"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none --kv-quant k8v8 \
	--prompt-tokens 1024 --max-tokens 30 --max-ctx 9000
why="$(st_check 0)"
want "runs" "$(wc -l <"$home/baseline_argv" | tr -d ' ')" 2
want "runs with the cell's options" "$(argv_lines_with "$home/baseline_argv" "--model $CELL_MODEL" "--prompt-tokens 1024" "--max-tokens 30" "--max-ctx 9000" "--emit-token-ids" "--log debug")" 2
want "runs recording as diagnostic" "$(argv_lines_with "$home/baseline_argv" "--record --run-class diagnostic")" 2
stamp="$(sed -n 's/.*--label codec_inertness:\([0-9]\{8\}T[0-9]\{6\}Z\):none .*/\1/p' "$home/baseline_argv")"
want "label of the none run" "$(argv_lines_with "$home/baseline_argv" "--label codec_inertness:$stamp:none ")" 1
want "label of the k8v8 run" "$(argv_lines_with "$home/baseline_argv" "--label codec_inertness:$stamp:k8v8 ")" 1
want "runs given --git-sha or --warmup" "$(argv_lines_with "$home/baseline_argv" "--git-sha")$(argv_lines_with "$home/baseline_argv" "--warmup")" 00
want "max_ctx column" "$(inertness_rows "$home" | cut -d, -f4 | sort -u)" 9000
st_verdict inertness_record_argv "$why" "every run records as diagnostic under one codec_inertness stamp, emits its ids and logs at debug"

home="$(new_home inertness_kv_boundary)"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none --kv-quant iso3 --kv-boundary-layers 0,0
why="$(st_check 0)"
want "runs given the boundary" "$(argv_lines_with "$home/baseline_argv" "--kv-boundary-layers 0,0")" 2
want "kv_boundary column" "$(inertness_rows "$home" | python3 -c 'import csv, sys; print(" ".join(r[14] for r in csv.reader(sys.stdin)))')" "0,0 0,0"
want "fields per row" "$(inertness_rows "$home" | python3 -c 'import csv, sys; print(" ".join(str(len(r)) for r in csv.reader(sys.stdin)))')" "17 17"
st_verdict inertness_kv_boundary "$why" "--kv-boundary-layers reaches every run and fills the kv_boundary column as one quoted field"

home="$(new_home inertness_record_pending)"
control "$home" pending_on "--kv-quant k8v8 "
control "$home" inertness "$(printf '%s\n' '1 5000 1,2,3 0' '2 4000 1,2,3 0' '3 3000 4,5,6 0')"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none --kv-quant k8v8 --kv-quant iso3
why="$(st_check 75 --has "record pending: $home/metrics/buffer/pending/stub.json")"
want "recorded_run_class/record_status" "$(inertness_rows "$home" | cut -d, -f6,7,16,17 | tr '\n' ' ')" \
	'none,0,diagnostic,landed k8v8,0,diagnostic,pending iso3,0,diagnostic,landed '
want "the pending run kept its reading" "$(inertness_rows "$home" | sed -n 2p | cut -d, -f8,9)" "4000,$IDS_123_SHA"
want "runs" "$(wc -l <"$home/baseline_argv" | tr -d ' ')" 3
st_verdict inertness_record_pending "$why" "a run that measured and whose record did not land keeps exit_code 0 and its reading, is marked pending, and the sweep ends 75"

home="$(new_home inertness_run_fails)"
control "$home" crash_on "--kv-quant k8v8 "
control "$home" inertness "$(printf '%s\n' '1 5000 1,2,3 0' '3 3000 4,5,6 0')"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none --kv-quant k8v8 --kv-quant iso3
why="$(st_check 0)"
want "rows" "$(inertness_rows "$home" | cut -d, -f6- | tr '\n' ' ')" \
	"none,0,5000,$IDS_123_SHA,0,1,10.0,1.0,$STUB_SHA,\"\",diagnostic,landed k8v8,1,,,0,,,,$STUB_SHA,\"\",, iso3,0,3000,$IDS_456_SHA,0,1,10.0,1.0,$STUB_SHA,\"\",diagnostic,landed "
st_verdict inertness_run_fails "$why" "a run that fails keeps its row with its own exit code and no reading, no record class, and the sweep goes on"

home="$(new_home inertness_run_without_ids)"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none
why="$(st_check 0)"
want "ids_sha and store_skipped" "$(inertness_rows "$home" | cut -d, -f8,9,10)" "1000,,0"
st_verdict inertness_run_without_ids "$why" "a run that printed no token ids has an empty digest, not the digest of nothing"

home="$(new_home inertness_claim_held)"
control "$home" claim_held 1
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none --kv-quant k8v8
why="$(st_check 11)"
want "runs while the claim is held" "$(cat "$home/baseline_argv" 2>/dev/null || true)" ""
st_verdict inertness_claim_held "$why" "a Metal claim another process holds ends the sweep with exit 11, before any run"

home="$(new_home inertness_appends_under_one_header)"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant k8v8
why="$(st_check 0)"
want "header lines" "$(grep -c '^timestamp,' "$home/bench/codec_inertness.csv" || true)" 1
want "rows" "$(inertness_rows "$home" | wc -l | tr -d ' ')" 2
st_verdict inertness_appends_under_one_header "$why" "a second sweep appends its rows under the header already on disk"

home="$(new_home inertness_foreign_header)"
mkdir -p "$home/bench"
printf 'timestamp,model,codec\n' >"$home/bench/codec_inertness.csv"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none
why="$(st_check 2)"
want "runs against a file of another header" "$(cat "$home/baseline_argv" 2>/dev/null || true)" ""
want "file left as it was" "$(cat "$home/bench/codec_inertness.csv")" "timestamp,model,codec"
st_verdict inertness_foreign_header "$why" "a CSV whose header is not this one's is refused before any run and left as it was"

home="$(new_home inertness_csv_and_binary_flags)"
make_stub "$ST_WORK/rmlx_named"
run_tool "$home" "$FAKE" --measure inertness --model "$CELL_MODEL" --kv-quant none --csv "$ST_WORK/named.csv" --binary "$ST_WORK/rmlx_named"
why="$(st_check 0)"
want "binary that ran" "$(sort -u "$home/ran")" "$ST_WORK/rmlx_named"
want "rows in the named file" "$(sed 1d "$ST_WORK/named.csv" | wc -l | tr -d ' ')" 1
want "files under the data root" "$(ls "$home/bench" 2>/dev/null || true)" ""
st_verdict inertness_csv_and_binary_flags "$why" "--csv names the file the rows go to and --binary the rmlx that runs"

usage_case() { # <name> <needle> <args...>
	local name="$1" needle="$2"
	shift 2
	run_tool "$(new_home "$name")" "$FAKE" "$@"
	why="$(st_check 2 --has "$needle")"
	st_verdict "$name" "$why"
}
usage_case usage_no_arguments "--model is required without --preset"
usage_case usage_no_kv_quant "--kv-quant is required without --preset" --model "$CELL_MODEL"
usage_case usage_unknown_argument "unknown argument: --bogus" --bogus
usage_case usage_option_without_value "--model requires a value" --model
usage_case usage_unknown_preset "unknown preset: nope" --preset nope
usage_case usage_canary_without_models_root "--preset canary needs --models-root" --preset canary
usage_case usage_canary_with_cell_option "--preset canary takes none of: --model" --preset canary --models-root "$MODELS" --model "$CELL_MODEL"
usage_case usage_cell_with_canary_option "belong to --preset canary" --model "$CELL_MODEL" --kv-quant k8v8 --include-26b
usage_case usage_record_class "--record takes diagnostic" --model "$CELL_MODEL" --kv-quant k8v8 --record gate
usage_case usage_zero_runs "take a positive integer" --model "$CELL_MODEL" --kv-quant k8v8 --runs 0
usage_case usage_missing_model "model path not found: $ST_WORK/no_such_model" --model "$ST_WORK/no_such_model" --kv-quant k8v8
usage_case usage_unknown_measure "unknown measure: nope" --measure nope --model "$CELL_MODEL"
usage_case usage_canary_with_measure "--preset canary takes none of: --measure" --preset canary --models-root "$MODELS" --measure inertness
usage_case usage_inertness_with_rate_option "--measure inertness takes none of: --warmup --runs" --measure inertness --model "$CELL_MODEL" --warmup 2 --runs 3
usage_case usage_rate_with_inertness_option "--csv belong to --measure inertness" --model "$CELL_MODEL" --kv-quant k8v8 --csv "$ST_WORK/x.csv"
usage_case usage_rate_with_two_codecs "--kv-quant takes one codec without --measure inertness" --model "$CELL_MODEL" --kv-quant k8v8 --kv-quant iso3
usage_case usage_inertness_missing_model "model path not found: $ST_WORK/no_such_model" --measure inertness --model "$ST_WORK/no_such_model"

st_done
