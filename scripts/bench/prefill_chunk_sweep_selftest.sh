#!/usr/bin/env bash
# Recall test for the prefill-chunk sweep: slot order, env name, records, paired verdict, exits
#
# Recall test for: bench/prefill_chunk_sweep.sh
#
# Each case drives the sweep in a fake checkout against a stub `rmlx` (`bench` answers with the JSON
# `rmlx bench --json` prints, `metrics record` keeps the record it is handed), a stub `ps` that
# shows an idle or a busy host, and a stub `date` that moves one second per reading. No model and
# no GPU runs. The cases assert the level that occupies each slot, the env var each slot gets, that
# a slot records nothing, the records built from the stub's numbers, the paired verdict, the
# token and zero-measurement refusals, the host refusals, a window that was not sampled, and a
# refused record.
#
# Usage:  bash scripts/bench/prefill_chunk_sweep_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init prefill-chunk-sweep-selftest

# make_stubs <fake root> — the stub binaries a sweep runs. Controls are files under $RMLX_HOME:
#   values.tsv    `<field> <level> <row 0> <row 1> ...`: what the slots of that level report in the
#                 field (ttft_ms, prefill_tps, decode_tps, kv_cache_bytes), the last row's value
#                 standing for later rows; a field not listed reports a constant
#   alt_digest    the level whose slots generate other tokens than the rest
#   bench_fail    `bench` exits 1
#   claim_held    `claim run` exits 11
#   hog           a foreign process burns CPU on the host
#   refuse_level  `metrics record` refuses the record whose chunk is this level
make_stubs() {
	local root="$1"
	mkdir -p "$root/bin" "$root/stubbin" "$root/prompts" "$root/models/ns__m-4bit"
	echo '{"dtype": "bfloat16"}' >"$root/models/ns__m-4bit/config.json"
	echo '{"messages": []}' >"$root/prompts/longctx_4k.json"
	echo '{"messages": []}' >"$root/prompts/longctx_8k.json"

	cat >"$root/bin/rmlx" <<'STUB'
#!/usr/bin/env bash
set -u
value_of() { # <field> <level> <row> <default>
	awk -v f="$1" -v l="$2" -v r="$3" -v d="$4" '
		$1 == f && $2 == l { i = r + 3; if (i > NF) i = NF; print $i; found = 1; exit }
		END { if (!found) print d }' "$RMLX_HOME/values.tsv"
}
case "$1" in
claim)
	if [ -e "$RMLX_HOME/claim_held" ]; then
		echo "the Metal claim is held by pid 1" >&2
		exit 11
	fi
	;;
metrics)
	case "$2" in
	identity) echo '{"backend":"rmlx","backend_version":"9.9.9","git_sha":"abc1234"}' ;;
	record)
		echo "$*" >>"$RMLX_HOME/recorded.log"
		if [ -e "$RMLX_HOME/refuse_level" ] &&
			grep -q "\"prefill_chunk=$(cat "$RMLX_HOME/refuse_level")\"" "$4"; then
			echo "Error: validate record: refused" >&2
			exit 1
		fi
		mkdir -p "$RMLX_HOME/recorded"
		cp "$4" "$RMLX_HOME/recorded/"
		rm -f "$4"
		;;
	esac
	;;
bench)
	[ ! -e "$RMLX_HOME/bench_fail" ] || { echo "bench failed" >&2; exit 1; }
	args="$*"
	ptok="" runs=2
	while [ "$#" -gt 0 ]; do
		case "$1" in
		--prompt-tokens) ptok="$2"; shift ;;
		--runs) runs="$2"; shift ;;
		esac
		shift
	done
	envs="$(env | sort | sed -n 's/^\(RMLX_PREFILL_CHUNK[A-Z0-9_]*=.*\)$/\1 /p' | tr -d '\n')"
	level="${envs#*=}"
	level="${level%% *}"
	printf 'ptok=%s\t%s\t%s\n' "$ptok" "$envs" "$args" >>"$RMLX_HOME/slots.log"
	counter="$RMLX_HOME/calls_p${ptok}_c${level}"
	row="$(cat "$counter" 2>/dev/null || echo 0)"
	echo $((row + 1)) >"$counter"
	digest=digest_a
	if [ -e "$RMLX_HOME/alt_digest" ] && [ "$(cat "$RMLX_HOME/alt_digest")" = "$level" ]; then
		digest=digest_b
	fi
	detail=""
	for ((i = 0; i < runs; i++)); do
		detail="${detail:+$detail,}{\"ttft_ms\":$(value_of ttft_ms "$level" "$row" 100)"
		detail="$detail,\"prefill_tps\":$(value_of prefill_tps "$level" "$row" 1000)"
		detail="$detail,\"decode_tps\":$(value_of decode_tps "$level" "$row" 50)"
		detail="$detail,\"kv_cache_bytes\":$(value_of kv_cache_bytes "$level" "$row" 4096)}"
	done
	printf '{"token_digest":"%s","prompt_tokens":%s,"gen_tokens":100,"arch":"qwen3","prompt":"longctx","runs_detail":[%s]}\n' \
		"$digest" "$ptok" "$detail"
	;;
esac
STUB

	cat >"$root/stubbin/ps" <<'STUB'
#!/bin/sh
n=$(cat "$RMLX_HOME/ps_calls" 2>/dev/null || echo 0)
echo $((n + 1)) >"$RMLX_HOME/ps_calls"
i=0
while [ "$i" -lt 30 ]; do
	echo "$((100 + i)) 0:01.00 /usr/sbin/proc$i"
	i=$((i + 1))
done
[ ! -e "$RMLX_HOME/hog" ] || echo "999 $((n + 1)):00.00 /usr/bin/hog"
STUB

	cat >"$root/stubbin/date" <<'STUB'
#!/bin/sh
if [ "$*" = "+%s" ]; then
	n=$(cat "$RMLX_HOME/clock" 2>/dev/null || echo 1000)
	echo $((n + 1)) >"$RMLX_HOME/clock"
	echo $((n + 1))
else
	exec /bin/date "$@"
fi
STUB
	chmod +x "$root/bin/rmlx" "$root/stubbin/ps" "$root/stubbin/date"
}

FAKE="$ST_WORK/repo"
mkdir -p "$FAKE" || st_die "cannot build a fixture: $FAKE"
ln -s "$REPO_ROOT/scripts" "$FAKE/scripts"
make_stubs "$FAKE"
MODEL="$FAKE/models/ns__m-4bit"
BIN="$FAKE/bin/rmlx"

new_home() { # <name> — a fresh $RMLX_HOME
	local home="$ST_WORK/home_$1"
	mkdir -p "$home"
	: >"$home/values.tsv"
	printf '%s' "$home"
}

# run_raw <home> <sweep args...> — the sweep alone in an empty environment, given exactly these args.
run_raw() {
	local home="$1"
	shift
	st_run env -i PATH="$FAKE/stubbin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$ST_WORK" RMLX_HOME="$home" \
		bash "$FAKE/scripts/bench/prefill_chunk_sweep.sh" "$@"
}

# run_sweep <home> <sweep args...> — three levels, one prompt length, three rows; later args win.
run_sweep() {
	local home="$1"
	shift
	run_raw "$home" --model "$MODEL" --arch-key qwen3 --binary "$BIN" --levels "256 512 1024" \
		--prompt-tokens 4096 --out "$home/out" "$@"
}

# slot_levels <home> <ptok> — the chunk level of each slot of that prompt length, in the order run.
slot_levels() {
	awk -F'\t' -v p="ptok=$2" '$1 == p { split($2, a, /[= ]/); printf "%s ", a[2] }' "$1/slots.log"
}

# square_is_latin <levels run> <n> — "latin" when each (slot position, level) pair occurs exactly once.
square_is_latin() {
	echo "$1" | awk -v n="$2" '{
		for (i = 1; i <= NF; i++) seen[(i - 1) % n, $i]++
		for (k in seen) { pairs++; if (seen[k] != 1) bad = 1 }
		print (pairs == n * n && !bad) ? "latin" : "not-latin"
	}'
}

# summarize_records <dir> — one line per record filed, in order.
summarize_records() {
	python3 -I - "$1" <<'PY'
import glob, json, sys
for path in sorted(glob.glob(sys.argv[1] + "/rec_*.json")):
    r = json.load(open(path))
    cells = " ".join("%s=%g" % (m["name"], m["value"]) for m in r["metrics"])
    print(r["prompt_tokens"], r["decode_config"], r["run_class"], cells)
PY
}

why=""
want() { # <what> <got> <expected> — note a difference
	[ "$2" = "$3" ] || why="${why:+$why; }$1: got [$2], want [$3]"
}

# --- slot order ---

home="$(new_home latin)"
run_sweep "$home"
why="$(st_check 0)"
want "levels in slot order" "$(slot_levels "$home" 4096)" "256 512 1024 512 1024 256 1024 256 512 "
want "square" "$(square_is_latin "$(slot_levels "$home" 4096)" 3)" latin
st_verdict latin_order "$why" "over three rows each level takes each slot position once, in the cyclic order"

home="$(new_home latin_2x2)"
run_sweep "$home" --levels "128 4096"
why="$(st_check 0)"
want "levels in slot order" "$(slot_levels "$home" 4096)" "128 4096 4096 128 "
st_verdict latin_order_two_levels "$why" "two levels run 128 4096 then 4096 128"

# --- the env var ---

home="$(new_home env_name)"
run_sweep "$home" --arch-key qwen3_5_moe --levels "256 512" --rows 2
why="$(st_check 0 --has "arch key   qwen3_5_moe  ->  RMLX_PREFILL_CHUNK_QWEN3_5_MOE")"
want "env of each slot" "$(awk -F'\t' '{ print $2 }' "$home/slots.log" | tr -d ' ' | tr '\n' ',')" \
	"RMLX_PREFILL_CHUNK_QWEN3_5_MOE=256,RMLX_PREFILL_CHUNK_QWEN3_5_MOE=512,RMLX_PREFILL_CHUNK_QWEN3_5_MOE=512,RMLX_PREFILL_CHUNK_QWEN3_5_MOE=256,"
st_verdict env_name_derived_from_arch_key "$why" "--arch-key qwen3_5_moe sets RMLX_PREFILL_CHUNK_QWEN3_5_MOE on each slot and no other chunk variable"

# --- slots record nothing; the records ---

home="$(new_home slots_off)"
run_sweep "$home"
why="$(st_check 0 --has "Not recorded.")"
want "slots without --metrics off" "$(grep -c -v -- ' --metrics off ' "$home/slots.log" || true)" 0
want "bench calls" "$(wc -l <"$home/slots.log" | tr -d ' ')" 9
want "record calls" "$(ls "$home/recorded.log" 2>/dev/null || true)" ""
st_verdict slots_record_nothing "$why" "every slot runs --metrics off, and a sweep without --record files no record"

home="$(new_home records)"
cat >"$home/values.tsv" <<'VALUES'
decode_tps 256 50
decode_tps 512 60
decode_tps 1024 70
prefill_tps 256 900
prefill_tps 512 1100
prefill_tps 1024 1200
VALUES
run_sweep "$home" --prompt-tokens "4096 8192" --record --db "$home/runs.db"
why="$(st_check 0)"
want "records filed" "$(summarize_records "$home/recorded")" "4096 prefill_chunk=256 diagnostic decode_tps_warm=50 kv_cache_bytes=4096 prefill_tps=900 ttft_warm_ms=100
4096 prefill_chunk=512 diagnostic decode_tps_warm=60 kv_cache_bytes=4096 prefill_tps=1100 ttft_warm_ms=100
4096 prefill_chunk=1024 diagnostic decode_tps_warm=70 kv_cache_bytes=4096 prefill_tps=1200 ttft_warm_ms=100
8192 prefill_chunk=256 diagnostic decode_tps_warm=50 kv_cache_bytes=4096 prefill_tps=900 ttft_warm_ms=100
8192 prefill_chunk=512 diagnostic decode_tps_warm=60 kv_cache_bytes=4096 prefill_tps=1100 ttft_warm_ms=100
8192 prefill_chunk=1024 diagnostic decode_tps_warm=70 kv_cache_bytes=4096 prefill_tps=1200 ttft_warm_ms=100"
want "record calls" "$(wc -l <"$home/recorded.log" | tr -d ' ')" 6
want "record calls through --db" "$(grep -c -- "--db $home/runs.db\$" "$home/recorded.log")" 6
st_verdict records_one_per_cell "$why" "one diagnostic record per prompt length and level, carrying the stub's numbers, filed through --db"

# --- the verdict ---

home="$(new_home paired)"
cat >"$home/values.tsv" <<'VALUES'
prefill_tps 256 100 1000 10000
prefill_tps 512 120 900 11000
VALUES
run_sweep "$home"
why="$(st_check 0 --has "level    512: median     900.00 [   120.00,  11000.00] n=6  paired vs 256: 1.1000 [0.9000, 1.2000]")"
st_verdict paired_verdict_not_pooled "$why" "pooled the 512 chunk is slower than 256 (900 against 1000), paired within rows it is faster (median ratio 1.1000)"

# --- the correctness refusals ---

home="$(new_home tokens)"
echo 512 >"$home/alt_digest"
run_sweep "$home" --record
why="$(st_check 1 --has "CORRECTNESS FAILURE: chunking changed the generated tokens." --has "prompt_tokens=4096:" --has "Nothing was recorded.")"
want "record calls" "$(ls "$home/recorded.log" 2>/dev/null || true)" ""
st_verdict exit_1_token_mismatch "$why" "two levels of one cell that generate different tokens end 1, naming the prompt length, and file nothing"

home="$(new_home zeroed)"
echo "decode_tps 512 0" >"$home/values.tsv"
run_sweep "$home" --record
why="$(st_check 1 --has "MEASUREMENT FAILURE" --has "level 512: decode_tps_warm")"
want "record calls" "$(ls "$home/recorded.log" 2>/dev/null || true)" ""
st_verdict exit_1_zero_measurement "$why" "a slot that reports zero ends 1, naming the level and the metric, and files nothing"

# --- the refusals that are 125 ---

home="$(new_home busy)"
: >"$home/hog"
run_sweep "$home" --record
why="$(st_check 125 --has "TAINTED" --has "/usr/bin/hog" --has "Nothing was recorded.")"
want "records built" "$(ls "$home/out/records" 2>/dev/null || true)" ""
want "record calls" "$(ls "$home/recorded.log" 2>/dev/null || true)" ""
st_verdict exit_125_busy_host "$why" "a foreign process burning CPU over the slots ends 125 and builds no record"

home="$(new_home busy_at_threshold)"
: >"$home/hog"
run_sweep "$home" --busy-pct 6000
why="$(st_check 125 --has "slot p4096 r0 s0 level 256: busy 6000.0 /usr/bin/hog")"
st_verdict exit_125_busy_at_the_threshold "$why" "a window whose foreign process burns exactly --busy-pct of one core is busy: the threshold is inclusive"

home="$(new_home not_sampled)"
mkdir -p "$home/out/raw"
: >"$home/out/raw/cpu_p4096_c256_r0_s0_after.not-sampled"
run_sweep "$home"
why="$(st_check 0 --has "row 0 slot 0 level   256  not-sampled - -" --has "row 0 slot 1 level   512  quiet 0.0 -" --lacks "TAINTED")"
st_verdict not_sampled_window_is_reported_and_not_a_taint "$why" "a slot window whose snapshot carries the not-sampled marker is reported as not sampled and taints nothing, while the other slots are still classified"

home="$(new_home claim)"
: >"$home/claim_held"
run_sweep "$home"
why="$(st_check 125 --has "another process holds the Metal context (claim probe exit 11)")"
want "bench calls" "$(ls "$home/slots.log" 2>/dev/null || true)" ""
st_verdict exit_125_metal_claim_held "$why" "a held Metal claim ends 125 before any slot runs"

home="$(new_home no_binary)"
run_sweep "$home" --binary "$FAKE/bin/nope"
why="$(st_check 125 --has "binary not found / not executable")"
st_verdict exit_125_missing_binary "$why" "a missing binary ends 125"

home="$(new_home no_model)"
run_sweep "$home" --model "$FAKE/models/nope"
why="$(st_check 125 --has "model path not found")"
st_verdict exit_125_missing_model "$why" "a missing model ends 125"

home="$(new_home no_prompt)"
run_sweep "$home" --prompt-tokens "4096 3072"
why="$(st_check 125 --has "no canonical prompt for 3072 tokens")"
st_verdict exit_125_missing_prompt "$why" "a prompt length with no canonical prompt ends 125"

home="$(new_home odd_prompt)"
run_sweep "$home" --prompt-tokens 1000
why="$(st_check 125 --has "--prompt-tokens 1000 is not a positive multiple of 1024")"
st_verdict exit_125_prompt_length_not_a_multiple "$why" "a prompt length that is no multiple of 1024 ends 125"

home="$(new_home no_model_arg)"
run_raw "$home" --arch-key qwen3 --binary "$BIN"
why="$(st_check 125 --has "--model is required")"
st_verdict exit_125_no_model_argument "$why" "a run with no --model ends 125"

home="$(new_home no_value)"
run_raw "$home" --model
why="$(st_check 125 --has "--model requires a value")"
st_verdict exit_125_option_without_value "$why" "an option with no value ends 125"

home="$(new_home unknown_arg)"
run_sweep "$home" --nope
why="$(st_check 125 --has "unknown argument: --nope")"
st_verdict exit_125_unknown_argument "$why" "an unknown argument ends 125"

home="$(new_home rows)"
run_sweep "$home" --rows 4
why="$(st_check 125 --has "--rows 4 is not a multiple of the 3 levels")"
want "bench calls" "$(ls "$home/slots.log" 2>/dev/null || true)" ""
st_verdict exit_125_unbalanced_rows "$why" "rows that are no multiple of the levels end 125"

home="$(new_home one_run)"
run_sweep "$home" --runs 1
why="$(st_check 125 --has "--runs must be at least 2")"
st_verdict exit_125_single_run "$why" "a single measured run ends 125"

home="$(new_home slot_failed)"
: >"$home/bench_fail"
run_sweep "$home"
why="$(st_check 125 --has "slot row=0 pos=0 level=256 failed (rc=1)")"
st_verdict exit_125_slot_failed "$why" "a slot whose bench fails ends 125, naming the slot"

# --- a refused record ---

home="$(new_home refused)"
echo 512 >"$home/refuse_level"
run_sweep "$home" --record
why="$(st_check 75 --has "ERROR: record not filed (exit 1)" --has "$home/out/records/rec_001.json")"
want "records attempted" "$(sed 's|.*/||' "$home/recorded.log" | tr '\n' ' ')" "rec_000.json rec_001.json rec_002.json "
want "records filed" "$(summarize_records "$home/recorded" | awk '{ print $2 }' | tr '\n' ' ')" "prefill_chunk=256 prefill_chunk=1024 "
want "refused record left" "$(ls "$home/out/records/rec_001.json" 2>/dev/null | tr '\n' ' ')" "$home/out/records/rec_001.json "
st_verdict exit_75_refused_record_goes_on "$why" "a refused record is named, the later cells are still filed, and the run ends 75"

st_done
