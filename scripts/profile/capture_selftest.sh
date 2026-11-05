#!/usr/bin/env bash
# Recall test for the GPU-instrument launcher: both captures and the host preflight, against shims
#
# Recall test for: profile/capture.sh gputrace mst preflight
#
# Each case runs the launcher in a fake checkout (the real scripts, a stub `rmlx` per build profile)
# under a PATH of shims for `xcrun`, `xcode-select`, `DevToolsSecurity`, `codesign`, `plutil` and
# `cargo`, so no GPU, no Xcode and no model is touched. The cases assert the argv the engine and
# the instrument are handed, the bundle directory and its retention, every refusal's exit code and
# reason, and, for the preflight, a host that passes and one case per missing prerequisite. Every
# shim logs its argv one element per line, so a quoting slip shows, and the capture variable it
# was started with, so a launch that should not carry it shows.
#
# Usage:  bash scripts/profile/capture_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init profile-capture-selftest

SHIMS="$ST_WORK/shims"
SYSBIN="$ST_WORK/sysbin"
MODEL="$ST_WORK/models/host__m"
MODEL_N="WORK/models/host__m"
mkdir -p "$SHIMS" "$SYSBIN" "$MODEL" || st_die "cannot build a fixture: $ST_WORK"

# The cases run under a PATH of these and the shims alone, so a host `xctrace` is never found.
for tool in awk basename bash cat cut date dirname du env find grep head ls mkdir python3 rm sed seq sleep sort stat tail touch tr true wc; do
	path="$(type -P "$tool")" || st_die "the host has no $tool"
	ln -s "$path" "$SYSBIN/$tool"
done

# `logcall <name> [argv...]` — what every shim runs first: argv one element per line and a `----` line
# to $CAP_CTL/<name>.argv, and the capture variable to $CAP_CTL/env.log.
cat >"$SHIMS/logcall" <<'SHIM'
#!/usr/bin/env bash
name="$1"
shift
{
	[ "$#" -eq 0 ] || printf '%s\n' "$@"
	echo ----
} >>"$CAP_CTL/$name.argv"
echo "$name MTL_CAPTURE_ENABLED=${MTL_CAPTURE_ENABLED-unset}" >>"$CAP_CTL/env.log"
SHIM

# `du` is what the bundle cap runs after a capture: it logs, then measures.
cat >"$SHIMS/du" <<'SHIM'
#!/usr/bin/env bash
logcall du "$@"
exec /usr/bin/du "$@"
SHIM

# The stub engine. Controls are files under $CAP_CTL: no_feature_flag, no_trace, baseline_rc,
# pending, claim_held. A baseline run is logged as "<profile directory> <argv>".
cat >"$ST_WORK/rmlx_stub" <<'STUB'
#!/usr/bin/env bash
set -u
profile="$(basename "$(dirname "$0")")"
case "$1" in
claim)
	logcall rmlx-claim "$@"
	if [ -e "$CAP_CTL/claim_held" ]; then
		echo "error: the Metal claim is held by pid 1" >&2
		exit 11
	fi
	shift 3
	exec "$@"
	;;
baseline)
	if [ "${2:-}" = --help ]; then
		logcall rmlx-probe "$@"
		[ -e "$CAP_CTL/no_feature_flag" ] || echo "      --gpu-capture <PATH>"
		echo "      --record"
		exit 0
	fi
	logcall rmlx "$profile" "$@"
	prev=""
	for a in "$@"; do
		if [ "$prev" = --gpu-capture ] && [ ! -e "$CAP_CTL/no_trace" ]; then mkdir -p "$a"; fi
		prev="$a"
	done
	if [ -e "$CAP_CTL/baseline_rc" ]; then
		echo "error: load failed" >&2
		exit "$(cat "$CAP_CTL/baseline_rc")"
	fi
	echo "baseline: decode_tps=10.0"
	if [ -e "$CAP_CTL/pending" ]; then
		echo "error: baseline: measured, but the record did not reach runs.db; it is kept at /x/pending/stub.json for \`rmlx metrics record --replay-pending\`" >&2
		exit 75
	fi
	;;
esac
STUB

# `xcrun`: `-f xctrace` answers, `-sdk macosx metal` is the Metal toolchain probe, and `xctrace
# record` / `export` write what the controls say. Controls: no_xctrace, no_metal, no_bundle, no_log,
# no_export, export_rc, record_rc, mode (`pending`: this run's record did not land; `other`: another
# process's record did not).
cat >"$SHIMS/xcrun" <<'SHIM'
#!/usr/bin/env bash
logcall xcrun "$@"
case "$1" in
-f) [ ! -e "$CAP_CTL/no_xctrace" ]; exit ;;
-sdk)
	[ ! -e "$CAP_CTL/no_metal" ] || exit 1
	echo "Apple metal version 32023.883"
	exit 0
	;;
esac
[ "$1" = xctrace ] || exit 9
if [ "$2" = export ]; then
	out=""
	prev=""
	for a in "$@"; do
		[ "$prev" = --output ] && out="$a"
		prev="$a"
	done
	[ -e "$CAP_CTL/no_export" ] || printf '<trace/>\n' >"$out"
	exit "$(cat "$CAP_CTL/export_rc" 2>/dev/null || echo 0)"
fi
logcall xctrace "$@"
shift 2
out="" home=""
while [ $# -gt 0 ] && [ "$1" != -- ]; do
	case "$1" in
	--output) out="$2"; shift ;;
	--env) case "$2" in RMLX_HOME=*) home="${2#RMLX_HOME=}" ;; esac; shift ;;
	esac
	shift
done
shift
logcall launched "$@"
[ -n "$home" ] || { echo "shim: the child was not handed RMLX_HOME" >&2; exit 9; }
label="" prev=""
for a in "$@"; do
	[ "$prev" = --label ] && label="$a"
	prev="$a"
done
[ -e "$CAP_CTL/no_bundle" ] || mkdir -p "$out"
mode="$(cat "$CAP_CTL/mode" 2>/dev/null || true)"
mkdir -p "$home/logs" "$home/metrics/buffer/pending"
if [ ! -e "$CAP_CTL/no_log" ]; then
	printf '%s\n' "{\"fields\":{\"message\":\"rmlx start\",\"cmd\":\"Baseline { label: Some(\\\"$label\\\") }\"}}" \
		'{"fields":{"message":"generate: prefill","prefill_ms":123.4}}' >"$home/logs/20261024-000000-9.9.9.jsonl"
	printf '%s\n' '{"fields":{"message":"rmlx start","cmd":"Info"}}' >"$home/logs/20261024-000001-9.9.9.jsonl"
	touch -t 203001010000 "$home/logs/20261024-000001-9.9.9.jsonl"
fi
[ "$mode" != pending ] || printf '{"notes": "label=%s"}\n' "$label" >"$home/metrics/buffer/pending/this.json"
[ "$mode" != other ] || printf '{"notes": "label=perf_ab:other:A:1"}\n' >"$home/metrics/buffer/pending/other.json"
exit "$(cat "$CAP_CTL/record_rc" 2>/dev/null || echo 0)"
SHIM

cat >"$SHIMS/cargo" <<'SHIM'
#!/usr/bin/env bash
logcall cargo "$@"
if [ -e "$CAP_CTL/cargo_rc" ]; then
	echo "error: parser refused the table" >&2
	exit "$(cat "$CAP_CTL/cargo_rc")"
fi
echo "stub summary"
SHIM

# The host probes of the preflight. Controls: xcode_dir (empty: none selected), dts, codesign_rc,
# entitlements (the blob codesign prints). `plutil` reads a dotted key as a key path, as the real
# one does, so it answers only a key whose dots are escaped.
cat >"$SHIMS/xcode-select" <<'SHIM'
#!/usr/bin/env bash
logcall xcode-select "$@"
[ -e "$CAP_CTL/xcode_dir" ] || { echo /Applications/Xcode.app/Contents/Developer; exit 0; }
dir="$(cat "$CAP_CTL/xcode_dir")"
[ -n "$dir" ] || exit 2
echo "$dir"
SHIM
cat >"$SHIMS/DevToolsSecurity" <<'SHIM'
#!/usr/bin/env bash
logcall DevToolsSecurity "$@"
cat "$CAP_CTL/dts" 2>/dev/null || echo "Developer mode is currently enabled."
SHIM
cat >"$SHIMS/codesign" <<'SHIM'
#!/usr/bin/env bash
logcall codesign "$@"
if [ -e "$CAP_CTL/codesign_rc" ]; then
	echo "codesign: not signed" >&2
	exit "$(cat "$CAP_CTL/codesign_rc")"
fi
if [ -e "$CAP_CTL/entitlements" ]; then
	cat "$CAP_CTL/entitlements"
else
	echo '<?xml version="1.0"?><plist version="1.0"><dict><key>com.apple.security.get-task-allow</key><true/></dict></plist>'
fi
SHIM
cat >"$SHIMS/plutil" <<'SHIM'
#!/usr/bin/env bash
logcall plutil "$@"
body="$(cat)"
case "$1" in
-lint) printf '%s' "$body" | grep -q '<plist' ;;
-extract)
	case "$2" in *[!\\].*) echo "plutil: no value at key path $2" >&2; exit 1 ;; esac
	key="${2//\\/}"
	value="$(printf '%s' "$body" | sed -n "s|.*<key>$key</key><\([a-z]*\)/>.*|\1|p")"
	[ -n "$value" ] || exit 1
	echo "$value"
	;;
esac
SHIM
chmod +x "$ST_WORK/rmlx_stub" "$SHIMS"/*

ROOT="" CTL="" CAP_HOME=""

# make_root <name> — a fake checkout with the real scripts and a stub engine per build profile, and
# its own control directory; sets ROOT, CTL and clears CAP_HOME (RMLX_HOME of the run, unset by default).
make_root() {
	ROOT="$ST_WORK/root_$1"
	CTL="$ST_WORK/ctl_$1"
	CAP_HOME=""
	mkdir -p "$ROOT/target/release-debug" "$ROOT/target/release" "$CTL" || st_die "cannot build a fixture: $ROOT"
	ln -s "$REPO_ROOT/scripts" "$ROOT/scripts"
	cp "$ST_WORK/rmlx_stub" "$ROOT/target/release-debug/rmlx"
	cp "$ST_WORK/rmlx_stub" "$ROOT/target/release/rmlx"
}

# norm — the checkout's and the scratch directory's paths and the run's timestamps made fixed.
norm() { sed -e "s|$ROOT/||g" -e "s|$ST_WORK/|WORK/|g" -e 's/[0-9]\{8\}-[0-9]\{6\}/STAMP/g'; }
settle() { ST_OUT="$(printf '%s\n' "$ST_OUT" | norm)"; }
# logged <file> — one line for each call the shim logged, its elements joined by a space.
logged() {
	if [ -e "$CTL/$1" ]; then
		awk '$0 == "----" { print line; line = ""; n = 0; next } { line = line (n++ ? " " : "") $0 }' "$CTL/$1" | norm
	fi
}
# arg_after <file> <flag> — the element that follows <flag> in each call, whole.
arg_after() {
	if [ -e "$CTL/$1" ]; then awk -v flag="$2" 'prev == flag { print } { prev = $0 }' "$CTL/$1" | norm; fi
}
# env_set — the shims that were started with the capture variable set.
env_set() {
	if [ -e "$CTL/env.log" ]; then grep -v 'MTL_CAPTURE_ENABLED=unset$' "$CTL/env.log" || true; fi
}
listing() { { ls -1 "$1" 2>/dev/null || true; } | norm | tr '\n' ' '; }

run_gputrace() {
	st_run env -i PATH="$SHIMS:$SYSBIN" HOME="$ST_WORK" CAP_CTL="$CTL" ${CAP_HOME:+"RMLX_HOME=$CAP_HOME"} \
		bash "$ROOT/scripts/profile/capture.sh" gputrace "$@"
	settle
}
run_mst() {
	st_run env -i PATH="$SHIMS:$SYSBIN" HOME="$ST_WORK" CAP_CTL="$CTL" ${CAP_HOME:+"RMLX_HOME=$CAP_HOME"} \
		bash "$ROOT/scripts/profile/capture.sh" mst "$@"
	settle
}
run_preflight() {
	st_run env -i PATH="$SHIMS:$SYSBIN" HOME="$ST_WORK" CAP_CTL="$CTL" ${CAP_HOME:+"RMLX_HOME=$CAP_HOME"} \
		bash "$ROOT/scripts/profile/capture.sh" preflight "$@"
	settle
}

why=""
want() { # <what> <got> <expected> — note a difference
	[ "$2" = "$3" ] || why="${why:+$why; }$1: got [$2], want [$3]"
}

# plant <dir> <suffix> <n> — n bundles old1..old<n> named <suffix>, each older than the next, with the
# table and channel files a System Trace leaves beside its bundle.
plant() {
	local i
	mkdir -p "$1"
	for ((i = 1; i <= $3; i++)); do
		mkdir -p "$1/old$i.$2"
		: >"$1/old$i.gpu-intervals.xml"
		: >"$1/old$i.channels.csv"
		touch -t "$(printf '2020%02d010000' "$i")" "$1/old$i.$2"
	done
}

GP_RUN="release-debug baseline --record --run-class diagnostic --label gpu_capture:STAMP --model $MODEL_N"
GP_TRACE=".rmlx/traces/host__m-none-4096tok-STAMP.gputrace"
MST_RUN="target/release/rmlx baseline --record --run-class diagnostic --label mst_capture:STAMP --model $MODEL_N"
MST_DIR=".rmlx/traces/mst"
MST_BASE="host__m-none-4096tok-STAMP"

# ===== gputrace =====

make_root gp_default
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 0 --has "capturing: codec=none prompt=4096 skip=4 steps=8 gen=18" --has "trace:     $GP_TRACE" \
	--has "done: $GP_TRACE" --has "— newest bundle (use --all to override)" --has "Offline, no Xcode:")"
want "engine argv" "$(logged rmlx.argv)" "$GP_RUN --kv-quant none --prompt-tokens 4096 --max-tokens 18 --max-ctx 4626 --max-prompt-tokens 4160 --gpu-capture $GP_TRACE --gpu-capture-skip 4 --gpu-capture-steps 8"
want "started with the capture variable" "$(env_set)" "rmlx MTL_CAPTURE_ENABLED=1"
want "bundles" "$(listing "$ROOT/.rmlx/traces")" "host__m-none-4096tok-STAMP.gputrace "
st_verdict gpu_capture_default_run "$why" "the engine runs the release-debug build with the capture variable, the window defaults and a bundle under .rmlx/traces"

make_root gp_home
CAP_HOME="$ST_WORK/gp_home_dir"
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 0)"
want "bundles" "$(listing "$ST_WORK/gp_home_dir/traces")" "host__m-none-4096tok-STAMP.gputrace "
want "default directory" "$(listing "$ROOT/.rmlx")" ""
st_verdict gpu_capture_default_directory_is_under_rmlx_home "$why" "the bundle is filed under <RMLX_HOME>/traces, not under the checkout"

make_root gp_whole_args
MODEL_SP="$ST_WORK/models/host m"
mkdir -p "$MODEL_SP"
run_gputrace --kv-quant none --model "$MODEL_SP" --keep-all
why="$(st_check 0)"
want "model" "$(arg_after rmlx.argv --model)" "WORK/models/host m"
want "trace" "$(arg_after rmlx.argv --gpu-capture)" ".rmlx/traces/host m-none-4096tok-STAMP.gputrace"
st_verdict gpu_capture_hands_a_path_with_a_space_to_the_engine_whole "$why" "a model path with a space reaches the engine as one argument, and so does the bundle path"

make_root gp_records
run_gputrace --kv-quant none --model "$MODEL" --keep-all
why="$(st_check 0)"
want "recorded as diagnostic" "$(logged rmlx.argv | grep -c -- '--record --run-class diagnostic --label gpu_capture:STAMP' || true)" 1
st_verdict gpu_capture_records_diagnostic "$why" "the capture run is recorded as diagnostic"

make_root gp_pending
: >"$CTL/pending"
run_gputrace --kv-quant none --model "$MODEL" --out-dir "$ST_WORK/traces_pending" --keep-all
why="$(st_check 75 --has "WARNING: the capture succeeded, but its diagnostic record did not land" \
	--has "retention: --keep-all" --has "done: WORK/traces_pending/host__m-none-4096tok-STAMP.gputrace" \
	--has "INCOMPLETE: the trace stands, its diagnostic record is pending (named above)")"
st_verdict gpu_capture_with_a_pending_record_keeps_its_trace "$why" "a capture whose record did not land keeps its trace, runs its retention and ends 75"

make_root gp_prompt
run_gputrace --kv-quant k8v8 --model "$MODEL" --prompt-tokens 8192
why="$(st_check 0 --has "capturing: codec=k8v8 prompt=8192 skip=4 steps=8 gen=18")"
want "engine argv" "$(logged rmlx.argv)" "$GP_RUN --kv-quant k8v8 --prompt-tokens 8192 --max-tokens 18 --max-ctx 8722 --max-prompt-tokens 8256 --gpu-capture .rmlx/traces/host__m-k8v8-8192tok-STAMP.gputrace --gpu-capture-skip 4 --gpu-capture-steps 8"
st_verdict gpu_capture_prompt_tokens_sizes_the_context "$why" "--prompt-tokens names the prompt, the bundle and the context the engine is given"

make_root gp_window
run_gputrace --kv-quant none --model "$MODEL" --skip 6 --steps 10
why="$(st_check 0 --has "skip=6 steps=10 gen=22")"
want "engine argv" "$(logged rmlx.argv)" "$GP_RUN --kv-quant none --prompt-tokens 4096 --max-tokens 22 --max-ctx 4630 --max-prompt-tokens 4160 --gpu-capture $GP_TRACE --gpu-capture-skip 6 --gpu-capture-steps 10"
st_verdict gpu_capture_window_sets_the_generation "$why" "--skip and --steps are passed on, and the default generation holds the window and four more steps"

make_root gp_gen
run_gputrace --kv-quant none --model "$MODEL" --gen 20
why="$(st_check 0 --has "gen=20")"
want "engine argv" "$(logged rmlx.argv)" "$GP_RUN --kv-quant none --prompt-tokens 4096 --max-tokens 20 --max-ctx 4628 --max-prompt-tokens 4160 --gpu-capture $GP_TRACE --gpu-capture-skip 4 --gpu-capture-steps 8"
st_verdict gpu_capture_gen_is_passed_on "$why" "--gen is the generation length and sizes the context"

make_root gp_outdir
run_gputrace --kv-quant none --model "$MODEL" --out-dir "$ST_WORK/out_g" --keep-all
why="$(st_check 0 --has "trace:     WORK/out_g/host__m-none-4096tok-STAMP.gputrace")"
want "bundles" "$(listing "$ST_WORK/out_g")" "host__m-none-4096tok-STAMP.gputrace "
want "default directory" "$(listing "$ROOT/.rmlx")" ""
st_verdict gpu_capture_out_dir "$why" "--out-dir holds the bundle and the default directory is not made"

make_root gp_retention
plant "$ROOT/.rmlx/traces" gputrace 8
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 0 --has "prune   old1.gputrace" --has "prune   old3.gputrace" --lacks "prune   old4.gputrace")"
want "bundles left" "$(ls -d "$ROOT"/.rmlx/traces/*.gputrace | wc -l | tr -d ' ')" 6
want "started with the capture variable" "$(env_set)" "rmlx MTL_CAPTURE_ENABLED=1"
want "newest old bundle" "$(ls -d "$ROOT/.rmlx/traces/old8.gputrace" | norm)" ".rmlx/traces/old8.gputrace"
st_verdict gpu_capture_retention_prunes_the_oldest "$why" "nine bundles are cut to the newest six, the new one among them"

make_root gp_keep_all
plant "$ROOT/.rmlx/traces" gputrace 8
run_gputrace --kv-quant none --model "$MODEL" --keep-all
why="$(st_check 0 --has "retention: --keep-all, cap not enforced (9 bundles in .rmlx/traces)" --lacks "prune ")"
want "bundles left" "$(ls -d "$ROOT"/.rmlx/traces/*.gputrace | wc -l | tr -d ' ')" 9
st_verdict gpu_capture_keep_all_enforces_no_cap "$why" "--keep-all leaves every bundle and counts them"

make_root gp_unknown
run_gputrace --kv-quant none --model "$MODEL" --bogus
why="$(st_check 2 --has "unknown argument")"
st_verdict gpu_capture_refuses_an_unknown_argument "$why" "an argument the capture does not take ends 2"

make_root gp_no_binary_flag
run_gputrace --kv-quant none --model "$MODEL" --binary "$ROOT/target/release/rmlx"
why="$(st_check 2 --has "unknown argument")"
want "engine runs" "$(logged rmlx.argv)" ""
st_verdict gpu_capture_takes_no_binary_flag "$why" "--binary belongs to the System Trace, not to the capture"

make_root gp_usage
run_gputrace
why="$(st_check 2 --has "--kv-quant <codec> --model <snapshot-abs-path>")"
run_gputrace --model "$MODEL"
why="${why:+$why; }$(st_check 2 --has "--kv-quant <codec> --model <snapshot-abs-path>")"
run_gputrace --kv-quant none
why="${why:+$why; }$(st_check 2 --has "--kv-quant <codec> --model <snapshot-abs-path>")"
st_verdict gpu_capture_refuses_a_missing_codec_or_model "$why" "no arguments, no codec and no model each end 2 with the usage"

make_root gp_short_gen
run_gputrace --kv-quant none --model "$MODEL" --gen 5
why="$(st_check 2 --has "--gen 5 cannot hold a 4-skip / 8-step window." --has "Raise it to at least 14")"
want "engine runs" "$(logged rmlx.argv)" ""
st_verdict gpu_capture_refuses_a_generation_shorter_than_the_window "$why" "a --gen below skip + steps + 2 ends 2 before the engine runs"

make_root gp_no_bin
rm -f "$ROOT/target/release-debug/rmlx"
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 1 --has "target/release-debug/rmlx not found" --has "make build-capture")"
st_verdict gpu_capture_refuses_a_missing_binary "$why" "no release-debug build ends 1 naming make build-capture"

make_root gp_no_feature
: >"$CTL/no_feature_flag"
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 1 --has "was built WITHOUT the metal-capture feature" --has "make build-capture")"
want "engine runs" "$(logged rmlx.argv)" ""
st_verdict gpu_capture_refuses_a_binary_without_the_feature "$why" "a binary with no --gpu-capture flag ends 1"

make_root gp_preflight
echo /Library/Developer/CommandLineTools >"$CTL/xcode_dir"
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 1 --has "FAIL  xcode-select points at Command Line Tools" --has "preflight FAILED")"
want "engine runs" "$(logged rmlx.argv)" ""
want "bundle directory" "$(listing "$ROOT/.rmlx")" ""
st_verdict gpu_capture_refuses_a_host_that_fails_the_preflight "$why" "a failing host prerequisite ends 1 before the engine runs or a directory is made"

make_root gp_run_fails
echo 3 >"$CTL/baseline_rc"
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 3 --has "capture run failed (exit 3)")"
st_verdict gpu_capture_ends_with_the_engines_exit_code "$why" "a failed engine run ends with its exit code"

make_root gp_no_trace
: >"$CTL/no_trace"
run_gputrace --kv-quant none --model "$MODEL"
why="$(st_check 1 --has "ERROR: run reported success but no trace exists at $GP_TRACE")"
st_verdict gpu_capture_refuses_success_without_a_trace "$why" "an engine that exits 0 and wrote no bundle ends 1"

# ===== mst =====

make_root mst_default
run_mst --model "$MODEL"
why="$(st_check 0 --has "recording:  model=host__m codec=none prompt=4096 gen=400 limit=12s" --has "trace:      $MST_DIR/$MST_BASE.trace" \
	--has "running:    $MST_RUN --kv-quant none --prompt-tokens 4096 --max-tokens 400 --max-ctx 5008" \
	--has "skip:       123 ms — measured prefill_ms, read back from this run's log" \
	--has "== full window (prefill included)" --has "== decode-only window (first 123 ms skipped)" \
	--has "bundle: $MST_DIR/$MST_BASE.trace" --has "table:  $MST_DIR/$MST_BASE.gpu-intervals.xml" \
	--has "csv:    $MST_DIR/$MST_BASE.channels.csv" --has "stub summary")"
want "instrument argv" "$(logged xctrace.argv)" "xctrace record --template Metal System Trace --no-prompt --output $MST_DIR/$MST_BASE.trace --time-limit 12s --env RMLX_HOME=.rmlx --launch -- $MST_RUN --kv-quant none --prompt-tokens 4096 --max-tokens 400 --max-ctx 5008"
want "summaries" "$(logged cargo.argv)" "run -q --release -p rmlx-mlx --features metal-capture --example gpu_timeline -- --input $MST_DIR/$MST_BASE.gpu-intervals.xml --process rmlx --skip-ms 0
run -q --release -p rmlx-mlx --features metal-capture --example gpu_timeline -- --input $MST_DIR/$MST_BASE.gpu-intervals.xml --process rmlx --skip-ms 123 --csv $MST_DIR/$MST_BASE.channels.csv"
want "files" "$(listing "$ROOT/$MST_DIR")" "$MST_BASE.gpu-intervals.xml $MST_BASE.trace "
want "started with the capture variable" "$(env_set)" ""
st_verdict mst_capture_default_run "$why" "the release build is launched under xctrace with the window defaults, exported and summarised twice, nothing started with the capture variable"

make_root mst_whole_args
MODEL_SP="$ST_WORK/models/host m"
mkdir -p "$MODEL_SP"
run_mst --model "$MODEL_SP"
why="$(st_check 0)"
SP_BASE="host m-none-4096tok-STAMP"
want "template" "$(arg_after xctrace.argv --template)" "Metal System Trace"
want "bundle" "$(arg_after xctrace.argv --output)" "$MST_DIR/$SP_BASE.trace"
want "model" "$(arg_after launched.argv --model)" "WORK/models/host m"
want "summary input" "$(arg_after cargo.argv --input)" "$MST_DIR/$SP_BASE.gpu-intervals.xml
$MST_DIR/$SP_BASE.gpu-intervals.xml"
st_verdict mst_capture_hands_each_argument_to_the_instrument_whole "$why" "the template name, a model path with a space and the bundle and table paths each reach the instrument and the summariser as one argument"

make_root mst_records
run_mst --model "$MODEL"
why="$(st_check 0)"
want "launched argv" "$(logged launched.argv | grep -c -- 'baseline --record --run-class diagnostic --label mst_capture:STAMP' || true)" 1
st_verdict mst_capture_records_diagnostic "$why" "the traced run is launched recording as diagnostic"

make_root mst_pending
echo pending >"$CTL/mode"
CAP_HOME="$ST_WORK/mst_home"
run_mst --model "$MODEL" --out-dir "$ST_WORK/mst_home.traces"
why="$(st_check 75 --has "record pending: WORK/mst_home/metrics/buffer/pending/this.json" --has "bundle: " \
	--has "skip:       123 ms" --has "INCOMPLETE: the trace stands, its diagnostic record is pending: WORK/mst_home/metrics/buffer/pending/this.json")"
want "child handed" "$(logged xctrace.argv | grep -c -- '--env RMLX_HOME=WORK/mst_home --launch' || true)" 1
st_verdict mst_capture_with_a_pending_record_keeps_its_trace "$why" "a traced run whose record did not land is found by its label, past a newer log of another process, and ends 75"

make_root mst_other
echo other >"$CTL/mode"
CAP_HOME="$ST_WORK/mst_home_other"
run_mst --model "$MODEL" --out-dir "$ST_WORK/mst_home_other.traces"
why="$(st_check 0 --lacks "record pending")"
st_verdict mst_capture_ignores_another_runs_pending_record "$why" "a traced run whose record landed is not charged with another process's pending record"

make_root mst_pending_export
echo pending >"$CTL/mode"
: >"$CTL/no_export"
CAP_HOME="$ST_WORK/mst_home_noexport"
run_mst --model "$MODEL" --out-dir "$ST_WORK/mst_home_noexport.traces"
why="$(st_check 1 --has "record pending: WORK/mst_home_noexport/metrics/buffer/pending/this.json")"
st_verdict mst_capture_names_a_pending_record_when_the_export_fails "$why" "a traced run whose export failed still names its pending buffer file"

make_root mst_home
CAP_HOME="$ST_WORK/mst_home_dir"
run_mst --model "$MODEL"
why="$(st_check 0)"
want "bundles" "$(listing "$ST_WORK/mst_home_dir/traces/mst")" "$MST_BASE.gpu-intervals.xml $MST_BASE.trace "
want "default directory" "$(listing "$ROOT/.rmlx")" ""
st_verdict mst_capture_default_directory_is_under_rmlx_home "$why" "the bundle is filed under <RMLX_HOME>/traces/mst, not under the checkout"

make_root mst_flags
run_mst --model "$MODEL" --kv-quant k8v8 --prompt-tokens 8192 --max-tokens 600 --time-limit 18 --skip-ms 77
why="$(st_check 0 --has "recording:  model=host__m codec=k8v8 prompt=8192 gen=600 limit=18s" --lacks "skip:       " \
	--has "== decode-only window (first 77 ms skipped)")"
want "instrument argv" "$(logged xctrace.argv)" "xctrace record --template Metal System Trace --no-prompt --output $MST_DIR/host__m-k8v8-8192tok-STAMP.trace --time-limit 18s --env RMLX_HOME=.rmlx --launch -- $MST_RUN --kv-quant k8v8 --prompt-tokens 8192 --max-tokens 600 --max-ctx 9304"
want "decode window" "$(logged cargo.argv | sed -n 2p | sed 's/.*--skip-ms/--skip-ms/')" "--skip-ms 77 --csv $MST_DIR/host__m-k8v8-8192tok-STAMP.channels.csv"
st_verdict mst_capture_flags_reach_the_engine_and_the_instrument "$why" "--kv-quant, --prompt-tokens, --max-tokens, --time-limit and --skip-ms reach the launch and the summary"

make_root mst_skip_zero
run_mst --model "$MODEL" --skip-ms 0
why="$(st_check 0 --lacks "== decode-only window")"
want "summaries" "$(logged cargo.argv | sed 's/.*--skip-ms/--skip-ms/')" "--skip-ms 0
--skip-ms 0 --csv $MST_DIR/$MST_BASE.channels.csv"
st_verdict mst_capture_skip_zero_prints_one_window "$why" "--skip-ms 0 prints the full window only and still writes the channel file"

make_root mst_no_log
: >"$CTL/no_log"
run_mst --model "$MODEL"
why="$(st_check 0 --has "WARNING: no generate: prefill event in <no log>; the" --has "summary below INCLUDES prefill")"
want "summaries" "$(logged cargo.argv | sed 's/.*--skip-ms/--skip-ms/')" "--skip-ms 0
--skip-ms 0 --csv $MST_DIR/$MST_BASE.channels.csv"
st_verdict mst_capture_warns_when_the_prefill_cannot_be_read "$why" "a run with no log of its own is summarised whole, and says so"

make_root mst_binary
cp "$ST_WORK/rmlx_stub" "$ST_WORK/custom_rmlx"
run_mst --model "$MODEL" --binary "$ST_WORK/custom_rmlx"
why="$(st_check 0 --has "running:    WORK/custom_rmlx baseline --record")"
want "summary process" "$(logged cargo.argv | sed -n 1p | sed 's/.*--process/--process/')" "--process custom_rmlx --skip-ms 0"
st_verdict mst_capture_binary "$why" "--binary is launched, probed for the claim and named as the process to summarise"

make_root mst_outdir
run_mst --model "$MODEL" --out-dir "$ST_WORK/out_m"
why="$(st_check 0 --has "bundle: WORK/out_m/$MST_BASE.trace")"
want "files" "$(listing "$ST_WORK/out_m")" "$MST_BASE.gpu-intervals.xml $MST_BASE.trace "
want "default directory" "$(listing "$ROOT/.rmlx/traces")" ""
st_verdict mst_capture_out_dir "$why" "--out-dir holds the bundle and its table"

make_root mst_retention
plant "$ROOT/$MST_DIR" trace 7
run_mst --model "$MODEL"
why="$(st_check 0 --has "retention: removed old1.trace" --has "beyond the newest 5" --lacks "removed old4.trace")"
want "kept" "$(listing "$ROOT/$MST_DIR" | tr ' ' '\n' | grep -c '\.trace$' || true)" 5
want "sidecars of a removed bundle" "$(ls -d "$ROOT/$MST_DIR"/old1.* "$ROOT/$MST_DIR"/old3.* 2>/dev/null | wc -l | tr -d ' ')" 0
want "sidecars of a kept bundle" "$(ls -d "$ROOT/$MST_DIR"/old7.* | wc -l | tr -d ' ')" 3
st_verdict mst_capture_retention_keeps_the_newest_five "$why" "eight bundles are cut to the newest five, the new one among them, and the removed bundles' table and channel files go too"

make_root mst_keep
plant "$ROOT/$MST_DIR" trace 7
run_mst --model "$MODEL" --keep 3
why="$(st_check 0 --has "beyond the newest 3)" --lacks "removed old6.trace")"
want "kept" "$(listing "$ROOT/$MST_DIR" | tr ' ' '\n' | grep '\.trace$' | sort | tr '\n' ' ')" "$MST_BASE.trace old6.trace old7.trace "
st_verdict mst_capture_keep_bounds_the_bundles "$why" "--keep 3 leaves the new bundle and the two newest before it"

make_root mst_keep_zero
plant "$ROOT/$MST_DIR" trace 3
run_mst --model "$MODEL" --keep 0
why="$(st_check 0 --has "removed old3.trace")"
want "kept" "$(listing "$ROOT/$MST_DIR" | tr ' ' '\n' | grep '\.trace$' | tr '\n' ' ')" "$MST_BASE.trace "
st_verdict mst_capture_keep_zero_spares_the_new_bundle "$why" "--keep 0 removes every earlier bundle and never the one just written"

make_root mst_retention_on_failure
plant "$ROOT/$MST_DIR" trace 7
: >"$CTL/no_export"
run_mst --model "$MODEL"
why="$(st_check 1 --has "retention: removed old3.trace" --lacks "removed old4.trace")"
st_verdict mst_capture_retention_runs_when_the_export_fails "$why" "a run that fails after its bundle still bounds the directory, counting that bundle"

make_root mst_retention_no_bundle
plant "$ROOT/$MST_DIR" trace 7
run_mst --model "$ST_WORK/models/absent"
why="$(st_check 1 --has "model snapshot not found: WORK/models/absent" --has "removed old2.trace" --lacks "removed old3.trace")"
st_verdict mst_capture_retention_runs_before_a_bundle_exists "$why" "a run that fails before it writes a bundle keeps the full count"

make_root mst_unknown
run_mst --model "$MODEL" --bogus
why="$(st_check 2 --has "unknown argument" --has "usage:")"
st_verdict mst_capture_refuses_an_unknown_argument "$why" "an argument the System Trace does not take ends 2 with the usage"

make_root mst_no_kv_flag_skip
run_mst --model "$MODEL" --skip 4
why="$(st_check 2 --has "unknown argument")"
st_verdict mst_capture_does_not_take_the_capture_window_flags "$why" "--skip belongs to the capture, not to the System Trace"

make_root mst_help
run_mst --help
why="$(st_check 0 --has "usage:" --has "--skip-ms N")"
st_verdict mst_capture_help "$why" "--help prints the usage and ends 0"

make_root mst_no_model
run_mst
why="$(st_check 2 --has "usage:")"
st_verdict mst_capture_refuses_a_missing_model_argument "$why" "no --model ends 2 with the usage"

for flag in --time-limit --max-tokens --keep --prompt-tokens --skip-ms; do
	make_root "mst_nan${flag#--}"
	run_mst --model "$MODEL" "$flag" abc
	why="$(st_check 2 --has "ERROR: $flag must be a non-negative integer, got 'abc'")"
	st_verdict "mst_capture_refuses_a_non_number_for_${flag#--}" "$why" "$flag abc ends 2 before anything runs"
done

make_root mst_prompt_size
run_mst --model "$MODEL" --prompt-tokens 1000
why="$(st_check 2 --has "ERROR: --prompt-tokens 1000 has no prompt fixture." --has "Valid: 4096, 8192, 16384, 32768, 65536, 131072")"
st_verdict mst_capture_refuses_a_prompt_size_without_a_fixture "$why" "a size with no checked-in prompt ends 2 naming the valid ones"

make_root mst_no_bin
rm -f "$ROOT/target/release/rmlx"
run_mst --model "$MODEL"
why="$(st_check 1 --has "target/release/rmlx not found" --has "make build")"
st_verdict mst_capture_refuses_a_missing_binary "$why" "no release build ends 1 naming make build"

make_root mst_no_model_dir
run_mst --model "$ST_WORK/models/absent"
why="$(st_check 1 --has "ERROR: model snapshot not found: WORK/models/absent")"
st_verdict mst_capture_refuses_a_missing_snapshot "$why" "a snapshot directory that does not exist ends 1"

make_root mst_no_xctrace
: >"$CTL/no_xctrace"
run_mst --model "$MODEL"
why="$(st_check 1 --has "ERROR: xctrace not available. Select full Xcode:")"
want "instrument runs" "$(logged xctrace.argv)" ""
st_verdict mst_capture_refuses_a_host_without_xctrace "$why" "neither xctrace on the path nor xcrun finding it ends 1"

make_root mst_claim
: >"$CTL/claim_held"
run_mst --model "$MODEL"
why="$(st_check 1 --has "ERROR: another process holds the Metal claim" --has "Stop it first, by the PID the refusal above names.")"
want "instrument runs" "$(logged xctrace.argv)" ""
st_verdict mst_capture_refuses_while_the_metal_claim_is_held "$why" "a held claim ends 1 and starts no recording"

make_root mst_no_bundle
: >"$CTL/no_bundle"
run_mst --model "$MODEL"
why="$(st_check 1 --has "ERROR: no bundle at $MST_DIR/$MST_BASE.trace — the recording did not start.")"
st_verdict mst_capture_refuses_a_recording_that_left_no_bundle "$why" "an instrument that wrote nothing ends 1"

make_root mst_record_rc
echo 2 >"$CTL/record_rc"
run_mst --model "$MODEL"
why="$(st_check 0 --has "note: xctrace record exited 2 — expected when --time-limit ends a")"
st_verdict mst_capture_does_not_gate_on_the_instruments_exit_code "$why" "a non-zero xctrace exit with a bundle and a table is a note, not a failure"

make_root mst_export
: >"$CTL/no_export"
run_mst --model "$MODEL"
why="$(st_check 1 --has "ERROR: export produced no table." --has "which is what --attach produces")"
st_verdict mst_capture_refuses_an_export_without_a_table "$why" "an export that wrote no table ends 1"

make_root mst_export_rc
echo 4 >"$CTL/export_rc"
run_mst --model "$MODEL"
why="$(st_check 1 --has "ERROR: export produced no table.")"
st_verdict mst_capture_refuses_a_failed_export "$why" "an export that exits non-zero ends 1 even with a table"

make_root mst_summary
echo 7 >"$CTL/cargo_rc"
run_mst --model "$MODEL"
why="$(st_check 7 --has "ERROR: summarising $MST_DIR/$MST_BASE.gpu-intervals.xml failed (exit 7)" --has "'contains no rows'" \
	--has "target/release/rmlx baseline --record")"
st_verdict mst_capture_ends_with_the_summarisers_exit_code "$why" "a summary the parser refused ends with its code and says what each refusal means"

# ===== preflight =====

make_root pf_pass
run_preflight
why="$(st_check 0 --has "gputrace preflight (binary: target/release-debug/rmlx)" \
	--has "ok    Xcode selected: /Applications/Xcode.app/Contents/Developer" --has "ok    developer mode enabled" \
	--has "ok    get-task-allow entitlement present" --has "ok    Metal toolchain present: Apple metal version" \
	--has "preflight ok — Apple's GPU tools can attach to this binary on this host" --lacks "FAIL")"
st_verdict preflight_passes_on_a_ready_host "$why" "a host with Xcode, developer mode and an entitled binary ends 0"

make_root pf_probes
run_preflight
why="$(st_check 0)"
want "xcode-select" "$(logged xcode-select.argv)" "-p"
want "DevToolsSecurity" "$(logged DevToolsSecurity.argv)" "-status"
want "codesign" "$(logged codesign.argv)" "-d --entitlements - --xml target/release-debug/rmlx"
want "plutil key" "$(arg_after plutil.argv -extract)" 'com\.apple\.security\.get-task-allow'
want "started with the capture variable" "$(env_set)" ""
st_verdict preflight_points_each_probe_at_what_it_checks "$why" "each host probe is given the arguments that read what the preflight reports, the entitlement key with its dots escaped"

make_root pf_binary
cp "$ST_WORK/rmlx_stub" "$ST_WORK/other_rmlx"
run_preflight --binary "$ST_WORK/other_rmlx"
why="$(st_check 0 --has "gputrace preflight (binary: WORK/other_rmlx)")"
want "codesign" "$(logged codesign.argv)" "-d --entitlements - --xml WORK/other_rmlx"
st_verdict preflight_binary "$why" "--binary names the binary that is checked, and codesign reads that one"

make_root pf_binary_space
mkdir -p "$ST_WORK/bin dir"
cp "$ST_WORK/rmlx_stub" "$ST_WORK/bin dir/rmlx"
run_preflight --binary "$ST_WORK/bin dir/rmlx"
why="$(st_check 0)"
want "codesign target" "$(arg_after codesign.argv --xml)" "WORK/bin dir/rmlx"
st_verdict preflight_hands_a_binary_path_with_a_space_to_codesign_whole "$why" "a binary path with a space reaches codesign as one argument"

make_root pf_no_metal
: >"$CTL/no_metal"
run_preflight
why="$(st_check 0 --has "warn  Metal toolchain not installed" --has "xcodebuild -downloadComponent MetalToolchain" --has "preflight ok")"
st_verdict preflight_warns_without_the_metal_toolchain "$why" "a missing Metal toolchain is a warning and the host still passes"

make_root pf_no_xcode
: >"$CTL/xcode_dir"
run_preflight
why="$(st_check 1 --has "FAIL  xcode-select has no developer directory selected." --has "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" --has "preflight FAILED")"
st_verdict preflight_fails_with_no_xcode_selected "$why" "no developer directory ends 1 naming the fix"

make_root pf_clt
echo /Library/Developer/CommandLineTools >"$CTL/xcode_dir"
run_preflight
why="$(st_check 1 --has "FAIL  xcode-select points at Command Line Tools, not Xcode (/Library/Developer/CommandLineTools)." --has "preflight FAILED")"
st_verdict preflight_fails_with_command_line_tools "$why" "Command Line Tools instead of Xcode ends 1"

make_root pf_dev_off
echo "Developer mode is currently disabled." >"$CTL/dts"
run_preflight
why="$(st_check 1 --has "FAIL  developer mode is disabled" --has "sudo DevToolsSecurity -enable" --has "preflight FAILED")"
st_verdict preflight_fails_with_developer_mode_off "$why" "developer mode off ends 1 naming the fix"

make_root pf_dev_unreadable
echo "dyld: library not loaded" >"$CTL/dts"
run_preflight
why="$(st_check 1 --has "FAIL  could not read developer-mode status; DevToolsSecurity said: dyld: library not loaded" --has "preflight FAILED")"
st_verdict preflight_fails_when_developer_mode_is_unreadable "$why" "an output it cannot read is not taken for enabled"

make_root pf_no_bin
rm -f "$ROOT/target/release-debug/rmlx"
run_preflight
why="$(st_check 1 --has "FAIL  capture binary not found at target/release-debug/rmlx." --has "make build-capture" --has "preflight FAILED")"
st_verdict preflight_fails_without_the_binary "$why" "a missing binary ends 1 naming make build-capture"

make_root pf_codesign
echo 1 >"$CTL/codesign_rc"
run_preflight
why="$(st_check 1 --has "FAIL  codesign could not read target/release-debug/rmlx (exit 1)" --has "preflight FAILED")"
st_verdict preflight_fails_when_codesign_cannot_read_the_binary "$why" "a binary codesign refuses ends 1"

make_root pf_unsigned
: >"$CTL/entitlements"
run_preflight
why="$(st_check 1 --has "FAIL  target/release-debug/rmlx carries no entitlements at all" --has "preflight FAILED")"
hint="$(printf '%s\n' "$ST_OUT" | sed -n 's|.*--entitlements \(scripts/[^ ]*\) target/release-debug/rmlx.*|\1|p')"
[ -f "$ROOT/$hint" ] || why="${why:+$why; }the re-sign hint names a file that does not exist: [$hint]"
st_verdict preflight_fails_with_no_entitlements "$why" "a linker-signed binary ends 1, and the re-sign hint names the entitlement file"

make_root pf_blob
echo "not a plist" >"$CTL/entitlements"
run_preflight
why="$(st_check 1 --has "FAIL  codesign printed an entitlement blob that is not a readable plist." --has "preflight FAILED")"
st_verdict preflight_fails_on_an_unreadable_entitlement_blob "$why" "a blob it cannot read is neither entitled nor not"

make_root pf_false
echo '<plist version="1.0"><dict><key>com.apple.security.get-task-allow</key><false/></dict></plist>' >"$CTL/entitlements"
run_preflight
why="$(st_check 1 --has "FAIL  target/release-debug/rmlx has com.apple.security.get-task-allow set to 'false', not true." --has "preflight FAILED")"
st_verdict preflight_fails_when_get_task_allow_is_false "$why" "an entitlement set to false ends 1"

make_root pf_absent
echo '<plist version="1.0"><dict><key>other</key><true/></dict></plist>' >"$CTL/entitlements"
run_preflight
why="$(st_check 1 --has "FAIL  target/release-debug/rmlx carries no com.apple.security.get-task-allow entitlement." --has "preflight FAILED")"
st_verdict preflight_fails_without_the_get_task_allow_key "$why" "a binary entitled for something else ends 1"

make_root pf_several
echo /Library/Developer/CommandLineTools >"$CTL/xcode_dir"
echo "Developer mode is currently disabled." >"$CTL/dts"
run_preflight
why="$(st_check 1 --has "FAIL  xcode-select points at Command Line Tools" --has "FAIL  developer mode is disabled" --has "ok    get-task-allow entitlement present")"
st_verdict preflight_reports_every_failure "$why" "two unmet prerequisites are both reported, and the one that holds still shows"

make_root pf_unknown
run_preflight --bogus
why="$(st_check 2 --has "unknown argument")"
st_verdict preflight_refuses_an_unknown_argument "$why" "an argument the preflight does not take ends 2"

make_root pf_no_value
run_preflight --binary
why="$(st_check 2 --has "--binary needs a")"
st_verdict preflight_refuses_a_binary_flag_without_a_path "$why" "--binary last, with no path, ends 2"

make_root pf_help
run_preflight --help
why="$(st_check 0 --has "usage:" --has "--binary")"
st_verdict preflight_help "$why" "--help prints the usage and ends 0"

# ===== a flag given an empty value =====

for flag in --kv-quant --model --prompt-tokens --skip --steps --gen --out-dir; do
	make_root "gp_empty${flag#--}"
	run_gputrace --kv-quant none --model "$MODEL" "$flag" ""
	why="$(st_check 2 --has "ERROR: $flag needs a value")"
	want "engine runs" "$(logged rmlx.argv)" ""
	want "directory" "$(listing "$ROOT/.rmlx")" ""
	st_verdict "gpu_capture_refuses_an_empty_${flag#--}" "$why" "$flag \"\" ends 2 before anything runs"
done

for flag in --model --kv-quant --prompt-tokens --max-tokens --time-limit --skip-ms --binary --keep --out-dir; do
	make_root "mst_empty${flag#--}"
	run_mst --model "$MODEL" "$flag" ""
	why="$(st_check 2 --has "ERROR: $flag needs a value")"
	want "instrument runs" "$(logged xctrace.argv)" ""
	want "directory" "$(listing "$ROOT/.rmlx")" ""
	st_verdict "mst_capture_refuses_an_empty_${flag#--}" "$why" "$flag \"\" ends 2 before anything runs"
done

make_root pf_empty_binary
run_preflight --binary ""
why="$(st_check 2 --has "ERROR: --binary needs a value")"
want "probes run" "$(logged codesign.argv)" ""
st_verdict preflight_refuses_an_empty_binary "$why" "--binary \"\" ends 2 before anything is probed"

make_root dispatch_none
run_raw() {
	st_run env -i PATH="$SHIMS:$SYSBIN" HOME="$ST_WORK" CAP_CTL="$CTL" bash "$ROOT/scripts/profile/capture.sh" "$@"
	settle
}
run_raw
why="$(st_check 2 --has "usage:")"
st_verdict capture_refuses_a_missing_subcommand "$why" "no subcommand ends 2 with the usage"
run_raw frobnicate --model "$MODEL"
why="$(st_check 2 --has "usage:")"
st_verdict capture_refuses_an_unknown_subcommand "$why" "a subcommand that is not gputrace, mst or preflight ends 2 with the usage"

st_done
