#!/usr/bin/env bash
# Recall test for the cross-backend cell tool: the codec name it files, where its files go, its exits
#
# Recall test for: bench/cross_backend.sh
#
# Each case drives the tool in a fake checkout against a stub `rmlx` (its `serve` writes a run log,
# its `claim run` stands in for every other backend's server), a stub `curl` and a stub `uv` that
# stands for the Cross-Backend-Bench harness and keeps its argv. No backend, model or harness runs.
# The cases assert the `--quant` the harness is handed (the codec name a cell is filed under), that
# the progress line and the server, harness and oMLX models files sit under RMLX_HOME, and the
# exit code and message of each refusal.
#
# Usage:  bash scripts/bench/cross_backend_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init cross-backend-selftest

# make_stubs <fake root> — the stub binaries a cell runs. Controls are files under $RMLX_HOME:
#   stated_codec  the codec name the stub server's run log states; absent, the log states none
#   decoy_codec   a second run log, written by another pid, that states this codec
#   claim_held    `rmlx serve` exits 11 at once
make_stubs() {
	local root="$1"
	mkdir -p "$root/target/release" "$root/stubbin" "$root/cbb"
	cat >"$root/target/release/rmlx" <<'STUB'
#!/usr/bin/env bash
set -u
echo "$*" >>"$RMLX_HOME/rmlx_argv"
case "$1" in
metrics)
	echo '{"backend":"rmlx","backend_version":"9.9.9","git_sha":"abc1234"}'
	;;
serve)
	[ ! -e "$RMLX_HOME/claim_held" ] || exit 11
	mkdir -p "$RMLX_HOME/logs"
	if [ -e "$RMLX_HOME/decoy_codec" ]; then
		printf '%s\n%s\n' \
			'{"fields":{"message":"rmlx start","pid":99999}}' \
			"{\"fields\":{\"message\":\"cache-type resolved\",\"kv_quant\":\"$(cat "$RMLX_HOME/decoy_codec")\"}}" \
			>>"$RMLX_HOME/logs/$(date +%s)-99999.jsonl"
	fi
	log="$RMLX_HOME/logs/$(date +%s)-$$.jsonl"
	printf '{"fields":{"message":"rmlx start","pid":%s}}\n' "$$" >>"$log"
	if [ -e "$RMLX_HOME/stated_codec" ]; then
		printf '{"fields":{"message":"cache-type resolved","kv_quant":"%s"}}\n' "$(cat "$RMLX_HOME/stated_codec")" >>"$log"
	fi
	exec /bin/sleep 30
	;;
claim)
	/bin/sleep 30 &
	child=$!
	trap 'kill "$child"; exit 143' TERM
	wait "$child"
	;;
esac
STUB
	cat >"$root/stubbin/uv" <<'STUB'
#!/usr/bin/env bash
for a in "$@"; do echo "argv=$a"; done >"$STUB_OUT"
echo "[1/2] ttft=100.0ms decode=50.00tps"
echo "[2/2] ttft=10.0ms decode=60.00tps"
STUB
	printf '#!/bin/sh\nexit 0\n' >"$root/stubbin/curl"
	printf '#!/bin/sh\nexec /bin/sleep 0.3\n' >"$root/stubbin/sleep"
	chmod +x "$root/target/release/rmlx" "$root/stubbin/uv" "$root/stubbin/curl" "$root/stubbin/sleep"
}

FAKE="$ST_WORK/repo"
mkdir -p "$FAKE" || st_die "cannot build a fixture: $FAKE"
ln -s "$REPO_ROOT/scripts" "$FAKE/scripts"
make_stubs "$FAKE"

new_home() { # <name> — a fresh $RMLX_HOME
	local home="$ST_WORK/home_$1"
	mkdir -p "$home"
	printf '%s' "$home"
}

# run_cell <home> <tag> <backend> <kv-quant> — one cell of the tool, alone in an empty environment.
run_cell() {
	local home="$1" tag="$2"
	shift 2
	st_run env -i PATH="$FAKE/stubbin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$ST_WORK" \
		RMLX_HOME="$home" CROSS_BENCH_ROOT="$FAKE/cbb" STUB_OUT="$home/harness_argv" \
		LLAMA_CPP_ROOT="$ST_WORK/no_llama" \
		bash "$FAKE/scripts/bench/cross_backend.sh" "$tag" "$ST_WORK/models/m-$tag" "$@" 5 3 mxfp8
}

# filed_quant <home> — the `--quant` the harness was handed: the name the cell is filed under.
filed_quant() {
	awk '$0 == "argv=--quant" { getline; sub(/^argv=/, ""); print; exit }' "$1/harness_argv" 2>/dev/null || true
}

why=""
want() { # <what> <got> <expected> — note a difference
	[ "$2" = "$3" ] || why="${why:+$why; }$1: got [$2], want [$3]"
}

# --- the codec name an rmlx cell is filed under ---

home="$(new_home rmlx_mixed)"
printf '%s' mixed_k8g64_v4g64 >"$home/stated_codec"
run_cell "$home" e2b rmlx mixed
why="$(st_check 0)"
want "filed as" "$(filed_quant "$home")" "mxfp8 / mixed_k8g64_v4g64"
want "server started with" "$(grep -c -- '--kv-quant mixed' "$home/rmlx_argv")" 1
st_verdict rmlx_mixed_files_the_name_its_log_states "$why" "an rmlx cell run with --kv-quant mixed is filed under the codec its run log states, not k8v4"

home="$(new_home rmlx_k8v4)"
printf '%s' k8v4 >"$home/stated_codec"
run_cell "$home" e2b rmlx k8v4
why="$(st_check 0)"
want "filed as" "$(filed_quant "$home")" "mxfp8 / k8v4"
st_verdict rmlx_k8v4_files_the_name_its_log_states "$why" "an rmlx cell run with --kv-quant k8v4 is filed under k8v4"

home="$(new_home rmlx_auto)"
printf '%s' k8v8 >"$home/stated_codec"
run_cell "$home" e2b rmlx auto
why="$(st_check 0)"
want "filed as" "$(filed_quant "$home")" "mxfp8 / k8v8"
st_verdict rmlx_auto_files_the_resolved_name "$why" "an rmlx cell run with --kv-quant auto is filed under the codec the server resolved"

home="$(new_home rmlx_decoy)"
printf '%s' mixed_k8g64_v4g64 >"$home/stated_codec"
printf '%s' k8v8 >"$home/decoy_codec"
run_cell "$home" e2b rmlx mixed
why="$(st_check 0)"
want "filed as" "$(filed_quant "$home")" "mxfp8 / mixed_k8g64_v4g64"
st_verdict rmlx_other_process_log_is_not_read "$why" "a run log another process wrote during the cell does not name the cell's codec"

home="$(new_home rmlx_no_codec)"
run_cell "$home" e2b rmlx mixed
why="$(st_check 4 --has "ERROR: cell e2b rmlx mixed")"
want "harness ran" "$(ls "$home/harness_argv" 2>/dev/null || true)" ""
want "progress line" "$(grep -c 'e2b | rmlx | mixed | success=false | coh=false | NO_KV_NAME' "$home/bench/cross_backend_progress.md")" 1
st_verdict rmlx_log_states_no_codec "$why" "a server whose run log states no codec files nothing for the cell and the tool ends 4 naming the cell"

# --- where the files go ---

home="$(new_home paths_rmlx)"
printf '%s' k8v4 >"$home/stated_codec"
run_cell "$home" e2b rmlx k8v4
why="$(st_check 0)"
want "progress line" "$(grep -c 'e2b | rmlx | k8v4 | success=true' "$home/bench/cross_backend_progress.md")" 1
want "server log" "$(ls "$home/tmp/bench_e2b_rmlx_k8v4.log" 2>/dev/null || true)" "$home/tmp/bench_e2b_rmlx_k8v4.log"
want "harness log" "$(grep -c 'decode=60.00tps' "$home/tmp/bench_harness_e2b_rmlx_k8v4.log" 2>/dev/null || true)" 1
st_verdict progress_and_logs_under_rmlx_home "$why" "the progress line, the server log and the harness log are all under RMLX_HOME"

home="$(new_home paths_omlx)"
mkdir -p "$home/tmp/omlx_models"
: >"$home/tmp/omlx_models/stale"
run_cell "$home" e2b omlx none
why="$(st_check 0)"
want "omlx models dir" "$(grep -c -- "--model-dir $home/tmp/omlx_models " "$home/rmlx_argv")" 1
want "model link" "$(readlink "$home/tmp/omlx_models/gemma4-e2b" 2>/dev/null || true)" "$ST_WORK/models/m-e2b"
want "stale entry" "$(ls "$home/tmp/omlx_models/stale" 2>/dev/null || true)" ""
st_verdict omlx_models_dir_under_rmlx_home "$why" "the oMLX models directory is under RMLX_HOME and the launch clears it"

# --- the other backends keep the alias table ---

home="$(new_home alias_turboquant)"
run_cell "$home" e2b mlx-lm-turboquant turboquant
why="$(st_check 0)"
want "filed as" "$(filed_quant "$home")" "mxfp8 / turbo4"
st_verdict mlx_lm_turboquant_alias_files_turbo4 "$why" "mlx-lm-turboquant with turboquant is filed as turbo4"

home="$(new_home alias_native)"
run_cell "$home" e2b paroquant native
why="$(st_check 0)"
want "filed as" "$(filed_quant "$home")" "mxfp8 / none"
st_verdict paroquant_alias_files_none "$why" "paroquant with native is filed as none"

# --- refusals ---

home="$(new_home exit_unknown_backend)"
run_cell "$home" e2b nope none
why="$(st_check 1 --has "unknown backend nope")"
want "progress line" "$(grep -c 'UNKNOWN_BACKEND' "$home/bench/cross_backend_progress.md")" 1
st_verdict exit_1_unknown_backend "$why" "an unknown backend ends 1 and says so"

home="$(new_home exit_server_not_ready)"
: >"$home/claim_held"
run_cell "$home" e2b rmlx k8v4
why="$(st_check 2 --has "CELL FAIL server-not-ready" --has "server_exit=11")"
want "harness ran" "$(ls "$home/harness_argv" 2>/dev/null || true)" ""
st_verdict exit_2_server_not_ready "$why" "a server that exits first ends the cell 2 and the message carries the server's 11"

home="$(new_home exit_no_llama_server)"
run_cell "$home" e2b llama.cpp f16
why="$(st_check 5 --has "CELL FAIL no-llama-server")"
st_verdict exit_5_no_llama_server "$why" "a llama.cpp cell with no llama-server binary ends 5"

home="$(new_home exit_mlx_lm_quantised)"
run_cell "$home" e2b mlx-lm k8v4
why="$(st_check 6 --has "bare mlx-lm serves unquantised KV only")"
st_verdict exit_6_bare_mlx_lm_quantised "$why" "bare mlx-lm given a quantised KV_QUANT ends 6"

st_done
