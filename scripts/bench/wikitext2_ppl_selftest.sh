#!/usr/bin/env bash
# Recall test for the perplexity harness: a pending record keeps its reading, a failed run ends 1
#
# Recall test for: bench/wikitext2_ppl.py
#
# Each case drives the tool over a cached text and a stub `rmlx` that prints a perplexity line and, when
# told to, exits 75 naming a buffer file, as the real binary does when the record did not reach the metrics
# DB. The cases assert the exit code, that the number is kept on stdout and that the buffer file is named
# on stderr. Two more cases run the tool without --cache-dir: a relative RMLX_HOME is refused before any
# fetch, and an unset one puts the cache under the checkout's .rmlx. No download, model or GPU is involved.
#
# Usage:  bash scripts/bench/wikitext2_ppl_selftest.sh
# Exit:   0 every case held; 1 a case did not, or none ran; 2 a fixture could not be built

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init wikitext2-ppl-selftest

MODEL="$ST_WORK/models/host__m"
mkdir -p "$MODEL" "$ST_WORK/wikitext_cache"
printf '{"torch_dtype": "float16"}\n' >"$MODEL/config.json"
printf 'some text\n' >"$ST_WORK/wikitext_cache/wiki.test.raw.txt"

# A stub `rmlx`: with STUB_PPL_CRASH set it fails without a reading.
cat >"$ST_WORK/rmlx_ppl" <<'STUB'
#!/usr/bin/env bash
[ -n "${STUB_PPL_CRASH:-}" ] && exit 1
echo '{"ppl": 7.5, "scored_tokens": 10, "windows": 1, "ctx_window": 4096, "stride": 2048}'
echo "error: ppl: measured, but the record did not reach runs.db; it is kept at /x/pending/ppl.json for \`rmlx metrics record --replay-pending\`" >&2
echo "rMLX exits with code 75." >&2
exit 75
STUB
chmod +x "$ST_WORK/rmlx_ppl"

ppl_case() { # <crash> — the tool over the cached text, with stdout and stderr apart
	STUB_PPL_CRASH="$1" python3 "$REPO_ROOT/scripts/bench/wikitext2_ppl.py" --model "$MODEL" \
		--run-class diagnostic --rmlx-binary "$ST_WORK/rmlx_ppl" --cache-dir "$ST_WORK/wikitext_cache" \
		>"$ST_WORK/ppl.out" 2>"$ST_WORK/ppl.err"
}

# A PPL run that measured and whose record did not land keeps its number, names the buffer file and
# ends 75; one that failed to measure ends 1.
status=0
ppl_case "" || status=$?
why=""
[ "$status" -eq 75 ] || why="exit=$status (want 75)"
grep -q '"ppl": 7.5' "$ST_WORK/ppl.out" || why="${why:+$why; }the PPL line was not kept"
grep -q 'kept at /x/pending/ppl.json' "$ST_WORK/ppl.err" || why="${why:+$why; }the buffer file is not named"
st_verdict ppl_with_a_pending_record_keeps_its_number "$why" "a PPL whose record did not land is printed, its file named, and the run ends 75"

status=0
ppl_case 1 || status=$?
why=""
[ "$status" -eq 1 ] || why="exit=$status (want 1)"
st_verdict ppl_that_did_not_measure_fails "$why" "a PPL run that failed to measure ends 1"

# Without --cache-dir the cache follows RMLX_HOME the way the binary does. HOME is a scratch directory
# and the proxy points at a closed port, so a run that reached for the old default or the network fails
# on its own and touches neither the real home nor the network.
SCRATCH_REPO="$ST_WORK/repo"
mkdir -p "$SCRATCH_REPO/scripts/bench" "$SCRATCH_REPO/scripts/lib" "$SCRATCH_REPO/.rmlx/cache/wikitext-2" "$ST_WORK/fakehome"
cp "$REPO_ROOT/scripts/bench/wikitext2_ppl.py" "$SCRATCH_REPO/scripts/bench/"
cp "$REPO_ROOT/scripts/lib/rmlx_home.py" "$SCRATCH_REPO/scripts/lib/"
cp "$ST_WORK/wikitext_cache/wiki.test.raw.txt" "$SCRATCH_REPO/.rmlx/cache/wikitext-2/"

# home_case <env arg>...: the scratch checkout's copy of the tool, no --cache-dir, under the environment the arguments give to `env`.
home_case() {
	(cd "$ST_WORK" && env "$@" HOME="$ST_WORK/fakehome" HTTPS_PROXY=http://127.0.0.1:9 https_proxy=http://127.0.0.1:9 \
		python3 "$SCRATCH_REPO/scripts/bench/wikitext2_ppl.py" --model "$MODEL" --run-class diagnostic \
		--rmlx-binary "$ST_WORK/rmlx_ppl")
}

st_case "cache dir: a relative RMLX_HOME is refused before any fetch" 1 \
	--has "RMLX_HOME=rel/home must be an absolute path" --lacks "downloading" -- home_case RMLX_HOME=rel/home
st_case "cache dir: an unset RMLX_HOME is the checkout's .rmlx" 75 \
	--has "cache hit at $(cd "$SCRATCH_REPO" && pwd -P)/.rmlx/cache/wikitext-2/wiki.test.raw.txt" -- home_case -u RMLX_HOME

st_done
