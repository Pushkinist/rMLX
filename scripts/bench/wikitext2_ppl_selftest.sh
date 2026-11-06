#!/usr/bin/env bash
# Recall test for the perplexity harness: a pending record keeps its reading, a failed run ends 1
#
# Recall test for: bench/wikitext2_ppl.py
#
# Each case drives the tool over a cached text and a stub `rmlx` that prints a perplexity line and, when
# told to, exits 75 naming a buffer file, as the real binary does when the record did not reach the metrics
# DB. The cases assert the exit code, that the number is kept on stdout and that the buffer file is named
# on stderr. No download, model or GPU is involved.
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

st_done
