#!/usr/bin/env bash
# perf_ab_ingest_selftest.sh — mutation check for scripts/ingest/perf_ab_ingest.py.
#
# WHY THIS EXISTS
#
# That script is the only writer that turns a throwaway A/B experiment into
# permanent rows in the append-only metrics store. Every guard in it exists to
# stop one specific wrong row, and a wrong row there cannot be taken back out.
# Six refusals asserted only by reading the source are six refusals nobody has
# watched fire.
#
# Each case drives the real script over a synthetic `perf_ab.sh` result file and
# a stub `rmlx` binary, in --dry-run, so nothing here can reach `runs.db`. The
# suite asserts the exit code AND the reason text, because a guard that refuses
# for the wrong reason is a guard that will stop refusing when that reason moves.
#
# Exit codes: 0 — every case behaved; 1 — at least one did not.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/ingest/perf_ab_ingest.py"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_ab_ingest_selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

PASSED=0
FAILED=0

# A stub `rmlx` whose `metrics identity --json` answers like the real one.
STUB="$WORK/rmlx"
cat >"$STUB" <<'STUBEOF'
#!/usr/bin/env bash
echo '{"backend":"rmlx","backend_version":"9.9.9","build_profile":"release-perf","hardware_tag":"m5_max_128gb"}'
STUBEOF
chmod +x "$STUB"
STUB_SHA16="$(shasum -a 256 "$STUB" | cut -c1-16)"

PROMPT="$WORK/prompt.json"
printf '{"messages":[{"role":"user","content":"hi"}]}' >"$PROMPT"

# make_result <name> [overrides-json] -- a minimal but complete perf_ab.sh
# result. Overrides are ONE JSON object of dotted paths, e.g.
# '{"results.0.taint": "busy"}'. A JSON argument rather than key=value pairs
# because several of the values under test contain spaces.
make_result() {
	local name="$1"
	local path="$WORK/$name.json"
	# The default cannot be written inline: a literal `}` inside a `${...}`
	# default closes the expansion.
	local overrides="${2:-}"
	[[ -z "$overrides" ]] && overrides='{}'
	STUB="$STUB" STUB_SHA16="$STUB_SHA16" OUT="$path" OVERRIDES="$overrides" python3 - <<'PY'
import json, os
stub, sha, out = os.environ["STUB"], os.environ["STUB_SHA16"], os.environ["OUT"]
arm = lambda label, args: {"label": label, "binary": stub, "sha256_16": sha, "args": args}
cell_arm = lambda: {"median_tps": 100.0, "sd_tps": 1.0, "min_tps": 99.0,
                    "max_tps": 101.0, "n": 4, "median_gen_alloc_mb": 40.0,
                    "median_kv_cache_bytes": 1000000123, "tps": [99.0, 100.0, 101.0, 100.0]}
res = {
    "ts_utc": "20260820T090151Z",
    "pattern": "ABBA BAAB",
    "slots_per_model": 8,
    "shape": {"prompt_tokens": 4096, "max_tokens": 100, "max_ctx": 8192},
    "arm_a": arm("none", "--kv-quant none"),
    "arm_b": arm("mixed", "--kv-quant mixed_k8g64_v4g64"),
    "waivers": {"null_arms": False, "busy_host": False,
                "token_divergence": False, "busy_pct_raised": False},
    "results": [{"model": "snap", "prompt_tokens": 3770,
                 "arm_a": cell_arm(), "arm_b": cell_arm(),
                 "ratio_b_over_a": 1.0, "kv_cache_ratio_b_over_a": 1.0,
                 "verdict": "INCONCLUSIVE", "tokens": "identical",
                 "tokens_per_run": 100, "taint": ""}],
}
for dotted, val in json.loads(os.environ["OVERRIDES"]).items():
    parts = dotted.split(".")
    tgt = res
    for part in parts[:-1]:
        tgt = tgt[int(part)] if part.isdigit() else tgt[part]
    tgt[parts[-1]] = val
open(out, "w").write(json.dumps(res))
PY
	echo "$path"
}

# check <name> <want-exit> <what-it-proves> <result-file> [extra args...] [GREP:pat]
check() {
	local name="$1" want="$2" what="$3" result="$4"
	shift 4
	local args=() greps=()
	local a
	for a in "$@"; do
		case "$a" in
		GREP:*) greps+=("${a#GREP:}") ;;
		*) args+=("$a") ;;
		esac
	done
	local out="$WORK/$name.log" got=0
	set +e
	python3 "$SCRIPT" --dry-run "$result" \
		--model snap --model-namespace mlx-community --weight-quant mxfp8 \
		--arm-a-kv-quant none --arm-b-kv-quant mixed_k8g64_v4g64 \
		--rmlx-bin "$STUB" ${args[@]+"${args[@]}"} >"$out" 2>&1
	got=$?
	set -e

	local bad=""
	[[ "$got" -ne "$want" ]] && bad="exit=$got (want $want)"
	local g
	for g in ${greps[@]+"${greps[@]}"}; do
		grep -qE "$g" "$out" || bad="${bad:+$bad; }missing: $g"
	done
	if [[ -z "$bad" ]]; then
		PASSED=$((PASSED + 1))
		printf '  ok   %-38s exit=%s — %s\n' "$name" "$got" "$what"
	else
		FAILED=$((FAILED + 1))
		printf '  FAIL %-38s %s — %s\n' "$name" "$bad" "$what"
		sed 's/^/       | /' "$out" | tail -4
	fi
}

echo "perf_ab_ingest selftest: refusal checks"

CLEAN="$(make_result clean)"
check accepts_a_clean_result 0 \
	"a clean result produces two records" "$CLEAN" \
	"GREP:\"kv_quant\": \"none\"" \
	"GREP:\"kv_quant\": \"mixed_k8g64_v4g64\"" \
	"GREP:\"prompt_tokens\": 3770"

# --- the taint / waiver family ------------------------------------------------

TAINTED="$(make_result tainted '{"results.0.taint": "WindowServer at 60%"}')"
check tainted_refused 2 \
	"a TAINTED run is refused rather than recorded as clean" "$TAINTED" \
	"GREP:refusing: run is TAINTED"

check tainted_accepted_carries_taint 0 \
	"--accept-tainted records it and carries the taint into notes" "$TAINTED" \
	--accept-tainted "GREP:TAINTED: WindowServer at 60"

# A raised --busy-pct removes the gate that would have tainted, so the result
# looks clean for exactly the reason it must not.
WAIVED="$(make_result waived '{"waivers.busy_pct_raised": true}')"
check raised_busy_pct_refused 2 \
	"an empty taint behind a raised --busy-pct is refused, not trusted" "$WAIVED" \
	"GREP:interference"

check waivers_reach_notes 0 \
	"a waived guard is named in the recorded notes" "$WAIVED" \
	--accept-tainted "GREP:guards waived: busy_pct_raised"

# --- identity ----------------------------------------------------------------

STALE="$(make_result stale '{"arm_a.sha256_16": "deadbeefdeadbeef"}')"
check rebuilt_binary_refused 1 \
	"a binary rebuilt since the run is refused, not stamped onto the row" "$STALE" \
	"GREP:recorded sha256:deadbeefdeadbeef"

MISSING="$(make_result missing '{"arm_a.binary": "/nonexistent/rmlx"}')"
check absent_binary_refused 1 \
	"a binary that no longer exists cannot supply an identity" "$MISSING" \
	"GREP:no longer exists"

# --- cell key ----------------------------------------------------------------

WRONGCODEC="$(make_result wrongcodec '{"arm_b.args": "--kv-quant k8v8"}')"
check codec_mismatch_refused 1 \
	"a declared codec the arm did not run is refused" "$WRONGCODEC" \
	"GREP:ran --kv-quant k8v8"

NOCODEC="$(make_result nocodec '{"arm_b.args": "--max-tokens 100"}')"
check codec_absent_refused 1 \
	"an arm naming no codec is a different failure from naming the wrong one" "$NOCODEC" \
	"GREP:recorded no --kv-quant"

# clap resolves a repeated flag last-wins; taking the first would confirm a
# codec the slot did not run.
RELAST="$(make_result relast '{"arm_b.args": "--kv-quant none --kv-quant mixed_k8g64_v4g64"}')"
check repeated_codec_flag_takes_the_last 0 \
	"a repeated --kv-quant is read the way clap resolves it" "$RELAST"

PTOK="$(make_result ptok)"
check prompt_tokens_mismatch_refused 1 \
	"a supplied prompt_tokens that disagrees with the run is refused" "$PTOK" \
	--prompt-tokens 9999 "GREP:disagrees with the 3770"

check prompt_tokens_agreement_accepted 0 \
	"a supplied prompt_tokens that agrees is a cross-check, not an override" "$PTOK" \
	--prompt-tokens 3770 "GREP:\"prompt_tokens\": 3770"

# --- residency ---------------------------------------------------------------

# The four cells recorded on this branch came from a harness that emitted only
# the report table's 0.1 MB field. That path still ingests, and says so.
LEGACY="$(make_result legacy '{"results.0.arm_a.median_kv_cache_bytes": null, "results.0.arm_b.median_kv_cache_bytes": null, "results.0.arm_a.median_kv_cache_mb": 1000.0, "results.0.arm_b.median_kv_cache_mb": 2000.0}')"
check legacy_mb_field_discloses_resolution 0 \
	"a pre-byte-column result records, with its resolution stated in notes" "$LEGACY" \
	"GREP:\"kv_cache_bytes\"" \
	"GREP:0.1 MB display"

REFUSED="$(make_result kvrefused '{"results.0.arm_a.median_kv_cache_bytes": null}')"
check kv_refusal_records_null_not_zero 0 \
	"an arm with no residency figure records null, never 0" "$REFUSED" \
	"GREP:\"value\": null"

# --- shape -------------------------------------------------------------------

TWO="$(make_result two)"
python3 - "$TWO" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["results"].append(json.loads(json.dumps(d["results"][0])))
json.dump(d, open(p, "w"))
PY
check multi_model_result_refused 1 \
	"a two-model result cannot take one model label" "$TWO" \
	"GREP:compares 2 models"

echo ""
if [[ "$FAILED" -ne 0 ]]; then
	echo "perf_ab_ingest selftest: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
	exit 1
fi
echo "perf_ab_ingest selftest: ok ($PASSED cases)"
