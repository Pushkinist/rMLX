#!/usr/bin/env bash
# published_table_selftest.sh — mutation check for scripts/lib/published_table.py.
#
# WHY THIS EXISTS
#
# The emitter's whole job is to put a bound next to a measurement, and a
# percentage-of-a-bound is the shape that goes wrong quietly: if the ceiling is
# derived at the wrong context, the wrong codec, the wrong checkpoint or the
# wrong host, the column still renders, still looks plausible and is still
# wrong by a factor nobody can see from the page. So the numeric cases below do
# not check that the column is *present* — they re-derive the ceiling and the
# resident floor from the fixture snapshot's own safetensors header and
# `config.json`, in arithmetic that shares no code with `perf_ceiling.py`, and
# require the rendered figure to match it inside the precision it is printed at.
#
# That independent derivation is only possible because the fixture is chosen to
# make it possible: a dense model, no MoE, no sliding window, no tied
# embeddings, and `--kv-quant none`, for which the KV byte model is
# `layers x kv_heads x head_dim x 2 sides x 2 bytes` per token and nothing else.
# A codec with a boundary floor or a packed store would need the byte model
# re-implemented here, which would be a second producer of the arithmetic
# `make check-kv-byte-model-parity` exists to keep single. The codec and
# context are instead checked to be *passed through* — change either in the
# result and the rendered ceiling must move.
#
# Every case asserts a literal exit code and, for a refusal, the reason: a gate
# that refuses for the wrong reason stops refusing the moment that reason moves.
#
# No GPU, no model, no server, no DB. The "snapshot" is a safetensors header
# with no tensor data behind it.
#
# Exit codes: 0 — every case behaved; 1 — at least one did not; 2 — the fixtures
# themselves could not be built.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EMITTER="${REPO_ROOT}/scripts/lib/published_table.py"
FIXTURES="${REPO_ROOT}/scripts/fixtures/published_table"
SNAPSHOT="${FIXTURES}/fixture__published-table-16L"

[ -r "${EMITTER}" ] || { echo "ERROR: missing ${EMITTER}" >&2; exit 2; }
[ -d "${SNAPSHOT}" ] || { echo "ERROR: missing ${SNAPSHOT}" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_published_table_selftest.XXXXXX")" || exit 2
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0

# ── Case driver ───────────────────────────────────────────────────────────────
#
# table_case <name> <want-exit> <what-it-proves> <grep-or-empty> [OPT...]
#
#   PLAIN:<py>   mutate the plain result   (object bound as `result`)
#   MTP:<py>     mutate the mtp result
#   DFLASH:<py>  mutate the dflash result
#   ONLY:a,b     which results to pass, by key (default plain,mtp,dflash)
#   ARGS:<flag>  an extra flag for the emitter
#   ABSENT:<re>  the output must NOT match this
table_case() {
    CASE_NAME="$1"
    local want="$2"
    CASE_WHAT="$3"
    local pattern="$4"
    shift 4

    local dir="${WORK}/case_${CASE_NAME}"
    mkdir -p "${dir}"
    CASE_OUT="${WORK}/${CASE_NAME}.out"
    CASE_ERR="${WORK}/${CASE_NAME}.err"
    CASE_BAD=""

    local only="plain,mtp,dflash" absent="" model="${SNAPSHOT}"
    local -a extra=() mutations=()
    local a
    for a in "$@"; do
        case "$a" in
        PLAIN:*|MTP:*|DFLASH:*) mutations+=("$a") ;;
        ONLY:*) only="${a#ONLY:}" ;;
        MODEL:*) model="${a#MODEL:}" ;;
        ARGS:*) extra+=("${a#ARGS:}") ;;
        ABSENT:*) absent="${a#ABSENT:}" ;;
        esac
    done

    local key
    for key in plain mtp dflash; do
        local mutation=""
        for a in ${mutations[@]+"${mutations[@]}"}; do
            case "$a" in
            PLAIN:*) [ "${key}" = plain ] && mutation="${a#PLAIN:}" ;;
            MTP:*) [ "${key}" = mtp ] && mutation="${a#MTP:}" ;;
            DFLASH:*) [ "${key}" = dflash ] && mutation="${a#DFLASH:}" ;;
            esac
        done
        if ! python3 - "${FIXTURES}/result_${key}.json" "${dir}/${key}.json" \
                "${mutation}" <<'PY'
import json, pathlib, sys

src, out, mutation = sys.argv[1:4]
result = json.loads(pathlib.Path(src).read_text(encoding="utf-8"))
if mutation:
    exec(mutation)
pathlib.Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")
PY
        then
            CASE_BAD="the fixture could not be built"
            return
        fi
    done

    local -a inputs=()
    for key in ${only//,/ }; do inputs+=("${dir}/${key}.json"); done

    python3 "${EMITTER}" "${inputs[@]}" --model "${model}" \
        ${extra[@]+"${extra[@]}"} >"${CASE_OUT}" 2>"${CASE_ERR}"
    local got=$?
    [ "${got}" -ne "${want}" ] && CASE_BAD="exit=${got} (want ${want})"
    if [ -n "${pattern}" ] && ! grep -qE -- "${pattern}" "${CASE_OUT}" "${CASE_ERR}"; then
        note_bad "missing /${pattern}/"
    fi
    if [ -n "${absent}" ] && grep -qE -- "${absent}" "${CASE_OUT}"; then
        note_bad "unexpected /${absent}/"
    fi
}

note_bad() {
    if [ -z "${CASE_BAD}" ]; then CASE_BAD="$1"; else CASE_BAD="${CASE_BAD}; $1"; fi
}

verdict() {
    if [ -z "${CASE_BAD}" ]; then
        printf 'ok    %-44s %s\n' "${CASE_NAME}" "${CASE_WHAT}"
        PASSED=$((PASSED + 1))
    else
        printf 'FAIL  %-44s %s\n' "${CASE_NAME}" "${CASE_WHAT}"
        printf '        %s\n' "${CASE_BAD}"
        tail -4 "${CASE_ERR}" | sed 's/^/        | /'
        FAILED=$((FAILED + 1))
    fi
}

# ── The independent oracle ────────────────────────────────────────────────────
#
# Re-derives what the emitter should have printed, from the fixture snapshot's
# own header and config, sharing no code with perf_ceiling.py. `check_number`
# takes a Markdown cell out of the rendered table and compares.
oracle() {
    python3 - "${SNAPSHOT}" "$@" <<'PY'
import json, pathlib, struct, sys

snapshot = pathlib.Path(sys.argv[1])
what, ctx, max_ctx = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
cfg = json.loads((snapshot / "config.json").read_text())

blob = (snapshot / "model.safetensors").read_bytes()
n = struct.unpack("<Q", blob[:8])[0]
header = json.loads(blob[8:8 + n])
sizes = {k: v["data_offsets"][1] - v["data_offsets"][0]
         for k, v in header.items() if k != "__metadata__"}

# Untied embeddings: the input table is gathered one row at a time and never
# streamed, but it is held. Nothing else in this fixture is excluded.
assert cfg["tie_word_embeddings"] is False
streamed = sum(v for k, v in sizes.items() if k != "model.embed_tokens.weight")
resident_weights = sum(sizes.values())

# `--kv-quant none` on a dense stack with no sliding window: two bf16 sides,
# every layer, every kv head.
per_token = (cfg["num_hidden_layers"] * cfg["num_key_value_heads"]
             * cfg["head_dim"] * 2 * 2)
ring = min(1 << (ctx - 1).bit_length(), max_ctx)

values = {
    "ceiling_tps": 614e9 / (streamed + per_token * ctx),
    "resident_floor_gb": (resident_weights + per_token * ring) / 1e9,
    "resident_weights_gb": resident_weights / 1e9,
    "kv_resident_gb": per_token * ring / 1e9,
}
print(f"{values[what]:.6f}")
PY
}

# check_number <name> <what-it-proves> <rendered> <oracle-value> <tolerance>
check_number() {
    CASE_NAME="$1"
    CASE_WHAT="$2"
    CASE_BAD=""
    CASE_ERR="${WORK}/${CASE_NAME}.err"
    : >"${CASE_ERR}"
    local got="$3" want="$4" tol="$5"
    local ok
    ok="$(python3 -c 'import sys
got, want, tol = (float(x) for x in sys.argv[1:4])
print("yes" if abs(got - want) <= tol else f"{got} vs {want}")' "${got}" "${want}" "${tol}")"
    [ "${ok}" = yes ] || note_bad "rendered ${ok}"
}

# The Markdown cell at <row-match> / <column-index>, from a rendered table.
cell_at() {
    python3 - "$1" "$2" "$3" <<'PY'
import pathlib, re, sys

path, row_match, index = sys.argv[1], sys.argv[2], int(sys.argv[3])
for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
    if line.startswith("|") and re.search(row_match, line):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        print(cells[index])
        break
else:
    print("NO-ROW")
PY
}

echo "published_table_selftest: the emitter, against a header-only fixture snapshot"
echo

# ── The bound is the right number, not merely a number ───────────────────────

table_case ceiling_matches_an_independent_derivation 0 \
    "the ceiling column is bandwidth over this snapshot's own bytes" ""
RENDERED="${CASE_OUT}"
verdict

# mt_bench:1024 — prompt 85.4, completion 381.2, so the window middle is 276.
CEIL_RENDERED="$(cell_at "${RENDERED}" '^\| `mt_bench:1024` \| `plain`' 6)"
CTX_RENDERED="$(cell_at "${RENDERED}" '^\| `mt_bench:1024` \| `plain`' 9)"
check_number ceiling_value "the printed ceiling is the derived one, to 0.01 tok/s" \
    "${CEIL_RENDERED}" "$(oracle ceiling_tps 276 8192)" 0.01
verdict

check_number ceiling_ctx "the ceiling is taken at the middle of the decode window" \
    "${CTX_RENDERED}" 276 0
verdict

CASE_NAME=percent_of_ceiling_is_the_quotient
CASE_WHAT="the percentage is the measured rate over that ceiling"
CASE_BAD=""; CASE_ERR="${WORK}/pct.err"; : >"${CASE_ERR}"
PCT_RENDERED="$(cell_at "${RENDERED}" '^\| `mt_bench:1024` \| `plain`' 7)"
RATE_RENDERED="$(cell_at "${RENDERED}" '^\| `mt_bench:1024` \| `plain`' 4)"
check_number percent_of_ceiling_is_the_quotient \
    "the percentage is the measured rate over that ceiling" \
    "${PCT_RENDERED%\%}" \
    "$(python3 -c 'import sys; print(float(sys.argv[1]) / float(sys.argv[2]) * 100)' \
        "${RATE_RENDERED}" "${CEIL_RENDERED}")" 0.05
verdict

# The fixed-prompt block: 1355 prompt + 1024 completion, middle 1867, ring 2048.
FLOOR_RENDERED="$(cell_at "${RENDERED}" '^\| peak `phys_footprint`' 2)"
check_number resident_floor_matches_an_independent_derivation \
    "the resident floor is this snapshot's held weights plus the ring's KV" \
    "${FLOOR_RENDERED}" "$(oracle resident_floor_gb 1867 8192)" 0.005
verdict

# ── The bound is derived from the run's own inputs ───────────────────────────

# `mixed_k8g64_v4g64` and not `k8v8`: on this stack `k8v8` decodes off its bf16
# mirror, so its streamed bytes are `none`'s to the byte and a pass-through test
# against it could never fail. The mixed codec reads its packed store.
table_case ceiling_follows_the_codec 0 \
    "a run recorded at another codec is priced at that codec, not at none" \
    'priced at `mixed_k8g64_v4g64`' \
    PLAIN:'result["kv_quant"] = "mixed_k8g64_v4g64"' \
    MTP:'result["kv_quant"] = "mixed_k8g64_v4g64"' \
    DFLASH:'result["kv_quant"] = "mixed_k8g64_v4g64"'
MIXED_CEIL="$(cell_at "${CASE_OUT}" '^\| `mt_bench:1024` \| `plain`' 6)"
MIXED_FLOOR="$(cell_at "${CASE_OUT}" '^\| peak `phys_footprint`' 2)"
[ "${MIXED_CEIL}" = "${CEIL_RENDERED}" ] && \
    note_bad "the mixed codec priced identically to none (${MIXED_CEIL}) — the codec is not reaching perf_ceiling.py"
[ "${MIXED_FLOOR}" = "${FLOOR_RENDERED}" ] && \
    note_bad "the mixed codec's resident floor equals none's (${MIXED_FLOOR})"
verdict

table_case ceiling_follows_the_context_ceiling 0 \
    "the run's --max-ctx sizes the ring the resident floor counts" \
    '--max-ctx 1024' \
    PLAIN:'result["ctx_max"] = 1024' \
    MTP:'result["ctx_max"] = 1024' \
    DFLASH:'result["ctx_max"] = 1024'
SMALL_RING_FLOOR="$(cell_at "${CASE_OUT}" '^\| peak `phys_footprint`' 2)"
check_number ceiling_follows_the_context_ceiling \
    "the run's --max-ctx sizes the ring the resident floor counts" \
    "${SMALL_RING_FLOOR}" "$(oracle resident_floor_gb 1867 1024)" 0.005
verdict

table_case codec_the_engine_cannot_name_is_refused 1 \
    "a codec perf_ceiling.py cannot parse stops the table, it does not blank a column" \
    'perf_ceiling.py could not price' \
    PLAIN:'result["kv_quant"] = "k9v9"' \
    MTP:'result["kv_quant"] = "k9v9"' \
    DFLASH:'result["kv_quant"] = "k9v9"'
verdict

# ── The bound belongs to this checkpoint and this host ───────────────────────

table_case another_checkpoints_ceiling_is_refused 1 \
    "a ceiling from the wrong snapshot renders as plausibly as a right one" \
    'was measured on fixture/other-model' \
    PLAIN:'result["model"] = "other-model"' \
    MTP:'result["model"] = "other-model"' \
    DFLASH:'result["model"] = "other-model"'
verdict

table_case another_hosts_bandwidth_is_refused 1 \
    "a percentage of another machine's ceiling is a percentage of nothing" \
    "perf_ceiling.py's bandwidth constant was measured on" \
    PLAIN:'result["hardware_tag"] = "m3_ultra_256gb"' \
    MTP:'result["hardware_tag"] = "m3_ultra_256gb"' \
    DFLASH:'result["hardware_tag"] = "m3_ultra_256gb"'
verdict

table_case another_hosts_bandwidth_can_be_stated 0 \
    "...and naming that host's bandwidth makes the substitution deliberate" \
    '819 GB/s on `m3_ultra_256gb`' \
    ARGS:--bandwidth-gbs ARGS:819 \
    PLAIN:'result["hardware_tag"] = "m3_ultra_256gb"' \
    MTP:'result["hardware_tag"] = "m3_ultra_256gb"' \
    DFLASH:'result["hardware_tag"] = "m3_ultra_256gb"'
verdict

table_case missing_snapshot_is_refused 1 \
    "a snapshot that is not there is named, not silently priced at zero" \
    'is not a directory' \
    ONLY:plain MODEL:"${WORK}/no-such-snapshot"
verdict

# ── One ceiling must describe the whole decode window ────────────────────────

table_case ceiling_that_moves_across_the_window_is_refused 3 \
    "a window the ceiling falls across is reported per context, not at its middle" \
    'past the 5% band one number is allowed' \
    PLAIN:'result["cells"]["math_500:4096"]["completion_tokens_mean"] = 40000' \
    MTP:'result["cells"]["math_500:4096"]["completion_tokens_mean"] = 40000' \
    DFLASH:'result["cells"]["math_500:4096"]["completion_tokens_mean"] = 40000'
verdict

table_case the_spread_band_is_a_band_not_a_constant 0 \
    "...and widening the band is the deliberate way to print it anyway" "" \
    ARGS:--ceiling-spread-pct ARGS:40 \
    PLAIN:'result["cells"]["math_500:4096"]["completion_tokens_mean"] = 40000' \
    MTP:'result["cells"]["math_500:4096"]["completion_tokens_mean"] = 40000' \
    DFLASH:'result["cells"]["math_500:4096"]["completion_tokens_mean"] = 40000'
verdict

# ── Several results are one comparison, or they are not a table ──────────────

table_case one_mode_twice_is_refused 1 \
    "a mode compared against itself is not a speedup" \
    'the same engine mode' \
    DFLASH:'result["decode_config"] = "mtp_sidecar/block=7"'
verdict

table_case two_codecs_are_refused 1 \
    "rows compared down a column must have been measured at one codec" \
    'kv_quant=' \
    MTP:'result["kv_quant"] = "k8v8"'
verdict

table_case two_output_budgets_are_refused 1 \
    "a macro at two different budgets is two different headline figures" \
    'macro_max_tokens=' \
    MTP:'result["protocol"]["macro_max_tokens"] = 512'
verdict

table_case two_binaries_are_refused 1 \
    "a speedup between two binaries is not a speedup between two engine modes" \
    'is not a speedup between two engine modes' \
    MTP:'result["binary"]["sha256"] = "f" * 64'
verdict

table_case two_sample_sets_are_refused 1 \
    "modes that measured different cells are not rows of one table" \
    'measured cells' \
    MTP:'result["cells"].pop("humaneval:1024")'
verdict

table_case two_sample_counts_are_refused 1 \
    "a cell measured over 80 samples in one mode and 40 in another is two cells" \
    'samples and' \
    MTP:'result["cells"]["humaneval:1024"]["samples"] = 40'
verdict

# ── What the reader is told ──────────────────────────────────────────────────

table_case a_synthetic_run_says_so_at_the_top 0 \
    "a fixture rendering announces itself before any number" \
    'THIS TABLE HOLDS NO PUBLISHABLE MEASUREMENT'
verdict

table_case a_real_run_carries_no_banner 0 \
    "...and the banner is not boilerplate that would be there either way" "" \
    ABSENT:'HOLDS NO PUBLISHABLE MEASUREMENT' \
    PLAIN:'result["synthetic_arms"] = False' \
    MTP:'result["synthetic_arms"] = False' \
    DFLASH:'result["synthetic_arms"] = False'
verdict

table_case an_unverified_sample_root_is_disclosed 0 \
    "samples that are not the pinned ones reach the banner" \
    'unverified root' \
    PLAIN:'result["synthetic_arms"] = False; result["unverified_samples"] = True' \
    MTP:'result["synthetic_arms"] = False' \
    DFLASH:'result["synthetic_arms"] = False'
verdict

table_case a_taint_is_disclosed 0 \
    "a busy or throttled host reaches the banner with its own reason" \
    'tainted.*thermal state was throttled' \
    PLAIN:'result["synthetic_arms"] = False; result["host"]["taint"] = "the thermal state was throttled"' \
    MTP:'result["synthetic_arms"] = False' \
    DFLASH:'result["synthetic_arms"] = False'
verdict

table_case the_three_disclosures_are_in_the_header 0 \
    "two-turn MT-Bench, the macro's definition and the fixed seed are stated" \
    'only the first turn is measured'
grep -q 'macro average is one cell per dataset' "${CASE_OUT}" || \
    note_bad "the macro's definition is not stated"
grep -q 'seed is held fixed across all three passes' "${CASE_OUT}" || \
    note_bad "the seed policy is not stated"
grep -qE '\| 0 of 80 \|' "${CASE_OUT}" || \
    note_bad "divergent_samples is not in the table"
verdict

table_case the_pinned_choices_are_in_the_header 0 \
    "max output tokens, thinking, warmup and the memory counter are printed" \
    'max output tokens . 1024'
grep -q 'thinking tokens . on, counted as output' "${CASE_OUT}" || \
    note_bad "the thinking-token choice is not printed"
grep -q '1 untimed request per pass' "${CASE_OUT}" || \
    note_bad "the warmup is not printed"
grep -q 'peak `phys_footprint`' "${CASE_OUT}" || \
    note_bad "the memory counter is not printed"
grep -q 'engine default, identical in all three passes' "${CASE_OUT}" || \
    note_bad "the seed policy chunk 3 recorded is not printed"
verdict

table_case an_unstable_mean_is_withheld_not_printed 0 \
    "a refused mean leaves no rate and no percentage behind it" \
    '\*\*UNSTABLE\*\*' \
    PLAIN:'result["cells"]["mt_bench:1024"]["stable"] = False'
grep -qE '^\| `mt_bench:1024` \| `plain` \|.*\| \*\*UNSTABLE\*\* \|.*\| — \|' "${CASE_OUT}" || \
    note_bad "an unstable cell still printed a percentage of its ceiling"
verdict

table_case no_speculative_ceiling_is_invented 0 \
    "a speculative arm is priced against the autoregressive bound, and says so" \
    'there is no speculative ceiling here'
grep -q 'AR ceiling' "${CASE_OUT}" || note_bad "the column does not say which bound it is"
grep -qE '\| 1[0-9][0-9]\.[0-9]% \|' "${CASE_OUT}" || \
    note_bad "no speculative row exceeded the AR bound, so the >100% case is untested"
verdict

table_case a_resized_block_quotes_no_block_kept 0 \
    "a loop that resized its block is not scored against the block it was given" \
    '— \(adaptive block\)'
grep -qE '^\| `mt_bench:1024` \| `mtp_sidecar/block=7` \|.*\| 48\.7% \|' "${CASE_OUT}" || \
    note_bad "a fixed-block loop did not get a block-kept figure"
verdict

table_case a_fixed_block_loop_is_scored_against_its_block 0 \
    "...and the same drafter without the resize term is" \
    '\| 18\.0% \|' \
    DFLASH:'result["decode_config"] = "dflash/block=16"'
verdict

table_case the_speedup_needs_a_plain_arm 0 \
    "with no autoregressive row there is nothing to be a multiple of" "" \
    ONLY:mtp,dflash \
    ABSENT:'\| 1\.679 \|'
verdict

table_case the_fixed_prompt_block_is_the_plain_arms 0 \
    "the protocol's second figure is the autoregressive one and only it has one" \
    'The fixed-length prompt — 1355 tokens'
if [ "$(grep -c 'The fixed-length prompt' "${CASE_OUT}")" -ne 1 ]; then
    note_bad "the fixed-prompt block was rendered more than once"
fi
verdict

table_case an_absent_prefill_anchor_is_absent_not_guessed 0 \
    "without a runs.db anchor the input-speed bound is empty and says why" \
    'input-speed bound is empty on purpose'
verdict

echo
echo "published_table_selftest: ${PASSED} passed, ${FAILED} failed"
[ "${FAILED}" -eq 0 ] || exit 1
exit 0
