#!/usr/bin/env bash
# published_ingest_selftest.sh — mutation check for
# scripts/ingest/published_ingest.py.
#
# WHY THIS EXISTS
#
# The bench harness produces a number; this is the step that makes it permanent.
# `observations` is append-only, so a row filed under the wrong prompt, the
# wrong binary or a sample set that has since moved cannot be taken back — it
# can only be out-voted by later rows that happen to be right. Every defence
# here is therefore a refusal, and a refusal nothing watches is a refusal that
# stops firing the moment its reason moves.
#
# Each case starts from one synthetic result over the REAL checked-in sample
# sets, applies one edit, and asserts the literal exit code and the reason. The
# `rmlx` binary is a file with the marker literals in it; the recorder is a stub
# that records what it was handed. No GPU, no model, no DB.
#
# Exit codes: 0 — every case behaved; 1 — at least one did not; 2 — the fixtures
# themselves could not be built.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INGEST="${REPO_ROOT}/scripts/ingest/published_ingest.py"
PUBLISHED="${REPO_ROOT}/prompts/published"

[ -r "${INGEST}" ] || { echo "ERROR: missing ${INGEST}" >&2; exit 2; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_published_ingest_selftest.XXXXXX")" || exit 2
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0

# ── The binary the run "measured" ─────────────────────────────────────────────
#
# The ingester re-hashes it and re-scans it for the marker literals the harness
# recorded, so the fixture has to be a real file with those bytes in it.
BINARY="${WORK}/rmlx"
cat >"${BINARY}" <<'BINEOF'
#!/bin/sh
# generate_streaming: TTFT
# generate: ITL stats (M30)
# generate: host categorical sampler active
# cache-type resolved
# mtp_generate: done
exit 0
BINEOF
chmod +x "${BINARY}"

# ── The recorder ──────────────────────────────────────────────────────────────
#
# `rmlx metrics record --file <path>` in the shape the ingester calls it, so the
# queue discipline is exercised: each record is moved into `pending/` only for
# the moment the recorder claims it, and taken back out either way.
STUB_RMLX="${WORK}/stub_rmlx"
cat >"${STUB_RMLX}" <<STUBEOF
#!/bin/sh
# Log which file was handed over, from where.
echo "\$4" >>"${WORK}/recorded.txt"
exit \${STUB_RECORD_EXIT:-0}
STUBEOF
chmod +x "${STUB_RMLX}"

# ── The base result ───────────────────────────────────────────────────────────
#
# Two cells over two real samples each, three passes, plus a fixed-prompt block.
# The sample ids and bodies are the checked-in ones, so the three-way digest
# assert has real inputs on all three sides.
BASE="${WORK}/base.json"
python3 - "${PUBLISHED}" "${BINARY}" "${BASE}" <<'PY' || exit 2
import hashlib, json, pathlib, sys

published, binary, out = (pathlib.Path(p) for p in sys.argv[1:4])
manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))

chosen = []
for entry in manifest["datasets"][:2]:
    doc = json.loads((published / entry["file"]).read_text(encoding="utf-8"))
    for sample in doc["samples"][:2]:
        chosen.append((entry["key"], sample))

samples = []
for number in (1, 2, 3):
    for dataset, sample in chosen:
        samples.append({
            "pass": number,
            "cell": f"{dataset}@1024",
            "dataset": dataset,
            "sample_id": sample["id"],
            "body_sha256": sample["body_sha256"],
            "max_tokens": 1024,
            "prompt_tokens": 512,
            "completion_tokens": 900,
            "ttft_ms": 210.0 + number,
            "decode_tps": 40.0 + number * 0.1,
            "client_decode_tps": 40.0,
        })

fixed_messages = [{"role": "user", "content": "Summarise this.\n\nA fixed body."}]
fixed_digest = hashlib.sha256(
    json.dumps(fixed_messages, ensure_ascii=False, separators=(",", ":")).encode()
).hexdigest()

blob = binary.read_bytes()
markers = {
    m: blob.count(m.encode())
    for m in (
        "generate_streaming: TTFT",
        "generate: ITL stats (M30)",
        "generate: host categorical sampler active",
        "cache-type resolved",
    )
}


def summary(values):
    mean = sum(values) / len(values)
    return {
        "pass_means": values,
        "mean": mean,
        "range_pct": (max(values) - min(values)) / mean * 100.0,
        "stable": True,
    }


out.write_text(json.dumps({
    "schema_version": 1,
    "backend": "rmlx",
    "backend_version": "0.4.1",
    "build_profile": "release-perf",
    "hardware_tag": "m5_max_128gb",
    "synthetic_arms": False,
    "unverified_samples": False,
    "arm": "plain",
    "model_namespace": "mlx-community",
    "model": "stub-model",
    "weight_quant": "mxfp8",
    "kv_quant": "none",
    "ctx_max": 8192,
    "ts_utc": "2026-09-06T00:00:00Z",
    "samples_root": str(published),
    "range_refusal_pct": 5,
    "protocol": {
        "passes": 3,
        "warmups_per_pass": 1,
        "macro_max_tokens": 1024,
        "sampling_resolved": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                              "min_p": 0.0, "seed": 42919},
        "thinking": "on, counted as output",
        "seed_policy": "engine default, identical in all three passes",
    },
    "binary": {
        "path": str(binary),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
        "arm": "plain",
        "markers": markers,
    },
    "host": {"pass_windows": ["quiet 0.0 -"] * 3,
             "thermal": ["pass 1 start nominal"],
             "thermal_source": "pmset -g therm",
             "taint": ""},
    "cells": {},
    "macro": {},
    "fixed_prompt": {
        "prompt_tokens": 1355,
        "target_tokens": 1355,
        "max_tokens": 1024,
        "body_sha256": fixed_digest,
        "corpus": "longctx_4k.json",
        "corpus_sha256": "a" * 64,
        "memory_poll_ms": 250,
        "words": 504,
        "filler_word": "the",
        "filler_reps": 0,
        "messages": fixed_messages,
        "decode_tps": summary([40.0, 40.1, 40.2]),
        "prefill_tps": summary([4500.0, 4501.0, 4502.0]),
        "ttft_ms": {"pass_means": [300.0, 301.0, 302.0], "mean": 301.0,
                    "range_pct": 0.66},
        "completion_tokens": {"pass_means": [900.0] * 3, "mean": 900.0,
                              "range_pct": 0.0},
        "phys_footprint_bytes": {"run_values": [31_000_000_000] * 3,
                                 "max": 31_000_000_000, "min": 31_000_000_000},
        "rss_bytes": {"run_values": [29_000_000_000] * 3,
                      "max": 29_000_000_000, "min": 29_000_000_000},
    },
    "samples": samples,
}, indent=1), encoding="utf-8")
print(f"base result: {len(chosen)} samples x 3 passes")
PY

# ── Case driver ───────────────────────────────────────────────────────────────

# ingest_case <name> <want-exit> <what-it-proves> <grep> [MUTATE:python] [ARGS:x]
#
# The mutation runs over `result` (the parsed base) and `work` (this case's
# directory), and whatever it leaves in `result` is what the ingester reads.
ingest_case() {
    CASE_NAME="$1"
    local want="$2"
    CASE_WHAT="$3"
    local pattern="$4"
    shift 4

    local dir="${WORK}/case_${CASE_NAME}"
    mkdir -p "${dir}"
    CASE_OUT="${WORK}/${CASE_NAME}.log"
    CASE_BAD=""
    CASE_HOME="${dir}/home"

    local mutation="" extra_args=() a
    for a in "$@"; do
        case "$a" in
        MUTATE:*) mutation="${a#MUTATE:}" ;;
        ARGS:*) extra_args+=("${a#ARGS:}") ;;
        esac
    done

    CASE_RESULT="${dir}/result.json"
    if ! python3 - "${BASE}" "${CASE_RESULT}" "${dir}" "${mutation}" <<'PY'
import json, pathlib, sys

base, out, work, mutation = sys.argv[1:5]
result = json.loads(pathlib.Path(base).read_text(encoding="utf-8"))
work = pathlib.Path(work)
if mutation:
    exec(mutation)
pathlib.Path(out).write_text(json.dumps(result, indent=1), encoding="utf-8")
PY
    then
        CASE_BAD="the fixture could not be built"
        return
    fi

    rm -f "${WORK}/recorded.txt"
    env RMLX_HOME="${CASE_HOME}" RMLX_REPO_ROOT="${REPO_ROOT}" \
        python3 "${INGEST}" "${CASE_RESULT}" \
        --rmlx-bin "${STUB_RMLX}" \
        ${extra_args[@]+"${extra_args[@]}"} >"${CASE_OUT}" 2>&1
    local got=$?
    [ "${got}" -ne "${want}" ] && CASE_BAD="exit=${got} (want ${want})"
    if [ -n "${pattern}" ] && ! grep -qE "${pattern}" "${CASE_OUT}"; then
        note_bad "missing /${pattern}/"
    fi
}

note_bad() {
    if [ -z "${CASE_BAD}" ]; then CASE_BAD="$1"; else CASE_BAD="${CASE_BAD}; $1"; fi
}

verdict() {
    if [ -z "${CASE_BAD}" ]; then
        printf 'ok    %-46s %s\n' "${CASE_NAME}" "${CASE_WHAT}"
        PASSED=$((PASSED + 1))
    else
        printf 'FAIL  %-46s %s\n' "${CASE_NAME}" "${CASE_WHAT}"
        printf '        %s\n' "${CASE_BAD}"
        tail -6 "${CASE_OUT}" | sed 's/^/        | /'
        FAILED=$((FAILED + 1))
    fi
}

# The records this case staged, as a JSON array.
staged_records() {
    python3 - "${CASE_HOME}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1]) / "bench" / "spec_bench_published" / "records"
files = sorted(root.glob("*.json")) if root.is_dir() else []
print(json.dumps([json.loads(f.read_text(encoding="utf-8")) for f in files]))
PY
}

# recs_check <python expression over `recs`>
recs_check() {
    staged_records | python3 -c 'import json, sys
recs = json.load(sys.stdin)
print(eval(sys.argv[1]))' "$1"
}

echo "published_ingest_selftest: one synthetic result over the checked-in samples"
echo

# ── What gets recorded ────────────────────────────────────────────────────────

ingest_case one_record_per_sample_plus_the_fixed_prompt 0 \
    "the measured things are the samples, and the fixed prompt is one more" \
    "^5 records$" ARGS:--dry-run
verdict

ingest_case sample_rows_carry_their_own_prompt 0 \
    "each row's prompt_id is the body that sample was measured on" ""
[ "$(recs_check "len(recs)")" = "5" ] || note_bad "$(recs_check "len(recs)") records"
[ "$(recs_check "sum(1 for r in recs if r['prompt']['name'].startswith('published/fixed_'))")" = "1" ] ||
    note_bad "no fixed-prompt record"
[ "$(recs_check "all(r['n_measure'] == 3 for r in recs)")" = "True" ] ||
    note_bad "a record does not say it is a mean of three"
# The body submitted for each sample is the one the measurement named, checked
# here against the result's own recorded digest rather than against the
# ingester's copy of the rule.
WANT_BODIES="$(python3 - "${CASE_RESULT}" <<'BODIES'
import json, pathlib, sys

result = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(sorted({s["body_sha256"] for s in result["samples"]})))
BODIES
)"
[ "$(recs_check "__import__('json').dumps(sorted(__import__('hashlib').sha256(__import__('json').dumps(r['prompt']['body'], ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() for r in recs if not r['prompt']['name'].startswith('published/fixed_')))")" = "${WANT_BODIES}" ] ||
    note_bad "the submitted bodies are not the measured ones"
[ "$(recs_check "sorted({m['name'] for r in recs for m in r['metrics']})")" = \
  "['accept_rate', 'accepted_per_step', 'decode_tps_warm', 'peak_phys_footprint_mb', 'peak_rss_mb', 'prefill_tps', 'tokens_per_round', 'ttft_warm_ms']" ] ||
    note_bad "metrics=$(recs_check "sorted({m['name'] for r in recs for m in r['metrics']})")"
verdict

# A plain run has no round loop, so its round figures are null and not zero: a
# zero there ranks in `bests` as a measured one.
ingest_case plain_rows_file_no_round_figure_and_no_decode_config 0 \
    "the plain arm records nulls where a drafter would have numbers" ""
[ "$(recs_check "all(m['value'] is None for r in recs for m in r['metrics'] if m['name'] in ('tokens_per_round', 'accepted_per_step', 'accept_rate'))")" = "True" ] ||
    note_bad "a plain row filed a round figure"
[ "$(recs_check "any('decode_config' in r for r in recs)")" = "False" ] ||
    note_bad "a plain row named a decode_config"
verdict

ingest_case speculative_rows_carry_the_engines_decode_config 0 \
    "a non-default engine configuration reaches the cell key" "" \
    'MUTATE:result["arm"] = "speculative"
result["decode_config"] = "mtp/block=5"
for row in result["samples"]:
    row["tokens_per_round"] = 2.5
    row["accepted_per_step"] = 1.5
    row["accept_rate"] = 0.6
del result["fixed_prompt"]'
[ "$(recs_check "sorted({r.get('decode_config') for r in recs})")" = "['mtp/block=5']" ] ||
    note_bad "decode_config=$(recs_check "sorted({r.get('decode_config') for r in recs})")"
[ "$(recs_check "all(m['value'] == 2.5 for r in recs for m in r['metrics'] if m['name'] == 'tokens_per_round')")" = "True" ] ||
    note_bad "a speculative row lost its tokens_per_round"
verdict

# The seed the engine resolved, not one this script decided. It is what the
# run-to-run spread in `notes` is a spread of.
ingest_case seed_and_temperature_are_the_engines 0 \
    "the row records the sampling the engine resolved" ""
[ "$(recs_check "sorted({r['seed'] for r in recs})")" = "[42919]" ] ||
    note_bad "seed=$(recs_check "sorted({r['seed'] for r in recs})")"
[ "$(recs_check "sorted({r['temperature'] for r in recs})")" = "[0.6]" ] ||
    note_bad "temperature=$(recs_check "sorted({r['temperature'] for r in recs})")"
[ "$(recs_check "all('seed engine default, identical in all three passes' in r['notes'] for r in recs)")" = "True" ] ||
    note_bad "a row does not say what its spread is a spread of"
verdict

ingest_case identity_is_the_runs_not_this_scripts 0 \
    "the identity fields are the ones stamped when the run was made" ""
[ "$(recs_check "sorted({r['backend_version'] for r in recs})")" = "['0.4.1']" ] ||
    note_bad "backend_version=$(recs_check "sorted({r['backend_version'] for r in recs})")"
[ "$(recs_check "sorted({r['build_profile'] for r in recs})")" = "['release-perf']" ] ||
    note_bad "build_profile=$(recs_check "sorted({r['build_profile'] for r in recs})")"
verdict

# ── The digest assert ─────────────────────────────────────────────────────────
#
# Three digests of one body: the one recorded with the measurement, the one in
# the sample manifest, and the one this recorder gives the body it is about to
# submit. A row whose prompt_id resolves to a body other than the measured one
# is attributed to a prompt that was never sent, and every later join on it
# splits with nothing saying so.

ingest_case measured_digest_that_moved_is_refused 2 \
    "a measurement recorded against another body files no row" \
    "They are not one prompt" \
    'MUTATE:moved = result["samples"][0]["sample_id"]
for row in result["samples"]:
    if row["sample_id"] == moved:
        row["body_sha256"] = "f" * 64'
verdict

# The other way one sample's digest can move: the passes disagree with each
# other rather than with the manifest. Different guard, different reason, and a
# fixture that reaches the three-way assert would never reach this one.
ingest_case passes_that_named_different_bodies_are_refused 2 \
    "passes that recorded different bodies did not measure one prompt" \
    "they did not measure one prompt" \
    'MUTATE:result["samples"][0]["body_sha256"] = "f" * 64'
verdict

# The manifest's leg and the recorder's leg are layered behind
# `published_samples.py verify`, which is re-run here: an edited sample set
# cannot reach the digest comparison because the root stops re-deriving first.
ingest_case sample_set_that_moved_is_refused 2 \
    "a sample set edited between the run and the record files no row" \
    "no longer re-derive from what" \
    'MUTATE:import shutil
copy = work / "samples"
shutil.copytree(result["samples_root"], copy)
doc = json.loads((copy / "mt_bench.json").read_text(encoding="utf-8"))
doc["samples"][0]["messages"][0]["content"] += " edited"
(copy / "mt_bench.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
result["samples_root"] = str(copy)'
verdict

ingest_case measured_sample_absent_from_the_sets_is_refused 2 \
    "a measurement of a sample the sets do not hold files no row" \
    "no sample of that id" \
    'MUTATE:absent = result["samples"][0]["sample_id"]
for row in result["samples"]:
    if row["sample_id"] == absent:
        row["sample_id"] = "mt_bench/999"'
verdict

ingest_case fixed_prompt_body_that_does_not_hash_is_refused 2 \
    "the fixed prompt's body must hash to the address recorded for it" \
    "does not hash to the digest the run recorded" \
    'MUTATE:result["fixed_prompt"]["messages"][0]["content"] += " edited"'
verdict

# ── Provenance ────────────────────────────────────────────────────────────────

ingest_case rebuilt_binary_is_refused 2 \
    "a binary rebuilt between the run and the record files no row" \
    "was rebuilt after the measurement" \
    'MUTATE:import pathlib
p = pathlib.Path(result["binary"]["path"])
copy = work / "rmlx"
copy.write_bytes(p.read_bytes() + b"\n# rebuilt\n")
result["binary"]["path"] = str(copy)'
verdict

# A digest can be made to agree by editing the record. The markers cannot: they
# are the literals the readings were read off, and a binary without them could
# not have produced the run.
ingest_case binary_whose_markers_went_missing_is_refused 2 \
    "a binary that agrees on its digest but lost a marker files no row" \
    "no longer contains" \
    'MUTATE:import hashlib, pathlib
p = pathlib.Path(result["binary"]["path"])
blob = p.read_bytes().replace(b"generate: ITL stats (M30)", b"")
copy = work / "rmlx"
copy.write_bytes(blob)
result["binary"]["path"] = str(copy)
result["binary"]["sha256"] = hashlib.sha256(blob).hexdigest()'
verdict

# A result written by something other than this harness is refused with the
# field it is missing, not with a traceback: an operator reading a stack trace
# cannot tell a malformed input from a broken ingester.
ingest_case result_missing_a_field_is_refused 2 \
    "a result this harness did not write is refused by name, not by traceback" \
    "is missing something every record needs" \
    'MUTATE:del result["protocol"]["seed_policy"]'
verdict

ingest_case result_without_a_binary_is_refused 2 \
    "a result that never said which binary it measured files no row" \
    "carries no binary identity" \
    'MUTATE:del result["binary"]'
verdict

# ── The gates ─────────────────────────────────────────────────────────────────

ingest_case synthetic_arms_is_refused_with_no_waiver 2 \
    "a run against a stub server has no measurement in it to accept" \
    "There is no waiver for this" \
    'MUTATE:result["synthetic_arms"] = True'
verdict

ingest_case unverified_samples_is_refused_with_no_waiver 2 \
    "a run on an unpinned sample copy is not a published measurement" \
    "not a published measurement" \
    'MUTATE:result["unverified_samples"] = True'
verdict

ingest_case taint_is_refused_without_the_waiver 2 \
    "a run taken on a machine that was not quiet is not recorded silently" \
    "run is TAINTED" \
    'MUTATE:result["host"]["taint"] = "thermal pass 2 mid: throttled=62; "'
verdict

ingest_case taint_with_the_waiver_reaches_the_row 0 \
    "the waiver records the run and carries the taint into the row" "" \
    'MUTATE:result["host"]["taint"] = "thermal pass 2 mid: throttled=62; "' \
    ARGS:--accept-tainted
[ "$(recs_check "all('TAINTED: thermal pass 2 mid: throttled=62' in r['notes'] for r in recs)")" = "True" ] ||
    note_bad "a row does not carry the taint: $(recs_check "recs[0]['notes']")"
verdict

# Thermal is recorded beside the numbers whether or not it tainted: a clean run
# that never says what state the machine was in is a claim nobody can check.
ingest_case thermal_reaches_the_row_even_when_it_did_not_taint 0 \
    "every row says what the machine's thermal state was" ""
[ "$(recs_check "all('thermal (pmset -g therm)' in r['notes'] for r in recs)")" = "True" ] ||
    note_bad "a row does not name its thermal reading: $(recs_check "recs[0]['notes']")"
verdict

ingest_case a_sample_measured_twice_is_refused 2 \
    "a mean over the passes that produced a row is not a mean over the passes" \
    "measured 2 times where the protocol" \
    'MUTATE:first = result["samples"][0]
result["samples"] = [s for s in result["samples"]
                     if not (s["sample_id"] == first["sample_id"] and s["pass"] == 3)]'
verdict

ingest_case one_body_with_two_lengths_is_refused 2 \
    "a prompt the server counted two ways is not one prompt" \
    "one body does not have two lengths" \
    'MUTATE:result["samples"][0]["prompt_tokens"] = 999'
verdict

# ── The queue ─────────────────────────────────────────────────────────────────
#
# Anything sitting in `metrics/buffer/pending/` is, by contract, work the next
# `--replay-pending` sweep claims — and that sweep quarantines what it cannot
# ingest. So a record is moved in only for the moment the recorder claims it.

ingest_case staging_is_outside_the_pending_queue 0 \
    "records written without --record cannot be claimed by a replay sweep" \
    "not ingested"
[ -d "${CASE_HOME}/metrics/buffer/pending" ] &&
    note_bad "the pending queue was created by a run that did not record"
[ "$(recs_check "len(recs)")" = "5" ] || note_bad "$(recs_check "len(recs)") staged"
verdict

ingest_case recording_leaves_the_queue_empty 0 \
    "each record passes through the queue and comes back out" \
    "ingested 5/5" ARGS:--record
[ -z "$(ls -A "${CASE_HOME}/metrics/buffer/pending" 2>/dev/null)" ] ||
    note_bad "a record was left in the pending queue"
[ "$(wc -l < "${WORK}/recorded.txt" | tr -d ' ')" = "5" ] ||
    note_bad "the recorder saw $(wc -l < "${WORK}/recorded.txt" | tr -d ' ') files"
grep -q "metrics/buffer/pending/" "${WORK}/recorded.txt" ||
    note_bad "the recorder was handed a path outside the pending queue"
verdict

STUB_RECORD_EXIT=1 ingest_case a_failed_ingest_keeps_its_record 1 \
    "a record the recorder rejected stays in staging, not in the queue" \
    "ingest FAILED" ARGS:--record
[ -z "$(ls -A "${CASE_HOME}/metrics/buffer/pending" 2>/dev/null)" ] ||
    note_bad "a rejected record was left in the pending queue"
[ "$(recs_check "len(recs)")" = "5" ] ||
    note_bad "$(recs_check "len(recs)") records survived in staging"
verdict

echo
echo "passed=${PASSED} failed=${FAILED}"
[ "${FAILED}" -eq 0 ] || exit 1
