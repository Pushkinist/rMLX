#!/usr/bin/env bash
# check_published_samples_fixtures.sh — recall test for
# `published_samples.py verify`.
#
# WHY
#   The verifier's whole job is to refuse. It reads a manifest, re-derives what
#   the manifest claims, and reports agreement — which is exactly the shape of a
#   check that quietly stops checking. A digest that is never compared, a
#   selection that is never re-drawn and an empty dataset list all look like
#   "ok" from the outside.
#
#   Each case below is a synthetic root: a copy of prompts/published/ with one
#   edit. Every case asserts the literal exit code AND greps the reason, because
#   a gate that refuses for the wrong reason stops refusing when that reason
#   moves. Cases that edit a sample file also refresh the manifest's recorded
#   file digest, so the edit reaches the checks past the digest rather than
#   stopping at it — that is the realistic drift, someone regenerating the
#   manifest around a file that is already wrong.
#
# Exit 0 = every case behaved. Exit 1 = at least one did not.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERIFY="${REPO_ROOT}/scripts/published_samples.py"
PUBLISHED="${REPO_ROOT}/prompts/published"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_published_samples.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0

# root <name> — a scan root holding an unedited copy of the published sample
# sets, plus an "upstream/" directory reconstructed from the checked-in samples'
# own source records so the --sources path can be exercised without network.
root() {
	local dir="${WORK}/$1"
	mkdir -p "${dir}"
	cp "${PUBLISHED}"/*.json "${dir}/"
	echo "${dir}"
}

# upstream <dir> — rebuild the pinned upstream files this repo drew from. Only
# the datasets whose whole pool is checked in can be reconstructed, so the
# --sources cases use a stub source whose digest the fixture writes into the
# manifest itself; what they prove is that the comparison happens, not what the
# real upstream bytes are.
upstream() {
	local dir="$1"
	mkdir -p "${dir}/upstream"
	python3 - "${dir}" <<'PY'
import hashlib, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
for entry in man["datasets"]:
    field = entry["pool_id_field"]
    lines = [
        json.dumps({field: pid}, ensure_ascii=False) for pid in entry["pool_ids"]
    ]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    if entry["source"]["encoding"] == "jsonl.gz":
        import gzip

        raw = gzip.compress(raw)
    (root / "upstream" / entry["source"]["cache_file"]).write_bytes(raw)
    entry["source"]["sha256"] = hashlib.sha256(raw).hexdigest()
(root / "manifest.json").write_text(
    json.dumps(man, ensure_ascii=False, indent=2) + "\n"
)
PY
}

# refresh <dir> <dataset-key> — recompute the manifest's recorded byte length and
# digest for one sample file, so an edit to that file reaches the later checks.
refresh() {
	python3 - "$1" "$2" <<'PY'
import hashlib, json, pathlib, sys

root, key = pathlib.Path(sys.argv[1]), sys.argv[2]
man = json.loads((root / "manifest.json").read_text())
for entry in man["datasets"]:
    if entry["key"] == key:
        blob = (root / entry["file"]).read_bytes()
        entry["file_bytes"] = len(blob)
        entry["file_sha256"] = hashlib.sha256(blob).hexdigest()
        break
else:
    raise SystemExit(f"no dataset {key}")
(root / "manifest.json").write_text(
    json.dumps(man, ensure_ascii=False, indent=2) + "\n"
)
PY
}

# judge <name> <want-exit> <what-it-proves> [grep-pattern] [--sources]
judge() {
	local name="$1" want="$2" what="$3" pattern="${4:-}" with_sources="${5:-}"
	local dir="${WORK}/${name}" out="${WORK}/${name}.log" got=0
	local -a args=(verify --root "${dir}")
	[ -n "${with_sources}" ] && args+=(--sources "${dir}/upstream")
	set +e
	python3 "${VERIFY}" "${args[@]}" >"${out}" 2>&1
	got=$?
	set -e
	local bad=""
	[ "${got}" -ne "${want}" ] && bad="exit=${got} (want ${want})"
	if [ -n "${pattern}" ] && ! grep -qE "${pattern}" "${out}"; then
		bad="${bad:+${bad}; }missing /${pattern}/"
	fi
	if [ -z "${bad}" ]; then
		printf 'ok    %-26s %s\n' "${name}" "${what}"
		PASSED=$((PASSED + 1))
	else
		printf 'FAIL  %-26s %s\n' "${name}" "${what}"
		printf '        %s\n' "${bad}"
		sed 's/^/        | /' "${out}"
		FAILED=$((FAILED + 1))
	fi
}

# run_absent <name> <pattern> — a reason the case must NOT report.
run_absent() {
	if grep -qE "$2" "${WORK}/$1.log"; then
		printf 'FAIL  %-26s %s\n' "$1" "unexpected /$2/"
		FAILED=$((FAILED + 1))
		PASSED=$((PASSED - 1))
	fi
}

# run_extra <name> <pattern> — a second assertion on a case already judged.
run_extra() {
	if ! grep -qE "$2" "${WORK}/$1.log"; then
		printf 'FAIL  %-26s %s\n' "$1" "missing /$2/"
		FAILED=$((FAILED + 1))
		PASSED=$((PASSED - 1))
	fi
}

echo "check_published_samples fixtures"
echo

# ── the tree as it stands ─────────────────────────────────────────────────────

d="$(root unedited)"
judge unedited 0 "the checked-in sets verify, and the count proves the scan was not empty" \
	"ok \(336 samples across 3 datasets, offline\)"

# ── digest and length of a checked-in file ────────────────────────────────────

d="$(root flipped_byte)"
python3 - "${d}/mt_bench.json" <<'PY'
import pathlib, sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
old = "Compose an engaging travel blog post"
assert old in s, "fixture anchor missing"
p.write_text(s.replace(old, "Compose an engaging travel blog pest", 1))
PY
judge flipped_byte 1 "one changed byte at unchanged length is a digest mismatch" \
	"mt_bench.json digest mismatch"

d="$(root truncated_file)"
python3 - "${d}/math_500.json" <<'PY'
import pathlib, sys

p = pathlib.Path(sys.argv[1])
b = p.read_bytes()
p.write_bytes(b[: len(b) - 4096])
PY
judge truncated_file 2 "a byte-truncated file names its length before it fails to parse" \
	"math_500.json size changed"
run_extra truncated_file "is unreadable"

# ── the recorded seed ─────────────────────────────────────────────────────────

d="$(root wrong_seed)"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
for entry in man["datasets"]:
    if entry["key"] == "math_500":
        assert entry["sampling"]["seed"] == 1729, "fixture anchor missing"
        entry["sampling"]["seed"] = 1730
(root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
PY
judge wrong_seed 1 "a seed the selection was not drawn with is caught and located" \
	"math_500: selection does not re-derive from the recorded seed"
run_extra wrong_seed "first divergence at index"

# ── the selection vs the file ─────────────────────────────────────────────────

d="$(root id_not_selected)"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
entry = next(e for e in man["datasets"] if e["key"] == "humaneval")
spare = next(p for p in entry["pool_ids"] if p not in entry["selected_ids"])
doc = json.loads((root / "humaneval.json").read_text())
doc["samples"][3]["source_id"] = spare
doc["samples"][3]["id"] = f"humaneval/{spare}"
(root / "humaneval.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
PY
refresh "${d}" humaneval
judge id_not_selected 1 "a sample the selection never drew is named" \
	"is not in the selected set"
run_extra id_not_selected "is absent from humaneval.json"

d="$(root duplicate_sample)"
python3 - "${d}" <<'PY'
import copy, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
doc = json.loads((root / "mt_bench.json").read_text())
doc["samples"][5] = copy.deepcopy(doc["samples"][4])
(root / "mt_bench.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
PY
refresh "${d}" mt_bench
judge duplicate_sample 1 "a sample counted twice does not pass as a full set" \
	"contains a duplicate sample id"

d="$(root sample_dropped)"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
doc = json.loads((root / "math_500.json").read_text())
del doc["samples"][7]
(root / "math_500.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
PY
refresh "${d}" math_500
judge sample_dropped 1 "a short set is refused with both counts" \
	"holds 127 samples, manifest records 128"

# ── the content address, and laundering it ────────────────────────────────────

d="$(root stale_body_digest)"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
doc = json.loads((root / "mt_bench.json").read_text())
s = doc["samples"][2]
s["body_sha256"] = ("1" if s["body_sha256"][0] == "0" else "0") + s["body_sha256"][1:]
(root / "mt_bench.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
PY
refresh "${d}" mt_bench
judge stale_body_digest 1 "a recorded body digest that does not hash the messages is caught" \
	"body digest mismatch"

d="$(root messages_laundered)"
python3 - "${d}" <<'PY'
import hashlib, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
doc = json.loads((root / "math_500.json").read_text())
s = doc["samples"][1]
s["messages"][0]["content"] = "Solve it.\n\n" + s["messages"][0]["content"]
canonical = json.dumps(s["messages"], ensure_ascii=False, separators=(",", ":"))
s["body_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
(root / "math_500.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
PY
refresh "${d}" math_500
judge messages_laundered 1 "an edited prompt with every digest re-blessed still fails on the template" \
	"messages do not render from the recorded template"

# ── the pinned upstream revision ──────────────────────────────────────────────

d="$(root upstream_moved)"
upstream "${d}"
python3 - "${d}/upstream/mt_bench.question.jsonl" <<'PY'
import pathlib, sys

p = pathlib.Path(sys.argv[1])
p.write_bytes(p.read_bytes() + b'{"question_id": 999}\n')
PY
judge upstream_moved 1 "an upstream file that moved off the pinned revision is named" \
	"differs from the pinned revision" --sources
run_extra upstream_moved "b494d0c6b4e7935f1764f8439e75da3e66beccc7"

d="$(root pool_reordered)"
upstream "${d}"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
entry = next(e for e in man["datasets"] if e["key"] == "math_500")
ids = entry["pool_ids"]
ids[0], ids[1] = ids[1], ids[0]
(root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
PY
judge pool_reordered 1 "a pool the selection still re-derives from is checked against upstream anyway" \
	"pool ids do not match the upstream file" --sources
run_absent pool_reordered "does not re-derive"

d="$(root sources_missing)"
upstream "${d}"
rm -f "${d}/upstream/math_500.test.jsonl"
judge sources_missing 2 "an absent upstream file fails closed rather than skipping the check" \
	"upstream file math_500.test.jsonl not found" --sources

# ── the manifest itself ───────────────────────────────────────────────────────

d="$(root no_datasets)"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
man["datasets"] = []
(root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
PY
judge no_datasets 2 "a manifest with nothing to check is a failure, not a pass" \
	"declares no datasets"

d="$(root dataset_dropped)"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
man["datasets"] = [e for e in man["datasets"] if e["key"] != "humaneval"]
(root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
PY
judge dataset_dropped 2 "checking two of the three sets is refused, not reported as ok" \
	"are not the sources this script builds"

d="$(root schema_bumped)"
python3 - "${d}" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
man = json.loads((root / "manifest.json").read_text())
man["schema_version"] = 2
(root / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=2) + "\n")
PY
judge schema_bumped 2 "a manifest this script does not read is refused, not guessed at" \
	"schema_version is 2"

d="$(root manifest_unreadable)"
printf '{"schema_version": 1, "datasets": [' >"${d}/manifest.json"
judge manifest_unreadable 2 "a half-written manifest fails closed" "manifest is unreadable"

d="$(root manifest_missing)"
rm -f "${d}/manifest.json"
judge manifest_missing 2 "no manifest is a failure, not an empty clean run" "manifest .* not found"

d="$(root sample_file_missing)"
rm -f "${d}/humaneval.json"
judge sample_file_missing 2 "a sample file the manifest names and the tree lacks fails closed" \
	"sample file humaneval.json not found"

echo
echo "passed=${PASSED} failed=${FAILED}"
[ "${FAILED}" -eq 0 ] || exit 1
