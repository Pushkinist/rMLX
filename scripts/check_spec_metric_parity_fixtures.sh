#!/usr/bin/env bash
# check_spec_metric_parity_fixtures.sh — recall test for
# check_spec_metric_parity.sh.
#
# WHY
#   The parity gate reads two files with awk. A gate that reads source with awk
#   fails silently the day either file is reformatted — it finds nothing, sees
#   two empty sets, and reports agreement. This branch has already shipped a
#   whitelist and a control that both passed while measuring nothing, so the
#   parity gate does not get to be trusted on its shape alone.
#
#   Each case is a synthetic scan root: a copy of the repository's two files
#   with one edit. Every case asserts the literal exit code AND greps the reason,
#   because a gate that refuses for the wrong reason stops refusing when that
#   reason moves.
#
# Exit 0 = every case behaved. Exit 1 = at least one did not.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${REPO_ROOT}/scripts/check_spec_metric_parity.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/rmlx_spec_metric_parity.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

PASSED=0
FAILED=0

# root <name> — a scan root holding both files the gate reads, unedited.
root() {
	local dir="${WORK}/$1"
	mkdir -p "${dir}/scripts" "${dir}/crates/rmlx-metrics/src"
	cp "${GATE}" "${dir}/scripts/"
	cp "${REPO_ROOT}/crates/rmlx-metrics/src/registry.rs" "${dir}/crates/rmlx-metrics/src/"
	cp "${REPO_ROOT}/scripts/spec_bench.sh" "${dir}/scripts/"
	echo "${dir}"
}

# case <name> <want-exit> <what-it-proves> <grep-pattern>
judge() {
	local name="$1" want="$2" what="$3" pattern="${4:-}"
	local dir="${WORK}/${name}" out="${WORK}/${name}.log" got=0
	set +e
	bash "${dir}/scripts/check_spec_metric_parity.sh" >"${out}" 2>&1
	got=$?
	set -e
	local bad=""
	[ "${got}" -ne "${want}" ] && bad="exit=${got} (want ${want})"
	if [ -n "${pattern}" ] && ! grep -qE "${pattern}" "${out}"; then
		bad="${bad:+${bad}; }missing /${pattern}/"
	fi
	if [ -z "${bad}" ]; then
		printf 'ok    %-34s %s\n' "${name}" "${what}"
		PASSED=$((PASSED + 1))
	else
		printf 'FAIL  %-34s %s\n' "${name}" "${what}"
		printf '        %s\n' "${bad}"
		sed 's/^/        | /' "${out}"
		FAILED=$((FAILED + 1))
	fi
}

echo "check_spec_metric_parity fixtures"
echo

d="$(root unedited)"
judge unedited 0 "the tree as it stands agrees" "metrics declared and recorded"

d="$(root declared_only)"
python3 - "${d}/crates/rmlx-metrics/src/registry.rs" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = '    ("loop_ms_per_round", SpecRole::Derived),\n];'
assert old in s, "fixture anchor missing"
p.write_text(s.replace(old, '    ("loop_ms_per_round", SpecRole::Derived),\n    ("resteer_ms_per_round", SpecRole::Derived),\n];', 1))
PY
judge declared_only 1 "a metric declared and never recorded is named" "never recorded by spec_bench.sh"

d="$(root recorded_only)"
python3 - "${d}/scripts/spec_bench.sh" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = '    ("loop_ms_per_round", "loop_ms_per_round", float),\n)'
assert old in s, "fixture anchor missing"
p.write_text(s.replace(old, '    ("loop_ms_per_round", "loop_ms_per_round", float),\n    ("resteer_ms_per_round", "resteer_ms_per_round", float),\n)', 1))
PY
judge recorded_only 1 "a metric recorded and never declared is named" "not declared in registry::SPEC_METRICS"

d="$(root dropped_from_bench)"
python3 - "${d}/scripts/spec_bench.sh" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = '    ("tokens_per_round", "tokens_per_round", float),\n'
assert old in s, "fixture anchor missing"
p.write_text(s.replace(old, '', 1))
PY
judge dropped_from_bench 1 "a name deleted from the bench is the parity branch, not the unreadable one" "tokens_per_round"

d="$(root indented_python)"
python3 - "${d}/scripts/spec_bench.sh" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = 'SPEC_METRICS = ('
assert old in s, "fixture anchor missing"
p.write_text(s.replace(old, '    SPEC_METRICS = (', 1))
PY
judge indented_python 2 "a reformat that moves the marker off column 0 fails closed" "no SPEC_METRICS entries found"

d="$(root renamed_registry_const)"
python3 - "${d}/crates/rmlx-metrics/src/registry.rs" <<'PY'
import sys, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text()
old = 'pub const SPEC_METRICS: &[(&str, SpecRole)] = &['
assert old in s, "fixture anchor missing"
p.write_text(s.replace(old, 'pub const SPECULATIVE_METRICS: &[(&str, SpecRole)] = &[', 1))
PY
judge renamed_registry_const 2 "a rename of the oracle fails closed rather than reporting agreement" "no SPEC_METRICS entries found"

echo
echo "passed=${PASSED} failed=${FAILED}"
[ "${FAILED}" -eq 0 ] || exit 1
