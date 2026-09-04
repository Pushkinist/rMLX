#!/usr/bin/env bash
# check_spec_metric_parity.sh — CI gate: the speculative metrics the engine
# declares are the ones the bench records.
#
# WHY
#   `rmlx_metrics::registry::SPEC_METRICS` is where a speculative metric is
#   declared, and a Rust test already forces the markdown export to render every
#   derived one. Nothing forced the *bench* to record them. `spec_bench.sh` maps
#   each metric name to a key of `scripts/lib/spec_round_log.py`'s output and a
#   cast; the key and the cast are the script's own business, but the metric
#   names are a verbatim second copy of the Rust list. Add a tenth metric and
#   the Rust test forces a column into the table while the bench never writes a
#   value for it, so the column renders `-` on every row for ever — the
#   single-producer property holding inside Rust and leaking at the language
#   boundary.
#
# ORACLE
#   The Rust list. This gate does not check it is right, only that there is
#   effectively one of it — the same contract as
#   `check-kv-boundary-default-parity`.
#
# Exit 0 = clean. Exit 1 = the two sets differ. Exit 2 = either side is unreadable.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTRY="${REPO_ROOT}/crates/rmlx-metrics/src/registry.rs"
BENCH="${REPO_ROOT}/scripts/spec_bench.sh"

# The `SPEC_METRICS` array literal in the registry: every `("name", SpecRole::_)`.
registry_names() {
	awk '
		/^pub const SPEC_METRICS: &\[\(&str, SpecRole\)\] = &\[/ { inside = 1; next }
		inside && /^\];/ { exit }
		inside && match($0, /"[a-z0-9_]+"/) {
			print substr($0, RSTART + 1, RLENGTH - 2)
		}
	' "${REGISTRY}" | sort -u
}

# The `SPEC_METRICS` tuple in the bench script: the first element of each row.
bench_names() {
	awk '
		/^SPEC_METRICS = \(/ { inside = 1; next }
		inside && /^\)/ { exit }
		inside && match($0, /\("[a-z0-9_]+"/) {
			print substr($0, RSTART + 2, RLENGTH - 3)
		}
	' "${BENCH}" | sort -u
}

WANT="$(registry_names)"
GOT="$(bench_names)"

if [[ -z "${WANT}" ]]; then
	echo "ERROR: no SPEC_METRICS entries found in ${REGISTRY#"${REPO_ROOT}/"}." >&2
	echo "That list is this gate's oracle; a rename has to update this script too." >&2
	exit 2
fi
if [[ -z "${GOT}" ]]; then
	echo "ERROR: no SPEC_METRICS entries found in ${BENCH#"${REPO_ROOT}/"}." >&2
	exit 2
fi

MISSING="$(comm -23 <(echo "${WANT}") <(echo "${GOT}") || true)"
EXTRA="$(comm -13 <(echo "${WANT}") <(echo "${GOT}") || true)"

status=0
if [[ -n "${MISSING}" ]]; then
	echo "ERROR: declared in registry::SPEC_METRICS and never recorded by spec_bench.sh:" >&2
	echo "${MISSING}" | sed 's/^/  /' >&2
	echo "  Its column in the markdown export renders '-' on every row." >&2
	status=1
fi
if [[ -n "${EXTRA}" ]]; then
	echo "ERROR: recorded by spec_bench.sh and not declared in registry::SPEC_METRICS:" >&2
	echo "${EXTRA}" | sed 's/^/  /' >&2
	echo "  Nothing gives it a unit, a direction or plausible bounds." >&2
	status=1
fi
if [[ "${status}" -eq 0 ]]; then
	echo "check-spec-metric-parity: ok ($(echo "${WANT}" | wc -l | tr -d ' ') metrics declared and recorded)"
fi
exit "${status}"
