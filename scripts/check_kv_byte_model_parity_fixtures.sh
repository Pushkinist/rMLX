#!/usr/bin/env bash
# scripts/check_kv_byte_model_parity_fixtures.sh — recall test for
# `check_kv_byte_model_parity.sh`: each fixture is one edit to an otherwise
# valid manifest, and asserts both the exit code and *which* check fired.
#
# WHY BOTH
#   The parity gate reports a disagreement (exit 1) and an inability to look
#   (exit 2) on different exits on purpose. A recall test that asserted only
#   "non-zero" would pass just as happily against a gate that had stopped being
#   able to read a manifest at all — which is the failure mode it exists to rule
#   out.
#
#   Nothing here hard-codes a byte count. The passing fixture's values are
#   produced by `perf_ceiling.py --byte-model` from a skeleton, so a change to
#   either byte model moves the fixture with it and these cases keep testing the
#   gate's plumbing rather than last year's arithmetic.
#
# EXIT CODES
#   0  every fixture produced the expected exit code and reason
#   1  a fixture did not
#   2  the fixtures themselves could not be built

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="${REPO_ROOT}/scripts/check_kv_byte_model_parity.sh"
PERF_CEILING="${REPO_ROOT}/scripts/perf_ceiling.py"

[ -x "${GATE}" ] || { echo "ERROR: missing ${GATE}" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || {
    echo "ERROR: python3 not on PATH" >&2
    exit 2
}

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# Skeleton: two codecs whose values differ under every arm this gate guards
# (a ring store and the mirror-conditional family), both topologies, two head
# dimensions. The trailing value is a placeholder — `--byte-model` replaces it.
cat >"${WORK}/skeleton" <<'EOF'
KVBYTES	iso3_sym	0	4096	128	8	0
KVBYTES	iso3_sym	0	4096	256	2	0
KVFLOOR	iso3_sym	0	36	x	x
KVBYTES	iso3_sym	1	4096	128	8	0
KVBYTES	iso3_sym	1	4096	256	2	0
KVFLOOR	iso3_sym	1	36	x	x
KVBYTES	mixed_k8g64_v4g64	0	4096	128	8	0
KVBYTES	mixed_k8g64_v4g64	0	4096	256	2	0
KVFLOOR	mixed_k8g64_v4g64	0	36	x	x
KVBYTES	mixed_k8g64_v4g64	1	4096	128	8	0
KVBYTES	mixed_k8g64_v4g64	1	4096	256	2	0
KVFLOOR	mixed_k8g64_v4g64	1	36	x	x
EOF

python3 "${PERF_CEILING}" --byte-model <"${WORK}/skeleton" >"${WORK}/rows" ||
    { echo "ERROR: could not build the fixture rows" >&2; exit 2; }
n_rows=$(awk 'END { print NR }' "${WORK}/rows")
[ "${n_rows}" -eq 12 ] ||
    { echo "ERROR: skeleton produced ${n_rows} rows, expected 12" >&2; exit 2; }

wrap() { # wrap ROWS_FILE OUT_FILE [DECLARED_COUNT]
    local rows="$1" out="$2" declared="${3:-}"
    [ -n "${declared}" ] || declared=$(awk 'END { print NR }' "${rows}")
    {
        echo "KVBYTES-BEGIN"
        cat "${rows}"
        printf 'KVBYTES-END\t%s\n' "${declared}"
    } >"${out}"
}

failures=0
check() { # check LABEL MANIFEST WANT_EXIT WANT_PATTERN
    local label="$1" manifest="$2" want_exit="$3" want_pat="$4"
    local out rc
    out="$(bash "${GATE}" "${manifest}" 2>&1)"
    rc=$?
    if [ "${rc}" -ne "${want_exit}" ]; then
        echo "FAIL  ${label}: exit ${rc}, expected ${want_exit}" >&2
        echo "${out}" | head -5 >&2
        failures=$((failures + 1))
        return
    fi
    if ! grep -qE -- "${want_pat}" <<<"${out}"; then
        echo "FAIL  ${label}: exit ${rc} as expected but no reason matching /${want_pat}/" >&2
        echo "${out}" | head -5 >&2
        failures=$((failures + 1))
        return
    fi
    echo "ok    ${label}  (exit ${rc}, reason matched)"
}

# 0 — the unmutated manifest passes. Without this the whole file could be
# passing because the gate rejects everything.
wrap "${WORK}/rows" "${WORK}/good.manifest"
check "clean manifest passes" "${WORK}/good.manifest" 0 "byte-model rows"

# 1 — one byte count off by one: a disagreement, not an environment problem.
awk -F'\t' 'BEGIN { OFS = "\t" }
    NR == 1 && $1 == "KVBYTES" { $7 = $7 + 1 }
    { print }' "${WORK}/rows" >"${WORK}/rows.wrongbyte"
wrap "${WORK}/rows.wrongbyte" "${WORK}/wrongbyte.manifest"
check "one wrong byte count" "${WORK}/wrongbyte.manifest" 1 "disagrees with the engine"

# 2 — one boundary codec off: the layer vector is checked, not just the bytes.
awk -F'\t' 'BEGIN { OFS = "\t" }
    $1 == "KVFLOOR" && $2 ~ /^mixed_/ && $3 == "0" { $5 = "k8v8" }
    { print }' "${WORK}/rows" >"${WORK}/rows.wrongfloor"
wrap "${WORK}/rows.wrongfloor" "${WORK}/wrongfloor.manifest"
check "one wrong boundary codec" "${WORK}/wrongfloor.manifest" 1 "disagrees with the engine"

# 3 — only the dense topology swept. The Mixed family's mirror is the whole
# difference between the two, so a one-sided manifest checks half the model.
awk -F'\t' '$3 == "0"' "${WORK}/rows" >"${WORK}/rows.onetopology"
wrap "${WORK}/rows.onetopology" "${WORK}/onetopology.manifest"
check "single topology" "${WORK}/onetopology.manifest" 2 "topologies"

# 4 — only one head dimension. A wrong sideband width and a wrong per-row term
# agree at exactly one head_dim.
awk -F'\t' '$1 == "KVFLOOR" || $5 == "128"' "${WORK}/rows" >"${WORK}/rows.oneshape"
wrap "${WORK}/rows.oneshape" "${WORK}/oneshape.manifest"
check "single head_dim" "${WORK}/oneshape.manifest" 2 "shape"

# 5 — the END sentinel over-counts: the manifest was truncated in transit and a
# diff over what survived would pass.
wrap "${WORK}/rows" "${WORK}/truncated.manifest" 999
check "truncated manifest" "${WORK}/truncated.manifest" 2 "truncated"

# 6 — no BEGIN sentinel: the emitter never ran. Must not read as "no
# disagreements found".
cat "${WORK}/rows" >"${WORK}/nobegin.manifest"
check "no BEGIN sentinel" "${WORK}/nobegin.manifest" 2 "did not run"

# 7 — sentinels but no rows.
: >"${WORK}/rows.empty"
wrap "${WORK}/rows.empty" "${WORK}/empty.manifest" 0
check "empty manifest" "${WORK}/empty.manifest" 2 "the manifest is empty"

if [ "${failures}" -gt 0 ]; then
    echo >&2
    echo "ERROR: ${failures} fixture(s) did not reproduce the expected gate behaviour." >&2
    exit 1
fi
echo "OK: the byte-model parity gate fires on every fixture, with the right reason."
