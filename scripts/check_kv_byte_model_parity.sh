#!/usr/bin/env bash
# scripts/check_kv_byte_model_parity.sh — CI gate: the KV byte model in
# `scripts/perf_ceiling.py` agrees, byte for byte, with the one the engine uses.
#
# USAGE
#   check_kv_byte_model_parity.sh [MANIFEST]
#
#   With no argument it derives the manifest by running the emitter test. With a
#   MANIFEST file it reads a pre-captured one instead — how
#   `check_kv_byte_model_parity_fixtures.sh` drives it, one mutation at a time.
#
# WHY
#   `perf_ceiling.py` prices a codec without building the workspace, so it
#   carries a second copy of `estimated_resident_bytes_per_layer`,
#   `packed_side_bytes`, `feeds_bf16_{k,v}_at_decode` and `boundary_floor`. A
#   second producer with no gate between it and the first is not a risk, it is a
#   schedule: that copy came to model a 4-byte iso/rotor ring sideband after the
#   store narrowed to 2 (iso 16.25 bits per value against a stored 12.125 — on
#   the wrong side of bf16), and to charge the `Mixed` family two bf16 mirrors on
#   architectures that stopped keeping them (22.50 against 6.50). Both drifts
#   changed which codec the figures recommend, and neither failed anything.
#
#   The engine is the oracle. This gate does not check that either model is
#   *right*; it checks that there is effectively one of them.
#
# ORACLE
#   `cargo test -p rmlx-models --lib emit_kv_byte_model_manifest` prints one
#   `KVBYTES` row per (codec, topology, shape) and one `KVFLOOR` row per (codec,
#   topology), swept over `ALL_KV_QUANTS`. That list's completeness is pinned
#   against the compiler-checked `variant_index`, so a new codec reaches this
#   gate without anyone adding it to a list here.
#
#   `perf_ceiling.py --byte-model` reads that manifest on stdin and re-emits it
#   from its own arithmetic. The codec sweep, both topologies, the shapes and
#   the layer count therefore come from the Rust side alone — the Python half
#   chooses nothing about what is covered, only what each row's value is.
#
# EXIT CODES
#   0  gate ran and passed
#   1  gate ran and found a disagreement
#   2  gate could not run (build failure, missing file, no python3)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_SRC="${1:-}"
PERF_CEILING="${REPO_ROOT}/scripts/perf_ceiling.py"

die_env() {
    echo "ERROR (gate could not run): $*" >&2
    exit 2
}
die_violation() {
    echo "ERROR: $*" >&2
    exit 1
}

[ -f "${PERF_CEILING}" ] || die_env "missing scripts/perf_ceiling.py"
command -v python3 >/dev/null 2>&1 ||
    die_env "python3 not on PATH — the second byte model is a python script"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# ── Oracle: the engine's byte model, derived from the type ───────────────────
if [ -n "${MANIFEST_SRC}" ]; then
    [ -f "${MANIFEST_SRC}" ] || die_env "missing manifest ${MANIFEST_SRC}"
    cat "${MANIFEST_SRC}" >"${WORK}/manifest.raw"
    : >"${WORK}/manifest.err"
    cargo_status=0
else
    command -v cargo >/dev/null 2>&1 ||
        die_env "cargo not on PATH — the byte model comes from the crate"
    (
        cd "${REPO_ROOT}" || exit 1
        cargo test -q -p rmlx-models --lib -- \
            --exact kv_cache::tests::tests::emit_kv_byte_model_manifest --nocapture
    ) >"${WORK}/manifest.raw" 2>"${WORK}/manifest.err"
    cargo_status=$?
fi

if ! grep -q '^KVBYTES-BEGIN$' "${WORK}/manifest.raw"; then
    echo "--- manifest stdout ---" >&2
    tail -n 30 "${WORK}/manifest.raw" >&2
    echo "--- manifest stderr ---" >&2
    tail -n 30 "${WORK}/manifest.err" >&2
    die_env "the byte-model manifest did not run (emitter exit ${cargo_status})"
fi

if [ "${cargo_status}" -ne 0 ]; then
    # Reached the emitter and then failed: the derivation itself is broken.
    # That is a violation, not an environment problem.
    tail -n 30 "${WORK}/manifest.raw" >&2
    die_violation "the byte-model manifest itself failed — see above"
fi

grep -E '^KV(BYTES|FLOOR)	' "${WORK}/manifest.raw" >"${WORK}/rust.tsv" || true
declared=$(grep '^KVBYTES-END	' "${WORK}/manifest.raw" | head -1 | cut -f2)
actual=$(awk 'END { print NR }' "${WORK}/rust.tsv")

[ -n "${declared}" ] || die_env "the manifest printed no END sentinel"
[ "${declared}" = "${actual}" ] ||
    die_env "manifest truncated: END says ${declared} rows, read ${actual}"
[ "${actual}" -gt 0 ] || die_env "the manifest is empty"

codecs=$(cut -f2 "${WORK}/rust.tsv" | sort -u | awk 'END { print NR }')
topologies=$(cut -f3 "${WORK}/rust.tsv" | sort -u | awk 'END { print NR }')
shapes=$(awk -F'\t' '$1 == "KVBYTES" { print $5 "x" $6 }' "${WORK}/rust.tsv" |
    sort -u | awk 'END { print NR }')
[ "${topologies}" -eq 2 ] ||
    die_env "the manifest covers ${topologies} topologies, not both — \
shares_kv moves the Mixed family by two whole mirrors and a one-sided sweep \
would not see it"
[ "${shapes}" -ge 2 ] ||
    die_env "the manifest covers ${shapes} head-dim shape(s) — the ring and \
rotor cadences carry a per-row term, and a single shape cannot tell a wrong \
sideband width from a wrong per-row term"

# ── The second model, re-emitting the same rows ──────────────────────────────
if ! python3 "${PERF_CEILING}" --byte-model \
    <"${WORK}/rust.tsv" >"${WORK}/py.tsv" 2>"${WORK}/py.err"; then
    tail -n 20 "${WORK}/py.err" >&2
    die_env "perf_ceiling.py --byte-model failed to run"
fi

py_rows=$(awk 'END { print NR }' "${WORK}/py.tsv")
[ "${py_rows}" -eq "${actual}" ] ||
    die_env "perf_ceiling.py re-emitted ${py_rows} of ${actual} rows — it \
dropped rows rather than disagreeing with them, so a diff would under-report"

if ! diff -u "${WORK}/rust.tsv" "${WORK}/py.tsv" >"${WORK}/parity.diff"; then
    echo "ERROR: scripts/perf_ceiling.py disagrees with the engine's byte model." >&2
    echo >&2
    echo "  '-' is the engine (crates/rmlx-kv-quant, crates/rmlx-models)." >&2
    echo "  '+' is scripts/perf_ceiling.py." >&2
    echo "  Columns: kind, codec, shares_kv, then the shape, then the value." >&2
    echo >&2
    grep -E '^[-+]KV' "${WORK}/parity.diff" >&2
    echo >&2
    echo "The engine is the oracle. Move the script, not the engine." >&2
    exit 1
fi

echo "OK: ${actual} byte-model rows (${codecs} KV codecs x ${topologies} \
topologies x ${shapes} shapes) agree between the engine and \
scripts/perf_ceiling.py."
