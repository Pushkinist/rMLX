#!/usr/bin/env bash
# scripts/check_kv_codec_disposition.sh — CI gate: what the user-facing surfaces
# say a KV codec does, and what the runtime classifiers say it does, agree.
#
# WHY
#   Seventeen of the twenty-eight `KvQuant` variants never run. Their decode
#   reads the bf16 mirror on both axes, so `exit_prefill` skips the encode and
#   calls `storage.clear_payload()`: at runtime they are byte-identical to
#   `--kv-quant none` in both resident bytes and generated tokens. That is not
#   visible from the codec's name, its `docs/KV_QUANT.md` section (which
#   describes a packed store nothing builds), or a `--help` line that lists it
#   beside a live codec. An operator who reads the help, picks a name and
#   reads back the resolved-codec log line has no way to learn it did nothing.
#
#   The disposition is not a property anyone can keep in their head: it moves
#   the day a codec grows a decode kernel over its own store, and it moves one
#   codec at a time. So the surfaces are checked against the code, not reviewed.
#
# ORACLE
#   `cargo test -p rmlx-kv-quant --lib emit_kv_codec_disposition_manifest`
#   prints one line per codec, swept over `ALL_KV_QUANTS` and classified by
#   `KvQuant::materialises_packed_store()` — the disjunction of the three
#   predicates `exit_prefill` gates the encode on:
#
#       decode_reads_packed_store()  feeds_bf16_k_at_decode()  feeds_bf16_v_at_decode()
#
#   `ALL_KV_QUANTS`'s completeness is pinned by
#   `all_kv_quants_names_every_variant_once` against the compiler-checked
#   `variant_index`, so a new enum variant cannot slip past this gate by being
#   absent from a list. Nothing here is hand-written per codec.
#
# RULE 1 (CLI help, coverage)
#   Every inert codec named anywhere in the `--kv-quant` / `--kv-bits` help
#   text must also appear inside an INERT block in that text. A name that is
#   only listed beside the live codecs is a name the help says works.
#
# RULE 2 (CLI help, converse)
#   Nothing inside an INERT block may be a codec that is NOT inert. A block
#   that over-claims retires a working codec in the reader's head, and it is
#   how the block goes stale when a codec is wired up.
#
# RULE 3 (docs, coverage)
#   Every inert codec must be named in at least one INERT banner in
#   docs/KV_QUANT.md. The per-variant sections describe pack formats and bit
#   rates in the present tense; without the banner they describe a store the
#   codec does not build.
#
# RULE 4 (docs, converse)
#   Nothing named in a banner may be a codec that is NOT inert.
#
# RULE 5 (docs, placement)
#   A banner must open within 3 lines of a `### ` heading — it belongs at the
#   head of the section it qualifies, not buried in one.
#
# RULE 6 (the help is actually reached)
#   Every `--kv-quant` argument in the CLI must take its help from the shared
#   constants. Five copies drift; one that is checked does not.
#
# SCOPE
#   Rules 3-5 check that an inert codec carries a banner *somewhere* in
#   docs/KV_QUANT.md under a section heading, not that it is the heading of its
#   own section — headings in that file are prose and not machine-addressable.
#
# EXIT CODES
#   0  gate ran and passed
#   1  gate ran and found a violation
#   2  gate could not run (build failure, missing file, unparseable surface)
#      — never conflated with 1, because "no violations found" and "nothing was
#      looked at" are the same exit code in a naive script and the second one
#      passes silently forever.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_MAIN="${REPO_ROOT}/crates/rmlx-cli/src/main.rs"
KV_DOC="${REPO_ROOT}/docs/KV_QUANT.md"

BANNER_MARKER='[*][*]INERT on this build[*][*]'
HELP_INERT_MARKER='^[[:space:]]*INERT[[:space:]]*—'

die_env() {
    echo "ERROR (gate could not run): $*" >&2
    exit 2
}
die_violation() {
    echo "ERROR: $*" >&2
    exit 1
}

for f in "${CLI_MAIN}" "${KV_DOC}"; do
    [ -f "$f" ] || die_env "missing ${f#"${REPO_ROOT}/"}"
done

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# ── Oracle: the disposition manifest, derived from the type ──────────────────
if ! command -v cargo >/dev/null 2>&1; then
    die_env "cargo not on PATH — the disposition manifest comes from the crate"
fi

(
    cd "${REPO_ROOT}" || exit 1
    cargo test -q -p rmlx-kv-quant --lib -- \
        --exact quant::quant_tests::emit_kv_codec_disposition_manifest --nocapture
) >"${WORK}/manifest.raw" 2>"${WORK}/manifest.err"
cargo_status=$?

if ! grep -q '^KVQUANT-DISPOSITION-BEGIN$' "${WORK}/manifest.raw"; then
    # Never reached the emitter: build failure, filtered-away test, wrong path.
    echo "--- cargo stdout ---" >&2
    tail -n 30 "${WORK}/manifest.raw" >&2
    echo "--- cargo stderr ---" >&2
    tail -n 30 "${WORK}/manifest.err" >&2
    die_env "the disposition manifest did not run (cargo exit ${cargo_status})"
fi

if [ "${cargo_status}" -ne 0 ]; then
    # Reached the emitter and then failed: the derivation itself is broken
    # (duplicate surface stem, empty stem). That is a violation, not an
    # environment problem, and the two must not share an exit code.
    tail -n 30 "${WORK}/manifest.raw" >&2
    die_violation "the disposition manifest itself failed — see above"
fi

grep '^KVQUANT-DISPOSITION	' "${WORK}/manifest.raw" >"${WORK}/manifest" || true
declared=$(grep '^KVQUANT-DISPOSITION-END	' "${WORK}/manifest.raw" | head -1 | cut -f2)
actual=$(grep -c '^KVQUANT-DISPOSITION	' "${WORK}/manifest.raw")

[ -n "${declared}" ] || die_env "the manifest printed no END sentinel"
[ "${declared}" = "${actual}" ] ||
    die_env "manifest truncated: END says ${declared} codecs, read ${actual}"
[ "${actual}" -gt 0 ] || die_env "the manifest is empty"

# ── Surface 1: the CLI help constants ────────────────────────────────────────
# Extract each named `const X: &str = "..."` body. The consts are the whole
# `--kv-quant` / `--kv-bits` help surface: rule 6 pins that the arguments take
# their text from them and nowhere else.
extract_const() {
    awk -v name="$1" '
        $0 ~ ("^const " name ": &str = ") { inside = 1; next }
        inside { print }
        inside && /";$/ { exit }
    ' "${CLI_MAIN}"
}

: >"${WORK}/help.txt"
for c in KV_QUANT_HELP KV_QUANT_LONG_HELP KV_BITS_LONG_HELP; do
    extract_const "$c" >"${WORK}/const.$c"
    [ -s "${WORK}/const.$c" ] ||
        die_env "could not extract const ${c} from crates/rmlx-cli/src/main.rs"
    cat "${WORK}/const.$c" >>"${WORK}/help.txt"
done

# The INERT blocks inside the help: from a marker line to the next blank line.
awk '
    /^[[:space:]]*INERT[[:space:]]*—/ { inside = 1 }
    inside && /^[[:space:]]*$/        { inside = 0 }
    inside                            { print }
' "${WORK}/help.txt" >"${WORK}/help_inert.txt"

if [ ! -s "${WORK}/help_inert.txt" ]; then
    die_violation "the --kv-quant/--kv-bits help declares no INERT block.
Seventeen of the codecs it can name do nothing at runtime. If that stopped
being true, this gate's manifest would say so — check its output first.
A block opens with a line matching: ${HELP_INERT_MARKER}"
fi

# Rule 6: every --kv-quant argument reaches the shared constants.
args=$(grep -cE '^ +kv_quant: (String|Option<String>),$' "${CLI_MAIN}")
short=$(grep -c 'help = KV_QUANT_HELP' "${CLI_MAIN}")
long=$(grep -c 'long_help = KV_QUANT_LONG_HELP' "${CLI_MAIN}")
if [ "${args}" -eq 0 ]; then
    die_env "found no --kv-quant argument declarations to check"
fi
if [ "${args}" != "${short}" ] || [ "${args}" != "${long}" ]; then
    die_violation "${args} --kv-quant arguments, but ${short} use \
help = KV_QUANT_HELP and ${long} use long_help = KV_QUANT_LONG_HELP.
An argument with its own help text is one this gate does not read."
fi

# ── Surface 2: the docs banners ──────────────────────────────────────────────
# A banner is a maximal run of consecutive `> ` lines whose first line carries
# the marker. Rule 5 (placement) is checked in the same pass.
awk -v marker="${BANNER_MARKER}" '
    /^### / { last_heading = NR }
    /^> / {
        if (!inside) {
            if ($0 ~ marker) {
                inside = 1
                if (last_heading == 0 || NR - last_heading > 3) {
                    printf "PLACEMENT\t%d\n", NR
                }
            } else {
                next
            }
        }
        print "BANNER\t" $0
        next
    }
    { inside = 0 }
' "${KV_DOC}" >"${WORK}/banners.raw" ||
    die_env "the banner scan of docs/KV_QUANT.md failed"


grep '^PLACEMENT	' "${WORK}/banners.raw" >"${WORK}/placement" || true
grep '^BANNER	' "${WORK}/banners.raw" | cut -f2- >"${WORK}/banners.txt" || true

if [ -s "${WORK}/placement" ]; then
    echo "ERROR: an INERT banner does not open within 3 lines of a '### ' heading:" >&2
    while IFS=$'\t' read -r _ line; do
        echo "  docs/KV_QUANT.md:${line}" >&2
    done <"${WORK}/placement"
    echo >&2
    echo "A banner qualifies the section it heads. Buried in the body it is a" >&2
    echo "remark; at the head it is the first thing the reader sees." >&2
    exit 1
fi

if [ ! -s "${WORK}/banners.txt" ]; then
    die_violation "docs/KV_QUANT.md carries no INERT banner at all — every \
per-variant section for an inert codec needs one (marker: **INERT on this build**)"
fi

# ── The four content rules ───────────────────────────────────────────────────
# Match a stem on a word boundary. `iso3` must not match inside `iso3_sym`, and
# `k8vturbo2` must not match inside `k8vturbo2tcq`, so an EXACT stem is fenced
# on both sides and a PREFIX stem on the left only.
match_stem() {
    local stem="$1" mode="$2" file="$3"
    local pattern
    if [ "${mode}" = "EXACT" ]; then
        pattern="(^|[^A-Za-z0-9_])${stem}([^A-Za-z0-9_]|\$)"
    else
        pattern="(^|[^A-Za-z0-9_])${stem}"
    fi
    grep -qE -- "${pattern}" "${file}"
}

violations=0
report() {
    echo "  $*" >&2
    violations=$((violations + 1))
}

inert_count=0
while IFS=$'\t' read -r _ _idx display stem mode class _rs _bk _bv; do
    case "${class}" in
        INERT)
            inert_count=$((inert_count + 1))
            # Rule 1: named in the help at all → must be in an INERT block.
            if match_stem "${stem}" "${mode}" "${WORK}/help.txt" &&
                ! match_stem "${stem}" "${mode}" "${WORK}/help_inert.txt"; then
                report "RULE 1  '${display}' is inert but the CLI help names it outside the INERT block"
            fi
            # Rule 3: must carry a docs banner.
            if ! match_stem "${stem}" "${mode}" "${WORK}/banners.txt"; then
                report "RULE 3  '${display}' is inert but no docs/KV_QUANT.md INERT banner names it"
            fi
            ;;
        LIVE | BASELINE)
            # Rule 2 / Rule 4: the converse, on both surfaces.
            if match_stem "${stem}" "${mode}" "${WORK}/help_inert.txt"; then
                report "RULE 2  '${display}' is ${class}, not inert, but the CLI help's INERT block names it"
            fi
            if match_stem "${stem}" "${mode}" "${WORK}/banners.txt"; then
                report "RULE 4  '${display}' is ${class}, not inert, but a docs/KV_QUANT.md INERT banner names it"
            fi
            ;;
        *)
            die_env "unknown disposition class '${class}' for '${display}'"
            ;;
    esac
done <"${WORK}/manifest"

if [ "${violations}" -gt 0 ]; then
    echo >&2
    echo "ERROR: ${violations} disposition mismatch(es) between the code and the" >&2
    echo "user-facing surfaces. The code is the oracle — either the codec's" >&2
    echo "classification moved (fix the help and docs/KV_QUANT.md to match), or" >&2
    echo "a surface was written from the codec's name instead of its behaviour." >&2
    echo >&2
    echo "Manifest as derived (index, name, stem, match, class, reads_store, bf16_k, bf16_v):" >&2
    cat "${WORK}/manifest" >&2
    exit 1
fi

echo "OK: ${actual} KV codecs classified from the type (${inert_count} inert); \
CLI help and docs/KV_QUANT.md agree with all of them."
