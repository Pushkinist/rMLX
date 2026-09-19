#!/usr/bin/env bash
# scripts/gpu_test_halves.sh — which half of the GPU suite each classified test
# belongs to. One row per test, `<half><TAB><crate><TAB><test>`.
#
# WHY
#   The whole GPU suite is hours of wall, and a change pays for all of it to
#   learn about the part it touched. The suite is therefore partitioned in two,
#   and `scripts/run_gpu_tests.sh --half <name>` runs one of them. A partition
#   is only a gate if it is computed in ONE place: two readers deriving it
#   separately would eventually disagree, and a test both of them called the
#   other's would run under neither.
#
# THE RULE
#   A classified GPU test is in the `codec` half if and only if its declaring
#   file is
#     * under a workspace member that `rmlx-models` depends on, or
#     * under the model layer's own `src/`, or
#     * in a file that SELECTS a KV codec — it names `DEFAULT_KV_QUANT`, or a
#       `KvQuant::<V>` whose `<V>` is a variant of `ALL_KV_QUANTS` other than
#       `None`.
#   Every other classified GPU test is in the `rest` half.
#
#   The third clause is what makes this a partition by what a change can BREAK
#   rather than by build topology: a file that selected a codec has chosen one,
#   and a codec change can move what it asserts. A file naming only `None` has
#   pinned the codec off to measure something else — a drafter, a router, a CLI
#   flag — and is in `rest`. The clause reads the FILE, not the test: a file
#   sweeping `Mixed` and `None` is one codec suite, and splitting it would put
#   two cells of one sweep in two gates.
#
#   Every fact is read from the tree and none is a test name: the member list
#   from the workspace manifest, the dependency edge from the model layer's
#   manifest, the declaring file from the classifier, the codec names from
#   `ALL_KV_QUANTS`. A test that moves file moves half; a test that is renamed
#   does not.
#
# THE CODEC NAMES ARE DERIVED, NOT A PATTERN
#   A bare `KvQuant::[A-Za-z0-9_]+` needle is not a codec test: it matches
#   `KvQuant::FromStr` and `KvQuant::materialises_packed_store`, a trait path
#   and a method, neither of which names a codec. So the names come from
#   `ALL_KV_QUANTS` — the list a codec must already be on to be constructible
#   anywhere. An EMPTY derived set is a hard refusal, not an empty match: a
#   producer that read zero codec names would place every integration binary in
#   `rest` and report a clean partition while the codec half guarded nothing.
#
#   Both the list and the needle read the line's CODE — comments removed and
#   string bodies blanked, through `scripts/lib/awk_text.sh`, as the sibling
#   source gates do. A commented-out `KvQuant::K8V8` is a codec the file no
#   longer selects, and a codec named inside a string literal is text a program
#   prints rather than a codec it constructs; either one placing a whole
#   integration binary in the codec half would make the half a property of the
#   prose in it.
#
#   `DEFAULT_KV_QUANT` is in the needle for the same reason. The suites that
#   select their codec through it name no variant, so they sit in `rest` only
#   for as long as that default is `None`. Reading the constant instead of its
#   current value is what keeps the partition from moving silently when the
#   default does.
#
# USAGE
#   bash scripts/gpu_test_halves.sh
#   bash scripts/gpu_test_halves.sh --root <dir>
#
#   `--root` is the tree to CLASSIFY, forwarded to the classifier and read for
#   the manifests; it is what points this at a synthetic fixture workspace. The
#   codec names always come from this checkout's codec crate, because a fixture
#   tree carries no codec crate source and an empty set is a refusal.
#
# Exit 0 = the partition, on stdout. Exit 2 = it could not be computed, with
# the reason named; there is no exit 1, because this script judges nothing.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${REPO_ROOT}"

# shellcheck source=lib/awk_text.sh
. "${REPO_ROOT}/scripts/lib/awk_text.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
        -h|--help)
            echo "Usage: gpu_test_halves.sh [--root <dir>]"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1' (expected --root <dir>)." >&2; exit 2 ;;
    esac
done

if [ ! -d "${ROOT}" ]; then
    echo "ERROR: --root '${ROOT}' is not a directory." >&2
    exit 2
fi
ROOT="$(cd "${ROOT}" && pwd)"

QUANT_RS="${REPO_ROOT}/crates/rmlx-kv-quant/src/quant.rs"
if [ ! -f "${QUANT_RS}" ]; then
    echo "ERROR: ${QUANT_RS} not found — the codec names cannot be derived." >&2
    exit 2
fi

# The variants of ALL_KV_QUANTS, less `None`. Read from the const's own body so
# a codec added to the enum but not to that list is not a codec this can select
# — it is not constructible through the list either.
codec_names="$(awk "${AWK_TEXT_FNS}"'
    { code = decomment($0) }
    /^pub const ALL_KV_QUANTS/ { in_list = 1; next }
    in_list && /^\];/ { in_list = 0 }
    in_list {
        # Every variant on the line, not the first: two on one line would
        # otherwise narrow the set silently, which is the empty-set refusal
        # below arriving one name at a time.
        while (match(code, /KvQuant::[A-Za-z_][A-Za-z0-9_]*/)) {
            name = substr(code, RSTART + 9, RLENGTH - 9)
            if (name != "None") print name
            code = substr(code, RSTART + RLENGTH)
        }
    }
' "${QUANT_RS}" | sort -u)"

if [ -z "${codec_names}" ]; then
    echo "ERROR: derived 0 codec names from ALL_KV_QUANTS in ${QUANT_RS}." >&2
    echo "Every file would then select no codec and the codec half would guard" >&2
    echo "nothing while reporting a clean partition; refusing to emit one." >&2
    exit 2
fi

codec_alt="$(printf '%s' "${codec_names}" | tr '\n' '|' | sed 's/|$//')"
# A codec name is a whole identifier after `KvQuant::`. Without the trailing
# class `KvQuant::Iso3Sym` would read as `Iso3`, and `KvQuant::NoneOfIt` as a
# codec; ERE is leftmost-longest, so the alternation needs no ordering.
CODEC_NEEDLE="(DEFAULT_KV_QUANT([^A-Za-z0-9_]|\$)|KvQuant::(${codec_alt})([^A-Za-z0-9_]|\$))"

CARGO_TOML="${ROOT}/Cargo.toml"
if [ ! -f "${CARGO_TOML}" ]; then
    echo "ERROR: ${CARGO_TOML} not found — cannot resolve workspace members." >&2
    exit 2
fi

# Same one-member-per-line array the classifier reads; a layout change that
# narrows the parse fails there first.
members=()
while IFS= read -r m; do
    [ -n "$m" ] && members+=("$m")
done < <(awk '
    /^[[:space:]]*members[[:space:]]*=[[:space:]]*\[/ { inm = 1; next }
    inm {
        line = $0
        if (line ~ /\]/) { inm = 0 }
        gsub(/[",]/, "", line)
        gsub(/[[:space:]]/, "", line)
        gsub(/\[/, "", line); gsub(/\]/, "", line)
        if (line != "" && line !~ /^#/) { print line }
    }
' "${CARGO_TOML}")

if [ ${#members[@]} -eq 0 ]; then
    echo "ERROR: parsed 0 workspace members from ${CARGO_TOML}." >&2
    exit 2
fi

# `<pkg name><TAB><member dir>` per member, and the model layer's own dir.
member_dirs=""
models_dir=""
for m in "${members[@]}"; do
    manifest="${ROOT}/${m}/Cargo.toml"
    [ -f "${manifest}" ] || continue
    pkg="$(awk -F'"' '/^name[[:space:]]*=/{print $2; exit}' "${manifest}")"
    [ -n "${pkg}" ] || continue
    member_dirs="${member_dirs}${pkg}"$'\t'"${m}"$'\n'
    [ "${pkg}" = "rmlx-models" ] && models_dir="${m}"
done

if [ -z "${models_dir}" ]; then
    echo "ERROR: no workspace member is named rmlx-models under ${ROOT}." >&2
    echo "The half rule is stated relative to the model layer; without it there" >&2
    echo "is no partition to compute." >&2
    exit 2
fi

# The model layer's [dependencies] section — the edge the rule's first clause
# reads. dev-dependencies are deliberately out: a dev-dep is what the model
# layer's own tests use, not what it stands on.
models_deps="$(awk '
    /^\[dependencies\]/ { in_deps = 1; next }
    /^\[/ { in_deps = 0 }
    in_deps && match($0, /^[A-Za-z_][A-Za-z0-9_-]*/) {
        print substr($0, RSTART, RLENGTH)
    }
' "${ROOT}/${models_dir}/Cargo.toml")"

# The member directories the rule's first clause covers, one per line.
codec_dirs=""
while IFS=$'\t' read -r pkg dir; do
    [ -n "${pkg}" ] || continue
    case $'\n'"${models_deps}"$'\n' in
        *$'\n'"${pkg}"$'\n'*) codec_dirs="${codec_dirs}${dir}"$'\n' ;;
    esac
done <<< "${member_dirs}"

listing="$(bash "${REPO_ROOT}/scripts/check_gpu_tests_ignored.sh" --list-files --root "${ROOT}" 2>/dev/null)"
if [ -z "${listing}" ]; then
    echo "ERROR: check_gpu_tests_ignored.sh --list-files produced no GPU tests for ${ROOT}." >&2
    exit 2
fi

# One grep per declaring file, not per test: the clause is a property of the
# file, and the classification names a file once per test it declares.
selects_codec=""
while IFS= read -r file; do
    [ -n "${file}" ] || continue
    if awk -v NEEDLE="${CODEC_NEEDLE}" "${AWK_TEXT_FNS}"'
        blank_strings(decomment($0)) ~ NEEDLE { found = 1 }
        END { exit(found ? 0 : 1) }
    ' "${file}" 2>/dev/null; then
        selects_codec="${selects_codec}${file}"$'\n'
    fi
done < <(printf '%s\n' "${listing}" | cut -f3 | sort -u)

while IFS=$'\t' read -r crate test file; do
    [ -n "${crate}" ] || continue
    rel="${file#"${ROOT}"/}"
    half="rest"
    case "${rel}" in
        "${models_dir}/src/"*) half="codec" ;;
    esac
    if [ "${half}" = "rest" ]; then
        while IFS= read -r dir; do
            [ -n "${dir}" ] || continue
            case "${rel}" in
                "${dir}/"*) half="codec"; break ;;
            esac
        done <<< "${codec_dirs}"
    fi
    if [ "${half}" = "rest" ]; then
        case $'\n'"${selects_codec}" in
            *$'\n'"${file}"$'\n'*) half="codec" ;;
        esac
    fi
    printf '%s\t%s\t%s\n' "${half}" "${crate}" "${test}"
done <<< "${listing}"
