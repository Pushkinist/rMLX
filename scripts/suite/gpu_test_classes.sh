#!/usr/bin/env bash
# Print the half (codec|rest) or threadgroup setting (on|off) of each classified GPU test.
#
# Usage:  bash scripts/suite/gpu_test_classes.sh half [--root <dir>]
#         bash scripts/suite/gpu_test_classes.sh threadgroup [--root <dir>]
#         half         codec|rest, crate, test: the partition `run_gpu_tests.sh --half` runs one side of
#         threadgroup  on|off, crate, test: the threadgroup-memory validation setting of each test
#         --root       the tree to classify; points the tool at a synthetic fixture workspace
# Exit:   0 the rows, on stdout; 2 they could not be computed, with the reason named
# Set:    -uo — a failed classifier listing or codec-name read is tested and refused, not fatal
# Not a mode of: check/gpu_tests.sh — that tool classifies the population and judges it (exit 1 on a
#                violation); this one reads the finished listing and judges nothing, so it has no exit 1
#
# One reader of the population, two predicates. Both modes print the same row shape, one row per
# classified test, and `scripts/suite/run_gpu_tests.sh` is the only consumer. A partition is only a gate if
# it is computed in ONE place: two readers deriving it separately would eventually disagree, and a
# test both of them called the other's would run under neither.
#
# HALF
#   The whole GPU suite is hours of wall, and a change pays for all of it to learn about the part
#   it touched, so the suite is partitioned in two and `run_gpu_tests.sh --half <name>` runs one.
#
#   A classified GPU test is in the `codec` half if and only if its declaring file is
#     * under a workspace member that `rmlx-models` depends on, or
#     * under the model layer's own `src/`, or
#     * in a file that SELECTS a KV codec: it names `DEFAULT_KV_QUANT`, or a `KvQuant::<V>` whose
#       `<V>` is a variant of `ALL_KV_QUANTS` other than `None`.
#   Every other classified GPU test is in the `rest` half.
#
#   The third clause makes this a partition by what a change can BREAK rather than by build
#   topology: a file that selected a codec has chosen one, and a codec change can move what it
#   asserts. A file naming only `None` has pinned the codec off to measure something else (a
#   drafter, a router, a CLI flag) and is in `rest`. The clause reads the FILE, not the test: a
#   file sweeping `Mixed` and `None` is one codec suite, and splitting it would put two cells of
#   one sweep in two gates.
#
#   Every fact is read from the tree and none is a test name: the member list from the workspace
#   manifest, the dependency edge from the model layer's manifest, the declaring file from the
#   classifier, the codec names from `ALL_KV_QUANTS`. A test that moves file moves half; a test
#   that is renamed does not.
#
#   The codec names are derived, not a pattern. A bare `KvQuant::[A-Za-z0-9_]+` needle matches
#   `KvQuant::FromStr` and `KvQuant::materialises_packed_store`, a trait path and a method, neither
#   of which names a codec. So the names come from `ALL_KV_QUANTS`, the list a codec must already be
#   on to be constructible anywhere. An EMPTY derived set is a hard refusal, not an empty match: a
#   reader that found zero codec names would place every integration binary in `rest` and report a
#   clean partition while the codec half guarded nothing.
#
#   Both the list and the needle read the line's CODE, comments removed and string bodies blanked,
#   through `scripts/lib/awk_text.sh`, as the sibling source gates do. A commented-out
#   `KvQuant::K8V8` is a codec the file no longer selects, and a codec named inside a string literal
#   is text a program prints rather than a codec it constructs.
#
#   `DEFAULT_KV_QUANT` is in the needle for the same reason. The suites that select their codec
#   through it name no variant, so they sit in `rest` only for as long as that default is `None`.
#   Reading the constant instead of its current value keeps the partition from moving silently when
#   the default does.
#
#   The codec names always come from this checkout's codec crate, because a fixture tree carries no
#   codec crate source and an empty set is a refusal.
#
# THREADGROUP
#   `MTL_SHADER_VALIDATION_THREADGROUP_MEMORY=1` makes MLX's own NAX kernels return wrong values,
#   differently on each run: the routed-expert `gather_qmm` (`affine_gather_qmm_rhs_nax`) in a
#   Qwen3.6 MoE prefill, and the f32 `affine_qmm_t_nax` in a PARO forward. With the threadgroup
#   instrumentation off and everything else on, the same calls are bit-identical to an unvalidated
#   run on every repeat. So a test whose GPU work is MLX's kernels cannot be judged with it on, and
#   rMLX's own `.metal` kernels, the ones this repo can get wrong in threadgroup memory, keep it.
#
#   A DISPATCHER is a source file that embeds, with `include_str!`, a `.metal` file under a
#   directory `scripts/lib/metal_dirs.sh` names. Its ENTRY NAMES are the names of the fns a non-test
#   dispatcher defines. A classified GPU test runs with threadgroup validation `on` if and only if
#     * its declaring file is itself a dispatcher, or
#     * its body, or the body of a fn in the same file it calls (followed transitively, by name),
#       names an entry name.
#   Every other classified GPU test runs with it `off`. Comments and the contents of string
#   literals are not read. Every fact is read from the tree and none is a test name.
#
#   What it cannot see:
#     * A test that reaches an rMLX kernel only through production code (a `KvCache` update, a
#       storage append, a model forward) names no entry and runs `off`. Following production calls
#       would put every checkpoint test `on`, since a model forward reaches the gated-delta and KV
#       kernels.
#     * A helper in another file (`tests/common`) is not followed.
#     * An entry name is matched by name, not by path: a test that names a dispatcher's helper
#       (`dtype_tag`, `is_supported_d`) without dispatching runs `on`.
#
#   A classification in which either setting is empty is refused: the point is the split, and a
#   reader that put every test on one side would run the suite with one setting and report a clean
#   split.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${REPO_ROOT}"

# shellcheck source=../lib/awk_text.sh
. "${REPO_ROOT}/scripts/lib/awk_text.sh"
# shellcheck source=../lib/workspace_members.sh
. "${REPO_ROOT}/scripts/lib/workspace_members.sh"

USAGE="Usage: gpu_test_classes.sh {half|threadgroup} [--root <dir>]"

# The classified tests, `<crate><TAB><test><TAB><declaring file>` per line, in ${listing}. The
# classifier's stderr is shown only when it fails: on success its NOTE blocks are the runner's own
# listing's to print, and repeating them here would show each twice. $1 names what is not computed
# from a listing that failed.
read_classified() {
    list_err="$(mktemp "${TMPDIR:-/tmp}/gpu_test_classes.XXXXXX")" || {
        echo "ERROR: cannot make a scratch file." >&2
        exit 2
    }
    trap 'rm -f "${list_err}"' EXIT
    listing="$(bash "${REPO_ROOT}/scripts/check/gpu_tests.sh" list --files --root "${ROOT}" 2>"${list_err}")"
    list_rc=$?
    if [ "${list_rc}" -ne 0 ]; then
        cat "${list_err}" >&2
        echo "ERROR: check/gpu_tests.sh list --files failed (exit ${list_rc}) for ${ROOT}; no $1 is computed from a listing it did not finish." >&2
        exit 2
    fi
    if [ -z "${listing}" ]; then
        echo "ERROR: check/gpu_tests.sh list --files produced no GPU tests for ${ROOT}." >&2
        exit 2
    fi
}

# The facts the half rule reads from the tree: the codec needle, the model layer and the member
# directories it depends on.
half_setup() {
    QUANT_RS="${REPO_ROOT}/crates/rmlx-kv-quant/src/quant.rs"
    if [ ! -f "${QUANT_RS}" ]; then
        echo "ERROR: ${QUANT_RS} not found — the codec names cannot be derived." >&2
        exit 2
    fi

    # The variants of ALL_KV_QUANTS, less `None`. Read from the const's own body so a codec added
    # to the enum but not to that list is not a codec this can select.
    codec_names="$(awk "${AWK_TEXT_FNS}"'
        { code = decomment($0) }
        /^pub const ALL_KV_QUANTS/ { in_list = 1; next }
        in_list && /^\];/ { in_list = 0 }
        in_list {
            # Every variant on the line, not the first: two on one line would otherwise narrow
            # the set silently, which is the empty-set refusal below arriving one name at a time.
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
    # A codec name is a whole identifier after `KvQuant::`. Without the trailing class
    # `KvQuant::Iso3Sym` would read as `Iso3`, and `KvQuant::NoneOfIt` as a codec; ERE is
    # leftmost-longest, so the alternation needs no ordering.
    CODEC_NEEDLE="(DEFAULT_KV_QUANT([^A-Za-z0-9_]|\$)|KvQuant::(${codec_alt})([^A-Za-z0-9_]|\$))"

    CARGO_TOML="${ROOT}/Cargo.toml"
    if [ ! -f "${CARGO_TOML}" ]; then
        echo "ERROR: ${CARGO_TOML} not found — cannot resolve workspace members." >&2
        exit 2
    fi

    # Same one-member-per-line array the classifier reads; a layout change that narrows the parse
    # fails there first.
    members=()
    while IFS= read -r m; do
        members+=("$m")
    done < <(workspace_members "${CARGO_TOML}")

    if [ ${#members[@]} -eq 0 ]; then
        echo "ERROR: parsed 0 workspace members from ${CARGO_TOML}." >&2
        exit 2
    fi

    # `<pkg name><TAB><member dir>` per member, and the model layer's own dir.
    member_dirs=""
    models_dir=""
    for m in ${members[@]+"${members[@]}"}; do
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

    # The model layer's [dependencies] section, the edge the first clause reads. dev-dependencies
    # are deliberately out: a dev-dep is what the model layer's own tests use, not what it stands on.
    models_deps="$(awk '
        /^\[dependencies\]/ { in_deps = 1; next }
        /^\[/ { in_deps = 0 }
        in_deps && match($0, /^[A-Za-z_][A-Za-z0-9_-]*/) {
            print substr($0, RSTART, RLENGTH)
        }
    ' "${ROOT}/${models_dir}/Cargo.toml")"

    # The member directories the first clause covers, one per line.
    codec_dirs=""
    while IFS=$'\t' read -r pkg dir; do
        [ -n "${pkg}" ] || continue
        case $'\n'"${models_deps}"$'\n' in
            *$'\n'"${pkg}"$'\n'*) codec_dirs="${codec_dirs}${dir}"$'\n' ;;
        esac
    done <<< "${member_dirs}"
}

half_rows() {
    # One grep per declaring file, not per test: the clause is a property of the file, and the
    # classification names a file once per test it declares.
    selects_codec=""
    while IFS= read -r file; do
        [ -n "${file}" ] || continue
        # awk exits 1 when the file names no codec; any other status is a file it could not read.
        awk_rc=0
        awk -v NEEDLE="${CODEC_NEEDLE}" "${AWK_TEXT_FNS}"'
            blank_strings(decomment($0)) ~ NEEDLE { found = 1 }
            END { exit(found ? 0 : 1) }
        ' "${file}" || awk_rc=$?
        case "${awk_rc}" in
            0) selects_codec="${selects_codec}${file}"$'\n' ;;
            1) ;;
            *)
                echo "ERROR: could not read ${file} to see whether it selects a codec." >&2
                exit 2
                ;;
        esac
    done < <(printf '%s\n' "${listing}" | cut -f3 | sort -u)

    while IFS=$'\t' read -r crate test file; do
        [ -n "${crate}" ] || continue
        rel="${file#"${ROOT}"/}"
        half="rest"
        case "${rel}" in
            "$(member_prefix "${models_dir}")src/"*) half="codec" ;;
        esac
        if [ "${half}" = "rest" ]; then
            while IFS= read -r dir; do
                [ -n "${dir}" ] || continue
                if [ "${dir}" = . ]; then
                    case "${rel}" in
                        src/* | tests/*) half="codec"; break ;;
                    esac
                else
                    case "${rel}" in
                        "${dir}/"*) half="codec"; break ;;
                    esac
                fi
            done <<< "${codec_dirs}"
        fi
        if [ "${half}" = "rest" ]; then
            case $'\n'"${selects_codec}" in
                *$'\n'"${file}"$'\n'*) half="codec" ;;
            esac
        fi
        printf '%s\t%s\t%s\n' "${half}" "${crate}" "${test}"
    done <<< "${listing}"
}

# The gated directories, resolved against the tree being classified.
threadgroup_setup() {
    METAL_DIRS=()
    local scripts_dir="${REPO_ROOT}/scripts"
    REPO_ROOT="${ROOT}"
    # shellcheck source=../lib/metal_dirs.sh
    . "${scripts_dir}/lib/metal_dirs.sh"
    REPO_ROOT="$(cd "${scripts_dir}/.." && pwd)"
    if [ ${#METAL_DIRS[@]} -eq 0 ]; then
        echo "ERROR: scripts/lib/metal_dirs.sh names no gated directory." >&2
        exit 2
    fi
}

threadgroup_rows() {
    # The program is read into a variable so stdin stays free for the listing.
    read -r -d '' SPLIT_PY <<'PY' || true
import os
import re
import sys

root, gated = sys.argv[1], [os.path.realpath(d) for d in sys.argv[2:]]
INCLUDE = re.compile(r'include_str!\(\s*"([^"]*\.metal)"\s*\)')
FN = re.compile(r'^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?'
                r'fn\s+([A-Za-z_][A-Za-z0-9_]*)')
IDENT = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def code(line):
    """The line with its `//` comment removed and string bodies blanked."""
    out, in_str, i = [], False, 0
    while i < len(line):
        c = line[i]
        if in_str:
            if c == '\\':
                out.append('  ')
                i += 2
                continue
            if c == '"':
                in_str = False
                out.append('"')
            else:
                out.append(' ')
            i += 1
            continue
        if c == '"':
            in_str = True
            out.append('"')
        elif line.startswith('//', i):
            break
        else:
            out.append(c)
        i += 1
    return ''.join(out)


def decomment(line):
    """The line with its `//` comment removed, string literals kept."""
    in_str, i = False, 0
    while i < len(line):
        c = line[i]
        if in_str and c == '\\':
            i += 2
            continue
        if c == '"':
            in_str = not in_str
        elif not in_str and line.startswith('//', i):
            return line[:i]
        i += 1
    return line


def refuse(message):
    sys.stderr.write(f'ERROR: {message}\n')
    sys.exit(2)


def read_lines(path):
    try:
        with open(path, errors='replace') as f:
            return f.read().split('\n')
    except OSError as e:
        refuse(f'cannot read {path}: {e.strerror}')


def embeds_gated_metal(path):
    for line in read_lines(path):
        for lit in INCLUDE.findall(decomment(line)):
            target = os.path.realpath(os.path.join(os.path.dirname(path), lit))
            if os.path.isfile(target) and any(target.startswith(g + os.sep) for g in gated):
                return True
    return False


def fn_bodies(path):
    """fn name -> list of body texts, by rustfmt layout: a fn closes on the
    first line that is its own indent followed by `}`."""
    lines = read_lines(path)
    bodies, i = {}, 0
    while i < len(lines):
        m = FN.match(lines[i])
        if not m:
            i += 1
            continue
        indent, name = m.group(1), m.group(2)
        body, j = [code(lines[i])], i + 1
        if not code(lines[i]).rstrip().endswith(('}', ';')):
            while j < len(lines) and lines[j].rstrip() != indent + '}':
                body.append(code(lines[j]))
                j += 1
        bodies.setdefault(name, []).append('\n'.join(body))
        i = j + 1
    return bodies


def is_test_file(path):
    base = os.path.basename(path)
    return base.endswith('_tests.rs') or base == 'tests.rs' or f'{os.sep}tests{os.sep}' in path


dispatchers = set()
entries = set()
for top, dirs, files in os.walk(os.path.join(root, 'crates'),
                                onerror=lambda e: refuse(f'cannot list {e.filename}: {e.strerror}')):
    dirs[:] = [d for d in dirs if d != 'target']
    for name in files:
        path = os.path.join(top, name)
        if name.endswith('.rs') and embeds_gated_metal(path):
            dispatchers.add(path)
            if not is_test_file(path):
                entries.update(fn_bodies(path))

if not dispatchers:
    refuse('no source embeds a .metal file from a gated directory, so no test can be placed on.')

rows, on, off, parsed = [], 0, 0, {}
for line in sys.stdin:
    parts = line.rstrip('\n').split('\t')
    if len(parts) != 3:
        continue
    crate, test, path = parts
    reached = path in dispatchers
    if not reached:
        if path not in parsed:
            parsed[path] = fn_bodies(path)
        bodies = parsed[path]
        seen, todo = set(), [test]
        while todo and not reached:
            fn = todo.pop()
            if fn in seen:
                continue
            seen.add(fn)
            for body in bodies.get(fn, []):
                tokens = set(IDENT.findall(body))
                if tokens & entries:
                    reached = True
                    break
                todo.extend(t for t in tokens if t in bodies and t not in seen)
    setting = 'on' if reached else 'off'
    on += reached
    off += not reached
    rows.append(f'{setting}\t{crate}\t{test}')

if on == 0 or off == 0:
    sys.stderr.write(f'ERROR: the threadgroup split collapsed — {on} test(s) on, {off} off.\n'
                     'Every classified GPU test would run with one setting while the split reads\n'
                     'as computed. See docs/GPU_TESTS.md.\n')
    sys.exit(2)
print('\n'.join(rows))
PY

    printf '%s\n' "${listing}" | python3 -c "${SPLIT_PY}" "${ROOT}" ${METAL_DIRS[@]+"${METAL_DIRS[@]}"}
}

mode="${1:-}"
case "${mode}" in
    half | threadgroup) shift ;;
    -h | --help)
        echo "${USAGE}"
        exit 0 ;;
    *) echo "ERROR: unknown mode '${mode}' (expected half or threadgroup)." >&2; exit 2 ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --root)
            if [ $# -lt 2 ]; then
                echo "ERROR: --root needs a value (expected --root <dir>)." >&2
                exit 2
            fi
            ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "${USAGE}"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1' (expected --root <dir>)." >&2; exit 2 ;;
    esac
done

if [ ! -d "${ROOT}" ]; then
    echo "ERROR: --root '${ROOT}' is not a directory." >&2
    exit 2
fi
ROOT="$(cd "${ROOT}" && pwd)"

case "${mode}" in
    half)
        half_setup
        read_classified partition
        half_rows ;;
    threadgroup)
        threadgroup_setup
        read_classified split
        threadgroup_rows ;;
esac
