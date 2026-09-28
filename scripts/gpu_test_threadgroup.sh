#!/usr/bin/env bash
# scripts/gpu_test_threadgroup.sh — which classified GPU tests run with Metal's
# threadgroup-memory validation. One row per test, `<on|off><TAB><crate><TAB><test>`.
#
# WHY TWO SETTINGS
#   `MTL_SHADER_VALIDATION_THREADGROUP_MEMORY=1` makes MLX's own NAX kernels
#   return wrong values, differently on each run: the routed-expert
#   `gather_qmm` (`affine_gather_qmm_rhs_nax`) in a Qwen3.6 MoE prefill, and
#   the f32 `affine_qmm_t_nax` in a PARO forward. With the threadgroup
#   instrumentation off and everything else on, the same calls are
#   bit-identical to an unvalidated run on every repeat. So a test whose GPU
#   work is MLX's kernels cannot be judged with it on, and rMLX's own `.metal`
#   kernels — the ones this repo can get wrong in threadgroup memory — keep it.
#
# THE RULE
#   A DISPATCHER is a source file that embeds, with `include_str!`, a `.metal`
#   file under a directory `scripts/metal_dirs.sh` names. Its ENTRY NAMES are
#   the names of the fns a non-test dispatcher defines.
#
#   A classified GPU test runs with threadgroup validation `on` if and only if
#     * its declaring file is itself a dispatcher, or
#     * its body, or the body of a fn in the same file it calls (followed
#       transitively, by name), names an entry name.
#   Every other classified GPU test runs with it `off`. Comments and the
#   contents of string literals are not read.
#
#   Every fact is read from the tree and none is a test name: the gated
#   directories from `scripts/metal_dirs.sh`, the embedding from the
#   `include_str!`, the entry names from the dispatchers' fn items, the
#   declaring file from the classifier.
#
# WHAT IT CANNOT SEE
#   * A test that reaches an rMLX kernel only through production code — a
#     `KvCache` update, a storage append, a model forward — names no entry and
#     runs `off`. Following production calls would put every checkpoint test
#     `on`, since a model forward reaches the gated-delta and KV kernels.
#   * A helper in another file (`tests/common`) is not followed.
#   * An entry name is matched by name, not by path: a test that names a
#     dispatcher's helper (`dtype_tag`, `is_supported_d`) without dispatching
#     runs `on`.
#
# A classification in which either setting is empty is refused: the point is
# the split, and a producer that put every test on one side would run the suite
# with one setting and report a clean split.
#
# USAGE
#   bash scripts/gpu_test_threadgroup.sh
#   bash scripts/gpu_test_threadgroup.sh --root <dir>
#
# Exit 0 = the rows, on stdout. Exit 2 = they could not be computed, with the
# reason named.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${REPO_ROOT}"

while [ $# -gt 0 ]; do
    case "$1" in
        --root) ROOT="${2:?--root needs a value}"; shift 2 ;;
        -h|--help)
            echo "Usage: gpu_test_threadgroup.sh [--root <dir>]"
            exit 0 ;;
        *) echo "ERROR: unknown argument '$1' (expected --root <dir>)." >&2; exit 2 ;;
    esac
done

if [ ! -d "${ROOT}" ]; then
    echo "ERROR: --root '${ROOT}' is not a directory." >&2
    exit 2
fi
ROOT="$(cd "${ROOT}" && pwd)"

# The gated directories, resolved against the tree being classified.
METAL_DIRS=()
SCRIPTS_DIR="${REPO_ROOT}/scripts"
REPO_ROOT="${ROOT}"
# shellcheck source=metal_dirs.sh
. "${SCRIPTS_DIR}/metal_dirs.sh"
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
if [ ${#METAL_DIRS[@]} -eq 0 ]; then
    echo "ERROR: scripts/metal_dirs.sh names no gated directory." >&2
    exit 2
fi

listing="$(bash "${REPO_ROOT}/scripts/check_gpu_tests_ignored.sh" --list-files --root "${ROOT}" 2>/dev/null)"
if [ -z "${listing}" ]; then
    echo "ERROR: check_gpu_tests_ignored.sh --list-files produced no GPU tests for ${ROOT}." >&2
    exit 2
fi

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


def embeds_gated_metal(path):
    try:
        lines = open(path, errors='replace').read().split('\n')
    except OSError:
        return False
    for line in lines:
        for lit in INCLUDE.findall(decomment(line)):
            target = os.path.realpath(os.path.join(os.path.dirname(path), lit))
            if os.path.isfile(target) and any(target.startswith(g + os.sep) for g in gated):
                return True
    return False


def fn_bodies(path):
    """fn name -> list of body texts, by rustfmt layout: a fn closes on the
    first line that is its own indent followed by `}`."""
    lines = open(path, errors='replace').read().split('\n')
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
for top, dirs, files in os.walk(os.path.join(root, 'crates')):
    dirs[:] = [d for d in dirs if d != 'target']
    for name in files:
        path = os.path.join(top, name)
        if name.endswith('.rs') and embeds_gated_metal(path):
            dispatchers.add(path)
            if not is_test_file(path):
                entries.update(fn_bodies(path))

if not dispatchers:
    sys.exit('ERROR: no source embeds a .metal file from a gated directory, so no test '
             'can be placed on.')

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

printf '%s\n' "${listing}" | python3 -c "${SPLIT_PY}" "${ROOT}" "${METAL_DIRS[@]}"
