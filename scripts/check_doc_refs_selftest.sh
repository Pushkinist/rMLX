#!/usr/bin/env bash
# scripts/check_doc_refs_selftest.sh — recall test for check_doc_refs.py.
# doc-refs: fixture — the docs/ paths below belong to the synthetic repo.
#
# Builds one synthetic git repo at run time (a doc pair, a CLAUDE.md map, a
# crate file, a script), commits it as the base, then copies it once per case,
# applies one edit to the copy's working tree, and runs the checker with
# `--base HEAD`. Every case asserts the exit code and the reason in the output.
# Nothing is written to the real tree.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="${REPO_ROOT}/scripts/check_doc_refs.py"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
BASE="${WORK}/base"

mkdir -p "${BASE}/docs" "${BASE}/crates/c/src" "${BASE}/scripts"
cat >"${BASE}/docs/A.md" <<'EOF'
# A

## 1. Intro

The engine reads this doc.

## 2. Contract

### 2.1 Defaults

The default is `x`.
The null was a bit-width result, not a context result.

## Retry Envelope

A retried request carries the same id.

## Old story

A dated measurement that nothing cites.
EOF
cat >"${BASE}/docs/B.md" <<'EOF'
# B

See [the contract](A.md#2-contract), [the map](../CLAUDE.md) and
[the source](../crates/c/src/lib.rs).
EOF
cat >"${BASE}/CLAUDE.md" <<'EOF'
# guide

| Doc | Topic |
|---|---|
| [`docs/A.md`](docs/A.md) | A |
| [`docs/B.md`](docs/B.md) | B |
EOF
cat >"${BASE}/crates/c/src/lib.rs" <<'EOF'
//! Defaults: see `docs/A.md` §2.1. Retries: `docs/A.md` § Retry Envelope.
//! Why: docs/A.md, "The null was a bit-width result".
//! Stale: `docs/GONE.md`.
//!
//! [`docs/B.md`]: ../../../docs/B.md
EOF
cat >"${BASE}/scripts/s.py" <<'EOF'
# The default: docs/A.md:11
# Retry detail: docs/A.md#retry-envelope
EOF
cat >"${BASE}/scripts/fx.sh" <<'EOF'
# doc-refs: fixture
# Builds a tree holding docs/NOPE.md.
EOF
git -C "${BASE}" init -q
git -C "${BASE}" add -A
git -C "${BASE}" -c user.name=selftest -c user.email=selftest@invalid commit -q -m base

FAILED=0
PASSED=0

# case <name> <want-exit> <needle> <what it proves> <edit-command...>
# The edit runs with the case's copy as the working directory.
case_() {
    local name="$1" want="$2" needle="$3" what="$4"
    shift 4
    local dir="${WORK}/${name}"
    cp -R "${BASE}" "${dir}"
    (cd "${dir}" && "$@") || {
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (edit did not apply) — %s\n' "${name}" "${what}"
        return
    }
    local out got
    out="$(python3 "${TOOL}" --root "${dir}" --base HEAD 2>&1)"
    got=$?
    if [ "${got}" = "${want}" ] && printf '%s' "${out}" | grep -qF -- "${needle}"; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-34s — %s\n' "${name}" "${what}"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (want exit %s + %q, got exit %s) — %s\n' "${name}" "${want}" "${needle}" "${got}" "${what}"
        printf '%s\n' "${out}" | sed 's/^/         /'
    fi
}

edit() { # edit <file> <old> <new> — replace every <old>; an edit that changes nothing is an error
    python3 - "$@" <<'EOF'
import sys
path, old, new = sys.argv[1:4]
s = open(path).read()
if old not in s:
    sys.exit(f"edit: {old!r} not in {path}")
open(path, "w").write(s.replace(old, new))
EOF
}

append() { # append <file> <text>
    printf '%s' "$2" >>"$1"
}

move_retry_section_to_b() {
    local section=$'## Retry Envelope\n\nA retried request carries the same id.\n\n'
    edit docs/A.md "${section}" '' &&
        append docs/B.md $'\n'"${section}" &&
        edit crates/c/src/lib.rs '`docs/A.md` § Retry' '`docs/B.md` § Retry' &&
        edit scripts/s.py 'docs/A.md#retry' 'docs/B.md#retry'
}

move_gone_citation_to_script() {
    edit crates/c/src/lib.rs $'//! Stale: `docs/GONE.md`.\n' '' &&
        append scripts/s.py $'# Stale: docs/GONE.md\n'
}

noop() { :; }

echo "check_doc_refs selftest:"

case_ clean 0 "OK: " \
    "an unedited tree passes" noop
case_ fixture_file_skipped 0 "not scanned (1 file(s) marked 'doc-refs: fixture'): scripts/fx.sh" \
    "a file marked as a fixture is not scanned, and the run names it" noop
case_ fixture_marker_removed 1 "PATH docs/NOPE.md — resolves to nothing" \
    "the same file without its marker is scanned" \
    edit scripts/fx.sh $'# doc-refs: fixture\n' ''
case_ fixture_marker_below_header 1 "PATH docs/NOPE.md — resolves to nothing" \
    "a marker below the first ten lines does not count" \
    edit scripts/fx.sh $'# doc-refs: fixture\n' $'\n\n\n\n\n\n\n\n\n\n# doc-refs: fixture\n'
case_ carried_is_noted 0 "note: 1 reference(s) already broken at HEAD" \
    "a reference already broken at the base is carried, not failed" noop
case_ unconsumed_prose_deleted 0 "OK: " \
    "deleting a section nothing reads passes" \
    edit docs/A.md $'## Old story\n\nA dated measurement that nothing cites.\n' $''
case_ cited_section_moved_with_citations 0 "OK: " \
    "moving a cited section to another doc passes when every citation follows it" \
    move_retry_section_to_b
case_ anchor_heading_deleted 1 "LINK docs/A.md '2-contract' — resolves to nothing" \
    "deleting a heading a Markdown link anchors on fails" \
    edit docs/A.md $'## 2. Contract\n' $''
case_ numbered_section_renumbered 1 "SECTION docs/A.md '2.1' — resolves to nothing" \
    "renumbering a section cited by number fails" \
    edit docs/A.md $'### 2.1 Defaults' $'### 2.2 Defaults'
case_ numbered_section_retitled 1 "SECTION docs/A.md '2.1' — re-pointed: was '2.1 Defaults', now '2.1 Limits'" \
    "a cited section number that now names another title fails" \
    edit docs/A.md $'### 2.1 Defaults' $'### 2.1 Limits'
case_ named_section_deleted 1 "SECTION docs/A.md 'Retry' — resolves to nothing" \
    "deleting a heading cited by name fails" \
    edit docs/A.md $'## Retry Envelope\n' $''
case_ bare_anchor_deleted 1 "ANCHOR docs/A.md 'retry-envelope' — resolves to nothing" \
    "deleting a heading a bare doc#anchor names fails" \
    edit docs/A.md $'## Retry Envelope\n' $'## Retries\n'
case_ heading_fenced 1 "ANCHOR docs/A.md 'retry-envelope' — resolves to nothing" \
    "a heading inside a code fence is not an anchor" \
    edit docs/A.md $'## Retry Envelope\n' $'```\n## Retry Envelope\n```\n'
case_ quoted_phrase_deleted 1 "QUOTE docs/A.md 'The null was a bit-width result' — resolves to nothing" \
    "deleting a phrase cited in quotes fails" \
    edit docs/A.md $'The null was a bit-width result, not a context result.\n' $''
case_ cited_line_shifted 1 "LINE docs/A.md '11' — re-pointed: was 'The default is \`x\`.'" \
    "deleting text above a line citation re-points it and fails" \
    edit docs/A.md $'The engine reads this doc.\n\n' $''
case_ linked_doc_deleted 1 "LINK docs/B.md — resolves to nothing" \
    "deleting a doc that is linked fails" \
    rm docs/B.md
case_ relative_link_broken 1 "LINK crates/c/src/gone.rs — resolves to nothing" \
    "a relative link written in a doc that points at nothing fails" \
    edit docs/B.md $'lib.rs' $'gone.rs'
case_ new_doc_unmapped 1 "MAP docs/C.md — resolves to nothing" \
    "a new top-level doc with no CLAUDE.md map row fails" \
    bash -c 'printf "# C\n" > docs/C.md'
case_ map_row_wrong_target 1 "MAPROW docs/B.md 'docs/A.md' — resolves to nothing" \
    "a map row whose label and link name different docs fails" \
    edit CLAUDE.md $'[`docs/A.md`](docs/A.md)' $'[`docs/A.md`](docs/B.md)'
case_ new_citation_to_nothing 1 "SECTION docs/A.md '9' — resolves to nothing" \
    "a new citation to a section that never existed fails" \
    append crates/c/src/lib.rs $'//! More: `docs/A.md` §9.\n'
case_ carried_reference_moved 0 "note: 1 reference(s) already broken at HEAD" \
    "moving an already-broken reference to another file carries it" \
    move_gone_citation_to_script
case_ carried_reference_copied 1 "PATH docs/GONE.md — resolves to nothing" \
    "a second copy of an already-broken reference is new and fails" \
    append crates/c/src/lib.rs $'//! Again: `docs/GONE.md`.\n'

run_exit() { # run_exit <name> <want-exit> <needle> <what> <checker args...>
    local name="$1" want="$2" needle="$3" what="$4"
    shift 4
    local out got
    out="$(python3 "${TOOL}" "$@" 2>&1)"
    got=$?
    if [ "${got}" = "${want}" ] && printf '%s' "${out}" | grep -qF -- "${needle}"; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-34s — %s\n' "${name}" "${what}"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (want exit %s + %q, got exit %s) — %s\n' "${name}" "${want}" "${needle}" "${got}" "${what}"
        printf '%s\n' "${out}" | sed 's/^/         /'
    fi
}

run_exit absolute_mode_lists_carried 1 "PATH docs/GONE.md — resolves to nothing" \
    "without --base, a reference broken since the base is a failure" --root "${BASE}"
run_exit unknown_base 2 "could not run" \
    "a base ref that does not exist is exit 2, not a pass" --root "${BASE}" --base no-such-ref
mkdir -p "${WORK}/not_git"
run_exit not_a_git_tree 2 "could not run" \
    "a root that is not a git tree is exit 2, not a pass" --root "${WORK}/not_git"
mkdir -p "${WORK}/no_docs"
git -C "${WORK}/no_docs" init -q
run_exit no_docs 2 "no tracked docs/*.md" \
    "a tree with no docs is exit 2, not a clean scan" --root "${WORK}/no_docs"

echo "check_doc_refs selftest: ${PASSED} passed, ${FAILED} failed"
[ "${FAILED}" -eq 0 ]
