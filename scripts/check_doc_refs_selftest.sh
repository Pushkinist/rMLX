#!/usr/bin/env bash
# scripts/check_doc_refs_selftest.sh — recall test for check_doc_refs.py.
# doc-refs: fixture — the docs/ paths below belong to the synthetic repo.
#
# Builds one synthetic git repo at run time (three docs, a CLAUDE.md map, a
# CHANGELOG, a crate file, scripts), commits it as the base, then copies it
# once per case, applies one edit to the copy's working tree, and runs the
# checker with `--base HEAD`. Every case asserts the exit code and the reason
# in the output. Nothing is written to the real tree.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="${REPO_ROOT}/scripts/check_doc_refs.py"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
BASE="${WORK}/base"

mkdir -p "${BASE}/docs" "${BASE}/crates/c/src" "${BASE}/scripts"
cat >"${BASE}/docs/A.md" <<'EOF'
# A

## Retry budget

A request retries at most twice.

## 1. Intro

The engine reads this doc.

## 2. Contract

### 2.1 Defaults

The default is `x`.
The null was a bit-width result, not a context result.

## The auto default

The auto default is bf16 on every arch.

## Retry Envelope policy

The old policy text.

## Retry Envelope

A retried request carries the same id.

## `MetalKernel` — handles

One handle per kernel.

## Known-bad rows — already in the DB

Two rows.

## Cost of the host path

Host sampling costs one millisecond.

## Evict-to-budget (runtime)

The evictor drops the oldest entry.

## Old story

A dated measurement that only the changelog cites.

## Scratch

Notes that nothing reads.

## Pointers

The retry cap is in § "Retry budget".
Both `B.md` or §"Retry budget" hold it. The table § below lists them.
The cap is § line 3 of the table.

**Budget rule.** Two retries at most.

## Scope — the long form

The long form.

## Scope

The short form.
EOF
cat >"${BASE}/docs/B.md" <<'EOF'
# B

See [the contract](A.md#2-contract), [the map](../CLAUDE.md) and
[the source](../crates/c/src/lib.rs).

Host cost: `docs/A.md` § *Cost of the host path*.
Eviction: `docs/A.md` §
> "Evict-to-budget (runtime)".
Readback: `docs/A.md`
§ "Evict-to-budget (runtime)".
EOF
printf '# Guide\n\nSome text.\n**Glued.** Not a paragraph start.\n\n**Binary** has no period.\n' >"${BASE}/docs/GUIDE.md"
printf '# Out\n\nSee `docs/NOWHERE.md`.\n' >"${BASE}/docs/OUT.md"
cat >"${BASE}/CLAUDE.md" <<'EOF'
# guide

| Doc | Topic |
|---|---|
| [`docs/A.md`](docs/A.md) | A |
| [`docs/B.md`](docs/B.md) | B |
| [`docs/GUIDE.md`](docs/GUIDE.md) | Guide |
| [`docs/OUT.md`](docs/OUT.md) | Out |
EOF
cat >"${BASE}/CHANGELOG.md" <<'EOF'
- Recorded in `docs/A.md` § "Old story".
EOF
cat >"${BASE}/crates/c/src/lib.rs" <<'EOF'
//! Defaults: see `docs/A.md` §2.1 and `docs/A.md` § Defaults.
//! Why: docs/A.md, "The null was a bit-width result".
//! Auto: `docs/A.md` "The auto default".
//! Kernels: `docs/A.md` § `MetalKernel`.
//! Stale: `docs/GONE.md`.
//! Missing: `docs/GUIDE.md` § "Nothing here"; planned: `docs/LATER.md` § "Plan".
//!
//! [`docs/B.md`]: ../../../docs/B.md
EOF
cat >"${BASE}/crates/c/src/t.rs" <<'EOF'
fn f() {
    panic!("update docs/A.md \"Host sampling costs one \
            millisecond\" first");
}
EOF
cat >"${BASE}/crates/c/src/wrap.rs" <<'EOF'
/// Wrapped: see `docs/A.md`
/// § "Evict-to-budget (runtime)".
fn a() {}
const RAW: &str = r#"docs/A.md
§ "Evict-to-budget""#;
const ESC: &str = "docs/A.md \
    § \"Evict-to-budget (runtime)\"";
EOF
cat >"${BASE}/crates/c/src/more.rs" <<'EOF'
//! Rule: `docs/A.md` § "Budget rule".
//! Scope: `docs/A.md` § "Scope".
//! Moved: `docs/GUIDE.md` § Zeta.
//! Glued: `docs/GUIDE.md` § "Glued"; binary: `docs/GUIDE.md` § "Binary".
EOF
cat >"${BASE}/crates/c/m.sql" <<'EOF'
-- Cost: docs/A.md under "Cost of the host path".
EOF
cat >"${BASE}/scripts/s.py" <<'EOF'
# The default: docs/A.md:15 ("The default is x")
# Retry detail: docs/A.md#retry-envelope
# Envelope: docs/A.md § "Retry Envelope"
# Rows: docs/A.md, section "Known-bad rows"
# Guide: GUIDE.md
# Wrapped: docs/A.md
# § "Evict-to-budget (runtime)"
EOF
cat >"${BASE}/scripts/fx_selftest.sh" <<'EOF'
# doc-refs: fixture
# Builds a tree holding docs/NOPE.md.
EOF
cat >"${BASE}/scripts/old_selftest.sh" <<'EOF'
# Reads docs/A.md.
EOF
commit_all() { # commit_all <repo> <message>
    git -C "$1" add -A &&
        git -C "$1" -c user.name=selftest -c user.email=selftest@invalid commit -q -m "$2"
}
git -C "${BASE}" init -q
commit_all "${BASE}" base

FAILED=0
PASSED=0

verdict() { # verdict <name> <want-exit> <needle> <what> <got-exit> <output>
    local name="$1" want="$2" needle="$3" what="$4" got="$5" out="$6"
    if [ "${got}" = "${want}" ] && printf '%s' "${out}" | grep -qF -- "${needle}"; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-38s — %s\n' "${name}" "${what}"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-38s (want exit %s + %q, got exit %s) — %s\n' "${name}" "${want}" "${needle}" "${got}" "${what}"
        printf '%s\n' "${out}" | sed 's/^/         /'
    fi
}

# case_ <name> <want-exit> <needle> <what it proves> <edit-command...>
# The edit runs with the case's copy as the working directory.
case_() {
    local name="$1" want="$2" needle="$3" what="$4"
    shift 4
    local dir="${WORK}/${name}"
    cp -R "${BASE}" "${dir}"
    (cd "${dir}" && "$@") || {
        FAILED=$((FAILED + 1))
        printf '  FAIL %-38s (edit did not apply) — %s\n' "${name}" "${what}"
        return
    }
    local out got
    out="$(python3 "${TOOL}" --root "${dir}" --base HEAD 2>&1)"
    got=$?
    verdict "${name}" "${want}" "${needle}" "${what}" "${got}" "${out}"
}

# run_exit <name> <want-exit> <needle> <what> <checker args...>
run_exit() {
    local name="$1" want="$2" needle="$3" what="$4"
    shift 4
    local out got
    out="$(python3 "${TOOL}" "$@" 2>&1)"
    got=$?
    verdict "${name}" "${want}" "${needle}" "${what}" "${got}" "${out}"
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
        edit scripts/s.py 'docs/A.md#retry' 'docs/B.md#retry' &&
        edit scripts/s.py 'docs/A.md § "Retry' 'docs/B.md § "Retry'
}

move_gone_citation_to_script() {
    edit crates/c/src/lib.rs $'//! Stale: `docs/GONE.md`.\n' '' &&
        append scripts/s.py $'# Stale: docs/GONE.md\n'
}

delete_guide_doc() {
    rm docs/GUIDE.md && edit CLAUDE.md $'| [`docs/GUIDE.md`](docs/GUIDE.md) | Guide |\n' ''
}

add_marked_doc() {
    printf '# doc-refs: fixture\n\nSee `docs/NOPE.md`.\n' >docs/C.md &&
        append CLAUDE.md $'| [`docs/C.md`](docs/C.md) | C |\n'
}

create_later_doc() {
    printf '# Later\n' >docs/LATER.md &&
        append CLAUDE.md $'| [`docs/LATER.md`](docs/LATER.md) | Later |\n'
}

cite_ambiguous_memo() {
    append docs/A.md $'\n## Memo — first\n\nOne.\n\n## Memo — second\n\nTwo.\n' &&
        append crates/c/src/more.rs $'//! Memo: `docs/A.md` § "Memo".\n'
}

noop() { :; }

echo "check_doc_refs selftest:"

case_ clean 0 "OK: " \
    "an unedited tree passes" noop
case_ base_is_printed 0 "base: HEAD (" \
    "the run prints the base it compared with" noop
case_ carried_is_noted 0 "note: 8 reference(s) already broken at the base, carried:" \
    "a reference already broken at the base is carried, not failed" noop
case_ unconsumed_prose_deleted 0 "OK: " \
    "deleting a section nothing reads passes" \
    edit docs/A.md $'## Scratch\n\nNotes that nothing reads.\n' ''
case_ cited_section_moved_with_citations 0 "OK: " \
    "moving a cited section to another doc passes when every citation follows it" \
    move_retry_section_to_b

case_ fixture_file_skipped 0 "not scanned (1 file(s) marked 'doc-refs: fixture'): scripts/fx_selftest.sh" \
    "a selftest marked as a fixture is not scanned, and the run names it" noop
case_ fixture_marker_removed 1 "PATH docs/NOPE.md — resolves to nothing" \
    "the same file without its marker is scanned" \
    edit scripts/fx_selftest.sh $'# doc-refs: fixture\n' ''
case_ fixture_marker_below_header 1 "PATH docs/NOPE.md — resolves to nothing" \
    "a marker below the first ten lines does not count" \
    edit scripts/fx_selftest.sh $'# doc-refs: fixture\n' $'\n\n\n\n\n\n\n\n\n\n# doc-refs: fixture\n'
case_ fixture_marker_in_doc_ignored 1 "docs/C.md:3: PATH docs/NOPE.md — resolves to nothing" \
    "a doc whose H1 is the marker is still scanned" \
    add_marked_doc
case_ fixture_marker_in_crate_ignored 1 "crates/c/src/x.rs:2: PATH docs/NOPE.md — resolves to nothing" \
    "a crate file carrying the marker is still scanned" \
    bash -c 'printf "// doc-refs: fixture\n//! \`docs/NOPE.md\`\n" > crates/c/src/x.rs'
case_ fixture_marker_in_other_script_ignored 1 "scripts/tool.sh:2: PATH docs/NOPE.md — resolves to nothing" \
    "a script not named *_selftest.sh or *_fixtures.sh is still scanned" \
    bash -c 'printf "# doc-refs: fixture\n# docs/NOPE.md\n" > scripts/tool.sh'
case_ fixture_marker_gained 1 "scripts/old_selftest.sh: gained the 'doc-refs: fixture' marker" \
    "a selftest that existed at the base and gains the marker fails" \
    edit scripts/old_selftest.sh $'# Reads' $'# doc-refs: fixture\n# Reads'
case_ new_fixture_file_skipped 0 "not scanned (2 file(s) marked 'doc-refs: fixture')" \
    "a new selftest may carry the marker" \
    bash -c 'printf "# doc-refs: fixture\n# docs/NOPE.md\n" > scripts/new_fixtures.sh'

case_ anchor_heading_deleted 1 "LINK docs/A.md '2-contract' — resolves to nothing" \
    "deleting a heading a Markdown link anchors on fails" \
    edit docs/A.md $'## 2. Contract\n' ''
case_ numbered_section_renumbered 1 "SECTION docs/A.md '2.1' — resolves to nothing" \
    "renumbering a section cited by number fails" \
    edit docs/A.md $'### 2.1 Defaults' $'### 2.2 Defaults'
case_ numbered_section_retitled 1 "SECTION docs/A.md '2.1' — re-pointed: was '2.1 Defaults', now '2.1 Limits'" \
    "a cited section number that now names another title fails" \
    edit docs/A.md $'### 2.1 Defaults' $'### 2.1 Limits'
case_ section_words_deleted 1 "SECTION docs/A.md 'Defaults' — resolves to nothing" \
    "deleting a heading cited as an unquoted name after § fails" \
    edit docs/A.md $'### 2.1 Defaults\n' ''
case_ section_quoted_shared_word 1 "scripts/s.py:3: SECTION docs/A.md 'Retry Envelope' — resolves to nothing" \
    "deleting a heading cited as § \"Phrase\" fails, though an earlier heading starts with its title" \
    edit docs/A.md $'## Retry Envelope\n' ''
case_ section_ident_deleted 1 "SECTION docs/A.md 'MetalKernel' — resolves to nothing" \
    "deleting a heading cited as § and a backticked identifier fails" \
    edit docs/A.md $'## `MetalKernel` — handles\n' ''
case_ section_named_in_prose_deleted 1 "SECTION docs/A.md 'Known-bad rows' — resolves to nothing" \
    "deleting a heading cited as section \"Phrase\" fails" \
    edit docs/A.md $'## Known-bad rows — already in the DB\n' ''
case_ section_under_deleted 1 "crates/c/m.sql:1: SECTION docs/A.md 'Cost of the host path' — resolves to nothing" \
    "deleting a heading cited as under \"Phrase\" fails" \
    edit docs/A.md $'## Cost of the host path\n' ''
case_ section_italic_deleted 1 "docs/B.md:6: SECTION docs/A.md 'Cost of the host path' — resolves to nothing" \
    "deleting a heading cited as § *Phrase* fails" \
    edit docs/A.md $'## Cost of the host path\n' ''
case_ section_after_line_break_deleted 1 "SECTION docs/A.md 'Evict-to-budget (runtime)' — resolves to nothing" \
    "deleting a heading cited by a phrase on the line after § fails" \
    edit docs/A.md $'## Evict-to-budget (runtime)\n' ''
case_ same_doc_section_deleted 1 "docs/A.md:55: SECTION docs/A.md 'Retry budget' — resolves to nothing" \
    "deleting a heading that a § with no doc name cites from the same doc fails" \
    edit docs/A.md $'## Retry budget\n' ''
case_ same_doc_section_after_unparsed_doc_name 1 "docs/A.md:56: SECTION docs/A.md 'Retry budget' — resolves to nothing" \
    "a § right after a doc name whose own parse found nothing still cites the same doc" \
    edit docs/A.md $'## Retry budget\n' ''
case_ wrapped_rust_doc_comment 1 "crates/c/src/wrap.rs:1: SECTION docs/A.md 'Evict-to-budget (runtime)' — resolves to nothing" \
    "a doc name on one /// line and § \"Phrase\" on the next is one citation" \
    edit docs/A.md $'## Evict-to-budget (runtime)\n' ''
case_ wrapped_raw_string 1 "crates/c/src/wrap.rs:4: SECTION docs/A.md 'Evict-to-budget' — resolves to nothing" \
    "a doc name and § on the next line of a raw string are one citation" \
    edit docs/A.md $'## Evict-to-budget (runtime)\n' ''
case_ wrapped_escaped_quotes 1 "crates/c/src/wrap.rs:6: SECTION docs/A.md 'Evict-to-budget (runtime)' — resolves to nothing" \
    "a doc name, a string continuation, and § with escaped quotes are one citation" \
    edit docs/A.md $'## Evict-to-budget (runtime)\n' ''
case_ wrapped_hash_comment 1 "scripts/s.py:6: SECTION docs/A.md 'Evict-to-budget (runtime)' — resolves to nothing" \
    "a doc name on one # line and § on the next is one citation" \
    edit docs/A.md $'## Evict-to-budget (runtime)\n' ''
case_ wrapped_markdown_line 1 "docs/B.md:9: SECTION docs/A.md 'Evict-to-budget (runtime)' — resolves to nothing" \
    "a doc name ending one Markdown line and § starting the next is one citation" \
    edit docs/A.md $'## Evict-to-budget (runtime)\n' ''
case_ duplicate_title_refused 1 "names 'Retry budget', and 2 headings answer to it; make the heading unique" \
    "a citation to a title that two headings carry fails" \
    append docs/A.md $'\n## Retry budget\n\nAgain.\n'
case_ short_name_ambiguity_refused 1 "names 'Memo — first', and 2 headings answer to it; make the heading unique" \
    "a phrase that two titles answer by short name, with no exact title, fails" \
    cite_ambiguous_memo
case_ unquoted_name_is_its_own_carry_key 1 "crates/c/src/more.rs:3: SECTION docs/GUIDE.md 'Zeta' — resolves to nothing" \
    "a carried unquoted citation that turns quoted is a new citation, not the carried one" \
    edit crates/c/src/more.rs '§ Zeta.' '§ "Zeta".'
case_ edited_doc_carries_nothing_in 1 "SECTION docs/GUIDE.md 'Nothing here' — resolves to nothing; this change edits that doc, so fix it here" \
    "a change that edits a doc must fix every broken reference into it" \
    append docs/GUIDE.md $'\nMore text.\n'
case_ edited_doc_carries_nothing_out 1 "docs/OUT.md:3: PATH docs/NOWHERE.md — resolves to nothing; this change edits that doc, so fix it here" \
    "a change that edits a doc must fix every broken reference out of it" \
    append docs/OUT.md $'\nMore text.\n'
case_ created_doc_carries_nothing 1 "SECTION docs/LATER.md 'Plan' — resolves to nothing; this change edits that doc, so fix it here" \
    "a doc created since the base is never in the carried set" \
    create_later_doc
case_ bare_anchor_deleted 1 "ANCHOR docs/A.md 'retry-envelope' — resolves to nothing" \
    "deleting a heading a bare doc#anchor names fails" \
    edit docs/A.md $'## Retry Envelope\n' $'## Retries\n'
case_ heading_fenced 1 "ANCHOR docs/A.md 'retry-envelope' — resolves to nothing" \
    "a heading inside a code fence is not an anchor" \
    edit docs/A.md $'## Retry Envelope\n' $'```\n## Retry Envelope\n```\n'

case_ quoted_phrase_deleted 1 "QUOTE docs/A.md 'The null was a bit-width result' — resolves to nothing" \
    "deleting a phrase cited in quotes fails" \
    edit docs/A.md $'The null was a bit-width result, not a context result.\n' ''
case_ escaped_quote_phrase_deleted 1 "crates/c/src/t.rs:2: QUOTE docs/A.md 'Host sampling costs one millisecond' — resolves to nothing" \
    "a phrase in escaped quotes inside a string literal, wrapped by a line continuation, is checked" \
    edit docs/A.md $'Host sampling costs one millisecond.\n' ''
case_ quoted_phrase_does_not_run_on 1 "QUOTE docs/A.md 'Scratch pad rules' — resolves to nothing" \
    "a quoted phrase that only starts with a heading title does not name that heading" \
    append crates/c/src/lib.rs $'//! Notes: docs/A.md "Scratch pad rules".\n'
case_ quoted_heading_kept_only_in_body 1 "QUOTE docs/A.md 'The auto default' — re-pointed: was 'heading: The auto default', now 'body text'" \
    "a quoted phrase that named a heading fails when only body text still holds it" \
    edit docs/A.md $'## The auto default\n' ''
case_ cited_line_shifted 1 "LINE docs/A.md '15' — re-pointed: was 'The default is \`x\`.'" \
    "deleting text above a line citation re-points it and fails" \
    edit docs/A.md $'The engine reads this doc.\n\n' ''
case_ line_quote_checked 1 "QUOTE docs/A.md 'The default is x' — resolves to nothing" \
    "the phrase quoted after a line citation is checked too" \
    edit docs/A.md $'The default is `x`.' $'The default is `y`.'

case_ linked_doc_deleted 1 "LINK docs/B.md — resolves to nothing" \
    "deleting a doc that is linked fails" \
    rm docs/B.md
case_ bare_doc_name_deleted 1 "scripts/s.py:5: PATH docs/GUIDE.md — resolves to nothing" \
    "deleting a doc named only by its bare file name fails" \
    delete_guide_doc
case_ relative_link_broken 1 "LINK crates/c/src/gone.rs — resolves to nothing" \
    "a relative link written in a doc that points at nothing fails" \
    edit docs/B.md 'lib.rs' 'gone.rs'
case_ new_doc_unmapped 1 "CLAUDE.md (documentation map): MAP docs/C.md — resolves to nothing" \
    "a new top-level doc with no CLAUDE.md map row fails" \
    bash -c 'printf "# C\n" > docs/C.md'
case_ map_row_wrong_target 1 "MAPROW docs/B.md 'docs/A.md' — resolves to nothing" \
    "a map row whose label and link name different docs fails" \
    edit CLAUDE.md '[`docs/A.md`](docs/A.md)' '[`docs/A.md`](docs/B.md)'
case_ new_citation_to_nothing 1 "SECTION docs/A.md '9' — resolves to nothing" \
    "a new citation to a section that never existed fails" \
    append crates/c/src/lib.rs $'//! More: `docs/A.md` §9.\n'
case_ carried_reference_moved 0 "note: 8 reference(s) already broken at the base, carried:" \
    "moving an already-broken reference to another file carries it" \
    move_gone_citation_to_script
case_ carried_reference_copied 1 "PATH docs/GONE.md — resolves to nothing" \
    "a second copy of an already-broken reference is new and fails" \
    append crates/c/src/lib.rs $'//! Again: `docs/GONE.md`.\n'
case_ changelog_never_fails 0 "note: 1 broken reference(s) in CHANGELOG.md (released history; never fails):" \
    "a CHANGELOG reference broken by a cut is printed and does not fail" \
    edit docs/A.md $'## Old story\n\nA dated measurement that only the changelog cites.\n\n' ''

list_out="$(python3 "${TOOL}" --root "${BASE}" --list 2>&1)"
# list_case <name> <present|absent> <extended regex over --list lines> <what it proves>
list_case() {
    local name="$1" mode="$2" pattern="$3" what="$4" got=absent
    printf '%s\n' "${list_out}" | grep -qE -- "${pattern}" && got=present
    if [ "${got}" = "${mode}" ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-38s — %s\n' "${name}" "${what}"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-38s (want %s: %q) — %s\n' "${name}" "${mode}" "${pattern}" "${what}"
    fi
}
T=$'\t'
list_case same_doc_citation_listed present "^SECTION${T}docs/A.md:56${T}docs/A.md${T}Retry budget${T}Retry budget$" \
    "the --list inventory holds the same-doc citation and what it resolves to"
list_case run_in_heading_resolves present "^SECTION${T}crates/c/src/more.rs:1${T}docs/A.md${T}Budget rule${T}Budget rule$" \
    "a **Title.** paragraph opening is a heading"
list_case run_in_needs_paragraph_start present "^SECTION${T}crates/c/src/more.rs:4${T}docs/GUIDE.md${T}Glued${T}BROKEN$" \
    "a bold opening that does not start a paragraph is not a heading"
list_case run_in_needs_period_or_colon present "^SECTION${T}crates/c/src/more.rs:4${T}docs/GUIDE.md${T}Binary${T}BROKEN$" \
    "a bold opening with no period inside and no colon after is not a heading"
list_case exact_title_beats_short_name present "^SECTION${T}crates/c/src/more.rs:2${T}docs/A.md${T}Scope${T}Scope$" \
    "an exact title wins over an earlier heading that has the phrase as its short name"
list_case dash_short_name_resolves present "^SECTION${T}scripts/s.py:4${T}docs/A.md${T}Known-bad rows${T}Known-bad rows — already in the DB$" \
    "the part of a title before ' — ' names the heading"
list_case paren_short_name_resolves present "^SECTION${T}crates/c/src/wrap.rs:4${T}docs/A.md${T}Evict-to-budget${T}Evict-to-budget \(runtime\)$" \
    "a title without its trailing parenthetical names the heading"
list_case prose_below_is_not_a_section absent "^SECTION${T}[^${T}]*${T}[^${T}]*${T}(below|above)( [^${T}]*)?${T}" \
    "§ below and § above are prose, not a section name"
list_case prose_line_is_not_a_section absent "^SECTION${T}[^${T}]*${T}[^${T}]*${T}lines?( [^${T}]*)?${T}" \
    "§ line N is prose, not a section name"

run_exit absolute_mode_fails_on_an_old_break 1 "PATH docs/GONE.md — resolves to nothing" \
    "without a base, a reference broken since the base is a failure" --root "${BASE}"
run_exit unknown_base 2 "could not run" \
    "a base ref that does not exist is exit 2, not a pass" --root "${BASE}" --base no-such-ref
mkdir -p "${WORK}/not_git"
run_exit not_a_git_tree 2 "could not run" \
    "a root that is not a git tree is exit 2, not a pass" --root "${WORK}/not_git"
mkdir -p "${WORK}/no_docs"
git -C "${WORK}/no_docs" init -q
run_exit no_docs 2 "no tracked docs/*.md" \
    "a tree with no docs is exit 2, not a clean scan" --root "${WORK}/no_docs"

# --base auto: origin/main at the base commit, origin/next/x one commit above
# it, HEAD one commit above that.
AUTO="${WORK}/auto"
cp -R "${BASE}" "${AUTO}"
c0=$(git -C "${AUTO}" rev-parse HEAD)
printf 'next\n' >"${AUTO}/next.txt" && commit_all "${AUTO}" next
c1=$(git -C "${AUTO}" rev-parse HEAD)
printf 'work\n' >"${AUTO}/work.txt" && commit_all "${AUTO}" work
run_exit auto_without_refs 2 "no origin/main or origin/next/* ref" \
    "--base auto with no origin ref is exit 2" --root "${AUTO}" --base auto
git -C "${AUTO}" update-ref refs/remotes/origin/main "${c0}"
git -C "${AUTO}" update-ref refs/remotes/origin/next/x "${c1}"
run_exit auto_picks_nearest 0 "base: origin/next/x (merge-base ${c1:0:12})" \
    "--base auto picks the ref that leaves the fewest commits on HEAD" --root "${AUTO}" --base auto
git -C "${AUTO}" update-ref refs/remotes/origin/next/x "$(git -C "${AUTO}" rev-parse HEAD)"
run_exit auto_skips_a_ref_holding_head 0 "base: origin/main (merge-base ${c0:0:12})" \
    "--base auto skips a ref that already contains HEAD" --root "${AUTO}" --base auto
git -C "${AUTO}" update-ref refs/remotes/origin/main "$(git -C "${AUTO}" rev-parse HEAD)"
run_exit auto_falls_back_to_head 0 "base: HEAD (merge-base" \
    "--base auto compares with HEAD when every ref contains it" --root "${AUTO}" --base auto

# A hotfix on main that next/ lacks: main no longer contains next/, and next/
# does not contain main.
HOTFIX="${WORK}/hotfix"
cp -R "${BASE}" "${HOTFIX}"
h0=$(git -C "${HOTFIX}" rev-parse HEAD)
printf 'next\n' >"${HOTFIX}/next.txt" && commit_all "${HOTFIX}" next
next_tip=$(git -C "${HOTFIX}" rev-parse HEAD)
git -C "${HOTFIX}" checkout -q -b hotfix "${h0}"
printf 'fix\n' >"${HOTFIX}/fix.txt" && commit_all "${HOTFIX}" hotfix
main_tip=$(git -C "${HOTFIX}" rev-parse HEAD)
git -C "${HOTFIX}" update-ref refs/remotes/origin/main "${main_tip}"
git -C "${HOTFIX}" update-ref refs/remotes/origin/next/x "${next_tip}"
git -C "${HOTFIX}" checkout -q --detach "${next_tip}"
run_exit auto_on_next_tip_uses_main 0 "base: origin/main (merge-base ${h0:0:12})" \
    "HEAD at a next/ tip that main does not contain compares with the merge-base with main" \
    --root "${HOTFIX}" --base auto
git -C "${HOTFIX}" checkout -q --detach "${main_tip}"
run_exit auto_on_main_uses_next 0 "base: origin/next/x (merge-base ${h0:0:12})" \
    "HEAD at the main tip compares with the merge-base with the next/ ref" \
    --root "${HOTFIX}" --base auto

echo "check_doc_refs selftest: ${PASSED} passed, ${FAILED} failed"
[ "${FAILED}" -eq 0 ]
