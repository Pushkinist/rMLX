#!/usr/bin/env bash
# scripts/check_doc_size_selftest.sh — recall test for the doc-size gate,
# `python3 scripts/lib/debt_report.py --check-doc-size`.
#
# doc-refs: fixture
#
# Each case builds a throwaway git work tree, runs the gate against it and
# asserts the literal exit code and, for every failure, the line naming why.
#
# Exit 0 = every case held. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$REPO_ROOT/scripts/lib/debt_report.py"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAILED=0
PASSED=0

# bytes <path> <n>: a file of exactly n bytes.
bytes() {
    python3 -c 'import sys; open(sys.argv[1], "w").write("x" * (int(sys.argv[2]) - 1) + "\n")' "$1" "$2"
}

# fresh <name>: a git work tree whose docs/ holds one small doc and, as the
# real tree does, the one temporary exception over the cap.
fresh() {
    local root="$WORK/$1"
    mkdir -p "$root/docs"
    git -C "$root" init -q || exit 1
    printf '# Small\n\nShort doc.\n' >"$root/docs/SMALL.md"
    bytes "$root/docs/METRICS_DB.md" 66000
    printf '%s' "$root"
}

# case_run <name> <what> <want-exit> <needle|-> <root>
case_run() {
    local name="$1" what="$2" want="$3" needle="$4" root="$5"
    local out status
    out=$(python3 "$TOOL" --root "$root" --check-doc-size 2>&1)
    status=$?
    if [ "$status" != "$want" ]; then
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (want exit %s, got %s) — %s\n%s\n' "$name" "$want" "$status" "$what" "$out"
    elif [ "$needle" != "-" ] && ! printf '%s' "$out" | grep -qF -- "$needle"; then
        FAILED=$((FAILED + 1))
        printf '  FAIL %-34s (missing %q) — %s\n%s\n' "$name" "$needle" "$what" "$out"
    else
        PASSED=$((PASSED + 1))
        printf '  ok   %-34s — %s\n' "$name" "$what"
    fi
}

root=$(fresh clean)
case_run clean_tree "every doc within the cap passes" \
    0 "check-doc-size: ok (2 docs measured" "$root"

root=$(fresh boundary_at_cap)
bytes "$root/docs/AT_CAP.md" 40960
case_run boundary_at_cap_passes "a doc of exactly 40,960 B is at the cap, not over it" \
    0 "check-doc-size: ok (3 docs measured" "$root"

root=$(fresh boundary_over_cap)
bytes "$root/docs/OVER_CAP.md" 40961
case_run boundary_one_byte_over_fails "a doc of 40,961 B is over the cap and is named" \
    1 "FAIL docs/OVER_CAP.md  40.0 KiB (40961 B) is over the 40 KiB cap" "$root"

root=$(fresh kb_not_kib)
bytes "$root/docs/KB.md" 40500
case_run cap_counts_in_kib "40,500 B is over 40 KB and within 40 KiB: it passes" \
    0 "check-doc-size: ok" "$root"

root=$(fresh nested_over)
mkdir -p "$root/docs/deep/er"
bytes "$root/docs/deep/er/NESTED.md" 50000
case_run nested_doc_over_fails "a doc below docs/ in a subdirectory is measured and named" \
    1 "FAIL docs/deep/er/NESTED.md" "$root"

root=$(fresh two_over)
bytes "$root/docs/A_OVER.md" 45000
bytes "$root/docs/B_OVER.md" 46000
case_run two_over_first_named "the first of two docs over the cap is named" \
    1 "FAIL docs/A_OVER.md" "$root"
case_run two_over_second_named "the second is named too: the gate lists every doc, not the first" \
    1 "FAIL docs/B_OVER.md" "$root"

root=$(fresh ignored_over)
mkdir -p "$root/docs/private"
printf 'docs/private/\n' >"$root/.gitignore"
bytes "$root/docs/private/HIDDEN.md" 50000
case_run ignored_doc_skipped "a git-ignored doc over the cap is not measured" \
    0 "check-doc-size: ok (2 docs measured" "$root"

root=$(fresh marker_under_cap)
printf '<!-- size-exempt: it is long on purpose -->\n# Marked\n' >"$root/docs/MARKED.md"
case_run marker_in_doc_fails "a doc carrying the marker fails even within the cap" \
    1 "FAIL docs/MARKED.md  carries a size-exempt marker" "$root"

root=$(fresh marker_over_cap)
{ printf 'size-exempt: generated\n'; python3 -c 'print("y" * 45000)'; } >"$root/docs/BIG_MARKED.md"
case_run marker_does_not_exempt_size "the marker does not lift the cap: the doc is named as over it" \
    1 "FAIL docs/BIG_MARKED.md  44.0 KiB (45024 B) is over the 40 KiB cap" "$root"
case_run marker_over_cap_also_named "and the marker is named as a failure of its own" \
    1 "FAIL docs/BIG_MARKED.md  carries a size-exempt marker" "$root"

n=0
for leader in "- " "> " "* " "## " "1. " "<!-- - "; do
    n=$((n + 1))
    tag="$n"
    root=$(fresh "marker_leader_$tag")
    printf '# Doc\n\n%ssize-exempt: long on purpose\n' "$leader" >"$root/docs/LEAD.md"
    case_run "marker_after_leader_${tag}" "a marker after the Markdown leader '${leader}' still fails" \
        1 "FAIL docs/LEAD.md  carries a size-exempt marker" "$root"
done

root=$(fresh upper_case_suffix)
bytes "$root/docs/SHOUT.MD" 45000
case_run upper_case_suffix_measured "a doc spelled .MD is measured like .md" \
    1 "FAIL docs/SHOUT.MD" "$root"

root=$(fresh marker_in_prose)
printf '# Prose\n\nThe gate refuses a `size-exempt:` line in any doc.\n' >"$root/docs/PROSE.md"
case_run marker_mid_line_is_prose "a mid-line mention is prose, not a marker" \
    0 "check-doc-size: ok" "$root"

root=$(fresh marker_in_script)
mkdir -p "$root/scripts"
printf '# size-exempt: planted by a selftest\n' >"$root/scripts/x_selftest.sh"
case_run marker_in_script_not_scanned "the marker in a selftest script is outside the scanned docs" \
    0 "check-doc-size: ok" "$root"

root=$(fresh exception_over)
case_run exception_over_cap_passes "the one temporary exception over the cap passes and is printed" \
    0 "check-doc-size: docs/METRICS_DB.md 64.5 KiB, temporary exception:" "$root"

root=$(fresh exception_at_ceiling)
bytes "$root/docs/METRICS_DB.md" 66397
case_run exception_at_ceiling_passes "the exception at its recorded size passes" \
    0 "temporary exception:" "$root"

root=$(fresh exception_grown)
bytes "$root/docs/METRICS_DB.md" 66398
case_run exception_grown_fails "the exception one byte past its recorded size fails" \
    1 "FAIL docs/METRICS_DB.md  66398 B grew past its temporary exception's 66397 B" "$root"

root=$(fresh exception_within)
bytes "$root/docs/METRICS_DB.md" 40960
case_run exception_within_cap_is_stale "an exception whose doc is within the cap fails until removed" \
    1 "FAIL docs/METRICS_DB.md  stale temporary exception (40960 B is within the cap)" "$root"

root=$(fresh exception_gone)
rm -f "$root/docs/METRICS_DB.md"
case_run exception_gone_is_stale "an exception whose doc is gone fails until removed" \
    1 "FAIL docs/METRICS_DB.md  stale temporary exception (the doc is gone)" "$root"

root="$WORK/no_docs"
mkdir -p "$root"
git -C "$root" init -q || exit 1
case_run no_docs_dir_unmeasurable "a root with no docs/ cannot be measured: exit 2" \
    2 "check-doc-size: unavailable (" "$root"

root="$WORK/empty_docs"
mkdir -p "$root/docs"
git -C "$root" init -q || exit 1
case_run empty_docs_unmeasurable "a docs/ with no doc is exit 2, never a clean pass" \
    2 "unavailable (no docs/**/*.md under" "$root"

root="$WORK/not_git"
mkdir -p "$root/docs"
bytes "$root/docs/LARGE.md" 50000
case_run outside_git_unmeasurable "outside a git work tree the ignored set is unknown: exit 2" \
    2 "unavailable (git check-ignore exit 128" "$root"

case_run missing_root_unmeasurable "a root that does not exist is exit 2" \
    2 "is not a directory" "$WORK/does-not-exist"

if [ "$FAILED" -ne 0 ]; then
    echo "check-doc-size selftest: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi
echo "check-doc-size selftest: ok ($PASSED cases)"
