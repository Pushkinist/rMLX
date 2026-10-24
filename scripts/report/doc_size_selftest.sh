#!/usr/bin/env bash
# Recall test for debt_report.py --check-doc-size: throwaway git trees, asserting exit and reason.
#
# Recall test for: report/debt_report.py --check-doc-size
#
# doc-refs: fixture
#
# Each case builds a throwaway git work tree, runs the gate against it and
# asserts the literal exit code and, for every failure, the line naming why.
#
# Usage:  bash scripts/report/doc_size_selftest.sh
# Exit:   0 every case held; 1 at least one did not; 2 a scan root could not be
#         built

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="$REPO_ROOT/scripts/report/debt_report.py"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init check-doc-size-selftest

# bytes <path> <n>: a file of exactly n bytes.
bytes() {
    python3 -c 'import sys; open(sys.argv[1], "w").write("x" * (int(sys.argv[2]) - 1) + "\n")' "$1" "$2"
}

# fresh <name>: a git work tree whose docs/ holds one small doc.
fresh() {
    local root
    root="$(st_git_root "$1")"
    mkdir -p "$root/docs"
    printf '# Small\n\nShort doc.\n' >"$root/docs/SMALL.md"
    printf '%s' "$root"
}

# Every gate run is bounded, so a scan that backtracks without end is a failed
# case (exit 124) rather than a hung selftest. `timeout(1)` is not on stock macOS.
CASE_TIMEOUT_S=5
bounded() {
    python3 -c '
import subprocess, sys
try:
    sys.exit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)
except subprocess.TimeoutExpired:
    print(f"timed out after {sys.argv[1]} s")
    sys.exit(124)
' "$CASE_TIMEOUT_S" "$@"
}

# doc_size_gate <root>: the doc-size gate over <root>, bounded.
doc_size_gate() {
    bounded python3 "$TOOL" --root "$1" --check-doc-size
}

root=$(fresh clean)
st_case clean_tree 0 --what "every doc within the cap passes" \
    --has "check-doc-size: ok (1 docs measured" -- doc_size_gate "$root"

root=$(fresh boundary_at_cap)
bytes "$root/docs/AT_CAP.md" 40960
st_case boundary_at_cap_passes 0 --what "a doc of exactly 40,960 B is at the cap, not over it" \
    --has "check-doc-size: ok (2 docs measured" -- doc_size_gate "$root"

root=$(fresh boundary_over_cap)
bytes "$root/docs/OVER_CAP.md" 40961
st_case boundary_one_byte_over_fails 1 --what "a doc of 40,961 B is over the cap and is named" \
    --has "FAIL docs/OVER_CAP.md  40.0 KiB (40961 B) is over the 40 KiB cap" -- doc_size_gate "$root"

root=$(fresh kb_not_kib)
bytes "$root/docs/KB.md" 40500
st_case cap_counts_in_kib 0 --what "40,500 B is over 40 KB and within 40 KiB: it passes" \
    --has "check-doc-size: ok" -- doc_size_gate "$root"

root=$(fresh nested_over)
mkdir -p "$root/docs/deep/er"
bytes "$root/docs/deep/er/NESTED.md" 50000
st_case nested_doc_over_fails 1 --what "a doc below docs/ in a subdirectory is measured and named" \
    --has "FAIL docs/deep/er/NESTED.md" -- doc_size_gate "$root"

root=$(fresh two_over)
bytes "$root/docs/A_OVER.md" 45000
bytes "$root/docs/B_OVER.md" 46000
st_case two_over_first_named 1 --what "the first of two docs over the cap is named" \
    --has "FAIL docs/A_OVER.md" -- doc_size_gate "$root"
st_case two_over_second_named 1 --what "the second is named too: the gate lists every doc, not the first" \
    --has "FAIL docs/B_OVER.md" -- doc_size_gate "$root"

root=$(fresh ignored_over)
mkdir -p "$root/docs/private"
printf 'docs/private/\n' >"$root/.gitignore"
bytes "$root/docs/private/HIDDEN.md" 50000
st_case ignored_doc_skipped 0 --what "a git-ignored doc over the cap is not measured" \
    --has "check-doc-size: ok (1 docs measured" -- doc_size_gate "$root"

root=$(fresh marker_under_cap)
printf '<!-- size-exempt: it is long on purpose -->\n# Marked\n' >"$root/docs/MARKED.md"
st_case marker_in_doc_fails 1 --what "a doc carrying the marker fails even within the cap" \
    --has "FAIL docs/MARKED.md  carries a size-exempt marker" -- doc_size_gate "$root"

root=$(fresh marker_over_cap)
{ printf 'size-exempt: generated\n'; python3 -c 'print("y" * 45000)'; } >"$root/docs/BIG_MARKED.md"
st_case marker_does_not_exempt_size 1 --what "the marker does not lift the cap: the doc is named as over it" \
    --has "FAIL docs/BIG_MARKED.md  44.0 KiB (45024 B) is over the 40 KiB cap" -- doc_size_gate "$root"
st_case marker_over_cap_also_named 1 --what "and the marker is named as a failure of its own" \
    --has "FAIL docs/BIG_MARKED.md  carries a size-exempt marker" -- doc_size_gate "$root"

n=0
for leader in "- " "> " "* " "+ " "## " "1. " "12. " "<!-- - "; do
    n=$((n + 1))
    tag="$n"
    root=$(fresh "marker_leader_$tag")
    printf '# Doc\n\n%ssize-exempt: long on purpose\n' "$leader" >"$root/docs/LEAD.md"
    st_case "marker_after_leader_${tag}" 1 --what "a marker after the Markdown leader '${leader}' still fails" \
        --has "FAIL docs/LEAD.md  carries a size-exempt marker" -- doc_size_gate "$root"
done

root=$(fresh marker_scan_linear)
printf '%s\n' "$(printf '#%.0s' $(seq 64))x" >"$root/docs/HASHES.md"
st_case marker_scan_is_linear 0 --what "a line of 64 '#' and no marker is scanned in bounded time" \
    --has "check-doc-size: ok (2 docs measured" -- doc_size_gate "$root"

root=$(fresh upper_case_suffix)
bytes "$root/docs/SHOUT.MD" 45000
st_case upper_case_suffix_measured 1 --what "a doc spelled .MD is measured like .md" \
    --has "FAIL docs/SHOUT.MD" -- doc_size_gate "$root"

root=$(fresh marker_in_prose)
printf '# Prose\n\nThe gate refuses a `size-exempt:` line in any doc.\n' >"$root/docs/PROSE.md"
st_case marker_mid_line_is_prose 0 --what "a mid-line mention is prose, not a marker" \
    --has "check-doc-size: ok" -- doc_size_gate "$root"

root=$(fresh marker_in_script)
mkdir -p "$root/scripts"
printf '# size-exempt: planted by a selftest\n' >"$root/scripts/x_selftest.sh"
st_case marker_in_script_not_scanned 0 --what "the marker in a selftest script is outside the scanned docs" \
    --has "check-doc-size: ok" -- doc_size_gate "$root"

root=$(fresh undecodable)
printf '\377\n- size-exempt: after a byte that is not UTF-8\n' >"$root/docs/BYTES.md"
st_case undecodable_doc_still_scanned 1 --what "a doc with a byte that is not UTF-8 is read, and its marker is named" \
    --has "FAIL docs/BYTES.md  carries a size-exempt marker" -- doc_size_gate "$root"

# doc_size_gate_fake_git <root>: the gate with a `git` that answers check-ignore with exit 99.
doc_size_gate_fake_git() {
    mkdir -p "$ST_WORK/fakegit"
    printf '#!/bin/sh\nexit 99\n' >"$ST_WORK/fakegit/git"
    chmod +x "$ST_WORK/fakegit/git"
    PATH="$ST_WORK/fakegit:$PATH" doc_size_gate "$1"
}
root=$(fresh git_cannot_say)
st_case git_check_ignore_exit_99_unmeasurable 2 --what "any check-ignore exit but 0 and 1 is a question git did not answer: exit 2" \
    --has "unavailable (git check-ignore exit 99" -- doc_size_gate_fake_git "$root"

root="$(st_git_root no_docs)"
st_case no_docs_dir_unmeasurable 2 --what "a root with no docs/ cannot be measured: exit 2" \
    --has "check-doc-size: unavailable (" -- doc_size_gate "$root"

root="$(st_git_root empty_docs)"
mkdir -p "$root/docs"
st_case empty_docs_unmeasurable 2 --what "a docs/ with no doc is exit 2, never a clean pass" \
    --has "unavailable (no docs/**/*.md under" -- doc_size_gate "$root"

root="$ST_WORK/not_git"
mkdir -p "$root/docs"
bytes "$root/docs/LARGE.md" 50000
st_case outside_git_unmeasurable 2 --what "outside a git work tree the ignored set is unknown: exit 2" \
    --has "unavailable (git check-ignore exit 128" -- doc_size_gate "$root"

st_case missing_root_unmeasurable 2 --what "a root that does not exist is exit 2" \
    --has "is not a directory" -- doc_size_gate "$ST_WORK/does-not-exist"

# With no --root the gate reads the tree the tool sits in: a copy of the tool
# under a fixture tree's scripts/report/.
root=$(fresh default_root)
mkdir -p "$root/scripts/report"
cp "$REPO_ROOT"/scripts/report/*.py "$root/scripts/report/"
st_case default_root_is_the_tools_tree 0 --what "with no --root the gate measures the tree the tool sits in" \
    --has "check-doc-size: ok (1 docs measured" -- bounded python3 "$root/scripts/report/debt_report.py" --check-doc-size

st_done
