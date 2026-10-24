#!/usr/bin/env bash
# Recall test for debt_report.py --scripts: planted trees, asserting exit code and rule:file:line.
# doc-refs: fixture — the docs/ paths below belong to the planted trees.
#
# Recall test for: report/debt_report.py --scripts
# Usage:  bash scripts/report/scripts_gate_selftest.sh [<debt_report.py under test>]
# Exit:   0 every case held; 1 a case did not; 2 a fixture could not be built
# LOC-exempt: one planted tree per case, and the cases share the fixture helpers at the top
#
# Each case copies one small valid tree (a committed base), plants one edit and
# runs the gate with `--base HEAD`, so a planted file is new. Every tree keeps
# four clean files in scope, so a must-fail case cannot pass through an empty
# scope. The last case runs the gate on the real tree and holds the pinned
# counts it prints to the pin file. It runs on `set -uo pipefail`: a failing
# case must not stop the cases after it.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="${1:-$REPO_ROOT/scripts/report/debt_report.py}"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init scripts-gate-selftest

PINS_REL=scripts/report/scripts_gate_pins.txt
NAM='# Not a mode of: check/ok.sh — a planted entry point'

# gate <root>: the scripts gate over <root>, the base being its own HEAD.
gate() {
    python3 "$TOOL" --scripts --root "$1" --base HEAD
}

# put <root> <rel>: write stdin to <root>/scripts/<rel>, executable when it starts with a shebang.
put() {
    mkdir -p "$(dirname "$1/scripts/$2")"
    cat >"$1/scripts/$2"
    if [ "$(head -c 2 "$1/scripts/$2")" = '#!' ]; then chmod +x "$1/scripts/$2"; fi
}

# entry_sh <line 2> <line 3> <usage> <exit> <caller> <not-a-mode> <body>: a shell
# entry point whose header lines are the arguments. A bare `#` leaves a field out.
entry_sh() {
    printf '%s\n' '#!/usr/bin/env bash' "$1" "$2" "$3" "$4" "$5" "$6" 'set -euo pipefail' "$7"
}

# valid_sh <body>: a shell entry point every rule accepts, with <body> as line 9.
valid_sh() {
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
        '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" "$1"
}

# nocaller_sh <body>: a shell entry point with no Caller: line, so only a run makes it callable.
nocaller_sh() {
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
        '# Exit:   0 clean; 1 failed' '#' "$NAM" "$1"
}

# recall_sh <caller line> <field line>...: a planted recall test; each argument after the first is a line
# of its Recall test for: field. The caller line is a bare # when a recipe runs it.
recall_sh() {
    local caller=$1 first=1 line
    shift
    printf '%s\n' '#!/usr/bin/env bash' '# A planted recall test.' '#'
    for line in "$@"; do
        if [ "$first" = 1 ]; then
            printf '# Recall test for: %s\n' "$line"
            first=0
        else
            printf '#                  %s\n' "$line"
        fi
    done
    printf '%s\n' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 held' "$caller" "$NAM" 'set -euo pipefail' 'exit 0'
}

GATE_TIMEOUT_S=60

# bounded_gate <root>: the gate over <root>, killed after GATE_TIMEOUT_S seconds.
bounded_gate() {
    st_bounded "$GATE_TIMEOUT_S" python3 "$TOOL" --scripts --root "$1" --base HEAD
}

# gate_at <root> <ref>: the gate with <ref> as its base.
gate_at() {
    python3 "$TOOL" --scripts --root "$1" --base "$2"
}

# reindex <root>: regenerate INDEX.md, so a case isolates the rule it plants.
reindex() {
    st_bounded "$GATE_TIMEOUT_S" python3 "$TOOL" --scripts --root "$1" --base HEAD --write-index >/dev/null 2>&1 || true
}

# row_line <root> <rel>: the INDEX.md line that holds the row for <rel>.
row_line() {
    grep -n -F "| \`$2\` |" "$1/scripts/INDEX.md" | head -n 1 | cut -d: -f1
}

commit_all() {
    git -C "$1" add -A &&
        git -C "$1" -c user.name=selftest -c user.email=selftest@invalid -c commit.gpgsign=false \
            commit -q -m "$2" || st_die "cannot commit the fixture tree $1"
}

BASE="$(st_git_root base)"
printf 'check-ok:\n\t@bash scripts/check/ok.sh\n\ntool:\n\t@python3 scripts/report/tool.py\n' >"$BASE/Makefile"
put "$BASE" check/ok.sh <<'EOF'
#!/usr/bin/env bash
# CI gate: the fixture entry point that every rule accepts.
#
# Usage:  bash scripts/check/ok.sh
# Exit:   0 clean; 1 a violation
set -euo pipefail
. "$(dirname "$0")/../lib/helper.sh"
helper
[ -z "${NOPE:-}" ] || exit 1
exit 0
EOF
put "$BASE" lib/helper.sh <<'EOF'
# A helper the fixture's entry point sources.
#
# Module of: check/ok.sh
helper() { :; }
EOF
put "$BASE" report/tool.py <<'EOF'
#!/usr/bin/env python3
"""The fixture's Python entry point; it imports its module.

Usage:  python3 scripts/report/tool.py
Exit:   0 always
"""

import tool_mod

raise SystemExit(0)
EOF
put "$BASE" report/tool_mod.py <<'EOF'
"""A module the fixture's Python entry point imports.

Module of: report/tool.py
"""
EOF
commit_all "$BASE" files
reindex "$BASE"
commit_all "$BASE" base

# fresh <name>: a copy of the base tree, to plant one edit in.
fresh() {
    cp -R "$BASE" "$ST_WORK/$1" || st_die "cannot copy the base tree to $1"
    printf '%s' "$ST_WORK/$1"
}

root=$(fresh clean)
st_case clean_tree 0 --what "the base tree passes with no pin" \
    --has "check-scripts: ok (4 files, 0 pinned)" --has "pinned: none" -- gate "$root"
st_case base_is_printed 0 --what "the run names the base it used" \
    --has "check-scripts: base HEAD (" -- gate "$root"

# ---- index-stale ---------------------------------------------------------

root=$(fresh index_new_file)
valid_sh 'exit 0' | put "$root" check/x.sh
line=$(row_line "$root" check/ok.sh)
st_case index_missing_row 1 --what "a new file with no row: named at the line after its neighbour's row" \
    --has "index-stale: INDEX.md:$((line + 1)): missing row for check/x.sh" --lacks "not-a-mode" -- gate "$root"

root=$(fresh index_edited_row)
line=$(row_line "$root" check/ok.sh)
python3 - "$root/scripts/INDEX.md" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("fixture entry point that every rule accepts", "an edited purpose")
open(path, "w").write(text)
EOF
st_case index_edited_row 1 --what "a row edited by hand is named at its own line" \
    --has "index-stale: INDEX.md:${line}: row for check/ok.sh differs" -- gate "$root"

root=$(fresh index_deleted_file)
line=$(row_line "$root" lib/helper.sh)
rm "$root/scripts/lib/helper.sh"
st_case index_deleted_file 1 --what "a deleted file's row is named at its own line" \
    --has "index-stale: INDEX.md:${line}: row for lib/helper.sh names no file" -- gate "$root"

root=$(fresh index_preamble)
python3 - "$root/scripts/INDEX.md" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("Never edit it by hand.", "Edit it freely.")
open(path, "w").write(text)
EOF
st_case index_preamble_edited 1 --what "a change outside the rows is still stale" \
    --has "index-stale: INDEX.md:" --has "differs from the generated text" -- gate "$root"

root=$(fresh index_longer)
printf '\n' >>"$root/scripts/INDEX.md"
line=$(($(grep -c '' "$root/scripts/INDEX.md") + 1))
st_case index_only_longer 1 --what "an INDEX.md that differs only in length is named at the first line past the text, with no traceback" \
    --has "index-stale: INDEX.md:${line}: differs from the generated text" --lacks "Traceback" -- gate "$root"

root=$(fresh index_absent)
rm "$root/scripts/INDEX.md"
st_case index_absent 1 --what "no INDEX.md is stale, not clean" \
    --has "index-stale: INDEX.md:1: missing" -- gate "$root"

root=$(fresh index_written)
valid_sh 'exit 0' | put "$root" check/x.sh
st_case write_index_then_pass 0 --what "--write-index regenerates the rows and the check then passes" \
    --has "scripts-index: wrote scripts/INDEX.md (5 rows)" -- python3 "$TOOL" --scripts --root "$root" --base HEAD --write-index
st_case index_written_passes 0 --what "the regenerated index is what the check expects" \
    --has "check-scripts: ok (5 files, 0 pinned)" -- gate "$root"

root=$(fresh index_write_refused)
valid_sh 'exit 0' | put "$root" check/x.sh
before=$(cksum <"$root/scripts/INDEX.md")
st_case write_index_unknown_base 2 --what "--write-index with a base that is no commit refuses" \
    --has "is not a commit" -- python3 "$TOOL" --scripts --root "$root" --base no-such-ref --write-index
st_case write_index_unknown_base_writes_nothing 0 --what "and the INDEX.md it would have rewritten is as it was" \
    --has "$before" -- sh -c 'cksum <"$1/scripts/INDEX.md"' sh "$root"

root=$(fresh index_written_still_checks)
put "$root" stray.sh <<'EOF'
#!/usr/bin/env bash
# A top-level file the class rule refuses.
set -euo pipefail
EOF
st_case write_index_runs_the_rules 1 --what "--write-index writes the index and still reports every other rule" \
    --has "class: stray.sh:1:" --lacks "index-stale" -- python3 "$TOOL" --scripts --root "$root" --base HEAD --write-index

# ---- class ---------------------------------------------------------------

root=$(fresh class_top)
valid_sh 'exit 0' | put "$root" x.sh
reindex "$root"
st_case class_top_level_file 1 --what "a file at the top of scripts/" \
    --has "class: x.sh:1:" -- gate "$root"

root=$(fresh class_newdir)
valid_sh 'exit 0' | put "$root" newdir/x.sh
reindex "$root"
st_case class_unknown_directory 1 --what "a file in a directory outside the class list" \
    --has "class: newdir/x.sh:1:" -- gate "$root"

root=$(fresh class_readme)
printf '# scripts/ — rules\n' >"$root/scripts/README.md"
reindex "$root"
st_case class_readme_allowed 0 --what "README.md at the top is not a file of the tree" \
    --has "check-scripts: ok (4 files" -- gate "$root"

# ---- purpose -------------------------------------------------------------

root=$(fresh purpose_missing)
entry_sh '#' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case purpose_missing 1 --what "line 2 is a bare comment: no purpose" \
    --has "purpose: check/x.sh:2:" -- gate "$root"

root=$(fresh purpose_code_first)
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' | put "$root" check/x.sh
reindex "$root"
st_case purpose_code_on_line_2 1 --what "code right after the shebang: no purpose" \
    --has "purpose: check/x.sh:2:" -- gate "$root"

root=$(fresh purpose_wrapped)
entry_sh '# A purpose line that carries on' '# onto the next line.' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' | put "$root" check/x.sh
reindex "$root"
st_case purpose_wrapped 1 --what "the purpose wraps onto line 3, which is named" \
    --has "purpose: check/x.sh:3:" -- gate "$root"

root=$(fresh purpose_label_next)
entry_sh '# CI gate: a purpose followed at once by a field.' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' '#' "$NAM" 'exit 0' | put "$root" check/x.sh
reindex "$root"
st_case purpose_then_field_not_wrapped 0 --what "a field on line 3 is not a wrapped purpose" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh purpose_marker_next)
entry_sh '# CI gate: a purpose followed at once by a marker.' '# doc-refs: fixture — planted' \
    '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case purpose_then_marker_not_wrapped 0 --what "a doc-refs marker on line 3 is not a wrapped purpose" \
    --has "check-scripts: ok (5 files" -- gate "$root"

long100="# $(printf '%*s' 100 '' | tr ' ' a)"
root=$(fresh purpose_100)
entry_sh "$long100" '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case purpose_100_characters_passes 0 --what "a purpose of exactly 100 characters is within the limit" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh purpose_101)
entry_sh "${long100}a" '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case purpose_101_characters 1 --what "a purpose of 101 characters is over the limit" \
    --has "purpose: check/x.sh:2: purpose is 101 characters, over 100" -- gate "$root"

root=$(fresh purpose_py_missing)
put "$root" report/y.py <<'EOF'
#!/usr/bin/env python3
"""
Usage:  python3 scripts/report/y.py
Exit:   0 always
Not a mode of: report/tool.py — a planted entry point
Caller: by hand
"""
raise SystemExit(0)
EOF
reindex "$root"
st_case purpose_python_docstring_missing 1 --what "a docstring that opens on a bare line has no purpose" \
    --has "purpose: report/y.py:2:" -- gate "$root"

root=$(fresh purpose_py_ok)
put "$root" report/y.py <<'EOF'
#!/usr/bin/env python3
"""A planted Python entry point with its purpose on the docstring's first line.

Usage:  python3 scripts/report/y.py
Exit:   0 always
Not a mode of: report/tool.py — a planted entry point
Caller: by hand
"""
raise SystemExit(0)
EOF
reindex "$root"
st_case purpose_python_docstring_ok 0 --what "the same file with a purpose passes" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh purpose_data_file)
printf '%s\n' 'no comment at all' >"$root/scripts/report/data.txt"
reindex "$root"
st_case purpose_data_file_without_comment 1 --what "a data file is held to the rule too" \
    --has "purpose: report/data.txt:1:" -- gate "$root"

# ---- usage ---------------------------------------------------------------

root=$(fresh usage_missing)
entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case usage_missing 1 --what "an entry point without a Usage: line" \
    --has "usage: check/x.sh:1:" -- gate "$root"

root=$(fresh usage_lib_exempt)
printf '%s\n' '# A second helper with no usage line.' '#' '# Module of: check/ok.sh' |
    put "$root" lib/second.sh
printf '. "$(dirname "$0")/../lib/second.sh"\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case usage_not_asked_of_lib 0 --what "a lib/ file is not an entry point" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh usage_py_no_shebang)
put "$root" report/z.py <<'EOF'
"""A Python file with no shebang, which no one runs directly.

Nothing here is an entry point, so no Usage: line is asked of it.
"""
EOF
reindex "$root"
st_case usage_not_asked_without_shebang 1 --what "no shebang and nobody runs it: not an entry point, so an orphan and no Usage: asked" \
    --has "orphan: report/z.py:1:" --lacks "usage: report/z.py" -- gate "$root"

root=$(fresh usage_py_imported)
put "$root" report/z.py <<'EOF'
"""A Python file with no shebang that the tool imports without a Module of: line.

Imported, never run: not an entry point and not an orphan.
"""
EOF
printf 'import z\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case usage_not_asked_of_an_imported_file 0 --what "an import is a caller: neither an entry point nor an orphan" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh usage_run_without_shebang)
printf '%s\n' '# A planted script with no shebang that a recipe runs.' 'exit 0' | put "$root" check/noshebang.sh
printf 'check-noshebang:\n\t@bash scripts/check/noshebang.sh\n' >>"$root/Makefile"
reindex "$root"
st_case usage_asked_of_a_file_a_recipe_runs 1 --what "no shebang, but a recipe runs it: an entry point" \
    --has "usage: check/noshebang.sh:1:" -- gate "$root"

root=$(fresh usage_run_in_lib)
printf '%s\n' '#!/usr/bin/env bash' '# A planted file in lib/ that a recipe runs.' 'exit 0' | put "$root" lib/runme.sh
printf 'check-runme:\n\t@bash scripts/lib/runme.sh\n' >>"$root/Makefile"
reindex "$root"
st_case usage_asked_of_a_lib_file_a_recipe_runs 1 --what "a lib/ file a recipe runs is an entry point" \
    --has "usage: lib/runme.sh:1:" -- gate "$root"

root=$(fresh usage_empty)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:' '# Exit:   0 clean' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case usage_empty 1 --what "a Usage: line that says nothing" \
    --has "usage: check/x.sh:4: \`Usage:\` says nothing" -- gate "$root"

root=$(fresh usage_module_with_shebang)
put "$root" report/tool_three.py <<'EOF'
#!/usr/bin/env python3
"""A module that also carries a shebang.

Module of: report/tool.py
"""
EOF
printf 'import tool_three\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case usage_not_asked_of_a_module 1 --what "a Module of: line takes a file out of the entry points; its shebang is the finding" \
    --has "shebang: report/tool_three.py:1:" --lacks "usage:" -- gate "$root"

# ---- exit-undeclared -----------------------------------------------------

root=$(fresh exit_seven)
valid_sh 'exit 7' | put "$root" check/x.sh
reindex "$root"
st_case exit_undeclared_shell 1 --what "exit 7 on line 9 is not in the Exit: line" \
    --has "exit-undeclared: check/x.sh:9: 7 is not declared" -- gate "$root"

root=$(fresh exit_twice)
valid_sh "$(printf 'exit 7\nexit 7\n')" | put "$root" check/x.sh
reindex "$root"
st_case exit_undeclared_reported_once 1 --what "a code used twice is named once, at its first line" \
    --has "exit-undeclared: check/x.sh:9: 7" --lacks "check/x.sh:10:" -- gate "$root"

root=$(fresh exit_passthrough)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   passthrough of rmlx' '# Caller: by hand' "$NAM" 'exit "$rc"' | put "$root" check/x.sh
reindex "$root"
st_case exit_passthrough_not_literal 0 --what "exit \"\$rc\" under a passthrough declaration" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_declared)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 7 a violation' '# Caller: by hand' "$NAM" 'exit 7' | put "$root" check/x.sh
reindex "$root"
st_case exit_declared_passes 0 --what "exit 7 named in the Exit: line" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_range)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 3-10 specific to this script' '# Caller: by hand' "$NAM" 'exit 7' | put "$root" check/x.sh
reindex "$root"
st_case exit_range_declared 0 --what "a code inside a declared range" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_hidden)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean' '# Caller: by hand' "$NAM" "$(printf '%s\n' \
        '# exit 7 in a comment' \
        '# done; exit 7 in a comment' \
        'echo "exit 7 in a string"' \
        'echo "done; exit 7"' \
        "echo 'then exit 7 in single quotes'" \
        'cat <<TXT' \
        'exit 7' \
        'TXT' \
        'python3 -c '"'"'import sys; sys.exit(9)'"'"'' \
        'exit 0')" | put "$root" check/x.sh
reindex "$root"
st_case exit_in_comment_string_heredoc 0 --what "exit in a comment, a string, a heredoc body or python -c is not the script's exit" \
    --has "check-scripts: ok (5 files" -- gate "$root"

n=0
for body in 'true || exit 6' 'true && exit 6' '{ exit 6; }' 'case x in x) exit 6;; esac' '( exit 6 )' 'true; exit 6'; do
    n=$((n + 1))
    root=$(fresh "exit_position_$n")
    valid_sh "$body" | put "$root" check/x.sh
    reindex "$root"
    st_case "exit_position_$n" 1 --what "exit 6 after: $body" \
        --has "exit-undeclared: check/x.sh:9: 6" -- gate "$root"
done

root=$(fresh exit_en_dash)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 3–10 specific to this script' '# Caller: by hand' "$NAM" 'exit 7' | put "$root" check/x.sh
reindex "$root"
st_case exit_en_dash_range 0 --what "a range written with an en dash" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_dash_heredoc)
valid_sh "$(printf 'cat <<-TXT\n\texit 7\n\tTXT\nexit 6')" | put "$root" check/x.sh
reindex "$root"
st_case exit_after_tab_indented_heredoc 1 --what "<<- ends at a tab-indented tag, so the exit after it is read" \
    --has "exit-undeclared: check/x.sh:12: 6" --lacks ": 7 is not declared" -- gate "$root"

root=$(fresh exit_prose_digit)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed; reads the last 30 lines of the log' '# Caller: by hand' "$NAM" 'exit 30' |
    put "$root" check/x.sh
reindex "$root"
st_case exit_digit_in_prose_is_not_declared 1 --what "30 in the prose of Exit: declares nothing" \
    --has "exit-undeclared: check/x.sh:9: 30 is not declared" -- gate "$root"

root=$(fresh exit_equals_form)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    "$(printf '# Exit:   0 = clean\n#         4 = refused')" '# Caller: by hand' "$NAM" 'exit 4' |
    put "$root" check/x.sh
reindex "$root"
st_case exit_n_equals_meaning_declares 0 --what "N = meaning, on a continuation line, declares N" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_substitution_quotes)
dq='"'
sq="'"
valid_sh "v=${dq}\$(echo ${sq}${dq}${sq})${dq}"$'\n'"exit 6" | put "$root" check/x.sh
reindex "$root"
st_case exit_after_a_substitution_with_quotes 1 --what "a quote inside \$( ) in a string does not hide the exit after it" \
    --has "exit-undeclared: check/x.sh:10: 6" -- gate "$root"

root=$(fresh exit_after_then)
valid_sh 'if [ -n "${X:-}" ]; then exit 5; fi' | put "$root" check/x.sh
reindex "$root"
st_case exit_after_then 1 --what "exit after then is read" \
    --has "exit-undeclared: check/x.sh:9: 5" -- gate "$root"

root=$(fresh exit_python)
put "$root" report/y.py <<'EOF'
#!/usr/bin/env python3
"""A planted Python entry point that exits with an undeclared code.

Usage:  python3 scripts/report/y.py
Exit:   0 always
Not a mode of: report/tool.py — a planted entry point
Caller: by hand
"""
import sys

sys.exit(4)
EOF
reindex "$root"
st_case exit_undeclared_python 1 --what "sys.exit(4) on line 11" \
    --has "exit-undeclared: report/y.py:11: 4 is not declared" -- gate "$root"

root=$(fresh exit_python_string)
put "$root" report/y.py <<'EOF'
#!/usr/bin/env python3
"""A planted Python entry point whose exit text is in a string.

Usage:  python3 scripts/report/y.py
Exit:   0 always
Not a mode of: report/tool.py — a planted entry point
Caller: by hand
"""
print("sys.exit(4)")
raise SystemExit(0)
EOF
reindex "$root"
st_case exit_python_in_string 0 --what "sys.exit(4) inside a string is not a call" \
    --has "check-scripts: ok (5 files" -- gate "$root"

n=0
for stmt in 'sys.exit(5)' 'exit(5)' 'quit(5)' 'os._exit(5)' 'raise SystemExit(5)'; do
    n=$((n + 1))
    root=$(fresh "exit_python_spelling_$n")
    printf '%s\n' '#!/usr/bin/env python3' '"""A planted Python entry point that exits with an undeclared code.' '' \
        'Usage:  python3 scripts/report/y.py' 'Exit:   0 always' \
        'Not a mode of: report/tool.py — a planted entry point' 'Caller: by hand' '"""' \
        'import os, sys' "$stmt" | put "$root" report/y.py
    reindex "$root"
    st_case "exit_python_spelling_$n" 1 --what "$stmt on line 10" \
        --has "exit-undeclared: report/y.py:10: 5 is not declared" -- gate "$root"
done

# ---- caller --------------------------------------------------------------

root=$(fresh caller_none)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
reindex "$root"
st_case caller_none 1 --what "nothing runs it and it does not say it is by hand" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_by_hand_ok)
valid_sh 'exit 0' | put "$root" check/x.sh
reindex "$root"
st_case caller_by_hand_declared 0 --what "Caller: by hand on a file nothing runs" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh caller_stale_by_hand)
python3 - "$root/scripts/check/ok.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Exit:   0 clean; 1 a violation\n", "# Exit:   0 clean; 1 a violation\n# Caller: by hand\n")
open(path, "w").write(text)
EOF
reindex "$root"
st_case caller_by_hand_but_run 1 --what "Caller: by hand on a file the Makefile runs, named at its own line" \
    --has "caller: check/ok.sh:6: declares \`Caller: by hand\` but something runs it" -- gate "$root"

root=$(fresh caller_longer_name)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/ab.sh
printf 'check-lab:\n\t@bash scripts/check/lab.sh\n' >>"$root/Makefile"
reindex "$root"
st_case caller_name_inside_a_longer_name 1 --what "lab.sh in a recipe is not a caller of ab.sh" \
    --has "caller: check/ab.sh:1:" -- gate "$root"

root=$(fresh caller_suffix)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
printf 'check-bak:\n\t@bash scripts/check/x.sh.bak\n' >>"$root/Makefile"
reindex "$root"
st_case caller_name_with_a_suffix 1 --what "x.sh.bak in a recipe is not a caller of x.sh" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_prose)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
printf 'say:\n\t@echo "run scripts/check/x.sh by hand"\n\t@printf "%%s\\n" scripts/check/x.sh\n\t# bash scripts/check/x.sh\n' >>"$root/Makefile"
printf '# bash scripts/check/x.sh\necho scripts/check/x.sh\nusage() { echo "see check/x.sh"; }\n' >>"$root/scripts/lib/helper.sh"
reindex "$root"
st_case caller_named_only_in_text 1 --what "a comment, an echo and a printf name it and run nothing" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_script_runs_it)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
printf 'bash "$(dirname "$0")/x.sh"\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case caller_another_script_runs_it 0 --what "another script running it is a caller" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh caller_in_subshell)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
printf 'out=$(bash scripts/check/x.sh 2>&1)\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case caller_inside_command_substitution 0 --what "a run inside \$( ) is a caller" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh caller_self)
valid_sh 'bash scripts/check/x.sh' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
reindex "$root"
st_case caller_not_itself 1 --what "a script that runs itself has no caller" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_data_files)
printf '%s\n' '# A planted data file that a script only names in text.' >"$root/scripts/report/said.txt"
printf '%s\n' '# A planted data file that a script reads.' >"$root/scripts/report/read.txt"
printf 'echo scripts/report/said.txt\ncat scripts/report/read.txt\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case caller_data_file_named_in_echo 0 --what "a data file named only by echo has no reader" \
    --has '| `report/said.txt` | none |' -- cat "$root/scripts/INDEX.md"
st_case caller_data_file_read 0 --what "a data file passed to cat is read by the script" \
    --has '| `report/read.txt` | check/ok.sh |' -- cat "$root/scripts/INDEX.md"

n=0
for body in 'out=`bash scripts/check/x.sh`' 'timeout 5 bash scripts/check/x.sh' 'env A=1 scripts/check/x.sh' \
    'uv run scripts/check/x.sh' 'run_case name -- scripts/check/x.sh arg' 'FOO=1 bash -x scripts/check/x.sh'; do
    n=$((n + 1))
    root=$(fresh "caller_spelling_$n")
    valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
        put "$root" check/x.sh
    printf '%s\n' "$body" >>"$root/scripts/check/ok.sh"
    reindex "$root"
    st_case "caller_spelling_$n" 0 --what "run as: $body" \
        --has "check-scripts: ok (5 files" -- gate "$root"
done

root=$(fresh caller_multi_target)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
printf 'multi-a multi-b:\n\t@bash scripts/check/x.sh\n' >>"$root/Makefile"
reindex "$root"
st_case caller_every_target_of_a_rule 0 --what "a rule with two target names credits both" \
    --has '| `check/x.sh` | make multi-a, make multi-b |' -- cat "$root/scripts/INDEX.md"

root=$(fresh caller_assign_unused)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'SUBJECT=scripts/check/x.sh\ngrep -c TODO "$SUBJECT" >/dev/null || true\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case caller_assignment_only_as_an_argument 1 --what "VAR=path with \$VAR used as an argument runs nothing" \
    --has "caller: check/x.sh:1:" -- gate "$root"

n=0
for use in 'bash "$SUBJECT"' 'st_run "$SUBJECT"' 'st_case name 0 -- "$SUBJECT"' 'env -u A "$SUBJECT"' \
    'T2="$SUBJECT"; bash "${T2}"' 'timeout 5 "$SUBJECT"' 'bounded python3 "$SUBJECT"' \
    'st_bounded 5 bash "$SUBJECT"'; do
    n=$((n + 1))
    root=$(fresh "caller_assign_used_$n")
    nocaller_sh 'exit 0' | put "$root" check/x.sh
    printf 'SUBJECT=scripts/check/x.sh\n%s\n' "$use" >>"$root/scripts/check/ok.sh"
    reindex "$root"
    st_case "caller_assign_used_$n" 0 --what "VAR=path, later used as: $use" \
        --has "check-scripts: ok (5 files" -- gate "$root"
done

root=$(fresh caller_assign_used_before)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'bash "$SUBJECT"\nSUBJECT=scripts/check/x.sh\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case caller_variable_used_before_it_is_set 1 --what "\$VAR expanded before VAR=path credits nothing" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_python_constant)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf '\nSUBJECT = "scripts/check/x.sh"\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case caller_python_bare_constant 1 --what "a Python string constant outside a command vector is no caller" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_shell_continuation)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'echo \\\nscripts/check/x.sh\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case caller_shell_continuation_joined 1 --what "a name on the continuation line of an echo in a script is text" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_make_continuation)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'check-echo:\n\t@echo \\\n\t\tscripts/check/x.sh\n' >>"$root/Makefile"
reindex "$root"
st_case caller_recipe_continuation_joined 1 --what "a name on the continuation line of an echo is text" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_make_continuation_run)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'check-cont:\n\t@bash \\\n\t\tscripts/check/x.sh\n' >>"$root/Makefile"
reindex "$root"
st_case caller_recipe_continuation_run 0 --what "bash on one line and the script on the next is a run" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh caller_make_continuation_flush)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'check-flush:\n\t@bash \\\nscripts/check/x.sh\n' >>"$root/Makefile"
reindex "$root"
st_case caller_recipe_continuation_at_column_0 0 --what "a continuation line that starts at column 0 still belongs to the recipe" \
    --has "check-scripts: ok (5 files" -- gate "$root"

n=0
for wrapper in 'exec' 'env A=1' 'command' 'nohup' 'nice -n 5' 'timeout 5' 'sudo' 'builtin'; do
    n=$((n + 1))
    root=$(fresh "caller_wrapper_$n")
    nocaller_sh 'exit 0' | put "$root" check/x.sh
    printf '%s scripts/check/x.sh\n' "$wrapper" >>"$root/scripts/check/ok.sh"
    reindex "$root"
    st_case "caller_wrapper_$n" 0 --what "run behind: $wrapper" \
        --has "check-scripts: ok (5 files" -- gate "$root"
done

n=0
for form in 'source scripts/lib/second.sh' '. scripts/lib/second.sh'; do
    n=$((n + 1))
    root=$(fresh "caller_source_$n")
    printf '%s\n' '# A second helper that nothing runs.' '#' '# Module of: check/ok.sh' |
        put "$root" lib/second.sh
    printf '%s\n' "$form" >>"$root/scripts/check/ok.sh"
    reindex "$root"
    st_case "caller_source_$n" 0 --what "$form: a caller, and a sourced file is no entry point" \
        --has '| `lib/second.sh` | check/ok.sh |' -- cat "$root/scripts/INDEX.md"
    st_case "caller_source_clean_$n" 0 --what "and it is not asked for a Usage: line" \
        --has "check-scripts: ok (5 files" -- gate "$root"
done

root=$(fresh caller_nested_substitution)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'bash "$(dirname "$(dirname "$0")")/check/x.sh"\n' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case caller_nested_substitution 0 --what "a path built from a nested \$( ) in double quotes" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh caller_cell_by_hand)
valid_sh 'exit 0' | put "$root" check/x.sh
reindex "$root"
st_case caller_cell_by_hand 0 --what "an entry nothing runs, declared by hand, reads by hand in the index" \
    --has '| `check/x.sh` | by hand |' -- cat "$root/scripts/INDEX.md"

root=$(fresh caller_python_string)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/x.sh
printf 'import subprocess\n\nsubprocess.run(["bash", "scripts/check/x.sh"])\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case caller_python_path_string 0 --what "a Python file naming the path in a call is a caller" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh caller_python_from_import)
put "$root" report/tool_four.py <<'EOF'
"""A planted module imported with from.

Module of: report/tool.py
"""
EOF
printf 'from tool_four import nothing\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case caller_python_from_import 0 --what "from <module> import names the importing tool as the module's caller" \
    --has '| `report/tool_four.py` | report/tool.py |' -- cat "$root/scripts/INDEX.md"

root=$(fresh caller_same_basename)
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" check/dup.sh
valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    put "$root" report/dup.sh
printf 'check-dup:\n\t@bash scripts/check/dup.sh\n' >>"$root/Makefile"
reindex "$root"
st_case caller_two_files_one_basename 1 --what "a recipe naming check/dup.sh is no caller of report/dup.sh" \
    --has "caller: report/dup.sh:1:" --lacks "caller: check/dup.sh" -- gate "$root"

# ---- callers that are not scripts ---------------------------------------

n=0
for snippet in \
    'SUBJECT = "scripts/check/x.sh"\nsubprocess.run(["bash", SUBJECT])\n' \
    'CMD = ["bash", "scripts/check/x.sh"]\nsubprocess.check_call(CMD)\n' \
    'SCRIPT = Path(__file__).resolve().parents[1] / "check" / "x.sh"\nsubprocess.check_output(["bash", str(SCRIPT)])\n' \
    'BASE = Path(__file__).resolve().parents[1]\nSCRIPT = BASE / "check" / "x.sh"\nsubprocess.Popen(["bash", str(SCRIPT)])\n' \
    'NAME = "x.sh"\nSCRIPT = Path(__file__).resolve().parents[1] / NAME\nsubprocess.run(["bash", str(SCRIPT)])\n' \
    'SUBJECT = "scripts/check/x.sh"\nsubprocess.call(["bash", SUBJECT])\n'; do
    n=$((n + 1))
    root=$(fresh "caller_py_bound_$n")
    nocaller_sh 'exit 0' | put "$root" check/x.sh
    printf '\nimport subprocess\nfrom pathlib import Path\n\n%b' "$snippet" >>"$root/scripts/report/tool.py"
    reindex "$root"
    st_case "caller_python_bound_name_$n" 0 --what "a name bound to the path and passed to a subprocess call is a caller" \
        --has "check-scripts: ok (5 files" -- gate "$root"
done

root=$(fresh caller_py_from_import)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf '\nfrom subprocess import run\n\nSUBJECT = "scripts/check/x.sh"\nrun(["bash", SUBJECT])\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case caller_python_from_subprocess_import 0 --what "run imported from subprocess is a subprocess call" \
    --has "check-scripts: ok (5 files" -- gate "$root"

n=0
for snippet in \
    'SUBJECT = "scripts/check/x.sh"\nprint(SUBJECT)\n' \
    'SUBJECT = "scripts/check/x.sh"\nsize = len(SUBJECT)\n' \
    'SUBJECT = "scripts/check/x.sh"\nsubprocess.run(["true"])\n' \
    'SUBJECT = "scripts/check/x.sh"\nsubprocess.run(["bash", "-c", "true"], cwd=SUBJECT)\n' \
    'def run(path):\n    return path\n\nSUBJECT = "scripts/check/x.sh"\nrun(["bash", SUBJECT])\n'; do
    n=$((n + 1))
    root=$(fresh "caller_py_unbound_$n")
    nocaller_sh 'exit 0' | put "$root" check/x.sh
    printf '\nimport subprocess\n\n%b' "$snippet" >>"$root/scripts/report/tool.py"
    reindex "$root"
    st_case "caller_python_name_not_run_$n" 1 --what "a bound name that no subprocess call takes as its program is no caller" \
        --has "caller: check/x.sh:1:" -- gate "$root"
done

root=$(fresh caller_workflow)
nocaller_sh 'exit 0' | put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'name: ci\njobs:\n  gates:\n    steps:\n      - run: bash scripts/check/x.sh\n' >"$root/.github/workflows/ci.yml"
reindex "$root"
st_case caller_workflow_inline_run 0 --what "a hosted-CI run step is a caller" \
    --has "check-scripts: ok (5 files" -- gate "$root"
st_case caller_workflow_cell 0 --what "and the index names the workflow file" \
    --has '| `check/x.sh` | ci.yml |' -- cat "$root/scripts/INDEX.md"

root=$(fresh caller_workflow_block)
nocaller_sh 'exit 0' | put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'name: ci\njobs:\n  gates:\n    steps:\n      - name: gates\n        run: |\n          set -e\n          bash scripts/check/x.sh\n      - name: after\n        run: echo done\n' >"$root/.github/workflows/ci.yml"
reindex "$root"
st_case caller_workflow_block_run 0 --what "a line of a run: block scalar is a caller" \
    --has '| `check/x.sh` | ci.yml |' -- cat "$root/scripts/INDEX.md"

root=$(fresh caller_workflow_folded)
nocaller_sh 'exit 0' | put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'jobs:\n  gates:\n    steps:\n      - run: >\n          bash scripts/check/x.sh\n' >"$root/.github/workflows/ci.yml"
reindex "$root"
st_case caller_workflow_folded_run 0 --what "a run: > folded scalar is a caller too" \
    --has '| `check/x.sh` | ci.yml |' -- cat "$root/scripts/INDEX.md"

root=$(fresh caller_workflow_text)
nocaller_sh 'exit 0' | put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'name: scripts/check/x.sh\n# bash scripts/check/x.sh\njobs:\n  gates:\n    steps:\n      - name: scripts/check/x.sh\n        run: echo scripts/check/x.sh\n      - run: |\n          echo done\n        shell: bash scripts/check/x.sh\n        env:\n          SUBJECT: scripts/check/x.sh\n' >"$root/.github/workflows/ci.yml"
reindex "$root"
st_case caller_workflow_text_only 1 --what "a workflow name, comment, env value, echo and a key after the run block name the file and run nothing" \
    --has "caller: check/x.sh:1:" -- gate "$root"

root=$(fresh caller_pre_commit)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'repos:\n  - repo: local\n    hooks:\n      - id: x\n        entry: bash scripts/check/x.sh\n        language: system\n' >"$root/.pre-commit-config.yaml"
reindex "$root"
st_case caller_pre_commit_entry 0 --what "a pre-commit entry is a caller, shown as pre-commit" \
    --has '| `check/x.sh` | pre-commit |' -- cat "$root/scripts/INDEX.md"

root=$(fresh caller_pre_commit_text)
nocaller_sh 'exit 0' | put "$root" check/x.sh
printf 'repos:\n  - repo: local\n    hooks:\n      - id: x\n        # entry: bash scripts/check/x.sh\n        name: scripts/check/x.sh\n        entry: true\n        exclude: ^scripts/check/x.sh$\n' >"$root/.pre-commit-config.yaml"
reindex "$root"
st_case caller_pre_commit_text_only 1 --what "a commented entry, a name and an exclude pattern run nothing" \
    --has "caller: check/x.sh:1:" -- gate "$root"

sq="'"
dq='"'
n=0
for src in \
    'const S: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/check/x.sh");' \
    "fn f() { let c = ${sq}${dq}${sq}; let p = ${dq}../../scripts/check/x.sh${dq}; }" \
    "fn f() { let r = r#${dq}a${dq}b${dq}#; let p = ${dq}../../scripts/check/x.sh${dq}; }" \
    "fn f() { let s = ${dq}a\\${dq}b${dq}; let p = ${dq}../../scripts/check/x.sh${dq}; }" \
    "fn f() { let c = ${sq}\\${sq}${sq}; let p = ${dq}../../scripts/check/x.sh${dq}; }" \
    "fn f() { let c = ${sq}\\${dq}${sq}; let p = ${dq}../../scripts/check/x.sh${dq}; }" \
    "fn f() -> &${sq}static str { /* ${dq}scripts/lib/no${dq} */ ${dq}/../../scripts/check/x.sh${dq} }"; do
    n=$((n + 1))
    root=$(fresh "caller_crate_$n")
    nocaller_sh 'exit 0' | put "$root" check/x.sh
    mkdir -p "$root/crates/c/src"
    printf '%s\n' "$src" >"$root/crates/c/src/lib.rs"
    reindex "$root"
    st_case "caller_crate_literal_$n" 0 --what "a string literal that is the whole path is a caller, named by its crate file" \
        --has '| `check/x.sh` | crates/c/src/lib.rs |' -- cat "$root/scripts/INDEX.md"
done

n=0
for src in \
    '// "scripts/check/x.sh"' \
    '/// "scripts/check/x.sh"' \
    '/* "../../scripts/check/x.sh" */' \
    '/* a /* nested */ "scripts/check/x.sh" */' \
    'const S: &str = "see scripts/check/x.sh for the rest";' \
    'const S: &str = "scripts/check/x.sh is the gate";' \
    'const S: &str = "`scripts/check/x.sh`";' \
    'const S: &str = "scripts/check";'; do
    n=$((n + 1))
    root=$(fresh "caller_crate_text_$n")
    nocaller_sh 'exit 0' | put "$root" check/x.sh
    mkdir -p "$root/crates/c/src"
    printf '%s\n' "$src" >"$root/crates/c/src/lib.rs"
    reindex "$root"
    st_case "caller_crate_text_only_$n" 1 --what "not a caller: $src" \
        --has "caller: check/x.sh:1:" --lacks "dead-path" -- gate "$root"
done

root=$(fresh caller_crate_run)
printf '%s\n' '# A planted script with no shebang that a crate runs.' 'exit 0' | put "$root" check/noshebang.sh
mkdir -p "$root/crates/c/src"
printf 'fn run() { Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/check/noshebang.sh")); }\n' >"$root/crates/c/src/lib.rs"
reindex "$root"
st_case caller_crate_command_new_is_a_run 1 --what "a path that is the argument of Command::new makes the file an entry point" \
    --has "usage: check/noshebang.sh:1:" -- gate "$root"

root=$(fresh caller_crate_read)
printf '%s\n' '# A planted script with no shebang that a crate only reads.' 'exit 0' | put "$root" check/noshebang.sh
mkdir -p "$root/crates/c/src"
printf 'const S: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/check/noshebang.sh"));\n' >"$root/crates/c/src/lib.rs"
reindex "$root"
st_case caller_crate_include_str_is_a_read 0 --what "a path that is only read is a caller and no entry point" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh index_xml_purpose)
printf '%s\n' '<?xml version="1.0"?>' '<!--' '  A planted property list read by a script.' '-->' >"$root/scripts/report/x.entitlements"
reindex "$root"
st_case index_xml_comment_is_the_purpose 0 --what "the first text line of an XML comment is the purpose" \
    --has '| `report/x.entitlements` | none | A planted property list read by a script. |' -- cat "$root/scripts/INDEX.md"

# ---- not-a-mode, not-a-mode-stale ----------------------------------------

root=$(fresh nam_missing)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
    '# Caller: by hand' '#' 'exit 0' | put "$root" check/y.sh
reindex "$root"
st_case not_a_mode_missing 1 --what "a new entry point with no Not a mode of: line" \
    --has "not-a-mode: check/y.sh:1:" -- gate "$root"

root=$(fresh nam_no_reason)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
    '# Caller: by hand' '# Not a mode of: check/ok.sh' 'exit 0' | put "$root" check/y.sh
reindex "$root"
st_case not_a_mode_without_reason 1 --what "the line names a tool and gives no reason" \
    --has "not-a-mode: check/y.sh:7: \`Not a mode of:\` gives no reason" -- gate "$root"

root=$(fresh nam_old_file)
st_case not_a_mode_old_file_exempt 0 --what "ok.sh is in the base and carries no such line" \
    --has "check-scripts: ok (4 files" -- gate "$root"

root=$(fresh nam_recall_test)
printf '%s\n' '#!/usr/bin/env bash' '# Recall test for ok.sh: a planted recall test.' '#' \
    '# Recall test for: check/ok.sh' '# Usage:  bash scripts/check/ok_selftest.sh' '# Exit:   0 held' \
    '# Caller: by hand' |
    put "$root" check/ok_selftest.sh
reindex "$root"
st_case not_a_mode_recall_test_exempt 0 --what "a new recall test that names an existing file belongs to it" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh nam_selftest_name_only)
printf '%s\n' '#!/usr/bin/env bash' '# Deploys the release bottle; not a test at all.' '#' \
    '# Usage:  bash scripts/check/deploy_selftest.sh' '# Exit:   0 ok' '# Caller: by hand' 'exit 0' |
    put "$root" check/deploy_selftest.sh
reindex "$root"
st_case not_a_mode_name_alone_is_no_recall_test 1 --what "a _selftest.sh name with no Recall test for: line" \
    --has "not-a-mode: check/deploy_selftest.sh:1:" -- gate "$root"

root=$(fresh nam_recall_nothing)
printf '%s\n' '#!/usr/bin/env bash' '# A new tool that claims to be a recall test.' '#' \
    '# Recall test for: nothing/at_all.sh' '# Usage:  bash scripts/check/tool2.sh' '# Exit:   0 ok' \
    '# Caller: by hand' 'exit 0' | put "$root" check/tool2.sh
reindex "$root"
st_case not_a_mode_recall_test_of_nothing 1 --what "Recall test for: names a file that does not exist" \
    --has "not-a-mode: check/tool2.sh:1:" -- gate "$root"

root=$(fresh nam_recall_itself)
printf '%s\n' '#!/usr/bin/env bash' '# A new tool that claims to be its own recall test.' '#' \
    '# Recall test for: check/tool2.sh' '# Usage:  bash scripts/check/tool2.sh' '# Exit:   0 ok' \
    '# Caller: by hand' 'exit 0' | put "$root" check/tool2.sh
reindex "$root"
st_case not_a_mode_recall_test_of_itself 1 --what "Recall test for: names the file itself" \
    --has "not-a-mode: check/tool2.sh:1:" -- gate "$root"

root=$(fresh nam_module_nowhere)
printf '%s\n' '#!/usr/bin/env bash' '# A new entry point hidden behind a Module of: line.' '#' \
    '# Module of: check/does_not_exist.sh' 'exit 7' | put "$root" check/hidden.sh
reindex "$root"
st_case not_a_mode_module_of_nothing 1 --what "Module of: names no file: an entry point under every rule" \
    --has "not-a-mode: check/hidden.sh:1:" --has "usage: check/hidden.sh:1:" -- gate "$root"

root=$(fresh nam_module_itself)
printf '%s\n' '#!/usr/bin/env bash' '# A new entry point that is a module of itself.' '#' \
    '# Module of: check/hidden.sh' 'exit 7' | put "$root" check/hidden.sh
reindex "$root"
st_case not_a_mode_module_of_itself 1 --what "Module of: names the file itself" \
    --has "not-a-mode: check/hidden.sh:1:" -- gate "$root"

root=$(fresh nam_renamed)
git -C "$root" config diff.renames false
git -C "$root" mv scripts/check/ok.sh scripts/check/ok2.sh
printf 'check-ok:\n\t@bash scripts/check/ok2.sh\n\ntool:\n\t@python3 scripts/report/tool.py\n' >"$root/Makefile"
sed -i.bak 's|Module of: check/ok.sh|Module of: check/ok2.sh|' "$root/scripts/lib/helper.sh"
sed -i.bak 's|scripts/check/ok.sh|scripts/check/ok2.sh|' "$root/scripts/check/ok2.sh"
/bin/rm -f "$root/scripts/lib/helper.sh.bak" "$root/scripts/check/ok2.sh.bak"
reindex "$root"
commit_all "$root" "move ok.sh"
st_case not_a_mode_moved_file_is_not_new 0 --what "a committed git mv of an old file is not a new file" \
    --has "check-scripts: ok (4 files" --lacks "not-a-mode" -- gate_at "$root" HEAD^

root=$(fresh nam_unstaged_move)
mv "$root/scripts/check/ok.sh" "$root/scripts/check/ok2.sh"
printf 'check-ok:\n\t@bash scripts/check/ok2.sh\n\ntool:\n\t@python3 scripts/report/tool.py\n' >"$root/Makefile"
reindex "$root"
st_case not_a_mode_unstaged_move_is_new 1 --what "a plain mv, not staged, reads as a new file" \
    --has "not-a-mode: check/ok2.sh:1:" -- gate "$root"

root=$(fresh nam_module)
printf '%s\n' '"""A planted module of the fixture tool.' '' 'Module of: report/tool.py' '"""' |
    put "$root" report/tool_two.py
printf 'import tool_two\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case not_a_mode_module_exempt 0 --what "a new module belongs to its tool" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh nam_stale)
python3 - "$root/scripts/check/ok.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("set -euo", "# Not a mode of: check/gone.sh — a tool deleted since\nset -euo")
open(path, "w").write(text)
EOF
reindex "$root"
st_case not_a_mode_stale 1 --what "an old file names a tool that no longer exists, at its own line" \
    --has "not-a-mode-stale: check/ok.sh:6: names check/gone.sh, which is not another entry point" -- gate "$root"

root=$(fresh nam_names_existing)
python3 - "$root/scripts/check/ok.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("set -euo", "# Not a mode of: report/tool.py — a tool that exists\nset -euo")
open(path, "w").write(text)
EOF
reindex "$root"
st_case not_a_mode_names_existing 0 --what "the same line naming an existing entry point passes" \
    --has "check-scripts: ok (4 files" -- gate "$root"

n=0
for target in 'check/ok.sh' 'README.md' 'check' 'lib/helper.sh' 'INDEX.md'; do
    n=$((n + 1))
    root=$(fresh "nam_target_$n")
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
        '# Caller: by hand' "# Not a mode of: $target — a reason" 'exit 0' | put "$root" check/x.sh
    reindex "$root"
    if [ "$target" = check/ok.sh ]; then
        st_case "not_a_mode_target_$n" 0 --what "naming another entry point ($target) passes" \
            --has "check-scripts: ok (5 files" -- gate "$root"
        entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
            '# Caller: by hand' "# Not a mode of: check/x.sh — a reason" 'exit 0' | put "$root" check/x.sh
        reindex "$root"
        st_case "not_a_mode_target_itself" 1 --what "naming the file itself fails" \
            --has "not-a-mode-stale: check/x.sh:7: names check/x.sh, which is not another entry point" -- gate "$root"
    else
        st_case "not_a_mode_target_$n" 1 --what "naming $target is not naming another entry point" \
            --has "not-a-mode-stale: check/x.sh:7: names $target, which is not another entry point" -- gate "$root"
    fi
done

n=0
for separator in ' -- ' ' - '; do
    n=$((n + 1))
    root=$(fresh "nam_separator_$n")
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
        '# Caller: by hand' "# Not a mode of: check/ok.sh${separator}a reason" 'exit 0' | put "$root" check/x.sh
    reindex "$root"
    st_case "not_a_mode_separator_$n" 0 --what "the tool and its reason, separated by '${separator}'" \
        --has "check-scripts: ok (5 files" -- gate "$root"
done

# ---- pins ----------------------------------------------------------------

PINNABLE='# pinnable: class purpose usage exit-undeclared caller recall-missing lib-consumers lib-run shebang exec-bit dead-path'
pins_file() {
    printf '%s\n' '# Accepted violations of the fixture tree.' '#' "$PINNABLE" "$@"
}

# pinned_base <name> [header]: the base tree plus check/p.sh, which has no Usage: line and
# an undeclared exit 7, both pinned; committed, so these pins are the base's.
pinned_base() {
    local r
    r=$(fresh "$1")
    entry_sh '# CI gate: a pinned entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$NAM" 'exit 7' |
        put "$r" check/p.sh
    printf '%s\n' '# Accepted violations of the fixture tree.' '#' "${2:-$PINNABLE}" \
        'usage check/p.sh' 'exit-undeclared check/p.sh 7' >"$r/$PINS_REL"
    reindex "$r"
    commit_all "$r" "pin p.sh"
    printf '%s' "$r"
}

root=$(fresh pin_holds)
entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
pins_file 'usage check/x.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_accepts_a_violation 0 --what "a pinned violation passes and is counted" \
    --has "pinned: usage 1" --has "check-scripts: ok (6 files, 1 pinned)" \
    --has "pins: HEAD has no pin list; growth not checked" -- gate "$root"

root=$(fresh pin_other_rule)
entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$NAM" 'exit 7' |
    put "$root" check/x.sh
pins_file 'usage check/x.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_does_not_cover_another_rule 1 --what "a pin for usage does not hide an undeclared exit" \
    --has "exit-undeclared: check/x.sh:9: 7" --lacks "usage: check/x.sh" -- gate "$root"

root=$(fresh pin_stale)
valid_sh 'exit 0' | put "$root" check/x.sh
pins_file 'usage check/x.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_stale_file_now_passes 1 --what "a pinned file that now passes fails, at the pin's own line" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: usage check/x.sh: no longer violated" -- gate "$root"

root=$(fresh pin_stale_gone)
pins_file 'usage check/gone.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_stale_file_gone 1 --what "a pin on a file that does not exist" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: usage check/gone.sh" -- gate "$root"

root=$(fresh pin_unknown_rule)
pins_file 'no-such-rule check/ok.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_stale_unknown_rule 1 --what "a pin naming no rule" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: no-such-rule check/ok.sh" -- gate "$root"

root=$(fresh pin_index)
pins_file 'index-stale INDEX.md' >"$root/$PINS_REL"
reindex "$root"
st_case pin_cannot_hold_index_stale 1 --what "index-stale cannot be pinned" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: index-stale INDEX.md" -- gate "$root"

root=$(pinned_base pin_base_holds)
st_case pin_list_equal_to_the_base_holds 0 --what "pins equal to the base's: nothing grew, no bootstrap note" \
    --has "pinned: usage 1, exit-undeclared 1" --lacks "pin-new" --lacks "has no pin list" -- gate "$root"

root=$(pinned_base pin_growth)
entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/foo.sh
printf 'usage check/foo.sh\n' >>"$root/$PINS_REL"
reindex "$root"
st_case pin_new_for_a_new_file 1 --what "a pin for a file the base did not pin, for a rule the base pinned" \
    --has "pin-new: report/scripts_gate_pins.txt:6: usage check/foo.sh is not pinned at the base" -- gate "$root"

root=$(pinned_base pin_duplicate)
printf 'usage check/p.sh\n' >>"$root/$PINS_REL"
st_case pin_duplicate 1 --what "the same pin twice" \
    --has "pin-dup: report/scripts_gate_pins.txt:6: usage check/p.sh is already pinned at line 4" -- gate "$root"

root=$(pinned_base pin_unpinnable)
printf 'not-a-mode check/p.sh\n' >>"$root/$PINS_REL"
st_case pin_unpinnable_rule 1 --what "not-a-mode cannot be pinned" \
    --has "pin-stale: report/scripts_gate_pins.txt:6: not-a-mode check/p.sh: not-a-mode cannot be pinned" -- gate "$root"

root=$(pinned_base pin_unpinnable_violation)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/foo.sh' '# Exit:   0 clean' \
    '# Caller: by hand' '#' 'exit 0' | put "$root" check/foo.sh
printf 'not-a-mode check/foo.sh\n' >>"$root/$PINS_REL"
reindex "$root"
st_case pin_does_not_hide_an_unpinnable_violation 1 --what "a pin for not-a-mode leaves the not-a-mode violation standing" \
    --has "not-a-mode: check/foo.sh:1:" \
    --has "pin-stale: report/scripts_gate_pins.txt:6: not-a-mode check/foo.sh: not-a-mode cannot be pinned" -- gate "$root"

root=$(pinned_base pin_header "# pinnable: class purpose")
st_case pin_header_mismatch 1 --what "the pinnable line differs from the gate's rule set" \
    --has "pin-header: report/scripts_gate_pins.txt:3:" -- gate "$root"

root=$(pinned_base pin_new_exit)
printf 'exit 99\n' >>"$root/scripts/check/p.sh"
st_case pin_exit_code_beyond_the_set 1 --what "a pinned file grows another undeclared exit" \
    --has "exit-undeclared: check/p.sh:10: 99 is not declared" -- gate "$root"

root=$(pinned_base pin_exit_declared)
python3 - "$root/scripts/check/p.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Exit:   0 clean\n", "# Exit:   0 clean; 7 a violation\n")
open(path, "w").write(text)
EOF
st_case pin_exit_only_code_declared 1 --what "the file's only undeclared code is now declared: the pin is stale" \
    --has "pin-stale: report/scripts_gate_pins.txt:5: exit-undeclared check/p.sh: no longer violated" -- gate "$root"

root=$(pinned_base pin_exit_declared_other)
python3 - "$root/scripts/check/p.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Exit:   0 clean\n", "# Exit:   0 clean; 7 a violation\n") + "exit 99\n"
open(path, "w").write(text)
EOF
st_case pin_exit_code_now_declared 1 --what "a code in the pinned set is declared while another is not" \
    --has "pin-stale: report/scripts_gate_pins.txt:5: exit-undeclared check/p.sh: code 7 is fixed or gone" \
    --has "exit-undeclared: check/p.sh:10: 99 is not declared" -- gate "$root"

root=$(pinned_base pin_exit_grown_set)
printf 'exit 99\n' >>"$root/scripts/check/p.sh"
python3 - "$root/$PINS_REL" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("check/p.sh 7\n", "check/p.sh 7,99\n")
open(path, "w").write(text)
EOF
st_case pin_exit_set_grows 1 --what "the pinned code set gains a code the base did not pin" \
    --has "pin-new: report/scripts_gate_pins.txt:5: exit-undeclared check/p.sh: code 99 is not pinned at the base" \
    --lacks "exit-undeclared: check/p.sh" -- gate "$root"

root=$(pinned_base pin_two_rules)
python3 - "$root/scripts/check/p.sh" <<'EOF'
import sys

path = sys.argv[1]
lines = open(path).read().split("\n")
lines[3] = "# Usage:  bash scripts/check/p.sh"
open(path, "w").write("\n".join(lines))
EOF
st_case pin_one_of_two_rules_fixed 1 --what "a file pinned for two rules that fixes one: that pin is stale, the other holds" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: usage check/p.sh: no longer violated" \
    --lacks "exit-undeclared check/p.sh: code" -- gate "$root"

root=$(pinned_base pin_moved)
git -C "$root" mv scripts/check/p.sh scripts/check/q.sh
python3 - "$root/$PINS_REL" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("check/p.sh", "check/q.sh")
open(path, "w").write(text)
EOF
reindex "$root"
st_case pin_follows_a_moved_file 0 --what "a git mv with its pin lines renamed keeps the pins" \
    --has "pinned: usage 1, exit-undeclared 1" --lacks "pin-new" -- gate "$root"

root=$(pinned_base pin_new_rule "# pinnable: class purpose usage exit-undeclared")
nocaller_sh 'exit 0' | put "$root" check/r.sh
python3 - "$root/$PINS_REL" "$PINNABLE" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# pinnable: class purpose usage exit-undeclared\n", sys.argv[2] + "\n", 1)
open(path, "w").write(text)
EOF
printf 'caller check/r.sh\n' >>"$root/$PINS_REL"
reindex "$root"
st_case pin_seeds_a_rule_the_base_lacked 0 --what "a rule absent from the base's pinnable line may seed pins" \
    --has "pinned: usage 1, exit-undeclared 1, caller 1" -- gate "$root"

# ---- recall-missing, recall-stale ----------------------------------------

# gate_body: the body of a planted tool that answers --lib, so that mode occurs in its text.
GATE_BODY='[ "${1:-}" = --lib ] || exit 0'

# make_ci <root> <text>: append <text> (printf %b) to the fixture Makefile.
make_ci() {
    printf '%b' "$2" >>"$1/Makefile"
}

root=$(fresh recall_make_call)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@$(MAKE) --no-print-directory gates\n\ngates:\n\t@bash scripts/check/gate.sh\n'
reindex "$root"
st_case recall_missing_through_make_call 1 --what "a tool reached only through a \$(MAKE) line of ci has no recall test" \
    --has "recall-missing: check/gate.sh:1: mode default is run by \`make ci\` and no \`Recall test for:\` names it" -- gate "$root"

root=$(fresh recall_prerequisite)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci: gates\n\ngates:\n\t@bash scripts/check/gate.sh\n'
reindex "$root"
st_case recall_missing_through_prerequisite 1 --what "a tool run by a prerequisite of ci has no recall test" \
    --has "recall-missing: check/gate.sh:1: mode default" -- gate "$root"

root=$(fresh recall_cycle)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@$(MAKE) a\n\na:\n\t@$(MAKE) ci\n\t@bash scripts/check/gate.sh\n'
reindex "$root"
st_case recall_closure_survives_a_cycle 1 --what "targets that call each other end the walk and the tool inside is still found" \
    --has "recall-missing: check/gate.sh:1: mode default" --lacks "timed out" -- bounded_gate "$root"

root=$(fresh recall_not_in_ci)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@echo hi\n\nother:\n\t@bash scripts/check/gate.sh\n'
reindex "$root"
st_case recall_tool_outside_ci_asks_nothing 0 --what "a tool only another target runs needs no recall test" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh recall_advisory)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || true\n'
reindex "$root"
st_case recall_advisory_line_asks_nothing 0 --what "a recipe line ending in || true gates nothing" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh recall_advisory_then_more)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || true; echo done\n'
reindex "$root"
st_case recall_advisory_line_must_end_there 1 --what "|| true that is followed by another command does not make the line advisory" \
    --has "recall-missing: check/gate.sh:1: mode default" -- gate "$root"

root=$(fresh recall_covered)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '#' 'check/gate.sh' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate_selftest.sh\n'
reindex "$root"
st_case recall_covered_by_a_test_ci_runs 0 --what "a recall test that ci runs covers the default mode, and asks for none itself" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh recall_test_not_run)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '# Caller: by hand' 'check/gate.sh' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n'
reindex "$root"
st_case recall_test_ci_does_not_run 1 --what "a recall test that make ci never runs does not cover the tool" \
    --has "recall-missing: check/gate.sh:1: mode default: its recall test check/gate_selftest.sh is not run by \`make ci\`" -- gate "$root"

root=$(fresh recall_mode_unnamed)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '#' 'check/gate.sh' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
reindex "$root"
st_case recall_mode_not_named 1 --what "ci runs the tool with --lib and the test names only the default mode" \
    --has "recall-missing: check/gate.sh:1: mode --lib is run by" --lacks "mode default" -- gate "$root"

root=$(fresh recall_mode_named)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '#' 'check/gate.sh --lib' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
reindex "$root"
st_case recall_mode_named 0 --what "the test names --lib: covered" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh recall_non_mode_argument)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '#' 'check/gate.sh' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh some/path.txt "$(X)" 42\n\t@bash scripts/check/gate_selftest.sh\n'
reindex "$root"
st_case recall_argument_that_is_no_mode_is_the_default 0 --what "a path, a substitution and a number as first argument leave the default mode" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh recall_two_modes_one_line)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '#' 'check/gate.sh default --lib' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
reindex "$root"
st_case recall_two_modes_on_one_line 0 --what "one field line may name several modes" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh recall_two_modes_two_lines)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '#' 'check/gate.sh' 'check/gate.sh --lib' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
reindex "$root"
st_case recall_two_modes_on_two_lines 0 --what "a field of two lines covers both" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh recall_stale_tool)
valid_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '# Caller: by hand' 'check/gone.sh' | put "$root" check/gate_selftest.sh
reindex "$root"
st_case recall_stale_names_no_file 1 --what "Recall test for: names a file that does not exist, at the field's line" \
    --has "recall-stale: check/gate_selftest.sh:4: names check/gone.sh, which is no file of the tree" -- gate "$root"

root=$(fresh recall_stale_mode)
valid_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '# Caller: by hand' 'check/gate.sh --nope' | put "$root" check/gate_selftest.sh
reindex "$root"
st_case recall_stale_mode_not_in_tool 1 --what "a mode token that occurs nowhere in the tool" \
    --has "recall-stale: check/gate_selftest.sh:4: check/gate.sh: \`--nope\` occurs nowhere in check/gate.sh" -- gate "$root"

root=$(fresh recall_stale_token)
valid_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '# Caller: by hand' 'check/gate.sh (all modes)' | put "$root" check/gate_selftest.sh
reindex "$root"
st_case recall_stale_prose_after_the_tool 1 --what "words after the tool are mode tokens, and a parenthesis is not one" \
    --has "recall-stale: check/gate_selftest.sh:4: check/gate.sh: \`(all\` is not a mode token" -- gate "$root"

root=$(fresh recall_stale_self)
recall_sh '# Caller: by hand' 'check/gate_selftest.sh' | put "$root" check/gate_selftest.sh
reindex "$root"
st_case recall_stale_names_itself 1 --what "a recall test of itself" \
    --has "recall-stale: check/gate_selftest.sh:4: \`Recall test for:\` names the file itself" -- gate "$root"

root=$(fresh recall_stale_second_line)
valid_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '# Caller: by hand' 'check/gate.sh' 'check/gone.sh' | put "$root" check/gate_selftest.sh
reindex "$root"
st_case recall_stale_second_line 1 --what "the second line of the field is the one named" \
    --has "recall-stale: check/gate_selftest.sh:5: names check/gone.sh" --lacks "gate_selftest.sh:4" -- gate "$root"

root=$(fresh recall_valid_claim)
valid_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '# Caller: by hand' 'check/gate.sh --lib' | put "$root" check/gate_selftest.sh
reindex "$root"
st_case recall_valid_claim 0 --what "a claim on a tool and a mode that occurs in it" \
    --has "check-scripts: ok (6 files" -- gate "$root"

# ---- lib-consumers, lib-run ----------------------------------------------

# lib_sh [<last header line>]: a planted lib/ file, no shebang, sourced and never run.
lib_sh() {
    printf '%s\n' '# A planted lib file.' '#' "${1:-# Sourced, not executed.}" 'liby() { :; }'
}

SRC_Y='. "$(dirname "$0")/../lib/y.sh"'

root=$(fresh libc_one)
lib_sh | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case lib_consumers_one 1 --what "a lib/ file one script sources" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s), needs 2 or a \`Module of:\` line" -- gate "$root"

root=$(fresh libc_two_scripts)
lib_sh | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
valid_sh "$SRC_Y" | put "$root" check/z.sh
reindex "$root"
st_case lib_consumers_two_scripts 0 --what "two scripts source it" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh libc_script_and_crate)
lib_sh | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
mkdir -p "$root/crates/c/src"
printf 'const S: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/y.sh"));\n' >"$root/crates/c/src/lib.rs"
reindex "$root"
st_case lib_consumers_script_and_crate 0 --what "one script and one crates/ source count as two" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh libc_zero)
lib_sh | put "$root" lib/y.sh
reindex "$root"
st_case lib_consumers_none 1 --what "a lib/ file nothing uses, and it is not an orphan" \
    --has "lib-consumers: lib/y.sh:1: 0 consumer(s)" --lacks "orphan" -- gate "$root"

root=$(fresh libc_twice)
lib_sh | put "$root" lib/y.sh
printf '%s\n%s\n' "$SRC_Y" "$SRC_Y" >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case lib_consumers_one_file_counts_once 1 --what "one script that sources it twice is one consumer" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s)" -- gate "$root"

root=$(fresh libc_recall_test)
lib_sh | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
recall_sh '# Caller: by hand' 'check/ok.sh' | sed '$d' | put "$root" check/ok_selftest.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok_selftest.sh"
reindex "$root"
st_case lib_consumers_recall_test_not_counted 1 --what "a script and a recall test: the recall test does not count" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s)" -- gate "$root"

root=$(fresh libc_selftest_support)
lib_sh | put "$root" lib/selftest_support.sh
printf '. "$(dirname "$0")/../lib/selftest_support.sh"\n' >>"$root/scripts/check/ok.sh"
recall_sh '# Caller: by hand' 'check/ok.sh' | sed '$d' | put "$root" check/ok_selftest.sh
printf '. "$(dirname "$0")/../lib/selftest_support.sh"\n' >>"$root/scripts/check/ok_selftest.sh"
reindex "$root"
st_case lib_consumers_selftest_support_counts_recall_tests 0 --what "lib/selftest_support.sh: a script and a recall test are two" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh libc_hook)
printf '%s\n' '# A planted lib data file.' >"$root/scripts/lib/data.txt"
printf '%s\n' 'cat "$(dirname "$0")/../lib/data.txt"' >>"$root/scripts/check/ok.sh"
printf 'repos:\n  - repo: local\n    hooks:\n      - id: x\n        entry: cat scripts/lib/data.txt\n' >"$root/.pre-commit-config.yaml"
reindex "$root"
st_case lib_consumers_pre_commit_is_no_consumer 1 --what "a hook that reads a lib/ file is not a script or a crate: one consumer" \
    --has "lib-consumers: lib/data.txt:1: 1 consumer(s)" -- gate "$root"

root=$(fresh libc_module_of)
lib_sh '# Module of: check/ok.sh' | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case lib_consumers_module_of_exempts 0 --what "Module of: its one consumer: not counted" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh libc_module_of_missing)
lib_sh '# Module of: check/none.sh' | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case lib_consumers_module_of_naming_nothing 1 --what "a Module of: that names no file exempts nothing" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s)" --has "module-of: lib/y.sh:3: names check/none.sh" -- gate "$root"

# lib_py: a planted lib/ Python file with the header of an entry point, for the cases where something runs it.
lib_py() {
    printf '%s\n' '"""A planted lib module.' '' 'Usage:  python3 scripts/lib/y.py' 'Exit:   0 always' \
        'Not a mode of: report/tool.py — a planted entry point' '"""'
}

root=$(fresh librun_recipe)
lib_py | put "$root" lib/y.py
make_ci "$root" 'lib-y:\n\t@python3 scripts/lib/y.py\n'
reindex "$root"
st_case lib_run_by_a_recipe 1 --what "a recipe runs a lib/ file, named by its target" \
    --has "lib-run: lib/y.py:1: lib/ is sourced or imported, never run; run by make:lib-y" -- gate "$root"

root=$(fresh librun_script)
lib_py | put "$root" lib/y.py
printf '%s\n' 'python3 "$(dirname "$0")/../lib/y.py"' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case lib_run_by_a_script 1 --what "a script runs a lib/ file, named by its path" \
    --has "lib-run: lib/y.py:1: lib/ is sourced or imported, never run; run by check/ok.sh" -- gate "$root"

root=$(fresh librun_workflow)
lib_py | put "$root" lib/y.py
mkdir -p "$root/.github/workflows"
printf 'jobs:\n  gates:\n    steps:\n      - run: python3 scripts/lib/y.py\n' >"$root/.github/workflows/ci.yml"
reindex "$root"
st_case lib_run_by_a_workflow 1 --what "a hosted-CI step runs a lib/ file" \
    --has "run by ci.yml" -- gate "$root"

root=$(fresh librun_pre_commit)
lib_py | put "$root" lib/y.py
printf 'repos:\n  - repo: local\n    hooks:\n      - id: y\n        entry: python3 scripts/lib/y.py\n' >"$root/.pre-commit-config.yaml"
reindex "$root"
st_case lib_run_by_pre_commit 1 --what "a pre-commit entry runs a lib/ file" \
    --has "run by pre-commit" -- gate "$root"

root=$(fresh librun_crate)
lib_py | put "$root" lib/y.py
mkdir -p "$root/crates/c/src"
printf 'fn run() { Command::new("python3").arg(1); Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/y.py")); }\n' >"$root/crates/c/src/lib.rs"
reindex "$root"
st_case lib_run_by_a_crate 1 --what "Command::new on the path runs a lib/ file" \
    --has "run by crates/c/src/lib.rs" -- gate "$root"

root=$(fresh librun_crate_read)
lib_py | put "$root" lib/y.py
mkdir -p "$root/crates/c/src"
printf 'fn run() { Command::new("python3").arg(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/y.py")); }\nconst S: &str = include_str!("../../../scripts/lib/y.py");\n' >"$root/crates/c/src/lib.rs"
printf 'import y\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case lib_run_crate_reads_and_imports_are_not_runs 0 --what "an argument after Command::new, include_str! and an import read the file" \
    --has "check-scripts: ok (" -- gate "$root"

root=$(fresh librun_sourced)
lib_sh '# Module of: check/ok.sh' | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case lib_run_sourcing_is_no_run 0 --what "sourcing a lib/ file is what it is for" \
    --has "check-scripts: ok (5 files" -- gate "$root"

# ---- module-of -----------------------------------------------------------

# mod_py <named file>: a planted Python module whose third line is `Module of:`.
mod_py() {
    printf '%s\n' '"""A planted module.' '' "Module of: $1" '"""'
}

root=$(fresh modof_missing)
mod_py report/none.py | put "$root" report/m.py
reindex "$root"
st_case module_of_names_no_file 1 --what "Module of: names a file that does not exist" \
    --has "module-of: report/m.py:3: names report/none.py, which is no file of the tree" -- gate "$root"

root=$(fresh modof_self)
mod_py report/m.py | put "$root" report/m.py
reindex "$root"
st_case module_of_names_itself 1 --what "Module of: names the file itself" \
    --has "module-of: report/m.py:3: \`Module of:\` names the file itself" -- gate "$root"

root=$(fresh modof_unreached)
mod_py report/tool.py | put "$root" report/m.py
reindex "$root"
st_case module_of_tool_does_not_import_it 1 --what "the named tool never imports the module" \
    --has "module-of: report/m.py:3: report/tool.py does not run, source or import it" -- gate "$root"

root=$(fresh modof_transitive)
printf '%s\n' '"""A planted module.' '' 'Module of: report/tool.py' '"""' 'import leaf' | put "$root" report/mid.py
mod_py report/tool.py | put "$root" report/leaf.py
printf 'import mid\n' >>"$root/scripts/report/tool.py"
reindex "$root"
st_case module_of_through_another_module 0 --what "the tool imports mid, mid imports leaf: both are its modules" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh modof_directory)
mod_py report | put "$root" report/m.py
reindex "$root"
st_case module_of_names_a_directory 1 --what "Module of: names a directory" \
    --has "module-of: report/m.py:3: names report, which is no file of the tree" -- gate "$root"

# ---- shebang, exec-bit ---------------------------------------------------

root=$(fresh shebang_py_bash)
printf '%s\n' '#!/usr/bin/env bash' '"""A planted Python entry point with a shell shebang.' '' \
    'Usage:  python3 scripts/report/tool.py' 'Exit:   0 always' \
    'Not a mode of: report/tool.py — a planted entry point' 'Caller: by hand' '"""' 'raise SystemExit(0)' |
    put "$root" report/y.py
reindex "$root"
st_case shebang_python_with_bash 1 --what "a .py entry point with a bash shebang" \
    --has "shebang: report/y.py:1: entry point has #!/usr/bin/env bash, expected #!/usr/bin/env python3" -- gate "$root"

root=$(fresh shebang_sh_bin_bash)
valid_sh 'exit 0' | sed '1s|.*|#!/bin/bash|' | put "$root" check/y.sh
reindex "$root"
st_case shebang_sh_not_env 1 --what "#!/bin/bash is not #!/usr/bin/env bash" \
    --has "shebang: check/y.sh:1: entry point has #!/bin/bash, expected #!/usr/bin/env bash" -- gate "$root"

root=$(fresh shebang_sh_python)
valid_sh 'exit 0' | sed '1s|.*|#!/usr/bin/env python3|' | put "$root" check/y.sh
reindex "$root"
st_case shebang_sh_with_python 1 --what "a .sh entry point with a python shebang" \
    --has "shebang: check/y.sh:1: entry point has #!/usr/bin/env python3, expected #!/usr/bin/env bash" -- gate "$root"

root=$(fresh shebang_none)
printf '%s\n' '# A planted script with no shebang that a recipe runs.' 'exit 0' | put "$root" check/noshebang.sh
printf 'check-noshebang:\n\t@bash scripts/check/noshebang.sh\n' >>"$root/Makefile"
reindex "$root"
st_case shebang_missing_on_a_run_file 1 --what "an entry point a recipe runs, with no shebang" \
    --has "shebang: check/noshebang.sh:1: entry point has no shebang, expected #!/usr/bin/env bash" -- gate "$root"

root=$(fresh shebang_lib)
printf '%s\n' '#!/usr/bin/env bash' '# A planted lib file with a shebang.' 'liby() { :; }' | put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case shebang_on_a_lib_file 1 --what "a lib/ file carries a shebang" \
    --has "shebang: lib/y.sh:1: a module or lib/ file carries a shebang" -- gate "$root"

root=$(fresh exec_untracked)
valid_sh 'exit 0' | put "$root" check/y.sh
chmod -x "$root/scripts/check/y.sh"
reindex "$root"
st_case exec_bit_missing_on_an_untracked_entry 1 --what "an untracked entry point is judged by its file mode" \
    --has "exec-bit: check/y.sh:1: entry point without the exec bit" -- gate "$root"

root=$(fresh exec_tracked)
valid_sh 'exit 0' | put "$root" check/y.sh
chmod -x "$root/scripts/check/y.sh"
reindex "$root"
commit_all "$root" "adds y"
st_case exec_bit_missing_on_a_tracked_entry 1 --what "a tracked entry point is judged by the index" \
    --has "exec-bit: check/y.sh:1: entry point without the exec bit" -- gate "$root"
chmod +x "$root/scripts/check/y.sh"
st_case exec_bit_index_wins_over_the_file 1 --what "the file system says executable, the index does not: the index wins" \
    --has "exec-bit: check/y.sh:1:" -- gate "$root"
git -C "$root" update-index --chmod=+x scripts/check/y.sh
st_case exec_bit_set_in_the_index 0 --what "with the bit in the index the entry point passes" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exec_module)
chmod +x "$root/scripts/report/tool_mod.py"
git -C "$root" update-index --chmod=+x scripts/report/tool_mod.py
st_case exec_bit_on_a_module 1 --what "a module with the exec bit" \
    --has "exec-bit: report/tool_mod.py:1: a module or lib/ file with the exec bit" -- gate "$root"

root=$(fresh exec_lib)
chmod +x "$root/scripts/lib/helper.sh"
git -C "$root" update-index --chmod=+x scripts/lib/helper.sh
st_case exec_bit_on_a_lib_file 1 --what "a lib/ file with the exec bit" \
    --has "exec-bit: lib/helper.sh:1: a module or lib/ file with the exec bit" -- gate "$root"

# ---- orphan --------------------------------------------------------------

root=$(fresh orphan_data)
printf '%s\n' '# A planted data file nothing reads.' >"$root/scripts/report/data.txt"
reindex "$root"
st_case orphan_ignores_data_files 0 --what "a data file nothing reads is not an orphan" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh orphan_sh)
printf '%s\n' '# A planted shell file with no shebang that nothing uses.' 'exit 0' | put "$root" check/o.sh
reindex "$root"
st_case orphan_shell_file 1 --what "no shebang, no caller, no Module of:" \
    --has "orphan: check/o.sh:1: no shebang and nothing runs, sources or imports it" -- gate "$root"

# ---- dead-path -----------------------------------------------------------

root=$(fresh dead_path)
printf '%s\n' '# see scripts/check/gone.sh' >>"$root/scripts/check/ok.sh"
make_ci "$root" 'check-gone:\n\t@echo scripts/check/gone.sh\n'
reindex "$root"
ok_line=$(grep -c '' "$root/scripts/check/ok.sh")
make_line=$(grep -c '' "$root/Makefile")
st_case dead_path_in_a_script_and_the_makefile 1 --what "scripts/check/gone.sh named in a script and in a recipe" \
    --has "dead-path: check/ok.sh:${ok_line}: scripts/check/gone.sh names no file" \
    --has "dead-path: ../Makefile:${make_line}: scripts/check/gone.sh names no file" -- gate "$root"

root=$(fresh dead_path_alive)
{
    printf '%s\n' '# scripts/check/ok.sh is a file, scripts/check a directory, scripts/check/ok.sh. ends a sentence'
    printf '%s\n' '# patterns: scripts/nope/*.sh scripts/nope/<name>.sh scripts/nope/{a,b}.sh scripts/nope/$X scripts/nope/file(_x)?\.sh'
    printf '%s\n' '# more patterns: scripts/nope/file[ab].sh scripts/nope/file?.sh scripts/nope/file+.sh scripts/nope/a|b scripts/nope/file\.sh'
    printf '%s\n' '# the bare directory: scripts/ and scripts/lib/'
} >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case dead_path_names_that_exist_or_are_patterns 0 --what "files, directories, trailing dots and patterns are not dead" \
    --has "check-scripts: ok (4 files" -- gate "$root"

root=$(fresh dead_path_ignored)
printf '%s\n' 'scripts/check/out.json' >"$root/.gitignore"
printf '%s\n' '# writes scripts/check/out.json' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case dead_path_git_ignored_is_a_produced_file 0 --what "a path git ignores is a produced file" \
    --has "check-scripts: ok (4 files" -- gate "$root"

root=$(fresh dead_path_not_ignored)
printf '%s\n' '# writes scripts/check/out.json' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case dead_path_same_token_not_ignored 1 --what "the same token with no ignore rule is dead" \
    --has "dead-path: check/ok.sh:" --has "scripts/check/out.json names no file" -- gate "$root"

root=$(fresh dead_path_recall_test)
valid_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '# Caller: by hand' 'check/gate.sh' | put "$root" check/gate_selftest.sh
printf '%s\n' 'echo scripts/check/planted.sh' >>"$root/scripts/check/gate_selftest.sh"
reindex "$root"
st_case dead_path_recall_test_plants_trees 0 --what "a recall test plants files: its tokens are not read" \
    --has "check-scripts: ok (6 files" -- gate "$root"

root=$(fresh dead_path_docs)
mkdir -p "$root/docs" "$root/.github/workflows"
printf 'one\nsee scripts/check/gone.sh\n' >"$root/docs/A.md"
printf 'see scripts/check/gone.sh\n' >"$root/CLAUDE.md"
printf 'see scripts/check/gone.sh\n' >"$root/CONTRIBUTING.md"
printf 'see scripts/check/gone.sh\n' >"$root/README.md"
printf 'see scripts/check/gone.sh\n' >"$root/CHANGELOG.md"
printf 'jobs:\n  g:\n    steps:\n      - run: echo done # scripts/check/gone.sh\n' >"$root/.github/workflows/ci.yml"
printf 'repos: []  # scripts/check/gone.sh\n' >"$root/.pre-commit-config.yaml"
reindex "$root"
st_case dead_path_in_guides_docs_and_configs 1 --what "a doc, the guides, a workflow and the pre-commit config, but not the changelog" \
    --has "dead-path: ../docs/A.md:2: scripts/check/gone.sh" --has "dead-path: ../CLAUDE.md:1:" \
    --has "dead-path: ../CONTRIBUTING.md:1:" --has "dead-path: ../README.md:1:" \
    --has "dead-path: ../.github/workflows/ci.yml:4:" --has "dead-path: ../.pre-commit-config.yaml:1:" \
    --lacks "CHANGELOG" -- gate "$root"

root=$(fresh dead_path_crate)
mkdir -p "$root/crates/c/src"
printf 'const S: &str = include_str!("../../../scripts/check/gone.sh");\n' >"$root/crates/c/src/lib.rs"
reindex "$root"
st_case dead_path_in_a_crate_literal 1 --what "a crates/ string literal naming a file that is gone" \
    --has "dead-path: ../crates/c/src/lib.rs:1: scripts/check/gone.sh names no file" -- gate "$root"

# ---- pins of the new rules ----------------------------------------------

root=$(fresh pinset_recall_holds)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n'
pins_file 'recall-missing check/gate.sh default' >"$root/$PINS_REL"
reindex "$root"
st_case pin_set_recall_missing_holds 0 --what "a pinned mode passes and the rule is counted" \
    --has "pinned: recall-missing 1" -- gate "$root"

root=$(fresh pinset_recall_new_mode)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n'
pins_file 'recall-missing check/gate.sh default' >"$root/$PINS_REL"
reindex "$root"
st_case pin_set_recall_missing_new_mode 1 --what "a mode outside the pinned set fails the rule" \
    --has "recall-missing: check/gate.sh:1: mode --lib" --lacks "mode default" -- gate "$root"

root=$(fresh pinset_recall_stale_mode)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
recall_sh '#' 'check/gate.sh' | put "$root" check/gate_selftest.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
pins_file 'recall-missing check/gate.sh default,--lib' >"$root/$PINS_REL"
reindex "$root"
st_case pin_set_recall_missing_stale_mode 1 --what "a pinned mode a recall test now covers is stale while the other pinned mode holds" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: recall-missing check/gate.sh: mode default is fixed or gone" \
    --lacks "mode --lib" -- gate "$root"

root=$(fresh pinset_lib_run)
lib_py | put "$root" lib/y.py
make_ci "$root" 'liba:\n\t@python3 scripts/lib/y.py\n'
pins_file 'lib-run lib/y.py make:liba' 'lib-consumers lib/y.py' >"$root/$PINS_REL"
reindex "$root"
st_case pin_set_lib_run_holds 0 --what "a pinned running site passes" \
    --has "pinned: lib-consumers 1, lib-run 1" -- gate "$root"
printf 'libb:\n\t@python3 scripts/lib/y.py\n' >>"$root/Makefile"
reindex "$root"
st_case pin_set_lib_run_new_site 1 --what "a second site outside the pin fails" \
    --has "lib-run: lib/y.py:1:" --has "run by make:libb" --lacks "run by make:liba" -- gate "$root"

root=$(fresh pinset_dead_path)
printf '%s\n' '# see scripts/check/gone.sh' >>"$root/scripts/check/ok.sh"
pins_file 'dead-path check/ok.sh check/gone.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_set_dead_path_holds 0 --what "a pinned dead token passes" \
    --has "pinned: dead-path 1" -- gate "$root"
printf '%s\n' '# and scripts/check/other.sh' >>"$root/scripts/check/ok.sh"
reindex "$root"
st_case pin_set_dead_path_new_token 1 --what "a second dead token in the pinned file fails" \
    --has "scripts/check/other.sh names no file" --lacks "gone.sh names" -- gate "$root"

root=$(fresh pinset_dead_path_stale)
pins_file 'dead-path check/ok.sh check/gone.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_set_dead_path_stale_token 1 --what "a pinned token no longer present is stale" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: dead-path check/ok.sh: no longer violated" -- gate "$root"

root=$(fresh pinset_file_level)
valid_sh 'exit 0' | put "$root" check/y.sh
chmod -x "$root/scripts/check/y.sh"
pins_file 'exec-bit check/y.sh' >"$root/$PINS_REL"
reindex "$root"
st_case pin_exec_bit_holds 0 --what "a pinned missing exec bit passes" \
    --has "pinned: exec-bit 1" -- gate "$root"
chmod +x "$root/scripts/check/y.sh"
st_case pin_exec_bit_stale 1 --what "and is stale once the bit is set" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: exec-bit check/y.sh: no longer violated" -- gate "$root"

root=$(fresh pinset_base_growth)
nocaller_sh "$GATE_BODY" | put "$root" check/gate.sh
make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n'
pins_file 'recall-missing check/gate.sh default' >"$root/$PINS_REL"
reindex "$root"
commit_all "$root" "pin gate.sh"
make_ci "$root" '\t@bash scripts/check/gate.sh --lib\n'
pins_file 'recall-missing check/gate.sh default,--lib' >"$root/$PINS_REL"
reindex "$root"
st_case pin_new_set_member 1 --what "a mode added to a pin the base already had" \
    --has "pin-new: report/scripts_gate_pins.txt:4: recall-missing check/gate.sh: mode --lib is not pinned at the base" -- gate "$root"

# ---- scope ---------------------------------------------------------------

root=$(fresh scope_fixtures)
mkdir -p "$root/scripts/fixtures/gate"
printf 'no header at all\n' >"$root/scripts/fixtures/gate/nohdr.sh"
st_case scope_fixtures_excluded 0 --what "a headerless file under fixtures/ is not scanned" \
    --has "check-scripts: ok (4 files" -- gate "$root"

root=$(fresh scope_ignored)
printf 'scripts/check/scratch.sh\n' >"$root/.gitignore"
printf 'no header at all\n' >"$root/scripts/check/scratch.sh"
st_case scope_git_ignored_excluded 0 --what "a git-ignored file is not in scope" \
    --has "check-scripts: ok (4 files" -- gate "$root"

root=$(fresh scope_untracked)
printf 'no header at all\n' >"$root/scripts/check/scratch.sh"
reindex "$root"
st_case scope_untracked_included 1 --what "the same file, not ignored, is scanned" \
    --has "purpose: check/scratch.sh:" -- gate "$root"

# ---- unavailable ---------------------------------------------------------

root="$ST_WORK/ceiling/nogit"
mkdir -p "$root/scripts/check"
valid_sh 'exit 0' >"$root/scripts/check/x.sh"
nogit_gate() {
    GIT_CEILING_DIRECTORIES="$ST_WORK" gate "$root"
}
st_case unavailable_outside_git 2 --what "outside a git work tree the population is unknown" \
    --has "check-scripts: unavailable (" --has "is not inside a git work tree" -- nogit_gate

root=$(fresh no_scripts_dir)
rm -rf "$root/scripts"
st_case unavailable_no_files 2 --what "a tree with no file under scripts/ is not a pass" \
    --has "check-scripts: unavailable (no file under scripts/" -- gate "$root"

st_case unavailable_missing_root 2 --what "a root that does not exist" \
    --has "check-scripts: unavailable (" -- gate "$ST_WORK/does-not-exist"

root=$(fresh base_unknown)
st_case unavailable_unknown_base 2 --what "a base that is no commit" \
    --has "check-scripts: unavailable (--base no-such-ref is not a commit)" -- \
    python3 "$TOOL" --scripts --root "$root" --base no-such-ref

st_case unavailable_no_origin_main 2 --what "no --base and no origin/main to take one from" \
    --has "check-scripts: unavailable (no origin/main ref" -- \
    python3 "$TOOL" --scripts --root "$root"

st_case flags_need_scripts 2 --what "--write-index without --scripts is a usage error" \
    --has "need --scripts" -- python3 "$TOOL" --root "$root" --write-index

# ---- the default base ----------------------------------------------------

# auto <root>: the gate with no --base, which takes the merge-base with origin/main.
auto() {
    python3 "$TOOL" --scripts --root "$1"
}

# plant_unnamed <root> <name>: commit a new entry point with no Not a mode of: line.
plant_unnamed() {
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
        '# Caller: by hand' '#' 'exit 0' | put "$1" "check/$2.sh"
    reindex "$1"
    commit_all "$1" "adds $2"
}

root=$(fresh auto_ahead)
git -C "$root" update-ref refs/remotes/origin/main HEAD
plant_unnamed "$root" y
st_case base_is_merge_base_with_origin_main 1 --what "a branch ahead of origin/main: files added since the merge-base are new" \
    --has "check-scripts: base origin/main (" --has "not-a-mode: check/y.sh:1:" -- auto "$root"

root=$(fresh auto_tip)
plant_unnamed "$root" y
git -C "$root" update-ref refs/remotes/origin/main HEAD
st_case base_at_origin_main_is_the_parent 1 --what "HEAD is origin/main: the base is its parent, so the last commit's files are new" \
    --has "check-scripts: base HEAD^ (" --has "not-a-mode: check/y.sh:1:" -- auto "$root"

root=$(fresh auto_tip_old)
plant_unnamed "$root" y
plant_unnamed "$root" z
python3 - "$root/scripts/check/z.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Caller: by hand\n#\n", "# Caller: by hand\n# Not a mode of: check/ok.sh — a planted entry point\n")
open(path, "w").write(text)
EOF
reindex "$root"
commit_all "$root" "names z"
git -C "$root" update-ref refs/remotes/origin/main HEAD
st_case base_parent_leaves_older_files_alone 0 --what "a file committed before the parent is not new" \
    --has "check-scripts: base HEAD^ (" --lacks "not-a-mode: check/y.sh" -- auto "$root"

# ---- the real tree -------------------------------------------------------

# The pin counts are literals, so a chunk that shrinks a pin lowers them in the same commit
# and growth shows in the diff as a raised literal.
st_case real_tree 0 --what "the checked-in tree passes with exactly these pins" \
    --has "pinned: class 97, purpose 6, usage 87, exit-undeclared 86, caller 16, recall-missing 14, lib-consumers 10, lib-run 19, shebang 1, exec-bit 5, dead-path 1" \
    --has "342 pinned)" -- gate "$REPO_ROOT"

st_done
