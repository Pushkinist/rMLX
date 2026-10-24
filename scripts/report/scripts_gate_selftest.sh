#!/usr/bin/env bash
# Recall test for debt_report.py --scripts: planted trees, asserting exit code and rule:file:line.
#
# Recall test for: report/debt_report.py --scripts
# Usage:  bash scripts/report/scripts_gate_selftest.sh [<debt_report.py under test>]
# Exit:   0 every case held; 1 a case did not; 2 a fixture could not be built
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

# put <root> <rel>: write stdin to <root>/scripts/<rel>, executable.
put() {
    mkdir -p "$(dirname "$1/scripts/$2")"
    cat >"$1/scripts/$2"
    chmod +x "$1/scripts/$2"
}

# entry_sh <line 2> <line 3> <usage> <exit> <caller> <not-a-mode> <body>: a shell
# entry point whose header lines are the arguments. A bare `#` leaves a field out.
entry_sh() {
    printf '%s\n' '#!/usr/bin/env bash' "$1" "$2" "$3" "$4" "$5" "$6" 'set -euo pipefail' "$7"
}

# valid_sh <body>: a shell entry point every rule accepts, with <body> as line 9.
valid_sh() {
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
        '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" "$1"
}

# nocaller_sh <body>: a shell entry point with no Caller: line, so only a run makes it callable.
nocaller_sh() {
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
        '# Exit:   0 clean; 1 failed' '#' "$NAM" "$1"
}

# gate_at <root> <ref>: the gate with <ref> as its base.
gate_at() {
    python3 "$TOOL" --scripts --root "$1" --base "$2"
}

# reindex <root>: regenerate INDEX.md, so a case isolates the rule it plants.
reindex() {
    python3 "$TOOL" --scripts --root "$1" --base HEAD --write-index >/dev/null 2>&1 || true
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
#!/usr/bin/env bash
# A helper the fixture's entry point sources.
#
# Sourced, not executed.
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
entry_sh '#' '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
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
entry_sh '# A purpose line that carries on' '# onto the next line.' '# Usage:  bash scripts/check/x.sh' \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' | put "$root" check/x.sh
reindex "$root"
st_case purpose_wrapped 1 --what "the purpose wraps onto line 3, which is named" \
    --has "purpose: check/x.sh:3:" -- gate "$root"

root=$(fresh purpose_label_next)
entry_sh '# CI gate: a purpose followed at once by a field.' '# Usage:  bash scripts/check/x.sh' \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' '#' "$NAM" 'exit 0' | put "$root" check/x.sh
reindex "$root"
st_case purpose_then_field_not_wrapped 0 --what "a field on line 3 is not a wrapped purpose" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh purpose_marker_next)
entry_sh '# CI gate: a purpose followed at once by a marker.' '# doc-refs: fixture — planted' \
    '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case purpose_then_marker_not_wrapped 0 --what "a doc-refs marker on line 3 is not a wrapped purpose" \
    --has "check-scripts: ok (5 files" -- gate "$root"

long100="# $(printf '%*s' 100 '' | tr ' ' a)"
root=$(fresh purpose_100)
entry_sh "$long100" '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
    put "$root" check/x.sh
reindex "$root"
st_case purpose_100_characters_passes 0 --what "a purpose of exactly 100 characters is within the limit" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh purpose_101)
entry_sh "${long100}a" '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$NAM" 'exit 0' |
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
printf '%s\n' '#!/usr/bin/env bash' '# A second helper with no usage line.' '#' '# Sourced, not executed.' |
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
st_case usage_not_asked_without_shebang 0 --what "no shebang and nobody runs it: not an entry point" \
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
st_case usage_not_asked_of_a_module 0 --what "a Module of: line takes a file out of the entry points" \
    --has "check-scripts: ok (5 files" -- gate "$root"

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
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
    '# Exit:   passthrough of rmlx' '# Caller: by hand' "$NAM" 'exit "$rc"' | put "$root" check/x.sh
reindex "$root"
st_case exit_passthrough_not_literal 0 --what "exit \"\$rc\" under a passthrough declaration" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_declared)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
    '# Exit:   0 clean; 7 a violation' '# Caller: by hand' "$NAM" 'exit 7' | put "$root" check/x.sh
reindex "$root"
st_case exit_declared_passes 0 --what "exit 7 named in the Exit: line" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_range)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
    '# Exit:   0 clean; 3-10 specific to this script' '# Caller: by hand' "$NAM" 'exit 7' | put "$root" check/x.sh
reindex "$root"
st_case exit_range_declared 0 --what "a code inside a declared range" \
    --has "check-scripts: ok (5 files" -- gate "$root"

root=$(fresh exit_hidden)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
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
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
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
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
    '# Exit:   0 clean; 1 failed; reads the last 30 lines of the log' '# Caller: by hand' "$NAM" 'exit 30' |
    put "$root" check/x.sh
reindex "$root"
st_case exit_digit_in_prose_is_not_declared 1 --what "30 in the prose of Exit: declares nothing" \
    --has "exit-undeclared: check/x.sh:9: 30 is not declared" -- gate "$root"

root=$(fresh exit_equals_form)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' \
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
    'T2="$SUBJECT"; bash "${T2}"' 'timeout 5 "$SUBJECT"' 'bounded python3 "$SUBJECT"'; do
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
    printf '%s\n' '#!/usr/bin/env bash' '# A second helper that nothing runs.' '#' '# Sourced, not executed.' |
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

root=$(fresh index_xml_purpose)
printf '%s\n' '<?xml version="1.0"?>' '<!--' '  A planted property list read by a script.' '-->' >"$root/scripts/report/x.entitlements"
reindex "$root"
st_case index_xml_comment_is_the_purpose 0 --what "the first text line of an XML comment is the purpose" \
    --has '| `report/x.entitlements` | none | A planted property list read by a script. |' -- cat "$root/scripts/INDEX.md"

# ---- not-a-mode, not-a-mode-stale ----------------------------------------

root=$(fresh nam_missing)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean' \
    '# Caller: by hand' '#' 'exit 0' | put "$root" check/y.sh
reindex "$root"
st_case not_a_mode_missing 1 --what "a new entry point with no Not a mode of: line" \
    --has "not-a-mode: check/y.sh:1:" -- gate "$root"

root=$(fresh nam_no_reason)
entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean' \
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
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean' \
        '# Caller: by hand' "# Not a mode of: $target — a reason" 'exit 0' | put "$root" check/x.sh
    reindex "$root"
    if [ "$target" = check/ok.sh ]; then
        st_case "not_a_mode_target_$n" 0 --what "naming another entry point ($target) passes" \
            --has "check-scripts: ok (5 files" -- gate "$root"
        entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean' \
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
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean' \
        '# Caller: by hand' "# Not a mode of: check/ok.sh${separator}a reason" 'exit 0' | put "$root" check/x.sh
    reindex "$root"
    st_case "not_a_mode_separator_$n" 0 --what "the tool and its reason, separated by '${separator}'" \
        --has "check-scripts: ok (5 files" -- gate "$root"
done

# ---- pins ----------------------------------------------------------------

PINNABLE='# pinnable: class purpose usage exit-undeclared caller'
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
    --has "pin-stale: report/scripts_gate_pins.txt:5: exit-undeclared check/p.sh: code 7 is declared or gone" \
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
python3 - "$root/$PINS_REL" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("exit-undeclared\n", "exit-undeclared caller\n", 1)
open(path, "w").write(text)
EOF
printf 'caller check/r.sh\n' >>"$root/$PINS_REL"
reindex "$root"
st_case pin_seeds_a_rule_the_base_lacked 0 --what "a rule absent from the base's pinnable line may seed pins" \
    --has "pinned: usage 1, exit-undeclared 1, caller 1" -- gate "$root"

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
    entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/x.sh' '# Exit:   0 clean' \
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
    --has "pinned: class 97, purpose 6, usage 88, exit-undeclared 86, caller 16" \
    --has "293 pinned)" -- gate "$REPO_ROOT"

st_done
