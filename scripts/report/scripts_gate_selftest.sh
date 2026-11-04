#!/usr/bin/env bash
# Recall test for debt_report.py --scripts: planted trees, asserting exit code and rule:file:line.
# doc-refs: fixture — the docs/ paths below belong to the planted trees.
#
# Recall test for: report/debt_report.py --scripts
# Usage:  bash scripts/report/scripts_gate_selftest.sh [<debt_report.py under test>]
# Exit:   0 every case held; 1 a case did not; 2 a fixture could not be built
# LOC-exempt: one planted tree per case; the builders are in lib/selftest_support.sh
#
# The core and structural rules; scripts_gate_content_selftest.sh holds the per-file ones.
# Each case copies one small valid tree (a committed base), plants one edit and
# runs the gate with `--base HEAD`, so a planted file is new. The last case runs
# the gate on the real tree and holds the pinned counts it prints to the pin file.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="${1:-$REPO_ROOT/scripts/report/debt_report.py}"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init scripts-gate-selftest

sg_init

root=$(sg_fresh clean)
st_case clean_tree 0 --what "the base tree passes with no pin" \
    --has "check-scripts: ok (4 files, 0 pinned)" --has "pinned: none" -- sg_gate "$root"
st_case base_is_printed 0 --what "the run names the base it used" \
    --has "check-scripts: base HEAD (" -- sg_gate "$root"

# ---- index-stale ---------------------------------------------------------

root=$(sg_fresh index_new_file)
sg_valid_sh 'exit 0' | sg_put "$root" check/x.sh
line=$(sg_row_line "$root" check/ok.sh)
st_case index_missing_row 1 --what "a new file with no row: named at the line after its neighbour's row" \
    --has "index-stale: INDEX.md:$((line + 1)): missing row for check/x.sh" --lacks "not-a-mode" -- sg_gate "$root"

root=$(sg_fresh index_edited_row)
line=$(sg_row_line "$root" check/ok.sh)
python3 - "$root/scripts/INDEX.md" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("fixture entry point that every rule accepts", "an edited purpose")
open(path, "w").write(text)
EOF
st_case index_edited_row 1 --what "a row edited by hand is named at its own line" \
    --has "index-stale: INDEX.md:${line}: row for check/ok.sh differs" -- sg_gate "$root"

root=$(sg_fresh index_deleted_file)
line=$(sg_row_line "$root" lib/helper.sh)
rm "$root/scripts/lib/helper.sh"
st_case index_deleted_file 1 --what "a deleted file's row is named at its own line" \
    --has "index-stale: INDEX.md:${line}: row for lib/helper.sh names no file" -- sg_gate "$root"

root=$(sg_fresh index_preamble)
python3 - "$root/scripts/INDEX.md" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("Never edit it by hand.", "Edit it freely.")
open(path, "w").write(text)
EOF
st_case index_preamble_edited 1 --what "a change outside the rows is still stale" \
    --has "index-stale: INDEX.md:" --has "differs from the generated text" -- sg_gate "$root"

root=$(sg_fresh index_longer)
printf '\n' >>"$root/scripts/INDEX.md"
line=$(($(grep -c '' "$root/scripts/INDEX.md") + 1))
st_case index_only_longer 1 --what "an INDEX.md that differs only in length is named at the first line past the text, with no traceback" \
    --has "index-stale: INDEX.md:${line}: differs from the generated text" --lacks "Traceback" -- sg_gate "$root"

root=$(sg_fresh index_absent)
rm "$root/scripts/INDEX.md"
st_case index_absent 1 --what "no INDEX.md is stale, not clean" \
    --has "index-stale: INDEX.md:1: missing" -- sg_gate "$root"

root=$(sg_fresh index_written)
sg_valid_sh 'exit 0' | sg_put "$root" check/x.sh
st_case write_index_then_pass 0 --what "--write-index regenerates the rows and the check then passes" \
    --has "scripts-index: wrote scripts/INDEX.md (5 rows)" -- python3 "$TOOL" --scripts --root "$root" --base HEAD --write-index
st_case index_written_passes 0 --what "the regenerated index is what the check expects" \
    --has "check-scripts: ok (5 files, 0 pinned)" -- sg_gate "$root"

root=$(sg_fresh index_write_refused)
sg_valid_sh 'exit 0' | sg_put "$root" check/x.sh
before=$(cksum <"$root/scripts/INDEX.md")
st_case write_index_unknown_base 2 --what "--write-index with a base that is no commit refuses" \
    --has "is not a commit" -- python3 "$TOOL" --scripts --root "$root" --base no-such-ref --write-index
st_case write_index_unknown_base_writes_nothing 0 --what "and the INDEX.md it would have rewritten is as it was" \
    --has "$before" -- sh -c 'cksum <"$1/scripts/INDEX.md"' sh "$root"

root=$(sg_fresh index_written_still_checks)
sg_put "$root" stray.sh <<'EOF'
#!/usr/bin/env bash
# A top-level file the class rule refuses.
set -euo pipefail
EOF
st_case write_index_runs_the_rules 1 --what "--write-index writes the index and still reports every other rule" \
    --has "class: stray.sh:1:" --lacks "index-stale" -- python3 "$TOOL" --scripts --root "$root" --base HEAD --write-index

# ---- class ---------------------------------------------------------------

root=$(sg_fresh class_top)
sg_valid_sh 'exit 0' | sg_put "$root" x.sh
sg_reindex "$root"
st_case class_top_level_file 1 --what "a file at the top of scripts/" \
    --has "class: x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh class_newdir)
sg_valid_sh 'exit 0' | sg_put "$root" newdir/x.sh
sg_reindex "$root"
st_case class_unknown_directory 1 --what "a file in a directory outside the class list" \
    --has "class: newdir/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh class_readme)
printf '# scripts/ — rules\n' >"$root/scripts/README.md"
sg_reindex "$root"
st_case class_readme_allowed 0 --what "README.md at the top is not a file of the tree" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

# ---- purpose -------------------------------------------------------------

root=$(sg_fresh purpose_missing)
sg_entry_sh '#' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case purpose_missing 1 --what "line 2 is a bare comment: no purpose" \
    --has "purpose: check/x.sh:2:" -- sg_gate "$root"

root=$(sg_fresh purpose_code_first)
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case purpose_code_on_line_2 1 --what "code right after the shebang: no purpose" \
    --has "purpose: check/x.sh:2:" -- sg_gate "$root"

root=$(sg_fresh purpose_wrapped)
sg_entry_sh '# A purpose line that carries on' '# onto the next line.' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case purpose_wrapped 1 --what "the purpose wraps onto line 3, which is named" \
    --has "purpose: check/x.sh:3:" -- sg_gate "$root"

root=$(sg_fresh purpose_label_next)
sg_entry_sh '# CI gate: a purpose followed at once by a field.' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' '#' "$SG_NAM" 'exit 0' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case purpose_then_field_not_wrapped 0 --what "a field on line 3 is not a wrapped purpose" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh purpose_marker_next)
sg_entry_sh '# CI gate: a purpose followed at once by a marker.' '# doc-refs: fixture — planted' \
    '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case purpose_then_marker_not_wrapped 0 --what "a doc-refs marker on line 3 is not a wrapped purpose" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

long100="# $(printf '%*s' 100 '' | tr ' ' a)"
root=$(sg_fresh purpose_100)
sg_entry_sh "$long100" '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case purpose_100_characters_passes 0 --what "a purpose of exactly 100 characters is within the limit" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh purpose_101)
sg_entry_sh "${long100}a" '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case purpose_101_characters 1 --what "a purpose of 101 characters is over the limit" \
    --has "purpose: check/x.sh:2: purpose is 101 characters, over 100" -- sg_gate "$root"

root=$(sg_fresh purpose_py_missing)
sg_put "$root" report/y.py <<'EOF'
#!/usr/bin/env python3
"""
Usage:  python3 scripts/report/y.py
Exit:   0 always
Not a mode of: report/tool.py — a planted entry point
Caller: by hand
"""
raise SystemExit(0)
EOF
sg_reindex "$root"
st_case purpose_python_docstring_missing 1 --what "a docstring that opens on a bare line has no purpose" \
    --has "purpose: report/y.py:2:" -- sg_gate "$root"

root=$(sg_fresh purpose_py_ok)
sg_put "$root" report/y.py <<'EOF'
#!/usr/bin/env python3
"""A planted Python entry point with its purpose on the docstring's first line.

Usage:  python3 scripts/report/y.py
Exit:   0 always
Not a mode of: report/tool.py — a planted entry point
Caller: by hand
"""
raise SystemExit(0)
EOF
sg_reindex "$root"
st_case purpose_python_docstring_ok 0 --what "the same file with a purpose passes" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh purpose_data_file)
printf '%s\n' 'no comment at all' >"$root/scripts/report/data.txt"
sg_reindex "$root"
st_case purpose_data_file_without_comment 1 --what "a data file is held to the rule too" \
    --has "purpose: report/data.txt:1:" -- sg_gate "$root"

# ---- usage ---------------------------------------------------------------

root=$(sg_fresh usage_missing)
sg_entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case usage_missing 1 --what "an entry point without a Usage: line" \
    --has "usage: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh usage_lib_exempt)
printf '%s\n' '# A second helper with no usage line.' '#' '# Module of: check/ok.sh' |
    sg_put "$root" lib/second.sh
printf '. "$(dirname "$0")/../lib/second.sh"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case usage_not_asked_of_lib 0 --what "a lib/ file is not an entry point" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh usage_py_no_shebang)
sg_put "$root" report/z.py <<'EOF'
"""A Python file with no shebang, which no one runs directly.

Nothing here is an entry point, so no Usage: line is asked of it.
"""
EOF
sg_reindex "$root"
st_case usage_not_asked_without_shebang 1 --what "no shebang and nobody runs it: not an entry point, so an orphan and no Usage: asked" \
    --has "orphan: report/z.py:1:" --lacks "usage: report/z.py" -- sg_gate "$root"

root=$(sg_fresh usage_py_imported)
sg_put "$root" report/z.py <<'EOF'
"""A Python file with no shebang that the tool imports without a Module of: line.

Imported, never run: not an entry point and not an orphan.
"""
EOF
printf 'import z\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case usage_not_asked_of_an_imported_file 0 --what "an import is a caller: neither an entry point nor an orphan" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh usage_run_without_shebang)
printf '%s\n' '# A planted script with no shebang that a recipe runs.' 'exit 0' | sg_put "$root" check/noshebang.sh
printf 'check-noshebang:\n\t@bash scripts/check/noshebang.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case usage_asked_of_a_file_a_recipe_runs 1 --what "no shebang, but a recipe runs it: an entry point" \
    --has "usage: check/noshebang.sh:1:" -- sg_gate "$root"

root=$(sg_fresh usage_run_in_lib)
printf '%s\n' '#!/usr/bin/env bash' '# A planted file in lib/ that a recipe runs.' 'exit 0' | sg_put "$root" lib/runme.sh
printf 'check-runme:\n\t@bash scripts/lib/runme.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case usage_asked_of_a_lib_file_a_recipe_runs 1 --what "a lib/ file a recipe runs is an entry point" \
    --has "usage: lib/runme.sh:1:" -- sg_gate "$root"

root=$(sg_fresh usage_empty)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:' '# Exit:   0 clean' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case usage_empty 1 --what "a Usage: line that says nothing" \
    --has "usage: check/x.sh:4: \`Usage:\` says nothing" -- sg_gate "$root"

root=$(sg_fresh usage_module_with_shebang)
sg_put "$root" report/tool_three.py <<'EOF'
#!/usr/bin/env python3
"""A module that also carries a shebang.

Module of: report/tool.py
"""
EOF
printf 'import tool_three\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case usage_not_asked_of_a_module 1 --what "a Module of: line takes a file out of the entry points; its shebang is the finding" \
    --has "shebang: report/tool_three.py:1:" --lacks "usage:" -- sg_gate "$root"

# ---- exit-undeclared -----------------------------------------------------

root=$(sg_fresh exit_seven)
sg_valid_sh 'exit 7' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_undeclared_shell 1 --what "exit 7 on line 9 is not in the Exit: line" \
    --has "exit-undeclared: check/x.sh:9: 7 is not declared" -- sg_gate "$root"

root=$(sg_fresh exit_twice)
sg_valid_sh "$(printf 'exit 7\nexit 7\n')" | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_undeclared_reported_once 1 --what "a code used twice is named once, at its first line" \
    --has "exit-undeclared: check/x.sh:9: 7" --lacks "check/x.sh:10:" -- sg_gate "$root"

root=$(sg_fresh exit_passthrough)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   passthrough of rmlx' '# Caller: by hand' "$SG_NAM" 'exit "$rc"' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_passthrough_not_literal 0 --what "exit \"\$rc\" under a passthrough declaration" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh exit_declared)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 7 a violation' '# Caller: by hand' "$SG_NAM" 'exit 7' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_declared_passes 0 --what "exit 7 named in the Exit: line" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh exit_range)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 3-10 specific to this script' '# Caller: by hand' "$SG_NAM" 'exit 7' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_range_declared 0 --what "a code inside a declared range" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh exit_hidden)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean' '# Caller: by hand' "$SG_NAM" "$(printf '%s\n' \
        '# exit 7 in a comment' \
        '# done; exit 7 in a comment' \
        'echo "exit 7 in a string"' \
        'echo "done; exit 7"' \
        "echo 'then exit 7 in single quotes'" \
        'cat <<TXT' \
        'exit 7' \
        'TXT' \
        'python3 -c '"'"'import sys; sys.exit(9)'"'"'' \
        'exit 0')" | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_in_comment_string_heredoc 0 --what "exit in a comment, a string, a heredoc body or python -c is not the script's exit" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

n=0
for body in 'true || exit 6' 'true && exit 6' '{ exit 6; }' 'case x in x) exit 6;; esac' '( exit 6 )' 'true; exit 6'; do
    n=$((n + 1))
    root=$(sg_fresh "exit_position_$n")
    sg_valid_sh "$body" | sg_put "$root" check/x.sh
    sg_reindex "$root"
    st_case "exit_position_$n" 1 --what "exit 6 after: $body" \
        --has "exit-undeclared: check/x.sh:9: 6" -- sg_gate "$root"
done

root=$(sg_fresh exit_en_dash)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 3–10 specific to this script' '# Caller: by hand' "$SG_NAM" 'exit 7' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_en_dash_range 0 --what "a range written with an en dash" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh exit_dash_heredoc)
sg_valid_sh "$(printf 'cat <<-TXT\n\texit 7\n\tTXT\nexit 6')" | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_after_tab_indented_heredoc 1 --what "<<- ends at a tab-indented tag, so the exit after it is read" \
    --has "exit-undeclared: check/x.sh:12: 6" --lacks ": 7 is not declared" -- sg_gate "$root"

root=$(sg_fresh exit_prose_digit)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed; reads the last 30 lines of the log' '# Caller: by hand' "$SG_NAM" 'exit 30' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_digit_in_prose_is_not_declared 1 --what "30 in the prose of Exit: declares nothing" \
    --has "exit-undeclared: check/x.sh:9: 30 is not declared" -- sg_gate "$root"

root=$(sg_fresh exit_equals_form)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    "$(printf '# Exit:   0 = clean\n#         4 = refused')" '# Caller: by hand' "$SG_NAM" 'exit 4' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_n_equals_meaning_declares 0 --what "N = meaning, on a continuation line, declares N" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh exit_substitution_quotes)
dq='"'
sq="'"
sg_valid_sh "v=${dq}\$(echo ${sq}${dq}${sq})${dq}"$'\n'"exit 6" | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_after_a_substitution_with_quotes 1 --what "a quote inside \$( ) in a string does not hide the exit after it" \
    --has "exit-undeclared: check/x.sh:10: 6" -- sg_gate "$root"

root=$(sg_fresh exit_after_then)
sg_valid_sh 'if [ -n "${X:-}" ]; then exit 5; fi' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case exit_after_then 1 --what "exit after then is read" \
    --has "exit-undeclared: check/x.sh:9: 5" -- sg_gate "$root"

root=$(sg_fresh exit_python)
sg_put "$root" report/y.py <<'EOF'
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
sg_reindex "$root"
st_case exit_undeclared_python 1 --what "sys.exit(4) on line 11" \
    --has "exit-undeclared: report/y.py:11: 4 is not declared" -- sg_gate "$root"

root=$(sg_fresh exit_python_string)
sg_put "$root" report/y.py <<'EOF'
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
sg_reindex "$root"
st_case exit_python_in_string 0 --what "sys.exit(4) inside a string is not a call" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

n=0
for stmt in 'sys.exit(5)' 'exit(5)' 'quit(5)' 'os._exit(5)' 'raise SystemExit(5)'; do
    n=$((n + 1))
    root=$(sg_fresh "exit_python_spelling_$n")
    printf '%s\n' '#!/usr/bin/env python3' '"""A planted Python entry point that exits with an undeclared code.' '' \
        'Usage:  python3 scripts/report/y.py' 'Exit:   0 always' \
        'Not a mode of: report/tool.py — a planted entry point' 'Caller: by hand' '"""' \
        'import os, sys' "$stmt" | sg_put "$root" report/y.py
    sg_reindex "$root"
    st_case "exit_python_spelling_$n" 1 --what "$stmt on line 10" \
        --has "exit-undeclared: report/y.py:10: 5 is not declared" -- sg_gate "$root"
done

# ---- caller --------------------------------------------------------------

root=$(sg_fresh caller_none)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case caller_none 1 --what "nothing runs it and it does not say it is by hand" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_by_hand_ok)
sg_valid_sh 'exit 0' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case caller_by_hand_declared 0 --what "Caller: by hand on a file nothing runs" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_stale_by_hand)
python3 - "$root/scripts/check/ok.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Exit:   0 clean; 1 a violation\n", "# Exit:   0 clean; 1 a violation\n# Caller: by hand\n")
open(path, "w").write(text)
EOF
sg_reindex "$root"
st_case caller_by_hand_but_run 1 --what "Caller: by hand on a file the Makefile runs, named at its own line" \
    --has "caller: check/ok.sh:6: declares \`Caller: by hand\` but something runs it" -- sg_gate "$root"

root=$(sg_fresh caller_longer_name)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/ab.sh
printf 'check-lab:\n\t@bash scripts/check/lab.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case caller_name_inside_a_longer_name 1 --what "lab.sh in a recipe is not a caller of ab.sh" \
    --has "caller: check/ab.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_suffix)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
printf 'check-bak:\n\t@bash scripts/check/x.sh.bak\n' >>"$root/Makefile"
sg_reindex "$root"
st_case caller_name_with_a_suffix 1 --what "x.sh.bak in a recipe is not a caller of x.sh" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_prose)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
printf 'say:\n\t@echo "run scripts/check/x.sh by hand"\n\t@printf "%%s\\n" scripts/check/x.sh\n\t# bash scripts/check/x.sh\n' >>"$root/Makefile"
printf '# bash scripts/check/x.sh\necho scripts/check/x.sh\nusage() { echo "see check/x.sh"; }\n' >>"$root/scripts/lib/helper.sh"
sg_reindex "$root"
st_case caller_named_only_in_text 1 --what "a comment, an echo and a printf name it and run nothing" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_script_runs_it)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
printf 'bash "$(dirname "$0")/x.sh"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_another_script_runs_it 0 --what "another script running it is a caller" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_in_subshell)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
printf 'out=$(bash scripts/check/x.sh 2>&1)\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_inside_command_substitution 0 --what "a run inside \$( ) is a caller" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_path_in_another_directory)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'REPO_ROOT=.\nbash "$REPO_ROOT/scripts/no_such_dir/x.sh"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_path_through_a_directory_that_has_no_such_file 1 --what "a path under scripts/ that names no file credits no file of the same base name" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_path_under_repo_root)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'REPO_ROOT=.\nbash "$REPO_ROOT/scripts/check/x.sh"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_path_under_a_variable_and_scripts 0 --what "\$REPO_ROOT/scripts/check/x.sh is a caller of check/x.sh" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_path_dot_scripts)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'bash ./scripts/check/x.sh\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_path_dot_slash_scripts 0 --what "./scripts/check/x.sh is a caller of check/x.sh" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_path_sibling)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'HERE="$(cd "$(dirname "$0")" && pwd)"\nbash "$HERE/x.sh"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_path_sibling_by_a_variable 0 --what "\$HERE/x.sh, no scripts/ in the word, still credits the one file of that name" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_path_wrong_tail)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'REPO_ROOT=.\nbash "${REPO_ROOT}/scripts/lib/x.sh"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_path_with_the_wrong_directory 1 --what "scripts/lib/x.sh is not scripts/check/x.sh" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

n=0
for spelling in '/opt/other/scripts/check/x.sh' '../scripts/check/x.sh' '$HOME/other/scripts/check/x.sh' \
    'x/$REPO_ROOT/scripts/check/x.sh'; do
    n=$((n + 1))
    root=$(sg_fresh "caller_other_tree_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    printf '%s\n' 'REPO_ROOT=.' "bash $spelling" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    st_case "caller_path_in_another_tree_$n" 1 --what "bash $spelling names a file of another tree: no caller of check/x.sh" \
        --has "caller: check/x.sh:1:" -- sg_gate "$root"
done

n=0
for spelling in '"${REPO_ROOT:-.}/scripts/check/x.sh"' '"$(git rev-parse --show-toplevel)/scripts/check/x.sh"' \
    '"$(pwd)"/scripts/check/x.sh' '${REPO_ROOT:=.}/scripts/check/x.sh'; do
    n=$((n + 1))
    root=$(sg_fresh "caller_root_expr_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    printf '%s\n' 'REPO_ROOT=.' "bash $spelling" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    st_case "caller_path_behind_root_expression_$n" 0 --what "bash $spelling is a run of check/x.sh" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

root=$(sg_fresh caller_default_expansion)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'REPO_ROOT=.\nTOOL="${1:-$REPO_ROOT/scripts/check/x.sh}"\nbash "$TOOL"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_default_of_a_parameter_expansion 0 --what "VAR=\"\${1:-<path>}\" then bash \"\$VAR\": the default is the path that runs" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_default_expansion_unused)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'REPO_ROOT=.\nTOOL="${1:-$REPO_ROOT/scripts/check/x.sh}"\necho "$TOOL"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_default_expansion_only_printed 1 --what "the same assignment, only printed: no caller" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_single_quoted_substitution)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
cat >>"$root/scripts/check/ok.sh" <<'EOF'
printf '%s\n' 'out=$(bash scripts/check/x.sh 2>&1)'
EOF
sg_reindex "$root"
st_case caller_substitution_inside_single_quotes_is_text 1 --what "a \$( ) in single quotes is no run: the shell never expands it" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_double_quoted_substitution)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
cat >>"$root/scripts/check/ok.sh" <<'EOF'
out="it's $(bash scripts/check/x.sh 2>&1)"
EOF
sg_reindex "$root"
st_case caller_substitution_inside_double_quotes_runs 0 --what "a \$( ) in double quotes runs, also behind an apostrophe" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_escaped_apostrophe)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
cat >>"$root/scripts/check/ok.sh" <<'EOF'
echo \' "$(bash scripts/check/x.sh 2>&1)"
EOF
sg_reindex "$root"
st_case caller_substitution_after_an_escaped_apostrophe 0 --what "a backslashed apostrophe opens no single quote: the substitution after it runs" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_variable_after_scripts)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
cat >>"$root/scripts/check/ok.sh" <<'EOF'
REPO_ROOT=.
sub=check
bash "$REPO_ROOT/scripts/$sub/x.sh"
EOF
sg_reindex "$root"
st_case caller_path_read_from_its_last_expansion 0 --what "scripts/\$sub/x.sh: only the part after the last expansion is read, and x.sh is the one file of that name" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_self)
sg_valid_sh 'bash scripts/check/x.sh' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
sg_reindex "$root"
st_case caller_not_itself 1 --what "a script that runs itself has no caller" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_data_files)
printf '%s\n' '# A planted data file that a script only names in text.' >"$root/scripts/report/said.txt"
printf '%s\n' '# A planted data file that a script reads.' >"$root/scripts/report/read.txt"
printf 'echo scripts/report/said.txt\ncat scripts/report/read.txt\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_data_file_named_in_echo 0 --what "a data file named only by echo has no reader" \
    --has '| `report/said.txt` | none |' -- cat "$root/scripts/INDEX.md"
st_case caller_data_file_read 0 --what "a data file passed to cat is read by the script" \
    --has '| `report/read.txt` | check/ok.sh |' -- cat "$root/scripts/INDEX.md"

n=0
for body in 'out=`bash scripts/check/x.sh`' 'timeout 5 bash scripts/check/x.sh' 'env A=1 scripts/check/x.sh' \
    'uv run scripts/check/x.sh' 'run_case name -- scripts/check/x.sh arg' 'FOO=1 bash -x scripts/check/x.sh'; do
    n=$((n + 1))
    root=$(sg_fresh "caller_spelling_$n")
    sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
        sg_put "$root" check/x.sh
    printf '%s\n' "$body" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    st_case "caller_spelling_$n" 0 --what "run as: $body" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

root=$(sg_fresh caller_multi_target)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
printf 'multi-a multi-b:\n\t@bash scripts/check/x.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case caller_every_target_of_a_rule 0 --what "a rule with two target names credits both" \
    --has '| `check/x.sh` | make multi-a, make multi-b |' -- cat "$root/scripts/INDEX.md"

root=$(sg_fresh caller_assign_unused)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'SUBJECT=scripts/check/x.sh\ngrep -c TODO "$SUBJECT" >/dev/null || true\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_assignment_only_as_an_argument 1 --what "VAR=path with \$VAR used as an argument runs nothing" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

n=0
for use in 'bash "$SUBJECT"' 'st_run "$SUBJECT"' 'st_case name 0 -- "$SUBJECT"' 'env -u A "$SUBJECT"' \
    'T2="$SUBJECT"; bash "${T2}"' 'timeout 5 "$SUBJECT"' 'bounded python3 "$SUBJECT"' \
    'st_bounded 5 bash "$SUBJECT"'; do
    n=$((n + 1))
    root=$(sg_fresh "caller_assign_used_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    printf 'SUBJECT=scripts/check/x.sh\n%s\n' "$use" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    st_case "caller_assign_used_$n" 0 --what "VAR=path, later used as: $use" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

root=$(sg_fresh caller_assign_used_before)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'bash "$SUBJECT"\nSUBJECT=scripts/check/x.sh\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_variable_used_before_it_is_set 1 --what "\$VAR expanded before VAR=path credits nothing" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_python_constant)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf '\nSUBJECT = "scripts/check/x.sh"\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case caller_python_bare_constant 1 --what "a Python string constant outside a command vector is no caller" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_shell_continuation)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'echo \\\nscripts/check/x.sh\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_shell_continuation_joined 1 --what "a name on the continuation line of an echo in a script is text" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_make_continuation)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'check-echo:\n\t@echo \\\n\t\tscripts/check/x.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case caller_recipe_continuation_joined 1 --what "a name on the continuation line of an echo is text" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_make_continuation_run)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'check-cont:\n\t@bash \\\n\t\tscripts/check/x.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case caller_recipe_continuation_run 0 --what "bash on one line and the script on the next is a run" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_make_continuation_flush)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'check-flush:\n\t@bash \\\nscripts/check/x.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case caller_recipe_continuation_at_column_0 0 --what "a continuation line that starts at column 0 still belongs to the recipe" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

n=0
for wrapper in 'exec' 'env A=1' 'command' 'nohup' 'nice -n 5' 'timeout 5' 'sudo' 'builtin'; do
    n=$((n + 1))
    root=$(sg_fresh "caller_wrapper_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    printf '%s scripts/check/x.sh\n' "$wrapper" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    st_case "caller_wrapper_$n" 0 --what "run behind: $wrapper" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

n=0
for form in 'source scripts/lib/second.sh' '. scripts/lib/second.sh'; do
    n=$((n + 1))
    root=$(sg_fresh "caller_source_$n")
    printf '%s\n' '# A second helper that nothing runs.' '#' '# Module of: check/ok.sh' |
        sg_put "$root" lib/second.sh
    printf '%s\n' "$form" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    st_case "caller_source_$n" 0 --what "$form: a caller, and a sourced file is no entry point" \
        --has '| `lib/second.sh` | check/ok.sh |' -- cat "$root/scripts/INDEX.md"
    st_case "caller_source_clean_$n" 0 --what "and it is not asked for a Usage: line" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

root=$(sg_fresh caller_nested_substitution)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'bash "$(dirname "$(dirname "$0")")/check/x.sh"\n' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case caller_nested_substitution 0 --what "a path built from a nested \$( ) in double quotes" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_cell_by_hand)
sg_valid_sh 'exit 0' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case caller_cell_by_hand 0 --what "an entry nothing runs, declared by hand, reads by hand in the index" \
    --has '| `check/x.sh` | by hand |' -- cat "$root/scripts/INDEX.md"

root=$(sg_fresh caller_python_string)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/x.sh
printf 'import subprocess\n\nsubprocess.run(["bash", "scripts/check/x.sh"])\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case caller_python_path_string 0 --what "a Python file naming the path in a call is a caller" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh caller_python_from_import)
sg_put "$root" report/tool_four.py <<'EOF'
"""A planted module imported with from.

Module of: report/tool.py
"""
EOF
printf 'from tool_four import nothing\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case caller_python_from_import 0 --what "from <module> import names the importing tool as the module's caller" \
    --has '| `report/tool_four.py` | report/tool.py |' -- cat "$root/scripts/INDEX.md"

root=$(sg_fresh caller_same_basename)
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" check/dup.sh
sg_valid_sh 'exit 0' | python3 -c 'import sys; sys.stdout.write(sys.stdin.read().replace("# Caller: by hand\n", "#\n"))' |
    sg_put "$root" report/dup.sh
printf 'check-dup:\n\t@bash scripts/check/dup.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case caller_two_files_one_basename 1 --what "a recipe naming check/dup.sh is no caller of report/dup.sh" \
    --has "caller: report/dup.sh:1:" --lacks "caller: check/dup.sh" -- sg_gate "$root"

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
    root=$(sg_fresh "caller_py_bound_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    printf '\nimport subprocess\nfrom pathlib import Path\n\n%b' "$snippet" >>"$root/scripts/report/tool.py"
    sg_reindex "$root"
    st_case "caller_python_bound_name_$n" 0 --what "a name bound to the path and passed to a subprocess call is a caller" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

root=$(sg_fresh caller_py_from_import)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf '\nfrom subprocess import run\n\nSUBJECT = "scripts/check/x.sh"\nrun(["bash", SUBJECT])\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case caller_python_from_subprocess_import 0 --what "run imported from subprocess is a subprocess call" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

n=0
for snippet in \
    'SUBJECT = "scripts/check/x.sh"\nprint(SUBJECT)\n' \
    'SUBJECT = "scripts/check/x.sh"\nsize = len(SUBJECT)\n' \
    'SUBJECT = "scripts/check/x.sh"\nsubprocess.run(["true"])\n' \
    'SUBJECT = "scripts/check/x.sh"\nsubprocess.run(["bash", "-c", "true"], cwd=SUBJECT)\n' \
    'def run(path):\n    return path\n\nSUBJECT = "scripts/check/x.sh"\nrun(["bash", SUBJECT])\n'; do
    n=$((n + 1))
    root=$(sg_fresh "caller_py_unbound_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    printf '\nimport subprocess\n\n%b' "$snippet" >>"$root/scripts/report/tool.py"
    sg_reindex "$root"
    st_case "caller_python_name_not_run_$n" 1 --what "a bound name that no subprocess call takes as its program is no caller" \
        --has "caller: check/x.sh:1:" -- sg_gate "$root"
done

root=$(sg_fresh caller_workflow)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'name: ci\njobs:\n  gates:\n    steps:\n      - run: bash scripts/check/x.sh\n' >"$root/.github/workflows/ci.yml"
sg_reindex "$root"
st_case caller_workflow_inline_run 0 --what "a hosted-CI run step is a caller" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"
st_case caller_workflow_cell 0 --what "and the index names the workflow file" \
    --has '| `check/x.sh` | ci.yml |' -- cat "$root/scripts/INDEX.md"

root=$(sg_fresh caller_workflow_block)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'name: ci\njobs:\n  gates:\n    steps:\n      - name: gates\n        run: |\n          set -e\n          bash scripts/check/x.sh\n      - name: after\n        run: echo done\n' >"$root/.github/workflows/ci.yml"
sg_reindex "$root"
st_case caller_workflow_block_run 0 --what "a line of a run: block scalar is a caller" \
    --has '| `check/x.sh` | ci.yml |' -- cat "$root/scripts/INDEX.md"

root=$(sg_fresh caller_workflow_folded)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'jobs:\n  gates:\n    steps:\n      - run: >\n          bash scripts/check/x.sh\n' >"$root/.github/workflows/ci.yml"
sg_reindex "$root"
st_case caller_workflow_folded_run 0 --what "a run: > folded scalar is a caller too" \
    --has '| `check/x.sh` | ci.yml |' -- cat "$root/scripts/INDEX.md"

root=$(sg_fresh caller_workflow_text)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
mkdir -p "$root/.github/workflows"
printf 'name: scripts/check/x.sh\n# bash scripts/check/x.sh\njobs:\n  gates:\n    steps:\n      - name: scripts/check/x.sh\n        run: echo scripts/check/x.sh\n      - run: |\n          echo done\n        shell: bash scripts/check/x.sh\n        env:\n          SUBJECT: scripts/check/x.sh\n' >"$root/.github/workflows/ci.yml"
sg_reindex "$root"
st_case caller_workflow_text_only 1 --what "a workflow name, comment, env value, echo and a key after the run block name the file and run nothing" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_pre_commit)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'repos:\n  - repo: local\n    hooks:\n      - id: x\n        entry: bash scripts/check/x.sh\n        language: system\n' >"$root/.pre-commit-config.yaml"
sg_reindex "$root"
st_case caller_pre_commit_entry 0 --what "a pre-commit entry is a caller, shown as pre-commit" \
    --has '| `check/x.sh` | pre-commit |' -- cat "$root/scripts/INDEX.md"

root=$(sg_fresh caller_pre_commit_text)
sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
printf 'repos:\n  - repo: local\n    hooks:\n      - id: x\n        # entry: bash scripts/check/x.sh\n        name: scripts/check/x.sh\n        entry: true\n        exclude: ^scripts/check/x.sh$\n' >"$root/.pre-commit-config.yaml"
sg_reindex "$root"
st_case caller_pre_commit_text_only 1 --what "a commented entry, a name and an exclude pattern run nothing" \
    --has "caller: check/x.sh:1:" -- sg_gate "$root"

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
    root=$(sg_fresh "caller_crate_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    mkdir -p "$root/crates/c/src"
    printf '%s\n' "$src" >"$root/crates/c/src/lib.rs"
    sg_reindex "$root"
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
    root=$(sg_fresh "caller_crate_text_$n")
    sg_nocaller_sh 'exit 0' | sg_put "$root" check/x.sh
    mkdir -p "$root/crates/c/src"
    printf '%s\n' "$src" >"$root/crates/c/src/lib.rs"
    sg_reindex "$root"
    st_case "caller_crate_text_only_$n" 1 --what "not a caller: $src" \
        --has "caller: check/x.sh:1:" --lacks "dead-path" -- sg_gate "$root"
done

root=$(sg_fresh caller_crate_run)
printf '%s\n' '# A planted script with no shebang that a crate runs.' 'exit 0' | sg_put "$root" check/noshebang.sh
mkdir -p "$root/crates/c/src"
printf 'fn run() { Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/check/noshebang.sh")); }\n' >"$root/crates/c/src/lib.rs"
sg_reindex "$root"
st_case caller_crate_command_new_is_a_run 1 --what "a path that is the argument of Command::new makes the file an entry point" \
    --has "usage: check/noshebang.sh:1:" -- sg_gate "$root"

root=$(sg_fresh caller_crate_read)
printf '%s\n' '# A planted script with no shebang that a crate only reads.' 'exit 0' | sg_put "$root" check/noshebang.sh
mkdir -p "$root/crates/c/src"
printf 'const S: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/check/noshebang.sh"));\n' >"$root/crates/c/src/lib.rs"
sg_reindex "$root"
st_case caller_crate_include_str_is_a_read 0 --what "a path that is only read is a caller and no entry point" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh index_xml_purpose)
printf '%s\n' '<?xml version="1.0"?>' '<!--' '  A planted property list read by a script.' '-->' >"$root/scripts/report/x.entitlements"
sg_reindex "$root"
st_case index_xml_comment_is_the_purpose 0 --what "the first text line of an XML comment is the purpose" \
    --has '| `report/x.entitlements` | none | A planted property list read by a script. |' -- cat "$root/scripts/INDEX.md"

# ---- not-a-mode, not-a-mode-stale ----------------------------------------

root=$(sg_fresh nam_missing)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
    '# Caller: by hand' '#' 'exit 0' | sg_put "$root" check/y.sh
sg_reindex "$root"
st_case not_a_mode_missing 1 --what "a new entry point with no Not a mode of: line" \
    --has "not-a-mode: check/y.sh:1:" -- sg_gate "$root"

root=$(sg_fresh nam_no_reason)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
    '# Caller: by hand' '# Not a mode of: check/ok.sh' 'exit 0' | sg_put "$root" check/y.sh
sg_reindex "$root"
st_case not_a_mode_without_reason 1 --what "the line names a tool and gives no reason" \
    --has "not-a-mode: check/y.sh:7: \`Not a mode of:\` gives no reason" -- sg_gate "$root"

root=$(sg_fresh nam_old_file)
st_case not_a_mode_old_file_exempt 0 --what "ok.sh is in the base and carries no such line" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

root=$(sg_fresh nam_recall_test)
printf '%s\n' '#!/usr/bin/env bash' '# Recall test for ok.sh: a planted recall test.' '#' \
    '# Recall test for: check/ok.sh' '# Usage:  bash scripts/check/ok_selftest.sh' '# Exit:   0 held' \
    '# Caller: by hand' 'set -euo pipefail' 'bash scripts/check/ok.sh' |
    sg_put "$root" check/ok_selftest.sh
sg_reindex "$root"
st_case not_a_mode_recall_test_exempt 0 --what "a new recall test that names an existing file belongs to it" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh nam_selftest_name_only)
printf '%s\n' '#!/usr/bin/env bash' '# Deploys the release bottle; not a test at all.' '#' \
    '# Usage:  bash scripts/check/deploy_selftest.sh' '# Exit:   0 ok' '# Caller: by hand' 'exit 0' |
    sg_put "$root" check/deploy_selftest.sh
sg_reindex "$root"
st_case not_a_mode_name_alone_is_no_recall_test 1 --what "a _selftest.sh name with no Recall test for: line" \
    --has "not-a-mode: check/deploy_selftest.sh:1:" -- sg_gate "$root"

root=$(sg_fresh nam_recall_nothing)
printf '%s\n' '#!/usr/bin/env bash' '# A new tool that claims to be a recall test.' '#' \
    '# Recall test for: nothing/at_all.sh' '# Usage:  bash scripts/check/tool2.sh' '# Exit:   0 ok' \
    '# Caller: by hand' 'exit 0' | sg_put "$root" check/tool2.sh
sg_reindex "$root"
st_case not_a_mode_recall_test_of_nothing 1 --what "Recall test for: names a file that does not exist" \
    --has "not-a-mode: check/tool2.sh:1:" -- sg_gate "$root"

root=$(sg_fresh nam_recall_itself)
printf '%s\n' '#!/usr/bin/env bash' '# A new tool that claims to be its own recall test.' '#' \
    '# Recall test for: check/tool2.sh' '# Usage:  bash scripts/check/tool2.sh' '# Exit:   0 ok' \
    '# Caller: by hand' 'exit 0' | sg_put "$root" check/tool2.sh
sg_reindex "$root"
st_case not_a_mode_recall_test_of_itself 1 --what "Recall test for: names the file itself" \
    --has "not-a-mode: check/tool2.sh:1:" -- sg_gate "$root"

root=$(sg_fresh nam_module_nowhere)
printf '%s\n' '#!/usr/bin/env bash' '# A new entry point hidden behind a Module of: line.' '#' \
    '# Module of: check/does_not_exist.sh' 'exit 7' | sg_put "$root" check/hidden.sh
sg_reindex "$root"
st_case not_a_mode_module_of_nothing 1 --what "Module of: names no file: an entry point under every rule" \
    --has "not-a-mode: check/hidden.sh:1:" --has "usage: check/hidden.sh:1:" -- sg_gate "$root"

root=$(sg_fresh nam_module_itself)
printf '%s\n' '#!/usr/bin/env bash' '# A new entry point that is a module of itself.' '#' \
    '# Module of: check/hidden.sh' 'exit 7' | sg_put "$root" check/hidden.sh
sg_reindex "$root"
st_case not_a_mode_module_of_itself 1 --what "Module of: names the file itself" \
    --has "not-a-mode: check/hidden.sh:1:" -- sg_gate "$root"

root=$(sg_fresh nam_renamed)
git -C "$root" config diff.renames false
git -C "$root" mv scripts/check/ok.sh scripts/check/ok2.sh
printf 'check-ok:\n\t@bash scripts/check/ok2.sh\n\ntool:\n\t@python3 scripts/report/tool.py\n' >"$root/Makefile"
sed -i.bak 's|Module of: check/ok.sh|Module of: check/ok2.sh|' "$root/scripts/lib/helper.sh"
sed -i.bak 's|scripts/check/ok.sh|scripts/check/ok2.sh|' "$root/scripts/check/ok2.sh"
/bin/rm -f "$root/scripts/lib/helper.sh.bak" "$root/scripts/check/ok2.sh.bak"
sg_reindex "$root"
sg_commit_all "$root" "move ok.sh"
st_case not_a_mode_moved_file_is_not_new 0 --what "a committed git mv of an old file is not a new file" \
    --has "check-scripts: ok (4 files" --lacks "not-a-mode" -- sg_gate_at "$root" HEAD^

root=$(sg_fresh nam_unstaged_move)
mv "$root/scripts/check/ok.sh" "$root/scripts/check/ok2.sh"
printf 'check-ok:\n\t@bash scripts/check/ok2.sh\n\ntool:\n\t@python3 scripts/report/tool.py\n' >"$root/Makefile"
sg_reindex "$root"
st_case not_a_mode_unstaged_move_is_new 1 --what "a plain mv, not staged, reads as a new file" \
    --has "not-a-mode: check/ok2.sh:1:" -- sg_gate "$root"

root=$(sg_fresh nam_module)
printf '%s\n' '"""A planted module of the fixture tool.' '' 'Module of: report/tool.py' '"""' |
    sg_put "$root" report/tool_two.py
printf 'import tool_two\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case not_a_mode_module_exempt 0 --what "a new module belongs to its tool" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh nam_stale)
python3 - "$root/scripts/check/ok.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("set -euo", "# Not a mode of: check/gone.sh — a tool deleted since\nset -euo")
open(path, "w").write(text)
EOF
sg_reindex "$root"
st_case not_a_mode_stale 1 --what "an old file names a tool that no longer exists, at its own line" \
    --has "not-a-mode-stale: check/ok.sh:6: names check/gone.sh, which is not another entry point" -- sg_gate "$root"

root=$(sg_fresh nam_names_existing)
python3 - "$root/scripts/check/ok.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("set -euo", "# Not a mode of: report/tool.py — a tool that exists\nset -euo")
open(path, "w").write(text)
EOF
sg_reindex "$root"
st_case not_a_mode_names_existing 0 --what "the same line naming an existing entry point passes" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

n=0
for target in 'check/ok.sh' 'README.md' 'check' 'lib/helper.sh' 'INDEX.md'; do
    n=$((n + 1))
    root=$(sg_fresh "nam_target_$n")
    sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
        '# Caller: by hand' "# Not a mode of: $target — a reason" 'exit 0' | sg_put "$root" check/x.sh
    sg_reindex "$root"
    if [ "$target" = check/ok.sh ]; then
        st_case "not_a_mode_target_$n" 0 --what "naming another entry point ($target) passes" \
            --has "check-scripts: ok (5 files" -- sg_gate "$root"
        sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
            '# Caller: by hand' "# Not a mode of: check/x.sh — a reason" 'exit 0' | sg_put "$root" check/x.sh
        sg_reindex "$root"
        st_case "not_a_mode_target_itself" 1 --what "naming the file itself fails" \
            --has "not-a-mode-stale: check/x.sh:7: names check/x.sh, which is not another entry point" -- sg_gate "$root"
    else
        st_case "not_a_mode_target_$n" 1 --what "naming $target is not naming another entry point" \
            --has "not-a-mode-stale: check/x.sh:7: names $target, which is not another entry point" -- sg_gate "$root"
    fi
done

n=0
for separator in ' -- ' ' - '; do
    n=$((n + 1))
    root=$(sg_fresh "nam_separator_$n")
    sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
        '# Caller: by hand' "# Not a mode of: check/ok.sh${separator}a reason" 'exit 0' | sg_put "$root" check/x.sh
    sg_reindex "$root"
    st_case "not_a_mode_separator_$n" 0 --what "the tool and its reason, separated by '${separator}'" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

# ---- pins ----------------------------------------------------------------

# pinned_base <name> [header]: the base tree plus check/p.sh, which has no Usage: line and
# an undeclared exit 7, both pinned; committed, so these pins are the base's.
pinned_base() {
    local r
    r=$(sg_fresh "$1")
    sg_entry_sh '# CI gate: a pinned entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$SG_NAM" 'exit 7' |
        sg_put "$r" check/p.sh
    printf '%s\n' '# Accepted violations of the fixture tree.' '#' "${2:-$SG_PINNABLE}" \
        'usage check/p.sh' 'exit-undeclared check/p.sh 7' >"$r/$SG_PINS_REL"
    sg_reindex "$r"
    sg_commit_all "$r" "pin p.sh"
    printf '%s' "$r"
}

root=$(sg_fresh pin_holds)
sg_entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/x.sh
sg_pins_file 'usage check/x.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_accepts_a_violation 0 --what "a pinned violation passes and is counted" \
    --has "pinned: usage 1" --has "check-scripts: ok (6 files, 1 pinned)" \
    --has "pins: HEAD has no pin list; growth not checked" -- sg_gate "$root"

root=$(sg_fresh pin_other_rule)
sg_entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$SG_NAM" 'exit 7' |
    sg_put "$root" check/x.sh
sg_pins_file 'usage check/x.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_does_not_cover_another_rule 1 --what "a pin for usage does not hide an undeclared exit" \
    --has "exit-undeclared: check/x.sh:9: 7" --lacks "usage: check/x.sh" -- sg_gate "$root"

root=$(sg_fresh pin_stale)
sg_valid_sh 'exit 0' | sg_put "$root" check/x.sh
sg_pins_file 'usage check/x.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_stale_file_now_passes 1 --what "a pinned file that now passes fails, at the pin's own line" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: usage check/x.sh: no longer violated" -- sg_gate "$root"

root=$(sg_fresh pin_stale_gone)
sg_pins_file 'usage check/gone.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_stale_file_gone 1 --what "a pin on a file that does not exist" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: usage check/gone.sh" -- sg_gate "$root"

root=$(sg_fresh pin_unknown_rule)
sg_pins_file 'no-such-rule check/ok.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_stale_unknown_rule 1 --what "a pin naming no rule" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: no-such-rule check/ok.sh" -- sg_gate "$root"

root=$(sg_fresh pin_index)
sg_pins_file 'index-stale INDEX.md' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_cannot_hold_index_stale 1 --what "index-stale cannot be pinned" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: index-stale INDEX.md" -- sg_gate "$root"

root=$(sg_fresh pin_field_dup)
sg_pins_file 'field-dup check/ok.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_cannot_hold_field_dup 1 --what "field-dup cannot be pinned" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: field-dup check/ok.sh" -- sg_gate "$root"

root=$(pinned_base pin_base_holds)
st_case pin_list_equal_to_the_base_holds 0 --what "pins equal to the base's: nothing grew, no bootstrap note" \
    --has "pinned: usage 1, exit-undeclared 1" --lacks "pin-new" --lacks "has no pin list" -- sg_gate "$root"

root=$(pinned_base pin_growth)
sg_entry_sh '# CI gate: a planted entry point.' '#' '#' '# Exit:   0 clean' '# Caller: by hand' "$SG_NAM" 'exit 0' |
    sg_put "$root" check/foo.sh
printf 'usage check/foo.sh\n' >>"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_new_for_a_new_file 1 --what "a pin for a file the base did not pin, for a rule the base pinned" \
    --has "pin-new: report/scripts_gate_pins.txt:6: usage check/foo.sh is not pinned at the base" -- sg_gate "$root"

root=$(pinned_base pin_duplicate)
printf 'usage check/p.sh\n' >>"$root/$SG_PINS_REL"
st_case pin_duplicate 1 --what "the same pin twice" \
    --has "pin-dup: report/scripts_gate_pins.txt:6: usage check/p.sh is already pinned at line 4" -- sg_gate "$root"

root=$(pinned_base pin_unpinnable)
printf 'not-a-mode check/p.sh\n' >>"$root/$SG_PINS_REL"
st_case pin_unpinnable_rule 1 --what "not-a-mode cannot be pinned" \
    --has "pin-stale: report/scripts_gate_pins.txt:6: not-a-mode check/p.sh: not-a-mode cannot be pinned" -- sg_gate "$root"

root=$(pinned_base pin_unpinnable_violation)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/foo.sh' '# Exit:   0 clean' \
    '# Caller: by hand' '#' 'exit 0' | sg_put "$root" check/foo.sh
printf 'not-a-mode check/foo.sh\n' >>"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_does_not_hide_an_unpinnable_violation 1 --what "a pin for not-a-mode leaves the not-a-mode violation standing" \
    --has "not-a-mode: check/foo.sh:1:" \
    --has "pin-stale: report/scripts_gate_pins.txt:6: not-a-mode check/foo.sh: not-a-mode cannot be pinned" -- sg_gate "$root"

root=$(pinned_base pin_header "# pinnable: class purpose")
st_case pin_header_mismatch 1 --what "the pinnable line differs from the gate's rule set" \
    --has "pin-header: report/scripts_gate_pins.txt:3:" -- sg_gate "$root"

root=$(pinned_base pin_new_exit)
printf 'exit 99\n' >>"$root/scripts/check/p.sh"
st_case pin_exit_code_beyond_the_set 1 --what "a pinned file grows another undeclared exit" \
    --has "exit-undeclared: check/p.sh:10: 99 is not declared" -- sg_gate "$root"

root=$(pinned_base pin_exit_declared)
python3 - "$root/scripts/check/p.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Exit:   0 clean\n", "# Exit:   0 clean; 7 a violation\n")
open(path, "w").write(text)
EOF
st_case pin_exit_only_code_declared 1 --what "the file's only undeclared code is now declared: the pin is stale" \
    --has "pin-stale: report/scripts_gate_pins.txt:5: exit-undeclared check/p.sh: no longer violated" -- sg_gate "$root"

root=$(pinned_base pin_exit_declared_other)
python3 - "$root/scripts/check/p.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Exit:   0 clean\n", "# Exit:   0 clean; 7 a violation\n") + "exit 99\n"
open(path, "w").write(text)
EOF
st_case pin_exit_code_now_declared 1 --what "a code in the pinned set is declared while another is not" \
    --has "pin-stale: report/scripts_gate_pins.txt:5: exit-undeclared check/p.sh: code 7 is fixed or gone" \
    --has "exit-undeclared: check/p.sh:10: 99 is not declared" -- sg_gate "$root"

root=$(pinned_base pin_exit_grown_set)
printf 'exit 99\n' >>"$root/scripts/check/p.sh"
python3 - "$root/$SG_PINS_REL" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("check/p.sh 7\n", "check/p.sh 7,99\n")
open(path, "w").write(text)
EOF
st_case pin_exit_set_grows 1 --what "the pinned code set gains a code the base did not pin" \
    --has "pin-new: report/scripts_gate_pins.txt:5: exit-undeclared check/p.sh: code 99 is not pinned at the base" \
    --lacks "exit-undeclared: check/p.sh" -- sg_gate "$root"

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
    --lacks "exit-undeclared check/p.sh: code" -- sg_gate "$root"

root=$(pinned_base pin_moved)
git -C "$root" mv scripts/check/p.sh scripts/check/q.sh
python3 - "$root/$SG_PINS_REL" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("check/p.sh", "check/q.sh")
open(path, "w").write(text)
EOF
sg_reindex "$root"
st_case pin_follows_a_moved_file 0 --what "a git mv with its pin lines renamed keeps the pins" \
    --has "pinned: usage 1, exit-undeclared 1" --lacks "pin-new" -- sg_gate "$root"

root=$(pinned_base pin_new_rule "# pinnable: class purpose usage exit-undeclared")
sg_nocaller_sh 'exit 0' | sg_put "$root" check/r.sh
python3 - "$root/$SG_PINS_REL" "$SG_PINNABLE" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# pinnable: class purpose usage exit-undeclared\n", sys.argv[2] + "\n", 1)
open(path, "w").write(text)
EOF
printf 'caller check/r.sh\n' >>"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_seeds_a_rule_the_base_lacked 0 --what "a rule absent from the base's pinnable line may seed pins" \
    --has "pinned: usage 1, exit-undeclared 1, caller 1" -- sg_gate "$root"

# ---- recall-missing, recall-stale ----------------------------------------

# gate_body: the body of a planted tool that answers --lib, so that mode occurs in its text.
GATE_BODY='[ "${1:-}" = --lib ] || exit 0'

root=$(sg_fresh recall_make_call)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(MAKE) --no-print-directory gates\n\ngates:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case recall_missing_through_make_call 1 --what "a tool reached only through a \$(MAKE) line of ci has no recall test" \
    --has "recall-missing: check/gate.sh:1: mode default is run by \`make ci\` and no \`Recall test for:\` names it" -- sg_gate "$root"

root=$(sg_fresh recall_prerequisite)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci: gates\n\ngates:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case recall_missing_through_prerequisite 1 --what "a tool run by a prerequisite of ci has no recall test" \
    --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh recall_cycle)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(MAKE) a\n\na:\n\t@$(MAKE) ci\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case recall_closure_survives_a_cycle 1 --what "targets that call each other end the walk and the tool inside is still found" \
    --has "recall-missing: check/gate.sh:1: mode default" --lacks "timed out" -- sg_bounded_gate "$root"

root=$(sg_fresh recall_not_in_ci)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@echo hi\n\nother:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case recall_tool_outside_ci_asks_nothing 0 --what "a tool only another target runs needs no recall test" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh silenced_gate)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || true\n'
sg_reindex "$root"
st_case silenced_gate_is_reported 1 --what "|| true on a gate: the line throws its exit code away" \
    --has "silenced: check/gate.sh:1: mode default is run by \`make ci\` on a line that throws its exit code away" \
    --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_advisory_purpose)
sg_entry_sh '# Advisory report: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed' '#' "$SG_NAM" "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || true\n'
sg_reindex "$root"
st_case silenced_purpose_line_exempts_nothing 1 --what "a purpose line that says Advisory does not excuse a silenced line" \
    --has "silenced: check/gate.sh:1: mode default" --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_with_recall_test)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || true\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case silenced_gate_with_a_recall_test_is_reported 1 --what "a recall test for the tool does not make its silenced line harmless" \
    --has "silenced: check/gate.sh:1: mode default" --lacks "recall-missing" -- sg_gate "$root"

root=$(sg_fresh silenced_or_exit)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || exit 1\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case silenced_or_exit_1_is_a_guarded_tail 1 --what "|| exit 1 is a ||: the shape rule cannot tell it keeps the failure, so the mode is pinned" \
    --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"

for form in '|| :' '|| exit 0' '|| echo skipped' '|| printf x' '; true' '; :' '|| true; echo done' '&& echo ok || true'; do
    root=$(sg_fresh "silenced_form_$(printf '%s' "$form" | tr -c 'a-z' _)")
    sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
    sg_make_ci "$root" "ci:\\n\\t@bash scripts/check/gate.sh $form\\n"
    sg_reindex "$root"
    st_case "silenced_form_${form// /_}" 1 --what "a recipe line that ends in: $form" \
        --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"
done

root=$(sg_fresh silenced_dash_prefix)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t-@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_dash_prefix 1 --what "a - in front of the command makes make ignore its failure" \
    --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_in_a_string)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh "|| true"\n'
sg_reindex "$root"
st_case silenced_words_in_a_string_are_an_argument 1 --what "|| true inside quotes is an argument, not a silenced line" \
    --lacks "silenced:" --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_make_call)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(MAKE) --no-print-directory gates || true\n\ngates:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_through_a_make_call 1 --what "|| true on a \$(MAKE) line silences every tool of the target it calls" \
    --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_make_call_loud_too)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci: gates\n\t@$(MAKE) --no-print-directory gates || true\n\ngates:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_target_reached_loudly_too 1 --what "a target ci also reaches through a prerequisite is not silenced" \
    --lacks "silenced:" --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_nested_make_call)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(MAKE) --no-print-directory outer || true\n\nouter:\n\t@$(MAKE) --no-print-directory inner\n\ninner:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_through_two_make_calls 1 --what "a target only a silenced line reaches silences the \$(MAKE) lines of its own, down to the tool" \
    --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_pin_holds)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || true\n'
sg_pins_file 'recall-missing check/gate.sh default' 'silenced check/gate.sh default' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case silenced_pin_holds 0 --what "a pinned silenced mode passes and the rule is counted" \
    --has "pinned: recall-missing 1, silenced 1" -- sg_gate "$root"

root=$(sg_fresh silenced_pin_new_mode)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh || true\n\t@bash scripts/check/gate.sh --lib || true\n'
sg_pins_file 'recall-missing check/gate.sh default,--lib' 'silenced check/gate.sh default' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case silenced_pin_new_mode 1 --what "a second silenced mode of a pinned tool is outside the pinned set" \
    --has "silenced: check/gate.sh:1: mode --lib" --lacks "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_pin_stale)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate_selftest.sh\n'
sg_pins_file 'silenced check/gate.sh default' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case silenced_pin_stale 1 --what "the line no longer discards the exit code: the pin is stale" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: silenced check/gate.sh: no longer violated" -- sg_gate "$root"

# A line is silenced unless its tool is a simple command, last on the line or followed only by && <command>.
silencing_shapes=(
    'TOOL | cat' 'TOOL; echo done' 'TOOL || (true)' 'TOOL || /usr/bin/true' 'TOOL || { true; }'
    'if TOOL; then :; fi' 'TOOL &' 'TOOL; exit 0 ;' '( TOOL )' '{ TOOL; }' '! TOOL' 'x=$$(TOOL)'
    'echo "$$(TOOL)"' 'TOOL || true && echo ok' 'TOOL 2>&1 | tee log.txt' 'TOOL |& cat' 'TOOL || TOOL'
)
n=0
for form in "${silencing_shapes[@]}"; do
    n=$((n + 1))
    root=$(sg_fresh "silenced_shape_$n")
    sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
    sg_make_ci "$root" "ci:\\n\\t@${form//TOOL/bash scripts/check/gate.sh}\\n"
    sg_reindex "$root"
    st_case "silenced_shape_$n" 1 --what "the recipe line @$form throws the tool's exit code away" \
        --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"
done

root=$(sg_fresh silenced_shape_variable)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'QUIET := || true\n\nci:\n\t@bash scripts/check/gate.sh $(QUIET)\n'
sg_reindex "$root"
st_case silenced_shape_through_a_variable 1 --what "a make variable that expands to || true silences the line it is on" \
    --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_shape_keep_going)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(MAKE) -k gates\n\ngates:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_shape_make_keep_going 1 --what "\$(MAKE) -k goes on after a failure: the tools of its target are silenced" \
    --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_shape_make_plain)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(MAKE) --no-print-directory -j2 gates\n\ngates:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_shape_make_without_keep_going 1 --what "\$(MAKE) with other options silences nothing" \
    --lacks "silenced:" --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_shape_controls)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate2.sh
sg_recall_sh '#' 'check/gate.sh' 'check/gate2.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@cd . && bash scripts/check/gate.sh >/dev/null 2>&1 && bash scripts/check/gate2.sh\n\t@time bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case silenced_shape_controls 0 --what "a tool after cd &&, with redirections, then && another tool, then a timed tool: nothing is silenced" \
    --has "check-scripts: ok (7 files" -- sg_gate "$root"

for flag in -i -ik --keep-going --ignore-errors; do
    root=$(sg_fresh "silenced_shape_make_$(printf '%s' "$flag" | tr -c 'a-z' _)")
    sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
    sg_make_ci "$root" "ci:\\n\\t@\$(MAKE) $flag gates\\n\\ngates:\\n\\t@bash scripts/check/gate.sh\\n"
    sg_reindex "$root"
    st_case "silenced_shape_make_option_${flag//-/_}" 1 --what "\$(MAKE) $flag goes on after a failure" \
        --has "silenced: check/gate.sh:1: mode default" -- sg_gate "$root"
done

root=$(sg_fresh silenced_shape_substitution_argument)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh $$(cd . && pwd)\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case silenced_shape_substitution_as_an_argument 0 --what "a \$( ) argument holds operators of its own and silences nothing" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh silenced_shape_shell_variable_head)
sg_make_ci "$root" 'ci:\n\t@$$SHELL -c true\n'
sg_reindex "$root"
st_case silenced_shape_shell_expansion_as_command_word 0 --what "\$\$SHELL is the shell's variable, not a make expansion the gate cannot follow" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

root=$(sg_fresh silenced_shape_curdir_head)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(CURDIR)/scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_shape_path_command_word 1 --what "\$(CURDIR)/scripts/... as the command word is the tool, not an unresolved expansion" \
    --lacks "unavailable" --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_shape_shell_substitution_is_not_a_variable)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'QUIET := || true\n\nci:\n\t@bash scripts/check/gate.sh $$(QUIET)\n'
sg_reindex "$root"
st_case silenced_shape_double_dollar_is_the_shell 1 --what "\$\$(QUIET) is a shell substitution, not the make variable QUIET" \
    --lacks "silenced:" --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh silenced_shape_call)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'quietly = $(1) || true\n\nci:\n\t@$(call quietly,bash scripts/check/gate.sh)\n'
sg_reindex "$root"
st_case silenced_shape_make_call_is_unavailable 2 --what "a \$(call ...) in a closure recipe line cannot be followed" \
    --has 'check-scripts: unavailable (`$(call ...)` in the recipe line' -- sg_gate "$root"

root=$(sg_fresh silenced_shape_command_word)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@$(RUNNER) bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_shape_unresolved_command_word_is_unavailable 2 --what "a command word the gate cannot expand is exit 2, not a pass" \
    --has 'check-scripts: unavailable (`$(RUNNER)` is a command word of a recipe line of the `make ci` closure' -- sg_gate "$root"

root=$(sg_fresh silenced_shape_command_word_known)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'RUNNER := env\n\nci:\n\t@$(RUNNER) bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case silenced_shape_known_command_word_is_expanded 1 --what "the same word defined once as env is followed: the tool is found and asks for a recall test" \
    --lacks "unavailable" --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh recall_covered)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_covered_by_a_test_ci_runs 0 --what "a recall test that ci runs covers the default mode, and asks for none itself" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_test_not_run)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '# Caller: by hand' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n'
sg_reindex "$root"
st_case recall_test_ci_does_not_run 1 --what "a recall test that make ci never runs does not cover the tool" \
    --has "recall-missing: check/gate.sh:1: mode default: its recall test check/gate_selftest.sh is not run by \`make ci\`" -- sg_gate "$root"

root=$(sg_fresh recall_mode_unnamed)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_mode_not_named 1 --what "ci runs the tool with --lib and the test names only the default mode" \
    --has "recall-missing: check/gate.sh:1: mode --lib is run by" --lacks "mode default" -- sg_gate "$root"

root=$(sg_fresh recall_mode_named)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --lib' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_mode_named 0 --what "the test names --lib: covered" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_non_mode_argument)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh some/path.txt "$(X)" 42\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_argument_that_is_no_mode_is_the_default 0 --what "a path, a substitution and a number as first argument leave the default mode" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_two_modes_one_line)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh default --lib' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_two_modes_on_one_line 0 --what "one field line may name several modes" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_two_modes_two_lines)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' 'check/gate.sh --lib' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_two_modes_on_two_lines 0 --what "a field of two lines covers both" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_stale_tool)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '# Caller: by hand' 'check/gone.sh' | sg_put "$root" check/gate_selftest.sh
sg_reindex "$root"
st_case recall_stale_names_no_file 1 --what "Recall test for: names a file that does not exist, at the field's line" \
    --has "recall-stale: check/gate_selftest.sh:4: names check/gone.sh, which is no file of the tree" -- sg_gate "$root"

root=$(sg_fresh recall_stale_mode)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '# Caller: by hand' 'check/gate.sh --nope' | sg_put "$root" check/gate_selftest.sh
sg_reindex "$root"
st_case recall_stale_mode_not_in_tool 1 --what "a mode token that occurs nowhere in the tool" \
    --has "recall-stale: check/gate_selftest.sh:4: check/gate.sh: \`--nope\` occurs nowhere in check/gate.sh" -- sg_gate "$root"

root=$(sg_fresh recall_stale_token)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '# Caller: by hand' 'check/gate.sh (all modes)' | sg_put "$root" check/gate_selftest.sh
sg_reindex "$root"
st_case recall_stale_prose_after_the_tool 1 --what "words after the tool are mode tokens, and a parenthesis is not one" \
    --has "recall-stale: check/gate_selftest.sh:4: check/gate.sh: \`(all\` is not a mode token" -- sg_gate "$root"

root=$(sg_fresh recall_stale_self)
sg_recall_sh '# Caller: by hand' 'check/gate_selftest.sh' | sg_put "$root" check/gate_selftest.sh
sg_reindex "$root"
st_case recall_stale_names_itself 1 --what "a recall test of itself" \
    --has "recall-stale: check/gate_selftest.sh:4: \`Recall test for:\` names the file itself" -- sg_gate "$root"

root=$(sg_fresh recall_stale_second_line)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '# Caller: by hand' 'check/gate.sh' 'check/gone.sh' | sg_put "$root" check/gate_selftest.sh
sg_reindex "$root"
st_case recall_stale_second_line 1 --what "the second line of the field is the one named" \
    --has "recall-stale: check/gate_selftest.sh:5: names check/gone.sh" --lacks "gate_selftest.sh:4" -- sg_gate "$root"

root=$(sg_fresh recall_valid_claim)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '# Caller: by hand' 'check/gate.sh --lib' | sg_put "$root" check/gate_selftest.sh
sg_reindex "$root"
st_case recall_valid_claim 0 --what "a claim on a tool and a mode that occurs in it" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_claim_not_run)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sed '/^bash scripts\//d' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_claim_of_a_tool_it_does_not_run 1 --what "a test that names a tool and never runs it covers nothing and is itself reported" \
    --has "recall-stale: check/gate_selftest.sh:4: names check/gate.sh, which this file does not run, source or import" \
    --has "recall-missing: check/gate.sh:1: mode default is run by" -- sg_gate "$root"

root=$(sg_fresh recall_claim_not_run_exempts_nothing)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' \
    '# Recall test for: report/tool.py' '#' 'bin=target/release/rmlx' | sg_put "$root" check/newgate.sh
printf '%s\n' 'cat scripts/check/gone_table.txt' >>"$root/scripts/check/newgate.sh"
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/newgate.sh\n'
sg_reindex "$root"
st_case recall_claim_not_run_exempts_nothing 1 --what "a new gate with a claim on a file it never runs is not a recall test: four rules still apply" \
    --has "not-a-mode: check/newgate.sh:1:" --has "recall-missing: check/newgate.sh:1: mode default" \
    --has "dead-path: check/newgate.sh:10: scripts/check/gone_table.txt names no file" \
    --has "binary-literal: check/newgate.sh:9:" --has "recall-stale: check/newgate.sh:6:" -- sg_gate "$root"

root=$(sg_fresh recall_claim_in_a_file_not_named_as_a_test)
sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean; 1 failed' \
    '# Recall test for: report/tool.py' '#' 'bin=target/release/rmlx' | sg_put "$root" check/newgate.sh
printf '%s\n' 'python3 scripts/report/tool.py' 'cat scripts/check/gone_table.txt' >>"$root/scripts/check/newgate.sh"
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/newgate.sh\n'
sg_reindex "$root"
st_case recall_claim_in_a_file_not_named_as_a_test 1 --what "a new gate that runs the tool it claims, under a name that is not *_selftest or *_fixtures: four rules still apply" \
    --has "recall-stale: check/newgate.sh:6: a \`Recall test for:\` claim needs a file named *_selftest.<ext> or *_fixtures.<ext>" \
    --has "not-a-mode: check/newgate.sh:1:" --has "recall-missing: check/newgate.sh:1: mode default" \
    --has "dead-path: check/newgate.sh:11: scripts/check/gone_table.txt names no file" \
    --has "binary-literal: check/newgate.sh:9:" -- sg_gate "$root"

root=$(sg_fresh recall_claim_in_a_fixtures_file)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_fixtures.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate_fixtures.sh\n'
sg_reindex "$root"
st_case recall_claim_in_a_fixtures_file 0 --what "a file named *_fixtures.sh that runs the tool holds its claim" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_claim_without_the_underscore)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gateselftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gateselftest.sh\n'
sg_reindex "$root"
st_case recall_claim_needs_the_underscore 1 --what "gateselftest.sh does not end in _selftest.sh" \
    --has "recall-stale: check/gateselftest.sh:4: a \`Recall test for:\` claim needs a file named" -- sg_gate "$root"

root=$(sg_fresh recall_claim_in_a_file_named_selftest_in_the_middle)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/selftest_gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/selftest_gate.sh\n'
sg_reindex "$root"
st_case recall_claim_needs_the_suffix 1 --what "selftest at the start of a name is not the suffix" \
    --has "recall-stale: check/selftest_gate.sh:4: a \`Recall test for:\` claim needs a file named" \
    --has "recall-missing: check/gate.sh:1: mode default" -- sg_gate "$root"

root=$(sg_fresh recall_claim_mode_not_in_test)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --lib' | sed 's|^bash scripts/check/gate.sh --lib$|bash scripts/check/gate.sh|' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_claim_of_a_mode_the_test_never_runs 1 --what "the test names --lib and only runs the default mode: the claim is stale and the mode uncovered" \
    --has "recall-stale: check/gate_selftest.sh:4: check/gate.sh: \`--lib\` occurs nowhere in the code of check/gate_selftest.sh" \
    --has "recall-missing: check/gate.sh:1: mode --lib is run by" -- sg_gate "$root"

root=$(sg_fresh recall_claim_mode_in_comment)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --lib' | sed 's|^bash scripts/check/gate.sh --lib$|bash scripts/check/gate.sh|' | sed 's|^exit 0$|# bash scripts/check/gate.sh --lib\nexit 0|' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_claim_of_a_mode_named_in_a_comment 1 --what "a mode token that only a comment of the test holds is not exercised" \
    --has "recall-stale: check/gate_selftest.sh:4: check/gate.sh: \`--lib\` occurs nowhere in the code of check/gate_selftest.sh" -- sg_gate "$root"

root=$(sg_fresh recall_claim_one_of_two_holds)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --lib --nope' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case recall_claim_with_one_good_mode_and_one_bad 1 --what "the mode that holds covers its line; the other is reported" \
    --has "recall-stale: check/gate_selftest.sh:4: check/gate.sh: \`--nope\` occurs nowhere in check/gate.sh" \
    --lacks "recall-missing" --lacks "not-a-mode" -- sg_gate "$root"

root=$(sg_fresh recall_claim_python_test)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_put "$root" check/gate_selftest.py <<'EOF'
#!/usr/bin/env python3
"""A planted Python recall test.

Recall test for: check/gate.sh --lib
Usage:  python3 scripts/check/gate_selftest.py
Exit:   0 held
"""

import subprocess

subprocess.run(["bash", "scripts/check/gate.sh", "--lib"])
EOF
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@python3 scripts/check/gate_selftest.py\n'
sg_reindex "$root"
st_case recall_claim_python_test_holds 0 --what "a Python recall test whose command vector holds the mode covers it" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh closure_var_make_call)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'NEW_GATES := check-g\n\ncheck-g:\n\t@bash scripts/check/gate.sh\n\nci:\n\t@$(MAKE) --no-print-directory $(NEW_GATES)\n'
sg_reindex "$root"
st_case closure_follows_a_variable_in_a_make_call 1 --what "a target named through a variable is in the closure: its tool asks for a recall test" \
    --has "recall-missing: check/gate.sh:1: mode default is run by" -- sg_gate "$root"

root=$(sg_fresh closure_var_prerequisite)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'NEW_GATES = check-g other-g\n\ncheck-g:\n\t@bash scripts/check/gate.sh\n\nother-g:\n\t@echo hi\n\nci: $(NEW_GATES)\n'
sg_reindex "$root"
st_case closure_follows_a_variable_prerequisite 1 --what "a prerequisite named through a two-word variable is followed" \
    --has "recall-missing: check/gate.sh:1: mode default is run by" -- sg_gate "$root"

root=$(sg_fresh closure_var_unknown)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci: $(UNSET_GATES)\n'
sg_reindex "$root"
st_case closure_unknown_variable_is_unavailable 2 --what "a prerequisite the gate cannot expand is exit 2, not a pass" \
    --has 'check-scripts: unavailable (`$(UNSET_GATES)` in the prerequisites of `ci` is a make expansion the gate cannot follow' -- sg_gate "$root"

root=$(sg_fresh closure_var_twice)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'NEW_GATES := check-g\nNEW_GATES := other\n\nci:\n\t@$(MAKE) $(NEW_GATES)\n'
sg_reindex "$root"
st_case closure_variable_defined_twice_is_unavailable 2 --what "a variable defined twice is not simple" \
    --has 'in a `$(MAKE)` line of `ci` is a make expansion' -- sg_gate "$root"

root=$(sg_fresh closure_var_nested)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'BASE := check\nNEW_GATES := $(BASE)-g\n\nci: $(NEW_GATES)\n'
sg_reindex "$root"
st_case closure_variable_with_an_expansion_is_unavailable 2 --what "a value that expands another variable is not simple" \
    --has 'in the prerequisites of `ci` is a make expansion' -- sg_gate "$root"

root=$(sg_fresh closure_var_embedded)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'WHICH := g\n\nci: check-$(WHICH)\n'
sg_reindex "$root"
st_case closure_variable_inside_a_word_is_unavailable 2 --what "a variable inside a word is not followed" \
    --has 'check-$(WHICH)` in the prerequisites of `ci`' -- sg_gate "$root"

root=$(sg_fresh mode_value_option)
sg_nocaller_sh 'case "${1:-}" in --root) shift 2 ;; esac; case "${1:-}" in --strict) exit 1 ;; esac' | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --strict' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --root . --strict\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case mode_skips_an_option_with_a_value 0 --what "--root . takes its value: the mode of the line is --strict, which the test names" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh mode_value_option_not_a_mode)
sg_nocaller_sh 'case "${1:-}" in --root) shift 2 ;; esac; case "${1:-}" in --strict) exit 1 ;; esac' | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --root' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --root . --strict\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case mode_option_with_a_value_is_no_mode 1 --what "a test that names --root covers nothing the line runs" \
    --has "recall-missing: check/gate.sh:1: mode --strict is run by" -- sg_gate "$root"

root=$(sg_fresh mode_value_only)
sg_nocaller_sh 'case "${1:-}" in --base) shift 2 ;; esac' | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --base "$(BASE_ARG)"\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case mode_line_with_only_an_option_and_its_value_is_the_default 0 --what "--base VALUE alone is the default mode" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh mode_positional_first)
sg_nocaller_sh 'case "${1:-}" in --lib) exit 0 ;; esac' | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh some/path.txt --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case mode_first_argument_that_is_no_mode_ends_the_scan 0 --what "a path first, then --lib: the line has the mode default, as the first argument decides" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh mode_equals)
sg_nocaller_sh '[ "${1:-}" = --mode=a ] || [ "${1:-}" = --mode=b ] || exit 0' | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --mode=a' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --mode=a\n\t@bash scripts/check/gate.sh --mode=b\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case mode_equals_form_is_a_mode_as_written 1 --what "--mode=a and --mode=b are two modes: the test names one" \
    --has "recall-missing: check/gate.sh:1: mode --mode=b is run by" --lacks "mode --mode=a" --lacks "mode default" -- sg_gate "$root"

root=$(sg_fresh mode_equals_both)
sg_nocaller_sh '[ "${1:-}" = --mode=a ] || [ "${1:-}" = --mode=b ] || exit 0' | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --mode=a --mode=b' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --mode=a\n\t@bash scripts/check/gate.sh --mode=b\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case mode_equals_both_named 0 --what "a test that names both spellings covers both lines" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh recall_claim_python_docstring)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_put "$root" check/gate_selftest.py <<'EOF'
#!/usr/bin/env python3
"""A planted Python recall test.

Recall test for: check/gate.sh --lib
Usage:  python3 scripts/check/gate_selftest.py --lib
Exit:   0 held
"""

import subprocess

subprocess.run(["bash", "scripts/check/gate.sh"])
EOF
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@python3 scripts/check/gate_selftest.py\n'
sg_reindex "$root"
st_case recall_claim_of_a_mode_only_the_docstring_names 1 --what "a mode token in the docstring of a Python test is not code" \
    --has "recall-stale: check/gate_selftest.py:4: check/gate.sh: \`--lib\` occurs nowhere in the code of check/gate_selftest.py" -- sg_gate "$root"

for kind in class function async; do
    root=$(sg_fresh "recall_claim_python_${kind}_docstring")
    sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
    {
        printf '%s\n' '#!/usr/bin/env python3' '"""A planted Python recall test.' '' 'Recall test for: check/gate.sh --lib' \
            'Usage:  python3 scripts/check/gate_selftest.py' 'Exit:   0 held' '"""' '' 'import subprocess' ''
        case "$kind" in
        class) printf '%s\n' 'class Case:' '    """Runs the gate with --lib."""' '' '    def run(self):' \
            '        subprocess.run(["bash", "scripts/check/gate.sh"])' '' '' 'Case().run()' ;;
        function) printf '%s\n' 'def run():' '    """Runs the gate with --lib."""' '    subprocess.run(["bash", "scripts/check/gate.sh"])' '' '' 'run()' ;;
        async) printf '%s\n' 'async def run():' '    """Runs the gate with --lib."""' '    subprocess.run(["bash", "scripts/check/gate.sh"])' '' '' 'run()' ;;
        esac
    } | sg_put "$root" check/gate_selftest.py
    sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --lib\n\t@python3 scripts/check/gate_selftest.py\n'
    sg_reindex "$root"
    st_case "recall_claim_of_a_mode_only_a_${kind}_docstring_names" 1 --what "a mode token in the docstring of a $kind of a Python test is not code" \
        --has "recall-stale: check/gate_selftest.py:4: check/gate.sh: \`--lib\` occurs nowhere in the code of check/gate_selftest.py" -- sg_gate "$root"
done

root=$(sg_fresh mode_equals_then_positional)
sg_nocaller_sh '[ "${1:-}" = --mode=a ] || exit 0' | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh --mode=a' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh --mode=a some/path.txt\n\t@bash scripts/check/gate_selftest.sh\n'
sg_reindex "$root"
st_case mode_equals_form_takes_no_value 0 --what "--mode=a followed by a path is still the mode --mode=a" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

# ---- field-dup -----------------------------------------------------------

root=$(sg_fresh field_dup_recall)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
printf '%s\n' '#!/usr/bin/env bash' '# A planted recall test.' '#' \
    '# Recall test for: check/gate.sh' '# Recall test for: check/gate.sh --lib' '#' \
    '# Usage:  bash scripts/check/gate_selftest.sh' '# Exit:   0 held' '# Caller: by hand' "$SG_NAM" \
    'set -euo pipefail' 'bash scripts/check/gate.sh' 'bash scripts/check/gate.sh --lib' 'exit 0' |
    sg_put "$root" check/gate_selftest.sh
sg_reindex "$root"
st_case field_dup_recall_test_twice 1 --what "a second Recall test for: line is not dropped without a word: both lines are named" \
    --has "field-dup: check/gate_selftest.sh:5: \`Recall test for:\` is declared twice, at lines 4 and 5" -- sg_gate "$root"

root=$(sg_fresh field_dup_usage)
sg_entry_sh '# CI gate: a planted entry point.' '#' \
    "$(printf '%s\n' '# Usage:  bash scripts/check/ok.sh' '# Usage:  bash scripts/check/ok.sh --lib')" \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case field_dup_any_field 1 --what "the rule reads every field, not the recall claim alone" \
    --has "field-dup: check/x.sh:5: \`Usage:\` is declared twice, at lines 4 and 5" -- sg_gate "$root"

root=$(sg_fresh field_continuation)
sg_entry_sh '# CI gate: a planted entry point.' '#' \
    "$(printf '%s\n' '# Usage:  bash scripts/check/ok.sh' '#         bash scripts/check/ok.sh --lib')" \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'exit 0' | sg_put "$root" check/x.sh
sg_reindex "$root"
st_case field_continuation_lines_are_one_field 0 --what "a field that goes on over lines with no field name is one field" \
    --has "check-scripts: ok (5 files" --lacks "field-dup" -- sg_gate "$root"

# ---- lib-consumers, lib-run ----------------------------------------------

SRC_Y='. "$(dirname "$0")/../lib/y.sh"'

root=$(sg_fresh libc_one)
sg_lib_sh | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case lib_consumers_one 1 --what "a lib/ file one script sources" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s), needs 2 or a \`Module of:\` line" -- sg_gate "$root"

root=$(sg_fresh libc_two_scripts)
sg_lib_sh | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_valid_sh "$SRC_Y" | sg_put "$root" check/z.sh
sg_reindex "$root"
st_case lib_consumers_two_scripts 0 --what "two scripts source it" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh libc_script_and_crate)
sg_lib_sh | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
mkdir -p "$root/crates/c/src"
printf 'const S: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/y.sh"));\n' >"$root/crates/c/src/lib.rs"
sg_reindex "$root"
st_case lib_consumers_script_and_crate 0 --what "one script and one crates/ source count as two" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh libc_zero)
sg_lib_sh | sg_put "$root" lib/y.sh
sg_reindex "$root"
st_case lib_consumers_none 1 --what "a lib/ file nothing uses, and it is not an orphan" \
    --has "lib-consumers: lib/y.sh:1: 0 consumer(s)" --lacks "orphan" -- sg_gate "$root"

root=$(sg_fresh libc_twice)
sg_lib_sh | sg_put "$root" lib/y.sh
printf '%s\n%s\n' "$SRC_Y" "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case lib_consumers_one_file_counts_once 1 --what "one script that sources it twice is one consumer" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s)" -- sg_gate "$root"

root=$(sg_fresh libc_recall_test)
sg_lib_sh | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_recall_sh '# Caller: by hand' 'check/ok.sh' | sed '$d' | sg_put "$root" check/ok_selftest.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok_selftest.sh"
sg_reindex "$root"
st_case lib_consumers_recall_test_not_counted 1 --what "a script and a recall test: the recall test does not count" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s)" -- sg_gate "$root"

root=$(sg_fresh libc_selftest_support)
sg_lib_sh | sg_put "$root" lib/selftest_support.sh
printf '. "$(dirname "$0")/../lib/selftest_support.sh"\n' >>"$root/scripts/check/ok.sh"
sg_recall_sh '# Caller: by hand' 'check/ok.sh' | sed '$d' | sg_put "$root" check/ok_selftest.sh
printf '. "$(dirname "$0")/../lib/selftest_support.sh"\n' >>"$root/scripts/check/ok_selftest.sh"
sg_reindex "$root"
st_case lib_consumers_selftest_support_counts_recall_tests 0 --what "lib/selftest_support.sh: a script and a recall test are two" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh libc_hook)
printf '%s\n' '# A planted lib data file.' >"$root/scripts/lib/data.txt"
printf '%s\n' 'cat "$(dirname "$0")/../lib/data.txt"' >>"$root/scripts/check/ok.sh"
printf 'repos:\n  - repo: local\n    hooks:\n      - id: x\n        entry: cat scripts/lib/data.txt\n' >"$root/.pre-commit-config.yaml"
sg_reindex "$root"
st_case lib_consumers_pre_commit_is_no_consumer 1 --what "a hook that reads a lib/ file is not a script or a crate: one consumer" \
    --has "lib-consumers: lib/data.txt:1: 1 consumer(s)" -- sg_gate "$root"

root=$(sg_fresh libc_module_of)
sg_lib_sh '# Module of: check/ok.sh' | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case lib_consumers_module_of_exempts 0 --what "Module of: its one consumer: not counted" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh libc_module_of_missing)
sg_lib_sh '# Module of: check/none.sh' | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case lib_consumers_module_of_naming_nothing 1 --what "a Module of: that names no file exempts nothing" \
    --has "lib-consumers: lib/y.sh:1: 1 consumer(s)" --has "module-of: lib/y.sh:3: names check/none.sh" -- sg_gate "$root"

# lib_py: a planted lib/ Python file with the header of an entry point, for the cases where something runs it.
lib_py() {
    printf '%s\n' '"""A planted lib module.' '' 'Usage:  python3 scripts/lib/y.py' 'Exit:   0 always' \
        'Not a mode of: report/tool.py — a planted entry point' '"""'
}

root=$(sg_fresh librun_recipe)
lib_py | sg_put "$root" lib/y.py
sg_make_ci "$root" 'lib-y:\n\t@python3 scripts/lib/y.py\n'
sg_reindex "$root"
st_case lib_run_by_a_recipe 1 --what "a recipe runs a lib/ file, named by its target" \
    --has "lib-run: lib/y.py:1: lib/ is sourced or imported, never run; run by make:lib-y" -- sg_gate "$root"

root=$(sg_fresh librun_script)
lib_py | sg_put "$root" lib/y.py
printf '%s\n' 'python3 "$(dirname "$0")/../lib/y.py"' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case lib_run_by_a_script 1 --what "a script runs a lib/ file, named by its path" \
    --has "lib-run: lib/y.py:1: lib/ is sourced or imported, never run; run by check/ok.sh" -- sg_gate "$root"

root=$(sg_fresh librun_workflow)
lib_py | sg_put "$root" lib/y.py
mkdir -p "$root/.github/workflows"
printf 'jobs:\n  gates:\n    steps:\n      - run: python3 scripts/lib/y.py\n' >"$root/.github/workflows/ci.yml"
sg_reindex "$root"
st_case lib_run_by_a_workflow 1 --what "a hosted-CI step runs a lib/ file" \
    --has "run by ci.yml" -- sg_gate "$root"

root=$(sg_fresh librun_pre_commit)
lib_py | sg_put "$root" lib/y.py
printf 'repos:\n  - repo: local\n    hooks:\n      - id: y\n        entry: python3 scripts/lib/y.py\n' >"$root/.pre-commit-config.yaml"
sg_reindex "$root"
st_case lib_run_by_pre_commit 1 --what "a pre-commit entry runs a lib/ file" \
    --has "run by pre-commit" -- sg_gate "$root"

root=$(sg_fresh librun_crate)
lib_py | sg_put "$root" lib/y.py
mkdir -p "$root/crates/c/src"
printf 'fn run() { Command::new("python3").arg(1); Command::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/y.py")); }\n' >"$root/crates/c/src/lib.rs"
sg_reindex "$root"
st_case lib_run_by_a_crate 1 --what "Command::new on the path runs a lib/ file" \
    --has "run by crates/c/src/lib.rs" -- sg_gate "$root"

root=$(sg_fresh librun_crate_read)
lib_py | sg_put "$root" lib/y.py
mkdir -p "$root/crates/c/src"
printf 'const S: &str = include_str!("../../../scripts/lib/y.py");\n' >"$root/crates/c/src/lib.rs"
printf 'import y\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case lib_run_crate_reads_and_imports_are_not_runs 0 --what "include_str! and an import read the file" \
    --has "check-scripts: ok (" -- sg_gate "$root"

n=0
for form in 'Command::new("python3").arg(PATH)' 'Command::new("python").arg("-u").arg(PATH)' 'Command::new("python3").args([PATH, "x"])' \
    'Command::new("bash").arg(PATH)' 'Command::new("sh").arg(PATH).output()'; do
    n=$((n + 1))
    root=$(sg_fresh "librun_crate_interpreter_$n")
    lib_py | sg_put "$root" lib/y.py
    mkdir -p "$root/crates/c/src"
    path='concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/y.py")'
    printf 'fn run() { let _ = %s; }\n' "${form//PATH/$path}" >"$root/crates/c/src/lib.rs"
    sg_reindex "$root"
    st_case "lib_run_by_a_crate_through_an_interpreter_$n" 1 --what "$form: the file is run" \
        --has "lib-run: lib/y.py:1:" --has "run by crates/c/src/lib.rs" -- sg_gate "$root"
done

n=0
for form in 'Command::new("cargo").arg(PATH)' 'Command::new("python3").env("X", PATH)' 'Command::new("python3"); let _ = include_str!(PATH)' \
    'Command::new("python3").arg("-c").arg("x"); let _ = (PATH)'; do
    n=$((n + 1))
    root=$(sg_fresh "librun_crate_not_interpreter_$n")
    lib_py | sg_put "$root" lib/y.py
    mkdir -p "$root/crates/c/src"
    path='concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/y.py")'
    printf 'fn run() { let _ = %s; }\n' "${form//PATH/$path}" >"$root/crates/c/src/lib.rs"
    printf 'import y\n' >>"$root/scripts/report/tool.py"
    sg_reindex "$root"
    st_case "lib_run_crate_not_a_run_$n" 0 --what "$form: the file is only named" \
        --has "check-scripts: ok (" -- sg_gate "$root"
done

root=$(sg_fresh librun_sourced)
sg_lib_sh '# Module of: check/ok.sh' | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case lib_run_sourcing_is_no_run 0 --what "sourcing a lib/ file is what it is for" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

# ---- module-of -----------------------------------------------------------

# mod_py <named file>: a planted Python module whose third line is `Module of:`.
mod_py() {
    printf '%s\n' '"""A planted module.' '' "Module of: $1" '"""'
}

root=$(sg_fresh modof_missing)
mod_py report/none.py | sg_put "$root" report/m.py
sg_reindex "$root"
st_case module_of_names_no_file 1 --what "Module of: names a file that does not exist" \
    --has "module-of: report/m.py:3: names report/none.py, which is no file of the tree" -- sg_gate "$root"

root=$(sg_fresh modof_self)
mod_py report/m.py | sg_put "$root" report/m.py
sg_reindex "$root"
st_case module_of_names_itself 1 --what "Module of: names the file itself" \
    --has "module-of: report/m.py:3: \`Module of:\` names the file itself" -- sg_gate "$root"

root=$(sg_fresh modof_unreached)
mod_py report/tool.py | sg_put "$root" report/m.py
sg_reindex "$root"
st_case module_of_tool_does_not_import_it 1 --what "the named tool never imports the module" \
    --has "module-of: report/m.py:3: report/tool.py does not run, source or import it" -- sg_gate "$root"

root=$(sg_fresh modof_transitive)
printf '%s\n' '"""A planted module.' '' 'Module of: report/tool.py' '"""' 'import leaf' | sg_put "$root" report/mid.py
mod_py report/tool.py | sg_put "$root" report/leaf.py
printf 'import mid\n' >>"$root/scripts/report/tool.py"
sg_reindex "$root"
st_case module_of_through_another_module 0 --what "the tool imports mid, mid imports leaf: both are its modules" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh modof_directory)
mod_py report | sg_put "$root" report/m.py
sg_reindex "$root"
st_case module_of_names_a_directory 1 --what "Module of: names a directory" \
    --has "module-of: report/m.py:3: names report, which is no file of the tree" -- sg_gate "$root"

# ---- shebang, exec-bit ---------------------------------------------------

root=$(sg_fresh shebang_py_bash)
printf '%s\n' '#!/usr/bin/env bash' '"""A planted Python entry point with a shell shebang.' '' \
    'Usage:  python3 scripts/report/tool.py' 'Exit:   0 always' \
    'Not a mode of: report/tool.py — a planted entry point' 'Caller: by hand' '"""' 'raise SystemExit(0)' |
    sg_put "$root" report/y.py
sg_reindex "$root"
st_case shebang_python_with_bash 1 --what "a .py entry point with a bash shebang" \
    --has "shebang: report/y.py:1: entry point has #!/usr/bin/env bash, expected #!/usr/bin/env python3" -- sg_gate "$root"

root=$(sg_fresh shebang_sh_bin_bash)
sg_valid_sh 'exit 0' | sed '1s|.*|#!/bin/bash|' | sg_put "$root" check/y.sh
sg_reindex "$root"
st_case shebang_sh_not_env 1 --what "#!/bin/bash is not #!/usr/bin/env bash" \
    --has "shebang: check/y.sh:1: entry point has #!/bin/bash, expected #!/usr/bin/env bash" -- sg_gate "$root"

root=$(sg_fresh shebang_sh_python)
sg_valid_sh 'exit 0' | sed '1s|.*|#!/usr/bin/env python3|' | sg_put "$root" check/y.sh
sg_reindex "$root"
st_case shebang_sh_with_python 1 --what "a .sh entry point with a python shebang" \
    --has "shebang: check/y.sh:1: entry point has #!/usr/bin/env python3, expected #!/usr/bin/env bash" -- sg_gate "$root"

root=$(sg_fresh shebang_none)
printf '%s\n' '# A planted script with no shebang that a recipe runs.' 'exit 0' | sg_put "$root" check/noshebang.sh
printf 'check-noshebang:\n\t@bash scripts/check/noshebang.sh\n' >>"$root/Makefile"
sg_reindex "$root"
st_case shebang_missing_on_a_run_file 1 --what "an entry point a recipe runs, with no shebang" \
    --has "shebang: check/noshebang.sh:1: entry point has no shebang, expected #!/usr/bin/env bash" -- sg_gate "$root"

root=$(sg_fresh shebang_lib)
printf '%s\n' '#!/usr/bin/env bash' '# A planted lib file with a shebang.' 'liby() { :; }' | sg_put "$root" lib/y.sh
printf '%s\n' "$SRC_Y" >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case shebang_on_a_lib_file 1 --what "a lib/ file carries a shebang" \
    --has "shebang: lib/y.sh:1: a module or lib/ file carries a shebang" -- sg_gate "$root"

root=$(sg_fresh exec_untracked)
sg_valid_sh 'exit 0' | sg_put "$root" check/y.sh
chmod -x "$root/scripts/check/y.sh"
sg_reindex "$root"
st_case exec_bit_missing_on_an_untracked_entry 1 --what "an untracked entry point is judged by its file mode" \
    --has "exec-bit: check/y.sh:1: entry point without the exec bit" -- sg_gate "$root"

root=$(sg_fresh exec_tracked)
sg_valid_sh 'exit 0' | sg_put "$root" check/y.sh
chmod -x "$root/scripts/check/y.sh"
sg_reindex "$root"
sg_commit_all "$root" "adds y"
st_case exec_bit_missing_on_a_tracked_entry 1 --what "a tracked entry point is judged by the index" \
    --has "exec-bit: check/y.sh:1: entry point without the exec bit" -- sg_gate "$root"
chmod +x "$root/scripts/check/y.sh"
st_case exec_bit_index_wins_over_the_file 1 --what "the file system says executable, the index does not: the index wins" \
    --has "exec-bit: check/y.sh:1:" -- sg_gate "$root"
git -C "$root" update-index --chmod=+x scripts/check/y.sh
st_case exec_bit_set_in_the_index 0 --what "with the bit in the index the entry point passes" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh exec_module)
chmod +x "$root/scripts/report/tool_mod.py"
git -C "$root" update-index --chmod=+x scripts/report/tool_mod.py
st_case exec_bit_on_a_module 1 --what "a module with the exec bit" \
    --has "exec-bit: report/tool_mod.py:1: a module or lib/ file with the exec bit" -- sg_gate "$root"

root=$(sg_fresh exec_lib)
chmod +x "$root/scripts/lib/helper.sh"
git -C "$root" update-index --chmod=+x scripts/lib/helper.sh
st_case exec_bit_on_a_lib_file 1 --what "a lib/ file with the exec bit" \
    --has "exec-bit: lib/helper.sh:1: a module or lib/ file with the exec bit" -- sg_gate "$root"

# ---- orphan --------------------------------------------------------------

root=$(sg_fresh orphan_data)
printf '%s\n' '# A planted data file nothing reads.' >"$root/scripts/report/data.txt"
sg_reindex "$root"
st_case orphan_ignores_data_files 0 --what "a data file nothing reads is not an orphan" \
    --has "check-scripts: ok (5 files" -- sg_gate "$root"

root=$(sg_fresh orphan_sh)
printf '%s\n' '# A planted shell file with no shebang that nothing uses.' 'exit 0' | sg_put "$root" check/o.sh
sg_reindex "$root"
st_case orphan_shell_file 1 --what "no shebang, no caller, no Module of:" \
    --has "orphan: check/o.sh:1: no shebang and nothing runs, sources or imports it" -- sg_gate "$root"

# ---- dead-path -----------------------------------------------------------

root=$(sg_fresh dead_path)
printf '%s\n' '# see scripts/check/gone.sh' >>"$root/scripts/check/ok.sh"
sg_make_ci "$root" 'check-gone:\n\t@echo scripts/check/gone.sh\n'
sg_reindex "$root"
ok_line=$(grep -c '' "$root/scripts/check/ok.sh")
make_line=$(grep -c '' "$root/Makefile")
st_case dead_path_in_a_script_and_the_makefile 1 --what "scripts/check/gone.sh named in a script and in a recipe" \
    --has "dead-path: check/ok.sh:${ok_line}: scripts/check/gone.sh names no file" \
    --has "dead-path: ../Makefile:${make_line}: scripts/check/gone.sh names no file" -- sg_gate "$root"

root=$(sg_fresh dead_path_alive)
{
    printf '%s\n' '# scripts/check/ok.sh is a file, scripts/check a directory, scripts/check/ok.sh. ends a sentence'
    printf '%s\n' '# patterns: scripts/nope/*.sh scripts/nope/<name>.sh scripts/nope/{a,b}.sh scripts/nope/$X scripts/nope/file(_x)?\.sh'
    printf '%s\n' '# more patterns: scripts/nope/file[ab].sh scripts/nope/file?.sh scripts/nope/file+.sh scripts/nope/a|b scripts/nope/file\.sh'
    printf '%s\n' '# the bare directory: scripts/ and scripts/lib/'
} >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case dead_path_names_that_exist_or_are_patterns 0 --what "files, directories, trailing dots and patterns are not dead" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

root=$(sg_fresh dead_path_ignored)
printf '%s\n' 'scripts/check/out.json' >"$root/.gitignore"
printf '%s\n' '# writes scripts/check/out.json' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case dead_path_git_ignored_is_a_produced_file 0 --what "a path git ignores is a produced file" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

root=$(sg_fresh dead_path_not_ignored)
printf '%s\n' '# writes scripts/check/out.json' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case dead_path_same_token_not_ignored 1 --what "the same token with no ignore rule is dead" \
    --has "dead-path: check/ok.sh:" --has "scripts/check/out.json names no file" -- sg_gate "$root"

root=$(sg_fresh dead_path_recall_test)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '# Caller: by hand' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
printf '%s\n' 'echo scripts/check/planted.sh' >>"$root/scripts/check/gate_selftest.sh"
sg_reindex "$root"
st_case dead_path_recall_test_plants_trees 0 --what "a recall test plants files: its tokens are not read" \
    --has "check-scripts: ok (6 files" -- sg_gate "$root"

root=$(sg_fresh dead_path_docs)
mkdir -p "$root/docs" "$root/.github/workflows"
printf 'one\nsee scripts/check/gone.sh\n' >"$root/docs/A.md"
printf 'see scripts/check/gone.sh\n' >"$root/CLAUDE.md"
printf 'see scripts/check/gone.sh\n' >"$root/CONTRIBUTING.md"
printf 'see scripts/check/gone.sh\n' >"$root/README.md"
printf 'see scripts/check/gone.sh\n' >"$root/CHANGELOG.md"
printf 'jobs:\n  g:\n    steps:\n      - run: echo done # scripts/check/gone.sh\n' >"$root/.github/workflows/ci.yml"
printf 'repos: []  # scripts/check/gone.sh\n' >"$root/.pre-commit-config.yaml"
sg_reindex "$root"
st_case dead_path_in_guides_docs_and_configs 1 --what "a doc, the guides, a workflow and the pre-commit config, but not the changelog" \
    --has "dead-path: ../docs/A.md:2: scripts/check/gone.sh" --has "dead-path: ../CLAUDE.md:1:" \
    --has "dead-path: ../CONTRIBUTING.md:1:" --has "dead-path: ../README.md:1:" \
    --has "dead-path: ../.github/workflows/ci.yml:4:" --has "dead-path: ../.pre-commit-config.yaml:1:" \
    --lacks "CHANGELOG" -- sg_gate "$root"

root=$(sg_fresh dead_path_crate)
mkdir -p "$root/crates/c/src"
printf 'const S: &str = include_str!("../../../scripts/check/gone.sh");\n' >"$root/crates/c/src/lib.rs"
sg_reindex "$root"
st_case dead_path_in_a_crate_literal 1 --what "a crates/ string literal naming a file that is gone" \
    --has "dead-path: ../crates/c/src/lib.rs:1: scripts/check/gone.sh names no file" -- sg_gate "$root"

n=0
for spelling in './scripts/check/gone.sh' '$REPO_ROOT/scripts/check/gone.sh' '${REPO_ROOT}/scripts/check/gone.sh' \
    '$(CURDIR)/scripts/check/gone.sh' '"$REPO_ROOT"/scripts/check/gone.sh' '"${REPO_ROOT}"/scripts/check/gone.sh' \
    '${REPO_ROOT:-.}/scripts/check/gone.sh' '"${REPO_ROOT:-.}"/scripts/check/gone.sh' \
    '$(git rev-parse --show-toplevel)/scripts/check/gone.sh' '"$(git rev-parse --show-toplevel)"/scripts/check/gone.sh'; do
    n=$((n + 1))
    root=$(sg_fresh "deadroot_$n")
    printf '%s\n' "# runs $spelling" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    line=$(grep -c '' "$root/scripts/check/ok.sh")
    st_case "dead_path_behind_root_expression_$n" 1 --what "scripts/check/gone.sh behind the repository-root expression $spelling" \
        --has "dead-path: check/ok.sh:${line}: scripts/check/gone.sh names no file" -- sg_gate "$root"
done

for spelling in '../scripts/check/gone.sh' '/opt/other/scripts/check/gone.sh' '$HOME/other/scripts/check/gone.sh' \
    '$REPO_ROOT/../scripts/check/gone.sh' 'my-scripts/check/gone.sh' 'x/$REPO_ROOT/scripts/check/gone.sh'; do
    n=$((n + 1))
    root=$(sg_fresh "deadother_$n")
    printf '%s\n' "# runs $spelling" >>"$root/scripts/check/ok.sh"
    sg_reindex "$root"
    st_case "dead_path_skips_other_tree_$n" 0 --what "$spelling is another tree or another name: not read" \
        --has "check-scripts: ok (4 files" -- sg_gate "$root"
done

root=$(sg_fresh dead_path_default_expansion)
printf '%s\n' 'TOOL="${1:-$REPO_ROOT/scripts/check/gone.sh}"' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
line=$(grep -c '' "$root/scripts/check/ok.sh")
st_case dead_path_in_a_parameter_default 1 --what "the default of \${1:-\$REPO_ROOT/scripts/...} is a path" \
    --has "dead-path: check/ok.sh:${line}: scripts/check/gone.sh names no file" -- sg_gate "$root"

root=$(sg_fresh dead_path_root_prefix_docs)
mkdir -p "$root/docs"
printf 'run `./scripts/check/gone.sh` first\nrun `../scripts/check/gone.sh` too\n' >"$root/docs/B.md"
sg_reindex "$root"
st_case dead_path_in_a_doc_behind_dot_slash 1 --what "./scripts/ in a doc is read; ../scripts/ on the next line is not" \
    --has "dead-path: ../docs/B.md:1: scripts/check/gone.sh names no file" --lacks "docs/B.md:2" -- sg_gate "$root"

root=$(sg_fresh dead_path_makefile_curdir)
sg_make_ci "$root" 'gone-target:\n\tbash $(CURDIR)/scripts/check/gone.sh\n'
sg_reindex "$root"
make_line=$(grep -c '' "$root/Makefile")
st_case dead_path_in_the_makefile_behind_curdir 1 --what "\$(CURDIR)/scripts/ in a recipe" \
    --has "dead-path: ../Makefile:${make_line}: scripts/check/gone.sh names no file" -- sg_gate "$root"

# ---- pins of the new rules ----------------------------------------------

root=$(sg_fresh pinset_recall_holds)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n'
sg_pins_file 'recall-missing check/gate.sh default' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_set_recall_missing_holds 0 --what "a pinned mode passes and the rule is counted" \
    --has "pinned: recall-missing 1" -- sg_gate "$root"

root=$(sg_fresh pinset_recall_new_mode)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n'
sg_pins_file 'recall-missing check/gate.sh default' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_set_recall_missing_new_mode 1 --what "a mode outside the pinned set fails the rule" \
    --has "recall-missing: check/gate.sh:1: mode --lib" --lacks "mode default" -- sg_gate "$root"

root=$(sg_fresh pinset_recall_stale_mode)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_recall_sh '#' 'check/gate.sh' | sg_put "$root" check/gate_selftest.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n\t@bash scripts/check/gate.sh --lib\n\t@bash scripts/check/gate_selftest.sh\n'
sg_pins_file 'recall-missing check/gate.sh default,--lib' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_set_recall_missing_stale_mode 1 --what "a pinned mode a recall test now covers is stale while the other pinned mode holds" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: recall-missing check/gate.sh: mode default is fixed or gone" \
    --lacks "mode --lib" -- sg_gate "$root"

root=$(sg_fresh pinset_lib_run)
lib_py | sg_put "$root" lib/y.py
sg_make_ci "$root" 'liba:\n\t@python3 scripts/lib/y.py\n'
sg_pins_file 'lib-run lib/y.py make:liba' 'lib-consumers lib/y.py' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_set_lib_run_holds 0 --what "a pinned running site passes" \
    --has "pinned: lib-consumers 1, lib-run 1" -- sg_gate "$root"
printf 'libb:\n\t@python3 scripts/lib/y.py\n' >>"$root/Makefile"
sg_reindex "$root"
st_case pin_set_lib_run_new_site 1 --what "a second site outside the pin fails" \
    --has "lib-run: lib/y.py:1:" --has "run by make:libb" --lacks "run by make:liba" -- sg_gate "$root"

root=$(sg_fresh pinset_dead_path)
printf '%s\n' '# see scripts/check/gone.sh' >>"$root/scripts/check/ok.sh"
sg_pins_file 'dead-path check/ok.sh check/gone.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_set_dead_path_holds 0 --what "a pinned dead token passes" \
    --has "pinned: dead-path 1" -- sg_gate "$root"
printf '%s\n' '# and scripts/check/other.sh' >>"$root/scripts/check/ok.sh"
sg_reindex "$root"
st_case pin_set_dead_path_new_token 1 --what "a second dead token in the pinned file fails" \
    --has "scripts/check/other.sh names no file" --lacks "gone.sh names" -- sg_gate "$root"

root=$(sg_fresh pinset_dead_path_stale)
sg_pins_file 'dead-path check/ok.sh check/gone.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_set_dead_path_stale_token 1 --what "a pinned token no longer present is stale" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: dead-path check/ok.sh: no longer violated" -- sg_gate "$root"

root=$(sg_fresh pinset_file_level)
sg_valid_sh 'exit 0' | sg_put "$root" check/y.sh
chmod -x "$root/scripts/check/y.sh"
sg_pins_file 'exec-bit check/y.sh' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_exec_bit_holds 0 --what "a pinned missing exec bit passes" \
    --has "pinned: exec-bit 1" -- sg_gate "$root"
chmod +x "$root/scripts/check/y.sh"
st_case pin_exec_bit_stale 1 --what "and is stale once the bit is set" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: exec-bit check/y.sh: no longer violated" -- sg_gate "$root"

root=$(sg_fresh pinset_base_growth)
sg_nocaller_sh "$GATE_BODY" | sg_put "$root" check/gate.sh
sg_make_ci "$root" 'ci:\n\t@bash scripts/check/gate.sh\n'
sg_pins_file 'recall-missing check/gate.sh default' >"$root/$SG_PINS_REL"
sg_reindex "$root"
sg_commit_all "$root" "pin gate.sh"
sg_make_ci "$root" '\t@bash scripts/check/gate.sh --lib\n'
sg_pins_file 'recall-missing check/gate.sh default,--lib' >"$root/$SG_PINS_REL"
sg_reindex "$root"
st_case pin_new_set_member 1 --what "a mode added to a pin the base already had" \
    --has "pin-new: report/scripts_gate_pins.txt:4: recall-missing check/gate.sh: mode --lib is not pinned at the base" -- sg_gate "$root"

# ---- scope ---------------------------------------------------------------

root=$(sg_fresh scope_fixtures)
mkdir -p "$root/scripts/fixtures/gate"
printf 'no header at all\n' >"$root/scripts/fixtures/gate/nohdr.sh"
st_case scope_fixtures_excluded 0 --what "a headerless file under fixtures/ is not scanned" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

root=$(sg_fresh scope_ignored)
printf 'scripts/check/scratch.sh\n' >"$root/.gitignore"
printf 'no header at all\n' >"$root/scripts/check/scratch.sh"
st_case scope_git_ignored_excluded 0 --what "a git-ignored file is not in scope" \
    --has "check-scripts: ok (4 files" -- sg_gate "$root"

root=$(sg_fresh scope_untracked)
printf 'no header at all\n' >"$root/scripts/check/scratch.sh"
sg_reindex "$root"
st_case scope_untracked_included 1 --what "the same file, not ignored, is scanned" \
    --has "purpose: check/scratch.sh:" -- sg_gate "$root"

# ---- unavailable ---------------------------------------------------------

root="$ST_WORK/ceiling/nogit"
mkdir -p "$root/scripts/check"
sg_valid_sh 'exit 0' >"$root/scripts/check/x.sh"
nogit_gate() {
    GIT_CEILING_DIRECTORIES="$ST_WORK" sg_gate "$root"
}
st_case unavailable_outside_git 2 --what "outside a git work tree the population is unknown" \
    --has "check-scripts: unavailable (" --has "is not inside a git work tree" -- nogit_gate

root=$(sg_fresh no_scripts_dir)
rm -rf "$root/scripts"
st_case unavailable_no_files 2 --what "a tree with no file under scripts/ is not a pass" \
    --has "check-scripts: unavailable (no file under scripts/" -- sg_gate "$root"

st_case unavailable_missing_root 2 --what "a root that does not exist" \
    --has "check-scripts: unavailable (" -- sg_gate "$ST_WORK/does-not-exist"

root=$(sg_fresh base_unknown)
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
    sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 clean' \
        '# Caller: by hand' '#' 'exit 0' | sg_put "$1" "check/$2.sh"
    sg_reindex "$1"
    sg_commit_all "$1" "adds $2"
}

root=$(sg_fresh auto_ahead)
git -C "$root" update-ref refs/remotes/origin/main HEAD
plant_unnamed "$root" y
st_case base_is_merge_base_with_origin_main 1 --what "a branch ahead of origin/main: files added since the merge-base are new" \
    --has "check-scripts: base origin/main (" --has "not-a-mode: check/y.sh:1:" -- auto "$root"

root=$(sg_fresh auto_tip)
plant_unnamed "$root" y
git -C "$root" update-ref refs/remotes/origin/main HEAD
st_case base_at_origin_main_is_the_parent 1 --what "HEAD is origin/main: the base is its parent, so the last commit's files are new" \
    --has "check-scripts: base HEAD^ (" --has "not-a-mode: check/y.sh:1:" -- auto "$root"

root=$(sg_fresh auto_tip_old)
plant_unnamed "$root" y
plant_unnamed "$root" z
python3 - "$root/scripts/check/z.sh" <<'EOF'
import sys

path = sys.argv[1]
text = open(path).read().replace("# Caller: by hand\n#\n", "# Caller: by hand\n# Not a mode of: check/ok.sh — a planted entry point\n")
open(path, "w").write(text)
EOF
sg_reindex "$root"
sg_commit_all "$root" "names z"
git -C "$root" update-ref refs/remotes/origin/main HEAD
st_case base_parent_leaves_older_files_alone 0 --what "a file committed before the parent is not new" \
    --has "check-scripts: base HEAD^ (" --lacks "not-a-mode: check/y.sh" -- auto "$root"

# ---- the case runner under strict mode ------------------------------------

# runner_child <body file>: run the body in a fresh bash under set -euo pipefail with the case
# runner sourced and initialised, so a defect that only strict mode turns into a silent abort shows.
runner_child() {
    {
        printf '%s\n' 'set -euo pipefail' ". \"$REPO_ROOT/scripts/lib/selftest_support.sh\"" 'st_init child'
        cat "$1"
    } >"$1.sh"
    bash "$1.sh"
}

body="$ST_WORK/runner_big_output"
cat >"$body" <<'EOF'
st_case big 0 -- bash -c 'yes xxxxxxxxxx | head -n 20000; exit 1'
st_done
EOF
st_case runner_failing_case_with_big_output 1 --what "a failing case whose output outgrows the pipe still prints its FAIL line, its output head and the summary" \
    --has "  FAIL big" --has "    | xxxxxxxxxx" --has "child: 0 passed, 1 failed" -- runner_child "$body"

body="$ST_WORK/runner_row_missing"
cat >"$body" <<'EOF'
mkdir -p "$ST_WORK/r/scripts"
printf '| `a.sh` | by hand | x |\n' >"$ST_WORK/r/scripts/INDEX.md"
line=$(sg_row_line "$ST_WORK/r" no/such.sh)
echo "reached $line"
EOF
st_case runner_row_missing 2 --what "sg_row_line on a row that is not there ends the harness with exit 2 and names the row" \
    --has "scripts/INDEX.md has no row for no/such.sh" --lacks "reached" -- runner_child "$body"

body="$ST_WORK/runner_row_present"
cat >"$body" <<'EOF'
mkdir -p "$ST_WORK/r/scripts"
printf '# preamble\n| `a.sh` | by hand | x |\n' >"$ST_WORK/r/scripts/INDEX.md"
line=$(sg_row_line "$ST_WORK/r" a.sh)
echo "row at line $line"
st_finish_early
EOF
st_case runner_row_present 0 --what "sg_row_line on a row that is there prints its line number" \
    --is "row at line 2" -- runner_child "$body"

body="$ST_WORK/runner_skip"
cat >"$body" <<'EOF'
st_case held 0 -- true
st_skip needs_root "this user can read a mode-000 file"
st_done
EOF
st_case runner_skip_is_in_the_summary 0 --what "a skipped case is neither passed nor failed, and the summary line says how many were skipped" \
    --has "  skip needs_root - this user can read a mode-000 file" --has "child: 1 passed, 0 failed, 1 skipped" -- runner_child "$body"

body="$ST_WORK/runner_no_skip"
cat >"$body" <<'EOF'
st_case held 0 -- true
st_done
EOF
st_case runner_summary_without_a_skip_is_unchanged 0 --what "with no skipped case the summary line is the two counts and nothing more" \
    --is "  ok   held
child: 1 passed, 0 failed" -- runner_child "$body"

body="$ST_WORK/runner_only_skips"
cat >"$body" <<'EOF'
st_skip only "cannot run here"
st_done
EOF
st_case runner_a_run_of_skips_only_is_no_case_ran 1 --what "a run in which every case was skipped checked nothing, and is not a clean run" \
    --has "child: 0 passed, 0 failed, 1 skipped" --has "child: no case ran" -- runner_child "$body"

body="$ST_WORK/runner_unbound"
cat >"$body" <<'EOF'
st_case held 0 -- true
echo "$NO_SUCH_VARIABLE"
st_done
EOF
st_case runner_unbound_expansion_is_exit_1 1 --what "a harness that dies on an unbound variable before its summary exits 1 and says so, where bash 3.2 would exit 0" \
    --has "child: the harness ended before its summary" -- runner_child "$body"

body="$ST_WORK/runner_exit_before_done"
cat >"$body" <<'EOF'
st_case held 0 -- true
exit 0
EOF
st_case runner_exit_0_before_the_summary_is_exit_1 1 --what "an exit 0 that skips st_done is a harness that ended before its summary" \
    --has "child: the harness ended before its summary" -- runner_child "$body"

body="$ST_WORK/runner_finish_early"
cat >"$body" <<'EOF'
st_finish_early
st_case never 0 -- false
st_done
EOF
st_case runner_finish_early_is_exit_0 0 --what "st_finish_early ends the harness with exit 0 and no complaint" \
    --lacks "ended before its summary" -- runner_child "$body"

body="$ST_WORK/runner_die"
cat >"$body" <<'EOF'
st_die "the fixture is broken"
EOF
st_case runner_die_stays_exit_2 2 --what "st_die ends the harness with exit 2, not the exit 1 of a harness without a summary" \
    --has "child: the fixture is broken" --lacks "ended before its summary" -- runner_child "$body"

# ---- the real tree -------------------------------------------------------

# The pin counts are literals, so a chunk that shrinks a pin lowers them in the same commit
# and growth shows in the diff as a raised literal. Beside the line counts, the member total of
# every set rule and the figure sum of every measure rule are literals too: a figure raised
# inside a pin, or a name added to one, changes no line count.
# pin_totals: those totals, read from the checked-in pin file.
pin_totals() {
    awk '
/^#/ || NF == 0 { next }
{
    n = split($3, parts, ",")
    if ($1 ~ /^(exit-undeclared|recall-missing|silenced|lib-run|dead-path|name|lib-shadow|env-knob|unguarded-array)$/) {
        members[$1] += n
    } else if ($1 ~ /^(size|heredoc|twin|binary-literal)$/) {
        for (i = 1; i <= n; i++) sum[$1] += parts[i]
    }
}
END {
    split("exit-undeclared recall-missing silenced lib-run dead-path name lib-shadow env-knob unguarded-array", sets, " ")
    split("size heredoc twin binary-literal", measures, " ")
    line = ""
    for (i = 1; i <= 9; i++) line = line (i > 1 ? ", " : "") sets[i] " " members[sets[i]] + 0
    print "members: " line
    line = ""
    for (i = 1; i <= 4; i++) line = line (i > 1 ? ", " : "") measures[i] " " sum[measures[i]] + 0
    print "figures: " line
}' "$REPO_ROOT/scripts/report/scripts_gate_pins.txt"
}
st_case real_tree_pin_totals 0 --what "the checked-in pin file holds exactly these member totals and figure sums" \
    --is "$(printf '%s\n%s' 'members: exit-undeclared 88, recall-missing 7, silenced 3, lib-run 29, dead-path 1, name 10, lib-shadow 3, env-knob 87, unguarded-array 32' 'figures: size 14673, heredoc 1278, twin 1234, binary-literal 43')" -- pin_totals
st_case real_tree 0 --what "the checked-in tree passes with exactly these pins" \
    --has "pinned: class 59, purpose 6, usage 54, exit-undeclared 47, caller 11, recall-missing 7, silenced 2, lib-consumers 10, lib-run 19, shebang 1, exec-bit 5, dead-path 1, set-mode 22, name 10, size 10, heredoc 14, twin 15, lib-shadow 3, binary-literal 22, env-knob 33, unguarded-array 12" \
    --has "363 pinned)" -- sg_gate "$REPO_ROOT"

st_done
