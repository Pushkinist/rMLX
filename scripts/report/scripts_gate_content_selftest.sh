#!/usr/bin/env bash
# Recall test for the per-file rules of debt_report.py --scripts: planted files, exit code, rule:line.
#
# Recall test for: report/debt_report.py --scripts --shellcheck
# Usage:  bash scripts/report/scripts_gate_content_selftest.sh [<debt_report.py under test>]
# Exit:   0 every case held; 1 a case did not; 2 a fixture could not be built
# Set:    -uo — a failing case must not stop the cases after it
#
# The rules that read one file: set-mode, py-floor, name, size, heredoc, twin,
# lib-shadow, binary-literal, env-knob, and the advisory shellcheck mode. Each
# case plants one file in a copy of the base tree that lib/selftest_support.sh
# builds, and runs the gate with `--base HEAD`. Planted shell comes from
# heredocs, which the gate does not read as this file's own code.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOOL="${1:-$REPO_ROOT/scripts/report/debt_report.py}"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"
st_init scripts-gate-content-selftest
sg_init

# sh_entry <root> <rel> <set line> [<extra header line>]: an entry point whose body, on stdin,
# follows the set line and a `:` line, so the first body line is line 10.
sh_entry() {
    {
        printf '%s\n' '#!/usr/bin/env bash' '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
            '# Exit:   0 clean; 1 failed'
        [ -z "${4:-}" ] || printf '%s\n' "$4"
        printf '%s\n' '# Caller: by hand' "$SG_NAM" "$3" ':'
        cat
    } | sg_put "$1" "$2"
}

# py_entry <root> <rel>: a Python entry point whose body, on stdin, follows the docstring.
py_entry() {
    {
        printf '%s\n' '#!/usr/bin/env python3' '"""A planted Python entry point.' '' 'Usage:  python3 scripts/report/y.py' \
            'Exit:   0 always' 'Not a mode of: report/tool.py — a planted entry point' 'Caller: by hand' '"""'
        cat
    } | sg_put "$1" "$2"
}

# words <n> <prefix>: n distinct lines. The digit fold would merge numbered ones.
words() {
    python3 -c '
import itertools, string, sys

names = ("".join(p) for p in itertools.product(string.ascii_lowercase, repeat=3))
for name in itertools.islice(names, int(sys.argv[1])):
    print(sys.argv[2] + name)
' "$1" "$2"
}

# padded <lines>: a valid entry point of exactly that many lines, on stdout.
padded() {
    sg_valid_sh ':'
    /usr/bin/yes ': pad' | head -n "$(($1 - 9))"
}

# verdict <name> <want exit> <what> <root> [<check>...]: regenerate the INDEX.md of <root>, so the case
# isolates the rule it plants, then hold the gate over it to the checks of st_case.
verdict() {
    local name=$1 want=$2 what=$3 root=$4
    shift 4
    sg_reindex "$root"
    st_case "$name" "$want" --what "$what" "$@" -- sg_gate "$root"
}

# ---- set-mode ------------------------------------------------------------

for form in 'set -euo pipefail' 'set -Eeuo pipefail' 'set -eu -o pipefail' 'set -e -u -o pipefail'; do
    root=$(sg_fresh "setmode_ok_$(printf '%s' "$form" | tr -c 'a-zA-Z' _)")
    sh_entry "$root" check/y.sh "$form" <<'EOF'
:
EOF
    sg_reindex "$root"
    st_case "set_mode_accepts_${form// /_}" 0 --what "strict mode as: $form" \
        --has "check-scripts: ok (5 files" -- sg_gate "$root"
done

root=$(sg_fresh setmode_none)
sh_entry "$root" check/y.sh ':' <<'EOF'
:
EOF
verdict set_mode_missing 1 "an entry point with no set at all" "$root" \
    --has 'set-mode: check/y.sh:1: no `set -euo pipefail`'

root=$(sg_fresh setmode_no_pipefail)
sh_entry "$root" check/y.sh 'set -eu' <<'EOF'
:
EOF
verdict set_mode_without_pipefail 1 "set -eu: a failing pipeline stage is lost" "$root" \
    --has 'set-mode: check/y.sh:1: no `set -euo pipefail`'

root=$(sg_fresh setmode_no_u)
sh_entry "$root" check/y.sh 'set -eo pipefail' <<'EOF'
:
EOF
verdict set_mode_without_nounset 1 "set -eo pipefail: an unset variable is silent" "$root" \
    --has 'set-mode: check/y.sh:1: no `set -euo pipefail`'

root=$(sg_fresh setmode_in_string)
sh_entry "$root" check/y.sh ':' <<'EOF'
echo "set -euo pipefail"   # set -euo pipefail
EOF
verdict set_mode_in_a_string_or_comment 1 "the words in a string or a comment set nothing" "$root" \
    --has 'set-mode: check/y.sh:1: no `set -euo pipefail`'

root=$(sg_fresh setmode_keyword)
sh_entry "$root" check/y.sh ':' <<'EOF'
if true; then set -euo pipefail; fi
EOF
verdict set_mode_after_a_keyword 0 "a set command after then counts" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh setmode_uo)
sh_entry "$root" check/y.sh 'set -uo pipefail' <<'EOF'
:
EOF
verdict set_mode_uo_needs_a_reason 1 "set -uo pipefail with no Set: field" "$root" \
    --has 'set-mode: check/y.sh:1: `set -uo pipefail` without `Set: -uo — <reason>`'

root=$(sg_fresh setmode_uo_reason)
sh_entry "$root" check/y.sh 'set -uo pipefail' '# Set:    -uo — a failing case must not stop the cases after it' <<'EOF'
:
EOF
verdict set_mode_uo_with_reason 0 "the same with the reason in the header" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh setmode_uo_no_reason)
sh_entry "$root" check/y.sh 'set -uo pipefail' '# Set:    -uo' <<'EOF'
:
EOF
verdict set_mode_uo_empty_reason 1 "a Set: field that gives no reason exempts nothing" "$root" \
    --has 'set-mode: check/y.sh:1: `set -uo pipefail` without'

root=$(sg_fresh setmode_stale)
sh_entry "$root" check/y.sh 'set -euo pipefail' '# Set:    -uo — a failing case must not stop the cases after it' <<'EOF'
:
EOF
verdict set_mode_declared_on_a_strict_file 1 "a Set: field on a file that runs under the strict mode is stale" "$root" \
    --has 'set-mode: check/y.sh:6: `Set:` is declared but the file runs under'

root=$(sg_fresh setmode_purpose)
printf '%s\n' '#!/usr/bin/env bash' '# CI gate: a planted entry point.' \
    '# Set:    -uo — a failing case must not stop the cases after it' '#' '# Usage:  bash scripts/check/ok.sh' \
    '# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" 'set -uo pipefail' ':' | sg_put "$root" check/y.sh
verdict set_mode_field_after_purpose_is_no_wrap 0 "a Set: field right after the purpose is not a wrapped purpose" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh setmode_pin)
sh_entry "$root" check/y.sh ':' <<'EOF'
:
EOF
sg_pins_file 'set-mode check/y.sh' >"$root/$SG_PINS_REL"
verdict set_mode_pin_holds 0 "a pinned entry point without the strict mode passes and is counted" "$root" \
    --has "pinned: set-mode 1"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
:
EOF
st_case set_mode_pin_stale 1 --what "and the pin is stale once the file has the strict mode" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: set-mode check/y.sh: no longer violated" -- sg_gate "$root"

# ---- py-floor ------------------------------------------------------------

root=$(sg_fresh pyfloor_match)
py_entry "$root" report/y.py <<'EOF'
def f(x):
    match x:
        case 1:
            return 2
    return 3
EOF
verdict py_floor_match_statement 1 "a match statement does not parse as Python 3.9" "$root" \
    --has "py-floor: report/y.py:" --has "does not parse as Python 3.9"

root=$(sg_fresh pyfloor_union)
py_entry "$root" report/y.py <<'EOF'
def f(x: int | None) -> int:
    return 1
EOF
verdict py_floor_union_in_argument 1 "int | None evaluated at def time" "$root" \
    --has "py-floor: report/y.py:9:" --has "needs \`from __future__ import annotations\`"

root=$(sg_fresh pyfloor_union_return)
py_entry "$root" report/y.py <<'EOF'
def f(x) -> dict[str, int] | None:
    return None
EOF
verdict py_floor_union_in_return 1 "a union in a return annotation" "$root" \
    --has "py-floor: report/y.py:9:"

root=$(sg_fresh pyfloor_union_module)
py_entry "$root" report/y.py <<'EOF'
LIMIT: int | None = None
EOF
verdict py_floor_union_in_module_variable 1 "a module-level variable annotation is evaluated" "$root" \
    --has "py-floor: report/y.py:9:"

root=$(sg_fresh pyfloor_future)
py_entry "$root" report/y.py <<'EOF'
from __future__ import annotations


def f(x: int | None) -> dict[str, int] | None:
    return None
EOF
verdict py_floor_future_import 0 "the same annotations with the future import" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh pyfloor_local)
py_entry "$root" report/y.py <<'EOF'
def f():
    y: int | None = None
    return y
EOF
verdict py_floor_local_annotation 0 "a variable annotation inside a function is not evaluated" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh pyfloor_string)
py_entry "$root" report/y.py <<'EOF'
def f(x: "int | None", y=1 | 2):
    return x
EOF
verdict py_floor_string_annotation_and_expression 0 "a quoted annotation and a | outside an annotation" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh pyfloor_kinds)
py_entry "$root" report/y.py <<'EOF'
def a(*, k: int | None = None): pass
def b(*args: int | None): pass
def c(**kw: int | None): pass
def d(p: int | None, /): pass
EOF
verdict py_floor_every_argument_kind 1 "keyword-only, *args, **kwargs and positional-only arguments" "$root" \
    --has "py-floor: report/y.py:9:" --has "py-floor: report/y.py:10:" \
    --has "py-floor: report/y.py:11:" --has "py-floor: report/y.py:12:"

root=$(sg_fresh pyfloor_other_ops)
py_entry "$root" report/y.py <<'EOF'
def e(x: Annotated[int, 1 + 2], y: "a" "b" = 1): pass
EOF
verdict py_floor_other_operators 0 "an annotation with + inside is no union" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh pyfloor_pin)
py_entry "$root" report/y.py <<'EOF'
LIMIT: int | None = None
EOF
sg_pins_file 'py-floor report/y.py' >"$root/$SG_PINS_REL"
verdict py_floor_cannot_be_pinned 1 "py-floor is not a pinnable rule" "$root" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: py-floor report/y.py: py-floor cannot be pinned" \
    --has "py-floor: report/y.py:9:"

# ---- name ----------------------------------------------------------------

root=$(sg_fresh name_hyphen)
sg_valid_sh ':' | sg_put "$root" check/x-y.sh
verdict name_hyphen_in_a_file_name 1 "a hyphen in a file name" "$root" \
    --has "name: check/x-y.sh:1: a hyphen in the file name"

ROOT_UP='$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)'
for spec in 'ROOT|ROOT' 'TOP|TOP' 'REPO_ROOT|' 'REPO_DIR|REPO_DIR'; do
    var=${spec%%|*}
    root=$(sg_fresh "name_root_$var")
    sh_entry "$root" check/y.sh 'set -euo pipefail' <<EOF
$var="$ROOT_UP"
EOF
    sg_reindex "$root"
    if [ -n "${spec#*|}" ]; then
        st_case "name_shell_root_variable_$var" 1 --what "$var holds the directory two levels above check/y.sh" \
            --has "name: check/y.sh:10: the repository root variable is $var; name it REPO_ROOT" -- sg_gate "$root"
    else
        st_case "name_shell_root_variable_$var" 0 --what "REPO_ROOT is the one name" \
            --has "check-scripts: ok (5 files" -- sg_gate "$root"
    fi
done

root=$(sg_fresh name_not_root)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW='$(cd "$(dirname "$0")/../.." && pwd)'
EOF
verdict name_directory_above_scripts_is_no_root 0 "the script's own directory, scripts/ itself and a quoted literal are no repository root" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh name_dollar0)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EOF
verdict name_dirname_of_dollar_zero 1 "dirname of \$0 is the file's own location too" "$root" \
    --has "name: check/y.sh:10: the repository root variable is ROOT"

root=$(sg_fresh name_toplevel)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
TOP="$(git rev-parse --show-toplevel)"
EOF
verdict name_work_tree_top_level 1 "the work tree's top level is a repository root" "$root" \
    --has "name: check/y.sh:10: the repository root variable is TOP"

for spec in 'ROOT = Path(__file__).resolve().parents[2]|ROOT' 'ROOT = Path(__file__).resolve().parent.parent.parent|ROOT' \
    'REPO_ROOT = Path(__file__).resolve().parents[2]|' 'HERE = Path(__file__).resolve().parent|' \
    'OTHER = Path(sys.argv[1]).resolve().parents[2]|' \
    'DOCS = Path(__file__).resolve().parents[2] / "docs"|' 'ROOT = Path(__file__).resolve().parents[1]|' \
    'ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))|ROOT'; do
    line=${spec%|*}
    var=${spec##*|}
    root=$(sg_fresh "name_py_${#line}_${line:0:4}")
    py_entry "$root" report/y.py <<EOF
import os
from pathlib import Path

$line
EOF
    sg_reindex "$root"
    if [ -n "$var" ]; then
        st_case "name_python_root_${#line}_$var" 1 --what "Python: $line" \
            --has "name: report/y.py:12: the repository root variable is $var" -- sg_gate "$root"
    else
        st_case "name_python_no_root_${#line}" 0 --what "Python: $line" \
            --has "check-scripts: ok (5 files" -- sg_gate "$root"
    fi
done

root=$(sg_fresh name_pin)
sg_valid_sh ':' | sg_put "$root" check/x-y.sh
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EOF
sg_pins_file 'name check/x-y.sh hyphen' 'name check/y.sh ROOT' >"$root/$SG_PINS_REL"
verdict name_pin_holds 0 "a pinned hyphen and a pinned root variable pass" "$root" \
    --has "pinned: name 2"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TOP="$(git rev-parse --show-toplevel)"
EOF
st_case name_pin_new_member 1 --what "a second root variable in a pinned file fails" \
    --has "name: check/y.sh:11: the repository root variable is TOP" --lacks "variable is ROOT" -- sg_gate "$root"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EOF
st_case name_pin_stale_member 1 --what "a pinned name that is fixed is stale while the other holds" \
    --has "pin-stale: report/scripts_gate_pins.txt:5: name check/y.sh: no longer violated" -- sg_gate "$root"

# ---- size ----------------------------------------------------------------

root=$(sg_fresh size_1001)
padded 1001 | sg_put "$root" check/y.sh
verdict size_over_the_limit 1 "1001 lines with no LOC-exempt" "$root" \
    --has "size: check/y.sh:1: 1001 lines, over 1000"

root=$(sg_fresh size_1000)
padded 1000 | sg_put "$root" check/y.sh
verdict size_at_the_limit 0 "1000 lines is within the limit" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh size_exempt)
/usr/bin/yes ': pad' | head -n 1200 | sh_entry "$root" check/y.sh 'set -euo pipefail' '# LOC-exempt: one planted case per line'
verdict size_with_loc_exempt 0 "a header LOC-exempt reason exempts the file" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh size_exempt_empty)
/usr/bin/yes ': pad' | head -n 1200 | sh_entry "$root" check/y.sh 'set -euo pipefail' '# LOC-exempt:'
verdict size_exempt_without_a_reason 1 "LOC-exempt with no reason exempts nothing" "$root" \
    --has "size: check/y.sh:1: 1210 lines"

for figure in '' ' many'; do
    root=$(sg_fresh "size_pin_figure_${#figure}")
    padded 1001 | sg_put "$root" check/y.sh
    sg_pins_file "size check/y.sh$figure" >"$root/$SG_PINS_REL"
    verdict "size_pin_without_a_figure_${#figure}" 1 "a measure pin needs a number: '$figure'" "$root" \
        --has "pin-figure: report/scripts_gate_pins.txt:4: size check/y.sh needs a figure" \
        --has "size: check/y.sh:1: 1001 lines"
done

root=$(sg_fresh size_pin)
padded 1003 | sg_put "$root" check/y.sh
sg_pins_file 'size check/y.sh 1003' >"$root/$SG_PINS_REL"
verdict size_pin_holds 0 "a pinned size passes and is counted" "$root" \
    --has "pinned: size 1"
sg_commit_all "$root" "pin y.sh"
padded 1006 | sg_put "$root" check/y.sh
st_case size_pin_grew 1 --what "a pinned file that grew fails size-pin" \
    --has "size-pin: check/y.sh:1: size figure is 1006, pinned 1003; fix the file, do not raise the pin" -- sg_gate "$root"
padded 1001 | sg_put "$root" check/y.sh
st_case size_pin_shrank 1 --what "one that shrank and is still over the limit asks for the pin to be lowered" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: size check/y.sh: figure is 1001, pinned 1003; lower the pin" -- sg_gate "$root"
padded 1000 | sg_put "$root" check/y.sh
st_case size_pin_back_within_the_limit 1 --what "one back within the limit is no longer violated" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: size check/y.sh: no longer violated" -- sg_gate "$root"
padded 1003 | sg_put "$root" check/y.sh
sg_pins_file 'size check/y.sh 1010' >"$root/$SG_PINS_REL"
st_case size_pin_raised_above_the_base 1 --what "a pin raised above the base's fails pin-new" \
    --has "pin-new: report/scripts_gate_pins.txt:4: size check/y.sh: figure 1010 is above the base's 1003" -- sg_gate "$root"

# ---- heredoc -------------------------------------------------------------

# py_heredoc <lines> [<tag>]: a python3 heredoc of that many body lines.
py_heredoc() {
    printf '%s\n' "python3 - <<'${2:-PY}'"
    /usr/bin/yes 'print(1)' | head -n "$1"
    printf '%s\n' "${2:-PY}"
}

root=$(sg_fresh heredoc_21)
{ sg_valid_sh ':'; py_heredoc 21; } | sg_put "$root" check/y.sh
verdict heredoc_over_the_limit 1 "a 21-line Python heredoc" "$root" \
    --has "heredoc: check/y.sh:10: a Python body of 21 lines, over 20"

root=$(sg_fresh heredoc_20)
{ sg_valid_sh ':'; py_heredoc 20; } | sg_put "$root" check/y.sh
verdict heredoc_at_the_limit 0 "a 20-line one is within it" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_dash_c)
{
    sg_valid_sh ':'
    printf '%s\n' "python3 -c '"
    /usr/bin/yes 'print(1)' | head -n 21
    printf '%s\n' "' arg"
} | sg_put "$root" check/y.sh
verdict heredoc_python_dash_c_over_the_limit 1 "a 21-line python3 -c string" "$root" \
    --has "heredoc: check/y.sh:10: a Python body of 21 lines"

root=$(sg_fresh heredoc_dash_c_20)
{
    sg_valid_sh ':'
    printf '%s\n' "python3 -c '"
    /usr/bin/yes 'print(1)' | head -n 20
    printf '%s\n' "' arg"
} | sg_put "$root" check/y.sh
verdict heredoc_python_dash_c_at_the_limit 0 "a 20-line one" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_dash_c_quoted)
{
    sg_valid_sh ':'
    printf '%s\n' 'python3 -c "'
    /usr/bin/yes 'print(\"x\")' | head -n 21
    printf '%s\n' '" arg'
} | sg_put "$root" check/y.sh
verdict heredoc_python_dash_c_escaped_quote 1 "a double-quoted -c string runs on past an escaped quote" "$root" \
    --has "heredoc: check/y.sh:10: a Python body of 21 lines"

root=$(sg_fresh heredoc_one_liner)
{ sg_valid_sh ':'; printf '%s\n' "python3 -c 'print(1)'" "python3 -c \"print(2)\""; } | sg_put "$root" check/y.sh
verdict heredoc_one_line_dash_c 0 "one-line python3 -c strings" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_other_command)
{
    sg_valid_sh ':'
    printf '%s\n' "cat <<'TXT' >/dev/null"
    /usr/bin/yes 'plain text' | head -n 40
    printf '%s\n' TXT
} | sg_put "$root" check/y.sh
verdict heredoc_of_another_command 0 "a 40-line heredoc that feeds cat" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_planted)
{
    sg_valid_sh ':'
    printf '%s\n' "cat <<'TXT' >/dev/null"
    py_heredoc 30 PY
    printf '%s\n' TXT
} | sg_put "$root" check/y.sh
verdict heredoc_inside_another_heredoc 0 "Python in the body of a cat heredoc is text, not a Python body" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_string)
{
    sg_valid_sh ':'
    printf '%s\n' "echo \"python3 - <<'PY'\""
    /usr/bin/yes ': after' | head -n 30
} | sg_put "$root" check/y.sh
verdict heredoc_opener_in_a_string 0 "an opener inside a string opens nothing" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_dash)
{
    sg_valid_sh ':'
    printf '%s\n' "python3 - <<-PY" $'\tprint(1)' $'\tPY'
    /usr/bin/yes ': after' | head -n 30
} | sg_put "$root" check/y.sh
verdict heredoc_tab_terminator 0 "a <<- heredoc ends at its tab-indented terminator" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_herestring)
{
    sg_valid_sh ':'
    printf '%s\n' "python3 -c 'pass' <<<word"
    /usr/bin/yes ': after' | head -n 30
} | sg_put "$root" check/y.sh
verdict heredoc_here_string 0 "a here-string opens no heredoc" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh heredoc_pin)
{ sg_valid_sh ':'; py_heredoc 25; py_heredoc 22; } | sg_put "$root" check/y.sh
sg_pins_file 'heredoc check/y.sh 25,22' >"$root/$SG_PINS_REL"
verdict heredoc_pin_holds 0 "two pinned over-limit bodies pass and are counted once per file" "$root" \
    --has "pinned: heredoc 1"
{ sg_valid_sh ':'; py_heredoc 26; py_heredoc 22; } | sg_put "$root" check/y.sh
st_case heredoc_pin_longest_grew 1 --what "the longest body grew" \
    --has "heredoc-pin: check/y.sh:10: heredoc figure is 26,22, pinned 25,22" -- sg_gate "$root"
{ sg_valid_sh ':'; py_heredoc 25; py_heredoc 22; py_heredoc 21; } | sg_put "$root" check/y.sh
st_case heredoc_pin_gained_a_body 1 --what "a third over-limit body" \
    --has "heredoc-pin: check/y.sh:10: heredoc figure is 25,22,21, pinned 25,22" -- sg_gate "$root"
{ sg_valid_sh ':'; py_heredoc 25; py_heredoc 20; } | sg_put "$root" check/y.sh
st_case heredoc_pin_lost_a_body 1 --what "a body that fell to the limit asks for the pin to be lowered" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: heredoc check/y.sh: figure is 25, pinned 25,22" -- sg_gate "$root"

# ---- twin ----------------------------------------------------------------

# twin_of <n>: an entry point with n significant lines: the set line and n - 1 distinct ones.
twin_of() {
    sg_valid_sh "$(words "$(($1 - 1))" ': w_')"
}

root=$(sg_fresh twin_copy)
twin_of 45 | sg_put "$root" check/a.sh
twin_of 45 | sg_put "$root" check/b.sh
verdict twin_copy_of_a_script 1 "a script copied under a new name" "$root" \
    --has "twin: check/a.sh:1: 45 matched significant lines with check/b.sh"

root=$(sg_fresh twin_small)
twin_of 15 | sg_put "$root" check/a.sh
twin_of 15 | sg_put "$root" check/b.sh
verdict twin_small_files_are_no_twins 0 "two identical files under 20 significant lines" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_half)
twin_of 30 | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 14 ': w_')"; words 15 ': v_'; } | sg_put "$root" check/b.sh
verdict twin_half_of_the_smaller 1 "half of a 20-line-plus file's significant lines match" "$root" \
    --has "twin: check/a.sh:1: " --has "matched significant lines with check/b.sh"

root=$(sg_fresh twin_just_under_half)
twin_of 30 | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 12 ': w_')"; words 18 ': v_'; } | sg_put "$root" check/b.sh
verdict twin_under_half 0 "fewer than half of the smaller file's lines match" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_forty)
twin_of 101 | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 39 ': w_')"; words 100 ': v_'; } | sg_put "$root" check/b.sh
verdict twin_forty_matched_lines 1 "40 shared lines, under half of the smaller file, are a twin" "$root" \
    --has "twin: check/a.sh:1: " --has "with check/b.sh"

root=$(sg_fresh twin_thirty_nine)
twin_of 101 | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 38 ': w_')"; words 100 ': v_'; } | sg_put "$root" check/b.sh
verdict twin_below_forty_matched_lines 0 "39 shared lines, under half of the smaller file" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_reversed)
twin_of 61 | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 60 ': w_' | sed '1!G;h;$!d')"; } | sg_put "$root" check/b.sh
verdict twin_same_lines_in_reverse_order 0 "the same lines in the opposite order share no run: the bound alone would call them twins" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_languages)
{ sg_valid_sh "$(words 30 'v_' | sed 's/$/=1/')"; } | sg_put "$root" check/a.sh
py_entry "$root" report/y.py < <(words 30 'v_' | sed 's/$/=1/')
verdict twin_other_language_is_no_twin 0 "a shell file and a Python file with the same lines are not compared" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_comments)
{ sg_valid_sh "$(words 24 ': w_')"; words 30 '# c_'; /usr/bin/yes '}' | head -n 30; } | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 24 ': v_')"; words 30 '# c_'; /usr/bin/yes '}' | head -n 30; } | sg_put "$root" check/b.sh
verdict twin_comments_and_closers_are_not_significant 0 "30 shared comment lines and 30 shared closers do not make a twin" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_floor_19)
twin_of 19 | sg_put "$root" check/a.sh
twin_of 19 | sg_put "$root" check/b.sh
verdict twin_below_the_ratio_floor 0 "two identical files of 19 significant lines" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_floor_20)
twin_of 20 | sg_put "$root" check/a.sh
twin_of 20 | sg_put "$root" check/b.sh
verdict twin_at_the_ratio_floor 1 "two identical files of 20" "$root" \
    --has "twin: check/a.sh:1: 20 matched significant lines with check/b.sh"

root=$(sg_fresh twin_odd_half)
twin_of 31 | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 14 ': w_')"; words 20 ': v_'; } | sg_put "$root" check/b.sh
verdict twin_half_rounds_up 0 "15 of 31 lines is under half: it takes 16" "$root" \
    --has "check-scripts: ok (6 files"

root=$(sg_fresh twin_pin)
twin_of 45 | sg_put "$root" check/a.sh
twin_of 45 | sg_put "$root" check/b.sh
sg_pins_file 'twin check/a.sh+check/b.sh 45' >"$root/$SG_PINS_REL"
verdict twin_pin_holds 0 "a pinned pair passes and is counted" "$root" \
    --has "pinned: twin 1"
twin_of 50 | sg_put "$root" check/a.sh
twin_of 50 | sg_put "$root" check/b.sh
st_case twin_pin_grew 1 --what "a pinned pair whose figure grew fails twin-pin" \
    --has "twin-pin: check/a.sh:1: twin figure is 50, pinned 45" -- sg_gate "$root"
twin_of 45 | sg_put "$root" check/a.sh
{ sg_valid_sh "$(words 24 ': w_')"; words 40 ': v_'; } | sg_put "$root" check/b.sh
st_case twin_pin_shrank 1 --what "one that shrank asks for the pin to be lowered" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: twin check/a.sh+check/b.sh: figure is 25, pinned 45" -- sg_gate "$root"

root=$(sg_fresh twin_pin_moved)
twin_of 45 | sg_put "$root" check/a.sh
twin_of 45 | sg_put "$root" check/b.sh
sg_pins_file 'twin check/a.sh+check/b.sh 45' >"$root/$SG_PINS_REL"
sg_reindex "$root"
sg_commit_all "$root" "pin the pair"
git -C "$root" mv scripts/check/a.sh scripts/check/z.sh
sg_pins_file 'twin check/b.sh+check/z.sh 45' >"$root/$SG_PINS_REL"
verdict twin_pin_follows_a_moved_file 0 "a pinned pair keeps its pin when one file moves and the pin line moves with it" "$root" \
    --has "pinned: twin 1" --lacks "pin-new"

# ---- lib-shadow ----------------------------------------------------------

root=$(sg_fresh libshadow)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
helper() { :; }
EOF
verdict lib_shadow_function 1 "a function lib/helper.sh defines, defined again in a script" "$root" \
    --has "lib-shadow: check/y.sh:10: helper() is defined in lib/helper.sh"

root=$(sg_fresh libshadow_keyword)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
function helper {
    :
}
EOF
verdict lib_shadow_function_keyword_form 1 "the function keyword form" "$root" \
    --has "lib-shadow: check/y.sh:10: helper() is defined in lib/helper.sh"

root=$(sg_fresh libshadow_other)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
helper_two() { :; }
cat <<'TXT' >/dev/null
helper() { :; }
TXT
# helper() { :; }
EOF
verdict lib_shadow_other_names_heredocs_comments 0 "another name, a heredoc body and a comment define nothing" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh libshadow_recall)
sg_recall_sh '# Caller: by hand' 'check/ok.sh' | sed 's/^exit 0$/helper() { :; }/' | sg_put "$root" check/y_selftest.sh
verdict lib_shadow_recall_test_is_not_exempt 1 "a recall test that redefines a lib function no longer exercises it" "$root" \
    --has "lib-shadow: check/y_selftest.sh:"

root=$(sg_fresh libshadow_pin)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
helper() { :; }
EOF
sg_pins_file 'lib-shadow check/y.sh helper' >"$root/$SG_PINS_REL"
verdict lib_shadow_pin_holds 0 "a pinned function passes and is counted" "$root" \
    --has "pinned: lib-shadow 1"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
EOF
st_case lib_shadow_pin_stale 1 --what "and is stale once the function is gone" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: lib-shadow check/y.sh: no longer violated" -- sg_gate "$root"

root=$(sg_fresh libshadow_pin_other)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
helper() { :; }
EOF
sg_pins_file 'lib-shadow check/y.sh other_function' >"$root/$SG_PINS_REL"
verdict lib_shadow_pin_names_another_function 1 "a pin for another function accepts nothing" "$root" \
    --has "lib-shadow: check/y.sh:10: helper() is defined in lib/helper.sh" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: lib-shadow check/y.sh: function other_function is fixed or gone"

# ---- binary-literal ------------------------------------------------------

root=$(sg_fresh binary)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
BIN=target/release/rmlx
BIN2="$REPO_ROOT/target/$PROFILE/rmlx"
EOF
verdict binary_literal_sites 1 "target/<profile>/rmlx written out, with a variable profile too" "$root" \
    --has 'binary-literal: check/y.sh:10: `target/release/rmlx` is written out' \
    --has 'binary-literal: check/y.sh:11: `target/$PROFILE/rmlx` is written out'

root=$(sg_fresh binary_other)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
# target/release/rmlx in a comment
BIN=target/release/rmlx-server
BIN2=target/release/rmlx_main
EOF
verdict binary_literal_not_a_site 0 "a comment, rmlx-server and rmlx_main are no binary literal" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh binary_identity)
printf '%s\n' '# The planted identity library.' '#' '# Module of: check/ok.sh' 'BIN=target/release/rmlx' | sg_put "$root" lib/identity.sh
printf '%s\n' '. "$(dirname "$0")/../lib/identity.sh"' >>"$root/scripts/check/ok.sh"
verdict binary_literal_in_identity_library 0 "lib/identity.sh may write it" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh binary_recall)
sg_recall_sh '# Caller: by hand' 'check/ok.sh' | sed 's/^exit 0$/STUB=target\/release\/rmlx/' | sg_put "$root" check/y_selftest.sh
verdict binary_literal_in_recall_test 0 "a recall test that plants a stub binary is exempt" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh binary_named_selftest)
sh_entry "$root" check/y_selftest.sh 'set -euo pipefail' <<'EOF'
BIN=target/release/rmlx
EOF
verdict binary_literal_name_alone_is_no_exemption 1 "a _selftest.sh name with no Recall test for: line is not a recall test" "$root" \
    --has "binary-literal: check/y_selftest.sh:10:"

root=$(sg_fresh binary_pin)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
BIN=target/release/rmlx
BIN2=target/debug/rmlx
EOF
sg_pins_file 'binary-literal check/y.sh 2' >"$root/$SG_PINS_REL"
verdict binary_literal_pin_holds 0 "two pinned sites pass and are counted once" "$root" \
    --has "pinned: binary-literal 1"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
BIN=target/release/rmlx
BIN2=target/debug/rmlx
BIN3=target/release-perf/rmlx
EOF
st_case binary_literal_pin_grew 1 --what "a third site fails binary-literal-pin" \
    --has "binary-literal-pin: check/y.sh:10: binary-literal figure is 3, pinned 2" -- sg_gate "$root"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
BIN=target/release/rmlx
EOF
st_case binary_literal_pin_shrank 1 --what "one site fewer asks for the pin to be lowered" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: binary-literal check/y.sh: figure is 1, pinned 2" -- sg_gate "$root"

# ---- env-knob ------------------------------------------------------------

root=$(sg_fresh env_forms)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
: "${KNOB_A:-x}"
: "${KNOB_B-x}"
: "${KNOB_C:=x}"
: "${KNOB_D:?must be set}"
EOF
verdict env_knob_read_forms 1 "the four default forms of an environment read" "$root" \
    --has "env-knob: check/y.sh:10: reads the environment variable KNOB_A" \
    --has "env-knob: check/y.sh:11: reads the environment variable KNOB_B" \
    --has "env-knob: check/y.sh:12: reads the environment variable KNOB_C" \
    --has "env-knob: check/y.sh:13: reads the environment variable KNOB_D"

root=$(sg_fresh env_not_reads)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
: "${KNOB_A}" "${KNOB_B:+x}" "${knob_c:-x}" "${1:-x}" "${#KNOB_D}" "${HOME:-/}" "${PATH:-}" "${TMPDIR:-/tmp}"
# : "${KNOB_E:-x}"
cat <<'TXT' >/dev/null
: "${KNOB_F:-x}"
TXT
EOF
verdict env_knob_not_a_read 0 "a plain expansion, :+, lower case, a parameter, a comment, a heredoc and the system names" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh env_assigned)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
KNOB_A=1
: "${KNOB_A:-x}"
local KNOB_B=2
: "${KNOB_B:-x}"
for KNOB_C in a b; do
    : "${KNOB_C:-x}"
done
KNOB_D="${KNOB_D:-x}"
EOF
verdict env_knob_assigned_earlier 1 "a name assigned on an earlier line is no environment read; one assigned from its own default is" "$root" \
    --has "env-knob: check/y.sh:17: reads the environment variable KNOB_D" --lacks "KNOB_A" --lacks "KNOB_B" --lacks "KNOB_C"

for read in 'os.environ["KNOB_P"]' 'os.environ.get("KNOB_P")' 'os.environ.get("KNOB_P", "x")' 'os.getenv("KNOB_P")'; do
    root=$(sg_fresh "env_py_${#read}")
    py_entry "$root" report/y.py <<EOF
import os

VALUE = $read
EOF
    sg_reindex "$root"
    st_case "env_knob_python_${#read}" 1 --what "Python: $read" \
        --has "env-knob: report/y.py:11: reads the environment variable KNOB_P" -- sg_gate "$root"
done

root=$(sg_fresh env_py_not_reads)
py_entry "$root" report/y.py <<'EOF'
import os

name = "KNOB_P"
os.environ["KNOB_Q"] = "1"
VALUE = (os.environ.get(name), os.environ["HOME"], os.getenv("lower"), os.environ.get("TMPDIR"))
EOF
verdict env_knob_python_not_reads 0 "a store, a computed name, a lower-case name and the system names" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh env_declared)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
declare KNOB_E
read -r KNOB_F
export KNOB_G
: "${KNOB_E:-x}" "${KNOB_F:-x}" "${KNOB_G:-x}"
EOF
verdict env_knob_declared_names 0 "a name declared by declare, read or export is the file's own" "$root" \
    --has "check-scripts: ok (5 files"

root=$(sg_fresh env_py_imported)
py_entry "$root" report/y.py <<'EOF'
from os import environ, getenv

A = environ["KNOB_P"]
B = getenv("KNOB_Q")
EOF
verdict env_knob_python_imported_names 1 "environ and getenv imported from os" "$root" \
    --has "env-knob: report/y.py:11: reads the environment variable KNOB_P" \
    --has "env-knob: report/y.py:12: reads the environment variable KNOB_Q"

root=$(sg_fresh env_pin)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
: "${KNOB_A:-x}" "${KNOB_B:-x}"
EOF
sg_pins_file 'env-knob check/y.sh KNOB_A,KNOB_B' >"$root/$SG_PINS_REL"
verdict env_knob_pin_holds 0 "pinned names pass and the file is counted once" "$root" \
    --has "pinned: env-knob 1"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
: "${KNOB_A:-x}" "${KNOB_B:-x}" "${KNOB_C:-x}"
EOF
st_case env_knob_pin_new_name 1 --what "a name outside the pin fails" \
    --has "env-knob: check/y.sh:10: reads the environment variable KNOB_C" --lacks "KNOB_A" -- sg_gate "$root"
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
: "${KNOB_A:-x}"
EOF
st_case env_knob_pin_stale_name 1 --what "a pinned name that is gone is stale" \
    --has "pin-stale: report/scripts_gate_pins.txt:4: env-knob check/y.sh: variable KNOB_B is fixed or gone" -- sg_gate "$root"

root=$(sg_fresh env_pin_new)
sh_entry "$root" check/y.sh 'set -euo pipefail' <<'EOF'
: "${KNOB_A:-x}" "${KNOB_B:-x}"
EOF
sg_pins_file 'env-knob check/y.sh KNOB_A' >"$root/$SG_PINS_REL"
sg_reindex "$root"
sg_commit_all "$root" "pin KNOB_A"
sg_pins_file 'env-knob check/y.sh KNOB_A,KNOB_B' >"$root/$SG_PINS_REL"
st_case env_knob_pin_cannot_grow 1 --what "a name added to a pin the base already had" \
    --has "pin-new: report/scripts_gate_pins.txt:4: env-knob check/y.sh: variable KNOB_B is not pinned at the base" -- sg_gate "$root"

# ---- shellcheck ----------------------------------------------------------

root=$(sg_fresh shellcheck)
STUB="$ST_WORK/stub"
mkdir -p "$STUB"
{
    printf '%s\n' '#!/bin/sh' 'echo "$@" >"$0.args"'
    printf '%s\n' 'echo "scripts/check/ok.sh:3:1: warning: planted finding [SC0000]"' 'exit 1'
} >"$STUB/shellcheck"
chmod +x "$STUB/shellcheck"
st_case shellcheck_prints_findings 0 --what "a finding is printed with a summary and the exit is 0 whatever shellcheck exits" \
    --has "scripts/check/ok.sh:3:1: warning: planted finding [SC0000]" \
    --has "shellcheck-report: 1 finding(s) in 2 shell file(s)" -- env PATH="$STUB:$PATH" python3 "$TOOL" --shellcheck --root "$root"
st_case shellcheck_gets_the_shell_files 0 --what "it is handed the shell files of scripts/ and nothing else" \
    --is "--format=gcc -- scripts/check/ok.sh scripts/lib/helper.sh" -- sh -c 'cat "$1.args"' sh "$STUB/shellcheck"

NOBIN="$ST_WORK/nobin"
mkdir -p "$NOBIN"
ln -s "$(command -v python3)" "$NOBIN/python3"
ln -s "$(command -v git)" "$NOBIN/git"
st_case shellcheck_absent 0 --what "no shellcheck on PATH: a skipped line and exit 0" \
    --has "shellcheck-report: skipped: shellcheck not on PATH" -- env PATH="$NOBIN" "$NOBIN/python3" "$TOOL" --shellcheck --root "$root"

NOTREE="$ST_WORK/ceiling/notree"
mkdir -p "$NOTREE"
shellcheck_outside_git() {
    GIT_CEILING_DIRECTORIES="$ST_WORK" env PATH="$STUB:$PATH" python3 "$TOOL" --shellcheck --root "$NOTREE"
}
st_case shellcheck_outside_git 0 --what "a tree that is no git work tree is reported, not failed" \
    --has "shellcheck-report: unavailable (" -- shellcheck_outside_git

# ---- dead-path: the README and the case runner ----------------------------

root=$(sg_fresh dead_readme)
printf '# rules\n\nSee scripts/check/gone.sh and scripts/check/ok.sh.\n' >"$root/scripts/README.md"
verdict dead_path_in_the_readme 1 "scripts/README.md is read: a token that names no file" "$root" \
    --has "dead-path: ../scripts/README.md:3: scripts/check/gone.sh names no file" --lacks "scripts/check/ok.sh names"

root=$(sg_fresh dead_support)
printf '%s\n' '# A planted case runner.' '#' '# Module of: check/ok.sh' '# Usage: bash scripts/check/gone.sh' 'st_x() { :; }' |
    sg_put "$root" lib/selftest_support.sh
printf '%s\n' '. "$(dirname "$0")/../lib/selftest_support.sh"' >>"$root/scripts/check/ok.sh"
verdict dead_path_not_read_in_the_case_runner 0 "lib/selftest_support.sh plants trees, so its paths are not read" "$root" \
    --has "check-scripts: ok (5 files"

st_done
