#!/usr/bin/env bash
# Recall test for check_personal_data.sh: throwaway git trees, asserting exit code and file:line.
#
# Each case builds a throwaway git tree, plants one edit, runs the gate over it
# and asserts the exit code and, for a violation, the rule and the file:line
# the gate printed; for a refusal, the reason. The planted values are built
# from parts at run time, so this file holds no line the gate would report.

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/check_personal_data.sh"
# shellcheck source=scripts/lib/selftest_support.sh
. "$HERE/lib/selftest_support.sh"
st_init check-personal-data-selftest

AT='@'

new_tree() {
    local d
    d="$(st_git_root "$1")"
    printf 'clean line\n' >"$d/README.md"
    printf '%s\n' "$d"
}

t="$(new_tree clean)"
st_case "a clean tree passes" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"

t="$(new_tree email)"
printf 'line one\nmail jane.doe%sgmail.com here\n' "$AT" >"$t/notes.txt"
st_case "a personal address fails" 1 --has "email: notes.txt:2:" -- bash "$GATE" "$t"

t="$(new_tree email-upper)"
printf 'Contact: Jane%sMail.Example.Ru\n' "$AT" >"$t/a.json"
st_case "an address in any case fails" 1 --has "email: a.json:1:" -- bash "$GATE" "$t"

t="$(new_tree placeholder)"
printf 'a %sexample.invalid b %susers.noreply.github.com c %sexample.com\n' \
    "x$AT" "1+me$AT" "y$AT" >"$t/ok.txt"
st_case "placeholder and noreply domains pass" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"

t="$(new_tree mixed)"
printf 'x%sexample.invalid then y%scorp.io\n' "$AT" "$AT" >"$t/m.txt"
st_case "a real address after a placeholder on one line fails" 1 --has "email: m.txt:1:" -- bash "$GATE" "$t"

t="$(new_tree users)"
printf 'path /%s/alice/src\n' "Users" >"$t/s.sh"
st_case "a macOS home path fails" 1 --has "home-path: s.sh:1:" -- bash "$GATE" "$t"

t="$(new_tree users-placeholder)"
printf 'path /%s/<you>/src\n' "Users" >"$t/s.md"
st_case "an angle-bracket placeholder home path passes" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"

t="$(new_tree varfolders)"
printf 'tmp /private/var/%s/ab/T/x\n' "folders" >"$t/v.txt"
st_case "a per-user temp path fails" 1 --has "home-path: v.txt:1:" -- bash "$GATE" "$t"

t="$(new_tree linux-home)"
printf 'db /home/user/.rmlx/metrics/runs.db\n' >"$t/l.md"
st_case "a Linux placeholder home path passes" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"

t="$(new_tree session)"
printf 'see https://claude.ai/code/%s_01ABC\n' "session" >"$t/pr.md"
st_case "an assistant session link fails" 1 --has "session-link: pr.md:1:" -- bash "$GATE" "$t"

t="$(new_tree ignored)"
printf '*.local\n' >"$t/.gitignore"
printf 'jane%sgmail.com\n' "$AT" >"$t/notes.local"
st_case "a git-ignored file is out of scope" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"

t="$(new_tree untracked)"
printf 'jane%sgmail.com\n' "$AT" >"$t/new.txt"
st_case "an untracked, not-ignored file is in scope" 1 --has "email: new.txt:1:" -- bash "$GATE" "$t"

t="$(new_tree binary)"
printf 'jane%sgmail.com\000\001' "$AT" >"$t/blob.bin"
st_case "a binary file is skipped" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"

for domain in example.org example.net noreply.github.com; do
    t="$(new_tree "placeholder-$domain")"
    printf 'a x%s%s b\n' "$AT" "$domain" >"$t/ok.txt"
    st_case "the placeholder domain $domain passes" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"
done

t="$(new_tree scratch-path)"
printf 'tmp /private/tmp/%s/x\n' "claude" >"$t/c.txt"
st_case "an assistant scratch path fails" 1 --has "home-path: c.txt:1:" -- bash "$GATE" "$t"

t="$(new_tree tracked)"
printf 'jane%sgmail.com\n' "$AT" >"$t/staged.txt"
git -C "$t" add staged.txt
st_case "a staged file is in scope" 1 --has "email: staged.txt:1:" -- bash "$GATE" "$t"

t="$(new_tree own-files)"
mkdir -p "$t/scripts"
printf 'jane%sgmail.com\n' "$AT" >"$t/scripts/check_personal_data.sh"
printf 'jane%sgmail.com\n' "$AT" >"$t/scripts/check_personal_data_selftest.sh"
st_case "the gate and its selftest are not scanned" 0 --has "check-personal-data: ok" -- bash "$GATE" "$t"

t="$(new_tree two-on-a-line)"
printf 'jane%sgmail.com and joe%scorp.io\n' "$AT" "$AT" >"$t/two.txt"
st_case "two addresses on one line are one violation" 1 --has "FAIL: 1 violation(s)" -- bash "$GATE" "$t"

mkdir -p "$ST_WORK/nogit"
printf 'x\n' >"$ST_WORK/nogit/a.txt"
st_case "a root outside a git work tree cannot be scanned" 2 --has "git ls-files exit" -- bash "$GATE" "$ST_WORK/nogit"

st_case "a missing root cannot be scanned" 2 --has "is not a directory" -- bash "$GATE" "$ST_WORK/does-not-exist"

t="$(st_git_root empty)"
st_case "an empty scope cannot be scanned" 2 --has "no in-scope file" -- bash "$GATE" "$t"

st_done
