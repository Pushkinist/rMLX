#!/usr/bin/env bash
# scripts/check_personal_data_selftest.sh — recall test for
# check_personal_data.sh.
#
# Each case builds a throwaway git tree, plants one edit, runs the gate over it
# and asserts the exit code and, for a violation, the rule and the file:line
# the gate printed; for a refusal, the reason. The planted values are built
# from parts at run time, so this file holds no line the gate would report.

set -uo pipefail
export LC_ALL=C

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/check_personal_data.sh"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/personal-data-selftest.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

AT='@'
pass=0
fail=0

new_tree() {
    local d="$WORK/$1"
    mkdir -p "$d"
    git -C "$d" init -q
    printf 'clean line\n' >"$d/README.md"
    printf '%s\n' "$d"
}

# expect <name> <tree> <exit> <needle>
expect() {
    local name="$1" tree="$2" want="$3" needle="$4" out rc
    out="$(bash "$GATE" "$tree" 2>&1)"
    rc=$?
    if [ "$rc" -eq "$want" ] && grep -qF -- "$needle" <<<"$out"; then
        pass=$((pass + 1))
        echo "  PASS  $name"
    else
        fail=$((fail + 1))
        echo "  FAIL  $name (exit $rc, wanted $want; looking for: $needle)"
        sed 's/^/        /' <<<"$out"
    fi
}

t="$(new_tree clean)"
expect "a clean tree passes" "$t" 0 "check-personal-data: ok"

t="$(new_tree email)"
printf 'line one\nmail jane.doe%sgmail.com here\n' "$AT" >"$t/notes.txt"
expect "a personal address fails" "$t" 1 "email: notes.txt:2:"

t="$(new_tree email-upper)"
printf 'Contact: Jane%sMail.Example.Ru\n' "$AT" >"$t/a.json"
expect "an address in any case fails" "$t" 1 "email: a.json:1:"

t="$(new_tree placeholder)"
printf 'a %sexample.invalid b %susers.noreply.github.com c %sexample.com\n' \
    "x$AT" "1+me$AT" "y$AT" >"$t/ok.txt"
expect "placeholder and noreply domains pass" "$t" 0 "check-personal-data: ok"

t="$(new_tree mixed)"
printf 'x%sexample.invalid then y%scorp.io\n' "$AT" "$AT" >"$t/m.txt"
expect "a real address after a placeholder on one line fails" "$t" 1 "email: m.txt:1:"

t="$(new_tree users)"
printf 'path /%s/alice/src\n' "Users" >"$t/s.sh"
expect "a macOS home path fails" "$t" 1 "home-path: s.sh:1:"

t="$(new_tree users-placeholder)"
printf 'path /%s/<you>/src\n' "Users" >"$t/s.md"
expect "an angle-bracket placeholder home path passes" "$t" 0 "check-personal-data: ok"

t="$(new_tree varfolders)"
printf 'tmp /private/var/%s/ab/T/x\n' "folders" >"$t/v.txt"
expect "a per-user temp path fails" "$t" 1 "home-path: v.txt:1:"

t="$(new_tree linux-home)"
printf 'db /home/user/.rmlx/metrics/runs.db\n' >"$t/l.md"
expect "a Linux placeholder home path passes" "$t" 0 "check-personal-data: ok"

t="$(new_tree session)"
printf 'see https://claude.ai/code/%s_01ABC\n' "session" >"$t/pr.md"
expect "an assistant session link fails" "$t" 1 "session-link: pr.md:1:"

t="$(new_tree ignored)"
printf '*.local\n' >"$t/.gitignore"
printf 'jane%sgmail.com\n' "$AT" >"$t/notes.local"
expect "a git-ignored file is out of scope" "$t" 0 "check-personal-data: ok"

t="$(new_tree untracked)"
printf 'jane%sgmail.com\n' "$AT" >"$t/new.txt"
expect "an untracked, not-ignored file is in scope" "$t" 1 "email: new.txt:1:"

t="$(new_tree binary)"
printf 'jane%sgmail.com\000\001' "$AT" >"$t/blob.bin"
expect "a binary file is skipped" "$t" 0 "check-personal-data: ok"

mkdir -p "$WORK/nogit"
printf 'x\n' >"$WORK/nogit/a.txt"
expect "a root outside a git work tree cannot be scanned" "$WORK/nogit" 2 "git ls-files exit"

expect "a missing root cannot be scanned" "$WORK/does-not-exist" 2 "is not a directory"

t="$WORK/empty"
mkdir -p "$t"
git -C "$t" init -q
expect "an empty scope cannot be scanned" "$t" 2 "no in-scope file"

echo "check_personal_data selftest: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
