# The case runner and shims the self-tests share; source it.
#
# The case runner (st_*) counts cases and prints one summary line. A harness
# calls st_init, then st_case for a case that is a command, an exit code and
# lines of output, or st_run + st_check + st_verdict for a case that also
# inspects what the command left behind, then ends with st_done. Exit codes of
# a harness on the runner: 0 every case held, 1 one did not or none ran,
# 2 a fixture could not be built or a case was written with a bad needle.
# st_init installs the EXIT trap, so it replaces any trap set before it.
#
# The two spec-bench tests drive a harness against a stub server and need a
# stand-in for `python3` that stands between the harness and the client reader.
# They and bench_llama_ab_selftest.sh need a wait that cannot hang.

# st_init <suite> — zero the counters and make the scratch directory ST_WORK,
# removed when the harness exits.
st_init() {
	ST_SUITE="$1"
	ST_PASSED=0
	ST_FAILED=0
	ST_OUT=""
	ST_RC=0
	local tmp="${TMPDIR:-/tmp}"
	ST_WORK="$(mktemp -d "${tmp%/}/${1}.XXXXXX")" || {
		echo "${1}: cannot make a scratch directory" >&2
		exit 2
	}
	trap st_cleanup EXIT
}

st_cleanup() {
	[ -z "${ST_WORK:-}" ] || rm -rf "$ST_WORK"
}

# st_die <why> — the harness itself is broken: exit 2, not a red case. Called
# inside $(...) it leaves a mark that the next st_run, st_verdict or st_done
# turns into the same exit in the harness's own shell.
st_die() {
	echo "${ST_SUITE}: $1" >&2
	: >"$ST_WORK/.broken"
	exit 2
}

st_abort_if_broken() {
	[ ! -e "$ST_WORK/.broken" ] || exit 2
}

# st_git_root <name> — print $ST_WORK/<name>, made and initialised as a git work tree.
st_git_root() {
	local root="$ST_WORK/$1"
	mkdir -p "$root" && git -C "$root" init -q || st_die "cannot build a fixture: git init in $root"
	printf '%s' "$root"
}

# st_run <cmd...> — run it with stdout and stderr together. ST_OUT holds the
# output and ST_RC the exit code.
st_run() {
	st_abort_if_broken
	ST_OUT=$("$@" 2>&1) && ST_RC=0 || ST_RC=$?
}

# st_check <want-exit> [--has S]... [--lacks S]... [--is S] — judge the last
# st_run: print one reason per line for each condition that fails, nothing when
# all hold. --has and --lacks match a fixed one-line, non-empty string anywhere
# in the output; --is wants the whole output to equal S.
st_check() {
	local want="$1"
	shift
	[ "$ST_RC" -eq "$want" ] || echo "exit $ST_RC, want $want"
	while [ "$#" -gt 0 ]; do
		case "$1" in
		--is) [ "$ST_OUT" = "${2-}" ] || echo "output is '$ST_OUT', want '${2-}'" ;;
		--has | --lacks)
			case "${2-}" in
			"" | *$'\n'*) st_die "$1 takes one non-empty line, got '${2-}'" ;;
			esac
			if [ "$1" = --has ]; then
				grep -qF -- "$2" <<<"$ST_OUT" || echo "no '$2' in the output"
			else
				! grep -qF -- "$2" <<<"$ST_OUT" || echo "'$2' is in the output"
			fi
			;;
		*) echo "st_check: unknown option '$1'" ;;
		esac
		shift 2 || break
	done
}

# st_verdict <name> <why> [<what>] — count the case as held when <why> is
# empty; otherwise count it as failed and print <why> and the head of ST_OUT.
st_verdict() {
	local name="$1" why="$2" what="${3:-}"
	st_abort_if_broken
	if [ -z "$why" ]; then
		ST_PASSED=$((ST_PASSED + 1))
		printf '  ok   %s%s\n' "$name" "${what:+ - $what}"
	else
		ST_FAILED=$((ST_FAILED + 1))
		printf '  FAIL %s%s\n' "$name" "${what:+ - $what}"
		printf '%s\n' "$why" | sed 's/^/         /'
		[ -z "$ST_OUT" ] || printf '%s\n' "$ST_OUT" | head -40 | sed 's/^/    | /'
	fi
	ST_OUT=""
}

# st_case <name> <want-exit> [--what S] [--has S]... [--lacks S]... [--is S]
# -- <cmd...> — run the command and hold it to an exact exit code and to its output.
st_case() {
	local name="$1" want="$2" what="" checks=()
	shift 2
	while [ "$#" -gt 0 ] && [ "$1" != -- ]; do
		if [ "$1" = --what ]; then what="${2-}"; else checks+=("$1" "${2-}"); fi
		shift 2 || break
	done
	[ "${1-}" = -- ] || { st_verdict "$name" "st_case: no '--' before the command" "$what"; return; }
	shift
	st_run "$@"
	st_verdict "$name" "$(st_check "$want" ${checks[@]+"${checks[@]}"})" "$what"
}

# st_done — print the summary line; fail when a case failed or none ran.
st_done() {
	st_abort_if_broken
	printf '%s: %d passed, %d failed\n' "$ST_SUITE" "$ST_PASSED" "$ST_FAILED"
	if [ "$((ST_PASSED + ST_FAILED))" -eq 0 ]; then
		echo "${ST_SUITE}: no case ran" >&2
		return 1
	fi
	[ "$ST_FAILED" -eq 0 ]
}

# write_client_shim <dir> <delayed> [<step>] — write <dir>/bin/python3, a
# stand-in for `python3` that watches the client reader and runs everything else
# unchanged.
#
# The Nth start of the client reader leaves the file <dir>/mark.N and adds its
# arguments to <dir>/reader_argv. When <delayed> is a number, that client sleeps
# one second before it leaves its mark. 0 delays none. The stub server reads the
# marks (`--marks <dir>`), so it can say for each request whether the client that
# sent it was already running.
#
# With <step> the client reader runs on a clock of its own: `time.monotonic()`
# starts at 100 s and moves <step> seconds with each call, and the reader calls
# it once for each line it reads. A run on the client clock then has a rate that
# depends on the stream's shape and on nothing else, not on this host.
write_client_shim() {
	local dir="$1" delayed="$2" step="${3:-}" env_line=""
	mkdir -p "${dir}/bin"
	if [ -n "${step}" ]; then
		mkdir -p "${dir}/clock"
		cat >"${dir}/clock/sitecustomize.py" <<CLOCKEOF
import time

_now = [100.0]


def _monotonic():
    _now[0] += ${step}
    return _now[0]


time.monotonic = _monotonic
CLOCKEOF
		env_line="export PYTHONPATH=\"${dir}/clock\""
	fi
	cat >"${dir}/bin/python3" <<SHIMEOF
#!/bin/sh
case "\$1" in
*/sse_decode_window.py)
	n=\$(cat "${dir}/count" 2>/dev/null || echo 0); n=\$((n + 1)); echo \$n >"${dir}/count"
	script="\$1"; shift
	echo "\$*" >>"${dir}/reader_argv"
	[ "\$n" = "${delayed}" ] && /bin/sleep 1
	${env_line}
	: >"${dir}/mark.\$n"
	set -- "\$script" "\$@"
	;;
esac
PATH="\${PATH#${dir}/bin:}"
exec python3 "\$@"
SHIMEOF
	chmod +x "${dir}/bin/python3"
}

# wait_until <seconds> <command...> — poll every 0.1 s until the command
# succeeds. Returns 1 when the limit is reached first, so a hang is a failure
# with a reason and not a gate that never ends. The limit is read from the
# shell's own clock.
wait_until() {
	local limit="$1" started=${SECONDS}
	shift
	until "$@"; do
		if ((SECONDS - started >= limit)); then
			return 1
		fi
		/bin/sleep 0.1
	done
}

# st_bounded <seconds> <command...> — run the command, killed after <seconds>
# with exit 124 and a "timed out" line, so a loop is a red case and not a hung
# run. `timeout(1)` is not on stock macOS.
st_bounded() {
	python3 -c '
import subprocess, sys

try:
    sys.exit(subprocess.run(sys.argv[2:], timeout=float(sys.argv[1])).returncode)
except subprocess.TimeoutExpired:
    print(f"timed out after {sys.argv[1]} s")
    sys.exit(124)
' "$@"
}

# ---- planted scripts/ trees --------------------------------------------------
#
# The recall tests of `debt_report.py --scripts` plant one edit in a copy of a
# small valid tree and run the gate over it with `--base HEAD`. They share the
# builders below. A test sets TOOL (the debt_report.py under test), sources this
# file, calls st_init and then sg_init, which builds the committed base tree.
# Every tree keeps four clean files in scope, so a must-fail case cannot pass
# through an empty scope.

SG_PINS_REL=scripts/report/scripts_gate_pins.txt
SG_NAM='# Not a mode of: check/ok.sh — a planted entry point'
SG_TIMEOUT_S=60
SG_PINNABLE='# pinnable: class purpose usage exit-undeclared caller recall-missing lib-consumers lib-run shebang exec-bit dead-path set-mode name size heredoc twin lib-shadow binary-literal env-knob'

# sg_gate <root> — the scripts gate over <root>, the base being its own HEAD.
sg_gate() {
	python3 "$TOOL" --scripts --root "$1" --base HEAD
}

# sg_bounded_gate <root> — sg_gate, killed after SG_TIMEOUT_S seconds.
sg_bounded_gate() {
	st_bounded "$SG_TIMEOUT_S" python3 "$TOOL" --scripts --root "$1" --base HEAD
}

# sg_gate_at <root> <ref> — the gate with <ref> as its base.
sg_gate_at() {
	python3 "$TOOL" --scripts --root "$1" --base "$2"
}

# sg_reindex <root> — regenerate INDEX.md, so a case isolates the rule it plants.
sg_reindex() {
	st_bounded "$SG_TIMEOUT_S" python3 "$TOOL" --scripts --root "$1" --base HEAD --write-index >/dev/null 2>&1 || true
}

# sg_put <root> <rel> — write stdin to <root>/scripts/<rel>, executable when it starts with a shebang.
sg_put() {
	mkdir -p "$(dirname "$1/scripts/$2")"
	cat >"$1/scripts/$2"
	if [ "$(head -c 2 "$1/scripts/$2")" = '#!' ]; then chmod +x "$1/scripts/$2"; fi
}

# sg_commit_all <root> <message> — commit everything in <root>.
sg_commit_all() {
	git -C "$1" add -A &&
		git -C "$1" -c user.name=selftest -c user.email=selftest@invalid -c commit.gpgsign=false \
			commit -q -m "$2" || st_die "cannot commit the fixture tree $1"
}

# sg_row_line <root> <rel> — the INDEX.md line that holds the row for <rel>.
sg_row_line() {
	grep -n -F "| \`$2\` |" "$1/scripts/INDEX.md" | head -n 1 | cut -d: -f1
}

# sg_entry_sh <line 2> <line 3> <usage> <exit> <caller> <not-a-mode> <body> — a shell
# entry point whose header lines are the arguments. A bare `#` leaves a field out.
sg_entry_sh() {
	printf '%s\n' '#!/usr/bin/env bash' "$1" "$2" "$3" "$4" "$5" "$6" 'set -euo pipefail' "$7"
}

# sg_valid_sh <body> — a shell entry point every rule accepts, with <body> as line 9.
sg_valid_sh() {
	sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
		'# Exit:   0 clean; 1 failed' '# Caller: by hand' "$SG_NAM" "$1"
}

# sg_nocaller_sh <body> — a shell entry point with no Caller: line, so only a run makes it callable.
sg_nocaller_sh() {
	sg_entry_sh '# CI gate: a planted entry point.' '#' '# Usage:  bash scripts/check/ok.sh' \
		'# Exit:   0 clean; 1 failed' '#' "$SG_NAM" "$1"
}

# sg_recall_sh <caller line> <field line>... — a planted recall test; each argument after the first
# is a line of its Recall test for: field. The caller line is a bare # when a recipe runs it.
sg_recall_sh() {
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
	printf '%s\n' '#' '# Usage:  bash scripts/check/ok.sh' '# Exit:   0 held' "$caller" "$SG_NAM" 'set -euo pipefail' 'exit 0'
}

# sg_lib_sh [<last header line>] — a planted lib/ file, no shebang, sourced and never run.
sg_lib_sh() {
	printf '%s\n' '# A planted lib file.' '#' "${1:-# Sourced, not executed.}" 'liby() { :; }'
}

# sg_make_ci <root> <text> — append <text> (printf %b) to the fixture Makefile.
sg_make_ci() {
	printf '%b' "$2" >>"$1/Makefile"
}

# sg_pins_file <line>... — a pin file for the fixture tree: its header, then the lines.
sg_pins_file() {
	printf '%s\n' '# Accepted violations of the fixture tree.' '#' "$SG_PINNABLE" "$@"
}

# sg_init — build the committed base tree in $SG_BASE: an entry point that sources its
# module and a Python entry point that imports its module.
sg_init() {
	SG_BASE="$(st_git_root base)"
	printf 'check-ok:\n\t@bash scripts/check/ok.sh\n\ntool:\n\t@python3 scripts/report/tool.py\n' >"$SG_BASE/Makefile"
	sg_put "$SG_BASE" check/ok.sh <<'EOF'
#!/usr/bin/env bash
# CI gate: the fixture entry point that every rule accepts.
#
# Usage:  bash scripts/check/ok.sh
# Exit:   0 clean; 1 a violation
set -euo pipefail
. "$(dirname "$0")/../lib/helper.sh"
helper
[ "$#" -eq 0 ] || exit 1
exit 0
EOF
	sg_put "$SG_BASE" lib/helper.sh <<'EOF'
# A helper the fixture's entry point sources.
#
# Module of: check/ok.sh
helper() { :; }
EOF
	sg_put "$SG_BASE" report/tool.py <<'EOF'
#!/usr/bin/env python3
"""The fixture's Python entry point; it imports its module.

Usage:  python3 scripts/report/tool.py
Exit:   0 always
"""

import tool_mod

raise SystemExit(0)
EOF
	sg_put "$SG_BASE" report/tool_mod.py <<'EOF'
"""A module the fixture's Python entry point imports.

Module of: report/tool.py
"""
EOF
	sg_commit_all "$SG_BASE" files
	sg_reindex "$SG_BASE"
	sg_commit_all "$SG_BASE" base
}

# sg_fresh <name> — print the path of a copy of the base tree, to plant one edit in.
sg_fresh() {
	cp -R "$SG_BASE" "$ST_WORK/$1" || st_die "cannot copy the base tree to $1"
	printf '%s' "$ST_WORK/$1"
}
