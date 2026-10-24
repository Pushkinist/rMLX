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
