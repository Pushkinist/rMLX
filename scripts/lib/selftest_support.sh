#!/usr/bin/env bash
# selftest_support.sh — what the spec-bench self-tests share. Source it.
#
# The two spec-bench tests drive a harness against a stub server and need a
# stand-in for `python3` that stands between the harness and the client reader.
# They and bench_llama_ab_selftest.sh need a wait that cannot hang.

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
