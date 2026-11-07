# Resolve the runtime home the way the binary does, for the shell tools; source it.
#
# Sourced, not executed. The twin of lib/rmlx_home.py.
#
#     source "$REPO_ROOT/scripts/lib/rmlx_home.sh"
#     RMLX_HOME="$(rmlx_home "$REPO_ROOT")" || exit 2
#
# `rmlx_home <repo_root>` prints $RMLX_HOME when it is set and absolute, `<repo_root>/.rmlx` when it
# is unset. A relative or empty value is refused: the line `RMLX_HOME=<value> must be an absolute
# path` goes to stderr and the status is 1. A tool passes the setup or usage exit its header declares.
rmlx_home() {
	local root="${1:?rmlx_home: pass the checkout root}"
	if [[ -z "${RMLX_HOME+set}" ]]; then
		printf '%s/.rmlx\n' "$root"
	elif [[ "$RMLX_HOME" == /* ]]; then
		printf '%s\n' "$RMLX_HOME"
	else
		echo "RMLX_HOME=$RMLX_HOME must be an absolute path" >&2
		return 1
	fi
}
