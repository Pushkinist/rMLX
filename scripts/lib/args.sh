# Argument-parsing helpers shared by the bench harnesses; source it.
#
# Sourced, not executed.
#
#     source "$REPO_ROOT/scripts/lib/args.sh"
#     while [[ $# -gt 0 ]]; do
#         case "$1" in
#         --model) need_value "$1" $#; MODEL="$2"; shift 2 ;;
#
# `need_value <flag> <count>` exits 125 when fewer than two arguments are left, so a flag given
# last without its value is refused instead of reading an empty `$2`.
need_value() {
	if [[ $2 -lt 2 ]]; then
		echo "ERROR: $1 requires a value" >&2
		exit 125
	fi
}
