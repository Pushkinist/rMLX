#!/usr/bin/env bash
# Diff the function sets of two .gputrace bundles: what did A's window reference
# that B's did not, and the other way round.
#
# This is the A/B that motivates capturing at all — "what does iso3_sym dispatch
# that none does not", one codec against another, or one commit against another.
# It was first done by hand, reverse-engineering names out of the archive; this
# is the same operation with the parser and its layout checks in
# scripts/gputrace_kernels.sh, so a bundle it cannot read is an error rather
# than an empty diff that reads like "no difference".
#
# Usage:
#   bash scripts/gputrace_diff.sh <a.gputrace> <b.gputrace>
#   bash scripts/gputrace_diff.sh <a.gputrace> <b.gputrace> --set unused
#
# Exit: 0 = compared (whether or not they differ), 2 = usage, 3 = a bundle could
#       not be parsed (the reason comes from gputrace_kernels.sh)

set -uo pipefail

A=""
B=""
SET="used"

while [ $# -gt 0 ]; do
	case "$1" in
	--set)
		if [ $# -lt 2 ]; then
			echo "--set needs used|unused" >&2
			exit 2
		fi
		SET="$2"
		shift 2
		;;
	-h | --help)
		echo "usage: $0 <a.gputrace> <b.gputrace> [--set used|unused]"
		exit 0
		;;
	-*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	*)
		if [ -z "$A" ]; then
			A="$1"
		elif [ -z "$B" ]; then
			B="$1"
		else
			echo "expected exactly two bundles (got a third: $1)" >&2
			exit 2
		fi
		shift
		;;
	esac
done

if [ -z "$A" ] || [ -z "$B" ]; then
	echo "usage: $0 <a.gputrace> <b.gputrace> [--set used|unused]" >&2
	exit 2
fi
case "$SET" in
used | unused) ;;
*)
	echo "--set must be used or unused (got '$SET')" >&2
	exit 2
	;;
esac

cd "$(dirname "$0")/.." || exit 1
here="$(pwd)"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/gputrace-diff.XXXXXX") || exit 1
trap 'rm -rf "$tmp"' EXIT

# The kernel lister does the parsing and owns every layout check. Its failures
# are ours: a diff computed from a half-read bundle is worse than no diff.
list() {
	local bundle="$1" out="$2"
	if ! bash "$here/scripts/gputrace_kernels.sh" "$bundle" --set "$SET" --names-only >"$out"; then
		echo "ERROR: could not list functions for $bundle (see above) — no diff produced." >&2
		return 1
	fi
	sort -u "$out" -o "$out"
}

list "$A" "$tmp/a" || exit 3
list "$B" "$tmp/b" || exit 3

only_a=$(comm -23 "$tmp/a" "$tmp/b")
only_b=$(comm -13 "$tmp/a" "$tmp/b")
shared=$(comm -12 "$tmp/a" "$tmp/b" | wc -l | tr -d ' ')

count() {
	if [ -z "$1" ]; then echo 0; else printf '%s\n' "$1" | wc -l | tr -d ' '; fi
}

echo "set:    $SET functions"
echo "A:      $A ($(wc -l <"$tmp/a" | tr -d ' ') named)"
echo "B:      $B ($(wc -l <"$tmp/b" | tr -d ' ') named)"
echo "shared: $shared"
echo ""
echo "only in A ($(count "$only_a")):"
[ -n "$only_a" ] && printf '%s\n' "$only_a" | sed 's/^/  /'
echo ""
echo "only in B ($(count "$only_b")):"
[ -n "$only_b" ] && printf '%s\n' "$only_b" | sed 's/^/  /'

if [ -z "$only_a" ] && [ -z "$only_b" ]; then
	echo ""
	echo "identical named function sets"
fi

# Names the parser could not read are per-bundle and not part of either list;
# gputrace_kernels.sh reports their count for each bundle.
exit 0
