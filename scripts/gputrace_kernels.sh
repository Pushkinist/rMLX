#!/usr/bin/env bash
# List the Metal functions a captured window referenced, straight out of the
# bundle — no Xcode, no replay.
#
# This is the one thing a .gputrace answers offline (docs/PROFILING.md §5): it
# proves a kernel *fired* instead of inferring it from timing, which is what
# every "is the codec's own kernel running, or is it decoding through the bf16
# mirror?" question needs.
#
# Where the names come from. A bundle holds two function/pipeline archives, both
# in Apple's "MTSP" container:
#
#   device-resources-0x<addr>         referenced by the captured window
#   unused-device-resources-0x<addr>  recorded by the capture layer as unused
#
# Inside, each function record is the literal string `function` followed by its
# name, so a `function`-marker scan recovers the list. Some records carry an
# opaque 16-hex object id instead of a name (the container stores that name by
# reference) — those are counted and reported separately rather than dropped,
# because "33 functions, 6 of which this parser could not name" and "33
# functions" are different facts.
#
# The layout is Apple's and is NOT a stable contract, so every structural
# assumption is checked and a violation is a hard error naming what broke — an
# empty list must never be mistaken for "no kernels ran". The checks:
#
#   - the container magic is MTSP;
#   - the referenced set holds at least one `function` record;
#   - at most half the records are unreadable (past that the record shape has
#     moved, and any list would be a guess);
#   - at least one record yields a readable name.
#
# Individual unreadable records are not fatal — they are counted and printed.
#
# Usage:
#   bash scripts/gputrace_kernels.sh <bundle.gputrace>
#   bash scripts/gputrace_kernels.sh <bundle.gputrace> --set unused
#   bash scripts/gputrace_kernels.sh <bundle.gputrace> --set used --names-only
#
# Exit: 0 = parsed, 2 = usage / no such bundle, 3 = bundle layout not understood

set -uo pipefail

BUNDLE=""
SET="all"
NAMES_ONLY=0

while [ $# -gt 0 ]; do
	case "$1" in
	--set)
		if [ $# -lt 2 ]; then
			echo "--set needs used|unused|all" >&2
			exit 2
		fi
		SET="$2"
		shift 2
		;;
	--names-only)
		NAMES_ONLY=1
		shift
		;;
	-h | --help)
		echo "usage: $0 <bundle.gputrace> [--set used|unused|all] [--names-only]"
		exit 0
		;;
	-*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	*)
		if [ -n "$BUNDLE" ]; then
			echo "only one bundle at a time (got '$BUNDLE' and '$1')" >&2
			exit 2
		fi
		BUNDLE="$1"
		shift
		;;
	esac
done

if [ -z "$BUNDLE" ]; then
	echo "usage: $0 <bundle.gputrace> [--set used|unused|all] [--names-only]" >&2
	exit 2
fi
case "$SET" in
used | unused | all) ;;
*)
	echo "--set must be used, unused or all (got '$SET')" >&2
	exit 2
	;;
esac
if [ "$NAMES_ONLY" = "1" ] && [ "$SET" = "all" ]; then
	echo "--names-only needs --set used or --set unused (a merged list is ambiguous)" >&2
	exit 2
fi
if [ ! -d "$BUNDLE" ]; then
	echo "ERROR: $BUNDLE is not a .gputrace bundle directory" >&2
	exit 2
fi

BUNDLE="${BUNDLE%/}"

# Resolve the two archives. `delta-device-resources-*` is skipped: it holds the
# incremental resource updates, not the function table.
used_file=""
unused_file=""
for f in "$BUNDLE"/device-resources-0x*; do
	[ -e "$f" ] || continue
	used_file="$f"
done
for f in "$BUNDLE"/unused-device-resources-0x*; do
	[ -e "$f" ] || continue
	unused_file="$f"
done

if [ -z "$used_file" ]; then
	echo "ERROR: no device-resources-0x* file in $BUNDLE." >&2
	echo "  Either this is not a Metal GPU trace, or the bundle layout changed." >&2
	echo "  Check what it holds:  ls '$BUNDLE' | grep -v '^MTL'" >&2
	exit 3
fi

# check_magic <file> — the archives start with the four bytes "MTSP".
check_magic() {
	local magic
	magic=$(head -c 4 "$1" 2>/dev/null)
	if [ "$magic" != "MTSP" ]; then
		echo "ERROR: $(basename "$1") does not start with the expected MTSP magic (got '${magic}')." >&2
		echo "  The capture container changed format; this parser is out of date." >&2
		return 1
	fi
	return 0
}

# parse_functions <file> <label> <require-nonempty:0|1>
# Writes names to $names_out, object ids to $opaque_out, and the number of
# records whose value could not be read to $unresolved_out. Returns 1 on a
# structural violation, having said which one.
#
# A record is `CSuwuw` / `function` / <value>, but the container also emits raw
# pointer bytes that sometimes decode as a short printable run, so the value is
# not always the very next string:
#
#     CSuwuw · function · @!Ac · vn_copybfloat16bfloat16 · @!Ac · CiulSl
#
# The scan therefore looks ahead a few strings, skipping anything not shaped
# like a name or an object id, and stops at the next record (`CSuwuw` or
# `function`). A record the scan cannot resolve is counted, never silently
# dropped: "39 records, 33 named" and "39 records, 33 named, 6 unreadable" are
# different facts about the same window.
names_out=""
opaque_out=""
marker_count=0
unresolved_out=0
parse_functions() {
	local file="$1" label="$2" require="$3"
	local raw values markers

	check_magic "$file" || return 1

	raw=$(strings -n 4 "$file" | awk '
		$0 == "function" {
			if (pend > 0) unresolved++
			pend = 4
			next
		}
		pend > 0 {
			# Container type tags ("CSuwuw", "CiulSl", "Ciuli", ...) are a C
			# followed by field codes; hitting one means this record had no
			# readable value, so stop rather than adopt the next record"s name.
			if ($0 ~ /^C[iulwSbtU@<>3]+$/) { unresolved++; pend = 0; next }
			if ($0 ~ /^([0-9A-F]{12,}|[A-Za-z_][A-Za-z0-9_]*)$/) { print; pend = 0; next }
			pend--
			if (pend == 0) unresolved++
			next
		}
		END { if (pend > 0) unresolved++; print "#unresolved " unresolved + 0 }
	')

	markers=$(strings -n 4 "$file" | grep -cx 'function')
	marker_count="$markers"
	unresolved_out=$(printf '%s\n' "$raw" | sed -n 's/^#unresolved //p')
	unresolved_out=${unresolved_out:-0}
	values=$(printf '%s\n' "$raw" | grep -v '^#unresolved ')

	if [ "$markers" -eq 0 ]; then
		if [ "$require" = "1" ]; then
			echo "ERROR: $(basename "$file") holds no 'function' records at all." >&2
			echo "  A captured decode window always references functions, so this is a" >&2
			echo "  layout change, not an empty result. Inspect it with:" >&2
			echo "    strings -n 4 '$file' | sort | uniq -c | sort -rn | head" >&2
			return 1
		fi
		names_out=""
		opaque_out=""
		return 0
	fi

	# Most records unreadable means the record shape moved, not that this
	# window happened to use anonymous functions.
	if [ "$unresolved_out" -gt $((markers / 2)) ]; then
		echo "ERROR: $(basename "$file"): $unresolved_out of $markers function records unreadable." >&2
		echo "  That is most of them — the record layout changed. Inspect it with:" >&2
		echo "    strings -n 4 '$file' | grep -A3 -x function | head -30" >&2
		return 1
	fi

	opaque_out=$(printf '%s\n' "$values" | grep -E '^[0-9A-F]{12,}$' | sort -u)
	names_out=$(printf '%s\n' "$values" | grep -vE '^[0-9A-F]{12,}$' | sort -u)

	if [ -z "$names_out" ]; then
		echo "ERROR: $(basename "$file"): $markers function records, none with a readable name." >&2
		echo "  Names are no longer stored inline; this parser cannot report the set." >&2
		return 1
	fi
	return 0
}

count_lines() {
	if [ -z "$1" ]; then echo 0; else printf '%s\n' "$1" | wc -l | tr -d ' '; fi
}

emit_set() {
	local label="$1" names="$2" opaque="$3" markers="$4" file="$5" unresolved="$6"
	if [ "$NAMES_ONLY" = "1" ]; then
		[ -n "$names" ] && printf '%s\n' "$names"
		return 0
	fi
	local tail=""
	[ "$unresolved" -gt 0 ] && tail=", $unresolved unreadable"
	echo ""
	echo "$label ($(basename "$file")): $markers function records — $(count_lines "$names") named, $(count_lines "$opaque") stored by object id$tail"
	[ -n "$names" ] && printf '%s\n' "$names" | sed 's/^/  /'
	return 0
}

rc=0

if [ "$SET" = "used" ] || [ "$SET" = "all" ]; then
	parse_functions "$used_file" "used" 1 || exit 3
	used_names="$names_out"
	used_opaque="$opaque_out"
	used_markers="$marker_count"
	used_unresolved="$unresolved_out"
fi
if [ "$SET" = "unused" ] || [ "$SET" = "all" ]; then
	if [ -z "$unused_file" ]; then
		if [ "$NAMES_ONLY" = "1" ]; then
			echo "ERROR: $BUNDLE has no unused-device-resources-0x* file." >&2
			echo "  Nothing to list; a silent empty list would read as 'nothing unused'." >&2
			exit 3
		fi
		unused_absent=1
	else
		# Zero unused functions is a legitimate outcome here (unlike the
		# referenced set), so an empty parse is reported, not rejected.
		parse_functions "$unused_file" "unused" 0 || exit 3
		unused_absent=0
		unused_names="$names_out"
		unused_opaque="$opaque_out"
		unused_markers="$marker_count"
		unused_unresolved="$unresolved_out"
	fi
fi

if [ "$NAMES_ONLY" = "0" ]; then
	echo "bundle: $BUNDLE"
fi

if [ "$SET" = "used" ] || [ "$SET" = "all" ]; then
	emit_set "referenced by the window" "$used_names" "$used_opaque" "$used_markers" \
		"$used_file" "$used_unresolved"
fi
if [ "$SET" = "unused" ] || [ "$SET" = "all" ]; then
	if [ "$unused_absent" = "1" ]; then
		echo ""
		echo "recorded unused: no unused-device-resources-0x* file in this bundle"
	else
		emit_set "recorded unused by the capture layer" "$unused_names" "$unused_opaque" \
			"$unused_markers" "$unused_file" "$unused_unresolved"
	fi
fi

exit $rc
