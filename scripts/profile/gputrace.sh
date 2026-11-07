#!/usr/bin/env bash
# Read a .gputrace bundle offline: the Metal functions it names, a diff of two, or a summary.
#
# A bundle holds two function/pipeline archives in Apple's "MTSP" container
# (docs/PROFILING.md §5):
#
#   device-resources-0x<addr>         referenced by the captured window
#   unused-device-resources-0x<addr>  recorded by the capture layer as unused
#
# Inside, each function record is the literal string `function` followed by its
# name, so a `function`-marker scan recovers the list. Some records carry an
# opaque 16-hex object id instead of a name: those are counted and reported
# separately, because "33 functions, 6 of which this parser could not name" and
# "33 functions" are different facts.
#
# The layout is Apple's and not a stable contract, so every structural assumption
# is checked and a violation is a hard error naming what broke; an empty list
# must never be mistaken for "no kernels ran". The checks:
#
#   - the container magic is MTSP;
#   - the referenced set holds at least one `function` record;
#   - at most half the records are unreadable;
#   - at least one record yields a readable name.
#
# Individual unreadable records are not fatal: they are counted and printed.
#
# Modes:
#   kernels  list the functions of one bundle (--set used|unused|all, --names-only)
#   diff     the function sets of two bundles, in both directions (--set used|unused)
#   summary  what a bundle captured, its size, and whether Xcode has profiled it
#
# Usage:  bash scripts/profile/gputrace.sh kernels <bundle.gputrace> [--set used|unused|all] [--names-only]
#         bash scripts/profile/gputrace.sh diff <a.gputrace> <b.gputrace> [--set used|unused]
#         bash scripts/profile/gputrace.sh summary <bundle.gputrace> [<bundle.gputrace> ...]
# Exit:   0 done; 1 diff could not make its scratch directory; 2 usage or no such bundle; 3 bundle layout not understood
# Set:    -uo — a failed parse is a status the code reads and reports, not a reason to stop
# Not a mode of: profile/capture.sh — that one records a bundle on the GPU; this one only reads a finished one

set -uo pipefail

PROG="$0"

# check_magic <file>: the archives start with the four bytes "MTSP".
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

# parse_functions <file> <require-nonempty:0|1>
# Writes names to $names_out, object ids to $opaque_out, the number of function
# records to $marker_count and the number whose value could not be read to
# $unresolved_out. Returns 1 on a structural violation, having said which one.
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
# dropped.
names_out=""
opaque_out=""
marker_count=0
unresolved_out=0
parse_functions() {
	local file="$1" require="$2"
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

# emit_set <label> <names> <opaque> <markers> <file> <unresolved> <names-only:0|1>
emit_set() {
	local label="$1" names="$2" opaque="$3" markers="$4" file="$5" unresolved="$6" names_only="$7"
	if [ "$names_only" = "1" ]; then
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

# list_functions <bundle> <used|unused|all> <names-only:0|1>: the kernels report.
# Returns 2 for a path that is no bundle, 3 for a layout that is not understood.
list_functions() {
	local given="$1" set="$2" names_only="$3"
	local bundle="${given%/}" used_file="" unused_file="" f
	local used_names="" used_opaque="" used_markers=0 used_unresolved=0
	local unused_names="" unused_opaque="" unused_markers=0 unused_unresolved=0 unused_absent=0

	if [ ! -d "$given" ]; then
		echo "ERROR: $given is not a .gputrace bundle directory" >&2
		return 2
	fi

	# `delta-device-resources-*` is skipped: it holds the incremental resource
	# updates, not the function table.
	for f in "$bundle"/device-resources-0x*; do
		[ -e "$f" ] || continue
		used_file="$f"
	done
	for f in "$bundle"/unused-device-resources-0x*; do
		[ -e "$f" ] || continue
		unused_file="$f"
	done

	if [ -z "$used_file" ]; then
		echo "ERROR: no device-resources-0x* file in $bundle." >&2
		echo "  Either this is not a Metal GPU trace, or the bundle layout changed." >&2
		echo "  Check what it holds:  ls '$bundle' | grep -v '^MTL'" >&2
		return 3
	fi

	if [ "$set" = "used" ] || [ "$set" = "all" ]; then
		parse_functions "$used_file" 1 || return 3
		used_names="$names_out"
		used_opaque="$opaque_out"
		used_markers="$marker_count"
		used_unresolved="$unresolved_out"
	fi
	if [ "$set" = "unused" ] || [ "$set" = "all" ]; then
		if [ -z "$unused_file" ]; then
			if [ "$names_only" = "1" ]; then
				echo "ERROR: $bundle has no unused-device-resources-0x* file." >&2
				echo "  Nothing to list; a silent empty list would read as 'nothing unused'." >&2
				return 3
			fi
			unused_absent=1
		else
			# Zero unused functions is legitimate (unlike the referenced set), so an
			# empty parse is reported, not rejected.
			parse_functions "$unused_file" 0 || return 3
			unused_names="$names_out"
			unused_opaque="$opaque_out"
			unused_markers="$marker_count"
			unused_unresolved="$unresolved_out"
		fi
	fi

	if [ "$names_only" = "0" ]; then
		echo "bundle: $bundle"
	fi
	if [ "$set" = "used" ] || [ "$set" = "all" ]; then
		emit_set "referenced by the window" "$used_names" "$used_opaque" "$used_markers" \
			"$used_file" "$used_unresolved" "$names_only"
	fi
	if [ "$set" = "unused" ] || [ "$set" = "all" ]; then
		if [ "$unused_absent" = "1" ]; then
			echo ""
			echo "recorded unused: no unused-device-resources-0x* file in this bundle"
		else
			emit_set "recorded unused by the capture layer" "$unused_names" "$unused_opaque" \
				"$unused_markers" "$unused_file" "$unused_unresolved" "$names_only"
		fi
	fi
	return 0
}

mode_kernels() {
	local bundle="" set="all" names_only=0
	while [ $# -gt 0 ]; do
		case "$1" in
		--set)
			if [ $# -lt 2 ]; then
				echo "--set needs used|unused|all" >&2
				return 2
			fi
			set="$2"
			shift 2
			;;
		--names-only)
			names_only=1
			shift
			;;
		-h | --help)
			echo "usage: $PROG kernels <bundle.gputrace> [--set used|unused|all] [--names-only]"
			return 0
			;;
		-*)
			echo "unknown argument: $1" >&2
			return 2
			;;
		*)
			if [ -n "$bundle" ]; then
				echo "only one bundle at a time (got '$bundle' and '$1')" >&2
				return 2
			fi
			bundle="$1"
			shift
			;;
		esac
	done

	if [ -z "$bundle" ]; then
		echo "usage: $PROG kernels <bundle.gputrace> [--set used|unused|all] [--names-only]" >&2
		return 2
	fi
	case "$set" in
	used | unused | all) ;;
	*)
		echo "--set must be used, unused or all (got '$set')" >&2
		return 2
		;;
	esac
	if [ "$names_only" = "1" ] && [ "$set" = "all" ]; then
		echo "--names-only needs --set used or --set unused (a merged list is ambiguous)" >&2
		return 2
	fi
	list_functions "$bundle" "$set" "$names_only"
}

mode_diff() {
	local a="" b="" set="used"
	while [ $# -gt 0 ]; do
		case "$1" in
		--set)
			if [ $# -lt 2 ]; then
				echo "--set needs used|unused" >&2
				return 2
			fi
			set="$2"
			shift 2
			;;
		-h | --help)
			echo "usage: $PROG diff <a.gputrace> <b.gputrace> [--set used|unused]"
			return 0
			;;
		-*)
			echo "unknown argument: $1" >&2
			return 2
			;;
		*)
			if [ -z "$a" ]; then
				a="$1"
			elif [ -z "$b" ]; then
				b="$1"
			else
				echo "expected exactly two bundles (got a third: $1)" >&2
				return 2
			fi
			shift
			;;
		esac
	done

	if [ -z "$a" ] || [ -z "$b" ]; then
		echo "usage: $PROG diff <a.gputrace> <b.gputrace> [--set used|unused]" >&2
		return 2
	fi
	case "$set" in
	used | unused) ;;
	*)
		echo "--set must be used or unused (got '$set')" >&2
		return 2
		;;
	esac

	# Global, not local: the EXIT trap runs after this function has returned.
	tmp=$(mktemp -d "${TMPDIR:-/tmp}/gputrace-diff.XXXXXX") || return 1
	trap 'rm -rf "$tmp"' EXIT

	# A diff computed from a half-read bundle is worse than no diff, so a bundle
	# that cannot be listed ends the run.
	list_sorted() {
		local bundle="$1" out="$2"
		if ! list_functions "$bundle" "$set" 1 >"$out"; then
			echo "ERROR: could not list functions for $bundle (see above) — no diff produced." >&2
			return 1
		fi
		sort -u "$out" -o "$out"
	}
	list_sorted "$a" "$tmp/a" || return 3
	list_sorted "$b" "$tmp/b" || return 3

	local only_a only_b shared
	only_a=$(comm -23 "$tmp/a" "$tmp/b")
	only_b=$(comm -13 "$tmp/a" "$tmp/b")
	shared=$(comm -12 "$tmp/a" "$tmp/b" | wc -l | tr -d ' ')

	echo "set:    $set functions"
	echo "A:      $a ($(wc -l <"$tmp/a" | tr -d ' ') named)"
	echo "B:      $b ($(wc -l <"$tmp/b" | tr -d ' ') named)"
	echo "shared: $shared"
	echo ""
	echo "only in A ($(count_lines "$only_a")):"
	[ -n "$only_a" ] && printf '%s\n' "$only_a" | sed 's/^/  /'
	echo ""
	echo "only in B ($(count_lines "$only_b")):"
	[ -n "$only_b" ] && printf '%s\n' "$only_b" | sed 's/^/  /'

	if [ -z "$only_a" ] && [ -z "$only_b" ]; then
		echo ""
		echo "identical named function sets"
	fi
	# Names the parser could not read are per-bundle and not part of either list;
	# kernels reports their count for each bundle.
	return 0
}

# Shape checks are pure parameter expansion on purpose: `printf ... | grep -q`
# under `set -o pipefail` reports failure when grep exits before the writer
# finishes, which would turn a well-formed name into "does not follow the
# convention".
is_digits() {
	case "$1" in
	"") return 1 ;;
	*[!0-9]*) return 1 ;;
	*) return 0 ;;
	esac
}

mode_summary() {
	if [ $# -eq 0 ]; then
		echo "usage: $PROG summary <bundle.gputrace> [<bundle.gputrace> ...]" >&2
		return 2
	fi
	case "${1:-}" in
	-h | --help)
		echo "usage: $PROG summary <bundle.gputrace> [<bundle.gputrace> ...]"
		return 0
		;;
	esac

	local rc=0 raw bundle base stem name_rest hms ymd toks codec model tok_count
	local missing dev_res f total cap entries prof prof_h meta frames unused
	for raw in "$@"; do
		bundle="${raw%/}"
		if [ ! -d "$bundle" ]; then
			echo "ERROR: $bundle is not a .gputrace bundle directory" >&2
			rc=2
			continue
		fi

		base=$(basename "$bundle")
		echo "bundle:   $bundle"

		# Parsed right-anchored: <model>-<codec>-<N>tok-<YYYYMMDD>-<HHMMSS>. The
		# model tag may be absent in bundles captured before the harness added it,
		# which is reported as such.
		stem="${base%.gputrace}"
		if [ "$stem" = "$base" ]; then
			echo "  name:   '$base' does not end in .gputrace — not a trace bundle name"
		fi
		name_rest="$stem"
		hms="${name_rest##*-}"
		name_rest="${name_rest%-*}"
		ymd="${name_rest##*-}"
		name_rest="${name_rest%-*}"
		toks="${name_rest##*-}"
		name_rest="${name_rest%-*}"
		codec="${name_rest##*-}"
		model="${name_rest%-*}"
		[ "$model" = "$codec" ] && model=""

		tok_count="${toks%tok}"
		if is_digits "$hms" && [ "${#hms}" -eq 6 ] &&
			is_digits "$ymd" && [ "${#ymd}" -eq 8 ] &&
			[ "$tok_count" != "$toks" ] && is_digits "$tok_count"; then
			echo "  model:  ${model:-<not in name — captured before the harness stamped it>}"
			echo "  codec:  $codec"
			echo "  prompt: $tok_count tokens"
			echo "  when:   $ymd $hms"
		else
			echo "  name:   '$base' does not follow the harness convention"
			echo "          (<model>-<codec>-<prompt>tok-<YYYYMMDD>-<HHMMSS>.gputrace);"
			echo "          model, codec and prompt size are unknown for this bundle."
		fi

		missing=""
		[ -f "$bundle/capture" ] || missing="$missing capture"
		[ -f "$bundle/metadata" ] || missing="$missing metadata"
		dev_res=""
		for f in "$bundle"/device-resources-0x*; do
			[ -e "$f" ] && dev_res="$f"
		done
		[ -n "$dev_res" ] || missing="$missing device-resources-0x*"
		if [ -n "$missing" ]; then
			echo "ERROR: $bundle is missing:$missing" >&2
			echo "  Either this is not a Metal GPU trace, or the bundle layout changed." >&2
			rc=3
			echo ""
			continue
		fi

		total=$(du -sh "$bundle" 2>/dev/null | awk '{print $1}')
		cap=$(du -h "$bundle/capture" 2>/dev/null | awk '{print $1}')
		entries=$(find "$bundle" -maxdepth 1 -mindepth 1 | wc -l | tr -d ' ')
		echo "  size:   ${total:-unknown} total, ${cap:-unknown} command stream, $entries entries"

		# A capture holds no timing; only Xcode's GUI Profile replay writes a
		# .gpuprofiler_raw, so its absence is the normal state.
		prof=$(find "$bundle" -maxdepth 2 -name '*gpuprofiler_raw*' -print -quit 2>/dev/null)
		if [ -n "$prof" ]; then
			prof_h=$(du -h "$prof" 2>/dev/null | awk '{print $1}')
			echo "  timing: PROFILED — $(basename "$prof") (${prof_h:-size unknown}) holds a replay's counters"
		else
			echo "  timing: NOT PROFILED — no .gpuprofiler_raw (the normal state: only"
			echo "          Xcode's GUI Profile replay writes one). What this bundle does"
			echo "          answer, offline:"
			echo "            bash scripts/profile/gputrace.sh kernels '$bundle'"
			echo "          For wall-clock GPU timing use Metal System Trace instead —"
			echo "          docs/PROFILING.md §5."
		fi

		meta=$(plutil -p "$bundle/metadata" 2>&1)
		if [ $? -ne 0 ] || [ -z "$meta" ]; then
			echo "ERROR: $bundle/metadata is not a readable property list:" >&2
			head -3 <<<"$meta" | sed 's/^/    /' >&2
			rc=3
			echo ""
			continue
		fi
		frames=$(printf '%s\n' "$meta" | awk -F'=> ' '/captured_frames_count/ {print $2}')
		echo "  frames: ${frames:-<no captured_frames_count key — metadata layout changed>}"
		unused=$(printf '%s\n' "$meta" | awk -F'=> ' '/unused[A-Za-z]*Count/ && $2+0 > 0 {
			gsub(/[" ]/, "", $1); sub(/.*\./, "", $1); printf "%s=%s ", $1, $2 }')
		[ -n "$unused" ] && echo "  unused: $unused"

		echo ""
	done
	return "$rc"
}

if [ $# -eq 0 ]; then
	echo "usage: $PROG {kernels|diff|summary} <bundle.gputrace>..." >&2
	exit 2
fi
mode="$1"
shift
case "$mode" in
kernels) mode_kernels "$@" ;;
diff) mode_diff "$@" ;;
summary) mode_summary "$@" ;;
*)
	echo "unknown mode: $mode (expected kernels, diff or summary)" >&2
	echo "usage: $PROG {kernels|diff|summary} <bundle.gputrace>..." >&2
	exit 2
	;;
esac
exit $?
