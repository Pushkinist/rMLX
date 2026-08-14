#!/usr/bin/env bash
# Summarise a .gputrace bundle: what it is, how big it is, and — the part that
# is otherwise invisible — whether it has been profiled in Xcode yet.
#
# A bundle is several GB of opaque blobs. Two facts are worth having before
# opening one:
#
#   1. what it captured. The harness encodes that in the name
#      (<model>-<codec>-<prompt>tok-<timestamp>.gputrace), so this reads it back
#      rather than guessing from the contents.
#   2. whether a .gpuprofiler_raw is present. A capture holds no timing at all;
#      only Xcode's GUI Profile replay ever writes that file, and it cannot be
#      driven headlessly. Reported because its absence is the normal state and
#      people mistake it for a broken capture — for wall-clock GPU timing use
#      Metal System Trace instead (docs/PROFILING.md §5).
#
# The bundle layout is Apple's and is not a stable contract, so the structural
# reads (metadata plist, capture stream, resource archive) are checked and a
# violation is a hard error naming what broke — never a summary with silently
# missing fields.
#
# Usage:
#   bash scripts/gputrace_summary.sh <bundle.gputrace> [<bundle.gputrace> ...]
#   bash scripts/gputrace_summary.sh .rmlx/traces/*.gputrace
#
# Exit: 0 = summarised, 2 = usage / no such bundle, 3 = bundle layout not understood

set -uo pipefail

if [ $# -eq 0 ]; then
	echo "usage: $0 <bundle.gputrace> [<bundle.gputrace> ...]" >&2
	exit 2
fi
case "${1:-}" in
-h | --help)
	echo "usage: $0 <bundle.gputrace> [<bundle.gputrace> ...]"
	exit 0
	;;
esac

rc=0

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

for raw in "$@"; do
	bundle="${raw%/}"
	if [ ! -d "$bundle" ]; then
		echo "ERROR: $bundle is not a .gputrace bundle directory" >&2
		rc=2
		continue
	fi

	base=$(basename "$bundle")
	echo "bundle:   $bundle"

	# --- name fields --------------------------------------------------------
	# Parsed right-anchored: <model>-<codec>-<N>tok-<YYYYMMDD>-<HHMMSS>. The
	# model tag may be absent in bundles captured before the harness added it,
	# which is reported as such — an empty model field is never printed as if
	# it had been read.
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

	# --- structure ----------------------------------------------------------
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

	# --- profiled yet? ------------------------------------------------------
	prof=$(find "$bundle" -maxdepth 2 -name '*gpuprofiler_raw*' -print -quit 2>/dev/null)
	if [ -n "$prof" ]; then
		prof_h=$(du -h "$prof" 2>/dev/null | awk '{print $1}')
		echo "  timing: PROFILED — $(basename "$prof") (${prof_h:-size unknown}) holds a replay's counters"
	else
		echo "  timing: NOT PROFILED — no .gpuprofiler_raw (the normal state: only"
		echo "          Xcode's GUI Profile replay writes one). What this bundle does"
		echo "          answer, offline:"
		echo "            bash scripts/gputrace_kernels.sh '$bundle'"
		echo "          For wall-clock GPU timing use Metal System Trace instead —"
		echo "          docs/PROFILING.md §5."
	fi

	# --- metadata -----------------------------------------------------------
	meta=$(plutil -p "$bundle/metadata" 2>&1)
	if [ $? -ne 0 ] || [ -z "$meta" ]; then
		echo "ERROR: $bundle/metadata is not a readable property list:" >&2
		printf '%s\n' "$meta" | head -3 | sed 's/^/    /' >&2
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

exit $rc
