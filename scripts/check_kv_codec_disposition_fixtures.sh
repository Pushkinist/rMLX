#!/usr/bin/env bash
# scripts/check_kv_codec_disposition_fixtures.sh — recall test for
# `check_kv_codec_disposition.sh`.
#
# That gate has seven rules and two exit codes, and every one of them is one regex
# edit away from matching nothing and going permanently green. This repo has
# shipped three gates that were each individually unable to fail, so a gate's
# detection power is measured here rather than assumed.
#
# HOW
#   `scripts/fixtures/kv_codec_disposition/base/` is a synthetic scan root — a
#   stand-in `main.rs`, a stand-in `KV_QUANT.md`, and a pre-captured
#   `manifest.raw` — that the gate passes clean. Each case copies it and makes
#   exactly ONE edit, so the rule that fires is attributable to that edit and
#   nothing else. The gate reads the scan root through its normal code path;
#   only the manifest's source differs from a production run.
#
# WHY THE EXPECTED MESSAGE IS CHECKED, NOT JUST THE EXIT CODE
#   All seven rules exit 1, so a corpus that asserts only the exit code cannot
#   tell "RULE 3 fired" from "RULE 3 is dead and the fixture happened to trip
#   RULE 1" — which is how a gate gets redder for the wrong reason and still
#   looks like it works. The two manifest cases assert
#   exit 2 for the same reason in the other direction: "nothing was looked at"
#   must never read as "a violation was found", let alone as green.
#
# Exit 0 = every fixture produced its expected exit code and message.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_kv_codec_disposition.sh"
BASE="$ROOT/scripts/fixtures/kv_codec_disposition/base"

for f in main.rs KV_QUANT.md manifest.raw; do
    [ -f "$BASE/$f" ] || {
        echo "check-kv-codec-disposition fixtures: missing $BASE/$f" >&2
        exit 1
    }
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Rewrite a file in place. `sed -i` is not portable across BSD and GNU sed.
edit() {
    local file="$1" expr="$2"
    sed "$expr" "$file" >"$file.new" && mv -f "$file.new" "$file"
}

# Build the scan root for one case: the base, plus exactly one edit.
build_case() {
    local name="$1" dir="$2"
    mkdir -p "$dir"
    cp -f "$BASE/main.rs" "$BASE/KV_QUANT.md" "$BASE/manifest.raw" "$dir/"

    case "$name" in
        clean) ;;
        rule1_inert_named_outside_block)
            # fixinert moves out of the INERT block and in beside the live codec.
            edit "$dir/main.rs" \
                's/^      fixinert, fixinert2\.$/      fixinert2./; s/^      fixlive\.";$/      fixlive, fixinert.";/'
            ;;
        rule2_live_named_inside_inert_block)
            edit "$dir/main.rs" 's/^      fixinert, fixinert2\.$/      fixinert, fixinert2, fixlive./'
            ;;
        rule3_inert_without_a_banner)
            edit "$dir/KV_QUANT.md" '/^> \*\*INERT on this build\*\* — `fixinert2`/d'
            ;;
        rule4_live_named_in_a_banner)
            edit "$dir/KV_QUANT.md" \
                's/^\(> \*\*INERT on this build\*\* — `fixinert`.*\)$/\1 So does fixlive./'
            ;;
        rule5_banner_buried_in_the_section)
            edit "$dir/KV_QUANT.md" \
                's/^### fixinert$/### fixinert\n\nOne paragraph of prose.\n\nA second paragraph, pushing the banner past the third line./'
            ;;
        rule6_kv_quant_inline_help)
            edit "$dir/main.rs" 's/help = KV_QUANT_HELP/help = "KV cache quantization codec."/'
            ;;
        rule6_kv_bits_inline_help)
            edit "$dir/main.rs" 's/long_help = KV_BITS_LONG_HELP/long_help = "Bit-width alias."/'
            ;;
        rule6_kv_preset_inline_help)
            edit "$dir/main.rs" 's/long_help = KV_PRESET_LONG_HELP/long_help = "Named preset."/'
            ;;
        rule1_inert_named_only_in_preset_help)
            # The preset help's INERT block loses its only line; fixinert stays
            # named in that same constant's Presets list. Nothing fires unless
            # the --kv-preset constant is read as a surface of its own.
            edit "$dir/main.rs" '/^      fixslow resolves to fixinert\.$/d'
            ;;
        rule7_ratio_written_into_the_help)
            edit "$dir/main.rs" 's/^      fixlive\.";$/      fixlive, at 1.05x the baseline.";/'
            ;;
        manifest_truncated)
            # One codec line lost; the END sentinel still claims four.
            edit "$dir/manifest.raw" '/	fixlive	/d'
            ;;
        manifest_unknown_class)
            edit "$dir/manifest.raw" 's/	LIVE	/	WOBBLE	/'
            ;;
        *)
            echo "check-kv-codec-disposition fixtures: no such case '$name'" >&2
            return 1
            ;;
    esac
}

# case | expected-exit | expected-substring | what it proves
CASES=(
    "clean|0|OK: 4 KV codecs classified|the synthetic scan root passes as itself"

    "rule1_inert_named_outside_block|1|RULE 1  'fixinert'|an inert codec listed beside the live ones is caught"
    "rule2_live_named_inside_inert_block|1|RULE 2  'fixlive'|a live codec inside the help's INERT block is caught"
    "rule3_inert_without_a_banner|1|RULE 3  'fixinert2'|an inert codec whose docs banner was removed is caught"
    "rule4_live_named_in_a_banner|1|RULE 4  'fixlive'|a live codec named in a docs banner is caught"
    "rule5_banner_buried_in_the_section|1|RULE 5|a banner pushed 4 lines below its heading is caught"
    "rule6_kv_quant_inline_help|1|RULE 6|a --kv-quant argument with its own help text is caught"
    "rule6_kv_bits_inline_help|1|RULE 6|a --kv-bits argument with its own long help is caught"
    "rule6_kv_preset_inline_help|1|RULE 6|a --kv-preset argument with its own long help is caught"
    "rule1_inert_named_only_in_preset_help|1|RULE 1  'fixinert'|the --kv-preset help is read, not just the --kv-quant one"
    "rule7_ratio_written_into_the_help|1|RULE 7|a resident-KV ratio typed into the help is caught"

    "manifest_truncated|2|manifest truncated|a short manifest is an environment error, not a violation"
    "manifest_unknown_class|2|unknown disposition class|a class the gate cannot read is an environment error"
)

FAILED=0
PASSED=0

for case in "${CASES[@]}"; do
    IFS='|' read -r name want_exit want_msg what <<<"$case"
    dir="$WORK/$name"
    if ! build_case "$name" "$dir"; then
        FAILED=$((FAILED + 1))
        continue
    fi
    out=$("$GATE" "$dir" 2>&1)
    got=$?

    if [ "$got" -eq "$want_exit" ] && printf '%s' "$out" | grep -qF -- "$want_msg"; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-38s exit=%s — %s\n' "$name" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-38s exit=%s (want %s, want message "%s") — %s\n' \
            "$name" "$got" "$want_exit" "$want_msg" "$what"
        printf '%s\n' "$out" | sed 's/^/       | /'
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "check-kv-codec-disposition fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-kv-codec-disposition fixtures: ok ($PASSED cases)"
