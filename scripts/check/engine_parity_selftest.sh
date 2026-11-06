#!/usr/bin/env bash
# Recall test for the four engine-parity modes: one case group per mode, rule and exit code.
#
# Recall test for: check/engine_parity.sh kv-byte-model kv-codec-disposition kv-boundary-default spec-metrics
#
# Each mode holds a restatement of an engine fact to the engine, and each is one regex edit away
# from matching nothing and going permanently green. Every case makes one edit to a manifest or
# to a synthetic scan root and asserts the literal exit code and the reason: a gate that
# refuses for the wrong reason stops refusing when that reason moves, and "nothing was looked at"
# (exit 2) must never read as "a violation was found" (exit 1) or as green.
#
# The four modes are named in one place each, the functions `gate_byte_model`,
# `gate_codec_disposition`, `gate_boundary_default` and `gate_spec_metrics`. The byte-model
# cases need no byte count: the passing manifest is produced by `perf_ceiling.py --byte-model`
# from a skeleton. The codec-disposition cases copy `scripts/fixtures/kv_codec_disposition/base/`
# and make one edit. The boundary and spec-metric modes read fixed repo paths, so each case runs
# a copy of the tool inside a scratch tree holding the files it reads. Two cases in the
# byte-model group run the tool with no mode and with an unknown one.
#
# Usage:  bash scripts/check/engine_parity_selftest.sh [<kv-byte-model|kv-codec-disposition|
#         kv-boundary-default|spec-metrics>]
#         no group runs the cases of all four
# Exit:   0 every case held; 1 a case did not, or none ran; 2 an unknown group, or a fixture
#         that could not be built

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/lib/selftest_support.sh
. "$REPO_ROOT/scripts/lib/selftest_support.sh"

GROUP="${1:-all}"
case "$GROUP" in
kv-byte-model) st_init check-kv-byte-model-parity-fixtures ;;
kv-codec-disposition) st_init check-kv-codec-disposition-fixtures ;;
kv-boundary-default) st_init check-kv-boundary-default-parity-fixtures ;;
spec-metrics) st_init check-spec-metric-parity-fixtures ;;
all) st_init engine-parity-selftest ;;
*)
    echo "usage: bash scripts/check/engine_parity_selftest.sh [<kv-byte-model|kv-codec-disposition|kv-boundary-default|spec-metrics>]" >&2
    exit 2
    ;;
esac

PERF_CEILING="$REPO_ROOT/scripts/profile/perf_ceiling.py"
CODEC_BASE="$REPO_ROOT/scripts/fixtures/kv_codec_disposition/base"
ENGINE_PARITY="$REPO_ROOT/scripts/check/engine_parity.sh"

# ---- the gates, one function each ------------------------------------------------------------

# gate_byte_model <manifest>: the byte-model gate over a pre-captured manifest.
gate_byte_model() {
    bash "$ENGINE_PARITY" kv-byte-model "$1"
}

# gate_codec_disposition <scan-root> [<tool-file>]: the disposition mode over a scan root,
# or an edited copy of the tool.
gate_codec_disposition() {
    bash "${2:-$ENGINE_PARITY}" kv-codec-disposition "$1"
}

# gate_boundary_default <scan-root>: the tool and the resolver module it imports, copied into the
# root and run there.
gate_boundary_default() {
    mkdir -p "$1/scripts/lib" "$1/scripts/check"
    cp -f "$ENGINE_PARITY" "$1/scripts/check/"
    [ -e "$1/scripts/lib/kv_boundary_default.py" ] ||
        cp -f "$REPO_ROOT/scripts/lib/kv_boundary_default.py" "$1/scripts/lib/"
    env -u RMLX_REPO_ROOT bash "$1/scripts/check/engine_parity.sh" kv-boundary-default
}

# gate_spec_metrics <scan-root>: the tool, copied into the root and run there.
gate_spec_metrics() {
    mkdir -p "$1/scripts/check"
    cp -f "$ENGINE_PARITY" "$1/scripts/check/"
    bash "$1/scripts/check/engine_parity.sh" spec-metrics
}

# ---- shared planting -------------------------------------------------------------------------

# replace_once <file> <old> <new>: plant one edit; an anchor that is not there ends the harness.
replace_once() {
    python3 -I -c '
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
if sys.argv[2] not in s:
    sys.exit(1)
p.write_text(s.replace(sys.argv[2], sys.argv[3], 1))
' "$@" || st_die "fixture anchor missing in $1"
}

# ---- kv-byte-model ---------------------------------------------------------------------------

# wrap <rows-file> <out-file> [<declared-count>]: rows between the manifest sentinels.
wrap() {
    local rows="$1" out="$2" declared="${3:-}"
    [ -n "$declared" ] || declared=$(awk 'END { print NR }' "$rows")
    {
        echo "KVBYTES-BEGIN"
        cat "$rows"
        printf 'KVBYTES-END\t%s\n' "$declared"
    } >"$out"
}

# bm <label> <manifest> <want-exit> <reason>
bm() {
    st_case "$1" "$3" --has "$4" -- gate_byte_model "$2"
}

byte_model_cases() {
    local w="$ST_WORK/bm"
    mkdir -p "$w"
    # Two codecs whose values differ under every arm the gate guards (a ring store and the
    # mirror-conditional family), both topologies, two head dimensions. The trailing value is a
    # placeholder that `--byte-model` replaces.
    cat >"$w/skeleton" <<'EOF'
KVBYTES	iso3_sym	0	4096	128	8	0
KVBYTES	iso3_sym	0	4096	256	2	0
KVFLOOR	iso3_sym	0	36	-
KVBYTES	iso3_sym	1	4096	128	8	0
KVBYTES	iso3_sym	1	4096	256	2	0
KVFLOOR	iso3_sym	1	36	-
KVBYTES	mixed_k8g64_v4g64	0	4096	128	8	0
KVBYTES	mixed_k8g64_v4g64	0	4096	256	2	0
KVFLOOR	mixed_k8g64_v4g64	0	36	-
KVBYTES	mixed_k8g64_v4g64	1	4096	128	8	0
KVBYTES	mixed_k8g64_v4g64	1	4096	256	2	0
KVFLOOR	mixed_k8g64_v4g64	1	36	-
EOF
    python3 "$PERF_CEILING" --byte-model <"$w/skeleton" >"$w/rows" ||
        st_die "could not build the byte-model fixture rows"
    local n_rows short_floor
    n_rows=$(awk 'END { print NR }' "$w/rows")
    [ "$n_rows" -eq 12 ] || st_die "skeleton produced $n_rows rows, expected 12"

    # A KVFLOOR row carries one entry per layer; a case that edits a tail entry is vacuous
    # against a two-index sample.
    short_floor=$(awk -F'\t' '$1 == "KVFLOOR" {
            n = split($5, v, ",")
            if (n != $4 + 0) { bad++ }
        } END { print bad + 0 }' "$w/rows")
    [ "$short_floor" -eq 0 ] ||
        st_die "$short_floor KVFLOOR row(s) are not one entry per layer"

    wrap "$w/rows" "$w/good.manifest"
    bm "clean manifest passes" "$w/good.manifest" 0 "byte-model rows"

    awk -F'\t' 'BEGIN { OFS = "\t" }
        NR == 1 && $1 == "KVBYTES" { $7 = $7 + 1 }
        { print }' "$w/rows" >"$w/rows.wrongbyte"
    wrap "$w/rows.wrongbyte" "$w/wrongbyte.manifest"
    bm "one wrong byte count" "$w/wrongbyte.manifest" 1 "disagrees with the engine"

    awk -F'\t' 'BEGIN { OFS = "\t" }
        $1 == "KVFLOOR" && $2 ~ /^mixed_/ && $3 == "0" {
            n = split($5, v, ",")
            v[1] = "k8v8"
            out = v[1]
            for (i = 2; i <= n; i++) { out = out "," v[i] }
            $5 = out
        }
        { print }' "$w/rows" >"$w/rows.wronghead"
    wrap "$w/rows.wronghead" "$w/wronghead.manifest"
    bm "wrong head-layer codec" "$w/wronghead.manifest" 1 "disagrees with the engine"

    awk -F'\t' 'BEGIN { OFS = "\t" }
        $1 == "KVFLOOR" && $2 ~ /^mixed_/ && $3 == "0" {
            n = split($5, v, ",")
            v[n] = "k8v4"
            out = v[1]
            for (i = 2; i <= n; i++) { out = out "," v[i] }
            $5 = out
        }
        { print }' "$w/rows" >"$w/rows.wrongtail"
    wrap "$w/rows.wrongtail" "$w/wrongtail.manifest"
    bm "wrong tail-layer codec" "$w/wrongtail.manifest" 1 "disagrees with the engine"

    awk -F'\t' '$3 == "0"' "$w/rows" >"$w/rows.onetopology"
    wrap "$w/rows.onetopology" "$w/onetopology.manifest"
    bm "single topology" "$w/onetopology.manifest" 2 "topologies"

    awk -F'\t' '$1 == "KVFLOOR" || $5 == "128"' "$w/rows" >"$w/rows.oneshape"
    wrap "$w/rows.oneshape" "$w/oneshape.manifest"
    bm "single head_dim" "$w/oneshape.manifest" 2 "shape"

    wrap "$w/rows" "$w/truncated.manifest" 999
    bm "truncated manifest" "$w/truncated.manifest" 2 "truncated"

    cat "$w/rows" >"$w/nobegin.manifest"
    bm "no BEGIN sentinel" "$w/nobegin.manifest" 2 "did not run"

    : >"$w/rows.empty"
    wrap "$w/rows.empty" "$w/empty.manifest" 0
    bm "empty manifest" "$w/empty.manifest" 2 "the manifest is empty"

    st_case "usage_no_mode" 2 --has "usage: engine_parity.sh" -- bash "$ENGINE_PARITY"
    st_case "usage_unknown_mode" 2 --has "usage: engine_parity.sh" -- bash "$ENGINE_PARITY" no-such-mode
}

# ---- kv-codec-disposition --------------------------------------------------------------------

CODEC_BASE_FILES=(main.rs KV_CODECS.md KV_ROTATION_CODECS.md KV_QUANT.md manifest.raw)

# Rewrite a file in place. `sed -i` is not portable across BSD and GNU sed.
edit() {
    local file="$1" expr="$2"
    sed "$expr" "$file" >"$file.new" && mv -f "$file.new" "$file"
}

# Build the scan root for one case: the base, plus exactly one edit.
build_case() {
    local name="$1" dir="$2" f
    mkdir -p "$dir"
    for f in "${CODEC_BASE_FILES[@]}"; do
        cp -f "$CODEC_BASE/$f" "$dir/"
    done

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
        edit "$dir/KV_ROTATION_CODECS.md" '/^> \*\*INERT on this build\*\* — `fixinert2`/d'
        ;;
    rule4_live_named_in_a_banner)
        edit "$dir/KV_CODECS.md" \
            's/^\(> \*\*INERT on this build\*\* — `fixinert`.*\)$/\1 So does fixlive./'
        ;;
    rule5_banner_buried_in_the_section)
        edit "$dir/KV_CODECS.md" \
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
        # The preset help's INERT block loses its only line; fixinert stays named in that same
        # constant's Presets list. Nothing fires unless the --kv-preset constant is read as a
        # surface of its own.
        edit "$dir/main.rs" '/^      fixslow resolves to fixinert\.$/d'
        ;;
    rule7_ratio_written_into_the_help)
        edit "$dir/main.rs" 's/^      fixlive\.";$/      fixlive, at 1.05x the baseline.";/'
        ;;
    rule7_ratio_with_multiplication_sign)
        # The spelling the ratio rows of docs/KV_ROTATION_CODECS.md use. A pattern keyed on `x`
        # alone is a gate a reviewer walks past by typing this.
        edit "$dir/main.rs" 's/^      fixlive\.";$/      fixlive, at 1.05× the baseline.";/'
        ;;
    rule9_listing_pointer_without_call_site)
        # The help still tells the operator to run --list-cache-types; the listing it points at
        # no longer prints. Nothing else notices.
        edit "$dir/main.rs" '/^        print_kv_quant_residency_table();$/d'
        ;;
    rule8_help_reference_without_a_readable_const)
        # clap renders this argument's long help and the gate cannot read it: no
        # `const FIXTURE_MISSING_LONG_HELP: &str` exists.
        printf '\n%s\n' \
            '        #[arg(long, long_help = other::FIXTURE_MISSING_LONG_HELP)]' \
            >>"$dir/main.rs"
        ;;
    rule7_ratio_in_a_constant_from_another_module)
        # The constant lives in a second file and is referenced from the first BY PATH, the
        # shape a bare-identifier scan reads past.
        cat >"$dir/other.rs" <<'OTHER'
pub(crate) const FIXTURE_MODULE_LONG_HELP: &str = "\
A help text that lives in its own module and measures 1.42x the baseline.";
OTHER
        printf '\n%s\n' \
            '        #[arg(long, long_help = other::FIXTURE_MODULE_LONG_HELP)]' \
            >>"$dir/main.rs"
        ;;
    rule7_integer_ratio)
        # No decimal point. `2x bf16` is a ratio and reads as one.
        edit "$dir/main.rs" 's/^      fixlive\.";$/      fixlive, at 2x the baseline.";/'
        ;;
    rule10_banner_in_an_unlisted_doc)
        # fixinert2's section moves to KV_QUANT.md, a doc the gate does not list.
        sed -n '/^### fixinert2$/,$p' "$dir/KV_ROTATION_CODECS.md" >>"$dir/KV_QUANT.md"
        edit "$dir/KV_ROTATION_CODECS.md" '/^### fixinert2$/,$d'
        ;;
    rule10_banner_in_an_unlisted_nested_doc)
        mkdir -p "$dir/models"
        sed -n '/^### fixinert2$/,$p' "$dir/KV_ROTATION_CODECS.md" >"$dir/models/NOTES.md"
        edit "$dir/KV_ROTATION_CODECS.md" '/^### fixinert2$/,$d'
        ;;
    rule10_empty_list)
        # The list is config inside the gate, so this case runs an edited copy of the gate.
        sed 's/^BANNER_DOCS=(.*)$/BANNER_DOCS=()/' "$ENGINE_PARITY" >"$dir/gate.sh"
        ;;
    rule10_duplicate_list_entry)
        sed 's/^BANNER_DOCS=(\(.*\))$/BANNER_DOCS=(\1 KV_CODECS.md)/' "$ENGINE_PARITY" >"$dir/gate.sh"
        ;;
    rule10_listed_doc_missing)
        rm -f "$dir/KV_ROTATION_CODECS.md"
        ;;
    rule11_codec_in_two_listed_docs)
        sed -n '/^### fixinert$/,/^exactly the paragraph/p' "$dir/KV_CODECS.md" \
            >>"$dir/KV_ROTATION_CODECS.md"
        ;;
    banner_moved_between_listed_docs)
        # fixinert's section moves to the other listed doc. The list, not one doc, is the
        # surface: this passes.
        sed -n '/^### fixinert$/,/^exactly the paragraph/p' "$dir/KV_CODECS.md" \
            >>"$dir/KV_ROTATION_CODECS.md"
        edit "$dir/KV_CODECS.md" '/^### fixinert$/,/^exactly the paragraph/d'
        ;;
    manifest_truncated)
        # One codec line lost; the END sentinel still claims four.
        edit "$dir/manifest.raw" '/	fixlive	/d'
        ;;
    manifest_unknown_class)
        edit "$dir/manifest.raw" 's/	LIVE	/	WOBBLE	/'
        ;;
    *)
        st_die "no such codec-disposition case '$name'"
        ;;
    esac
}

# case | expected-exit | expected-substring | what it proves
CODEC_CASES=(
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
    "rule7_ratio_with_multiplication_sign|1|RULE 7|the same ratio spelled with U+00D7 is caught"
    "rule7_integer_ratio|1|RULE 7|an integer ratio with no decimal point is caught"
    "rule8_help_reference_without_a_readable_const|1|RULE 8|clap help the gate cannot read is caught"
    "rule7_ratio_in_a_constant_from_another_module|1|RULE 7|a ratio in a help constant from a second module is caught"
    "rule9_listing_pointer_without_call_site|1|RULE 9|the help's --list-cache-types pointer with no call site is caught"
    "rule10_banner_in_an_unlisted_doc|1|RULE 10  docs/KV_QUANT.md|a banner moved to a doc the gate does not list is caught"
    "rule10_banner_in_an_unlisted_nested_doc|1|RULE 10  docs/models/NOTES|a banner in a subdirectory of docs/ is caught"
    "rule10_empty_list|2|BANNER_DOCS is empty|an empty banner-doc list is a config error, not a pass"
    "rule10_duplicate_list_entry|2|BANNER_DOCS lists KV_CODECS.md twice|a doc listed twice is a config error"
    "rule10_listed_doc_missing|2|a doc BANNER_DOCS lists|a listed doc that is gone is an environment error"
    "rule11_codec_in_two_listed_docs|1|RULE 11  'fixinert' is named in 2|a second banner for one codec, in the other listed doc, is caught"
    "banner_moved_between_listed_docs|0|OK: 4 KV codecs classified|a banner moved from one listed doc to the other passes"

    "manifest_truncated|2|manifest truncated|a short manifest is an environment error, not a violation"
    "manifest_unknown_class|2|unknown disposition class|a class the gate cannot read is an environment error"
)

codec_disposition_cases() {
    local f entry name want_exit want_msg what dir gate
    for f in "${CODEC_BASE_FILES[@]}"; do
        [ -f "$CODEC_BASE/$f" ] || st_die "missing $CODEC_BASE/$f"
    done
    for entry in "${CODEC_CASES[@]}"; do
        IFS='|' read -r name want_exit want_msg what <<<"$entry"
        dir="$ST_WORK/$name"
        build_case "$name" "$dir"
        gate="$ENGINE_PARITY"
        if [ -f "$dir/gate.sh" ]; then gate="$dir/gate.sh"; fi
        st_case "$name" "$want_exit" --what "$what" --has "$want_msg" -- gate_codec_disposition "$dir" "$gate"
    done
}

# ---- kv-boundary-default ---------------------------------------------------------------------

# bd_root <name>: a scan root holding the files the boundary gate reads, in agreement at 2,8.
bd_root() {
    local dir="$ST_WORK/bd/$1"
    mkdir -p "$dir/crates/rmlx-core/src" "$dir/crates/rmlx-cli/src" "$dir/docs"
    cat >"$dir/crates/rmlx-core/src/kv_boundary.rs" <<'EOF'
pub const DEFAULT_BOUNDARY_HEAD_N: usize = 2;
pub const DEFAULT_BOUNDARY_TAIL_N: usize = 8;
EOF
    cat >"$dir/crates/rmlx-cli/src/main.rs" <<'EOF'
        /// Layers held at the boundary floor, written
        /// as `<head>,<tail>`. Default `2,8`.
EOF
    cat >"$dir/docs/CLI.md" <<'EOF'
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | Leading and trailing layers held at the boundary floor. |
| `--kv-boundary-layers` | `HEAD,TAIL` | `2,8` | See "KV codec flags". |
EOF
    printf '%s' "$dir"
}

# bd_case <name> <want-exit> <reason> <file> <old> <new>: plant one edit in the file, run the gate.
bd_case() {
    local name="$1" want="$2" reason="$3" dir
    dir="$(bd_root "$name")"
    replace_once "$dir/$4" "$5" "$6"
    st_case "$name" "$want" --has "$reason" -- gate_boundary_default "$dir"
}

boundary_default_cases() {
    local dir

    dir="$(bd_root clean)"
    st_case "clean" 0 --what "the engine, the CLI help, docs/CLI.md and the resolver agree" \
        --has "OK: the default KV boundary '2,8' is named identically" \
        -- gate_boundary_default "$dir"

    bd_case "cli_help_disagrees" 1 "crates/rmlx-cli/src/main.rs says the default is '3,8'" \
        crates/rmlx-cli/src/main.rs 'Default `2,8`' 'Default `3,8`'
    bd_case "cli_help_names_no_pair" 1 "names no \`Default \`H,T\`\` pair" \
        crates/rmlx-cli/src/main.rs 'Default `2,8`' 'The default is two and eight'
    bd_case "cli_help_second_pair_disagrees" 1 "says the default is '2,9'" \
        crates/rmlx-cli/src/main.rs 'Default `2,8`.' $'Default `2,8`.\n        /// Default `2,9`.'
    bd_case "doc_row_disagrees" 1 "docs/CLI.md says the default is '2,9'" \
        docs/CLI.md '| `2,8` | Leading' '| `2,9` | Leading'
    bd_case "doc_second_row_disagrees" 1 "docs/CLI.md says the default is '4,8'" \
        docs/CLI.md '| `2,8` | See' '| `4,8` | See'

    dir="$(bd_root doc_names_no_flag)"
    printf 'No flag is documented here.\n' >"$dir/docs/CLI.md"
    st_case "doc_names_no_flag" 1 --has "documents no --kv-boundary-layers flag" -- gate_boundary_default "$dir"

    dir="$(bd_root doc_rows_state_no_default)"
    printf '%s\n' '`--kv-boundary-layers` is on serve only.' >"$dir/docs/CLI.md"
    st_case "doc_rows_state_no_default" 1 --has "rows that state no default" -- gate_boundary_default "$dir"

    bd_case "oracle_constant_renamed" 2 "cannot read DEFAULT_BOUNDARY_{HEAD,TAIL}_N" \
        crates/rmlx-core/src/kv_boundary.rs 'DEFAULT_BOUNDARY_TAIL_N' 'DEFAULT_TAIL_N'

    dir="$(bd_root oracle_file_missing)"
    rm -f "$dir/crates/rmlx-core/src/kv_boundary.rs"
    st_case "oracle_file_missing" 2 --has "missing crates/rmlx-core/src/kv_boundary.rs" \
        -- gate_boundary_default "$dir"

    dir="$(bd_root cli_source_missing)"
    rm -f "$dir/crates/rmlx-cli/src/main.rs"
    st_case "cli_source_missing" 2 --has "missing crates/rmlx-cli/src/main.rs" -- gate_boundary_default "$dir"

    dir="$(bd_root doc_missing)"
    rm -f "$dir/docs/CLI.md"
    st_case "doc_missing" 2 --has "missing docs/CLI.md" -- gate_boundary_default "$dir"

    dir="$(bd_root resolver_disagrees)"
    mkdir -p "$dir/scripts/lib"
    cp -f "$REPO_ROOT/scripts/lib/kv_boundary_default.py" "$dir/scripts/lib/"
    replace_once "$dir/scripts/lib/kv_boundary_default.py" 'return int(match.group("value"))' \
        'return int(match.group("value")) + 1'
    st_case "resolver_disagrees" 1 --has "resolves '3,9', engine '2,8'" -- gate_boundary_default "$dir"

    dir="$(bd_root resolver_cannot_find_constant)"
    mkdir -p "$dir/scripts/lib"
    cp -f "$REPO_ROOT/scripts/lib/kv_boundary_default.py" "$dir/scripts/lib/"
    replace_once "$dir/scripts/lib/kv_boundary_default.py" 'pub const DEFAULT_BOUNDARY_{}_N' \
        'pub const DEFAULT_{}_N'
    st_case "resolver_cannot_find_constant" 1 --has "cannot resolve the default" -- gate_boundary_default "$dir"

    dir="$(bd_root resolver_unimportable)"
    mkdir -p "$dir/scripts/lib"
    printf 'raise ImportError("planted")\n' >"$dir/scripts/lib/kv_boundary_default.py"
    st_case "resolver_unimportable" 1 --has "cannot resolve the default" -- gate_boundary_default "$dir"
}

# ---- spec-metrics ----------------------------------------------------------------------------

# sm_root <name>: a scan root holding both files the spec-metric gate reads, unedited.
sm_root() {
    local dir="$ST_WORK/spec/$1"
    mkdir -p "$dir/scripts/bench" "$dir/crates/rmlx-metrics/src"
    cp -f "$REPO_ROOT/crates/rmlx-metrics/src/registry.rs" "$dir/crates/rmlx-metrics/src/"
    cp -f "$REPO_ROOT/scripts/bench/serve_bench.sh" "$dir/scripts/bench/"
    printf '%s' "$dir"
}

# sm_judge <name> <want-exit> <what-it-proves> <reason> [<second-reason>]
sm_judge() {
    local name="$1" want="$2" what="$3" reason="$4" extra="${5:-}" checks=()
    checks=(--what "$what" --has "$reason")
    if [ -n "$extra" ]; then checks+=(--has "$extra"); fi
    st_case "$name" "$want" "${checks[@]}" -- gate_spec_metrics "$ST_WORK/spec/$name"
}

spec_metrics_cases() {
    local d

    sm_root unedited >/dev/null
    sm_judge unedited 0 "the tree as it stands agrees" "metrics declared and recorded"

    d="$(sm_root declared_only)"
    replace_once "$d/crates/rmlx-metrics/src/registry.rs" \
        $'    ("loop_ms_per_round", SpecRole::Derived),\n];' \
        $'    ("loop_ms_per_round", SpecRole::Derived),\n    ("resteer_ms_per_round", SpecRole::Derived),\n];'
    sm_judge declared_only 1 "a metric declared and never recorded is named" "never recorded by serve_bench.sh"

    d="$(sm_root recorded_only)"
    replace_once "$d/scripts/bench/serve_bench.sh" \
        $'    ("loop_ms_per_round", "loop_ms_per_round", float),\n)' \
        $'    ("loop_ms_per_round", "loop_ms_per_round", float),\n    ("resteer_ms_per_round", "resteer_ms_per_round", float),\n)'
    sm_judge recorded_only 1 "a metric recorded and never declared is named" "not declared in registry::SPEC_METRICS"

    d="$(sm_root dropped_from_bench)"
    replace_once "$d/scripts/bench/serve_bench.sh" $'    ("tokens_per_round", "tokens_per_round", float),\n' ''
    sm_judge dropped_from_bench 1 "a name deleted from the bench is the parity branch, and in the right direction" "never recorded by serve_bench.sh" "tokens_per_round"

    d="$(sm_root indented_python)"
    replace_once "$d/scripts/bench/serve_bench.sh" 'SPEC_METRICS = (' '    SPEC_METRICS = ('
    sm_judge indented_python 2 "a reformat that moves the marker off column 0 fails closed, naming the bench" "no SPEC_METRICS entries found in scripts/bench/serve_bench.sh"

    d="$(sm_root renamed_registry_const)"
    replace_once "$d/crates/rmlx-metrics/src/registry.rs" \
        'pub const SPEC_METRICS: &[(&str, SpecRole)] = &[' \
        'pub const SPECULATIVE_METRICS: &[(&str, SpecRole)] = &['
    sm_judge renamed_registry_const 2 "a rename of the oracle fails closed, naming the registry" "no SPEC_METRICS entries found in crates/rmlx-metrics/src/registry.rs"
}

# ---- run -------------------------------------------------------------------------------------

case "$GROUP" in
kv-byte-model) byte_model_cases ;;
kv-codec-disposition) codec_disposition_cases ;;
kv-boundary-default) boundary_default_cases ;;
spec-metrics) spec_metrics_cases ;;
all)
    byte_model_cases
    codec_disposition_cases
    boundary_default_cases
    spec_metrics_cases
    ;;
esac

st_done
