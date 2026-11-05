#!/usr/bin/env bash
# CI gate: a restatement of an engine fact agrees with the engine, one mode per fact.
#
# Usage:  engine_parity.sh kv-byte-model [MANIFEST]
#         engine_parity.sh kv-codec-disposition [SCAN_ROOT]
#         engine_parity.sh kv-boundary-default
#         engine_parity.sh spec-metrics
# Exit:   0 the restatement agrees; 1 it disagrees; 2 usage error, or the gate could not run
#         (build failure, missing or unreadable file, no python3)
# Set:    -uo — a probe that finds nothing (grep, diff, the emitter test) is a verdict the script reads
# Not a mode of: check/tree_lines.sh — that one matches forbidden lines; these compare against an engine-derived oracle
#
# Each mode holds one restatement to the engine, which is the oracle: no mode checks the engine is
# right, only that there is effectively one copy of the fact.
#
# kv-byte-model [MANIFEST]
#   `scripts/perf_ceiling.py` prices a codec without building the workspace, so it carries a second
#   copy of `estimated_resident_bytes_per_layer`, `packed_side_bytes`, `feeds_bf16_{k,v}_at_decode`
#   and `boundary_floor`. That copy drifted twice (a 4-byte ring sideband after the store narrowed
#   to 2; two bf16 mirrors for the Mixed family on architectures that stopped keeping them), and
#   each drift changed which codec the figures recommend without failing anything.
#   Oracle: `cargo test -p rmlx-models --lib emit_kv_byte_model_manifest` prints one `KVBYTES` row
#   per (codec, topology, shape) and one `KVFLOOR` row per (codec, topology), swept over
#   `ALL_KV_QUANTS`. `perf_ceiling.py --byte-model` re-emits the rows from its own arithmetic, so
#   the sweep, both topologies, the shapes and the layer count come from the Rust side alone.
#   With MANIFEST it reads a pre-captured manifest instead of running the emitter.
#
# kv-codec-disposition [SCAN_ROOT]
#   Most `KvQuant` variants never run: their decode reads the bf16 mirror, so `exit_prefill` skips
#   the encode and the codec is byte-identical to `--kv-quant none` in resident bytes and tokens.
#   Nothing in a codec's name, its per-variant doc section or a `--help` line beside a live codec
#   says so. The disposition moves one codec at a time, so the surfaces are checked, not reviewed.
#   Oracle: `cargo test -p rmlx-kv-quant --lib emit_kv_codec_disposition_manifest` prints one line
#   per codec over `ALL_KV_QUANTS`, classified by `KvQuant::materialises_packed_store()`. The
#   list's completeness is pinned by `descriptor_has_one_arm_per_listed_codec`.
#   With SCAN_ROOT it reads `main.rs`, a pre-captured `manifest.raw` and the banner docs from that
#   directory, which stands in for `docs/`.
#   RULE 1  every inert codec named in the `--kv-quant`/`--kv-bits`/`--kv-preset` help is also named
#           inside an INERT block of that same constant.
#   RULE 2  nothing inside an INERT block is a codec that is not inert.
#   RULE 3  every inert codec is named in an INERT banner of one of BANNER_DOCS.
#   RULE 4  nothing named in a banner is a codec that is not inert.
#   RULE 5  a banner opens within 3 lines of a `### ` heading.
#   RULE 6  every `--kv-quant`, `--kv-bits` and `--kv-preset` argument takes its help from the
#           shared constants, checked per argument.
#   RULE 7  no resident-KV ratio (`0.44x`, `2x`, `1.4×`) is written into the help: that figure has
#           one producer, `rmlx info --list-cache-types`.
#   RULE 8  every identifier clap renders as `help =` / `long_help =` anywhere in the CLI crate is a
#           constant this gate can read. The scope is derived from the clap attributes, not listed.
#   RULE 9  while the help points at `--list-cache-types`, both listing functions have a live call.
#   RULE 10 the banners are read from BANNER_DOCS only; a banner in another doc fails, a missing doc
#           is exit 2, and an empty list or a doc listed twice is exit 2.
#   RULE 11 each inert codec is named in exactly one banner across BANNER_DOCS.
#
# kv-boundary-default
#   `rmlx_core::kv_boundary::DEFAULT_BOUNDARY_{HEAD,TAIL}_N` is the one definition. Two surfaces
#   restate the pair as text a compiler never reads: the `--kv-boundary-layers` long help and the
#   flag's rows in docs/CLI.md; `scripts/lib/kv_boundary_default.py` must resolve to it too.
#   `decode_config` NULL means "the engine at its defaults", so cell identity is downstream of it.
#
# spec-metrics
#   `rmlx_metrics::registry::SPEC_METRICS` declares each speculative metric; `bench/serve_bench.sh`
#   holds a verbatim second copy of the names. A name declared and never recorded renders `-` in
#   the markdown export on every row for ever; a name recorded and never declared has no unit,
#   direction or bounds.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The docs that carry the INERT banners, as paths under docs/ (RULE 10).
BANNER_DOCS=(KV_CODECS.md KV_ROTATION_CODECS.md)

WORK=""
trap '[ -z "${WORK}" ] || rm -rf "${WORK}"' EXIT

die_env() {
    echo "ERROR (gate could not run): $*" >&2
    exit 2
}
die_violation() {
    echo "ERROR: $*" >&2
    exit 1
}
new_work() {
    WORK="$(mktemp -d)"
}

# load_manifest <label> <from> <begin> <end> <row-regex> <unit> <rows-out> <src> <package> <test>
#
# Reads an emitter manifest into <rows-out> (the rows matching <row-regex>) and leaves their count
# in MANIFEST_ROWS. <src> is a pre-captured manifest file; empty runs the emitter test. Refuses a
# manifest that never began, an emitter that failed after beginning, one with no END line, a count
# that disagrees with the END line, and an empty one.
load_manifest() {
    local label="$1" from="$2" begin="$3" end="$4" row_regex="$5" unit="$6" rows_out="$7"
    local src="$8" package="$9" test_path="${10}" cargo_status declared

    if [ -n "${src}" ]; then
        cat "${src}" >"${WORK}/manifest.raw"
        : >"${WORK}/manifest.err"
        cargo_status=0
    else
        command -v cargo >/dev/null 2>&1 ||
            die_env "cargo not on PATH — ${from} comes from the crate"
        (
            cd "${REPO_ROOT}" || exit 1
            cargo test -q -p "${package}" --lib -- --exact "${test_path}" --nocapture
        ) >"${WORK}/manifest.raw" 2>"${WORK}/manifest.err"
        cargo_status=$?
    fi

    if ! grep -q "^${begin}\$" "${WORK}/manifest.raw"; then
        echo "--- manifest stdout ---" >&2
        tail -n 30 "${WORK}/manifest.raw" >&2
        echo "--- manifest stderr ---" >&2
        tail -n 30 "${WORK}/manifest.err" >&2
        die_env "the ${label} manifest did not run (emitter exit ${cargo_status})"
    fi

    if [ "${cargo_status}" -ne 0 ]; then
        tail -n 30 "${WORK}/manifest.raw" >&2
        die_violation "the ${label} manifest itself failed — see above"
    fi

    grep -E "${row_regex}" "${WORK}/manifest.raw" >"${rows_out}" || true
    declared=$(grep -m1 "^${end}"$'\t' "${WORK}/manifest.raw" | cut -f2)
    MANIFEST_ROWS=$(awk 'END { print NR }' "${rows_out}")

    [ -n "${declared}" ] || die_env "the manifest printed no END sentinel"
    [ "${declared}" = "${MANIFEST_ROWS}" ] ||
        die_env "manifest truncated: END says ${declared} ${unit}, read ${MANIFEST_ROWS}"
    [ "${MANIFEST_ROWS}" -gt 0 ] || die_env "the manifest is empty"
}

# ── kv-byte-model ────────────────────────────────────────────────────────────

mode_kv_byte_model() {
    local manifest_src="${1:-}" perf_ceiling="${REPO_ROOT}/scripts/perf_ceiling.py"
    local actual codecs topologies shapes py_rows

    [ -f "${perf_ceiling}" ] || die_env "missing scripts/perf_ceiling.py"
    command -v python3 >/dev/null 2>&1 ||
        die_env "python3 not on PATH — the second byte model is a python script"
    new_work
    if [ -n "${manifest_src}" ]; then
        [ -f "${manifest_src}" ] || die_env "missing manifest ${manifest_src}"
    fi

    load_manifest byte-model "the byte model" KVBYTES-BEGIN KVBYTES-END \
        '^KV(BYTES|FLOOR)'$'\t' rows "${WORK}/rust.tsv" "${manifest_src}" \
        rmlx-models kv_cache::tests::tests::emit_kv_byte_model_manifest
    actual="${MANIFEST_ROWS}"

    codecs=$(cut -f2 "${WORK}/rust.tsv" | sort -u | awk 'END { print NR }')
    topologies=$(cut -f3 "${WORK}/rust.tsv" | sort -u | awk 'END { print NR }')
    shapes=$(awk -F'\t' '$1 == "KVBYTES" { print $5 "x" $6 }' "${WORK}/rust.tsv" |
        sort -u | awk 'END { print NR }')
    [ "${topologies}" -eq 2 ] ||
        die_env "the manifest covers ${topologies} topologies, not both — \
shares_kv moves the Mixed family by two whole mirrors and a one-sided sweep \
would not see it"
    [ "${shapes}" -ge 2 ] ||
        die_env "the manifest covers ${shapes} head-dim shape(s) — the ring and \
rotor cadences carry a per-row term, and a single shape cannot tell a wrong \
sideband width from a wrong per-row term"

    if ! python3 "${perf_ceiling}" --byte-model \
        <"${WORK}/rust.tsv" >"${WORK}/py.tsv" 2>"${WORK}/py.err"; then
        tail -n 20 "${WORK}/py.err" >&2
        die_env "perf_ceiling.py --byte-model failed to run"
    fi

    py_rows=$(awk 'END { print NR }' "${WORK}/py.tsv")
    [ "${py_rows}" -eq "${actual}" ] ||
        die_env "perf_ceiling.py re-emitted ${py_rows} of ${actual} rows — it \
dropped rows rather than disagreeing with them, so a diff would under-report"

    if ! diff -u "${WORK}/rust.tsv" "${WORK}/py.tsv" >"${WORK}/parity.diff"; then
        echo "ERROR: scripts/perf_ceiling.py disagrees with the engine's byte model." >&2
        echo >&2
        echo "  '-' is the engine (crates/rmlx-kv-quant, crates/rmlx-models)." >&2
        echo "  '+' is scripts/perf_ceiling.py." >&2
        echo "  Columns: kind, codec, shares_kv, then the shape, then the value." >&2
        echo >&2
        grep -E '^[-+]KV' "${WORK}/parity.diff" >&2
        echo >&2
        echo "The engine is the oracle. Move the script, not the engine." >&2
        exit 1
    fi

    echo "OK: ${actual} byte-model rows (${codecs} KV codecs x ${topologies} \
topologies x ${shapes} shapes) agree between the engine and \
scripts/perf_ceiling.py."
}

# ── kv-codec-disposition ─────────────────────────────────────────────────────

BANNER_MARKER='[*][*]INERT on this build[*][*]'
HELP_INERT_MARKER='^[[:space:]]*INERT[[:space:]]*—'

# Match a stem on a word boundary. `iso3` must not match inside `iso3_sym`, and
# `k8vturbo2` must not match inside `k8vturbo2tcq`, so an EXACT stem is fenced
# on both sides and a PREFIX stem on the left only.
stem_pattern() {
    local stem="$1" mode="$2"
    if [ "${mode}" = "EXACT" ]; then
        printf '%s' "(^|[^A-Za-z0-9_])${stem}([^A-Za-z0-9_]|\$)"
    else
        printf '%s' "(^|[^A-Za-z0-9_])${stem}"
    fi
}
match_stem() {
    grep -qE -- "$(stem_pattern "$1" "$2")" "$3"
}
# The banners that name a stem, one `<doc>:<line>` per banner.
banners_naming() {
    cut -f2- "${WORK}/banners.txt" | grep -nE -- "$(stem_pattern "$1" "$2")" |
        cut -d: -f1 | while IFS= read -r n; do
            sed -n "${n}p" "${WORK}/banners.txt" | cut -f1
        done
}

# Extract one `const X: &str = "..."` body from wherever in the CLI crate it is
# defined. `pub` / `pub(crate)` and an indented definition are accepted.
extract_const() {
    # shellcheck disable=SC2046  # the source list is newline-free by construction
    awk -v name="$1" '
        $0 ~ ("^[[:space:]]*(pub(\\([a-z]+\\))?[[:space:]]+)?const " name ": &str = ") {
            inside = 1; next
        }
        inside { print }
        inside && /";$/ { exit }
    ' $(cat "${WORK}/cli_sources")
}

# The file that defines a help constant, for an error message that can be acted on.
const_home() {
    grep -lE "^[[:space:]]*(pub(\([a-z]+\))?[[:space:]]+)?const $1: &str = " \
        $(cat "${WORK}/cli_sources") 2>/dev/null | sed -n 1p
}

violations=0
report() {
    echo "  $*" >&2
    violations=$((violations + 1))
}

mode_kv_codec_disposition() {
    local scan_root="${1:-}" cli_main cli_src docs_dir docs_label manifest_src cli_label
    local f d duplicate doc_list_label inert_count actual c rel listed

    docs_label="docs"
    if [ -n "${scan_root}" ]; then
        cli_main="${scan_root}/main.rs"
        cli_src="${scan_root}"
        docs_dir="${scan_root}"
        manifest_src="${scan_root}/manifest.raw"
    else
        cli_main="${REPO_ROOT}/crates/rmlx-cli/src/main.rs"
        cli_src="${REPO_ROOT}/crates/rmlx-cli/src"
        docs_dir="${REPO_ROOT}/docs"
        manifest_src=""
    fi
    cli_label="${cli_main#"${REPO_ROOT}/"}"

    for f in "${cli_main}" ${manifest_src:+"${manifest_src}"}; do
        [ -f "$f" ] || die_env "missing ${f#"${REPO_ROOT}/"}"
    done
    # `${#BANNER_DOCS[@]}` is safe under `set -u` on an empty array in bash 3.2;
    # `"${BANNER_DOCS[@]}"` is not, so the count is read first and every expansion is guarded.
    [ "${#BANNER_DOCS[@]}" -gt 0 ] ||
        die_env "BANNER_DOCS is empty: the gate would read no INERT banner (RULE 10)"
    duplicate=$(printf '%s\n' ${BANNER_DOCS[@]+"${BANNER_DOCS[@]}"} | sort | uniq -d | sed -n 1p)
    [ -z "${duplicate}" ] ||
        die_env "BANNER_DOCS lists ${duplicate} twice (RULE 10)"
    doc_list_label=""
    for d in ${BANNER_DOCS[@]+"${BANNER_DOCS[@]}"}; do
        doc_list_label="${doc_list_label:+${doc_list_label}, }${docs_label}/${d}"
    done
    for d in ${BANNER_DOCS[@]+"${BANNER_DOCS[@]}"}; do
        [ -f "${docs_dir}/${d}" ] ||
            die_env "missing ${docs_label}/${d}, a doc BANNER_DOCS lists (RULE 10)"
    done

    new_work
    load_manifest disposition "the disposition manifest" KVQUANT-DISPOSITION-BEGIN \
        KVQUANT-DISPOSITION-END '^KVQUANT-DISPOSITION'$'\t' codecs "${WORK}/manifest" \
        "${manifest_src}" rmlx-kv-quant quant::quant_tests::emit_kv_codec_disposition_manifest
    actual="${MANIFEST_ROWS}"

    # Read off the manifest rather than typed here: a hand-written count goes stale exactly when
    # this gate is doing its job.
    inert_count=$(awk -F'\t' '$6 == "INERT" { n++ } END { print n + 0 }' "${WORK}/manifest")

    # ── Surface 1: the CLI help constants ────────────────────────────────────
    # Every `.rs` in the CLI crate: a help constant lives wherever its module does.
    find "${cli_src}" -name '*.rs' -type f | sort >"${WORK}/cli_sources"
    [ -s "${WORK}/cli_sources" ] ||
        die_env "no .rs sources under ${cli_src#"${REPO_ROOT}/"}"

    # RULE 8: the scope is derived. A reference may be a bare identifier or a path
    # (`startup::X`, `crate::a::X`); the last path segment is the constant's name. A
    # `help = "literal"` is not collected on purpose: rule 6 already refuses one.
    grep -rhoE '(long_)?help = (([A-Za-z_][A-Za-z0-9_]*)::)*[A-Z][A-Z0-9_]*' \
        $(cat "${WORK}/cli_sources") |
        awk '{ print $3 }' | sed 's/.*:://' | sort -u >"${WORK}/referenced"
    [ -s "${WORK}/referenced" ] ||
        die_env "found no \`help = IDENT\` clap attribute under \
${cli_src#"${REPO_ROOT}/"} — the attribute shape stopped matching, so this \
gate would be reading nothing"

    : >"${WORK}/unreadable"
    while IFS= read -r c; do
        [ -n "$(const_home "$c")" ] || printf '%s\n' "$c" >>"${WORK}/unreadable"
    done <"${WORK}/referenced"
    if [ -s "${WORK}/unreadable" ]; then
        echo "ERROR: clap renders help this gate cannot read:" >&2
        while IFS= read -r c; do
            echo "  RULE 8  ${c} is used as clap help but no \`const ${c}: &str\` \
was found in ${cli_src#"${REPO_ROOT}/"}" >&2
        done <"${WORK}/unreadable"
        echo >&2
        echo "Rules 1, 2 and 7 read the constants named here and nothing else, so" >&2
        echo "a ratio or a dead codec name in that text is invisible to all three." >&2
        echo "Give it the \`const NAME: &str = \"...\";\` shape the extractor reads," >&2
        echo "or the gate is scanning past a surface an operator is shown." >&2
        echo "A violation and not an environment error: the text arrives in the" >&2
        echo "change that added it, and its author is who can shape it." >&2
        exit 1
    fi

    HELP_CONSTS="$(tr '\n' ' ' <"${WORK}/referenced")"

    : >"${WORK}/help.txt"
    : >"${WORK}/help_inert.txt"
    for c in ${HELP_CONSTS}; do
        extract_const "$c" >"${WORK}/const.$c"
        [ -s "${WORK}/const.$c" ] ||
            die_env "const ${c}, defined in $(const_home "$c"), has a definition \
this gate can find but a body it cannot read — the opening and closing shape \
the extractor keys on stopped matching"
        cat "${WORK}/const.$c" >>"${WORK}/help.txt"
        # The INERT blocks inside this one constant: from a marker line to the next blank line.
        # Kept per constant, not pooled: a codec declared inert in the `--kv-preset` block must
        # not launder a `--kv-quant` block that lists it beside the live codecs.
        awk '
            /^[[:space:]]*INERT[[:space:]]*—/ { inside = 1 }
            inside && /^[[:space:]]*$/        { inside = 0 }
            inside                            { print }
        ' "${WORK}/const.$c" >"${WORK}/inert.$c"
        cat "${WORK}/inert.$c" >>"${WORK}/help_inert.txt"
    done

    if [ ! -s "${WORK}/help_inert.txt" ]; then
        die_violation "the --kv-quant/--kv-bits help declares no INERT block.
${inert_count} of the ${actual} codecs it can name do nothing at runtime. If
that stopped being true, this gate's manifest would say so — check its output
first. A block opens with a line matching: ${HELP_INERT_MARKER}"
    fi

    # RULE 6: every --kv-quant / --kv-bits / --kv-preset argument reaches the shared constants.
    # Per argument, not by comparing counts: N arguments and N `help =` attributes is also what
    # one argument carrying two and another carrying none looks like. The attribute block of an
    # argument opens at its `#[` and runs to the declaration.
    awk '
        /^[[:space:]]*#\[/ { block = "" }
        { block = block $0 "\n" }
        /^[[:space:]]+kv_quant: (String|Option<String>),$/ {
            quant_args++
            if (block !~ /[^_]help = KV_QUANT_HELP/) {
                printf "MISS\t%d\t--kv-quant\thelp = KV_QUANT_HELP\n", NR
            }
            if (block !~ /long_help = KV_QUANT_LONG_HELP/) {
                printf "MISS\t%d\t--kv-quant\tlong_help = KV_QUANT_LONG_HELP\n", NR
            }
        }
        /^[[:space:]]+kv_bits: Option<f32>,$/ {
            bits_args++
            if (block !~ /long_help = KV_BITS_LONG_HELP/) {
                printf "MISS\t%d\t--kv-bits\tlong_help = KV_BITS_LONG_HELP\n", NR
            }
        }
        /^[[:space:]]+kv_preset: Option<KvPresetArg>,$/ {
            preset_args++
            if (block !~ /long_help = KV_PRESET_LONG_HELP/) {
                printf "MISS\t%d\t--kv-preset\tlong_help = KV_PRESET_LONG_HELP\n", NR
            }
        }
        END { printf "COUNT\t%d\t%d\t%d\n", quant_args, bits_args, preset_args }
    ' "${cli_main}" >"${WORK}/argcheck"

    quant_args=$(awk -F'\t' '$1 == "COUNT" { print $2 }' "${WORK}/argcheck")
    bits_args=$(awk -F'\t' '$1 == "COUNT" { print $3 }' "${WORK}/argcheck")
    preset_args=$(awk -F'\t' '$1 == "COUNT" { print $4 }' "${WORK}/argcheck")
    if [ "${quant_args}" -eq 0 ] || [ "${bits_args}" -eq 0 ] || [ "${preset_args}" -eq 0 ]; then
        die_env "found ${quant_args} --kv-quant, ${bits_args} --kv-bits and \
${preset_args} --kv-preset argument declarations in ${cli_label} — one of the \
three shapes stopped matching, so the gate would be checking nothing"
    fi

    grep '^MISS	' "${WORK}/argcheck" >"${WORK}/argmiss" || true
    if [ -s "${WORK}/argmiss" ]; then
        echo "ERROR: an argument does not take its help from the shared constants:" >&2
        while IFS=$'\t' read -r _ line flag want; do
            echo "  RULE 6  ${cli_label}:${line}  ${flag} is missing ${want}" >&2
        done <"${WORK}/argmiss"
        echo >&2
        echo "An argument with its own help text is one this gate does not read." >&2
        exit 1
    fi

    # RULE 7: the pattern is the shape of a ratio in every spelling the tree uses: an integer or
    # decimal followed by `x`, `X` or the Unicode multiplication sign, on a digit boundary so a
    # version or a group size is not a ratio. One definition serves the rule and its message.
    RATIO_SHAPE='(^|[^0-9.])[0-9]+(\.[0-9]+)?(x|X|×)([^A-Za-z0-9_]|$)'
    grep -nE "${RATIO_SHAPE}" "${WORK}/help.txt" >"${WORK}/ratios" || true
    if [ -s "${WORK}/ratios" ]; then
        echo "ERROR: the --kv-quant/--kv-bits/--kv-preset help writes a resident-KV \
ratio:" >&2
        while IFS= read -r line; do
            echo "  RULE 7  ${line}" >&2
        done <"${WORK}/ratios"
        echo >&2
        echo "That figure has one producer. Quote none and point the operator at" >&2
        echo "\`rmlx info --list-cache-types\`, which computes it per codec and per" >&2
        echo "topology from the same byte model the engine allocates against." >&2
        echo "Shape matched: ${RATIO_SHAPE}" >&2
        exit 1
    fi

    # RULE 9: the pattern matches a call statement, not the `use` that imports the name and not
    # the `fn` that defines it, so removing the call is caught while the symbol is still there.
    if grep -q -- '--list-cache-types' "${WORK}/help.txt"; then
        missing_call=0
        for fn in print_cache_type_table print_kv_quant_residency_table; do
            # shellcheck disable=SC2046
            if ! grep -qE "^[[:space:]]*${fn}\(\);" $(cat "${WORK}/cli_sources"); then
                echo "  RULE 9  ${cli_src#"${REPO_ROOT}/"}  ${fn}() is never called" >&2
                missing_call=$((missing_call + 1))
            fi
        done
        if [ "${missing_call}" -gt 0 ]; then
            echo >&2
            echo "ERROR: the help sends the operator to \`rmlx info --list-cache-types\`" >&2
            echo "for a figure it deliberately does not quote, and ${missing_call} of the" >&2
            echo "listings that command prints has no call site. The pointer is the only" >&2
            echo "place those numbers are published; a dangling one is worse than the" >&2
            echo "stale literals it replaced." >&2
            exit 1
        fi
    fi

    # ── Surface 2: the docs banners ──────────────────────────────────────────
    # RULE 10: a banner outside BANNER_DOCS is a banner no rule below reads.
    find "${docs_dir}" -name '*.md' -type f | sort >"${WORK}/all_docs" ||
        die_env "cannot list the docs under ${docs_label}"
    [ -s "${WORK}/all_docs" ] || die_env "no .md docs under ${docs_label}"
    : >"${WORK}/unlisted"
    while IFS= read -r f; do
        rel="${f#"${docs_dir}/"}"
        listed=0
        for d in ${BANNER_DOCS[@]+"${BANNER_DOCS[@]}"}; do
            [ "${rel}" = "${d}" ] && listed=1
        done
        if [ "${listed}" -eq 0 ] && grep -qE "^> .*${BANNER_MARKER}" "$f"; then
            printf '%s\n' "${docs_label}/${rel}" >>"${WORK}/unlisted"
        fi
    done <"${WORK}/all_docs"
    if [ -s "${WORK}/unlisted" ]; then
        echo "ERROR: an INERT banner sits in a doc the gate does not list:" >&2
        while IFS= read -r f; do
            echo "  RULE 10  ${f}" >&2
        done <"${WORK}/unlisted"
        echo >&2
        echo "The banners are read from ${doc_list_label} and nowhere else." >&2
        echo "Move the banner and its section back, or add the doc to BANNER_DOCS." >&2
        exit 1
    fi

    # A banner is a maximal run of consecutive `> ` lines whose first line carries the marker.
    # RULE 5 (placement) is checked in the same pass. Each banner is written as one line,
    # `<doc>:<line>` then its text, so rule 11 counts banners rather than lines.
    : >"${WORK}/banners.raw"
    for d in ${BANNER_DOCS[@]+"${BANNER_DOCS[@]}"}; do
        awk -v marker="${BANNER_MARKER}" -v doc="${docs_label}/${d}" '
            function flush() {
                if (text != "") print "BANNER\t" start "\t" text
                text = ""
            }
            /^### / { last_heading = NR }
            /^> / {
                if (!inside) {
                    if ($0 ~ marker) {
                        inside = 1
                        start = doc ":" NR
                        if (last_heading == 0 || NR - last_heading > 3) {
                            printf "PLACEMENT\t%s:%d\n", doc, NR
                        }
                    } else {
                        next
                    }
                }
                text = text " " $0
                next
            }
            { inside = 0; flush() }
            END { flush() }
        ' "${docs_dir}/${d}" >>"${WORK}/banners.raw" ||
            die_env "the banner scan of ${docs_label}/${d} failed"
    done

    grep '^PLACEMENT	' "${WORK}/banners.raw" >"${WORK}/placement" || true
    grep '^BANNER	' "${WORK}/banners.raw" | cut -f2- >"${WORK}/banners.txt" || true

    if [ -s "${WORK}/placement" ]; then
        echo "ERROR: an INERT banner does not open within 3 lines of a '### ' heading:" >&2
        while IFS=$'\t' read -r _ where; do
            echo "  RULE 5  ${where}" >&2
        done <"${WORK}/placement"
        echo >&2
        echo "A banner qualifies the section it heads. Buried in the body it is a" >&2
        echo "remark; at the head it is the first thing the reader sees." >&2
        exit 1
    fi

    if [ ! -s "${WORK}/banners.txt" ]; then
        die_violation "${doc_list_label} carry no INERT banner at all — every \
per-variant section for an inert codec needs one (marker: **INERT on this build**)"
    fi

    # ── The content rules ────────────────────────────────────────────────────
    while IFS=$'\t' read -r _ _idx display stem mode class _rs _bk _bv; do
        case "${class}" in
            INERT)
                # RULE 1: named in a help constant at all → that same constant must carry an
                # INERT block naming it.
                for c in ${HELP_CONSTS}; do
                    if match_stem "${stem}" "${mode}" "${WORK}/const.$c" &&
                        ! match_stem "${stem}" "${mode}" "${WORK}/inert.$c"; then
                        report "RULE 1  '${display}' is inert but ${c} names it outside an INERT block"
                    fi
                done
                # RULE 3: carries a docs banner. RULE 11: exactly one.
                banners_naming "${stem}" "${mode}" >"${WORK}/naming"
                named=$(wc -l <"${WORK}/naming" | tr -d ' ')
                if [ "${named}" -eq 0 ]; then
                    report "RULE 3  '${display}' is inert but no INERT banner in ${doc_list_label} names it"
                elif [ "${named}" -gt 1 ]; then
                    report "RULE 11  '${display}' is named in ${named} INERT banners: $(tr '\n' ' ' <"${WORK}/naming")"
                fi
                ;;
            LIVE | BASELINE)
                # RULE 2 / RULE 4: the converse, on both surfaces.
                if match_stem "${stem}" "${mode}" "${WORK}/help_inert.txt"; then
                    report "RULE 2  '${display}' is ${class}, not inert, but the CLI help's INERT block names it"
                fi
                banners_naming "${stem}" "${mode}" >"${WORK}/naming"
                if [ -s "${WORK}/naming" ]; then
                    report "RULE 4  '${display}' is ${class}, not inert, but the INERT banner at $(tr '\n' ' ' <"${WORK}/naming")names it"
                fi
                ;;
            *)
                die_env "unknown disposition class '${class}' for '${display}'"
                ;;
        esac
    done <"${WORK}/manifest"

    if [ "${violations}" -gt 0 ]; then
        echo >&2
        echo "ERROR: ${violations} disposition mismatch(es) between the code and the" >&2
        echo "user-facing surfaces. The code is the oracle — either the codec's" >&2
        echo "classification moved (fix the help and the banner docs to match), or" >&2
        echo "a surface was written from the codec's name instead of its behaviour." >&2
        echo >&2
        echo "Manifest as derived (index, name, stem, match, class, reads_store, bf16_k, bf16_v):" >&2
        cat "${WORK}/manifest" >&2
        exit 1
    fi

    echo "OK: ${actual} KV codecs classified from the type (${inert_count} inert); \
CLI help and the INERT banners in ${doc_list_label} agree with all of them."
}

# ── kv-boundary-default ──────────────────────────────────────────────────────

mode_kv_boundary_default() {
    local source="${REPO_ROOT}/crates/rmlx-core/src/kv_boundary.rs"
    local cli_main="${REPO_ROOT}/crates/rmlx-cli/src/main.rs"
    local cli_doc="${REPO_ROOT}/docs/CLI.md"
    local f head_n tail_n want fail help_pairs doc_rows doc_pairs py_pair pair

    for f in "${source}" "${cli_main}" "${cli_doc}"; do
        [ -f "$f" ] || die_env "missing ${f#"${REPO_ROOT}/"}"
    done

    head_n="$(sed -n "s/^pub const DEFAULT_BOUNDARY_HEAD_N: usize = \([0-9]*\);.*/\1/p" "${source}" | sed -n 1p)"
    tail_n="$(sed -n "s/^pub const DEFAULT_BOUNDARY_TAIL_N: usize = \([0-9]*\);.*/\1/p" "${source}" | sed -n 1p)"
    if [[ -z "${head_n}" || -z "${tail_n}" ]]; then
        echo "ERROR: cannot read DEFAULT_BOUNDARY_{HEAD,TAIL}_N from ${source#"${REPO_ROOT}/"}." >&2
        echo "That file is the oracle for this gate; a rename has to update it here too." >&2
        exit 2
    fi
    want="${head_n},${tail_n}"

    fail=0

    # The CLI long help: one occurrence, spelled ``Default `H,T`.``
    help_pairs="$(sed -n 's/.*Default `\([0-9]*,[0-9]*\)`.*/\1/p' "${cli_main}" || true)"
    if [[ -z "${help_pairs}" ]]; then
        echo "ERROR: crates/rmlx-cli/src/main.rs names no \`Default \`H,T\`\` pair." >&2
        echo "The --kv-boundary-layers long help states the default; this gate reads it." >&2
        fail=1
    else
        while IFS= read -r pair; do
            [[ -z "${pair}" ]] && continue
            if [[ "${pair}" != "${want}" ]]; then
                echo "ERROR: crates/rmlx-cli/src/main.rs says the default is '${pair}'," >&2
                echo "       the engine applies '${want}'." >&2
                fail=1
            fi
        done <<<"${help_pairs}"
    fi

    # docs/CLI.md flag rows.
    doc_rows="$(grep -c -- '--kv-boundary-layers' "${cli_doc}" || true)"
    if [[ "${doc_rows}" -eq 0 ]]; then
        echo "ERROR: docs/CLI.md documents no --kv-boundary-layers flag." >&2
        fail=1
    fi
    doc_pairs="$(sed -n 's/.*`--kv-boundary-layers` | `HEAD,TAIL` | `\([0-9]*,[0-9]*\)`.*/\1/p' "${cli_doc}" || true)"
    if [[ -z "${doc_pairs}" ]]; then
        echo "ERROR: docs/CLI.md has --kv-boundary-layers rows that state no default." >&2
        fail=1
    else
        while IFS= read -r pair; do
            [[ -z "${pair}" ]] && continue
            if [[ "${pair}" != "${want}" ]]; then
                echo "ERROR: docs/CLI.md says the default is '${pair}', the engine applies '${want}'." >&2
                fail=1
            fi
        done <<<"${doc_pairs}"
    fi

    # The shared Python resolver actually resolves.
    py_pair="$(cd "${REPO_ROOT}" && python3 -c "
import sys
sys.path.insert(0, 'scripts/lib')
from kv_boundary_default import kv_boundary_default
print('%d,%d' % kv_boundary_default())
" 2>&1)" || {
        echo "ERROR: scripts/lib/kv_boundary_default.py cannot resolve the default:" >&2
        echo "${py_pair}" >&2
        exit 1
    }
    if [[ "${py_pair}" != "${want}" ]]; then
        echo "ERROR: scripts/lib/kv_boundary_default.py resolves '${py_pair}', engine '${want}'." >&2
        fail=1
    fi

    if [[ "${fail}" -ne 0 ]]; then
        echo >&2
        echo "The engine's constants are the oracle. Update the surface, not the constant," >&2
        echo "unless the default itself is what moved." >&2
        exit 1
    fi

    echo "OK: the default KV boundary '${want}' is named identically by the engine, the CLI help, docs/CLI.md and the ingest resolver ($((doc_rows)) doc mentions)."
}

# ── spec-metrics ─────────────────────────────────────────────────────────────

mode_spec_metrics() {
    local registry="${REPO_ROOT}/crates/rmlx-metrics/src/registry.rs"
    local bench="${REPO_ROOT}/scripts/bench/serve_bench.sh"
    local want got missing extra status

    # The `SPEC_METRICS` array literal in the registry: every `("name", SpecRole::_)`.
    want="$(awk '
        /^pub const SPEC_METRICS: &\[\(&str, SpecRole\)\] = &\[/ { inside = 1; next }
        inside && /^\];/ { exit }
        inside && match($0, /"[a-z0-9_]+"/) {
            print substr($0, RSTART + 1, RLENGTH - 2)
        }
    ' "${registry}" | sort -u)"
    # The `SPEC_METRICS` tuple in the bench script: the first element of each row.
    got="$(awk '
        /^SPEC_METRICS = \(/ { inside = 1; next }
        inside && /^\)/ { exit }
        inside && match($0, /\("[a-z0-9_]+"/) {
            print substr($0, RSTART + 2, RLENGTH - 3)
        }
    ' "${bench}" | sort -u)"

    if [[ -z "${want}" ]]; then
        echo "ERROR: no SPEC_METRICS entries found in ${registry#"${REPO_ROOT}/"}." >&2
        echo "That list is this gate's oracle; a rename has to update this script too." >&2
        exit 2
    fi
    if [[ -z "${got}" ]]; then
        echo "ERROR: no SPEC_METRICS entries found in ${bench#"${REPO_ROOT}/"}." >&2
        exit 2
    fi

    missing="$(comm -23 <(echo "${want}") <(echo "${got}") || true)"
    extra="$(comm -13 <(echo "${want}") <(echo "${got}") || true)"

    status=0
    if [[ -n "${missing}" ]]; then
        echo "ERROR: declared in registry::SPEC_METRICS and never recorded by serve_bench.sh:" >&2
        echo "${missing}" | sed 's/^/  /' >&2
        echo "  Its column in the markdown export renders '-' on every row." >&2
        status=1
    fi
    if [[ -n "${extra}" ]]; then
        echo "ERROR: recorded by serve_bench.sh and not declared in registry::SPEC_METRICS:" >&2
        echo "${extra}" | sed 's/^/  /' >&2
        echo "  Nothing gives it a unit, a direction or plausible bounds." >&2
        status=1
    fi
    if [[ "${status}" -eq 0 ]]; then
        echo "check-spec-metric-parity: ok ($(echo "${want}" | wc -l | tr -d ' ') metrics declared and recorded)"
    fi
    exit "${status}"
}

usage() {
    echo "usage: engine_parity.sh {kv-byte-model [MANIFEST]|kv-codec-disposition [SCAN_ROOT]|kv-boundary-default|spec-metrics}" >&2
    exit 2
}

[ "$#" -gt 0 ] || usage
mode="$1"
shift
case "${mode}" in
kv-byte-model) mode_kv_byte_model "$@" ;;
kv-codec-disposition) mode_kv_codec_disposition "$@" ;;
kv-boundary-default) mode_kv_boundary_default ;;
spec-metrics) mode_spec_metrics ;;
*) usage ;;
esac
