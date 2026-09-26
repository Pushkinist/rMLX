#!/usr/bin/env bash
# scripts/check_kv_codec_disposition.sh — CI gate: what the user-facing surfaces
# say a KV codec does, and what the runtime classifiers say it does, agree.
#
# USAGE
#   check_kv_codec_disposition.sh [SCAN_ROOT]
#
#   With no argument it reads the real surfaces and derives the manifest by
#   running the emitter test. With a SCAN_ROOT it reads `main.rs`, a
#   pre-captured `manifest.raw` and the banner docs from that directory instead,
#   which stands in for `docs/` — how `check_kv_codec_disposition_fixtures.sh`
#   drives it, one mutation at a time.
#
# WHY
#   Most of the `KvQuant` variants never run — the gate prints how many on every
#   pass, so the number lives in one place and cannot be read here after it
#   moved. Their decode
#   reads the bf16 mirror on both axes, so `exit_prefill` skips the encode and
#   calls `storage.clear_payload()`: at runtime they are byte-identical to
#   `--kv-quant none` in both resident bytes and generated tokens. That is not
#   visible from the codec's name, its per-variant doc section (which
#   describes a packed store nothing builds), or a `--help` line that lists it
#   beside a live codec. An operator who reads the help, picks a name and
#   reads back the resolved-codec log line has no way to learn it did nothing.
#
#   The disposition is not a property anyone can keep in their head: it moves
#   the day a codec grows a decode kernel over its own store, and it moves one
#   codec at a time. So the surfaces are checked against the code, not reviewed.
#
# ORACLE
#   `cargo test -p rmlx-kv-quant --lib emit_kv_codec_disposition_manifest`
#   prints one line per codec, swept over `ALL_KV_QUANTS` and classified by
#   `KvQuant::materialises_packed_store()` — the disjunction of the three
#   predicates `exit_prefill` gates the encode on:
#
#       decode_reads_packed_store()  feeds_bf16_k_at_decode()  feeds_bf16_v_at_decode()
#
#   `ALL_KV_QUANTS`'s completeness is pinned by
#   `descriptor_has_one_arm_per_listed_codec`, which counts the arms of the
#   compiler-checked codec descriptor match out of the source and compares them
#   to the list's length — a variant absent from the list can be constructed
#   nowhere in the crate, so nothing that sweeps the list could see it. A new
#   enum variant therefore cannot slip past this gate by being absent from a
#   list. Nothing here is hand-written per codec.
#
# RULE 1 (CLI help, coverage)
#   Every inert codec named anywhere in the `--kv-quant` / `--kv-bits` /
#   `--kv-preset` help text must also appear inside an INERT block in that text.
#   A name that is only listed beside the live codecs is a name the help says
#   works. `--kv-preset` is in scope because a preset is a codec under another
#   name: five of its seven targets are inert, and while the block sat outside
#   this list it said so nowhere.
#
# RULE 2 (CLI help, converse)
#   Nothing inside an INERT block may be a codec that is NOT inert. A block
#   that over-claims retires a working codec in the reader's head, and it is
#   how the block goes stale when a codec is wired up.
#
# RULE 3 (docs, coverage)
#   Every inert codec must be named in an INERT banner in one of the
#   BANNER_DOCS. The per-variant sections describe pack formats and bit
#   rates in the present tense; without the banner they describe a store the
#   codec does not build.
#
# RULE 4 (docs, converse)
#   Nothing named in a banner may be a codec that is NOT inert.
#
# RULE 5 (docs, placement)
#   A banner must open within 3 lines of a `### ` heading — it belongs at the
#   head of the section it qualifies, not buried in one.
#
# RULE 10 (docs, where the banners are)
#   The banners are read from BANNER_DOCS, a fixed list of docs, not a glob. A
#   banner in any other doc under docs/ fails: it is a banner no rule reads. A
#   listed doc that is missing is exit 2: the list names a surface that is gone.
#   An empty list, or a doc listed twice, is exit 2 too: the list is config.
#
# RULE 11 (docs, one banner per codec)
#   Each inert codec is named in exactly one banner across BANNER_DOCS. A
#   second banner is a copy that the next edit to one of them leaves stale.
#
# RULE 6 (the help is actually reached)
#   Every `--kv-quant`, `--kv-bits` and `--kv-preset` argument in the CLI must
#   take its help from the shared constants. Copies drift; one that is checked
#   does not. Checked per argument, not by comparing two counts: "five arguments
#   and five `help =` attributes" is also what one argument carrying two of them
#   and another carrying none looks like.
#
# RULE 9 (the listing the help points at is still printed)
#   The help carries no ratio; it tells the operator to run
#   `rmlx info --list-cache-types`. That makes the listing's call site part of
#   the help, and a call site is deletable without breaking a build or a test
#   that only checks the rendering. So whenever a scoped help constant names
#   `--list-cache-types`, both listing functions must have a live call.
#
# RULE 8 (every text clap shows an operator is a text this gate reads)
#   Rules 1, 2 and 7 read a set of help constants, and the set used to be typed
#   into this script. A hand-written scope list is the same disease those rules
#   exist to cure: it was missing `--kv-preset` for as long as nobody remembered
#   the flag existed, then `CACHE_TYPE_K_LONG_HELP` and `CACHE_TYPE_V_LONG_HELP`
#   after that, and a constant defined outside `main.rs` would have been next.
#
#   So there is no list. The set is DERIVED from the clap attributes across the
#   whole CLI crate: every identifier used as `help = X` or `long_help = X` is a
#   string an operator is shown, and that is the definition of in scope. Scoped
#   by shape, so it needs no exclusions either — a constant clap never renders
#   is not help, and a help constant in a new module is picked up by being
#   referenced, whichever file it lives in.
#
#   The rule itself is then the one thing that derivation can still get wrong:
#   an identifier clap renders whose definition this gate cannot extract. That
#   is a text shown to an operator and read by nothing, so it fails.
#
# RULE 7 (no ratio is written into the help)
#   No resident-KV ratio may appear in any of these constants, in any spelling
#   this tree uses: `0.44x`, `1.406x`, `2x`, and the same three with the Unicode
#   multiplication sign `×`, which is what the ratio rows of
#   docs/KV_ROTATION_CODECS.md are written with. Keyed on the SHAPE of such a figure, not on a list
#   of the ones that were there -- a corrected number is the same defect as a
#   stale one, and a pattern that matches one spelling is a gate that a reviewer
#   can walk past by typing the other.
#   Such a figure has one producer — `KvQuant::estimated_resident_bytes_per_layer`,
#   which `rmlx info --list-cache-types` prints — and a second copy of it in a
#   string literal is correct only by hand. The four that used to sit here were
#   carried unchanged through the two commits that moved the stores underneath
#   them, and ended up filing the four codecs that compress most under a heading
#   that said they were larger than bf16. Rules 1-5 could not see it: they are
#   all set-membership rules on codec names and none of them reads a number.
#
# SCOPE
#   Rules 3-5 check that an inert codec carries a banner in one of the
#   BANNER_DOCS under a section heading, not that it is the heading of its
#   own section — headings in that file are prose and not machine-addressable.
#
# EXIT CODES
#   0  gate ran and passed
#   1  gate ran and found a violation
#   2  gate could not run (build failure, missing file, unparseable surface)
#      — never conflated with 1, because "no violations found" and "nothing was
#      looked at" are the same exit code in a naive script and the second one
#      passes silently forever.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_ROOT="${1:-}"

# The docs that carry the INERT banners, as paths under docs/ (RULE 10).
BANNER_DOCS=(KV_CODECS.md KV_ROTATION_CODECS.md)

if [ -n "${SCAN_ROOT}" ]; then
    CLI_MAIN="${SCAN_ROOT}/main.rs"
    CLI_SRC="${SCAN_ROOT}"
    DOCS_DIR="${SCAN_ROOT}"
    MANIFEST_SRC="${SCAN_ROOT}/manifest.raw"
    # The scan root stands in for docs/, so a message names the doc the same
    # way in both modes.
    DOCS_LABEL="docs"
else
    CLI_MAIN="${REPO_ROOT}/crates/rmlx-cli/src/main.rs"
    # The whole crate, not one file: a help constant is wherever its module is.
    CLI_SRC="${REPO_ROOT}/crates/rmlx-cli/src"
    DOCS_DIR="${REPO_ROOT}/docs"
    DOCS_LABEL="docs"
    MANIFEST_SRC=""
fi
CLI_LABEL="${CLI_MAIN#"${REPO_ROOT}/"}"

BANNER_MARKER='[*][*]INERT on this build[*][*]'
HELP_INERT_MARKER='^[[:space:]]*INERT[[:space:]]*—'

die_env() {
    echo "ERROR (gate could not run): $*" >&2
    exit 2
}
die_violation() {
    echo "ERROR: $*" >&2
    exit 1
}

for f in "${CLI_MAIN}" ${MANIFEST_SRC:+"${MANIFEST_SRC}"}; do
    [ -f "$f" ] || die_env "missing ${f#"${REPO_ROOT}/"}"
done
# `${#BANNER_DOCS[@]}` is safe under `set -u` on an empty array in bash 3.2;
# `"${BANNER_DOCS[@]}"` is not, so the count is read first.
[ "${#BANNER_DOCS[@]}" -gt 0 ] ||
    die_env "BANNER_DOCS is empty: the gate would read no INERT banner (RULE 10)"
duplicate=$(printf '%s\n' "${BANNER_DOCS[@]}" | sort | uniq -d | head -1)
[ -z "${duplicate}" ] ||
    die_env "BANNER_DOCS lists ${duplicate} twice (RULE 10)"
DOC_LIST_LABEL=""
for d in "${BANNER_DOCS[@]}"; do
    DOC_LIST_LABEL="${DOC_LIST_LABEL:+${DOC_LIST_LABEL}, }${DOCS_LABEL}/${d}"
done
for d in "${BANNER_DOCS[@]}"; do
    [ -f "${DOCS_DIR}/${d}" ] ||
        die_env "missing ${DOCS_LABEL}/${d}, a doc BANNER_DOCS lists (RULE 10)"
done

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

# ── Oracle: the disposition manifest, derived from the type ──────────────────
if [ -n "${MANIFEST_SRC}" ]; then
    # Fixture mode: the manifest was captured ahead of time. Everything below
    # this point is the production path unchanged, including the two sentinel
    # checks — that is what makes the fixtures a recall test of the real gate.
    cat "${MANIFEST_SRC}" >"${WORK}/manifest.raw"
    : >"${WORK}/manifest.err"
    cargo_status=0
else
    if ! command -v cargo >/dev/null 2>&1; then
        die_env "cargo not on PATH — the disposition manifest comes from the crate"
    fi

    (
        cd "${REPO_ROOT}" || exit 1
        cargo test -q -p rmlx-kv-quant --lib -- \
            --exact quant::quant_tests::emit_kv_codec_disposition_manifest --nocapture
    ) >"${WORK}/manifest.raw" 2>"${WORK}/manifest.err"
    cargo_status=$?
fi

if ! grep -q '^KVQUANT-DISPOSITION-BEGIN$' "${WORK}/manifest.raw"; then
    # Never reached the emitter: build failure, filtered-away test, wrong path.
    echo "--- manifest stdout ---" >&2
    tail -n 30 "${WORK}/manifest.raw" >&2
    echo "--- manifest stderr ---" >&2
    tail -n 30 "${WORK}/manifest.err" >&2
    die_env "the disposition manifest did not run (emitter exit ${cargo_status})"
fi

if [ "${cargo_status}" -ne 0 ]; then
    # Reached the emitter and then failed: the derivation itself is broken
    # (duplicate surface stem, empty stem). That is a violation, not an
    # environment problem, and the two must not share an exit code.
    tail -n 30 "${WORK}/manifest.raw" >&2
    die_violation "the disposition manifest itself failed — see above"
fi

grep '^KVQUANT-DISPOSITION	' "${WORK}/manifest.raw" >"${WORK}/manifest" || true
declared=$(grep '^KVQUANT-DISPOSITION-END	' "${WORK}/manifest.raw" | head -1 | cut -f2)
actual=$(grep -c '^KVQUANT-DISPOSITION	' "${WORK}/manifest.raw")

[ -n "${declared}" ] || die_env "the manifest printed no END sentinel"
[ "${declared}" = "${actual}" ] ||
    die_env "manifest truncated: END says ${declared} codecs, read ${actual}"
[ "${actual}" -gt 0 ] || die_env "the manifest is empty"

# How many codecs are inert, read off the manifest rather than typed here — a
# hand-written count goes stale exactly when this gate is doing its job.
inert_count=$(awk -F'\t' '$6 == "INERT" { n++ } END { print n + 0 }' "${WORK}/manifest")

# ── Surface 1: the CLI help constants ────────────────────────────────────────
# Every `.rs` in the CLI crate. A help constant lives wherever its module does,
# and a gate that reads one file is a gate with a move away from a blind spot.
find "${CLI_SRC}" -name '*.rs' -type f | sort >"${WORK}/cli_sources"
[ -s "${WORK}/cli_sources" ] ||
    die_env "no .rs sources under ${CLI_SRC#"${REPO_ROOT}/"}"

# Extract one `const X: &str = "..."` body from wherever in the crate it is
# defined. `pub` / `pub(crate)` and an indented definition are accepted: the
# gate must not care which module a constant sits in.
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

# Which file defines it, for an error message that can be acted on.
const_home() {
    grep -lE "^[[:space:]]*(pub(\([a-z]+\))?[[:space:]]+)?const $1: &str = " \
        $(cat "${WORK}/cli_sources") 2>/dev/null | head -1
}

# RULE 8: the scope is derived, not declared.
#
# In scope = every identifier clap renders as help. That is what "operator
# facing" means; it needs no exclusion list, because a constant clap never
# renders is not help. A reference may be a bare identifier or a path
# (`startup::X`, `crate::a::X`) — a constant outside the argument's own module
# is referenced by path, and both forms put the same text in front of an
# operator — so the last path segment is taken as the constant's name. A
# `help = "literal"` is not collected on purpose: rule 6 already refuses one on
# the arguments this gate is about.
grep -rhoE '(long_)?help = (([A-Za-z_][A-Za-z0-9_]*)::)*[A-Z][A-Z0-9_]*' \
    $(cat "${WORK}/cli_sources") |
    awk '{ print $3 }' | sed 's/.*:://' | sort -u >"${WORK}/referenced"
[ -s "${WORK}/referenced" ] ||
    die_env "found no \`help = IDENT\` clap attribute under \
${CLI_SRC#"${REPO_ROOT}/"} — the attribute shape stopped matching, so this \
gate would be reading nothing"

: >"${WORK}/unreadable"
while IFS= read -r c; do
    [ -n "$(const_home "$c")" ] || printf '%s\n' "$c" >>"${WORK}/unreadable"
done <"${WORK}/referenced"
if [ -s "${WORK}/unreadable" ]; then
    echo "ERROR: clap renders help this gate cannot read:" >&2
    while IFS= read -r c; do
        echo "  RULE 8  ${c} is used as clap help but no \`const ${c}: &str\` \
was found in ${CLI_SRC#"${REPO_ROOT}/"}" >&2
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
    # The INERT blocks inside this one constant: from a marker line to the next
    # blank line. Kept per constant, not pooled: a codec declared inert in the
    # `--kv-preset` block must not launder a `--kv-quant` block that lists it
    # beside the live codecs. Those are two surfaces and an operator reads one.
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

# Rule 6: every --kv-quant / --kv-bits / --kv-preset argument reaches the shared
# constants. Read from main.rs, where the arguments are declared; if they move,
# the COUNT check below finds zero of a shape and stops the gate rather than
# passing over an empty scan.
# Per argument, not by comparing counts: N arguments and N `help =` attributes
# is also what one argument carrying two and another carrying none looks like.
# The attribute block that belongs to an argument opens at its `#[` and runs to
# the declaration; a `help =` on some other argument cannot launder this one.
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
' "${CLI_MAIN}" >"${WORK}/argcheck"

quant_args=$(awk -F'\t' '$1 == "COUNT" { print $2 }' "${WORK}/argcheck")
bits_args=$(awk -F'\t' '$1 == "COUNT" { print $3 }' "${WORK}/argcheck")
preset_args=$(awk -F'\t' '$1 == "COUNT" { print $4 }' "${WORK}/argcheck")
if [ "${quant_args}" -eq 0 ] || [ "${bits_args}" -eq 0 ] || [ "${preset_args}" -eq 0 ]; then
    die_env "found ${quant_args} --kv-quant, ${bits_args} --kv-bits and \
${preset_args} --kv-preset argument declarations in ${CLI_LABEL} — one of the \
three shapes stopped matching, so the gate would be checking nothing"
fi

grep '^MISS	' "${WORK}/argcheck" >"${WORK}/argmiss" || true
if [ -s "${WORK}/argmiss" ]; then
    echo "ERROR: an argument does not take its help from the shared constants:" >&2
    while IFS=$'\t' read -r _ line flag want; do
        echo "  RULE 6  ${CLI_LABEL}:${line}  ${flag} is missing ${want}" >&2
    done <"${WORK}/argmiss"
    echo >&2
    echo "An argument with its own help text is one this gate does not read." >&2
    exit 1
fi

# ── Rule 7: no resident-KV ratio is written into the help ───────────────────
# A ratio in these constants is a hand-maintained copy of
# `KvQuant::estimated_resident_bytes_per_layer`, and the copy is what goes
# stale. `rmlx info --list-cache-types` prints the computed figure; the help
# points at it.
#
# The pattern is the *shape* of such a figure, across every spelling in the
# tree: an integer or decimal followed by `x`, `X` or the Unicode multiplication
# sign, on a digit boundary so a version or a group size is not a ratio.
# `RATIO_SHAPE` is one definition shared with the reason string, so the rule and
# the message it prints cannot describe different patterns.
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

# ── Rule 9: the pointer the help hands the operator still resolves ──────────
# `grep -qE '^[[:space:]]*fn\(\);'` matches a statement, not the `use` that
# imports the name and not the `fn` that defines it — so removing the call is
# caught even though the symbol is still in the file.
if grep -q -- '--list-cache-types' "${WORK}/help.txt"; then
    missing_call=0
    for fn in print_cache_type_table print_kv_quant_residency_table; do
        # shellcheck disable=SC2046
        if ! grep -qE "^[[:space:]]*${fn}\(\);" $(cat "${WORK}/cli_sources"); then
            echo "  RULE 9  ${CLI_SRC#"${REPO_ROOT}/"}  ${fn}() is never called" >&2
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

# ── Surface 2: the docs banners ──────────────────────────────────────────────
# RULE 10: a banner outside BANNER_DOCS is a banner no rule below reads.
find "${DOCS_DIR}" -name '*.md' -type f | sort >"${WORK}/all_docs" ||
    die_env "cannot list the docs under ${DOCS_LABEL}"
[ -s "${WORK}/all_docs" ] || die_env "no .md docs under ${DOCS_LABEL}"
: >"${WORK}/unlisted"
while IFS= read -r f; do
    rel="${f#"${DOCS_DIR}/"}"
    listed=0
    for d in "${BANNER_DOCS[@]}"; do
        [ "${rel}" = "${d}" ] && listed=1
    done
    if [ "${listed}" -eq 0 ] && grep -qE "^> .*${BANNER_MARKER}" "$f"; then
        printf '%s\n' "${DOCS_LABEL}/${rel}" >>"${WORK}/unlisted"
    fi
done <"${WORK}/all_docs"
if [ -s "${WORK}/unlisted" ]; then
    echo "ERROR: an INERT banner sits in a doc the gate does not list:" >&2
    while IFS= read -r f; do
        echo "  RULE 10  ${f}" >&2
    done <"${WORK}/unlisted"
    echo >&2
    echo "The banners are read from ${DOC_LIST_LABEL} and nowhere else." >&2
    echo "Move the banner and its section back, or add the doc to BANNER_DOCS." >&2
    exit 1
fi

# A banner is a maximal run of consecutive `> ` lines whose first line carries
# the marker. Rule 5 (placement) is checked in the same pass. Each banner is
# written as one line, `<doc>:<line>` then its text, so rule 11 can count
# banners rather than lines.
: >"${WORK}/banners.raw"
for d in "${BANNER_DOCS[@]}"; do
    awk -v marker="${BANNER_MARKER}" -v doc="${DOCS_LABEL}/${d}" '
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
    ' "${DOCS_DIR}/${d}" >>"${WORK}/banners.raw" ||
        die_env "the banner scan of ${DOCS_LABEL}/${d} failed"
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
    die_violation "${DOC_LIST_LABEL} carry no INERT banner at all — every \
per-variant section for an inert codec needs one (marker: **INERT on this build**)"
fi

# ── The four content rules ───────────────────────────────────────────────────
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

violations=0
report() {
    echo "  $*" >&2
    violations=$((violations + 1))
}

while IFS=$'\t' read -r _ _idx display stem mode class _rs _bk _bv; do
    case "${class}" in
        INERT)
            # Rule 1: named in a help constant at all → that same constant must
            # carry an INERT block naming it.
            for c in ${HELP_CONSTS}; do
                if match_stem "${stem}" "${mode}" "${WORK}/const.$c" &&
                    ! match_stem "${stem}" "${mode}" "${WORK}/inert.$c"; then
                    report "RULE 1  '${display}' is inert but ${c} names it outside an INERT block"
                fi
            done
            # Rule 3: must carry a docs banner. Rule 11: exactly one.
            banners_naming "${stem}" "${mode}" >"${WORK}/naming"
            named=$(wc -l <"${WORK}/naming" | tr -d ' ')
            if [ "${named}" -eq 0 ]; then
                report "RULE 3  '${display}' is inert but no INERT banner in ${DOC_LIST_LABEL} names it"
            elif [ "${named}" -gt 1 ]; then
                report "RULE 11  '${display}' is named in ${named} INERT banners: $(tr '\n' ' ' <"${WORK}/naming")"
            fi
            ;;
        LIVE | BASELINE)
            # Rule 2 / Rule 4: the converse, on both surfaces.
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
CLI help and the INERT banners in ${DOC_LIST_LABEL} agree with all of them."
