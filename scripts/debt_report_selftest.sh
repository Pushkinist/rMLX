#!/usr/bin/env bash
# scripts/debt_report_selftest.sh — fixture test for scripts/lib/debt_report.py.
#
# WHY
#   The report always exits 0 (advisory) — see scripts/debt_report.sh — so its
#   detection power is not in an exit code. Each case below asserts that a
#   specific fact actually appears (or is absent) in the printed text, rather
#   than trusting that the tool ran without checking what it found.
#
# HOW
#   scripts/fixtures/debt_report/base/ is a synthetic scan root: a twin file
#   pair, a same-naming-shape pair with unrelated bodies, a same-naming-shape
#   pair just under the similarity threshold, and the six speculative
#   round-loop drivers. The two size-critical docs (over/under the 200 KB
#   threshold) are generated into a throwaway copy of the fixture at run time
#   rather than committed, so this test does not carry ~250 KB of filler into
#   the tree's own churn count. The churn section needs real git history,
#   which the base fixture does not have on its own (it is a subtree of this
#   repo's working copy) — it builds its own throwaway two-commit, one-tag
#   repo in a temp dir instead.
#
# Exit 0 = every case found what it was supposed to. Exit 1 = at least one did not.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$REPO_ROOT/scripts/lib/debt_report.py"
BASE="$REPO_ROOT/scripts/fixtures/debt_report/base"

for f in \
    Makefile \
    crates/rmlx-kv-quant/src/storage/codec_alpha3.rs \
    crates/rmlx-kv-quant/src/storage/codec_alpha4.rs \
    crates/rmlx-kv-quant/src/storage/codec_beta3.rs \
    crates/rmlx-kv-quant/src/storage/codec_beta4.rs \
    crates/rmlx-kv-quant/src/storage/codec_gamma3.rs \
    crates/rmlx-kv-quant/src/storage/codec_gamma4.rs \
    crates/rmlx-kv-quant/src/storage/counters_allow.rs \
    crates/rmlx-kv-quant/src/storage/counters_debt.rs \
    crates/rmlx-kv-quant/src/storage/counters_debt_tests.rs \
    crates/rmlx-kv-quant/src/storage/counters_oversized.rs \
    crates/rmlx-kv-quant/src/storage/counters_oversized_exempt.rs \
    crates/rmlx-models/src/speculative/cached.rs \
    crates/rmlx-models/src/speculative/mtp.rs \
    crates/rmlx-models/src/speculative/dflash.rs \
    crates/rmlx-models/src/speculative/dflash2.rs \
    crates/rmlx-models/src/speculative/eagle3.rs \
    crates/rmlx-models/src/speculative/gemma4_assistant.rs \
    crates/rmlx-models/src/speculative/mod.rs \
    docs/SMALL.md
do
    [ -f "$BASE/$f" ] || {
        echo "debt-report selftest: missing $BASE/$f" >&2
        exit 1
    }
done

FAILED=0
PASSED=0

# check <name> <what it proves> <contains|absent> <needle> <haystack-var-name>
check() {
    local name="$1" what="$2" mode="$3" needle="$4" hay="$5"
    local found=1
    printf '%s' "${!hay}" | grep -qF -- "$needle" && found=0
    if { [ "$mode" = "contains" ] && [ "$found" -eq 0 ]; } ||
        { [ "$mode" = "absent" ] && [ "$found" -ne 0 ]; }; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-32s — %s\n' "$name" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-32s (mode=%s needle=%q) — %s\n' "$name" "$mode" "$needle" "$what"
    fi
}

STATIC_WORK="$(mktemp -d)"
cp -R "$BASE" "$STATIC_WORK/base"

python3 - "$STATIC_WORK/base/docs" <<'EOF'
import sys
docs = sys.argv[1]
big = ("This is a filler line documenting a codec variant in detail.\n" * 4300)
open(f"{docs}/BIG.md", "w").write("# Big doc\n\n" + big)  # ~257 KB, over the 200 KB threshold
almost = ("This is a filler line documenting a codec variant in detail.\n" * 3200)
open(f"{docs}/ALMOST.md", "w").write("# Almost doc\n\n" + almost)  # ~191 KB, under the threshold
EOF

BASE_OUT=$(python3 "$TOOL" --root "$STATIC_WORK/base" --since HEAD)

check "twin_pair_reported" \
    "a planted twin pair (differ only by the trailing digit) is reported" \
    contains "codec_alpha3.rs <-> crates/rmlx-kv-quant/src/storage/codec_alpha4.rs: 100.0% shared" \
    BASE_OUT

check "non_twin_pair_absent" \
    "a pair with the same naming shape but unrelated bodies is not reported" \
    absent "codec_beta" \
    BASE_OUT

check "near_threshold_pair_absent" \
    "a pair measuring 57.1% shared (below the 60% line) is not reported" \
    absent "codec_gamma" \
    BASE_OUT

check "fn_level_append_shared" \
    "fn-level similarity is asserted, not just the file-level percentage — deleting the fn loop or extract_fns leaves this absent" \
    contains "fn append (crates/rmlx-kv-quant/src/storage/codec_alpha3.rs:6 <-> crates/rmlx-kv-quant/src/storage/codec_alpha4.rs:6): 100.0% shared" \
    BASE_OUT

check "fn_level_byte_size_shared" \
    "a second named fn-level line from the same twin pair is asserted" \
    contains "fn byte_size (crates/rmlx-kv-quant/src/storage/codec_alpha3.rs:12 <-> crates/rmlx-kv-quant/src/storage/codec_alpha4.rs:12): 100.0% shared" \
    BASE_OUT

check "round_loop_group_named" \
    "the round-loop drivers group is printed" \
    contains "--- round-loop drivers (crates/rmlx-models/src/speculative) ---" \
    BASE_OUT

check "round_loop_driver_count" \
    "all 8 signature-matching drivers are found — 6 pub loops plus the 2 private cached ones a name list misses" \
    contains "  8 driver(s) found" \
    BASE_OUT

for entry in \
    "mtp_generate:crates/rmlx-models/src/speculative/mtp.rs:5" \
    "dflash_generate:crates/rmlx-models/src/speculative/dflash.rs:5" \
    "dflash2_generate:crates/rmlx-models/src/speculative/dflash2.rs:5" \
    "eagle3_generate:crates/rmlx-models/src/speculative/eagle3.rs:5" \
    "mtp_assistant_generate:crates/rmlx-models/src/speculative/gemma4_assistant.rs:5" \
    "spec_generate_greedy:crates/rmlx-models/src/speculative/mod.rs:5" \
    "spec_generate_greedy_cached:crates/rmlx-models/src/speculative/cached.rs:5" \
    "spec_generate_stochastic_cached:crates/rmlx-models/src/speculative/cached.rs:21"
do
    driver="${entry%%:*}"
    where="${entry#*:}"
    check "round_loop_has_${driver}" \
        "the round-loop group resolves ${driver} to its actual file:line, not just its name" \
        contains "  ${driver}" \
        BASE_OUT
    check "round_loop_path_${driver}" \
        "${driver}'s resolved path is the one printed" \
        contains "${where}" \
        BASE_OUT
done

check "doc_over_threshold_listed" \
    "BIG.md, generated over the 200 KB threshold, is listed" \
    contains "docs/BIG.md" \
    BASE_OUT

check "doc_under_threshold_absent" \
    "SMALL.md, well under the threshold, is not listed" \
    absent "SMALL.md" \
    BASE_OUT

check "doc_just_under_threshold_absent" \
    "ALMOST.md, generated just under the threshold, is not listed" \
    absent "ALMOST.md" \
    BASE_OUT

check "dead_path_not_attempted" \
    "dead-path counting says plainly that it is not attempted" \
    contains "dead-path (zero non-test callers): not attempted" \
    BASE_OUT

check "counter_allow_sites" \
    "the one planted #[allow(...)] site is counted" \
    contains "#[allow(...)] sites (non-test source only): 1" \
    BASE_OUT

check "counter_debt_comments" \
    "both planted debt comments are counted — one in source, one in a _tests.rs file, proving the population is source + tests" \
    contains "debt-marker comments (inert|dormant|deferred|kept for|future-reference|no longer; source + tests): 2" \
    BASE_OUT

check "counter_check_targets" \
    "the fixture Makefile's two check-*: targets are counted, the non-check target is not" \
    contains "check-* Make targets: 2" \
    BASE_OUT

check "counter_oversized" \
    "the 1200-line file is counted; its LOC-exempt-marked twin is not" \
    contains "files >1000 LOC without a LOC-exempt marker (non-test source only): 1" \
    BASE_OUT

# ---- negative case: a renamed driver is discovered under its new name, not --
# ---- its old one, and the group's total count is unaffected by a rename ----

RENAME_WORK="$(mktemp -d)"
cp -R "$BASE" "$RENAME_WORK/base"
sed -i.bak 's/pub fn mtp_generate(/pub fn zzz_renamed_driver(/' \
    "$RENAME_WORK/base/crates/rmlx-models/src/speculative/mtp.rs"
rm -f "$RENAME_WORK/base/crates/rmlx-models/src/speculative/mtp.rs.bak"

RENAME_OUT=$(python3 "$TOOL" --root "$RENAME_WORK/base" --since HEAD)

check "renamed_driver_old_name_absent" \
    "renaming a driver's fn (not its signature) removes its old name from the group" \
    absent "mtp_generate" \
    RENAME_OUT

check "renamed_driver_new_name_present" \
    "discovery is signature-driven, not name-driven — the renamed fn is still found, under its new name" \
    contains "  zzz_renamed_driver" \
    RENAME_OUT

check "renamed_driver_new_path_present" \
    "the renamed fn's resolved path is still the one printed" \
    contains "crates/rmlx-models/src/speculative/mtp.rs:5" \
    RENAME_OUT

check "renamed_driver_count_unchanged" \
    "the total driver count is unaffected by a pure rename (still 8) — a naive name-keyed implementation would lose one" \
    contains "  8 driver(s) found" \
    RENAME_OUT

rm -rf "$STATIC_WORK" "$RENAME_WORK"

# ---- churn on a synthetic two-commit repo -----------------------------------

RATIO_WORK="$(mktemp -d)"
trap 'rm -rf "$RATIO_WORK"' EXIT

git_ratio() { git -c commit.gpgsign=false -C "$RATIO_WORK" "$@"; }

git_ratio init -q || {
    echo "debt-report selftest: git init failed for the ratio fixture" >&2
    exit 1
}
git_ratio config user.email "debt-report-selftest@example.invalid" || exit 1
git_ratio config user.name "debt-report-selftest" || exit 1
mkdir -p "$RATIO_WORK/crates/fixture-crate/src" "$RATIO_WORK/docs" || exit 1

printf 'line1\nline2\nline3\nline4\nline5\n' >"$RATIO_WORK/crates/fixture-crate/src/lib.rs"
printf 'docline1\ndocline2\ndocline3\ndocline4\n' >"$RATIO_WORK/docs/A.md"
# A top-level *.md file, outside docs/ — proves the churn section's combined
# pathspec still covers it; dropping the "*.md" half of that call would go
# unnoticed if only docs/A.md (already covered by "docs") ever changed.
printf 'r1\nr2\nr3\n' >"$RATIO_WORK/README.md"

git_ratio add -A || exit 1
git_ratio commit -q -m "initial" || {
    echo "debt-report selftest: initial commit failed for the ratio fixture" >&2
    exit 1
}
git_ratio tag v0.0.1 || exit 1

# +4/-2 in the source file, +3/-1 in docs/A.md, +2/-1 in README.md — unique
# lines on both sides of each edit so the diff is unambiguous, not just
# plausible.
printf 'line1\nline2\nline3\nline6\nline7\nline8\nline9\n' >"$RATIO_WORK/crates/fixture-crate/src/lib.rs"
printf 'docline1\ndocline2\ndocline3\ndocline5\ndocline6\ndocline7\n' >"$RATIO_WORK/docs/A.md"
printf 'r1\nr2\nr4\nr5\n' >"$RATIO_WORK/README.md"
git_ratio add -A || exit 1
git_ratio commit -q -m "update" || {
    echo "debt-report selftest: update commit failed for the ratio fixture" >&2
    exit 1
}

RATIO_OUT=$(python3 "$TOOL" --root "$RATIO_WORK")

check "ratio_tag_resolved" \
    "the default ref (no --since) resolves to the planted tag" \
    contains "=== churn (summed over commits) since v0.0.1 ===" \
    RATIO_OUT

check "ratio_source_computed" \
    "source insertions/deletions match the planted diff (+4/-2)" \
    contains "source (crates/): +4 / -2  (ratio 2.00x)" \
    RATIO_OUT

check "ratio_docs_computed" \
    "docs/A.md and the top-level README.md are both in the combined figure (+3/-1 and +2/-1), not double-counted" \
    contains "docs (docs/, *.md): +5 / -2  (ratio 2.50x)" \
    RATIO_OUT

# A ref that never existed: the section must say so, not print a silent 0.
UNAVAILABLE_OUT=$(python3 "$TOOL" --root "$RATIO_WORK" --since v9.9.9-does-not-exist)

check "ratio_unavailable_on_bad_ref" \
    "an unresolvable --since is reported as unavailable, not read as zero churn" \
    contains "unavailable (" \
    UNAVAILABLE_OUT

if [ "$FAILED" -ne 0 ]; then
    echo "debt-report selftest: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi
echo "debt-report selftest: ok ($PASSED cases)"
