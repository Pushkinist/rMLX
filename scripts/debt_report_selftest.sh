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
#   pair just under the similarity threshold, one speculative fn carrying both
#   the driver signature and a constructed RoundTotals (the sole round-loop
#   driver) beside seven that carry the signature alone, and two planted impl
#   RoundDrafter bodies for the matched-lines figure, two same-axis rotor
#   storage files (a three-member V group, a two-member K group, plus a test
#   file matching the same glob) and five update_rotor* fns beside one
#   update_affine, for the two rotor populations, the same shape at two
#   members per group in quant_iso_*.rs and update_iso* — plus an
#   iso_v_update / iso_sym_update pair that only an unanchored name pattern
#   reaches — for the two iso populations, and two per-arch hydrate
#   bodies beside a hydrate_from_ssd and a test-path copy, for the
#   ssd-hydrate population. A group of three is what
#   separates every-pair-in-a-group from consecutive-only pairing; a width
#   spelled as its own segment (update_rotor_5_sym) is what separates a
#   separator-collapsing key from one that leaves a doubled separator behind.
#   The two size-critical
#   docs (over/under the 200 KB threshold) are generated into a throwaway copy
#   of the fixture at run time rather than committed, so this test does not
#   carry ~250 KB of filler into the tree's own churn count. The churn section
#   needs real git history, which the base fixture does not have on its own
#   (it is a subtree of this repo's working copy) — it builds its own
#   throwaway two-commit, one-tag repo in a temp dir instead.
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
    crates/rmlx-kv-quant/src/storage/quant_rotor_v2.rs \
    crates/rmlx-kv-quant/src/storage/quant_rotor_v3.rs \
    crates/rmlx-kv-quant/src/storage/quant_rotor_v4.rs \
    crates/rmlx-kv-quant/src/storage/quant_rotor_v3_tests.rs \
    crates/rmlx-kv-quant/src/storage/quant_rotor_k3.rs \
    crates/rmlx-kv-quant/src/storage/quant_rotor_k4.rs \
    crates/rmlx-kv-quant/src/storage/quant_iso_v3.rs \
    crates/rmlx-kv-quant/src/storage/quant_iso_v4.rs \
    crates/rmlx-kv-quant/src/storage/quant_iso_k3.rs \
    crates/rmlx-kv-quant/src/storage/quant_iso_k4.rs \
    crates/rmlx-kv-quant/src/storage/quant_k_turbo3.rs \
    crates/rmlx-kv-quant/src/storage/quant_k_turbo4.rs \
    crates/rmlx-kv-ssd/src/block_io.rs \
    crates/rmlx-kv-quant/src/kvcache/update.rs \
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
    crates/rmlx-models/src/speculative/round_impl_alpha.rs \
    crates/rmlx-models/src/speculative/round_impl_beta.rs \
    crates/rmlx-models/src/hydrate_alpha/prompt_cache.rs \
    crates/rmlx-models/src/hydrate_alpha/prompt_cache_tests.rs \
    crates/rmlx-models/src/hydrate_beta/prompt_cache.rs \
    crates/rmlx-models/src/hydrate_gamma/prompt_cache.rs \
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

# check_exit <name> <what it proves> <expected> <actual>
check_exit() {
    local name="$1" what="$2" want="$3" got="$4"
    if [ "$want" = "$got" ]; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-32s — %s\n' "$name" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-32s (want exit %s, got %s) — %s\n' "$name" "$want" "$got" "$what"
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
    "population (a) — the driver signature AND a constructed RoundTotals, check_spec_charge.sh's own rule — finds exactly the one fn that carries both, not every signature match" \
    contains "  1 driver(s) found" \
    BASE_OUT

check "round_loop_has_mtp_generate" \
    "the sole (signature + RoundTotals) fn is resolved to its actual file:line" \
    contains "  mtp_generate                     crates/rmlx-models/src/speculative/mtp.rs:5" \
    BASE_OUT

check "round_loop_no_pairwise" \
    "one driver has no pair to compare against" \
    contains "  no pairwise similarity (fewer than two drivers)" \
    BASE_OUT

# A dead producer (check_spec_charge.sh missing/broken) also reports 0
# drivers, which would make every absent-check below pass for the wrong
# reason. This pins that BASE_OUT is a genuinely healthy scan, not that.
check "base_out_not_unavailable" \
    "the healthy scan the exclude-checks below read from is not itself a swallowed error" \
    absent "  unavailable (" \
    BASE_OUT

# Each of these carries the driver signature but never constructs a
# RoundTotals — under the old signature-only rule every one of them was
# reported as a driver; under population (a) none is.
for driver in \
    dflash_generate \
    dflash2_generate \
    eagle3_generate \
    mtp_assistant_generate \
    spec_generate_greedy \
    spec_generate_greedy_cached \
    spec_generate_stochastic_cached
do
    check "round_loop_excludes_${driver}" \
        "${driver} has the signature and no RoundTotals — population (a) excludes it" \
        absent "  ${driver}" \
        BASE_OUT
done

# ---- negative case: renaming the driver AND dropping its RoundTotals in ----
# ---- the same fixture copy is a refusal, never a silent 0 -----------------

TOTALS_WORK="$(mktemp -d)"
cp -R "$BASE" "$TOTALS_WORK/base"
sed -i.bak \
    -e '/let _totals = RoundTotals { total: out.len() as u32 };/d' \
    -e 's/pub fn mtp_generate(/pub fn zzz_renamed_and_detotaled(/' \
    "$TOTALS_WORK/base/crates/rmlx-models/src/speculative/mtp.rs"
rm -f "$TOTALS_WORK/base/crates/rmlx-models/src/speculative/mtp.rs.bak"

TOTALS_OUT=$(python3 "$TOOL" --root "$TOTALS_WORK/base" --since HEAD)

check "totals_removed_reports_unavailable" \
    "renaming the sole driver and removing its RoundTotals leaves check_spec_charge.sh's own population empty, which is a refusal (exit 2, 'found no round loop'), not a silent 0" \
    contains "  unavailable (check_spec_charge.sh --list-drivers failed (exit 2): check-spec-charge: found no round loop" \
    TOTALS_OUT

check "totals_removed_no_driver_count_line" \
    "once the section is unavailable, no stale 'driver(s) found' line follows it" \
    absent "driver(s) found" \
    TOTALS_OUT

check "totals_removed_old_name_absent" \
    "the renamed, detotaled fn's old name is gone from the section too" \
    absent "mtp_generate" \
    TOTALS_OUT

rm -rf "$TOTALS_WORK"

# ---- a dead producer (check_spec_charge.sh unreachable) is unavailable, ---
# ---- never a silent 0 — the concrete failure the guard above rules out ----

DEAD_OUT=$(python3 - "$REPO_ROOT/scripts/lib" "$STATIC_WORK/base" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import debt_report as dr
from pathlib import Path
dr.CHECK_SPEC_CHARGE_SCRIPT = Path("/nonexistent/check_spec_charge.sh")
lines = []
dr.report_sibling_similarity(Path(sys.argv[2]), lines)
print("\n".join(lines))
EOF
)

check "dead_producer_reports_unavailable" \
    "pointing the gate script at a path that does not exist surfaces as unavailable" \
    contains "  unavailable (check_spec_charge.sh not found at /nonexistent/check_spec_charge.sh)" \
    DEAD_OUT

check "dead_producer_no_driver_count_line" \
    "the stale 'driver(s) found' line is not printed once the section is unavailable — this is the shape the guard above rules out for the healthy runs" \
    absent "driver(s) found" \
    DEAD_OUT

# ---- two same-named, same-file drivers join 1:1 by (file, fn, line) — 2 --
# ---- drivers, not the 4 a name-only join would produce --------------------

DUP_WORK="$(mktemp -d)"
cp -R "$BASE" "$DUP_WORK/base"
cat >>"$DUP_WORK/base/crates/rmlx-models/src/speculative/mtp.rs" <<'EOF'

mod inner {
    use super::ProbeStep;

    pub fn mtp_generate(
        prompt_ids: &[u32],
        n_tokens: usize,
        step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        for _ in 0..n_tokens {
            let next = prompt_ids.last().copied().unwrap_or(0) + 1;
            if step_fn(&ProbeStep { token: next }).is_none() {
                break;
            }
            out.push(next);
        }
        let _totals = RoundTotals { total: out.len() as u32 };
        out
    }
}
EOF

DUP_OUT=$(python3 "$TOOL" --root "$DUP_WORK/base" --since HEAD)

check "duplicate_named_driver_count_is_two_not_four" \
    "two same-named, same-file drivers at different lines join 1:1 by (file, fn, line) — a name-only join would extend by both Python matches for each of the 2 gate-listed lines and read 4" \
    contains "  2 driver(s) found" \
    DUP_OUT

rm -rf "$DUP_WORK"

# ---- a gate-listed driver Python's own scan excludes (moved under a -------
# ---- speculative/tests/ subdirectory) is reported unresolved, not dropped -

MOVED_WORK="$(mktemp -d)"
cp -R "$BASE" "$MOVED_WORK/base"
sed -i.bak '/let _totals = RoundTotals { total: out.len() as u32 };/d' \
    "$MOVED_WORK/base/crates/rmlx-models/src/speculative/mtp.rs"
rm -f "$MOVED_WORK/base/crates/rmlx-models/src/speculative/mtp.rs.bak"

mkdir -p "$MOVED_WORK/base/crates/rmlx-models/src/speculative/tests"
cat >"$MOVED_WORK/base/crates/rmlx-models/src/speculative/tests/moved_driver.rs" <<'EOF'
pub struct ProbeStep {
    pub token: u32,
}

pub fn moved_generate(
    prompt_ids: &[u32],
    n_tokens: usize,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
) -> Vec<u32> {
    let mut out = Vec::new();
    for _ in 0..n_tokens {
        let next = prompt_ids.last().copied().unwrap_or(0) + 1;
        if step_fn(&ProbeStep { token: next }).is_none() {
            break;
        }
        out.push(next);
    }
    let _totals = RoundTotals { total: out.len() as u32 };
    out
}
EOF

MOVED_OUT=$(python3 "$TOOL" --root "$MOVED_WORK/base" --since HEAD)

check "unresolved_driver_reported_not_zero" \
    "check_spec_charge.sh excludes only by filename (*_tests.rs / tests.rs) and still lists a driver under a tests/ subdirectory; rust_files() excludes the whole subdirectory by path component and cannot resolve it — this reads as 1 listed, 0 resolved, not a silent 0 driver(s) found" \
    contains "  1 listed, 0 resolved" \
    MOVED_OUT

check "unresolved_driver_named" \
    "the unresolved entry names the fn and where the gate found it, so nothing is silently dropped" \
    contains "    unresolved: moved_generate crates/rmlx-models/src/speculative/tests/moved_driver.rs:5" \
    MOVED_OUT

rm -rf "$MOVED_WORK"

# ---- --matched-lines: one producer for the campaign's duplication figure --

DRIVERS_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines drivers)

check "matched_lines_drivers_single_item" \
    "one driver has no pair, so the summed matched-line count over the driver population is 0" \
    contains "round-loop drivers (crates/rmlx-models/src/speculative): 0 matched lines over 12 body lines (1 item(s), 0 pair(s))" \
    DRIVERS_ML

IMPLS_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines impls)

check "matched_lines_impls_planted_pair" \
    "the two planted impl RoundDrafter bodies are identical but for their type name, which sits outside the extracted body — the matched-line count is the full 7-line body, known in advance" \
    contains "impl RoundDrafter bodies (crates/rmlx-models/src/speculative): 7 matched lines over 14 body lines (2 item(s), 1 pair(s))" \
    IMPLS_ML

# A body edit lowers the matched-line count below the planted 7 — the figure
# reads the actual bodies each time, not a cached constant.
DIVERGED_WORK="$(mktemp -d)"
cp -R "$BASE" "$DIVERGED_WORK/base"
sed -i.bak 's/base \* 2/base * 3/' \
    "$DIVERGED_WORK/base/crates/rmlx-models/src/speculative/round_impl_beta.rs"
rm -f "$DIVERGED_WORK/base/crates/rmlx-models/src/speculative/round_impl_beta.rs.bak"

DIVERGED_ML=$(python3 "$TOOL" --root "$DIVERGED_WORK/base" --matched-lines impls)

check "matched_lines_impls_diverged_pair_drops" \
    "editing one planted body's last statement drops the matched count from the planted 7 to 6" \
    contains "impl RoundDrafter bodies (crates/rmlx-models/src/speculative): 6 matched lines over 14 body lines (2 item(s), 1 pair(s))" \
    DIVERGED_ML

rm -rf "$DIVERGED_WORK"

# ---- the empty-population rule is not rotor-only: it governs every ---------
# ---- population, and it changed what an impls scan with no bodies prints ---

EMPTY_IMPLS_WORK="$(mktemp -d)"
cp -R "$BASE" "$EMPTY_IMPLS_WORK/base"
rm -f "$EMPTY_IMPLS_WORK/base/crates/rmlx-models/src/speculative"/round_impl_*.rs

EMPTY_IMPLS_ML=$(python3 "$TOOL" --root "$EMPTY_IMPLS_WORK/base" --matched-lines impls 2>&1)
EMPTY_IMPLS_STATUS=$?

check "matched_lines_impls_empty_unavailable" \
    "the speculative directory is there and holds no impl RoundDrafter body: unavailable, where this used to print 0 matched lines over 0 body lines (0 item(s), 0 pair(s)) and exit 0 — the rule is the population's, not the rotor populations'" \
    contains "debt-report --matched-lines impls: unavailable (crates/rmlx-models/src/speculative: population is empty)" \
    EMPTY_IMPLS_ML

check_exit "matched_lines_impls_empty_exit" \
    "an empty impls population exits 1, like every other unavailable one" \
    1 "$EMPTY_IMPLS_STATUS"

rm -rf "$EMPTY_IMPLS_WORK"

# ---- --matched-lines: the two rotor populations ---------------------------
#
# Each is derived from the fixture by a glob plus a name rule, never a file or
# fn list: the non-test quant_rotor_*.rs files under the storage directory,
# and the update_rotor* fns of the update file, both paired inside a group
# sharing the name with every digit run removed. The fixture plants two
# same-axis pairs per population, so an implementation that paired every item
# with every other would report 6 (storage) or 6 (updates) pairs, not 2.

ROTOR_STORAGE_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines rotor-storage 2>&1)
ROTOR_STORAGE_STATUS=$?

check "matched_lines_rotor_storage_pairs" \
    "the five planted storage files are a three-member V group and a two-member K group: every pair inside a group, so 3 + 1 = 4 pairs (14 folded lines of 15 per V pair, all 9 for K), not the ten an all-pairs population compares and not the 3 pairs / 37 lines consecutive-only pairing would leave; quant_rotor_v3_tests.rs matches the glob and is excluded as a test path" \
    contains "rotor storage twins (crates/rmlx-kv-quant/src/storage): 51 matched lines over 63 body lines (5 item(s), 4 pair(s))" \
    ROTOR_STORAGE_ML

check_exit "matched_lines_rotor_storage_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$ROTOR_STORAGE_STATUS"

check "matched_lines_rotor_storage_label_own_root" \
    "the label names the population's own root, taken from the population entry" \
    contains "rotor storage twins (crates/rmlx-kv-quant/src/storage):" \
    ROTOR_STORAGE_ML

check "matched_lines_rotor_storage_label_not_spec_dir" \
    "the label does not name SPEC_DIR — the module constant the two speculative populations use, which a shared render would print beside every figure" \
    absent "speculative" \
    ROTOR_STORAGE_ML

ROTOR_UPDATES_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines rotor-updates 2>&1)
ROTOR_UPDATES_STATUS=$?

check "matched_lines_rotor_updates_pairs" \
    "the five planted update_rotor* fns are a two-member plain family and a three-member sym family, 1 + 3 = 4 pairs; update_rotor_5_sym spells its width as its own segment and still joins the sym group, which a key that left the doubled separator behind would split off (2 pairs, 7 matched); update_affine in the same impl is outside the prefix — widening the prefix to update_ would read 6 item(s)" \
    contains "rotor update twins (crates/rmlx-kv-quant/src/kvcache/update*.rs): 15 matched lines over 23 body lines (5 item(s), 4 pair(s))" \
    ROTOR_UPDATES_ML

check_exit "matched_lines_rotor_updates_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$ROTOR_UPDATES_STATUS"

check "matched_lines_rotor_updates_label_own_root" \
    "a population root can be a glob over several files, and it is the one printed" \
    contains "rotor update twins (crates/rmlx-kv-quant/src/kvcache/update*.rs):" \
    ROTOR_UPDATES_ML

# ---- split: a family's bodies move to their own file, the figure holds -----
#
# The population is a glob over the update files, so a body that changes file
# without changing a line must leave every figure where it was. A file list, or
# a pairing order that followed the file layout, would move one.

SPLIT_WORK="$(mktemp -d)"
cp -R "$BASE" "$SPLIT_WORK/base"
python3 - "$SPLIT_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" \
    "$SPLIT_WORK/base/crates/rmlx-kv-quant/src/kvcache/update_rotor.rs" <<'EOF'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
moved = []
for name in ("update_rotor3_sym", "update_rotor4_sym", "update_rotor_5_sym"):
    match = re.search(r"\n    fn " + name + r"\(.*?\n    \}\n", text, flags=re.S)
    assert match, name
    moved.append(match.group(0))
    text = text[: match.start()] + "\n" + text[match.end() :]
open(src, "w").write(text)
open(dst, "w").write("impl KvCache {" + "".join(moved) + "}\n")
EOF

SPLIT_UPDATES_ML=$(python3 "$TOOL" --root "$SPLIT_WORK/base" --matched-lines rotor-updates 2>&1)
SPLIT_UPDATES_STATUS=$?

check "matched_lines_rotor_updates_split_unchanged" \
    "moving the three sym bodies into a family file of their own leaves the figure, the item count and the pair count exactly where they were — the glob finds them, and the name-sorted pairing does not care which file holds a body" \
    contains "rotor update twins (crates/rmlx-kv-quant/src/kvcache/update*.rs): 15 matched lines over 23 body lines (5 item(s), 4 pair(s))" \
    SPLIT_UPDATES_ML

check_exit "matched_lines_rotor_updates_split_exit" \
    "a split population is still a measurement, exit 0" \
    0 "$SPLIT_UPDATES_STATUS"

SPLIT_BODIES_ML=$(python3 "$TOOL" --root "$SPLIT_WORK/base" --matched-lines update-bodies 2>&1)

check "matched_lines_update_bodies_split_unchanged" \
    "the whole-prefix population holds too: the same 14 bodies over two files read the same figure they read over one" \
    contains "update_-prefixed fns of the update files (crates/rmlx-kv-quant/src/kvcache/update*.rs): 248 matched lines over 63 body lines (14 item(s), 91 pair(s))" \
    SPLIT_BODIES_ML

rm -rf "$SPLIT_WORK"

# ---- orientation: an asymmetric pair, measured the same whichever way -----
#
# `difflib.SequenceMatcher` is not symmetric: it anchors on the longest match
# it finds in its first argument. The two bodies below share four lines read
# one way and three the other, and they are the only pair in this file that
# does — every other planted body is too short or too uniform for the
# asymmetry to show, which is why a producer that measured one way round and
# ordered its pairs by the file layout passed all of them.
#
# Two arrangements, because one alone cannot hold both rules. The bodies are
# the same; which name carries which decides whether the name order reaches
# the larger reading or the smaller one.

orient_tree() { # orient_tree ROOT FIRST_BODY SECOND_BODY
    local root="$1" first="$2" second="$3"
    rm -rf "$root"
    cp -R "$BASE" "$root"
    local dir="$root/crates/rmlx-kv-quant/src/kvcache"
    printf 'impl KvCache {\n    fn update_zz(&mut self) {\n%b    }\n}\n' "$first" >"$dir/update.rs"
    printf 'impl KvCache {\n    fn update_aa(&mut self) {\n%b    }\n}\n' "$second" >"$dir/update_rotor.rs"
}

# `b a c a` read against `a a a b` matches three lines; the other way round,
# four.
ORIENT_BAC="        b();\n        a();\n        c();\n        a();\n"
ORIENT_AAB="        a();\n        a();\n        a();\n        b();\n"
ORIENT_EXPECTED="update_-prefixed fns of the update files (crates/rmlx-kv-quant/src/kvcache/update*.rs): 4 matched lines over 12 body lines (2 item(s), 1 pair(s))"

ORIENT_WORK="$(mktemp -d)"

# The first name carries the body that reads the smaller count when it is
# measured first. Only measuring both ways round reaches 4 here.
orient_tree "$ORIENT_WORK/base" "$ORIENT_AAB" "$ORIENT_BAC"
ORIENT_ML=$(python3 "$TOOL" --root "$ORIENT_WORK/base" --matched-lines update-bodies 2>&1)
ORIENT_STATUS=$?

check "matched_lines_orientation_both_ways" \
    "the pair shares four lines one way and three the other, and the name order reaches the three: a one-directional measure reads 3 here, so this case is what holds the measure to the larger of the two" \
    contains "$ORIENT_EXPECTED" \
    ORIENT_ML

check_exit "matched_lines_orientation_both_ways_exit" \
    "a measured population exits 0" \
    0 "$ORIENT_STATUS"

# The same two bodies with the names swapped. Here the name order reaches the
# larger count on its own, so this case is the one that fails if the pair list
# is left in collector order while the measure is one-directional.
orient_tree "$ORIENT_WORK/base" "$ORIENT_BAC" "$ORIENT_AAB"
ORIENT_NAMED_ML=$(python3 "$TOOL" --root "$ORIENT_WORK/base" --matched-lines update-bodies 2>&1)

check "matched_lines_orientation_name_order" \
    "swapping which name carries which body leaves the figure where it was — with the pair list in collector order and a one-directional measure this reads 3" \
    contains "$ORIENT_EXPECTED" \
    ORIENT_NAMED_ML

rm -rf "$ORIENT_WORK"

# ---- collapsed: the width twin is gone, the population is still found -----

COLLAPSED_WORK="$(mktemp -d)"
cp -R "$BASE" "$COLLAPSED_WORK/base"
rm -f "$COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/storage/quant_rotor_v2.rs" \
    "$COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/storage/quant_rotor_v4.rs" \
    "$COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/storage/quant_rotor_k4.rs"
python3 - "$COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" <<'EOF'
import re
import sys

path = sys.argv[1]
text = open(path).read()
# Drop the two 4-bit entries the way the collapse did — the 3-bit ones stay.
for name in ("update_rotor4", "update_rotor4_sym", "update_rotor_5_sym"):
    text = re.sub(r"\n    fn " + name + r"\(.*?\n    \}\n", "\n", text, flags=re.S)
open(path, "w").write(text)
EOF

COLLAPSED_STORAGE_ML=$(python3 "$TOOL" --root "$COLLAPSED_WORK/base" --matched-lines rotor-storage 2>&1)
COLLAPSED_STORAGE_STATUS=$?

check "matched_lines_rotor_storage_collapsed_zero" \
    "deleting the 4-bit files leaves one item per axis and so no pair: 0 matched lines with the population still found, which is a different answer from an empty population" \
    contains "rotor storage twins (crates/rmlx-kv-quant/src/storage): 0 matched lines over 24 body lines (2 item(s), 0 pair(s))" \
    COLLAPSED_STORAGE_ML

check_exit "matched_lines_rotor_storage_collapsed_exit" \
    "a collapsed population is a measured 0, exit 0" \
    0 "$COLLAPSED_STORAGE_STATUS"

COLLAPSED_UPDATES_ML=$(python3 "$TOOL" --root "$COLLAPSED_WORK/base" --matched-lines rotor-updates 2>&1)
COLLAPSED_UPDATES_STATUS=$?

check "matched_lines_rotor_updates_collapsed_zero" \
    "deleting the two 4-bit entries leaves one body per family: 0 matched lines over the 9 body lines that remain, population still found" \
    contains "rotor update twins (crates/rmlx-kv-quant/src/kvcache/update*.rs): 0 matched lines over 9 body lines (2 item(s), 0 pair(s))" \
    COLLAPSED_UPDATES_ML

check_exit "matched_lines_rotor_updates_collapsed_exit" \
    "a collapsed population is a measured 0, exit 0" \
    0 "$COLLAPSED_UPDATES_STATUS"

rm -rf "$COLLAPSED_WORK"

# ---- --matched-lines: the two iso populations -----------------------------
#
# Same rule as the rotor pair and a different glob and prefix, both supplied at
# the registration site. The fixture plants one two-member group per axis, so a
# population that paired every item with every other would report 6 pairs, not
# 2, and one that read the rotor glob would report the rotor files' figure.

ISO_STORAGE_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines iso-storage 2>&1)
ISO_STORAGE_STATUS=$?

check "matched_lines_iso_storage_pairs" \
    "the four planted iso storage files are a two-member V group and a two-member K group: 1 + 1 = 2 pairs, not the 6 an all-pairs population compares; the rotor files in the same directory are outside this glob and would read 9 item(s) if the glob widened" \
    contains "iso storage twins (crates/rmlx-kv-quant/src/storage): 23 matched lines over 48 body lines (4 item(s), 2 pair(s))" \
    ISO_STORAGE_ML

check_exit "matched_lines_iso_storage_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$ISO_STORAGE_STATUS"

check "matched_lines_iso_storage_label_own_root" \
    "the label names the population's own root, and the iso and rotor storage populations share it" \
    contains "iso storage twins (crates/rmlx-kv-quant/src/storage):" \
    ISO_STORAGE_ML

ISO_UPDATES_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines iso-updates 2>&1)
ISO_UPDATES_STATUS=$?

check "matched_lines_iso_updates_pairs" \
    "the six planted iso fns are the four update_iso* ones plus iso_v_update / iso_sym_update, which an anchored ^update_iso pattern misses entirely — that reads 4 item(s); this population pairs every item with every other, so 6 items give 15 pairs, and a width key would give 2 pairs and never compare the same-width bodies at all; the update_rotor* fns carry no iso token and would read 11 item(s) if the pattern widened to update_, while a token pattern that did not admit a glued width digit would drop update_iso3 / update_iso4 and read 4" \
    contains "iso update fns (crates/rmlx-kv-quant/src/kvcache/update*.rs): 41 matched lines over 28 body lines (6 item(s), 15 pair(s))" \
    ISO_UPDATES_ML

check_exit "matched_lines_iso_updates_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$ISO_UPDATES_STATUS"

# ---- collapsed: the iso width twin is gone, the population is still found --

ISO_COLLAPSED_WORK="$(mktemp -d)"
cp -R "$BASE" "$ISO_COLLAPSED_WORK/base"
rm -f "$ISO_COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/storage/quant_iso_v4.rs" \
    "$ISO_COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/storage/quant_iso_k4.rs"
python3 - "$ISO_COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" <<'EOF'
import re
import sys

path = sys.argv[1]
text = open(path).read()
# Drop the two 4-bit entries the way the collapse did — the 3-bit ones stay.
for name in ("update_iso4", "update_iso_k_only_4"):
    text = re.sub(r"\n    fn " + name + r"\(.*?\n    \}\n", "\n", text, flags=re.S)
open(path, "w").write(text)
EOF

ISO_COLLAPSED_STORAGE_ML=$(python3 "$TOOL" --root "$ISO_COLLAPSED_WORK/base" --matched-lines iso-storage 2>&1)
ISO_COLLAPSED_STORAGE_STATUS=$?

check "matched_lines_iso_storage_collapsed_zero" \
    "deleting the 4-bit files leaves one item per axis and so no pair: 0 matched lines with the population still found, which is a different answer from an empty population" \
    contains "iso storage twins (crates/rmlx-kv-quant/src/storage): 0 matched lines over 24 body lines (2 item(s), 0 pair(s))" \
    ISO_COLLAPSED_STORAGE_ML

check_exit "matched_lines_iso_storage_collapsed_exit" \
    "a collapsed population is a measured 0, exit 0" \
    0 "$ISO_COLLAPSED_STORAGE_STATUS"

ISO_COLLAPSED_UPDATES_ML=$(python3 "$TOOL" --root "$ISO_COLLAPSED_WORK/base" --matched-lines iso-updates 2>&1)
ISO_COLLAPSED_UPDATES_STATUS=$?

check "matched_lines_iso_updates_collapsed_width_twin" \
    "deleting the two 4-bit entries drops the figure from 41 over 6 items to 17 over 4, population still found — a width collapse moves this counter even though it is not keyed on width" \
    contains "iso update fns (crates/rmlx-kv-quant/src/kvcache/update*.rs): 17 matched lines over 19 body lines (4 item(s), 6 pair(s))" \
    ISO_COLLAPSED_UPDATES_ML

check_exit "matched_lines_iso_updates_collapsed_exit" \
    "a collapsed population is still a measurement, exit 0" \
    0 "$ISO_COLLAPSED_UPDATES_STATUS"

rm -rf "$ISO_COLLAPSED_WORK"

# ---- the same-width residual the width key cannot see, and its collapse ----
#
# `iso_v_update` / `iso_sym_update` are one width and two entries. A
# `width_pair_key` population puts them in two groups and compares them never,
# so it would read the same figure whether the duplication is there or not.
# This pair is what says the iso update population can move on it.

ISO_SAMEWIDTH_WORK="$(mktemp -d)"
cp -R "$BASE" "$ISO_SAMEWIDTH_WORK/base"
python3 - "$ISO_SAMEWIDTH_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" <<'EOF'
import re
import sys

path = sys.argv[1]
text = open(path).read()
# Collapse the planted same-width twin the way the engine collapsed its own:
# one body, one caller left.
text = re.sub(r"\n    fn iso_sym_update\(.*?\n    \}\n", "\n", text, flags=re.S)
open(path, "w").write(text)
EOF

ISO_SAMEWIDTH_ML=$(python3 "$TOOL" --root "$ISO_SAMEWIDTH_WORK/base" --matched-lines iso-updates 2>&1)
ISO_SAMEWIDTH_STATUS=$?

check "matched_lines_iso_updates_same_width_twin_drops" \
    "collapsing the planted same-width twin drops the figure from 41 over 6 items to 27 over 5 — the duplication a width key is blind to by construction is one this population reports" \
    contains "iso update fns (crates/rmlx-kv-quant/src/kvcache/update*.rs): 27 matched lines over 23 body lines (5 item(s), 10 pair(s))" \
    ISO_SAMEWIDTH_ML

check_exit "matched_lines_iso_updates_same_width_exit" \
    "still a measurement, exit 0" \
    0 "$ISO_SAMEWIDTH_STATUS"

rm -rf "$ISO_SAMEWIDTH_WORK"

# ---- --matched-lines: the three turbo populations -------------------------
#
# Same rule as the rotor and iso pairs, and one departure: the third population
# roots on a file in another crate, because the turbo width twins are not all
# in one place. Each entry supplies its own glob or pattern at the registration
# site, so widening any of the three is visible as a moved item count below.

TURBO_STORAGE_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines turbo-storage 2>&1)
TURBO_STORAGE_STATUS=$?

check "matched_lines_turbo_storage_pairs" \
    "the two planted quant_k_turbo*.rs files are one group and so one pair; the rotor and iso files share the directory and would read 11 item(s) if the glob widened to quant_*" \
    contains "turbo storage twins (crates/rmlx-kv-quant/src/storage): 15 matched lines over 39 body lines (2 item(s), 1 pair(s))" \
    TURBO_STORAGE_ML

check_exit "matched_lines_turbo_storage_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$TURBO_STORAGE_STATUS"

check "matched_lines_turbo_storage_label_own_root" \
    "the label names the population's own root, which all three storage populations share" \
    contains "turbo storage twins (crates/rmlx-kv-quant/src/storage):" \
    TURBO_STORAGE_ML

TURBO_UPDATES_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines turbo-updates 2>&1)
TURBO_UPDATES_STATUS=$?

check "matched_lines_turbo_updates_pairs" \
    "the pattern is anchored on the symmetric entries, so the two planted update_tsym* fns are the whole population: 2 item(s), 1 pair. The planted update_k8vturbo3 / update_k8vturbo2 are the same file's other turbo width pair and belong to a different collapse — a (turbo|tsym) pattern reads 4 item(s) and 2 pair(s) and would never reach a measured 0 for this one" \
    contains "turbo update twins (crates/rmlx-kv-quant/src/kvcache/update*.rs): 4 matched lines over 10 body lines (2 item(s), 1 pair(s))" \
    TURBO_UPDATES_ML

check_exit "matched_lines_turbo_updates_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$TURBO_UPDATES_STATUS"

TURBO_SSD_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines turbo-ssd 2>&1)
TURBO_SSD_STATUS=$?

check "matched_lines_turbo_ssd_pairs" \
    "the eight planted block_io helpers are four two-member groups, so four pairs; the read_quant_k beside them carries no turbo token and would read 9 item(s) if the pattern widened to (read|write|shape)" \
    contains "turbo ssd helper twins (crates/rmlx-kv-ssd/src/block_io.rs): 12 matched lines over 30 body lines (8 item(s), 4 pair(s))" \
    TURBO_SSD_ML

check_exit "matched_lines_turbo_ssd_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$TURBO_SSD_STATUS"

check "matched_lines_turbo_ssd_label_own_root" \
    "the figure prints the file it was measured over, in the crate it lives in — which is why this is its own population and not a widened turbo-storage glob" \
    contains "turbo ssd helper twins (crates/rmlx-kv-ssd/src/block_io.rs):" \
    TURBO_SSD_ML

check "matched_lines_turbo_ssd_not_storage_root" \
    "and it does not print the kv-quant storage directory, which a shared root would" \
    absent "turbo ssd helper twins (crates/rmlx-kv-quant" \
    TURBO_SSD_ML

# ---- collapsed: the turbo width twins are gone, the populations remain -----
#
# The third case is the one that pins the digit-free SSD pattern: after the
# collapse those helpers keep their names without a width suffix, and a pattern
# that required a digit would find nothing and report unavailable instead of a
# measured 0.

TURBO_COLLAPSED_WORK="$(mktemp -d)"
cp -R "$BASE" "$TURBO_COLLAPSED_WORK/base"
rm -f "$TURBO_COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/storage/quant_k_turbo4.rs"
python3 - "$TURBO_COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" \
    "$TURBO_COLLAPSED_WORK/base/crates/rmlx-kv-ssd/src/block_io.rs" <<'EOF'
import re
import sys

update_path, block_io_path = sys.argv[1], sys.argv[2]

text = open(update_path).read()
# Drop the 4-bit entry the way the collapse does — the 3-bit one stays.
text = re.sub(r"\n    fn update_tsym4\(.*?\n    \}\n", "\n", text, flags=re.S)
open(update_path, "w").write(text)

text = open(block_io_path).read()
# The SSD helpers collapse by losing the width, not by one width winning: the
# 4-bit bodies go and the 3-bit ones are renamed without their suffix.
for name in ("k_turbo4_shape", "write_quant_k_turbo4", "read_tsym4", "read_quant_k_turbo4"):
    text = re.sub(r"\n    fn " + name + r"\(.*?\n    \}\n", "\n", text, flags=re.S)
for old, new in (
    ("k_turbo3_shape", "k_turbo_shape"),
    ("write_quant_k_turbo3", "write_quant_k_turbo"),
    ("read_tsym3", "read_tsym"),
    ("read_quant_k_turbo3", "read_quant_k_turbo"),
):
    text = text.replace(old, new)
open(block_io_path, "w").write(text)
EOF

TURBO_COLLAPSED_STORAGE_ML=$(python3 "$TOOL" --root "$TURBO_COLLAPSED_WORK/base" --matched-lines turbo-storage 2>&1)
TURBO_COLLAPSED_STORAGE_STATUS=$?

check "matched_lines_turbo_storage_collapsed_zero" \
    "deleting the 4-bit file leaves one item and so no pair: 0 matched lines with the population still found, which is a different answer from an empty population" \
    contains "turbo storage twins (crates/rmlx-kv-quant/src/storage): 0 matched lines over 23 body lines (1 item(s), 0 pair(s))" \
    TURBO_COLLAPSED_STORAGE_ML

check_exit "matched_lines_turbo_storage_collapsed_exit" \
    "a collapsed population is a measured 0, exit 0" \
    0 "$TURBO_COLLAPSED_STORAGE_STATUS"

TURBO_COLLAPSED_UPDATES_ML=$(python3 "$TOOL" --root "$TURBO_COLLAPSED_WORK/base" --matched-lines turbo-updates 2>&1)
TURBO_COLLAPSED_UPDATES_STATUS=$?

check "matched_lines_turbo_updates_collapsed_zero" \
    "deleting the 4-bit entry leaves one body: 0 matched lines over the 5 that remain, population still found — and the other turbo width pair in the same file, which this population deliberately does not name, cannot hold the figure above 0" \
    contains "turbo update twins (crates/rmlx-kv-quant/src/kvcache/update*.rs): 0 matched lines over 5 body lines (1 item(s), 0 pair(s))" \
    TURBO_COLLAPSED_UPDATES_ML

check_exit "matched_lines_turbo_updates_collapsed_exit" \
    "a collapsed population is a measured 0, exit 0" \
    0 "$TURBO_COLLAPSED_UPDATES_STATUS"

TURBO_COLLAPSED_SSD_ML=$(python3 "$TOOL" --root "$TURBO_COLLAPSED_WORK/base" --matched-lines turbo-ssd 2>&1)
TURBO_COLLAPSED_SSD_STATUS=$?

check "matched_lines_turbo_ssd_collapsed_zero" \
    "the four collapsed helpers keep their turbo token and lose their width, so the population is still found at 4 item(s) and reports a measured 0; a pattern that required a width digit would report unavailable here, which is the answer a collapse must not produce" \
    contains "turbo ssd helper twins (crates/rmlx-kv-ssd/src/block_io.rs): 0 matched lines over 15 body lines (4 item(s), 0 pair(s))" \
    TURBO_COLLAPSED_SSD_ML

check_exit "matched_lines_turbo_ssd_collapsed_exit" \
    "a collapsed population is a measured 0, exit 0" \
    0 "$TURBO_COLLAPSED_SSD_STATUS"

rm -rf "$TURBO_COLLAPSED_WORK"

# ---- --matched-lines: the ssd-hydrate population, both tree shapes --------
#
# The rule is two exact fn names, because one name spans only one tree: the
# per-arch `hydrate` bodies before the blanket impl, the short `from_hydrated`
# constructors after it. The fixture plants both shapes, so dropping either
# name from the rule turns one of the two figures below red — the pre-collapse
# figure by count, the post-collapse one by going unavailable.

SSD_HYDRATE_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines ssd-hydrate 2>&1)
SSD_HYDRATE_STATUS=$?

check "matched_lines_ssd_hydrate_before" \
    "the two planted per-arch hydrate bodies differ only in the entry name the struct literal carries, so 13 of each 14-line body match; hydrate_from_ssd is a third name and is outside the rule, and the hydrate in prompt_cache_tests.rs is a test path — either one counted would read 3 item(s)" \
    contains "ssd hydrate twins (crates/rmlx-models/src): 13 matched lines over 28 body lines (2 item(s), 1 pair(s))" \
    SSD_HYDRATE_ML

check_exit "matched_lines_ssd_hydrate_before_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$SSD_HYDRATE_STATUS"

# The collapsed tree: each planted arch keeps a short `from_hydrated`
# constructor and no `hydrate` body at all, which is what the blanket impl
# leaves behind.
HYDRATE_WORK="$(mktemp -d)"
cp -R "$BASE" "$HYDRATE_WORK/base"
python3 - "$HYDRATE_WORK/base/crates/rmlx-models/src/hydrate_alpha/prompt_cache.rs" \
    "$HYDRATE_WORK/base/crates/rmlx-models/src/hydrate_beta/prompt_cache.rs" <<'EOF'
import pathlib
import re
import sys

for path in sys.argv[1:]:
    p = pathlib.Path(path)
    text = p.read_text()
    m = re.search(r"impl SsdHydrate<(\w+)> for SsdHydrator \{.*?\n\}\n", text, re.S)
    entry = m.group(1)
    collapsed = (
        f"impl HydratedEntry for {entry} {{\n"
        "    const SHARES_KV: bool = false;\n"
        "\n"
        "    fn from_hydrated(block: HydratedBlock, block_hashes: Vec<u64>) -> Self {\n"
        "        Self {\n"
        "            prompt_token_ids: block.prompt_ids,\n"
        "            block_hashes,\n"
        "            kv_caches: block.kv_caches,\n"
        "            is_ssd_hydrated: true,\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    p.write_text(text[: m.start()] + collapsed + text[m.end() :])
EOF

COLLAPSED_HYDRATE_ML=$(python3 "$TOOL" --root "$HYDRATE_WORK/base" --matched-lines ssd-hydrate 2>&1)
COLLAPSED_HYDRATE_STATUS=$?

check "matched_lines_ssd_hydrate_after" \
    "the collapsed tree has no hydrate body left: the population is the two short from_hydrated constructors, whose 8-line bodies are identical because the entry name sits in the impl header. A rule naming hydrate alone reads unavailable here" \
    contains "ssd hydrate twins (crates/rmlx-models/src): 8 matched lines over 16 body lines (2 item(s), 1 pair(s))" \
    COLLAPSED_HYDRATE_ML

check_exit "matched_lines_ssd_hydrate_after_exit" \
    "the collapsed population is still found, so it is a measurement, exit 0" \
    0 "$COLLAPSED_HYDRATE_STATUS"

rm -rf "$HYDRATE_WORK"

# ---- --matched-lines: the update_-prefixed fns of the update file ---------
#
# The three family populations above each read one codec's twins. This one
# reads every `update_`-prefixed fn of the update file, which is the population
# the "one update body per store shape" step has to shrink. Its pattern is the
# anchored `update_` prefix, so it holds the fns the three family patterns
# select **and** the ones that belong to no family — which is the point: a
# family-keyed counter cannot see a body shared across families.

UPDATE_BODIES_ML=$(python3 "$TOOL" --root "$STATIC_WORK/base" --matched-lines update-bodies 2>&1)
UPDATE_BODIES_STATUS=$?

check "matched_lines_update_bodies_before" \
    "every fn of the update file whose name starts update_, paired with every other: the rotor, iso and turbo per-variant bodies plus update_affine, which belongs to no family and which all three family patterns miss — the iso population's own iso_v_update / iso_sym_update do not carry the prefix, and an unanchored pattern would read 16 item(s)" \
    contains "update_-prefixed fns of the update files (crates/rmlx-kv-quant/src/kvcache/update*.rs): 248 matched lines over 63 body lines (14 item(s), 91 pair(s))" \
    UPDATE_BODIES_ML

check_exit "matched_lines_update_bodies_before_exit" \
    "a population with members is a measurement, exit 0" \
    0 "$UPDATE_BODIES_STATUS"

# One body per store shape, the step's end state: the per-variant bodies are
# gone and one shared body is left. The figure must fall to a measured 0 with
# the population still found, which is a different answer from unavailable.
UPDATE_COLLAPSED_WORK="$(mktemp -d)"
cp -R "$BASE" "$UPDATE_COLLAPSED_WORK/base"
python3 - "$UPDATE_COLLAPSED_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" <<'EOF'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
# Every per-variant body but the first goes; the dispatch enters the one that
# is left. This is the shape the restructure's second step lands.
names = re.findall(r"\n    fn (update_\w+)\(", text)
for name in names[1:]:
    text = re.sub(r"\n    fn " + name + r"\(.*?\n    \}\n", "\n", text, flags=re.S)
path.write_text(text)
EOF

UPDATE_COLLAPSED_ML=$(python3 "$TOOL" --root "$UPDATE_COLLAPSED_WORK/base" --matched-lines update-bodies 2>&1)
UPDATE_COLLAPSED_STATUS=$?

check "matched_lines_update_bodies_after" \
    "one body left and so no pair: a measured 0 with the population still found, which the restructure's second step is judged on" \
    contains "update_-prefixed fns of the update files (crates/rmlx-kv-quant/src/kvcache/update*.rs): 0 matched lines over 4 body lines (1 item(s), 0 pair(s))" \
    UPDATE_COLLAPSED_ML

check_exit "matched_lines_update_bodies_after_exit" \
    "a collapsed population is a measured 0, exit 0" \
    0 "$UPDATE_COLLAPSED_STATUS"

rm -rf "$UPDATE_COLLAPSED_WORK"

# The population emptied: no fn of the file carries the prefix at all. Renaming
# a body out of the prefix is how this differs from collapsing one, and the two
# must not print the same thing.
UPDATE_EMPTY_WORK="$(mktemp -d)"
cp -R "$BASE" "$UPDATE_EMPTY_WORK/base"
python3 - "$UPDATE_EMPTY_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" <<'EOF'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_text(path.read_text().replace("fn update_", "fn apply_"))
EOF

UPDATE_EMPTY_ML=$(python3 "$TOOL" --root "$UPDATE_EMPTY_WORK/base" --matched-lines update-bodies 2>&1)
UPDATE_EMPTY_STATUS=$?

check "matched_lines_update_bodies_empty_unavailable" \
    "renaming every body out of the prefix empties the population: unavailable, not the 0 a collapsed population reads" \
    contains "debt-report --matched-lines update-bodies: unavailable (crates/rmlx-kv-quant/src/kvcache/update*.rs: population is empty)" \
    UPDATE_EMPTY_ML

check_exit "matched_lines_update_bodies_empty_exit" \
    "an unavailable population exits 1" \
    1 "$UPDATE_EMPTY_STATUS"

rm -rf "$UPDATE_EMPTY_WORK"

# ---- absent: zero members, and a missing root, are unavailable not 0 ------

ABSENT_WORK="$(mktemp -d)"
cp -R "$BASE" "$ABSENT_WORK/base"
rm -f "$ABSENT_WORK/base/crates/rmlx-kv-quant/src/storage"/quant_rotor_*.rs \
    "$ABSENT_WORK/base/crates/rmlx-kv-quant/src/storage"/quant_iso_*.rs \
    "$ABSENT_WORK/base/crates/rmlx-kv-quant/src/storage"/quant_k_turbo*.rs
python3 - "$ABSENT_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs" \
    "$ABSENT_WORK/base/crates/rmlx-kv-ssd/src/block_io.rs" <<'EOF'
import sys

update_path, block_io_path = sys.argv[1], sys.argv[2]

text = open(update_path).read().replace("fn update_rotor", "fn update_affine_rotor")
# The iso pattern is unanchored and keys on the codec token, so emptying that
# population means removing the token, not moving it off the front.
text = text.replace("fn update_iso", "fn update_quat")
text = text.replace("fn iso_", "fn quat_")
# The turbo update pattern keys on the codec token as a whole segment and not
# on a prefix, so emptying that population means removing the token.
text = text.replace("fn update_tsym", "fn update_sym_lloyd")
open(update_path, "w").write(text)

# The SSD pattern keys on the codec token and admits no width digit, so
# emptying that population means removing the token.
text = open(block_io_path).read()
text = text.replace("turbo", "lloyd").replace("tsym", "sym_lloyd")
open(block_io_path, "w").write(text)
EOF

EMPTY_STORAGE_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines rotor-storage 2>&1)
EMPTY_STORAGE_STATUS=$?

check "matched_lines_rotor_storage_empty_unavailable" \
    "the directory is there and nothing in it matches the glob: unavailable, never a 0 indistinguishable from a collapsed population" \
    contains "debt-report --matched-lines rotor-storage: unavailable (crates/rmlx-kv-quant/src/storage: population is empty)" \
    EMPTY_STORAGE_ML

check_exit "matched_lines_rotor_storage_empty_exit" \
    "an unavailable population exits 1 — a measurement with no figure behind it must not read as a passing one" \
    1 "$EMPTY_STORAGE_STATUS"

EMPTY_UPDATES_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines rotor-updates 2>&1)
EMPTY_UPDATES_STATUS=$?

check "matched_lines_rotor_updates_empty_unavailable" \
    "renaming every update_rotor* fn out of the prefix empties the population: unavailable, not 0" \
    contains "debt-report --matched-lines rotor-updates: unavailable (crates/rmlx-kv-quant/src/kvcache/update*.rs: population is empty)" \
    EMPTY_UPDATES_ML

check_exit "matched_lines_rotor_updates_empty_exit" \
    "an unavailable population exits 1" \
    1 "$EMPTY_UPDATES_STATUS"

ISO_EMPTY_STORAGE_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines iso-storage 2>&1)
ISO_EMPTY_STORAGE_STATUS=$?

check "matched_lines_iso_storage_empty_unavailable" \
    "the directory is there and nothing in it matches the iso glob: unavailable, never a 0 indistinguishable from a collapsed population" \
    contains "debt-report --matched-lines iso-storage: unavailable (crates/rmlx-kv-quant/src/storage: population is empty)" \
    ISO_EMPTY_STORAGE_ML

check_exit "matched_lines_iso_storage_empty_exit" \
    "an unavailable population exits 1" \
    1 "$ISO_EMPTY_STORAGE_STATUS"

ISO_EMPTY_UPDATES_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines iso-updates 2>&1)
ISO_EMPTY_UPDATES_STATUS=$?

check "matched_lines_iso_updates_empty_unavailable" \
    "renaming the codec token out of every iso fn empties the population: unavailable, not 0" \
    contains "debt-report --matched-lines iso-updates: unavailable (crates/rmlx-kv-quant/src/kvcache/update*.rs: population is empty)" \
    ISO_EMPTY_UPDATES_ML

check_exit "matched_lines_iso_updates_empty_exit" \
    "an unavailable population exits 1" \
    1 "$ISO_EMPTY_UPDATES_STATUS"

TURBO_EMPTY_STORAGE_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines turbo-storage 2>&1)
TURBO_EMPTY_STORAGE_STATUS=$?

check "matched_lines_turbo_storage_empty_unavailable" \
    "the directory is there and nothing in it matches the turbo glob: unavailable, never a 0 indistinguishable from a collapsed population" \
    contains "debt-report --matched-lines turbo-storage: unavailable (crates/rmlx-kv-quant/src/storage: population is empty)" \
    TURBO_EMPTY_STORAGE_ML

check_exit "matched_lines_turbo_storage_empty_exit" \
    "an unavailable population exits 1" \
    1 "$TURBO_EMPTY_STORAGE_STATUS"

TURBO_EMPTY_UPDATES_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines turbo-updates 2>&1)
TURBO_EMPTY_UPDATES_STATUS=$?

check "matched_lines_turbo_updates_empty_unavailable" \
    "renaming the codec token out of every update_tsym* fn empties the population: unavailable, not 0 — moving it off the front would not, since the pattern matches the token wherever in the name it sits" \
    contains "debt-report --matched-lines turbo-updates: unavailable (crates/rmlx-kv-quant/src/kvcache/update*.rs: population is empty)" \
    TURBO_EMPTY_UPDATES_ML

check_exit "matched_lines_turbo_updates_empty_exit" \
    "an unavailable population exits 1" \
    1 "$TURBO_EMPTY_UPDATES_STATUS"

TURBO_EMPTY_SSD_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines turbo-ssd 2>&1)
TURBO_EMPTY_SSD_STATUS=$?

check "matched_lines_turbo_ssd_empty_unavailable" \
    "removing the turbo token from every block_io helper empties the population: unavailable, not 0" \
    contains "debt-report --matched-lines turbo-ssd: unavailable (crates/rmlx-kv-ssd/src/block_io.rs: population is empty)" \
    TURBO_EMPTY_SSD_ML

check_exit "matched_lines_turbo_ssd_empty_exit" \
    "an unavailable population exits 1" \
    1 "$TURBO_EMPTY_SSD_STATUS"

rm -rf "$ABSENT_WORK/base/crates/rmlx-kv-quant/src/storage"
rm -f "$ABSENT_WORK/base/crates/rmlx-kv-quant/src/kvcache/update.rs"
rm -f "$ABSENT_WORK/base/crates/rmlx-kv-ssd/src/block_io.rs"

MISSING_STORAGE_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines rotor-storage 2>&1)
MISSING_STORAGE_STATUS=$?

check "matched_lines_rotor_storage_missing_root" \
    "a missing root names itself, and is told apart from a root that is there and empty" \
    contains "debt-report --matched-lines rotor-storage: unavailable (crates/rmlx-kv-quant/src/storage is not a directory)" \
    MISSING_STORAGE_ML

check_exit "matched_lines_rotor_storage_missing_exit" \
    "a missing root exits 1" \
    1 "$MISSING_STORAGE_STATUS"

MISSING_UPDATES_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines rotor-updates 2>&1)
MISSING_UPDATES_STATUS=$?

check "matched_lines_rotor_updates_missing_root" \
    "a glob root that matches no file names itself, and is told apart from a root whose files hold no matching fn" \
    contains "debt-report --matched-lines rotor-updates: unavailable (crates/rmlx-kv-quant/src/kvcache/update*.rs matches no file)" \
    MISSING_UPDATES_ML

check_exit "matched_lines_rotor_updates_missing_exit" \
    "a missing root exits 1" \
    1 "$MISSING_UPDATES_STATUS"

ISO_MISSING_STORAGE_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines iso-storage 2>&1)
ISO_MISSING_STORAGE_STATUS=$?

check "matched_lines_iso_storage_missing_root" \
    "a missing root names itself, and is told apart from a root that is there and empty" \
    contains "debt-report --matched-lines iso-storage: unavailable (crates/rmlx-kv-quant/src/storage is not a directory)" \
    ISO_MISSING_STORAGE_ML

check_exit "matched_lines_iso_storage_missing_exit" \
    "a missing root exits 1" \
    1 "$ISO_MISSING_STORAGE_STATUS"

ISO_MISSING_UPDATES_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines iso-updates 2>&1)
ISO_MISSING_UPDATES_STATUS=$?

check "matched_lines_iso_updates_missing_root" \
    "a glob root that matches no file names itself, and is told apart from a root whose files hold no matching fn" \
    contains "debt-report --matched-lines iso-updates: unavailable (crates/rmlx-kv-quant/src/kvcache/update*.rs matches no file)" \
    ISO_MISSING_UPDATES_ML

check_exit "matched_lines_iso_updates_missing_exit" \
    "a missing root exits 1" \
    1 "$ISO_MISSING_UPDATES_STATUS"

TURBO_MISSING_STORAGE_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines turbo-storage 2>&1)
TURBO_MISSING_STORAGE_STATUS=$?

check "matched_lines_turbo_storage_missing_root" \
    "a missing root names itself, and is told apart from a root that is there and empty" \
    contains "debt-report --matched-lines turbo-storage: unavailable (crates/rmlx-kv-quant/src/storage is not a directory)" \
    TURBO_MISSING_STORAGE_ML

check_exit "matched_lines_turbo_storage_missing_exit" \
    "a missing root exits 1" \
    1 "$TURBO_MISSING_STORAGE_STATUS"

TURBO_MISSING_SSD_ML=$(python3 "$TOOL" --root "$ABSENT_WORK/base" --matched-lines turbo-ssd 2>&1)
TURBO_MISSING_SSD_STATUS=$?

check "matched_lines_turbo_ssd_missing_root" \
    "the third population's root is a file in another crate, and a missing one names that file and not the storage directory" \
    contains "debt-report --matched-lines turbo-ssd: unavailable (crates/rmlx-kv-ssd/src/block_io.rs is not a file)" \
    TURBO_MISSING_SSD_ML

check_exit "matched_lines_turbo_ssd_missing_exit" \
    "a missing root exits 1" \
    1 "$TURBO_MISSING_SSD_STATUS"

rm -rf "$ABSENT_WORK"

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
    "discovery is structural (signature + RoundTotals), not name-driven — the renamed fn is still found, under its new name" \
    contains "  zzz_renamed_driver" \
    RENAME_OUT

check "renamed_driver_new_path_present" \
    "the renamed fn's resolved path is still the one printed" \
    contains "crates/rmlx-models/src/speculative/mtp.rs:5" \
    RENAME_OUT

check "renamed_driver_count_unchanged" \
    "the total driver count is unaffected by a pure rename (still 1) — a naive name-keyed implementation would lose it" \
    contains "  1 driver(s) found" \
    RENAME_OUT

check "rename_out_not_unavailable" \
    "the rename-case scan the old-name-absent check above reads from is a genuinely healthy scan, not a dead-producer artifact" \
    absent "  unavailable (" \
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
