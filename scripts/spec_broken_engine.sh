#!/usr/bin/env bash
# Run the answer-equivalence gate against a deliberately broken round loop.
#
# The gate's constants are set against a measured population: every pair as
# shipped, and a broken engine per pair. Those readings are worth no more than
# the ability to take them again, so each broken engine is a named edit here
# rather than a number in a document nobody can regenerate.
#
#   scripts/spec_broken_engine.sh <engine> <test-name>
#
# Engines, all on the EAGLE-3 round loop:
#
#   shipped              the tree as it stands, no edit
#   correction-restricted  the round's correction left on the drafter's
#                          restricted argmax, which widens the declared
#                          vocabulary boundary from "sometimes at an accepted
#                          position" to "always at the correction"
#   rejected-draft-kept    the verifier's rollback target one position short,
#                          so every partial round keeps one rejected draft
#
# The edit is applied to a working tree that must be clean of it beforehand:
# the pattern being replaced has to occur exactly once, and two occurrences is
# a hard failure rather than a reason to take the first — an edit that lands
# somewhere other than where the run is named for reports a result that was
# never taken. The file is restored from a byte snapshot and the restore is
# verified by digest, so a failed or interrupted run cannot leave the tree
# mutated.
#
# Environment: RMLX_O_MODELS_ROOT and RMLX_DRAFT_TEST_MODEL, as the gate itself
# documents. No GPU work happens here that the gate would not do on its own.
set -uo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TARGET="$ROOT/crates/rmlx-models/src/speculative/eagle3/mod.rs"

engine="${1:-}"
test_name="${2:-}"
if [[ -z "$engine" || -z "$test_name" ]]; then
    sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi

case "$engine" in
shipped)
    find=""
    replace=""
    ;;
correction-restricted)
    find='            tokens[full_pos] = u32::from_le_bytes(corr_bytes[..4].try_into().unwrap());'
    replace='            let _ = &corr_bytes;'
    ;;
rejected-draft-kept)
    find='        let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32);'
    replace='        let v_target = v_offset_before - (draft_tokens.len() as i32 - accept as i32) + 1;'
    ;;
*)
    echo "unknown engine: $engine" >&2
    exit 2
    ;;
esac

snapshot=""
restore() {
    [[ -n "$snapshot" ]] || return 0
    /bin/cp -f "$snapshot" "$TARGET"
    local now
    now="$(shasum -a 256 "$TARGET" | cut -d' ' -f1)"
    if [[ "$now" != "$original_digest" ]]; then
        echo "FATAL: $TARGET was not restored ($now != $original_digest)" >&2
        exit 3
    fi
    rm -f "$snapshot"
    echo "restored $TARGET, digest $original_digest"
}

if [[ -n "$find" ]]; then
    occurrences="$(grep -cF -- "$find" "$TARGET")"
    if [[ "$occurrences" != "1" ]]; then
        echo "FATAL: the '$engine' edit matches $occurrences lines of $TARGET, and it must \
match exactly one — the run would not be the run it is named for" >&2
        exit 2
    fi
    snapshot="$(mktemp)"
    /bin/cp -f "$TARGET" "$snapshot"
    original_digest="$(shasum -a 256 "$snapshot" | cut -d' ' -f1)"
    trap restore EXIT
    : >"$TARGET"
    while IFS= read -r line; do
        if [[ "$line" == "$find" ]]; then
            printf '%s\n' "$replace" >>"$TARGET"
        else
            printf '%s\n' "$line" >>"$TARGET"
        fi
    done <"$snapshot"
    echo "applied '$engine' to $TARGET"
fi

echo "running $test_name against the '$engine' engine"
cargo test -p rmlx-models --test spec_greedy_equivalence -- \
    --ignored --nocapture --test-threads=1 "$test_name"
status=$?
echo "$test_name against '$engine': cargo test exit $status"
exit "$status"
