#!/usr/bin/env bash
# scripts/check_kernel_dtype_contract_fixtures.sh — recall test for the
# `check_kernel_dtype_contract.sh` gate.
#
# A source gate is only worth its runtime if it still fires when the pattern it
# looks for is spelled differently, hidden inside a tuple, or separated from
# its marker by an attribute block. Each fixture under
# `scripts/fixtures/kernel_dtype_contract/` is a synthetic scan root paired
# with the exit code the gate must produce and, for the failing cases, the
# function it must name. Four of them are bypasses that were reproduced against
# an earlier revision of the gate: a rustfmt-split declaration it could not see,
# a second escaping buffer that one cast exempted, a derived cast on an input
# that counted as a guard, and a marker that leaked across a `const`. Asserting the named function — not just the exit code
# — is what stops a gate that fails for the wrong reason from counting as
# recall.
#
# Exit 0 = every fixture produced its expected exit code and message.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE="$ROOT/scripts/check_kernel_dtype_contract.sh"
FIX="$ROOT/scripts/fixtures/kernel_dtype_contract"

# fixture | min-fns | expected-exit | expected-substring | what it proves
CASES=(
    "clean|1|0|check-kernel-dtype-contract: ok|a derived .astype( on the way out passes"
    "unguarded_return|1|1|leaky_sdpa|an f32 kernel output returned as-is is caught"
    "f32_literal_astype|1|1|upcast_only_sdpa|the .astype(Dtype::F32) upcast is not a guard"
    "literal_width_astype|1|1|pinned_width_sdpa|a literal-width .astype( is not a guard"
    "marker_above|1|0|check-kernel-dtype-contract: ok|an f32-out-ok marker in the block above exempts the fn"
    "marker_through_attributes|1|0|check-kernel-dtype-contract: ok|a marker separated by a multi-line attribute still applies"
    "marker_does_not_leak|1|1|leaky_sdpa|one marker exempts one function, not the next one"
    "tuple_scales|1|1|quantize_gpu|an f32 scale buffer returned in a tuple is caught"
    "partial_tuple_guard|1|1|quantize_partial_guard_gpu|casting one of two escaping f32 buffers is not enough"
    "multiline_add_output_shape|1|1|split_decl_sdpa|a rustfmt-split add_output_shape is still seen"
    "derived_astype_on_input|1|1|input_cast_sdpa|a derived cast BEFORE the dispatch guards an input, not the output"
    "marker_over_const|1|1|leaky_after_const_sdpa|a marker above a const does not leak onto the next fn"
    "nested_dir|1|1|leaky_sdpa|a dispatcher in a sub-directory is still scanned"
    "not_a_dispatcher|1|1|expected at least|a file with no MetalKernelInvoke is out of scope"
    "test_file_excluded|1|1|expected at least|a *_tests.rs dispatcher is out of scope"
    "empty_tree|1|1|expected at least|an empty scan root fails instead of passing vacuously"
    "clean|2|1|expected at least|fewer dispatcher functions than the pinned floor fails"
)

FAILED=0
PASSED=0

for case in "${CASES[@]}"; do
    IFS='|' read -r name min want want_msg what <<<"$case"
    out=$("$GATE" "$FIX/$name" "$min" 2>&1)
    got=$?
    if [ "$got" -eq "$want" ] && printf '%s' "$out" | grep -qF -- "$want_msg"; then
        PASSED=$((PASSED + 1))
        printf '  ok   %-26s min=%s exit=%s — %s\n' "$name" "$min" "$got" "$what"
    else
        FAILED=$((FAILED + 1))
        printf '  FAIL %-26s min=%s exit=%s (want %s, want message %s) — %s\n' \
            "$name" "$min" "$got" "$want" "$want_msg" "$what"
        printf '%s\n' "$out" | sed 's/^/       | /'
    fi
done

if [ "$FAILED" -ne 0 ]; then
    echo "check-kernel-dtype-contract fixtures: FAIL ($FAILED of $((PASSED + FAILED)))" >&2
    exit 1
fi

echo "check-kernel-dtype-contract fixtures: ok ($PASSED cases)"
