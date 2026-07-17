#!/usr/bin/env bash
# scripts/check_no_decode_swallow.sh — CI gate: a failed decode step or a failed
# prefill chunk must propagate, never be swallowed into a successful-looking run.
#
# THE BUG THIS PINS
#   Decode: a decode step that errored used to log a warning and `break`, handing
#   the caller the tokens produced so far. The server then saw a non-EOS last
#   token and reported `finish_reason="length"` — byte-identical to hitting the
#   token cap.
#
#   Prefill: a prefill chunk that errored used to log a warning and hand the
#   caller "no logits" with no cause. The caller returned its empty step list and
#   the run completed successfully, reporting `ttft_ms=0 decode_tps=0.000
#   prefill_tps=0.0` — a fabricated zero, emitted as a measurement, with exit 0.
#
#   Both shapes defeat the same thing. Every automated gate in this repo (bench
#   harness, perf canary, regression gate, smoke probes) reads `finish_reason`,
#   token counts, and throughput fields — never the log. A failure that reports
#   as an ordinary outcome passes all of them, and a zero is as dangerous as a
#   plausible number: any harness that averages, medians, or records without
#   refusing zeros silently absorbs it.
#
# WHY A GREP GATE
#   The decode loop and the chunked prefill are copied per-arch (the shared
#   pipelined loop and shared `chunked_prefill`, plus the bitnet / laguna /
#   qwen2 hand-written ones). A unit test needs a concrete `&Model` per arch, so
#   the per-arch copies are effectively untestable at unit scope — and a fix
#   hand-applied to some copies but not others is precisely the "looks complete,
#   isn't" failure this rule exists to prevent. The shared loops carry real unit
#   tests (`pipelined_decode_propagates_step_failure`,
#   `chunked_prefill_propagates_chunk_failure`); this gate covers every copy of
#   the shape.
#
# THE RULE
#   Every failure log site listed below must propagate the cause, either by
#   returning it directly (`return Err`) or — where a mandatory cleanup sweep
#   must run first — by capturing it (`= Some(e)`) for return after the sweep.
#   A `break` with no capture is the swallow, and a `return Ok(` on a failure
#   path is the fabricated success this gate exists to kill.
#
#   The deferred-capture form is not a loophole: the shared `chunked_prefill`
#   cannot `return Err` at the log site because every cache that entered prefill
#   must run `exit_prefill` before the function leaves, or the remaining caches
#   keep un-finalized prefill state that corrupts any later reuse. The cause is
#   captured, the sweep runs, the cause propagates.
#
# NOT YET COVERED
#   The image-prefill failure sites (`image prefill failed`, and the image-path
#   `exit_prefill quantization failed` in gemma3) still `return Ok(steps)`.
#   Adding those anchors to PREFILL_ANCHORS below is a one-line change that goes
#   red on exactly those sites — do it with the fix, not before.
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Sites that must propagate at the log site itself — no cleanup is owed.
DECODE_ANCHORS=(
    'decode step failed'
)

# Sites that may defer propagation until after a mandatory cleanup sweep.
PREFILL_ANCHORS=(
    'prefill chunk failed'
    'prefill chunk cache eval failed'
)

# Lines of context after the anchor in which the verdict must appear. The log
# macro spans several lines (fields, message, closing paren) before the control
# flow; 8 covers the widest current site with room to spare.
CONTEXT=8

violations=0

# Emit "file:line" for every non-test site logging $1.
anchor_sites() {
    grep -rn --include='*.rs' -F "$1" crates/ 2>/dev/null \
        | grep -v '_tests\.rs:' | cut -d: -f1,2 || true
}

# $1 = anchor, $2 = "direct" (return Err only) or "deferred" (capture allowed).
check_anchor() {
    local anchor="$1" mode="$2" site f ln end window
    while IFS= read -r site; do
        [ -z "${site}" ] && continue
        f="${site%%:*}"
        ln="${site##*:}"
        end=$((ln + CONTEXT))
        window="$(sed -n "${ln},${end}p" "${f}")"

        # A fabricated success on a failure path — never allowed, either mode.
        if printf '%s' "${window}" | grep -q 'return Ok('; then
            echo "VIOLATION: ${f}:${ln} — '${anchor}' returns Ok on the failure path."
            echo "    A failure reported as a successful run is the bug this gate pins."
            printf '%s\n' "${window}" | sed 's/^/    | /'
            violations=$((violations + 1))
            continue
        fi

        if printf '%s' "${window}" | grep -q 'return Err'; then
            continue
        fi
        # Deferred propagation: cause captured now, returned after the sweep.
        if [ "${mode}" = "deferred" ] && printf '%s' "${window}" | grep -q '= Some(e)'; then
            continue
        fi

        if [ "${mode}" = "deferred" ]; then
            echo "VIOLATION: ${f}:${ln} — '${anchor}' neither returns the cause ('return Err')"
            echo "    nor captures it for propagation after the cleanup sweep ('= Some(e)')"
            echo "    within ${CONTEXT} lines."
        else
            echo "VIOLATION: ${f}:${ln} — '${anchor}' is not followed by 'return Err' within ${CONTEXT} lines."
        fi
        printf '%s\n' "${window}" | sed 's/^/    | /'
        violations=$((violations + 1))
    done < <(anchor_sites "${anchor}")
}

for anchor in "${DECODE_ANCHORS[@]}"; do
    check_anchor "${anchor}" direct
done
for anchor in "${PREFILL_ANCHORS[@]}"; do
    check_anchor "${anchor}" deferred
done

if [ "${violations}" -gt 0 ]; then
    echo
    echo "FAIL: ${violations} failure site(s) swallow the error."
    echo "A decode step that fails must abort the request — returning the tokens"
    echo "produced so far is reported as finish_reason=\"length\", indistinguishable"
    echo "from a clean token-cap stop. A prefill chunk that fails must abort too —"
    echo "returning no logits without a cause completes the run and reports zeros"
    echo "as a measurement. Every automated gate reads those fields, not the log."
    exit 1
fi

echo "OK: every decode-step and prefill-chunk failure site propagates (no swallow)."
