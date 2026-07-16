#!/usr/bin/env bash
# scripts/check_no_decode_swallow.sh — CI gate: a failed decode step must
# propagate, never be swallowed into a short-but-successful generation.
#
# THE BUG THIS PINS
#   A decode step that errored used to log a warning and `break`, handing the
#   caller the tokens produced so far. The server then saw a non-EOS last token
#   and reported `finish_reason="length"` — byte-identical to hitting the token
#   cap. Every automated gate in this repo (bench harness, perf canary,
#   regression gate, smoke probes) reads exactly `finish_reason` + token counts,
#   so a stream that died mid-flight passed all of them.
#
# WHY A GREP GATE
#   The decode loop is copied per-arch (the shared pipelined loop plus the
#   bitnet / laguna / qwen2 hand-written ones). A unit test needs a concrete
#   `&Model` per arch, so the per-arch copies are effectively untestable at unit
#   scope — and a fix hand-applied to some copies but not others is precisely
#   the "looks complete, isn't" failure this rule exists to prevent. The shared
#   loop carries a real unit test (`pipelined_decode_propagates_step_failure`);
#   this gate covers every copy of the shape.
#
# THE RULE
#   Every `decode step failed` log site must `return Err` within the next few
#   lines. A `break` there is the swallow.
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ANCHOR='decode step failed'
# Lines of context after the anchor in which the verdict must appear. The log
# macro spans several lines (fields, message, closing paren) before the control
# flow; 8 covers the widest current site with room to spare.
CONTEXT=8

violations=0

# Every site that logs the anchor, as "file:line".
while IFS= read -r site; do
    [ -z "${site}" ] && continue
    f="${site%%:*}"
    ln="${site##*:}"
    end=$((ln + CONTEXT))
    window="$(sed -n "${ln},${end}p" "${f}")"
    if ! printf '%s' "${window}" | grep -q 'return Err'; then
        echo "VIOLATION: ${f}:${ln} — 'decode step failed' is not followed by 'return Err' within ${CONTEXT} lines."
        printf '%s\n' "${window}" | sed 's/^/    | /'
        violations=$((violations + 1))
    fi
done < <(grep -rn --include='*.rs' -F "${ANCHOR}" crates/ | grep -v '_tests\.rs:' | cut -d: -f1,2 || true)

if [ "${violations}" -gt 0 ]; then
    echo
    echo "FAIL: ${violations} decode-step failure site(s) swallow the error."
    echo "A decode step that fails must abort the request — returning the tokens"
    echo "produced so far is reported as finish_reason=\"length\", indistinguishable"
    echo "from a clean token-cap stop, and every automated gate reads that field."
    exit 1
fi

echo "OK: every decode-step failure site propagates (no swallow)."
