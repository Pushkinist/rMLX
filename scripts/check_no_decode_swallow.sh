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
#   NaN detection: a prefill whose logits contained NaN used to be detected,
#   emit one junk token (greedy selection over an all-NaN row returns index 0
#   whatever the model computed), and `return Ok` — logging nothing at any
#   level. The run reported one token, `decode_tps` ~0.06, and exit 0.
#
#   Both shapes defeat the same thing. Every automated gate in this repo (bench
#   harness, perf canary, regression gate, smoke probes) reads `finish_reason`,
#   token counts, and throughput fields — never the log. A failure that reports
#   as an ordinary outcome passes all of them, and a zero is as dangerous as a
#   plausible number: any harness that averages, medians, or records without
#   refusing zeros silently absorbs it.
#
# WHY A GREP GATE
#   The decode loop and the chunked prefill are copied per-arch. A unit test
#   needs a concrete `&Model` per arch, so the per-arch copies are effectively
#   untestable at unit scope — and a fix hand-applied to some copies but not
#   others is precisely the "looks complete, isn't" failure this rule exists to
#   prevent. The shared loops carry real unit tests
#   (`pipelined_decode_propagates_step_failure`,
#   `chunked_prefill_propagates_chunk_failure`); this gate covers the copies.
#
# WHY THE SWALLOW RULE IS KEYED ON SHAPE, NOT ON MESSAGE TEXT
#   An earlier revision anchored on log message text ("prefill chunk failed").
#   Two live swallows evaded it purely by wording ("prefix tail chunk failed",
#   "tail chunk failed") and the gate reported OK with the pinned bug present.
#   Message-keyed anchors are whack-a-mole. RULE 1 therefore anchors on the
#   structural marker every failure-log site in the arch layer shares —
#   `error = %e` — and looks for the swallow itself.
#
# THE RULES
#   1. SWALLOW (shape-keyed, arch layer): no failure-log site may return Ok or
#      set a `*_ok = false` swallow flag from inside its own failure arm.
#   2. DECODE (message-keyed): every `decode step failed` site must `return Err`.
#   3. SWEEP (message-keyed): the shared chunked_prefill must CAPTURE its cause
#      (`= Some(e)`), not `return Err` inline — see RULE 3 below.
#   4. DETECT (shape-keyed, arch layer): a site that COUNTS NaN in a logit row
#      must propagate the finding. RULE 1 cannot see this shape at all: it keys
#      on `error = %e`, and a detection that logs nothing has no anchor.
#
# SCOPE (what this does NOT cover)
#   RULE 1 scans crates/rmlx-models/src only — the arch generate/prefill paths
#   where this class lives. Failure sites in other crates are not covered.
#   RULE 1 keys on `error = %e`; a failure-log site that does not carry the
#   error as a structured field is invisible to it (and is a traceability bug
#   in its own right — see the tracing rules in CLAUDE.md). RULE 4 exists
#   precisely because that anchor is missing from a detect-and-discard site.
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Lines of context after an anchor in which the verdict must appear. The log
# macro spans several lines (fields, message, closing paren) before the control
# flow; 8 covers the widest current site with room to spare.
CONTEXT=8

violations=0

# Indent (leading spaces) of a line. rustfmt guarantees consistent indentation,
# which is what makes the nesting test below meaningful.
indent_of() { printf '%s' "$1" | awk '{ match($0, /[^ ]/); print RSTART - 1 }'; }

# ── RULE 1: no swallow at any arch-layer failure-log site ────────────────────
#
# Anchor: `error = %e`, the structural marker shared by every failure-log site
# in the arch layer.
#
# Verdict: a `return Ok(` or `*_ok = false` inside the anchor's own failure arm
# is the swallow. The discriminator is NESTING, not distance: a swallow is
# nested at or below the log statement, whereas a correct degrade-and-continue
# site (`warn!` + fall through, e.g. "skipping prompt cache store") is followed
# only by later SIBLING blocks, which are dedented relative to it.
#
# The `- 4` slack is one macro-continuation level: the anchor is often an
# argument line of a multi-line `tracing::error!(...)`, one level deeper than
# the statement it belongs to.
check_swallow() {
    local site f ln anchor_line a_ind limit win l l_ind
    while IFS= read -r site; do
        [ -z "${site}" ] && continue
        f="${site%%:*}"
        ln="${site##*:}"
        anchor_line="$(sed -n "${ln}p" "${f}")"
        a_ind="$(indent_of "${anchor_line}")"
        limit=$((a_ind - 4))
        win="$(sed -n "${ln},$((ln + CONTEXT))p" "${f}")"
        while IFS= read -r l; do
            printf '%s' "${l}" | grep -q 'return Ok(\|_ok = false' || continue
            l_ind="$(indent_of "${l}")"
            [ "${l_ind}" -ge "${limit}" ] || continue
            echo "VIOLATION: ${f}:${ln} — failure-log site swallows the cause."
            echo "    A 'return Ok(' / '*_ok = false' inside the failure arm reports a failed"
            echo "    prefill or decode as a successful run. Propagate the cause instead."
            printf '%s\n' "${win}" | sed 's/^/    | /'
            violations=$((violations + 1))
            break
        done <<< "${win}"
    done < <(grep -rn --include='*.rs' -F 'error = %e' crates/rmlx-models/src/ \
        | grep -v '_tests\.rs:' | cut -d: -f1,2 || true)
}

# ── RULE 2: a failed decode step must return Err at the site ─────────────────
check_decode() {
    local site f ln win
    while IFS= read -r site; do
        [ -z "${site}" ] && continue
        f="${site%%:*}"
        ln="${site##*:}"
        win="$(sed -n "${ln},$((ln + CONTEXT))p" "${f}")"
        if ! printf '%s' "${win}" | grep -q 'return Err'; then
            echo "VIOLATION: ${f}:${ln} — 'decode step failed' is not followed by 'return Err' within ${CONTEXT} lines."
            printf '%s\n' "${win}" | sed 's/^/    | /'
            violations=$((violations + 1))
        fi
    done < <(grep -rn --include='*.rs' -F 'decode step failed' crates/ \
        | grep -v '_tests\.rs:' | cut -d: -f1,2 || true)
}

# ── RULE 3: the shared chunked_prefill must capture, not return inline ───────
#
# These two sites are the ONLY ones that owe a cleanup sweep: chunked_prefill
# calls `enter_prefill` on every cache up front, so it must run `exit_prefill`
# on every cache before it leaves. A `return Err` at the log site skips the
# sweep for the remaining caches, stranding them `in_prefill = true` — the next
# decode on a stuck cache errors or corrupts KV. So here, and only here, the
# correct form is to capture the cause and let the post-loop sweep run first.
#
# Message-keyed on purpose: "owes a sweep" is a property of this one function,
# not a shape visible to grep. RULE 1 still covers these sites against the
# swallow itself, and `chunked_prefill_runs_exit_sweep_on_failure` pins the
# ordering invariant with a real test — this rule is the cheap early warning.
SWEEP_ANCHORS=(
    'prefill chunk failed'
    'prefill chunk cache eval failed'
)

check_sweep() {
    local anchor="$1" site f ln win
    while IFS= read -r site; do
        [ -z "${site}" ] && continue
        f="${site%%:*}"
        ln="${site##*:}"
        # Only the shared helper owes a sweep; the per-arch copies that never
        # call enter_prefill correctly return inline and are covered by RULE 1.
        case "${f}" in
        *decode_loop.rs) ;;
        *) continue ;;
        esac
        win="$(sed -n "${ln},$((ln + CONTEXT))p" "${f}")"
        if printf '%s' "${win}" | grep -q 'return Err'; then
            echo "VIOLATION: ${f}:${ln} — '${anchor}' returns Err inline, skipping the exit_prefill sweep."
            echo "    Every cache that entered prefill must run exit_prefill before this"
            echo "    function leaves. Capture the cause ('= Some(e)') and let the sweep run."
            printf '%s\n' "${win}" | sed 's/^/    | /'
            violations=$((violations + 1))
        elif ! printf '%s' "${win}" | grep -q '= Some(e)'; then
            echo "VIOLATION: ${f}:${ln} — '${anchor}' neither captures the cause ('= Some(e)')"
            echo "    for propagation after the exit_prefill sweep, nor propagates it at all."
            printf '%s\n' "${win}" | sed 's/^/    | /'
            violations=$((violations + 1))
        fi
    done < <(grep -rn --include='*.rs' -F "${anchor}" crates/ \
        | grep -v '_tests\.rs:' | cut -d: -f1,2 || true)
}

# ── RULE 4: a counted NaN must propagate, not be discarded ───────────────────
#
# Anchor: a call to `count_nan_in_bytes(` in the arch layer — the detector
# itself, an API name rather than a log message, so it cannot be evaded by
# rewording. (Definitions of the helper are skipped; only call sites count.)
#
# Verdict: within NAN_CONTEXT lines the finding must reach a propagating form —
# `reject_nan_prefill(` (the shared prefill guard) or a literal `return Err`.
# Neither present means the site computed the count and then dropped it, which
# is what produced a one-token run reported as success.
#
# The window is deliberately narrow: it is one error construction, one
# structured `error!`, and the return. "Somewhere later in the function" is not
# what this pins — the guard belongs immediately after the detection, before the
# offending token is selected and handed to `step_fn`.
#
# ORDERING. A `return Err` anywhere in the window is not enough: propagating
# AFTER `step_fn` has already shipped the token is the original bug shape with
# `Err` substituted for `Ok`. The caller has been handed a junk token either
# way, and on the server that junk token is on the wire and in `delivered`,
# which is exactly the configuration the Migratable classification in
# docs/SERVER.md argues is safe *because* it cannot happen. So a `step_fn` call
# appearing before the propagation inside the window is itself a violation.
#
# ANCHOR FLOORS. A grep gate that iterates whatever grep finds passes silently
# when it finds nothing, so one rename or one wrapper extraction makes it
# vacuous — which is how the shape this rule pins got past RULE 1 in the first
# place. Both counts are pinned to exact values. They are expected to change
# when an architecture is added or a path is refactored; the point is that the
# change has to be a deliberate edit here, re-verified, rather than a silent
# loss of coverage.
#
# KNOWN LIMIT, stated rather than papered over: these counts see call sites, not
# reachability. An arch that adds a correctly-guarded wrapper and then never
# calls it from its prefill path satisfies both the per-site check and the
# floors. Wrapper call sites are counted below precisely to keep that surface
# visible, but "every prefill path reaches a guard" is a review property, not a
# grep property.
NAN_CONTEXT=16

# Call sites of the detector in the arch layer.
EXPECTED_NAN_DETECT_SITES=11
# Call sites of the guard, counting the two wrappers that fan out to several
# prefill paths (qwen3_vl_moe text+image; the five speculative drivers).
EXPECTED_GUARD_CALL_SITES=17
GUARD_CALL_RE='reject_nan_prefill\(|guard_prefill_logits\(|guard_verifier_prefill_logits\('

nan_detect_sites() {
    grep -rn --include='*.rs' -F 'count_nan_in_bytes(' crates/rmlx-models/src/ \
        | grep -v '_tests\.rs:' | grep -v 'fn count_nan_in_bytes(' | cut -d: -f1,2 || true
}

guard_call_sites() {
    grep -rnE --include='*.rs' "${GUARD_CALL_RE}" crates/rmlx-models/src/ \
        | grep -v '_tests\.rs:' | grep -vE 'fn (reject_nan_prefill|guard_prefill_logits|guard_verifier_prefill_logits)\(' \
        | cut -d: -f1,2 || true
}

# Line offset of the first match of a pattern within the window, or empty.
# "No match" is a normal answer here, not a failure — without the `|| true` the
# script runs under `set -e`/`pipefail` and aborts silently with exit 1, which
# would look exactly like a passing gate that printed nothing.
first_offset() {
    printf '%s\n' "$2" | grep -nE "$1" | head -1 | cut -d: -f1 || true
}

check_nan_detect() {
    local site f ln win guard_at step_at n_detect n_guard
    while IFS= read -r site; do
        [ -z "${site}" ] && continue
        f="${site%%:*}"
        ln="${site##*:}"
        win="$(sed -n "${ln},$((ln + NAN_CONTEXT))p" "${f}")"
        guard_at="$(first_offset 'reject_nan_prefill\(|return Err' "${win}")"
        if [ -z "${guard_at}" ]; then
            echo "VIOLATION: ${f}:${ln} — NaN is counted here and never propagated."
            echo "    Within ${NAN_CONTEXT} lines there is no 'reject_nan_prefill(' and no"
            echo "    'return Err'. A NaN logit row makes greedy selection return a fixed"
            echo "    junk token; returning that as Ok reports a fault as a measurement."
            printf '%s\n' "${win}" | sed 's/^/    | /'
            violations=$((violations + 1))
            continue
        fi
        # `step_fn(` and `(ctx.step_fn)(` are calls; the bare `step_fn,` field
        # init inside a DecodeCtx literal is not, and must not match.
        step_at="$(first_offset 'step_fn\(|step_fn\)\(' "${win}")"
        if [ -n "${step_at}" ] && [ "${step_at}" -lt "${guard_at}" ]; then
            echo "VIOLATION: ${f}:${ln} — the token is emitted BEFORE the NaN is propagated."
            echo "    'step_fn' is called at window line ${step_at}, the propagation at"
            echo "    ${guard_at}. Erring after emitting is the original bug shape with Err"
            echo "    in place of Ok: the caller still received the junk token, and on the"
            echo "    server it is already on the wire and in the retry envelope's"
            echo "    'delivered' prefix. Raise before the emit."
            printf '%s\n' "${win}" | sed 's/^/    | /'
            violations=$((violations + 1))
        fi
    done < <(nan_detect_sites)

    n_detect="$(nan_detect_sites | grep -c . || true)"
    n_guard="$(guard_call_sites | grep -c . || true)"
    if [ "${n_detect}" -ne "${EXPECTED_NAN_DETECT_SITES}" ]; then
        echo "VIOLATION: NaN detector call sites = ${n_detect}, expected ${EXPECTED_NAN_DETECT_SITES}."
        echo "    This rule iterates whatever grep finds, so it goes vacuous the moment the"
        echo "    anchors disappear. If the change is intended, re-verify every prefill path"
        echo "    still reaches a guard and update EXPECTED_NAN_DETECT_SITES."
        nan_detect_sites | sed 's/^/    | /'
        violations=$((violations + 1))
    fi
    if [ "${n_guard}" -ne "${EXPECTED_GUARD_CALL_SITES}" ]; then
        echo "VIOLATION: NaN guard call sites = ${n_guard}, expected ${EXPECTED_GUARD_CALL_SITES}."
        echo "    A prefill path that stopped calling the guard does not remove a detector"
        echo "    anchor, so only this count sees it. If the change is intended, re-verify"
        echo "    coverage and update EXPECTED_GUARD_CALL_SITES."
        guard_call_sites | sed 's/^/    | /'
        violations=$((violations + 1))
    fi
}

check_swallow
check_decode
for anchor in "${SWEEP_ANCHORS[@]}"; do
    check_sweep "${anchor}"
done
check_nan_detect

if [ "${violations}" -gt 0 ]; then
    echo
    echo "FAIL: ${violations} failure site(s) swallow the error, skip the prefill"
    echo "sweep, or discard a detected NaN."
    echo "A decode step that fails must abort the request — returning the tokens"
    echo "produced so far is reported as finish_reason=\"length\", indistinguishable"
    echo "from a clean token-cap stop. A prefill chunk that fails must abort too —"
    echo "returning no logits without a cause completes the run and reports zeros"
    echo "as a measurement. A NaN logit row is the same class one step earlier:"
    echo "greedy selection over it returns a fixed junk token, and returning that"
    echo "as Ok reports the fault as a one-token generation with exit 0."
    echo "Every automated gate reads those fields, not the log."
    exit 1
fi

echo "OK: no swallow at any arch-layer failure-log site; decode + prefill propagate;"
echo "    every counted NaN is propagated."
