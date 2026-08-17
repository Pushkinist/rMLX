#!/usr/bin/env bash
# scripts/check_eval_lock.sh — CI gate: every FFI call that can reach
# `mlx::core::eval_impl` must be made under the process-wide evaluation lock.
#
# THE BUG THIS PINS
#   The linked MLX (0.31.x) resolves a CPU stream's `CommandEncoder` through a
#   *process-global* `std::unordered_map<int, CommandEncoder>` that
#   `mlx/backend/cpu/encoder.cpp::get_command_encoder` fills lazily, on the
#   evaluating thread, with no synchronisation. Default CPU streams are
#   per-thread, so every thread that evaluates mints its own stream index and
#   performs its own insert into that one shared map. Two inserts in flight
#   together rehash it under a third thread's bucket walk.
#
#   The result is a SIGSEGV (or SIGTRAP, or an infinite spin on a bucket chain
#   that became circular) *inside MLX*, on a thread named after whichever test
#   happened to be running. libtest names no failing test, because none failed
#   — the process died. `cargo test` runs one OS thread per test, so this
#   reached `make ci` as an intermittent crash of a whole test binary that
#   never reproduced in isolation and was indistinguishable from a real
#   regression in the branch under test.
#
#   `rmlx_mlx::with_eval_lock` contains it by serialising evaluation
#   process-wide.
#
# WHY A GREP GATE AND NOT A TEST
#   The reproducer for this (`concurrent_first_eval_reproducer`) is
#   probabilistic — measured at roughly one failure in twelve runs without the
#   lock. That is far too weak to gate on: eleven times in twelve, deleting the
#   lock would go green. The *structural* regressions, though, are exactly the
#   ones a text gate catches deterministically:
#
#     1. Someone drops the lock from a guarded call site.
#     2. Someone adds a call to one of the other C entry points that evaluate.
#
#   Case 2 is the live risk. Only two of the seventeen reachable entry points
#   are called today; the other fifteen sit in the generated bindings, one call
#   away, and look completely innocuous — `mlx_array_item_float32` reads like a
#   scalar accessor, not an evaluation.
#
# THE REACH-SET (derived from the generated mlx-c bindings, not guessed)
#   Reaches `eval_impl`, therefore needs the lock:
#     - mlx_array_eval          (called here, guarded)
#     - mlx_async_eval          (called here, guarded)
#     - mlx_eval                (not called here)
#     - mlx_array_item_*        (14 of them, not called here) — each goes
#                               through `array::item<T>()`, which calls
#                               `eval()` before reading the value; see
#                               `item()` in mlx/array.h.
#   Does NOT evaluate, needs no lock:
#     - mlx_array_data_*        (plain pointer accessors)
#
# THE RULES
#   RULE 1  No call to `sys::mlx_eval(` or `sys::mlx_array_item_*(` anywhere.
#           These have no guarded wrapper, so any call is unguarded by
#           construction. If one is genuinely needed, wrap it in
#           `with_eval_lock` and add it to RULE 2's allowed set here.
#   RULE 2  Every call to `sys::mlx_array_eval(` / `sys::mlx_async_eval(` must
#           have `with_eval_lock` within the 3 lines ending at the call, so a
#           rustfmt line-wrap does not trip it.
#
# WHAT THIS GATE CANNOT REACH — know the boundary before trusting it
#   * It is a text scan. A call made through an alias (`use ... as f; f()`), a
#     macro expansion, a function pointer or a transmute is invisible to it.
#   * The reach-set above is hand-derived from the bindings as they are today.
#     An mlx-c bump that adds a new evaluating entry point will not be flagged
#     until someone extends RULE 1's pattern list. Re-derive it when the
#     mlx / mlx-c pin moves.
#   * It proves lexical adjacency, not that the lock is held at runtime. A
#     `with_eval_lock` call on a neighbouring but unrelated line would satisfy
#     it. The closure-taking signature of `with_eval_lock` is what makes that
#     shape hard to write by accident.
#   * It says nothing about concurrent *graph construction*, which never
#     reaches `eval_impl`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCAN_DIR="${REPO_ROOT}/crates"

if [ ! -d "${SCAN_DIR}" ]; then
    echo "check_eval_lock: no crates dir at ${SCAN_DIR}" >&2
    exit 1
fi

fail=0

# ---- RULE 1: entry points with no guarded wrapper must not be called -------
#
# Anchored on `sys::` + an open paren, so prose in comments and docs that names
# these functions in backticks does not trip the gate.
banned_hits="$(grep -rnE 'sys::(mlx_eval|mlx_array_item_[A-Za-z0-9_]*)[[:space:]]*\(' \
    --include='*.rs' "${SCAN_DIR}" || true)"

if [ -n "${banned_hits}" ]; then
    fail=1
    echo "check_eval_lock: RULE 1 — call to an MLX entry point that evaluates but has no guarded wrapper." >&2
    echo "${banned_hits}" | sed 's/^/  /' >&2
    echo >&2
    echo "  These reach mlx::core::eval_impl and therefore the unsynchronised" >&2
    echo "  process-global CPU command-encoder map. Wrap the call in" >&2
    echo "  rmlx_mlx::with_eval_lock and add it to this gate's allowed set." >&2
    echo >&2
fi

# ---- RULE 2: guarded entry points must be called under the lock ------------

guarded_hits="$(grep -rnE 'sys::(mlx_array_eval|mlx_async_eval)[[:space:]]*\(' \
    --include='*.rs' "${SCAN_DIR}" || true)"

if [ -z "${guarded_hits}" ]; then
    fail=1
    echo "check_eval_lock: RULE 2 — no call to sys::mlx_array_eval / sys::mlx_async_eval found at all." >&2
    echo "  The gate keys on those call sites; if they were renamed or removed," >&2
    echo "  this gate is no longer checking anything and must be updated." >&2
    echo >&2
fi

while IFS= read -r hit; do
    [ -z "${hit}" ] && continue
    file="${hit%%:*}"
    rest="${hit#*:}"
    line="${rest%%:*}"

    # Window: the call line and the 3 lines above it, so a rustfmt wrap of
    # `with_eval_lock(|| unsafe { ... })` across lines still satisfies the gate.
    start=$(( line > 3 ? line - 3 : 1 ))
    window="$(sed -n "${start},${line}p" "${file}")"

    if ! printf '%s' "${window}" | grep -q 'with_eval_lock'; then
        fail=1
        echo "check_eval_lock: RULE 2 — evaluation FFI call not under the evaluation lock:" >&2
        echo "  ${file}:${line}" >&2
        echo "  Wrap it: with_eval_lock(|| unsafe { sys::... })" >&2
        echo >&2
    fi
done <<< "${guarded_hits}"

if [ "${fail}" -ne 0 ]; then
    echo "check_eval_lock: FAILED" >&2
    exit 1
fi

echo "OK: every MLX evaluation FFI call is made under the evaluation lock."
