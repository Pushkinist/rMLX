#!/usr/bin/env bash
# scripts/check_eval_lock.sh — CI gate: every FFI call that can reach
# `mlx::core::eval_impl` must be made under the process-wide evaluation lock,
# and nothing that runs under that lock may evaluate.
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
#   probabilistic — roughly one failure in twelve runs without the lock. Far
#   too weak to gate on: eleven times in twelve, deleting the lock would go
#   green. The *structural* regressions are the ones a text gate catches
#   deterministically, and they are the realistic ones:
#
#     1. Someone drops the lock from a guarded call site.
#     2. Someone adds a call to one of the many other C entry points that
#        evaluate. This is the live risk — only three of the twenty-four are
#        called today, and most of the rest look completely innocuous.
#        `mlx_array_item_float32` reads as a scalar accessor;
#        `mlx_save_safetensors` is the write side of `rmlx convert`;
#        `mlx_array_tostring` is what an `impl Debug for Array` reaches for.
#     3. Someone makes a compiled-closure body evaluate, which self-deadlocks
#        against the non-reentrant mutex (RULE 3).
#
# HOW THE REACH-SET WAS DERIVED — re-run this when the mlx / mlx-c pin moves
#   It is not guesswork and must not become guesswork. Reverse reachability
#   over the linked dylibs, computed twice and cross-checked:
#
#     otool -tvV "$(brew --prefix mlx-c)/lib/libmlxc.dylib" > /tmp/mlxc.s
#     otool -tvV "$(brew --prefix mlx)/lib/libmlx.dylib"    > /tmp/mlx.s
#
#   Take the transitive closure of callers backwards from BOTH
#   `mlx::core::eval_impl` and the hazard symbol itself,
#   `mlx::core::cpu::get_command_encoder(Stream)`, then intersect with the
#   exported `mlx_*` C ABI. Both directions gave the same 24 below.
#
#   Two of the paths are non-obvious and worth restating, because a reader who
#   only greps for "eval" will miss them:
#     * `mlx_array_item_*` -> `array::item<T>()` -> `array::eval()`
#       (visible in `mlx/array.h` — `item()` evaluates before reading).
#     * `mlx_closure_apply` -> compiled closure body -> `compile_fuse` ->
#       `Compiled::Compiled` -> `print_constant` -> `array::item<T>()` ->
#       `array::eval()`. `print_constant` bakes scalar constants into the
#       kernel library name; see `mlx/backend/common/compiled.cpp`.
#
# THE RULES
#   RULE 1  No call to an evaluating entry point that has no guarded wrapper.
#           If one is genuinely needed, wrap it in `with_eval_lock` and move it
#           from RULE 1's list to RULE 2's.
#   RULE 2  Every call to a guarded entry point must be lexically inside a
#           `with_eval_lock` closure: either on the call line itself, or inside
#           an enclosing block whose opening line calls it.
#   RULE 3  No `Closure::from_fn` body may evaluate. Those bodies run on the
#           calling thread *inside* `mlx_closure_apply`, with the lock already
#           held, so an evaluation there self-deadlocks on a non-reentrant
#           mutex — a hang, which is one of this defect's own symptoms.
#
# WHAT THIS GATE CANNOT REACH — know the boundary before trusting it
#   * It is a text scan. A call reached through an alias (`use ... as f; f()`),
#     a macro expansion, a function pointer or a transmute is invisible to it.
#     The one alternate *spelling* that existed — `sys::ffi::mlx_*`, valid
#     while `mod ffi` was `pub(crate)` — was closed in the code instead of the
#     regex: `crates/rmlx-mlx/src/sys.rs` now declares `mod ffi` privately, so
#     `sys::mlx_*` is the only way to name these functions.
#   * RULE 3 is one level deep. It sees `.eval()` written directly in a closure
#     body, not an evaluation reached through a helper the body calls. No op
#     wrapper or `MetalKernel::apply` evaluates today, which is what makes one
#     level sufficient; that is an assumption, not a proof.
#   * The reach-set is a snapshot of the current pin. A newly-added evaluating
#     entry point will not be flagged until someone re-runs the derivation
#     above and extends RULE 1.
#   * It proves lexical structure, not runtime behaviour. A `with_eval_lock`
#     that took no lock would satisfy it. The mutual-exclusion unit test
#     (`with_eval_lock_serialises_concurrent_callers`) covers that.
#   * It says nothing about concurrent *graph construction*, which never
#     reaches `eval_impl`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Optional scan root, so the fixture suite can point the gate at a synthetic
# tree. Defaults to the real one.
SCAN_DIR="${1:-${REPO_ROOT}/crates}"

if [ ! -d "${SCAN_DIR}" ]; then
    echo "check_eval_lock: no scan dir at ${SCAN_DIR}" >&2
    exit 1
fi

# Entry points that evaluate. Split by whether this crate owns a guarded
# wrapper for them. Anchored on `sys::` + open paren, so prose naming these
# functions in comments and docs does not trip the gate.
#
# `(ffi::)?` is belt-and-braces. The real fix for that spelling is in the code —
# `sys.rs` declares `mod ffi` privately, so `sys::ffi::mlx_*` does not resolve —
# but matching it too costs one alternation and keeps the gate honest if that
# module's visibility is ever widened again.
BANNED_RE='sys::(ffi::)?(mlx_eval|mlx_array_item_[A-Za-z0-9_]*|mlx_array_tostring|mlx_save[A-Za-z0-9_]*|mlx_load_gguf)[[:space:]]*\('
GUARDED_RE='sys::(ffi::)?(mlx_array_eval|mlx_async_eval|mlx_closure_apply)[[:space:]]*\('

fail=0

# ---- RULE 1: entry points with no guarded wrapper must not be called -------

banned_hits="$(grep -rnE "${BANNED_RE}" --include='*.rs' "${SCAN_DIR}" || true)"

if [ -n "${banned_hits}" ]; then
    fail=1
    echo "check_eval_lock: RULE 1 — call to an MLX entry point that evaluates but has no guarded wrapper." >&2
    echo "${banned_hits}" | sed 's/^/  /' >&2
    echo >&2
    echo "  These reach mlx::core::eval_impl and therefore the unsynchronised" >&2
    echo "  process-global CPU command-encoder map. Every mlx-c call belongs in" >&2
    echo "  the rmlx-mlx crate; wrap it in that crate's with_eval_lock and move" >&2
    echo "  the symbol into this gate's guarded set." >&2
    echo >&2
fi

# ---- RULE 2: guarded entry points must be called under the lock ------------

guarded_hits="$(grep -rnE "${GUARDED_RE}" --include='*.rs' "${SCAN_DIR}" || true)"

if [ -z "${guarded_hits}" ]; then
    fail=1
    echo "check_eval_lock: RULE 2 — no call to any guarded MLX entry point found at all." >&2
    echo "  The gate keys on those call sites; if they were renamed or removed," >&2
    echo "  this gate is no longer checking anything and must be updated." >&2
    echo >&2
fi

# For a hit at line L, pass if `with_eval_lock` appears on L itself or on the
# opening line of any block enclosing L. Walking *openers* — successively lower
# indentation — rather than a fixed line window is what makes this immune to
# both a long comment inside the closure (the window bug this replaced) and to
# a guarded sibling call earlier in the same function (a sibling is not an
# enclosing opener, so it never launders a later unguarded call).
check_guarded() {
    awk -v target="$1" '
        NR <= target {
            line = $0
            match(line, /^[ \t]*/)
            ind[NR] = RLENGTH
            txt[NR] = line
        }
        NR == target {
            if (txt[target] ~ /with_eval_lock/) { print "ok"; exit }
            cur = ind[target]
            for (i = target - 1; i >= 1; i--) {
                if (txt[i] ~ /^[ \t]*$/) continue
                if (ind[i] < cur) {
                    if (txt[i] ~ /with_eval_lock/) { print "ok"; exit }
                    cur = ind[i]
                    if (cur == 0) break
                }
            }
            print "bad"
            exit
        }
    ' "$2"
}

while IFS= read -r hit; do
    [ -z "${hit}" ] && continue
    file="${hit%%:*}"
    rest="${hit#*:}"
    line="${rest%%:*}"

    if [ "$(check_guarded "${line}" "${file}")" != "ok" ]; then
        fail=1
        echo "check_eval_lock: RULE 2 — evaluation FFI call not under the evaluation lock:" >&2
        echo "  ${file}:${line}" >&2
        echo "  Wrap it: with_eval_lock(|| unsafe { sys::... })" >&2
        echo >&2
    fi
done <<< "${guarded_hits}"

# ---- RULE 3: nothing that runs under the lock may evaluate ----------------

closure_hits="$(grep -rnE 'Closure::from_fn[[:space:]]*\(' --include='*.rs' "${SCAN_DIR}" || true)"

# Brace-match the closure body and look for a direct evaluation inside it.
check_closure_body() {
    awk -v start="$1" '
        NR < start { next }
        {
            line = $0
            if (!started) {
                p = index(line, "{")
                if (p == 0) next
                started = 1
                rest = substr(line, p)
            } else {
                rest = line
            }
            if (rest ~ /\.(eval|async_eval|to_bytes)[[:space:]]*\(/) bad = 1
            n = length(rest)
            for (i = 1; i <= n; i++) {
                c = substr(rest, i, 1)
                if (c == "{") depth++
                else if (c == "}") {
                    depth--
                    # `exit` still runs END in awk, so mark it handled or the
                    # verdict gets printed twice and every caller reads "bad".
                    if (depth == 0) { done = 1; print (bad ? "bad" : "ok"); exit }
                }
            }
        }
        END { if (!done) print (bad ? "bad" : "ok") }
    ' "$2"
}

while IFS= read -r hit; do
    [ -z "${hit}" ] && continue
    file="${hit%%:*}"
    rest="${hit#*:}"
    line="${rest%%:*}"

    # Doc-comment examples are prose, not code.
    src_line="$(sed -n "${line}p" "${file}")"
    case "${src_line}" in
        *"//!"*|*"///"*) continue ;;
    esac

    if [ "$(check_closure_body "${line}" "${file}")" != "ok" ]; then
        fail=1
        echo "check_eval_lock: RULE 3 — a Closure::from_fn body evaluates:" >&2
        echo "  ${file}:${line}" >&2
        echo "  That body runs inside mlx_closure_apply with the evaluation lock" >&2
        echo "  already held, so evaluating there self-deadlocks on a" >&2
        echo "  non-reentrant mutex. Move the evaluation outside the closure." >&2
        echo >&2
    fi
done <<< "${closure_hits}"

if [ "${fail}" -ne 0 ]; then
    echo "check_eval_lock: FAILED" >&2
    exit 1
fi

echo "OK: every MLX evaluation FFI call is made under the evaluation lock (24-symbol reach-set)."
