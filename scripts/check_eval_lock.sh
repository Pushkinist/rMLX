#!/usr/bin/env bash
# scripts/check_eval_lock.sh — CI gate: every FFI call that can reach
# `mlx::core::eval_impl` must be made under the process-wide evaluation lock,
# and nothing that runs under that lock may take it again.
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
#   too weak to gate on. The *structural* regressions are the ones a text gate
#   catches deterministically, and they are the realistic ones:
#
#     1. Someone drops the lock from a guarded call site.
#     2. Someone adds a call to one of the many other C entry points that
#        evaluate. This is the live risk — only three of the twenty-five are
#        called today, and most of the rest look completely innocuous.
#        `mlx_array_item_float32` reads as a scalar accessor;
#        `mlx_save_safetensors` is the write side of `rmlx convert`;
#        `mlx_array_tostring` is what an `impl Debug for Array` reaches for.
#     3. Someone makes a compiled-closure body take the lock again, which
#        self-deadlocks against the non-reentrant mutex (RULE 3).
#
#   What it does NOT catch is a `with_eval_lock` that stopped locking: the
#   lexical structure is unchanged, so this gate stays green. That half is
#   covered by the unit test
#   `with_eval_lock_serialises_concurrent_callers`. The two are complementary
#   by construction, verified by mutation — neither alone covers this defect.
#
# HOW THE REACH-SET WAS DERIVED — re-run this when the mlx / mlx-c pin moves
#   It is not guesswork and must not become guesswork. TWO passes, because one
#   of them is structurally blind to a quarter of the problem.
#
#   Pass 1 — automated, direct calls (yields 24 symbols):
#     otool -tvV "$(brew --prefix mlx-c)/lib/libmlxc.dylib" > /tmp/mlxc.s
#     otool -tvV "$(brew --prefix mlx)/lib/libmlx.dylib"    > /tmp/mlx.s
#   Take the transitive closure of callers backwards from BOTH
#   `mlx::core::eval_impl` and the hazard symbol itself,
#   `mlx::core::cpu::get_command_encoder(Stream)`, then intersect with the
#   exported `mlx_*` C ABI.
#
#   Pass 2 — by hand, INDIRECT dispatch (adds 1, for 25 total):
#   Pass 1 CANNOT see a call made through a `std::function` — the jump is a
#   `blr` on a vtable slot, so reverse reachability over a disassembly does not
#   traverse it. `mlx_closure_apply` is exactly that shape and does NOT appear
#   in pass 1's output. It reaches evaluation anyway:
#     mlx_closure_apply -> compiled closure body -> compile_fuse ->
#     Compiled::Compiled -> print_constant -> array::item<T>() -> array::eval()
#   `print_constant` bakes scalar constants into the kernel library name; see
#   `mlx/backend/common/compiled.cpp`.
#
#   ** If you re-run pass 1 at a pin bump you will get 24 and `mlx_closure_apply`
#   will be missing. That is the automated pass's blind spot, not a stale
#   entry — do NOT "correct" the gate by deleting the closure guard. **
#
#   Other closure-taking entry points to re-audit the same way at a pin bump,
#   none called today: `mlx_export_function`, `mlx_imported_function_apply`,
#   `mlx_vjp`, `mlx_jvp`, `mlx_value_and_grad`, `mlx_custom_function`,
#   `mlx_checkpoint`. (`mlx_compile` does not invoke the closure;
#   `mlx_fast_metal_kernel_apply` is lazy.)
#
# THE RULES
#   RULE 1  No call to an evaluating entry point that has no guarded wrapper.
#           If one is genuinely needed, wrap it in `with_eval_lock` and move it
#           from RULE 1's list to RULE 2's.
#   RULE 2  Every call to a guarded entry point must be lexically inside a
#           `with_eval_lock` closure: either on the call line itself, or inside
#           an enclosing block whose opening line calls it.
#   RULE 3  No `Closure::from_fn` body may take the evaluation lock. Those
#           bodies run on the calling thread *inside* `mlx_closure_apply`, with
#           the lock already held, so taking it again self-deadlocks on a
#           non-reentrant mutex — a hang, which is one of this defect's own
#           symptoms. The ban is on *taking the lock*, which is broader than
#           "evaluating": `Closure::apply` itself takes it, so a body that
#           applies another compiled closure deadlocks too. That is not a
#           hypothetical shape — it is the first thing a "fuse two fused
#           kernels" refactor produces.
#
# WHAT THIS GATE CANNOT REACH — know the boundary before trusting it
#   * It is a text scan. A call reached through an alias (`use ... as f; f()`),
#     a macro expansion, a function pointer or a transmute is invisible to it.
#     The one alternate *spelling* that existed — `sys::ffi::mlx_*`, valid
#     while `mod ffi` was `pub(crate)` — was closed in the code instead of the
#     regex: `crates/rmlx-mlx/src/sys.rs` now declares `mod ffi` privately, so
#     `sys::mlx_*` is the only way to name these functions.
#   * RULE 3 only sees bodies written INLINE at the `Closure::from_fn` call.
#     `Closure::from_fn(named_fn)` moves the body out of reach entirely, and
#     that is a natural refactor of a one-line delegation like
#     `gated_delta_msl.rs`'s. It is also one level deep: it sees a lock-taking
#     call written in the body, not one reached through a helper the body
#     calls. No op wrapper or `MetalKernel::apply` takes the lock today, which
#     is what makes one level sufficient — an assumption, not a proof.
#   * Line comments and double-quoted strings are stripped before matching, so
#     braces and lock names inside them are inert. Rust RAW strings
#     (`r#"..."#`), char literals (`'{'`) and block comments (`/* */`) are NOT
#     handled; an unbalanced brace in one of those would end RULE 3's body scan
#     early and leave the remainder unscanned.
#   * The reach-set is a snapshot of the current pin, and pass 1 is blind to
#     indirect dispatch (see above).
#   * It proves lexical structure, not runtime behaviour.
#   * It says nothing about concurrent *graph construction*, which never
#     reaches `eval_impl`.
#
# AWK PORTABILITY — measured, not reasoned about
#   This gate and its 26-fixture corpus give identical verdicts under gawk
#   5.4.1, mawk 1.3.4 and BSD awk 20200816, each under both LC_ALL=C and
#   LC_ALL=en_US.UTF-8, with all 17 gate mutations killed in all six
#   combinations. The awk below is POSIX-only on purpose: no `gensub`, no `\s`,
#   no `{n,m}` intervals (mawk's support is version-dependent) and no octal
#   escapes in bracket expressions (BSD awk accepts `[\300-\337]`, Linux awk
#   rejects it outright). `length`/`substr` do go character-based rather than
#   byte-based under a UTF-8 locale, but `strip()` only ever reconstructs by
#   concatenation, so its output is byte-identical either way. The scripts call
#   bare `awk`, so re-checking this means putting a shim named `awk` first on
#   PATH — there is no AWK= override to reach for.

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

# Shared awk helper: drop `//` comments and double-quoted string bodies, so
# neither a laundering comment nor a brace in a literal can steer the scans.
STRIP_FN='
function strip(line,   i, n, c, out, instr, esc) {
    n = length(line); out = ""; instr = 0; esc = 0
    for (i = 1; i <= n; i++) {
        c = substr(line, i, 1)
        if (instr) {
            if (esc) { esc = 0; continue }
            if (c == "\\") { esc = 1; continue }
            if (c == "\"") { instr = 0 }
            continue
        }
        if (c == "\"") { instr = 1; continue }
        if (c == "/" && substr(line, i + 1, 1) == "/") break
        out = out c
    }
    return out
}'

# Drop hits whose match survives only inside a comment or a string literal.
# All three rules go through this, so "is this a real call site?" is answered
# the same way everywhere — an inconsistency here is what let a doc comment
# naming `sys::mlx_eval(v)` fail RULE 1 while the same text was inert to
# RULE 2 and RULE 3.
filter_real_calls() {
    local re="$1" hit file line text stripped
    while IFS= read -r hit; do
        [ -z "${hit}" ] && continue
        file="${hit%%:*}"
        text="${hit#*:}"
        line="${text%%:*}"
        stripped="$(sed -n "${line}p" "${file}" | awk "${STRIP_FN}"'{ print strip($0) }')"
        if printf '%s\n' "${stripped}" | grep -qE "${re}"; then
            printf '%s:%s:%s\n' "${file}" "${line}" "${stripped}"
        fi
    done
}

fail=0

# ---- RULE 1: entry points with no guarded wrapper must not be called -------

banned_hits="$(grep -rnE "${BANNED_RE}" --include='*.rs' "${SCAN_DIR}" \
    | filter_real_calls "${BANNED_RE}" || true)"

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

guarded_hits="$(grep -rnE "${GUARDED_RE}" --include='*.rs' "${SCAN_DIR}" \
    | filter_real_calls "${GUARDED_RE}" || true)"

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
# both a long comment inside the closure and to a guarded sibling call earlier
# in the same function (a sibling is not an enclosing opener, so it never
# launders a later unguarded call). Comments are stripped first, so neither a
# trailing `// with_eval_lock: ...` nor a column-0 comment inside the closure
# can launder or sever the chain.
check_guarded() {
    awk -v target="$1" "${STRIP_FN}"'
        NR <= target {
            line = strip($0)
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
        echo "  Wrap the call itself in with_eval_lock(|| ...). A comment saying" >&2
        echo "  the caller already holds the lock does not satisfy this gate and" >&2
        echo "  is not a substitute for holding it." >&2
        echo >&2
    fi
done <<< "${guarded_hits}"

# ---- RULE 3: nothing running under the lock may take it again -------------

closure_hits="$(grep -rnE 'Closure::from_fn[[:space:]]*\(' --include='*.rs' "${SCAN_DIR}" || true)"

# Brace-match the closure body (over comment- and string-stripped text) and
# look for anything that would take the evaluation lock: a direct evaluation,
# or an application of another compiled closure, which takes it internally.
check_closure_body() {
    awk -v start="$1" "${STRIP_FN}"'
        NR < start { next }
        {
            line = strip($0)
            if (!started) {
                p = index(line, "{")
                if (p == 0) next
                started = 1
                rest = substr(line, p)
            } else {
                rest = line
            }
            if (rest ~ /(\.|Array::)(eval|async_eval|to_bytes)[[:space:]]*\(/) bad = 1
            if (rest ~ /\.apply[[:space:]]*\([[:space:]]*&\[/) bad = 1
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
        echo "check_eval_lock: RULE 3 — a Closure::from_fn body takes the evaluation lock:" >&2
        echo "  ${file}:${line}" >&2
        echo "  That body runs inside mlx_closure_apply with the lock already" >&2
        echo "  held, so evaluating — or applying another compiled closure," >&2
        echo "  which takes the lock internally — self-deadlocks on a" >&2
        echo "  non-reentrant mutex. Move it outside the closure." >&2
        echo >&2
    fi
done <<< "${closure_hits}"

if [ "${fail}" -ne 0 ]; then
    echo "check_eval_lock: FAILED" >&2
    exit 1
fi

echo "OK: every MLX evaluation FFI call is made under the evaluation lock (25-symbol reach-set)."
