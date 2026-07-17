#!/usr/bin/env bash
# scripts/check_gpu_tests_ignored.sh — CI gate: fail if a GPU-touching test in
# rmlx-kv-quant is missing the `#[ignore]` Metal-context attribute.
#
# WHY
#   A shared Metal context cannot be driven from parallel test threads. A GPU
#   test left un-ignored runs under the default `cargo test` alongside its
#   siblings and aborts the whole binary ("Rust cannot catch foreign
#   exceptions"), taking every other test in the crate down with it. Such tests
#   pass when run in isolation, so a PR's own targeted run looks green — the
#   drift only surfaces on a full `cargo test --workspace`.
#
# WHAT COUNTS AS GPU-TOUCHING (shape, not message)
#   A `#[test]` fn that reaches `Device::Gpu`, directly in its own body or
#   transitively through a file-local helper it calls. The check keys on that
#   shape alone — it never matches on the ignore *reason* text, which varies
#   across the crate and would make the gate green while the class stayed live.
#
# THE FIX FOR A VIOLATION — pick the one that is true:
#   * The test really drives the GPU -> add the attribute:
#       #[ignore = "GPU Metal context — run in isolation: \
#                   cargo test <filter> -- --include-ignored --test-threads=1"]
#   * The test only exercises a CPU-side guard that returns before any
#     device-parameterized op -> pass `Device::Cpu`, and leave it un-ignored so
#     it keeps running in the default gate.
#
# PARSING
#   Relies on rustfmt layout (already enforced by `make fmt-check`): top-level
#   `fn` items start at column 0 and close with `}` at column 0. That makes
#   brace/string depth tracking unnecessary.
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_SRC="${REPO_ROOT}/crates/rmlx-kv-quant/src"

violations=""

while IFS= read -r -d '' f; do
    out=$(awk -v fname="$f" '
        # ── Multi-line attribute continuation ────────────────────────────────
        in_attr {
            attrs = attrs " " $0
            if ($0 ~ /^\)\]/) { in_attr = 0 }
            next
        }
        # ── Attribute at column 0 ───────────────────────────────────────────
        /^#\[/ {
            attrs = attrs " " $0
            if ($0 !~ /\]$/) { in_attr = 1 }
            next
        }
        # ── Comments / doc comments keep the pending attribute block ────────
        /^\/\// { next }
        # ── Top-level fn item ───────────────────────────────────────────────
        /^(pub[^ ]* )?(async )?fn [a-zA-Z0-9_]+/ {
            name = $0
            sub(/^(pub[^ ]* )?(async )?fn /, "", name)
            sub(/[^a-zA-Z0-9_].*$/, "", name)
            order[++n] = name
            fn_attrs[name] = attrs
            body[name] = $0
            cur = name
            in_fn = 1
            attrs = ""
            next
        }
        # ── Close of a top-level item ───────────────────────────────────────
        /^\}/ {
            in_fn = 0
            cur = ""
            attrs = ""
            next
        }
        # Body lines, with line comments stripped: prose that names a GPU
        # helper or Device::Gpu must not make the fn look GPU-touching.
        in_fn {
            line = $0
            sub(/\/\/.*$/, "", line)
            body[cur] = body[cur] " " line
            next
        }
        # Any other column-0 line drops a dangling attribute block.
        /^[^[:space:]]/ { attrs = "" }

        END {
            # Seed: fns that name Device::Gpu in their own body.
            for (f_ in body) {
                gpu[f_] = (body[f_] ~ /Device::Gpu/) ? 1 : 0
            }
            # Fixed point: a fn that *calls* a GPU-reaching fn is GPU-reaching.
            # Match a call site (`name(`), not a bare mention, so a name in a
            # doc reference or a string does not propagate.
            changed = 1
            while (changed) {
                changed = 0
                for (caller in body) {
                    if (gpu[caller]) { continue }
                    for (callee in body) {
                        if (caller == callee || !gpu[callee]) { continue }
                        if (body[caller] ~ ("(^|[^a-zA-Z0-9_])" callee "[[:space:]]*\\(")) {
                            gpu[caller] = 1
                            changed = 1
                            break
                        }
                    }
                }
            }
            for (i = 1; i <= n; i++) {
                f_ = order[i]
                if (fn_attrs[f_] !~ /#\[test\]/) { continue }
                if (!gpu[f_]) { continue }
                if (fn_attrs[f_] ~ /#\[ignore/) { continue }
                printf "  %s: %s\n", fname, f_
            }
        }
    ' "$f")
    if [ -n "$out" ]; then
        violations="${violations}${out}"$'\n'
    fi
done < <(find "${CRATE_SRC}" -name "*_tests.rs" -not -path "*/target/*" -print0)

if [ -n "$violations" ]; then
    echo "ERROR: GPU-touching tests in rmlx-kv-quant missing the #[ignore] Metal-context attribute:" >&2
    printf '%s' "$violations" >&2
    echo >&2
    echo "A shared Metal context cannot be driven from parallel test threads: an" >&2
    echo "un-ignored GPU test aborts the whole test binary under a plain" >&2
    echo "\`cargo test\`, killing every other test in the crate." >&2
    echo >&2
    echo "Add to each test that really drives the GPU:" >&2
    echo "  #[ignore = \"GPU Metal context — run in isolation: cargo test <filter> -- --include-ignored --test-threads=1\"]" >&2
    echo >&2
    echo "If the test only exercises a CPU-side guard that returns before any" >&2
    echo "device-parameterized op, pass Device::Cpu instead and leave it un-ignored." >&2
    exit 1
fi

echo "OK: every GPU-touching test in rmlx-kv-quant carries #[ignore]."
