#!/usr/bin/env bash
# scripts/check_gpu_tests_ignored.sh — CI gate: fail if a GPU-touching test in
# rmlx-kv-quant is missing the `#[ignore]` Metal-context attribute.
#
# WHY
#   A shared Metal context cannot be driven from parallel test threads. A GPU
#   test left un-ignored runs under the default `cargo test` alongside its
#   siblings and aborts the whole binary ("Rust cannot catch foreign
#   exceptions"), taking every other test in that binary down with it. Such
#   tests pass when run in isolation, so a PR's own targeted run looks green.
#   The abort is load-dependent — a couple of parallel GPU tests can pass for a
#   long time before enough of them tip the binary over — so "it passes today"
#   is not evidence of compliance. That is why this is a static gate.
#
# SCOPE
#   Both test roots of the crate: unit tests (`src/**/*_tests.rs`) and the
#   integration binaries (`tests/*.rs`). `cargo test -p rmlx-kv-quant --lib`
#   does not build `tests/` at all, so a check scoped to `src/` alone would
#   mirror exactly the blind spot it exists to catch.
#
# WHAT COUNTS AS GPU-TOUCHING (shape, not message)
#   A `#[test]` fn that reaches `Device::Gpu`, directly in its own body or
#   transitively through a helper it calls (including generic helpers called
#   with a turbofish, and methods on an inherent impl). The check keys on that
#   shape alone — never on the ignore *reason* text, which varies across the
#   crate and would make the gate green while the class stayed live.
#
# THE FIX FOR A VIOLATION — pick the one that is true:
#   * The test really drives the GPU -> add the attribute:
#       #[ignore = "GPU Metal context — run in isolation: \
#                   cargo test <filter> -- --ignored --test-threads=1"]
#   * The test only exercises a guard that returns before any GPU work ->
#     pass `Device::Cpu`, and leave it un-ignored so it keeps running in the
#     default gate.
#
# Also emits a non-fatal WARNING for the converse — a test whose `#[ignore]`
# claims a Metal context but which never reaches `Device::Gpu`. That is a test
# that has silently stopped running. It is a warning, not a failure: some
# ignores are legitimately non-GPU (e.g. "requires mlx runtime").
#
# PARSING
#   Relies on rustfmt layout (already enforced by `make fmt-check`): a `fn`
#   item's closing brace sits at the same indent as its `fn` keyword. That
#   makes brace/string depth tracking unnecessary.
#
# Exit 0 = clean. Exit 1 = violation (or a scan that found nothing to check).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_SRC="${REPO_ROOT}/crates/rmlx-kv-quant/src"
CRATE_TESTS="${REPO_ROOT}/crates/rmlx-kv-quant/tests"

# Fail closed. A moved or renamed crate must break this gate loudly rather than
# let it report OK over an empty scan — this crate has been extracted once
# already (see the workspace dep graph in CLAUDE.md).
for root in "${CRATE_SRC}" "${CRATE_TESTS}"; do
    if [ ! -d "${root}" ]; then
        echo "ERROR: ${root} does not exist — this gate is scanning nothing." >&2
        echo "The crate moved or was renamed; update the roots in check_gpu_tests_ignored.sh." >&2
        exit 1
    fi
done

scan_files=()
while IFS= read -r -d '' f; do scan_files+=("$f"); done < <(
    find "${CRATE_SRC}" -name "*_tests.rs" -not -path "*/target/*" -print0
)
while IFS= read -r -d '' f; do scan_files+=("$f"); done < <(
    find "${CRATE_TESTS}" -name "*.rs" -not -path "*/target/*" -print0
)

if [ ${#scan_files[@]} -eq 0 ]; then
    echo "ERROR: matched 0 test files under ${CRATE_SRC} and ${CRATE_TESTS}." >&2
    echo "A gate that scans nothing passes everything; refusing to report OK." >&2
    exit 1
fi

violations=""
warnings=""

for f in "${scan_files[@]}"; do
    out=$(awk -v fname="$f" '
        # ── Multi-line attribute continuation ────────────────────────────────
        in_attr {
            attrs = attrs " " $0
            if ($0 ~ /^[[:space:]]*\)\]/) { in_attr = 0 }
            next
        }
        # ── Attribute (any indent) ──────────────────────────────────────────
        /^[[:space:]]*#\[/ {
            attrs = attrs " " $0
            if ($0 !~ /\][[:space:]]*$/) { in_attr = 1 }
            next
        }
        # ── Comments / doc comments keep the pending attribute block ────────
        /^[[:space:]]*\/\// { next }

        # ── fn item at any indent (top-level or inside an inherent impl) ────
        !in_fn && /^[[:space:]]*(pub[^ ]* )?(async )?fn [a-zA-Z0-9_]+/ {
            line = $0
            indent = line
            sub(/[^[:space:]].*$/, "", indent)   # leading whitespace only
            name = line
            sub(/^[[:space:]]*(pub[^ ]* )?(async )?fn /, "", name)
            sub(/[^a-zA-Z0-9_].*$/, "", name)
            order[++n] = name
            fn_attrs[name] = attrs
            body[name] = line
            cur = name
            in_fn = 1
            close_marker = indent "}"
            attrs = ""
            next
        }
        # ── Close of the captured fn: `}` at the fn keyword indent ──────────
        in_fn && $0 == close_marker {
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
        # Any other line drops a dangling attribute block.
        /^[[:space:]]*[^[:space:]]/ { attrs = "" }

        END {
            # Seed: fns that name Device::Gpu in their own body.
            for (f_ in body) {
                gpu[f_] = (body[f_] ~ /Device::Gpu/) ? 1 : 0
            }
            # Fixed point: a fn that *calls* a GPU-reaching fn is GPU-reaching.
            # Match a call site — `name(` or `name::<T>(` (turbofish) — not a
            # bare mention, so a name in a doc reference or a string does not
            # propagate. A leading `.` is admitted so method calls count.
            changed = 1
            while (changed) {
                changed = 0
                for (caller in body) {
                    if (gpu[caller]) { continue }
                    for (callee in body) {
                        if (caller == callee || !gpu[callee]) { continue }
                        if (body[caller] ~ ("(^|[^a-zA-Z0-9_])" callee \
                                            "([[:space:]]*::[[:space:]]*<[^;{]*>)?[[:space:]]*\\(")) {
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
                has_ignore = (fn_attrs[f_] ~ /#\[ignore/)
                if (gpu[f_] && !has_ignore) {
                    printf "V  %s: %s\n", fname, f_
                }
                # Converse: an ignore claiming a Metal context on a test that
                # never reaches one. Warn — the test has stopped running.
                if (!gpu[f_] && has_ignore && fn_attrs[f_] ~ /#\[ignore[^]]*([Mm]etal|GPU)/) {
                    printf "W  %s: %s\n", fname, f_
                }
            }
        }
    ' "$f")
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        case "$line" in
            "V  "*) violations="${violations}  ${line#V  }"$'\n' ;;
            "W  "*) warnings="${warnings}  ${line#W  }"$'\n' ;;
        esac
    done <<< "$out"
done

if [ -n "$warnings" ]; then
    echo "WARNING: tests whose #[ignore] claims a Metal context but that never reach Device::Gpu:" >&2
    printf '%s' "$warnings" >&2
    echo "  -> an ignored CPU test is a test that silently stopped running." >&2
    echo "     Drop the #[ignore] (and pass Device::Cpu) if it does not touch the GPU." >&2
    echo >&2
fi

if [ -n "$violations" ]; then
    echo "ERROR: GPU-touching tests in rmlx-kv-quant missing the #[ignore] Metal-context attribute:" >&2
    printf '%s' "$violations" >&2
    echo >&2
    echo "A shared Metal context cannot be driven from parallel test threads: an" >&2
    echo "un-ignored GPU test aborts the whole test binary under a plain" >&2
    echo "\`cargo test\`, killing every other test in it." >&2
    echo >&2
    echo "Add to each test that really drives the GPU:" >&2
    echo "  #[ignore = \"GPU Metal context — run in isolation: cargo test <filter> -- --ignored --test-threads=1\"]" >&2
    echo >&2
    echo "If the test only exercises a guard that returns before any GPU work," >&2
    echo "pass Device::Cpu instead and leave it un-ignored." >&2
    exit 1
fi

echo "OK: every GPU-touching test in rmlx-kv-quant carries #[ignore] (${#scan_files[@]} files scanned)."
