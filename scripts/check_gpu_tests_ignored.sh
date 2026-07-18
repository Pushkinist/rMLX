#!/usr/bin/env bash
# scripts/check_gpu_tests_ignored.sh — CI gate: fail if a GPU-touching test in
# ANY workspace member crate is missing the `#[ignore]` Metal-context attribute.
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
#   Every member crate of the workspace (read from `Cargo.toml [workspace]
#   members`, never hard-coded), across both of its test roots:
#     * unit tests under `src/` — files named `<name>_tests.rs` OR bare
#       `tests.rs` (the file-size convention in CLAUDE.md allows both; a glob
#       of `*_tests.rs` alone silently misses every `tests.rs`).
#     * integration binaries under `tests/*.rs`.
#   `cargo test -p <crate> --lib` does not build `tests/` at all, so a check
#   scoped to `src/` alone would mirror exactly the blind spot it exists to
#   catch.
#
# WHAT COUNTS AS GPU-TOUCHING (shape, not message)
#   A `#[test]` fn that reaches `Device::Gpu`, directly in its own body or
#   transitively through a helper it calls (including generic helpers called
#   with a turbofish, and methods on an inherent impl). `Device::Gpu` bound to
#   a module-scope `const NAME: Device = Device::Gpu;` counts too — a body that
#   uses NAME is GPU-touching. The check keys on that shape alone — never on
#   the ignore *reason* text, which varies across the tree and would make the
#   gate green while the class stayed live.
#
# KNOWN LIMITATION (file-local fixed point)
#   The reachability fixed point runs per file. A `#[test]` that reaches Metal
#   ONLY through a helper defined in another file (e.g. a `tests/common/mod.rs`
#   `run_golden_test`) is not seen. Those helpers' callers are `#[ignore]`d
#   today, so nothing is un-caught now, but a new un-ignored cross-file caller
#   would slip through. Closing this needs a whole-tree (not per-file) pass and
#   is tracked as a separate follow-up.
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
# FILE EXEMPTION (device-as-value false positives)
#   Detection is by shape: naming `Device::Gpu`. In compute crates that always
#   means a Metal dispatch. In higher crates `Device` is a plain config enum —
#   a test may pass `Device::Gpu` to a PURE function that only branches on it
#   (e.g. a CLI truncation-policy resolver) and never touches Metal. No local
#   shape separates `pure_fn(.., Device::Gpu, ..)` from `mlx_op(.., Device::Gpu)`,
#   and the seed must stay broad (the real GPU tests bind `let d = Device::Gpu;`,
#   so any narrower match would miss them). Such a file opts out with a marker
#   on its own line:
#       // gpu-test-gate: exempt — <reason>
#   The file is then skipped entirely. Use it ONLY for files that contain no
#   Metal-driving test; a reviewer audits each marker, and the file must stay
#   free of GPU-dispatching tests (a new one there would be silently uncaught).
#
# PARSING
#   Relies on rustfmt layout (already enforced by `make fmt-check`): a `fn`
#   item's closing brace sits at the same indent as its `fn` keyword. That
#   makes brace/string depth tracking unnecessary.
#
# Exit 0 = clean. Exit 1 = violation (or a scan that found nothing to check).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CARGO_TOML="${REPO_ROOT}/Cargo.toml"

# Fail closed: the workspace manifest is the source of the member list. If it
# is gone the gate is scanning nothing.
if [ ! -f "${CARGO_TOML}" ]; then
    echo "ERROR: ${CARGO_TOML} not found — cannot resolve workspace members." >&2
    exit 1
fi

# Read `[workspace] members = [ ... ]` — the single source of the crate list.
# Never hard-code a crate root: a moved/renamed crate must surface as a
# fail-closed error below, not as a silently narrowed scan.
members=()
while IFS= read -r m; do
    [ -n "$m" ] && members+=("$m")
done < <(awk '
    /^[[:space:]]*members[[:space:]]*=[[:space:]]*\[/ { inm = 1; next }
    inm {
        line = $0
        if (line ~ /\]/) { inm = 0 }
        gsub(/[",]/, "", line)
        gsub(/[[:space:]]/, "", line)
        gsub(/\[/, "", line)
        gsub(/\]/, "", line)
        if (line != "" && line !~ /^#/) { print line }
    }
' "${CARGO_TOML}")

if [ ${#members[@]} -eq 0 ]; then
    echo "ERROR: parsed 0 workspace members from ${CARGO_TOML}." >&2
    echo "A gate that scans nothing passes everything; refusing to report OK." >&2
    exit 1
fi

# Fail closed per member. A member listed in Cargo.toml whose dir or `src/` is
# missing means the crate moved or was renamed — break loudly rather than let
# the gate report OK over a narrowed scan (this workspace has extracted crates
# before; see the dep graph in CLAUDE.md).
scan_files=()
for crate in "${members[@]}"; do
    crate_dir="${REPO_ROOT}/${crate}"
    src_dir="${crate_dir}/src"
    tests_dir="${crate_dir}/tests"

    if [ ! -d "${crate_dir}" ]; then
        echo "ERROR: member crate '${crate}' from Cargo.toml has no directory at ${crate_dir}." >&2
        echo "The crate moved or was renamed; fix the members list or the path." >&2
        exit 1
    fi
    if [ ! -d "${src_dir}" ]; then
        echo "ERROR: ${src_dir} does not exist — this gate is scanning nothing for '${crate}'." >&2
        echo "The crate moved or was renamed; update Cargo.toml members." >&2
        exit 1
    fi

    # Unit tests: match both `<name>_tests.rs` and bare `tests.rs`.
    while IFS= read -r -d '' f; do scan_files+=("$f"); done < <(
        find "${src_dir}" \( -name "*_tests.rs" -o -name "tests.rs" \) \
            -not -path "*/target/*" -print0
    )
    # Integration binaries (optional per crate).
    if [ -d "${tests_dir}" ]; then
        while IFS= read -r -d '' f; do scan_files+=("$f"); done < <(
            find "${tests_dir}" -name "*.rs" -not -path "*/target/*" -print0
        )
    fi
done

if [ ${#scan_files[@]} -eq 0 ]; then
    echo "ERROR: matched 0 test files across ${#members[@]} workspace members." >&2
    echo "A gate that scans nothing passes everything; refusing to report OK." >&2
    exit 1
fi

violations=""
warnings=""

for f in "${scan_files[@]}"; do
    # A file may opt out of the GPU-dispatch check with an explicit marker
    # (see FILE EXEMPTION in the header). Skip it entirely.
    if grep -q "gpu-test-gate: exempt" "$f"; then
        continue
    fi
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

        # ── Module-scope `const NAME: Device = Device::Gpu;` ────────────────
        # A test that uses NAME (rather than the Device::Gpu literal) is still
        # GPU-touching; record the alias so the seed below can see it.
        !in_fn && /^[[:space:]]*const[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:[[:space:]]*Device[[:space:]]*=[[:space:]]*Device::Gpu/ {
            cname = $0
            sub(/^[[:space:]]*const[[:space:]]+/, "", cname)
            sub(/[^A-Za-z0-9_].*$/, "", cname)
            gpu_const[cname] = 1
            attrs = ""
            next
        }

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
            # Seed: fns that name Device::Gpu (literal) or a module-scope
            # `const … = Device::Gpu` alias in their own body.
            for (f_ in body) {
                is_gpu = (body[f_] ~ /Device::Gpu/) ? 1 : 0
                if (!is_gpu) {
                    for (c in gpu_const) {
                        if (body[f_] ~ ("(^|[^A-Za-z0-9_])" c "([^A-Za-z0-9_]|$)")) {
                            is_gpu = 1
                            break
                        }
                    }
                }
                gpu[f_] = is_gpu
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
    echo "WARNING (non-fatal): #[ignore] claims a Metal context but no Device::Gpu is visible in-file:" >&2
    printf '%s' "$warnings" >&2
    echo "  -> If the test truly does not touch the GPU, drop the #[ignore] (and pass" >&2
    echo "     Device::Cpu) — an ignored CPU test is a test that silently stopped running." >&2
    echo "  -> But the seed is file-local: a test reaching Metal ONLY through a helper" >&2
    echo "     defined in another module (e.g. a kernel-builder in the parent) is a false" >&2
    echo "     positive here and the #[ignore] is correct. Verify before removing it." >&2
    echo >&2
fi

if [ -n "$violations" ]; then
    echo "ERROR: GPU-touching tests missing the #[ignore] Metal-context attribute:" >&2
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

echo "OK: every GPU-touching test carries #[ignore] (${#scan_files[@]} files across ${#members[@]} workspace members)."
