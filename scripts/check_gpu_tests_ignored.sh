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
# REACHABILITY IS WHOLE-CRATE, NOT PER-FILE
#   The fn call-graph is built across ALL of a crate's scanned files together
#   (not one file at a time). A `#[test]` that reaches Metal ONLY through a
#   helper defined in a DIFFERENT scanned file — the canonical case being a
#   `tests/common/mod.rs` helper such as `run_golden_test` that binds
#   `let device = Device::Gpu;` and is called as `common::run_golden_test(..)`
#   from each `tests/<arch>_golden_tokens.rs` binary — is now seen. Call
#   resolution is collision-safe (two files may define same-named helpers):
#     * an UNqualified call `helper(..)` binds to a same-file definition only;
#     * a QUALIFIED call `module::helper(..)` binds to a helper defined in the
#       file whose module name is `module` (the directory name for a `mod.rs`,
#       otherwise the file stem).
#   So a CPU-only `helper()` in file A is never tainted by a same-named GPU
#   `helper()` in unrelated file B — the unqualified call binds to A's own.
#
# RESIDUAL (documented honestly — the gate narrows the blind spot, it does not
# claim to erase every path):
#   * A helper defined in a NON-scanned regular source file (e.g.
#     `crates/rmlx-models/src/paroquant_msl.rs`) reached from its sibling
#     `*_tests.rs` via `use super::*` is not traced — only the crate's SCANNED
#     test roots are in the graph. (The present instance, `kernel_rpt1()`,
#     builds a `MetalKernel` and never names `Device::Gpu`, so no gate shape
#     matches it regardless of reachability.)
#   * An UNqualified cross-file call resolved through a glob import
#     (`use module::*; helper()`) is not traced across files — cross-file
#     binding requires the `module::` qualifier. This is the price of
#     collision-safety and it fails toward MISSING a path, never toward a
#     false positive.
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
# ignores are legitimately non-GPU (e.g. "requires mlx runtime"), and a helper
# in a non-scanned source file (see RESIDUAL) is a legitimate reason a
# Metal-driving test looks GPU-free to this gate.
#
# PER-TEST EXEMPTION (device-as-value false positives)
#   Detection is by shape: naming `Device::Gpu`. In compute crates that always
#   means a Metal dispatch. In higher crates `Device` is a plain config enum —
#   a test may pass `Device::Gpu` to a PURE function that only branches on it
#   (e.g. a CLI truncation-policy resolver) and never touches Metal. No local
#   shape separates `pure_fn(.., Device::Gpu, ..)` from `mlx_op(.., Device::Gpu)`,
#   and the seed must stay broad (the real GPU tests bind `let d = Device::Gpu;`,
#   so any narrower match would miss them). Such a test opts out with a
#   line-leading marker inside its OWN attribute/comment block:
#       // gpu-test-gate: exempt
#   The exemption is scoped to that one `#[test]` fn — NOT the whole file — so a
#   Metal-driving test added to the same file still trips the gate. The marker
#   must lead its line and sit in the fn's attribute block; a copy inside a fn
#   body does not exempt. A reviewer audits each marker; use it ONLY for a test
#   that passes `Device::Gpu` as a value and never dispatches Metal.
#
# PARSING
#   Relies on rustfmt layout (already enforced by `make fmt-check`): a `fn`
#   item's closing brace sits at the same indent as its `fn` keyword. That
#   makes brace/string depth tracking unnecessary.
#
# Exit 0 = clean. Exit 1 = violation (or a scan that found nothing to check).

set -euo pipefail

# --list: print the COMPLIANT set (GPU-touching tests that carry #[ignore]) as
# `<crate><TAB><fn>` and exit, instead of enforcing. `scripts/run_gpu_tests.sh`
# consumes this so the rule and the runner cannot cover different populations —
# a GPU test this gate mandates is by construction a GPU test that gets run.
LIST_MODE=0
if [ "${1:-}" = "--list" ]; then
    LIST_MODE=1
    shift
fi

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
#
# NOTE: this parser assumes the one-member-per-line array layout. `make
# fmt-check` (rustfmt) does NOT format `Cargo.toml`, so that layout is a
# convention, not an enforced invariant — a reformat to a single-line array or
# an inline comment could silently narrow the scan. The plausibility floor
# below (parsed members vs. crate dirs on disk) guards against exactly that.
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

# Plausibility floor: every crate under crates/ (a dir with its own Cargo.toml)
# must be a parsed member. If the parse yields fewer, the members array layout
# changed and the scan silently narrowed — the exact scope-loss class this gate
# guards against. Fail closed. (A crate deliberately excluded from the workspace
# would trip this; that is a reviewable event, not a silent narrowing.)
disk_crates=0
if [ -d "${REPO_ROOT}/crates" ]; then
    disk_crates=$(find "${REPO_ROOT}/crates" -mindepth 2 -maxdepth 2 \
        -name Cargo.toml -not -path "*/target/*" | wc -l | tr -d ' ')
fi
if [ "${#members[@]}" -lt "${disk_crates}" ]; then
    echo "ERROR: parsed ${#members[@]} workspace members but ${disk_crates} crate dirs exist under crates/." >&2
    echo "The members array in ${CARGO_TOML} likely changed layout and narrowed the scan." >&2
    echo "Restore the one-member-per-line layout, or reconcile the members list." >&2
    exit 1
fi

# The awk detector, shared across every crate's scan. Reads all of a crate's
# scanned files in one pass (FILENAME distinguishes them) and builds a
# whole-crate fn call-graph, so a cross-file helper is reachable. Emits
# `V  <file>: <fn>` for a violation and `W  <file>: <fn>` for the converse
# warning. Kept in a variable so the per-crate loop invokes one identical
# program over each crate's file list.
read -r -d '' AWK_DETECT <<'AWK' || true
    # New file: reset the per-file parse state and derive this file's module
    # name (directory name for a mod.rs, otherwise the file stem). The module
    # name is how a QUALIFIED cross-file call `module::helper(..)` binds.
    FNR == 1 {
        in_attr = 0; attrs = ""; in_fn = 0; curid = 0; close_marker = ""
        nseg = split(FILENAME, seg, "/")
        base = seg[nseg]
        if (base == "mod.rs" && nseg >= 2) {
            curmod = seg[nseg - 1]
        } else {
            curmod = base
            sub(/\.rs$/, "", curmod)
        }
    }

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
    # ── Per-test exemption marker (line-leading, in the attr block) ─────
    # Folds into the pending attribute block so it attaches to the NEXT fn
    # only. Guarded by !in_fn so a copy inside a body cannot exempt.
    !in_fn && /^[[:space:]]*\/\/[[:space:]]*gpu-test-gate:[[:space:]]*exempt([[:space:]]|$)/ {
        attrs = attrs " GATE_EXEMPT"
        next
    }
    # ── Comments / doc comments keep the pending attribute block ────────
    /^[[:space:]]*\/\// { next }

    # ── Module-scope `const NAME: Device = Device::Gpu;` ────────────────
    # A test that uses NAME (rather than the Device::Gpu literal) is still
    # GPU-touching; record the alias, scoped to THIS file, so a same-named
    # const in another file does not leak across the crate.
    !in_fn && /^[[:space:]]*const[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:[[:space:]]*Device[[:space:]]*=[[:space:]]*Device::Gpu/ {
        cname = $0
        sub(/^[[:space:]]*const[[:space:]]+/, "", cname)
        sub(/[^A-Za-z0-9_].*$/, "", cname)
        gpu_const[FILENAME SUBSEP cname] = 1
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
        G++
        order[++ord_n] = G
        name_of[G] = name
        file_of[G] = FILENAME
        mod_of[G] = curmod
        attrs_of[G] = attrs
        body_of[G] = line
        curid = G
        in_fn = 1
        close_marker = indent "}"
        attrs = ""
        next
    }
    # ── Close of the captured fn: `}` at the fn keyword indent ──────────
    in_fn && $0 == close_marker {
        in_fn = 0
        curid = 0
        attrs = ""
        next
    }
    # Body lines, with line comments stripped: prose that names a GPU
    # helper or Device::Gpu must not make the fn look GPU-touching.
    in_fn {
        line = $0
        sub(/\/\/.*$/, "", line)
        body_of[curid] = body_of[curid] " " line
        next
    }
    # Any other line drops a dangling attribute block.
    /^[[:space:]]*[^[:space:]]/ { attrs = "" }

    END {
        # ── Seed: fns naming Device::Gpu (literal) or a file-scoped
        #    `const … = Device::Gpu` alias in their own body. Seeds feed a
        #    worklist; the GPU set is small, so we pop each GPU fn once and
        #    taint its callers rather than sweep every pair every round.
        qh = 0; qt = 0
        for (g = 1; g <= G; g++) {
            b = body_of[g]
            isg = (b ~ /Device::Gpu/) ? 1 : 0
            if (!isg) {
                f = file_of[g]
                for (key in gpu_const) {
                    si = index(key, SUBSEP)
                    kf = substr(key, 1, si - 1)
                    kc = substr(key, si + 1)
                    if (kf == f && b ~ ("(^|[^A-Za-z0-9_])" kc "([^A-Za-z0-9_]|$)")) {
                        isg = 1
                        break
                    }
                }
            }
            if (isg) { gpu[g] = 1; queue[++qt] = g }
        }

        # ── Fixed point via worklist. Pop a GPU fn h; any not-yet-GPU fn
        #    that CALLS h becomes GPU. Resolution is collision-safe:
        #      * same file  -> unqualified/method call `h(` (any qualifier
        #        char admitted, as a file has no two free fns of one name);
        #      * other file -> QUALIFIED call `mod_of[h] :: h (`.
        #    A cheap index() substring prune avoids compiling the anchored
        #    regex for the vast majority of (caller, callee) pairs.
        while (qh < qt) {
            h = queue[++qh]
            nm = name_of[h]
            hf = file_of[h]
            hq = mod_of[h]
            same_re = "(^|[^a-zA-Z0-9_])" nm "([[:space:]]*::[[:space:]]*<[^;{]*>)?[[:space:]]*\\("
            qual_re = "(^|[^A-Za-z0-9_])" hq "[[:space:]]*::[[:space:]]*" nm "([[:space:]]*::[[:space:]]*<[^;{]*>)?[[:space:]]*\\("
            for (c = 1; c <= G; c++) {
                if (gpu[c]) { continue }
                if (index(body_of[c], nm) == 0) { continue }
                matched = 0
                if (file_of[c] == hf) {
                    if (body_of[c] ~ same_re) { matched = 1 }
                }
                if (!matched && body_of[c] ~ qual_re) { matched = 1 }
                if (matched) {
                    gpu[c] = 1
                    queue[++qt] = c
                }
            }
        }

        # ── Report, in source order across the crate's files ────────────
        for (i = 1; i <= ord_n; i++) {
            g = order[i]
            if (attrs_of[g] !~ /#\[test\]/) { continue }
            has_ignore = (attrs_of[g] ~ /#\[ignore/)
            exempt = (attrs_of[g] ~ /GATE_EXEMPT/)
            if (gpu[g] && !has_ignore && !exempt) {
                printf "V  %s: %s\n", file_of[g], name_of[g]
            }
            # The compliant set: GPU-touching AND ignored. This is exactly the
            # population `scripts/run_gpu_tests.sh` must execute, so it is
            # derived from the same classifier rather than from a second,
            # drifting list. Exempt fns are device-as-value, not Metal.
            if (gpu[g] && has_ignore && !exempt) {
                printf "T  %s: %s\n", file_of[g], name_of[g]
            }
            # Converse: an ignore claiming a Metal context on a test that
            # never reaches one. Warn — the test may have stopped running
            # (or reaches Metal only via a non-scanned source-file helper).
            if (!gpu[g] && has_ignore && attrs_of[g] ~ /#\[ignore[^]]*([Mm]etal|GPU)/) {
                printf "W  %s: %s\n", file_of[g], name_of[g]
            }
        }
    }
AWK

# Fail closed per member. A member listed in Cargo.toml whose dir or `src/` is
# missing means the crate moved or was renamed — break loudly rather than let
# the gate report OK over a narrowed scan (this workspace has extracted crates
# before; see the dep graph in CLAUDE.md).
total_files=0
violations=""
warnings=""
gpu_tests=""

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

    # The cargo package name, for `--list` consumers that feed it to
    # `cargo test -p`. Read from the manifest rather than assumed equal to the
    # directory basename: the two happen to match for every member today, but a
    # mismatch would surface as an opaque "package not found" from cargo instead
    # of failing at this gate, which is the fail-closed discipline the rest of
    # this script follows.
    pkg_name="$(awk -F'"' '/^name[[:space:]]*=/{print $2; exit}' "${crate_dir}/Cargo.toml" 2>/dev/null)"
    if [ -z "${pkg_name}" ]; then
        echo "ERROR: could not read [package] name from ${crate_dir}/Cargo.toml." >&2
        exit 1
    fi

    # Gather this crate's scanned files. Whole-crate reachability needs every
    # scanned file of the crate in ONE awk pass, so collect them per crate.
    crate_files=()
    # Unit tests: match both `<name>_tests.rs` and bare `tests.rs`.
    while IFS= read -r -d '' f; do crate_files+=("$f"); done < <(
        find "${src_dir}" \( -name "*_tests.rs" -o -name "tests.rs" \) \
            -not -path "*/target/*" -print0
    )
    # Integration binaries + their shared modules (optional per crate).
    if [ -d "${tests_dir}" ]; then
        while IFS= read -r -d '' f; do crate_files+=("$f"); done < <(
            find "${tests_dir}" -name "*.rs" -not -path "*/target/*" -print0
        )
    fi

    [ ${#crate_files[@]} -eq 0 ] && continue
    total_files=$((total_files + ${#crate_files[@]}))

    out=$(awk "$AWK_DETECT" "${crate_files[@]}")
    while IFS= read -r line; do
        [ -z "$line" ] && continue
        case "$line" in
            "V  "*) violations="${violations}  ${line#V  }"$'\n' ;;
            "W  "*) warnings="${warnings}  ${line#W  }"$'\n' ;;
            "T  "*) gpu_tests="${gpu_tests}${pkg_name}"$'\t'"${line##*: }"$'\n' ;;
        esac
    done <<< "$out"
done

if [ "${total_files}" -eq 0 ]; then
    echo "ERROR: matched 0 test files across ${#members[@]} workspace members." >&2
    echo "A gate that scans nothing passes everything; refusing to report OK." >&2
    exit 1
fi

if [ "${LIST_MODE}" -eq 1 ]; then
    if [ -z "${gpu_tests}" ]; then
        echo "ERROR: classified 0 GPU-touching #[ignore] tests across ${total_files} files." >&2
        echo "The detector found nothing to run; refusing to emit an empty list." >&2
        exit 1
    fi
    printf '%s' "${gpu_tests}"
    exit 0
fi

if [ -n "$warnings" ]; then
    echo "WARNING (non-fatal): #[ignore] claims a Metal context but no Device::Gpu is reachable in the scanned roots:" >&2
    printf '%s' "$warnings" >&2
    echo "  -> If the test truly does not touch the GPU, drop the #[ignore] (and pass" >&2
    echo "     Device::Cpu) — an ignored CPU test is a test that silently stopped running." >&2
    echo "  -> But reachability covers only the crate's scanned test roots: a test that" >&2
    echo "     reaches Metal ONLY through a helper in a non-scanned source file (e.g. a" >&2
    echo "     kernel-builder in the parent module) is a false positive here and the" >&2
    echo "     #[ignore] is correct. Verify before removing it." >&2
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

echo "OK: every GPU-touching test carries #[ignore] (${total_files} files across ${#members[@]} workspace members)."
