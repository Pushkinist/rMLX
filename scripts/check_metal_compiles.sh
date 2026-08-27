#!/usr/bin/env bash
# scripts/check_metal_compiles.sh — CI gate: every `.metal` kernel compiles with
# the native Metal compiler, so an MSL syntax error surfaces here instead of on
# the first GPU dispatch.
#
# Where the kernels are
# ---------------------
# One directory per crate that ships MSL, each with its own
# `probes/kernels.manifest`:
#
#   crates/rmlx-kv-quant/src/metal   KV-cache codecs
#   crates/rmlx-models/src/metal     per-arch kernels (ParoQuant, GatedDeltaNet)
#   crates/rmlx-mlx/src/metal        the MLX-JIT language-version probe
#
# Directory scope, not crate scope: a `.metal` file is gated by living in one of
# these directories, wherever its Rust dispatcher sits.
#
# Why the files are not compiled directly
# ---------------------------------------
# These hold kernel *bodies*: MLX generates the function signature and buffer
# declarations at dispatch, so a body on its own is a sequence of statements at
# file scope — not a translation unit. Three things must be supplied to compile
# one:
#
#   1. The codec's header (codebook / rotation constants). For codecs whose
#      header is a static file, that file is prepended. For codecs whose header
#      is generated in Rust at dispatch, `probes/*.hdr.metal` holds a captured
#      representative header.
#   2. A function to hold the body, with the body's buffers in scope. The probe
#      declares them as local aliases (not kernel parameters) so it needs no
#      per-kernel signature or buffer-index bookkeeping.
#   3. The values MLX injects at dispatch that are neither buffers nor header
#      constants: template dtypes, template ints, and scalar 0-D inputs. Those
#      come from the manifest's optional fourth field as `#define`s.
#
# `probes/kernels.manifest` supplies all three, per body.
#
# The probe checks *syntax and name resolution* of the real kernel text. It is
# not a numerical or dispatch-shape check — that is what the KV parity tests and
# the real-model smoke cover.
#
# Language versions
# -----------------
# MLX's runtime JIT compiles custom kernel bodies at Metal 4.0 — observed, not
# inferred: `rmlx_nax_probe_gpu` in crates/rmlx-mlx/src/metal_kernel_tests.rs
# reads `__METAL_VERSION__` from inside a JIT'd body and gets 400. So every body
# is compiled twice:
#
#   metal3.1  the floor. Keeps new syntax from creeping in unnoticed. It is
#             3.1 and not 3.0 because `bfloat` — the stored dtype of the KV
#             codecs' scale and norm planes, and so the declared type of every
#             kernel parameter that reads one — is a Metal 3.1 type. A 3.0 floor
#             would have to be held by textually hiding those declarations from
#             the gate, which is the one thing the gate exists to prevent.
#   metal4.0  what production actually compiles at, and the only version where
#             `__HAVE_TENSOR__` is defined.
#
# The second pass is the one that matters for a kernel guarded by
# `#if __HAVE_TENSOR__`: at the baseline that guard is inactive, the body compiles
# to an empty translation unit, and the gate would go green having checked
# nothing. Such a body is therefore never compiled without the guard — it is
# either checked for real or reported as SKIPPED, never quietly passed.
#
# The capability is probed by asserting the guard and the cooperative-tensor
# includes, not by testing that the driver accepts the `-std` flag. A toolchain
# that accepts the flag but leaves the guard undefined would otherwise compile a
# guarded body through its `#else` arm at both passes: the same vacuous pass,
# reached another way.
#
# Manifest coverage
# -----------------
# Every `.metal` file in a gated directory must be named by that directory's
# manifest, as a body or as a `../`-prefixed header. An unlisted body is checked
# by nothing, which is the exact failure mode this gate exists to prevent, so it
# is a hard failure rather than a silent omission.
#
# Toolchain policy — one rule, not two
# -----------------------------------
# "This box's toolchain cannot do X" gets the same answer whatever X is (the
# Metal compiler is absent, or it is too old for the Metal 4 pass):
#
#   --strict      hard failure. CI passes it, and CI must never report green
#                 while checking less than the gate claims to check.
#   otherwise     loud notice, reduced run, exit 0. A contributor on an older
#                 Xcode keeps a working `make ci`; the parts that can be checked
#                 still are, and the parts that cannot are named on stdout.
#
# Splitting that rule — skipping for a missing compiler but failing for an old
# one — breaks the dev loop for everyone whose Xcode predates Metal 4, over a
# diagnostic kernel that ships nothing.
#
# Compiling `.metal` needs the toolchain, not a GPU, so this gate runs for real
# in CI even though the runner has no usable Metal device. Detection and
# enforcement stay in one place so the two cannot drift.
#
# Exit 0 = everything checkable compiled (skips are reported and counted).
# Exit 1 = a kernel failed to compile, a kernel is missing from its manifest, or
# --strict and the toolchain is missing or cannot do the Metal 4 pass.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Directories holding gated `.metal` kernels. Single-sourced with the format
# gate so the two cannot drift apart.
# shellcheck source=scripts/metal_dirs.sh
. "$(dirname "${BASH_SOURCE[0]}")/metal_dirs.sh"

# The floor, and the version MLX's JIT was observed to use.
BASELINE_STD="metal3.1"
TENSOR_STD="metal4.0"
# Defined from metal4.0 onwards; gates the cooperative-tensor surface.
TENSOR_GUARD="__HAVE_TENSOR__"

STRICT=0
for arg in "$@"; do
    case "${arg}" in
        --strict) STRICT=1 ;;
        *) echo "usage: $(basename "$0") [--strict]" >&2; exit 2 ;;
    esac
done

# Probe by *executing* the compiler, not by resolving its path. With Xcode
# selected but the separately-downloaded Metal Toolchain component absent,
# `xcrun -f metal` happily prints a path while every invocation fails with
# "cannot execute tool 'metal' due to missing Metal Toolchain" — which would
# otherwise be reported as a compile failure for every kernel in the manifest.
metal_unavailable_reason=""
if ! command -v xcrun >/dev/null 2>&1; then
    metal_unavailable_reason="xcrun not found"
elif ! metal_probe="$(xcrun -sdk macosx metal --version 2>&1)"; then
    case "${metal_probe}" in
        *"missing Metal Toolchain"*)
            metal_unavailable_reason="Xcode is selected but the Metal Toolchain component is not installed; run: xcodebuild -downloadComponent MetalToolchain"
            ;;
        *)
            metal_unavailable_reason="Metal compiler not usable (needs full Xcode, not just the Command Line Tools): xcode-select -s /Applications/Xcode.app"
            ;;
    esac
fi

if [ -n "${metal_unavailable_reason}" ]; then
    if [ "${STRICT}" = 1 ]; then
        echo "ERROR: --strict: ${metal_unavailable_reason}" >&2
        echo "       Refusing to pass by skipping." >&2
        exit 1
    fi
    echo "SKIP: ${metal_unavailable_reason}"
    echo "      MSL compile gate not run."
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

# Two capabilities, probed separately because they gate different things.
#
#   TENSOR_STD_OK    the driver accepts -std=<TENSOR_STD>. Gates the second pass
#                    over EVERY body.
#   TENSOR_GUARD_OK  at that version <TENSOR_GUARD> is actually defined AND the
#                    cooperative-tensor headers resolve. Gates whether a body
#                    guarded by it can be checked at all.
#
# Probing the flag alone is not enough for the second: a toolchain that accepts
# `-std=metal4.0` but leaves the guard undefined would compile a guarded body
# through its `#else` arm at both passes — green having validated an empty tensor
# path, which is the same vacuous pass the second pass exists to close, reached
# another way. So this probe asserts the guard and the include, not the flag.
printf '#include <metal_stdlib>\n' > "${TMP}/std_probe.metal"
if xcrun -sdk macosx metal "-std=${TENSOR_STD}" -c "${TMP}/std_probe.metal" \
        -o "${TMP}/std_probe.air" >/dev/null 2>&1; then
    TENSOR_STD_OK=1
    STDS=("${BASELINE_STD}" "${TENSOR_STD}")
else
    TENSOR_STD_OK=0
    STDS=("${BASELINE_STD}")
fi

TENSOR_GUARD_OK=0
if [ "${TENSOR_STD_OK}" = 1 ]; then
    {
        echo '#include <metal_stdlib>'
        echo "#if !${TENSOR_GUARD}"
        echo "#error \"${TENSOR_GUARD} is not defined at this language version\""
        echo '#endif'
        echo '#include <metal_tensor>'
        echo '#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>'
        echo 'using namespace mpp::tensor_ops;'
    } > "${TMP}/guard_probe.metal"
    if xcrun -sdk macosx metal "-std=${TENSOR_STD}" -c "${TMP}/guard_probe.metal" \
            -o "${TMP}/guard_probe.air" >/dev/null 2>&1; then
        TENSOR_GUARD_OK=1
    fi
fi

# A toolchain that cannot do the second pass is the same class of problem as a
# missing compiler, and gets the same policy: hard failure under --strict (CI,
# where checking less while reporting green is the whole defect), a loud notice
# and a reduced run otherwise. A contributor on an older Xcode still gets a
# working `make ci`.
if [ "${TENSOR_STD_OK}" = 0 ] || [ "${TENSOR_GUARD_OK}" = 0 ]; then
    if [ "${TENSOR_STD_OK}" = 0 ]; then
        reduced_reason="cannot compile at -std=${TENSOR_STD}"
    else
        reduced_reason="compiles at -std=${TENSOR_STD} but does not define ${TENSOR_GUARD} (or cannot resolve the cooperative-tensor headers)"
    fi
    if [ "${STRICT}" = 1 ]; then
        echo "ERROR: --strict: this toolchain ${reduced_reason}." >&2
        echo "       ${TENSOR_STD} is the language version MLX's JIT uses in production," \
             "so part of the gate" >&2
        echo "       would not run. Refusing to pass by checking less. Select an Xcode with" >&2
        echo "       a Metal 4 toolchain (xcode-select -s), then:" \
             "xcodebuild -downloadComponent MetalToolchain" >&2
        exit 1
    fi
    echo "NOTE: this toolchain ${reduced_reason}."
    echo "      Checking ${STDS[*]}; bodies guarded by ${TENSOR_GUARD} are SKIPPED," \
         "not silently passed."
    echo "      Install a Metal 4 toolchain to run the whole gate. CI runs it with" \
         "--strict, which refuses this."
fi

failed=()
checked=0
skipped=0

for METAL_DIR in "${METAL_DIRS[@]}"; do
    PROBE_DIR="${METAL_DIR}/probes"
    MANIFEST="${PROBE_DIR}/kernels.manifest"
    rel_dir="${METAL_DIR#"${REPO_ROOT}/"}"

    if [ ! -d "${METAL_DIR}" ]; then
        echo "ERROR: missing kernel directory: ${rel_dir}" >&2
        failed+=("${rel_dir} (missing directory)")
        continue
    fi
    if [ ! -f "${MANIFEST}" ]; then
        echo "ERROR: missing ${MANIFEST}" >&2
        failed+=("${rel_dir} (missing manifest)")
        continue
    fi

    # Names the manifest refers to, for the coverage check below.
    referenced="${TMP}/referenced_$(echo "${rel_dir}" | tr '/' '_').txt"
    : > "${referenced}"

    while IFS= read -r line; do
        # Skip comments / blanks.
        case "${line}" in ''|'#'*) continue ;; esac

        body="$(echo "${line}" | cut -d'|' -f1 | xargs)"
        header="$(echo "${line}" | cut -d'|' -f2 | xargs)"
        buffers="$(echo "${line}" | cut -d'|' -f3 | xargs)"
        defines="$(echo "${line}" | cut -d'|' -f4 | xargs)"

        echo "${body}" >> "${referenced}"
        case "${header}" in ../*) echo "${header#../}" >> "${referenced}" ;; esac

        if [ ! -f "${METAL_DIR}/${body}" ]; then
            echo "ERROR: manifest references missing body: ${rel_dir}/${body}" >&2
            failed+=("${rel_dir}/${body} (missing)")
            continue
        fi

        # Resolve the header path early: the guard can live in either half of a
        # body/header pair, and a pair whose guard is entirely in the header
        # would otherwise escape the check below.
        hdr_path=""
        if [ "${header}" != "-" ]; then
            case "${header}" in
                ../*) hdr_path="${METAL_DIR}/${header#../}" ;;
                *)    hdr_path="${PROBE_DIR}/${header}" ;;
            esac
            if [ ! -f "${hdr_path}" ]; then
                echo "ERROR: manifest references missing header: ${rel_dir}/probes -> ${header}" >&2
                failed+=("${rel_dir}/${body} (missing header ${header})")
                continue
            fi
        fi

        # A body behind the Metal 4 guard cannot be checked without it: the
        # guard is inactive, the guarded text vanishes, and compiling the
        # remainder would validate nothing. `--strict` already refused above, so
        # reaching here means a dev box — skip the body loudly rather than
        # pretend it passed.
        if [ "${TENSOR_GUARD_OK}" = 0 ] \
                && grep -q -F -- "${TENSOR_GUARD}" "${METAL_DIR}/${body}" \
                       ${hdr_path:+"${hdr_path}"}; then
            echo "NOTE: SKIP ${rel_dir}/${body} — guarded by ${TENSOR_GUARD}," \
                 "which this toolchain does not provide."
            echo "      Compiling it here would check an empty tensor path." \
                 "CI (--strict) checks it for real."
            skipped=$((skipped + 1))
            continue
        fi

        probe="${TMP}/probe_${body}"
        {
            echo '#include <metal_stdlib>'
            echo 'using namespace metal;'
            echo
            if [ -n "${hdr_path}" ]; then
                cat "${hdr_path}"
            fi
            echo
            echo 'kernel void rmlx_msl_compile_probe('
            echo '    device uint*   probe_u [[buffer(0)]],'
            echo '    device float*  probe_f [[buffer(1)]],'
            echo '    device int*    probe_i [[buffer(2)]],'
            echo '    device bfloat* probe_b [[buffer(3)]],'
            echo '    uint3 thread_position_in_grid          [[thread_position_in_grid]],'
            echo '    uint3 threadgroup_position_in_grid     [[threadgroup_position_in_grid]],'
            echo '    uint3 thread_position_in_threadgroup   [[thread_position_in_threadgroup]],'
            echo '    uint  thread_index_in_threadgroup      [[thread_index_in_threadgroup]],'
            echo '    uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],'
            echo '    uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],'
            echo '    uint  simdgroups_per_threadgroup       [[simdgroups_per_threadgroup]]) {'
            # Buffer aliases: the names this body expects, at their dispatch dtype.
            IFS=',' read -ra bufs <<< "${buffers}"
            for b in "${bufs[@]}"; do
                name="${b%%:*}"
                type="${b##*:}"
                case "${type}" in
                    u) echo "    device uint* ${name} = probe_u; (void)${name};" ;;
                    i) echo "    device int* ${name} = probe_i; (void)${name};" ;;
                    f) echo "    device float* ${name} = probe_f; (void)${name};" ;;
                    b) echo "    device bfloat* ${name} = probe_b; (void)${name};" ;;
                    *)
                        echo "ERROR: ${rel_dir}/${body}: unknown buffer type '${type}' for '${name}' (want u, i, f or b)" >&2
                        exit 1
                        ;;
                esac
            done
            # Dispatch-time values MLX injects that are neither buffers nor
            # header constants. Emitted here, immediately ahead of the body, so
            # a common name (`T`) cannot collide with the header or the probe's
            # own signature.
            if [ -n "${defines}" ] && [ "${defines}" != "-" ]; then
                IFS=',' read -ra defs <<< "${defines}"
                for d in "${defs[@]}"; do
                    echo "#define ${d%%=*} ${d#*=}"
                done
            fi
            echo
            cat "${METAL_DIR}/${body}"
            echo
            echo '}'
        } > "${probe}" || { failed+=("${rel_dir}/${body} (probe assembly)"); continue; }

        for std in "${STDS[@]}"; do
            if ! err="$(xcrun -sdk macosx metal "-std=${std}" -c "${probe}" \
                    -o "${TMP}/out.air" 2>&1)"; then
                echo "FAIL: ${rel_dir}/${body} at -std=${std}" >&2
                echo "${err}" | head -20 >&2
                echo >&2
                failed+=("${rel_dir}/${body} (-std=${std})")
            fi
        done
        checked=$((checked + 1))
    done < "${MANIFEST}"

    # Coverage: a `.metal` file no manifest line names is checked by nothing.
    shopt -s nullglob
    for f in "${METAL_DIR}"/*.metal; do
        base="$(basename "${f}")"
        if ! grep -qxF "${base}" "${referenced}"; then
            echo "ERROR: ${rel_dir}/${base} is not named by ${rel_dir}/probes/kernels.manifest," \
                 "so nothing compiles it." >&2
            echo "       Add it as a body line, or as a '../${base}' header on the line that uses it." >&2
            failed+=("${rel_dir}/${base} (not in manifest)")
        fi
    done
    shopt -u nullglob
done

if [ ${#failed[@]} -gt 0 ]; then
    echo "ERROR: ${#failed[@]} .metal kernel check(s) failed:" >&2
    for f in "${failed[@]}"; do echo "  ${f}" >&2; done
    exit 1
fi

if [ "${skipped}" -gt 0 ]; then
    echo "OK: ${checked} .metal kernels compile clean at ${STDS[*]};" \
         "${skipped} skipped (see the ${TENSOR_GUARD} notes above)."
else
    echo "OK: ${checked} .metal kernels compile clean at ${STDS[*]}."
fi
