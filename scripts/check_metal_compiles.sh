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
#   metal3.0  the floor. Keeps new syntax from creeping in unnoticed.
#   metal4.0  what production actually compiles at, and the only version where
#             `__HAVE_TENSOR__` is defined.
#
# The second pass is the one that matters for a kernel guarded by
# `#if __HAVE_TENSOR__`: at metal3.0 that guard is inactive, the body compiles
# to an empty translation unit, and the gate would go green having checked
# nothing. A body naming that guard is therefore REJECTED outright when the
# toolchain cannot compile at metal4.0 — refusing to pass by skipping, the same
# rule `--strict` applies to a missing compiler.
#
# Manifest coverage
# -----------------
# Every `.metal` file in a gated directory must be named by that directory's
# manifest, as a body or as a `../`-prefixed header. An unlisted body is checked
# by nothing, which is the exact failure mode this gate exists to prevent, so it
# is a hard failure rather than a silent omission.
#
# The Metal compiler ships with full Xcode, not the Command Line Tools, so it is
# absent on a Command-Line-Tools-only dev box. When it is missing the gate skips
# rather than fails.
#
# `--strict` turns a missing compiler — or a compiler too old for the Metal 4
# pass — into a hard failure. CI passes it: the runner ships a toolchain that can
# do both passes, so degrading to one there would mean the gate silently checked
# less than it claims. Compiling `.metal` needs the toolchain, not a GPU, so this
# gate runs for real in CI even though the runner has no usable Metal device.
# Detection and enforcement stay in one place so the two cannot drift.
#
# Exit 0 = all compiled (or skipped). Exit 1 = a kernel failed to compile, a
# kernel is missing from its manifest, a `__HAVE_TENSOR__` body could not be
# checked, or --strict and the Metal compiler is missing or pre-Metal-4.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Directories holding gated `.metal` kernels. Each needs a
# `probes/kernels.manifest`. Add a directory here when a crate starts shipping
# MSL — nothing else discovers it.
METAL_DIRS=(
    "${REPO_ROOT}/crates/rmlx-kv-quant/src/metal"
    "${REPO_ROOT}/crates/rmlx-models/src/metal"
    "${REPO_ROOT}/crates/rmlx-mlx/src/metal"
)

# The floor, and the version MLX's JIT was observed to use.
BASELINE_STD="metal3.0"
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

# Can this toolchain compile at the Metal 4 language version? An older Xcode
# cannot, which is not by itself a failure — but it does make every
# `__HAVE_TENSOR__` body uncheckable, handled per body below.
printf '#include <metal_stdlib>\n' > "${TMP}/std_probe.metal"
if xcrun -sdk macosx metal "-std=${TENSOR_STD}" -c "${TMP}/std_probe.metal" \
        -o "${TMP}/std_probe.air" >/dev/null 2>&1; then
    TENSOR_STD_OK=1
    STDS=("${BASELINE_STD}" "${TENSOR_STD}")
elif [ "${STRICT}" = 1 ]; then
    echo "ERROR: --strict: this toolchain cannot compile at -std=${TENSOR_STD}," \
         "which is the language version" >&2
    echo "       MLX's JIT uses in production, so half the gate would not run." >&2
    echo "       Refusing to pass by checking less. Select an Xcode with a Metal 4" >&2
    echo "       toolchain (xcode-select -s), then: xcodebuild -downloadComponent MetalToolchain" >&2
    exit 1
else
    TENSOR_STD_OK=0
    STDS=("${BASELINE_STD}")
    echo "NOTE: this toolchain cannot compile at -std=${TENSOR_STD};" \
         "checking ${BASELINE_STD} only."
    echo "      A kernel guarded by ${TENSOR_GUARD} will be reported as" \
         "uncheckable rather than skipped."
fi

failed=()
checked=0

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

        # A body behind the Metal 4 guard cannot be checked at the baseline
        # version: the guard is inactive there and the body vanishes.
        if [ "${TENSOR_STD_OK}" = 0 ] \
                && grep -q -- "${TENSOR_GUARD}" "${METAL_DIR}/${body}"; then
            echo "ERROR: ${rel_dir}/${body} is guarded by ${TENSOR_GUARD}," \
                 "which is only defined at -std=${TENSOR_STD}." >&2
            echo "       This toolchain cannot compile at that version, so the" \
                 "body would be checked as an empty" >&2
            echo "       translation unit. Refusing to pass by skipping;" \
                 "install a Metal 4 toolchain." >&2
            failed+=("${rel_dir}/${body} (${TENSOR_GUARD}, uncheckable)")
            continue
        fi

        probe="${TMP}/probe_${body}"
        {
            echo '#include <metal_stdlib>'
            echo 'using namespace metal;'
            echo
            if [ "${header}" != "-" ]; then
                case "${header}" in
                    ../*) hdr_path="${METAL_DIR}/${header#../}" ;;
                    *)    hdr_path="${PROBE_DIR}/${header}" ;;
                esac
                if [ ! -f "${hdr_path}" ]; then
                    echo "ERROR: manifest references missing header: ${header}" >&2
                    exit 1
                fi
                cat "${hdr_path}"
            fi
            echo
            echo 'kernel void rmlx_msl_compile_probe('
            echo '    device uint*  probe_u [[buffer(0)]],'
            echo '    device float* probe_f [[buffer(1)]],'
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
                if [ "${type}" = "u" ]; then
                    echo "    device uint* ${name} = probe_u; (void)${name};"
                else
                    echo "    device float* ${name} = probe_f; (void)${name};"
                fi
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

echo "OK: ${checked} .metal kernels compile clean at ${STDS[*]}."
