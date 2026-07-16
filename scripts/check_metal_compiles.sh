#!/usr/bin/env bash
# scripts/check_metal_compiles.sh — CI gate: every KV `.metal` kernel compiles
# with the native Metal compiler, so an MSL syntax error surfaces here instead
# of on the first GPU dispatch.
#
# Why the files are not compiled directly
# ---------------------------------------
# `crates/rmlx-kv-quant/src/metal/*.metal` holds kernel *bodies*: MLX generates
# the function signature and buffer declarations at dispatch, so a body on its
# own is a sequence of statements at file scope — not a translation unit. Two
# things must be supplied to compile one:
#
#   1. The codec's header (codebook / rotation constants). For codecs whose
#      header is a static file, that file is prepended. For codecs whose header
#      is generated in Rust at dispatch, `probes/*.hdr.metal` holds a captured
#      representative header.
#   2. A function to hold the body, with the body's buffers in scope. The probe
#      declares them as local aliases (not kernel parameters) so it needs no
#      per-kernel signature or buffer-index bookkeeping.
#
# `probes/kernels.manifest` supplies both, per body.
#
# The probe checks *syntax and name resolution* of the real kernel text. It is
# not a numerical or dispatch-shape check — that is what the KV parity tests and
# the real-model smoke cover.
#
# The Metal compiler ships with full Xcode, not the Command Line Tools, so it is
# absent on a Command-Line-Tools-only dev box. When it is missing the gate skips
# rather than fails.
#
# `--strict` turns a missing compiler into a hard failure. CI passes it: the
# GitHub macOS runner ships full Xcode, so a skip there would mean the gate
# silently protected nothing. Compiling `.metal` needs the toolchain, not a GPU,
# so this gate runs for real in CI even though the runner has no usable Metal
# device. Detection and enforcement stay in one place so the two cannot drift.
#
# Exit 0 = all compiled (or skipped). Exit 1 = a kernel failed to compile, or
# --strict and the Metal compiler is missing.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METAL_DIR="${REPO_ROOT}/crates/rmlx-kv-quant/src/metal"
PROBE_DIR="${METAL_DIR}/probes"
MANIFEST="${PROBE_DIR}/kernels.manifest"

STRICT=0
for arg in "$@"; do
    case "${arg}" in
        --strict) STRICT=1 ;;
        *) echo "usage: $(basename "$0") [--strict]" >&2; exit 2 ;;
    esac
done

if ! command -v xcrun >/dev/null 2>&1 || ! xcrun -sdk macosx -f metal >/dev/null 2>&1; then
    if [ "${STRICT}" = 1 ]; then
        echo "ERROR: --strict: Metal compiler not found (xcrun -sdk macosx -f metal)." >&2
        echo "       Refusing to pass by skipping. This needs full Xcode, not just the" >&2
        echo "       Command Line Tools: 'xcode-select -s /Applications/Xcode.app'." >&2
        exit 1
    fi
    echo "SKIP: Metal compiler not found (needs full Xcode, not just the Command Line Tools);"
    echo "      MSL compile gate not run. Install Xcode + 'xcode-select -s /Applications/Xcode.app'."
    exit 0
fi

if [ ! -f "${MANIFEST}" ]; then
    echo "ERROR: missing ${MANIFEST}" >&2
    exit 1
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

failed=()
checked=0

while IFS= read -r line; do
    # Skip comments / blanks.
    case "${line}" in ''|'#'*) continue ;; esac

    body="$(echo "${line}" | cut -d'|' -f1 | xargs)"
    header="$(echo "${line}" | cut -d'|' -f2 | xargs)"
    buffers="$(echo "${line}" | cut -d'|' -f3 | xargs)"

    if [ ! -f "${METAL_DIR}/${body}" ]; then
        echo "ERROR: manifest references missing body: ${body}" >&2
        failed+=("${body} (missing)")
        continue
    fi

    probe="${TMP}/probe_${body}"
    {
        echo '#include <metal_stdlib>'
        echo 'using namespace metal;'
        echo
        # Dequantize kernels are templated on the output dtype by MLX; pin it.
        echo '#define OutT float'
        echo
        if [ "${header}" != "-" ]; then
            hdr_path="${PROBE_DIR}/${header}"
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
        echo '    uint3 thread_position_in_grid        [[thread_position_in_grid]],'
        echo '    uint3 threadgroup_position_in_grid   [[threadgroup_position_in_grid]],'
        echo '    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]) {'
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
        echo
        cat "${METAL_DIR}/${body}"
        echo
        echo '}'
    } > "${probe}" || { failed+=("${body} (probe assembly)"); continue; }

    if ! err="$(xcrun -sdk macosx metal -std=metal3.0 -c "${probe}" -o "${TMP}/out.air" 2>&1)"; then
        echo "FAIL: ${body}" >&2
        echo "${err}" | head -20 >&2
        echo >&2
        failed+=("${body}")
    fi
    checked=$((checked + 1))
done < "${MANIFEST}"

if [ ${#failed[@]} -gt 0 ]; then
    echo "ERROR: ${#failed[@]} KV .metal kernel(s) failed to compile:" >&2
    for f in "${failed[@]}"; do echo "  ${f}" >&2; done
    exit 1
fi

echo "OK: ${checked} KV .metal kernels compile clean."
