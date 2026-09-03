#!/usr/bin/env bash
# check_kv_boundary_default_parity.sh — CI gate: every surface that names the
# default KV boundary-layer counts names the pair the engine actually applies.
#
# WHY
#   `rmlx_core::kv_boundary::DEFAULT_BOUNDARY_{HEAD,TAIL}_N` is the one
#   definition — it lives in the root crate so `rmlx-models` (which applies the
#   counts) and `rmlx-metrics` (which must recognise a `decode_config` spelling
#   them out) cannot drift. Two surfaces still restate the pair as literal text
#   a compiler never reads: the `--kv-boundary-layers` long help, and the flag's
#   rows in docs/CLI.md. A stale one there is not cosmetic — it tells an
#   operator that omitting the flag gives them a configuration it does not, and
#   `decode_config` NULL is defined as "the engine at its defaults", so the
#   whole cell-identity contract is downstream of this number.
#
#   The Python ingest side is not checked here because it does not restate the
#   value at all: `scripts/lib/kv_boundary_default.py` reads the constants and
#   raises if it cannot find them.
#
# ORACLE
#   The Rust constants. This gate does not check they are RIGHT, only that
#   there is effectively one of them — the same contract as
#   `check-kv-byte-model-parity`.
#
# Exit 0 = clean. Exit 1 = a surface disagrees. Exit 2 = the oracle is unreadable.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${REPO_ROOT}/crates/rmlx-core/src/kv_boundary.rs"
CLI_MAIN="${REPO_ROOT}/crates/rmlx-cli/src/main.rs"
CLI_DOC="${REPO_ROOT}/docs/CLI.md"

const_of() { # $1 = HEAD | TAIL
	sed -n "s/^pub const DEFAULT_BOUNDARY_$1_N: usize = \([0-9]*\);.*/\1/p" "${SOURCE}" | head -1
}

HEAD_N="$(const_of HEAD)"
TAIL_N="$(const_of TAIL)"
if [[ -z "${HEAD_N}" || -z "${TAIL_N}" ]]; then
	echo "ERROR: cannot read DEFAULT_BOUNDARY_{HEAD,TAIL}_N from ${SOURCE#"${REPO_ROOT}/"}." >&2
	echo "That file is the oracle for this gate; a rename has to update it here too." >&2
	exit 2
fi
WANT="${HEAD_N},${TAIL_N}"

fail=0

# ── The CLI long help ───────────────────────────────────────────────────────
# One occurrence, spelled ``Default `H,T`.``
help_pairs="$(sed -n 's/.*Default `\([0-9]*,[0-9]*\)`.*/\1/p' "${CLI_MAIN}" || true)"
if [[ -z "${help_pairs}" ]]; then
	echo "ERROR: crates/rmlx-cli/src/main.rs names no \`Default \`H,T\`\` pair." >&2
	echo "The --kv-boundary-layers long help states the default; this gate reads it." >&2
	fail=1
else
	while IFS= read -r pair; do
		[[ -z "${pair}" ]] && continue
		if [[ "${pair}" != "${WANT}" ]]; then
			echo "ERROR: crates/rmlx-cli/src/main.rs says the default is '${pair}'," >&2
			echo "       the engine applies '${WANT}'." >&2
			fail=1
		fi
	done <<<"${help_pairs}"
fi

# ── docs/CLI.md flag rows ───────────────────────────────────────────────────
doc_rows="$(grep -c -- '--kv-boundary-layers' "${CLI_DOC}" || true)"
if [[ "${doc_rows}" -eq 0 ]]; then
	echo "ERROR: docs/CLI.md documents no --kv-boundary-layers flag." >&2
	fail=1
fi
doc_pairs="$(sed -n 's/.*`--kv-boundary-layers` | `HEAD,TAIL` | `\([0-9]*,[0-9]*\)`.*/\1/p' "${CLI_DOC}" || true)"
if [[ -z "${doc_pairs}" ]]; then
	echo "ERROR: docs/CLI.md has --kv-boundary-layers rows that state no default." >&2
	fail=1
else
	while IFS= read -r pair; do
		[[ -z "${pair}" ]] && continue
		if [[ "${pair}" != "${WANT}" ]]; then
			echo "ERROR: docs/CLI.md says the default is '${pair}', the engine applies '${WANT}'." >&2
			fail=1
		fi
	done <<<"${doc_pairs}"
fi

# ── The shared Python resolver actually resolves ────────────────────────────
py_pair="$(cd "${REPO_ROOT}" && python3 -c "
import sys
sys.path.insert(0, 'scripts/lib')
from kv_boundary_default import kv_boundary_default
print('%d,%d' % kv_boundary_default())
" 2>&1)" || {
	echo "ERROR: scripts/lib/kv_boundary_default.py cannot resolve the default:" >&2
	echo "${py_pair}" >&2
	exit 1
}
if [[ "${py_pair}" != "${WANT}" ]]; then
	echo "ERROR: scripts/lib/kv_boundary_default.py resolves '${py_pair}', engine '${WANT}'." >&2
	fail=1
fi

if [[ "${fail}" -ne 0 ]]; then
	echo >&2
	echo "The engine's constants are the oracle. Update the surface, not the constant," >&2
	echo "unless the default itself is what moved." >&2
	exit 1
fi

echo "OK: the default KV boundary '${WANT}' is named identically by the engine, the CLI help, docs/CLI.md and the ingest resolver ($((doc_rows)) doc mentions)."
