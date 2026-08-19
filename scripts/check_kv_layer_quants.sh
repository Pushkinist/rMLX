#!/usr/bin/env bash
# scripts/check_kv_layer_quants.sh — CI gate: the per-layer KV codec vector has
# exactly one producer, and every path that builds a per-layer cache stack
# either uses it or declares that it deliberately does not.
#
# WHY
#   Three places must agree on which codec each decoder layer gets: the arch
#   loops that BUILD the caches, the SSD attach that folds the vector into
#   `layout_key`, and the per-request prompt-cache seed. The last two DESCRIBE
#   what the first builds. A second copy of the
#   `kv_quant_for_layer(i, n_layers, base, TAIL_N, HEAD_N)` loop is therefore
#   not a style duplication — it is how a description stops matching the thing
#   described, silently, the next time the constants or the rule move. The
#   failure mode is not a crash: the layout key stops moving when the policy
#   changes, and a stale on-disk block is handed to a request whose layers were
#   built differently.
#
# RULE 1 (single producer)
#   `kv_quant_for_layer(` may only be called inside
#   crates/rmlx-models/src/kv_cache/ — the function, its producer
#   `kv_layer_quants`, and their unit tests. Everywhere else, including tests,
#   calls `kv_layer_quants(n_layers, base)`: an arch test that mirrors
#   production construction has to mirror it through the same producer or it
#   stops being a mirror.
#
# RULE 2 (declared uniformity)
#   A non-test file under crates/rmlx-models/src that constructs per-layer
#   caches (`KvCache::with_quant_max_seq[_window]`) must either call
#   `kv_layer_quants(` or carry the marker
#
#       // kv-layer-quants: uniform — <reason>
#
#   Rule 1 alone cannot see a hand-rolled `if i < 2 { K8V8 }`; this rule makes
#   a per-layer stack that does not go through the producer a deliberate,
#   written-down decision rather than an accident. Every current marker is a
#   scratch stack that is never spilled and never keyed.
#
# Exit 0 = clean. Exit 1 = violation found.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_DIR="crates/rmlx-models/src/kv_cache/"

# ── Rule 1 ────────────────────────────────────────────────────────────────
rule1=()
while IFS= read -r f; do
    rel="${f#"${REPO_ROOT}/"}"
    case "${rel}" in
        "${POLICY_DIR}"*) continue ;;
    esac
    rule1+=("${rel}")
done < <(
    grep -rl --include='*.rs' 'kv_quant_for_layer(' "${REPO_ROOT}/crates" 2>/dev/null || true
)

if [ ${#rule1[@]} -gt 0 ]; then
    echo "ERROR: kv_quant_for_layer( called outside ${POLICY_DIR}:" >&2
    for f in "${rule1[@]}"; do echo "  $f" >&2; done
    echo >&2
    echo "Use the single producer instead:" >&2
    echo "  let quants = kv_layer_quants(n_layers, kv_quant);   // crate::kv_cache" >&2
    echo "Two copies of the loop let the SSD layout key and the prompt-cache seed" >&2
    echo "describe a per-layer mixture the arch no longer builds." >&2
    exit 1
fi

# ── Rule 2 ────────────────────────────────────────────────────────────────
rule2=()
while IFS= read -r f; do
    rel="${f#"${REPO_ROOT}/"}"
    case "${rel}" in
        "${POLICY_DIR}"*) continue ;;
        *_tests.rs|*/tests.rs|*/tests/*) continue ;;
    esac
    grep -q 'kv_layer_quants(' "$f" && continue
    grep -qE '^[[:space:]]*//[[:space:]]*kv-layer-quants: uniform' "$f" && continue
    rule2+=("${rel}")
done < <(
    grep -rl --include='*.rs' -E 'KvCache::with_quant_max_seq(_window)?\(' \
        "${REPO_ROOT}/crates/rmlx-models/src" 2>/dev/null || true
)

if [ ${#rule2[@]} -gt 0 ]; then
    echo "ERROR: per-layer cache stack built without the single producer, and without" >&2
    echo "declaring that it is uniform on purpose:" >&2
    for f in "${rule2[@]}"; do echo "  $f" >&2; done
    echo >&2
    echo "Either build the stack from the producer:" >&2
    echo "  kv_layer_quants(n_layers, kv_quant).into_iter().enumerate().map(|(i, q)| …)" >&2
    echo "or, if this stack is deliberately uniform (a scratch stack that is never" >&2
    echo "spilled and never keyed), record why with a line-leading marker:" >&2
    echo "  // kv-layer-quants: uniform — <reason>" >&2
    exit 1
fi

echo "OK: kv_quant_for_layer has one producer, and every per-layer cache stack uses it or declares uniformity."
