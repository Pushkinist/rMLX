//! Cross-crate drift guard: `rmlx_kv_quant::KvQuant`'s canonical grammar vs
//! the `rmlx-metrics` kv_quant allow-list mirror.
//!
//! `rmlx-metrics` deliberately does not depend on `rmlx-kv-quant` (see
//! CLAUDE.md's workspace dep graph — pulling the MLX-bound codec types into
//! the metrics crate is an Ask-before dep edge), so
//! `rmlx_metrics::identity::canonicalize_kv_quant` hand-mirrors the
//! `<KvQuant as Display>` / `<KvQuant as FromStr>` grammar as plain string
//! matching. `rmlx-models` is the one crate in the workspace that depends on
//! both `rmlx-kv-quant` and `rmlx-metrics`, so this is where the two sides
//! can be checked against each other without introducing that forbidden
//! dep edge anywhere else.
//!
//! If a new `KvQuant` variant is added (or an existing Display form
//! changes) without updating the metrics-side mirror, this test fails —
//! enforcing the single-source-of-truth contract the allow-list is
//! supposed to have, without actually sharing the dependency.

use rmlx_kv_quant::KvQuant;
use rmlx_metrics::identity::canonicalize_kv_quant;

#[test]
fn metrics_kv_quant_allow_list_accepts_every_kv_quant_variant() {
    for q in KvQuant::one_of_each_variant() {
        let display = q.to_string();
        canonicalize_kv_quant(&display).unwrap_or_else(|e| {
            panic!(
                "rmlx-metrics kv_quant allow-list rejected {display:?} (KvQuant variant {q:?}): {e}\n\
                 update `is_valid_kv_quant_token` in crates/rmlx-metrics/src/identity.rs \
                 to mirror the current KvQuant grammar in crates/rmlx-kv-quant/src/quant.rs"
            )
        });
    }
}
