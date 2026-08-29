//! SSD layout-key disambiguation tests.
//!
//! The `layout_key` is what stops two different KV layouts from sharing cached
//! blocks in one namespace. Two properties are pinned here against a real
//! on-disk SQLite index:
//!
//! 1. Two layout keys coexist under the same prompt digest — the composite
//!    `(hash, layout_key)` PK does not let one overwrite the other.
//! 2. A block spilled under the previous per-layer codec policy is a MISS for a
//!    request built under the current one.
//!
//! Both run in `make ci`: they need no model, only a temp dir. The end-to-end
//! variant that drove a live `generate_greedy` was a `panic!("deferred")` stub
//! that no gate ever executed, and was deleted rather than left reading as
//! coverage.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

use rmlx_kv_ssd::ssd_index::SsdKvIndex;

/// Smoke test: even without a real model we can sanity-check the layout-key
/// salt indirectly — two synthetic `(hash, layout_key)` rows must coexist in
/// the same namespace's index DB without overwriting one another. Runs in
/// CI alongside the rest of the workspace (NOT `#[ignore]`) so the
/// composite-PK guarantee is exercised end-to-end against a real on-disk
/// SQLite file.
#[test]
fn two_layout_keys_share_namespace_without_collision() {
    let tmp = tempfile::TempDir::new().unwrap();
    let db = tmp.path().join("index.db");
    let idx = SsdKvIndex::open_at(&db).unwrap();

    let hash = "0123456789abcdef".to_string();
    let layout_a: u64 = 0xa55a_5aa5_a55a_5aa5;
    let layout_b: u64 = 0x5aa5_a55a_5aa5_a55a;

    idx.record(
        &hash,
        layout_a,
        std::path::Path::new("/tmp/layout_a.kvb"),
        "Arch/snap",
        "k8v4",
        1024,
    )
    .unwrap();
    idx.record(
        &hash,
        layout_b,
        std::path::Path::new("/tmp/layout_b.kvb"),
        "Arch/snap",
        "k8v8",
        2048,
    )
    .unwrap();

    let row_a = idx.lookup(&hash, layout_a).unwrap().expect("layout A row");
    let row_b = idx.lookup(&hash, layout_b).unwrap().expect("layout B row");

    assert_eq!(row_a.kv_quant, "k8v4");
    assert_eq!(row_b.kv_quant, "k8v8");
    assert_ne!(row_a.layout_key, row_b.layout_key);
    assert_ne!(row_a.byte_size, row_b.byte_size);
}

/// A block spilled under the **previous** boundary-promotion policy must not
/// hydrate into a request running the current one.
///
/// `--kv-quant none` used to be built as a bf16/K8V8 mixture: the first 2 and
/// last 8 layers were promoted to a packed q8_0 store. It is uniform bf16 now.
/// Both are "none" as far as the requested codec goes, so the base codec alone
/// cannot tell the two layouts apart — the layout key has to fold the per-layer
/// vector, which is what this test pins against a real on-disk index.
///
/// Vacuity guard: the same lookup under the request's own key must hit, so a
/// lookup that missed for any other reason (wrong hash, empty table) fails the
/// test rather than passing it.
#[test]
fn a_block_written_under_the_old_mixture_does_not_hydrate_into_a_none_request() {
    use rmlx_kv_quant::KvQuant;
    use rmlx_kv_ssd::compute_layout_key;
    use rmlx_models::kv_cache::{kv_layer_quants, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};

    let n_layers = 36usize; // Ternary-Bonsai-8B: every layer full-attention.
    let (arch, kv_heads, head_dim) = ("Qwen3ForCausalLM", 8usize, 128usize);

    // What a `--kv-quant none` request builds today — from the single producer
    // the arch loops build their caches from, not a second copy of it.
    let current: Vec<KvQuant> = kv_layer_quants(n_layers, KvQuant::None, false);
    // What it used to build: boundary layers promoted to K8V8. Spelled out on
    // purpose — this arm is a frozen historical layout, so deriving it from the
    // live policy would make the comparison below vacuous by construction.
    let legacy: Vec<KvQuant> = (0..n_layers)
        .map(|i| {
            if i < LAYER_ADAPTIVE_HEAD_N || i >= n_layers - LAYER_ADAPTIVE_TAIL_N {
                KvQuant::K8V8
            } else {
                KvQuant::None
            }
        })
        .collect();
    assert_ne!(
        current, legacy,
        "fixture is vacuous: the current policy still produces the legacy mixture"
    );

    let key_current = compute_layout_key(arch, &current, kv_heads, head_dim, KvQuant::None);
    let key_legacy = compute_layout_key(arch, &legacy, kv_heads, head_dim, KvQuant::None);

    let tmp = tempfile::TempDir::new().unwrap();
    let idx = SsdKvIndex::open_at(&tmp.path().join("index.db")).unwrap();
    let hash = "fedcba9876543210".to_string();
    idx.record(
        &hash,
        key_legacy,
        &tmp.path().join("legacy.kvb"),
        "Qwen3ForCausalLM/bonsai-8b",
        "none",
        4096,
    )
    .unwrap();

    assert!(
        idx.lookup(&hash, key_current).unwrap().is_none(),
        "a block spilled under the old bf16/K8V8 mixture must be a MISS for a \
         request whose layers are uniform bf16"
    );
    idx.record(
        &hash,
        key_current,
        &tmp.path().join("current.kvb"),
        "Qwen3ForCausalLM/bonsai-8b",
        "none",
        4096,
    )
    .unwrap();
    assert!(
        idx.lookup(&hash, key_current).unwrap().is_some(),
        "the request must still hit its OWN block — otherwise the miss above proves nothing"
    );
}
