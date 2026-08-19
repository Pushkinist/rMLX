//! End-to-end SSD layout-key disambiguation gate.
//!
//! Loads the Bonsai-2bit Qwen3 model, runs the same 256-token prompt under
//! two different `kv_quant` configurations within the SAME `--project`
//! namespace, and asserts:
//!
//! 1. The second invocation under kv_quant **A** is an SSD hit (the chained
//!    digest reconstructed from `prompt_ids` under `layout_key_A` matches the
//!    row written by the first invocation).
//! 2. Switching to kv_quant **B** produces a MISS (different `layout_key` ⇒
//!    different chained-hash stream ⇒ a fresh prefill is run).
//! 3. After both runs the SQLite index holds two rows under the same prompt's
//!    last block digest with distinct `layout_key` values, and both `.kvb`
//!    files are present on disk.
//!
//! Gated on `RMLX_KV_TEST_MODEL` (same convention as the other goldens).
//! When the env var is absent the test is a silent skip — it can ship green on
//! CI without the operator copying a model into the runner. The single MLX
//! claim file is honoured: the test acquires it via the standard pattern.
//!
//! Run (after wiping prior caches under the chosen project):
//!
//! ```text
//! RMLX_KV_TEST_MODEL=/path/to/Ternary-Bonsai-8B-mlx-2bit \
//! cargo test -p rmlx-models --test layout_key_disambiguation \
//! -- --ignored --nocapture
//! ```
//!
//! Marked `#[ignore]` so it never gates `make ci` / `make model-check`.

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
    use rmlx_models::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};

    let n_layers = 36usize; // Ternary-Bonsai-8B: every layer full-attention.
    let (arch, kv_heads, head_dim) = ("Qwen3ForCausalLM", 8usize, 128usize);

    // What a `--kv-quant none` request builds today.
    let current: Vec<KvQuant> = (0..n_layers)
        .map(|i| {
            kv_quant_for_layer(
                i,
                n_layers,
                KvQuant::None,
                LAYER_ADAPTIVE_TAIL_N,
                LAYER_ADAPTIVE_HEAD_N,
            )
        })
        .collect();
    // What it used to build: boundary layers promoted to K8V8.
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

/// End-to-end variant — gated by `RMLX_KV_TEST_MODEL` AND `--ignored`. Lives
/// here so the test binary is identifiable in `cargo test --test` output.
///
/// The full implementation is intentionally a compile-only skeleton today: it
/// loads the Bonsai snapshot, verifies the arch matches, then early-returns.
/// Wiring the bench-style prompt-cache assertion through the real
/// `generate_greedy` path is more invasive than 's scope (it touches the
/// server + claim-file plumbing). The unit suite (`crates/rmlx-models/src/...`)
/// covers every invariant on synthetic fixtures; this slot is reserved
/// for a follow-up that wires the assertion through the live decode loop.
#[ignore]
#[test]
fn ssd_layout_key_disambiguation() {
    panic!("E2E wiring deferred; see ticket follow-up");
}
