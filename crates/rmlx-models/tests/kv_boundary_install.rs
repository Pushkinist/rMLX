//! The installed boundary reaches the single per-layer producer.
//!
//! `install_kv_boundary` is first-call-wins and process-global, so this is a
//! test binary of its own: one install, one observation. The unit tests in
//! `kv_cache/tests.rs` cover the boundary→vector mapping across many values
//! through the private explicit-boundary entry point; what they cannot cover,
//! and this file exists for, is that the *installed* value is the one
//! `kv_layer_quants` reads.

use rmlx_kv_quant::KvQuant;
use rmlx_models::kv_cache::{install_kv_boundary, kv_layer_quants, KvBoundary};

const BASE: KvQuant = KvQuant::Mixed {
    k_bits: 4,
    v_bits: 4,
    k_group_size: 64,
    v_group_size: 64,
};

#[test]
fn installed_boundary_is_the_one_the_producer_uses() {
    let default_promoted = promoted_indices(&kv_layer_quants(12, BASE, false));
    assert_eq!(
        default_promoted,
        vec![0, 1, 4, 5, 6, 7, 8, 9, 10, 11],
        "before any install the producer runs the shipped 2 head + 8 tail counts"
    );

    install_kv_boundary(Some(KvBoundary {
        head_n: 1,
        tail_n: 2,
    }));

    assert_eq!(
        promoted_indices(&kv_layer_quants(12, BASE, false)),
        vec![0, 10, 11],
        "the producer reads the installed counts, not the constants"
    );

    // First call wins: a later, different install is dropped rather than
    // re-pointing a stack that is already built.
    install_kv_boundary(Some(KvBoundary {
        head_n: 5,
        tail_n: 5,
    }));
    assert_eq!(
        promoted_indices(&kv_layer_quants(12, BASE, false)),
        vec![0, 10, 11],
    );
}

fn promoted_indices(vector: &[KvQuant]) -> Vec<usize> {
    vector
        .iter()
        .enumerate()
        .filter_map(|(i, q)| (*q != BASE).then_some(i))
        .collect()
}
