//! The installed boundary reaches the single per-layer producer.
//!
//! `install_kv_boundary` is first-call-wins and process-global, so this is a
//! test binary of its own: one install, one observation. The unit tests in
//! `kv_cache/tests.rs` cover the boundary→vector mapping across many values
//! through the private explicit-boundary entry point; what they cannot cover,
//! and this file exists for, is that the *installed* value is the one
//! `kv_layer_quants` reads.

use rmlx_kv_quant::KvQuant;
use rmlx_models::kv_cache::{
    install_kv_boundary, kv_layer_quants, KvBoundary, KvBoundaryInstallError,
};

const BASE: KvQuant = KvQuant::Mixed {
    k_bits: 4,
    v_bits: 4,
    k_group_size: 64,
    v_group_size: 64,
};

#[test]
#[allow(
    clippy::expect_used,
    reason = "test asserts the install contract itself: an unexpected Err must fail loudly with the reason attached"
)]
fn installed_boundary_is_the_one_the_producer_uses() {
    install_kv_boundary(Some(KvBoundary {
        head_n: 1,
        tail_n: 2,
    }))
    .expect("the first install, before any read, is accepted");

    assert_eq!(
        promoted_indices(&kv_layer_quants(12, BASE, false)),
        vec![0, 10, 11],
        "the producer reads the installed counts, not the constants"
    );

    // Re-installing the same value is a no-op, not a conflict: two startup
    // paths agreeing about the configuration is not an error.
    install_kv_boundary(Some(KvBoundary {
        head_n: 1,
        tail_n: 2,
    }))
    .expect("an install that agrees with the one in force changes nothing");

    // A later, different install is refused. Accepting it would re-point a
    // policy that has already built caches, an SSD layout key and a
    // prompt-cache seed.
    let err = install_kv_boundary(Some(KvBoundary {
        head_n: 5,
        tail_n: 5,
    }))
    .expect_err("a second, different install must be refused");
    assert!(
        matches!(err, KvBoundaryInstallError::AlreadyInstalled { .. }),
        "expected AlreadyInstalled, got {err:?}"
    );
    assert_eq!(
        promoted_indices(&kv_layer_quants(12, BASE, false)),
        vec![0, 10, 11],
        "the refused install changed nothing"
    );
}

fn promoted_indices(vector: &[KvQuant]) -> Vec<usize> {
    vector
        .iter()
        .enumerate()
        .filter_map(|(i, q)| (*q != BASE).then_some(i))
        .collect()
}
