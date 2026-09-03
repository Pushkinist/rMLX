//! An install that arrives after the boundary has been read is refused.
//!
//! `active_kv_boundary` answers with the default when nothing is installed, so
//! a read before the install is indistinguishable from a default run. Move an
//! eager preload above the install and it builds its caches at the default
//! while every key written afterwards describes the requested boundary — no
//! error, no warning, and a stored block handed to a request whose layers were
//! built differently.
//!
//! Its own test binary because the latch is process-global and one-way: a
//! process can demonstrate the read-then-install order or the install-then-read
//! order, never both.

use rmlx_kv_quant::KvQuant;
use rmlx_models::kv_cache::{
    active_kv_boundary, install_kv_boundary, kv_layer_quants, KvBoundary, KvBoundaryInstallError,
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
    clippy::panic,
    reason = "test asserts the install contract itself: an unexpected Ok, or the wrong refusal, must fail loudly with the value attached"
)]
fn an_install_after_the_first_read_is_refused() {
    // The read a stray preload would make. It is answered with the default,
    // and that is exactly the problem: nothing downstream can tell this apart
    // from a run that asked for the default.
    let read = active_kv_boundary();
    assert_eq!(read, KvBoundary::default());
    let built = kv_layer_quants(12, BASE, false);

    let err = install_kv_boundary(Some(KvBoundary {
        head_n: 0,
        tail_n: 0,
    }))
    .expect_err("an install after a read must be refused, not warned about");
    let KvBoundaryInstallError::AlreadyRead { read, requested } = err else {
        panic!("expected AlreadyRead, got {err:?}")
    };
    assert_eq!(read, KvBoundary::default());
    assert_eq!(
        requested,
        KvBoundary {
            head_n: 0,
            tail_n: 0
        }
    );

    assert_eq!(
        kv_layer_quants(12, BASE, false),
        built,
        "the refused install did not change the vector the earlier read described"
    );
}
