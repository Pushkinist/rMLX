//! The gate's positive control.
//!
//! Run by `scripts/run_gpu_tests.sh` before it trusts a clean scan, never as
//! part of the correctness suite — the runner excludes this name from the
//! population it derives from the ignore-rule classifier.

use super::{dispatch_out_of_bounds_store, shader_validation_enabled};
use rmlx_mlx::Device;

#[test]
#[ignore = "drives the GPU; run serialized via scripts/run_gpu_tests.sh"]
fn shader_validation_canary_emits_an_invalid_access_report() {
    // Refuses to dispatch uninstrumented. With validation on the store is
    // caught and dropped; without it, this would be a genuine out-of-bounds
    // write into an MLX-owned buffer, which is not something to do for a test.
    if !shader_validation_enabled() {
        println!(
            "SKIP shader_validation_canary: MTL_SHADER_VALIDATION is not set, so the \
             deliberate out-of-bounds store would be a real one."
        );
        return;
    }
    // The dispatch itself must SUCCEED. An invalid device store does not fail
    // the command buffer: it is dropped, `cb.error` stays nil, and the process
    // exits 0. The gate's assertion is on the diagnostic text this produces, so
    // an error here means the canary never ran, not that it worked.
    #[allow(
        clippy::expect_used,
        reason = "the canary failing to dispatch must abort loudly: a silent skip would let the gate trust a scan that proved nothing"
    )]
    dispatch_out_of_bounds_store(Device::Gpu).expect("canary kernel must dispatch");
}
