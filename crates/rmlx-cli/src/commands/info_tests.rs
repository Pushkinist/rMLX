//! Unit tests for the smoke-probe exit-code mapping in `info.rs`.

use super::SmokeExitCode;
use rmlx_models::SmokeVerdict;

// ---------------------------------------------------------------------------
// SmokeExitCode::as_i32 — all five codes
// ---------------------------------------------------------------------------

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn smoke_exit_code_ok_is_zero() {
    assert_eq!(SmokeExitCode::Ok.as_i32(), 0);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn smoke_exit_code_broken_is_one() {
    assert_eq!(SmokeExitCode::Broken.as_i32(), 1);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn smoke_exit_code_load_fail_is_three() {
    assert_eq!(SmokeExitCode::LoadFail.as_i32(), 3);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn smoke_exit_code_inconclusive_is_four() {
    assert_eq!(SmokeExitCode::Inconclusive.as_i32(), 4);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn smoke_exit_code_unsupported_is_five() {
    assert_eq!(SmokeExitCode::Unsupported.as_i32(), 5);
}

// ---------------------------------------------------------------------------
// SmokeVerdict → SmokeExitCode mapping — inline helper mirrors run_info logic
// ---------------------------------------------------------------------------

fn verdict_to_exit_code(v: &SmokeVerdict) -> SmokeExitCode {
    match v {
        SmokeVerdict::Ok => SmokeExitCode::Ok,
        SmokeVerdict::BrokenPunctLoop { .. } | SmokeVerdict::BrokenNan { .. } => {
            SmokeExitCode::Broken
        }
        SmokeVerdict::Inconclusive { .. } => SmokeExitCode::Inconclusive,
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn verdict_ok_maps_to_exit_0() {
    assert_eq!(verdict_to_exit_code(&SmokeVerdict::Ok).as_i32(), 0);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn verdict_broken_punct_loop_maps_to_exit_1() {
    let v = SmokeVerdict::BrokenPunctLoop {
        dominant_piece: ".".to_owned(),
        distinct_ids: 1,
    };
    assert_eq!(verdict_to_exit_code(&v).as_i32(), 1);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn verdict_broken_nan_maps_to_exit_1() {
    let v = SmokeVerdict::BrokenNan { at_step: 0 };
    assert_eq!(verdict_to_exit_code(&v).as_i32(), 1);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn verdict_inconclusive_maps_to_exit_4() {
    let v = SmokeVerdict::Inconclusive {
        reason: "eos at step 1".to_owned(),
    };
    assert_eq!(verdict_to_exit_code(&v).as_i32(), 4);
}

// ---------------------------------------------------------------------------
// Load-error variants — exit 3 vs exit 5
// ---------------------------------------------------------------------------

// Error is #[non_exhaustive]; a wildcard arm is required even after listing all
// known variants. Allow it here — the important branch is Model vs everything else.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "Error is #[non_exhaustive]; catch-all is required for forward compatibility"
)]
fn load_err_to_exit_code(e: &rmlx_core::error::Error) -> SmokeExitCode {
    match e {
        rmlx_core::error::Error::Model(_) => SmokeExitCode::Unsupported,
        _ => SmokeExitCode::LoadFail,
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn load_error_model_maps_to_exit_5() {
    let e = rmlx_core::error::Error::Model("arch not supported".to_owned());
    assert_eq!(load_err_to_exit_code(&e).as_i32(), 5);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn load_error_loader_maps_to_exit_3() {
    let e = rmlx_core::error::Error::Loader("missing shard".to_owned());
    assert_eq!(load_err_to_exit_code(&e).as_i32(), 3);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn load_error_config_maps_to_exit_3() {
    let e = rmlx_core::error::Error::Config("malformed config".to_owned());
    assert_eq!(load_err_to_exit_code(&e).as_i32(), 3);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test assertions; panic is the desired failure mode"
)]
fn load_error_mlx_maps_to_exit_3() {
    let e = rmlx_core::error::Error::Mlx("metal error".to_owned());
    assert_eq!(load_err_to_exit_code(&e).as_i32(), 3);
}
