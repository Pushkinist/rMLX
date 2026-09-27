// The latch is process-global and one-way, so every test that sets it runs in
// a child process of this test binary; the parent never sets it.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::*;
use std::process::{Command, Stdio};

const CHILD_MARKER: &str = "started-by-a-gpu-latch-parent-test";
const CHILD_DONE: &str = "gpu-latch-child done";

fn started_by_parent() -> bool {
    std::env::args().any(|arg| arg == CHILD_MARKER)
}

/// Run the ignored child test `name` of this binary and assert it passed and
/// reached its last line, so a filter that matched nothing cannot pass.
fn run_child(name: &str) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            &format!("gpu_latch_tests::{name}"),
            CHILD_MARKER,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{name} failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains(CHILD_DONE),
        "{name} did not run to its end:\n{stdout}"
    );
}

fn assert_forbidden<T: std::fmt::Debug>(got: Result<T>, op: &str) {
    match got {
        Err(Error::GpuForbidden { op: named }) => assert_eq!(named, op),
        other => panic!("{op}: expected GpuForbidden, got {other:?}"),
    }
}

fn cpu_sum() -> Vec<u8> {
    let bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = Array::from_bytes(&bytes, &[2], Dtype::F32).unwrap();
    let sum = add(&a, &a, Device::Cpu).unwrap();
    sum.eval().unwrap();
    sum.to_bytes().unwrap()
}

#[test]
fn gpu_latch_is_unset_until_forbidden() {
    assert!(
        !gpu_forbidden(),
        "no test in this process may set the latch"
    );
}

#[test]
fn metal_api_refused_under_cpu() {
    run_child("metal_api_refused_under_cpu_child");
}

#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn metal_api_refused_under_cpu_child() {
    if !started_by_parent() {
        return;
    }
    forbid_gpu();
    assert!(gpu_forbidden());
    assert_forbidden(metal::is_available(), "metal::is_available");
    assert_forbidden(
        metal::device_info_size("max_recommended_working_set_size"),
        "metal::device_info_size",
    );
    assert_forbidden(metal::set_wired_limit(1 << 30), "metal::set_wired_limit");
    assert_forbidden(
        metal::set_wired_limit_to_recommended(),
        "metal::set_wired_limit_to_recommended",
    );
    assert_forbidden(ensure_gpu_default_stream(), "ensure_gpu_default_stream");
    #[cfg(feature = "metal-capture")]
    assert_forbidden(
        metal_capture::CaptureScope::start(std::path::Path::new("unused.gputrace")),
        "metal_capture::start",
    );
    let expected: Vec<u8> = [2.0f32, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(cpu_sum(), expected, "a CPU op still runs after the latch");
    println!("{CHILD_DONE}");
}

#[test]
fn gpu_stream_refused_under_cpu() {
    run_child("gpu_stream_refused_under_cpu_child");
}

// gpu-test-gate: exempt  the latch refuses the GPU stream before any FFI call.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn gpu_stream_refused_under_cpu_child() {
    if !started_by_parent() {
        return;
    }
    forbid_gpu();
    assert!(gpu_forbidden());
    let bytes: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = Array::from_bytes(&bytes, &[1], Dtype::F32).unwrap();
    assert_forbidden(add(&a, &a, Device::Gpu), "GPU stream");
    println!("{CHILD_DONE}");
}
