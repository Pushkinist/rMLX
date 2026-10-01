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
    forbid_gpu().unwrap();
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
    forbid_gpu().unwrap();
    assert!(gpu_forbidden());
    let bytes: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = Array::from_bytes(&bytes, &[1], Dtype::F32).unwrap();
    assert_forbidden(add(&a, &a, Device::Gpu), "GPU stream");
    println!("{CHILD_DONE}");
}

#[test]
fn an_op_that_mlx_builds_itself_runs_on_the_cpu_under_cpu() {
    run_child("an_op_that_mlx_builds_itself_runs_on_the_cpu_under_cpu_child");
}

/// An f32 input against bf16 scales: affine `quantized_matmul` builds
/// `astype(scales, f32)` itself, on the default stream of the MLX default
/// device. With the GPU still the default device, that op lands on a GPU
/// stream of this thread, and the MLX thread cannot evaluate it. The oracle
/// dequantizes the same weight to bf16 and multiplies in f32, so it agrees to
/// bf16 rounding.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn an_op_that_mlx_builds_itself_runs_on_the_cpu_under_cpu_child() {
    if !started_by_parent() {
        return;
    }
    forbid_gpu().unwrap();
    let values: Vec<f32> = (0..64 * 64)
        .map(|i| ((i % 17) as f32 - 8.0) / 8.0)
        .collect();
    let w = Array::from_f32_slice(&values, &[64, 64])
        .unwrap()
        .astype(Dtype::Bf16, Device::Cpu)
        .unwrap();
    let (packed, scales, biases) = quantize(&w, 64, 4, Device::Cpu).unwrap();
    assert_eq!(scales.dtype(), Dtype::Bf16, "the scales must stay bf16");
    let x = Array::from_f32_slice(&[0.5; 64], &[1, 64]).unwrap();
    let got = quantized_matmul(
        &x,
        &packed,
        &scales,
        Some(&biases),
        64,
        4,
        "affine",
        true,
        Device::Cpu,
    )
    .unwrap()
    .to_bytes()
    .unwrap_or_else(|e| panic!("an op that MLX built itself did not run on the CPU: {e}"));
    let dense = dequantize(
        &packed,
        &scales,
        Some(&biases),
        64,
        4,
        "affine",
        Device::Cpu,
    )
    .unwrap()
    .astype(Dtype::F32, Device::Cpu)
    .unwrap();
    let dense_t = dense.transpose(&[1, 0], Device::Cpu).unwrap();
    let want = matmul(&x, &dense_t, Device::Cpu)
        .unwrap()
        .to_bytes()
        .unwrap();
    let floats = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let (got, want) = (floats(&got), floats(&want));
    assert_eq!(got.len(), 64);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-2 * (1.0 + w.abs()),
            "column {i}: {g} against the oracle {w}"
        );
    }
    println!("{CHILD_DONE}");
}
