//! Tests for the MLX thread: an op built on any thread evaluates on any
//! thread, and only `mlx_thread.rs` gets a stream from mlx-c.

use std::sync::mpsc;
use std::time::Duration;

use super::*;
use crate::{add, Array, Device};

/// The longest the cross-thread read may take. A cross-thread evaluation can
/// wait forever instead of failing, and it holds the evaluation lock while it
/// waits, so every later test of this binary would wait too.
const READ_LIMIT: Duration = Duration::from_secs(60);

#[allow(
    clippy::exit,
    reason = "a hung evaluation cannot be stopped, and it holds the lock every later test needs"
)]
fn within_read_limit<T>(done: &mpsc::Receiver<T>, test: &str) -> T {
    match done.recv_timeout(READ_LIMIT) {
        Ok(value) => value,
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{test}: the reader panicked"),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            eprintln!(
                "{test}: the cross-thread read did not finish within {READ_LIMIT:?}. It holds \
                 the evaluation lock, so this test binary ends here."
            );
            std::process::exit(101);
        }
    }
}

fn lazy_sum(device: Device) -> Array {
    let a = Array::from_f32_slice(&[1.0, 2.0], &[2]).unwrap();
    add(&a, &a, device).unwrap()
}

/// Build a lazy op on one thread and keep that thread alive and idle, as a
/// blocking-pool worker is between requests. Evaluate the op on a second
/// thread. The oracle builds and evaluates on one thread.
fn assert_crosses_threads(test: &str, device: Device) {
    let expected: Vec<u8> = [2.0f32, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let oracle = std::thread::spawn(move || lazy_sum(device).to_bytes().unwrap())
        .join()
        .unwrap();
    assert_eq!(oracle, expected, "{test}: the one-thread oracle");

    let (built_tx, built_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let builder = std::thread::spawn(move || {
        built_tx.send(lazy_sum(device)).unwrap();
        release_rx.recv().ok();
    });
    let lazy = built_rx.recv().unwrap();
    assert!(
        !lazy.is_available().unwrap(),
        "{test}: the op must still be lazy when it crosses"
    );
    let (read_tx, read_rx) = mpsc::channel();
    std::thread::spawn(move || {
        read_tx
            .send(lazy.to_bytes().map_err(|e| e.to_string()))
            .ok();
    });
    let crossed = within_read_limit(&read_rx, test);
    release_tx.send(()).ok();
    builder.join().unwrap();
    let crossed = crossed.unwrap_or_else(|e| panic!("{test}: an op built on another thread: {e}"));
    assert_eq!(
        crossed, expected,
        "{test}: differs from the one-thread oracle"
    );
}

#[test]
fn a_cpu_op_built_on_one_thread_evaluates_on_another() {
    assert_crosses_threads(
        "a_cpu_op_built_on_one_thread_evaluates_on_another",
        Device::Cpu,
    );
}

#[test]
#[ignore = "requires Metal GPU; run with `make gpu-test`"]
fn a_gpu_op_built_on_one_thread_evaluates_on_another() {
    assert_crosses_threads(
        "a_gpu_op_built_on_one_thread_evaluates_on_another",
        Device::Gpu,
    );
}

#[test]
fn the_error_message_of_a_call_reaches_the_calling_thread() {
    LAST_ERROR.with(|slot| slot.set(None));
    let value = run(|| {
        LAST_ERROR.with(|slot| slot.set(Some("from the MLX thread".to_owned())));
        7
    })
    .unwrap();
    assert_eq!(value, 7);
    assert_eq!(
        LAST_ERROR.with(Cell::take).as_deref(),
        Some("from the MLX thread"),
        "check_status after a call on the MLX thread must read the message it left"
    );
}

#[test]
fn a_panic_on_the_mlx_thread_reaches_the_caller_and_the_thread_goes_on() {
    let caught = panic::catch_unwind(|| run(|| -> i32 { panic!("inside a job") }));
    assert!(caught.is_err(), "the panic must reach the caller");
    assert_eq!(
        run(|| 5).unwrap(),
        5,
        "the MLX thread must take the next job"
    );
}

/// MLX builds some ops inside other ops on the default stream of the default
/// device, not on the stream the caller passed. After one CPU op, the default
/// CPU and GPU streams of a thread must be the MLX thread's streams.
#[test]
fn a_thread_that_built_an_op_defaults_to_the_mlx_threads_streams() {
    let (cpu, gpu) = std::thread::spawn(|| {
        drop(lazy_sum(Device::Cpu));
        // SAFETY: new references to this thread's default streams, freed
        // below; the MLX thread's handles live for the process.
        unsafe {
            let default = sys::mlx_default_cpu_stream_new();
            let cpu = sys::mlx_stream_equal(default, CPU_STREAM.get().unwrap().0);
            sys::mlx_stream_free(default);
            let gpu = GPU_STREAM.get().map(|mlx| {
                let default = sys::mlx_default_gpu_stream_new();
                let same = sys::mlx_stream_equal(default, mlx.0);
                sys::mlx_stream_free(default);
                same
            });
            (cpu, gpu)
        }
    })
    .join()
    .unwrap();
    assert!(
        cpu,
        "the default CPU stream of a building thread is its own"
    );
    assert_eq!(
        gpu,
        metal_available().then_some(true),
        "the default GPU stream of a building thread is its own, or the MLX thread has none"
    );
}

/// Every `sys::` call whose name holds `stream` in a non-test source file of
/// this crate. Only `mlx_thread.rs` may make one: a stream from anywhere else
/// is a stream of the calling thread, and an op built on it evaluates only
/// there.
#[test]
fn every_stream_comes_from_the_mlx_thread() {
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    let mut dirs = vec![src.clone()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    let is_test = |p: &std::path::Path| {
        let name = p.file_name().unwrap().to_string_lossy();
        name == "tests.rs" || name.ends_with("_tests.rs")
    };
    let mut scanned = 0;
    let mut outside = Vec::new();
    let mut inside = 0;
    for file in files.iter().filter(|f| !is_test(f)) {
        scanned += 1;
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or_default();
            let calls = code
                .match_indices("sys::mlx_")
                .filter(|(at, _)| {
                    let name: String = code[at + 5..]
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    name.contains("stream") && code[at + 5 + name.len()..].starts_with('(')
                })
                .count();
            if calls == 0 {
                continue;
            }
            if file.file_name().is_some_and(|f| f == "mlx_thread.rs") {
                inside += calls;
            } else {
                outside.push(format!(
                    "{}:{}: {}",
                    file.strip_prefix(&src).unwrap().display(),
                    n + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(scanned > 10, "the scan read {scanned} files");
    assert!(
        inside >= 2,
        "the scan found {inside} stream calls in mlx_thread.rs, which gets both streams"
    );
    assert!(
        outside.is_empty(),
        "a stream from outside the MLX thread; build ops through `with_stream`:\n{}",
        outside.join("\n")
    );
}
