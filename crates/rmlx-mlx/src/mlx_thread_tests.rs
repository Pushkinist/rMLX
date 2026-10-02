//! Tests for the MLX thread: an op built on any thread evaluates on any
//! thread, and only `mlx_thread.rs` names an mlx-c stream function.

use std::sync::mpsc;

use super::*;
use crate::within_limit::{within_limit, INIT_LIMIT, LIMIT};
use crate::{add, Array, Device};

fn lazy_sum(device: Device) -> Array {
    let a = Array::from_f32_slice(&[1.0, 2.0], &[2]).unwrap();
    add(&a, &a, device).unwrap()
}

/// Build and evaluate [`lazy_sum`] on one new thread, under [`INIT_LIMIT`]:
/// the first MLX evaluation of a test can include the one-time MLX init.
fn first_evaluation(device: Device) -> Vec<u8> {
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        done_tx
            .send(lazy_sum(device).to_bytes().map_err(|e| e.to_string()))
            .ok();
    });
    within_limit(
        &done_rx,
        INIT_LIMIT,
        "the first MLX evaluation of this test (it can include the one-time MLX init)",
    )
    .unwrap_or_else(|| panic!("the thread of the first MLX evaluation panicked"))
    .unwrap_or_else(|e| panic!("the first MLX evaluation failed: {e}"))
}

/// Wait for the one-time MLX init of this process, so that a later [`LIMIT`]
/// counts only its own step. Under Miri no MLX runs.
fn after_mlx_init() {
    if cfg!(miri) {
        return;
    }
    assert_eq!(first_evaluation(Device::Cpu).len(), 8);
}

/// Build a lazy op on one thread and keep that thread alive and idle, as a
/// blocking-pool worker is between requests. Evaluate the op on a second
/// thread. The oracle builds and evaluates on one thread.
fn assert_crosses_threads(test: &str, device: Device) {
    let expected: Vec<u8> = [2.0f32, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let oracle = first_evaluation(device);
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
    let crossed = within_limit(&read_rx, LIMIT, "the cross-thread read")
        .unwrap_or_else(|| panic!("{test}: the reader panicked"));
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

/// A job on the MLX thread that hands off again runs the inner job in place.
/// A second hand-off from the MLX thread would wait forever for `turn`, which
/// the thread that posted the outer job holds until that job is done.
#[test]
fn a_hand_off_from_the_mlx_thread_runs_in_place() {
    after_mlx_init();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        done_tx.send(run(|| run(|| 5))).ok();
    });
    let nested = within_limit(&done_rx, LIMIT, "a nested hand-off")
        .unwrap_or_else(|| panic!("the nested hand-off panicked"));
    assert_eq!(nested.unwrap().unwrap(), 5);
}

/// Eight threads hand off at the same time, and each must get back the value
/// of its own job. Without `turn`, a second post replaces the first: that job
/// is lost, or the MLX thread runs it after its waiter returned. Alone, this
/// test then fails on its assertion or ends with a time-limit exit. In the
/// full test binary the process can also crash, or hang with no exit: an
/// untimed test that lost its job waits forever.
#[test]
fn concurrent_hand_offs_each_get_their_own_value() {
    const THREADS: usize = 8;
    // Miri interleaves the threads at random, so it needs fewer hand-offs.
    // Under Miri the clock moves with each step that Miri runs, so 2000
    // hand-offs take more time than `LIMIT`.
    const HAND_OFFS: usize = if cfg!(miri) { 100 } else { 2000 };
    after_mlx_init();
    let (done_tx, done_rx) = mpsc::channel();
    for t in 0..THREADS {
        let done_tx = done_tx.clone();
        std::thread::spawn(move || {
            let wrong = (0..HAND_OFFS)
                .filter(|&i| run(move || (t, i)).ok() != Some((t, i)))
                .count();
            done_tx.send(wrong).ok();
        });
    }
    drop(done_tx);
    for _ in 0..THREADS {
        let wrong = within_limit(&done_rx, LIMIT, "concurrent hand-offs")
            .unwrap_or_else(|| panic!("a hand-off thread panicked"));
        assert_eq!(
            wrong, 0,
            "{wrong} of {HAND_OFFS} hand-offs of one thread did not return the value of its job"
        );
    }
}

/// The job and its result live on the stack of the waiting thread. A job that
/// borrows a local and writes through a `&mut` reads back on that thread.
#[test]
fn a_job_borrows_from_the_stack_of_the_waiting_thread() {
    let input = [1u32, 2, 3];
    let mut written = 0u32;
    let sum = run(|| {
        written = 9;
        input.iter().sum::<u32>()
    })
    .unwrap();
    assert_eq!((sum, written), (6, 9));
}

#[test]
fn a_null_stream_error_names_the_reason_from_mlx_and_empties_the_slot() {
    LAST_ERROR.with(|slot| slot.set(Some("no Metal device".to_owned())));
    let message = no_stream(Device::Cpu).to_string();
    assert!(
        message.contains("no Metal device"),
        "the reason from mlx-c is lost: {message}"
    );
    assert_eq!(
        LAST_ERROR.with(Cell::take),
        None,
        "a stale reason would name a later, unrelated failure"
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

/// `source` with every comment blanked and every line kept. String and char
/// literals stay, so a stream function named in one (a `link_name`, a `dlsym`
/// argument) still counts, and a `//` inside a literal starts no comment.
fn without_comments(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let at = |i: usize| chars.get(i).copied();
    let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    let starts_token =
        |j: usize| j == 0 || !at(j - 1).is_some_and(|p| p.is_alphanumeric() || p == '_');
    while let Some(c) = at(i) {
        let raw_prefix = starts_token(i)
            || (i > 0 && matches!(at(i - 1), Some('b' | 'c')) && starts_token(i - 1));
        if c == '/' && at(i + 1) == Some('/') {
            while let Some(c) = at(i).filter(|&c| c != '\n') {
                out.push(blank(c));
                i += 1;
            }
        } else if c == '/' && at(i + 1) == Some('*') {
            let mut depth = 0;
            while let Some(c) = at(i) {
                if c == '/' && at(i + 1) == Some('*') {
                    depth += 1;
                    out.push_str("  ");
                    i += 2;
                } else if c == '*' && at(i + 1) == Some('/') {
                    depth -= 1;
                    out.push_str("  ");
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    out.push(blank(c));
                    i += 1;
                }
            }
        } else if c == 'r' && raw_prefix && matches!(at(i + 1), Some('"' | '#')) {
            let hashes = chars[i + 1..].iter().take_while(|&&h| h == '#').count();
            if at(i + 1 + hashes) != Some('"') {
                out.push(c);
                i += 1;
                continue;
            }
            let close: String = std::iter::once('"')
                .chain("#".repeat(hashes).chars())
                .collect();
            let body_start = i + 2 + hashes;
            let rest: String = chars[body_start..].iter().collect();
            let end = rest.find(&close).map_or(chars.len(), |at| {
                body_start + rest[..at].chars().count() + close.chars().count()
            });
            out.extend(&chars[i..end]);
            i = end;
        } else if c == '"' {
            out.push(c);
            i += 1;
            while let Some(c) = at(i) {
                out.push(c);
                i += 1;
                if c == '\\' {
                    out.extend(at(i));
                    i += 1;
                } else if c == '"' {
                    break;
                }
            }
        } else if c == '\'' && at(i + 1) == Some('\\') {
            let end = (i + 3..chars.len())
                .find(|&j| chars[j] == '\'')
                .map_or(chars.len(), |j| j + 1);
            out.extend(&chars[i..end]);
            i = end;
        } else if c == '\'' && at(i + 2) == Some('\'') {
            out.extend(&chars[i..i + 3]);
            i += 3;
        } else {
            out.push(c);
            i += 1;
        }
    }
    out
}

/// The mlx-c stream functions that `code` names: every `mlx_` identifier that
/// holds `stream`, except the handle type `mlx_stream`, and every
/// `mlx_synchronize` function, which waits on a stream.
fn stream_functions(code: &str) -> Vec<&str> {
    code.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .map(|name| name.trim_start_matches('_'))
        .filter(|name| {
            name.starts_with("mlx_synchronize")
                || (name.starts_with("mlx_") && name.contains("stream") && *name != "mlx_stream")
        })
        .collect()
}

/// The scan below must see a stream function however a file names it.
#[test]
fn the_stream_scan_sees_every_spelling_of_a_stream_function() {
    let found = |source: &str| stream_functions(&without_comments(source)).len();
    let named = [
        (
            "use sys::mlx_default_cpu_stream_new as f;\nunsafe { f() };",
            "an alias",
        ),
        (
            "use crate::sys::*;\nunsafe { mlx_default_gpu_stream_new() };",
            "a glob import",
        ),
        ("let make = sys::mlx_stream_new_device;", "a fn pointer"),
        (
            "let url = \"http://x\"; unsafe { sys::mlx_stream_new() };",
            "a `//` in a string",
        ),
        (
            "let raw = r#\"/*\"#; unsafe { sys::mlx_stream_new() };",
            "a `/*` in a raw string",
        ),
        (
            "let q = '\"'; unsafe { sys::mlx_stream_new() };",
            "a quote in a char literal",
        ),
        ("unsafe { sys::mlx_synchronize(s) };", "a synchronize"),
        (
            "unsafe { sys::mlx_synchronize_default() };",
            "a default synchronize",
        ),
        (
            "let b = br\"\\\"; let s = \"//\"; unsafe { sys::mlx_stream_new() };",
            "a raw byte string that ends in a backslash",
        ),
        (
            "let b = br#\"\\\"#; let s = \"//\"; unsafe { sys::mlx_stream_new() };",
            "a raw byte string with hashes that ends in a backslash",
        ),
        (
            "let c = cr\"\\\"; let s = \"//\"; unsafe { sys::mlx_stream_new() };",
            "a raw C string that ends in a backslash",
        ),
        (
            "#[link_name = \"mlx_default_cpu_stream_new\"]\nfn f();",
            "a link name",
        ),
        (
            "unsafe { sys::_mlx_stream_private(s) };",
            "a leading underscore",
        ),
    ];
    for (source, spelling) in named {
        assert_eq!(
            found(source),
            1,
            "{spelling} hides a stream function: {source}"
        );
    }
    let not_named = [
        ("fn build(s: sys::mlx_stream) {}", "the handle type"),
        ("// unsafe { sys::mlx_stream_new() }", "a line comment"),
        (
            "/* a /* nested */ sys::mlx_stream_new() */",
            "a nested block comment",
        ),
        (
            "let q = '\"'; // sys::mlx_stream_new()",
            "a comment after a quote in a char literal",
        ),
        (
            "fn f<'a>(s: &'a str) -> &'a str { s } // sys::mlx_stream_new()",
            "a comment after lifetimes",
        ),
        (
            "let s = \"\\\"\"; // sys::mlx_stream_new()",
            "a comment after an escaped quote",
        ),
    ];
    for (source, spelling) in not_named {
        assert_eq!(
            found(source),
            0,
            "{spelling} names no stream function: {source}"
        );
    }
}

/// Every mlx-c stream function that a non-test source file of this crate
/// names. Only `mlx_thread.rs` may name one: a stream from anywhere else is a
/// stream of the calling thread, and an op built on it evaluates only there.
///
/// The scan reads names, not calls. It cannot see a name that a macro builds
/// from parts, a symbol looked up from a string built at run time, or a crate
/// other than this one; no other crate can reach `sys`.
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
    let mut inside = Vec::new();
    for file in files.iter().filter(|f| !is_test(f)) {
        scanned += 1;
        let code = without_comments(&std::fs::read_to_string(file).unwrap());
        for (n, line) in code.lines().enumerate() {
            let names = stream_functions(line);
            if names.is_empty() {
                continue;
            }
            if file.strip_prefix(&src) == Ok(std::path::Path::new("mlx_thread.rs")) {
                inside.extend(names.iter().map(|name| (*name).to_owned()));
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
    for name in [
        "mlx_default_cpu_stream_new",
        "mlx_default_gpu_stream_new",
        "mlx_set_default_stream",
    ] {
        assert!(
            inside.iter().any(|found| found == name),
            "the scan did not find {name} in mlx_thread.rs, which calls it; found {inside:?}"
        );
    }
    assert!(
        outside.is_empty(),
        "a stream from outside the MLX thread; build ops through `with_stream`:\n{}",
        outside.join("\n")
    );
}
