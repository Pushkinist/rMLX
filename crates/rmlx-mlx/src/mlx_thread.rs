//! The MLX thread: the one thread that owns the streams every op is built on,
//! and the only thread that evaluates.
//!
//! MLX keeps the command encoder of a stream in the thread that created the
//! stream, and an evaluation looks for the encoder only in the evaluating
//! thread (`mlx/backend/cpu/encoder.cpp` from MLX 0.32, the Metal device on
//! every MLX). An op records its stream when it is built. So an array built on
//! the stream of one thread fails when another thread evaluates it ("There is
//! no Stream(cpu, N) in current thread."), or waits forever.
//!
//! This thread creates one CPU stream and one GPU stream. [`crate::with_stream`]
//! builds every op, on every thread, on one of the two, and makes them the
//! default streams of the building thread too, because MLX builds some ops
//! inside other ops on the default stream of the default device.
//! [`crate::with_eval_lock`] runs every evaluation here. So an array built on any thread evaluates from any thread,
//! and no loader, cache or request path has to evaluate its arrays before
//! another thread uses them.
//!
//! Only this module gets a stream from mlx-c:
//! `every_stream_comes_from_the_mlx_thread` fails on a stream call in any
//! other source file of the crate.

use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc;
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};

use crate::{check_status, sys, Device, LAST_ERROR};

type Job = Box<dyn FnOnce() + Send>;

thread_local! {
    static ON_MLX_THREAD: Cell<bool> = const { Cell::new(false) };
    /// The default streams of this thread are already the MLX thread's streams.
    static DEFAULTS_SET: Cell<bool> = const { Cell::new(false) };
}

/// A stream of the MLX thread. The handle is never freed: MLX keeps a stream
/// and its encoder for the life of the process.
#[derive(Clone, Copy)]
struct Stream(sys::mlx_stream);

// SAFETY: an `mlx_stream` points to an immutable `{device, index}` value that
// MLX reads from any thread. Only the MLX thread uses the encoder behind it.
unsafe impl Send for Stream {}
// SAFETY: as for `Send`; the value behind the handle never changes.
unsafe impl Sync for Stream {}

static CPU_STREAM: OnceLock<Stream> = OnceLock::new();
static GPU_STREAM: OnceLock<Stream> = OnceLock::new();

fn queue() -> Result<&'static mpsc::Sender<Job>> {
    static QUEUE: OnceLock<std::result::Result<mpsc::Sender<Job>, String>> = OnceLock::new();
    QUEUE
        .get_or_init(|| {
            let (jobs, incoming) = mpsc::channel::<Job>();
            std::thread::Builder::new()
                .name("rmlx-mlx".to_owned())
                // The stack of a main thread, not the 2 MiB of a spawned one:
                // every evaluation of every command runs here.
                .stack_size(8 << 20)
                .spawn(move || {
                    ON_MLX_THREAD.with(|on| on.set(true));
                    for job in incoming {
                        job();
                    }
                })
                .map(|_| jobs)
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("the MLX thread did not start: {e}")))
}

fn stopped() -> Error {
    Error::Mlx("the MLX thread stopped".to_owned())
}

/// Run `f` on the MLX thread and return its value. On the MLX thread, run `f`
/// in place.
///
/// The mlx-c error message that `f` leaves on the MLX thread moves to the
/// calling thread, so a `check_status` after this call reads it. A panic in
/// `f` continues on the calling thread.
pub(crate) fn run<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> Result<T> {
    if ON_MLX_THREAD.with(Cell::get) {
        return Ok(f());
    }
    let (done, finished) = mpsc::sync_channel(1);
    let job: Job = Box::new(move || {
        let value = panic::catch_unwind(AssertUnwindSafe(f));
        let error = LAST_ERROR.with(Cell::take);
        done.send((value, error)).ok();
    });
    queue()?.send(job).map_err(|_| stopped())?;
    let (value, error) = finished.recv().map_err(|_| stopped())?;
    LAST_ERROR.with(|slot| slot.set(error));
    match value {
        Ok(value) => Ok(value),
        Err(payload) => panic::resume_unwind(payload),
    }
}

/// The MLX thread's stream for `device`. The first call on a thread also makes
/// the MLX thread's streams the default streams of that thread.
pub(crate) fn stream(device: Device) -> Result<sys::mlx_stream> {
    let stream = mlx_stream(device)?;
    set_thread_defaults()?;
    Ok(stream)
}

/// The MLX thread's stream for `device`, created on the MLX thread at first
/// use.
fn mlx_stream(device: Device) -> Result<sys::mlx_stream> {
    let cell = match device {
        Device::Cpu => &CPU_STREAM,
        Device::Gpu => &GPU_STREAM,
    };
    match cell.get() {
        Some(stream) => Ok(stream.0),
        None => Ok(run(move || default_stream(device, cell))??.0),
    }
}

/// Make the MLX thread's streams the default streams of the calling thread,
/// for both devices: MLX builds some ops inside other ops on the default
/// stream of the default device, whatever device the caller builds on. The GPU
/// is left out after [`crate::forbid_gpu`] and on a Mac without Metal.
fn set_thread_defaults() -> Result<()> {
    if ON_MLX_THREAD.with(Cell::get) || DEFAULTS_SET.with(Cell::get) {
        return Ok(());
    }
    let mut devices = vec![Device::Cpu];
    if !crate::gpu_forbidden() && metal_available() {
        devices.push(Device::Gpu);
    }
    for device in devices {
        let stream = mlx_stream(device)?;
        // SAFETY: the handle is valid for the life of the process; MLX copies
        // the stream value into this thread's default slot for its device.
        let status = unsafe { sys::mlx_set_default_stream(stream) };
        // SAFETY: called immediately after the C function on this thread.
        unsafe { check_status(status, "mlx_set_default_stream") }?;
    }
    DEFAULTS_SET.with(|set| set.set(true));
    Ok(())
}

fn metal_available() -> bool {
    let mut available = false;
    // SAFETY: the out-pointer is a stack `bool` we own.
    let status = unsafe { sys::mlx_metal_is_available(&raw mut available) };
    status == 0 && available
}

/// Only the MLX thread calls this, so no other thread can set `cell` first.
fn default_stream(device: Device, cell: &'static OnceLock<Stream>) -> Result<Stream> {
    if let Some(stream) = cell.get() {
        return Ok(*stream);
    }
    // SAFETY: each call returns a new reference to the default stream of the
    // calling thread. MLX creates that stream and its encoder at first use.
    let handle = unsafe {
        match device {
            Device::Cpu => sys::mlx_default_cpu_stream_new(),
            Device::Gpu => sys::mlx_default_gpu_stream_new(),
        }
    };
    if handle.ctx.is_null() {
        return Err(Error::Mlx(format!("MLX gave no default {device:?} stream")));
    }
    Ok(*cell.get_or_init(|| Stream(handle)))
}

#[cfg(test)]
#[path = "mlx_thread_tests.rs"]
mod mlx_thread_tests;
