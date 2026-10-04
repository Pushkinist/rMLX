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
//! [`crate::with_eval_lock`] runs every evaluation here. So an array built on
//! any thread evaluates from any thread, and no loader, cache or request path
//! has to evaluate its arrays before another thread uses them.
//!
//! A hand-off allocates nothing: the job and its result stay on the stack of
//! the waiting thread, and the MLX thread reaches them through one mailbox.
//!
//! Only this module names an mlx-c stream function:
//! `every_stream_comes_from_the_mlx_thread` fails on one in any other source
//! file of the crate.

use std::cell::Cell;
use std::marker::PhantomData;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::Thread;

use rmlx_core::error::{Error, Result};

use crate::{check_status, sys, Device, LAST_ERROR};

thread_local! {
    static ON_MLX_THREAD: Cell<bool> = const { Cell::new(false) };
    /// The default streams of this thread are already the MLX thread's streams.
    static DEFAULTS_SET: Cell<bool> = const { Cell::new(false) };
}

/// A job on the stack of the thread that waits for it: a pointer to the
/// closure, and [`call`] for the type of that closure.
#[derive(Clone, Copy)]
struct JobRef {
    job: *mut (),
    call: unsafe fn(*mut ()),
}

// SAFETY: `Posted::new` makes a `JobRef` only from a closure that is `Send`.
unsafe impl Send for JobRef {}

/// Call the closure of type `F` that `job` points to.
///
/// # Safety
/// `job` points to a live `F`, and nothing else uses that `F` during the call.
unsafe fn call<F: FnMut()>(job: *mut ()) {
    // SAFETY: the caller's contract.
    unsafe { (*job.cast::<F>())() }
}

/// A posted job and the thread that waits for it.
struct Post {
    job: JobRef,
    waiter: Thread,
}

/// Where a waiting thread puts its job for the MLX thread.
struct Mailbox {
    /// Held by the one thread whose job is posted, until it sees `done`.
    turn: Mutex<()>,
    post: Mutex<Option<Post>>,
    /// Set when `post` holds a job that the MLX thread has not taken.
    posted: AtomicBool,
    /// Set when the MLX thread has run the posted job.
    done: AtomicBool,
}

static MAILBOX: Mailbox = Mailbox {
    turn: Mutex::new(()),
    post: Mutex::new(None),
    posted: AtomicBool::new(false),
    done: AtomicBool::new(false),
};

/// Park until `flag` is set. The thread that sets `flag` unparks this one
/// after it.
fn wait_for(flag: &AtomicBool) {
    while !flag.load(Ordering::Acquire) {
        std::thread::park();
    }
}

/// A job posted to the MLX thread. It holds `turn` and borrows the job. Its
/// drop waits for `done` before it gives up `turn`, so no exit from
/// `hand_off`, a return or an unwind, frees a job that the MLX thread can run.
struct Posted<'job> {
    mailbox: &'static Mailbox,
    _turn: MutexGuard<'static, ()>,
    _job: PhantomData<&'job mut ()>,
}

impl<'job> Posted<'job> {
    /// Post `job` and wake the MLX thread.
    ///
    /// # Safety
    /// Drop the returned value. Do not forget or leak it: the MLX thread can
    /// run `job` until the drop sees `done`.
    unsafe fn new<F: FnMut() + Send>(
        mailbox: &'static Mailbox,
        mlx_thread: &Thread,
        job: &'job mut F,
    ) -> Self {
        let turn = mailbox.turn.lock().unwrap_or_else(PoisonError::into_inner);
        *mailbox.post.lock().unwrap_or_else(PoisonError::into_inner) = Some(Post {
            job: JobRef {
                job: std::ptr::from_mut(job).cast(),
                call: call::<F>,
            },
            waiter: std::thread::current(),
        });
        let posted = Self {
            mailbox,
            _turn: turn,
            _job: PhantomData,
        };
        mailbox.posted.store(true, Ordering::Release);
        mlx_thread.unpark();
        posted
    }
}

impl Drop for Posted<'_> {
    fn drop(&mut self) {
        wait_for(&self.mailbox.done);
        self.mailbox.done.store(false, Ordering::Relaxed);
    }
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

/// The mailbox, and the MLX thread to wake for it. The first call starts the
/// thread.
fn mailbox() -> Result<(&'static Mailbox, &'static Thread)> {
    static STARTED: OnceLock<std::result::Result<Thread, String>> = OnceLock::new();
    STARTED
        .get_or_init(|| {
            std::thread::Builder::new()
                .name("rmlx-mlx".to_owned())
                // The stack of a main thread, not the 2 MiB of a spawned one:
                // every evaluation of every command runs here.
                .stack_size(8 << 20)
                .spawn(|| serve(&MAILBOX))
                .map(|handle| handle.thread().clone())
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(|thread| (&MAILBOX, thread))
        .map_err(|e| Error::Mlx(format!("the MLX thread did not start: {e}")))
}

/// The loop of the MLX thread: take each posted job, run it, mark it done.
fn serve(mailbox: &Mailbox) {
    ON_MLX_THREAD.with(|on| on.set(true));
    loop {
        wait_for(&mailbox.posted);
        mailbox.posted.store(false, Ordering::Relaxed);
        let post = mailbox
            .post
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let Some(Post { job, waiter }) = post else {
            continue;
        };
        // SAFETY: a `Posted` made `job` from a closure of the type that
        // `job.call` expects. It holds `turn`, so the next `done` is for this
        // job. It borrows the closure and is dropped, never forgotten, so the
        // closure lives until its drop sees that `done`.
        unsafe { (job.call)(job.job) };
        mailbox.done.store(true, Ordering::Release);
        waiter.unpark();
    }
}

/// Run `job` on the MLX thread and return when it is done: the drop of the
/// `Posted` waits for it.
fn hand_off<F: FnMut() + Send>(job: &mut F) -> Result<()> {
    let (mailbox, mlx_thread) = mailbox()?;
    // SAFETY: `_posted` drops at the end of this function.
    let _posted = unsafe { Posted::new(mailbox, mlx_thread, job) };
    Ok(())
}

/// Run `f` on the MLX thread and return its value. On the MLX thread, run `f`
/// in place.
///
/// The mlx-c error message that `f` leaves on the MLX thread moves to the
/// calling thread, so a `check_status` after this call reads it. A panic in
/// `f` continues on the calling thread.
pub(crate) fn run<T: Send>(f: impl FnOnce() -> T + Send) -> Result<T> {
    if ON_MLX_THREAD.with(Cell::get) {
        return Ok(f());
    }
    let mut f = Some(f);
    let mut outcome = None;
    hand_off(&mut || {
        if let Some(f) = f.take() {
            let value = panic::catch_unwind(AssertUnwindSafe(f));
            outcome = Some((value, LAST_ERROR.with(Cell::take)));
        }
    })?;
    let Some((value, error)) = outcome else {
        return Err(Error::Mlx("the MLX thread did not run the job".to_owned()));
    };
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
/// is left out after [`crate::forbid_gpu`], which also makes the CPU the
/// default device, and on a Mac without Metal.
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

/// Wait until the GPU has run every command buffer of the MLX thread's GPU
/// stream and Metal has called their completion handlers.
///
/// For memory measurement only; no engine path calls it. An evaluation returns
/// when its outputs are written. MLX frees the temporaries of a command buffer
/// in the completion handler of that buffer, which Metal can call after the
/// evaluation returned. Until then [`crate::mlx_active_memory_bytes`] still
/// counts them. After this call it does not: a memory measurement calls this
/// before each reading.
///
/// Do not call it from a job that runs on the MLX thread (a compiled closure
/// body): that job holds the evaluation lock, which this call takes. Such a
/// call returns an error and does not wait.
///
/// # Errors
/// [`Error::GpuForbidden`] after [`crate::forbid_gpu`]: no GPU stream is
/// created and no Metal API is called. An error for a call from the MLX
/// thread, when the MLX thread cannot start, when MLX gives no GPU stream, or
/// when a command buffer of the stream failed.
pub fn synchronize_gpu() -> Result<()> {
    crate::check_gpu_allowed("synchronize_gpu")?;
    if ON_MLX_THREAD.with(Cell::get) {
        return Err(Error::Mlx(
            "synchronize_gpu: called from a job on the MLX thread, which holds the evaluation \
             lock that this call takes"
                .to_owned(),
        ));
    }
    crate::install_error_handler();
    let stream = Stream(mlx_stream(Device::Gpu)?);
    let status = crate::with_eval_lock(move || {
        // The whole `Stream` moves into the job: its field alone is not `Send`.
        let stream = stream;
        // SAFETY: the handle is valid for the life of the process, and the job
        // runs on the MLX thread, which owns the encoder behind it.
        unsafe { sys::mlx_synchronize(stream.0) }
    })?;
    // SAFETY: `with_eval_lock` moved the error message of the call to this
    // thread's error slot, and no mlx-c call ran since.
    unsafe { check_status(status, "mlx_synchronize") }
}

/// Whether the MLX thread has created its GPU stream.
#[cfg(test)]
pub(crate) fn has_gpu_stream() -> bool {
    GPU_STREAM.get().is_some()
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
        return Err(no_stream(device));
    }
    Ok(*cell.get_or_init(|| Stream(handle)))
}

/// The error for a null stream handle, with the reason that mlx-c left in the
/// error slot of this thread. The slot is empty afterwards.
fn no_stream(device: Device) -> Error {
    let reason = LAST_ERROR
        .with(Cell::take)
        .unwrap_or_else(|| "mlx-c gave no reason".to_owned());
    Error::Mlx(format!("MLX gave no default {device:?} stream: {reason}"))
}

#[cfg(test)]
#[path = "mlx_thread_tests.rs"]
mod mlx_thread_tests;

#[cfg(test)]
#[path = "synchronize_tests.rs"]
mod synchronize_tests;
