//! Metal GPU trace capture over a bounded window of steady-state decode.
//!
//! Compiled only under the `metal-capture` feature. An ordinary build contains
//! none of this — no state, no branch on the decode path, and no way to reach
//! `MTLCaptureManager` at all.
//!
//! # Why a window
//!
//! A whole-run capture is dominated by weight load and prefill and is unusably
//! large. Kernel work studies steady-state decode, so the driver skips the first
//! `skip` decode steps (pipeline warm-up, first-touch kernel compilation) and
//! captures the next `steps`.
//!
//! # Shape
//!
//! [`Window`] is the pure policy: fed one tick per decode-step boundary, it says
//! when to open and when to close. [`arm`] / [`step`] / [`finish`] wrap it in a
//! process-global driver that owns the [`CaptureScope`]. The decode loop calls
//! [`step`] once per step and knows nothing about paths, counts, or Metal, so
//! the hook is model- and codec-agnostic.
//!
//! # Prerequisite outside our control
//!
//! Apple only inserts the Metal capture layer into a process when the
//! `MTL_CAPTURE_ENABLED` environment variable is set at launch — a framework
//! toggle, not an rMLX configuration knob. rMLX never keys behaviour off it; it
//! is read once, purely so a missing capture layer reports the fix instead of
//! mlx-c's bare "Capture layer is not inserted".
//!
//! # mlx-c API
//!
//! `mlx_metal_start_capture(path: *const c_char) -> int` — 0 = ok, non-0 = error.
//! `mlx_metal_stop_capture() -> int` — 0 = ok, non-0 = error.
//! Both live in `mlx/c/metal.h` (mlx-c 0.6.0).

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use crate::{check_status, install_error_handler, sys, Result};
use rmlx_core::error::Error;

// ---------------------------------------------------------------------------
// RAII scope
// ---------------------------------------------------------------------------

/// RAII guard that starts a Metal GPU trace on construction and stops it on
/// drop (or when [`CaptureScope::stop`] is called explicitly).
///
/// Only one scope may be active at a time — the Metal capture manager is
/// process-global. The driver in this module owns the single scope; construct
/// one directly only in tests.
#[allow(missing_debug_implementations)]
pub struct CaptureScope {
    stopped: bool,
}

impl CaptureScope {
    /// Start a Metal capture writing to `path`.
    ///
    /// Returns `Err` if the capture layer is not inserted, the path already
    /// exists, or Metal is unavailable.
    pub fn start(path: &Path) -> Result<Self> {
        install_error_handler();
        let path_str = path.to_str().ok_or_else(|| {
            Error::Mlx(format!(
                "CaptureScope::start: trace path is not valid UTF-8: {}",
                path.display()
            ))
        })?;
        let path_c = CString::new(path_str)
            .map_err(|e| Error::Mlx(format!("CaptureScope::start: path contains NUL: {e}")))?;
        // SAFETY: path_c is a valid NUL-terminated string that outlives the call.
        let status = unsafe { sys::mlx_metal_start_capture(path_c.as_ptr()) };
        // SAFETY: called immediately after the C function on the same thread.
        unsafe { check_status(status, "metal_capture::start") }?;
        Ok(Self { stopped: false })
    }

    /// Stop the capture explicitly. A no-op if already stopped.
    pub fn stop(&mut self) -> Result<()> {
        if self.stopped {
            return Ok(());
        }
        self.stopped = true;
        install_error_handler();
        // SAFETY: no preconditions; mlx_metal_stop_capture is idempotent per docs.
        let status = unsafe { sys::mlx_metal_stop_capture() };
        // SAFETY: called immediately after the C function on the same thread.
        unsafe { check_status(status, "metal_capture::stop") }
    }
}

impl Drop for CaptureScope {
    fn drop(&mut self) {
        // Best-effort: Drop cannot propagate. `finish` is the path that reports
        // a failing stop; this only covers an early unwind.
        if let Err(e) = self.stop() {
            tracing::error!(error = %e, "metal capture failed to stop on drop");
        }
    }
}

// ---------------------------------------------------------------------------
// Window policy (pure)
// ---------------------------------------------------------------------------

/// What the driver must do at the decode-step boundary just observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed three-state instruction from Window::tick to its single driver"
)]
pub enum Action {
    /// Nothing to do at this boundary.
    Idle,
    /// Open the capture scope before the step that is about to run.
    Open,
    /// Close the capture scope; the requested step budget is spent.
    Close,
}

/// Pure window policy: ticked once per decode-step boundary, it decides when the
/// capture opens and closes. Holds no Metal state, so it is unit-testable.
///
/// A tick is observed *before* the step it precedes runs. With `skip = 4` and
/// `steps = 8` the scope opens before step 5 and closes before step 13, so
/// steps 5..=12 — eight whole steps — are inside the trace.
#[derive(Debug, Clone, Copy)]
pub struct Window {
    skip: u32,
    steps: u32,
    seen: u32,
    opened: bool,
    closed: bool,
}

impl Window {
    /// Build a window that skips `skip` decode steps and then captures `steps`.
    #[must_use]
    pub const fn new(skip: u32, steps: u32) -> Self {
        Self {
            skip,
            steps,
            seen: 0,
            opened: false,
            closed: false,
        }
    }

    /// Observe one decode-step boundary and report the action it triggers.
    pub const fn tick(&mut self) -> Action {
        self.seen += 1;
        if !self.opened && self.seen > self.skip {
            self.opened = true;
            return Action::Open;
        }
        if self.opened && !self.closed && self.seen > self.skip + self.steps {
            self.closed = true;
            return Action::Close;
        }
        Action::Idle
    }

    /// Decode-step boundaries observed so far.
    #[must_use]
    pub const fn seen(&self) -> u32 {
        self.seen
    }

    /// Boundaries the generation must reach for the window to open at all.
    #[must_use]
    pub const fn steps_needed_to_open(&self) -> u32 {
        self.skip + 1
    }

    /// Whether the window ever opened.
    #[must_use]
    pub const fn opened(&self) -> bool {
        self.opened
    }

    /// Whether the requested step budget was spent (the window closed on its
    /// own rather than at the end of a short generation).
    #[must_use]
    pub const fn closed(&self) -> bool {
        self.closed
    }

    /// Whole decode steps inside the window once generation has ended.
    ///
    /// Call only after the decode loop returns: it assumes the step following
    /// the last observed boundary has run to completion, which is true at the
    /// end of the loop and false mid-loop.
    #[must_use]
    pub const fn captured_steps_at_end(&self) -> u32 {
        if !self.opened {
            return 0;
        }
        if self.closed {
            return self.steps;
        }
        self.seen - self.skip
    }
}

// ---------------------------------------------------------------------------
// Request validation
// ---------------------------------------------------------------------------

/// Validate a capture request. Pure — takes the capture-layer state rather than
/// reading the environment, so it is testable outside a Metal process.
///
/// Runs before the model loads: a rejected request must cost seconds, not a full
/// weight load followed by a failure at the first decode step.
pub fn validate(path: &Path, layer_inserted: bool, steps: u32) -> Result<()> {
    if !layer_inserted {
        return Err(Error::Mlx(
            "GPU capture requested but Apple's Metal capture layer is not inserted in \
             this process. Relaunch with MTL_CAPTURE_ENABLED=1 in the environment \
             (Metal reads it at launch and it cannot be inserted afterwards), or use \
             scripts/gpu_capture.sh, which sets it."
                .to_owned(),
        ));
    }
    if steps == 0 {
        return Err(Error::Mlx(
            "GPU capture window is zero steps wide; ask for at least one decode step".to_owned(),
        ));
    }
    if path.exists() {
        return Err(Error::Mlx(format!(
            "GPU capture destination already exists: {}. Metal refuses to overwrite a \
             trace bundle; remove it or pick another path.",
            path.display()
        )));
    }
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() && !dir.is_dir() => Err(Error::Mlx(format!(
            "GPU capture destination directory does not exist: {}",
            dir.display()
        ))),
        _ => Ok(()),
    }
}

/// Decode steps a generation must produce for a `skip`/`steps` window to open,
/// fill, and close inside the decode loop.
///
/// The loop drives one boundary per token after the prefill token, and the
/// closing boundary is itself a step that has to exist — hence the `+ 2`.
#[must_use]
pub const fn min_tokens_for_window(skip: u32, steps: u32) -> u32 {
    skip + steps + 2
}

/// Whether Apple's Metal capture layer is present in this process.
///
/// The layer is inserted at launch when `MTL_CAPTURE_ENABLED` is set; there is
/// no in-process way to insert it afterwards.
#[must_use]
pub fn capture_layer_inserted() -> bool {
    std::env::var_os("MTL_CAPTURE_ENABLED").is_some_and(|v| v != "0")
}

// ---------------------------------------------------------------------------
// Process-global driver
// ---------------------------------------------------------------------------

/// Result of [`finish`] — what the run actually produced.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed outcome set reported once per run by the CLI capture driver"
)]
pub enum Outcome {
    /// No capture was requested.
    Disabled,
    /// A trace was written. `complete` is false when generation stopped (EOS,
    /// token cap) before the requested step budget was spent.
    Captured {
        /// Path of the `.gputrace` bundle.
        path: PathBuf,
        /// Whole decode steps inside the trace.
        steps: u32,
        /// Whether the full requested budget was captured.
        complete: bool,
    },
    /// Capture was armed but generation never reached the window's first step.
    NeverOpened {
        /// Decode-step boundaries the generation did reach.
        seen: u32,
        /// Boundaries it needed to reach for the window to open.
        needed: u32,
    },
}

struct Driver {
    path: PathBuf,
    window: Window,
    scope: Option<CaptureScope>,
}

static DRIVER: Mutex<Option<Driver>> = Mutex::new(None);

/// A poisoned driver lock means a panic unwound through a capture step. The
/// state behind it is a counter plus an `Option<CaptureScope>`, both of which
/// stay meaningful after an unwind, so recovering beats aborting the process.
fn driver() -> MutexGuard<'static, Option<Driver>> {
    DRIVER.lock().unwrap_or_else(|poisoned| {
        tracing::warn!("metal capture driver lock was poisoned; recovering");
        poisoned.into_inner()
    })
}

/// Arm the decode-window capture for this process. A second arm while one is
/// live is an error, not a silent replacement.
pub fn arm(path: PathBuf, skip: u32, steps: u32) -> Result<()> {
    validate(&path, capture_layer_inserted(), steps)?;
    let mut guard = driver();
    if guard.is_some() {
        return Err(Error::Mlx(
            "GPU capture is already armed for this process".to_owned(),
        ));
    }
    tracing::info!(
        path = %path.display(),
        skip_steps = skip,
        capture_steps = steps,
        "GPU capture armed; window opens after the skipped decode steps"
    );
    *guard = Some(Driver {
        path,
        window: Window::new(skip, steps),
        scope: None,
    });
    Ok(())
}

/// Observe one decode-step boundary. Called by the shared decode loop, once per
/// step, before the step's forward pass.
///
/// A capture that cannot start is an error, not a warning: a run that quietly
/// produces no trace is the failure mode this path exists to remove.
pub fn step() -> Result<()> {
    let mut guard = driver();
    let Some(d) = guard.as_mut() else {
        return Ok(());
    };
    match d.window.tick() {
        Action::Idle => Ok(()),
        Action::Open => {
            d.scope = Some(CaptureScope::start(&d.path)?);
            tracing::info!(
                path = %d.path.display(),
                after_steps = d.window.seen() - 1,
                "GPU capture window open"
            );
            Ok(())
        }
        Action::Close => {
            if let Some(mut scope) = d.scope.take() {
                scope.stop()?;
            }
            tracing::info!(
                path = %d.path.display(),
                captured_steps = d.window.captured_steps_at_end(),
                "GPU capture window closed"
            );
            Ok(())
        }
    }
}

/// Stop any live capture and report what the run produced. Disarms the driver.
pub fn finish() -> Result<Outcome> {
    let mut guard = driver();
    let Some(mut d) = guard.take() else {
        return Ok(Outcome::Disabled);
    };
    if let Some(mut scope) = d.scope.take() {
        scope.stop()?;
    }
    if !d.window.opened() {
        return Ok(Outcome::NeverOpened {
            seen: d.window.seen(),
            needed: d.window.steps_needed_to_open(),
        });
    }
    if !d.path.exists() {
        return Err(Error::Mlx(format!(
            "GPU capture window opened and closed but no trace bundle exists at {}",
            d.path.display()
        )));
    }
    Ok(Outcome::Captured {
        path: d.path,
        steps: d.window.captured_steps_at_end(),
        complete: d.window.closed(),
    })
}

#[cfg(test)]
#[path = "metal_capture_tests.rs"]
mod metal_capture_tests;
