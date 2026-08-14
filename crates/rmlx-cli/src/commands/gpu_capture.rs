// CLI binary: user-facing output. tracing carries the structured record; the
// operator needs the trace path on stdout to open it.
#![allow(clippy::print_stdout)]
//! Driver for `rmlx baseline --gpu-capture` — the debug-only Metal GPU trace of
//! a bounded window of steady-state decode.
//!
//! Compiled only under the `metal-capture` feature; without it neither this
//! module nor the flags that reach it exist, so a release binary cannot capture.
//!
//! The whole point of this file is that failure is loud. Every way a capture can
//! fail to happen — no capture layer, occupied destination, a generation too
//! short to reach the window, a window that opened but wrote nothing — ends in a
//! non-zero exit with the reason, never in a run that quietly produced no trace.

use std::path::Path;

use anyhow::{bail, Result};
use rmlx_mlx::metal_capture::{self, Outcome};

/// Validate and arm the capture window. Returns whether a capture was requested,
/// which [`report`] needs to decide whether silence is acceptable.
///
/// Called before the model loads so a bad request costs seconds rather than a
/// full weight load.
pub(crate) fn arm(path: Option<&Path>, skip: u32, steps: u32, max_tokens: u32) -> Result<bool> {
    let Some(path) = path else {
        return Ok(false);
    };
    // The decode loop drives one window tick per token after the prefill token,
    // so a generation shorter than this can never close the window. Catching it
    // here is the difference between a clear message and a trace that silently
    // holds fewer steps than asked for.
    let need = metal_capture::min_tokens_for_window(skip, steps);
    if max_tokens < need {
        bail!(
            "--gpu-capture needs at least {need} generated tokens to open, fill and close a \
             {skip}-skip / {steps}-step window, but --max-tokens is {max_tokens}. \
             Raise --max-tokens to {need} or shrink the window."
        );
    }
    metal_capture::arm(path.to_path_buf(), skip, steps)?;
    Ok(true)
}

/// Stop any live capture and turn the outcome into an exit status plus an
/// operator-facing summary. Call once, after generation, whether it succeeded
/// or not.
pub(crate) fn report(requested: bool) -> Result<()> {
    describe(metal_capture::finish()?, requested)
}

/// Turn a finished capture into an exit status and an operator summary.
/// Separate from [`report`] so every outcome — including the ones that need a
/// real generation to reach — is reachable from a test.
fn describe(outcome: Outcome, requested: bool) -> Result<()> {
    match outcome {
        Outcome::Disabled => {
            if requested {
                bail!(
                    "GPU capture was armed for this run but the driver was already disarmed \
                     when the run ended. Nothing was written. This is an internal \
                     inconsistency — a second capture driver ran in the same process."
                );
            }
            Ok(())
        }
        // Zero steps means the decode loop was never entered at all, which is a
        // different fault from a generation that was merely too short: the arch
        // does not route through the shared loop the hook lives in. Saying
        // "raise --max-tokens" there would send the operator down a dead end.
        Outcome::NeverOpened { seen: 0, .. } => bail!(
            "GPU capture window never opened: the run produced no decode steps at all. \
             Either generation stopped at the prefill token, or this architecture does not \
             decode through the shared loop the capture hook lives in \
             (rmlx_models::decode_loop::pipelined_decode)."
        ),
        Outcome::NeverOpened { seen, needed } => bail!(
            "GPU capture window never opened: generation produced {seen} decode steps but the \
             window opens at step {needed}. Lower --gpu-capture-skip or raise --max-tokens."
        ),
        Outcome::Captured {
            path,
            steps,
            complete,
        } => {
            let bytes = bundle_bytes(&path);
            if !complete {
                tracing::warn!(
                    path = %path.display(),
                    captured_steps = steps,
                    "GPU capture window closed early — generation stopped before the step budget was spent"
                );
            }
            tracing::info!(
                path = %path.display(),
                captured_steps = steps,
                bytes,
                complete,
                "GPU capture written"
            );
            println!(
                "gpu-capture: {} ({steps} decode steps, {bytes} bytes{})",
                path.display(),
                if complete { "" } else { ", SHORT" }
            );
            println!("gpu-capture: open with  open '{}'", path.display());
            Ok(())
        }
    }
}

/// Total size of a `.gputrace` bundle. Bundles are directories; a read error
/// contributes nothing rather than aborting a successful capture.
fn bundle_bytes(path: &Path) -> u64 {
    fn walk(dir: &Path) -> u64 {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return 0;
        };
        entries
            .flatten()
            .map(|e| match e.file_type() {
                Ok(t) if t.is_dir() => walk(&e.path()),
                Ok(_) => e.metadata().map_or(0, |m| m.len()),
                Err(_) => 0,
            })
            .sum()
    }
    match std::fs::metadata(path) {
        Ok(m) if m.is_dir() => walk(path),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

#[cfg(test)]
#[path = "gpu_capture_tests.rs"]
mod gpu_capture_tests;
