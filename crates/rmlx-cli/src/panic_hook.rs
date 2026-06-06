//! Process-wide panic hook with `tracing::error!`.
//!
//! Installs a global hook via [`std::panic::set_hook`] that emits a single
//! structured `tracing::error!` event before the process unwinds (or aborts,
//! depending on the active Cargo profile — see CLAUDE.md hard rule 9).
//!
//! The hook is idempotent: a [`std::sync::Once`] guard prevents reinstallation
//! if [`install`] is called more than once (which is not expected in practice
//! but guarded as a defensive measure).
//!
//! Backtrace capture:
//! - If `RUST_BACKTRACE` is set (any value), `Backtrace::capture()` is called
//!   and the backtrace text is emitted as the `backtrace` field.
//! - If `RUST_BACKTRACE` is not set, the field is omitted to avoid the cost of
//!   an empty/disabled backtrace in steady-state production.
//!
//! Sidecar file:
//! - A `panic-<pid>.txt` file is also written to the rmlx logs dir for
//!   post-mortem analysis when the JSON log layer cannot be flushed in time.

use std::backtrace::BacktraceStatus;
use std::sync::Once;

use tracing::error;

/// Global install gate.
static HOOK_INSTALLED: Once = Once::new();

/// Install the panic hook.
///
/// Safe to call multiple times — only the first call installs the hook.
/// The `logs_dir` argument is the directory for the sidecar text file.
pub(crate) fn install(logs_dir: std::path::PathBuf) {
    HOOK_INSTALLED.call_once(move || {
        std::panic::set_hook(Box::new(move |info| {
            // — Payload ───────────────────────────────────────────────────────
            let payload: &str = info
                .payload()
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
                .unwrap_or("(non-string payload)");

            // — Location ──────────────────────────────────────────────────────
            let (loc_file, loc_line, loc_col) = info
                .location()
                .map_or(("<unknown>", 0, 0), |l| (l.file(), l.line(), l.column()));

            // — Thread name ───────────────────────────────────────────────────
            let thread = std::thread::current();
            let thread_name = thread.name().unwrap_or("<unnamed>");

            // — Backtrace (only when RUST_BACKTRACE is set) ───────────────────
            //
            // Backtrace::capture() respects the RUST_BACKTRACE / RUST_LIB_BACKTRACE
            // env vars: it returns BacktraceStatus::Disabled when neither is set.
            // We skip the field entirely in that case to keep the log concise.
            let bt = std::backtrace::Backtrace::capture();
            let has_bt = bt.status() == BacktraceStatus::Captured;

            if has_bt {
                let bt_str = bt.to_string();
                error!(
                    target: "panic",
                    payload,
                    location.file = loc_file,
                    location.line = loc_line,
                    location.column = loc_col,
                    thread = thread_name,
                    backtrace = %bt,
                    "rMLX panic",
                );
                // — Sidecar file ──────────────────────────────────────────────
                // Written in addition to the tracing event so that a crash-at-flush
                // still leaves a readable artefact on disk.
                let sidecar = logs_dir.join(format!("panic-{}.txt", std::process::id()));
                let _ = std::fs::write(
                    &sidecar,
                    format!(
                        "payload: {payload}\nfile: {loc_file}\nline: {loc_line}\ncol: {loc_col}\nthread: {thread_name}\n{bt_str}\n"
                    ),
                );
            } else {
                error!(
                    target: "panic",
                    payload,
                    location.file = loc_file,
                    location.line = loc_line,
                    location.column = loc_col,
                    thread = thread_name,
                    "rMLX panic",
                );
                let sidecar = logs_dir.join(format!("panic-{}.txt", std::process::id()));
                let _ = std::fs::write(
                    &sidecar,
                    format!(
                        "payload: {payload}\nfile: {loc_file}\nline: {loc_line}\ncol: {loc_col}\nthread: {thread_name}\n"
                    ),
                );
            }
        }));
    });
}
