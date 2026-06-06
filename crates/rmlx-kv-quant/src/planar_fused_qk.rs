//! Process-global toggle for the PlanarQuant fused-QK MSL kernel.
//!
//! Mirrors [`rotor_qjl_enabled`](crate::rotor_qjl::rotor_qjl_enabled) precedent:
//! CLI > env > default. Default is **on**; the toggle is the ablation / bench
//! knob. The dispatch site reads it from the SDPA hot path on every (b, hq, s)
//! score, so it must be cheap.
//!
//! # Why a runtime toggle
//!
//! The kernel + dispatch land as a non-default-changing perf lever. Defaulting
//! **on** delivers the speedup; the **off** path is the regression probe —
//! `--planar-fused-qk off` reverts to the dequant+SDPA legacy path and lets
//! benches diff the win cleanly.
//!
//! # No env-var
//!
//! Unlike `rotor_qjl`, this toggle is **CLI-only** — there is no
//! `RMLX_PLANAR_FUSED_QK` env var.  The flag is resolved at process startup
//! by `rmlx-cli`, never afterwards.  This keeps tests env-lock-free and
//! avoids the POSIX `setenv` race the rotor-qjl test pattern works around.

use std::sync::OnceLock;

/// CLI-set override.  Once set by `install_planar_fused_qk`, takes precedence
/// over the default.  Subsequent installs are silently ignored — the first
/// wins (matches `install_rotor_qjl`).
static PLANAR_FUSED_QK_CLI: OnceLock<bool> = OnceLock::new();

/// Install the CLI-resolved `--planar-fused-qk` flag value at startup.
///
/// Called from `rmlx-cli` after argument parsing.  May be called at most once
/// per process; later calls are dropped.
pub fn install_planar_fused_qk(enabled: bool) {
    if PLANAR_FUSED_QK_CLI.set(enabled).is_err() {
        tracing::debug!(
            requested = enabled,
            "install_planar_fused_qk: already installed; ignoring duplicate set"
        );
        return;
    }
    if enabled {
        tracing::info!("PlanarQuant fused-QK kernel ENABLED (default; CLI --planar-fused-qk=on)");
    } else {
        tracing::info!("PlanarQuant fused-QK kernel DISABLED (CLI --planar-fused-qk=off)");
    }
}

/// Returns `true` when the fused-QK kernel should be used on PlanarQuant K
/// caches.  Default `true`; flip via `--planar-fused-qk off`.
#[inline]
pub fn planar_fused_qk_enabled() -> bool {
    PLANAR_FUSED_QK_CLI.get().copied().unwrap_or(true)
}

#[cfg(test)]
#[path = "planar_fused_qk_tests.rs"]
mod planar_fused_qk_tests;
