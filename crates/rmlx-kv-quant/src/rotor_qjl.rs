//! Process-global toggle for the optional 1-bit QJL residual on
//! rotor K-side codecs (rotor3_sym / rotor4_sym / k_rotor3 / k_rotor4).
//!
//! Mirrors the [`paged_kv_enabled`](crate::paged::paged_kv_enabled) precedent:
//! CLI > env > default. Default is **off** (`false`).
//!
//! # Why off by default
//!
//! The QJL sideband has no MSL kernel, so turning it on forces the rotor K
//! encode + dequant onto the CPU on every decode step — the GPU sits idle and
//! the codec decodes at single-digit (often sub-1) TPS. With QJL off the rotor
//! K path runs the Metal fused flash-decode-over-quant kernel, recovering
//! roughly 16-70x decode and 3-4x prefill/TTFT. Measured across two
//! architectures and a context sweep (short prompts and a long-context needle),
//! the 1-bit residual bought no measurable accuracy — identical temp=0 output
//! and identical needle retrieval on vs off. So the fast Metal path is the
//! default, and QJL is the opt-in fidelity / ablation knob.
//!
//! # Storage cost
//!
//! When enabled, one signed-bit per `head_dim` element per token is appended
//! to each token's payload. At `head_dim=128` this is 16 bytes/token/head —
//! amortised by the rotor-table savings (rotor4_sym K-side compression goes
//! from ~4.7× to ~4.1× with QJL on at `head_dim=128`).
//!
//! # Wire format
//!
//! Packed u8 row-major, shape `[B, kv_h, max_seq, ceil(head_dim/8)]`. Bit
//! order matches the Python `rotorquant/turboquant/rotorquant.py` reference
//! (LSB = element 0, MSB = element 7). One bit per residual sign.

use std::sync::OnceLock;

/// Environment variable mirror of the CLI flag.
///
/// `RMLX_ROTOR_QJL=1` (or `on`, `true`, `yes`) explicitly enables; any other
/// value (or absence) leaves QJL **disabled** per the default.
const ROTOR_QJL_ENV: &str = "RMLX_ROTOR_QJL";

/// CLI-set override. Once set by `install_rotor_qjl`, takes precedence over the
/// env var so the per-launch `--rotor-qjl on|off` setting is authoritative.
static ROTOR_QJL_CLI: OnceLock<bool> = OnceLock::new();

/// Install the CLI-resolved `--rotor-qjl` flag value at startup.
///
/// Called from `rmlx-cli` after argument parsing. May be called at most once
/// per process; subsequent calls are silently ignored (the first wins).
pub fn install_rotor_qjl(enabled: bool) {
    if ROTOR_QJL_CLI.set(enabled).is_err() {
        tracing::debug!(
            requested = enabled,
            "install_rotor_qjl: already installed; ignoring duplicate set"
        );
        return;
    }
    if enabled {
        tracing::info!(
            "rotor K-side QJL residual ENABLED (CLI --rotor-qjl=on) — opt-in; disables the \
             rotor Metal fused-decode path and runs the K encode + dequant on CPU"
        );
    } else {
        tracing::info!("rotor K-side QJL residual DISABLED (default; CLI --rotor-qjl=off)");
    }
}

/// Returns `true` when the K-side QJL residual codec is enabled.
///
/// Resolution order: CLI install > env var > default `false`. Not cached — the
/// cold-path call sites (`KvStorage::new`, `update_*_sym`, `update_rotor_k_*`)
/// re-read on every construction so env changes after first call still propagate.
pub fn rotor_qjl_enabled() -> bool {
    if let Some(b) = ROTOR_QJL_CLI.get().copied() {
        return b;
    }
    match std::env::var(ROTOR_QJL_ENV) {
        Ok(v) => {
            let trimmed = v.trim();
            matches!(trimmed, "1" | "on" | "true" | "yes")
        }
        Err(_) => false,
    }
}

/// `true` when the CLI override (`install_rotor_qjl`) has been installed in this
/// process. Test-only helper: lets cross-module tests that toggle the env var
/// detect the CLI shadow and skip the env-based assertions.
#[cfg(test)]
pub(crate) fn rotor_qjl_cli_is_set() -> bool {
    ROTOR_QJL_CLI.get().is_some()
}

#[cfg(test)]
#[path = "rotor_qjl_tests.rs"]
mod rotor_qjl_tests;
