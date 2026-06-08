//! Process-global toggle for the optional 1-bit QJL residual on
//! rotor K-side codecs (rotor3_sym / rotor4_sym / k_rotor3 / k_rotor4).
//!
//! Mirrors the [`paged_kv_enabled`](crate::paged::paged_kv_enabled) precedent:
//! CLI > env > default. Default is **on** (`true`) per spec — the K-side
//! residual is the load-bearing fidelity lift for the rotor K codecs, and
//! turning it off is the explicit ablation/bench knob.
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
/// `RMLX_ROTOR_QJL=0` (or `off`, `false`) explicitly disables; any other value
/// (or absence) leaves QJL **enabled** per spec default.
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
        tracing::info!("rotor K-side QJL residual ENABLED (default; CLI --rotor-qjl=on)");
    } else {
        tracing::info!("rotor K-side QJL residual DISABLED (CLI --rotor-qjl=off)");
    }
}

/// Returns `true` when the K-side QJL residual codec is enabled.
///
/// Resolution order: CLI install > env var > default `true`. Not cached — the
/// cold-path call sites (`KvStorage::new`, `update_*_sym`, `update_rotor_k_*`)
/// re-read on every construction so env changes after first call still propagate.
pub fn rotor_qjl_enabled() -> bool {
    if let Some(b) = ROTOR_QJL_CLI.get().copied() {
        return b;
    }
    match std::env::var(ROTOR_QJL_ENV) {
        Ok(v) => {
            let trimmed = v.trim();
            !matches!(trimmed, "0" | "off" | "false" | "no")
        }
        Err(_) => true,
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
