//! Paged-KV feature flag: resolver functions and process-global state.

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default page size in tokens (32 tokens per page).
///
/// 32 is the TurboQuant / PlanarQuant group size, so each page holds exactly
/// one complete quantiser group per element — no partial-group cross-page
/// boundary issues.
pub(super) const DEFAULT_PAGE_TOKENS: i32 = 32;

// ── Global feature flag ────────────────────────────────────────────────────────

/// process-global CLI-installed override for the paged-KV toggle.
static PAGED_KV_CLI: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static PAGED_KV_PAGE_TOKENS_CLI: std::sync::OnceLock<i32> = std::sync::OnceLock::new();

/// pure resolver — CLI > default(false). Public for unit tests.
pub fn resolve_paged_kv(cli: bool) -> bool {
    cli
}

/// pure resolver for `--paged-kv-page-tokens` (positive integer).
pub fn resolve_paged_kv_page_tokens(cli: Option<i32>) -> i32 {
    if let Some(n) = cli {
        if n > 0 {
            return n;
        }
    }
    DEFAULT_PAGE_TOKENS
}

/// install the CLI-resolved paged-KV config at serve startup. First
/// call wins; later calls are no-ops (warn on disagreement).
#[allow(clippy::cognitive_complexity)]
pub fn install_paged_kv(cli_enabled: bool, cli_page_tokens: Option<i32>) {
    let final_enabled = resolve_paged_kv(cli_enabled);
    let final_tokens = resolve_paged_kv_page_tokens(cli_page_tokens);

    if PAGED_KV_CLI.set(final_enabled).is_err() {
        let existing = PAGED_KV_CLI.get().copied().unwrap_or(false);
        if existing != final_enabled {
            tracing::warn!(
                existing,
                requested = final_enabled,
                "install_paged_kv called more than once; keeping the first value"
            );
        }
    }
    if PAGED_KV_PAGE_TOKENS_CLI.set(final_tokens).is_err() {
        let existing = PAGED_KV_PAGE_TOKENS_CLI.get().copied().unwrap_or(0);
        if existing != final_tokens {
            tracing::warn!(
                existing,
                requested = final_tokens,
                "install_paged_kv page_tokens disagreement; keeping the first value"
            );
        }
    }
    if final_enabled {
        tracing::info!(
            page_tokens = final_tokens,
            "paged-KV ENABLED via CLI flag (--paged-kv)"
        );
    } else {
        tracing::debug!("paged-KV disabled (default / CLI off)");
    }
}

/// Returns `true` when paged KV is enabled. Reads the process-global override
/// installed by `install_paged_kv`; returns `false` when not yet installed.
pub fn paged_kv_enabled() -> bool {
    PAGED_KV_CLI.get().copied().unwrap_or(false)
}

/// Returns the page size in tokens. Reads the process-global override
/// installed by `install_paged_kv`; returns the default when not yet installed.
pub fn paged_kv_page_tokens() -> i32 {
    PAGED_KV_PAGE_TOKENS_CLI
        .get()
        .copied()
        .unwrap_or(DEFAULT_PAGE_TOKENS)
}
