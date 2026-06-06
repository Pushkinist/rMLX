//! Paged-KV feature flag: resolver functions and process-global state.

// ── Constants ─────────────────────────────────────────────────────────────────

/// Environment-variable name to enable paged KV allocation.
///
/// `RMLX_PAGED_KV=1` opts in. Any other value (or absent) keeps the
/// contiguous-growth path.
const PAGED_KV_ENV: &str = "RMLX_PAGED_KV";

/// Environment-variable name to control the per-page token count.
///
/// Default: 32 tokens per page. Must be a positive integer.
const KV_PAGE_SIZE_ENV: &str = "RMLX_KV_PAGE_SIZE";

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

/// pure resolver — CLI > env > default(false). Public for unit tests.
pub fn resolve_paged_kv(cli: bool, env: Option<&str>) -> bool {
    if cli {
        return true;
    }
    if let Some(s) = env {
        return s.trim() == "1";
    }
    false
}

/// pure resolver for `--paged-kv-page-tokens` (positive integer).
pub fn resolve_paged_kv_page_tokens(cli: Option<i32>, env: Option<&str>) -> i32 {
    if let Some(n) = cli {
        if n > 0 {
            return n;
        }
    }
    if let Some(s) = env {
        if let Ok(n) = s.trim().parse::<i32>() {
            if n > 0 {
                return n;
            }
        }
    }
    DEFAULT_PAGE_TOKENS
}

/// install the CLI-resolved paged-KV config at serve startup. First
/// call wins; later calls are no-ops (warn on disagreement).
#[allow(clippy::cognitive_complexity)]
pub fn install_paged_kv(cli_enabled: bool, cli_page_tokens: Option<i32>) {
    let env_enabled = std::env::var(PAGED_KV_ENV).ok();
    let env_tokens = std::env::var(KV_PAGE_SIZE_ENV).ok();
    let final_enabled = resolve_paged_kv(cli_enabled, env_enabled.as_deref());
    let final_tokens = resolve_paged_kv_page_tokens(cli_page_tokens, env_tokens.as_deref());

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

/// Returns `true` when paged KV is enabled (CLI > env > default false).
///
/// Reads the process-global override installed by `install_paged_kv`;
/// falls back to `RMLX_PAGED_KV=1` env for compat. Not cached — called only
/// from `KvStorage::new` (cold path), so re-resolving avoids OnceLock desync
/// when env is set after a pre-install read.
pub fn paged_kv_enabled() -> bool {
    if let Some(b) = PAGED_KV_CLI.get().copied() {
        return b;
    }
    std::env::var(PAGED_KV_ENV)
        .ok()
        .is_some_and(|v| v.trim() == "1")
}

/// Returns the page size in tokens. Not cached (see `paged_kv_enabled`).
pub fn paged_kv_page_tokens() -> i32 {
    if let Some(n) = PAGED_KV_PAGE_TOKENS_CLI.get().copied() {
        return n;
    }
    std::env::var(KV_PAGE_SIZE_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(DEFAULT_PAGE_TOKENS)
}
