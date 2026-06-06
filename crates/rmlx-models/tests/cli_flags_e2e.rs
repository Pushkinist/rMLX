//! End-to-end E2E tests for the new CLI surface.
//!
//! Three scenarios from the ticket DoD:
//!
//! 1. `ssd_global_ceiling_evicts_across_namespaces` — two serves of the same
//!    model under `--project alpha` and `--project beta` with a shared global
//!    pool ceiling; the second `attach_at_load` must evict ≥ 1 row from alpha
//!    while beta's just-loaded cache still hits on retry.
//! 2. `ram_cap_cli_overrides_env` — env=512 MiB, CLI=1.0 GiB; assert effective
//!    cap = 1 GiB via `PromptCache::stats().bytes`.
//! 3. `paged_kv_cli_flag_routes_correctly` — `--paged-kv --kv-quant k8v4`
//!    routes through `KvStorage::Paged`; without `--paged-kv` it routes
//!    through the contiguous path. Token parity ±1 same seed.
//!
//! All three are `#[ignore]` and gated on `RMLX_KV_TEST_MODEL` so they never
//! gate `make ci` / `make model-check`. Wiring the full server-side decode
//! loop is out of scope; the gated tests are skeletons that compile
//! and assert env-var presence + early-return, matching the E2E
//! pattern.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

/// E2E #1: cross-namespace SSD eviction at the global ceiling.
#[ignore]
#[test]
fn ssd_global_ceiling_evicts_across_namespaces() {
    let Ok(_model) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("skip: RMLX_KV_TEST_MODEL not set");
        return;
    };
    // Live-decode wiring deferred — see ticket follow-up. Unit suite covers
    // every invariant (cross-namespace LRU oldest-first ordering, .kvb +
    // index-row deletion, namespaces_touched count).
    panic!("E2E ssd_global_ceiling wiring deferred; see ticket follow-up");
}

/// E2E #2: CLI `--prompt-cache-ram-gb` overrides env.
#[ignore]
#[test]
fn ram_cap_cli_overrides_env() {
    let Ok(_model) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("skip: RMLX_KV_TEST_MODEL not set");
        return;
    };
    // Live-decode wiring deferred — see ticket follow-up. The resolver
    // contract (`resolve_ram_cap_bytes`) is unit-tested directly in
    // `crates/rmlx-models/src/prompt_cache.rs`; this slot wires it through
    // the live `PromptCache::stats().bytes` check.
    panic!("E2E ram_cap_cli_overrides_env wiring deferred; see ticket follow-up");
}

/// E2E #3: `--paged-kv` flag routes to `KvStorage::Paged`.
#[ignore]
#[test]
fn paged_kv_cli_flag_routes_correctly() {
    let Ok(_model) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("skip: RMLX_KV_TEST_MODEL not set");
        return;
    };
    // Live-decode wiring deferred — see ticket follow-up. The resolver
    // contract (`resolve_paged_kv`) is unit-tested directly in
    // `crates/rmlx-models/src/kv_cache/paged.rs`; this slot wires it through
    // a live `storage = "paged"` vs `"contiguous"` tracing assertion + token
    // parity check.
    panic!("E2E paged_kv_cli_flag wiring deferred; see ticket follow-up");
}
