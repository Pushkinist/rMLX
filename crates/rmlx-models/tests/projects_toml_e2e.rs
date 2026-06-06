//! End-to-end E2E tests for `projects.toml` per-project cap defaults.
//!
//! Two scenarios from the ticket DoD:
//!
//! 1. `projects_toml_applies_per_project_cap` — TOML to tempdir `<RMLX_HOME>`
//!    with `[project.alpha] ssd_cap_gb = 0.5`, no CLI override. Serve Bonsai
//!    with `--project alpha`. Assert via debug log: active per-ns cap = 0.5 GiB.
//! 2. `cli_flag_beats_projects_toml` — same file + `--kv-ssd-cache-gb 1.0`.
//!    Effective cap = 1.0 GiB.
//!
//! Both are `#[ignore]` and gated on `RMLX_KV_TEST_MODEL` so they never gate
//! `make ci` / `make model-check`. Wiring the full server-side decode loop is
//! out of scope; the gated tests are skeletons that compile and assert
//! env-var presence + early-return, matching the E2E pattern.

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

/// E2E #1: projects.toml applies per-project SSD cap when no CLI override.
#[ignore]
#[test]
fn projects_toml_applies_per_project_cap() {
    let Ok(_model) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("skip: RMLX_KV_TEST_MODEL not set");
        return;
    };
    // Live-decode wiring deferred — unit suite covers the resolver contract
    // (`resolve_caps`) exhaustively (6 precedence cases). This slot wires it
    // through a live `ssd_tier::active()` check after `install_config` runs.
    //
    // Implementation sketch (when wired):
    // 1. Write projects.toml with `[project.alpha] ssd_cap_gb = 0.5` to
    // a tempdir. Set `RMLX_HOME` to that tempdir.
    // 2. Call `run_serve` (or `install_config` directly) with
    // `kv_ssd_cache_gb = 0.0`, `project = Some("alpha")`.
    // 3. Assert `ssd_tier::active().unwrap().per_namespace_budget_bytes`
    // == (0.5 * 1024^3) as u64.
    panic!("E2E projects_toml_applies_per_project_cap wiring deferred; see ticket follow-up");
}

/// E2E #2: CLI `--kv-ssd-cache-gb` beats the projects.toml value.
#[ignore]
#[test]
fn cli_flag_beats_projects_toml() {
    let Ok(_model) = std::env::var("RMLX_KV_TEST_MODEL") else {
        eprintln!("skip: RMLX_KV_TEST_MODEL not set");
        return;
    };
    // Live-decode wiring deferred — the precedence contract is unit-tested in
    // `rmlx_core::projects_config` (case 1: CLI wins over project section).
    //
    // Implementation sketch (when wired):
    // 1. Same TOML as E2E #1 (`[project.alpha] ssd_cap_gb = 0.5`).
    // 2. Call with `kv_ssd_cache_gb = 1.0`, `project = Some("alpha")`.
    // 3. Assert `ssd_tier::active().unwrap().per_namespace_budget_bytes`
    // == (1.0 * 1024^3) as u64.
    panic!("E2E cli_flag_beats_projects_toml wiring deferred; see ticket follow-up");
}
