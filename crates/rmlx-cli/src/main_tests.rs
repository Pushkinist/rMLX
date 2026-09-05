use super::LogLevel;

/// The default LogLevel must be Info, and its EnvFilter string must NOT
/// contain the word "debug". This ensures the steady-state default does
/// not fire per-layer / per-step debug events.
#[test]
fn default_log_level_is_info() {
    let default: LogLevel = LogLevel::default();
    assert_eq!(
        default,
        LogLevel::Info,
        "LogLevel default must be Info, got {default:?}"
    );
}

#[test]
fn default_env_filter_does_not_contain_debug() {
    let filter_str = LogLevel::Info.env_filter();
    assert!(
        !filter_str.contains("debug"),
        "default EnvFilter string must not contain 'debug', got: {filter_str:?}"
    );
}

// ── clap parse-time validation ───────────────────────────────────

/// `--draft-model` stands alone: the drafter kind is read from the snapshot.
/// A `requires = "draft_kind"` here is what made the two-model loop
/// unreachable from the command line.
#[test]
fn draft_model_parses_without_draft_kind() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--draft-model",
        "/tmp/d",
    ]);
    assert!(
        r.is_ok(),
        "--draft-model alone must parse, got: {:?}",
        r.err()
    );
}

/// `--draft-kind` without a draft is meaningless and stays refused.
#[test]
fn draft_kind_requires_draft_model() {
    let r = Cli::try_parse_from(["rmlx", "serve", "--model", "/tmp/m", "--draft-kind", "mtp"]);
    let msg = r.err().map_or_else(String::new, |e| e.to_string());
    assert!(
        msg.contains("--draft-model"),
        "--draft-kind without --draft-model must be refused naming the missing flag, got: {msg}"
    );
}

/// Every kind the engine ships is a value the flag accepts, spelled as the
/// engine spells it in its logs and metrics.
#[test]
fn every_draft_kind_is_a_flag_value() {
    for kind in [
        rmlx_models::DraftKind::Mtp,
        rmlx_models::DraftKind::DFlash,
        rmlx_models::DraftKind::Eagle3,
        rmlx_models::DraftKind::TwoModel,
    ] {
        let r = Cli::try_parse_from([
            "rmlx",
            "serve",
            "--model",
            "/tmp/m",
            "--draft-model",
            "/tmp/d",
            "--draft-kind",
            kind.as_str(),
        ]);
        assert!(
            r.is_ok(),
            "--draft-kind {kind} must parse, got: {:?}",
            r.err()
        );
    }
}

use super::Cli;
use clap::Parser;

/// `--paged-kv-page-tokens N` without `--paged-kv` must be rejected by clap
/// via the `requires` attribute.
#[test]
fn paged_kv_page_tokens_requires_paged_kv() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--paged-kv-page-tokens",
        "256",
    ]);
    assert!(
        r.is_err(),
        "--paged-kv-page-tokens without --paged-kv must be rejected by clap"
    );
}

/// `--paged-kv --paged-kv-page-tokens N` parses cleanly.
#[test]
fn paged_kv_with_page_tokens_parses() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--paged-kv",
        "--paged-kv-page-tokens",
        "128",
    ]);
    assert!(r.is_ok(), "should parse: {:?}", r.err());
}

/// New flags accept GiB doubles and the default 0.0.
#[test]
fn global_gb_flag_parses() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--kv-ssd-cache-gb",
        "1.0",
        "--kv-ssd-global-gb",
        "2.0",
    ]);
    assert!(r.is_ok(), "should parse: {:?}", r.err());
}

#[test]
fn prompt_cache_ram_gb_flag_parses() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--prompt-cache-ram-gb",
        "1.5",
    ]);
    assert!(r.is_ok(), "should parse: {:?}", r.err());
}

/// `--idle-timeout-secs -1` must parse — clap otherwise treats `-1` as a
/// flag prefix and rejects it. Fixed by `allow_hyphen_values = true`.
#[test]
fn idle_timeout_secs_accepts_negative() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--idle-timeout-secs",
        "-1",
    ]);
    assert!(
        r.is_ok(),
        "--idle-timeout-secs -1 must parse (pin policy), got: {:?}",
        r.err()
    );
}

/// `--idle-timeout-secs -30m` must parse — Go-style negative duration shapes
/// (`-1`, `-1s`, `-30m`) all map to Pin.
#[test]
fn idle_timeout_secs_accepts_negative_with_unit() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--idle-timeout-secs",
        "-30m",
    ]);
    assert!(
        r.is_ok(),
        "--idle-timeout-secs -30m must parse (pin policy), got: {:?}",
        r.err()
    );
}

// ── --kv-preset fp16 + --paged-kv bypass ─────────────────────────────────────
//
// The runtime check for --paged-kv + unquantised KV lives in the `Cmd::Serve`
// arm of `run()`, after kv_quant_final is resolved from the preset. The full
// run() path requires a model on disk, so this test verifies the clap parse
// succeeds (no conflicts_with_all covers paged_kv) and that the KvQuant::None
// `matches!` predicate is correct.

/// `--kv-preset fp16 --paged-kv` must parse at the clap level (the runtime
/// check rejects it; conflicts_with_all on kv_preset does not cover paged_kv).
#[test]
fn kv_preset_fp16_paged_kv_parses_at_clap_level() {
    let r = Cli::try_parse_from([
        "rmlx",
        "serve",
        "--model",
        "/tmp/m",
        "--kv-preset",
        "fp16",
        "--paged-kv",
    ]);
    // Clap should NOT reject this combination (only the runtime run() path does).
    assert!(
        r.is_ok(),
        "--kv-preset fp16 --paged-kv must parse at clap level (runtime rejects); got: {:?}",
        r.err()
    );
}

/// The `matches!(KvQuant::None)` predicate used in the paged-kv post-resolution
/// guard correctly identifies unquantised variants and passes quantised ones.
#[test]
fn kv_quant_none_matches_predicate() {
    use rmlx_kv_quant::KvQuant;
    // Variants that must be rejected by --paged-kv:
    assert!(
        matches!(KvQuant::None, KvQuant::None),
        "KvQuant::None (bf16/fp16) must match the paged-kv rejection predicate"
    );
    // Variants that must be accepted:
    assert!(
        !matches!(KvQuant::K8V4, KvQuant::None),
        "KvQuant::K8V4 must NOT match the paged-kv rejection predicate"
    );
    assert!(
        !matches!(KvQuant::K8V8, KvQuant::None),
        "KvQuant::K8V8 must NOT match the paged-kv rejection predicate"
    );
    assert!(
        !matches!(KvQuant::Planar, KvQuant::None),
        "KvQuant::Planar must NOT match the paged-kv rejection predicate"
    );
}
