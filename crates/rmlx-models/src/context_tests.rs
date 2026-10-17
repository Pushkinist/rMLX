//! Tests for the single context-ceiling resolution.

#![allow(
    clippy::expect_used,
    reason = "unit-test scaffolding: a panic is how an assertion failure surfaces"
)]

use super::{rebind_to_ceiling, resolve_context, ContextLimits, ContextScaling, ScalingSource};
use rmlx_core::error::Error;
use rmlx_kv_quant::{KvCache, KvQuant, KV_MAX_SEQ_DEFAULT};
use rmlx_mlx::{Array, Device};

/// Bonsai-8B's shape: a 65 536 trained window declared as YaRN ×4 over 16 384.
fn bonsai_limits() -> ContextLimits {
    ContextLimits {
        trained_max: 65_536,
        scaling: Some(ContextScaling {
            factor: 4.0,
            original_max: 16_384.0,
            source: ScalingSource::Config,
        }),
        scaling_supported: true,
    }
}

/// Qwen3-8B's shape: a 40 960 trained window and no `rope_scaling`.
fn unscaled_qwen3_limits() -> ContextLimits {
    ContextLimits {
        trained_max: 40_960,
        scaling: None,
        scaling_supported: true,
    }
}

// ── Refusal ──────────────────────────────────────────────────────────────────

/// One token past the trained window is refused, not clamped.
#[test]
fn one_past_the_trained_window_is_refused() {
    let err = resolve_context(&unscaled_qwen3_limits(), Some(40_961))
        .expect_err("a request past the positional capacity must be refused");
    let Error::ContextCeilingExceeded {
        requested,
        positional_max,
        trained_max,
        ..
    } = err
    else {
        panic!("expected ContextCeilingExceeded, got: {err}");
    };
    assert_eq!(requested, 40_961);
    assert_eq!(positional_max, 40_960);
    assert_eq!(trained_max, 40_960);
}

/// The refusal names both numbers and the flag that would lift it — the
/// operator must not have to read the source to learn the cause.
#[test]
fn refusal_names_both_numbers_and_the_lift() {
    let err = resolve_context(&unscaled_qwen3_limits(), Some(131_072)).expect_err("refused");
    let msg = err.to_string();
    assert!(msg.contains("131072"), "requested value missing: {msg}");
    assert!(msg.contains("40960"), "capacity missing: {msg}");
    assert!(
        msg.contains("max_position_embeddings"),
        "cause missing: {msg}"
    );
    assert!(msg.contains("--yarn-factor"), "lift missing: {msg}");
}

/// On an architecture with no RoPE-scaling implementation the refusal says so
/// instead of naming a flag that would do nothing.
#[test]
fn refusal_is_honest_when_the_arch_cannot_scale() {
    let err =
        resolve_context(&ContextLimits::trained_only(131_072), Some(262_144)).expect_err("refused");
    let msg = err.to_string();
    assert!(
        msg.contains("no RoPE-scaling support"),
        "expected the arch-limitation wording, got: {msg}"
    );
    assert!(
        !msg.contains("--yarn-factor"),
        "must not offer a flag this arch ignores: {msg}"
    );
}

/// The refusal wording is architecture-agnostic: it is keyed off the numbers
/// and the scaling mechanism, never off a model or architecture name.
#[test]
fn refusal_wording_is_arch_agnostic() {
    let msgs = [
        resolve_context(&unscaled_qwen3_limits(), Some(131_072)),
        resolve_context(&ContextLimits::trained_only(131_072), Some(262_144)),
        resolve_context(&bonsai_limits(), Some(262_144)),
    ]
    .map(|r| r.expect_err("refused").to_string().to_lowercase());
    for msg in &msgs {
        for name in ["qwen", "gemma", "bonsai", "llama", "mistral"] {
            assert!(!msg.contains(name), "arch-specific wording in: {msg}");
        }
    }
}

// ── Lifting ──────────────────────────────────────────────────────────────────

/// A checkpoint-declared scaling sets the capacity; a request inside it is
/// served without a flag.
#[test]
fn declared_scaling_sets_the_capacity() {
    let limits = ContextLimits {
        trained_max: 40_960,
        scaling: Some(ContextScaling {
            factor: 4.0,
            original_max: 40_960.0,
            source: ScalingSource::Config,
        }),
        scaling_supported: true,
    };
    let ctx = resolve_context(&limits, Some(131_072)).expect("declared scaling covers 131072");
    assert_eq!(ctx.positional_max, 163_840);
    assert_eq!(ctx.ceiling, 131_072);
}

/// An operator-requested scaling lifts the ceiling past the checkpoint's own
/// declared window — the case the flag exists for.
#[test]
fn operator_scaling_lifts_past_the_declared_window() {
    let mut limits = bonsai_limits();
    assert!(
        resolve_context(&limits, Some(131_072)).is_err(),
        "without the flag the declared 65536 window stands"
    );
    limits.scaling = Some(ContextScaling {
        factor: 8.0,
        original_max: 16_384.0,
        source: ScalingSource::Operator,
    });
    let ctx = resolve_context(&limits, Some(131_072)).expect("--yarn-factor 8 lifts to 131072");
    assert_eq!(ctx.positional_max, 131_072);
    assert_eq!(ctx.ceiling, 131_072);
    assert_eq!(ctx.initial_max_seq, KV_MAX_SEQ_DEFAULT);
}

/// A scaling that reaches less than the trained window never lowers it.
#[test]
fn scaling_never_lowers_the_trained_window() {
    let limits = ContextLimits {
        trained_max: 65_536,
        scaling: Some(ContextScaling {
            factor: 2.0,
            original_max: 16_384.0,
            source: ScalingSource::Operator,
        }),
        scaling_supported: true,
    };
    assert_eq!(limits.positional_max(), 65_536);
}

/// A factor at or below 1.0 extends nothing.
#[test]
fn non_extending_factor_adds_no_capacity() {
    for factor in [0.5_f32, 1.0] {
        let s = ContextScaling {
            factor,
            original_max: 16_384.0,
            source: ScalingSource::Operator,
        };
        assert_eq!(s.extended_max(), 0, "factor {factor} must not extend");
    }
}

// ── Ceiling / ring sizing (the pre-existing lazy-grow policy) ────────────────

/// A large `--max-ctx` becomes a ceiling, not the initial ring size.
#[test]
fn large_override_starts_the_ring_lazily() {
    let ctx = resolve_context(&ContextLimits::trained_only(262_144), Some(140_000))
        .expect("under capacity");
    assert_eq!(ctx.initial_max_seq, KV_MAX_SEQ_DEFAULT);
    assert_eq!(ctx.ceiling, 140_000);
}

/// A sub-default ceiling also caps the initial ring — never pre-grow past it.
#[test]
fn sub_default_ceiling_caps_the_initial_ring() {
    let ctx =
        resolve_context(&ContextLimits::trained_only(131_072), Some(2048)).expect("under capacity");
    assert_eq!(ctx.initial_max_seq, 2048);
    assert_eq!(ctx.ceiling, 2048);
}

/// No override: the ceiling is `min(capacity, KV_MAX_SEQ_DEFAULT)`.
#[test]
fn no_override_uses_the_capacity_default_chain() {
    let ctx = resolve_context(&ContextLimits::trained_only(131_072), None).expect("no override");
    assert_eq!(ctx.ceiling, KV_MAX_SEQ_DEFAULT);
    let ctx = resolve_context(&ContextLimits::trained_only(2048), None).expect("no override");
    assert_eq!(ctx.ceiling, 2048);
    assert_eq!(ctx.initial_max_seq, 2048);
}

/// An architecture that does not expose `max_position_embeddings` reports an
/// unknown capacity, and an override then stands alone.
#[test]
fn unknown_capacity_accepts_any_override() {
    let ctx = resolve_context(&ContextLimits::trained_only(0), Some(64_000))
        .expect("no capacity to check against");
    assert_eq!(ctx.positional_max, 0);
    assert_eq!(ctx.ceiling, 64_000);
    let ctx = resolve_context(&ContextLimits::trained_only(0), None).expect("no override");
    assert_eq!(ctx.ceiling, KV_MAX_SEQ_DEFAULT);
}

/// A non-positive override is treated as unset, not as a zero-token ceiling.
#[test]
fn non_positive_override_falls_back_to_the_default_chain() {
    for n in [0, -1] {
        let ctx = resolve_context(&ContextLimits::trained_only(131_072), Some(n))
            .expect("treated as unset");
        assert_eq!(ctx.ceiling, KV_MAX_SEQ_DEFAULT, "override {n}");
    }
}

/// A scaling wide enough to overflow `i32` saturates instead of wrapping into
/// a negative capacity that would refuse every request.
#[test]
fn absurd_scaling_saturates() {
    let s = ContextScaling {
        factor: 1e9,
        original_max: 1e9,
        source: ScalingSource::Operator,
    };
    assert_eq!(s.extended_max(), i32::MAX);
}

// ── One resolution, enumerated ───────────────────────────────────────────────

/// Every place a context ceiling is **resolved**, by workspace-relative path.
///
/// Not every place one is applied — a bound is read far more widely than it is
/// decided, and this list is the deciding. Adding a call site is fine; leaving
/// it off is not. The list is what makes "one resolution" checkable: a reviewer
/// reads it and sees every place a context bound is arrived at.
const CEILING_CONSUMERS: &[&str] = &[
    // KV ring sizing, one per architecture generate path.
    "crates/rmlx-models/src/gemma4/generate/mod.rs",
    "crates/rmlx-models/src/qwen3.rs",
    "crates/rmlx-models/src/qwen3_5_moe/generate.rs",
    "crates/rmlx-models/src/qwen3_vl_moe/generate.rs",
    // Server: the load-time ceiling behind the admission guard.
    "crates/rmlx-server/src/engine/arch_generator.rs",
    "crates/rmlx-server/src/engine/speculative.rs",
    // Server: a per-request `max_ctx` override.
    "crates/rmlx-server/src/openai/chat.rs",
    // CLI: the default `--max-prompt-tokens` cap.
    "crates/rmlx-cli/src/commands/baseline.rs",
    "crates/rmlx-cli/src/commands/bench.rs",
    // Speculative: the verifier's limits bound the pair, and every round loop
    // resolves them here, with the cache stack they size.
    "crates/rmlx-models/src/speculative/round_common.rs",
];

/// The only places allowed to name `max_position_embeddings` at all — the
/// field, the accessor, or a message that quotes it. Everything else must go
/// through [`resolve_context`], because a second `min(mpe, …)` chain is exactly
/// the drift this module removed.
///
/// Scanning the bare identifier rather than the `max_position_embeddings()`
/// accessor form is deliberate: the field read
/// (`model.cfg.max_position_embeddings`) was the dominant shape, so an
/// accessor-only scan let a hand-rolled clamp into any generate path
/// unnoticed. Every architecture now folds its raw field into a
/// [`ContextLimits`] inside its config parser, and the generate paths read
/// that.
///
/// Three kinds of entry belong on this list and nothing else:
/// * **config parsers and the arch accessor** — where the raw field is read
///   once and folded into [`ContextLimits`];
/// * **`rmlx info`** — it prints the config field verbatim;
/// * **message text** — `error.rs` and `arch/loader.rs` quote the field name
///   to the operator. A string is not a formula.
///
/// YaRN's `original_max_position_embeddings` is a different quantity and is
/// masked out of the scan, so `rope.rs` and the drafter RoPE code stay off it.
const RAW_MPE_READERS: &[&str] = &[
    // Config parsers: the one fold from raw field to ContextLimits.
    "crates/rmlx-loader/src/config.rs",
    "crates/rmlx-models/src/arch/mod.rs",
    "crates/rmlx-models/src/bitnet/config.rs",
    "crates/rmlx-models/src/gemma4/config.rs",
    "crates/rmlx-models/src/jina_v4/config.rs",
    "crates/rmlx-models/src/qwen3.rs",
    "crates/rmlx-models/src/qwen3_5_moe/config.rs",
    "crates/rmlx-models/src/qwen3_vl_moe/config.rs",
    "crates/rmlx-models/src/qwen3_vl_moe/loader.rs",
    "crates/rmlx-models/src/qwen3_vl_moe/model.rs",
    // Operator-facing text that quotes the field name.
    "crates/rmlx-core/src/error.rs",
    "crates/rmlx-models/src/arch/loader.rs",
    // `rmlx info` prints the config field verbatim.
    "crates/rmlx-cli/src/commands/info.rs",
];

/// Workspace root, from this crate's manifest directory.
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root resolves from CARGO_MANIFEST_DIR")
}

/// Is `path` test source? Test files are excluded from every scan: a fixture
/// naming a field is not a production consumer of it.
fn is_test_source(path: &std::path::Path) -> bool {
    let name = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or("");
    name.ends_with("_tests.rs")
        || name == "tests.rs"
        || path.components().any(|c| c.as_os_str() == "tests")
}

/// Workspace-relative paths of every `.rs` file under `crates/` containing
/// `needle`, excluding this module's own source and every test source.
///
/// `original_max_position_embeddings` is masked out before the search so a
/// scan for `max_position_embeddings` does not collect YaRN's anchor field,
/// which is a different quantity.
fn files_containing(needle: &str) -> Vec<String> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let root = workspace_root();
    let mut files = Vec::new();
    walk(&root.join("crates"), &mut files);
    let mut hits: Vec<String> = files
        .iter()
        .filter(|p| {
            let name = p
                .file_name()
                .and_then(std::ffi::OsStr::to_str)
                .unwrap_or("");
            name != "context.rs" && name != "context_tests.rs" && !is_test_source(p)
        })
        .filter(|p| {
            std::fs::read_to_string(p).is_ok_and(|body| {
                body.lines()
                    .map(|l| l.replace("original_max_position_embeddings", ""))
                    .any(|l| l.contains(needle) && !l.trim_start().starts_with("//"))
            })
        })
        .filter_map(|p| p.strip_prefix(&root).ok())
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .collect();
    hits.sort();
    hits.dedup();
    hits
}

/// Every place that resolves a context ceiling is on the enumerated list, and
/// every entry on the list still resolves one.
#[test]
fn ceiling_consumers_are_enumerated() {
    let mut found = files_containing("resolve_context(");
    found.sort();
    found.dedup();
    let mut expected: Vec<String> = CEILING_CONSUMERS.iter().map(|s| (*s).to_owned()).collect();
    expected.sort();
    assert_eq!(
        found, expected,
        "the set of context-ceiling consumers changed; add the new call site to \
         CEILING_CONSUMERS (or drop the stale entry) so 'one resolution' stays checkable"
    );
}

/// No second ceiling formula: `max_position_embeddings` is named only where a
/// context bound is not what is being computed.
#[test]
fn raw_positional_limit_has_no_unlisted_readers() {
    let found = files_containing("max_position_embeddings");
    let mut expected: Vec<String> = RAW_MPE_READERS.iter().map(|s| (*s).to_owned()).collect();
    expected.sort();
    assert_eq!(
        found, expected,
        "a new reader of max_position_embeddings appeared; fold it into a ContextLimits in \
         the architecture's config parser and route context bounds through resolve_context \
         instead of clamping to the raw field by hand"
    );
}

// ── A reused prefix under another request's ceiling ──────────────────────────

/// A cache holding `n` positions, filled under `ceiling`.
#[allow(
    clippy::unwrap_used,
    reason = "unit-test fixture: the CPU update path is infallible on a correctly-shaped array"
)]
fn cache_filled_under(ceiling: i32, n: i32) -> KvCache {
    let data: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
    let kv = Array::from_f32_slice(&data, &[1, 1, n, 2]).unwrap();
    let mut cache = KvCache::with_quant_max_seq(KvQuant::None, 1024).with_max_seq_ceiling(ceiling);
    cache.update(&kv, &kv, Device::Cpu).unwrap();
    cache
}

/// `n` more positions appended, as a resumed tail or a decode step does.
#[allow(
    clippy::unwrap_used,
    reason = "unit-test fixture: the CPU update path is infallible on a correctly-shaped array"
)]
fn append(cache: &mut KvCache, n: i32) -> rmlx_core::error::Result<()> {
    let data: Vec<f32> = (0..n * 2).map(|i| i as f32).collect();
    let kv = Array::from_f32_slice(&data, &[1, 1, n, 2]).unwrap();
    cache.update(&kv, &kv, Device::Cpu).map(|_| ())
}

/// A request under a higher ceiling than the stored one is not held to the stored
/// one: its tail grows past it, as a cold run under its own ceiling would.
#[test]
fn a_reused_prefix_grows_past_the_ceiling_it_was_stored_under() {
    let mut stored = cache_filled_under(500, 400);
    assert!(
        append(&mut stored, 150).is_err(),
        "control: the stored ceiling refuses 550 positions"
    );

    let mut rebound = rebind_to_ceiling(vec![cache_filled_under(500, 400)], 4096, 400)
        .expect("a prompt under the new ceiling");
    let cache = rebound.first_mut().expect("one cache");
    assert!(
        append(cache, 150).is_ok(),
        "the request's own, higher ceiling lets the tail grow"
    );
}

/// A request under a lower ceiling than the stored one is refused when its
/// prompt is above it, as a cold prefill of that prompt is, and is held to it
/// when the prompt is under it.
#[test]
fn a_reused_prefix_obeys_a_lower_ceiling_than_the_one_it_was_stored_under() {
    let refused = rebind_to_ceiling(vec![cache_filled_under(4096, 400)], 300, 400);
    assert!(
        matches!(
            refused,
            Err(Error::KvCeilingExceeded {
                requested: 400,
                ceiling: 300
            })
        ),
        "a prompt above the ceiling is refused with the typed ceiling error"
    );

    let mut rebound = rebind_to_ceiling(vec![cache_filled_under(4096, 400)], 450, 400)
        .expect("a prompt under the new ceiling");
    let cache = rebound.first_mut().expect("one cache");
    assert!(
        append(cache, 100).is_err(),
        "the request's own, lower ceiling stops its growth"
    );
}

/// A cache of codec `quant` with `max_seq` positions of capacity, filled to `n`
/// under `ceiling`, optionally a sliding window of `window` positions.
#[allow(
    clippy::unwrap_used,
    reason = "unit-test fixture: the CPU update path is infallible on a correctly-shaped array"
)]
fn grown_cache(quant: KvQuant, max_seq: i32, ceiling: i32, n: i32, window: Option<i32>) -> KvCache {
    let data: Vec<f32> = (0..n * 128).map(|i| (i % 97) as f32).collect();
    let kv = Array::from_f32_slice(&data, &[1, 1, n, 128]).unwrap();
    let mut cache =
        KvCache::with_quant_max_seq_window(quant, max_seq, window).with_max_seq_ceiling(ceiling);
    cache.update(&kv, &kv, Device::Cpu).unwrap();
    cache
}

#[allow(
    clippy::unwrap_used,
    reason = "unit-test fixture: the CPU update path is infallible on a correctly-shaped array"
)]
fn append_wide(cache: &mut KvCache, n: i32) -> rmlx_core::error::Result<()> {
    let data: Vec<f32> = (0..n * 128).map(|i| (i % 89) as f32).collect();
    let kv = Array::from_f32_slice(&data, &[1, 1, n, 128]).unwrap();
    cache.update(&kv, &kv, Device::Cpu).map(|_| ())
}

/// The capacity, not only the ceiling, follows the new request: a cache filled to
/// 400 of a 500-position capacity under a 500 ceiling, rebound to 4096, takes
/// 200 more positions. Stored under each codec family and as a sliding window.
#[test]
fn a_rebound_cache_grows_its_capacity_past_the_stored_ceiling() {
    for (label, quant, window) in [
        ("none", KvQuant::None, None),
        ("k8v8", KvQuant::K8V8, None),
        ("none window", KvQuant::None, Some(128)),
    ] {
        let mut stored = grown_cache(quant, 500, 500, 400, window);
        let stored_refuses = append_wide(&mut stored, 200).is_err();
        assert!(
            window.is_some() || stored_refuses,
            "{label}: control: the stored ceiling refuses 600 positions"
        );

        let mut rebound =
            rebind_to_ceiling(vec![grown_cache(quant, 500, 500, 400, window)], 4096, 400)
                .expect("a prompt under the new ceiling");
        let cache = rebound.first_mut().expect("one cache");
        assert_eq!(
            cache.offset(),
            400,
            "{label}: rebinding leaves the content alone"
        );
        assert!(
            append_wide(cache, 200).is_ok(),
            "{label}: the request's higher ceiling lets the capacity grow past 500"
        );
        assert_eq!(cache.offset(), 600, "{label}");
        // A window holds `window` positions however long the sequence runs.
        assert!(
            window.is_some() || cache.max_seq() >= 600,
            "{label}: capacity {} must cover the 600 positions held",
            cache.max_seq()
        );
    }
}
