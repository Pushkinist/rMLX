// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Shared CLI flag-parsing helpers and Metal claim acquisition.
//!
//! Centralises the parsing logic used by multiple subcommands (`serve`,
//! `info`, `baseline`, `eval`) so the same string → type conversions and
//! claim-file semantics are applied consistently.
//!
//! # Public API
//!
//! - [`parse_device`] — `"cpu"` / `"gpu"` string → [`rmlx_mlx::Device`].
//! - [`acquire_claim_for_device`] — acquire the single-MLX-process claim
//!   file before any MLX call; aborts with a clear message on contention.
//! - [`parse_kv_quant`] — `--kv-quant` string → `Option<KvQuant>`.
//! - [`parse_kv_preset`] — `--kv-preset` name → [`KvPresetArg`] via the
//!   static preset table. `"auto"` yields `KvPresetArg::Auto`; unknown names
//!   return a clap `InvalidValue` error with an available-names hint.
//! - [`KvPresetArg`] — parsed `--kv-preset` value: either a resolved
//!   [`KvQuant`] or the sentinel `Auto` variant.
//! - [`resolve_preset_arg`] — convert a `KvPresetArg` to a `KvQuant`;
//!   `Auto` is `DEFAULT_KV_QUANT`.
//! - [`parse_cache_type`] — `--cache-type` string → [`CacheType`].
//! - [`build_cache_type_spec`] — combine `--ctk` / `--ctv` / `--kv-quant`
//!   / `--cache-type` aliases into a resolved `CacheType`.
//! - [`resolve_kv_quant`] — apply model-side KV-quant capability caps to
//!   the user-requested quant, falling back gracefully.
//! - [`resolve_model_flags`] — full flag resolution for a loaded model.
//! - [`parse_max_ctx`] — `Option<u32>` → `Option<i32>` with overflow guard.
//! - [`parse_kv_bits_fractional`] / [`parse_kv_bits_combo`] — fractional
//!   and combo KV-bit-width string parsers used by baseline / eval.

#![allow(clippy::cognitive_complexity)]
use rmlx_mlx::Device;
use rmlx_server::{try_claim, ClaimError};
use tracing::error;

use crate::commands::preset_table::{lookup_preset, PresetError, AVAILABLE_NAMES};

/// Parse the `--device` flag value into a `Device`.
pub(crate) fn parse_device(s: &str) -> anyhow::Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "gpu" => Ok(Device::Gpu),
        other => Err(anyhow::anyhow!(
            "--device must be 'cpu' or 'gpu', got '{other}'"
        )),
    }
}

/// Acquire the Metal claim file for `device` and `port`.
///
/// - `Device::Gpu` → calls `try_claim(port)`. On conflict, logs the error and
///   exits with code 11 (CLAUDE.md mandate).
/// - `Device::Cpu` → no-op (returns `None`).
pub(crate) fn acquire_claim_for_device(
    device: Device,
    port: u16,
) -> anyhow::Result<Option<rmlx_server::MetalClaim>> {
    if device == Device::Cpu {
        return Ok(None);
    }
    match try_claim(port) {
        Ok(claim) => Ok(Some(claim)),
        Err(ClaimError::AlreadyHeld {
            holder_pid,
            port: p,
        }) => {
            error!(
                holder_pid,
                port = p,
                "Metal claim held by another rMLX process — refusing to start"
            );
            eprintln!(
                "error: another rMLX process (PID {holder_pid}) holds the Metal claim for port {p}.\n\
                 Hint: stop it with `kill {holder_pid}` or via the /v1/models/<id>/unload API.\n\
                 rMLX exits with code 11."
            );
            std::process::exit(11);
        }
        Err(e) => {
            error!(
                error = %e,
                port,
                "D-class startup: Metal claim I/O error — cannot acquire GPU lock"
            );
            Err(anyhow::anyhow!("Metal claim: {e}"))
        }
    }
}

/// Parse the `--kv-quant` flag value into an optional `KvQuant` override.
///
/// `"auto"` → `None` — caller resolves it via [`resolve_kv_quant`] once the
/// model's `config.json` is loaded.
/// `"mixed"` → `Some(KvQuant::Mixed { k_bits:8, v_bits:4, k_group_size:64, v_group_size:64 })`
/// (mixed-precision quantized SDPA path; K=8-bit / V=4-bit affine, group=64).
/// Backwards-compatible short alias for the canonical `mixed_k8g64_v4g64` form.
///
/// Every other value is delegated to `<KvQuant as FromStr>::from_str`, which
/// accepts the canonical strings `none` / `k8v4` / `k8v8` / `planar`, the
/// aliases `bf16` / `f16` for `none`, and the long-form `mixed_k<kb>g<kg>_v<vb>g<vg>`.
pub(crate) fn parse_kv_quant(s: &str) -> anyhow::Result<Option<rmlx_kv_quant::KvQuant>> {
    use rmlx_kv_quant::KvQuant;
    use std::str::FromStr;
    if s == "auto" {
        return Ok(None);
    }
    if s == "mixed" {
        return Ok(Some(KvQuant::Mixed {
            k_bits: 8,
            v_bits: 4,
            k_group_size: 64,
            v_group_size: 64,
        }));
    }
    KvQuant::from_str(s)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("--kv-quant: {e}"))
}

/// Parsed value for the `--kv-preset` flag.
///
/// A clap `value_parser` must return a concrete type.  `KvPresetArg` carries
/// either a fully-resolved `KvQuant` (for named presets) or the `Auto`
/// sentinel (for `--kv-preset auto`), which [`resolve_preset_arg`] turns into
/// `DEFAULT_KV_QUANT`.
#[derive(Debug, Clone, Copy)]
pub(crate) enum KvPresetArg {
    /// A named preset that was resolved at parse time.
    Resolved(rmlx_kv_quant::KvQuant),
    /// `--kv-preset auto` — resolved to `DEFAULT_KV_QUANT`.
    Auto,
}

/// Parse the `--kv-preset` flag value into a [`KvPresetArg`].
///
/// This is a clap `value_parser`-compatible function: on success it returns a
/// `KvPresetArg`; on failure it returns a `String` error so clap wraps it as
/// an `InvalidValue` usage error with an available-names hint.
///
/// - `"auto"` → `Ok(KvPresetArg::Auto)`.
/// - Any named preset → `Ok(KvPresetArg::Resolved(kv_quant))`.
/// - Unknown names → `Err(...)` with the `AVAILABLE_NAMES` hint.
///
/// The clap `conflicts_with_all` on `kv_preset` ensures that `--kv-quant`,
/// `--cache-type-k`, `--cache-type-v`, and `--kv-bits` are mutually exclusive
/// with `--kv-preset` at parse time — no explicit runtime check needed here.
pub(crate) fn parse_kv_preset(name: &str) -> Result<KvPresetArg, String> {
    if name == "auto" {
        return Ok(KvPresetArg::Auto);
    }
    match lookup_preset(name) {
        Ok(spec) => Ok(KvPresetArg::Resolved(spec.kv_quant)),
        Err(PresetError::Reserved) => {
            // lookup_preset still returns Reserved for "auto", but we handle
            // "auto" above — this branch is unreachable in practice.
            Ok(KvPresetArg::Auto)
        }
        Err(PresetError::Unknown) => Err(format!(
            "unknown kv-preset '{name}'; available: auto, {AVAILABLE_NAMES}"
        )),
    }
}

/// Resolve a [`KvPresetArg`] to a concrete `KvQuant`.
///
/// For `KvPresetArg::Resolved` this is a trivial unwrap.
///
/// `KvPresetArg::Auto` is [`rmlx_models::kv_cache::DEFAULT_KV_QUANT`] — the
/// same constant `--kv-quant auto` resolves to, read from the same place. Two
/// auto surfaces that resolve independently are two defaults that can disagree,
/// and this one did: it ran a unified-memory decision tree and returned a
/// "compressing" preset under pressure, while every preset it could return
/// holds resident KV byte-identical to bf16. It answered a memory question with
/// something that has no memory effect.
pub(crate) fn resolve_preset_arg(arg: KvPresetArg) -> rmlx_kv_quant::KvQuant {
    match arg {
        KvPresetArg::Resolved(kq) => kq,
        KvPresetArg::Auto => rmlx_models::kv_cache::DEFAULT_KV_QUANT,
    }
}

/// Parse a `--cache-type-k` / `--cache-type-v` tag string into a [`CacheType`].
///
/// Accepts all canonical tags from §D1 plus documented aliases (`f16`, `none`,
/// `turbo4`). Returns an error for unknown tags or llama.cpp legacy block-32
/// codecs that rMLX does not implement.
pub(crate) fn parse_cache_type(s: &str) -> anyhow::Result<rmlx_models::kv_cache::CacheType> {
    rmlx_models::kv_cache::parse_cache_type_str(s).map_err(anyhow::Error::from)
}

/// Build an optional [`CacheTypeSpec`] from raw `--cache-type-k` / `--cache-type-v`
/// flag values.
///
/// - Both `None` → `Ok(None)` (no per-side override; auto-resolver runs).
/// - Either `Some(...)` → `Ok(Some(CacheTypeSpec { k, v }))` with
///   [`CacheType::Auto`] substituted for the absent side.
pub(crate) fn build_cache_type_spec(
    ctk: Option<&str>,
    ctv: Option<&str>,
) -> anyhow::Result<Option<rmlx_models::kv_cache::CacheTypeSpec>> {
    use rmlx_models::kv_cache::{CacheType, CacheTypeSpec};
    if ctk.is_none() && ctv.is_none() {
        return Ok(None);
    }
    let k = match ctk {
        Some(s) => parse_cache_type(s)?,
        None => CacheType::Auto,
    };
    let v = match ctv {
        Some(s) => parse_cache_type(s)?,
        None => CacheType::Auto,
    };
    Ok(Some(CacheTypeSpec { k, v }))
}

/// Refuse `--paged-kv` when the resolved KV codec keeps no packed store.
///
/// Paged KV is a layout for a codec's packed store — a block table over
/// quantised pages. `KvQuant::None` has no store to page, so the combination
/// is not a slower mode, it is meaningless.
///
/// The bar is deliberately the same one this check has always used —
/// "unquantised" — and not `materialises_packed_store()`. The mirror-family
/// codecs (`K8V8`, `K8V4`, `Planar*`, …) no longer build a store either, so the
/// stricter predicate would newly refuse `--kv-quant k8v8 --paged-kv`, which is
/// a separate change with its own blast radius. Widening this belongs with the
/// work that makes paged KV do anything at all.
///
/// This fires under `--kv-quant auto` now that auto is unquantised bf16, and
/// that is deliberate. The alternative — promoting `auto` to a quantised codec
/// because some *other* flag was passed — would be a second codec resolver
/// keyed on something that is not the codec flag, invisible in `--kv-quant`'s
/// own surface. One flag, named by the operator, is the honest form.
///
/// Returns `Some(message)` when the invocation must be refused. The message
/// names the resolved codec and what to pass, because "requires K8V4 / K8V8 /
/// Planar" alone does not tell an operator who passed no codec flag at all why
/// they are being refused.
pub(crate) fn reject_paged_kv_without_store(
    paged_kv: bool,
    resolved: Option<rmlx_kv_quant::KvQuant>,
) -> Option<String> {
    if !paged_kv {
        return None;
    }
    let kq = resolved.unwrap_or(rmlx_models::kv_cache::DEFAULT_KV_QUANT);
    if !matches!(kq, rmlx_kv_quant::KvQuant::None) {
        return None;
    }
    Some(format!(
        "--paged-kv needs a KV codec that keeps a packed store to page; \
         the resolved codec is `{kq}`, which keeps none. \
         `--kv-quant auto` is unquantised bf16, so --paged-kv must be given a \
         quantised codec explicitly: try `--kv-quant k8v8 --paged-kv`."
    ))
}

/// Resolve the final [`KvQuant`] from the parsed `--kv-quant` and
/// `--cache-type-{k,v}` flag values plus the loaded [`rmlx_loader::ModelConfig`].
///
/// Between `load_config` and `load_model`, every command runs the resolver and
/// fails-fast with `EX_CONFIG` (exit 78) on `ResolveError`.
///
/// Branch logic on the override pair:
/// - `(Some(kq), None)` → preset override wins.
/// - `(None, Some(spec))` → resolve the per-side spec. A side the operator left
///   `auto` takes the named side's canonical partner, not the engine default —
///   see [`rmlx_models::kv_cache::resolve_cache_type`]. On `Err`, log + hint +
///   `exit(78)`.
/// - `(None, None)` → [`rmlx_models::kv_cache::DEFAULT_KV_QUANT`].
/// - `(Some(_), Some(_))` → defense-in-depth (clap should have rejected this);
///   log + hint + `exit(78)`. No panic.
///
/// On success: emits a `tracing::info!` with `arch`, `head_dim`, resolved
/// `KvQuant`. If the resolved quant is non-`None` AND the arch is Gemma3 or
/// Gemma4, emits an additional one-shot `info!` disclosing the SWA-stays-bf16
/// rule (§D6.7).
///
/// fractional `--kv-bits` values (e.g. `3.5`) dispatch via
/// [`parse_kv_bits_fractional`] → floor K / ceil V before the arch resolver
/// runs. The `QwenMoeKBitsTooLow` guard fires at resolve time when the floor
/// K is < 8 on a Qwen MoE model.
pub(crate) fn resolve_kv_quant(
    model_cfg: &rmlx_loader::ModelConfig,
    kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    cts_override: Option<rmlx_models::kv_cache::CacheTypeSpec>,
) -> rmlx_kv_quant::KvQuant {
    use rmlx_kv_quant::KvQuant;
    use rmlx_models::kv_cache::{
        resolve_cache_type, validate_resolved_kv_quant, ResolverContext, DEFAULT_KV_QUANT,
    };

    let arch_class = model_cfg
        .architectures
        .first()
        .map_or("(empty)", String::as_str);
    let head_dim_opt = model_cfg.head_dim();
    let auto = DEFAULT_KV_QUANT;

    let final_kv_quant = match (kv_quant_override, cts_override) {
        (Some(kq), None) => {
            // Preset path: run the post-resolve arch invariants even though
            // the user bypassed the cache-type spec resolver. This catches
            // e.g. `--kv-quant mixed_k8g128_v4g64` on a Gemma4 model at
            // startup (exit 78) instead of crashing at first prefill.
            if let Err(e) = validate_resolved_kv_quant(arch_class, &kq) {
                tracing::error!(
                    error = %e,
                    arch = arch_class,
                    "kv-quant preset rejected by arch invariant"
                );
                eprintln!("error: {e}");
                eprintln!("see docs/KV_CACHE.md for supported codecs and combinations");
                std::process::exit(78);
            }
            kq
        }
        (None, None) => auto,
        (None, Some(spec)) => {
            let ctx = ResolverContext {
                arch_class,
                head_dim: head_dim_opt,
            };
            match resolve_cache_type(spec, ctx, auto) {
                Ok(kq) => kq,
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        arch = arch_class,
                        head_dim = ?head_dim_opt,
                        "cache-type validation failed"
                    );
                    eprintln!("error: {e}");
                    eprintln!("see docs/KV_CACHE.md for supported codecs and combinations");
                    std::process::exit(78);
                }
            }
        }
        (Some(_), Some(_)) => {
            // Defense-in-depth: clap `conflicts_with("kv_quant")` should have
            // rejected this with exit 2 already. No panic — fail fast with
            // EX_CONFIG.
            tracing::error!(
                arch = arch_class,
                "both --kv-quant preset and --cache-type-* per-side codecs supplied — these are mutually exclusive"
            );
            eprintln!("error: --kv-quant and --cache-type-k/--cache-type-v are mutually exclusive");
            eprintln!("see docs/KV_CACHE.md for supported codecs and combinations");
            std::process::exit(78);
        }
    };

    tracing::info!(
        arch = arch_class,
        head_dim = ?head_dim_opt,
        kv_quant = ?final_kv_quant,
        "cache-type resolved"
    );

    // §D6.7 disclosure: SWA layers always use bf16 regardless of --ctk/--ctv.
    if !matches!(final_kv_quant, KvQuant::None)
        && (arch_class.starts_with("Gemma3") || arch_class.starts_with("Gemma4"))
    {
        tracing::info!("SWA layers always use bf16 — only full-attention layers are quantized");
    }

    final_kv_quant
}

/// Reject a zero `--max-prompt-tokens`; `truncate(0)` would empty the prompt.
pub(crate) fn parse_max_prompt_tokens(v: usize) -> anyhow::Result<usize> {
    if v == 0 {
        return Err(anyhow::anyhow!("--max-prompt-tokens must be >= 1, got 0"));
    }
    Ok(v)
}

/// Parse the `--max-ctx` flag value into an optional `i32` override.
///
/// `None` → no override (arch derives from `max_position_embeddings`, capped at 4096).
/// `Some(n)` with `n >= 256` → use `n` directly as the KV buffer size.
/// `Some(n)` with `n < 256` → validation error.
pub(crate) fn parse_max_ctx(v: Option<u32>) -> anyhow::Result<Option<i32>> {
    match v {
        None => Ok(None),
        Some(n) if n < 256 => Err(anyhow::anyhow!("--max-ctx must be >= 256, got {n}")),
        Some(n) => Ok(Some(n as i32)),
    }
}

/// Parse a fractional `--kv-bits` value into an asymmetric K-floor/V-ceil
/// [`KvQuant::Mixed`].
///
/// Only called when `bits` is non-integer (e.g. `3.5`). Integer values are
/// dispatched to [`parse_kv_bits_combo`] unchanged.
///
/// # Floor/ceil mapping (mirrors mlx-vlm `_ensure_codecs`)
///
/// ```text
/// k_bits = floor(bits) (e.g. 3.5 → 3)
/// v_bits = ceil(bits) (e.g. 3.5 → 4)
/// ```
///
/// Both sides use `group_size` as their group size.
///
/// # Rejected inputs
///
/// - `floor(bits)` outside `{3, 4, 5, 6, 8}` → error (K floor not supported by
///   MLX affine quantizer).
/// - `ceil(bits)` outside `{3, 4, 5, 6, 8}` → error (V ceil not supported).
/// - `group_size == 0`, `group_size > 65535`, or any `group_size` outside
///   `{32, 64, 128}` → error, via [`rmlx_kv_quant::validate_mixed_side`].
/// - The value is not strictly fractional (integer path should have been taken).
pub(crate) fn parse_kv_bits_fractional(
    bits: f32,
    group_size: usize,
) -> anyhow::Result<rmlx_kv_quant::KvQuant> {
    use rmlx_kv_quant::KvQuant;

    const VALID_BITS: &[u8] = &[3, 4, 5, 6, 8];

    if group_size == 0 {
        return Err(anyhow::anyhow!("--kv-group-size must be > 0, got 0"));
    }

    let k_bits = bits.floor() as u8;
    let v_bits = bits.ceil() as u8;

    if !VALID_BITS.contains(&k_bits) {
        return Err(anyhow::anyhow!(
            "--kv-bits {bits}: K floor={k_bits} is not a supported MLX affine bit-width; \
             supported set is {{{}}}",
            VALID_BITS
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if !VALID_BITS.contains(&v_bits) {
        return Err(anyhow::anyhow!(
            "--kv-bits {bits}: V ceil={v_bits} is not a supported MLX affine bit-width; \
             supported set is {{{}}}",
            VALID_BITS
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let group_size = kv_group_size_u16(group_size)?;
    // Both sides reach the MLX affine quantizer, so both have to be a pair it
    // implements — the same check `--kv-quant mixed_*` runs. The bit-width
    // screens above do not cover the group size, so without this the alias
    // flags could build a codec `KvQuant::from_str` refuses to spell back.
    for (side, side_bits) in [('k', k_bits), ('v', v_bits)] {
        rmlx_kv_quant::validate_mixed_side(side, side_bits, group_size)
            .map_err(|e| anyhow::anyhow!("--kv-bits {bits} --kv-group-size {group_size}: {e}"))?;
    }

    Ok(KvQuant::Mixed {
        k_bits,
        v_bits,
        k_group_size: group_size,
        v_group_size: group_size,
    })
}

/// Narrow a `--kv-group-size` to the `u16` the codec field holds.
///
/// A plain `as u16` wraps: `--kv-group-size 65600` would arrive as 64 and pass
/// every check below it while naming a size nobody asked for.
fn kv_group_size_u16(group_size: usize) -> anyhow::Result<u16> {
    u16::try_from(group_size)
        .map_err(|_| anyhow::anyhow!("--kv-group-size must be <= 65535, got {group_size}"))
}

/// Parse `--kv-bits` + `--kv-group-size` integer aliases into a concrete [`KvQuant`].
///
/// mlx-lm ergonomics: users pass integer bit-widths and group sizes instead of
/// the named preset strings.
///
/// # Mapping table
///
/// | (bits, group_size) | → KvQuant |
/// |--------------------|-----------|
/// | (8, 128) | K8V8 (rMLX MSL q8_0, both sides) |
/// | (8, 64) | Mixed { k=8,g=64 / v=8,g=64 } |
/// | (4, 64) | Mixed { k=8,g=64 / v=4,g=64 } (mlx-lm K=8 default) |
/// | (4, 32) | Mixed { k=8,g=64 / v=4,g=32 } |
/// | (3, 64) | Mixed { k=8,g=64 / v=3,g=64 } |
/// | any unmapped | Mixed { k_bits=8, v_bits=bits, k_group_size=64, v_group_size=group_size } |
///
/// The unmapped fallback mirrors mlx-lm's `maybe_quantize_kv_cache` default:
/// K stays at 8-bit (group=64) and V uses the caller-specified bits/group_size.
///
/// # Rejected inputs
///
/// - `bits` outside `{2, 3, 4, 5, 6, 8}` → error (not supported by MLX affine quantizer).
/// - `group_size == 0`, `group_size > 65535`, or any `group_size` outside
///   `{32, 64, 128}` → error, via [`rmlx_kv_quant::validate_mixed_side`]. The
///   named `(8, 128)` preset resolves to `K8V8` and does not reach it.
pub(crate) fn parse_kv_bits_combo(
    bits: u8,
    group_size: usize,
) -> anyhow::Result<rmlx_kv_quant::KvQuant> {
    use rmlx_kv_quant::KvQuant;

    // 2-bit is valid here because parse_kv_bits_combo always keeps K at
    // 8-bit (the Mixed fallback below) — the V side carries the requested bits.
    // Pure 2-bit K is gated in combo_to_kv_quant, not reachable from this path.
    const VALID_BITS: &[u8] = &[2, 3, 4, 5, 6, 8];
    if !VALID_BITS.contains(&bits) {
        return Err(anyhow::anyhow!(
            "--kv-bits must be one of {{{}}}, got {bits} \
             (bits outside this set are not supported by the MLX affine quantizer)",
            VALID_BITS
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if group_size == 0 {
        return Err(anyhow::anyhow!("--kv-group-size must be > 0, got 0"));
    }

    let group_size = kv_group_size_u16(group_size)?;

    // The one named preset that maps to a non-Mixed variant.
    if (bits, group_size) == (8, 128) {
        return Ok(KvQuant::K8V8);
    }

    // Everything else takes mlx-lm's default: K stays at 8-bit (group=64), V
    // uses the caller's bits/group_size. The V side is therefore the one that
    // varies (K is pinned to a pair the validator accepts), and it reaches the
    // MLX affine quantizer — so it has to be a pair that quantizer implements,
    // the same check `--kv-quant mixed_*` runs. The bit-width screen above does
    // not cover the group size.
    rmlx_kv_quant::validate_mixed_side('v', bits, group_size)
        .map_err(|e| anyhow::anyhow!("--kv-bits {bits} --kv-group-size {group_size}: {e}"))?;

    Ok(KvQuant::Mixed {
        k_bits: 8,
        v_bits: bits,
        k_group_size: 64,
        v_group_size: group_size,
    })
}

/// Parse and resolve all model-related CLI flags in one call.
///
/// Shared preamble for `serve` (single-model path), `chat`, `info`, and
/// `baseline`: runs `parse_kv_quant` → `build_cache_type_spec` →
/// `parse_max_ctx` → `parse_device` → emits an `info!` span for the
/// resolved device → loads `config.json` via `rmlx_loader::load_config` →
/// resolves the final [`KvQuant`] via [`resolve_kv_quant`].
///
/// `kv_bits` + `kv_group_size`: when both are `Some`, they are resolved via
/// [`parse_kv_bits_combo`] and used as the `kv_quant_override` (the
/// `--kv-quant` string is ignored in that case — clap `conflicts_with` prevents
/// both from being set simultaneously). When `kv_bits` is `Some` but
/// `kv_group_size` is `None`, `kv_group_size` defaults to 64 (mlx-lm default).
///
/// `cmd_name` is a short label used only in the `info!` log line
/// (e.g. `"rmlx serve"`, `"rmlx chat"`, `"rmlx info"`, `"rmlx baseline"`).
///
/// Returns `(device, kv_quant, max_ctx_override)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_model_flags(
    model: &std::path::Path,
    kv_quant: &str,
    ctk: Option<&str>,
    ctv: Option<&str>,
    max_ctx: Option<u32>,
    device: &str,
    cmd_name: &str,
    kv_bits: Option<f32>,
    kv_group_size: Option<usize>,
) -> anyhow::Result<(Device, rmlx_kv_quant::KvQuant, Option<i32>)> {
    // --kv-bits / --kv-group-size: resolve to a KvQuant before the normal
    // preset path. clap conflicts_with prevents --kv-bits from appearing
    // alongside --kv-quant / --cache-type-k / --cache-type-v, so if kv_bits
    // is Some here, we skip parse_kv_quant / build_cache_type_spec entirely.
    //
    // fractional values (e.g. 3.5) dispatch to parse_kv_bits_fractional
    // (floor K / ceil V); integer values dispatch to parse_kv_bits_combo
    // (mapping, unchanged).
    let (kv_quant_opt, cts_override) = if let Some(bits) = kv_bits {
        let gs = kv_group_size.unwrap_or(64);
        let kq = if bits.fract() == 0.0 {
            parse_kv_bits_combo(bits as u8, gs)?
        } else {
            parse_kv_bits_fractional(bits, gs)?
        };
        tracing::info!(
            kv_bits = bits,
            kv_group_size = gs,
            kv_quant = ?kq,
            "{cmd_name}: --kv-bits resolved"
        );
        (Some(kq), None)
    } else {
        (parse_kv_quant(kv_quant)?, build_cache_type_spec(ctk, ctv)?)
    };
    let max_ctx_override = parse_max_ctx(max_ctx)?;
    let dev = parse_device(device)?;
    tracing::info!(device, "{cmd_name}: resolved device");
    let cfg = rmlx_loader::load_config(model).map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
    let kv_quant_final = resolve_kv_quant(&cfg, kv_quant_opt, cts_override);
    Ok((dev, kv_quant_final, max_ctx_override))
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
