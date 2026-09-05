//! rMLX binary entry.
//!
//! Parses subcommands, sets up tracing to `logs/<run-id>.jsonl`, prints version.

// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::float_cmp,
        clippy::ignore_without_reason,
    )
)]
// Allocator selection — mutually exclusive:
//
// dhat-heap feature (OFF by default): replaces the global allocator with dhat's
// profiling allocator for heap-allocation analysis. See docs/PROFILING.md §4.
// Build: cargo build --features rmlx-cli/dhat-heap --bin rmlx
// On exit, writes dhat-heap.json to the current directory.
//
// Default (no feature): jemalloc, which reduces fragmentation for long-running
// servers with KV-cache churn on macOS (libmalloc degrades under multi-threaded
// short-lived alloc patterns). Apple Silicon only per CLAUDE.md §1.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static GLOBAL: dhat::Alloc = dhat::Alloc;

#[cfg(not(feature = "dhat-heap"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

mod commands;
mod panic_hook;
mod startup;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use rmlx_core::runinfo::make_run_id;
use rmlx_metrics::events::EventRecorder;
use rmlx_server::SENTINEL_PORT;
use startup::{
    init_tracing, print_cache_type_table, print_kv_quant_residency_table, LogLevel, MetricsArg,
};
use tracing::info;

use commands::metrics::{dispatch as metrics_dispatch, MetricsCmd};

/// Clap `ValueEnum` adapter for `rmlx_models::DraftKind`.
///
/// `rmlx-models` carries no clap dep, so the `ValueEnum` impl lives here.
/// Conversion to the model-crate type is via `From<DraftKindArg>`. Each value
/// is spelled as `DraftKind::as_str` spells it, so the flag, the log fields
/// and the metrics `decode_config` say one thing.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum DraftKindArg {
    /// Multi-Token Prediction sidecar (Qwen3.5-family head, Gemma4 assistant).
    Mtp,
    /// Draft-Flash attention-based draft head.
    #[value(name = "dflash")]
    DFlash,
    /// EAGLE-3 speculative drafter.
    Eagle3,
    /// A separate full draft model of the verifier's family.
    #[value(name = "two_model")]
    TwoModel,
}

impl From<DraftKindArg> for rmlx_models::DraftKind {
    fn from(a: DraftKindArg) -> Self {
        match a {
            DraftKindArg::Mtp => rmlx_models::DraftKind::Mtp,
            DraftKindArg::DFlash => rmlx_models::DraftKind::DFlash,
            DraftKindArg::Eagle3 => rmlx_models::DraftKind::Eagle3,
            DraftKindArg::TwoModel => rmlx_models::DraftKind::TwoModel,
        }
    }
}

/// `--draft-block-size` at parse time: the engine's floor, refused before a
/// model loads rather than on the first request after two have.
fn parse_draft_block_size(s: &str) -> Result<usize, String> {
    let block: usize = s
        .parse()
        .map_err(|e| format!("not a whole number of tokens: {e}"))?;
    if block < rmlx_server::MIN_DRAFT_BLOCK_SIZE {
        return Err(format!(
            "a block of {block} leaves no room for a draft token; it must be at least {}",
            rmlx_server::MIN_DRAFT_BLOCK_SIZE
        ));
    }
    Ok(block)
}

/// CLI value-enum wrapper for [`rmlx_models::prefix_index::PrefixIndexKind`].
///
/// review MEDIUM-3: lets clap reject garbage values at parse-time
/// (`possible values: linear, radix` usage error) rather than after the
/// downstream `String::parse()` returns an anyhow. The model crate carries no
/// clap dep, so the `ValueEnum` impl lives here.
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq, Default)]
enum PrefixIndexKindArg {
    /// O(slots × n_blocks) linear scan; byte-identical to pre-.
    #[default]
    Linear,
    /// NVIDIA Dynamo positional radix tree.
    Radix,
}

impl From<PrefixIndexKindArg> for rmlx_models::prefix_index::PrefixIndexKind {
    fn from(v: PrefixIndexKindArg) -> Self {
        match v {
            PrefixIndexKindArg::Linear => Self::Linear,
            PrefixIndexKindArg::Radix => Self::Radix,
        }
    }
}

use commands::parse::KvPresetArg;
use commands::serve::{
    FusedQkMode, PlanarFlashDecodeMode, RotKFusedMode, SparseAttnMode, TurboFlashMode,
};
use commands::{
    acquire_claim_for_device, build_cache_type_spec, kv_bits_u8, parse_device, parse_kv_bits_combo,
    parse_kv_bits_fractional, parse_kv_boundary_layers, parse_kv_preset, parse_kv_quant,
    parse_max_ctx, parse_max_prompt_tokens, resolve_model_flags, resolve_preset_arg, run_baseline,
    run_bench, run_healthcheck, run_info, run_kv_calibrate, run_ppl, run_serve,
};

/// Long-help body shared by `--cache-type-k` / `--cache-type-v` on every
/// subcommand. Mirrors the §D1 table at a high level; full set is in
/// `docs/KV_CACHE.md` and `rmlx info --list-cache-types`.
const CACHE_TYPE_K_LONG_HELP: &str = "\
Per-side KV cache codec for the K (key) tensor (`--cache-type-k` / `--ctk`).

Mutually exclusive with `--kv-quant`. Pass either a preset (`--kv-quant`) or
per-side codecs (`--cache-type-k`/`--cache-type-v`), never both.

  Examples: q8_g128 (rMLX MSL 8-bit, K+V), q4_g64 (MLX affine 4-bit, V), tq4 (TurboQuant 4-bit, V-only), planar4 (PlanarQuant 4-bit, V-only), bf16 (unquantized, K+V).
  See `rmlx info --list-cache-types` for the full set, or docs/KV_CACHE.md.
  Note for llama.cpp users: rMLX uses double-dash --ctk (not single-dash -ctk).";

const CACHE_TYPE_V_LONG_HELP: &str = "\
Per-side KV cache codec for the V (value) tensor (`--cache-type-v` / `--ctv`).

Mutually exclusive with `--kv-quant`. Pass either a preset (`--kv-quant`) or
per-side codecs (`--cache-type-k`/`--cache-type-v`), never both.

  Examples: q8_g128 (rMLX MSL 8-bit, K+V), q4_g64 (MLX affine 4-bit, V), tq4 (TurboQuant 4-bit, V-only), planar4 (PlanarQuant 4-bit, V-only), bf16 (unquantized, K+V).
  See `rmlx info --list-cache-types` for the full set, or docs/KV_CACHE.md.
  Note for llama.cpp users: rMLX uses double-dash --ctv (not single-dash -ctv).";

/// Long-help for `--kv-preset`. Documents the preset table and conflict rules.
///
/// Read by `make check-kv-codec-disposition` alongside the `--kv-quant` and
/// `--kv-bits` help: every preset here resolves to a codec, and the gate holds
/// the disposition this text claims against the one the runtime classifiers
/// give. The `--kv-quant` name of each target is spelled out for that reason as
/// much as for the reader's — a gate that searches for a codec cannot find it
/// under a variant name that never reaches the command line.
const KV_PRESET_LONG_HELP: &str = "\
Named KV-cache preset. Bundles K+V codec choice into a single flag.

Mutually exclusive with `--kv-quant`, `--cache-type-k`, `--cache-type-v`,
and `--kv-bits`. Passing any of those alongside `--kv-preset` is a hard
clap error (exit 2).

NO PRESET BELOW REDUCES MEMORY BELOW fp16 — fp16 is the unquantised reference,
and every other name resolves to a codec whose decode reads the bf16 mirror, so
its packed store is never built: the served request holds the same resident KV
as fp16 and emits the same tokens. Pinned by `no_preset_is_a_memory_lever`,
which sweeps the preset table and fails the build the day one of these targets
starts building a store. Decode throughput against fp16 is INCONCLUSIVE, so
they are not known to cost anything either. They are kept because the names
appear in recorded bench rows and each is its codec's entry point, not because
one of them is a smaller cache.

  INERT — every preset target except fp16:
      speed resolves to tsym3, quality to tsym4, planar to planar, planar3 to
      planar3, k_only_planar to planar_k. q8 resolves to k8v8, which is inert
      too. Decode reads the bf16 mirror on both axes for all six, so the packed
      store is never built and the codec math never runs.

Special value:
  auto      -- the same codec `--kv-quant auto` resolves to (unquantised bf16).
               It does not inspect the hardware: the memory-pressure selector
               that used to live here returned a preset that saves no bytes.

Presets:
  fp16          -- bf16 both sides (unquantised; KvQuant::None)
  q8            -- symmetric 8-bit K+V (KvQuant::K8V8)
  speed         -- TurboSym3 (symmetric WHT-3 K+V; rejected on Qwen MoE)
  quality       -- TurboSym4 (WHT-4 symmetric K+V; rejected on Qwen MoE — see arch guard note*)
  planar        -- PlanarQuant V-side (KvQuant::Planar)
  planar3       -- PlanarQuant 3-bit V-side (KvQuant::Planar3)
  k_only_planar -- PlanarQuant K-side, V bf16 (KvQuant::PlanarK; rejected on Qwen MoE)

*Arch guard: --kv-preset quality resolves to TurboSym4 (symmetric WHT-4 K+V), rejected on \
Qwen MoE (PPL disaster path). --kv-preset speed resolves to TurboSym3 (symmetric WHT-3 K+V), \
also rejected on Qwen MoE (K-side 3-bit PPL disaster). \
See docs/KV_QUANT.md sections \"Preset semantics\" and \"Codec disposition\".";

/// Short help for `--kv-quant`, shared by every subcommand that takes it.
///
/// Names no codec and quotes no ratio on purpose. A name in a one-line help
/// cannot carry its disposition, and a ratio typed into a help string is a
/// second copy of a number the engine already computes — the one this line used
/// to carry named two codec families as the only ones under bf16 when six are.
const KV_QUANT_HELP: &str = "\
KV cache quantization codec. Default \"auto\" = unquantised bf16 on every arch. \
Which codecs hold less resident KV than bf16 depends on the architecture — \
`rmlx info --list-cache-types` prints the computed figure per codec. See --help.";

/// Long-help for `--kv-quant`, shared by every subcommand that takes it.
///
/// The codec lists here are checked against the runtime classifiers by
/// `make check-kv-codec-disposition`: a name in the INERT block that starts
/// reading its own packed store, or an inert name listed anywhere else, fails
/// the build. The same gate rejects a resident-KV ratio written into this text
/// at all — that figure has one producer, `rmlx info --list-cache-types`, which
/// computes it from the engine's byte model instead of quoting it.
const KV_QUANT_LONG_HELP: &str = "\
KV cache quantization codec. Default \"auto\", which resolves to unquantised
bf16 (\"bf16\", alias \"none\") on every architecture and every context length.

HOW MUCH RESIDENT KV A CODEC HOLDS IS NOT PRINTED HERE. It is architecture-
conditional, and `rmlx info --list-cache-types` computes it per codec, for a
dense stack and a cross-layer-KV (shared-KV) stack side by side, from the same
byte model the engine allocates against. Read that listing before picking a
name to save memory; pick a name here for what its decode does.

Mutually exclusive with `--kv-preset`, `--cache-type-k`, `--cache-type-v`,
and `--kv-bits`.

  bf16 (alias none; what auto resolves to)
      Unquantised bf16 K and V. The reference the listing's ratios are
      against.

  INERT — accepted, but does nothing:
      k8v4, k8v8, planar, planar3, planar_k, k8vturbo2, k8vturbo2tcq,
      k8vturbo3, k8vturbo3tcq, tsym3, tsym4, iso3, iso4, rotor3, rotor4,
      rotor_k_3_asym_v<vb>_g<vg>, rotor_k_4_asym_v<vb>_g<vg>.
      Decode reads the bf16 mirror on both axes, so the packed store is never
      built and the codec math never runs. Resident KV and generated tokens
      measure identical to bf16, on two architectures at two contexts; decode
      throughput against it is INCONCLUSIVE, so selecting one is not known to
      cost anything either. It simply does not do what the name says.

  Runs its codec, decoding over its own packed store:
      mixed_k<kb>g<kg>_v<vb>g<vg>, rot_k_v<vb>g<vg>, iso3_sym, iso4_sym,
      k_iso3, k_iso4, rotor3_sym, rotor4_sym, k_rotor3, k_rotor4.
      Two mechanisms decide what that costs, and they pull opposite ways:
      * mixed_* / rot_k_* keep a bf16 K/V mirror beside their store only where
        the model's layers share K/V, because that mirror is what the consumer
        layers read. There they hold the store AND two full bf16 buffers;
        everywhere else the mirror is not built and the store is the whole cost.
        One codec, two answers — the listing prints both.
      * The iso and rotor families keep no mirror on any architecture. Their
        rate is the ring's, which is fixed by head_dim and by the sideband
        width — not by the bit width in the name, which is a codebook. The two
        members of a family are byte-identical for that reason.
      Whether either is *below* bf16 is the listing's business, not this text's.

Per-codec detail: docs/KV_QUANT.md, section \"Codec disposition — what every
codec in the tree is for\".";

/// Long-help for `--kv-bits`. Mirrors mlx-lm's `kv_bits` / `kv_group_size` ergonomics.
///
/// Same disposition gate as [`KV_QUANT_LONG_HELP`] — the codec names here are
/// checked against the runtime classifiers, in both directions.
const KV_BITS_LONG_HELP: &str = "\
Bit-width alias for KV cache quantization (mlx-lm ergonomics).
Accepts integer or fractional values (e.g. 3, 4, 3.5, 4.5).

Mutually exclusive with `--kv-quant`, `--kv-preset`, `--cache-type-k`, and
`--cache-type-v`. Pass `--kv-bits` + optionally `--kv-group-size` instead of a
preset string.

WHAT THIS FLAG SHRINKS DEPENDS ON THE MODEL. Every value except (8, 128)
resolves to the mixed_* codec, whose bf16 K/V mirror is retained on exactly one
kind of architecture: one whose layers share K/V across layers (Gemma4), where
that mirror is what the consumer layers read. There the cache holds the packed
store AND both mirrors and is LARGER than plain bf16. Where the layers do not
share K/V, the mirror is not built and the packed store is the whole cache.
`rmlx info --list-cache-types` computes both figures per codec; this flag names
no ratio because the answer is not one number. The single exception below is
neither smaller nor larger:

  INERT — accepted, but does nothing:
      k8v8, which is what (8, 128) resolves to.
      Decode reads the bf16 mirror on both axes, so its packed store is never
      built: resident KV and generated tokens measure identical to bf16.

`--kv-bits` cannot reach a codec smaller than bf16 on a shared-KV architecture:
every value it maps to is in the mirrored family. The codecs that compress there
are reached through `--kv-quant`; `rmlx info --list-cache-types` names which.

Integer mapping:
  --kv-bits 8 --kv-group-size 128  → k8v8 (see INERT above)
  --kv-bits 4 --kv-group-size 64   → mixed_k8g64_v4g64  [mlx-lm default]
  --kv-bits 4 --kv-group-size 32   → mixed_k8g64_v4g32
  --kv-bits 3 --kv-group-size 64   → mixed_k8g64_v3g64
  other (bits, group_size)         → mixed_k8g64_v<bits>g<group_size>

Fractional mapping (MLX affine, K=floor / V=ceil — not TurboQuant):
  --kv-bits 3.5 --kv-group-size 64 → mixed_k3g64_v4g64
  --kv-bits 4.5 --kv-group-size 64 → mixed_k4g64_v5g64
  Fractional bits: K=floor(bits), V=ceil(bits), both sides use group_size.

K always defaults to 8-bit / group=64 for integer values (mlx-lm K=8 convention).
When --kv-group-size is omitted, group_size defaults to 64.
Valid bits: 2, 3, 4, 5, 6, 8 for integers, 3..8 for the floor and ceil of a
fraction. Valid group sizes: 32, 64, 128 — the set the MLX affine
quantizer implements. Anything else is a parse error, not a mode.";

/// Long-help for `--kv-boundary-layers`, shared by every subcommand that takes it.
const KV_BOUNDARY_LAYERS_LONG_HELP: &str = "\
How many leading and trailing decoder layers are held at the boundary floor,
as `<head>,<tail>`. Default `2,8`.

A boundary layer runs the quality floor instead of the requested codec. For a
codec whose widths are parameters (`mixed_*`, `rot_k_*`) the floor is that
codec's own 8-bit form. For every other quantizing codec the floor is a target
that materialises no packed store and decodes at model dtype, so those boundary
layers are bf16 layers and cost bf16 bytes.

The counts are therefore a memory knob as well as a quality one, and the
default is inherited rather than derived: raising `head` or `tail` spends more
bytes at the floor, and `0,0` turns the promotion off entirely and runs the
requested codec on every layer.

How many layers this actually moves is a property of the model, not the flag.
A windowed (sliding-attention) layer runs the bf16 rotating ring whatever it is
handed, and a shared-KV consumer layer owns no cache at all, so on an
architecture whose head and tail are all windowed or all consumers the flag
changes nothing. `rmlx info --list-cache-types` reports the per-layer mix.

Runs at a non-default value are recorded in `decode_config` as
`kv_boundary/head=<h>,kv_boundary/tail=<t>`, so they rank as their own cell and
never against a default-boundary run.";

#[derive(Parser, Debug)]
#[command(name = "rmlx", version, about = "Rust-native MLX inference server")]
struct Cli {
    /// Log verbosity preset (info|debug|verbose). `RUST_LOG` overrides this
    /// when set. Default `info`.
    #[arg(long, value_enum, global = true, default_value_t = LogLevel::Info)]
    log: LogLevel,
    /// Metrics recording level (off|events|full). Default `full`.
    ///
    /// `off` writes nothing — the drainer never spawns and `runs.db` is never
    /// opened or created. `events` keeps the runtime event stream but records
    /// no bench observations. Reading (`rmlx metrics best|export|query`) works
    /// in every mode.
    #[arg(long = "metrics", value_enum, global = true, default_value_t = MetricsArg::Full)]
    metrics_mode: MetricsArg,
    /// Total size cap for `<RMLX_HOME>/logs/` in megabytes. When the directory
    /// exceeds this limit at startup, the oldest `.jsonl` files are deleted
    /// until the total is within the cap. `0` disables rotation (logs grow
    /// unbounded). Default 100.
    ///
    /// Env: `RMLX_LOG_CAP_MB`.
    #[arg(
        long,
        global = true,
        env = "RMLX_LOG_CAP_MB",
        default_value_t = 100u64,
        value_name = "MB"
    )]
    log_cap_mb: u64,
    /// Toggle the K-side 1-bit QJL residual for the rotor3_sym /
    /// rotor4_sym / k_rotor3 / k_rotor4 codecs. Default `off`: QJL has no Metal
    /// kernel, so turning it on forces the rotor K path onto CPU (single-digit
    /// TPS) with no measured accuracy gain; off routes the rotor K encode
    /// through the Metal fused-decode kernel. Pass `--rotor-qjl on` to opt into
    /// the residual for ablation / fidelity study.
    /// Env fallback `RMLX_ROTOR_QJL=1` honored when this flag is absent.
    #[arg(long, value_enum, global = true, default_value_t = RotorQjlArg::Off)]
    rotor_qjl: RotorQjlArg,
    /// Route pre-softmax QK over PlanarQuant-packed K through the fused MSL
    /// kernel (`planar_fused_qk`).  Default `on`; `off` reverts to the legacy
    /// dequant+SDPA path (ablation / bench baseline).  Affects only
    /// `KvStorage::PlanarK` caches.  No env fallback — CLI-only.
    #[arg(long, value_enum, global = true, default_value_t = PlanarFusedQkArg::On)]
    planar_fused_qk: PlanarFusedQkArg,
    /// Generalized fused-QK kernels for q8 / turbo3 / turbo4 / iso / rotor
    /// K-packed caches.  Default `auto`.
    ///
    /// `auto` (default): HOLD — kernel stubs present but not dispatching.
    /// Auto stays OFF until codec implementations land and NIAH gates pass;
    /// a pre-existing `RMLX_FUSED_QK=1` is still honoured.
    /// `on`: resolve the gate on.
    /// `off`: HARD override — resolves off even with `RMLX_FUSED_QK=1` set.
    #[arg(long, value_enum, global = true, default_value_t = FusedQkMode::Auto)]
    fused_qk: FusedQkMode,
    /// Two-phase sparse-attention dispatch (phase1_score +
    /// phase2_sparse_attend).  Default `auto`.
    ///
    /// `auto` (default): HOLD — warm-TTFT dormant by design on normal generate
    /// flows. Auto stays OFF until seedless workloads demonstrate measurable
    /// speedup; a pre-existing `RMLX_SPARSE_ATTN=1` is still honoured.
    /// `on`: resolve the gate on.
    /// `off`: HARD override — resolves off even with `RMLX_SPARSE_ATTN=1` set.
    #[arg(long, value_enum, global = true, default_value_t = SparseAttnMode::Auto)]
    sparse_attn: SparseAttnMode,
    /// TurboFlash MSL attention kernel. Default `auto`.
    ///
    /// `auto` (default): resolves OFF on every host. On the one storage it
    /// serves (K8V4, `kv_seq > 4096`) the kernel decodes 2.0–4.25× slower than
    /// the generic path — the loss grows with `kv_seq` — and holds ~722 MB more
    /// resident KV.
    ///
    /// It also changes the generated tokens, for a reason that is not a kernel
    /// defect: it is the only K8V4 configuration in which the 4-bit V codec
    /// runs at decode at all (the generic path reads the bf16 mirror), so the
    /// difference is that codec's ≈0.997 fidelity floor. Against a
    /// dequant-then-SDPA reference over its *own* packed buffers the kernel
    /// agrees to ≤2 bf16 ULP.
    ///
    /// That ratio predates the dtype fix: the dispatcher used to return its f32
    /// kernel output uncast, which promoted the whole decode graph while the
    /// gate was on. Read it as an upper bound on the kernel's own cost. The
    /// direction, and this default, are unchanged. HOLD until a decode
    /// re-measurement clears it; a pre-existing
    /// `RMLX_TURBO_FLASH=1` is still honoured, and logs a `warn!` naming the
    /// cost because the kernel then runs while the flag reads `auto`.
    /// `on`: resolve the gate on (ablation, and the escape hatch for that
    /// re-measurement).
    /// `off`: hard override — resolves off even with `RMLX_TURBO_FLASH=1` set.
    ///
    /// Global: every subcommand resolves this the same way, so `rmlx bench`
    /// and `rmlx baseline` measure the kernel configuration `rmlx serve` runs.
    #[arg(long, value_enum, global = true, default_value_t = TurboFlashMode::Auto)]
    turbo_flash: TurboFlashMode,
    /// Enable the TurboFlash lock variant. Default OFF.
    ///
    /// Skips bf16 K/V buffer maintenance once the persistent flash buffers are seeded.
    /// Has no effect unless `--turbo-flash` (or `RMLX_TURBO_FLASH=1`) is also active.
    /// There is no `off` arm: passing the flag resolves lock-on, and when it is
    /// absent `RMLX_TURBO_FLASH_LOCK=1` is still honoured (back-compat), so
    /// clearing it means unsetting the variable.
    #[arg(long, global = true, default_value_t = false)]
    turbo_flash_lock: bool,
    /// PlanarQuant flash-decode MSL kernel. Default `auto`.
    ///
    /// `auto` (default): OFF on every host — the warm-TTFT bf16-K seed shadows
    /// the kernel on the normal generate flow, so there is no measurable TPS
    /// win to flip Auto for; a pre-existing `RMLX_PLANAR_FLASH_DECODE=1` is
    /// still honoured.
    /// `on`: resolve the gate on.
    /// `off`: HARD override — resolves off even with
    /// `RMLX_PLANAR_FLASH_DECODE=1` set.
    ///
    /// Only takes effect for PlanarK-storage layers (i.e.
    /// `--kv-quant planar_k`); other KV variants fall through unchanged.
    #[arg(long, value_enum, global = true, default_value_t = PlanarFlashDecodeMode::Auto)]
    planar_flash_decode: PlanarFlashDecodeMode,
    /// Fused FWHT + affine-quantize MSL kernel for the rot_k codec families.
    /// Default `auto`.
    ///
    /// `auto` (default): OFF — the rotate-by-matmul path is the validated one;
    /// a pre-existing `RMLX_ROT_K_FUSED=1` is still honoured.
    /// `on`: force the fused kernel (ablation / bench).
    /// `off`: HARD override — ignores `RMLX_ROT_K_FUSED=1` in the shell.
    ///
    /// Only affects caches whose codec rotates K (`--kv-quant
    /// rot_k_v<bits>g<group>`); every other codec ignores it.
    #[arg(long, value_enum, global = true, default_value_t = RotKFusedMode::Auto)]
    rot_k_fused: RotKFusedMode,
    #[command(subcommand)]
    cmd: Cmd,
}

/// On|off toggle for the K-side QJL residual.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum RotorQjlArg {
    On,
    Off,
}

impl RotorQjlArg {
    fn enabled(self) -> bool {
        matches!(self, RotorQjlArg::On)
    }
}

/// On|off toggle for the PlanarQuant fused-QK MSL kernel.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
enum PlanarFusedQkArg {
    On,
    Off,
}

impl PlanarFusedQkArg {
    fn enabled(self) -> bool {
        matches!(self, PlanarFusedQkArg::On)
    }
}

#[derive(Subcommand, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "Serve is the primary CLI variant; clap-derived structs are stack-allocated once at startup and not stored in collections"
)]
enum Cmd {
    /// Serve OpenAI + Anthropic chat-completion HTTP API.
    Serve {
        /// Path to a model snapshot directory (optional; empty registry if omitted).
        /// Mutually exclusive with --registry.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Path to a JSON registry file listing model snapshots.
        /// Format: {"models":[{"id":"name","path":"/abs/path"},...]}.
        /// Mutually exclusive with --model.
        #[arg(long)]
        registry: Option<PathBuf>,
        /// Named launch profile from `<RMLX_HOME>/profiles.toml`.
        /// Loads `[profile.<name>]` as defaults; any CLI flag the user passes
        /// overrides the profile value. See `rmlx profile list`.
        #[arg(long)]
        profile: Option<String>,
        /// TCP port to listen on. Default 8080 (or the profile's `port`).
        #[arg(long)]
        port: Option<u16>,
        /// Host/IP to bind. Default 127.0.0.1 (or the profile's `host`).
        #[arg(long)]
        host: Option<String>,
        /// Device to run inference on: "cpu" or "gpu".
        /// Defaults to "gpu". Chunked prefill (Stage-3.2b) resolves the Metal watchdog
        /// timeout on long prompts. Use --device cpu to fall back to CPU.
        #[arg(long)]
        device: Option<String>,
        #[arg(long, help = KV_QUANT_HELP, long_help = KV_QUANT_LONG_HELP)]
        kv_quant: Option<String>,
        /// Named KV-cache preset (see long-help). Mutually exclusive with
        /// `--kv-quant`, `--cache-type-k`, `--cache-type-v`, and `--kv-bits`.
        #[arg(
            long,
            value_name = "NAME",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v", "kv_bits"],
            value_parser = parse_kv_preset,
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
        /// Per-side KV cache codec for K (see long-help).
        #[arg(
            long = "cache-type-k",
            visible_alias = "ctk",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_K_LONG_HELP,
        )]
        cache_type_k: Option<String>,
        /// Per-side KV cache codec for V (see long-help).
        #[arg(
            long = "cache-type-v",
            visible_alias = "ctv",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_V_LONG_HELP,
        )]
        cache_type_v: Option<String>,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "BITS",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v"],
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Group size for --kv-bits (default 64). See --kv-bits long-help.
        #[arg(long, value_name = "N", requires = "kv_bits")]
        kv_group_size: Option<usize>,
        /// Head/tail layer counts held at the KV boundary floor (see long-help).
        #[arg(
            long,
            value_name = "HEAD,TAIL",
            value_parser = parse_kv_boundary_layers,
            long_help = KV_BOUNDARY_LAYERS_LONG_HELP,
        )]
        kv_boundary_layers: Option<rmlx_models::kv_cache::KvBoundary>,
        /// Maximum context length (tokens) the run may address. Bounded by the
        /// checkpoint's positional capacity — `max_position_embeddings`,
        /// extended by a `rope_scaling` the config declares or by
        /// `--yarn-factor`. A value above that capacity is refused, naming
        /// both numbers; it is never clamped. When unset, the ceiling is
        /// `min(capacity, 4096)`. Must be >= 256 when set.
        #[arg(long)]
        max_ctx: Option<u32>,
        /// Idle keep-alive — unload the model after this much idle time.
        ///
        /// Accepts an integer count of seconds (`30`, `900`) OR a Go-style
        /// duration string (`30s`, `15m`, `2h`, `24h`). Negative (`-1`)
        /// pins the model forever; `0` unloads immediately after each response.
        ///
        /// Default: `15m` (900 s). Override per request with the `keep_alive`
        /// body field on native rMLX routes (OpenAI/Anthropic-compat routes
        /// ignore the field but still reset the timer on use).
        #[arg(long, value_name = "DURATION", allow_hyphen_values = true)]
        idle_timeout_secs: Option<String>,
        /// Number of prompt-cache slots for multi-slot prefix matching.
        /// Each slot holds a post-prefill KV snapshot; on a prefix match
        /// the cached state is cloned and the new tail is prefilled.
        /// Default 4. Set to 1 for legacy single-slot exact-match behaviour.
        #[arg(long)]
        prompt_cache_slots: Option<usize>,
        /// Drafter snapshot for speculative decoding: a sidecar head (MTP,
        /// DFlash, EAGLE-3) or a smaller full model of the verifier's family.
        /// Which one it is is read from its config.json. A full draft model
        /// must carry the verifier's tokenizer; that is checked at load time.
        #[arg(long)]
        draft_model: Option<PathBuf>,
        /// Drafter kind, for a `--draft-model` whose config.json does not
        /// declare one. Refused when it contradicts what the snapshot declares.
        /// Requires `--draft-model`. Env: `MLX_VLM_DRAFT_KIND`.
        ///
        /// Values: mtp, dflash, eagle3, two_model
        #[arg(long, value_enum, requires = "draft_model", env = "MLX_VLM_DRAFT_KIND")]
        draft_kind: Option<DraftKindArg>,
        /// Speculative round block: tokens the verifier scores per round, its
        /// own token included, so the drafter proposes one fewer. The same
        /// number for every drafter kind. Must be ≥ 2. Default 5.
        /// Env: `MLX_VLM_DRAFT_BLOCK_SIZE` (fallback when flag is absent).
        #[arg(long, value_name = "N", env = "MLX_VLM_DRAFT_BLOCK_SIZE", value_parser = parse_draft_block_size)]
        draft_block_size: Option<usize>,
        /// Per-request `max_tokens` ceiling. Requests with a higher value
        /// receive HTTP 400 `invalid_request_error` instead of being silently
        /// clamped. Only lowers the server's structural ceiling of 1 048 576
        /// completion tokens, which applies when this flag is absent.
        #[arg(long)]
        max_tokens_cap: Option<u32>,
        /// Server-startup cap on per-request wall-clock timeout, in seconds (A8).
        ///
        /// Every request (including SSE streams) is bounded by this timeout.
        /// The `X-Request-Timeout-Seconds` header can lower the effective timeout
        /// per request, but never exceed this cap. 0 = no timeout (disabled).
        /// Default 600 (10 minutes).
        #[arg(long)]
        max_timeout_secs: Option<u64>,
        /// Run the 8-token smoke probe on first model load (B5). Default OFF.
        ///
        /// When set, the server runs `classify_smoke` on the snapshot before
        /// placing it in the serving slot. If the verdict is `BrokenPunctLoop`
        /// or `BrokenNan`, the load is rejected with HTTP 503 and the server
        /// continues waiting for a valid model request. Applies to every
        /// model swap, not just the first load.
        ///
        /// When absent (default), no startup probe is run — zero overhead on the
        /// hot path. Use `rmlx info --probe-smoke` for offline auditing instead.
        #[arg(long, default_value_t = false)]
        require_smoke_probe: bool,
        /// Maximum number of models held resident in GPU memory at once (C4).
        ///
        /// Default 1 — byte-equivalent to the single-slot behaviour: loading
        /// a different model evicts the current one (implicit swap). With a
        /// value > 1, up to N models stay resident and the least-recently-used
        /// model is evicted only when a new one is requested past capacity.
        /// All forward passes are still serialised process-wide (single Metal
        /// context per process).
        #[arg(long)]
        max_loaded_models: Option<usize>,
        /// Maximum number of admitted-and-in-flight requests before new ones
        /// are rejected with HTTP 429 `server queue full` (C5 Slice A).
        ///
        /// Requests serialise on the single GPU; this bounds the FIFO
        /// admission queue depth so a burst cannot stack unbounded
        /// `spawn_blocking` threads. Admitted requests are served in strict
        /// FIFO arrival order (fairness fix). `0` = unlimited (FIFO +
        /// queue metrics still apply, no 429). Default 64.
        #[arg(long)]
        max_queue_depth: Option<usize>,
        /// Server-startup default temperature applied when a request omits the
        /// `temperature` field (G4). Precedence: request > this > model
        /// generation_config.json > hard-coded 1.0.
        ///
        /// Set to 0.0 for deterministic greedy decoding across all requests
        /// (parity with mlx_lm.server `--temp 0`). Must be in [0.0, 2.0].
        /// When absent (default), behaviour is unchanged from before this flag.
        #[arg(long)]
        default_temperature: Option<f32>,
        /// Server-startup default for thinking mode on Qwen3-family models.
        ///
        /// `--enable-thinking false` suppresses the open `<think>` block (no-think
        /// mode) for all requests unless a per-request `enable_thinking` field
        /// overrides it. `--enable-thinking true` is the explicit opt-in (same as
        /// the current default when the flag is absent).
        ///
        /// Precedence: per-request `enable_thinking` > this > absent (= enabled).
        /// When absent (default), the Qwen3.6 template default is preserved:
        /// an open `<think>` block byte-identical to HF `apply_chat_template`.
        #[arg(long, value_name = "BOOL")]
        enable_thinking: Option<bool>,
        /// Server-startup default image-token budget for Gemma4-unified vision.
        ///
        /// Raises the soft-token budget the vision preprocessor allocates per
        /// image, preserving more resolution for dense inputs (e.g. tables).
        /// Clamped to the model's safe upper bound (1120). When absent
        /// (default), the snapshot's `processor_config.json` `max_soft_tokens`
        /// (typically 280) is used — behaviour unchanged.
        ///
        /// Precedence: per-request `image_max_tokens` > this > config default.
        /// A no-op for text-only requests and non-Gemma4-unified vision archs.
        #[arg(long, value_name = "N")]
        image_max_tokens: Option<usize>,
        /// SSD prompt-cache tier budget, in GiB. GLOBAL ceiling over the
        /// active cache namespace's on-disk KV blocks.
        ///
        /// Default 0 = no per-namespace ceiling. With `--kv-ssd-global-gb` also
        /// 0 that means the tier is OFF (RAM-only prompt cache, unchanged
        /// behaviour); with a global pool set, the tier is on and the pool
        /// ceiling governs this namespace on its own.
        /// When the tier is on, RAM-evicted prompt-cache snapshots spill to
        /// `<RMLX_HOME>/cache/kv/<namespace>/` and a RAM miss is served from the
        /// longest cached block-aligned prefix on disk. The namespace is the
        /// model id unless `--project` overrides it. Eviction within the
        /// namespace is LRU-by-size: the ceiling is enforced at model load
        /// (index pruned + evicted-to-budget) and again after every block the
        /// spill thread writes, so it holds for the life of the process and not
        /// only at startup. Because `rmlx serve` runs a single MLX process, one
        /// namespace is active at a time, so this per-namespace ceiling is also
        /// the global ceiling.
        #[arg(long, value_name = "GIB", default_value_t = 0.0)]
        kv_ssd_cache_gb: f64,
        /// SSD prompt-cache namespace. Additive per-project cache:
        /// blocks land in `<RMLX_HOME>/cache/kv/<NAME>/` with their own index
        /// and `--kv-ssd-cache-gb` budget, isolated from other projects.
        ///
        /// Requires `--kv-ssd-cache-gb > 0` (the tier must be ON). When absent,
        /// the namespace defaults to the model id.
        #[arg(long, value_name = "NAME")]
        project: Option<String>,
        /// SSD global pool ceiling across ALL namespaces under
        /// `<RMLX_HOME>/cache/kv/*`, in GiB.
        ///
        /// Default `0` = no global ceiling (per-namespace `--kv-ssd-cache-gb`
        /// stands alone). When `> 0`, every model load runs a cross-namespace
        /// LRU sweep at startup: rows are evicted oldest-first across the
        /// union of all namespaces until the pool sum is ≤ this budget. Then
        /// the active namespace's own per-namespace eviction runs (as today).
        ///
        /// Independent of `--kv-ssd-cache-gb`. If the per-namespace value
        /// exceeds the global budget, the per-namespace ceiling is implicitly
        /// clamped (`min(per_ns, global)`), and a warning is emitted.
        ///
        /// Precedence (per-namespace ceiling effective value): the tighter of
        /// the two when both are > 0; whichever one is set when only one is;
        /// and no ceiling at all when neither is (the tier is off). A zero on
        /// either flag is "unconfigured", never "a ceiling of zero bytes".
        #[arg(long, value_name = "GIB", default_value_t = 0.0)]
        kv_ssd_global_gb: f64,
        /// RAM cap for the in-process prompt cache, in GiB.
        ///
        /// Precedence: CLI > default 2 GiB. Applies to every per-arch prompt
        /// cache constructed during
        /// this process — controls `PromptCache::new` via the
        /// `install_ram_cap` resolver.
        #[arg(long, value_name = "GIB")]
        prompt_cache_ram_gb: Option<f64>,
        /// Enable the paged-KV block-table storage path.
        ///
        /// OFF by default — the contiguous KV path is unchanged. When ON:
        /// every freshly constructed K8V4 / K8V8 / Planar cache routes through
        /// `KvStorage::Paged` (block-table allocator) instead of the
        /// contiguous-growth path.
        ///
        /// Restrictions enforced at CLI parse time:
        /// - `--paged-kv` + `--kv-quant bf16` (or `--kv-quant none`) is rejected
        ///   (the paged path supports K8V4 / K8V8 / Planar only).
        /// - `--paged-kv` + `--cache-type-k rot_k*` is rejected (RotK
        ///   are not paged-compatible — they ride the Mixed quantized-SDPA path).
        ///
        #[arg(long, default_value_t = false)]
        paged_kv: bool,
        /// per-page token count for `--paged-kv` (positive integer).
        ///
        /// Requires `--paged-kv` (clap `requires`). Default 32 (TurboQuant /
        /// PlanarQuant group size — exactly one quantiser group per element
        /// per page, no cross-page partial groups).
        #[arg(long, value_name = "N", requires = "paged_kv")]
        paged_kv_page_tokens: Option<i32>,
        /// prompt-cache longest-prefix index strategy.
        ///
        /// `linear` (default) — O(slots × n_blocks) scan, byte-identical to
        /// the pre-path. The radix tree is still maintained in
        /// parallel (for the differential bench) but unused on lookup.
        ///
        /// `radix` — NVIDIA Dynamo positional radix tree, O(n_blocks) lookup
        /// independent of slot count. Opt-in pending the bench gate
        /// (≥2× linear at N≥32 with <5% mem overhead → default flip). The
        /// linear path remains the bisect-safe fallback either way.
        #[arg(
            long,
            value_name = "KIND",
            value_enum,
            default_value_t = PrefixIndexKindArg::Linear,
        )]
        prefix_index: PrefixIndexKindArg,
        /// Enable the in-process adaptive admission controller (default OFF).
        ///
        /// When enabled, the controller dynamically adjusts `max_queue_depth` based
        /// on observed request latency vs the SLA targets (`--step-target-ms`,
        /// `--itl-target-ms`). Also enables anticipatory 503 rejection: if the
        /// 2D OLS regression predicts end-to-end step > 2× step target for the next
        /// admission, the request is rejected with 503 + `Retry-After` instead of
        /// being queued.
        ///
        /// When absent (default), the open-loop FIFO queue path is used.
        #[arg(long, default_value_t = false)]
        adaptive_admission: bool,
        /// End-to-end step SLA target in milliseconds for the adaptive controller.
        ///
        /// M2: this is the admission→final-token wall-clock target (not TTFT per se).
        /// Anticipatory 503 fires when `est_step > 2 × step_target`. Default 500 ms.
        /// `--ttft-target-ms` is accepted as a hidden alias for backward compatibility.
        #[arg(
            long = "step-target-ms",
            alias = "ttft-target-ms",
            value_name = "MS",
            default_value_t = rmlx_server::admission::DEFAULT_STEP_TARGET_MS
        )]
        ttft_target_ms: u64,
        /// ITL SLA target in milliseconds for the adaptive controller.
        ///
        /// Requires `--adaptive-admission`. Queue depth is lowered when sustained
        /// `est_itl > itl_target` and raised when `est_itl < itl_target × 0.80`.
        /// Default 50 ms (Dynamo default).
        #[arg(long, value_name = "MS", default_value_t = rmlx_server::admission::DEFAULT_ITL_TARGET_MS)]
        itl_target_ms: u64,
        /// Enable adaptive prefill-chunk sizing (default OFF).
        ///
        /// Requires `--adaptive-admission`. When enabled, the admission controller
        /// also adjusts the process-wide prefill chunk size based on the same ITL
        /// regression: raises the chunk when load is below the deadband (< 80 % of
        /// `--itl-target-ms`), lowers it after `HOLD_TICKS` consecutive overload
        /// ticks. Operates independently of queue-depth adjustment. Bounds: 32–2048.
        ///
        /// OFF by default — the chunk tuning is higher-risk and defaults are locked
        /// from p0b-ttft bench data. Use only when the bench shows clear headroom.
        #[arg(long, default_value_t = false)]
        adaptive_prefill_chunk: bool,
        /// Path to a Whisper model snapshot directory (e.g. mlx-community/whisper-large-v3-mlx).
        ///
        /// Required to serve `POST /v1/audio/transcriptions` and `/v1/audio/translations`.
        /// Env: `RMLX_WHISPER_MODEL_PATH`.
        #[arg(long, env = "RMLX_WHISPER_MODEL_PATH", value_name = "PATH")]
        whisper_model_path: Option<PathBuf>,
        /// Path to a Whisper tokenizer directory containing `tokenizer.json`.
        ///
        /// mlx-community Whisper snapshots do not ship tokenizer files; download
        /// the openai/whisper-large-v3 tokenizer and point here (or set
        /// RMLX_WHISPER_TOKENIZER_PATH). When absent, audio endpoints return 503.
        /// Env: `RMLX_WHISPER_TOKENIZER_PATH`.
        #[arg(long, env = "RMLX_WHISPER_TOKENIZER_PATH", value_name = "PATH")]
        whisper_tokenizer_path: Option<PathBuf>,
        /// Path to the Qwen3-TTS model snapshot directory.
        ///
        /// Required to serve `POST /v1/audio/speech`. The snapshot must contain
        /// `config.json` and `model.safetensors` (or shards).
        /// Env: `RMLX_TTS_MODEL_PATH`.
        ///
        /// NOTE: Phase 4b (neural codec decoder) is not yet implemented; the
        /// endpoint returns HTTP 501 until Phase 4b lands. The flag is accepted
        /// now so server startup configs can be written ahead of time.
        #[arg(long, env = "RMLX_TTS_MODEL_PATH", value_name = "PATH")]
        tts_model_path: Option<PathBuf>,
        /// Path to the Qwen3-TTS speech tokenizer (codec decoder) snapshot directory.
        ///
        /// Required together with `--tts-model-path`. Points to the
        /// `Qwen3-TTS-Tokenizer-12Hz` snapshot (neural codec decoder).
        /// Env: `RMLX_TTS_TOKENIZER_PATH`.
        #[arg(long, env = "RMLX_TTS_TOKENIZER_PATH", value_name = "PATH")]
        tts_tokenizer_path: Option<PathBuf>,
        /// Byte budget for the multimodal encoder-output cache.
        ///
        /// Caches vision-tower (and audio-encoder) outputs keyed on the
        /// post-preprocess pixel/PCM content hash so repeated calls with
        /// identical inputs skip the encoder forward entirely. `0` disables
        /// the cache. Default 512 MiB.
        ///
        /// Env: `RMLX_MM_CACHE_BYTES`.
        #[arg(
            long,
            env = "RMLX_MM_CACHE_BYTES",
            value_name = "BYTES",
            default_value_t = 512 * 1024 * 1024
        )]
        mm_cache_bytes: usize,
        /// Maximum number of active sessions held in the LRU session cache.
        ///
        /// Each active session reserves an extra prompt-cache slot so multi-turn
        /// conversations are not evicted by concurrent single-turn requests. When
        /// the limit is reached the least-recently-used session is dropped (its
        /// KV tensors are not deleted — only the slot reservation is lost).
        ///
        /// Default 64. Env: `RMLX_SESSION_CACHE_MAX_SESSIONS`.
        #[arg(
            long,
            env = "RMLX_SESSION_CACHE_MAX_SESSIONS",
            value_name = "N",
            default_value_t = 64usize
        )]
        session_cache_max_sessions: usize,
        /// YARN RoPE scale factor. Extends the context window a Qwen3-family
        /// checkpoint can address to `factor * original` and raises the
        /// ceiling `--max-ctx` is bounded by. Overrides a `rope_scaling` the
        /// config declares; must be > 1.0 to take effect. Output quality past
        /// the checkpoint's trained window is the operator's risk, and every
        /// run past it is logged.
        ///
        /// Env: `RMLX_YARN_FACTOR`.
        #[arg(long, env = "RMLX_YARN_FACTOR", value_name = "FLOAT")]
        yarn_factor: Option<f32>,
        /// Pre-extension context size `--yarn-factor` interpolates from.
        /// When absent, the checkpoint's declared
        /// `original_max_position_embeddings` is used, falling back to its
        /// `max_position_embeddings`.
        ///
        /// Env: `RMLX_YARN_ORIGINAL_MAX`.
        #[arg(long, env = "RMLX_YARN_ORIGINAL_MAX", value_name = "U32")]
        yarn_original_max: Option<u32>,
    },
    /// One-off REPL chat for sanity-checking a model.
    Chat {
        #[arg(long)]
        model: PathBuf,
        /// Device to run inference on: "cpu" or "gpu".
        /// Defaults to "gpu". Chunked prefill (Stage-3.2b) resolves the Metal watchdog timeout.
        #[arg(long, default_value = "gpu")]
        device: String,
        #[arg(
            long,
            default_value = "auto",
            help = KV_QUANT_HELP,
            long_help = KV_QUANT_LONG_HELP
        )]
        kv_quant: String,
        /// Named KV-cache preset (see long-help). Mutually exclusive with
        /// `--kv-quant`, `--cache-type-k`, `--cache-type-v`, and `--kv-bits`.
        #[arg(
            long,
            value_name = "NAME",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v", "kv_bits"],
            value_parser = parse_kv_preset,
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
        /// Per-side KV cache codec for K (see long-help).
        #[arg(
            long = "cache-type-k",
            visible_alias = "ctk",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_K_LONG_HELP,
        )]
        cache_type_k: Option<String>,
        /// Per-side KV cache codec for V (see long-help).
        #[arg(
            long = "cache-type-v",
            visible_alias = "ctv",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_V_LONG_HELP,
        )]
        cache_type_v: Option<String>,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "BITS",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v"],
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Group size for --kv-bits (default 64). See --kv-bits long-help.
        #[arg(long, value_name = "N", requires = "kv_bits")]
        kv_group_size: Option<usize>,
        /// Maximum context length (tokens) the run may address. Bounded by the
        /// checkpoint's positional capacity — `max_position_embeddings`,
        /// extended by a `rope_scaling` the config declares or by
        /// `--yarn-factor`. A value above that capacity is refused, naming
        /// both numbers; it is never clamped. When unset, the ceiling is
        /// `min(capacity, 4096)`. Must be >= 256 when set.
        #[arg(long)]
        max_ctx: Option<u32>,
    },
    /// Transcribe an audio file to text / subtitles (speech-to-text).
    ///
    /// Arch-dispatched on the snapshot's `config.json` (Whisper today; the
    /// dispatch is a seam for future ASR architectures). The input container is
    /// decoded and resampled to 16 kHz mono internally, so any
    /// `.m4a`/`.wav`/`.mp3`/`.flac`/… works directly. Output goes to stdout, or
    /// to `--output` when given.
    Transcribe {
        /// Input audio file (any Symphonia-supported container).
        #[arg(value_name = "AUDIO")]
        audio: PathBuf,
        /// Model snapshot directory (Whisper). Env: `RMLX_WHISPER_MODEL_PATH`.
        #[arg(long, env = "RMLX_WHISPER_MODEL_PATH", value_name = "PATH")]
        model: PathBuf,
        /// Companion tokenizer directory. Whisper snapshots ship no
        /// `tokenizer.json`; point this at the `openai/whisper-large-v3`
        /// tokenizer dir. Env: `RMLX_WHISPER_TOKENIZER_PATH`. Falls back to the
        /// model dir when absent.
        #[arg(long, env = "RMLX_WHISPER_TOKENIZER_PATH", value_name = "PATH")]
        tokenizer: Option<PathBuf>,
        /// Output format: txt | json | srt | vtt.
        #[arg(long, default_value = "txt")]
        format: String,
        /// Language code (`en`, `fr`, …) or `auto` for detection.
        #[arg(long, default_value = "auto")]
        language: String,
        /// Translate to English instead of transcribing in the source language.
        #[arg(long, default_value_t = false)]
        translate: bool,
        /// Write output to this file instead of stdout.
        #[arg(long, value_name = "PATH")]
        output: Option<PathBuf>,
        /// Device: "cpu" or "gpu" (default "gpu").
        #[arg(long, default_value = "gpu")]
        device: String,
    },
    /// Print arch + quant info for a snapshot, no inference.
    Info {
        /// Path to the model snapshot directory.
        ///
        /// Required unless `--list-cache-types` is set (which prints the §D1
        /// codec table without loading a model).
        #[arg(long, required_unless_present = "list_cache_types")]
        model: Option<PathBuf>,
        /// Device to run inference on: "cpu" or "gpu".
        /// Defaults to "gpu". Chunked prefill (Stage-3.2b) resolves the Metal watchdog timeout.
        /// Only relevant when --probe-forward or --probe-smoke is set.
        #[arg(long, default_value = "gpu")]
        device: String,
        /// Run a single-token forward pass and print top-1 token + max logit.
        /// Token 2 (BOS) is used. Requires the model to be Gemma4ForConditionalGeneration.
        #[arg(long, default_value_t = false)]
        probe_forward: bool,
        /// Run the 8-token smoke probe and classify the snapshot.
        ///
        /// Exit codes: 0 = ok (coherent), 1 = broken (BrokenPunctLoop or BrokenNan),
        /// 3 = load-fail (supported arch failed to load), 4 = inconclusive (too few steps),
        /// 5 = unsupported (architecture not handled). Exit 2 is reserved by clap.
        #[arg(long, default_value_t = false)]
        probe_smoke: bool,
        #[arg(
            long,
            default_value = "auto",
            help = KV_QUANT_HELP,
            long_help = KV_QUANT_LONG_HELP
        )]
        kv_quant: String,
        /// Named KV-cache preset (see long-help). Mutually exclusive with
        /// `--kv-quant`, `--cache-type-k`, `--cache-type-v`, and `--kv-bits`.
        #[arg(
            long,
            value_name = "NAME",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v", "kv_bits"],
            value_parser = parse_kv_preset,
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
        /// Per-side KV cache codec for K (see long-help).
        #[arg(
            long = "cache-type-k",
            visible_alias = "ctk",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_K_LONG_HELP,
        )]
        cache_type_k: Option<String>,
        /// Per-side KV cache codec for V (see long-help).
        #[arg(
            long = "cache-type-v",
            visible_alias = "ctv",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_V_LONG_HELP,
        )]
        cache_type_v: Option<String>,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "BITS",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v"],
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Group size for --kv-bits (default 64). See --kv-bits long-help.
        #[arg(long, value_name = "N", requires = "kv_bits")]
        kv_group_size: Option<usize>,
        /// Print the full KV cache codec table (§D1) and exit.
        /// No model load is attempted when this flag is set.
        #[arg(long, default_value_t = false)]
        list_cache_types: bool,
        /// Maximum context length (tokens) the run may address. Bounded by the
        /// checkpoint's positional capacity — `max_position_embeddings`,
        /// extended by a `rope_scaling` the config declares or by
        /// `--yarn-factor`. A value above that capacity is refused, naming
        /// both numbers; it is never clamped. When unset, the ceiling is
        /// `min(capacity, 4096)`. Must be >= 256 when set.
        #[arg(long)]
        max_ctx: Option<u32>,
    },
    /// Manage the metrics SQLite database (schema init, health checks, backup/restore).
    Metrics(MetricsCmd),
    /// Check rMLX readiness: claim file, HTTP /health, registry loadability,
    /// metrics DB, disk space, and process memory.
    ///
    /// Emits one JSON line per check (or plain text with --human).
    /// Exit 0 = all green, 1 = any red, 2 = internal error.
    ///
    /// Default (no --full) path is MLX-free: safe to run repeatedly without
    /// holding the Metal context. --full loads MLX for the smoke probe;
    /// ensure no other rMLX instance is running before using --full.
    Healthcheck {
        /// Check every registered model's loadability.
        /// Mutually exclusive with --model.
        #[arg(long, conflicts_with = "model")]
        registry: Option<PathBuf>,
        /// Check a single model snapshot's loadability.
        /// Mutually exclusive with --registry.
        #[arg(long, conflicts_with = "registry")]
        model: Option<PathBuf>,
        /// Also probe a live server on this port (claim + HTTP checks).
        #[arg(long)]
        port: Option<u16>,
        /// Path to the metrics SQLite DB.
        /// Defaults to env RMLX_METRICS_DB or metrics/runs.db.
        #[arg(long)]
        db: Option<PathBuf>,
        /// Minimum free disk space in GiB for metrics/ and logs/ directories.
        /// Default 5.
        #[arg(long, default_value_t = 5)]
        min_disk_gb: u64,
        /// Also run the MLX smoke probe per model (loads MLX — slow, exclusive Metal).
        #[arg(long, default_value_t = false)]
        full: bool,
        /// Emit plain OK/FAIL text instead of JSON lines.
        #[arg(long, default_value_t = false)]
        human: bool,
    },
    /// Record performance baseline for a model snapshot.
    ///
    /// Measures load time, time-to-first-token, tokens/sec, and peak RSS.
    /// Appends one row to metrics/baseline.csv and one JSONL record to
    /// metrics/<run-id>.jsonl. Prints a one-line summary to stdout.
    Baseline {
        /// Path to the model snapshot directory.
        #[arg(long)]
        model: PathBuf,
        /// Path to the prompt file. Defaults to the bundled fixture. Mutually
        /// exclusive with `--prompt-tokens`.
        #[arg(
            long,
            default_value = "crates/rmlx-cli/tests/fixtures/baseline_prompt.txt",
            conflicts_with = "prompt_tokens"
        )]
        prompt: PathBuf,
        /// Select a canonical bench prompt from `prompts/longctx_<N/1024>k.json`
        /// (e.g. `--prompt-tokens 4096` → `prompts/longctx_4k.json`). Bench
        /// harness convenience flag; mutually exclusive with `--prompt`.
        #[arg(long, value_name = "N")]
        prompt_tokens: Option<u32>,
        /// Device: "cpu" or "gpu". Defaults to "gpu" (chunked prefill resolves watchdog, Stage-3.2b).
        #[arg(long, default_value = "gpu")]
        device: String,
        /// Number of tokens to generate. Default 32. Visible alias `--gen-tokens`.
        #[arg(long, visible_alias = "gen-tokens", default_value_t = 32)]
        max_tokens: u32,
        /// Short label identifying the prompt fixture. Defaults to the prompt filename.
        /// Written to the `prompt` column of baseline.csv.
        #[arg(long, default_value = "")]
        prompt_label: String,
        #[arg(
            long,
            default_value = "auto",
            help = KV_QUANT_HELP,
            long_help = KV_QUANT_LONG_HELP
        )]
        kv_quant: String,
        /// Named KV-cache preset (see long-help). Mutually exclusive with
        /// `--kv-quant`, `--cache-type-k`, `--cache-type-v`, and `--kv-bits`.
        #[arg(
            long,
            value_name = "NAME",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v", "kv_bits"],
            value_parser = parse_kv_preset,
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
        /// Per-side KV cache codec for K (see long-help).
        #[arg(
            long = "cache-type-k",
            visible_alias = "ctk",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_K_LONG_HELP,
        )]
        cache_type_k: Option<String>,
        /// Per-side KV cache codec for V (see long-help).
        #[arg(
            long = "cache-type-v",
            visible_alias = "ctv",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_V_LONG_HELP,
        )]
        cache_type_v: Option<String>,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "BITS",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v"],
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Group size for --kv-bits (default 64). See --kv-bits long-help.
        #[arg(long, value_name = "N", requires = "kv_bits")]
        kv_group_size: Option<usize>,
        /// Head/tail layer counts held at the KV boundary floor (see long-help).
        #[arg(
            long,
            value_name = "HEAD,TAIL",
            value_parser = parse_kv_boundary_layers,
            long_help = KV_BOUNDARY_LAYERS_LONG_HELP,
        )]
        kv_boundary_layers: Option<rmlx_models::kv_cache::KvBoundary>,
        /// Maximum context length (tokens) the run may address. Visible alias
        /// `--ctx-max`. Bounded by the
        /// checkpoint's positional capacity — `max_position_embeddings`,
        /// extended by a `rope_scaling` the config declares or by
        /// `--yarn-factor`. A value above that capacity is refused, naming
        /// both numbers; it is never clamped. When unset, the ceiling is
        /// `min(capacity, 4096)`. Must be >= 256 when set.
        #[arg(long, visible_alias = "ctx-max")]
        max_ctx: Option<u32>,
        /// Truncate the tokenized prompt to at most this many tokens.
        /// Defaults to the run's resolved context ceiling (see `--max-ctx`),
        /// so raising the ceiling is what admits a longer prompt. Passing this
        /// flag explicitly opts into truncation on `--device gpu`. Must be >= 1.
        #[arg(long, value_name = "N")]
        max_prompt_tokens: Option<usize>,
        /// Opt into silently truncating a too-long prompt to the resolved
        /// context ceiling on `--device gpu` instead of erroring.
        /// `--device cpu` always truncates (real O(N^2) forward cost) regardless
        /// of this flag. Has no effect when `--max-prompt-tokens` is passed
        /// explicitly (that is itself an opt-in to truncation).
        #[arg(long, default_value_t = false)]
        allow_truncate: bool,
        /// Free-form label stamped into the metrics record's `notes` column
        /// (used by the bench harness to group cells under a campaign name).
        #[arg(long)]
        label: Option<String>,
        /// Emit one §8.5 universal `RunRecord` to `<RMLX_HOME>/metrics/buffer/pending/`
        /// at the end of the run and ingest it into `runs.db` in-process. The
        /// resulting row is visible to `rmlx metrics champions` / the `bests`
        /// view immediately. Used by the bench harness.
        #[arg(long, default_value_t = false)]
        record: bool,
        /// Commit SHA to record as provenance on the emitted metrics row
        /// (only meaningful with `--record`). Optional caller-supplied
        /// value — the binary cannot honestly know what commit it was
        /// built from, so this is never derived or guessed. Absent by
        /// default (`git_sha` is `NULL`).
        #[arg(long, value_name = "SHA")]
        git_sha: Option<String>,
        /// Root directory to search for canonical bench prompt files
        /// (`longctx_<N>k.json`). When unset, the binary walks up from the
        /// current working directory looking for a `prompts/` subdirectory that
        /// contains `longctx_4k.json`, then falls back to `prompts/` relative to
        /// cwd. This flag overrides both the cwd-walk and the fallback.
        ///
        /// Env: `RMLX_PROMPTS_DIR`.
        #[arg(long, env = "RMLX_PROMPTS_DIR", value_name = "PATH")]
        prompts_dir: Option<PathBuf>,
        /// YARN RoPE scale factor. Extends the context window a Qwen3-family
        /// checkpoint can address to `factor * original` and raises the
        /// ceiling `--max-ctx` is bounded by. Overrides a `rope_scaling` the
        /// config declares; must be > 1.0 to take effect. Output quality past
        /// the checkpoint's trained window is the operator's risk, and every
        /// run past it is logged.
        /// Env: `RMLX_YARN_FACTOR`.
        #[arg(long, env = "RMLX_YARN_FACTOR", value_name = "FLOAT")]
        yarn_factor: Option<f32>,
        /// Pre-extension context size `--yarn-factor` interpolates from.
        /// When absent, the checkpoint's declared
        /// `original_max_position_embeddings` is used, falling back to its
        /// `max_position_embeddings`.
        /// Env: `RMLX_YARN_ORIGINAL_MAX`.
        #[arg(long, env = "RMLX_YARN_ORIGINAL_MAX", value_name = "U32")]
        yarn_original_max: Option<u32>,
        /// Print the exact generated token-id sequence as a
        /// `baseline: token_ids=<comma-separated>` line after the summary.
        ///
        /// For A/B harnesses: two arms that produce different tokens are not
        /// comparable on speed, and decoded text cannot prove they match
        /// (different id sequences can decode to the same string). Off by
        /// default — the line is long and of no use to a human reader.
        #[arg(long, default_value_t = false)]
        emit_token_ids: bool,
        /// Write a Metal GPU trace of a bounded window of steady-state decode
        /// to PATH (a `.gputrace` bundle, opened in Xcode). Debug builds only —
        /// this flag exists only when the binary is built with
        /// `--features rmlx-cli/metal-capture`.
        ///
        /// The process must have been launched with `MTL_CAPTURE_ENABLED=1`;
        /// Metal inserts the capture layer at launch and cannot do so later.
        /// `scripts/gpu_capture.sh` handles both.
        ///
        /// Capture perturbs every timing this command measures, so it cannot be
        /// combined with `--record`.
        #[cfg(feature = "metal-capture")]
        #[arg(long, value_name = "PATH", conflicts_with = "record")]
        gpu_capture: Option<PathBuf>,
        /// Decode steps to run before the GPU-capture window opens. Skipping the
        /// first steps keeps first-touch kernel compilation and pipeline warm-up
        /// out of the trace.
        #[cfg(feature = "metal-capture")]
        #[arg(long, value_name = "N", default_value_t = 4, requires = "gpu_capture")]
        gpu_capture_skip: u32,
        /// Decode steps inside the GPU-capture window. Keep it small — every
        /// captured dispatch is serialised into the bundle.
        #[cfg(feature = "metal-capture")]
        #[arg(long, value_name = "N", default_value_t = 8, requires = "gpu_capture")]
        gpu_capture_steps: u32,
    },
    /// Repeated-run decode instrument: TTFT, ITL p50/p99, decode TPS and
    /// filled-prefix KV bytes for one (model, KV codec, context, generation)
    /// cell, as a median plus the observed run-to-run range.
    ///
    /// Unlike `baseline` (one run, one row appended to the metrics store),
    /// `bench` runs the cell `--warmup` + `--runs` times in one process,
    /// prints the spread, and writes nothing. It aborts rather than print a
    /// number whose measurement conditions did not hold — a run served from
    /// the prompt cache (so its TTFT is a replay time) and a KV-byte figure
    /// the run did not itself report are both hard errors.
    Bench {
        /// Path to the model snapshot directory.
        #[arg(long)]
        model: PathBuf,
        /// Path to the prompt file. Defaults to the bundled fixture. Mutually
        /// exclusive with `--prompt-tokens`.
        #[arg(
            long,
            default_value = "crates/rmlx-cli/tests/fixtures/baseline_prompt.txt",
            conflicts_with = "prompt_tokens"
        )]
        prompt: PathBuf,
        /// Select a canonical bench prompt from `prompts/longctx_<N/1024>k.json`
        /// (e.g. `--prompt-tokens 4096` → `prompts/longctx_4k.json`). Mutually
        /// exclusive with `--prompt`.
        #[arg(long, value_name = "N")]
        prompt_tokens: Option<u32>,
        /// Device: "cpu" or "gpu".
        #[arg(long, default_value = "gpu")]
        device: String,
        /// Tokens to generate per run. Visible alias `--gen-tokens`.
        #[arg(long, visible_alias = "gen-tokens", default_value_t = 128)]
        max_tokens: u32,
        /// Measured runs. Must be >= 2 — a single run has no observable spread.
        #[arg(long, default_value_t = 3)]
        runs: u32,
        /// Discarded warmup runs before the measured ones.
        #[arg(long, default_value_t = 1)]
        warmup: u32,
        #[arg(
            long,
            default_value = "auto",
            help = KV_QUANT_HELP,
            long_help = KV_QUANT_LONG_HELP
        )]
        kv_quant: String,
        /// Named KV-cache preset (see `rmlx baseline --help` long-help).
        /// Mutually exclusive with `--kv-quant`, `--cache-type-k`,
        /// `--cache-type-v` and `--kv-bits`.
        #[arg(
            long,
            value_name = "NAME",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v", "kv_bits"],
            value_parser = parse_kv_preset,
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
        /// Per-side KV cache codec for K (see long-help).
        #[arg(
            long = "cache-type-k",
            visible_alias = "ctk",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_K_LONG_HELP,
        )]
        cache_type_k: Option<String>,
        /// Per-side KV cache codec for V (see long-help).
        #[arg(
            long = "cache-type-v",
            visible_alias = "ctv",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_V_LONG_HELP,
        )]
        cache_type_v: Option<String>,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "BITS",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v"],
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Group size for --kv-bits (default 64). See --kv-bits long-help.
        #[arg(long, value_name = "N", requires = "kv_bits")]
        kv_group_size: Option<usize>,
        /// Head/tail layer counts held at the KV boundary floor (see long-help).
        #[arg(
            long,
            value_name = "HEAD,TAIL",
            value_parser = parse_kv_boundary_layers,
            long_help = KV_BOUNDARY_LAYERS_LONG_HELP,
        )]
        kv_boundary_layers: Option<rmlx_models::kv_cache::KvBoundary>,
        /// Maximum context length (tokens) the run may address. Visible alias
        /// `--ctx-max`. Bounded by the
        /// checkpoint's positional capacity — `max_position_embeddings`,
        /// extended by a `rope_scaling` the config declares or by
        /// `--yarn-factor`. A value above that capacity is refused, naming
        /// both numbers; it is never clamped. When unset, the ceiling is
        /// `min(capacity, 4096)`. Must be >= 256 when set.
        #[arg(long, visible_alias = "ctx-max")]
        max_ctx: Option<u32>,
        /// Truncate the tokenized prompt to at most this many tokens.
        /// Defaults to the run's resolved context ceiling (see `--max-ctx`).
        /// Same device-dependent semantics as
        /// `rmlx baseline --max-prompt-tokens`.
        #[arg(long, value_name = "N")]
        max_prompt_tokens: Option<usize>,
        /// Opt into silently truncating a too-long prompt on `--device gpu`
        /// instead of erroring.
        #[arg(long, default_value_t = false)]
        allow_truncate: bool,
        /// Emit the summary as one JSON object instead of the text table.
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Root directory to search for canonical bench prompt files
        /// (`longctx_<N>k.json`). Env: `RMLX_PROMPTS_DIR`.
        #[arg(long, env = "RMLX_PROMPTS_DIR", value_name = "PATH")]
        prompts_dir: Option<PathBuf>,
        /// Sampling temperature. `0` (the default) is greedy: the GPU argmax
        /// path, with no logits row read back to the host. Any positive value
        /// routes the cell through the host sampler instead, which is the only
        /// way to bench that path — every other shape this binary measures is
        /// greedy. The seed is the fixed default, so runs stay comparable.
        #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
        temperature: f32,
        /// Nucleus threshold. Requires `--temperature` > 0; `1.0` (the default)
        /// disables it. Its cost is an ordering pass over the whole vocabulary,
        /// so bench it separately from temperature alone.
        #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
        top_p: f32,
        /// Top-k cutoff. Requires `--temperature` > 0; `0` (the default)
        /// disables it. Also an ordering pass over the whole vocabulary. Needed
        /// to reproduce the served default: several snapshots ship a `top_k` in
        /// `generation_config.json`, so a request that omits sampling fields
        /// gets one.
        #[arg(long, default_value_t = 0, value_name = "N")]
        top_k: u32,
        /// Sign-aware multiplicative repetition penalty over the trailing
        /// 20-token window. `1.0` (the default) is the exact no-op. A value
        /// other than 1 reads the logits row to the host even at temperature 0.
        #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
        repetition_penalty: f32,
    },
    /// Manage named server profiles in `<RMLX_HOME>/profiles.toml`.
    Profile {
        #[command(subcommand)]
        cmd: ProfileCmd,
    },
    /// — offline evaluation harness (currently only `ppl`).
    Eval {
        #[command(subcommand)]
        cmd: EvalCmd,
    },
    /// CPU-only KV calibration pass: computes top-K high-precision indices
    /// per KV head from weight L2 norms and writes `kv_calib.json`.
    ///
    /// Requires no MLX/Metal context. Safe to run while `rmlx serve` is active.
    /// Output path defaults to `<model>/kv_calib.json`.
    ///
    /// Recipes:
    ///   turbo2       — ~25% high-precision dims (turboquant25 internal)
    ///   turbo2_tcq   — same ratio as turbo2 with TCQ codec
    ///   turbo3       — ~50% high-precision dims (turboquant35 internal)
    ///   turbo3_tcq   — same ratio as turbo3 with TCQ codec
    ///   turbo4       — same ratio as turbo3 (turboquant35 internal)
    ///   head_budget    — per-layer-per-head sparse-attn budgets (K-norm²
    ///                    proxy). Legacy v1 recipe; superseded by `softmax_mass`.
    ///   softmax_mass   — true softmax-mass calibration. Computes real
    ///                    Q@K^T -> softmax -> cumulative-mass top-K per
    ///                    (layer, head) using a long-context calibration
    ///                    prompt set. Writes head_budgets.json schema v2.
    ///   k_norm_proxy   — explicit alias for the legacy K-norm² proxy recipe,
    ///                    recorded as schema v2 with the proxy label.
    #[command(name = "kv-calibrate")]
    KvCalibrate {
        /// Path to the MLX model snapshot directory (must contain config.json + safetensors).
        #[arg(value_name = "MODEL")]
        model: PathBuf,
        /// Calibration recipe. Controls the outlier ratio and internal codec variant.
        #[arg(
            long,
            default_value = "turbo3",
            value_parser = [
                "turbo2", "turbo2_tcq", "turbo3", "turbo3_tcq", "turbo4",
                "head_budget", "softmax_mass", "k_norm_proxy",
            ],
        )]
        recipe: String,
        /// Output path for kv_calib.json (or head_budgets.json for the head-budget
        /// family recipes). Default: <model>/<filename>.
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
        /// Calibration prompt JSON file (head-budget family recipes only).
        /// Default: <repo>/prompts/calibration_long_context.json,
        /// falling back to calibration_default.json when absent.
        /// Ignored for weight-norm recipes.
        #[arg(long, value_name = "PATH")]
        prompts: Option<PathBuf>,
        /// Cumulative softmax-mass coverage target (head-budget family recipes
        /// only). Must be in [0.50, 1.00]. Default 0.95. Ignored for weight-norm
        /// recipes.
        #[arg(long, value_name = "FLOAT")]
        mass_threshold: Option<f32>,
        /// Minimum per-(layer, head) budget. Guards against pathological
        /// single-mass distributions producing a 1-slot budget.
        /// Default 16. Ignored for weight-norm recipes.
        #[arg(long, value_name = "U32", default_value_t = 16)]
        target_mass_budget_floor: u32,
    },
}

/// `rmlx eval <subcommand>`.
#[derive(Subcommand, Debug)]
enum EvalCmd {
    /// Compute perplexity over a text corpus using sliding-window NLL.
    ///
    /// Loads `--model`, tokenizes `--text-file`, and runs the native PPL
    /// scorer (Qwen3 family only — Bonsai is the smoke target).
    /// Prints one JSON line to stdout: `{"ppl":..,"mean_nll":..,"scored_tokens":..,"windows":..}`.
    /// When `--corpus wikitext-2` (or any non-empty value) is supplied, also
    /// ingests one §8.5 universal `RunRecord` into `<RMLX_HOME>/metrics/runs.db`
    /// under op `ppl_wikitext2`.
    ///
    /// **A KV cache is opt-in here.** With no KV flag the scorer forwards each
    /// window once and reads every position's logits out of that pass — no
    /// cache exists, so no codec and no layer policy can affect the number.
    /// Passing `--kv-quant` (or `--kv-preset` / `--kv-bits` /
    /// `--cache-type-*`) switches it to teacher-forcing the window through a
    /// real per-layer cache, one forward per scored token, so each NLL comes
    /// off the decode path a request runs. The two modes are recorded as
    /// different `decode_config` cells.
    Ppl {
        /// Path to the model snapshot directory (MLX format).
        #[arg(long)]
        model: PathBuf,
        /// Path to the corpus text file (raw UTF-8).
        #[arg(long, value_name = "PATH")]
        text_file: PathBuf,
        /// Number of tokens forwarded per window. Default 4096.
        #[arg(long, default_value_t = 4096)]
        ctx_window: usize,
        /// Stride between consecutive windows. Default 2048.
        #[arg(long, default_value_t = 2048)]
        stride: usize,
        /// Corpus identifier (currently only `wikitext-2` is recognised by the
        /// harness; the scorer accepts any string for the op-name tag).
        /// When empty, no metrics row is written.
        #[arg(long, default_value = "")]
        corpus: String,
        /// Device: "cpu" or "gpu". Default "gpu".
        #[arg(long, default_value = "gpu")]
        device: String,
        /// Cap the number of tokens fed to the scorer. `0` = use the whole
        /// corpus. Defaults to `0`; the wikitext-2 harness sets this when
        /// it wants a quick smoke run.
        #[arg(long, default_value_t = 0)]
        max_tokens: usize,
        /// Commit SHA to record as provenance on the emitted metrics row.
        /// Optional caller-supplied value — the binary cannot honestly know
        /// what commit it was built from, so this is never derived or
        /// guessed. Absent by default (`git_sha` is `NULL`).
        #[arg(long, value_name = "SHA")]
        git_sha: Option<String>,
        #[arg(long, help = KV_QUANT_HELP, long_help = KV_QUANT_LONG_HELP)]
        kv_quant: Option<String>,
        /// Named KV-cache preset (see long-help). Mutually exclusive with
        /// `--kv-quant`, `--cache-type-k`, `--cache-type-v`, and `--kv-bits`.
        #[arg(
            long,
            value_name = "NAME",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v", "kv_bits"],
            value_parser = parse_kv_preset,
            long_help = KV_PRESET_LONG_HELP,
        )]
        kv_preset: Option<KvPresetArg>,
        /// Per-side KV cache codec for K (see long-help).
        #[arg(
            long = "cache-type-k",
            visible_alias = "ctk",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_K_LONG_HELP,
        )]
        cache_type_k: Option<String>,
        /// Per-side KV cache codec for V (see long-help).
        #[arg(
            long = "cache-type-v",
            visible_alias = "ctv",
            value_name = "TAG",
            conflicts_with = "kv_quant",
            long_help = CACHE_TYPE_V_LONG_HELP,
        )]
        cache_type_v: Option<String>,
        /// Integer bit-width KV quantization alias. See long-help.
        #[arg(
            long,
            value_name = "BITS",
            conflicts_with_all = &["kv_quant", "cache_type_k", "cache_type_v"],
            long_help = KV_BITS_LONG_HELP,
        )]
        kv_bits: Option<f32>,
        /// Group size for --kv-bits (default 64). See --kv-bits long-help.
        #[arg(long, value_name = "N", requires = "kv_bits")]
        kv_group_size: Option<usize>,
        /// Head/tail layer counts held at the KV boundary floor (see long-help).
        /// Only meaningful alongside a KV codec — the cacheless scorer has no
        /// per-layer cache to apply it to.
        #[arg(
            long,
            value_name = "HEAD,TAIL",
            value_parser = parse_kv_boundary_layers,
            long_help = KV_BOUNDARY_LAYERS_LONG_HELP,
        )]
        kv_boundary_layers: Option<rmlx_models::kv_cache::KvBoundary>,
    },
}

/// `rmlx profile <subcommand>`. Scope is `serve` profiles only;
/// chat/baseline profiles are follow-on.
#[derive(Subcommand, Debug)]
enum ProfileCmd {
    /// List the names of all defined profiles, one per line.
    List,
}

// — defaults for the profile-bindable `serve` flags. Kept here (not as
// clap `default_value_t`) so the clap layer can distinguish "flag not passed"
// (`None` → use profile) from "flag passed". Final value resolves as
// `cli.or(profile).unwrap_or(DEFAULT)`.
const DEFAULT_SERVE_PORT: u16 = 8080;
const DEFAULT_SERVE_HOST: &str = "127.0.0.1";
const DEFAULT_SERVE_DEVICE: &str = "gpu";
const DEFAULT_SERVE_KV_QUANT: &str = "auto";
const DEFAULT_PROMPT_CACHE_SLOTS: usize = 4;
const DEFAULT_MAX_TOKENS_CAP: u32 = rmlx_server::bounds::MAX_COMPLETION_TOKENS;
const DEFAULT_MAX_TIMEOUT_SECS: u64 = 600;
const DEFAULT_MAX_LOADED_MODELS: usize = 1;
const DEFAULT_MAX_QUEUE_DEPTH: usize = 64;

/// Locate the canonical bench-prompt directory (`prompts/longctx_<N>k.json`).
///
/// An explicit `--prompts-dir` wins; otherwise walk up from cwd looking for a
/// `prompts/` subdirectory that contains `longctx_4k.json` (the workspace-root
/// convention the bench harnesses use), falling back to `prompts/` relative to
/// cwd. Shared by `baseline` and `bench` so both resolve `--prompt-tokens N`
/// against the same directory.
fn resolve_prompts_root(prompts_dir: Option<PathBuf>) -> PathBuf {
    prompts_dir
        .or_else(|| {
            let mut cur = std::env::current_dir().ok()?;
            loop {
                let p = cur.join("prompts");
                if p.join("longctx_4k.json").exists() {
                    return Some(p);
                }
                if !cur.pop() {
                    return None;
                }
            }
        })
        .unwrap_or_else(|| PathBuf::from("prompts"))
}

/// Whether this invocation asks for a GPU trace capture.
///
/// Read before the metrics kill switch is resolved: a captured run's numbers are
/// instrumentation artefacts, not measurements, and must not reach any metrics
/// surface. See the call site in [`main`].
#[cfg(feature = "metal-capture")]
fn gpu_capture_requested(cmd: &Cmd) -> bool {
    matches!(
        cmd,
        Cmd::Baseline {
            gpu_capture: Some(_),
            ..
        }
    )
}

/// Without the capture feature there is no flag to ask with.
#[cfg(not(feature = "metal-capture"))]
const fn gpu_capture_requested(_cmd: &Cmd) -> bool {
    false
}

/// Refuse to measure against an MLX that is not the validated pair.
///
/// The pin exists because a drifted MLX changes prefill throughput by ~3.8x
/// while leaving output correct and decode flat — so the number a bench
/// produces is wrong and nothing about the run looks wrong. Any prefill or
/// TTFT figure measured across the pin boundary is not comparable to one from
/// the other side, which makes recording it worse than not measuring.
///
/// Checked here, in the process that is about to take the measurement, because
/// that is the only place the answer describes the library actually loaded.
/// `scripts/mlx_preflight.sh` reads the package manager's symlinks, which are
/// not necessarily what this binary was linked against: `MLX_PREFIX` and
/// `MLX_C_PREFIX` (`crates/rmlx-mlx/build.rs`) can point a build at an install
/// the preflight never inspects.
fn refuse_to_measure_off_the_pin(command: &str) -> Result<()> {
    let check = rmlx_mlx::pin_check();
    if check.matches || !check.enforcement.is_binding() {
        tracing::debug!(
            detail = %check.detail,
            enforcement = ?check.enforcement,
            command,
            "MLX pin check cleared the measurement path"
        );
        return Ok(());
    }
    anyhow::bail!(
        "{command} refuses to measure: {}. Prefill and TTFT measured against an \
         unvalidated MLX are not comparable to any recorded number. Restore the pair \
         with `make mlx-restore-pin`, or see docs/FFI.md.",
        check.detail
    )
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "fn main() is the top-level CLI dispatch — splitting fragments the \
              subcommand wiring, which must remain co-located for clap arg resolution"
)]
fn main() -> Result<()> {
    // dhat profiler — instantiated FIRST so it covers all subsequent allocations.
    // Dropped at end of main, which triggers the JSON write to dhat-heap.json.
    // This block compiles to nothing when the feature is off.
    #[cfg(feature = "dhat-heap")]
    let _dhat = dhat::Profiler::new_heap();

    // Parse CLI first so `--log` can shape the tracing filter.
    let cli = Cli::parse();
    let run_id = make_run_id();

    // Resolve the metrics kill switch exactly once, before anything can open
    // the DB or spawn the drainer. Every writer reads it from here; no call
    // site carries its own toggle.
    //
    // A GPU-capture run is instrumentation, not measurement: the capture layer
    // serialises every dispatch and collapses decode to single-digit TPS, so
    // every number the run produces is false. Force the switch off for it. The
    // clap conflict with `--record` covers the §8.5 observations row only —
    // the `events` rows and the `metrics/baseline.csv` append happen outside
    // it, so without this a hand-run capture still wrote a ~2.5 TPS row into
    // append-only surfaces.
    let capture_forces_metrics_off = gpu_capture_requested(&cli.cmd);
    rmlx_metrics::mode::init(if capture_forces_metrics_off {
        rmlx_metrics::mode::MetricsMode::Off
    } else {
        cli.metrics_mode.mode()
    });

    // Set RUST_BACKTRACE before init_tracing so the call happens while the
    // process is genuinely single-threaded — no background threads exist yet
    // (the tracing_appender non-blocking writer is spawned inside init_tracing
    // below). RUST_BACKTRACE is only ever read by subsequent threads, never
    // written, so there is no data race.
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        #[allow(
            unsafe_code,
            reason = "set_var is safe here: called before init_tracing spawns the \
                      tracing_appender background thread; no other thread exists at \
                      this point in main(). RUST_BACKTRACE is only read by subsequent \
                      threads, so there is no data race."
        )]
        unsafe {
            std::env::set_var("RUST_BACKTRACE", "full");
        }
    }
    let _guard = init_tracing(&run_id, cli.log, cli.log_cap_mb)?;

    // Record the nax-GEMM-kernel capability of the MLX this process loaded.
    // `rmlx-metrics` cannot read this itself (see `identity::set_mlx_nax`
    // doc) — `rmlx-cli` is the one binary that links both `rmlx-mlx` and
    // `rmlx-metrics`, so it is the only place that can supply it.
    //
    // After `init_tracing` on purpose, and still well before the first
    // `RunIdentity::get()` / `EventRecorder::record`: the probe warns when it
    // cannot read the metallib, and that warning is the only record of *why* a
    // run ends up stamped `unknown` in an append-only table.
    rmlx_metrics::identity::set_mlx_nax(rmlx_mlx::nax_capability());

    // Reported after tracing is up, since the decision above predates it.
    if capture_forces_metrics_off {
        info!(
            "--gpu-capture: metrics forced off — a capture-distorted run writes no events \
             row, no baseline.csv row and no observation"
        );
    }

    // Install the rotor-QJL toggle before any cache construction.
    // This is a process-wide one-shot OnceLock; safe to call once at startup.
    rmlx_kv_quant::rotor_qjl::install_rotor_qjl(cli.rotor_qjl.enabled());

    // Install the planar-fused-QK toggle before any cache construction.
    // Process-wide one-shot OnceLock; default ON per spec.
    rmlx_kv_quant::planar_fused_qk::install_planar_fused_qk(cli.planar_fused_qk.enabled());

    // Resolve every kernel gate into the process-default DispatchPolicy before
    // any cache is built. This happens here, not inside `run_serve`, so that
    // every subcommand that runs inference — `bench`, `baseline`, `eval`,
    // `chat`, `generate` — sees the same kernel configuration the server does.
    // Resolving them only for `serve` made the measurement commands silently
    // benchmark a different kernel set than the one production runs.
    rmlx_core::set_dispatch_policy(commands::serve::resolve_dispatch_policy(
        cli.fused_qk,
        cli.sparse_attn,
        cli.turbo_flash,
        cli.turbo_flash_lock,
        cli.planar_flash_decode,
        cli.rot_k_fused,
        rmlx_core::DispatchPolicy::from_env(),
    ));

    // Install the process-wide panic hook.
    // Emits tracing::error! with structured fields (payload, location, thread,
    // optional backtrace) and writes a sidecar txt file to the logs dir.
    // The hook is idempotent (Once guard inside install()).
    panic_hook::install(rmlx_core::paths::logs_dir());

    // `pid` so a log can be attributed to the process that wrote it. Two runs
    // starting in the same second share a run-id, and therefore a log path;
    // a reader that has to know whose numbers it is holding needs an identity
    // the filename cannot give it.
    info!(
        version = env!("CARGO_PKG_VERSION"),
        %run_id,
        pid = std::process::id(),
        ?cli.cmd,
        "rmlx start"
    );

    // `rmlx metrics` subcommands are DB-admin only: they own their own
    // connection (possibly to a different DB via `--db`/`RMLX_METRICS_DB`)
    // and emit no per-event records, so they short-circuit before opening
    // an `EventRecorder` — otherwise concurrent test subprocesses contend
    // on the workspace `.rmlx/metrics/runs.db` lock.
    if let Cmd::Metrics(cmd) = cli.cmd {
        return metrics_dispatch(cmd);
    }

    // `rmlx profile …` is a pure file-read admin command — it touches no model
    // and opens no metrics recorder, so it short-circuits like `metrics`.
    if let Cmd::Profile { cmd } = &cli.cmd {
        return match cmd {
            ProfileCmd::List => commands::run_profile_list(),
        };
    }

    // `rmlx kv-calibrate` runs without opening the EventRecorder; the
    // GPU-loading recipes (`head_budget`, `softmax_mass`) acquire the
    // single-MLX claim internally and are mutually exclusive with a live
    // `rmlx serve`. Weight-norm recipes (turbo*) remain CPU-only and are safe
    // to co-run with serve.
    if let Cmd::KvCalibrate {
        model,
        recipe,
        out,
        prompts,
        mass_threshold,
        target_mass_budget_floor,
    } = &cli.cmd
    {
        return run_kv_calibrate(
            model,
            recipe,
            out.as_deref(),
            prompts.as_deref(),
            *mass_threshold,
            *target_mass_budget_floor,
        );
    }

    // D-class startup site: EventRecorder::open failure is fatal; tracing is
    // already initialised here, so we emit an error event before propagating.
    let sink = EventRecorder::open(&run_id).map_err(|e| {
        tracing::error!(error = %e, "D-class startup: metrics EventRecorder::open failed");
        anyhow::anyhow!("metrics open: {e}")
    })?;

    #[allow(
        clippy::unreachable,
        reason = "Cmd::Metrics, Cmd::Profile, and Cmd::KvCalibrate are handled by `return`-ing \
                  early blocks above; reaching these arms means the early-return guard was \
                  removed — a BUG"
    )]
    #[allow(
        clippy::match_same_arms,
        reason = "Metrics + Profile + KvCalibrate arms are unreachable!() guards documenting \
                  distinct early-return commands above; collapsing them would lose the \
                  Cmd-specific BUG narrative"
    )]
    match cli.cmd {
        Cmd::Metrics(_) => unreachable!("handled above"),
        Cmd::Profile { .. } => unreachable!("handled above"),
        Cmd::KvCalibrate { .. } => unreachable!("handled above"),
        Cmd::Serve {
            model,
            registry,
            profile,
            port,
            host,
            device,
            kv_quant,
            kv_preset,
            cache_type_k,
            cache_type_v,
            kv_bits,
            kv_group_size,
            max_ctx,
            idle_timeout_secs,
            prompt_cache_slots,
            draft_model,
            draft_kind,
            draft_block_size,
            max_tokens_cap,
            max_timeout_secs,
            require_smoke_probe,
            max_loaded_models,
            max_queue_depth,
            default_temperature,
            enable_thinking,
            image_max_tokens,
            kv_ssd_cache_gb,
            project,
            kv_ssd_global_gb,
            prompt_cache_ram_gb,
            paged_kv,
            paged_kv_page_tokens,
            prefix_index,
            adaptive_admission,
            ttft_target_ms,
            itl_target_ms,
            adaptive_prefill_chunk,
            whisper_model_path,
            whisper_tokenizer_path,
            tts_model_path,
            tts_tokenizer_path,
            mm_cache_bytes,
            session_cache_max_sessions,
            yarn_factor,
            yarn_original_max,
            kv_boundary_layers,
        } => {
            rmlx_models::kv_cache::install_kv_boundary(kv_boundary_layers)?;
            // load + merge the named profile (if any). Precedence is
            // CLI > profile > hard-coded default. Each bindable flag is `Option`
            // at the clap layer, so `cli.or(profile)` honours "flag not passed"
            // (None → take profile) vs "flag passed" (Some → CLI wins).
            let prof = match profile.as_deref() {
                Some(name) => Some(commands::profile::ProfilesFile::load()?.get(name)?.clone()),
                None => None,
            };
            let p = prof.as_ref();

            let model = model.or_else(|| p.and_then(|x| x.model.clone()));
            let registry = registry.or_else(|| p.and_then(|x| x.registry.clone()));
            let port = port
                .or_else(|| p.and_then(|x| x.port))
                .unwrap_or(DEFAULT_SERVE_PORT);
            let host = host
                .or_else(|| p.and_then(|x| x.host.clone()))
                .unwrap_or_else(|| DEFAULT_SERVE_HOST.to_string());
            let device = device
                .or_else(|| p.and_then(|x| x.device.clone()))
                .unwrap_or_else(|| DEFAULT_SERVE_DEVICE.to_string());
            let kv_quant = kv_quant
                .or_else(|| p.and_then(|x| x.kv_quant.clone()))
                .unwrap_or_else(|| DEFAULT_SERVE_KV_QUANT.to_string());
            let max_ctx = max_ctx.or_else(|| p.and_then(|x| x.max_ctx));
            // Idle keep-alive — CLI string > profile u64 (legacy) > unset.
            //
            // When unset everywhere, `run_serve` falls back to the 15 min default.
            // `Option::None` here signals "no CLI override" — the layered resolver wins.
            let idle_timeout_spec: Option<String> = idle_timeout_secs
                .or_else(|| p.and_then(|x| x.idle_timeout_secs.map(|n| n.to_string())));
            let prompt_cache_slots = prompt_cache_slots
                .or_else(|| p.and_then(|x| x.prompt_cache_slots))
                .unwrap_or(DEFAULT_PROMPT_CACHE_SLOTS);
            let draft_model = draft_model.or_else(|| p.and_then(|x| x.draft_model.clone()));
            // draft_kind / draft_block_size have no profile key; a profile's
            // draft_model runs at the kind its snapshot declares.
            let draft_kind: Option<rmlx_models::DraftKind> = draft_kind.map(Into::into);
            let max_tokens_cap = max_tokens_cap
                .or_else(|| p.and_then(|x| x.max_tokens_cap))
                .unwrap_or(DEFAULT_MAX_TOKENS_CAP);
            let max_timeout_secs = max_timeout_secs
                .or_else(|| p.and_then(|x| x.max_timeout_secs))
                .unwrap_or(DEFAULT_MAX_TIMEOUT_SECS);
            let max_loaded_models = max_loaded_models
                .or_else(|| p.and_then(|x| x.max_loaded_models))
                .unwrap_or(DEFAULT_MAX_LOADED_MODELS);
            let max_queue_depth = max_queue_depth
                .or_else(|| p.and_then(|x| x.max_queue_depth))
                .unwrap_or(DEFAULT_MAX_QUEUE_DEPTH);
            let default_temperature =
                default_temperature.or_else(|| p.and_then(|x| x.default_temperature));
            // `enable_thinking` has no profile key yet; CLI only.
            // Future: add to ServeProfile when profile support is needed.

            if let Some(name) = profile.as_deref() {
                info!(
                    profile = name,
                    %host, port, device = %device, kv_quant = %kv_quant,
                    "resolved serve config from profile (CLI flags override)"
                );
            }

            if model.is_some() && registry.is_some() {
                return Err(anyhow::anyhow!(
                    "--model and --registry are mutually exclusive"
                ));
            }
            // G4: validate --default-temperature at startup.
            if let Some(t) = default_temperature {
                if !(0.0..=2.0).contains(&t) {
                    return Err(anyhow::anyhow!(
                        "--default-temperature must be in [0.0, 2.0], got {t}"
                    ));
                }
                info!(default_temperature = t, "G4: --default-temperature set");
            }
            // log --enable-thinking at startup.
            if let Some(et) = enable_thinking {
                info!(enable_thinking = et, "--enable-thinking set");
            }
            // validate the SSD prompt-cache tier flags.
            if kv_ssd_cache_gb < 0.0 {
                return Err(anyhow::anyhow!(
                    "--kv-ssd-cache-gb must be >= 0 (0 = tier off), got {kv_ssd_cache_gb}"
                ));
            }
            if project.is_some() && kv_ssd_cache_gb <= 0.0 {
                return Err(anyhow::anyhow!(
                    "--project requires --kv-ssd-cache-gb > 0 (the SSD tier must be enabled)"
                ));
            }
            if kv_ssd_cache_gb > 0.0 {
                info!(
                    kv_ssd_cache_gb,
                    project = project.as_deref().unwrap_or("(model_id)"),
                    "SSD prompt-cache tier requested"
                );
            }
            // validate the new SSD / prompt-cache / paged-KV flags.
            if kv_ssd_global_gb < 0.0 {
                return Err(anyhow::anyhow!(
                    "--kv-ssd-global-gb must be >= 0 (0 = no global ceiling), got {kv_ssd_global_gb}"
                ));
            }
            if kv_ssd_global_gb > 0.0 && kv_ssd_cache_gb > kv_ssd_global_gb {
                tracing::warn!(
                    kv_ssd_cache_gb,
                    kv_ssd_global_gb,
                    "--kv-ssd-cache-gb exceeds --kv-ssd-global-gb; per-namespace ceiling clamped to global at startup"
                );
            }
            if let Some(g) = prompt_cache_ram_gb {
                if !g.is_finite() || g < 0.0 {
                    return Err(anyhow::anyhow!(
                        "--prompt-cache-ram-gb must be a finite non-negative number, got {g}"
                    ));
                }
            }
            // review MEDIUM-3: clap `value_enum` already rejects
            // garbage at parse-time (usage error mentioning the possible
            // values). The downstream wiring just converts the wrapper
            // into the model-crate type.
            let prefix_index_kind: rmlx_models::prefix_index::PrefixIndexKind = prefix_index.into();
            info!(?prefix_index_kind, "--prefix-index parsed");
            if paged_kv {
                // Reject --paged-kv with rot_k* cache type (rotation-based K is
                // not paged-compatible — Mixed quantized-SDPA dispatch).
                // This check runs early because cache_type_k is a raw string flag
                // unaffected by kv_preset resolution.
                if let Some(ctk) = cache_type_k.as_deref() {
                    let lc = ctk.to_ascii_lowercase();
                    if lc.starts_with("rot_k") {
                        return Err(anyhow::anyhow!(
                            "--paged-kv is incompatible with --cache-type-k {ctk} (rot_k* not paged-compatible)"
                        ));
                    }
                }
                if let Some(n) = paged_kv_page_tokens {
                    if n <= 0 {
                        return Err(anyhow::anyhow!(
                            "--paged-kv-page-tokens must be a positive integer, got {n}"
                        ));
                    }
                }
                // Note: bf16/unquantised rejection for --paged-kv is checked
                // AFTER kv_quant_final is resolved below. The old string-based
                // check on kv_quant here would miss --kv-preset fp16.
                info!(
                    page_tokens = ?paged_kv_page_tokens,
                    "--paged-kv requested"
                );
            }
            // Validate --image-max-tokens: zero is rejected (matches the HTTP
            // 400 the request path returns for image_max_tokens == 0). High
            // values are silently clamped downstream, consistent with the
            // request path which only errors on zero.
            if image_max_tokens == Some(0) {
                return Err(anyhow::anyhow!("--image-max-tokens must be > 0"));
            }
            // projects.toml loading + cap resolution wired in run_serve.
            // `draft_kind_flag`, not `draft_kind`: the resolved kind is
            // logged under that name by the generator once the snapshot is read.
            if let Some(dp) = draft_model.as_deref() {
                info!(
                    draft_model = %dp.display(),
                    draft_kind_flag = ?draft_kind,
                    draft_block_size = ?draft_block_size,
                    "speculative decoding — draft flags"
                );
            }
            // Resolve --cache-type-* against the model config when a single
            // --model is supplied. Registry-mode resolution is per-model and
            // happens inside the loader closure; passing --cache-type-* with
            // --registry is rejected here (per-arch resolution would need to
            // re-run for every model in the registry, out of scope for v0.0.1).
            //
            // --kv-preset pre-resolution. parse_kv_preset returns a
            // KvPresetArg which is either Resolved(KvQuant) or Auto;
            // resolve_preset_arg turns Auto into DEFAULT_KV_QUANT.
            let (dev, kv_quant_final, max_ctx_override) = if let Some(ref model_path) = model {
                if let Some(preset_arg) = kv_preset {
                    let max_ctx_override = parse_max_ctx(max_ctx)?;
                    let dev = parse_device(&device)?;
                    info!(device = %device, "rmlx serve: resolved device");
                    let cfg = rmlx_loader::load_config(model_path)
                        .map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
                    let preset_kq = resolve_preset_arg(preset_arg);
                    info!(kv_quant = ?preset_kq, "--kv-preset resolved");
                    let kq = commands::parse::resolve_kv_quant(&cfg, Some(preset_kq), None);
                    (dev, Some(kq), max_ctx_override)
                } else {
                    let (d, kq, ctx) = resolve_model_flags(
                        model_path,
                        &kv_quant,
                        cache_type_k.as_deref(),
                        cache_type_v.as_deref(),
                        max_ctx,
                        &device,
                        "rmlx serve",
                        kv_bits,
                        kv_group_size,
                    )?;
                    (d, Some(kq), ctx)
                }
            } else {
                // Registry mode: parse flags individually (no model to resolve against).
                // --kv-bits in registry mode is resolved directly (no arch validation).
                // fractional dispatch mirrors resolve_model_flags.
                //
                // `--kv-preset auto` used to be rejected here with exit 78 and
                // "auto-selection needs config.json to estimate model size".
                // That reason is gone: `resolve_preset_arg` reads a constant and
                // opens nothing, so there is nothing for a missing config.json
                // to prevent — and `--kv-quant auto`, the same constant under
                // another flag, was accepted on the same command line. Two
                // spellings of one default must not disagree about whether they
                // are allowed.
                let kv_quant_opt = if let Some(preset_arg) = kv_preset {
                    let preset_kq = resolve_preset_arg(preset_arg);
                    info!(kv_quant = ?preset_kq, "rmlx serve registry: --kv-preset applied");
                    Some(preset_kq)
                } else if let Some(bits) = kv_bits {
                    let gs = kv_group_size.unwrap_or(64);
                    let kq = if bits.fract() == 0.0 {
                        parse_kv_bits_combo(kv_bits_u8(bits)?, gs)?
                    } else {
                        parse_kv_bits_fractional(bits, gs)?
                    };
                    Some(kq)
                } else {
                    parse_kv_quant(&kv_quant)?
                };
                let cts_override = if kv_preset.is_some() || kv_bits.is_some() {
                    None
                } else {
                    build_cache_type_spec(cache_type_k.as_deref(), cache_type_v.as_deref())?
                };
                let max_ctx_override = parse_max_ctx(max_ctx)?;
                let dev = parse_device(&device)?;
                info!(device = %device, "rmlx serve: resolved device");
                if cts_override.is_some() {
                    tracing::error!(
                        "--cache-type-k/--cache-type-v requires --model; \
                         per-arch resolution does not apply to --registry mode"
                    );
                    eprintln!(
                        "error: --cache-type-k/--cache-type-v requires --model (not --registry)"
                    );
                    eprintln!("see docs/KV_CACHE.md for supported codecs and combinations");
                    std::process::exit(78);
                }
                (dev, kv_quant_opt, max_ctx_override)
            };

            // Validate --paged-kv against the fully resolved KvQuant
            // (post-preset resolution). This covers --kv-preset fp16 and any
            // other path that yields KvQuant::None (unquantised / bf16).
            if let Some(msg) =
                commands::parse::reject_paged_kv_without_store(paged_kv, kv_quant_final)
            {
                return Err(anyhow::anyhow!(msg));
            }

            // Acquire Metal claim for GPU runs; CPU-only skips.
            let _claim = acquire_claim_for_device(dev, port)?;
            // Build YARN override from CLI flags. None when either flag is absent.
            let yarn_override = yarn_factor.map(|factor| rmlx_models::qwen3::YarnOverride {
                factor,
                original_max: yarn_original_max.map_or(0.0, |v| v as f32),
            });
            run_serve(
                model.as_deref(),
                registry.as_deref(),
                &host,
                port,
                &device,
                kv_quant_final,
                max_ctx_override,
                idle_timeout_spec,
                prompt_cache_slots,
                draft_model.as_deref(),
                draft_kind,
                draft_block_size,
                max_tokens_cap,
                max_timeout_secs,
                require_smoke_probe,
                max_loaded_models,
                max_queue_depth,
                default_temperature,
                enable_thinking,
                kv_ssd_cache_gb,
                kv_ssd_global_gb,
                project,
                prompt_cache_ram_gb,
                paged_kv,
                paged_kv_page_tokens,
                prefix_index_kind,
                adaptive_admission,
                ttft_target_ms,
                itl_target_ms,
                adaptive_prefill_chunk,
                whisper_model_path,
                whisper_tokenizer_path,
                tts_model_path,
                tts_tokenizer_path,
                mm_cache_bytes,
                session_cache_max_sessions,
                yarn_override,
                image_max_tokens,
                &sink,
            )?;
        }
        Cmd::Chat {
            model,
            device,
            kv_quant,
            kv_preset,
            cache_type_k,
            cache_type_v,
            kv_bits,
            kv_group_size,
            max_ctx,
        } => {
            // Load config + run the cache-type resolver before any model
            // load. Even though `chat` is a stub today, validating the flag
            // combination here keeps the CLI failure semantics consistent.
            //
            // --kv-preset pre-resolution. resolve_preset_arg turns
            // KvPresetArg::Auto into DEFAULT_KV_QUANT.
            let (dev, _kv_quant_final, _max_ctx_override) = if let Some(preset_arg) = kv_preset {
                let max_ctx_override = parse_max_ctx(max_ctx)?;
                let dev = parse_device(&device)?;
                let cfg = rmlx_loader::load_config(&model)
                    .map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
                let preset_kq = resolve_preset_arg(preset_arg);
                info!(kv_quant = ?preset_kq, "--kv-preset resolved");
                let kq = commands::parse::resolve_kv_quant(&cfg, Some(preset_kq), None);
                (dev, kq, max_ctx_override)
            } else {
                resolve_model_flags(
                    &model,
                    &kv_quant,
                    cache_type_k.as_deref(),
                    cache_type_v.as_deref(),
                    max_ctx,
                    &device,
                    "rmlx chat",
                    kv_bits,
                    kv_group_size,
                )?
            };
            let _claim = acquire_claim_for_device(dev, SENTINEL_PORT)?;
            println!("rmlx chat   model={}  device={device}", model.display());
        }
        Cmd::Transcribe {
            audio,
            model,
            tokenizer,
            format,
            language,
            translate,
            output,
            device,
        } => {
            let dev = parse_device(&device)?;
            // ASR holds Metal; acquire the single-MLX claim like the other
            // model-loading subcommands.
            let _claim = acquire_claim_for_device(dev, SENTINEL_PORT)?;
            let args = commands::transcribe::TranscribeArgs {
                audio: &audio,
                model: &model,
                tokenizer: tokenizer.as_deref(),
                format: &format,
                language: &language,
                translate,
            };
            let rendered = commands::transcribe::run_transcribe(&args, dev)?;
            match output {
                Some(path) => {
                    std::fs::write(&path, rendered.as_bytes())
                        .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;
                    println!("wrote {}", path.display());
                }
                None => println!("{rendered}"),
            }
        }
        Cmd::Info {
            model,
            device,
            probe_forward,
            probe_smoke,
            kv_quant,
            kv_preset,
            cache_type_k,
            cache_type_v,
            kv_bits,
            kv_group_size,
            list_cache_types,
            max_ctx,
        } => {
            // `--list-cache-types` prints the §D1 codec table and exits
            // before any model load. clap's `required_unless_present` guarantees
            // `model` is Some whenever this flag is absent.
            if list_cache_types {
                print_cache_type_table();
                print_kv_quant_residency_table();
                return Ok(());
            }
            let model = model.expect("clap required_unless_present guarantees model is Some");
            // Always run the cache-type resolver (it loads config + fails
            // fast on invalid combos even when no probe is requested). The
            // resolved KvQuant is only handed downstream when a probe will
            // actually load the model.
            //
            // --kv-preset pre-resolution. resolve_preset_arg turns
            // KvPresetArg::Auto into DEFAULT_KV_QUANT.
            let (dev, kv_quant_final, max_ctx_override) = if let Some(preset_arg) = kv_preset {
                let max_ctx_override = parse_max_ctx(max_ctx)?;
                let dev = parse_device(&device)?;
                let cfg = rmlx_loader::load_config(&model)
                    .map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
                let preset_kq = resolve_preset_arg(preset_arg);
                info!(kv_quant = ?preset_kq, "--kv-preset resolved");
                let kq = commands::parse::resolve_kv_quant(&cfg, Some(preset_kq), None);
                (dev, kq, max_ctx_override)
            } else {
                resolve_model_flags(
                    &model,
                    &kv_quant,
                    cache_type_k.as_deref(),
                    cache_type_v.as_deref(),
                    max_ctx,
                    &device,
                    "rmlx info",
                    kv_bits,
                    kv_group_size,
                )?
            };
            let kv_quant_resolved = if probe_forward || probe_smoke {
                Some(kv_quant_final)
            } else {
                None
            };
            // Claim only when a probe is requested (probes use the MLX runtime).
            let _claim = if probe_forward || probe_smoke {
                Some(acquire_claim_for_device(dev, SENTINEL_PORT)?)
            } else {
                None
            };
            let exit_code = run_info(
                &model,
                probe_forward,
                probe_smoke,
                dev,
                kv_quant_resolved,
                max_ctx_override,
                &sink,
            )?;
            let code = exit_code.as_i32();
            if code != 0 {
                std::process::exit(code);
            }
        }
        Cmd::Healthcheck {
            registry,
            model,
            port,
            db,
            min_disk_gb,
            full,
            human,
        } => {
            // Resolve DB path using the same logic as `rmlx metrics`.
            let db_path = if let Some(p) = db {
                p
            } else if let Ok(env) = std::env::var("RMLX_METRICS_DB") {
                PathBuf::from(env)
            } else {
                rmlx_core::paths::metrics_db_path()
            };

            let exit_code = run_healthcheck(
                registry.as_deref(),
                model.as_deref(),
                port,
                &db_path,
                min_disk_gb,
                full,
                human,
            )?;
            std::process::exit(exit_code);
        }
        Cmd::Baseline {
            model,
            prompt,
            prompt_tokens,
            device,
            max_tokens,
            prompt_label,
            kv_quant,
            kv_preset,
            cache_type_k,
            cache_type_v,
            kv_bits,
            kv_group_size,
            max_ctx,
            max_prompt_tokens,
            allow_truncate,
            label: bench_label,
            record,
            git_sha,
            prompts_dir,
            yarn_factor,
            yarn_original_max,
            emit_token_ids,
            #[cfg(feature = "metal-capture")]
            gpu_capture,
            #[cfg(feature = "metal-capture")]
            gpu_capture_skip,
            #[cfg(feature = "metal-capture")]
            gpu_capture_steps,
            kv_boundary_layers,
        } => {
            rmlx_models::kv_cache::install_kv_boundary(kv_boundary_layers)?;
            refuse_to_measure_off_the_pin("rmlx baseline")?;

            // Arm the GPU-capture window before anything expensive happens: a
            // request that cannot be honoured must cost seconds, not a full
            // weight load followed by a failure at the first decode step.
            #[cfg(feature = "metal-capture")]
            let capture_requested = commands::gpu_capture::arm(
                gpu_capture.as_deref(),
                gpu_capture_skip,
                gpu_capture_steps,
                max_tokens,
            )?;

            let max_prompt_tokens = max_prompt_tokens.map(parse_max_prompt_tokens).transpose()?;
            // --kv-preset pre-resolution. resolve_preset_arg turns
            // KvPresetArg::Auto into DEFAULT_KV_QUANT.
            let (dev, kv_quant_resolved, max_ctx_override) = if let Some(preset_arg) = kv_preset {
                let max_ctx_override = parse_max_ctx(max_ctx)?;
                let dev = parse_device(&device)?;
                let cfg = rmlx_loader::load_config(&model)
                    .map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
                let preset_kq = resolve_preset_arg(preset_arg);
                info!(kv_quant = ?preset_kq, "--kv-preset resolved");
                let kq = commands::parse::resolve_kv_quant(&cfg, Some(preset_kq), None);
                (dev, kq, max_ctx_override)
            } else {
                resolve_model_flags(
                    &model,
                    &kv_quant,
                    cache_type_k.as_deref(),
                    cache_type_v.as_deref(),
                    max_ctx,
                    &device,
                    "rmlx baseline",
                    kv_bits,
                    kv_group_size,
                )?
            };
            let _claim = acquire_claim_for_device(dev, SENTINEL_PORT)?;

            // Resolve --prompt-tokens → canonical longctx file when present.
            // The prompts/ dir lives at the workspace root; locate it via the
            // --prompts-dir flag (env: RMLX_PROMPTS_DIR) or a cwd-walk.
            let prompts_root = resolve_prompts_root(prompts_dir);

            let (effective_prompt_path, prompt_id_opt, prompt_body_opt): (
                PathBuf,
                Option<String>,
                Option<serde_json::Value>,
            ) = match prompt_tokens {
                Some(n) => {
                    let (path, name) =
                        commands::baseline::resolve_prompt_tokens_file(&prompts_root, n)?;
                    // Read the canonical JSON for embedding in PromptRef::ByBody.
                    let raw = std::fs::read_to_string(&path)?;
                    let obj: serde_json::Value = serde_json::from_str(&raw)?;
                    let body = if commands::baseline::is_chat_fixture(&obj) {
                        obj["messages"].clone()
                    } else {
                        obj
                    };
                    (path, Some(name), Some(body))
                }
                None => (prompt, None, None),
            };

            // Derive prompt label from filename when not explicitly given.
            let label_str = if prompt_label.is_empty() {
                prompt_id_opt.clone().unwrap_or_else(|| {
                    effective_prompt_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("unknown")
                        .to_owned()
                })
            } else {
                prompt_label
            };

            let bench_label_ref = bench_label.as_deref();
            let prompt_id_ref = prompt_id_opt.as_deref();
            let git_sha_ref = git_sha.as_deref();
            let record_args = if record {
                Some(commands::baseline::BaselineRecordArgs {
                    label: bench_label_ref,
                    prompt_id: prompt_id_ref,
                    prompt_body: prompt_body_opt,
                    kv_quant: kv_quant_resolved,
                    git_sha: git_sha_ref,
                })
            } else {
                None
            };

            // Build YARN override from CLI flags. None when yarn_factor is absent.
            let yarn_override = yarn_factor.map(|factor| rmlx_models::qwen3::YarnOverride {
                factor,
                original_max: yarn_original_max.map_or(0.0, |v| v as f32),
            });
            let baseline_result = run_baseline(
                &model,
                &effective_prompt_path,
                &device,
                max_tokens,
                &run_id,
                &label_str,
                Some(kv_quant_resolved),
                max_ctx_override,
                max_prompt_tokens,
                allow_truncate,
                yarn_override,
                emit_token_ids,
                &sink,
                record_args,
            );
            // Always stop and report the capture, including when the run failed —
            // otherwise a live scope leaks and the trace is never finalised. The
            // run's own error still wins, since it is the root cause.
            #[cfg(feature = "metal-capture")]
            let capture_result = commands::gpu_capture::report(capture_requested);
            baseline_result?;
            #[cfg(feature = "metal-capture")]
            capture_result?;
        }
        Cmd::Bench {
            model,
            prompt,
            prompt_tokens,
            device,
            max_tokens,
            runs,
            warmup,
            kv_quant,
            kv_preset,
            cache_type_k,
            cache_type_v,
            kv_bits,
            kv_group_size,
            max_ctx,
            max_prompt_tokens,
            allow_truncate,
            json,
            prompts_dir,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            kv_boundary_layers,
        } => {
            rmlx_models::kv_cache::install_kv_boundary(kv_boundary_layers)?;
            let max_prompt_tokens = max_prompt_tokens.map(parse_max_prompt_tokens).transpose()?;
            // Same KV resolution ladder as `baseline`, so a cell benched here
            // and a cell recorded there name the same codec.
            refuse_to_measure_off_the_pin("rmlx bench")?;

            let (dev, kv_quant_resolved, max_ctx_override) = if let Some(preset_arg) = kv_preset {
                let max_ctx_override = parse_max_ctx(max_ctx)?;
                let dev = parse_device(&device)?;
                let cfg = rmlx_loader::load_config(&model)
                    .map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
                let preset_kq = resolve_preset_arg(preset_arg);
                info!(kv_quant = ?preset_kq, "--kv-preset resolved");
                let kq = commands::parse::resolve_kv_quant(&cfg, Some(preset_kq), None);
                (dev, kq, max_ctx_override)
            } else {
                resolve_model_flags(
                    &model,
                    &kv_quant,
                    cache_type_k.as_deref(),
                    cache_type_v.as_deref(),
                    max_ctx,
                    &device,
                    "rmlx bench",
                    kv_bits,
                    kv_group_size,
                )?
            };
            let _claim = acquire_claim_for_device(dev, SENTINEL_PORT)?;

            let prompts_root = resolve_prompts_root(prompts_dir);
            let (prompt_path, prompt_label) =
                commands::bench::resolve_prompt(&prompts_root, prompt, prompt_tokens)?;

            run_bench(commands::bench::BenchArgs {
                model,
                prompt: prompt_path,
                prompt_label,
                device: dev,
                max_tokens,
                runs,
                warmup,
                kv_quant: kv_quant_resolved,
                max_ctx: max_ctx_override,
                max_prompt_tokens,
                allow_truncate,
                json,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
            })?;
        }
        Cmd::Eval { cmd } => match cmd {
            EvalCmd::Ppl {
                model,
                text_file,
                ctx_window,
                stride,
                corpus,
                device,
                max_tokens,
                git_sha,
                kv_quant,
                kv_preset,
                cache_type_k,
                cache_type_v,
                kv_bits,
                kv_group_size,
                kv_boundary_layers,
            } => {
                // A KV codec is opt-in here: with none of these flags the
                // scorer keeps its cacheless full-window forward, which is what
                // every PPL row already in the DB was measured with. Asking for
                // one switches it to the teacher-forced path that runs the
                // cache the decode loop runs.
                let kv_requested = kv_quant.is_some()
                    || kv_preset.is_some()
                    || kv_bits.is_some()
                    || cache_type_k.is_some()
                    || cache_type_v.is_some();
                if kv_boundary_layers.is_some() && !kv_requested {
                    return Err(anyhow::anyhow!(
                        "--kv-boundary-layers needs a KV codec: the default scorer runs no \
                         per-layer cache, so the boundary counts would change nothing. Pass \
                         --kv-quant (or --kv-preset / --kv-bits) as well."
                    ));
                }
                rmlx_models::kv_cache::install_kv_boundary(kv_boundary_layers)?;
                // Same KV resolution ladder as `baseline` and `bench`, so a
                // codec scored here is the codec those two measure.
                let (dev, kv_quant_resolved) = if !kv_requested {
                    (parse_device(&device)?, None)
                } else if let Some(preset_arg) = kv_preset {
                    let dev = parse_device(&device)?;
                    let cfg = rmlx_loader::load_config(&model)
                        .map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
                    let preset_kq = resolve_preset_arg(preset_arg);
                    info!(kv_quant = ?preset_kq, "--kv-preset resolved");
                    (
                        dev,
                        Some(commands::parse::resolve_kv_quant(
                            &cfg,
                            Some(preset_kq),
                            None,
                        )),
                    )
                } else {
                    let (dev, kq, _) = resolve_model_flags(
                        &model,
                        kv_quant.as_deref().unwrap_or("auto"),
                        cache_type_k.as_deref(),
                        cache_type_v.as_deref(),
                        None,
                        &device,
                        "rmlx eval ppl",
                        kv_bits,
                        kv_group_size,
                    )?;
                    (dev, Some(kq))
                };
                // claim the Metal GPU before loading the model so a
                // running `rmlx serve` does not contend on the single-process
                // claim file. Mirrors `run_baseline`'s acquire pattern --
                // `SENTINEL_PORT` flags the CLI-side (non-HTTP) claim holder.
                let _claim = acquire_claim_for_device(dev, SENTINEL_PORT)?;
                run_ppl(
                    &model,
                    &text_file,
                    ctx_window,
                    stride,
                    &corpus,
                    &device,
                    max_tokens,
                    &run_id,
                    git_sha.as_deref(),
                    kv_quant_resolved,
                )?;
            }
        },
    }
    Ok(())
}

#[cfg(test)]
#[path = "main_tests.rs"]
mod tests;
