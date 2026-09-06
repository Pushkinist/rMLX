//! Startup utilities: tracing init, log rotation, and info-table rendering.
//!
//! Extracted from `main.rs` to reduce the entry-point file size. All items
//! here are `pub(crate)`; none cross the crate boundary.

// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::PathBuf;

use anyhow::Result;
use tracing_appender::non_blocking::WorkerGuard;

// ---------------------------------------------------------------------------
// LogLevel — clap ValueEnum for the --log flag
// ---------------------------------------------------------------------------

/// Log verbosity preset. Mapped to a default `EnvFilter` at startup.
/// Passing `--log` does NOT override an explicit `RUST_LOG` — that escape
/// hatch is preserved for power users.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum LogLevel {
    /// info on every crate (default). Per-layer / per-token trace! stays off.
    #[default]
    Info,
    /// debug on rmlx crates. Per-token trace! stays off.
    Debug,
    /// trace on rmlx crates — per-token, per-FFI, per-layer events, and the
    /// speculative round loops' charged phase split. Everything else stays at
    /// info: a bare `trace` here sets the *global* default, which puts every
    /// dependency's trace events in the log and satisfies engine-side
    /// `tracing::enabled!` checks that were meant to be opt-in by target.
    /// Use to debug single tokens / individual model / cache decisions.
    Verbose,
}

impl LogLevel {
    pub(crate) fn env_filter(self) -> &'static str {
        match self {
            LogLevel::Info => "info,rmlx=info",
            LogLevel::Debug => "debug,rmlx=debug",
            LogLevel::Verbose => "info,rmlx=trace",
        }
    }
}

// ---------------------------------------------------------------------------
// MetricsArg — clap ValueEnum for the --metrics flag
// ---------------------------------------------------------------------------

/// How much telemetry this process writes to `runs.db`.
///
/// Resolved once at startup into `rmlx_metrics::mode`, which every writer then
/// consults — no per-call-site toggle. Disables *writing* only: the `rmlx
/// metrics` read commands (`best`, `export`, `query`, …) work in every mode,
/// as do the explicitly user-invoked `record` / `migrate` writes.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum MetricsArg {
    /// No DB writes at all. The drainer task is never spawned and the SQLite
    /// file is never opened or created.
    Off,
    /// Runtime `events` only — no bench `observations`.
    Events,
    /// Everything (default).
    #[default]
    Full,
}

impl MetricsArg {
    pub(crate) fn mode(self) -> rmlx_metrics::mode::MetricsMode {
        use rmlx_metrics::mode::MetricsMode;
        match self {
            MetricsArg::Off => MetricsMode::Off,
            MetricsArg::Events => MetricsMode::Events,
            MetricsArg::Full => MetricsMode::Full,
        }
    }
}

// ---------------------------------------------------------------------------
// init_tracing
// ---------------------------------------------------------------------------

/// Set up tracing to both stderr (human) and `<RMLX_HOME>/logs/<run-id>.jsonl`
/// (machine-readable JSON). Returns the worker guard that flushes on drop;
/// it must outlive `main`.
///
/// Filter precedence: `RUST_LOG` (escape hatch) > `--log` flag preset.
/// The default size-cap rotation sweep runs before the appender is opened
/// so the newly-created file is never the one we delete.
///
/// `log_cap_mb` is the resolved value of `--log-cap-mb` (default 100). The
/// caller is responsible for supplying the final value; no env read happens here.
pub(crate) fn init_tracing(run_id: &str, level: LogLevel, log_cap_mb: u64) -> Result<WorkerGuard> {
    use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let logs_dir = rmlx_core::paths::logs_dir();
    rotate_logs(&logs_dir, log_cap_mb);

    let file_appender = tracing_appender::rolling::never(&logs_dir, format!("{run_id}.jsonl"));
    let (nb_writer, guard) = tracing_appender::non_blocking(file_appender);

    // `RUST_LOG` wins when set (compat with existing scripts + power-users).
    // Otherwise the `--log` preset selects the filter string.
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level.env_filter()));

    // Logs go to stderr so commands that print data on stdout (e.g.
    // `rmlx metrics export --markdown > file`) are not polluted by tracing.
    let stdout_layer = fmt::layer()
        .with_target(true)
        .compact()
        .with_writer(std::io::stderr);
    let file_layer = fmt::layer().json().with_writer(nb_writer);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer)
        .with(file_layer)
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing init: {e}"))?;

    Ok(guard)
}

// ---------------------------------------------------------------------------
// rotate_logs
// ---------------------------------------------------------------------------

/// Delete oldest `*.jsonl` files in `dir` until total bytes ≤ cap.
/// Best-effort: any FS error during scan or delete is logged at WARN and
/// swallowed — log rotation must never bring down the server. Runs once at
/// startup, before the new run's appender is opened (so the in-flight file
/// is never a candidate for deletion).
///
/// `cap_mb` is the caller-resolved value (from `--log-cap-mb`, default 100).
pub(crate) fn rotate_logs(dir: &std::path::Path, cap_mb: u64) {
    let cap_bytes = cap_mb.saturating_mul(1024 * 1024);

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    let mut files: Vec<(PathBuf, u64, std::time::SystemTime)> = Vec::new();
    for ent in entries.flatten() {
        let p = ent.path();
        if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Ok(meta) = ent.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
        files.push((p, meta.len(), mtime));
    }

    let total: u64 = files.iter().map(|(_, sz, _)| sz).sum();
    if total <= cap_bytes {
        return;
    }

    // Sort oldest-first so we can pop the head of the queue.
    files.sort_by_key(|(_, _, mt)| *mt);

    let mut over = total - cap_bytes;
    for (path, size, _) in files {
        if over == 0 {
            break;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {
                tracing::debug!(
                    path = %path.display(),
                    size,
                    "log rotation: dropped oldest"
                );
                over = over.saturating_sub(size);
            }
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "log rotation: failed to delete"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// print_cache_type_table
// ---------------------------------------------------------------------------

/// Print the full §D1 KV cache codec table to stdout.
///
/// Self-contained: no external file dependency. Columns: tag, codec, bits,
/// group, sides. Triggered by `rmlx info --list-cache-types`.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported variants; exhaustive expansion would require updating on every new variant"
)]
pub(crate) fn print_cache_type_table() {
    use rmlx_kv_quant::storage::ring_bits_per_value;
    use rmlx_models::kv_cache::CacheType;
    // Header.
    println!(
        "{:<10} {:<28} {:<5} {:<6} sides",
        "tag", "codec", "bits", "group"
    );
    println!(
        "{:<10} {:<28} {:<5} {:<6} -----",
        "---", "-----", "----", "-----"
    );
    for ct in CacheType::all() {
        #[allow(
            clippy::match_same_arms,
            reason = "CacheType variants enumerated separately for documentation clarity; \
                      collapsing affine families into one arm would obscure the per-bit codec mapping"
        )]
        let codec = match ct {
            CacheType::Auto => "auto (engine picks)",
            CacheType::Bf16 => "bf16 (unquantized)",
            CacheType::Q8G128 => "rMLX MSL affine",
            CacheType::Q8G64 | CacheType::Q8G32 => "MLX affine",
            CacheType::Q6G64 | CacheType::Q5G64 => "MLX affine",
            CacheType::Q4G128 | CacheType::Q4G64 | CacheType::Q4G32 => "MLX affine",
            CacheType::Q3G64 => "MLX affine",
            CacheType::Q2G64 => "MLX affine (V-only)",
            CacheType::Tq4 => "TurboQuant rotation",
            CacheType::Planar4 => "PlanarQuant rotation (4-bit)",
            CacheType::Planar3 => "PlanarQuant rotation (3-bit)",
            CacheType::PlanarK4 => "PlanarQuant K-rotation",
            CacheType::RotK => "Hadamard K-rotation",
            CacheType::Iso3 => "IsoQuant rotation (3-bit)",
            CacheType::Iso4 => "IsoQuant rotation (4-bit)",
            CacheType::Rotor3 => "Clifford rotor3 (3-bit)",
            CacheType::Rotor4 => "Clifford rotor4 (4-bit)",
            CacheType::Turbo3Tcq => "TurboQuant + Viterbi trellis (3-bit)",
            CacheType::Turbo2Tcq => "TurboQuant + Viterbi trellis (2-bit)",
            CacheType::IsoK3 => "IsoQuant K-rotation (3-bit)",
            CacheType::IsoK4 => "IsoQuant K-rotation (4-bit)",
            CacheType::RotorK3 => "Clifford rotor K-side (3-bit)",
            CacheType::RotorK4 => "Clifford rotor K-side (4-bit)",
            // Symmetric WHT-3 K+V.
            CacheType::TurboSym3 => "TurboQuant sym K+V (3-bit)",
        };
        let bits = match ct.bits() {
            Some(b) => format!("{b}"),
            None => "—".to_string(),
        };
        let group = match ct.group_size() {
            Some(g) => format!("{g}"),
            None => "—".to_string(),
        };
        // §D6.3 — rotation codecs are V-only; bf16 + Q8G128 + Auto + MLX-affine
        // codecs cover both sides (subject to §D6 invariants on the V side).
        #[allow(
            clippy::match_same_arms,
            reason = "TurboSym3 listed explicitly for documentation clarity; \
                      collapsing into wildcard would obscure that it covers K+V"
        )]
        let sides = match ct {
            // q2_g64 is V-side only (2-bit K gated in combo_to_kv_quant).
            // Planar3 is V-side only like Planar4.
            // Iso3 is V-side only.
            CacheType::Tq4
            | CacheType::Planar4
            | CacheType::Planar3
            | CacheType::Iso3
            | CacheType::Iso4
            | CacheType::Rotor3
            | CacheType::Rotor4
            | CacheType::Turbo3Tcq
            | CacheType::Turbo2Tcq
            | CacheType::Q2G64 => "V",
            // K-side rotation codecs: rot_k (RotK), planar_k4 (PlanarK4),
            // iso_k_3 / iso_k_4, rotor_k_3 / rotor_k_4.
            CacheType::RotK
            | CacheType::PlanarK4
            | CacheType::IsoK3
            | CacheType::IsoK4
            | CacheType::RotorK3
            | CacheType::RotorK4 => "K",
            CacheType::TurboSym3 => "K, V",
            _ => "K, V",
        };
        println!(
            "{:<10} {:<28} {:<5} {:<6} {}",
            ct.tag(),
            codec,
            bits,
            group,
            sides,
        );
    }
    // The nominal width in a rotation codec's name is its codebook, not what
    // the store spends. The iso and rotor tags decode from a shared GPU ring
    // that costs one whole u32 code word plus one sideband scale per group
    // whatever the codebook width, so a 3-bit and a 4-bit member of the same
    // family occupy byte-identical space; planar spends an f32 per PAIR and is
    // wider than either. The two ring figures are computed from the ring's own
    // `ring_bits_per_value`, the single producer of a stored-rate figure, so
    // narrowing the sideband moves this listing without anyone editing it.
    // Said here because this listing is where the tags are chosen.
    let head_dim: u64 = 128;
    let iso_rate = ring_bits_per_value(head_dim, head_dim / 4);
    let rotor_rate = ring_bits_per_value(head_dim, head_dim.div_ceil(3));
    println!();
    println!("note: the bit width in an iso_*/rotor_*/planar* name is its codebook, not");
    println!("      what it stores: each spends a whole u32 code word per group (planar,");
    println!("      per PAIR), so the 3-bit and the 4-bit member of a family are");
    println!("      byte-identical. Stored rate at head_dim {head_dim}, against bf16's 16.00:");
    println!("        iso_k_3/4, iso_v_3/4 on the ring    {iso_rate:6.3}  (12 + 16/head_dim)");
    println!("        rotor_k_3/4, rotor_v_3/4            {rotor_rate:6.3}  ((48*ceil(D/3)+16)/D)");
    println!("        planar3, planar4, planar_k4         22.000  (at every head_dim)");
    println!("      Only iso is under bf16, and only because its scale and norm planes");
    println!("      are 16 bits: a sideband change cannot fix rotor's u32-per-3-values");
    println!("      code cadence, and planar is the widest cell on this menu.");
    println!("      The ring is what a served request holds for iso_k_*/rotor_k_* and");
    println!("      for both axes of iso3_sym/iso4_sym/rotor3_sym/rotor4_sym. An");
    println!("      iso_v_3/iso_v_4 chosen on its own (K stays q8) builds NO packed");
    println!("      store at all and decodes from the bf16 mirror — it stores nothing");
    println!("      and saves nothing. The rotor and planar figures are also measured");
    println!("      from real encoder output by the crate rate gate (kv_rate_tests).");
    println!("      For which pairings actually build a store see docs/KV_QUANT.md");
    println!("      \"Codec disposition\".");
}

// ---------------------------------------------------------------------------
// print_kv_quant_residency_table
// ---------------------------------------------------------------------------

/// Head dimension the `--kv-quant` residency listing is computed at.
///
/// The iso and rotor rings carry a per-row sideband term (`16 / head_dim` at a
/// bf16 sideband), so their rate — and only theirs — moves with this. Every
/// other cadence is per-value and this figure is head-dim-invariant.
const RESIDENCY_HEAD_DIM: u64 = 128;

/// KV heads and token count the listing is computed at. The ratio is invariant
/// in both: every term in the byte model is linear in `seq * kv_heads`.
const RESIDENCY_KV_HEADS: u64 = 8;
const RESIDENCY_SEQ: u64 = 4096;

/// Marker that opens the residency listing. The disposition gate binds the
/// help's `rmlx info --list-cache-types` pointer to a live call site; this is
/// what a test asserts the rendered listing starts with.
const RESIDENCY_TABLE_HEADING: &str =
    "--kv-quant codecs — resident KV one GLOBAL (full-attention) layer holds,";

/// Render what each `--kv-quant` codec holds on a global layer, computed.
///
/// This is the **one** place a resident-KV ratio is published to an operator.
/// Every figure comes out of
/// [`rmlx_kv_quant::KvQuant::estimated_resident_bytes_per_layer`], the same
/// producer the resolve-time net-benefit warning and the crate's byte-model
/// gates read, so narrowing a store, dropping a mirror or adding a codec moves
/// this listing without anyone editing it. `--kv-quant`'s help quotes no ratio
/// and points here instead; `make check-kv-codec-disposition` keeps it that way
/// and binds the pointer to [`print_kv_quant_residency_table`]'s call site.
///
/// Both topologies are rendered because `shares_kv` is not a codec property. The
/// `mixed_*` / `rot_k_*` bf16 K/V mirror is what a cross-layer-KV architecture's
/// consumer layers read, so it is retained there and elided everywhere else —
/// the same codec is well under bf16 on one stack and well over it on the
/// other, and a single column could only be right about one of them.
///
/// Separate from the printing so `startup_tests.rs` can hold the rendered text
/// to the type: a listing checked only by eye is a listing that can quietly
/// stop covering a codec.
pub(crate) fn kv_quant_residency_table() -> String {
    use std::fmt::Write as _;

    use rmlx_kv_quant::ALL_KV_QUANTS;

    let bf16 = rmlx_kv_quant::KvQuant::None.estimated_resident_bytes_per_layer(
        RESIDENCY_SEQ,
        RESIDENCY_HEAD_DIM,
        RESIDENCY_KV_HEADS,
        false,
    );
    // Two axes (K and V) per stored position.
    let values = RESIDENCY_SEQ * RESIDENCY_HEAD_DIM * RESIDENCY_KV_HEADS * 2;

    let mut out = String::new();
    // Every `write!` into a `String` is infallible; the `Result` is discarded
    // once here rather than at each call site.
    let _ = writeln!(out);
    let _ = writeln!(out, "{RESIDENCY_TABLE_HEADING}");
    let _ = writeln!(
        out,
        "computed at head_dim {RESIDENCY_HEAD_DIM} from the engine's own byte model. A windowed"
    );
    let _ = writeln!(
        out,
        "(SWA) layer runs the bf16 rotating ring whatever codec is set, so a"
    );
    let _ = writeln!(
        out,
        "whole-model figure on such an arch is this diluted toward 1.00x."
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "{:<26} {:^19}  {:^19}  disposition",
        "codec", "dense arch", "shared-KV arch"
    );
    let _ = writeln!(
        out,
        "{:<26} {:^19}  {:^19}  -----------",
        "-----", "bits/value  x bf16", "bits/value  x bf16"
    );
    for &q in ALL_KV_QUANTS {
        let cell = |shares_kv: bool| {
            let bytes = q.estimated_resident_bytes_per_layer(
                RESIDENCY_SEQ,
                RESIDENCY_HEAD_DIM,
                RESIDENCY_KV_HEADS,
                shares_kv,
            );
            let bits_per_value = (bytes * 8) as f64 / values as f64;
            let ratio = bytes as f64 / bf16 as f64;
            format!("{bits_per_value:10.3}  {ratio:5.3}x")
        };
        let disposition = if q == rmlx_kv_quant::KvQuant::None {
            "unquantised reference"
        } else if q.materialises_packed_store() {
            "runs its codec"
        } else {
            "INERT — no store built"
        };
        let _ = writeln!(
            out,
            "{:<26} {:>19}  {:>19}  {}",
            q.to_string(),
            cell(false),
            cell(true),
            disposition
        );
    }
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "A parametric name (mixed_*, rot_k_v*, rotor_k_*_asym_*) is listed at one"
    );
    let _ = writeln!(
        out,
        "representative width; pass your own and the arithmetic follows it."
    );
    let _ = writeln!(
        out,
        "An INERT codec decodes from the bf16 mirror on both axes, so its packed"
    );
    let _ = writeln!(
        out,
        "store is never built and it holds exactly what bf16 holds."
    );
    out
}

/// Print [`kv_quant_residency_table`] to stdout. Triggered by
/// `rmlx info --list-cache-types`, which is where the `--kv-quant` help sends
/// an operator who wants a resident-KV figure.
pub(crate) fn print_kv_quant_residency_table() {
    print!("{}", kv_quant_residency_table());
}

#[cfg(test)]
#[path = "startup_tests.rs"]
mod startup_tests;
