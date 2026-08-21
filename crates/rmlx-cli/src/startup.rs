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
    /// trace on rmlx crates — per-token, per-FFI, per-layer events.
    /// Use to debug single tokens / individual model / cache decisions.
    Verbose,
}

impl LogLevel {
    pub(crate) fn env_filter(self) -> &'static str {
        match self {
            LogLevel::Info => "info,rmlx=info",
            LogLevel::Debug => "debug,rmlx=debug",
            LogLevel::Verbose => "trace,rmlx=trace",
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
    // that costs one whole u32 code word plus one f32 scale per group whatever
    // the codebook width, so a 3-bit and a 4-bit member of the same family
    // occupy byte-identical space; planar spends an f32 per PAIR and is wider
    // still. None of the three is a compression format at this layout. Said
    // here because this listing is where the tags are chosen.
    println!();
    println!("note: the bit width in an iso_*/rotor_*/planar* name is its codebook, not");
    println!("      what it stores. None of the three is a compression format at the");
    println!("      layout it ships: each spends a whole u32 code word and an f32 scale");
    println!("      per group (planar, per PAIR), so all land ABOVE bf16's 16.00 bits");
    println!("      per value. At head_dim 128:");
    println!("        iso_k_3/4, iso_v_3/4         16.25   (16 + 32/head_dim)");
    println!("        rotor_k_3/4, rotor_v_3/4     21.75   ((64*ceil(D/3)+32)/D)");
    println!("        planar3, planar4, planar_k4  22.00   (at every head_dim)");
    println!("      so planar is the widest cell on this menu, not iso or rotor. Each");
    println!("      pair is byte-identical across its 3-bit and 4-bit member. The");
    println!("      iso/rotor rates are derived from the ring allocation; the rotor and");
    println!("      planar figures are also measured by the crate rate gate");
    println!("      (kv_rate_tests), which measures iso in its wider CPU-block form.");
    println!("      For which pairings actually build such a store see docs/KV_QUANT.md");
    println!("      \"Codec disposition\".");
}
