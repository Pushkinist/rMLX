// ---------------------------------------------------------------------------
// LoadPhases — per-load timing breakdown
// ---------------------------------------------------------------------------

use std::sync::Mutex;

/// Timing breakdown for one model load.
///
/// Captured in `load_model` and stored in `LAST_LOAD_PHASES` (global).
/// Read by `read_load_phases()` and surfaced via `/metrics/cache`.
///
/// ## Phase definitions
///
/// - `mmap_ms`: time to open + mmap all safetensors shards (pre-measurement via
///   `load_shard_index + ShardSet::open`). The arch loader re-opens the same files
///   hitting OS page cache, adding <1 ms. mmap(2) creates virtual mappings only —
///   real I/O (page faults) happens during tensor decode and is captured in `dequant_ms`.
/// - `dequant_ms`: time for the arch-specific loader to copy safetensors bytes into
///   MLX Array objects. MLX arrays are lazy — bytes are copied from mmap at
///   `Array::from_*` time; GPU dispatch is deferred until the first forward pass.
/// - `gpu_residency_ms`: always 0. MLX does not eagerly push arrays to GPU during
///   loading (dispatch is deferred). Measuring this would require calling mlx_eval
///   inside the loader, which was not done to avoid changing load semantics.
/// - `first_kernel_ready_ms`: time for a no-op warmup dispatch (empty Array to_bytes)
///   immediately after the arch loader returns. Captures MLX JIT compile + Metal
///   pipeline-state creation latency for the first kernel.
/// - `total_load_ms`: wall-clock time from `load_model` entry to exit.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed struct — five load-phase timer fields; adding a timer requires updating load_model and read_load_phases"
)]
#[derive(Debug, Clone, Default)]
pub struct LoadPhases {
    /// Wall-clock milliseconds spent in the mmap + safetensors parsing phase.
    pub mmap_ms: u64,
    /// Wall-clock milliseconds spent dequantizing or copying weight tensors.
    pub dequant_ms: u64,
    /// Wall-clock milliseconds until all weight arrays are resident on GPU.
    pub gpu_residency_ms: u64,
    /// Wall-clock milliseconds until the first Metal kernel dispatch completes.
    pub first_kernel_ready_ms: u64,
    /// Total wall-clock milliseconds for the entire `load_model` call.
    pub total_load_ms: u64,
}

/// Most-recent load-phase timings. Written by `load_model`; read by
/// `read_load_phases()`. Zero-initialised until the first load completes.
pub(super) static LAST_LOAD_PHASES: Mutex<LoadPhases> = Mutex::new(LoadPhases {
    mmap_ms: 0,
    dequant_ms: 0,
    gpu_residency_ms: 0,
    first_kernel_ready_ms: 0,
    total_load_ms: 0,
});

/// Read the load-phase timings from the most recent `load_model` call.
///
/// Returns `None` when no model has been loaded yet (all fields are 0).
pub fn read_load_phases() -> Option<LoadPhases> {
    let guard = LAST_LOAD_PHASES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if guard.total_load_ms == 0 {
        None
    } else {
        Some(guard.clone())
    }
}
