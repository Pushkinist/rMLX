//! Shared test utilities for `rmlx-kv-quant`.
//!
//! # Vectorized-vs-scalar parity harness
//!
//! [`vectorized_parity_check`] runs a CPU-scalar path and a GPU/MSL path on
//! the same input and asserts that the max-abs-error between their outputs
//! does not exceed `tol`.  All four per-codec parity tests use this helper.
//!
//! # `RMLX_SKIP_GPU` env-var skip
//!
//! [`skip_if_no_gpu_env`] returns `true` when `RMLX_SKIP_GPU=1` is set.
//! Parity tests call it at the top of the test body:
//!
//! ```ignore
//! #[test]
//! #[ignore = "GPU Metal context — run explicitly"]
//! fn my_parity_test() {
//!     if crate::test_utils::skip_if_no_gpu_env() { return; }
//!     // ... test body ...
//! }
//! ```
//!
//! `RMLX_SKIP_GPU=1` wins even when `--include-ignored` is passed; `#[ignore]`
//! still gates the default test run.
//!
//! # Per-codec tolerance policy
//!
//! | Codec family | Tolerance | Rationale |
//! |---|---|---|
//! | Integer / packed codes (bit-level) | exact | GPU layout == CPU pack |
//! | TurboQuant V4 (codebook lookup) | 5e-3 | f32 rounding in lookup path |
//! | PlanarQuant V4 (codebook + rotation) | 5e-3 | f32 rounding in lookup path |
//! | K8VTurbo3 V (3-bit codebook lookup) | 1e-3 | tighter: 3-bit centroids smaller |
//! | rot_k FWHT + affine q8 | 0.10 | one 8-bit quant step for D=128 FWHT range |
//! | q8_0 group-128 affine | 5e-3 | f32 rounding in min/max scan |

/// Process-global lock for every test in this binary that touches the
/// environment — as a **writer or a reader**.
///
/// The granularity is the whole environment, not one variable: `setenv` is UB
/// against a concurrent `getenv` of *any* key, so a per-variable lock would be
/// unsound. Acquire it via [`env_lock`].
///
/// Hold it for the **entire** test body (set → build → encode/decode → assert →
/// clear), not just across the mutation. A reader that samples an env-backed
/// gate without it observes another test's in-flight mutation and fails
/// intermittently — an unexplained flake that teaches everyone to re-run rather
/// than investigate, and launders real failures in the process.
///
/// `rmlx-kv-ssd` keeps its own lock: it is a separate crate, so its tests run in
/// a separate binary (separate process) and share no environment with this one.
///
/// # Usage
///
/// The guard restores the managed keys when it drops, so a test sets what it
/// needs and does not clean up:
///
/// ```ignore
/// #[test]
/// fn my_test() {
///     let _guard = crate::test_utils::env_lock();
///     // SAFETY: env lock held — no concurrent env reader/writer.
///     unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0"); }
///     // ... test body; no restore needed, `_guard` handles it ...
/// }
/// ```
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Environment keys the test suite mutates, snapshotted and restored by
/// [`EnvGuard`].
///
/// Only `RMLX_ROTOR_QJL` qualifies: it is the one process-global that tests
/// write, and it is deliberately not `OnceLock`-latched, so a leaked value
/// changes what every later test observes. Keys that tests merely *read*
/// (`RMLX_SKIP_GPU`, `RMLX_TURBO_FLASH`) need the lock, not restoration.
const MANAGED_ENV_KEYS: [&str; 1] = ["RMLX_ROTOR_QJL"];

/// Holds [`ENV_LOCK`] and restores [`MANAGED_ENV_KEYS`] to their pre-acquisition
/// values on drop — including while unwinding from a failed assertion.
///
/// That last part is the point. Every writer in this suite is shaped
/// `set_var` → `assert!` → restore, so a failing assertion used to skip its own
/// restore and leak the value into every subsequent test; the next reader then
/// failed with "test assumes the default QJL-off state" and buried the assertion
/// that actually broke. Restoring in `Drop` makes the writers unwind-safe
/// without each of them having to be.
pub(crate) struct EnvGuard {
    /// Dropped after `Drop::drop` returns, so the restore below runs while the
    /// lock is still held.
    _lock: std::sync::MutexGuard<'static, ()>,
    prev: [(&'static str, Option<String>); MANAGED_ENV_KEYS.len()],
}

impl Drop for EnvGuard {
    #[allow(unsafe_code, reason = "env restore under the lock this guard holds")]
    fn drop(&mut self) {
        for (key, prev) in &self.prev {
            // SAFETY: `_lock` is still held (fields drop after this fn returns),
            // so there is no concurrent env reader or writer in this binary.
            match prev {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
}

/// Acquire [`ENV_LOCK`], snapshotting [`MANAGED_ENV_KEYS`] for restoration.
///
/// Poisoning is recovered rather than propagated. That is sound *because* of
/// the `Drop` restore above: the panicking test's guard put the managed keys
/// back on its way out, so the environment this caller inherits is the one it
/// would have seen had that test passed. Propagating instead would replace every
/// later test's real failure with "env lock poisoned", hiding the one that broke
/// behind a cascade.
pub(crate) fn env_lock() -> EnvGuard {
    let lock = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    EnvGuard {
        _lock: lock,
        prev: MANAGED_ENV_KEYS.map(|key| (key, std::env::var(key).ok())),
    }
}

/// Whether an `RMLX_SKIP_GPU` value means "skip": strictly `Some("1")`.
///
/// Split out from [`skip_if_no_gpu_env`] so the membership rule can be tested
/// without touching the process environment. Setting `RMLX_SKIP_GPU` to probe
/// this would be unsound: [`skip_if_no_gpu_env`] is read at the top of every
/// GPU test in this binary, and none of them hold the lock, so a transient
/// write can flip a live test between "run" and "silent skip" — a false green,
/// or a Metal test un-ignored into a parallel run.
pub(crate) fn skip_value_means_skip(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Returns `true` if the `RMLX_SKIP_GPU` environment variable is set to `"1"`.
///
/// Parity tests call this at the top of their body as an additional opt-out
/// beyond `#[ignore]`.  When `RMLX_SKIP_GPU=1`, the test exits silently
/// without touching the GPU, even when run with `--include-ignored`.
pub(crate) fn skip_if_no_gpu_env() -> bool {
    skip_value_means_skip(std::env::var("RMLX_SKIP_GPU").ok().as_deref())
}

/// Run `cpu_path` and `msl_path` on `input` and assert max-abs-error ≤ `tol`.
///
/// # Arguments
///
/// * `cpu_path` — CPU scalar encode+decode, takes a `&[f32]` slice, returns
///   `Vec<f32>` of dequantized values.
/// * `msl_path` — GPU/MSL encode+decode, same signature.
/// * `input` — raw f32 input values (flat, in whatever shape the codec expects).
/// * `tol` — max-abs-error tolerance (see per-codec table in module docs).
/// * `name` — codec name for the panic message.
///
/// # Panics
///
/// Panics if the two outputs differ in length, or if any element pair exceeds
/// `tol`. On failure the panic message includes `name`, the observed error,
/// and the tolerance.
///
/// # Unit-test coverage
///
/// `identity_codec_parity_check_passes` in `test_utils_tests.rs` verifies the
/// helper with a trivial identity codec (no error) and a 1e-7 tolerance.
pub(crate) fn vectorized_parity_check<F1, F2>(
    cpu_path: F1,
    msl_path: F2,
    input: &[f32],
    tol: f32,
    name: &str,
) where
    F1: FnOnce(&[f32]) -> Vec<f32>,
    F2: FnOnce(&[f32]) -> Vec<f32>,
{
    let cpu_out = cpu_path(input);
    let msl_out = msl_path(input);

    assert_eq!(
        cpu_out.len(),
        msl_out.len(),
        "[{name}] CPU and MSL outputs have different lengths: {} vs {}",
        cpu_out.len(),
        msl_out.len(),
    );

    let max_err = cpu_out
        .iter()
        .zip(msl_out.iter())
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    if max_err > tol {
        let first_diff_idx = cpu_out
            .iter()
            .zip(msl_out.iter())
            .position(|(&c, &g)| (c - g).abs() > tol)
            .expect("position must exist when max_err > tol");
        panic!(
            "[{name}] CPU vs MSL max-abs-error {max_err:.2e} exceeds tolerance {tol:.2e}. \
             First divergence near index {first_diff_idx}.",
        );
    }
}

// ── Cosine-similarity gate ───────────────────────────────────────────────────

/// Pinned LCG seed for all codec cosine tests.
///
/// Never replace with `thread_rng` — tests must be deterministic.
pub(crate) const TEST_SEED: u64 = 0x0000_00C0_FFEE_BEEF;

/// Summary statistics from [`cosine_similarity_per_row`].
pub(crate) struct CosineStats {
    /// Mean cosine similarity across all rows.
    pub mean: f32,
    /// Minimum cosine similarity across all rows.
    pub min: f32,
    /// Number of rows processed.
    pub n_rows: usize,
}

/// Compute per-row cosine similarity between `reference` and `decoded`.
///
/// Both slices must have the same length and that length must be a non-zero
/// multiple of `head_dim`. Each row of `head_dim` f32 elements is treated as
/// one vector; cosine similarity `dot(a,b) / (||a|| × ||b||)` is accumulated
/// in f64 for precision. If either vector in a row is all-zero the cosine for
/// that row is defined as `1.0` (perfect match — both are zero).
///
/// Returns [`CosineStats`] with `mean`, `min`, and `n_rows`.
///
/// # Panics
///
/// Panics (test-only) if `reference.len() != decoded.len()`, `head_dim == 0`,
/// or `head_dim` does not evenly divide `reference.len()`.
pub(crate) fn cosine_similarity_per_row(
    reference: &[f32],
    decoded: &[f32],
    head_dim: usize,
) -> CosineStats {
    assert_eq!(
        reference.len(),
        decoded.len(),
        "cosine_similarity_per_row: reference and decoded lengths differ ({} vs {})",
        reference.len(),
        decoded.len(),
    );
    assert!(
        head_dim > 0 && reference.len().is_multiple_of(head_dim),
        "cosine_similarity_per_row: head_dim={head_dim} does not evenly divide len={}",
        reference.len(),
    );

    let n_rows = reference.len() / head_dim;
    let mut sum_cos = 0.0f64;
    let mut min_cos = f64::MAX;

    for row in 0..n_rows {
        let start = row * head_dim;
        let a = &reference[start..start + head_dim];
        let b = &decoded[start..start + head_dim];

        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for (&ai, &bi) in a.iter().zip(b.iter()) {
            let af = f64::from(ai);
            let bf = f64::from(bi);
            dot += af * bf;
            norm_a += af * af;
            norm_b += bf * bf;
        }

        let denom = norm_a.sqrt() * norm_b.sqrt();
        // Zero-vector: both sides are zero — define cosine as 1.0 (exact match).
        let cos = if denom < 1e-30 { 1.0 } else { dot / denom };
        sum_cos += cos;
        if cos < min_cos {
            min_cos = cos;
        }
    }

    CosineStats {
        mean: (sum_cos / n_rows as f64) as f32,
        min: min_cos as f32,
        n_rows,
    }
}

/// Generate `n` f32 values in `[-1.0, 1.0]` from a pinned LCG with `seed`.
///
/// Uses the Knuth LCG (`mul = 6364136223846793005`,
/// `add = 1442695040888963407`) — same parameters as the existing
/// `turboquant_tests.rs` and `planarquant_tests.rs` fixtures.
///
/// Range: `state >> 32` extracts the upper 32 bits as a u32 in `[0, u32::MAX]`;
/// dividing by `u32::MAX` gives `[0.0, 1.0]`; `* 2.0 - 1.0` maps to `[-1.0, 1.0]`.
/// (Previously used `>> 33` which only extracted 31 bits, biasing output to `[-1.0, ~0.0)`.)
pub(crate) fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Upper 32 bits → uniform [0, 1].  Fixed from >> 33 (biased negative half).
            let frac = (state >> 32) as u32 as f32 / u32::MAX as f32;
            frac * 2.0 - 1.0
        })
        .collect()
}

/// Apply a normalized Walsh–Hadamard transform (FWHT) in-place, row-wise.
///
/// `buf` is processed as `buf.len() / n` rows of `n` elements each. After the
/// call each row holds `H_n × row / sqrt(n)`, where `H_n` is the Sylvester
/// Hadamard matrix. This is the CPU reference for the rot_k Hadamard rotation
/// (`rot_k::hadamard_rotation` + `rot_k::rotate_last_axis` use MLX matmul for
/// the production path).
///
/// Since `H_n` is symmetric and self-inverse (`H_n H_n = n I`), normalizing
/// by `1 / sqrt(n)` gives an orthogonal matrix `R = H_n / sqrt(n)` satisfying
/// `R R = I`. Therefore calling `fwht_normalize` twice is the identity.
///
/// # Panics
///
/// Panics (test-only) if `n == 0`, `n` is not a power of two, or
/// `buf.len()` is not a multiple of `n`.
pub(crate) fn fwht_normalize(buf: &mut [f32], n: usize) {
    assert!(
        n > 0 && n.is_power_of_two(),
        "fwht_normalize: n={n} must be a positive power-of-two"
    );
    assert!(
        buf.len().is_multiple_of(n),
        "fwht_normalize: buf.len()={} is not a multiple of n={n}",
        buf.len()
    );

    let inv_sqrt_n = 1.0f32 / (n as f32).sqrt();

    for row in buf.chunks_exact_mut(n) {
        // Cooley-Tukey Hadamard butterfly, in-place.
        let mut h = 1usize;
        while h < n {
            let step = h * 2;
            let mut i = 0;
            while i < n {
                for j in i..i + h {
                    let a = row[j];
                    let b = row[j + h];
                    row[j] = a + b;
                    row[j + h] = a - b;
                }
                i += step;
            }
            h = step;
        }
        for v in row.iter_mut() {
            *v *= inv_sqrt_n;
        }
    }
}

// ── Unit tests for the harness itself ────────────────────────────────────────

#[cfg(test)]
#[path = "test_utils_tests.rs"]
mod test_utils_tests;
