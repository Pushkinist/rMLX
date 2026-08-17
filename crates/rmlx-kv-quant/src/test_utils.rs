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

/// Generate `n` standard-normal f32 values from the same pinned Knuth LCG as
/// [`lcg_data`], via Box–Muller.
///
/// Deterministic for a fixed `seed`; never `thread_rng`. Pairs of uniforms are
/// consumed per pair of outputs; an odd `n` discards the second variate of the
/// final pair.
///
/// `u1` is drawn on `(0, 1]` (`+1` before the divide) so `ln(u1)` is finite;
/// `u2` on `[0, 1)`. Accumulated in f64 and narrowed once, so the output is
/// bit-stable across optimisation levels.
pub(crate) fn gaussian_data(n: usize, seed: u64) -> Vec<f32> {
    const TWO_POW_32: f64 = 4_294_967_296.0;
    let mut state = seed;
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 32) as u32
    };

    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = (f64::from(next_u32()) + 1.0) / TWO_POW_32; // (0, 1]
        let u2 = f64::from(next_u32()) / TWO_POW_32; // [0, 1)
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}

/// Generate `[n_rows, head_dim]` f32 modelling the **K-cache outlier-channel**
/// shape: i.i.d. standard-normal base with `n_outlier_channels` persistent
/// channels scaled by `outlier_ratio`.
///
/// # Why this shape
///
/// Uniform / i.i.d. fixtures are already close to maximally incoherent, so a
/// rotation cannot improve them and a gate built on one cannot see rotation
/// quality at all (see [`incoherence_per_row`]). Real K-cache activations are
/// not i.i.d.: the reported structure is a handful of **fixed channels whose
/// magnitude is large in every token**, which is why the KV-quantization
/// literature quantizes Keys per-channel and Values per-token, and why the
/// rotation-KV family exists at all.
///
/// * KIVI (Liu et al., ICML 2024, arXiv:2402.02750) — the Key cache has a few
///   fixed channels with large magnitudes; the Value cache has no such pattern.
/// * KVQuant (Hooper et al., NeurIPS 2024, arXiv:2401.18079) — same
///   observation; motivates per-channel Key quantization.
/// * LLM.int8() (Dettmers et al., 2022, arXiv:2208.07339) — emergent outlier
///   features occupy a fraction of a percent of the hidden dimensions at
///   magnitudes on the order of 20× the rest. That report is the source of the
///   `20.0` default ratio in [`OUTLIER_RATIO`]; it is **not** the source of the
///   channel count, see [`OUTLIER_CHANNELS`].
/// * QuIP (Chee et al., 2023, arXiv:2307.13304) and QuaRot (Ashkboos et al.,
///   2024, arXiv:2404.00456) — a full-dimension (randomized) Hadamard is the
///   standard remedy, and the incoherence parameter is the standard measure of
///   whether it worked.
///
/// This is a **model** of that shape, not a capture of any particular layer:
/// the base is exactly i.i.d. Gaussian and every outlier channel gets the same
/// ratio. It is adversarial in the one axis that matters here — a few
/// coordinates carry most of the mass — and it is reproducible from a seed.
///
/// # Channel placement
///
/// Indices are `(i · head_dim / n + i · OUTLIER_CHANNEL_TWIST) mod head_dim`:
/// an even spread so the channels cover the row, plus a stride so their
/// **intra-block offsets differ**. The twist is load-bearing. A plain even
/// spread at `head_dim = 128` with 4 channels gives 0 / 32 / 64 / 96, every one
/// of which sits at offset 0 of a block of 2, 3 or 4 — the most aligned
/// placement available, not a neutral one. That is harmless for iso, whose left
/// quaternion product has the same largest column magnitude in every slot, but
/// planar's per-pair Givens search is not symmetric under swapping the pair, so
/// a constant offset would put an unmeasured bias into a pinned number. With
/// the twist the four channels land on 0 / 37 / 74 / 111, whose offsets are
/// distinct mod 4, cover both residues mod 2, and take three of three values
/// mod 3.
///
/// # Panics
///
/// Panics (test-only) if `head_dim == 0` or
/// `n_outlier_channels > head_dim`.
pub(crate) fn outlier_channel_data(
    n_rows: usize,
    head_dim: usize,
    n_outlier_channels: usize,
    outlier_ratio: f32,
    seed: u64,
) -> Vec<f32> {
    assert!(head_dim > 0, "outlier_channel_data: head_dim must be > 0");
    assert!(
        n_outlier_channels <= head_dim,
        "outlier_channel_data: n_outlier_channels={n_outlier_channels} exceeds head_dim={head_dim}",
    );

    let mut data = gaussian_data(n_rows * head_dim, seed);
    for channel in outlier_channels(head_dim, n_outlier_channels) {
        for row in 0..n_rows {
            // row < n_rows and channel < head_dim, so the index is in bounds.
            data[row * head_dim + channel] *= outlier_ratio;
        }
    }
    data
}

/// Stride applied on top of the even spread in [`outlier_channels`] so the
/// chosen channels do not all share one intra-block offset.
///
/// 5 is coprime to 2, 3 and 4 — the block sizes of every rotation family here —
/// so successive channels advance through the offsets rather than repeating one.
const OUTLIER_CHANNEL_TWIST: usize = 5;

/// Which channels [`outlier_channel_data`] scales. See its "Channel placement".
///
/// The twist is coprime to the block sizes but not to every `head_dim`, so for
/// some counts the raw formula collides. Colliding slots walk forward to the
/// next free channel, which keeps the result exactly `n_outlier_channels`
/// distinct channels for any `n_outlier_channels <= head_dim` — without that, a
/// channel scaled twice would get `ratio²` and the count would silently be an
/// over-estimate.
///
/// # Panics
///
/// Panics (test-only) if `n_outlier_channels > head_dim`.
pub(crate) fn outlier_channels(head_dim: usize, n_outlier_channels: usize) -> Vec<usize> {
    assert!(
        n_outlier_channels <= head_dim,
        "outlier_channels: {n_outlier_channels} channels requested of head_dim={head_dim}",
    );
    let mut taken = vec![false; head_dim];
    let mut out = Vec::with_capacity(n_outlier_channels);
    for i in 0..n_outlier_channels {
        let mut channel =
            (i * head_dim / n_outlier_channels + i * OUTLIER_CHANNEL_TWIST) % head_dim;
        while taken[channel] {
            channel = (channel + 1) % head_dim;
        }
        taken[channel] = true;
        out.push(channel);
    }
    out
}

/// Rows in the canonical outlier fixture — enough that the p99 of the per-row
/// statistic is meaningful.
pub(crate) const OUTLIER_ROWS: usize = 256;

/// `head_dim` of the canonical outlier fixture. 128 is a real shipped head_dim
/// (Bonsai / Qwen3) and a power of two, so the full-dimension Hadamard applies.
pub(crate) const OUTLIER_HEAD_DIM: usize = 128;

/// Outlier channels in the canonical fixture: 4 of 128 = 3.1%.
///
/// Deliberately **denser than the literature**, which puts emergent outlier
/// features at a fraction of a percent of the hidden dimensions. The density is
/// chosen so that every affine group of 64 — the group `rot_k` sets its 8-bit
/// scale over — contains an outlier, which is the condition under which a
/// full-dimension rotation has something to recover across the whole row. At a
/// literature-faithful 0.8% (one channel of 128) one of the two groups is clean
/// and the row-averaged gain roughly halves; the fixture would still be
/// adversarial, just less uniformly so.
///
/// `outlier_channel_count_monotonically_raises_incoherence` sweeps this the way
/// `outlier_ratio_monotonically_raises_incoherence` sweeps the magnitude, so
/// neither parameter is a bare assertion.
pub(crate) const OUTLIER_CHANNELS: usize = 4;

/// Magnitude ratio of an outlier channel to the Gaussian base — the order of
/// magnitude reported for emergent outlier features (arXiv:2208.07339).
pub(crate) const OUTLIER_RATIO: f32 = 20.0;

/// The canonical outlier-channel fixture every rotation gate measures against.
pub(crate) fn outlier_fixture() -> Vec<f32> {
    outlier_channel_data(
        OUTLIER_ROWS,
        OUTLIER_HEAD_DIM,
        OUTLIER_CHANNELS,
        OUTLIER_RATIO,
        TEST_SEED,
    )
}

// ── Incoherence statistic ────────────────────────────────────────────────────

/// Summary statistics from [`incoherence_per_row`].
pub(crate) struct IncoherenceStats {
    /// Mean of `mu` across all rows.
    pub mean: f64,
    /// 99th percentile of `mu` (nearest-rank) — the tail a quantizer's range
    /// has to cover.
    pub p99: f64,
    /// Maximum `mu` across all rows.
    pub max: f64,
    /// Number of rows processed.
    pub n_rows: usize,
}

/// Per-row incoherence `mu(x) = sqrt(d) · max_i |x_i| / ||x||_2`.
///
/// `mu = 1` for a perfectly flat vector and `mu = sqrt(d)` for a one-hot
/// vector, so it is a normalized peak-to-RMS ratio: exactly the quantity a
/// uniform quantizer's range is set by, and exactly what a decorrelating
/// rotation is supposed to reduce. The name and normalization follow QuIP
/// (arXiv:2307.13304).
///
/// An all-zero row is defined as `mu = 1.0` — it is flat, and the ratio is
/// otherwise `0/0`.
///
/// # Panics
///
/// Panics (test-only) if `head_dim == 0` or does not evenly divide `x.len()`,
/// or if `x` is empty.
pub(crate) fn incoherence_per_row(x: &[f32], head_dim: usize) -> IncoherenceStats {
    assert!(
        head_dim > 0 && !x.is_empty() && x.len().is_multiple_of(head_dim),
        "incoherence_per_row: head_dim={head_dim} does not evenly divide a non-empty len={}",
        x.len(),
    );

    let sqrt_d = (head_dim as f64).sqrt();
    let mut mus: Vec<f64> = x
        .chunks_exact(head_dim)
        .map(|row| {
            let peak = row
                .iter()
                .fold(0.0f64, |acc, &v| acc.max(f64::from(v).abs()));
            let l2 = row
                .iter()
                .fold(0.0f64, |acc, &v| f64::from(v).mul_add(f64::from(v), acc))
                .sqrt();
            if l2 <= 0.0 {
                1.0
            } else {
                sqrt_d * peak / l2
            }
        })
        .collect();

    let n_rows = mus.len();
    let mean = mus.iter().sum::<f64>() / n_rows as f64;
    mus.sort_by(f64::total_cmp);
    // Nearest-rank p99: the smallest value at or above 99% of the sorted rows.
    let rank = ((n_rows as f64) * 0.99).ceil().max(1.0) as usize;
    let p99 = mus[rank.min(n_rows) - 1];
    let max = mus[n_rows - 1];

    IncoherenceStats {
        mean,
        p99,
        max,
        n_rows,
    }
}

// ── Stored-rate accounting ───────────────────────────────────────────────────

/// Bits a bf16 KV buffer spends per stored value.
///
/// The floor every KV codec is measured against. A codec whose store lands
/// above this is not compressing — it is paying a codebook's *nominal* width in
/// docs while spending more than the uncompressed baseline in memory, which is
/// the failure mode the rotor and iso families shipped with.
pub(crate) const BF16_BITS_PER_VALUE: f64 = 16.0;

/// Stored bits per input value, from an encoder's **actual** output buffers.
///
/// `stored_bytes` is the summed heap size of everything the encode call
/// produced — codes, per-group scales, per-group rotations, per-token norms,
/// any sideband — and `n_values` the number of input values those buffers
/// describe. Deriving the rate this way rather than from a codec's advertised
/// bit width is the point: the two disagree by 5.4x for rotor3, because the
/// nominal width counts only the codes and only the ones that carry
/// information.
///
/// Returns `f64::INFINITY` for `n_values == 0` — a store with no values has no
/// meaningful rate, and an infinite one fails every ceiling rather than passing
/// them all.
pub(crate) fn stored_bits_per_value(stored_bytes: u64, n_values: usize) -> f64 {
    if n_values == 0 {
        return f64::INFINITY;
    }
    (stored_bytes * 8) as f64 / n_values as f64
}

// ── Rate-distortion reference ────────────────────────────────────────────────

/// dB per bit for a scalar quantizer: `20 · log10(2)`.
///
/// One extra bit halves the quantizer step, so it buys this much SQNR. Used to
/// convert a dB shortfall into the directly meaningful unit — see
/// [`wasted_bits`].
pub(crate) const DB_PER_BIT: f64 = 6.020_599_913_279_624;

/// SQNR in dB achievable by a **fixed-rate Lloyd-Max quantizer on the standard
/// normal**, indexed by `bits - 1` for `bits ∈ {1, 2, 3, 4}`.
///
/// These are *not* the rate-distortion bound. The bound is `6.02 · b` dB and an
/// optimally entropy-coded scalar quantizer sits ~1.53 dB below it (the gap
/// between cubic quantizer cells and sphere packing); a fixed-rate Lloyd-Max
/// quantizer is lower still. The values here are `10 · log10(1 / D)` for the
/// classical minimum mean-square distortions `D` of Max (1960),
/// "Quantizing for Minimum Distortion", IRE Trans. Inf. Theory 6(1), Table I:
///
/// | bits | levels | `D` | anchor |
/// |---|---|---|---|
/// | 1 | 2 | 0.363 4 | 4.396 dB |
/// | 2 | 4 | 0.117 5 | 9.300 dB |
/// | 3 | 8 | 0.034 54 | 14.616 dB |
/// | 4 | 16 | 0.009 497 | 20.224 dB |
///
/// The anchor assumes the quantizer is **matched to the source** — it knows
/// sigma and spends no rate saying so. Every codec here instead derives a scale
/// from a per-group maximum and stores it, so the comparison is not
/// apples-to-apples in the codec's favour: the codec pays extra rate for the
/// scale and can therefore legitimately land *above* the anchor. What the
/// anchor is good for is the other direction — landing well below it at the
/// same nominal bit width means the bits are not buying what they should.
pub(crate) const LLOYD_MAX_GAUSSIAN_SQNR_DB: [f64; 4] = [4.396, 9.300, 14.616, 20.224];

/// [`LLOYD_MAX_GAUSSIAN_SQNR_DB`] for `bits`.
///
/// # Panics
///
/// Panics (test-only) for `bits` outside `1..=4` — no anchor is tabulated.
pub(crate) fn lloyd_max_anchor_db(bits: u8) -> f64 {
    assert!(
        (1..=4).contains(&bits),
        "lloyd_max_anchor_db: no Lloyd-Max anchor tabulated for bits={bits}",
    );
    LLOYD_MAX_GAUSSIAN_SQNR_DB[usize::from(bits) - 1]
}

/// Signal-to-quantization-noise ratio in dB: `10 · log10(P_signal / P_error)`.
///
/// Both powers are accumulated in f64. A zero error power returns
/// `f64::INFINITY` (a lossless round-trip), and a zero signal power returns
/// `f64::NEG_INFINITY` unless the error is also zero.
///
/// # Panics
///
/// Panics (test-only) if the two slices differ in length or are empty.
pub(crate) fn sqnr_db(reference: &[f32], decoded: &[f32]) -> f64 {
    assert_eq!(
        reference.len(),
        decoded.len(),
        "sqnr_db: reference and decoded lengths differ ({} vs {})",
        reference.len(),
        decoded.len(),
    );
    assert!(!reference.is_empty(), "sqnr_db: empty input");

    let mut signal = 0.0f64;
    let mut error = 0.0f64;
    for (&r, &d) in reference.iter().zip(decoded.iter()) {
        let rf = f64::from(r);
        let e = rf - f64::from(d);
        signal = rf.mul_add(rf, signal);
        error = e.mul_add(e, error);
    }
    if error <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal / error).log10()
}

/// Shortfall of `measured_db` against `anchor_db`, expressed in bits.
///
/// Positive means the codec is that many bits short of the anchor; negative
/// means it is ahead of it. See [`DB_PER_BIT`].
pub(crate) fn wasted_bits(measured_db: f64, anchor_db: f64) -> f64 {
    (anchor_db - measured_db) / DB_PER_BIT
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
