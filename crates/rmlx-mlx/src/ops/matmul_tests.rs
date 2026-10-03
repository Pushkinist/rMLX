use super::*;
use crate::ops::{concatenate, conv1d, max_axis, maximum, negative, subtract};
use crate::Dtype;

/// Group sizes at or above the K tile. A group this wide always divides the
/// split-K partition evenly, so the guard must never fire for one.
const GROUP_SIZES: [i32; 3] = [32, 64, 128];

fn is_tile_whole(m: i32, n: i32, k: i32, group_size: i32) -> bool {
    splitk_k_partition(m, n, k, group_size).is_none_or(|p| p % SPLITK_K_TILE == 0)
}

/// The shapes a real nvfp4 checkpoint dispatches: `hidden_size` and the four
/// attention projections of a gemma-4-class text tower, plus a couple of
/// narrower / wider cells to spread the search.
const SHAPES: [(i32, i32); 8] = [
    (2048, 2560),
    (512, 2560),
    (256, 2560),
    (1024, 2560),
    (2560, 2048),
    (4096, 2560),
    (128, 1024),
    (64, 4096),
];

/// The partition MLX picks for the shape that corrupts a real nvfp4 checkpoint
/// is not a whole number of K tiles, and growing the batch by one tile fixes it.
///
/// Pinned as literals so a change to the mirrored split-K arithmetic has to
/// restate what it believes MLX does, rather than silently agreeing with
/// itself.
#[test]
fn narrow_group_partition_is_not_tile_whole_at_the_checkpoint_shape() {
    // k_proj / v_proj of a gemma-4-class nvfp4 tower, prefilling 16 tokens.
    assert_eq!(splitk_k_partition(16, 512, 2560, 16), Some(80));
    assert_ne!(80 % SPLITK_K_TILE, 0, "80 is 2.5 tiles, not a whole count");
    assert_eq!(splitk_safe_rows(16, 512, 2560, 16), Some(33));
    assert_eq!(splitk_k_partition(33, 512, 2560, 16), Some(160));
    assert_eq!(160 % SPLITK_K_TILE, 0);

    // The per-layer input gate of the same tower, one tile further up.
    assert_eq!(splitk_k_partition(48, 256, 2560, 16), Some(80));
    assert_eq!(splitk_safe_rows(48, 256, 2560, 16), Some(65));
    assert_eq!(splitk_k_partition(65, 256, 2560, 16), Some(128));
}

/// Whatever row count the guard hands back is one MLX splits into whole K
/// tiles, and it never shrinks the batch.
#[test]
fn padded_row_count_is_always_tile_whole_and_never_shrinks() {
    for (n, k) in SHAPES {
        for m in 1..1024 {
            match splitk_safe_rows(m, n, k, 16) {
                None => assert!(
                    m < qmv_batch_limit_floor(k, n) || is_tile_whole(m, n, k, 16),
                    "declined to pad an unsafe shape m={m} n={n} k={k}"
                ),
                Some(padded) => {
                    assert!(padded > m, "padding must grow the batch: {padded} <= {m}");
                    assert!(
                        is_tile_whole(padded, n, k, 16),
                        "padded to a still-unsafe row count m={m} -> {padded} n={n} k={k}"
                    );
                }
            }
        }
    }
}

/// How far the guard will grow a batch, pinned.
///
/// Declining to pad past some ratio is not available as a remedy: the batches
/// that need the largest pads are exactly the small ones, and MLX's real
/// vector/tiled crossover is architecture-dependent, so a batch this guard
/// declines may still be one the tiled kernel runs — and then it is corrupt.
/// The growth is bounded instead by [`qmv_batch_limit_floor`], and this test
/// records what that bound actually buys. It is a budget, not a threshold:
/// raising it silently is the regression to catch.
#[test]
fn padding_growth_stays_within_the_recorded_budget() {
    let mut worst = (1, 0, 0, 0, 0);
    for (n, k) in SHAPES {
        for m in 1..2048 {
            if let Some(padded) = splitk_safe_rows(m, n, k, 16) {
                if padded * worst.0 > worst.1 * m {
                    worst = (m, padded, m, n, k);
                }
            }
        }
    }
    let (m, padded, _, n, k) = worst;
    // Worst case over SHAPES: m=14 grows to 65 rows at n=128, k=1024.
    assert!(
        padded * 14 <= m * 65,
        "padding grew {m} -> {padded} ({:.2}x) at n={n} k={k}; budget is 65/14 = 4.64x",
        f64::from(padded) / f64::from(m)
    );
}

/// Negative control. A group at least as wide as the K tile always divides the
/// partition evenly, so the guard must be inert for every codec except the
/// narrow-group one — no padding, on any shape or batch.
#[test]
fn group_sizes_at_or_above_the_tile_never_pad() {
    for group_size in GROUP_SIZES {
        for (n, k) in SHAPES {
            for m in 1..1024 {
                assert_eq!(
                    splitk_safe_rows(m, n, k, group_size),
                    None,
                    "padded a wide-group shape gs={group_size} m={m} n={n} k={k}"
                );
                assert!(is_tile_whole(m, n, k, group_size));
            }
        }
    }
}

/// Below the shape's vector-kernel floor the tiled kernel cannot run on any
/// Apple GPU, so single-token decode must stay on the untouched path however
/// the shape scores.
#[test]
fn batches_below_the_vector_kernel_floor_never_pad() {
    for (n, k) in SHAPES {
        for m in 1..qmv_batch_limit_floor(k, n) {
            assert_eq!(
                splitk_safe_rows(m, n, k, 16),
                None,
                "padded below the vector-kernel floor m={m} n={n} k={k}"
            );
        }
    }
}

/// The floor is the minimum over every branch of MLX's `get_qmv_batch_limit`,
/// so it can never exceed the limit any Apple GPU actually uses.
#[test]
fn vector_kernel_floor_is_the_minimum_over_every_upstream_branch() {
    // (arch_size == 'd', arch_gen in {13,14}) -> the three shape tiers.
    let branches = |k: i32, n: i32| -> [i32; 4] {
        let tier = |big: [i32; 3]| {
            if k <= 2048 && n <= 2048 {
                big[0]
            } else if k <= 4096 && n <= 4096 {
                big[1]
            } else {
                big[2]
            }
        };
        [
            tier([32, 18, 12]), // 'd', any gen
            tier([14, 10, 6]),  // non-'d', gen 13/14
            tier([32, 18, 12]), // 'd', other gen
            tier([18, 12, 10]), // non-'d', other gen
        ]
    };
    for (n, k) in SHAPES {
        let smallest = branches(k, n).into_iter().min().unwrap_or(i32::MAX);
        assert_eq!(
            qmv_batch_limit_floor(k, n),
            smallest,
            "floor for k={k} n={n} is not the minimum upstream branch"
        );
    }
}

/// The guard exists only for MLX releases whose `qmm_splitk` aligns the split-K
/// partition to `group_size` alone. Upstream aligns it to `max(group_size, 32)`
/// after 0.32; against such a build every pad this guard performs is pointless
/// work that stays numerically correct, so nothing else would ever report it.
///
/// mlx-c exposes the linked version but no way to ask which arithmetic it
/// carries, so this is the drift alarm: it fails the build rather than letting
/// the guard rot into a silent prefill tax.
#[test]
#[allow(
    clippy::expect_used,
    reason = "an MLX whose version is unknown cannot be cleared of the drift this test exists for"
)]
fn linked_mlx_still_carries_the_misaligned_split_k_partition() {
    let (major, minor, patch) =
        crate::loaded_mlx_version().expect("the loaded MLX version could not be read");
    assert!(
        (major, minor) <= (0, 32),
        "linked MLX is {major}.{minor}.{patch}, past the last release whose qmm_splitk aligns \
         the split-K partition to group_size alone. Re-check upstream: if this \
         build aligns to max(group_size, 32), delete qmv_batch_limit_floor, \
         splitk_k_partition, splitk_safe_rows, their call in quantized_matmul \
         and these tests — the guard is then dead weight, not protection."
    );
}

/// Counts events this module emits, so a test can observe whether
/// `quantized_matmul` actually took the pad path rather than only checking a
/// result the unpadded path would produce too.
struct PadEventCounter(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl tracing::Subscriber for PadEventCounter {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        // Never cache a decision: `enabled` must run for every event so a
        // stale global interest cannot silence the count.
        tracing::subscriber::Interest::sometimes()
    }
    fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
        Some(tracing::level_filters::LevelFilter::TRACE)
    }
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        meta.target().starts_with("rmlx_mlx::ops::matmul")
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        if event
            .metadata()
            .target()
            .starts_with("rmlx_mlx::ops::matmul")
        {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

fn deterministic_bf16(shape: &[i32], seed: u32, device: Device) -> Array {
    let count: i32 = shape.iter().product();
    let mut state = seed | 1;
    let mut bytes = Vec::with_capacity(count as usize * 4);
    for _ in 0..count {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
        bytes.extend_from_slice(&(unit * 2.0 - 1.0).to_le_bytes());
    }
    #[allow(
        clippy::expect_used,
        reason = "fixture construction from bytes built in this fn; a failure here is a test bug"
    )]
    Array::from_bytes(&bytes, shape, Dtype::F32)
        .expect("from_bytes")
        .astype(Dtype::Bf16, device)
        .expect("astype bf16")
}

/// The wrapper takes the pad/slice round trip without disturbing the caller's
/// rank, on the `[batch, seq, in_features]` shape every `Linear` passes.
///
/// This is the only test that drives `quantized_matmul` itself on a rank the
/// GPU test does not cover; deleting the guard's wiring leaves every pure-
/// function test above green.
#[test]
#[allow(
    clippy::expect_used,
    reason = "CPU fixture setup in this fn; a failure here is a test bug"
)]
fn wrapper_restores_the_caller_shape_through_the_pad_round_trip() {
    let device = Device::Cpu;
    let (batch, seq, k, n) = (2, 8, 2560, 512);
    assert_eq!(
        splitk_safe_rows(batch * seq, n, k, 16),
        Some(33),
        "fixture must be a shape the guard pads"
    );

    let w = deterministic_bf16(&[n, k], 0x51D3, device);
    let (codes, scales, _) = quantize_mode(&w, 16, 4, "nvfp4", device).expect("quantize");
    let x = deterministic_bf16(&[batch, seq, k], 0x7A21, device);

    let pads = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let got =
        tracing::subscriber::with_default(PadEventCounter(std::sync::Arc::clone(&pads)), || {
            quantized_matmul(&x, &codes, &scales, None, 16, 4, "nvfp4", true, device)
        })
        .expect("quantized_matmul");
    assert_eq!(
        pads.load(std::sync::atomic::Ordering::SeqCst),
        1,
        "quantized_matmul did not take the pad path; the guard is not wired in"
    );
    assert_eq!(got.shape(), vec![batch, seq, n], "rank or extent changed");

    // Same rows, presented flat: the pad must not make the result depend on how
    // the caller grouped them.
    let flat = x.reshape(&[batch * seq, k], device).expect("reshape");
    let flat_out = quantized_matmul(&flat, &codes, &scales, None, 16, 4, "nvfp4", true, device)
        .expect("quantized_matmul flat");
    assert_eq!(flat_out.shape(), vec![batch * seq, n]);
    let diff = subtract(
        &got.reshape(&[batch * seq, n], device).expect("reshape"),
        &flat_out,
        device,
    )
    .expect("subtract");
    assert_eq!(max_abs(&diff, device), 0.0, "grouping changed the result");
}

/// The correctness claim, on the device that actually has the defect: a narrow
/// group nvfp4 matmul at the corrupting shape agrees with dequantize-then-dense,
/// and a wide group at the same shape — which never routes through the guard —
/// agrees to the same tolerance as the control.
///
/// The bar is measured, not guessed. Stubbing `splitk_safe_rows` to `None` puts
/// m=16 at relative 0.515. With the guard in place the narrow group peaks at
/// 0.0075, *below* the 0.0094 the never-guarded wide-group control reaches at
/// the same shapes — so what is left is bf16 round-trip noise, not residue.
/// The 0.05 bar sits 5.3x above that noise and 10x below the failure it has to
/// catch; retighten from these two numbers, not by taste.
#[test]
#[ignore = "drives Metal; run under make gpu-test"]
#[allow(
    clippy::expect_used,
    reason = "GPU fixture setup in this fn; a failure here is a test bug"
)]
fn narrow_group_quantized_matmul_matches_dense_reference_at_the_corrupting_shape() {
    let device = Device::Gpu;
    let (n, k) = (512, 2560);

    for (mode, group_size, bits) in [("nvfp4", 16, 4), ("mxfp4", 32, 4)] {
        let w = deterministic_bf16(&[n, k], 0x51D3, device);
        let (codes, scales, _) =
            quantize_mode(&w, group_size, bits, mode, device).expect("quantize");
        let dense = dequantize(&codes, &scales, None, group_size, bits, mode, device)
            .expect("dequantize")
            .astype(Dtype::F32, device)
            .expect("astype");

        for m in [8, 16, 20, 32, 33, 64] {
            let x = deterministic_bf16(&[m, k], 0x7A21 + m as u32, device);
            let got = quantized_matmul(
                &x, &codes, &scales, None, group_size, bits, mode, true, device,
            )
            .expect("quantized_matmul")
            .astype(Dtype::F32, device)
            .expect("astype");
            let want = matmul(
                &x.astype(Dtype::F32, device).expect("astype"),
                &dense.transpose(&[1, 0], device).expect("transpose"),
                device,
            )
            .expect("matmul");

            assert_eq!(got.shape(), vec![m, n], "output shape changed");

            let diff = subtract(&got, &want, device).expect("subtract");
            let relative = max_abs(&diff, device) / max_abs(&want, device);
            assert!(
                relative < 0.05,
                "{mode} gs={group_size} m={m}: relative error {relative} against dense reference"
            );
        }
    }
}

#[allow(
    clippy::expect_used,
    reason = "reduction over a materialised test array; a failure here is a test bug"
)]
fn max_abs(a: &Array, device: Device) -> f32 {
    let neg = negative(a, device).expect("negative");
    let mag = maximum(a, &neg, device).expect("maximum");
    let flat_len: i32 = mag.shape().iter().product();
    let flat = mag.reshape(&[flat_len], device).expect("reshape");
    let peak = max_axis(&flat, 0, device).expect("max_axis");
    let host = peak.astype(Dtype::F32, device).expect("astype");
    host.eval().expect("eval");
    let bytes = host.to_bytes().expect("to_bytes");
    f32::from_le_bytes(bytes[..4].try_into().expect("4 bytes"))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Thirty-one bf16 rows of width `k`: every value NaN when `scale` is NaN,
/// otherwise varied values of magnitude up to `|scale|`. Varied rows quantize
/// to non-zero codes, so a tail lane reaching a kept column would move it
/// through its codes as well as through its scale and bias.
#[allow(
    clippy::expect_used,
    reason = "GPU fixture setup in this fn; a failure here is a test bug"
)]
fn tail_rows(k: i32, scale: f32, device: Device) -> Array {
    let values: Vec<f32> = if scale.is_nan() {
        vec![f32::NAN; 31 * k as usize]
    } else {
        let unit = deterministic_bf16(&[31, k], 0x7A11, device)
            .astype(Dtype::F32, device)
            .expect("astype f32")
            .to_bytes()
            .expect("to_bytes");
        unit.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes")) * scale)
            .collect()
    };
    Array::from_f32_slice(&values, &[31, k])
        .expect("tail")
        .astype(Dtype::Bf16, device)
        .expect("astype")
}

/// A 32-row quantized weight whose rows past the first are `tail`, sliced to
/// its first row. The slice is a view into the 32-row buffers, so the rows an
/// `alN_false` tile reads past `N = 1` are the parent's rows, whose content the
/// caller chose. The flag says whether any code of the tail rows is non-zero.
#[allow(
    clippy::expect_used,
    reason = "GPU fixture setup in this fn; a failure here is a test bug"
)]
fn one_row_view_over_tail(
    row: &Array,
    tail: &Array,
    mode: &str,
    group_size: i32,
    bits: i32,
    device: Device,
) -> (Array, Array, Option<Array>, bool) {
    let parent = concatenate(&[row, tail], 0, device).expect("concatenate");
    let (codes, scales, biases) =
        quantize_mode(&parent, group_size, bits, mode, device).expect("quantize");
    let rows = |a: &Array, from: i32, to: i32| {
        let cols = a.shape()[1];
        a.slice(&[from, 0], &[to, cols], &[1, 1], device)
            .expect("slice")
    };
    let tail_codes_nonzero = rows(&codes, 1, 32)
        .to_bytes()
        .expect("to_bytes")
        .iter()
        .any(|b| *b != 0);
    let biases = (mode == "affine").then(|| rows(&biases, 0, 1));
    (
        rows(&codes, 0, 1),
        rows(&scales, 0, 1),
        biases,
        tail_codes_nonzero,
    )
}

/// Rows of the product under test. MLX leaves its vector kernel at the row
/// count `get_qmv_batch_limit` returns, and the highest return on any Apple
/// GPU is 33 (32 in MLX 0.31.2). The test does not rely on this figure: it
/// fails when the product stays on the vector kernel.
const SPLITK_ROWS: i32 = 64;

/// A batch height MLX runs on its vector kernel on every Apple GPU: the lowest
/// return of `get_qmv_batch_limit` is 6.
const VECTOR_KERNEL_ROWS: i32 = 2;

/// Whether the rows MLX's `qmm_t` split-K loader reads past the end of an
/// `N = 1` weight can reach the output. The same kept row is multiplied three
/// ways: out of its own one-row buffers (where the tail read leaves the
/// allocation and shader validation reports it), and out of views whose tail
/// rows are NaN, or varied values up to `+-3e4` with non-zero codes. Any lane
/// of a tail row that reached a kept column would make the outputs differ bit
/// for bit.
///
/// That a view's tile really reads its parent's tail rows is not asserted
/// here. On an MLX whose loader reads past the weight (every release before
/// 0.32.3) the evidence is the shader-validation census, where the fresh arm
/// reports 4 loads per mode and the view arms none.
///
/// **How the test knows the product ran on split-K.** MLX gives a transposed,
/// unbatched quantized product to its vector kernel below a row count that
/// depends on the GPU, and to `qmm_t_splitk` at or above it
/// (`QuantizedMatmul::eval_gpu`, `M >= vector_limit`). mlx-c reports neither
/// the limit nor the kernel. The two kernels sum in a different order, so the
/// test multiplies the same rows two more ways: in batches of
/// [`VECTOR_KERNEL_ROWS`], which every GPU runs on the vector kernel, and four
/// rows as one batch, also below every limit. The four-row batch must equal
/// its two-row batches bit for bit, which shows the vector kernel does not
/// depend on the batch height. The full batch must then differ from its
/// two-row batches, or the product did not leave the vector kernel and the
/// tail arms below prove nothing about split-K.
///
/// The printed digest is the fresh arm's output; comparing it between a run
/// under `make gpu-test` and one without shader validation says whether the
/// validation layer itself changes what this kernel computes.
#[test]
#[ignore = "drives Metal; run under make gpu-test"]
#[allow(
    clippy::expect_used,
    reason = "GPU fixture setup in this fn; a failure here is a test bug"
)]
fn split_k_tail_row_reads_never_reach_the_output() {
    let device = Device::Gpu;
    let (m, k) = (SPLITK_ROWS, 2048);
    let x = deterministic_bf16(&[m, k], 0x3C1E, device);
    let row = deterministic_bf16(&[1, k], 0x0B0E, device);

    for (mode, group_size, bits) in [("affine", 64, 8), ("affine", 64, 4), ("mxfp8", 32, 8)] {
        let product_of = |x: &Array, codes: &Array, scales: &Array, biases: Option<&Array>| {
            let y = quantized_matmul(
                x, codes, scales, biases, group_size, bits, mode, true, device,
            )
            .expect("quantized_matmul");
            assert_eq!(y.shape(), vec![x.shape()[0], 1], "output shape changed");
            y.to_bytes().expect("to_bytes")
        };
        let product = |codes: &Array, scales: &Array, biases: Option<&Array>| {
            product_of(&x, codes, scales, biases)
        };

        let (codes, scales, biases) =
            quantize_mode(&row, group_size, bits, mode, device).expect("quantize");
        let biases = (mode == "affine").then_some(biases);
        let fresh = product(&codes, &scales, biases.as_ref());

        let on_the_vector_kernel = |rows: i32| -> Vec<u8> {
            (0..rows / VECTOR_KERNEL_ROWS)
                .flat_map(|batch| {
                    let from = batch * VECTOR_KERNEL_ROWS;
                    let pair = x
                        .slice(&[from, 0], &[from + VECTOR_KERNEL_ROWS, k], &[1, 1], device)
                        .expect("slice");
                    product_of(&pair, &codes, &scales, biases.as_ref())
                })
                .collect()
        };
        let four = x.slice(&[0, 0], &[4, k], &[1, 1], device).expect("slice");
        assert_eq!(
            product_of(&four, &codes, &scales, biases.as_ref()),
            on_the_vector_kernel(4),
            "{mode} b{bits}: four rows as one batch differ from the same rows in batches \
             of {VECTOR_KERNEL_ROWS}, so a difference at {m} rows would not show which \
             kernel ran"
        );
        let vector = on_the_vector_kernel(m);
        let moved = fresh
            .chunks_exact(2)
            .zip(vector.chunks_exact(2))
            .filter(|(a, b)| a != b)
            .count();
        println!("split-k reach {mode} gs={group_size} b={bits}: {moved} of {m} rows differ");
        assert!(
            moved > 0,
            "{mode} b{bits}: {m} rows as one batch equal the same rows in batches of \
             {VECTOR_KERNEL_ROWS} bit for bit, so the product ran on the vector kernel \
             and this test did not reach split-K"
        );
        for repeat in 1..4 {
            assert_eq!(
                product(&codes, &scales, biases.as_ref()),
                fresh,
                "{mode} b{bits}: repeat {repeat} of the same product differs"
            );
        }
        println!(
            "split-k tail digest {mode} gs={group_size} b={bits}: {:016x}",
            fnv1a64(&fresh)
        );

        for scale in [f32::NAN, 3.0e4, -3.0e4] {
            let tail = tail_rows(k, scale, device);
            let (codes, scales, biases, tail_codes_nonzero) =
                one_row_view_over_tail(&row, &tail, mode, group_size, bits, device);
            if !scale.is_nan() {
                assert!(
                    tail_codes_nonzero,
                    "{mode} b{bits}: the tail rows at scale {scale} quantized to all-zero codes"
                );
            }
            assert_eq!(
                product(&codes, &scales, biases.as_ref()),
                fresh,
                "{mode} b{bits}: a tail row at scale {scale} reached the output"
            );
        }
    }
}

/// Whether the output-channel rows MLX's implicit-GEMM conv weight loader reads
/// past the end of the weight can reach the output. With 64-wide N tiles the
/// loader's bound check only exists for 8-wide tiles, so at `C_out = 96` the
/// second tile reads rows 96..128 of a 96-row weight. Same method as
/// `split_k_tail_row_reads_never_reach_the_output`: the fresh arm reads past
/// its own allocation, the view arms read a parent's NaN / +-3e4 rows, and any
/// tail lane reaching a kept channel makes the outputs differ bit for bit.
///
/// The shape is the audio codec's decoder residual conv: `bm = 64` needs an
/// implicit M of at least 8192 and 64 or more input channels.
#[test]
#[ignore = "drives Metal; run under make gpu-test"]
#[allow(
    clippy::expect_used,
    reason = "GPU fixture setup in this fn; a failure here is a test bug"
)]
fn implicit_gemm_conv_tail_channel_reads_never_reach_the_output() {
    let device = Device::Gpu;
    let (len, channels, taps, kept) = (8192, 96, 7, 96);
    let f32_of = |a: Array| a.astype(Dtype::F32, device).expect("astype f32");
    let x = f32_of(deterministic_bf16(&[1, len, channels], 0x5EED, device));
    let weight = f32_of(deterministic_bf16(&[kept, taps, channels], 0xC0DE, device));

    let conv = |w: &Array| {
        let y = conv1d(&x, w, 1, taps / 2, 1, 1, device).expect("conv1d");
        assert_eq!(y.shape(), vec![1, len, kept], "output shape changed");
        y.to_bytes().expect("to_bytes")
    };

    let fresh = conv(&weight);
    for repeat in 1..4 {
        assert_eq!(
            conv(&weight),
            fresh,
            "repeat {repeat} of the same conv differs"
        );
    }
    println!("implicit-gemm conv tail digest: {:016x}", fnv1a64(&fresh));

    for tail in [f32::NAN, 3.0e4, -3.0e4] {
        let tail_rows = Array::from_f32_slice(
            &vec![tail; (32 * taps * channels) as usize],
            &[32, taps, channels],
        )
        .expect("tail");
        let parent = concatenate(&[&weight, &tail_rows], 0, device).expect("concatenate");
        let view = parent
            .slice(&[0, 0, 0], &[kept, taps, channels], &[1, 1, 1], device)
            .expect("slice");
        assert_eq!(
            conv(&view),
            fresh,
            "a tail channel row of {tail} reached the output"
        );
    }
}
