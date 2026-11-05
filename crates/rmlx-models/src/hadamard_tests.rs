//! The signed block Hadamard transform against a host f64 oracle.
//!
//! The oracle is the contract's own definition, built with no MLX op: the
//! normalised Sylvester matrix `H[i][j] = (-1)^popcount(i & j) / sqrt(block)`,
//! one block at a time along the last axis; forward `H (s * x)`, inverse
//! `s * (H x)`. The host defects the bound must reject are checked in their
//! own test, which needs no transform. All tests run on `Device::Cpu`.
//!
//! Not covered here: a packed linear `dequantize(W') @ T(x)` with its
//! no-transform control, and the refusal of the hot-vocab head. Both belong
//! with the tests of the Qwen3.5 Hadamard `Linear` variant.

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use super::{Direction, SignedBlockHadamard};

const BLOCK: usize = 1024;
/// The input widths of the Ternary-Bonsai-2-27B pack: hidden size, the
/// attention / delta-net output width, and the MLP intermediate size.
const PACK_WIDTHS: [usize; 3] = [5120, 6144, 17408];
/// The vendor runtime's `fwht`, run on the MLX 0.32.0 CPU device with the
/// pack's own signs, was at most 2.8e-7 from this oracle at every pack width.
/// The bound leaves a margin of about 35 times.
const F32_BOUND: f64 = 1e-5;

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn uniform(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u32 << 23) as f32) - 1.0
    }
    fn sign(&mut self) -> f32 {
        if self.next_u64() >> 63 == 0 {
            1.0
        } else {
            -1.0
        }
    }
}

pub(crate) fn uniform(seed: u64, n: usize) -> Vec<f32> {
    let mut g = Lcg(seed);
    (0..n).map(|_| g.uniform()).collect()
}

pub(crate) fn signs(seed: u64, n: usize) -> Vec<f32> {
    let mut g = Lcg(seed);
    (0..n).map(|_| g.sign()).collect()
}

pub(crate) fn read_f32(a: &Array) -> Result<Vec<f32>> {
    let f = a.astype(Dtype::F32, Device::Cpu)?;
    f.eval()?;
    let bytes = f.to_bytes()?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn sylvester_sign(i: usize, j: usize) -> f64 {
    if (i & j).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

/// `H x` per `block` along the flat row-major buffer, `scale` per block.
fn host_hadamard(x: &[f64], block: usize, scale: f64) -> Vec<f64> {
    let table: Vec<f64> = (0..block * block)
        .map(|n| sylvester_sign(n / block, n % block))
        .collect();
    let mut out = Vec::with_capacity(x.len());
    for chunk in x.chunks_exact(block) {
        for row in table.chunks_exact(block) {
            out.push(row.iter().zip(chunk).map(|(h, v)| h * v).sum::<f64>() * scale);
        }
    }
    out
}

pub(crate) fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().map(|&v| f64::from(v)).collect()
}

/// `s` repeated over every row of `x`.
fn times_signs(x: &[f64], s: &[f32]) -> Vec<f64> {
    x.iter()
        .zip(s.iter().cycle())
        .map(|(v, sg)| v * f64::from(*sg))
        .collect()
}

pub(crate) fn oracle_forward(x: &[f64], s: &[f32], block: usize) -> Vec<f64> {
    host_hadamard(&times_signs(x, s), block, 1.0 / (block as f64).sqrt())
}

pub(crate) fn oracle_inverse(x: &[f64], s: &[f32], block: usize) -> Vec<f64> {
    times_signs(&host_hadamard(x, block, 1.0 / (block as f64).sqrt()), s)
}

pub(crate) fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn max_abs_diff_f32(a: &[f32], b: &[f64]) -> f64 {
    max_abs_diff(&widen(a), b)
}

fn oracle(direction: Direction, x: &[f64], s: &[f32]) -> Vec<f64> {
    match direction {
        Direction::Forward => oracle_forward(x, s, BLOCK),
        Direction::Inverse => oracle_inverse(x, s, BLOCK),
    }
}

fn run(direction: Direction, t: &SignedBlockHadamard, x: &Array, device: Device) -> Result<Array> {
    match direction {
        Direction::Forward => t.forward(x, device),
        Direction::Inverse => t.inverse(x, device),
    }
}

/// Spacing of `dtype`'s values at the magnitude of `v`: 7 mantissa bits for
/// bf16, 10 for f16 with its subnormal floor of 2^-24.
pub(crate) fn spacing(dtype: Dtype, v: f64) -> f64 {
    let (mantissa_bits, floor) = if dtype == Dtype::F16 {
        (10.0, 2f64.powi(-24))
    } else {
        (7.0, 2f64.powi(-133))
    };
    if v == 0.0 {
        return floor;
    }
    2f64.powf(v.abs().log2().floor() - mantissa_bits).max(floor)
}

/// Round-to-nearest-even of `v` to `dtype`, from f64 in one step.
pub(crate) fn round_to(dtype: Dtype, v: f64) -> f64 {
    let q = spacing(dtype, v);
    (v / q).round_ties_even() * q
}

#[test]
fn the_oracle_is_the_sylvester_matrix_and_its_own_inverse() {
    let h4: Vec<f64> = (0..16).map(|n| sylvester_sign(n / 4, n % 4)).collect();
    #[rustfmt::skip]
    let sylvester_4 = [
        1.0,  1.0,  1.0,  1.0,
        1.0, -1.0,  1.0, -1.0,
        1.0,  1.0, -1.0, -1.0,
        1.0, -1.0, -1.0,  1.0,
    ];
    assert_eq!(h4, sylvester_4, "H_4 must be [[H_2, H_2], [H_2, -H_2]]");

    let s = signs(11, 2 * BLOCK);
    let x = widen(&uniform(12, 2 * BLOCK));
    let round_trip = oracle_inverse(&oracle_forward(&x, &s, BLOCK), &s, BLOCK);
    let d = max_abs_diff(&round_trip, &x);
    assert!(d < 1e-12, "oracle inverse(forward(x)) misses x by {d:.3e}");
}

/// Each defect a transform can carry is computed on the host and must miss the
/// oracle by more than ten bounds, so the value tests' bound can reject it.
#[test]
fn host_defects_miss_the_oracle_by_more_than_ten_bounds() {
    let scale = 1.0 / (BLOCK as f64).sqrt();
    for (n, &width) in PACK_WIDTHS.iter().enumerate() {
        let s = signs(100 + n as u64, width);
        let x = widen(&uniform(200 + n as u64, width));
        let forward = oracle_forward(&x, &s, BLOCK);
        let inverse = oracle_inverse(&x, &s, BLOCK);
        for (what, defect, oracle) in [
            (
                "signs after H",
                times_signs(&host_hadamard(&x, BLOCK, scale), &s),
                &forward,
            ),
            ("no signs", host_hadamard(&x, BLOCK, scale), &forward),
            (
                "no 1/sqrt(block)",
                host_hadamard(&times_signs(&x, &s), BLOCK, 1.0),
                &forward,
            ),
            ("block 512", oracle_forward(&x, &s, BLOCK / 2), &forward),
            ("inverse = forward", forward.clone(), &inverse),
        ] {
            let d = max_abs_diff(&defect, oracle);
            assert!(
                d > 10.0 * F32_BOUND,
                "width {width}: defect `{what}` misses the oracle by only {d:.3e} \
                 (bound {F32_BOUND:.1e}); the bound cannot reject it"
            );
        }
    }
}

/// Worked by hand: `s * x = [1, -2, 3, -4]`; the rows of `H_4` give
/// `[-2, 10, 0, -4]`, halved `[-1, 5, 0, -2]`. Inverse: `H_4 x = [10, -2, -4, 0]`,
/// halved `[5, -1, -2, 0]`, times `s` `[5, 1, -2, 0]`. Every value is exact in f32.
fn check_block_four_by_hand(device: Device) -> Result<()> {
    let t = SignedBlockHadamard::new(4, &[1.0, -1.0, 1.0, -1.0])?;
    let x = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4])?;
    assert_eq!(
        read_f32(&t.forward(&x, device)?)?,
        vec![-1.0, 5.0, 0.0, -2.0]
    );
    assert_eq!(
        read_f32(&t.inverse(&x, device)?)?,
        vec![5.0, 1.0, -2.0, 0.0]
    );
    Ok(())
}

fn check_forward_at_each_pack_width(device: Device) -> Result<()> {
    for (n, &width) in PACK_WIDTHS.iter().enumerate() {
        let s = signs(100 + n as u64, width);
        let x32 = uniform(200 + n as u64, width);
        let oracle = oracle_forward(&widen(&x32), &s, BLOCK);

        let t = SignedBlockHadamard::new(BLOCK, &s)?;
        let out = t.forward(&Array::from_f32_slice(&x32, &[1, width as i32])?, device)?;
        assert_eq!(out.dtype(), Dtype::F32, "width {width}: f32 in, f32 out");
        assert_eq!(out.shape(), vec![1, width as i32], "width {width}: shape");
        let d = max_abs_diff_f32(&read_f32(&out)?, &oracle);
        assert!(
            d <= F32_BOUND,
            "width {width}: forward misses the oracle by {d:.3e}"
        );
    }
    Ok(())
}

fn check_signed_basis_vector(device: Device) -> Result<()> {
    let width = 5120;
    let (b, k) = (3, 777);
    let s = signs(300, width);
    let at = b * BLOCK + k;
    let x: Vec<f32> = s
        .iter()
        .enumerate()
        .map(|(n, &sg)| if n == at { sg } else { 0.0 })
        .collect();

    let t = SignedBlockHadamard::new(BLOCK, &s)?;
    let out = read_f32(&t.forward(&Array::from_f32_slice(&x, &[1, width as i32])?, device)?)?;

    let expected: Vec<f32> = (0..width)
        .map(|n| {
            if n / BLOCK == b {
                (sylvester_sign(k, n % BLOCK) / 32.0) as f32
            } else {
                0.0
            }
        })
        .collect();
    assert_eq!(
        out, expected,
        "H (s * (s_k e_k)) must be row k of H / 32 inside block {b} and zero elsewhere"
    );
    Ok(())
}

fn check_inverse_at_each_pack_width(device: Device) -> Result<()> {
    for (n, &width) in PACK_WIDTHS.iter().enumerate() {
        let s = signs(400 + n as u64, width);
        let x32 = uniform(500 + n as u64, width);
        let x = widen(&x32);
        let oracle = oracle_inverse(&x, &s, BLOCK);

        let t = SignedBlockHadamard::new(BLOCK, &s)?;
        let input = Array::from_f32_slice(&x32, &[1, width as i32])?;
        let inv = t.inverse(&input, device)?;
        assert_eq!(inv.dtype(), Dtype::F32, "width {width}: f32 in, f32 out");
        assert_eq!(inv.shape(), vec![1, width as i32], "width {width}: shape");
        let d = max_abs_diff_f32(&read_f32(&inv)?, &oracle);
        assert!(
            d <= F32_BOUND,
            "width {width}: inverse misses the oracle by {d:.3e}"
        );

        let round_trip = t.inverse(&t.forward(&input, device)?, device)?;
        let d = max_abs_diff_f32(&read_f32(&round_trip)?, &x);
        assert!(
            d <= 2.0 * F32_BOUND,
            "width {width}: inverse(forward(x)) misses x by {d:.3e}"
        );
    }
    Ok(())
}

fn check_leading_axes(device: Device) -> Result<()> {
    let width = 5120;
    let s = signs(600, width);
    let x32 = uniform(601, 2 * 3 * width);
    let x = Array::from_f32_slice(&x32, &[2, 3, width as i32])?;
    let t = SignedBlockHadamard::new(BLOCK, &s)?;
    for direction in [Direction::Forward, Direction::Inverse] {
        let out = run(direction, &t, &x, device)?;
        assert_eq!(out.shape(), vec![2, 3, width as i32], "{direction:?}");
        let d = max_abs_diff_f32(&read_f32(&out)?, &oracle(direction, &widen(&x32), &s));
        assert!(
            d <= F32_BOUND,
            "{direction:?} of [2, 3, {width}] misses the oracle by {d:.3e}"
        );
    }
    Ok(())
}

const LOW_WIDTH: usize = 5120;
const LOW_ROWS: usize = 4;

/// One low-precision case: the input, the oracle and the oracle rounded to
/// the input dtype.
struct LowPrecisionCase {
    x: Array,
    oracle: Vec<f64>,
    rounded: Vec<f64>,
}

impl LowPrecisionCase {
    fn new(dtype: Dtype, direction: Direction, s: &[f32]) -> Result<Self> {
        let x = Array::from_f32_slice(
            &uniform(701, LOW_ROWS * LOW_WIDTH),
            &[LOW_ROWS as i32, LOW_WIDTH as i32],
        )?
        .astype(dtype, Device::Cpu)?;
        let oracle = oracle(direction, &widen(&read_f32(&x)?), s);
        let rounded = oracle.iter().map(|&v| round_to(dtype, v)).collect();
        Ok(Self { x, oracle, rounded })
    }

    fn exact_share(&self, out: &[f32]) -> f64 {
        out.iter()
            .zip(&self.rounded)
            .filter(|(a, b)| f64::from(**a).to_bits() == b.to_bits())
            .count() as f64
            / out.len() as f64
    }
}

/// A bf16 or f16 activation is widened to f32, transformed and rounded once,
/// in both directions: at least 99.5% of outputs equal the rounded oracle and
/// none is more than one spacing of the dtype away.
fn check_low_precision(device: Device) -> Result<()> {
    let s = signs(700, LOW_WIDTH);
    let t = SignedBlockHadamard::new(BLOCK, &s)?;
    for dtype in [Dtype::Bf16, Dtype::F16] {
        for direction in [Direction::Forward, Direction::Inverse] {
            let case = LowPrecisionCase::new(dtype, direction, &s)?;
            let label = format!("{direction:?} {dtype:?}");
            let out = run(direction, &t, &case.x, device)?;
            assert_eq!(
                out.dtype(),
                dtype,
                "{label}: the output keeps the input dtype"
            );
            let got = read_f32(&out)?;
            let share = case.exact_share(&got);
            assert!(
                share >= 0.995,
                "{label}: only {share:.4} of outputs equal the rounded oracle"
            );
            for (g, o) in got.iter().zip(&case.oracle) {
                let d = (f64::from(*g) - o).abs();
                assert!(
                    d <= spacing(dtype, *o) + 1e-6,
                    "{label}: output {g} is {d:.3e} from oracle {o}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn block_four_by_hand() -> Result<()> {
    check_block_four_by_hand(Device::Cpu)
}

#[test]
fn forward_matches_the_f64_oracle_at_each_pack_width() -> Result<()> {
    check_forward_at_each_pack_width(Device::Cpu)
}

#[test]
fn forward_of_a_signed_basis_vector_is_a_sylvester_row() -> Result<()> {
    check_signed_basis_vector(Device::Cpu)
}

#[test]
fn inverse_matches_the_oracle_and_undoes_forward() -> Result<()> {
    check_inverse_at_each_pack_width(Device::Cpu)
}

#[test]
fn leading_axes_are_kept() -> Result<()> {
    check_leading_axes(Device::Cpu)
}

#[test]
fn a_low_precision_input_is_transformed_in_f32_and_rounded_once() -> Result<()> {
    check_low_precision(Device::Cpu)
}

/// MLX's CPU op keeps its butterfly in the input dtype, so a transform run in
/// bf16 or f16 rounds at each of the ten stages. That transform must miss the
/// exact-share bound of the low-precision test, in both directions.
#[test]
fn a_transform_run_in_the_input_dtype_misses_the_low_precision_bound() -> Result<()> {
    let s = signs(700, LOW_WIDTH);
    for dtype in [Dtype::Bf16, Dtype::F16] {
        let s_low = Array::from_f32_slice(&s, &[LOW_WIDTH as i32])?.astype(dtype, Device::Cpu)?;
        for direction in [Direction::Forward, Direction::Inverse] {
            let case = LowPrecisionCase::new(dtype, direction, &s)?;
            let in_dtype = match direction {
                Direction::Forward => {
                    let signed = rmlx_mlx::multiply(&case.x, &s_low, Device::Cpu)?;
                    let blocks = signed.reshape(&[-1, BLOCK as i32], Device::Cpu)?;
                    rmlx_mlx::hadamard_transform(&blocks, Some(1.0 / 32.0), Device::Cpu)?
                }
                Direction::Inverse => {
                    let blocks = case.x.reshape(&[-1, BLOCK as i32], Device::Cpu)?;
                    let h = rmlx_mlx::hadamard_transform(&blocks, Some(1.0 / 32.0), Device::Cpu)?;
                    let h = h.reshape(&[LOW_ROWS as i32, LOW_WIDTH as i32], Device::Cpu)?;
                    rmlx_mlx::multiply(&h, &s_low, Device::Cpu)?
                }
            };
            assert_eq!(in_dtype.dtype(), dtype, "the control runs in {dtype:?}");
            let share = case.exact_share(&read_f32(&in_dtype)?);
            assert!(
                share < 0.9,
                "{direction:?} {dtype:?}: a transform run in {dtype:?} equals the rounded \
                 oracle on {share:.4} of outputs; the bound cannot tell it from the f32 path"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "Metal: GPU twin of block_four_by_hand; needs the Metal claim (make gpu-test)"]
fn hadamard_block_four_by_hand_on_gpu() -> Result<()> {
    check_block_four_by_hand(Device::Gpu)
}

#[test]
#[ignore = "Metal: GPU twin of forward_matches_the_f64_oracle_at_each_pack_width; needs the Metal claim (make gpu-test)"]
fn hadamard_forward_matches_the_f64_oracle_at_each_pack_width_on_gpu() -> Result<()> {
    check_forward_at_each_pack_width(Device::Gpu)
}

#[test]
#[ignore = "Metal: GPU twin of forward_of_a_signed_basis_vector_is_a_sylvester_row; needs the Metal claim (make gpu-test)"]
fn hadamard_forward_of_a_signed_basis_vector_is_a_sylvester_row_on_gpu() -> Result<()> {
    check_signed_basis_vector(Device::Gpu)
}

#[test]
#[ignore = "Metal: GPU twin of inverse_matches_the_oracle_and_undoes_forward; needs the Metal claim (make gpu-test)"]
fn hadamard_inverse_matches_the_oracle_and_undoes_forward_on_gpu() -> Result<()> {
    check_inverse_at_each_pack_width(Device::Gpu)
}

#[test]
#[ignore = "Metal: GPU twin of leading_axes_are_kept; needs the Metal claim (make gpu-test)"]
fn hadamard_leading_axes_are_kept_on_gpu() -> Result<()> {
    check_leading_axes(Device::Gpu)
}

#[test]
#[ignore = "Metal: GPU twin of a_low_precision_input_is_transformed_in_f32_and_rounded_once; needs the Metal claim (make gpu-test)"]
fn hadamard_a_low_precision_input_is_transformed_in_f32_and_rounded_once_on_gpu() -> Result<()> {
    check_low_precision(Device::Gpu)
}

fn refused(r: &Result<SignedBlockHadamard>) -> bool {
    matches!(r, Err(Error::Model(_)))
}

#[test]
fn new_refuses_what_the_contract_does_not_define() {
    assert!(
        refused(&SignedBlockHadamard::new(1000, &signs(800, 2000))),
        "block not a power of two"
    );
    assert!(
        refused(&SignedBlockHadamard::new(0, &signs(801, 1024))),
        "block 0"
    );
    assert!(
        refused(&SignedBlockHadamard::new(1, &signs(801, 1024))),
        "block 1"
    );
    assert!(
        refused(&SignedBlockHadamard::new(8192, &signs(801, 8192))),
        "block above the largest the kernel is checked at"
    );
    assert!(
        refused(&SignedBlockHadamard::new(BLOCK, &signs(802, 1536))),
        "width not a multiple of block"
    );
    assert!(refused(&SignedBlockHadamard::new(BLOCK, &[])), "no signs");
    for bad in [0.0, 2.0, -0.5, f32::NAN] {
        let s: Vec<f32> = signs(803, 2 * BLOCK)
            .into_iter()
            .enumerate()
            .map(|(n, v)| if n == 1500 { bad } else { v })
            .collect();
        assert!(
            refused(&SignedBlockHadamard::new(BLOCK, &s)),
            "sign value {bad}"
        );
    }
}

#[test]
fn forward_and_inverse_refuse_an_input_that_is_not_the_sign_width() -> Result<()> {
    let t = SignedBlockHadamard::new(BLOCK, &signs(900, 5120))?;
    for width in [4096, 6144] {
        let x = Array::from_f32_slice(&uniform(901, width), &[1, width as i32])?;
        for (what, r) in [
            ("forward", t.forward(&x, Device::Cpu)),
            ("inverse", t.inverse(&x, Device::Cpu)),
        ] {
            assert!(
                matches!(&r, Err(Error::Model(m)) if m.contains("5120") && m.contains(&width.to_string())),
                "{what} of a width-{width} input must refuse naming both widths, got {:?}",
                r.err()
            );
        }
    }
    Ok(())
}

const BONSAI_2_SLUG: &str = "prism-ml__Ternary-Bonsai-2-27B-mlx-2bit";

/// The real pack's contract, read on the host: `config.json`, `hadamard.json`,
/// the shard header and the 402 `.signs` tensors, no weight. 401 forward
/// modules (273 of width 5120, 64 of 6144, 64 of 17408), the embedding as the
/// one inverse module, and a transform built from each sign vector. The pack
/// check that `load_model` and `serve` run passes it.
#[test]
fn the_ternary_bonsai_2_pack_contract_reads_and_builds_its_transforms() {
    let test = "the_ternary_bonsai_2_pack_contract_reads_and_builds_its_transforms";
    let Some(dir) =
        crate::test_snapshot::snapshot_by_slug(test, BONSAI_2_SLUG, &["prism_hadamard_qwen35"])
    else {
        return;
    };
    let contract = match rmlx_loader::HadamardContract::read(&dir) {
        Ok(Some(c)) => c,
        other => panic!("{}: want a contract, got {other:?}", dir.display()),
    };
    assert_eq!(contract.block, BLOCK);
    assert_eq!(contract.base_model_type, "qwen3_5");
    assert_eq!(contract.tensor_prefix, "language_model.");
    assert_eq!(
        contract.signs.keys().copied().collect::<Vec<_>>(),
        PACK_WIDTHS.to_vec()
    );
    let per_width: Vec<usize> = PACK_WIDTHS
        .iter()
        .map(|w| contract.forward.values().filter(|v| *v == w).count())
        .collect();
    assert_eq!(contract.forward.len(), 401);
    assert_eq!(per_width, vec![273, 64, 64]);
    assert_eq!(
        contract.inverse.iter().collect::<Vec<_>>(),
        vec![(&"language_model.model.embed_tokens".to_owned(), &5120)]
    );
    for (width, s) in &contract.signs {
        if let Err(e) = SignedBlockHadamard::new(contract.block, s) {
            panic!("width {width}: the pack's signs build no transform: {e}");
        }
    }
    let cfg = rmlx_loader::load_config(&dir)
        .unwrap_or_else(|e| panic!("{}: config.json: {e}", dir.display()));
    if let Err(e) = crate::arch::check_hadamard_pack(&cfg, &dir) {
        panic!("the pack check refuses the real pack: {e}");
    }
}
