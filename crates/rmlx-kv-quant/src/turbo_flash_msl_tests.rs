use super::*;
use rmlx_core::DispatchPolicy;

#[test]
fn test_turbo_flash_disabled_by_default() {
    // The default policy selects the generic path. This guards against
    // accidentally flipping default-ON.
    assert!(
        !DispatchPolicy::default().turbo_flash,
        "TurboFlash must be default-OFF"
    );
}

#[test]
fn test_smoke_probe_no_corruption() {
    // Clean output — diverse token IDs, no run of 4.
    let tokens = [1u32, 5, 3, 7, 2, 8, 4, 6, 9, 10, 11, 12];
    assert!(!smoke_probe_check(&tokens), "should not detect corruption");
}

#[test]
fn test_smoke_probe_detects_corruption() {
    // Run of 4 identical tokens — corruption signature.
    // Use a temporary AtomicBool state by resetting after test.
    let saved = FORCED_FALLBACK.load(Ordering::Relaxed);
    FORCED_FALLBACK.store(false, Ordering::Relaxed);
    let tokens = [1u32, 5, 106, 106, 106, 106, 3];
    assert!(
        smoke_probe_check(&tokens),
        "should detect corruption (run of 4 identical token 106)"
    );
    // Restore state (tests may run in any order).
    FORCED_FALLBACK.store(saved, Ordering::Relaxed);
}

#[test]
fn test_turbo_flash_should_run_gated() {
    // should_run requires:
    // 1. policy.turbo_flash
    // 2. !corrupted
    // 3. q_seq == 1
    // 4. kv_seq > policy.turbo_flash_min_kv_seq

    let off = DispatchPolicy::default();
    assert!(!turbo_flash_should_run(&off, 1, 8192));
    assert!(!turbo_flash_should_run(&off, 1, 100));

    // Each remaining condition is checked against a policy that satisfies the
    // gate, so a dropped condition cannot pass by accident.
    let on = DispatchPolicy {
        turbo_flash: true,
        ..DispatchPolicy::default()
    };
    assert!(turbo_flash_should_run(&on, 1, 8192), "gate must open");
    assert!(
        !turbo_flash_should_run(&on, 1, 100),
        "kv_seq below the policy threshold must not run"
    );
    assert!(
        !turbo_flash_should_run(&on, 2, 8192),
        "prefill (q_seq > 1) must not run"
    );
    let low_threshold = DispatchPolicy {
        turbo_flash: true,
        turbo_flash_min_kv_seq: 0,
        ..DispatchPolicy::default()
    };
    assert!(
        turbo_flash_should_run(&low_threshold, 1, 100),
        "the threshold must come from the policy, not a constant"
    );
}

// ── Kernel vs. its codec reference ────────────────────────────────────────────
//
// The correctness question the `--turbo-flash` HOLD could not answer: does the
// kernel compute the right thing *for its codec*? Turning the gate off does not
// answer it — on K8V4 the generic path reads the bf16 mirror and never touches
// the 4-bit V store, so a gate-off run is a bf16 attention that any correct
// tq4-V kernel must also differ from. `turbo_flash_reference_sdpa` unpacks the
// same packed buffers and runs an ordinary SDPA, so the codec's error is common
// to both arms and only the kernel's own arithmetic is left.

/// Deterministic pseudo-random data in [-1, 1). Same LCG constants as the
/// other kernel integration tests so a failing seed round-trips across reports.
fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 32) as u32 as f32 / u32::MAX as f32).mul_add(2.0, -1.0)
        })
        .collect()
}

#[allow(
    clippy::expect_used,
    reason = "test helper: fixed in-bounds shapes cannot fail to build; expect documents that"
)]
fn bf16_from(data: &[f32], shape: &[i32], device: Device) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32)
        .expect("from_bytes")
        .astype(Dtype::Bf16, device)
        .expect("astype bf16")
}

#[allow(
    clippy::expect_used,
    reason = "test helper: the array is built in this test and is materialisable; expect documents that"
)]
fn collect_f32(a: &Array, device: Device) -> Vec<f32> {
    let a = a.astype(Dtype::F32, device).expect("astype f32");
    a.eval().expect("eval");
    a.to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Per-row cosine (`row_len` = head_dim) — min over rows.
#[allow(
    clippy::indexing_slicing,
    reason = "test: row bounds are a * exact multiple of row_len, established two lines above"
)]
fn min_row_cosine(a: &[f32], b: &[f32], row_len: usize) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine: length mismatch");
    assert!(
        row_len > 0 && a.len().is_multiple_of(row_len),
        "cosine: ragged rows"
    );
    let mut worst = f32::INFINITY;
    for r in 0..(a.len() / row_len) {
        let (ra, rb) = (
            &a[r * row_len..(r + 1) * row_len],
            &b[r * row_len..(r + 1) * row_len],
        );
        let dot: f64 = ra
            .iter()
            .zip(rb)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let na: f64 = ra.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = rb.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let c = if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            (dot / (na * nb)) as f32
        };
        worst = worst.min(c);
    }
    worst
}

/// Tolerance for kernel-vs-reference. Both arms consume identical codes and
/// scales, so the codec's ~0.997 floor is *not* in this number: what is left is
/// block tiling, online-softmax ordering and the bf16 rounding of the two
/// outputs. Measured 0.9999956 (Bonsai-8B geometry) and 0.9999962 (Qwen3.6-35B
/// geometry), so this floor sits ~2x tighter in `1 - cos` than the observed
/// error and three orders of magnitude inside the ~0.997 codec floor a kernel
/// defect would otherwise have room to hide under.
const KERNEL_VS_REFERENCE_MIN_COSINE: f32 = 0.99999;

/// Companion bound on the elementwise difference, stated in bf16 ULPs at the
/// reference's own magnitude rather than as an absolute number — an absolute
/// tolerance passes or fails on how large the attention output happens to be.
/// bf16 keeps 8 mantissa bits, so one ULP at magnitude `m` is `m * 2^-8`.
/// Measured 1.4 ULP (Bonsai-8B geometry) and 1.7 ULP (Qwen3.6-35B geometry).
const KERNEL_VS_REFERENCE_MAX_ULPS: f32 = 4.0;

/// One cell of the kernel-vs-reference comparison.
///
/// `t_active` is deliberately not a multiple of the kernel's 64-token block and
/// `t_stride` is deliberately larger than it: that is the production shape (a
/// ring provisioned to `max_seq`, filled to `kv_seq`), and it is the shape in
/// which a stride or tail-block bug shows up.
#[allow(
    clippy::expect_used,
    reason = "test: every fallible call here is on a fixture built in this fn; expect documents the invariant"
)]
fn assert_kernel_matches_reference(head_dim: i32, kv_h: i32, heads_per_kv: i32) {
    let device = Device::Gpu;
    let b = 1i32;
    let n_q_heads = kv_h * heads_per_kv;
    let t_active = 200i32;
    let t_stride = 256i32;

    let kv_shape = [b, kv_h, t_stride, head_dim];
    let n_kv = kv_shape.iter().map(|&d| d as usize).product::<usize>();
    let k_bf16 = bf16_from(&lcg(n_kv, 0x7F1A_2B3C_4D5E_0001), &kv_shape, device);
    let v_bf16 = bf16_from(&lcg(n_kv, 0x7F1A_2B3C_4D5E_0002), &kv_shape, device);

    let (k_codes, k_scales) =
        crate::q8_msl::q8_quantize_gpu(&k_bf16, device).expect("q8 quantize K");
    let (v_codes, v_scales) =
        crate::turboquant_msl::turbo_quantize_v4_gpu(&v_bf16, device).expect("tq4 quantize V");

    let q_shape = [b, n_q_heads, 1, head_dim];
    let n_q = q_shape.iter().map(|&d| d as usize).product::<usize>();
    let q_raw = bf16_from(&lcg(n_q, 0x7F1A_2B3C_4D5E_0003), &q_shape, device);
    // Pre-scaled, as both arms' contract requires.
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = {
        let sc = rmlx_mlx::scalar_f32(scale)
            .astype(Dtype::Bf16, device)
            .expect("scale cast");
        rmlx_mlx::multiply(&q_raw, &sc, device).expect("pre-scale Q")
    };

    let before = turbo_flash_dispatch_count();
    let kernel = turbo_flash_sdpa(
        &q, &k_codes, &k_scales, &v_codes, &v_scales, None, b, n_q_heads, kv_h, t_active, t_stride,
        head_dim, device,
    )
    .expect("turbo_flash_sdpa");
    let after = turbo_flash_dispatch_count();
    assert!(
        after > before,
        "head_dim={head_dim}: the kernel must actually dispatch, else this \
         comparison proves nothing"
    );

    let reference = turbo_flash_reference_sdpa(
        &q, &k_codes, &k_scales, &v_codes, &v_scales, None, b, n_q_heads, kv_h, t_active, t_stride,
        head_dim, device,
    )
    .expect("turbo_flash_reference_sdpa");
    assert_eq!(
        turbo_flash_dispatch_count(),
        after,
        "head_dim={head_dim}: the reference arm must not dispatch the kernel — \
         a reference that reached the same code would compare it with itself"
    );

    assert_eq!(
        kernel.shape(),
        reference.shape(),
        "head_dim={head_dim}: arms must agree on output shape"
    );
    assert_eq!(
        kernel.dtype(),
        reference.dtype(),
        "head_dim={head_dim}: arms must agree on output dtype — comparing a \
         promoted arm against a bf16 one measures the promotion, not the kernel"
    );

    let a = collect_f32(&kernel, device);
    let r = collect_f32(&reference, device);
    let cos = min_row_cosine(&a, &r, head_dim as usize);
    let max_abs = a
        .iter()
        .zip(&r)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let ref_scale = r.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);

    // One bf16 ULP at the reference's magnitude: bf16 keeps 8 mantissa bits.
    let ulp = ref_scale * f32::exp2(-8.0);
    let ulps = max_abs / ulp;

    assert!(
        cos >= KERNEL_VS_REFERENCE_MIN_COSINE,
        "head_dim={head_dim}: kernel vs its own codec reference cosine {cos} \
         below {KERNEL_VS_REFERENCE_MIN_COSINE} (max abs diff {max_abs} = \
         {ulps} bf16 ULP, |ref|max {ref_scale}) — this is the kernel's own \
         error, the codec's cancels between the arms"
    );
    assert!(
        ulps <= KERNEL_VS_REFERENCE_MAX_ULPS,
        "head_dim={head_dim}: kernel vs its own codec reference differs by \
         {ulps} bf16 ULP (max abs {max_abs}, |ref|max {ref_scale}), above \
         {KERNEL_VS_REFERENCE_MAX_ULPS} — cosine {cos} can stay high while a \
         single element is wrong, so both bounds are asserted"
    );
}

// The two shapes below are the attention geometries of the two architectures
// on which the kernel actually dispatches, taken from their `config.json`.
// The kernel keys off shape, never off an arch name, so naming them here is
// coverage bookkeeping — it is what makes this "two architectures" and not two
// arbitrary head_dims.

/// Ternary-Bonsai-8B-2bit geometry: `head_dim` 128, 8 KV heads, 32 query heads.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --tests -- --ignored --test-threads=1"]
fn turbo_flash_matches_its_codec_reference_at_bonsai_8b_geometry() {
    assert_kernel_matches_reference(128, 8, 4);
}

/// Qwen3.6-35B-A3B geometry: `head_dim` 256, 2 KV heads, 16 query heads — the
/// other head_dim the kernel is wired for, and a much wider GQA fan-out.
#[test]
#[ignore = "GPU Metal context: cargo test -p rmlx-kv-quant --tests -- --ignored --test-threads=1"]
fn turbo_flash_matches_its_codec_reference_at_qwen36_35b_geometry() {
    assert_kernel_matches_reference(256, 2, 8);
}

/// The reference must refuse exactly what the kernel refuses, *for the same
/// stated reason*.
///
/// Asserting "both returned Err" is not enough and was tried: with the
/// `head_dim ∈ {128, 256}` rule deleted the arms still both errored — one from
/// deeper in the kernel, one from the dequant — and a both-are-Err check stayed
/// green through the mutation. Comparing the reason text is what makes it fail.
// gpu-test-gate: exempt
#[test]
fn reference_and_kernel_refuse_the_same_shapes_for_the_same_reason() {
    let device = Device::Gpu;
    // A one-element dummy is enough: both arms validate shape before touching
    // any buffer, so nothing is dispatched on these paths.
    #[allow(
        clippy::expect_used,
        reason = "test: a 1-element array cannot fail to build; expect documents that"
    )]
    let dummy = Array::from_bytes(&[0u8; 4], &[1], Dtype::F32).expect("dummy");
    let before = turbo_flash_dispatch_count();
    for (head_dim, t_active, t_stride) in [(64i32, 8i32, 8i32), (100, 8, 8), (128, 16, 8)] {
        let kernel = turbo_flash_sdpa(
            &dummy, &dummy, &dummy, &dummy, &dummy, None, 1, 1, 1, t_active, t_stride, head_dim,
            device,
        );
        let reference = turbo_flash_reference_sdpa(
            &dummy, &dummy, &dummy, &dummy, &dummy, None, 1, 1, 1, t_active, t_stride, head_dim,
            device,
        );
        let cell = format!("head_dim={head_dim} t_active={t_active} t_stride={t_stride}");
        let (Err(ke), Err(re)) = (kernel, reference) else {
            panic!("{cell}: both arms must refuse");
        };
        // The two messages name their own caller and are otherwise the same
        // sentence; normalise the one name that differs by construction.
        let reason = |m: String| m.replace("turbo_flash_reference_sdpa", "turbo_flash_sdpa");
        assert_eq!(
            reason(ke.to_string()),
            reason(re.to_string()),
            "{cell}: the arms refused for different reasons — the reference is \
             only a reference where it accepts exactly what the kernel accepts"
        );
    }
    assert_eq!(
        turbo_flash_dispatch_count(),
        before,
        "a refused shape must not have reached the kernel"
    );
}
