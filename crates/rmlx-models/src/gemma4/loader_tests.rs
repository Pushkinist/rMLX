use super::classify_load_oom;
use rmlx_core::{Error, OomPhase};

#[test]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn j3_oom_classify_alloc_string_becomes_oom() {
    // The exact shape MLX throws on a Metal allocation failure.
    let e = Error::Mlx("Array::eval: [malloc_or_wait] Unable to allocate 9000000000 bytes".into());
    match classify_load_oom(e) {
        Error::Oom { phase, .. } => assert_eq!(phase, OomPhase::LoadWeights),
        other => panic!("expected Error::Oom, got {other:?}"),
    }
}

#[test]
fn j3_oom_classify_shape_error_stays_mlx() {
    // A non-OOM MLX failure must NOT be misclassified as OOM (false 507 +
    // wrong "evict & retry" is worse than an honest generic error).
    let e = Error::Mlx("reshape: total size mismatch 12 vs 16".into());
    assert!(
        matches!(classify_load_oom(e), Error::Mlx(_)),
        "shape error must stay Error::Mlx"
    );
}

#[test]
fn j3_oom_classify_non_mlx_untouched() {
    let e = Error::Loader("missing config.json".into());
    assert!(matches!(classify_load_oom(e), Error::Loader(_)));
}
