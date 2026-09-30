use super::*;

#[test]
fn only_the_same_c_api_on_both_sides_is_a_match() {
    for compiled in [CApi::V0_6, CApi::V0_7] {
        assert_eq!(
            CApiVerdict::classify(compiled, compiled),
            CApiVerdict::Match(compiled)
        );
        assert_eq!(
            CApiVerdict::classify(compiled, compiled.other()),
            CApiVerdict::Mismatch {
                compiled,
                loaded: compiled.other()
            }
        );
    }
}

#[test]
fn a_mismatch_is_an_error_that_names_both_c_apis_and_the_call() {
    assert!(CApiVerdict::Match(CApi::V0_7).require_match().is_ok());
    let err = CApiVerdict::Mismatch {
        compiled: CApi::V0_6,
        loaded: CApi::V0_7,
    }
    .require_match()
    .expect_err("a mismatch must not pass")
    .to_string();
    for needle in [
        "compiled against the mlx-c 0.6 C API",
        "has the mlx-c 0.7 C API",
        "mlx_fast_scaled_dot_product_attention",
        "cargo clean -p rmlx-mlx",
    ] {
        assert!(err.contains(needle), "{needle:?} missing from: {err}");
    }
}

/// Positive and negative control of the probe itself: a probe that answers
/// the same for every name cannot tell the two C APIs apart.
#[test]
fn the_export_probe_finds_a_present_symbol_and_not_an_absent_one() {
    assert!(process_exports("mlx_array_new"));
    assert!(!process_exports("mlx_rmlx_no_such_function"));
}

/// On a matched pair the probe must see the C API the build compiled. A
/// binary built against one mlx-c and run against the other fails here by
/// design: that is the mismatch the probe exists to find.
#[test]
fn the_loaded_c_api_is_the_compiled_one() {
    assert_eq!(
        verdict(),
        CApiVerdict::Match(CApi::COMPILED),
        "the loaded libmlxc.dylib has another C API than the one compiled in"
    );
}
