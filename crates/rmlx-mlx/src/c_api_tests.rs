use super::*;

#[test]
fn only_the_same_c_api_on_both_sides_is_a_match() {
    for compiled in CApi::ALL {
        for loaded in CApi::ALL {
            let want = if compiled == loaded {
                CApiVerdict::Match(compiled)
            } else {
                CApiVerdict::Mismatch { compiled, loaded }
            };
            assert_eq!(CApiVerdict::classify(compiled, loaded), want);
        }
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
        "C API mismatch",
        "compiled against the mlx-c 0.6 C API",
        "has the mlx-c 0.7 C API",
        "mlx_fast_scaled_dot_product_attention",
        "mlx_gather_qmm",
        "cargo clean -p rmlx-mlx",
        "brew reinstall rmlx",
        "release tarball",
    ] {
        assert!(err.contains(needle), "{needle:?} missing from: {err}");
    }
}

/// Each C API has its own name in the message, so a mismatch between any two
/// reads as one.
#[test]
fn every_c_api_has_its_own_name() {
    let names: Vec<_> = CApi::ALL.iter().map(|api| api.name()).collect();
    assert_eq!(names, ["mlx-c 0.6", "mlx-c 0.7", "mlx-c 0.8"]);
}

/// Positive and negative control of the probe itself: a probe that answers
/// the same for every name cannot tell the C APIs apart.
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
