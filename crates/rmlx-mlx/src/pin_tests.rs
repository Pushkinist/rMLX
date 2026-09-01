//! Tests for the loaded-MLX pin gate.
//!
//! Two layers, and they cover different things. The classifier is pure and its
//! whole matrix is asserted from injected observations, so every verdict —
//! including the ones this machine cannot produce — is exercised. The gate
//! itself reads the real process and is the only thing that can catch a
//! symlink that moved after the build.

use std::path::{Path, PathBuf};

use rmlx_core::apple_gpu::apple_silicon_generation;

use super::{
    keg_version_from, parse_pin, pin_check, verdict, KernelScan, LinkedPair, MlxPin, PinVerdict,
    PIN_SRC,
};
use crate::nax::{loaded_library_path, NAX_GEMM_KERNEL};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn pin() -> MlxPin {
    MlxPin {
        mlx: "0.31.2".to_owned(),
        mlx_c: "0.6.0_2".to_owned(),
    }
}

/// An observation of the pinned pair, correctly resolved and scanned. Every
/// case below starts here and breaks exactly one thing, so the assertion is
/// about that one thing and not about the fixture.
fn pinned_pair() -> LinkedPair {
    LinkedPair {
        mlx_path: Some(PathBuf::from(
            "/opt/homebrew/Cellar/mlx/0.31.2/lib/libmlx.dylib",
        )),
        mlx_c_path: Some(PathBuf::from(
            "/opt/homebrew/Cellar/mlx-c/0.6.0_2/lib/libmlxc.dylib",
        )),
        mlx_keg: Some("0.31.2".to_owned()),
        mlx_c_keg: Some("0.6.0_2".to_owned()),
        kernels: KernelScan::Scanned {
            present: true,
            metallib: PathBuf::from("/opt/homebrew/Cellar/mlx/0.31.2/lib/mlx.metallib"),
        },
    }
}

// ---------------------------------------------------------------------------
// Pin file parsing
// ---------------------------------------------------------------------------

#[test]
fn the_checked_in_pin_declares_both_formulas() {
    // `verdict()` reads this file at compile time. A pin that does not parse
    // makes the gate unable to check anything, so it must fail here first.
    let Some(parsed) = parse_pin(PIN_SRC) else {
        panic!("the checked-in mlx-pin.txt must declare both formulas");
    };
    assert!(!parsed.mlx.is_empty() && !parsed.mlx_c.is_empty());
}

#[test]
fn parse_pin_ignores_comments_and_blank_lines() {
    let src = "# a comment\n\n  mlx    0.31.2  \n# another\nmlx-c  0.6.0_2\n";
    assert_eq!(parse_pin(src), Some(pin()));
}

#[test]
fn parse_pin_rejects_a_half_declared_pair() {
    // Pinning one half is the exact mistake the pair exists to prevent.
    assert_eq!(parse_pin("mlx 0.31.2\n"), None);
    assert_eq!(parse_pin("mlx-c 0.6.0_2\n"), None);
}

#[test]
fn parse_pin_rejects_unknown_or_malformed_lines() {
    assert_eq!(parse_pin("mlx 0.31.2\nmlx-c 0.6.0_2\nmlx-rs 1.0\n"), None);
    assert_eq!(parse_pin("mlx\nmlx-c 0.6.0_2\n"), None);
    assert_eq!(parse_pin("mlx 0.31.2 extra\nmlx-c 0.6.0_2\n"), None);
}

#[test]
fn an_unparsable_pin_says_so_rather_than_checking_nothing() {
    // A broken pin file is a defect in the tree, not a property of the host,
    // and it leaves the gate with nothing to compare against. It gets its own
    // verdict so it can never be read as a pass.
    let report = PinVerdict::PinUnparsable.report();
    assert!(report.contains("mlx-pin.txt"), "{report}");
    assert!(!PinVerdict::PinUnparsable.is_match());
}

// ---------------------------------------------------------------------------
// Keg version reading
// ---------------------------------------------------------------------------

#[test]
fn keg_version_reads_the_homebrew_revision_suffix() {
    // The `_2` suffix is the load-bearing part: same upstream 0.6.0, built
    // against a different mlx.
    assert_eq!(
        keg_version_from(Path::new("/opt/homebrew/Cellar/mlx-c/0.6.0_2"), "mlx-c"),
        Some("0.6.0_2".to_owned())
    );
    assert_eq!(
        keg_version_from(Path::new("/opt/homebrew/Cellar/mlx/0.31.2"), "mlx"),
        Some("0.31.2".to_owned())
    );
}

#[test]
fn keg_version_declines_non_keg_layouts() {
    assert_eq!(
        keg_version_from(Path::new("/usr/local/mlx-c"), "mlx-c"),
        None
    );
    assert_eq!(keg_version_from(Path::new("/"), "mlx"), None);
    // Right leaf, wrong formula — a `mlx-c` keg must not be read as `mlx`.
    assert_eq!(
        keg_version_from(Path::new("/opt/homebrew/Cellar/mlx/0.31.2"), "mlx-c"),
        None
    );
}

// ---------------------------------------------------------------------------
// The classifier matrix
// ---------------------------------------------------------------------------

#[test]
fn the_pinned_pair_with_the_kernels_is_the_only_match() {
    let observed = pinned_pair();
    assert_eq!(
        observed.classify(&pin()),
        PinVerdict::Match {
            mlx: "0.31.2".to_owned(),
            mlx_c: "0.6.0_2".to_owned(),
            metallib: PathBuf::from("/opt/homebrew/Cellar/mlx/0.31.2/lib/mlx.metallib"),
        }
    );
    assert!(observed.classify(&pin()).is_match());
}

#[test]
fn a_metallib_without_the_kernels_outranks_every_other_finding() {
    // The expensive, invisible failure. It is reported even when the version
    // also disagrees, because the version is the cheaper thing to notice.
    let observed = LinkedPair {
        mlx_keg: Some("0.32.0".to_owned()),
        mlx_path: Some(PathBuf::from(
            "/opt/homebrew/Cellar/mlx/0.32.0/lib/libmlx.dylib",
        )),
        kernels: KernelScan::Scanned {
            present: false,
            metallib: PathBuf::from("/opt/homebrew/Cellar/mlx/0.32.0/lib/mlx.metallib"),
        },
        ..pinned_pair()
    };
    let got = observed.classify(&pin());
    assert_eq!(
        got,
        PinVerdict::KernelsMissing {
            metallib: PathBuf::from("/opt/homebrew/Cellar/mlx/0.32.0/lib/mlx.metallib"),
            mlx: Some("0.32.0".to_owned()),
        }
    );
    assert!(got.report().contains(NAX_GEMM_KERNEL));
    assert!(!got.is_match());
}

#[test]
fn either_half_of_the_pair_drifting_is_a_mismatch() {
    // The pair is the validated unit: mlx and mlx-c are ABI-coupled.
    let drifted_mlx = LinkedPair {
        mlx_keg: Some("0.32.0".to_owned()),
        ..pinned_pair()
    };
    assert_eq!(
        drifted_mlx.classify(&pin()),
        PinVerdict::VersionMismatch {
            formula: "mlx",
            resolved: "0.32.0".to_owned(),
            pinned: "0.31.2".to_owned(),
        }
    );
    let drifted_mlx_c = LinkedPair {
        mlx_c_keg: Some("0.6.0_3".to_owned()),
        ..pinned_pair()
    };
    assert_eq!(
        drifted_mlx_c.classify(&pin()),
        PinVerdict::VersionMismatch {
            formula: "mlx-c",
            resolved: "0.6.0_3".to_owned(),
            pinned: "0.6.0_2".to_owned(),
        }
    );
}

#[test]
fn a_library_dyld_never_listed_is_not_a_version_finding() {
    // "Could not look" must not be classified as "looked and disagreed".
    for (broken, library) in [
        (
            LinkedPair {
                mlx_path: None,
                ..pinned_pair()
            },
            "libmlx.dylib",
        ),
        (
            LinkedPair {
                mlx_c_path: None,
                ..pinned_pair()
            },
            "libmlxc.dylib",
        ),
    ] {
        let got = broken.classify(&pin());
        assert_eq!(got, PinVerdict::NotLoaded { library });
        assert!(got.report().contains(library));
    }
}

#[test]
fn a_non_keg_layout_is_unverifiable_rather_than_matching() {
    // A hand-built tree or a wheel has no version to compare. Passing it would
    // be a green that means "I could not check".
    let mlx_hand_built = LinkedPair {
        mlx_keg: None,
        mlx_path: Some(PathBuf::from("/usr/local/mlx/lib/libmlx.dylib")),
        ..pinned_pair()
    };
    assert_eq!(
        mlx_hand_built.classify(&pin()),
        PinVerdict::NotAKeg {
            formula: "mlx",
            resolved: Some(PathBuf::from("/usr/local/mlx/lib/libmlx.dylib")),
        }
    );
    let mlx_c_hand_built = LinkedPair {
        mlx_c_keg: None,
        mlx_c_path: Some(PathBuf::from("/usr/local/mlx-c/lib/libmlxc.dylib")),
        ..pinned_pair()
    };
    assert_eq!(
        mlx_c_hand_built.classify(&pin()),
        PinVerdict::NotAKeg {
            formula: "mlx-c",
            resolved: Some(PathBuf::from("/usr/local/mlx-c/lib/libmlxc.dylib")),
        }
    );
    assert!(!mlx_c_hand_built.classify(&pin()).is_match());
}

#[test]
fn an_unreadable_metallib_is_unverified_not_absent() {
    // The distinction the whole gate turns on: a scan that could not run has
    // established nothing, and must not read as either presence or absence.
    let unreadable = LinkedPair {
        kernels: KernelScan::Unverified {
            metallib: Some(PathBuf::from(
                "/opt/homebrew/Cellar/mlx/0.31.2/lib/mlx.metallib",
            )),
        },
        ..pinned_pair()
    };
    assert_eq!(
        unreadable.classify(&pin()),
        PinVerdict::KernelsUnverified {
            metallib: Some(PathBuf::from(
                "/opt/homebrew/Cellar/mlx/0.31.2/lib/mlx.metallib"
            )),
        }
    );
    let no_path = LinkedPair {
        kernels: KernelScan::Unverified { metallib: None },
        ..pinned_pair()
    };
    assert_eq!(
        no_path.classify(&pin()),
        PinVerdict::KernelsUnverified { metallib: None }
    );
    assert!(!no_path.classify(&pin()).is_match());
}

#[test]
fn no_two_verdicts_read_the_same() {
    // The point of spelling every failure as its own variant is that an
    // operator can tell them apart. Two variants that render identically would
    // put "could not look" and "looked and found nothing" back in one bucket,
    // which is the defect this type exists to prevent.
    let all = [
        pinned_pair().classify(&pin()),
        PinVerdict::PinUnparsable,
        PinVerdict::NotLoaded {
            library: "libmlx.dylib",
        },
        PinVerdict::KernelsMissing {
            metallib: PathBuf::from("/keg/lib/mlx.metallib"),
            mlx: Some("0.32.0".to_owned()),
        },
        PinVerdict::VersionMismatch {
            formula: "mlx",
            resolved: "0.32.0".to_owned(),
            pinned: "0.31.2".to_owned(),
        },
        PinVerdict::NotAKeg {
            formula: "mlx",
            resolved: Some(PathBuf::from("/usr/local/mlx/lib/libmlx.dylib")),
        },
        PinVerdict::KernelsUnverified { metallib: None },
        PinVerdict::KernelsUnverified {
            metallib: Some(PathBuf::from("/keg/lib/mlx.metallib")),
        },
    ];
    let mut seen: Vec<String> = Vec::new();
    for found in &all {
        let report = found.report();
        assert!(!report.is_empty(), "{found:?} renders nothing");
        assert!(
            !seen.contains(&report),
            "{found:?} renders the same line as another verdict: {report}"
        );
        seen.push(report);
    }
    // Only one of them may be read as a pass.
    assert_eq!(all.iter().filter(|v| v.is_match()).count(), 1);
}

/// The pin file is the only place the pair is written down.
///
/// The bench preflight and the restore script both act on these versions, and
/// each used to carry its own copy — so a pin bump silently left them
/// restoring and validating the previous pair. Both now read the pin.
///
/// Fail-closed by construction: the scripts are pulled in with `include_str!`,
/// so a renamed or deleted one is a compile error rather than a scan that
/// quietly matches nothing.
#[test]
fn no_other_file_writes_the_pinned_versions_down() {
    const PREFLIGHT: &str = include_str!("../../../scripts/mlx_preflight.sh");
    const RESTORE: &str = include_str!("../../../scripts/mlx_restore_pin.sh");

    let Some(pinned) = parse_pin(PIN_SRC) else {
        panic!("the checked-in mlx-pin.txt must parse for this gate to have needles");
    };
    for needle in [pinned.mlx.as_str(), pinned.mlx_c.as_str()] {
        // Positive control: a needle the pin file itself does not contain would
        // make every assertion below pass for the wrong reason.
        assert!(
            PIN_SRC.contains(needle),
            "{needle:?} is not in the pin file, so searching for it proves nothing"
        );
        for (name, body) in [
            ("scripts/mlx_preflight.sh", PREFLIGHT),
            ("scripts/mlx_restore_pin.sh", RESTORE),
        ] {
            assert!(
                !body.contains(needle),
                "{name} writes {needle:?} down itself; read it from mlx-pin.txt instead, \
                 or a pin bump leaves this file on the old pair"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// dyld resolution
// ---------------------------------------------------------------------------

#[test]
fn dyld_lists_both_halves_of_the_pair() {
    // This test binary links both dylibs, so dyld must be able to name both.
    // That is what makes the gate below possible on a distributed binary,
    // where the build machine's prefix is the wrong answer.
    for library in ["libmlx.dylib", "libmlxc.dylib"] {
        let Some(path) = loaded_library_path(library) else {
            panic!("{library} is linked, so dyld must list it");
        };
        assert_eq!(path.file_name().and_then(|n| n.to_str()), Some(library));
        assert!(path.is_absolute(), "dyld reports absolute paths: {path:?}");
    }
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// The gate cannot know whether it applies to a host it cannot identify.
///
/// It scopes itself to Neural-Accelerator-class hardware, because that is the
/// only hardware the pinned kernels exist for. An unidentifiable chip makes
/// that scoping a guess, and a gate that guesses its own subject is a gate
/// that can pass by accident.
#[test]
fn the_gate_can_tell_which_host_it_is_on() {
    assert!(
        apple_silicon_generation().is_some(),
        "the chip could not be identified, so the pin gate cannot tell whether it applies"
    );
}

/// The MLX this process loaded is the pair `mlx-pin.txt` declares, and its
/// metallib carries the kernels the pin exists to buy.
///
/// Keyed on dyld's answer, not on a build-time constant or a config file: the
/// `opt` symlink that both dylibs' install names point at can move after the
/// build, and cargo cannot see it move backwards.
///
/// Scoped to Neural-Accelerator-class hosts, derived from the chip rather than
/// from a list of machines. Earlier Apple Silicon legitimately ships zero of
/// these kernels at every MLX version, so the pinned pair buys nothing there
/// and demanding it would be noise on the majority of Macs.
#[test]
fn linked_mlx_matches_the_pinned_pair() {
    let found = pin_check();
    if found.enforced {
        assert!(
            found.matches,
            "this Mac has a GPU Neural Accelerator and is not running the validated MLX pair: {}",
            found.detail
        );
        return;
    }
    // Not Neural-Accelerator hardware, so the pinned kernels buy nothing here
    // and the pair is not required. The probe still has to have worked: a host
    // this gate does not bind is not a host it may decline to look at, or the
    // scoping becomes a way to pass without checking anything.
    assert!(
        !matches!(
            verdict(),
            PinVerdict::PinUnparsable | PinVerdict::NotLoaded { .. }
        ),
        "the loaded MLX could not be identified at all: {}",
        found.detail
    );
}
