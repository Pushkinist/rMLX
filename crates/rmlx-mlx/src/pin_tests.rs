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
    is_keg_version, keg_version_from, parse_pin, pin_check, verdict, Library, LinkedPair, MlxPin,
    PinVerdict, PIN_SRC,
};
use crate::nax::KernelScan;
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
        mlx: keg("mlx", "0.31.2", "libmlx.dylib"),
        mlx_c: keg("mlx-c", "0.6.0_2", "libmlxc.dylib"),
        kernels: KernelScan::Scanned {
            present: true,
            metallib: PathBuf::from("/opt/homebrew/Cellar/mlx/0.31.2/lib/mlx.metallib"),
        },
    }
}

/// A half resolved into the keg Homebrew would have laid it out in.
fn keg(formula: &str, version: &str, dylib: &str) -> Library {
    Library::Keg {
        version: version.to_owned(),
        resolved: PathBuf::from(format!(
            "/opt/homebrew/Cellar/{formula}/{version}/lib/{dylib}"
        )),
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

#[test]
fn parse_pin_rejects_a_repeated_formula() {
    // Two `mlx` lines leave "which one is the pin?" to line order. The pair is
    // the validated unit; a doubled half is not one.
    assert_eq!(parse_pin("mlx 0.31.2\nmlx 0.32.0\nmlx-c 0.6.0_2\n"), None);
    assert_eq!(
        parse_pin("mlx 0.31.2\nmlx-c 0.6.0_2\nmlx-c 0.6.0_3\n"),
        None
    );
}

#[test]
fn parse_pin_rejects_versions_that_are_not_keg_names() {
    // These strings name directories the restore script removes, copies over
    // and repoints symlinks at. `..` is the one that matters: it reaches the
    // whole Cellar. The rest close the same door from other directions.
    for hostile in ["..", ".", "../mlx", "a/b", "-rf", "$(id)", "", "\u{7f}"] {
        assert!(
            !is_keg_version(hostile),
            "{hostile:?} must not be accepted as a keg version"
        );
        let src = format!("mlx {hostile}\nmlx-c 0.6.0_2\n");
        assert_eq!(parse_pin(&src), None, "pin file accepted {hostile:?}");
    }
    // ...while the shapes Homebrew actually produces stay accepted.
    for real in ["0.31.2", "0.6.0_2", "1.2.3_1", "2026.1", "1.0.0-rc.1", "3"] {
        assert!(is_keg_version(real), "{real:?} is a real keg name");
    }
}

/// The shell parser and this one must call the same files usable.
///
/// The bench preflight and the restore script run before any binary exists to
/// ask, so they carry their own parser. If it were the more permissive of the
/// two, a pin the Rust gate refuses to check would still drive a restore.
#[test]
fn the_shell_pin_parser_agrees_with_the_rust_one() {
    let dir = std::env::temp_dir().join(format!("rmlx-pin-parity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/lib/mlx_pin.sh");

    let corpus = [
        "mlx 0.31.2\nmlx-c 0.6.0_2\n",
        "# c\n\n  mlx    0.31.2  \n# c\nmlx-c  0.6.0_2\n",
        "mlx 0.31.2\n",
        "mlx-c 0.6.0_2\n",
        "",
        "mlx 0.31.2\nmlx-c 0.6.0_2\nmlx-rs 1.0\n",
        "mlx\nmlx-c 0.6.0_2\n",
        "mlx 0.31.2 extra\nmlx-c 0.6.0_2\n",
        "mlx 0.31.2\nmlx 0.32.0\nmlx-c 0.6.0_2\n",
        "mlx ..\nmlx-c 0.6.0_2\n",
        "mlx 0.31.2\nmlx-c ../../etc\n",
        "mlx -rf\nmlx-c 0.6.0_2\n",
        "mlx 0.31.2\nmlx-c 0.6.0_2 # trailing\n",
    ];
    let mut accepted = 0_usize;
    for (index, src) in corpus.iter().enumerate() {
        let path = dir.join(format!("pin-{index}.txt"));
        std::fs::write(&path, src).expect("write fixture");
        let out = std::process::Command::new("bash")
            .arg("-c")
            .arg(format!(
                r#"source "{script}"; mlx_pin_load "{}" >/dev/null 2>&1"#,
                path.display()
            ))
            .output()
            .expect("bash must run");
        let shell_accepts = out.status.success();
        let rust = parse_pin(src);
        assert_eq!(
            shell_accepts,
            rust.is_some(),
            "parsers disagree on {src:?}: shell accepts={shell_accepts}, rust={rust:?}"
        );
        accepted += usize::from(shell_accepts);
    }
    std::fs::remove_dir_all(&dir).ok();

    // Controls: a corpus that is accepted everywhere, or rejected everywhere,
    // would agree for the wrong reason.
    assert!(
        accepted > 0,
        "no fixture was accepted; agreement is vacuous"
    );
    assert!(
        accepted < corpus.len(),
        "no fixture was rejected; agreement is vacuous"
    );
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
        mlx: keg("mlx", "0.32.0", "libmlx.dylib"),
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
        mlx: keg("mlx", "0.32.0", "libmlx.dylib"),
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
        mlx_c: keg("mlx-c", "0.6.0_3", "libmlxc.dylib"),
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
                mlx: Library::NotLoaded,
                ..pinned_pair()
            },
            "libmlx.dylib",
        ),
        (
            LinkedPair {
                mlx_c: Library::NotLoaded,
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
        mlx: Library::NotAKeg {
            resolved: PathBuf::from("/usr/local/mlx/lib/libmlx.dylib"),
        },
        ..pinned_pair()
    };
    assert_eq!(
        mlx_hand_built.classify(&pin()),
        PinVerdict::NotAKeg {
            formula: "mlx",
            resolved: PathBuf::from("/usr/local/mlx/lib/libmlx.dylib"),
        }
    );
    let mlx_c_hand_built = LinkedPair {
        mlx_c: Library::NotAKeg {
            resolved: PathBuf::from("/usr/local/mlx-c/lib/libmlxc.dylib"),
        },
        ..pinned_pair()
    };
    assert_eq!(
        mlx_c_hand_built.classify(&pin()),
        PinVerdict::NotAKeg {
            formula: "mlx-c",
            resolved: PathBuf::from("/usr/local/mlx-c/lib/libmlxc.dylib"),
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
            resolved: PathBuf::from("/usr/local/mlx/lib/libmlx.dylib"),
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

/// No script writes the pinned versions down itself.
///
/// The bench preflight and the restore script both act on these versions, and
/// each used to carry its own copy — so a pin bump silently left them
/// restoring and validating the previous pair.
///
/// Scoped to `scripts/`, and the set is read from the directory rather than
/// listed here, so a *new* script hardcoding the pair is caught too. Prose in
/// `docs/` legitimately names concrete versions when explaining which bottle
/// regressed; that is documentation, not a second declaration something acts
/// on.
#[test]
fn no_script_writes_the_pinned_versions_down() {
    let scripts_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts"));
    let Some(pinned) = parse_pin(PIN_SRC) else {
        panic!("the checked-in mlx-pin.txt must parse for this gate to have needles");
    };
    let needles = [pinned.mlx.as_str(), pinned.mlx_c.as_str()];
    for needle in needles {
        // Positive control: a needle the pin file itself does not contain would
        // make every assertion below pass for the wrong reason.
        assert!(
            PIN_SRC.contains(needle),
            "{needle:?} is not in the pin file, so searching for it proves nothing"
        );
    }

    let mut scanned = 0_usize;
    let mut offenders: Vec<String> = Vec::new();
    for entry in walk_shell_scripts(scripts_dir) {
        let Ok(body) = std::fs::read_to_string(&entry) else {
            panic!("could not read {}", entry.display());
        };
        scanned += 1;
        for needle in needles {
            if body.contains(needle) {
                offenders.push(format!("{} writes {needle:?}", entry.display()));
            }
        }
    }

    // An empty walk — a moved directory, a changed extension — would report
    // success having looked at nothing.
    assert!(
        scanned >= 2,
        "walked {scanned} scripts under {}; the scan found nothing to check",
        scripts_dir.display()
    );
    assert!(
        offenders.is_empty(),
        "read the pair from crates/rmlx-mlx/mlx-pin.txt instead, or a pin bump leaves \
         these on the old pair: {offenders:?}"
    );
}

/// Every `.sh` under `dir`, recursively. Panics rather than skipping on an
/// unreadable directory: a walk that cannot look must not report zero hits.
fn walk_shell_scripts(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let entries =
        std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot walk {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("readable dir entry").path();
        if path.is_dir() {
            found.extend(walk_shell_scripts(&path));
        } else if path.extension().is_some_and(|ext| ext == "sh") {
            found.push(path);
        }
    }
    found
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
    assert!(
        pin_check().enforcement.host_is_known(),
        "the verdict must carry the same answer the probe gave"
    );
}

/// Every enforcement state is reachable from a host class, and the three are
/// distinct. Asserted over injected families, so the two cells this machine
/// cannot be are covered too.
#[test]
fn host_class_maps_onto_whether_the_pin_binds() {
    use super::{enforcement_for, PinEnforcement};

    assert_eq!(enforcement_for(Some(10)), PinEnforcement::Binding);
    assert_eq!(enforcement_for(Some(11)), PinEnforcement::Binding);
    assert_eq!(
        enforcement_for(Some(9)),
        PinEnforcement::NotApplicable { gpu_family: 9 }
    );
    assert_eq!(
        enforcement_for(Some(7)),
        PinEnforcement::NotApplicable { gpu_family: 7 }
    );
    assert_eq!(enforcement_for(None), PinEnforcement::UnknownHost);

    // "Does not bind" and "cannot tell" must not be the same answer, and only
    // one of them is a host the gate may quietly pass.
    assert!(!enforcement_for(Some(7)).is_binding());
    assert!(!enforcement_for(None).is_binding());
    assert!(enforcement_for(Some(7)).host_is_known());
    assert!(!enforcement_for(None).host_is_known());

    // Each state says something different out loud.
    let described: Vec<String> = [Some(10), Some(7), None]
        .into_iter()
        .map(|family| enforcement_for(family).describe())
        .collect();
    assert_eq!(
        described
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        3,
        "an operator must be able to tell the three apart: {described:?}"
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
    if found.enforcement.is_binding() {
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
