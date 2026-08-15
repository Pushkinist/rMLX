//! Coverage for `build.rs`'s pin parsing and version/capability resolution.
//!
//! The helpers are `include!`d from the same file the build script includes: a
//! build script cannot be imported, and this logic decides whether a known
//! 3.8x GPU-matmul cliff gets reported. Silent breakage here would turn the
//! pin into decoration, so it is tested against the real checked-in pin file.

include!("../build_support.rs");

/// The pin file as shipped. Bumping it is expected; breaking its shape is not.
const PIN_SRC: &str = include_str!("../mlx-pin.txt");

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "the checked-in pin file must parse; a None here is the failure under test"
)]
fn checked_in_pin_declares_both_formulas() {
    let pin = parse_pin(PIN_SRC).unwrap();
    assert!(
        !pin.mlx.is_empty() && !pin.mlx_c.is_empty(),
        "both halves of the pair must be declared, got {pin:?}"
    );
}

#[test]
fn parse_pin_ignores_comments_and_blank_lines() {
    let src = "# a comment\n\n  mlx    0.31.2  \n# another\nmlx-c  0.6.0_2\n";
    assert_eq!(
        parse_pin(src),
        Some(MlxPin {
            mlx: "0.31.2".to_owned(),
            mlx_c: "0.6.0_2".to_owned(),
        })
    );
}

#[test]
fn parse_pin_rejects_a_half_declared_pair() {
    // Pinning one half is the exact mistake the pair exists to prevent, so it
    // must fail the build rather than silently check only mlx.
    assert_eq!(parse_pin("mlx 0.31.2\n"), None);
    assert_eq!(parse_pin("mlx-c 0.6.0_2\n"), None);
}

#[test]
fn parse_pin_rejects_unknown_or_malformed_lines() {
    assert_eq!(parse_pin("mlx 0.31.2\nmlx-c 0.6.0_2\nmlx-rs 1.0\n"), None);
    assert_eq!(parse_pin("mlx\nmlx-c 0.6.0_2\n"), None);
    assert_eq!(parse_pin("mlx 0.31.2 extra\nmlx-c 0.6.0_2\n"), None);
}

/// The pinned pair the drift cases are stated against.
fn pin_fixture() -> MlxPin {
    MlxPin {
        mlx: "0.31.2".to_owned(),
        mlx_c: "0.6.0_2".to_owned(),
    }
}

#[test]
fn no_drift_when_both_halves_match() {
    assert!(!pin_drift("0.31.2", "0.6.0_2", &pin_fixture()));
}

#[test]
fn drift_when_either_half_differs() {
    // Either half alone is drift: the pair is the validated unit.
    assert!(pin_drift("0.32.0", "0.6.0_2", &pin_fixture()));
    assert!(pin_drift("0.31.2", "0.6.0_3", &pin_fixture()));
    assert!(pin_drift("0.32.0", "0.6.0_3", &pin_fixture()));
}

#[test]
fn unknown_never_reports_drift() {
    // "unknown" is "cannot verify", not "mismatch" — a non-keg layout or an
    // unreadable header must not warn at someone whose install is merely
    // shaped differently.
    assert!(!pin_drift("unknown", "unknown", &pin_fixture()));
    assert!(!pin_drift("unknown", "0.6.0_2", &pin_fixture()));
    assert!(!pin_drift("0.31.2", "unknown", &pin_fixture()));
    // ...but a known-bad half still drifts even when the other is unknown.
    assert!(pin_drift("unknown", "0.6.0_3", &pin_fixture()));
    assert!(pin_drift("0.32.0", "unknown", &pin_fixture()));
}

#[test]
fn keg_version_reads_the_homebrew_revision_suffix() {
    // The `_2` suffix is the load-bearing part: same upstream 0.6.0, built
    // against a different mlx.
    assert_eq!(
        keg_version_from(
            std::path::Path::new("/opt/homebrew/Cellar/mlx-c/0.6.0_2"),
            "mlx-c"
        ),
        Some("0.6.0_2".to_owned())
    );
    assert_eq!(
        keg_version_from(
            std::path::Path::new("/opt/homebrew/Cellar/mlx/0.31.2"),
            "mlx"
        ),
        Some("0.31.2".to_owned())
    );
}

#[test]
fn keg_version_declines_non_keg_layouts() {
    // A wheel or hand-built tree has no keg version; the pin must stay quiet
    // instead of reading a directory name as a version.
    assert_eq!(
        keg_version_from(std::path::Path::new("/usr/local/mlx-c"), "mlx-c"),
        None
    );
    assert_eq!(keg_version_from(std::path::Path::new("/"), "mlx"), None);
    // Right leaf, wrong formula.
    assert_eq!(
        keg_version_from(
            std::path::Path::new("/opt/homebrew/Cellar/mlx/0.31.2"),
            "mlx-c"
        ),
        None
    );
}

#[test]
fn mlx_version_comes_from_the_header_macros() {
    let header = "#pragma once\n\
                  #define MLX_VERSION_MAJOR 0\n\
                  #define MLX_VERSION_MINOR 31\n\
                  #define MLX_VERSION_PATCH 2\n";
    assert_eq!(read_mlx_version(header), "0.31.2");
}

#[test]
fn mlx_version_degrades_to_unknown() {
    // An unreadable header means "cannot verify", not "mismatch" — callers key
    // off this string to stay quiet rather than warn wrongly.
    assert_eq!(read_mlx_version(""), "unknown");
    assert_eq!(read_mlx_version("#define MLX_VERSION_MAJOR 0\n"), "unknown");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "reads from an in-memory slice, which cannot fail"
)]
fn needle_is_found_across_a_chunk_boundary() {
    let needle = NAX_GEMM_KERNEL.as_bytes();
    // Straddle a read boundary: the match exists only if the tail is carried.
    let mut haystack = vec![b'.'; 7];
    haystack.extend_from_slice(needle);
    haystack.extend_from_slice(&[b'.'; 7]);
    for chunk in [1, 3, 8, 4096] {
        assert!(
            contains_needle(haystack.as_slice(), needle, chunk).unwrap(),
            "missed a boundary-straddling match at chunk={chunk}"
        );
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "reads from an in-memory slice, which cannot fail"
)]
fn needle_absent_reports_false() {
    let needle = NAX_GEMM_KERNEL.as_bytes();
    assert!(!contains_needle(b"".as_slice(), needle, 8).unwrap());
    assert!(!contains_needle(b"steel_gemm_fused_na".as_slice(), needle, 8).unwrap());
    // A near-miss must not count: this is the 0-vs-360 distinction itself.
    assert!(!contains_needle(b"steel_gemm_fused_max".as_slice(), needle, 8).unwrap());
}

#[test]
fn chip_generation_reads_the_marketing_number() {
    assert_eq!(chip_generation("Apple M5 Max"), Some(5));
    assert_eq!(chip_generation("Apple M5"), Some(5));
    assert_eq!(chip_generation("Apple M1"), Some(1));
    assert_eq!(chip_generation("Apple M2 Pro"), Some(2));
    assert_eq!(chip_generation("Apple M4 Max"), Some(4));
    assert_eq!(chip_generation("  Apple M3 Ultra  "), Some(3));
}

#[test]
fn chip_generation_declines_non_apple_silicon_strings() {
    assert_eq!(chip_generation(""), None);
    assert_eq!(
        chip_generation("Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz"),
        None
    );
    // Right prefix, no digits after it.
    assert_eq!(chip_generation("Apple M"), None);
}

#[test]
fn is_na_class_host_is_true_only_from_m5_onward() {
    // The exact boundary this whole gate exists to draw: M1-M4 never had a
    // Neural Accelerator, M5 and later do.
    assert!(!is_na_class_host("Apple M1"));
    assert!(!is_na_class_host("Apple M2 Pro"));
    assert!(!is_na_class_host("Apple M3 Max"));
    assert!(!is_na_class_host("Apple M4"));
    assert!(is_na_class_host("Apple M5"));
    assert!(is_na_class_host("Apple M5 Max"));
    // Unidentifiable input must not be guessed as NA-class.
    assert!(!is_na_class_host(""));
    assert!(!is_na_class_host(
        "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz"
    ));
}

#[test]
fn nax_capability_str_matrix() {
    // The exact tri-state recorded in RMLX_MLX_NAX / events.mlx_nax: a
    // confirmed scan result maps straight through, an unreadable metallib
    // must not be guessed as either present or absent.
    assert_eq!(nax_capability_str(Some(true)), "present");
    assert_eq!(nax_capability_str(Some(false)), "absent");
    assert_eq!(nax_capability_str(None), "unknown");
}

#[test]
fn nax_warning_level_matrix() {
    // NA-class host, kernels confirmed missing -> loud (today's behaviour).
    assert_eq!(nax_warning_level(true, false), NaxWarningLevel::Loud);
    // Non-NA host, kernels missing -> silent: exactly the macos-14 CI case.
    assert_eq!(nax_warning_level(false, false), NaxWarningLevel::Silent);
    // Kernels present -> silent regardless of host.
    assert_eq!(nax_warning_level(true, true), NaxWarningLevel::Silent);
    assert_eq!(nax_warning_level(false, true), NaxWarningLevel::Silent);
}

#[test]
fn should_report_drift_matrix() {
    // No drift -> never report, regardless of kernel presence.
    assert!(!should_report_drift(false, false));
    assert!(!should_report_drift(true, false));
    // Drift, kernels present -> report (the original, untouched drift note).
    assert!(should_report_drift(false, true));
    // Drift, kernels confirmed missing -> stays suppressed on ANY host: this
    // is the exact regression case. Before NA-class gating existed, a
    // confirmed absence always took the loud branch and the drift branch was
    // structurally unreachable underneath it. A non-NA host (macos-14 / M1)
    // with both a confirmed absence and a version drift must still end up
    // fully silent, not fall through to a different warning.
    assert!(!should_report_drift(true, true));
}

#[test]
fn ci_m1_runner_with_missing_kernels_stays_silent() {
    // The exact scenario the bug report describes: GitHub's macos-14 runners
    // are M1, which never had a Neural Accelerator, so the metallib
    // legitimately ships no nax kernels there and that must not be alarming.
    // This cannot be run against real CI from this dev machine (it is an M5
    // and cannot reproduce the missing-kernel finding), so the M1 brand
    // string is asserted directly through the same pure decision the real
    // build uses.
    let ci_host = is_na_class_host("Apple M1");
    assert!(!ci_host);
    assert_eq!(nax_warning_level(ci_host, false), NaxWarningLevel::Silent);
}

#[test]
fn dev_m5_host_with_kernels_present_stays_silent() {
    // This machine, run for real: M5, and its Homebrew mlx 0.31.2 ships the
    // nax kernels. The one cell of the matrix actually exercised on this box.
    let dev_host = is_na_class_host("Apple M5 Max");
    assert!(dev_host);
    assert_eq!(nax_warning_level(dev_host, true), NaxWarningLevel::Silent);
}

/// The pin fixture with an mlx-c value distinct from the resolved one, so
/// tests can tell "the pin's pair" from "the pair currently in scope" apart
/// in assertions.
fn nax_message_fixture_pin() -> MlxPin {
    MlxPin {
        mlx: "0.31.2".to_owned(),
        mlx_c: "0.6.0_2".to_owned(),
    }
}

#[test]
fn loud_message_reports_the_absence_without_the_ships_them_contradiction() {
    let pin = nax_message_fixture_pin();
    let lines = nax_missing_kernel_lines(
        "/opt/homebrew/opt/mlx",
        "0.32.0",
        "0.6.0_3",
        &pin,
        NAX_GEMM_KERNEL,
        "crates/rmlx-mlx/mlx-pin.txt",
    );
    let joined = lines.join("\n");

    // States the finding plainly.
    assert!(joined.contains(&format!("ships no {NAX_GEMM_KERNEL} kernels")));

    // Must not restate the fixed self-contradiction: claiming, in the same
    // breath as reporting an absence, that the pinned pair's metallib ships
    // the very kernels just found missing.
    assert!(
        !joined.contains("whose metallib ships them"),
        "must not assert what an uninspected bottle contains: {joined}"
    );
    // Must instead hedge: the pin records what one bottle had, not a promise
    // about this or any other bottle.
    assert!(
        joined.contains("does not guarantee"),
        "must hedge that the version pin alone promises nothing about kernel presence: {joined}"
    );
}

#[test]
fn loud_message_never_hard_fails() {
    // The report is pure string-building with no I/O and no panics: whatever
    // the inputs, it always yields lines to print with `cargo:warning=`,
    // never a build abort. A public user on any layout must still build.
    let pin = nax_message_fixture_pin();
    let lines = nax_missing_kernel_lines(
        "",
        "unknown",
        "unknown",
        &pin,
        NAX_GEMM_KERNEL,
        "crates/rmlx-mlx/mlx-pin.txt",
    );
    assert!(!lines.is_empty());
}
