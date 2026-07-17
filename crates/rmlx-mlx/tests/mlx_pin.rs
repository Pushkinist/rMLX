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
    let needle = b"steel_gemm_fused_nax";
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
    let needle = b"steel_gemm_fused_nax";
    assert!(!contains_needle(b"".as_slice(), needle, 8).unwrap());
    assert!(!contains_needle(b"steel_gemm_fused_na".as_slice(), needle, 8).unwrap());
    // A near-miss must not count: this is the 0-vs-360 distinction itself.
    assert!(!contains_needle(b"steel_gemm_fused_max".as_slice(), needle, 8).unwrap());
}
