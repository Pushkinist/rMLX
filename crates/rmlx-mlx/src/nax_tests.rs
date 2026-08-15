//! Tests for the runtime Neural-Accelerator GEMM-kernel probe.
//!
//! The load-bearing property is the **host-class gate**, and it cannot be
//! covered by running on this machine: only one class of hardware is available
//! at a time, and the case that must never regress is a pre-M5 host staying
//! silent about a metallib with no nax kernels. So the chip is stubbed as a
//! brand string through the same parser the runtime path uses, and the
//! metallib as a small file with or without the kernel name in it. Both halves
//! of the matrix are asserted in both directions: a guard never observed
//! firing, and never observed staying quiet when it should, is not known to
//! work.

use std::path::{Path, PathBuf};

use rmlx_core::apple_gpu::parse_apple_generation;

use super::{
    contains_nax_kernel, evaluate, is_na_class, loaded_libmlx_path, loaded_metallib_path,
    NaxFinding, LIBMLX_FILE, METALLIB_FILE, NAX_GEMM_KERNEL,
};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A file in the temp dir holding `body`, removed when the guard drops.
struct Fixture {
    path: PathBuf,
}

impl Fixture {
    fn new(name: &str, body: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!("rmlx-nax-{}-{name}", std::process::id()));
        std::fs::write(&path, body).expect("write nax fixture");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Stand-in for a metallib that carries the kernels, with enough filler that
/// the needle is not the only thing in the file.
fn with_nax() -> Vec<u8> {
    let mut body = vec![b'\x00'; 4096];
    body.extend_from_slice(NAX_GEMM_KERNEL.as_bytes());
    body.extend_from_slice(b"_nn_bfloat16_bfloat16_bm64");
    body.extend_from_slice(&vec![b'\x00'; 4096]);
    body
}

/// Stand-in for a metallib built with the kernel family compiled out. Carries
/// the *neighbouring* kernel names so the scan has to distinguish them rather
/// than just noticing the file is different.
fn without_nax() -> Vec<u8> {
    let mut body = vec![b'\x00'; 4096];
    body.extend_from_slice(b"steel_gemm_fused_nn_bfloat16_bfloat16_bm64");
    body.extend_from_slice(b"steel_attention_bfloat16_bq64_bk32_bd128");
    body.extend_from_slice(&vec![b'\x00'; 4096]);
    body
}

// ---------------------------------------------------------------------------
// Host class — stubbed brand strings
// ---------------------------------------------------------------------------

/// Every Apple Silicon generation before M5 lacks a Neural Accelerator, so its
/// zero-nax MLX is correct and must never be treated as NA-class. This is the
/// majority of Macs and the case a false warning would ruin.
#[test]
fn pre_m5_brand_strings_are_not_na_class() {
    for brand in [
        "Apple M1",
        "Apple M1 Pro",
        "Apple M1 Ultra",
        "Apple M2 Pro",
        "Apple M3 Max",
        "Apple M4",
        "Apple M4 Pro",
    ] {
        assert!(
            !is_na_class(parse_apple_generation(brand)),
            "{brand} has no Neural Accelerator and must not be NA-class"
        );
    }
}

/// M5 is where the Neural Accelerator arrives; later generations keep it.
#[test]
fn m5_and_later_brand_strings_are_na_class() {
    for brand in ["Apple M5", "Apple M5 Pro", "Apple M5 Max", "Apple M6 Ultra"] {
        assert!(
            is_na_class(parse_apple_generation(brand)),
            "{brand} has a Neural Accelerator and must be NA-class"
        );
    }
}

/// A chip the parser cannot identify asserts nothing, so it is not NA-class —
/// the probe stays quiet rather than guessing.
#[test]
fn unidentifiable_hosts_are_not_na_class() {
    for brand in [
        "Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz",
        "Apple",
        "Apple M",
        "",
        "   ",
    ] {
        assert!(
            !is_na_class(parse_apple_generation(brand)),
            "{brand:?} cannot be identified and must not be NA-class"
        );
    }
}

// ---------------------------------------------------------------------------
// The host-class x kernel-presence matrix
// ---------------------------------------------------------------------------

/// The one cell that speaks: Neural-Accelerator hardware running an MLX whose
/// metallib has no nax kernels.
#[test]
fn na_class_host_with_no_kernels_warns() {
    let fixture = Fixture::new("na-absent", &without_nax());
    for brand in ["Apple M5", "Apple M5 Max", "Apple M6 Ultra"] {
        let finding = evaluate(parse_apple_generation(brand), Some(fixture.path()));
        assert_eq!(
            finding,
            NaxFinding::Scanned {
                kernels_present: false
            },
            "{brand} must scan and confirm the absence"
        );
        assert!(
            finding.warrants_warning(),
            "{brand} on a nax-less metallib is the whole point of the probe: {finding:?}"
        );
    }
}

/// Same hardware, a metallib that carries the kernels: nothing to say.
#[test]
fn na_class_host_with_kernels_is_silent() {
    let fixture = Fixture::new("na-present", &with_nax());
    let finding = evaluate(parse_apple_generation("Apple M5 Max"), Some(fixture.path()));

    assert_eq!(
        finding,
        NaxFinding::Scanned {
            kernels_present: true
        },
        "the scan must confirm the kernels it was pointed at"
    );
    assert!(
        !finding.warrants_warning(),
        "a working stack must not warn: {finding:?}"
    );
}

/// The regression that would be worse than shipping nothing: the identical
/// nax-less metallib on a pre-M5 host, where it is correct, must stay silent.
///
/// M4 is in the list on purpose — it is the generation immediately below the
/// Neural Accelerator, so it is the one that flips if the host-class threshold
/// ever slips by one.
#[test]
fn pre_m5_host_with_no_kernels_is_silent() {
    let fixture = Fixture::new("pre-m5-absent", &without_nax());
    for brand in ["Apple M1", "Apple M2 Pro", "Apple M3 Max", "Apple M4 Pro"] {
        let finding = evaluate(parse_apple_generation(brand), Some(fixture.path()));
        assert!(
            !finding.warrants_warning(),
            "{brand} legitimately ships zero nax kernels and must never be warned: {finding:?}"
        );
    }
}

/// A pre-M5 host does not merely ignore the metallib — it never opens it. The
/// fixture here *does* carry the kernels, so a scan that had run would have
/// produced `Scanned`; `NotNaClass` is the proof that the host-class gate
/// short-circuits ahead of the file access, which is what makes the probe free
/// on the hardware it has nothing to say about.
#[test]
fn pre_m5_host_never_scans_the_metallib() {
    let fixture = Fixture::new("pre-m5-present", &with_nax());
    for brand in ["Apple M1", "Apple M2 Pro", "Apple M3 Max", "Apple M4 Pro"] {
        assert_eq!(
            evaluate(parse_apple_generation(brand), Some(fixture.path())),
            NaxFinding::NotNaClass,
            "{brand} must not read the metallib at all"
        );
    }
}

/// An unidentifiable chip is handled as not-NA-class end to end, not just in
/// the predicate: no scan, no warning.
#[test]
fn unidentified_host_neither_scans_nor_warns() {
    let fixture = Fixture::new("unknown-host", &without_nax());
    let finding = evaluate(
        parse_apple_generation("Intel(R) Xeon(R) W-2191B"),
        Some(fixture.path()),
    );

    assert_eq!(finding, NaxFinding::NotNaClass);
    assert!(!finding.warrants_warning(), "{finding:?}");
}

/// "Could not look" is not "absent". An NA-class host whose metallib is
/// missing or unreadable establishes nothing, so it must stay silent rather
/// than report a capability it never observed.
#[test]
fn na_class_host_with_unreadable_metallib_is_silent() {
    let missing = std::env::temp_dir().join(format!(
        "rmlx-nax-{}-does-not-exist/{METALLIB_FILE}",
        std::process::id()
    ));
    let finding = evaluate(parse_apple_generation("Apple M5 Max"), Some(&missing));

    assert_eq!(
        finding,
        NaxFinding::Unverified,
        "an unreadable metallib is unverified, not confirmed absent"
    );
    assert!(!finding.warrants_warning(), "{finding:?}");
}

/// No metallib path at all (dyld had no `libmlx.dylib` to point at) is the
/// same "cannot verify" state.
#[test]
fn na_class_host_with_no_metallib_path_is_silent() {
    let finding = evaluate(parse_apple_generation("Apple M5 Max"), None);

    assert_eq!(finding, NaxFinding::Unverified);
    assert!(!finding.warrants_warning(), "{finding:?}");
}

// ---------------------------------------------------------------------------
// The scan
// ---------------------------------------------------------------------------

#[test]
fn scan_finds_the_kernel_family_anywhere_in_the_stream() {
    let needle = NAX_GEMM_KERNEL.as_bytes();
    let filler = vec![b'q'; 300];

    let at_start = [needle, &filler].concat();
    let at_end = [&filler[..], needle].concat();
    let in_middle = [&filler[..], needle, &filler].concat();

    for (label, body) in [
        ("start", at_start),
        ("end", at_end),
        ("middle", in_middle),
        ("whole stream", needle.to_vec()),
    ] {
        assert!(
            contains_nax_kernel(body.as_slice(), 64).expect("scan"),
            "needle at {label} must be found"
        );
    }
}

#[test]
fn scan_reports_absence_rather_than_near_misses() {
    for (label, body) in [
        ("empty stream", Vec::new()),
        ("no kernels at all", vec![b'\x00'; 4096]),
        // Shares the whole prefix but not the `_nax` suffix — the classic
        // false positive if the scan stopped comparing early.
        ("prefix only", b"steel_gemm_fused_nn_bfloat16".to_vec()),
        // Many first-byte hits, no match: exercises the skip loop's advance.
        (
            "first-byte hits",
            b"ssssssteel_ssteel_gemm_fused_na".to_vec(),
        ),
        // Truncated one byte short of the needle.
        ("truncated needle", b"steel_gemm_fused_na".to_vec()),
    ] {
        assert!(
            !contains_nax_kernel(body.as_slice(), 8).expect("scan"),
            "{label} must not read as a match"
        );
    }
}

/// The carry path: a needle split across two reads still matches. Chunk sizes
/// are chosen so the split lands at every offset inside the needle.
#[test]
fn scan_matches_a_needle_straddling_two_reads() {
    let needle = NAX_GEMM_KERNEL.as_bytes();
    for split in 1..needle.len() {
        let body = [&vec![b'z'; split][..], needle].concat();
        assert!(
            contains_nax_kernel(body.as_slice(), split).expect("scan"),
            "needle split after {split} byte(s) must still be found"
        );
    }
}

/// A chunk far smaller than the needle still works — the retained tail grows
/// the window until a full needle fits.
#[test]
fn scan_works_with_a_chunk_smaller_than_the_needle() {
    let body = [&vec![b'z'; 40][..], NAX_GEMM_KERNEL.as_bytes(), b"tail"].concat();
    assert!(contains_nax_kernel(body.as_slice(), 1).expect("scan"));
}

/// The scan reads a real file through the same entry point `evaluate` uses.
#[test]
fn scan_reads_a_file_on_disk() {
    let present = Fixture::new("disk-present", &with_nax());
    let absent = Fixture::new("disk-absent", &without_nax());

    assert_eq!(
        super::metallib_has_nax_kernels(present.path()).ok(),
        Some(true)
    );
    assert_eq!(
        super::metallib_has_nax_kernels(absent.path()).ok(),
        Some(false)
    );
    assert!(
        super::metallib_has_nax_kernels(Path::new("/nonexistent/mlx.metallib")).is_err(),
        "a missing file must surface as an error, not as a confirmed absence"
    );
}

// ---------------------------------------------------------------------------
// dyld image lookup
// ---------------------------------------------------------------------------

/// This test binary links `libmlx.dylib` (`build.rs` emits the link directive
/// and asserts the dylib exists), so dyld must be able to name it. That is
/// what makes the runtime probe possible at all on a distributed binary, where
/// the build machine's prefix is the wrong answer.
#[test]
fn dyld_names_the_loaded_libmlx() {
    let path = loaded_libmlx_path().expect("libmlx.dylib is linked, so dyld must list it");

    assert_eq!(
        path.file_name().and_then(|n| n.to_str()),
        Some(LIBMLX_FILE),
        "the image found must be libmlx itself, not something merely nearby: {path:?}"
    );
    assert!(
        path.is_absolute(),
        "dyld reports absolute image paths: {path:?}"
    );
}

/// The metallib is derived as libmlx's sibling, which is where MLX colocates
/// it and therefore where MLX itself will load kernels from.
#[test]
fn metallib_path_is_the_sibling_of_the_loaded_libmlx() {
    let libmlx = loaded_libmlx_path().expect("libmlx.dylib is linked");
    let metallib = loaded_metallib_path().expect("a located libmlx always yields a metallib path");

    assert_eq!(metallib.parent(), libmlx.parent());
    assert_eq!(
        metallib.file_name().and_then(|n| n.to_str()),
        Some(METALLIB_FILE)
    );
}
