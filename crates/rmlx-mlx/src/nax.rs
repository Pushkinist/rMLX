//! Runtime check: does the MLX **actually loaded** into this process carry the
//! Neural-Accelerator GEMM kernels?
//!
//! `build.rs` scans the metallib it compiled against and bakes the answer into
//! `crate::NAX_CAPABILITY`. That constant describes the machine that *built*
//! the binary, which is the wrong machine for anything shipped: a prebuilt
//! Homebrew bottle or release tarball links `libmlx.dylib` through a
//! package-manager `opt` symlink, so it loads whatever MLX the installing user
//! happens to have — a different version, a different bottle, a different
//! capability. This module answers the question for the library dyld actually
//! resolved, which is the only answer that survives distribution.
//!
//! # Only Neural-Accelerator-class hosts hear anything
//!
//! Apple Silicon before M5 has no GPU Neural Accelerator, so its MLX
//! legitimately ships zero of these kernels at every version. That is the
//! majority of Macs. Warning there would be noise on hardware that cannot use
//! the kernels either way, and a warning most people learn to ignore is worse
//! than no warning at all — it would bury the one host where the absence
//! costs something. So the host-class gate runs first, and when it says no,
//! nothing else runs: no file is opened and nothing is logged above `debug`.
//!
//! # What the absence costs
//!
//! These are the GEMM kernels, so the loss is confined to GEMM-bound work:
//! prefill / time-to-first-token measured 2.2-3.7x slower without them.
//! Decode is bandwidth-bound, never reaches this path, and stays flat — which
//! is exactly why the loss hides. Output remains correct.

use std::io::Read;
use std::path::{Path, PathBuf};

/// The GEMM kernel family only Neural-Accelerator hardware can run.
///
/// Spelled here as well as in `build.rs`: a build script cannot import from
/// the crate it builds, so the two copies are structural rather than an
/// oversight. Both must name the same family, or the build-time and runtime
/// answers describe different things.
const NAX_GEMM_KERNEL: &str = "steel_gemm_fused_nax";

/// Apple GPU family that introduced the per-core Neural Accelerator.
///
/// [`rmlx_core::apple_gpu`] maps M1 -> 7, M2 -> 8, M3 / M4 -> 9, M5 -> 10, and
/// one family per generation after that, so `>= 10` is "M5 or later". Do not
/// confuse the Neural Accelerator with the Neural *Engine* (ANE): every Mac
/// since M1 has an ANE, it is unrelated to this GEMM path, and keying off it
/// would make every Mac NA-class.
const NA_CLASS_GPU_FAMILY: u8 = 10;

/// Read size for the metallib scan. The file is ~124-158 MB, which no startup
/// path should hold in memory at once.
const SCAN_CHUNK: usize = 1 << 20;

/// The Metal library MLX colocates with `libmlx.dylib` and loads kernels from.
const METALLIB_FILE: &str = "mlx.metallib";

/// The dylib file name dyld reports for MLX itself.
const LIBMLX_FILE: &str = "libmlx.dylib";

// dyld's read-only accessors over the loaded-image list, from libSystem. Used
// instead of a prefix baked in at build time because a baked prefix describes
// the build machine's package layout, and the point of this module is to
// describe the library that was really loaded.
unsafe extern "C" {
    fn _dyld_image_count() -> u32;
    fn _dyld_get_image_name(image_index: u32) -> *const std::ffi::c_char;
}

/// What the probe established, kept separate from what it should say about it.
///
/// Three states, not a host-class flag beside a tri-state scan result: the two
/// are not independent, because a host with no Neural Accelerator is never
/// scanned at all. Spelling that as a variant means "did not look" cannot be
/// mistaken for "looked and found nothing" — and reporting a capability that
/// was never observed is how a probe starts lying.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum NaxFinding {
    /// This host has no GPU Neural Accelerator, or its chip could not be
    /// identified. The kernels cannot matter here, so nothing was scanned.
    NotNaClass,
    /// Neural-Accelerator-class host, but the metallib could not be inspected
    /// — no path to it, or it would not read. Establishes nothing.
    Unverified,
    /// Neural-Accelerator-class host, and the metallib answered.
    Scanned {
        /// Whether the scan found the nax GEMM kernel family.
        kernels_present: bool,
    },
}

impl NaxFinding {
    /// Whether this finding is worth saying out loud.
    ///
    /// Loud only for a confirmed absence on hardware that can use the kernels.
    /// Present is silent, a non-NA host is silent even when they are absent,
    /// and an uninspectable metallib is silent because it establishes nothing.
    pub(crate) const fn warrants_warning(&self) -> bool {
        matches!(
            self,
            Self::Scanned {
                kernels_present: false
            }
        )
    }
}

/// Whether an Apple GPU family number is Neural-Accelerator-class.
///
/// `None` — an unidentifiable chip, a non-Apple-Silicon host, a failed sysctl
/// — is treated as "not NA-class": a probe that cannot identify the hardware
/// has nothing to assert and must stay quiet rather than guess.
const fn is_na_class(gpu_family: Option<u8>) -> bool {
    matches!(gpu_family, Some(family) if family >= NA_CLASS_GPU_FAMILY)
}

/// Probe the host class, then — only if it matters — the metallib.
///
/// Both inputs are injected rather than read here so the full host-class x
/// kernel-presence matrix is testable on one machine: this crate only ever
/// runs on one Mac at a time, and the case that must never regress (a pre-M5
/// host staying silent about a metallib with no nax kernels) is unreachable on
/// the hardware that would run the tests.
fn evaluate(gpu_family: Option<u8>, metallib: Option<&Path>) -> NaxFinding {
    if !is_na_class(gpu_family) {
        // Returns before touching the filesystem, not merely before deciding
        // to speak: on a host with no Neural Accelerator the answer cannot
        // change any outcome, so the scan must not cost anything either. The
        // variant is what proves it — a scan that had run would have produced
        // `Scanned`.
        return NaxFinding::NotNaClass;
    }
    let Some(path) = metallib else {
        return NaxFinding::Unverified;
    };
    match metallib_has_nax_kernels(path) {
        Ok(kernels_present) => NaxFinding::Scanned { kernels_present },
        Err(e) => {
            tracing::debug!(
                metallib = %path.display(),
                error = %e,
                "MLX nax-kernel probe could not read the metallib; capability unverified"
            );
            NaxFinding::Unverified
        }
    }
}

/// Warn once when this host has a Neural Accelerator and the MLX it loaded
/// cannot reach it.
///
/// Called from the crate's one-shot MLX init alongside the build/runtime
/// version-skew warning. The two are complementary: the skew warning catches a
/// library that changed underneath a built binary, this catches a library that
/// is the expected one but was built without the kernels.
pub(crate) fn warn_if_nax_kernels_missing() {
    let gpu_family = rmlx_core::apple_gpu::apple_silicon_generation();
    let metallib = loaded_metallib_path();
    let finding = evaluate(gpu_family, metallib.as_deref());

    // A confirmed absence always has a path to name — `warrants_warning` needs
    // a scan result, and only a readable path produces one. Binding both keeps
    // that an observation rather than an assumption with a fallback.
    if let (true, Some(path)) = (finding.warrants_warning(), metallib.as_deref()) {
        tracing::warn!(
            gpu_family = ?gpu_family,
            metallib = %path.display(),
            kernel = NAX_GEMM_KERNEL,
            "this Mac has a GPU Neural Accelerator, but the MLX it loaded ships no \
             Neural-Accelerator GEMM kernels. Prefill / time-to-first-token measured \
             2.2-3.7x slower without them. Decode is bandwidth-bound, never reaches \
             this path, and is unaffected — so output stays correct and only TTFT \
             regresses, which is why the loss reads as a model-code defect rather than \
             a toolchain one. Published rMLX prefill numbers assume these kernels are \
             present; TTFT measured here is not comparable to them. Verify with \
             `strings {} | grep -c {NAX_GEMM_KERNEL}` (want non-zero). Some package \
             bottles omit the family entirely while a build of the same MLX version \
             from another source carries it, so this is about the build, not the \
             version number.",
            path.display(),
        );
    } else {
        tracing::debug!(
            gpu_family = ?gpu_family,
            finding = ?finding,
            metallib = ?metallib,
            "MLX nax-kernel probe: nothing to report"
        );
    }
}

/// Path of the `mlx.metallib` belonging to the `libmlx.dylib` dyld loaded.
fn loaded_metallib_path() -> Option<PathBuf> {
    Some(loaded_libmlx_path()?.parent()?.join(METALLIB_FILE))
}

/// Walk dyld's loaded-image list for `libmlx.dylib`.
///
/// `None` when no such image is loaded, which on a build that links MLX means
/// the list could not be read — the caller treats that as "cannot verify".
fn loaded_libmlx_path() -> Option<PathBuf> {
    use std::os::unix::ffi::OsStrExt as _;

    // SAFETY: both functions are libSystem's read-only accessors over dyld's
    // loaded-image list. `index` is bounded by the count read immediately
    // before it, and `_dyld_get_image_name` returns either null or a
    // NUL-terminated string dyld owns for the image's lifetime; the bytes are
    // copied out before this returns. The list is mutated only by dlopen /
    // dlclose, and this runs from the crate's one-shot MLX init, which does
    // neither.
    unsafe {
        for index in 0.._dyld_image_count() {
            let raw = _dyld_get_image_name(index);
            if raw.is_null() {
                continue;
            }
            let path = Path::new(std::ffi::OsStr::from_bytes(
                std::ffi::CStr::from_ptr(raw).to_bytes(),
            ));
            if path.file_name() == Some(std::ffi::OsStr::new(LIBMLX_FILE)) {
                return Some(path.to_path_buf());
            }
        }
    }
    None
}

/// Whether `path` carries the nax GEMM kernel family.
fn metallib_has_nax_kernels(path: &Path) -> std::io::Result<bool> {
    contains_nax_kernel(std::fs::File::open(path)?, SCAN_CHUNK)
}

/// Whether `reader` yields [`NAX_GEMM_KERNEL`] anywhere, scanning in
/// `chunk`-sized reads.
///
/// Skips ahead on the needle's first byte rather than comparing at every
/// offset. That is the difference between this and the equivalent scan in
/// `build_support.rs`, and it is why this one can sit on a startup path: a
/// per-offset comparison over ~124 MB takes seconds, this takes ~47 ms for a
/// full pass and ~2 ms in the common case, where the first match arrives a
/// couple of MB in and ends the scan early.
///
/// `chunk` is a parameter so the carry path — a match straddling two reads —
/// is testable without a 124 MB fixture.
fn contains_nax_kernel<R: Read>(mut reader: R, chunk: usize) -> std::io::Result<bool> {
    let needle = NAX_GEMM_KERNEL.as_bytes();
    let Some(&first) = needle.first() else {
        return Ok(false);
    };
    let mut buf = vec![0_u8; chunk];
    let mut window: Vec<u8> = Vec::new();
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => return Ok(false),
            Ok(n) => n,
            // An interrupted read is not a failed probe: the caller reads an
            // error as "cannot inspect" and goes quiet, which is the outcome
            // this scan exists to prevent.
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        // `Read` guarantees n <= buf.len(); `get` states that without a panic
        // path and keeps the copy a memcpy.
        let Some(filled) = buf.get(..n) else {
            return Err(std::io::Error::other(
                "reader reported more bytes than the buffer holds",
            ));
        };
        window.extend_from_slice(filled);

        let mut from = 0_usize;
        while let Some(offset) = window
            .get(from..)
            .and_then(|tail| tail.iter().position(|&b| b == first))
        {
            let start = from + offset;
            if window
                .get(start..)
                .is_some_and(|tail| tail.starts_with(needle))
            {
                return Ok(true);
            }
            from = start + 1;
        }

        // Keep the last needle-1 bytes: a match straddling this read and the
        // next one lives entirely inside that tail plus what comes after.
        let consumed = window.len().saturating_sub(needle.len().saturating_sub(1));
        window.drain(..consumed);
    }
}

#[cfg(test)]
#[path = "nax_tests.rs"]
mod nax_tests;
