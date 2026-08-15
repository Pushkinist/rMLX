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
//! costs something. So the host-class gate runs ahead of every other step: on
//! a host with no Neural Accelerator neither the dyld image walk nor the
//! metallib open happens, and nothing is logged above `debug`.
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
/// Spelled here as well as in `build_support.rs`: a build script cannot import
/// from the crate it builds, so the two copies are structural rather than an
/// oversight. Both must name the same family, or the build-time and runtime
/// answers describe different things — `build_side_names_the_same_kernel_family`
/// pins them together, because nothing in the compiler does.
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
        /// The metallib that answered. Carried in the variant rather than
        /// re-derived at the warning site: the warning has to name the file it
        /// read, and a second, independently-obtained path could disagree with
        /// this one — which would silently demote a confirmed absence to a
        /// `debug!` instead of failing.
        metallib: PathBuf,
    },
}

impl NaxFinding {
    /// The metallib to name out loud, or `None` when there is nothing to say.
    ///
    /// Loud only for a confirmed absence on hardware that can use the kernels.
    /// Present is silent, a non-NA host is silent even when they are absent,
    /// and an uninspectable metallib is silent because it establishes nothing.
    ///
    /// Returning the path rather than a bool is what keeps the decision and the
    /// thing the warning talks about a single observation: both come out of the
    /// same match arm.
    pub(crate) fn warning_target(&self) -> Option<&Path> {
        match self {
            Self::Scanned {
                kernels_present: false,
                metallib,
            } => Some(metallib),
            Self::NotNaClass
            | Self::Unverified
            | Self::Scanned {
                kernels_present: true,
                ..
            } => None,
        }
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
        Ok(kernels_present) => NaxFinding::Scanned {
            kernels_present,
            metallib: path.to_path_buf(),
        },
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
    let metallib = metallib_to_scan(gpu_family);
    // `evaluate` applies the host-class gate again. That is not redundant
    // bookkeeping: it has to stay a total function of its inputs, which is what
    // makes the host-class x kernel-presence matrix testable on a machine that
    // is only ever one of those hosts.
    let finding = evaluate(gpu_family, metallib.as_deref());

    if let Some(path) = finding.warning_target() {
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

/// The metallib worth scanning on this host, or `None` when there is none.
///
/// The host-class gate sits here, ahead of the dyld image walk, so a host with
/// no Neural Accelerator skips the walk as well as the metallib open. Walking
/// every loaded image is not "nothing runs", which is what the module claims
/// pre-M5 hosts pay.
///
/// Named rather than folded into its one caller so that skip is observable:
/// the caller does I/O and emits tracing, but this is a total function of the
/// host class, and in a process that has MLX loaded a `None` here can only mean
/// the walk did not happen.
fn metallib_to_scan(gpu_family: Option<u8>) -> Option<PathBuf> {
    is_na_class(gpu_family).then(loaded_metallib_path).flatten()
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
    // loaded-image list, which is process-wide state — so the invariant that
    // has to hold is process-wide too, not a property of this thread. Other
    // threads are already running by this point (the tracing appender, and
    // tokio workers under `serve`), so assume the list can move underneath the
    // loop and check what that can do:
    //
    // - The count is a snapshot. A concurrent load only appends, so a stale
    //   count under-reports and the walk misses a newly added image — it never
    //   reads out of range. A concurrent unload would make `index` stale in the
    //   other direction, and `_dyld_get_image_name` answers an out-of-range
    //   index with null, which the null check skips. Either way this degrades
    //   to a missed image, never to a bad read.
    // - The name pointer belongs to dyld and stays valid for as long as the
    //   image is loaded; the bytes are copied into an owned `PathBuf` before
    //   this returns. Only unloading that image could free it, and nothing in
    //   this process unloads one: the workspace calls neither `dlopen` nor
    //   `dlclose`, and MLX and Metal load images without ever unloading them.
    //   So there is no window in which the copy could race a free.
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

/// Whether `hay` contains `needle`, skipping ahead on `first` — the needle's
/// first byte — rather than comparing at every offset.
///
/// That skip is the difference between this and the equivalent scan in
/// `build_support.rs`, and it is why this one can sit on a startup path: a
/// per-offset comparison over a ~150 MB metallib takes seconds.
fn has_needle(hay: &[u8], needle: &[u8], first: u8) -> bool {
    let mut from = 0_usize;
    while let Some(offset) = hay
        .get(from..)
        .and_then(|tail| tail.iter().position(|&b| b == first))
    {
        let start = from + offset;
        if hay
            .get(start..)
            .is_some_and(|tail| tail.starts_with(needle))
        {
            return true;
        }
        from = start + 1;
    }
    false
}

/// Whether `reader` yields [`NAX_GEMM_KERNEL`] anywhere, scanning in
/// `chunk`-sized reads.
///
/// Each read is searched where it lands. Only the last needle-1 bytes are
/// carried forward, and only they are ever copied — a match that begins any
/// earlier was already either found or ruled out in the read it began in. The
/// file is ~124-158 MB and the buffer this needs is 19 bytes, so nothing
/// between those two sizes should be memcpy'd on a startup path.
///
/// `chunk` is a parameter so the carry path — a match straddling two reads —
/// is testable without a 124 MB fixture.
fn contains_nax_kernel<R: Read>(mut reader: R, chunk: usize) -> std::io::Result<bool> {
    let needle = NAX_GEMM_KERNEL.as_bytes();
    let Some(&first) = needle.first() else {
        return Ok(false);
    };
    if chunk == 0 {
        // A zero-length buffer makes `read` answer `Ok(0)` without touching the
        // stream, which the loop below would take for end-of-file and report as
        // a confirmed absence — established by reading nothing. Refuse, so the
        // caller lands in `Unverified` instead.
        return Err(std::io::Error::other(
            "metallib scan needs a non-zero read size",
        ));
    }
    // The most a match starting in one read can extend into the next.
    let tail_len = needle.len().saturating_sub(1);
    let mut buf = vec![0_u8; chunk];
    let mut carry: Vec<u8> = Vec::with_capacity(tail_len);
    let mut bridge: Vec<u8> = Vec::with_capacity(tail_len.saturating_mul(2));
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
        // path.
        let Some(filled) = buf.get(..n) else {
            return Err(std::io::Error::other(
                "reader reported more bytes than the buffer holds",
            ));
        };

        // Matches that *start* in the carried tail. They run at most needle-1
        // bytes past it, so joining that much of this read is enough — the read
        // is never copied wholesale.
        if !carry.is_empty() {
            bridge.clear();
            bridge.extend_from_slice(&carry);
            if let Some(head) = filled.get(..tail_len.min(filled.len())) {
                bridge.extend_from_slice(head);
            }
            if has_needle(&bridge, needle, first) {
                return Ok(true);
            }
        }
        // Matches that start in this read are found in it directly, in place.
        if has_needle(filled, needle, first) {
            return Ok(true);
        }

        // Carry the last needle-1 bytes of the stream so far. A match beginning
        // inside them is exactly the one neither search above could have
        // settled, and it is all the next read needs.
        let keep_from = filled.len().saturating_sub(tail_len);
        if keep_from > 0 {
            // This read alone supplies the whole tail; nothing older survives.
            carry.clear();
        }
        if let Some(tail) = filled.get(keep_from..) {
            carry.extend_from_slice(tail);
        }
        let excess = carry.len().saturating_sub(tail_len);
        carry.drain(..excess);
    }
}

#[cfg(test)]
#[path = "nax_tests.rs"]
mod nax_tests;
