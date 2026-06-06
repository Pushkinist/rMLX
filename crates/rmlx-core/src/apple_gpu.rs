// unsafe_code: sysctl(3) FFI via libc::sysctlbyname — Apple-only system call.
#![allow(unsafe_code)]

//! Apple Silicon GPU-family detection.
//!
//! Returns the Apple GPU family number (7, 8, 9, 10, …) the host machine
//! belongs to, derived from the CPU brand string the xnu kernel exposes
//! through `machdep.cpu.brand_string`.
//!
//! # Mapping
//!
//! Per Apple's [Metal Feature Set
//! Tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) and
//! the public `MTLGPUFamily` enum:
//!
//! | CPU      | GPU family |
//! |----------|------------|
//! | M1 / M1 Pro / M1 Max / M1 Ultra | Apple7  |
//! | M2 / M2 Pro / M2 Max / M2 Ultra | Apple8  |
//! | M3 / M3 Pro / M3 Max / M3 Ultra | Apple9  |
//! | M4 / M4 Pro / M4 Max            | Apple9  |
//! | M5 / M5 Pro / M5 Max / M5 Ultra | Apple10 |
//!
//! M4 stays on Apple9 — Apple kept the same GPU feature set on M4. Apple10
//! arrives with M5.
//!
//! # Why sysctl, not Metal `supportsFamily`?
//!
//! The Metal API would be authoritative but requires a `MTLDevice` handle
//! and Objective-C interop. The sysctl path is:
//! - Pure user-space, no extra Metal binding surface to maintain.
//! - Side-effect free (no Metal context allocation).
//! - Available before MLX initialises.
//! - Mirrors the existing [`crate::unified_memory`] pattern.
//!
//! Unknown / unparseable brand strings (e.g. a future "Apple M6 Pro" we
//! haven't taught the mapping yet) fall back to `None`. Callers MUST treat
//! `None` as "unknown — apply the conservative default" (typically the
//! Apple10+ rule, which is the safe choice when hazards are involved).

/// Apple GPU family number for the current host.
///
/// Returns `Some(n)` where `n` is the GPU family (`7..=10` today). Returns
/// `None` when:
///   - The host is not macOS / Apple Silicon (cross-compile target);
///   - `sysctlbyname` fails;
///   - The brand string cannot be parsed (unknown CPU);
///   - The CPU is older than M1 (intel macs — out of scope per CLAUDE.md
///     hard rule 1).
///
/// Callers MUST treat `None` as "unknown" and apply the **conservative**
/// default — for example, the TurboFlash gate defaults to OFF on `None`
/// because the Apple10 hazard is the worst-case failure mode.
pub fn apple_silicon_generation() -> Option<u8> {
    #[cfg(target_os = "macos")]
    {
        let brand = imp::query_cpu_brand_string()?;
        parse_apple_generation(&brand)
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

/// Convert an `M<digit>` model number to its GPU family.
///
/// `m1 → 7`, `m2 → 8`, `m3 → 9`, `m4 → 9`, `m5 → 10`, `m6+ → 10 + (m - 5)`.
///
/// Returns `None` for `m == 0` (i.e. unparseable). Public so tests can
/// exercise the mapping without going through the sysctl path.
pub fn family_for_m_number(m: u8) -> Option<u8> {
    match m {
        0 => None,
        1 => Some(7),
        2 => Some(8),
        // M3 and M4 share the same GPU family (Apple9) — Apple kept the
        // GPU feature set static across the M3 → M4 generation bump.
        3 | 4 => Some(9),
        5 => Some(10),
        // Future-proofing: assume one family bump per generation past M5.
        // Conservative: if Apple keeps a family static like they did M3→M4,
        // this overstates the family number, but the gate logic
        // (Apple ≥10 = hazard) still does the right thing.
        n => Some(10 + (n - 5)),
    }
}

/// Parse the CPU brand string into a GPU family number.
///
/// Recognises the "Apple M<digit>" prefix (case-insensitive). Anything else
/// (Intel CPUs, future Apple naming changes) returns `None`.
///
/// Public for unit tests; callers should prefer
/// [`apple_silicon_generation()`] which combines the sysctl query.
pub fn parse_apple_generation(brand: &str) -> Option<u8> {
    let s = brand.trim();
    // Strip trailing NUL bytes that sysctl includes in the buffer.
    let s = s.trim_end_matches('\0').trim();
    // Match "Apple M<n>" prefix (case-insensitive on "apple": Apple / apple / APPLE).
    let rest = s
        .strip_prefix("Apple ")
        .or_else(|| s.strip_prefix("apple "))
        .or_else(|| s.strip_prefix("APPLE "))?;
    let rest = rest.strip_prefix('M').or_else(|| rest.strip_prefix('m'))?;
    // Take the leading decimal digits.
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    let m: u8 = digits.parse().ok()?;
    family_for_m_number(m)
}

#[cfg(target_os = "macos")]
mod imp {
    /// Read `machdep.cpu.brand_string` via `sysctlbyname`.
    ///
    /// Returns the brand string (e.g. `"Apple M3 Max"`) or `None` on any
    /// sysctl error. Buffer is 128 bytes — Apple Silicon brand strings are
    /// short (`"Apple M3 Max"` = 12 bytes), so this is comfortably oversized.
    pub(super) fn query_cpu_brand_string() -> Option<String> {
        // SAFETY contract:
        // - `"machdep.cpu.brand_string\0"` is a valid C string; the MIB has
        //   been stable on every macOS Apple Silicon release.
        // - First sysctlbyname call with NULL oldp + populated oldlenp asks
        //   the kernel for the required buffer size — standard probe.
        // - Second call fills the buffer; we cap `size` so even a malicious
        //   kernel cannot overflow `buf`.
        // - We treat any non-zero return as failure and bail.
        const MAX: usize = 128;
        let name = c"machdep.cpu.brand_string";
        let mut size: usize = 0;
        let probe = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                std::ptr::null_mut(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if probe != 0 {
            return None;
        }
        if size == 0 || size > MAX {
            return None;
        }
        let mut buf = vec![0u8; size];
        let ret = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                buf.as_mut_ptr().cast::<libc::c_void>(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret != 0 {
            return None;
        }
        // Strip trailing NUL the kernel inserts.
        if let Some(nul) = buf.iter().position(|b| *b == 0) {
            buf.truncate(nul);
        }
        String::from_utf8(buf).ok()
    }
}

#[cfg(test)]
#[path = "apple_gpu_tests.rs"]
mod apple_gpu_tests;
