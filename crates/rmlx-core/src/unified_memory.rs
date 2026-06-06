// unsafe_code: sysctl(3) FFI via libc::sysctlbyname — Apple-only system call.
#![allow(unsafe_code)]

//! Unified memory capacity query via `sysctl hw.memsize`.
//!
//! Returns the total unified DRAM installed on this Apple Silicon Mac.  On
//! Apple Silicon, CPU and GPU share the same physical DRAM pool — querying
//! `hw.memsize` gives the total available to both compute units.
//!
//! # Usage
//!
//! ```rust
//! use rmlx_core::unified_memory::unified_memory_gb;
//!
//! let gb = unified_memory_gb().unwrap_or(8.0); // caller-defined fallback
//! ```
//!
//! # Platform note
//!
//! Only implemented on macOS (Apple Silicon). On other targets the function
//! is a stub that always returns `None`.  rMLX is Apple Silicon–only by
//! CLAUDE.md hard rule, so the stub path exists purely for conditional-
//! compilation correctness.

/// Query total unified memory from the macOS sysctl `hw.memsize` MIB.
///
/// Returns `Some(gb)` where `gb` is the installed DRAM in gigabytes
/// (e.g. `Some(64.0)` on a 64 GB M3 Max).  Returns `None` on any
/// `sysctlbyname` error — the caller should fall back to a conservative
/// default (8 GB is the recommended safe floor).
///
/// # Implementation
///
/// Calls `libc::sysctlbyname("hw.memsize", ...)` to read a `u64` byte
/// count directly from the xnu kernel — no Metal API, no IOKit.  This
/// is the same call used by `sysctl hw.memsize` at the shell prompt.
pub fn unified_memory_gb() -> Option<f32> {
    #[cfg(target_os = "macos")]
    {
        imp::query_hw_memsize()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

#[cfg(target_os = "macos")]
mod imp {
    /// Inner implementation: calls `sysctlbyname("hw.memsize")` and converts
    /// the returned `u64` byte count to gigabytes.
    pub(super) fn query_hw_memsize() -> Option<f32> {
        // SAFETY:
        // - `"hw.memsize\0"` is a valid NUL-terminated C string accepted by
        //   `sysctlbyname` on all macOS versions that support Apple Silicon.
        // - `oldp` points to a `u64` local on the stack; `oldlenp` is its
        //   sizeof (8 bytes).  The kernel writes exactly 8 bytes on success.
        // - `newp` is NULL and `newlen` is 0 — we are reading, not writing.
        // - Return value is checked: 0 = success, -1 = error (errno set).
        // Stable xnu ABI, unchanged across all Apple Silicon macOS releases.
        let mut value: u64 = 0;
        let mut size = size_of::<u64>();
        let name = c"hw.memsize";
        let ret = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                (&raw mut value).cast::<libc::c_void>(),
                &raw mut size,
                std::ptr::null_mut(),
                0,
            )
        };
        if ret != 0 {
            return None;
        }
        if size != size_of::<u64>() {
            return None;
        }
        if value == 0 {
            return None;
        }
        // Convert bytes → GB (SI, 1e9 bytes per GB — consistent with Apple's
        // "About This Mac" display and the auto-selector math in preset_table.rs).
        Some(value as f32 / 1e9_f32)
    }
}

#[cfg(test)]
#[path = "unified_memory_tests.rs"]
mod unified_memory_tests;
