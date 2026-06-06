// unsafe_code: Mach VM stat FFI — mach_task_basic_info / task_vm_info kernel calls
#![allow(unsafe_code)]

//! Apple Mach VM stats — RSS and physical-footprint telemetry.
//!
//! Wraps two `task_info` kernel flavours via raw libc FFI:
//! - `MACH_TASK_BASIC_INFO` (flavour 20) — resident-set size and virtual size.
//! - `TASK_VM_INFO` (flavour 22) — `phys_footprint`, `internal`, `compressed`,
//!   and `external` byte counts.
//!
//! The public entry point is [`read_proc_mem`]. On non-macOS targets the
//! function is a no-op stub that returns `Ok(None)`.
//!
//! # Public API
//!
//! - [`read_proc_mem`] — single call returning [`ProcMem`] snapshot.
//! - [`ProcMem`] — plain-data struct with RSS, virtual, and footprint fields.
//!
//! # Invariants
//!
//! This module is a pure telemetry primitive. It does not register metrics or
//! write to any sink — callers own that wiring.

/// Process-memory telemetry via the macOS `task_info` kernel API.
///
/// Provides [`read_proc_mem`], which performs two `task_info` calls:
/// - `MACH_TASK_BASIC_INFO` (flavour 20) → RSS + virtual size.
/// - `TASK_VM_INFO` (flavour 22) → phys_footprint, internal,
///   compressed, external bytes.
///
/// # Consumer note
/// This module is a pure telemetry primitive. Wiring into the F1 metrics
/// drainer / `MetricKind` registry is deferred to **J6** (healthcheck
/// endpoint), which will own the registered `MetricKind::ProcMem` variant.
/// Do not add ad-hoc drainer calls here — unregistered metrics are WARN-dropped
/// per the metrics-DB hard rule.
#[cfg(target_os = "macos")]
mod imp {
    use libc::{
        kern_return_t, mach_msg_type_number_t, mach_task_basic_info, natural_t, task_flavor_t,
        task_info, task_info_t, KERN_SUCCESS, MACH_TASK_BASIC_INFO, MACH_TASK_BASIC_INFO_COUNT,
    };
    use std::mem;
    use std::ptr;

    use crate::error::{Error, Result};

    // -------------------------------------------------------------------------
    // task_vm_info — hand-declared because libc 0.2 does not expose it.
    // Layout verified against:
    // /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/mach/task_info.h
    //
    // C offsetof checks (confirmed by runtime C program on this machine):
    // virtual_size=0 region_count=8 page_size=12 resident_size=16
    // internal=48 external=64 compressed=120 phys_footprint=144
    // sizeof(task_vm_info)=372 _Alignof(task_vm_info)=4
    // sizeof(natural_t)=4 TASK_VM_INFO_COUNT=93
    //
    // `repr(C, packed(4))` is required: the C struct has _Alignof == 4 (not 8)
    // due to mixed integer_t / mach_vm_size_t fields. Without packed(4) Rust
    // infers align=8 and sizeof rounds to 376.
    // -------------------------------------------------------------------------

    /// Kernel flavour constant for `task_vm_info`.
    /// Defined in `<mach/task_info.h>` as `#define TASK_VM_INFO 22`.
    const TASK_VM_INFO: task_flavor_t = 22;

    /// Minimal `task_vm_info` mirror (fields through `phys_footprint` only).
    ///
    /// The rev2–rev7 extension fields are captured by `_tail` so that
    /// `sizeof == 372` and `COUNT == 93` match the kernel expectation exactly.
    ///
    /// All `mach_vm_size_t` fields → `u64`. `integer_t` → `i32`.
    #[repr(C, packed(4))]
    struct TaskVmInfo {
        virtual_size: u64,                // offset   0
        region_count: i32,                // offset   8
        page_size: i32,                   // offset  12
        resident_size: u64,               // offset  16
        resident_size_peak: u64,          // offset  24
        device: u64,                      // offset  32
        device_peak: u64,                 // offset  40
        internal: u64,                    // offset  48
        internal_peak: u64,               // offset  56
        external: u64,                    // offset  64
        external_peak: u64,               // offset  72
        reusable: u64,                    // offset  80
        reusable_peak: u64,               // offset  88
        purgeable_volatile_pmap: u64,     // offset  96
        purgeable_volatile_resident: u64, // offset 104
        purgeable_volatile_virtual: u64,  // offset 112
        compressed: u64,                  // offset 120
        compressed_peak: u64,             // offset 128
        compressed_lifetime: u64,         // offset 136
        phys_footprint: u64,              // offset 144
        // rev2–rev7 fields (min_address … ledger_tag_neural_nofootprint_peak).
        // Not read; present so sizeof == 372 and COUNT == 93.
        _tail: [u8; 372 - 152], // offset 152 → 371
    }

    /// COUNT = sizeof(task_vm_info_data_t) / sizeof(natural_t) = 372 / 4 = 93.
    const TASK_VM_INFO_COUNT: mach_msg_type_number_t =
        (size_of::<TaskVmInfo>() / size_of::<natural_t>()) as mach_msg_type_number_t;

    // Compile-time layout guards.
    const _: () = assert!(
        size_of::<TaskVmInfo>() == 372,
        "TaskVmInfo size mismatch vs task_vm_info_data_t (expected 372 bytes)"
    );
    const _: () = assert!(
        TASK_VM_INFO_COUNT == 93,
        "TASK_VM_INFO_COUNT != 93 — unexpected natural_t size?"
    );

    // -------------------------------------------------------------------------
    // Public types
    // -------------------------------------------------------------------------

    /// Snapshot of process memory counters drawn from the macOS kernel.
    ///
    /// All fields are in bytes.
    ///
    /// | Field | Source | Meaning |
    /// |---|---|---|
    /// | `rss_bytes` | `MACH_TASK_BASIC_INFO.resident_size` | Pages physically in RAM right now. |
    /// | `virtual_bytes` | `MACH_TASK_BASIC_INFO.virtual_size` | Total VM address space committed. |
    /// | `phys_footprint_bytes` | `TASK_VM_INFO.phys_footprint` | **Apple-recommended pressure metric** — what Activity Monitor and the kernel OOM killer use. |
    /// | `internal_bytes` | `TASK_VM_INFO.internal` | Anonymous / heap pages (not file-backed). |
    /// | `compressed_bytes` | `TASK_VM_INFO.compressed` | Pages held by the macOS memory compressor ("soft swap"). |
    /// | `external_bytes` | `TASK_VM_INFO.external` | File-backed pages — primarily mmap'd weight tensors in rMLX. |
    #[derive(Debug, Clone, Copy)]
    #[allow(
        clippy::exhaustive_structs,
        reason = "internal closed struct — fields mirror mach kernel task_info fields; adding a field requires a new mach API call site"
    )]
    pub struct ProcMem {
        /// Resident set size: pages physically in RAM right now
        /// (`MACH_TASK_BASIC_INFO.resident_size`).
        pub rss_bytes: u64,
        /// Total virtual address space committed
        /// (`MACH_TASK_BASIC_INFO.virtual_size`).
        pub virtual_bytes: u64,
        /// Physical memory footprint as reported by Activity Monitor and the
        /// kernel pressure subsystem (`TASK_VM_INFO.phys_footprint`).
        pub phys_footprint_bytes: u64,
        /// Anonymous / heap pages — malloc arenas, KV-cache buffers
        /// (`TASK_VM_INFO.internal`).
        pub internal_bytes: u64,
        /// Pages currently held by the macOS memory compressor — effectively
        /// "soft swap" (`TASK_VM_INFO.compressed`).
        pub compressed_bytes: u64,
        /// File-backed pages — primarily mmap'd safetensors weight files
        /// (`TASK_VM_INFO.external`).
        pub external_bytes: u64,
    }

    // -------------------------------------------------------------------------
    // Implementation
    // -------------------------------------------------------------------------

    /// Read a [`ProcMem`] snapshot for the current process.
    ///
    /// Performs two `task_info` kernel calls:
    /// 1. `MACH_TASK_BASIC_INFO` → `rss_bytes`, `virtual_bytes`.
    /// 2. `TASK_VM_INFO` → `phys_footprint_bytes`, `internal_bytes`,
    ///    `compressed_bytes`, `external_bytes`.
    ///
    /// Returns `Err(Error::Other(_))` if either call returns non-zero
    /// `kern_return_t`, with the raw code in the message.
    pub fn read_proc_mem() -> Result<ProcMem> {
        let (rss_bytes, virtual_bytes) = read_basic_info()?;
        let (phys_footprint_bytes, internal_bytes, compressed_bytes, external_bytes) =
            read_vm_info()?;
        Ok(ProcMem {
            rss_bytes,
            virtual_bytes,
            phys_footprint_bytes,
            internal_bytes,
            compressed_bytes,
            external_bytes,
        })
    }

    fn read_basic_info() -> Result<(u64, u64)> {
        let mut info = mach_task_basic_info {
            virtual_size: 0,
            resident_size: 0,
            resident_size_max: 0,
            user_time: libc::time_value_t {
                seconds: 0,
                microseconds: 0,
            },
            system_time: libc::time_value_t {
                seconds: 0,
                microseconds: 0,
            },
            policy: 0,
            suspend_count: 0,
        };
        let mut count: mach_msg_type_number_t = MACH_TASK_BASIC_INFO_COUNT;

        // SAFETY: `libc::mach_task_self_` is the current task's Mach port,
        // always valid for the process lifetime. `info` is sized for
        // MACH_TASK_BASIC_INFO (COUNT = 12 × natural_t = 48 bytes).
        // Stable Apple kernel ABI; only `info` and `count` are written.
        //
        // libc deprecated `mach_task_self` in 0.2.55 pointing to the `mach2`
        // crate, but `mach2` is not a workspace dep. Reading `mach_task_self_`
        // directly (what the deprecated wrapper does) is the correct approach.
        #[allow(deprecated)]
        let kr: kern_return_t = unsafe {
            task_info(
                libc::mach_task_self(),
                MACH_TASK_BASIC_INFO,
                &raw mut info as task_info_t,
                &raw mut count,
            )
        };

        if kr != KERN_SUCCESS {
            return Err(Error::Other(format!(
                "task_info(MACH_TASK_BASIC_INFO) failed: kern_return_t={kr}"
            )));
        }
        Ok((info.resident_size, info.virtual_size))
    }

    fn read_vm_info() -> Result<(u64, u64, u64, u64)> {
        // SAFETY: TaskVmInfo is repr(C, packed(4)) with a u8-array tail;
        // mem::zeroed() is valid — all fields are integer types.
        let mut vm: TaskVmInfo = unsafe { mem::zeroed() };
        let mut count: mach_msg_type_number_t = TASK_VM_INFO_COUNT;

        // SAFETY: `libc::mach_task_self_` is the current task port (always valid).
        // `vm` is sized for TASK_VM_INFO (COUNT = 93 × natural_t = 372 bytes),
        // matching the kernel struct verified by C offsetof/sizeof checks.
        // Only `vm` and `count` are written; no aliasing.
        #[allow(deprecated)]
        let kr: kern_return_t = unsafe {
            task_info(
                libc::mach_task_self(),
                TASK_VM_INFO,
                &raw mut vm as task_info_t,
                &raw mut count,
            )
        };

        if kr != KERN_SUCCESS {
            return Err(Error::Other(format!(
                "task_info(TASK_VM_INFO) failed: kern_return_t={kr}"
            )));
        }

        // Read fields via ptr::read_unaligned because TaskVmInfo is packed(4):
        // u64 fields are only guaranteed 4-byte aligned, not 8-byte aligned.
        // ptr::read_unaligned handles this safely on AArch64 (ARM64).
        // SAFETY: `vm` is fully initialised by the kernel call above.
        let (phys_footprint, internal, compressed, external) = unsafe {
            (
                ptr::read_unaligned(ptr::addr_of!(vm.phys_footprint)),
                ptr::read_unaligned(ptr::addr_of!(vm.internal)),
                ptr::read_unaligned(ptr::addr_of!(vm.compressed)),
                ptr::read_unaligned(ptr::addr_of!(vm.external)),
            )
        };

        Ok((phys_footprint, internal, compressed, external))
    }
}

// Re-export so callers use `rmlx_core::mach_mem::{ProcMem, read_proc_mem}`.
#[cfg(target_os = "macos")]
pub use imp::{read_proc_mem, ProcMem};

// Tests live in the sibling file `mach_mem_tests.rs` (matches project convention).
// Declared at file scope so `#[path]` resolves relative to `src/`, not `src/mach_mem/imp/`.
#[cfg(all(test, target_os = "macos"))]
#[path = "mach_mem_tests.rs"]
mod mach_mem_tests;
