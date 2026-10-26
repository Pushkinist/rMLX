// unsafe_code: mmap(MAP_FIXED) over a read-only shared file mapping, to drop its resident pages in place
#![allow(unsafe_code)]

//! Releasing the resident pages of a read-only shared file mapping.
//!
//! macOS keeps every touched page of a file mapping resident in the process
//! until the range is unmapped: `madvise` `MADV_DONTNEED` and `MADV_FREE` leave
//! it resident, and `msync(MS_INVALIDATE)` releases it but also evicts the file
//! from the page cache. Mapping the same file range over itself with
//! `MAP_FIXED` drops the process's pages and leaves the page cache alone, so a
//! later read faults the same bytes back in from the cache.

use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;

/// The VM page size in bytes, or `None` when the system does not report a
/// power of two.
pub fn page_size() -> Option<usize> {
    // SAFETY: `sysconf` has no preconditions.
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(size).ok().filter(|p| p.is_power_of_two())
}

/// The whole pages inside the byte range `start..end`: `start` rounded up and
/// `end` rounded down to a multiple of `page`. `None` when no whole page fits.
pub fn inner_page_range(start: usize, end: usize, page: usize) -> Option<Range<usize>> {
    let first = start.checked_next_multiple_of(page)?;
    let last = end - end.checked_rem(page)?;
    (first < last).then_some(first..last)
}

/// Maps `file` from `offset` over `addr..addr + len`, read-only and shared,
/// replacing the mapping there. The pages the process held for that range are
/// released; the next read re-faults them from the page cache.
///
/// # Safety
///
/// `addr` and `len` are multiples of the page size, and `addr..addr + len`
/// lies inside a live `PROT_READ`, `MAP_SHARED` mapping of `file` whose byte at
/// `addr` is the file's byte at `offset`, which stays mapped for the call.
pub unsafe fn remap_in_place(addr: usize, len: usize, file: &File, offset: u64) -> io::Result<()> {
    let offset = libc::off_t::try_from(offset)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "file offset exceeds off_t"))?;
    // SAFETY: by the contract the new mapping covers only pages of the old one,
    // with the same file, offset, length, protection and sharing, so every byte
    // of the range reads the same before and after and a `&[u8]` borrowing it
    // stays valid. Pages outside the range are not touched, so a concurrent
    // reader of other bytes of the mapping is unaffected.
    let placed = unsafe {
        libc::mmap(
            addr as *mut c_void,
            len,
            libc::PROT_READ,
            libc::MAP_SHARED | libc::MAP_FIXED,
            file.as_raw_fd(),
            offset,
        )
    };
    if placed == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    if placed as usize != addr {
        // SAFETY: `placed` is a mapping of `len` bytes this call created and
        // nothing borrows.
        unsafe { libc::munmap(placed, len) };
        return Err(io::Error::other(format!(
            "mmap placed the range at {placed:p}, not at {addr:#x}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "mapped_pages_tests.rs"]
mod mapped_pages_tests;
