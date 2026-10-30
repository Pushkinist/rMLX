// unsafe_code: msync(MS_INVALIDATE) over a read-only shared file mapping, and mincore to observe it
#![allow(unsafe_code)]

//! Dropping the pages of a read-only shared file mapping once its bytes are
//! no longer needed.
//!
//! On macOS `madvise` `MADV_DONTNEED` and `MADV_FREE` do not release a touched
//! page of a file mapping. Unmapping it, or mapping it again with `MAP_FIXED`,
//! releases it from the process but leaves it in the page cache, which the
//! kernel did not reclaim under a load burst: it compressed the Metal copies
//! instead. `msync(MS_INVALIDATE)` releases it from both.

use std::ffi::{c_char, c_void};
use std::fs::File;
use std::io;
use std::ops::Range;
use std::os::fd::AsRawFd;
use std::path::Path;

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

/// Drops the pages of `addr..addr + len` from the process and from the page
/// cache. The range stays mapped, and the next read faults the same file
/// bytes back in from disk.
///
/// # Safety
///
/// `addr` is page-aligned and `addr..addr + len` lies inside a live
/// `PROT_READ`, `MAP_SHARED` file mapping. Never call it on anonymous or
/// private memory: `MS_INVALIDATE` there is unmeasured and may discard data.
pub unsafe fn invalidate_file_pages(addr: usize, len: usize) -> io::Result<()> {
    // SAFETY: by the contract the range is a shared file mapping. Another writer
    // of the file can dirty its pages, but a synchronous `msync` writes back a
    // dirty page before `MS_INVALIDATE` drops it, so no data is lost. The
    // mapping stays in place with no holes, so a `&[u8]` into it stays valid and
    // reads the same file bytes.
    let rc = unsafe {
        libc::msync(
            addr as *mut c_void,
            len,
            libc::MS_SYNC | libc::MS_INVALIDATE,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// Which pages of the file at `path` are in the page cache, one entry per
/// page, read through a fresh mapping. Observes the page cache for tests only:
/// it shows what [`invalidate_file_pages`] evicts.
#[doc(hidden)]
pub fn file_pages_in_cache(path: &Path) -> io::Result<Vec<bool>> {
    let file = File::open(path)?;
    let len = usize::try_from(file.metadata()?.len()).map_err(io::Error::other)?;
    if len == 0 {
        return Ok(Vec::new());
    }
    let page = page_size().ok_or_else(|| io::Error::other("no power-of-two page size"))?;
    // SAFETY: a new read-only shared mapping of a file this call opened.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let mut vec: Vec<c_char> = vec![0; len.div_ceil(page)];
    // SAFETY: `addr..addr + len` is the mapping created above, and `vec` holds
    // one byte for each of its pages.
    let rc = unsafe { libc::mincore(addr, len, vec.as_mut_ptr()) };
    let in_cache = if rc == 0 {
        Ok(vec
            .iter()
            .map(|&b| i32::from(b) & libc::MINCORE_INCORE != 0)
            .collect())
    } else {
        Err(io::Error::last_os_error())
    };
    // SAFETY: `addr` is the mapping of `len` bytes created above; nothing
    // borrows it.
    unsafe { libc::munmap(addr, len) };
    in_cache
}

#[cfg(test)]
#[path = "mapped_pages_tests.rs"]
mod mapped_pages_tests;
