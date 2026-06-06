use super::imp::*;
use std::process::Command;

/// Validate that `read_proc_mem()` RSS is in the right ballpark of
/// `ps` output, plus strong sanity invariants.
///
/// Band = max(50 % of ps RSS, 64 MiB). `read_proc_mem()` and `ps`
/// sample at different instants; under parallel `cargo test
/// --workspace` the process RSS shifts by tens of MB between the two
/// reads (other test threads alloc/free), so a tight band is a
/// TOCTOU flake, not an FFI defect. The generous band still catches
/// every failure mode this test exists for: wrong units (1024×
/// off), uninitialized/garbage, wrong struct field (wildly off).
///
/// Also asserts `phys_footprint_bytes > 0` (process has physical
/// pages), `external_bytes > 0` (test binary is file-backed — the
/// executable itself is mmap'd by the kernel), and that RSS is in a
/// sane absolute range.
#[test]
fn rss_matches_ps() {
    let mem = read_proc_mem().expect("read_proc_mem() should succeed on macOS");

    // Cross-check with ps(1).
    let pid = std::process::id();
    let out = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .expect("ps command failed");

    let rss_kib: u64 = String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("ps output should be a number");

    let rss_ps = rss_kib * 1024;
    let delta = (mem.rss_bytes as i64 - rss_ps as i64).unsigned_abs();
    // Band = max(50% of ps RSS, 64 MiB) — robust to sampling skew
    // under parallel test load (see doc comment). Catches unit /
    // garbage / wrong-field errors without flaking on TOCTOU.
    let tolerance = (rss_ps / 2).max(64 * 1024 * 1024);

    assert!(
        delta <= tolerance,
        "RSS out of ballpark: read_proc_mem={} B  ps×1024={} B  delta={} B  band={} B (max(50%,64MiB))",
        mem.rss_bytes,
        rss_ps,
        delta,
        tolerance,
    );

    // Sane absolute range: a live macOS process is >1 MiB RSS and
    // (on a 128 GB box) well under 256 GiB. Guards against a zeroed
    // / wildly-wrong struct read that a relative band could miss
    // when ps RSS itself is small.
    assert!(
        mem.rss_bytes > 1024 * 1024 && mem.rss_bytes < 256 * 1024 * 1024 * 1024,
        "rss_bytes {} B outside sane absolute range (1 MiB .. 256 GiB)",
        mem.rss_bytes,
    );

    assert!(
        mem.phys_footprint_bytes > 0,
        "phys_footprint_bytes should be > 0"
    );
    assert!(
        mem.external_bytes > 0,
        "external_bytes should be > 0 (test binary is file-backed)"
    );

    // Print for run-log evidence.
    println!(
        "ProcMem {{ rss={} MB, virtual={} MB, phys_footprint={} MB, \
         internal={} MB, compressed={} MB, external={} MB }}",
        mem.rss_bytes >> 20,
        mem.virtual_bytes >> 20,
        mem.phys_footprint_bytes >> 20,
        mem.internal_bytes >> 20,
        mem.compressed_bytes >> 20,
        mem.external_bytes >> 20,
    );
    println!(
        "ps rss={} MB  mach rss={} MB  delta={} KB",
        rss_ps >> 20,
        mem.rss_bytes >> 20,
        delta / 1024,
    );
}
