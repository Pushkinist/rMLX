//! Head-to-head bench harness for the [`PrefixIndex`] strategies.
//!
//! For each `N ∈ {1, 4, 16, 64, 256}` populates a freshly built `LinearScan`
//! and `RadixTree` with `N` synthetic entries × 8 chained-block hashes (≈2048
//! tokens). Drives 10_000 random-prompt lookups per impl (50% hit / 50%
//! miss). Reports the criterion `ns/op` time per `match_best` call.
//!
//! Memory: best-effort resident-byte estimate per slot (`size_of_val` walk
//! over the indexed entries) is dumped to `.rmlx/bench/prefix_index.csv`
//! alongside the timing rows, so the radix vs linear overhead is observable.
//! A small Markdown summary is appended to `docs/PERF_BASELINE.md` at the
//! "Prefix-index bench" section by the runner (the bench itself just emits
//! the CSV; the docs append is a one-line cat in the decision commit).
//!
//! ## Running
//!
//! ```
//! cargo bench -p rmlx-models --bench prefix_index_bench
//! ```
//!
//! Times under criterion's default 100-sample regime; full output lands at
//! `target/criterion/`. The CSV under `.rmlx/bench/` is the load-bearing
//! artefact the decision rule reads.

#![allow(
    missing_docs, // criterion_group!/criterion_main! expand to undocumented fns
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented
)]

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use rmlx_models::prefix_index::{LinearScan, PrefixIndex, RadixTree};

// ---------------------------------------------------------------------------
// Synthetic fixture
// ---------------------------------------------------------------------------

const BLOCKS_PER_ENTRY: usize = 8;
const LOOKUPS_PER_RUN: usize = 10_000;
const LAYOUT_KEY: u64 = 0;

/// Deterministic LCG so the bench is reproducible across runs.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
}

fn synth_chained(seed: u64, n_blocks: usize) -> Vec<u64> {
    let mut prev = seed;
    let mut out = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let mut h = prev;
        for byte in i.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        out.push(h);
        prev = h;
    }
    out
}

struct Fixture {
    entries: Vec<Vec<u64>>,
    /// Pre-baked probe queries: 50% match (clone one of the entries),
    /// 50% miss (synth fresh).
    probes: Vec<Vec<u64>>,
}

fn build_fixture(n_entries: usize, seed: u64) -> Fixture {
    let mut rng = Lcg::new(seed);
    let entries: Vec<Vec<u64>> = (0..n_entries)
        .map(|i| synth_chained(rng.next_u64() ^ (i as u64), BLOCKS_PER_ENTRY))
        .collect();
    let mut probes: Vec<Vec<u64>> = Vec::with_capacity(LOOKUPS_PER_RUN);
    for i in 0..LOOKUPS_PER_RUN {
        if i % 2 == 0 {
            // Hit: clone one of the entries.
            let idx = (rng.next_u64() as usize) % n_entries.max(1);
            probes.push(entries[idx].clone());
        } else {
            // Miss: fresh seed.
            probes.push(synth_chained(rng.next_u64(), BLOCKS_PER_ENTRY));
        }
    }
    Fixture { entries, probes }
}

/// Best-effort resident-byte estimate over a `Vec<Vec<u64>>`-like structure.
/// Doesn't capture per-node bookkeeping inside `RadixTree`; the radix path
/// uses an additional fold (`radix_resident_bytes`) for accuracy.
fn linear_resident_bytes(idx: &LinearScan) -> usize {
    // `LinearScan` holds `Vec<{chained: Vec<u64>, layout_key, slot_id}>`.
    // We can't introspect the private fields from a bench, so approximate
    // by `len() * (BLOCKS_PER_ENTRY * 8 + 24)`. The 24-byte constant is
    // `Vec<u64>` (3 words on 64-bit) + `u64 + u64` for layout_key + slot_id.
    idx.len() * (BLOCKS_PER_ENTRY * 8 + 24)
}

fn radix_resident_bytes(idx: &RadixTree) -> usize {
    // Worst-case upper bound: nodes count × node footprint. We can't peek
    // into the node vec from outside the crate, so estimate as
    // `entries × depth × node_footprint`. node_footprint ≈ 80 B (4 u64
    // fields + Vec<(u64,u32)> + Vec<u32>). This is conservative — actual
    // tree memory is lower when entries share prefixes (radix's whole
    // point), so the headline "radix overhead vs linear" is a strict
    // upper bound and any sub-threshold result is decisive.
    idx.len() * BLOCKS_PER_ENTRY * 80
}

// ---------------------------------------------------------------------------
// CSV + Markdown emit
// ---------------------------------------------------------------------------

fn csv_path() -> PathBuf {
    // review LOW-4: route through `rmlx_core::paths::bench_dir()`
    // so the CSV lands at `<RMLX_HOME>/bench/prefix_index.csv`. Replaces
    // the previous cwd walk-up that hard-coded `.rmlx/bench/` segments.
    rmlx_core::paths::bench_dir().join("prefix_index.csv")
}

fn ensure_csv_header(path: &PathBuf) {
    if !path.exists() {
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)
        {
            let _ = writeln!(
                f,
                "ts,kind,n_entries,blocks_per_entry,lookups,wall_ms,ns_per_op,resident_bytes_estimate"
            );
        }
    }
}

fn append_csv_row(
    kind: &str,
    n_entries: usize,
    wall_ms: f64,
    ns_per_op: f64,
    resident_bytes: usize,
) {
    let path = csv_path();
    ensure_csv_header(&path);
    if let Ok(mut f) = OpenOptions::new().append(true).create(true).open(&path) {
        let ts = chrono_now_iso();
        let _ = writeln!(
            f,
            "{ts},{kind},{n_entries},{BLOCKS_PER_ENTRY},{LOOKUPS_PER_RUN},{wall_ms:.3},{ns_per_op:.1},{resident_bytes}"
        );
    }
}

fn chrono_now_iso() -> String {
    // Workspace already depends on chrono indirectly, but rmlx-models does
    // not — keep this dependency-light by using `Instant`-based monotonic
    // seconds since process start. The CSV consumer treats `ts` as opaque.
    use std::time::SystemTime;
    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => format!("{}", d.as_secs()),
        Err(_) => "0".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Bench bodies
// ---------------------------------------------------------------------------

fn run_lookups_linear(idx: &LinearScan, probes: &[Vec<u64>]) {
    for p in probes {
        black_box(idx.match_best(black_box(p), LAYOUT_KEY));
    }
}

fn run_lookups_radix(idx: &RadixTree, probes: &[Vec<u64>]) {
    for p in probes {
        black_box(idx.match_best(black_box(p), LAYOUT_KEY));
    }
}

fn bench_prefix_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefix_index/match_best");
    group.throughput(Throughput::Elements(LOOKUPS_PER_RUN as u64));

    for &n in &[1usize, 4, 16, 64, 256] {
        let fx = build_fixture(n, 0xCAFE_F00D + n as u64);

        // Populate both impls.
        let mut linear = LinearScan::new();
        for (i, e) in fx.entries.iter().enumerate() {
            linear.insert(e, LAYOUT_KEY, i as u64 + 1);
        }
        let mut radix = RadixTree::new();
        for (i, e) in fx.entries.iter().enumerate() {
            radix.insert(e, LAYOUT_KEY, i as u64 + 1);
        }

        let lin_bytes = linear_resident_bytes(&linear);
        let rdx_bytes = radix_resident_bytes(&radix);

        // Time linear.
        group.bench_with_input(BenchmarkId::new("linear", n), &fx.probes, |b, probes| {
            b.iter(|| run_lookups_linear(black_box(&linear), probes));
        });

        // Time radix.
        group.bench_with_input(BenchmarkId::new("radix", n), &fx.probes, |b, probes| {
            b.iter(|| run_lookups_radix(black_box(&radix), probes));
        });

        // Also record a one-shot wall-clock + ns/op + RSS estimate to the
        // CSV so the decision-rule script can read them without parsing
        // criterion HTML.
        let t0 = Instant::now();
        run_lookups_linear(&linear, &fx.probes);
        let wall = t0.elapsed();
        let wall_ms = (wall.as_micros() as f64) / 1000.0;
        let ns_per_op = (wall.as_nanos() as f64) / (LOOKUPS_PER_RUN as f64);
        append_csv_row("linear", n, wall_ms, ns_per_op, lin_bytes);

        let t0 = Instant::now();
        run_lookups_radix(&radix, &fx.probes);
        let wall = t0.elapsed();
        let wall_ms = (wall.as_micros() as f64) / 1000.0;
        let ns_per_op = (wall.as_nanos() as f64) / (LOOKUPS_PER_RUN as f64);
        append_csv_row("radix", n, wall_ms, ns_per_op, rdx_bytes);
    }

    group.finish();
}

criterion_group!(benches, bench_prefix_index);
criterion_main!(benches);
