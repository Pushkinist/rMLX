// Pure helpers behind `build.rs`'s MLX prefix/version resolution.
//
// `include!`d by `build.rs` and by `tests/mlx_pin.rs` rather than imported: a
// build script cannot depend on the crate it builds, and this logic decides
// whether a known perf cliff gets reported, so it needs coverage that
// `cargo test` actually runs. Everything here is pure — all I/O and all
// `cargo:` directives stay in `build.rs`.
//
// Paths and traits are fully qualified: an `include!`d file cannot own imports
// without colliding with whichever file pulled it in.

/// The GEMM kernel family whose presence in `mlx.metallib` is the capability
/// the pin exists to guarantee. Missing entirely in some bottles.
///
/// Lives here rather than in `build.rs` so the build script, its tests and the
/// runtime probe do not each carry their own copy of the string. The runtime
/// probe (`src/nax.rs`) still needs its own — a build script cannot be imported
/// from the crate it builds — and `build_side_names_the_same_kernel_family`
/// pins that one to this one.
const NAX_GEMM_KERNEL: &str = "steel_gemm_fused_nax";

/// The MLX / mlx-c pair declared by `mlx-pin.txt`.
#[derive(Debug, PartialEq, Eq)]
struct MlxPin {
    mlx: String,
    mlx_c: String,
}

/// Parse the pinned pair out of `mlx-pin.txt`.
///
/// Format: one `<formula> <version>` per line, `#` comments and blank lines
/// ignored. Both formulas are mandatory and any other line is an error: the
/// pair is the unit that was validated, so a half-declared or typo'd pin would
/// quietly stop checking the very coupling it exists to enforce.
fn parse_pin(src: &str) -> Option<MlxPin> {
    let mut mlx = None;
    let mut mlx_c = None;
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next(), fields.next()) {
            (Some("mlx"), Some(v), None) => mlx = Some(v.to_owned()),
            (Some("mlx-c"), Some(v), None) => mlx_c = Some(v.to_owned()),
            _ => return None,
        }
    }
    Some(MlxPin {
        mlx: mlx?,
        mlx_c: mlx_c?,
    })
}

/// The Homebrew keg version an already-canonicalized prefix points at, e.g.
/// `/opt/homebrew/Cellar/mlx-c/0.6.0_2` -> `0.6.0_2`.
///
/// This is the only version identity mlx-c has: it ships no version header, and
/// the part that matters is precisely the Homebrew revision suffix (`_2` vs
/// `_3`) — same upstream 0.6.0, different build, different mlx ABI. A layout
/// that is not a keg (a wheel, a hand-built tree) yields `None`: the pin has
/// nothing to say there and must stay quiet rather than guess.
fn keg_version_from(real: &std::path::Path, formula: &str) -> Option<String> {
    let version = real.file_name()?.to_str()?;
    let parent = real.parent()?.file_name()?.to_str()?;
    (parent == formula).then(|| version.to_owned())
}

/// Whether the resolved pair differs from the pinned one.
///
/// `"unknown"` means the version could not be established (a non-keg layout, an
/// unreadable header) — that is "cannot verify", not "mismatch", so it never
/// reports drift. Guessing there would warn at people whose install is simply
/// shaped differently.
fn pin_drift(mlx: &str, mlx_c: &str, pin: &MlxPin) -> bool {
    (mlx != "unknown" && mlx != pin.mlx) || (mlx_c != "unknown" && mlx_c != pin.mlx_c)
}

/// Parse `MLX_VERSION_{MAJOR,MINOR,PATCH}` out of the text of MLX's `version.h`.
///
/// The header is authoritative for the tree we compile against — a Cellar
/// directory name is only a Homebrew convention, and MLX is also installed
/// other ways. Returns `"unknown"` when the header cannot be parsed; callers
/// treat that as "cannot verify" and stay quiet rather than crying wolf.
fn read_mlx_version(src: &str) -> String {
    let field = |name: &str| -> Option<String> {
        src.lines()
            .find_map(|l| l.trim().strip_prefix(&format!("#define {name} ")))
            .map(|v| v.trim().to_owned())
    };
    match (
        field("MLX_VERSION_MAJOR"),
        field("MLX_VERSION_MINOR"),
        field("MLX_VERSION_PATCH"),
    ) {
        (Some(a), Some(b), Some(c)) => format!("{a}.{b}.{c}"),
        _ => "unknown".to_owned(),
    }
}

/// Parse the Apple chip generation number out of a `machdep.cpu.brand_string`
/// value, e.g. `"Apple M5 Max"` -> `Some(5)`.
///
/// `None` covers anything that is not the `"Apple M<n>[ variant]"` shape
/// Apple Silicon brand strings have used since M1 — an Intel brand string, an
/// empty one, or a malformed reading. `is_na_class_host` treats that the same
/// as "not NA-class": a probe that cannot identify the chip has nothing to
/// assert and must stay quiet rather than guess.
fn chip_generation(brand_string: &str) -> Option<u32> {
    let after_m = brand_string.trim().strip_prefix("Apple M")?;
    let digits: String = after_m.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

/// Whether this host is Neural-Accelerator-class hardware — the GPU matmul
/// path the `steel_gemm_fused_nax*` kernels exist for.
///
/// NA arrived with the M5 generation (macOS; A19 Pro on iOS, out of scope
/// here since this crate targets macOS only). Do not confuse this with the
/// Neural *Engine* (ANE): every Mac since M1 has an ANE, and it is unrelated
/// to this GEMM path — gating on it would make every Mac "NA-class" and
/// bring back the exact false positive this check exists to remove.
fn is_na_class_host(brand_string: &str) -> bool {
    chip_generation(brand_string).is_some_and(|gen| gen >= 5)
}

/// The two outcomes for the missing-nax-kernel report: shout only where the
/// absence actually costs the host something.
#[derive(Debug, PartialEq, Eq)]
enum NaxWarningLevel {
    /// Missing kernels cost this host nothing (no Neural Accelerator to feed,
    /// or they are simply present) — say nothing about it.
    Silent,
    /// A Neural-Accelerator-class host is missing the kernels it needs.
    Loud,
}

/// Decide whether the missing-nax-kernel report should be loud, given
/// whether this host would benefit from the kernels and whether the metallib
/// scan found them.
///
/// Pure so the full host x kernel-presence matrix is unit-testable without a
/// metallib to scan or a real Mac model to probe: kernels present is silent
/// regardless of host, and a non-NA (or unidentifiable) host is silent even
/// when kernels are absent — only "NA-class host, confirmed absent" is loud.
/// This is precisely the case GitHub's `macos-14` (M1) runners hit: kernels
/// absent, but not NA-class, so silent.
fn nax_warning_level(na_class_host: bool, kernels_present: bool) -> NaxWarningLevel {
    if na_class_host && !kernels_present {
        NaxWarningLevel::Loud
    } else {
        NaxWarningLevel::Silent
    }
}

/// Whether the separate "MLX pin drift" note should print, alongside — not
/// instead of — the nax-missing-kernel report above.
///
/// A confirmed kernel absence takes over the report entirely, on any host:
/// this mirrors the pre-existing `if kernels_missing { <nax report> } else if
/// drift { <drift note> }` exclusivity exactly. Without this, gating the
/// drift note on `NaxWarningLevel` alone would let it fire on a non-NA host
/// whose kernels are missing — a state that was structurally unreachable
/// before this file started distinguishing NA-class hosts, and a regression
/// on the very audience (`macos-14` / M1 CI) this distinguishing exists for:
/// that build would go from "loud" to "a different warning" instead of
/// silent. `drift` alone is also stale prose once kernels are confirmed
/// missing — its text asserts the kernels "are present".
fn should_report_drift(kernels_missing: bool, drift: bool) -> bool {
    drift && !kernels_missing
}

/// Build the loud missing-nax-kernel report lines (each printed by the
/// caller as its own `cargo:warning=`).
///
/// Pure: takes everything it needs to render, so the exact text — including
/// that it never asserts a property of a bottle it did not inspect — is
/// covered by tests without a real metallib or Homebrew install. Every claim
/// here is scoped to `mlx_prefix`'s own metallib, the one the scan just
/// read; it must never assert what the pinned pair's bottle contains in the
/// abstract, since that bottle was not the one inspected.
fn nax_missing_kernel_lines(
    mlx_prefix: &str,
    mlx_version: &str,
    mlx_c_version: &str,
    pin: &MlxPin,
    kernel: &str,
    pin_file_display: &str,
) -> Vec<String> {
    vec![
        format!(
            "MLX at {mlx_prefix} (mlx {mlx_version}, mlx-c {mlx_c_version}) ships no {kernel} \
             kernels in lib/mlx.metallib."
        ),
        "  Those are the Neural-Accelerator GEMM path, and this is Neural-Accelerator-class \
         hardware: without them GPU matmul throughput measured ~3.8x lower and prefill \
         2.2-3.7x slower. Decode is bandwidth-bound and looks normal, so the symptom mimics \
         a model-code defect."
            .to_string(),
        format!(
            "  rMLX validates against mlx {} + mlx-c {}: on the bottle that pair was checked \
             against, the kernels were present, but bottle contents vary by build runner — the \
             version pin alone does not guarantee this or any other bottle carries them. Only \
             the metallib scan above does. Some Homebrew bottles omit the family entirely; a \
             non-bottle build of the same version can be fine.",
            pin.mlx, pin.mlx_c
        ),
        "  Fix (Homebrew; both kegs must already be in the Cellar — check with \
         `ls /opt/homebrew/Cellar/mlx`):"
            .to_owned(),
        format!(
            "    ln -sfn ../Cellar/mlx/{} /opt/homebrew/opt/mlx && ln -sfn \
             ../Cellar/mlx-c/{} /opt/homebrew/opt/mlx-c && brew pin mlx mlx-c && cargo \
             clean -p rmlx-mlx",
            pin.mlx, pin.mlx_c
        ),
        format!(
            "  Repoint both: mlx and mlx-c are ABI-coupled and a mismatched pair aborts at load. \
             The `cargo clean` is required, not tidiness: repointing to an older keg moves file \
             times backwards, and cargo only re-runs a build script for a *newer* one — so a \
             plain rebuild would silently keep these stale bindings. Pin: {pin_file_display}; \
             rationale and un-pin steps: docs/FFI.md."
        ),
    ]
}

/// Render the metallib nax-kernel scan as the tri-state string recorded in
/// run identity (`RMLX_MLX_NAX`, then `events.mlx_nax`).
///
/// Reuses the exact `Option<bool>` `check_mlx_pin` already computes from the
/// metallib scan — `Some(true)`/`Some(false)` are a confirmed presence or
/// absence, `None` is "no metallib to inspect" (unreadable or missing file).
/// The third case is `"unknown"`, not a guess in either direction: a build
/// that cannot look must not report a capability it never observed.
fn nax_capability_str(fast_gemm: Option<bool>) -> &'static str {
    match fast_gemm {
        Some(true) => "present",
        Some(false) => "absent",
        None => "unknown",
    }
}

/// Whether `reader` yields `needle` anywhere, scanning in `chunk`-sized reads.
///
/// The metallib this scans is ~150 MB, which a build script has no business
/// holding in memory. `chunk` is a parameter so the carry path — a needle that
/// straddles two reads — is testable without a 150 MB fixture.
fn contains_needle<R: std::io::Read>(
    mut reader: R,
    needle: &[u8],
    chunk: usize,
) -> std::io::Result<bool> {
    let mut buf = vec![0_u8; chunk];
    let mut window: Vec<u8> = Vec::new();
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => return Ok(false),
            Ok(n) => n,
            // A signal must not be mistaken for a failed probe: the caller reads
            // an error as "cannot inspect" and goes quiet, which is precisely
            // the outcome this scan exists to prevent.
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        };
        // `Read` guarantees n <= buf.len(); `get` states that without a panic
        // path, and keeps the copy a memcpy rather than a byte-at-a-time loop.
        let Some(filled) = buf.get(..n) else {
            return Err(std::io::Error::other(
                "reader reported more bytes than the buffer holds",
            ));
        };
        window.extend_from_slice(filled);
        if window.windows(needle.len()).any(|w| w == needle) {
            return Ok(true);
        }
        // Keep the last needle-1 bytes: a match straddling this read and the
        // next one lives entirely inside that tail plus what comes after.
        let consumed = window.len().saturating_sub(needle.len().saturating_sub(1));
        window.drain(..consumed);
    }
}
