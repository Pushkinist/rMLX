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
