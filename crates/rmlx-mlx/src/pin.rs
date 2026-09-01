//! Is the MLX **this process loaded** the pair rMLX is validated against?
//!
//! `mlx-pin.txt` names one mlx and one mlx-c build. Drifting off that pair is
//! silent: output stays correct, decode stays flat, and only GEMM-bound work —
//! prefill / time-to-first-token — collapses. A whole campaign of prefill
//! numbers can be invalidated with nothing red anywhere.
//!
//! # Why this is not a build script
//!
//! Cargo re-runs a build script only when a `rerun-if-changed` path is *newer*
//! than the last run, and it stats through symlinks. Repointing a package
//! manager's `opt` symlink at an older keg moves the observed mtime backwards,
//! so the script does not re-run — and cargo replays its cached output, which
//! means a stale "everything checks out" is printed about a stack that has
//! since changed. A check that cannot fail in the direction that matters is
//! not a check, so this one runs where it can observe the truth: in a process
//! that has the library mapped.
//!
//! # Two independent facts, one of them load-bearing
//!
//! The keg **versions** prove the pair is the validated one, which matters
//! because mlx and mlx-c are ABI-coupled. The metallib **contents** prove the
//! capability the pin exists to buy. Contents are the load-bearing half:
//! package bottles of the same version vary by build runner, so a version
//! match is not evidence the kernels are there.

use std::path::{Path, PathBuf};

use crate::nax::{
    loaded_library_path, loaded_metallib_path, metallib_has_nax_kernels, NAX_GEMM_KERNEL,
};

/// The validated pair, as checked in. Read at compile time from the same file
/// the maintainer edits, so there is no second declaration to drift.
const PIN_SRC: &str = include_str!("../mlx-pin.txt");

/// Path of the pin file, for messages that tell the reader where to look.
const PIN_FILE_DISPLAY: &str = "crates/rmlx-mlx/mlx-pin.txt";

/// The dylib file names dyld reports for the two halves of the pair, and the
/// Homebrew formula whose keg directory carries each one's version.
const MLX_LIB: &str = "libmlx.dylib";
const MLX_FORMULA: &str = "mlx";
const MLX_C_LIB: &str = "libmlxc.dylib";
const MLX_C_FORMULA: &str = "mlx-c";

/// The MLX / mlx-c pair declared by `mlx-pin.txt`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MlxPin {
    pub(crate) mlx: String,
    pub(crate) mlx_c: String,
}

/// Parse the pinned pair out of `mlx-pin.txt`.
///
/// Format: one `<formula> <version>` per line, `#` comments and blank lines
/// ignored. Both formulas are mandatory and any other line is an error: the
/// pair is the unit that was validated, so a half-declared or typo'd pin would
/// quietly stop checking the very coupling it exists to enforce.
pub(crate) fn parse_pin(src: &str) -> Option<MlxPin> {
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
/// that is not a keg (a wheel, a hand-built tree) yields `None`, which stays
/// distinguishable from a version that was read and disagreed.
pub(crate) fn keg_version_from(real: &Path, formula: &str) -> Option<String> {
    let version = real.file_name()?.to_str()?;
    let parent = real.parent()?.file_name()?.to_str()?;
    (parent == formula).then(|| version.to_owned())
}

/// Everything the probe read off the running process.
///
/// `None` is always "could not establish", never a guess in either direction —
/// which is what keeps "did not look" from being classified as "looked and
/// found nothing".
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct LinkedPair {
    /// Where dyld says `libmlx.dylib` came from, canonicalized.
    pub(crate) mlx_path: Option<PathBuf>,
    /// Where dyld says `libmlxc.dylib` came from, canonicalized.
    pub(crate) mlx_c_path: Option<PathBuf>,
    /// Keg version of the resolved `libmlx.dylib`, when it lives in a keg.
    pub(crate) mlx_keg: Option<String>,
    /// Keg version of the resolved `libmlxc.dylib`, when it lives in a keg.
    pub(crate) mlx_c_keg: Option<String>,
    /// What the metallib scan established.
    pub(crate) kernels: KernelScan,
}

/// The metallib scan, carrying the file that answered.
///
/// The path lives in the variant rather than beside it so a verdict can never
/// name a file the scan did not read — and so "no metallib, or it would not
/// open" cannot be paired with a scan result at all.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum KernelScan {
    /// Nothing to inspect, or it would not read. Establishes nothing.
    Unverified { metallib: Option<PathBuf> },
    /// The metallib answered.
    Scanned {
        /// Whether it carries [`NAX_GEMM_KERNEL`].
        present: bool,
        /// The file that answered.
        metallib: PathBuf,
    },
}

/// What the probe concluded about the loaded pair.
///
/// Every way of not being the validated pair is its own variant, so a caller
/// cannot collapse an inconclusive probe into a pass. The three inconclusive
/// ones — [`Self::NotLoaded`], [`Self::NotAKeg`], [`Self::KernelsUnverified`]
/// — each say what specifically could not be established.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PinVerdict {
    /// Both halves resolve to the pinned kegs and the metallib carries the
    /// kernels. The only passing verdict.
    Match {
        mlx: String,
        mlx_c: String,
        metallib: PathBuf,
    },
    /// `mlx-pin.txt` does not declare a pair, so there is nothing to check
    /// against.
    PinUnparsable,
    /// dyld has no image with this file name. In a binary that links MLX this
    /// means the image list could not be read, not that MLX is absent.
    NotLoaded { library: &'static str },
    /// The metallib was read and does not carry the kernels — the expensive
    /// failure, and the one a version number cannot detect.
    KernelsMissing {
        metallib: PathBuf,
        /// The resolved keg version, when the layout has one to read.
        mlx: Option<String>,
    },
    /// A resolved keg version disagrees with the pin.
    VersionMismatch {
        formula: &'static str,
        resolved: String,
        pinned: String,
    },
    /// The resolved library is not inside a keg, so it has no version to
    /// compare. Distinct from a mismatch: nothing was established.
    NotAKeg {
        formula: &'static str,
        resolved: Option<PathBuf>,
    },
    /// The metallib could not be inspected, so the capability is unknown.
    KernelsUnverified { metallib: Option<PathBuf> },
}

impl LinkedPair {
    /// Judge the observation against a pinned pair.
    ///
    /// Pure, and total over the observation: the whole matrix is exercised
    /// without a Homebrew install to point at.
    ///
    /// Precedence is by cost, not by the order the facts were read. A missing
    /// kernel family is reported ahead of a version disagreement because it is
    /// the failure that is both expensive and invisible; a version
    /// disagreement is reported ahead of the inconclusive states because it is
    /// the one that is actually known to be wrong.
    pub(crate) fn classify(&self, pin: &MlxPin) -> PinVerdict {
        for (path, library) in [(&self.mlx_path, MLX_LIB), (&self.mlx_c_path, MLX_C_LIB)] {
            if path.is_none() {
                return PinVerdict::NotLoaded { library };
            }
        }
        if let KernelScan::Scanned {
            present: false,
            metallib,
        } = &self.kernels
        {
            return PinVerdict::KernelsMissing {
                metallib: metallib.clone(),
                mlx: self.mlx_keg.clone(),
            };
        }
        for (keg, path, formula, pinned) in [
            (&self.mlx_keg, &self.mlx_path, MLX_FORMULA, &pin.mlx),
            (&self.mlx_c_keg, &self.mlx_c_path, MLX_C_FORMULA, &pin.mlx_c),
        ] {
            match keg {
                Some(resolved) if resolved != pinned => {
                    return PinVerdict::VersionMismatch {
                        formula,
                        resolved: resolved.clone(),
                        pinned: pinned.clone(),
                    };
                }
                Some(_) => {}
                // `path` is `Some` here: a `None` would have returned
                // `NotLoaded` above, before any version was looked at.
                None => {
                    return PinVerdict::NotAKeg {
                        formula,
                        resolved: path.clone(),
                    };
                }
            }
        }
        match &self.kernels {
            KernelScan::Scanned { metallib, .. } => PinVerdict::Match {
                mlx: pin.mlx.clone(),
                mlx_c: pin.mlx_c.clone(),
                metallib: metallib.clone(),
            },
            KernelScan::Unverified { metallib } => PinVerdict::KernelsUnverified {
                metallib: metallib.clone(),
            },
        }
    }
}

impl PinVerdict {
    /// Whether this process is running the validated pair.
    pub(crate) const fn is_match(&self) -> bool {
        matches!(self, Self::Match { .. })
    }

    /// One line naming what was found and, when it is not the pinned pair,
    /// what specifically could not be established or disagreed.
    pub(crate) fn report(&self) -> String {
        match self {
            Self::Match {
                mlx,
                mlx_c,
                metallib,
            } => format!(
                "loaded mlx {mlx} + mlx-c {mlx_c}; {} carries {NAX_GEMM_KERNEL}",
                metallib.display()
            ),
            Self::PinUnparsable => format!(
                "{PIN_FILE_DISPLAY} must declare one `mlx <version>` line and one \
                 `mlx-c <version>` line (plus `#` comments)"
            ),
            Self::NotLoaded { library } => format!(
                "dyld lists no {library} in this process, so the loaded MLX cannot be \
                 identified at all"
            ),
            Self::KernelsMissing { metallib, mlx } => format!(
                "{} (mlx {}) carries no {NAX_GEMM_KERNEL} kernels — GPU matmul measured \
                 ~3.8x slower and prefill 2.2-3.7x slower without them, while output and \
                 decode look normal. Repoint both halves of the pair to {PIN_FILE_DISPLAY} \
                 and see docs/FFI.md",
                metallib.display(),
                mlx.as_deref().unwrap_or("version unreadable")
            ),
            Self::VersionMismatch {
                formula,
                resolved,
                pinned,
            } => format!(
                "dyld resolved {formula} {resolved}, but {PIN_FILE_DISPLAY} pins {pinned}. \
                 mlx and mlx-c are ABI-coupled and repoint as a pair; see docs/FFI.md"
            ),
            Self::NotAKeg { formula, resolved } => format!(
                "the loaded {formula} is {}, which is not a Homebrew keg, so its version \
                 cannot be read and the pin cannot be verified here",
                resolved
                    .as_ref()
                    .map_or_else(|| "<unnamed>".to_owned(), |p| p.display().to_string())
            ),
            Self::KernelsUnverified { metallib } => match metallib {
                Some(path) => format!(
                    "could not read {} to check for {NAX_GEMM_KERNEL}, so the capability the \
                     pin exists for is unverified",
                    path.display()
                ),
                None => format!(
                    "no mlx.metallib beside the loaded libmlx.dylib, so {NAX_GEMM_KERNEL} \
                     could not be checked for"
                ),
            },
        }
    }
}

/// The pin verdict for this process, plus whether this host is bound by it.
///
/// Flat and stringly on purpose: [`PinVerdict`] stays crate-private so the
/// judgement has exactly one producer, and callers outside this crate get the
/// two bits they can act on — is it a match, and does it have to be.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct PinCheck {
    /// Whether the loaded MLX is the pair `mlx-pin.txt` declares, with a
    /// metallib that carries the kernels the pin exists to buy.
    pub matches: bool,
    /// Whether this Mac has a GPU Neural Accelerator, which is the only
    /// hardware the pinned kernels exist for and therefore the only hardware
    /// the pin binds. Earlier Apple Silicon ships zero of them at every MLX
    /// version, so requiring the pinned pair there would be noise.
    pub enforced: bool,
    /// One line naming what was found, and what disagreed or could not be
    /// established.
    pub detail: String,
}

/// Check the MLX this process loaded against the checked-in pin.
#[must_use]
pub fn pin_check() -> PinCheck {
    let found = verdict();
    PinCheck {
        matches: found.is_match(),
        enforced: crate::nax::is_na_class(rmlx_core::apple_gpu::apple_silicon_generation()),
        detail: found.report(),
    }
}

/// Read the pinned pair, the loaded pair, and the loaded metallib, then judge.
fn verdict() -> PinVerdict {
    let Some(pin) = parse_pin(PIN_SRC) else {
        return PinVerdict::PinUnparsable;
    };
    observe().classify(&pin)
}

/// Read the two loaded libraries and the metallib beside `libmlx.dylib`.
fn observe() -> LinkedPair {
    let resolve =
        |file: &str| loaded_library_path(file).and_then(|path| std::fs::canonicalize(path).ok());
    let mlx_path = resolve(MLX_LIB);
    let mlx_c_path = resolve(MLX_C_LIB);
    LinkedPair {
        mlx_keg: keg_prefix(mlx_path.as_deref())
            .and_then(|prefix| keg_version_from(prefix, MLX_FORMULA)),
        mlx_c_keg: keg_prefix(mlx_c_path.as_deref())
            .and_then(|prefix| keg_version_from(prefix, MLX_C_FORMULA)),
        kernels: scan_loaded_metallib(),
        mlx_path,
        mlx_c_path,
    }
}

/// Scan the `mlx.metallib` beside the loaded `libmlx.dylib`.
fn scan_loaded_metallib() -> KernelScan {
    let Some(metallib) = loaded_metallib_path() else {
        return KernelScan::Unverified { metallib: None };
    };
    match metallib_has_nax_kernels(&metallib) {
        Ok(present) => KernelScan::Scanned { present, metallib },
        Err(e) => {
            tracing::debug!(
                metallib = %metallib.display(),
                error = %e,
                "MLX pin probe could not read the metallib; capability unverified"
            );
            KernelScan::Unverified {
                metallib: Some(metallib),
            }
        }
    }
}

/// The keg root of a canonicalized `<keg>/lib/<dylib>` path.
///
/// Two parents up from the dylib, which is the layout every Homebrew keg has.
/// Anything shallower is not a keg and yields `None` rather than a guess.
fn keg_prefix(dylib: Option<&Path>) -> Option<&Path> {
    dylib?.parent()?.parent()
}

#[cfg(test)]
#[path = "pin_tests.rs"]
mod pin_tests;
