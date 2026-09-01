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

use crate::nax::{loaded_library_path, loaded_metallib_scan, KernelScan, NAX_GEMM_KERNEL};

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
/// ignored. Both formulas are mandatory, neither may repeat, and any other
/// line is an error: the pair is the unit that was validated, so a
/// half-declared, doubled or typo'd pin would quietly stop checking the very
/// coupling it exists to enforce.
///
/// `scripts/lib/mlx_pin.sh` applies the same grammar for the bench preflight
/// and the restore script, which run before any binary exists to ask. They are
/// held together by `the_shell_pin_parser_agrees_with_the_rust_one`.
pub(crate) fn parse_pin(src: &str) -> Option<MlxPin> {
    let mut mlx = None;
    let mut mlx_c = None;
    for line in src.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let slot = match (fields.next(), fields.next(), fields.next()) {
            (Some("mlx"), Some(v), None) => (&mut mlx, v),
            (Some("mlx-c"), Some(v), None) => (&mut mlx_c, v),
            _ => return None,
        };
        let (slot, version) = slot;
        if slot.is_some() || !is_keg_version(version) {
            return None;
        }
        *slot = Some(version.to_owned());
    }
    Some(MlxPin {
        mlx: mlx?,
        mlx_c: mlx_c?,
    })
}

/// Whether a token is shaped like a Homebrew keg directory name.
///
/// An allowlist, not a sanity check. These values name directories under the
/// Cellar that `scripts/mlx_restore_pin.sh` removes, copies over and repoints
/// symlinks at, so a token such as `..` would reach far outside a keg. Keeping
/// the rule here as well as in the shell means neither parser can be the one
/// that accepts it.
fn is_keg_version(token: &str) -> bool {
    let mut chars = token.chars();
    chars.next().is_some_and(|c| c.is_ascii_alphanumeric())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-'))
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
    /// What the resolved `libmlx.dylib` turned out to be.
    pub(crate) mlx: Library,
    /// What the resolved `libmlxc.dylib` turned out to be.
    pub(crate) mlx_c: Library,
    /// What the metallib scan established.
    pub(crate) kernels: KernelScan,
}

/// One half of the pair, as dyld resolved it.
///
/// A state rather than a pair of `Option`s: "not loaded" has no path to name
/// and "not a keg" has no version, and spelling those as variants is what
/// stops a verdict from rendering an empty path where a real one was assumed.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Library {
    /// dyld listed no image with this file name.
    NotLoaded,
    /// Resolved, but not inside a Homebrew keg, so it has no version.
    NotAKeg { resolved: PathBuf },
    /// Resolved inside a keg, whose directory name is the version.
    Keg { version: String, resolved: PathBuf },
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
        resolved: PathBuf,
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
        for (half, library) in [(&self.mlx, MLX_LIB), (&self.mlx_c, MLX_C_LIB)] {
            if matches!(half, Library::NotLoaded) {
                return PinVerdict::NotLoaded { library };
            }
        }
        let metallib = match &self.kernels {
            KernelScan::Scanned {
                present: false,
                metallib,
            } => {
                return PinVerdict::KernelsMissing {
                    metallib: metallib.clone(),
                    mlx: self.mlx.version().map(ToOwned::to_owned),
                };
            }
            KernelScan::Scanned {
                present: true,
                metallib,
            } => Some(metallib),
            KernelScan::Unverified { .. } => None,
        };
        let mlx = match self.mlx.against(MLX_FORMULA, &pin.mlx) {
            Ok(version) => version,
            Err(verdict) => return verdict,
        };
        let mlx_c = match self.mlx_c.against(MLX_C_FORMULA, &pin.mlx_c) {
            Ok(version) => version,
            Err(verdict) => return verdict,
        };
        match metallib {
            Some(metallib) => PinVerdict::Match {
                // What dyld resolved, never the pinned strings. They agree only
                // because `against` returned on disagreement, and a verdict has
                // to report what it observed.
                mlx: mlx.to_owned(),
                mlx_c: mlx_c.to_owned(),
                metallib: metallib.clone(),
            },
            None => PinVerdict::KernelsUnverified {
                metallib: self.kernels.metallib().cloned(),
            },
        }
    }
}

impl Library {
    /// The keg version, when this half resolved into a keg.
    pub(crate) fn version(&self) -> Option<&str> {
        match self {
            Self::Keg { version, .. } => Some(version),
            Self::NotLoaded | Self::NotAKeg { .. } => None,
        }
    }

    /// The version this half resolved to, or the verdict that stops the check.
    ///
    /// Returning `&str` rather than `Option<&str>` is what lets the caller
    /// build a `Match` without a fallback: every way of not having a comparable
    /// version leaves through `Err`.
    fn against(&self, formula: &'static str, pinned: &str) -> Result<&str, PinVerdict> {
        match self {
            Self::Keg { version, .. } if version == pinned => Ok(version),
            Self::Keg { version, .. } => Err(PinVerdict::VersionMismatch {
                formula,
                resolved: version.clone(),
                pinned: pinned.to_owned(),
            }),
            Self::NotAKeg { resolved } => Err(PinVerdict::NotAKeg {
                formula,
                resolved: resolved.clone(),
            }),
            Self::NotLoaded => Err(PinVerdict::NotLoaded { library: formula }),
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
                resolved.display()
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
    /// Whether the pin binds this host, and whether that could be established.
    pub enforcement: PinEnforcement,
    /// One line naming what was found, and what disagreed or could not be
    /// established — including the host class, which decides whether any of it
    /// is a failure.
    pub detail: String,
}

/// Whether the pin binds this host.
///
/// Three states, not a bool. "This is an M1, the pinned kernels do not exist
/// for it" and "the chip could not be identified, so nobody knows whether they
/// would" are both *not binding*, but only the first is a clean pass — the
/// second is the gate unable to tell what it is looking at, which is how the
/// host scoping turns into a way to succeed without checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PinEnforcement {
    /// Neural-Accelerator-class hardware: the pinned kernels exist for it, so
    /// the pair is required.
    Binding,
    /// Identified as pre-Neural-Accelerator Apple Silicon, which ships zero of
    /// these kernels at every MLX version. The pin buys nothing here.
    NotApplicable {
        /// The Apple GPU family, as `rmlx_core::apple_gpu` numbers them.
        gpu_family: u8,
    },
    /// The chip could not be identified, so whether the pin binds is unknown.
    UnknownHost,
}

impl PinEnforcement {
    /// Whether a mismatch is a failure on this host.
    #[must_use]
    pub const fn is_binding(self) -> bool {
        matches!(self, Self::Binding)
    }

    /// Whether the host class itself could be established.
    #[must_use]
    pub const fn host_is_known(self) -> bool {
        !matches!(self, Self::UnknownHost)
    }

    fn describe(self) -> String {
        match self {
            Self::Binding => "this Mac has a GPU Neural Accelerator, so the pinned pair is \
                              required"
                .to_owned(),
            Self::NotApplicable { gpu_family } => format!(
                "Apple GPU family {gpu_family} has no Neural Accelerator, so the pinned \
                 kernels do not exist for it and the pin does not bind here"
            ),
            Self::UnknownHost => "the chip could not be identified, so whether the pin binds \
                                  here is unknown"
                .to_owned(),
        }
    }
}

/// Check the MLX this process loaded against the checked-in pin.
#[must_use]
pub fn pin_check() -> PinCheck {
    let found = verdict();
    let enforcement = enforcement_for(rmlx_core::apple_gpu::apple_silicon_generation());
    PinCheck {
        matches: found.is_match(),
        enforcement,
        // The host class is in the operator-facing line, not only in the
        // status: without it an inapplicable gate and a gate that could not
        // tell read identically.
        detail: format!("{} ({})", found.report(), enforcement.describe()),
    }
}

/// Map the probed Apple GPU family onto whether the pin binds.
fn enforcement_for(gpu_family: Option<u8>) -> PinEnforcement {
    match gpu_family {
        None => PinEnforcement::UnknownHost,
        Some(family) if crate::nax::is_na_class(Some(family)) => PinEnforcement::Binding,
        Some(gpu_family) => PinEnforcement::NotApplicable { gpu_family },
    }
}

/// Read the pinned pair, the loaded pair, and the loaded metallib, then judge.
fn verdict() -> PinVerdict {
    let Some(pin) = parse_pin(PIN_SRC) else {
        return PinVerdict::PinUnparsable;
    };
    observe().classify(&pin)
}

/// Read both loaded libraries, and the metallib the shared probe scanned.
fn observe() -> LinkedPair {
    LinkedPair {
        mlx: resolve_library(MLX_LIB, MLX_FORMULA),
        mlx_c: resolve_library(MLX_C_LIB, MLX_C_FORMULA),
        kernels: loaded_metallib_scan().clone(),
    }
}

/// Ask dyld for `file`, canonicalize it once, and read its keg version from
/// that same canonical path.
///
/// Canonicalizing once matters: the path dyld reports runs through the package
/// manager's `opt` symlink, which is the thing this module exists to distrust.
/// Deriving the version from one read and anything else from another would
/// straddle it.
fn resolve_library(file: &str, formula: &str) -> Library {
    let Some(reported) = loaded_library_path(file) else {
        return Library::NotLoaded;
    };
    let Ok(resolved) = std::fs::canonicalize(&reported) else {
        return Library::NotAKeg { resolved: reported };
    };
    match keg_prefix(&resolved).and_then(|prefix| keg_version_from(prefix, formula)) {
        Some(version) => Library::Keg { version, resolved },
        None => Library::NotAKeg { resolved },
    }
}

/// The keg root of a canonicalized `<keg>/lib/<dylib>` path.
///
/// Two parents up from the dylib, which is the layout every Homebrew keg has.
/// Anything shallower is not a keg and yields `None` rather than a guess.
fn keg_prefix(dylib: &Path) -> Option<&Path> {
    dylib.parent()?.parent()
}

#[cfg(test)]
#[path = "pin_tests.rs"]
mod pin_tests;
