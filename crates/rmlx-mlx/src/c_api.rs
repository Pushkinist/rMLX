//! The mlx-c C API this crate is compiled against, and the one this process
//! loaded.
//!
//! mlx-c 0.7.0 adds `bool force_fused` in front of the stream argument of
//! `mlx_fast_scaled_dot_product_attention` and keeps the symbol name. dyld
//! links a binary built against one C API to a `libmlxc.dylib` of the other
//! with no error, and the call then reads its stream from the wrong register.
//! This happens when Homebrew moves `mlx-c` under an installed `rmlx`, and to
//! a release tarball on a Mac with the other `mlx-c`.
//!
//! [`verdict`] compares the two once per process. The SDPA wrapper refuses the
//! call on a mismatch, and the one-shot MLX init logs it as an error.

use std::ffi::{c_char, c_void, CString};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};

/// An mlx-c C API that `rmlx-mlx` compiles against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CApi {
    /// mlx-c 0.6: `mlx_fast_scaled_dot_product_attention` takes 9 arguments.
    V0_6,
    /// mlx-c 0.7: the same function takes 10, with `force_fused`.
    V0_7,
}

impl CApi {
    /// The C API of the headers `build.rs` compiled this crate against.
    pub(crate) const COMPILED: Self = if cfg!(mlxc_c_api_0_7) {
        Self::V0_7
    } else {
        Self::V0_6
    };

    /// The other C API.
    #[cfg(test)]
    pub(crate) const fn other(self) -> Self {
        match self {
            Self::V0_6 => Self::V0_7,
            Self::V0_7 => Self::V0_6,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::V0_6 => "mlx-c 0.6",
            Self::V0_7 => "mlx-c 0.7",
        }
    }
}

/// The compiled C API against the loaded one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CApiVerdict {
    /// The loaded `libmlxc.dylib` has the C API this crate was compiled for.
    Match(CApi),
    /// It has the other one. The SDPA call must not be made.
    Mismatch { compiled: CApi, loaded: CApi },
}

impl CApiVerdict {
    pub(crate) const fn classify(compiled: CApi, loaded: CApi) -> Self {
        match (compiled, loaded) {
            (CApi::V0_6, CApi::V0_6) | (CApi::V0_7, CApi::V0_7) => Self::Match(compiled),
            _ => Self::Mismatch { compiled, loaded },
        }
    }

    /// `Err` naming both C APIs and the fix on a mismatch.
    pub(crate) fn require_match(self) -> Result<()> {
        match self {
            Self::Match(_) => Ok(()),
            Self::Mismatch { compiled, loaded } => {
                Err(Error::Mlx(mismatch_message(compiled, loaded)))
            }
        }
    }
}

pub(crate) fn mismatch_message(compiled: CApi, loaded: CApi) -> String {
    format!(
        "mlx-c C API mismatch: this binary was compiled against the {} C API, but the \
         loaded libmlxc.dylib has the {} C API. mlx_fast_scaled_dot_product_attention takes \
         a different argument list in the two, so rMLX does not call it. Rebuild rMLX \
         against the loaded mlx-c (`cargo clean -p rmlx-mlx`, then build again), or load \
         the mlx-c it was built against. See docs/MLX_PAIR.md, \"Two mlx-c C APIs\".",
        compiled.name(),
        loaded.name(),
    )
}

/// The verdict for this process, probed once.
pub(crate) fn verdict() -> CApiVerdict {
    static VERDICT: OnceLock<CApiVerdict> = OnceLock::new();
    *VERDICT.get_or_init(|| CApiVerdict::classify(CApi::COMPILED, loaded()))
}

/// The C API of the loaded `libmlxc.dylib`, from whether it exports the
/// marker function of the 0.7 C API (declared once, in `build_support.rs`).
fn loaded() -> CApi {
    if process_exports(env!("RMLX_MLXC_0_7_MARKER")) {
        CApi::V0_7
    } else {
        CApi::V0_6
    }
}

/// Log the verdict. Called from the one-shot MLX init, before any op runs.
pub(crate) fn report_at_init() {
    match verdict() {
        CApiVerdict::Match(api) => tracing::debug!(c_api = api.name(), "mlx-c C API matches"),
        CApiVerdict::Mismatch { compiled, loaded } => tracing::error!(
            compiled = compiled.name(),
            loaded = loaded.name(),
            "{}",
            mismatch_message(compiled, loaded)
        ),
    }
}

// libSystem. `RTLD_DEFAULT` is `((void *) -2)` in `<dlfcn.h>`.
unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// Whether any image loaded in this process exports `symbol`.
pub(crate) fn process_exports(symbol: &str) -> bool {
    let Ok(symbol) = CString::new(symbol) else {
        return false;
    };
    let rtld_default = std::ptr::without_provenance_mut::<c_void>(usize::MAX - 1);
    // SAFETY: `dlsym` only reads the NUL-terminated name, which lives until
    // the call returns. The returned address is compared with null and never
    // dereferenced or called.
    let found = unsafe { dlsym(rtld_default, symbol.as_ptr()) };
    !found.is_null()
}

#[cfg(test)]
#[path = "c_api_tests.rs"]
mod c_api_tests;
