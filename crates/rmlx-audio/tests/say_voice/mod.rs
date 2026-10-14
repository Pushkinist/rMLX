//! The `say` voice of the English clips of the model tests, of this crate
//! and of `rmlx-server`.
//!
//! The default voice of a host is not reproducible: it can give one of two
//! renders of the same sentence at random. A named voice gives the same
//! samples on each run.

#![allow(dead_code, unreachable_pub)]

use std::path::Path;

/// The `say` voice of every English clip.
pub const ENGLISH_VOICE: &str = "Eddy (English (US))";

/// Whether `say` lists `voice` as installed. `say -v <voice>` does not tell:
/// for a voice it does not have, it speaks with the default voice and exits 0.
/// `false` also when `say` cannot run.
pub fn voice_is_installed(voice: &str) -> bool {
    let Ok(listed) = std::process::Command::new("say").args(["-v", "?"]).output() else {
        return false;
    };
    // A line is the name, then the locale: `Anna                de_DE    # ...`.
    // After a part of a name, the next word is the rest of the name.
    String::from_utf8_lossy(&listed.stdout).lines().any(|line| {
        line.strip_prefix(voice).is_some_and(|rest| {
            rest.split_whitespace()
                .next()
                .is_some_and(|locale| locale.contains('_'))
        })
    })
}

/// Speak `text` with `voice` to `path`, as a 16 kHz mono 16-bit WAV. `false`
/// when `say` or the voice is not available; nothing is spoken then.
pub fn say_wav(voice: &str, path: &Path, text: &str) -> bool {
    voice_is_installed(voice)
        && std::process::Command::new("say")
            .args(["-v", voice, "-o"])
            .arg(path)
            .args(["--data-format=LEI16@16000", text])
            .status()
            .is_ok_and(|status| status.success())
}

/// [`say_wav`] with [`ENGLISH_VOICE`].
pub fn say_english_wav(path: &Path, text: &str) -> bool {
    say_wav(ENGLISH_VOICE, path, text)
}
