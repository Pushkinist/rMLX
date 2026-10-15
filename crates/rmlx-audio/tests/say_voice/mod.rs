//! The `say` voice of the English clips of the model tests, of this crate
//! and of `rmlx-server`.
//!
//! The default voice of a host is not reproducible: it can give one of two
//! renders of the same sentence at random. A named voice gives the same
//! samples on each run.

#![allow(dead_code, unreachable_pub)]

use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The `say` voice of every English clip.
pub const ENGLISH_VOICE: &str = "Eddy (English (US))";

/// Whether `say` lists `voice` as installed. `say -v <voice>` does not tell:
/// for a voice it does not have, it speaks with the default voice and exits 0.
/// `false` also when `say` cannot run.
pub fn voice_is_installed(voice: &str) -> bool {
    let Ok(listed) = Command::new("say").args(["-v", "?"]).output() else {
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

/// The longest a `say` or `ffmpeg` child of a clip may run. The longest clip
/// of the tests, 30 s of speech, takes 0.6 s to speak and 0.05 s to convert.
const CHILD_BOUND: Duration = Duration::from_secs(30);

/// Run `command`, the tool `tool`, to its end: whether it exits with success.
/// `false` when it cannot start. A child that runs past [`CHILD_BOUND`] hangs:
/// it is killed and waited for, and the test fails. A hang is not a tool that
/// is not available, so it is not a stand-down.
#[allow(
    clippy::panic,
    reason = "a hung child fails the test that waits for it"
)]
pub fn run_bounded(tool: &str, command: &mut Command) -> bool {
    let Ok(mut child) = command.stdout(Stdio::null()).stderr(Stdio::null()).spawn() else {
        return false;
    };
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) if started.elapsed() < CHILD_BOUND => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Ok(None) => {
                let _ = child.kill();
                let _ = child.wait();
                panic!(
                    "{tool} ran past the bound of {} s and was killed",
                    CHILD_BOUND.as_secs()
                );
            }
            Err(_) => return false,
        }
    }
}

/// Speak `text` with `voice` to `path`, as a 16 kHz mono 16-bit WAV. `false`
/// when `say` or the voice is not available; nothing is spoken then.
pub fn say_wav(voice: &str, path: &Path, text: &str) -> bool {
    voice_is_installed(voice)
        && run_bounded(
            "say",
            Command::new("say")
                .args(["-v", voice, "-o"])
                .arg(path)
                .args(["--data-format=LEI16@16000", text]),
        )
}

/// [`say_wav`] with [`ENGLISH_VOICE`].
pub fn say_english_wav(path: &Path, text: &str) -> bool {
    say_wav(ENGLISH_VOICE, path, text)
}
