//! The memory of a transcription request, from the upload to the tokenizer load.
//!
//! The route holds the uploaded file, then decodes it to 16 kHz mono. Beside
//! the request body, that may hold no more than the file once, its 16 kHz
//! output and a fixed slack: no copy of the upload and no copy of the samples.
//! The configured snapshot paths do not exist, so the request ends at the
//! tokenizer load, the first step after the decode. That also bounds what it
//! sees: a file kept alive through the model load and the transcription is
//! not measured here.
//!
//! This binary holds one test, so nothing else allocates while it counts. Its
//! global allocator counts every Rust allocation of the process, the client's
//! included, and a reallocation as both blocks at once, whether or not the
//! allocator moves it.

#![allow(unsafe_code, reason = "a global allocator implements an unsafe trait")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stderr,
    reason = "test assertions and the per-case figures"
)]

mod stub_server;

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rmlx_audio::wav::WavEncoder;
use rmlx_server::ModelRegistry;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::SeqCst) + bytes;
    PEAK.fetch_max(live, Ordering::SeqCst);
}

// SAFETY: every call goes to `System` unchanged; the counts change nothing
// else.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size());
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grow(new_size);
        // SAFETY: the caller's contract, passed on.
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
        moved
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// What the route may hold beside the file and its output: the HTTP and
/// multipart read buffers, the reader's buffer and one packet's samples.
const SLACK_BYTES: usize = 2 << 20;

/// A PCM-16 WAV of `secs` seconds: a different tone on each channel.
#[allow(
    clippy::cast_precision_loss,
    reason = "sample indices of a one-minute clip are exact in f32"
)]
fn wav(rate: u32, channels: u16, secs: u32) -> Vec<u8> {
    let frames = rate * secs;
    let samples: Vec<f32> = (0..frames)
        .flat_map(|i| {
            (0..channels).map(move |c| {
                let hz = 440.0 * f32::from(c + 1);
                (std::f32::consts::TAU * hz * i as f32 / rate as f32).sin() * 0.5
            })
        })
        .collect();
    WavEncoder::encode(&samples, rate, channels).expect("encode")
}

/// POST `audio` as the `file` field of a transcription form, after the field
/// `before` (its name and its value) when there is one, and the most bytes
/// live at once while the request runs, above what was live before it.
async fn peak_of_request(
    port: u16,
    audio: &[u8],
    before: Option<&(String, Vec<u8>)>,
) -> (String, usize) {
    let mut body = Vec::with_capacity(audio.len() + 512);
    if let Some((name, value)) = before {
        body.reserve(name.len() + value.len());
        body.extend_from_slice(b"--XBOUNDARY\r\nContent-Disposition: form-data; name=\"");
        body.extend_from_slice(name.as_bytes());
        body.extend_from_slice(b"\"\r\n\r\n");
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(
        b"--XBOUNDARY\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
          Content-Type: application/octet-stream\r\n\r\n",
    );
    body.extend_from_slice(audio);
    body.extend_from_slice(b"\r\n--XBOUNDARY--\r\n");
    let head = format!(
        "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
         Content-Type: multipart/form-data; boundary=XBOUNDARY\r\nContent-Length: {}\r\n\r\n",
        body.len()
    );
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let mut response = Vec::with_capacity(1 << 16);

    let before = LIVE.load(Ordering::SeqCst);
    PEAK.store(before, Ordering::SeqCst);
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(&body).await.unwrap();
    // A refusal sent before the upload is fully read closes with unread bytes,
    // which the kernel turns into a reset. The response read before it stands.
    if let Err(e) = stream.read_to_end(&mut response).await {
        assert_eq!(e.kind(), std::io::ErrorKind::ConnectionReset, "{e}");
    }
    let peak = PEAK.load(Ordering::SeqCst) - before;
    (String::from_utf8_lossy(&response).into_owned(), peak)
}

#[tokio::test]
async fn a_transcription_request_holds_the_file_once_and_its_output() {
    let dir = tempfile::tempdir().unwrap();
    let mut state = stub_server::not_ready_state(ModelRegistry::default());
    state.whisper_model_path = Some(dir.path().join("no-model"));
    state.whisper_tokenizer_path = Some(dir.path().join("no-tokenizer"));
    let port = stub_server::start_server_with_state(state).await;

    // After the first two cases, a field before the file: the route reads
    // and drops what it does not keep, and never holds a long value. A long
    // field name is not here: the multipart parser holds a part's headers
    // before the route sees them (about five times a 24 MiB name).
    let big = 24 << 20;
    let cases = [
        (
            "16 kHz mono WAV",
            wav(16_000, 1, 60),
            16_000 * 60,
            None,
            "tokenizer:",
        ),
        (
            "48 kHz stereo WAV",
            wav(48_000, 2, 60),
            16_000 * 60,
            None,
            "tokenizer:",
        ),
        (
            "1 s WAV after an 8 MiB ignored field",
            wav(16_000, 1, 1),
            16_000,
            Some(("ignored".to_owned(), vec![b'x'; 8 << 20])),
            "tokenizer:",
        ),
        (
            "1 s WAV after a 24 MiB model",
            wav(16_000, 1, 1),
            16_000,
            Some(("model".to_owned(), vec![b'x'; big])),
            "tokenizer:",
        ),
        (
            "1 s WAV after a 24 MiB language of control bytes",
            wav(16_000, 1, 1),
            16_000,
            Some(("language".to_owned(), vec![1; big])),
            "language",
        ),
    ];
    let mut over = Vec::new();
    for (name, audio, output_samples, before, ends_at) in &cases {
        let (response, peak) = peak_of_request(port, audio, before.as_ref()).await;
        assert!(
            response.contains(ends_at),
            "{name}: the request must end at {ends_at}: {}",
            response.get(..512).unwrap_or(&response)
        );
        let allowed = audio.len() + output_samples * size_of::<f32>() + SLACK_BYTES;
        eprintln!(
            "{name}: file {} B, peak beside the request body {peak} B",
            audio.len()
        );
        if peak > allowed {
            over.push(format!("{name}: {peak} B > {allowed} B"));
        }
    }
    assert!(
        over.is_empty(),
        "over the file, its output and slack: {over:?}"
    );
}
