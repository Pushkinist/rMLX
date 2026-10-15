use super::*;
use std::io::Write;
use std::net::TcpListener;

// Minimal 1×1 transparent PNG (46 bytes), for inline tests.
// Generated via: python3 -c "import base64,zlib,struct; ..."
const TINY_PNG_B64: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn tiny_png_bytes() -> Vec<u8> {
    base64_decode(TINY_PNG_B64).unwrap()
}

// ── 1. data: URI ──────────────────────────────────────────────────────────

#[test]
fn data_uri_png() {
    let uri = format!("data:image/png;base64,{TINY_PNG_B64}");
    let bytes = load_image(&uri, DEFAULT_HTTP_TIMEOUT).unwrap();
    assert_eq!(bytes, tiny_png_bytes());
}

#[test]
fn data_uri_no_comma_rejected() {
    assert!(load_image("data:image/png;base64", DEFAULT_HTTP_TIMEOUT).is_err());
}

#[test]
fn data_uri_non_base64_rejected() {
    assert!(load_image("data:image/png,hello", DEFAULT_HTTP_TIMEOUT).is_err());
}

// ── 2. Raw base64 ─────────────────────────────────────────────────────────

#[test]
fn raw_base64_roundtrip() {
    // "foo" → "Zm9v"
    let bytes = load_image("Zm9v", DEFAULT_HTTP_TIMEOUT).unwrap();
    assert_eq!(bytes, b"foo");
}

#[test]
fn raw_base64_whitespace_tolerated() {
    // spaces / newlines inside base64 are stripped
    let bytes = load_image("Zm 9v\n", DEFAULT_HTTP_TIMEOUT).unwrap();
    assert_eq!(bytes, b"foo");
}

// ── 3. File path ──────────────────────────────────────────────────────────

#[test]
fn file_path_absolute() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("img.png");
    std::fs::write(&path, tiny_png_bytes()).unwrap();
    let bytes = load_image(path.to_str().unwrap(), DEFAULT_HTTP_TIMEOUT).unwrap();
    assert_eq!(bytes, tiny_png_bytes());
}

#[test]
fn file_path_missing_returns_err() {
    assert!(load_image("/no/such/file.png", DEFAULT_HTTP_TIMEOUT).is_err());
}

/// A `~/` source that does not exist names the source as sent: no home
/// directory of the server, no OS error text.
#[test]
fn a_missing_home_file_message_holds_no_server_path() {
    let source = "~/rmlx-no-such-dir/img.png";
    let msg = load_image(source, DEFAULT_HTTP_TIMEOUT).unwrap_err();
    assert_eq!(msg, "cannot read image file ~/rmlx-no-such-dir/img.png");
    let home = std::env::var("HOME").expect("HOME is set");
    assert!(!msg.contains(&home), "{msg}");
    assert!(!msg.contains("os error"), "{msg}");
}

/// A file that is missing and a path that cannot be read as a file give the
/// same text, so the message does not tell the two apart.
#[test]
fn every_file_read_failure_has_one_text() {
    let dir = tempfile::tempdir().unwrap();
    let directory = dir.path().to_str().unwrap();
    let missing = dir.path().join("none.png");
    let missing = missing.to_str().unwrap();
    for source in [directory, missing] {
        let msg = load_image(source, DEFAULT_HTTP_TIMEOUT).unwrap_err();
        assert_eq!(msg, format!("cannot read image file {source}"));
    }
}

/// A source longer than the quoted limit is cut, not repeated whole.
#[test]
fn a_long_source_is_cut_in_the_message() {
    let source = format!("/{}", "a".repeat(500));
    let msg = load_image(&source, DEFAULT_HTTP_TIMEOUT).unwrap_err();
    assert_eq!(
        msg,
        format!(
            "cannot read image file /{}...",
            "a".repeat(QUOTED_SOURCE_CHARS - 1)
        )
    );
}

// ── 4. HTTP URL (localhost) ────────────────────────────────────────────────

/// Spin up a bare-bones HTTP/1.1 server on a random port, serve a single
/// response, then shut down. No external network required.
fn serve_once(body: Vec<u8>) -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        // drain the request headers
        let mut buf = [0u8; 4096];
        let _ = std::io::Read::read(&mut stream, &mut buf);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.write_all(&body).unwrap();
    });
    port
}

#[test]
fn url_fetch_localhost() {
    let port = serve_once(tiny_png_bytes());
    let url = format!("http://127.0.0.1:{port}/img.png");
    let bytes = load_image(&url, DEFAULT_HTTP_TIMEOUT).unwrap();
    assert_eq!(bytes, tiny_png_bytes());
}

/// A transport failure has one fixed text: no host, no port, no cause.
#[test]
fn url_transport_failure_has_one_text() {
    let port = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let url = format!("http://127.0.0.1:{port}/img.png");
    let msg = load_image(&url, Duration::from_millis(500)).unwrap_err();
    assert_eq!(msg, "the image URL could not be fetched");
}

/// A status the server got is named.
#[test]
fn url_status_is_named() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 4096];
        let _ = std::io::Read::read(&mut stream, &mut buf);
        stream
            .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });
    let url = format!("http://127.0.0.1:{port}/img.png");
    let msg = load_image(&url, DEFAULT_HTTP_TIMEOUT).unwrap_err();
    assert_eq!(msg, "the image URL answered 404");
}

/// Serve a fake response whose Content-Length exceeds the cap.
#[test]
fn url_oversize_content_length_rejected() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut buf = [0u8; 4096];
        let _ = std::io::Read::read(&mut stream, &mut buf);
        // Announce a body larger than HTTP_MAX_BYTES.
        let fake_cl = HTTP_MAX_BYTES + 1;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: {fake_cl}\r\nConnection: close\r\n\r\n"
        );
        stream.write_all(response.as_bytes()).unwrap();
        // Don't actually send that many bytes; the CL pre-check should fire.
    });
    let url = format!("http://127.0.0.1:{port}/huge.png");
    let result = load_image(&url, DEFAULT_HTTP_TIMEOUT);
    assert!(
        result.is_err(),
        "expected oversize rejection, got {result:?}"
    );
    let msg = result.unwrap_err();
    assert!(
        msg.contains("too large"),
        "error should mention 'too large': {msg}"
    );
}

// ── 5. Live network (ignored in offline CI) ───────────────────────────────

/// Fetch a real tiny PNG from a stable public host.
/// Run explicitly: `cargo test -p rmlx-server -- --ignored url_fetch_live`
#[test]
#[ignore]
fn url_fetch_live() {
    // 1×1 PNG served by the httpbin project (Cloudflare-backed, stable).
    let url = "https://httpbin.org/image/png";
    let bytes = load_image(url, DEFAULT_HTTP_TIMEOUT).unwrap();
    assert!(!bytes.is_empty(), "expected non-empty PNG from {url}");
    // Verify it's a valid PNG magic header.
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "expected PNG magic");
}
