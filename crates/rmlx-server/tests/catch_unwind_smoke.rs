//! Integration smoke test for the `CatchUnwindLayer`.
//!
//! Proves that a panicking handler returns HTTP 500 and does NOT drop the
//! connection or crash the worker — subsequent requests still succeed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr
)]

use axum::routing::get;
use axum::Router;
use rmlx_server::catch_unwind::CatchUnwindLayer;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Test router ───────────────────────────────────────────────────────────────

/// Build a minimal router with:
/// - `/__test_panic` → always panics.
/// - `/health`       → returns 200 OK.
///
/// The `CatchUnwindLayer` is the outermost layer, matching production placement
/// in `build_router`.
fn panic_test_router() -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/__test_panic", get(panic_handler))
        .layer(CatchUnwindLayer::new())
}

async fn health() -> axum::http::StatusCode {
    axum::http::StatusCode::OK
}

async fn panic_handler() -> axum::http::StatusCode {
    panic!("CatchUnwind test panic — deliberate");
}

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Bind the test router on a random port, start serving, return the port.
async fn start() -> u16 {
    let router = panic_test_router();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    // Wait until the server is actually accepting connections (up to 250 ms).
    for _ in 0..50 {
        if TcpStream::connect(("127.0.0.1", port)).await.is_ok() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    port
}

/// Send a raw HTTP/1.1 GET request and return the status line + body.
async fn get_raw(port: u16, path: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect");

    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .await
        .expect("read response");

    let response = String::from_utf8_lossy(&response).into_owned();

    // Parse the status code from "HTTP/1.1 NNN ..."
    let status: u16 = response
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .expect("parse status");

    (status, response)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn panic_recovery_handler_returns_500_and_next_request_succeeds() {
    let port = start().await;

    // 1. A panicking handler must return 500 (not a connection drop / RST).
    let (status_panic, body_panic) = get_raw(port, "/__test_panic").await;
    assert_eq!(
        status_panic, 500,
        "panicking handler must return 500, got {status_panic}; body: {body_panic}"
    );

    // 2. A subsequent request to a normal route must succeed (200).
    //    This proves the worker survived the panic.
    let (status_health, _) = get_raw(port, "/health").await;
    assert_eq!(
        status_health, 200,
        "health check after panic must be 200, got {status_health}"
    );
}

/// Run the recovery scenario 10 times to confirm deterministic behaviour.
///
/// (The `for i in 1..=10` loop from the DoD is embedded here to keep the
/// CI gate inside `cargo test` rather than requiring a shell loop.)
#[tokio::test]
async fn panic_recovery_is_deterministic() {
    let port = start().await;

    for round in 1_u32..=10 {
        let (status_panic, _) = get_raw(port, "/__test_panic").await;
        assert_eq!(
            status_panic, 500,
            "round {round}: expected 500, got {status_panic}"
        );

        let (status_health, _) = get_raw(port, "/health").await;
        assert_eq!(
            status_health, 200,
            "round {round}: expected 200 after panic, got {status_health}"
        );
    }
}
