//! `rmlx serve` under real signals: the bounds the binary passes, the signal
//! handlers and the routes of the server under the shutdown layer.
//!
//! Each test starts the built binary with `--device cpu` and no model, so it
//! takes no Metal claim, loads nothing and reads no weights. The open request
//! is a chat request whose body never arrives in full: its handler waits for
//! the body, as a handler waits for a job.
//!
//! Not covered: the Metal claim of a GPU server at its exit. A test cannot
//! take the machine-wide claim without refusing a server that runs on this
//! Mac. `claim_tests.rs` of `rmlx-server` holds the rule below the process.
//! So no test fails when `run_serve` hands `None` to `block_on_serve` in place
//! of its claim. The proof of that one line is a run of a real GPU server
//! with a chat request in flight at the signal: the log has `Metal claim
//! stays held until the process is gone`, and the next start is not refused.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "test assertions: a failed setup step is a test failure, and the stderr of the child is its cause"
)]

use std::fs::File;
use std::io::{Read as _, Write as _};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// The reply bound of the server, read from the doc that states it. A unit
/// test of `rmlx-server` holds that sentence to the constant of the server,
/// so this file has no copy of the value.
fn reply_bound() -> Duration {
    let doc = include_str!("../../../docs/SERVER_SHUTDOWN.md");
    let (before, _) = doc
        .split_once(" s to send its reply")
        .expect("the doc states the reply bound");
    let seconds = before.rsplit(' ').next().expect("a number before the unit");
    Duration::from_secs(seconds.parse().expect("the reply bound in seconds"))
}

/// The bound on every wait of a test. It is far above the 10 s exit bound, so
/// a loaded machine does not fail a correct server.
const WAIT: Duration = Duration::from_secs(60);

/// A running `rmlx serve`. The process is killed when the test ends early,
/// and a test that fails prints what the process wrote to stderr.
struct Server {
    child: Child,
    port: u16,
    stderr: PathBuf,
    _home: tempfile::TempDir,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.child.kill().ok();
        self.child.wait().ok();
        if std::thread::panicking() {
            let stderr = std::fs::read_to_string(&self.stderr).unwrap_or_default();
            eprintln!("stderr of rmlx serve:\n{stderr}");
        }
    }
}

impl Server {
    /// Start the server and wait until it answers `/health`.
    fn start() -> Self {
        let home = tempfile::tempdir().unwrap();
        let stderr = home.path().join("stderr.txt");
        // The port is free when this bind releases it, and the child binds it
        // a moment later. If another process takes it in between, the child
        // fails its bind and exits, and the loop below fails the test at once
        // with the bind error of the child.
        let port = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let child = Command::new(env!("CARGO_BIN_EXE_rmlx"))
            .env("RMLX_HOME", home.path())
            .args(["serve", "--device", "cpu", "--metrics", "off"])
            .args(["--max-timeout-secs", "0", "--port", &port.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(File::create(&stderr).unwrap())
            .spawn()
            .expect("rmlx serve starts");
        let mut server = Self {
            child,
            port,
            stderr,
            _home: home,
        };
        let started = Instant::now();
        while !server.is_healthy() {
            let exited = server.child.try_wait().unwrap();
            assert!(
                exited.is_none(),
                "rmlx serve exited at its start: {exited:?}"
            );
            assert!(started.elapsed() < WAIT, "the server did not come up");
            std::thread::sleep(Duration::from_millis(50));
        }
        server
    }

    fn connect(&self) -> Option<TcpStream> {
        let stream = TcpStream::connect(("127.0.0.1", self.port)).ok()?;
        stream.set_read_timeout(Some(WAIT)).unwrap();
        Some(stream)
    }

    fn is_healthy(&self) -> bool {
        let Some(mut stream) = self.connect() else {
            return false;
        };
        let request = b"GET /health HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\r\n";
        let mut reply = String::new();
        stream.write_all(request).is_ok()
            && stream.read_to_string(&mut reply).is_ok()
            && reply.starts_with("HTTP/1.1 200")
    }

    /// Open a chat request whose body does not arrive. The server answers
    /// `100 Continue` when the handler starts to read the body, so on return
    /// the request is in flight.
    fn open_chat_request(&self) -> TcpStream {
        let mut stream = self.connect().expect("connect");
        let head = b"POST /v1/chat/completions HTTP/1.1\r\nHost: test\r\n\
            Connection: close\r\nContent-Type: application/json\r\n\
            Content-Length: 4096\r\nExpect: 100-continue\r\n\r\n";
        stream.write_all(head).unwrap();
        let mut interim = Vec::new();
        while !interim.ends_with(b"\r\n\r\n") {
            let mut byte = [0_u8];
            stream.read_exact(&mut byte).expect("the interim reply");
            interim.push(byte[0]);
        }
        assert!(interim.starts_with(b"HTTP/1.1 100"), "{interim:?}");
        stream
    }

    fn signal(&self, signal: &str) {
        let status = Command::new("/bin/kill")
            .args([signal, &self.child.id().to_string()])
            .status()
            .expect("run kill");
        assert!(status.success(), "kill {signal} failed");
    }

    /// The exit code. Panics when the process is not gone inside `WAIT`.
    fn exit_code(&mut self) -> Option<i32> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status.code();
            }
            assert!(started.elapsed() < WAIT, "the server did not exit");
            std::thread::sleep(Duration::from_millis(20));
        }
    }
}

/// The whole reply of the open request, which must be the 503 of the
/// shutdown in the error envelope of the server.
fn read_shutdown_503(mut request: TcpStream) {
    let mut reply = String::new();
    request.read_to_string(&mut reply).expect("the final reply");
    assert!(reply.starts_with("HTTP/1.1 503"), "{reply}");
    assert!(reply.contains(r#""type":"service_unavailable""#), "{reply}");
    assert!(reply.contains("the server is shutting down"), "{reply}");
}

/// One SIGTERM with a request in flight: the request gets its 503 at the
/// reply bound and not before it, and the process exits with code 0.
#[test]
fn a_sigterm_answers_the_open_request_at_the_reply_bound_and_exits() {
    let mut server = Server::start();
    let request = server.open_chat_request();

    let signalled = Instant::now();
    server.signal("-TERM");

    read_shutdown_503(request);
    assert!(
        signalled.elapsed() >= reply_bound(),
        "the 503 came {:?} after the signal, before the reply bound",
        signalled.elapsed()
    );
    assert_eq!(server.exit_code(), Some(0));
}

/// A second signal ends the wait: the request gets its 503 and the process
/// exits with code 0, both before the reply bound of the first signal. The
/// two signals differ, so the kernel cannot merge them into one.
#[test]
fn a_second_signal_answers_the_open_request_and_exits_before_the_reply_bound() {
    let mut server = Server::start();
    let request = server.open_chat_request();

    let signalled = Instant::now();
    server.signal("-TERM");
    server.signal("-INT");

    read_shutdown_503(request);
    assert_eq!(server.exit_code(), Some(0));
    assert!(
        signalled.elapsed() < reply_bound(),
        "the exit came {:?} after the signals: the second one was not read",
        signalled.elapsed()
    );
}

/// With no request in flight one SIGINT ends the process before the reply
/// bound, with code 0.
#[test]
fn a_sigint_with_nothing_in_flight_exits_at_once() {
    let mut server = Server::start();

    let signalled = Instant::now();
    server.signal("-INT");

    assert_eq!(server.exit_code(), Some(0));
    assert!(
        signalled.elapsed() < reply_bound(),
        "the idle exit took {:?}",
        signalled.elapsed()
    );
}
