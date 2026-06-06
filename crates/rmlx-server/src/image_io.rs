//! Shared image-source loader: HTTP(S) URL, `data:` URI, raw base64, file path.
//!
//! Used by both `/v1/embeddings` (jina-v4 image input) and the chat image path
//!. Returns raw image bytes that callers then hand to
//! their codec (e.g. [`rmlx_models::jina_v4::preprocess_image_bytes`]).
//!
//! ## Input variants (auto-detected)
//! 1. `https://…` / `http://…` — blocking HTTP GET, 10 s timeout, 50 MiB cap.
//! 2. `data:[<type>];base64,<b64>` — inline data URI (RFC 2397).
//! 3. `/<path>` / `./…` / `~/…` or any path that `Path::is_file()` returns true

//! 4. Anything else — treated as raw base64.

use std::time::Duration;

use tracing::{debug, instrument};

/// Maximum number of bytes accepted from an HTTP response (50 MiB).
const HTTP_MAX_BYTES: usize = 50 * 1024 * 1024;

/// Default HTTP timeout used by [`load_image`].
pub const DEFAULT_HTTP_TIMEOUT: Duration = Duration::from_secs(10);

/// Load raw image bytes from `source`.
///
/// `timeout` applies only to HTTP(S) fetches; it is ignored for all other
/// source kinds.
///
/// Returns `Err(String)` with a human-readable message on failure.
#[instrument(skip(source), fields(kind = %detect_kind(source)), level = "debug")]
pub fn load_image(source: &str, timeout: Duration) -> Result<Vec<u8>, String> {
    let s = source.trim();

    // ── 1. HTTP(S) URL ────────────────────────────────────────────────────────
    if s.starts_with("https://") || s.starts_with("http://") {
        return fetch_url(s, timeout);
    }

    // ── 2. data: URI ──────────────────────────────────────────────────────────
    if let Some(rest) = s.strip_prefix("data:") {
        let comma = rest
            .find(',')
            .ok_or_else(|| "malformed data URI (no comma)".to_owned())?;
        let meta = &rest[..comma];
        let payload = &rest[comma + 1..];
        if !meta.contains("base64") {
            return Err("only base64 data URIs are supported".to_owned());
        }
        return base64_decode(payload).map_err(|e| format!("data URI base64 decode: {e}"));
    }

    // ── 3. File path ──────────────────────────────────────────────────────────
    let looks_pathish = s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("~/")
        || std::path::Path::new(s).is_file();
    if looks_pathish {
        let p = if let Some(home_rel) = s.strip_prefix("~/") {
            match std::env::var_os("HOME") {
                Some(h) => std::path::PathBuf::from(h).join(home_rel),
                None => std::path::PathBuf::from(s),
            }
        } else {
            std::path::PathBuf::from(s)
        };
        return std::fs::read(&p)
            .map_err(|e| format!("cannot read image file {}: {e}", p.display()));
    }

    // ── 4. Raw base64 ─────────────────────────────────────────────────────────
    base64_decode(s).map_err(|e| format!("base64 decode: {e} (not a data URI or readable path)"))
}

/// One-word label for tracing field (no I/O).
fn detect_kind(s: &str) -> &'static str {
    let s = s.trim();
    if s.starts_with("https://") || s.starts_with("http://") {
        "url"
    } else if s.starts_with("data:") {
        "data_uri"
    } else if s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("~/")
        || std::path::Path::new(s).is_file()
    {
        "file"
    } else {
        "base64"
    }
}

/// Blocking HTTP(S) GET with `timeout` and a [`HTTP_MAX_BYTES`] response cap.
fn fetch_url(url: &str, timeout: Duration) -> Result<Vec<u8>, String> {
    debug!(
        url,
        timeout_ms = timeout.as_millis(),
        "fetching image via HTTP"
    );

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(timeout))
        .build()
        .into();

    let mut response = agent
        .get(url)
        .call()
        .map_err(|e| format!("HTTP fetch {url}: {e}"))?;

    // Honour Content-Length pre-check to refuse clearly oversized payloads
    // before reading the body.
    if let Some(cl) = response.headers().get("content-length") {
        let len: usize = cl.to_str().ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if len > HTTP_MAX_BYTES {
            return Err(format!(
                "image response too large: Content-Length {len} > {HTTP_MAX_BYTES} bytes"
            ));
        }
    }

    // read_to_vec has a built-in 10 MiB default cap; override to our 50 MiB
    // limit so the cap is defined in one place (HTTP_MAX_BYTES above).
    let bytes = response
        .body_mut()
        .with_config()
        .limit(HTTP_MAX_BYTES as u64)
        .read_to_vec()
        .map_err(|e| format!("reading HTTP body from {url}: {e}"))?;

    debug!(url, bytes = bytes.len(), "image fetch complete");
    Ok(bytes)
}

// ── Internal base64 decoder (RFC 4648 standard alphabet, optional padding) ───

pub(crate) fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut bits = 0u32;
    let mut nbits = 0u32;
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    for &c in input.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c).ok_or_else(|| format!("invalid base64 char {:?}", c as char))?;
        bits = (bits << 6) | u32::from(v);
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
        }
    }
    if out.is_empty() {
        return Err("empty base64 payload".to_owned());
    }
    Ok(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "image_io_tests.rs"]
mod tests;
