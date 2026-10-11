//! The multipart form of the transcription routes: field bounds and buffer
//! sizing, on forms built in the test.

use super::{parse_multipart, AudioFormFields, IgnoredFields};

/// The ignored-field log event is bounded: a count, at most eight names, each
/// at most 64 bytes, cut at a character boundary.
#[test]
fn ignored_fields_are_counted_and_their_names_bounded() {
    let mut ignored = IgnoredFields::default();
    let long = "é".repeat(1 << 20);
    for _ in 0..400_000 {
        ignored.note(&long);
    }
    ignored.note("prompt");
    assert_eq!(ignored.count, 400_001);
    assert_eq!(ignored.names.len(), 8);
    for name in &ignored.names {
        assert_eq!(name.len(), 64, "{name}");
        assert!(name.chars().all(|c| c == 'é'));
    }
    let mut few = IgnoredFields::default();
    few.note("prompt");
    few.note("include[]");
    assert_eq!(
        few,
        IgnoredFields {
            count: 2,
            names: vec!["prompt".to_owned(), "include[]".to_owned()],
        }
    );
}

/// The form `parse_multipart` reads from `body`, a multipart body with the
/// boundary `B`, given its length as the request's `Content-Length`.
async fn parsed(body: Vec<u8>) -> AudioFormFields {
    use axum::extract::FromRequest;

    let len = body.len();
    let request = axum::http::Request::builder()
        .header("content-type", "multipart/form-data; boundary=B")
        .body(axum::body::Body::from(body))
        .expect("a request");
    let mut multipart = axum::extract::Multipart::from_request(request, &())
        .await
        .expect("a multipart body");
    parse_multipart(&mut multipart, Some(len))
        .await
        .expect("the form is taken")
}

/// A four-byte file after one other field.
fn form_before_file(name: &str, value: &[u8]) -> Vec<u8> {
    let mut body =
        format!("--B\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n").into_bytes();
    body.extend_from_slice(value);
    body.extend_from_slice(
        b"\r\n--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a\"\r\n\r\nRIFF\r\n--B--\r\n",
    );
    body
}

/// `model` is only logged: a 1 MiB value is taken, and the form keeps its
/// first 64 bytes.
#[tokio::test]
async fn a_long_model_is_kept_cut_to_64_bytes() {
    let fields = parsed(form_before_file("model", &vec![b'm'; 1 << 20])).await;
    assert_eq!(fields.model, "m".repeat(64));
}

/// The file's buffer is sized from `Content-Length` less the fields before
/// it, their names included: a 1 MiB field name before a four-byte file
/// reserves no megabyte.
#[tokio::test]
async fn the_file_buffer_leaves_out_the_names_before_it() {
    let name = "n".repeat(1 << 20);
    let fields = parsed(form_before_file(&name, b"x")).await;
    assert_eq!(fields.audio_bytes, b"RIFF");
    assert!(
        fields.audio_bytes.capacity() < 4096,
        "a {} B buffer",
        fields.audio_bytes.capacity()
    );
}
