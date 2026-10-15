//! The multipart form of the transcription and translation routes.

use axum::extract::multipart::{Field, MultipartError, MultipartRejection};
use axum::extract::Multipart;
use axum::http::StatusCode;
use rmlx_audio::transcribe::requested_language;
use tracing::warn;

use crate::tts_sampling::MIN_SAMPLING_VALUE;

use super::{
    bad_field, cut, too_large, unprocessable, Refusal, LOGGED_NAME_BYTES, MAX_AUDIO_BYTES,
    MAX_REQUEST_BYTES,
};

/// The most parts a form may hold, the file included.
const MAX_FORM_PARTS: usize = 64;

/// The most bytes the route keeps of a field other than `file`: more than any
/// value it takes, so the refusals that echo a value stay short.
const SHORT_FIELD_BYTES: usize = 64;

/// The fields of a form the route does not read: how many, and the first
/// names, each cut, for one log event of bounded size.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct IgnoredFields {
    pub(super) count: usize,
    pub(super) names: Vec<String>,
}

impl IgnoredFields {
    /// The most names the log event lists.
    const NAMES: usize = 8;

    pub(super) fn note(&mut self, name: &str) {
        self.count += 1;
        if self.names.len() < Self::NAMES {
            self.names.push(cut(name, LOGGED_NAME_BYTES).to_owned());
        }
    }
}

pub(super) struct AudioFormFields {
    pub(super) audio_bytes: Vec<u8>,
    pub(super) model: String,
    /// `auto`, or the language the request named: a code or an English name.
    pub(super) language: String,
    pub(super) response_format: ResponseFormat,
    pub(super) temperature: f32,
    /// The `prompt` text, whole: the transport limit of the form bounds it.
    pub(super) prompt: String,
    /// `timestamp_granularities` holds `word`: the reply holds word times.
    pub(super) words: bool,
    /// `stream` is true: the reply is server-sent events. Only `json` streams.
    pub(super) stream: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum ResponseFormat {
    #[default]
    Json,
    Text,
    VerboseJson,
    Srt,
    Vtt,
}

impl ResponseFormat {
    /// An unknown value is an error, not the default: the route answers 422.
    fn parse_or_default(s: &str) -> Result<Self, String> {
        match s {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "verbose_json" => Ok(Self::VerboseJson),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            other => Err(format!(
                "unsupported response_format '{other}'; must be one of: json, text, verbose_json, srt, vtt"
            )),
        }
    }
}

/// Read the form. `body_len`, the request's `Content-Length`, less the bytes of
/// the fields before the file (names and values), sizes the file's buffer, so
/// the upload is one allocation and is never copied. Every other field is read
/// chunk by chunk: a field the route reads keeps its first
/// `SHORT_FIELD_BYTES` bytes, and one longer than that is refused, so no
/// refusal echoes more of the client's text, except `prompt`, which is kept
/// whole; a field the route ignores keeps nothing. After a form that is not refused, one `warn` event counts and names
/// the ignored fields.
pub(super) async fn parse_multipart(
    multipart: &mut Multipart,
    body_len: Option<usize>,
) -> Result<AudioFormFields, Refusal> {
    let unreadable = |e: MultipartError| unreadable_form(&e, body_len);
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut model = "whisper-large-v3".to_owned();
    let mut language = "auto".to_owned();
    let mut response_format = ResponseFormat::Json;
    let mut temperature = 0.0_f32;
    let mut prompt = String::new();
    let mut words = false;
    let mut stream = false;
    let mut ignored = IgnoredFields::default();
    let mut parts = 0_usize;
    // Bytes of the fields read before the file: not the file's.
    let mut read_before_file = 0_usize;

    while let Some(mut field) = multipart.next_field().await.map_err(unreadable)? {
        parts += 1;
        if parts > MAX_FORM_PARTS {
            return Err(unprocessable(format!(
                "the form has more than {MAX_FORM_PARTS} parts"
            )));
        }
        let name = field.name().unwrap_or("").to_owned();
        read_before_file += name.len();
        match name.as_str() {
            "file" => {
                let reserve = body_len.unwrap_or(0).saturating_sub(read_before_file);
                let mut bytes = Vec::with_capacity(reserve.min(MAX_AUDIO_BYTES));
                while let Some(chunk) = field.chunk().await.map_err(unreadable)? {
                    bytes.extend_from_slice(&chunk);
                }
                audio_bytes = Some(bytes);
            }
            // Only logged: a longer value is kept cut, not refused.
            "model" => {
                let (text, len) = read_short(&mut field, body_len).await?;
                read_before_file += len;
                model = text;
            }
            "language" => {
                let text = short_value(&name, read_short(&mut field, body_len).await?)?;
                if !text.is_empty() {
                    requested_language(&text).map_err(|e| bad_field("language", e.to_string()))?;
                }
                language = if text.is_empty() {
                    "auto".to_owned()
                } else {
                    text
                };
            }
            "response_format" => {
                let text = short_value(&name, read_short(&mut field, body_len).await?)?;
                response_format = ResponseFormat::parse_or_default(&text)
                    .map_err(|message| bad_field("response_format", message))?;
            }
            "temperature" => {
                let text = short_value(&name, read_short(&mut field, body_len).await?)?;
                let t = text.parse::<f32>().map_err(|_| {
                    bad_field(
                        "temperature",
                        format!(
                            "invalid temperature '{text}'; must be 0 or a finite float >= 1e-6"
                        ),
                    )
                })?;
                // Below the bound the sampler's scaled logit is not finite.
                if !t.is_finite() || t < 0.0 || (t > 0.0 && t < MIN_SAMPLING_VALUE) {
                    return Err(bad_field(
                        "temperature",
                        format!(
                            "temperature {t} is out of range; must be 0 or a finite float >= 1e-6"
                        ),
                    ));
                }
                temperature = t;
            }
            "prompt" => {
                let mut bytes = Vec::new();
                while let Some(chunk) = field.chunk().await.map_err(unreadable)? {
                    bytes.extend_from_slice(&chunk);
                }
                read_before_file += bytes.len();
                prompt = String::from_utf8(bytes)
                    .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
            }
            // OpenAI's SDKs send an array field under its name with `[]`.
            "timestamp_granularities" | "timestamp_granularities[]" => {
                let text = short_value(&name, read_short(&mut field, body_len).await?)?;
                match text.as_str() {
                    "segment" => {}
                    "word" => words = true,
                    other => {
                        return Err(bad_field(
                            &name,
                            format!("unknown {name} '{other}'; must be 'word' or 'segment'"),
                        ));
                    }
                }
            }
            "stream" => {
                let text = short_value(&name, read_short(&mut field, body_len).await?)?;
                match text.to_ascii_lowercase().as_str() {
                    "false" | "0" | "" => stream = false,
                    "true" | "1" => stream = true,
                    _ => {
                        return Err(bad_field(
                            "stream",
                            format!("invalid stream '{text}'; must be true or false"),
                        ));
                    }
                }
            }
            other => {
                ignored.note(other);
                while let Some(chunk) = field.chunk().await.map_err(unreadable)? {
                    read_before_file += chunk.len();
                }
            }
        }
    }
    if ignored.count > 0 {
        warn!(
            ignored = ignored.count,
            fields = ?ignored.names,
            "form fields ignored"
        );
    }

    let audio_bytes =
        audio_bytes.ok_or_else(|| bad_field("file", "missing required field: file".to_owned()))?;
    if audio_bytes.is_empty() {
        return Err(bad_field("file", "audio file is empty".to_owned()));
    }
    if words && response_format != ResponseFormat::VerboseJson {
        return Err(bad_field(
            "timestamp_granularities",
            "timestamp_granularities 'word' needs response_format verbose_json".to_owned(),
        ));
    }

    if stream && response_format != ResponseFormat::Json {
        return Err(bad_field(
            "stream",
            "stream=true needs response_format json; the other formats are sent whole when \
             the transcription ends"
                .to_owned(),
        ));
    }

    Ok(AudioFormFields {
        audio_bytes,
        model,
        language,
        response_format,
        temperature,
        prompt,
        words,
        stream,
    })
}

/// The first `SHORT_FIELD_BYTES` bytes of a field as text, and the field's
/// length in bytes. The bytes past the limit are read and dropped.
async fn read_short(
    field: &mut Field<'_>,
    body_len: Option<usize>,
) -> Result<(String, usize), Refusal> {
    let mut kept = Vec::new();
    let mut len = 0;
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|e| unreadable_form(&e, body_len))?
    {
        let room = SHORT_FIELD_BYTES
            .saturating_sub(kept.len())
            .min(chunk.len());
        kept.extend_from_slice(chunk.get(..room).unwrap_or_default());
        len += chunk.len();
    }
    Ok((String::from_utf8_lossy(&kept).into_owned(), len))
}

/// The text of the field `name` read by [`read_short`], refused when the field
/// is longer than the route keeps: the refusal names the field and echoes none
/// of it.
fn short_value(name: &str, (text, len): (String, usize)) -> Result<String, Refusal> {
    if len > SHORT_FIELD_BYTES {
        return Err(bad_field(
            name,
            format!("{name} is over {SHORT_FIELD_BYTES} bytes"),
        ));
    }
    Ok(text)
}

/// A request that the multipart extractor refused before the form was read:
/// its status and text, in the error envelope.
pub(super) fn rejected_form(rejection: &MultipartRejection) -> Refusal {
    Refusal {
        status: rejection.status(),
        ..unprocessable(rejection.body_text())
    }
}

/// A form the server could not read. A body over the transport limit is 413
/// naming the audio file cap; any other fault is 422 with the parser's text.
fn unreadable_form(e: &MultipartError, body_len: Option<usize>) -> Refusal {
    if e.status() != StatusCode::PAYLOAD_TOO_LARGE {
        return unprocessable(e.to_string());
    }
    let size = body_len.map_or_else(String::new, |n| format!(" {n} bytes,"));
    too_large(format!(
        "request body is{size} over the limit of {MAX_REQUEST_BYTES} bytes: an audio file of \
         at most {MAX_AUDIO_BYTES} bytes (25 MiB) and 1 MiB of other form fields"
    ))
}

#[cfg(test)]
#[path = "form_tests.rs"]
mod form_tests;
