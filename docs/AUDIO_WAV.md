# WAV I/O

`WavDecoder::decode` is Symphonia-backed. It reads WAV (PCM), MP3, MP4/M4A
(AAC) and ADTS (AAC). FLAC, Ogg and the other Symphonia formats are not
compiled in, and their probe fails. It returns `DecodedAudio`: mono f32 at the
input's own sample rate, the length the file declares, the end of the last
packet read, how the reader stopped, and the audio lost (`AudioLoss`).
`WavDecoder::decode_resampled(bytes, rate, max_secs, cancelled)` returns the
same with the samples at `rate`; Whisper and the Gemma4 audio input take it
at 16 kHz. `max_secs` is `MAX_AUDIO_SECS` on the transcription routes, no cap
for `rmlx transcribe`, and on the chat `input_audio` path what the context
holds (see "Duration cap"). `cancelled` is asked before each packet; when it
says the caller is gone the decode stops with `WavError::Cancelled`.

**Resample and memory.** The channels are averaged and the result is
resampled by linear interpolation packet by packet. The samples are the same,
bit for bit, as a linear resample of the whole mono stream. More than two
channels are a plain average, each channel 1/n, the LFE included; `ffmpeg
-ac 1` uses a downmix matrix that drops the LFE (an LFE-only 5.1 file is
silence there and 1/6 of its level here; ffmpeg 9.0.2).

The decoder takes a file rate, and a resample rate, of 4 kHz to 768 kHz
(`SAMPLE_RATES`) and refuses any other (`WavError::SampleRate`). The floor
bounds a resample to 16 kHz at four outputs an input: an 8-bit mono WAV at
4 kHz gives 16 bytes of output a byte of file.

Beside the file's bytes, a decode holds its output and about one packet: no
copy of the bytes, no buffer of all channels, no buffer at the file's own
rate. The output is sized from the length the file declares or Symphonia
guesses, never past the frames the bytes can hold: one a sample's bytes for
a WAV (a WAV whose data length is a placeholder is sized from that alone),
256 a byte for a compressed stream (an AAC-LC frame of 1024 samples takes at
least 4 bytes). For a well-formed file the declared length is the smaller.
When the stream holds more, the buffer grows, and holds its old and its new
copy while it moves. The silence for dropped packets is at most 5 s
(`LOST_CAP_SECS`): see "Lost audio".

The output at 16 kHz is 64,000 bytes per second of audio: 3.84 MB a minute,
230.4 MB an hour.

**Duration cap.** The server's 25 MiB upload cap is a size, not a length,
and compressed audio of that size holds more than 7 hours. The duration cap,
`MAX_AUDIO_SECS`, bounds the audio itself on the server; `rmlx transcribe`
decodes a local file of any length:

| Bound | Value | What reaches it | Memory at the bound |
|---|---|---|---|
| Upload cap (server only) | 25 MiB (26,214,400 bytes) of file | 13.7 min of 16 kHz mono PCM-16 WAV, 2.3 min of 48 kHz stereo PCM-16 WAV, 27.3 min of MP3 at 128 kbit/s, 54.6 min at 64 kbit/s, 7.3 h at 8 kbit/s | the file |
| Duration cap | 3 h (10,800 s) of audio | every file the server decodes, of any size | 691.2 MB of 16 kHz f32 output, beside the file |

A file that declares more than the cap is refused before any packet decodes
(`WavError::TooLong`, `the file declares … s of audio, over the 10800 s the
decoder takes`; 422 `audio decode: …` on the server). A stream that declares
no length (ADTS, an MP3 with no LAME tag, a WAV through a pipe) is refused
before its output passes the cap. The end padding a LAME tag declares is read
before the trim and is not counted; the cap applies to what the trim leaves,
so an MP3 that ends before its padding is refused if it holds more than the
cap. A one-hour 16 kHz mono WAV is 115 MB and
is refused by the upload cap, not by the duration cap.

A chat `input_audio` clip (Gemma4) is capped lower: at the whole seconds
whose audio tokens fit the context left after the prompt and the two audio
markers, 25 tokens a second (640 samples a token on both audio branches),
and at most `MAX_AUDIO_SECS`. With an 8192-token context and a 100-token
prompt that is 323 s. A clip over it is a 400, refused before the mel.

The peak of a read and a decode to 16 kHz, counted on the
heap with the file's bytes: 3.0 times the file for a 16 kHz mono PCM-16 WAV
(the f32 output alone is twice the file), 1.33 times for a 48 kHz stereo
PCM-16 WAV, and the file plus 230.9 MB for an hour of 48 kHz stereo MP3 at
64 kbit/s.

**Declared length.** A WAV header, an MP4 track and the LAME tag of an MP3
declare the stream length; a length of 0 (a fragmented MP4) is no length.
Symphonia also fills a length for an ADTS stream and for an MP3 with no LAME
tag, as a guess from the bitrate; the decoder does not take a guess as a
declared length. A Xing or VBRI frame count with no LAME tag is exact, but
Symphonia does not say whether it read one or guessed, so it is not used
either.

A WAV data length of `0x7FFF0000` or more is a placeholder, not a length,
and the stream is not held to it. Writers that cannot seek back to patch the
header leave one: `0xFFFFFFFF` (ffmpeg to a pipe), `0x7FFFF000` rounded down
to the block (sox 14.4.2 to a pipe), `0x80000000` (arecord to stdout). Below
the bound, the length is held to: a WAV cut short of it loses the rest, and
the shortfall counts as lost audio.

Known limits:

- A real WAV with a data length of `0x7FFF0000` (2 GiB) or more that is cut
  decodes as a complete shorter file.
- The header shapes some writers leave on a pipe fail Symphonia's probe, as
  they did before this check: sox after 14.4.2 (RIFF length 36, data
  `0xFFFFFFFF`) and libsndfile (RIFF length 8, data 0).

**Trim.** The MP3 encoder delay (frames before time 0) is left out. The
padding a LAME tag declares is cut when the stream ends where the tag says.
When the stream holds more than the tag declares (two MP3 files joined, or a
frame count that is low), all of it is kept and a `warn` says so; `ffmpeg`
also decodes the frames past the count.

**Lost audio.** Two things lose audio:

- A packet that does not decode is dropped: a `DecodeError`, an `IoError` (the
  AAC decoder reads past the end of a mangled packet) or `Unsupported` (a
  feature bit set by damage). Its length of silence takes its place, so the
  audio after it stays at its time on the stream and the segment times hold.
- The stream ends short of its declared length.

`AudioLoss` gives the dropped packets, their seconds, the first and last
position, the first error and the shortfall. The limit scales with the
stream: `lost_limit_secs` is 1 % (`LOST_FRACTION`) of the stream's length
(the declared length, or the end reached when that is later), never below
0.2 s (`LOST_FLOOR_SECS`, more than one frame of any compiled codec at
8 kHz) and never above 5 s (`LOST_CAP_SECS`). A 5 s clip may lose 0.2 s, a
100 s file 1 s, a file of 500 s or more 5 s. Up to the limit the decode goes
on and a `warn` event, `audio lost under the limit`, carries `lost_secs`,
`limit_secs`, `dropped_packets`, `dropped_secs`, `short_secs` and
`first_drop_at_secs` as fields. Above it, the decode is refused. The limit
does not depend on where in the file the loss is: one bad frame at the end
and one in the middle are treated the same. The three numbers are a default
the owner may change.

Dropped packets that hold more than `LOST_CAP_SECS`, the largest limit, are
refused when that point is reached, before their silence is written; the
message then gives the limit of the declared length, or of the length read
so far when the file declares none.

**Refusals.** Each is a `WavError`; the message gives the time in seconds.

- `LostAudio`: more lost than the limit for the stream's length. For example
  an MP3 cut short of its LAME length, or a second MP3 at another sample rate
  appended to the first.
- `ReadFailed`: the container reader returns an error other than running out
  of bytes. An MP4 cut inside its media data, an ADTS header that is damaged.
- `PacketFailed`: the decoder fails in a way a damaged packet does not
  explain (it needs a reset, or it hit a limit), or no packet of the stream
  decodes (a feature the decoder does not support); the error is the
  decoder's on the first packet.
- `Empty`: the stream holds no packet.

A `warn` event comes before each refusal, with the declared and reached
seconds and the end. A `debug` event, `audio stream read`, gives the same for
every decode, with the end: `EndOfStream` (the reader reported it) or
`EndOfBytes` (the bytes ran out inside a packet). When the bytes run out
inside a packet of a stream that declares no length and is not a WAV, a
`warn` event, `the bytes end inside a packet; the file may be cut`, gives
`reached_secs`.

An M4A decodes 1000 to 3000 samples more than `ffmpeg` gives (measured 1152 to
2816): Symphonia does not read the MP4 edit list, so the AAC encoder's
priming at the start and its padding at the end stay in the samples.

**What no check sees.** A stream that declares no length ends where its bytes
end: a WAV written to a pipe, an MP3 with no LAME tag, an ADTS stream. Such a
file, cut short, decodes as a complete shorter file. The MP3 and ADTS readers
also skip bytes that are not a frame, with no error and no timestamp gap:
frames that garbage replaced in the middle of such a stream are missing from
the samples and from the count. With a LAME tag the missing frames show as a
shortfall at the end and count as lost.

`WavEncoder::encode` writes a 44-byte RIFF PCM-16 LE header, then the samples.
Each f32 in `[-1, 1]` is clamped and rounded to `i16`.
