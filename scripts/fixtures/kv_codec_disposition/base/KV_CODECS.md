# Synthetic KV_CODECS.md

Stand-in for `docs/KV_CODECS.md`, the first doc the gate lists in
`BANNER_DOCS`, read only by `scripts/check_kv_codec_disposition_fixtures.sh`.
Each per-variant section below mirrors the shape the gate scans for: a `### `
heading, then an INERT banner within three lines of it when the codec is inert,
then prose.

### fixinert

> **INERT on this build** — `fixinert` decode reads the bf16 mirror on both axes, so its packed store is never built.

Pack-format prose, present tense, describing a store nothing builds. This is
exactly the paragraph the banner above exists to qualify.

### fixlive

Prose. No banner: this codec decodes over its own packed store, so a banner
here would retire a working codec in the reader's head.

### fixbase

Prose. The unquantised baseline has no packed store to skip, so it earns no
banner either.
