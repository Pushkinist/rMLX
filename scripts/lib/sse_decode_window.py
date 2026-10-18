#!/usr/bin/env python3
"""Time a streamed chat-completions response over its decode window.

Sends one `stream: true` request, stamps each chunk as it arrives, and reports
the rate over the same window the engine reports: first content token to last,
`(n - 1) / (t_last - t_first)`. Prefill, connection setup and the process spawn
all land before the first content chunk and are outside it, which is what makes
this number comparable with `rmlx baseline`'s `decode_tps` and with the
`decode_tps` a speculative round loop logs (docs/SPEC_READING_A_RUN.md). Dividing the
completion tokens by the whole request duration measures something else and
reads low by the prefill.

This process sends the request itself. A reader fed by a separate `curl` could
start after the first chunks arrived and stamp them all at its own start, which
shrinks the window and reads the rate high; here nothing can arrive before the
reader exists.

By default each chunk is stamped with this process's clock when it is read.
Under the explicit argument `--server-stamps` the stamps come from the server
instead: the line `: sent-at=<seconds>` before a content chunk is that chunk's
stamp, in the server's time base. Only a self-test passes that argument, with a
stub server that sends the stamps. A stamp that is not a finite number, does not
follow the one before it, or is missing on a content chunk is an error that
names it. Without the argument a `: sent-at=` line is an ordinary comment. The
rMLX server sends no stamps.

A response with fewer than two content chunks has no interval and no rate; the
`decode_tps` line is then absent rather than zero, because a zero in that slot
is averaged and ranked as a real throughput.

The window is timed from content-chunk arrivals. A completion's token count is
normally *larger* than that — a stop token is counted and carries no content —
so the two disagreeing is not by itself a fault. More content chunks than
counted tokens is: the two numbers are then describing different things and
neither can be trusted, so that direction is refused.

The opposite hazard, a server batching several tokens into one chunk, makes the
window read low by the batching factor and is **not** detectable from these
counts — a batched stream looks exactly like one with uncounted stop tokens.
What catches it is the caller comparing this rate against the engine's own
reading of the same window; see `scripts/spec_bench.sh`.

Usage: `sse_decode_window.py --url URL --body-file FILE [--raw FILE]`. The file
holds the JSON request body, which is POSTed with `Content-Type:
application/json`.

With `--complete FILE` the request is not timed and not streamed: the body asks
for a whole response, this process writes it to `FILE` and prints the digest of
its text, reasoning first and then content, computed by the same function as
the streamed digest. For a response whose reasoning comes before its content,
which is how a model writes them, the two digests of one answer are equal. The
output is `answer_sha256=<hex>`.

Output (stdout), one `key=value` per line:

    clock=<source>      `client` when this process timed the chunks,
                        `server_stamps` when the server's stamps did. A rate
                        read from stamps is not a measurement of any client.
    tokens=<n>          completion tokens from the usage chunk, or — when it
                        carried none — the count of content chunks. A caller
                        that must not publish a client derivation under an
                        engine field's name reads `completion_tokens` instead.
    completion_tokens=<n>  from the usage chunk; omitted when it carried none
    content_chunks=<n>  chunks that carried text — the window's token count
    prompt_tokens=<n>   from the usage chunk; omitted when it carried none
    decode_tps=<f>      omitted when the response has no measurable window
    answer_sha256=<hex> sha256 of the whole completion, reasoning included
    preview=<text>      first 64 characters of the completion, newlines folded

`answer_sha256` is what a caller comparing two responses reads. `preview` is 64
characters, and two answers that agree for a sentence and then diverge share it,
so comparing previews would call them one answer. The digest covers
`reasoning_content` as well as `content`, for the same reason the window counts
both: they are tokens the model generated, and a run that thought differently
answered differently.

`--raw` is truncated before the request is sent and holds this request's lines
only, also when the request fails.

Exit codes: 0 — read; 2 — `--raw` or the `--complete` file could not be
written; 3 — more content chunks arrived than the completion had tokens; 4 — the
request could not be sent, the response did not end as a complete response (a
chunked body without its last chunk, a body shorter than its length), or a
`--complete` response was not a 200; 5 — a server stamp is unusable; 6 — a
`--complete` response is not a chat completion.
"""

import argparse
import hashlib
import http.client
import json
import math
import sys
import time
import urllib.parse

PREVIEW_CHARS = 64
SENT_AT = ": sent-at="


class StampError(Exception):
    """A server stamp that cannot be used as a time."""


class ChunkCountMismatch(Exception):
    """More content chunks arrived than the completion is said to have tokens."""


def read_stamped(stream, server_stamps=False):
    """(clock_time, server_stamp, line) for every line of `stream`.

    `clock_time` is this process's clock when the line was read. `server_stamp`
    is the text after `: sent-at=` on the line before, when `server_stamps` is
    set, and None otherwise.

    Explicit `readline()` rather than iteration: iterating a buffered stream
    reads ahead, which would stamp a whole block of tokens with the arrival
    time of the last one and collapse the window.
    """
    sent_at = None
    while True:
        line = stream.readline()
        if not line:
            return
        text = line.decode("utf-8", errors="replace")
        if server_stamps and text.startswith(SENT_AT):
            sent_at = text[len(SENT_AT) :].strip()
            continue
        yield time.monotonic(), sent_at, text
        sent_at = None


def post(url, body):
    """The open response to a JSON POST of `body` to `url`."""
    target = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(target.hostname, target.port)
    connection.request(
        "POST",
        target.path or "/",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    return connection.getresponse()


def ended_early(response):
    """Why `response` stopped before its framing says it should have, or None.

    The standard library reports a chunked body that stops at a chunk boundary
    as an end of file, so the framing state is read here after the last line.
    A body with neither a chunk framing nor a length (a close-delimited one)
    has no end to compare with.
    """
    if getattr(response, "chunked", False) and response.chunk_left is not None:
        return "the chunked body ended without its last chunk"
    length = getattr(response, "length", None)
    if length:
        return f"the body ended {length} bytes short of its length"
    return None


def server_time(stamp, index, previous):
    """The server stamp of content chunk `index` as a time, or a StampError."""
    if stamp is None:
        raise StampError(f"content chunk {index} has no sent-at stamp")
    try:
        value = float(stamp)
    except ValueError:
        raise StampError(
            f"content chunk {index}: sent-at={stamp!r} is not a number"
        ) from None
    if not math.isfinite(value):
        raise StampError(f"content chunk {index}: sent-at={stamp} is not a finite time")
    if previous is not None and value <= previous:
        raise StampError(
            f"content chunk {index}: sent-at={stamp} does not come after "
            f"chunk {index - 1}, which is at {previous}"
        )
    return value


def parse(stamped, server_stamps=False):
    """Content-chunk times, the completion text, and any usage counts."""
    arrivals = []
    text = []
    usage_tokens = None
    prompt_tokens = None
    for clock_time, stamp, line in stamped:
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        choices = chunk.get("choices") or [{}]
        delta = choices[0].get("delta") or {}
        piece = delta.get("content") or delta.get("reasoning_content") or ""
        if piece:
            if server_stamps:
                previous = arrivals[-1] if arrivals else None
                arrivals.append(server_time(stamp, len(arrivals) + 1, previous))
            else:
                arrivals.append(clock_time)
            text.append(piece)
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            if "completion_tokens" in usage:
                usage_tokens = usage["completion_tokens"]
            if "prompt_tokens" in usage:
                prompt_tokens = usage["prompt_tokens"]
    return arrivals, "".join(text), usage_tokens, prompt_tokens


def answer_digest(text):
    """The digest of a whole completion, the one every caller compares."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def report(arrivals, text, usage_tokens, prompt_tokens, clock):
    """The `key=value` lines for one response."""
    if usage_tokens is not None and len(arrivals) > usage_tokens:
        raise ChunkCountMismatch(
            f"{len(arrivals)} content chunks arrived for a completion of "
            f"{usage_tokens} tokens; a chunk cannot carry less than a token, so "
            "these two counts are not describing the same stream"
        )
    tokens = usage_tokens if usage_tokens is not None else len(arrivals)
    lines = [f"clock={clock}", f"tokens={tokens}", f"content_chunks={len(arrivals)}"]
    if usage_tokens is not None:
        lines.append(f"completion_tokens={usage_tokens}")
    if prompt_tokens is not None:
        lines.append(f"prompt_tokens={prompt_tokens}")
    if len(arrivals) >= 2:
        window = arrivals[-1] - arrivals[0]
        if window > 0:
            lines.append(f"decode_tps={(len(arrivals) - 1) / window:.6f}")
    lines.append(f"answer_sha256={answer_digest(text)}")
    preview = text[:PREVIEW_CHARS].replace("\n", " ").replace("\r", " ")
    lines.append(f"preview={preview}")
    return lines


def run(args, raw):
    """Send the request, read the response, print the report. Returns the exit code."""
    try:
        with open(args.body_file, "rb") as handle:
            body = handle.read()
        response = post(args.url, body)
    except (OSError, http.client.HTTPException) as exc:
        print(f"sse_decode_window: request to {args.url} failed: {exc}", file=sys.stderr)
        return 4

    def tee(source):
        for item in source:
            raw.append(f"{item[0]:.6f} {item[2]}")
            yield item

    try:
        arrivals, text, usage_tokens, prompt_tokens = parse(
            tee(read_stamped(response, args.server_stamps)), args.server_stamps
        )
        response.read()
        cause = ended_early(response)
        if cause:
            raise http.client.HTTPException(cause)
    except StampError as exc:
        print(f"sse_decode_window: unusable server stamp: {exc}", file=sys.stderr)
        return 5
    except (OSError, http.client.HTTPException) as exc:
        print(
            f"sse_decode_window: the response from {args.url} did not end as a "
            f"complete response: {exc}",
            file=sys.stderr,
        )
        return 4

    clock = "server_stamps" if args.server_stamps else "client"
    try:
        lines = report(arrivals, text, usage_tokens, prompt_tokens, clock)
    except ChunkCountMismatch as exc:
        print(f"sse_decode_window: {exc}", file=sys.stderr)
        return 3

    print("\n".join(lines))
    return 0


def run_complete(args):
    """Send the request, keep the whole response, print its digest. Returns the exit code."""
    try:
        with open(args.body_file, "rb") as handle:
            body = handle.read()
        response = post(args.url, body)
        data = response.read()
    except (OSError, http.client.HTTPException) as exc:
        print(f"sse_decode_window: request to {args.url} failed: {exc}", file=sys.stderr)
        return 4
    if response.status != 200:
        print(
            f"sse_decode_window: {args.url} answered {response.status}, not 200",
            file=sys.stderr,
        )
        return 4
    try:
        with open(args.complete, "wb") as handle:
            handle.write(data)
    except OSError as exc:
        print(f"sse_decode_window: cannot write {args.complete}: {exc}", file=sys.stderr)
        return 2
    try:
        completion = json.loads(data)
        message = completion["choices"][0]["message"]
        text = (message.get("reasoning_content") or "") + (message.get("content") or "")
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        print(
            f"sse_decode_window: the response is not a chat completion: {exc!r}",
            file=sys.stderr,
        )
        return 6
    print(f"answer_sha256={answer_digest(text)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="chat-completions endpoint")
    parser.add_argument(
        "--body-file", required=True, help="file holding the JSON request body"
    )
    parser.add_argument(
        "--raw",
        default=None,
        help="also write the arrival-stamped body here, for triage",
    )
    parser.add_argument(
        "--complete",
        default=None,
        metavar="FILE",
        help="send an unstreamed request, write the response to FILE and print its digest",
    )
    parser.add_argument(
        "--server-stamps",
        action="store_true",
        help="time the chunks by the server's `: sent-at=` stamps (self-tests only)",
    )
    args = parser.parse_args(argv)

    if args.complete:
        return run_complete(args)

    raw_handle = None
    if args.raw:
        try:
            raw_handle = open(args.raw, "w", encoding="utf-8")
        except OSError as exc:
            print(f"sse_decode_window: cannot write {args.raw}: {exc}", file=sys.stderr)
            return 2

    raw = []
    status = run(args, raw)

    if raw_handle is not None:
        try:
            with raw_handle:
                raw_handle.write("".join(raw))
        except OSError as exc:
            print(f"sse_decode_window: cannot write {args.raw}: {exc}", file=sys.stderr)
            return 2
    return status


if __name__ == "__main__":
    sys.exit(main())
