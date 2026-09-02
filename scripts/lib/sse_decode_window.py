#!/usr/bin/env python3
"""Time a streamed chat-completions response over its decode window.

Reads an OpenAI-compatible `stream: true` body on stdin, stamping each chunk as
it arrives, and reports the rate over the same window the engine reports:
first content token to last, `(n - 1) / (t_last - t_first)`. Prefill, connection
setup and the process spawn all land before the first content chunk and are
outside it, which is what makes this number comparable with `rmlx baseline`'s
`decode_tps` and with the `decode_tps` a speculative round loop logs
(docs/SPECULATIVE.md). Dividing the completion tokens by the whole request
duration measures something else and reads low by the prefill.

A response with fewer than two content chunks has no interval and no rate; the
`decode_tps` line is then absent rather than zero, because a zero in that slot
is averaged and ranked as a real throughput.

The window is timed from chunk arrivals, so it is only a window over tokens if
the tokens arrived one per chunk. When the response carries a usage block that
disagrees with the number of content chunks, this refuses rather than reporting
a rate that is low by the batching factor.

Output (stdout), one `key=value` per line:

    tokens=<n>          completion tokens, from the usage chunk when present
    decode_tps=<f>      omitted when the response has no measurable window
    preview=<text>      first 64 characters of the completion, newlines folded

Exit codes: 0 — read; 2 — `--raw` could not be written; 3 — the tokens did not
arrive one per chunk.
"""

import argparse
import json
import sys
import time

PREVIEW_CHARS = 64


def read_stamped(stream):
    """(arrival_time, line) for every line of `stream`, as it arrives.

    Explicit `readline()` rather than iteration: iterating a buffered stream
    reads ahead, which would stamp a whole block of tokens with the arrival
    time of the last one and collapse the window.
    """
    while True:
        line = stream.readline()
        if not line:
            return
        yield time.monotonic(), line.decode("utf-8", errors="replace")


def parse(stamped):
    """Content-chunk arrival times, the completion text, and any usage count."""
    arrivals = []
    text = []
    usage_tokens = None
    for arrival, line in stamped:
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
            arrivals.append(arrival)
            text.append(piece)
        usage = chunk.get("usage")
        if isinstance(usage, dict) and "completion_tokens" in usage:
            usage_tokens = usage["completion_tokens"]
    return arrivals, "".join(text), usage_tokens


class ChunkCountMismatch(Exception):
    """The response batched tokens into chunks, so arrivals cannot time them."""


def report(arrivals, text, usage_tokens):
    """The `key=value` lines for one response."""
    if usage_tokens is not None and usage_tokens != len(arrivals):
        raise ChunkCountMismatch(
            f"usage reports {usage_tokens} completion tokens against "
            f"{len(arrivals)} content chunks; chunk arrivals can only time a "
            "window whose tokens arrived one per chunk"
        )
    tokens = usage_tokens if usage_tokens is not None else len(arrivals)
    lines = [f"tokens={tokens}"]
    if len(arrivals) >= 2:
        window = arrivals[-1] - arrivals[0]
        if window > 0:
            lines.append(f"decode_tps={(len(arrivals) - 1) / window:.6f}")
    preview = text[:PREVIEW_CHARS].replace("\n", " ").replace("\r", " ")
    lines.append(f"preview={preview}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default=None,
        help="also write the arrival-stamped body here, for triage",
    )
    args = parser.parse_args()

    raw = []
    stamped = read_stamped(sys.stdin.buffer)
    if args.raw:

        def tee(source):
            for arrival, line in source:
                raw.append(f"{arrival:.6f} {line}")
                yield arrival, line

        stamped = tee(stamped)

    arrivals, text, usage_tokens = parse(stamped)

    if args.raw:
        try:
            with open(args.raw, "w", encoding="utf-8") as handle:
                handle.write("".join(raw))
        except OSError as exc:
            print(f"sse_decode_window: cannot write {args.raw}: {exc}", file=sys.stderr)
            return 2

    try:
        lines = report(arrivals, text, usage_tokens)
    except ChunkCountMismatch as exc:
        print(f"sse_decode_window: {exc}", file=sys.stderr)
        return 3

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
