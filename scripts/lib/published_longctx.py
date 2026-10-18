#!/usr/bin/env python3
"""The long-context row: one canonical prompt, plain decode, a cold cache.

The protocol's fixed prompt is 1 355 tokens, which says nothing about how time to
first token and the decode rate grow with the context. The row this builds is the
same measurement on `prompts/longctx_<size>.json`, one of the checked-in canonical
prompts: time to first token, input speed by both its definitions, the decode rate
over the window after it, and the resident peaks, three runs, plain decode only.

  payload  render the request body of one prompt file, and say what the file
           holds and what context ceiling the server needs for it.
  run      assemble one run from the engine's account of it, the client's and
           the memory poll.

WHAT THE LABEL CLAIMS. The file is checked in, so unlike the fixed prompt it is not
fitted: it is sent as it is, and the token count is whatever this checkpoint's
tokenizer makes of it. The file names the size it was built for. A server count
more than a quarter away from that is not a prompt of that size, and the run is
refused rather than filed under the size in its name.

WHAT IS SENT. The prompt's messages verbatim, at temperature 0 and with a short
output budget: the row is about the context, and a sampled long answer would put
the host sampler into a decode rate that is meant to be the engine's.

THE CEILING. The harness's other servers run at its default context, which a
32k prompt does not fit, so each of these runs on a server of its own started at a
ceiling a quarter above the prompt's size. The ring grows lazily to it, so the
ceiling costs nothing the prompt does not.

Exit codes: 0 — rendered or assembled; 1 — the readings do not describe a run of
that prompt; 2 — an input could not be read.
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import published_fixed_run  # noqa: E402
from published_samples import body_sha256  # noqa: E402

# How far the server's count may sit from the size the file was built for.
SIZE_TOLERANCE = 0.25


class RunError(Exception):
    """The readings do not describe a run of that prompt."""


def ctx_ceiling(target_tokens, max_tokens):
    """The context a server needs for a prompt of this size, with room to spare."""
    return target_tokens + target_tokens // 4 + max_tokens


def payload(args):
    path = pathlib.Path(args.file)
    blob = path.read_bytes()
    doc = json.loads(blob)
    messages = doc["messages"]
    target = int(doc["target_tokens"])
    body = {
        "model": args.model_id,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "enable_thinking": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    pathlib.Path(args.out).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    meta = {
        "label": doc["label"],
        "file": path.name,
        "file_sha256": hashlib.sha256(blob).hexdigest(),
        "body_sha256": body_sha256(messages),
        "target_tokens": target,
        "max_tokens": args.max_tokens,
        "ctx_max": ctx_ceiling(target, args.max_tokens),
    }
    pathlib.Path(args.meta).write_text(json.dumps(meta) + "\n", encoding="utf-8")
    print(f"{meta['label']}: {target} tokens, ceiling {meta['ctx_max']}")


def run(args):
    meta = json.loads(pathlib.Path(args.meta).read_text(encoding="utf-8"))
    engine = json.loads(pathlib.Path(args.engine).read_text(encoding="utf-8"))
    rows = engine["requests"]
    if len(rows) != 1:
        raise RunError(f"the engine reported {len(rows)} requests for this run, not one")
    client = published_fixed_run.read_kv(args.client)
    what = f"the {meta['label']} prompt"
    figures = published_fixed_run.cold_request(rows[0], client, what, args.cross_check_pct)

    target = meta["target_tokens"]
    off = abs(figures["prompt_tokens"] - target) / target
    if off > SIZE_TOLERANCE:
        raise RunError(
            f"the server counted {figures['prompt_tokens']} tokens in {what}, "
            f"{off * 100:.0f}% away from the {target} it was built for and past the "
            f"{SIZE_TOLERANCE * 100:.0f}% allowed: this is not a prompt of that size"
        )
    phys, rss, samples = published_fixed_run.peak_memory(args.memory, f"{what}'s request")
    return {
        "pass": args.pass_number,
        **meta,
        **figures,
        "phys_footprint_bytes": phys,
        "rss_bytes": rss,
        "memory_samples": samples,
        "memory_poll_ms": args.memory_poll_ms,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    pl = sub.add_parser("payload", help="render one prompt file as a request body")
    pl.add_argument("--file", required=True, help="prompts/longctx_<size>.json")
    pl.add_argument("--model-id", required=True)
    pl.add_argument("--max-tokens", type=int, required=True)
    pl.add_argument("--out", required=True, help="where to write the request body")
    pl.add_argument("--meta", required=True, help="where to write what the file holds")

    rn = sub.add_parser("run", help="assemble one run")
    rn.add_argument("--engine", required=True, help="published_run_log.py output")
    rn.add_argument("--client", required=True, help="the client-side key=value block")
    rn.add_argument("--memory", required=True, help="the memory poll file")
    rn.add_argument("--meta", required=True, help="the payload step's meta")
    rn.add_argument("--pass-number", type=int, required=True, dest="pass_number")
    rn.add_argument("--memory-poll-ms", type=int, required=True)
    rn.add_argument("--cross-check-pct", type=float, default=10.0)

    args = ap.parse_args()
    try:
        if args.command == "payload":
            payload(args)
        else:
            json.dump(run(args), sys.stdout)
            print()
    except (RunError, published_fixed_run.RunError) as exc:
        print(f"published_longctx: {exc}", file=sys.stderr)
        return 1
    except (OSError, KeyError, ValueError) as exc:
        print(f"published_longctx: cannot read an input: {exc!r}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
