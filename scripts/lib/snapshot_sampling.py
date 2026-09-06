#!/usr/bin/env python3
"""Read the sampling defaults a snapshot states, for a run that sends none.

The published protocol measures a checkpoint under its own sampling, and the way
to send a checkpoint's own sampling is to send no sampling field at all: the
engine then resolves request > server default > `generation_config.json` >
a hard-coded fallback. A snapshot that states nothing does not make the sampling
unknown — it makes it the fallback, and the fallback would be published as that
checkpoint's default.

So the three fields the protocol names are required, because each has a fallback
that is not this checkpoint's:

    temperature   -> 1.0
    top_p         -> 1.0
    top_k         -> 0, which disables top-k entirely

`repetition_penalty` is deliberately not required: the protocol does not name it
and its fallback is 1.0, the neutral element rather than another setting.

This is a pre-flight refusal, so hours are not spent before the fallback is
discovered. It is NOT the published figure — what a run actually sampled under
is read back from the engine's own resolved-sampler event.

Output (stdout), one `key=value` per line:

    temperature=<f>   what the snapshot states; the caller needs it to know
                      whether the engine will resolve a sampler at all
    sampled=<true|false>

Exit codes: 0 — read; 1 — the snapshot states no sampling this protocol can
run under; 2 — the file is absent or unreadable.
"""

import argparse
import json
import pathlib
import sys

REQUIRED = ("temperature", "top_p", "top_k")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot", help="a model snapshot directory")
    args = ap.parse_args()

    path = pathlib.Path(args.snapshot) / "generation_config.json"
    if not path.is_file():
        print(
            f"snapshot_sampling: {path} does not exist: the request sends no "
            "sampling field, so the engine would fall back to temperature 1.0, "
            "top_p 1.0 and top-k disabled, and those would be published as this "
            "checkpoint own defaults",
            file=sys.stderr,
        )
        return 2
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"snapshot_sampling: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(cfg, dict):
        print(f"snapshot_sampling: {path} is not a JSON object", file=sys.stderr)
        return 2

    missing = [k for k in REQUIRED if cfg.get(k) is None]
    if missing:
        print(
            f"snapshot_sampling: {path} states no {', '.join(missing)}: the "
            "request sends no sampling field, so the engine would fall back to "
            "its hard-coded default and that would be published as this "
            "checkpoint own default",
            file=sys.stderr,
        )
        return 1

    temperature = cfg["temperature"]
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        print(
            f"snapshot_sampling: {path} states temperature={temperature!r}, "
            "which is not a number",
            file=sys.stderr,
        )
        return 1

    print(f"temperature={temperature}")
    # The engine writes its resolved-sampler event only when the sampler is
    # active, so whether a run has one to read back is the checkpoint to say.
    print(f"sampled={'true' if temperature > 0 else 'false'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
