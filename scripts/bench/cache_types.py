"""Python half of baseline_cells.sh --preset cache-types: a cell's run record and a smoke text check.

Module of: bench/baseline_cells.sh

`record` writes the one run record of a measured cell: the median decode rate of its
measured runs, with their stddev, filed as `gate`. Its `kv_quant` is the codec name the
caller read from the run's own log; this module maps no flags to a name. `repeat` reads a
smoke run's stderr log on stdin. It prints how often the most repeated four-word sequence
of the last `decoded="..."` field occurs, `none` when the log holds no such field, or
`empty` when the field holds no word.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from decoded_text import decoded_text  # noqa: E402

FOUR_GRAM = 4


def number(raw: str) -> float | None:
    """An empty reading is a missing measurement, not a zero."""
    raw = raw.strip()
    return float(raw) if raw else None


def model_identity(model_path: str) -> tuple[str, str]:
    """The namespace and name of a snapshot directory `<namespace>__<name>`."""
    base = Path(model_path).name
    if "__" in base:
        namespace, name = base.split("__", 1)
        return namespace, name
    return "local", base


def build_record(args: argparse.Namespace) -> dict:
    namespace, model = model_identity(args.model_path)
    prompt_file = Path(args.prompt_file)
    git_sha = {"git_sha": args.git_sha} if args.git_sha else {}
    return {
        **json.loads(args.identity_json),
        **git_sha,
        "model_namespace": namespace,
        "model": model,
        "weight_quant": args.weight_quant,
        "kv_quant": args.kv_quant,
        "ctx_max": args.ctx_max,
        "prompt": {"name": prompt_file.stem, "body": json.loads(prompt_file.read_text())},
        "ts_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "n_warmups": args.warmups,
        "n_measure": args.measured,
        "notes": f"label={args.label} ctk={args.ctk or 'auto'} ctv={args.ctv or 'auto'} "
        f"warmup={args.warmups} measured={args.measured}",
        "description": f"baseline_cells cache-types {args.tag} {args.label}",
        "run_class": "gate",
        "metrics": [
            {"name": "decode_tps_warm", "value": number(args.median), "stddev": number(args.stddev)},
            {"name": "ttft_warm_ms", "value": number(args.ttft)},
            {"name": "model_load_ms", "value": number(args.load)},
            {"name": "peak_rss_mb", "value": number(args.rss)},
        ],
    }


def max_repeat(text: str) -> int:
    """How often the most repeated run of four whitespace-separated words occurs in `text`."""
    words = text.split()
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - FOUR_GRAM + 1):
        gram = tuple(words[i : i + FOUR_GRAM])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values(), default=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("repeat")
    record = commands.add_parser("record")
    for name in ("out", "identity-json", "model-path", "weight-quant", "kv-quant", "prompt-file", "label", "tag"):
        record.add_argument(f"--{name}", required=True)
    for name in ("ctk", "ctv", "git-sha", "median", "stddev", "ttft", "load", "rss"):
        record.add_argument(f"--{name}", default="")
    for name in ("ctx-max", "prompt-tokens", "max-tokens", "warmups", "measured"):
        record.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()
    if args.command == "repeat":
        text = decoded_text(sys.stdin.read())
        if text is None:
            print("none")
        elif not text.split():
            print("empty")
        else:
            print(max_repeat(text))
        return 0
    Path(args.out).write_text(json.dumps(build_record(args), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
