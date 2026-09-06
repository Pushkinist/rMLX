#!/usr/bin/env python3
"""Render one request payload per checked-in sample, and index them.

A cell is a `<dataset>:<max output tokens>` pair, and the protocol measures four
of them: the three datasets at the pinned budget, plus MATH-500 again at the
longer one. This turns the sample sets under a root into one payload file per
cell and sample, and a tab-separated index naming them.

Two ways the cell table and the sample sets can drift apart, both refused here:
a cell naming a dataset the root does not hold, and a dataset the root holds
that no cell measures. The second is the quiet one — a checked-in sample set
that no pass ever sends is a sample set nobody is reading, and the macro average
silently stops covering it.

The `messages` array is copied out of the sample verbatim. The workspace builds
serde_json with `preserve_order`, so a message re-emitted as `{content, role}`
has a different content address than the `{role, content}` that was checked in,
and a later join on that address splits with nothing saying so.

The payload carries no temperature, top_p, top_k or seed: the published sampling
is the checkpoint's own, and the way to send the checkpoint's own is to send
none. Thinking is asked for explicitly — it is a pinned choice of the protocol,
not a template default this is willing to inherit.

`warmup.json` is written beside the cells and left out of the index. Its prompt
is not in any sample set: an untimed warmup on the first measured sample would
leave that one sample facing a warm prompt cache while every other is cold, and
the protocol defines input speed over a cold one. A warmup is there to make the
weights resident and the kernels compiled, which any prompt does.

Exit codes: 0 — written; 1 — the cell table and the root do not describe one
measurement, or a dataset is empty; 2 — the root could not be read.
"""

import argparse
import json
import pathlib
import sys


class RootError(Exception):
    """The cell table and the sample root do not describe one measurement."""


def load_root(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return {d["key"]: d["file"] for d in manifest["datasets"]}


def check_coverage(files, cells):
    named = {dataset for dataset, _ in cells}
    missing = sorted(d for d in named if d not in files)
    if missing:
        raise RootError(f"the sample root holds no dataset {', '.join(missing)}")
    unmeasured = sorted(k for k in files if k not in named)
    if unmeasured:
        raise RootError(
            f"the sample root holds {', '.join(unmeasured)}, which no cell "
            "measures: a checked-in dataset that no pass sends is a sample set "
            "nobody is reading"
        )


WARMUP_PROMPT = "Say hello."
WARMUP_MAX_TOKENS = 64


def payload(model_id, messages, max_tokens):
    return {
        "model": model_id,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "enable_thinking": True,
        "stream": True,
        # The prompt length recorded with a request is the one the server
        # counted, and it only says so when asked.
        "stream_options": {"include_usage": True},
    }


def write_json(path, body):
    path.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")


def render(root, out, model_id, cells):
    """(cell, dataset, max_tokens, sample_id, body_sha256, payload_path) rows."""
    files = load_root(root)
    check_coverage(files, cells)

    out.mkdir(parents=True, exist_ok=True)
    write_json(
        out / "warmup.json",
        payload(
            model_id,
            [{"role": "user", "content": WARMUP_PROMPT}],
            WARMUP_MAX_TOKENS,
        ),
    )

    rows = []
    for dataset, max_tokens in cells:
        doc = json.loads((root / files[dataset]).read_text(encoding="utf-8"))
        cell = f"{dataset}@{max_tokens}"
        if not doc["samples"]:
            raise RootError(f"{cell}: the dataset holds no sample")
        cell_dir = out / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(doc["samples"]):
            path = cell_dir / f"{i:05d}.json"
            write_json(path, payload(model_id, sample["messages"], max_tokens))
            rows.append(
                (cell, dataset, str(max_tokens), sample["id"],
                 sample["body_sha256"], str(path))
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples-root", required=True)
    ap.add_argument("--out", required=True, help="directory for the payload files")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--index", required=True, help="where to write the index")
    ap.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="DATASET:MAX_TOKENS",
        help="one measured cell; repeatable",
    )
    args = ap.parse_args()

    cells = []
    for spec in args.cell:
        dataset, sep, max_tokens = spec.partition(":")
        if not sep or not max_tokens.isdigit():
            print(f"published_payloads: --cell {spec!r} is not DATASET:MAX_TOKENS",
                  file=sys.stderr)
            return 1
        cells.append((dataset, max_tokens))

    root = pathlib.Path(args.samples_root)
    try:
        rows = render(root, pathlib.Path(args.out), args.model_id, cells)
    except (OSError, ValueError, KeyError) as exc:
        print(f"published_payloads: cannot read {root}: {exc}", file=sys.stderr)
        return 2
    except RootError as exc:
        print(f"published_payloads: {root}: {exc}", file=sys.stderr)
        return 1

    pathlib.Path(args.index).write_text(
        "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"{len(rows)} requests over {len(cells)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
