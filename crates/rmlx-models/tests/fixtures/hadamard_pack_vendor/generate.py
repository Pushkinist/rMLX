"""Regenerate the vendor reference logits of the Ternary-Bonsai-2-27B Hadamard pack.

rMLX runs no Python. This script exists so the committed fixture is
reproducible: it loads the pack with the runtime the pack ships
(`runtime/vision_artifact.py`, `load_vl_model`) and writes what that runtime
computes for three fixed token sequences.

    python3 -m venv venv
    venv/bin/pip install -r <pack>/runtime/requirements.txt
    rmlx claim run -- venv/bin/python -I generate.py <pack> [--full <dir>]

It runs on the GPU, so it holds the Metal claim (`docs/CLI.md` "Claim file").

`logits.safetensors`, beside this file, holds for each sequence and position:

  * `ids` [3, 64]: the token ids, from the pack's `tokenizer.json`;
  * `reference_top_ids`, `reference_top_logits` [3, 64, 20]: the 20 largest
    logits of the runtime as shipped (f16 embedding output, f16 scales), in
    descending order, and `reference_logsumexp` [3, 64] over the vocabulary;
  * `floor_top_logits` [3, 64, 20], `floor_logsumexp`, `floor_argmax`
    [3, 64]: the same runtime after mlx-vlm's own bf16 cast (every floating
    parameter the model's `cast_predicate` accepts, as `mlx_vlm.convert`
    casts), at the reference's top ids. This is the noise floor of bf16
    activations.

`--full <dir>` also writes the full logits of both runs as float32 `.npy`.

The consumer is `crates/rmlx-models/src/qwen3_5_moe/pack_reference_tests.rs`.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LENGTH = 64
TOP = 20

PROMPTS = [
    "<|im_start|>user\nExplain how a refrigerator moves heat from its inside to the "
    "room, and why the back of it feels warm.<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\nA refrigerator pumps a refrigerant around a closed loop. "
    "Inside the cabinet the refrigerant evaporates at low pressure, which absorbs heat "
    "from the food. A compressor then squeezes the vapour, so its pressure and its "
    "temperature rise above the temperature of the room.",
    "def merge_sorted(left, right):\n"
    '    """Merge two sorted lists into one sorted list."""\n'
    "    result = []\n"
    "    i = j = 0\n"
    "    while i < len(left) and j < len(right):\n"
    "        if left[i] <= right[j]:\n"
    "            result.append(left[i])\n"
    "            i += 1\n"
    "        else:\n"
    "            result.append(right[j])\n"
    "            j += 1\n"
    "    result.extend(left[i:])\n"
    "    result.extend(right[j:])\n"
    "    return result\n",
    "A train leaves the station at 9:15 and travels 240 kilometres at an average speed "
    "of 80 kilometres per hour. It then stops for 20 minutes and continues for another "
    "150 kilometres at 100 kilometres per hour. To find the arrival time, first compute "
    "the time of each part of the journey: 240 / 80 = 3 hours, and 150 / 100 = 1.5 hours.",
]


def logsumexp(rows):
    top = rows.max(axis=-1, keepdims=True)
    return (top[..., 0] + np.log(np.exp(rows - top).sum(axis=-1))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--full", type=Path)
    args = parser.parse_args()
    pack = args.pack.resolve()
    sys.path.insert(0, str(pack / "runtime"))

    import mlx.core as mx
    import mlx_vlm
    from mlx.utils import tree_flatten, tree_unflatten
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer
    from vision_artifact import load_vl_model

    if mx.default_device() != mx.gpu:
        raise SystemExit("the reference runs on the GPU")

    tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
    ids = []
    for text in PROMPTS:
        row = tokenizer.encode(text, add_special_tokens=False).ids
        if len(row) < LENGTH:
            raise SystemExit(f"a prompt has {len(row)} tokens, fewer than {LENGTH}")
        ids.append(row[:LENGTH])
    ids = np.array(ids, dtype=np.int32)

    model, _, _ = load_vl_model(pack, load_processor=False)
    lm = model.language_model

    def run(label):
        rows = []
        for row in ids:
            lm._position_ids = None
            lm._rope_deltas = None
            out = lm(mx.array(row[None, :]))
            logits = getattr(out, "logits", out)
            sink = []
            lm.model(
                mx.array(row[None, :]),
                position_ids=lm._position_ids,
                capture_layer_ids=[0, 63],
                hidden_sink=sink,
            )
            rows.append(np.array(logits[0].astype(mx.float32)))
        print(
            f"{label}: logits {logits.dtype}, hidden after layer 0 {sink[0].dtype}, "
            f"after layer 63 {sink[1].dtype}",
            flush=True,
        )
        return np.stack(rows)

    reference = run("as shipped")
    again = run("as shipped, again")
    print("as shipped, run to run: max abs", float(np.abs(reference - again).max()), flush=True)
    del again

    cast = lm.cast_predicate
    lm.update(
        tree_unflatten(
            [
                (k, v.astype(mx.bfloat16) if cast(k) and mx.issubdtype(v.dtype, mx.floating) else v)
                for k, v in tree_flatten(lm.parameters())
            ]
        )
    )
    lm.model.embed_tokens.dtype = mx.bfloat16
    floor = run("bf16 cast")

    top_ids = np.argsort(-reference, axis=-1, kind="stable")[..., :TOP].astype(np.int32)
    tensors = {
        "ids": ids,
        "reference_top_ids": top_ids,
        "reference_top_logits": np.take_along_axis(reference, top_ids, axis=-1),
        "reference_logsumexp": logsumexp(reference),
        "floor_top_logits": np.take_along_axis(floor, top_ids, axis=-1),
        "floor_logsumexp": logsumexp(floor),
        "floor_argmax": floor.argmax(axis=-1).astype(np.int32),
    }
    metadata = {
        "mlx": mx.__version__,
        "mlx_vlm": mlx_vlm.__version__,
        "config_sha256": hashlib.sha256((pack / "config.json").read_bytes()).hexdigest(),
        "prompts": json.dumps(PROMPTS),
    }
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(HERE / "logits.safetensors"), metadata)
    if args.full:
        args.full.mkdir(parents=True, exist_ok=True)
        np.save(args.full / "reference.npy", reference)
        np.save(args.full / "floor.npy", floor)
    print("wrote", HERE / "logits.safetensors", flush=True)


if __name__ == "__main__":
    main()
