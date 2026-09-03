#!/usr/bin/env python3
"""Read a model snapshot's identity columns out of the snapshot itself.

`model_namespace`, `model` and `weight_quant` are cell identity, so a bench
script that hard-codes them files every run under whatever it was written for.
They are facts about the directory being served and are read from it:

* namespace and model from the directory name, split on `__` (docs/METRICS_DB.md
  §5.1). A name without the separator is `local/<name>`.
* `weight_quant` from `config.json`, which is the checkpoint's own statement:
  `quantization.mode` when that names a format (`mxfp8`), the bit width when it
  names a *scheme* instead (`affine` at 8 bits is `8bit`), `paro` for a
  ParoQuant checkpoint, and the storage dtype for an unquantized one. Anything
  that does not land on the §5.2 whitelist is refused rather than guessed at — a
  wrong label here puts the measurement in another checkpoint's cell.

Output (stdout), one `key=value` per line:

    model_namespace=<ns>
    model=<name>
    weight_quant=<label>

Exit codes: 0 — read; 2 — no readable `config.json`; 6 — the quantization it
describes has no §5.2 label.
"""

import argparse
import json
import os
import sys

# docs/METRICS_DB.md §5.2, mirroring identity::WEIGHT_QUANT_WHITELIST.
WEIGHT_QUANT_WHITELIST = {
    "bf16", "fp16", "mxfp8", "mxfp4", "nvfp4", "q8_0", "q4_k_m",
    "2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "paro",
}

DTYPE_LABELS = {"bfloat16": "bf16", "float16": "fp16"}


def split_model_dir(name):
    """`ns__model` → (ns, model); a bare name is namespaced `local`."""
    if "__" in name:
        namespace, model = name.split("__", 1)
        return namespace, model
    return "local", name


def weight_quant(config):
    """The §5.2 label this checkpoint's own metadata describes, or None."""
    quant_config = config.get("quantization_config") or {}
    if quant_config.get("quant_method") == "paroquant":
        return "paro"

    quant = config.get("quantization")
    if isinstance(quant, dict):
        # `mode` names the format for mxfp/nvfp checkpoints; for affine ones it
        # names the *scheme* ("affine"), which is not a format, and the bit
        # width is what distinguishes them.
        mode = quant.get("mode")
        if isinstance(mode, str) and mode in WEIGHT_QUANT_WHITELIST:
            return mode
        bits = quant.get("bits")
        if isinstance(bits, int):
            return f"{bits}bit"
        return None

    return DTYPE_LABELS.get(config.get("dtype") or config.get("torch_dtype"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="path to a model snapshot directory")
    args = parser.parse_args()

    path = args.snapshot.rstrip("/")
    namespace, model = split_model_dir(os.path.basename(path))

    try:
        with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"snapshot_identity: {path}/config.json: {exc}", file=sys.stderr)
        return 2

    label = weight_quant(config)
    if label not in WEIGHT_QUANT_WHITELIST:
        print(
            f"snapshot_identity: {path} describes weight quantization {label!r}, "
            "which is not one of the labels the metrics DB records",
            file=sys.stderr,
        )
        return 6

    print(f"model_namespace={namespace}")
    print(f"model={model}")
    print(f"weight_quant={label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
