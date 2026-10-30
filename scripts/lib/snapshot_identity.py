#!/usr/bin/env python3
"""Read a model snapshot's identity columns out of the snapshot itself.

`model_namespace`, `model` and `weight_quant` are cell identity, so a bench
script that hard-codes them files every run under whatever it was written for.
They are facts about the directory being served and are read from it:

* namespace and model from the directory name, split on `__` (docs/METRICS_DB.md
  §5.1). A name without the separator is `local/<name>`.
* `weight_quant` from `config.json`, which is the checkpoint's own statement:
  `paro` for a ParoQuant checkpoint, else `quantization.mode` lowercased
  (`mxfp8`), or the bit width when the mode is absent or `affine`, which names
  a *scheme* (`affine` at 8 bits is `8bit`), else the storage dtype for an
  unquantized one. Anything that does not land on the §5.2 whitelist is
  refused rather than guessed at — a wrong label here puts the measurement in
  another checkpoint's cell.

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
    "1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "paro",
}

DTYPE_LABELS = {"bfloat16": "bf16", "float16": "fp16"}


def split_model_dir(name):
    """`ns__model` → (ns, model); a bare name is namespaced `local`.

    The namespace is a free-form label recorded as given; an empty part is
    refused with ValueError, as the Rust `identity::split_model_path` does.
    """
    if "__" not in name:
        return "local", name
    namespace, model = name.split("__", 1)
    if not namespace or not model:
        raise ValueError(f"{name!r}: model_namespace and model must not be empty")
    return namespace, model


def weight_quant(config):
    """The §5.2 label this checkpoint's own metadata describes, or None.

    Kept in step with `ModelConfig::weight_quant_label` in
    `crates/rmlx-loader/src/config.rs`, the label `rmlx baseline --record` and
    the server's drainer file the same snapshot under, by one shared table:
    `crates/rmlx-loader/tests/fixtures/weight_quant_labels.json`.
    """
    quant_config = config.get("quantization_config") or {}
    if quant_config.get("quant_method") == "paroquant":
        return "paro"

    quant = config.get("quantization")
    if isinstance(quant, dict):
        # `mode` names the format for mxfp/nvfp checkpoints and passes through
        # lowercased; for affine ones it names the *scheme* ("affine", or absent),
        # and the bit width is what distinguishes them. The whitelist is the
        # caller's check, not this function's.
        mode = quant.get("mode")
        if isinstance(mode, str) and mode and mode.lower() != "affine":
            return mode.lower()
        bits = quant.get("bits")
        if isinstance(bits, int):
            return f"{bits}bit"
        return None

    return DTYPE_LABELS.get(config.get("dtype") or config.get("torch_dtype"))


class UnlabelledSnapshot(Exception):
    """A snapshot whose weights have no label the metrics DB records.

    `exit_code` is this script's exit code for the case: 2 when `config.json`
    cannot be read, 6 when the quantization it describes has no §5.2 label.
    """

    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code


def snapshot_weight_quant(snapshot_dir):
    """The §5.2 weight label of the snapshot at `snapshot_dir`, read from its
    `config.json`; raises `UnlabelledSnapshot` rather than guess one."""
    config_path = os.path.join(snapshot_dir, "config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError) as exc:
        raise UnlabelledSnapshot(f"{config_path}: {exc}", 2) from None
    label = weight_quant(config)
    if label not in WEIGHT_QUANT_WHITELIST:
        raise UnlabelledSnapshot(
            f"{snapshot_dir} describes weight quantization {label!r}, "
            "which is not one of the labels the metrics DB records",
            6,
        )
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="path to a model snapshot directory")
    args = parser.parse_args()

    path = args.snapshot.rstrip("/")
    try:
        namespace, model = split_model_dir(os.path.basename(path))
    except ValueError as exc:
        print(f"snapshot_identity: {exc}", file=sys.stderr)
        return 2

    try:
        label = snapshot_weight_quant(path)
    except UnlabelledSnapshot as exc:
        print(f"snapshot_identity: {exc}", file=sys.stderr)
        return exc.exit_code

    print(f"model_namespace={namespace}")
    print(f"model={model}")
    print(f"weight_quant={label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
