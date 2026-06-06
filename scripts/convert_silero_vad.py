"""
Convert Silero VAD v4 ONNX weights (16kHz branch only) to safetensors.

Run once at asset-preparation time. The output file is committed to
crates/rmlx-audio/assets/silero_vad_16k.safetensors and loaded at runtime
by the rmlx-audio VAD engine. Do NOT run at server startup.

Usage:
    uv run scripts/convert_silero_vad.py <silero_vad.onnx> <output.safetensors>

Dependencies (via paroquant venv or any Python 3.9+ env):
    uv pip install onnx

## Architecture (16kHz path)

    input [batch, T_samples]
      → Pad (64 samples on each side)
      → Unsqueeze → Conv1d(stft.forward_basis_buffer [258, 1, 256], stride=128)
      → split real/imag → magnitude sqrt(r² + i²) → [batch, 129, T_frames]
    encoder.0: Conv1d [128, 129, 3], pad=1, stride=1 + ReLU
    encoder.1: Conv1d [64, 128, 3], pad=1, stride=1 + ReLU
    encoder.2: Conv1d [64, 64, 3], pad=1, stride=1 + ReLU
    encoder.3: Conv1d [128, 64, 3], pad=1, stride=1 + ReLU
    decoder.rnn: LSTM(input=128, hidden=128)
      weight_ih [512, 128], weight_hh [512, 128]
      bias_ih [512],        bias_hh [512]
      state input [2, batch, 128] = [h_n, c_n] stacked
    decoder.2: Conv1d [1, 128, 1] + sigmoid → VAD probability per frame

Source: https://github.com/snakers4/silero-vad (MIT License)
"""

import sys
import json
import struct
import numpy as np

try:
    import onnx
    import onnx.numpy_helper as nph
except ImportError:
    print("ERROR: onnx not installed. Run: uv pip install onnx", file=sys.stderr)
    sys.exit(1)

ONNX_PATH = sys.argv[1] if len(sys.argv) > 1 else "silero_vad.onnx"
OUT_PATH  = sys.argv[2] if len(sys.argv) > 2 else "crates/rmlx-audio/assets/silero_vad_16k.safetensors"

model = onnx.load(ONNX_PATH)
graph = model.graph

# Extract the then_branch — this is the sr==16000 path.
# The outer graph has: Equal(sr, 16000) → If(then=16kHz, else=8kHz).
if_node = next(n for n in graph.node if n.op_type == "If")
then_branch = next(attr.g for attr in if_node.attribute if attr.name == "then_branch")

# Gather all Constant nodes → weight tensors.
# All model weights are inlined as ONNX Constant nodes (no external initializers).
raw: dict[str, np.ndarray] = {}
for node in then_branch.node:
    if node.op_type == "Constant":
        name = node.output[0]
        for attr in node.attribute:
            if attr.name == "value":
                arr = nph.to_array(attr.t)
                key = name.replace("If_0_then_branch__Inline_0__", "")
                raw[key] = arr

EXPORT_KEYS = [
    "stft.forward_basis_buffer",     # [258, 1, 256] – real STFT via learned conv
    "encoder.0.reparam_conv.weight", # [128, 129, 3]
    "encoder.0.reparam_conv.bias",   # [128]
    "encoder.1.reparam_conv.weight", # [64, 128, 3]
    "encoder.1.reparam_conv.bias",   # [64]
    "encoder.2.reparam_conv.weight", # [64, 64, 3]
    "encoder.2.reparam_conv.bias",   # [64]
    "encoder.3.reparam_conv.weight", # [128, 64, 3]
    "encoder.3.reparam_conv.bias",   # [128]
    "decoder.rnn.weight_ih",         # [512, 128] – LSTM input→hidden
    "decoder.rnn.weight_hh",         # [512, 128] – LSTM hidden→hidden
    "decoder.rnn.bias_ih",           # [512]
    "decoder.rnn.bias_hh",           # [512]
    "decoder.decoder.2.weight",      # [1, 128, 1]
    "decoder.decoder.2.bias",        # [1]
]

# Build safetensors: uint64-LE header-length + JSON header + packed f32 blobs.
metadata: dict = {}
data_parts: list[bytes] = []
offset = 0

for key in EXPORT_KEYS:
    if key not in raw:
        print(f"ERROR: key '{key}' not found in ONNX model", file=sys.stderr)
        sys.exit(1)
    arr = raw[key].astype(np.float32, copy=False)
    nbytes = arr.nbytes
    metadata[key] = {
        "dtype": "F32",
        "shape": list(arr.shape),
        "data_offsets": [offset, offset + nbytes],
    }
    data_parts.append(arr.tobytes())
    offset += nbytes

header_json = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
# Pad to 8-byte boundary.
pad = (8 - len(header_json) % 8) % 8
header_json += b" " * pad
header_len = struct.pack("<Q", len(header_json))

with open(OUT_PATH, "wb") as f:
    f.write(header_len)
    f.write(header_json)
    for part in data_parts:
        f.write(part)

import os
size_kb = os.path.getsize(OUT_PATH) / 1024
print(f"Wrote {OUT_PATH} ({size_kb:.1f} KB, {len(EXPORT_KEYS)} tensors)")
for key in EXPORT_KEYS:
    arr = raw[key]
    print(f"  {key}: {list(arr.shape)} f32")
